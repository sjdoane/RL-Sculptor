"""Sculpt-run orchestration.

Per Prompt 8 R3 this is NOT a parallel state tracker — runs are
JobManager jobs of kind `sculpt_run`. This module provides:

  - `run_sculpt_job(...)`: the async callable handed to `JobManager.submit`.
    It spawns `python -m sculptor.cli run ...` as a subprocess, captures
    stdout line-by-line, parses additive `[SCULPT-EVENT]` markers, and
    runs filesystem watchers (the primary source of truth per R2).

  - `build_iterations_summary(job)`: pure view over `job.events` that
    produces the `IterEventSummary` list for REST detail responses.

Filesystem watchers are authoritative. Stdout markers supplement them
(early_stop reason, run-level metadata) but if fs and stdout disagree,
the fs wins.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from backend.services.cuda_errors import CudaErrorClass, classify
from backend.services.job_manager import Job


EVENT_TAG = "[SCULPT-EVENT]"
ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


_GEN_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")

#: §Ship 51: gate the L2 task-derived calibration path (novel-task steer-rights
#: via K independently-authored competence ladders). DEFAULT ON (Sam's call,
#: 2026-06-15) — set RS_TASK_DERIVED_CALIBRATION=0 to disable. Adds ~3 metric-
#: blind LLM author calls (~60s, pre-GPU) on a NOVEL-task launch only; the 5
#: built-in families take the unchanged built-in calibration path. The firewall
#: still gates steering on a genuine grant, so an enabled-but-failing
#: calibration only ever runs observe-only.
_TASK_DERIVED_ENABLED = os.getenv("RS_TASK_DERIVED_CALIBRATION", "1") == "1"

#: §Ship 53 / §Metric-quality laws (LAW 9): gate the L3 adversarial gaming-
#: archetype check (an independent, metric-blind author proposes OFF-GOAL gaming
#: policies; a metric that scores any in competent territory is GAMEABLE). DEFAULT
#: ON (Sam's call, 2026-06-19) for HIGH-STAKES acceptance — granting steer-rights
#: to a NOVEL-task metric (a one-shot per-launch decision), and AUDIT-ONLY probing
#: of the hand-authored spec_* ground truth a generated metric calibrates against
#: (the surface that never ran on the metric that scored g1-kick-v5). Set
#: RS_ADVERSARIAL_ARCHETYPES=0 to disable. Cost: one metric-blind author call per
#: high-stakes launch (Sam approved). It can DENY a task-derived grant (never the
#: built-in fence — that probe is record-and-warn). Routine per-iteration metric
#: generation makes NO adversarial call, so it stays OFF for routine work by
#: construction. Observe-only stays never-silent: a deny/finding names the policy
#: + its score.
_ADVERSARIAL_ENABLED = os.getenv("RS_ADVERSARIAL_ARCHETYPES", "1") == "1"

#: §Ship 42: dropdown sentinel — "generate the objective metric at launch as the
#: run's first phase" (vs picking an existing built-in / gen:<id>). Ship 43 runs
#: the generation pre-phase and rewrites fitness_metric to gen:<new id> before
#: the cmd is built; until then (or if it yields no accepted metric) the run is
#: blind. Keep in sync with the frontend value in NewRunDialog.tsx / types.ts.
LAUNCH_GEN_SENTINEL = "generate-at-launch"


def _resolve_fitness_metric(project_dir: Path, fitness_metric: str) -> Optional[str]:
    """§Ship 35: map a UI fitness_metric value to the string the sculpt
    CLI's resolve_fitness_fn understands. A generated metric is
    "gen:<id>" → the project's `metrics/<id>/metric.py` path; a built-in
    name passes through. Returns None for an unresolvable/unsafe gen ref
    (the caller drops it → blind loop, never a failed run)."""
    if fitness_metric == LAUNCH_GEN_SENTINEL:
        # §Ship 42: the deferred-generation sentinel must never reach the CLI as
        # a metric name. Ship 43 intercepts it BEFORE this (a generation
        # pre-phase that rewrites it to gen:<id>); reaching here means
        # launch-gen is off/failed → run blind.
        return None
    if fitness_metric.startswith("gen:"):
        gid = fitness_metric[len("gen:"):]
        # §Ship 35 review: validate the id so a crafted ref can't traverse
        # outside the project's metrics dir.
        if not _GEN_ID_RE.match(gid):
            print(f"[run_manager] invalid generated-metric id {gid!r}; "
                  f"running blind", flush=True)
            return None
        metric_py = project_dir / "metrics" / gid / "metric.py"
        if metric_py.is_file():
            return str(metric_py)
        print(f"[run_manager] generated metric {gid!r} not found at "
              f"{metric_py}; running blind", flush=True)
        return None
    return fitness_metric


def steer_allowed(project_dir: Path, fitness_metric: str) -> bool:
    """§Ship 35 review (CRITICAL): the backend — not just the UI — must
    forbid an uncalibrated generated metric from STEERING (self-grading
    firewall). Built-in metrics + calibrated generated metrics may steer;
    an accepted-but-uncalibrated generated metric may only observe."""
    if not fitness_metric.startswith("gen:"):
        return True
    gid = fitness_metric[len("gen:"):]
    if not _GEN_ID_RE.match(gid):
        return False
    meta = project_dir / "metrics" / gid / "meta.json"
    try:
        import json as _json

        return bool(_json.loads(meta.read_text(encoding="utf-8")).get("calibrated"))
    except Exception:  # noqa: BLE001 — unreadable meta → not calibrated → observe
        return False


# ── §Ship 43: launch-time objective-metric generation (run-phase 0) ────
def _robot_task_id(project_dir: Path) -> Optional[str]:
    """Best-effort adapter task_id from config.toml — a robot hint for the
    metric generator (e.g. "Mjlab-Velocity-Flat-Unitree-G1"). Never fatal."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover — py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with (project_dir / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return None
    tid = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
    return str(tid) if isinstance(tid, str) else None


#: §Ship 45: initial generation + up to N-1 user-triggered retries before the
#: run falls back to blind.
_MAX_LAUNCH_GEN_ATTEMPTS = 4


async def _await_gen_decision(
    control_path: Path, cancel: asyncio.Event, *, timeout_s: float = 1800.0,
) -> str:
    """§Ship 45: after a launch-gen rejection, poll the control sidecar for the
    user's decision — "retry" (regenerate) or "blind" (run without a metric).
    The pre-phase holds NO GPU (the sculpt subprocess hasn't spawned), so the
    wait is cheap; a long timeout falls back to "blind" so an unattended run
    still completes rather than hanging. Cancel / stop → "blind"."""
    start_seq = int(read_control_file(control_path).get("gen_decision_seq", 0) or 0)
    waited = 0.0
    step = 1.5
    while waited < timeout_s:
        if cancel.is_set():
            return "blind"
        ctrl = read_control_file(control_path)
        if ctrl.get("stop"):
            return "blind"
        if int(ctrl.get("gen_decision_seq", 0) or 0) != start_seq:
            return "retry" if ctrl.get("gen_decision") == "retry" else "blind"
        await asyncio.sleep(step)
        waited += step
    return "blind"


async def _generate_at_launch(
    job: Job, project_dir: Path, behavior_goal: str,
    control_path: Path, cancel: asyncio.Event, *, n_candidates: int = 1,
) -> Optional[str]:
    """§Ship 43/44/45: generate the objective metric as the run's FIRST phase,
    streaming the Ship-40 stage events into THIS run's event stream, then
    auto-calibrate it. Returns "gen:<id>" on acceptance, else None (→ blind).
    Rejections are surfaced as events (never silent) with the exact validation
    reasons + reviewer concerns; §Ship 45 then PAUSES for a one-click retry
    decision (bounded by `_MAX_LAUNCH_GEN_ATTEMPTS`) before falling back to
    blind. Never raises — a failed generation degrades to a blind run.

    The blocking, multi-LLM-call generation runs in a worker thread; its
    on_event callback fires in that thread, so events are marshalled back onto
    the event loop with call_soon_threadsafe (Job.emit's subscriber queues are
    not thread-safe)."""
    from backend.services import metric_store, sculptor_bridge

    loop = asyncio.get_running_loop()
    robot_hint = _robot_task_id(project_dir)

    def _emit_threadsafe(ev: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(job.emit, ev)

    def _on_event(ev: dict[str, Any]) -> None:
        # Stream into the run timeline (WS) AND the Ship-40 sidecar (parity
        # with the standalone generate UI). Both are best-effort.
        _emit_threadsafe({"type": "metric_generation_progress", "source": "launch_gen", **ev})
        metric_store.write_progress(project_dir, {"active": True, **ev})

    for attempt in range(_MAX_LAUNCH_GEN_ATTEMPTS):
        job.emit({"type": "metric_generation_started", "source": "launch_gen",
                  "behavior_goal": behavior_goal,
                  "attempt": attempt + 1, "max": _MAX_LAUNCH_GEN_ATTEMPTS})
        try:
            rec = await asyncio.to_thread(
                metric_store.generate, project_dir, behavior_goal,
                robot_hint=robot_hint, review=True,
                n_candidates=n_candidates, on_event=_on_event)
        except asyncio.CancelledError:
            # §Ship 45 review (MEDIUM): clear the Ship-40 progress sidecar on
            # Stop. CancelledError is a BaseException that bypasses the
            # `except Exception` below, which otherwise leaves the sidecar stuck
            # at {active:true} → a phantom "generating" spinner in the standalone
            # Generate UI until the next generation. (The orphaned worker thread
            # is bounded by the LLM client's own timeout — to_thread can't be
            # cancelled mid-call.) Mirrors the standalone route's finally-clear.
            metric_store.clear_progress(project_dir)
            raise
        except Exception as e:  # noqa: BLE001 — generation never fails the run
            metric_store.clear_progress(project_dir)
            job.emit({"type": "metric_generation_failed", "source": "launch_gen",
                      "error": f"{type(e).__name__}: {e}"})
            return None
        metric_store.clear_progress(project_dir)

        if rec.get("accepted"):
            gid = rec.get("id")
            job.emit({"type": "metric_generated", "source": "launch_gen",
                      "gen_id": gid, "accepted": True,
                      "behavior_goal": behavior_goal})
            # §Ship 44: auto-calibrate vs the family's built-in ground truth
            # (offline, no GPU). On pass the metric earns steer-rights (the
            # cmd-build's steer_allowed reads the now-calibrated meta.json);
            # else it stays observe-only — the firewall never lets an
            # uncalibrated metric steer.
            builtin = sculptor_bridge.resolve_calibration_builtin(behavior_goal, robot_hint)
            if builtin and gid:
                job.emit({"type": "metric_calibration_started", "source": "launch_gen",
                          "gen_id": gid, "builtin": builtin})
                try:
                    cal = await asyncio.to_thread(
                        metric_store.calibrate, project_dir, gid, builtin)
                    job.emit({"type": "metric_calibration_done", "source": "launch_gen",
                              "gen_id": gid, "builtin": builtin, "method": "builtin",
                              "calibrated": bool(cal.get("calibrated")),
                              "spearman": (cal.get("calibration") or {}).get("spearman"),
                              "trust": (cal.get("trust") or {}).get("trust")})
                except Exception as e:  # noqa: BLE001 — calibration failure ≠ run failure
                    job.emit({"type": "metric_calibration_done", "source": "launch_gen",
                              "gen_id": gid, "builtin": builtin, "calibrated": False,
                              "error": f"{type(e).__name__}: {e}"})
                # §Metric-quality laws (LAW 9): AUDIT-ONLY adversarial probe of the
                # HAND-AUTHORED ground-truth spec_* metric this generated metric is
                # calibrating AGAINST — the gate that never ran on spec_g1_kick, the
                # metric that scored g1-kick-v5. Records + warns; NEVER revokes the
                # fence (built-ins are the trusted calibration anchor). Flag-gated;
                # scoped to families with a curated loser set (kick today). Bounded
                # by a timeout; never fails the run.
                if _ADVERSARIAL_ENABLED and sculptor_bridge.has_spec_audit(builtin):
                    job.emit({"type": "metric_spec_audit_started",
                              "source": "launch_gen", "builtin": builtin})
                    try:
                        audit = await asyncio.wait_for(
                            asyncio.to_thread(sculptor_bridge.audit_builtin_spec_metric,
                                              builtin, behavior_goal, robot_hint),
                            timeout=180.0)
                        job.emit({"type": "metric_spec_audit", "source": "launch_gen",
                                  "builtin": builtin, "audit_only": True,
                                  "ran": bool(audit.get("ran")),
                                  "gameable": bool(audit.get("gameable")),
                                  "worst_name": audit.get("worst_name"),
                                  "worst_gaming": audit.get("worst_gaming"),
                                  "coverage_gaps": audit.get("coverage_gaps"),
                                  "reason": audit.get("reason")})
                    except asyncio.TimeoutError:
                        job.emit({"type": "metric_spec_audit", "source": "launch_gen",
                                  "builtin": builtin, "audit_only": True, "ran": False,
                                  "reason": "spec audit timed out (>180s) — not enforced"})
                    except Exception as e:  # noqa: BLE001 — audit never affects the run
                        job.emit({"type": "metric_spec_audit", "source": "launch_gen",
                                  "builtin": builtin, "audit_only": True, "ran": False,
                                  "error": f"{type(e).__name__}: {e}"})
            elif _TASK_DERIVED_ENABLED:
                # §Ship 51: novel task (no built-in) → earn steer-rights by
                # ranking K independently-authored competence ladders. No GPU is
                # held (pre-phase); bounded by a timeout so an unattended run
                # still completes. Never fails the run — any failure mode is an
                # observe-only reason (the firewall keeps an uncalibrated metric
                # from steering regardless).
                job.emit({"type": "metric_calibration_started", "source": "launch_gen",
                          "gen_id": gid, "method": "task_derived", "k_sources": 3})
                # §round-5: stamp a calibration token so a >300s orphan (asyncio.to_thread
                # can't be cancelled) cannot persist calibrated=true AFTER we surface
                # 'observe-only' — on timeout we re-stamp and the orphan's late write is a
                # no-op. None-safe: stamp_cal_token never raises.
                cal_token = metric_store.stamp_cal_token(project_dir, gid)
                try:
                    cal = await asyncio.wait_for(
                        asyncio.to_thread(metric_store.calibrate_task_derived,
                                          project_dir, gid, behavior_goal, robot_hint,
                                          adversarial=_ADVERSARIAL_ENABLED,
                                          expect_token=cal_token, require_token=True),
                        timeout=300.0)
                    c = cal.get("calibration") or {}
                    adv = c.get("adversarial") or {}
                    job.emit({"type": "metric_calibration_done", "source": "launch_gen",
                              "gen_id": gid, "method": "task_derived",
                              "calibrated": bool(cal.get("calibrated")),
                              "spearman": c.get("rho_min"),
                              "rho_min": c.get("rho_min"),
                              "agreement_fraction": c.get("agreement_fraction"),
                              "adversarial_ran": bool(adv.get("ran")),
                              "gameable": bool(adv.get("gameable")),
                              "trust": (cal.get("trust") or {}).get("trust"),
                              "reason": c.get("reason")})
                except asyncio.TimeoutError:
                    # §round-5/6: force calibrated=false + re-stamp the token (atomically)
                    # so the still-running orphan thread's late write is rejected AND can't
                    # resurrect calibrated=true behind this observe-only verdict (even via a
                    # lost-update race). Best-effort; never fails the run.
                    try:
                        metric_store.supersede_calibration(project_dir, gid)
                    except Exception:  # noqa: BLE001
                        pass
                    job.emit({"type": "metric_calibration_done", "source": "launch_gen",
                              "gen_id": gid, "method": "task_derived", "calibrated": False,
                              "reason": "task-derived calibration timed out (>300s) — observe-only"})
                except Exception as e:  # noqa: BLE001 — calibration ≠ run failure
                    job.emit({"type": "metric_calibration_done", "source": "launch_gen",
                              "gen_id": gid, "method": "task_derived", "calibrated": False,
                              "error": f"{type(e).__name__}: {e}",
                              "reason": "task-derived calibration crashed — observe-only"})
            else:
                # Reached only when there is no built-in AND the task-derived
                # flag is off — name both so the observe-only state is specific.
                job.emit({"type": "metric_calibration_skipped", "source": "launch_gen",
                          "gen_id": gid,
                          "reason": "no matching built-in ground truth; "
                                    "task-derived calibration disabled — observe-only"})
            return f"gen:{gid}"

        # Rejected — surface WHY (validation gate reasons + reviewer concerns).
        can_retry = (attempt + 1 < _MAX_LAUNCH_GEN_ATTEMPTS) and not cancel.is_set()
        job.emit({"type": "metric_generation_rejected", "source": "launch_gen",
                  "gen_id": rec.get("id"),
                  "reasons": list(rec.get("reasons") or []),
                  "concerns": list((rec.get("review") or {}).get("concerns") or []),
                  "can_retry": can_retry})
        if not can_retry:
            return None
        # §Ship 45: pause for a one-click decision — retry generation or
        # continue blind. No GPU is held (the sculpt subprocess hasn't spawned).
        job.emit({"type": "metric_generation_awaiting_decision", "source": "launch_gen",
                  "gen_id": rec.get("id"),
                  "attempt": attempt + 1, "max": _MAX_LAUNCH_GEN_ATTEMPTS})
        decision = await _await_gen_decision(control_path, cancel)
        if decision != "retry":
            return None
        # loop → regenerate
    return None


# ── §Ship 39 (H1): interactive control sidecar ─────────────────────────
def control_file_path(project_dir: Path, run_id: str) -> Path:
    """Deterministic path for a run's interactive control sidecar. Both the
    runner (writes the initial state) and the PATCH /control route (writes
    updates) compute it the same way → no job-lookup race."""
    return Path(project_dir) / "runs" / f"_control_{run_id}.json"


def write_control_file(path: Path, data: dict[str, Any]) -> None:
    """Atomically write the control sidecar (tmp + rename) so the sculpt
    subprocess never reads a half-written file at its poll."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def read_control_file(path: Path) -> dict[str, Any]:
    """Read the control sidecar; missing / unreadable → a safe auto default."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        pass
    return {"mode": "auto", "resume_token": 0, "feedback": None, "stop": False}


# ── public API ────────────────────────────────────────────────────────
def run_sculpt_job(
    *,
    project_dir: Path,
    run_params: dict[str, Any],
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Return an async callable for `JobManager.submit`.

    `run_params` keys: behavior_goal (str), iterations (int), no_kg
    (bool), dry_run (bool).
    """
    behavior_goal = str(run_params["behavior_goal"])
    iterations = int(run_params.get("iterations", 10))
    no_kg = bool(run_params.get("no_kg", False))
    dry_run = bool(run_params.get("dry_run", False))
    # Phase-4 overrides from NewRunDialog → Advanced tab. Pre-fix these
    # 4 fields were silently dropped between the route and the sculpt
    # CLI — user saw "1500 rsl_rl iters" even when they typed 100.
    # Any field here also needs a matching CLI flag in sculptor/cli.py's
    # `run` command OR an env var the CLI reads.
    training_iterations = run_params.get("training_iterations")
    # Per-run hardware overrides must reach the subprocess; the UI presents
    # them as launch-scoped safety controls (especially important on 8 GiB
    # laptop GPUs), so silently falling back to config.toml is unsafe.
    num_envs_override = run_params.get("num_envs_override")
    device_override = run_params.get("device_override")
    # `expand_kg` is a sculpt-loop feature not wired up to the CLI yet —
    # pass-through flag only.
    expand_kg = bool(run_params.get("expand_kg", False))
    # §Ship-7: rollout-video + RL knobs. Forwarded to `sculpt run` as
    # long-form CLI flags (see sculptor/cli.py::run). None means the
    # runner or config.toml default wins.
    max_episode_steps = run_params.get("max_episode_steps")
    playback_speed = run_params.get("playback_speed")
    render_every = run_params.get("render_every")
    rollout_fps = run_params.get("rollout_fps")
    render_width = run_params.get("render_width")
    render_height = run_params.get("render_height")
    rollout_episodes = run_params.get("rollout_episodes")
    seed = run_params.get("seed")
    auto_adjust_physics = run_params.get("auto_adjust_physics")
    early_stop_enabled = run_params.get("early_stop_enabled")
    early_stop_patience = run_params.get("early_stop_patience")
    # §Ship 34: objective fitness-in-the-loop (spec-metric name). Forwarded
    # to `sculpt run --fitness-metric`; None = the blind loop.
    fitness_metric = run_params.get("fitness_metric")
    # §Ship 35: observe vs steer (default steer). Only meaningful with a
    # metric set; harmless otherwise.
    fitness_mode = run_params.get("fitness_mode")
    # §best-of-N: candidates to sample for a generate-at-launch metric (1 →
    # single-shot). Only used on the LAUNCH_GEN_SENTINEL path below.
    metric_n_candidates = int(run_params.get("metric_n_candidates", 1) or 1)
    # §Ship 48: patience for the fitness-plateau early-stop (the live early
    # stop; the early_stop_* knobs above are a no-op for it). Only meaningful
    # with a metric set. None → sculpt-lib default (2).
    fitness_patience = run_params.get("fitness_patience")
    # §Ship 39 (H1): interactive start mode. "manual" = pause for human
    # feedback at each iteration boundary; "auto" = run straight through.
    # A control sidecar is ALWAYS written (deterministic path) so the
    # Auto/Manual toggle works at ANY point mid-run, regardless of start mode.
    start_mode = run_params.get("start_mode")
    start_mode = start_mode if start_mode in ("manual", "auto") else "auto"

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        # Prepare env — strip empty ANTHROPIC_API_KEY so the .env file
        # in the project (or sculptor's own) can win; matches the
        # sculptor/__init__.py behavior.
        env = {k: v for k, v in os.environ.items() if v or k != "ANTHROPIC_API_KEY"}
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Point the sculpt subprocess at the same KG path the UI reads
        # through — `project_kg_db_path` returns the shared user-wide DB
        # for new projects and the legacy per-project DB only when one
        # already exists on disk. Hardcoding `<project>/kg/graph.db`
        # here used to create an empty legacy DB on every first run,
        # which then shadowed the shared DB (papers=46) and left the UI
        # showing "No papers in the KG yet" for that project.
        from backend.services.kg_store import project_kg_db_path

        env["SCULPTOR_KG_PATH"] = str(project_kg_db_path(project_dir))

        # §Ship 23d: inject the UI's saved remote-GPU settings as
        # SCULPTOR_REMOTE_* env vars (they win over any [remote] table
        # in config.toml). Empty dict when disabled — fully local.
        # Projects live at <projects_root>/<slug>, so parent is root.
        from backend.services.remote_settings import remote_env

        env.update(remote_env(project_dir.parent))

        # §Ship 39 (H1): the interactive control sidecar. §Ship 45: written
        # BEFORE the launch-gen pre-phase so the PATCH /control route can deliver
        # a retry/continue decision while generation is paused on a rejection.
        control_path = control_file_path(project_dir, job.job_id)
        write_control_file(control_path, {
            "mode": start_mode, "resume_token": 0,
            "feedback": None, "stop": False,
        })

        # §Ship 43: launch-time generation — if the user picked "Generate a
        # metric from this goal (at launch)", generate the objective metric as
        # the run's FIRST phase (events streamed into this run's stream), then
        # steer the rest of the run with it (observe-only until calibrated).
        # Opt-in via the dropdown sentinel; SCULPTOR_LAUNCH_GEN=0 disables it
        # (→ blind, the Ship-42 fallback). Other fitness_metric values are
        # untouched, so default run behavior stays green.
        eff_fitness_metric = fitness_metric
        eff_fitness_mode = fitness_mode
        if (eff_fitness_metric == LAUNCH_GEN_SENTINEL
                and os.environ.get("SCULPTOR_LAUNCH_GEN", "1") != "0"
                and not cancel.is_set()):
            eff_fitness_metric = await _generate_at_launch(
                job, project_dir, behavior_goal, control_path, cancel,
                n_candidates=metric_n_candidates)
            # §Ship 44: a launch-generated metric STEERS iff it earned
            # steer-rights via launch-time calibration; request steer and let
            # the steer_allowed firewall below downgrade to observe otherwise.
            if eff_fitness_metric is not None:
                eff_fitness_mode = "steer"

        cmd = [
            sys.executable, "-m", "sculptor.cli",
            "run", behavior_goal,
            "--config", str(project_dir / "config.toml"),
            "--iterations", str(iterations),
            # Always resume: sculpt_run skips iter N's training when
            # `iter_N/checkpoint.pt` is already on disk (per-phase
            # resume in sculpt.py). So an overnight run that errors
            # at iter 7 can be retried in the morning without redoing
            # the first 7 iters' 22 min of GPU time each. On a fresh
            # project, resume is a no-op — `latest_n_before_loop=0` so
            # `start_iter=0` either way.
            "--resume",
        ]
        if no_kg:
            cmd.append("--no-kg")
        if dry_run:
            cmd.append("--dry-run")
        if training_iterations is not None:
            cmd += ["--steps-per-iter", str(int(training_iterations))]
        if num_envs_override is not None:
            cmd += ["--num-envs", str(int(num_envs_override))]
        if device_override is not None:
            cmd += ["--device", str(device_override)]
        if max_episode_steps is not None:
            cmd += ["--max-episode-steps", str(int(max_episode_steps))]
        if playback_speed is not None:
            cmd += ["--playback-speed", str(float(playback_speed))]
        if render_every is not None:
            cmd += ["--render-every", str(int(render_every))]
        if rollout_fps is not None:
            cmd += ["--rollout-fps", str(float(rollout_fps))]
        if render_width is not None:
            cmd += ["--render-width", str(int(render_width))]
        if render_height is not None:
            cmd += ["--render-height", str(int(render_height))]
        if rollout_episodes is not None:
            cmd += ["--rollout-episodes", str(int(rollout_episodes))]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        if auto_adjust_physics is not None:
            # typer's --flag/--no-flag convention.
            cmd.append(
                "--auto-adjust-physics" if auto_adjust_physics
                else "--no-auto-adjust-physics"
            )
        if early_stop_enabled is not None:
            cmd.append(
                "--early-stop" if early_stop_enabled else "--no-early-stop"
            )
        if early_stop_patience is not None:
            cmd += ["--early-stop-patience", str(int(early_stop_patience))]
        final_fitness_mode: Optional[str] = None
        if eff_fitness_metric:
            # §Ship 35: a generated metric is referenced as "gen:<id>";
            # resolve it to the project's metric.py path the CLI can load.
            # A built-in name passes through unchanged. An unresolvable
            # gen ref is dropped (blind loop) rather than failing the run.
            # §Ship 43: eff_fitness_metric is the LAUNCH-GENERATED gen:<id>
            # when the sentinel was used (else the original selection).
            resolved = _resolve_fitness_metric(project_dir, str(eff_fitness_metric))
            if resolved is not None:
                cmd += ["--fitness-metric", resolved]
                # §Ship 35 review: downgrade steer→observe for an uncalibrated
                # generated metric (backend-enforced, not just the UI). §Ship 44:
                # a launch-generated metric that PASSED launch-time calibration
                # has calibrated=true in its meta → steer_allowed lets it steer.
                eff_mode = eff_fitness_mode
                if eff_mode == "steer" and not steer_allowed(
                        project_dir, str(eff_fitness_metric)):
                    eff_mode = "observe"
                if eff_mode in ("observe", "steer"):
                    cmd += ["--fitness-mode", str(eff_mode)]
                    final_fitness_mode = eff_mode
                # §Ship 48: forward the fitness-plateau patience (the live
                # early stop). Only meaningful alongside a resolved metric;
                # sculpt ignores it without a fitness_fn.
                if fitness_patience is not None:
                    cmd += ["--fitness-patience", str(int(fitness_patience))]

        # §Ship 39 (H1): the interactive control sidecar (written above, before
        # the launch-gen pre-phase) lets the sculpt subprocess poll for the
        # Auto/Manual toggle at any iteration boundary.
        cmd += ["--control-file", str(control_path)]
        job.params["control_file"] = str(control_path)
        job.params["mode"] = start_mode

        job.params.setdefault("cmd", cmd)
        job.params.setdefault("behavior_goal", behavior_goal)
        job.params.setdefault("iterations", iterations)
        job.params.setdefault("no_kg", no_kg)
        job.params.setdefault("dry_run", dry_run)
        if training_iterations is not None:
            job.params.setdefault("training_iterations", int(training_iterations))
        if num_envs_override is not None:
            job.params.setdefault("num_envs_override", int(num_envs_override))
        if device_override is not None:
            job.params.setdefault("device_override", str(device_override))
        if expand_kg:
            job.params.setdefault("expand_kg", expand_kg)
        # §Ship-7: stash the new params so the Runs-tab summary can
        # surface them (non-None entries only to keep the payload lean).
        for key, val in (
            ("max_episode_steps", max_episode_steps),
            ("playback_speed", playback_speed),
            ("render_every", render_every),
            ("rollout_fps", rollout_fps),
            ("rollout_episodes", rollout_episodes),
            ("seed", seed),
            ("auto_adjust_physics", auto_adjust_physics),
            ("early_stop_enabled", early_stop_enabled),
            ("early_stop_patience", early_stop_patience),
            # §Ship 43: persist the RESOLVED metric (the launch-generated
            # gen:<id> when the sentinel was used) so the Runs view shows what
            # actually steers; None (rejected/blind) is skipped below. §Ship 44:
            # persist the POST-firewall effective mode (steer iff calibrated).
            ("fitness_metric", eff_fitness_metric),
            ("fitness_mode", final_fitness_mode),
            ("fitness_patience", fitness_patience),
        ):
            if val is not None:
                job.params.setdefault(key, val)

        # Mirror all stdout into a per-run log file so the "10k lines on
        # disk" reconnect history from Prompt 8 is durable after the
        # in-memory ring rolls off.
        runs_dir = project_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        log_path = runs_dir / f"_run_{job.job_id}.log"
        job.params["log_file"] = str(log_path)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merged stream
            env=env,
            creationflags=creationflags,
        )
        job.params["pid"] = proc.pid

        # Start cooperative cancel task — when cancel is signalled we
        # send CTRL_BREAK on Windows / SIGTERM elsewhere, then SIGKILL
        # after a grace period if needed.
        killer_task = asyncio.create_task(
            _kill_on_cancel(proc, cancel, grace_s=5.0)
        )

        # Filesystem watcher. Bounded to this project's `runs/` dir +
        # `rewards/` so we pick up iter dirs and new reward files.
        watcher_task = asyncio.create_task(
            _fs_watcher(job, project_dir, cancel)
        )

        # stdout reader.
        reader_task = asyncio.create_task(
            _stream_stdout(job, proc, log_path)
        )

        # Heartbeat: periodically re-emit metric history + scan for any
        # iter dirs we might have missed (watchfiles on Windows
        # sometimes misses the dir-creation event).
        heartbeat_task = asyncio.create_task(
            _heartbeat(job, project_dir, cancel)
        )

        rc = await proc.wait()
        cancel.set()
        for t in (reader_task, watcher_task, heartbeat_task, killer_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Final summary from disk (canonical: sculpt wrote these).
        metric_history = _read_metric_history(project_dir)
        iterations_run = len(metric_history)
        if rc == 0:
            job.emit({
                "type": "run_completed",
                "return_code": rc,
                "iterations_run": iterations_run,
                "primary_metric_history": metric_history,
            })
        else:
            classification = _classify_run_failure(log_path, project_dir)
            friendly = (
                classification.title
                if classification.kind != "unknown"
                else f"sculpt exited with code {rc}"
            )
            job.status = "errored"
            job.error = friendly
            # Stash the full classification on job.params so the REST
            # view (`_run_summary`) can surface it without re-parsing
            # events. UI renders a one-click remediation button keyed
            # off `error_classification.action.kind`.
            job.params["error_classification"] = {
                "kind": classification.kind,
                "title": classification.title,
                "detail": classification.detail,
                "suggestions": list(classification.suggestions or []),
                "problem_type": classification.problem_type,
                "action": classification.action,
            }
            job.emit({
                "type": "run_errored",
                "return_code": rc,
                "iterations_run": iterations_run,
                "error": friendly,
                "error_kind": classification.kind,
                "error_detail": classification.detail,
                "error_suggestions": classification.suggestions,
                "error_problem_type": classification.problem_type,
                "error_action": classification.action,
            })

        return {
            "return_code": rc,
            "iterations_run": iterations_run,
            "primary_metric_history": metric_history,
            "log_file": str(log_path),
            "iterations": _iter_events(job),
        }

    return _runner


# ── stdout capture ────────────────────────────────────────────────────
async def _stream_stdout(
    job: Job,
    proc: asyncio.subprocess.Process,
    log_path: Path,
) -> None:
    """Read `proc.stdout` line-by-line. Every line is emitted as
    `log_line`; lines starting with `[SCULPT-EVENT] {...}` are ALSO
    parsed and emitted as typed events (the JSON payload's `type`
    becomes the event type)."""
    assert proc.stdout is not None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = log_path.open("a", encoding="utf-8", buffering=1)
    except OSError:
        logf = None

    try:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if logf is not None:
                try:
                    logf.write(text + "\n")
                except OSError:
                    pass
            job.emit({"type": "log_line", "text": text})
            if text.startswith(EVENT_TAG):
                payload_str = text[len(EVENT_TAG):].strip()
                try:
                    payload = json.loads(payload_str)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(payload, dict) and payload.get("type"):
                    ev = dict(payload)
                    # Mark its provenance so the frontend can dedupe
                    # against filesystem-derived events of the same type.
                    ev["source"] = "stdout"
                    job.emit(ev)
    finally:
        if logf is not None:
            try:
                logf.close()
            except OSError:
                pass


# ── kill on cancel ────────────────────────────────────────────────────
async def _kill_on_cancel(
    proc: asyncio.subprocess.Process,
    cancel: asyncio.Event,
    *,
    grace_s: float,
) -> None:
    await cancel.wait()
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            # terminate() on Windows sends CTRL_BREAK_EVENT to the
            # process group. We spawned with CREATE_NEW_PROCESS_GROUP
            # so this only hits the child.
            proc.terminate()
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return


# ── filesystem watcher (primary truth, per R2) ───────────────────────
async def _fs_watcher(
    job: Job, project_dir: Path, cancel: asyncio.Event
) -> None:
    """Stream filesystem events from the project dir and map them to
    typed run events. Safe to run alongside stdout parsing — they can
    emit overlapping events, but the filesystem ones are tagged
    `source=fs` and should be preferred by the frontend on conflict."""
    try:
        from watchfiles import Change, awatch
    except Exception:  # noqa: BLE001
        return

    seen_iters: set[int] = set()
    seen_iter_done: set[int] = set()
    seen_rollouts: set[int] = set()
    seen_rewards: set[int] = set()
    seen_citations: set[tuple[int, str]] = set()
    seen_realism: set[int] = set()

    # Pre-SEED the dedup sets with iterations/artifacts/rewards already on disk
    # at run start (NO emit) so a RESUMED run never surfaces the PRIOR run's
    # iters as 'running' / re-applies their edits. The live subprocess stdout
    # re-emits iter_started/iter_completed for the iters this run executes, and
    # the awatch loop below catches anything CREATED during this run (incl. a
    # fresh run's iter_0 written after this seed).
    _preseed_seen(
        project_dir,
        seen_iters=seen_iters,
        seen_iter_done=seen_iter_done,
        seen_rollouts=seen_rollouts,
        seen_rewards=seen_rewards,
        seen_citations=seen_citations,
        seen_realism=seen_realism,
    )

    watched = [project_dir / "runs", project_dir / "rewards", project_dir / "reports"]
    for p in watched:
        p.mkdir(parents=True, exist_ok=True)
    try:
        async for changes in awatch(
            *watched,
            stop_event=_StopEventAdapter(cancel),
            step=250,
            recursive=True,
        ):
            _handle_fs_changes(
                job, project_dir, changes,
                seen_iters=seen_iters,
                seen_iter_done=seen_iter_done,
                seen_rollouts=seen_rollouts,
                seen_rewards=seen_rewards,
                seen_citations=seen_citations,
                seen_realism=seen_realism,
            )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — never let the watcher take down the job
        return


class _StopEventAdapter:
    """watchfiles expects a threading-style stop event. Adapt asyncio.Event."""

    def __init__(self, cancel: asyncio.Event) -> None:
        self._cancel = cancel

    def is_set(self) -> bool:
        return self._cancel.is_set()


def _preseed_seen(
    project_dir: Path,
    *,
    seen_iters: set[int],
    seen_iter_done: set[int],
    seen_rollouts: set[int],
    seen_rewards: set[int],
    seen_citations: set[tuple[int, str]],
    seen_realism: set[int] | None = None,
) -> None:
    """Populate the watcher's dedup sets with iterations/artifacts/rewards
    ALREADY on disk when this run STARTS — WITHOUT emitting any event.

    Those belong to a PRIOR run (or this run is a `--resume`), so the watcher
    must NOT surface them as 'started/running' iteration cards or re-applied
    edits in THIS run's timeline. (The bug this fixes: a resumed run showed the
    previous run's iters perpetually RUNNING, because the fs watcher emitted
    `iter_started` for every on-disk iter_<n> dir but `iter_completed` only ever
    comes from the live subprocess stdout for the iters it actually runs.)

    The live subprocess stdout drives the timeline for the iters this run
    executes — it re-emits `iter_started`/`iter_completed` for the resumed range
    — and the `awatch` loop + `_handle_fs_changes` catch anything CREATED during
    this run. Artifact sets are keyed on artifact VALIDITY (the same predicates
    `_check_iter_artifacts` uses), not mere dir existence, so a half-written
    prior iter can't strand a stray rollout_done/diagnosed. `seen_citations` is
    covered transitively: it is only written inside `_emit_edit_applied`, which
    is gated by `seen_rewards`."""
    _ = seen_citations  # transitively covered via seen_rewards; kept for parity
    runs_dir = project_dir / "runs"
    if runs_dir.is_dir():
        for d in runs_dir.iterdir():
            m = ITER_DIR_RE.match(d.name)
            if not m:
                continue
            n = int(m.group(1))
            seen_iters.add(n)
            mp4 = d / "rollout" / "rollout.mp4"
            if mp4.is_file() and mp4.stat().st_size > 2048:
                seen_rollouts.add(n)
            if seen_realism is not None and (d / "realism_audit.json").is_file():
                try:
                    if isinstance(json.loads(
                            (d / "realism_audit.json").read_text(encoding="utf-8")),
                            dict):
                        seen_realism.add(n)
                except Exception:  # noqa: BLE001
                    pass
            if (d / "diagnosis.json").is_file():
                try:
                    if json.loads((d / "diagnosis.json").read_text(encoding="utf-8")):
                        seen_iter_done.add(n)
                except Exception:  # noqa: BLE001
                    pass

    rewards_dir = project_dir / "rewards"
    if rewards_dir.is_dir():
        for p in rewards_dir.iterdir():
            mv = re.fullmatch(r"v(\d+)\.py", p.name)
            if mv and int(mv.group(1)) > 0:
                seen_rewards.add(int(mv.group(1)))


def _handle_fs_changes(
    job: Job,
    project_dir: Path,
    changes: set[tuple[int, str]],
    *,
    seen_iters: set[int],
    seen_iter_done: set[int],
    seen_rollouts: set[int],
    seen_rewards: set[int],
    seen_citations: set[tuple[int, str]],
    seen_realism: set[int] | None = None,
) -> None:
    runs_dir = project_dir / "runs"
    rewards_dir = project_dir / "rewards"
    for change_type, raw_path in changes:
        p = Path(raw_path)
        try:
            rel = p.relative_to(project_dir)
        except ValueError:
            continue
        parts = rel.parts

        # runs/iter_<n>/...
        if len(parts) >= 2 and parts[0] == "runs":
            m = ITER_DIR_RE.match(parts[1])
            if m:
                n = int(m.group(1))
                if n not in seen_iters:
                    seen_iters.add(n)
                    job.emit({"type": "iter_started", "source": "fs", "iter": n})
                iter_dir = runs_dir / parts[1]
                _check_iter_artifacts(
                    job, iter_dir, n,
                    seen_rollouts=seen_rollouts,
                    seen_iter_done=seen_iter_done,
                    seen_realism=seen_realism,
                )

        # rewards/v<n>.py
        if len(parts) == 2 and parts[0] == "rewards":
            mv = re.fullmatch(r"v(\d+)\.py", parts[1])
            if mv:
                v = int(mv.group(1))
                if v > 0 and v not in seen_rewards and (rewards_dir / parts[1]).is_file():
                    seen_rewards.add(v)
                    _emit_edit_applied(job, rewards_dir, v, seen_citations)

        # reports/metric_history.json
        if parts == ("reports", "metric_history.json"):
            history = _read_metric_history(project_dir)
            if history:
                job.emit({
                    "type": "metric_history",
                    "source": "fs",
                    "history": history,
                })


def _check_iter_artifacts(
    job: Job,
    iter_dir: Path,
    n: int,
    *,
    seen_rollouts: set[int],
    seen_iter_done: set[int],
    seen_realism: set[int] | None = None,
) -> None:
    mp4 = iter_dir / "rollout" / "rollout.mp4"
    # An MP4 becomes visible before ffmpeg writes its closing moov atom.  The
    # rollout runner writes behavior.json only after video encoding and all
    # trajectory artifacts are closed, so use it as the generic readiness
    # marker.  Emitting on file-size alone permanently marked the iteration as
    # seen while the clip worker was still receiving "moov atom not found".
    behavior = iter_dir / "rollout" / "behavior.json"
    if (n not in seen_rollouts and mp4.is_file()
            and mp4.stat().st_size > 2048 and behavior.is_file()):
        seen_rollouts.add(n)
        job.emit({
            "type": "rollout_done",
            "source": "fs",
            "iter": n,
            "size_bytes": mp4.stat().st_size,
        })
        # Kick off live-clip rendering. Fire-and-forget — streamer
        # handles its own concurrency gate and never blocks us here.
        streamer = _streamer_for(job)
        project_dir = iter_dir.parent.parent  # runs/ → project root
        asyncio.create_task(
            streamer.maybe_render(job, n, project_dir)
        )
    # §7.3: realism audit. Emit BEFORE `diagnosed` so the iter timeline
    # picks up the verdict chip alongside failure_modes once both files
    # land (sculpt.py writes realism_audit.json between rollout and
    # diagnose, so the fs order holds).
    if seen_realism is not None and n not in seen_realism:
        audit_path = iter_dir / "realism_audit.json"
        if audit_path.is_file():
            try:
                audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                audit_payload = None
            if isinstance(audit_payload, dict):
                seen_realism.add(n)
                job.emit({
                    "type": "realism_audited",
                    "source": "fs",
                    "iter": n,
                    "verdict": audit_payload.get("verdict"),
                    "audit": audit_payload,
                })
    diag = iter_dir / "diagnosis.json"
    if diag.is_file():
        try:
            payload = json.loads(diag.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = None
        if payload and n not in seen_iter_done:
            seen_iter_done.add(n)
            job.emit({
                "type": "diagnosed",
                "source": "fs",
                "iter": n,
                "failure_modes": list(payload.get("failure_modes") or []),
                "confidence": payload.get("confidence"),
                "n_edits": len(payload.get("proposed_edits") or []),
            })


# ── per-job rollout streamer (lazy-constructed) ──────────────────────
def _streamer_for(job: Job):
    """Returns a RolloutStreamer attached to this job. Lazy so imports
    stay lightweight for tests that don't need clip rendering."""
    from backend.services.rollout_streamer import RolloutStreamer

    s = getattr(job, "_rollout_streamer", None)
    if s is None:
        s = RolloutStreamer()
        job._rollout_streamer = s  # type: ignore[attr-defined]
    return s


def _emit_edit_applied(
    job: Job,
    rewards_dir: Path,
    new_version: int,
    seen_citations: set[tuple[int, str]],
) -> None:
    new_path = rewards_dir / f"v{new_version}.py"
    prev_path = rewards_dir / f"v{new_version - 1}.py"
    if not new_path.is_file():
        return
    try:
        new_refs = _extract_reference_arxiv_ids(new_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        new_refs = []
    try:
        prev_refs = (
            _extract_reference_arxiv_ids(prev_path.read_text(encoding="utf-8"))
            if prev_path.is_file() else []
        )
    except Exception:  # noqa: BLE001
        prev_refs = []
    added = [r for r in new_refs if r not in prev_refs]
    job.emit({
        "type": "edit_applied",
        "source": "fs",
        "iter": new_version - 1,  # conventional mapping
        "reward_version_after": new_version,
        "reward_version_before": new_version - 1,
        "paper_refs": new_refs,
    })
    for aid in added:
        key = (new_version, aid)
        if key in seen_citations:
            continue
        seen_citations.add(key)
        job.emit({
            "type": "citation_added",
            "source": "fs",
            "iter": new_version - 1,
            "reward_version": new_version,
            "arxiv_id": aid,
        })


def _extract_reference_arxiv_ids(source: str) -> list[str]:
    """AST-safe extraction of REWARD_SPEC.references[*].arxiv_id — no
    `exec`, matches the pattern we use in reward_store.py."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in tree.body:
        target = None
        value_node = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "REWARD_SPEC":
                    target, value_node = t, node.value
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "REWARD_SPEC" and node.value is not None:
                target, value_node = node.target, node.value
        if target and value_node is not None:
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, SyntaxError):
                return []
            if not isinstance(value, dict):
                return []
            refs = value.get("references") or []
            return [
                str(r["arxiv_id"])
                for r in refs
                if isinstance(r, dict) and "arxiv_id" in r
            ]
    return []


# ── heartbeat ─────────────────────────────────────────────────────────
async def _heartbeat(
    job: Job, project_dir: Path, cancel: asyncio.Event
) -> None:
    """Periodically refresh job.progress + metric_history.  The
    filesystem watcher covers rapid events; the heartbeat covers slow
    or missed ones."""
    while not cancel.is_set():
        try:
            history = _read_metric_history(project_dir)
            iterations_run = len(history)
            requested = int(job.params.get("iterations") or 0)
            job.progress = (
                min(1.0, iterations_run / requested) if requested > 0 else None
            )
            if history:
                job.params["primary_metric_history"] = history
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(cancel.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            continue
        else:
            return


def _read_metric_history(project_dir: Path) -> list[Optional[float]]:
    path = project_dir / "reports" / "metric_history.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    hist = payload.get("history") if isinstance(payload, dict) else None
    if not isinstance(hist, list):
        return []
    out: list[Optional[float]] = []
    for v in hist:
        if v is None:
            out.append(None)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(None)
    return out


# ── iteration summary from events ─────────────────────────────────────
def _iter_events(job: Job) -> list[dict[str, Any]]:
    """Fold the job's event log into a per-iteration summary. Used both
    by REST detail responses and by the final `result` dict on job
    completion so the frontend can reconstruct full timeline state from
    a single GET."""
    by_iter: dict[int, dict[str, Any]] = {}
    for ev in job.events:
        etype = ev.get("type")
        iter_idx = ev.get("iter")
        if not isinstance(iter_idx, int):
            continue
        slot = by_iter.setdefault(
            iter_idx,
            {
                "iter_index": iter_idx,
                "status": "running",
                "started_at": None,
                "completed_at": None,
                "reward_version_before": None,
                "reward_version_after": None,
                "primary_metric": None,
                "metric_delta": None,
                "failure_modes": [],
                "edit_count": None,
                "paper_refs": [],
                "rollout_ready": False,
                "diagnosed": False,
                "realism_audit": None,
                "physics_edit_suggestion": None,
                # §Ship 48: edits the diagnoser WANTED but couldn't ground
                # because the adapter doesn't expose the needed field
                # (requires_env_extension). None until an iter defers ≥1.
                "env_extension_suggestion": None,
                # §env generalization: env-curriculum change applied at
                # this iter's boundary (None until env_spec_updated fires).
                "env_spec_update": None,
                # §Ship 34: objective fitness-in-the-loop (None for blind runs).
                "fitness": None,
                "best_fitness": None,
                # §Convergence loop 1: dense sub-success progress (ranking
                # signal below the completion gate). None when the metric
                # doesn't emit progress_score.
                "progress": None,
            },
        )
        if etype == "iter_started":
            slot["started_at"] = ev.get("ts")
            if "reward_version_before" in ev:
                slot["reward_version_before"] = ev.get("reward_version_before")
        elif etype == "rollout_done":
            slot["rollout_ready"] = True
        elif etype == "physics_edit_suggested":
            # §7.4: store the prompt text so the UI can surface an "apply
            # fix" chip + pre-fill the Physics tab's textarea. Keep the
            # top_joints list around so the chip tooltip matches what the
            # realism audit surfaced.
            prompt = ev.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                slot["physics_edit_suggestion"] = {
                    "prompt": prompt,
                    "verdict": ev.get("verdict"),
                    "top_joints_saturation":
                        ev.get("top_joints_saturation") or [],
                    "auto_apply_state": "pending",
                }
        elif etype in (
            "physics_auto_apply_started",
            "physics_auto_applied",
            "physics_auto_apply_rejected",
            "physics_auto_apply_errored",
            "physics_auto_apply_skipped",
        ):
            # §Ship-8c hotfix (critique medium-2): track sculpt-side
            # auto-apply progress so the UI can disable the "apply
            # physics fix" chip once sculpt starts doing the work.
            sug = slot.get("physics_edit_suggestion") or {}
            sug["auto_apply_state"] = {
                "physics_auto_apply_started": "in_progress",
                "physics_auto_applied": "applied",
                "physics_auto_apply_rejected": "rejected",
                "physics_auto_apply_errored": "errored",
                "physics_auto_apply_skipped": "skipped",
            }[etype]
            if etype == "physics_auto_applied":
                sug["auto_apply_summary"] = ev.get("summary")
            if etype in ("physics_auto_apply_rejected", "physics_auto_apply_errored"):
                sug["auto_apply_reason"] = (
                    ev.get("rejected_reason") or ev.get("reason")
                )
            if etype == "physics_auto_apply_skipped":
                sug["auto_apply_reason"] = ev.get("reason")
            if sug:
                slot["physics_edit_suggestion"] = sug
        elif etype == "requires_env_extension":
            # §Ship 48: the diagnoser flagged ≥1 edit it wants but can't
            # ground against the adapter contract. Surface it (informational
            # chip — env extension is a code change, never auto-applied) so a
            # structurally-blocked skill (the g1-kick-v3 stall: every kick
            # term deferred for want of per-foot channels) is never silent.
            terms = [str(x) for x in (ev.get("terms") or []) if str(x)]
            if terms:
                slot["env_extension_suggestion"] = {
                    "terms": terms,
                    "rationales": [str(x) for x in (ev.get("rationales") or [])],
                }
        elif etype == "realism_audited":
            # §7.3: prefer the richer fs-emitted `audit` dict (full payload
            # read from realism_audit.json) over the stdout-emitted fields.
            audit = ev.get("audit")
            if isinstance(audit, dict):
                slot["realism_audit"] = audit
            elif slot["realism_audit"] is None:
                # Stdout-only fields — minimal dict to let UI show verdict.
                slot["realism_audit"] = {
                    "verdict": ev.get("verdict"),
                    "torque_saturation_frac": ev.get("torque_saturation_frac"),
                    "any_joint_saturation_max": ev.get("any_joint_saturation_max"),
                    "joint_vel_p99_max": ev.get("joint_vel_p99_max"),
                    "joint_limit_violation_frac": ev.get("joint_limit_violation_frac"),
                    "top_joints_saturation": ev.get("top_joints_saturation") or [],
                }
        elif etype == "diagnosed":
            slot["diagnosed"] = True
            slot["failure_modes"] = list(ev.get("failure_modes") or [])
        elif etype == "edit_applied":
            slot["reward_version_after"] = ev.get("reward_version_after")
            slot["reward_version_before"] = slot["reward_version_before"] or ev.get("reward_version_before")
            slot["paper_refs"] = list(ev.get("paper_refs") or [])
        elif etype == "iter_completed":
            slot["status"] = "completed"
            slot["completed_at"] = ev.get("ts")
            if slot["primary_metric"] is None and ev.get("primary_metric") is not None:
                slot["primary_metric"] = ev.get("primary_metric")
            if slot["metric_delta"] is None and ev.get("metric_delta") is not None:
                slot["metric_delta"] = ev.get("metric_delta")
            if not slot["failure_modes"] and ev.get("failure_modes"):
                slot["failure_modes"] = list(ev.get("failure_modes") or [])
            slot["edit_count"] = ev.get("edit_count")
            if ev.get("paper_refs"):
                slot["paper_refs"] = list(ev.get("paper_refs") or [])
            if ev.get("reward_version_after") is not None:
                slot["reward_version_after"] = ev.get("reward_version_after")
        elif etype == "iter_fitness":
            # §Ship 34: persist per-iter objective fitness into the REST
            # timeline so the Runs-tab fitness chip survives a reload
            # (the live WS path renders it too, but history rebuilds here).
            if isinstance(ev.get("fitness"), (int, float)):
                slot["fitness"] = float(ev.get("fitness"))
            if isinstance(ev.get("best_so_far"), (int, float)):
                slot["best_fitness"] = float(ev.get("best_so_far"))
            if isinstance(ev.get("progress"), (int, float)):
                slot["progress"] = float(ev.get("progress"))
        elif etype == "best_reward_selected":
            if isinstance(ev.get("fitness"), (int, float)):
                slot["best_fitness"] = float(ev.get("fitness"))
        elif etype == "env_spec_updated":
            # §env generalization: the diagnoser's env-curriculum change
            # (applied + rejected with reasons) for this iter's boundary.
            slot["env_spec_update"] = {
                "new_version": ev.get("new_version"),
                "applied": list(ev.get("applied") or []),
                "rejected": list(ev.get("rejected") or []),
            }

    # §fix: a sequential sculpt loop runs ONE iter at a time, so only the
    # highest-started iter can still be running. A LOWER iter still marked
    # "running" lost its `iter_completed` (a dropped stdout line, or a
    # crash-then-resume) — the loop demonstrably advanced past it, so reconcile
    # it to "completed" rather than stranding a stale RUNNING card. The current
    # (highest-started) iter keeps its event-derived status. Defense-in-depth
    # alongside the watcher pre-seed: `completed` is otherwise STRICTLY
    # stdout-`iter_completed`-driven with no artifact fallback.
    started_idxs = [i for i, s in by_iter.items() if s.get("started_at")]
    if started_idxs:
        max_started = max(started_idxs)
        for i, s in by_iter.items():
            if i < max_started and s["status"] == "running":
                s["status"] = "completed"

    return [by_iter[k] for k in sorted(by_iter.keys())]


def build_iterations_summary(job: Job) -> list[dict[str, Any]]:
    return _iter_events(job)


# ── error classification on non-zero exit ─────────────────────────────
def _classify_run_failure(
    log_path: Path, project_dir: Path,
) -> CudaErrorClass:
    """Scan the run's log for known failure signatures. Tail only — the
    actual error message is almost always in the last few KiB."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")[-32768:]
    except OSError:
        text = ""
    num_envs = _read_adapter_num_envs(project_dir)
    return classify(text, current_num_envs=num_envs)


def _read_adapter_num_envs(project_dir: Path) -> Optional[int]:
    """Pull `[adapter].config.num_envs` from config.toml, if present.
    Used to make OOM suggestions concrete. Returns None on any error."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover — py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with (project_dir / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
    except (OSError, Exception):  # noqa: BLE001
        return None
    adapter_cfg = (cfg.get("adapter") or {}).get("config") or {}
    ne = adapter_cfg.get("num_envs")
    if isinstance(ne, bool):
        return None
    if isinstance(ne, (int, float)):
        return int(ne)
    return None
