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
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

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


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_starting_skill_load_event(
    event: dict[str, Any],
    *,
    expected_checkpoint: Path,
    expected_sha256: str,
    initialization_mode: str,
    require_unadapted: bool = False,
) -> dict[str, Any] | None:
    """Return an exact runtime load receipt or reject contradictory evidence."""
    if event.get("type") != "warm_start_loaded":
        return None
    source_value = event.get("source")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("warm_start_loaded has no source checkpoint")
    source = Path(source_value).expanduser().resolve(strict=True)
    expected = Path(expected_checkpoint).expanduser().resolve(strict=True)
    if source != expected:
        raise ValueError(
            f"worker loaded {source}, not selected starting skill {expected}"
        )
    actual_sha256 = _file_sha256(source)
    event_sha256 = event.get("source_sha256")
    if event_sha256 != expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(
            "warm_start_loaded full digest does not match the immutable "
            "starting-skill pin"
        )
    raw_keys = event.get("load_cfg_keys")
    if not isinstance(raw_keys, list) or not all(
        isinstance(key, str) for key in raw_keys
    ):
        raise ValueError("warm_start_loaded has no structural load keys")
    keys = sorted(raw_keys)
    expected_keys = (
        ["actor", "critic"]
        if initialization_mode == "actor_critic"
        else ["actor"]
    )
    if Counter(keys) != Counter(expected_keys):
        raise ValueError(
            f"worker loaded roles {keys}, expected exactly {expected_keys}"
        )
    raw_loaded = event.get("loaded_checkpoint")
    if not isinstance(raw_loaded, str) or not raw_loaded:
        raise ValueError(
            "warm_start_loaded has no actual loaded checkpoint path"
        )
    loaded = Path(raw_loaded).expanduser().resolve(strict=True)
    loaded_sha256 = _file_sha256(loaded)
    if event.get("loaded_checkpoint_sha256") != loaded_sha256:
        raise ValueError(
            "warm_start_loaded actual digest does not match loaded bytes"
        )
    adapted = event.get("adapted")
    if not isinstance(adapted, bool):
        raise ValueError("warm_start_loaded has no explicit adaptation fact")
    if require_unadapted and adapted:
        raise ValueError(
            "exact recovery snapshots cannot be adapted before loading"
        )
    if not adapted:
        if loaded != expected or loaded_sha256 != expected_sha256:
            raise ValueError(
                "unadapted warm start did not load the selected checkpoint bytes"
            )
    else:
        derived_from = event.get("derived_from")
        if (
            not isinstance(derived_from, dict)
            or derived_from.get("source") != str(source)
            or derived_from.get("source_sha256") != expected_sha256
            or event.get("policy_contract_migration")
            != "zero_initialized_event_phase_observation"
        ):
            raise ValueError(
                "adapted warm start lacks exact source and migration lineage"
            )
    return {
        "source": str(source),
        "source_sha256": actual_sha256,
        "loaded_checkpoint": str(loaded),
        "loaded_checkpoint_sha256": loaded_sha256,
        "adapted": adapted,
        "load_cfg_keys": keys,
        "initialization_mode": initialization_mode,
    }


def resolve_project_robot_slug(project_dir: Path) -> str:
    """Resolve the exact policy/reference robot namespace.

    Kept as a compatibility name for existing run/mission callers; the shared
    resolver distinguishes this namespace from the robot-catalog asset slug.
    """
    from backend.services.project_robot import resolve_project_reference_robot

    return resolve_project_reference_robot(project_dir)


def resolve_starting_skill_target(
    project_dir: Path,
    *,
    require_policy_contract: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the current project interface and its immutable receipt.

    Starting-skill admission happens before a job enters the queue, while the
    subprocess consumes the project's *current* ``config.toml``.  This helper
    is intentionally shared by the API and worker so they cannot disagree
    about adapter, task, robot, or policy-contract identity.

    The first result is the compatibility target payload; the second is the
    compact receipt pinned into ``run_params`` and re-created immediately
    before launch.  Reference-only imports do not need to import mjlab merely
    to resolve their project identity, but their adapter/task/robot tuple is
    still pinned exactly.
    """
    project_dir = Path(project_dir)
    config = tomllib.loads(
        (project_dir / "config.toml").read_text(encoding="utf-8")
    )
    adapter = config.get("adapter") or {}
    adapter_class = adapter.get("class")
    adapter_config = adapter.get("config") or {}
    task_id = adapter_config.get("task_id") or adapter_config.get("env_id")
    if not isinstance(adapter_class, str) or not adapter_class:
        raise ValueError("project adapter class is missing")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("project adapter task_id/env_id is missing")

    robot_slug = resolve_project_robot_slug(project_dir)

    policy_contract = None
    policy_contract_sha256 = None
    if require_policy_contract:
        from sculptor.policy_contract import (
            build_project_policy_contract,
            contract_fingerprint,
        )

        policy_contract = build_project_policy_contract(project_dir)
        policy_contract_sha256 = contract_fingerprint(policy_contract)

    target = {
        "adapter_class": adapter_class,
        "task_id": task_id,
        "robot_slug": robot_slug,
        "compatibility_contract": policy_contract,
    }
    receipt = {
        "schema": 1,
        "adapter_class": adapter_class,
        "task_id": task_id,
        "robot_slug": robot_slug,
        "policy_contract_required": require_policy_contract,
        "policy_contract_sha256": policy_contract_sha256,
    }
    return target, receipt


def resolve_warm_start_checkpoint(
    project_dir: Path, iteration: int,
) -> Path:
    """Resolve an explicit UI warm-start within this project's run history.

    The UI supplies an iteration number rather than a path. Only a non-empty
    promoted checkpoint directly under ``runs/iter_N`` is eligible, and
    symlinks escaping ``runs`` fail closed. ``checkpoint.pt`` is preferred
    when an adapter left both formats behind.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("warm-start iteration must be an integer")
    if iteration < 0:
        raise ValueError("warm-start iteration must be non-negative")

    runs_root = (project_dir / "runs").resolve()
    iter_dir = runs_root / f"iter_{iteration}"
    for name in ("checkpoint.pt", "checkpoint.zip"):
        candidate = iter_dir / name
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        try:
            resolved.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError(
                f"warm-start checkpoint escapes project runs: {candidate}"
            ) from exc
        if resolved.is_file() and resolved.stat().st_size > 0:
            return resolved

    raise FileNotFoundError(
        f"no non-empty checkpoint.pt or checkpoint.zip for iter_{iteration}"
    )


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
    wait is cheap.  A long wait returns ``timeout`` rather than being mistaken
    for an explicit blind acknowledgement; cancel and stop are likewise
    distinct."""
    start_seq = int(read_control_file(control_path).get("gen_decision_seq", 0) or 0)
    waited = 0.0
    step = 1.5
    while waited < timeout_s:
        if cancel.is_set():
            return "cancel"
        ctrl = read_control_file(control_path)
        if ctrl.get("stop"):
            return "stop"
        if int(ctrl.get("gen_decision_seq", 0) or 0) != start_seq:
            return "retry" if ctrl.get("gen_decision") == "retry" else "blind"
        await asyncio.sleep(step)
        waited += step
    return "timeout"


async def _generate_at_launch(
    job: Job, project_dir: Path, behavior_goal: str,
    control_path: Path, cancel: asyncio.Event, *, n_candidates: int = 1,
) -> Optional[str]:
    """§Ship 43/44/45: generate the objective metric as the run's FIRST phase,
    streaming the Ship-40 stage events into THIS run's event stream, then
    auto-calibrate it. Returns "gen:<id>" on acceptance, else None.  The caller
    may proceed blind only when an explicit request-time or interactive
    acknowledgement was recorded.
    Rejections are surfaced as events (never silent) with the exact validation
    reasons + reviewer concerns; §Ship 45 then PAUSES for a one-click retry
    decision (bounded by `_MAX_LAUNCH_GEN_ATTEMPTS`) before falling back to
    blind. Never raises — the caller owns the final objective contract.

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
            if decision == "blind":
                # This acknowledgement happened after the immutable request,
                # while generation was visibly paused. Preserve it separately
                # instead of rewriting the original request field.
                job.params["blind_fitness_runtime_acknowledged"] = True
                job.params["blind_fitness_runtime_acknowledgement_source"] = (
                    "metric_generation_decision"
                )
                job.emit({
                    "type": "blind_fitness_acknowledged",
                    "source": "metric_generation_decision",
                })
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


def _restore_promoted_training_inputs(project_dir: Path) -> dict[str, Any]:
    """Restore mutable training pointers from the promoted atomic tuple.

    ``selection_current.json`` is the authority.  The full selection and every
    referenced artifact hash are verified before either convenience pointer is
    changed.  The artifact-store lock prevents a concurrent promotion from
    changing the selection between verification and restoration.

    This is deliberately generic: refs are resolved by artifact kind/version,
    never by robot or task name.
    """
    from sculptor.edit import _write_current_reexport
    from sculptor.env_spec import load_env_spec, repoint_env_current
    from sculptor.world.artifacts import WorldArtifactStore
    from sculptor.world.project import load_selected_world

    project_dir = Path(project_dir).expanduser().resolve()
    store = WorldArtifactStore(project_dir)
    with store.locked():
        verified_store, selection, _bundle = load_selected_world(
            store.selection_path,
        )
        reward_ref = selection.refs.get("reward")
        env_ref = selection.refs.get("env_spec")
        if reward_ref is None or reward_ref.kind != "reward":
            raise ValueError("promoted selection has no valid reward ref")
        if env_ref is None or env_ref.kind != "env_spec":
            raise ValueError("promoted selection has no valid env_spec ref")

        # resolve_ref rechecks each immutable artifact's SHA-256.  Confinement
        # additionally prevents a hand-crafted selection from repointing the
        # mutable inputs outside their project-local stores.
        reward_path = verified_store.resolve_ref(reward_ref)
        env_path = verified_store.resolve_ref(env_ref)
        rewards_dir = (project_dir / "rewards").resolve()
        env_dir = (project_dir / "env").resolve()
        if (reward_path.parent != rewards_dir
                or reward_path.name != f"{reward_ref.version}.py"):
            raise ValueError("promoted reward ref is outside the reward store")
        if (env_path.parent != env_dir
                or env_path.name != f"{env_ref.version}.json"):
            raise ValueError("promoted env_spec ref is outside the env store")

        # Validate both sources before either mutable pointer is changed.
        load_env_spec(env_path)
        compile(reward_path.read_text(encoding="utf-8"), str(reward_path), "exec")
        _write_current_reexport(rewards_dir, reward_path)
        repoint_env_current(env_dir, env_ref.version)

    return {
        "selection_version": selection.selection_version,
        "tuple_hash": selection.tuple_hash,
        "reward_version": reward_ref.version,
        "reward_sha256": reward_ref.sha256,
        "env_spec_version": env_ref.version,
        "env_spec_sha256": env_ref.sha256,
    }


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
    render_env_index = run_params.get("render_env_index")
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
    resume_exact_tuple = bool(run_params.get("resume_exact_tuple", False))
    warm_start_iteration = run_params.get("warm_start_iteration")
    warm_start_snapshot = run_params.get("warm_start_snapshot")
    expected_recovery_snapshot_receipt = run_params.get(
        "recovery_snapshot_receipt"
    )
    starting_skill_id = run_params.get("starting_skill_id")
    expected_starting_skill_target_receipt = run_params.get(
        "starting_skill_target_receipt"
    )
    expected_starting_skill_manifest_digest = run_params.get(
        "expected_starting_skill_manifest_digest"
    )
    expected_contract_provenance_receipt = run_params.get(
        "compatibility_contract_provenance_receipt"
    )
    acknowledge_legacy_reconstructed_initialization = bool(
        run_params.get("acknowledge_legacy_reconstructed_initialization", False)
    )
    initialization_mode = run_params.get("initialization_mode") or (
        "actor_critic"
        if warm_start_iteration is not None or warm_start_snapshot is not None
        else "actor_only"
    )
    reference_clip_id = run_params.get("reference_clip_id")
    reference_robot = run_params.get("reference_robot")
    reference_feasibility = run_params.get("reference_feasibility")
    expected_active_reference_authority = run_params.get(
        "active_reference_authority"
    )
    authored_world_receipt_contract_present = (
        "authored_world_receipt" in run_params
    )
    expected_authored_world_receipt = run_params.get(
        "authored_world_receipt"
    )
    expected_warm_start_policy_contract_receipt = run_params.get(
        "warm_start_policy_contract_receipt"
    )
    blind_contract_present = "acknowledge_blind_fitness" in run_params
    acknowledge_blind_fitness = bool(
        run_params.get("acknowledge_blind_fitness", False)
    )
    objective_fitness_receipt = dict(
        run_params.get("objective_fitness_receipt") or {
            "requested_metric": fitness_metric,
            "objective_requested": bool(str(fitness_metric or "").strip()),
            "blind_ablation_acknowledged": acknowledge_blind_fitness,
            "dry_run": dry_run,
            "authorization": (
                "dry_run"
                if dry_run
                else (
                    "objective_requested"
                    if str(fitness_metric or "").strip()
                    else "legacy_unreceipted"
                )
            ),
        }
    )

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.params.setdefault(
            "acknowledge_blind_fitness", acknowledge_blind_fitness,
        )
        job.params.setdefault(
            "objective_fitness_receipt", objective_fitness_receipt,
        )
        job.emit({
            "type": "objective_fitness_request_resolved",
            "source": "ui_launch",
            **objective_fitness_receipt,
        })
        if (
            blind_contract_present
            and not dry_run
            and not str(fitness_metric or "").strip()
            and not acknowledge_blind_fitness
        ):
            raise RuntimeError(
                "live training has no objective fitness and no explicit "
                "blind-ablation acknowledgement"
            )
        warm_start_checkpoint: Optional[Path] = None
        warm_start_sha256: Optional[str] = None
        verified_warm_start_policy_contract_receipt: Optional[
            dict[str, Any]
        ] = None
        verified_recovery_snapshot_receipt: Optional[dict[str, Any]] = None
        if warm_start_snapshot is not None:
            if warm_start_iteration is not None or starting_skill_id is not None:
                raise RuntimeError(
                    "interrupted snapshot was combined with another policy "
                    "source after route admission"
                )
            if not isinstance(warm_start_snapshot, dict):
                raise RuntimeError("interrupted snapshot request is malformed")
            if not isinstance(expected_recovery_snapshot_receipt, dict):
                raise RuntimeError(
                    "interrupted snapshot has no immutable admission receipt"
                )
            if not bool(
                warm_start_snapshot.get("acknowledge_interrupted_snapshot")
            ):
                raise RuntimeError(
                    "interrupted snapshot acknowledgement was not preserved"
                )
            try:
                from backend.services.recovery_snapshots import (
                    resolve_recovery_snapshot,
                )

                warm_start_checkpoint, verified_recovery_snapshot_receipt = (
                    await asyncio.to_thread(
                        resolve_recovery_snapshot,
                        project_dir,
                        snapshot_id=str(warm_start_snapshot["snapshot_id"]),
                        checkpoint_sha256=str(
                            warm_start_snapshot["checkpoint_sha256"]
                        ),
                        receipt_digest=str(
                            warm_start_snapshot["receipt_digest"]
                        ),
                    )
                )
                warm_start_sha256 = await asyncio.to_thread(
                    _file_sha256, warm_start_checkpoint,
                )
            except Exception as exc:
                job.emit({
                    "type": "warm_start_snapshot_failed",
                    "source": "worker_launch",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(
                    "could not re-attest the interrupted PPO snapshot; the "
                    "sculpt subprocess was not started"
                ) from exc
            if verified_recovery_snapshot_receipt != (
                expected_recovery_snapshot_receipt
            ):
                raise RuntimeError(
                    "interrupted snapshot receipt changed after route "
                    "admission; the sculpt subprocess was not started"
                )
            provenance_status = verified_recovery_snapshot_receipt.get(
                "provenance_status"
            )
            if (
                provenance_status == "legacy_reconstructed"
                and not bool(
                    warm_start_snapshot.get(
                        "acknowledge_legacy_reconstructed_snapshot"
                    )
                )
            ):
                raise RuntimeError(
                    "legacy snapshot reconstruction acknowledgement was not "
                    "preserved"
                )
            source_payload = verified_recovery_snapshot_receipt["source"]
            checkpoint_payload = verified_recovery_snapshot_receipt[
                "checkpoint"
            ]
            job.params["recovery_snapshot_receipt_revalidated"] = (
                verified_recovery_snapshot_receipt
            )
            job.emit({
                "type": "warm_start_snapshot_resolved",
                "source": "worker_launch",
                "snapshot_id": verified_recovery_snapshot_receipt[
                    "snapshot_id"
                ],
                "iteration": source_payload["iteration"],
                "ppo_step": checkpoint_payload["ppo_step"],
                "last_observed_ppo_iteration": source_payload[
                    "last_observed_ppo_step"
                ],
                "checkpoint": str(warm_start_checkpoint),
                "checkpoint_sha256": warm_start_sha256,
                "receipt_digest": verified_recovery_snapshot_receipt[
                    "receipt_digest"
                ],
                "provenance_status": provenance_status,
                "load_cfg_keys": ["actor", "critic"],
                "optimizer_resume": False,
            })
            if not isinstance(
                expected_warm_start_policy_contract_receipt, dict
            ):
                raise RuntimeError(
                    "interrupted snapshot has no immutable policy-contract "
                    "receipt"
                )
            try:
                from sculptor.policy_contract import (
                    build_recovery_snapshot_warm_start_contract_receipt,
                )

                target_selection_path = Path(
                    expected_warm_start_policy_contract_receipt["target"][
                        "selection_path"
                    ]
                )
                verified_warm_start_policy_contract_receipt = (
                    await asyncio.to_thread(
                        build_recovery_snapshot_warm_start_contract_receipt,
                        project_dir,
                        recovery_receipt=(
                            verified_recovery_snapshot_receipt
                        ),
                        target_selection_path=target_selection_path,
                    )
                )
            except Exception as exc:
                job.emit({
                    "type": "warm_start_snapshot_contract_failed",
                    "source": "worker_launch",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(
                    "could not re-attest the interrupted snapshot policy "
                    "contract; the sculpt subprocess was not started"
                ) from exc
            if verified_warm_start_policy_contract_receipt != (
                expected_warm_start_policy_contract_receipt
            ):
                raise RuntimeError(
                    "interrupted snapshot policy-contract receipt changed "
                    "after route admission"
                )
            job.params[
                "warm_start_policy_contract_receipt_revalidated"
            ] = verified_warm_start_policy_contract_receipt
            job.emit({
                "type": "warm_start_snapshot_contract_verified",
                "source": "worker_launch",
                "snapshot_id": verified_recovery_snapshot_receipt[
                    "snapshot_id"
                ],
                "source_contract_sha256": (
                    verified_warm_start_policy_contract_receipt["source"]
                    ["contract_sha256"]
                ),
                "target_contract_sha256": (
                    verified_warm_start_policy_contract_receipt["target"]
                    ["contract_sha256"]
                ),
                "compatibility": (
                    verified_warm_start_policy_contract_receipt[
                        "compatibility"
                    ]
                ),
            })
        if warm_start_iteration is not None:
            try:
                warm_start_checkpoint = await asyncio.to_thread(
                    resolve_warm_start_checkpoint,
                    project_dir,
                    warm_start_iteration,
                )
                warm_start_sha256 = await asyncio.to_thread(
                    _file_sha256, warm_start_checkpoint,
                )
            except Exception as exc:
                job.emit({
                    "type": "warm_start_checkpoint_failed",
                    "source": "ui_launch",
                    "iteration": warm_start_iteration,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(
                    "could not resolve the explicitly selected warm-start "
                    "checkpoint; the sculpt subprocess was not started"
                ) from exc
            job.emit({
                "type": "warm_start_checkpoint_resolved",
                "source": "ui_launch",
                "iteration": int(warm_start_iteration),
                "checkpoint": str(warm_start_checkpoint),
                "checkpoint_sha256": warm_start_sha256,
            })
            if (
                expected_authored_world_receipt is not None
                and not isinstance(
                    expected_warm_start_policy_contract_receipt, dict
                )
            ):
                raise RuntimeError(
                    "authored-world warm start has no immutable policy-contract "
                    "receipt; the sculpt subprocess was not started"
                )
            if isinstance(
                expected_warm_start_policy_contract_receipt, dict
            ):
                try:
                    from sculptor.policy_contract import (
                        build_iteration_warm_start_contract_receipt,
                    )

                    target_payload = (
                        expected_warm_start_policy_contract_receipt["target"]
                    )
                    target_selection_path = Path(
                        target_payload["selection_path"]
                    )
                    verified_warm_start_policy_contract_receipt = (
                        await asyncio.to_thread(
                            build_iteration_warm_start_contract_receipt,
                            project_dir,
                            int(warm_start_iteration),
                            target_selection_path=target_selection_path,
                        )
                    )
                except Exception as exc:
                    job.emit({
                        "type": "warm_start_policy_contract_failed",
                        "source": "worker_launch",
                        "iteration": int(warm_start_iteration),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise RuntimeError(
                        "could not re-attest the warm-start source/target "
                        "policy contracts; the sculpt subprocess was not "
                        "started"
                    ) from exc
                if (
                    verified_warm_start_policy_contract_receipt
                    != expected_warm_start_policy_contract_receipt
                ):
                    job.emit({
                        "type": "warm_start_policy_contract_failed",
                        "source": "worker_launch",
                        "iteration": int(warm_start_iteration),
                        "reason": "receipt_changed_after_admission",
                    })
                    raise RuntimeError(
                        "warm-start policy-contract receipt changed after route "
                        "admission; the sculpt subprocess was not started"
                    )
                job.params[
                    "warm_start_policy_contract_receipt_revalidated"
                ] = verified_warm_start_policy_contract_receipt
                job.emit({
                    "type": "warm_start_policy_contract_verified",
                    "source": "worker_launch",
                    "iteration": int(warm_start_iteration),
                    "source_contract_sha256": (
                        verified_warm_start_policy_contract_receipt[
                            "source"
                        ]["contract_sha256"]
                    ),
                    "target_contract_sha256": (
                        verified_warm_start_policy_contract_receipt[
                            "target"
                        ]["contract_sha256"]
                    ),
                    "compatibility": (
                        verified_warm_start_policy_contract_receipt[
                            "compatibility"
                        ]
                    ),
                })
        starting_skill_record = None
        if starting_skill_id is not None:
            from sculptor.skill_library import SkillLibrary, SkillLibraryError

            library = SkillLibrary()
            starting_skill_record = library.load(str(starting_skill_id))
            if starting_skill_record is None:
                job.emit({
                    "type": "starting_skill_revalidation_failed",
                    "source": "ui_launch",
                    "starting_skill_id": str(starting_skill_id),
                    "reason": "missing_or_invalid_immutable_metadata",
                })
                raise RuntimeError(
                    f"starting skill {starting_skill_id!r} is missing or failed "
                    "immutable metadata revalidation before launch; the sculpt "
                    "subprocess was not started"
                )
            if (
                not expected_starting_skill_manifest_digest
                or starting_skill_record.manifest_digest
                != expected_starting_skill_manifest_digest
            ):
                job.emit({
                    "type": "starting_skill_manifest_mismatch",
                    "source": "ui_launch",
                    "starting_skill_id": str(starting_skill_id),
                    "expected_manifest_digest": (
                        expected_starting_skill_manifest_digest
                    ),
                    "actual_manifest_digest": (
                        starting_skill_record.manifest_digest
                    ),
                })
                raise RuntimeError(
                    "starting skill manifest changed after selection; the "
                    "sculpt subprocess was not started"
                )
            if initialization_mode not in starting_skill_record.initialization_modes:
                raise RuntimeError(
                    f"initialization mode {initialization_mode!r} is not admitted "
                    f"for skill {starting_skill_id}"
                )
            if initialization_mode != "reference_only":
                from sculptor.compatibility_provenance import (
                    CompatibilityProvenanceError,
                    build_launch_acknowledgement_receipt,
                )

                try:
                    current_contract_provenance_receipt = (
                        build_launch_acknowledgement_receipt(
                            status=(
                                starting_skill_record
                                .compatibility_contract_provenance_status
                            ),
                            provenance_digest=(
                                starting_skill_record
                                .compatibility_contract_provenance_digest
                            ),
                            acknowledged=(
                                acknowledge_legacy_reconstructed_initialization
                            ),
                            initialization_mode=str(initialization_mode),
                        )
                    )
                except CompatibilityProvenanceError as exc:
                    job.emit({
                        "type": "starting_skill_provenance_failed",
                        "source": "worker_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": str(exc),
                    })
                    raise RuntimeError(
                        "starting-skill compatibility provenance could not be "
                        "revalidated; the sculpt subprocess was not started"
                    ) from exc
                if (
                    current_contract_provenance_receipt
                    != expected_contract_provenance_receipt
                ):
                    job.emit({
                        "type": "starting_skill_provenance_failed",
                        "source": "worker_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "reason": "receipt_changed_after_admission",
                    })
                    raise RuntimeError(
                        "starting-skill compatibility provenance receipt changed "
                        "after route admission; the sculpt subprocess was not "
                        "started"
                    )
                job.params[
                    "compatibility_contract_provenance_receipt_revalidated"
                ] = current_contract_provenance_receipt
                job.emit({
                    "type": "starting_skill_provenance_verified",
                    "source": "worker_launch",
                    "starting_skill_id": str(starting_skill_id),
                    **current_contract_provenance_receipt,
                })
            elif (
                expected_contract_provenance_receipt is not None
                or acknowledge_legacy_reconstructed_initialization
            ):
                raise RuntimeError(
                    "reference-only initialization cannot carry a policy "
                    "compatibility-provenance acknowledgement"
                )
            try:
                from sculptor.skill_bundle import ImportTarget, compatibility_for

                current_target_payload, current_target_receipt = (
                    await asyncio.to_thread(
                        resolve_starting_skill_target,
                        project_dir,
                        require_policy_contract=(
                            initialization_mode != "reference_only"
                        ),
                    )
                )
            except Exception as exc:
                job.emit({
                    "type": "starting_skill_target_contract_mismatch",
                    "source": "worker_launch",
                    "starting_skill_id": str(starting_skill_id),
                    "expected_target": expected_starting_skill_target_receipt,
                    "actual_target": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(
                    "could not re-attest the project target contract; the "
                    "sculpt subprocess was not started"
                ) from exc
            if (
                not isinstance(expected_starting_skill_target_receipt, dict)
                or current_target_receipt
                != expected_starting_skill_target_receipt
            ):
                job.emit({
                    "type": "starting_skill_target_contract_mismatch",
                    "source": "worker_launch",
                    "starting_skill_id": str(starting_skill_id),
                    "expected_target": expected_starting_skill_target_receipt,
                    "actual_target": current_target_receipt,
                })
                raise RuntimeError(
                    "project target changed after starting-skill admission; "
                    "the sculpt subprocess was not started"
                )
            current_target = ImportTarget(**current_target_payload)
            current_compatibility = compatibility_for(
                starting_skill_record, current_target,
            )
            if (
                current_compatibility["reasons"]
                or initialization_mode
                not in current_compatibility["allowed_initialization_modes"]
            ):
                job.emit({
                    "type": "starting_skill_target_contract_mismatch",
                    "source": "worker_launch",
                    "starting_skill_id": str(starting_skill_id),
                    "expected_target": expected_starting_skill_target_receipt,
                    "actual_target": current_target_receipt,
                    "compatibility": current_compatibility,
                })
                raise RuntimeError(
                    "starting skill is no longer compatible with the project "
                    "target; the sculpt subprocess was not started"
                )
            job.params.setdefault(
                "starting_skill_target_receipt", current_target_receipt,
            )
            if initialization_mode != "reference_only":
                try:
                    warm_start_checkpoint = await asyncio.to_thread(
                        library.checkpoint_path_for, starting_skill_record,
                    )
                except SkillLibraryError as exc:
                    job.emit({
                        "type": "starting_skill_integrity_failed",
                        "source": "ui_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": str(exc),
                    })
                    raise RuntimeError(
                        "starting skill failed its launch-time integrity check"
                    ) from exc
                warm_start_sha256 = starting_skill_record.checkpoint_sha256
                try:
                    from sculptor.policy_contract import (
                        build_skill_warm_start_contract_receipt,
                    )

                    current_policy_receipt = (
                        build_skill_warm_start_contract_receipt(
                            skill_id=starting_skill_record.skill_id,
                            manifest_digest=str(
                                starting_skill_record.manifest_digest
                            ),
                            checkpoint_sha256=(
                                starting_skill_record.checkpoint_sha256
                            ),
                            tensor_signature_sha256=(
                                starting_skill_record.tensor_signature_sha256
                            ),
                            source_contract=(
                                starting_skill_record.compatibility_contract
                                or {}
                            ),
                            target_contract=(
                                current_target_payload[
                                    "compatibility_contract"
                                ] or {}
                            ),
                            target_receipt=current_target_receipt,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "could not re-attest the starting-skill policy "
                        "contract receipt; the sculpt subprocess was not "
                        "started"
                    ) from exc
                receipt_supplied = isinstance(
                    expected_warm_start_policy_contract_receipt, dict
                )
                if (
                    not receipt_supplied
                    or current_policy_receipt
                    != expected_warm_start_policy_contract_receipt
                ):
                    job.emit({
                        "type": "warm_start_policy_contract_failed",
                        "source": "worker_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "reason": "receipt_changed_after_admission",
                    })
                    raise RuntimeError(
                        "starting-skill policy-contract receipt changed after "
                        "route admission; the sculpt subprocess was not "
                        "started"
                    )
                if (
                    current_policy_receipt is not None
                    and verified_warm_start_policy_contract_receipt is not None
                ):
                    raise RuntimeError(
                        "multiple policy warm-start receipts were admitted"
                    )
                if current_policy_receipt is not None:
                    verified_warm_start_policy_contract_receipt = (
                        current_policy_receipt
                    )
                    job.params[
                        "warm_start_policy_contract_receipt_revalidated"
                    ] = current_policy_receipt
                    job.emit({
                        "type": "warm_start_policy_contract_verified",
                        "source": "worker_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "source_contract_sha256": current_policy_receipt[
                            "source"
                        ]["contract_sha256"],
                        "target_contract_sha256": current_policy_receipt[
                            "target"
                        ]["contract_sha256"],
                        "compatibility": current_policy_receipt[
                            "compatibility"
                        ],
                    })
            else:
                from sculptor.refs import library as reference_library

                if not (
                    starting_skill_record.reference_clip_id
                    and starting_skill_record.reference_robot
                    and starting_skill_record.reference_sha256
                ):
                    raise RuntimeError(
                        "reference-only starting skill has no attested reference"
                    )
                clip_path = reference_library.clip_dir(
                    starting_skill_record.reference_robot,
                    starting_skill_record.reference_clip_id,
                ) / reference_library.CLIP_FILENAME
                provenance_path = reference_library.clip_dir(
                    starting_skill_record.reference_robot,
                    starting_skill_record.reference_clip_id,
                ) / reference_library.PROVENANCE_FILENAME
                try:
                    actual_reference_sha = await asyncio.to_thread(
                        _file_sha256, clip_path,
                    )
                except OSError as exc:
                    job.emit({
                        "type": "starting_skill_integrity_failed",
                        "source": "ui_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": f"reference unavailable: {exc}",
                    })
                    raise RuntimeError(
                        "starting-skill reference disappeared before launch"
                    ) from exc
                if actual_reference_sha != starting_skill_record.reference_sha256:
                    job.emit({
                        "type": "starting_skill_integrity_failed",
                        "source": "ui_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": "reference digest mismatch",
                        "expected_reference_sha256": (
                            starting_skill_record.reference_sha256
                        ),
                        "actual_reference_sha256": actual_reference_sha,
                    })
                    raise RuntimeError(
                        "starting-skill reference failed its launch-time "
                        "integrity check"
                    )
                expected_provenance_sha = (
                    starting_skill_record.reference_provenance_sha256
                )
                if not expected_provenance_sha:
                    raise RuntimeError(
                        "imported reference has no canonical provenance "
                        "identity; re-import the starting skill"
                    )
                try:
                    from sculptor.skill_bundle import (
                        reference_source_provenance_sha256,
                    )

                    actual_provenance_sha = await asyncio.to_thread(
                        reference_source_provenance_sha256,
                        json.loads(
                            provenance_path.read_text(encoding="utf-8")
                        ),
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    job.emit({
                        "type": "starting_skill_integrity_failed",
                        "source": "ui_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": f"reference provenance unavailable: {exc}",
                    })
                    raise RuntimeError(
                        "starting-skill reference provenance disappeared or "
                        "became invalid before launch"
                    ) from exc
                if actual_provenance_sha != expected_provenance_sha:
                    job.emit({
                        "type": "starting_skill_integrity_failed",
                        "source": "ui_launch",
                        "starting_skill_id": str(starting_skill_id),
                        "error": "reference provenance identity mismatch",
                        "expected_reference_provenance_sha256": (
                            expected_provenance_sha
                        ),
                        "actual_reference_provenance_sha256": (
                            actual_provenance_sha
                        ),
                    })
                    raise RuntimeError(
                        "starting-skill reference provenance failed its "
                        "launch-time integrity check"
                    )
            job.emit({
                "type": "starting_skill_resolved",
                "source": "ui_launch",
                "starting_skill_id": str(starting_skill_id),
                "initialization_mode": str(initialization_mode),
                "manifest_digest": starting_skill_record.manifest_digest,
                "checkpoint_sha256": warm_start_sha256,
                "reference_clip_id": starting_skill_record.reference_clip_id,
                "reference_robot": starting_skill_record.reference_robot,
                "reference_sha256": starting_skill_record.reference_sha256,
                "tensor_signature_sha256": (
                    starting_skill_record.tensor_signature_sha256
                ),
                "trust_status": starting_skill_record.trust_status,
            })

        # Re-resolve the exact reward selected by current.py after queueing.
        # A promoted reward may embed a reference even when the browser motion
        # picker is empty, so this authority must agree byte-for-byte with the
        # route receipt before any Tier-D or subprocess work begins.
        from sculptor.reference_authority import (
            ActiveReferenceAuthorityError,
            resolve_active_reference_authority,
        )

        try:
            current_active_reference = await asyncio.to_thread(
                resolve_active_reference_authority,
                project_dir / "rewards",
            )
        except ActiveReferenceAuthorityError as exc:
            current_active_reference = None
            active_reference_error = str(exc)
        else:
            active_reference_error = None
        current_active_reference_receipt = (
            current_active_reference.to_dict()
            if current_active_reference is not None
            else None
        )
        if (
            active_reference_error is not None
            or current_active_reference_receipt
            != expected_active_reference_authority
        ):
            job.emit({
                "type": "active_reference_authority_failed",
                "source": "worker_launch",
                "expected": expected_active_reference_authority,
                "actual": current_active_reference_receipt,
                "error": active_reference_error,
            })
            raise RuntimeError(
                "active reference reward changed or became invalid after "
                "route admission; the sculpt subprocess was not started"
            )
        if current_active_reference is not None:
            if (
                str(reference_clip_id)
                != current_active_reference.reference_clip_id
                or str(reference_robot)
                != current_active_reference.reference_robot
            ):
                raise RuntimeError(
                    "queued reference identity disagrees with the active "
                    "reward; the sculpt subprocess was not started"
                )
            job.emit({
                "type": "active_reference_reward_attested",
                "source": "worker_launch",
                **current_active_reference_receipt,
            })

        # Re-attest the exact feasibility artifacts after queueing and before
        # subprocess creation.  The route stores a digest pin from
        # verify_tierd_certificate; this closes the TOCTOU window if the clip,
        # rollout, or provenance changes while a job waits in the queue.
        if (
            reference_clip_id is not None
            and reference_robot is not None
            and isinstance(reference_feasibility, dict)
        ):
            from sculptor.refs import library as reference_library
            expected_status = reference_feasibility.get("status")
            from sculptor.refs.track import (
                TierDAdmissionError,
                require_tierd_admission,
                require_tierd_target_compatibility,
                verify_tierd_certificate,
            )

            if expected_status == "tierd_verified":
                try:
                    expected_target_robot = reference_feasibility.get(
                        "target_robot"
                    )
                    current_target_robot = resolve_project_robot_slug(
                        project_dir
                    )
                    if (
                        expected_target_robot != current_target_robot
                        or str(reference_robot) != current_target_robot
                    ):
                        raise TierDAdmissionError(
                            "reference/project robot identity changed after "
                            "route admission"
                        )
                    certificate = await asyncio.to_thread(
                        require_tierd_admission,
                        str(reference_robot),
                        str(reference_clip_id),
                        expected_clip_sha256=reference_feasibility.get(
                            "clip_sha256"
                        ),
                        expected_certificate_sha256=reference_feasibility.get(
                            "certificate_sha256"
                        ),
                        expected_rollout_sha256=reference_feasibility.get(
                            "rollout_sha256"
                        ),
                        expected_execution_contract_sha256=(
                            reference_feasibility.get(
                                "execution_contract_sha256"
                            )
                        ),
                        expected_execution_boundary_sha256=(
                            reference_feasibility.get(
                                "execution_boundary_sha256"
                            )
                        ),
                    )
                    certificate = await asyncio.to_thread(
                        require_tierd_target_compatibility,
                        certificate,
                        project_dir,
                        target_robot=current_target_robot,
                    )
                    certificate_reason = None
                except (TierDAdmissionError, ValueError) as exc:
                    job.emit({
                        "type": "reference_feasibility_integrity_failed",
                        "source": "ui_launch",
                        "reference_robot": str(reference_robot),
                        "reference_clip_id": str(reference_clip_id),
                        "error": str(exc),
                    })
                    raise RuntimeError(
                        "reference feasibility artifacts changed after "
                        "admission; the sculpt subprocess was not started"
                    ) from exc
            else:
                certificate, certificate_reason = await asyncio.to_thread(
                    verify_tierd_certificate,
                    str(reference_robot),
                    str(reference_clip_id),
                )
            if certificate is not None:
                if (
                    expected_status != "tierd_verified"
                    or certificate.clip_content_sha256
                    != reference_feasibility.get("clip_sha256")
                    or certificate.rollout_sha256
                    != reference_feasibility.get("rollout_sha256")
                    or certificate.certificate_sha256
                    != reference_feasibility.get("certificate_sha256")
                    or certificate.execution_contract_sha256
                    != reference_feasibility.get("execution_contract_sha256")
                    or certificate.execution_boundary_sha256
                    != reference_feasibility.get("execution_boundary_sha256")
                ):
                    job.emit({
                        "type": "reference_feasibility_integrity_failed",
                        "source": "ui_launch",
                        "reference_robot": str(reference_robot),
                        "reference_clip_id": str(reference_clip_id),
                        "error": "Tier-D certificate differs from route pin",
                    })
                    raise RuntimeError(
                        "reference feasibility artifacts changed after "
                        "admission; the sculpt subprocess was not started"
                    )
                job.emit({
                    "type": "reference_feasibility_admitted",
                    "source": "ui_launch",
                    "status": "tierd_verified",
                    "tier": "D",
                    "kinematic_only": False,
                    "training_authorized": True,
                    "reference_robot": str(reference_robot),
                    "target_robot": current_target_robot,
                    "reference_clip_id": str(reference_clip_id),
                    "clip_sha256": certificate.clip_content_sha256,
                    "rollout_sha256": certificate.rollout_sha256,
                    "certificate_sha256": certificate.certificate_sha256,
                    "execution_contract_sha256": (
                        certificate.execution_contract_sha256
                    ),
                    "execution_boundary_sha256": (
                        certificate.execution_boundary_sha256
                    ),
                })
            else:
                if (
                    not dry_run
                    or expected_status != "kinematic_reference_inspection_only"
                ):
                    job.emit({
                        "type": "reference_feasibility_integrity_failed",
                        "source": "ui_launch",
                        "reference_robot": str(reference_robot),
                        "reference_clip_id": str(reference_clip_id),
                        "error": certificate_reason or "Tier-D certificate missing",
                    })
                    raise RuntimeError(
                        "live reference-backed training has no verified Tier-D "
                        "certificate; the sculpt subprocess was not started"
                    )
                clip_path = reference_library.clip_dir(
                    str(reference_robot), str(reference_clip_id),
                ) / reference_library.CLIP_FILENAME
                try:
                    current_clip_sha = await asyncio.to_thread(
                        _file_sha256, clip_path,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        "kinematic reference disappeared before dry-run launch"
                    ) from exc
                if current_clip_sha != reference_feasibility.get("clip_sha256"):
                    job.emit({
                        "type": "reference_feasibility_integrity_failed",
                        "source": "ui_launch",
                        "reference_robot": str(reference_robot),
                        "reference_clip_id": str(reference_clip_id),
                        "error": "Tier-K clip digest differs from route pin",
                    })
                    raise RuntimeError(
                        "kinematic reference changed after admission; the "
                        "sculpt subprocess was not started"
                    )
                job.emit({
                    "type": "reference_feasibility_admitted",
                    "source": "ui_launch",
                    "status": "kinematic_reference_inspection_only",
                    "tier": "K",
                    "kinematic_only": True,
                    "training_authorized": False,
                    "scope": "contract_and_reference_resolution",
                    "inspection_only": True,
                    "training_invoked": False,
                    "checkpoint_published": False,
                    "reference_robot": str(reference_robot),
                    "reference_clip_id": str(reference_clip_id),
                    "clip_sha256": current_clip_sha,
                    "rollout_sha256": None,
                    "reason": certificate_reason,
                })
                # Tier K proves only a kinematic/reference contract.  Returning
                # here is the safety boundary: no sculpt subprocess exists, so
                # adapter.train, rollout, reward edits, lineage production, and
                # checkpoint publication are all impossible on this path.
                inspection_receipt = {
                    "inspection_only": True,
                    "reference_robot": str(reference_robot),
                    "reference_clip_id": str(reference_clip_id),
                    "clip_sha256": current_clip_sha,
                    "training_invoked": False,
                    "rollout_invoked": False,
                    "checkpoint_published": False,
                    "scope": "contract_and_reference_resolution",
                }
                job.params["reference_inspection_receipt"] = inspection_receipt
                job.emit({
                    "type": "reference_inspection_completed",
                    "source": "ui_launch",
                    **inspection_receipt,
                })
                return inspection_receipt

        if resume_exact_tuple:
            try:
                restored = await asyncio.to_thread(
                    _restore_promoted_training_inputs, project_dir,
                )
            except Exception as exc:
                job.emit({
                    "type": "promoted_tuple_restore_failed",
                    "source": "ui_resume",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(
                    "could not restore the promoted training tuple; "
                    "the sculpt subprocess was not started"
                ) from exc
            job.emit({
                "type": "promoted_tuple_restored",
                "source": "ui_resume",
                **restored,
            })

        # Prepare env — strip empty ANTHROPIC_API_KEY so the .env file
        # in the project (or sculptor's own) can win; matches the
        # sculptor/__init__.py behavior.
        env = {k: v for k, v in os.environ.items() if v or k != "ANTHROPIC_API_KEY"}
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if warm_start_sha256:
            env["SCULPTOR_WARM_START_CHECKPOINT_SHA256"] = str(
                warm_start_sha256
            )
        if verified_warm_start_policy_contract_receipt is not None:
            verified_contract = verified_warm_start_policy_contract_receipt
            env[
                "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON"
            ] = json.dumps(
                verified_contract,
                sort_keys=True,
                separators=(",", ":"),
            )
            env["SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON"] = json.dumps(
                verified_contract["target"]["contract"],
                sort_keys=True,
                separators=(",", ":"),
            )
            env["SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256"] = str(
                verified_contract["target"]["contract_sha256"]
            )
            env["SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256"] = str(
                verified_contract["source"]["contract_sha256"]
            )
            env["SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON"] = json.dumps(
                verified_contract["compatibility"],
                sort_keys=True,
                separators=(",", ":"),
            )
        if starting_skill_record is not None:
            env["SCULPTOR_STARTING_SKILL_ID"] = str(starting_skill_record.skill_id)
            env["SCULPTOR_STARTING_SKILL_INIT_MODE"] = str(initialization_mode)
            if starting_skill_record.manifest_digest:
                env["SCULPTOR_STARTING_SKILL_MANIFEST_DIGEST"] = str(
                    starting_skill_record.manifest_digest
                )
            if starting_skill_record.compatibility_contract_digest:
                env["SCULPTOR_STARTING_SKILL_CONTRACT_DIGEST"] = str(
                    starting_skill_record.compatibility_contract_digest
                )
            if starting_skill_record.compatibility_contract_provenance_status:
                env["SCULPTOR_STARTING_SKILL_CONTRACT_PROVENANCE_STATUS"] = str(
                    starting_skill_record.compatibility_contract_provenance_status
                )
            if starting_skill_record.compatibility_contract_provenance_digest:
                env["SCULPTOR_STARTING_SKILL_CONTRACT_PROVENANCE_DIGEST"] = str(
                    starting_skill_record.compatibility_contract_provenance_digest
                )
            env["SCULPTOR_STARTING_SKILL_LEGACY_ACKNOWLEDGED"] = (
                "1" if acknowledge_legacy_reconstructed_initialization else "0"
            )
            if starting_skill_record.tensor_signature_sha256:
                env["SCULPTOR_STARTING_SKILL_TENSOR_SIGNATURE"] = str(
                    starting_skill_record.tensor_signature_sha256
                )
            if warm_start_sha256:
                env["SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256"] = str(
                    warm_start_sha256
                )
            if starting_skill_record.reference_clip_id:
                env["SCULPTOR_STARTING_SKILL_REFERENCE_CLIP"] = str(
                    starting_skill_record.reference_clip_id
                )
            if starting_skill_record.reference_sha256:
                env["SCULPTOR_STARTING_SKILL_REFERENCE_SHA256"] = str(
                    starting_skill_record.reference_sha256
                )
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
        # Opt-in via the dropdown sentinel; SCULPTOR_LAUNCH_GEN=0 disables it.
        # Without a separately recorded blind-ablation acknowledgement that
        # now rejects before subprocess creation instead of silently training
        # without the requested objective.
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

        runtime_blind_acknowledged = bool(
            job.params.get("blind_fitness_runtime_acknowledged", False)
        )
        if (
            blind_contract_present
            and not dry_run
            and not str(eff_fitness_metric or "").strip()
            and not acknowledge_blind_fitness
            and not runtime_blind_acknowledged
        ):
            receipt = dict(job.params.get("objective_fitness_receipt") or {})
            receipt.update({
                "effective_metric": None,
                "authorization": "rejected_no_effective_objective",
            })
            job.params["objective_fitness_receipt"] = receipt
            job.emit({
                "type": "objective_fitness_contract_rejected",
                "source": "ui_launch",
                **receipt,
            })
            raise RuntimeError(
                "the requested objective metric was unavailable and no "
                "explicit blind-ablation acknowledgement was recorded; the "
                "sculpt subprocess was not started"
            )

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
        if warm_start_checkpoint is not None:
            cmd += ["--init-policy", str(warm_start_checkpoint)]
            cmd += ["--init-policy-mode", str(initialization_mode)]
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
        if render_env_index is not None:
            cmd += ["--render-env-index", str(int(render_env_index))]
        if rollout_episodes is not None:
            cmd += ["--rollout-episodes", str(int(rollout_episodes))]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        if reference_clip_id is not None and reference_robot is not None:
            cmd += [
                "--reference-clip", str(reference_clip_id),
                "--reference-robot", str(reference_robot),
            ]
        if isinstance(expected_active_reference_authority, dict):
            expected_reward_sha = expected_active_reference_authority.get(
                "reward_sha256"
            )
            if isinstance(expected_reward_sha, str):
                cmd += [
                    "--expected-active-reference-reward-sha256",
                    expected_reward_sha,
                ]
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
        resolved_fitness_metric: Optional[str] = None
        if eff_fitness_metric:
            # §Ship 35: a generated metric is referenced as "gen:<id>";
            # resolve it to the project's metric.py path the CLI can load.
            # A built-in name passes through unchanged. An unresolvable
            # gen ref is dropped (blind loop) rather than failing the run.
            # §Ship 43: eff_fitness_metric is the LAUNCH-GENERATED gen:<id>
            # when the sentinel was used (else the original selection).
            resolved = _resolve_fitness_metric(project_dir, str(eff_fitness_metric))
            if resolved is not None:
                resolved_fitness_metric = resolved
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

        if (
            blind_contract_present
            and not dry_run
            and resolved_fitness_metric is None
            and not acknowledge_blind_fitness
            and not runtime_blind_acknowledged
        ):
            receipt = dict(job.params.get("objective_fitness_receipt") or {})
            receipt.update({
                "effective_metric": eff_fitness_metric,
                "resolved_metric": None,
                "authorization": "rejected_no_effective_objective",
            })
            job.params["objective_fitness_receipt"] = receipt
            job.emit({
                "type": "objective_fitness_contract_rejected",
                "source": "ui_launch",
                **receipt,
            })
            raise RuntimeError(
                "the requested objective metric could not be resolved and no "
                "explicit blind-ablation acknowledgement was recorded; the "
                "sculpt subprocess was not started"
            )

        objective_receipt = dict(
            job.params.get("objective_fitness_receipt") or {}
        )
        objective_receipt.update({
            "effective_metric": eff_fitness_metric,
            "resolved_metric": resolved_fitness_metric,
            "effective_mode": final_fitness_mode,
            "blind_at_subprocess_start": resolved_fitness_metric is None,
            "authorization": (
                "dry_run"
                if dry_run
                else (
                    "objective_resolved"
                    if resolved_fitness_metric is not None
                    else (
                        "blind_ablation_acknowledged"
                        if acknowledge_blind_fitness
                        else "runtime_blind_ablation_acknowledged"
                    )
                )
            ),
        })
        job.params["objective_fitness_receipt"] = objective_receipt
        job.emit({
            "type": "objective_fitness_effective",
            "source": "ui_launch",
            **objective_receipt,
        })

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
        if warm_start_checkpoint is not None:
            if warm_start_iteration is not None:
                job.params.setdefault(
                    "warm_start_iteration", int(warm_start_iteration),
                )
            if verified_recovery_snapshot_receipt is not None:
                job.params.setdefault(
                    "warm_start_snapshot_id",
                    verified_recovery_snapshot_receipt["snapshot_id"],
                )
                job.params.setdefault(
                    "warm_start_snapshot_receipt_digest",
                    verified_recovery_snapshot_receipt["receipt_digest"],
                )
            job.params.setdefault(
                "warm_start_checkpoint", str(warm_start_checkpoint),
            )
            job.params.setdefault(
                "warm_start_checkpoint_sha256", warm_start_sha256,
            )
        if starting_skill_record is not None:
            job.params.setdefault("starting_skill_id", starting_skill_record.skill_id)
            job.params.setdefault("initialization_mode", str(initialization_mode))
            job.params.setdefault(
                "starting_skill_manifest_digest",
                starting_skill_record.manifest_digest,
            )
            job.params.setdefault(
                "starting_skill_checkpoint_sha256", warm_start_sha256,
            )
            job.params.setdefault(
                "starting_skill_compatibility_contract_digest",
                starting_skill_record.compatibility_contract_digest,
            )
            job.params.setdefault(
                "starting_skill_compatibility_contract_provenance_status",
                starting_skill_record.compatibility_contract_provenance_status,
            )
            job.params.setdefault(
                "starting_skill_compatibility_contract_provenance_digest",
                starting_skill_record.compatibility_contract_provenance_digest,
            )
            job.params.setdefault(
                "starting_skill_tensor_signature_sha256",
                starting_skill_record.tensor_signature_sha256,
            )
            job.params.setdefault(
                "starting_skill_reference_sha256",
                starting_skill_record.reference_sha256,
            )
            job.params.setdefault(
                "starting_skill_trust_status",
                starting_skill_record.trust_status,
            )
        # §Ship-7: stash the new params so the Runs-tab summary can
        # surface them (non-None entries only to keep the payload lean).
        for key, val in (
            ("max_episode_steps", max_episode_steps),
            ("playback_speed", playback_speed),
            ("render_every", render_every),
            ("rollout_fps", rollout_fps),
            ("render_env_index", render_env_index),
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
            ("reference_clip_id", reference_clip_id),
            ("reference_robot", reference_robot),
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

        # Snapshot checkpoint state immediately before spawn.  This session
        # records requested/effective inputs only after process creation, an
        # initialization edge only on the worker's successful-load event, and
        # production edges only for checkpoint bytes new to this invocation.
        from backend.services.artifact_lineage import RunLineageSession

        reference_sha256 = None
        if isinstance(reference_feasibility, dict):
            raw_reference_sha = reference_feasibility.get("clip_sha256")
            if isinstance(raw_reference_sha, str):
                reference_sha256 = raw_reference_sha
        if reference_sha256 is None and starting_skill_record is not None:
            reference_sha256 = starting_skill_record.reference_sha256
        requested_lineage_mode = (
            str(initialization_mode)
            if (
                starting_skill_record is not None
                or warm_start_iteration is not None
                or verified_recovery_snapshot_receipt is not None
            )
            else "auto_resume"
        )
        lineage = RunLineageSession(
            project_dir=project_dir,
            project_slug=str(job.project_slug or project_dir.name),
            run_id=job.job_id,
            requested_initialization_mode=requested_lineage_mode,
            no_kg=no_kg,
            reference_robot=(
                str(reference_robot) if reference_robot is not None else None
            ),
            reference_clip_id=(
                str(reference_clip_id) if reference_clip_id is not None else None
            ),
            reference_sha256=reference_sha256,
            starting_skill_record=starting_skill_record,
        )

        # The route admits and pins the promoted world, but jobs can wait in a
        # queue. Re-run the same integrity + robot check at the last
        # responsible moment and compare the exact tuple receipt before any
        # subprocess (and therefore any GPU work) exists.
        from backend.services import world_store

        try:
            current_world = await asyncio.to_thread(
                world_store.training_preflight, project_dir,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed before spawn
            job.emit({
                "type": "authored_world_revalidation_failed",
                "source": "worker_launch",
                "reason": "integrity_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise RuntimeError(
                "could not re-attest the authored world; the sculpt "
                "subprocess was not started"
            ) from exc
        if current_world is not None and not bool(current_world.get("ok")):
            errors = current_world.get("errors") or ["unknown integrity error"]
            job.emit({
                "type": "authored_world_revalidation_failed",
                "source": "worker_launch",
                "reason": "integrity_error",
                "errors": [str(error) for error in errors],
            })
            raise RuntimeError(
                "authored world integrity changed after route admission; the "
                "sculpt subprocess was not started"
            )
        if (
            current_world is not None
            and current_world.get("robot_matches_project") is False
        ):
            job.emit({
                "type": "authored_world_revalidation_failed",
                "source": "worker_launch",
                "reason": "robot_mismatch",
                "world_robot": current_world.get("world_robot"),
                "project_robot": current_world.get("project_robot"),
            })
            raise RuntimeError(
                "authored world targets another robot; re-author it for the "
                "project robot before launching"
            )
        actual_authored_world_receipt = (
            None
            if current_world is None
            else {
                "selection_version": current_world.get("selection_version"),
                "tuple_hash": current_world.get("tuple_hash"),
                "world_robot": current_world.get("world_robot"),
                "project_robot": current_world.get("project_robot"),
            }
        )
        if (
            authored_world_receipt_contract_present
            and actual_authored_world_receipt != expected_authored_world_receipt
        ):
            job.emit({
                "type": "authored_world_revalidation_failed",
                "source": "worker_launch",
                "reason": "selection_changed",
                "expected_receipt": expected_authored_world_receipt,
                "actual_receipt": actual_authored_world_receipt,
            })
            raise RuntimeError(
                "authored world changed after route admission; the sculpt "
                "subprocess was not started"
            )
        if actual_authored_world_receipt is not None:
            job.params["authored_world_receipt_revalidated"] = (
                actual_authored_world_receipt
            )
            job.emit({
                "type": "authored_world_revalidated",
                "source": "worker_launch",
                **actual_authored_world_receipt,
            })

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
        try:
            lineage.record_started()
        except Exception as exc:  # noqa: BLE001 - preserve worker, expose gap
            job.emit({
                "type": "lineage_record_failed",
                "phase": "run_started",
                "error": f"{type(exc).__name__}: {exc}",
            })

        starting_skill_load_required = bool(
            (
                (
                    starting_skill_record is not None
                    and starting_skill_record.policy_roles
                )
                or verified_recovery_snapshot_receipt is not None
            )
            and initialization_mode != "reference_only"
        )
        verified_starting_skill_load: dict[str, Any] | None = None
        starting_skill_load_error: str | None = None

        def _observe_lineage_event(event: dict[str, Any]) -> None:
            nonlocal verified_starting_skill_load, starting_skill_load_error
            if starting_skill_load_required and event.get("type") == "warm_start_loaded":
                try:
                    if warm_start_checkpoint is None or warm_start_sha256 is None:
                        raise ValueError(
                            "selected starting skill has no resolved checkpoint pin"
                        )
                    receipt = _verify_starting_skill_load_event(
                        event,
                        expected_checkpoint=warm_start_checkpoint,
                        expected_sha256=warm_start_sha256,
                        initialization_mode=str(initialization_mode),
                        require_unadapted=(
                            verified_recovery_snapshot_receipt is not None
                        ),
                    )
                    if receipt is None:
                        raise ValueError("worker load event was not recognized")
                    if (
                        verified_starting_skill_load is not None
                        and verified_starting_skill_load != receipt
                    ):
                        raise ValueError(
                            "worker emitted conflicting starting-skill load receipts"
                        )
                    verified_starting_skill_load = receipt
                    job.params["starting_skill_load_receipt"] = receipt
                except Exception as exc:  # noqa: BLE001 - fail after drain
                    starting_skill_load_error = f"{type(exc).__name__}: {exc}"
                    job.emit({
                        "type": "starting_skill_load_rejected",
                        "source": "worker_stdout",
                        "error": starting_skill_load_error,
                    })
                    return
            try:
                lineage.observe_event(event)
            except Exception as exc:  # noqa: BLE001 - event remains visible
                if (
                    starting_skill_load_required
                    and event.get("type") == "warm_start_loaded"
                ):
                    starting_skill_load_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                job.emit({
                    "type": "lineage_observation_rejected",
                    "phase": str(event.get("type") or "unknown"),
                    "error": f"{type(exc).__name__}: {exc}",
                })

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
            _stream_stdout(
                job, proc, log_path, on_event=_observe_lineage_event,
            )
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

        starting_skill_load_failure = None
        if starting_skill_load_required:
            if starting_skill_load_error is not None:
                starting_skill_load_failure = starting_skill_load_error
            elif verified_starting_skill_load is None:
                starting_skill_load_failure = (
                    "worker exited without an exact warm_start_loaded receipt "
                    "for the selected starting skill"
                )

        # Final summary from disk (canonical: sculpt wrote these).
        metric_history = _read_metric_history(project_dir)
        iterations_run = len(metric_history)
        if starting_skill_load_failure is None:
            try:
                output_policies = lineage.record_outputs()
            except Exception as exc:  # noqa: BLE001 - outcome remains preserved
                job.emit({
                    "type": "lineage_record_failed",
                    "phase": "run_outputs",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            else:
                if output_policies:
                    job.emit({
                        "type": "lineage_outputs_recorded",
                        "policy_sha256": [
                            policy.sha256 for policy in output_policies
                        ],
                    })
        else:
            job.emit({
                "type": "lineage_outputs_quarantined",
                "reason": "starting_skill_load_unproven",
                "detail": starting_skill_load_failure,
            })
        if rc == 0 and starting_skill_load_failure is None:
            job.emit({
                "type": "run_completed",
                "return_code": rc,
                "iterations_run": iterations_run,
                "primary_metric_history": metric_history,
            })
        elif starting_skill_load_failure is not None:
            friendly = "selected starting skill was not proven loaded"
            job.status = "errored"
            job.error = friendly
            job.params["error_classification"] = {
                "kind": "starting_skill_load_unproven",
                "title": friendly,
                "detail": starting_skill_load_failure,
                "suggestions": [
                    "Inspect the worker log and adapter warm-start support.",
                    "Relaunch only after the exact policy/load-role receipt appears.",
                ],
                "problem_type": "/problems/starting-skill-load-unproven",
                "action": None,
            }
            job.emit({
                "type": "run_errored",
                "return_code": rc,
                "iterations_run": iterations_run,
                "error": friendly,
                "error_kind": "starting_skill_load_unproven",
                "error_detail": starting_skill_load_failure,
                "error_suggestions": job.params["error_classification"][
                    "suggestions"
                ],
                "error_problem_type": (
                    "/problems/starting-skill-load-unproven"
                ),
                "error_action": None,
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
                "evidence": classification.evidence,
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
                "error_evidence": classification.evidence,
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
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
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
                    # Under its own key: `source` is a field the emitters
                    # already use for real data — a warm start's checkpoint
                    # path, a selection's origin, a clip's dataset — and
                    # overwriting it silently replaced all of those with
                    # the string "stdout" by the time they reached the UI.
                    ev["origin"] = "stdout"
                    ev.setdefault("source", "stdout")
                    job.emit(ev)
                    if on_event is not None:
                        on_event(ev)
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
        from watchfiles import awatch
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
    started_iter_indices: set[int] = set()
    active_iter: int | None = None
    for ev in job.events:
        etype = ev.get("type")
        iter_idx = ev.get("iter")
        if etype == "iter_started" and isinstance(iter_idx, int):
            active_iter = iter_idx
            started_iter_indices.add(iter_idx)
        elif not isinstance(iter_idx, int) and etype == "iter_progress":
            # The inner runner has no outer index; ordered progress belongs to
            # the most recently started outer iteration.
            iter_idx = active_iter
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
                "failure_stage": None,
                "ppo_iterations_completed": None,
                "ppo_iterations_requested": None,
                "checkpoint_preserved": False,
                "checkpoint_sha256": None,
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
        elif etype == "iter_progress":
            rl_iter = ev.get("rl_iter")
            rl_total = ev.get("rl_total")
            if type(rl_iter) is int and type(rl_total) is int:
                slot["ppo_iterations_completed"] = rl_iter
                slot["ppo_iterations_requested"] = rl_total
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
    # An event timestamp is optional metadata, not proof that the iteration
    # started.  Use the explicit lifecycle event so terminal reconciliation
    # also works for durable/replayed logs whose ``iter_started`` record did
    # not carry ``ts``.
    started_idxs = sorted(started_iter_indices)
    if started_idxs:
        max_started = max(started_idxs)
        for i, s in by_iter.items():
            if i < max_started and s["status"] == "running":
                s["status"] = "completed"
        terminal_slot = by_iter[max_started]
        if terminal_slot["status"] == "running" and job.status in {
            "errored", "stopped",
        }:
            terminal_slot["status"] = job.status
            terminal_event = next(
                (
                    event for event in reversed(job.events)
                    if event.get("type") in {"run_errored", "run_stopped"}
                ),
                None,
            )
            if terminal_event is not None:
                terminal_slot["completed_at"] = terminal_event.get("ts")

        raw_classification = job.params.get("error_classification")
        if isinstance(raw_classification, dict):
            evidence = raw_classification.get("evidence")
            if (
                isinstance(evidence, dict)
                and evidence.get("iteration") == max_started
            ):
                terminal_slot["failure_stage"] = evidence.get("failure_stage")
                terminal_slot["ppo_iterations_completed"] = evidence.get(
                    "rl_iter"
                )
                terminal_slot["ppo_iterations_requested"] = evidence.get(
                    "rl_total"
                )
                terminal_slot["checkpoint_preserved"] = (
                    evidence.get("checkpoint_preserved") is True
                )
                checkpoint_sha = evidence.get("checkpoint_sha256")
                if isinstance(checkpoint_sha, str):
                    terminal_slot["checkpoint_sha256"] = checkpoint_sha

    return [by_iter[k] for k in sorted(by_iter.keys())]


def build_iterations_summary(job: Job) -> list[dict[str, Any]]:
    return _iter_events(job)


# ── error classification on non-zero exit ─────────────────────────────
def _post_training_rollout_failure_evidence(
    log_path: Path, project_dir: Path,
) -> dict[str, Any] | None:
    """Prove completed/reused training, a later rollout failure, and checkpoint."""
    active_iteration: int | None = None
    terminal: dict[str, int] | None = None
    terminal_source: Literal["ppo_progress", "train_skip"] | None = None
    attested_checkpoint: Path | None = None
    terminal_line = -1
    rollout_started_after_terminal = False
    rollout_failure_after_terminal = False
    try:
        handle = Path(log_path).open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with handle:
        for line_number, raw_line in enumerate(handle):
            marker = raw_line.find(EVENT_TAG)
            if marker >= 0:
                payload_text = raw_line[marker + len(EVENT_TAG):].strip()
                try:
                    event = json.loads(payload_text)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict):
                    if (
                        event.get("type") == "iter_started"
                        and type(event.get("iter")) is int
                        and event["iter"] >= 0
                    ):
                        active_iteration = int(event["iter"])
                        # An outer iteration is a hard evidence boundary.  A
                        # terminal progress record from an earlier iteration
                        # must never classify a later rollout failure.
                        terminal = None
                        terminal_source = None
                        attested_checkpoint = None
                        terminal_line = -1
                        rollout_started_after_terminal = False
                        rollout_failure_after_terminal = False
                    if event.get("type") == "iter_progress":
                        rl_iter = event.get("rl_iter")
                        rl_total = event.get("rl_total")
                        event_iteration = event.get("iter")
                        if (
                            type(rl_iter) is int
                            and type(rl_total) is int
                            and rl_total > 0
                            and rl_iter == rl_total
                            and active_iteration is not None
                            and (
                                type(event_iteration) is not int
                                or event_iteration == active_iteration
                            )
                        ):
                            terminal = {
                                "iteration": active_iteration,
                                "rl_iter": rl_iter,
                                "rl_total": rl_total,
                            }
                            terminal_source = "ppo_progress"
                            attested_checkpoint = None
                            terminal_line = line_number
                            rollout_started_after_terminal = False
                            rollout_failure_after_terminal = False
                    if (
                        event.get("type") == "phase_skipped"
                        and event.get("phase") == "train"
                        and type(event.get("iter")) is int
                        and event.get("iter") == active_iteration
                        and isinstance(event.get("checkpoint"), str)
                    ):
                        # `_train_or_resume` emits this only after validating
                        # the local promoted checkpoint.  Bind the structured
                        # attestation to the exact iteration path again here;
                        # the mere presence of some checkpoint elsewhere must
                        # not upgrade an evaluation failure.
                        checkpoint_path = Path(event["checkpoint"])
                        expected_names = {"checkpoint.pt", "checkpoint.zip"}
                        expected = (
                            Path(project_dir)
                            / "runs"
                            / f"iter_{active_iteration}"
                            / checkpoint_path.name
                        )
                        try:
                            checkpoint_attested = (
                                checkpoint_path.is_absolute()
                                and checkpoint_path.name in expected_names
                                and Path(os.path.abspath(checkpoint_path))
                                == Path(os.path.abspath(expected))
                                and not expected.is_symlink()
                                and expected.is_file()
                                and expected.stat().st_size > 0
                            )
                        except OSError:
                            checkpoint_attested = False
                        if checkpoint_attested:
                            terminal = {"iteration": active_iteration}
                            terminal_source = "train_skip"
                            attested_checkpoint = expected
                            terminal_line = line_number
                            rollout_started_after_terminal = False
                            rollout_failure_after_terminal = False
                    if (
                        terminal is not None
                        and line_number > terminal_line
                        and event.get("type") == "rollout_started"
                    ):
                        rollout_started_after_terminal = True
            if (
                terminal is not None
                and line_number > terminal_line
                and "mjlab rollout runner exited" in raw_line.lower()
            ):
                rollout_failure_after_terminal = True
    if terminal is None or not rollout_failure_after_terminal:
        return None
    checkpoint: Path | None = attested_checkpoint
    if terminal_source == "ppo_progress":
        iter_dir = Path(project_dir) / "runs" / f"iter_{terminal['iteration']}"
        for name in ("checkpoint.pt", "checkpoint.zip"):
            candidate = iter_dir / name
            try:
                if (
                    not candidate.is_symlink()
                    and candidate.is_file()
                    and candidate.stat().st_size > 0
                ):
                    checkpoint = candidate
                    break
            except OSError:
                continue
    if checkpoint is None:
        return None
    try:
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or checkpoint.stat().st_size <= 0
        ):
            return None
        checkpoint_sha256 = _file_sha256(checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size
    except OSError:
        return None
    evidence: dict[str, Any] = {
        "failure_stage": "evaluation",
        "iteration": terminal["iteration"],
        "checkpoint_preserved": True,
        "checkpoint_name": checkpoint.name,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "rollout_started": rollout_started_after_terminal,
    }
    if terminal_source == "ppo_progress":
        evidence.update({
            "rl_iter": terminal["rl_iter"],
            "rl_total": terminal["rl_total"],
        })
    else:
        evidence.update({
            "training_skipped": True,
            "checkpoint_attested": True,
        })
    return evidence


def _classify_run_failure(
    log_path: Path, project_dir: Path,
) -> CudaErrorClass:
    """Classify a non-zero run without conflating train and eval stages."""
    post_training = _post_training_rollout_failure_evidence(
        log_path, project_dir,
    )
    if post_training is not None:
        iteration = int(post_training["iteration"])
        rl_iter = post_training.get("rl_iter")
        rl_total = post_training.get("rl_total")
        if type(rl_iter) is int and type(rl_total) is int:
            stage_detail = (
                f"PPO reached {rl_iter}/{rl_total} for iter {iteration} and "
                "the final checkpoint was preserved"
            )
        else:
            stage_detail = (
                f"Iter {iteration} reused its attested local training "
                "checkpoint"
            )
        return CudaErrorClass(
            kind="post_training_rollout_failed",
            title="Training checkpoint preserved; evaluation failed",
            detail=(
                f"{stage_detail}, but rollout/evaluation "
                "failed before completion and objective evidence were written."
            ),
            suggestions=[
                "Treat this checkpoint as an interrupted recovery input, not "
                "a completed policy.",
                "Fix the rollout/evaluation failure, then relaunch from the "
                "attested actor+critic snapshot.",
            ],
            problem_type="/problems/post-training-rollout-failed",
            evidence=post_training,
        )
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
