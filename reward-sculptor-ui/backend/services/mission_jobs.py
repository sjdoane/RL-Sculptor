"""Background runners for mission jobs (Ship 18a).

Two job kinds, two strategies (per Ship 18a plan-review):

  * `mission_decompose` — IN-PROCESS via `asyncio.to_thread`.
    Decompose is a single Claude call + pydantic validation
    (~30-90s); subprocess fork buys nothing here. The async wrap
    keeps the event loop unblocked.

  * `mission_execute` — SUBPROCESS via `sculpt mission-run ...`.
    Mission execution chains N stages (each a sculpt_run subprocess)
    spanning hours. Subprocess isolation matches the proven
    `sculpt_run` pattern: kill semantics, log durability, GPU
    process isolation.

Both runners' output is tee'd into `job.events`:
  - `mission_decompose`: emits {`mission_decompose_started`, `...
    completed`, `... errored`} from inside this process.
  - `mission_execute`: parses `[SCULPT-EVENT] {...}` markers from
    the subprocess stdout (mirrors run_manager's stream pattern).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

# Eager sculptor import: same rationale as kg_jobs.py — `.env` must
# load before any Anthropic-key check downstream.
import sculptor  # noqa: F401

from backend.services import mission_store
from backend.services.job_manager import Job, JobManager


# Stdout marker the sculpt CLI uses for structured events.
_EVENT_TAG = "[SCULPT-EVENT]"

# §Ship 21e: guard for stage_name before it's used to build a path.
import re as _re
_SAFE_STAGE_NAME = _re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# §Ship 21: stage lifecycle event types. The streamer creates a child
# Job (kind="mission_stage_run") on `stage_started`; tee's iter_*
# events to it; closes it on a stage-terminal event.
_STAGE_OPEN_EVENT = "stage_started"
_STAGE_CLOSE_EVENTS = frozenset({
    "stage_succeeded",
    "stage_failed",
    "stage_skipped",
})
# Events emitted from inside the per-stage sculpt_run subprocess that
# the parent's streamer sees AFTER `stage_started` and BEFORE the
# stage-close event. These get tee'd to the active child Job's event
# stream so the Runs tab's existing iter timeline / metric chart /
# log viewer machinery renders the stage exactly like a top-level
# sculpt_run.
_STAGE_TEE_EVENTS = frozenset({
    "iter_started",
    "iter_completed",
    "iter_progress",
    "iter_rolled_out",
    "iter_diagnosed",
    "rollout_done",
    "stage_warm_start_resolved",
    "warm_start_skipped",
    "stage_warm_start_chosen",
    "stage_scaffolded",
    "stage_v1_materialized",
    "stage_completed_training",
    "stage_criterion_evaluated",
    "stage_extension_skipped",
    "stage_extension_exhausted",
    "stage_extended",
    "stage_skill_published",
    "stage_skill_publish_skipped",
    "skill_warm_start_skipped",
    "feedback_read_degraded",
    "stage_redecomposition_started",
    "stage_redecomposed",
    "redecomposition_skipped",
    "stage_redecomposition_failed",
})


# ── In-process decompose ─────────────────────────────────────────────
def run_mission_decompose_job(
    *,
    project_dir: Path,
    project_slug: str,
    goal: str,
    mission_slug: str,
    no_kg: bool = False,
    run_defaults: Optional[dict[str, Any]] = None,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Async runner that calls `sculptor.decompose.decompose_task` and
    persists the resulting mission to `<project_dir>/.missions/
    <mission_slug>/mission.json`. In-process via `asyncio.to_thread`.

    Args
    ----
    project_dir : project root with `config.toml`.
    project_slug : for event payload routing.
    goal : NL behavior goal.
    mission_slug : pre-resolved unique slug (caller derived via
        `mission_store.derive_unique_mission_slug`).
    no_kg : skip KG context to Claude (faster, less grounded).
    run_defaults : §Ship 21a — optional run-time defaults set up
        front via the NewMissionDialog Advanced tab. Persisted on the
        Mission so RunMissionDialog can pre-fill when the user later
        clicks Run mission. Validated by the route layer against
        RunMissionRequest's pydantic shape before reaching here.
    """

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.05
        job.message = f"decomposing into stages (mission_slug={mission_slug})"
        job.emit({
            "type": "mission_decompose_started",
            "mission_slug": mission_slug,
            "project_slug": project_slug,
            "goal": goal,
        })

        def _do_decompose() -> dict[str, Any]:
            from sculptor.adapters.base import load_adapter
            from sculptor.decompose import decompose_task
            from sculptor.kg.store import SculptorKG
            from sculptor.mission import save_mission

            config_path = project_dir / "config.toml"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"{config_path} not found — is this a sculpt project?"
                )

            adapter = load_adapter(config_path)
            reward_contract = adapter.reward_contract()

            kg_store = None if no_kg else SculptorKG()
            try:
                mission = decompose_task(
                    goal, reward_contract, kg_store=kg_store,
                )
            finally:
                if kg_store is not None:
                    kg_store.close()

            md = mission_store.mission_dir(project_dir, mission_slug)
            mission.mission_dir = str(md.resolve())
            # §Ship 21a: persist run_defaults on the Mission BEFORE
            # save so the first GET /missions/{slug} after decompose
            # completes already includes them.
            if run_defaults:
                mission.run_defaults = dict(run_defaults)
            save_mission(mission, md)
            return {
                "mission_slug": mission_slug,
                "n_stages": len(mission.stages),
                "mission_dir": str(md.resolve()),
                "decomposition_rationale": mission.decomposition_rationale,
            }

        try:
            result = await asyncio.to_thread(_do_decompose)
        except Exception as e:  # noqa: BLE001
            job.emit({
                "type": "mission_decompose_errored",
                "mission_slug": mission_slug,
                "error_kind": type(e).__name__,
                "error": str(e),
            })
            # Re-raise so the JobManager marks the job errored with
            # this exception's text in `job.error`.
            raise

        job.progress = 1.0
        job.message = (
            f"decomposed into {result['n_stages']} stages"
        )
        job.emit({
            "type": "mission_decompose_completed",
            **result,
        })
        return result

    return _runner


# ── Subprocess execute ───────────────────────────────────────────────
def _build_mission_run_flags(run_kwargs: dict[str, Any]) -> list[str]:
    """Translate the RunMissionRequest body into `sculpt mission-run`
    CLI flags. Skips fields whose value is None (defer to typer's
    Option default) and skips False booleans (typer interprets the
    flag as ON when present). Ship 19d.
    """
    flags: list[str] = []
    int_flags = [
        ("iterations_override", "--iterations-override"),
        ("steps_per_iter", "--steps-per-iter"),
        ("seed", "--seed"),
        ("criterion_stability_window", "--criterion-stability-window"),
        ("max_extensions_per_stage", "--max-extensions-per-stage"),
    ]
    for key, flag in int_flags:
        v = run_kwargs.get(key)
        if v is not None:
            flags += [flag, str(int(v))]
    float_flags = [
        ("extension_factor", "--extension-factor"),
        ("extension_improvement_threshold",
         "--extension-improvement-threshold"),
    ]
    for key, flag in float_flags:
        v = run_kwargs.get(key)
        if v is not None:
            flags += [flag, str(float(v))]
    bool_flags = [
        ("early_stop_on_criterion", "--early-stop-on-criterion"),
        ("extend_on_improvement", "--extend-on-improvement"),
    ]
    for key, flag in bool_flags:
        if run_kwargs.get(key) is True:
            flags.append(flag)
    return flags


def run_mission_execute_job(
    *,
    project_dir: Path,
    project_slug: str,
    mission_slug: str,
    run_kwargs: Optional[dict[str, Any]] = None,
    job_manager: Optional[JobManager] = None,
) -> Callable[[Job, asyncio.Event], Awaitable[dict[str, Any]]]:
    """Async runner that spawns `sculpt mission-run` as a subprocess
    and streams its `[SCULPT-EVENT]` markers into `job.events`.
    Mirrors run_manager's pattern but doesn't need the heavy
    filesystem watcher / heartbeat — mission events are dense enough
    that stdout is the canonical event stream.

    §Ship-19d `run_kwargs` (optional) — translated into CLI flags
    for `sculpt mission-run` so the UI's RunMissionDialog can knob
    iterations / Goal A / Goal B without touching mission.json.
    """
    extra_flags = _build_mission_run_flags(run_kwargs or {})

    async def _runner(job: Job, cancel: asyncio.Event) -> dict[str, Any]:
        job.progress = 0.0
        job.message = f"running mission {mission_slug}"
        job.emit({
            "type": "mission_execute_started",
            "mission_slug": mission_slug,
            "project_slug": project_slug,
            "run_kwargs": dict(run_kwargs or {}),
        })

        # Build cmd. `sculpt` is a typer entry point; spawn via
        # `python -m sculptor.cli` so we don't depend on PATH having
        # the entry-point shim (works in `uv run`, `pip install`,
        # editable installs uniformly).
        cmd = [
            sys.executable, "-m", "sculptor.cli",
            "mission-run",
            str(project_dir.resolve()),
            mission_slug,
            *extra_flags,
        ]

        env = dict(os.environ)
        # Mirror runs.py: opt out of wandb prompts in mission-execute
        # subprocesses too — every stage spawns its own sculpt-run
        # subprocess inheriting this env.
        env.setdefault("WANDB_MODE", "disabled")

        # Per-job log file under the mission dir for durability.
        md = mission_store.mission_dir(project_dir, mission_slug)
        md.mkdir(parents=True, exist_ok=True)
        log_path = md / f"_execute_{job.job_id}.log"
        job.params["log_file"] = str(log_path)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        job.params["pid"] = proc.pid

        killer = asyncio.create_task(_kill_on_cancel(proc, cancel))
        reader = asyncio.create_task(_stream_stdout(
            job, proc, log_path,
            job_manager=job_manager,
            project_slug=project_slug,
            mission_slug=mission_slug,
        ))

        rc = await proc.wait()
        cancel.set()
        for t in (reader, killer):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if rc == 0:
            job.emit({
                "type": "mission_execute_completed",
                "return_code": rc,
                "mission_slug": mission_slug,
            })
        else:
            job.status = "errored"
            job.error = f"sculpt mission-run exited with code {rc}"
            job.emit({
                "type": "mission_execute_errored",
                "return_code": rc,
                "mission_slug": mission_slug,
                "error": job.error,
            })

        return {
            "return_code": rc,
            "mission_slug": mission_slug,
            "log_file": str(log_path),
        }

    return _runner


# ── Subprocess plumbing (modeled on run_manager.py) ──────────────────
async def _stream_stdout(
    job: Job,
    proc: asyncio.subprocess.Process,
    log_path: Path,
    *,
    job_manager: Optional[JobManager] = None,
    project_slug: Optional[str] = None,
    mission_slug: Optional[str] = None,
) -> None:
    """Tee subprocess stdout into job.events: every line as `log_line`,
    plus parsed `[SCULPT-EVENT]` JSON as a typed event.

    §Ship 21: when `job_manager` is supplied, register a child
    `mission_stage_run` Job for each stage_started event seen so the
    stage appears as a first-class entry in /runs (with its own iter
    timeline, metric chart, log viewer). Tee subsequent iter_* + log
    events to that child until the stage closes.
    """
    assert proc.stdout is not None

    try:
        logf = log_path.open("a", encoding="utf-8", buffering=1)
    except OSError:
        logf = None

    # Active child Job for the currently-running stage, or None.
    active_stage_job: Optional[Job] = None

    def _close_active_stage(final_status: str, error_msg: Optional[str] = None) -> None:
        nonlocal active_stage_job
        if active_stage_job is None:
            return
        from datetime import datetime as _dt, timezone as _tz
        active_stage_job.status = final_status  # type: ignore[assignment]
        active_stage_job.ended_at = _dt.now(tz=_tz.utc)
        if error_msg is not None:
            active_stage_job.error = error_msg
        active_stage_job = None

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
            log_event = {"type": "log_line", "text": text}
            job.emit(log_event)
            # §Ship 21: tee log lines to the active child stage job too
            # so the per-stage Runs detail pane sees stdout in real time.
            if active_stage_job is not None:
                active_stage_job.emit(dict(log_event))

            if not text.startswith(_EVENT_TAG):
                continue
            payload_str = text[len(_EVENT_TAG):].strip()
            try:
                payload = json.loads(payload_str)
            except Exception:  # noqa: BLE001
                continue
            if not (isinstance(payload, dict) and payload.get("type")):
                continue
            ev = dict(payload)
            ev["source"] = "stdout"
            ev_type = ev["type"]

            # Always emit on the parent (mission_execute) job.
            job.emit(ev)

            # §Ship 21: child stage-job lifecycle.
            if (
                ev_type == _STAGE_OPEN_EVENT
                and job_manager is not None
                and project_slug is not None
                and mission_slug is not None
            ):
                # Close any prior unclosed stage (defensive — should not
                # happen in normal flow because each stage emits a
                # close event, but if the orchestrator crashed mid-
                # stage the previous one is still "running"; mark it
                # errored so the UI's status is consistent).
                if active_stage_job is not None:
                    _close_active_stage(
                        "errored",
                        error_msg="stage_started fired without prior close event",
                    )
                # §Ship 21e (review fix): sanitize stage_name before it
                # enters job.params — it later builds a filesystem path
                # (routes/runs.py:_resolve_run_root → FileResponse).
                # Stage names are snake_case ≤32 chars by decompose-time
                # validation, but defense-in-depth against a corrupt /
                # hand-edited mission.json: reject anything outside the
                # safe charset and fall back to a non-traversing label.
                raw_stage_name = ev.get("stage_name") or "stage_unknown"
                stage_name = (
                    raw_stage_name
                    if _SAFE_STAGE_NAME.match(str(raw_stage_name))
                    else "stage_invalid"
                )
                stage_dir = ev.get("stage_dir")
                goal_text = ev.get("goal_text") or ""
                eff_max = ev.get("effective_max_iterations")
                stage_index = ev.get("stage_index")
                active_stage_job = job_manager.register_passive_job(
                    kind="mission_stage_run",
                    project_slug=project_slug,
                    parent_id=job.job_id,
                    params={
                        "mission_slug": mission_slug,
                        "stage_name": stage_name,
                        "stage_index": stage_index,
                        "stage_dir": stage_dir,
                        "behavior_goal": goal_text,
                        "iterations_requested": eff_max,
                        "parent_mission_goal": job.params.get("mission_slug"),
                    },
                )
                # Also emit the stage_started on the child for replay.
                active_stage_job.emit(dict(ev))
                continue

            if active_stage_job is not None and ev_type in _STAGE_TEE_EVENTS:
                active_stage_job.emit(dict(ev))

            if active_stage_job is not None and ev_type in _STAGE_CLOSE_EVENTS:
                # Map stage event type → job status. stage_succeeded /
                # stage_skipped → completed; stage_failed → errored.
                final = (
                    "errored" if ev_type == "stage_failed" else "completed"
                )
                err = (
                    ev.get("reason") or ev.get("detail")
                    if ev_type == "stage_failed"
                    else None
                )
                # Emit the close event on the child too so the UI
                # observes the terminal transition.
                active_stage_job.emit(dict(ev))
                _close_active_stage(final, error_msg=err)
    finally:
        # Parent subprocess ended (success or kill). Any still-open
        # child stage gets marked errored so the Runs UI doesn't show
        # a perpetually-running row.
        if active_stage_job is not None:
            _close_active_stage(
                "errored",
                error_msg="parent mission_execute terminated mid-stage",
            )
        if logf is not None:
            try:
                logf.close()
            except OSError:
                pass


async def _kill_on_cancel(
    proc: asyncio.subprocess.Process,
    cancel: asyncio.Event,
    *,
    grace_s: float = 5.0,
) -> None:
    """SIGTERM → wait grace_s → SIGKILL, mirroring run_manager."""
    await cancel.wait()
    if proc.returncode is not None:
        return
    try:
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
