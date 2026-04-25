"""Sculpt-run REST routes.

Runs are JobManager jobs of kind `sculpt_run`. This router is a thin
view over `JobManager` — there is no parallel state tracker per
Prompt 8 R3. The WebSocket route is in this same module for locality.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from backend.models.project import ProblemDetail
from backend.models.run import IterEventSummary, RunDetail, RunParams, RunSummary
from backend.services.job_manager import Job, JobManager
from backend.services.project_store import ProjectStore
from backend.services.run_manager import build_iterations_summary, run_sculpt_job


router = APIRouter(tags=["runs"])


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ────────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: Optional[str] = None,
    type_: str = "about:blank",
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _find_run(jobs: JobManager, slug: str, run_id: str) -> Optional[Job]:
    job = jobs.get(run_id)
    if job is None:
        return None
    # §Ship 21: also accept mission_stage_run kind so per-stage rows
    # in /runs are addressable by the standard run-detail / clip /
    # rollout / WS routes.
    if job.kind not in ("sculpt_run", "mission_stage_run"):
        return None
    if job.project_slug != slug:
        return None
    return job


def _resolve_run_root(job: Job, project_dir: Path) -> Path:
    """§Ship 21: where this run's `runs/iter_*/` artifacts live.

    For top-level sculpt_run jobs: `<project_dir>/runs/`.
    For mission_stage_run jobs: `<project_dir>/.missions/<mission_slug>/
    stages/<stage_name>/runs/` — each stage scaffolds its own mini-
    project with its own runs/ tree.
    """
    if job.kind == "mission_stage_run":
        mission_slug = job.params.get("mission_slug")
        stage_name = job.params.get("stage_name")
        if mission_slug and stage_name:
            return (
                project_dir / ".missions" / str(mission_slug)
                / "stages" / str(stage_name) / "runs"
            )
    return project_dir / "runs"


def _run_summary(job: Job) -> RunSummary:
    params = job.params or {}
    iters = build_iterations_summary(job)
    completed = sum(1 for it in iters if it.get("status") == "completed")
    current = next(
        (it["iter_index"] for it in iters if it.get("status") == "running"),
        None,
    )
    history_from_params = params.get("primary_metric_history")
    if isinstance(history_from_params, list):
        metric_history = [
            (float(v) if isinstance(v, (int, float)) else None)
            for v in history_from_params
        ]
    else:
        metric_history = [
            (it.get("primary_metric") if isinstance(it.get("primary_metric"), (int, float)) else None)
            for it in iters
        ]
    classification_raw = params.get("error_classification")
    classification = None
    if isinstance(classification_raw, dict):
        from backend.models.run import ErrorClassification

        try:
            classification = ErrorClassification(**classification_raw)
        except Exception:  # noqa: BLE001
            classification = None

    # §Ship 21: surface mission/stage context for mission_stage_run
    # rows so the Runs sidebar can group them under their parent
    # mission, and the detail pane can route per-stage rewards.
    return RunSummary(
        run_id=job.job_id,
        project_slug=job.project_slug or "",
        status=job.status,
        behavior_goal=str(params.get("behavior_goal") or ""),
        iterations_requested=int(params.get("iterations_requested") or params.get("iterations") or 0),
        iterations_completed=completed,
        current_iter_index=current,
        primary_metric_history=metric_history,
        started_at=job.started_at,
        ended_at=job.ended_at,
        error=job.error,
        error_classification=classification,
        kind=job.kind,  # type: ignore[arg-type]
        parent_id=job.parent_id,
        mission_slug=params.get("mission_slug"),
        stage_name=params.get("stage_name"),
        stage_index=params.get("stage_index"),
    )


# ── POST /projects/{slug}/runs ────────────────────────────────────────
@router.post(
    "/projects/{slug}/runs",
    response_model=RunSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        412: {"model": ProblemDetail},
    },
)
def launch_run(
    slug: str,
    body: RunParams,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run already active",
            detail=(
                "This project already has a sculpt run in the running "
                "/ queued state. Stop it before launching another."
            ),
            type_="/problems/job-busy",
        )

    # Live runs require ANTHROPIC_API_KEY unless --dry-run is set.
    if not body.dry_run:
        import os

        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "ANTHROPIC_API_KEY not set",
                detail=(
                    "Live sculpt runs need ANTHROPIC_API_KEY to diagnose + "
                    "edit. Set it in the environment or pass dry_run=true."
                ),
                type_="/problems/no-api-key",
            )

    project_dir = Path(detail.project_dir)
    run_params: dict[str, Any] = body.model_dump()
    job = jobs.submit(
        kind="sculpt_run",
        project_slug=slug,
        fn=run_sculpt_job(project_dir=project_dir, run_params=run_params),
        params=run_params,
    )
    return _run_summary(job)


# ── GET /projects/{slug}/runs ─────────────────────────────────────────
@router.get(
    "/projects/{slug}/runs",
    response_model=list[RunSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_runs(
    slug: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    # §Ship 21: include `mission_stage_run` rows (per-stage child jobs
    # registered by mission_jobs._stream_stdout). Sidebar groups them
    # under their parent mission.
    sculpt_runs = jobs.list(kind="sculpt_run", project_slug=slug)
    stage_runs = jobs.list(kind="mission_stage_run", project_slug=slug)
    return [_run_summary(j) for j in (*sculpt_runs, *stage_runs)]


# ── GET /projects/{slug}/runs/{run_id} ────────────────────────────────
@router.get(
    "/projects/{slug}/runs/{run_id}",
    response_model=RunDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_run(
    slug: str,
    run_id: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "run not found",
            detail=f"no sculpt_run with id {run_id!r} in project {slug!r}",
            type_="/problems/not-found",
        )
    summary = _run_summary(job)
    # §Ship 21: stage runs don't have a top-level RunParams (the
    # parent mission_execute owns those); synthesize a minimal one
    # from the stage's recorded fields so RunDetail's existing shape
    # holds. behavior_goal becomes the stage's goal_text.
    iters_for_params = int(
        job.params.get("iterations_requested")
        or job.params.get("iterations")
        or 1
    )
    params = RunParams(
        behavior_goal=str(job.params.get("behavior_goal") or ""),
        iterations=iters_for_params,
        no_kg=bool(job.params.get("no_kg") or False),
        dry_run=bool(job.params.get("dry_run") or False),
    )
    iterations = [
        IterEventSummary(**it) for it in build_iterations_summary(job)
    ]
    return RunDetail(
        **summary.model_dump(),
        params=params,
        iterations=iterations,
        stdout_tail=list(job.log_ring),
        total_event_count=len(job.events),
    )


# ── DELETE /projects/{slug}/runs/{run_id} ─────────────────────────────
@router.delete(
    "/projects/{slug}/runs/{run_id}",
    response_model=RunSummary,
    responses={404: {"model": ProblemDetail}},
)
def kill_run(
    slug: str,
    run_id: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "run not found",
            type_="/problems/not-found",
        )
    jobs.stop(run_id)
    job.emit({"type": "run_stopped", "source": "user"})
    return _run_summary(job)


# ── WS /projects/{slug}/runs/{run_id}/events ──────────────────────────
@router.websocket("/ws/projects/{slug}/runs/{run_id}/events")
async def run_events_ws(
    websocket: WebSocket, slug: str, run_id: str,
) -> None:
    jobs: JobManager = websocket.app.state.job_manager
    store: ProjectStore = websocket.app.state.project_store

    await websocket.accept()
    if store.get(slug) is None:
        await websocket.send_json({
            "type": "error",
            "error": f"project {slug!r} not found",
        })
        await websocket.close(code=4404)
        return
    job = _find_run(jobs, slug, run_id)
    if job is None:
        await websocket.send_json({
            "type": "error",
            "error": f"run {run_id!r} not found in project {slug!r}",
        })
        await websocket.close(code=4404)
        return

    # Replay: send a `connected` snapshot + the buffered events. The
    # frontend uses `seq` to dedupe on reconnect. We cap the replay at
    # the last 200 log lines + ALL typed events (typed events are small).
    replay_log_count = 0
    replay_events: list[dict[str, Any]] = []
    log_budget = 200
    for ev in reversed(job.events):
        if ev.get("type") == "log_line":
            if replay_log_count < log_budget:
                replay_events.append(ev)
                replay_log_count += 1
        else:
            replay_events.append(ev)
    replay_events.reverse()

    try:
        await websocket.send_json({
            "type": "connected",
            "status": job.status,
            "total_event_count": len(job.events),
            "replay_count": len(replay_events),
        })
        for ev in replay_events:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    queue = job.subscribe()
    # If the job already terminated we still want the client to receive
    # any final events that landed between replay and subscribe — the
    # job.events append is atomic, so replaying once covers it.
    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json(
                {"type": "terminal", "status": job.status}
            )
        finally:
            job.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    # Live loop.
    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                break
            ev = get_task.result()
            try:
                await websocket.send_json(ev)
            except (WebSocketDisconnect, RuntimeError):
                break
            if ev.get("type") in ("run_completed", "run_errored", "run_stopped"):
                # Give the client a moment to receive, then close.
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "terminal", "status": job.status}
                    )
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _drain_client(ws: WebSocket) -> None:
    """Keep the receive side pumping so the client can disconnect cleanly.
    We don't expect any inbound messages — the WS is one-directional —
    but the server must still await `receive` to notice close frames."""
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return


# ── WS /frames (live clip push) ───────────────────────────────────────
_FRAME_EVENT_TYPES = {"new_clip", "clip_skipped"}


@router.websocket("/ws/projects/{slug}/runs/{run_id}/frames")
async def run_frames_ws(
    websocket: WebSocket, slug: str, run_id: str,
) -> None:
    """Thin filtered view of the job's event stream. Only forwards
    `new_clip` and `clip_skipped` events. Replay mirrors /events — on
    connect the client gets `connected` + every prior clip event so
    they can populate the replay strip without an extra REST round."""
    jobs: JobManager = websocket.app.state.job_manager
    store: ProjectStore = websocket.app.state.project_store

    await websocket.accept()
    if store.get(slug) is None:
        await websocket.send_json({"type": "error", "error": f"project {slug!r} not found"})
        await websocket.close(code=4404)
        return
    job = _find_run(jobs, slug, run_id)
    if job is None:
        await websocket.send_json({"type": "error", "error": f"run {run_id!r} not found"})
        await websocket.close(code=4404)
        return

    clips = [ev for ev in job.events if ev.get("type") in _FRAME_EVENT_TYPES]
    try:
        await websocket.send_json({
            "type": "connected",
            "status": job.status,
            "replay_count": len(clips),
        })
        for ev in clips:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json({"type": "terminal", "status": job.status})
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()
        return

    queue = job.subscribe()
    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                break
            ev = get_task.result()
            etype = ev.get("type")
            if etype in _FRAME_EVENT_TYPES:
                try:
                    await websocket.send_json(ev)
                except (WebSocketDisconnect, RuntimeError):
                    break
            if etype in ("run_completed", "run_errored", "run_stopped"):
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "terminal", "status": job.status})
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


# ── clip + rollout file serving (Range-aware) ─────────────────────────
@router.get(
    "/projects/{slug}/runs/{run_id}/clips/{file}",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_clip_file(
    slug: str, run_id: str, file: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(404, "project not found", type_="/problems/not-found")
    if _find_run(jobs, slug, run_id) is None:
        return _problem(404, "run not found", type_="/problems/not-found")

    # Harden: only allow "iter_<N>.mp4" filenames, no traversal.
    if not _CLIP_NAME_RE.match(file):
        return _problem(
            404, "clip not found",
            detail=f"{file!r} is not a valid clip filename",
            type_="/problems/not-found",
        )
    project_dir = Path(detail.project_dir)
    path = project_dir / "uploads" / "live_clips" / run_id / file
    try:
        resolved = path.resolve()
        resolved.relative_to(
            (project_dir / "uploads" / "live_clips" / run_id).resolve()
        )
    except (ValueError, OSError):
        return _problem(
            404, "clip not found", type_="/problems/not-found"
        )
    if not resolved.is_file():
        return _problem(
            404, "clip not found",
            detail=f"no clip at {resolved.name}",
            type_="/problems/not-found",
        )
    return FileResponse(resolved, media_type="video/mp4")


@router.get(
    "/projects/{slug}/runs/{run_id}/iterations/{iter_index}/rollout",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_iter_rollout(
    slug: str, run_id: str, iter_index: int,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(404, "project not found", type_="/problems/not-found")
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(404, "run not found", type_="/problems/not-found")
    project_dir = Path(detail.project_dir)
    # §Ship 21: stage runs live under .missions/<m>/stages/<s>/runs/
    runs_root = _resolve_run_root(job, project_dir)
    path = runs_root / f"iter_{iter_index}" / "rollout" / "rollout.mp4"
    if not path.is_file() or path.stat().st_size < 2048:
        return _problem(
            404, "rollout not available",
            detail=(
                f"iter {iter_index} has no rollout.mp4 yet (either "
                "still rendering or the run errored before this iter)"
            ),
            type_="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")


import re
_CLIP_NAME_RE = re.compile(r"^iter_\d+\.mp4$")
