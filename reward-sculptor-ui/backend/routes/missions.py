"""Mission orchestrator REST + WebSocket routes (Ship 18a).

Endpoints (all under `/projects/{slug}/missions`):

  POST   /                              — start a `mission_decompose` job
                                          → 202 + JobSummary
  GET    /                              — list missions in this project
                                          → 200 + list[MissionSummary]
  GET    /{mission_slug}                — full mission detail
                                          → 200 + MissionDetail | 404
  POST   /{mission_slug}/run            — start a `mission_execute` job
                                          → 202 + JobSummary
  DELETE /{mission_slug}                — hard-delete the mission dir
                                          → 200 + DeleteMissionResponse
                                          | 409 if active job

  WS     /ws/projects/{slug}/missions/{mission_slug}/events
                                        — replay + tee the active
                                          decompose / execute job's events

Source-of-truth invariant (per Ship 18a plan-review):
  `mission.json` is canonical; GET handlers read it on every call via
  `mission_store.load_mission_*`. JobManager events overlay during in-
  flight jobs (active_job_id / active_job_kind on the response).
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
from fastapi.responses import JSONResponse

from backend.models.kg import JobDetail, JobSummary
from backend.models.mission import (
    CreateMissionRequest,
    DeleteMissionResponse,
    MissionDetail,
    MissionSummary,
    RunMissionRequest,
)
from backend.models.project import ProblemDetail
from backend.services import mission_store
from backend.services.job_manager import JobManager
from backend.services.mission_jobs import (
    run_mission_decompose_job,
    run_mission_execute_job,
)
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["missions"])
ws_router = APIRouter()


# ── DI ───────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ──────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: Optional[str] = None,
    type_: str = "about:blank",
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail,
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _project_dir(store: ProjectStore, slug: str) -> Optional[Path]:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir)


# ── POST /projects/{slug}/missions  (decompose) ──────────────────────
# §Ship-19c response_model fix: was JobSummary (no `params`), so the
# frontend couldn't tell what mission_slug got assigned and the auto-
# open detail dialog + optimistic cache insert silently no-op'd. Use
# JobDetail (which extends JobSummary with `params` + `result`) so
# `params.mission_slug` is on the wire.
@router.post(
    "/projects/{slug}/missions",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
    },
)
def create_mission(
    slug: str,
    body: CreateMissionRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    # Audit fix #E (CRITICAL): two concurrent POSTs without a
    # `mission_slug` override would BOTH read the same `existing_slugs`
    # snapshot and derive the SAME slug, then both jobs would write
    # to the same `<slug>/mission.json`. Acquire the per-project lock
    # to serialize slug-derivation + job-submission. The lock is held
    # ONLY across the cheap reservation step (creating the empty
    # mission_dir to claim the slug); the actual decompose runs
    # without holding the lock so other endpoints aren't blocked.
    try:
        project_lock = store.acquire_lock(slug, timeout=5.0)
    except Exception as e:  # noqa: BLE001 — BusyError or similar
        return _problem(
            status.HTTP_409_CONFLICT,
            "project is busy",
            detail=(
                f"another write is in progress on project {slug!r}: "
                f"{type(e).__name__}: {e}"
            ),
            type_="/problems/state-conflict",
        )

    try:
        existing_slugs = set(mission_store.list_mission_slugs(project_dir))
        if body.mission_slug is not None:
            if body.mission_slug in existing_slugs:
                return _problem(
                    status.HTTP_409_CONFLICT,
                    "mission slug already exists",
                    detail=(
                        f"mission_slug={body.mission_slug!r} is already in "
                        f"use under project {slug!r}. Pick another or omit "
                        f"`mission_slug` to auto-derive."
                    ),
                    type_="/problems/state-conflict",
                )
            mission_slug = body.mission_slug
        else:
            mission_slug = mission_store.derive_unique_mission_slug(
                body.goal, existing_slugs,
            )
        # Reserve the slug atomically by mkdir before releasing the
        # lock — `list_mission_slugs` filters to dirs containing
        # mission.json, but our slug-collision set was JUST `existing
        # _slugs` (dir names). Add an `.in_progress` marker file so a
        # racing list call sees the dir but treats it as "claimed not
        # yet decomposed."
        reserved_dir = mission_store.mission_dir(project_dir, mission_slug)
        reserved_dir.mkdir(parents=True, exist_ok=True)
        # Empty marker — decompose will write mission.json here.
        (reserved_dir / ".decompose_pending").touch()

        # §Ship 21a: thread run_defaults from the NewMissionDialog
        # Advanced tab through to the decompose job so the resulting
        # mission.json carries them; RunMissionDialog pre-fills from
        # them on first open. body.run_defaults is already validated
        # via the RunMissionRequest pydantic shape.
        run_defaults_dict = (
            body.run_defaults.model_dump(exclude_none=False)
            if body.run_defaults is not None else None
        )
        job = jobs.submit(
            kind="mission_decompose",
            project_slug=slug,
            fn=run_mission_decompose_job(
                project_dir=project_dir,
                project_slug=slug,
                goal=body.goal,
                mission_slug=mission_slug,
                no_kg=body.no_kg,
                run_defaults=run_defaults_dict,
                gen_stage_metrics=body.gen_stage_metrics,
                stage_metric_candidates=body.stage_metric_candidates,
            ),
            params={
                "mission_slug": mission_slug,
                "goal": body.goal,
                "no_kg": body.no_kg,
                # Surface in params so the frontend optimistic-cache
                # write can show the user the Advanced settings round-
                # tripped successfully.
                "run_defaults": run_defaults_dict,
            },
        )
    finally:
        project_lock.release()
    # §Ship-19c: return JobDetail (has `params`) so the frontend's
    # auto-open detail dialog can read mission_slug from the response.
    return job.to_detail()


# ── GET /projects/{slug}/missions  (list) ────────────────────────────
@router.get(
    "/projects/{slug}/missions",
    response_model=list[MissionSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_missions(
    slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    out: list[MissionSummary] = []
    for mission_slug in mission_store.list_mission_slugs(project_dir):
        active = jobs.active_mission_job(slug, mission_slug)
        summary = mission_store.load_mission_summary(
            project_dir, slug, mission_slug,
            active_job_kind=active.kind if active else None,
            active_job_id=active.job_id if active else None,
        )
        if summary is not None:
            out.append(summary)
    # Newest-first (mirror project list ordering).
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


# ── GET /projects/{slug}/missions/{mission_slug}  (detail) ───────────
@router.get(
    "/projects/{slug}/missions/{mission_slug}",
    response_model=MissionDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    active = jobs.active_mission_job(slug, mission_slug)
    detail = mission_store.load_mission_detail(
        project_dir, slug, mission_slug,
        active_job_kind=active.kind if active else None,
        active_job_id=active.job_id if active else None,
    )
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            detail=(
                f"no mission {mission_slug!r} under project {slug!r}. "
                f"Available: {mission_store.list_mission_slugs(project_dir)}"
            ),
            type_="/problems/not-found",
        )
    return detail


# ── POST /projects/{slug}/missions/{mission_slug}/run  (execute) ─────
@router.post(
    "/projects/{slug}/missions/{mission_slug}/run",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def run_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    body: Optional[RunMissionRequest] = None,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if not mission_store.mission_json_path(
        project_dir, mission_slug,
    ).is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            detail=(
                f"no mission {mission_slug!r} under project {slug!r}"
            ),
            type_="/problems/not-found",
        )

    # Concurrent-mission guard: at most one active decompose/execute
    # PER (project, mission) pair.
    if jobs.active_mission_job(slug, mission_slug) is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                f"mission {mission_slug!r} already has an active "
                f"decompose or execute job. Wait for it to finish "
                f"or call POST /jobs/{{id}}/stop to cancel."
            ),
            type_="/problems/state-conflict",
        )

    # GPU contention guard: refuse if any sculpt_run / mission_execute
    # is in flight cross-project (per Ship 18a plan-review).
    if jobs.has_any_active_gpu_job():
        return _problem(
            status.HTTP_409_CONFLICT,
            "GPU is busy",
            detail=(
                "Another sculpt_run or mission_execute is in flight. "
                "Mission execution requires exclusive GPU access "
                "(spawns per-stage sculpt_run subprocesses)."
            ),
            type_="/problems/state-conflict",
        )

    # §Ship-19d: optional run-time knobs from the RunMissionDialog.
    # An empty / missing body is the Ship 16 default (per-stage budget
    # from mission.json, no Goal A/B). The backend forwards these as
    # CLI flags to `sculpt mission-run` (see run_mission_execute_job).
    run_kwargs: dict[str, Any] = {}
    if body is not None:
        # Pydantic model_dump excludes unset fields with exclude_none.
        run_kwargs = body.model_dump(exclude_none=False)

    job = jobs.submit(
        kind="mission_execute",
        project_slug=slug,
        fn=run_mission_execute_job(
            project_dir=project_dir,
            project_slug=slug,
            mission_slug=mission_slug,
            run_kwargs=run_kwargs,
            # §Ship 21: pass JobManager so the streamer can register
            # per-stage child Jobs (mission_stage_run kind) on the fly.
            job_manager=jobs,
        ),
        params={"mission_slug": mission_slug, **run_kwargs},
    )
    return job.to_detail()


# ── DELETE /projects/{slug}/missions/{mission_slug} ──────────────────
@router.delete(
    "/projects/{slug}/missions/{mission_slug}",
    response_model=DeleteMissionResponse,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def delete_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            type_="/problems/not-found",
        )

    if not mission_store.mission_dir(project_dir, mission_slug).is_dir():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            type_="/problems/not-found",
        )

    if jobs.active_mission_job(slug, mission_slug) is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                "Stop the active decompose / execute job before "
                "deleting the mission."
            ),
            type_="/problems/state-conflict",
        )

    freed = mission_store.delete_mission(project_dir, mission_slug)
    return DeleteMissionResponse(
        mission_slug=mission_slug, freed_bytes=freed,
    )


# ── WS /ws/projects/{slug}/missions/{mission_slug}/events ────────────
@ws_router.websocket(
    "/ws/projects/{slug}/missions/{mission_slug}/events",
)
async def mission_events_ws(
    websocket: WebSocket,
    slug: str,
    mission_slug: str,
) -> None:
    """Replay + live-tee the active mission_decompose or
    mission_execute job's events for a (project, mission) pair.

    Mirrors the runs WebSocket pattern at runs.py — same envelope:
    `connected` snapshot, replay (last 200 log_line + all typed
    events), live forward, `terminal` on job end.
    """
    await websocket.accept()
    app = websocket.app
    store: ProjectStore = app.state.project_store
    jobs: JobManager = app.state.job_manager

    if store.get(slug) is None:
        await websocket.send_json({
            "type": "error",
            "error": f"project {slug!r} not found",
        })
        await websocket.close(code=4404)
        return

    job = jobs.active_mission_job(slug, mission_slug)
    if job is None:
        # Allow connecting even when no active job — the UI may want
        # to subscribe in advance. Send a pseudo-connected with an
        # empty event list and close cleanly.
        await websocket.send_json({
            "type": "connected",
            "status": "no_active_job",
            "total_event_count": 0,
            "replay_count": 0,
        })
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    # Replay: last 200 log_line + ALL typed events.
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
            "kind": job.kind,
            "job_id": job.job_id,
            "total_event_count": len(job.events),
            "replay_count": len(replay_events),
        })
        for ev in replay_events:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    queue = job.subscribe()
    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json(
                {"type": "terminal", "status": job.status},
            )
        finally:
            job.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
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
            terminal_types = (
                "mission_decompose_completed", "mission_decompose_errored",
                "mission_execute_completed", "mission_execute_errored",
                "mission_completed", "mission_halted_terminal",
            )
            if ev.get("type") in terminal_types:
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "terminal", "status": job.status},
                    )
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _drain_client(ws: WebSocket) -> None:
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return
