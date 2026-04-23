"""Job-status endpoints.

  GET   /jobs/{job_id}             — detail
  GET   /jobs                      — list (filterable by kind/status/slug)
  POST  /jobs/{job_id}/stop        — cooperative cancel
  WS    /ws/jobs/{job_id}/events   — live event stream (log_line + status)

UI polls GET /jobs/{job_id} every 3s (per Prompt 7 precondition R3)
until status is terminal. Long-running jobs (reward prompt-edit,
physics prompt-edit, KG extract) emit `log_line` events via
`job.emit()`; the WS endpoint replays buffered events on connect +
streams live ones. Mirrors the contract of
`/ws/projects/{slug}/runs/{run_id}/events` but keyed on job_id so
Rewards / Physics / KG tabs can subscribe without inventing a slug.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Optional

from fastapi import (
    APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse

from backend.models.kg import JobDetail, JobKind, JobStatus, JobSummary
from backend.models.project import ProblemDetail
from backend.services.job_manager import JobManager


router = APIRouter(prefix="/jobs", tags=["jobs"])

# Separate router without the `/jobs` prefix — used to register the
# `/ws/jobs/{job_id}/events` WebSocket at the same mount depth as
# `/ws/projects/.../runs/.../events` (which also lives on a
# no-prefix runs router). Wired into the app in backend/main.py.
ws_router = APIRouter(tags=["jobs"])


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _problem(
    status_code: int, title: str, *, detail: str | None = None,
    type_: str = "about:blank", **extra: Any,
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


@router.get("", response_model=list[JobSummary])
def list_jobs(
    kind: Optional[JobKind] = Query(None),
    status_filter: Optional[JobStatus] = Query(None, alias="status"),
    project_slug: Optional[str] = Query(None),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    return [
        j.to_summary()
        for j in jobs.list(kind=kind, status=status_filter, project_slug=project_slug)
    ]


@router.get(
    "/{job_id}",
    response_model=JobDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> Any:
    job = jobs.get(job_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "job not found",
            detail=f"no job with id {job_id!r}",
            type_="/problems/not-found",
        )
    return job.to_detail()


@router.post(
    "/{job_id}/stop",
    response_model=JobSummary,
    responses={404: {"model": ProblemDetail}},
)
def stop_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager)
) -> Any:
    job = jobs.stop(job_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "job not found",
            detail=f"no job with id {job_id!r}",
            type_="/problems/not-found",
        )
    return job.to_summary()


# ── WS /ws/jobs/{job_id}/events ───────────────────────────────────────
# Registered on the unprefixed `ws_router` so the final path matches
# the run-events convention (`/ws/projects/.../runs/.../events`) and
# passes through the Vite dev proxy unchanged (see 2026-04-21 20:45
# Change Log on the Vite `/ws` proxy-rewrite bug).
@ws_router.websocket("/ws/jobs/{job_id}/events")
async def job_events_ws(
    websocket: WebSocket, job_id: str,
) -> None:
    """Stream a job's events (log_line + terminal). Replays the full
    buffered event list on connect, then tails live emits. Closes with
    a `terminal` marker once the job reaches a terminal state."""
    await _job_events_ws_impl(websocket, job_id)


async def _job_events_ws_impl(websocket: WebSocket, job_id: str) -> None:
    jobs: JobManager = websocket.app.state.job_manager
    await websocket.accept()
    job = jobs.get(job_id)
    if job is None:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "error": f"job {job_id!r} not found",
            })
            await websocket.close(code=4404)
        return

    # Replay: send `connected` snapshot + every buffered event. Events
    # are small dicts (one log line ≈ 100 B); no budget needed unlike
    # runs where the log volume is 10⁵+.
    try:
        await websocket.send_json({
            "type": "connected",
            "status": job.status,
            "total_event_count": len(job.events),
            "replay_count": len(job.events),
        })
        for ev in job.events:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    # Terminal jobs: send the final marker + close. No subscribe needed.
    if job.status in ("completed", "errored", "stopped"):
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "terminal", "status": job.status})
            await websocket.close()
        return

    # Live loop.
    queue = job.subscribe()
    recv_task = asyncio.create_task(_drain_ws_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED,
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
            # Detect terminal via job.status each iteration; we don't
            # know which event kind marks terminal across job kinds.
            if job.status in ("completed", "errored", "stopped"):
                with contextlib.suppress(Exception):
                    await websocket.send_json({
                        "type": "terminal", "status": job.status,
                    })
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task
        with contextlib.suppress(Exception):
            await websocket.close()


async def _drain_ws_client(ws: WebSocket) -> None:
    """Read + discard client frames so a half-closed client triggers
    WebSocketDisconnect instead of the server waiting forever on the
    send side. Mirrors routes/runs.py::_drain_client."""
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        return
