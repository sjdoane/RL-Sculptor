"""§Ship 35: auto-generated objective-metric endpoints.

  GET  /projects/{slug}/metrics                 — list generated metrics
  POST /projects/{slug}/metrics/generate        — generate (LLM; ~1-2 min)
  POST /projects/{slug}/metrics/{mid}/calibrate — earn steer-rights

Generation is an LLM round-trip; it runs in a threadpool so the event
loop isn't blocked. Synchronous (no job kind) — a one-off action with a
loading state in the UI.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

_GEN_ID_RE = re.compile(r"^gen_\d+$")

from fastapi import APIRouter, Depends, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from backend.models.metric import GenerateMetricRequest, MetricSummary
from backend.services import metric_store, sculptor_bridge
from backend.services.project_store import ProjectStore

router = APIRouter(tags=["metrics"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def _problem(code: int, title: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"type": "about:blank", "title": title, "status": code,
                 "detail": detail},
    )


def _robot_hint(project_dir: Path) -> Optional[str]:
    """Best-effort task_id from the project's config.toml (a hint for the
    generator); never fatal."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        cfg = tomllib.loads((project_dir / "config.toml").read_text(encoding="utf-8"))
        return ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
    except Exception:  # noqa: BLE001
        return None


@router.get("/projects/{slug}/metrics", response_model=list[MetricSummary])
async def list_project_metrics(slug: str, store: ProjectStore = Depends(get_store)):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    return metric_store.list_metrics(Path(detail.project_dir))


@router.post("/projects/{slug}/metrics/generate", response_model=MetricSummary)
async def generate_project_metric(
    slug: str, body: GenerateMetricRequest,
    store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    if not sculptor_bridge.sculptor_ok():
        return _problem(status.HTTP_503_SERVICE_UNAVAILABLE,
                        "Sculptor unavailable", sculptor_bridge.sculptor_error())
    project_dir = Path(detail.project_dir)
    # §Ship 40: stream pipeline progress to a sidecar the UI polls (this is a
    # 1-2 min, multi-LLM-call op — the user should see it advancing).
    metric_store.write_progress(project_dir, {
        "active": True, "stage": "starting", "message": "Starting generation…"})

    def _on_event(ev: dict) -> None:
        metric_store.write_progress(project_dir, {"active": True, **ev})

    try:
        rec = await run_in_threadpool(
            metric_store.generate, project_dir, body.behavior_goal,
            robot_hint=_robot_hint(project_dir), review=body.review,
            on_event=_on_event)
        if body.calibrate_against and rec.get("accepted"):
            try:
                metric_store.write_progress(project_dir, {
                    "active": True, "stage": "calibrating",
                    "message": f"Calibrating vs {body.calibrate_against}…"})
                rec = await run_in_threadpool(
                    metric_store.calibrate, project_dir, rec["id"],
                    body.calibrate_against)
            except Exception:  # noqa: BLE001 — calibration failure ≠ generation failure
                pass
    finally:
        metric_store.clear_progress(project_dir)
    return rec


@router.get("/projects/{slug}/metrics/generate/progress")
async def metric_generate_progress(
    slug: str, store: ProjectStore = Depends(get_store),
):
    """§Ship 40: current generation progress for the project (polled by the UI
    while a generate is in flight). `{active: false}` when idle."""
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    return metric_store.read_progress(Path(detail.project_dir))


@router.post("/projects/{slug}/metrics/{mid}/calibrate", response_model=MetricSummary)
async def calibrate_project_metric(
    slug: str, mid: str, against: str,
    store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    # §Ship 35 review: validate the id (path-traversal guard) and the
    # ground-truth metric name before touching the filesystem.
    if not _GEN_ID_RE.match(mid):
        return _problem(status.HTTP_400_BAD_REQUEST, "Invalid metric id")
    if against not in sculptor_bridge.builtin_spec_metric_names():
        return _problem(status.HTTP_400_BAD_REQUEST, "Unknown built-in metric",
                        f"valid: {sculptor_bridge.builtin_spec_metric_names()}")
    try:
        return await run_in_threadpool(
            metric_store.calibrate, Path(detail.project_dir), mid, against)
    except FileNotFoundError as e:
        return _problem(status.HTTP_404_NOT_FOUND, "Metric not found", str(e))
    except KeyError as e:
        return _problem(status.HTTP_400_BAD_REQUEST, "Unknown built-in metric", str(e))
