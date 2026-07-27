"""Trained-policy endpoints — listing + deployment-bundle export.

  GET /projects/{slug}/policies                      — exportable iterations
  GET /projects/{slug}/policies/{iter_index}/export  — build + download zip

Disk is the source of truth (runs/iter_*/checkpoint.*), so policies stay
listable and exportable after a backend restart even though JobManager
runs are in-memory. An optional `run_id` query scopes both endpoints to a
mission stage's own runs tree (.missions/<m>/stages/<s>/runs) via the same
resolution the runs router uses.

Bundle building is synchronous — a bundle is a zip of small files plus an
ONNX/TorchScript trace of a ~500k-param MLP, a few seconds at worst
(same trade-off as POST /reports/build).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse

from backend.models.project import ProblemDetail
from backend.models.run import PolicySummary
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore
from backend.routes.runs import _find_run, _resolve_run_root


router = APIRouter(tags=["policies"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _problem(code: int, title: str, **extra: Any) -> JSONResponse:
    body = ProblemDetail(title=title, status=code, **{
        k: v for k, v in extra.items() if k in {"detail", "type", "instance"}
    }).model_dump()
    body.update({k: v for k, v in extra.items() if k not in body})
    return JSONResponse(
        status_code=code, content=body,
        media_type="application/problem+json")


def _resolve_roots(
    slug: str,
    run_id: Optional[str],
    store: ProjectStore,
    jobs: JobManager,
) -> tuple[Optional[Path], Optional[Path], Optional[JSONResponse]]:
    """(project_dir, runs_root, error). Mirrors the runs router's
    resolution so stage runs export from their own runs tree."""
    detail = store.get(slug)
    if detail is None:
        return None, None, _problem(
            404, "project not found",
            detail=f"no project with slug {slug!r}",
            type="/problems/not-found")
    project_dir = Path(detail.project_dir)
    if run_id is None:
        return project_dir, project_dir / "runs", None
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return None, None, _problem(
            404, "run not found",
            detail=f"no run with id {run_id!r} in project {slug!r} "
                   "(runs are in-memory; after a backend restart omit "
                   "run_id to export from the project's runs on disk)",
            type="/problems/not-found")
    return project_dir, _resolve_run_root(job, project_dir), None


# ── GET /projects/{slug}/policies ─────────────────────────────────────
@router.get(
    "/projects/{slug}/policies",
    response_model=list[PolicySummary],
    responses={404: {"model": ProblemDetail}, 503: {"model": ProblemDetail}},
)
def list_policies(
    slug: str,
    run_id: Optional[str] = None,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    _, runs_root, err = _resolve_roots(slug, run_id, store, jobs)
    if err is not None:
        return err
    try:
        from sculptor.export import list_exportable_iters
    except Exception as e:  # noqa: BLE001
        return _problem(
            503, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable")
    return [PolicySummary(**row) for row in list_exportable_iters(runs_root)]


# ── GET /projects/{slug}/policies/{iter_index}/export ─────────────────
@router.get(
    "/projects/{slug}/policies/{iter_index}/export",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/zip": {}}},
        404: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def export_policy(
    slug: str,
    iter_index: int,
    run_id: Optional[str] = None,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    project_dir, runs_root, err = _resolve_roots(slug, run_id, store, jobs)
    if err is not None:
        return err
    if iter_index < 0:
        return _problem(
            404, "iteration not found",
            detail="iter_index must be >= 0", type="/problems/not-found")
    try:
        from sculptor.export import ExportError, export_policy_bundle
    except Exception as e:  # noqa: BLE001
        return _problem(
            503, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable")

    try:
        result = export_policy_bundle(
            project_dir, iter_index=iter_index, runs_root=runs_root)
    except ExportError as e:
        return _problem(
            404, "no exportable checkpoint",
            detail=str(e), type="/problems/not-found")
    except Exception as e:  # noqa: BLE001 — degrade loudly, never 500-blank
        return _problem(
            500, "export failed",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/export-failed")

    return FileResponse(
        result.bundle_path,
        media_type="application/zip",
        filename=result.bundle_path.name,
    )
