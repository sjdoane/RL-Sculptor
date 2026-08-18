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

import json
import math
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse

from backend.models.project import ProblemDetail
from backend.models.run import PolicySummary
from backend.routes.runs import _find_run, _resolve_run_root
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore

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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _selection_candidates(
    runs_root: Path,
) -> tuple[Optional[int], Optional[str], dict[int, dict[str, Any]]]:
    """Read a canonical keep-best receipt without guessing from recency.

    A partially-written or hand-edited document is not enough to preselect a
    policy: the top-level index/source and matching candidate marker must all
    agree. This is intentionally stricter than merely finding the newest
    checkpoint.
    """
    doc = _load_json_object(runs_root.parent / "reports" / "selection.json")
    selected = doc.get("selected_iter_index")
    source = doc.get("selection_source")
    rows = doc.get("candidates")
    candidates: dict[int, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("iter_index")
            if isinstance(index, int) and not isinstance(index, bool):
                candidates[index] = row
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or not isinstance(source, str)
        or not source.strip()
        or candidates.get(selected, {}).get("selected") is not True
    ):
        return None, None, candidates
    return selected, source.strip(), candidates


def _evidence_value(
    components: dict[str, Any], keys: tuple[str, ...],
) -> Optional[dict[str, Any]]:
    for key in keys:
        value = components.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        if key.endswith("_frac") or key in {"completion_gate"}:
            kind = "fraction"
        elif "frame" in key:
            kind = "frames"
        elif "count" in key:
            kind = "count"
        elif "score" in key or key.startswith("ch_"):
            kind = "score"
        else:
            kind = "value"
        return {"key": key, "value": float(value), "kind": kind}
    return None


def _metric_text(metric: dict[str, Any], key: str) -> Optional[str]:
    value = metric.get(key)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return None
        text = str(value).strip()
        return text or None
    return None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _policy_receipt_fields(
    runs_root: Path,
    row: dict[str, Any],
    *,
    selected_iter: Optional[int],
    selection_source: Optional[str],
    selection_candidates: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    iter_index = int(row["iter_index"])
    iter_dir = runs_root / f"iter_{iter_index}"
    fitness_doc = _load_json_object(iter_dir / "fitness.json")
    components = fitness_doc.get("components")
    if not isinstance(components, dict):
        components = {}
    metric = fitness_doc.get("metric")
    if not isinstance(metric, dict):
        metric = {}

    route = _evidence_value(components, (
        "route_complete_frac", "actual_route_complete_frac",
        "order_ok_frac", "success_seen_frac", "completion_gate",
    ))
    contact = None
    if components.get("contact_evidence_ok") != 0:
        contact = _evidence_value(components, (
            "contact_free_frac", "forbidden_contact_free_frac",
            "contact_frac", "forbidden_contact_count",
        ))
    hold = _evidence_value(components, (
        "strict_hold_frac", "hold_ok_frac", "full_hold_frac", "hold_frac",
        "strict_hold_count", "hold_frames", "proof_frames", "ch_hold",
    ))
    evidence_count = sum(value is not None for value in (route, contact, hold))
    evidence_status = (
        "complete" if evidence_count == 3
        else "partial" if evidence_count > 0
        else "unavailable"
    )

    candidate = selection_candidates.get(iter_index, {})
    criterion = candidate.get("criterion_pass")
    criterion_status = (
        "passed" if criterion is True
        else "failed" if criterion is False
        else "not_recorded"
    )
    return {
        "metric_id": _metric_text(metric, "id"),
        "metric_version": _metric_text(metric, "version"),
        "metric_source": _metric_text(metric, "source"),
        "metric_sha256": _metric_text(metric, "sha256"),
        "criterion_status": criterion_status,
        "evidence_status": evidence_status,
        "route_evidence": route,
        "contact_evidence": contact,
        "hold_evidence": hold,
        "rollout_available": _nonempty_file(
            iter_dir / "rollout" / "rollout.mp4"
        ),
        "selected": selected_iter == iter_index,
        "selection_source": (
            selection_source if selected_iter == iter_index else None
        ),
    }


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
    selected, selection_source, candidates = _selection_candidates(runs_root)
    return [
        PolicySummary(
            **row,
            **_policy_receipt_fields(
                runs_root,
                row,
                selected_iter=selected,
                selection_source=selection_source,
                selection_candidates=candidates,
            ),
        )
        for row in list_exportable_iters(runs_root)
    ]


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
