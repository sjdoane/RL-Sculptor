"""Production status/receipt endpoints for per-mode objective evidence."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse

from backend.models.project import ProblemDetail
from backend.services import mode_evidence
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["mode-evidence"])


def _problem(code: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ProblemDetail(
            type="/problems/mode-evidence-unavailable",
            title=title,
            status=code,
            detail=detail,
        ).model_dump(),
        media_type="application/problem+json",
    )


def _project_dir(store: ProjectStore, slug: str):
    detail = store.get(slug)
    return None if detail is None else detail.project_dir


@router.get(
    "/projects/{slug}/mode-evidence",
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def get_mode_evidence(
    slug: str,
    request: Request,
    clip_id: str = Query(..., min_length=1),
    robot: str = Query(..., min_length=1),
) -> Any:
    project_dir = _project_dir(request.app.state.project_store, slug)
    if project_dir is None:
        return _problem(status.HTTP_404_NOT_FOUND, "project not found", slug)
    try:
        return mode_evidence.status(
            project_dir, expected_clip_id=clip_id, expected_robot=robot
        )
    except mode_evidence.ModeEvidenceError as exc:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mode evidence context is not current",
            str(exc),
        )


@router.post(
    "/projects/{slug}/mode-evidence/receipt",
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def record_mode_evidence_receipt(
    slug: str,
    request: Request,
    clip_id: str = Query(..., min_length=1),
    robot: str = Query(..., min_length=1),
) -> Any:
    project_dir = _project_dir(request.app.state.project_store, slug)
    if project_dir is None:
        return _problem(status.HTTP_404_NOT_FOUND, "project not found", slug)
    try:
        receipt = mode_evidence.build_readiness_receipt(
            project_dir, expected_clip_id=clip_id, expected_robot=robot
        )
        path = mode_evidence.persist_readiness_receipt(project_dir, receipt)
    except mode_evidence.ModeEvidenceError as exc:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mode evidence context is not current",
            str(exc),
        )
    return dict(receipt, recorded=True, receipt_path=str(path))
