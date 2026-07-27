"""Trash endpoints — chunk A1 non-destructive delete.

  GET    /trash                    → list every trash entry (newest first)
  POST   /trash/{entry_id}/restore → move an entry back to origin_path
  DELETE /trash/{entry_id}         → permanent purge (requires ?confirm=)

`DELETE /projects/{slug}` and the mission-delete endpoint move into the
trash rather than hard-deleting (see `backend.services.trash`); this
router is the only place a trashed entry can be permanently destroyed.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import JSONResponse

from backend.models.project import ProblemDetail
from backend.models.trash import RestoreTrashResponse, TrashEntry
from backend.services import trash


router = APIRouter(prefix="/trash", tags=["trash"])


# ── helpers ────────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: str | None = None,
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


# ── endpoints ──────────────────────────────────────────────────────────
@router.get("", response_model=list[TrashEntry])
def list_trash() -> list[TrashEntry]:
    return [TrashEntry(**row) for row in trash.list_trash()]


@router.post(
    "/{entry_id}/restore",
    response_model=RestoreTrashResponse,
    responses={
        400: {"model": ProblemDetail, "description": "malformed entry_id"},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail, "description": "origin no longer exists"},
    },
)
def restore_trash_entry(entry_id: str) -> Any:
    try:
        restored_path = trash.restore(entry_id)
    except trash.InvalidEntryIdError as e:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "malformed trash entry_id",
            detail=str(e),
            type_="/problems/validation-error",
        )
    except trash.RestoreConflictError as e:
        return _problem(
            status.HTTP_409_CONFLICT,
            "restore target is gone",
            detail=str(e),
            type_="/problems/state-conflict",
        )
    except trash.EntryNotFoundError as e:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "trash entry not found",
            detail=str(e),
            type_="/problems/not-found",
        )
    return RestoreTrashResponse(
        entry_id=entry_id, restored_path=str(restored_path),
    )


@router.delete(
    "/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    responses={
        400: {"model": ProblemDetail, "description": "missing/mismatched confirm"},
        404: {"model": ProblemDetail},
    },
)
def purge_trash_entry(
    entry_id: str,
    confirm: Optional[str] = Query(
        default=None,
        description=(
            "Must equal `entry_id` — echo-confirmation guard against "
            "an automated or mistaken DELETE permanently destroying "
            "the wrong entry."
        ),
    ),
) -> Any:
    if confirm != entry_id:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "confirm mismatch",
            detail=(
                "purge requires `?confirm=<entry_id>` to equal the "
                "entry_id being purged — this permanently destroys "
                "the trashed data, there is no further undo."
            ),
            type_="/problems/validation-error",
        )
    try:
        trash.purge(entry_id)
    except trash.InvalidEntryIdError as e:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "malformed trash entry_id",
            detail=str(e),
            type_="/problems/validation-error",
        )
    except trash.EntryNotFoundError as e:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "trash entry not found",
            detail=str(e),
            type_="/problems/not-found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
