"""Saved-missions library endpoints (chunk A4).

Missions auto-archive into a durable `saved/` store as they run (see
`sculptor.archive` + the orchestrator hooks landed in A3). This router
exposes that store over HTTP plus a manual "save a copy now" action:

  GET    /saved                                    — list every entry
  GET    /saved/{entry_id}                         — full manifest + mission.json
  GET    /saved/{entry_id}/file/{relpath:path}      — serve an archived file
  DELETE /saved/{entry_id}                          — move the entry to trash
  POST   /projects/{slug}/missions/{mission_slug}/save
                                                     — schedule a `mission_save`
                                                       job (202 + JobDetail)

Disk (`sculptor.archive.list_saved` / `read_manifest`) is the source of
truth for GET/DELETE — saved entries must stay listable after a backend
restart even though JobManager history is in-memory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.models.kg import JobDetail
from backend.models.project import ProblemDetail
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore
from backend.services.saved_jobs import run_mission_save_job
from backend.services import mission_store


router = APIRouter(tags=["saved"])

# Entry ids are minted by sculptor.archive._entry_id as
# "<project_slug>--<mission_slug>--<UTC stamp>". Anchored full-match so
# a traversal-shaped id ("..", "/", "\\") is rejected before any join.
_ENTRY_ID_RE = re.compile(
    r"^[a-z0-9][a-z0-9_-]{0,63}--[a-z0-9][a-z0-9_-]{0,63}--\d{8}T\d{6}Z$"
)


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ────────────────────────────────────────────────────────────
def _problem(code: int, title: str, **extra: Any) -> JSONResponse:
    body = ProblemDetail(title=title, status=code, **{
        k: v for k, v in extra.items() if k in {"detail", "type", "instance"}
    }).model_dump()
    body.update({k: v for k, v in extra.items() if k not in body})
    return JSONResponse(
        status_code=code, content=body,
        media_type="application/problem+json")


def _valid_entry_id(entry_id: str) -> bool:
    return bool(_ENTRY_ID_RE.fullmatch(entry_id))


def _entry_dir(entry_id: str) -> Path:
    from sculptor.archive import saved_root

    return saved_root() / entry_id


def _read_manifest_or_none(entry_dir: Path) -> Optional[dict[str, Any]]:
    from sculptor.archive import read_manifest

    if not entry_dir.is_dir():
        return None
    try:
        return read_manifest(entry_dir)
    except (OSError, ValueError):
        return None


def _slim_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """List-view projection of a full manifest — per-stage status +
    kept-checkpoint COUNT (not the full kept_checkpoints list) and the
    first video relpath across all stages as a thumbnail hint."""
    stages = manifest.get("stages") or []
    stage_rows = []
    first_video: Optional[str] = None
    for s in stages:
        if not isinstance(s, dict):
            continue
        stage_rows.append({
            "name": s.get("name"),
            "status": s.get("status"),
            "best_metric": s.get("best_metric"),
            "n_kept_checkpoints": len(s.get("kept_checkpoints") or []),
        })
        if first_video is None:
            videos = s.get("videos") or []
            if videos:
                first_video = videos[0]
    return {
        "entry_id": manifest.get("entry_id"),
        "project_slug": manifest.get("project_slug"),
        "mission_slug": manifest.get("mission_slug"),
        "goal": manifest.get("goal"),
        "created_at": manifest.get("created_at"),
        "stages": stage_rows,
        "total_bytes": manifest.get("total_bytes"),
        "thumbnail_video": first_video,
    }


# ── GET /saved ─────────────────────────────────────────────────────────
@router.get("/saved")
def list_saved_entries() -> Any:
    from sculptor.archive import list_saved, saved_root

    manifests = list_saved(saved_root())
    # sculptor.archive.list_saved already sorts newest-first by
    # created_at; keep that ordering rather than re-deriving it here.
    return [_slim_row(m) for m in manifests]


# ── GET /saved/{entry_id} ────────────────────────────────────────────
@router.get(
    "/saved/{entry_id}",
    responses={404: {"model": ProblemDetail}},
)
def get_saved_entry(entry_id: str) -> Any:
    if not _valid_entry_id(entry_id):
        return _problem(
            404, "saved entry not found",
            detail=f"malformed entry_id {entry_id!r}",
            type="/problems/not-found")
    entry_dir = _entry_dir(entry_id)
    manifest = _read_manifest_or_none(entry_dir)
    if manifest is None:
        return _problem(
            404, "saved entry not found",
            detail=f"no saved entry {entry_id!r}",
            type="/problems/not-found")

    mission_json: Optional[dict[str, Any]] = None
    mission_json_path = entry_dir / "mission.json"
    if mission_json_path.is_file():
        import json as _json

        try:
            mission_json = _json.loads(
                mission_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mission_json = None

    return {**manifest, "mission_json": mission_json}


# ── GET /saved/{entry_id}/file/{relpath} ─────────────────────────────
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".md": "text/markdown",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".toml": "text/plain",
    ".yml": "text/plain",
    ".yaml": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".npz": "application/octet-stream",
}


def _media_type_for(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


@router.get(
    "/saved/{entry_id}/file/{relpath:path}",
    response_class=FileResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_saved_file(entry_id: str, relpath: str) -> Any:
    if not _valid_entry_id(entry_id):
        return _problem(
            404, "saved entry not found",
            detail=f"malformed entry_id {entry_id!r}",
            type="/problems/not-found")
    entry_dir = _entry_dir(entry_id)
    if not entry_dir.is_dir():
        return _problem(
            404, "saved entry not found",
            detail=f"no saved entry {entry_id!r}",
            type="/problems/not-found")

    candidate = entry_dir / relpath
    try:
        resolved = candidate.resolve()
        resolved.relative_to(entry_dir.resolve())
    except (ValueError, OSError):
        return _problem(
            404, "file not found",
            detail=f"{relpath!r} escapes the saved entry",
            type="/problems/not-found")
    if not resolved.is_file():
        return _problem(
            404, "file not found",
            detail=f"no file at {relpath!r} in saved entry {entry_id!r}",
            type="/problems/not-found")

    return FileResponse(resolved, media_type=_media_type_for(resolved))


# ── DELETE /saved/{entry_id} ─────────────────────────────────────────
@router.delete(
    "/saved/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    responses={404: {"model": ProblemDetail}},
)
def delete_saved_entry(entry_id: str) -> Any:
    from fastapi import Response

    if not _valid_entry_id(entry_id):
        return _problem(
            404, "saved entry not found",
            detail=f"malformed entry_id {entry_id!r}",
            type="/problems/not-found")
    entry_dir = _entry_dir(entry_id)
    if not entry_dir.is_dir():
        return _problem(
            404, "saved entry not found",
            detail=f"no saved entry {entry_id!r}",
            type="/problems/not-found")

    manifest = _read_manifest_or_none(entry_dir)
    mission_slug = (
        manifest.get("mission_slug") if manifest else None
    ) or entry_id

    from backend.services import trash

    # Consistency with project/mission delete: even deleting a saved
    # mission is recoverable via /trash rather than a hard rmtree.
    trash.move_to_trash(
        entry_dir, kind="saved", slug=str(mission_slug), origin=entry_dir,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── POST /projects/{slug}/missions/{mission_slug}/save ───────────────
class SaveMissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # stage name -> list of iter indices to pin a checkpoint for,
    # regardless of final/best status. `None` / omitted = no pins.
    pinned: Optional[dict[str, list[int]]] = None


@router.post(
    "/projects/{slug}/missions/{mission_slug}/save",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"model": ProblemDetail}},
)
def save_mission(
    slug: str,
    mission_slug: str,
    body: Optional[SaveMissionRequest] = Body(default=None),
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            404, "project not found",
            detail=f"no project with slug {slug!r}",
            type="/problems/not-found")
    project_dir = Path(detail.project_dir)

    if mission_slug not in mission_store.list_mission_slugs(project_dir):
        return _problem(
            404, "mission not found",
            detail=f"no mission {mission_slug!r} under project {slug!r}",
            type="/problems/not-found")

    pinned = (body.pinned if body is not None else None) or {}
    pinned_sets = {k: set(v) for k, v in pinned.items()}

    job = jobs.submit(
        kind="mission_save",
        project_slug=slug,
        fn=run_mission_save_job(
            project_dir=project_dir,
            project_slug=slug,
            mission_slug=mission_slug,
            pinned=pinned_sets,
        ),
        params={"mission_slug": mission_slug, "pinned": pinned},
    )
    return job.to_detail()
