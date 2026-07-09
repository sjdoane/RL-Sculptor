"""Reference-library endpoints (§R1_BUILD_SPEC decision 11).

Top-level disk-truth library — same precedent as `routes/saved.py`
(regex id guard, slim-row listing, FileResponse with resolve/
relative_to traversal guard, media-type map) — plus one mission-scoped
attach/detach pair.

  GET    /references?robot=&q=&k=&llm=            — search or listing
  GET    /references/{clip_id}                     — detail (provenance
                                                       + index row)
  GET    /references/{clip_id}/preview             — preview.png
  GET    /references/{clip_id}/file/clip.npz       — clip download
  POST   /projects/{slug}/missions/{ms}/stages/{stage}/reference
                                                     — attach a clip_id
                                                       to a stage
  DELETE /projects/{slug}/missions/{ms}/stages/{stage}/reference
                                                     — clear it

v1 is g1-only (§decision 11: "no robot path segment in v1 API" — the
`robot` query param on GET /references is a FILTER, not a path
segment; robot for the single-clip routes is resolved by looking the
clip_id up in the index).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.models.project import ProblemDetail
from backend.services import mission_store
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["references"])

# §decision 4/11: clip_id charset — lowercase alnum + `_-`, must start
# alnum, max 96 chars. Anchored full-match so a traversal-shaped id
# ("..", "/", "\\") is rejected before any join. Mirrors
# `sculptor.refs.library.CLIP_ID_RE` verbatim (re-declared here so the
# route layer doesn't have to import sculptor at module scope just to
# validate a path segment).
_CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")

# Same plain-slug-component guard `routes/missions.py` applies to
# mission_slug / stage before they hit the filesystem.
_SAFE_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


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


def _valid_clip_id(clip_id: str) -> bool:
    return bool(_CLIP_ID_RE.fullmatch(clip_id))


def _find_index_row(clip_id: str) -> Optional[dict[str, Any]]:
    """Look up one clip's slim index row by id (v1 = g1 only, but this
    scans every robot dir the index happens to carry rather than
    hard-coding "g1" — a future multi-robot library needs no route
    change)."""
    from sculptor.refs import library

    for row in library.read_index():
        if row.get("clip_id") == clip_id:
            return row
    return None


def _clip_dir_for(clip_id: str, robot: str) -> Path:
    from sculptor.refs import library

    return library.clip_dir(robot, clip_id)


# ── GET /references ──────────────────────────────────────────────────
@router.get("/references")
def list_or_search_references(
    robot: str = "g1",
    q: Optional[str] = Query(default=None),
    k: int = Query(default=10, ge=1, le=100),
    llm: int = Query(default=0),
) -> Any:
    """`q` present → deterministic-by-default search via
    `sculptor.refs.retrieve.search` (§decision 11: `llm=0/1` query
    param controls `use_llm`, default OFF for the UI's as-you-type
    path — an LLM rerank on every keystroke would be slow and costly).
    `q` absent → the full slim index listing, filtered to `robot`."""
    from sculptor.refs import library, retrieve

    if q is not None and q.strip():
        matches = retrieve.search(q, robot=robot, k=k, use_llm=bool(llm))
        return [
            {
                "clip_id": m.clip_id,
                "text": m.text,
                "score": m.score,
                "match_confidence": m.match_confidence,
                "reason": m.reason,
                "tier": m.tier,
                "license": m.license,
                "n_frames": m.n_frames,
                "fps": m.fps,
                "duration_s": m.duration_s,
                "rerank": m.rerank,
            }
            for m in matches
        ]

    rows = [r for r in library.read_index() if r.get("robot") == robot]
    return rows[:k] if k else rows


# ── GET /references/{clip_id} ────────────────────────────────────────
@router.get(
    "/references/{clip_id}",
    responses={404: {"model": ProblemDetail}},
)
def get_reference_detail(clip_id: str) -> Any:
    if not _valid_clip_id(clip_id):
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"malformed clip_id {clip_id!r}",
            type_="/problems/not-found")

    row = _find_index_row(clip_id)
    if row is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"no reference clip {clip_id!r}",
            type_="/problems/not-found")

    from sculptor.refs import library

    provenance: Optional[dict[str, Any]] = None
    try:
        provenance = library.read_provenance(row["robot"], clip_id)
    except (OSError, ValueError):
        provenance = None

    return {"index_row": row, "provenance": provenance}


# ── GET /references/{clip_id}/preview ────────────────────────────────
@router.get(
    "/references/{clip_id}/preview",
    response_class=FileResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_reference_preview(clip_id: str) -> Any:
    from sculptor.refs import library

    if not _valid_clip_id(clip_id):
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"malformed clip_id {clip_id!r}",
            type_="/problems/not-found")

    row = _find_index_row(clip_id)
    if row is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"no reference clip {clip_id!r}",
            type_="/problems/not-found")

    preview_path = _clip_dir_for(clip_id, row["robot"]) / library.PREVIEW_FILENAME
    if not preview_path.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND, "preview not found",
            detail=f"no preview.png for reference clip {clip_id!r}",
            type_="/problems/not-found")

    return FileResponse(preview_path, media_type="image/png")


# ── GET /references/{clip_id}/file/clip.npz ──────────────────────────
@router.get(
    "/references/{clip_id}/file/clip.npz",
    response_class=FileResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_reference_clip_file(clip_id: str) -> Any:
    from sculptor.refs import library

    if not _valid_clip_id(clip_id):
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"malformed clip_id {clip_id!r}",
            type_="/problems/not-found")

    row = _find_index_row(clip_id)
    if row is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"no reference clip {clip_id!r}",
            type_="/problems/not-found")

    clip_dir = _clip_dir_for(clip_id, row["robot"])
    candidate = clip_dir / library.CLIP_FILENAME
    try:
        resolved = candidate.resolve()
        resolved.relative_to(clip_dir.resolve())
    except (ValueError, OSError):
        return _problem(
            status.HTTP_404_NOT_FOUND, "file not found",
            detail=f"clip.npz escapes the reference clip dir for {clip_id!r}",
            type_="/problems/not-found")
    if not resolved.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND, "file not found",
            detail=f"no clip.npz for reference clip {clip_id!r}",
            type_="/problems/not-found")

    return FileResponse(resolved, media_type="application/octet-stream")


# ── POST/DELETE /projects/{slug}/missions/{mission_slug}/stages/{stage}/reference
class AttachReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str


def _project_dir(store: ProjectStore, slug: str) -> Optional[Path]:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir)


def _validate_mission_and_stage(
    project_dir: Path, slug: str, mission_slug: str, stage: str,
) -> Any:
    """Returns None on success, else a `_problem(...)` JSONResponse to
    return immediately. Validates against mission.json's stage list —
    NOT the on-disk stages/<stage>/ training dir (§decision 11 /
    commit 8b0bfa3: a pending stage that hasn't trained yet has no
    training dir but is exactly the kind of stage a user wants to
    attach a reference to before it ever runs)."""
    if (not _SAFE_PATH_SEGMENT.match(mission_slug)
            or not _SAFE_PATH_SEGMENT.match(stage)):
        return _problem(
            status.HTTP_404_NOT_FOUND, "invalid path segment",
            detail=(
                f"mission_slug={mission_slug!r} / stage={stage!r} must "
                "each be a plain slug component"
            ),
            type_="/problems/not-found")
    if mission_slug not in mission_store.list_mission_slugs(project_dir):
        return _problem(
            status.HTTP_404_NOT_FOUND, "mission not found",
            detail=f"no mission {mission_slug!r} under project {slug!r}",
            type_="/problems/not-found")

    from sculptor.mission import load_mission

    mission = load_mission(mission_store.mission_dir(project_dir, mission_slug))
    stage_names = {s.name for s in mission.stages}
    if stage not in stage_names:
        return _problem(
            status.HTTP_404_NOT_FOUND, "stage not found",
            detail=(
                f"no stage {stage!r} in mission {mission_slug!r} "
                f"(project {slug!r})"
            ),
            type_="/problems/not-found")
    return None


def _active_job_conflict(
    jobs: JobManager, slug: str, mission_slug: str,
) -> Any:
    """409 while ANY mission-scoped job (decompose/execute/stage-run/
    metric-regen) is live for this mission — same breadth as the
    regenerate-metric endpoint's guard (§decision 11: "same guard as
    the regenerate-metric endpoint"). Attaching/detaching a reference
    mutates mission.json via the same non-atomic save_mission path
    those jobs also write through."""
    job = jobs.active_mission_scoped_job(slug, mission_slug)
    if job is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                f"mission {mission_slug!r} has an active {job.kind!r} "
                f"job; wait for it to finish before changing this "
                f"stage's reference."
            ),
            type_="/problems/state-conflict")
    return None


@router.post(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/reference",
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def attach_stage_reference(
    slug: str,
    mission_slug: str,
    stage: str,
    body: AttachReferenceRequest,
    request: Request,
) -> Any:
    store: ProjectStore = request.app.state.project_store
    jobs: JobManager = request.app.state.job_manager

    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    if not _valid_clip_id(body.clip_id):
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"malformed clip_id {body.clip_id!r}",
            type_="/problems/not-found")

    row = _find_index_row(body.clip_id)
    if row is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"no reference clip {body.clip_id!r}",
            type_="/problems/not-found")

    err = _validate_mission_and_stage(project_dir, slug, mission_slug, stage)
    if err is not None:
        return err

    err = _active_job_conflict(jobs, slug, mission_slug)
    if err is not None:
        return err

    from sculptor.mission import load_mission, save_mission

    md = mission_store.mission_dir(project_dir, mission_slug)
    mission = load_mission(md)
    for s in mission.stages:
        if s.name == stage:
            s.reference_clip_id = body.clip_id
            s.reference_tier = row.get("tier")
            # §decision 11: "match_confidence (null for manual attach)"
            # — this endpoint is a direct attach-by-clip_id, never fed a
            # search-result's match_confidence, so it is always None.
            s.reference_match_confidence = None
            break
    save_mission(mission, md)

    return {
        "stage": stage,
        "reference_clip_id": body.clip_id,
        "reference_tier": row.get("tier"),
        "reference_match_confidence": None,
    }


@router.delete(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/reference",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def detach_stage_reference(
    slug: str,
    mission_slug: str,
    stage: str,
    request: Request,
) -> Any:
    from fastapi import Response

    store: ProjectStore = request.app.state.project_store
    jobs: JobManager = request.app.state.job_manager

    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    err = _validate_mission_and_stage(project_dir, slug, mission_slug, stage)
    if err is not None:
        return err

    err = _active_job_conflict(jobs, slug, mission_slug)
    if err is not None:
        return err

    from sculptor.mission import load_mission, save_mission

    md = mission_store.mission_dir(project_dir, mission_slug)
    mission = load_mission(md)
    for s in mission.stages:
        if s.name == stage:
            s.reference_clip_id = None
            s.reference_tier = None
            s.reference_match_confidence = None
            break
    save_mission(mission, md)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
