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

v1 has "no robot path segment in v1 API" (§decision 11) — the `robot`
query param on GET /references is a FILTER (default "g1", the only
robot v1 originally shipped with), not a path segment; robot for the
single-clip routes is resolved by looking the clip_id up in the index,
never hardcoded. Neither of those is actually g1-specific: `robot` is a
real filter over whatever robots `sculptor.refs.library`'s index
carries (verified against a t1 clip — §Problem 2, 2026-07-11), so a
second robot (e.g. t1, populated via `sculptor.refs.retarget`'s GMR
pipeline) is already listable/searchable/attachable today through this
same v1 surface, just by passing `?robot=t1` — no route change needed.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.models.kg import JobSummary
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
    k: int = Query(default=10, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    llm: int = Query(default=0),
) -> Any:
    """`q` present → deterministic-by-default search via
    `sculptor.refs.retrieve.search` (§decision 11: `llm=0/1` query
    param controls `use_llm`, default OFF for the UI's as-you-type
    path — an LLM rerank on every keystroke would be slow and costly).
    `q` absent → the slim index listing, filtered to `robot`.

    Still list-shaped, because the typeahead callers depend on that. Use
    `GET /references/browse` for the paginated, faceted library view — this
    endpoint's `k=10` default silently made a ~6000-clip library look like
    ten alphabetically-first clips, and a freshly composed motion unfindable
    except by typing its id back into a semantic search box."""
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
    if offset:
        rows = rows[offset:]
    return rows[:k] if k else rows


@router.get("/references/browse")
def browse_references(
    robot: str = "g1",
    q: Optional[str] = Query(default=None),
    label: Optional[str] = Query(default=None),
    tier: Optional[str] = Query(default=None),
    composed: Optional[bool] = Query(default=None),
    min_duration_s: Optional[float] = Query(default=None),
    max_duration_s: Optional[float] = Query(default=None),
    sort: str = Query(default="recent"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Any:
    """The library, paginated, with the facet counts needed to navigate it.

    `q` here is a plain substring filter over id, text and labels —
    deliberately NOT the embedding search. Browsing wants "show me everything
    matching, in a stable order, with a total", and semantic retrieval gives
    neither a total nor a stable page boundary. `/references?q=` remains the
    semantic path, and the two are complementary: search when you know what
    the motion looks like, browse when you want to see what exists.
    """
    from sculptor.refs import library

    #: The slim index is an append-only JSONL, so file order IS registration
    #: order. That is the only recency signal there is — no row carries a
    #: timestamp — and it is exactly what "show me the clip I just composed"
    #: needs.
    all_rows = library.read_index()
    rows = [dict(r, _seq=i) for i, r in enumerate(all_rows)
            if r.get("robot") == robot]

    def _is_composed(r: dict) -> bool:
        return "composed" in (r.get("labels") or [])

    facets = {
        "tiers": _counts(rows, lambda r: str(r.get("tier") or "?")),
        "labels": _label_counts(rows),
        "composed": sum(1 for r in rows if _is_composed(r)),
        "total": len(rows),
    }

    if q and q.strip():
        # AND over whitespace tokens against a normalized haystack. A plain
        # substring match found nothing for "balance beam", because the ids
        # are `balance_on_beam03_poses_100_jpos` — the separator, not the
        # words, was the mismatch.
        tokens = [t for t in _normalize(q).split() if t]
        rows = [r for r in rows if all(t in _haystack(r) for t in tokens)]
    if label:
        rows = [r for r in rows if label in (r.get("labels") or [])]
    if tier:
        rows = [r for r in rows if str(r.get("tier") or "") == tier]
    if composed is not None:
        rows = [r for r in rows if _is_composed(r) is composed]
    if min_duration_s is not None:
        rows = [r for r in rows
                if float(r.get("duration_s") or 0) >= min_duration_s]
    if max_duration_s is not None:
        rows = [r for r in rows
                if float(r.get("duration_s") or 0) <= max_duration_s]

    if sort == "duration":
        rows.sort(key=lambda r: float(r.get("duration_s") or 0), reverse=True)
    elif sort == "name":
        rows.sort(key=lambda r: str(r.get("clip_id", "")))
    else:
        # "recent" puts composites first, then newest registration. The bulk
        # corpus was appended after the composites, so a pure recency sort
        # buries the clip you just made under 6000 corpus rows — which is the
        # single thing this sort exists to surface.
        rows.sort(key=lambda r: (_is_composed(r), r["_seq"]), reverse=True)

    total = len(rows)
    page = []
    for r in rows[offset:offset + limit]:
        r = dict(r)
        r.pop("_seq", None)
        r["composed"] = _is_composed(r)
        page.append(r)
    return {
        "rows": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "robot": robot,
        "facets": facets,
    }


def _project_reference_robot(project_dir: Path) -> str:
    """The library namespace this project's clips must come from.

    Mirrors `sculptor.sculpt._stage_reference_robot_slug`, which is what
    actually resolves the clip at training time — deliberately the same
    task_id-shaped string ("Mjlab-Velocity-Flat-Unitree-G1"). Returns "" when
    it cannot tell, so an unknown adapter never blocks an attach.
    """
    try:
        from sculptor.sculpt import _STAGE_REFERENCE_ROBOT_SLUGS
    except Exception:  # noqa: BLE001 — never block an attach on an import
        return ""
    config_path = project_dir / "config.toml"
    if not config_path.is_file():
        return ""
    try:
        import tomllib
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
        task_id = str(
            ((cfg.get("adapter") or {}).get("config") or {})
            .get("task_id", "") or "")
    except Exception:  # noqa: BLE001 — best-effort resolution only
        return ""
    hint = task_id.lower()
    for candidate in _STAGE_REFERENCE_ROBOT_SLUGS:
        if candidate in hint:
            return candidate
    return ""


def _normalize(s: str) -> str:
    """Lowercase, and treat `_`/`-`/`.` as word breaks."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower())


def _haystack(row: dict) -> str:
    parts = [row.get("clip_id", ""), row.get("text", "")]
    parts.extend(str(x) for x in (row.get("labels") or []))
    return _normalize(" ".join(str(p) for p in parts))


def _counts(rows: list[dict], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[key(r)] = out.get(key(r), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _label_counts(rows: list[dict], *, top: int = 24) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for lab in (r.get("labels") or []):
            out[str(lab)] = out.get(str(lab), 0) + 1
    ranked = sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(ranked[:top])


# ── POST /references/compose ─────────────────────────────────────────
class ComposeSegment(BaseModel):
    """One span of one already-registered library clip."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    t_start_s: Optional[float] = None
    t_end_s: Optional[float] = None
    label: Optional[str] = None


class ComposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    robot: str = "g1"
    segments: list[ComposeSegment]
    text: str = ""
    labels: list[str] = []
    blend_s: float = 0.20
    target_fps: Optional[float] = None
    # `strict=False` still MEASURES every seam — it only declines to refuse.
    # Exposed because a marginal composite is sometimes worth eyeballing in
    # the preview before deciding, and the seam report says how bad it is.
    strict: bool = True


@router.post(
    "/references/compose",
    responses={400: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
def compose_reference_clip(body: ComposeRequest) -> Any:
    """Compose spans of several solved clips into one novel candidate clip.

    This is the "solve for a motion nobody recorded" path: the goal's phases
    each exist in some clip, just never in the same one. The result registers
    at tier K and is explicitly NOT certified — `sculptor.refs.track` is what
    promotes it — so the response returns the seam report for the caller to
    judge before spending a tracking run.
    """
    from sculptor.refs import library
    from sculptor.refs.compose import ComposeError, compose_and_register

    if not _valid_clip_id(body.clip_id):
        return _problem(
            status.HTTP_400_BAD_REQUEST, "invalid clip_id",
            detail=f"{body.clip_id!r} must match {_CLIP_ID_RE.pattern}")
    if len(body.segments) < 2:
        return _problem(
            status.HTTP_400_BAD_REQUEST, "need at least 2 segments",
            detail="a single span is a crop, not a composition")
    for seg in body.segments:
        if not _valid_clip_id(seg.clip_id):
            return _problem(
                status.HTTP_400_BAD_REQUEST, "invalid source clip_id",
                detail=f"{seg.clip_id!r} must match {_CLIP_ID_RE.pattern}")
    if _find_index_row(body.clip_id) is not None:
        return _problem(
            status.HTTP_409_CONFLICT, "clip_id already exists",
            detail=f"{body.clip_id!r} is already in the library")

    try:
        composed = compose_and_register(
            body.robot,
            [seg.model_dump(exclude_none=True) for seg in body.segments],
            clip_id=body.clip_id,
            text=body.text,
            labels=body.labels,
            blend_s=body.blend_s,
            target_fps=body.target_fps,
            strict=body.strict,
        )
    except ComposeError as exc:
        # Every ComposeError is a caller-fixable statement about the spans
        # (they do not meet, the joint sets differ, a source is missing), so
        # it is a 400 carrying the actual measurement, not a 500.
        return _problem(
            status.HTTP_400_BAD_REQUEST, "cannot compose these segments",
            detail=str(exc))

    library.rebuild_index()
    prov = composed.provenance
    return {
        "clip_id": composed.clip_id,
        "robot": composed.robot,
        "tier": prov.get("tier"),
        "certified": False,
        "license": prov.get("license"),
        "attribution": prov.get("attribution"),
        "parent_clip_ids": (prov.get("source") or {}).get("parent_clip_ids", []),
        "qc": prov.get("qc", {}),
        "next_step": (
            "Kinematic candidate only — momentum is not conserved across a "
            "seam. Certify with sculptor.refs.track before training on it."),
    }


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
    mutates mission.json via the same save_mission path those jobs
    also write through (atomic tmp+rename per write, but concurrent
    writers still race whole-file: last-write-wins)."""
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

    # A Stage stores a clip id with no embodiment beside it, and at training
    # time `_stage_reference_robot_slug` re-derives the robot from the
    # project's own task_id. `_find_index_row` scans every robot directory, so
    # attaching a g1 clip to a Go1 project used to succeed, render correctly
    # in the stage card, and then fail hours later with
    # `reference_tracking_seed_failed` — unrecoverable without hand-editing
    # mission.json, because the mismatch is not representable in the state.
    project_robot = _project_reference_robot(project_dir)
    clip_robot = str(row.get("robot") or "")
    if project_robot and clip_robot and clip_robot != project_robot:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "reference clip is for a different robot",
            detail=(
                f"{body.clip_id!r} is retargeted to {clip_robot!r}, but this "
                f"project trains {project_robot!r}. A stage records only the "
                f"clip id, so the robot is re-derived from the project at "
                f"training time and this would fail the stage after the run "
                f"had already started."),
            type_="/problems/reference-motion")

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


# ── the OGMP mode automaton + its reward scaffold ──────────────────────
# `ModeTimeline.tsx` already draws the automaton at compose time by mirroring
# the derivation in TypeScript. These endpoints are for the half that cannot be
# mirrored: turning it into reward code. `sculptor.mode_rewards` emits a module
# whose per-mode gating is derived from the graph rather than authored, because
# both real Tier-D failures in this repo were phase-clock bugs — see
# HANDOFF.md §12.
class ModeRewardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    robot: str = "g1"
    goal: str = ""
    #: Include the reference-tracking backbone. Default on: without it every
    #: mode is a stub paying zero, so the module is not trainable until every
    #: mode has been authored.
    tracking: bool = True
    #: Filename under `<project>/rewards/`. Never a path — see `_reward_dest`.
    filename: str = "mode_reward_v0.py"
    overwrite: bool = False


def _episode_horizon_s(project_dir: Path) -> Optional[float]:
    """The episode the mode automaton has to cover, or None if unknowable.

    A per-mode reward's windows are raw CLIP seconds. When the project has an
    authored world those are the wrong clock: on platform-ascent-showcase a
    6.92 s composite gated a 20 s episode, so `active_mode` clamped 75% of
    every episode into the terminal mode (which pays stillness) and the entry
    window demanded a speed above the runtime's own command cap. Three sculpt
    iterations later the robot had never left the ground. See
    `sculptor.mode_rewards.scale_windows`.

    Read from the promoted selection's task ref rather than the compiler's
    `traversal_window_s`: that value is derived at runtime and never persisted
    (it reaches stderr and nothing else), while `episode_length_s` is on disk
    and is the quantity the automaton actually has to span.

    None whenever anything is missing — no authored world, no promoted
    selection, malformed json — so projects without a world keep clip time and
    byte-identical output.
    """
    sel = project_dir / "env" / "selection_current.json"
    try:
        refs = (json.loads(sel.read_text(encoding="utf-8")) or {}).get("refs")
        task_rel = ((refs or {}).get("task") or {}).get("path")
        if not task_rel:
            return None
        task = json.loads(
            (project_dir / task_rel).read_text(encoding="utf-8")) or {}
        value = (((task.get("shared") or {}).get("termination") or {})
                 .get("episode_length_s"))
        horizon = float(value)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return horizon if horizon > 0.0 else None


def _load_mode_graph(clip_id: str, robot: str):
    """`(clip, graph)` or a `_problem(...)` response. Never raises."""
    from sculptor.modes import ModeError, modes_from_composition
    from sculptor.reference import load_clip
    from sculptor.refs.library import CLIP_FILENAME

    if not _valid_clip_id(clip_id):
        return None, _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"malformed clip_id {clip_id!r}",
            type_="/problems/not-found")
    path = _clip_dir_for(clip_id, robot) / CLIP_FILENAME
    if not path.is_file():
        return None, _problem(
            status.HTTP_404_NOT_FOUND, "reference clip not found",
            detail=f"no clip at {path}", type_="/problems/not-found")
    try:
        clip = load_clip(path)
        return (clip, modes_from_composition(clip, clip_id=clip_id)), None
    except ModeError as e:
        # The common case is a single-clip reference: there is one mode and no
        # transition to derive, which is a 422 (the request is well-formed, the
        # clip just is not a composite) rather than a 404 or a 500.
        return None, _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "reference has no mode automaton", detail=str(e),
            type_="/problems/not-a-composite")
    except (OSError, ValueError) as e:
        return None, _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "reference clip could not be read", detail=str(e),
            type_="/problems/unreadable-clip")


@router.get(
    "/references/{clip_id}/modes",
    responses={404: {"model": ProblemDetail}, 422: {"model": ProblemDetail}},
)
def get_reference_modes(clip_id: str, robot: str = Query("g1")) -> Any:
    """The hybrid automaton derived from a composite's own provenance."""
    from sculptor.mode_rewards import mode_windows_s

    loaded, err = _load_mode_graph(clip_id, robot)
    if err is not None:
        return err
    _clip, graph = loaded
    windows = mode_windows_s(graph)
    return {
        "clip_id": clip_id,
        "fps": graph.fps,
        "modes": [
            {"name": m.name,
             "frame_range": list(m.frame_range),
             "start_s": windows[m.name][0],
             "end_s": windows[m.name][1],
             "source_clip_id": m.source_clip_id}
            for m in graph.modes
        ],
        "transitions": [
            {"from_mode": t.from_mode, "to_mode": t.to_mode,
             "guard_kind": t.guard.kind,
             "at_phase": t.guard.at_phase,
             "expression": t.guard.expression}
            for t in graph.transitions
        ],
    }


def _reward_dest(project_dir: Path, filename: str) -> Optional[Path]:
    """`<project>/rewards/<filename>`, or None when `filename` tries to escape.

    The name reaches this from a request body, so it is validated rather than
    trusted: anything with a separator, a parent ref, or a non-`.py` suffix is
    refused outright instead of being sanitized into something that looks
    accepted.
    """
    name = (filename or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", name):
        return None
    if name.startswith(".") or ".." in name:
        return None
    return project_dir / "rewards" / name


def _chain_name(stem: str, mode_name: str) -> str:
    """`mode_reward_v0` -> `mode_reward_v1`, anything else -> `<stem>_<mode>.py`.

    Authoring is one mode per call so the versions chain. Naming after the mode
    when there is no version to bump keeps a hand-placed scaffold from silently
    overwriting itself on the second call.
    """
    from sculptor.mode_rewards import mode_ident

    m = re.fullmatch(r"(.*?)(\d+)", stem)
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1}.py"
    return f"{stem}_{mode_ident(mode_name)}.py"


@router.get(
    "/projects/{slug}/mode-rewards",
    responses={404: {"model": ProblemDetail}},
)
def list_mode_rewards(slug: str, request: Request) -> Any:
    """Every `mode_reward_v*.py` on disk, with per-mode authored state.

    These files were previously write-only. `reward_store._V_RE` matches
    `^v(\\d+)\\.py$`, so they are invisible in the Rewards tab; there was no
    endpoint to read them; and all authoring progress lived in one component's
    `useState`. Reloading the page therefore showed a panel whose only button
    was "Scaffold reward", which overwrote the very bodies the reload had
    hidden. This is the read side that makes the panel resumable.
    """
    from sculptor.mode_rewards import authored_modes

    store: ProjectStore = request.app.state.project_store
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    rewards_dir = project_dir / "rewards"
    out: list[dict[str, Any]] = []
    for path in sorted(rewards_dir.glob("mode_reward_v*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        authored = authored_modes(source)
        if not authored:
            continue
        windows = _mode_windows_from_source(source)
        clip_id = _spec_str_from_source(source, "reference_clip_id")
        out.append({
            "filename": path.name,
            "path": str(path),
            "clip_id": clip_id,
            "mtime": path.stat().st_mtime,
            "modes": [
                {"name": name,
                 "start_s": windows.get(name, (0.0, 0.0))[0],
                 "end_s": windows.get(name, (0.0, 0.0))[1],
                 "authored": bool(done)}
                for name, done in authored.items()
            ],
            "unauthored": [n for n, done in authored.items() if not done],
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)

    # What a run would actually train. `promote_mode_reward` copies a
    # `mode_reward_v<n>.py` into the version chain as `v<n>.py` and repoints
    # `current.py`; until then the authored bodies are inert. Reporting the
    # promoted side here is what lets the Behavior flow say "4 modes will
    # train" rather than "4 modes exist on disk somewhere".
    promoted: dict[str, Any] | None = None
    best_n = -1
    best_path: Path | None = None
    for path in rewards_dir.glob("v*.py"):
        m = re.fullmatch(r"v(\d+)", path.stem)
        if m and int(m.group(1)) > best_n:
            best_n, best_path = int(m.group(1)), path
    if best_path is not None:
        try:
            source = best_path.read_text(encoding="utf-8")
        except OSError:
            source = ""
        authored = authored_modes(source) if source else {}
        if authored:
            windows = _mode_windows_from_source(source)
            promoted = {
                "version": best_n,
                "filename": best_path.name,
                "clip_id": _spec_str_from_source(source, "reference_clip_id"),
                "modes": [
                    {"name": name,
                     "start_s": windows.get(name, (0.0, 0.0))[0],
                     "end_s": windows.get(name, (0.0, 0.0))[1],
                     "authored": bool(done)}
                    for name, done in authored.items()
                ],
                "unauthored": [n for n, done in authored.items() if not done],
            }

    return {"mode_rewards": out, "promoted": promoted}


def _mode_windows_from_source(source: str) -> dict[str, tuple[float, float]]:
    """`MODE_WINDOWS_S` without importing the module.

    Reading it by literal_eval keeps this endpoint side-effect free — the
    module's top level builds numpy tables and we only want the windows.
    """
    m = re.search(r"^MODE_WINDOWS_S: dict = (\{.*?\n\})", source, re.M | re.S)
    if not m:
        return {}
    try:
        raw = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for name, span in (raw or {}).items():
        try:
            out[str(name)] = (float(span[0]), float(span[1]))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _spec_str_from_source(source: str, key: str) -> str:
    """A string field of `REWARD_SPEC`, read out of the source text.

    Parsed, not matched. The scaffold writes the dict over many lines with
    double-quoted keys, and every promotion runs it back out as a single-line
    `repr` with single-quoted keys — so a line-anchored pattern for one shape
    returned "" for the other, and `/mode-rewards` reported an empty clip_id
    for exactly the promoted rewards the field exists to identify.
    """
    from sculptor.mode_rewards import reward_spec_from_source

    val = reward_spec_from_source(source).get(key)
    return val if isinstance(val, str) else ""


@router.post(
    "/projects/{slug}/references/{clip_id}/mode-reward",
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail},
               422: {"model": ProblemDetail}},
)
def scaffold_mode_reward(
    slug: str, clip_id: str, body: ModeRewardRequest, request: Request,
) -> Any:
    """Write a per-mode reward scaffold into the project's `rewards/`.

    The modes come back with `authored: false` — every body is a stub paying
    nothing, and the point of the scaffold is the gating, not the terms. With
    the tracking backbone the module is still trainable as-is; authoring adds
    mode-specific task terms on top of it.
    """
    from sculptor.mode_rewards import (authored_modes,
                                       generate_mode_reward_scaffold,
                                       mode_windows_s,
                                       validate_mode_reward_source)
    from sculptor.modes import ModeError

    store: ProjectStore = request.app.state.project_store
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    if clip_id != body.clip_id:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "clip_id mismatch",
            detail=f"path says {clip_id!r}, body says {body.clip_id!r}",
            type_="/problems/validation-error")

    loaded, err = _load_mode_graph(clip_id, body.robot)
    if err is not None:
        return err
    clip, graph = loaded

    dest = _reward_dest(project_dir, body.filename)
    if dest is None:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid filename",
            detail=f"{body.filename!r} must be a bare .py filename",
            type_="/problems/validation-error")
    if dest.exists() and not body.overwrite:
        return _problem(
            status.HTTP_409_CONFLICT, "reward file already exists",
            detail=f"{dest.name} exists; regenerating would discard any "
                   "authored mode bodies. Pass overwrite=true to replace it.",
            type_="/problems/conflict")

    try:
        source = generate_mode_reward_scaffold(
            graph, behavior_goal=body.goal, clip_id=clip_id,
            clip=clip if body.tracking else None,
            horizon_s=_episode_horizon_s(project_dir))
    except ModeError as e:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "could not scaffold a reward for this automaton", detail=str(e),
            type_="/problems/validation-error")

    errors = validate_mode_reward_source(source, graph)
    if errors:   # a scaffold failing its own validator is a bug here, not input
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "scaffold failed validation",
            detail="; ".join(errors), type_="/problems/validation-error")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source, encoding="utf-8")

    authored = authored_modes(source)
    windows = mode_windows_s(graph)
    return {
        "path": str(dest),
        "filename": dest.name,
        "clip_id": clip_id,
        "tracking": bool(body.tracking),
        "modes": [
            {"name": name,
             "start_s": windows[name][0],
             "end_s": windows[name][1],
             "authored": bool(done)}
            for name, done in authored.items()
        ],
        "unauthored": [n for n, done in authored.items() if not done],
    }


class ModeAuthorRequest(BaseModel):
    """POST /projects/{slug}/references/{clip_id}/mode-reward/author — Claude
    writes ONE mode's reward bodies into an existing scaffold."""

    model_config = ConfigDict(extra="forbid")
    clip_id: str
    robot: str = "g1"
    #: Which mode to author. One per request: the scaffold's gating is already
    #: correct, so the only thing a model can get wrong is one window's terms.
    mode: str
    #: The scaffold to author into — a bare filename under the project's
    #: `rewards/`, as returned by the scaffold endpoint.
    filename: str = "mode_reward_v0.py"
    #: Where to write the result. Defaults to overwriting nothing: the caller
    #: chains modes by passing the previous response's `filename` back in.
    out_filename: Optional[str] = None
    goal: str = ""
    mode_goal: str = ""


@router.post(
    "/projects/{slug}/references/{clip_id}/mode-reward/author",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail},
               422: {"model": ProblemDetail}, 503: {"model": ProblemDetail}},
)
def author_mode_reward(
    slug: str, clip_id: str, body: ModeAuthorRequest, request: Request,
) -> Any:
    """Author one mode's reward terms. Fires a background job; poll
    `GET /jobs/{job_id}`.

    The scaffold's per-mode gating is generated, not authored, so what Claude
    is asked for here is one function body (plus its batched twin) rather than
    a whole reward — which is what keeps a bad edit scoped to one window.
    """
    import os

    store: ProjectStore = request.app.state.project_store
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    if clip_id != body.clip_id:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "clip_id mismatch",
            detail=f"path says {clip_id!r}, body says {body.clip_id!r}",
            type_="/problems/validation-error")

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for mode authoring",
            detail="Set ANTHROPIC_API_KEY in the shell launching the backend "
                   "OR in ../RewardSculptor/.env, then restart.",
            type_="/problems/anthropic-key-missing")

    loaded, err = _load_mode_graph(clip_id, body.robot)
    if err is not None:
        return err
    _clip, graph = loaded
    try:
        graph.mode(body.mode)
    except KeyError:
        names = ", ".join(m.name for m in graph.modes)
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown mode",
            detail=f"{clip_id} has no mode {body.mode!r}; have: {names}",
            type_="/problems/validation-error")

    src = _reward_dest(project_dir, body.filename)
    if src is None:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid filename",
            detail=f"{body.filename!r} must be a bare .py filename",
            type_="/problems/validation-error")
    if not src.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND, "scaffold not found",
            detail=f"{src.name} does not exist — scaffold the mode reward "
                   "first (POST .../mode-reward).",
            type_="/problems/not-found")

    out_name = body.out_filename or _chain_name(src.stem, body.mode)
    dest = _reward_dest(project_dir, out_name)
    if dest is None:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid out_filename",
            detail=f"{out_name!r} must be a bare .py filename",
            type_="/problems/validation-error")
    if dest == src:
        # Authoring in place would leave no way back to the scaffold if the
        # edit is bad, and the caller chains by filename anyway.
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "out_filename same as source",
            detail="authoring writes a new file; pass a different "
                   "out_filename or let it be derived.",
            type_="/problems/validation-error")

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT, "sculpt run in progress",
            detail="Reward edits are locked while a sculpt run is active.",
            type_="/problems/state-conflict")
    for existing in jobs.list(project_slug=slug):
        if existing.kind == "mode_author" and existing.status in (
                "queued", "running"):
            # Two concurrent authoring jobs would race on the chained file.
            return _problem(
                status.HTTP_409_CONFLICT, "mode authoring already active",
                detail="Another mode is being authored. Wait for it to "
                       "finish — modes are authored one at a time.",
                type_="/problems/job-busy",
                active_job_id=existing.job_id)

    from backend.services.mode_jobs import run_mode_author_job

    job = jobs.submit(
        kind="mode_author",  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_mode_author_job(
            project_dir=project_dir, reward_path=src, out_path=dest,
            clip_id=clip_id, robot=body.robot, mode=body.mode,
            behavior_goal=body.goal, mode_goal=body.mode_goal),
        params={"clip_id": clip_id, "mode": body.mode,
                "filename": src.name, "out_filename": dest.name},
    )
    return job.to_summary()


class ModePromoteRequest(BaseModel):
    """POST .../mode-reward/promote — put an authored mode reward into the
    project's reward version chain so a run actually uses it."""

    model_config = ConfigDict(extra="forbid")
    filename: str
    #: Promote even though some modes are still stubs. A stub pays nothing, so
    #: this trains a reward that is blank across part of the episode — which is
    #: legitimate for a bare scaffold (the tracking backbone alone is the
    #: Tier-D path) and a mistake otherwise, hence explicit.
    allow_unauthored: bool = False


@router.post(
    "/projects/{slug}/references/{clip_id}/mode-reward/promote",
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail},
               422: {"model": ProblemDetail}},
)
def promote_mode_reward_route(
    slug: str, clip_id: str, body: ModePromoteRequest, request: Request,
) -> Any:
    """Make the authored reward the one a run will train.

    Authoring writes `mode_reward_v<n>.py`, which is NOT a version — only
    `v<n>.py` is, and `rewards/current.py` (what every adapter imports) points
    at one. Without this step, pressing Run after authoring trains whatever
    `current.py` pointed at before, silently. Synchronous: it is a file copy
    plus a probe, not a model call.
    """
    from sculptor.mode_rewards import ModeAuthorError, promote_mode_reward

    store: ProjectStore = request.app.state.project_store
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found")

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT, "sculpt run in progress",
            detail="Repointing current.py under a live run would swap the "
                   "reward mid-training. Stop the run first.",
            type_="/problems/state-conflict")

    src = _reward_dest(project_dir, body.filename)
    if src is None:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid filename",
            detail=f"{body.filename!r} must be a bare .py filename",
            type_="/problems/validation-error")
    if not src.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND, "reward not found",
            detail=f"{src.name} does not exist in this project's rewards/",
            type_="/problems/not-found")

    contract = None
    try:
        from sculptor.adapters.base import load_adapter
        contract = load_adapter(project_dir / "config.toml").reward_contract()
    except Exception:  # noqa: BLE001 — probe is a bonus, absence is not fatal
        contract = None

    try:
        out = promote_mode_reward(
            src, contract=contract, allow_unauthored=body.allow_unauthored)
    except ModeAuthorError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "reward not promotable", detail=str(e),
            type_="/problems/state-conflict")

    return {"version": out["version"], "filename": out["filename"],
            "path": out["path"], "clip_id": clip_id,
            "unauthored": out["unauthored"],
            "source_filename": out["source_filename"]}
