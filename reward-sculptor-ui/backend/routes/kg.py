"""Knowledge-graph endpoints.

  GET   /projects/{slug}/kg/papers
  GET   /projects/{slug}/kg/papers/{arxiv_id}
  GET   /projects/{slug}/kg/techniques
  GET   /projects/{slug}/kg/stats
  GET   /projects/{slug}/kg/pending-seeds   ← seeds in kg_seeds.yml not yet ingested
  POST  /projects/{slug}/kg/seeds           ← async job (ingest + optional extract)
  GET   /projects/{slug}/kg/graph.html      ← pre-built pyvis HTML

Prompt 7 precondition R3: the POST returns the job handle immediately
(202 Accepted), UI closes the dialog, and subsequent progress polling
uses GET /jobs/{job_id}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse

from backend.models.kg import (
    AddSeedsRequest,
    JobSummary,
    KGStats,
    PaperDetail,
    PaperSummary,
    ResearchTopicRequest,
    TechniqueSummary,
)
from backend.models.project import ProblemDetail
from backend.services import kg_store
from backend.services.job_manager import Job, JobManager
from backend.services.project_store import BusyError, ProjectStore


router = APIRouter(tags=["kg"])


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


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


def _project_dir(store: ProjectStore, slug: str) -> Path | None:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir)


# ── list + detail ──────────────────────────────────────────────────────
@router.get(
    "/projects/{slug}/kg/papers",
    response_model=list[PaperSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_papers(
    slug: str,
    extracted: Optional[bool] = Query(None),
    search: Optional[str] = Query(None, max_length=120),
    store: ProjectStore = Depends(get_store),
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    return kg_store.list_papers(
        project_dir, extracted=extracted, search=search
    )


@router.get(
    "/projects/{slug}/kg/papers/{arxiv_id}",
    response_model=PaperDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_paper(
    slug: str, arxiv_id: str, store: ProjectStore = Depends(get_store)
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    detail = kg_store.get_paper_detail(project_dir, arxiv_id)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "paper not found",
            detail=f"no paper with arxiv_id={arxiv_id!r} in project KG",
            type_="/problems/not-found",
        )
    return detail


@router.get(
    "/projects/{slug}/kg/techniques",
    response_model=list[TechniqueSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_techniques(
    slug: str, store: ProjectStore = Depends(get_store)
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    return kg_store.list_techniques(project_dir)


@router.get(
    "/projects/{slug}/kg/stats",
    response_model=KGStats,
    responses={404: {"model": ProblemDetail}},
)
def kg_stats(
    slug: str, store: ProjectStore = Depends(get_store)
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    return kg_store.get_stats(project_dir)


@router.get(
    "/projects/{slug}/kg/pending-seeds",
    responses={404: {"model": ProblemDetail}},
)
def kg_pending_seeds(
    slug: str, store: ProjectStore = Depends(get_store)
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    seeds = store.pending_seeds(slug)
    try:
        ingested = {p.arxiv_id for p in kg_store.list_papers(project_dir)}
    except Exception:  # noqa: BLE001
        ingested = set()
    pending = [s for s in seeds if s.get("arxiv_id") not in ingested]
    return {"pending": pending, "ingested_count": len(ingested)}


# ── POST seeds (async) ─────────────────────────────────────────────────
@router.post(
    "/projects/{slug}/kg/seeds",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        412: {"model": ProblemDetail},
    },
)
async def add_seeds(
    slug: str,
    body: AddSeedsRequest,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )

    for existing in jobs.list(project_slug=slug):
        if existing.kind in ("kg_ingest", "kg_extract", "kg_ingest_extract") \
                and existing.status in ("queued", "running"):
            return _problem(
                status.HTTP_409_CONFLICT,
                "KG job already active",
                detail=(
                    "An ingest/extract job is still running for this "
                    "project. Wait for it to finish before adding more."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "project busy",
            detail=str(e), type_="/problems/state-conflict",
        )
    try:
        seeds_dicts = [s.model_dump(exclude_none=True) for s in body.seeds]
        store.append_seeds(slug, seeds_dicts)
    finally:
        lock.release()

    # Per R3: return the job handle immediately. The client polls
    # GET /jobs/{id} every 3s for progress.
    from backend.services.kg_jobs import run_ingest_extract_job

    job_kind = "kg_ingest_extract" if body.auto_extract else "kg_ingest"
    job = jobs.submit(
        kind=job_kind,  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_ingest_extract_job(
            project_dir=project_dir,
            auto_extract=body.auto_extract,
        ),
        params={
            "seeds_added": [s.model_dump(exclude_none=True) for s in body.seeds],
            "auto_extract": body.auto_extract,
        },
    )
    return job.to_summary()


# ── bulk-ingest curated seeds (M7: one-click global seeds) ──────────────
_GLOBAL_SEEDS_PATH = (
    Path(__file__).resolve().parents[3]
    / "RewardSculptor" / "examples" / "kg_seeds_global.yml"
)


@router.post(
    "/projects/{slug}/kg/ingest-global",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def ingest_global_seeds(
    slug: str,
    request: Request,
    auto_extract: bool = Query(default=True),
    store: ProjectStore = Depends(get_store),
) -> Any:
    """One-click ingest of the curated ~50-paper KG seeds file at
    `RewardSculptor/examples/kg_seeds_global.yml`. Appends to the
    project's `kg_seeds.yml` (idempotent on arxiv_id — already-ingested
    papers skip), fires an ingest+extract job, returns the job handle.

    The UI wires this to a single "Bulk-seed from canonical library"
    button on the KG tab so users don't need to type anything.
    """
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )

    if not _GLOBAL_SEEDS_PATH.is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "global seeds file not found",
            detail=(
                f"Expected {_GLOBAL_SEEDS_PATH} — is the sculptor repo "
                f"at its usual location?"
            ),
            type_="/problems/seeds-missing",
        )

    # Refuse if another KG job is already in flight.
    for existing in jobs.list(project_slug=slug):
        if existing.kind in ("kg_ingest", "kg_extract", "kg_ingest_extract") \
                and existing.status in ("queued", "running"):
            return _problem(
                status.HTTP_409_CONFLICT,
                "KG job already active",
                detail=(
                    "An ingest/extract job is still running for this "
                    "project. Wait for it to finish before triggering "
                    "another bulk ingest."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    # Parse the global seeds YAML + append to the project's kg_seeds.yml.
    import yaml
    try:
        with _GLOBAL_SEEDS_PATH.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "global seeds YAML malformed",
            detail=str(e), type_="/problems/seeds-malformed",
        )
    papers = doc.get("papers") or []
    if not isinstance(papers, list) or not papers:
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "global seeds YAML has no papers",
            type_="/problems/seeds-empty",
        )
    seeds_to_add: list[dict[str, Any]] = []
    for p in papers:
        if not isinstance(p, dict):
            continue
        aid = p.get("arxiv_id")
        if not aid:
            continue
        seeds_to_add.append({k: v for k, v in p.items() if v is not None})

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "project busy",
            detail=str(e), type_="/problems/state-conflict",
        )
    try:
        store.append_seeds(slug, seeds_to_add)
    finally:
        lock.release()

    from backend.services.kg_jobs import run_ingest_extract_job

    job_kind = "kg_ingest_extract" if auto_extract else "kg_ingest"
    job = jobs.submit(
        kind=job_kind,  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_ingest_extract_job(
            project_dir=project_dir,
            auto_extract=auto_extract,
        ),
        params={
            "seeds_added_count": len(seeds_to_add),
            "auto_extract": auto_extract,
            "preset": "global",
        },
    )
    return job.to_summary()


# ── POST /projects/{slug}/kg/research ────────────────────────────────
@router.post(
    "/projects/{slug}/kg/research",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def research_topic_endpoint(
    slug: str,
    body: ResearchTopicRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Prompt-time KG research (M7 P2). Claude picks arxiv IDs relevant
    to `topic`, dedupes against the KG, ingests + extracts the new
    ones.

    Writes target the same resolved KG that other routes use
    (`kg_store.project_kg_db_path`) — i.e. the shared DB for projects
    created under M7 Phase 1, or the legacy per-project DB if one is
    still around.

    Returns a 202 + job handle; poll via `GET /jobs/{job_id}`. The
    payload at `result.new_papers` is the set of arxiv IDs Claude
    recommended (after dedup); `coverage_note` is Claude's free-form
    note about gaps when the topic is thinly covered.
    """
    import os

    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY required for prompt-time research",
            detail=(
                "Set ANTHROPIC_API_KEY in the shell launching the "
                "backend OR in ../RewardSculptor/.env, then restart."
            ),
            type_="/problems/anthropic-key-missing",
        )

    # Mutex with other KG jobs on this project. A global shared KG
    # means concurrent writes across projects are possible; the
    # JobManager's per-project lock is conservative but cheap.
    for existing in jobs.list(project_slug=slug):
        if existing.kind in (
            "kg_ingest", "kg_extract", "kg_ingest_extract", "kg_research"
        ) and existing.status in ("queued", "running"):
            return _problem(
                status.HTTP_409_CONFLICT,
                "KG job already active",
                detail=(
                    "An ingest / extract / research job is still "
                    "running for this project. Wait for it to finish "
                    "before firing another."
                ),
                type_="/problems/job-busy",
                active_job_id=existing.job_id,
            )

    from backend.services.kg_jobs import run_research_job

    job = jobs.submit(
        kind="kg_research",  # type: ignore[arg-type]
        project_slug=slug,
        fn=run_research_job(
            project_dir=project_dir,
            topic=body.topic,
            max_papers=body.max_papers,
            auto_extract=body.auto_extract,
        ),
        params={
            "topic": body.topic,
            "max_papers": body.max_papers,
            "auto_extract": body.auto_extract,
        },
    )
    return job.to_summary()


# ── pyvis HTML ─────────────────────────────────────────────────────────
@router.get(
    "/projects/{slug}/kg/graph.html",
    response_class=FileResponse,
    responses={
        200: {"content": {"text/html": {}}},
        404: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
    },
)
def kg_graph_html(
    slug: str,
    regenerate: bool = Query(False),
    store: ProjectStore = Depends(get_store),
) -> Any:
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )

    html_path = project_dir / "reports" / "kg.html"
    if regenerate or not html_path.is_file() or html_path.stat().st_size == 0:
        try:
            from sculptor.kg.viz import build_kg_html

            from sculptor.kg.store import SculptorKG
            db = kg_store.project_kg_db_path(project_dir)
            db.parent.mkdir(parents=True, exist_ok=True)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json

            prov_path = project_dir / "reports" / "provenance.json"
            provenance = None
            if prov_path.is_file():
                try:
                    provenance = _json.loads(prov_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    provenance = None
            with SculptorKG(db) as kg:
                build_kg_html(
                    kg,
                    out_path=html_path,
                    provenance=provenance,
                    title=f"{slug} — Knowledge Graph",
                )
        except Exception as e:  # noqa: BLE001
            return _problem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "graph render failed",
                detail=f"{type(e).__name__}: {e}",
                type_="/problems/preview-failed",
            )

    return FileResponse(html_path, media_type="text/html")


# ── §7.7 / §Ship-7: heal stub titles via UI ─────────────────────────────
@router.post(
    "/projects/{slug}/kg/heal-stubs",
    responses={
        200: {"description": "Heal results per arxiv_id."},
        404: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
    },
)
def kg_heal_stubs(
    slug: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Re-ingest Paper nodes whose title is still `arxiv:XXXX.XXXXX`.

    Stubs get left behind when the arxiv API rate-limits a bulk ingest
    (§7.7). This route scans the shared KG (resolved via
    `project_kg_db_path`) and calls `heal_stub_titles` with
    `force=True`. Response: `{"results": {arxiv_id: "healed" |
    "still_stubbed" | "error: ..."}, "summary": {...counts}}`.
    Synchronous — typically finishes in ~2 min for ~10 stubs on a
    healthy network; UI should show a spinner.
    """
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            type_="/problems/not-found",
        )
    try:
        from sculptor.kg.ingest import heal_stub_titles
        from sculptor.kg.store import SculptorKG

        db = kg_store.project_kg_db_path(project_dir)
        db.parent.mkdir(parents=True, exist_ok=True)
        with SculptorKG(db) as kg:
            results = heal_stub_titles(store=kg)
    except Exception as e:  # noqa: BLE001
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "heal-stubs failed",
            detail=f"{type(e).__name__}: {e}",
            type_="/problems/preview-failed",
        )
    healed = sum(1 for v in results.values() if v == "healed")
    still = sum(1 for v in results.values() if v == "still_stubbed")
    errored = len(results) - healed - still
    return {
        "results": results,
        "summary": {
            "healed": healed,
            "still_stubbed": still,
            "errored": errored,
            "total": len(results),
        },
    }
