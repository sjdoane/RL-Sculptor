"""Robot-library endpoints (MJLAB_PIVOT_DESIGN §5.1).

GET /library/robots?category=&training_support=&search=
GET /library/robots/{slug}
GET /library/robots/{slug}/thumbnail
GET /library/categories
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import FileResponse, JSONResponse

from backend.models.library import (
    AdapterInfoResponse,
    LibraryListResponse,
    LibraryPreconfiguredTask,
    LibraryReference,
    LibraryRobotResponse,
)
from backend.models.project import ProblemDetail
from backend.services.robot_library import (
    CATEGORIES,
    RobotEntry,
    get_library,
)


router = APIRouter(prefix="/library", tags=["library"])


_THUMBNAIL_ROOT = (
    Path(__file__).resolve().parents[2] / "frontend" / "public" / "robots"
)


def _entry_to_response(entry: RobotEntry) -> LibraryRobotResponse:
    return LibraryRobotResponse(
        slug=entry.slug,
        display_name=entry.display_name,
        category=entry.category,
        description=entry.description,
        source=entry.source,
        menagerie_package=entry.menagerie_package,
        training_support=entry.training_support,
        is_smoke_test_target=entry.is_smoke_test_target,
        preconfigured_tasks=[
            LibraryPreconfiguredTask(
                task_id=t.task_id,
                display_name=t.display_name,
                recommended_num_envs=t.recommended_num_envs,
            )
            for t in entry.preconfigured_tasks
        ],
        references=[
            LibraryReference(kind=r.kind, url=r.url, citation=r.citation)
            for r in entry.references
        ],
        thumbnail_path=entry.thumbnail_path,
        demote_note=entry.demote_note,
    )


@router.get("/categories", response_model=list[str])
def list_categories() -> list[str]:
    return list(CATEGORIES)


@router.get("/adapters", response_model=list[AdapterInfoResponse])
def list_adapters() -> list[AdapterInfoResponse]:
    """M5: enumerate ready + coming-soon adapters for the UI's project-
    creation dropdown. Ordered ready first, then coming-soon by name.
    """
    from sculptor.adapters import ADAPTER_REGISTRY  # type: ignore

    entries = list(ADAPTER_REGISTRY.values())
    entries.sort(key=lambda a: (a.status != "ready", a.name))
    return [
        AdapterInfoResponse(
            name=a.name,
            display_name=a.display_name,
            class_path=a.class_path,
            status=a.status,
            supported_robot_categories=list(a.supported_robot_categories),
            adoption_guide_url=a.adoption_guide_url,
            estimated_effort=a.estimated_effort,
        )
        for a in entries
    ]


@router.get("/robots", response_model=LibraryListResponse)
def list_robots(
    category: Optional[str] = Query(default=None),
    training_support: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
) -> LibraryListResponse:
    library = get_library()
    entries = library.list_robots(
        category=category,
        training_support=training_support,
        search=search,
    )
    return LibraryListResponse(
        robots=[_entry_to_response(e) for e in entries],
        total=len(entries),
        categories=list(CATEGORIES),
    )


@router.get(
    "/robots/{slug}",
    response_model=LibraryRobotResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_robot(slug: str):  # type: ignore[no-untyped-def]
    library = get_library()
    entry = library.get_robot(slug)
    if entry is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ProblemDetail(
                type="/problems/not-found",
                title="robot not found",
                status=404,
                detail=f"no library robot with slug {slug!r}",
            ).model_dump(),
            media_type="application/problem+json",
        )
    return _entry_to_response(entry)


@router.get(
    "/robots/{slug}/thumbnail",
    responses={404: {"model": ProblemDetail}},
)
def get_thumbnail(slug: str):  # type: ignore[no-untyped-def]
    library = get_library()
    entry = library.get_robot(slug)
    if entry is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ProblemDetail(
                type="/problems/not-found",
                title="robot not found",
                status=404,
                detail=f"no library robot with slug {slug!r}",
            ).model_dump(),
            media_type="application/problem+json",
        )
    # thumbnail_path is stored relative to frontend/public/, e.g.
    # "robots/unitree_g1.webp". Resolve against the repo root.
    rel = entry.thumbnail_path.lstrip("/")
    candidate = _THUMBNAIL_ROOT.parent / rel  # frontend/public/<rel>
    if not candidate.is_file():
        # Fallback to .png next to the declared .webp (generation script
        # may produce either format).
        alt_png = candidate.with_suffix(".png")
        if alt_png.is_file():
            candidate = alt_png
        else:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ProblemDetail(
                    type="/problems/thumbnail-missing",
                    title="thumbnail not yet generated",
                    status=404,
                    detail=(
                        f"No thumbnail at {candidate} — run "
                        f"scripts/generate_library_thumbnails.py to produce."
                    ),
                ).model_dump(),
                media_type="application/problem+json",
            )
    return FileResponse(candidate)
