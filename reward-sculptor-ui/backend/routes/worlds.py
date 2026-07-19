"""Environment-authoring endpoints (env-authoring item 5).

  POST /projects/{slug}/worlds/author        — prompt → draft + questions
  POST /projects/{slug}/worlds/author/apply  — answers → admit + promote
  GET  /projects/{slug}/worlds/selection     — the promoted tuple
  GET  /projects/{slug}/worlds/lineage       — immutable selection history

Authoring and admission are CPU-bound sculptor calls (offline author +
gate chain with a model compile, seconds not minutes) — run in a
threadpool, synchronous response, mirroring the metrics route. Apply
takes the per-project write lock: it promotes the authoritative
selection the sculpt loop trains under.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from backend.models.world import (
    ApplyWorldRequest,
    AuthorWorldRequest,
    WorldApplySummary,
    WorldDraftSummary,
)
from backend.services import sculptor_bridge, world_store
from backend.services.project_store import BusyError, ProjectStore

router = APIRouter(tags=["worlds"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def _problem(code: int, title: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"type": "about:blank", "title": title, "status": code,
                 "detail": detail},
    )


@router.post("/projects/{slug}/worlds/author",
             response_model=WorldDraftSummary)
async def author_world(
    slug: str, body: AuthorWorldRequest,
    store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    if not sculptor_bridge.sculptor_ok():
        return _problem(status.HTTP_503_SERVICE_UNAVAILABLE,
                        "Sculptor unavailable", sculptor_bridge.sculptor_error())
    try:
        return await run_in_threadpool(
            world_store.author, Path(detail.project_dir), body.prompt,
            robot_capability_id=body.robot_capability_id,
            kg_grounding=body.kg_grounding)
    except Exception as exc:  # noqa: BLE001 — AuthoringError et al. → 422
        return _problem(status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "Authoring failed", f"{type(exc).__name__}: {exc}")


@router.post("/projects/{slug}/worlds/author/apply",
             response_model=WorldApplySummary)
async def apply_world(
    slug: str, body: ApplyWorldRequest,
    store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    if not sculptor_bridge.sculptor_ok():
        return _problem(status.HTTP_503_SERVICE_UNAVAILABLE,
                        "Sculptor unavailable", sculptor_bridge.sculptor_error())
    # Sculptor imports only after the availability guard — an unimportable
    # sculptor must surface as the 503 above, not an ImportError 500.
    from sculptor.world.author import AuthoringError, StaleClarificationError
    from sculptor.world.project import WorldPromotionError

    try:
        lock = store.acquire_lock(slug)
    except BusyError as exc:
        return _problem(status.HTTP_409_CONFLICT, "Project busy", str(exc))
    try:
        return await run_in_threadpool(
            world_store.apply, Path(detail.project_dir),
            body.session_id,
            [answer.model_dump() for answer in body.answers])
    except world_store.UnknownSessionError:
        return _problem(status.HTTP_404_NOT_FOUND, "Authoring session not found")
    except (world_store.StaleDraftError, StaleClarificationError) as exc:
        return _problem(status.HTTP_409_CONFLICT, "Stale draft", str(exc))
    except (ValueError, AuthoringError, WorldPromotionError) as exc:
        # includes admission-gate rejections — the detail carries the
        # gate/violation summary for the UI to display
        return _problem(status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "World not admitted", f"{type(exc).__name__}: {exc}")
    finally:
        lock.release()


@router.get("/projects/{slug}/worlds/selection")
async def world_selection(
    slug: str, store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    payload = await run_in_threadpool(
        world_store.selection, Path(detail.project_dir))
    if payload is None:
        return _problem(status.HTTP_404_NOT_FOUND, "No authored world",
                        "This project has no promoted world selection yet.")
    return payload


@router.get("/projects/{slug}/worlds/lineage")
async def world_lineage(
    slug: str, store: ProjectStore = Depends(get_store),
):
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    return await run_in_threadpool(
        world_store.lineage, Path(detail.project_dir))


@router.get("/projects/{slug}/worlds/validate")
async def world_validate(
    slug: str, store: ProjectStore = Depends(get_store),
):
    """Integrity check: tuple hash, per-artifact byte hashes (tamper
    evidence), and schema re-validation of the authoritative selection."""
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    try:
        return await run_in_threadpool(
            world_store.validate, Path(detail.project_dir))
    except FileNotFoundError as exc:
        return _problem(status.HTTP_404_NOT_FOUND, "No authored world",
                        str(exc))


@router.get("/projects/{slug}/worlds/curriculum")
async def world_curriculum(
    slug: str, store: ProjectStore = Depends(get_store),
):
    """Terrain-curriculum progression of the most recent run."""
    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    return await run_in_threadpool(
        world_store.curriculum, Path(detail.project_dir))


@router.get("/projects/{slug}/worlds/preview")
async def world_preview(
    slug: str, angle: str = "iso", regenerate: bool = False,
    store: ProjectStore = Depends(get_store),
):
    """PNG of the materialized evaluation scene (cached per selection
    version + camera angle)."""
    from backend.services.preview_renderer import CAMERA_ANGLES, PreviewError

    detail = store.get(slug)
    if detail is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Project not found")
    if angle not in CAMERA_ANGLES:
        return _problem(status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "Unknown camera angle",
                        f"angle must be one of {sorted(CAMERA_ANGLES)}")
    try:
        path = await run_in_threadpool(
            world_store.preview, Path(detail.project_dir),
            angle=angle, regenerate=regenerate)
    except FileNotFoundError as exc:
        return _problem(status.HTTP_404_NOT_FOUND, "No authored world",
                        str(exc))
    except PreviewError as exc:
        code = (status.HTTP_503_SERVICE_UNAVAILABLE
                if exc.kind == "mujoco_import"
                else status.HTTP_422_UNPROCESSABLE_ENTITY)
        return _problem(code, "Preview unavailable", str(exc))
    return FileResponse(path, media_type="image/png")
