"""Robot-source + static-preview endpoints.

Three POST/GET shapes (spec: Prompt 4):
  POST /projects/{slug}/robot/library  → library pick, sets env_id in config.toml
  POST /projects/{slug}/robot/urdf     → multipart upload (.urdf or .xml MJCF) + optional .zip meshes
  GET  /projects/{slug}/robot          → current state
  GET  /projects/{slug}/preview        → cached PNG (renders on first request)

Per R5: every write endpoint acquires the per-project filelock.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from backend.models.project import ProblemDetail
from backend.models.robot import (
    LIBRARY_ROBOT_MAP,
    LibraryPickRequest,
    RobotStateResponse,
)
from backend.services import preview_renderer
from backend.services.job_manager import JobManager
from backend.services.preview_renderer import PreviewError
from backend.services.project_store import BusyError, ProjectStore


router = APIRouter(tags=["robot"])


# Default arxiv seeds seeded into kg_seeds.yml on library pick. Same
# four papers the sculptor's examples/hopper reference project uses,
# picked to be reasonable for any of the 5 MuJoCo locomotion robots.
DEFAULT_LIBRARY_SEEDS: list[dict[str, str]] = [
    {
        "arxiv_id": "1707.06347",
        "title": "Proximal Policy Optimization Algorithms",
        "rationale": "Training algorithm used by the GymSB3Adapter.",
    },
    {
        "arxiv_id": "1606.01540",
        "title": "OpenAI Gym",
        "rationale": "The env framework Gymnasium is a fork of.",
    },
    {
        "arxiv_id": "1709.06560",
        "title": "Deep Reinforcement Learning That Matters",
        "rationale": "Variance + reporting pitfalls in deep RL benchmarking.",
    },
    {
        "arxiv_id": "1801.00690",
        "title": "DeepMind Control Suite",
        "rationale": "Bounded reward structure + tolerance kernels.",
    },
]


# Shared limits. Tuned per spec: 50 MB combined for URDF uploads.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = 20 * 1024 * 1024
MAX_ZIP_RATIO = 100

# Paths under <project>/
ROBOT_SUBDIR = Path("uploads") / "robot"

ALLOWED_MODEL_SUFFIXES = {".urdf", ".xml", ".mjcf"}


# ── DI ─────────────────────────────────────────────────────────────────
def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


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


def _require_project(store: ProjectStore, slug: str) -> Optional[JSONResponse]:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    return None


def _build_robot_response(
    store: ProjectStore, slug: str
) -> RobotStateResponse:
    rs = store.read_robot_source(slug)
    env_id = store.read_env_id(slug) if rs.get("kind") == "library" else rs.get("env_id")
    kind = rs.get("kind") or "none"
    if kind not in ("library", "mjcf", "urdf", "none"):
        kind = "none"
    uploaded_at = rs.get("uploaded_at")
    upl_dt: Optional[datetime] = None
    if isinstance(uploaded_at, str):
        try:
            upl_dt = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
        except ValueError:
            upl_dt = None
    return RobotStateResponse(
        kind=kind,  # type: ignore[arg-type]
        library_name=rs.get("library_name"),
        env_id=env_id,
        model_file=rs.get("model_file"),
        original_filename=rs.get("original_filename"),
        size_bytes=rs.get("size_bytes"),
        uploaded_at=upl_dt,
        mesh_paths=list(rs.get("mesh_paths") or []),
    )


# ── GET /projects/{slug}/robot ────────────────────────────────────────
@router.get(
    "/projects/{slug}/robot",
    response_model=RobotStateResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_robot(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    err = _require_project(store, slug)
    if err is not None:
        return err
    return _build_robot_response(store, slug)


# ── POST /projects/{slug}/robot/library ───────────────────────────────
@router.post(
    "/projects/{slug}/robot/library",
    response_model=RobotStateResponse,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def set_library_robot(
    slug: str,
    body: LibraryPickRequest,
    store: ProjectStore = Depends(get_store),
) -> Any:
    err = _require_project(store, slug)
    if err is not None:
        return err

    env_id = LIBRARY_ROBOT_MAP[body.robot_name]

    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "project busy",
            detail=str(e), type_="/problems/state-conflict",
        )
    try:
        try:
            store.set_env_id(slug, env_id)
        except (FileNotFoundError, RuntimeError) as e:
            return _problem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "config.toml update failed",
                detail=str(e),
            )
        store.write_robot_source(
            slug,
            {
                "kind": "library",
                "library_name": body.robot_name,
                "library_slug": body.robot_name,
                "reference_robot": body.robot_name,
                "env_id": env_id,
                "model_file": None,
                "original_filename": None,
                "size_bytes": None,
                "uploaded_at": None,
                "mesh_paths": [],
            },
        )
        store.invalidate_preview(slug)
        # Seed the project's kg_seeds.yml with sane defaults so the KG
        # tab has something to show. Idempotent on arxiv_id — safe to
        # call repeatedly.
        store.append_seeds(slug, list(DEFAULT_LIBRARY_SEEDS))
    finally:
        lock.release()

    return _build_robot_response(store, slug)


# ── POST /projects/{slug}/robot/urdf ──────────────────────────────────
@router.post(
    "/projects/{slug}/robot/urdf",
    response_model=RobotStateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        413: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
    },
)
async def upload_urdf(
    slug: str,
    model_file: UploadFile = File(..., description=".urdf or .xml/.mjcf model"),
    meshes_zip: Optional[UploadFile] = File(
        None, description="optional zip of mesh files referenced by the model"
    ),
    store: ProjectStore = Depends(get_store),
) -> Any:
    err = _require_project(store, slug)
    if err is not None:
        return err

    # ── extension check ──────────────────────────────────────────────
    orig_name = model_file.filename or "model"
    suffix = Path(orig_name).suffix.lower()
    if suffix not in ALLOWED_MODEL_SUFFIXES:
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "unsupported model file extension",
            detail=(
                f"filename {orig_name!r} has suffix {suffix!r}; "
                f"allowed: {sorted(ALLOWED_MODEL_SUFFIXES)}"
            ),
            type_="/problems/validation-error",
        )

    # ── read bytes + size cap ────────────────────────────────────────
    model_bytes = await model_file.read()
    total = len(model_bytes)
    zip_bytes: Optional[bytes] = None
    if meshes_zip is not None:
        zip_bytes = await meshes_zip.read()
        total += len(zip_bytes)
    if total > MAX_UPLOAD_BYTES:
        return _problem(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "upload exceeds size cap",
            detail=(
                f"combined size {total} B > {MAX_UPLOAD_BYTES} B "
                f"(50 MB). Split meshes out of the model or compress further."
            ),
            type_="/problems/upload-too-large",
        )
    if not model_bytes:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "empty model file",
            type_="/problems/validation-error",
        )

    # ── acquire lock + write to a staging area ───────────────────────
    try:
        lock = store.acquire_lock(slug)
    except BusyError as e:
        return _problem(
            status.HTTP_409_CONFLICT, "project busy",
            detail=str(e), type_="/problems/state-conflict",
        )
    try:
        project_detail = store.get(slug)
        assert project_detail is not None
        project_dir = Path(project_detail.project_dir)
        robot_dir = project_dir / ROBOT_SUBDIR
        staging = project_dir / "uploads" / ".robot-incoming"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        # Sanitize filename: keep alnum/_/./- only, avoid path traversal.
        safe_name = _sanitize_filename(orig_name)
        model_path = staging / safe_name
        model_path.write_bytes(model_bytes)

        mesh_paths: list[str] = []
        if zip_bytes is not None:
            try:
                mesh_paths = _extract_mesh_zip(zip_bytes, staging)
            except _ZipReject as e:
                return _problem(
                    status.HTTP_400_BAD_REQUEST,
                    "zip rejected",
                    detail=str(e),
                    type_="/problems/zip-path-traversal",
                )

        # ── validate parseable ──────────────────────────────────────
        try:
            summary = preview_renderer.validate_model_file(model_path)
        except PreviewError as e:
            # Clean staging on any parse failure.
            shutil.rmtree(staging, ignore_errors=True)
            type_ = (
                "/problems/urdf-unsupported"
                if e.kind == "urdf_parse"
                else "/problems/model-parse"
            )
            return _problem(
                status.HTTP_400_BAD_REQUEST,
                "MuJoCo could not load the uploaded model",
                detail=str(e),
                type_=type_,
                parse_kind=e.kind,
            )

        # ── promote staging → uploads/robot ─────────────────────────
        if robot_dir.exists():
            shutil.rmtree(robot_dir)
        robot_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(robot_dir)

        kind = "urdf" if suffix == ".urdf" else "mjcf"
        final_model_rel = str(ROBOT_SUBDIR / safe_name).replace("\\", "/")
        final_mesh_rel = [
            str((ROBOT_SUBDIR / m).as_posix()) for m in mesh_paths
        ]
        store.write_robot_source(
            slug,
            {
                "kind": kind,
                "library_name": None,
                "env_id": None,
                "model_file": final_model_rel,
                "original_filename": orig_name,
                "size_bytes": int(total),
                "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
                "mesh_paths": final_mesh_rel,
                "mujoco_summary": summary,
            },
        )
        store.invalidate_preview(slug)
    finally:
        lock.release()

    return _build_robot_response(store, slug)


# ── GET /projects/{slug}/preview ──────────────────────────────────────
@router.get(
    "/projects/{slug}/preview",
    responses={
        200: {"content": {"image/png": {}}},
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
    },
    response_class=Response,
)
async def get_preview(
    slug: str,
    angle: str = Query(
        preview_renderer.DEFAULT_ANGLE,
        description="Camera angle preset — one of iso|front|side|top.",
    ),
    regenerate: bool = Query(
        False,
        description="Force a re-render even if a cached PNG exists for this angle.",
    ),
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    err = _require_project(store, slug)
    if err is not None:
        return err

    if angle not in preview_renderer.CAMERA_ANGLES:
        return _problem(
            status.HTTP_400_BAD_REQUEST,
            "invalid camera angle",
            detail=(
                f"angle={angle!r} is not one of "
                f"{sorted(preview_renderer.CAMERA_ANGLES)}"
            ),
            type_="/problems/validation-error",
        )

    project_detail = store.get(slug)
    assert project_detail is not None
    project_dir = Path(project_detail.project_dir)

    preview_file = preview_renderer.preview_path(project_dir, angle)
    if (
        not regenerate
        and preview_file.is_file()
        and preview_file.stat().st_size > 0
    ):
        return FileResponse(preview_file, media_type="image/png")

    rs = store.read_robot_source(slug)
    # Ensure env_id populated for library picks, since sidecar may omit it
    # if the project was upgraded from an older schema.
    if rs.get("kind") == "library" and not rs.get("env_id"):
        rs["env_id"] = store.read_env_id(slug)

    # If the user asked for regenerate, drop the cached PNG for this
    # angle first so a failed re-render doesn't silently return stale.
    if regenerate and preview_file.is_file():
        try:
            preview_file.unlink()
        except OSError:
            pass

    try:
        await preview_renderer.render_static_async(project_dir, rs, angle)
    except PreviewError as e:
        # §7.6: a `kind="render"` failure during an active sculpt run is
        # almost always GPU contention (EGL context can't allocate while
        # mjlab subprocess owns the CUDA device). Return 503 Service
        # Unavailable so the frontend retries after the run, instead of
        # 500 which reads as a permanent failure in the UI.
        if e.kind == "render" and jobs.has_any_active_sculpt_run():
            return _problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "preview busy — sculpt run is using the GPU",
                detail=(
                    f"Preview render failed while a sculpt run is active. "
                    f"This is almost always GPU contention — the preview will "
                    f"re-render after the run completes. Underlying error: {e}"
                ),
                type_="/problems/preview-busy",
                render_kind=e.kind,
            )
        status_code = (
            status.HTTP_409_CONFLICT
            if e.kind == "not_configured"
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        type_ = (
            "/problems/state-conflict"
            if e.kind == "not_configured"
            else "/problems/preview-failed"
        )
        return _problem(
            status_code,
            "preview unavailable",
            detail=str(e),
            type_=type_,
            render_kind=e.kind,
        )
    return FileResponse(preview_file, media_type="image/png")


# ── helpers: filename sanitization + zip extraction ───────────────────
def _sanitize_filename(name: str) -> str:
    """Keep alphanumerics, dots, underscores, hyphens. Prevent traversal."""
    import re as _re

    base = Path(name).name  # strip any leading directory
    safe = _re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    safe = safe.strip(".")
    if not safe:
        safe = "model.xml"
    return safe[:120]


class _ZipReject(Exception):
    pass


def _extract_mesh_zip(zip_bytes: bytes, dest: Path) -> list[str]:
    """Extract a mesh zip into `dest`. Returns relative paths of extracted
    files. Raises _ZipReject on any zip-slip, symlink, oversized entry,
    or pathologic compression ratio."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise _ZipReject(f"not a valid zip file: {e}") from e

    dest_resolved = dest.resolve()
    extracted: list[str] = []

    for info in zf.infolist():
        name = info.filename
        if not name or name.endswith("/"):
            continue  # skip directories
        # Path-traversal + absolute-path + symlink checks.
        if name.startswith("/") or ":" in name[:3]:
            raise _ZipReject(f"absolute path in zip: {name!r}")
        if any(part == ".." for part in Path(name).parts):
            raise _ZipReject(f"'..' in zip entry path: {name!r}")
        # symlink bit: external_attr upper 16 bits store POSIX mode.
        posix_mode = (info.external_attr >> 16) & 0xFFFF
        if posix_mode and (posix_mode & 0o170000) == 0o120000:
            raise _ZipReject(f"symlink entries are not allowed: {name!r}")
        if info.file_size > MAX_ZIP_ENTRY_BYTES:
            raise _ZipReject(
                f"entry {name!r} uncompressed size "
                f"{info.file_size} > {MAX_ZIP_ENTRY_BYTES}"
            )
        if (
            info.compress_size > 0
            and info.file_size / max(info.compress_size, 1) > MAX_ZIP_RATIO
        ):
            raise _ZipReject(
                f"entry {name!r} compression ratio "
                f"{info.file_size / max(info.compress_size, 1):.0f}:1 "
                f"> {MAX_ZIP_RATIO}:1 — suspected zip bomb"
            )

        target = (dest / name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as e:
            raise _ZipReject(
                f"zip entry {name!r} resolves outside extraction root"
            ) from e
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 256)
        extracted.append(name.replace("\\", "/"))
    return extracted
