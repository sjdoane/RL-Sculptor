"""Static-preview renderer.

Two entry points depending on the project's robot source:

- **Library picks** → `gymnasium.make(env_id, render_mode="rgb_array")`.
  This matches how sculptor renders rollouts, so if sculpt run works
  the preview works.
- **Uploaded MJCF/URDF** → `mujoco.MjModel.from_xml_path` +
  `mujoco.Renderer`. For .urdf, two parse attempts are tried before
  giving up, and the error message names which ones failed so users
  have something actionable.

All rendering happens inside `asyncio.to_thread` — MuJoCo's C++
renderer holds the GIL. The rendered frame is encoded to PNG and
cached at `<project>/uploads/preview.png`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np


PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Camera presets. Library-card thumbnails always use `iso`; the preview
# panel's angle dropdown calls through with any of these. Distances are
# scaled by the model's bounding-box size so small + large robots both
# frame up reasonably.
CAMERA_ANGLES: dict[str, dict[str, float]] = {
    "iso":   {"azimuth":  45.0, "elevation": -20.0, "distance_mul": 2.5},
    "front": {"azimuth":  90.0, "elevation": -10.0, "distance_mul": 2.5},
    "side":  {"azimuth":   0.0, "elevation": -10.0, "distance_mul": 2.5},
    "top":   {"azimuth":   0.0, "elevation": -89.0, "distance_mul": 3.5},
}
DEFAULT_ANGLE = "iso"


def preview_path(project_dir: Path, angle: str = DEFAULT_ANGLE) -> Path:
    return project_dir / "uploads" / f"preview_{angle}.png"


class PreviewError(Exception):
    """Raised when the preview can't be rendered. Carries a `kind` tag so
    routes can pick an appropriate HTTP status + problem type."""

    def __init__(self, message: str, *, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


# ── public API ────────────────────────────────────────────────────────
def resolve_library_xml(env_id: str) -> Path | None:
    """Resolve a Gymnasium env_id like 'Hopper-v4' to its MJCF asset path.
    None if the env isn't one of the bundled MuJoCo ones."""
    try:
        import gymnasium as gym
    except Exception:  # noqa: BLE001
        return None
    stem = env_id.split("-")[0]
    asset_dir = Path(gym.__file__).resolve().parent / "envs" / "mujoco" / "assets"
    candidates = [
        asset_dir / f"{stem.lower()}.xml",
        asset_dir / f"{_camel_to_snake(stem)}.xml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _camel_to_snake(s: str) -> str:
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def render_static(
    project_dir: Path,
    robot_source: dict[str, Any],
    angle: str = DEFAULT_ANGLE,
    *,
    size: tuple[int, int] = (PREVIEW_WIDTH, PREVIEW_HEIGHT),
    out_path: Path | None = None,
) -> Path:
    """Render a PNG preview synchronously at the given camera angle.

    `robot_source` shapes:
      - library: `{"kind": "library", "library_name": "hopper", "env_id": "Hopper-v4"}`
      - upload:  `{"kind": "mjcf"|"urdf", "model_file": "uploads/robot/foo.xml", ...}`

    Returns the PNG path. Defaults to `<project>/uploads/preview_<angle>.png`;
    override with `out_path` (used by the thumbnail generator).
    """
    if angle not in CAMERA_ANGLES:
        raise PreviewError(
            f"unknown camera angle {angle!r}; "
            f"expected one of {sorted(CAMERA_ANGLES)}",
            kind="bad_angle",
        )

    kind = robot_source.get("kind")
    if kind is None or kind == "none":
        raise PreviewError(
            "no robot configured for this project", kind="not_configured"
        )

    if kind == "library":
        # `library_slug` is the canonical identifier written by the
        # library-driven project-create flow (`_finalize_library_scaffold`).
        # `library_name` + `env_id` are the legacy fields from the
        # POST /robot/library route. Either may be populated; a post-
        # library-slug project generally has slug + menagerie_package
        # but no env_id (mjlab uses task_id in config.toml, not env_id).
        library_slug = robot_source.get("library_slug")
        library_name = robot_source.get("library_name")
        env_id = robot_source.get("env_id")

        # sculpt_init stamps kind="library" with no slug/name until the
        # user picks a robot; gym env_id template value is "CHANGE_ME".
        nothing_configured = (
            not library_slug
            and not library_name
            and env_id in (None, "", "CHANGE_ME")
        )
        if nothing_configured:
            raise PreviewError(
                "no robot has been chosen yet — pick one from the library "
                "or upload a URDF/MJCF.",
                kind="not_configured",
            )

        # Two resolution paths, tried in order:
        #   1. Library-slug + Menagerie (mjlab_ready + preview_only library
        #      entries; new project-create flow). Returns None if the
        #      entry isn't menagerie-sourced.
        #   2. Gymnasium bundled asset by env_id (legacy gym_sb3 path).
        xml_path: Path | None = None
        slug_to_resolve = library_slug or library_name
        if slug_to_resolve:
            try:
                from backend.services.robot_library import get_library

                xml_path = get_library().resolve_menagerie_path(slug_to_resolve)
            except Exception:  # noqa: BLE001 — fall through to the gym path
                xml_path = None
        if xml_path is None and env_id not in (None, "", "CHANGE_ME"):
            xml_path = resolve_library_xml(env_id)
        if xml_path is None:
            raise PreviewError(
                f"no MuJoCo asset found for "
                f"library_slug={library_slug!r} / env_id={env_id!r}. "
                "For Menagerie robots, confirm `robot_descriptions` is "
                "installed and the library entry's menagerie_package is "
                "valid; for gym robots, confirm env_id matches a bundled "
                "gymnasium.envs.mujoco asset.",
                kind="env_make",
            )
    elif kind in ("mjcf", "urdf"):
        model_rel = robot_source.get("model_file")
        if not model_rel:
            raise PreviewError(
                f"{kind} robot is selected but no model_file is recorded",
                kind="not_configured",
            )
        xml_path = project_dir / model_rel
    else:
        raise PreviewError(
            f"unknown robot kind {kind!r}", kind="bad_kind"
        )

    frame = _render_mjcf(xml_path, angle=angle, size=size)

    out = out_path or preview_path(project_dir, angle)
    out.parent.mkdir(parents=True, exist_ok=True)
    _encode_png(frame, out)
    return out


# §7.6: serialize preview renders across the backend process. Two
# concurrent `mujoco.Renderer` constructions can race on the EGL
# context, especially on WSL2's software-EGL path — leaving the second
# request stuck in a half-initialized state until the first releases.
# An asyncio.Lock held across `asyncio.to_thread(render_static, ...)`
# costs ~500 ms of extra serial latency in the worst case (still faster
# than the render itself) and eliminates the contention-500 class of
# failures Sam hit during the cartwheel run (CONTEXT 00:15, symptom 1
# of project-tab breakage).
_PREVIEW_RENDER_LOCK: asyncio.Lock | None = None


def _get_preview_lock() -> asyncio.Lock:
    """Lazy-init so the Lock is bound to the current event loop. Creating
    an `asyncio.Lock` at module import time would tie it to whichever
    loop was running then, which breaks in test suites that spin up
    fresh loops per case."""
    global _PREVIEW_RENDER_LOCK
    if _PREVIEW_RENDER_LOCK is None:
        _PREVIEW_RENDER_LOCK = asyncio.Lock()
    return _PREVIEW_RENDER_LOCK


async def render_static_async(
    project_dir: Path,
    robot_source: dict[str, Any],
    angle: str = DEFAULT_ANGLE,
) -> Path:
    async with _get_preview_lock():
        return await asyncio.to_thread(
            render_static, project_dir, robot_source, angle
        )


# ── implementation ────────────────────────────────────────────────────
def _posed_data(model: Any) -> Any:
    """MjData posed for a legible still: keyframe 0 when the model has
    one (robot assets and compiled evaluation scenes carry an
    init/home keyframe), else the model default. Without this, a
    quadruped renders at qpos0 with fully extended legs — tall, spindly,
    and easy to mistake for a humanoid at thumbnail size."""
    import mujoco

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _render_mjcf(
    xml_path: Path,
    angle: str = DEFAULT_ANGLE,
    size: tuple[int, int] = (PREVIEW_WIDTH, PREVIEW_HEIGHT),
) -> np.ndarray:
    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"mujoco import failed: {type(e).__name__}: {e}",
            kind="mujoco_import",
        ) from e
    if not xml_path.is_file():
        raise PreviewError(
            f"model file not found: {xml_path}", kind="model_missing"
        )
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"MuJoCo failed to parse {xml_path.name}: "
            f"{type(e).__name__}: {e}",
            kind="parse",
        ) from e
    try:
        data = _posed_data(model)
        camera = _build_camera(model, data, angle)
        width, height = size
        with mujoco.Renderer(
            model, height=height, width=width
        ) as renderer:
            renderer.update_scene(data, camera)
            frame = renderer.render()
    except PreviewError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"MuJoCo render failed for {xml_path.name}: "
            f"{type(e).__name__}: {e}",
            kind="render",
        ) from e
    return np.asarray(frame, dtype=np.uint8)


def render_model_preview(
    model_path: Path,
    out_path: Path,
    angle: str = DEFAULT_ANGLE,
    *,
    size: tuple[int, int] = (PREVIEW_WIDTH, PREVIEW_HEIGHT),
) -> Path:
    """Render a compiled model file (.mjb binary or MJCF .xml) to a PNG.

    Used by the authored-world preview: the materialized evaluation MJB
    embeds terrain heightfields, objects, zones, and the robot, so the
    render shows exactly the scene evaluation replays and scores."""
    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"mujoco import failed: {type(e).__name__}: {e}",
            kind="mujoco_import",
        ) from e
    if not model_path.is_file():
        raise PreviewError(
            f"model file not found: {model_path}", kind="model_missing")
    try:
        if model_path.suffix == ".mjb":
            model = mujoco.MjModel.from_binary_path(str(model_path))
        else:
            model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"MuJoCo failed to load {model_path.name}: "
            f"{type(e).__name__}: {e}",
            kind="parse",
        ) from e
    try:
        data = _posed_data(model)
        camera = _build_camera(model, data, angle)
        width, height = size
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera)
            frame = renderer.render()
    except PreviewError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"MuJoCo render failed for {model_path.name}: "
            f"{type(e).__name__}: {e}",
            kind="render",
        ) from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _encode_png(np.asarray(frame, dtype=np.uint8), out_path)
    return out_path


def _build_camera(model: Any, data: Any, angle: str) -> Any:
    """Build a free MjvCamera framed on the robot.

    Compute the bounding box over the NON-PLANE geoms in world frame
    (after mj_forward) so an arena floor doesn't blow up the scale.
    Fall back to body COM positions if the model has no non-plane
    geoms for some reason.
    """
    import mujoco
    import numpy as np

    cfg = CAMERA_ANGLES[angle]

    geom_positions = np.array(data.geom_xpos)       # world-frame, (ngeom, 3)
    geom_sizes = np.array(model.geom_size)          # half-sizes/radii, (ngeom, 3)
    geom_types = np.array(model.geom_type)

    non_plane = geom_types != mujoco.mjtGeom.mjGEOM_PLANE
    if non_plane.any():
        pos = geom_positions[non_plane]
        half = geom_sizes[non_plane].max(axis=1)
        lo = (pos - half[:, None]).min(axis=0)
        hi = (pos + half[:, None]).max(axis=0)
    elif model.nbody > 1:
        # All-plane models are weird but tolerated — use body COMs.
        body_pos = np.array(data.xipos[1:model.nbody])
        lo = body_pos.min(axis=0) - 0.25
        hi = body_pos.max(axis=0) + 0.25
    else:
        lo = np.array([-0.5, -0.5, 0.0])
        hi = np.array([0.5, 0.5, 1.0])

    span = float(np.max(hi - lo))
    span = max(span, 0.5)  # don't zoom in too tight on tiny robots

    # Lookat = bbox center lifted a bit toward the robot's COM if
    # there's a non-world body. Keeps the eye aimed at the subject
    # rather than at the ground between its feet.
    center = (lo + hi) / 2.0
    if model.nbody > 1:
        body_com = np.array(data.xipos[1])
        center[2] = (center[2] + body_com[2]) / 2.0

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = cfg["azimuth"]
    cam.elevation = cfg["elevation"]
    cam.distance = span * cfg["distance_mul"]
    cam.lookat[:] = center.tolist()
    return cam


def _encode_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path, format="PNG")


# ── MJCF / URDF parse validation (used by robot-upload route) ─────────
def validate_model_file(xml_path: Path) -> dict[str, Any]:
    """Verify the file is parseable by MuJoCo. Returns a small summary
    dict on success. Raises PreviewError(kind="parse" | "urdf_parse")
    with an actionable message on failure.

    For .urdf files, two parse paths are tried in order:
      1. `mujoco.MjModel.from_xml_path` — MuJoCo 3.x parses URDF directly.
      2. `mujoco.MjSpec.from_file` + `spec.compile()` — URDF→MJCF
         conversion via the newer spec API.

    If both fail, the error message names both attempts so the user
    can see which capability is missing.
    """
    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        raise PreviewError(
            f"mujoco import failed: {type(e).__name__}: {e}",
            kind="mujoco_import",
        ) from e

    if not xml_path.is_file():
        raise PreviewError(
            f"model file not found: {xml_path}", kind="model_missing"
        )

    is_urdf = xml_path.suffix.lower() == ".urdf"

    # Path 1: direct load (works for MJCF always, URDF in 3.x+)
    direct_err: str | None = None
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        return {
            "ok": True,
            "path": "direct",
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
        }
    except Exception as e:  # noqa: BLE001
        direct_err = f"{type(e).__name__}: {e}"

    if not is_urdf:
        # MJCF failed — this is a parse error, surface verbatim.
        raise PreviewError(
            f"MuJoCo rejected {xml_path.name}: {direct_err}",
            kind="parse",
        )

    # Path 2: MjSpec (URDF-specific conversion path in MuJoCo 3.x)
    spec_err: str | None = None
    if hasattr(mujoco, "MjSpec") and hasattr(mujoco.MjSpec, "from_file"):
        try:
            spec = mujoco.MjSpec.from_file(str(xml_path))
            model = spec.compile()
            return {
                "ok": True,
                "path": "mjspec",
                "nbody": int(model.nbody),
                "njnt": int(model.njnt),
                "ngeom": int(model.ngeom),
            }
        except Exception as e:  # noqa: BLE001
            spec_err = f"{type(e).__name__}: {e}"
    else:
        spec_err = (
            "mujoco.MjSpec.from_file is unavailable in this MuJoCo version "
            f"({getattr(mujoco, '__version__', 'unknown')}); "
            "URDF→MJCF conversion requires MuJoCo >= 3.0."
        )

    # Both URDF paths failed. Name them specifically so the user knows
    # what to convert externally.
    raise PreviewError(
        f"URDF parsing failed for {xml_path.name}. "
        f"Tried (1) MjModel.from_xml_path: {direct_err}. "
        f"Tried (2) MjSpec.from_file: {spec_err}. "
        "Convert the URDF to MJCF externally (e.g. using MuJoCo's "
        "`mjpython` compiler or `obj2mjcf`) and re-upload the resulting "
        ".xml file.",
        kind="urdf_parse",
    )
