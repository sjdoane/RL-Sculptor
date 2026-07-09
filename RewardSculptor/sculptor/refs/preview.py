"""Keyframe-strip PNG previews for reference clips (§R1_BUILD_SPEC
decision 8).

Renders a reference clip (`sculptor.reference` clip dict, root_pos_xy +
root_pos_z + root_quat_wxyz + joint_pos) as a single horizontal-strip PNG
of 12 keyframes (`np.linspace` over T, ~180 px/tile), the same
sampling idiom `_mjlab_runner.py`'s rollout keyframe export uses. The G1
MJCF is resolved programmatically from the installed `mjlab` package
(`mjlab.asset_zoo.robots.unitree_g1.xmls.g1.xml` on this box — see
`resolve_g1_mjcf`'s docstring for how it's found without hardcoding that
path). qpos is built from the free-joint root (xyz + wxyz quat, already
canonical MuJoCo order per §decision 1) and every joint set BY NAME via
`model.joint(name).qposadr` — never a positional/index assumption.

Rendering never blocks ingest: any failure to create an offscreen
GL/EGL context (or resolve the MJCF, or anything else MuJoCo-side)
raises the typed `PreviewUnavailable`, which `ingest.py`/the CLI catch
and log — the clip stays valid, it just has no preview.png.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

N_TILES = 12
TILE_PX = 180


class PreviewUnavailable(Exception):
    """Preview rendering could not run in this environment (no GL/EGL
    context, MJCF unresolvable, etc). Callers (ingest, CLI) catch this
    and skip preview generation — it must never fail a clip ingest."""


# ── G1 MJCF resolution ───────────────────────────────────────────────────
def resolve_g1_mjcf() -> Path:
    """Locate the G1 MJCF shipped inside the installed `mjlab` package.

    Resolved programmatically (not a hardcoded path) by importing
    `mjlab`, walking its package directory for
    `asset_zoo/robots/unitree_g1/xmls/g1.xml`, and falling back to a
    package-tree walk for any `*g1*.xml` under `asset_zoo` if the
    module's internal layout ever moves. Verified against the real
    installed package during this build:
    `<venv>/lib/python3.13/site-packages/mjlab/asset_zoo/robots/
    unitree_g1/xmls/g1.xml` — joint order in that file's 29 hinge
    joints (after the one free joint) matches `G1_29` exactly, and
    every joint's `qposadr` is sequential starting at 7 (verified with
    a real `mujoco.MjModel.from_xml_path` load), consistent with
    §decision-1's canonical layout.
    """
    try:
        import mjlab
    except Exception as e:  # noqa: BLE001
        raise PreviewUnavailable(
            f"mjlab package not importable: {type(e).__name__}: {e}") from e

    pkg_dir = Path(mjlab.__file__).resolve().parent
    candidate = pkg_dir / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
    if candidate.is_file():
        return candidate

    # Layout moved — fall back to a bounded search under asset_zoo for
    # any G1 xml rather than hard-failing on a path that used to work.
    asset_zoo = pkg_dir / "asset_zoo"
    if asset_zoo.is_dir():
        matches = sorted(asset_zoo.rglob("*g1*.xml"))
        if matches:
            return matches[0]

    raise PreviewUnavailable(
        f"could not find a G1 MJCF under {pkg_dir} "
        "(expected asset_zoo/robots/unitree_g1/xmls/g1.xml)")


# ── qpos assembly (pure — unit-testable without a GL context) ───────────
def build_qpos_frame(
    model, *, root_pos_xy: np.ndarray, root_pos_z: float,
    root_quat_wxyz: np.ndarray, joint_names: list[str], joint_pos: np.ndarray,
) -> np.ndarray:
    """Build one `(model.nq,)` qpos vector for a single frame.

    Root free-joint qpos[0:7] = [x, y, z, qw, qx, qy, qz] (MuJoCo
    convention — `root_quat_wxyz` is already in that order per
    §decision 1, no conversion here). Every joint angle is written to
    `model.joint(name).qposadr[0]` — resolved BY NAME each time, never
    by a positional/index assumption, so a model whose joint order
    doesn't match `joint_names`' order still lands correctly (and a
    genuinely absent name raises `KeyError` from mujoco's name lookup
    rather than silently misplacing a value).
    """
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:2] = root_pos_xy
    qpos[2] = root_pos_z
    qpos[3:7] = root_quat_wxyz
    for name, angle in zip(joint_names, joint_pos):
        adr = int(model.joint(name).qposadr[0])
        qpos[adr] = float(angle)
    return qpos


# ── rendering ─────────────────────────────────────────────────────────
def _render_frame(model, data, *, size: int = TILE_PX) -> np.ndarray:
    import mujoco

    with mujoco.Renderer(model, height=size, width=size) as renderer:
        renderer.update_scene(data, camera=_build_fit_camera(model, data))
        return np.asarray(renderer.render(), dtype=np.uint8)


def _build_fit_camera(model, data):
    """Bounding-box-fit free camera over non-plane geoms — same idiom as
    `reward-sculptor-ui/backend/services/preview_renderer.py`'s
    `_build_camera` (kept independent/duplicated rather than imported:
    that module lives in the backend venv, not the sculptor venv this
    package runs in)."""
    import mujoco

    cam = mujoco.MjvCamera()
    geom_pos = np.array(data.geom_xpos)
    geom_size = np.array(model.geom_size)
    geom_type = np.array(model.geom_type)
    non_plane = geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    if non_plane.any():
        pos = geom_pos[non_plane]
        half = geom_size[non_plane].max(axis=1)
        lo = (pos - half[:, None]).min(axis=0)
        hi = (pos + half[:, None]).max(axis=0)
    else:
        lo = np.array([-0.5, -0.5, 0.0])
        hi = np.array([0.5, 0.5, 1.0])
    span = max(float(np.max(hi - lo)), 0.5)
    center = (lo + hi) / 2.0
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 45.0
    cam.elevation = -20.0
    cam.distance = span * 2.2
    cam.lookat[:] = center.tolist()
    return cam


def _keyframe_indices(n_frames: int, n_tiles: int = N_TILES) -> np.ndarray:
    return np.linspace(0, max(0, n_frames - 1), num=min(n_tiles, n_frames)).astype(int)


def render_keyframe_strip(
    clip: dict, *, mjcf_path: Optional[Path] = None, tile_px: int = TILE_PX,
) -> np.ndarray:
    """Render a clip to an `(tile_px, tile_px * n_tiles, 3)` uint8 strip.

    Raises `PreviewUnavailable` on ANY failure: MJCF unresolvable,
    mujoco import failure, GL/EGL context creation failure, or a clip
    missing the fields needed to pose the robot (`root_pos_xy`,
    `root_quat_wxyz`, `joint_pos`, `joint_names` — all §R1 optional
    clip keys; a clip ingested without them simply can't be posed).
    """
    for key in ("root_pos_xy", "root_quat_wxyz", "joint_pos", "joint_names"):
        if clip.get(key) is None:
            raise PreviewUnavailable(
                f"clip is missing {key!r} — cannot pose the robot for preview")

    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        raise PreviewUnavailable(
            f"mujoco import failed: {type(e).__name__}: {e}") from e

    xml_path = mjcf_path or resolve_g1_mjcf()
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as e:  # noqa: BLE001
        raise PreviewUnavailable(
            f"MuJoCo failed to load {xml_path}: {type(e).__name__}: {e}") from e

    z = clip["root_pos_z"]
    idxs = _keyframe_indices(int(z.shape[0]))
    joint_names = list(clip["joint_names"])
    tiles: list[np.ndarray] = []
    try:
        for i in idxs:
            qpos = build_qpos_frame(
                model,
                root_pos_xy=clip["root_pos_xy"][i],
                root_pos_z=float(z[i]),
                root_quat_wxyz=clip["root_quat_wxyz"][i],
                joint_names=joint_names,
                joint_pos=clip["joint_pos"][i],
            )
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            tiles.append(_render_frame(model, data, size=tile_px))
    except PreviewUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 — GL/EGL context or render failure
        raise PreviewUnavailable(
            f"MuJoCo render failed: {type(e).__name__}: {e}") from e

    return np.concatenate(tiles, axis=1)


def render_preview_png(
    clip: dict, out_path: Path | str, *, mjcf_path: Optional[Path] = None,
) -> Path:
    """Render + encode the keyframe strip to `out_path` (PNG). Raises
    `PreviewUnavailable` on any failure — caller decides whether that's
    fatal (it should not be, for ingest)."""
    strip = render_keyframe_strip(clip, mjcf_path=mjcf_path)
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        raise PreviewUnavailable(
            f"PIL/Pillow not available for PNG encode: {type(e).__name__}: {e}") from e
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path, format="PNG")
    return out_path
