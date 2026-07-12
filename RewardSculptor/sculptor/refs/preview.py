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
from typing import Callable, Optional

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


# ── T1 MJCF resolution (§Problem 2, 2026-07-11) ──────────────────────────
#: Default GMR checkout root — kept as an independent constant (not
#: imported from `sculptor.refs.retarget.DEFAULT_GMR_ROOT`) so this
#: module has no import-time dependency on the retarget module, mirroring
#: `_build_fit_camera`'s "kept independent/duplicated rather than
#: imported" precedent above. Same value either way: `~/tools/GMR`.
_DEFAULT_GMR_ROOT = Path.home() / "tools" / "GMR"


def resolve_t1_mjcf(*, gmr_root: Optional[Path] = None) -> Path:
    """Locate the Booster T1 (29dof) MJCF that GMR itself retargets
    against — R3's cross-robot proof used this exact asset.

    Unlike G1, mjlab ships NO Booster T1 asset at all: verified
    empirically against the installed package, `mjlab.asset_zoo.robots`
    contains only `unitree_g1`, `unitree_go1`, `i2rt_yam` — no t1/booster
    entry — so there is no mjlab-internal MJCF to fall back to. The only
    verified source is the GMR checkout itself (`~/tools/GMR`, the same
    tree `sculptor.refs.retarget` shells out to for the actual
    retargeting): its `general_motion_retargeting/params.py` defines
    `ROBOT_XML_DICT["booster_t1_29dof"] = ASSET_ROOT /
    "booster_t1_29dof" / "t1_mocap.xml"` (`ASSET_ROOT = <gmr_root>/
    assets`) — read directly off that file during this build (never
    imported: GMR is a separate py3.10 venv/package the sculptor py3.13
    venv does not, and per `retarget.py`'s module docstring must not,
    co-install). Verified against the real checkout on this box: the
    file exists, loads under `mujoco.MjModel.from_xml_path` with exactly
    28 joints (1 free joint + 27 named hinges — matches our T1 clips'
    `joint_names` count exactly; a naive text grep for `joint name="..."`
    over-counts to 29 because the XML also has 2 commented-out
    alternate-range `<!-- <joint name="..."/> -->` lines for the two
    Elbow_Yaw joints, which `mujoco.MjModel.from_xml_path` correctly
    ignores), and its named joints are exactly the set our T1 clips use
    (see `_gmr_driver.py:_robot_joint_names`).

    Raises `PreviewUnavailable` (never blocks ingest/retarget) if no GMR
    checkout is found at `gmr_root` (default `~/tools/GMR`) — this is a
    local-dev-machine dependency, not something every environment has."""
    root = gmr_root or _DEFAULT_GMR_ROOT
    candidate = root / "assets" / "booster_t1_29dof" / "t1_mocap.xml"
    if candidate.is_file():
        return candidate
    raise PreviewUnavailable(
        f"could not find the Booster T1 MJCF at {candidate} — expected a "
        f"GMR checkout under {root} (general_motion_retargeting/assets/"
        f"booster_t1_29dof/t1_mocap.xml); mjlab ships no T1 asset at all "
        f"so there is no fallback search location")


#: robot slug -> zero-arg MJCF resolver. `"g1_hands"` intentionally maps
#: to the plain G1 resolver (pre-existing behavior, unchanged by this
#: fix — mjlab's asset_zoo has no separate with-hands G1 asset either,
#: so there is nothing better to resolve to yet; a future fix would add
#: its own entry here rather than touch this dict's shape) so this dict
#: only changes behavior for `"t1"`, which previously silently fell
#: through to `resolve_g1_mjcf()` too (posing T1 joint angles on a G1
#: skeleton — always failed, just with an opaque MuJoCo joint-lookup
#: error instead of an actionable one).
_MJCF_RESOLVERS: dict[str, Callable[[], Path]] = {
    "g1": resolve_g1_mjcf,
    "g1_hands": resolve_g1_mjcf,
    "t1": resolve_t1_mjcf,
}


def resolve_mjcf_for_robot(robot: str) -> Path:
    """Resolve the MJCF to pose `robot`'s clips for preview rendering.
    Robot-symmetric dispatch (§Problem 2): every caller that used to
    unconditionally default to `resolve_g1_mjcf()` regardless of which
    robot the clip actually belongs to should call this instead.

    A robot with no registered resolver raises `PreviewUnavailable` with
    a precise, actionable reason — never silently falls back to G1 (that
    would pose the wrong skeleton's joint angles and either crash with a
    confusing MuJoCo error or, worse, mis-render)."""
    resolver = _MJCF_RESOLVERS.get(robot)
    if resolver is None:
        raise PreviewUnavailable(
            f"no MJCF resolver registered for robot={robot!r} — preview "
            f"skipped (clip remains valid with no preview.png); known "
            f"robots: {sorted(_MJCF_RESOLVERS)}")
    return resolver()


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
