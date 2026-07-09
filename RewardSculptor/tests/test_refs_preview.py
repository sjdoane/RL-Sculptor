"""§R1_BUILD_SPEC W2: reference-clip preview rendering
(`sculptor/refs/preview.py`).

Two kinds of test here:
  1. Pure qpos-assembly logic against a minimal hand-written MJCF (no
     GL/EGL context needed — `build_qpos_frame` only touches
     `model.joint(name).qposadr`, a compile-time model property).
  2. A real end-to-end render attempt against the actual installed
     mjlab G1 MJCF, which SKIPS (not fails) with the `PreviewUnavailable`
     reason if this environment can't create a GL/EGL context — preview
     generation must never be a hard test/ingest dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.robot_manifest import G1_29
from sculptor.refs.preview import (
    N_TILES,
    PreviewUnavailable,
    build_qpos_frame,
    render_keyframe_strip,
    render_preview_png,
    resolve_g1_mjcf,
)

mujoco = pytest.importorskip("mujoco")


# ── minimal hand-written MJCF: free joint + 2 hinge joints ───────────────
_MINI_MJCF = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint name="root_free"/>
      <geom type="sphere" size="0.1"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="alpha_joint" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="beta_joint" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
          <body name="link3" pos="0.2 0 0">
            <joint name="gamma_joint" type="hinge" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def mini_model():
    return mujoco.MjModel.from_xml_string(_MINI_MJCF)


# ── pure qpos assembly (name -> address mapping) ─────────────────────────
def test_build_qpos_frame_places_root_at_0_to_7(mini_model) -> None:
    qpos = build_qpos_frame(
        mini_model,
        root_pos_xy=np.array([1.5, -2.5]),
        root_pos_z=0.42,
        root_quat_wxyz=np.array([0.9, 0.1, 0.2, 0.3]),
        joint_names=["alpha_joint", "beta_joint", "gamma_joint"],
        joint_pos=np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(qpos[0:2], [1.5, -2.5])
    assert qpos[2] == pytest.approx(0.42)
    np.testing.assert_allclose(qpos[3:7], [0.9, 0.1, 0.2, 0.3])


def test_build_qpos_frame_places_joints_by_name_not_position(mini_model) -> None:
    """The MJCF's compile-time joint order is alpha(qposadr=7), beta(8),
    gamma(9) (free joint occupies 0:7) — feed the angles in a
    DELIBERATELY shuffled order and assert each value still lands at
    its OWN qposadr, proving the mapping is by name, not by list
    position."""
    assert int(mini_model.joint("alpha_joint").qposadr[0]) == 7
    assert int(mini_model.joint("beta_joint").qposadr[0]) == 8
    assert int(mini_model.joint("gamma_joint").qposadr[0]) == 9

    qpos = build_qpos_frame(
        mini_model,
        root_pos_xy=np.zeros(2), root_pos_z=0.0,
        root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        # Shuffled: gamma first, then alpha, then beta.
        joint_names=["gamma_joint", "alpha_joint", "beta_joint"],
        joint_pos=np.array([0.111, 0.222, 0.333]),
    )
    assert qpos[7] == pytest.approx(0.222)  # alpha got its own value
    assert qpos[8] == pytest.approx(0.333)  # beta got its own value
    assert qpos[9] == pytest.approx(0.111)  # gamma got its own value


def test_build_qpos_frame_unknown_joint_name_raises(mini_model) -> None:
    with pytest.raises(Exception):  # noqa: B017 — mujoco's own KeyError-ish type
        build_qpos_frame(
            mini_model,
            root_pos_xy=np.zeros(2), root_pos_z=0.0,
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_names=["not_a_real_joint"],
            joint_pos=np.array([0.0]),
        )


# ── G1 MJCF resolution ───────────────────────────────────────────────────
def test_resolve_g1_mjcf_finds_a_real_file() -> None:
    """mjlab must be installed in this venv (§R1_BUILD_SPEC ground
    truth: 'Python for sculptor work' has mujoco/torch/numpy). If this
    fails in some other environment, that's the documented
    PreviewUnavailable path, not a test bug — but the sculptor venv this
    suite runs in DOES have mjlab, so assert it resolves for real."""
    path = resolve_g1_mjcf()
    assert path.is_file()
    assert path.name == "g1.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    # Joint order after the free joint must match G1_29 exactly, and
    # every qposadr must be sequential starting at 7 (§decision 1
    # canonical layout) — the assumption `build_qpos_frame`'s by-name
    # lookup doesn't even need, but worth pinning as a regression guard.
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    assert joint_names[1:] == list(G1_29)
    for offset, name in enumerate(G1_29):
        assert int(model.joint(name).qposadr[0]) == 7 + offset


# ── real end-to-end render — skip gracefully, don't fail, if no GL ──────
def _synthetic_clip() -> dict:
    n = N_TILES + 3  # not an exact multiple of N_TILES, exercise linspace
    z = np.full(n, 0.78)
    xy = np.zeros((n, 2))
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    joints = 0.1 * np.sin(
        np.linspace(0.0, np.pi, n))[:, None] * np.ones((1, len(G1_29)))
    return {
        "root_pos_z": z, "root_pos_xy": xy, "root_quat_wxyz": quat,
        "joint_pos": joints, "joint_names": list(G1_29), "fps": 30.0,
    }


def test_render_keyframe_strip_real_attempt_or_documented_skip(tmp_path: Path) -> None:
    """Build a tiny synthetic ingested-format clip and ACTUALLY attempt
    to render it. If this environment can create a GL/EGL context, the
    strip must be exactly N_TILES tiles wide with the documented tile
    size. If it can't, `PreviewUnavailable` must be raised (never a
    bare mujoco exception) and the test skips with that reason instead
    of failing — preview generation must never block ingest, and by the
    same principle must never block this test suite on a GL-less box."""
    clip = _synthetic_clip()
    try:
        strip = render_keyframe_strip(clip)
    except PreviewUnavailable as e:
        pytest.skip(f"no GL/EGL context in this environment: {e}")
        return
    assert strip.ndim == 3
    assert strip.shape[2] == 3
    assert strip.shape[0] == 180
    assert strip.shape[1] == 180 * N_TILES


def test_render_preview_png_real_attempt_or_documented_skip(tmp_path: Path) -> None:
    clip = _synthetic_clip()
    out_path = tmp_path / "preview.png"
    try:
        result = render_preview_png(clip, out_path)
    except PreviewUnavailable as e:
        pytest.skip(f"no GL/EGL context in this environment: {e}")
        return
    assert result == out_path
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_keyframe_strip_missing_fields_raises_preview_unavailable() -> None:
    """A clip ingested WITHOUT the R1-optional posing keys (root_pos_xy /
    root_quat_wxyz / joint_pos / joint_names — e.g. an older/minimal
    clip) can't be posed; this must raise `PreviewUnavailable` (so
    ingest skips gracefully) rather than a raw KeyError."""
    clip = {
        "root_pos_z": np.full(20, 0.78),
        "fps": 30.0,
    }
    with pytest.raises(PreviewUnavailable):
        render_keyframe_strip(clip)


# ── wired into ingest (§decision 8: "must never block ingest") ──────────
def test_ingest_clip_bytes_generates_preview_when_not_disabled(tmp_path: Path) -> None:
    """`no_preview=False` (the default) makes `ingest_clip_bytes` render
    a real `preview.png` beside the clip. If this box can't create a GL
    context the ingest must still succeed (per `_try_render_preview`'s
    best-effort contract) — assert accordingly either way rather than
    assuming GL availability."""
    import csv
    import io

    from sculptor.refs import library
    from sculptor.refs.ingest import ingest_clip_bytes

    n = 50
    t = np.linspace(0.0, 1.0, n)
    xy = np.zeros((n, 2))
    z = np.full(n, 0.78)
    quat = np.zeros((n, 4))
    quat[:, 3] = 1.0  # xyzw identity
    joints = 0.1 * np.sin(np.outer(t, np.linspace(1, 2, len(G1_29))) * np.pi)
    rows = np.concatenate([xy, z[:, None], quat, joints], axis=1)

    buf = io.StringIO()
    w = csv.writer(buf)
    for row in rows:
        w.writerow([f"{v:.6f}" for v in row])
    raw = buf.getvalue().encode("utf-8")

    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/dance1-3_subject1.csv", stem="dance1-3_subject1",
        robot="g1", root=tmp_path, no_preview=False)

    preview_path = (
        library.clip_dir("g1", result.clip_id, root=tmp_path)
        / library.PREVIEW_FILENAME)
    try:
        resolve_g1_mjcf()
    except PreviewUnavailable:
        pytest.skip("mjlab G1 MJCF not resolvable in this environment")
        return
    # mjlab resolved and mujoco is importable (module-level importorskip
    # already passed) — on THIS box rendering is known to work (proven
    # by the direct render_preview_png tests above), so the preview
    # must actually exist, and the index must reflect it.
    assert preview_path.is_file()
    rows_idx = library.rebuild_index(root=tmp_path)
    row = next(r for r in rows_idx if r["clip_id"] == result.clip_id)
    assert row["has_preview"] is True


def test_ingest_clip_bytes_no_preview_flag_skips_rendering(tmp_path: Path) -> None:
    import csv
    import io

    from sculptor.refs import library
    from sculptor.refs.ingest import ingest_clip_bytes

    n = 50
    rows = np.zeros((n, 36))
    rows[:, 2] = 0.78
    rows[:, 6] = 1.0  # quat w (xyzw index 3 -> col 6)

    buf = io.StringIO()
    w = csv.writer(buf)
    for row in rows:
        w.writerow([f"{v:.6f}" for v in row])
    raw = buf.getvalue().encode("utf-8")

    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/dance1-4_subject1.csv", stem="dance1-4_subject1",
        robot="g1", root=tmp_path, no_preview=True)

    preview_path = (
        library.clip_dir("g1", result.clip_id, root=tmp_path)
        / library.PREVIEW_FILENAME)
    assert not preview_path.is_file()
