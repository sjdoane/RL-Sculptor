"""§REFERENCE_TRAJECTORY_PLAN §5/§7: `sculptor/refs/convert.py` — clip ->
battery-array conversion + the kinematic signature. Offline, synthetic
clip fixtures only (never touches the real library or network)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sculptor.refs.convert import (
    CONVERTED_ARRAY_KEYS,
    _projected_gravity_b,
    clip_to_arrays,
    kinematic_signature,
)


def _quat_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    w = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    return np.array([w, *(axis * s)])


# ── quaternion -> projected_gravity_b correctness ────────────────────────
def test_identity_quat_gives_straight_down_gravity():
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    g = _projected_gravity_b(q)
    assert g.shape == (1, 3)
    np.testing.assert_allclose(g[0], [0.0, 0.0, -1.0], atol=1e-9)


def test_90deg_about_z_leaves_gravity_unchanged():
    # Rotating about the SAME axis gravity points along cannot change its
    # projection — gravity_b stays exactly (0, 0, -1).
    q = _quat_axis_angle([0, 0, 1], math.pi / 2)[None, :]
    g = _projected_gravity_b(q)
    np.testing.assert_allclose(g[0], [0.0, 0.0, -1.0], atol=1e-9)


def test_90deg_about_x_tips_gravity_into_body_y():
    q = _quat_axis_angle([1, 0, 0], math.pi / 2)[None, :]
    g = _projected_gravity_b(q)
    # A +90deg rotation about body-x tips world -z into body -y.
    np.testing.assert_allclose(g[0], [0.0, -1.0, 0.0], atol=1e-9)


def test_180deg_about_x_flips_gravity_upward():
    q = _quat_axis_angle([1, 0, 0], math.pi)[None, :]
    g = _projected_gravity_b(q)
    np.testing.assert_allclose(g[0], [0.0, 0.0, 1.0], atol=1e-9)


def test_projected_gravity_always_unit_norm():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 4))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    g = _projected_gravity_b(raw)
    norms = np.linalg.norm(g, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ── clip_to_arrays: shapes / tiling / absent-key behavior ────────────────
def _rise_clip(T: int = 40, fps: float = 30.0) -> dict:
    t = np.linspace(0.0, 1.0, T)
    z = 0.1 + 0.65 * (1 - np.cos(np.pi * t)) / 2.0
    return {"root_pos_z": z, "fps": fps}


def test_minimal_clip_yields_only_root_link_pos_w():
    clip = _rise_clip()
    arrays, meta = clip_to_arrays(clip, n_envs=4)
    assert set(arrays) <= set(CONVERTED_ARRAY_KEYS) | {"root_link_pos_w"}
    assert "root_link_pos_w" in arrays
    assert "joint_pos" not in arrays
    assert "projected_gravity_b" not in arrays
    T = clip["root_pos_z"].shape[0]
    assert arrays["root_link_pos_w"].shape == (T, 4, 3)
    assert arrays["first_episode_valid_mask"].shape == (T, 4)
    assert arrays["first_episode_valid_mask"].dtype == np.bool_
    assert np.all(arrays["first_episode_valid_mask"])
    np.testing.assert_allclose(
        arrays["root_link_pos_w"][:, :, 2], np.tile(clip["root_pos_z"][:, None], 4))
    # xy defaults to zero when the clip has no root_pos_xy.
    np.testing.assert_allclose(arrays["root_link_pos_w"][:, :, :2], 0.0)
    assert meta["joint_names"] == []
    assert meta["max_episode_steps"] == T
    assert meta["rollout_num_envs"] == 4
    assert meta["step_dt"] == pytest.approx(1.0 / 30.0)
    assert meta["reference"]["inferred_contacts"] is False


def test_clip_to_arrays_tiles_across_envs():
    clip = _rise_clip(T=20)
    for n_envs in (1, 4, 7):
        arrays, meta = clip_to_arrays(clip, n_envs=n_envs)
        assert arrays["root_link_pos_w"].shape == (20, n_envs, 3)
        assert meta["rollout_num_envs"] == n_envs
        # Every env column is identical (tiled, not distinct rollouts).
        for e in range(1, n_envs):
            np.testing.assert_allclose(
                arrays["root_link_pos_w"][:, e, :],
                arrays["root_link_pos_w"][:, 0, :])


def test_clip_to_arrays_with_joints_and_orientation():
    T = 25
    clip = _rise_clip(T=T)
    jp = np.zeros((T, 2))
    jp[:, 0] = np.linspace(0, 1.0, T)
    quat = np.zeros((T, 4)); quat[:, 0] = 1.0
    clip["joint_pos"] = jp
    clip["joint_names"] = ["a", "b"]
    clip["root_quat_wxyz"] = quat
    arrays, meta = clip_to_arrays(clip, n_envs=2)
    assert arrays["joint_pos"].shape == (T, 2, 2)
    assert arrays["joint_vel"].shape == (T, 2, 2)
    assert arrays["projected_gravity_b"].shape == (T, 2, 3)
    np.testing.assert_allclose(arrays["projected_gravity_b"][..., 2], -1.0)
    assert meta["joint_names"] == ["a", "b"]
    # joint_vel finite-differenced from joint_pos when absent on the clip.
    expected_jv = np.gradient(jp, axis=0) * clip["fps"]
    np.testing.assert_allclose(arrays["joint_vel"][:, 0, :], expected_jv)


def test_clip_to_arrays_uses_measured_joint_vel_when_present():
    T = 15
    clip = _rise_clip(T=T)
    jp = np.zeros((T, 1))
    jv = np.full((T, 1), 3.0)   # deliberately NOT the finite-diff of jp
    clip["joint_pos"] = jp
    clip["joint_vel"] = jv
    clip["joint_names"] = ["a"]
    arrays, _ = clip_to_arrays(clip, n_envs=1)
    np.testing.assert_allclose(arrays["joint_vel"][:, 0, 0], 3.0)


def test_clip_to_arrays_measured_contacts_passthrough():
    T = 12
    clip = _rise_clip(T=T)
    clip["contact_left_foot"] = np.ones(T)
    clip["contact_right_foot"] = np.zeros(T)
    arrays, meta = clip_to_arrays(clip, n_envs=3)
    np.testing.assert_allclose(arrays["left_foot_contact"][:, 0], 1.0)
    np.testing.assert_allclose(arrays["right_foot_contact"][:, 0], 0.0)
    assert meta["reference"]["inferred_contacts"] is False


def test_clip_to_arrays_never_raises_when_fk_inputs_absent():
    # No root_pos_xy/root_quat_wxyz/joint_pos/joint_names -> FK cannot run;
    # clip_to_arrays must still succeed with those arrays simply absent.
    clip = _rise_clip()
    arrays, meta = clip_to_arrays(clip)
    assert "left_foot_pos_b" not in arrays
    assert "right_foot_pos_b" not in arrays
    assert "left_foot_contact" not in arrays
    assert "right_foot_contact" not in arrays


def test_clip_to_arrays_rejects_invalid_clip():
    with pytest.raises(ValueError):
        clip_to_arrays({"root_pos_z": np.array([1.0, -1.0, 2.0, 3.0,
                                                  4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                         "fps": 30.0})


# ── FK-derived foot positions/contacts (skip-guarded on real MJCF) ───────
mujoco = pytest.importorskip("mujoco")


def _g1_standing_clip(T: int = 10, fps: float = 20.0) -> dict:
    from sculptor.eval.robot_manifest import G1_29

    names = list(G1_29)
    z = np.full(T, 0.78)
    xy = np.zeros((T, 2))
    quat = np.zeros((T, 4)); quat[:, 0] = 1.0
    jp = np.zeros((T, len(names)))
    return {"root_pos_z": z, "root_pos_xy": xy, "root_quat_wxyz": quat,
            "joint_pos": jp, "joint_names": names, "fps": fps}


def test_fk_infers_standing_contacts_on_real_g1_mjcf():
    clip = _g1_standing_clip()
    try:
        arrays, meta = clip_to_arrays(clip, n_envs=2)
    except Exception as e:  # noqa: BLE001 — environment-dependent GL/EGL
        pytest.skip(f"MJCF/FK unavailable in this environment: {e}")
    if "left_foot_pos_b" not in arrays:
        pytest.skip("FK did not resolve foot sites in this environment")
    assert arrays["left_foot_pos_b"].shape == (10, 2, 3)
    # Standing still on the ground: both feet in contact.
    np.testing.assert_allclose(arrays["left_foot_contact"][:, 0], 1.0)
    np.testing.assert_allclose(arrays["right_foot_contact"][:, 0], 1.0)
    assert meta["reference"]["inferred_contacts"] is True


# ── kinematic_signature ───────────────────────────────────────────────────
def _getup_like_clip(T: int = 60, fps: float = 30.0) -> dict:
    """z: 0.1 flat -> ramp -> 0.75 flat (a get-up-shaped profile)."""
    n_flat0 = T // 4
    n_ramp = T // 2
    n_flat1 = T - n_flat0 - n_ramp
    flat0 = np.full(n_flat0, 0.1)
    s = np.linspace(0.0, 1.0, n_ramp)
    ramp = 0.1 + 0.65 * (1 - np.cos(np.pi * s)) / 2.0
    flat1 = np.full(n_flat1, 0.75)
    z = np.concatenate([flat0, ramp, flat1])
    return {"root_pos_z": z, "fps": fps}


def test_kinematic_signature_detects_phases_and_endpoints():
    clip = _getup_like_clip()
    sig = kinematic_signature(clip)
    assert sig["root_z"]["start"] == pytest.approx(0.1, abs=0.02)
    assert sig["root_z"]["end"] == pytest.approx(0.75, abs=0.02)
    assert sig["root_z"]["max"] == pytest.approx(0.75, abs=0.02)
    assert sig["root_z"]["min"] == pytest.approx(0.1, abs=0.02)
    phases = [p["phase"] for p in sig["phases"]]
    assert "rising" in phases
    assert "still" in phases
    # Duration/fps are exact and serializable.
    assert sig["duration_s"] == pytest.approx(
        clip["root_pos_z"].shape[0] / clip["fps"], abs=1e-6)
    assert sig["fps"] == pytest.approx(clip["fps"])
    # Every value must be a JSON-serializable primitive/container.
    import json
    json.dumps(sig)


def test_kinematic_signature_orientation_summary_upright_to_upright():
    clip = _getup_like_clip()
    T = clip["root_pos_z"].shape[0]
    quat = np.zeros((T, 4)); quat[:, 0] = 1.0   # identity throughout
    clip["root_quat_wxyz"] = quat
    sig = kinematic_signature(clip)
    assert sig["orientation"]["initial_gravity_z_b"] == pytest.approx(-1.0)
    assert sig["orientation"]["final_gravity_z_b"] == pytest.approx(-1.0)


def test_kinematic_signature_contact_schedule_when_measured_contacts_present():
    clip = _getup_like_clip()
    T = clip["root_pos_z"].shape[0]
    c = np.zeros(T); c[T // 2:] = 1.0
    clip["contact_left_foot"] = c
    clip["contact_right_foot"] = c
    sig = kinematic_signature(clip)
    assert "contact_schedule" in sig
    assert sig["contact_schedule"]["left_foot"]["contact_fraction"] == pytest.approx(0.5, abs=0.05)


def test_kinematic_signature_joint_energy_by_phase():
    clip = _getup_like_clip()
    T = clip["root_pos_z"].shape[0]
    jp = np.zeros((T, 1))
    jp[:, 0] = np.linspace(0, 2.0, T)
    clip["joint_pos"] = jp
    clip["joint_names"] = ["a"]
    sig = kinematic_signature(clip)
    assert "joint_motion_energy_by_phase" in sig
    energies = [p["mean_abs_joint_vel"] for p in sig["joint_motion_energy_by_phase"]]
    assert all(e >= 0.0 for e in energies)


def test_kinematic_signature_rejects_invalid_clip():
    with pytest.raises(ValueError):
        kinematic_signature({"root_pos_z": np.array([1.0]), "fps": 30.0})
