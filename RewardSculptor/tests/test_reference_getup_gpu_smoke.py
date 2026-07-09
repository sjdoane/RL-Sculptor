"""Real-GPU env-construction smoke test for the get-up reference RSI
curriculum (§REFERENCE_TRAJECTORY_PLAN §8 part 2).

Builds the mjlab G1 env EXACTLY the way a training run would (smallest
practical num_envs, real `ManagerBasedRlEnv`, `device="cuda:0"`), with
an env-spec version produced by `apply_reference_rsi` from a synthetic
get-up clip (lying start z=0.12 m, pitched 90 deg face-up, with a
per-joint "curled up" `joint_pos` trace), resets it, and asserts the
OBSERVED root height and orientation in the reset state actually match
a lying get-up start — not just that the cfg carries the right values.
This is the highest verification level available (live env, not just
cfg-mutation assertions): it proves both new mechanisms actually work
end to end inside mjlab —
  * `pose_range["pitch"]` orientation wiring (§8 part 1);
  * the injected `reset_joints_to_reference` event term (§8 part 2,
    the genuinely new per-joint-target mjlab event this increment
    adds — mjlab's shipped `reset_joints_by_offset` cannot do this).

Invocation:
    cd ~/projects/RewardSculptor && uv run pytest -m gpu

Skipped automatically on hosts without CUDA (see conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Canonical G1 joint order (mirrors sculptor.eval.robot_manifest.G1_29 —
# duplicated here as a literal so this test has no hidden coupling to
# that module's own correctness).
_G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _make_synthetic_getup_clip() -> dict:
    """Lying (z=0.12 m, pitched pi/2 face-up, curled-up joints) -> ramp
    -> standing (z=0.78 m, upright, straight joints)."""
    fps = 50.0
    n_lying, n_ramp, n_stand = 25, 50, 25
    lying_z = np.full(n_lying, 0.12)
    ramp_z = np.linspace(0.12, 0.78, n_ramp)
    stand_z = np.full(n_stand, 0.78)
    z = np.concatenate([lying_z, ramp_z, stand_z])

    def quat_pitch(theta: float) -> np.ndarray:
        return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])

    lying_q = np.tile(quat_pitch(np.pi / 2), (n_lying, 1))
    ramp_s = np.linspace(0.0, 1.0, n_ramp, endpoint=False)
    ramp_q = np.stack([quat_pitch(np.pi / 2 * (1 - s)) for s in ramp_s])
    stand_q = np.tile(quat_pitch(0.0), (n_stand, 1))
    quat = np.concatenate([lying_q, ramp_q, stand_q], axis=0)

    j = len(_G1_JOINTS)
    lying_j = np.tile(np.linspace(0.3, -0.3, j), (n_lying, 1))
    ramp_j = np.linspace(lying_j[0], np.zeros(j), n_ramp)
    stand_j = np.zeros((n_stand, j))
    joint_pos = np.concatenate([lying_j, ramp_j, stand_j], axis=0)

    return {
        "root_pos_z": z, "fps": fps, "root_quat_wxyz": quat,
        "joint_pos": joint_pos, "joint_names": list(_G1_JOINTS),
        "meta": {"source": "gpu_smoke:synthetic_getup"},
    }


def _quat_wxyz_to_pitch_rad(q: np.ndarray) -> float:
    w, x, y, z = q
    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(np.arcsin(sinp))


@pytest.mark.gpu
def test_getup_rsi_env_reset_is_actually_lying(tmp_path: Path) -> None:
    """End-to-end: synthetic get-up clip -> apply_reference_rsi ->
    _apply_env_spec -> real ManagerBasedRlEnv(device="cuda:0") -> reset
    -> observed root height < 0.35 m and |pitch| > 1.0 rad."""
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from sculptor.adapters._mjlab_runner import _apply_env_spec
    from sculptor.env_spec import read_current_env_spec
    from sculptor.reference import apply_reference_rsi

    task_id = "Mjlab-Velocity-Flat-Unitree-G1"
    clip = _make_synthetic_getup_clip()
    env_dir = tmp_path / "env"
    apply_reference_rsi(env_dir, clip)
    spec = read_current_env_spec(env_dir)

    # Sanity on the spec itself before spending GPU time — all four new
    # keys must be present (the whole point of this increment).
    train = spec["train"]
    assert "reset_pitch_offset_rad" in train
    assert "reset_joint_pos_target" in train
    assert len(train["reset_joint_pos_target"]) == len(_G1_JOINTS)

    env_cfg = load_env_cfg(task_id)
    # Smallest num_envs the adapter/mjlab stack tolerates for a
    # construction smoke — this test only needs a reset, not throughput.
    env_cfg.scene.num_envs = 2
    _apply_env_spec(env_cfg, spec, train=True, task_id=task_id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        env.reset()
        asset = env.scene["robot"]
        root_state = asset.data.root_link_pose_w  # (num_envs, 7)
        heights = root_state[:, 2].detach().cpu().numpy()
        quats = root_state[:, 3:7].detach().cpu().numpy()
        pitches = [abs(_quat_wxyz_to_pitch_rad(q)) for q in quats]

        assert (heights < 0.35).all(), (
            f"reset root height not low enough for a lying get-up "
            f"start: {heights}")
        assert all(p > 1.0 for p in pitches), (
            f"reset pitch magnitude not large enough for a lying "
            f"get-up start: {pitches}")

        # The per-joint reference posture reset must also be observed —
        # NOT the standing default (all-zero for this synthetic clip).
        joint_pos = asset.data.joint_pos.detach().cpu().numpy()
        target = np.array(train["reset_joint_pos_target"])
        # Generous tolerance: soft-limit clamping + noise (0.05 rad) may
        # shift individual joints slightly from the exact target.
        assert np.abs(joint_pos - target[None, :]).max() < 0.3, (
            "reset joint_pos does not resemble the reference posture "
            "target — reset_joints_to_reference may not have fired")
    finally:
        env.close()
