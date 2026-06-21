"""tests/test_realism.py — §7.3 physics-realism audit.

CPU-only unit tests for `sculptor.adapters.realism.audit_rollout`.
Builds synthetic trajectory.npz + mjcf_limits.json fixtures, calls the
audit, asserts per-metric correctness + verdict thresholds. No mjlab /
GPU required — the audit is numpy-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.adapters.realism import (
    _MILD_ANY_JOINT_SAT,
    _NAT_LIMIT_VIOLATION_FRAC,
    _NEAR_FORCERANGE_FRAC,
    _NOMINAL_JOINT_VEL_RAD_S,
    _SEVERE_ANY_JOINT_SAT,
    _SEVERE_OVERALL_SAT,
    audit_rollout,
    naturalness_channel,
    steer_fitness,
)


# ── Fixtures ──────────────────────────────────────────────────────────────
def _write_trajectory(
    path: Path,
    actuator_force: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    *,
    include_required_base: bool = True,
) -> None:
    payload: dict[str, np.ndarray] = {
        "actuator_force": actuator_force.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
    }
    if include_required_base:
        payload["rewards"] = np.zeros(actuator_force.shape[0], dtype=np.float32)
        payload["episode_id"] = np.zeros(actuator_force.shape[0], dtype=np.int32)
    np.savez_compressed(path, **payload)


def _write_limits(
    path: Path,
    *,
    forceranges: list[list[float]] | None = None,
    joint_ranges: list[list[float]] | None = None,
    actuator_names: list[str] | None = None,
    joint_names: list[str] | None = None,
) -> None:
    payload = {
        "actuator_forceranges": forceranges or [],
        "joint_ranges": joint_ranges or [],
        "actuator_names": actuator_names or [],
        "joint_names": joint_names or [],
    }
    path.write_text(json.dumps(payload))


# ── Verdict thresholds ────────────────────────────────────────────────────
def test_audit_ok_verdict_on_healthy_policy(tmp_path: Path) -> None:
    """3 joints, 100 steps, 8 envs, all forces at 10% forcerange, vels
    at 5 rad/s, positions well within range. Must verdict=ok."""
    T, N, A = 100, 8, 3
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.full((T, N, A), 5.0)    # 10% of 50 Nm
    pos = np.full((T, N, A), 0.0)       # center of range
    vel = np.full((T, N, A), 5.0)       # 1/6 of nominal 30 rad/s
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
        actuator_names=["act0", "act1", "act2"],
        joint_names=["j0", "j1", "j2"],
    )
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "ok", out
    assert out["torque_saturation_frac"] == pytest.approx(0.0)
    assert out["any_joint_saturation_max"] == pytest.approx(0.0)
    assert out["joint_limit_violation_frac"] == pytest.approx(0.0)


def test_audit_severe_verdict_when_single_joint_saturated(tmp_path: Path) -> None:
    """One joint pinned at forcerange ceiling across 100% of steps →
    any_joint_saturation_max ≥ _SEVERE_ANY_JOINT_SAT → severe."""
    T, N, A = 50, 4, 3
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.full((T, N, A), 5.0)
    # Joint 1 fully saturated.
    forces[:, :, 1] = 49.9  # > 0.95 * 50
    pos = np.zeros((T, N, A))
    vel = np.full((T, N, A), 2.0)
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
        actuator_names=["act_hip", "act_knee", "act_ankle"],
        joint_names=["j_hip", "j_knee", "j_ankle"],
    )
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "severe", out
    assert out["any_joint_saturation_max"] == pytest.approx(1.0)
    # Top saturated joint must be the one we pinned.
    top = out["top_joints_saturation"]
    assert top[0]["name"] == "act_knee", top
    assert top[0]["value"] == pytest.approx(1.0)


def test_audit_mild_verdict_at_boundary_saturation(tmp_path: Path) -> None:
    """Saturation of ~15% on any joint → mild (past _MILD_ but below _SEVERE_)."""
    T, N, A = 100, 4, 2
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.full((T, N, A), 5.0)
    # Joint 0 saturated on 15% of steps.
    forces[:15, :, 0] = 49.9
    pos = np.zeros((T, N, A))
    vel = np.full((T, N, A), 2.0)
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
        actuator_names=["a0", "a1"],
        joint_names=["j0", "j1"],
    )
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "mild", out
    assert _MILD_ANY_JOINT_SAT <= out["any_joint_saturation_max"] < _SEVERE_ANY_JOINT_SAT


def test_audit_severe_on_high_joint_velocity(tmp_path: Path) -> None:
    """No torque saturation but peak velocity > 3× nominal → severe."""
    T, N, A = 50, 4, 2
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.zeros((T, N, A))
    pos = np.zeros((T, N, A))
    # Joint 0 spinning at 100 rad/s — well past 3× nominal (90).
    vel = np.full((T, N, A), 2.0)
    vel[:, :, 0] = 100.0
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
    )
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "severe", out
    assert out["joint_vel_p99_max"] == pytest.approx(100.0)


def test_audit_mild_on_moderate_joint_velocity(tmp_path: Path) -> None:
    """Peak velocity 1.5-3× nominal → mild."""
    T, N, A = 50, 4, 2
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.zeros((T, N, A))
    pos = np.zeros((T, N, A))
    # 2× nominal (60 rad/s, > 1.5× but < 3×).
    vel = np.full((T, N, A), 60.0)
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
    )
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "mild", out


# ── Per-joint limit violation ──────────────────────────────────────────────
def test_audit_reports_joint_limit_violation(tmp_path: Path) -> None:
    T, N, A = 100, 4, 2
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.zeros((T, N, A))
    pos = np.zeros((T, N, A))
    # Joint 0 out of range [+0.5 < 1.0] for 40% of steps.
    pos[:40, :, 0] = 1.5
    vel = np.full((T, N, A), 2.0)
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
        joint_names=["hip", "knee"],
    )
    out = audit_rollout(traj, limits)
    # 40% of (T, N) pairs violate on joint 0, 0% on joint 1.
    # Overall mean = 20%.
    assert out["joint_limit_violation_frac"] == pytest.approx(0.20, abs=1e-3)
    top_limit = out["top_joints_limit_violation"]
    assert top_limit[0]["name"] == "hip"
    assert top_limit[0]["value"] == pytest.approx(0.40)


# ── Error paths: never raise, always return verdict=unknown ────────────────
def test_audit_missing_trajectory_returns_unknown(tmp_path: Path) -> None:
    limits = tmp_path / "mjcf_limits.json"
    _write_limits(limits)
    out = audit_rollout(tmp_path / "not_there.npz", limits)
    assert out["verdict"] == "unknown"
    assert "trajectory" in out["reason"].lower()


def test_audit_missing_limits_returns_unknown(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory.npz"
    _write_trajectory(
        traj, np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.float32),
    )
    out = audit_rollout(traj, tmp_path / "not_there.json")
    assert out["verdict"] == "unknown"
    assert "limits" in out["reason"].lower()


def test_audit_trajectory_without_expanded_fields_returns_unknown(
    tmp_path: Path,
) -> None:
    """Pre-§7.1 trajectory.npz (only `rewards` + `episode_id`) → unknown."""
    traj = tmp_path / "trajectory.npz"
    np.savez_compressed(
        traj, rewards=np.zeros(5, dtype=np.float32),
        episode_id=np.zeros(5, dtype=np.int32),
    )
    limits = tmp_path / "mjcf_limits.json"
    _write_limits(limits, forceranges=[[-50, 50]], joint_ranges=[[-1, 1]])
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "unknown"
    assert "expanded fields" in out["reason"]


def test_audit_empty_limits_gracefully_skips_metrics(tmp_path: Path) -> None:
    """Limits file present but empty arrays → audit still runs (with
    zero saturation/violation metrics), never crashes."""
    T, N, A = 10, 2, 2
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    _write_trajectory(
        traj, np.full((T, N, A), 100.0),  # would saturate if limits present
        np.zeros((T, N, A)),
        np.full((T, N, A), 5.0),
    )
    _write_limits(limits, forceranges=[], joint_ranges=[])
    out = audit_rollout(traj, limits)
    assert out["verdict"] in ("ok", "mild", "severe")  # doesn't crash
    # Saturation cannot be computed without forceranges → stays at 0.
    assert out["torque_saturation_frac"] == pytest.approx(0.0)
    # Joint velocity still computed.
    assert out["joint_vel_p99_max"] == pytest.approx(5.0, abs=0.5)


def test_audit_corrupt_npz_returns_unknown(tmp_path: Path) -> None:
    """Non-npz file at the trajectory path → unknown, not a crash."""
    traj = tmp_path / "trajectory.npz"
    traj.write_text("not an npz file")
    limits = tmp_path / "mjcf_limits.json"
    _write_limits(limits)
    out = audit_rollout(traj, limits)
    assert out["verdict"] == "unknown"


# ── Named joint reporting ──────────────────────────────────────────────────
def test_audit_top_joints_use_positional_fallback_when_names_missing(
    tmp_path: Path,
) -> None:
    """No actuator_names in limits → audit reports `actuator_0`, etc."""
    T, N, A = 10, 2, 3
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.zeros((T, N, A))
    forces[:, :, 2] = 49.9  # saturate joint 2
    pos = np.zeros((T, N, A))
    vel = np.full((T, N, A), 2.0)
    _write_trajectory(traj, forces, pos, vel)
    # No names.
    _write_limits(
        limits, forceranges=[[-50, 50]] * A, joint_ranges=[[-1, 1]] * A,
    )
    out = audit_rollout(traj, limits)
    # Fallback name format is `joint_<index>` (shared prefix across
    # top-saturation / top-vel / top-limit surfaces — the audit doesn't
    # distinguish actuator vs joint labels when names are missing).
    top_name = out["top_joints_saturation"][0]["name"]
    assert top_name == "joint_2", top_name


# ── §LAW 7: naturalness channel (down-weight + flag; hard-reject on limit) ──


def test_naturalness_channel_hard_rejects_joint_limit_exploit():
    """A genuine joint-LIMIT violation (exploiting past the joint stops) is the
    only hard reject — steer credit goes to 0."""
    nat = naturalness_channel({"verdict": "severe", "joint_limit_violation_frac": 0.2})
    assert nat["hard_reject"] is True
    assert nat["steer_factor"] == 0.0
    assert nat["flag"] == "joint_limit_exploit"


def test_naturalness_channel_downweights_severe_without_limit():
    """A 'severe' verdict WITHOUT a limit violation (e.g. an 18 rad/s whip) is
    DOWN-WEIGHTED + flagged, never rejected — an aggressive-but-valid kick can
    legitimately be severe (Sam's policy)."""
    nat = naturalness_channel({"verdict": "severe", "joint_limit_violation_frac": 0.0})
    assert nat["hard_reject"] is False
    assert 0.0 < nat["steer_factor"] < 1.0
    assert nat["flag"] == "unnatural_severe"


def test_naturalness_channel_passes_ok_unknown():
    for verdict in ("ok", "unknown"):
        nat = naturalness_channel({"verdict": verdict, "joint_limit_violation_frac": 0.0})
        assert nat["hard_reject"] is False
        assert nat["steer_factor"] == 1.0
    # None / non-dict audit is a no-op pass (a missing rollout must not suppress steer)
    assert naturalness_channel(None)["steer_factor"] == 1.0


def test_naturalness_channel_mild_down_weights():
    """§kick-fix (legitimate bar-raise): a 'mild' verdict now DOWN-WEIGHTS steer
    credit ×0.75 (was full credit 1.0) — on g1-kick-v6 the violent kicks read
    mild/severe at 2-3.7× the real motor limit and the loop kept selecting them."""
    nat = naturalness_channel({"verdict": "mild", "joint_limit_violation_frac": 0.0})
    assert nat["hard_reject"] is False
    assert nat["steer_factor"] == 0.75
    assert nat["flag"] == "unnatural_mild"


def test_steer_fitness_gates_separately_from_display():
    """steer_fitness gates the value used for SELECTION; a hard reject zeros it,
    a severe verdict scales it, and ok / None pass through unchanged (so the
    displayed fitness — passed separately — is never corrupted)."""
    reject = naturalness_channel({"verdict": "severe", "joint_limit_violation_frac": 0.5})
    severe = naturalness_channel({"verdict": "severe", "joint_limit_violation_frac": 0.0})
    mild = naturalness_channel({"verdict": "mild", "joint_limit_violation_frac": 0.0})
    ok = naturalness_channel({"verdict": "ok", "joint_limit_violation_frac": 0.0})
    assert steer_fitness(0.9, reject) == 0.0                 # exploit can't win 'best'
    assert steer_fitness(0.9, severe) == pytest.approx(0.45)  # down-weighted ×0.5
    assert steer_fitness(0.9, mild) == pytest.approx(0.675)   # §kick-fix: mild ×0.75
    assert steer_fitness(0.9, ok) == pytest.approx(0.9)       # unchanged
    assert steer_fitness(0.9, None) == pytest.approx(0.9)     # no audit → unchanged
    assert steer_fitness(None, reject) is None                # no fitness → None


def test_naturalness_hard_reject_uses_per_joint_max_not_mean(tmp_path: Path) -> None:
    """§kick-fix: a focused SINGLE-joint limit exploit on a HIGH-DOF robot must
    still hard-reject. On a 29-DOF G1 one joint past its stop for 20% of (step,env)
    pairs dilutes to ~0.20/29 ≈ 0.007 in the J-averaged mean — under the 1% gate —
    so the audit exposes a per-joint MAX and the naturalness channel thresholds
    THAT (the 'any joint past its stop = exploit' guarantee is DOF-independent)."""
    T, N, A = 100, 8, 29
    traj = tmp_path / "trajectory.npz"
    limits = tmp_path / "mjcf_limits.json"
    forces = np.zeros((T, N, A))
    pos = np.zeros((T, N, A))
    pos[:20, :, 0] = 1.5            # joint 0 out of [-1, 1] for 20% of steps
    vel = np.full((T, N, A), 2.0)
    _write_trajectory(traj, forces, pos, vel)
    _write_limits(
        limits, forceranges=[[-50, 50]] * A,
        joint_ranges=[[-1.0, 1.0]] * A,
        joint_names=[f"j{i}" for i in range(A)],
    )
    out = audit_rollout(traj, limits)
    # The mean is diluted under the gate; the per-joint max captures the exploit.
    assert out["joint_limit_violation_frac"] < _NAT_LIMIT_VIOLATION_FRAC
    assert out["joint_limit_violation_max"] == pytest.approx(0.20, abs=1e-3)
    nat = naturalness_channel(out)
    assert nat["hard_reject"] is True
    assert nat["steer_factor"] == 0.0
    assert nat["flag"] == "joint_limit_exploit"


def test_actuator_limits_report_speed_and_torque(tmp_path):
    """§reports: per-motor speed (vs no-load velocity_limit) + torque (vs effort
    limit) utilization, JOINT-aligned. Torque present only when joint_torque is
    persisted; speed always. Limits resolved by joint name."""
    from sculptor.adapters.realism import actuator_limits_report

    T, E = 20, 4
    names = ["left_knee_joint", "left_hip_pitch_joint", "left_elbow_joint"]
    # knee at ~10 rad/s (limit 20 → 50%), 100 N·m torque (limit 139 → 72%)
    jv = np.zeros((T, E, 3)); jt = np.zeros((T, E, 3))
    jv[..., 0] = 10.0; jt[..., 0] = 100.0
    jv[..., 1] = 16.0; jt[..., 1] = 44.0     # hip_pitch: 16/32=50%, 44/88=50%
    jv[..., 2] = 3.7; jt[..., 2] = 2.5
    np.savez(tmp_path / "trajectory.npz", joint_vel=jv, joint_torque=jt,
             actuator_force=jt, projected_gravity_b=np.zeros((T, E, 3)))
    (tmp_path / "mjcf_limits.json").write_text(
        json.dumps({"joint_names": names}), encoding="utf-8")

    rep = actuator_limits_report(tmp_path / "trajectory.npz", tmp_path / "mjcf_limits.json")
    assert rep["ok"] and rep["has_torque"] and rep["n_motors"] == 3
    knee = next(m for m in rep["motors"] if m["name"] == "left_knee_joint")
    assert knee["velocity_limit"] == 20.0 and knee["effort_limit"] == 139.0
    assert knee["speed_util_p99"] == pytest.approx(0.5, abs=0.01)
    assert knee["torque_util_p99"] == pytest.approx(100.0 / 139.0, abs=0.01)

    # no joint_torque → speed only, has_torque False (older rollouts)
    np.savez(tmp_path / "t2.npz", joint_vel=jv, projected_gravity_b=np.zeros((T, E, 3)))
    rep2 = actuator_limits_report(tmp_path / "t2.npz", tmp_path / "mjcf_limits.json")
    assert rep2["ok"] and not rep2["has_torque"]
    assert "torque_util_p99" not in rep2["motors"][0]
    assert rep2["motors"][0]["speed_util_p99"] == pytest.approx(0.5, abs=0.01)

    # missing joint_vel → graceful not-ok (never raises)
    np.savez(tmp_path / "t3.npz", rewards=np.zeros(T))
    assert actuator_limits_report(tmp_path / "t3.npz", tmp_path / "mjcf_limits.json")["ok"] is False


def test_joint_effort_limit_map():
    from sculptor.adapters.realism import joint_effort_limit
    assert joint_effort_limit("left_knee_joint") == 139.0
    assert joint_effort_limit("left_hip_pitch_joint") == 88.0
    assert joint_effort_limit("left_elbow_joint") == 25.0
    assert joint_effort_limit("FL_calf_joint") == 35.55      # Go1
    assert joint_effort_limit("mystery_joint") is None       # unknown → no torque bar


def test_joint_velocity_limit_g1_map_and_non_g1_fallback():
    """§kick-fix (byte-identical fence): G1 joints resolve to their real mjlab
    no-load speeds; EVERY non-G1 joint (Go1 'FL_hip_joint'/'FL_thigh_joint'/
    'FL_calf_joint', cartpole 'slider') falls to the generic nominal 30 — so the
    compound-token map can never silently re-scale another robot's audit."""
    from sculptor.adapters.realism import joint_velocity_limit, _NOMINAL_JOINT_VEL_RAD_S
    assert joint_velocity_limit("left_knee_joint") == 20.0
    assert joint_velocity_limit("right_hip_roll_joint") == 20.0
    assert joint_velocity_limit("left_hip_pitch_joint") == 32.0
    assert joint_velocity_limit("left_ankle_pitch_joint") == 37.0
    assert joint_velocity_limit("left_shoulder_pitch_joint") == 37.0
    assert joint_velocity_limit("left_wrist_pitch_joint") == 22.0
    for non_g1 in ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                   "RR_hip_joint", "slider", "cart", "pole", ""):
        assert joint_velocity_limit(non_g1) == _NOMINAL_JOINT_VEL_RAD_S


def test_verdict_per_joint_velocity_limit_term_byte_identical_without_real_limits():
    """The new per-joint velocity term defaults 0.0 and is OR'd in, so a robot with
    no real per-joint limits (vel_limit_max_ratio=0.0) gets the EXACT pre-fix
    verdict for every (sat, vel_multiplier) combination."""
    from sculptor.adapters.realism import _verdict
    cases = [(0.0, 0.0, 0.0), (0.4, 0.6, 0.0), (0.0, 0.0, 3.5),
             (0.0, 0.15, 1.6), (0.2, 0.2, 0.5)]
    for overall, anyj, velm in cases:
        assert _verdict(overall, anyj, velm, 0.0) == _verdict(overall, anyj, velm)
    # and the new term independently escalates when a real limit IS exceeded
    assert _verdict(0.0, 0.0, 0.0, 1.6) == "severe"   # ≥1.5× a no-load speed
    assert _verdict(0.0, 0.0, 0.0, 1.1) == "mild"     # ≥1.0×
