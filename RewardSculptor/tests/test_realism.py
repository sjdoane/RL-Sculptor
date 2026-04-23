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
    _NEAR_FORCERANGE_FRAC,
    _NOMINAL_JOINT_VEL_RAD_S,
    _SEVERE_ANY_JOINT_SAT,
    _SEVERE_OVERALL_SAT,
    audit_rollout,
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
