"""sculptor/adapters/realism.py — physics-realism audit (§7.3).

Given an expanded `trajectory.npz` (§7.1) and a snapshot of the model's
actuator forceranges + joint ranges (`mjcf_limits.json`, also written by
the rollout runner in §7.1), compute three headline metrics:

  * **torque_saturation_frac** — fraction of (step, env, joint) triples
    where `|actuator_force| > 0.95 * max_abs_forcerange`. A policy that
    exploits unrealistic motor response (the cartwheel-test failure Sam
    hit on 2026-04-22) pins its joints near the forcerange ceiling
    across most of the rollout.
  * **joint_vel_p99** — per-joint 99th-percentile of `|joint_vel|`.
    Nominal joint velocities for Go1/G1 hardware are <30 rad/s (see
    ANYmal 1901.08652 §3). A trained policy exceeding this by 1.5× on
    ANY joint is tagged `mild`; 3× is tagged `severe` even without
    torque saturation.
  * **joint_limit_violation_frac** — fraction of (step, env, joint)
    triples where `joint_pos < range_lo` or `> range_hi`. Nonzero is
    usually a reward-hacking signal — the default reward terms clip
    at limits; the policy is finding a way past them.

The audit returns a verdict in `{"ok", "mild", "severe", "unknown"}`
that downstream consumers (diagnose prompt, UI chip, §7.4 auto-adjust
trigger) can switch on. `unknown` is returned when inputs are missing
or malformed — NEVER raise; the audit is best-effort and must not
block the iteration.

Functions are pure / stdlib-only (numpy at most) so they unit-test
without mjlab on a CPU-only machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# Thresholds tuned against the cartwheel-test evidence (2026-04-22):
#   - Sam's failing policy: all 29 G1 joints > 50% saturation, verdict=severe.
#   - A healthy Go1 walker run: 2-3% saturation, verdict=ok.
#   - Boundary cases at 10-30% saturation → mild (reward needs adjustment
#     but the MJCF itself is probably ok).
_SEVERE_ANY_JOINT_SAT = 0.3
_SEVERE_OVERALL_SAT = 0.5
_MILD_ANY_JOINT_SAT = 0.1
_NOMINAL_JOINT_VEL_RAD_S = 30.0  # ANYmal 1901.08652 §3
_SEVERE_VEL_MULT = 3.0
_MILD_VEL_MULT = 1.5
_NEAR_FORCERANGE_FRAC = 0.95  # "saturated" if |force| exceeds 95% of forcerange


def _safe_load_npz(path: Path) -> dict[str, np.ndarray] | None:
    """Return the npz contents as a plain dict, or None on any error.
    Never raises — the audit must degrade to verdict=unknown rather
    than explode."""
    try:
        if not path.is_file():
            return None
        with np.load(path) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    except Exception:  # noqa: BLE001
        return None


def _safe_load_json(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _per_joint_saturation(
    actuator_force: np.ndarray, forceranges: np.ndarray,
) -> np.ndarray:
    """Return per-joint saturation fraction, shape (num_actuators,).

    `actuator_force`: (T, N_env, num_actuators).
    `forceranges`: (num_actuators, 2).
    """
    # max absolute forcerange per joint — handles asymmetric ranges
    # (e.g. a joint with ctrlrange [-10, 50] saturates at |force|=50).
    ceil = np.maximum(np.abs(forceranges[:, 0]), np.abs(forceranges[:, 1]))
    # Guard zero / tiny ceilings (should not happen for real motors, but
    # synthetic tests may hand us zeros).
    ceil = np.where(ceil > 1e-6, ceil, np.inf)
    abs_force = np.abs(actuator_force)  # (T, N, A)
    over = abs_force > (_NEAR_FORCERANGE_FRAC * ceil)  # broadcasts last dim
    # Mean across (T, N) → per-joint fraction.
    return over.mean(axis=(0, 1)).astype(np.float64)


def _per_joint_vel_p99(joint_vel: np.ndarray) -> np.ndarray:
    """99th percentile of |joint_vel|, per joint."""
    abs_vel = np.abs(joint_vel)  # (T, N, J)
    flat = abs_vel.reshape(-1, abs_vel.shape[-1])  # (T*N, J)
    return np.percentile(flat, 99.0, axis=0).astype(np.float64)


def _per_joint_limit_violation(
    joint_pos: np.ndarray, joint_ranges: np.ndarray,
) -> np.ndarray:
    """Fraction of (step, env) pairs where joint_pos is outside its range."""
    lo = joint_ranges[:, 0]  # (J,)
    hi = joint_ranges[:, 1]  # (J,)
    # (T, N, J) → (T, N, J) booleans.
    out_of_range = (joint_pos < lo) | (joint_pos > hi)
    return out_of_range.mean(axis=(0, 1)).astype(np.float64)


def _verdict(
    overall_sat: float, any_joint_sat: float, vel_multiplier: float,
) -> str:
    if (
        any_joint_sat >= _SEVERE_ANY_JOINT_SAT
        or overall_sat >= _SEVERE_OVERALL_SAT
        or vel_multiplier >= _SEVERE_VEL_MULT
    ):
        return "severe"
    if (
        any_joint_sat >= _MILD_ANY_JOINT_SAT
        or vel_multiplier >= _MILD_VEL_MULT
    ):
        return "mild"
    return "ok"


def _top_joints(
    names: list[str], per_joint: np.ndarray, k: int = 3,
) -> list[dict]:
    """Top-k joints by `per_joint` value, as dicts with name + value."""
    if per_joint.size == 0:
        return []
    k = min(k, per_joint.size)
    idxs = np.argsort(per_joint)[::-1][:k]
    return [
        {"name": names[i] if i < len(names) else f"joint_{i}",
         "value": round(float(per_joint[i]), 4)}
        for i in idxs
    ]


def audit_rollout(
    trajectory_path: Path | str,
    limits_path: Path | str,
) -> dict[str, Any]:
    """Run the realism audit. Returns a dict with the schema:

        {
          "verdict": "ok" | "mild" | "severe" | "unknown",
          "reason": str,  # when verdict == "unknown"
          "torque_saturation_frac": float,        # overall mean
          "any_joint_saturation_max": float,      # worst single joint
          "joint_vel_p99_max": float,             # worst single joint
          "joint_limit_violation_frac": float,    # overall mean
          "top_joints_saturation": [
              {"name": "...", "value": 0.47},
              ...
          ],
          "top_joints_vel": [...],
          "top_joints_limit_violation": [...],
          "n_actuators": int,
          "n_joints": int,
          "n_steps": int,
        }

    Never raises; failures return verdict=unknown with a reason.
    """
    traj = _safe_load_npz(Path(trajectory_path))
    limits = _safe_load_json(Path(limits_path))
    if traj is None:
        return {"verdict": "unknown",
                "reason": f"trajectory file missing or unreadable: {trajectory_path}"}
    if limits is None:
        return {"verdict": "unknown",
                "reason": f"limits file missing or unreadable: {limits_path}"}

    # Feature-detect expanded trajectory fields (§7.1). Older npzs that
    # only contain `rewards` + `episode_id` get verdict=unknown with an
    # informative reason, NOT a crash.
    required = ("actuator_force", "joint_pos", "joint_vel")
    missing = [k for k in required if k not in traj]
    if missing:
        return {"verdict": "unknown",
                "reason": f"trajectory missing expanded fields (§7.1): {missing}"}

    actuator_force = traj["actuator_force"]     # (T, N, A)
    joint_pos = traj["joint_pos"]               # (T, N, J)
    joint_vel = traj["joint_vel"]               # (T, N, J)

    # Shape sanity.
    if actuator_force.ndim != 3 or joint_pos.ndim != 3 or joint_vel.ndim != 3:
        return {"verdict": "unknown",
                "reason": (
                    f"expected (T, N, K) arrays; got shapes "
                    f"actuator_force={actuator_force.shape}, "
                    f"joint_pos={joint_pos.shape}, "
                    f"joint_vel={joint_vel.shape}"
                )}

    T, N, A = actuator_force.shape
    _, _, J = joint_pos.shape
    if joint_vel.shape != (T, N, J):
        return {"verdict": "unknown",
                "reason": f"joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}"}

    # Limits.
    forceranges_list = limits.get("actuator_forceranges") or []
    joint_ranges_list = limits.get("joint_ranges") or []
    actuator_names = list(limits.get("actuator_names") or [])
    joint_names = list(limits.get("joint_names") or [])

    forceranges = np.asarray(forceranges_list, dtype=np.float64)
    joint_ranges = np.asarray(joint_ranges_list, dtype=np.float64)

    # Metric 1: torque saturation — skipped silently if shapes don't match.
    sat_per_joint: np.ndarray = np.zeros(A, dtype=np.float64)
    if forceranges.shape == (A, 2):
        sat_per_joint = _per_joint_saturation(actuator_force, forceranges)
    overall_sat = float(sat_per_joint.mean()) if sat_per_joint.size else 0.0
    any_joint_sat = float(sat_per_joint.max()) if sat_per_joint.size else 0.0

    # Metric 2: joint velocity p99.
    vel_p99_per_joint = _per_joint_vel_p99(joint_vel)
    vel_p99_max = float(vel_p99_per_joint.max()) if vel_p99_per_joint.size else 0.0
    vel_multiplier = vel_p99_max / _NOMINAL_JOINT_VEL_RAD_S

    # Metric 3: joint-limit violation.
    limit_viol_per_joint: np.ndarray = np.zeros(J, dtype=np.float64)
    if joint_ranges.shape == (J, 2):
        limit_viol_per_joint = _per_joint_limit_violation(joint_pos, joint_ranges)
    overall_limit_viol = (
        float(limit_viol_per_joint.mean()) if limit_viol_per_joint.size else 0.0
    )

    verdict = _verdict(overall_sat, any_joint_sat, vel_multiplier)

    return {
        "verdict": verdict,
        "torque_saturation_frac": round(overall_sat, 4),
        "any_joint_saturation_max": round(any_joint_sat, 4),
        "joint_vel_p99_max": round(vel_p99_max, 2),
        "joint_vel_multiplier_vs_nominal": round(vel_multiplier, 2),
        "joint_limit_violation_frac": round(overall_limit_viol, 4),
        "top_joints_saturation": _top_joints(actuator_names, sat_per_joint),
        "top_joints_vel": _top_joints(joint_names, vel_p99_per_joint),
        "top_joints_limit_violation": _top_joints(joint_names, limit_viol_per_joint),
        "n_actuators": int(A),
        "n_joints": int(J),
        "n_steps": int(T),
    }
