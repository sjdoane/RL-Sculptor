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
_NOMINAL_JOINT_VEL_RAD_S = 30.0  # ANYmal 1901.08652 §3 — generic fallback
_SEVERE_VEL_MULT = 3.0
_MILD_VEL_MULT = 1.5
_NEAR_FORCERANGE_FRAC = 0.95  # "saturated" if |force| exceeds 95% of forcerange

# §kick-fix (Sam 2026-06-20): real per-joint motor NO-LOAD SPEEDS for the Unitree
# G1, from mjlab `g1_constants.py` (the 5020/7520-14/7520-22/4010 ElectricActuator
# velocity_limits). mjlab DECLARES these but never passes them to the sim actuator,
# so a trained policy drives joints 2-3.7× past them (knee p99 43-73 vs 20 rad/s on
# g1-kick-v6) and the generic 30 nominal didn't flag it. Keyed by COMPOUND joint-name
# tokens (e.g. "hip_pitch", never bare "hip") so a non-G1 robot whose joints don't
# match (Go1 "FL_hip_joint"/"FL_thigh_joint") falls through to the nominal — the
# byte-identical fence for every other robot. A joint past its no-load speed is
# physically impossible (motor torque → 0 there), so exceeding it escalates the
# verdict. (Tuple order is first-match; tokens are disjoint so order is immaterial.)
_G1_JOINT_VEL_LIMITS: tuple[tuple[str, float], ...] = (
    ("knee", 20.0), ("hip_roll", 20.0),            # 7520-22
    ("hip_pitch", 32.0), ("hip_yaw", 32.0), ("waist_yaw", 32.0),   # 7520-14
    ("wrist_pitch", 22.0), ("wrist_yaw", 22.0),    # 4010
    ("shoulder", 37.0), ("elbow", 37.0), ("wrist_roll", 37.0),     # 5020
    ("ankle", 37.0), ("waist_pitch", 37.0), ("waist_roll", 37.0),  # 2×5020
)
_SEVERE_VEL_LIMIT_RATIO = 1.5  # ≥1.5× a real no-load speed → severe (clearly unphysical)
_MILD_VEL_LIMIT_RATIO = 1.0    # at/above a real no-load speed → mild

# §Metric-quality laws (LAW 7): naturalness-channel policy (Sam, 2026-06-18 —
# "down-weight + flag, hard-reject only on joint-limit violation").
_NAT_LIMIT_VIOLATION_FRAC = 0.01  # any real breach past the joint stops = exploit
_NAT_SEVERE_STEER_FACTOR = 0.5    # down-weight a 'severe' (vel/torque) rollout's steer credit
_NAT_MILD_STEER_FACTOR = 0.75     # §kick-fix: 'mild' now also down-weights (was 1.0)


def joint_velocity_limit(name: str) -> float:
    """The real motor no-load speed (rad/s) for a joint, by name. G1 motors from
    mjlab `g1_constants` (compound tokens); ANY non-matching joint — including every
    joint of a non-G1 robot — returns the generic nominal `_NOMINAL_JOINT_VEL_RAD_S`,
    so other robots' audits stay byte-identical."""
    n = (name or "").lower()
    for tok, lim in _G1_JOINT_VEL_LIMITS:
        if tok in n:
            return lim
    return _NOMINAL_JOINT_VEL_RAD_S


#: §reports: real per-motor TORQUE (effort) limits, N·m, cited from the SAME mjlab
#: constants as the velocity limits (g1_constants effort_limit; go1_constants). Used
#: only by the actuator-limits report (name-keyed, like the velocity map). G1: knee/
#: hip_roll 139 (7520-22), hip_pitch/yaw/waist_yaw 88 (7520-14), arms 25 (5020),
#: wrist_pitch/yaw 5 (4010), ankle/waist_pitch/roll 50 (2×5020). Go1: hip/thigh 23.7,
#: calf 35.55.
_JOINT_EFFORT_LIMITS: tuple[tuple[str, float], ...] = (
    ("knee", 139.0), ("hip_roll", 139.0),
    ("hip_pitch", 88.0), ("hip_yaw", 88.0), ("waist_yaw", 88.0),
    ("elbow", 25.0), ("shoulder", 25.0), ("wrist_roll", 25.0),
    ("wrist_pitch", 5.0), ("wrist_yaw", 5.0),
    ("ankle", 50.0), ("waist_pitch", 50.0), ("waist_roll", 50.0),
    ("thigh", 23.7), ("calf", 35.55), ("hip_joint", 23.7),   # Go1
)


def joint_effort_limit(name: str) -> "float | None":
    """The real motor torque (effort) limit (N·m) for a joint, by name, or None when
    unknown (the report then omits the torque-limit bar for that joint)."""
    n = (name or "").lower()
    for tok, lim in _JOINT_EFFORT_LIMITS:
        if tok in n:
            return lim
    return None


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
    vel_limit_max_ratio: float = 0.0,
) -> str:
    """`vel_limit_max_ratio` (§kick-fix): the worst per-joint velp99 ÷ that joint's
    REAL no-load speed, over joints with a known limit (0.0 when none known). It is
    OR'd in with — never replaces — the torque + generic-velocity terms, so a robot
    with no real per-joint limits (default 0.0) is byte-identical to before."""
    if (
        any_joint_sat >= _SEVERE_ANY_JOINT_SAT
        or overall_sat >= _SEVERE_OVERALL_SAT
        or vel_multiplier >= _SEVERE_VEL_MULT
        or vel_limit_max_ratio >= _SEVERE_VEL_LIMIT_RATIO
    ):
        return "severe"
    if (
        any_joint_sat >= _MILD_ANY_JOINT_SAT
        or vel_multiplier >= _MILD_VEL_MULT
        or vel_limit_max_ratio >= _MILD_VEL_LIMIT_RATIO
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

    # §kick-fix: per-joint velocity vs the REAL motor no-load speed, when known.
    # Counts ONLY joints with a real (non-nominal) limit, so a robot whose joints
    # all fall to the nominal contributes 0.0 → the verdict is byte-identical.
    vel_limit_max_ratio = 0.0
    if joint_names and len(joint_names) == vel_p99_per_joint.size:
        for _jn, _vp in zip(joint_names, vel_p99_per_joint):
            _lim = joint_velocity_limit(_jn)
            if _lim != _NOMINAL_JOINT_VEL_RAD_S:
                vel_limit_max_ratio = max(vel_limit_max_ratio, float(_vp) / _lim)

    # Metric 3: joint-limit violation.
    limit_viol_per_joint: np.ndarray = np.zeros(J, dtype=np.float64)
    if joint_ranges.shape == (J, 2):
        limit_viol_per_joint = _per_joint_limit_violation(joint_pos, joint_ranges)
    overall_limit_viol = (
        float(limit_viol_per_joint.mean()) if limit_viol_per_joint.size else 0.0
    )
    # §kick-fix: the WORST single joint's violation fraction. The naturalness
    # hard-reject (a real breach PAST the joint stops is an unambiguous physics
    # exploit) must threshold this, not the J-averaged mean — on a 29-DOF G1 a
    # focused single-joint exploit dilutes to ~frac/J in the mean and slips under
    # the 1% gate. Per-joint MAX preserves the "any joint past its stop = exploit"
    # guarantee regardless of DOF count.
    overall_limit_viol_max = (
        float(limit_viol_per_joint.max()) if limit_viol_per_joint.size else 0.0
    )

    verdict = _verdict(overall_sat, any_joint_sat, vel_multiplier,
                       vel_limit_max_ratio)

    return {
        "verdict": verdict,
        "torque_saturation_frac": round(overall_sat, 4),
        "any_joint_saturation_max": round(any_joint_sat, 4),
        "joint_vel_p99_max": round(vel_p99_max, 2),
        "joint_vel_multiplier_vs_nominal": round(vel_multiplier, 2),
        "joint_vel_limit_max_ratio": round(vel_limit_max_ratio, 2),
        "joint_limit_violation_frac": round(overall_limit_viol, 4),
        "joint_limit_violation_max": round(overall_limit_viol_max, 4),
        "top_joints_saturation": _top_joints(actuator_names, sat_per_joint),
        "top_joints_vel": _top_joints(joint_names, vel_p99_per_joint),
        "top_joints_limit_violation": _top_joints(joint_names, limit_viol_per_joint),
        "n_actuators": int(A),
        "n_joints": int(J),
        "n_steps": int(T),
    }


def actuator_limits_report(
    trajectory_path: Path | str, limits_path: Path | str,
) -> dict[str, Any]:
    """Per-motor joint SPEED (vs the real no-load `velocity_limit`) and TORQUE (vs
    the real `effort_limit`) utilization for one rollout — the data behind the
    reports 'actuator limits' charts that VISUALLY CONFIRM a policy respects each
    motor's real envelope (§actuator-limit enforcement). Everything is JOINT-ORDERED:
    speed from `joint_vel`, torque from `joint_torque` (= `qfrc_actuator`, the
    actuator force in joint space — added §reports), limits from the name-keyed maps.
    SPEED is always present; TORQUE only when `joint_torque` was persisted (older
    rollouts omit it → `has_torque: False`). NEVER raises.

    Each motor record: `{name, speed_p99, speed_max, velocity_limit,
    velocity_limit_known, speed_util_p99/max}` plus, when torque is available,
    `{torque_p99, torque_max, effort_limit, torque_util_p99/max}`. `*_util_*` is
    value ÷ limit (≤ 1.0 means within the envelope)."""
    traj = _safe_load_npz(Path(trajectory_path))
    if traj is None:
        return {"ok": False, "reason": "trajectory missing or unreadable",
                "has_torque": False, "motors": []}
    limits = _safe_load_json(Path(limits_path)) or {}
    jv = traj.get("joint_vel")
    if jv is None or getattr(jv, "ndim", 0) != 3:
        return {"ok": False, "reason": "trajectory missing joint_vel (§7.1 fields)",
                "has_torque": False, "motors": []}

    T, _, J = jv.shape
    joint_names = list(limits.get("joint_names") or [])
    abs_v = np.abs(jv).reshape(-1, J)
    sp_p99 = np.percentile(abs_v, 99.0, axis=0)
    sp_max = abs_v.max(axis=0)

    # Torque in JOINT space (qfrc_actuator), aligned with joint_vel — present only
    # when the rollout persisted it (§reports addition).
    jt = traj.get("joint_torque")
    tq_p99 = tq_max = None
    if jt is not None and getattr(jt, "ndim", 0) == 3 and jt.shape[2] == J:
        abs_t = np.abs(jt).reshape(-1, J)
        tq_p99 = np.percentile(abs_t, 99.0, axis=0)
        tq_max = abs_t.max(axis=0)

    motors: list[dict[str, Any]] = []
    for i in range(J):
        name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        vlim = joint_velocity_limit(name)
        rec: dict[str, Any] = {
            "name": name,
            "speed_p99": round(float(sp_p99[i]), 3),
            "speed_max": round(float(sp_max[i]), 3),
            "velocity_limit": round(float(vlim), 3),
            "velocity_limit_known": bool(vlim != _NOMINAL_JOINT_VEL_RAD_S),
            "speed_util_p99": round(float(sp_p99[i]) / vlim, 4),
            "speed_util_max": round(float(sp_max[i]) / vlim, 4),
        }
        if tq_p99 is not None:
            elim = joint_effort_limit(name)
            rec["torque_p99"] = round(float(tq_p99[i]), 3)
            rec["torque_max"] = round(float(tq_max[i]), 3)
            rec["effort_limit"] = (round(float(elim), 3) if elim else None)
            rec["torque_util_p99"] = (round(float(tq_p99[i]) / elim, 4) if elim else None)
            rec["torque_util_max"] = (round(float(tq_max[i]) / elim, 4) if elim else None)
        motors.append(rec)
    return {"ok": True, "n_steps": int(T), "n_motors": len(motors),
            "has_torque": tq_p99 is not None, "motors": motors}


def naturalness_channel(audit: dict[str, Any] | None) -> dict[str, Any]:
    """§Metric-quality laws (LAW 7): map a realism audit (`audit_rollout`) to a
    NATURALNESS decision kept SEPARATE from task-correctness. Policy (Sam,
    2026-06-18 — "down-weight + flag, hard-reject only on joint-limit violation"):

      * HARD-REJECT only on a genuine joint-LIMIT violation (the policy is
        exploiting PAST the joint stops — an unambiguous physics exploit): the
        iteration earns NO steer credit (`steer_factor=0.0`).
      * a `"severe"` verdict WITHOUT a limit violation (an 18 rad/s whip /
        torque saturation) is DOWN-WEIGHTED + flagged, NOT rejected — a
        genuinely aggressive-but-valid kick legitimately has a high vel-p99.
      * `"ok"`/`"mild"`/`"unknown"` pass at full credit (`steer_factor=1.0`).

    Returns `{verdict, hard_reject, steer_factor, flag, reason}`. Never raises; a
    None/empty/`unknown` audit is a no-op pass (factor 1.0) so a missing rollout
    never silently suppresses steering."""
    if not isinstance(audit, dict):
        return {"verdict": "unknown", "hard_reject": False,
                "steer_factor": 1.0, "flag": None, "reason": "no audit"}
    verdict = str(audit.get("verdict", "unknown"))
    # §kick-fix: threshold the WORST joint (joint_limit_violation_max), not the
    # J-averaged mean — a single-joint limit exploit on a high-DOF robot dilutes
    # under the gate in the mean. Fall back to the mean key for older audit dicts
    # (and direct test injection) so this stays a no-op-on-missing, never-raise read.
    limit_viol = float(
        audit.get("joint_limit_violation_max",
                  audit.get("joint_limit_violation_frac", 0.0)) or 0.0)
    if limit_viol > _NAT_LIMIT_VIOLATION_FRAC:
        return {
            "verdict": verdict, "hard_reject": True, "steer_factor": 0.0,
            "flag": "joint_limit_exploit",
            "reason": (f"joint-limit violation {limit_viol:.3f} > "
                       f"{_NAT_LIMIT_VIOLATION_FRAC} — exploiting past the joint "
                       f"stops; no steer credit (hard reject)"),
        }
    if verdict == "severe":
        return {
            "verdict": verdict, "hard_reject": False,
            "steer_factor": _NAT_SEVERE_STEER_FACTOR, "flag": "unnatural_severe",
            "reason": (f"severe realism verdict (torque/velocity) — steer credit "
                       f"down-weighted ×{_NAT_SEVERE_STEER_FACTOR} + flagged, not "
                       f"rejected (an aggressive-but-valid motion can be severe)"),
        }
    if verdict == "mild":
        # §kick-fix: a 'mild' verdict (1.0–1.5× nominal, or ≥1× a real motor no-load
        # speed) now DOWN-WEIGHTS steer credit too (was full credit). On g1-kick-v6
        # the violent kicks read mild/severe at 2–3.7× the real knee limit and the
        # loop kept selecting them; this denies them best-selection credit.
        return {
            "verdict": verdict, "hard_reject": False,
            "steer_factor": _NAT_MILD_STEER_FACTOR, "flag": "unnatural_mild",
            "reason": (f"mild realism verdict — steer credit down-weighted "
                       f"×{_NAT_MILD_STEER_FACTOR} + flagged"),
        }
    return {
        "verdict": verdict, "hard_reject": False, "steer_factor": 1.0,
        "flag": None, "reason": "",
    }


def steer_fitness(
    fitness: float | None, naturalness: dict[str, Any] | None,
) -> float | None:
    """§LAW 7: the naturalness-GATED value used for STEER SELECTION (best-by-
    fitness, fitness early-stop) — DISTINCT from the displayed fitness, which
    stays the TRUE task score. A `hard_reject` → 0.0 (a joint-limit exploit can
    never win 'best'); otherwise `fitness × steer_factor`. Passes None through
    (no fitness this iter). With a None/ok naturalness the factor is 1.0, so the
    steer value EQUALS the displayed fitness — byte-identical to the pre-LAW-7
    loop for every non-severe, non-exploiting iteration."""
    if fitness is None:
        return None
    nat = naturalness or {}
    if nat.get("hard_reject"):
        return 0.0
    return float(fitness) * float(nat.get("steer_factor", 1.0))
