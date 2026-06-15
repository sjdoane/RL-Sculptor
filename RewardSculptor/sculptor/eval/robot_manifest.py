"""Per-robot joint manifests (§Ship 49).

The required-joint-roles validation gate must answer, BEFORE the first GPU
rollout exists, "does this robot actually expose the joints the metric
needs?". That needs the robot's `joint_names` at launch time. The manifest
provides them, keyed by the adapter `task_id` / robot hint (e.g.
"Mjlab-Velocity-Flat-Unitree-G1"), captured verbatim from real rollouts so
the names and ORDER match what the runtime will see.

This is the GROUND-TRUTH ordering for the pre-run gate only; the runtime
still re-resolves against the LIVE `joint_names` of each rollout and
persists the result (so a robot/adapter change that drifts from the
manifest is caught at run time, not trusted blindly). An unknown robot is
returned as `None` — the gate then SKIPS (can't reject what it can't model)
and the runtime resolution + permutation gate remain the backstop.
"""
from __future__ import annotations

from typing import Optional, Sequence

# ── canonical joint orderings (captured from real mjlab rollouts) ────

#: Unitree G1 (29-DOF humanoid) — order from the entity joint_names API,
#: identical to the joint_pos/joint_vel buffer columns.
G1_29 = (
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

#: Unitree Go1 (12-DOF quadruped). FR/FL/RR/RL × hip(abduction)/thigh/calf.
GO1_12 = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

#: Robot-hint substring (lowercased) → canonical joint_names. Most specific
#: keys first; `robot_joint_names` scans in this order so "go1" beats a bare
#: "g1"-style fallback would-be collision (there is none today, but the
#: order documents intent).
_MANIFEST: tuple[tuple[str, Sequence[str]], ...] = (
    ("go2", GO1_12),       # Go2 shares the Go1 12-DOF leg layout
    ("go1", GO1_12),
    ("g1", G1_29),
)


def robot_joint_names(robot_hint: Optional[str]) -> Optional[list[str]]:
    """Canonical `joint_names` for a robot hint / task_id, or None if the
    robot is not in the manifest. Substring match on the lowercased hint so
    "Mjlab-Velocity-Flat-Unitree-G1" resolves to the G1 layout."""
    if not robot_hint:
        return None
    h = str(robot_hint).lower()
    for key, names in _MANIFEST:
        if key in h:
            return list(names)
    return None
