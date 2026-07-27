"""sculptor/refs/dr_calibration.py — task-matched domain randomization.

Two ideas from the lab review (docs/RESEARCH_DIRECTION.md §6), implemented
against the assets this repo already has.

**1. Size the robustness push to the behavior, not to a constant.**

The recommended heuristic: apply perturbations as *velocity pushes* rather
than forces, sized so the imparted momentum is roughly half the momentum of
the motion you expect from that behavior. mjlab's ``push_robot`` event is
already velocity-based (``velocity_range``), so the missing half was the
sizing — the magnitude was whatever a constant or an LLM guess supplied,
which means the same push that barely perturbs a run knocks over a crouch.

The sizing collapses to something simple. A push sets a base-velocity delta
on the *same body* whose momentum we are comparing against, so

    imparted / expected  =  (m · Δv) / (m · v_behavior)  =  Δv / v_behavior

and the mass cancels exactly. No mass estimate is needed, and the result is
robot-independent: the push velocity is just a fraction of the behavior's own
characteristic speed, which the attached reference clip already measures.

The 0.5 fraction is the reviewer's tacit heuristic — no paper states it — so
it is a *calibratable default*, not a law. It is a named argument, recorded
in the returned provenance, and meant to be tuned per robot.

**2. Randomize what the task is actually uncertain about.**

Randomizing every axis at full width is not free: BeyondMimic (2508.08241)
finds that over-wide randomization dilutes the control objective and yields an
over-conservative policy. A locomotion task genuinely needs terrain and
friction; a carry task needs payload mass; a task involving external forces
needs pushes. `dr_profile_for_task` selects axes by task type instead of
applying one flat always-on set.

Both functions are pure and offline: they read a clip and return plain dicts
that merge into an env spec. Nothing here touches the simulator.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

#: The reviewer's tacit heuristic: imparted momentum ~= half the behavior's.
#: Explicitly NOT a published constant — tune it per robot.
DEFAULT_PUSH_MOMENTUM_FRACTION = 0.5

#: Characteristic speed is a high percentile of the root speed rather than the
#: max: a single retargeting spike would otherwise set the push magnitude for
#: the whole behavior.
CHARACTERISTIC_SPEED_PERCENTILE = 90.0

#: Even a near-stationary behavior (balance, stand, hold) needs SOME push or
#: the robustness signal vanishes entirely. Floor in m/s.
MIN_PUSH_MPS = 0.15
#: Rail against a corrupt clip producing an unsurvivable push.
MAX_PUSH_MPS = 3.0


def characteristic_speed(clip: Mapping[str, Any]) -> float:
    """The behavior's representative root speed in m/s.

    Uses whatever translation channels the clip carries: horizontal speed from
    ``root_pos_xy`` and vertical from ``root_pos_z``, differentiated at the
    clip's own fps. A clip with no XY (height-only, e.g. a squat) still yields
    a meaningful vertical speed rather than zero.
    """
    fps = float(clip["fps"])
    z = np.asarray(clip["root_pos_z"], dtype=np.float64)
    if z.shape[0] < 2:
        return 0.0
    vz = np.gradient(z, 1.0 / fps)
    xy = clip.get("root_pos_xy")
    if xy is not None:
        xy = np.asarray(xy, dtype=np.float64)
        vxy = np.gradient(xy, 1.0 / fps, axis=0)
        speed = np.linalg.norm(np.column_stack([vxy, vz]), axis=1)
    else:
        speed = np.abs(vz)
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, CHARACTERISTIC_SPEED_PERCENTILE))


def push_events_for_behavior(
    clip: Mapping[str, Any],
    *,
    fraction: float = DEFAULT_PUSH_MOMENTUM_FRACTION,
    interval_s: tuple[float, float] = (3.0, 6.0),
    angular_fraction: float = 0.5,
) -> dict[str, Any]:
    """A ``push_events`` block sized to this behavior's own momentum.

    Returns the env-spec shape directly (``enabled``/``interval_s``/
    ``linear_mps``/``angular_radps``) plus a ``provenance`` block recording the
    measured speed and the fraction used — the fraction is a tunable default,
    so a run must be able to say which value produced its numbers.

    Raises ``ValueError`` for a non-positive fraction; a caller asking for a
    zero push should disable pushes explicitly instead.
    """
    if fraction <= 0.0:
        raise ValueError(
            f"fraction must be > 0 (got {fraction}); to disable pushes set "
            "push_events.enabled = false rather than sizing them to zero")

    speed = characteristic_speed(clip)
    raw = speed * float(fraction)
    linear = float(min(MAX_PUSH_MPS, max(MIN_PUSH_MPS, raw)))
    return {
        "enabled": True,
        "interval_s": [float(interval_s[0]), float(interval_s[1])],
        "linear_mps": round(linear, 4),
        # Angular pushes are scaled off the same budget. Expressed in rad/s
        # against a ~1 m-scale humanoid, so the numeric value carries over.
        "angular_radps": round(float(linear * angular_fraction), 4),
        "provenance": {
            "rule": "imparted momentum ~= fraction x behavior momentum; the "
                    "push and the behavior act on the same body, so mass "
                    "cancels and this is a pure velocity ratio",
            "characteristic_speed_mps": round(speed, 4),
            "speed_percentile": CHARACTERISTIC_SPEED_PERCENTILE,
            "fraction": float(fraction),
            "raw_push_mps": round(raw, 4),
            "clamped": bool(raw != linear),
            "caveat": "the 0.5 fraction is a practitioner heuristic, not a "
                      "published constant — calibrate per robot",
        },
    }


#: Which randomization axes each task type is genuinely uncertain about.
#: Keys are env-spec keys; a task's profile is the union of its entries with
#: the always-on core. Deliberately narrow — see the BeyondMimic caveat above.
_TASK_AXES: dict[str, tuple[str, ...]] = {
    # Feet-and-ground dominated: what varies in reality is the surface.
    "locomotion": ("friction_range", "body_friction_range",
                   "body_mass_scale_range"),
    # Carrying changes the inertia the controller must compensate, which is
    # exactly what a mass scale expresses.
    "carry": ("body_mass_scale_range", "pd_kp_scale_range",
              "pd_kd_scale_range"),
    # Contact-rich: the object's friction and the arm's gains dominate.
    "manipulation": ("body_friction_range", "pd_kp_scale_range",
                     "pd_kd_scale_range", "motor_strength_scale_range"),
    # Ballistic phases are decided by actuator authority, not by surface.
    "jump": ("motor_strength_scale_range", "body_mass_scale_range",
             "joint_armature_scale_range"),
    # Getting up scrapes the whole body along the ground.
    "getup": ("body_friction_range", "friction_range",
              "joint_damping_scale_range"),
    # Standing/balance is about holding against disturbance.
    "balance": ("pd_kp_scale_range", "pd_kd_scale_range",
                "body_mass_scale_range"),
}

#: Randomized for every task type: mass and motor strength are uncertain on
#: real hardware regardless of what the task is.
_CORE_AXES: tuple[str, ...] = (
    "body_mass_scale_range", "motor_strength_scale_range")

#: Moderate default widths. Mass follows the reviewer's explicit 0.75-1.5.
_AXIS_DEFAULTS: dict[str, list[float]] = {
    "body_mass_scale_range": [0.75, 1.5],
    "motor_strength_scale_range": [0.8, 1.2],
    "pd_kp_scale_range": [0.8, 1.2],
    "pd_kd_scale_range": [0.8, 1.2],
    "joint_damping_scale_range": [0.8, 1.2],
    "joint_armature_scale_range": [0.8, 1.2],
    "friction_range": [0.4, 1.4],
    "body_friction_range": [0.4, 1.4],
}


def dr_profile_for_task(
    task_type: str,
    *,
    clip: Optional[Mapping[str, Any]] = None,
    push_fraction: float = DEFAULT_PUSH_MOMENTUM_FRACTION,
) -> dict[str, Any]:
    """Randomization axes for `task_type`, as env-spec ``train`` keys.

    Unknown task types fall back to the core axes rather than raising: an
    unrecognized task should get conservative randomization, never none and
    never everything. When `clip` is supplied, a momentum-matched
    ``push_events`` block is included.
    """
    key = str(task_type or "").strip().lower()
    axes = tuple(dict.fromkeys(_CORE_AXES + _TASK_AXES.get(key, ())))
    profile: dict[str, Any] = {
        axis: list(_AXIS_DEFAULTS[axis]) for axis in axes
        if axis in _AXIS_DEFAULTS
    }
    if clip is not None:
        profile["push_events"] = push_events_for_behavior(
            clip, fraction=push_fraction)
    profile["dr_profile"] = {
        "task_type": key or "unknown",
        "recognized": key in _TASK_AXES,
        "axes": sorted(profile.keys() - {"push_events", "dr_profile"}),
        "rationale": (
            "task-matched axes only; randomizing everything at full width "
            "dilutes the control objective and yields an over-conservative "
            "policy (BeyondMimic 2508.08241)"),
    }
    return profile
