"""sculptor/reward.py — versioned, swappable reward function.

The env (`QuadrupedJumpMJXEnv` in the AME456 repo) imports `compute_reward`
and `REWARD_SPEC` from this module. Sculptor iterations replace this file
(or point at a sibling module) to mutate the reward without touching the env.

v0 is a verbatim port of the in-env reward block at the time of refactor
(quadruped_mjx_env.py:579–699 in the AME456 repo). Behavioral parity with
the pre-refactor env is enforced by tests/test_reward_parity.py.

The function is JAX-pure (jit/vmap compatible). All branching uses jp.where.
Component values in the returned dict are signed contributions — they sum
(in the documented order, see _accumulate_reward) to the scalar reward
within IEEE-754 float32.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jp


REWARD_SPEC: dict = {
    "version": "v1_backdrive_fix",
    "description": (
        "v1 (2026-04-20, Run 15 post-mortem): action-smoothness weight raised "
        "from 0.05 → 1.0 (20×) to make bang-bang action exploitation "
        "unprofitable. Companion to the env-side B_BACKDRIVE=0.3 backdrive-"
        "resistance fix. Rest is unchanged from v0: ten per-step components "
        "plus four end-of-episode bonuses, hard -1000 on NaN termination."
    ),
    "author": "human",
    "parent_hash": "v0",
    "hyperparameters": {
        # per-step
        "tilt_pen_w":              2.0,     # × tilt²
        "angvel_damp_w":           0.2,     # × (|roll_rate| + |pitch_rate|)
        "horiz_vel_pen_w":         2.0,     # × (vx² + vy²)
        "upward_vel_w":            5.0,     # × max(0, body_vz), gated
        "upward_vel_n_feet_min":   3,       # gate
        "upward_vel_tilt_max_rad": 0.524,   # gate (~30°)
        "flight_height_w":         150.0,   # × min(max(0, height_gain), 0.4)²
        "flight_height_cap_m":     0.4,
        "cycle_bonus_w":           100.0,   # × clip((peak/0.03)², 0, 16) × upright
        "cycle_jump_ref_height_m": 0.03,
        "cycle_height_scale_clip": 16.0,
        "cycle_upright_full_tilt_rad": 0.436,  # ~25°
        "drift_dead_zone_m":       0.15,
        "drift_pen_w":             3.0,     # × max(0, xy_drift - 0.15)²
        "action_smoothness_w":     1.0,     # × Σ(action_t - action_{t-1})² (v1: was 0.05)
        "torque_cost_w":           0.0001,  # × Σ τ_motor²

        # termination + end-of-episode
        "term_penalty":            -1000.0,
        "end_max_peak_w":          3000.0,
        "end_successive_jump_w":   100.0,
        "end_successive_jump_cap": 5.0,
        "end_mean_peak_w":         500.0,
        "end_never_jumped_pen":    -500.0,
        "end_never_jumped_thresh_m": 0.001,
    },
    # paper_id strings, populated when the KG is live; v0 was authored by hand.
    "references": [],
}


def compute_reward(
    state: Any,
    action: Any,
    next_state: Any,
    info: dict,
) -> Tuple[Any, dict]:
    """Compute the per-step reward and a per-component breakdown.

    Args:
        state:      EnvPipelineState before the step. Reads jump_count and
                    cycle_peak_sum (used by the end-of-episode bonuses).
        action:     jax.Array (5,), already clipped to [-1, 1] by the env.
                    Not used by v0 (action smoothness is precomputed in info).
        next_state: EnvPipelineState after the step. Reads max_cycle_peak_gain.
        info:       Transient scalars/arrays computed by the env step.
                    Required keys (all jax.Array unless noted):
                      tilt, roll_rate, pitch_rate,
                      vx, vy, body_vz,
                      n_feet_on, all_feet_off,
                      height_gain, peak_gain, current_cycle_max_tilt,
                      cycle_completed, action_rate, xy_drift,
                      tau_motor, terminated, done.

    Returns:
        (reward, components) where reward is a JAX scalar and components is a
        dict[str, jax.Array] of signed per-term contributions. The components
        sum to `reward` in the order documented below (within float32 ULPs).
    """
    del action  # unused in v0; reserved for future smoothness-on-raw-action.

    tilt                   = info["tilt"]
    roll_rate              = info["roll_rate"]
    pitch_rate             = info["pitch_rate"]
    vx                     = info["vx"]
    vy                     = info["vy"]
    body_vz                = info["body_vz"]
    n_feet_on              = info["n_feet_on"]
    all_feet_off           = info["all_feet_off"]
    height_gain            = info["height_gain"]
    peak_gain              = info["peak_gain"]
    current_cycle_max_tilt = info["current_cycle_max_tilt"]
    cycle_completed        = info["cycle_completed"]
    action_rate            = info["action_rate"]
    xy_drift               = info["xy_drift"]
    tau_motor              = info["tau_motor"]
    terminated             = info["terminated"]
    done                   = info["done"]

    new_max_cycle_peak_gain = next_state.max_cycle_peak_gain
    prev_jump_count         = state.jump_count
    prev_cycle_peak_sum     = state.cycle_peak_sum

    # ── Build the scalar reward in the SAME order as the original env block.
    # Each component is computed with the same expression as the original so
    # IEEE-754 float32 arithmetic matches bit-exactly. Components are also
    # captured in the dict for diagnostic attribution.

    reward = jp.float32(0.0)

    # 1. Tilt²  (line 599 in original)
    reward = reward - 2.0 * tilt * tilt
    rew_tilt_pen = -2.0 * tilt * tilt

    # 2. Angular-velocity damping  (602)
    reward = reward - 0.2 * (roll_rate + pitch_rate)
    rew_angvel_damp = -0.2 * (roll_rate + pitch_rate)

    # 2b. Horizontal velocity²  (613)
    reward = reward - 2.0 * (vx ** 2 + vy ** 2)
    rew_horiz_vel_pen = -2.0 * (vx ** 2 + vy ** 2)

    # 3. Upward velocity, grounded + level gate  (621–622)
    rew_upward_vel = jp.where(
        (n_feet_on >= 3) & (tilt < jp.float32(0.524)),
        5.0 * jp.maximum(0.0, body_vz), 0.0)
    reward = reward + rew_upward_vel

    # 4. Quadratic flight height  (638–640)
    height_gain_capped = jp.minimum(jp.maximum(0.0, height_gain), 0.4)
    rew_flight_height = jp.where(
        all_feet_off, 150.0 * height_gain_capped ** 2, 0.0)
    reward = reward + rew_flight_height

    # 5. Cycle completion bonus, height² × upright  (646–651)
    height_scale = jp.clip(
        (peak_gain / jp.float32(0.03)) ** 2, 0.0, 16.0)
    cycle_upright = jp.maximum(
        0.0, 1.0 - current_cycle_max_tilt / jp.float32(0.436))
    rew_cycle_bonus = jp.where(
        cycle_completed, 100.0 * height_scale * cycle_upright, 0.0)
    reward = reward + rew_cycle_bonus

    # 6. XY drift, dead-zone  (659–660)
    drift_excess = jp.maximum(0.0, xy_drift - 0.15)
    reward = reward - 3.0 * drift_excess ** 2
    rew_drift_pen = -3.0 * drift_excess ** 2

    # 7. Action smoothness  (663) — v1 (2026-04-20): raised 0.05 → 1.0 to
    # make bang-bang action exploits unprofitable (Run 15 post-mortem).
    reward = reward - 1.0 * action_rate
    rew_action_smoothness = -1.0 * action_rate

    # 8. Torque cost  (666)
    reward = reward - 0.0001 * jp.sum(tau_motor ** 2)
    rew_torque_cost = -0.0001 * jp.sum(tau_motor ** 2)

    # Termination overrides everything per-step  (672)
    reward = jp.where(terminated, jp.float32(-1000.0), reward)
    # For attribution: the delta the termination clamp introduced.
    pre_term_reward = (
        rew_tilt_pen + rew_angvel_damp + rew_horiz_vel_pen
        + rew_upward_vel + rew_flight_height + rew_cycle_bonus
        + rew_drift_pen + rew_action_smoothness + rew_torque_cost)
    rew_termination = jp.where(
        terminated, jp.float32(-1000.0) - pre_term_reward, jp.float32(0.0))

    # 9. End-of-episode: 3000 × max single-cycle peak  (675–676)
    rew_end_max_peak = jp.where(
        done & ~terminated, 3000.0 * new_max_cycle_peak_gain, 0.0)
    reward = reward + rew_end_max_peak

    # 9b. End-of-episode: successive jumps, capped  (682–684)
    rew_end_successive = jp.where(
        done & ~terminated,
        100.0 * jp.minimum(prev_jump_count.astype(jp.float32), 5.0),
        0.0)
    reward = reward + rew_end_successive

    # 9c. End-of-episode: mean peak across cycles  (689–694)
    mean_peak = jp.where(
        prev_jump_count > 0,
        prev_cycle_peak_sum / jp.maximum(
            prev_jump_count.astype(jp.float32), 1.0),
        0.0)
    rew_end_mean_peak = jp.where(
        done & ~terminated, 500.0 * mean_peak, 0.0)
    reward = reward + rew_end_mean_peak

    # 10. End-of-episode: never-jumped penalty  (697–699)
    rew_end_never_jumped = jp.where(
        done & ~terminated
        & (new_max_cycle_peak_gain <= jp.float32(0.001)),
        -500.0, 0.0)
    reward = reward + rew_end_never_jumped

    components = {
        "tilt_pen":          rew_tilt_pen,
        "angvel_damp":       rew_angvel_damp,
        "horiz_vel_pen":     rew_horiz_vel_pen,
        "upward_vel":        rew_upward_vel,
        "flight_height":     rew_flight_height,
        "cycle_bonus":       rew_cycle_bonus,
        "drift_pen":         rew_drift_pen,
        "action_smoothness": rew_action_smoothness,
        "torque_cost":       rew_torque_cost,
        "termination_pen":   rew_termination,
        "end_max_peak":      rew_end_max_peak,
        "end_successive":    rew_end_successive,
        "end_mean_peak":     rew_end_mean_peak,
        "end_never_jumped":  rew_end_never_jumped,
    }
    return reward, components
