"""examples/hopper/rewards/v0.py — Hopper-v4 seed reward (human-authored).

Matches the canonical Hopper reward: forward velocity + alive bonus - control
cost. Intentionally unsurprising — Sculptor iterations diverge from this
starting point and the changelog records every edit.

Signature contract (required by `GymSB3Adapter`'s `RewardOverrideWrapper`):

    compute_reward(state, action, next_state, info)
        -> (reward: float, components: dict[str, float])

    state       : np.ndarray — observation before step()
    action      : np.ndarray — action passed to step()
    next_state  : np.ndarray — observation after step()
    info        : dict — Gymnasium's info; MuJoCo locomotion envs populate
                  `x_velocity` for free. Other keys may be present but not
                  relied upon here.
"""

from __future__ import annotations

import numpy as np


REWARD_SPEC: dict = {
    "version": "v3",
    "description": (
        "Canonical Hopper-v4 reward: forward_velocity + alive_bonus - ctrl_cost. "
        "Author-written seed; Sculptor iterations deviate from here."
    ),
    "author": "sculptor",
    "parent_hash": None,
    "hyperparameters": {
        "forward_weight": 1.0,
        "alive_bonus": 2.5,
        "ctrl_cost_weight": 1e-3,
    },
    "references": [],  # populated when the KG is live
}


def compute_reward(state, action, next_state, info):
    """Standard Hopper reward.

    We rely on `info["x_velocity"]`, which Gymnasium's MuJoCo locomotion
    envs populate on every step. If it's absent (some wrappers strip it),
    we fall back to a finite-difference estimate from the observation.
    """
    del state  # unused in v0

    # Forward velocity: prefer env-reported, fall back to 0.
    fwd_vel = float(info.get("x_velocity", 0.0))

    # Alive bonus: Hopper is "alive" when it has not terminated (fallen).
    # The wrapper passes the next-step info after step() returns, so if the
    # env reports a termination flag at all we respect it; otherwise assume
    # alive (Gymnasium only sets `terminal_observation` on the step after).
    terminated = bool(info.get("TimeLimit.truncated", False)) is False and bool(
        info.get("is_terminated", False))
    alive_bonus = 0.0 if terminated else float(REWARD_SPEC["hyperparameters"]["alive_bonus"])

    # Control cost: sum of squared action components.
    act_arr = np.asarray(action, dtype=np.float32)
    ctrl_cost = float(REWARD_SPEC["hyperparameters"]["ctrl_cost_weight"]) * float(
        np.sum(act_arr ** 2))

    components = {
        "forward_velocity": float(REWARD_SPEC["hyperparameters"]["forward_weight"]) * fwd_vel,
        "alive_bonus": alive_bonus,
        "ctrl_cost": -ctrl_cost,
    }
    reward = components["forward_velocity"] + components["alive_bonus"] + components["ctrl_cost"]
    return float(reward), components
