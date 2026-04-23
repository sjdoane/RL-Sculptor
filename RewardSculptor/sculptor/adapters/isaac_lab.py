"""sculptor/adapters/isaac_lab.py — Isaac Lab adapter (STUB).

Status
======
Scaffold only. `train()` and `rollout()` raise NotImplementedError. The
ABC surface, reward_contract, probe_component, and compute_behavior_metrics
are fully implemented so:

  - `sculptor.adapters.load_adapter("config.toml")` works against a
    config pointing here without blowing up.
  - The UI adapter picker can display a real "Coming soon" option backed
    by this class, with the completion checklist surfacing in tooltips.
  - Type-checking remains green across the codebase.

Target library
==============
Isaac Lab (https://github.com/isaac-sim/IsaacLab) — at least
`isaaclab >= 0.2.0`, CUDA 12.4+ required. Linux x86_64 only (Isaac Lab
has no macOS/WSL-native support as of the time of writing).

Minimal viable implementation path
==================================

1.  Install Isaac Lab via its installer (`./isaaclab.sh --install`). This
    brings in Isaac Sim which is heavy — budget 60+ GB of disk.

2.  Reward injection mirrors the mjlab pattern. Isaac Lab's manager-based
    env exposes `RewardManagerCfg` with a dict of `RewardTermCfg` entries.
    Add a single `SculptorRewardTerm` (class-based, with `reset(env_ids)`
    hook to zero history) and set all other task-shipped terms to
    `weight=0.0`. Set `RewardManagerCfg.scale_rewards_by_dt = False`.

    The sculptor reward module's `compute_reward_batched` operates on
    `(num_envs, *feature_shape)` torch tensors on CUDA and returns a
    `(num_envs,)` tensor. The SculptorRewardTerm's `__call__(env, **params)`
    snapshots state off `env.scene[robot_name].data.{joint_pos, joint_vel,
    root_lin_vel_b, root_ang_vel_b, projected_gravity_b, actuator_force}`
    + `env.command_manager.get_command(...)`, stores `prev_state`, and
    dispatches.

3.  Training entry point: subprocess-invoke Isaac Lab's
    `scripts/reinforcement_learning/rsl_rl/train.py` (or the rl_games
    variant, depending on which RL library the task targets).
    Isaac Lab writes checkpoints under `logs/rsl_rl/<experiment>/model_<N>.pt`;
    scan for the highest-numbered file after training completes.

4.  Rollout: subprocess-invoke Isaac Lab's corresponding `play.py` with
    `--checkpoint`, `--video`, and `--num_envs 1`. It writes an mp4 +
    exits.

5.  `compute_behavior_metrics` parses Isaac Lab's tensorboard event files
    (or the summary CSV if configured) for `mean_reward`,
    `mean_episode_length`, `success_rate` (for manipulation), and the
    task-specific custom metrics the author registered on the
    `MetricsManager`.

Known gotchas
=============

- **Headless mode.** Set `HEADLESS=1` + `LIVESTREAM=0` before spawning
  the subprocess or Isaac Sim will try to open a GUI and hang on remote
  hosts.
- **Reset semantics.** Isaac Lab's rsl_rl integration calls `env.reset()`
  AT iteration boundaries; the SculptorRewardTerm's `reset(env_ids)` hook
  must zero `prev_state[env_ids]` — not the whole buffer — or rewards
  spike on individual-env timeouts.
- **Device pinning.** Isaac Lab defaults to `cuda:0`; multi-GPU via
  `--headless --device cuda:N` but this lands post-M6.

Sculptor-specific overrides documented in docs/adapters.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)


_ADOPTION_GUIDE_URL = "docs/adapters/isaac.md"
_EFFORT = "4-8 hours for a senior Isaac Lab user"


@dataclass
class IsaacLabAdapter(SculptorAdapter):
    """Isaac Lab adapter — NOT YET IMPLEMENTED. See module docstring."""

    task: str = "Isaac-Velocity-Flat-Unitree-A1-v0"
    num_envs: int = 4096
    device: str = "cuda:0"
    max_iterations: int = 1500

    def reward_contract(self) -> RewardContract:
        # Defensible defaults derived from a canonical Isaac-Lab velocity
        # task. Replace with introspected values once `train()` lands.
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=[
                "qpos", "qvel", "base_lin_vel_b", "base_ang_vel_b",
                "projected_gravity_b", "actuator_force", "command_vel",
            ],
            expected_components=None,
            supports_batched=True,
            training_device="gpu",
            min_gpu_memory_gb=8.0,
            state_schema={
                "qpos": (18,),
                "qvel": (18,),
                "base_lin_vel_b": (3,),
                "base_ang_vel_b": (3,),
                "projected_gravity_b": (3,),
                "actuator_force": (12,),
                "command_vel": (3,),
            },
        )

    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        return ComponentProbe(
            ok=False,
            error=(
                "IsaacLabAdapter.probe_component is not implemented. "
                f"See {_ADOPTION_GUIDE_URL} for the completion checklist."
            ),
        )

    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict[str, Any]:
        return {
            "mean_return": 0.0,
            "mean_episode_length": 0,
            "fall_rate": 0.0,
            "adapter_status": "stub",
        }

    def train(
        self,
        reward_module_path: Path,
        output_dir: Path,
        steps: int,
        seed: int,
    ) -> TrainResult:
        raise NotImplementedError(
            f"Isaac Lab adapter not yet implemented. "
            f"Adoption guide: {_ADOPTION_GUIDE_URL}. "
            f"Estimated effort: {_EFFORT}."
        )

    def rollout(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        n_episodes: int,
    ) -> RolloutResult:
        raise NotImplementedError(
            f"Isaac Lab adapter not yet implemented. "
            f"Adoption guide: {_ADOPTION_GUIDE_URL}. "
            f"Estimated effort: {_EFFORT}."
        )
