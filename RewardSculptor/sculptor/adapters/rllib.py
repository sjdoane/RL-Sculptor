"""sculptor/adapters/rllib.py — Ray RLlib adapter (STUB).

Status
======
Scaffold only. `train()` and `rollout()` raise NotImplementedError. The
ABC surface is fully implemented so the class can be discovered, loaded,
and type-checked without the rest of the stack.

Target library
==============
Ray RLlib (https://docs.ray.io/en/latest/rllib/index.html) — at least
`ray[rllib] >= 2.10`. Works on CPU or GPU (RLlib handles device-placement
internally per-worker). Multi-node training is a first-class feature.

Minimal viable implementation path
==================================

1.  Install: `uv add "ray[rllib]>=2.10"`.

2.  Reward injection pattern: RLlib composes envs via `env_creator`
    callbacks registered on the algorithm's config. Wrap the base env
    in a `RewardOverrideWrapper` analog (see `gym_sb3.py` for the
    reference pattern) and register the wrapper as the env:

        config.environment(
            env=lambda env_cfg: RewardOverrideWrapper(
                gym.make(env_cfg["base_env_id"]),
                env_cfg["reward_module_path"],
            ),
            env_config={"base_env_id": self.env_id,
                        "reward_module_path": str(reward_module_path)},
        )

    Each rollout worker instantiates the env on its own process. The
    reward module is loaded per-worker; keep reward-module path
    absolute or shipped via Ray's `runtime_env` so every worker can
    resolve it.

3.  Training: `config.build().train()` — synchronous single-call
    trainer; iterate until `steps` budget is spent. Save checkpoint via
    `algo.save(checkpoint_dir)`; RLlib writes a multi-file directory
    (`checkpoint_<N>/`), not a single .pt. Adjust sculptor's
    `TrainResult.checkpoint_path` semantics to point at the directory.

4.  Rollout: `algo.from_checkpoint(path)` + `algo.compute_single_action`
    in a single-env loop. RLlib's built-in video recorder can attach to
    the env via `add_video_recorder` but reliability is mixed; prefer
    external frame capture via the env's `render_mode="rgb_array"` + the
    existing `_write_mp4` helper from `gym_sb3.py`.

5.  `compute_behavior_metrics` — parse `result_dict` returned by each
    `algo.train()` call. Standard keys: `env_runners/episode_reward_mean`,
    `env_runners/episode_len_mean`, `env_runners/num_env_steps_sampled`.

Known gotchas
=============

- **Worker remoting.** RLlib spawns rollout workers as Ray actors, which
  run in separate processes. Any sculptor in-memory state (loaded
  modules, counters) does NOT cross the actor boundary. All reward-module
  coordination must go through the filesystem (module path) or Ray's
  `runtime_env`.
- **Ray's process pool.** `ray.init()` takes several seconds. For short
  training budgets, this overhead dominates; document it. For M2,
  the adapter is scoped to single-node local training
  (`num_workers=0` + `num_envs_per_worker=4` is a sane default).
- **Checkpoint format.** RLlib's checkpoint is a directory, not a file.
  Sculptor's `TrainResult.checkpoint_path: Path` still works (Path can
  point at a dir), but `torch.load()` will not — rollout code must
  use `Algorithm.from_checkpoint`.

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


_ADOPTION_GUIDE_URL = "docs/adapters/rllib.md"
_EFFORT = "4-8 hours (worker coordination + checkpoint shape)"


@dataclass
class RllibAdapter(SculptorAdapter):
    """Ray RLlib adapter — NOT YET IMPLEMENTED. See module docstring."""

    env_id: str = "Hopper-v4"
    algorithm: str = "PPO"
    num_workers: int = 0
    num_envs_per_worker: int = 4
    framework: str = "torch"

    def reward_contract(self) -> RewardContract:
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=["x_velocity"],
            expected_components=None,
            supports_batched=False,  # RLlib adapter stays on scalar path
            training_device="any",
            min_gpu_memory_gb=None,
            state_schema=None,
        )

    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        return ComponentProbe(
            ok=False,
            error=(
                "RllibAdapter.probe_component is not implemented. "
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
            f"Ray RLlib adapter not yet implemented. "
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
            f"Ray RLlib adapter not yet implemented. "
            f"Adoption guide: {_ADOPTION_GUIDE_URL}. "
            f"Estimated effort: {_EFFORT}."
        )
