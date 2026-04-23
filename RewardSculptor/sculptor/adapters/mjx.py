"""sculptor/adapters/mjx.py — Brax / MJX adapter (STUB).

Status
======
Scaffold only. `train()` and `rollout()` raise NotImplementedError. The
ABC surface, reward_contract, probe_component, and compute_behavior_metrics
are fully implemented so the class can be discovered, loaded, and
type-checked without the rest of the stack.

Target library
==============
Brax (https://github.com/google/brax) + MJX — at least
`brax >= 0.12.0`, `jax >= 0.7.0`, `jaxlib-cuda12 >= 0.7.0`. CUDA 12.x
required; CPU-only jax works but is too slow to be useful.

Minimal viable implementation path
==================================

1.  Install: `uv add "brax>=0.12" "jax[cuda12]>=0.7" "mujoco-mjx>=3.3"`.

2.  Reward injection pattern: Brax envs accept a custom `reward_fn` at
    construction time OR expose their reward computation as a pure JAX
    function that can be replaced by wrapping `brax.envs.Env` subclass.
    The sculptor reward module must export `compute_reward_batched` as a
    JAX-pure callable (no Python branching on traced values). Verify via
    a JIT preflight in `train()`:

        @jax.jit
        def _preflight(sculpt_fn, dummy_state_batch, dummy_action_batch):
            return sculpt_fn(dummy_state_batch, dummy_action_batch, ...)

    If JIT compilation fails, surface the trace error in
    RewardValidationError and abort before spawning the trainer.

3.  Reference injection for the AME456 quadruped is already sketched in
    `RewardSculptor/sculptor/reward.py` (legacy v1 quadruped reward). The
    MJX env in AME456/quadruped_mjx_env.py imports that module; an MJX
    adapter would instead read the project's current v<n>.py and pass
    the module's `compute_reward_batched` into the env at
    `MJXAdapter._make_env(reward_fn=...)`.

4.  Training: use brax's PPO implementation
    (`brax.training.agents.ppo.train.train`) or import a research-style
    custom loop from Acme / JaxMARL — both are in common use for MJX
    workloads. Save params as a pickle of JAX `PyTree`; decode to torch
    at rollout time for compatibility with sculptor's trajectory.npz
    format.

5.  Rollout: drive the env at num_envs=1 on CPU for reproducible mp4s.
    Use `imageio-ffmpeg` for encoding (already a sculptor dep).

Known gotchas
=============

- **Pure-functional constraint.** Any Python dict of tensors the sculptor
  reward module wants to read MUST be converted to a JAX PyTree — no
  `torch.Tensor` survives the `@jax.jit` boundary. Explicit conversion
  in the SculptorRewardTerm wrapper.
- **Determinism.** Brax's scan over timesteps requires a seeded PRNG
  key threaded through every reward call. `compute_reward_batched` must
  accept an optional `rng` kwarg or be RNG-free.
- **vmap vs pmap.** Single-GPU (M6 scope) uses vmap; multi-GPU uses pmap
  and is deferred.

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


_ADOPTION_GUIDE_URL = "docs/adapters/mjx.md"
_EFFORT = "4-6 hours for a JAX-familiar contributor"


@dataclass
class MjxAdapter(SculptorAdapter):
    """Brax / MJX adapter — NOT YET IMPLEMENTED. See module docstring."""

    env_id: str = "ant"
    num_envs: int = 4096
    device: str = "cuda"
    num_timesteps: int = 50_000_000

    def reward_contract(self) -> RewardContract:
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=[
                "qpos", "qvel", "body_vel", "body_ang_vel", "contact_force",
            ],
            expected_components=None,
            supports_batched=True,
            training_device="gpu",
            min_gpu_memory_gb=6.0,
            state_schema={
                "qpos": (15,),
                "qvel": (14,),
                "body_vel": (3,),
                "body_ang_vel": (3,),
                "contact_force": (4,),
            },
        )

    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        return ComponentProbe(
            ok=False,
            error=(
                "MjxAdapter.probe_component is not implemented. "
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
            f"Brax / MJX adapter not yet implemented. "
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
            f"Brax / MJX adapter not yet implemented. "
            f"Adoption guide: {_ADOPTION_GUIDE_URL}. "
            f"Estimated effort: {_EFFORT}."
        )
