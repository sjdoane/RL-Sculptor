"""sculptor/adapters/base.py — the load-bearing public surface of Sculptor.

# ============================================================================
# Adapter interface — for external lab contributors
# ============================================================================
#
# A `SculptorAdapter` is the ONLY thing Sculptor needs to know about your
# codebase. The inner loop (generate -> train -> rollout -> diagnose -> edit) is
# adapter-agnostic; the literature-grounded editor is adapter-agnostic; the
# knowledge graph is adapter-agnostic. Everything env- or stack-specific lives
# behind this interface.
#
# The contract (must implement):
# ----------------------------------------------------------------------------
#   train(reward_module_path, output_dir, steps, seed) -> TrainResult
#   rollout(checkpoint_path, output_dir, n_episodes) -> RolloutResult
#   compute_behavior_metrics(rollout) -> dict[str, Any]
#   reward_contract() -> RewardContract
#   probe_component(reward_module_path) -> ComponentProbe
#
# Probe shape is adapter-defined because mjlab's zero-dummy state is a dict
# of zero tensors (per-key shapes from state_schema), whereas Gymnasium's is
# a zero np.float32 array of observation_space.shape.
#
# Batched reward path:
# ----------------------------------------------------------------------------
#   reward_batched(reward_module_path, state_batch, action_batch,
#                   next_state_batch, info_batch) -> (rewards, components)
#
#   Default implementation loops the scalar path with a runtime warning.
#   Adapters whose RewardContract declares `supports_batched=True` (mjlab)
#   override this with a true batched path that calls the reward module's
#   `compute_reward_batched` entry point, which operates on dicts of
#   (num_envs, *shape) tensors and returns a (num_envs,) reward tensor.
#
# What's fixed vs. adapter-defined:
# ----------------------------------------------------------------------------
#   Fixed by Sculptor:
#     - The dataclass shapes returned by the five methods.
#     - The reward-module API: `compute_reward(...) -> (reward, components)`
#       (scalar path, required for EVERY adapter) and OPTIONALLY
#       `compute_reward_batched(...)` when the module is targeted at a
#       batched adapter. The REWARD_SPEC dict format is unchanged apart
#       from the new `supports_batched: bool` field (defaults False).
#     - The output directory layout under `output_dir` (sculptor expects to
#       find `metrics.json`, `rollout.mp4`, `keyframes/`, `trajectory.npz`,
#       `behavior.json` by name).
#
#   Adapter-defined:
#     - Reward injection mechanism (see patterns below).
#     - Concrete obs/action specs (whatever your env uses).
#     - Behavior metrics (whatever your domain cares about).
#     - The exact `info` dict and state schema the reward function receives.
#     - Whether components are a closed set or open.
#
# Recommended reward-injection patterns by stack:
# ----------------------------------------------------------------------------
#   Gymnasium + SB3 / CleanRL  (the reference path; see `gym_sb3.py`)
#     Wrap the env in a `RewardOverrideWrapper` that loads the reward module
#     once at construction and calls its `compute_reward` inside `step()`.
#     supports_batched=False.
#
#   mjlab (manager-based, MuJoCo-Warp)  (see `mjlab.py`)
#     Patch the task cfg's `rewards` dict to contain a single active term
#     (`SculptorRewardTerm`) whose `func` is a class-based term that loads
#     the sculpted reward module and dispatches to its
#     `compute_reward_batched` on each step, returning a (num_envs,) tensor
#     on device. Other task-shipped reward terms are set to weight=0.0.
#     Also sets `cfg.scale_rewards_by_dt=False` so the module's raw
#     per-step reward survives to training without dt-scaling.
#
#   Isaac Gym / Isaac Lab  (stub — see `isaac_lab.py`)
#     Same shape as mjlab; patch the task's RewardManagerCfg.
#
#   Brax / MJX  (stub — see `mjx.py`)
#     Inject via `reward_fn=` at env init. JAX-pure.
#
#   RLlib  (stub — see `rllib.py`)
#     env_creator callback; worker-aware reward-module path shipping.
#
# Things adapters MUST NOT do:
# ----------------------------------------------------------------------------
#   - Train longer than the requested `steps` budget.
#   - Mutate the reward module file from inside `train()`.
#   - Add fields to the dataclasses that Sculptor depends on.
#   - Silently fall back to the env's native reward when the reward module
#     fails to load — raise.
# ============================================================================
"""

from __future__ import annotations

import importlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class TrainResult:
    """Outcome of a single training run."""

    checkpoint_path: Path
    metrics_dict: dict[str, float]
    component_means: dict[str, float]
    logs_path: Path


@dataclass
class RolloutResult:
    """Artifacts produced by a rollout."""

    video_path: Path
    keyframes_dir: Path
    trajectory_path: Path
    n_episodes: int


@dataclass
class ComponentProbe:
    """Result of running a reward module on zero-dummy inputs.

    Returned by `SculptorAdapter.probe_component`. The UI's pydantic
    `ComponentProbe` (backend/models/reward.py) mirrors this shape.
    """

    ok: bool
    components: Optional[dict[str, float]] = None
    total: Optional[float] = None
    error: Optional[str] = None


@dataclass
class RewardContract:
    """What a reward module for this adapter must look like."""

    observation_space_spec: Any
    action_space_spec: Any
    expected_info_keys: list[str] = field(default_factory=list)
    expected_components: Optional[list[str]] = None

    # Batched-reward support (MJLAB_PIVOT_DESIGN §2.2).
    # Defaults preserve the original scalar-only contract — existing reward
    # modules stay valid without modification.
    supports_batched: bool = False
    # Where the adapter trains. "cpu" = CPU only, "gpu" = requires CUDA,
    # "any" = either works. Consumed by project-creation validation.
    training_device: Literal["cpu", "gpu", "any"] = "any"
    # Conservative minimum VRAM (GiB) at the adapter's default num_envs.
    # Pre-creation VRAM check uses this as the floor; the adapter's runtime
    # probe refines a per-task coefficient after first training run.
    min_gpu_memory_gb: Optional[float] = None
    # For batched adapters: per-key feature shape of the state / next_state
    # dict (excluding the leading num_envs batch dim). e.g.
    # {"qpos": (18,), "qvel": (17,), "command_vel": (3,)}.
    # The reward module's compute_reward_batched receives dict[str, Tensor]
    # where tensor.shape == (num_envs, *state_schema[k]).
    state_schema: Optional[dict[str, tuple[int, ...]]] = None


class SculptorAdapter(ABC):
    """Bridge between Sculptor and a specific RL stack + env. See module doc."""

    @abstractmethod
    def train(
        self,
        reward_module_path: Path,
        output_dir: Path,
        steps: int,
        seed: int,
    ) -> TrainResult: ...

    @abstractmethod
    def rollout(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        n_episodes: int,
    ) -> RolloutResult: ...

    @abstractmethod
    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict[str, Any]: ...

    @abstractmethod
    def reward_contract(self) -> RewardContract: ...

    @abstractmethod
    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        """Run `compute_reward` on adapter-specific zero-dummy inputs.

        Must run in a subprocess — user-supplied reward code is not trusted
        inside the main process. Returns a ComponentProbe carrying
        (ok, components, total, error).
        """
        ...

    def reward_batched(
        self,
        reward_module_path: Path,
        state_batch: Any,
        action_batch: Any,
        next_state_batch: Any,
        info_batch: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Default batched reward: loop scalar path over the leading batch dim.

        Adapters whose `reward_contract().supports_batched` is True should
        override to call the reward module's `compute_reward_batched` entry
        point for real GPU-batched performance. The default here is a
        correctness fallback only — emits a RuntimeWarning and is O(N) on
        batch size N.

        Assumes:
          - state_batch / next_state_batch are either dicts of per-key
            tensors/arrays with leading batch dim N, or indexable sequences.
          - action_batch has leading batch dim N.
          - info_batch is a dict of per-key values; scalar-typed values are
            broadcast (same value given to every env).
        Returns:
          (rewards, components) where `rewards` is a numpy array of shape (N,)
          and `components` is dict[str, np.ndarray of shape (N,)].
        """
        import numpy as np

        warnings.warn(
            f"{type(self).__name__}.reward_batched using scalar fallback loop. "
            f"Override for GPU-batched performance.",
            RuntimeWarning,
            stacklevel=2,
        )

        def _slice(container: Any, i: int) -> Any:
            if isinstance(container, dict):
                return {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in container.items()}
            return container[i]

        def _info_at(info: Any, i: int) -> dict:
            if not isinstance(info, dict):
                return {}
            out = {}
            for k, v in info.items():
                try:
                    out[k] = v[i] if hasattr(v, "__getitem__") and not isinstance(v, str) else v
                except (IndexError, TypeError):
                    out[k] = v
            return out

        n = len(action_batch)
        mod = _import_reward_module(reward_module_path)
        rewards: list[float] = []
        comps_acc: dict[str, list[float]] = {}
        for i in range(n):
            r, c = mod.compute_reward(
                _slice(state_batch, i),
                action_batch[i],
                _slice(next_state_batch, i),
                _info_at(info_batch, i),
            )
            rewards.append(float(r))
            for k, v in c.items():
                comps_acc.setdefault(k, []).append(float(v))
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        comps_arr = {k: np.asarray(v, dtype=np.float32) for k, v in comps_acc.items()}
        return rewards_arr, comps_arr


def _import_reward_module(path: Path) -> Any:
    """Helper — import a reward module by file path.

    Kept at module level so subclasses can reuse it in probe_component
    subprocess scripts that need a scalar-path loader.
    """
    import importlib.util as _util

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mod_name = f"_sculptor_adapter_reward_{abs(hash(str(path)))}"
    spec = _util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not spec reward module at {path}")
    mod = _util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "compute_reward"):
        raise AttributeError(f"{path} must define compute_reward(state, action, next_state, info)")
    if not hasattr(mod, "REWARD_SPEC"):
        raise AttributeError(f"{path} must define REWARD_SPEC dict")
    return mod


# ── Phase 2: config-driven adapter loader ───────────────────────────────────
def load_adapter(config_path: Path) -> SculptorAdapter:
    """Read `config.toml`, import the adapter class, instantiate with config.

    The config file must have an `[adapter]` table with:
        class  = "fully.qualified.AdapterClassName"
        config = { ...kwargs passed to the adapter __init__... }
    """
    config_path = Path(config_path)
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    if "adapter" not in cfg:
        raise ValueError(f"{config_path}: missing [adapter] section")
    adapter_cfg = cfg["adapter"]
    dotted = adapter_cfg.get("class")
    if not dotted:
        raise ValueError(f"{config_path}: [adapter].class is required")
    init_kwargs = adapter_cfg.get("config", {}) or {}

    module_name, _, class_name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"adapter.class must be a dotted path, got {dotted!r}")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    instance = cls(**init_kwargs)
    if not isinstance(instance, SculptorAdapter):
        raise TypeError(
            f"{dotted} must subclass SculptorAdapter, got {type(instance).__name__}"
        )
    return instance
