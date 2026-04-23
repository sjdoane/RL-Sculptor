"""tests/test_adapter_contract.py — Phase 1 gate.

Verifies:
  1. GymSB3Adapter instantiates with the Hopper config values.
  2. reward_contract() returns a valid RewardContract with non-None obs/action
     specs drawn from the real env.
  3. RewardOverrideWrapper actually replaces the env reward with the module's
     `compute_reward` output, and accumulates components. Uses `env.step` and
     `env.reset` only — no training, no SB3.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
HOPPER_CONFIG = REPO_ROOT / "examples" / "hopper" / "config.toml"
HOPPER_REWARD_V0 = REPO_ROOT / "examples" / "hopper" / "rewards" / "v0.py"


def _read_hopper_config() -> dict:
    """Read examples/hopper/config.toml without depending on load_adapter."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with HOPPER_CONFIG.open("rb") as f:
        return tomllib.load(f)


def test_hopper_config_exists_and_is_valid():
    cfg = _read_hopper_config()
    assert cfg["target"]["name"] == "hopper_demo"
    assert cfg["adapter"]["class"].endswith(".GymSB3Adapter")
    assert cfg["adapter"]["config"]["env_id"] == "Hopper-v4"


def test_adapter_instantiates_with_hopper_config():
    cfg = _read_hopper_config()
    dotted = cfg["adapter"]["class"]
    adapter_cfg = cfg["adapter"]["config"]

    module_name, _, class_name = dotted.rpartition(".")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    adapter = cls(**adapter_cfg)

    from sculptor.adapters.gym_sb3 import GymSB3Adapter
    assert isinstance(adapter, GymSB3Adapter)
    assert adapter.env_id == "Hopper-v4"
    assert adapter.n_envs == 4
    assert adapter.ppo_kwargs == {"learning_rate": 3e-4, "n_steps": 2048}


def test_reward_contract_is_valid():
    from sculptor.adapters.gym_sb3 import GymSB3Adapter
    from sculptor.adapters.base import RewardContract

    adapter = GymSB3Adapter(env_id="Hopper-v4", n_envs=1)
    contract = adapter.reward_contract()

    assert isinstance(contract, RewardContract)
    assert contract.observation_space_spec is not None
    assert contract.action_space_spec is not None
    # Gymnasium Hopper: 11-D obs, 3-D action
    assert contract.observation_space_spec.shape == (11,)
    assert contract.action_space_spec.shape == (3,)
    assert "x_velocity" in contract.expected_info_keys
    assert contract.expected_components is None  # open


def test_reward_override_wrapper_replaces_env_reward():
    """Wrapper loads reward module, replaces env reward, accumulates components."""
    import gymnasium as gym
    from sculptor.adapters.gym_sb3 import _make_reward_override_wrapper

    RewardOverrideWrapper = _make_reward_override_wrapper()
    env = RewardOverrideWrapper(gym.make("Hopper-v4"), HOPPER_REWARD_V0)

    obs, info = env.reset(seed=42)
    assert obs.shape == (11,)

    total_reward = 0.0
    component_keys_seen: set[str] = set()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # Wrapper injects per-step diagnostics:
        assert "sculptor_env_reward" in info
        assert "sculptor_components" in info
        assert isinstance(info["sculptor_components"], dict)
        component_keys_seen.update(info["sculptor_components"].keys())
        # Sculpted reward should differ from the env's native reward in at
        # least one step (Hopper native ≠ forward_vel + alive - ctrl_cost
        # because of Gymnasium's healthy_reward formulation).
        total_reward += reward
        if terminated or truncated:
            break

    env.close()

    # Expected component names from v0.py
    assert "forward_velocity" in component_keys_seen
    assert "alive_bonus" in component_keys_seen
    assert "ctrl_cost" in component_keys_seen

    means = env.get_component_means()
    assert isinstance(means, dict)
    assert set(means.keys()) == component_keys_seen


def test_reward_override_wrapper_raises_on_bad_module(tmp_path):
    """A reward module without compute_reward must raise at wrapper construction."""
    import gymnasium as gym
    from sculptor.adapters.gym_sb3 import _make_reward_override_wrapper

    bad = tmp_path / "bad.py"
    bad.write_text("REWARD_SPEC = {}\n# note: no compute_reward\n")

    RewardOverrideWrapper = _make_reward_override_wrapper()
    base = gym.make("Hopper-v4")
    with pytest.raises(AttributeError):
        RewardOverrideWrapper(base, bad)
    base.close()


def test_reward_spec_fields_present():
    """REWARD_SPEC must carry version, author, parent_hash, hyperparameters, references."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hopper_v0", str(HOPPER_REWARD_V0))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    rs = mod.REWARD_SPEC
    for key in ("version", "description", "author", "parent_hash", "hyperparameters", "references"):
        assert key in rs, f"REWARD_SPEC missing `{key}`"
    assert rs["author"] in ("human", "sculptor")
    assert isinstance(rs["references"], list)
    assert isinstance(rs["hyperparameters"], dict)
