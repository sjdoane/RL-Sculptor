"""Real-GPU smoke test for MjlabAdapter (M2 verification #4).

Invocation:
    cd ~/projects/RewardSculptor && uv run pytest -m gpu

Requires:
    - NVIDIA GPU + CUDA 12.4+ installed.
    - `mjlab[cu128]` installed in the sculptor uv env.

Budget:
    - First run: ~5 min wall-clock (100 iters Go1 train + checkpoint write).
    - Subsequent runs: <10 s (fixture load from cache).
    - Force retrain with `pytest -m gpu --regenerate-fixtures`.

Cache path: tests/fixtures/go1_smoke_checkpoint.pt
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.gpu
def test_mjlab_train_produces_checkpoint(
    mjlab_go1_checkpoint: Path,
) -> None:
    """Session fixture produces the checkpoint on first use. Subsequent
    runs just load it."""
    assert mjlab_go1_checkpoint.is_file(), (
        f"mjlab_go1_checkpoint fixture did not produce {mjlab_go1_checkpoint}"
    )
    assert mjlab_go1_checkpoint.stat().st_size > 1024, (
        f"checkpoint at {mjlab_go1_checkpoint} is suspiciously small "
        f"({mjlab_go1_checkpoint.stat().st_size} bytes)"
    )


# ── S7 (T1 / T9): other-robot smoke gates ──────────────────────────
#
# These pin the humanoid (G1) and toy (Cartpole) training paths so a
# mjlab / sculptor bump that breaks their schema dispatch or
# subprocess CLI gets caught at GPU-smoke time instead of during an
# overnight run. Budget: ~2-3 min each at max_iterations=10,
# num_envs=512 — well below Go1's 5-min 100-iter fixture. No caching
# fixture; they run every time `pytest -m gpu` is invoked (rare by
# design). CPU-only hosts auto-skip via conftest.py.
#
# ANYmal (T10) intentionally omitted: mjlab's task registry
# (`mjlab.tasks.registry._REGISTRY`) ships only Go1, G1, Cartpole,
# and Yam tasks as of this mjlab version. The robot_library.yml lists
# `anybotics_anymal_c` as `mjlab_ready` but `preconfigured_tasks: []`
# — there's no MjlabAdapter-compatible task_id to drive. Revisit once
# mjlab lands an ANYmal task.


@pytest.mark.gpu
def test_mjlab_g1_short_train_smoke(tmp_path: Path) -> None:
    """G1 humanoid (29-dof) — schema dispatch + train loop. Catches
    the class of bugs where `_G1_STATE_SCHEMA` and the snapshot field
    reads drift apart (G1 has 29 joints vs Go1's 18, and would
    AttributeError silently if `_find_articulated_entity` returned
    the wrong entity)."""
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        num_envs=512,
        device="cuda:0",
        max_iterations=10,
    )
    # State schema assertion is CPU-cheap — fail fast if dispatch breaks.
    c = adapter.reward_contract()
    assert c.state_schema is not None
    assert c.state_schema["qpos"] == (29,), c.state_schema
    assert c.state_schema["actuator_force"] == (23,), c.state_schema

    output_dir = tmp_path / "g1_smoke"
    output_dir.mkdir()
    result = adapter.train(
        reward_module_path=None,
        output_dir=output_dir,
        steps=10,
        seed=1,
    )
    assert result.checkpoint_path.is_file(), (
        f"G1 training did not produce checkpoint at {result.checkpoint_path}"
    )
    assert result.checkpoint_path.stat().st_size > 1024


@pytest.mark.gpu
def test_tracking_reward_uses_reset_relative_height_on_real_g1(
    tmp_path: Path,
) -> None:
    """The generated tracking term runs on a real batched G1 env and its
    normalized reference root-Z matches the robot's non-zero world height."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from sculptor.adapters._mjlab_runner import _build_sculptor_term_class
    from sculptor.refs.track import generate_tracking_residual_reward_source

    env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        env.reset()
        robot = env.scene["robot"]
        joint_names = list(robot.joint_names)
        initial_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        clip = {
            # Deliberately normalized near zero: the real G1 root is around
            # 0.7 m, which reproduced the live dead-channel failure.
            "root_pos_z": np.full(20, 0.01, dtype=np.float64),
            "fps": 50.0,
            "joint_pos": np.tile(initial_joint_pos, (20, 1)),
            "joint_names": joint_names,
        }
        reward_path = tmp_path / "tracking_reward.py"
        reward_path.write_text(generate_tracking_residual_reward_source(
            clip=clip, clip_id="gpu-reset-relative", robot="g1"),
            encoding="utf-8")

        Term = _build_sculptor_term_class((
            "qpos", "qvel", "base_lin_vel_b", "base_ang_vel_b",
            "projected_gravity_b", "actuator_force", "command_vel",
        ))
        term = Term(SimpleNamespace(params={
            "reward_module_path": str(reward_path),
        }), env)
        reward = term(env)

        assert reward.shape == (2,)
        assert torch.isfinite(reward).all()
        # Position, zero velocity, and reset-relative root height all match;
        # the old absolute-height comparison scored only ~0.6 here.
        assert float(reward.min()) > 0.95
        assert term._base_height_anchor is not None
        assert torch.isfinite(term._base_height_anchor).all()
    finally:
        env.close()


@pytest.mark.gpu
def test_mjlab_cartpole_short_train_smoke(tmp_path: Path) -> None:
    """Cartpole (T1) — the fast-path toy task that sanity-checks the
    fixed-base schema (`_CARTPOLE_STATE_SCHEMA`: qpos/qvel/actuator_force)
    end-to-end on GPU. Budget ~30 s at max_iterations=10, num_envs=256.
    Pairs with the CPU-only unit tests in test_mjlab_adapter.py +
    test_edit.py shipped under S4 of this pass."""
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Cartpole-Balance",
        num_envs=256,
        device="cuda:0",
        max_iterations=10,
    )
    c = adapter.reward_contract()
    assert c.state_schema is not None
    assert set(c.state_schema.keys()) == {"qpos", "qvel", "actuator_force"}

    output_dir = tmp_path / "cartpole_smoke"
    output_dir.mkdir()
    result = adapter.train(
        reward_module_path=None,
        output_dir=output_dir,
        steps=10,
        seed=1,
    )
    assert result.checkpoint_path.is_file()
    assert result.checkpoint_path.stat().st_size > 1024


@pytest.mark.gpu
def test_mjlab_vram_probe_returns_coefficient(tmp_path: Path) -> None:
    """Pre-M3 gate (B): VRAM probe sanity.

    Instantiate MjlabAdapter for Go1, call the probe explicitly, assert:
      - per-env coefficient is in sane range 0.1–10 MB/env for Go1 at 64 envs
      - cache file is written with the correct (task_id, mjlab_version) key
      - second call hits the cache (same result, doesn't re-run subprocess)
    """
    from importlib.metadata import version
    import json as _json

    from sculptor.adapters.mjlab import MjlabAdapter, measure_vram_coefficient

    # Instantiate once to confirm task_id is valid.
    _ = MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1", device="cuda:0")

    cache = tmp_path / ".sculptor_cache" / "vram_coefficients.json"
    result = measure_vram_coefficient(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        num_envs=64,
        cache_file=cache,
    )
    assert result["ok"] is True, result
    assert result["probe_num_envs"] == 64
    assert result["per_env_bytes"] > 0

    # Range gate: 0.1–10 MB per env. Go1 at 64 envs is typically
    # 0.5–3 MB/env; anything outside this band is suspicious.
    MB = 1024 * 1024
    assert 0.1 * MB <= result["per_env_bytes"] <= 10 * MB, (
        f"per_env_bytes {result['per_env_bytes'] / MB:.3f} MB "
        f"outside 0.1-10 MB sanity range"
    )
    # 20% safety buffer kicks in.
    assert result["coefficient_bytes_per_env"] >= result["per_env_bytes"] * 1.15

    # Cache file written.
    assert cache.is_file(), "cache file not written"
    payload = _json.loads(cache.read_text())
    expected_key = f"Mjlab-Velocity-Flat-Unitree-Go1|{version('mjlab')}"
    assert payload.get("_cache_key") == expected_key

    # Second call — should hit cache (same result, fast path).
    cached_result = measure_vram_coefficient(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        num_envs=64,
        cache_file=cache,
    )
    assert cached_result["_cache_key"] == expected_key
    assert cached_result["per_env_bytes"] == result["per_env_bytes"]
