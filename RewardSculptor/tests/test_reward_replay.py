"""§RL_SCULPTOR_AUDIT §4.4 (loop 4b): edit anti-collapse screen.

Covers `MjlabAdapter.build_reward_replay` (trajectory.npz → batched
reward inputs) and edit.py's `_replay_reward_summary` /
`_screen_reward_on_replay` (net-negative-living reject). Offline only —
no GPU, no LLM.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sculptor.edit import (  # noqa: E402
    EditValidationError,
    _REPLAY_MIN_FRAMES,
    _replay_reward_summary,
    _screen_reward_on_replay,
)


# ── Toy reward modules ─────────────────────────────────────────────────────
def _module_with(fn) -> types.ModuleType:
    mod = types.ModuleType("_toy_reward")
    mod.compute_reward_batched = fn
    return mod


def _penalty_heavy(state, action, next_state, info):
    # v6-class: constant -2 living penalty dwarfing a 0.1 alive bonus.
    n = action.shape[0]
    alive = torch.full((n,), 0.1)
    seated = torch.full((n,), -2.0)
    return alive + seated, {"alive_bonus": alive, "seated_penalty": seated}


def _healthy(state, action, next_state, info):
    n = action.shape[0]
    alive = torch.full((n,), 0.1)
    shaped = info["base_height"].clamp(min=0.0)
    return alive + shaped, {"alive_bonus": alive, "height": shaped}


def _replay_batch(n: int = 128, fallen_frac: float = 0.0):
    state = {"qpos": torch.zeros((n, 4))}
    next_state = {"qpos": torch.zeros((n, 4))}
    action = torch.zeros((n, 2))
    fallen = torch.zeros(n)
    k = int(n * fallen_frac)
    if k:
        fallen[:k] = 1.0
    info = {
        "base_height": torch.full((n,), 0.7),
        "fallen": fallen,
    }
    return state, action, next_state, info


# ── _replay_reward_summary ────────────────────────────────────────────────
def test_summary_reports_mean_and_components() -> None:
    s = _replay_reward_summary(_module_with(_penalty_heavy), _replay_batch())
    assert s is not None
    assert s["mean_alive"] == pytest.approx(-1.9)
    assert s["component_means"]["seated_penalty"] == pytest.approx(-2.0)
    assert s["n_alive"] == 128


def test_summary_excludes_fallen_frames() -> None:
    def half_and_half(state, action, next_state, info):
        # -5 on fallen frames, +1 on living frames.
        r = torch.where(info["fallen"] > 0.5, -5.0, 1.0)
        return r, {"r": r}

    s = _replay_reward_summary(
        _module_with(half_and_half), _replay_batch(n=200, fallen_frac=0.5))
    assert s is not None
    assert s["mean_alive"] == pytest.approx(1.0)   # fallen frames masked out
    assert s["n_alive"] == 100


def test_summary_none_without_batched_path_or_inputs() -> None:
    assert _replay_reward_summary(types.ModuleType("empty"), _replay_batch()) is None
    assert _replay_reward_summary(_module_with(_healthy), None) is None


def test_summary_none_when_too_few_alive_frames() -> None:
    n = _REPLAY_MIN_FRAMES - 1
    s = _replay_reward_summary(_module_with(_healthy), _replay_batch(n=n))
    assert s is None


# ── _screen_reward_on_replay ──────────────────────────────────────────────
def test_screen_rejects_net_negative_living_reward() -> None:
    with pytest.raises(EditValidationError, match="reward-collapse screen"):
        _screen_reward_on_replay(_module_with(_penalty_heavy), _replay_batch())


def test_screen_reject_message_names_offending_component() -> None:
    with pytest.raises(EditValidationError, match="seated_penalty=-2.000"):
        _screen_reward_on_replay(_module_with(_penalty_heavy), _replay_batch())


def test_screen_includes_parent_baseline_when_supplied() -> None:
    parent = _replay_reward_summary(_module_with(_healthy), _replay_batch())
    with pytest.raises(EditValidationError, match="PARENT reward averages"):
        _screen_reward_on_replay(
            _module_with(_penalty_heavy), _replay_batch(), parent)


def test_screen_passes_healthy_reward_and_no_inputs() -> None:
    _screen_reward_on_replay(_module_with(_healthy), _replay_batch())
    _screen_reward_on_replay(_module_with(_penalty_heavy), None)  # screen off
    # Crash inside the module → no evidence → pass (variance/zero probes
    # own crash rejection).
    def boom(*a):  # noqa: ANN002
        raise RuntimeError("nope")
    _screen_reward_on_replay(_module_with(boom), _replay_batch())


# ── MjlabAdapter.build_reward_replay ──────────────────────────────────────
def _write_fake_trajectory(d: Path, *, T: int = 6, N: int = 3) -> None:
    rng = {}  # deterministic values, no RNG
    jp = np.arange(T * N * 29, dtype=np.float32).reshape(T, N, 29) * 1e-3
    root = np.zeros((T, N, 3), dtype=np.float32)
    root[..., 2] = 0.74
    root[-1, :, 2] = 0.80                      # last frame lifted
    pg = np.zeros((T, N, 3), dtype=np.float32)
    pg[..., 2] = -1.0                          # upright
    pg[:, N - 1, 2] = 0.5                      # env N-1 fallen throughout
    payload = {
        "rewards": np.zeros(T, dtype=np.float32),
        "episode_id": np.zeros(T * N, dtype=np.int32),
        "joint_pos": jp,
        "joint_vel": jp * 2.0,
        "action": jp[..., :29],
        "actuator_force": jp,
        "projected_gravity_b": pg,
        "root_link_pos_w": root,
        "left_foot_contact": np.ones((T, N), dtype=np.float32),
        "right_foot_contact": np.ones((T, N), dtype=np.float32),
        "left_foot_pos_b": np.full((T, N, 3), -0.7, dtype=np.float32),
        "right_foot_pos_b": np.full((T, N, 3), -0.7, dtype=np.float32),
    }
    np.savez_compressed(d / "trajectory.npz", **payload)
    (d / "behavior.json").write_text('{"step_dt": 0.02}')
    del rng


def test_build_reward_replay_shapes_and_info(tmp_path: Path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=64)
    _write_fake_trajectory(tmp_path)
    out = adapter.build_reward_replay(tmp_path)
    assert out is not None
    state, action, next_state, info = out
    F = action.shape[0]
    assert F == (6 - 1) * 3                       # all transitions kept
    assert state["qpos"].shape == (F, 29)
    assert next_state["qpos"].shape == (F, 29)
    # Info faithful to the runner's __call__ construction.
    assert info["base_height"].shape == (F,)
    assert float(info["base_height"].max()) == pytest.approx(0.80, abs=1e-6)
    # env N-1 has projected gravity z >= 0 → fallen.
    assert float(info["fallen"].max()) == 1.0
    assert float(info["fallen"].mean()) == pytest.approx(1 / 3, abs=1e-6)
    for k in ("left_foot_contact", "right_foot_height",
              "left_foot_swing_speed", "base_horizontal_speed"):
        assert info[k].shape == (F,)
    # Foot world height approx: root_z + pelvis-frame foot z ≈ 0.04.
    assert float(info["left_foot_height"][0]) == pytest.approx(0.04, abs=1e-3)


def test_build_reward_replay_subsamples_deterministically(tmp_path: Path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=64)
    _write_fake_trajectory(tmp_path, T=90, N=64)   # 5696 transitions > cap
    out1 = adapter.build_reward_replay(tmp_path)
    out2 = adapter.build_reward_replay(tmp_path)
    assert out1[1].shape[0] == adapter._REPLAY_MAX_FRAMES
    assert torch.equal(out1[3]["base_height"], out2[3]["base_height"])


def test_build_reward_replay_none_on_missing_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=64)
    assert adapter.build_reward_replay(tmp_path) is None          # no npz
    np.savez_compressed(tmp_path / "trajectory.npz",
                        rewards=np.zeros(4, dtype=np.float32))
    assert adapter.build_reward_replay(tmp_path) is None          # no core keys


def test_base_adapter_replay_defaults_to_none(tmp_path: Path) -> None:
    from sculptor.adapters.gym_sb3 import GymSB3Adapter

    adapter = GymSB3Adapter(env_id="Hopper-v4", n_envs=1)
    assert adapter.build_reward_replay(tmp_path) is None
