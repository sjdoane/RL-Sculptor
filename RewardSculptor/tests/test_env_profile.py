"""§RL_SCULPTOR_AUDIT §4.4 gap #5: goal-class env alignment.

`_apply_env_profile` retargets the mjlab walking-task mechanics that
fight a standing jump. Offline tests only — the cfg mutation is pure
attribute manipulation on the loaded task cfg, so a SimpleNamespace
fake covers it (same convention as test_ground_texture.py).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sculptor.adapters import _mjlab_runner


def _fake_velocity_cfg() -> SimpleNamespace:
    """Shape-faithful mock of mjlab's velocity task cfg (the fields
    _apply_env_profile touches, with the real default values)."""
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi),
    )
    twist = SimpleNamespace(
        ranges=ranges, rel_standing_envs=0.1, rel_heading_envs=0.3,
        rel_forward_envs=0.2, heading_command=True,
    )
    return SimpleNamespace(
        commands={"twist": twist},
        curriculum={"command_vel": object(), "terrain_levels": object()},
        events={
            "push_robot": object(),
            "reset_base": SimpleNamespace(params={
                "pose_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                    "z": (0.01, 0.05), "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            }),
        },
        terminations={
            "fell_over": SimpleNamespace(
                params={"limit_angle": math.radians(70.0)}),
            "time_out": SimpleNamespace(params={}),
        },
        episode_length_s=20.0,
    )


def test_jump_profile_zeroes_velocity_commands() -> None:
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg, "jump")
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (0.0, 0.0)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    # None, NOT (0,0): UniformVelocityCommand.__init__ rejects any truthy
    # heading range when heading_command=False (live failure 2026-07-01).
    assert twist.ranges.heading is None
    assert twist.rel_standing_envs == 1.0
    assert twist.rel_heading_envs == 0.0
    assert twist.rel_forward_envs == 0.0
    assert twist.heading_command is False


def test_jump_profile_removes_command_curriculum_and_pushes() -> None:
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg, "jump")
    # command_vel curriculum would RE-WIDEN the zeroed ranges at 5k/10k
    # steps; push_robot destroys launch/landing attempts.
    assert "command_vel" not in cfg.curriculum
    assert "push_robot" not in cfg.events
    # Unrelated entries untouched.
    assert "terrain_levels" in cfg.curriculum
    assert "reset_base" in cfg.events


def test_jump_profile_relaxes_fell_over_and_shortens_episode() -> None:
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg, "jump")
    # 70° terminated the episode before projected gravity could flip
    # sign (90°), so the reward-contract `fallen` signal never fired
    # during training AND termination was an escape from penalties.
    assert cfg.terminations["fell_over"].params["limit_angle"] == (
        pytest.approx(math.radians(120.0)))
    assert cfg.episode_length_s == 10.0


def test_empty_and_default_profiles_are_noops() -> None:
    for profile in ("", "default", None):
        cfg = _fake_velocity_cfg()
        _mjlab_runner._apply_env_profile(cfg, profile)
        assert cfg.commands["twist"].ranges.lin_vel_x == (-1.0, 1.0)
        assert "push_robot" in cfg.events
        assert cfg.episode_length_s == 20.0


def test_unknown_profile_warns_but_never_raises(capsys) -> None:
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg, "backflip")
    assert cfg.episode_length_s == 20.0            # untouched
    assert "unknown" in capsys.readouterr().err


def test_jump_profile_tolerates_partial_cfg() -> None:
    # A cfg missing every touched surface (fixed-base task, API drift)
    # must not raise — the defensive per-mutation contract.
    _mjlab_runner._apply_env_profile(SimpleNamespace(), "jump")
    _mjjab_partial = SimpleNamespace(commands={}, terminations={})
    _mjlab_runner._apply_env_profile(_mjjab_partial, "jump")


def test_adapter_rejects_unknown_env_profile() -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    with pytest.raises(ValueError, match="env_profile"):
        MjlabAdapter(
            task_id="Mjlab-Velocity-Flat-Unitree-Go1",
            env_profile="backflip",
        )


def test_adapter_passes_env_profile_to_train_subprocess(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from unittest.mock import patch

    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        num_envs=64,
        env_profile="jump",
    )
    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa: ANN001
        captured["cmd"] = cmd
        return _FakeCompleted()

    (tmp_path / "checkpoint.pt").write_bytes(b"stub")
    (tmp_path / "metrics.json").write_text("{}")
    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        adapter.train(
            reward_module_path=None, output_dir=tmp_path, steps=1, seed=1)
    cmd = captured["cmd"]
    assert "--env-profile" in cmd
    assert cmd[cmd.index("--env-profile") + 1] == "jump"


def test_adapter_passes_env_profile_to_rollout_subprocess(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from unittest.mock import patch

    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1",
        num_envs=64,
        env_profile="jump",
    )
    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa: ANN001
        captured["cmd"] = cmd
        return _FakeCompleted()

    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"stub")
    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        adapter.rollout(
            checkpoint_path=ckpt, output_dir=tmp_path, n_episodes=1)
    cmd = captured["cmd"]
    assert "--env-profile" in cmd
    assert cmd[cmd.index("--env-profile") + 1] == "jump"


def test_default_adapter_omits_env_profile_flag(tmp_path) -> None:
    """Byte-identical CLI for existing projects (no profile set)."""
    pytest.importorskip("mjlab")
    from unittest.mock import patch

    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1", num_envs=64)
    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa: ANN001
        captured["cmd"] = cmd
        return _FakeCompleted()

    (tmp_path / "checkpoint.pt").write_bytes(b"stub")
    (tmp_path / "metrics.json").write_text("{}")
    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        adapter.train(
            reward_module_path=None, output_dir=tmp_path, steps=1, seed=1)
    assert "--env-profile" not in captured["cmd"]


# ── §2026-07-04: RSI reset curriculum + explosive-motion PPO profile ───────
def test_jump_profile_adds_rsi_resets_in_train_only() -> None:
    """Reference-state initialization (gap #7) applies to TRAINING resets
    only: episodes start up to +0.40 m with vertical velocity in
    [-0.5, +2.0] m/s so the policy experiences flight/landing it cannot
    yet produce. Rollout keeps the standing start (the true task) — the
    metric's upright_start / return-to-start-height view must not see
    mid-air spawns."""
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg, "jump", train=True)
    params = cfg.events["reset_base"].params
    assert params["pose_range"]["z"] == (0.0, 0.40)
    assert params["velocity_range"]["z"] == (-0.5, 2.0)

    cfg2 = _fake_velocity_cfg()
    _mjlab_runner._apply_env_profile(cfg2, "jump", train=False)
    params2 = cfg2.events["reset_base"].params
    assert params2["pose_range"]["z"] == (0.01, 0.05)      # untouched
    assert params2["velocity_range"] == {}                 # untouched
    # Everything non-RSI still applies on the rollout side.
    assert "push_robot" not in cfg2.events
    assert cfg2.episode_length_s == 10.0


def test_rl_profile_doubles_entropy_for_jump_only() -> None:
    algo = SimpleNamespace(entropy_coef=0.01)
    rl_cfg = SimpleNamespace(algorithm=algo)
    _mjlab_runner._apply_rl_profile(rl_cfg, "jump")
    assert algo.entropy_coef == pytest.approx(0.02)
    # Non-jump profiles + missing fields are no-ops, never raises.
    algo2 = SimpleNamespace(entropy_coef=0.01)
    _mjlab_runner._apply_rl_profile(SimpleNamespace(algorithm=algo2), "")
    assert algo2.entropy_coef == 0.01
    _mjlab_runner._apply_rl_profile(SimpleNamespace(), "jump")
    _mjlab_runner._apply_rl_profile(
        SimpleNamespace(algorithm=SimpleNamespace()), "jump")
