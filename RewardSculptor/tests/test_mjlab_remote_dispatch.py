"""MjlabAdapter ↔ RemoteExecutor seam tests (§Ship 23).

Mirrors the `_run_with_cleanup` patch pattern from test_mjlab_adapter.py:
RemoteExecutor.execute is patched to capture the RunnerJob and drop the
artifacts train()/rollout() post-checks expect. Verifies:
  * enabled → executor dispatched, local subprocess NOT spawned;
  * disabled → byte-identical local behavior, executor never touched;
  * job shape (options, mirrored inputs, ordered artifacts, remote env);
  * rollout stays local unless `rollout_remote=true`;
  * warm-start checkpoint rides in input_paths;
  * the local-VRAM num_envs autocap is skipped when remote is enabled;
  * SCULPTOR_REMOTE_* env vars enable dispatch with no TOML at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mjlab")

from sculptor.adapters import _remote as _remote_mod  # noqa: E402
from sculptor.adapters import mjlab as mjlab_mod  # noqa: E402
from sculptor.adapters.base import RolloutResult, TrainResult  # noqa: E402

TASK = "Mjlab-Velocity-Flat-Unitree-Go1"
REMOTE_TABLE = {"enabled": True, "host": "1.2.3.4", "user": "root"}

TRAIN_ARTIFACTS = {"checkpoint.pt": b"CKPT", "metrics.json": b'{"loss": 1.0}'}
ROLLOUT_ARTIFACTS = {
    "behavior.json": b'{"mean_return": 1.0}',
    "trajectory.npz": b"NPZ",
    "rollout.mp4": b"MP4",
}


def _fake_execute(captured: dict, artifacts: dict[str, bytes]):
    def execute(self, job):
        captured["job"] = job
        captured["cfg"] = self.cfg
        for name, data in artifacts.items():
            (Path(job.output_dir) / name).write_bytes(data)
        return subprocess.CompletedProcess(["<remote>"], 0, "", "")

    return execute


def _reward_module(tmp_path: Path) -> Path:
    p = tmp_path / "v0.py"
    p.write_text(
        "REWARD_SPEC = {'version': 'v0'}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return (0.0, {})\n"
    )
    return p


# ── train: remote enabled ────────────────────────────────────────────


def test_train_dispatches_remotely_when_enabled(tmp_path: Path) -> None:
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=512, device="cuda:0", max_iterations=5,
        remote={**REMOTE_TABLE, "device": "cuda:1"},
    )
    captured: dict = {}

    def forbidden_local(cmd, env=None, timeout=None):  # noqa: ARG001
        raise AssertionError("local _run_with_cleanup must not run when remote enabled")

    with (
        patch.object(_remote_mod.RemoteExecutor, "execute",
                     _fake_execute(captured, TRAIN_ARTIFACTS)),
        patch.object(mjlab_mod, "_run_with_cleanup", side_effect=forbidden_local),
    ):
        result = adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=tmp_path / "out",
            steps=5,
            seed=7,
        )

    assert isinstance(result, TrainResult)
    assert result.checkpoint_path == (tmp_path / "out" / "checkpoint.pt").resolve()
    job = captured["job"]
    assert job.subcommand == "train"
    assert job.options["--task-id"] == TASK
    assert job.options["--num-envs"] == "512"
    assert job.options["--max-iterations"] == "5"
    assert job.options["--seed"] == "7"
    # Remote device override wins over the local adapter device …
    assert job.options["--device"] == "cuda:1"
    # … and pins CUDA_VISIBLE_DEVICES for the REMOTE card.
    assert job.remote_env["CUDA_VISIBLE_DEVICES"] == "1"
    # Schema keys still derived per-task (S8 regression surface).
    assert set(job.options["--schema-keys"].split(",")) >= {"qpos", "qvel", "command_vel"}
    # Reward module rides as an uploadable input, not a bare option.
    assert job.input_paths["--reward-module-path"] == _reward_module(tmp_path)
    # Ordered artifacts: completion key (resume key) LAST.
    assert job.required_artifacts == ("metrics.json", "checkpoint.pt")
    assert job.output_dir == (tmp_path / "out").resolve()


def test_train_remote_device_defaults_to_local_device(tmp_path: Path) -> None:
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1, remote=dict(REMOTE_TABLE),
    )
    captured: dict = {}
    with patch.object(_remote_mod.RemoteExecutor, "execute",
                      _fake_execute(captured, TRAIN_ARTIFACTS)):
        adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=tmp_path / "out", steps=1, seed=1,
        )
    assert captured["job"].options["--device"] == "cuda:0"
    assert captured["job"].remote_env["CUDA_VISIBLE_DEVICES"] == "0"


def test_train_warm_start_checkpoint_in_input_paths(tmp_path: Path) -> None:
    """§Ship 15 warm-start must be uploaded (mirror path), not passed
    as a local-only flag the remote host can't see."""
    prior = tmp_path / "stage1" / "checkpoint.pt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"PRIOR")
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1, remote=dict(REMOTE_TABLE),
    )
    captured: dict = {}
    with patch.object(_remote_mod.RemoteExecutor, "execute",
                      _fake_execute(captured, TRAIN_ARTIFACTS)):
        adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=tmp_path / "out", steps=1, seed=1,
            init_policy_path=prior,
        )
    job = captured["job"]
    assert job.input_paths["--load-pretrained-policy"] == prior.resolve()
    assert "--load-pretrained-policy" not in job.options


def test_train_remote_nonzero_exit_raises_like_local(tmp_path: Path) -> None:
    """A remote runner failure surfaces through the SAME RuntimeError
    path as a local one (recoverable iteration failure upstream)."""
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1, remote=dict(REMOTE_TABLE),
    )

    def failing_execute(self, job):
        return subprocess.CompletedProcess(
            ["<remote>"], 9, "remote stdout", "remote: synthetic boom",
        )

    with patch.object(_remote_mod.RemoteExecutor, "execute", failing_execute):
        with pytest.raises(RuntimeError) as ei:
            adapter.train(
                reward_module_path=None,
                output_dir=tmp_path / "out", steps=1, seed=1,
            )
    assert "exited 9" in str(ei.value)
    assert "synthetic boom" in str(ei.value)


# ── train: disabled → identical local behavior ───────────────────────


def test_train_local_when_remote_absent(tmp_path: Path) -> None:
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, device="cuda:0", max_iterations=1,
    )
    captured: dict = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_local(cmd, env=None, timeout=None):  # noqa: ARG001
        captured["cmd"] = cmd
        return _Done()

    out = tmp_path / "out"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"stub")
    (out / "metrics.json").write_text("{}")

    def forbidden_remote(self, job):  # noqa: ARG001
        raise AssertionError("RemoteExecutor must not run when remote is unset")

    with (
        patch.object(_remote_mod.RemoteExecutor, "execute", forbidden_remote),
        patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_local),
    ):
        adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=out, steps=1, seed=1,
        )
    assert "--task-id" in captured["cmd"]


def test_train_local_when_remote_disabled_table(tmp_path: Path) -> None:
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1,
        remote={"enabled": False, "host": "1.2.3.4"},
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"stub")
    (out / "metrics.json").write_text("{}")

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch.object(mjlab_mod, "_run_with_cleanup", return_value=_Done()) as local:
        adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=out, steps=1, seed=1,
        )
    assert local.called


# ── env-var injection path (the UI backend's mechanism) ──────────────


def test_env_vars_enable_remote_without_toml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SCULPTOR_REMOTE_ENABLED", "1")
    monkeypatch.setenv("SCULPTOR_REMOTE_HOST", "9.9.9.9")
    monkeypatch.setenv("SCULPTOR_REMOTE_USER", "root")
    adapter = mjlab_mod.MjlabAdapter(task_id=TASK, num_envs=64, max_iterations=1)
    captured: dict = {}
    with patch.object(_remote_mod.RemoteExecutor, "execute",
                      _fake_execute(captured, TRAIN_ARTIFACTS)):
        adapter.train(
            reward_module_path=_reward_module(tmp_path),
            output_dir=tmp_path / "out", steps=1, seed=1,
        )
    assert captured["cfg"].host == "9.9.9.9"


# ── rollout gating ───────────────────────────────────────────────────


def _checkpoint(tmp_path: Path) -> Path:
    p = tmp_path / "checkpoint.pt"
    p.write_bytes(b"CKPT")
    return p


def test_rollout_stays_local_by_default(tmp_path: Path) -> None:
    """Remote ENABLED but rollout_remote unset → rollout runs locally
    (short job; local keeps video preview robust)."""
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1, remote=dict(REMOTE_TABLE),
    )

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def forbidden_remote(self, job):  # noqa: ARG001
        raise AssertionError("rollout must stay local unless rollout_remote=true")

    with (
        patch.object(_remote_mod.RemoteExecutor, "execute", forbidden_remote),
        patch.object(mjlab_mod, "_run_with_cleanup", return_value=_Done()) as local,
    ):
        result = adapter.rollout(
            checkpoint_path=_checkpoint(tmp_path),
            output_dir=tmp_path / "ro", n_episodes=2,
        )
    assert local.called
    assert isinstance(result, RolloutResult)


def test_rollout_dispatches_remotely_when_opted_in(tmp_path: Path) -> None:
    adapter = mjlab_mod.MjlabAdapter(
        task_id=TASK, num_envs=64, max_iterations=1,
        remote={**REMOTE_TABLE, "rollout_remote": True},
    )
    captured: dict = {}

    def forbidden_local(cmd, env=None, timeout=None):  # noqa: ARG001
        raise AssertionError("local rollout must not run when rollout_remote=true")

    with (
        patch.object(_remote_mod.RemoteExecutor, "execute",
                     _fake_execute(captured, ROLLOUT_ARTIFACTS)),
        patch.object(mjlab_mod, "_run_with_cleanup", side_effect=forbidden_local),
    ):
        result = adapter.rollout(
            checkpoint_path=_checkpoint(tmp_path),
            output_dir=tmp_path / "ro", n_episodes=3,
            max_episode_steps=250, playback_speed=1.0,
        )
    job = captured["job"]
    assert job.subcommand == "rollout"
    assert job.options["--n-episodes"] == "3"
    assert job.options["--max-episode-steps"] == "250"
    assert job.input_paths["--checkpoint-path"] == _checkpoint(tmp_path).resolve()
    # Ordered: rollout.mp4 LAST — sculpt.py's skip check needs all three.
    assert job.required_artifacts == ("behavior.json", "trajectory.npz", "rollout.mp4")
    assert result.video_path == (tmp_path / "ro" / "rollout.mp4").resolve()


# ── num_envs autocap skip ────────────────────────────────────────────


def test_autocap_applies_locally_on_small_vram() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(4 * 2**30, 8 * 2**30)),
    ):
        with pytest.warns(RuntimeWarning, match="auto-capping num_envs"):
            adapter = mjlab_mod.MjlabAdapter(task_id=TASK, num_envs=4096)
    assert adapter.num_envs == 2048


def test_autocap_skipped_when_remote_enabled() -> None:
    """The local VRAM probe measures the wrong GPU when training runs
    on the pod — 4096 envs must survive on an 8 GiB laptop."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(4 * 2**30, 8 * 2**30)),
    ):
        adapter = mjlab_mod.MjlabAdapter(
            task_id=TASK, num_envs=4096, remote=dict(REMOTE_TABLE),
        )
    assert adapter.num_envs == 4096


def test_autocap_still_applies_when_remote_disabled() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(4 * 2**30, 8 * 2**30)),
    ):
        with pytest.warns(RuntimeWarning):
            adapter = mjlab_mod.MjlabAdapter(
                task_id=TASK, num_envs=4096,
                remote={"enabled": False, "host": "1.2.3.4"},
            )
    assert adapter.num_envs == 2048
