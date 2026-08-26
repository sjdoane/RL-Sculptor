"""tests/test_sculpt.py — offline tests for the orchestrator.

No live LLM. No training. Exercises:
  - `sculpt_init` scaffolds a project with the right files + git commit.
  - Removed metric-plateau auto-kill compatibility behavior.
  - `sculpt_run --dry-run` end-to-end by stubbing the adapter's train +
    rollout methods (no gymnasium / SB3 in the hot path) and going through
    the dry-run diagnose + edit shortcuts.
  - CHANGELOG.md, provenance.json, and git commits are produced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from sculptor.adapters.base import (
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)
from sculptor.sculpt import (
    _find_resume_start_iteration,
    _should_early_stop,
    _write_iteration_completion_marker,
    regenerate_reward_template,
    sculpt_init,
    sculpt_run,
)
from sculptor.run_manifests import (
    build_completion_manifest,
    build_rollout_input_manifest,
    build_train_input_manifest,
    file_identity,
    manifest_sha256,
    write_json_atomic,
)


def _write_reward_version(rewards_dir: Path, version: int) -> None:
    rewards_dir.mkdir(parents=True, exist_ok=True)
    (rewards_dir / f"v{version}.py").write_text(
        "def compute_reward(*args, **kwargs):\n    return 0.0, {}\n",
        encoding="utf-8",
    )


def _write_completion_marker(runs_dir: Path, version: int, **overrides) -> None:
    iter_dir = runs_dir / f"iter_{version}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = iter_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_required_phase_manifests(iter_dir)
    payload = _write_iteration_completion_marker(
        iter_dir,
        iter_index=version,
        checkpoint_path=checkpoint,
        reward_version_before=version,
        reward_version_after=version + 1,
        world_selection_hash=None,
    )
    payload.update(overrides)
    (iter_dir / "iteration_complete.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_required_phase_manifests(iter_dir: Path) -> Path:
    iteration = int(iter_dir.name.removeprefix("iter_"))
    checkpoint = iter_dir / "checkpoint.pt"
    reward = iter_dir / "reward_input.py"
    reward.write_text("reward = 1\n", encoding="utf-8")
    adapter = object()
    request = build_train_input_manifest(
        adapter=adapter,
        iteration=iteration,
        reward_module_path=reward,
        steps=100,
        seed=42,
        init_policy_path=None,
        init_policy_mode="actor_only",
    )
    effective = {
        **request,
        "request_manifest_sha256": manifest_sha256(request),
        "effective_initialization": {
            "mode": None,
            "policy": None,
            "forwarded_to_adapter": False,
        },
    }
    write_json_atomic(iter_dir / "train_request_manifest.json", request)
    write_json_atomic(iter_dir / "train_input_manifest.json", effective)
    write_json_atomic(
        iter_dir / "train_completion_manifest.json",
        build_completion_manifest(effective, [checkpoint]),
    )

    rollout = iter_dir / "rollout"
    rollout.mkdir(exist_ok=True)
    outputs = [
        rollout / "rollout.mp4",
        rollout / "trajectory.npz",
        rollout / "behavior.json",
    ]
    for path, content in zip(outputs, (b"video", b"trajectory", b"{}")):
        path.write_bytes(content)
    rollout_input = build_rollout_input_manifest(
        adapter=adapter,
        iteration=iteration,
        checkpoint_path=checkpoint,
        reward_module_path=reward,
        n_episodes=1,
        seed=42,
        max_episode_steps=100,
        playback_speed=1.0,
        render_every=1,
        fps=30.0,
        render_width=320,
        render_height=240,
        render_env_index=0,
    )
    write_json_atomic(
        rollout / "rollout_input_manifest.json", rollout_input,
    )
    write_json_atomic(
        rollout / "rollout_completion_manifest.json",
        build_completion_manifest(rollout_input, outputs),
    )
    assert file_identity(reward)["exists"] is True
    return reward


def test_resume_advances_past_completed_no_edit_iteration(tmp_path: Path):
    rewards = tmp_path / "rewards"
    runs = tmp_path / "runs"
    _write_reward_version(rewards, 3)
    _write_completion_marker(runs, 3)

    assert _find_resume_start_iteration(rewards, runs) == 4


def test_resume_advances_past_contiguous_completion_markers(tmp_path: Path):
    rewards = tmp_path / "rewards"
    runs = tmp_path / "runs"
    _write_reward_version(rewards, 3)
    _write_completion_marker(runs, 3)
    _write_completion_marker(runs, 4)

    assert _find_resume_start_iteration(rewards, runs) == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": 2},
        {"state": "running"},
        {"iter": 99},
    ],
)
def test_resume_reuses_iteration_with_invalid_completion_marker(
    tmp_path: Path,
    overrides: dict,
):
    rewards = tmp_path / "rewards"
    runs = tmp_path / "runs"
    _write_reward_version(rewards, 3)
    _write_completion_marker(runs, 3, **overrides)

    assert _find_resume_start_iteration(rewards, runs) == 3


def test_resume_does_not_jump_over_completion_marker_gap(tmp_path: Path):
    rewards = tmp_path / "rewards"
    runs = tmp_path / "runs"
    _write_reward_version(rewards, 3)
    _write_completion_marker(runs, 4)

    assert _find_resume_start_iteration(rewards, runs) == 3


def test_completion_marker_attests_exact_checkpoint_bytes(tmp_path: Path):
    iter_dir = tmp_path / "runs" / "iter_7"
    iter_dir.mkdir(parents=True)
    checkpoint = iter_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"exact actor and critic bytes")
    reward_input = _write_required_phase_manifests(iter_dir)

    _write_iteration_completion_marker(
        iter_dir,
        iter_index=7,
        checkpoint_path=checkpoint,
        reward_version_before=3,
        reward_version_after=4,
        world_selection_hash="a" * 64,
    )

    marker = json.loads(
        (iter_dir / "iteration_complete.json").read_text(encoding="utf-8")
    )
    assert marker["schema"] == 3
    assert marker["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert marker["checkpoint_bytes"] == checkpoint.stat().st_size

    rewards_dir = tmp_path / "rewards"
    _write_reward_version(rewards_dir, 7)
    assert _find_resume_start_iteration(rewards_dir, iter_dir.parent) == 8

    external = tmp_path / "checkpoint.pt"
    external.write_bytes(checkpoint.read_bytes())
    forged = dict(marker)
    forged["checkpoint"] = str(external)
    (iter_dir / "iteration_complete.json").write_text(
        json.dumps(forged), encoding="utf-8"
    )
    assert _find_resume_start_iteration(rewards_dir, iter_dir.parent) == 7

    (iter_dir / "iteration_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    checkpoint.write_bytes(b"x" * checkpoint.stat().st_size)
    assert _find_resume_start_iteration(rewards_dir, iter_dir.parent) == 7

    checkpoint.write_bytes(b"exact actor and critic bytes")
    _write_required_phase_manifests(iter_dir)
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=7,
        checkpoint_path=checkpoint,
        reward_version_before=3,
        reward_version_after=4,
        world_selection_hash="a" * 64,
    )
    reward_input.write_text("reward = 2\n", encoding="utf-8")
    assert _find_resume_start_iteration(rewards_dir, iter_dir.parent) == 7


# ── Metric-plateau auto-kill compatibility ───────────────────────────────
def test_early_stop_false_when_improving():
    assert _should_early_stop([1.0, 2.0, 3.0, 4.0]) is False


def test_early_stop_false_with_too_few_iterations():
    assert _should_early_stop([1.0, 2.0]) is False
    assert _should_early_stop([1.0, 2.0, 3.0]) is False


def test_early_stop_false_after_three_non_improving():
    # Metric-plateau auto-kill is disabled; reward dips are not halt signals.
    assert _should_early_stop([1.0, 5.0, 4.0, 4.5, 4.9]) is False


def test_early_stop_false_when_last_window_beats_prior():
    # prior max was 3, last three include a 4 → improvement.
    assert _should_early_stop([1.0, 2.0, 3.0, 4.0, 3.5, 3.8]) is False


# ── sculpt_init ──────────────────────────────────────────────────────────
def test_sculpt_init_scaffolds_expected_files(tmp_path: Path):
    target = tmp_path / "myproj"
    project = sculpt_init(target, adapter="gym_sb3")
    assert project == target.resolve()
    # Required files
    for rel in (
        "config.toml", "rewards/v0.py", "rewards/__init__.py", "rewards/current.py",
        "kg_seeds.yml", ".gitignore", "reports/.gitkeep", "runs/.gitkeep",
    ):
        assert (project / rel).is_file(), f"missing {rel}"
    # v0 has the minimum shape
    v0 = (project / "rewards/v0.py").read_text(encoding="utf-8")
    assert "REWARD_SPEC" in v0
    assert '"version": "v0"' in v0
    assert "def compute_reward" in v0
    # Config dotted class matches shortname lookup
    cfg = (project / "config.toml").read_text(encoding="utf-8")
    assert "sculptor.adapters.gym_sb3.GymSB3Adapter" in cfg
    # git repo with an initial commit
    r = subprocess.run(["git", "-C", str(project), "log", "--oneline"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "sculpt init" in r.stdout


def test_sculpt_init_refuses_non_empty_dir(tmp_path: Path):
    (tmp_path / "junk.txt").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sculpt_init(tmp_path, adapter="gym_sb3")


def test_sculpt_init_passes_dotted_adapter_through(tmp_path: Path):
    proj = sculpt_init(
        tmp_path / "proj2",
        adapter="some.custom.AdapterClass")
    cfg = (proj / "config.toml").read_text(encoding="utf-8")
    assert "some.custom.AdapterClass" in cfg


# ── adapter-aware v0 templates ───────────────────────────────────────────
def test_sculpt_init_gym_sb3_writes_scalar_template(tmp_path: Path):
    proj = sculpt_init(tmp_path / "gym", adapter="gym_sb3")
    v0 = (proj / "rewards" / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward(" in v0
    assert "def compute_reward_batched" not in v0
    assert "supports_batched" not in v0


def test_sculpt_init_mjlab_writes_batched_template(tmp_path: Path):
    proj = sculpt_init(tmp_path / "mjlab_proj", adapter="mjlab")
    v0 = (proj / "rewards" / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward(" in v0, "scalar path still required"
    assert "def compute_reward_batched(" in v0, "batched entry point missing"
    assert '"supports_batched": True' in v0
    # config.toml should pin the mjlab class (short-name resolved).
    cfg = (proj / "config.toml").read_text(encoding="utf-8")
    assert "sculptor.adapters.mjlab.MjlabAdapter" in cfg


def test_sculpt_init_mjlab_dotted_path_also_batched(tmp_path: Path):
    """Callers passing the fully-qualified class name should also get the
    batched template — not just the 'mjlab' short name."""
    proj = sculpt_init(
        tmp_path / "mjlab_dotted",
        adapter="sculptor.adapters.mjlab.MjlabAdapter",
    )
    v0 = (proj / "rewards" / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward_batched(" in v0


# ── regenerate_reward_template ───────────────────────────────────────────
def test_regenerate_reward_template_flips_scalar_to_batched(tmp_path: Path):
    """The migration path for already-broken projects: scaffolded under
    gym_sb3 (scalar), then [adapter].class swapped to mjlab post-hoc.
    regenerate_reward_template must detect that and rewrite v0.py."""
    proj = sculpt_init(tmp_path / "broken", adapter="gym_sb3")
    # Simulate backend's post-scaffold [adapter] rewrite to mjlab.
    cfg_path = proj / "config.toml"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace(
        "sculptor.adapters.gym_sb3.GymSB3Adapter",
        "sculptor.adapters.mjlab.MjlabAdapter",
    )
    cfg_path.write_text(cfg_text, encoding="utf-8")

    v0_before = (proj / "rewards" / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward_batched" not in v0_before  # precondition

    out = regenerate_reward_template(proj)
    assert out == proj / "rewards" / "v0.py"
    v0_after = (proj / "rewards" / "v0.py").read_text(encoding="utf-8")
    assert "def compute_reward_batched(" in v0_after
    assert '"supports_batched": True' in v0_after


def test_regenerate_reward_template_stays_scalar_for_gym(tmp_path: Path):
    proj = sculpt_init(tmp_path / "gym_ok", adapter="gym_sb3")
    out = regenerate_reward_template(proj)
    v0 = out.read_text(encoding="utf-8")
    assert "def compute_reward(" in v0
    assert "def compute_reward_batched" not in v0


def test_regenerate_reward_template_overwrites_user_edits(tmp_path: Path):
    """The contract is destructive by design — existing edits are lost
    (git history preserves the previous v0.py). The UI surfaces a warning
    before calling this; at the library level the behavior is simply
    'rewrite to the scaffold template for the current adapter'."""
    proj = sculpt_init(tmp_path / "mjlab_edited", adapter="mjlab")
    v0 = proj / "rewards" / "v0.py"
    v0.write_text(
        "# user's broken edit\ndef compute_reward(s,a,n,i): return 0,{}\n",
        encoding="utf-8",
    )
    regenerate_reward_template(proj)
    restored = v0.read_text(encoding="utf-8")
    assert "def compute_reward_batched(" in restored
    assert "user's broken edit" not in restored


def test_regenerate_reward_template_raises_on_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        regenerate_reward_template(tmp_path / "not_a_project")


def test_regenerate_reward_template_preserves_latest_current_when_iterated(
    tmp_path: Path,
):
    """Edge case: if the project has already iterated past v0, current.py
    must keep pointing at the latest — regenerating v0 is intended as a
    migration for unbroken projects but shouldn't regress ones that
    somehow progressed."""
    proj = sculpt_init(tmp_path / "iterated", adapter="gym_sb3")
    # Simulate an iteration: write v1.py + point current.py at it.
    rewards_dir = proj / "rewards"
    (rewards_dir / "v1.py").write_text(
        "REWARD_SPEC = {}\ndef compute_reward(s,a,n,i): return 1.0, {}\n",
        encoding="utf-8",
    )
    from sculptor.edit import _write_current_reexport
    _write_current_reexport(rewards_dir, rewards_dir / "v1.py")

    regenerate_reward_template(proj)
    current = (rewards_dir / "current.py").read_text(encoding="utf-8")
    assert "v1" in current, "current.py must still re-export v1, not v0"


def test_write_current_reexport_surfaces_compute_reward_batched(tmp_path: Path):
    """Regression: MjlabAdapter reads `compute_reward_batched` off the
    current.py module (not off v<n>.py directly). Before this fix the
    re-export shim only bound `compute_reward` and `REWARD_SPEC`, so the
    mjlab runner raised AttributeError inside `env.load_managers()` even
    though v0.py defined the batched path correctly. The generated
    current.py must conditionally re-bind `compute_reward_batched` when
    the latest reward module defines it."""
    import importlib.util

    from sculptor.edit import _write_current_reexport

    rewards_dir = tmp_path / "rewards"
    rewards_dir.mkdir()
    (rewards_dir / "__init__.py").write_text("", encoding="utf-8")
    (rewards_dir / "v0.py").write_text(
        "REWARD_SPEC = {'supports_batched': True}\n"
        "def compute_reward(s, a, n, i): return 1.0, {'x': 1.0}\n"
        "def compute_reward_batched(s, a, n, i):\n"
        "    import torch\n"
        "    r = torch.ones((a.shape[0],), device=a.device)\n"
        "    return r, {'x': r}\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards_dir, rewards_dir / "v0.py")
    current_src = (rewards_dir / "current.py").read_text(encoding="utf-8")
    assert "compute_reward_batched" in current_src, (
        "re-export must conditionally surface compute_reward_batched; "
        "scalar-only re-export hides the mjlab entry point"
    )

    # Load the generated current.py via the same file-path spec mjlab
    # uses, verify the batched attribute is importable as a module-level
    # binding (not just as _mod.compute_reward_batched).
    spec = importlib.util.spec_from_file_location(
        "reg_test_current", str(rewards_dir / "current.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compute_reward"), "scalar path still required"
    assert hasattr(mod, "compute_reward_batched"), (
        "batched path must be a module-level attribute on current.py "
        "so mjlab's SculptorRewardTerm can pick it up"
    )
    assert "compute_reward_batched" in mod.__all__


def test_write_current_reexport_is_transparent_to_reference_clock_surface(
    tmp_path: Path,
) -> None:
    """The worker receives current.py, so conditioning helpers must survive it."""
    import hashlib
    import importlib.util

    from sculptor.edit import _write_current_reexport

    rewards_dir = tmp_path / "rewards"
    rewards_dir.mkdir()
    selected = rewards_dir / "v12.py"
    selected.write_text(
        "REWARD_SPEC = {'reference_clock': {'schema': 1}}\n"
        "def compute_reward(s, a, n, i): return 0.0, {}\n"
        "def compute_reward_batched(s, a, n, i): return a[:, 0], {}\n"
        "def reference_clock_batched(step_count, device): return step_count\n"
        "def reference_target_index_batched(step_count, device): "
        "return step_count + 1\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards_dir, selected)

    spec = importlib.util.spec_from_file_location(
        "reg_test_reference_current", str(rewards_dir / "current.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name in (
        "compute_reward_batched",
        "reference_clock_batched",
        "reference_target_index_batched",
    ):
        assert getattr(mod, name) is getattr(mod._mod, name)
        assert name in mod.__all__
    assert mod.reference_clock_batched(4, "cpu") == 4
    assert mod.reference_target_index_batched(4, "cpu") == 5
    assert mod.SCULPTOR_REWARD_SELECTOR == {
        "schema": 1,
        "filename": "v12.py",
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }


def test_write_current_reexport_skips_batched_for_scalar_only_module(
    tmp_path: Path,
):
    """Scalar-only reward modules (gym_sb3 + other supports_batched=False
    adapters) must NOT grow a `compute_reward_batched` binding on the
    re-export — the conditional guard keeps both worlds working."""
    import importlib.util

    from sculptor.edit import _write_current_reexport

    rewards_dir = tmp_path / "rewards"
    rewards_dir.mkdir()
    (rewards_dir / "__init__.py").write_text("", encoding="utf-8")
    (rewards_dir / "v0.py").write_text(
        "REWARD_SPEC = {}\n"
        "def compute_reward(s, a, n, i): return 1.0, {}\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards_dir, rewards_dir / "v0.py")
    spec = importlib.util.spec_from_file_location(
        "reg_test_current_scalar", str(rewards_dir / "current.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compute_reward")
    assert not hasattr(mod, "compute_reward_batched")


# ── Stub adapter for dry-run end-to-end ──────────────────────────────────
class _StubAdapter(SculptorAdapter):
    """Minimal adapter that writes the artifacts diagnose+edit expect but
    does not touch gymnasium/SB3."""

    def __init__(self, project_root: Path, primary_metric_schedule):
        self._project_root = project_root
        self._schedule = list(primary_metric_schedule)
        self._train_calls = 0

    def reward_contract(self) -> RewardContract:
        return RewardContract(
            observation_space_spec=_FakeBox((4,)),
            action_space_spec=_FakeBox((2,)),
            expected_info_keys=["x_velocity"],
            expected_components=None,
        )

    def probe_component(self, reward_module_path):
        from sculptor.adapters.base import ComponentProbe
        return ComponentProbe(ok=True, components={"alive_bonus": 1.0}, total=1.0)

    def train(self, reward_module_path, output_dir, steps, seed):
        self._train_calls += 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "checkpoint.zip").write_bytes(b"stub-checkpoint")
        # primary_metric pulled from schedule (list of per-call values)
        mr = self._schedule.pop(0) if self._schedule else 0.0
        (output_dir / "metrics.json").write_text(json.dumps({
            "metrics": {
                "mean_return": mr, "std_return": 0.1, "n_eval_episodes": 5,
                "training_steps": steps, "n_envs": 1, "seed": seed,
            },
            "components": {"alive_bonus": 1.0},
        }), encoding="utf-8")
        (output_dir / "reward_spec.json").write_text(json.dumps({
            "version": "v_current", "author": "human",
            "hyperparameters": {"alive_bonus": 1.0},
            "references": [], "parent_hash": "",
            "description": "stub",
        }), encoding="utf-8")
        return TrainResult(
            checkpoint_path=output_dir / "checkpoint.zip",
            metrics_dict={"mean_return": mr},
            component_means={"alive_bonus": 1.0},
            logs_path=output_dir / "logs",
        )

    def rollout(
        self, checkpoint_path, output_dir, n_episodes, *, seed=None,
        reward_module_path=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "keyframes").mkdir(exist_ok=True)
        # one dummy PNG so diagnose has something to skip over
        (output_dir / "keyframes" / "frame_00.png").write_bytes(_TINY_PNG)
        (output_dir / "rollout.mp4").write_bytes(b"stub-video")
        import numpy as np

        np.savez_compressed(output_dir / "trajectory.npz",
                            rewards=np.zeros(10, dtype=np.float32),
                            episode_id=np.zeros(10, dtype=np.int32))
        behavior = {
            "n_episodes": n_episodes,
            "mean_return": 0.0,
            "max_episode_length": 100,
            "fall_rate": 0.5,
        }
        (output_dir / "behavior.json").write_text(json.dumps(behavior),
                                                  encoding="utf-8")
        return RolloutResult(
            video_path=output_dir / "rollout.mp4",
            keyframes_dir=output_dir / "keyframes",
            trajectory_path=output_dir / "trajectory.npz",
            n_episodes=n_episodes,
        )

    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict:
        return {"stub": True}


class _FakeBox:
    def __init__(self, shape):
        import numpy as np

        self.shape = shape
        self.dtype = np.float32

    def __repr__(self):
        return f"FakeBox(shape={self.shape})"


# 1x1 red PNG (same encoder as test_diagnose.py)
import struct
import zlib


def _png1() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xFF\x00\x00"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_TINY_PNG = _png1()


# ── Dry-run end-to-end ───────────────────────────────────────────────────
def _write_minimal_project(tmp_path: Path) -> Path:
    """Hand-built project dir — avoids gym_sb3.GymSB3Adapter pulling in MuJoCo."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "rewards").mkdir()
    (proj / "reports").mkdir()
    (proj / "runs").mkdir()
    (proj / "rewards" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "rewards" / "v0.py").write_text(textwrap.dedent('''
        """starter"""
        REWARD_SPEC = {
            "version": "v0",
            "parent_hash": "",
            "description": "starter",
            "author": "human",
            "hyperparameters": {"alive_bonus": 1.0},
            "references": [],
        }
        def compute_reward(state, action, next_state, info):
            a = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
            return a, {"alive_bonus": a}
    ''').strip() + "\n", encoding="utf-8")

    # Config points at a dummy adapter class we're going to monkeypatch in.
    (proj / "config.toml").write_text(textwrap.dedent(f'''
        [target]
        name = "dryrun"

        [adapter]
        class = "tests.test_sculpt._TestAdapterStub"
        config = {{ env_id = "Dummy-v0" }}

        [kg]
        environment_tag = "unit_test"

        [iteration]
        steps_per_iter = 1000
        primary_metric = "mean_return"
        behavior_metrics = ["max_episode_length", "fall_rate"]
        rollout_episodes = 2
    ''').strip() + "\n", encoding="utf-8")

    # git init
    subprocess.run(["git", "-C", str(proj), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(proj), "-c", "user.email=t@t",
                    "-c", "user.name=t", "add", "."], check=True)
    subprocess.run(["git", "-C", str(proj), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "init"], check=True)
    # Ensure we have a default identity for subsequent commits inside the loop.
    subprocess.run(["git", "-C", str(proj), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(proj), "config", "user.name", "t"], check=True)
    return proj


# Module-level adapter so load_adapter can import it.
_SCHEDULE: list[float] = []


class _TestAdapterStub(_StubAdapter):  # dotted path consumable by load_adapter
    def __init__(self, **_cfg):
        super().__init__(project_root=Path.cwd(), primary_metric_schedule=_SCHEDULE)


class _ResourceAdapterStub(_TestAdapterStub):
    """Generic adapter surface used to verify launch-scoped resources."""

    def __init__(
        self, num_envs: int = 1024, device: str = "cuda:0", **cfg,
    ):
        super().__init__(**cfg)
        self.num_envs = num_envs
        self.device = device


class _SeedEvidenceAdapter(_StubAdapter):
    """CPU stub with an explicit rollout seed and one injected failure."""

    def __init__(self, **_cfg):
        super().__init__(Path.cwd(), [1.0])

    def rollout(
        self, checkpoint_path, output_dir, n_episodes, *, seed=None,
        reward_module_path=None,
    ):
        if seed == 10_001:
            raise RuntimeError("injected seed failure")
        return super().rollout(checkpoint_path, output_dir, n_episodes)


class _PrimarySeedFailureAdapter(_SeedEvidenceAdapter):
    def rollout(
        self, checkpoint_path, output_dir, n_episodes, *, seed=None,
        reward_module_path=None,
    ):
        if seed == 10_000:
            raise RuntimeError("primary seed failed")
        return super().rollout(
            checkpoint_path,
            output_dir,
            n_episodes,
            seed=seed,
            reward_module_path=reward_module_path,
        )


def test_sculpt_run_dry_run_end_to_end(tmp_path: Path, monkeypatch):
    """3 iterations, dry-run, stub adapter — exercises the full orchestration
    path including CHANGELOG, provenance, git commits."""
    global _SCHEDULE
    _SCHEDULE = [1.0, 5.0, 3.0, 4.0, 4.5]  # improvement, then stall
    proj = _write_minimal_project(tmp_path)
    cfg = proj / "config.toml"

    result = sculpt_run(
        config_path=cfg, behavior_goal="dummy goal",
        iterations=3, resume=False, no_kg=True, dry_run=True,
    )
    # 3 iters ran (not early-stopped at 3).
    assert result.iterations_run == 3
    assert result.early_stopped is False
    # Each iter wrote an iter_dir and a new v<n>.py
    for i in range(3):
        assert (proj / "runs" / f"iter_{i}").is_dir()
        assert (proj / "runs" / f"iter_{i}" / "checkpoint.zip").is_file()
        assert (proj / "runs" / f"iter_{i}" / "rollout" / "behavior.json").is_file()
        assert (proj / "runs" / f"iter_{i}" / "diagnosis.json").is_file()
        completion = json.loads(
            (proj / "runs" / f"iter_{i}" / "iteration_complete.json").read_text(
                encoding="utf-8",
            )
        )
        assert completion["state"] == "completed"
        assert completion["iter"] == i
        assert completion["schema"] == 3
        checkpoint = proj / "runs" / f"iter_{i}" / "checkpoint.zip"
        assert completion["checkpoint_sha256"] == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        assert completion["checkpoint_bytes"] == checkpoint.stat().st_size
    for n in (1, 2, 3):
        assert (proj / "rewards" / f"v{n}.py").is_file()
    # current.py now re-exports v3
    current = (proj / "rewards" / "current.py").read_text(encoding="utf-8")
    assert "v3.py" in current

    # CHANGELOG, provenance, metric history
    changelog = (proj / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Iteration 0" in changelog and "Iteration 2" in changelog
    assert "alive_bonus" in changelog
    assert "(novel)" in changelog  # every dry-run edit is novel
    provenance = json.loads(
        (proj / "reports" / "provenance.json").read_text(encoding="utf-8"))
    # Dry-run edits are novel (paper_refs=[]), so provenance stays empty —
    # but the file must still be written.
    assert isinstance(provenance, dict)
    history = json.loads(
        (proj / "reports" / "metric_history.json").read_text(encoding="utf-8"))
    assert history["primary_metric"] == "mean_return"
    assert len(history["history"]) == 3

    # One git commit per iter was recorded (plus the scaffold commit).
    log = subprocess.run(
        ["git", "-C", str(proj), "log", "--oneline"],
        capture_output=True, text=True, check=True).stdout
    iter_commits = [line for line in log.splitlines() if "iter " in line]
    assert len(iter_commits) == 3


def test_incomplete_multiseed_batch_is_recorded_and_aborts_iteration(
    tmp_path: Path,
):
    proj = _write_minimal_project(tmp_path)
    config = proj / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "tests.test_sculpt._TestAdapterStub",
            "tests.test_sculpt._SeedEvidenceAdapter",
        ),
        encoding="utf-8",
    )

    def fitness(_iter_dir):
        return 0.5

    fitness.detail = lambda _iter_dir: {"spec_score": 0.5}  # type: ignore[attr-defined]
    fitness.detail_dir = lambda _rollout_dir: {  # type: ignore[attr-defined]
        "spec_score": 0.6
    }

    with pytest.raises(RuntimeError, match="objective evaluation incomplete"):
        sculpt_run(
            config_path=config,
            behavior_goal="record every requested evaluation seed",
            iterations=1,
            no_kg=True,
            dry_run=True,
            fitness_fn=fitness,
            eval_seeds=3,
            fresh_eval_seeds=0,
        )

    iter_dir = proj / "runs" / "iter_0"
    plan = json.loads(
        (iter_dir / "evaluation_plan.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (iter_dir / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert plan["requested_seeds"] == [10_000, 10_001, 10_002]
    assert receipt["requested_count"] == 3
    assert receipt["completed_count"] == 2
    assert receipt["complete"] is False
    assert [item["status"] for item in receipt["results"]] == [
        "succeeded", "failed", "succeeded",
    ]
    assert not (iter_dir / "fitness.json").exists()
    assert not (iter_dir / "diagnosis.json").exists()
    assert not (iter_dir / "iteration_complete.json").exists()
    assert _find_resume_start_iteration(
        proj / "rewards", proj / "runs",
    ) == 0


def test_primary_evaluation_failure_records_all_requested_seeds(
    tmp_path: Path,
):
    proj = _write_minimal_project(tmp_path)
    config = proj / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "tests.test_sculpt._TestAdapterStub",
            "tests.test_sculpt._PrimarySeedFailureAdapter",
        ),
        encoding="utf-8",
    )

    def fitness(_iter_dir):
        return 0.5

    fitness.detail = lambda _iter_dir: {"spec_score": 0.5}  # type: ignore[attr-defined]
    fitness.detail_dir = lambda _rollout_dir: {  # type: ignore[attr-defined]
        "spec_score": 0.5
    }

    with pytest.raises(RuntimeError, match="primary seed failed"):
        sculpt_run(
            config_path=config,
            behavior_goal="fail visibly",
            iterations=1,
            no_kg=True,
            dry_run=True,
            fitness_fn=fitness,
            eval_seeds=3,
            fresh_eval_seeds=0,
        )

    receipt = json.loads(
        (proj / "runs" / "iter_0" / "evaluation_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["status"] for item in receipt["results"]] == [
        "failed", "not_run", "not_run",
    ]
    assert [item["seed"] for item in receipt["results"]] == [
        10_000, 10_001, 10_002,
    ]
    iter_dir = proj / "runs" / "iter_0"
    assert not (iter_dir / "iteration_complete.json").exists()
    assert _find_resume_start_iteration(
        proj / "rewards", proj / "runs",
    ) == 0


def test_sculpt_run_applies_resource_overrides_without_mutating_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Hardware overrides are launch-scoped, generic, and reproducible."""
    import sculptor.sculpt as sculpt_module

    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)
    config = proj / "config.toml"
    original = config.read_text(encoding="utf-8").replace(
        "tests.test_sculpt._TestAdapterStub",
        "tests.test_sculpt._ResourceAdapterStub",
    )
    config.write_text(original, encoding="utf-8")

    loaded: dict[str, object] = {}
    real_load = sculpt_module.load_adapter

    def capture_load(path):
        adapter = real_load(path)
        loaded["adapter"] = adapter
        return adapter

    monkeypatch.setattr(sculpt_module, "load_adapter", capture_load)
    sculpt_run(
        config_path=config,
        behavior_goal="traverse rough terrain",
        iterations=1,
        no_kg=True,
        dry_run=True,
        num_envs=256,
        device="cpu",
    )

    adapter = loaded["adapter"]
    assert getattr(adapter, "num_envs") == 256
    assert getattr(adapter, "device") == "cpu"
    assert config.read_text(encoding="utf-8") == original
    context = json.loads(
        (proj / "reports" / "run_context.json").read_text(encoding="utf-8")
    )
    effective = context["config"]["effective"]["adapter"]["config"]
    assert effective["num_envs"] == 256
    assert effective["device"] == "cpu"


def test_sculpt_run_does_not_stop_after_three_non_improving(tmp_path: Path):
    global _SCHEDULE
    # best is iter 0 (10). Iters 1-3 all fall short, but metric dips are
    # no longer an auto-kill criterion.
    _SCHEDULE = [10.0, 1.0, 2.0, 3.0, 9.0]
    proj = _write_minimal_project(tmp_path)
    cfg = proj / "config.toml"

    result = sculpt_run(
        config_path=cfg, behavior_goal="dummy goal",
        iterations=5, resume=False, no_kg=True, dry_run=True,
    )
    assert result.early_stopped is False
    assert result.iterations_run == 5


def test_sculpt_run_resume_picks_up_at_latest_v(tmp_path: Path):
    """After a first --dry-run, re-running with --resume continues from the
    next iter rather than restarting at 0."""
    global _SCHEDULE
    _SCHEDULE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    proj = _write_minimal_project(tmp_path)
    cfg = proj / "config.toml"

    # First run: 2 iters.
    r1 = sculpt_run(
        config_path=cfg, behavior_goal="g",
        iterations=2, resume=False, no_kg=True, dry_run=True,
    )
    assert r1.iterations_run == 2
    # Now v0, v1, v2 exist.
    assert (proj / "rewards" / "v2.py").is_file()

    # Second run: --resume 1 iter. Should start at iter 2, write v3.
    r2 = sculpt_run(
        config_path=cfg, behavior_goal="g",
        iterations=1, resume=True, no_kg=True, dry_run=True,
    )
    assert r2.iterations_run == 1
    assert r2.completed_iters[0].iter_index == 2
    assert (proj / "rewards" / "v3.py").is_file()


# ── Per-phase resume regression: skip train when ckpt is on disk ────
def test_train_or_resume_rejects_unattested_checkpoint(tmp_path: Path):
    """A parseable checkpoint without exact input/completion receipts is
    stale evidence and must retrain rather than inherit the new request."""
    import torch
    from sculptor.adapters.base import TrainResult
    from sculptor.sculpt import _train_or_resume

    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    # Write a valid torch pickle so the integrity check passes.
    torch.save({"model": "ok"}, iter_dir / "checkpoint.pt")
    (iter_dir / "metrics.json").write_text('{"steps": 1500}', encoding="utf-8")

    reward = tmp_path / "reward.py"
    reward.write_text("reward = 1\n", encoding="utf-8")

    class _Adapter:
        called = False

        def train(self, **_kw):
            self.called = True
            torch.save({"model": "new"}, iter_dir / "checkpoint.pt")
            return TrainResult(
                checkpoint_path=iter_dir / "checkpoint.pt",
                metrics_dict={}, component_means={}, logs_path=iter_dir / "logs",
            )

    adapter = _Adapter()
    result = _train_or_resume(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=1500,
        seed=0,
    )
    assert isinstance(result, TrainResult)
    assert result.checkpoint_path == iter_dir / "checkpoint.pt"
    assert adapter.called is True
    assert (iter_dir / "train_input_manifest.json").is_file()
    assert (iter_dir / "train_completion_manifest.json").is_file()


def test_train_or_resume_falls_through_on_corrupt_checkpoint(tmp_path: Path):
    """If `checkpoint.pt` is present but torch.load explodes (SIGKILL
    mid-copy pre-atomic-write scenario), we must retrain rather than
    hand a rotted artifact to the rollout step."""
    from sculptor.sculpt import _train_or_resume

    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    (iter_dir / "checkpoint.pt").write_bytes(b"not-a-torch-pickle")
    reward = tmp_path / "reward.py"
    reward.write_text("reward = 1\n", encoding="utf-8")

    class _StubAdapter:
        def __init__(self):
            self.train_called = False

        def train(self, **_kw):
            self.train_called = True
            # Write a fake checkpoint so downstream logic has something.
            import torch
            torch.save({"model": "ok"}, iter_dir / "checkpoint.pt")
            from sculptor.adapters.base import TrainResult
            return TrainResult(
                checkpoint_path=iter_dir / "checkpoint.pt",
                metrics_dict={}, component_means={}, logs_path=iter_dir / "logs",
            )

    adapter = _StubAdapter()
    result = _train_or_resume(
        adapter=adapter, iter_index=0, iter_dir=iter_dir,
        reward_module_path=reward,
        steps=1500, seed=0,
    )
    assert adapter.train_called, "corrupt ckpt must trigger a fresh train"
    assert result is not None


def test_mode_admission_event_is_emitted_only_at_fresh_train_boundary(
    tmp_path: Path, monkeypatch,
):
    import sculptor.sculpt as sculpt_mod
    from sculptor.adapters.base import TrainResult

    event = {"type": "mode_execution_admitted", "reward_sha256": "a" * 64}
    observed: list[dict] = []
    monkeypatch.setattr(sculpt_mod, "_emit_event", observed.append)

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    reward = tmp_path / "reward.py"
    reward.write_text("reward = 1\n", encoding="utf-8")

    class _Adapter:
        def train(self, output_dir, **_kwargs):
            import torch

            assert observed == [event]
            checkpoint = Path(output_dir) / "checkpoint.pt"
            torch.save({"model": "ok"}, checkpoint)
            return TrainResult(
                checkpoint_path=checkpoint,
                metrics_dict={},
                component_means={},
                logs_path=fresh / "logs",
            )

    sculpt_mod._train_or_resume(
        adapter=_Adapter(),
        iter_index=0,
        iter_dir=fresh,
        reward_module_path=reward,
        steps=1,
        seed=0,
        pre_train_event=event,
    )

    observed.clear()
    sculpt_mod._train_or_resume(
        adapter=_Adapter(),
        iter_index=0,
        iter_dir=fresh,
        reward_module_path=reward,
        steps=1,
        seed=0,
        pre_train_event=event,
    )
    assert not any(item.get("type") == "mode_execution_admitted" for item in observed)


def test_rollout_or_resume_requires_exact_receipts(tmp_path: Path):
    """Legacy artifacts rerun once; exact receipts then skip; a partial
    output reruns again."""
    from sculptor.sculpt import _rollout_or_resume

    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    for name in ("rollout.mp4", "trajectory.npz", "behavior.json"):
        (rollout_dir / name).write_bytes(b"x")  # non-empty so the size check passes

    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")

    class _StubAdapter:
        def __init__(self):
            self.calls = 0

        def rollout(self, **_kw):
            self.calls += 1
            for name in ("rollout.mp4", "trajectory.npz", "behavior.json"):
                (rollout_dir / name).write_bytes(
                    f"{name}:{self.calls}".encode()
                )

    adapter = _StubAdapter()
    _rollout_or_resume(
        adapter=adapter, iter_index=0, rollout_dir=rollout_dir,
        checkpoint_path=checkpoint, n_episodes=6,
    )
    assert adapter.calls == 1

    _rollout_or_resume(
        adapter=adapter, iter_index=0, rollout_dir=rollout_dir,
        checkpoint_path=checkpoint, n_episodes=6,
    )
    assert adapter.calls == 1

    # Remove one artifact → rollout must re-run.
    (rollout_dir / "behavior.json").unlink()

    _rollout_or_resume(
        adapter=adapter, iter_index=0, rollout_dir=rollout_dir,
        checkpoint_path=checkpoint, n_episodes=6,
    )
    assert adapter.calls == 2, "missing behavior.json must trigger re-run"


# ── Metric-plateau early-stop compatibility ───────────────────────────────
def test_should_early_stop_default_never_fires() -> None:
    """Metric-plateau auto-kill is disabled even with legacy defaults."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop([1.0, 1.5, 2.0, 1.8, 1.9, 1.7]) is False


def test_should_early_stop_disabled_never_fires() -> None:
    """Legacy enabled=False remains accepted and never fires."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop(
        [1.0, 0.5, 0.2, 0.1, 0.05, 0.01], enabled=False,
    ) is False


def test_should_early_stop_patience_zero_disables() -> None:
    """Legacy patience=0 remains accepted and never fires."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop([1.0, 0.5, 0.2], patience=0) is False


def test_should_early_stop_patience_5_never_fires() -> None:
    """Legacy patience overrides are ignored by the disabled auto-kill."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop(
        [1.0, 2.0, 1.5, 1.4, 1.3, 1.2], patience=5,
    ) is False
    assert _should_early_stop(
        [1.0, 2.0, 1.5, 1.4, 1.3, 1.2, 1.1], patience=5,
    ) is False


def test_should_early_stop_handles_nones_in_history() -> None:
    """None entries are accepted; history still never halts the run."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop([None, None, None, None]) is False
    assert _should_early_stop([5.0, None, 3.0, None, 2.0, 1.0]) is False


def test_should_early_stop_negative_patience_disables() -> None:
    """Legacy negative patience remains accepted and never fires."""
    from sculptor.sculpt import _should_early_stop
    assert _should_early_stop([5.0, 1.0, 0.5], patience=-1) is False


def test_config_template_early_stop_defaults_parse() -> None:
    """Fresh `sculpt init` writes a config.toml whose [iteration] has
    legacy early-stop compatibility fields parseable by tomllib."""
    import tempfile
    import tomllib
    from pathlib import Path as _P
    from sculptor.sculpt import sculpt_init

    with tempfile.TemporaryDirectory() as td:
        proj = sculpt_init(_P(td) / "defaults_proj", adapter="gym_sb3")
        with (proj / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
        iter_cfg = cfg["iteration"]
        assert iter_cfg["early_stop_enabled"] is False
        assert iter_cfg["early_stop_patience"] == 3


def test_sculpt_run_ignores_legacy_early_stop_and_runs_full_budget(
    tmp_path, monkeypatch,
):
    """Even when the primary_metric is strictly decreasing every iter,
    the removed metric-plateau auto-kill lets the loop run to completion."""
    from sculptor.sculpt import sculpt_init, sculpt_run

    proj = sculpt_init(tmp_path / "long_overnight", adapter="gym_sb3")
    config_path = proj / "config.toml"

    # Force --dry-run so we never hit Claude / training. Dry-run also
    # caps training steps at 1000 but still runs _run_one_iter. Patch the
    # heavy adapter methods to return stub results so each iter finishes
    # in ~1ms.
    from sculptor.adapters.base import TrainResult, RolloutResult

    class _StubAdapter:
        def reward_contract(self):
            from sculptor.adapters.base import RewardContract
            import gymnasium as gym
            import numpy as np
            return RewardContract(
                observation_space_spec=gym.spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float32),
                action_space_spec=gym.spaces.Box(
                    -1, 1, shape=(2,), dtype=np.float32),
                expected_info_keys=[], expected_components=None,
            )

        def train(self, *, reward_module_path, output_dir, steps, seed):
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt = output_dir / "checkpoint.pt"
            ckpt.write_bytes(b"stub")
            return TrainResult(
                checkpoint_path=ckpt, metrics_dict={},
                component_means={}, logs_path=output_dir / "logs",
            )

        def rollout(self, *, checkpoint_path, output_dir, n_episodes, **_):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "rollout.mp4").write_bytes(b"stub" * 800)
            import numpy as np
            np.savez_compressed(
                output_dir / "trajectory.npz",
                rewards=np.zeros(1, dtype=np.float32),
                episode_id=np.zeros(1, dtype=np.int32),
            )
            # Emit a strictly-decreasing mean_return.
            import json
            mean = 10.0 - self._tick * 2.0
            self._tick += 1
            (output_dir / "behavior.json").write_text(json.dumps({
                "n_episodes": n_episodes,
                "mean_return": mean,
                "mean_episode_length": 42.0,
                "max_episode_length": 42,
            }))
            return RolloutResult(
                video_path=output_dir / "rollout.mp4",
                keyframes_dir=output_dir / "keyframes",
                trajectory_path=output_dir / "trajectory.npz",
                n_episodes=n_episodes,
            )

        def compute_behavior_metrics(self, rollout):
            import json
            return json.loads(
                (rollout.video_path.parent / "behavior.json").read_text()
            )

        _tick = 0

    monkeypatch.setattr(
        "sculptor.sculpt.load_adapter", lambda _p: _StubAdapter(),
    )

    result = sculpt_run(
        config_path=config_path,
        behavior_goal="stub goal",
        iterations=6,  # 6 strictly-decreasing iters; runs full budget.
        dry_run=True,
        no_kg=True,
    )
    assert result.iterations_run == 6
    assert result.early_stopped is False


# ── §2026-07-04: run-boundary trained-reward resolution ────────────────────
def test_current_reward_target_parses_reexport(tmp_path: Path):
    from sculptor.sculpt import _current_reward_target, _write_current_reexport

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v0.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (rewards / "v4.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")

    # No current.py yet → None.
    assert _current_reward_target(rewards) is None
    # Re-export target resolves even when a HIGHER version exists on disk.
    _write_current_reexport(rewards, rewards / "v0.py")
    assert _current_reward_target(rewards) == rewards / "v0.py"
    _write_current_reexport(rewards, rewards / "v4.py")
    assert _current_reward_target(rewards) == rewards / "v4.py"
    # Hand-edited/unrecognizable current.py → None (callers fall back).
    (rewards / "current.py").write_text("from .v4 import *\n", encoding="utf-8")
    assert _current_reward_target(rewards) is None
    # Recognizable line but the target file is gone → None.
    _write_current_reexport(rewards, rewards / "v0.py")
    (rewards / "v0.py").unlink()
    assert _current_reward_target(rewards) is None


def test_run_boundary_trains_and_edits_current_target(
        tmp_path: Path, capsys):
    """§2026-07-04 regression (tuck-jump iter 16): when a previous run's
    best-by-fitness selection repointed current.py at an OLDER version and
    a new run resumes, the iter must (a) record that version as
    reward_path_trained — keep-best must never keep a never-trained file,
    (b) apply its diagnosis edit to THAT source, and (c) report that
    version in the iter events — not the highest v<n>.py on disk."""
    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)
    rewards = proj / "rewards"
    # Simulate the previous run: a later edit v2 exists on disk with a
    # DISTINCT hyperparameter value, but best-selection kept v0.
    v2_src = (rewards / "v0.py").read_text(encoding="utf-8").replace(
        '"version": "v0"', '"version": "v2"').replace(
        '"alive_bonus": 1.0', '"alive_bonus": 7.0')
    (rewards / "v2.py").write_text(v2_src, encoding="utf-8")
    from sculptor.sculpt import _write_current_reexport
    _write_current_reexport(rewards, rewards / "v0.py")

    result = sculpt_run(
        config_path=proj / "config.toml", behavior_goal="dummy goal",
        iterations=1, resume=True, no_kg=True, dry_run=True,
    )
    (outcome,) = result.completed_iters
    assert outcome.iter_index == 2               # resume starts at latest_n

    # (a) the trained record follows current.py's re-export target.
    assert outcome.reward_path_trained == rewards / "v0.py"

    # (b) the new edit is still numbered v3 (monotonic) but its content
    # derives from v0 (the dry-run canned edit bumps alive_bonus by 0.5:
    # 1.0 → 1.5). Pre-fix it derived from v2 and read 7.5.
    v3 = (rewards / "v3.py").read_text(encoding="utf-8")
    assert '"alive_bonus": 1.5' in v3
    assert '"alive_bonus": 7.5' not in v3

    # (c) events report the version that actually trained.
    events = [
        json.loads(line.split("[SCULPT-EVENT] ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[SCULPT-EVENT] ")
    ]
    started = [e for e in events if e["type"] == "iter_started"]
    completed = [e for e in events if e["type"] == "iter_completed"]
    assert started and started[0]["reward_version_before"] == 0
    assert completed and completed[0]["reward_version_before"] == 0
    assert completed[0]["reward_version_after"] == 3


def test_iter_fitness_event_carries_components_dict(
        tmp_path: Path, capsys):
    """§D24 (F4): the `iter_fitness` SCULPT-EVENT must carry the same
    per-channel component breakdown the diagnoser sees, not scalars only.
    Before this, the D20/D23 hollow-success class (criterion reads
    'satisfied' while the certified fitness is ~0) was invisible in the
    live event stream — recovering WHICH channel zeroed the fitness
    required an offline recompute of the metric."""
    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)

    class _DetailFitnessFn:
        """Fake fitness_fn exposing the `.detail` accessor (see
        `sculptor.eval.spec_metrics.make_spec_fitness_fn`) so the iter's
        detail path runs without needing a real metric/rollout shape."""

        def __call__(self, iter_dir):
            return 0.0

        def detail(self, iter_dir):
            return {
                "spec_score": 0.0, "spec_name": "torso_righting",
                "gate_upright_frac": 1.0, "gate_reached_035": 0.0,
                "error": None,
            }

        metric_id = "test-metric"
        metric_version = "v1"
        metric_source = "test"
        metric_sha256 = "d" * 64

    result = sculpt_run(
        config_path=proj / "config.toml", behavior_goal="dummy goal",
        iterations=1, no_kg=True, dry_run=True,
        fitness_fn=_DetailFitnessFn(),
    )
    events = [
        json.loads(line.split("[SCULPT-EVENT] ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[SCULPT-EVENT] ")
    ]
    fitness_events = [e for e in events if e["type"] == "iter_fitness"]
    assert fitness_events, events
    assert fitness_events[0]["components"] == {
        "gate_upright_frac": 1.0, "gate_reached_035": 0.0,
    }

    # §Ship-56: the live `iter_fitness` event must also be MIRRORED to
    # `<iter_dir>/fitness.json` so a finished stage's per-iteration
    # fitness survives past the job log (the UI probes this file).
    (outcome,) = result.completed_iters
    fitness_json = Path(outcome.iter_dir) / "fitness.json"
    assert fitness_json.is_file()
    record = json.loads(fitness_json.read_text(encoding="utf-8"))
    assert record["iter"] == outcome.iter_index
    assert record["fitness"] == pytest.approx(0.0)
    assert record["components"] == {
        "gate_upright_frac": 1.0, "gate_reached_035": 0.0,
    }
    assert record["metric"] == {
        "id": "test-metric",
        "version": "v1",
        "source": "test",
        "sha256": "d" * 64,
    }
    assert record["source"] == "live"
    assert record["observe_only"] is False
    assert "recorded_at" in record


def test_no_fitness_json_written_when_fitness_fn_is_none(
        tmp_path: Path, capsys):
    """The blind default (`fitness_fn=None`) never runs the fitness detail
    path — `fitness.json` must not appear in the iter dir at all, not an
    empty/stale placeholder."""
    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)

    result = sculpt_run(
        config_path=proj / "config.toml", behavior_goal="dummy goal",
        iterations=1, no_kg=True, dry_run=True,
    )
    (outcome,) = result.completed_iters
    assert not (Path(outcome.iter_dir) / "fitness.json").exists()


def test_iter_fitness_event_components_null_on_plain_float_fallback(
        tmp_path: Path, capsys):
    """A `fitness_fn` with no `.detail` accessor (the plain-float
    fallback path) must carry `"components": None` on the event — never
    a crash, never a stale/empty dict masquerading as real data."""
    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)

    sculpt_run(
        config_path=proj / "config.toml", behavior_goal="dummy goal",
        iterations=1, no_kg=True, dry_run=True,
        fitness_fn=lambda iter_dir: 0.42,
    )
    events = [
        json.loads(line.split("[SCULPT-EVENT] ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[SCULPT-EVENT] ")
    ]
    fitness_events = [e for e in events if e["type"] == "iter_fitness"]
    assert fitness_events, events
    assert fitness_events[0]["fitness"] == pytest.approx(0.42)
    assert fitness_events[0]["components"] is None


def test_dangling_current_reexport_fails_closed_without_latest_fallback(
    tmp_path: Path,
):
    """A missing selected reward is lost authority, not permission to guess.

    Selecting the newest surviving version would silently change the reward
    the researcher chose. Launch must stop before adapter work and leave the
    selector untouched so the repair is explicit and reviewable.
    """
    global _SCHEDULE
    _SCHEDULE = [1.0]
    proj = _write_minimal_project(tmp_path)
    rewards = proj / "rewards"
    v1_src = (rewards / "v0.py").read_text(encoding="utf-8").replace(
        '"version": "v0"', '"version": "v1"')
    (rewards / "v1.py").write_text(v1_src, encoding="utf-8")
    from sculptor.sculpt import _current_reward_target, _write_current_reexport
    # current.py -> v0, then v0 vanishes (user cleanup).
    _write_current_reexport(rewards, rewards / "v0.py")
    (rewards / "v0.py").unlink()
    assert _current_reward_target(rewards) is None

    with pytest.raises(ValueError, match="selects a missing"):
        sculpt_run(
            config_path=proj / "config.toml", behavior_goal="dummy goal",
            iterations=1, resume=True, no_kg=True, dry_run=True,
        )
    assert "v0.py" in (rewards / "current.py").read_text(encoding="utf-8")
    assert (rewards / "v1.py").is_file()


def test_current_reward_target_parses_ui_backend_format(tmp_path: Path):
    """The UI backend (reward_store.py) writes current.py with a
    `_TARGET = Path(__file__).resolve().parent / 'v<n>.py'` line instead
    of sculptor's `_LATEST = _HERE / ...`. Both must resolve — a
    UI-rewritten current.py otherwise silently reverts the run-boundary
    fix to the buggy latest-version fallback. (Ported from the parallel
    worktree fix, commit 07c6da2.)"""
    from sculptor.sculpt import _current_reward_target

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v1.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (rewards / "v5.py").write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    (rewards / "current.py").write_text(
        '"""Auto-generated by the UI backend."""\n'
        "import importlib.util\n"
        "from pathlib import Path\n"
        "_TARGET = Path(__file__).resolve().parent / 'v1.py'\n"
        "_spec = importlib.util.spec_from_file_location(\n"
        '    "_sculpt_current_reward", _TARGET)\n',
        encoding="utf-8",
    )
    assert _current_reward_target(rewards) == rewards / "v1.py"
