from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.adapters.base import TrainResult
from sculptor.run_manifests import (
    build_rollout_input_manifest,
    build_train_input_manifest,
    input_manifest_matches,
    manifest_sha256,
    verify_iteration_completion_marker,
    write_json_atomic,
)
from sculptor.sculpt import (
    _rollout_or_resume,
    _train_or_resume,
    _write_iteration_completion_marker,
)


class _Adapter:
    env_id = "unit-env"
    robot = "unit-robot"
    control_dt = 0.02

    def __init__(self) -> None:
        self.train_calls = 0
        self.rollout_calls = 0
        self.fail_train = False
        self.fail_rollout = False

    def train(self, *, output_dir: Path, **_kwargs):
        if self.fail_train:
            raise AssertionError("exact train resume should have skipped")
        self.train_calls += 1
        import torch

        checkpoint = output_dir / "checkpoint.pt"
        torch.save({"call": self.train_calls}, checkpoint)
        return TrainResult(
            checkpoint_path=checkpoint,
            metrics_dict={},
            component_means={},
            logs_path=output_dir / "logs",
        )

    def rollout(self, *, output_dir: Path, **_kwargs):
        if self.fail_rollout:
            raise AssertionError("exact rollout resume should have skipped")
        self.rollout_calls += 1
        for name in ("rollout.mp4", "trajectory.npz", "behavior.json"):
            (output_dir / name).write_bytes(
                f"{name}:{self.rollout_calls}".encode()
            )


def _reward(tmp_path: Path, text: str = "reward = 1\n") -> Path:
    path = tmp_path / "reward.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_train_resume_requires_exact_input_and_completion_manifests(tmp_path):
    adapter = _Adapter()
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    reward = _reward(tmp_path)

    first = _train_or_resume(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=10,
        seed=7,
        manifest_context={"world": "a" * 64},
    )
    assert first.checkpoint_path.is_file()
    assert adapter.train_calls == 1

    adapter.fail_train = True
    resumed = _train_or_resume(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=10,
        seed=7,
        manifest_context={"world": "a" * 64},
    )
    assert resumed.checkpoint_path == first.checkpoint_path
    assert adapter.train_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 11),
        ("seed", 8),
        ("context", {"world": "b" * 64}),
        ("init_policy_mode", "actor_only"),
    ],
)
def test_train_manifest_change_forces_retrain(tmp_path, field, value):
    adapter = _Adapter()
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    reward = _reward(tmp_path)
    kwargs = {
        "adapter": adapter,
        "iter_index": 0,
        "iter_dir": iter_dir,
        "reward_module_path": reward,
        "steps": 10,
        "seed": 7,
        "init_policy_mode": "actor_critic",
        "manifest_context": {"world": "a" * 64},
    }
    _train_or_resume(**kwargs)
    kwargs[field if field != "context" else "manifest_context"] = value
    _train_or_resume(**kwargs)
    assert adapter.train_calls == 2


def test_reward_byte_change_forces_retrain(tmp_path):
    adapter = _Adapter()
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    reward = _reward(tmp_path)
    kwargs = dict(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=10,
        seed=7,
    )
    _train_or_resume(**kwargs)
    reward.write_text("reward = 2\n", encoding="utf-8")
    _train_or_resume(**kwargs)
    assert adapter.train_calls == 2


def test_checkpoint_byte_change_invalidates_train_completion(tmp_path):
    adapter = _Adapter()
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    reward = _reward(tmp_path)
    kwargs = dict(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=10,
        seed=7,
    )
    first = _train_or_resume(**kwargs)
    import torch

    torch.save({"tampered": True}, first.checkpoint_path)
    _train_or_resume(**kwargs)
    assert adapter.train_calls == 2


def test_rollout_resume_binds_checkpoint_reward_seed_render_and_outputs(tmp_path):
    adapter = _Adapter()
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    reward = _reward(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    kwargs = dict(
        adapter=adapter,
        iter_index=0,
        rollout_dir=rollout_dir,
        checkpoint_path=checkpoint,
        reward_module_path=reward,
        n_episodes=3,
        seed=101,
        render_width=320,
        manifest_context={"reset": {"mode": "default"}},
    )
    _rollout_or_resume(**kwargs)
    assert adapter.rollout_calls == 1

    adapter.fail_rollout = True
    _rollout_or_resume(**kwargs)
    assert adapter.rollout_calls == 1

    adapter.fail_rollout = False
    kwargs["seed"] = 102
    _rollout_or_resume(**kwargs)
    assert adapter.rollout_calls == 2

    # Output bytes are part of completion evidence too.
    (rollout_dir / "behavior.json").write_bytes(b"tampered")
    _rollout_or_resume(**kwargs)
    assert adapter.rollout_calls == 3


def test_manifest_helpers_reject_missing_required_inputs(tmp_path):
    adapter = _Adapter()
    with pytest.raises(ValueError, match="required manifest input is missing"):
        build_train_input_manifest(
            adapter=adapter,
            iteration=0,
            reward_module_path=tmp_path / "missing.py",
            steps=1,
            seed=0,
            init_policy_path=None,
            init_policy_mode="actor_critic",
        )
    with pytest.raises(ValueError, match="required manifest input is missing"):
        build_rollout_input_manifest(
            adapter=adapter,
            iteration=0,
            checkpoint_path=tmp_path / "missing.pt",
            reward_module_path=None,
            n_episodes=1,
            seed=None,
            max_episode_steps=None,
            playback_speed=None,
            render_every=None,
            fps=None,
            render_width=None,
            render_height=None,
            render_env_index=None,
        )


def test_input_manifest_comparison_is_canonical(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"schema": 1, "phase": "train", "nested": {"b": 2, "a": 1}}
    write_json_atomic(path, manifest)
    assert input_manifest_matches(path, manifest)
    assert len(manifest_sha256(manifest)) == 64
    assert not input_manifest_matches(path, {**manifest, "phase": "rollout"})


def _completed_schema3_iteration(tmp_path: Path) -> tuple[Path, Path]:
    adapter = _Adapter()
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir()
    reward = _reward(tmp_path)
    trained = _train_or_resume(
        adapter=adapter,
        iter_index=0,
        iter_dir=iter_dir,
        reward_module_path=reward,
        steps=10,
        seed=7,
    )
    _rollout_or_resume(
        adapter=adapter,
        iter_index=0,
        rollout_dir=rollout_dir,
        checkpoint_path=trained.checkpoint_path,
        reward_module_path=reward,
        n_episodes=1,
        seed=11,
    )
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=0,
        checkpoint_path=trained.checkpoint_path,
        reward_version_before=0,
        reward_version_after=None,
        world_selection_hash=None,
    )
    return iter_dir, reward


def test_schema3_completion_reverifies_all_phase_receipts(tmp_path):
    iter_dir, _ = _completed_schema3_iteration(tmp_path)

    receipt = verify_iteration_completion_marker(iter_dir)

    assert receipt is not None
    assert receipt["schema"] == 3
    assert receipt["checkpoint"] == "checkpoint.pt"
    assert len(receipt["phase_manifests_sha256"]) == 64


def test_schema3_completion_accepts_bound_objective_plan_and_results(tmp_path):
    iter_dir, _ = _completed_schema3_iteration(tmp_path)
    plan = {
        "schema": 1,
        "iteration": 0,
        "authority": "precommitted_objective_evaluation_seeds",
        "requested_count": 2,
        "requested_seeds": [10_000, 10_001],
        "rollout_episodes_per_seed": 2,
    }
    results = {
        "schema": 1,
        "iteration": 0,
        "plan": plan,
        "requested_count": 2,
        "completed_count": 2,
        "complete": True,
        "results": [
            {"seed": 10_000, "status": "succeeded"},
            {"seed": 10_001, "status": "succeeded"},
        ],
    }
    write_json_atomic(iter_dir / "evaluation_plan.json", plan)
    write_json_atomic(iter_dir / "evaluation_results.json", results)
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=0,
        checkpoint_path=iter_dir / "checkpoint.pt",
        reward_version_before=0,
        reward_version_after=None,
        world_selection_hash=None,
    )

    receipt = verify_iteration_completion_marker(iter_dir)

    assert receipt is not None
    assert set(receipt["phase_manifests"]) >= {
        "evaluation_plan.json", "evaluation_results.json",
    }

    (iter_dir / "evaluation_results.json").write_text(
        "{}\n", encoding="utf-8",
    )
    assert verify_iteration_completion_marker(iter_dir) is None


def test_schema3_completion_rejects_bound_incomplete_evaluation(tmp_path):
    iter_dir, _ = _completed_schema3_iteration(tmp_path)
    plan = {
        "schema": 1,
        "iteration": 0,
        "authority": "precommitted_objective_evaluation_seeds",
        "requested_count": 2,
        "requested_seeds": [10_000, 10_001],
        "rollout_episodes_per_seed": 2,
    }
    results = {
        "schema": 1,
        "iteration": 0,
        "plan": plan,
        "requested_count": 2,
        "completed_count": 1,
        "complete": False,
        "results": [
            {"seed": 10_000, "status": "succeeded"},
            {
                "seed": 10_001,
                "status": "failed",
                "error": "rollout failed",
            },
        ],
    }
    write_json_atomic(iter_dir / "evaluation_plan.json", plan)
    write_json_atomic(iter_dir / "evaluation_results.json", results)
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=0,
        checkpoint_path=iter_dir / "checkpoint.pt",
        reward_version_before=0,
        reward_version_after=None,
        world_selection_hash=None,
    )

    assert verify_iteration_completion_marker(iter_dir) is None


def test_schema3_completion_rejects_unknown_phase_receipt(tmp_path):
    iter_dir, _ = _completed_schema3_iteration(tmp_path)
    marker_path = iter_dir / "iteration_complete.json"
    import json

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["phase_manifests"]["unrecognized.json"] = {
        "sha256": "0" * 64,
        "bytes": 1,
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert verify_iteration_completion_marker(iter_dir) is None


@pytest.mark.parametrize(
    "tamper",
    ["phase-manifest", "rollout-output", "reward-input", "marker-receipt"],
)
def test_schema3_completion_fails_closed_on_bound_byte_change(
    tmp_path: Path, tamper: str,
):
    iter_dir, reward = _completed_schema3_iteration(tmp_path)
    if tamper == "phase-manifest":
        (iter_dir / "train_input_manifest.json").write_text(
            "{}\n", encoding="utf-8",
        )
    elif tamper == "rollout-output":
        (iter_dir / "rollout" / "behavior.json").write_bytes(b"changed")
    elif tamper == "reward-input":
        reward.write_text("reward = 2\n", encoding="utf-8")
    else:
        marker_path = iter_dir / "iteration_complete.json"
        import json

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["phase_manifests"]["train_input_manifest.json"][
            "sha256"
        ] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert verify_iteration_completion_marker(iter_dir) is None
