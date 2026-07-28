from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sculptor.reference import save_clip
from sculptor.reference_run import (
    ReferenceRunError,
    build_reference_guided_reward,
    load_exact_reference_motion,
    reference_input_hash,
)
from sculptor.refs import library


def _register_gait(root: Path, *, robot: str = "demo") -> str:
    clip_id = "walk_cycle"
    n = 90
    fps = 30.0
    t = np.arange(n, dtype=np.float64) / fps
    phase = 2.0 * np.pi * t / 0.75
    clip = {
        "root_pos_z": 0.7 + 0.01 * np.sin(phase),
        "root_pos_xy": np.stack([0.6 * t, np.zeros_like(t)], axis=1),
        "fps": fps,
        "joint_pos": np.stack([np.sin(phase), np.cos(phase)], axis=1),
        "joint_names": ["left_joint", "right_joint"],
        "meta": {"source": "test:gait"},
    }
    clip_path = save_clip(
        library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME,
        clip,
    )
    provenance = library.make_provenance(
        clip_id=clip_id,
        robot=robot,
        source={"kind": "unit-test"},
        license="CC0",
        attribution="test",
        content_sha256_=library.content_sha256(clip_path.read_bytes()),
        text="walk forward",
        fps_source=fps,
    )
    library.write_provenance(robot, clip_id, provenance, root=root)
    return clip_id


def test_exact_reference_pair_loads_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    clip_id = _register_gait(tmp_path)

    clip, provenance, clip_sha = load_exact_reference_motion(
        clip_id=clip_id, robot="demo",
    )

    assert provenance["robot"] == "demo"
    assert clip["joint_names"] == ["left_joint", "right_joint"]
    assert len(clip_sha) == 64
    with pytest.raises(ReferenceRunError, match="missing"):
        load_exact_reference_motion(clip_id=clip_id, robot="other")


def test_dry_run_builds_looping_immutable_motion_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    clip_id = _register_gait(tmp_path)

    built = build_reference_guided_reward(
        clip_id=clip_id,
        robot="demo",
        behavior_goal="weave through the course",
        reward_version="v4",
        reward_contract=object(),
        dry_run=True,
    )

    assert built.phase_mode == "loop"
    assert built.task_residual_authored is False
    assert built.target_sha256 in built.source
    assert '"type": "reference_tracking_residual"' in built.source


def test_live_build_accepts_only_bounded_residual_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor import reference_run

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    clip_id = _register_gait(tmp_path)

    def _bounded_edit(**kwargs):
        source = Path(kwargs["current_reward_path"]).read_text()
        source = source.replace(
            "    return 0.0\n\n\ndef compute_reward",
            "    return 0.1\n\n\ndef compute_reward",
        ).replace(
            "    return torch.zeros_like(like)\n\n\ndef compute_reward_batched",
            "    return torch.full_like(like, 0.1)\n\n\ndef compute_reward_batched",
        )
        target = Path(kwargs["current_reward_path"]).parent / (
            kwargs["new_iter_id"] + ".py"
        )
        target.write_text(source)
        return target

    monkeypatch.setattr(reference_run, "apply_prompt_edit", _bounded_edit)
    built = build_reference_guided_reward(
        clip_id=clip_id,
        robot="demo",
        behavior_goal="weave through the course",
        reward_version="v4",
        reward_contract=object(),
    )

    assert built.task_residual_authored is True
    assert "return 0.1" in built.source


def test_live_build_rejects_reference_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor import reference_run

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    clip_id = _register_gait(tmp_path)

    def _tampering_edit(**kwargs):
        source = Path(kwargs["current_reward_path"]).read_text()
        source = source.replace(
            "REFERENCE_TARGET_SHA256 = '",
            "REFERENCE_TARGET_SHA256 = 'tampered-",
            1,
        )
        target = Path(kwargs["current_reward_path"]).parent / (
            kwargs["new_iter_id"] + ".py"
        )
        target.write_text(source)
        return target

    monkeypatch.setattr(reference_run, "apply_prompt_edit", _tampering_edit)
    with pytest.raises(ReferenceRunError, match="immutable"):
        build_reference_guided_reward(
            clip_id=clip_id,
            robot="demo",
            behavior_goal="weave through the course",
            reward_version="v4",
            reward_contract=object(),
        )


def test_reference_input_hash_binds_goal_and_clip_content() -> None:
    base = reference_input_hash(
        clip_id="walk", robot="demo", clip_sha256="a" * 64,
        behavior_goal="weave",
    )
    assert base == reference_input_hash(
        clip_id="walk", robot="demo", clip_sha256="a" * 64,
        behavior_goal=" weave ",
    )
    assert base != reference_input_hash(
        clip_id="walk", robot="demo", clip_sha256="b" * 64,
        behavior_goal="weave",
    )


def test_reference_run_promotes_one_atomic_tuple_and_resumes_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _prepare_reference_guided_run
    from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

    refs_root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(refs_root))
    clip_id = _register_gait(refs_root)

    project = tmp_path / "project"
    rewards = project / "rewards"
    env_dir = project / "env"
    reports = project / "reports"
    rewards.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    reports.mkdir()
    reward_v0 = rewards / "v0.py"
    reward_v0.write_text(
        "REWARD_SPEC = {'version': 'v0'}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {'alive': 0.0}\n"
    )
    env_v0 = env_dir / "v0.json"
    env_v0.write_text("{}\n")

    store = WorldArtifactStore(project)
    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v0", reward_v0, base=project,
        ),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v0", env_v0, base=project,
        ),
    }
    for kind in (
        "world", "task", "resolved_eval", "channel_catalog",
        "clarifications",
    ):
        path = env_dir / f"{kind}_v1.json"
        path.write_text("{}\n")
        refs[kind] = ArtifactRef.from_path(kind, "v1", path, base=project)
    first = store.promote(refs, evaluation_lineage="test-lineage")

    adapter = SimpleNamespace(
        world_selection_path=str(env_dir / "selection_current.json"),
        reward_contract=lambda: object(),
    )
    installed = _prepare_reference_guided_run(
        adapter=adapter,
        project=project,
        rewards_dir=rewards,
        behavior_goal="weave through the course",
        clip_id=clip_id,
        robot="demo",
        kg_store=None,
        dry_run=True,
    )

    assert installed["reward_version"] == "v1"
    assert installed["tuple_hash"] != first.tuple_hash
    promoted = store.read_selection(Path(adapter.world_selection_path))
    assert promoted is not None
    assert promoted.refs["reward"].version == "v1"
    assert promoted.refs["env_spec"].version == "v0"

    reused = _prepare_reference_guided_run(
        adapter=adapter,
        project=project,
        rewards_dir=rewards,
        behavior_goal="weave through the course",
        clip_id=clip_id,
        robot="demo",
        kg_store=None,
        dry_run=True,
    )
    assert reused["tuple_hash"] == installed["tuple_hash"]
    assert not (rewards / "v2.py").exists()


def _mode_reward_source(clip_id: str) -> str:
    """The load-bearing shape of a promoted per-mode module: MODE_ORDER,
    MODE_WINDOWS_S, and the clip it was scaffolded from."""
    return (
        "REWARD_SPEC = {\n"
        "    'version': 'mode-reward-v1',\n"
        f"    \"reference_clip_id\": {clip_id!r},\n"
        "}\n"
        "MODE_WINDOWS_S: dict = {'a': (0.0, 1.0), 'b': (1.0, 2.0)}\n"
        "MODE_ORDER: list = ['a', 'b']\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {}\n"
        "def compute_reward_batched(state, action, next_state, info):\n"
        "    return 0.0, {}\n"
    )


def _reference_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A project with a promoted world selection, ready for a guided run.

    Mirrors `test_reference_run_promotes_one_atomic_tuple_and_resumes_
    idempotently` so the two tests exercise the same entry conditions.
    """
    from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

    refs_root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(refs_root))
    clip_id = _register_gait(refs_root)

    project = tmp_path / "project"
    rewards = project / "rewards"
    env_dir = project / "env"
    rewards.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    (project / "reports").mkdir()
    reward_v0 = rewards / "v0.py"
    reward_v0.write_text(
        "REWARD_SPEC = {'version': 'v0'}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {'alive': 0.0}\n"
    )
    env_v0 = env_dir / "v0.json"
    env_v0.write_text("{}\n")

    store = WorldArtifactStore(project)
    refs = {
        "reward": ArtifactRef.from_path("reward", "v0", reward_v0,
                                        base=project),
        "env_spec": ArtifactRef.from_path("env_spec", "v0", env_v0,
                                          base=project),
    }
    for kind in ("world", "task", "resolved_eval", "channel_catalog",
                 "clarifications"):
        path = env_dir / f"{kind}_v1.json"
        path.write_text("{}\n")
        refs[kind] = ArtifactRef.from_path(kind, "v1", path, base=project)
    store.promote(refs, evaluation_lineage="test-lineage")

    adapter = SimpleNamespace(
        world_selection_path=str(env_dir / "selection_current.json"),
        reward_contract=lambda: object(),
    )
    return clip_id, project, rewards, adapter, store


def test_a_promoted_per_mode_reward_for_this_clip_is_kept_not_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two installers, one `current.py`.

    `promote_mode_reward` copies an authored per-mode module into the chain
    and repoints `current.py`; `_prepare_reference_guided_run` builds a flat
    tracking reward as the NEXT version and repoints `current.py` too. Doing
    both for the same clip used to discard every authored mode body — they
    stayed on disk as `mode_reward_v<n>.py`, stopped being trained, and no
    event said so. A per-mode reward for this clip already carries the
    tracking backbone, so rebuilding is a strict downgrade.
    """
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch)
    (rewards / "v1.py").write_text(_mode_reward_source(clip_id))

    installed = _prepare_reference_guided_run(
        adapter=adapter,
        project=project,
        rewards_dir=rewards,
        behavior_goal="weave through the course",
        clip_id=clip_id,
        robot="demo",
        kg_store=None,
        dry_run=True,
    )

    assert not (rewards / "v2.py").exists(), "the authored modes were replaced"
    assert installed["reward_version"] == "v1"
    assert installed["source"] == "promoted_mode_reward"
    assert installed["phase_mode"] == "per_mode"
    # The tuple still names what actually trains, so lineage is not left
    # pointing at the starter reward.
    promoted = store.read_selection(Path(adapter.world_selection_path))
    assert promoted is not None
    assert promoted.refs["reward"].version == "v1"


def test_a_per_mode_reward_for_a_different_clip_does_not_block_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse is keyed on the clip, not on "is per-mode".

    Asking for a different motion than the promoted modes were authored
    against is a real request to change what the run tracks.
    """
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, _store = _reference_project(
        tmp_path, monkeypatch)
    (rewards / "v1.py").write_text(_mode_reward_source("some-other-clip--g1"))

    installed = _prepare_reference_guided_run(
        adapter=adapter,
        project=project,
        rewards_dir=rewards,
        behavior_goal="weave through the course",
        clip_id=clip_id,
        robot="demo",
        kg_store=None,
        dry_run=True,
    )

    assert (rewards / "v2.py").exists()
    assert installed["reward_version"] == "v2"
    assert installed.get("source") != "promoted_mode_reward"


def test_a_flat_reward_at_the_head_of_the_chain_still_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not fire on an ordinary project."""
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, _store = _reference_project(
        tmp_path, monkeypatch)

    installed = _prepare_reference_guided_run(
        adapter=adapter,
        project=project,
        rewards_dir=rewards,
        behavior_goal="weave through the course",
        clip_id=clip_id,
        robot="demo",
        kg_store=None,
        dry_run=True,
    )

    assert installed["reward_version"] == "v1"
    assert installed.get("source") != "promoted_mode_reward"
