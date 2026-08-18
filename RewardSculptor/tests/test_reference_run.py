from __future__ import annotations

import hashlib
import json
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
        "meta": {
            "source": "test:gait",
            "composition": {
                "segments": [
                    {"label": "a", "source_id": "walk-start"},
                    {"label": "b", "source_id": "walk-finish"},
                ],
                "seam_frames": [45],
            },
        },
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


def _mode_reward_source(
    clip_id: str,
    *,
    project: Path | None = None,
    refs_root: Path | None = None,
    robot: str = "demo",
) -> str:
    """The load-bearing shape of a promoted per-mode module: MODE_ORDER,
    MODE_WINDOWS_S, and the clip it was scaffolded from."""
    spec: dict = {
        "version": "mode-reward-v1",
        "reference_clip_id": clip_id,
    }
    if project is not None and refs_root is not None:
        from sculptor.mode_rewards import (
            MODE_BINDING_CONTEXT_REFS,
            build_mode_reward_binding,
        )

        from sculptor.modes import (
            build_mode_execution_manifest,
            mode_graph_sha256,
            modes_from_composition,
        )
        from sculptor.reference import load_clip

        selection = json.loads(
            (project / "env" / "selection_current.json").read_text()
        )
        context_refs = {
            kind: selection["refs"][kind]["sha256"]
            for kind in MODE_BINDING_CONTEXT_REFS
            if kind in selection["refs"]
        }
        clip_path = library.clip_dir(
            robot, clip_id, root=refs_root
        ) / library.CLIP_FILENAME
        graph = modes_from_composition(
            load_clip(clip_path), clip_id=clip_id
        )
        graph_sha = mode_graph_sha256(graph)
        manifest = build_mode_execution_manifest(graph).to_dict()
        spec["mode_execution_manifest"] = manifest
        spec["mode_binding"] = build_mode_reward_binding(
            clip_id=clip_id,
            robot=robot,
            clip_sha256=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
            graph_sha256=graph_sha,
            context_refs=context_refs,
            execution_manifest=manifest,
        )
    windows = (spec.get("mode_execution_manifest") or {}).get(
        "windows_s", {"a": [0.0, 1.0], "b": [1.0, 2.0]}
    )
    order = list(windows)
    return (
        "REWARD_SPEC = {\n"
        + "".join(f"    {key!r}: {value!r},\n" for key, value in spec.items())
        + "}\n"
        f"MODE_WINDOWS_S: dict = {windows!r}\n"
        f"MODE_ORDER: list = {order!r}\n"
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


def _promote_test_reward(
    *,
    project: Path,
    rewards: Path,
    adapter: SimpleNamespace,
    store,
    version: int,
    source: str,
) -> Path:
    """Make a reward genuinely current in both execution and tuple truth."""
    from sculptor.edit import _write_current_reexport
    from sculptor.world.artifacts import ArtifactRef

    reward = rewards / f"v{version}.py"
    reward.write_text(source, encoding="utf-8")
    _write_current_reexport(rewards, reward)
    selection = store.read_selection(project / "env" / "selection_current.json")
    assert selection is not None
    refs = dict(selection.refs)
    refs["reward"] = ArtifactRef.from_path(
        "reward", f"v{version}", reward, base=project
    )
    store.promote(
        refs, evaluation_lineage=selection.evaluation_lineage
    )
    adapter.world_selection_path = str(
        project / "env" / "selection_current.json"
    )
    return reward


def test_pretrain_mode_admission_rederives_exact_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _validated_mode_execution_admission

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch,
    )
    reward = _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source(
            clip_id,
            project=project,
            refs_root=tmp_path / "references",
        ),
    )
    selection = store.read_selection(
        project / "env" / "selection_current.json"
    )
    assert selection is not None
    selection_path = (
        project / "env" / f"selection_v{selection.selection_version}.json"
    )
    event = _validated_mode_execution_admission(
        project=project,
        reward_path=reward,
        selection_path=selection_path,
    )
    assert event is not None
    assert event["type"] == "mode_execution_admitted"
    assert event["source"] == "sculpt_run_worker"
    assert event["robot"] == "demo"
    assert event["clip_id"] == clip_id
    assert event["tuple_hash"] == selection.tuple_hash
    assert event["reward_sha256"] == hashlib.sha256(
        reward.read_bytes()
    ).hexdigest()


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
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source(
            clip_id,
            project=project,
            refs_root=tmp_path / "references",
        ),
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

    assert not (rewards / "v2.py").exists(), "the authored modes were replaced"
    assert installed["reward_version"] == "v1"
    assert installed["source"] == "promoted_mode_reward"
    assert installed["phase_mode"] == "per_mode"
    # The tuple still names what actually trains, so lineage is not left
    # pointing at the starter reward.
    promoted = store.read_selection(Path(adapter.world_selection_path))
    assert promoted is not None
    assert promoted.refs["reward"].version == "v1"


def test_promoted_mode_reward_cannot_self_attest_a_forged_graph_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutually consistent forged graph/binding must fail before training.

    The exact reference clip has its own independently derivable seam at frame
    45. Moving that seam to frame 30 and updating both the manifest and binding
    used to pass because admission trusted the reward's own graph digest.
    """
    from sculptor.mode_rewards import (
        MODE_BINDING_CONTEXT_REFS,
        _rewrite_reward_spec,
        build_mode_reward_binding,
    )
    from sculptor.modes import (
        Guard,
        Mode,
        ModeGraph,
        Transition,
        build_mode_execution_manifest,
        mode_graph_sha256,
    )
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    refs_root = tmp_path / "references"
    source = _mode_reward_source(
        clip_id, project=project, refs_root=refs_root
    )
    forged_graph = ModeGraph(
        modes=(Mode("a", (0, 30)), Mode("b", (30, 90))),
        transitions=(Transition("a", "b", Guard("phase", at_phase=1.0)),),
        fps=30.0,
        source={"kind": "composition", "clip_id": clip_id},
    )
    forged_manifest = build_mode_execution_manifest(forged_graph).to_dict()
    selection = store.read_selection(project / "env" / "selection_current.json")
    assert selection is not None
    context_refs = {
        kind: ref.sha256
        for kind, ref in selection.refs.items()
        if kind in MODE_BINDING_CONTEXT_REFS
    }
    clip_path = library.clip_dir(
        "demo", clip_id, root=refs_root
    ) / library.CLIP_FILENAME
    forged_binding = build_mode_reward_binding(
        clip_id=clip_id,
        robot="demo",
        clip_sha256=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        graph_sha256=mode_graph_sha256(forged_graph),
        context_refs=context_refs,
        execution_manifest=forged_manifest,
    )
    source = _rewrite_reward_spec(source, {
        "mode_execution_manifest": forged_manifest,
        "mode_binding": forged_binding,
    })
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=source,
    )

    with pytest.raises(
        ValueError,
        match="independently derived clip graph",
    ):
        _prepare_reference_guided_run(
            adapter=adapter,
            project=project,
            rewards_dir=rewards,
            behavior_goal="weave through the course",
            clip_id=clip_id,
            robot="demo",
            kg_store=None,
            dry_run=True,
        )


def test_promoted_mode_reward_resolves_current_not_highest_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    source = _mode_reward_source(
        clip_id,
        project=project,
        refs_root=tmp_path / "references",
    )
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=2,
        source=source,
    )
    # A later file can exist after keep-best points current.py back to v2.
    (rewards / "v4.py").write_text(source, encoding="utf-8")

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

    assert installed["reward_version"] == "v2"
    assert Path(installed["reward_path"]).name == "v2.py"


def test_promoted_mode_reward_fails_when_current_and_selection_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.edit import _write_current_reexport
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    source = _mode_reward_source(
        clip_id,
        project=project,
        refs_root=tmp_path / "references",
    )
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=2,
        source=source,
    )
    v4 = rewards / "v4.py"
    v4.write_text(source, encoding="utf-8")
    _write_current_reexport(rewards, v4)

    with pytest.raises(ValueError, match="current.py.*pinned selection"):
        _prepare_reference_guided_run(
            adapter=adapter,
            project=project,
            rewards_dir=rewards,
            behavior_goal="weave through the course",
            clip_id=clip_id,
            robot="demo",
            kg_store=None,
            dry_run=True,
        )


def test_same_named_phase_reward_without_exact_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source(clip_id),
    )

    with pytest.raises(ValueError, match="mode binding is stale or incomplete"):
        _prepare_reference_guided_run(
            adapter=adapter,
            project=project,
            rewards_dir=rewards,
            behavior_goal="weave through the course",
            clip_id=clip_id,
            robot="demo",
            kg_store=None,
            dry_run=True,
        )


def test_same_named_phase_reward_is_not_reused_after_clip_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    refs_root = tmp_path / "references"
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source(
            clip_id, project=project, refs_root=refs_root
        ),
    )
    clip_path = library.clip_dir(
        "demo", clip_id, root=refs_root
    ) / library.CLIP_FILENAME
    clip_path.write_bytes(clip_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="clip_sha256"):
        _prepare_reference_guided_run(
            adapter=adapter,
            project=project,
            rewards_dir=rewards,
            behavior_goal="weave through the course",
            clip_id=clip_id,
            robot="demo",
            kg_store=None,
            dry_run=True,
        )


def test_same_named_phase_reward_is_not_reused_after_world_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.sculpt import _prepare_reference_guided_run
    from sculptor.world.artifacts import ArtifactRef

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch
    )
    refs_root = tmp_path / "references"
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source(
            clip_id, project=project, refs_root=refs_root
        ),
    )
    current = store.read_selection(project / "env" / "selection_current.json")
    assert current is not None
    changed_task = project / "env" / "task_v2.json"
    changed_task.write_text('{"goal": "changed"}\n')
    changed_refs = dict(current.refs)
    changed_refs["task"] = ArtifactRef.from_path(
        "task", "v2", changed_task, base=project
    )
    store.promote(changed_refs, evaluation_lineage="changed-context")

    with pytest.raises(ValueError, match="context_refs"):
        _prepare_reference_guided_run(
            adapter=adapter,
            project=project,
            rewards_dir=rewards,
            behavior_goal="weave through the course",
            clip_id=clip_id,
            robot="demo",
            kg_store=None,
            dry_run=True,
        )


def test_official_rollout_persists_digest_bound_mode_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.eval.mode_metrics import mode_diagnostics_digest
    from sculptor.mode_rewards import (
        MODE_BINDING_CONTEXT_REFS,
        build_mode_reward_binding,
    )
    from sculptor.modes import (
        Guard,
        Mode,
        ModeGraph,
        Transition,
        build_mode_execution_manifest,
        mode_graph_sha256,
    )
    from sculptor.sculpt import _persist_mode_diagnostics
    from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

    project = tmp_path / "project"
    rewards = project / "rewards"
    env_dir = project / "env"
    iter_dir = project / "runs" / "iter_0"
    rollout_dir = iter_dir / "rollout"
    rewards.mkdir(parents=True)
    env_dir.mkdir(parents=True)
    rollout_dir.mkdir(parents=True)

    graph = ModeGraph(
        modes=(Mode("approach", (0, 50)), Mode("traverse", (50, 100))),
        transitions=(Transition(
            "approach", "traverse", Guard("phase", at_phase=1.0)
        ),),
        fps=50.0,
    )
    manifest = build_mode_execution_manifest(graph)
    env_spec = env_dir / "v0.json"
    env_spec.write_text("{}\n")
    refs: dict[str, ArtifactRef] = {
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v0", env_spec, base=project
        ),
    }
    for kind in (
        "world", "task", "resolved_eval", "channel_catalog",
        "clarifications",
    ):
        path = env_dir / f"{kind}_v1.json"
        path.write_text("{}\n")
        refs[kind] = ArtifactRef.from_path(kind, "v1", path, base=project)
    context_refs = {
        kind: refs[kind].sha256
        for kind in MODE_BINDING_CONTEXT_REFS
        if kind in refs
    }
    clip_sha = "a" * 64
    binding = build_mode_reward_binding(
        clip_id="composite-1",
        robot="g1",
        clip_sha256=clip_sha,
        graph_sha256=mode_graph_sha256(graph),
        context_refs=context_refs,
        execution_manifest=manifest.to_dict(),
    )
    reward = rewards / "v1.py"
    reward.write_text(
        "REWARD_SPEC = " + repr({
            "version": "mode-reward-v1",
            "reference_clip_id": "composite-1",
            "reference_robot": "g1",
            "mode_execution_manifest": manifest.to_dict(),
            "mode_binding": binding,
        }) + "\n"
        "MODE_WINDOWS_S = {'approach': (0.0, 1.0), "
        "'traverse': (1.0, 2.0)}\n"
        "MODE_ORDER = ['approach', 'traverse']\n"
    )
    refs["reward"] = ArtifactRef.from_path(
        "reward", "v1", reward, base=project
    )
    store = WorldArtifactStore(project)
    selection = store.promote(refs, evaluation_lineage="test")
    selection_path = env_dir / f"selection_v{selection.selection_version}.json"
    (iter_dir / "artifact_tuple.json").write_bytes(selection_path.read_bytes())

    mask = np.ones((100, 2), dtype=bool)
    mask[75:, 1] = False
    np.savez_compressed(
        rollout_dir / "trajectory.npz",
        first_episode_valid_mask=mask,
        joint_pos=np.zeros((100, 2, 4), dtype=np.float32),
        reward_term__tracking=np.ones((100, 2), dtype=np.float32),
        # Deliberately ragged metadata: the production loader excludes it,
        # while the source-file digest still binds its bytes.
        episode_id=np.arange(175, dtype=np.int32),
    )
    (rollout_dir / "behavior.json").write_text(json.dumps({
        "step_dt": 0.02,
        "rollout_num_envs": 2,
        "max_episode_steps": 100,
        "rendered_env_index": 1,
    }))

    monkeypatch.setattr(
        "sculptor.reference_run.load_exact_reference_motion",
        lambda **_kwargs: ({"fps": 50.0}, {"schema": 1}, clip_sha),
    )
    monkeypatch.setattr(
        "sculptor.modes.modes_from_composition",
        lambda _clip, clip_id=None: graph,
    )

    record = _persist_mode_diagnostics(
        project=project,
        iter_index=0,
        iter_dir=iter_dir,
        rollout_dir=rollout_dir,
        reward_path=reward,
        selection_path=selection_path,
    )

    assert record is not None
    persisted = json.loads(
        (iter_dir / "mode_metrics.json").read_text(encoding="utf-8")
    )
    assert persisted["diagnostic_digest"] == mode_diagnostics_digest(persisted)
    assert persisted["authority"]["classification"] == "diagnostic_only"
    assert persisted["modes"]["traverse"]["entered_env_count"] == 2
    assert persisted["modes"]["traverse"]["completed_window_env_count"] == 1
    assert persisted["transitions"][0]["fired_per_env"] == [True, True]
    assert "episode_id" not in persisted["rollout"]["time_indexed_channels"]


def test_official_mode_diagnostics_fail_closed_without_execution_manifest(
    tmp_path: Path,
) -> None:
    from sculptor.sculpt import _persist_mode_diagnostics

    reward = tmp_path / "reward.py"
    reward.write_text(
        "REWARD_SPEC = {'reference_clip_id': 'clip', "
        "'reference_robot': 'g1', 'mode_binding': {}}\n"
        "MODE_WINDOWS_S = {'mode': (0.0, 1.0)}\n"
        "MODE_ORDER = ['mode']\n"
    )

    with pytest.raises(ValueError, match="missing mode_execution_manifest"):
        _persist_mode_diagnostics(
            project=tmp_path,
            iter_index=0,
            iter_dir=tmp_path / "iter_0",
            rollout_dir=tmp_path / "rollout",
            reward_path=reward,
            selection_path=None,
        )


def test_a_per_mode_reward_for_a_different_clip_does_not_block_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse is keyed on the clip, not on "is per-mode".

    Asking for a different motion than the promoted modes were authored
    against is a real request to change what the run tracks.
    """
    from sculptor.sculpt import _prepare_reference_guided_run

    clip_id, project, rewards, adapter, store = _reference_project(
        tmp_path, monkeypatch)
    _promote_test_reward(
        project=project,
        rewards=rewards,
        adapter=adapter,
        store=store,
        version=1,
        source=_mode_reward_source("some-other-clip--g1"),
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
