from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.artifact_lineage import (
    LineageObservationError,
    RunLineageSession,
)
from sculptor.kg.lineage import LineageConflict
from sculptor.kg.schema import Relation, SoftwareEnvironment
from sculptor.kg.store import SculptorKG
from sculptor.reference import save_clip
from sculptor.refs import library as reference_library
from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore


def _seed_world(project: Path) -> tuple[str, str]:
    project.mkdir(parents=True, exist_ok=True)
    reward = project / "rewards" / "v0.py"
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text("def compute_reward(_obs): return 0.0\n", encoding="utf-8")
    env_spec = project / "env" / "v0.json"
    env_spec.parent.mkdir(parents=True, exist_ok=True)
    env_spec.write_text("{}\n", encoding="utf-8")
    store = WorldArtifactStore(project)
    refs = {
        "reward": ArtifactRef.from_path("reward", "v0", reward, base=project),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v0", env_spec, base=project,
        ),
        "world": store.write_json("world", {"kind": "test-world"}),
        "task": store.write_json("task", {"kind": "test-task"}),
        "resolved_eval": store.write_json(
            "resolved_eval", {"kind": "test-eval"},
        ),
        "channel_catalog": store.write_json(
            "channel_catalog", {"channels": []},
        ),
        "clarifications": store.write_json("clarifications", {}),
    }
    selection = store.promote(refs, evaluation_lineage="test-eval-v1")
    return (
        selection.tuple_hash,
        f"selection_v{selection.selection_version}.json",
    )


def _seed_reference(root: Path) -> tuple[str, str, str]:
    robot = "g1"
    clip_id = "lineage_motion"
    clip_dir = reference_library.clip_dir(robot, clip_id, root=root)
    clip_dir.mkdir(parents=True)
    n = 16
    clip_path = save_clip(clip_dir / reference_library.CLIP_FILENAME, {
        "root_pos_z": np.full(n, 0.78),
        "root_pos_xy": np.zeros((n, 2)),
        "fps": 30.0,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["j0", "j1"],
        "meta": {
            "source": "unit-test:composed",
            "composition": {
                "segments": [
                    {"label": "approach", "source_id": "source-a"},
                    {"label": "finish", "source_id": "source-b"},
                ],
                "seam_frames": [8],
            },
        },
    })
    digest = hashlib.sha256(clip_path.read_bytes()).hexdigest()
    provenance = reference_library.make_provenance(
        clip_id=clip_id,
        robot=robot,
        source={"kind": "unit-test"},
        license="CC0",
        attribution="unit test",
        content_sha256_=digest,
        fps_source=30.0,
    )
    reference_library.write_provenance(
        robot, clip_id, provenance, root=root,
    )
    return robot, clip_id, digest


def _install_mode_reward(
    project: Path,
    reference_root: Path,
    *,
    robot: str,
    clip_id: str,
    clip_sha256: str,
) -> tuple[dict, str, str]:
    from sculptor.mode_rewards import (
        MODE_BINDING_CONTEXT_REFS,
        build_mode_reward_binding,
        mode_execution_manifest_digest,
    )
    from sculptor.modes import (
        build_mode_execution_manifest,
        mode_graph_sha256,
        modes_from_composition,
    )
    from sculptor.reference import load_clip

    store = WorldArtifactStore(project)
    selection = store.read_selection(project / "env" / "selection_current.json")
    assert selection is not None
    clip_path = reference_library.clip_dir(
        robot, clip_id, root=reference_root,
    ) / reference_library.CLIP_FILENAME
    graph = modes_from_composition(load_clip(clip_path), clip_id=clip_id)
    graph_sha256 = mode_graph_sha256(graph)
    manifest = build_mode_execution_manifest(graph).to_dict()
    context_refs = {
        kind: ref.sha256
        for kind, ref in selection.refs.items()
        if kind in MODE_BINDING_CONTEXT_REFS
    }
    binding = build_mode_reward_binding(
        clip_id=clip_id,
        robot=robot,
        clip_sha256=clip_sha256,
        graph_sha256=graph_sha256,
        context_refs=context_refs,
        execution_manifest=manifest,
    )
    spec = {
        "version": "v1",
        "reference_clip_id": clip_id,
        "reference_robot": robot,
        "mode_windows_s": manifest["windows_s"],
        "mode_execution_manifest": manifest,
        "mode_binding": binding,
    }
    reward = project / "rewards" / "v1.py"
    reward.write_text(
        f"REWARD_SPEC = {spec!r}\n"
        f"MODE_WINDOWS_S = {manifest['windows_s']!r}\n"
        f"MODE_ORDER = {list(manifest['windows_s'])!r}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {}\n",
        encoding="utf-8",
    )
    refs = dict(selection.refs)
    refs["reward"] = ArtifactRef.from_path(
        "reward", "v1", reward, base=project,
    )
    promoted = store.promote(
        refs, evaluation_lineage=selection.evaluation_lineage,
    )
    selection_name = f"selection_v{promoted.selection_version}.json"
    event = {
        "type": "mode_execution_admitted",
        "source": "sculpt_run_worker",
        "reward_path": str(reward.resolve()),
        "reward_sha256": hashlib.sha256(reward.read_bytes()).hexdigest(),
        "robot": robot,
        "clip_id": clip_id,
        "clip_sha256": clip_sha256,
        "graph_sha256": graph_sha256,
        "execution_manifest_digest": mode_execution_manifest_digest(manifest),
        "context_refs": context_refs,
        "selection": selection_name,
        "tuple_hash": promoted.tuple_hash,
    }
    return event, selection_name, promoted.tuple_hash


def _write_run_context(
    project: Path,
    *,
    commit: str,
    dirty: bool,
    packages: object | None = None,
    diff_sha256: str | None = None,
    captured_at: str = "2026-08-17T00:00:00+00:00",
) -> tuple[dict, dict]:
    config_sha = hashlib.sha256(b"effective-config").hexdigest()
    context = {
        "schema": 2,
        "captured_at": captured_at,
        "code_git": {
            "available": True,
            "sha": commit,
            "branch": "codex/test",
            "dirty": dirty,
            **(
                {"diff_sha256": diff_sha256 or ("d" * 64)}
                if dirty
                else {}
            ),
        },
        "config": {"sha256": config_sha, "effective": {"seed": 42}},
        "seeds": {"base_seed": 42},
        "packages": (
            {"torch": "2.11.0", "reward-sculptor": "0.1.0"}
            if packages is None
            else packages
        ),
        "python": {
            "version": "3.12.4",
            "implementation": "CPython",
            "platform": "Linux-test",
        },
        "env": {
            "sculptor_vars_set": [],
            "remote_enabled": False,
            "remote_host": None,
        },
        "prompts": {"dir": "/immutable/prompts", "sha256": {}},
        "llm": {"edit": {"model_id": "test", "temperature": "sdk_default"}},
    }
    path = project / "reports" / "run_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, sort_keys=True), encoding="utf-8")
    event = {
        "type": "run_context_captured",
        "path": str(path),
        "code_sha": commit,
        "code_dirty": dirty,
        "config_sha256": config_sha,
        "base_seed": 42,
    }
    return context, event


def _lineage_session(project: Path, run_id: str) -> RunLineageSession:
    return RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id=run_id,
        requested_initialization_mode="auto_resume",
    )


def test_run_lineage_uses_observed_load_and_new_checkpoint_only(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    tuple_hash, selection_name = _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)

    old_checkpoint = project / "runs" / "iter_0" / "checkpoint.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b"old-policy")
    old_sha = hashlib.sha256(old_checkpoint.read_bytes()).hexdigest()

    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-lineage",
        requested_initialization_mode="auto_resume",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
    )
    session.record_started()
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
    })
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        "reference_robot": robot,
        "reference_clip_id": clip_id,
        "clip_sha256": clip_sha,
        "rollout_sha256": "7" * 64,
        "certificate_sha256": "8" * 64,
        "execution_contract_sha256": "9" * 64,
        "execution_boundary_sha256": "a" * 64,
        "target_robot": robot,
    })
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
        },
    })

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.TRACKS) == 1
        runs = kg.find_nodes(kind="TrainingRun")
        assert len(runs) == 1
        tracks_edge, _motion_id = kg.neighbors(
            runs[0].id, relation=Relation.TRACKS,
        )[0]
        assert tracks_edge.data == {
            "authority": "reference_feasibility_admitted+run_started",
            "verified": True,
            "tier": "D",
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "rollout_sha256": "7" * 64,
            "certificate_sha256": "8" * 64,
            "execution_contract_sha256": "9" * 64,
            "execution_boundary_sha256": "a" * 64,
            "target_robot": robot,
        }
        assert kg.count_edges(Relation.EXECUTES_IN) == 1
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0
        assert runs[0].selection_digest == tuple_hash

    load_event = {
        "type": "warm_start_loaded",
        "source": str(old_checkpoint),
        "source_sha256": old_sha,
        "source_sha8": old_sha[:8],
        "load_cfg_keys": ["critic", "actor"],
    }
    session.observe_event(load_event)
    session.observe_event(load_event)  # replay is idempotent

    new_checkpoint = project / "runs" / "iter_1" / "checkpoint.pt"
    new_checkpoint.parent.mkdir(parents=True)
    new_checkpoint.write_bytes(b"new-policy")
    outputs = session.record_outputs()
    assert [item.sha256 for item in outputs] == [
        hashlib.sha256(b"new-policy").hexdigest()
    ]
    session.record_outputs()  # disk/event replay remains idempotent

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 1
        assert kg.count_edges(Relation.PRODUCED) == 1
        assert kg.count_edges(Relation.DERIVED_FROM) == 1
        produced = kg.neighbors(runs[0].id, relation=Relation.PRODUCED)
        assert produced[0][1] == outputs[0].id


def test_reference_tracks_rejects_missing_or_untrusted_tierd_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    project.mkdir()
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    context = {
        "robot": robot,
        "clip_id": clip_id,
        "clip_sha256": clip_sha,
    }

    missing = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-no-tierd",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
    )
    missing.record_started()
    with pytest.raises(LineageObservationError, match="no prior verified Tier-D"):
        missing.observe_event({"type": "run_started", "reference_motion": context})

    spoofed = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-spoofed-tierd",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
    )
    spoofed.record_started()
    with pytest.raises(LineageObservationError, match="worker admission"):
        spoofed.observe_event({
            "type": "reference_feasibility_admitted",
            "source": "ui_launch",
            "reference_robot": robot,
            "reference_clip_id": clip_id,
            "clip_sha256": clip_sha,
            "rollout_sha256": "7" * 64,
            "certificate_sha256": "8" * 64,
            "execution_contract_sha256": "9" * 64,
            "execution_boundary_sha256": "a" * 64,
            "target_robot": robot,
        })

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.TRACKS) == 0


def test_mode_execution_event_records_exact_reverified_authority(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    event, selection_name, tuple_hash = _install_mode_reward(
        project,
        reference_root,
        robot=robot,
        clip_id=clip_id,
        clip_sha256=clip_sha,
    )
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-mode-lineage",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
    )
    session.record_started()
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        "reference_robot": robot,
        "reference_clip_id": clip_id,
        "clip_sha256": clip_sha,
        "rollout_sha256": "7" * 64,
        "certificate_sha256": "8" * 64,
        "execution_contract_sha256": "9" * 64,
        "execution_boundary_sha256": "a" * 64,
        "target_robot": robot,
    })
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
        },
    })
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
    })
    session.observe_event(event)
    session.observe_event(event)

    with SculptorKG(kg_path) as kg:
        artifacts = kg.find_nodes(kind="ModeExecutionArtifact")
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.reward_sha256 == event["reward_sha256"]
        assert artifact.graph_sha256 == event["graph_sha256"]
        assert artifact.execution_manifest_digest == event[
            "execution_manifest_digest"
        ]
        assert artifact.selection_digest == tuple_hash
        assert artifact.context_refs == event["context_refs"]
        assert kg.count_edges(Relation.USES_MODE_EXECUTION) == 1

    stale = {**event, "graph_sha256": "f" * 64}
    with pytest.raises(
        LineageObservationError, match="stale clip/graph/manifest",
    ):
        session.observe_event(stale)
    reward_path = Path(str(event["reward_path"]))
    reward_path.write_text(
        reward_path.read_text(encoding="utf-8") + "\n# post-admission drift\n",
        encoding="utf-8",
    )
    with pytest.raises(
        LineageObservationError, match="digest differs from reward bytes",
    ):
        session.observe_event(event)
    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.USES_MODE_EXECUTION) == 1


def test_run_context_lineage_distinguishes_clean_and_dirty_same_commit(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    project.mkdir()
    commit = "a" * 40

    clean = _lineage_session(project, "job-clean")
    clean.record_started()
    _context, clean_event = _write_run_context(
        project, commit=commit, dirty=False,
    )
    clean.observe_event(clean_event)
    clean.observe_event(clean_event)  # exact event replay is idempotent

    dirty = _lineage_session(project, "job-dirty")
    dirty.record_started()
    _context, dirty_event = _write_run_context(
        project, commit=commit, dirty=True,
    )
    dirty.observe_event(dirty_event)
    dirty.observe_event(dirty_event)

    with SculptorKG(kg_path) as kg:
        software = kg.find_nodes(kind="SoftwareEnvironment")
        assert len(software) == 2
        assert all(isinstance(node, SoftwareEnvironment) for node in software)
        assert {node.code_commit for node in software} == {commit}
        assert {node.code_dirty for node in software} == {False, True}
        assert len({node.id for node in software}) == 2
        dirty_node = next(node for node in software if node.code_dirty)
        clean_node = next(node for node in software if not node.code_dirty)
        assert dirty_node.captured_source_sha256 is None
        assert clean_node.captured_source_sha256 is None
        assert dirty_node.code_diff_digest == "d" * 64
        assert dirty_node.versions["torch"] == "2.11.0"
        assert dirty_node.runtime["python"]["version"] == "3.12.4"

        runs = {node.run_id: node for node in kg.find_nodes(kind="TrainingRun")}
        assert runs["job-clean"].code_commit == commit
        assert runs["job-dirty"].code_commit == commit
        assert kg.count_edges(Relation.EXECUTES_IN) == 2
        for run in runs.values():
            edges = kg.neighbors(run.id, relation=Relation.EXECUTES_IN)
            assert len(edges) == 1
            edge, destination = edges[0]
            assert destination in {node.id for node in software}
            assert edge.data["authority"] == "run_context_captured"
            assert edge.data["verified"] is True
            assert len(edge.data["captured_source_sha256"]) == 64


def test_dirty_software_identity_ignores_capture_time_but_separates_patches(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    project.mkdir()
    commit = "e" * 40

    first = _lineage_session(project, "job-patch-a-1")
    first.record_started()
    _context, event = _write_run_context(
        project,
        commit=commit,
        dirty=True,
        diff_sha256="1" * 64,
        captured_at="2026-08-17T00:00:00+00:00",
    )
    first.observe_event(event)

    replay = _lineage_session(project, "job-patch-a-2")
    replay.record_started()
    _context, event = _write_run_context(
        project,
        commit=commit,
        dirty=True,
        diff_sha256="1" * 64,
        captured_at="2026-08-17T00:05:00+00:00",
    )
    replay.observe_event(event)

    changed = _lineage_session(project, "job-patch-b")
    changed.record_started()
    _context, event = _write_run_context(
        project,
        commit=commit,
        dirty=True,
        diff_sha256="2" * 64,
        captured_at="2026-08-17T00:10:00+00:00",
    )
    changed.observe_event(event)

    with SculptorKG(kg_path) as kg:
        software = kg.find_nodes(kind="SoftwareEnvironment")
        assert len(software) == 2
        assert {node.code_diff_digest for node in software} == {
            "1" * 64,
            "2" * 64,
        }
        assert kg.count_edges(Relation.EXECUTES_IN) == 3


def test_run_context_replay_rejects_conflicting_software_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    project.mkdir()
    commit = "b" * 40
    session = _lineage_session(project, "job-conflict")
    session.record_started()

    _context, clean_event = _write_run_context(
        project, commit=commit, dirty=False,
    )
    session.observe_event(clean_event)
    _context, dirty_event = _write_run_context(
        project, commit=commit, dirty=True,
    )
    with pytest.raises(LineageConflict, match="two software contexts"):
        session.observe_event(dirty_event)

    with SculptorKG(kg_path) as kg:
        assert len(kg.find_nodes(kind="SoftwareEnvironment")) == 1
        assert kg.count_edges(Relation.EXECUTES_IN) == 1


def test_malformed_run_context_events_cannot_earn_lineage(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    project.mkdir()
    commit = "c" * 40

    _context, event = _write_run_context(
        project, commit=commit, dirty=False,
    )
    missing_path = dict(event)
    missing_path.pop("path")
    session = _lineage_session(project, "job-missing-path")
    session.record_started()
    with pytest.raises(LineageObservationError, match="lacks required"):
        session.observe_event(missing_path)

    outside = tmp_path / "outside.json"
    outside.write_text(
        (project / "reports" / "run_context.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    outside_event = {**event, "path": str(outside)}
    session = _lineage_session(project, "job-outside-path")
    session.record_started()
    with pytest.raises(LineageObservationError, match="canonical project report"):
        session.observe_event(outside_event)

    mismatched = {**event, "code_dirty": True}
    session = _lineage_session(project, "job-mismatched-summary")
    session.record_started()
    with pytest.raises(LineageObservationError, match="dirty flag differs"):
        session.observe_event(mismatched)

    _context, malformed_packages = _write_run_context(
        project, commit=commit, dirty=False, packages=["torch==2.11.0"],
    )
    session = _lineage_session(project, "job-malformed-runtime")
    session.record_started()
    with pytest.raises(LineageObservationError, match="packages is not a string map"):
        session.observe_event(malformed_packages)

    dirty_context, dirty_event = _write_run_context(
        project, commit=commit, dirty=True,
    )
    dirty_context["code_git"].pop("diff_sha256")
    (project / "reports" / "run_context.json").write_text(
        json.dumps(dirty_context, sort_keys=True), encoding="utf-8",
    )
    session = _lineage_session(project, "job-dirty-without-patch")
    session.record_started()
    with pytest.raises(LineageObservationError, match="no deterministic code diff"):
        session.observe_event(dirty_event)

    with SculptorKG(kg_path) as kg:
        assert kg.find_nodes(kind="SoftwareEnvironment") == []
        assert kg.count_edges(Relation.EXECUTES_IN) == 0
        assert all(
            run.code_commit is None
            for run in kg.find_nodes(kind="TrainingRun")
        )


def test_unverified_load_event_cannot_create_initialization_edge(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    _seed_world(project)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not-an-admitted-source")
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-rejected-observation",
        requested_initialization_mode="auto_resume",
    )
    session.record_started()

    try:
        session.observe_event({
            "type": "warm_start_loaded",
            "source": str(outside),
            "source_sha8": hashlib.sha256(outside.read_bytes()).hexdigest()[:8],
            "load_cfg_keys": ["actor"],
        })
    except LineageObservationError as exc:
        assert "outside project runs" in str(exc)
    else:  # pragma: no cover - makes the negative authority explicit
        raise AssertionError("outside checkpoint unexpectedly earned lineage")

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0


def test_no_kg_skips_every_lineage_side_effect(tmp_path: Path, monkeypatch) -> None:
    kg_path = tmp_path / "must-not-exist.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    project.mkdir()
    session = RunLineageSession(
        project_dir=project,
        project_slug="no-kg",
        run_id="job-no-kg",
        requested_initialization_mode="auto_resume",
        no_kg=True,
    )
    session.record_started()
    session.observe_event({
        "type": "warm_start_loaded",
        "source": str(tmp_path / "missing.pt"),
    })
    assert session.record_outputs() == []
    assert not kg_path.exists()


def test_run_manager_binds_session_to_worker_events_and_exit(
    tmp_path: Path, monkeypatch,
) -> None:
    from backend.services import artifact_lineage, run_manager
    from backend.services.job_manager import Job

    calls: list[object] = []

    class _SpySession:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def record_started(self) -> None:
            calls.append("started")

        def observe_event(self, event: dict) -> None:
            calls.append(("event", event["type"]))

        def record_outputs(self) -> list:
            calls.append("outputs")
            return []

    class _Stdout:
        def __init__(self):
            payload = {
                "type": "warm_start_loaded",
                "source": "/irrelevant-to-spy.pt",
                "source_sha8": "12345678",
                "load_cfg_keys": ["actor"],
            }
            self._lines = [
                (run_manager.EVENT_TAG + " " + json.dumps(payload) + "\n").encode()
            ]

        async def readline(self) -> bytes:
            await asyncio.sleep(0)
            return self._lines.pop(0) if self._lines else b""

    class _Proc:
        pid = 1234
        returncode = None

        def __init__(self):
            self.stdout = _Stdout()

        async def wait(self) -> int:
            await asyncio.sleep(0.02)
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    async def _fake_exec(*_args, **_kwargs):
        return _Proc()

    async def _wait_for_cancel(*args, **_kwargs):
        cancel = next(arg for arg in args if isinstance(arg, asyncio.Event))
        await cancel.wait()

    monkeypatch.setattr(artifact_lineage, "RunLineageSession", _SpySession)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(run_manager, "_fs_watcher", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_heartbeat", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_kill_on_cancel", _wait_for_cancel)

    project = tmp_path / "project"
    project.mkdir()
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={"behavior_goal": "lineage integration", "iterations": 1},
    )
    job = Job(
        job_id="job-lineage-integration",
        kind="sculpt_run",
        project_slug="lineage-integration",
        status="running",
    )
    result = asyncio.run(runner(job, asyncio.Event()))

    assert result["return_code"] == 0
    assert calls[0][0] == "init"
    assert calls[1] == "started"
    assert ("event", "warm_start_loaded") in calls
    assert calls[-1] == "outputs"
