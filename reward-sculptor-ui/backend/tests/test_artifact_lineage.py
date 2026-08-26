from __future__ import annotations

import asyncio
import hashlib
import json
import os
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


def _lineage_session(
    project: Path,
    run_id: str,
    *,
    warm_start_policy_contract_receipt: dict | None = None,
) -> RunLineageSession:
    return RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id=run_id,
        requested_initialization_mode="auto_resume",
        warm_start_policy_contract_receipt=(
            warm_start_policy_contract_receipt
        ),
    )


def _tierd_receipt(
    robot: str, clip_id: str, clip_sha256: str,
) -> dict[str, object]:
    return {
        "status": "tierd_verified",
        "tier": "D",
        "kinematic_only": False,
        "training_authorized": True,
        "reference_tracking_certificate_admitted": True,
        "reference_robot": robot,
        "target_robot": robot,
        "reference_clip_id": clip_id,
        "clip_sha256": clip_sha256,
        "rollout_sha256": "7" * 64,
        "certificate_sha256": "8" * 64,
        "execution_contract_sha256": "9" * 64,
        "execution_boundary_sha256": "a" * 64,
        "certification_scope": {
            "simulator": "mjlab",
            "control_dt_s": 0.02,
        },
    }


def _reference_schedule(robot: str, clip_id: str) -> dict[str, object]:
    return {
        "reference_robot": robot,
        "reference_clip_id": clip_id,
        "reference_target_sha256": "b" * 64,
        "phase_mode": "hold",
        "phase_duration_s": 1.0,
        "n_phase_targets": 32,
        "tracking_backbone_sha256": "c" * 64,
    }


def _reference_output_contract(
    reference_schedule: dict[str, object],
) -> dict[str, object]:
    from sculptor.reference_clock import build_reference_clock

    return {
        "schema": 4,
        "identity": {"task_id": "g1"},
        "reference_clock": build_reference_clock(
            clip_id=str(reference_schedule["reference_clip_id"]),
            robot=str(reference_schedule["reference_robot"]),
            target_sha256=str(
                reference_schedule["reference_target_sha256"]
            ),
            phase_mode=str(reference_schedule["phase_mode"]),
            phase_duration_s=float(reference_schedule["phase_duration_s"]),
            n_phase_targets=int(reference_schedule["n_phase_targets"]),
        ),
        "joints": {"ordered_names": ["j0", "j1"]},
        "observations": {"names": ["obs"], "shape": [1]},
        "actions": {"names": ["act"], "shape": [1]},
        "timing": {"control_dt_s": 0.02},
    }


def _strict_lineage_plan(
    reference_schedule: dict[str, object],
    *,
    expected_iterations: int,
    allowed_early_stop_sources: tuple[str, ...] = (),
) -> dict[str, object]:
    from sculptor.policy_contract import contract_fingerprint

    contract = _reference_output_contract(reference_schedule)
    return {
        "expected_iterations": expected_iterations,
        "allowed_early_stop_sources": allowed_early_stop_sources,
        "expected_output_robot": str(reference_schedule["reference_robot"]),
        "expected_output_policy_contract": contract,
        "expected_output_policy_contract_sha256": contract_fingerprint(
            contract
        ),
    }


def _initialization_receipt(
    source: Path,
    *,
    source_sha256: str,
    loaded: Path | None = None,
    loaded_sha256: str | None = None,
    roles: list[str] | None = None,
    migration: dict | None = None,
    source_contract_sha256: str | None = None,
    target_contract_sha256: str | None = None,
) -> dict[str, object]:
    loaded = source if loaded is None else loaded
    loaded_sha256 = source_sha256 if loaded_sha256 is None else loaded_sha256
    roles = ["actor"] if roles is None else sorted(roles)
    mode = "actor_critic" if "critic" in roles else "actor_only"
    return {
        "schema": 1,
        "requested": {
            "roles": roles,
            "initialization_mode": mode,
        },
        "resolved": {
            "roles": roles,
            "initialization_mode": mode,
            "checkpoint_sha256": source_sha256,
            "source_policy_contract_sha256": source_contract_sha256,
            "target_policy_contract_sha256": target_contract_sha256,
        },
        "observed": {
            "source": str(source.resolve()),
            "source_sha256": source_sha256,
            "loaded_checkpoint": str(loaded.resolve()),
            "loaded_checkpoint_sha256": loaded_sha256,
            "roles": roles,
            "load_cfg_keys": roles,
            "initialization_mode": mode,
            "adapted": loaded.resolve() != source.resolve(),
            "policy_contract_migration": migration,
            "effective_policy_contract_sha256": target_contract_sha256,
        },
    }


def _write_reference_output(
    project: Path,
    *,
    iteration: int,
    checkpoint_bytes: bytes,
    selection_name: str,
    input_checkpoint: Path | None,
    reference_schedule: dict[str, object],
    policy_contract: dict[str, object] | None = None,
) -> Path:
    from sculptor.policy_contract import contract_fingerprint
    from sculptor.runtime_inputs import (
        capture_environment_artifacts,
        environment_artifacts_for_phase,
    )

    output_dir = project / "runs" / f"iter_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    contract = (
        _reference_output_contract(reference_schedule)
        if policy_contract is None else policy_contract
    )
    contract_sha = contract_fingerprint(contract)
    sidecar = Path(str(checkpoint) + ".policy_contract.json")
    sidecar.write_text(json.dumps({
        "schema": 1,
        "checkpoint_sha256": checkpoint_sha,
        "policy_contract": contract,
        "policy_contract_sha256": contract_sha,
    }, sort_keys=True), encoding="utf-8")
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    selection_path = project / "env" / selection_name
    environment = environment_artifacts_for_phase(
        capture_environment_artifacts(world_selection_path=selection_path),
        "train",
    )
    reward_sha = environment["world_selection"]["refs"]["reward"]["sha256"]
    input_sha = (
        hashlib.sha256(input_checkpoint.read_bytes()).hexdigest()
        if input_checkpoint is not None else None
    )
    runtime = {
        "schema": "reward-sculptor-runner-artifacts-v2",
        "phase": "train",
        "reward_module_sha256": reward_sha,
        "environment_artifacts": environment,
        "input_checkpoint_requested_sha256": input_sha,
        "input_checkpoint_loaded_sha256": input_sha,
        "input_checkpoint_load_completed": input_checkpoint is not None,
        "output_checkpoint_sha256": checkpoint_sha,
        "output_policy_contract_sha256": contract_sha,
        "output_policy_contract_sidecar_sha256": sidecar_sha,
    }
    (output_dir / "metrics.json").write_text(json.dumps({
        "checkpoint_path": str(checkpoint.resolve()),
        "policy_contract_sidecar": str(sidecar.resolve()),
        "runtime_artifacts": runtime,
    }, sort_keys=True), encoding="utf-8")
    return checkpoint


def _ready_reference_session(
    tmp_path: Path,
    monkeypatch,
    *,
    expected_iterations: int,
    completed_iterations: int,
    allowed_early_stop_sources: tuple[str, ...] = (),
    emit_iter_completed: bool = True,
) -> tuple[
    RunLineageSession,
    Path,
    str,
    dict[str, object],
]:
    kg_path = tmp_path / "lineage.db"
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    tuple_hash, selection_name = _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-proof-boundary",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(
            schedule,
            expected_iterations=expected_iterations,
            allowed_early_stop_sources=allowed_early_stop_sources,
        ),
    )
    session.record_started()
    _context, context_event = _write_run_context(
        project, commit="2" * 40, dirty=False,
    )
    session.observe_event(context_event)
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    monkeypatch.setattr(
        "backend.services.artifact_lineage._rederive_reference_schedule",
        lambda *_args, **_kwargs: schedule,
    )
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "reward_path": str(
                (project / "rewards" / "v0.py").resolve()
            ),
        },
    })
    for iteration in range(completed_iterations):
        session.observe_event({"type": "iter_started", "iter": iteration})
        session.observe_event({
            "type": "artifact_tuple_pinned",
            "tuple_hash": tuple_hash,
            "selection": selection_name,
            "iter": iteration,
        })
        if emit_iter_completed:
            session.observe_event({
                "type": "iter_completed",
                "iter": iteration,
            })
    return session, project, selection_name, schedule


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
    schedule = _reference_schedule(robot, clip_id)

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
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    session.record_started()
    _context, context_event = _write_run_context(
        project, commit="d" * 40, dirty=False,
    )
    session.observe_event(context_event)
    session.observe_event({"type": "iter_started", "iter": 1})
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
        "iter": 1,
    })
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    monkeypatch.setattr(
        "backend.services.artifact_lineage._rederive_reference_schedule",
        lambda *_args, **_kwargs: schedule,
    )
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "reward_path": str((project / "rewards" / "v0.py").resolve()),
        },
    })

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.TRACKS) == 1
        runs = kg.find_nodes(kind="TrainingRun")
        assert len(runs) == 1
        tracks_edge, _motion_id = kg.neighbors(
            runs[0].id, relation=Relation.TRACKS,
        )[0]
        assert tracks_edge.data["tierd_receipt"] == _tierd_receipt(
            robot, clip_id, clip_sha,
        )
        assert tracks_edge.data["runtime_schedule"] == schedule
        assert kg.count_edges(Relation.EXECUTES_IN) == 3
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0
        assert runs[0].selection_digest == tuple_hash

    load_event = {
        "type": "warm_start_loaded",
        "source": str(old_checkpoint),
        "source_sha256": old_sha,
        "source_sha8": old_sha[:8],
        "load_cfg_keys": ["critic", "actor"],
    }
    session.observe_event(load_event)  # raw worker evidence alone is not authority
    initialization_receipt = _initialization_receipt(
        old_checkpoint,
        source_sha256=old_sha,
        roles=["critic", "actor"],
    )
    session.record_verified_initialization(initialization_receipt)
    session.record_verified_initialization(initialization_receipt)
    session.observe_event({"type": "iter_completed", "iter": 1})

    _write_reference_output(
        project,
        iteration=1,
        checkpoint_bytes=b"new-policy",
        selection_name=selection_name,
        input_checkpoint=old_checkpoint,
        reference_schedule=schedule,
    )
    outputs = session.record_outputs()
    assert [item.sha256 for item in outputs] == [
        hashlib.sha256(b"new-policy").hexdigest()
    ]
    session.record_outputs()  # disk/event replay remains idempotent
    proof = session.finalize_proof()
    assert proof["strict_reference_lineage"] is True
    assert proof["tierd_receipt"] == _tierd_receipt(robot, clip_id, clip_sha)
    assert proof["reference_runtime_schedule"] == schedule
    assert proof["iterations"][0]["world_tuple_sha256"] == tuple_hash
    assert proof["iterations"][0]["input_policy_sha256"] == old_sha
    assert proof["iterations"][0]["output_policy_sha256"] == outputs[0].sha256

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 2
        assert kg.count_edges(Relation.PRODUCED) == 2
        assert kg.count_edges(Relation.DERIVED_FROM) == 1
        produced = kg.neighbors(runs[0].id, relation=Relation.PRODUCED)
        assert produced[0][1] == outputs[0].id


def test_reference_lineage_reconstructs_multi_iteration_ancestry(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    tuple_hash, selection_name = _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-multi-iteration",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=2),
    )
    session.record_started()
    _context, context_event = _write_run_context(
        project, commit="e" * 40, dirty=False,
    )
    session.observe_event(context_event)
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    monkeypatch.setattr(
        "backend.services.artifact_lineage._rederive_reference_schedule",
        lambda *_args, **_kwargs: schedule,
    )
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "reward_path": str((project / "rewards" / "v0.py").resolve()),
        },
    })
    for iteration in (0, 1):
        session.observe_event({"type": "iter_started", "iter": iteration})
        session.observe_event({
            "type": "artifact_tuple_pinned",
            "tuple_hash": tuple_hash,
            "selection": selection_name,
            "iter": iteration,
        })
        session.observe_event({
            "type": "iter_completed",
            "iter": iteration,
        })
    output_0 = _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"iteration-zero-policy",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    output_1 = _write_reference_output(
        project,
        iteration=1,
        checkpoint_bytes=b"iteration-one-policy",
        selection_name=selection_name,
        input_checkpoint=output_0,
        reference_schedule=schedule,
    )
    outputs = session.record_outputs()
    proof = session.finalize_proof()

    assert [item.sha256 for item in outputs] == [
        hashlib.sha256(output_0.read_bytes()).hexdigest(),
        hashlib.sha256(output_1.read_bytes()).hexdigest(),
    ]
    assert [item["iteration"] for item in proof["iterations"]] == [0, 1]
    assert proof["iterations"][1]["input_policy_sha256"] == (
        proof["iterations"][0]["output_policy_sha256"]
    )
    with SculptorKG(kg_path) as kg:
        assert len(kg.find_nodes(kind="TrainingIteration")) == 2
        assert kg.count_edges(Relation.HAS_ITERATION) == 2
        assert kg.count_edges(Relation.EXECUTES_IN) == 4
        assert kg.count_edges(Relation.PRODUCED) == 4
        derived = kg.neighbors(outputs[1].id, relation=Relation.DERIVED_FROM)
        assert any(destination == outputs[0].id for _edge, destination in derived)


def test_rejected_strict_event_poisons_otherwise_complete_proof(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=1,
        completed_iterations=1,
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"complete-before-conflict",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()

    conflicting = {
        **schedule,
        "reference_target_sha256": "d" * 64,
    }
    with pytest.raises(
        LineageObservationError,
        match="conflicting reference runtime schedules",
    ):
        session.observe_event({
            "type": "reference_runtime_schedule_admitted",
            "source": "sculpt_run_boundary",
            **conflicting,
        })
    with pytest.raises(
        LineageObservationError,
        match="rejected authoritative lineage evidence",
    ):
        session.finalize_proof()


def test_fresh_one_cycle_reference_plan_is_proof_ready(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=1,
        completed_iterations=1,
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"fresh-one-cycle-policy",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()

    proof = session.finalize_proof()
    assert proof["iteration_plan"] == {
        "expected_count": 1,
        "allowed_early_stop_sources": [],
        "completed_count": 1,
        "early_stop": None,
    }
    assert [item["iteration"] for item in proof["iterations"]] == [0]


def test_reference_proof_rejects_truncated_iteration_plan(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=2,
        completed_iterations=1,
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"one-of-two-iterations",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()

    with pytest.raises(
        LineageObservationError,
        match="exact launch iteration plan",
    ):
        session.finalize_proof()


def test_reference_proof_requires_iter_completed_for_each_output(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=1,
        completed_iterations=1,
        emit_iter_completed=False,
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"checkpoint-without-completion-event",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()

    with pytest.raises(
        LineageObservationError,
        match="without exact iter_completed evidence",
    ):
        session.finalize_proof()


def test_reference_proof_accepts_launch_authorized_fitness_early_stop(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=2,
        completed_iterations=1,
        allowed_early_stop_sources=("fitness", "goodhart_onset"),
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"fitness-stopped-policy",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()
    session.observe_event({
        "type": "early_stop",
        "at_iter": 0,
        "source": "fitness",
        "reason": "fitness plateau: no new best in 2 iters",
    })

    proof = session.finalize_proof()
    plan = proof["iteration_plan"]
    assert plan["expected_count"] == 2
    assert plan["completed_count"] == 1
    assert plan["early_stop"]["at_iter"] == 0
    assert plan["early_stop"]["source"] == "fitness"
    assert plan["early_stop"]["reason"] == (
        "fitness plateau: no new best in 2 iters"
    )
    assert len(plan["early_stop"]["event_sha256"]) == 64


def test_reference_proof_rejects_unattested_manual_early_stop(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=2,
        completed_iterations=1,
        allowed_early_stop_sources=("fitness", "goodhart_onset"),
    )
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"manually-stopped-policy",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    session.record_outputs()

    with pytest.raises(
        LineageObservationError,
        match="not independently authorized at launch",
    ):
        session.observe_event({
            "type": "early_stop",
            "at_iter": 0,
            "source": "user",
            "reason": "stopped by user (interactive)",
        })
    with pytest.raises(
        LineageObservationError,
        match="rejected authoritative lineage evidence",
    ):
        session.finalize_proof()


def test_reference_output_contract_must_match_launch_resolved_target(
    tmp_path: Path, monkeypatch,
) -> None:
    session, project, selection_name, schedule = _ready_reference_session(
        tmp_path,
        monkeypatch,
        expected_iterations=1,
        completed_iterations=1,
    )
    wrong_contract = _reference_output_contract(schedule)
    wrong_contract["actions"] = {
        "names": ["act"],
        "shape": [2],
    }
    _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"self-consistent-wrong-interface",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
        policy_contract=wrong_contract,
    )

    with pytest.raises(
        LineageObservationError,
        match="independently launch-resolved target interface",
    ):
        session.record_outputs()
    with SculptorKG(tmp_path / "lineage.db") as kg:
        assert kg.count_edges(Relation.PRODUCED) == 0
        assert kg.count_edges(Relation.COMPATIBLE_WITH) == 0


def test_reference_output_sidecar_tamper_never_earns_production_lineage(
    tmp_path: Path, monkeypatch,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    tuple_hash, selection_name = _seed_world(project)
    robot = "g1"
    clip_id = "lineage-motion"
    clip_sha = "f" * 64
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-sidecar-tamper",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    session.record_started()
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    session.observe_event({"type": "iter_started", "iter": 0})
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
        "iter": 0,
    })
    checkpoint = _write_reference_output(
        project,
        iteration=0,
        checkpoint_bytes=b"tamper-target",
        selection_name=selection_name,
        input_checkpoint=None,
        reference_schedule=schedule,
    )
    sidecar = Path(str(checkpoint) + ".policy_contract.json")
    sidecar.write_text(
        sidecar.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        LineageObservationError, match="sidecar bytes differ",
    ):
        session.record_outputs()
    from sculptor.policy_contract import contract_fingerprint

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["policy_contract"]["reference_clock"][
        "reference_target_sha256"
    ] = "d" * 64
    payload["policy_contract_sha256"] = contract_fingerprint(
        payload["policy_contract"]
    )
    sidecar.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    metrics_path = checkpoint.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["runtime_artifacts"]["output_policy_contract_sha256"] = (
        payload["policy_contract_sha256"]
    )
    metrics["runtime_artifacts"][
        "output_policy_contract_sidecar_sha256"
    ] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        LineageObservationError, match="clock differs from run admission",
    ):
        session.record_outputs()
    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.PRODUCED) == 0
        assert kg.count_edges(Relation.COMPATIBLE_WITH) == 0


def test_reference_completion_rejects_missing_software_and_schedule(
    tmp_path: Path, monkeypatch,
) -> None:
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-missing-proof",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    session.record_started()
    with pytest.raises(
        LineageObservationError, match="software environment",
    ):
        session.finalize_proof()

    _context, context_event = _write_run_context(
        project, commit="f" * 40, dirty=False,
    )
    session.observe_event(context_event)
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    with pytest.raises(
        LineageObservationError, match="no admitted runtime schedule",
    ):
        session.observe_event({
            "type": "run_started",
            "reference_motion": {
                "robot": robot,
                "clip_id": clip_id,
                "clip_sha256": clip_sha,
                "reward_path": str(
                    (project / "rewards" / "v0.py").resolve()
                ),
            },
        })
    with pytest.raises(
        LineageObservationError, match="rejected authoritative lineage evidence",
    ):
        session.finalize_proof()


@pytest.mark.parametrize(
    "migration",
    [
        {
            "type": "zero_initialized_event_phase_observation",
            "from_schema": 2,
            "to_schema": 3,
            "observation_term": "authored_event_phase",
            "extension_width": 3,
            "ordered_phase_ids": ["route", "jump", "hold"],
            "optimizer_resume": False,
        },
        {
            "type": "zero_initialized_reference_clock_observation",
            "from_schema": 2,
            "to_schema": 4,
            "observation_term": "reference_phase",
            "extension_width": 1,
            "reference_clock_sha256": "c" * 64,
            "optimizer_resume": False,
        },
        {
            "type": "zero_initialized_observation_extensions",
            "from_schema": 2,
            "to_schema": 4,
            "extension_width": 4,
            "extensions": [
                {
                    "type": "zero_initialized_event_phase_observation",
                    "from_schema": 2,
                    "to_schema": 3,
                    "observation_term": "authored_event_phase",
                    "extension_width": 3,
                    "ordered_phase_ids": ["route", "jump", "hold"],
                    "optimizer_resume": False,
                },
                {
                    "type": "zero_initialized_reference_clock_observation",
                    "from_schema": 3,
                    "to_schema": 4,
                    "observation_term": "reference_phase",
                    "extension_width": 1,
                    "reference_clock_sha256": "c" * 64,
                    "optimizer_resume": False,
                },
            ],
            "optimizer_resume": False,
        },
    ],
    ids=["event-phase", "reference-clock", "combined"],
)
def test_migrated_warm_start_initializes_from_loaded_bytes_and_records_source(
    tmp_path: Path, monkeypatch, migration: dict,
) -> None:
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    _seed_world(project)
    source = project / "runs" / "iter_0" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"schema-2-policy")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = project / "runs" / "iter_1" / "warm_start_event_observation.pt"
    loaded.parent.mkdir(parents=True)
    loaded.write_bytes(b"schema-3-derived-policy")
    loaded_sha = hashlib.sha256(loaded.read_bytes()).hexdigest()

    contract_receipt = {
        "source": {"contract_sha256": "a" * 64},
        "target": {"contract_sha256": "b" * 64},
        "compatibility": migration,
    }
    session = _lineage_session(
        project,
        "job-migrated-load",
        warm_start_policy_contract_receipt=contract_receipt,
    )
    session.record_started()
    session.observe_event({"type": "iter_started", "iter": 1})
    session.record_verified_initialization(_initialization_receipt(
        source,
        source_sha256=source_sha,
        loaded=loaded,
        loaded_sha256=loaded_sha,
        roles=["actor", "critic"],
        migration=migration,
        source_contract_sha256="a" * 64,
        target_contract_sha256="b" * 64,
    ))

    with SculptorKG(kg_path) as kg:
        run = kg.find_nodes(kind="TrainingRun")[0]
        policies = {
            policy.sha256: policy
            for policy in kg.find_nodes(kind="PolicyArtifact")
        }
        assert source_sha in policies
        assert loaded_sha in policies
        initialized = kg.neighbors(
            run.id, relation=Relation.INITIALIZED_FROM,
        )
        assert initialized[0][1] == policies[loaded_sha].id
        derived = kg.neighbors(
            policies[loaded_sha].id, relation=Relation.DERIVED_FROM,
        )
        edge, target_id = derived[0]
        assert target_id == policies[source_sha].id
        assert edge.data["migration"] == migration
        assert edge.data["source_policy_contract_sha256"] == "a" * 64
        assert edge.data["effective_policy_contract_sha256"] == "b" * 64


def test_migrated_warm_start_rejects_forged_migration_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("RS_KG_PATH", str(tmp_path / "lineage.db"))
    project = tmp_path / "project"
    _seed_world(project)
    source = project / "runs" / "iter_0" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-policy")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = project / "runs" / "iter_1" / "warm_start_observation.pt"
    loaded.parent.mkdir(parents=True)
    loaded.write_bytes(b"derived-policy")
    loaded_sha = hashlib.sha256(loaded.read_bytes()).hexdigest()
    admitted = {
        "type": "zero_initialized_reference_clock_observation",
        "from_schema": 2,
        "to_schema": 4,
        "observation_term": "reference_phase",
        "extension_width": 1,
        "reference_clock_sha256": "c" * 64,
        "optimizer_resume": False,
    }
    session = _lineage_session(
        project,
        "job-forged-migration",
        warm_start_policy_contract_receipt={
            "source": {"contract_sha256": "a" * 64},
            "target": {"contract_sha256": "b" * 64},
            "compatibility": admitted,
        },
    )
    session.record_started()
    session.observe_event({"type": "iter_started", "iter": 1})
    forged = {**admitted, "reference_clock_sha256": "d" * 64}
    with pytest.raises(LineageObservationError, match="exact migration"):
        session.record_verified_initialization(_initialization_receipt(
            source,
            source_sha256=source_sha,
            loaded=loaded,
            loaded_sha256=loaded_sha,
            roles=["actor", "critic"],
            migration=forged,
            source_contract_sha256="a" * 64,
            target_contract_sha256="b" * 64,
        ))


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
    schedule = _reference_schedule(robot, clip_id)

    with pytest.raises(
        LineageObservationError, match="launch-admitted Tier-D receipt",
    ):
        RunLineageSession(
            project_dir=project,
            project_slug="lineage-project",
            run_id="job-no-tierd",
            requested_initialization_mode="reference_only",
            reference_robot=robot,
            reference_clip_id=clip_id,
            reference_sha256=clip_sha,
        )

    spoofed = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-spoofed-tierd",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    spoofed.record_started()
    with pytest.raises(LineageObservationError, match="worker admission"):
        spoofed.observe_event({
            "type": "reference_feasibility_admitted",
            "source": "ui_launch",
            **_tierd_receipt(robot, clip_id, clip_sha),
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
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-mode-lineage",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    session.record_started()
    session.observe_event({"type": "iter_started", "iter": 0})
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    monkeypatch.setattr(
        "backend.services.artifact_lineage._rederive_reference_schedule",
        lambda *_args, **_kwargs: schedule,
    )
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "reward_path": event["reward_path"],
        },
    })
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
        "iter": 0,
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
        assert kg.count_edges(Relation.USES_MODE_EXECUTION) == 2

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
        assert kg.count_edges(Relation.USES_MODE_EXECUTION) == 2


def test_reference_completion_rejects_required_mode_without_admission(
    tmp_path: Path, monkeypatch,
) -> None:
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    project = tmp_path / "project"
    _seed_world(project)
    robot, clip_id, clip_sha = _seed_reference(reference_root)
    mode_event, selection_name, tuple_hash = _install_mode_reward(
        project,
        reference_root,
        robot=robot,
        clip_id=clip_id,
        clip_sha256=clip_sha,
    )
    schedule = _reference_schedule(robot, clip_id)
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-mode-missing",
        requested_initialization_mode="reference_only",
        reference_robot=robot,
        reference_clip_id=clip_id,
        reference_sha256=clip_sha,
        reference_feasibility_receipt=_tierd_receipt(
            robot, clip_id, clip_sha,
        ),
        **_strict_lineage_plan(schedule, expected_iterations=1),
    )
    session.record_started()
    _context, context_event = _write_run_context(
        project, commit="1" * 40, dirty=False,
    )
    session.observe_event(context_event)
    session.observe_event({
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        **_tierd_receipt(robot, clip_id, clip_sha),
    })
    session.observe_event({
        "type": "reference_runtime_schedule_admitted",
        "source": "sculpt_run_boundary",
        **schedule,
    })
    monkeypatch.setattr(
        "backend.services.artifact_lineage._rederive_reference_schedule",
        lambda *_args, **_kwargs: schedule,
    )
    session.observe_event({
        "type": "run_started",
        "reference_motion": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_sha256": clip_sha,
            "reward_path": mode_event["reward_path"],
        },
    })
    session.observe_event({"type": "iter_started", "iter": 0})
    session.observe_event({
        "type": "artifact_tuple_pinned",
        "tuple_hash": tuple_hash,
        "selection": selection_name,
        "iter": 0,
    })
    session.observe_event({"type": "iter_completed", "iter": 0})
    with pytest.raises(
        LineageObservationError, match="lacks its required mode executor",
    ):
        session.finalize_proof()


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

    outside_sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    session.observe_event({
        "type": "warm_start_loaded",
        "source": str(outside),
        "source_sha8": outside_sha[:8],
        "load_cfg_keys": ["actor"],
    })
    session.observe_event({"type": "iter_started", "iter": 0})
    with pytest.raises(LineageObservationError, match="outside project runs"):
        session.record_verified_initialization(_initialization_receipt(
            outside, source_sha256=outside_sha,
        ))

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0


def test_project_actor_critic_lineage_is_earned_only_after_exact_load(
    tmp_path: Path, monkeypatch,
) -> None:
    """A project checkpoint earns no edge until both requested roles load."""
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    project = tmp_path / "project"
    _seed_world(project)
    checkpoint = project / "runs" / "iter_2" / "logs" / "model_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"interrupted-project-snapshot")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    session = RunLineageSession(
        project_dir=project,
        project_slug="lineage-project",
        run_id="job-actor-only-rejected",
        requested_initialization_mode="actor_critic",
    )
    session.record_started()
    session.observe_event({"type": "iter_started", "iter": 2})

    with pytest.raises(LineageObservationError, match="expected exactly"):
        session.record_verified_initialization(_initialization_receipt(
            checkpoint,
            source_sha256=checkpoint_sha,
            roles=["actor"],
        ))

    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0

    with pytest.raises(LineageObservationError, match="expected exactly"):
        malformed = _initialization_receipt(
            checkpoint,
            source_sha256=checkpoint_sha,
            roles=["actor", "critic"],
        )
        malformed["observed"]["load_cfg_keys"] = [
            "actor", "critic", "critic",
        ]
        session.record_verified_initialization(malformed)
    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0

    session.record_verified_initialization(_initialization_receipt(
        checkpoint,
        source_sha256=checkpoint_sha,
        roles=["critic", "actor"],
    ))
    with SculptorKG(kg_path) as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 2


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


def test_backend_suite_never_resolves_the_real_knowledge_graph() -> None:
    """Canary for the autouse `_isolate_knowledge_graph` conftest fixture.

    2026-08-25 audit: two tests in this file built a `RunLineageSession`
    without `RS_KG_PATH`, so full-suite runs wrote fixture nodes — including
    a fabricated TRACKS edge with placeholder Tier-D hashes — into the real
    `~/.local/share/sculptor/kg/graph.db`. If this canary fails, backend
    tests can pollute the developer's production knowledge graph again."""
    value = os.environ.get("RS_KG_PATH")
    assert value, "autouse KG isolation fixture must set RS_KG_PATH"
    resolved = Path(value).expanduser().resolve()
    shared_default = (Path.home() / ".local" / "share" / "sculptor").resolve()
    assert not str(resolved).startswith(str(shared_default))
