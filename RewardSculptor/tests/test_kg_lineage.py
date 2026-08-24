from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from sculptor.kg.lineage import (
    LineageConflict,
    build_mode_execution_artifact,
    record_imported_skill,
    record_mode_execution_admitted,
    record_policy_loaded,
    record_run_output,
    record_run_started,
)
from sculptor.kg.schema import (
    ArtifactAttestation,
    PolicyArtifact,
    ReferenceMotion,
    Relation,
    RobotEmbodiment,
    TrainingRun,
    WorldArtifact,
    make_artifact_attestation_id,
    make_policy_artifact_id,
    make_reference_motion_id,
    make_robot_embodiment_id,
    make_training_run_id,
    make_world_artifact_id,
)
from sculptor.kg.store import SculptorKG


def _policy(char: str = "a", *, fmt: str = "safetensors") -> PolicyArtifact:
    digest = char * 64
    return PolicyArtifact(
        id=make_policy_artifact_id(digest),
        sha256=digest,
        artifact_format=fmt,
        size_bytes=1024,
    )


def _attestation(char: str = "b") -> ArtifactAttestation:
    digest = char * 64
    return ArtifactAttestation(
        id=make_artifact_attestation_id(digest),
        manifest_digest=digest,
        trust_status="sanitized",
        source_format="safetensors",
        declared={"robot_slug": "g1", "alias": "parkour prior"},
    )


def _robot() -> RobotEmbodiment:
    digest = "c" * 64
    return RobotEmbodiment(
        id=make_robot_embodiment_id("g1", digest),
        slug="g1",
        contract_digest=digest,
        joint_names=["left_hip_pitch_joint"],
    )


def _run(mode: str = "actor_only") -> TrainingRun:
    return TrainingRun(
        id=make_training_run_id("parkour", "job-42"),
        project="parkour",
        run_id="job-42",
        requested_initialization_mode=mode,
        code_commit="deadbeef",
        selection_digest="d" * 64,
    )


def _reference_evidence(digest: str) -> dict[str, object]:
    scope = {"simulator": "mjlab", "cadence_hz": 50.0}
    tierd_receipt = {
        "status": "tierd_verified",
        "tier": "D",
        "kinematic_only": False,
        "training_authorized": True,
        "reference_tracking_certificate_admitted": True,
        "reference_robot": "g1",
        "target_robot": "g1",
        "reference_clip_id": "cartwheel",
        "clip_sha256": digest,
        "rollout_sha256": "7" * 64,
        "certificate_sha256": "8" * 64,
        "execution_contract_sha256": "9" * 64,
        "execution_boundary_sha256": "a" * 64,
        "certification_scope": scope,
    }
    schedule = {
        "reference_robot": "g1",
        "reference_clip_id": "cartwheel",
        "reference_target_sha256": "b" * 64,
        "phase_mode": "hold",
        "phase_duration_s": 1.0,
        "n_phase_targets": 32,
        "tracking_backbone_sha256": "c" * 64,
    }

    def digest_of(value: object) -> str:
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()

    return {
        "authority": "reference_feasibility_admitted+run_started",
        "verified": True,
        "tierd_receipt": tierd_receipt,
        "tierd_receipt_sha256": digest_of(tierd_receipt),
        "tier": "D",
        "robot": "g1",
        "clip_id": "cartwheel",
        "clip_sha256": digest,
        "rollout_sha256": "7" * 64,
        "certificate_sha256": "8" * 64,
        "execution_contract_sha256": "9" * 64,
        "execution_boundary_sha256": "a" * 64,
        "target_robot": "g1",
        "certification_scope": scope,
        "certification_scope_sha256": digest_of(scope),
        "runtime_schedule_authority": "reference_runtime_schedule_admitted",
        "runtime_schedule": schedule,
        "runtime_schedule_sha256": digest_of(schedule),
        "reference_target_sha256": "b" * 64,
        "phase_mode": "hold",
        "phase_duration_s": 1.0,
        "n_phase_targets": 32,
        "tracking_backbone_sha256": "c" * 64,
    }


def _initialization_receipt(policy: PolicyArtifact) -> dict[str, object]:
    return {
        "schema": 1,
        "requested": {
            "roles": ["actor"],
            "initialization_mode": "actor_only",
        },
        "resolved": {
            "roles": ["actor"],
            "initialization_mode": "actor_only",
            "checkpoint_sha256": policy.sha256,
        },
        "observed": {
            "roles": ["actor"],
            "load_cfg_keys": ["actor"],
            "initialization_mode": "actor_only",
            "source_sha256": policy.sha256,
            "loaded_checkpoint_sha256": policy.sha256,
        },
    }


@pytest.fixture
def kg(tmp_path: Path):
    with SculptorKG(tmp_path / "lineage.db") as store:
        yield store


def test_import_is_idempotent_and_does_not_invent_runtime_usage(kg) -> None:
    policy = _policy()
    attestation = _attestation()
    robot = _robot()
    motion_digest = "e" * 64
    motion = ReferenceMotion(
        id=make_reference_motion_id(motion_digest), sha256=motion_digest
    )

    for _ in range(2):
        record_imported_skill(
            kg,
            attestation=attestation,
            policy=policy,
            motion=motion,
            declared_target=robot,
            compatible_target=robot,
        )

    assert kg.count_edges(Relation.ATTESTS) == 2
    assert kg.count_edges(Relation.DECLARES_TARGET) == 1
    assert kg.count_edges(Relation.COMPATIBLE_WITH) == 1
    assert kg.count_edges(Relation.TRACKS) == 0
    assert kg.count_edges(Relation.EXECUTES_IN) == 0


def test_conflicting_facts_cannot_overwrite_content_identity(kg) -> None:
    record_imported_skill(kg, attestation=_attestation(), policy=_policy())

    with pytest.raises(LineageConflict, match="immutable KG node"):
        record_imported_skill(
            kg, attestation=_attestation(), policy=_policy(fmt="pickle")
        )
    assert kg.get_node(_policy().id).artifact_format == "safetensors"


def test_import_distinguishes_source_weights_from_server_conversion(kg) -> None:
    signature = "9" * 64
    source = PolicyArtifact(
        id=make_policy_artifact_id("5" * 64),
        sha256="5" * 64,
        artifact_format="safetensors",
        size_bytes=2048,
        tensor_inventory_digest=signature,
    )
    converted = PolicyArtifact(
        id=make_policy_artifact_id("6" * 64),
        sha256="6" * 64,
        artifact_format="native_pt",
        size_bytes=3072,
        tensor_inventory_digest=signature,
    )
    evidence = {
        "authority": "sanitized_safetensors_conversion",
        "source_retained": False,
        "output_retained": True,
        "source_sha256": source.sha256,
        "output_sha256": converted.sha256,
        "tensor_signature_sha256": signature,
    }

    record_imported_skill(
        kg,
        attestation=_attestation(),
        policy=converted,
        source_policy=source,
        policy_derivation_data=evidence,
        compatible_target=_robot(),
    )
    record_imported_skill(
        kg,
        attestation=_attestation(),
        policy=converted,
        source_policy=source,
        policy_derivation_data=evidence,
        compatible_target=_robot(),
    )

    attested = kg.neighbors(_attestation().id, relation=Relation.ATTESTS)
    assert [(edge.data, destination) for edge, destination in attested] == [
        ({"role": "uploaded_source_policy", "retained": False}, source.id),
    ]
    derived = kg.neighbors(converted.id, relation=Relation.DERIVED_FROM)
    assert [(edge.data, destination) for edge, destination in derived] == [
        (evidence, source.id),
    ]
    compatible = kg.neighbors(
        converted.id, relation=Relation.COMPATIBLE_WITH,
    )
    assert len(compatible) == 1


def test_import_rejects_unproven_policy_conversion(kg) -> None:
    signature = "9" * 64
    source = PolicyArtifact(
        id=make_policy_artifact_id("5" * 64),
        sha256="5" * 64,
        artifact_format="safetensors",
        tensor_inventory_digest=signature,
    )
    converted = PolicyArtifact(
        id=make_policy_artifact_id("6" * 64),
        sha256="6" * 64,
        artifact_format="native_pt",
        tensor_inventory_digest=signature,
    )
    with pytest.raises(LineageConflict, match="exact safetensors conversion"):
        record_imported_skill(
            kg,
            attestation=_attestation(),
            policy=converted,
            source_policy=source,
            policy_derivation_data={"authority": "manifest_declaration"},
        )
    assert kg.get_node(_attestation().id) is None
    assert kg.count_edges(Relation.ATTESTS) == 0


def test_initialization_edge_is_earned_only_after_observed_load(kg) -> None:
    run = _run()
    policy = _policy()
    record_run_started(kg, run=run)
    assert kg.count_edges(Relation.INITIALIZED_FROM) == 0

    record_policy_loaded(
        kg,
        run=run,
        policy=policy,
        transfer_mode="actor_only",
        checkpoint_sha256=policy.sha256,
        load_cfg_keys=["actor"],
        initialization_receipt=_initialization_receipt(policy),
    )

    edges = kg.neighbors(run.id, relation=Relation.INITIALIZED_FROM)
    assert len(edges) == 1
    assert edges[0][0].data["transfer_mode"] == "actor_only"
    assert edges[0][0].data["checkpoint_sha256"] == policy.sha256
    assert edges[0][0].data["load_cfg_keys"] == ["actor"]
    assert edges[0][0].data["authority"] == (
        "starting_policy_initialization_verified"
    )
    assert edges[0][0].data["receipt"] == _initialization_receipt(policy)
    assert kg.get_node(run.id).observed_initialization_mode == "actor_only"


def test_late_observed_run_identity_can_fill_once_but_not_change(kg) -> None:
    run = dataclasses.replace(_run(), code_commit=None, selection_digest=None)
    record_run_started(kg, run=run)
    observed = dataclasses.replace(
        run, code_commit="a" * 40, selection_digest="b" * 64,
    )
    record_run_started(kg, run=observed)
    stored = kg.get_node(run.id)
    assert stored.code_commit == "a" * 40
    assert stored.selection_digest == "b" * 64

    with pytest.raises(LineageConflict, match="changed its observed"):
        record_run_started(
            kg,
            run=dataclasses.replace(observed, selection_digest="c" * 64),
        )


def test_reference_only_run_tracks_motion_without_policy_edge(kg) -> None:
    run = _run("reference_only")
    digest = "f" * 64
    motion = ReferenceMotion(
        id=make_reference_motion_id(digest), sha256=digest
    )

    record_run_started(
        kg,
        run=run,
        effective_reference=motion,
        reference_edge_data=_reference_evidence(digest),
    )

    assert kg.count_edges(Relation.TRACKS) == 1
    assert kg.count_edges(Relation.INITIALIZED_FROM) == 0


def test_reference_lineage_fails_closed_without_exact_tierd_evidence(kg) -> None:
    run = _run("reference_only")
    digest = "f" * 64
    motion = ReferenceMotion(
        id=make_reference_motion_id(digest), sha256=digest,
    )

    with pytest.raises(LineageConflict, match="TRACKS requires exact"):
        record_run_started(kg, run=run, effective_reference=motion)
    with pytest.raises(LineageConflict, match="TRACKS requires exact"):
        record_run_started(
            kg,
            run=run,
            effective_reference=motion,
            reference_edge_data={
                **_reference_evidence(digest),
                "clip_sha256": "0" * 64,
            },
        )
    with pytest.raises(LineageConflict, match="TRACKS requires exact"):
        record_run_started(
            kg,
            run=run,
            effective_reference=motion,
            reference_edge_data={
                **_reference_evidence(digest),
                "execution_boundary_sha256": None,
            },
        )
    with pytest.raises(LineageConflict, match="TRACKS requires exact"):
        record_run_started(
            kg,
            run=run,
            effective_reference=motion,
            reference_edge_data={
                **_reference_evidence(digest),
                "target_robot": "go1",
            },
        )
    assert kg.count_edges(Relation.TRACKS) == 0


def test_only_validated_active_world_gets_execution_edge(kg) -> None:
    run = _run()
    digest = "1" * 64
    world = WorldArtifact(
        id=make_world_artifact_id(digest), sha256=digest, artifact_format="tuple"
    )
    with pytest.raises(LineageConflict, match="validated active"):
        record_run_started(
            kg,
            run=run,
            active_world=world,
            active_world_edge_data={"status": "staged", "validated": True},
        )

    record_run_started(
        kg,
        run=run,
        active_world=world,
        active_world_edge_data={
            "status": "active",
            "validated": True,
            "tuple_hash": digest,
        },
    )
    assert kg.count_edges(Relation.EXECUTES_IN) == 1


def test_mode_execution_authority_is_content_addressed_and_idempotent(kg) -> None:
    run = _run()
    artifact = build_mode_execution_artifact(
        reward_sha256="1" * 64,
        robot="g1",
        clip_id="cartwheel",
        clip_sha256="2" * 64,
        graph_sha256="3" * 64,
        execution_manifest_digest="4" * 64,
        selection_digest=run.selection_digest or "",
        context_refs={"world": "5" * 64, "task": "6" * 64},
    )
    world = WorldArtifact(
        id=make_world_artifact_id(run.selection_digest or ""),
        sha256=run.selection_digest or "",
        artifact_format="tuple",
    )
    record_run_started(
        kg,
        run=run,
        active_world=world,
        active_world_edge_data={"status": "active", "validated": True},
    )
    for _ in range(2):
        record_mode_execution_admitted(
            kg,
            run=run,
            artifact=artifact,
            selection_digest=run.selection_digest or "",
        )

    assert kg.get_node(artifact.id) == artifact
    edges = kg.neighbors(run.id, relation=Relation.USES_MODE_EXECUTION)
    assert len(edges) == 1
    assert edges[0][1] == artifact.id
    assert edges[0][0].data == {
        "authority": "mode_execution_admitted",
        "verified": True,
        "selection_digest": run.selection_digest,
    }


def test_mode_execution_authority_rejects_forged_identity_and_selection(kg) -> None:
    run = _run()
    artifact = build_mode_execution_artifact(
        reward_sha256="1" * 64,
        robot="g1",
        clip_id="cartwheel",
        clip_sha256="2" * 64,
        graph_sha256="3" * 64,
        execution_manifest_digest="4" * 64,
        selection_digest=run.selection_digest or "",
        context_refs={"world": "5" * 64},
    )
    world = WorldArtifact(
        id=make_world_artifact_id(run.selection_digest or ""),
        sha256=run.selection_digest or "",
        artifact_format="tuple",
    )
    record_run_started(
        kg,
        run=run,
        active_world=world,
        active_world_edge_data={"status": "active", "validated": True},
    )
    with pytest.raises(LineageConflict, match="identity disagrees"):
        record_mode_execution_admitted(
            kg,
            run=run,
            artifact=dataclasses.replace(artifact, graph_sha256="9" * 64),
            selection_digest=run.selection_digest or "",
        )
    with pytest.raises(LineageConflict, match="admitted selection"):
        record_mode_execution_admitted(
            kg,
            run=run,
            artifact=artifact,
            selection_digest="8" * 64,
        )
    unpinned = build_mode_execution_artifact(
        reward_sha256=artifact.reward_sha256,
        robot=artifact.robot,
        clip_id=artifact.clip_id,
        clip_sha256=artifact.clip_sha256,
        graph_sha256=artifact.graph_sha256,
        execution_manifest_digest=artifact.execution_manifest_digest,
        selection_digest="8" * 64,
        context_refs=artifact.context_refs,
    )
    with pytest.raises(LineageConflict, match="validated active world tuple"):
        record_mode_execution_admitted(
            kg,
            run=run,
            artifact=unpinned,
            selection_digest="8" * 64,
        )
    assert kg.count_edges(Relation.USES_MODE_EXECUTION) == 0


def test_output_policy_has_production_and_derivation_lineage(kg) -> None:
    run = _run()
    input_policy = _policy("2")
    output_policy = _policy("3")
    record_run_started(kg, run=run)
    record_policy_loaded(
        kg,
        run=run,
        policy=input_policy,
        transfer_mode="actor_only",
        checkpoint_sha256=input_policy.sha256,
        load_cfg_keys=["actor"],
        initialization_receipt=_initialization_receipt(input_policy),
    )
    record_run_output(
        kg,
        run=kg.get_node(run.id),
        output_policy=output_policy,
        input_policy=input_policy,
    )

    assert kg.count_edges(Relation.PRODUCED) == 1
    derived = kg.neighbors(output_policy.id, relation=Relation.DERIVED_FROM)
    assert derived[0][1] == input_policy.id
    for edge in kg.all_edges():
        assert kg.has_node(edge.src)
        assert kg.has_node(edge.dst)


def test_no_kg_is_an_explicit_no_op() -> None:
    record_imported_skill(None, attestation=_attestation(), policy=_policy())
    record_run_started(None, run=_run())
    record_policy_loaded(
        None,
        run=_run(),
        policy=_policy(),
        transfer_mode="actor_only",
        checkpoint_sha256=_policy().sha256,
        load_cfg_keys=["actor"],
    )
    record_run_output(None, run=_run(), output_policy=_policy("4"))
