"""Truthful, replay-safe lineage for imported skills and training runs.

The literature graph and runtime lineage share one store, but they do not
share an evidence standard.  These helpers add only facts proven at the
boundary named by the function: admission, prepared run context, observed
policy load, or produced checkpoint.  They deliberately do not infer that a
controller/world bundled beside a policy was executed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable, Mapping

from sculptor.kg.schema import (
    ArtifactAttestation,
    Edge,
    ModeExecutionArtifact,
    PolicyArtifact,
    ReferenceMotion,
    Relation,
    RobotEmbodiment,
    SoftwareEnvironment,
    TrainingRun,
    WorldArtifact,
    make_mode_execution_artifact_id,
    make_world_artifact_id,
)
from sculptor.kg.store import SculptorKG


class LineageConflict(ValueError):
    """A content identity was replayed with contradictory facts."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LineageConflict(
            "mode execution identity is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def build_mode_execution_artifact(
    *,
    reward_sha256: str,
    robot: str,
    clip_id: str,
    clip_sha256: str,
    graph_sha256: str,
    execution_manifest_digest: str,
    selection_digest: str,
    context_refs: dict[str, str],
) -> ModeExecutionArtifact:
    """Construct the one canonical identity for an admitted mode executor."""
    digests = {
        "reward_sha256": reward_sha256,
        "clip_sha256": clip_sha256,
        "graph_sha256": graph_sha256,
        "execution_manifest_digest": execution_manifest_digest,
        "selection_digest": selection_digest,
    }
    invalid = sorted(name for name, value in digests.items() if not _is_sha256(value))
    if invalid:
        raise LineageConflict(
            "mode execution has invalid SHA-256 field(s): " + ", ".join(invalid)
        )
    if not isinstance(robot, str) or not robot.strip():
        raise LineageConflict("mode execution requires a robot namespace")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise LineageConflict("mode execution requires a clip identity")
    if not isinstance(context_refs, dict) or not all(
        isinstance(key, str)
        and bool(key)
        and _is_sha256(value)
        for key, value in context_refs.items()
    ):
        raise LineageConflict(
            "mode execution context refs must map names to SHA-256 digests"
        )
    normalized_refs = {
        key: context_refs[key].lower() for key in sorted(context_refs)
    }
    context_refs_digest = _canonical_sha256(normalized_refs)
    identity = {
        "schema": 1,
        "reward_sha256": reward_sha256.lower(),
        "robot": robot.strip(),
        "clip_id": clip_id.strip(),
        "clip_sha256": clip_sha256.lower(),
        "graph_sha256": graph_sha256.lower(),
        "execution_manifest_digest": execution_manifest_digest.lower(),
        "selection_digest": selection_digest.lower(),
        "context_refs_digest": context_refs_digest,
        "context_refs": normalized_refs,
    }
    bundle_digest = _canonical_sha256(identity)
    return ModeExecutionArtifact(
        id=make_mode_execution_artifact_id(bundle_digest),
        bundle_digest=bundle_digest,
        reward_sha256=identity["reward_sha256"],
        robot=identity["robot"],
        clip_id=identity["clip_id"],
        clip_sha256=identity["clip_sha256"],
        graph_sha256=identity["graph_sha256"],
        execution_manifest_digest=identity["execution_manifest_digest"],
        selection_digest=identity["selection_digest"],
        context_refs_digest=context_refs_digest,
        context_refs=normalized_refs,
    )


def _same_node(existing: Any, incoming: Any) -> bool:
    if type(existing) is not type(incoming):
        return False
    if isinstance(existing, ArtifactAttestation):
        # Admission time is a local observation, not manifest content. A
        # replay may reconstruct the same attestation at another wall time.
        incoming = dataclasses.replace(incoming, admitted_at=existing.admitted_at)
    return existing == incoming


def _add_immutable(kg: SculptorKG, node: Any) -> None:
    existing = kg.get_node(node.id)
    if existing is None:
        kg.add_node(node, upsert=False)
        return
    if not _same_node(existing, node):
        raise LineageConflict(
            f"immutable KG node {node.id!r} already exists with different facts"
        )


def _add_edge_once(kg: SculptorKG, edge: Edge) -> None:
    if not kg.has_node(edge.src) or not kg.has_node(edge.dst):
        raise LineageConflict(
            f"cannot record dangling edge {edge.src!r} -> {edge.dst!r}"
        )
    existing = [
        candidate
        for candidate, other in kg.neighbors(
            edge.src, relation=edge.relation, direction="out"
        )
        if other == edge.dst
    ]
    if not existing:
        kg.add_edge(edge, upsert=False)
        return
    if len(existing) != 1 or existing[0].data != edge.data:
        raise LineageConflict(
            f"lineage edge {edge.src!r} -[{edge.relation.value}]-> "
            f"{edge.dst!r} already exists with different evidence"
        )


def _merge_run_observation(
    kg: SculptorKG,
    run: TrainingRun,
    *,
    observed_initialization_mode: str | None,
) -> TrainingRun:
    existing = kg.get_node(run.id)
    if existing is None:
        merged = dataclasses.replace(
            run, observed_initialization_mode=observed_initialization_mode
        )
        kg.add_node(merged, upsert=False)
        return merged
    if not isinstance(existing, TrainingRun):
        raise LineageConflict(f"{run.id!r} is not a TrainingRun")
    static_fields = ("project", "run_id", "requested_initialization_mode")
    if any(getattr(existing, field) != getattr(run, field) for field in static_fields):
        raise LineageConflict(f"training run {run.id!r} changed its requested facts")
    # These facts may become observable just after process creation
    # (run_context capture and the first immutable artifact-tuple pin).  They
    # can be filled once, never replaced by a conflicting later observation.
    needs_update = False
    for field in ("code_commit", "selection_digest"):
        old = getattr(existing, field)
        new = getattr(run, field)
        if old is not None and new is not None and old != new:
            raise LineageConflict(
                f"training run {run.id!r} changed its observed {field}"
            )
        if old is None and new is not None:
            setattr(existing, field, new)
            needs_update = True
    observed = existing.observed_initialization_mode
    if observed and observed_initialization_mode and observed != observed_initialization_mode:
        raise LineageConflict(
            f"training run {run.id!r} observed two initialization modes"
        )
    if observed is None and observed_initialization_mode is not None:
        existing.observed_initialization_mode = observed_initialization_mode
        needs_update = True
    if needs_update:
        kg.add_node(existing)
    return existing


def record_imported_skill(
    kg: SculptorKG | None,
    *,
    attestation: ArtifactAttestation,
    policy: PolicyArtifact | None = None,
    source_policy: PolicyArtifact | None = None,
    policy_derivation_data: dict[str, Any] | None = None,
    motion: ReferenceMotion | None = None,
    declared_target: RobotEmbodiment | None = None,
    compatible_target: RobotEmbodiment | None = None,
) -> None:
    """Record an admitted data bundle without inventing runtime usage.

    ``compatible_target`` must be supplied only after the tensor inventory and
    the full embodiment/policy contract have been validated. When an upload is
    sanitized into server-owned policy bytes, the manifest ATTESTS the distinct
    ``source_policy`` and the retained ``policy`` DERIVES_FROM it with exact
    conversion evidence. A bundled motion gets ATTESTS, never TRACKS;
    co-location is not execution.
    ``kg=None`` is the explicit `--no-kg`/test ablation and is a no-op.
    """
    if kg is None:
        return
    if compatible_target is not None and policy is None:
        raise LineageConflict("compatibility requires a policy artifact")
    evidence: dict[str, Any] | None = None
    if source_policy is not None:
        if policy is None:
            raise LineageConflict(
                "source policy conversion requires a retained policy artifact"
            )
        evidence = dict(policy_derivation_data or {})
        if (
            evidence.get("authority") != "sanitized_safetensors_conversion"
            or evidence.get("source_retained") is not False
            or evidence.get("output_retained") is not True
            or evidence.get("source_sha256") != source_policy.sha256
            or evidence.get("output_sha256") != policy.sha256
            or not _is_sha256(evidence.get("tensor_signature_sha256"))
            or source_policy.tensor_inventory_digest
            != evidence.get("tensor_signature_sha256")
            or policy.tensor_inventory_digest
            != evidence.get("tensor_signature_sha256")
            or source_policy.artifact_format != "safetensors"
            or source_policy.id == policy.id
        ):
            raise LineageConflict(
                "policy derivation requires exact safetensors conversion evidence"
            )
    _add_immutable(kg, attestation)
    attested_policy = source_policy if source_policy is not None else policy
    for artifact in (attested_policy, motion):
        if artifact is not None:
            _add_immutable(kg, artifact)
            _add_edge_once(
                kg,
                Edge(
                    attestation.id,
                    artifact.id,
                    Relation.ATTESTS,
                    data=(
                        {
                            "role": "uploaded_source_policy",
                            "retained": False,
                        }
                        if source_policy is not None and artifact is source_policy
                        else {}
                    ),
                ),
            )
    if source_policy is not None:
        assert policy is not None and evidence is not None
        _add_immutable(kg, policy)
        _add_edge_once(
            kg,
            Edge(
                policy.id,
                source_policy.id,
                Relation.DERIVED_FROM,
                data=evidence,
            ),
        )
    if declared_target is not None:
        _add_immutable(kg, declared_target)
        _add_edge_once(
            kg,
            Edge(
                attestation.id,
                declared_target.id,
                Relation.DECLARES_TARGET,
                data={"authority": "manifest_declaration"},
            ),
        )
    if compatible_target is not None:
        assert policy is not None
        _add_immutable(kg, compatible_target)
        _add_edge_once(
            kg,
            Edge(
                policy.id,
                compatible_target.id,
                Relation.COMPATIBLE_WITH,
                data={"authority": "exact_tensor_and_contract_validation"},
            ),
        )


def record_run_started(
    kg: SculptorKG | None,
    *,
    run: TrainingRun,
    effective_reference: ReferenceMotion | None = None,
    reference_edge_data: dict[str, Any] | None = None,
    active_world: WorldArtifact | None = None,
    active_world_edge_data: dict[str, Any] | None = None,
    software: SoftwareEnvironment | None = None,
    software_edge_data: dict[str, Any] | None = None,
) -> None:
    """Record the prepared context of an actual run, before weight loading."""
    if kg is None:
        return
    _merge_run_observation(kg, run, observed_initialization_mode=None)
    if effective_reference is not None:
        evidence = dict(reference_edge_data or {})
        if (
            evidence.get("verified") is not True
            or evidence.get("tier") != "D"
            or evidence.get("authority")
            != "reference_feasibility_admitted+run_started"
            or not isinstance(evidence.get("robot"), str)
            or not evidence["robot"]
            or not isinstance(evidence.get("clip_id"), str)
            or not evidence["clip_id"]
            or evidence.get("target_robot") != evidence.get("robot")
            or any(
                not _is_sha256(evidence.get(field))
                for field in (
                    "clip_sha256",
                    "rollout_sha256",
                    "certificate_sha256",
                    "execution_contract_sha256",
                    "execution_boundary_sha256",
                )
            )
            or evidence.get("clip_sha256") != effective_reference.sha256
        ):
            raise LineageConflict(
                "TRACKS requires exact target robot/clip identity and verified "
                "Tier-D clip, rollout, certificate, execution-contract, and "
                "execution-boundary evidence"
            )
        _add_immutable(kg, effective_reference)
        _add_edge_once(
            kg,
            Edge(
                run.id,
                effective_reference.id,
                Relation.TRACKS,
                data=evidence,
            ),
        )
    if active_world is not None:
        evidence = dict(active_world_edge_data or {})
        if evidence.get("validated") is not True or evidence.get("status") != "active":
            raise LineageConflict(
                "EXECUTES_IN requires the validated active world selection"
            )
        _add_immutable(kg, active_world)
        _add_edge_once(
            kg, Edge(run.id, active_world.id, Relation.EXECUTES_IN, data=evidence)
        )
    if software is not None:
        evidence = dict(software_edge_data or {})
        if (
            evidence.get("verified") is not True
            or evidence.get("authority") != "run_context_captured"
        ):
            raise LineageConflict(
                "software EXECUTES_IN requires a verified run_context_captured "
                "observation"
            )
        # A run may EXECUTE_IN both one world tuple and one software context,
        # so filter by destination kind.  Replaying the same capture is safe;
        # observing a second software identity for the same run is a conflict.
        for _edge, destination in kg.neighbors(
            run.id, relation=Relation.EXECUTES_IN, direction="out"
        ):
            existing_destination = kg.get_node(destination)
            if (
                isinstance(existing_destination, SoftwareEnvironment)
                and destination != software.id
            ):
                raise LineageConflict(
                    f"training run {run.id!r} observed two software contexts"
                )
        _add_immutable(kg, software)
        _add_edge_once(
            kg,
            Edge(
                run.id,
                software.id,
                Relation.EXECUTES_IN,
                data=evidence,
            ),
        )


def record_mode_execution_admitted(
    kg: SculptorKG | None,
    *,
    run: TrainingRun,
    artifact: ModeExecutionArtifact,
    selection_digest: str,
) -> None:
    """Record the exact mode schedule/reward proven at the pre-train boundary."""
    if kg is None:
        return
    if not _is_sha256(selection_digest):
        raise LineageConflict(
            "mode execution admission requires an immutable selection digest"
        )
    expected = build_mode_execution_artifact(
        reward_sha256=artifact.reward_sha256,
        robot=artifact.robot,
        clip_id=artifact.clip_id,
        clip_sha256=artifact.clip_sha256,
        graph_sha256=artifact.graph_sha256,
        execution_manifest_digest=artifact.execution_manifest_digest,
        selection_digest=artifact.selection_digest,
        context_refs=artifact.context_refs,
    )
    if artifact != expected:
        raise LineageConflict(
            "mode execution artifact identity disagrees with its exact facts"
        )
    if artifact.selection_digest != selection_digest:
        raise LineageConflict(
            "mode execution artifact differs from its admitted selection"
        )
    _merge_run_observation(
        kg, run, observed_initialization_mode=run.observed_initialization_mode,
    )
    expected_world_id = make_world_artifact_id(selection_digest)
    matching_world_edges = [
        edge
        for edge, destination in kg.neighbors(
            run.id, relation=Relation.EXECUTES_IN, direction="out",
        )
        if destination == expected_world_id
        and edge.data.get("validated") is True
        and edge.data.get("status") == "active"
    ]
    if len(matching_world_edges) != 1:
        raise LineageConflict(
            "mode execution selection is not a validated active world tuple "
            "for this run"
        )
    _add_immutable(kg, artifact)
    _add_edge_once(
        kg,
        Edge(
            run.id,
            artifact.id,
            Relation.USES_MODE_EXECUTION,
            data={
                "authority": "mode_execution_admitted",
                "verified": True,
                "selection_digest": selection_digest,
            },
        ),
    )


def record_policy_loaded(
    kg: SculptorKG | None,
    *,
    run: TrainingRun,
    policy: PolicyArtifact,
    transfer_mode: str,
    checkpoint_sha256: str,
    load_cfg_keys: Iterable[str],
    derived_from_policy: PolicyArtifact | None = None,
    derivation_data: Mapping[str, Any] | None = None,
) -> None:
    """Earn INITIALIZED_FROM after the runtime confirms a successful load."""
    if kg is None:
        return
    if checkpoint_sha256 != policy.sha256:
        raise LineageConflict("loaded checkpoint digest does not match policy node")
    _add_immutable(kg, policy)
    _merge_run_observation(
        kg, run, observed_initialization_mode=transfer_mode
    )
    _add_edge_once(
        kg,
        Edge(
            run.id,
            policy.id,
            Relation.INITIALIZED_FROM,
            data={
                "transfer_mode": transfer_mode,
                "checkpoint_sha256": checkpoint_sha256,
                "load_cfg_keys": sorted(str(key) for key in load_cfg_keys),
                "authority": "warm_start_loaded",
            },
        ),
    )
    if (
        derived_from_policy is not None
        and derived_from_policy.id != policy.id
    ):
        _add_immutable(kg, derived_from_policy)
        _add_edge_once(
            kg,
            Edge(
                policy.id,
                derived_from_policy.id,
                Relation.DERIVED_FROM,
                data={
                    "authority": "warm_start_loaded_derived_checkpoint",
                    **dict(derivation_data or {}),
                },
            ),
        )


def record_run_output(
    kg: SculptorKG | None,
    *,
    run: TrainingRun,
    output_policy: PolicyArtifact,
    input_policy: PolicyArtifact | None = None,
) -> None:
    """Record a checkpoint actually produced by a run and its ancestry."""
    if kg is None:
        return
    _merge_run_observation(
        kg,
        run,
        observed_initialization_mode=run.observed_initialization_mode,
    )
    _add_immutable(kg, output_policy)
    _add_edge_once(
        kg,
        Edge(
            run.id,
            output_policy.id,
            Relation.PRODUCED,
            data={"authority": "checkpoint_written"},
        ),
    )
    if input_policy is not None:
        _add_immutable(kg, input_policy)
        _add_edge_once(
            kg,
            Edge(
                output_policy.id,
                input_policy.id,
                Relation.DERIVED_FROM,
                data={"authority": "observed_initialization"},
            ),
        )
