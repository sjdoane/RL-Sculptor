"""Production adapters for immutable artifact/run lineage.

The core :mod:`sculptor.kg.lineage` helpers deliberately accept fully formed
facts.  This module is the narrow boundary that earns those facts from the UI
backend:

* bundle lineage is written only after the data-only importer returns an
  admitted, immutable ``SkillRecord``;
* run inputs come from the hash-verified world/reference stores;
* software identity comes from the worker's verified ``run_context.json``,
  never from the backend process's checkout;
* a mode executor is recorded only after the worker's pre-train admission is
  independently joined back to the pinned selection, reward bytes, Tier-D
  reference, derived graph, manifest, and binding;
* policy initialization is written only for a trusted worker's observed
  ``warm_start_loaded`` event, after its path and full digest re-verify; and
* output lineage is written only for checkpoint bytes created or changed
  after the worker was spawned.

Resolving a path, selecting a card in the UI, or finding an old checkpoint on
disk is intentionally insufficient evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.services.kg_store import project_kg_db_path
from sculptor.kg.lineage import (
    build_mode_execution_artifact,
    record_imported_skill,
    record_iteration_started,
    record_iteration_world,
    record_mode_execution_admitted,
    record_policy_loaded,
    record_run_output,
    record_run_started,
)
from sculptor.kg.schema import (
    ArtifactAttestation,
    PolicyArtifact,
    ReferenceMotion,
    RobotEmbodiment,
    SoftwareEnvironment,
    TrainingIteration,
    TrainingRun,
    WorldArtifact,
    make_artifact_attestation_id,
    make_policy_artifact_id,
    make_reference_motion_id,
    make_robot_embodiment_id,
    make_software_environment_id,
    make_training_iteration_id,
    make_training_run_id,
    make_world_artifact_id,
)
from sculptor.kg.store import SculptorKG
from sculptor.skill_bundle import ImportTarget
from sculptor.skill_library import SkillRecord


class LineageObservationError(ValueError):
    """A runtime event could not be tied to the selected immutable bytes."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise LineageObservationError(f"{label} is not a SHA-256 digest")
    return value.lower()


def _string_map(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise LineageObservationError(f"{label} is not a string map")
    return {str(key): str(item) for key, item in sorted(value.items())}


def _policy_format(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        return "stable_baselines3_zip"
    if path.suffix.lower() == ".pt":
        return "native_pt"
    return f"checkpoint_{path.suffix.lower().lstrip('.') or 'unknown'}"


def policy_artifact_from_checkpoint(
    path: Path,
    *,
    record: SkillRecord | None = None,
    expected_sha256: str | None = None,
) -> PolicyArtifact:
    """Re-verify checkpoint bytes and construct their intrinsic KG node."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise LineageObservationError(f"checkpoint is not a non-empty file: {path}")
    digest = _file_sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise LineageObservationError(
            "checkpoint digest changed before lineage observation: "
            f"expected {expected_sha256}, got {digest}"
        )
    use_record = record is not None and record.checkpoint_sha256 == digest
    return PolicyArtifact(
        id=make_policy_artifact_id(digest),
        sha256=digest,
        artifact_format=(
            str(record.checkpoint_format)
            if use_record and record is not None
            else _policy_format(resolved)
        ),
        size_bytes=resolved.stat().st_size,
        tensor_inventory_digest=(
            record.tensor_signature_sha256
            if use_record and record is not None
            else None
        ),
    )


def reference_motion_from_library(
    robot: str,
    clip_id: str,
    *,
    expected_sha256: str | None = None,
) -> ReferenceMotion:
    """Read the canonical registered clip and preserve only byte facts."""
    from sculptor.reference import load_clip
    from sculptor.refs import library as reference_library

    clip_path = (
        reference_library.clip_dir(str(robot), str(clip_id))
        / reference_library.CLIP_FILENAME
    ).resolve(strict=True)
    digest = _file_sha256(clip_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise LineageObservationError(
            f"reference {robot}/{clip_id} digest changed: "
            f"expected {expected_sha256}, got {digest}"
        )
    clip = load_clip(clip_path)
    return ReferenceMotion(
        id=make_reference_motion_id(digest),
        sha256=digest,
        fps=float(clip["fps"]),
        frame_count=int(clip["root_pos_z"].shape[0]),
        joint_names=[str(name) for name in (clip.get("joint_names") or [])],
    )


def _robot_from_contract(
    slug: str | None,
    contract: dict[str, Any] | None,
    digest: str | None,
) -> RobotEmbodiment | None:
    if not slug or not isinstance(contract, dict) or not digest:
        return None
    joints = contract.get("joints") or {}
    observations = contract.get("observations") or {}
    actions = contract.get("actions") or {}
    timing = contract.get("timing") or {}
    return RobotEmbodiment(
        id=make_robot_embodiment_id(str(slug), str(digest)),
        slug=str(slug),
        contract_digest=str(digest),
        joint_names=[str(name) for name in (joints.get("ordered_names") or [])],
        observation_contract=dict(observations),
        action_contract=dict(actions),
        control_dt_s=(
            float(timing["control_dt_s"])
            if timing.get("control_dt_s") is not None
            else None
        ),
    )


def _contract_slug(
    preferred: str | None, contract: dict[str, Any] | None,
) -> str | None:
    if preferred:
        return str(preferred)
    if isinstance(contract, dict):
        identity = contract.get("identity") or {}
        task_id = identity.get("task_id")
        if task_id:
            return str(task_id)
    return None


_TIERD_RECEIPT_FIELDS = (
    "status",
    "tier",
    "kinematic_only",
    "training_authorized",
    "reference_tracking_certificate_admitted",
    "reference_robot",
    "target_robot",
    "reference_clip_id",
    "clip_sha256",
    "rollout_sha256",
    "certificate_sha256",
    "execution_contract_sha256",
    "execution_boundary_sha256",
    "certification_scope",
)


def _normalized_tierd_receipt(
    value: Any, *, label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LineageObservationError(f"{label} is not an object")
    receipt = {key: value.get(key) for key in _TIERD_RECEIPT_FIELDS}
    if (
        receipt["status"] != "tierd_verified"
        or receipt["tier"] != "D"
        or receipt["kinematic_only"] is not False
        or receipt["training_authorized"] is not True
        or receipt["reference_tracking_certificate_admitted"] is not True
        or not isinstance(receipt["reference_robot"], str)
        or not receipt["reference_robot"]
        or receipt["target_robot"] != receipt["reference_robot"]
        or not isinstance(receipt["reference_clip_id"], str)
        or not receipt["reference_clip_id"]
        or not isinstance(receipt["certification_scope"], dict)
        or not receipt["certification_scope"]
    ):
        raise LineageObservationError(
            f"{label} lacks exact live Tier-D authority"
        )
    for key in (
        "clip_sha256",
        "rollout_sha256",
        "certificate_sha256",
        "execution_contract_sha256",
        "execution_boundary_sha256",
    ):
        receipt[key] = _optional_sha256(
            receipt[key], label=f"{label} {key}",
        )
        if receipt[key] is None:
            raise LineageObservationError(f"{label} lacks {key}")
    # JSON round-trip rejects opaque/mutable Python values and gives edge data
    # one deterministic structural representation.
    try:
        receipt["certification_scope"] = json.loads(json.dumps(
            receipt["certification_scope"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
    except (TypeError, ValueError) as exc:
        raise LineageObservationError(
            f"{label} certification scope is not canonical JSON"
        ) from exc
    return receipt


def _normalized_reference_schedule(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LineageObservationError("reference runtime schedule is not an object")
    if value.get("source") != "sculpt_run_boundary":
        raise LineageObservationError(
            "reference runtime schedule lacks the sculpt run-boundary authority"
        )
    robot = value.get("reference_robot")
    clip_id = value.get("reference_clip_id")
    phase_mode = value.get("phase_mode")
    phase_duration = value.get("phase_duration_s")
    target_count = value.get("n_phase_targets")
    if (
        not isinstance(robot, str)
        or not robot
        or not isinstance(clip_id, str)
        or not clip_id
        or not isinstance(phase_mode, str)
        or not phase_mode
        or isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count <= 0
        or isinstance(phase_duration, bool)
        or not isinstance(phase_duration, (int, float))
        or float(phase_duration) <= 0.0
    ):
        raise LineageObservationError(
            "reference runtime schedule has invalid robot/clip/clock facts"
        )
    target_sha = _optional_sha256(
        value.get("reference_target_sha256"),
        label="reference target digest",
    )
    backbone_sha = _optional_sha256(
        value.get("tracking_backbone_sha256"),
        label="reference tracking-backbone digest",
    )
    if target_sha is None or backbone_sha is None:
        raise LineageObservationError(
            "reference runtime schedule lacks exact target/backbone pins"
        )
    return {
        "reference_robot": robot,
        "reference_clip_id": clip_id,
        "reference_target_sha256": target_sha,
        "phase_mode": phase_mode,
        "phase_duration_s": float(phase_duration),
        "n_phase_targets": target_count,
        "tracking_backbone_sha256": backbone_sha,
    }


def _rederive_reference_schedule(
    project_dir: Path,
    *,
    robot: str,
    clip_id: str,
    reward_path: Path,
) -> dict[str, Any]:
    """Recompute the exact live reference target, clock, and backbone."""
    from sculptor.reference_run import require_exact_reference_tracking_backbone
    from sculptor.refs.track import (
        require_tierd_admission,
        require_tierd_runtime_reference,
        require_tierd_target_compatibility,
    )

    try:
        reward_path = Path(reward_path).expanduser().resolve(strict=True)
        reward_path.relative_to((Path(project_dir) / "rewards").resolve())
        reward_source = reward_path.read_text(encoding="utf-8")
        certificate = require_tierd_target_compatibility(
            require_tierd_admission(robot, clip_id),
            Path(project_dir),
            target_robot=robot,
        )
        clock = require_tierd_runtime_reference(certificate, reward_source)
        backbone = require_exact_reference_tracking_backbone(
            reward_source=reward_source,
            clip_id=clip_id,
            robot=robot,
        )
    except Exception as exc:
        raise LineageObservationError(
            f"cannot independently rederive reference runtime schedule: {exc}"
        ) from exc
    return {
        "reference_robot": robot,
        "reference_clip_id": clip_id,
        "reference_target_sha256": str(clock["reference_target_sha256"]),
        "phase_mode": str(clock["phase_mode"]),
        "phase_duration_s": (
            float(clock["phase_duration_s"])
            if clock["phase_duration_s"] is not None else None
        ),
        "n_phase_targets": int(clock["n_phase_targets"]),
        "tracking_backbone_sha256": str(backbone),
    }


def record_admitted_starting_skill(
    project_dir: Path,
    *,
    record: SkillRecord,
    receipt: dict[str, Any],
    target: ImportTarget,
) -> None:
    """Persist one successfully admitted import, replay-safely.

    Discarded world/controller bytes intentionally have no artifact nodes:
    their digests are declaration metadata on the attestation, not executable
    or retained world/controller artifacts.
    """
    manifest_digest = record.manifest_digest
    if not manifest_digest:
        raise LineageObservationError("admitted import has no manifest digest")
    declared = {
        "skill_id": record.skill_id,
        "alias": record.alias,
        "adapter_class": record.adapter_class,
        "task_id": record.task_id,
        "robot_slug": record.robot_slug,
        "initialization_modes": list(record.initialization_modes),
        "policy_roles": list(record.policy_roles),
        "reference_clip_id": record.reference_clip_id,
        "reference_robot": record.reference_robot,
        "world_bundle_sha256": record.world_bundle_sha256,
        "controller_kind": record.controller_kind,
        "controller_sha256": record.controller_sha256,
        "compatibility_contract_digest": record.compatibility_contract_digest,
        "tensor_contract_verified": bool(record.tensor_contract_verified),
        "source_weights_sha256": record.source_weights_sha256,
        "server_checkpoint_sha256": record.checkpoint_sha256,
        "tensor_signature_sha256": record.tensor_signature_sha256,
    }
    attestation = ArtifactAttestation(
        id=make_artifact_attestation_id(manifest_digest),
        manifest_digest=manifest_digest,
        trust_status=str(record.trust_status),
        source_format=str(
            record.source_format
            or ("reference_npz" if record.reference_clip_id else "unknown")
        ),
        declared=declared,
    )

    policy: PolicyArtifact | None = None
    source_policy: PolicyArtifact | None = None
    policy_derivation_data: dict[str, Any] | None = None
    if record.policy_roles:
        from sculptor.skill_library import SkillLibrary

        checkpoint = SkillLibrary().checkpoint_path_for(record)
        policy = policy_artifact_from_checkpoint(
            checkpoint, record=record, expected_sha256=record.checkpoint_sha256,
        )
        if (
            record.source_format != "safetensors"
            or record.source_weights_sha256 is None
            or record.tensor_signature_sha256 is None
        ):
            raise LineageObservationError(
                "admitted imported policy lacks exact safetensors conversion evidence"
            )
        source_sha = _optional_sha256(
            record.source_weights_sha256,
            label="admitted source weights digest",
        )
        tensor_signature = _optional_sha256(
            record.tensor_signature_sha256,
            label="admitted tensor signature digest",
        )
        if source_sha is None or tensor_signature is None:
            raise LineageObservationError(
                "admitted imported policy has incomplete source identity"
            )
        source_policy = PolicyArtifact(
            id=make_policy_artifact_id(source_sha),
            sha256=source_sha,
            artifact_format="safetensors",
            size_bytes=None,
            tensor_inventory_digest=tensor_signature,
        )
        policy_derivation_data = {
            "authority": "sanitized_safetensors_conversion",
            "source_retained": False,
            "output_retained": True,
            "source_sha256": source_sha,
            "output_sha256": policy.sha256,
            "tensor_signature_sha256": tensor_signature,
        }

    motion: ReferenceMotion | None = None
    if record.reference_clip_id and record.reference_robot:
        motion = reference_motion_from_library(
            record.reference_robot,
            record.reference_clip_id,
            expected_sha256=record.reference_sha256,
        )

    declared_target = _robot_from_contract(
        _contract_slug(record.robot_slug, record.compatibility_contract),
        record.compatibility_contract,
        record.compatibility_contract_digest,
    )
    allowed = set(
        (receipt.get("compatibility") or {}).get(
            "allowed_initialization_modes", []
        )
    )
    compatible_target: RobotEmbodiment | None = None
    if policy is not None and allowed.intersection({"actor_only", "actor_critic"}):
        from sculptor.policy_contract import contract_fingerprint

        if target.compatibility_contract is not None:
            target_digest = contract_fingerprint(target.compatibility_contract)
            compatible_target = _robot_from_contract(
                _contract_slug(
                    target.robot_slug, target.compatibility_contract,
                ),
                target.compatibility_contract,
                target_digest,
            )

    with SculptorKG(project_kg_db_path(Path(project_dir))) as kg:
        with kg.transaction():
            record_imported_skill(
                kg,
                attestation=attestation,
                policy=policy,
                source_policy=source_policy,
                policy_derivation_data=policy_derivation_data,
                motion=motion,
                declared_target=declared_target,
                compatible_target=compatible_target,
            )


def _checkpoint_snapshot(project_dir: Path) -> dict[Path, str]:
    snapshot: dict[Path, str] = {}
    runs_dir = Path(project_dir) / "runs"
    if not runs_dir.is_dir():
        return snapshot
    for iteration_dir in runs_dir.glob("iter_[0-9]*"):
        if not iteration_dir.is_dir():
            continue
        for name in ("checkpoint.pt", "checkpoint.zip"):
            candidate = iteration_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                resolved = candidate.resolve()
                snapshot[resolved] = _file_sha256(resolved)
    return snapshot


@dataclass(frozen=True)
class _VerifiedOutput:
    policy: PolicyArtifact
    iteration_index: int
    evidence: dict[str, Any]
    input_checkpoint_sha256: str | None
    world_tuple_sha256: str
    compatible_target: RobotEmbodiment


def _checkpoint_iteration_index(path: Path, project_dir: Path) -> int:
    try:
        relative = Path(path).resolve().relative_to(
            (Path(project_dir) / "runs").resolve()
        )
    except ValueError as exc:
        raise LineageObservationError(
            "output checkpoint is outside the project runs directory"
        ) from exc
    if (
        len(relative.parts) != 2
        or re.fullmatch(r"iter_[0-9]+", relative.parts[0]) is None
        or relative.parts[1] not in {"checkpoint.pt", "checkpoint.zip"}
    ):
        raise LineageObservationError(
            "output checkpoint is not a canonical iteration checkpoint"
        )
    return int(relative.parts[0].split("_", 1)[1])


def _verified_reference_output(
    project_dir: Path,
    checkpoint_path: Path,
    *,
    robot: str,
    expected_schedule: dict[str, Any],
    expected_target_contract: dict[str, Any],
    expected_target_contract_sha256: str,
    expected_target_robot: str,
) -> _VerifiedOutput:
    """Bind an output to runner metrics and its exact policy-contract sidecar."""
    from sculptor.policy_contract import contract_fingerprint
    from sculptor.reference_clock import (
        build_reference_clock,
        validate_reference_clock,
    )
    from sculptor.runtime_inputs import validate_environment_artifacts

    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    iteration_index = _checkpoint_iteration_index(checkpoint_path, project_dir)
    policy = policy_artifact_from_checkpoint(checkpoint_path)
    metrics_path = checkpoint_path.parent / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LineageObservationError(
            f"reference output has no valid runner metrics receipt: {exc}"
        ) from exc
    if not isinstance(metrics, dict):
        raise LineageObservationError("runner metrics receipt is not an object")
    try:
        observed_checkpoint_path = Path(
            str(metrics["checkpoint_path"])
        ).expanduser().resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise LineageObservationError(
            "runner metrics has no resolvable checkpoint path"
        ) from exc
    if observed_checkpoint_path != checkpoint_path:
        raise LineageObservationError(
            "runner metrics identifies a different output checkpoint"
        )
    runtime = metrics.get("runtime_artifacts")
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema") != "reward-sculptor-runner-artifacts-v2"
        or runtime.get("phase") != "train"
        or runtime.get("output_checkpoint_sha256") != policy.sha256
    ):
        raise LineageObservationError(
            "reference output differs from its exact runner runtime receipt"
        )
    environment = runtime.get("environment_artifacts")
    issues = validate_environment_artifacts(environment, phase="train")
    if issues:
        raise LineageObservationError(
            "runner output environment receipt is invalid: " + "; ".join(issues)
        )
    world_selection = environment.get("world_selection")
    if (
        not isinstance(world_selection, dict)
        or world_selection.get("present") is not True
    ):
        raise LineageObservationError(
            "reference output has no exact authored-world runtime receipt"
        )
    world_tuple = _optional_sha256(
        world_selection.get("tuple_hash"),
        label="output world tuple digest",
    )
    if world_tuple is None:
        raise LineageObservationError(
            "reference output authored-world receipt lacks a tuple digest"
        )
    world_refs = world_selection.get("refs")
    reward_ref = (
        world_refs.get("reward") if isinstance(world_refs, dict) else None
    )
    reward_sha = _optional_sha256(
        runtime.get("reward_module_sha256"),
        label="output reward-module digest",
    )
    if (
        reward_sha is None
        or not isinstance(reward_ref, dict)
        or reward_ref.get("sha256") != reward_sha
    ):
        raise LineageObservationError(
            "reference output reward bytes differ from its authored world"
        )

    sidecar_expected = Path(str(checkpoint_path) + ".policy_contract.json")
    raw_sidecar_path = metrics.get("policy_contract_sidecar")
    if not isinstance(raw_sidecar_path, str) or not raw_sidecar_path:
        raise LineageObservationError(
            "reference output has no policy-contract sidecar path"
        )
    try:
        sidecar_path = Path(raw_sidecar_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise LineageObservationError(
            "reference output policy-contract sidecar is unavailable"
        ) from exc
    if sidecar_path != sidecar_expected.resolve():
        raise LineageObservationError(
            "reference output policy-contract sidecar is not checkpoint-scoped"
        )
    sidecar_sha = _file_sha256(sidecar_path)
    if runtime.get("output_policy_contract_sidecar_sha256") != sidecar_sha:
        raise LineageObservationError(
            "reference output policy-contract sidecar bytes differ from receipt"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LineageObservationError(
            "reference output policy-contract sidecar is not valid JSON"
        ) from exc
    if not isinstance(sidecar, dict) or set(sidecar) != {
        "schema", "checkpoint_sha256", "policy_contract",
        "policy_contract_sha256",
    }:
        raise LineageObservationError(
            "reference output policy-contract sidecar is non-canonical"
        )
    contract = sidecar.get("policy_contract")
    contract_sha = _optional_sha256(
        sidecar.get("policy_contract_sha256"),
        label="output policy contract digest",
    )
    if (
        sidecar.get("schema") != 1
        or sidecar.get("checkpoint_sha256") != policy.sha256
        or not isinstance(contract, dict)
        or contract_sha is None
        or contract_fingerprint(contract) != contract_sha
        or runtime.get("output_policy_contract_sha256") != contract_sha
    ):
        raise LineageObservationError(
            "reference output policy contract disagrees with checkpoint or receipt"
        )
    try:
        observed_clock = validate_reference_clock(contract["reference_clock"])
        expected_clock = build_reference_clock(
            clip_id=str(expected_schedule["reference_clip_id"]),
            robot=str(expected_schedule["reference_robot"]),
            target_sha256=str(expected_schedule["reference_target_sha256"]),
            phase_mode=str(expected_schedule["phase_mode"]),
            phase_duration_s=float(expected_schedule["phase_duration_s"]),
            n_phase_targets=int(expected_schedule["n_phase_targets"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageObservationError(
            "reference output policy contract has no valid admitted clock"
        ) from exc
    if observed_clock != expected_clock:
        raise LineageObservationError(
            "reference output policy contract clock differs from run admission"
        )
    if (
        expected_target_robot != robot
        or contract != expected_target_contract
        or contract_sha != expected_target_contract_sha256
    ):
        raise LineageObservationError(
            "reference output policy contract differs from the independently "
            "launch-resolved target interface"
        )
    compatible_target = _robot_from_contract(
        expected_target_robot,
        expected_target_contract,
        expected_target_contract_sha256,
    )
    if compatible_target is None:
        raise LineageObservationError(
            "reference output policy contract has no exact embodiment identity"
        )

    load_completed = runtime.get("input_checkpoint_load_completed")
    input_sha = runtime.get("input_checkpoint_loaded_sha256")
    requested_input_sha = runtime.get("input_checkpoint_requested_sha256")
    if not isinstance(load_completed, bool):
        raise LineageObservationError(
            "runner output has no definitive input-checkpoint load fact"
        )
    if load_completed:
        input_sha = _optional_sha256(
            input_sha, label="output input-checkpoint digest",
        )
        requested_input_sha = _optional_sha256(
            requested_input_sha,
            label="output requested input-checkpoint digest",
        )
        if input_sha is None or requested_input_sha != input_sha:
            raise LineageObservationError(
                "runner input request/load receipts do not identify exact bytes"
            )
    elif input_sha is not None or requested_input_sha is not None:
        raise LineageObservationError(
            "runner reports input bytes without a completed checkpoint load"
        )
    evidence = {
        "iteration": iteration_index,
        "runtime_artifacts_sha256": _canonical_sha256(runtime),
        "reward_module_sha256": reward_sha,
        "input_checkpoint_requested_sha256": requested_input_sha,
        "input_checkpoint_loaded_sha256": input_sha,
        "input_checkpoint_load_completed": load_completed,
        "output_checkpoint_sha256": policy.sha256,
        "output_policy_contract_sha256": contract_sha,
        "output_policy_contract_sidecar_sha256": sidecar_sha,
        "world_tuple_sha256": world_tuple,
    }
    return _VerifiedOutput(
        policy=policy,
        iteration_index=iteration_index,
        evidence=evidence,
        input_checkpoint_sha256=input_sha,
        world_tuple_sha256=world_tuple,
        compatible_target=compatible_target,
    )


def _world_from_pinned_event(
    project_dir: Path, event: dict[str, Any],
) -> tuple[WorldArtifact, dict[str, Any]]:
    from sculptor.world.artifacts import WorldArtifactStore

    store = WorldArtifactStore(project_dir)
    selection_name = event.get("selection")
    expected_tuple = event.get("tuple_hash")
    iteration_index = event.get("iter")
    if not isinstance(selection_name, str) or not selection_name:
        raise LineageObservationError(
            "artifact_tuple_pinned has no immutable selection filename"
        )
    if not isinstance(expected_tuple, str) or not expected_tuple:
        raise LineageObservationError(
            "artifact_tuple_pinned has no tuple hash"
        )
    if (
        isinstance(iteration_index, bool)
        or not isinstance(iteration_index, int)
        or iteration_index < 0
    ):
        raise LineageObservationError(
            "artifact_tuple_pinned has no non-negative iteration index"
        )
    selection_path = store.env_dir / selection_name
    selection = store.read_selection(selection_path)
    if selection is None or selection.tuple_hash != expected_tuple:
        raise LineageObservationError(
            "worker artifact tuple does not match the immutable selection"
        )
    node = WorldArtifact(
        id=make_world_artifact_id(selection.tuple_hash),
        sha256=selection.tuple_hash,
        artifact_format="reward-sculptor-artifact-tuple",
    )
    reward_ref = selection.refs.get("reward")
    if reward_ref is None:
        raise LineageObservationError(
            "worker artifact tuple has no immutable reward reference"
        )
    try:
        reward_source = store.resolve_ref(reward_ref).read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise LineageObservationError(
            "worker artifact tuple reward cannot be re-read"
        ) from exc
    edge_data = {
        "status": "active",
        "validated": True,
        "authority": "artifact_tuple_pinned",
        "tuple_hash": selection.tuple_hash,
        "iteration": iteration_index,
        "mode_execution_required": all(
            marker in reward_source for marker in ("MODE_ORDER", "MODE_WINDOWS_S")
        ),
        "selection_version": selection.selection_version,
        "evaluation_lineage": selection.evaluation_lineage,
        "refs": {
            key: {"version": ref.version, "sha256": ref.sha256}
            for key, ref in sorted(selection.refs.items())
        },
    }
    return node, edge_data


def _mode_execution_from_admitted_event(
    project_dir: Path,
    event: dict[str, Any],
) -> tuple[Any, str]:
    """Re-verify one worker mode admission from canonical project bytes."""
    if event.get("source") != "sculpt_run_worker":
        raise LineageObservationError(
            "mode execution lineage requires the worker admission boundary"
        )
    project_dir = Path(project_dir).resolve()
    raw_reward_path = event.get("reward_path")
    if not isinstance(raw_reward_path, str) or not raw_reward_path:
        raise LineageObservationError(
            "mode_execution_admitted has no reward path"
        )
    try:
        reward_path = Path(raw_reward_path).expanduser().resolve(strict=True)
        reward_path.relative_to((project_dir / "rewards").resolve())
    except (OSError, ValueError) as exc:
        raise LineageObservationError(
            "mode execution reward is outside the project reward store"
        ) from exc
    if not reward_path.is_file():
        raise LineageObservationError("mode execution reward is not a file")
    if re.fullmatch(r"v[0-9]+\.py", reward_path.name) is None:
        raise LineageObservationError(
            "mode execution reward is not an immutable v<n>.py version"
        )
    reward_sha256 = _file_sha256(reward_path)
    event_reward_sha256 = _optional_sha256(
        event.get("reward_sha256"), label="mode reward digest",
    )
    if event_reward_sha256 != reward_sha256:
        raise LineageObservationError(
            "mode execution reward digest differs from reward bytes"
        )

    selection_name = event.get("selection")
    tuple_hash = _optional_sha256(
        event.get("tuple_hash"), label="mode execution tuple digest",
    )
    if (
        not isinstance(selection_name, str)
        or not selection_name
        or Path(selection_name).name != selection_name
        or re.fullmatch(r"selection_v[0-9]+\.json", selection_name) is None
        or tuple_hash is None
    ):
        raise LineageObservationError(
            "mode execution admission has no immutable selection identity"
        )
    from sculptor.world.artifacts import WorldArtifactStore

    store = WorldArtifactStore(project_dir)
    selection = store.read_selection(store.env_dir / selection_name)
    if selection is None or selection.tuple_hash != tuple_hash:
        raise LineageObservationError(
            "mode execution selection differs from immutable project bytes"
        )
    selected_reward = selection.refs.get("reward")
    if selected_reward is None:
        raise LineageObservationError(
            "mode execution selection has no reward ref"
        )
    selected_reward_path = Path(selected_reward.path)
    if not selected_reward_path.is_absolute():
        selected_reward_path = project_dir / selected_reward_path
    if (
        selected_reward_path.resolve() != reward_path
        or selected_reward.sha256 != reward_sha256
    ):
        raise LineageObservationError(
            "mode execution reward differs from the pinned selection"
        )

    try:
        source = reward_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LineageObservationError(
            "mode execution reward is not readable UTF-8 source"
        ) from exc
    from sculptor.mode_rewards import (
        MODE_BINDING_CONTEXT_REFS,
        mode_execution_manifest_digest,
        mode_reward_binding_errors,
        reward_spec_from_source,
    )
    from sculptor.modes import mode_graph_sha256, modes_from_composition
    from sculptor.reference_run import load_exact_reference_motion

    try:
        spec = reward_spec_from_source(source)
    except Exception as exc:  # noqa: BLE001 - normalize core parser errors
        raise LineageObservationError(
            "mode execution reward has no valid data-only REWARD_SPEC"
        ) from exc
    manifest = spec.get("mode_execution_manifest")
    binding = spec.get("mode_binding")
    if not isinstance(manifest, dict) or not isinstance(binding, dict):
        raise LineageObservationError(
            "mode execution reward has no exact manifest and binding"
        )
    robot = event.get("robot")
    clip_id = event.get("clip_id")
    if not all(isinstance(value, str) and value for value in (robot, clip_id)):
        raise LineageObservationError(
            "mode execution admission has no robot/clip identity"
        )
    assert isinstance(robot, str) and isinstance(clip_id, str)
    if (
        binding.get("robot") != robot
        or binding.get("clip_id") != clip_id
        or spec.get("reference_clip_id") != clip_id
        or str(spec.get("reference_robot") or binding.get("robot") or "")
        != robot
    ):
        raise LineageObservationError(
            "mode execution event disagrees with reward reference identity"
        )
    try:
        clip, _provenance, clip_sha256 = load_exact_reference_motion(
            clip_id=clip_id, robot=robot,
        )
        graph = modes_from_composition(clip, clip_id=clip_id)
        graph_sha256 = mode_graph_sha256(graph)
        manifest_digest = mode_execution_manifest_digest(manifest)
    except Exception as exc:  # noqa: BLE001 - exact clip/graph admission
        raise LineageObservationError(
            "mode execution reference/graph cannot be re-derived"
        ) from exc
    event_clip_sha = _optional_sha256(
        event.get("clip_sha256"), label="mode reference clip digest",
    )
    event_graph_sha = _optional_sha256(
        event.get("graph_sha256"), label="mode graph digest",
    )
    event_manifest_digest = _optional_sha256(
        event.get("execution_manifest_digest"),
        label="mode execution manifest digest",
    )
    if (
        event_clip_sha != clip_sha256
        or event_graph_sha != graph_sha256
        or event_manifest_digest != manifest_digest
    ):
        raise LineageObservationError(
            "mode execution event contains stale clip/graph/manifest facts"
        )
    context_refs = {
        kind: ref.sha256
        for kind, ref in selection.refs.items()
        if kind in MODE_BINDING_CONTEXT_REFS
    }
    if event.get("context_refs") != context_refs:
        raise LineageObservationError(
            "mode execution context refs differ from the pinned selection"
        )
    errors = mode_reward_binding_errors(
        spec,
        clip_id=clip_id,
        robot=robot,
        clip_sha256=clip_sha256,
        context_refs=context_refs,
        graph_sha256=graph_sha256,
        graph=graph,
        reward_source=source,
    )
    if errors:
        raise LineageObservationError(
            "mode execution reward binding is stale: " + "; ".join(errors)
        )
    try:
        artifact = build_mode_execution_artifact(
            reward_sha256=reward_sha256,
            robot=robot,
            clip_id=clip_id,
            clip_sha256=clip_sha256,
            graph_sha256=graph_sha256,
            execution_manifest_digest=manifest_digest,
            selection_digest=tuple_hash,
            context_refs=context_refs,
        )
    except ValueError as exc:
        raise LineageObservationError(str(exc)) from exc
    return artifact, tuple_hash


def _software_from_run_context_event(
    project_dir: Path,
    event: dict[str, Any],
) -> tuple[SoftwareEnvironment, dict[str, Any], str | None]:
    """Verify a worker-owned run-context capture and derive lineage facts.

    The event's summary is not accepted on trust: the path is confined to the
    project's canonical report, the exact bytes are hashed, and every summary
    field is checked against the parsed document. Dirty captures require the
    deterministic worktree digest emitted by the worker. The report digest is
    edge evidence, never software identity, because it includes captured_at.
    """
    required_event_fields = {
        "path", "code_sha", "code_dirty", "config_sha256", "base_seed",
    }
    missing = sorted(required_event_fields - event.keys())
    if missing:
        raise LineageObservationError(
            "run_context_captured lacks required field(s): " + ", ".join(missing)
        )

    raw_path = event.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise LineageObservationError("run_context_captured has no context path")
    try:
        context_path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise LineageObservationError(
            "run_context_captured context file is unavailable"
        ) from exc
    expected_path = (Path(project_dir) / "reports" / "run_context.json").resolve()
    if context_path != expected_path or not context_path.is_file():
        raise LineageObservationError(
            "run_context_captured path is outside the canonical project report"
        )

    try:
        source = context_path.read_bytes()
        context = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineageObservationError(
            "run_context_captured report is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(context, dict):
        raise LineageObservationError("run context root is not an object")
    source_sha256 = hashlib.sha256(source).hexdigest()

    schema = context.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
        raise LineageObservationError("run context has no valid schema version")
    code_git = context.get("code_git")
    if not isinstance(code_git, dict):
        raise LineageObservationError("run context code_git is not an object")
    available = code_git.get("available")
    if not isinstance(available, bool):
        raise LineageObservationError(
            "run context code_git.available is not boolean"
        )

    code_commit: str | None = None
    code_dirty: bool | None = None
    if available:
        raw_commit = code_git.get("sha")
        if (
            not isinstance(raw_commit, str)
            or _GIT_COMMIT_RE.fullmatch(raw_commit.lower()) is None
        ):
            raise LineageObservationError(
                "run context has no full hexadecimal code commit"
            )
        code_commit = raw_commit.lower()
        raw_dirty = code_git.get("dirty")
        if not isinstance(raw_dirty, bool):
            raise LineageObservationError(
                "run context has no definitive clean/dirty observation"
            )
        code_dirty = raw_dirty
    elif code_git.get("sha") is not None or code_git.get("dirty") is not None:
        raise LineageObservationError(
            "unavailable code_git contains contradictory commit facts"
        )

    if event.get("code_sha") != code_commit:
        raise LineageObservationError(
            "run_context_captured code commit differs from report"
        )
    if event.get("code_dirty") is not code_dirty:
        raise LineageObservationError(
            "run_context_captured dirty flag differs from report"
        )

    config = context.get("config")
    if not isinstance(config, dict):
        raise LineageObservationError("run context config is not an object")
    config_sha256 = _optional_sha256(
        config.get("sha256"), label="run context config.sha256",
    )
    if event.get("config_sha256") != config_sha256:
        raise LineageObservationError(
            "run_context_captured config digest differs from report"
        )
    seeds = context.get("seeds")
    if not isinstance(seeds, dict):
        raise LineageObservationError("run context seeds is not an object")
    base_seed = seeds.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise LineageObservationError("run context base seed is not an integer")
    if event.get("base_seed") != base_seed:
        raise LineageObservationError(
            "run_context_captured base seed differs from report"
        )

    packages = _string_map(context.get("packages"), label="run context packages")
    python = context.get("python")
    environment = context.get("env")
    prompts = context.get("prompts")
    llm = context.get("llm")
    for label, value in (
        ("python", python),
        ("env", environment),
        ("prompts", prompts),
        ("llm", llm),
    ):
        if not isinstance(value, dict):
            raise LineageObservationError(
                f"run context {label} is not an object"
            )

    tree_digest = _optional_sha256(
        code_git.get("tree_sha256"), label="run context code tree digest",
    )
    diff_digest = _optional_sha256(
        code_git.get("diff_sha256"), label="run context code diff digest",
    )
    if code_dirty is True and diff_digest is None:
        raise LineageObservationError(
            "dirty run context has no deterministic code diff digest"
        )
    if code_dirty is not True and diff_digest is not None:
        raise LineageObservationError(
            "non-dirty run context contains contradictory code diff evidence"
        )
    runtime = {
        "python": dict(python),
        "environment": dict(environment),
        "prompts": dict(prompts),
        "llm": dict(llm),
    }
    identity = {
        "capture_schema": schema,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "code_tree_digest": tree_digest,
        "code_diff_digest": diff_digest,
        "versions": packages,
        "runtime": runtime,
    }
    lock_digest = _canonical_sha256(identity)
    software = SoftwareEnvironment(
        id=make_software_environment_id(lock_digest),
        lock_digest=lock_digest,
        versions=packages,
        capture_schema=schema,
        captured_source_sha256=None,
        code_commit=code_commit,
        code_dirty=code_dirty,
        code_tree_digest=tree_digest,
        code_diff_digest=diff_digest,
        runtime=runtime,
    )
    edge_data = {
        "authority": "run_context_captured",
        "verified": True,
        "captured_source_sha256": source_sha256,
        "capture_schema": schema,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "code_tree_digest": tree_digest,
        "code_diff_digest": diff_digest,
        "config_sha256": config_sha256,
        "base_seed": base_seed,
        "software_identity_digest": lock_digest,
    }
    return software, edge_data, code_commit


@dataclass
class RunLineageSession:
    """Evidence accumulator for one actual UI-launched worker process."""

    project_dir: Path
    project_slug: str
    run_id: str
    requested_initialization_mode: str
    no_kg: bool = False
    reference_robot: str | None = None
    reference_clip_id: str | None = None
    reference_sha256: str | None = None
    reference_feasibility_receipt: dict[str, Any] | None = None
    starting_skill_record: SkillRecord | None = None
    warm_start_policy_contract_receipt: dict[str, Any] | None = None
    expected_iterations: int | None = None
    allowed_early_stop_sources: tuple[str, ...] = ()
    expected_output_robot: str | None = None
    expected_output_policy_contract: dict[str, Any] | None = None
    expected_output_policy_contract_sha256: str | None = None
    _baseline: dict[Path, str] = field(init=False, repr=False)
    _run: TrainingRun | None = field(default=None, init=False, repr=False)
    _input_policy: PolicyArtifact | None = field(
        default=None, init=False, repr=False,
    )
    _reference_feasibility: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _reference_confirmed: bool = field(default=False, init=False, repr=False)
    _reference_runtime_schedule: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _software_confirmed: bool = field(default=False, init=False, repr=False)
    _software_id: str | None = field(default=None, init=False, repr=False)
    _reference_id: str | None = field(default=None, init=False, repr=False)
    _iterations: dict[int, TrainingIteration] = field(
        default_factory=dict, init=False, repr=False,
    )
    _current_iteration: int | None = field(default=None, init=False, repr=False)
    _world_by_iteration: dict[int, str] = field(
        default_factory=dict, init=False, repr=False,
    )
    _mode_required_iterations: set[int] = field(
        default_factory=set, init=False, repr=False,
    )
    _mode_by_iteration: dict[int, str] = field(
        default_factory=dict, init=False, repr=False,
    )
    _input_policy_by_iteration: dict[int, PolicyArtifact] = field(
        default_factory=dict, init=False, repr=False,
    )
    _initialization_receipt_by_iteration: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _output_by_iteration: dict[int, PolicyArtifact] = field(
        default_factory=dict, init=False, repr=False,
    )
    _output_contract_by_iteration: dict[int, str] = field(
        default_factory=dict, init=False, repr=False,
    )
    _output_evidence_by_iteration: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _latest_selection_digest: str | None = field(
        default=None, init=False, repr=False,
    )
    _completed_event_by_iteration: dict[int, str] = field(
        default_factory=dict, init=False, repr=False,
    )
    _early_stop_evidence: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _fatal_observation_errors: list[str] = field(
        default_factory=list, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        if self.expected_iterations is not None and (
            isinstance(self.expected_iterations, bool)
            or not isinstance(self.expected_iterations, int)
            or self.expected_iterations <= 0
        ):
            raise LineageObservationError(
                "expected iteration count must be a positive integer"
            )
        allowed_sources = tuple(sorted(set(self.allowed_early_stop_sources)))
        if any(
            not isinstance(source, str)
            or source not in {"fitness", "goodhart_onset"}
            for source in allowed_sources
        ):
            raise LineageObservationError(
                "early-stop authority is not independently launch-attested"
            )
        self.allowed_early_stop_sources = allowed_sources
        reference_identity = (
            self.reference_robot,
            self.reference_clip_id,
            self.reference_sha256,
        )
        if any(value is not None for value in reference_identity) and not all(
            isinstance(value, str) and value for value in reference_identity
        ):
            raise LineageObservationError(
                "reference-guided lineage requires an exact robot, clip, and digest"
            )
        if self.reference_feasibility_receipt is not None:
            self.reference_feasibility_receipt = _normalized_tierd_receipt(
                self.reference_feasibility_receipt,
                label="launch Tier-D receipt",
            )
            if (
                self.reference_feasibility_receipt["reference_robot"]
                != self.reference_robot
                or self.reference_feasibility_receipt["reference_clip_id"]
                != self.reference_clip_id
                or self.reference_feasibility_receipt["clip_sha256"]
                != self.reference_sha256
            ):
                raise LineageObservationError(
                    "launch Tier-D receipt differs from the selected reference"
                )
        elif self.reference_robot is not None:
            raise LineageObservationError(
                "reference-guided lineage requires the launch-admitted Tier-D receipt"
            )
        if self.reference_robot is not None:
            from sculptor.policy_contract import contract_fingerprint

            if self.expected_iterations is None:
                raise LineageObservationError(
                    "reference-guided lineage requires an expected iteration plan"
                )
            if (
                self.expected_output_robot != self.reference_robot
                or not isinstance(self.expected_output_policy_contract, dict)
            ):
                raise LineageObservationError(
                    "reference-guided lineage requires the independently "
                    "launch-resolved target robot and policy contract"
                )
            expected_contract_sha = _optional_sha256(
                self.expected_output_policy_contract_sha256,
                label="launch-resolved output policy contract digest",
            )
            if (
                expected_contract_sha is None
                or contract_fingerprint(
                    self.expected_output_policy_contract
                ) != expected_contract_sha
            ):
                raise LineageObservationError(
                    "launch-resolved output policy contract fingerprint is invalid"
                )
            try:
                self.expected_output_policy_contract = json.loads(json.dumps(
                    self.expected_output_policy_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ))
            except (TypeError, ValueError) as exc:
                raise LineageObservationError(
                    "launch-resolved output policy contract is not canonical JSON"
                ) from exc
            self.expected_output_policy_contract_sha256 = expected_contract_sha
        # Snapshotting is read-only and needed only when lineage is enabled.
        self._baseline = {} if self.no_kg else _checkpoint_snapshot(self.project_dir)

    def _iteration(self, iteration_index: int) -> TrainingIteration:
        if self._run is None:
            raise LineageObservationError("iteration observed before run start")
        existing = self._iterations.get(iteration_index)
        if existing is not None:
            return existing
        iteration = TrainingIteration(
            id=make_training_iteration_id(
                self._run.project, self._run.run_id, iteration_index,
            ),
            project=self._run.project,
            run_id=self._run.run_id,
            iteration_index=iteration_index,
        )
        with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
            recorded = record_iteration_started(
                kg, run=self._run, iteration_index=iteration_index,
            )
        if recorded is None or recorded.id != iteration.id:
            raise LineageObservationError(
                "training iteration could not be recorded"
            )
        self._iterations[iteration_index] = iteration
        return iteration

    def record_started(self) -> None:
        """Record invocation after process creation, without inferred inputs."""
        if self.no_kg:
            return
        self._run = TrainingRun(
            id=make_training_run_id(self.project_slug, self.run_id),
            project=self.project_slug,
            run_id=self.run_id,
            requested_initialization_mode=self.requested_initialization_mode,
            # The backend process's checkout is not proof of the spawned
            # worker's code. This is filled only by run_context_captured.
            code_commit=None,
            selection_digest=None,
        )
        with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
            record_run_started(kg, run=self._run)

    def observe_event(self, event: dict[str, Any]) -> None:
        """Consume an event and persist any strict-lineage rejection as fatal."""
        if self.no_kg:
            return
        try:
            self._observe_event(event)
        except Exception as exc:
            if self.reference_robot is not None:
                event_type = (
                    event.get("type") if isinstance(event, dict) else None
                )
                failure = (
                    f"{str(event_type or 'unknown')}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if failure not in self._fatal_observation_errors:
                    self._fatal_observation_errors.append(failure)
            raise

    def _observe_event(self, event: dict[str, Any]) -> None:
        """Consume a parsed worker event; only successful facts earn edges."""
        if self._run is None:
            raise LineageObservationError(
                "worker event arrived before the run was recorded"
            )
        event_type = event.get("type")
        if event_type == "iter_started":
            iteration_index = event.get("iter")
            if (
                isinstance(iteration_index, bool)
                or not isinstance(iteration_index, int)
                or iteration_index < 0
            ):
                raise LineageObservationError(
                    "iter_started has no non-negative iteration index"
                )
            if iteration_index in self._iterations:
                if iteration_index != self._current_iteration:
                    raise LineageObservationError(
                        "worker replayed an out-of-order iter_started event"
                    )
                return
            if (
                self._current_iteration is not None
                and self._current_iteration
                not in self._completed_event_by_iteration
            ):
                raise LineageObservationError(
                    "worker started a new iteration before completing the prior one"
                )
            if self._early_stop_evidence is not None:
                raise LineageObservationError(
                    "worker started another iteration after its attested early stop"
                )
            if self._iterations and iteration_index != max(self._iterations) + 1:
                raise LineageObservationError(
                    "worker iteration sequence is not contiguous"
                )
            if (
                self.expected_iterations is not None
                and len(self._iterations) >= self.expected_iterations
            ):
                raise LineageObservationError(
                    "worker started more iterations than the launch plan"
                )
            self._iteration(iteration_index)
            self._current_iteration = iteration_index
            return
        if event_type == "iter_completed":
            iteration_index = event.get("iter")
            if (
                isinstance(iteration_index, bool)
                or not isinstance(iteration_index, int)
                or iteration_index < 0
            ):
                raise LineageObservationError(
                    "iter_completed has no non-negative iteration index"
                )
            if (
                iteration_index not in self._iterations
                or iteration_index != self._current_iteration
            ):
                raise LineageObservationError(
                    "iter_completed has no matching current iter_started authority"
                )
            try:
                event_sha256 = _canonical_sha256(event)
            except (TypeError, ValueError) as exc:
                raise LineageObservationError(
                    "iter_completed evidence is not canonical JSON"
                ) from exc
            prior = self._completed_event_by_iteration.get(iteration_index)
            if prior is not None and prior != event_sha256:
                raise LineageObservationError(
                    "worker emitted conflicting iter_completed evidence"
                )
            self._completed_event_by_iteration[iteration_index] = event_sha256
            return
        if event_type == "early_stop":
            iteration_index = event.get("at_iter")
            source = event.get("source")
            reason = event.get("reason")
            if (
                isinstance(iteration_index, bool)
                or not isinstance(iteration_index, int)
                or iteration_index not in self._completed_event_by_iteration
                or iteration_index != max(self._completed_event_by_iteration)
            ):
                raise LineageObservationError(
                    "early_stop has no matching completed terminal iteration"
                )
            if (
                not isinstance(source, str)
                or source not in self.allowed_early_stop_sources
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise LineageObservationError(
                    "early_stop was not independently authorized at launch"
                )
            evidence = {
                "at_iter": iteration_index,
                "source": source,
                "reason": reason.strip(),
                "event_sha256": _canonical_sha256(event),
            }
            if (
                self._early_stop_evidence is not None
                and self._early_stop_evidence != evidence
            ):
                raise LineageObservationError(
                    "worker emitted conflicting early_stop evidence"
                )
            self._early_stop_evidence = evidence
            return
        if event_type == "reference_feasibility_admitted":
            if event.get("source") != "sculpt_run_worker":
                raise LineageObservationError(
                    "reference feasibility lineage requires the worker admission"
                )
            observed = _normalized_tierd_receipt(
                event, label="worker Tier-D receipt",
            )
            expected = self.reference_feasibility_receipt
            if expected is None:
                raise LineageObservationError(
                    "worker Tier-D receipt has no launch-admitted authority"
                )
            if observed != expected:
                raise LineageObservationError(
                    "worker Tier-D receipt differs from launch admission"
                )
            robot = observed["reference_robot"]
            clip_id = observed["reference_clip_id"]
            clip_sha = observed["clip_sha256"]
            if self.reference_robot is not None and robot != self.reference_robot:
                raise LineageObservationError(
                    "Tier-D robot differs from launch selection"
                )
            if self.reference_clip_id is not None and clip_id != self.reference_clip_id:
                raise LineageObservationError(
                    "Tier-D clip differs from launch selection"
                )
            if self.reference_sha256 is not None and clip_sha != self.reference_sha256:
                raise LineageObservationError(
                    "Tier-D clip digest differs from launch admission"
                )
            evidence = {
                "authority": "reference_feasibility_admitted+run_started",
                "verified": True,
                "tierd_receipt": observed,
                "tierd_receipt_sha256": _canonical_sha256(observed),
                "tier": observed["tier"],
                "robot": robot,
                "clip_id": clip_id,
                "clip_sha256": clip_sha,
                "rollout_sha256": observed["rollout_sha256"],
                "certificate_sha256": observed["certificate_sha256"],
                "execution_contract_sha256": observed[
                    "execution_contract_sha256"
                ],
                "execution_boundary_sha256": observed[
                    "execution_boundary_sha256"
                ],
                "target_robot": observed["target_robot"],
                "certification_scope": observed["certification_scope"],
                "certification_scope_sha256": _canonical_sha256(
                    observed["certification_scope"]
                ),
            }
            if (
                self._reference_feasibility is not None
                and self._reference_feasibility != evidence
            ):
                raise LineageObservationError(
                    "worker emitted conflicting Tier-D reference evidence"
                )
            self._reference_feasibility = evidence
            return
        if event_type == "reference_runtime_schedule_admitted":
            schedule = _normalized_reference_schedule(event)
            if (
                self.reference_robot is not None
                and schedule["reference_robot"] != self.reference_robot
            ):
                raise LineageObservationError(
                    "reference schedule robot differs from launch selection"
                )
            if (
                self.reference_clip_id is not None
                and schedule["reference_clip_id"] != self.reference_clip_id
            ):
                raise LineageObservationError(
                    "reference schedule clip differs from launch selection"
                )
            if (
                self._reference_runtime_schedule is not None
                and self._reference_runtime_schedule != schedule
            ):
                raise LineageObservationError(
                    "worker emitted conflicting reference runtime schedules"
                )
            self._reference_runtime_schedule = schedule
            return
        if event_type == "run_context_captured":
            software, evidence, code_commit = _software_from_run_context_event(
                self.project_dir, event,
            )
            if code_commit is not None:
                self._run = replace(self._run, code_commit=code_commit)
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                record_run_started(
                    kg,
                    run=self._run,
                    software=software,
                    software_edge_data=evidence,
                )
            self._software_confirmed = True
            self._software_id = software.id
            return
        if event_type == "artifact_tuple_pinned":
            world, evidence = _world_from_pinned_event(
                self.project_dir, event,
            )
            if self._run.selection_digest is None:
                self._run = replace(
                    self._run, selection_digest=world.sha256,
                )
            self._latest_selection_digest = world.sha256
            iteration_index = int(evidence["iteration"])
            iteration = self._iteration(iteration_index)
            prior = self._world_by_iteration.get(iteration_index)
            if prior is not None and prior != world.sha256:
                raise LineageObservationError(
                    "training iteration observed two authored world tuples"
                )
            run_evidence = {
                key: value for key, value in evidence.items()
                if key != "iteration"
            }
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                with kg.transaction():
                    record_run_started(
                        kg,
                        run=self._run,
                        active_world=world,
                        active_world_edge_data=run_evidence,
                    )
                    record_iteration_world(
                        kg,
                        run=self._run,
                        iteration=iteration,
                        world=world,
                        evidence=evidence,
                    )
            self._world_by_iteration[iteration_index] = world.sha256
            if evidence["mode_execution_required"]:
                self._mode_required_iterations.add(iteration_index)
            return
        if event_type == "mode_execution_admitted":
            artifact, selection_digest = _mode_execution_from_admitted_event(
                self.project_dir, event,
            )
            evidence = self._reference_feasibility
            if (
                not self._reference_confirmed
                or evidence is None
                or evidence.get("robot") != artifact.robot
                or evidence.get("clip_id") != artifact.clip_id
                or evidence.get("clip_sha256") != artifact.clip_sha256
            ):
                raise LineageObservationError(
                    "mode execution has no matching verified Tier-D reference "
                    "confirmed by run_started"
                )
            if self._latest_selection_digest != selection_digest:
                raise LineageObservationError(
                    "mode execution was not admitted against the run's latest "
                    "pinned artifact tuple"
                )
            iteration_index = self._current_iteration
            if iteration_index is None:
                raise LineageObservationError(
                    "mode execution was admitted before iter_started"
                )
            if self._world_by_iteration.get(iteration_index) != selection_digest:
                raise LineageObservationError(
                    "mode execution selection differs from its iteration world"
                )
            prior_mode = self._mode_by_iteration.get(iteration_index)
            if prior_mode is not None and prior_mode != artifact.id:
                raise LineageObservationError(
                    "training iteration observed two mode executors"
                )
            iteration = self._iteration(iteration_index)
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                with kg.transaction():
                    record_mode_execution_admitted(
                        kg,
                        run=self._run,
                        artifact=artifact,
                        selection_digest=selection_digest,
                        iteration=iteration,
                    )
            self._mode_by_iteration[iteration_index] = artifact.id
            return
        if event_type == "run_started":
            context = event.get("reference_motion")
            if context is None:
                if self.reference_robot or self.reference_clip_id:
                    raise LineageObservationError(
                        "worker did not confirm the selected reference"
                    )
                return
            if not isinstance(context, dict):
                raise LineageObservationError(
                    "worker reference_motion context is not an object"
                )
            robot = context.get("robot")
            clip_id = context.get("clip_id")
            clip_sha = context.get("clip_sha256")
            if not all(isinstance(value, str) and value for value in (
                robot, clip_id, clip_sha,
            )):
                raise LineageObservationError(
                    "worker reference context lacks robot/clip/digest"
                )
            if self.reference_robot is not None and robot != self.reference_robot:
                raise LineageObservationError(
                    "worker reference robot differs from launch selection"
                )
            if self.reference_clip_id is not None and clip_id != self.reference_clip_id:
                raise LineageObservationError(
                    "worker reference clip differs from launch selection"
                )
            if self.reference_sha256 is not None and clip_sha != self.reference_sha256:
                raise LineageObservationError(
                    "worker reference digest differs from launch admission"
                )
            evidence = self._reference_feasibility
            if evidence is None:
                raise LineageObservationError(
                    "worker reference has no prior verified Tier-D admission"
                )
            if (
                evidence["robot"] != robot
                or evidence["clip_id"] != clip_id
                or evidence["clip_sha256"] != clip_sha
            ):
                raise LineageObservationError(
                    "worker reference differs from verified Tier-D evidence"
                )
            schedule = self._reference_runtime_schedule
            if schedule is None:
                raise LineageObservationError(
                    "worker reference has no admitted runtime schedule/backbone"
                )
            raw_reward_path = context.get("reward_path")
            if not isinstance(raw_reward_path, str) or not raw_reward_path:
                raise LineageObservationError(
                    "worker reference context has no exact reward path"
                )
            rederived_schedule = _rederive_reference_schedule(
                self.project_dir,
                robot=robot,
                clip_id=clip_id,
                reward_path=Path(raw_reward_path),
            )
            if schedule != rederived_schedule:
                raise LineageObservationError(
                    "worker reference schedule/backbone differs from reward bytes"
                )
            evidence = {
                **evidence,
                "runtime_schedule_authority": (
                    "reference_runtime_schedule_admitted"
                ),
                "runtime_schedule": schedule,
                "runtime_schedule_sha256": _canonical_sha256(schedule),
                **{
                    key: schedule[key]
                    for key in (
                        "reference_target_sha256",
                        "phase_mode",
                        "phase_duration_s",
                        "n_phase_targets",
                        "tracking_backbone_sha256",
                    )
                },
            }
            reference = reference_motion_from_library(
                robot, clip_id, expected_sha256=clip_sha,
            )
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                record_run_started(
                    kg,
                    run=self._run,
                    effective_reference=reference,
                    reference_edge_data=evidence,
                )
            self._reference_feasibility = evidence
            self._reference_confirmed = True
            self._reference_id = reference.id
            return
        # Raw warm-start events are necessary evidence, but the backend's
        # Requested/Resolved/Observed join is the sole initialization authority.
        # ``record_verified_initialization`` consumes that joined receipt after
        # run_manager has validated it.
        return

    def record_verified_initialization(
        self, receipt: dict[str, Any],
    ) -> PolicyArtifact:
        """Earn INITIALIZED_FROM from the UI's exact three-way authority."""
        if self.no_kg:
            raise LineageObservationError(
                "reference proof cannot record initialization with KG disabled"
            )
        if self._run is None:
            raise LineageObservationError(
                "policy initialization observed before run start"
            )
        if not isinstance(receipt, dict):
            raise LineageObservationError(
                "verified policy initialization receipt is not an object"
            )
        requested = receipt.get("requested")
        resolved = receipt.get("resolved")
        observed = receipt.get("observed")
        if (
            receipt.get("schema") != 1
            or not isinstance(requested, dict)
            or not isinstance(resolved, dict)
            or not isinstance(observed, dict)
        ):
            raise LineageObservationError(
                "verified policy initialization receipt is malformed"
            )
        raw_source = observed.get("source")
        if not isinstance(raw_source, str) or not raw_source:
            raise LineageObservationError(
                "verified initialization has no source checkpoint"
            )
        source = Path(raw_source).expanduser().resolve(strict=True)
        project_runs = (self.project_dir / "runs").resolve()
        admitted_skill_path: Path | None = None
        if self.starting_skill_record is not None and self.starting_skill_record.policy_roles:
            from sculptor.skill_library import SkillLibrary

            admitted_skill_path = SkillLibrary().checkpoint_path_for(
                self.starting_skill_record
            ).resolve()
        try:
            source.relative_to(project_runs)
            allowed = True
        except ValueError:
            allowed = admitted_skill_path is not None and source == admitted_skill_path
        if not allowed:
            raise LineageObservationError(
                "verified initialization source is outside project runs and the "
                "admitted starting skill"
            )
        source_policy = policy_artifact_from_checkpoint(
            source, record=self.starting_skill_record,
        )
        source_sha256 = observed.get("source_sha256")
        if source_sha256 != source_policy.sha256:
            raise LineageObservationError(
                "verified initialization digest does not match source bytes"
            )
        raw_loaded = observed.get("loaded_checkpoint")
        if not isinstance(raw_loaded, str) or not raw_loaded:
            raise LineageObservationError(
                "verified initialization has no effective checkpoint path"
            )
        loaded = Path(raw_loaded).expanduser().resolve(strict=True)
        try:
            loaded.relative_to(project_runs)
            loaded_allowed = True
        except ValueError:
            loaded_allowed = loaded == source
        if not loaded_allowed:
            raise LineageObservationError(
                "verified initialization checkpoint is outside project runs"
            )
        loaded_policy = (
            source_policy
            if loaded == source
            else policy_artifact_from_checkpoint(loaded)
        )
        loaded_sha256 = observed.get("loaded_checkpoint_sha256")
        if loaded_sha256 != loaded_policy.sha256:
            raise LineageObservationError(
                "verified initialization digest does not match effective bytes"
            )
        is_derived = loaded_policy.id != source_policy.id
        expected_contract_receipt = self.warm_start_policy_contract_receipt
        expected_source = (
            expected_contract_receipt.get("source")
            if isinstance(expected_contract_receipt, dict)
            else None
        )
        expected_target = (
            expected_contract_receipt.get("target")
            if isinstance(expected_contract_receipt, dict)
            else None
        )
        expected_migration = (
            expected_contract_receipt.get("compatibility")
            if isinstance(expected_contract_receipt, dict)
            else None
        )
        observed_migration = observed.get("policy_contract_migration")
        if is_derived and (
            observed.get("adapted") is not True
            or not isinstance(expected_source, dict)
            or not isinstance(expected_target, dict)
            or not isinstance(expected_migration, dict)
            or expected_migration.get("type")
            not in {
                "zero_initialized_event_phase_observation",
                "zero_initialized_reference_clock_observation",
                "zero_initialized_observation_extensions",
            }
            or expected_migration.get("optimizer_resume") is not False
            or observed_migration != expected_migration
            or resolved.get("source_policy_contract_sha256")
            != expected_source.get("contract_sha256")
            or observed.get("effective_policy_contract_sha256")
            != expected_target.get("contract_sha256")
        ):
            raise LineageObservationError(
                "derived warm-start checkpoint lacks exact migration lineage"
            )
        if not is_derived and observed.get("adapted") is True:
            raise LineageObservationError(
                "warm_start_loaded claims adaptation without derived bytes"
            )
        raw_keys = observed.get("load_cfg_keys")
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) for key in raw_keys
        ):
            raise LineageObservationError(
                "verified initialization has no structural load roles"
            )
        # Keep multiplicity: an exact role receipt is a structural contract,
        # not a set-membership claim.  Duplicate or extra roles must fail the
        # same equality check as a missing critic.
        keys = sorted(raw_keys)
        if "actor" not in keys:
            raise LineageObservationError(
                "warm_start_loaded did not report actor weight loading"
            )
        transfer_mode = "actor_critic" if "critic" in keys else "actor_only"
        expected_keys = (
            ["actor", "critic"]
            if self.requested_initialization_mode == "actor_critic"
            else ["actor"]
        )
        role_contract_is_explicit = self.requested_initialization_mode in {
            "actor_only", "actor_critic",
        }
        if role_contract_is_explicit and keys != expected_keys:
            raise LineageObservationError(
                f"worker loaded roles {keys}, expected exactly {expected_keys}"
            )
        iteration_index = self._current_iteration
        if iteration_index is None:
            raise LineageObservationError(
                "verified initialization was observed before iter_started"
            )
        iteration = self._iteration(iteration_index)
        canonical_receipt = json.loads(json.dumps(
            receipt, sort_keys=True, allow_nan=False,
        ))
        prior_receipt = self._initialization_receipt_by_iteration.get(
            iteration_index
        )
        if prior_receipt is not None and prior_receipt != canonical_receipt:
            raise LineageObservationError(
                "training iteration observed conflicting initialization receipts"
            )
        compatible_target: RobotEmbodiment | None = None
        if isinstance(expected_target, dict):
            target_contract = expected_target.get("contract")
            target_digest = expected_target.get("contract_sha256")
            if isinstance(target_contract, dict) and isinstance(target_digest, str):
                from sculptor.policy_contract import contract_fingerprint

                if contract_fingerprint(target_contract) != target_digest:
                    raise LineageObservationError(
                        "verified target policy contract fingerprint is invalid"
                    )
                compatible_target = _robot_from_contract(
                    self.reference_robot,
                    target_contract,
                    target_digest,
                )
        with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
            record_policy_loaded(
                kg,
                run=self._run,
                policy=loaded_policy,
                transfer_mode=transfer_mode,
                checkpoint_sha256=loaded_policy.sha256,
                load_cfg_keys=keys,
                initialization_receipt=canonical_receipt,
                iteration=iteration,
                compatible_target=compatible_target,
                derived_from_policy=(
                    source_policy if is_derived else None
                ),
                derivation_data=(
                    {
                        "migration": observed_migration,
                        "source_policy_contract_sha256": resolved.get(
                            "source_policy_contract_sha256"
                        ),
                        "effective_policy_contract_sha256": observed.get(
                            "effective_policy_contract_sha256"
                        ),
                        "source_checkpoint_sha256": source_policy.sha256,
                        "loaded_checkpoint_sha256": loaded_policy.sha256,
                    }
                    if is_derived
                    else None
                ),
            )
        self._input_policy = loaded_policy
        self._input_policy_by_iteration[iteration_index] = loaded_policy
        self._initialization_receipt_by_iteration[iteration_index] = (
            canonical_receipt
        )
        return loaded_policy

    def record_outputs(self) -> list[PolicyArtifact]:
        """Record only checkpoints absent from or changed since pre-spawn."""
        if self.no_kg:
            return []
        if self._run is None:
            raise LineageObservationError("run outputs observed before run start")
        after = _checkpoint_snapshot(self.project_dir)
        changed = [
            path for path, digest in sorted(after.items(), key=lambda item: str(item[0]))
            if self._baseline.get(path) != digest
        ]
        if self.reference_robot is not None:
            if self._reference_runtime_schedule is None:
                raise LineageObservationError(
                    "reference outputs have no admitted runtime schedule"
                )
            verified = [
                _verified_reference_output(
                    self.project_dir,
                    path,
                    robot=self.reference_robot,
                    expected_schedule=self._reference_runtime_schedule,
                    expected_target_contract=(
                        self.expected_output_policy_contract or {}
                    ),
                    expected_target_contract_sha256=str(
                        self.expected_output_policy_contract_sha256 or ""
                    ),
                    expected_target_robot=str(
                        self.expected_output_robot or ""
                    ),
                )
                for path in changed
            ]
            by_iteration: dict[int, _VerifiedOutput] = {}
            for output in verified:
                if output.iteration_index in by_iteration:
                    raise LineageObservationError(
                        "reference run produced two checkpoint formats for one "
                        "training iteration"
                    )
                if output.iteration_index not in self._iterations:
                    raise LineageObservationError(
                        "reference output has no matching iter_started authority"
                    )
                if (
                    self._world_by_iteration.get(output.iteration_index)
                    != output.world_tuple_sha256
                ):
                    raise LineageObservationError(
                        "reference output world differs from its iteration world"
                    )
                by_iteration[output.iteration_index] = output

            resolved_inputs: dict[int, PolicyArtifact | None] = {}
            earlier_outputs: dict[str, tuple[int, PolicyArtifact]] = {}
            for iteration_index in sorted(by_iteration):
                output = by_iteration[iteration_index]
                input_policy: PolicyArtifact | None = None
                if output.input_checkpoint_sha256 is not None:
                    explicit_input = self._input_policy_by_iteration.get(
                        iteration_index
                    )
                    if (
                        explicit_input is not None
                        and explicit_input.sha256
                        == output.input_checkpoint_sha256
                    ):
                        input_policy = explicit_input
                    else:
                        prior = earlier_outputs.get(
                            output.input_checkpoint_sha256
                        )
                        if prior is not None and prior[0] < iteration_index:
                            input_policy = prior[1]
                    if input_policy is None:
                        raise LineageObservationError(
                            "reference output input digest has no verified "
                            "initialization or earlier-iteration output"
                        )
                elif self._input_policy_by_iteration.get(iteration_index) is not None:
                    raise LineageObservationError(
                        "reference output denies its verified policy initialization"
                    )
                resolved_inputs[iteration_index] = input_policy
                earlier_outputs[output.policy.sha256] = (
                    iteration_index, output.policy,
                )

            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                with kg.transaction():
                    for iteration_index in sorted(by_iteration):
                        output = by_iteration[iteration_index]
                        record_run_output(
                            kg,
                            run=self._run,
                            output_policy=output.policy,
                            input_policy=resolved_inputs[iteration_index],
                            iteration=self._iterations[iteration_index],
                            output_evidence=output.evidence,
                            compatible_target=output.compatible_target,
                        )
            for iteration_index, output in by_iteration.items():
                self._output_by_iteration[iteration_index] = output.policy
                self._output_contract_by_iteration[iteration_index] = (
                    output.compatible_target.contract_digest
                )
                self._output_evidence_by_iteration[iteration_index] = dict(
                    output.evidence
                )
                input_policy = resolved_inputs[iteration_index]
                if input_policy is not None:
                    self._input_policy_by_iteration[iteration_index] = input_policy
            return [
                by_iteration[index].policy for index in sorted(by_iteration)
            ]

        outputs: list[PolicyArtifact] = []
        with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
            for path in changed:
                policy = policy_artifact_from_checkpoint(path)
                record_run_output(
                    kg,
                    run=self._run,
                    output_policy=policy,
                    input_policy=(
                        self._input_policy
                        if self._input_policy is not None
                        and self._input_policy.id != policy.id
                        else None
                    ),
                )
                outputs.append(policy)
        return outputs

    def finalize_proof(self) -> dict[str, Any]:
        """Return a proof-ready receipt or reject an incomplete reference run."""
        if self.reference_robot is None:
            return {
                "schema": 1,
                "strict_reference_lineage": False,
                "run_id": self.run_id,
            }
        if self.no_kg:
            raise LineageObservationError(
                "reference-guided completion cannot be proven with KG disabled"
            )
        if self._run is None:
            raise LineageObservationError("reference run has no training-run node")
        if self._fatal_observation_errors:
            raise LineageObservationError(
                "reference run contains rejected authoritative lineage evidence: "
                + " | ".join(self._fatal_observation_errors)
            )
        if not self._software_confirmed or self._software_id is None:
            raise LineageObservationError(
                "reference run has no verified observed software environment"
            )
        if (
            self.reference_feasibility_receipt is None
            or self._reference_feasibility is None
            or not self._reference_confirmed
            or self._reference_id is None
        ):
            raise LineageObservationError(
                "reference run has no exact launch/worker Tier-D TRACKS authority"
            )
        if self._reference_runtime_schedule is None:
            raise LineageObservationError(
                "reference run has no verified runtime target/clock/backbone"
            )
        if not self._iterations:
            raise LineageObservationError(
                "reference run has no observed training iterations"
            )

        iteration_receipts: list[dict[str, Any]] = []
        ordered_iterations = sorted(self._iterations)
        if ordered_iterations != list(range(
            ordered_iterations[0],
            ordered_iterations[0] + len(ordered_iterations),
        )):
            raise LineageObservationError(
                "reference run iteration sequence is not contiguous"
            )
        completed_iterations = sorted(self._completed_event_by_iteration)
        if completed_iterations != ordered_iterations:
            raise LineageObservationError(
                "reference run has an iteration without exact iter_completed evidence"
            )
        if len(ordered_iterations) != self.expected_iterations:
            early_stop = self._early_stop_evidence
            if (
                early_stop is None
                or len(ordered_iterations) >= int(self.expected_iterations or 0)
                or early_stop["at_iter"] != ordered_iterations[-1]
            ):
                raise LineageObservationError(
                    "reference run did not complete the exact launch iteration plan "
                    "or an independently authorized early stop"
                )
        first_iteration = ordered_iterations[0]
        for iteration_index in ordered_iterations:
            world_sha = self._world_by_iteration.get(iteration_index)
            output = self._output_by_iteration.get(iteration_index)
            output_contract = self._output_contract_by_iteration.get(
                iteration_index
            )
            output_evidence = self._output_evidence_by_iteration.get(
                iteration_index
            )
            if world_sha is None:
                raise LineageObservationError(
                    f"iteration {iteration_index} has no validated world tuple"
                )
            if (
                iteration_index in self._mode_required_iterations
                and iteration_index not in self._mode_by_iteration
            ):
                raise LineageObservationError(
                    f"iteration {iteration_index} lacks its required mode executor"
                )
            if output is None or output_contract is None or output_evidence is None:
                raise LineageObservationError(
                    f"iteration {iteration_index} has no verified output lineage"
                )
            input_policy = self._input_policy_by_iteration.get(iteration_index)
            initialization_receipt = (
                self._initialization_receipt_by_iteration.get(iteration_index)
            )
            if (
                iteration_index == first_iteration
                and self.requested_initialization_mode
                in {"actor_only", "actor_critic"}
                and initialization_receipt is None
            ):
                raise LineageObservationError(
                    "selected starting policy lacks its verified "
                    "Requested/Resolved/Observed initialization receipt"
                )
            if (
                input_policy is not None
                and output_evidence.get("input_checkpoint_loaded_sha256")
                != input_policy.sha256
            ):
                raise LineageObservationError(
                    f"iteration {iteration_index} input ancestry is inconsistent"
                )
            iteration_receipts.append({
                "iteration": iteration_index,
                "iteration_node_id": self._iterations[iteration_index].id,
                "world_tuple_sha256": world_sha,
                "mode_execution_required": (
                    iteration_index in self._mode_required_iterations
                ),
                "mode_execution_artifact_id": self._mode_by_iteration.get(
                    iteration_index
                ),
                "input_policy_sha256": (
                    input_policy.sha256 if input_policy is not None else None
                ),
                "input_authority": (
                    "starting_policy_initialization_verified"
                    if initialization_receipt is not None
                    else "runner_runtime_artifacts"
                    if input_policy is not None
                    else "runner_no_input_checkpoint"
                ),
                "initialization_receipt_sha256": (
                    _canonical_sha256(initialization_receipt)
                    if initialization_receipt is not None else None
                ),
                "iter_completed_event_sha256": (
                    self._completed_event_by_iteration[iteration_index]
                ),
                "output_policy_sha256": output.sha256,
                "output_policy_contract_sha256": output_contract,
                "runner_runtime_artifacts_sha256": output_evidence[
                    "runtime_artifacts_sha256"
                ],
            })

        receipt = {
            "schema": 1,
            "strict_reference_lineage": True,
            "authority": "reference_guided_completion_verified",
            "run_id": self.run_id,
            "training_run_node_id": self._run.id,
            "software_environment_node_id": self._software_id,
            "reference_motion_node_id": self._reference_id,
            "tierd_receipt": self.reference_feasibility_receipt,
            "reference_runtime_schedule": self._reference_runtime_schedule,
            "output_policy_target": {
                "robot": self.expected_output_robot,
                "policy_contract_sha256": (
                    self.expected_output_policy_contract_sha256
                ),
            },
            "iteration_plan": {
                "expected_count": self.expected_iterations,
                "allowed_early_stop_sources": list(
                    self.allowed_early_stop_sources
                ),
                "completed_count": len(ordered_iterations),
                "early_stop": self._early_stop_evidence,
            },
            "iterations": iteration_receipts,
        }
        receipt["proof_sha256"] = _canonical_sha256(receipt)
        return receipt


__all__ = [
    "LineageObservationError",
    "RunLineageSession",
    "policy_artifact_from_checkpoint",
    "record_admitted_starting_skill",
    "reference_motion_from_library",
]
