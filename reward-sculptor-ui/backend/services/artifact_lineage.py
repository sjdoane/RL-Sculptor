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
    TrainingRun,
    WorldArtifact,
    make_artifact_attestation_id,
    make_policy_artifact_id,
    make_reference_motion_id,
    make_robot_embodiment_id,
    make_software_environment_id,
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


def _world_from_pinned_event(
    project_dir: Path, event: dict[str, Any],
) -> tuple[WorldArtifact, dict[str, Any]]:
    from sculptor.world.artifacts import WorldArtifactStore

    store = WorldArtifactStore(project_dir)
    selection_name = event.get("selection")
    expected_tuple = event.get("tuple_hash")
    if not isinstance(selection_name, str) or not selection_name:
        raise LineageObservationError(
            "artifact_tuple_pinned has no immutable selection filename"
        )
    if not isinstance(expected_tuple, str) or not expected_tuple:
        raise LineageObservationError(
            "artifact_tuple_pinned has no tuple hash"
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
    edge_data = {
        "status": "active",
        "validated": True,
        "authority": "artifact_tuple_pinned",
        "tuple_hash": selection.tuple_hash,
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
    starting_skill_record: SkillRecord | None = None
    _baseline: dict[Path, str] = field(init=False, repr=False)
    _run: TrainingRun | None = field(default=None, init=False, repr=False)
    _input_policy: PolicyArtifact | None = field(
        default=None, init=False, repr=False,
    )
    _reference_feasibility: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _reference_confirmed: bool = field(default=False, init=False, repr=False)
    _latest_selection_digest: str | None = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        # Snapshotting is read-only and needed only when lineage is enabled.
        self._baseline = {} if self.no_kg else _checkpoint_snapshot(self.project_dir)

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
        """Consume a parsed worker event; only successful load events matter."""
        if self.no_kg:
            return
        if self._run is None:
            raise LineageObservationError(
                "worker event arrived before the run was recorded"
            )
        event_type = event.get("type")
        if event_type == "reference_feasibility_admitted":
            if event.get("source") != "sculpt_run_worker":
                raise LineageObservationError(
                    "reference feasibility lineage requires the worker admission"
                )
            robot = event.get("reference_robot")
            clip_id = event.get("reference_clip_id")
            if not isinstance(robot, str) or not robot:
                raise LineageObservationError(
                    "reference feasibility event has no robot namespace"
                )
            if not isinstance(clip_id, str) or not clip_id:
                raise LineageObservationError(
                    "reference feasibility event has no clip identity"
                )
            clip_sha = _optional_sha256(
                event.get("clip_sha256"), label="Tier-D clip digest",
            )
            rollout_sha = _optional_sha256(
                event.get("rollout_sha256"), label="Tier-D rollout digest",
            )
            certificate_sha = _optional_sha256(
                event.get("certificate_sha256"),
                label="Tier-D certificate digest",
            )
            execution_contract_sha = _optional_sha256(
                event.get("execution_contract_sha256"),
                label="Tier-D execution contract digest",
            )
            execution_boundary_sha = _optional_sha256(
                event.get("execution_boundary_sha256"),
                label="Tier-D execution boundary digest",
            )
            target_robot = event.get("target_robot")
            if not isinstance(target_robot, str) or target_robot != robot:
                raise LineageObservationError(
                    "Tier-D target robot differs from reference robot"
                )
            if any(value is None for value in (
                clip_sha,
                rollout_sha,
                certificate_sha,
                execution_contract_sha,
                execution_boundary_sha,
            )):
                raise LineageObservationError(
                    "reference feasibility event lacks exact Tier-D pins"
                )
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
                "tier": "D",
                "robot": robot,
                "clip_id": clip_id,
                "clip_sha256": clip_sha,
                "rollout_sha256": rollout_sha,
                "certificate_sha256": certificate_sha,
                "execution_contract_sha256": execution_contract_sha,
                "execution_boundary_sha256": execution_boundary_sha,
                "target_robot": target_robot,
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
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                record_run_started(
                    kg,
                    run=self._run,
                    active_world=world,
                    active_world_edge_data=evidence,
                )
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
            with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
                with kg.transaction():
                    record_mode_execution_admitted(
                        kg,
                        run=self._run,
                        artifact=artifact,
                        selection_digest=selection_digest,
                    )
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
            self._reference_confirmed = True
            return
        if event_type != "warm_start_loaded":
            return
        raw_source = event.get("source")
        if not isinstance(raw_source, str) or not raw_source:
            raise LineageObservationError("warm_start_loaded has no checkpoint path")
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
                "warm_start_loaded source is outside project runs and the "
                "admitted starting skill"
            )
        source_policy = policy_artifact_from_checkpoint(
            source, record=self.starting_skill_record,
        )
        source_sha256 = event.get("source_sha256")
        if source_sha256 != source_policy.sha256:
            raise LineageObservationError(
                "warm_start_loaded full digest does not match checkpoint bytes"
            )
        raw_loaded = event.get("loaded_checkpoint", raw_source)
        if not isinstance(raw_loaded, str) or not raw_loaded:
            raise LineageObservationError(
                "warm_start_loaded has no actual loaded checkpoint path"
            )
        loaded = Path(raw_loaded).expanduser().resolve(strict=True)
        try:
            loaded.relative_to(project_runs)
            loaded_allowed = True
        except ValueError:
            loaded_allowed = loaded == source
        if not loaded_allowed:
            raise LineageObservationError(
                "warm_start_loaded actual checkpoint is outside project runs"
            )
        loaded_policy = (
            source_policy
            if loaded == source
            else policy_artifact_from_checkpoint(loaded)
        )
        loaded_sha256 = event.get(
            "loaded_checkpoint_sha256", source_sha256,
        )
        if loaded_sha256 != loaded_policy.sha256:
            raise LineageObservationError(
                "warm_start_loaded actual digest does not match loaded bytes"
            )
        is_derived = loaded_policy.id != source_policy.id
        derived_from = event.get("derived_from")
        if is_derived and (
            event.get("adapted") is not True
            or not isinstance(derived_from, dict)
            or derived_from.get("source_sha256") != source_policy.sha256
            or event.get("policy_contract_migration")
            != "zero_initialized_event_phase_observation"
        ):
            raise LineageObservationError(
                "derived warm-start checkpoint lacks exact migration lineage"
            )
        if not is_derived and event.get("adapted") is True:
            raise LineageObservationError(
                "warm_start_loaded claims adaptation without derived bytes"
            )
        raw_keys = event.get("load_cfg_keys")
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) for key in raw_keys
        ):
            raise LineageObservationError(
                "warm_start_loaded has no structural load_cfg_keys evidence"
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
        with SculptorKG(project_kg_db_path(self.project_dir)) as kg:
            record_policy_loaded(
                kg,
                run=self._run,
                policy=loaded_policy,
                transfer_mode=transfer_mode,
                checkpoint_sha256=loaded_policy.sha256,
                load_cfg_keys=keys,
                derived_from_policy=(
                    source_policy if is_derived else None
                ),
                derivation_data=(
                    {
                        "migration": event.get(
                            "admitted_policy_contract_migration"
                        ),
                        "source_policy_contract_sha256": event.get(
                            "source_policy_contract_sha256"
                        ),
                        "effective_policy_contract_sha256": event.get(
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


__all__ = [
    "LineageObservationError",
    "RunLineageSession",
    "policy_artifact_from_checkpoint",
    "record_admitted_starting_skill",
    "reference_motion_from_library",
]
