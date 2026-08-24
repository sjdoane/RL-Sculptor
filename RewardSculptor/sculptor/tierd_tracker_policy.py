"""Pure contracts for exporting a certified Tier-D tracker actor.

The tracker checkpoint is trusted local state and may be converted to a
portable actor, but the certificate that shaped it is provenance only.  It
does not select a reference, world, controller, reward, optimizer state, or
mode executor in the importing project.

This module deliberately performs only JSON/type/hash validation.  It never
opens a file, imports a simulator, loads a checkpoint, or executes policy
bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

TIERD_TRACKER_POLICY_ORIGIN_SCHEMA = 1
TIERD_TRACKER_POLICY_ORIGIN_KIND = "reward-sculptor-tier-d-tracker-actor-origin"
TIERD_TRACKER_POLICY_ORIGIN_MEMBER = "provenance/tierd_tracker_origin.json"
TIERD_TRACKER_POLICY_SUMMARY_SCHEMA = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITIES = {
    "policy_roles": ["actor"],
    "initialization_modes": ["actor_only"],
    "optimizer_resume": False,
    "exact_resume": False,
    "reference_activation": False,
    "world_activation": False,
    "controller_activation": False,
    "mode_reuse_supported": False,
}
_ORIGIN_KEYS = {
    "schema",
    "kind",
    "robot",
    "clip_id",
    "capabilities",
    "source_artifacts",
    "checkpoint_policy_contract_sidecar",
    "certificate",
}
_SOURCE_ARTIFACT_KEYS = {
    "checkpoint_sha256",
    "checkpoint_policy_contract_sidecar_sha256",
    "certification_config_sha256",
    "reward_module_sha256",
    "portable_actor_safetensors_sha256",
}
_CHECKPOINT_SIDECAR_KEYS = {
    "schema",
    "checkpoint_sha256",
    "policy_contract",
    "policy_contract_sha256",
}
_CERTIFICATE_KEYS = {"sha256", "payload"}
_CERTIFICATE_PAYLOAD_KEYS = {"schema", "robot", "clip_id", "tierD"}
_SUMMARY_KEYS = {
    "schema",
    "kind",
    "robot",
    "clip_id",
    "policy_roles",
    "initialization_modes",
    "optimizer_resume",
    "exact_resume",
    "reference_activation",
    "world_activation",
    "controller_activation",
    "mode_reuse_supported",
    "source_checkpoint_sha256",
    "source_policy_contract_sha256",
    "portable_actor_safetensors_sha256",
    "checkpoint_policy_contract_sidecar_sha256",
    "certificate_sha256",
    "reference_clip_sha256",
    "rollout_sha256",
    "execution_contract_sha256",
    "execution_boundary_sha256",
    "tracker_iterations",
    "tracked_at",
}


class TierDTrackerPolicyError(ValueError):
    """A tracker-actor origin document is missing exact evidence."""


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical bytes used by Tier-D certificate and summary digests."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TierDTrackerPolicyError(
            "tracker policy provenance is not canonical JSON"
        ) from exc


def _json_clone(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TierDTrackerPolicyError(
                f"tracker policy provenance contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise TierDTrackerPolicyError(
        f"tracker policy provenance contains non-finite number {value!r}"
    )


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def parse_tierd_tracker_policy_origin(payload: bytes | str) -> dict[str, Any]:
    """Parse one retained origin member without ambiguous JSON extensions."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
        )
    except TierDTrackerPolicyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise TierDTrackerPolicyError(
            "tracker policy provenance is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TierDTrackerPolicyError(
            "tracker policy provenance root must be an object"
        )
    return value


def _require_exact_keys(
    value: Any, expected: set[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TierDTrackerPolicyError(
            f"{label} must contain exactly {sorted(expected)}"
        )
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TierDTrackerPolicyError(f"{label} must be lowercase SHA-256")
    return value


def _require_identity(robot: Any, clip_id: Any) -> tuple[str, str]:
    try:
        from sculptor.project_robot import validate_robot_namespace
        from sculptor.refs.library import validate_clip_id

        normalized_robot = validate_robot_namespace(robot)
        normalized_clip = validate_clip_id(clip_id)
    except (TypeError, ValueError) as exc:
        raise TierDTrackerPolicyError(
            f"tracker policy source identity is invalid: {exc}"
        ) from exc
    return normalized_robot, normalized_clip


def _contract_digest(contract: Any) -> str:
    if not isinstance(contract, dict):
        raise TierDTrackerPolicyError(
            "tracker policy source contract must be an object"
        )
    try:
        from sculptor.policy_contract import contract_fingerprint

        return contract_fingerprint(contract)
    except (TypeError, ValueError, KeyError) as exc:
        raise TierDTrackerPolicyError(
            f"tracker policy source contract is invalid: {exc}"
        ) from exc


def build_tierd_tracker_policy_origin(
    *,
    robot: str,
    clip_id: str,
    tier_d: dict[str, Any],
    certificate_sha256: str,
    checkpoint_policy_contract_sidecar: dict[str, Any],
    checkpoint_policy_contract_sidecar_sha256: str,
    certification_config_sha256: str,
    reward_module_sha256: str,
    portable_actor_safetensors_sha256: str,
) -> dict[str, Any]:
    """Build and self-validate the inert provenance for one actor export."""
    normalized_robot, normalized_clip = _require_identity(robot, clip_id)
    sidecar = _json_clone(checkpoint_policy_contract_sidecar)
    tier_d_copy = _json_clone(tier_d)
    from sculptor.refs.track import TIER_D_CERTIFICATE_SCHEMA

    origin = {
        "schema": TIERD_TRACKER_POLICY_ORIGIN_SCHEMA,
        "kind": TIERD_TRACKER_POLICY_ORIGIN_KIND,
        "robot": normalized_robot,
        "clip_id": normalized_clip,
        "capabilities": _json_clone(_CAPABILITIES),
        "source_artifacts": {
            "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
            "checkpoint_policy_contract_sidecar_sha256": (
                checkpoint_policy_contract_sidecar_sha256
            ),
            "certification_config_sha256": certification_config_sha256,
            "reward_module_sha256": reward_module_sha256,
            "portable_actor_safetensors_sha256": (
                portable_actor_safetensors_sha256
            ),
        },
        "checkpoint_policy_contract_sidecar": sidecar,
        "certificate": {
            "sha256": certificate_sha256,
            "payload": {
                "schema": TIER_D_CERTIFICATE_SCHEMA,
                "robot": normalized_robot,
                "clip_id": normalized_clip,
                "tierD": tier_d_copy,
            },
        },
    }
    validate_tierd_tracker_policy_origin(
        origin,
        source_contract=sidecar.get("policy_contract"),
        source_checkpoint_sha256=sidecar.get("checkpoint_sha256"),
        portable_actor_safetensors_sha256=portable_actor_safetensors_sha256,
    )
    return origin


def validate_tierd_tracker_policy_origin(
    origin: Any,
    *,
    source_contract: dict[str, Any],
    source_checkpoint_sha256: str,
    portable_actor_safetensors_sha256: str,
) -> dict[str, Any]:
    """Validate the complete source chain and return its compact summary.

    This proves internal byte/digest relationships retained by the portable
    artifact.  It intentionally does not re-open the source reference library
    or claim that its reference can execute in the importing project.
    """
    value = _require_exact_keys(origin, _ORIGIN_KEYS, label="tracker origin")
    if value.get("schema") != TIERD_TRACKER_POLICY_ORIGIN_SCHEMA:
        raise TierDTrackerPolicyError("tracker policy origin schema is unsupported")
    if value.get("kind") != TIERD_TRACKER_POLICY_ORIGIN_KIND:
        raise TierDTrackerPolicyError("tracker policy origin kind is unsupported")
    robot, clip_id = _require_identity(value.get("robot"), value.get("clip_id"))
    if value.get("capabilities") != _CAPABILITIES:
        raise TierDTrackerPolicyError(
            "tracker actor provenance may authorize actor-only initialization only"
        )

    expected_source_checkpoint = _require_sha256(
        source_checkpoint_sha256,
        label="source checkpoint sha256",
    )
    expected_contract_digest = _contract_digest(source_contract)
    source_artifacts = _require_exact_keys(
        value.get("source_artifacts"),
        _SOURCE_ARTIFACT_KEYS,
        label="tracker source_artifacts",
    )
    source_checkpoint = _require_sha256(
        source_artifacts.get("checkpoint_sha256"),
        label="tracker source checkpoint sha256",
    )
    sidecar_sha = _require_sha256(
        source_artifacts.get("checkpoint_policy_contract_sidecar_sha256"),
        label="tracker checkpoint policy-contract sidecar sha256",
    )
    certification_config_sha = _require_sha256(
        source_artifacts.get("certification_config_sha256"),
        label="tracker certification config sha256",
    )
    reward_sha = _require_sha256(
        source_artifacts.get("reward_module_sha256"),
        label="tracker reward module sha256",
    )
    portable_actor_sha = _require_sha256(
        source_artifacts.get("portable_actor_safetensors_sha256"),
        label="portable tracker actor safetensors sha256",
    )
    if source_checkpoint != expected_source_checkpoint:
        raise TierDTrackerPolicyError(
            "tracker provenance checkpoint differs from the portable source checkpoint"
        )
    if portable_actor_sha != _require_sha256(
        portable_actor_safetensors_sha256,
        label="expected portable tracker actor safetensors sha256",
    ):
        raise TierDTrackerPolicyError(
            "tracker provenance does not bind the admitted portable actor bytes"
        )

    sidecar = _require_exact_keys(
        value.get("checkpoint_policy_contract_sidecar"),
        _CHECKPOINT_SIDECAR_KEYS,
        label="tracker checkpoint policy-contract sidecar",
    )
    sidecar_contract = sidecar.get("policy_contract")
    if (
        sidecar.get("schema") != 1
        or sidecar.get("checkpoint_sha256") != source_checkpoint
        or sidecar_contract != source_contract
        or sidecar.get("policy_contract_sha256") != expected_contract_digest
        or _contract_digest(sidecar_contract) != expected_contract_digest
    ):
        raise TierDTrackerPolicyError(
            "tracker checkpoint sidecar does not bind the exported policy interface"
        )

    certificate = _require_exact_keys(
        value.get("certificate"),
        _CERTIFICATE_KEYS,
        label="tracker Tier-D certificate",
    )
    certificate_sha = _require_sha256(
        certificate.get("sha256"), label="Tier-D certificate sha256"
    )
    certificate_payload = _require_exact_keys(
        certificate.get("payload"),
        _CERTIFICATE_PAYLOAD_KEYS,
        label="tracker Tier-D certificate payload",
    )
    from sculptor.refs.track import (
        TIER_D_CERTIFICATE_SCHEMA,
        TIER_D_CERTIFICATION_SCOPE,
        validate_tierd_execution_contract,
    )

    if (
        certificate_payload.get("schema") != TIER_D_CERTIFICATE_SCHEMA
        or certificate_payload.get("robot") != robot
        or certificate_payload.get("clip_id") != clip_id
    ):
        raise TierDTrackerPolicyError(
            "tracker Tier-D certificate identity/schema is inconsistent"
        )
    if hashlib.sha256(canonical_json_bytes(certificate_payload)).hexdigest() != (
        certificate_sha
    ):
        raise TierDTrackerPolicyError("tracker Tier-D certificate digest mismatch")

    tier_d = certificate_payload.get("tierD")
    if not isinstance(tier_d, dict):
        raise TierDTrackerPolicyError("tracker Tier-D certificate block is missing")
    errors = tier_d.get("errors")
    if (
        tier_d.get("feasible") is False
        or not isinstance(errors, dict)
        or errors.get("feasible") is not True
        or errors.get("certification_scope") != TIER_D_CERTIFICATION_SCOPE
    ):
        raise TierDTrackerPolicyError(
            "tracker source did not retain a successful Tier-D verdict"
        )
    execution_contract = tier_d.get("execution_contract")
    contract_issues = validate_tierd_execution_contract(execution_contract)
    if contract_issues:
        raise TierDTrackerPolicyError(
            "tracker Tier-D execution contract is invalid: "
            + "; ".join(contract_issues)
        )
    assert isinstance(execution_contract, dict)
    execution_contract_sha = _require_sha256(
        tier_d.get("execution_contract_sha256"),
        label="Tier-D execution contract sha256",
    )
    execution_boundary_sha = _require_sha256(
        tier_d.get("execution_boundary_sha256"),
        label="Tier-D execution boundary sha256",
    )
    if (
        execution_contract.get("contract_sha256") != execution_contract_sha
        or execution_contract.get("execution_boundary_sha256") != execution_boundary_sha
    ):
        raise TierDTrackerPolicyError(
            "tracker Tier-D certificate and execution-contract digests differ"
        )

    donor = execution_contract.get("donor")
    runtime = execution_contract.get("runtime_artifacts")
    reference = execution_contract.get("reference")
    boundary = execution_contract.get("execution_boundary")
    if not all(
        isinstance(item, dict) for item in (donor, runtime, reference, boundary)
    ):
        raise TierDTrackerPolicyError(
            "tracker execution contract omits donor/runtime/reference evidence"
        )
    assert isinstance(donor, dict)
    assert isinstance(runtime, dict)
    assert isinstance(reference, dict)
    assert isinstance(boundary, dict)
    if (
        boundary.get("robot") != robot
        or reference.get("clip_id") != clip_id
        or donor.get("policy_contract") != source_contract
        or donor.get("policy_contract_sha256") != expected_contract_digest
        or donor.get("certification_config_sha256") != certification_config_sha
        or runtime.get("requested_reward_module_sha256") != reward_sha
        or runtime.get("final_checkpoint_sha256") != source_checkpoint
    ):
        raise TierDTrackerPolicyError(
            "tracker source artifacts differ from the certified execution contract"
        )
    observations = runtime.get("train_observations")
    if not isinstance(observations, list) or not observations:
        raise TierDTrackerPolicyError(
            "tracker execution contract has no completed training observations"
        )
    last_observation = observations[-1]
    if (
        not isinstance(last_observation, dict)
        or last_observation.get("output_checkpoint_sha256") != source_checkpoint
        or last_observation.get("output_policy_contract_sha256")
        != expected_contract_digest
        or last_observation.get("output_policy_contract_sidecar_sha256") != sidecar_sha
    ):
        raise TierDTrackerPolicyError(
            "tracker checkpoint/contract sidecar is not the certified final output"
        )
    requested_training = runtime.get("requested_training")
    iterations = tier_d.get("iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
        or not isinstance(requested_training, dict)
        or requested_training.get("iterations") != iterations
        or len(observations) != iterations
    ):
        raise TierDTrackerPolicyError(
            "tracker iteration evidence is incomplete or contradictory"
        )
    tracked_at = tier_d.get("tracked_at")
    if not isinstance(tracked_at, str) or not tracked_at.strip():
        raise TierDTrackerPolicyError("tracker certificate has no tracked_at time")

    clip_sha = _require_sha256(
        tier_d.get("clip_content_sha256"),
        label="Tier-D reference clip sha256",
    )
    rollout_sha = _require_sha256(
        tier_d.get("rollout_sha256"), label="Tier-D rollout sha256"
    )
    summary = {
        "schema": TIERD_TRACKER_POLICY_SUMMARY_SCHEMA,
        "kind": TIERD_TRACKER_POLICY_ORIGIN_KIND,
        "robot": robot,
        "clip_id": clip_id,
        **_json_clone(_CAPABILITIES),
        "source_checkpoint_sha256": source_checkpoint,
        "source_policy_contract_sha256": expected_contract_digest,
        "portable_actor_safetensors_sha256": portable_actor_sha,
        "checkpoint_policy_contract_sidecar_sha256": sidecar_sha,
        "certificate_sha256": certificate_sha,
        "reference_clip_sha256": clip_sha,
        "rollout_sha256": rollout_sha,
        "execution_contract_sha256": execution_contract_sha,
        "execution_boundary_sha256": execution_boundary_sha,
        "tracker_iterations": iterations,
        "tracked_at": tracked_at,
    }
    return validate_tierd_tracker_policy_summary(
        summary,
        source_contract=source_contract,
        source_checkpoint_sha256=source_checkpoint,
        portable_actor_safetensors_sha256=portable_actor_sha,
    )


def validate_tierd_tracker_policy_summary(
    summary: Any,
    *,
    source_contract: dict[str, Any],
    source_checkpoint_sha256: str,
    portable_actor_safetensors_sha256: str,
) -> dict[str, Any]:
    """Validate the compact metadata copy stored in a skill record."""
    value = _require_exact_keys(
        summary, _SUMMARY_KEYS, label="tracker policy provenance summary"
    )
    if (
        value.get("schema") != TIERD_TRACKER_POLICY_SUMMARY_SCHEMA
        or value.get("kind") != TIERD_TRACKER_POLICY_ORIGIN_KIND
    ):
        raise TierDTrackerPolicyError(
            "tracker policy provenance summary schema/kind is unsupported"
        )
    _require_identity(value.get("robot"), value.get("clip_id"))
    for key, expected in _CAPABILITIES.items():
        if value.get(key) != expected:
            raise TierDTrackerPolicyError(
                "tracker policy summary may authorize actor-only initialization only"
            )
    if value.get("source_checkpoint_sha256") != _require_sha256(
        source_checkpoint_sha256, label="source checkpoint sha256"
    ):
        raise TierDTrackerPolicyError(
            "tracker policy summary checkpoint differs from source checkpoint"
        )
    if value.get("source_policy_contract_sha256") != _contract_digest(source_contract):
        raise TierDTrackerPolicyError(
            "tracker policy summary contract differs from source contract"
        )
    if value.get("portable_actor_safetensors_sha256") != _require_sha256(
        portable_actor_safetensors_sha256,
        label="portable tracker actor safetensors sha256",
    ):
        raise TierDTrackerPolicyError(
            "tracker policy summary differs from the admitted portable actor bytes"
        )
    for key in (
        "source_checkpoint_sha256",
        "source_policy_contract_sha256",
        "portable_actor_safetensors_sha256",
        "checkpoint_policy_contract_sidecar_sha256",
        "certificate_sha256",
        "reference_clip_sha256",
        "rollout_sha256",
        "execution_contract_sha256",
        "execution_boundary_sha256",
    ):
        _require_sha256(value.get(key), label=key)
    iterations = value.get("tracker_iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise TierDTrackerPolicyError(
            "tracker policy summary iterations must be positive"
        )
    tracked_at = value.get("tracked_at")
    if not isinstance(tracked_at, str) or not tracked_at.strip():
        raise TierDTrackerPolicyError(
            "tracker policy summary tracked_at must be non-empty"
        )
    return _json_clone(value)


__all__ = [
    "TIERD_TRACKER_POLICY_ORIGIN_KIND",
    "TIERD_TRACKER_POLICY_ORIGIN_MEMBER",
    "TIERD_TRACKER_POLICY_ORIGIN_SCHEMA",
    "TIERD_TRACKER_POLICY_SUMMARY_SCHEMA",
    "TierDTrackerPolicyError",
    "build_tierd_tracker_policy_origin",
    "canonical_json_bytes",
    "parse_tierd_tracker_policy_origin",
    "validate_tierd_tracker_policy_origin",
    "validate_tierd_tracker_policy_summary",
]
