"""Fail-closed provenance for portable policy compatibility contracts.

Portable policy weights are useful only when the interface contract beside
them is itself attributable.  New runs persist the exact contract at training
time.  A narrow historical escape hatch can reconstruct one from retained,
immutable runtime evidence, but it is intentionally less trusted and can only
initialize actor/critic weights.

This module is deliberately pure and data-only.  It never imports a policy,
executes an uploaded controller, or treats a historical reconstruction as an
optimizer/exact resume.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any


PROVENANCE_SCHEMA = 1
ORIGIN_PERSISTED = "origin_persisted"
LEGACY_RECONSTRUCTED = "legacy_reconstructed"

ORIGIN_CONTRACT_MEMBER = "provenance/origin_policy_contract.json"
LEGACY_LOG_MEMBER = "provenance/origin_job.log"
LEGACY_CONFIG_MEMBER = "provenance/source_config.toml"
LEGACY_SOURCE_SELECTION_MEMBER = "provenance/selection_source.json"
LEGACY_OBSERVED_SELECTION_MEMBER = "provenance/selection_observed.json"


class CompatibilityProvenanceError(ValueError):
    """A policy-contract provenance claim cannot be verified."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def provenance_fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _descriptor(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _sha256(payload), "bytes": len(payload)}


def _capabilities(policy_roles: list[str]) -> dict[str, Any]:
    if policy_roles == ["actor"]:
        modes = ["actor_only"]
    elif policy_roles == ["actor", "critic"]:
        modes = ["actor_only", "actor_critic"]
    else:
        raise CompatibilityProvenanceError(
            "contract provenance supports actor or actor+critic weights only"
        )
    return {
        "initialization_modes": modes,
        "optimizer_resume": False,
        "exact_resume": False,
    }


def build_origin_persisted_provenance(
    *, contract_bytes: bytes, policy_roles: list[str]
) -> dict[str, Any]:
    """Describe an exact contract sidecar retained when training ran."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "status": ORIGIN_PERSISTED,
        "capabilities": _capabilities(policy_roles),
        "evidence": {
            "origin_policy_contract": _descriptor(
                ORIGIN_CONTRACT_MEMBER, contract_bytes
            )
        },
    }


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CompatibilityProvenanceError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except CompatibilityProvenanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityProvenanceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CompatibilityProvenanceError(f"{label} must be a JSON object")
    return value


def _shape(text: str) -> list[int]:
    values = [int(item) for item in re.findall(r"\d+", text)]
    if not values:
        raise CompatibilityProvenanceError(
            f"runtime interface row has no parseable shape: {text!r}"
        )
    return values


def _table_rows(lines: list[str], start: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*(\([^|]+\)|\d+)\s*\|$"
    )
    for line in lines[start:]:
        match = row_re.match(line.strip())
        if match:
            rows.append({"name": match.group(2), "shape": _shape(match.group(3))})
        elif rows and line.strip().startswith("+"):
            break
    return rows


def _parse_model(lines: list[str], label: str) -> dict[str, Any]:
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(label))
    except StopIteration as exc:
        raise CompatibilityProvenanceError(f"origin log has no {label}") from exc
    end = len(lines)
    for index in range(start + 1, min(len(lines), start + 40)):
        if lines[index].startswith(("Actor Model:", "Critic Model:", "[SCULPT-EVENT]")):
            end = index
            break
    block = lines[start:end]
    linear = [
        (int(a), int(b))
        for a, b in re.findall(
            r"Linear\(in_features=(\d+), out_features=(\d+), bias=True\)",
            "\n".join(block),
        )
    ]
    if len(linear) < 2:
        raise CompatibilityProvenanceError(f"{label} has no complete MLP topology")
    activation_matches = re.findall(r"\)\:\s*([A-Za-z]+)\(", "\n".join(block))
    known_activations = {
        "ELU", "GELU", "LeakyReLU", "ReLU", "SELU", "SiLU", "Tanh",
    }
    activations = [
        value.lower() for value in activation_matches if value in known_activations
    ]
    if not activations or len(set(activations)) != 1:
        raise CompatibilityProvenanceError(f"{label} activation is ambiguous")
    return {
        "input_dim": linear[0][0],
        "hidden_dims": [out_features for _, out_features in linear[:-1]],
        "output_dim": linear[-1][1],
        "activation": activations[0],
        "normalizer_present": any("EmpiricalNormalization" in line for line in block),
    }


def parse_runtime_interface(origin_log: bytes) -> dict[str, Any]:
    """Recover the exact interface rows printed by the historical worker."""
    try:
        text = origin_log.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CompatibilityProvenanceError("origin job log is not UTF-8") from exc
    lines = text.splitlines()

    action_header = re.compile(r"Active Action Terms \(shape:\s*(\d+)\)")
    observation_header = re.compile(
        r"Active Observation Terms in Group: '([^']+)' \(shape:\s*\((\d+),?\)\)"
    )
    actions: dict[str, Any] | None = None
    observations: dict[str, Any] = {}
    for index, line in enumerate(lines):
        action_match = action_header.search(line)
        if action_match and actions is None:
            actions = {
                "shape": [int(action_match.group(1))],
                "terms": _table_rows(lines, index + 1),
            }
        observation_match = observation_header.search(line)
        if observation_match and observation_match.group(1) not in observations:
            observations[observation_match.group(1)] = {
                "shape": [int(observation_match.group(2))],
                "terms": _table_rows(lines, index + 1),
            }
    if actions is None or set(observations) != {"actor", "critic"}:
        raise CompatibilityProvenanceError(
            "origin log lacks actor, critic, or action runtime interface rows"
        )

    timing: dict[str, float] = {}
    timing_labels = {
        "Physics step-size": "sim_timestep_s",
        "Environment step-size": "control_dt_s",
    }
    for line in lines:
        for printed, key in timing_labels.items():
            match = re.match(
                rf"^\|\s*{re.escape(printed)}\s*\|\s*([0-9.]+)\s*\|$",
                line.strip(),
            )
            if match and key not in timing:
                timing[key] = float(match.group(1))
    if set(timing) != {"sim_timestep_s", "control_dt_s"}:
        raise CompatibilityProvenanceError("origin log lacks controller timing rows")
    ratio = timing["control_dt_s"] / timing["sim_timestep_s"]
    if abs(ratio - round(ratio)) > 1e-9:
        raise CompatibilityProvenanceError("runtime timing has non-integral decimation")
    timing["decimation"] = int(round(ratio))

    events: list[dict[str, Any]] = []
    marker = "[SCULPT-EVENT] "
    for line in lines:
        if marker not in line:
            continue
        event = _parse_json_object(
            line.split(marker, 1)[1].encode("utf-8"),
            label="origin log SCULPT-EVENT row",
        )
        events.append(event)

    return {
        "actions": actions,
        "observations": observations,
        "policy": {
            "actor": _parse_model(lines, "Actor Model:"),
            "critic": _parse_model(lines, "Critic Model:"),
        },
        "timing": timing,
        "events": events,
    }


def _event(
    interface: dict[str, Any], event_type: str, *, iter_index: int | None = None
) -> dict[str, Any]:
    matches = [
        event
        for event in interface["events"]
        if event.get("type") == event_type
        and (iter_index is None or event.get("iter") == iter_index)
    ]
    if len(matches) != 1:
        raise CompatibilityProvenanceError(
            f"origin log must contain exactly one {event_type!r} event"
        )
    return matches[0]


def _material_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "tuple_hash": selection.get("tuple_hash"),
        "evaluation_lineage": selection.get("evaluation_lineage"),
        "refs": selection.get("refs"),
    }


def _validate_selection(selection: dict[str, Any], *, label: str) -> None:
    version = selection.get("selection_version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise CompatibilityProvenanceError(f"{label} has no positive selection_version")
    if not isinstance(selection.get("tuple_hash"), str):
        raise CompatibilityProvenanceError(f"{label} has no tuple_hash")
    if not isinstance(selection.get("refs"), dict):
        raise CompatibilityProvenanceError(f"{label} has no refs object")


def _contract_terms(contract: dict[str, Any], group: str) -> dict[str, Any]:
    observations = contract.get("observations") or {}
    if group == "actor":
        terms = observations.get("ordered_terms")
        shape = observations.get("shape")
    else:
        terms = observations.get("critic_ordered_terms")
        shape = observations.get("critic_shape")
    return {
        "shape": shape,
        "terms": [
            {"name": term.get("name"), "shape": term.get("shape")}
            for term in terms or []
        ],
    }


def _validate_interface_against_contract(
    interface: dict[str, Any], contract: dict[str, Any]
) -> None:
    for group in ("actor", "critic"):
        if interface["observations"][group] != _contract_terms(contract, group):
            raise CompatibilityProvenanceError(
                f"runtime {group} observation rows do not match the policy contract"
            )
    actions = contract.get("actions") or {}
    expected_actions = {
        "shape": actions.get("shape"),
        "terms": [
            {"name": name, "shape": actions.get("shape")}
            for name in actions.get("term_names") or []
        ],
    }
    if interface["actions"] != expected_actions:
        raise CompatibilityProvenanceError(
            "runtime action rows do not match the policy contract"
        )
    policy = contract.get("policy") or {}
    normalizer = policy.get("normalizer") or {}
    for role, output_dim in (("actor", actions.get("shape", [None])[0]), ("critic", 1)):
        declared = policy.get(role) or {}
        runtime = interface["policy"][role]
        expected = {
            "input_dim": _contract_terms(contract, role)["shape"][0],
            "hidden_dims": declared.get("hidden_dims"),
            "output_dim": output_dim,
            "activation": declared.get("activation"),
            "normalizer_present": bool(normalizer.get(f"{role}_present")),
        }
        if runtime != expected:
            raise CompatibilityProvenanceError(
                f"runtime {role} topology does not match the policy contract"
            )
    if interface["timing"] != (contract.get("timing") or {}):
        raise CompatibilityProvenanceError(
            "runtime controller timing does not match the policy contract"
        )


def reconstruct_legacy_evidence(
    *,
    origin_log: bytes,
    source_config: bytes,
    source_selection: bytes,
    observed_selection: bytes,
    contract: dict[str, Any],
    iter_index: int,
) -> dict[str, Any]:
    interface = parse_runtime_interface(origin_log)
    _validate_interface_against_contract(interface, contract)

    try:
        config = tomllib.loads(source_config.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CompatibilityProvenanceError("source config is not valid TOML") from exc
    identity = contract.get("identity") or {}
    adapter = config.get("adapter") or {}
    adapter_config = adapter.get("config") or {}
    if (
        adapter.get("class") != identity.get("adapter_class")
        or adapter_config.get("task_id") != identity.get("task_id")
    ):
        raise CompatibilityProvenanceError(
            "source config adapter/task identity does not match the contract"
        )

    source = _parse_json_object(source_selection, label="source selection")
    observed = _parse_json_object(observed_selection, label="observed selection")
    _validate_selection(source, label="source selection")
    _validate_selection(observed, label="observed selection")
    if _material_selection(source) != _material_selection(observed):
        raise CompatibilityProvenanceError(
            "historical selection alias changes material world identity"
        )

    context = _event(interface, "run_context_captured")
    authored = _event(interface, "authored_world_pinned")
    observed_event = _event(interface, "artifact_tuple_pinned", iter_index=iter_index)
    started = _event(interface, "iter_started", iter_index=iter_index)
    if context.get("config_sha256") != _sha256(source_config):
        raise CompatibilityProvenanceError(
            "source config digest does not match run_context_captured"
        )
    expected_source_name = f"selection_v{source['selection_version']}.json"
    expected_observed_name = f"selection_v{observed['selection_version']}.json"
    if (
        authored.get("selection") != expected_source_name
        or authored.get("tuple_hash") != source.get("tuple_hash")
        or observed_event.get("selection") != expected_observed_name
        or observed_event.get("tuple_hash") != observed.get("tuple_hash")
    ):
        raise CompatibilityProvenanceError(
            "origin log selection events do not match retained selection bytes"
        )
    if started.get("iter") != iter_index:
        raise CompatibilityProvenanceError("origin log does not prove source iteration")

    interface_without_events = dict(interface)
    interface_without_events.pop("events", None)
    return {
        "iter_index": iter_index,
        "code_sha": context.get("code_sha"),
        "code_dirty": context.get("code_dirty"),
        "config_sha256": context.get("config_sha256"),
        "source_selection_version": source.get("selection_version"),
        "observed_selection_version": observed.get("selection_version"),
        "material_tuple_hash": source.get("tuple_hash"),
        "runtime_interface": interface_without_events,
        "runtime_interface_sha256": provenance_fingerprint(interface_without_events),
    }


def build_legacy_reconstructed_provenance(
    *,
    origin_log: bytes,
    source_config: bytes,
    source_selection: bytes,
    observed_selection: bytes,
    contract: dict[str, Any],
    policy_roles: list[str],
    iter_index: int,
) -> dict[str, Any]:
    reconstruction = reconstruct_legacy_evidence(
        origin_log=origin_log,
        source_config=source_config,
        source_selection=source_selection,
        observed_selection=observed_selection,
        contract=contract,
        iter_index=iter_index,
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "status": LEGACY_RECONSTRUCTED,
        "capabilities": _capabilities(policy_roles),
        "evidence": {
            "origin_job_log": _descriptor(LEGACY_LOG_MEMBER, origin_log),
            "source_config": _descriptor(LEGACY_CONFIG_MEMBER, source_config),
            "source_selection": _descriptor(
                LEGACY_SOURCE_SELECTION_MEMBER, source_selection
            ),
            "observed_selection": _descriptor(
                LEGACY_OBSERVED_SELECTION_MEMBER, observed_selection
            ),
        },
        "reconstruction": reconstruction,
    }


def _validate_evidence_descriptor(
    raw: Any,
    *,
    expected_path: str,
    read_member: Callable[[str], bytes],
) -> bytes:
    if not isinstance(raw, dict) or raw.get("path") != expected_path:
        raise CompatibilityProvenanceError(
            f"contract provenance must retain {expected_path}"
        )
    try:
        payload = read_member(expected_path)
    except (KeyError, OSError) as exc:
        raise CompatibilityProvenanceError(
            f"contract provenance evidence is missing: {expected_path}"
        ) from exc
    if raw != _descriptor(expected_path, payload):
        raise CompatibilityProvenanceError(
            f"contract provenance descriptor mismatch: {expected_path}"
        )
    return payload


def validate_compatibility_contract_provenance(
    provenance: Any,
    *,
    contract: dict[str, Any],
    policy_roles: list[str],
    iter_index: int,
    read_member: Callable[[str], bytes],
) -> str:
    """Re-derive a provenance claim and return its status.

    All authoritative facts come from retained bytes.  The manifest's
    reconstruction summary is treated as a claim and must exactly match the
    independently parsed evidence.
    """
    if not isinstance(provenance, dict):
        raise CompatibilityProvenanceError(
            "trainable bundle is missing compatibility_contract_provenance"
        )
    status = provenance.get("status")
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise CompatibilityProvenanceError(
            "compatibility contract provenance schema is unsupported"
        )
    if provenance.get("capabilities") != _capabilities(policy_roles):
        raise CompatibilityProvenanceError(
            "contract provenance capabilities must be actor/critic initialization only"
        )
    evidence = provenance.get("evidence")
    if not isinstance(evidence, dict):
        raise CompatibilityProvenanceError("contract provenance evidence must be an object")

    if status == ORIGIN_PERSISTED:
        if set(evidence) != {"origin_policy_contract"} or "reconstruction" in provenance:
            raise CompatibilityProvenanceError(
                "origin_persisted provenance has unexpected evidence"
            )
        contract_bytes = _validate_evidence_descriptor(
            evidence.get("origin_policy_contract"),
            expected_path=ORIGIN_CONTRACT_MEMBER,
            read_member=read_member,
        )
        retained_contract = _parse_json_object(
            contract_bytes, label="origin policy contract"
        )
        if canonical_json_bytes(retained_contract) != canonical_json_bytes(contract):
            raise CompatibilityProvenanceError(
                "origin policy contract bytes do not match compatibility_contract"
            )
        return status

    if status != LEGACY_RECONSTRUCTED:
        raise CompatibilityProvenanceError(
            f"unsupported compatibility contract provenance status: {status!r}"
        )
    expected = {
        "origin_job_log": LEGACY_LOG_MEMBER,
        "source_config": LEGACY_CONFIG_MEMBER,
        "source_selection": LEGACY_SOURCE_SELECTION_MEMBER,
        "observed_selection": LEGACY_OBSERVED_SELECTION_MEMBER,
    }
    if set(evidence) != set(expected):
        raise CompatibilityProvenanceError(
            "legacy reconstruction evidence set is incomplete"
        )
    payloads = {
        role: _validate_evidence_descriptor(
            evidence.get(role), expected_path=path, read_member=read_member
        )
        for role, path in expected.items()
    }
    reconstructed = reconstruct_legacy_evidence(
        origin_log=payloads["origin_job_log"],
        source_config=payloads["source_config"],
        source_selection=payloads["source_selection"],
        observed_selection=payloads["observed_selection"],
        contract=contract,
        iter_index=iter_index,
    )
    if provenance.get("reconstruction") != reconstructed:
        raise CompatibilityProvenanceError(
            "legacy reconstruction summary does not match retained evidence"
        )
    return status


def evidence_member_paths(provenance: dict[str, Any]) -> list[str]:
    evidence = provenance.get("evidence") or {}
    paths: list[str] = []
    for row in evidence.values():
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            path = PurePosixPath(row["path"])
            if path.parts and path.parts[0] == "provenance":
                paths.append(path.as_posix())
    return sorted(paths)


def build_launch_acknowledgement_receipt(
    *,
    status: str | None,
    provenance_digest: str | None,
    acknowledged: bool,
    initialization_mode: str,
) -> dict[str, Any]:
    """Build the route/worker-stable receipt for provenance risk consent."""
    if status not in {ORIGIN_PERSISTED, LEGACY_RECONSTRUCTED}:
        raise CompatibilityProvenanceError(
            "policy initialization has no admitted contract provenance status"
        )
    if not isinstance(provenance_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance_digest
    ):
        raise CompatibilityProvenanceError(
            "policy initialization has no contract provenance digest"
        )
    if initialization_mode not in {"actor_only", "actor_critic"}:
        raise CompatibilityProvenanceError(
            "portable policy provenance permits actor/critic initialization only"
        )
    if status == LEGACY_RECONSTRUCTED and not acknowledged:
        raise CompatibilityProvenanceError(
            "legacy reconstructed compatibility requires explicit acknowledgement"
        )
    if status == ORIGIN_PERSISTED and acknowledged:
        raise CompatibilityProvenanceError(
            "legacy acknowledgement was supplied for an origin-persisted contract"
        )
    return {
        "schema": 1,
        "status": status,
        "provenance_digest": provenance_digest,
        "acknowledged": acknowledged,
        "initialization_mode": initialization_mode,
        "optimizer_resume": False,
        "exact_resume": False,
    }


__all__ = [
    "CompatibilityProvenanceError",
    "LEGACY_CONFIG_MEMBER",
    "LEGACY_LOG_MEMBER",
    "LEGACY_OBSERVED_SELECTION_MEMBER",
    "LEGACY_RECONSTRUCTED",
    "LEGACY_SOURCE_SELECTION_MEMBER",
    "ORIGIN_CONTRACT_MEMBER",
    "ORIGIN_PERSISTED",
    "build_legacy_reconstructed_provenance",
    "build_launch_acknowledgement_receipt",
    "build_origin_persisted_provenance",
    "evidence_member_paths",
    "provenance_fingerprint",
    "validate_compatibility_contract_provenance",
]
