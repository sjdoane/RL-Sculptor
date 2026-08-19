from __future__ import annotations

import hashlib
import json

import pytest

from sculptor.compatibility_provenance import (
    LEGACY_CONFIG_MEMBER,
    LEGACY_LOG_MEMBER,
    LEGACY_OBSERVED_SELECTION_MEMBER,
    LEGACY_RECONSTRUCTED,
    LEGACY_SOURCE_SELECTION_MEMBER,
    CompatibilityProvenanceError,
    build_legacy_reconstructed_provenance,
    validate_compatibility_contract_provenance,
)


def _contract() -> dict:
    term = {"name": "q", "source": "q", "shape": [4]}
    policy = {
        "class_name": "MLP",
        "hidden_dims": [8],
        "activation": "elu",
        "recurrent": {"type": None, "hidden_dim": 0, "num_layers": 0},
    }
    return {
        "schema": 1,
        "identity": {"adapter_class": "A", "task_id": "T"},
        "joints": {"ordered_names": ["j0", "j1"]},
        "observations": {
            "ordered_terms": [term],
            "shape": [4],
            "critic_ordered_terms": [term],
            "critic_shape": [4],
        },
        "actions": {
            "ordered_names": ["j0", "j1"],
            "term_names": ["joint_pos"],
            "shape": [2],
        },
        "policy": {
            "actor": policy,
            "critic": policy,
            "normalizer": {
                "present": False,
                "actor_present": False,
                "critic_present": False,
                "actor_shape": None,
                "critic_shape": None,
            },
        },
        "timing": {
            "sim_timestep_s": 0.002,
            "decimation": 10,
            "control_dt_s": 0.02,
        },
        "versions": {},
    }


def _selection(version: int) -> bytes:
    return json.dumps({
        "selection_version": version,
        "tuple_hash": "ced657" + "0" * 58,
        "evaluation_lineage": "iter38-source",
        "refs": {
            "reward": {"version": "v20", "sha256": "1" * 64},
            "env_spec": {"version": "v21", "sha256": "2" * 64},
        },
    }, sort_keys=True).encode("utf-8")


def _origin_log(config: bytes, *, action_width: int = 2) -> bytes:
    events = [
        {
            "type": "run_context_captured",
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "code_sha": "614b1fc" + "0" * 33,
            "code_dirty": False,
        },
        {
            "type": "authored_world_pinned",
            "selection": "selection_v47.json",
            "tuple_hash": "ced657" + "0" * 58,
        },
        {
            "type": "artifact_tuple_pinned",
            "iter": 38,
            "selection": "selection_v48.json",
            "tuple_hash": "ced657" + "0" * 58,
        },
        {"type": "iter_started", "iter": 38},
    ]
    lines = [
        "| Physics step-size | 0.002 |",
        "| Environment step-size | 0.02 |",
        "Active Action Terms (shape: 2)",
        f"| 0 | joint_pos | ({action_width},) |",
        "+---+---+---+",
        "Active Observation Terms in Group: 'actor' (shape: (4,))",
        "| 0 | q | (4,) |",
        "+---+---+---+",
        "Active Observation Terms in Group: 'critic' (shape: (4,))",
        "| 0 | q | (4,) |",
        "+---+---+---+",
        "Actor Model:",
        "  (0): Linear(in_features=4, out_features=8, bias=True)",
        "  (1): ELU(alpha=1.0)",
        "  (2): Linear(in_features=8, out_features=2, bias=True)",
        "Critic Model:",
        "  (0): Linear(in_features=4, out_features=8, bias=True)",
        "  (1): ELU(alpha=1.0)",
        "  (2): Linear(in_features=8, out_features=1, bias=True)",
    ]
    lines.extend(
        "[SCULPT-EVENT] " + json.dumps(event, sort_keys=True)
        for event in events
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _evidence() -> tuple[dict, dict[str, bytes]]:
    contract = _contract()
    config = b'[adapter]\nclass = "A"\nconfig = { task_id = "T" }\n'
    payloads = {
        LEGACY_LOG_MEMBER: _origin_log(config),
        LEGACY_CONFIG_MEMBER: config,
        LEGACY_SOURCE_SELECTION_MEMBER: _selection(47),
        LEGACY_OBSERVED_SELECTION_MEMBER: _selection(48),
    }
    provenance = build_legacy_reconstructed_provenance(
        origin_log=payloads[LEGACY_LOG_MEMBER],
        source_config=payloads[LEGACY_CONFIG_MEMBER],
        source_selection=payloads[LEGACY_SOURCE_SELECTION_MEMBER],
        observed_selection=payloads[LEGACY_OBSERVED_SELECTION_MEMBER],
        contract=contract,
        policy_roles=["actor", "critic"],
        iter_index=38,
    )
    return provenance, payloads


def test_legacy_reconstruction_rederives_runtime_and_material_selection_alias() -> None:
    contract = _contract()
    provenance, payloads = _evidence()

    status = validate_compatibility_contract_provenance(
        provenance,
        contract=contract,
        policy_roles=["actor", "critic"],
        iter_index=38,
        read_member=payloads.__getitem__,
    )

    assert status == LEGACY_RECONSTRUCTED
    reconstruction = provenance["reconstruction"]
    assert reconstruction["source_selection_version"] == 47
    assert reconstruction["observed_selection_version"] == 48
    assert len(reconstruction["runtime_interface_sha256"]) == 64
    assert provenance["capabilities"] == {
        "initialization_modes": ["actor_only", "actor_critic"],
        "optimizer_resume": False,
        "exact_resume": False,
    }


def test_legacy_reconstruction_rejects_material_selection_change() -> None:
    contract = _contract()
    provenance, payloads = _evidence()
    changed = json.loads(payloads[LEGACY_OBSERVED_SELECTION_MEMBER])
    changed["refs"]["env_spec"]["sha256"] = "f" * 64
    payloads[LEGACY_OBSERVED_SELECTION_MEMBER] = json.dumps(changed).encode()

    with pytest.raises(
        CompatibilityProvenanceError,
        match="changes material world identity|descriptor mismatch",
    ):
        validate_compatibility_contract_provenance(
            provenance,
            contract=contract,
            policy_roles=["actor", "critic"],
            iter_index=38,
            read_member=payloads.__getitem__,
        )


def test_legacy_reconstruction_rejects_runtime_interface_drift() -> None:
    contract = _contract()
    config = b'[adapter]\nclass = "A"\nconfig = { task_id = "T" }\n'

    with pytest.raises(
        CompatibilityProvenanceError,
        match="runtime action rows do not match",
    ):
        build_legacy_reconstructed_provenance(
            origin_log=_origin_log(config, action_width=3),
            source_config=config,
            source_selection=_selection(47),
            observed_selection=_selection(48),
            contract=contract,
            policy_roles=["actor", "critic"],
            iter_index=38,
        )
