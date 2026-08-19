from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sculptor.policy_contract import (
    _shape_for_observation_term,
    build_iteration_warm_start_contract_receipt,
    build_skill_warm_start_contract_receipt,
    compare_policy_contracts,
    contract_fingerprint,
    policy_contract_migration,
)
from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore


def _legacy_contract() -> dict:
    terms = [{"name": "proprio", "source": "joint_pos", "shape": [4]}]
    return {
        "schema": 2,
        "identity": {"adapter_class": "MjlabAdapter", "task_id": "task"},
        "joints": {"ordered_names": ["j0", "j1"]},
        "observations": {
            "ordered_terms": copy.deepcopy(terms),
            "shape": [4],
            "critic_ordered_terms": copy.deepcopy(terms),
            "critic_shape": [4],
        },
        "actions": {
            "ordered_names": ["j0", "j1"],
            "term_names": ["joint_pos"],
            "shape": [2],
        },
        "policy": {
            "actor": {"class_name": "MlpModel", "hidden_dims": [8]},
            "critic": {"class_name": "MlpModel", "hidden_dims": [8]},
            "normalizer": {
                "present": True,
                "actor_present": True,
                "critic_present": True,
                "actor_shape": [4],
                "critic_shape": [4],
            },
        },
        "timing": {
            "sim_timestep_s": 0.002,
            "decimation": 10,
            "control_dt_s": 0.02,
        },
        "versions": {
            "torch": "2.7", "mjlab": "1", "rsl_rl": "2", "adapter": "1",
        },
    }


def _event_contract(source: dict) -> dict:
    target = copy.deepcopy(source)
    target["schema"] = 3
    target["event_observation"] = {
        "schema": 1,
        "term_name": "authored_event_phase",
        "encoding": "one_hot",
        "ordered_phase_ids": ["route", "jump", "hold"],
    }
    phase_term = {
        "name": "authored_event_phase",
        "source": "authored_event_phase_observation",
        "shape": [3],
    }
    observations = target["observations"]
    observations["ordered_terms"].append(copy.deepcopy(phase_term))
    observations["critic_ordered_terms"].append(copy.deepcopy(phase_term))
    observations["shape"] = [7]
    observations["critic_shape"] = [7]
    target["policy"]["normalizer"]["actor_shape"] = [7]
    target["policy"]["normalizer"]["critic_shape"] = [7]
    return target


def test_authored_region_relative_observation_has_explicit_xyz_shape() -> None:
    shape = _shape_for_observation_term(
        name="authored_region_finish",
        source="authored_region_relative_observation",
        params={"center_m": (8.0, 0.0, 0.0)},
        joint_count=29,
        action_dim=29,
        command_cfg={},
        env_cfg=object(),
    )
    assert shape == [3]
    assert _shape_for_observation_term(
        name="broken",
        source="authored_region_relative_observation",
        params={"center_m": (1.0, 2.0)},
        joint_count=29,
        action_dim=29,
        command_cfg={},
        env_cfg=object(),
    ) is None


def test_grid_height_scan_shape_uses_inclusive_raycast_grid() -> None:
    class Pattern:
        size = (1.6, 1.0)
        resolution = 0.2

    class Sensor:
        name = "authored_height_scan"
        pattern = Pattern()

    class Scene:
        sensors = (Sensor(),)

    class Env:
        scene = Scene()

    assert _shape_for_observation_term(
        name="authored_height_scan",
        source="height_scan",
        params={"sensor_name": "authored_height_scan"},
        joint_count=29,
        action_dim=29,
        command_cfg={},
        env_cfg=Env(),
    ) == [54]

    Sensor.pattern.size = (1.55, 1.0)
    assert _shape_for_observation_term(
        name="authored_height_scan",
        source="height_scan",
        params={"sensor_name": "authored_height_scan"},
        joint_count=29,
        action_dim=29,
        command_cfg={},
        env_cfg=Env(),
    ) is None


def test_schema2_policy_admits_only_exact_zero_initialized_event_extension() -> None:
    source = _legacy_contract()
    target = _event_contract(source)

    migration = policy_contract_migration(source, target)
    assert migration == {
        "type": "zero_initialized_event_phase_observation",
        "from_schema": 2,
        "to_schema": 3,
        "observation_term": "authored_event_phase",
        "extension_width": 3,
        "ordered_phase_ids": ["route", "jump", "hold"],
        "optimizer_resume": False,
    }
    assert compare_policy_contracts(source, target) == []
    assert "effective_world" not in target

    # A different selected world with the same structural observation
    # interface produces the same target contract; world tuple/lineage belong
    # to launch provenance, not policy compatibility.
    other_world_same_interface = copy.deepcopy(target)
    assert policy_contract_migration(
        source, other_world_same_interface) == migration


def test_event_extension_migration_fails_closed_on_any_extra_change() -> None:
    source = _legacy_contract()
    target = _event_contract(source)
    target["timing"]["control_dt_s"] = 0.04
    assert policy_contract_migration(source, target) is None
    assert compare_policy_contracts(source, target)

    target = _event_contract(source)
    target["observations"]["ordered_terms"].reverse()
    assert policy_contract_migration(source, target) is None


def test_skill_warm_start_receipt_pins_cross_project_event_migration() -> None:
    source = _legacy_contract()
    target = _event_contract(source)

    receipt = build_skill_warm_start_contract_receipt(
        skill_id="abc123def456",
        manifest_digest="a" * 64,
        checkpoint_sha256="b" * 64,
        tensor_signature_sha256="c" * 64,
        source_contract=source,
        target_contract=target,
        target_receipt={
            "schema": 1,
            "adapter_class": "MjlabAdapter",
            "task_id": "task",
            "robot_slug": "g1",
            "policy_contract_required": True,
            "policy_contract_sha256": contract_fingerprint(target),
        },
    )

    assert receipt["kind"] == "starting_skill"
    assert receipt["source"]["contract"] == source
    assert receipt["target"]["contract"] == target
    assert receipt["compatibility"]["type"] == (
        "zero_initialized_event_phase_observation"
    )
    assert receipt["compatibility"]["optimizer_resume"] is False


def _selection_pair(project_dir: Path) -> tuple[Path, Path]:
    store = WorldArtifactStore(project_dir)
    reward_path = project_dir / "rewards" / "v1.py"
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text("def reward():\n    return 0\n", encoding="utf-8")
    env_spec_path = project_dir / "env" / "v1.json"
    env_spec_path.write_text("{}\n", encoding="utf-8")
    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v1", reward_path, base=project_dir,
        ),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v1", env_spec_path, base=project_dir,
        ),
        "world": store.write_json("world", {"shared": {}}),
        "task": store.write_json("task", {"shared": {}}),
        "resolved_eval": store.write_json("resolved_eval", {}),
        "channel_catalog": store.write_json("channel_catalog", {}),
        "clarifications": store.write_json("clarifications", {}),
    }
    source = store.promote(refs, evaluation_lineage="source")
    source_path = project_dir / "env" / (
        f"selection_v{source.selection_version}.json"
    )
    refs["task"] = store.write_json(
        "task", {"shared": {"event_sequence": {"id": "event"}}}
    )
    target = store.promote(refs, evaluation_lineage="target")
    target_path = project_dir / "env" / (
        f"selection_v{target.selection_version}.json"
    )
    return source_path, target_path


def test_iteration_warm_start_receipt_binds_exact_source_and_target(
    tmp_path: Path, monkeypatch,
) -> None:
    source_path, target_path = _selection_pair(tmp_path)
    tuple_path = tmp_path / "runs" / "iter_38" / "artifact_tuple.json"
    tuple_path.parent.mkdir(parents=True)
    shutil.copyfile(source_path, tuple_path)
    source_contract = _legacy_contract()
    target_contract = _event_contract(source_contract)

    def fake_build(_project_dir, *, world_selection_path):
        return (
            copy.deepcopy(source_contract)
            if Path(world_selection_path) == source_path
            else copy.deepcopy(target_contract)
        )

    monkeypatch.setattr(
        "sculptor.policy_contract.build_project_policy_contract", fake_build,
    )
    receipt = build_iteration_warm_start_contract_receipt(
        tmp_path, 38, target_selection_path=target_path,
    )
    assert receipt["source"]["selection_version"] == 1
    assert receipt["target"]["selection_version"] == 2
    assert receipt["compatibility"]["type"] == (
        "zero_initialized_event_phase_observation"
    )
    assert receipt["source"]["contract"] == source_contract
    assert receipt["target"]["contract"] == target_contract

    tampered = json.loads(tuple_path.read_text(encoding="utf-8"))
    tampered["evaluation_lineage"] = "tampered"
    tuple_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        build_iteration_warm_start_contract_receipt(
            tmp_path, 38, target_selection_path=target_path,
        )
