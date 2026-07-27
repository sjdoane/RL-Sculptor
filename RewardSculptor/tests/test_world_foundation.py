"""Adversarial contract tests for the prompt-driven world foundation.

These tests intentionally exercise invariants at the public artifact/schema
boundaries.  They use no mjlab runtime and should remain fast enough to run on
every foundation change.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import pytest

import sculptor.world.artifacts as artifact_module
from sculptor.world.artifacts import (
    ArtifactRef,
    WorldArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
)
from sculptor.world.capabilities import (
    CapabilityError,
    resolve_robot_capability,
    robot_capabilities,
)
from sculptor.world.task_spec import (
    parse_contact_selector,
    validate_task_spec,
)
from sculptor.world.world_spec import (
    apply_variation_value,
    resolve_pointer,
    validate_world_spec,
)


def _world() -> dict:
    return {
        "world_spec_version": 2,
        "meta": {
            "version": "v1",
            "parent": None,
            "source": "user",
            "prompt": "move the ball into the goal",
            "grounding": [],
            "parameter_provenance": {},
        },
        "shared": {
            "eval_seed": 1729,
            "robot": {
                "capability_id": "unitree_g1:base",
                "required_capabilities": ["locomotion", "push"],
            },
            "terrain": {"kind": "plane"},
            "obstacles": {
                "layout": "linear",
                "waypoints": "auto",
                "course": [
                    {
                        "id": "platform_01",
                        "element": "platform",
                        "nominal": {
                            "height_m": 0.25,
                            "length_m": 1.0,
                            "width_m": 1.0,
                        },
                    },
                    {
                        "id": "platform_02",
                        "element": "platform",
                        "nominal": {
                            "height_m": 0.40,
                            "length_m": 1.1,
                            "width_m": 1.0,
                        },
                    },
                ],
            },
            "objects": {
                "ball": {
                    "shape": "sphere",
                    "fixed": False,
                    "nominal": {
                        "radius_m": 0.11,
                        "mass_kg": 0.45,
                        "pose": {"position_m": [0.0, 0.0, 0.11]},
                    },
                },
            },
            "zones": {
                "start": {
                    "kind": "disk",
                    "center_m": [0.0, 0.0],
                    "radius_m": 1.0,
                },
                "goal_mouth": {
                    "kind": "box",
                    "center_m": [4.0, 0.0, 0.5],
                    "size_m": [0.3, 1.8, 1.0],
                },
            },
        },
        "train": {
            "variations": [
                {
                    "id": "second_platform_height",
                    "target": (
                        "/shared/obstacles/course/@platform_02/nominal/height_m"
                    ),
                    "class": "model_field",
                    "distribution": {
                        "kind": "uniform",
                        "low": 0.20,
                        "high": 0.60,
                    },
                },
            ],
            "curriculum": {
                "difficulty_range": [0.0, 1.0],
                "promotion": {
                    "signal": "waypoint_success",
                    "promote_above": 0.75,
                    "demote_below": 0.50,
                },
            },
        },
    }


def _generated_terrain_world() -> dict:
    world = _world()
    world["shared"]["terrain"] = {
        "kind": "generator",
        "layout": {
            "mode": "sampled_grid",
            "rows": 2,
            "cols": 2,
            "tile_size_m": [8.0, 8.0],
            "border_width_m": 2.0,
        },
        "evaluation_difficulty": 0.5,
        "sub_terrains": {
            "rough": {
                "type": "hf_random_uniform",
                "proportion": 1.0,
                "nominal": {
                    "noise_range_m": [0.02, 0.08],
                    "noise_step_m": 0.02,
                },
            },
        },
    }
    return world


def _task() -> dict:
    return {
        "task_spec_version": 1,
        "meta": {
            "version": "v1",
            "parent": None,
            "source": "user",
            "prompt": "move the ball into the goal",
            "grounding": [],
        },
        "shared": {
            "control_mode": "goal_directed",
            "goal": {
                "id": "score",
                "type": "object_to_region",
                "subject": "ball",
                "region": "goal_mouth",
                "success": {
                    "predicate": "inside",
                    "hold_s": 0.1,
                    "tolerance_m": 0.0,
                },
            },
            "contacts": {
                "desired": [
                    ["robot:left_foot|right_foot", "object:ball"],
                ],
                "forbidden": [["robot:torso", "object:ball"]],
                "terminate_on": [["robot:torso", "world:terrain"]],
            },
            "termination": {
                "fall": "capability_default",
                "out_of_bounds_m": 12.0,
                "success_ends_episode": False,
                "episode_length_s": 20.0,
            },
            "observations": {
                "proprioception": True,
                "height_scan": False,
                "object_relative": ["ball"],
                "region_relative": ["goal_mouth"],
            },
        },
        "train": {
            "goal_sampling": [
                {
                    "id": "ball_start",
                    "target": "object:ball.pose",
                    "distribution": {
                        "kind": "uniform_in_region",
                        "region": "start",
                    },
                },
            ],
            "scaffolds": [],
        },
    }


def _assert_unknown(errors: list[str], expected_path: str) -> None:
    assert any(expected_path in error and "unknown" in error for error in errors), (
        f"expected strict unknown-key error at {expected_path}; got {errors}"
    )


def test_world_rejects_unknown_keys_at_nested_boundaries() -> None:
    cases = []

    layout = _generated_terrain_world()
    layout["shared"]["terrain"]["layout"]["columnz"] = 2
    cases.append((layout, "shared.terrain.layout"))

    nominal = _generated_terrain_world()
    nominal["shared"]["terrain"]["sub_terrains"]["rough"]["nominal"][
        "noise_typo_m"
    ] = 0.1
    cases.append((nominal, "shared.terrain.sub_terrains.rough.nominal"))

    obj = _world()
    obj["shared"]["objects"]["ball"]["nominal"]["diameter_m"] = 0.22
    cases.append((obj, "shared.objects.ball.nominal"))

    for spec, path in cases:
        _assert_unknown(validate_world_spec(spec), path)


def test_task_rejects_unknown_keys_at_nested_boundaries() -> None:
    world = _world()
    assert validate_world_spec(world) == []

    success = _task()
    success["shared"]["goal"]["success"]["hold_frames"] = 5
    _assert_unknown(validate_task_spec(success, world=world), "shared.goal.success")

    sampling = _task()
    sampling["train"]["goal_sampling"][0]["distribution"]["seed"] = 3
    _assert_unknown(
        validate_task_spec(sampling, world=world),
        "train.goal_sampling[0].distribution",
    )


def test_stock_g1_cannot_be_mislabeled_as_a_gripper() -> None:
    installed = robot_capabilities()
    assert "unitree_g1:dual_gripper" not in installed
    assert "grasp" not in installed["unitree_g1:base"].capabilities

    with pytest.raises(CapabilityError, match="unknown robot capability"):
        resolve_robot_capability("unitree_g1:dual_gripper")

    world = _world()
    world["shared"]["robot"]["required_capabilities"] = ["grasp"]
    errors = validate_world_spec(world)
    assert any("missing required capabilities" in error and "grasp" in error
               for error in errors)


def test_arm_gripper_uses_the_same_generic_capability_contract() -> None:
    cap = resolve_robot_capability(
        "yam:parallel_gripper", required=["manipulation", "grasp"])
    assert cap.resolve_role("gripper") == ("lf_down", "rf_down")
    assert cap.resolve_semantic_role("grasp") == (
        "site", ("grasp_site",))
    assert cap.reward_state_schema["end_effector_pos_w"] == (3,)

    world = _world()
    world["shared"]["robot"] = {
        "capability_id": "yam:parallel_gripper",
        "required_capabilities": ["manipulation", "grasp"],
    }
    task = _task()
    task["shared"]["contacts"] = {
        "desired": [["robot:gripper", "object:ball"]],
        "forbidden": [],
        "terminate_on": [],
    }
    task["shared"]["observations"]["end_effector_relative"] = ["grasp"]
    assert validate_world_spec(world) == []
    assert validate_task_spec(task, world=world) == []


def test_required_capability_failure_is_precise() -> None:
    with pytest.raises(CapabilityError) as exc:
        resolve_robot_capability(
            "unitree_go1:base", required=["locomotion", "grasp"]
        )
    message = str(exc.value)
    assert "missing required capabilities ['grasp']" in message
    assert "installed=" in message


def test_variation_rejects_structural_and_class_mismatch() -> None:
    structural = _world()
    structural["train"]["variations"][0]["class"] = "structural"
    errors = validate_world_spec(structural)
    assert any("structural variation is forbidden" in error for error in errors)

    wrong_class = _generated_terrain_world()
    wrong_class["train"]["variations"] = [
        {
            "id": "roughness",
            "target": (
                "/shared/terrain/sub_terrains/rough/nominal/noise_range_m/1"
            ),
            "class": "model_field",
            "distribution": {"kind": "uniform", "low": 0.04, "high": 0.12},
        }
    ]
    errors = validate_world_spec(wrong_class)
    assert any("does not match registry class 'generator_parameter'" in error
               for error in errors)


def test_stable_id_pointer_targets_same_element_after_reorder() -> None:
    world = _world()
    assert validate_world_spec(world) == []
    world["shared"]["obstacles"]["course"].reverse()

    parent, key = resolve_pointer(
        world,
        "/shared/obstacles/course/@platform_02/nominal/height_m",
    )
    assert key == "height_m"
    assert parent[key] == pytest.approx(0.40)

    updated = apply_variation_value(world, "second_platform_height", 0.55)
    by_id = {
        item["id"]: item for item in updated["shared"]["obstacles"]["course"]
    }
    assert by_id["platform_02"]["nominal"]["height_m"] == pytest.approx(0.55)
    assert by_id["platform_01"]["nominal"]["height_m"] == pytest.approx(0.25)


def test_task_cross_references_and_contact_roles_are_resolved() -> None:
    world = _world()
    task = _task()
    assert validate_world_spec(world) == []
    assert validate_task_spec(task, world=world) == []
    assert parse_contact_selector(
        "robot:left_foot|right_foot", world=world
    ) == (
        "robot",
        ("left_ankle_roll_link", "right_ankle_roll_link"),
    )

    broken = copy.deepcopy(task)
    broken["shared"]["goal"]["subject"] = "missing_ball"
    broken["shared"]["goal"]["region"] = "missing_goal"
    broken["shared"]["contacts"]["desired"] = [
        ["robot:gripper", "object:missing_ball"]
    ]
    errors = validate_task_spec(broken, world=world)
    assert any("unknown object 'missing_ball'" in error for error in errors)
    assert any("unknown zone 'missing_goal'" in error for error in errors)
    assert any("robot role 'gripper' is unavailable" in error for error in errors)


def test_canonical_hash_is_mapping_order_and_format_independent() -> None:
    left = {
        "unicode": "uneven 地形",
        "nested": {"b": 2, "a": [3, 1]},
        "enabled": True,
    }
    right = {
        "enabled": True,
        "nested": {"a": [3, 1], "b": 2},
        "unicode": "uneven 地形",
    }
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_bytes(canonical_json_bytes(left)) == sha256_bytes(
        canonical_json_bytes(right)
    )


def _complete_refs(
    store: WorldArtifactStore, project: Path,
) -> dict[str, ArtifactRef]:
    rewards = project / "rewards"
    rewards.mkdir(parents=True, exist_ok=True)
    reward_path = rewards / "v1.py"
    reward_path.write_text("REWARD_SPEC = {}\n", encoding="utf-8")

    env_spec_path = store.env_dir / "v1.json"
    env_spec_path.write_text("{}\n", encoding="utf-8")

    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v1", reward_path, base=project
        ),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v1", env_spec_path, base=project
        ),
    }
    for kind in (
        "world",
        "task",
        "resolved_eval",
        "channel_catalog",
        "clarifications",
    ):
        refs[kind] = store.write_json(kind, {"kind": kind, "version": 1})
    return refs


def test_complete_selection_is_atomic_and_survives_failed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    first = store.promote(_complete_refs(store, project), evaluation_lineage="eval-a")
    before = store.selection_path.read_bytes()

    second_refs = dict(first.refs)
    second_refs["world"] = store.write_json(
        "world", {"kind": "world", "version": 2}
    )
    real_atomic_write = artifact_module._atomic_write

    def fail_commit(path: Path, data: bytes) -> None:
        if path == store.selection_path:
            raise OSError("simulated interruption at selection commit")
        real_atomic_write(path, data)

    monkeypatch.setattr(artifact_module, "_atomic_write", fail_commit)
    with pytest.raises(OSError, match="simulated interruption"):
        store.promote(second_refs, evaluation_lineage="eval-b")

    assert store.selection_path.read_bytes() == before
    selected = store.read_selection()
    assert selected is not None
    assert selected.tuple_hash == first.tuple_hash
    assert set(selected.refs) == {
        "reward",
        "env_spec",
        "world",
        "task",
        "resolved_eval",
        "channel_catalog",
        "clarifications",
    }


def test_selected_artifact_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    selection = store.promote(
        _complete_refs(store, project), evaluation_lineage="eval-a"
    )
    world_path = store.resolve_ref(selection.refs["world"])
    world_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="selected world hash mismatch"):
        store.read_selection()


def test_immutable_selection_snapshot_can_be_pinned(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    first = store.promote(
        _complete_refs(store, project), evaluation_lineage="eval-a")
    pinned = store.env_dir / f"selection_v{first.selection_version}.json"

    second_refs = dict(first.refs)
    second_refs["world"] = store.write_json(
        "world", {"kind": "world", "version": 2})
    second = store.promote(second_refs, evaluation_lineage="eval-b")

    assert store.read_selection().tuple_hash == second.tuple_hash
    assert store.read_selection(pinned).tuple_hash == first.tuple_hash


def test_incomplete_selection_is_rejected_on_write_and_read(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    refs = _complete_refs(store, project)
    refs.pop("clarifications")
    with pytest.raises(ValueError, match="missing required refs.*clarifications"):
        store.promote(refs, evaluation_lineage="eval-a")
    assert not store.selection_path.exists()

    complete = store.promote(
        _complete_refs(store, project), evaluation_lineage="eval-a"
    )
    payload = complete.to_dict()
    payload["refs"].pop("clarifications")
    payload["tuple_hash"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: asdict(ref)
                for key, ref in sorted(complete.refs.items())
                if key != "clarifications"
            }
        )
    )
    store.selection_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="missing required refs.*clarifications"):
        store.read_selection()
