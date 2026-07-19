"""Focused CPU tests for deterministic world compilation and admission."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np

from mjlab.scene import Scene, SceneCfg

from sculptor.world.compiler import (
    ResolvedEvaluation,
    apply_world_selection,
    compile_world,
    install_materialized_terrain_factory,
    materialized_terrain_types,
)
from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore
from sculptor.world.gates import run_admission_gates
from sculptor.world.capabilities import (
    build_robot_entity_cfg,
    resolve_robot_capability,
)


def _world(*, generated: bool = False, robot: str = "unitree_g1:base") -> dict:
    terrain = {"kind": "plane"}
    if generated:
        terrain = {
            "kind": "generator",
            "layout": {
                "mode": "sampled_grid", "rows": 1, "cols": 1,
                "tile_size_m": [4.0, 4.0], "border_width_m": 1.0,
            },
            "evaluation_difficulty": 0.5,
            "sub_terrains": {
                "rough": {
                    "type": "hf_random_uniform", "proportion": 1.0,
                    "nominal": {
                        "noise_range_m": [0.02, 0.06],
                        "noise_step_m": 0.02,
                    },
                },
            },
        }
    required = ["locomotion"] if robot == "unitree_g1:base" else ["grasp"]
    return {
        "world_spec_version": 2,
        "meta": {
            "version": "v1", "parent": None, "source": "user",
            "prompt": "put the ball in the goal", "grounding": [],
            "parameter_provenance": {},
        },
        "shared": {
            "eval_seed": 1729,
            "robot": {
                "capability_id": robot,
                "required_capabilities": required,
            },
            "terrain": terrain,
            "obstacles": {"layout": "linear", "waypoints": "auto", "course": []},
            "objects": {
                "ball": {
                    "shape": "sphere", "fixed": False,
                    "nominal": {
                        "radius_m": 0.08, "mass_kg": 0.2,
                        "pose": {"position_m": [0.35, 0.0, 0.8]},
                    },
                },
            },
            "zones": {
                "start": {
                    "kind": "disk", "center_m": [0.35, 0.0],
                    "radius_m": 0.25,
                },
                "goal": {
                    "kind": "box", "center_m": [0.6, 0.0, 0.2],
                    "size_m": [0.4, 0.4, 0.4],
                },
            },
        },
        "train": {
            "variations": [],
            "curriculum": {"difficulty_range": [0.0, 1.0]},
        },
    }


def _task(*, robot: str = "unitree_g1:base", height_scan: object = False) -> dict:
    desired = (["robot:left_foot", "object:ball"]
               if robot == "unitree_g1:base"
               else ["robot:gripper", "object:ball"])
    return {
        "task_spec_version": 1,
        "meta": {
            "version": "v1", "parent": None, "source": "user",
            "prompt": "put the ball in the goal", "grounding": [],
        },
        "shared": {
            "control_mode": "goal_directed",
            "goal": {
                "id": "place", "type": "object_to_region",
                "subject": "ball", "region": "goal",
                "success": {
                    "predicate": "inside", "hold_s": 0.1,
                    "tolerance_m": 0.0,
                },
            },
            "contacts": {
                "desired": [desired], "forbidden": [], "terminate_on": [],
            },
            "termination": {
                "fall": "capability_default", "out_of_bounds_m": 8.0,
                "success_ends_episode": False, "episode_length_s": 10.0,
            },
            "observations": {
                "proprioception": True, "height_scan": height_scan,
                "object_relative": ["ball"], "region_relative": ["goal"],
            },
        },
        "train": {"goal_sampling": [], "scaffolds": []},
    }


def test_generated_compilation_is_deterministic_and_robot_agnostic() -> None:
    world = _world(generated=True)
    task = _task(height_scan="auto")
    first = compile_world(world, task)
    second = compile_world(world, task)

    assert first.resolved_eval.compiled_model_hash == \
        second.resolved_eval.compiled_model_hash
    assert first.resolved_eval.manifest_hash == second.resolved_eval.manifest_hash
    assert "authored_height_scan" in first.task_runtime.to_dict()["sensor_names"]

    arm = compile_world(
        _world(robot="yam:parallel_gripper"),
        _task(robot="yam:parallel_gripper"),
    )
    contact = arm.task_runtime.contacts[0]
    assert contact.selectors[0] == "robot:gripper"
    assert contact.resolved[0]["names"] == ["lf_down", "rf_down"]
    sensor = arm.task_runtime.sensor_cfgs[0]
    assert sensor.secondary_policy == "first"
    assert sensor.secondary.entity == "ball"
    assert arm.robot.capability_id == "yam:parallel_gripper"


def test_materialized_terrain_replays_exact_heightfield(tmp_path: Path) -> None:
    root = tmp_path / "eval_assets"
    compiled = compile_world(
        _world(generated=True), _task(), materialize_dir=root)
    manifest = compiled.resolved_eval
    terrain_xml = tmp_path / manifest.materialized_assets["terrain_xml"]
    assert terrain_xml.is_file()

    install_materialized_terrain_factory()
    cfg_type, _ = materialized_terrain_types()
    frozen_cfg = cfg_type(
        terrain_type="generator", terrain_generator=None,
        terrain_xml_path=str(terrain_xml),
        terrain_origins_m=tuple(manifest.terrain["origins_m"]),
        frozen_flat_patches=manifest.terrain["flat_patches"],
        frozen_flat_patch_radii_m=manifest.terrain["flat_patch_radii_m"],
    )
    replay = Scene(SceneCfg(num_envs=1, terrain=frozen_cfg), device="cpu")
    replay_model = replay.compile()
    original = compiled._model
    assert replay_model.nhfield == original.nhfield
    np.testing.assert_array_equal(replay_model.hfield_data, original.hfield_data)


def test_admission_green_path_stamps_manifest() -> None:
    report, compiled = run_admission_gates(
        _world(), _task(), settle_steps=20)
    assert report.ok, [violation.to_dict() for violation in report.violations]
    assert compiled is not None
    assert compiled.resolved_eval.admission["ok"] is True
    rebuilt = ResolvedEvaluation.from_dict(compiled.resolved_eval.to_dict())
    assert rebuilt.manifest_hash == compiled.resolved_eval.manifest_hash


def test_admission_returns_independent_budget_placement_and_reach_violations() -> None:
    world = _world()
    world["shared"]["obstacles"]["course"] = [
        {
            "id": "gap_01", "element": "gap",
            "nominal": {"length_m": 2.0, "width_m": 1.0, "depth_m": 0.2},
        },
    ]
    world["shared"]["objects"]["second_ball"] = copy.deepcopy(
        world["shared"]["objects"]["ball"])
    world["shared"]["zones"]["goal"]["size_m"] = [0.05, 0.05, 0.05]
    task = _task()

    report, _ = run_admission_gates(world, task, settle_steps=5)
    codes = {violation.code for violation in report.violations}
    assert "object_overlap" in codes
    assert "gap_out_of_envelope" in codes
    assert "object_does_not_fit_region" in codes

    budget_world = _world(generated=True)
    layout = budget_world["shared"]["terrain"]["layout"]
    layout.update({"rows": 64, "cols": 64, "tile_size_m": [20.0, 20.0]})
    budget_report, _ = run_admission_gates(
        budget_world, _task(), settle_steps=1)
    assert "heightfield_texels_budget_exceeded" in {
        violation.code for violation in budget_report.violations}


def test_immutable_selection_applies_materialized_eval_without_regeneration(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    world = _world(generated=True)
    task = _task()
    task["shared"]["contacts"]["desired"] = []
    report, compiled = run_admission_gates(
        world, task, materialize_dir=store.env_dir / "eval_assets",
        settle_steps=20)
    assert report.ok and compiled is not None

    rewards = project / "rewards"
    rewards.mkdir(parents=True)
    reward_path = rewards / "v1.py"
    reward_path.write_text("REWARD_SPEC = {}\n", encoding="utf-8")
    env_spec_path = store.env_dir / "v1.json"
    env_spec_path.write_text("{}\n", encoding="utf-8")
    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v1", reward_path, base=project),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v1", env_spec_path, base=project),
        "world": store.write_json("world", world),
        "task": store.write_json("task", task),
        "resolved_eval": store.write_json(
            "resolved_eval", compiled.resolved_eval.to_dict()),
        "channel_catalog": store.write_json(
            "channel_catalog", compiled.channel_catalog.to_dict()),
        "clarifications": store.write_json("clarifications", {"items": []}),
    }
    selection = store.promote(refs, evaluation_lineage="eval-test")
    immutable = store.env_dir / f"selection_v{selection.selection_version}.json"

    base = SceneCfg(num_envs=1, entities={
        "robot": build_robot_entity_cfg(resolve_robot_capability(
            "unitree_g1:base")),
    })
    bundle = apply_world_selection(base, immutable, train=False)
    assert bundle.tuple_hash == selection.tuple_hash
    cfg_type, _ = materialized_terrain_types()
    assert isinstance(base.terrain, cfg_type)
    replay = Scene(base, device="cpu")
    replay_model = replay.compile()
    assert replay_model.nhfield == compiled._model.nhfield
    np.testing.assert_array_equal(
        replay_model.hfield_data, compiled._model.hfield_data)


# ── env-authoring §10.1: train-time terrain difficulty span ───────────────
def _span_world(rng, *, kind="generator", mode="curriculum_grid"):
    return {
        "shared": {"terrain": {"kind": kind, "layout": {"mode": mode}}},
        "train": {"curriculum": {"difficulty_range": rng}},
    }


def test_train_difficulty_span_well_formed_and_degenerate():
    from sculptor.world.compiler import train_difficulty_span

    assert train_difficulty_span(_span_world([0.1, 0.9])) == (0.1, 0.9)
    assert train_difficulty_span(_span_world([0.5, 0.5])) is None
    assert train_difficulty_span(_span_world([0.9, 0.1])) is None
    assert train_difficulty_span(_span_world([-0.1, 0.9])) is None
    assert train_difficulty_span(_span_world([0.1, 1.1])) is None
    assert train_difficulty_span(_span_world("bad")) is None
    assert train_difficulty_span(_span_world([0.1, 0.9], kind="plane")) is None
    assert train_difficulty_span(
        _span_world([0.1, 0.9], mode="sampled_grid")) is None


def test_expand_train_terrain_difficulty_mutates_only_train_scene():
    import types as _types

    from sculptor.world.compiler import expand_train_terrain_difficulty

    generator = _types.SimpleNamespace(difficulty_range=(0.45, 0.45))
    compiled = _types.SimpleNamespace(scene_cfg=_types.SimpleNamespace(
        terrain=_types.SimpleNamespace(terrain_generator=generator)))
    assert expand_train_terrain_difficulty(
        compiled, _span_world([0.0, 1.0])) is True
    assert generator.difficulty_range == (0.0, 1.0)

    # plane / missing generator: untouched, reported False
    flat = _types.SimpleNamespace(scene_cfg=_types.SimpleNamespace(
        terrain=_types.SimpleNamespace(terrain_generator=None)))
    assert expand_train_terrain_difficulty(
        flat, _span_world([0.0, 1.0])) is False


def test_authored_uneven_world_declares_full_span():
    """The offline uneven author's train.curriculum must produce a real
    span (this is what apply_world_selection(train=True) will widen to)."""
    from sculptor.world.author import author_environment
    from sculptor.world.compiler import train_difficulty_span

    draft = author_environment(
        "stay stable and walk on uneven rough terrain",
        robot_capability_id="unitree_g1:base")
    assert train_difficulty_span(draft.world_spec) == (0.0, 1.0)


def test_resolved_evaluation_with_admission_round_trips_course():
    """Live-UI regression: with_admission -> to_dict on a COURSE-BEARING
    manifest crashed ('dict' has no to_dict) because build left round-
    tripped course dicts unconverted; empty-course tests never tripped it."""
    from sculptor.world.compiler import ResolvedEvaluation, ResolvedPrimitive

    prim = ResolvedPrimitive(
        primitive_id="b__platform", source_id="b", shape="box",
        position_m=(1.0, 0.0, 0.1), size_m=(1.0, 1.2, 0.2))
    manifest = ResolvedEvaluation.build(
        world_hash="w", task_hash="t", compiler_hash="c",
        robot_capability_hash="rc", robot_asset_hash="ra",
        simulator_capability_hash="s", dependency_versions={},
        runtime_task_id=None, eval_seed=1, terrain={}, course=[prim],
        objects={}, zones={}, task_shared={}, channel_catalog_hash="cc",
        compiled_model_hash="cm", materialized_assets={}, admission={})
    stamped = manifest.with_admission({"ok": True})
    payload = stamped.to_dict()  # crashed before the fix
    assert payload["course"][0]["primitive_id"] == "b__platform"
    again = stamped.with_admission({"ok": False}).to_dict()
    assert again["course"] == payload["course"]
