"""Focused CPU tests for deterministic world compilation and admission."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco  # noqa: F401 - fail collection if the runtime is unavailable
import numpy as np
import pytest

from mjlab.scene import Scene, SceneCfg

from sculptor.world.compiler import (
    ResolvedEvaluation,
    _clearance_adjusted_waypoint_points,
    _horizon_aware_terminal_brake_radius,
    _horizon_aware_waypoint_cruise,
    _install_task_observations,
    _reconcile_waypoint_course,
    apply_world_selection,
    compile_task_runtime,
    compile_world,
    install_materialized_terrain_factory,
    materialized_terrain_types,
    reset_robot_along_waypoint_route,
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


def test_authored_task_observations_reach_actor_and_critic() -> None:
    """The manifest observation contract must change the policy input."""
    import torch
    from mjlab.managers.observation_manager import ObservationGroupCfg

    world = _world()
    task = _task()
    robot = resolve_robot_capability("unitree_g1:base")
    runtime = compile_task_runtime(world, task, robot)
    cfg = SimpleNamespace(observations={
        "actor": ObservationGroupCfg(terms={}),
        "critic": ObservationGroupCfg(terms={}),
    })
    _install_task_observations(
        cfg, runtime, zones=world["shared"]["zones"], robot=robot)

    expected = {"authored_object__ball", "authored_region__goal"}
    assert expected <= set(cfg.observations["actor"].terms)
    assert expected <= set(cfg.observations["critic"].terms)

    class Scene(dict):
        pass

    scene = Scene({
        "robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[11.0, 20.0, 0.7]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )),
        "ball": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[12.0, 21.0, 0.5]]),
        )),
    })
    scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
    env = SimpleNamespace(scene=scene)

    region_cfg = cfg.observations["actor"].terms["authored_region__goal"]
    object_cfg = cfg.observations["actor"].terms["authored_object__ball"]
    torch.testing.assert_close(
        region_cfg.func(env, **region_cfg.params),
        torch.tensor([[-0.4, 0.0, -0.5]]),
    )
    torch.testing.assert_close(
        object_cfg.func(env, **object_cfg.params),
        torch.tensor([[1.0, 1.0, -0.2]]),
    )


def test_explicit_waypoint_zones_align_commands_without_materialized_course() -> None:
    """A named-zone slalom gets a goal-conditioned, not fixed +X, command."""
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14),
    )
    command = SimpleNamespace(
        ranges=ranges, heading_command=True, rel_standing_envs=0.1,
        rel_heading_envs=1.0, rel_forward_envs=0.0,
    )
    reset = SimpleNamespace(params={
        "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0),
                       "yaw": (-3.14, 3.14)},
    })
    object_reset = SimpleNamespace(params={
        "asset_cfg": SimpleNamespace(name="box_01"),
        "pose_range": {
            "x": (0.0, 0.0), "y": (-0.5, 0.5), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    })
    env_cfg = SimpleNamespace(
        events={"reset_base": reset, "object_reset": object_reset},
        commands={"twist": command},
        curriculum={"command_vel": object()},
    )
    manifest = SimpleNamespace(
        task_shared={"goal": {
            "type": "waypoint_sequence",
            "waypoints": ["waypoint_01", "finish"],
            "success": {"tolerance_m": 0.35, "hold_s": 2.0},
        }},
        course=(),
        zones={
            "waypoint_01": {
                "kind": "disk", "center_m": [2.0, 0.85], "radius_m": 0.35,
            },
            "finish": {
                "kind": "disk", "center_m": [4.0, 0.0], "radius_m": 0.35,
            },
        },
    )

    adjustments = _reconcile_waypoint_course(env_cfg, manifest, train=True)

    routed = env_cfg.commands["twist"]
    assert routed.waypoints_m == ((2.0, 0.85, 0.0), (4.0, 0.0, 0.0))
    assert routed.tolerance_m == 0.35
    assert routed.ranges.lin_vel_x == (-1.0, 1.0)
    assert routed.ranges.lin_vel_y == (-1.0, 1.0)
    assert routed.ranges.ang_vel_z == (-1.5, 1.5)
    assert routed.ranges.heading is None
    assert reset.params["pose_range"]["yaw"] == (-0.08, 0.08)
    assert object_reset.params["pose_range"]["x"] == (0.0, 0.0)
    assert object_reset.params["pose_range"]["y"] == (-0.5, 0.5)
    assert "world_route_state_initialization" in env_cfg.events
    route_rsi = env_cfg.events["world_route_state_initialization"]
    assert route_rsi.mode == "reset"
    assert route_rsi.params["midroute_probability"] == 0.5
    assert route_rsi.params["terminal_fraction_within_midroute"] == 0.5
    assert routed.terminal_stop_at_predicate_boundary is True
    assert routed.terminal_slow_radius_m == pytest.approx(2.0)
    assert (
        routed.cruise_speed_mps * routed.terminal_min_speed_scale
        <= 0.10 + 1e-9
    )
    assert "command_vel" not in env_cfg.curriculum
    assert any("goal-conditioned waypoint traversal" in item
               for item in adjustments)
    assert any("collision-local interior" in item
               for item in adjustments)


def test_forbidden_object_waypoint_uses_embodiment_clearance_subtarget() -> None:
    """A command may use a safer point inside the same authored region."""
    robot = resolve_robot_capability("unitree_g1:base")
    manifest = SimpleNamespace(
        task_shared={
            "goal": {
                "type": "waypoint_sequence",
                "waypoints": ["waypoint", "finish"],
                "success": {"tolerance_m": 0.35},
            },
            "contacts": {
                "forbidden": [
                    ["robot:any", "object:box"],
                ],
            },
        },
        course=(),
        zones={
            "waypoint": {
                "kind": "disk",
                "center_m": [2.0, 0.85],
                "radius_m": 0.45,
            },
            "finish": {
                "kind": "disk",
                "center_m": [4.0, 0.0],
                "radius_m": 0.9,
            },
        },
        objects={
            "box": {
                "shape": "box",
                "nominal": {
                    "pose": {"position_m": [2.0, 0.0, 0.375]},
                    "size_m": [0.45, 0.45, 0.75],
                },
            },
        },
    )

    points, notes = _clearance_adjusted_waypoint_points(
        manifest, ["waypoint", "finish"], robot)

    object_radius = (2.0 * (0.45 / 2.0) ** 2) ** 0.5
    required_clearance = (
        robot.geometry.reach_radius_m + object_radius + 0.05)
    assert points[0][0] == pytest.approx(2.0)
    assert points[0][1] == pytest.approx(required_clearance)
    assert points[0][1] - 0.85 <= 0.8 * 0.35 + 1e-9
    assert points[1] == (4.0, 0.0, 0.0)
    assert len(notes) == 1
    assert "embodiment reach" in notes[0]
    assert "radial entry clearance" in notes[0]

    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5),
        heading=None,
    )
    env_cfg = SimpleNamespace(
        events={},
        commands={"twist": SimpleNamespace(
            ranges=ranges,
            entity_name="robot",
            debug_vis=False,
        )},
        curriculum={},
    )
    adjustments = _reconcile_waypoint_course(
        env_cfg, manifest, train=True, robot=robot)
    routed = env_cfg.commands["twist"]
    assert routed.waypoints_m == (
        (2.0, 0.85, 0.0),
        (4.0, 0.0, 0.0),
    )
    assert routed.predicate_waypoints_m == (
        (2.0, 0.85, 0.0),
        (4.0, 0.0, 0.0),
    )
    assert routed.clearance_shifts_m[0] == pytest.approx(
        (0.0, required_clearance - 0.85, 0.0))
    assert routed.clearance_shifts_m[1] == (0.0, 0.0, 0.0)
    stage = routed.clearance_staging_shifts_m[0]
    assert math.hypot(stage[0], stage[1]) == pytest.approx(0.45)
    assert stage[0] < 0.0
    assert stage[1] > 0.0
    # Following the stage ray into the immutable disk reaches its boundary
    # with the same obstacle-away clearance as the original safe subtarget.
    assert stage[1] * (0.35 / 0.45) == pytest.approx(
        required_clearance - 0.85)
    assert routed.clearance_staging_shifts_m[1] == (0.0, 0.0, 0.0)
    traversal = routed.clearance_traversal_shifts_m[0]
    assert traversal == pytest.approx(
        routed.clearance_shifts_m[0])
    assert math.hypot(traversal[0], traversal[1]) == pytest.approx(
        required_clearance - 0.85)
    assert routed.clearance_traversal_shifts_m[1] == (0.0, 0.0, 0.0)
    assert routed.tolerance_m == pytest.approx(0.35)
    assert routed.clearance_transition_slack_m == pytest.approx(0.025)
    assert routed.clearance_stage_capture_radius_m == pytest.approx(0.15)
    assert routed.intermediate_min_speed_scale == pytest.approx(0.35)
    assert routed.clearance_traversal_min_speed_scale == pytest.approx(1.0)
    assert routed.terminal_stop_at_predicate_boundary is False
    assert routed.terminal_min_speed_scale == pytest.approx(0.35)
    assert any("outside approach stage" in item
               for item in adjustments)
    assert any("frozen 0.350 m task-disk entry" in item
               for item in adjustments)
    rsi_points = env_cfg.events[
        "world_route_state_initialization"].params["waypoints_m"]
    assert rsi_points[0] == pytest.approx((
        2.0 + stage[0],
        0.85 + stage[1],
        0.0,
    ))
    assert rsi_points[1] == (4.0, 0.0, 0.0)


def test_clearance_stage_requires_plane_crossing_and_finite_width() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.world.compiler import _clearance_stage_reached

    centers = torch.tensor([
        [2.0, 0.85],
        [4.0, 0.0],
    ])
    shifts = torch.tensor([
        [-0.29, 0.344],
        [0.0, 0.0],
    ])
    positions = torch.tensor([
        [1.72, 1.19],  # Crossed the finite-width outside stage.
        [4.0, 0.0],  # Unadjusted waypoints have no separate stage.
    ])
    reached = _clearance_stage_reached(
        positions,
        centers,
        shifts,
        capture_radius_m=0.15,
        clearance_slack_m=0.025,
    )
    assert reached.tolist() == [True, False]

    # Entering the immutable disk is not itself approach-stage completion.
    positions[0] = torch.tensor([2.0, 1.05])
    assert not bool(_clearance_stage_reached(
        positions,
        centers,
        shifts,
        capture_radius_m=0.15,
        clearance_slack_m=0.025,
    )[0])

    # Crossing the plane far beside the planned approach cannot trigger it.
    stage_direction = shifts[0] / torch.linalg.norm(shifts[0])
    tangent = torch.tensor([-stage_direction[1], stage_direction[0]])
    positions[0] = centers[0] + shifts[0] + 0.2 * tangent
    assert not bool(_clearance_stage_reached(
        positions,
        centers,
        shifts,
        capture_radius_m=0.15,
        clearance_slack_m=0.025,
    )[0])


def test_waypoint_cruise_reserves_horizon_for_authored_terminal_hold() -> None:
    """Long routes accelerate generically without redefining the objective."""
    waypoints = (
        (2.0, 0.85, 0.0),
        (3.5, -0.85, 0.0),
        (5.0, 0.85, 0.0),
        (6.5, -0.85, 0.0),
        (8.0, 0.0, 0.0),
    )
    no_stages = tuple((0.0, 0.0, 0.0) for _ in waypoints)

    speed, path_length, traversal_window = (
        _horizon_aware_waypoint_cruise(
            waypoints,
            no_stages,
            episode_length_s=20.0,
            hold_s=2.0,
            max_speed_mps=1.0,
        )
    )

    assert path_length == pytest.approx(10.698696, abs=1e-6)
    assert traversal_window == pytest.approx(16.0)
    assert speed == pytest.approx(path_length / (16.0 * 0.70))
    assert 0.8 < speed < 1.0

    capped_speed, _, _ = _horizon_aware_waypoint_cruise(
        waypoints,
        no_stages,
        episode_length_s=20.0,
        hold_s=2.0,
        max_speed_mps=0.9,
    )
    assert capped_speed == pytest.approx(0.9)

    short_speed, _, _ = _horizon_aware_waypoint_cruise(
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        episode_length_s=20.0,
        hold_s=2.0,
        max_speed_mps=1.0,
    )
    assert short_speed == pytest.approx(0.8)


def test_terminal_brake_radius_preserves_staged_route_transition_slack() -> None:
    """A horizon-saturated staged route cannot spend all slack braking."""
    assert _horizon_aware_terminal_brake_radius(
        path_length_m=12.74,
        traversal_window_s=16.0,
        cruise_speed_mps=1.0,
        command_segment_count=9,
    ) == pytest.approx(0.5)

    assert _horizon_aware_terminal_brake_radius(
        path_length_m=4.0,
        traversal_window_s=16.0,
        cruise_speed_mps=0.8,
        command_segment_count=2,
    ) == pytest.approx(2.0)


def test_adjusted_waypoint_stages_then_synchronizes_on_frozen_disk() -> None:
    """The command never invents a second route-success predicate."""
    import torch

    robot_capability = resolve_robot_capability("unitree_g1:base")
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5),
        heading=None,
    )
    env_cfg = SimpleNamespace(
        events={},
        commands={"twist": SimpleNamespace(
            ranges=ranges,
            entity_name="robot",
            debug_vis=False,
        )},
        curriculum={},
    )
    manifest = SimpleNamespace(
        task_shared={
            "goal": {
                "type": "waypoint_sequence",
                "waypoints": ["waypoint", "finish"],
                "success": {"tolerance_m": 0.35},
            },
            "contacts": {
                "forbidden": [["robot:any", "object:box"]],
            },
        },
        course=(),
        zones={
            "waypoint": {
                "kind": "disk",
                "center_m": [2.0, 0.85],
                "radius_m": 0.45,
            },
            "finish": {
                "kind": "disk",
                "center_m": [4.0, 0.0],
                "radius_m": 0.9,
            },
        },
        objects={
            "box": {
                "shape": "box",
                "nominal": {
                    "pose": {"position_m": [2.0, 0.0, 0.375]},
                    "size_m": [0.45, 0.45, 0.75],
                },
            },
        },
    )
    _reconcile_waypoint_course(
        env_cfg, manifest, train=False, robot=robot_capability)
    cfg = env_cfg.commands["twist"]

    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros((1, 3)),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
        heading_w=torch.zeros(1),
    ))

    class FakeScene(dict):
        pass

    scene = FakeScene(robot=robot)
    scene.env_origins = torch.zeros((1, 3))
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene)
    term = cfg.build(env)
    term._update_command()
    assert term._waypoint_index.item() == 0
    assert not term._clearance_stage_complete.item()

    stage = (
        torch.tensor(cfg.predicate_waypoints_m[0][:2])
        + torch.tensor(cfg.clearance_staging_shifts_m[0][:2])
    )
    robot.data.root_link_pos_w[0, :2] = stage
    term._update_command()
    assert term._waypoint_index.item() == 0
    assert term._clearance_stage_complete.item()
    # Once staged, the command aims at the embodiment-safe cap inside the
    # immutable disk. The disk itself remains the only advancement authority.
    safe_target = (
        torch.tensor(cfg.predicate_waypoints_m[0][:2])
        + torch.tensor(cfg.clearance_traversal_shifts_m[0][:2])
    )
    expected_direction = safe_target - stage
    expected_direction = (
        expected_direction / torch.linalg.norm(expected_direction)
    )
    actual_direction = term.command[0, :2]
    actual_direction = actual_direction / torch.linalg.norm(actual_direction)
    torch.testing.assert_close(actual_direction, expected_direction)
    assert torch.linalg.norm(term.command[0, :2]) == pytest.approx(
        cfg.cruise_speed_mps)

    # A policy that drifted around the old outgoing target is pulled back to
    # the in-disk cap with full command authority instead of parking outside.
    safe_shift = torch.tensor(
        cfg.clearance_traversal_shifts_m[0][:2])
    robot.data.root_link_pos_w[0, :2] = (
        torch.tensor(cfg.predicate_waypoints_m[0][:2])
        + torch.tensor([0.36, safe_shift[1].item() + 0.13])
    )
    term._update_command()
    assert term._waypoint_index.item() == 0
    assert torch.linalg.norm(term.command[0, :2]) == pytest.approx(
        cfg.cruise_speed_mps)
    drift_direction = safe_target - robot.data.root_link_pos_w[0, :2]
    drift_direction = drift_direction / torch.linalg.norm(drift_direction)
    actual_direction = term.command[0, :2]
    actual_direction = actual_direction / torch.linalg.norm(actual_direction)
    torch.testing.assert_close(actual_direction, drift_direction)

    robot.data.root_link_pos_w[0, :2] = torch.tensor([2.0, 0.85])
    term._update_command()
    assert term._waypoint_index.item() == 1
    assert not term._clearance_stage_complete.item()
    assert term._clearance_followthrough_pending.item()
    expected_direction = (
        safe_target - robot.data.root_link_pos_w[0, :2])
    expected_direction = (
        expected_direction / torch.linalg.norm(expected_direction))
    actual_direction = term.command[0, :2]
    actual_direction = actual_direction / torch.linalg.norm(actual_direction)
    torch.testing.assert_close(actual_direction, expected_direction)

    # Objective truth has already advanced, but the next route target is not
    # released until the embodiment-safe radial component is recovered.
    robot.data.root_link_pos_w[0, :2] = safe_target
    term._update_command()
    assert term._waypoint_index.item() == 1
    assert not term._clearance_followthrough_pending.item()

    # Terminal raw entry also advances immediately and begins the authored
    # dwell phase on that same frame. The command independently retains a
    # small in-disk target until it has positional margin against settling
    # drift, then becomes exactly zero.
    finish_center = torch.tensor(cfg.predicate_waypoints_m[1][:2])
    robot.data.root_link_pos_w[0, :2] = (
        finish_center + torch.tensor([-0.34, 0.0]))
    term._update_command()
    assert term._waypoint_index.item() == 2
    assert not term._terminal_settle_complete.item()
    assert term.is_standing_env.item()
    assert torch.linalg.norm(term.command[0, :2]) == pytest.approx(
        cfg.cruise_speed_mps * cfg.terminal_min_speed_scale)

    robot.data.root_link_pos_w[0, :2] = (
        finish_center + torch.tensor([-0.20, 0.0]))
    term._update_command()
    assert term._terminal_settle_complete.item()
    assert term.is_standing_env.item()
    assert torch.linalg.norm(term.command[0, :2]) == pytest.approx(0.0)
    assert cfg.terminal_retention_radius_m == pytest.approx(0.25)

    # A stochastic early disk entry must still advance on the exact frame the
    # frozen objective advances, preventing persistent command/metric drift.
    robot.data.root_link_pos_w[0, :2] = safe_target
    fresh = cfg.build(env)
    fresh._clearance_stage_complete[:] = False
    fresh._waypoint_index[:] = 0
    fresh._update_command()
    assert fresh._waypoint_index.item() == 1


def test_route_rsi_places_robot_before_sampled_local_waypoint() -> None:
    import torch

    num_envs = 8
    origins = torch.stack((
        torch.arange(num_envs, dtype=torch.float32) * 10.0,
        torch.zeros(num_envs),
        torch.zeros(num_envs),
    ), dim=-1)
    default = torch.zeros((num_envs, 13), dtype=torch.float32)
    default[:, 2] = 0.8
    default[:, 3] = 1.0

    class FakeRobot:
        is_fixed_base = False
        data = SimpleNamespace(default_root_state=default)

        def write_root_state_to_sim(self, root_state, env_ids=None):
            self.written = root_state.clone()
            self.written_ids = env_ids.clone()

    class FakeScene(dict):
        pass

    robot = FakeRobot()
    scene = FakeScene(robot=robot)
    scene.env_origins = origins
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu", scene=scene)
    waypoints = (
        (2.0, 0.85, 0.0),
        (3.5, -0.85, 0.0),
        (5.0, 0.85, 0.0),
    )
    torch.manual_seed(4)
    reset_robot_along_waypoint_route(
        env,
        None,
        waypoints_m=waypoints,
        midroute_probability=1.0,
        approach_distance_m=(0.25, 0.55),
        lateral_jitter_m=0.12,
    )

    starts = env._sculptor_waypoint_start_index
    assert torch.all(starts >= 1)
    assert torch.all(starts < len(waypoints))
    local_xy = robot.written[:, :2] - origins[:, :2]
    targets = torch.as_tensor(waypoints)[starts, :2]
    distances = torch.linalg.norm(targets - local_xy, dim=-1)
    assert torch.all(distances >= 0.24)
    assert torch.all(distances <= 0.58)
    assert torch.all(robot.written[:, 7:13] == 0)

    torch.manual_seed(5)
    reset_robot_along_waypoint_route(
        env,
        None,
        waypoints_m=waypoints,
        midroute_probability=1.0,
        terminal_fraction_within_midroute=1.0,
        approach_distance_m=(0.25, 0.55),
        lateral_jitter_m=0.12,
    )
    assert torch.all(
        env._sculptor_waypoint_start_index == len(waypoints) - 1)


def test_waypoint_velocity_command_turns_and_stops_per_environment() -> None:
    """The command follows the active target and zeroes only completed envs."""
    import torch

    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5), heading=None,
    )
    command = SimpleNamespace(
        ranges=ranges, entity_name="robot", debug_vis=False,
    )
    env_cfg = SimpleNamespace(
        events={}, commands={"twist": command}, curriculum={},
    )
    manifest = SimpleNamespace(
        task_shared={"goal": {
            "type": "waypoint_sequence",
            "waypoints": ["left", "right", "finish"],
            "success": {"tolerance_m": 0.2, "hold_s": 2.0},
        }},
        course=(),
        zones={
            "left": {"kind": "disk", "center_m": [1.0, 1.0]},
            "right": {"kind": "disk", "center_m": [2.0, -1.0]},
            "finish": {"kind": "disk", "center_m": [3.0, 0.0]},
        },
    )
    _reconcile_waypoint_course(env_cfg, manifest, train=True)

    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros((2, 3)),
        root_link_lin_vel_b=torch.zeros((2, 3)),
        root_link_ang_vel_b=torch.zeros((2, 3)),
        heading_w=torch.zeros(2),
    ))

    class FakeScene(dict):
        pass

    scene = FakeScene(robot=robot)
    scene.env_origins = torch.zeros((2, 3))
    env = SimpleNamespace(num_envs=2, device="cpu", scene=scene)
    term = env_cfg.commands["twist"].build(env)
    term._update_command()
    assert term.command[0, 0] > 0
    assert term.command[0, 1] > 0
    assert term.command[0, 2] > 0

    robot.data.root_link_pos_w[0, :2] = torch.tensor([1.0, 1.0])
    term._update_command()
    assert term._waypoint_index.tolist() == [1, 0]
    assert term.command[0, 1] < 0
    assert term.command[0, 2] < 0
    assert term.command[1, 1] > 0

    robot.data.root_link_pos_w[0, :2] = torch.tensor([2.0, -1.0])
    term._update_command()
    robot.data.root_link_pos_w[0, :2] = torch.tensor([2.79, 0.0])
    term._update_command()
    terminal_entry_speed = torch.linalg.norm(term.command[0, :2])
    assert 0 < terminal_entry_speed <= 0.10 + 1e-6
    assert term.cfg.terminal_stop_at_predicate_boundary is True
    assert abs(term.command[0, 2]) < term.cfg.max_yaw_rate
    robot.data.root_link_pos_w[0, :2] = torch.tensor([3.0, 0.0])
    term._update_command()
    assert term._waypoint_index.tolist() == [3, 0]
    torch.testing.assert_close(term.command[0], torch.zeros(3))
    assert term.is_standing_env.tolist() == [True, False]
    assert torch.linalg.norm(term.command[1]) > 0


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


def test_authored_plane_removes_only_generator_dependent_curriculum(
    tmp_path: Path,
) -> None:
    """A real rough-task cfg must reset after an authored plane overlay.

    Regression for the first UI-launched parkour run: mjlab's retained
    terrain-level curriculum asserted because the authored plane correctly has
    no terrain generator.  Keep the unrelated command curriculum intact.
    """
    from mjlab.tasks.registry import load_env_cfg

    project = tmp_path / "project"
    store = WorldArtifactStore(project)
    world = _world(robot="unitree_go1:base")
    world["shared"]["robot"]["required_capabilities"] = ["locomotion"]
    task = _task(robot="unitree_go1:base")
    task["shared"]["contacts"]["desired"] = []
    report, compiled = run_admission_gates(
        world, task, materialize_dir=store.env_dir / "eval_assets",
        settle_steps=20,
        runtime_task_id="Mjlab-Velocity-Rough-Unitree-Go1",
    )
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

    env_cfg = load_env_cfg("Mjlab-Velocity-Rough-Unitree-Go1")
    assert "terrain_levels" in env_cfg.curriculum
    assert "command_vel" in env_cfg.curriculum
    bundle = apply_world_selection(
        env_cfg,
        immutable,
        train=True,
        runtime_task_id="Mjlab-Velocity-Rough-Unitree-Go1",
    )

    assert env_cfg.scene.terrain.terrain_generator is None
    assert "terrain_levels" not in env_cfg.curriculum
    assert "command_vel" in env_cfg.curriculum
    # The constraint budget is raised first: even this rough task's own
    # njmax=1500 / nconmax=35 are sized for ITS scene, not for the authored
    # course now standing in front of the robot.
    assert bundle.runtime_adjustments == (
        (
            "constraint budget for authored scene: njmax 1500→1536, "
            "nconmax 35→512 (task defaults are sized for the task's own "
            "scene; overflow drops contact rows silently and ends in NaN "
            "observations)"
        ),
        "curriculum:terrain_levels→removed(no live terrain generator)",
        (
            "physical scene alignment → object 'ball' → "
            "nominal local pose + env origin"
        ),
    )


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


# ── constraint budget for authored scenes ─────────────────────────────
def _sim_cfg(njmax, nconmax):
    """Minimal stand-in for mjlab's SimulationCfg — only the two knobs."""
    return SimpleNamespace(sim=SimpleNamespace(njmax=njmax, nconmax=nconmax))


def test_authored_scene_raises_flat_plane_constraint_budget():
    """The bug this closes: mjlab's G1 flat config pins njmax=300 for a bare
    plane. An authored box course overflows it on ~100% of steps, mjwarp drops
    contact rows silently, and training dies on NaN observations ~18 learning
    iterations later."""
    from sculptor.world.compiler import (
        AUTHORED_WORLD_NCONMAX, AUTHORED_WORLD_NJMAX,
        _reconcile_constraint_budget,
    )

    cfg = _sim_cfg(300, None)
    notes = _reconcile_constraint_budget(cfg)

    assert cfg.sim.njmax == AUTHORED_WORLD_NJMAX
    assert cfg.sim.nconmax == AUTHORED_WORLD_NCONMAX
    assert notes and "njmax 300→" in notes[0]
    # nconmax=None is mjwarp's heuristic, which measured WORSE than the pinned
    # default here — it must read as unset, not as "already big enough".
    assert "nconmax heuristic→" in notes[0]


def test_constraint_budget_never_shrinks_a_larger_task_default():
    """A task asking for more than the floor knows something about its own
    scene that we do not; lowering it would recreate the overflow the other
    way round."""
    from sculptor.world.compiler import _reconcile_constraint_budget

    cfg = _sim_cfg(9000, 4000)
    assert _reconcile_constraint_budget(cfg) == ()
    assert (cfg.sim.njmax, cfg.sim.nconmax) == (9000, 4000)


def test_constraint_budget_is_a_no_op_without_the_knobs():
    """gym_sb3 and friends have no sim cfg — must not raise."""
    from sculptor.world.compiler import _reconcile_constraint_budget

    assert _reconcile_constraint_budget(SimpleNamespace()) == ()
    assert _reconcile_constraint_budget(SimpleNamespace(sim=SimpleNamespace())) == ()


# ── env grid pitch vs. authored course footprint ──────────────────────
def _course_world(
    *lengths_m: float, generated: bool = False, lead_gap_m: float = 0.0,
) -> dict:
    """A world whose linear course is a run of platforms of the given lengths
    laid end to end, optionally starting `lead_gap_m` in front of the origin
    (a real authored course leaves the robot room to accelerate)."""
    world = _world(generated=generated)
    course = []
    if lead_gap_m:
        course.append({
            "id": "approach", "element": "gap",
            "nominal": {"length_m": lead_gap_m, "width_m": 1.2, "depth_m": 0.0},
            "variations": [],
        })
    for index, length in enumerate(lengths_m, start=1):
        course.append({
            "id": f"box_{index:02d}", "element": "platform",
            "nominal": {"length_m": length, "width_m": 1.2, "height_m": 0.3},
            "variations": [],
        })
    world["shared"]["obstacles"]["course"] = course
    world["shared"]["objects"] = {}
    return world


def _scene_cfg(env_spacing: float, *, terrain=None):
    """Minimal stand-in for mjlab's SceneCfg — only the pitch and terrain."""
    return SimpleNamespace(
        scene=SimpleNamespace(env_spacing=env_spacing, terrain=terrain))


def test_env_spacing_widens_to_clear_the_authored_course():
    """The bug this closes: mjlab shares one model across parallel envs and
    separates them by a 2.0 m grid pitch sized for a bare robot. An authored
    course reaching several metres forward of the origin gets repeated at every
    origin, so each robot spawns INSIDE a neighbour's boxes. Physics keeps
    stepping and per-env observations look ordinary — the only symptom is a
    rollout video full of interpenetrating boxes."""
    from sculptor.world.compiler import (
        AUTHORED_COURSE_CLEARANCE_M, _reconcile_env_spacing, authored_footprint_m,
    )

    world = _course_world(1.0, 1.1, 1.2)
    span_x, span_y = authored_footprint_m(world)
    cfg = _scene_cfg(2.0)

    notes = _reconcile_env_spacing(cfg, world)

    assert cfg.scene.env_spacing == pytest.approx(
        span_x + AUTHORED_COURSE_CLEARANCE_M)
    assert cfg.scene.env_spacing > span_x > 2.0
    assert notes and "env_spacing 2→" in notes[0]
    assert "spawn" in notes[0].lower() or "neighbouring" in notes[0]
    assert span_y < span_x  # the run is long in x, narrow in y


def test_env_spacing_measures_from_the_origin_not_the_first_box():
    """The robot spawns at the origin, so the span that must fit the pitch runs
    origin→far edge. Measuring first-box→last-box would under-report by the
    approach gap and leave the spawn inside a neighbour's geometry."""
    from sculptor.world.compiler import authored_footprint_m, resolve_course

    world = _course_world(1.0, 1.0, lead_gap_m=0.8)
    course = resolve_course(world)
    first_to_last = (
        max(p.position_m[0] + p.size_m[0] / 2 for p in course)
        - min(p.position_m[0] - p.size_m[0] / 2 for p in course))
    span_x, _ = authored_footprint_m(world)

    # The 0.8 m approach gap is part of what has to clear the neighbour.
    assert span_x == pytest.approx(first_to_last + 0.8)
    assert span_x > first_to_last
    assert span_x == pytest.approx(
        max(p.position_m[0] + p.size_m[0] / 2 for p in course))


def test_env_spacing_never_narrows_a_wider_task_default():
    """A task that already spreads its envs further apart knows something about
    its own scene; narrowing the pitch would create the overlap we are here to
    remove."""
    from sculptor.world.compiler import _reconcile_env_spacing

    cfg = _scene_cfg(50.0)
    assert _reconcile_env_spacing(cfg, _course_world(1.0, 1.0)) == ()
    assert cfg.scene.env_spacing == 50.0


def test_env_spacing_is_a_no_op_without_an_authored_course():
    """A bare-robot world has no geometry to clear — the task's own pitch is
    the right one and must be left alone."""
    from sculptor.world.compiler import _reconcile_env_spacing

    world = _world()
    world["shared"]["objects"] = {}
    cfg = _scene_cfg(2.0)
    assert _reconcile_env_spacing(cfg, world) == ()
    assert cfg.scene.env_spacing == 2.0


def test_env_spacing_is_a_no_op_without_the_knob():
    """Non-mjlab adapters have no scene cfg — must not raise."""
    from sculptor.world.compiler import _reconcile_env_spacing

    world = _course_world(1.0)
    assert _reconcile_env_spacing(SimpleNamespace(), world) == ()
    assert _reconcile_env_spacing(
        SimpleNamespace(scene=SimpleNamespace()), world) == ()


def test_generator_terrain_rejects_a_course_larger_than_its_tile():
    """Generator terrains take env origins from terrain tiles and ignore
    env_spacing, so an oversized course cannot be fixed by widening the pitch.
    Reaching the GPU anyway would reproduce the same silent interpenetration —
    refuse instead of pretending the scene is valid."""
    from sculptor.world.compiler import WorldCompileError, _reconcile_env_spacing

    terrain = SimpleNamespace(
        terrain_type="generator",
        terrain_generator=SimpleNamespace(size=(4.0, 4.0)))
    cfg = _scene_cfg(2.0, terrain=terrain)

    with pytest.raises(WorldCompileError, match="does not fit inside one terrain tile"):
        _reconcile_env_spacing(cfg, _course_world(2.0, 2.0, 2.0))


def test_generator_terrain_accepts_a_course_inside_its_tile():
    """The pitch is the tile's, not ours — a course that fits needs no note."""
    from sculptor.world.compiler import _reconcile_env_spacing

    terrain = SimpleNamespace(
        terrain_type="generator",
        terrain_generator=SimpleNamespace(size=(40.0, 40.0)))
    cfg = _scene_cfg(2.0, terrain=terrain)

    assert _reconcile_env_spacing(cfg, _course_world(1.0, 1.0)) == ()
    assert cfg.scene.env_spacing == 2.0


@pytest.mark.parametrize("spacing", [2.0, 2.5, 3.0, 4.0])
def test_reconciled_pitch_leaves_every_spawn_outside_every_neighbour(spacing):
    """The property that actually matters: after reconciliation, no repeat of
    the course at ±k·pitch may contain the origin where the robot spawns."""
    from sculptor.world.compiler import (
        _reconcile_env_spacing, resolve_course,
    )

    world = _course_world(1.0, 1.1, 1.2, 1.3)
    cfg = _scene_cfg(spacing)
    _reconcile_env_spacing(cfg, world)
    pitch = cfg.scene.env_spacing

    boxes = [(p.position_m[0] - p.size_m[0] / 2, p.position_m[0] + p.size_m[0] / 2)
             for p in resolve_course(world)]
    for k in list(range(-6, 0)) + list(range(1, 7)):
        for x0, x1 in boxes:
            assert not (x0 + k * pitch <= 0.0 <= x1 + k * pitch), (
                f"neighbour at {k * pitch:+.2f} m still swallows the spawn")


def test_constraint_budget_honors_env_overrides(monkeypatch):
    from sculptor.world.compiler import _reconcile_constraint_budget

    monkeypatch.setenv("RS_WORLD_NJMAX", "4096")
    monkeypatch.setenv("RS_WORLD_NCONMAX", "2048")
    cfg = _sim_cfg(300, 64)
    _reconcile_constraint_budget(cfg)
    assert (cfg.sim.njmax, cfg.sim.nconmax) == (4096, 2048)


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "0", "-5"])
def test_constraint_budget_ignores_garbage_overrides(monkeypatch, bad):
    """A typo'd env var must fall back to the measured floor, not disable the
    fix or allocate zero rows."""
    from sculptor.world.compiler import (
        AUTHORED_WORLD_NJMAX, _reconcile_constraint_budget,
    )

    monkeypatch.setenv("RS_WORLD_NJMAX", bad)
    cfg = _sim_cfg(300, 64)
    _reconcile_constraint_budget(cfg)
    assert cfg.sim.njmax == AUTHORED_WORLD_NJMAX


def test_constraint_budget_floor_exceeds_measured_peak():
    """Guards the constant against a future edit that trims it below what the
    box course actually needs. Measured peak was 625 rows/world at
    num_envs=1024 under random actions."""
    from sculptor.world.compiler import (
        AUTHORED_WORLD_NCONMAX, AUTHORED_WORLD_NJMAX,
    )

    MEASURED_PEAK_NEFC = 625
    assert AUTHORED_WORLD_NJMAX >= 2 * MEASURED_PEAK_NEFC
    assert AUTHORED_WORLD_NCONMAX >= 256


# ── route RSI must land the robot ON the course, not inside it ─────────
def _reset_env(num_envs: int = 16):
    """Fake env + robot for the route-RSI event, standing height 0.8 m."""
    import torch

    default = torch.zeros((num_envs, 13), dtype=torch.float32)
    default[:, 2] = 0.8
    default[:, 3] = 1.0

    class FakeRobot:
        is_fixed_base = False
        data = SimpleNamespace(default_root_state=default)
        written = None

        def write_root_state_to_sim(self, root_state, env_ids=None):
            self.written = root_state.clone()

    class FakeScene(dict):
        pass

    robot = FakeRobot()
    scene = FakeScene(robot=robot)
    scene.env_origins = torch.zeros((num_envs, 3), dtype=torch.float32)
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene), robot


def test_surface_height_reads_the_top_of_the_box_underfoot():
    """(x_min, x_max, y_min, y_max, top_z) rows; off the boxes the surface is
    the plane, and overlapping boxes resolve to the highest."""
    import torch

    from sculptor.world.compiler import _surface_height_at

    boxes = ((0.0, 1.0, -0.6, 0.6, 0.25), (0.5, 2.0, -0.6, 0.6, 0.40))
    xy = torch.tensor([[0.2, 0.0], [0.7, 0.0], [1.5, 0.0], [5.0, 0.0],
                       [0.2, 2.0]])
    got = _surface_height_at(xy, boxes)

    assert got.tolist() == pytest.approx([0.25, 0.40, 0.40, 0.0, 0.0])
    assert _surface_height_at(xy, ()).tolist() == [0.0] * 5


def test_route_rsi_stands_the_robot_on_the_platform_not_inside_it():
    """The bug this closes: the event rewrote x and y and left z at the
    flat-ground standing height, so every reset onto a platform drove the
    robot's shins through the box by the box's own height. MuJoCo resolves the
    interpenetration by pushing rather than erroring and the height observation
    still reads plausibly, so nothing downstream notices."""
    import torch

    from sculptor.world.compiler import reset_robot_along_waypoint_route

    # Three platforms of increasing height, waypoints at their centres.
    boxes = ((1.5, 2.5, -0.6, 0.6, 0.20),
             (3.0, 4.0, -0.6, 0.6, 0.35),
             (4.5, 5.5, -0.6, 0.6, 0.50))
    waypoints = ((2.0, 0.0, 0.10), (3.5, 0.0, 0.175), (5.0, 0.0, 0.25))

    env, robot = _reset_env()
    torch.manual_seed(11)
    reset_robot_along_waypoint_route(
        env, None, waypoints_m=waypoints, midroute_probability=1.0,
        approach_distance_m=(0.05, 0.15), lateral_jitter_m=0.05,
        support_boxes_m=boxes)

    from sculptor.world.compiler import _surface_height_at
    surface = _surface_height_at(robot.written[:, :2], boxes)
    clearance = robot.written[:, 2] - surface
    # Every robot stands its full default height above whatever is underfoot.
    assert torch.allclose(clearance, torch.full_like(clearance, 0.8), atol=1e-5)
    assert torch.any(surface > 0.0), "fixture must place some resets on a box"


def test_route_rsi_leaves_plane_resets_at_the_default_height():
    """A route point over open ground must not be lifted — the fix has to be
    the surface under the robot, not a blanket offset."""
    import torch

    from sculptor.world.compiler import reset_robot_along_waypoint_route

    waypoints = ((2.0, 0.0, 0.0), (3.5, 0.0, 0.0), (5.0, 0.0, 0.0))
    env, robot = _reset_env()
    torch.manual_seed(11)
    reset_robot_along_waypoint_route(
        env, None, waypoints_m=waypoints, midroute_probability=1.0,
        approach_distance_m=(0.05, 0.15), lateral_jitter_m=0.05,
        support_boxes_m=((10.0, 11.0, -0.6, 0.6, 0.5),))  # far from the route

    assert torch.allclose(
        robot.written[:, 2], torch.full((env.num_envs,), 0.8), atol=1e-6)


def test_route_rsi_without_support_boxes_still_runs():
    """Older callers pass no course; the event must keep working (at the
    default height) rather than raise."""
    import torch

    from sculptor.world.compiler import reset_robot_along_waypoint_route

    env, robot = _reset_env()
    torch.manual_seed(3)
    reset_robot_along_waypoint_route(
        env, None, waypoints_m=((2.0, 0.0, 0.0), (3.5, 0.0, 0.0)),
        midroute_probability=1.0, approach_distance_m=(0.25, 0.55),
        lateral_jitter_m=0.12)

    assert robot.written is not None
    assert torch.allclose(
        robot.written[:, 2], torch.full((env.num_envs,), 0.8), atol=1e-6)


def test_installed_route_rsi_carries_the_course_geometry():
    """The wiring, not just the function: the event as installed must receive
    the platform tops, or the fix above never runs in a real training job."""
    from sculptor.world.compiler import _reconcile_waypoint_course, resolve_course

    world = _course_world(1.0, 1.1, 1.2, lead_gap_m=0.8)
    course = resolve_course(world)
    events: dict = {}
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5), heading=None)
    env_cfg = SimpleNamespace(
        events=events,
        commands={"twist": SimpleNamespace(
            ranges=ranges, entity_name="robot", debug_vis=False)},
        curriculum={})
    manifest = SimpleNamespace(
        course=course,
        task_shared={"goal": {
            "type": "waypoint_sequence", "waypoints": "auto",
            "success": {"tolerance_m": 0.2, "hold_s": 0.0},
        }},
        zones={})

    _reconcile_waypoint_course(env_cfg, manifest, train=True, robot=None)

    term = events.get("world_route_state_initialization")
    assert term is not None, "route RSI was not installed"
    boxes = term.params["support_boxes_m"]
    assert len(boxes) == len(course)
    for (x0, x1, y0, y1, top), primitive in zip(boxes, course):
        assert top == pytest.approx(
            primitive.position_m[2] + primitive.size_m[2] / 2)
        assert x1 - x0 == pytest.approx(primitive.size_m[0])
        assert y1 - y0 == pytest.approx(primitive.size_m[1])
