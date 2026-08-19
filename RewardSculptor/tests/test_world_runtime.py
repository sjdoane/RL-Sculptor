from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sculptor.world.channels import ChannelCatalog, compile_channel_catalog
from sculptor.world.runtime import (
    TorchWorldRewardRuntime,
    WorldChannelRecorder,
    WorldChannelRuntime,
    WorldRuntimeError,
)
from sculptor.world.author import author_environment
from tests.test_world_foundation import _task, _world


class _Scene(dict):
    pass


def _runtime(*, num_envs: int = 2) -> tuple[WorldChannelRuntime, SimpleNamespace]:
    # The runtime contract is robot-generic.  Use the installed fixed-base
    # arm/gripper descriptor here so locomotion-oriented names cannot creep
    # into the sampler implementation.
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
    catalog = compile_channel_catalog(world, task)

    robot_pos = np.zeros((num_envs, 3), dtype=np.float32)
    ball_pos = np.repeat(
        np.asarray([[4.0, 0.0, 0.5]], dtype=np.float32), num_envs, axis=0)
    ball = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=ball_pos,
        root_link_quat_w=np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            num_envs, axis=0),
        root_link_lin_vel_w=np.zeros((num_envs, 3), dtype=np.float32),
        root_link_ang_vel_w=np.zeros((num_envs, 3), dtype=np.float32),
    ))
    scene = _Scene({
        "robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=robot_pos)),
        "ball": ball,
        "authored_contact__desired__0": SimpleNamespace(
            data=SimpleNamespace(found=np.ones((num_envs, 1), dtype=bool))),
    })
    env = SimpleNamespace(
        scene=scene, num_envs=num_envs, step_dt=0.05)
    manifest = SimpleNamespace(
        task_shared=task["shared"], zones=world["shared"]["zones"],
        objects=world["shared"]["objects"], course=(),
    )
    return WorldChannelRuntime(
        env, catalog=catalog, manifest=manifest), ball


def test_runtime_records_exact_catalog_and_enforces_metric_firewall() -> None:
    runtime, _ball = _runtime()
    recorder = WorldChannelRecorder(runtime)

    first = recorder.append()
    second = recorder.append()
    arrays = recorder.finalize()

    success_name = "goal__score__success"
    inside_name = "goal__score__inside"
    distance_name = "object__ball__to_region__goal_mouth__distance"
    assert first.channels[inside_name].all()
    assert not first.channels[success_name].any()
    assert second.channels[success_name].all()  # 0.1 s continuous hold
    assert np.array_equal(arrays[distance_name], np.zeros((2, 2)))
    assert arrays[success_name].dtype == np.bool_
    assert arrays[success_name].shape == (2, 2)
    assert arrays["channel_catalog_hash"].item() == runtime.catalog.catalog_hash

    # Completion truth exists in the persisted metric surface but never in
    # the reward-facing dictionary.  Shared distance/state remains available.
    assert success_name not in first.reward_info
    assert inside_name not in first.reward_info
    assert distance_name in first.reward_info
    assert "object__ball__pos_w" in first.reward_info


def test_success_hold_resets_after_region_exit() -> None:
    runtime, ball = _runtime(num_envs=1)
    assert not runtime.sample().channels["goal__score__success"].item()
    assert runtime.sample().channels["goal__score__success"].item()

    ball.data.root_link_pos_w[...] = np.asarray([[8.0, 0.0, 0.5]])
    outside = runtime.sample()
    assert not outside.channels["goal__score__inside"].item()
    assert not outside.channels["goal__score__success"].item()
    assert outside.channels[
        "object__ball__to_region__goal_mouth__distance"].item() > 0.0

    ball.data.root_link_pos_w[...] = np.asarray([[4.0, 0.0, 0.5]])
    assert not runtime.sample().channels["goal__score__success"].item()


def test_event_success_requires_hold_duration_inside_finish() -> None:
    draft = author_environment(
        "Slalom around four boxes, then jump at the finish and hold still "
        "for 2 seconds.",
        robot_capability_id="unitree_g1:base",
    )
    task = draft.task_spec
    world = draft.world_spec
    catalog = compile_channel_catalog(world, task)
    finish = np.asarray(
        world["shared"]["zones"]["finish"]["center_m"] + [0.8],
        dtype=np.float32,
    )[None, :]
    scene_items = {
        "robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=finish.copy(),
            root_link_quat_w=np.asarray(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            root_link_lin_vel_w=np.zeros((1, 3), dtype=np.float32),
            root_link_ang_vel_w=np.zeros((1, 3), dtype=np.float32),
        )),
    }
    for name, record in world["shared"]["objects"].items():
        position = np.asarray(
            record["nominal"]["pose"]["position_m"], dtype=np.float32
        )[None, :]
        scene_items[name] = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=position,
            root_link_quat_w=np.asarray(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            root_link_lin_vel_w=np.zeros((1, 3), dtype=np.float32),
            root_link_ang_vel_w=np.zeros((1, 3), dtype=np.float32),
        ))
    for index in range(4):
        scene_items[f"authored_contact__forbidden__{index}"] = (
            SimpleNamespace(data=SimpleNamespace(
                found=np.zeros((1, 1), dtype=bool),
            ))
        )
    scene = _Scene(scene_items)
    command = SimpleNamespace(
        event_sequence_id="route_jump_hold",
        event_phase=np.asarray([1], dtype=np.int32),
        event_phase_height_delta=np.asarray([0.2], dtype=np.float32),
        event_phase_vertical_velocity=np.asarray([0.0], dtype=np.float32),
        event_sequence_violation=np.asarray([False]),
    )

    class Manager:
        active_terms = ("route",)

        @staticmethod
        def get_term(name):
            assert name == "route"
            return command

    env = SimpleNamespace(
        scene=scene,
        num_envs=1,
        step_dt=0.02,
        command_manager=Manager(),
    )
    manifest = SimpleNamespace(
        task_shared=task["shared"],
        zones=world["shared"]["zones"],
        objects={
            name: {
                "fixed": True,
                "position_m": record["nominal"]["pose"]["position_m"],
            }
            for name, record in world["shared"]["objects"].items()
        },
        course=(),
    )
    runtime = WorldChannelRuntime(
        env, catalog=catalog, manifest=manifest)
    runtime._waypoint_index[:] = len(runtime._waypoints)
    runtime._last_waypoint_complete[:] = True
    success_name = "goal__complete_slalom_and_stop__success"

    def sample():
        env._sculptor_event_reward_snapshot = {
            "route_jump_hold": {
                "event_phase": command.event_phase.copy(),
                "event_phase_height_delta": (
                    command.event_phase_height_delta.copy()),
                "event_phase_vertical_velocity": (
                    command.event_phase_vertical_velocity.copy()),
                "event_sequence_violation": (
                    command.event_sequence_violation.copy()),
                "event_support_contact_0": np.asarray([True]),
                "event_support_contact_1": np.asarray([True]),
            },
        }
        return runtime.sample()

    # Raw route completion and arbitrary time in JUMP are not task success.
    for _ in range(110):
        assert not sample().channels[success_name].item()

    command.event_phase[:] = 2
    command.event_sequence_violation[:] = True
    invalid_hold = sample()
    phase_name = "event__route_jump_hold__phase"
    assert invalid_hold.channels[phase_name].item() == 2
    assert invalid_hold.reward_info[phase_name].item() == -1
    for _ in range(109):
        assert not sample().channels[success_name].item()
    command.event_sequence_violation[:] = False
    for _ in range(99):
        assert not sample().channels[success_name].item()
    assert sample().channels[success_name].item()

    # Leaving the immutable finish disk resets the declared HOLD dwell.
    scene["robot"].data.root_link_pos_w[0, 0] += 1.0
    assert not sample().channels[success_name].item()
    scene["robot"].data.root_link_pos_w[:] = finish
    for _ in range(99):
        assert not sample().channels[success_name].item()
    assert sample().channels[success_name].item()

    # Simulator reset clears temporal proof even if the next state is HOLD.
    runtime.reset(np.asarray([0]))
    assert not sample().channels[success_name].item()


def test_runtime_reset_clears_only_requested_environment_state() -> None:
    runtime, _ball = _runtime(num_envs=2)
    runtime.sample()
    assert runtime.sample().channels["goal__score__success"].all()
    runtime._waypoint_index[:] = [3, 4]
    runtime._last_waypoint_distance[:] = [1.5, 2.5]
    runtime._last_waypoint_complete[:] = True

    runtime.reset(np.asarray([0]))

    assert runtime._waypoint_index.tolist() == [0, 4]
    assert runtime._last_waypoint_distance.tolist() == [0.0, 2.5]
    assert runtime._last_waypoint_complete.tolist() == [False, True]
    after = runtime.sample().channels["goal__score__success"]
    assert after.tolist() == [False, True]
    with pytest.raises(WorldRuntimeError, match="reset env_ids must be within"):
        runtime.reset(np.asarray([runtime.num_envs]))


def test_torch_reward_runtime_never_exposes_metric_only_channels() -> None:
    torch = pytest.importorskip("torch")
    runtime, _ball = _runtime(num_envs=2)
    env = runtime.env
    env.device = "cpu"
    for entity_name in ("robot", "ball"):
        data = env.scene[entity_name].data
        for key, value in vars(data).items():
            setattr(data, key, torch.as_tensor(value))
    env.scene["authored_contact__desired__0"].data.found = torch.ones(
        (2, 1), dtype=torch.bool)

    reward_runtime = TorchWorldRewardRuntime(
        env, catalog=runtime.catalog, manifest=runtime.manifest)
    info = reward_runtime.sample()

    assert "object__ball__pos_w" in info
    assert "object__ball__to_region__goal_mouth__distance" in info
    assert "contact__desired__0" in info
    assert "goal__score__success" not in info
    assert "goal__score__inside" not in info


def test_event_channels_preserve_reward_time_transition_parity() -> None:
    torch = pytest.importorskip("torch")
    draft = author_environment(
        "Slalom around four boxes, then jump at the finish and hold still "
        "for 2 seconds.",
        robot_capability_id="unitree_g1:base",
    )
    full_catalog = compile_channel_catalog(
        draft.world_spec, draft.task_spec)
    event_specs = tuple(
        spec for spec in full_catalog.channels
        if (
            spec.producer.startswith("event_phase_")
            or spec.producer == "event_sequence_violation"
        )
    )
    catalog = ChannelCatalog.build(
        world_hash=full_catalog.world_hash,
        task_hash=full_catalog.task_hash,
        channels=event_specs,
    )
    finish = draft.world_spec["shared"]["zones"]["finish"]["center_m"]
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[finish[0], finish[1], 0.8]]),
        root_link_lin_vel_w=torch.tensor([[0.0, 0.0, 0.2]]),
    ))
    term = SimpleNamespace(
        event_sequence_id="route_jump_hold",
        event_phase=torch.tensor([0]),
        _event_phase_height_anchor=torch.tensor([float("nan")]),
        _event_takeoff_height_anchor=torch.tensor([float("nan")]),
        event_sequence_violation=torch.tensor([False]),
        event_support_contacts=torch.tensor([[True, True]]),
    )

    class Manager:
        active_terms = ("route",)

        @staticmethod
        def get_term(_name):
            return term

    scene = _Scene({"robot": robot})
    env = SimpleNamespace(
        scene=scene,
        num_envs=1,
        step_dt=0.02,
        device="cpu",
        command_manager=Manager(),
    )
    manifest = SimpleNamespace(
        task_shared=draft.task_spec["shared"],
        zones=draft.world_spec["shared"]["zones"],
        objects={},
        course=(),
    )
    reward_runtime = TorchWorldRewardRuntime(
        env, catalog=catalog, manifest=manifest)
    recorder_runtime = WorldChannelRuntime(
        env, catalog=catalog, manifest=manifest)
    recorder_runtime._waypoint_index[:] = len(recorder_runtime._waypoints)
    recorder_runtime._last_waypoint_complete[:] = True

    phase_name = "event__route_jump_hold__phase"
    delta_name = "event__route_jump_hold__phase_height_delta"
    velocity_name = "event__route_jump_hold__base_vertical_velocity"
    violation_name = "event__route_jump_hold__violation"

    def reward_then_record(after_reward=None):
        reward_values = reward_runtime.sample()
        if after_reward is not None:
            after_reward()
        recorded = recorder_runtime.sample().channels
        for name in (phase_name, delta_name, velocity_name):
            np.testing.assert_allclose(
                recorded[name], reward_values[name].cpu().numpy())
        return recorded

    # ROUTE reward is captured before the command transitions to JUMP.
    reward_then_record(after_reward=lambda: (
        term.event_phase.fill_(1),
        term._event_phase_height_anchor.fill_(0.8),
        term._event_takeoff_height_anchor.fill_(0.8),
    ))

    # At the apex, reward and trajectory share current physics-derived delta/vz.
    robot.data.root_link_pos_w[:, 2] = 1.05
    robot.data.root_link_lin_vel_w[:, 2] = 0.0
    apex = reward_then_record()
    assert apex[phase_name].item() == 1
    assert apex[delta_name].item() == pytest.approx(0.25)

    # Landing transition happens after reward; this row remains JUMP.
    robot.data.root_link_pos_w[:, 2] = 0.82
    robot.data.root_link_lin_vel_w[:, 2] = -0.1
    landing = reward_then_record(
        after_reward=lambda: term.event_phase.fill_(2))
    assert landing[phase_name].item() == 1

    # The next reward-time row is the first official HOLD row.
    hold = reward_then_record()
    assert hold[phase_name].item() == 2

    # Violation truth remains metric-only.  Shared physical shaping is zeroed
    # after invalidation, so an invalid JUMP cannot farm apex/velocity reward.
    term.event_phase.fill_(1)
    term.event_sequence_violation.fill_(True)
    robot.data.root_link_pos_w[:, 2] = 1.2
    robot.data.root_link_lin_vel_w[:, 2] = 2.0
    invalid_reward = reward_runtime.sample()
    invalid_recorded = recorder_runtime.sample().channels
    assert invalid_reward[phase_name].item() == -1
    assert invalid_recorded[phase_name].item() == 1
    assert invalid_recorded[violation_name].item()
    assert invalid_reward[delta_name].item() == 0.0
    assert invalid_reward[velocity_name].item() == 0.0
    assert invalid_recorded[delta_name].item() == pytest.approx(0.4)
    assert invalid_recorded[velocity_name].item() == pytest.approx(2.0)

    # A repeated jump invalidates a raw HOLD phase.  Evidence retains both the
    # raw terminal phase and violation latch, while reward sees only the
    # out-of-domain invalid sentinel and cannot farm terminal shaping.
    term.event_phase.fill_(2)
    invalid_hold_reward = reward_runtime.sample()
    invalid_hold_recorded = recorder_runtime.sample().channels
    assert invalid_hold_reward[phase_name].item() == -1
    assert invalid_hold_recorded[phase_name].item() == 2
    assert invalid_hold_recorded[violation_name].item()


def test_region_relative_channels_are_local_to_each_environment_origin() -> None:
    """Replicated worlds must expose the same local goal vector in every env."""
    torch = pytest.importorskip("torch")
    runtime, ball = _runtime(num_envs=2)
    origins = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]], dtype=np.float32)
    robot_local = np.asarray([1.0, 2.0, 0.0], dtype=np.float32)
    ball_local = np.asarray([4.0, 0.0, 0.5], dtype=np.float32)
    runtime.env.scene.env_origins = origins.copy()
    runtime.env.scene["robot"].data.root_link_pos_w = origins + robot_local
    ball.data.root_link_pos_w = origins + ball_local

    channel = "region__goal_mouth__relative"
    expected = np.repeat(
        np.asarray([[3.0, -2.0, 0.5]], dtype=np.float32), 2, axis=0)
    np.testing.assert_allclose(runtime.sample().channels[channel], expected)

    env = runtime.env
    env.device = "cpu"
    env.scene.env_origins = torch.as_tensor(origins)
    for entity_name in ("robot", "ball"):
        data = env.scene[entity_name].data
        for key, value in vars(data).items():
            setattr(data, key, torch.as_tensor(value))
    env.scene["authored_contact__desired__0"].data.found = torch.ones(
        (2, 1), dtype=torch.bool)
    reward_runtime = TorchWorldRewardRuntime(
        env, catalog=runtime.catalog, manifest=runtime.manifest)
    torch.testing.assert_close(
        reward_runtime.sample()[channel], torch.as_tensor(expected))


def test_waypoints_cannot_advance_against_misaligned_physical_geometry() -> None:
    """Virtual local zones must not score while fixed boxes remain global."""
    origins = np.asarray(
        [[7.0, -7.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    waypoint = np.asarray([2.0, 0.85, 0.0], dtype=np.float32)
    robot_pos = origins + waypoint
    # This is the production failure mode: one global box pose was reused for
    # every replicated environment instead of adding each env origin.
    box_pos = np.repeat(
        np.asarray([[2.0, 0.0, 0.375]], dtype=np.float32), 2, axis=0)
    scene = _Scene({
        "robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=robot_pos)),
        "box_01": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=box_pos)),
    })
    scene.env_origins = origins
    env = SimpleNamespace(scene=scene, num_envs=2, step_dt=0.02)
    catalog = ChannelCatalog.build(
        world_hash="0" * 64, task_hash="1" * 64, channels=())
    manifest = SimpleNamespace(
        task_shared={"goal": {
            "type": "waypoint_sequence",
            "waypoints": ["waypoint_01"],
            "success": {"tolerance_m": 0.35},
        }},
        zones={
            "waypoint_01": {
                "kind": "disk",
                "center_m": waypoint.tolist(),
                "radius_m": 0.35,
            },
        },
        objects={
            "box_01": {
                "fixed": True,
                "position_m": [2.0, 0.0, 0.375],
            },
        },
        course=(),
    )
    runtime = WorldChannelRuntime(
        env, catalog=catalog, manifest=manifest)

    runtime.sample()
    assert runtime._waypoint_index.tolist() == [0, 1]
    assert runtime._last_waypoint_complete.tolist() == [False, True]

    # Once each physical object is in nominal-local + env-origin coordinates,
    # both otherwise-identical local waypoint entries become authoritative.
    scene["box_01"].data.root_link_pos_w = origins + np.asarray(
        [2.0, 0.0, 0.375], dtype=np.float32)
    runtime.sample()
    assert runtime._waypoint_index.tolist() == [1, 1]
    assert runtime._last_waypoint_complete.tolist() == [True, True]


def test_runtime_reset_consumes_authoritative_route_rsi_index() -> None:
    runtime, _ball = _runtime(num_envs=2)
    runtime.env._sculptor_waypoint_start_index = np.asarray([2, 4])
    runtime.reset(np.asarray([1]))
    assert runtime._waypoint_index.tolist() == [0, 4]
