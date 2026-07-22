from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sculptor.world.channels import compile_channel_catalog
from sculptor.world.runtime import (
    TorchWorldRewardRuntime,
    WorldChannelRecorder,
    WorldChannelRuntime,
)
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
