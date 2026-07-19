from __future__ import annotations

import pytest

from sculptor.world.author import (
    apply_clarifications,
    author_environment,
    default_clarification_submission,
)
from sculptor.world.gates import run_admission_gates


@pytest.mark.parametrize(("prompt", "robot", "goal_type"), [
    (
        "Walk across uneven rough terrain and slopes to a finish region",
        "unitree_g1:base",
        "robot_to_region",
    ),
    (
        "Run a box parkour obstacle course with gaps and platforms",
        "unitree_go1:base",
        "waypoint_sequence",
    ),
    (
        "Use a gripper to move a ball into a goal region",
        "yam:parallel_gripper",
        "object_to_region",
    ),
])
def test_acceptance_prompt_compiles_through_all_admission_gates(
    prompt: str, robot: str, goal_type: str,
) -> None:
    draft = author_environment(prompt, robot_capability_id=robot)
    applied = apply_clarifications(
        draft, default_clarification_submission(draft, timeout=False))

    report, compiled = run_admission_gates(
        applied.world_spec, applied.task_spec, settle_steps=30)

    assert report.ok, [item.to_dict() for item in report.violations]
    assert compiled is not None
    assert compiled.robot.capability_id == robot
    assert applied.task_spec["shared"]["goal"]["type"] == goal_type
    assert compiled.resolved_eval.admission["ok"] is True
    assert compiled.channel_catalog.channels


def test_gripper_acceptance_uses_capabilities_not_robot_name() -> None:
    draft = author_environment(
        "Use a gripper to move a ball into a goal region",
        robot_capability_id="yam:parallel_gripper",
    )
    assert draft.world_spec["shared"]["robot"]["required_capabilities"] == [
        "grasp",
    ]
    assert draft.task_spec["shared"]["contacts"]["desired"] == [[
        "robot:gripper", "object:target_object",
    ]]
    # The task remains a generic entity/region composition; no soccer/G1/Yam
    # discriminator is encoded in the task or compiler-facing goal.
    assert draft.task_spec["shared"]["goal"]["type"] == "object_to_region"
