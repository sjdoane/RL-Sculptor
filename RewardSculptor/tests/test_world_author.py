"""Fast contract tests for robot-agnostic world authoring/clarification."""

from __future__ import annotations

import copy
import json

import pytest

from sculptor.world.author import (
    CLARIFICATION_VERSION,
    MAX_QUESTIONS_PER_PAGE,
    AuthoringError,
    ClarificationAnswer,
    ClarificationSubmission,
    StaleClarificationError,
    WorldAuthor,
    apply_clarifications,
    author_environment,
    default_clarification_submission,
)
from sculptor.world.author import _parse_count
from sculptor.world.capabilities import CapabilityError
from sculptor.world.task_spec import validate_task_spec
from sculptor.world.world_spec import validate_world_spec


_COMPACT_LOW_RAIL_PROMPT = (
    "Build a compact low-rail course with four low fixed rails centered at "
    "x=0.35, 0.85, 1.35, and 1.85 m at y=0. Each rail is 0.10 by 0.60 by "
    "0.06 m. Put ordered landing disks at x=0.65, 1.15, 1.65, and 2.15 m "
    "with radius 0.30 m, then a finish at (2.55, 0) with radius 0.45 m. "
    "Perform four distinct support-cycle hops without touching the rails, "
    "then hold in finish for 2 seconds in an 8 second episode."
)


@pytest.mark.parametrize(
    ("prompt", "robot", "goal_type"),
    [
        (
            "Stay stable and walk or jump on uneven terrain.",
            "unitree_g1:base",
            "robot_to_region",
        ),
        (
            "Learn parkour over a course of boxes.",
            "unitree_g1:base",
            "waypoint_sequence",
        ),
        (
            "Use a gripper robot to move an object into a goal region.",
            "yam:parallel_gripper",
            "object_to_region",
        ),
    ],
)
def test_offline_acceptance_drafts_are_strict_and_valid(
    prompt: str, robot: str, goal_type: str,
) -> None:
    draft = author_environment(prompt, robot_capability_id=robot)

    assert draft.capability_id == robot
    assert draft.task_spec["shared"]["goal"]["type"] == goal_type
    assert validate_world_spec(draft.world_spec) == []
    assert validate_task_spec(draft.task_spec, world=draft.world_spec) == []
    assert draft.world_spec["meta"]["parameter_provenance"]


def test_offline_slalom_jump_authors_exact_one_shot_event_program() -> None:
    prompt = (
        "Have the G1 slalom around four boxes without touching them, then "
        "jump at the finish and hold still for 2 seconds."
    )
    draft = author_environment(
        prompt,
        robot_capability_id="unitree_g1:base",
    )

    task = draft.task_spec
    event = task["shared"]["event_sequence"]
    assert [phase["id"] for phase in event["phases"]] == [
        "route", "jump", "hold",
    ]
    assert event["phases"][0]["until"] == {"event": "goal_complete"}
    assert event["phases"][1]["until"]["event"] \
        == "bilateral_support_cycle"
    assert event["phases"][1]["until"]["support_contacts"] == [
        ["robot:left_foot", "world:terrain"],
        ["robot:right_foot", "world:terrain"],
    ]
    assert event["phases"][1]["until"]["min_height_delta_m"] == 0.18
    assert event["phases"][2] == {
        "id": "hold", "terminal": True, "minimum_hold_s": 2.0,
    }
    assert task["shared"]["goal"]["success"]["hold_s"] == 0.0
    assert {
        float(zone["radius_m"])
        for zone in draft.world_spec["shared"]["zones"].values()
    } == {0.35}
    assert task["shared"]["termination"]["episode_length_s"] >= 24.0
    assert task["train"]["event_phase_sampling"] == {
        "route": 0.5, "jump": 0.4, "hold": 0.1,
    }
    assert validate_task_spec(task, world=draft.world_spec) == []

    too_short = copy.deepcopy(task)
    too_short["shared"]["event_sequence"]["phases"][2][
        "minimum_hold_s"
    ] = 1.99
    errors = validate_task_spec(too_short, world=draft.world_spec)
    assert any("minimum_hold_s" in error and "below 2" in error
               for error in errors)

    wrong_sampling = copy.deepcopy(task)
    wrong_sampling["train"]["event_phase_sampling"] = {
        "route": 0.6, "jump": 0.4, "hold": 0.1,
    }
    errors = validate_task_spec(wrong_sampling, world=draft.world_spec)
    assert any("probabilities must sum to 1" in error for error in errors)

    duplicate_support = copy.deepcopy(task)
    supports = duplicate_support["shared"]["event_sequence"]["phases"][1][
        "until"
    ]["support_contacts"]
    supports[1] = list(supports[0])
    errors = validate_task_spec(duplicate_support, world=draft.world_spec)
    assert any("distinct bodies" in error for error in errors)

    early_termination = copy.deepcopy(task)
    early_termination["shared"]["termination"][
        "success_ends_episode"
    ] = True
    errors = validate_task_spec(early_termination, world=draft.world_spec)
    assert any("event sequences require false" in error for error in errors)


def test_gripper_resolution_is_generic_and_impossible_requests_are_precise() -> None:
    # The unique installed grasp-capable descriptor is selected from declared
    # capabilities; the author contains no G1/task-name special case.
    draft = author_environment(
        "Use a gripper robot to move an object into a goal region."
    )
    assert draft.capability_id == "yam:parallel_gripper"
    assert "grasp" in draft.world_spec["shared"]["robot"][
        "required_capabilities"
    ]
    assert not any(
        question.parameter_path == "/world/shared/robot/capability_id"
        for question in draft.clarification_plan.questions
    )

    with pytest.raises(CapabilityError, match="missing required capabilities.*grasp"):
        author_environment(
            "Use a gripper to put the ball into a goal.",
            robot_capability_id="unitree_g1:base",
        )
    with pytest.raises(CapabilityError) as exc:
        author_environment(
            "Get the humanoid with a gripper to score a ball into a goal."
        )
    message = str(exc.value)
    assert "grasp" in message and "whole_body" in message
    assert "yam:parallel_gripper" in message


@pytest.mark.parametrize("prompt", [
    "Pick and place the ball in the goal region.",
    "Pick the ball up and place it in the goal region.",
])
def test_pick_and_place_phrasings_require_real_grasp_capability(
    prompt: str,
) -> None:
    draft = author_environment(prompt)

    assert draft.capability_id == "yam:parallel_gripper"
    assert "grasp" in draft.world_spec["shared"]["robot"][
        "required_capabilities"
    ]


def test_ambiguous_push_auto_selection_stays_robot_portable() -> None:
    draft = author_environment("Push the ball into the goal region.")
    robot_question = next(
        question for question in draft.clarification_plan.questions
        if question.parameter_path == "/world/shared/robot/capability_id"
    )

    # The default task uses a semantic role supported by at least two eligible
    # descriptors, so robot choice is a real bounded clarification rather than
    # an invalid ID-only substitution.
    assert len(robot_question.choices) >= 2
    assert draft.capability_id in {
        choice.value for choice in robot_question.choices
    }
    assert validate_task_spec(draft.task_spec, world=draft.world_spec) == []


def test_external_robot_descriptor_authors_without_robot_name_branches(
    tmp_path,
) -> None:
    descriptor = tmp_path / "pinch_arm.json"
    descriptor.write_text(json.dumps({
        "capability_id": "example:pinch_arm",
        "asset_id": "example_arm_asset",
        "asset_hash": "sha256:example",
        "root_body": "mount",
        "body_roles": {
            "base": ["mount"],
            "gripper": ["left_pad", "right_pad"],
            "end_effector": ["wrist_link"],
        },
        "site_roles": {"grasp": ["pinch_center"]},
        "capabilities": ["manipulation", "grasp", "push"],
        "geometry": {
            "standing_height_m": 0.0,
            "leg_length_m": 0.0,
            "foot_length_m": 0.0,
            "reach_radius_m": 0.6,
            "max_step_height_m": 0.0,
            "max_gap_m": 0.0,
        },
        "supported_commands": ["object_target"],
        "supported_observations": [
            "proprioception", "object_relative", "region_relative",
            "end_effector_relative",
        ],
        "contact_capacity": 16,
    }), encoding="utf-8")

    draft = author_environment(
        "Use a gripper robot to move an object into a goal region.",
        robot_capability_id="example:pinch_arm",
        robot_descriptor_paths=[descriptor],
    )

    assert draft.capability_id == "example:pinch_arm"
    assert draft.world_spec["shared"]["robot"]["descriptor_path"] \
        == str(descriptor.resolve())
    assert draft.task_spec["shared"]["contacts"]["desired"] == [[
        "robot:gripper", "object:target_object",
    ]]
    assert validate_world_spec(draft.world_spec) == []
    assert validate_task_spec(draft.task_spec, world=draft.world_spec) == []


def test_every_load_bearing_default_is_paginated_and_disclosed() -> None:
    draft = author_environment(
        "Learn parkour over a course of boxes.",
        robot_capability_id="unitree_g1:base",
    )
    questions = draft.clarification_plan.questions
    report = draft.underspecification_report

    assert len(questions) > MAX_QUESTIONS_PER_PAGE
    assert all(
        1 <= len(page.questions) <= MAX_QUESTIONS_PER_PAGE
        for page in draft.clarification_plan.pages
    )
    assert sum(len(page.questions) for page in draft.clarification_plan.pages) \
        == len(questions)
    queued = {question.parameter_path for question in questions}
    assert queued == set(report.defaulted_load_bearing_paths)
    assert queued == {
        record.path for record in report.parameters
        if record.load_bearing and record.provenance == "default"
    }
    for question in questions:
        assert 2 <= len(question.choices) <= 4
        assert question.default_choice_id in {
            choice.choice_id for choice in question.choices
        }
        payload = question.to_dict()
        assert payload["system_default"]["choice_id"] == "system_default"
        assert "default:" in payload["system_default"]["label"]
        assert question.default_reason


def test_timeout_defaults_record_provenance_and_revalidate_each_page() -> None:
    draft = author_environment(
        "Use a gripper robot to move an object into a goal region.",
        robot_capability_id="yam:parallel_gripper",
    )
    submission = default_clarification_submission(draft, timeout=True)
    applied = apply_clarifications(draft, submission)

    assert validate_world_spec(applied.world_spec) == []
    assert validate_task_spec(applied.task_spec, world=applied.world_spec) == []
    assert len(applied.clarification_ledger["answers"]) == len(
        draft.clarification_plan.questions
    )
    assert {
        answer["source"] for answer in applied.clarification_ledger["answers"]
    } == {"timeout_default"}
    assert applied.clarification_ledger["unanswered_question_ids"] == []
    assert applied.underspecification_report.defaulted_load_bearing_paths == ()
    for answer in applied.clarification_ledger["answers"]:
        assert applied.world_spec["meta"]["parameter_provenance"][
            answer["parameter_path"]
        ] == "timeout_default"


def test_user_answer_changes_value_and_is_attributed_to_user() -> None:
    draft = author_environment(
        "Stay stable and walk or jump on uneven terrain.",
        robot_capability_id="unitree_g1:base",
    )
    question = next(
        item for item in draft.clarification_plan.questions
        if item.parameter_path.endswith("/slope_deg")
    )
    alternative = next(
        choice for choice in question.choices
        if choice.choice_id != question.default_choice_id
    )
    default_submission = default_clarification_submission(
        draft, timeout=False)
    submission = ClarificationSubmission(
        version=CLARIFICATION_VERSION,
        draft_hash=draft.draft_hash,
        question_set_hash=draft.clarification_plan.question_set_hash,
        answers=tuple(
            ClarificationAnswer(
                question_id=answer.question_id,
                choice_id=(alternative.choice_id
                           if answer.question_id == question.question_id
                           else answer.choice_id),
                source=("user" if answer.question_id == question.question_id
                        else answer.source),
            )
            for answer in default_submission.answers
        ),
    )
    applied = apply_clarifications(draft, submission)

    assert applied.world_spec["shared"]["terrain"]["sub_terrains"]["slope"][
        "nominal"
    ]["slope_deg"] == alternative.value
    assert applied.world_spec["meta"]["parameter_provenance"][
        question.parameter_path
    ] == "user"


def test_stale_and_malformed_answer_envelopes_fail_before_apply() -> None:
    draft = author_environment(
        "Stay stable and walk on uneven terrain.",
        robot_capability_id="unitree_g1:base",
    )
    valid = default_clarification_submission(draft, timeout=False)

    stale_draft = ClarificationSubmission(
        version=valid.version,
        draft_hash="0" * 64,
        question_set_hash=valid.question_set_hash,
        answers=valid.answers,
    )
    with pytest.raises(StaleClarificationError, match="draft hash"):
        apply_clarifications(draft, stale_draft)

    stale_questions = ClarificationSubmission(
        version=valid.version,
        draft_hash=valid.draft_hash,
        question_set_hash="f" * 64,
        answers=valid.answers,
    )
    with pytest.raises(StaleClarificationError, match="question-set hash"):
        apply_clarifications(draft, stale_questions)

    question = draft.clarification_plan.questions[0]
    bad_source = ClarificationSubmission(
        version=valid.version,
        draft_hash=valid.draft_hash,
        question_set_hash=valid.question_set_hash,
        answers=(ClarificationAnswer(
            question_id=question.question_id,
            choice_id="system_default",
            source="user",
        ),),
    )
    with pytest.raises(AuthoringError, match="system_default answers"):
        apply_clarifications(draft, bad_source)

    incomplete = ClarificationSubmission(
        version=valid.version,
        draft_hash=valid.draft_hash,
        question_set_hash=valid.question_set_hash,
        answers=valid.answers[:-1],
    )
    with pytest.raises(AuthoringError, match="submission is incomplete"):
        apply_clarifications(draft, incomplete)


def test_injected_author_model_has_small_strict_interface() -> None:
    seed = author_environment(
        "Stay stable and walk on uneven terrain.",
        robot_capability_id="unitree_g1:base",
    )

    class Model:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def generate_authoring(self, request):
            self.requests.append(copy.deepcopy(dict(request)))
            return {
                "world_spec": copy.deepcopy(seed.world_spec),
                "task_spec": copy.deepcopy(seed.task_spec),
                "parameter_provenance": copy.deepcopy(
                    seed.world_spec["meta"]["parameter_provenance"]
                ),
            }

    model = Model()
    draft = WorldAuthor(model).author(
        "Stay stable and walk on uneven terrain.",
        robot_capability_id="unitree_g1:base",
    )
    assert validate_world_spec(draft.world_spec) == []
    assert model.requests[0]["selected_robot"]["capability_id"] \
        == "unitree_g1:base"
    assert "required_capabilities" in model.requests[0]

    class BadModel:
        def generate_authoring(self, request):
            return {
                "world_spec": copy.deepcopy(seed.world_spec),
                "task_spec": copy.deepcopy(seed.task_spec),
                "arbitrary_code": "do_not_execute()",
            }

    with pytest.raises(AuthoringError, match="unknown keys.*arbitrary_code"):
        WorldAuthor(BadModel()).author(
            "Stay stable and walk on uneven terrain.",
            robot_capability_id="unitree_g1:base",
        )


def test_parse_count_reads_requested_number():
    """The offline parkour template now reads a COUNT from the prompt — the
    "prompt for 4 boxes, get 3" bug was that the template ignored the number."""
    assert _parse_count("generate 4 boxes", default=3) == 4
    assert _parse_count("a parkour course with five platforms", default=3) == 5
    assert _parse_count("climb two boxes then jump off", default=3) == 2
    assert _parse_count(
        "four progressively taller, high-friction boxes in a straight line",
        default=3,
    ) == 4
    assert _parse_count("four low fixed rails", default=3) == 4
    # unquantified → nominal default
    assert _parse_count("build a parkour course", default=3) == 3
    # articles are not counts ("a course with 8 steps" is 8, not 1)
    assert _parse_count("a box course with 8 steps", default=3) == 8
    # clamped to a sane range
    assert _parse_count("999 boxes", default=3) == 12


def test_parkour_authors_requested_platform_count():
    draft = author_environment(
        "generate a parkour course with 4 boxes",
        robot_capability_id="unitree_g1:base")
    world = draft.world_spec
    assert validate_world_spec(world) == []
    platforms = [c["id"] for c in world["shared"]["obstacles"]["course"]
                 if c["element"] == "platform"]
    assert platforms == ["box_01", "box_02", "box_03", "box_04"]
    # variation targets stay valid ids for the authored length
    targets = {v["target"] for v in world["train"]["variations"]}
    course_ids = {c["id"] for c in world["shared"]["obstacles"]["course"]}
    for t in targets:
        ref = t.split("@")[1].split("/")[0]
        assert ref in course_ids, (t, course_ids)


def test_object_prompt_authors_ball_and_soccer_goal():
    draft = author_environment(
        "Use a gripper robot to move a ball into a soccer goal",
        robot_capability_id="yam:parallel_gripper")
    world = draft.world_spec
    assert validate_world_spec(world) == []
    objects = world["shared"]["objects"]
    assert objects["target_object"]["shape"] == "sphere"        # the ball
    assert objects["target_goal"]["shape"] == "frame"           # the soccer goal
    assert objects["target_goal"]["fixed"] is True


def test_object_prompt_without_goal_word_has_no_frame():
    draft = author_environment(
        "Use a gripper robot to move a cube into the region",
        robot_capability_id="yam:parallel_gripper")
    objects = draft.world_spec["shared"]["objects"]
    assert "target_goal" not in objects


def test_intent_routing_object_vs_parkour_precedence():
    """Object-manipulation prompts are not stolen by the generic box/platform
    parkour cues, and generic climb/box cues still ground as parkour."""
    from sculptor.world.author import _intent
    # object-manipulation wins even when it mentions a box/platform
    assert _intent("push the block onto the platform") == "object_to_region"
    assert _intent("move the cube into the goal region") == "object_to_region"
    assert _intent("score the ball into the soccer goal") == "object_to_region"
    # generic climb/box cues (no object task) → parkour
    assert _intent("generate 4 boxes") == "parkour"
    assert _intent("climb two boxes then jump off") == "parkour"
    assert _intent("climb onto the platform") == "parkour"
    # specific parkour cues still win
    assert _intent("parkour course of boxes") == "parkour"


def test_compact_low_rail_profile_is_exact_and_policy_compatible() -> None:
    """The named UI profile must preserve exact physical and observation truth."""
    from sculptor.world.author import _intent

    assert _intent(_COMPACT_LOW_RAIL_PROMPT) == "compact_low_rails"
    assert _intent("Build four low rails for hopping") == "compact_low_rails"
    draft = author_environment(
        _COMPACT_LOW_RAIL_PROMPT,
        robot_capability_id="unitree_g1:base",
    )
    world = draft.world_spec
    task = draft.task_spec

    assert validate_world_spec(world) == []
    assert validate_task_spec(task, world=world) == []
    assert draft.clarification_plan.questions == ()
    assert world["shared"]["robot"]["required_capabilities"] == [
        "jump", "locomotion",
    ]
    assert list(world["shared"]["objects"]) == [
        "rail_01", "rail_02", "rail_03", "rail_04",
    ]
    assert [
        item["nominal"]["pose"]["position_m"]
        for item in world["shared"]["objects"].values()
    ] == [
        [0.35, 0.0, 0.03],
        [0.85, 0.0, 0.03],
        [1.35, 0.0, 0.03],
        [1.85, 0.0, 0.03],
    ]
    assert all(
        item["shape"] == "box"
        and item["fixed"] is True
        and item["nominal"]["size_m"] == [0.10, 0.60, 0.06]
        for item in world["shared"]["objects"].values()
    )
    assert world["shared"]["zones"] == {
        "waypoint_01": {
            "kind": "disk", "center_m": [0.65, 0.0], "radius_m": 0.30,
        },
        "waypoint_02": {
            "kind": "disk", "center_m": [1.15, 0.0], "radius_m": 0.30,
        },
        "waypoint_03": {
            "kind": "disk", "center_m": [1.65, 0.0], "radius_m": 0.30,
        },
        "waypoint_04": {
            "kind": "disk", "center_m": [2.15, 0.0], "radius_m": 0.30,
        },
        "finish": {
            "kind": "disk", "center_m": [2.55, 0.0], "radius_m": 0.45,
        },
    }
    assert world["train"]["variations"] == []

    waypoints = [
        "waypoint_01", "waypoint_02", "waypoint_03", "waypoint_04", "finish",
    ]
    assert task["shared"]["goal"] == {
        "id": "complete_compact_rail_course",
        "type": "waypoint_sequence",
        "waypoints": waypoints,
        "success": {
            "predicate": "sequence_complete",
            "hold_s": 2.0,
            "tolerance_m": 0.30,
            "ordered": True,
        },
    }
    assert task["shared"]["contacts"]["forbidden"] == [
        ["robot:any", f"object:rail_{index:02d}"]
        for index in range(1, 5)
    ]
    assert task["shared"]["termination"]["episode_length_s"] == 8.0
    assert task["shared"]["termination"]["success_ends_episode"] is False
    assert task["shared"]["observations"] == {
        "proprioception": True,
        "height_scan": True,
        "object_relative": [],
        "region_relative": waypoints,
    }
    assert "event_sequence" not in task["shared"]


def test_slalom_preserves_obstacles_waypoints_and_terminal_dwell():
    from sculptor.world.author import _intent

    prompt = (
        "Run a slalom around four bright boxes through ordered waypoints, "
        "then enter a finish zone, stop, and remain there for 2 seconds."
    )
    assert _intent(prompt) == "slalom"
    draft = author_environment(
        prompt, robot_capability_id="unitree_g1:base")
    world, task = draft.world_spec, draft.task_spec
    assert validate_world_spec(world) == []
    assert validate_task_spec(task, world=world) == []
    boxes = {
        name: item for name, item in world["shared"]["objects"].items()
        if name.startswith("slalom_box_")
    }
    assert list(boxes) == [f"slalom_box_{i:02d}" for i in range(1, 5)]
    assert all(item["fixed"] and item["shape"] == "box"
               for item in boxes.values())
    goal = task["shared"]["goal"]
    assert goal["waypoints"] == [
        "waypoint_01", "waypoint_02", "waypoint_03", "waypoint_04", "finish"]
    assert goal["success"]["hold_s"] == 2.0
    assert task["shared"]["contacts"]["forbidden"] == [
        ["robot:any", f"object:slalom_box_{i:02d}"] for i in range(1, 5)]
    assert len(world["train"]["variations"]) == 4

    from sculptor.world.capabilities import resolve_robot_capability
    from sculptor.world.task_spec import parse_contact_selector

    cap = resolve_robot_capability("unitree_g1:base")
    resolved = parse_contact_selector("robot:any", world=world)
    assert resolved is not None
    assert resolved[0] == "robot"
    assert cap.root_body in resolved[1]


def test_fully_specified_slalom_does_not_repeat_clarifications():
    draft = author_environment(
        "Build a generous slalom with four bright orange boxes at roughly "
        "x=2.0, 3.5, 5.0, and 6.5. Each is 0.45 by 0.45 by 0.75 m. Use "
        "ordered waypoints approximately at alternating sides, a large finish "
        "zone, and stop there for 2 seconds without touching any box. Randomize "
        "each box lateral position by 0.08 m.",
        robot_capability_id="unitree_g1:base",
    )
    assert len(draft.clarification_plan.questions) <= 1
