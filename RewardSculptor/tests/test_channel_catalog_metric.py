from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.generated_metric import (
    _generated_metric_observables,
    compute_generated_metric,
)
from sculptor.eval.metric_validate import (
    _abstract_objective_program,
    _abstract_objective_probe,
    _abstract_route_window,
    discrimination_of_metric,
    validate_generated_metric,
)
from sculptor.eval.partition_gate import (
    build_partition_prompt_block,
    screen_edits,
)
from sculptor.world.author import author_environment
from sculptor.world.channels import (
    ChannelCatalog,
    catalog_fixture_arrays,
    compile_channel_catalog,
    validate_channel_catalog,
    validate_trajectory_channels,
)


def test_slalom_boxes_compile_as_planar_route_not_climbs():
    goal = (
        "Run a smooth slalom through four ordered waypoints, alternating "
        "around four boxes, then remain upright and still there continuously "
        "for 2 seconds."
    )
    assert _abstract_objective_program(goal) == [
        "move_forward", "recover", "dwell",
    ]
    assert _abstract_objective_program(
        "Jump onto four boxes and pause on each platform"
    ) == [
        "climb", "dwell", "climb", "dwell",
        "climb", "dwell", "climb", "dwell",
    ]


def test_explicit_repeated_hops_preserve_takeoff_landing_cardinality():
    goal = (
        "Move forward and perform four distinct one-leg hops over four low "
        "rails, landing after each, then recover upright and hold still."
    )
    assert _abstract_objective_program(goal) == [
        "move_forward",
        "jump_off", "land",
        "jump_off", "land",
        "jump_off", "land",
        "jump_off", "land",
        "recover", "dwell",
    ]


def test_geometry_count_does_not_invent_repeated_jumps():
    assert _abstract_objective_program(
        "Jump over four obstacles, land, recover upright, and hold still"
    ).count("jump_off") == 1


def test_repeated_hop_probe_preserves_four_flights_and_full_quiet_tail():
    goal = (
        "Perform four distinct one-leg hops over four low rails, landing "
        "after each, then recover upright and hold with horizontal, angular, "
        "joint, upright, and default-pose quiet for 100 uninterrupted frames."
    )
    program = _abstract_objective_program(goal)
    probe = _abstract_objective_probe(program, behavior_goal=goal)
    assert probe is not None
    air = (
        (probe["left_foot_contact"][:, 0] == 0.0)
        & (probe["right_foot_contact"][:, 0] == 0.0)
    )
    flight_starts = np.flatnonzero(air & ~np.r_[False, air[:-1]])
    assert flight_starts.size == 4

    tail = slice(-100, None)
    assert np.all(probe["left_foot_contact"][tail, 0] == 1.0)
    assert np.all(probe["right_foot_contact"][tail, 0] == 1.0)
    root_xy = probe["root_link_pos_w"][tail, 0, :2]
    assert np.max(np.linalg.norm(np.diff(root_xy, axis=0), axis=1)) == 0.0
    assert np.max(np.abs(probe["root_link_ang_vel_b"][tail, 0])) == 0.0
    assert np.max(np.abs(probe["joint_vel"][tail, 0])) == 0.0
    assert np.max(probe["default_pose_rms"][tail, 0]) == 0.0
    assert np.min(-probe["projected_gravity_b"][tail, 0, 2]) == 1.0


def test_competent_route_fixture_visits_regions_in_order_and_holds_finish():
    draft = author_environment(
        "Build a slalom around four boxes with ordered waypoints and a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    route_specs = sorted(
        (
            spec for spec in catalog.channels
            if spec.producer == "region_relative"
            and "sequence_index" in spec.source
        ),
        key=lambda spec: spec.source["sequence_index"],
    )
    assert len(route_specs) >= 2
    arrays = catalog_fixture_arrays(
        catalog, time_steps=180, num_envs=2, case="competent")
    first_hits = []
    for spec in route_specs:
        distance = np.linalg.norm(arrays[spec.name], axis=-1)
        first_hits.append(int(np.argmax(distance[:, 0] < 0.75)))
    assert first_hits == sorted(first_hits)
    assert len(set(first_hits)) == len(first_hits)
    finish_distance = np.linalg.norm(arrays[route_specs[-1].name], axis=-1)
    assert np.all(finish_distance[-100:] < 0.75)


def test_compound_event_catalog_exposes_phase_local_shaping_with_provenance():
    draft = author_environment(
        "Slalom around four boxes, then jump at the finish and hold still "
        "for 2 seconds.",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    by_name = catalog.by_name()
    expected = {
        "event__route_jump_hold__phase": "event_phase_state",
        "event__route_jump_hold__phase_height_delta": (
            "event_phase_height_delta"),
        "event__route_jump_hold__base_vertical_velocity": (
            "event_phase_vertical_velocity"),
    }
    for name, producer in expected.items():
        spec = by_name[name]
        assert spec.access == "shared_shaping"
        assert spec.producer == producer
        assert spec.source["event_sequence"] == "route_jump_hold"
        assert spec.source["phase_ids"] == ["route", "jump", "hold"]
        assert spec.source["event_program"]["phases"][1]["until"][
            "min_height_delta_m"
        ] == 0.18
    violation = by_name["event__route_jump_hold__violation"]
    assert violation.access == "metric_only"
    assert violation.producer == "event_sequence_violation"
    assert violation.source["event_program"]["phases"][2][
        "minimum_hold_s"
    ] == 2.0
    adversarial = catalog_fixture_arrays(
        catalog, time_steps=180, num_envs=1, case="event_violation")
    assert adversarial[violation.name][-1, 0]
    assert violation.name not in catalog.names(reward=True)
    supports = [
        by_name[f"event__route_jump_hold__support__{index}"]
        for index in range(2)
    ]
    assert [spec.source["support_selector"] for spec in supports] == [
        ["robot:left_foot", "world:terrain"],
        ["robot:right_foot", "world:terrain"],
    ]
    assert all(spec.access == "metric_only" for spec in supports)
    assert all(spec.name not in catalog.names(reward=True) for spec in supports)


def test_event_violation_fixture_must_fail_completion_gate(
    tmp_path: Path,
) -> None:
    draft = author_environment(
        "Slalom around four boxes, then jump at the finish and hold still "
        "for 2 seconds.",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    source = '''
import numpy as np

ABSTRACT_OBJECTIVE = {"phases": [
    "move_forward", "jump", "land", "dwell",
]}

def compute_spec(arrays, behavior, meta):
    valid = arrays["first_episode_valid_mask"]
    success = arrays["goal__complete_slalom_and_stop__success"]
    violation = arrays["event__route_jump_hold__violation"]
    completed = bool(np.any(success & valid))
    invalid = bool(np.any(violation & valid))
    score = 0.6 if (completed and invalid) else (0.9 if completed else 0.0)
    return {"spec_score": score, "completion_gate": score}
'''
    path = _write_metric(tmp_path, source, "soft_violation.py")
    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "navigate through four boxes, jump exactly once after the route, "
            "land bilaterally, and hold still for 2 seconds"
        ),
        channel_catalog=catalog,
    )
    assert result["archetype_scores"]["catalog_competent"] == pytest.approx(
        0.9
    )
    assert result["archetype_scores"][
        "catalog_event_violation"
    ] == pytest.approx(0.6)
    assert not result["gates"]["catalog_event_violation_fail_zero"]
    assert any(
        "invalid event attempts must fail" in reason
        for reason in result["reasons"]
    )


def _catalog() -> ChannelCatalog:
    world = {
        "shared": {
            "objects": {"ball": {}},
            "zones": {"goal_mouth": {}},
        }
    }
    task = {
        "shared": {
            "goal": {
                "id": "score",
                "type": "object_to_region",
                "subject": "ball",
                "region": "goal_mouth",
            },
            "observations": {"region_relative": ["goal_mouth"]},
            "contacts": {
                "desired": [
                    ["robot:end_effector", "object:ball"],
                    ["robot:left_foot", "world:terrain"],
                ],
                "forbidden": [["robot:torso", "object:ball"]],
                "terminate_on": [],
            },
        }
    }
    return compile_channel_catalog(world, task)


GOOD_OBJECT_METRIC = '''
import numpy as np

def compute_spec(arrays, behavior, meta):
    success = arrays["goal__score__success"].astype(float).mean()
    distance = arrays["object__ball__to_region__goal_mouth__distance"]
    progress = float(np.clip(1.0 - distance.mean(), 0.0, 1.0))
    return {"spec_score": float(success * progress),
            "completion_gate": float(success), "progress": progress}
'''


FOOT_REGION_RETENTION_METRIC = '''
import numpy as np

def compute_spec(arrays, behavior, meta):
    valid = arrays.get("first_episode_valid_mask")
    root = arrays.get("root_link_pos_w")
    finish = arrays.get("region__goal_mouth__relative")
    left = arrays.get("left_foot_pos_w")
    right = arrays.get("right_foot_pos_w")
    left_contact = arrays.get("left_foot_contact")
    right_contact = arrays.get("right_foot_contact")
    if any(x is None for x in (
        valid, root, finish, left, right, left_contact, right_contact,
    )):
        return {"spec_score": 0.0}
    passed = []
    landing_inside = []
    retained = []
    for env in range(root.shape[1]):
        keep = np.flatnonzero(valid[:, env])
        if keep.size < 3:
            passed.append(False); landing_inside.append(False); retained.append(False)
            continue
        lc = left_contact[keep, env] > 0.5
        rc = right_contact[keep, env] > 0.5
        both_air = (~lc) & (~rc)
        land = -1
        seen_air = False
        for index in range(keep.size):
            if both_air[index]:
                seen_air = True
            elif seen_air and lc[index] and rc[index]:
                land = index
                break
        foot_center = []
        for foot in (left, right):
            relative = (
                (foot[keep, env] - root[keep, env])[:, :2]
                - finish[keep, env, :2]
            )
            foot_center.append(np.linalg.norm(relative, axis=-1) <= 0.35)
        both_inside = foot_center[0] & foot_center[1]
        at_landing = bool(land >= 0 and both_inside[land])
        never_exited = bool(land >= 0 and np.all(both_inside[land:]))
        landing_inside.append(at_landing)
        retained.append(never_exited)
        passed.append(at_landing and never_exited)
    score = float(np.mean(passed))
    return {
        "spec_score": score,
        "landing_feet_inside_frac": float(np.mean(landing_inside)),
        "post_landing_retained_frac": float(np.mean(retained)),
    }
'''


DISTANCE_ONLY_METRIC = '''
import numpy as np

def compute_spec(arrays, behavior, meta):
    distance = arrays["object__ball__to_region__goal_mouth__distance"]
    return {"spec_score": float(np.clip(1.0 - distance.mean(), 0.0, 1.0))}
'''


DYNAMIC_KEY_METRIC = '''
def compute_spec(arrays, behavior, meta):
    key = "goal" + "__score__success"
    return {"spec_score": float(arrays[key].mean())}
'''


PROMPT_NATIVE_PARKOUR_METRIC = '''
import numpy as np

ABSTRACT_OBJECTIVE = {
    "phases": [
        "climb", "dwell", "climb", "dwell",
        "climb", "dwell", "climb", "dwell",
    ]
}

def compute_spec(arrays, behavior, meta):
    root = arrays["root_link_pos_w"]
    waypoint = arrays["goal__complete_course__waypoint_index"]
    valid = arrays.get("first_episode_valid_mask")
    left_contact = arrays.get("left_foot_contact")
    right_contact = arrays.get("right_foot_contact")
    success = arrays["goal__complete_course__success"]
    if valid is None or left_contact is None or right_contact is None:
        return {"spec_score": 0.0}
    passed = []
    for env in range(root.shape[1]):
        keep = np.flatnonzero(valid[:, env])
        if keep.size < 25:
            passed.append(False)
            continue
        lane_root = root[keep, env]
        lane_waypoint = waypoint[keep, env]
        flight = ((left_contact[keep, env] < 0.5)
                  & (right_contact[keep, env] < 0.5))
        starts = flight[1:] & (~flight[:-1])
        final_motion = np.linalg.norm(np.diff(lane_root[-25:], axis=0), axis=-1)
        passed.append(bool(
            lane_root[:, 2].max() - lane_root[:, 2].min() > 0.9
            and np.sum(np.diff(lane_waypoint) > 0) >= 4
            and starts.sum() >= 3
            and final_motion.mean() < 0.01
            and np.mean(success[keep[-3:], env]) >= 1.0
        ))
    score = float(np.mean(passed))
    return {
        "spec_score": score,
        "completion_gate": score,
    }
'''


def _write_metric(tmp_path: Path, source: str, name: str = "metric.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_catalog_is_canonical_strict_and_access_partitioned():
    catalog = _catalog()
    assert not validate_channel_catalog(catalog)
    assert catalog.to_dict()["catalog_hash"] == catalog.catalog_hash
    by_name = catalog.by_name()
    assert by_name["goal__score__success"].access == "metric_only"
    assert by_name[
        "object__ball__to_region__goal_mouth__distance"
    ].access == "shared_shaping"
    assert "object__ball__quat_w" in by_name
    assert "object__ball__ang_vel_w" in by_name
    assert "goal__score__success" not in catalog.names(reward=True)
    assert (
        "object__ball__to_region__goal_mouth__distance"
        in catalog.names(reward=True)
    )
    assert by_name["contact__desired__0"].source["index"] == 0
    assert by_name["contact__desired__1"].source == {
        "group": "desired",
        "index": 1,
        "selectors": ["robot:left_foot", "world:terrain"],
    }

    tampered = catalog.to_dict()
    tampered["channels"][0]["producer"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        ChannelCatalog.from_dict(tampered)

    unknown = catalog.to_dict()
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        ChannelCatalog.from_dict(unknown)

    missing_hash = catalog.to_dict()
    missing_hash.pop("catalog_hash")
    with pytest.raises(ValueError, match="missing"):
        ChannelCatalog.from_dict(missing_hash)


def test_trajectory_contract_checks_dtype_shape_budget_and_missing():
    catalog = _catalog()
    arrays = catalog_fixture_arrays(
        catalog, time_steps=12, num_envs=3, case="competent")
    assert not validate_trajectory_channels(
        arrays, catalog, catalog_hash=catalog.catalog_hash)

    bad_dtype = dict(arrays)
    bad_dtype["goal__score__success"] = np.ones((12, 3), dtype=np.int32)
    assert any("dtype" in e for e in validate_trajectory_channels(
        bad_dtype, catalog, catalog_hash=catalog.catalog_hash))

    missing = dict(arrays)
    missing.pop("object__ball__pos_w")
    assert any("missing" in e for e in validate_trajectory_channels(
        missing, catalog, catalog_hash=catalog.catalog_hash))

    inconsistent = dict(arrays)
    inconsistent["goal__score__success"] = np.ones((12, 2), dtype=bool)
    assert any("symbolic dimension N" in e for e in validate_trajectory_channels(
        inconsistent, catalog, catalog_hash=catalog.catalog_hash))

    too_large = catalog_fixture_arrays(
        catalog, time_steps=500, num_envs=500, case="competent")
    assert any("byte budget" in e for e in validate_trajectory_channels(
        too_large, catalog, catalog_hash=catalog.catalog_hash))


def test_validation_uses_exact_catalog_and_degenerate_object_fixtures(tmp_path):
    catalog = _catalog()
    good_path = _write_metric(tmp_path, GOOD_OBJECT_METRIC, "good.py")
    good = validate_generated_metric(
        GOOD_OBJECT_METRIC, good_path,
        behavior_goal="move the ball into the goal and hold it there",
        channel_catalog=catalog)
    assert good["ok"], good["reasons"]
    assert good["gates"]["catalog_completion_channel"]
    assert good["gates"]["catalog_degenerate_fixtures"]
    assert good["channel_catalog_hash"] == catalog.catalog_hash

    distance_path = _write_metric(tmp_path, DISTANCE_ONLY_METRIC, "distance.py")
    distance = validate_generated_metric(
        DISTANCE_ONLY_METRIC, distance_path,
        behavior_goal="move the ball into the goal and hold it there",
        channel_catalog=catalog)
    assert not distance["ok"]
    assert not distance["gates"]["catalog_completion_channel"]
    assert not distance["gates"]["catalog_degenerate_fixtures"]
    assert any("edge_camping" in reason for reason in distance["reasons"])

    dynamic_path = _write_metric(tmp_path, DYNAMIC_KEY_METRIC, "dynamic.py")
    dynamic = validate_generated_metric(
        DYNAMIC_KEY_METRIC, dynamic_path,
        behavior_goal="move the ball into the goal and hold it there",
        channel_catalog=catalog)
    assert not dynamic["gates"]["catalog_literal_array_access"]


def test_prompt_native_traversal_composes_physics_and_world_without_reference(
    tmp_path: Path,
) -> None:
    """A legitimate compound metric needs physical traversal AND authored
    completion in the same competent fixture; neither half alone may rescue it.

    This is the no-stored-trajectory path: the physical exemplar is retargeted
    from ``ABSTRACT_OBJECTIVE`` while the World catalog supplies the frozen
    waypoint state.
    """
    draft = author_environment(
        "Build four progressively taller boxes; jump onto each and pause",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    waypoint_channel = catalog.by_name()[
        "goal__complete_course__waypoint_index"]
    assert waypoint_channel.source["waypoint_count"] == 4
    competent = catalog_fixture_arrays(
        catalog, time_steps=120, num_envs=2, case="competent")
    waypoint_trace = competent["goal__complete_course__waypoint_index"]
    assert np.array_equal(np.unique(waypoint_trace), np.arange(5))
    assert np.all(np.diff(waypoint_trace, axis=0) >= 0)

    probe = _abstract_objective_probe(
        ["climb", "dwell"] * 4,
        behavior_goal="jump onto four boxes and pause on each",
    )
    assert probe is not None
    z = probe["root_link_pos_w"][:, 0, 2]
    supported = (
        (probe["left_foot_contact"][:, 0] > 0.5)
        & (probe["right_foot_contact"][:, 0] > 0.5)
        & (np.abs(np.gradient(z)) < 1e-9)
    )
    padded = np.pad(supported.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    stable_holds = [
        (start, end) for start, end in zip(starts, ends)
        if end - start >= 13 and float(np.mean(z[start:end])) > 0.6
    ]
    assert len(stable_holds) >= 4
    assert supported[-30:].all()
    path = _write_metric(tmp_path, PROMPT_NATIVE_PARKOUR_METRIC, "parkour.py")

    result = validate_generated_metric(
        PROMPT_NATIVE_PARKOUR_METRIC,
        path,
        behavior_goal=(
            "jump onto four progressively taller boxes, pause stably on every "
            "box, then continue through the course"
        ),
        channel_catalog=catalog,
    )

    assert result["ok"], result["reasons"]
    assert result["abstract_objective_program"] == [
        "climb", "dwell", "climb", "dwell",
        "climb", "dwell", "climb", "dwell",
    ]
    scores = result["archetype_scores"]
    assert scores["catalog_competent"] > 0.9
    assert scores["catalog_far_idle"] == pytest.approx(0.0)
    assert scores["prompt_competent"] > 0.9


def test_abstract_jump_then_land_is_one_continuous_flight_bout() -> None:
    """An explicit landing continues the preceding flight until touchdown.

    A fabricated grounded frame at the phase boundary turns one jump into two
    contact bouts and makes every honest exactly-once metric reject the validator's
    own competent example.
    """
    goal = (
        "navigate, jump exactly once, land, recover, and hold 100 "
        "uninterrupted post-completion frames"
    )
    probe = _abstract_objective_probe(
        ["move_forward", "jump", "land", "recover", "dwell"],
        behavior_goal=goal,
    )
    assert probe is not None

    flight = (
        (probe["left_foot_contact"][:, 0] < 0.5)
        & (probe["right_foot_contact"][:, 0] < 0.5)
    )
    changes = np.diff(np.pad(flight.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    assert len(starts) == len(ends) == 1
    assert ends[0] - starts[0] >= 3
    assert not flight[-35:].any()
    assert np.ptp(probe["root_link_pos_w"][flight, 0, 0]) == pytest.approx(0.0)

    root_xy = probe["root_link_pos_w"][:, 0, :2]
    horizontal_speed = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) / 0.02
    assert np.all(horizontal_speed[-100:] < 0.12)
    assert np.all(probe["left_foot_contact"][-100:, 0] > 0.5)
    assert np.all(probe["right_foot_contact"][-100:, 0] > 0.5)
    assert np.max(np.abs(probe["joint_vel"][-100:, 0])) < 1e-9


def test_competent_route_truth_precedes_post_route_jump() -> None:
    """Catalog progress and physical skill phases must share one timeline."""
    program = ["move_forward", "jump", "land", "recover", "dwell"]
    goal = (
        "navigate, jump exactly once, land, recover, and hold 100 "
        "uninterrupted post-completion frames"
    )
    probe = _abstract_objective_probe(
        program,
        behavior_goal=goal,
    )
    assert probe is not None
    probe_steps = probe["root_link_pos_w"].shape[0]
    route_window = _abstract_route_window(
        program, probe_steps=probe_steps, final_hold_steps=100,
    )
    assert route_window is not None
    route_start, route_completion = route_window

    draft = author_environment(
        "Build a slalom around four boxes with ordered waypoints and a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    arrays = catalog_fixture_arrays(
        catalog,
        time_steps=probe_steps,
        num_envs=2,
        case="competent",
        competent_route_start_step=route_start,
        competent_route_completion_step=route_completion,
    )
    waypoint_spec = next(
        spec for spec in catalog.channels if spec.producer == "waypoint_state"
    )
    waypoint_distance_spec = next(
        spec for spec in catalog.channels if spec.producer == "waypoint_distance"
    )
    waypoint = arrays[waypoint_spec.name][:, 0]
    waypoint_distance = arrays[waypoint_distance_spec.name][:, 0]
    terminal_index = int(waypoint_spec.source["waypoint_count"])
    terminal_frame = int(np.flatnonzero(waypoint >= terminal_index)[0])

    flight = (
        (probe["left_foot_contact"][:, 0] < 0.5)
        & (probe["right_foot_contact"][:, 0] < 0.5)
    )
    flight_start = int(np.flatnonzero(flight)[0])

    assert np.all(waypoint[:route_start] == 0)
    assert np.all(waypoint_distance[:route_start] > 0.9)
    assert terminal_frame == route_completion
    assert terminal_frame < flight_start
    assert np.all(waypoint_distance[route_completion:] == 0.0)


def test_abstract_route_window_abstains_after_extension_boundary() -> None:
    """Flat phase programs must not guess route ownership when ordering is mixed."""
    staged = _abstract_route_window(
        ["climb", "dwell", "climb", "dwell", "jump_off", "land"],
        probe_steps=180,
    )
    assert staged is not None

    ambiguous = _abstract_route_window(
        ["jump_off", "land", "move_forward", "recover"],
        probe_steps=180,
    )
    assert ambiguous is None


def test_prompt_native_planar_route_composes_motion_with_world_state(
    tmp_path: Path,
) -> None:
    """Planar authored routes need the same composed competent fixture as parkour."""
    draft = author_environment(
        "Build a slalom around four boxes with ordered waypoints and a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    success_name = next(
        name for name in catalog.names()
        if name.startswith("goal__") and name.endswith("__success")
    )
    source = f'''\
import numpy as np

ABSTRACT_OBJECTIVE = {{"phases": ["move_forward", "recover", "dwell"]}}

def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    success = arrays.get("{success_name}")
    valid = arrays.get("first_episode_valid_mask")
    if root is None or success is None or valid is None:
        return {{"spec_score": 0.0}}
    dt = float(behavior.get("step_dt", 0.02) or 0.02)
    passed = []
    for env in range(root.shape[1]):
        keep = np.flatnonzero(valid[:, env])
        if keep.size < round(2.0 / dt) + 1:
            passed.append(False)
            continue
        lane_root = root[keep, env]
        speed = np.linalg.norm(np.diff(lane_root, axis=0)[..., :2], axis=-1) / dt
        quiet = speed < 0.12
        run = 0
        best = 0
        for is_quiet in quiet:
            run = run + 1 if is_quiet else 0
            best = max(best, run)
        passed.append(bool(
            lane_root[-1, 0] - lane_root[0, 0] > 0.5
            and best >= round(2.0 / dt)
            and np.mean(success[keep[-3:], env]) >= 1.0
        ))
    return {{"spec_score": float(np.mean(passed))}}
'''
    path = _write_metric(tmp_path, source, "planar_route.py")

    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Run a slalom around four boxes through ordered waypoints, enter "
            "the finish zone, then stop upright and hold for 2 seconds"
        ),
        channel_catalog=catalog,
    )

    assert result["ok"], result["reasons"]
    assert result["abstract_objective_program"] == [
        "move_forward", "recover", "dwell",
    ]
    scores = result["archetype_scores"]
    assert scores["catalog_competent"] > 0.9
    assert scores["catalog_far_idle"] == pytest.approx(0.0)
    assert scores["prompt_competent"] > 0.9
    assert scores["catalog_interrupted_hold"] == pytest.approx(0.0)
    assert result["gates"]["continuous_hold_interruption"]
    assert result["gates"]["temporal_validity_channel"]
    assert result["gates"]["temporal_invalid_support"]
    assert scores["temporal_invalid_support"] == pytest.approx(0.0)


def test_temporal_metric_rejects_reset_flight_outside_official_support(
    tmp_path: Path,
) -> None:
    """Reset both-air frames plus an invalid sequence cannot establish flight.

    The validator's adversarial rollout places reset-like bilateral flight and
    the otherwise competent route in the invalid prefix, followed by only a
    short valid post-route state. Reading the mask as a no-op is rejected.
    """
    draft = author_environment(
        "Build a slalom with ordered waypoints and a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    success_name = next(
        name for name in catalog.names()
        if name.startswith("goal__") and name.endswith("__success")
    )
    source = f'''\
import numpy as np

ABSTRACT_OBJECTIVE = {{"phases": ["move_forward", "jump", "land", "dwell"]}}

def compute_spec(arrays, behavior, meta):
    root = arrays["root_link_pos_w"]
    left = arrays["left_foot_contact"]
    right = arrays["right_foot_contact"]
    success = arrays["{success_name}"]
    valid = arrays.get("first_episode_valid_mask")
    if valid is None:
        return {{"spec_score": 0.0}}
    flight = (left < 0.5) & (right < 0.5)
    starts = flight[1:] & (~flight[:-1])
    speed = np.linalg.norm(np.diff(root, axis=0)[..., :2], axis=-1) / 0.02
    route = root[-1, :, 0] - root[0, :, 0] > 0.5
    held = np.all(speed[-100:] < 0.12, axis=0)
    completed = success[-1] > 0.5
    passed = route & (starts.sum(axis=0) >= 1) & held & completed
    return {{"spec_score": float(np.mean(passed))}}
'''
    path = _write_metric(tmp_path, source, "ignores_validity.py")
    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Run through the ordered route, then jump and land, then hold "
            "still for 100 uninterrupted frames"
        ),
        channel_catalog=catalog,
    )

    assert result["gates"]["temporal_validity_channel"]
    assert not result["gates"]["temporal_invalid_support"]
    assert result["archetype_scores"]["temporal_invalid_support"] > 0.9
    assert any(
        "reset-like bilateral-flight prefix" in reason
        for reason in result["reasons"]
    )


def test_terminal_quiet_rejects_gravity_derivative_and_role_subset(
    tmp_path: Path,
) -> None:
    draft = author_environment(
        "Walk to a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    success_name = next(
        name for name in catalog.names()
        if name.startswith("goal__") and name.endswith("__success")
    )
    source = f'''\
import numpy as np

REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
ABSTRACT_OBJECTIVE = {{"phases": ["dwell"]}}

def compute_spec(arrays, behavior, meta):
    valid = arrays.get("first_episode_valid_mask")
    joint_vel = arrays["joint_vel"]
    gravity = arrays["projected_gravity_b"]
    recorded_angular = arrays.get("root_link_ang_vel_b")
    success = arrays["{success_name}"]
    if valid is None or recorded_angular is None:
        return {{"spec_score": 0.0}}
    roles = (meta or {{}}).get("joint_roles", {{}})
    knees = [roles[name] for name in REQUIRED_JOINT_ROLES if name in roles]
    passed = []
    for env in range(joint_vel.shape[1]):
        keep = np.flatnonzero(valid[:, env])
        if keep.size < 100 or not knees:
            passed.append(False)
            continue
        lane_joint = joint_vel[keep, env]
        derived_angular = np.linalg.norm(
            np.diff(gravity[keep, env], axis=0), axis=-1) / 0.02
        passed.append(bool(
            np.all(np.abs(lane_joint[-100:, knees]) < 0.12)
            and np.all(derived_angular[-99:] < 0.12)
            and success[keep[-1], env] > 0.5
        ))
    return {{"spec_score": float(np.mean(passed))}}
'''
    path = _write_metric(tmp_path, source, "narrow_quiet.py")
    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Hold 100 uninterrupted frames of base angular velocity and "
            "whole-body joint velocity quiet"
        ),
        channel_catalog=catalog,
    )

    assert result["gates"]["recorded_base_angular_channel"]
    assert result["gates"]["whole_body_joint_velocity_channel"]
    assert not result["gates"]["recorded_base_angular_violation"]
    assert not result["gates"]["whole_body_joint_velocity_violation"]
    assert result["archetype_scores"]["recorded_base_angular_violation"] > 0.9
    assert result["archetype_scores"]["whole_body_joint_velocity_violation"] > 0.9


def test_terminal_quiet_accepts_recorded_base_angular_and_all_joints(
    tmp_path: Path,
) -> None:
    draft = author_environment(
        "Walk to a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    success_name = next(
        name for name in catalog.names()
        if name.startswith("goal__") and name.endswith("__success")
    )
    source = f'''\
import numpy as np

ABSTRACT_OBJECTIVE = {{"phases": ["dwell"]}}

def compute_spec(arrays, behavior, meta):
    valid = arrays.get("first_episode_valid_mask")
    joint_vel = arrays["joint_vel"]
    angular = arrays.get("root_link_ang_vel_b")
    success = arrays["{success_name}"]
    if valid is None or angular is None:
        return {{"spec_score": 0.0}}
    passed = []
    for env in range(joint_vel.shape[1]):
        keep = np.flatnonzero(valid[:, env])
        if keep.size < 100:
            passed.append(False)
            continue
        passed.append(bool(
            np.all(np.abs(joint_vel[keep[-100:], env, :]) < 0.12)
            and np.all(np.linalg.norm(angular[keep[-100:], env], axis=-1) < 0.12)
            and success[keep[-1], env] > 0.5
        ))
    return {{"spec_score": float(np.mean(passed))}}
'''
    path = _write_metric(tmp_path, source, "whole_body_quiet.py")
    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Hold 100 uninterrupted frames of base angular velocity and "
            "whole-body joint velocity quiet"
        ),
        channel_catalog=catalog,
    )

    assert result["gates"]["temporal_invalid_support"]
    assert result["gates"]["recorded_base_angular_violation"]
    assert result["gates"]["whole_body_joint_velocity_violation"]
    assert result["archetype_scores"]["temporal_invalid_support"] == pytest.approx(0.0)
    assert result["archetype_scores"]["recorded_base_angular_violation"] == pytest.approx(0.0)
    assert result["archetype_scores"]["whole_body_joint_velocity_violation"] == pytest.approx(0.0)


def test_continuous_hold_rejects_quiet_sample_fraction_proxy(
    tmp_path: Path,
) -> None:
    draft = author_environment(
        "Build a slalom around four boxes with ordered waypoints and a finish zone",
        robot_capability_id="unitree_g1:base",
    )
    catalog = compile_channel_catalog(draft.world_spec, draft.task_spec)
    success_name = next(
        name for name in catalog.names()
        if name.startswith("goal__") and name.endswith("__success")
    )
    source = f'''\
import numpy as np

ABSTRACT_OBJECTIVE = {{"phases": ["move_forward", "recover", "dwell"]}}

def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    success = arrays.get("{success_name}")
    if root is None or success is None:
        return {{"spec_score": 0.0}}
    dt = float(behavior.get("step_dt", 0.02) or 0.02)
    speed = np.linalg.norm(
        np.diff(root, axis=0)[..., :2], axis=-1) / dt
    padded = np.pad(speed, ((2, 2), (0, 0)), mode="edge")
    smoothed_speed = sum(padded[i:i + speed.shape[0]] for i in range(5)) / 5.0
    quiet_fraction = np.mean(smoothed_speed[-90:] < 0.12, axis=0)
    dx = root[-1, :, 0] - root[0, :, 0]
    completed = np.mean(success[-3:] > 0.5, axis=0) >= 1.0
    physical = (dx > 0.5) & (quiet_fraction > 0.99)
    return {{"spec_score": float(np.mean(physical & completed))}}
'''
    path = _write_metric(tmp_path, source, "fraction_hold.py")

    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Run through the ordered slalom, enter the finish, then remain "
            "upright and still for 100 uninterrupted frames"
        ),
        channel_catalog=catalog,
    )

    assert not result["ok"]
    assert not result["gates"]["continuous_hold_interruption"]
    assert result["archetype_scores"]["catalog_interrupted_hold"] > 0.9
    hold_reason = next(
        reason for reason in result["reasons"]
        if "[continuous-hold]" in reason
    )
    assert "at least 100 consecutive raw-frame samples" in hold_reason
    assert "smoothing, capping" in hold_reason


def test_catalog_literal_access_allows_guards_but_rejects_aliases() -> None:
    from sculptor.eval.metric_validate import _catalog_array_access_violations

    guarded = '''\
def compute_spec(arrays, behavior, meta):
    if arrays is None:
        return {"spec_score": 0.0}
    if not isinstance(arrays, dict):
        return {"spec_score": 0.0}
    root = arrays.get("root_link_pos_w")
    return {"spec_score": 0.0 if root is None else 1.0}
'''
    aliased = '''\
def compute_spec(arrays, behavior, meta):
    local_arrays = arrays
    root = local_arrays.get("root_link_pos_w")
    return {"spec_score": 0.0 if root is None else 1.0}
'''

    assert _catalog_array_access_violations(guarded) == []
    violations = _catalog_array_access_violations(aliased)
    assert len(violations) == 1
    assert "line 2" in violations[0]


def test_runtime_requires_matching_catalog_hash_and_loads_only_allowlist(tmp_path):
    catalog = _catalog()
    metric_path = _write_metric(tmp_path, GOOD_OBJECT_METRIC)
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    arrays = catalog_fixture_arrays(
        catalog, time_steps=12, num_envs=3, case="competent")
    np.savez(
        rollout / "trajectory.npz",
        **arrays,
        channel_catalog_hash=np.asarray(catalog.catalog_hash),
        undeclared_secret=np.ones((12, 3), dtype=np.float32),
    )
    result = compute_generated_metric(
        metric_path, rollout, channel_catalog=catalog)
    assert result["spec_score"] == pytest.approx(1.0)

    np.savez(
        rollout / "trajectory.npz",
        **arrays,
        channel_catalog_hash=np.asarray("0" * 64),
    )
    mismatch = compute_generated_metric(
        metric_path, rollout, channel_catalog=catalog)
    assert mismatch["spec_score"] == 0.0
    assert "hash mismatch" in mismatch["error"]


@pytest.mark.parametrize(
    ("right_foot_overrides", "expected_landing", "expected_retained"),
    [
        ({}, 1.0, 1.0),
        ({4: 0.36}, 0.0, 0.0),
        ({7: 0.36}, 1.0, 0.0),
    ],
)
def test_world_foot_channels_prove_landing_and_veto_later_region_exit(
    tmp_path: Path,
    right_foot_overrides: dict[int, float],
    expected_landing: float,
    expected_retained: float,
) -> None:
    catalog = _catalog()
    metric_path = _write_metric(
        tmp_path, FOOT_REGION_RETENTION_METRIC, "foot_retention.py")
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    time_steps = 9
    arrays = catalog_fixture_arrays(
        catalog, time_steps=time_steps, num_envs=1, case="competent")

    root = np.zeros((time_steps, 1, 3), dtype=np.float32)
    root[..., 0] = 10.0  # replicated-environment world offset
    root[..., 2] = 0.8
    finish_relative = np.zeros((time_steps, 1, 3), dtype=np.float32)
    left = root.copy()
    right = root.copy()
    left[..., 0] += 0.10
    right[..., 0] -= 0.10
    for frame, offset in right_foot_overrides.items():
        right[frame, 0, 0] = root[frame, 0, 0] + offset
    left_contact = np.ones((time_steps, 1), dtype=np.float32)
    right_contact = np.ones((time_steps, 1), dtype=np.float32)
    left_contact[2:4] = 0.0
    right_contact[2:4] = 0.0

    arrays.update({
        "first_episode_valid_mask": np.ones(
            (time_steps, 1), dtype=bool),
        "root_link_pos_w": root,
        "region__goal_mouth__relative": finish_relative,
        "left_foot_pos_w": left,
        "right_foot_pos_w": right,
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
    })
    np.savez(
        rollout / "trajectory.npz",
        **arrays,
        channel_catalog_hash=np.asarray(catalog.catalog_hash),
    )

    result = compute_generated_metric(
        metric_path,
        rollout,
        behavior={"step_dt": 0.02},
        channel_catalog=catalog,
    )

    assert result["landing_feet_inside_frac"] == expected_landing
    assert result["post_landing_retained_frac"] == expected_retained
    assert result["spec_score"] == min(expected_landing, expected_retained)


@dataclass
class _Edit:
    target_term: str
    suggested_value: str = ""
    operation: str = "modify"


def test_best_of_n_observables_and_partition_use_catalog(tmp_path):
    catalog = _catalog()
    metric_path = _write_metric(tmp_path, GOOD_OBJECT_METRIC)
    observables = _generated_metric_observables(
        metric_path, channel_catalog=catalog)
    assert observables == {
        "goal__score__success",
        "object__ball__to_region__goal_mouth__distance",
    }
    discrimination = discrimination_of_metric(
        metric_path, channel_catalog=catalog)
    assert discrimination["catalog_axis"] is not None
    assert discrimination["catalog_axis"]["separation"] > 0.5

    screen = screen_edits(
        [_Edit("goal__score__success")],
        metric_observables=observables,
        channel_catalog=catalog)
    assert screen.channel_catalog_hash == catalog.catalog_hash
    assert screen.metric_only == ["goal__score__success"]
    assert screen.flagged_edits
    prompt = build_partition_prompt_block(
        observables, screen, channel_catalog=catalog)
    assert "reserved for the held-out judge" in prompt
    assert "goal__score__success" in prompt
