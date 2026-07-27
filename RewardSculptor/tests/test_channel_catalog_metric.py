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
    height_gain = float((root[..., 2].max(axis=0) - root[..., 2].min(axis=0)).mean())
    waypoint = arrays["goal__complete_course__waypoint_index"]
    waypoint_max = float(waypoint.max())
    ordered_steps = float(np.mean((np.diff(waypoint, axis=0) > 0).sum(axis=0) >= 4))
    left_contact = arrays.get("left_foot_contact")
    right_contact = arrays.get("right_foot_contact")
    if left_contact is None or right_contact is None:
        airborne_hops = 0.0
    else:
        flight = (left_contact < 0.5) & (right_contact < 0.5)
        starts = flight[1:] & (~flight[:-1])
        airborne_hops = float(np.mean(starts.sum(axis=0) >= 3))
    completed = float(arrays["goal__complete_course__success"].mean())
    physical = float(height_gain > 0.9)
    final_motion = np.linalg.norm(np.diff(root[-25:], axis=0), axis=-1)
    final_pause = float(final_motion.mean() < 0.01)
    progress = float(np.clip(waypoint_max / 4.0, 0.0, 1.0))
    return {
        "spec_score": float(
            physical * airborne_hops * final_pause
            * ordered_steps * progress * completed),
        "physical_traversal": physical,
        "airborne_hops": airborne_hops,
        "final_stable_pause": final_pause,
        "ordered_waypoint_steps": ordered_steps,
        "waypoint_progress": progress,
        "completion_gate": completed,
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
    if root is None or success is None:
        return {{"spec_score": 0.0}}
    dx = root[-1, :, 0] - root[0, :, 0]
    dt = float(behavior.get("step_dt", 0.02) or 0.02)
    speed = np.linalg.norm(
        np.diff(root, axis=0)[..., :2], axis=-1) / dt
    quiet = speed < 0.12
    run = np.zeros(root.shape[1])
    best = np.zeros(root.shape[1])
    for t in range(quiet.shape[0]):
        run = np.where(quiet[t], run + 1.0, 0.0)
        best = np.maximum(best, run)
    physical = (dx > 0.5) & (best >= round(2.0 / dt))
    completed = np.mean(success[-3:] > 0.5, axis=0) >= 1.0
    return {{"spec_score": float(np.mean(physical & completed))}}
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
    quiet_fraction = np.mean(speed[-100:] < 0.12, axis=0)
    dx = root[-1, :, 0] - root[0, :, 0]
    completed = np.mean(success[-3:] > 0.5, axis=0) >= 1.0
    physical = (dx > 0.5) & (quiet_fraction > 0.9)
    return {{"spec_score": float(np.mean(physical & completed))}}
'''
    path = _write_metric(tmp_path, source, "fraction_hold.py")

    result = validate_generated_metric(
        source,
        path,
        behavior_goal=(
            "Run through the ordered slalom, enter the finish, then remain "
            "upright and still continuously for at least 2 seconds"
        ),
        channel_catalog=catalog,
    )

    assert not result["ok"]
    assert not result["gates"]["continuous_hold_interruption"]
    assert result["archetype_scores"]["catalog_interrupted_hold"] > 0.9
    assert any("[continuous-hold]" in reason for reason in result["reasons"])


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
