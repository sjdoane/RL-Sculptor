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
    completed = float(arrays["goal__complete_course__success"].mean())
    physical = float(height_gain > 0.9)
    progress = float(np.clip(waypoint_max / 4.0, 0.0, 1.0))
    return {
        "spec_score": float(physical * ordered_steps * progress * completed),
        "physical_traversal": physical,
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
