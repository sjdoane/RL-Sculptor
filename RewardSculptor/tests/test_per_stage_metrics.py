"""§Ship 38 (M1): per-stage steering metrics — data model + decompose
authoring. GPU/LLM-free (the orchestration wiring is covered in
test_mission_run.py::test_mission_run_per_stage_steering_metric)."""
from __future__ import annotations

import pytest

from sculptor.decompose import (
    _StageModel,
    _render_fitness_metrics_block,
    _stages_from_model,
    _validate_steering_metrics,
)
from sculptor.mission import (
    Mission,
    MissionValidationError,
    Stage,
    validate_mission,
)


def _stage(**kw) -> Stage:
    base = dict(name="a", goal_text="g", success_criterion="metric > 0",
                max_iterations=3, parent_stage=None, reward_seed_prompt="seed here")
    base.update(kw)
    return Stage(**base)


def test_stage_steering_metric_roundtrip() -> None:
    s = _stage(steering_metric="g1_kick")
    d = s.to_dict()
    assert d["steering_metric"] == "g1_kick"
    assert Stage.from_dict(d).steering_metric == "g1_kick"
    # older mission.json without the field loads with None (backward-compat).
    d.pop("steering_metric")
    assert Stage.from_dict(d).steering_metric is None


def test_mission_validation_rejects_empty_steering_metric() -> None:
    m = Mission(goal="g", stages=[_stage(steering_metric="   ")],
                decomposition_model="x", decomposition_rationale="y")
    with pytest.raises(MissionValidationError):
        validate_mission(m, info_keys=set())


def test_mission_validation_accepts_none_and_named_metric() -> None:
    for sm in (None, "g1_kick", "some_generated_metric.py"):
        m = Mission(goal="g", stages=[_stage(steering_metric=sm)],
                    decomposition_model="x", decomposition_rationale="y")
        validate_mission(m, info_keys=set())   # must not raise


def test_validate_steering_metrics_rejects_unknown_spec_name() -> None:
    _validate_steering_metrics([_stage(steering_metric="g1_kick")])   # known → ok
    _validate_steering_metrics([_stage(steering_metric=None)])        # None → ok
    with pytest.raises(MissionValidationError):
        _validate_steering_metrics([_stage(steering_metric="not_a_metric")])


def test_stages_from_model_carries_and_normalizes_metric() -> None:
    sm = _StageModel(name="a", goal_text="g", success_criterion="metric > 0",
                     max_iterations=3, parent_stage=None,
                     reward_seed_prompt="seed prompt here",
                     steering_metric="g1_kick")
    assert _stages_from_model([sm])[0].steering_metric == "g1_kick"
    # whitespace-only normalizes to None at parse time.
    sm2 = _StageModel(name="b", goal_text="g", success_criterion="metric > 0",
                      max_iterations=3, parent_stage=None,
                      reward_seed_prompt="seed prompt here", steering_metric="   ")
    assert _stages_from_model([sm2])[0].steering_metric is None


def test_fitness_metrics_block_lists_builtin_names() -> None:
    block = _render_fitness_metrics_block()
    assert "AVAILABLE_FITNESS_METRICS" in block
    assert "g1_kick" in block and "go1_trot" in block and "cartpole_balance" in block
