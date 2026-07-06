"""tests/test_mission_metrics.py — §MISSION_METRIC_GRANULARITY.

Per-stage objective-metric generation at decomposition time. No live
LLM calls: `generate_objective_metric` is monkeypatched at its import
site inside `sculptor.eval`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.mission import Mission, Stage
from sculptor.mission_metrics import (
    generate_stage_metrics,
    resolve_stage_metric_ref,
)


def _mk_stage(name: str, **kw) -> Stage:
    return Stage(
        name=name,
        goal_text=f"goal for {name}",
        success_criterion="metric > 0.5",
        max_iterations=3,
        parent_stage=None,
        reward_seed_prompt="seed",
        **kw,
    )


def _mk_mission(tmp_path: Path, stages: list[Stage]) -> Mission:
    m = Mission(goal="stand then jump", stages=stages,
                decomposition_model="test", decomposition_rationale="test")
    m.mission_dir = str(tmp_path / "mission")
    Path(m.mission_dir).mkdir(parents=True, exist_ok=True)
    return m


# ── resolve_stage_metric_ref ───────────────────────────────────────────
def test_resolve_ref_spec_name_passes_through(tmp_path):
    assert resolve_stage_metric_ref("g1_kick", tmp_path) == "g1_kick"


def test_resolve_ref_absolute_path_passes_through(tmp_path):
    abs_ref = str(tmp_path / "m.py")
    assert resolve_stage_metric_ref(abs_ref, tmp_path / "other") == abs_ref


def test_resolve_ref_relative_py_is_anchored(tmp_path):
    got = resolve_stage_metric_ref("stages/s0/metric/metric.py", tmp_path)
    assert got == str(tmp_path / "stages/s0/metric/metric.py")


# ── generate_stage_metrics ─────────────────────────────────────────────
def test_accepted_metric_sets_relative_ref(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append({"goal": goal, "out_dir": Path(out_dir), **kw})
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    m = _mk_mission(tmp_path, [_mk_stage("stand_tall")])
    report = generate_stage_metrics(m, robot_hint="Mjlab-…-G1")
    assert report["generated"] == [
        {"stage": "stand_tall", "ref": "stages/stand_tall/metric/metric.py"}]
    assert m.stages[0].steering_metric == "stages/stand_tall/metric/metric.py"
    # ≤128 chars — inside the mission validator's steering_metric bound.
    assert len(m.stages[0].steering_metric) <= 128
    assert calls[0]["goal"] == "goal for stand_tall"
    assert calls[0]["out_dir"] == (
        Path(m.mission_dir) / "stages" / "stand_tall" / "metric")
    assert calls[0]["robot_hint"] == "Mjlab-…-G1"


def test_rejected_metric_leaves_fallback_and_reports(tmp_path, monkeypatch):
    def fake_gen(goal, out_dir, **kw):
        return {"accepted": False,
                "validation": {"reasons": ["degenerate: constant output"]}}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    m = _mk_mission(tmp_path, [_mk_stage("crouch_launch")])
    report = generate_stage_metrics(m)
    assert m.stages[0].steering_metric is None
    assert report["generated"] == []
    assert report["rejected"] == [
        {"stage": "crouch_launch", "reason": "degenerate: constant output"}]


def test_existing_steering_metric_and_succeeded_stage_are_skipped(
    tmp_path, monkeypatch
):
    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        raise AssertionError("generation must not run for skipped stages")

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("has_metric", steering_metric="g1_kick")
    s2 = _mk_stage("done_stage")
    s2.status = "succeeded"
    m = _mk_mission(tmp_path, [s1, s2])
    report = generate_stage_metrics(m)
    assert len(report["skipped"]) == 2
    assert report["generated"] == [] and report["rejected"] == []


def test_generation_crash_falls_back_and_never_raises(tmp_path, monkeypatch):
    def fake_gen(goal, out_dir, **kw):
        raise RuntimeError("API exploded")

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    m = _mk_mission(tmp_path, [_mk_stage("land_softly")])
    report = generate_stage_metrics(m)  # must not raise
    assert m.stages[0].steering_metric is None
    assert report["rejected"][0]["stage"] == "land_softly"
    assert "API exploded" in report["rejected"][0]["reason"]


def test_missing_mission_dir_raises(tmp_path):
    m = Mission(goal="g", stages=[_mk_stage("s")],
                decomposition_model="test", decomposition_rationale="test")
    m.mission_dir = None
    with pytest.raises(RuntimeError, match="mission_dir is None"):
        generate_stage_metrics(m)
