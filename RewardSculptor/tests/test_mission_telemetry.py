"""§Ship 25b (H2): decomposition-quality telemetry tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sculptor.sculpt import _write_mission_telemetry


def _stage_result(name: str, status: str, iters: int = 3,
                  satisfied: bool | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        stage_name=name, status=status, iterations_used=iters,
        criterion_satisfied=(status == "succeeded") if satisfied is None else satisfied,
    )


def _mission(n_stages: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        goal="do a flip",
        stages=[SimpleNamespace(name=f"s{i}") for i in range(n_stages)],
    )


def _result(*stage_results, completed=True, halted_reason=None) -> SimpleNamespace:
    return SimpleNamespace(
        stage_results=list(stage_results),
        completed=completed,
        halted_reason=halted_reason,
        halted_at_stage=None,
    )


def test_telemetry_written_to_mission_dir(tmp_path: Path) -> None:
    md = tmp_path / "mission"
    md.mkdir()
    events: list[dict] = []
    _write_mission_telemetry(
        _mission(3), md,
        _result(_stage_result("s0", "succeeded", 4),
                _stage_result("s1", "succeeded", 6),
                _stage_result("s2", "failed", 8, satisfied=False),
                completed=False, halted_reason="criterion_not_met"),
        n_stages_at_start=3, redecompositions=1, emit=events.append,
    )
    doc = json.loads((md / "telemetry.json").read_text(encoding="utf-8"))
    assert doc["n_stages_at_start"] == 3
    assert doc["n_stages_final"] == 3
    assert doc["stages_executed"] == 3
    assert doc["stages_succeeded"] == 2
    assert doc["stage_success_rate"] == round(2 / 3, 4)
    assert doc["redecompositions"] == 1
    assert doc["iterations_total"] == 18
    assert doc["completed"] is False
    assert doc["halted_reason"] == "criterion_not_met"
    assert [s["name"] for s in doc["per_stage"]] == ["s0", "s1", "s2"]
    assert [e["type"] for e in events] == ["mission_telemetry_written"]
    assert events[0]["stage_success_rate"] == round(2 / 3, 4)
    # Bare tmp dir (not a .missions layout) → no project aggregate.
    assert events[0]["aggregate_path"] is None


def test_telemetry_aggregates_under_missions_layout(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    md = project / ".missions" / "flip-mission"
    md.mkdir(parents=True)
    events: list[dict] = []
    _write_mission_telemetry(
        _mission(2), md,
        _result(_stage_result("s0", "succeeded")),
        n_stages_at_start=2, redecompositions=0, emit=events.append,
    )
    agg = json.loads(
        (project / "reports" / "mission_quality.json").read_text(encoding="utf-8")
    )
    assert len(agg["missions"]) == 1
    assert agg["missions"][0]["mission_slug"] == "flip-mission"
    assert events[0]["aggregate_path"].endswith("mission_quality.json")

    # Re-running the SAME mission replaces its record (no duplicates);
    # other missions accumulate.
    _write_mission_telemetry(
        _mission(2), md,
        _result(_stage_result("s0", "succeeded"), _stage_result("s1", "succeeded")),
        n_stages_at_start=2, redecompositions=0, emit=events.append,
    )
    md2 = project / ".missions" / "other-mission"
    md2.mkdir(parents=True)
    _write_mission_telemetry(
        _mission(1), md2,
        _result(_stage_result("a", "failed", satisfied=False), completed=False,
                halted_reason="x"),
        n_stages_at_start=1, redecompositions=0, emit=events.append,
    )
    agg = json.loads(
        (project / "reports" / "mission_quality.json").read_text(encoding="utf-8")
    )
    slugs = [m["mission_slug"] for m in agg["missions"]]
    assert slugs == ["flip-mission", "other-mission"]
    flip = agg["missions"][0]
    assert flip["stages_executed"] == 2, "rerun must replace, not duplicate"


def test_telemetry_corrupt_aggregate_recovers(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    md = project / ".missions" / "m"
    md.mkdir(parents=True)
    (project / "reports").mkdir()
    (project / "reports" / "mission_quality.json").write_text("{broken")
    events: list[dict] = []
    _write_mission_telemetry(
        _mission(1), md, _result(_stage_result("s0", "succeeded")),
        n_stages_at_start=1, redecompositions=0, emit=events.append,
    )
    agg = json.loads(
        (project / "reports" / "mission_quality.json").read_text(encoding="utf-8")
    )
    assert len(agg["missions"]) == 1
    assert [e["type"] for e in events] == ["mission_telemetry_written"]


def test_telemetry_never_raises(tmp_path: Path) -> None:
    """A telemetry failure must emit mission_telemetry_failed, not crash
    the mission's final accounting."""
    events: list[dict] = []
    # mission_dir is a FILE → write_json_atomic will fail.
    bogus = tmp_path / "not_a_dir"
    bogus.write_text("x")
    _write_mission_telemetry(
        _mission(1), bogus, _result(_stage_result("s0", "succeeded")),
        n_stages_at_start=1, redecompositions=0, emit=events.append,
    )
    assert [e["type"] for e in events] == ["mission_telemetry_failed"]


def test_telemetry_zero_executed_stages(tmp_path: Path) -> None:
    md = tmp_path / "m"
    md.mkdir()
    events: list[dict] = []
    _write_mission_telemetry(
        _mission(0), md, _result(completed=False, halted_reason="interrupted"),
        n_stages_at_start=0, redecompositions=0, emit=events.append,
    )
    doc = json.loads((md / "telemetry.json").read_text(encoding="utf-8"))
    assert doc["stage_success_rate"] is None
    assert doc["iterations_total"] == 0
