"""tests/test_mission_metrics.py — §MISSION_METRIC_GRANULARITY.

Per-stage objective-metric generation at decomposition time. No live
LLM calls: `generate_objective_metric` is monkeypatched at its import
site inside `sculptor.eval`.

§REFERENCE_TRAJECTORY_PLAN §7: the reference-clip-loading tests below use
a tmp `RS_REFERENCE_ROOT` library fixture (built via `sculptor.refs.library`
+ `sculptor.reference.save_clip`, same conventions as `test_refs_library.py`)
— no network, no real Anthropic calls.
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
    got = resolve_stage_metric_ref("stage_metrics/s0/metric.py", tmp_path)
    assert got == str(tmp_path / "stage_metrics/s0/metric.py")


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
        {"stage": "stand_tall", "ref": "stage_metrics/stand_tall/metric.py"}]
    assert m.stages[0].steering_metric == "stage_metrics/stand_tall/metric.py"
    # ≤128 chars — inside the mission validator's steering_metric bound.
    assert len(m.stages[0].steering_metric) <= 128
    assert calls[0]["goal"] == "goal for stand_tall"
    assert calls[0]["out_dir"] == (
        Path(m.mission_dir) / "stage_metrics" / "stand_tall")
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


# ── §mission-persistence increment 1: only_stages ──────────────────────
def test_only_stages_bypasses_existing_metric_guard_for_named_stage(
    tmp_path, monkeypatch,
):
    """A user-triggered regenerate targeting a specific stage must
    overwrite its EXISTING steering_metric (the normal whole-mission
    pass would skip it) while leaving every other stage untouched —
    not even reported as skipped, since it was never a candidate."""
    calls: list[str] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append(Path(out_dir).name)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("has_metric", steering_metric="g1_kick")
    s2 = _mk_stage("other_stage", steering_metric="g1_stand")
    m = _mk_mission(tmp_path, [s1, s2])

    report = generate_stage_metrics(m, only_stages=["has_metric"])

    # Only the named stage was regenerated.
    assert calls == ["has_metric"]
    assert report["generated"] == [
        {"stage": "has_metric", "ref": "stage_metrics/has_metric/metric.py"}]
    assert m.stages[0].steering_metric == "stage_metrics/has_metric/metric.py"
    # The un-named stage is completely untouched — not regenerated,
    # not reported as skipped.
    assert m.stages[1].steering_metric == "g1_stand"
    assert report["skipped"] == []
    assert report["rejected"] == []


def test_only_stages_still_skips_succeeded_stage(tmp_path, monkeypatch):
    """Even an explicit only_stages regenerate must not touch a stage
    that already succeeded — that guard is never bypassed."""
    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        raise AssertionError("generation must not run for a succeeded stage")

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("done_stage", steering_metric="g1_kick")
    s1.status = "succeeded"
    m = _mk_mission(tmp_path, [s1])

    report = generate_stage_metrics(m, only_stages=["done_stage"])
    assert report["generated"] == []
    assert report["skipped"] == [
        {"stage": "done_stage", "reason": "stage already succeeded"}]
    assert m.stages[0].steering_metric == "g1_kick"


def test_default_pass_skips_superseded_stage(tmp_path, monkeypatch):
    """The default whole-mission pass (only_stages=None) must skip a
    superseded stage — it is terminal and will never train again, so
    generating a metric for it wastes an LLM call."""
    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        raise AssertionError("generation must not run for a superseded stage")

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("old_parent")
    s1.status = "superseded"
    m = _mk_mission(tmp_path, [s1])

    report = generate_stage_metrics(m)
    assert report["generated"] == []
    assert report["skipped"] == [
        {"stage": "old_parent", "reason": "stage superseded"}]


# ── §REFERENCE_TRAJECTORY_PLAN §7: per-stage reference loading ─────────
def _write_fixture_clip(tmp_path: Path, robot: str, clip_id: str) -> Path:
    """Build a minimal valid on-disk library clip (provenance + clip.npz)
    under a tmp `RS_REFERENCE_ROOT`-shaped root, mirroring
    `test_refs_library.py`'s conventions. Returns the root."""
    from sculptor.reference import make_procedural_jump_clip, save_clip
    from sculptor.refs import library

    root = tmp_path / "refs_root"
    clip = make_procedural_jump_clip()
    save_clip(library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME, clip)
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot,
        source={"kind": "procedural"}, license="internal",
        attribution="test-fixture", content_sha256_="a" * 64)
    library.write_provenance(robot, clip_id, prov, root=root)
    return root


def test_generate_stage_metrics_loads_and_passes_reference(
    tmp_path, monkeypatch,
):
    """A stage with `reference_clip_id` set gets the clip loaded from the
    library and threaded through `generate_objective_metric(references=...)`."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    assert report["generated"] == [
        {"stage": "get_up", "ref": "stage_metrics/get_up/metric.py"}]
    assert len(calls) == 1
    refs = calls[0]["references"]
    assert refs is not None and len(refs) == 1
    clip_id, clip = refs[0]
    assert clip_id == "getup_demo_clip"
    assert "root_pos_z" in clip


def test_generate_stage_metrics_no_reference_clip_id_passes_none(
    tmp_path, monkeypatch,
):
    """A stage with no reference attached must pass references=None —
    byte-identical to the pre-existing call shape."""
    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    m = _mk_mission(tmp_path, [_mk_stage("plain_stage")])
    generate_stage_metrics(m)
    assert calls[0]["references"] is None


def test_generate_stage_metrics_reference_load_error_proceeds_without(
    tmp_path, monkeypatch,
):
    """A stage referencing a clip_id that doesn't exist on disk must NOT
    crash the pipeline — it proceeds with references=None and records
    `reference_load_error` on the report entry."""
    root = tmp_path / "empty_refs_root"
    root.mkdir()
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("get_up", reference_clip_id="nonexistent_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    assert calls[0]["references"] is None
    assert len(report["generated"]) == 1
    assert "reference_load_error" in report["generated"][0]
    assert report["generated"][0]["reference_load_error"]
