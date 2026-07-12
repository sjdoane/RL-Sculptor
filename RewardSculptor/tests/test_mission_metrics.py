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
    `test_refs_library.py`'s conventions. Returns the root.

    §F7: `content_sha256` is the REAL sha256 of the written clip.npz
    bytes (not a placeholder) — `verify_tierd_certificate` now
    recomputes this from disk, so a fake hash here would make the
    end-to-end real-cert wiring test deny for the wrong reason."""
    from sculptor.reference import make_procedural_jump_clip, save_clip
    from sculptor.refs import library

    root = tmp_path / "refs_root"
    clip = make_procedural_jump_clip()
    clip_path = library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME
    save_clip(clip_path, clip)
    content_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot,
        source={"kind": "procedural"}, license="internal",
        attribution="test-fixture", content_sha256_=content_sha)
    library.write_provenance(robot, clip_id, prov, root=root)
    return root


def test_generate_stage_metrics_loads_and_passes_reference(
    tmp_path, monkeypatch,
):
    """A stage with `reference_clip_id` set gets the clip loaded from the
    library and threaded through `generate_objective_metric(references=...)`.
    Reference-CALIBRATION wiring is a separate concern (see the dedicated
    tests below) — stubbed out here so this test stays about reference
    loading/threading only."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

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


# ── §REFERENCE_TRAJECTORY_PLAN §6 audit-finding close: calibration wiring ──
#
# `generate_stage_metrics` invokes `calibrate_metric_against_reference`
# right after an accepted stage metric's reference gets loaded — but ONLY
# for stages that actually have one. `calibrate_metric_against_reference`
# itself resolves the effective tier (never a caller-supplied string, see
# tests/test_reference_calibration.py); these tests check the PIPELINE
# seam: is it called with the right args, is it skipped when there's no
# reference, and does it crashing leave metric acceptance untouched.
def test_stage_with_reference_triggers_calibration_with_resolved_tier(
    tmp_path, monkeypatch,
):
    """A stage with a reference gets calibration invoked with the SAME
    (metric_path, clip_id, clip, robot) the reference-loading seam
    resolved — and the resolved tier/rights it returns land in both the
    report entry and an emitted event."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def fake_gen(goal, out_dir, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metric.py").write_text("def compute_spec(a, b, c): return {}")
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    cal_calls: list[dict] = []

    def fake_calibrate(module_path_or_source, clip_id, clip, **kw):
        cal_calls.append({"module_path_or_source": module_path_or_source,
                           "clip_id": clip_id, "clip": clip, **kw})
        return {"ok": True, "trust_tier": "reference:D:procedural",
                "rights": "steer", "cert_verified": True, "cert_reason": None}

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(mc, "calibrate_metric_against_reference", fake_calibrate)

    events: list[dict] = []
    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(
        m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1",
        on_event=events.append)

    assert len(cal_calls) == 1
    call = cal_calls[0]
    assert call["clip_id"] == "getup_demo_clip"
    assert call["robot"] == "g1"
    assert Path(call["module_path_or_source"]) == (
        Path(m.mission_dir) / "stage_metrics" / "get_up" / "metric.py")
    assert "root_pos_z" in call["clip"]

    entry = report["generated"][0]
    assert entry["reference_calibration"] == {
        "clip_id": "getup_demo_clip", "robot": "g1",
        "trust_tier": "reference:D:procedural", "rights": "steer",
        "ok": True, "cert_verified": True, "cert_reason": None,
    }
    cal_events = [e for e in events if e["type"] == "stage_metric_reference_calibrated"]
    assert len(cal_events) == 1
    assert cal_events[0]["stage"] == "get_up"
    assert cal_events[0]["trust_tier"] == "reference:D:procedural"
    assert cal_events[0]["rights"] == "steer"
    assert cal_events[0]["cert_verified"] is True


def test_stage_without_reference_never_calls_calibration(tmp_path, monkeypatch):
    """Byte-level no-op for a stage with no attached reference: calibration
    must not even be imported/called, and no `reference_calibration` key
    appears on the report entry."""
    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    def fake_calibrate(*a, **kw):  # pragma: no cover — must not run
        raise AssertionError("calibration must not run for a stage without a reference")

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(mc, "calibrate_metric_against_reference", fake_calibrate)

    m = _mk_mission(tmp_path, [_mk_stage("plain_stage")])
    report = generate_stage_metrics(m)

    assert report["generated"] == [
        {"stage": "plain_stage", "ref": "stage_metrics/plain_stage/metric.py"}]
    assert "reference_calibration" not in report["generated"][0]


def test_calibration_raising_never_breaks_metric_acceptance(tmp_path, monkeypatch):
    """A calibration crash is swallowed — the stage's metric is still
    accepted (steering_metric set, reported as generated), just without a
    `reference_calibration` entry, and no `stage_metric_reference_
    calibrated` event fires."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    def fake_calibrate(*a, **kw):
        raise RuntimeError("calibration blew up")

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(mc, "calibrate_metric_against_reference", fake_calibrate)

    events: list[dict] = []
    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m, on_event=events.append)  # must not raise

    assert report["generated"] == [
        {"stage": "get_up", "ref": "stage_metrics/get_up/metric.py"}]
    assert m.stages[0].steering_metric == "stage_metrics/get_up/metric.py"
    assert "reference_calibration" not in report["generated"][0]
    assert not any(e["type"] == "stage_metric_reference_calibrated" for e in events)


def test_calibration_wiring_end_to_end_resolves_tier_d_from_real_cert(
    tmp_path, monkeypatch,
):
    """Full integration (no mocked calibration): a clip that genuinely has
    a verified Tier-D certificate on disk earns steer through the WHOLE
    chain — mission_metrics -> calibrate_metric_against_reference ->
    verify_tierd_certificate — with no explicit tier ever passed by
    mission_metrics.py."""
    from sculptor.refs import library
    from sculptor.refs.track import TrackingErrors, update_provenance_tier_d

    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    # §F7: rollout_path must resolve INSIDE the library root — mirrors
    # `track_clip`'s own `clip_d / "tierD_rollout.npz"` convention.
    rollout_path = (
        library.clip_dir("g1", "getup_demo_clip", root=root)
        / "tierD_rollout.npz")
    rollout_path.write_bytes(b"a real tracking rollout for the wiring test")
    errs = TrackingErrors(
        mean_joint_err_rad=0.05, max_joint_err_rad=0.1, root_z_rmse_m=0.02,
        duration_coverage=1.0, common_joint_names=[], n_common_joints=0)
    update_provenance_tier_d(
        robot="g1", clip_id="getup_demo_clip", errors=errs, iterations=1,
        rollout_path=rollout_path, root=root)

    def fake_gen(goal, out_dir, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metric.py").write_text(
            "def compute_spec(a, b, c): return {'spec_score': 0.0}")
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    cal = report["generated"][0]["reference_calibration"]
    assert cal["cert_verified"] is True
    assert cal["trust_tier"] == "reference:D:procedural"
    # (rights depends on the trivial constant metric clearing the ladder
    # gate, which it won't — the load-bearing assertion here is that the
    # TIER resolved to D through the real cert chain, not steer/observe.)
