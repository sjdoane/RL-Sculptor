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

import json
from pathlib import Path

import pytest

from sculptor.mission import Mission, Stage
from sculptor.mission_metrics import (
    _compute_eval_reset_preview,
    generate_stage_metrics,
    load_stage_reference_clip,
    resolve_stage_metric_ref,
)


def _no_span(*a, **kw):
    """§D24 F1: `generate_stage_metrics` now runs the lazy span-backfill
    (`_backfill_stage_reference_span`) for any stage with a
    `reference_clip_id` and no span fields — which calls
    `sculptor.refs.spans.select_reference_span`, which by default makes a
    REAL Anthropic call. Every test in this module that attaches a
    reference clip WITHOUT itself testing the backfill must monkeypatch
    this in to keep the whole suite offline/deterministic (mirrors the
    `generate_objective_metric` stub pattern already used everywhere
    else here). Returns `(None, None)` — no span, no reason — so it is a
    byte-identical no-op for tests that don't care about the backfill at
    all (no `reference_span_backfill_reason` key gets added to report
    entries, no backfill event fires)."""
    return None, None


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


# ── §D24 F2: getup-shaped fixture clip (eval_reset_preview needs a
#    low-start archetype — the jump clip above derives no eval reset
#    at all) — mirrors test_start_pose_qc_settle.py's `_getup_clip`
#    (duplicated locally, same convention that file's own docstring
#    documents for cross-test-file synthetic clip builders). ─────────
def _getup_clip() -> dict:
    import numpy as np

    fps = 50.0

    def seg(dur, fn):
        n = max(2, int(round(dur * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    lying = seg(0.5, lambda s: np.full_like(s, 0.10))
    ramp = seg(1.0, lambda s: 0.10 + (0.75 - 0.10) * s)
    stand = seg(0.5, lambda s: np.full_like(s, 0.75))
    z = np.concatenate([lying, ramp, stand])
    return {"root_pos_z": z, "fps": fps}


def _write_getup_fixture_clip(tmp_path: Path, robot: str, clip_id: str) -> Path:
    from sculptor.reference import save_clip
    from sculptor.refs import library

    root = tmp_path / "refs_root_getup"
    clip = _getup_clip()
    clip_path = library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME
    save_clip(clip_path, clip)
    content_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot,
        source={"kind": "procedural"}, license="internal",
        attribution="test-fixture", content_sha256_=content_sha)
    library.write_provenance(robot, clip_id, prov, root=root)
    return root


# ── §D24 F2 item 1: _compute_eval_reset_preview (unit-level) ────────────
def test_compute_eval_reset_preview_none_for_non_getup_archetype():
    from sculptor.reference import make_procedural_jump_clip

    assert _compute_eval_reset_preview(
        make_procedural_jump_clip(), robot="g1") is None


def test_compute_eval_reset_preview_settled_success(monkeypatch):
    settled_scalars = {"reset_height_offset_m": -0.3}

    def fake_settle(pre, *, joint_names=None, robot="g1"):
        return {"scalars": settled_scalars, "delta_z_m": -0.02, "steps": 10,
                "converged": True, "duration_s": 0.2}

    monkeypatch.setattr("sculptor.reference.settle_reset", fake_settle)
    out = _compute_eval_reset_preview(_getup_clip(), robot="g1")
    assert out is not None
    assert out["settled"] is True
    assert out["reason"] is None
    assert out["scalars"] == settled_scalars
    assert out["delta_z_m"] == pytest.approx(-0.02)


def test_compute_eval_reset_preview_settle_unavailable_falls_back_unsettled(
    monkeypatch,
):
    from sculptor.reference import SettleUnavailable

    def raising_settle(pre, *, joint_names=None, robot="g1"):
        raise SettleUnavailable("mujoco unavailable in test")

    monkeypatch.setattr("sculptor.reference.settle_reset", raising_settle)
    out = _compute_eval_reset_preview(_getup_clip(), robot="g1")
    assert out is not None
    assert out["settled"] is False
    assert "mujoco unavailable" in out["reason"]
    assert out["scalars"] is not None  # the raw derive_eval_reset output


def test_compute_eval_reset_preview_rs_settle_reset_disabled(monkeypatch):
    monkeypatch.setenv("RS_SETTLE_RESET", "0")

    def boom(*a, **kw):  # pragma: no cover — must not run
        raise AssertionError("settle_reset must not be called when disabled")

    monkeypatch.setattr("sculptor.reference.settle_reset", boom)
    out = _compute_eval_reset_preview(_getup_clip(), robot="g1")
    assert out is not None
    assert out["settled"] is False
    assert out["reason"] == "RS_SETTLE_RESET=0"


def test_compute_eval_reset_preview_derive_failure_never_raises(monkeypatch):
    monkeypatch.setattr(
        "sculptor.reference.derive_eval_reset",
        lambda clip: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _compute_eval_reset_preview(_getup_clip(), robot="g1")
    assert out is not None
    assert out["settled"] is False
    assert out["scalars"] is None
    assert "boom" in out["reason"]


# ── §D24 F2 item 1: wiring into generate_stage_metrics ──────────────────
def test_eval_reset_preview_persisted_next_to_meta_json(tmp_path, monkeypatch):
    """A getup-archetype stage's cert-time eval-reset preview is computed
    via the SAME settle path the scaffold uses and persisted as
    `stage_metrics/<stage>/eval_reset_preview.json` — the same dir as
    `meta.json`."""
    root = _write_getup_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

    settled_scalars = {"reset_height_offset_m": -0.3, "reset_pitch_offset_rad": 0.1}

    def fake_settle(pre, *, joint_names=None, robot="g1"):
        return {"scalars": settled_scalars, "delta_z_m": -0.02, "steps": 10,
                "converged": True, "duration_s": 0.2}

    monkeypatch.setattr("sculptor.reference.settle_reset", fake_settle)

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    preview_path = (
        Path(m.mission_dir) / "stage_metrics" / "get_up"
        / "eval_reset_preview.json")
    assert preview_path.is_file()
    payload = json.loads(preview_path.read_text())
    assert payload["settled"] is True
    assert payload["scalars"]["reset_height_offset_m"] == pytest.approx(-0.3)


def test_eval_reset_preview_settle_unavailable_persists_unsettled(
    tmp_path, monkeypatch,
):
    """The settle-unavailable path (monkeypatched to raise) still
    persists a preview — the UNSETTLED `derive_eval_reset` output, with
    `settled: false` and a reason — never fatal, never silent."""
    root = _write_getup_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

    from sculptor.reference import SettleUnavailable

    def raising_settle(pre, *, joint_names=None, robot="g1"):
        raise SettleUnavailable("mujoco unavailable in test")

    monkeypatch.setattr("sculptor.reference.settle_reset", raising_settle)

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    preview_path = (
        Path(m.mission_dir) / "stage_metrics" / "get_up"
        / "eval_reset_preview.json")
    assert preview_path.is_file()
    payload = json.loads(preview_path.read_text())
    assert payload["settled"] is False
    assert "SettleUnavailable" in payload["reason"] or "mujoco" in payload["reason"]
    assert payload["scalars"] is not None


def test_generate_stage_metrics_threads_eval_reset_into_generation(
    tmp_path, monkeypatch,
):
    root = _write_getup_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)
    monkeypatch.setenv("RS_SETTLE_RESET", "0")  # keep this test physics-free

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
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    assert calls[0]["eval_reset"] is not None
    assert calls[0]["eval_reset"]["scalars"] is not None
    assert calls[0]["eval_reset"]["settled"] is False


def test_generate_stage_metrics_no_eval_reset_for_non_getup_clip(
    tmp_path, monkeypatch,
):
    """A jump-archetype (or any non-low-start) reference never produces
    an eval-reset preview — no file written, `eval_reset=None` passed
    through unchanged (byte-identical to pre-F2 behavior)."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

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
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    assert calls[0]["eval_reset"] is None
    preview_path = (
        Path(m.mission_dir) / "stage_metrics" / "get_up"
        / "eval_reset_preview.json")
    assert not preview_path.is_file()


# ── §D24 F2 (D deliverable): criterion re-grounding wiring ──────────────
def test_generate_stage_metrics_regrounds_criterion_on_fresh_backfill(
    tmp_path, monkeypatch,
):
    """The lazy backfill mechanism that fixes an EXISTING mission's
    criterion (e.g. g1-standing's torso_righting) without a full
    re-decomposition: `ground_stage_criterion` is invoked with the
    CROPPED clip exactly once, when the span is FRESHLY discovered."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def fake_select(clip, *, goal_text, start_pose=None, llm_call=None):
        return dict(_FAKE_SPAN), None

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", fake_select)

    ground_calls: list[dict] = []

    def fake_ground(stage, clip, **kw):
        ground_calls.append({"stage": stage.name, "n_frames": clip["root_pos_z"].shape[0]})
        stage.success_criterion = "metric > 0.9"
        return {"adopted": True, "rationale": "test"}

    monkeypatch.setattr("sculptor.decompose.ground_stage_criterion", fake_ground)

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1")

    assert len(ground_calls) == 1
    # The threaded clip is the CROPPED one (span [0.0, 1.0]s @ 50fps).
    assert ground_calls[0]["n_frames"] < 90
    assert m.stages[0].success_criterion == "metric > 0.9"

    # A SECOND pass (span already exists) must NOT re-ground.
    generate_stage_metrics(m, robot_hint="Mjlab-Velocity-Flat-Unitree-G1",
                            only_stages=["get_up"])
    assert len(ground_calls) == 1


# ── §D24 F1 item 5: load_stage_reference_clip (the one loader) ─────────
def test_load_stage_reference_clip_none_without_clip_id(tmp_path):
    s = _mk_stage("plain")
    assert load_stage_reference_clip(s, "g1") is None


def test_load_stage_reference_clip_full_clip_without_span(tmp_path, monkeypatch):
    """No span fields on the stage -> the FULL clip comes back unchanged,
    and `span_meta` is None (the explicit "no span applies" contract)."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    from sculptor.reference import make_procedural_jump_clip

    full_n = make_procedural_jump_clip()["root_pos_z"].shape[0]

    s = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    loaded = load_stage_reference_clip(s, "g1")
    assert loaded is not None
    clip_id, clip, span_meta = loaded
    assert clip_id == "getup_demo_clip"
    assert clip["root_pos_z"].shape[0] == full_n
    assert span_meta is None


def test_load_stage_reference_clip_crops_when_span_set(tmp_path, monkeypatch):
    """Span fields present -> the returned clip is CROPPED to them and
    `span_meta` mirrors the stage's persisted span."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    s = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_start_s=0.0, reference_span_end_s=1.0,
        reference_span_confidence=0.82, reference_span_method="llm+snap+qc")
    loaded = load_stage_reference_clip(s, "g1")
    assert loaded is not None
    clip_id, clip, span_meta = loaded
    assert clip_id == "getup_demo_clip"
    n = clip["root_pos_z"].shape[0]
    fps = float(clip["fps"])
    assert n / fps == pytest.approx(1.0, abs=1e-6)
    assert span_meta == {
        "t_start_s": 0.0, "t_end_s": 1.0,
        "confidence": 0.82, "method": "llm+snap+qc",
    }


def test_stage_reference_span_grounds_metric_gen_prompt_in_cropped_signature(
    tmp_path, monkeypatch,
):
    """§D24 F1: the metric-authoring prompt (`metric_gen._build_reference_
    signature_block`, threaded through `generate_stage_metrics`'s
    `references=` kwarg via the one loader) must be grounded in the
    CROPPED clip's signature when a stage carries a span, not the full
    clip's — this is what "the success-criterion authoring prompt is
    grounded in the CROPPED signature" cashes out to in this codebase
    (the generated metric is the trust-gated objective the stage
    actually steers by; see docs/internal/REFERENCE_BUILD_LOG.md D24)."""
    from sculptor.eval.metric_gen import _build_reference_signature_block

    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    s_full = _mk_stage("get_up_full", reference_clip_id="getup_demo_clip")
    s_span = _mk_stage(
        "get_up_span", reference_clip_id="getup_demo_clip",
        reference_span_start_s=0.0, reference_span_end_s=1.0,
        reference_span_confidence=0.8, reference_span_method="llm+snap+qc")

    _, clip_full, span_meta_full = load_stage_reference_clip(s_full, "g1")
    _, clip_span, span_meta_span = load_stage_reference_clip(s_span, "g1")
    assert span_meta_full is None
    assert span_meta_span is not None

    block_full, sig_full = _build_reference_signature_block(
        [("getup_demo_clip", clip_full)])
    block_span, sig_span = _build_reference_signature_block(
        [("getup_demo_clip", clip_span)])

    full_duration = sig_full["getup_demo_clip"]["duration_s"]
    span_duration = sig_span["getup_demo_clip"]["duration_s"]
    assert span_duration == pytest.approx(1.0, abs=1e-3)
    assert full_duration > span_duration + 0.1
    # The rendered prompt block itself carries the CROPPED duration, and
    # must not still carry the full clip's (distinguishable since they
    # differ by > 0.1s per the assertion above).
    assert str(span_duration) in block_span
    assert str(full_duration) not in block_span


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
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

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


def test_generate_stage_metrics_reference_load_error_rejects_stage(
    tmp_path, monkeypatch,
):
    """§M3 audit fix (docs/internal/REFERENCE_BUILD_LOG.md, fresh-context
    Opus adversarial audit): a stage referencing a clip_id that doesn't
    exist on disk must NOT silently downgrade to an UNGATED, no-
    reference metric acceptance — it must be REJECTED (never generated),
    `generate_objective_metric` must never be called for it, and the
    load error is surfaced on the reject entry's `reference_load_error`
    plus a loud `reason`."""
    root = tmp_path / "empty_refs_root"
    root.mkdir()
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage("get_up", reference_clip_id="nonexistent_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    assert calls == []
    assert report["generated"] == []
    assert len(report["rejected"]) == 1
    entry = report["rejected"][0]
    assert entry["stage"] == "get_up"
    assert "reference_load_error" in entry
    assert entry["reference_load_error"]
    assert "attached reference failed to load" in entry["reason"]
    assert "refusing certification without its intended exemplar" in entry["reason"]


def test_generate_stage_metrics_reference_crop_failure_rejects_stage(
    tmp_path, monkeypatch,
):
    """§M3: a clip that LOADS fine but fails to CROP (an out-of-range
    persisted span) is the same failure class as an unloadable clip on
    disk — must reject, never fall through to an ungated acceptance."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_start_s=9999.0, reference_span_end_s=10000.0,
        reference_span_confidence=0.9, reference_span_method="llm+snap+qc")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    assert calls == []
    assert report["generated"] == []
    assert len(report["rejected"]) == 1
    entry = report["rejected"][0]
    assert "reference_load_error" in entry
    assert "attached reference failed to load" in entry["reason"]


# ── §D24 F1 item 4c: lazy span backfill ─────────────────────────────────
_FAKE_SPAN = {
    "t_start_s": 0.0, "t_end_s": 1.0, "confidence": 0.82,
    "rationale": "test span", "method": "llm+snap+qc",
}


def test_generate_stage_metrics_backfills_span_once_and_persists(
    tmp_path, monkeypatch,
):
    """A stage with `reference_clip_id` but no span fields gets ONE
    `select_reference_span` call; the four span fields land on the
    stage (mutated in place), the loader threads the CROPPED clip into
    `generate_objective_metric(references=...)`, and — mirroring how
    every real caller of `generate_stage_metrics` re-saves mission.json
    right after (see `sculptor.mission.save_mission`) — the fields
    survive a save/reload round-trip. A second pass over the same
    (already-backfilled) stage must NOT call selection again."""
    from sculptor.mission import load_mission, save_mission

    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    span_calls: list[dict] = []

    def fake_select(clip, *, goal_text, start_pose=None, llm_call=None):
        span_calls.append({"goal_text": goal_text, "start_pose": start_pose})
        return dict(_FAKE_SPAN), None

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", fake_select)

    # §D24 F2: a FRESH span backfill also triggers criterion re-grounding
    # (a real Anthropic call by default) — stub it, this test is about
    # span backfill, not grounding (see test_criterion_ground.py /
    # the dedicated backfill-wiring test below for that).
    ground_calls: list[dict] = []
    monkeypatch.setattr(
        "sculptor.decompose.ground_stage_criterion",
        lambda stage, clip, **kw: ground_calls.append(
            {"stage": stage.name}) or {"adopted": False, "rationale": "stubbed"})

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

    assert len(span_calls) == 1
    assert len(ground_calls) == 1
    assert m.stages[0].reference_span_start_s == pytest.approx(0.0)
    assert m.stages[0].reference_span_end_s == pytest.approx(1.0)
    assert m.stages[0].reference_span_confidence == pytest.approx(0.82)
    assert m.stages[0].reference_span_method == "llm+snap+qc"

    # The threaded clip is the CROPPED one (~1.0s @ 50fps = ~50 frames),
    # not the full ~1.96s procedural clip (~98 frames).
    clip_id, cropped_clip = calls[0]["references"][0]
    assert clip_id == "getup_demo_clip"
    assert cropped_clip["root_pos_z"].shape[0] < 90

    # Mirrors the real caller contract: mutate in place, caller re-saves.
    save_mission(m, m.mission_dir)
    reloaded = load_mission(m.mission_dir)
    assert reloaded.stages[0].reference_span_start_s == pytest.approx(0.0)
    assert reloaded.stages[0].reference_span_end_s == pytest.approx(1.0)

    # Re-running must NOT re-select — the stage already has a span.
    report2 = generate_stage_metrics(
        reloaded, robot_hint="Mjlab-Velocity-Flat-Unitree-G1",
        only_stages=["get_up"])
    assert len(span_calls) == 1
    # §D24 F2: nor must it re-ground the criterion — that's gated on THIS
    # call being the one that freshly discovered the span.
    assert len(ground_calls) == 1
    assert "reference_span_backfill_reason" not in report2["generated"][0]
    assert "reference_span_backfill_reason" not in report["generated"][0]


def test_generate_stage_metrics_backfill_declined_skips_full_clip_certification(
    tmp_path, monkeypatch,
):
    """§H2 audit fix (docs/internal/REFERENCE_BUILD_LOG.md D23/D24/D25,
    fresh-context Opus adversarial audit): when span selection declines
    for a reason OTHER than `whole_clip` (low confidence, QC reject,
    ...) the sub-span question is UNRESOLVED — certifying against the
    full clip would reproduce the exact D23 class (a sub-phase goal
    certified against the wrong-scope clip). `generate_objective_metric`
    must NEVER be called; the stage is REJECTED with a loud reason
    naming the decline and falls back to the mission-level metric.

    §D24 W5 hardening (still in force): the SEMANTIC decline persists a
    `"declined:<reason>"` marker so the backfill never re-attempts
    selection on a later pass — see the dedicated retry tests below."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    select_calls: list[int] = []

    def fake_select(clip, *, goal_text, start_pose=None, llm_call=None):
        select_calls.append(1)
        return None, "low_confidence:0.42"

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", fake_select)

    calls: list[dict] = []

    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        calls.append(kw)
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    assert m.stages[0].reference_span_start_s is None
    assert m.stages[0].reference_span_method == "declined:low_confidence:0.42"
    assert calls == []  # full-clip certification must NEVER run
    assert report["generated"] == []
    assert len(report["rejected"]) == 1
    entry = report["rejected"][0]
    assert entry["stage"] == "get_up"
    assert entry["reference_span_backfill_reason"] == "low_confidence:0.42"
    assert (
        "reference span selection declined (low_confidence:0.42)"
        in entry["reason"])
    assert "D23 class" in entry["reason"]
    assert "mission-level fallback" in entry["reason"]

    # §D24 W5 hardening: THE regression this fix exists for — a second
    # `generate_stage_metrics` pass over the SAME (still-rejected) stage
    # must make ZERO further `select_reference_span` calls; the declined
    # marker is a standing verdict, not a retry target.
    generate_stage_metrics(m, only_stages=["get_up"])
    assert len(select_calls) == 1


def test_generate_stage_metrics_whole_clip_decline_still_certifies_full_clip(
    tmp_path, monkeypatch,
):
    """§H2: `declined:whole_clip` means the LLM affirmatively judged the
    goal covers the clip's ENTIRE motion — full-clip certification is
    CORRECT there (unlike every other decline reason) and must proceed
    exactly as before this fix."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def fake_select(clip, *, goal_text, start_pose=None, llm_call=None):
        return None, "whole_clip"

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", fake_select)

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
    report = generate_stage_metrics(m)

    assert m.stages[0].reference_span_method == "declined:whole_clip"
    assert len(calls) == 1  # full-clip certification DID run
    assert len(report["generated"]) == 1
    assert report["rejected"] == []

    from sculptor.reference import make_procedural_jump_clip

    full_n = make_procedural_jump_clip()["root_pos_z"].shape[0]
    clip_id, clip = calls[0]["references"][0]
    assert clip["root_pos_z"].shape[0] == full_n


def test_generate_stage_metrics_preexisting_declined_marker_skips_generation(
    tmp_path, monkeypatch,
):
    """§H2: a stage that ALREADY carries an unresolved `declined:<reason>`
    marker (e.g. set at decompose time by `_select_and_attach_span`, not
    via this module's own backfill) must be rejected the same way on its
    very FIRST `generate_stage_metrics` pass — never certifying against
    the full clip, and never re-attempting selection."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def boom(*a, **kw):  # pragma: no cover — must not run
        raise AssertionError("selection must not run — already declined")

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", boom)

    def fake_gen(goal, out_dir, **kw):  # pragma: no cover — must not run
        raise AssertionError(
            "generation must not run for an unresolved decline")

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    s1 = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_method="declined:qc_reject:duration<1.0s:0.300")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)

    assert report["generated"] == []
    assert len(report["rejected"]) == 1
    assert "qc_reject:duration<1.0s:0.300" in report["rejected"][0]["reason"]


def test_unresolved_span_decline_reason_none_absent_and_whole_clip():
    """Direct unit coverage of `_unresolved_span_decline_reason`: no
    marker or a whole_clip marker -> None (proceed normally); every other
    decline -> the bare reason string; a SUCCESS method (not a decline
    marker at all) -> None."""
    from sculptor.mission_metrics import _unresolved_span_decline_reason

    s = _mk_stage("s")
    assert _unresolved_span_decline_reason(s) is None

    s.reference_span_method = "declined:whole_clip"
    assert _unresolved_span_decline_reason(s) is None

    s.reference_span_method = "declined:low_confidence:0.42"
    assert _unresolved_span_decline_reason(s) == "low_confidence:0.42"

    s.reference_span_method = "declined:qc_reject:crop_error:boom"
    assert _unresolved_span_decline_reason(s) == "qc_reject:crop_error:boom"

    s.reference_span_method = "llm+snap+qc"  # a SUCCESS marker, not a decline
    assert _unresolved_span_decline_reason(s) is None


def test_backfill_llm_unavailable_persists_no_marker_and_is_retried(
    tmp_path, monkeypatch,
):
    """§D24 W5 hardening: an INFRA failure (llm_unavailable/parse_error/
    invalid_clip/signature_error) is NOT a standing verdict — no marker
    is persisted, so the VERY NEXT `generate_stage_metrics` pass retries
    selection (as opposed to the semantic-decline case above, which
    never retries)."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    select_calls: list[int] = []

    def fake_select(clip, *, goal_text, start_pose=None, llm_call=None):
        select_calls.append(1)
        return None, "llm_unavailable:APIConnectionError: timeout"

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", fake_select)

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage("get_up", reference_clip_id="getup_demo_clip")
    m = _mk_mission(tmp_path, [s1])
    generate_stage_metrics(m)
    assert m.stages[0].reference_span_start_s is None
    assert m.stages[0].reference_span_method is None
    assert len(select_calls) == 1

    generate_stage_metrics(m, only_stages=["get_up"])
    assert len(select_calls) == 2  # retried — no standing marker blocked it


def test_declined_marker_stage_still_certifies_against_full_clip(
    tmp_path, monkeypatch,
):
    """A stage carrying a `"declined:<reason>"` marker (start/end/
    confidence None) must resolve to the FULL clip through the ONE
    loader — the marker changes retry behavior, never the "no span
    applies" semantics `load_stage_reference_clip` already implements
    off of `reference_span_start_s`."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    from sculptor.reference import make_procedural_jump_clip

    full_n = make_procedural_jump_clip()["root_pos_z"].shape[0]

    s = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_method="declined:whole_clip")
    loaded = load_stage_reference_clip(s, "g1")
    assert loaded is not None
    clip_id, clip, span_meta = loaded
    assert clip_id == "getup_demo_clip"
    assert clip["root_pos_z"].shape[0] == full_n
    assert span_meta is None


def test_generate_stage_metrics_backfill_skipped_when_span_already_set(
    tmp_path, monkeypatch,
):
    """A stage that already carries span fields (e.g. set at decompose
    time) must never re-trigger selection during backfill."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    def boom(*a, **kw):  # pragma: no cover — must not run
        raise AssertionError("selection must not run when a span already exists")

    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", boom)

    def fake_gen(goal, out_dir, **kw):
        return {"accepted": True}

    import sculptor.eval as ev
    monkeypatch.setattr(ev, "generate_objective_metric", fake_gen)

    import sculptor.eval.metric_calibration as mc
    monkeypatch.setattr(
        mc, "calibrate_metric_against_reference", lambda *a, **kw: None)

    s1 = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_start_s=0.0, reference_span_end_s=1.0,
        reference_span_confidence=0.9, reference_span_method="llm+snap+qc")
    m = _mk_mission(tmp_path, [s1])
    report = generate_stage_metrics(m)  # must not raise
    assert "reference_span_backfill_reason" not in report["generated"][0]


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
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

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
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

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
    monkeypatch.setattr("sculptor.refs.spans.select_reference_span", _no_span)

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


def test_rejected_stage_clears_stale_steering_metric(tmp_path, monkeypatch):
    """LIVE FINDING (2026-07-12, post-audit): the runtime resolves
    `steering_metric or fitness_metric` WITHOUT re-checking acceptance,
    so a rejection must clear a stale pointer or the stage keeps
    steering by it — seen live twice: a grandfathered pre-D24 full-clip
    metric behind an unresolved span decline (feet_under_crouch), and a
    regen that overwrote an accepted metric.py with the REJECTED
    candidate while the pointer survived (drive_to_stand)."""
    root = _write_fixture_clip(tmp_path, "g1", "getup_demo_clip")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    s1 = _mk_stage(
        "get_up", reference_clip_id="getup_demo_clip",
        reference_span_method="declined:low_confidence:0.42")
    s1.steering_metric = "stage_metrics/get_up/metric.py"
    m = _mk_mission(tmp_path, [s1])
    # only_stages mirrors the per-stage regen endpoint - the one
    # path that bypasses the 'steering_metric already set'
    # whole-mission skip guard, and the path both live cases used.
    report = generate_stage_metrics(m, only_stages=["get_up"])

    assert len(report["rejected"]) == 1
    entry = report["rejected"][0]
    assert entry["steering_metric_cleared"] == "stage_metrics/get_up/metric.py"
    assert s1.steering_metric is None
