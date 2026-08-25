"""tests/test_reference_spans.py — D24 F1: `sculptor.refs.spans` (phase-cropped
stage reference sub-spans) + the `kinematic_signature` orientation-timeline
addition in `sculptor/refs/convert.py`.

D23 (docs/internal/REFERENCE_BUILD_LOG.md) diagnosed a live zero-fitness
regression: a stage goal that is a strict SUB-PHASE of its attached reference
clip can never pass certification against the FULL clip. These tests cover
the machinery that selects/crops the goal-aligned sub-span:

  * `crop_span` — frame-range cropping (real fixture + error paths);
  * `kinematic_signature`'s new `orientation.timeline` key (additive, quat-
    gated);
  * `select_reference_span` — the one-LLM-call span proposal, ALWAYS run
    against a mocked `llm_call` here (never touches the network), covering
    every documented rejection path plus determinism.

Uses the REAL fixture clip at fixtures/torso_righting_satup/reference_clip.npz
(the exact clip D23 diagnosed) for the shape-dependent assertions, and small
synthetic clips (mirroring test_refs_convert.py's conventions) for the pure
quat/no-quat signature case.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.reference import _archetype, load_clip, validate_clip
from sculptor.refs.convert import kinematic_signature
from sculptor.refs.perturb import perturbation_suite
from sculptor.refs.spans import (
    _span_qc,
    crop_span,
    select_reference_span,
    span_boundaries,
)

FIXTURE_CLIP = (
    Path(__file__).parent / "fixtures" / "torso_righting_satup" / "reference_clip.npz"
)


@pytest.fixture(scope="module")
def satup_clip() -> dict:
    return load_clip(FIXTURE_CLIP)


# ── crop_span: real fixture ─────────────────────────────────────────────
def test_crop_span_torso_righting_sub_span(satup_clip):
    """The exact D23/D24 sub-span: [0, 8.5] s stays low + rights the torso
    without ever reaching the clip's own standing end."""
    cropped = crop_span(satup_clip, 0.0, 8.5)
    assert validate_clip(cropped) == []
    assert _archetype(cropped) == "getup"
    assert float(np.max(cropped["root_pos_z"])) < 0.30

    n = cropped["root_pos_z"].shape[0]
    fps = float(cropped["fps"])
    assert n / fps == pytest.approx(8.5, abs=1e-6)

    sig = kinematic_signature(cropped)
    timeline = sig["orientation"]["timeline"]
    g_z_change = abs(timeline[-1]["g_z"] - timeline[0]["g_z"])
    assert g_z_change >= 0.15


def test_cropped_span_and_full_clip_agree_on_eval_reset(satup_clip):
    """§D29-1 (docs/internal/REFERENCE_BUILD_LOG.md) supersedes this
    test's original D24 F1 claim. PRE-§D29-1, `derive_reference_reset`'s
    reset_pitch_offset_rad/roll were computed as `start_window -
    end_window` — the clip's own END window leaked into the derivation,
    so a [0, 8.5]s span (whose end is still mid-recovery) measurably
    disagreed with the full clip (whose end is the true standing
    finish). §D29-1 replaced that cross-window subtraction with an
    analytic derivation FROM the START window's measured body-frame
    gravity alone (the D29 live disaster: subtracting a clip-relative
    "standing" reference produced an UPSIDE-DOWN prone reset) — the SAME
    fix D19 already made for `reset_height_offset_m` (anchored on the
    absolute G1-class standing constant, not the clip's own end). Both
    the full clip and the [0, 8.5]s span share the IDENTICAL start
    window (span cropping only moves the END), so every RSI/eval-reset
    key is now — correctly — BYTE-IDENTICAL between them; numerically
    confirmed on this fixture (both derive reset_pitch_offset_rad=
    -0.8825, reset_roll_offset_rad=0.037, reset_height_offset_m=-0.601).

    Phase-cropped SPANS still matter — for CERTIFICATION (a stage's
    metric must be certified against its own sub-phase, not the full
    clip's — D23/D24's actual motivating bug) and for the sunk-height
    termination guard (`min_base_height_termination_m`, `derive_rsi_
    train_keys`, anchored on `z.min()` over WHATEVER clip/span is
    passed in — see `test_crop_span_torso_righting_sub_span` above and
    `sculptor/refs/spans.py`/`mission_metrics.py` for the certification
    side) — RSI/eval-reset derivation specifically is just no longer one
    of the places that dependency shows up, now that BOTH its height and
    orientation channels are purely start-window-anchored."""
    from sculptor.reference import derive_eval_reset

    full_reset = derive_eval_reset(satup_clip)
    cropped = crop_span(satup_clip, 0.0, 8.5)
    span_reset = derive_eval_reset(cropped)

    assert full_reset is not None
    assert span_reset is not None
    assert "reset_pitch_offset_rad" in full_reset
    assert "reset_pitch_offset_rad" in span_reset
    assert full_reset == span_reset


@pytest.mark.parametrize("t_start,t_end", [
    (5.0, 2.0),      # inverted range
    (0.0, 100.0),    # beyond the clip's own duration
    (0.0, 0.02),     # crops to < 2 frames
])
def test_crop_span_rejects_bad_ranges(satup_clip, t_start, t_end):
    with pytest.raises(ValueError):
        crop_span(satup_clip, t_start, t_end)


def test_crop_span_feeds_perturbation_suite_without_error(satup_clip):
    """freeze/negative interaction sanity: a cropped clip is just another
    valid reference clip to the existing perturbation machinery."""
    cropped = crop_span(satup_clip, 0.0, 8.5)
    suite = perturbation_suite(cropped)
    assert set(suite) >= {
        "reversal", "freeze_start", "freeze_end", "shuffle",
        "trunc_25", "trunc_50", "trunc_75",
    }
    for name, variant in suite.items():
        assert validate_clip(variant) == [], f"{name} not validate_clip-clean"


def test_span_boundaries_always_includes_endpoints(satup_clip):
    boundaries = span_boundaries(satup_clip)
    assert boundaries[0] == 0.0
    n = satup_clip["root_pos_z"].shape[0]
    fps = float(satup_clip["fps"])
    # The exact max boundary may come from either the 6-decimal-rounded
    # duration_s or a 3-decimal-rounded phase-segment t_end that lands on
    # the same instant (both are stand-ins for "the clip's own end") —
    # tolerance covers that rounding-granularity gap, not a real drift.
    assert boundaries[-1] == pytest.approx(n / fps, abs=2e-3)
    assert boundaries == sorted(boundaries)
    assert len(boundaries) == len(set(boundaries))  # deduplicated


def test_span_qc_rejects_motionless_span(satup_clip):
    """A window inside the clip's long mid-clip 'still' plateau (per
    convert._phase_segments, ~1.2-3.1 s) has no z or orientation motion —
    the motion floor must reject it even though duration/bounds/QC-shape
    are all otherwise fine."""
    cropped = crop_span(satup_clip, 1.3, 3.0)
    reason = _span_qc(satup_clip, cropped, 1.3, 3.0, None)
    assert reason is not None
    assert reason.startswith("no_motion")


# ── kinematic_signature: orientation timeline (additive, quat-gated) ────
def _synthetic_getup_clip(T: int = 60, fps: float = 30.0, with_quat: bool = False) -> dict:
    """z: 0.1 flat -> ramp -> 0.75 flat (mirrors test_refs_convert.py's
    `_getup_like_clip`); optionally a root_quat_wxyz rotating from tipped
    (g_z ~ 0) to upright (g_z = -1) over the same duration."""
    n_flat0, n_ramp = T // 4, T // 2
    n_flat1 = T - n_flat0 - n_ramp
    flat0 = np.full(n_flat0, 0.1)
    s = np.linspace(0.0, 1.0, n_ramp)
    ramp = 0.1 + 0.65 * (1 - np.cos(np.pi * s)) / 2.0
    flat1 = np.full(n_flat1, 0.75)
    z = np.concatenate([flat0, ramp, flat1])
    clip: dict = {"root_pos_z": z, "fps": fps}
    if with_quat:
        theta = np.linspace(np.pi / 2, 0.0, T)  # tipped -> upright
        quat = np.zeros((T, 4))
        quat[:, 0] = np.cos(theta / 2.0)
        quat[:, 1] = np.sin(theta / 2.0)
        clip["root_quat_wxyz"] = quat
    return clip


def test_kinematic_signature_timeline_absent_without_quat():
    clip = _synthetic_getup_clip(with_quat=False)
    sig = kinematic_signature(clip)
    assert "orientation" not in sig


def test_kinematic_signature_timeline_present_with_quat():
    clip = _synthetic_getup_clip(with_quat=True)
    sig = kinematic_signature(clip)
    assert "timeline" in sig["orientation"]
    timeline = sig["orientation"]["timeline"]
    assert len(timeline) >= 2

    T = clip["root_pos_z"].shape[0]
    fps = float(clip["fps"])
    assert timeline[0]["t"] == 0.0
    # 3-decimal rounding in the timeline vs. an unrounded reference value.
    assert timeline[-1]["t"] == pytest.approx((T - 1) / fps, abs=1e-3)
    # First/last knots agree with the pre-existing initial/final scalars —
    # additive key, no change to existing values.
    assert timeline[0]["g_z"] == pytest.approx(
        sig["orientation"]["initial_gravity_z_b"], abs=1e-6)
    assert timeline[-1]["g_z"] == pytest.approx(
        sig["orientation"]["final_gravity_z_b"], abs=1e-6)
    json.dumps(sig)  # every value must stay JSON-serializable


# ── select_reference_span: mocked LLM (never touches the network) ──────
def _mock(text: str):
    return lambda prompt: text


#: The fixture's torso-righting sub-span [0, 8.5]s ends low (z ~0.14-0.16)
#: with the torso RIGHTED (g_z rights toward upright over the span) —
#: see test_crop_span_torso_righting_sub_span for the measured numbers.
#: Every mock payload below must carry a `expected_end` consistent with
#: whatever span it proposes (§F1 addendum).
_SATUP_EXPECTED_END = {"z_band": [0.10, 0.20], "g_z_more_upright": True}


def _good_payload(
    t_start=0.0, t_end=8.5, confidence=0.9, expected_end=None,
):
    return json.dumps({
        "t_start_s": t_start, "t_end_s": t_end, "confidence": confidence,
        "rationale": "sit-up phase: torso rights while pelvis stays low",
        "whole_clip": False,
        "expected_end": expected_end or _SATUP_EXPECTED_END,
    })


def test_select_reference_span_good_response_passes_qc(satup_clip):
    span, reason = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position",
        llm_call=_mock(_good_payload()))
    assert reason is None
    assert span is not None
    assert span["method"] == "llm+snap+qc"
    assert span["confidence"] == pytest.approx(0.9)
    assert 0.0 <= span["t_start_s"] < span["t_end_s"]
    assert span["t_end_s"] - span["t_start_s"] >= 1.0
    assert span["t_end_s"] <= 8.6  # snap only ever moves it onto a real boundary


def test_select_reference_span_low_confidence_rejected(satup_clip):
    span, reason = select_reference_span(
        satup_clip, goal_text="x", llm_call=_mock(_good_payload(confidence=0.3)))
    assert span is None
    assert reason is not None and reason.startswith("low_confidence")


def test_select_reference_span_whole_clip_rejected(satup_clip):
    n = satup_clip["root_pos_z"].shape[0]
    fps = float(satup_clip["fps"])
    payload = json.dumps({
        "t_start_s": 0.0, "t_end_s": n / fps, "confidence": 0.95,
        "rationale": "full get-up", "whole_clip": True,
        "expected_end": {"z_band": [0.70, 0.80], "g_z_more_upright": True},
    })
    span, reason = select_reference_span(
        satup_clip, goal_text="x", llm_call=_mock(payload))
    assert span is None
    assert reason == "whole_clip"


def test_select_reference_span_garbage_response_rejected(satup_clip):
    span, reason = select_reference_span(
        satup_clip, goal_text="x", llm_call=_mock("not json { at all"))
    assert span is None
    assert reason is not None and reason.startswith("parse_error")


# ── §F1 addendum: end-state self-consistency QC ─────────────────────────
def test_select_reference_span_missing_expected_end_is_parse_error(satup_clip):
    """`expected_end` is a required key (§F1 addendum) — an older-shaped
    response missing it must be treated exactly like any other malformed
    response: `parse_error`, never a silent default."""
    payload = json.dumps({
        "t_start_s": 0.0, "t_end_s": 8.5, "confidence": 0.9,
        "rationale": "sit-up phase", "whole_clip": False,
    })
    span, reason = select_reference_span(
        satup_clip, goal_text="x", llm_call=_mock(payload))
    assert span is None
    assert reason is not None and reason.startswith("parse_error")


def test_select_reference_span_end_state_consistent_accepted(satup_clip):
    """A span whose `expected_end` correctly describes its own measured
    end window (low z, rights toward upright) passes."""
    span, reason = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position",
        llm_call=_mock(_good_payload(
            expected_end={"z_band": [0.10, 0.20], "g_z_more_upright": True})))
    assert reason is None
    assert span is not None


def test_select_reference_span_end_state_wrong_z_band_rejected(satup_clip):
    """The correct [0, 8.5]s span, but a self-inconsistent `expected_end`
    claiming a STANDING z_band — must be rejected even though the span
    itself and its start-state are otherwise clean."""
    payload = _good_payload(
        t_start=0.0, t_end=8.5,
        expected_end={"z_band": [0.70, 0.80], "g_z_more_upright": True})
    span, reason = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position",
        llm_call=_mock(payload))
    assert span is None
    assert reason is not None and reason.startswith("qc_reject:end_state")


def test_select_reference_span_end_state_wrong_direction_rejected(satup_clip):
    """Correct z_band but a false directional claim (says the span does
    NOT become more upright when it measurably does) must also reject."""
    payload = _good_payload(
        t_start=0.0, t_end=8.5,
        expected_end={"z_band": [0.10, 0.20], "g_z_more_upright": False})
    span, reason = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position",
        llm_call=_mock(payload))
    assert span is None
    assert reason is not None and reason.startswith("qc_reject:end_state")


def test_select_reference_span_qc_rejects_over_extended_span(satup_clip):
    """§F1 addendum's motivating exploit: an OVER-EXTENDED span (0 ->
    11.1s, reaching the clip's own standing end per REFERENCE_BUILD_LOG.md
    D24_SPEC's measured numbers: z reaches ~0.78 by t=11.1s) whose
    `expected_end` still claims the true sit-up's LOW end state.
    `_span_qc`'s start-only checks would pass this identically to the
    correct [0, 8.5]s span (same start, same start_pose, real motion
    throughout) — only the end-state check catches it."""
    payload = _good_payload(
        t_start=0.0, t_end=11.1,
        expected_end={"z_band": [0.10, 0.20], "g_z_more_upright": True})
    span, reason = select_reference_span(
        satup_clip, goal_text="sit up", llm_call=_mock(payload))
    assert span is None
    assert reason is not None and reason.startswith("qc_reject:end_state")


def test_select_reference_span_qc_rejects_too_short_span(satup_clip):
    payload = _good_payload(t_start=0.0, t_end=0.5)
    span, reason = select_reference_span(
        satup_clip, goal_text="x", llm_call=_mock(payload))
    assert span is None
    assert reason is not None and reason.startswith("qc_reject:duration")


def test_select_reference_span_qc_rejects_start_pose_mismatch(satup_clip):
    """The fixture clip measures archetype='getup' (a lying/low start) — a
    stage authored with start_pose='standing' must be rejected even though
    the span itself is otherwise a clean sit-up sub-span."""
    span, reason = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position",
        start_pose="standing", llm_call=_mock(_good_payload()))
    assert span is None
    assert reason is not None
    assert reason.startswith("qc_reject:start_pose_mismatch")


def test_select_reference_span_never_raises_on_llm_failure(satup_clip):
    def boom(prompt):
        raise RuntimeError("network down")

    span, reason = select_reference_span(satup_clip, goal_text="x", llm_call=boom)
    assert span is None
    assert reason is not None and reason.startswith("llm_unavailable")


def test_select_reference_span_deterministic(satup_clip):
    call = _mock(_good_payload())
    span_a, reason_a = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position", llm_call=call)
    span_b, reason_b = select_reference_span(
        satup_clip, goal_text="right the torso to a seated position", llm_call=call)
    assert span_a == span_b
    assert reason_a == reason_b


def test_select_reference_span_empty_response_is_infra_not_semantic():
    """Live finding: fable-5 exhausted a 1024-token budget entirely on its
    thinking block and returned zero text blocks — the reason must be the
    diagnosable 'empty_response' (retryable infra), never a semantic
    decline that would persist a 'declined:' marker."""
    from sculptor.reference import load_clip
    from sculptor.refs.spans import is_semantic_decline, select_reference_span

    clip = load_clip(FIXTURE_CLIP)
    span, reason = select_reference_span(
        clip, goal_text="right the torso to sitting", llm_call=lambda p: "")
    assert span is None
    assert reason.startswith("empty_response:")
    assert is_semantic_decline(reason) is False


def _mock_span_call(t0, t1, band, up=True, conf=0.9):
    import json as _json
    return lambda p: _json.dumps({
        "t_start_s": t0, "t_end_s": t1, "confidence": conf,
        "rationale": "x", "whole_clip": False,
        "expected_end": {"z_band": band, "g_z_more_upright": up}})


def test_loose_z_band_rejected_as_uncommitted():
    """Opus audit H1 (PROVEN exploit): the end-state QC was vacuously
    satisfied by a wide z_band — the over-extended D23 span (0->11.2 s,
    ends standing) was ACCEPTED with z_band [0.05,0.80] or [0,1]. A band
    wider than max(0.15 m, 35% of the clip's z-range) is now rejected as
    an uncommitted claim; the honest tight band still accepts the honest
    span and still convicts the over-extended one."""
    from sculptor.reference import load_clip
    from sculptor.refs.spans import select_reference_span

    clip = load_clip(FIXTURE_CLIP)
    for band in ([0.05, 0.80], [0.0, 1.0], [0.10, 0.60]):
        span, reason = select_reference_span(
            clip, goal_text="sit up", llm_call=_mock_span_call(0.0, 11.2, band))
        assert span is None, band
        assert reason.startswith("qc_reject:end_state:"), reason
        assert "too wide" in reason
    # honest tight claim over the honest span: accepted
    span, reason = select_reference_span(
        clip, goal_text="sit up", llm_call=_mock_span_call(0.0, 8.4, [0.10, 0.20]))
    assert span is not None, reason
    # honest tight claim over the OVER-EXTENDED span: end-state mismatch
    span, reason = select_reference_span(
        clip, goal_text="sit up", llm_call=_mock_span_call(0.0, 11.2, [0.10, 0.20]))
    assert span is None
    assert "outside expected_end.z_band" in reason


def test_crop_span_clamps_rounded_end_boundary():
    """Opus audit M1 (PROVEN): _phase_segments rounds boundary times to
    3 decimals, so span_boundaries can emit an end time ~0.4 ms past the
    true duration (14.267 vs 14.2666...) — a legitimate reaches-the-end
    span then died with 'out of bounds' and fell back to the full clip.
    Rounding-scale overshoot now clamps; real overshoot still raises."""
    import pytest as _pytest
    from sculptor.reference import load_clip
    from sculptor.refs.spans import crop_span

    clip = load_clip(FIXTURE_CLIP)
    c = crop_span(clip, 3.0, 14.267)  # 0.4 ms past the true 14.2667 s end
    assert len(c["root_pos_z"]) == 338
    with _pytest.raises(ValueError):
        crop_span(clip, 0.0, 15.5)


def test_live_span_is_fast_completion_gated():
    """The accepted live span (0->8.1 s) must classify as reach-and-hold
    (ends_settled True) so reference_fast_completion is REQUIRED for it;
    clips still rising at their end (0->9.0, full clip) must not."""
    from sculptor.reference import load_clip
    from sculptor.refs.perturb import ends_settled
    from sculptor.refs.spans import crop_span

    clip = load_clip(FIXTURE_CLIP)
    assert ends_settled(crop_span(clip, 0.0, 8.1)) is True
    assert ends_settled(crop_span(clip, 0.0, 9.0)) is False
    assert ends_settled(clip) is False


def test_end_state_direction_skipped_when_no_headroom():
    """LIVE FALSE-REJECT (g1-standing-up drive_to_stand, 2026-07-12): a
    crouch->stand clip's PELVIS is near-upright for the whole motion
    (d13_crouch_to_ready: g_z -0.996 -> -0.995), so no honest answer to
    "does the span end more upright" can clear the direction threshold —
    the stage was pinned to a declined marker after one click. When the
    span STARTS near-upright (g_z <= -0.75) the direction question is
    skipped in either polarity; the z-band commitment still binds."""
    import numpy as np
    from sculptor.refs.spans import select_reference_span

    # Synthetic crouch->rise clip: pelvis upright throughout (identity
    # quat => g_z = -1), z rises 0.1 -> 0.4 then holds.
    n, fps = 120, 30.0
    z = np.concatenate([np.full(30, 0.10), np.linspace(0.10, 0.40, 60),
                        np.full(30, 0.40)])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    clip = {"root_pos_z": z, "fps": fps, "root_quat_wxyz": quat}

    def call(up):
        import json as _json
        return lambda p: _json.dumps({
            "t_start_s": 0.0, "t_end_s": 4.0, "confidence": 0.9,
            "rationale": "x", "whole_clip": False,
            "expected_end": {"z_band": [0.35, 0.45], "g_z_more_upright": up}})

    # BOTH polarities of the vacuous claim are accepted when the z-band
    # commitment holds — the direction question has no headroom here.
    for up in (True, False):
        span, reason = select_reference_span(
            clip, goal_text="drive up to ready", llm_call=call(up))
        assert span is not None, (up, reason)
    # ... and the z-band still convicts a wrong end-height claim.
    import json as _json
    bad = lambda p: _json.dumps({
        "t_start_s": 0.0, "t_end_s": 4.0, "confidence": 0.9,
        "rationale": "x", "whole_clip": False,
        "expected_end": {"z_band": [0.05, 0.15], "g_z_more_upright": True}})
    span, reason = select_reference_span(
        clip, goal_text="drive up to ready", llm_call=bad)
    assert span is None
    assert "outside expected_end.z_band" in reason


def test_snapped_end_boundary_past_duration_is_clamped_and_accepted():
    """LIVE D28 FINDING (second occurrence of the M1 rounding class): a
    span proposal snapped onto a phase boundary that _phase_segments had
    ROUNDED past the true clip duration (4.742 vs 4.741667 on the a10
    clip) was rejected by _span_qc's own unclamped bounds re-check even
    though crop_span clamps — and the persisted span fields would have
    carried the out-of-bounds number. The clamp now happens ONCE at the
    source (post-snap), so crop, QC, and persistence agree."""
    from sculptor.reference import load_clip
    from sculptor.refs.spans import select_reference_span

    clip = load_clip(FIXTURE_CLIP)
    duration = len(clip["root_pos_z"]) / float(clip["fps"])  # 14.26667
    # _phase_segments' final rounded boundary sits just PAST duration.
    span, reason = select_reference_span(
        clip, goal_text="drive up to standing and settle",
        llm_call=_mock_span_call(8.1, 14.267, [0.45, 0.60], up=True))
    assert span is not None, reason
    assert span["t_end_s"] <= duration + 1e-9


# ── Tier-D tracking reward: mjlab batched contract ─────────────────────
# MjlabAdapter REQUIRES compute_reward_batched; without it a Tier-D
# certification cannot train on the mjlab path at all, which is how the
# certification loop silently stopped working after mjlab's batched contract
# landed. These pin the entry point and its numerical agreement with the
# scalar path.
def _tierd_module(tmp_path):
    import importlib.util

    import numpy as np

    from sculptor.refs.track import generate_tracking_reward_source

    n_phase, n_joints = 8, 4
    rng = np.random.default_rng(0)
    src = generate_tracking_reward_source(
        clip_id="unit--g1",
        robot="g1",
        joint_names=[f"j{i}" for i in range(n_joints)],
        target_joint_pos=rng.normal(size=(n_phase, n_joints)),
        target_root_z=np.linspace(0.0, 0.05, n_phase),
        episode_len_steps=100,
    )
    path = tmp_path / "current.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tierd_rw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tierd_reward_declares_the_batched_contract(tmp_path):
    mod = _tierd_module(tmp_path)
    assert mod.REWARD_SPEC.get("supports_batched") is True
    assert hasattr(mod, "compute_reward_batched")
    assert hasattr(mod, "compute_reward")


def test_tierd_batched_scores_an_on_target_pose_at_one(tmp_path):
    torch = pytest.importorskip("torch")
    mod = _tierd_module(tmp_path)

    i = 3
    step = (i + 0.5) / mod.N_PHASE * mod.EPISODE_LEN_STEPS
    target = torch.as_tensor(mod.TARGET_JOINT_POS[i], dtype=torch.float64)
    qpos = target.repeat(4, 1)
    z0, tz = float(mod.TARGET_ROOT_Z[0]), float(mod.TARGET_ROOT_Z[i])
    info = {
        "episode_length": torch.full((4,), float(step), dtype=torch.float64),
        "base_height": torch.full((4,), z0 + (tz - z0), dtype=torch.float64),
    }
    total, comp = mod.compute_reward_batched(None, None, {"qpos": qpos}, info)
    assert float(comp["joint_tracking"][0]) == pytest.approx(
        mod.JOINT_TERM_COEFFICIENT, abs=1e-9)
    assert float(comp["root_tracking"][0]) == pytest.approx(
        mod.ROOT_TERM_COEFFICIENT, abs=1e-9)
    assert float(total[0]) == pytest.approx(mod.REWARD_TOTAL_SCALE, abs=1e-9)


def test_tierd_batched_joint_kernel_matches_the_scalar_path(tmp_path):
    """Same two-Gaussian formula, two state layouts. A divergence here means
    a Tier-D verdict computed during training disagrees with the one the
    scalar contract documents."""
    import numpy as np

    torch = pytest.importorskip("torch")
    mod = _tierd_module(tmp_path)

    i, off = 3, 0.3
    step = (i + 0.5) / mod.N_PHASE * mod.EPISODE_LEN_STEPS
    target = mod.TARGET_JOINT_POS[i]

    qpos = torch.as_tensor(target + off, dtype=torch.float64).repeat(4, 1)
    info = {
        "episode_length": torch.full((4,), float(step), dtype=torch.float64),
        "base_height": torch.zeros(4, dtype=torch.float64),
    }
    _, batched = mod.compute_reward_batched(None, None, {"qpos": qpos}, info)

    scalar_qpos = np.zeros(7 + mod.N_JOINTS)
    scalar_qpos[2] = float(mod.TARGET_ROOT_Z[i])
    scalar_qpos[7:] = target + off
    _, scalar = mod.compute_reward(
        None, None, {"qpos": scalar_qpos}, {"episode_length": int(step)})

    assert float(batched["joint_tracking"][0]) == pytest.approx(
        scalar["joint_tracking"], abs=1e-12)
    expected_kernel = float(np.exp(-mod.JOINT_ERR_WEIGHT * off ** 2))
    assert float(batched["joint_tracking"][0]) == pytest.approx(
        mod.JOINT_TERM_COEFFICIENT * expected_kernel, abs=1e-12)


def test_tierd_batched_root_uses_a_delta_not_an_absolute_height(tmp_path):
    """Origin-relative retargeted clips (most of the library) carry root_z
    near 0 while a standing G1 base is ~0.74 m. An absolute comparison would
    saturate the kernel at zero for every frame however well it tracked."""
    torch = pytest.importorskip("torch")
    mod = _tierd_module(tmp_path)

    i = 4
    step = (i + 0.5) / mod.N_PHASE * mod.EPISODE_LEN_STEPS
    qpos = torch.as_tensor(mod.TARGET_JOINT_POS[i], dtype=torch.float64).repeat(2, 1)
    z0, tz = float(mod.TARGET_ROOT_Z[0]), float(mod.TARGET_ROOT_Z[i])
    # A base sitting 0.74 m up but tracking the reference's DELTA perfectly.
    info = {
        "episode_length": torch.full((2,), float(step), dtype=torch.float64),
        "base_height_delta": torch.full((2,), tz - z0, dtype=torch.float64),
    }
    _, comp = mod.compute_reward_batched(None, None, {"qpos": qpos}, info)
    assert float(comp["root_tracking"][0]) == pytest.approx(
        mod.ROOT_TERM_COEFFICIENT, abs=1e-9)
