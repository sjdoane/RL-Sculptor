"""tests/test_refs_synth.py — D28 F-SYNTH: `sculptor.refs.synth`
(the last-resort synthetic-exemplar certification tier).

No network: `synthesize_reference_clip`'s `llm_call` is ALWAYS mocked
here (mirrors `tests/test_reference_spans.py`'s conventions for
`select_reference_span`). Covers:

  * strict-parse / mechanical-QC rejections (bad times, absurd z, low
    confidence, wide end band, start-pose archetype mismatch, keyframe
    count, duration bounds, mixed g_z);
  * the mechanical synthesis itself — cosine-eased interpolation
    smoothness (no velocity spikes), the g_z -> pitch -> quat -> g_z
    round-trip, determinism;
  * commitments honored/violated (uprightness direction, no-headroom
    skip, z-band checks);
  * infra failure paths (empty response, LLM unavailable, garbage
    response);
  * an END-TO-END integration proving the EXISTING six-gate reference
    battery (`sculptor.eval.metric_validate`, unmodified) certifies an
    honest metric against a synthetic clip exactly as it would a real
    reference — reusing `tests/test_reference_anchored_validation.py`'s
    harness pattern, grounded on the REAL a10-family fixture signature
    at `tests/fixtures/torso_righting_satup/reference_clip.npz` as an
    analogous-signature donor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.refs.synth import (
    _cosine_ease_interp,
    _pitch_sign,
    synthesize_reference_clip,
)

FIXTURE_CLIP = (
    Path(__file__).parent / "fixtures" / "torso_righting_satup" / "reference_clip.npz"
)


def _mock(payload: dict):
    return lambda prompt: json.dumps(payload)


def _good_payload(**overrides) -> dict:
    base = {
        "keyframes": [
            {"t": 0.0, "root_z": 0.15, "g_z": -0.6},
            {"t": 1.0, "root_z": 0.20, "g_z": -0.7},
            {"t": 2.5, "root_z": 0.45, "g_z": -0.85},
            {"t": 4.0, "root_z": 0.74, "g_z": -0.98},
        ],
        "duration_s": 4.0,
        "rationale": "crouch to stand sketch",
        "confidence": 0.8,
        "commitments": {
            "start_z_band": [0.10, 0.20],
            "end_z_band": [0.68, 0.78],
            "end_more_upright_than_start": True,
        },
    }
    base.update(overrides)
    return base


# ── happy path ────────────────────────────────────────────────────────────
def test_good_sketch_synthesizes_a_valid_clip():
    from sculptor.reference import validate_clip

    clip, reason = synthesize_reference_clip(
        "stand up from a crouch", start_pose="crouched",
        llm_call=_mock(_good_payload()))
    assert reason is None
    assert clip is not None
    assert validate_clip(clip) == []
    assert clip["fps"] == pytest.approx(30.0)
    assert clip["root_quat_wxyz"].shape == (clip["root_pos_z"].shape[0], 4)
    assert clip["meta"]["source"] == "synthetic"
    assert clip["meta"]["confidence"] == pytest.approx(0.8)
    assert clip["meta"]["start_pose"] == "crouched"


def test_all_null_gz_yields_no_orientation_channel():
    payload = _good_payload()
    for kf in payload["keyframes"]:
        kf["g_z"] = None
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert reason is None
    assert clip is not None
    assert "root_quat_wxyz" not in clip


def test_deterministic_given_same_inputs():
    call = _mock(_good_payload())
    clip_a, reason_a = synthesize_reference_clip(
        "stand up from a crouch", start_pose="crouched", llm_call=call)
    clip_b, reason_b = synthesize_reference_clip(
        "stand up from a crouch", start_pose="crouched", llm_call=call)
    assert reason_a is None and reason_b is None
    np.testing.assert_array_equal(clip_a["root_pos_z"], clip_b["root_pos_z"])
    np.testing.assert_array_equal(
        clip_a["root_quat_wxyz"], clip_b["root_quat_wxyz"])
    assert clip_a["meta"] == clip_b["meta"]


# ── infra failure paths ──────────────────────────────────────────────────
def test_never_raises_on_llm_failure():
    def boom(prompt):
        raise RuntimeError("network down")

    clip, reason = synthesize_reference_clip("x", llm_call=boom)
    assert clip is None
    assert reason is not None and reason.startswith("llm_unavailable")


def test_empty_response_is_infra_not_semantic():
    clip, reason = synthesize_reference_clip("x", llm_call=lambda p: "")
    assert clip is None
    assert reason == "empty_response:llm returned no text blocks"


def test_garbage_response_is_parse_error():
    clip, reason = synthesize_reference_clip(
        "x", llm_call=lambda p: "not json { at all")
    assert clip is None
    assert reason is not None and reason.startswith("parse_error")


# ── strict-parse rejections (schema shape) ───────────────────────────────
def test_non_increasing_times_rejected():
    payload = _good_payload()
    payload["keyframes"][2]["t"] = 0.5  # out of order
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")
    assert "strictly increasing" in reason


def test_first_keyframe_must_start_at_zero():
    payload = _good_payload()
    payload["keyframes"][0]["t"] = 0.2
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")


def test_last_keyframe_must_equal_duration():
    payload = _good_payload(duration_s=5.0)  # last kf t is still 4.0
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")


@pytest.mark.parametrize("n", [3, 13])
def test_keyframe_count_out_of_bounds_rejected(n):
    kfs = [{"t": float(i), "root_z": 0.1 + 0.02 * i, "g_z": None} for i in range(n)]
    payload = {
        "keyframes": kfs, "duration_s": float(n - 1), "rationale": "x",
        "confidence": 0.8,
        "commitments": {
            "start_z_band": [0.05, 0.2], "end_z_band": [0.2, 0.4],
            "end_more_upright_than_start": False,
        },
    }
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")
    assert "keyframes length" in reason


def test_mixed_gz_null_and_numeric_rejected():
    payload = _good_payload()
    payload["keyframes"][1]["g_z"] = None
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")
    assert "never a mix" in reason


def test_missing_commitments_key_is_parse_error():
    payload = _good_payload()
    del payload["commitments"]["end_more_upright_than_start"]
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("parse_error")


# ── mechanical QC rejections ──────────────────────────────────────────────
def test_low_confidence_rejected():
    payload = _good_payload(confidence=0.3)
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason == "low_confidence:0.30"


@pytest.mark.parametrize("duration_s,last_t", [(1.0, 1.0), (35.0, 35.0)])
def test_duration_out_of_range_rejected(duration_s, last_t):
    payload = _good_payload(duration_s=duration_s)
    n = len(payload["keyframes"])
    for i, kf in enumerate(payload["keyframes"]):
        kf["t"] = last_t * i / (n - 1)
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:duration_out_of_range")


def test_absurd_root_z_rejected():
    payload = _good_payload()
    payload["keyframes"][-1]["root_z"] = 5.0
    payload["commitments"]["end_z_band"] = [4.9, 5.0]
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:root_z_out_of_range")


def test_wide_end_band_rejected():
    """H1 lesson (spans.py's _end_state_qc): an uncommitted (too wide)
    end_z_band is a rejection, not a free pass."""
    payload = _good_payload()
    payload["commitments"]["end_z_band"] = [0.05, 0.78]
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:commitments:end_z_band_too_wide")


def test_start_pose_archetype_mismatch_rejected():
    """A sketch whose OWN root_z keyframes never dip into a lying/low
    start (archetype != getup/mid_start-for-sit-crouch) contradicts a
    claimed supine/prone/sitting/crouched start_pose — independent of
    orientation (the quaternion's hemisphere is derived FROM start_pose
    in this module, so the orientation half of the QC check is
    self-consistent by construction; the ARCHETYPE half is the
    genuinely independent signal this synthesis can violate)."""
    payload = {
        "keyframes": [
            {"t": 0.0, "root_z": 0.50, "g_z": None},
            {"t": 1.0, "root_z": 0.55, "g_z": None},
            {"t": 2.5, "root_z": 0.65, "g_z": None},
            {"t": 4.0, "root_z": 0.74, "g_z": None},
        ],
        "duration_s": 4.0,
        "rationale": "x",
        "confidence": 0.8,
        "commitments": {
            "start_z_band": [0.45, 0.55], "end_z_band": [0.68, 0.78],
            "end_more_upright_than_start": False,
        },
    }
    clip, reason = synthesize_reference_clip(
        "x", start_pose="supine", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:start_pose_mismatch")
    assert "does not measure as a lying start" in reason


def test_start_z_band_mismatch_rejected():
    payload = _good_payload()
    payload["commitments"]["start_z_band"] = [0.60, 0.70]  # sketch actually starts ~0.15
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:commitments:start_z_band_mismatch")


def test_uprightness_claim_violated_rejected():
    """The sketch DOES become more upright (g_z -0.6 -> -0.98) but the
    commitment claims it does NOT — must reject."""
    payload = _good_payload()
    payload["commitments"]["end_more_upright_than_start"] = False
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert clip is None
    assert reason.startswith("qc_reject:commitments:uprightness_mismatch")


def test_uprightness_claim_honored_accepted():
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(_good_payload()))
    assert reason is None
    assert clip is not None


def test_no_headroom_skips_direction_check():
    """A sketch that STARTS already near-upright (g_z <= -0.75) has no
    headroom for "does it end more upright" — the direction claim is
    skipped entirely (never rejects), mirroring spans.py's no-headroom
    guard."""
    payload = _good_payload()
    for kf in payload["keyframes"]:
        kf["g_z"] = -0.9  # already upright throughout
    payload["commitments"]["end_more_upright_than_start"] = True  # unverifiable claim
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert reason is None
    assert clip is not None


# ── interpolation smoothness / quat round-trip ───────────────────────────
def test_interpolation_smoothness_bounded():
    """Cosine-eased interpolation has ZERO derivative at every keyframe
    boundary (no velocity spikes, D12 hygiene) — the PEAK |dz/dt| over a
    segment is exactly `(pi/2) * naive_avg_rate` for a single monotone
    segment, never larger."""
    payload = {
        "keyframes": [
            {"t": 0.0, "root_z": 0.15, "g_z": None},
            {"t": 0.5, "root_z": 0.15, "g_z": None},
            {"t": 2.0, "root_z": 0.74, "g_z": None},
            {"t": 3.0, "root_z": 0.74, "g_z": None},
        ],
        "duration_s": 3.0,
        "rationale": "quick stand",
        "confidence": 0.9,
        "commitments": {
            "start_z_band": [0.10, 0.20], "end_z_band": [0.68, 0.78],
            "end_more_upright_than_start": False,
        },
    }
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert reason is None
    vz = clip["root_vel_z"]
    assert np.isfinite(vz).all()
    naive_rate = (0.74 - 0.15) / 1.5
    peak = float(np.max(np.abs(vz)))
    # Cosine-ease peak derivative is (pi/2) * naive average rate.
    assert peak == pytest.approx(np.pi / 2 * naive_rate, rel=0.05)
    # And boundary frames (keyframe times) sit near zero relative slope —
    # nowhere close to the peak.
    fps = float(clip["fps"])
    assert abs(vz[0]) < 0.05 * peak
    assert abs(vz[-1]) < 0.05 * peak


def test_cosine_ease_interp_exact_at_keyframe_times():
    t_kf = np.array([0.0, 1.0, 2.5, 4.0])
    v_kf = np.array([0.1, 0.3, 0.6, 0.8])
    out = _cosine_ease_interp(t_kf, t_kf, v_kf)
    np.testing.assert_allclose(out, v_kf, atol=1e-12)


def test_quat_gz_round_trip_via_kinematic_signature():
    """The first/last frame ALWAYS lands exactly on the first/last
    keyframe time (`t_frames[0]=0=t_kf[0]`, `t_frames[-1]=duration_s=
    t_kf[-1]`), so the synthesized clip's own start/end g_z (read via
    `kinematic_signature`'s orientation timeline) must reproduce the
    sketch's g_z keyframes exactly."""
    from sculptor.refs.convert import kinematic_signature

    payload = _good_payload()
    clip, reason = synthesize_reference_clip("x", llm_call=_mock(payload))
    assert reason is None
    sig = kinematic_signature(clip)
    timeline = sig["orientation"]["timeline"]
    assert timeline[0]["g_z"] == pytest.approx(
        payload["keyframes"][0]["g_z"], abs=1e-3)
    assert timeline[-1]["g_z"] == pytest.approx(
        payload["keyframes"][-1]["g_z"], abs=1e-3)
    assert sig["root_z"]["start"] == pytest.approx(
        payload["keyframes"][0]["root_z"], abs=1e-3)
    assert sig["root_z"]["end"] == pytest.approx(
        payload["keyframes"][-1]["root_z"], abs=1e-3)


def test_pitch_sign_supine_is_negative_others_positive():
    assert _pitch_sign("supine") == -1.0
    assert _pitch_sign("prone") == 1.0
    assert _pitch_sign("crouched") == 1.0
    assert _pitch_sign(None) == 1.0


# ── analogous signatures grounding ────────────────────────────────────────
def test_analogous_signatures_are_threaded_into_the_prompt():
    from sculptor.reference import load_clip
    from sculptor.refs.convert import kinematic_signature

    donor = load_clip(FIXTURE_CLIP)
    donor_sig = kinematic_signature(donor)

    captured: dict = {}

    def capturing_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps(_good_payload())

    clip, reason = synthesize_reference_clip(
        "x", analogous_signatures=[donor_sig], llm_call=capturing_call)
    assert reason is None
    assert "ANALOGOUS CLIP SIGNATURES" in captured["prompt"]
    assert str(donor_sig["root_z"]["start"]) in captured["prompt"]


def test_no_analogous_signatures_notes_robot_constants_only():
    captured: dict = {}

    def capturing_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps(_good_payload())

    clip, reason = synthesize_reference_clip("x", llm_call=capturing_call)
    assert reason is None
    assert "none available" in captured["prompt"]


# ── integration: the UNCHANGED six-gate reference battery on a synthetic
#    exemplar grounded on a REAL fixture signature ────────────────────────
def test_synthetic_clip_passes_the_six_gate_battery_with_an_honest_metric():
    """`sculptor.eval.metric_validate.validate_generated_metric` (read-only
    per the D28 spec — not modified by this feature) must certify an
    honest metric against a synthetic clip exactly as it would a real
    reference. Grounded on the REAL fixture's kinematic signature as an
    analogous-signature donor (a mismatched clip's numbers are still real
    physical data for this robot, per the D28 spec)."""
    from sculptor.eval.metric_validate import validate_generated_metric
    from sculptor.refs.convert import kinematic_signature
    from sculptor.refs.perturb import perturbation_suite
    from sculptor.reference import derive_eval_reset, load_clip, validate_clip

    donor = load_clip(FIXTURE_CLIP)
    donor_sig = kinematic_signature(donor)

    payload = {
        "keyframes": [
            {"t": 0.0, "root_z": 0.12, "g_z": -0.05},
            {"t": 1.0, "root_z": 0.15, "g_z": -0.2},
            {"t": 3.0, "root_z": 0.30, "g_z": -0.6},
            {"t": 4.5, "root_z": 0.45, "g_z": -0.85},
            {"t": 6.0, "root_z": 0.45, "g_z": -0.85},
        ],
        "duration_s": 6.0,
        "rationale": "prone push-up to kneel",
        "confidence": 0.85,
        "commitments": {
            "start_z_band": [0.08, 0.18], "end_z_band": [0.40, 0.50],
            "end_more_upright_than_start": True,
        },
    }
    clip, reason = synthesize_reference_clip(
        "push up from prone to a kneeling position",
        analogous_signatures=[donor_sig],
        llm_call=_mock(payload))
    assert reason is None, reason
    assert validate_clip(clip) == []

    # The six-gate battery must run UNCHANGED on a jointless synthetic
    # clip — the perturbation suite itself must build cleanly.
    suite = perturbation_suite(clip)
    assert set(suite) >= {
        "reversal", "freeze_start", "freeze_end", "shuffle", "root_only",
        "trunc_25", "trunc_50", "trunc_75",
    }
    for name, variant in suite.items():
        assert validate_clip(variant) == [], f"{name} not validate_clip-clean"

    eval_reset = derive_eval_reset(clip)
    assert eval_reset is not None  # a low ("getup") start archetype
    eval_reset_preview = {"scalars": eval_reset, "settled": True, "reason": None}

    honest_metric = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = min(15, n)
    rise = np.clip(z[-q:].mean() - z[:min(5, n)].min(), 0.0, None)
    gz = g[..., 2] / np.maximum(np.linalg.norm(g, axis=-1), 1e-9)
    upright_end = np.clip((-gz[-q:].mean() - 0.3) / 0.6, 0.0, 1.0)
    val = float(np.clip(rise / 0.33, 0.0, 1.0) * upright_end)
    return {"spec_score": val}
'''
    p = Path(__file__).parent / "_synth_integration_metric_tmp.py"
    try:
        p.write_text(honest_metric, encoding="utf-8")
        v = validate_generated_metric(
            honest_metric, p,
            references=[("synthetic:push_up_to_kneel", clip)],
            eval_reset=eval_reset_preview)
        assert v["ok"] is True, v["reasons"]
        gates = v["gates"]
        for gate_name in (
            "reference_nondegeneracy", "reference_monotonicity",
            "reference_negatives", "reference_complete_then_hold",
            "reference_fast_completion", "reference_settled_start",
        ):
            key = f"{gate_name}:synthetic:push_up_to_kneel"
            assert gates.get(key) is True, (gate_name, v["reasons"])
    finally:
        p.unlink(missing_ok=True)
