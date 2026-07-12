"""§REFERENCE_TRAJECTORY_PLAN §5: reference-anchored validation gates
wired into `sculptor.eval.metric_validate.validate_generated_metric` via
the `references` keyword. GPU-free — metrics written to temp .py files,
exercised through the REAL validation pipeline (mirrors
tests/test_generated_metric.py's `_write` + `validate_generated_metric`
pattern)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.metric_validate import (
    _score_reference_entry,
    validate_generated_metric,
)

# A synthetic "get-up"-shaped reference: root_z rises 0.1 -> 0.75 m over
# the middle half of the clip, flat before and after (a stylised get-up).
_T = 80
_FPS = 40.0


def _rising_clip() -> dict:
    n_flat0 = _T // 4
    n_ramp = _T // 2
    n_flat1 = _T - n_flat0 - n_ramp
    flat0 = np.full(n_flat0, 0.1)
    s = np.linspace(0.0, 1.0, n_ramp)
    ramp = 0.1 + 0.65 * (1 - np.cos(np.pi * s)) / 2.0
    flat1 = np.full(n_flat1, 0.75)
    z = np.concatenate([flat0, ramp, flat1])
    return {"root_pos_z": z, "fps": _FPS}


# An HONEST metric: rewards end-window height ONLY when it actually rose
# from a lower start (end*rise, not end alone) — end alone would be gamed
# by "hold the end pose the whole time" (freeze_end), which the
# reference_negatives gate is specifically built to catch. Windows are
# FIXED-SIZE (hold-invariant): fraction-of-episode windows violate the
# RELATIVE-TIME authoring rule and die on complete_then_hold_x4 — that
# defect is pinned separately by FRACTION_WINDOW_METRIC below.
HONEST_GETUP = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    w = min(15, n)
    first_w = z[:w].mean()
    final_w = z[-w:].mean()
    rise = np.clip(final_w - first_w, 0.0, None)
    val = float(np.clip(final_w * rise / 0.75, 0.0, 1.0))
    return {"spec_score": val}
'''

# The pre-D24 "honest" fixture, kept verbatim as a PINNED EXPLOIT: its
# fraction-of-episode windows (first/last quarter of n) read held terminal
# frames into the START window once a long hold is appended, so it narrowly
# clears complete_then_hold at x1 but collapses at x4 — the exact D23 live
# shape (brief transition, long hold). Must FAIL the gate.
FRACTION_WINDOW_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    first_q = z[:max(1, int(0.25 * n))].mean()
    final_q = z[int(0.75 * n):].mean()
    rise = np.clip(final_q - first_q, 0.0, None)
    val = float(np.clip(final_q * rise / 0.75, 0.0, 1.0))
    return {"spec_score": val}
'''

# A DEGENERATE metric: constant output regardless of input.
CONSTANT_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''

# A REVERSED-reward metric: rewards HIGH z at the START and LOW z at the
# END — literally scores the reversed/falling motion, not the get-up.
REVERSED_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    first_q = z[:max(1, int(0.25 * n))].mean()
    final_q = z[int(0.75 * n):].mean()
    fall = np.clip(first_q - final_q, 0.0, None)
    val = float(np.clip(first_q * fall / 0.75, 0.0, 1.0))
    return {"spec_score": val}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_honest_metric_clears_all_four_reference_gates(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_nondegeneracy:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_monotonicity:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_negatives:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_complete_then_hold:getup1"] is True, v["reasons"]
    assert v["ok"] is True, v["reasons"]

    ref_result = v["references"][0]
    assert ref_result["clip_id"] == "getup1"
    assert ref_result["gates"] == {
        "reference_nondegeneracy": True,
        "reference_monotonicity": True,
        "reference_negatives": True,
        "reference_complete_then_hold": True,
    }
    # speed_slow/speed_fast are scored + recorded but not gate keys.
    assert "speed_slow" in ref_result["scores"]
    assert "speed_fast" in ref_result["scores"]
    assert "complete_then_hold" in ref_result["scores"]
    assert f"reference:getup1" in v["archetype_scores"]
    assert f"reference:getup1:trunc_50" in v["archetype_scores"]
    assert f"reference:getup1:complete_then_hold" in v["archetype_scores"]


def _late_rising_clip() -> dict:
    """Lies STILL for the first half, then rises — the real-mocap pacing
    (fallAndGetUp segments) that broke the original strict-monotone gate:
    an honest righting metric scores trunc_25 == trunc_50 == 0.0 (a
    plateau, not an inversion). D8 in REFERENCE_BUILD_LOG."""
    n_flat0 = _T // 2
    n_ramp = _T // 4
    n_flat1 = _T - n_flat0 - n_ramp
    flat0 = np.full(n_flat0, 0.1)
    s = np.linspace(0.0, 1.0, n_ramp)
    ramp = 0.1 + 0.65 * (1 - np.cos(np.pi * s)) / 2.0
    flat1 = np.full(n_flat1, 0.75)
    z = np.concatenate([flat0, ramp, flat1])
    return {"root_pos_z": z, "fps": _FPS}


def test_honest_metric_passes_on_late_rising_clip_plateau(tmp_path):
    # Regression pin for D8: uneven mocap pacing (long lying prefix)
    # must NOT fail monotonicity — plateaus at zero are honest.
    clip = _late_rising_clip()
    p = _write(tmp_path, "honest_late.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup_late", clip)])
    assert v["gates"]["reference_monotonicity:getup_late"] is True, v["reasons"]
    assert v["ok"] is True, v["reasons"]
    sc = v["references"][0]["scores"]
    # The pacing really does plateau: both early truncations tie.
    assert sc["trunc_25"] == sc["trunc_50"]
    # And full still discriminates clearly against the earliest prefix.
    assert sc["full"] >= sc["trunc_25"] + 0.1


# The Opus-audit gaming metric (D18): rewards time-height correlation of
# the BASE only — steady pelvis rise, zero dependence on posture. Before
# the root_only negative existed this passed ALL reference gates (proven
# live): a policy levitating its pelvis while staying supine earned 1.0.
PELVIS_RISE_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    t = np.linspace(0.0, 1.0, n).reshape(n, 1)
    zc = z - z.mean(axis=0, keepdims=True)
    tc = t - 0.5
    denom = np.sqrt((zc ** 2).sum(axis=0) * (tc ** 2).sum(axis=0)) + 1e-9
    corr = (zc * tc).sum(axis=0) / denom
    rise = np.clip(z[-max(1, n // 10):].mean(axis=0) - z[:max(1, n // 10)].mean(axis=0), 0.0, None)
    val = float(np.clip(corr.mean(), 0.0, 1.0) * np.clip(rise.mean() / 0.3, 0.0, 1.0))
    return {"spec_score": val}
'''


def _clip_with_posture() -> dict:
    """Rising clip that ALSO carries orientation + joints, so root_only
    has posture channels to freeze."""
    clip = _rising_clip()
    n = clip["root_pos_z"].shape[0]
    t = np.linspace(0.0, 1.0, n)
    # supine (pitch -pi/2-ish) -> upright: quat about y from -1.5 to 0
    ang = -1.5 * (1.0 - t)
    quat = np.stack([np.cos(ang / 2), np.zeros(n), np.sin(ang / 2),
                     np.zeros(n)], axis=1)
    clip["root_quat_wxyz"] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    clip["joint_pos"] = np.stack([np.linspace(1.2, 0.0, n),
                                  np.linspace(-0.8, 0.1, n)], axis=1)
    clip["joint_names"] = ["a", "b"]
    return clip


def test_pelvis_rise_gaming_metric_fails_root_only_negative(tmp_path):
    # D18 regression pin: displacement-only metrics die on root_only.
    clip = _clip_with_posture()
    p = _write(tmp_path, "pelvis.py", PELVIS_RISE_METRIC)
    v = validate_generated_metric(
        PELVIS_RISE_METRIC, p, references=[("getup1", clip)])
    assert v["ok"] is False
    assert v["gates"]["reference_negatives:getup1"] is False, v["reasons"]
    sc = v["references"][0]["scores"]
    # root_only carries the same root trajectory -> the gamer scores it
    # like the real thing; that similarity is exactly what convicts it.
    assert sc["root_only"] > 0.5


# Posture-AWARE honest metric: height rise x ends-upright (body-frame
# gravity-z near -1 at the end). This is what D18 forces get-up metrics
# to be — a height-only metric (even an "honest" one) is now correctly
# convicted by root_only on posture-carrying clips.
HONEST_GETUP_POSTURE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = min(15, n)
    rise = np.clip(z[-q:].mean() - z[:q].mean(), 0.0, None)
    gz = g[..., 2] / np.maximum(np.linalg.norm(g, axis=-1), 1e-9)
    upright_end = np.clip((-gz[-q:].mean() - 0.3) / 0.6, 0.0, 1.0)
    val = float(np.clip(rise / 0.4, 0.0, 1.0) * upright_end)
    return {"spec_score": val}
'''


def test_posture_aware_honest_metric_passes_with_root_only(tmp_path):
    clip = _clip_with_posture()
    p = _write(tmp_path, "honest_posture.py", HONEST_GETUP_POSTURE)
    v = validate_generated_metric(
        HONEST_GETUP_POSTURE, p, references=[("getup1", clip)])
    assert v["ok"] is True, v["reasons"]
    sc = v["references"][0]["scores"]
    # root_only holds the SUPINE start orientation while the root rises:
    # the posture-aware metric sees no uprightness and scores it low.
    assert sc["root_only"] < sc["full"] - 0.3


def test_height_only_metric_convicted_on_posture_clip(tmp_path):
    # The old "honest" height-only fixture is now correctly rejected on
    # posture-carrying clips — get-up metrics MUST read orientation.
    clip = _clip_with_posture()
    p = _write(tmp_path, "height_only.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_negatives:getup1"] is False


def test_posture_less_clip_has_no_root_only_entry(tmp_path):
    # For root-channels-only clips root_only would equal the original —
    # it must be absent, not a universal conviction.
    clip = _rising_clip()
    p = _write(tmp_path, "honest3.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert "root_only" not in v["references"][0]["scores"]
    assert v["ok"] is True, v["reasons"]
    p = _write(tmp_path, "const.py", CONSTANT_METRIC)
    v = validate_generated_metric(
        CONSTANT_METRIC, p, references=[("getup1", clip)])
    assert v["gates"]["reference_nondegeneracy:getup1"] is False
    assert v["ok"] is False
    assert any("nondegeneracy" in r and "getup1" in r for r in v["reasons"])


def test_reversed_reward_metric_fails_monotonicity_and_negatives(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "rev.py", REVERSED_METRIC)
    v = validate_generated_metric(
        REVERSED_METRIC, p, references=[("getup1", clip)])
    assert v["gates"]["reference_monotonicity:getup1"] is False
    assert v["gates"]["reference_negatives:getup1"] is False
    assert v["ok"] is False


def test_no_references_leaves_result_shape_and_behavior_unchanged(tmp_path):
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(HONEST_GETUP, p)
    assert v["references"] == []
    assert not any(k.startswith("reference_") for k in v["gates"])
    assert not any(k.startswith("reference:") for k in v["archetype_scores"])


def test_multiple_references_each_get_their_own_gates(tmp_path):
    clip_a = _rising_clip()
    # A second reference: same shape but scaled to a lower final height —
    # still a legitimate rise, independent gate outcomes expected.
    z_b = _rising_clip()["root_pos_z"] * 0.9
    clip_b = {"root_pos_z": z_b, "fps": _FPS}
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip_a), ("getup2", clip_b)])
    assert "reference_nondegeneracy:getup1" in v["gates"]
    assert "reference_nondegeneracy:getup2" in v["gates"]
    ids = {r["clip_id"] for r in v["references"]}
    assert ids == {"getup1", "getup2"}


def test_reference_replaces_vacuous_probe_path_for_family_none_goal(tmp_path):
    """The synthetic battery is uninformative for a get-up-shaped goal (every
    archetype sits near a fixed standing/fallen height — no rise). Without a
    reference this would fall to the goal-agnostic selectivity probe (which
    can't see a pure root-height metric, since its probes don't move
    root_link_pos_w in this pattern); WITH a reference attached, the honest
    metric must be certified via the reference gates instead."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v_with_ref = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v_with_ref["nondegeneracy_vacuous"] is False
    assert v_with_ref["gates"]["nondegeneracy"] is True
    assert any("deferred to attached reference" in r for r in v_with_ref["reasons"])


def test_metric_crashing_on_a_reference_fails_the_existing_bounded_gate(tmp_path):
    """A metric that raises while scoring a reference is caught by
    `_score_reference_entry`'s own except-clause (nan = no signal), which
    then fails `reference_nondegeneracy` — no bypass of the crash-handling
    convention every other archetype uses."""
    crashy = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays["root_link_pos_w"]
    bad = 1.0 / 0.0  # ZeroDivisionError on every call
    return {"spec_score": 0.5}
'''
    clip = _rising_clip()
    p = _write(tmp_path, "crashy.py", crashy)
    v = validate_generated_metric(crashy, p, references=[("getup1", clip)])
    # The metric crashes on every fixed archetype too, so `bounded` is
    # already False — the reference block is skipped by its own guard
    # (gates.get("bounded")), so no reference_* keys are added at all.
    assert v["gates"]["bounded"] is False
    assert v["ok"] is False
    assert not any(k.startswith("reference_") for k in v["gates"])


def test_score_reference_entry_never_raises_on_a_crashing_metric():
    """Unit-level check of the reference scorer's own crash handling
    (independent of whether the fixed battery also crashes): a metric that
    raises scores `nan` — "no signal", exactly like `_score`'s existing
    archetype error path — never propagates the exception."""

    def crashy_fn(arrays, behavior, meta):
        raise RuntimeError("boom")

    clip = _rising_clip()
    score, _meta = _score_reference_entry(crashy_fn, clip, [])
    assert np.isnan(score)


# ── §D24 F3: `reference_complete_then_hold` — REQUIRED-HIGH positive gate ──
def test_honest_completion_style_metric_passes_complete_then_hold_gate(tmp_path):
    """A correct completion-style metric (started low, reached the goal,
    tolerant of an end-window hold) must score
    `complete_then_hold(reference)` at least `max(0.5, 0.8 * full)` — the
    HONEST_GETUP fixture is exactly this shape (started-low first-quarter
    read, reached-goal final-quarter read, no penalty for the final quarter
    staying flat)."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_complete_then_hold:getup1"] is True, v["reasons"]
    ref_result = v["references"][0]
    full = ref_result["scores"]["full"]
    cth = ref_result["scores"]["complete_then_hold"]
    assert cth >= max(0.5, 0.8 * full) - 1e-9


def _end_motion_clip() -> dict:
    """Continues rising gently through its LAST quarter (unlike
    `_rising_clip`'s dead-flat tail) — a clip an "the end must still be
    changing" metric legitimately certifies on the FULL reference, so a
    failure on `complete_then_hold` below isolates THAT metric's own defect
    rather than one already caught by nondegeneracy/monotonicity."""
    n_flat0, n_ramp, n_tail = 20, 40, 20
    flat0 = np.full(n_flat0, 0.1)
    s = np.linspace(0.0, 1.0, n_ramp)
    ramp = 0.1 + 0.6 * (1 - np.cos(np.pi * s)) / 2.0   # 0.1 -> 0.70
    tail = np.linspace(0.70, 0.75, n_tail)              # still slowly rising
    z = np.concatenate([flat0, ramp, tail])
    return {"root_pos_z": z, "fps": _FPS}


# An H2-class defect (D24 F3): in ADDITION to an honest rise/completion
# read, this metric requires the height to still be CHANGING in the final
# window. A rollout that reaches the goal and then HOLDS it (exactly what
# `complete_then_hold` constructs) reads as "stopped acting" and scores 0,
# even though it passes nondegeneracy/monotonicity/negatives cleanly on the
# still-drifting `_end_motion_clip` reference.
PUNISH_END_STILLNESS_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = max(2, n // 4)
    first_q = z[:q].mean()
    final_q = z[-q:].mean()
    rise = np.clip(final_q - first_q, 0.0, None)
    completion = float(final_q >= 0.5)
    end_motion = np.abs(np.diff(z[-q:], axis=0)).mean()
    motion_gate = float(np.clip(end_motion / 0.002, 0.0, 1.0))
    val = float(np.clip(completion * np.clip(rise / 0.5, 0.0, 1.0) * motion_gate, 0.0, 1.0))
    return {"spec_score": val, "motion_gate": motion_gate}
'''


def test_punish_end_stillness_metric_fails_complete_then_hold_with_scores_in_reason(
    tmp_path,
):
    clip = _end_motion_clip()
    p = _write(tmp_path, "punish_stillness.py", PUNISH_END_STILLNESS_METRIC)
    v = validate_generated_metric(
        PUNISH_END_STILLNESS_METRIC, p, references=[("getup1", clip)])
    # The other three reference gates are clean — this metric's ONLY defect
    # is punishing a held completion.
    assert v["gates"]["reference_nondegeneracy:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_monotonicity:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_negatives:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_complete_then_hold:getup1"] is False, v["reasons"]
    assert v["ok"] is False

    ref_result = v["references"][0]
    full = ref_result["scores"]["full"]
    cth = ref_result["scores"]["complete_then_hold"]
    assert cth < max(0.5, 0.8 * full)
    # archetype_scores carries the complete_then_hold entry regardless of
    # pass/fail.
    assert "reference:getup1:complete_then_hold" in v["archetype_scores"]
    assert v["archetype_scores"]["reference:getup1:complete_then_hold"] == pytest.approx(cth)
    # D12 idiom: the named numbers ride the failure reason.
    reason = next(r for r in ref_result["reasons"] if "complete_then_hold" in r)
    assert f"{cth:.3f}" in reason
    assert f"{full:.3f}" in reason


def test_fraction_window_metric_fails_complete_then_hold_at_x4(tmp_path):
    """PINNED EXPLOIT (found by the D24 F3 verification pass): a metric
    reading fraction-of-episode-length windows survives the x1 hold ratio
    by a narrow margin but collapses when the hold dominates — appending a
    long hold drags held terminal frames into its "start" window and kills
    the rise term, reproducing the D23 zero on a correct completion. The
    x4 ratio exists precisely to convict this class; a single-ratio gate
    let it through."""
    clip = _rising_clip()
    p = _write(tmp_path, "fraction_window.py", FRACTION_WINDOW_METRIC)
    v = validate_generated_metric(
        FRACTION_WINDOW_METRIC, p, references=[("getup1", clip)])
    assert v["gates"]["reference_complete_then_hold:getup1"] is False, v["reasons"]
    assert v["ok"] is False
    sc = v["references"][0]["scores"]
    threshold = max(0.5, 0.8 * sc["full"])
    # x1 narrowly clears the bar (the exploit) — x4 collapses (the conviction).
    assert sc["complete_then_hold"] >= threshold - 1e-9
    assert sc["complete_then_hold_x4"] < threshold
    assert "reference:getup1:complete_then_hold_x4" in v["archetype_scores"]
    reason = next(
        r for r in v["references"][0]["reasons"] if "complete_then_hold" in r)
    assert "x4" in reason


def test_freeze_end_rejected_and_complete_then_hold_passes_together(tmp_path):
    """The two gates have OPPOSITE polarity and must coexist on the same
    honest metric: `freeze_end` (never transitions, start == end) stays a
    rejected negative, while `complete_then_hold` (transitions, then holds
    the end) is a required positive — proving a hold-tolerant metric that
    also gates on the start being AWAY FROM the goal (HONEST_GETUP's
    `rise = final_q - first_q` factor) is not forced to choose between
    them."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_negatives:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_complete_then_hold:getup1"] is True, v["reasons"]
    sc = v["references"][0]["scores"]
    # freeze_end (start-from-goal, no transition) scores degenerate...
    assert sc["freeze_end"] < 0.1
    # ...while complete_then_hold (transition, then hold) scores high.
    assert sc["complete_then_hold"] >= 0.5
    assert v["ok"] is True, v["reasons"]
