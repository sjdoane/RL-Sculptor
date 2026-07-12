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
# RELATIVE-TIME authoring rule and die on complete_then_hold_x24 — that
# defect is pinned separately by FRACTION_WINDOW_METRIC below.
HONEST_GETUP = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    # start read: EARLIEST frames only (FAST-COMPLETION rule — a policy can
    # finish the whole transition inside any wide start window); min() is a
    # robust floor read. End read: a wide window is safe (held terminal).
    first_w = z[:min(5, n)].min()
    final_w = z[-min(15, n):].mean()
    rise = np.clip(final_w - first_w, 0.0, None)
    val = float(np.clip(final_w * rise / 0.75, 0.0, 1.0))
    return {"spec_score": val}
'''

# The live D24 defect shape, pinned: a metric whose started-away read is a
# WIDE start-window MEAN. It passes every hold gate (prefix reads are
# hold-invariant) but a 16x fast completion finishes INSIDE the window —
# the completed state leaks in, the gate zeroes, reproducing the third
# live zero-on-correct-behavior.
WIDE_START_WINDOW_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    started_low = z[:min(20, n)].mean() <= 0.2
    final_w = z[-min(15, n):].mean()
    rise = np.clip(final_w - z[:min(20, n)].mean(), 0.0, None)
    if not started_low or rise < 0.3:
        return {"spec_score": 0.0}
    val = float(np.clip(final_w * rise / 0.75, 0.0, 1.0))
    return {"spec_score": val}
'''

# The pre-D24 "honest" fixture shape, kept as a PINNED EXPLOIT FAMILY:
# fraction-of-episode windows (first/last n/k frames) read held terminal
# frames into the START window once the appended hold exceeds k-1 times the
# motion, so they clear complete_then_hold at small hold ratios but collapse
# at the realistic one (D23 live shape: ~0.5 s motion, 9.5 s hold, ~19x).
# A second verification pass proved k in [5, 20) also evaded a x4 second
# ratio — hence the gate's x24, which exceeds the whole realistic envelope.
# Every k here must FAIL the gate.
def _fraction_window_metric(k: int) -> str:
    return f'''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {{"spec_score": 0.0}}
    z = root[..., 2]
    n = z.shape[0]
    first_q = z[:max(1, n // {k})].mean()
    final_q = z[n - max(1, n // {k}):].mean()
    rise = np.clip(final_q - first_q, 0.0, None)
    val = float(np.clip(final_q * rise / 0.75, 0.0, 1.0))
    return {{"spec_score": val}}
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
    # _rising_clip ends settled (flat tail) -> fast_completion is GATED.
    assert ref_result["gates"] == {
        "reference_nondegeneracy": True,
        "reference_monotonicity": True,
        "reference_negatives": True,
        "reference_complete_then_hold": True,
        "reference_fast_completion": True,
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
    # start read: EARLIEST frames (FAST-COMPLETION rule); end reads wide.
    q = min(15, n)
    rise = np.clip(z[-q:].mean() - z[:min(5, n)].min(), 0.0, None)
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


@pytest.mark.parametrize("k", [4, 10, 15, 20])
def test_fraction_window_metric_family_fails_complete_then_hold(tmp_path, k):
    """PINNED EXPLOIT FAMILY (two D24 verification passes): metrics reading
    first/last n/k windows survive small hold ratios but collapse once the
    appended hold exceeds k-1 times the motion — held terminal frames leak
    into the "start" window and kill the rise term, reproducing the D23
    zero on a correct completion. k=4 fell to the original x4 gate; k in
    [5, 20) evaded x4 while failing the realistic 19x live shape — the x24
    ratio convicts the whole family up to the operating envelope."""
    clip = _rising_clip()
    src = _fraction_window_metric(k)
    p = _write(tmp_path, f"fraction_window_{k}.py", src)
    v = validate_generated_metric(src, p, references=[("getup1", clip)])
    assert v["gates"]["reference_complete_then_hold:getup1"] is False, (
        k, v["reasons"])
    assert v["ok"] is False
    sc = v["references"][0]["scores"]
    threshold = max(0.5, 0.8 * sc["full"])
    # x1 clears the bar (the evasion) — x24 collapses (the conviction).
    assert sc["complete_then_hold"] >= threshold - 1e-9, (k, sc)
    assert sc["complete_then_hold_x24"] < threshold, (k, sc)
    assert "reference:getup1:complete_then_hold_x24" in v["archetype_scores"]
    reason = next(
        r for r in v["references"][0]["reasons"] if "complete_then_hold" in r)
    assert "x24" in reason


def test_wide_start_window_metric_fails_fast_completion(tmp_path):
    """THIRD live zero-on-correct-behavior (D24): a freshly certified
    metric read its started-away state as a 0.5 s window MEAN; the policy
    completed the 8.1 s righting span in ~0.5 s, the completed state
    leaked into the window, gate zeroed a perfect rollout. The 16x
    fast_completion positive convicts that read at certification."""
    clip = _rising_clip()
    p = _write(tmp_path, "wide_window.py", WIDE_START_WINDOW_METRIC)
    v = validate_generated_metric(
        WIDE_START_WINDOW_METRIC, p, references=[("getup1", clip)])
    assert v["gates"]["reference_fast_completion:getup1"] is False, v["reasons"]
    assert v["ok"] is False
    assert "reference:getup1:fast_completion" in v["archetype_scores"]
    reason = next(
        r for r in v["references"][0]["reasons"] if "fast_completion" in r)
    assert "start" in reason.lower()


def test_fast_completion_abstains_for_non_settled_end(tmp_path):
    """A clip still MOVING at its end (gait/rhythm shaped) must NOT gate
    fast_completion (D3: a 16x-sped gait is a different motion; convicting
    an honest pace-sensitive metric would be a false reject) — recorded
    with an explicit abstain marker instead."""
    n = _T
    z = np.concatenate([np.full(n // 4, 0.1),
                        0.1 + 0.65 * np.linspace(0.0, 1.0, n - n // 4)])
    clip = {"root_pos_z": z, "fps": _FPS}  # rising to the very last frame
    p = _write(tmp_path, "honest_moving_end.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("moving_end", clip)])
    gates = v["references"][0]["gates"]
    assert "reference_fast_completion" not in gates
    sc = v["references"][0]["scores"]
    assert "fast_completion" in sc
    assert sc.get("fast_completion_abstained") is True


def test_hold_invariant_metric_passes_all_hold_ratios(tmp_path):
    """The gate's intent is achievable: a fixed-size-window (hold-invariant)
    metric scores identically at every hold ratio and passes."""
    clip = _rising_clip()
    p = _write(tmp_path, "hold_invariant.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_complete_then_hold:getup1"] is True, v["reasons"]
    sc = v["references"][0]["scores"]
    assert sc["complete_then_hold_x24"] == pytest.approx(
        sc["complete_then_hold"], abs=1e-6)


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


# ── §D24 F2: `reference_settled_start` — a POSITIVE gate requiring the
# metric to score a variant STARTING from the stage's ACTUAL certified
# eval-reset state (not the clip's own frame 0) at least as well as the
# full reference. Uses the REAL torso_righting fixture (the exact clip
# D23 diagnosed), cropped to the canonical [0, 8.5]s span
# (test_reference_spans.py's own convention) — the shipped
# eval_reset.json's height offset (-0.6644, absolute z 0.0756) genuinely
# differs from the clip's own frame-0 root_pos_z (0.1613).
FIXTURE_CLIP_PATH = (
    Path(__file__).parent / "fixtures" / "torso_righting_satup" / "reference_clip.npz"
)


@pytest.fixture(scope="module")
def torso_righting_span() -> dict:
    from sculptor.reference import load_clip
    from sculptor.refs.spans import crop_span

    clip = load_clip(FIXTURE_CLIP_PATH)
    return crop_span(clip, 0.0, 8.5)


#: `{"scalars": ..., "settled": bool, "reason": ...}` — the shape
#: `mission_metrics._compute_eval_reset_preview` returns. Z-only
#: (no pitch/roll) deliberately — exercises `settled_start_hold`'s
#: z-only fallback path (`orientation_adjusted: False`).
_EVAL_RESET_PREVIEW = {
    "scalars": {"reset_height_offset_m": -0.6644},
    "settled": True,
    "reason": None,
}

#: An HONEST started-low-tolerant metric: height gate accepts a WIDE
#: band (both the clip's own frame-0 z=0.161 and the settled z=0.076
#: clear it), the real signal is the g_z uprightness change.
HONEST_TORSO_RIGHT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    gz = g[..., 2]
    n = z.shape[0]
    q = min(10, n)
    started_low = float(np.clip((0.30 - z[:q].mean()) / 0.20, 0.0, 1.0))
    rights = float(np.clip((gz[:q].mean() - gz[-q:].mean()) / 0.15, 0.0, 1.0))
    val = float(np.clip(started_low * rights, 0.0, 1.0))
    return {"spec_score": val}
'''

#: The H2 defect (D23 class): hard-gates the start window to be CLOSE to
#: the clip's OWN frame-0 height (narrow band around 0.1429, the
#: cropped span's first-10-frame mean) — silently mis-scores a rollout
#: that actually starts from the settled eval-reset (0.076 m, outside
#: the band).
NARROW_START_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    gz = g[..., 2]
    n = z.shape[0]
    q = min(10, n)
    start_match = float(abs(z[:q].mean() - 0.1429) < 0.03)
    rights = float(np.clip((gz[:q].mean() - gz[-q:].mean()) / 0.15, 0.0, 1.0))
    val = float(np.clip(start_match * rights, 0.0, 1.0))
    return {"spec_score": val}
'''


def test_honest_started_low_tolerant_metric_passes_settled_start_gate(
    tmp_path, torso_righting_span,
):
    p = _write(tmp_path, "honest_torso.py", HONEST_TORSO_RIGHT)
    v = validate_generated_metric(
        HONEST_TORSO_RIGHT, p, references=[("torso", torso_righting_span)],
        eval_reset=_EVAL_RESET_PREVIEW)
    assert v["gates"]["reference_settled_start:torso"] is True, v["reasons"]
    assert v["ok"] is True, v["reasons"]

    ref_result = v["references"][0]
    assert ref_result.get("settled_start_abstained") is None
    assert ref_result["settled_start_orientation_adjusted"] is False
    full = ref_result["scores"]["full"]
    ss = ref_result["scores"]["settled_start"]
    assert ss >= max(0.5, 0.8 * full) - 1e-9
    assert "reference:torso:settled_start" in v["archetype_scores"]


def test_frame0_calibrated_metric_fails_settled_start_gate_with_scores_in_reason(
    tmp_path, torso_righting_span,
):
    """The H2 defect made concrete: a metric that hard-gates the start
    window to the CLIP'S OWN frame-0 numbers fails when the rollout
    actually starts from a measurably different settled state."""
    p = _write(tmp_path, "narrow_start.py", NARROW_START_METRIC)
    v = validate_generated_metric(
        NARROW_START_METRIC, p, references=[("torso", torso_righting_span)],
        eval_reset=_EVAL_RESET_PREVIEW)
    assert v["gates"]["reference_settled_start:torso"] is False, v["reasons"]
    assert v["ok"] is False

    ref_result = v["references"][0]
    full = ref_result["scores"]["full"]
    ss = ref_result["scores"]["settled_start"]
    assert ss < max(0.5, 0.8 * full)
    # D12 idiom: the named numbers ride the failure reason.
    reason = next(r for r in ref_result["reasons"] if "settled_start" in r)
    assert f"{ss:.3f}" in reason
    assert "certified eval-reset" in reason


def test_no_eval_reset_preview_abstains_never_silently_passes_or_fails(
    tmp_path, torso_righting_span,
):
    """When the caller supplies no `eval_reset` (the default), the gate
    is OMITTED from `gates` entirely — never a silent pass, never a
    silent fail — and `ok` is unaffected (byte-identical to pre-F2
    behavior)."""
    p = _write(tmp_path, "honest_torso2.py", HONEST_TORSO_RIGHT)
    v_with = validate_generated_metric(
        HONEST_TORSO_RIGHT, p, references=[("torso", torso_righting_span)],
        eval_reset=_EVAL_RESET_PREVIEW)
    v_without = validate_generated_metric(
        HONEST_TORSO_RIGHT, p, references=[("torso", torso_righting_span)])

    assert "reference_settled_start:torso" not in v_without["gates"]
    assert v_without["references"][0]["settled_start_abstained"] is True
    assert "settled_start" not in v_without["references"][0]["scores"]
    assert v_without["ok"] == v_with["ok"] is True


def test_settled_start_with_orientation_records_adjustment(tmp_path):
    """When the eval-reset preview carries pitch AND roll, and the clip
    carries `root_quat_wxyz`, the prepend's orientation is ALSO
    corrected — `settled_start_orientation_adjusted` records True."""
    clip = _clip_with_posture()
    p = _write(tmp_path, "posture_honest.py", HONEST_GETUP_POSTURE)
    eval_reset = {
        "scalars": {
            "reset_height_offset_m": -0.65,
            "reset_pitch_offset_rad": 0.3,
            "reset_roll_offset_rad": 0.0,
        },
        "settled": True, "reason": None,
    }
    v = validate_generated_metric(
        HONEST_GETUP_POSTURE, p, references=[("getup1", clip)],
        eval_reset=eval_reset)
    assert v["references"][0]["settled_start_orientation_adjusted"] is True


def test_keyword_family_does_not_block_reference_defer(tmp_path):
    """LIVE D28 FINDING (g1-standing prone mission): family inference is a
    keyword guess — 'drive up ... keep balance' routed two GET-UP stage
    goals to a builtin family (jump / cartpole depending on phrasing) whose
    archetype battery scores every entry 0.000 for a get-up metric. The old
    `family is None` term then blocked the D4 reference-defer and the
    uninformative battery rejected metrics that had passed ALL SIX
    reference gates as 'near-constant'. The defer now fires whenever the
    battery is uninformative and references are attached, family or not."""
    from sculptor.eval.metric_validate import resolve_behavior_family

    live_goal = (
        "From a low crouch with both feet planted and bearing weight, "
        "drive up through the legs, extending hips and knees to raise "
        "the pelvis to full standing height while bringing the torso "
        "fully upright, then keep balance briefly at the top.")
    # Precondition of the bug: the goal DOES resolve to a builtin family.
    assert resolve_behavior_family(live_goal, None) is not None

    clip = _rising_clip()
    p = _write(tmp_path, "honest_family.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)],
        behavior_goal=live_goal)
    assert v["ok"] is True, v["reasons"]
    assert v["gates"]["nondegeneracy"] is True
    assert any("deferred to attached reference" in r for r in v["reasons"])
