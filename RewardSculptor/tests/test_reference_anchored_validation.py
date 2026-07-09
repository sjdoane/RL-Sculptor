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


# An HONEST metric: rewards final-quarter height ONLY when it actually rose
# from a lower start (final*rise, not final alone) — final alone would be
# gamed by "hold the end pose the whole time" (freeze_end), which the
# reference_negatives gate is specifically built to catch.
HONEST_GETUP = '''import numpy as np
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


def test_honest_metric_clears_all_three_reference_gates(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", HONEST_GETUP)
    v = validate_generated_metric(
        HONEST_GETUP, p, references=[("getup1", clip)])
    assert v["gates"]["reference_nondegeneracy:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_monotonicity:getup1"] is True, v["reasons"]
    assert v["gates"]["reference_negatives:getup1"] is True, v["reasons"]
    assert v["ok"] is True, v["reasons"]

    ref_result = v["references"][0]
    assert ref_result["clip_id"] == "getup1"
    assert ref_result["gates"] == {
        "reference_nondegeneracy": True,
        "reference_monotonicity": True,
        "reference_negatives": True,
    }
    # speed_slow/speed_fast are scored + recorded but not gate keys.
    assert "speed_slow" in ref_result["scores"]
    assert "speed_fast" in ref_result["scores"]
    assert f"reference:getup1" in v["archetype_scores"]
    assert f"reference:getup1:trunc_50" in v["archetype_scores"]


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


def test_constant_metric_fails_reference_nondegeneracy(tmp_path):
    clip = _rising_clip()
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
