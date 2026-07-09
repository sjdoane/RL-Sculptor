"""§REFERENCE_TRAJECTORY_PLAN §6: reference-derived calibration ladders —
steer-rights for a NOVEL motion by ranking a competence ladder built FROM a
single attached reference clip. Fully offline/deterministic: no LLM call,
no library/network dependency (a synthetic rising clip fixture stands in
for a retrieved reference).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.metric_calibration import (
    calibrate_metric_against_reference,
    compute_trust,
)

# ── fixtures ────────────────────────────────────────────────────────────

#: A clean monotone-rising synthetic clip: root height ramps from 0.55 m to
#: 1.05 m with a smooth joint oscillation riding along. The height range is
#: kept ABOVE the fixed battery's degenerate anchor (still/chaotic at
#: z=0.5, fallen at z=0.1 -> anchor 0.5 for a pure-height metric) so an
#: HONEST metric's ladder orders correctly rung-by-rung, including rung 0.
_T = 150
_FPS = 50.0


def _rising_clip() -> dict:
    z = np.linspace(0.55, 1.05, _T)
    t = np.linspace(0.0, 1.0, _T)
    jp = np.stack([0.3 * np.sin(2 * np.pi * t), 0.2 * np.cos(2 * np.pi * t)], axis=1)
    return {
        "root_pos_z": z.astype(np.float64),
        "fps": _FPS,
        "joint_pos": jp,
        "joint_names": ["j0", "j1"],
    }


#: An honest metric: final-decile mean root height, normalized. Monotone in
#: the clip's own construction (degrade damps root height toward the START
#: pose which is LOWER; truncate cuts off the rise early) so it should rank
#: the ladder correctly and clear the firewall (the degenerate battery sits
#: at height 0.5, well under this clip's ramp).
_HONEST_HEIGHT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = z[int(0.9 * n):]
    val = float(np.clip(np.mean(q) / 1.2, 0.0, 1.0))
    return {"spec_score": val}
'''

_CONSTANT = '''def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''

#: The mirror image of the honest metric: LOW final height scores HIGH.
_REVERSED_HEIGHT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    q = z[int(0.9 * n):]
    val = float(np.clip(1.0 - np.mean(q) / 1.2, 0.0, 1.0))
    return {"spec_score": val}
'''

#: A metric that ranks the height ladder well ENOUGH (0.6 weight) but adds a
#: confound (0.4 weight) rewarding near-zero total joint excursion — the
#: degenerate battery's `still`/`chaotic` archetypes carry EXACTLY zero
#: joint_pos on this clip's own two-joint layout(*), so the confound scores
#: the degenerate anchor competitively with the full reference. This must
#: trip the firewall (rung-0-vs-full-clip margin), independent of rho.
#: (*) the archetype battery is a DIFFERENT 12-joint synthetic body; the
#: confound reads absolute joint_pos magnitude generically, which is still
#: near-zero on the quiet/near-still archetypes.
_FIREWALL_GAMEABLE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos")
    root = arrays.get("root_link_pos_w")
    if jp is None or root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    n = z.shape[0]
    height = float(np.clip(np.mean(z[int(0.9 * n):]) / 1.2, 0.0, 1.0))
    quiet = float(np.clip(1.0 - np.mean(np.abs(jp)) / 0.3, 0.0, 1.0))
    val = float(np.clip(0.6 * height + 0.4 * quiet, 0.0, 1.0))
    return {"spec_score": val}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


# ── tests ───────────────────────────────────────────────────────────────


def test_honest_metric_grants_steer_on_tier_d(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res["ok"] is True
    assert res["method"] == "reference"
    assert res["spearman"] >= 0.7
    assert res["rho_min"] == res["spearman"]
    assert res["rights"] == "steer"
    assert res["trust_tier"] == "reference:D:hf_dataset"
    assert res["clip_id"] == "rise_clip"
    assert res["tier"] == "D"
    assert res["source_kind"] == "hf_dataset"


def test_honest_metric_grants_observe_on_tier_k(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="K", source_kind="hf_dataset")

    assert res["ok"] is True
    assert res["rights"] == "observe"
    assert res["trust_tier"] == "reference:K:hf_dataset"


def test_ladder_order_recorded_correctly(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    ladder = res["ladder"]
    names = [r["rung"] for r in ladder]
    ranks = [r["intended_rank"] for r in ladder]
    assert names == [
        "degenerate_battery", "degrade_heavy", "degrade_light",
        "truncate_25", "truncate_50", "truncate_75", "full_clip",
    ]
    assert ranks == [0, 1, 2, 3, 4, 5, 6]
    # Every rung recorded a finite numeric score.
    for rung in ladder:
        assert np.isfinite(rung["score"])


def test_constant_metric_denied(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["spearman"] == 0.0


def test_reversed_metric_denied(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "reversed.py", _REVERSED_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["spearman"] < 0.0


def test_firewall_denies_degenerate_battery_gaming(tmp_path):
    """A metric that ranks the ladder somewhat but scores the fixed battery's
    degenerate anchor competitively with the full reference must be denied
    by the firewall — even though it may show SOME rank correlation."""
    clip = _rising_clip()
    p = _write(tmp_path, "gameable.py", _FIREWALL_GAMEABLE)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res["ok"] is False
    assert res["rights"] == "none"
    ladder = {r["rung"]: r["score"] for r in res["ladder"]}
    # The firewall margin is violated: full_clip does not clear
    # degenerate_battery by the required spread.
    assert ladder["full_clip"] < ladder["degenerate_battery"] + 0.1
    assert "degenerate anchor" in res["reason"]


def test_trust_tier_string_exact(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    for tier, source_kind in (("D", "hf_dataset"), ("K", "video"), ("D", "generated")):
        res = calibrate_metric_against_reference(
            p, "rise_clip", clip, tier=tier, source_kind=source_kind)
        assert res["trust_tier"] == f"reference:{tier}:{source_kind}"


def test_rights_none_when_not_ok(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    for tier in ("D", "K"):
        res = calibrate_metric_against_reference(
            p, "rise_clip", clip, tier=tier, source_kind="hf_dataset")
        assert res["rights"] == "none"


def test_compute_trust_consumes_reference_result_without_error(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    trust = compute_trust(res)
    assert "trust" in trust
    assert 0.0 <= trust["trust"] <= 1.0
    assert trust["method"] == "reference"
    assert trust["rho_min"] == res["rho_min"]
    assert trust["agreement_fraction"] == res["agreement_fraction"]


def test_compute_trust_consumes_denied_reference_result_without_error(tmp_path):
    clip = _rising_clip()
    p = _write(tmp_path, "const.py", _CONSTANT)
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    trust = compute_trust(res)
    assert "trust" in trust
    assert 0.0 <= trust["trust"] <= 1.0


def test_load_failure_never_raises(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("this is not ( valid python", encoding="utf-8")
    clip = _rising_clip()
    res = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res["ok"] is False
    assert res["rights"] == "none"
    assert res["error"] is not None


def test_determinism(tmp_path):
    """Everything is seeded — two calls on the same inputs must produce byte-
    identical ladders (degrade()'s seed=0 default)."""
    clip = _rising_clip()
    p = _write(tmp_path, "honest.py", _HONEST_HEIGHT)
    res1 = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")
    res2 = calibrate_metric_against_reference(
        p, "rise_clip", clip, tier="D", source_kind="hf_dataset")

    assert res1["ladder"] == res2["ladder"]
    assert res1["spearman"] == res2["spearman"]
