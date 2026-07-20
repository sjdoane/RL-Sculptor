"""§Ship 50: L1 task-agnostic axioms — each controlled-perturbation invariant
catches its targeted Goodhart leak, every GOOD_*/built-in reference metric
passes, and a novel-orientation (handstand) task is NOT false-rejected.
GPU-free; metrics are written to temp .py files."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sculptor.eval.generated_metric import load_generated_metric, read_required_roles
from sculptor.eval.metric_axioms import check_metric_axioms
from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.eval.spec_metrics import _SPEC_FNS
import tests.test_generated_metric as TG


def _load(src: str):
    d = Path(tempfile.mkdtemp())
    p = d / "m.py"
    p.write_text(src, encoding="utf-8")
    return load_generated_metric(p), read_required_roles(p)


def _axioms(src: str, *, family=None):
    fn, roles = _load(src)
    return check_metric_axioms(fn, family=family, required_roles=roles)


# ── reference good metrics pass every axiom ──────────────────────────


@pytest.mark.parametrize("name,family", [
    ("GOOD", "locomotion"),
    ("GOOD_KICK", "kick"),
    ("GOOD_JUMP", "jump"),
    ("GOOD_FLOSS", "floss"),
])
def test_good_metrics_pass_all_axioms(name, family):
    res = _axioms(getattr(TG, name), family=family)
    assert res["ok"], (name, res["reasons"], res["details"])
    assert all(res["axioms"].values()), (name, res["axioms"])


@pytest.mark.parametrize("builtin", list(_SPEC_FNS))
def test_builtins_pass_all_axioms(builtin):
    """The hand-authored ground truths are the calibration fence — they must
    never trip their own L1 axioms (cartpole reads only behavior, so its
    perturbations are inert → vacuously passes)."""
    fam = {"g1_kick": "kick", "g1_floss": "floss", "g1_jump": "jump",
           "go1_trot": "locomotion", "cartpole_balance": "cartpole",
           "object_lift_hold": "manipulation"}[builtin]
    res = check_metric_axioms(_SPEC_FNS[builtin], family=fam)
    assert res["ok"], (builtin, res["reasons"], res["details"])


# ── A1: uprightness monotonicity ─────────────────────────────────────

FALL_REWARDER = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = arrays.get("projected_gravity_b")
    if g is None:
        return {"spec_score": 0.0}
    n = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    toppled = float((np.abs(n[..., 0]) > 0.85).mean())   # high when on its side
    return {"spec_score": float(np.clip(toppled, 0.0, 1.0))}
'''


def test_fall_rewarder_fails_uprightness():
    res = _axioms(FALL_REWARDER, family="kick")
    assert res["axioms"]["uprightness_monotone"] is False
    assert any("uprightness" in r for r in res["reasons"])
    assert not res["ok"]


# ── A2: no reward for chaos ──────────────────────────────────────────


def test_flail_rewarder_fails_chaos():
    """FLAIL_REWARDER (motion magnitude) ranks archetypes badly AND its score
    rises under injected chaos — the controlled A2 catch."""
    res = _axioms(TG.FLAIL_REWARDER, family="kick")
    assert res["axioms"]["no_reward_for_chaos"] is False
    assert any("chaos" in r for r in res["reasons"])


# ── A3: stationary skills must not reward travel ─────────────────────

# A "kick" metric that wrongly blends leg motion with FORWARD TRAVEL credit.
TRAVEL_KICK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); root = arrays.get("root_link_pos_w")
    g = arrays.get("projected_gravity_b")
    if jv is None or root is None or g is None:
        return {"spec_score": 0.0}
    up = float((g[..., 2] < -0.85).mean())
    legpeak = 1.0 - float(np.exp(-np.abs(jv).max() / 8.0))
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    travel = 1.0 - float(np.exp(-disp / 3.0))
    return {"spec_score": float(np.clip(up * 0.5 * (legpeak + travel), 0.0, 1.0))}
'''


def test_travel_kick_fails_stationarity():
    res = _axioms(TRAVEL_KICK, family="kick")
    assert res["axioms"]["stationary_no_travel"] is False
    assert any("stationarity" in r for r in res["reasons"])


def test_stationarity_axiom_scoped_to_stationary_families():
    """For LOCOMOTION the travel axiom must NOT run — travel is the target.
    GOOD (a forward-travel metric) has no stationarity axiom and passes."""
    res = _axioms(TG.GOOD, family="locomotion")
    assert "stationary_no_travel" not in res["axioms"]
    assert res["ok"]


# ── invariances (L0 cannot express these) ────────────────────────────

# Rewards ABSOLUTE world x-position (fails to difference) — a coordinate leak.
ABSPOS_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    up = float((g[..., 2] < -0.85).mean())
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-np.abs(root[..., 0]).mean() / 50.0)), 0.0, 1.0))}
'''

# Reads RAW gravity magnitude as a pseudo-uprightness — units-fragile.
GRAVMAG_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = arrays.get("projected_gravity_b")
    if g is None:
        return {"spec_score": 0.0}
    return {"spec_score": float(np.clip(1.0 - np.exp(-np.abs(g[..., 2]).mean() / 5.0), 0.0, 1.0))}
'''

# Rewards SIGNED forward (+x) heading rather than distance travelled.
HEADING_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); g = arrays.get("projected_gravity_b")
    if root is None or g is None:
        return {"spec_score": 0.0}
    up = float((g[..., 2] < -0.85).mean())
    dx = float((root[-1, :, 0] - root[0, :, 0]).mean())
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-max(dx, 0.0) / 3.0)), 0.0, 1.0))}
'''


def test_translation_invariance_catches_absolute_position():
    res = _axioms(ABSPOS_HACK, family="locomotion")
    assert res["axioms"]["translation_invariant"] is False
    assert any("translation" in r for r in res["reasons"])


def test_gravity_scale_invariance_catches_raw_magnitude():
    res = _axioms(GRAVMAG_HACK, family="locomotion")
    assert res["axioms"]["gravity_scale_invariant"] is False
    assert any("gravity_scale" in r for r in res["reasons"])


def test_yaw_invariance_catches_absolute_heading():
    res = _axioms(HEADING_HACK, family="locomotion")
    assert res["axioms"]["yaw_rotation_invariant"] is False
    assert any("yaw" in r for r in res["reasons"])


def test_good_metrics_invariant_under_frame_and_units():
    """Every GOOD_*/built-in reference metric is exactly invariant under the
    three physical symmetries (the zero-false-rejection guarantee)."""
    for src in (TG.GOOD, TG.GOOD_KICK, TG.GOOD_JUMP, TG.GOOD_FLOSS):
        res = _axioms(src, family="kick")
        for inv in ("translation_invariant", "gravity_scale_invariant",
                    "yaw_rotation_invariant"):
            assert res["axioms"][inv] is True, (src[:40], inv, res["details"])


# ── novel-orientation task is not false-rejected ─────────────────────

HANDSTAND = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = arrays.get("projected_gravity_b")
    if g is None:
        return {"spec_score": 0.0}
    n = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    inverted = float((n[..., 2] > 0.85).mean())          # upside-down = handstand
    return {"spec_score": float(np.clip(inverted, 0.0, 1.0))}
'''


def test_handstand_metric_not_false_rejected():
    """A handstand metric scores the upright biped battery ~0, so the
    perturbations stay ~0 and the universal axioms pass vacuously — L1 never
    rejects a task whose competent behavior the battery cannot represent.
    Crucially the uprightness SWEEP tilts toward horizontal (never inversion),
    so even a metric that did score this battery would not be penalised for an
    inverted target."""
    res = _axioms(HANDSTAND, family=None)        # 'handstand' resolves to no family
    assert res["ok"], (res["reasons"], res["details"])
    assert res["axioms"]["uprightness_monotone"] is True


# ── §LAW 13: M1 uprightness is scoped to an UPRIGHT torso target ──────


def test_uprightness_axiom_scoped_to_upright_torso():
    """The M1 sweep penalises a fall-rewarder as an UPRIGHT skill but is SKIPPED
    for a horizontal target (flip/dive/roll) — the same metric that competently
    rewards a horizontal torso must not be M1-rejected."""
    fn, _ = _load(FALL_REWARDER)
    upright = check_metric_axioms(fn, family="kick", torso_target="upright")
    flip = check_metric_axioms(fn, family="kick", torso_target="horizontal")
    assert upright["axioms"]["uprightness_monotone"] is False      # penalised when upright
    assert "uprightness_monotone" not in flip["axioms"]            # skipped for a flip
    assert flip["details"].get("uprightness_monotone_skipped")
    assert not any("uprightness" in r for r in flip["reasons"])


def test_resolve_torso_target_keywords():
    from sculptor.eval.metric_validate import resolve_torso_target

    assert resolve_torso_target("do a backflip") == "horizontal"
    assert resolve_torso_target("dive forward and roll") == "horizontal"
    assert resolve_torso_target("hold a handstand") == "any"
    assert resolve_torso_target("kick forward with one leg") == "upright"
    assert resolve_torso_target(None) == "upright"


def test_validate_does_not_uprightness_reject_a_flip_goal():
    """End-to-end through validate_generated_metric: a flip goal resolves
    torso_target='horizontal', so the uprightness axiom never fires (the metric
    may still fail OTHER gates, but not for holding a non-upright torso)."""
    d = Path(tempfile.mkdtemp()); p = d / "m.py"; p.write_text(FALL_REWARDER)
    v = validate_generated_metric(FALL_REWARDER, p, behavior_goal="do a backflip")
    assert "uprightness_monotone" not in v["axioms"]["axioms"]
    assert not any("uprightness" in r for r in v["axioms"]["reasons"])


# ── determinism + validate integration ───────────────────────────────


def test_axioms_deterministic():
    fn, roles = _load(TG.GOOD_KICK)
    a = check_metric_axioms(fn, family="kick", required_roles=roles)
    b = check_metric_axioms(fn, family="kick", required_roles=roles)
    assert a["axioms"] == b["axioms"] and a["details"] == b["details"]


def test_validate_folds_axioms_into_gate_and_record():
    """validate_generated_metric runs the L1 gate, exposes the per-axiom
    breakdown, and a GOOD metric passes it."""
    fn_src = TG.GOOD
    d = Path(tempfile.mkdtemp()); p = d / "m.py"; p.write_text(fn_src)
    v = validate_generated_metric(fn_src, p, behavior_goal="trot forward")
    assert v["gates"]["axioms"] is True
    assert v["axioms"]["ok"] is True
    assert "no_reward_for_chaos" in v["axioms"]["axioms"]
    assert v["ok"]


def test_validate_rejects_fall_rewarder_via_axioms():
    d = Path(tempfile.mkdtemp()); p = d / "m.py"; p.write_text(FALL_REWARDER)
    v = validate_generated_metric(FALL_REWARDER, p, behavior_goal="kick forward")
    assert v["gates"]["axioms"] is False
    assert not v["ok"]
