"""§Ship 35: generated-metric runtime (load/compute/resolve) + the
validation gates. GPU-free; metrics are written to temp .py files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.generated_metric import (
    compute_generated_metric,
    make_generated_fitness_fn,
    resolve_fitness_fn,
)
from sculptor.eval.metric_validate import validate_generated_metric

# A valid task metric: forward travel × uprightness, physical + bounded.
GOOD = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None:
        return {"spec_score": 0.0}
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    up = float(np.mean(grav[..., 2] < -0.85))
    speed = 1.0 - float(np.exp(-disp / 2.0))
    return {"spec_score": float(np.clip(speed * up, 0.0, 1.0)), "disp": disp, "up": up}
'''

BAD_IMPORT = '''import os
import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''

BAD_ARRAY = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    x = arrays["contact_forces"]
    return {"spec_score": float(np.clip(x.mean(), 0, 1))}
'''

REWARDS_STILLNESS = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel")
    motion = float(np.mean(np.abs(jv))) if jv is not None else 0.0
    return {"spec_score": float(max(0.0, 1.0 - motion))}
'''

NONDETERMINISTIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": float(np.random.random())}
'''

# §Ship 35 review: __builtins__ is in every module namespace without an
# import → a code-exec escape if the safety gate only blocks imports.
BUILTINS_ESCAPE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    b = __builtins__
    return {"spec_score": 0.5}
'''

# §Ship 36: a metric that merely rewards motion MAGNITUDE — it scores a
# standing-and-flailing policy (the G1 kick hack) as highly as a competent
# one. The `upright_flail` archetype + non-degeneracy gate must reject it.
FLAIL_REWARDER = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    if jv is None or grav is None:
        return {"spec_score": 0.0}
    motion = float(np.mean(np.abs(jv)))
    up = float(np.mean(grav[..., 2] < -0.85))
    return {"spec_score": float(np.clip(motion / 5.0 * up, 0.0, 1.0))}
'''

# §Ship 41/49: a CORRECT kick metric — leg-velocity bursts from a STATIONARY,
# upright, standing-height stance. It needs all four arrays (returns 0.0 if any
# is missing), so the old locomotion-only `active` archetype scored it ~0
# (stationary → no travel) and the ladder (no root/joint_pos) couldn't
# calibrate it. §Ship 49: declares REQUIRED_JOINT_ROLES and reads the knees via
# meta["joint_roles"] — the safe, permutation-robust pattern (the old
# `jv[..., 2:4]` hard-coded indices that point at left_hip_yaw+left_knee on a
# real 29-DOF G1, and the permutation gate now rejects them).
GOOD_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w"); jp = arrays.get("joint_pos")
    if jv is None or grav is None or root is None or jp is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) if isinstance(meta, dict) else {}
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    xy = root[..., :2]
    drift = float(np.linalg.norm(xy.max(0) - xy.min(0), axis=-1).mean())
    stationary = float(np.exp(-drift / 0.4))
    zmed = float(np.median(root[..., 2]))
    height = float(np.clip((zmed - 0.45) / 0.20, 0.0, 1.0))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    # §LAW 4: signed FORWARD direction from foot position (pelvis frame). Abstains
    # (1.0) when the channel is absent so the footless calibration ladder still
    # ranks; a rearward/sideways kick (foot moves backward) scores ~0.
    direction = 1.0
    feet = [f for f in (arrays.get("left_foot_pos_b"), arrays.get("right_foot_pos_b"))
            if f is not None]
    if feet:
        ds = []
        for foot in feet:
            a = foot[..., 0]; dev = a - np.median(a, axis=0, keepdims=True)
            fwd = np.clip(dev, 0.0, None).max(axis=0)
            back = np.clip(-dev, 0.0, None).max(axis=0)
            ds.append((fwd / (fwd + back + 1e-6)) * (1.0 - np.exp(-fwd / 0.12)))
        direction = float(np.maximum.reduce(ds).mean()) if len(ds) > 1 else float(ds[0].mean())
    return {"spec_score": float(np.clip(up * stationary * height * burst * direction, 0.0, 1.0))}
'''

# §Ship 41: a correct JUMP metric — vertical base excursion while upright.
GOOD_JUMP = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    apex = float(np.clip((z - np.median(z, axis=0)).max(axis=0).mean(), 0.0, None))
    up = float(np.mean(grav[..., 2] < -0.85))
    return {"spec_score": float(np.clip((1.0 - np.exp(-apex / 0.3)) * up, 0.0, 1.0))}
'''

# §Ship 41: a correct FLOSS metric — hip<->arm ANTI-PHASE (negative correlation),
# not mere motion magnitude. A magnitude metric would score upright_flail high
# and be rejected; this keys on structure, so it beats flail.
GOOD_FLOSS = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos"); grav = arrays.get("projected_gravity_b")
    if jp is None or grav is None:
        return {"spec_score": 0.0}
    names = (meta or {}).get("joint_names", []) if isinstance(meta, dict) else []
    if len(names) != jp.shape[2]:
        return {"spec_score": 0.0}
    hips = [i for i, n in enumerate(names) if "hip" in n.lower()]
    arms = [i for i, n in enumerate(names)
            if "shoulder" in n.lower() or "elbow" in n.lower()]
    if not hips or not arms:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    h = jp[..., hips].mean(axis=2); a = jp[..., arms].mean(axis=2)
    h = h - h.mean(axis=0, keepdims=True); a = a - a.mean(axis=0, keepdims=True)
    den = h.std(axis=0) * a.std(axis=0) + 1e-9
    corr = float(((h * a).mean(axis=0) / den).mean())   # ~ -1 for anti-phase
    anti = float(np.clip(-corr, 0.0, 1.0))
    amp = float((h.std(axis=0).mean() + a.std(axis=0).mean()) / 2.0)
    moving = 1.0 - float(np.exp(-amp / 0.2))
    return {"spec_score": float(np.clip(anti * moving * up, 0.0, 1.0))}
'''

# §Ship 41 review (CRITICAL): a peak-joint-SPEED reward-hack — rewards a single
# fast joint spike, so it scores `chaotic` (upright random thrashing, the
# highest-peak archetype) above the real positives. Must be rejected.
PEAK_SPEED_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); g = arrays.get("projected_gravity_b")
    if jv is None or g is None:
        return {"spec_score": 0.0}
    n = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    up = float((n[..., 2] < -0.85).mean())
    return {"spec_score": float(up * (1.0 - np.exp(-np.abs(jv).max() / 8.0)))}
'''

# §Ship 47: a kick metric that rewards upright leg-velocity bursts but does
# NOT gate on a stationary base — the gen_005 failure that stalled g1-kick-v3.
# It scores a forward WALKER high (gait hip/knee swings look like bursts), so
# the walker ceiling must reject it for the kick family.
WALKER_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    if jv is None or grav is None:
        return {"spec_score": 0.0}
    names = (meta or {}).get("joint_names", []) if isinstance(meta, dict) else []
    legs = [i for i, n in enumerate(names)
            if "hip" in n.lower() or "knee" in n.lower()]
    sub = jv[..., legs] if (legs and len(names) == jv.shape[2]) else jv
    up = float(np.mean(grav[..., 2] < -0.85))
    peak = float(np.abs(sub).max())
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-peak / 6.0)), 0.0, 1.0))}
'''

# §Metric-quality laws (LAW 4): a kick metric that gates on stationarity +
# uprightness + leg burst but does NOT read foot DIRECTION — it scores a
# rearward "mule kick" identically to a forward one, so it is gameable by the
# g1-kick-v5 "kicks behind it" hack. The kick-family direction gate must reject
# it (active_kick_behind scores == active_kick).
DIRECTION_BLIND_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w"); jp = arrays.get("joint_pos")
    if jv is None or grav is None or root is None or jp is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) if isinstance(meta, dict) else {}
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    xy = root[..., :2]
    drift = float(np.linalg.norm(xy.max(0) - xy.min(0), axis=-1).mean())
    stationary = float(np.exp(-drift / 0.4))
    zmed = float(np.median(root[..., 2]))
    height = float(np.clip((zmed - 0.45) / 0.20, 0.0, 1.0))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    return {"spec_score": float(np.clip(up * stationary * height * burst, 0.0, 1.0))}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_validate_good_metric_passes(tmp_path):
    p = _write(tmp_path, "good.py", GOOD)
    v = validate_generated_metric(GOOD, p)
    assert v["ok"], v["reasons"]
    assert all(v["gates"].values())
    # active archetype must outscore still/fallen.
    s = v["archetype_scores"]
    assert s["active"] > s["still"] and s["active"] > s["fallen"]


def test_validate_rejects_forbidden_import(tmp_path):
    p = _write(tmp_path, "bi.py", BAD_IMPORT)
    v = validate_generated_metric(BAD_IMPORT, p)
    assert not v["ok"]
    assert v["gates"]["ast_safety"] is False
    assert any("os" in r for r in v["reasons"])


def test_validate_rejects_builtins_escape(tmp_path):
    p = _write(tmp_path, "be.py", BUILTINS_ESCAPE)
    v = validate_generated_metric(BUILTINS_ESCAPE, p)
    assert not v["ok"]
    assert v["gates"]["ast_safety"] is False
    assert any("__builtins__" in r for r in v["reasons"])


def test_validate_rejects_unavailable_array(tmp_path):
    p = _write(tmp_path, "ba.py", BAD_ARRAY)
    v = validate_generated_metric(BAD_ARRAY, p)
    assert not v["ok"]
    assert v["gates"]["array_contract"] is False
    assert any("contact_forces" in r for r in v["reasons"])


def test_validate_rejects_rewarding_stillness(tmp_path):
    p = _write(tmp_path, "still.py", REWARDS_STILLNESS)
    v = validate_generated_metric(REWARDS_STILLNESS, p)
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False


def test_validate_rejects_flail_rewarder(tmp_path):
    """§Ship 36: a motion-magnitude metric that scores the stand-and-flail
    hack as high as competent motion fails the new upright_flail gate."""
    p = _write(tmp_path, "flail.py", FLAIL_REWARDER)
    v = validate_generated_metric(FLAIL_REWARDER, p)
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False
    assert any("upright_flail" in r for r in v["reasons"]), v["reasons"]
    assert "upright_flail" in v["archetype_scores"]


def test_validate_rejects_nondeterministic(tmp_path):
    p = _write(tmp_path, "nd.py", NONDETERMINISTIC)
    v = validate_generated_metric(NONDETERMINISTIC, p)
    assert not v["ok"]
    assert v["gates"]["determinism"] is False


# ── §Ship 41: behavior-family non-degeneracy ─────────────────────────


def test_resolve_behavior_family():
    from sculptor.eval.metric_validate import resolve_behavior_family as rf

    assert rf("Repeatedly kick forward with one leg in sharp strikes") == "kick"
    assert rf("perform a continuous flossing dance") == "floss"
    assert rf("jump as high as you can, repeatedly") == "jump"
    assert rf("trot forward in a straight line") == "locomotion"
    assert rf("balance the pole upright") == "cartpole"
    assert rf("do a fancy spin move") is None
    # robot-family fallback for a quadruped with no behavior token
    assert rf("move", robot_hint="Mjlab-Velocity-Flat-Unitree-Go1") == "locomotion"
    # §Ship 41 review: WORD matching, not substring — "Hopper" must NOT be jump
    assert rf("Canonical Hopper-v4 reward: forward_velocity + alive_bonus") == "locomotion"
    assert rf("bound forward across the floor", robot_hint="Go1") == "locomotion"
    assert rf("strike a balance and stay centered") == "cartpole"


def test_validate_rejects_peak_speed_hack(tmp_path):
    """§Ship 41 review (CRITICAL fix): a peak-joint-speed reward-hack scores the
    `chaotic` archetype (random thrash) highest; chaotic is now a required-loser
    so the hack is rejected under EVERY goal (matched + unmatched)."""
    p = _write(tmp_path, "peak.py", PEAK_SPEED_HACK)
    for goal in (None, "do a spin", "kick forward", "jump high"):
        v = validate_generated_metric(PEAK_SPEED_HACK, p, behavior_goal=goal)
        assert not v["ok"], (goal, v["archetype_scores"])
        assert v["gates"]["nondegeneracy"] is False
        assert any("chaotic" in r for r in v["reasons"]), v["reasons"]


def test_calibrate_rejects_subresolution_joint_pos_drift(tmp_path):
    """§Ship 41 review (HIGH fix): a degenerate metric reading joint_pos
    magnitude drifts ~1e-7 across the enriched (cumsum joint_pos) ladders; the
    round-before-guard in spearman() must keep it from spuriously calibrating."""
    from sculptor.eval.metric_calibration import calibrate_metric

    src = ('import numpy as np\n'
           'def compute_spec(arrays, behavior, meta):\n'
           '    jp = arrays.get("joint_pos")\n'
           '    if jp is None: return {"spec_score": 0.0}\n'
           '    return {"spec_score": float(1.0 - np.exp(-float(np.mean(np.abs(jp))) / 1e6))}\n')
    p = _write(tmp_path, "drift.py", src)
    for builtin in ("g1_kick", "g1_floss", "g1_jump"):
        cal = calibrate_metric(p, builtin, threshold=0.7)
        assert not cal["ok"], (builtin, cal)


def test_validate_kick_metric_passes_with_family(tmp_path):
    """§Ship 41: a correct STATIONARY kick metric passes non-degeneracy when
    the kick family is resolved — and the locomotion `active` anchor is ~0 for
    it (exactly why the old single-archetype gate false-rejected it)."""
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    v = validate_generated_metric(
        GOOD_KICK, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["ok"], v["reasons"]
    assert v["family"] == "kick"
    s = v["archetype_scores"]
    assert s["active"] < 0.1 < s["active_kick"], s


def test_validate_jump_metric_passes_with_family(tmp_path):
    p = _write(tmp_path, "jump.py", GOOD_JUMP)
    v = validate_generated_metric(GOOD_JUMP, p, behavior_goal="jump repeatedly")
    assert v["ok"], v["reasons"]
    assert v["family"] == "jump"
    assert v["archetype_scores"]["active_jump"] > 0.3, v["archetype_scores"]


def test_validate_floss_metric_passes_with_family(tmp_path):
    p = _write(tmp_path, "floss.py", GOOD_FLOSS)
    v = validate_generated_metric(
        GOOD_FLOSS, p, behavior_goal="flossing dance, arms in opposition")
    assert v["ok"], v["reasons"]
    assert v["family"] == "floss"


# ── §Ship 47: forward-walker Goodhart (the g1-kick-v3 0.59 failure) ──────


def test_walker_archetype_present_and_caught_for_kick(tmp_path):
    """A kick metric that rewards leg bursts WITHOUT a stationarity gate
    (the gen_005 failure) scores the forward `walker` archetype high; the
    family-scoped ceiling rejects it even though its `active_kick` score
    keeps it above every relative negative."""
    p = _write(tmp_path, "walkhack.py", WALKER_HACK)
    v = validate_generated_metric(
        WALKER_HACK, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["family"] == "kick"
    assert "walker" in v["archetype_scores"]
    assert v["archetype_scores"]["walker"] > 0.3, v["archetype_scores"]
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False
    assert any("walker" in r for r in v["reasons"]), v["reasons"]


def test_good_kick_metric_clears_walker_ceiling(tmp_path):
    """The correct stationary kick metric gates on a stationary base, so it
    scores the forward walker BELOW the ceiling — the walker check must not
    false-reject it."""
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    v = validate_generated_metric(
        GOOD_KICK, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["archetype_scores"]["walker"] < 0.3, v["archetype_scores"]
    assert v["ok"], v["reasons"]


def test_validate_rejects_direction_blind_kick(tmp_path):
    """§LAW 4: a kick metric that ignores foot DIRECTION scores a rearward kick
    identically to a forward one — the kick-family gate rejects it (the
    'kicks behind it' g1-kick-v5 hack a stationarity-only metric can't catch)."""
    p = _write(tmp_path, "blind.py", DIRECTION_BLIND_KICK)
    v = validate_generated_metric(
        DIRECTION_BLIND_KICK, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["family"] == "kick"
    s = v["archetype_scores"]
    assert s["active_kick_behind"] == pytest.approx(s["active_kick"]), s
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False
    assert any("active_kick_behind" in r for r in v["reasons"]), v["reasons"]


def test_good_kick_clears_direction_and_balance_hacks(tmp_path):
    """§LAW 1/2/4: the direction-aware GOOD_KICK scores every documented v5 hack
    (rear kick, one-leg balance, partial twitch) BELOW its forward-kick positive
    and is accepted for the kick family."""
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    v = validate_generated_metric(
        GOOD_KICK, p, behavior_goal="repeatedly kick forward with one leg")
    s = v["archetype_scores"]
    for hack in ("active_kick_behind", "one_leg_balance", "partial_kick"):
        assert s[hack] < s["active_kick"], (hack, s)
    assert s["active_kick_behind"] < 0.1, s    # direction kills the rear kick
    assert v["ok"], v["reasons"]


# ── novel-task non-degeneracy: vacuous-pass selectivity probe ─────────
# A SELECTIVE novel-task metric (no behavior family) that the fixed archetype
# battery cannot represent — e.g. toe-touch / squat gated on a pelvis DIP-AND-
# RETURN no archetype performs — used to be false-rejected as "near-constant"
# (every archetype 0.0), leaving the run blind. The selectivity probe rescues it.
TOE_TOUCH_NOVEL = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos"); root = arrays.get("root_link_pos_w")
    pg = arrays.get("projected_gravity_b")
    if jp is None or root is None or pg is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) or {}
    idx = [roles[r] for r in ("left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee") if r in roles]
    if not idx:
        return {"spec_score": 0.0}
    z = root[..., 2]
    drop = float(np.mean(z[:5].mean(axis=0) - z.min(axis=0)))               # pelvis dips
    ret = float(np.mean(np.abs(z[-5:].mean(axis=0) - z[:5].mean(axis=0))))  # and returns
    rom = float(np.mean(np.max(jp[..., idx], axis=0) - np.min(jp[..., idx], axis=0)))
    up0 = float(np.mean(pg[:5, ..., 2].mean(axis=0) < -0.85))
    gate = 1.0
    gate *= 1.0 if drop > 0.20 else 0.0
    gate *= 1.0 if ret < 0.08 else 0.0
    gate *= 1.0 if up0 > 0.7 else 0.0
    amp = float(np.clip(rom / 1.2, 0.0, 1.0))
    return {"spec_score": float(np.clip(gate * amp, 0.0, 1.0))}
'''

ALL_ZERO = '''def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.0}
'''


def test_validate_novel_selective_metric_passes_vacuously(tmp_path):
    """A correct novel-task metric (toe-touch, gated on a pelvis dip-and-return)
    scores EVERY fixed archetype ~0 because none performs the goal. It must still
    PASS — vacuously — because it is selective on the goal-agnostic probe, instead
    of being false-rejected as 'near-constant' (which leaves the run blind)."""
    p = _write(tmp_path, "toe.py", TOE_TOUCH_NOVEL)
    v = validate_generated_metric(
        TOE_TOUCH_NOVEL, p, behavior_goal="touch your toes then stand back up",
        robot_hint="unitree_g1")
    assert v["family"] is None                        # genuinely novel (no family)
    assert all(abs(s) <= 1e-3 for s in v["archetype_scores"].values()), v["archetype_scores"]
    assert v["gates"]["nondegeneracy"] is True, v["reasons"]
    assert v["nondegeneracy_vacuous"] is True
    assert v["ok"], v["reasons"]


def test_validate_allzero_rejected_not_vacuous(tmp_path):
    """The vacuous-pass door does NOT re-open the all-zero hole: a constant-0
    metric is non-selective on the probe → rejected, and NOT marked vacuous."""
    p = _write(tmp_path, "zero.py", ALL_ZERO)
    v = validate_generated_metric(
        ALL_ZERO, p, behavior_goal="touch your toes then stand back up")
    assert v["gates"]["nondegeneracy"] is False
    assert v["nondegeneracy_vacuous"] is False
    assert not v["ok"]
    assert any("not selective on the probe" in r for r in v["reasons"]), v["reasons"]


def test_validate_family_metric_not_vacuous(tmp_path):
    """Byte-identical family path: a kick metric resolves a family, its positive
    archetype lights up, so the battery is informative and the vacuous branch is
    never entered (nondegeneracy_vacuous stays False)."""
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    v = validate_generated_metric(
        GOOD_KICK, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["family"] == "kick"
    assert v["nondegeneracy_vacuous"] is False
    assert v["ok"], v["reasons"]


def test_selectivity_probe_is_deterministic():
    """The probe is pure-numpy + offline: identical across calls (same discipline
    as the determinism gate), and a dip-rewarding fn separates competent from
    degenerate (the property the vacuous pass relies on)."""
    import numpy as np

    from sculptor.eval.metric_validate import _NAMES_12, _selectivity_probe

    def fn(arrays, behavior, meta):
        z = arrays["root_link_pos_w"][..., 2]
        drop = float(np.mean(z[:5].mean(axis=0) - z.min(axis=0)))
        return {"spec_score": float(np.clip(drop / 0.35, 0.0, 1.0))}

    meta = {"joint_names": list(_NAMES_12)}
    a = _selectivity_probe(fn, meta)
    b = _selectivity_probe(fn, meta)
    assert a == b                                     # deterministic
    assert a["competent"] >= 0.1 and a["spread"] >= 0.1   # dip lights up, degenerate ~0


def test_resolve_goal_frame_keywords():
    """§LAW 0: the goal frame resolves direction / support / torso from the NL
    goal, with SAFE forward+double+upright defaults for a plain kick."""
    from sculptor.eval.metric_validate import resolve_goal_frame

    assert resolve_goal_frame("repeatedly kick forward with one leg") == {
        "goal_axis": "+x", "support_mode": "double", "torso_target": "upright"}
    assert resolve_goal_frame("do a mule kick backward")["goal_axis"] == "-x"
    assert resolve_goal_frame(
        "balance on one foot like a flamingo")["support_mode"] == "single"
    assert resolve_goal_frame("jump as high as you can")["support_mode"] == "flight"
    assert resolve_goal_frame("do a fancy spin")["goal_axis"] is None
    assert resolve_goal_frame("do a backflip")["torso_target"] == "horizontal"


def test_kick_direction_gate_abstains_for_rearward_goal(tmp_path):
    """§LAW 0: the rear-kick direction gate fires for a FORWARD kick goal but
    ABSTAINS for a rearward (mule) kick — so a metric is not false-rejected on
    direction for a goal whose competent motion IS rearward. A direction-blind
    metric is the probe: it trips the gate forward, not rearward."""
    p = _write(tmp_path, "blind.py", DIRECTION_BLIND_KICK)
    fwd = validate_generated_metric(DIRECTION_BLIND_KICK, p, behavior_goal="kick forward")
    rear = validate_generated_metric(
        DIRECTION_BLIND_KICK, p, behavior_goal="do a mule kick backward")
    assert fwd["goal_frame"]["goal_axis"] == "+x"
    assert rear["goal_frame"]["goal_axis"] == "-x"
    assert any("active_kick_behind" in r for r in fwd["reasons"])
    assert not any("active_kick_behind" in r for r in rear["reasons"])


def test_walker_ceiling_skipped_for_locomotion(tmp_path):
    """The walker ceiling is scoped to stationary skills — a locomotion metric
    (for which a walker IS the target) is NOT rejected by it."""
    p = _write(tmp_path, "loco.py", GOOD)
    v = validate_generated_metric(GOOD, p, behavior_goal="trot forward in a straight line")
    assert v["family"] == "locomotion"
    assert v["archetype_scores"]["walker"] > 0.3, v["archetype_scores"]  # walker scores high…
    assert v["ok"], v["reasons"]                                          # …and that's fine


def test_validate_flail_rejected_under_kick_family(tmp_path):
    """§Ship 41 anti-regression: the stand-and-flail hack stays rejected even
    under a non-locomotion family — the negatives are unchanged HARD anchors."""
    p = _write(tmp_path, "flail.py", FLAIL_REWARDER)
    v = validate_generated_metric(FLAIL_REWARDER, p, behavior_goal="kick forward")
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False
    assert any("upright_flail" in r for r in v["reasons"]), v["reasons"]


def test_validate_stillness_rejected_unmatched(tmp_path):
    """§Ship 41: with NO family matched (the richer default battery incl. the
    jump positive, which has zero base-motion), a stillness-rewarder is STILL
    rejected — a quiet stance must not anchor the metric."""
    p = _write(tmp_path, "still.py", REWARDS_STILLNESS)
    v = validate_generated_metric(REWARDS_STILLNESS, p, behavior_goal="do a spin")
    assert not v["ok"]
    assert v["gates"]["nondegeneracy"] is False


def test_calibrate_4array_metric_now_passes(tmp_path):
    """§Ship 41: a kick metric that needs root_link_pos_w/joint_pos scored 0 on
    every (2-array) g1_kick ladder rung → Spearman 0 → never steered. The
    enriched ladder lets it calibrate."""
    from sculptor.eval.metric_calibration import calibrate_metric

    p = _write(tmp_path, "kick.py", GOOD_KICK)
    cal = calibrate_metric(p, "g1_kick", threshold=0.7)
    assert cal["ok"] and cal["spearman"] >= 0.7, cal
    assert any(s > 0 for s in cal["gen_scores"]), cal  # not all-zero anymore


def test_compute_and_resolve_generated_metric(tmp_path):
    p = _write(tmp_path, "good.py", GOOD)
    # synthetic rollout dir: forward-travelling, upright.
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    T, Ee = 50, 4
    root = np.zeros((T, Ee, 3), dtype=np.float32); root[..., 2] = 0.5
    root[..., 0] = (np.arange(T) * 0.05)[:, None]
    grav = np.zeros((T, Ee, 3), dtype=np.float32); grav[..., 2] = -1.0
    np.savez(rollout / "trajectory.npz", root_link_pos_w=root,
             projected_gravity_b=grav)
    (rollout / "behavior.json").write_text(json.dumps({"step_dt": 0.02}), encoding="utf-8")

    out = compute_generated_metric(p, rollout)
    assert 0.0 <= out["spec_score"] <= 1.0 and out["spec_score"] > 0.5
    # fitness fn scores iter_dir/rollout.
    fit = make_generated_fitness_fn(p)
    assert fit(tmp_path) == pytest.approx(out["spec_score"])
    # resolver dispatches a .py path to the generated fitness fn.
    assert resolve_fitness_fn(str(p))(tmp_path) == pytest.approx(out["spec_score"])
    # missing rollout → honest 0.0, never raises.
    assert make_generated_fitness_fn(p)(tmp_path / "nope") == 0.0


# ── generator (mock LLM) + calibration ────────────────────────────────

CONSTANT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''


class _Block:
    def __init__(self, text): self.type = "text"; self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _Parsed:
    def __init__(self, parsed): self.parsed_output = parsed


class _Messages:
    def __init__(self, src, approved, gaming_exploit=""):
        self._src = src; self._approved = approved
        self._gaming_exploit = gaming_exploit; self.creates = 0
    def create(self, **kw):
        self.creates += 1
        return _Resp("```python\n" + self._src + "\n```")
    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        return _Parsed(MetricReview(approved=self._approved,
                                    concerns=[] if self._approved else ["gameable"],
                                    summary="x", gaming_exploit=self._gaming_exploit))


class _FakeClient:
    def __init__(self, src, approved=True, gaming_exploit=""):
        self.messages = _Messages(src, approved, gaming_exploit)


def test_generate_objective_metric_accepts_good(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "m"
    rec = generate_objective_metric(
        "trot forward", out, client=_FakeClient(GOOD, approved=True), max_attempts=2)
    assert rec["accepted"] and rec["validation_passed"]
    assert rec["review"]["approved"]
    assert (out / "metric.py").is_file() and (out / "meta.json").is_file()
    assert rec["calibrated"] is False  # steer-rights not granted yet


def test_generate_objective_metric_rejects_bad_and_retries(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    client = _FakeClient(REWARDS_STILLNESS, approved=True)
    rec = generate_objective_metric(
        "trot forward", tmp_path / "m2", client=client, max_attempts=3)
    assert not rec["accepted"] and not rec["validation_passed"]
    assert client.messages.creates == 3  # retried up to the cap on failure


def test_generate_objective_metric_review_can_veto(tmp_path):
    from sculptor.eval.metric_gen import generate_objective_metric
    # validation passes but the reviewer rejects → not accepted.
    rec = generate_objective_metric(
        "trot forward", tmp_path / "m3",
        client=_FakeClient(GOOD, approved=False), max_attempts=1)
    assert rec["validation_passed"] and not rec["accepted"]


def test_single_review_named_exploit_forces_reject(tmp_path):
    """§LAW 9: on the PRODUCTION single-reviewer path (review_models unset — what
    sculptor_bridge / the CLI actually use), a reviewer that NAMES a gaming_exploit
    yet returns approved=True is coerced to a reject, just like the panel path. The
    'named exploit but approved' contradiction must not slip through on the path
    every production caller takes."""
    from sculptor.eval.metric_gen import generate_objective_metric
    rec = generate_objective_metric(
        "trot forward", tmp_path / "se",
        client=_FakeClient(GOOD, approved=True, gaming_exploit="stand and flail"),
        max_attempts=1)
    assert rec["validation_passed"] and not rec["accepted"]
    assert not rec["review"]["approved"]
    assert any("stand and flail" in c for c in rec["review"]["concerns"])


def test_generate_emits_progress_events(tmp_path):
    """§Ship 40: on_event streams the pipeline stages so the UI can show
    live progress (generate → validate → review → done)."""
    from sculptor.eval.metric_gen import generate_objective_metric
    events: list = []
    generate_objective_metric(
        "trot forward", tmp_path / "mp", client=_FakeClient(GOOD, approved=True),
        max_attempts=2, on_event=events.append)
    stages = [e["stage"] for e in events]
    assert stages[0] == "generating"
    assert "validating" in stages
    assert "reviewing" in stages                 # GOOD passes validation → review
    assert stages[-1] == "done" and events[-1]["accepted"] is True
    assert all(e.get("message") for e in events)  # every event is human-readable


def test_generate_emits_regenerating_on_validation_failure(tmp_path):
    """§Ship 40: a failing candidate emits a `regenerating` stage each retry
    and never reaches `reviewing`."""
    from sculptor.eval.metric_gen import generate_objective_metric
    events: list = []
    generate_objective_metric(
        "trot forward", tmp_path / "mp2",
        client=_FakeClient(REWARDS_STILLNESS, approved=True),
        max_attempts=3, on_event=events.append)
    stages = [e["stage"] for e in events]
    assert stages.count("generating") == 3       # all attempts
    assert "regenerating" in stages
    assert "reviewing" not in stages             # never passed validation
    assert stages[-1] == "done" and events[-1]["accepted"] is False


# ── §best-of-N: sample N candidates, select the most-discriminating valid one ──
# Opt-in n_candidates>1 samples N candidates and selects the most-discriminating
# VALID one via the OFFLINE graded discriminator; default 1 is byte-identical.

# A binary "did the pelvis dip" metric: passes validation (selective on the probe)
# but does NOT grade — every competent rung scores 1.0, so its discrimination is
# LOWER than a smoothly-grading toe-touch (no monotonic separation across rungs).
COARSE_DIP = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]; drop = float(np.mean(z[:6].mean(0) - z.min(0)))
    return {"spec_score": 1.0 if drop > 0.20 else 0.0}
'''


class _CycleMessages:
    """Serves a different source per .create call (best-of-N) and records the
    temperature / thinking / user-prompt of each call; the reviewer always approves."""

    def __init__(self, srcs):
        self._srcs = list(srcs); self.creates = 0
        self.temps: list = []; self.thinking: list = []; self.users: list = []

    def create(self, **kw):
        self.temps.append(kw.get("temperature"))
        self.thinking.append(kw.get("thinking"))
        self.users.append(kw.get("messages", [{}])[-1].get("content", ""))
        src = self._srcs[self.creates % len(self._srcs)]; self.creates += 1
        return _Resp("```python\n" + src + "\n```")

    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        return _Parsed(MetricReview(approved=True, concerns=[], summary="ok"))


class _CycleClient:
    def __init__(self, *srcs): self.messages = _CycleMessages(srcs)


def test_best_of_n_selects_most_discriminating(tmp_path):
    """KEYSTONE: COARSE (binary dip, low discrimination) is sampled FIRST and a
    smoothly-grading toe-touch (high discrimination) SECOND — best-of-N selects the
    SECOND, proving it ranks by the offline discriminator, not first-valid."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "bon"
    client = _CycleClient(COARSE_DIP, TOE_TOUCH_NOVEL)
    rec = generate_objective_metric(
        "touch your toes then stand back up", out, robot_hint="unitree_g1",
        client=client, n_candidates=2)
    assert rec["accepted"] and rec["selected_candidate"] == 1
    assert (out / "metric.py").read_text().strip() == TOE_TOUCH_NOVEL.strip()
    discs = {c["candidate"]: c.get("discrimination") for c in rec["candidates"]}
    assert discs[1] > discs[0]                    # the selected one is more discriminating
    # REGRESSION: the generator uses extended thinking, which the API rejects for any
    # temperature != 1.0 (a 400 that fails the call instantly). best-of-N must NOT vary
    # temperature — it decorrelates by FRAMING. So every candidate uses temp 1.0 with
    # thinking on, and the per-candidate prompts differ.
    assert client.messages.temps == [1.0, 1.0]
    assert all(t == {"type": "adaptive"} for t in client.messages.thinking)
    assert client.messages.users[0] != client.messages.users[1]   # framing decorrelation
    assert not list(out.glob("candidate_*.py"))   # non-winner files cleaned up


class _SelectiveReviewMessages(_CycleMessages):
    """Like _CycleMessages but the reviewer REJECTS any source containing `reject_marker`
    and approves the rest — to test best-of-N review-in-order."""

    def __init__(self, srcs, reject_marker):
        super().__init__(srcs); self._marker = reject_marker; self.reviewed: list = []

    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        content = kw.get("messages", [{}])[-1].get("content", "")
        self.reviewed.append(content)
        rejected = self._marker in content
        return _Parsed(MetricReview(approved=not rejected,
                                    concerns=["gameable"] if rejected else [],
                                    summary="x"))


class _SelectiveReviewClient:
    def __init__(self, srcs, reject_marker):
        self.messages = _SelectiveReviewMessages(srcs, reject_marker)


def test_best_of_n_reviews_in_order_and_accepts_approved(tmp_path):
    """KEYSTONE: the most-discriminating valid candidate is REVIEW-REJECTED, but a
    less-discriminating valid sibling is APPROVED — best-of-N reviews best-first and
    ACCEPTS the approved one (promoting it to the result), instead of letting the
    discrimination-winner's review veto sink the whole run."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "rio"
    # cand 0 = TOE_TOUCH_NOVEL (disc ~2.0, reviewed FIRST) — REJECTED via its marker;
    # cand 1 = COARSE_DIP (disc ~1.0, valid) — APPROVED.
    client = _SelectiveReviewClient([TOE_TOUCH_NOVEL, COARSE_DIP],
                                    reject_marker="REQUIRED_JOINT_ROLES")
    rec = generate_objective_metric(
        "touch your toes then stand back up", out, robot_hint="unitree_g1",
        client=client, n_candidates=2)
    assert rec["accepted"], rec                          # the approved sibling carried it
    assert rec["selected_candidate"] == 1                # the lower-disc, review-APPROVED one
    assert (out / "metric.py").read_text().strip() == COARSE_DIP.strip()
    assert len(client.messages.reviewed) == 2            # rejected top, then approved next


def test_review_feedback_retry_recovers(tmp_path):
    """A metric that passes validation but the reviewer VETOES is no longer a dead-end:
    the reviewer's concerns are fed back, a corrected metric is generated + re-reviewed,
    and an accepted metric is recovered (the review-failure analogue of the validation-
    feedback retry)."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "rfr"
    # 1st sample is reviewer-REJECTED (marker); the review-retry's sample is APPROVED.
    client = _SelectiveReviewClient([TOE_TOUCH_NOVEL, COARSE_DIP],
                                    reject_marker="REQUIRED_JOINT_ROLES")
    rec = generate_objective_metric(
        "touch your toes then stand back up", out, robot_hint="unitree_g1",
        client=client, n_candidates=1)
    assert rec["accepted"], rec                       # the review-feedback retry recovered
    assert (out / "metric.py").read_text().strip() == COARSE_DIP.strip()
    assert len(client.messages.reviewed) == 2         # rejected, then approved after feedback
    assert any(a.get("review_retry") for a in rec["attempts"])    # a review-retry happened


def test_review_veto_gives_up_after_bounded_retries(tmp_path):
    """A persistently-vetoed metric stops after _MAX_REVIEW_RETRIES (no infinite loop):
    not accepted, and the bounded retries are recorded."""
    from sculptor.eval.metric_gen import _MAX_REVIEW_RETRIES, generate_objective_metric
    out = tmp_path / "rvg"
    rec = generate_objective_metric(
        "trot forward", out, client=_FakeClient(GOOD, approved=False), n_candidates=1)
    assert rec["validation_passed"] and not rec["accepted"]
    assert sum(1 for a in rec["attempts"] if a.get("review_retry")) == _MAX_REVIEW_RETRIES


def test_best_of_n_ties_keep_first_valid(tmp_path):
    """Equal discrimination → the FIRST valid candidate (lowest index) wins, preserving
    today's first-valid behavior."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "tie"
    client = _CycleClient(TOE_TOUCH_NOVEL, TOE_TOUCH_NOVEL)
    rec = generate_objective_metric(
        "touch your toes then stand back up", out, robot_hint="unitree_g1",
        client=client, n_candidates=2)
    assert rec["accepted"] and rec["selected_candidate"] == 0


def test_n_candidates_one_is_byte_identical(tmp_path):
    """Default n_candidates=1 → the unchanged single-shot path: one author call, no
    candidate records, no selection (the only diff is three inert record keys)."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "one"
    client = _FakeClient(GOOD, approved=True)
    rec = generate_objective_metric("trot forward", out, client=client,
                                    n_candidates=1, max_attempts=2)
    assert rec["accepted"]
    assert rec["candidates"] is None and rec["selected_candidate"] is None
    assert rec["n_candidates"] == 1
    assert client.messages.creates == 1


def test_best_of_n_all_invalid_rejects(tmp_path):
    """All N candidates fail validation → not accepted, run stays blind (no metric can
    steer), as the single path. Each candidate failure is recorded and metric.py is
    still written (the contract)."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "bad"
    client = _CycleClient(REWARDS_STILLNESS, REWARDS_STILLNESS, REWARDS_STILLNESS)
    rec = generate_objective_metric("trot forward", out, robot_hint="go1",
                                    client=client, n_candidates=3)
    assert not rec["accepted"] and not rec["validation_passed"]
    assert rec["selected_candidate"] is None
    assert len(rec["candidates"]) == 3 and all(not c["ok"] for c in rec["candidates"])
    assert (out / "metric.py").is_file()
    assert not list(out.glob("candidate_*.py"))   # cleaned up even on all-invalid


def test_best_of_n_all_api_error_surfaces_reason(tmp_path):
    """When EVERY candidate's generation call fails (e.g. a 400), best-of-N surfaces a
    SPECIFIC reason (never a silent 'rejected') so the UI shows WHY — the failure mode
    that masked the temperature/thinking 400 the user hit."""
    from sculptor.eval.metric_gen import generate_objective_metric

    class _BoomMessages:
        def create(self, **kw):
            raise RuntimeError("Error code: 400 - temperature may only be set to 1 "
                               "when thinking is enabled")

        def parse(self, **kw):
            from sculptor.eval.metric_gen import MetricReview
            return _Parsed(MetricReview(approved=True))

    class _BoomClient:
        def __init__(self): self.messages = _BoomMessages()

    out = tmp_path / "boom"
    rec = generate_objective_metric("touch your toes then stand back up", out,
                                    robot_hint="unitree_g1", client=_BoomClient(),
                                    n_candidates=3)
    assert not rec["accepted"] and not rec["validation_passed"]
    assert rec["selected_candidate"] is None
    reasons = (rec["validation"] or {}).get("reasons") or []
    assert any("API error" in r and "thinking" in r for r in reasons), reasons
    assert len(rec["candidates"]) == 3 and all(c.get("api_error") for c in rec["candidates"])


def test_best_of_n_falls_back_to_feedback_retry(tmp_path):
    """best-of-N is NEVER WORSE than single-shot: when all N candidates fail validation,
    it falls back to the retry-WITH-FEEDBACK loop (seeded with the candidate failures), so
    a goal the LLM only nails after correction still succeeds — instead of best-of-N
    trading away the feedback loop for diversity and rejecting."""
    from sculptor.eval.metric_gen import generate_objective_metric
    out = tmp_path / "fb"
    # 2 invalid candidates, then the VALID source the feedback-retry produces.
    client = _CycleClient(REWARDS_STILLNESS, REWARDS_STILLNESS, GOOD)
    rec = generate_objective_metric("trot forward", out, robot_hint="go1",
                                    client=client, n_candidates=2)
    assert rec["accepted"] and rec["validation_passed"]   # the fallback rescued it
    assert len(rec["candidates"]) == 2                     # the 2 failed candidates recorded
    assert rec["selected_candidate"] is None               # no best-of-N winner
    assert client.messages.creates >= 3                    # 2 candidates + ≥1 fallback
    # the fallback was SEEDED with the candidate failures (real feedback retry).
    assert any("FAILED these validation gates" in u for u in client.messages.users)


def test_sample_source_raises_on_max_tokens_truncation():
    """A max_tokens-truncated response (adaptive thinking ate the budget) raises a CLEAR
    'truncated' error instead of returning incomplete code that fails downstream as a
    baffling 'missing compute_spec'. A normal (end_turn) response is unaffected."""
    import pytest

    from sculptor.eval.metric_gen import _sample_source

    class _Blk:
        type = "text"; text = "import numpy as np\ndef compute_sp"      # cut off

    class _Truncated:
        stop_reason = "max_tokens"; content = [_Blk()]

    class _ClientTrunc:
        messages = type("M", (), {"create": lambda self, **kw: _Truncated()})()

    with pytest.raises(RuntimeError, match="truncated at max_tokens"):
        _sample_source(_ClientTrunc(), "sys", "user", model="m")

    class _Ok:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text",
                   "text": "```python\ndef compute_spec(a, b, m): return {'spec_score': 0.0}\n```"})()]

    class _ClientOk:
        messages = type("M", (), {"create": lambda self, **kw: _Ok()})()

    assert "def compute_spec" in _sample_source(_ClientOk(), "sys", "user", model="m")


def test_graded_discrimination_ranks_and_is_deterministic():
    """The offline selector: a SHARP, smoothly-grading metric out-scores a COARSE
    binary one and a degenerate one, and is byte-deterministic across calls (it must
    add no nondeterminism to selection)."""
    import numpy as np

    from sculptor.eval.generated_metric import inject_joint_roles
    from sculptor.eval.metric_validate import _NAMES_12, graded_discrimination

    def _disc(src):
        ns: dict = {}; exec(src, ns)
        meta = {"joint_names": list(_NAMES_12)}
        inject_joint_roles(meta, ["left_hip_pitch", "right_hip_pitch",
                                  "left_knee", "right_knee"], lenient=True)
        return graded_discrimination(ns["compute_spec"], meta)

    sharp = _disc(TOE_TOUCH_NOVEL)
    coarse = _disc(COARSE_DIP)
    zero = _disc("def compute_spec(a, b, m): return {'spec_score': 0.0}\n")
    still = _disc("import numpy as np\ndef compute_spec(a, b, m):\n"
                  "    jv = a.get('joint_vel')\n"
                  "    if jv is None: return {'spec_score': 0.0}\n"
                  "    return {'spec_score': float(np.exp(-np.mean(np.abs(jv)) / 0.1))}\n")
    assert sharp["score"] > coarse["score"] > zero["score"]
    assert still["score"] < zero["score"]         # a stillness-rewarder is anti-graded
    assert sharp["fold_axis"]["monotonicity"] == 1.0
    assert _disc(TOE_TOUCH_NOVEL) == sharp        # deterministic


def test_graded_discrimination_is_amplitude_invariant():
    """The selector clamps spec_score to its [0,1] contract, so a high-AMPLITUDE coarse
    metric (a binary gate returning 5.0) saturates to [1,1,1] and CANNOT out-rank a
    smooth monotone grader on raw output scale — amplitude is an author artifact, not
    discrimination (the review's confirmed mis-rank)."""
    import numpy as np

    from sculptor.eval.generated_metric import inject_joint_roles
    from sculptor.eval.metric_validate import _NAMES_12, graded_discrimination

    def _disc(src):
        ns: dict = {}; exec(src, ns)
        meta = {"joint_names": list(_NAMES_12)}
        inject_joint_roles(meta, ["left_hip_pitch", "right_hip_pitch",
                                  "left_knee", "right_knee"], lenient=True)
        return graded_discrimination(ns["compute_spec"], meta)

    big_binary = _disc("import numpy as np\ndef compute_spec(a, b, m):\n"
                       "    root = a.get('root_link_pos_w')\n"
                       "    if root is None: return {'spec_score': 0.0}\n"
                       "    z = root[..., 2]; drop = float(np.mean(z[:6].mean(0) - z.min(0)))\n"
                       "    return {'spec_score': 5.0 if drop > 0.2 else 0.0}\n")
    smooth = _disc(TOE_TOUCH_NOVEL)
    assert big_binary["fold_axis"]["separation"] <= 1.0     # clamped to the contract
    assert big_binary["fold_axis"]["monotonicity"] == 0.0   # binary → saturates, no grading
    assert smooth["score"] > big_binary["score"]            # smooth monotone wins


# ── §Ship 55 (LAW 9): VLM metric-review Panel A ───────────────────────────
# A diversified, blinded, multi-model review panel. The mock routes each
# (model, lens) to a configured verdict and records (model, system, messages)
# so the tests can assert blinding + image use without any network call.


def _sys_text(system) -> str:
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in (system or []) if isinstance(b, dict))


def _msgs_text(messages) -> str:
    out = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(out)


def _detect_lens(system) -> str | None:
    from sculptor.eval.metric_gen import _LENSES
    t = _sys_text(system)
    for lens, appendix in _LENSES.items():
        if appendix and appendix in t:
            return lens
    return None


def _has_image(messages) -> bool:
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "image" for b in c):
            return True
    return False


class _PanelMessages:
    """`verdicts`: lens -> {approved, concerns, summary, gaming_exploit} or
    {"raise": True}; "*" is the default. Records every parse call."""

    def __init__(self, src, verdicts):
        self.src = src; self.verdicts = verdicts; self.calls = []; self.creates = 0

    def create(self, **kw):
        self.creates += 1
        return _Resp("```python\n" + self.src + "\n```")

    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        lens = _detect_lens(kw.get("system"))
        self.calls.append({"model": kw.get("model", ""), "lens": lens,
                           "system": _sys_text(kw.get("system")),
                           "messages": kw.get("messages"),
                           "has_image": _has_image(kw.get("messages"))})
        spec = self.verdicts.get(lens, self.verdicts.get("*", {"approved": True}))
        if spec.get("raise"):
            raise RuntimeError("reviewer api down")
        return _Parsed(MetricReview(
            approved=spec.get("approved", True), concerns=spec.get("concerns", []),
            summary=spec.get("summary", "x"), gaming_exploit=spec.get("gaming_exploit", "")))


class _PanelClient:
    def __init__(self, src, verdicts): self.messages = _PanelMessages(src, verdicts)


def _PANEL_MODELS():
    import sculptor.eval.metric_gen as mg
    return list(mg.REVIEW_PANEL_MODELS)


def test_reviewers_exclude_author():
    import sculptor.eval.metric_gen as mg
    r = mg._reviewers_from_models(list(mg.REVIEW_PANEL_MODELS), "claude-opus-4-8")
    assert all(m != "claude-opus-4-8" for m, _ in r)
    assert len(r) == len(mg._LENS_ORDER)


def test_panel_dispatch_is_byte_identical_on_default(tmp_path, monkeypatch):
    """review_models=None → the single _review_metric runs and the panel never
    does; a list flips it. Pins the dispatch so a panel can't silently fire on the
    default path."""
    import sculptor.eval.metric_gen as mg
    calls = {"single": 0, "panel": 0}
    real_single, real_panel = mg._review_metric, mg._review_metric_panel
    monkeypatch.setattr(mg, "_review_metric",
                        lambda *a, **k: (calls.__setitem__("single", calls["single"] + 1)
                                         or real_single(*a, **k)))
    monkeypatch.setattr(mg, "_review_metric_panel",
                        lambda *a, **k: (calls.__setitem__("panel", calls["panel"] + 1)
                                         or real_panel(*a, **k)))
    mg.generate_objective_metric("trot forward", tmp_path / "a",
                                 client=_FakeClient(GOOD, approved=True), max_attempts=1)
    assert calls == {"single": 1, "panel": 0}
    mg.generate_objective_metric("trot forward", tmp_path / "b",
                                 client=_PanelClient(GOOD, {"*": {"approved": True}}),
                                 review_models=_PANEL_MODELS(), max_attempts=1)
    assert calls == {"single": 1, "panel": 1}


def test_panel_review_out_keys_exact_both_paths(tmp_path):
    """The aggregate `review` stays EXACTLY {approved,concerns,summary} (the UI
    contract); panel provenance lives in the separate `review_panel`."""
    import sculptor.eval.metric_gen as mg
    rec1 = mg.generate_objective_metric("trot forward", tmp_path / "s",
                                        client=_FakeClient(GOOD, approved=True), max_attempts=1)
    assert set(rec1["review"]) == {"approved", "concerns", "summary"}
    assert rec1["review_panel"] is None
    rec2 = mg.generate_objective_metric("trot forward", tmp_path / "p",
                                        client=_PanelClient(GOOD, {"*": {"approved": True}}),
                                        review_models=_PANEL_MODELS(), max_attempts=1)
    assert set(rec2["review"]) == {"approved", "concerns", "summary"}
    assert rec2["accepted"] and rec2["review"]["approved"]
    assert rec2["review_panel"]["panel"] and "veto_by" in rec2["review_panel"]


def test_panel_one_lens_veto_rejects(tmp_path):
    """Asymmetric veto: a single rejecting lens sinks the metric, and the
    (model, lens) is named in review.concerns (never-silent)."""
    import sculptor.eval.metric_gen as mg
    rec = mg.generate_objective_metric(
        "trot forward", tmp_path / "v",
        client=_PanelClient(GOOD, {"consistency": {"approved": False, "concerns": ["weighted sum"]},
                                   "*": {"approved": True}}),
        review_models=_PANEL_MODELS(), max_attempts=1)
    assert not rec["accepted"] and not rec["review"]["approved"]
    assert any("veto" in c and "consistency" in c for c in rec["review"]["concerns"])
    assert rec["review_panel"]["veto_by"]


def test_panel_named_exploit_forces_reject(tmp_path):
    """A non-empty gaming_exploit forces a reject even if the reviewer returned
    approved=True — a 'named exploit but approved' contradiction can't slip through."""
    import sculptor.eval.metric_gen as mg
    rec = mg.generate_objective_metric(
        "trot forward", tmp_path / "x",
        client=_PanelClient(GOOD, {"completeness": {"approved": True,
                                                    "gaming_exploit": "stand and flail"},
                                   "*": {"approved": True}}),
        review_models=_PANEL_MODELS(), max_attempts=1)
    assert not rec["accepted"]
    assert any("stand and flail" in c for c in rec["review"]["concerns"])


def test_panel_all_crash_is_not_approved(tmp_path):
    """All reviewers crash → fail-closed overall (n_approve 0 < quorum) → rejected,
    with the inconclusive/quorum reason surfaced."""
    import sculptor.eval.metric_gen as mg
    rec = mg.generate_objective_metric("trot forward", tmp_path / "c",
                                       client=_PanelClient(GOOD, {"*": {"raise": True}}),
                                       review_models=_PANEL_MODELS(), max_attempts=1)
    assert not rec["accepted"] and not rec["review"]["approved"]
    assert any("quorum" in c or "inconclusive" in c for c in rec["review"]["concerns"])


def test_panel_thin_survivor_is_not_approved(tmp_path):
    """The quorum FLOOR: 3 of 4 text lenses crash, 1 approves → 1 < ceil(4/2)=2 →
    NOT approved. A lone survivor cannot carry the panel."""
    import sculptor.eval.metric_gen as mg
    rec = mg.generate_objective_metric(
        "trot forward", tmp_path / "thin",
        client=_PanelClient(GOOD, {"completeness": {"approved": True}, "*": {"raise": True}}),
        review_models=_PANEL_MODELS(), max_attempts=1)
    assert not rec["accepted"]


def test_panel_vlm_lens_skips_without_keyframes(tmp_path):
    """At metric-gen there are no keyframes → the VLM (naturalness) lens SKIPS,
    makes no call, and is excluded from the eligible quorum count."""
    import sculptor.eval.metric_gen as mg
    client = _PanelClient(GOOD, {"*": {"approved": True}})
    rec = mg.generate_objective_metric("trot forward", tmp_path / "noimg",
                                       client=client, review_models=_PANEL_MODELS(),
                                       max_attempts=1)
    panel = rec["review_panel"]["panel"]
    nat = next(p for p in panel if p["lens"] == "naturalness")
    assert nat["status"] == "skip"
    assert rec["review_panel"]["n_eligible"] == 4
    assert not any(c["lens"] == "naturalness" for c in client.messages.calls)


def test_panel_vlm_lens_uses_images_with_keyframes(tmp_path):
    """When keyframes ARE passed (the future post-rollout re-review), the VLM lens
    encodes them into image content blocks and votes — the headline VLM feature."""
    import sculptor.eval.metric_gen as mg
    kfs = []
    for i in range(2):
        p = tmp_path / f"kf{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8))
        kfs.append(p)
    client = _PanelClient(GOOD, {"*": {"approved": True}})
    rec = mg.generate_objective_metric("trot forward", tmp_path / "vlm", client=client,
                                       review_models=_PANEL_MODELS(), review_keyframes=kfs,
                                       max_attempts=1)
    assert rec["accepted"]
    nat = [c for c in client.messages.calls if c["lens"] == "naturalness"]
    assert nat and nat[0]["has_image"]
    statuses = {p["lens"]: p["status"] for p in rec["review_panel"]["panel"]}
    assert statuses["naturalness"] == "approved"
    assert rec["review_panel"]["n_eligible"] == 5     # the VLM lens now counts


def test_panel_blinding_author_excluded_and_absent(tmp_path):
    """Blinding: the author model id is never a reviewer model, and never appears
    in any reviewer's system or user payload. Uses an author that IS in the pool
    (claude-opus-4-8) so the exclusion path is genuinely exercised (not vacuous)."""
    import sculptor.eval.metric_gen as mg
    author = "claude-opus-4-8"
    assert author in mg.REVIEW_PANEL_MODELS              # the exclusion must do real work
    client = _PanelClient(GOOD, {"*": {"approved": True}})
    mg.generate_objective_metric("trot forward", tmp_path / "blind", client=client,
                                 model=author, review_models=_PANEL_MODELS(), max_attempts=1)
    assert client.messages.calls
    used_models = {c["model"] for c in client.messages.calls}
    assert author not in used_models                     # excluded from the roster
    assert used_models <= set(mg.REVIEW_PANEL_MODELS) - {author}
    for c in client.messages.calls:
        assert author not in c["system"]
        assert author not in _msgs_text(c["messages"])


def test_calibration_good_metric_correlates(tmp_path):
    from sculptor.eval.metric_calibration import calibrate_metric
    p = _write(tmp_path, "good.py", GOOD)
    cal = calibrate_metric(p, "go1_trot", threshold=0.7)
    assert cal["ok"] and cal["spearman"] >= 0.7, cal


def test_calibration_constant_metric_fails(tmp_path):
    from sculptor.eval.metric_calibration import calibrate_metric
    p = _write(tmp_path, "const.py", CONSTANT)
    cal = calibrate_metric(p, "go1_trot", threshold=0.7)
    assert not cal["ok"]  # constant → no rank info → spearman 0
