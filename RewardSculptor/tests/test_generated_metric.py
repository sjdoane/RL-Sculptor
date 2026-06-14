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

# §Ship 41: a CORRECT kick metric — leg-velocity bursts from a STATIONARY,
# upright, standing-height stance. It needs all four arrays (returns 0.0 if any
# is missing), so the old locomotion-only `active` archetype scored it ~0
# (stationary → no travel) and the ladder (no root/joint_pos) couldn't
# calibrate it. The kick family archetype + enriched ladder fix both.
GOOD_KICK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w"); jp = arrays.get("joint_pos")
    if jv is None or grav is None or root is None or jp is None:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    xy = root[..., :2]
    drift = float(np.linalg.norm(xy.max(0) - xy.min(0), axis=-1).mean())
    stationary = float(np.exp(-drift / 0.4))
    zmed = float(np.median(root[..., 2]))
    height = float(np.clip((zmed - 0.45) / 0.20, 0.0, 1.0))
    knee_peak = float(np.abs(jv[..., 2:4]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    return {"spec_score": float(np.clip(up * stationary * height * burst, 0.0, 1.0))}
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
    def __init__(self, src, approved): self._src = src; self._approved = approved; self.creates = 0
    def create(self, **kw):
        self.creates += 1
        return _Resp("```python\n" + self._src + "\n```")
    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        return _Parsed(MetricReview(approved=self._approved,
                                    concerns=[] if self._approved else ["gameable"],
                                    summary="x"))


class _FakeClient:
    def __init__(self, src, approved=True): self.messages = _Messages(src, approved)


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
