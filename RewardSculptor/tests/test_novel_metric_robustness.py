"""§novel-metric robustness: regression tests for the generation-pipeline fixes
that drove the auto-objective-metric accept rate up on NOVEL fold/compound goals
(toe-touch, squat, bow, wave, "bend then wave"). Each test pins one fix found by
running real best-of-N generations and reading the actual rejected sources:

  Fix 1 — the selectivity probe is enriched (gesture + sequenced compound probes)
          and supplies PHYSICAL rad/s velocity + the optional foot channels, so a
          compound/multi-phase or velocity-thresholding novel metric is no longer
          false-rejected as "competent 0.000".
  Fix 2 — the static joint-index ban no longer flags a 3-vector axis read
          (root[:, :, 2] / projected_gravity_b[:, :, 2]); only joint_pos/joint_vel
          columns are flagged.
  Fix 3 — `type(e).__name__` (the never-raise diagnostic idiom) passes AST safety;
          the dangerous traversal dunders (__class__, …) stay blocked.
  Fix 4 — the vacuous-branch entry keys on the POSITIVE archetypes (~0), so a novel
          metric that gives a fixed NEGATIVE a little credit still reaches the probe
          (the fixed negatives are folded into the degenerate anchor for safety).
  Fix 5 — on a vacuous pass the selectivity-probe scores are surfaced to the
          independent reviewer (so it does not false-flag the all-zero fixed battery
          as near-constant).

Round 2 (a second real best-of-N batch across bow/wave/twist/march + an adversarial
review of the round-1 commit found further fixes):
  Fix 6 — a FOLD/bow/gesture goal no longer MIS-resolves to a behavior family from a
          stray directional word ("bend FORWARD into a bow" used to → locomotion via
          "forward"); resolve_behavior_family requires a real gait verb (or a bare
          directional cue with NO posture verb, and never "in place"), so a true novel
          goal resolves to None and reaches the probe. The vacuous entry KEEPS the
          `family is None` scope, so a RECOGNIZED-family metric that scores its family
          positive ~0 (a wrong-behavior kick metric) is rejected on the normal path,
          NOT vacuously rescued by the goal-agnostic gesture probe (round-3 review).
  Fix 7 — the forward-WALKER is folded into the vacuous degenerate anchor UNLESS the
          goal is forward-directional, catching an in-place novel metric that rewards
          walking without false-rejecting a legitimately forward goal.
  Fix 8 — _graded_fold_rung / _graded_posture_rung share the probe's `_physical_vel`
          (rad/s) so the best-of-N selector and the validator never diverge in unit.
  Fix 9 — a probe-selective metric that a folded DEGENERATE archetype also scores high
          is reported as "gameable: ... '<archetype>' ..." (actionable feedback) rather
          than a misleading "near-constant".
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sculptor.eval.metric_validate import (
    _ast_safety,
    _graded_fold_rung,
    _NAMES_12,
    _physical_vel,
    _selectivity_probe,
    validate_generated_metric,
)

_ROBOT = "Mjlab-Velocity-Flat-Unitree-G1"


def _val(src: str, goal: str) -> dict:
    d = Path(tempfile.mkdtemp()) / "m.py"
    d.write_text(src, encoding="utf-8")
    return validate_generated_metric(src, d, behavior_goal=goal, robot_hint=_ROBOT)


# ── Fix 1: enriched + physical-velocity probe ────────────────────────────────

# A compound "bend → touch toes → stand → WAVE arms" metric: a sharp completion
# gate requiring bent (torso tilt) AND recovered (upright) AND a distinct arm WAVE
# (shoulder velocity oscillation) AFTER recovery, times a min of channels. This is
# the exact real artifact that had all 4 candidates rejected ("competent 0.000")
# before the probe gained a sequenced bend-then-gesture rollout + physical velocity.
COMPOUND_BEND_WAVE = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_hip_pitch","right_hip_pitch","left_shoulder_pitch","right_shoulder_pitch"]
def compute_spec(arrays, behavior, meta):
    try:
        pg=arrays.get("projected_gravity_b"); jp=arrays.get("joint_pos"); jv=arrays.get("joint_vel")
        if pg is None or jp is None or jv is None: return {"spec_score":0.0}
        roles=(meta or {}).get("joint_roles",{}) or {}
        need=["left_hip_pitch","right_hip_pitch","left_shoulder_pitch","right_shoulder_pitch"]
        if not all(r in roles for r in need): return {"spec_score":0.0}
        lhp,rhp,lsp,rsp=(int(roles[r]) for r in need)
        gx=pg[...,0]; gz=pg[...,2]
        hip=0.5*(jp[...,lhp]+jp[...,rhp]); shv=0.5*(jv[...,lsp]+jv[...,rsp]); sh=0.5*(jp[...,lsp]+jp[...,rsp])
        E=pg.shape[1]
        bent=np.zeros(E,bool); rec=np.zeros(E,bool); waved=np.zeros(E,bool)
        for e in range(E):
            be=(gx[:,e]>0.4)&(np.abs(hip[:,e])>0.5)
            if not be.any(): continue
            tb=int(np.argmax(be)); bent[e]=True
            ue=(gz[tb+1:,e]<-0.9)&(np.abs(hip[tb+1:,e])<0.35)
            if not ue.any(): continue
            tu=tb+1+int(np.argmax(ue)); rec[e]=True
            seg=sh[tu:,e]; vseg=shv[tu:,e]
            if seg.size<4: continue
            if float(np.max(seg)-np.min(seg))>0.7:
                sign=np.sign(np.where(np.abs(vseg)>0.6,vseg,0)); sc=0; last=0
                for s in sign:
                    if s!=0:
                        if last!=0 and s!=last: sc+=1
                        last=s
                if sc>=2: waved[e]=True
        gate=float((bent&rec&waved).mean())
        c_bend=float(np.mean(1.0-np.exp(-np.maximum(np.max(gx,axis=0)-0.35,0.0)/0.3)))
        c_wave=1.0 if waved.any() else 0.0
        return {"spec_score":float(np.clip(gate*min(c_bend,c_wave),0,1))}
    except Exception as ex:
        return {"spec_score":0.0,"reason":f"exc:{type(ex).__name__}"}
'''

# A genuinely-selective wave: arms must be HELD up (sustained mean elevation) AND
# oscillate — so a fast-flail/chaos rollout (oscillation, no sustained raise) scores 0.
PURE_WAVE_GOOD = '''import numpy as np
REQUIRED_JOINT_ROLES=["left_shoulder_pitch","right_shoulder_pitch"]
def compute_spec(arrays,behavior,meta):
    jp=arrays.get("joint_pos"); jv=arrays.get("joint_vel")
    if jp is None or jv is None: return {"spec_score":0.0}
    roles=(meta or {}).get("joint_roles",{}) or {}
    idx=[roles[r] for r in ("left_shoulder_pitch","right_shoulder_pitch") if r in roles]
    if not idx: return {"spec_score":0.0}
    sh=jp[...,idx].mean(-1); shv=jv[...,idx].mean(-1)
    raised=float(np.mean(np.mean(sh,axis=0)>0.5))
    osc=0.0
    for e in range(sh.shape[1]):
        v=shv[:,e]; sign=np.sign(np.where(np.abs(v)>0.6,v,0)); sc=0; last=0
        for s in sign:
            if s!=0:
                if last!=0 and s!=last: sc+=1
                last=s
        osc+=sc
    osc/=sh.shape[1]
    gate=1.0 if (raised>0.7 and osc>=2) else 0.0
    amp=float(np.clip(np.mean(np.max(sh,axis=0))/1.0,0,1))
    return {"spec_score":float(np.clip(gate*amp,0,1))}
'''

# A GAMEABLE wave: only checks oscillation AMPLITUDE (no sustained raise), so a
# fast whole-body flail games it. Must be REJECTED (the fixed flail/chaos negatives
# are folded into the degenerate anchor).
PURE_WAVE_FLAIL = '''import numpy as np
REQUIRED_JOINT_ROLES=["left_shoulder_pitch","right_shoulder_pitch"]
def compute_spec(arrays,behavior,meta):
    jp=arrays.get("joint_pos"); jv=arrays.get("joint_vel")
    if jp is None or jv is None: return {"spec_score":0.0}
    roles=(meta or {}).get("joint_roles",{}) or {}
    idx=[roles[r] for r in ("left_shoulder_pitch","right_shoulder_pitch") if r in roles]
    if not idx: return {"spec_score":0.0}
    sh=jp[...,idx].mean(-1); shv=jv[...,idx].mean(-1)
    amp=float(np.mean(np.max(sh,axis=0)-np.min(sh,axis=0)))
    osc=0.0
    for e in range(sh.shape[1]):
        v=shv[:,e]; sign=np.sign(np.where(np.abs(v)>0.6,v,0)); sc=0; last=0
        for s in sign:
            if s!=0:
                if last!=0 and s!=last: sc+=1
                last=s
        osc+=sc
    osc/=sh.shape[1]
    gate=1.0 if (amp>0.6 and osc>=2) else 0.0
    return {"spec_score":float(np.clip(gate*np.clip(amp/2.0,0,1),0,1))}
'''


def test_compound_bend_then_wave_passes_vacuously():
    """The real toe-touch+wave compound metric (all 4 candidates rejected pre-fix)
    now passes validation: the probe has a sequenced bend-then-gesture rollout the
    multi-phase completion gate can satisfy, with physical rad/s velocity for the
    arm-oscillation channel."""
    v = _val(COMPOUND_BEND_WAVE,
             "bend over touch your toes stand up then wave your arms in the air")
    assert v["ok"], v["reasons"]
    assert v["nondegeneracy_vacuous"] is True
    assert v["selectivity_probe"]["competent"] >= 0.5
    assert v["selectivity_probe"]["spread"] >= 0.1


def test_pure_wave_metric_passes_vacuously():
    """A selective wave metric (sustained raise + oscillation) passes via the gesture
    probe (C3) — impossible before the probe gained an arm-raise+oscillate rollout."""
    v = _val(PURE_WAVE_GOOD, "raise both arms overhead and wave them side to side")
    assert v["ok"], v["reasons"]
    assert v["nondegeneracy_vacuous"] is True


def test_flail_gameable_wave_rejected():
    """A wave metric that an undirected whole-body flail games is REJECTED — the
    fixed flail/chaos negatives are folded into the vacuous degenerate anchor."""
    v = _val(PURE_WAVE_FLAIL, "raise both arms overhead and wave them side to side")
    assert not v["ok"], v["reasons"]


def test_selectivity_probe_supplies_physical_velocity():
    """The probe's joint_vel is PHYSICAL rad/s (gradient/dt). A metric that fires only
    on a >5 rad/s joint velocity lights up — it scored 0 everywhere when the probe used
    bare rad/frame gradients (~50x too small)."""
    def fn(arrays, behavior, meta):
        jv = arrays.get("joint_vel")
        if jv is None:
            return {"spec_score": 0.0}
        return {"spec_score": 1.0 if float(np.max(np.abs(jv))) > 5.0 else 0.0}

    meta = {"joint_names": list(_NAMES_12)}
    probe = _selectivity_probe(fn, meta)
    assert probe["competent"] >= 0.1            # a gesture/sequence probe exceeds 5 rad/s
    assert probe["degenerate"] == 0.0           # still/fallen have zero velocity


def test_selectivity_probe_supplies_foot_channels():
    """The probe supplies the optional foot channels (planted), so a metric that READS
    foot_contact / foot_pos_b without a None-guard is exercised, not auto-zeroed."""
    seen = {"contact": False, "pos": False}

    def fn(arrays, behavior, meta):
        if arrays.get("left_foot_contact") is not None:
            seen["contact"] = True
        if arrays.get("left_foot_pos_b") is not None:
            seen["pos"] = True
        return {"spec_score": 0.0}

    _selectivity_probe(fn, {"joint_names": list(_NAMES_12)})
    assert seen["contact"] and seen["pos"]


# ── Fix 2: joint-index gate no longer flags 3-vector axis reads ──────────────

THREE_VEC_SLICE = '''import numpy as np
REQUIRED_JOINT_ROLES=["left_hip_pitch","right_hip_pitch","left_knee","right_knee"]
def compute_spec(arrays,behavior,meta):
    jp=arrays.get("joint_pos"); root=arrays.get("root_link_pos_w"); pg=arrays.get("projected_gravity_b")
    if jp is None or root is None or pg is None: return {"spec_score":0.0}
    roles=(meta or {}).get("joint_roles",{}) or {}
    idx=[roles[r] for r in ("left_hip_pitch","right_hip_pitch","left_knee","right_knee") if r in roles]
    if not idx: return {"spec_score":0.0}
    z=root[:, :, 2]; gz=pg[:, :, 2]                  # explicit-slice 3-vector axis reads (LEGIT)
    drop=float(np.mean(z[:5].mean(axis=0)-z.min(axis=0)))
    ret=float(np.mean(np.abs(z[-5:].mean(axis=0)-z[:5].mean(axis=0))))
    rom=float(np.mean(np.max(jp[...,idx],axis=0)-np.min(jp[...,idx],axis=0)))
    up=float(np.mean(gz[:5].mean(axis=0)<-0.85))
    gate=(1.0 if drop>0.2 else 0.0)*(1.0 if ret<0.08 else 0.0)*(1.0 if up>0.7 else 0.0)
    return {"spec_score":float(np.clip(gate*np.clip(rom/1.2,0,1),0,1))}
'''

JOINT_COL_SLICE = '''import numpy as np
def compute_spec(arrays,behavior,meta):
    jv=arrays.get("joint_vel")
    if jv is None: return {"spec_score":0.0}
    return {"spec_score":float(np.clip(np.abs(jv[:, :, 0]).mean(),0,1))}
'''


def test_three_vector_explicit_slice_not_flagged():
    """root_link_pos_w[:, :, 2] and projected_gravity_b[:, :, 2] are legitimate
    3-vector axis reads — the static joint-index ban must NOT fire (it used to, on
    every `[:, :, N]`, false-rejecting good toe-touch/bow metrics)."""
    v = _val(THREE_VEC_SLICE, "touch your toes then stand back up")
    assert v["gates"]["no_raw_joint_index"] is True, v["reasons"]
    assert v["ok"], v["reasons"]


def test_joint_column_hardcode_still_flagged():
    """A real hard-coded JOINT column (joint_vel[:, :, 0]) is still flagged."""
    v = _val(JOINT_COL_SLICE, "trot forward")
    assert v["gates"]["no_raw_joint_index"] is False
    assert any("joint-index" in r for r in v["reasons"])


# ── Fix 3: benign __name__ allowed, dangerous dunders blocked ────────────────

NAME_DUNDER_OK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    try:
        root = arrays.get("root_link_pos_w")
        if root is None:
            return {"spec_score": 0.0}
        return {"spec_score": float(np.clip(root[..., 2].mean(), 0, 1))}
    except Exception as ex:
        return {"spec_score": 0.0, "err": type(ex).__name__}
'''

CLASS_DUNDER_BAD = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    x = (0).__class__
    return {"spec_score": 0.5}
'''


def test_name_dunder_allowed():
    """`type(e).__name__` — the never-raise diagnostic idiom the generator emits
    constantly — passes AST safety (a name string carries no escape vector)."""
    assert _ast_safety(NAME_DUNDER_OK) == []


def test_traversal_dunders_still_blocked():
    """The class-traversal escape vector (__class__ → __bases__ → __subclasses__)
    stays blocked — only __name__ is allowlisted."""
    probs = _ast_safety(CLASS_DUNDER_BAD)
    assert any("__class__" in p for p in probs)


# ── Fix 4: vacuous-branch entry keys on positives, folds fixed negatives ─────

DIP_TINY_FALLEN_CREDIT = '''import numpy as np
REQUIRED_JOINT_ROLES=["left_hip_pitch","right_hip_pitch","left_knee","right_knee"]
def compute_spec(arrays,behavior,meta):
    jp=arrays.get("joint_pos"); root=arrays.get("root_link_pos_w"); pg=arrays.get("projected_gravity_b")
    if jp is None or root is None or pg is None: return {"spec_score":0.0}
    roles=(meta or {}).get("joint_roles",{}) or {}
    idx=[roles[r] for r in ("left_hip_pitch","right_hip_pitch","left_knee","right_knee") if r in roles]
    if not idx: return {"spec_score":0.0}
    z=root[...,2]
    drop=float(np.mean(z[:5].mean(0)-z.min(0))); ret=float(np.mean(np.abs(z[-5:].mean(0)-z[:5].mean(0))))
    up=float(np.mean(pg[:5,...,2].mean(0)<-0.85))
    gate=1.0 if (drop>0.2 and ret<0.08 and up>0.7) else 0.0
    amp=float(np.clip(drop/0.35,0,1))
    tiny=0.05*float(np.clip(np.mean(np.abs(pg[...,0])),0,1))   # small credit only the fallen archetype gets
    return {"spec_score":float(np.clip(gate*amp+tiny,0,1))}
'''


def test_vacuous_entry_keys_on_positives_not_whole_battery():
    """A novel dip metric that gives a fixed NEGATIVE (fallen) a little credit
    (>1e-3) still ENTERS the vacuous branch and passes non-degeneracy — the entry
    keys on the POSITIVE archetypes being ~0, not the whole battery. Pre-fix the
    fallen>1e-3 score kicked it to the normal path, where with all positives ~0 the
    'still(0)>=active(0)' tie false-rejected it."""
    v = _val(DIP_TINY_FALLEN_CREDIT, "touch your toes then stand back up")
    assert v["archetype_scores"]["fallen"] > 1e-3        # a negative scored above the floor
    assert v["nondegeneracy_vacuous"] is True            # yet it still reached the probe
    assert v["gates"]["nondegeneracy"] is True           # and the probe found it selective


# ── Fix 5: selectivity-probe scores surfaced to the reviewer on a vacuous pass ─

class _RecParsed:
    def __init__(self, parsed): self.parsed_output = parsed


class _RecMessages:
    def __init__(self): self.users: list[str] = []

    def parse(self, **kw):
        from sculptor.eval.metric_gen import MetricReview
        msgs = kw.get("messages", [])
        self.users.append(str(msgs[0]["content"]) if msgs else "")
        return _RecParsed(MetricReview(approved=True, concerns=[], summary="ok"))


class _RecClient:
    def __init__(self):
        self.messages = _RecMessages()


def test_review_payload_includes_selectivity_when_vacuous():
    from sculptor.eval.metric_gen import _review_metric
    c = _RecClient()
    _review_metric(c, "m", "goal", "src", {"still": 0.0},
                   selectivity={"competent": 0.83, "degenerate": 0.0, "spread": 0.83})
    assert "selectivity_probe" in c.messages.users[0]
    assert "0.83" in c.messages.users[0]


def test_review_payload_omits_selectivity_when_none():
    """Byte-identical to the prior single-reviewer payload when there is no vacuous
    selectivity probe (selectivity defaults None → key absent)."""
    from sculptor.eval.metric_gen import _review_metric
    c = _RecClient()
    _review_metric(c, "m", "goal", "src", {"still": 0.0})
    assert "selectivity_probe" not in c.messages.users[0]


# ── Fix 6: mis-resolved family still reaches the probe ───────────────────────

# A genuine bow/bend metric (torso pitches forward then recovers upright). The goal
# "bend forward …" MIS-resolves to family=locomotion via the stray word "forward",
# yet the metric scores every fixed positive (incl the locomotion `active` walker) ~0.
BOW_TILT_RETURN = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    pg = arrays.get("projected_gravity_b")
    if pg is None:
        return {"spec_score": 0.0}
    gx = pg[..., 0]; gz = pg[..., 2]
    tilt = float(np.mean(np.max(gx, axis=0)))                       # torso pitched forward at peak
    ret = float(np.mean(np.abs(gz[-5:].mean(0) - gz[:5].mean(0))))  # returns upright
    up0 = float(np.mean(gz[:5].mean(0) < -0.85))
    gate = (1.0 if tilt > 0.5 else 0.0) * (1.0 if ret < 0.15 else 0.0) * (1.0 if up0 > 0.7 else 0.0)
    return {"spec_score": float(np.clip(gate * np.clip(tilt, 0.0, 1.0), 0.0, 1.0))}
'''


def test_fold_goal_with_directional_word_resolves_to_none():
    """§round-3 fix: a FOLD/bow goal that contains a stray directional word ('bend
    FORWARD into a deep bow') no longer MIS-resolves to locomotion — the posture verb
    (bend/bow) suppresses the bare-'forward' locomotion cue — so it resolves to None and
    reaches the selectivity probe (instead of being false-rejected on the locomotion
    normal path where its forward-travel positive scores ~0)."""
    from sculptor.eval.metric_validate import resolve_behavior_family
    assert resolve_behavior_family("bend forward into a deep bow and return upright") is None
    v = _val(BOW_TILT_RETURN, "bend forward into a deep bow and return upright")
    assert v["family"] is None
    assert v["nondegeneracy_vacuous"] is True
    assert v["ok"], v["reasons"]


def test_resolve_family_precision():
    """The precise locomotion resolver: a real gait verb OR a bare directional/velocity
    cue with NO posture verb resolves to locomotion; an 'in place' skill and a posture
    goal do NOT — without breaking the Hopper 'forward_velocity' case."""
    from sculptor.eval.metric_validate import resolve_behavior_family as rf
    assert rf("trot forward in a straight line") == "locomotion"           # gait verb
    assert rf("Canonical Hopper-v4 reward: forward_velocity + alive_bonus") == "locomotion"
    assert rf("march in place lifting your knees up high") is None         # in-place ≠ travel
    assert rf("lean forward and bow deeply then return to standing") is None
    assert rf("reach down and touch the floor then rise") is None


def test_good_family_metric_stays_on_normal_path():
    """A GOOD family metric lights up its own positive archetype → battery informative →
    not vacuous (the builtin family path is unchanged)."""
    from tests.test_generated_metric import GOOD_KICK
    v = _val(GOOD_KICK, "repeatedly kick forward with one leg")
    assert v["family"] == "kick"
    assert v["nondegeneracy_vacuous"] is False
    assert v["ok"], v["reasons"]


# A wrong-behavior metric for a RECOGNIZED family: rewards a sustained arms-overhead
# hold (a gesture), which the goal-agnostic gesture probe (C3) scores high. For a KICK
# goal it must NOT vacuously pass — the family ground truth is authoritative.
ARMS_OVERHEAD_FOR_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_shoulder_pitch", "right_shoulder_pitch", "left_elbow", "right_elbow"]
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos"); pg = arrays.get("projected_gravity_b")
    if jp is None or pg is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) or {}
    idx = [roles[r] for r in ("left_shoulder_pitch","right_shoulder_pitch","left_elbow","right_elbow") if r in roles]
    if not idx:
        return {"spec_score": 0.0}
    raised = float(np.mean(np.clip(jp[..., idx].mean(0).mean(-1) / 2.0, 0, 1)))
    up = float(np.mean(pg[..., 2] < -0.85))
    return {"spec_score": float(np.clip(raised * up, 0, 1))}
'''


def test_wrong_behavior_recognized_family_not_vacuous():
    """§round-3 review regression: an arms-overhead metric for a KICK goal scores the
    fixed kick positive ~0, so for the RECOGNIZED kick family it must take the normal
    path and be REJECTED — NOT vacuously rescued by the goal-agnostic gesture probe.
    (Restoring the family-is-None scope on the vacuous entry closes this.)"""
    v = _val(ARMS_OVERHEAD_FOR_KICK, "kick a ball forward with your right leg")
    assert v["family"] == "kick"
    assert v["nondegeneracy_vacuous"] is False
    assert not v["ok"], v["reasons"]


# ── Fix 7: walker folded for in-place goals, not forward goals ───────────────

# Rewards EITHER a pelvis dip OR walking-at-standing-height — gameable on an in-place
# goal (walk instead of dipping), legitimate on a forward goal.
DIP_OR_WALK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_hip_pitch", "right_hip_pitch"]
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); pg = arrays.get("projected_gravity_b")
    if root is None or pg is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    drop = float(np.mean(z[:5].mean(0) - z.min(0)))
    ret = float(np.mean(np.abs(z[-5:].mean(0) - z[:5].mean(0))))
    up = float(np.mean(pg[:5, ..., 2].mean(0) < -0.85))
    dip = (1.0 if (drop > 0.2 and ret < 0.08 and up > 0.7) else 0.0) * float(np.clip(drop / 0.35, 0, 1))
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    standing = float(np.mean(z.mean(0) > 0.65))
    walk = (1.0 if standing > 0.7 else 0.0) * float(1 - np.exp(-disp / 1.0))
    return {"spec_score": float(np.clip(max(dip, walk), 0, 1))}
'''


def test_walker_folded_for_inplace_goal():
    """An in-place novel metric that also rewards forward walking is REJECTED in the
    vacuous branch — the forward walker is folded into the degenerate anchor and named."""
    v = _val(DIP_OR_WALK, "touch your toes then stand back up")
    assert not v["ok"]
    assert any("walker" in r for r in v["reasons"]), v["reasons"]


def test_walker_not_folded_for_forward_goal():
    """The same metric is NOT walker-rejected for a forward-directional goal (goal_axis
    +x, but family None so it reaches the probe) — folding the walker there would
    false-reject a legitimately forward novel goal."""
    from sculptor.eval.metric_validate import resolve_goal_frame
    goal = "lunge forward and touch the floor then rise"
    assert resolve_goal_frame(goal)["goal_axis"] == "+x"   # forward-directional
    v = _val(DIP_OR_WALK, goal)
    assert v["family"] is None                             # reaches the probe
    assert v["nondegeneracy_vacuous"] is True
    assert v["ok"], v["reasons"]


# ── Fix 8: graded rungs share the probe's physical rad/s velocity ────────────

def test_graded_rungs_use_physical_velocity():
    """The best-of-N selector's rung builders share the probe's `_physical_vel` (rad/s),
    so the SELECTOR and the VALIDATOR never feed joint_vel in divergent units."""
    rung = _graded_fold_rung(0.35, 1.2)
    assert np.allclose(rung["joint_vel"], _physical_vel(rung["joint_pos"]))
    # rad/s is ~1/dt larger than a bare per-frame gradient (the divergence the fix closes)
    bare = np.gradient(rung["joint_pos"], axis=0)
    assert np.max(np.abs(rung["joint_vel"])) > 10.0 * np.max(np.abs(bare)) - 1e-9


# ── Fix 9: a flail/walker-gameable probe-selective metric is named, not "near-constant" ─

# Scores a competent torso oscillation high, but a fast whole-body flail oscillates the
# same joint just as much — gameable (no frequency bound / no isolation).
TWIST_FLAIL_GAMEABLE = '''import numpy as np
REQUIRED_JOINT_ROLES = ["waist_yaw"]
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos"); jv = arrays.get("joint_vel"); pg = arrays.get("projected_gravity_b")
    if jp is None or jv is None or pg is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) or {}
    if "waist_yaw" not in roles:
        return {"spec_score": 0.0}
    w = jp[..., int(roles["waist_yaw"])]
    p2p = float(np.mean(np.max(w, axis=0) - np.min(w, axis=0)))
    mid = 0.5 * (np.max(w, axis=0) + np.min(w, axis=0))
    bidi = float(np.mean(np.minimum(np.max(w, axis=0) - mid, mid - np.min(w, axis=0))))
    up = float(np.mean(pg[..., 2] < -0.85))
    # Large peak-to-peak (> 0.9 rad): the small slow `active` archetype (p2p ~0.8) does
    # NOT trip it (so the metric enters the vacuous branch), but a fast whole-body
    # `upright_flail` (p2p ~2.4) games it (no frequency bound / no joint isolation).
    gate = 1.0 if (p2p > 0.9 and bidi > 0.08 and up > 0.7) else 0.0
    return {"spec_score": float(np.clip(gate * np.clip(p2p / 1.4, 0, 1), 0, 1))}
'''


def test_flail_gameable_metric_named_not_near_constant():
    """A probe-selective metric a fixed degenerate archetype also scores high is reported
    as 'gameable: ... <archetype>' (actionable), not a misleading 'near-constant'."""
    v = _val(TWIST_FLAIL_GAMEABLE, "twist your torso to the left and right repeatedly")
    assert not v["ok"]
    assert any("gameable" in r for r in v["reasons"]), v["reasons"]
    # the probe DID find it selective (so the message must not claim near-constant)
    assert v["selectivity_probe"]["competent"] >= 0.5
    assert not any("near-constant" in r for r in v["reasons"])


def test_balance_goal_with_velocity_resolves_to_cartpole():
    """§round-5 review: a balance/cartpole goal that mentions 'velocity' (in _directional)
    must resolve to cartpole, NOT locomotion — the cartpole check now precedes the
    locomotion check, so the wrong go1_trot calibration anchor isn't selected. The Hopper
    forward_velocity goal (no balance token) still resolves to locomotion."""
    from sculptor.eval.metric_validate import resolve_behavior_family as rf
    assert rf("minimize joint velocity while balancing the pole") == "cartpole"
    assert rf("balance the cartpole with low velocity") == "cartpole"
    assert rf("balance the pole upright and keep the cart centered") == "cartpole"
    assert rf("Canonical Hopper-v4 reward: forward_velocity + alive_bonus") == "locomotion"


def test_gait_plus_balance_resolves_to_locomotion():
    """§round-6: a real GAIT verb + an incidental 'balance' token resolves to LOCOMOTION,
    not cartpole — a gait verb is a far stronger locomotion signal (the round-5
    cartpole-before-locomotion reorder over-corrected; cartpole now requires not-gait).
    A pure balance/cartpole goal (no gait) still resolves to cartpole."""
    from sculptor.eval.metric_validate import resolve_behavior_family as rf
    assert rf("walk while balancing a load") == "locomotion"
    assert rf("trot steadily, balancing the body") == "locomotion"
    assert rf("run while balancing on a beam") == "locomotion"
    assert rf("minimize joint velocity while balancing the pole") == "cartpole"  # no gait
    assert rf("balance the pole upright and keep the cart centered") == "cartpole"


# ── §round-9 SECURITY: _ast_safety must contain untrusted metric code ─────────
# Round-9 found _ast_safety did NOT contain an untrusted (LLM-authored, behavior_goal-
# derived) metric: numpy is a full IO/pickle/native surface and several escapes are
# AST-blind. These pin each reproduced vector as REJECTED, and confirm legit metrics pass.
import pytest as _pytest


@_pytest.mark.parametrize("name,src", [
    ("numpy_savetxt", "import numpy\ndef compute_spec(a,b,m):\n"
                      "    numpy.savetxt('/tmp/x', numpy.array([1.0]))\n    return {'spec_score':0.5}\n"),
    ("np_load_allow_pickle", "import numpy as np\ndef compute_spec(a,b,m):\n"
                             "    np.load('/tmp/x', allow_pickle=True)\n    return {'spec_score':0.5}\n"),
    ("reduce_gadget", "import numpy as np\nclass _P:\n    def __reduce__(self):\n"
                      "        return (print, ('x',))\ndef compute_spec(a,b,m):\n    return {'spec_score':0.5}\n"),
    ("format_dunder", "def compute_spec(a,b,m):\n"
                      "    s = '{f.__globals__[__builtins__]}'.format(f=compute_spec)\n    return {'spec_score':0.5}\n"),
    ("ctypeslib_import", "import numpy.ctypeslib\ndef compute_spec(a,b,m):\n    return {'spec_score':0.5}\n"),
    ("from_numpy_import_save", "from numpy import save\ndef compute_spec(a,b,m):\n    return {'spec_score':0.5}\n"),
    ("star_import", "from numpy import *\ndef compute_spec(a,b,m):\n    return {'spec_score':0.5}\n"),
    ("ndarray_tofile", "import numpy as np\ndef compute_spec(a,b,m):\n"
                       "    np.array([1.0]).tofile('/tmp/x')\n    return {'spec_score':0.5}\n"),
    ("breakpoint_call", "def compute_spec(a,b,m):\n    breakpoint()\n    return {'spec_score':0.5}\n"),
])
def test_ast_safety_blocks_escape_vectors(name, src):
    """Each round-9-reproduced sandbox-escape vector is REJECTED by the static gate."""
    assert _ast_safety(src), f"{name} must be flagged by _ast_safety"


def test_ast_safety_accepts_legit_metric():
    """A normal numpy physical-quantity metric (incl the type(e).__name__ diagnostic and
    f-strings) still passes — the hardening must not false-reject real metrics."""
    legit = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    try:
        jp = arrays.get("joint_pos"); root = arrays.get("root_link_pos_w")
        if jp is None or root is None:
            return {"spec_score": 0.0, "reason": "missing_arrays"}
        rom = float(np.mean(np.max(jp, axis=0) - np.min(jp, axis=0)))
        drop = float(np.mean(root[..., 2][:5].mean(0) - root[..., 2].min(0)))
        return {"spec_score": float(np.clip(rom * drop, 0.0, 1.0)),
                "rom": rom, "ch_drop": drop}
    except Exception as ex:
        return {"spec_score": 0.0, "reason": f"exc:{type(ex).__name__}"}
'''
    assert _ast_safety(legit) == []


def test_load_generated_module_enforces_ast_safety(tmp_path):
    """§round-10: the static gate is enforced at the load chokepoint, so EVERY exec path
    (runtime scorer, calibration, best-of-N) is screened — not only validation. A metric
    that bypassed validation (e.g. a future edit-metric endpoint) and reaches the loader
    with a forbidden construct is REFUSED at load instead of exec'ing arbitrary code."""
    from sculptor.eval.generated_metric import load_generated_module
    unsafe = tmp_path / "metric.py"
    unsafe.write_text(
        "import os\n"
        "def compute_spec(arrays, behavior, meta):\n"
        "    os.system('echo pwn')\n"
        "    return {'spec_score': 0.5}\n",
        encoding="utf-8")
    with _pytest.raises(ImportError, match="_ast_safety violations"):
        load_generated_module(unsafe)


def test_load_generated_module_loads_safe_metric(tmp_path):
    """The load chokepoint must still load a legitimate numpy metric unchanged."""
    from sculptor.eval.generated_metric import load_generated_module
    safe = tmp_path / "metric.py"
    safe.write_text(
        "import numpy as np\n"
        "def compute_spec(arrays, behavior, meta):\n"
        "    return {'spec_score': float(np.clip(0.5, 0.0, 1.0))}\n",
        encoding="utf-8")
    mod = load_generated_module(safe)
    assert callable(getattr(mod, "compute_spec", None))


def test_compute_generated_metric_degrades_on_unsafe_source(tmp_path):
    """The never-raise runtime scorer turns a load-refusal into an honest 0.0 (it does
    NOT propagate the ImportError into the training loop)."""
    from sculptor.eval.generated_metric import compute_generated_metric
    import numpy as np
    mp = tmp_path / "metric.py"
    mp.write_text(
        "import os\n"
        "def compute_spec(arrays, behavior, meta):\n"
        "    return {'spec_score': 0.9}\n",
        encoding="utf-8")
    rd = tmp_path / "roll"; rd.mkdir()
    np.savez(rd / "trajectory.npz", joint_pos=np.zeros((10, 4, 12)))
    (rd / "behavior.json").write_text("{}", encoding="utf-8")
    out = compute_generated_metric(mp, rd)
    assert out.get("spec_score") == 0.0
    assert "error" in out
