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
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sculptor.eval.metric_validate import (
    _ast_safety,
    _NAMES_12,
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
