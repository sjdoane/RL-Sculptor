"""§Ship 51: L2 task-derived calibration — a novel-task metric earns
steer-rights by ranking K independently-authored competence ladders. All
LLM calls are mocked; the synthesizer + gate are deterministic and offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.ladder_synth import (
    CompetenceLadder,
    Coordination,
    GamingArchetype,
    GamingArchetypeSet,
    Group,
    MotionSpec,
    RoleQuery,
    render_ladder,
)
from sculptor.eval.metric_calibration import (
    adversarial_archetype_gate,
    adversarial_archetype_gate_spec,
    calibrate_metric,
    calibrate_task_derived,
    kick_required_losers,
    spearman,
    spearman_midrank,
)
from sculptor.eval.robot_manifest import G1_29

G1 = list(G1_29)

# ── metrics under test ────────────────────────────────────────────────

GOOD_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w")
    if jv is None or grav is None or root is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {})
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    drift = float(np.linalg.norm(root[..., :2].max(0) - root[..., :2].min(0), axis=-1).mean())
    stationary = float(np.exp(-drift / 0.4))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    return {"spec_score": float(np.clip(up * stationary * burst, 0.0, 1.0))}
'''

# A WRONG metric for "kick": it rewards forward TRAVEL (mistakes walking for
# kicking). It scores ~0 on stationary kick ladders.
BAD_TRAVEL = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-disp / 2.0)), 0.0, 1.0))}
'''

CONSTANT = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    return {"spec_score": 0.5}
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


# ── canned ladders + a mock author client ─────────────────────────────


def _kick_swing(peak: float, count: int) -> Group:
    return Group(name="swing", mode="burst", peak_radps=peak, burst_count=count,
                 role_query=RoleQuery(segments=["hip", "knee"],
                                      axes=["pitch", None], sides=["left"]))


def kick_ladder() -> CompetenceLadder:
    """A monotone, stationary kick ladder (what an honest author emits)."""
    return CompetenceLadder(competence_axis="left swing-leg sagittal burst speed", rungs=[
        MotionSpec(uprightness=0.6, base_height_m=0.7),
        MotionSpec(uprightness=0.8, base_height_m=0.7, groups=[_kick_swing(3.0, 3)]),
        MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_kick_swing(6.0, 3)]),
        MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_kick_swing(9.0, 4)]),
    ])


def travel_ladder() -> CompetenceLadder:
    """A locomotion-flavoured ladder (varies forward speed) — what a colluder
    aligned to the BAD travel-metric would emit. An honest kick author would
    NOT produce this."""
    return CompetenceLadder(competence_axis="forward base speed", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=0.7, forward_speed_mps=s)
        for s in (0.1, 0.5, 1.0, 1.6)
    ])


def degenerate_yaw_ladder() -> CompetenceLadder:
    return CompetenceLadder(competence_axis="yaw spin rate", rungs=[
        MotionSpec(degenerate_axis=True, degenerate_reason="spin is yaw, unobservable")
        for _ in range(4)
    ])


def near_constant_ladder() -> CompetenceLadder:
    """All rungs near-identical — the SPREAD/distinct sanity should drop it."""
    return CompetenceLadder(competence_axis="barely anything", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=0.7) for _ in range(4)
    ])


# §calibration JSON-mode: the authors now use messages.CREATE + JSON parse (the
# CompetenceLadder/GamingArchetypeSet schemas are too complex for the API grammar
# compiler used by messages.parse). The mocks return a completion whose text block is
# the JSON dump of the authored object — exactly what _author_structured parses.
class _Blk:
    type = "text"
    def __init__(self, text): self.text = text


class _Resp:
    stop_reason = "end_turn"
    def __init__(self, text): self.content = [_Blk(text)]


def _as_json_resp(obj) -> _Resp:
    return _Resp(json.dumps(obj.model_dump()))


class _LadderMessages:
    def __init__(self, ladders):
        self._ladders = list(ladders); self.calls = 0; self.payloads = []

    def create(self, **kw):
        # Record the user payload so a test can assert no metric leaked into it.
        self.payloads.append(kw.get("messages", [{}])[-1].get("content", ""))
        lad = self._ladders[self.calls % len(self._ladders)]
        self.calls += 1
        return _as_json_resp(lad)


class _FakeLadderClient:
    def __init__(self, *ladders): self.messages = _LadderMessages(ladders)


# ── spearman_midrank: the false-grant fix ─────────────────────────────


def test_spearman_midrank_rejects_saturating_and_last_only():
    rung = list(range(5))
    # the argsort spearman FALSE-GRANTS these (ties get sequential ranks):
    assert spearman([0.1, 0.9, 0.9, 0.9, 0.9], rung) == pytest.approx(1.0)
    assert spearman([0.5, 0.5, 0.5, 0.5, 0.9], rung) == pytest.approx(1.0)
    # midrank correctly scores them below the 0.8 bar:
    assert spearman_midrank([0.1, 0.9, 0.9, 0.9, 0.9], rung) == pytest.approx(0.707, abs=0.01)
    assert spearman_midrank([0.5, 0.5, 0.5, 0.5, 0.9], rung) == pytest.approx(0.707, abs=0.01)
    # a genuinely monotone metric stays high; a constant is 0.
    assert spearman_midrank([0.0, 0.2, 0.5, 0.7, 0.9], rung) == pytest.approx(1.0)
    assert spearman_midrank([0.5] * 5, rung) == 0.0


# ── synthesizer ───────────────────────────────────────────────────────


def test_synth_deterministic():
    specs = kick_ladder().rungs
    a = render_ladder(specs, G1)["rungs"][2][0]["joint_vel"]
    b = render_ladder(specs, G1)["rungs"][2][0]["joint_vel"]
    assert np.array_equal(a, b)


def test_render_prepends_fallen_anchor():
    out = render_ladder(kick_ladder().rungs, G1)
    assert out["n"] == 5 and not out["degenerate"]
    anchor_grav = out["rungs"][0][0]["projected_gravity_b"]
    assert float((anchor_grav[..., 2] < -0.85).mean()) == 0.0   # fully fallen


def test_degenerate_yaw_ladder_flagged():
    out = render_ladder(degenerate_yaw_ladder().rungs, G1)
    assert out["degenerate"]
    assert "yaw" in out["reason"] or "inexpressible" in out["reason"]


# ── calibrate_task_derived ────────────────────────────────────────────


def test_three_honest_grants(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    client = _FakeLadderClient(kick_ladder(), kick_ladder(), kick_ladder())
    cal = calibrate_task_derived(p, "repeatedly kick forward with the left leg",
                                 robot_hint="Unitree-G1", client=client)
    assert cal["ok"], cal
    assert cal["method"] == "task_derived"
    assert cal["n_valid"] == 3 and cal["agreement_fraction"] >= 2 / 3
    assert cal["rho_min"] >= 0.5
    assert cal["spearman"] == cal["rho_min"]          # UI mirror


def test_one_colluder_cannot_grant_bad_metric(tmp_path):
    """KEYSTONE: a WRONG (travel) metric, with 2 honest kick ladders + 1
    colluding travel ladder, is DENIED — the honest two give it no usable
    evidence, so a single aligned source cannot carry the grant."""
    p = _write(tmp_path, "bad.py", BAD_TRAVEL)
    client = _FakeLadderClient(kick_ladder(), kick_ladder(), travel_ladder())
    cal = calibrate_task_derived(p, "kick forward with the left leg",
                                 robot_hint="Unitree-G1", client=client)
    assert not cal["ok"], cal
    assert cal["n_valid"] < 2                          # honest ladders = no evidence
    # and the GOOD metric IS granted on the very same ladders:
    pg = _write(tmp_path, "good.py", GOOD_KICK)
    calg = calibrate_task_derived(pg, "kick forward with the left leg",
                                  robot_hint="Unitree-G1", client=client)
    assert calg["ok"], calg


# A pure-motion metric that does NOT gate uprightness — it scores the fallen+
# tremor anchor (rung 0) HIGH, so the absolute-separation gate must deny it.
MOTION_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel")
    if jv is None:
        return {"spec_score": 0.0}
    return {"spec_score": float(np.clip(np.mean(np.abs(jv)) / 3.0, 0.0, 1.0))}
'''


def test_separation_anchor_denies_motion_rewarder(tmp_path):
    """A flail/motion rewarder scores the fallen+jittering anchor ABOVE the
    competent rungs (negative separation), so it fails the absolute-separation
    anchor and earns no grant — even though it might rank some rungs."""
    p = _write(tmp_path, "motion.py", MOTION_HACK)
    client = _FakeLadderClient(kick_ladder(), kick_ladder(), kick_ladder())
    cal = calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1", client=client)
    assert not cal["ok"]
    assert all(s.get("separation", 0) < 0.2 for s in cal["sources"] if "separation" in s)
    assert any("separate" in (s.get("skip_reason") or "") for s in cal["sources"])


def test_near_constant_ladder_observe_only(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    client = _FakeLadderClient(near_constant_ladder(), near_constant_ladder(),
                               near_constant_ladder())
    cal = calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1",
                                 client=client)
    assert not cal["ok"]
    assert "usable ladder" in cal["reason"] or cal["n_valid"] < 2


def test_degenerate_yaw_observe_only(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    client = _FakeLadderClient(degenerate_yaw_ladder(), degenerate_yaw_ladder(),
                               degenerate_yaw_ladder())
    cal = calibrate_task_derived(p, "spin in place", robot_hint="Unitree-G1",
                                 client=client)
    assert not cal["ok"]
    assert any(s.get("degenerate") for s in cal["sources"])


def test_unknown_robot_observe_only(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    cal = calibrate_task_derived(p, "kick forward", robot_hint="MysteryBot",
                                 client=_FakeLadderClient(kick_ladder()))
    assert not cal["ok"] and "unknown robot" in cal["reason"]


def test_metric_source_never_in_author_payload(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    client = _FakeLadderClient(kick_ladder(), kick_ladder(), kick_ladder())
    calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1", client=client)
    for payload in client.messages.payloads:
        assert "def compute_spec" not in payload
        assert "REQUIRED_JOINT_ROLES" not in payload


def test_provenance_recorded(tmp_path):
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    client = _FakeLadderClient(kick_ladder(), kick_ladder(), kick_ladder())
    cal = calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1", client=client)
    assert len(cal["sources"]) == 3
    s0 = cal["sources"][0]
    assert s0["model_id"] and s0["payload_sha256"] and s0["style_id"] == 0
    # every source saw the SAME shared context (goal/robot/joint_names).
    ctx = {s["context_sha256"] for s in cal["sources"]}
    assert len(ctx) == 1


def test_metric_load_failure_denies(tmp_path):
    p = _write(tmp_path, "broken.py", "this is not python {{{")
    cal = calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1",
                                 client=_FakeLadderClient(kick_ladder()))
    assert not cal["ok"] and "failed to load" in cal["reason"]


def test_never_raises_on_author_crash(tmp_path):
    class _BoomMessages:
        def create(self, **kw): raise RuntimeError("api down")
    class _Boom:
        messages = _BoomMessages()
    p = _write(tmp_path, "kick.py", GOOD_KICK)
    cal = calibrate_task_derived(p, "kick forward", robot_hint="Unitree-G1", client=_Boom())
    assert not cal["ok"] and cal["n_valid"] == 0       # honest observe-only, no raise


# ── §Ship 53: L3 adversarial gaming archetypes ───────────────────────

# GOOD_KICK gates STATIONARITY, so a "walk-away while kicking" gaming policy
# scores ~0; but its raw |jv| is tremor-gameable (a hole the stationary ladders
# never test). GAMEABLE_KICK drops the stationarity gate — it ranks the same
# stationary ladder (passes L2) yet scores a travelling kicker as competent.
GAMEABLE_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    if jv is None or grav is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {})
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    return {"spec_score": float(np.clip(up * burst, 0.0, 1.0))}   # NO stationarity gate
'''


def gaming_travel() -> GamingArchetypeSet:
    """OFF-GOAL gaming policies for a STATIONARY kick — all travel/fall, which a
    stationarity-gated metric crushes but an un-gated one scores as competent."""
    return GamingArchetypeSet(goal_restated="kick from a stationary stance", archetypes=[
        GamingArchetype(name="walk_away_kicking", strategy="travel while swinging the leg",
            motion=MotionSpec(uprightness=1.0, base_height_m=0.7,
                              forward_speed_mps=1.2, groups=[_kick_swing(9.0, 4)])),
        GamingArchetype(name="fall_rhythmically", strategy="collapse in a rhythm",
            motion=MotionSpec(uprightness=0.3, base_height_m=0.5, hop_height_m=0.3, hop_count=4)),
        GamingArchetype(name="drift_sideways", strategy="sidestep while swinging",
            motion=MotionSpec(uprightness=1.0, base_height_m=0.7,
                              lateral_speed_mps=1.0, groups=[_kick_swing(9.0, 4)]))])


def gaming_tremor() -> GamingArchetypeSet:
    """A single upright-jitter policy — fakes 'activity' for any raw-|vel| metric."""
    return GamingArchetypeSet(goal_restated="kick from a stance", archetypes=[
        GamingArchetype(name="stand_and_flail", strategy="jitter upright, no real kick",
            motion=MotionSpec(uprightness=1.0, base_height_m=0.7, tremor=1.6))])


class _BothMessages:
    """A mock that serves BOTH the ladder author (CompetenceLadder) and the
    adversarial author (GamingArchetypeSet) over the JSON-mode `create` interface,
    branching on the payload (the gaming payload carries `n_archetypes`, the ladder
    payload does not)."""

    def __init__(self, ladders, gaming, gaming_raises=False):
        self._ladders = list(ladders)
        self._gaming = gaming
        self._gaming_raises = gaming_raises
        self.ladder_calls = 0
        self.gaming_calls = 0
        self.ladder_payloads: list = []
        self.gaming_payloads: list = []

    def create(self, **kw):
        content = kw.get("messages", [{}])[-1].get("content", "")
        if "n_archetypes" in content:
            self.gaming_calls += 1
            self.gaming_payloads.append(content)
            if self._gaming_raises:
                raise RuntimeError("gaming author api down")
            return _as_json_resp(self._gaming)
        self.ladder_calls += 1
        self.ladder_payloads.append(content)
        lad = self._ladders[self.ladder_calls - 1] if self.ladder_calls - 1 < len(self._ladders) \
            else self._ladders[(self.ladder_calls - 1) % len(self._ladders)]
        return _as_json_resp(lad)


class _FakeBothClient:
    def __init__(self, ladders, gaming, gaming_raises=False):
        self.messages = _BothMessages(ladders, gaming, gaming_raises)


def _three_kicks():
    return [kick_ladder(), kick_ladder(), kick_ladder()]


def test_adversarial_grants_robust_metric(tmp_path):
    """A stationarity-gated metric ranks the ladders AND scores every travel/
    fall gaming policy ~0 → still GRANTED with the L3 gate enabled."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward with the left leg from a stance",
                                 robot_hint="Unitree-G1", client=client, adversarial=True)
    assert cal["ok"], cal
    adv = cal["adversarial"]
    assert adv and adv["ran"] and not adv["gameable"]
    assert client.messages.gaming_calls == 1     # exactly one blind adversary call
    assert max(a["score"] for a in adv["archetypes"]) < adv["ceiling"]


def test_adversarial_denies_travel_gameable_metric(tmp_path):
    """KEYSTONE: a metric with NO stationarity gate ranks the stationary ladder
    perfectly (passes L2) but scores a travelling kicker as competent — L3's
    independent adversary catches the hole the ladders structurally can't."""
    p = _write(tmp_path, "gameable.py", GAMEABLE_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward with the left leg from a stance",
                                 robot_hint="Unitree-G1", client=client, adversarial=True)
    assert cal["rho_min"] >= 0.5                  # the ladders DID grant (L2 passed)
    assert not cal["ok"]                          # ...but L3 denies
    assert cal["adversarial"]["gameable"]
    assert "walk_away_kicking" in cal["reason"] and "gameable" in cal["reason"]


def test_adversarial_catches_tremor_hole_l2_missed(tmp_path):
    """The honest ladders only put tremor in the FALLEN anchor, so a raw-|jv|
    metric ranks them fine (L2 grants). The adversary's upright-jitter policy
    exposes that the metric rewards stationary flailing → DENIED."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_tremor())
    cal = calibrate_task_derived(p, "kick forward with the left leg from a stance",
                                 robot_hint="Unitree-G1", client=client, adversarial=True)
    assert cal["rho_min"] >= 0.5                  # L2 granted
    assert not cal["ok"] and cal["adversarial"]["gameable"]
    assert "stand_and_flail" in cal["reason"]


def test_adversarial_flag_off_skips_the_llm_breadth_pass(tmp_path):
    """adversarial=False → NO LLM gaming-author call is made (the path stays cheap).
    §round-13: the kick gate's deterministic losers are opt-in (a canonical kick routes
    to the built-in path), so a kick goal without the opt-in is NOT enforced and the
    GAMEABLE_KICK grant survives — but the gate record, if present, made no LLM call."""
    p = _write(tmp_path, "gameable.py", GAMEABLE_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward from a stance",
                                 robot_hint="Unitree-G1", client=client, adversarial=False)
    assert cal["ok"]                              # kick gate is opt-in → not enforced here
    adv = cal["adversarial"]
    assert adv is None or not adv.get("archetypes")   # no LLM breadth pass was run
    assert client.messages.gaming_calls == 0


def test_adversarial_author_is_metric_blind(tmp_path):
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                           client=client, adversarial=True)
    assert client.messages.gaming_payloads          # the adversary WAS called
    for payload in client.messages.gaming_payloads:
        assert "def compute_spec" not in payload
        assert "REQUIRED_JOINT_ROLES" not in payload


def test_adversarial_author_crash_not_enforced(tmp_path):
    """An adversary-call failure is NO evidence, not a denial — the L2 grant stands and the
    inconclusive author reason is recorded (never-silent). §round-19: the deterministic
    goal-blind losers now run for kicks too, so the gate RAN (carried the verdict) despite the
    author crash; GOOD_KICK scores those losers ~0 → grant survives."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel(), gaming_raises=True)
    cal = calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                                 client=client, adversarial=True)
    assert cal["ok"]                                # grant survives the crash
    adv = cal["adversarial"]
    assert adv and adv["ran"] and not adv["gameable"]            # deterministic losers carried it
    assert adv["required_losers"]                               # firewall was NOT empty
    assert "inconclusive" in adv["author_note"]                # crash still recorded (never-silent)


def test_adversarial_skipped_when_l2_denies(tmp_path):
    """The gate only spends a call when the ladders already grant — a metric
    denied at L2 never reaches L3 (adversarial stays None)."""
    p = _write(tmp_path, "bad.py", BAD_TRAVEL)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                                 client=client, adversarial=True)
    assert not cal["ok"] and cal["n_valid"] < 2     # denied at L2
    assert cal["adversarial"] is None
    assert client.messages.gaming_calls == 0


# Passes L2 (ranks the stationary kick ladder) but RAISES on any travelling
# rollout — i.e. it crashes on the walk-away/drift gaming archetypes.
RAISE_ON_TRAVEL = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is not None:
        disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
        if disp > 0.5:
            raise RuntimeError("metric boom on a travelling rollout")
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    roles = (meta or {}).get("joint_roles", {})
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if jv is None or grav is None or not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    return {"spec_score": float(np.clip(up * burst, 0.0, 1.0))}
'''


def test_adversarial_never_raises_when_metric_crashes_on_archetype(tmp_path):
    """Invariant 1 (NEVER raises): a metric that CRASHES on a gaming archetype
    degrades that probe to 'skipped' (no evidence) — the gate returns a record
    and the L2 grant stands, rather than the crash propagating."""
    p = _write(tmp_path, "raiser.py", RAISE_ON_TRAVEL)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                                 client=client, adversarial=True)
    assert cal["rho_min"] >= 0.5 and cal["ok"]          # L2 granted, gate didn't deny
    adv = cal["adversarial"]
    assert adv and not adv["gameable"]
    skipped = [a for a in adv["archetypes"] if "render/score error" in a.get("skipped", "")]
    assert skipped                                       # the travel probes were skipped


def test_adversarial_provenance_recorded(tmp_path):
    p = _write(tmp_path, "gameable.py", GAMEABLE_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                                 client=client, adversarial=True)
    adv = cal["adversarial"]
    assert adv["payload_sha256"] and adv["context_sha256"] and adv["response_sha256"]
    assert adv["model_id"] and "competent_ref" in adv
    names = {a["name"] for a in adv["archetypes"]}
    assert "walk_away_kicking" in names and all("score" in a for a in adv["archetypes"])


# ── regression: the 5 built-in families are UNCHANGED ─────────────────


@pytest.mark.parametrize("builtin", ["g1_kick", "g1_floss", "g1_jump",
                                     "go1_trot", "cartpole_balance"])
def test_builtin_calibration_unaffected(tmp_path, builtin):
    """calibrate_metric (argsort spearman, threshold 0.7) is untouched — the
    task-derived path is a disjoint sibling."""
    p = _write(tmp_path, "good.py", GOOD_KICK if "kick" in builtin else CONSTANT)
    cal = calibrate_metric(p, builtin, threshold=0.7)
    assert cal["builtin"] == builtin and "spearman" in cal and "rho_min" not in cal


# ── §Metric-quality laws (LAW 9): kick required-losers + spec_* gate scope ────
# The deterministic kick hacks WITH foot_pos_b (the direction channel render_rung
# can't render), now scored against BOTH a hand-authored spec_* metric (the
# surface that never existed → why the gate never ran on the metric that scored
# v5) and, opt-in, a generated metric.

from sculptor.eval.joint_resolver import (  # noqa: E402
    LEG_SAGITTAL_AXES,
    LEG_SAGITTAL_SEGMENTS,
    select_joints,
)
from sculptor.eval.metric_calibration import _kick_competent_reference  # noqa: E402
from sculptor.eval.spec_metrics import (  # noqa: E402
    burstiness,
    horizontal_speed,
    spec_g1_kick,
    upright_mask,
    uprightness,
)


def _old_form_kick(arrays, behavior, meta):
    """The PRE-rebuild, DIRECTION-BLIND partial-credit kick form
    (intensity·uprightness·stationarity — NO completion gate, NO signed direction):
    a faithful stand-in for the spec_g1_kick that reward-hacked g1-kick-v5. It
    scores a rear kick == a forward kick (direction-blind) → the gate must flag it."""
    import numpy as _np
    names = list((meta or {}).get("joint_names") or [])
    jv = arrays["joint_vel"]
    legs = (select_joints(names, segments=LEG_SAGITTAL_SEGMENTS, axes=LEG_SAGITTAL_AXES)
            if len(names) == jv.shape[2] else None)
    g = arrays["projected_gravity_b"]
    b = burstiness(jv, joint_indices=legs or None, valid_mask=upright_mask(g))
    intensity = 1.0 - float(_np.exp(-b["burst_p95"] / 4.0))
    up = uprightness(g)
    root = arrays.get("root_link_pos_w")
    speed = float(horizontal_speed(root)["speed_per_frame"]) if root is not None else 0.0
    stat = float(_np.clip(_np.exp(-speed / 0.01), 0.0, 1.0))
    return {"spec_score": float(_np.clip(intensity * up * stat, 0.0, 1.0))}


# A DIRECTION-AWARE generated kick metric: reads the left foot's anterior (x) to
# score a forward fraction, so a rear-kick loser scores ~0. Ranks the (footless)
# ladder via direction-abstain (=1.0). The opt-in robust counterpart to GOOD_KICK.
DIR_AWARE_KICK = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w")
    if jv is None or grav is None or root is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {})
    knees = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not knees:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    drift = float(np.linalg.norm(root[..., :2].max(0) - root[..., :2].min(0), axis=-1).mean())
    stationary = float(np.exp(-drift / 0.4))
    knee_peak = float(np.abs(jv[..., knees]).max(axis=2).max(axis=0).mean())
    burst = 1.0 - float(np.exp(-knee_peak / 8.0))
    direction = 1.0
    lf = arrays.get("left_foot_pos_b")
    if lf is not None:
        a = lf[..., 0].astype(np.float64)
        base = np.median(a, axis=0, keepdims=True); dev = a - base
        fwd = np.clip(dev, 0.0, None).max(axis=0); back = np.clip(-dev, 0.0, None).max(axis=0)
        direction = float(np.clip((fwd / (fwd + back + 1e-6)).mean(), 0.0, 1.0))
    return {"spec_score": float(np.clip(up * stationary * burst * direction, 0.0, 1.0))}
'''

_KICK_GOAL = "kick forward with the left leg from a stance"
_KICK_CHANS = ["direction", "completion", "amplitude"]


def test_kick_required_losers_frame_scoping():
    """LAW 0: frame-ambiguous losers are DROPPED so a novel kick variant is never
    false-denied. A plain forward kick gets all four; a mule kick drops the rear-
    direction loser; an explicit one-leg / lateral goal drops its support/direction
    loser (incl. the 'balancing on one leg' resolve gap the review found)."""
    g1 = list(G1_29)
    full = {l["name"] for l in kick_required_losers(g1, _KICK_GOAL, "Unitree-G1")}
    assert full == {"partial_kick", "whip_and_fall", "active_kick_behind", "one_leg_balance"}
    mule = {l["name"] for l in kick_required_losers(g1, "mule kick backward", "Unitree-G1")}
    assert "active_kick_behind" not in mule and "one_leg_balance" in mule
    oneleg = {l["name"] for l in kick_required_losers(
        g1, "kick a ball while balancing on one leg", "Unitree-G1")}
    assert "one_leg_balance" not in oneleg
    lateral = {l["name"] for l in kick_required_losers(g1, "spin kick to the side", "Unitree-G1")}
    assert "active_kick_behind" not in lateral


def test_kick_required_losers_unknown_robot_is_empty():
    """An unresolvable left leg → no probes (a coverage gap), never a wrong-joint
    deny."""
    assert kick_required_losers(["jointA", "jointB"], _KICK_GOAL) == []


def test_adversarial_gate_spec_g1_kick_not_gameable():
    """The v5 anchor: the gate now RUNS on the hand-authored spec_g1_kick (it never
    did — generated-only). The rebuilt completion_gate·min(channels) form crushes
    every kick hack (kick-behind/one-leg/partial/whip) far below the ceiling, and a
    competent forward kick pins competent_ref ≈0.78 → NOT gameable, audit-only."""
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    rec = adversarial_archetype_gate_spec(
        "g1_kick", _KICK_GOAL, client=client, robot_hint="Unitree-G1")
    assert rec["ran"] and not rec["gameable"], rec
    assert rec["competent_ref"] >= 0.75
    losers = {l["name"]: l["score"] for l in rec["required_losers"] if "score" in l}
    assert set(losers) == {"partial_kick", "whip_and_fall",
                           "active_kick_behind", "one_leg_balance"}
    assert max(losers.values()) < rec["ceiling"]
    assert rec["coverage_gaps"] == []        # forward double-support → full coverage
    assert rec["audit_only"] is True and rec["builtin"] == "g1_kick"


def test_adversarial_gate_spec_unknown_builtin_raises():
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    with pytest.raises(KeyError):
        adversarial_archetype_gate_spec("not_a_metric", _KICK_GOAL, client=client)


def test_adversarial_gate_old_form_kick_is_flagged():
    """Teeth (regression-lock): the OLD direction-blind partial-credit kick form
    (what scored g1-kick-v5) is FLAGGED gameable by the rear-kick required-loser —
    proving the gate has teeth and that the rebuild (test above) is what defeats it."""
    names = list(G1_29)
    losers = kick_required_losers(names, "kick forward from a stance", "Unitree-G1")
    competent_ref = _old_form_kick(*_kick_competent_reference(names))["spec_score"]
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    rec = adversarial_archetype_gate(
        _old_form_kick, [], names, competent_ref, client=client,
        required_losers=losers, scored_channels=_KICK_CHANS)
    assert rec["gameable"] and rec["worst_name"] == "active_kick_behind"
    assert "active_kick_behind" in rec["reason"]


def test_adversarial_gate_required_losers_run_on_author_crash():
    """WIRING: required-losers are scored even when the LLM author call CRASHES —
    a direction-blind metric is still flagged by the rear loser (the gate no longer
    early-returns past loser scoring on an author failure)."""
    names = list(G1_29)
    losers = kick_required_losers(names, "kick forward from a stance", "Unitree-G1")
    competent_ref = _old_form_kick(*_kick_competent_reference(names))["spec_score"]
    client = _FakeBothClient(_three_kicks(), gaming_travel(), gaming_raises=True)
    rec = adversarial_archetype_gate(
        _old_form_kick, [], names, competent_ref, client=client, required_losers=losers)
    assert rec["ran"] and rec["gameable"]            # losers ran despite the crash
    assert rec["worst_name"] == "active_kick_behind"
    assert not rec["archetypes"]                      # no LLM archetype scored


def test_adversarial_gate_clean_path_byte_identical():
    """required_losers/scored_channels unset → no loser scoring, no coverage keys
    (the Ship-53 path, exercised byte-identically by the suite above)."""
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    rec = adversarial_archetype_gate(
        lambda a, b, m: {"spec_score": 0.0}, [], list(G1_29), 0.8, client=client)
    assert rec["required_losers"] == []
    assert "coverage" not in rec and "coverage_gaps" not in rec
    assert rec["ran"] and not rec["gameable"]


def test_adversarial_gate_coverage_gap_flagged_not_denied():
    """A mule kick drops the direction loser (LAW 0); the per-channel coverage
    obligation FLAGS 'direction' as uncovered but NEVER denies on the gap."""
    names = list(G1_29)
    losers = kick_required_losers(names, "mule kick backward", "Unitree-G1")
    competent_ref = spec_g1_kick(*_kick_competent_reference(names))["spec_score"]
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    rec = adversarial_archetype_gate(
        spec_g1_kick, [], names, competent_ref, client=client,
        required_losers=losers, scored_channels=_KICK_CHANS)
    assert rec["coverage"]["direction"] is False
    assert "direction" in rec["coverage_gaps"]
    assert not rec["gameable"]               # a gap is a flag, not a deny


def test_adversarial_required_losers_opt_in_denies_direction_blind(tmp_path):
    """OPT-IN kick losers on the GENERATED task-derived path: a DIRECTION-BLIND
    metric (GOOD_KICK) ranks the kick ladder (L2 grants) but is DENIED by the rear
    loser — and only because the loser's roles are injected (knees resolve →
    ~0.63), the masking bug the review flagged."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(
        p, "kick forward from a stance", robot_hint="Unitree-G1", client=client,
        adversarial=True, adversarial_required_losers=True)
    assert cal["rho_min"] >= 0.5                       # ladders granted (L2)
    assert not cal["ok"] and cal["adversarial"]["gameable"]
    assert cal["adversarial"]["worst_name"] == "active_kick_behind"


def test_adversarial_required_losers_opt_in_grants_direction_aware(tmp_path):
    """The same opt-in path GRANTS a DIRECTION-AWARE metric — the rear loser scores
    ~0 (the foot swings backward), so the losers don't deny a robust metric."""
    p = _write(tmp_path, "dir.py", DIR_AWARE_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(
        p, "kick forward from a stance", robot_hint="Unitree-G1", client=client,
        adversarial=True, adversarial_required_losers=True)
    assert cal["ok"], cal
    adv = cal["adversarial"]
    assert adv["ran"] and not adv["gameable"]
    rear = next(l for l in adv["required_losers"] if l["name"] == "active_kick_behind")
    assert rear["score"] < adv["ceiling"]


def test_adversarial_required_losers_off_runs_general_firewall_for_kicks(tmp_path):
    """§round-19 [HIGH FALSE GRANT] fix C: adversarial_required_losers default OFF no longer
    means NO firewall on a novel kick goal (that was the bug — the gate scored zero losers and
    fail-open-granted a confound). It now runs the GENERAL goal-blind losers; the flag only
    swaps in the DEDICATED kick losers when ON. GOOD_KICK scores the general losers ~0 so it
    still GRANTS — but the firewall is present (non-empty), not absent."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(
        p, "kick forward from a stance", robot_hint="Unitree-G1", client=client,
        adversarial=True)
    assert cal["ok"], cal
    names = {l["name"] for l in cal["adversarial"]["required_losers"]}
    assert names == {"do_nothing_upright", "jitter_in_place", "collapse_and_stay_down"}


# ── §fold-and-return primitive: STEERING for fold/squat/sit-to-stand/toe-touch ──
# render_rung's base_height_m is a monotone ramp and CANNOT render a pelvis dip-and-
# return, so a correct toe-touch/squat metric could rank no competence ladder and
# stayed observe-only. The fold primitive (fold_depth_m + a "fold" group mode) renders
# the dip-and-return that lets it rank a ladder and EARN steer-rights.

from sculptor.eval.generated_metric import (  # noqa: E402
    inject_joint_roles,
    load_generated_metric,
    read_required_roles,
)
from sculptor.eval.ladder_synth import _FOLD_ARC, render_rung  # noqa: E402
from sculptor.eval.metric_calibration import _TD_SEPARATION_MIN  # noqa: E402

# A faithful, compact toe-touch metric (the on-disk g1-toe-touching/gen_001 form):
# completion_gate · min(pelvis dip-and-return, hip ROM, knee ROM); upright at the
# ENDS only; no-travel veto. Scores the full fold-and-return ≈0.83 (the C1 probe).
TOE_TOUCH = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee"]
def compute_spec(arrays, behavior, meta):
    jp = arrays.get("joint_pos"); root = arrays.get("root_link_pos_w")
    pg = arrays.get("projected_gravity_b")
    if jp is None or root is None or pg is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {}) or {}
    hip = [roles[r] for r in ("left_hip_pitch", "right_hip_pitch") if r in roles]
    knee = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
    if not hip or not knee:
        return {"spec_score": 0.0}
    edge = max(3, jp.shape[0] // 20)
    z = root[..., 2]
    z0 = np.mean(z[:edge], axis=0); z1 = np.mean(z[-edge:], axis=0); zmin = np.min(z, axis=0)
    drop = z0 - zmin; ret = np.abs(z1 - z0)
    amp_drop = float(np.mean(np.clip((drop - 0.20) / 0.15, 0.0, 1.0)))
    ret_ch = float(np.mean(np.exp(-(ret / 0.05) ** 2)))
    hr = np.mean(np.max(jp[..., hip], axis=0) - np.min(jp[..., hip], axis=0), axis=-1)
    hip_ch = float(np.mean(np.clip((hr - 0.7) / 0.6, 0.0, 1.0)))
    kr = np.mean(np.max(jp[..., knee], axis=0) - np.min(jp[..., knee], axis=0), axis=-1)
    knee_ch = float(np.mean(np.clip((kr - 0.5) / 0.7, 0.0, 1.0)))
    gz = pg[..., 2]
    up0 = float(np.mean(np.mean(gz[:edge], axis=0) < -0.85))
    up1 = float(np.mean(np.mean(gz[-edge:], axis=0) < -0.85))
    xy = root[..., :2]
    travel = np.sqrt(np.sum((np.max(xy, axis=0) - np.min(xy, axis=0)) ** 2, axis=-1))
    no_travel = float(np.mean(travel < 0.35))
    gate = 1.0
    gate *= 1.0 if float(np.mean(drop > 0.20)) > 0.7 else 0.0
    gate *= 1.0 if float(np.mean(ret < 0.075)) > 0.7 else 0.0
    gate *= 1.0 if up0 > 0.7 else 0.0
    gate *= 1.0 if up1 > 0.7 else 0.0
    gate *= 1.0 if no_travel > 0.7 else 0.0
    return {"spec_score": float(np.clip(gate * min(amp_drop, ret_ch, hip_ch, knee_ch), 0.0, 1.0))}
'''


def _fold_group(amp: float) -> Group:
    """A `fold`-mode group driving the sagittal hips + knees (both sides) through a
    flex-and-return arc of ROM `amp` — the joints a toe-touch metric reads."""
    return Group(name="legs", mode="fold", amplitude_rad=amp,
                 role_query=RoleQuery(segments=["hip", "knee"], axes=["pitch", None],
                                      sides=["left", "right"]))


def fold_ladder() -> CompetenceLadder:
    """A monotone fold-and-return ladder: shallow→full pelvis dip co-varied with
    small→full leg ROM (what an honest toe-touch/squat author emits)."""
    return CompetenceLadder(competence_axis="pelvis fold depth + leg flexion ROM", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=d, groups=[_fold_group(a)])
        for d, a in [(0.24, 0.80), (0.28, 0.95), (0.32, 1.15), (0.35, 1.30)]])


def test_fold_default_is_byte_identical():
    """Default fold_depth_m is 0 and a non-fold spec leaves the pelvis FLAT — the
    primitive only triggers on the targeted condition, so every existing rung
    (the 5 families, the kick hacks) renders unchanged."""
    assert MotionSpec().fold_depth_m == 0.0
    a, _, _ = render_rung(MotionSpec(uprightness=1.0, base_height_m=0.7), G1)
    assert np.allclose(a["root_link_pos_w"][:, 0, 2], 0.7)            # no dip
    # _FOLD_ARC is the 0→1→0 arc, peaking at mid, returning to ~0.
    assert _FOLD_ARC[0] == 0.0 and _FOLD_ARC[len(_FOLD_ARC) // 2] == pytest.approx(1.0)
    assert _FOLD_ARC[-1] < 1e-2


def test_fold_rung_dips_pelvis_and_flexes_joints():
    """A fold rung dips the pelvis by ~fold_depth at mid and returns to start, and a
    fold-mode group flexes its joints to ~amplitude and returns — in phase."""
    a, _, _ = render_rung(
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=0.35,
                   groups=[_fold_group(1.2)]), G1)
    z = a["root_link_pos_w"][:, 0, 2]
    assert z[0] == pytest.approx(0.7) and z[-1] == pytest.approx(0.7, abs=1e-2)
    assert z.min() == pytest.approx(0.35, abs=1e-3)                   # dips by 0.35
    assert z.argmin() == len(z) // 2                                  # at mid (in phase)
    hp = G1.index("left_hip_pitch_joint"); kn = G1.index("left_knee_joint")
    jp = a["joint_pos"][:, 0, :]
    for j in (hp, kn):
        assert jp[0, j] == pytest.approx(0.0) and jp[-1, j] == pytest.approx(0.0, abs=1e-2)
        assert jp[:, j].max() == pytest.approx(1.2, abs=1e-3)        # ROM = amplitude
    # a non-targeted joint (a wrist) is untouched.
    assert np.allclose(jp[:, G1.index("left_wrist_roll_joint")], 0.0)


def test_fold_ladder_ranks_toe_touch_monotone(tmp_path):
    """KEYSTONE: the real toe-touch metric ranks a rendered fold ladder MONOTONICALLY
    with separation ≥ the calibration threshold — exactly what earns steer-rights."""
    p = _write(tmp_path, "toe.py", TOE_TOUCH)
    fn = load_generated_metric(p); roles = read_required_roles(p)
    out = render_ladder(fold_ladder().rungs, G1)
    assert not out["degenerate"] and out["n"] == 5
    scores = []
    for arrays, beh, meta in out["rungs"]:
        inject_joint_roles(meta, roles)
        scores.append(float(fn(arrays, beh, meta).get("spec_score", 0.0)))
    assert scores == sorted(scores)                                  # monotone
    assert len(set(round(s, 6) for s in scores)) == 5                # distinct
    assert scores[-1] - scores[0] >= _TD_SEPARATION_MIN              # separated
    assert spearman_midrank(scores, list(range(len(scores)))) >= 0.8
    assert scores[-1] >= 0.8                                          # full fold ≈ ideal


def test_calibrate_task_derived_grants_fold_metric(tmp_path):
    """End-to-end: with three independently-authored fold ladders, the toe-touch
    metric earns the task-derived grant — the steering the dip-and-return unlocks."""
    p = _write(tmp_path, "toe.py", TOE_TOUCH)
    client = _FakeLadderClient(fold_ladder(), fold_ladder(), fold_ladder())
    cal = calibrate_task_derived(p, "touch your toes then stand back up",
                                 robot_hint="Unitree-G1", client=client)
    assert cal["ok"], cal
    assert cal["n_valid"] == 3 and cal["rho_min"] >= 0.5
    assert all(s.get("separation", 0) >= _TD_SEPARATION_MIN
               for s in cal["sources"] if "separation" in s)


def test_fold_mode_is_single_arc_regardless_of_period():
    """A fold is one synchronized posture arc over the rollout — period_frames/phase
    do not fragment it into reps (so it stays in phase with the pelvis dip)."""
    base = render_rung(MotionSpec(groups=[_fold_group(1.0)]), G1)[0]["joint_pos"]
    g = _fold_group(1.0); g.period_frames = 12; g.phase = 1.3; g.within_group_phase_spread = 2.0
    varied = render_rung(MotionSpec(groups=[g]), G1)[0]["joint_pos"]
    assert np.array_equal(base, varied)


def test_fold_dip_clamps_pelvis_at_floor():
    """A deep fold from a LOW base degrades to a physical posture (z ≥ 0), never a
    sub-floor pelvis (degrade-to-safe). The keystone fold (base 0.7, depth 0.35) is far
    above the floor, so the clamp never fires there — byte-identical for real ladders."""
    a, _, _ = render_rung(MotionSpec(base_height_m=0.2, fold_depth_m=0.6), G1)
    assert a["root_link_pos_w"][:, 0, 2].min() >= 0.0
    b, _, _ = render_rung(MotionSpec(base_height_m=0.7, fold_depth_m=0.35), G1)
    assert b["root_link_pos_w"][:, 0, 2].min() == pytest.approx(0.35, abs=1e-3)


# A pelvis-DEPTH-ONLY proxy: scores vertical excursion alone — no completion gate, no
# leg ROM, no return-check, no uprightness. It tracks the dominant axis of a naive fold
# ladder, so a DISCRIMINATING ladder must DENY it (the firewall the review asked for).
DEPTH_ONLY = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None:
        return {"spec_score": 0.0}
    z = root[..., 2]
    return {"spec_score": float(np.clip((z.max(0) - z.min(0)).mean(), 0.0, 1.0))}
'''


def discriminating_fold_ladder() -> CompetenceLadder:
    """A fold ladder whose LOW rung is a DEEP pelvis dip with NO leg flexion — the real
    gate·min(hip ROM, knee ROM, dip) metric crushes it to 0 via the empty ROM channels,
    but a pelvis-depth-only proxy scores it HIGH and ranks the ladder NON-monotonically.
    This is what makes the ladder test the metric instead of tracking dip depth."""
    return CompetenceLadder(competence_axis="completed fold (dip + leg flexion)", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=0.35),          # deep dip, NO flex
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=0.28, groups=[_fold_group(0.95)]),
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=0.32, groups=[_fold_group(1.15)]),
        MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=0.35, groups=[_fold_group(1.30)]),
    ])


def test_fold_ladder_discriminates_real_metric_from_depth_proxy(tmp_path):
    """The fold ladder is DISCRIMINATING, not depth-trackable: with a deep-dip-NO-flex
    gaming low rung, the real gate·min(channels) toe-touch metric earns the grant while a
    pelvis-DEPTH-ONLY proxy is DENIED (it scores the flex-less rung high and breaks rank).
    Proves 'gate · min(channels)' is load-bearing — the firewall the review flagged."""
    real = _write(tmp_path, "toe.py", TOE_TOUCH)
    proxy = _write(tmp_path, "depth.py", DEPTH_ONLY)

    def _cal(p):
        return calibrate_task_derived(
            p, "touch your toes then stand back up", robot_hint="Unitree-G1",
            client=_FakeLadderClient(discriminating_fold_ladder(),
                                     discriminating_fold_ladder(),
                                     discriminating_fold_ladder()))

    real_cal, proxy_cal = _cal(real), _cal(proxy)
    assert real_cal["ok"], real_cal                  # the honest metric grants
    assert not proxy_cal["ok"], proxy_cal            # the depth-only proxy is DENIED
    # gen_scores index 0 = the prepended fallen anchor; index 1 = the deep-dip-no-flex rung.
    real_scores = next(s["gen_scores"] for s in real_cal["sources"] if "gen_scores" in s)
    proxy_scores = next(s["gen_scores"] for s in proxy_cal["sources"] if "gen_scores" in s)
    assert real_scores[1] == 0.0                     # real metric: no flex → min channel 0
    assert proxy_scores[1] > proxy_scores[2]         # proxy ranks the flex-less rung too high


# ── §calibration JSON-mode authors (the steering-path fix) ────────────────────
# The CompetenceLadder / GamingArchetypeSet schemas are too complex for the API
# grammar compiler used by messages.parse (it 400s "schema is too complex" / hangs
# "grammar compilation timed out"), so the authors now use messages.CREATE + JSON
# parse. These pin the JSON extraction + that the author round-trips a real ladder.
from sculptor.eval.metric_calibration import _author_structured, _extract_json_obj


def test_extract_json_obj_handles_fence_and_prose():
    assert _extract_json_obj('{"a": 1}') == {"a": 1}
    assert _extract_json_obj('```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json_obj('Here is the ladder:\n{"a": 3}\nDone.') == {"a": 3}
    # trailing prose after a complete object is tolerated (raw_decode stops at the close)
    assert _extract_json_obj('{"a": 4, "b": [1,2]} trailing junk')["b"] == [1, 2]


def test_extract_json_obj_raises_on_garbage():
    import pytest as _pytest
    for bad in ("", "   ", "no json here", "[1,2,3]"):
        with _pytest.raises(ValueError):
            _extract_json_obj(bad)


def test_author_structured_roundtrips_via_create():
    """_author_structured calls messages.CREATE (not parse) and JSON-parses the text
    block into the pydantic schema — even when the model wraps it in a code fence."""
    lad = fold_ladder()

    class _M:
        def __init__(self): self.created = 0; self.used_parse = False
        def parse(self, **kw): self.used_parse = True; raise AssertionError("must not call parse")
        def create(self, **kw):
            self.created += 1
            return _Resp("```json\n" + json.dumps(lad.model_dump()) + "\n```")

    class _C:
        def __init__(self): self.messages = _M()

    c = _C()
    out = _author_structured(c, "m", "sys-prompt", {"behavior_goal": "x"}, CompetenceLadder)
    assert c.messages.created == 1 and not c.messages.used_parse
    assert [r.fold_depth_m for r in out.rungs] == [r.fold_depth_m for r in lad.rungs]


def test_author_structured_truncation_raises():
    """A max_tokens-truncated author response raises (the caller records a skip — a
    truncated/garbled author NEVER grants)."""
    import pytest as _pytest

    class _Trunc:
        stop_reason = "max_tokens"
        content = [_Blk('{"competence_axis": "x", "rungs": [')]   # incomplete

    class _M:
        def create(self, **kw): return _Trunc()
    class _C:
        messages = _M()
    with _pytest.raises(ValueError):
        _author_structured(_C(), "m", "sys", {"g": 1}, CompetenceLadder)


# ── §round-4 review hardening: length-robust echo-guard + all-candidate JSON scan ──
from sculptor.eval.metric_calibration import _echoes_source, _iter_json_objs


def test_echo_guard_length_robust():
    """The soft anti-collusion echo-guard catches a SHORT source and a source echoed
    only at its TAIL — the prior fixed-stride scan left <40-char and tail regions
    unscanned (round-4 review finding)."""
    short = "def compute_spec(x): return 1"          # 29 chars (<= window)
    assert _echoes_source(short, "blah " + short + " blah")
    assert not _echoes_source(short, "totally unrelated text")
    long = "A" * 60 + "UNIQUE_TAIL_TOKEN_1234567890XYZ"   # tail beyond a full stride
    assert _echoes_source(long, "noise " + long[-40:] + " noise")   # tail-only echo caught
    assert not _echoes_source("", "anything") and not _echoes_source("x", "")
    assert not _echoes_source("tiny", "tiny")           # below _ECHO_MIN → not flagged


def test_iter_json_objs_skips_decoy_before_genuine():
    """_iter_json_objs yields ALL objects in order, so _author_structured can skip a
    malformed/decoy block that appears BEFORE the genuine ladder (round-4 finding)."""
    text = '```json\n{"not": "a ladder"}\n```\nthen the real one:\n{"competence_axis": "real", "rungs": []}'
    objs = list(_iter_json_objs(text))
    assert {"not": "a ladder"} in objs
    assert any(o.get("competence_axis") == "real" for o in objs)


def test_author_structured_skips_decoy_picks_valid():
    """When the first JSON object fails schema validation, _author_structured tries the
    next candidate and returns the first that VALIDATES (decoy can't shadow genuine)."""
    from sculptor.eval.metric_calibration import _author_structured
    lad = fold_ladder()
    decoy = '{"rungs": "this is not a list, fails validation"}'
    body = decoy + "\n\n" + json.dumps(lad.model_dump())

    class _M:
        def create(self, **kw): return _Resp(body)
    class _C:
        messages = _M()
    out = _author_structured(_C(), "m", "sys", {"g": 1}, CompetenceLadder)
    assert [r.fold_depth_m for r in out.rungs] == [r.fold_depth_m for r in lad.rungs]


# ── §round-5 review: empty-shadow false-grant guard + RecursionError containment ──
def test_author_structured_rejects_empty_shadow_picks_real_set():
    """§round-5 FALSE-GRANT fix: a trivial leading JSON object ({} / {"x":1}) VALIDATES
    against GamingArchetypeSet/CompetenceLadder with EMPTY content (all-default schemas),
    so it must NOT shadow the genuine non-empty set — else the gaming gate sees 0
    archetypes and fails OPEN (a gameable metric grants). _author_structured now skips
    empty-content objects and returns the first NON-empty one."""
    from sculptor.eval.metric_calibration import _author_structured
    real = GamingArchetypeSet(goal_restated="kick", archetypes=[
        GamingArchetype(name="flail", strategy="jitter",
                        motion=MotionSpec(uprightness=1.0, tremor=1.8))])

    class _M:
        def __init__(self, text): self._t = text
        def create(self, **kw): return _Resp(self._t)
    class _C:
        def __init__(self, text): self.messages = _M(text)

    # empty {} BEFORE the real set → must return the real (non-empty) set
    shadow = 'note: {"example": true} ' + json.dumps(real.model_dump())
    out = _author_structured(_C(shadow), "m", "sys", {"g": 1}, GamingArchetypeSet)
    assert len(out.archetypes) == 1 and out.goal_restated == "kick"
    # ladder shadow too
    lad = fold_ladder()
    lout = _author_structured(_C('{} ' + json.dumps(lad.model_dump())), "m", "sys",
                              {"g": 1}, CompetenceLadder)
    assert len(lout.rungs) == len(lad.rungs)


def test_author_structured_all_empty_raises_failsafe():
    """If EVERY candidate object is empty/trivial, _author_structured RAISES → the caller
    records a skip (fail-SAFE: inconclusive, never a silent empty accept that fails the
    gaming gate open)."""
    import pytest as _pytest
    from sculptor.eval.metric_calibration import _author_structured

    class _M:
        def create(self, **kw): return _Resp('{} {"x":1} {"foo":"bar"}')
    class _C:
        messages = _M()
    with _pytest.raises(Exception):
        _author_structured(_C(), "m", "sys", {"g": 1}, GamingArchetypeSet)


def test_iter_json_objs_contains_recursion_error(monkeypatch):
    """§round-5: deeply-nested JSON makes raw_decode raise RecursionError (NOT a
    ValueError) — _iter_json_objs must CONTAIN it (skip that brace, keep scanning),
    never propagate. Monkeypatch raw_decode to raise RecursionError on the first brace
    (deterministic across recursion-limit differences); the second genuine object is
    still found."""
    import json as _json
    from sculptor.eval import metric_calibration as mc
    real = _json.JSONDecoder.raw_decode
    state = {"n": 0}

    def boom(self, s, *a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise RecursionError("nested too deep")
        return real(self, s, *a, **k)

    monkeypatch.setattr(_json.JSONDecoder, "raw_decode", boom)
    out = list(mc._iter_json_objs('{"x": 1} {"competence_axis": "ok", "rungs": []}'))
    assert {"competence_axis": "ok", "rungs": []} in out   # contained, scan continued


# ── §round-6 review: a NaN spec_score must NOT false-grant ─────────────────────
NAN_TOP_RUNG = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_hip_pitch","right_hip_pitch","left_knee","right_knee"]
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None: return {"spec_score": 0.0}
    z = root[..., 2]
    drop = float(np.mean(z[:5].mean(0) - z.min(0)))
    # finite + increasing on shallow rungs, but NaN on the DEEPEST rung (fold_ladder top
    # = 0.35) — pre-fix this ranked as max (rho~1.0) and slipped the std/separation gates
    # (`nan < x` is False), false-granting; post-fix NaN is coerced to 0.
    if drop > 0.33:
        return {"spec_score": float("nan")}
    return {"spec_score": float(np.clip(drop / 0.4, 0.0, 1.0))}
'''


def test_nan_spec_score_does_not_false_grant(tmp_path):
    """§round-6 FALSE-GRANT fix: a metric returning NaN on the top ladder rung must NOT
    earn steer-rights — NaN is coerced to 0.0 (penalized), so it can't manufacture a
    spurious rho/separation. Uses the deterministic fold ladders (no LLM)."""
    p = _write(tmp_path, "nan.py", NAN_TOP_RUNG)
    client = _FakeLadderClient(fold_ladder(), fold_ladder(), fold_ladder())
    cal = calibrate_task_derived(p, "touch your toes then stand back up",
                                 robot_hint="Unitree-G1", client=client)
    assert cal["ok"] is False        # the NaN top rung does NOT grant
    # and no source reports a spurious near-perfect rho off the NaN
    for s in cal.get("sources", []):
        if s.get("rho") is not None:
            assert s["rho"] < 0.99 or s.get("separation", 0) < 0.2


def test_adversarial_gate_nan_hack_is_gameable():
    """§round-7 FALSE-GRANT fix: a gaming probe the metric scores NaN must be treated as
    GAMEABLE (fail-closed) — `NaN > worst` is False, so a NaN hack would otherwise drop
    out of `worst` and ESCAPE the denial. The required-loser path runs regardless of the
    LLM author."""
    import numpy as np
    from sculptor.eval.metric_calibration import _NAMES_12, adversarial_archetype_gate

    def nan_metric(arrays, behavior, meta):
        return {"spec_score": float("nan")}     # scores every hack NaN

    loser = {"name": "tremor_hack", "channel": "amplitude",
             "arrays": {"joint_pos": np.zeros((120, 4, 12)),
                        "joint_vel": np.zeros((120, 4, 12)),
                        "projected_gravity_b": np.tile([0, 0, -1.0], (120, 4, 1)),
                        "root_link_pos_w": np.zeros((120, 4, 3))},
             "behavior": {"max_episode_steps": 120, "rollout_num_envs": 4, "step_dt": 0.02},
             "meta": {"joint_names": list(_NAMES_12)}}

    class _NoAuthor:
        def create(self, **kw): raise RuntimeError("no author needed")
    class _C:
        messages = _NoAuthor()

    rec = adversarial_archetype_gate(
        nan_metric, [], list(_NAMES_12), competent_ref=0.78, client=_C(),
        required_losers=[loser], scored_channels=["amplitude"])
    assert rec["ran"] is True
    assert rec["gameable"] is True          # NaN hack → gameable, not silently dropped
    assert rec["ok"] is False


# ── §round-13 RED-TEAM FALSE-GRANT fixes ─────────────────────────────────────
# The red-team CONSTRUCTED false grants: a depth-only / posture-only proxy earned
# steer-rights because the blind ladders co-vary the confound with the rung and the
# only goal-aware defense (the adversarial gate) was off-by-default / kick-only /
# fail-open. The deterministic goal-blind losers now run ALWAYS for a novel grant.

def test_round13_naive_ladder_denies_depth_proxy_via_deterministic_loser(tmp_path):
    """With a NAIVE monotone fold ladder (what a real blind LLM emits — NOT the synthetic
    discriminating ladder), a pelvis-DEPTH-ONLY proxy used to FALSE-GRANT. The always-on
    deterministic gate (collapse_and_stay_down) now DENIES it in the DEFAULT path
    (adversarial=False), while the honest gate·min toe-touch metric still GRANTS — no
    reliance on the ladder happening to be discriminating."""
    real = _write(tmp_path, "toe.py", TOE_TOUCH)
    proxy = _write(tmp_path, "depth.py", DEPTH_ONLY)

    def _cal(p):
        return calibrate_task_derived(
            p, "touch your toes then stand back up", robot_hint="Unitree-G1",
            client=_FakeLadderClient(fold_ladder(), fold_ladder(), fold_ladder()))

    real_cal, proxy_cal = _cal(real), _cal(proxy)
    assert real_cal["ok"], real_cal                      # honest metric still grants
    assert not proxy_cal["ok"], proxy_cal                # depth proxy now DENIED by default
    assert proxy_cal["adversarial"]["gameable"]
    assert "collapse_and_stay_down" in (proxy_cal["reason"] or "")


def test_round13_general_losers_flag_confound_proxies_not_real_metric():
    """Gate-level: the general deterministic losers flag a posture-only AND a depth-only
    proxy (gameable) but score an honest fold metric ~0 (not gameable)."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate)
    from sculptor.eval.generated_metric import load_generated_metric
    import tempfile, os
    losers = general_required_losers(G1, "wave your arm then lower it")

    def _gate(fn, roles=None):
        return adversarial_archetype_gate(
            fn, roles or [], G1, competent_ref=1.0, client=None,
            required_losers=losers, scored_channels=["posture", "completion"],
            author=False)

    def posture_only(arrays, behavior, meta):
        import numpy as np
        g = arrays.get("projected_gravity_b"); root = arrays.get("root_link_pos_w")
        if g is None or root is None:
            return {"spec_score": 0.0}
        up = float(np.mean(np.clip(-g[..., 2], 0, 1)))
        z = float(np.mean(root[..., 2]) / 0.7)
        return {"spec_score": float(np.clip(0.5 * up + 0.5 * z, 0.0, 1.0))}

    def depth_only(arrays, behavior, meta):
        import numpy as np
        root = arrays.get("root_link_pos_w")
        if root is None:
            return {"spec_score": 0.0}
        z = root[..., 2]
        return {"spec_score": float(np.clip((z.max(0) - z.min(0)).mean(), 0.0, 1.0))}

    assert _gate(posture_only)["gameable"]               # do_nothing_upright scores it max
    assert _gate(depth_only)["gameable"]                 # collapse_and_stay_down scores it high
    # an honest fold metric scores every general loser ~0 → not gameable
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "toe.py")
        open(mp, "w").write(TOE_TOUCH)
        rec = _gate(load_generated_metric(mp), [])
    assert not rec["gameable"], rec


def test_round13_terminal_down_goal_omits_collapse_loser():
    """A goal whose competent end-state is DOWN (lie/rest, no return) must NOT use
    collapse_and_stay_down as a loser (it is ON-goal there → would false-deny)."""
    from sculptor.eval.metric_calibration import general_required_losers
    fold = [l["name"] for l in general_required_losers(G1, "touch your toes then stand up")]
    lie = [l["name"] for l in general_required_losers(G1, "lie down on the floor and rest")]
    assert "collapse_and_stay_down" in fold
    assert "collapse_and_stay_down" not in lie           # on-goal → not a loser
    assert "do_nothing_upright" in lie                   # still off-goal everywhere


def test_round13_role_query_axes_null_does_not_break_set():
    """§round-13 fail-open fix: the blind gaming/ladder author commonly emits
    `axes: null`; the strict list type used to raise ValidationError and drop the whole
    GamingArchetypeSet → the gate fell to ran=False (not enforced → false grant). A null
    now coerces to the default instead of nuking the set."""
    from sculptor.eval.ladder_synth import RoleQuery
    rq = RoleQuery.model_validate({"segments": ["arm"], "axes": None})
    assert rq.axes == ["pitch", None]                    # coerced, no ValidationError
    rq2 = RoleQuery.model_validate({"segments": ["arm"], "axes": "pitch"})
    assert rq2.axes == ["pitch"]                         # scalar wrapped


# ── §round-14 verify the round-13 fix: close the fail-open + 2 false-rejects ──

def test_round15_loser_raise_is_runtime_equivalent_zero_no_fail_open():
    """§round-15 (supersedes round-14's fail-closed): a metric that RAISES on a loser is
    scored 0.0 and COUNTED (ran=True → NO fail-open via an empty set), mirroring the
    runtime contract (compute_generated_metric returns 0.0 on a raise). This is correct
    because such a metric would ALSO raise on the corresponding real degenerate POLICY at
    runtime → score it 0.0 → it CANNOT reward that policy → it is not gaming via the loser.
    (Round-14 fail-closed false-rejected honest onset/phase metrics; round-13 skip
    fail-opened via ran=False.)"""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate)
    losers = general_required_losers(G1, "touch your toes then stand back up")

    def raises_on_static(arrays, behavior, meta):
        # an HONEST return-gated fold metric that TIMES the fold via an onset index —
        # raises IndexError on a non-descending loser, exactly like a real generated
        # metric, and scores a collapse 0.0 (no up-return), so it is NOT a depth proxy.
        import numpy as np
        pz = arrays["root_link_pos_w"][..., 2].mean(1)
        onset = np.nonzero(pz < pz[0] - 1e-3)[0][0]   # empty -> IndexError on a static loser
        depth = pz[0] - pz.min(); ret = pz[-1] - pz.min()
        return {"spec_score": float(np.clip(min(depth / 0.3, ret / 0.3), 0.0, 1.0))}

    rec = adversarial_archetype_gate(
        raises_on_static, [], G1, competent_ref=1.0, client=None,
        required_losers=losers, scored_channels=["posture", "completion"], author=False)
    assert rec["ran"] is True                          # losers counted → NO fail-open
    assert all(l.get("score") == 0.0 for l in rec["required_losers"])  # raise → 0.0
    assert rec["gameable"] is False                    # honest metric NOT false-rejected


def test_round14_static_hold_goal_drops_stillness_losers():
    """§round-14 HIGH: a balance/steady goal's competent behavior IS a still upright
    stance, so do_nothing_upright/jitter are ON-goal there and must be dropped (else a
    genuine balance metric is false-denied). collapse_and_stay_down still applies."""
    from sculptor.eval.metric_calibration import general_required_losers
    hold = [l["name"] for l in general_required_losers(G1, "stand still and balance steadily")]
    active = [l["name"] for l in general_required_losers(G1, "wave your arm then lower it")]
    assert "do_nothing_upright" not in hold and "jitter_in_place" not in hold
    assert "collapse_and_stay_down" in hold          # a balance task must not collapse
    assert "do_nothing_upright" in active            # an active goal keeps the stillness losers

    def balance_metric(arrays, behavior, meta):
        import numpy as np
        g = arrays.get("projected_gravity_b")
        return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0, 1)))} if g is not None else {"spec_score": 0.0}

    from sculptor.eval.metric_calibration import adversarial_archetype_gate
    rec = adversarial_archetype_gate(
        balance_metric, [], G1, competent_ref=1.0, client=None,
        required_losers=general_required_losers(G1, "stand still and balance steadily"),
        scored_channels=["posture", "completion"], author=False)
    assert not rec["gameable"]                        # balance metric NOT false-denied


def test_round14_small_gesture_metric_not_false_rejected():
    """§round-14 MEDIUM: the shrunk jitter (ROM ~0.07 rad) must NOT false-reject an honest
    small-amplitude gesture metric (target ~0.3 rad), while still catching a 'rewards any
    joint motion' velocity proxy (jitter's 15 Hz gives high velocity at small ROM)."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate)
    losers = general_required_losers(G1, "give a small shoulder shrug gesture")

    def gentle_gesture(arrays, behavior, meta):  # rewards ROM normalized by a 0.3 rad target
        import numpy as np
        jp = arrays.get("joint_pos")
        if jp is None:
            return {"spec_score": 0.0}
        return {"spec_score": float(np.clip((jp.max(0) - jp.min(0)).mean() / 0.30, 0.0, 1.0))}

    rec = adversarial_archetype_gate(
        gentle_gesture, [], G1, competent_ref=1.0, client=None,
        required_losers=losers, scored_channels=["posture", "completion"], author=False)
    js = next(l["score"] for l in rec["required_losers"] if l["name"] == "jitter_in_place")
    assert js < 0.5                                   # gesture reads jitter's small ROM low
    assert not rec["gameable"]                        # → NOT false-rejected

    def any_velocity(arrays, behavior, meta):  # gaming: rewards generic joint velocity
        import numpy as np
        jv = arrays.get("joint_vel")
        if jv is None:
            return {"spec_score": 0.0}
        return {"spec_score": float(np.clip(np.abs(jv).mean() / 1.5, 0.0, 1.0))}

    rec2 = adversarial_archetype_gate(
        any_velocity, [], G1, competent_ref=1.0, client=None,
        required_losers=losers, scored_channels=["posture", "completion"], author=False)
    assert rec2["gameable"]                           # a velocity proxy is STILL caught


def test_round15_loser_set_never_empty_and_off_goal_per_class():
    """§round-15 DEFECT 2: the carve-outs must NEVER empty the loser set, and every loser
    must be OFF-goal for its goal class (else the gate is unenforced → false grant)."""
    from sculptor.eval.metric_calibration import general_required_losers
    cases = {
        "touch your toes then stand back up": {"do_nothing_upright", "jitter_in_place", "collapse_and_stay_down"},
        "balance on one leg":                 {"collapse_and_stay_down"},                       # still-upright is on-goal
        # §round-19/21 fix: a terminal-down goal drops collapse_and_stay_down (on-goal) but adds
        # TWO thrashing probes — collapse_and_thrash (constant-low, stillness channel) and
        # descend_and_thrash (pelvis ramp, descent channel) — so a low-only OR a descent-only
        # proxy at the low posture is no longer free, without false-rejecting an honest still-low.
        "lie down to rest":                   {"do_nothing_upright", "jitter_in_place", "collapse_and_thrash", "descend_and_thrash"},
        "lie still and rest":                 {"do_nothing_upright", "jitter_in_place", "collapse_and_thrash", "descend_and_thrash"},
    }
    for goal, expect in cases.items():
        names = {l["name"] for l in general_required_losers(G1, goal)}
        assert names, f"empty loser set for {goal!r} → gate unenforced"
        assert names == expect, f"{goal!r}: {names} != {expect}"


def test_round15_static_hold_intersection_denies_confound_proxy():
    """§round-15 DEFECT 2: a static-hold ∩ terminal-down goal ('lie still') kept a NON-empty
    loser set, so a still-upright confound proxy is DENIED (was a FALSE GRANT)."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate)

    def lie_still_proxy(arrays, behavior, meta):  # rewards stillness OR lowness, never lies down
        import numpy as np
        root = arrays["root_link_pos_w"]; jv = arrays["joint_vel"]
        low = float(np.clip(1 - np.mean(root[..., 2]) / 0.7, 0, 1))
        still = float(1 - np.clip(np.abs(jv).mean(), 0, 1))
        return {"spec_score": max(low, 0.8 * still)}

    rec = adversarial_archetype_gate(
        lie_still_proxy, [], G1, competent_ref=1.0, client=None,
        required_losers=general_required_losers(G1, "lie still and rest"),
        scored_channels=["posture", "completion"], author=False)
    assert rec["gameable"]                              # do_nothing (still+upright) scores 0.8 ≥ ceiling


def test_round15_steady_adverb_does_not_drop_posture_defense():
    """§round-15 DEFECT 3: an active goal with a stillness ADVERB ('a steady wave') must NOT
    be misread as static-hold (which dropped do_nothing → a posture proxy false-granted)."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate, _goal_is_static_hold)
    assert not _goal_is_static_hold("give a steady wave with your arm")   # active verb wins
    assert not _goal_is_static_hold("walk forward with a stable gait")
    assert _goal_is_static_hold("balance on one leg")                    # true balance still detected
    assert _goal_is_static_hold("hold a motionless stance")

    def posture_only(arrays, behavior, meta):
        import numpy as np
        g = arrays["projected_gravity_b"]
        return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0, 1)))}

    rec = adversarial_archetype_gate(
        posture_only, [], G1, competent_ref=1.0, client=None,
        required_losers=general_required_losers(G1, "give a steady wave with your arm"),
        scored_channels=["posture", "completion"], author=False)
    assert rec["gameable"]                              # do_nothing_upright still defends → denied


def test_round15_balance_metric_not_false_rejected():
    """§round-15: a genuine balance metric (rewards uprightness) on a static-hold goal is
    NOT false-rejected — do_nothing/jitter are dropped there (on-goal), only collapse runs."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate)

    def balance_metric(arrays, behavior, meta):
        import numpy as np
        g = arrays["projected_gravity_b"]
        return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0, 1)))}

    rec = adversarial_archetype_gate(
        balance_metric, [], G1, competent_ref=1.0, client=None,
        required_losers=general_required_losers(G1, "balance on one leg"),
        scored_channels=["posture", "completion"], author=False)
    assert not rec["gameable"]                          # balance metric grants-side


# ── §round-16 static-hold classification fixes ───────────────────────────────

def test_round16_locomotion_with_stillness_keeps_posture_losers():
    """§round-16 [HIGH FALSE GRANT]: a locomotion goal carrying an incidental stillness
    adverb ('stay still then dash forward') must NOT be read as static-hold — else the
    posture losers are dropped and an idle-upright proxy false-grants."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate, _goal_is_static_hold)
    for g in ["stay still then suddenly dash forward", "motionless crawl forward",
              "sprint while staying still in the upper body", "shuffle forward quietly"]:
        assert not _goal_is_static_hold(g), g
        names = {l["name"] for l in general_required_losers(G1, g)}
        assert "do_nothing_upright" in names                  # posture defense retained

    def idle_upright(arrays, behavior, meta):
        import numpy as np
        g = arrays["projected_gravity_b"]; root = arrays["root_link_pos_w"]
        up = float(np.mean(np.clip(-g[..., 2], 0, 1)))
        travel = float(min(abs(np.mean(root[..., 0, 0])) / 3.0, 1.0))
        return {"spec_score": 0.55 * up + 0.45 * travel}

    rec = adversarial_archetype_gate(
        idle_upright, [], G1, competent_ref=1.0, client=None,
        required_losers=general_required_losers(G1, "stay still then suddenly dash forward"),
        scored_channels=["posture", "completion"], author=False)
    assert rec["gameable"]                                    # do_nothing_upright catches it


def test_round16_tokenless_balance_phrasings_not_false_rejected():
    """§round-16 [MEDIUM FALSE REJECT]: a balance goal phrased WITHOUT the word 'balance'
    ('stand on one leg', 'remain upright on one foot', 'do not fall over') must classify
    static-hold so a genuine balance metric is not false-denied (the synthesizer cannot
    render single-leg contact, so a one-legged balance is indistinguishable from
    do_nothing_upright)."""
    from sculptor.eval.metric_calibration import (
        general_required_losers, adversarial_archetype_gate, _goal_is_static_hold)

    def balance(arrays, behavior, meta):
        import numpy as np
        g = arrays["projected_gravity_b"]
        return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0, 1)))}

    for g in ["stand on one leg", "remain upright on one foot", "hold a one-legged stance",
              "do not fall over", "keep your equilibrium"]:
        assert _goal_is_static_hold(g), g
        names = {l["name"] for l in general_required_losers(G1, g)}
        assert "do_nothing_upright" not in names             # on-goal → dropped
        rec = adversarial_archetype_gate(
            balance, [], G1, competent_ref=1.0, client=None,
            required_losers=general_required_losers(G1, g),
            scored_channels=["posture", "completion"], author=False)
        assert not rec["gameable"], g                        # genuine balance metric not denied


def test_round16_active_goal_with_balance_word_stays_active():
    """An ACTIVE goal that mentions balance/upright as a modifier keeps the posture losers
    (the active verb wins) — guards against the locomotion false-grant via 'balance'."""
    from sculptor.eval.metric_calibration import _goal_is_static_hold
    assert not _goal_is_static_hold("wave while balancing on one leg")
    assert not _goal_is_static_hold("walk forward staying upright")
    assert not _goal_is_static_hold("touch your toes then stand upright")


# ── §round-17 LADDER-DERIVED posture (root-cause fix for keyword brittleness) ──

def _shoulder_grp(amp):
    return Group(name="arm", mode="oscillate", amplitude_rad=amp,
                 role_query=RoleQuery(segments=["shoulder", "elbow"], axes=["pitch", None]))

def _salute_ladder():  # ascending arm motion → top rung is NOT a still hold
    return CompetenceLadder(competence_axis="arm raise toward salute", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_shoulder_grp(a)])
        for a in (0.1, 0.5, 0.9, 1.3)])

def _balance_ladder():  # ascending uprightness, no motion → top rung IS a still hold
    return CompetenceLadder(competence_axis="upright balance", rungs=[
        MotionSpec(uprightness=u, base_height_m=0.7) for u in (0.2, 0.6, 0.85, 1.0)])


def test_round17_ladder_posture_classifies_top_rung():
    """§round-17 root-cause: the static-hold/terminal-down decision is read from the blind
    AUTHORED ladder's TOP rung (goal-text-independent), not a brittle keyword classifier."""
    from sculptor.eval.metric_calibration import _ladder_posture
    assert _ladder_posture([_salute_ladder().rungs[-1]]) == (False, False)   # arm motion
    assert _ladder_posture([_balance_ladder().rungs[-1]]) == (True, False)   # still upright
    assert _ladder_posture([fold_ladder().rungs[-1]]) == (False, False)      # pelvis fold
    assert _ladder_posture([]) == (None, None)                              # no evidence → fallback


def test_round17_salute_ladder_denies_idle_proxy_end_to_end(tmp_path):
    """§round-17 [HIGH FALSE GRANT] fix end-to-end: an upright-and-moving idle proxy on a
    gesture goal whose AUTHORED ladder has arm motion at the top is DENIED — the ladder
    posture (not the goal phrasing 'salute while staying upright') keeps do_nothing_upright.
    The brittle keyword classifier had mis-read this as static-hold and dropped the defense."""
    IDLE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = np.asarray(arrays["projected_gravity_b"], float)
    jv = np.asarray(arrays["joint_vel"], float)
    return {"spec_score": float((g[..., 2] < -0.85).mean() * np.tanh(np.abs(jv).mean()))}
'''
    p = _write(tmp_path, "idle.py", IDLE)
    cal = calibrate_task_derived(
        p, "salute while staying upright", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_salute_ladder(), _salute_ladder(), _salute_ladder()))
    assert not cal["ok"], cal                                   # idle proxy DENIED
    assert cal["adversarial"]["gameable"]
    assert "do_nothing_upright" in {l["name"] for l in cal["adversarial"]["required_losers"]}


def test_round17_balance_ladder_grants_balance_metric_end_to_end(tmp_path):
    """§round-17: a genuine balance metric whose AUTHORED ladder is a still-upright hold is
    GRANTED — the ladder posture drops do_nothing_upright (on-goal) so it is not denied."""
    BAL = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = np.asarray(arrays["projected_gravity_b"], float)
    return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0.0, 1.0)))}
'''
    p = _write(tmp_path, "bal.py", BAL)
    cal = calibrate_task_derived(
        p, "balance steadily on both feet", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_balance_ladder(), _balance_ladder(), _balance_ladder()))
    adv = cal["adversarial"] or {}
    assert not adv.get("gameable")                             # balance metric NOT false-denied
    assert "do_nothing_upright" not in {l["name"] for l in adv.get("required_losers", [])}


# ── §round-18 ladder-posture: a SUBTLE active gesture is not a still hold ──────

def test_round18_subtle_gesture_ladder_keeps_velocity_defense(tmp_path):
    """§round-18 [HIGH FALSE GRANT] fix: a SUBTLE active-gesture ladder (group amplitude
    ≤ 0.1 rad) must NOT be read as a still hold — _spec_is_static_hold keys on the presence
    of commanded joint motion, not an amplitude threshold. So the velocity defense
    (jitter_in_place) is kept and a 'rewards any joint velocity' idle proxy is DENIED."""
    from sculptor.eval.metric_calibration import _spec_is_static_hold

    def _g(amp, mode="oscillate", off=0.0):
        return Group(name="arm", mode=mode, amplitude_rad=amp, offset_rad=off, period_frames=40,
                     role_query=RoleQuery(segments=["shoulder", "elbow"], axes=["pitch", None], sides=["left"]))

    # a subtle oscillation IS active; a held offset IS a distinctive posture; both NON-static.
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_g(0.099)]))
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_g(0.0, "hold", 0.8)]))
    # the whole-body tremor AND noise channels are motion too (round-18: previously ignored).
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7, tremor=1.5))
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7, noise=0.15))
    # a true balance/hold top (no motion group, no tremor/noise) stays static.
    assert _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7))

    def subtle_wave_ladder():
        return CompetenceLadder(competence_axis="subtle left-arm wave", rungs=[
            MotionSpec(uprightness=1.0, base_height_m=0.7, groups=[_g(a)])
            for a in (0.0, 0.034, 0.067, 0.099)])

    VEL = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = np.asarray(arrays["projected_gravity_b"], float); jv = np.asarray(arrays["joint_vel"], float)
    up = float((g[..., 2] < -0.85).mean())
    return {"spec_score": float(np.clip(up * (1 - np.exp(-np.abs(jv).mean() / 0.1)), 0.0, 1.0))}
'''
    p = _write(tmp_path, "vel.py", VEL)
    cal = calibrate_task_derived(
        p, "give a small subtle wave with the left arm", robot_hint="Unitree-G1",
        client=_FakeLadderClient(subtle_wave_ladder(), subtle_wave_ladder(), subtle_wave_ladder()))
    assert not cal["ok"], cal                                   # velocity proxy DENIED
    assert "jitter_in_place" in {l["name"] for l in cal["adversarial"]["required_losers"]}


# ── §round-19 base_height_m: the last unread MotionSpec motion channel ─────────

def test_round19_static_hold_reads_base_height_channel():
    """§round-19 [completeness]: base_height_m was the one MotionSpec motion field
    _spec_is_static_hold did not read, so a vertical RAMP (rise/descend = commanded motion)
    or a held NON-nominal height (squat) was mis-classified as a STILL UPRIGHT hold and
    DROPPED the do_nothing/jitter losers. A nominal-standing hold (≥0.55, no ramp) stays
    static so a genuine balance metric is still not false-denied."""
    from sculptor.eval.metric_calibration import _spec_is_static_hold
    # nominal standing hold (what a genuine balance ladder authors) → STILL static
    assert _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.7))
    assert _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.6))
    assert _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=[0.7, 0.68]))
    # a vertical RAMP is motion; a held squat is a non-standing posture → NOT static
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=[0.3, 0.7]))
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=[0.7, 0.45]))
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.45))
    assert not _spec_is_static_hold(MotionSpec(uprightness=1.0, base_height_m=0.3))


def test_round19_standup_ladder_keeps_posture_losers_end_to_end(tmp_path):
    """§round-19 end-to-end: a 'stand up from a crouch and hold' goal whose AUTHORED ladder
    has a RISING base_height ramp at the top must KEEP do_nothing/jitter (the ramp is motion,
    not a still hold). A genuine RISE metric scores do_nothing ~0 so it still GRANTS; the
    posture defense is no longer silently dropped by the unread height channel."""
    def standup_ladder():
        return CompetenceLadder(competence_axis="rise from crouch to standing", rungs=[
            MotionSpec(uprightness=1.0, base_height_m=0.30),
            MotionSpec(uprightness=1.0, base_height_m=[0.30, 0.45]),
            MotionSpec(uprightness=1.0, base_height_m=[0.30, 0.60]),
            MotionSpec(uprightness=1.0, base_height_m=[0.30, 0.70]),
        ])
    HONEST_RISE = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None: return {"spec_score": 0.0}
    z = np.asarray(root)[..., 2].mean(axis=1)
    rise = float(z[-1] - z[0]); final = float(z[-1])
    return {"spec_score": float(np.clip(min(rise / 0.35, final / 0.7), 0.0, 1.0))}
'''
    p = _write(tmp_path, "rise.py", HONEST_RISE)
    cal = calibrate_task_derived(
        p, "stand up from a crouch and hold standing", robot_hint="Unitree-G1",
        client=_FakeLadderClient(standup_ladder(), standup_ladder(), standup_ladder()))
    losers = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    assert "do_nothing_upright" in losers and "jitter_in_place" in losers   # ramp ≠ still hold
    assert cal["ok"], cal                                                   # honest rise still GRANTS


# ── §round-19 convergence red-team fixes (terminal_down / kick-firewall / lie-down) ──

def _squat_ladder():
    return CompetenceLadder(competence_axis="squat depth (upright)", rungs=[
        MotionSpec(uprightness=1.0, base_height_m=h) for h in (0.62, 0.50, 0.40, 0.30)])


def test_round19_upright_squat_keeps_collapse_loser_denies_depth_proxy(tmp_path):
    """§round-19 [HIGH FALSE GRANT] fix A: a held UPRIGHT deep squat (base_height<=0.35 but
    upright) must NOT be classified terminal-down — _spec_is_terminal_down's base_height branch
    is now gated on non-uprightness. So collapse_and_stay_down is KEPT and a dip-DEPTH-only
    proxy (gamed by collapsing) is DENIED, while an honest upright-gated squat metric GRANTS."""
    from sculptor.eval.metric_calibration import _spec_is_terminal_down
    assert not _spec_is_terminal_down(MotionSpec(uprightness=1.0, base_height_m=0.30))  # upright squat
    assert _spec_is_terminal_down(MotionSpec(uprightness=0.0, base_height_m=0.12))      # lie/fallen
    DEPTH = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None: return {"spec_score": 0.0}
    z = float(np.asarray(root)[..., 2].mean())
    return {"spec_score": float(np.clip((0.7 - z) / 0.6, 0.0, 1.0))}
'''
    HONEST = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None: return {"spec_score": 0.0}
    z = float(np.asarray(root)[..., 2].mean()); up = float((grav[..., 2] < -0.85).mean())
    return {"spec_score": float(np.clip((0.7 - z) / 0.6, 0.0, 1.0) * up)}
'''
    cal_bad = calibrate_task_derived(_write(tmp_path, "depth.py", DEPTH),
        "hold a deep squat", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_squat_ladder(), _squat_ladder(), _squat_ladder()))
    losers = {l["name"] for l in (cal_bad["adversarial"] or {}).get("required_losers", [])}
    assert "collapse_and_stay_down" in losers                 # kept (squat ≠ terminal-down)
    assert not cal_bad["ok"], cal_bad                         # depth confound DENIED
    cal_ok = calibrate_task_derived(_write(tmp_path, "honest.py", HONEST),
        "hold a deep squat", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_squat_ladder(), _squat_ladder(), _squat_ladder()))
    assert cal_ok["ok"], cal_ok                               # honest upright squat GRANTS


def test_round19_novel_kick_runs_firewall_default_path(tmp_path):
    """§round-19 [HIGH FALSE GRANT] fix C: a NOVEL kick goal lands on the task-derived path;
    its `if fam=="kick"` branch used to set req_losers=None unless an unset flag was on, so the
    gate scored ZERO losers and the firewall was OFF. It now runs the general goal-blind losers
    ALWAYS — an honest kick still GRANTS, but the firewall is no longer absent."""
    cal = calibrate_task_derived(_write(tmp_path, "kick.py", GOOD_KICK),
        "kick forward with the left leg from a stance", robot_hint="Unitree-G1",
        client=_FakeLadderClient(kick_ladder(), kick_ladder(), kick_ladder()))
    adv = cal["adversarial"] or {}
    names = {l["name"] for l in adv.get("required_losers", [])}
    assert names == {"do_nothing_upright", "jitter_in_place", "collapse_and_stay_down"}  # firewall RAN
    assert adv.get("ran") is True
    assert cal["ok"], cal                                     # honest kick still GRANTS


def _lie_ladder():
    return CompetenceLadder(competence_axis="lie down and rest still", rungs=[
        MotionSpec(uprightness=0.6, base_height_m=0.5),
        MotionSpec(uprightness=0.3, base_height_m=0.35),
        MotionSpec(uprightness=0.1, base_height_m=0.22),
        MotionSpec(uprightness=0.0, base_height_m=0.12)])


def test_round19_lie_down_stillness_loser_denies_low_only_proxy(tmp_path):
    """§round-19 [FALSE GRANT] fix B: for a terminal-down 'rest still' goal collapse is dropped
    (on-goal) and do_nothing/jitter are rendered UPRIGHT, so the stillness channel was unprobed
    at the low posture. A new collapse_and_thrash loser (low pelvis + violent motion) catches a
    low-height-ONLY proxy; an honest lie-STILL metric (low·stillness) scores it ~0 and GRANTS."""
    LOW = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    if root is None: return {"spec_score": 0.0}
    z = float(np.asarray(root)[..., 2].mean())
    return {"spec_score": float(np.clip(1.0 - z / 0.7, 0.0, 1.0))}
'''
    HONEST = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); jv = arrays.get("joint_vel")
    if root is None or jv is None: return {"spec_score": 0.0}
    z = float(np.asarray(root)[..., 2].mean()); low = float(np.clip(1.0 - z / 0.7, 0.0, 1.0))
    still = float(np.exp(-np.abs(np.asarray(jv)).mean() / 0.3))
    return {"spec_score": float(np.clip(low * still, 0.0, 1.0))}
'''
    cal_bad = calibrate_task_derived(_write(tmp_path, "low.py", LOW),
        "lie down on the floor and rest still", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_lie_ladder(), _lie_ladder(), _lie_ladder()))
    losers = {l["name"] for l in (cal_bad["adversarial"] or {}).get("required_losers", [])}
    assert "collapse_and_thrash" in losers                    # stillness probe present
    assert not cal_bad["ok"], cal_bad                         # stillness-blind proxy DENIED
    cal_ok = calibrate_task_derived(_write(tmp_path, "lie.py", HONEST),
        "lie down on the floor and rest still", robot_hint="Unitree-G1",
        client=_FakeLadderClient(_lie_ladder(), _lie_ladder(), _lie_ladder()))
    assert cal_ok["ok"], cal_ok                               # honest lie-still GRANTS


# ── §round-20: HOLD vs TRANSITION — the posture classifiers read the WHOLE ladder ──
# All three round-20 defects shared one root cause: classifying from the TOP rung alone
# conflates "HOLD this posture" with "TRANSITION to / move AT this posture".

def test_round20_motion_and_span_aware_posture_classifiers():
    """§round-20 unit: terminal_down now requires the down top to be STILL (a writhe/duck is
    active-low, not lie/rest); _ladder_has_crouched_rung detects a crouch→stand TRANSITION
    ladder (held-standing top but low start rungs)."""
    from sculptor.eval.metric_calibration import (
        _spec_has_commanded_motion, _spec_is_terminal_down, _ladder_has_crouched_rung)

    def osc(amp):
        return Group(name="b", mode="oscillate", amplitude_rad=amp,
                     role_query=RoleQuery(segments=["hip", "knee"], axes=["pitch", None]))
    still_lie = MotionSpec(uprightness=0.0, base_height_m=0.12)
    writhe = MotionSpec(uprightness=0.0, base_height_m=0.13, groups=[osc(1.4)])
    assert not _spec_has_commanded_motion(still_lie)
    assert _spec_has_commanded_motion(writhe)
    assert _spec_is_terminal_down(still_lie)        # still down → lie/rest
    assert not _spec_is_terminal_down(writhe)        # low + motion → active, NOT lie/rest
    # a crouch→stand transition (held heights 0.40..0.70) vs a balance hold (all nominal)
    assert _ladder_has_crouched_rung([MotionSpec(uprightness=1.0, base_height_m=h) for h in (0.40, 0.52, 0.62, 0.70)])
    assert not _ladder_has_crouched_rung([MotionSpec(uprightness=u, base_height_m=0.7) for u in (0.2, 0.6, 0.85, 1.0)])


def test_round20_sit_to_stand_keeps_do_nothing_denies_height_confound(tmp_path):
    """§round-20 [HIGH FALSE GRANT] fix (defect 3): a crouch/sit→stand ladder has a held-standing
    TOP rung (static_hold=True per-rung), but do_nothing (already standing, never rose) is OFF-goal.
    The crouch-span check suppresses static_hold → do_nothing/jitter are KEPT → a height-ONLY
    confound (full credit to an already-standing robot) is DENIED."""
    def held_height_ladder():
        return CompetenceLadder(competence_axis="final standing height", rungs=[
            MotionSpec(uprightness=1.0, base_height_m=h) for h in (0.40, 0.52, 0.62, 0.70)])
    HEIGHT_ONLY = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b")
    if root is None or grav is None: return {"spec_score": 0.0}
    up = float((grav[..., 2] < -0.85).mean()); h = float(np.asarray(root)[-1, :, 2].mean())
    return {"spec_score": float(np.clip(up * np.clip(h / 0.7, 0.0, 1.0), 0.0, 1.0))}
'''
    cal = calibrate_task_derived(_write(tmp_path, "height.py", HEIGHT_ONLY),
        "stand up from a seated position", robot_hint="Unitree-G1",
        client=_FakeLadderClient(held_height_ladder(), held_height_ladder(), held_height_ladder()))
    names = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    assert "do_nothing_upright" in names                 # span check kept it (transition, not hold)
    assert not cal["ok"], cal                            # height confound DENIED


def test_round20_writhe_keeps_collapse_grants_honest(tmp_path):
    """§round-20 [HIGH FALSE REJECT] fix (defect 2): a 'writhe/thrash on the ground' goal is
    terminal-LOW but WITH on-goal motion, so it is NOT lie/rest — collapse_and_thrash must NOT be
    injected (it IS the on-goal end-state → would false-reject). collapse_and_stay_down (still) is
    kept; an honest writhe metric scores it ~0 and GRANTS."""
    def writhe_grp(amp):
        return Group(name="body", mode="oscillate", amplitude_rad=amp, period_frames=18,
                     role_query=RoleQuery(segments=["hip", "knee", "shoulder", "elbow"], axes=["pitch", None]))
    def writhe_ladder():
        return CompetenceLadder(competence_axis="thrash amplitude on the ground", rungs=[
            MotionSpec(uprightness=0.0, base_height_m=0.13, groups=[writhe_grp(a)]) for a in (0.2, 0.6, 1.0, 1.4)])
    HONEST = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w"); grav = arrays.get("projected_gravity_b"); jv = arrays.get("joint_vel")
    if root is None or grav is None or jv is None: return {"spec_score": 0.0}
    z = float(np.asarray(root)[..., 2].mean()); low = float(np.clip(1.0 - z / 0.7, 0.0, 1.0))
    down = float((grav[..., 2] > -0.3).mean()); motion = float(1.0 - np.exp(-np.abs(np.asarray(jv)).mean() / 1.0))
    return {"spec_score": float(np.clip(low * down * motion, 0.0, 1.0))}
'''
    cal = calibrate_task_derived(_write(tmp_path, "writhe.py", HONEST),
        "writhe and thrash on the ground", robot_hint="Unitree-G1",
        client=_FakeLadderClient(writhe_ladder(), writhe_ladder(), writhe_ladder()))
    names = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    assert "collapse_and_stay_down" in names and "collapse_and_thrash" not in names  # active-low, not lie/rest
    assert cal["ok"], cal                                # honest writhe NOT false-rejected


def test_round20_active_duck_keeps_collapse_denies_descent_confound(tmp_path):
    """§round-20 [HIGH FALSE GRANT] fix (defect 1): an active 'duck down and hold low' goal has a
    low-uprightness top rung but HOLDS a bent-leg posture (commanded), so it is NOT lie/rest —
    collapse_and_stay_down (a descent ramp) is KEPT and catches a descent-magnitude confound that
    reads only root-z drop."""
    def duck_grp(off):
        return Group(name="legs", mode="hold", offset_rad=off,
                     role_query=RoleQuery(segments=["hip", "knee"], axes=["pitch", None], sides=["left", "right"]))
    def duck_ladder():
        return CompetenceLadder(competence_axis="crouch depth", rungs=[
            MotionSpec(uprightness=[1.0, U], base_height_m=[0.7, H], groups=[duck_grp(A)])
            for (U, H, A) in [(0.7, 0.55, 0.3), (0.5, 0.45, 0.6), (0.35, 0.37, 0.9), (0.25, 0.28, 1.2)]])
    DROP_ONLY = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    z = np.asarray(arrays["root_link_pos_w"])[..., 2]
    drop = float(np.mean(z[:5].mean(0) - z.min(0)))
    return {"spec_score": float(np.clip(drop / 0.45, 0.0, 1.0))}
'''
    cal = calibrate_task_derived(_write(tmp_path, "drop.py", DROP_ONLY),
        "duck down into a low crouch and hold it low", robot_hint="Unitree-G1",
        client=_FakeLadderClient(duck_ladder(), duck_ladder(), duck_ladder()))
    names = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    # §round-21: a duck holds a bent-leg posture (a zero-velocity hold offset → not dynamic
    # motion), so it classifies terminal-down; the descend_and_thrash ramp probes the descent
    # channel and catches a descent-magnitude confound.
    assert "descend_and_thrash" in names
    assert not cal["ok"], cal                            # descent confound DENIED


# ── §round-21: per-loser idle floor, AND-gated static_hold, two-probe terminal-down ──

def test_round21_floor_confound_denied_pure_idle_floor(tmp_path):
    """§round-21 [HIGH FALSE GRANT] fix: an additive uprightness-gated FLOOR confound
    (up·(FLOOR + (1-FLOOR)·goal)) pays do_nothing_upright the FLOOR (up to 49%), under the 0.5
    ceiling, so it used to GRANT. do_nothing_upright now carries a strict per-loser floor — paying
    the pure-idle anchor any non-trivial credit is gameable. An honest multiplicative fold metric
    scores do_nothing 0 and GRANTS."""
    def fold_d():
        return CompetenceLadder(competence_axis="toe-touch depth", rungs=[
            MotionSpec(uprightness=1.0, base_height_m=0.7, fold_depth_m=d) for d in (0.0, 0.15, 0.3, 0.45, 0.6)])
    FLOOR = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = np.asarray(arrays["root_link_pos_w"]); g = np.asarray(arrays["projected_gravity_b"])
    up = float((g[..., 2] < -0.85).mean()); z = root[..., 2].mean(1)
    dip = float(np.clip((z[:5].mean() - z.min()) / 0.6, 0, 1))
    return {"spec_score": float(np.clip(up * (0.49 + 0.51 * dip), 0, 1))}
'''
    HONEST = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    root = np.asarray(arrays["root_link_pos_w"]); g = np.asarray(arrays["projected_gravity_b"]); z = root[..., 2].mean(1)
    dip = float(np.clip((z[:5].mean() - z.min()) / 0.6, 0, 1)); ret = float(np.clip(1 - abs(z[-1] - z[0]) / 0.2, 0, 1))
    up = float((g[..., 2] < -0.85).mean())
    return {"spec_score": float(np.clip(dip * ret * up, 0, 1))}
'''
    cal_bad = calibrate_task_derived(_write(tmp_path, "floor.py", FLOOR),
        "bend down and touch your toes", robot_hint="Unitree-G1",
        client=_FakeLadderClient(fold_d(), fold_d(), fold_d()))
    assert not cal_bad["ok"], cal_bad
    assert cal_bad["adversarial"]["worst_name"] == "do_nothing_upright"
    cal_ok = calibrate_task_derived(_write(tmp_path, "fold.py", HONEST),
        "bend down and touch your toes", robot_hint="Unitree-G1",
        client=_FakeLadderClient(fold_d(), fold_d(), fold_d()))
    assert cal_ok["ok"], cal_ok


def test_round21_active_gesture_posture_confound_denied(tmp_path):
    """§round-21 [HIGH FALSE GRANT] fix: a postural-stability ladder for an ACTIVE gesture goal
    made the per-rung static_hold drop do_nothing. static_hold is now AND-gated with the goal-text
    classifier ('wave your arm' is NOT a static hold), so do_nothing is KEPT and a posture/height
    confound is DENIED."""
    def postural():
        return CompetenceLadder(competence_axis="postural stability", rungs=[
            MotionSpec(uprightness=u, base_height_m=0.7) for u in (0.3, 0.6, 0.85, 1.0)])
    CONF = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = np.asarray(arrays["projected_gravity_b"]); root = np.asarray(arrays["root_link_pos_w"])
    up = float((g[..., 2] < -0.85).mean()); tall = float(np.clip((root[..., 2].mean() - 0.55) / 0.15, 0, 1))
    return {"spec_score": float(np.clip(up * tall, 0, 1))}
'''
    cal = calibrate_task_derived(_write(tmp_path, "conf.py", CONF),
        "wave your right arm", robot_hint="Unitree-G1",
        client=_FakeLadderClient(postural(), postural(), postural()))
    names = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    assert "do_nothing_upright" in names                 # AND-gate kept it (active goal)
    assert not cal["ok"], cal


def test_round21_balance_low_fall_rung_not_false_rejected(tmp_path):
    """§round-21 [HIGH FALSE REJECT] fix: a blind balance author renders the FALL failure rung as a
    pelvis DROP (low base_height AND low uprightness). _ladder_has_crouched_rung now requires the
    low rung to ALSO be UPRIGHT to count as a crouch target, so the balance metric is not denied."""
    def bal_fall(fbh):
        return CompetenceLadder(competence_axis="upright balance", rungs=[
            MotionSpec(uprightness=0.4, base_height_m=fbh), MotionSpec(uprightness=0.7, base_height_m=0.7),
            MotionSpec(uprightness=0.9, base_height_m=0.7), MotionSpec(uprightness=1.0, base_height_m=0.7)])
    BAL = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    g = np.asarray(arrays["projected_gravity_b"]); jv = np.asarray(arrays["joint_vel"])
    return {"spec_score": float(np.mean(np.clip(-g[..., 2], 0, 1)) * np.exp(-np.abs(jv).mean() / 0.5))}
'''
    for fbh in (0.45, 0.30):
        cal = calibrate_task_derived(_write(tmp_path, f"bal{int(fbh*100)}.py", BAL),
            "balance on one leg without falling over", robot_hint="Unitree-G1",
            client=_FakeLadderClient(bal_fall(fbh), bal_fall(fbh), bal_fall(fbh)))
        assert cal["ok"], (fbh, cal)


def test_round21_settled_limb_lie_down_not_false_rejected(tmp_path):
    """§round-21 [FALSE REJECT] fix: a lie-down ladder whose top rung holds a settled limb (a
    zero-velocity hold offset) must still classify terminal-down — a held static offset is a
    POSTURE, not motion (dynamic_only=True). An honest descend-and-rest metric GRANTS."""
    def settled():
        return CompetenceLadder(competence_axis="descend and rest", rungs=[
            MotionSpec(uprightness=1.0, base_height_m=0.7), MotionSpec(uprightness=0.6, base_height_m=0.45),
            MotionSpec(uprightness=0.2, base_height_m=0.25),
            MotionSpec(uprightness=0.0, base_height_m=0.12,
                       groups=[Group(name="arm", mode="hold", offset_rad=0.02,
                                     role_query=RoleQuery(segments=["shoulder"], axes=["pitch", None]))])])
    HONEST = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    z = float(np.asarray(arrays["root_link_pos_w"])[..., 2].mean()); jv = np.asarray(arrays["joint_vel"])
    return {"spec_score": float(np.clip((1 - z / 0.7) * np.exp(-np.abs(jv).mean() / 0.3), 0, 1))}
'''
    cal = calibrate_task_derived(_write(tmp_path, "rest.py", HONEST),
        "lie down on the floor and rest", robot_hint="Unitree-G1",
        client=_FakeLadderClient(settled(), settled(), settled()))
    names = {l["name"] for l in (cal["adversarial"] or {}).get("required_losers", [])}
    assert "collapse_and_thrash" in names                # terminal-down (hold offset ignored)
    assert cal["ok"], cal


def test_round21_no_competence_anchor_is_inconclusive():
    """§round-21 [LOW] fix: competent_ref ≤ 0 has no usable anchor, so the gate is INCONCLUSIVE
    (not auto-flagged gameable with worst_name=None as the 0-ceiling bug did)."""
    from sculptor.eval.metric_calibration import adversarial_archetype_gate, general_required_losers
    losers = general_required_losers(G1, "wave your right arm")
    rec = adversarial_archetype_gate(lambda a, b, m: {"spec_score": 0.0}, [], G1,
                                     competent_ref=0.0, client=None,
                                     required_losers=losers, author=False)
    assert not rec["gameable"]                            # NOT a deny
    assert "inconclusive" in rec["reason"]
