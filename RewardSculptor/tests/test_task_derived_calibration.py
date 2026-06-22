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


def test_adversarial_flag_off_is_a_noop(tmp_path):
    """Default off: the grant is byte-identical to Ship 51 and NO adversary call
    is made (the minimal novel-task path stays cheap)."""
    p = _write(tmp_path, "gameable.py", GAMEABLE_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(p, "kick forward from a stance",
                                 robot_hint="Unitree-G1", client=client, adversarial=False)
    assert cal["ok"]                              # granted (would be denied if enforced)
    assert cal["adversarial"] is None
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
    """An adversary-call failure is NO evidence, not a denial — the L2 grant
    stands and the inconclusive reason is recorded (never-silent)."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel(), gaming_raises=True)
    cal = calibrate_task_derived(p, "kick forward from a stance", robot_hint="Unitree-G1",
                                 client=client, adversarial=True)
    assert cal["ok"]                                # grant survives the crash
    adv = cal["adversarial"]
    assert adv and not adv["ran"] and not adv["gameable"]
    assert "inconclusive" in adv["reason"]


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


def test_adversarial_required_losers_opt_in_off_is_byte_identical(tmp_path):
    """adversarial_required_losers default OFF → no losers injected (Ship-53 byte-
    identical): GOOD_KICK is GRANTED (it would be denied with losers on)."""
    p = _write(tmp_path, "good.py", GOOD_KICK)
    client = _FakeBothClient(_three_kicks(), gaming_travel())
    cal = calibrate_task_derived(
        p, "kick forward from a stance", robot_hint="Unitree-G1", client=client,
        adversarial=True)
    assert cal["ok"] and not cal["adversarial"].get("required_losers")


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
