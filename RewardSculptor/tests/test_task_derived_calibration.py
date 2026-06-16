"""§Ship 51: L2 task-derived calibration — a novel-task metric earns
steer-rights by ranking K independently-authored competence ladders. All
LLM calls are mocked; the synthesizer + gate are deterministic and offline.
"""
from __future__ import annotations

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
    calibrate_metric,
    calibrate_task_derived,
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


class _LadderParsed:
    def __init__(self, ladder): self.parsed_output = ladder


class _LadderMessages:
    def __init__(self, ladders):
        self._ladders = list(ladders); self.calls = 0; self.payloads = []

    def parse(self, **kw):
        # Record the user payload so a test can assert no metric leaked into it.
        self.payloads.append(kw.get("messages", [{}])[-1].get("content", ""))
        lad = self._ladders[self.calls % len(self._ladders)]
        self.calls += 1
        return _LadderParsed(lad)


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
        def parse(self, **kw): raise RuntimeError("api down")
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
    adversarial author (GamingArchetypeSet), branching on `output_format`."""

    def __init__(self, ladders, gaming, gaming_raises=False):
        self._ladders = list(ladders)
        self._gaming = gaming
        self._gaming_raises = gaming_raises
        self.ladder_calls = 0
        self.gaming_calls = 0
        self.ladder_payloads: list = []
        self.gaming_payloads: list = []

    def parse(self, **kw):
        content = kw.get("messages", [{}])[-1].get("content", "")
        if getattr(kw.get("output_format"), "__name__", "") == "GamingArchetypeSet":
            self.gaming_calls += 1
            self.gaming_payloads.append(content)
            if self._gaming_raises:
                raise RuntimeError("gaming author api down")
            return _LadderParsed(self._gaming)
        self.ladder_calls += 1
        self.ladder_payloads.append(content)
        lad = self._ladders[self.ladder_calls - 1] if self.ladder_calls - 1 < len(self._ladders) \
            else self._ladders[(self.ladder_calls - 1) % len(self._ladders)]
        return _LadderParsed(lad)


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
