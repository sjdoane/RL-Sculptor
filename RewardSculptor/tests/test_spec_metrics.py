"""§Ship 26 (E1): spec-metric tests on synthetic signals with known
ground truth — a metric that can't separate constructed good/bad
behavior has no business grading real ones."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval import BENCHMARKS, compute_spec_metrics, get_benchmark
from sculptor.eval.spec_metrics import (
    burstiness,
    horizontal_speed,
    kick_events_score,
    periodicity,
    spec_cartpole_balance,
    spec_g1_floss,
    spec_g1_jump,
    spec_g1_kick,
    spec_go1_trot,
    uprightness,
)

T, E, J = 200, 4, 12
RNG = np.random.default_rng(7)


def _upright_g(t: int = T, e: int = E) -> np.ndarray:
    g = np.zeros((t, e, 3), dtype=np.float32)
    g[..., 2] = -1.0
    return g


def _fallen_g(t: int = T, e: int = E) -> np.ndarray:
    g = np.zeros((t, e, 3), dtype=np.float32)
    g[..., 0] = 1.0  # gravity sideways in body frame = lying down
    return g


def _periodic_joints(period: int = 20, amp: float = 0.4) -> np.ndarray:
    jp = np.zeros((T, E, J), dtype=np.float32)
    t = np.arange(T)
    for j in range(3):  # three driven joints, phase-offset
        jp[:, :, j] = amp * np.sin(
            2 * np.pi * t / period + j * np.pi / 3
        )[:, None]
    jp += RNG.normal(0, 0.005, jp.shape)  # sensor-ish noise
    return jp


def _static_joints() -> np.ndarray:
    return RNG.normal(0, 0.003, (T, E, J)).astype(np.float32)


def _noise_joints() -> np.ndarray:
    return RNG.normal(0, 0.3, (T, E, J)).astype(np.float32)


# ── uprightness ──────────────────────────────────────────────────────


def test_uprightness_separates_upright_from_fallen() -> None:
    assert uprightness(_upright_g()) == 1.0
    assert uprightness(_fallen_g()) == 0.0


def test_uprightness_normalizes_scaled_gravity() -> None:
    """Audit L1: 9.81-scaled gravity vectors must not silently disable
    the gate."""
    assert uprightness(_upright_g() * 9.81) == 1.0
    assert uprightness(_fallen_g() * 9.81) == 0.0


# ── periodicity ──────────────────────────────────────────────────────


def test_periodicity_high_for_oscillation() -> None:
    out = periodicity(_periodic_joints())
    assert out["periodicity"] > 0.5
    assert 15 <= out["dominant_period_frames"] <= 25  # truth: 20


def test_periodicity_survives_per_env_random_phase() -> None:
    """Audit C1 regression: envs oscillate OUT OF PHASE in reality
    (resets re-randomize). Env-mean-before-FFT cancelled the signal
    (perfect flossing scored 0.22); incoherent power averaging must
    recover it."""
    jp = np.zeros((T, 64, J), dtype=np.float32)
    t = np.arange(T)
    phases = RNG.uniform(0, 2 * np.pi, 64)
    for j in range(3):
        jp[:, :, j] = 0.4 * np.sin(
            2 * np.pi * t[:, None] / 20 + phases[None, :] + j * np.pi / 3
        )
    jp += RNG.normal(0, 0.005, jp.shape)
    out = periodicity(jp)
    assert out["periodicity"] > 0.8, out
    assert 15 <= out["dominant_period_frames"] <= 25


def test_periodicity_zero_for_static() -> None:
    out = periodicity(_static_joints())
    assert out["periodicity"] == 0.0  # amplitude gate


def test_periodicity_low_for_white_noise() -> None:
    out = periodicity(_noise_joints())
    assert out["periodicity"] < 0.2


# ── burstiness ───────────────────────────────────────────────────────


def _bursty_vel() -> np.ndarray:
    jv = RNG.normal(0, 0.05, (T, E, J)).astype(np.float32)
    for t0 in range(25, T, 50):  # a kick every 50 frames
        jv[t0:t0 + 3, :, 4] = 8.0
    return jv


def _humming_vel() -> np.ndarray:
    t = np.arange(T)
    jv = np.zeros((T, E, J), dtype=np.float32)
    jv[:, :, 2] = (2.0 * np.sin(2 * np.pi * t / 20))[:, None]
    return jv


def test_burstiness_separates_kicks_from_hum() -> None:
    kick = burstiness(_bursty_vel())
    hum = burstiness(_humming_vel())
    assert kick["burst_ratio_p95"] > 3.0
    assert hum["burst_ratio_p95"] < 2.0


def test_burstiness_cancels_control_rate_tremor() -> None:
    """The calibrated-on-real-data behavior: standing policies tremble
    (±sign alternation at the frame rate, |v| ≈ 6) and must NOT read
    as bursts after signed smoothing."""
    jv = np.zeros((T, E, J), dtype=np.float32)
    jv[:, :, 3] = 6.0 * ((-1.0) ** np.arange(T))[:, None]  # pure tremor
    out = burstiness(jv)
    assert out["burst_p95"] < 1.5, out  # 6.0 raw → ~1.2 after 5-frame mean


def test_burstiness_upright_mask_gates_falling_bursts() -> None:
    """Audit H2: bursts during non-upright windows must not count."""
    jv = _bursty_vel()
    mask = np.ones((T, E), dtype=bool)
    for t0 in range(25, T, 50):           # fallen exactly around bursts
        mask[t0 - 5:t0 + 8, :] = False
    gated = burstiness(jv, valid_mask=mask)
    ungated = burstiness(jv)
    assert ungated["burst_p95"] > 3.0
    assert gated["burst_p95"] < 1.0, gated


def test_burstiness_joint_subset() -> None:
    """Audit H1: bursts outside the selected joints are invisible."""
    jv = _bursty_vel()                     # bursts on joint 4
    out_legs = burstiness(jv, joint_indices=[0, 1, 2])
    out_with = burstiness(jv, joint_indices=[4])
    assert out_with["burst_p95"] > 3.0
    assert out_legs["burst_p95"] < 1.0


# ── horizontal speed ─────────────────────────────────────────────────


def test_speed_straight_line() -> None:
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[:, :, 0] = (np.arange(T) * 0.03)[:, None]
    out = horizontal_speed(p)
    assert out["speed_per_frame"] == pytest.approx(0.03, rel=1e-3)
    assert out["straightness"] == pytest.approx(1.0, rel=1e-3)


def test_speed_random_walk_is_not_straight() -> None:
    steps = RNG.normal(0, 0.05, (T - 1, E, 2))
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[1:, :, :2] = np.cumsum(steps, axis=0)
    out = horizontal_speed(p)
    assert out["straightness"] < 0.4


def test_speed_ignores_reset_teleports() -> None:
    """Audit M2: a mid-capture auto-reset warps the env back to spawn;
    that step must not count as path, and per-segment nets must sum."""
    p = np.zeros((T, E, 3), dtype=np.float32)
    v = 0.03
    # Two straight segments with a teleport home in the middle.
    half = T // 2
    p[:half, :, 0] = (np.arange(half) * v)[:, None]
    p[half:, :, 0] = (np.arange(T - half) * v)[:, None]  # restarts at 0
    out = horizontal_speed(p)
    assert out["teleport_steps"] > 0
    assert out["straightness"] > 0.95, out  # both segments dead straight
    # net ≈ sum of both segment nets ≈ v * (T - 2)
    assert out["net_displacement"] == pytest.approx(v * (T - 2), rel=0.05)


# ── composite specs ──────────────────────────────────────────────────


def test_floss_spec_rewards_upright_oscillation_only() -> None:
    good = spec_g1_floss(
        {"joint_pos": _periodic_joints(), "projected_gravity_b": _upright_g()},
        {},
    )
    fallen = spec_g1_floss(
        {"joint_pos": _periodic_joints(), "projected_gravity_b": _fallen_g()},
        {},
    )
    static = spec_g1_floss(
        {"joint_pos": _static_joints(), "projected_gravity_b": _upright_g()},
        {},
    )
    assert good["spec_score"] > 0.5
    assert fallen["spec_score"] == 0.0
    assert static["spec_score"] == 0.0


def test_kick_spec_rewards_upright_bursts_only() -> None:
    good = spec_g1_kick(
        {"joint_vel": _bursty_vel(), "projected_gravity_b": _upright_g()}, {},
    )
    hum = spec_g1_kick(
        {"joint_vel": _humming_vel(), "projected_gravity_b": _upright_g()}, {},
    )
    fallen = spec_g1_kick(
        {"joint_vel": _bursty_vel(), "projected_gravity_b": _fallen_g()}, {},
    )
    assert good["spec_score"] > 0.5
    assert good["spec_score"] > hum["spec_score"] * 2
    assert fallen["spec_score"] == 0.0


def test_trot_spec_rewards_straight_upright_travel() -> None:
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[:, :, 0] = (np.arange(T) * 0.03)[:, None]
    p[:, :, 2] = 0.30                       # go1 stance height
    good = spec_go1_trot(
        {"root_link_pos_w": p, "projected_gravity_b": _upright_g()}, {},
    )
    still = np.zeros((T, E, 3), dtype=np.float32)
    still[:, :, 2] = 0.30
    still_out = spec_go1_trot(
        {"root_link_pos_w": still, "projected_gravity_b": _upright_g()}, {},
    )
    assert good["spec_score"] > 0.6
    assert still_out["spec_score"] == 0.0


def test_trot_spec_rejects_belly_crawl() -> None:
    """Audit H3: travel with a LEVEL torso at ground height (crawling /
    dragging) must not pass — uprightness alone can't see altitude."""
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[:, :, 0] = (np.arange(T) * 0.03)[:, None]
    p[:, :, 2] = 0.08                       # belly height
    crawl = spec_go1_trot(
        {"root_link_pos_w": p, "projected_gravity_b": _upright_g()}, {},
    )
    assert crawl["spec_score"] == 0.0
    assert crawl["height_gate"] == 0.0


# ── jump spec (§Ship 41) ─────────────────────────────────────────────


def _jump_root(height: float = 0.5, n_hops: int = 4) -> np.ndarray:
    """Repeated vertical hops from a 0.55 m crouch, NO horizontal travel."""
    z = np.full(T, 0.55)
    step = T // (n_hops + 1)
    for h in range(n_hops):
        start = step * (h + 1)
        for k in range(20):
            if start + k < T:
                z[start + k] = 0.55 + height * np.sin(np.pi * k / 20)
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[:, :, 2] = z[:, None]
    return p


def test_jump_spec_rewards_vertical_hops_only() -> None:
    good = spec_g1_jump(
        {"root_link_pos_w": _jump_root(0.6, 4),
         "projected_gravity_b": _upright_g()}, {},
    )
    still = np.zeros((T, E, 3), dtype=np.float32); still[:, :, 2] = 0.55
    still_out = spec_g1_jump(
        {"root_link_pos_w": still, "projected_gravity_b": _upright_g()}, {},
    )
    assert good["spec_score"] > 0.5, good
    assert still_out["spec_score"] == 0.0


def test_jump_spec_ignores_forward_travel() -> None:
    """A forward walker (horizontal travel, constant height) is NOT jumping —
    the spec is vertical-only, so locomotion scores zero here."""
    p = np.zeros((T, E, 3), dtype=np.float32)
    p[:, :, 0] = (np.arange(T) * 0.03)[:, None]
    p[:, :, 2] = 0.55
    out = spec_g1_jump(
        {"root_link_pos_w": p, "projected_gravity_b": _upright_g()}, {},
    )
    assert out["spec_score"] == 0.0


def test_jump_spec_rejects_fallen() -> None:
    out = spec_g1_jump(
        {"root_link_pos_w": _jump_root(0.6, 4),
         "projected_gravity_b": _fallen_g()}, {},
    )
    assert out["spec_score"] == 0.0


def test_jump_ladder_is_monotone() -> None:
    """§Ship 41: spec_g1_jump must order its calibration ladder (so a
    generated jump metric can earn steer-rights via Spearman)."""
    from sculptor.eval.metric_calibration import _ladder

    scores = [spec_g1_jump(a, b, m)["spec_score"] for a, b, m in _ladder("g1_jump")]
    assert scores == sorted(scores), scores
    assert scores[-1] > scores[0]


def test_jump_spec_rejects_vibration_noise() -> None:
    """§Ship 41 review (HIGH fix): pure sensor-noise vibration must NOT outscore
    a real jump — smoothing + robust-apex + completed-cycle counting reject it
    (the raw version scored noise 0.52, above every real ladder rung)."""
    rng = np.random.default_rng(0)
    z = 0.55 + rng.normal(0, 0.08, (T, E)).astype(np.float32)
    p = np.zeros((T, E, 3), dtype=np.float32); p[:, :, 2] = z
    noise = spec_g1_jump(
        {"root_link_pos_w": p, "projected_gravity_b": _upright_g()}, {},
    )
    real = spec_g1_jump(
        {"root_link_pos_w": _jump_root(0.6, 4),
         "projected_gravity_b": _upright_g()}, {},
    )
    assert noise["spec_score"] < real["spec_score"], (noise, real)
    assert noise["spec_score"] < 0.3, noise


def test_jump_spec_rejects_monotonic_climb() -> None:
    """§Ship 41 review (MEDIUM fix): a monotonic climb (elevator) has launches
    but no descents → 0 completed launch-and-land cycles → not a jump."""
    z = 0.55 + 0.02 * np.arange(T)
    p = np.zeros((T, E, 3), dtype=np.float32); p[:, :, 2] = z[:, None]
    out = spec_g1_jump(
        {"root_link_pos_w": p, "projected_gravity_b": _upright_g()}, {},
    )
    assert out["spec_score"] == 0.0, out


_NAMES_12 = [
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_joint", "right_ankle_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "torso_joint", "neck_joint",
]


def _floss_joints(antiphase: bool = True) -> np.ndarray:
    """Hips oscillating, arms at the same frequency — in opposition
    (genuine floss) or in phase (not a floss)."""
    jp = np.zeros((T, E, J), dtype=np.float32)
    t = np.arange(T)
    hip = 0.4 * np.sin(2 * np.pi * t / 20)
    arm_phase = np.pi if antiphase else 0.0
    arm = 0.4 * np.sin(2 * np.pi * t / 20 + arm_phase)
    for j in (0, 1):              # hips
        jp[:, :, j] = hip[:, None]
    for j in (6, 7, 8, 9):        # shoulders + elbows
        jp[:, :, j] = arm[:, None]
    jp += RNG.normal(0, 0.005, jp.shape)
    return jp


def test_floss_structure_rewards_antiphase_with_names() -> None:
    meta = {"joint_names": _NAMES_12}
    good = spec_g1_floss(
        {"joint_pos": _floss_joints(antiphase=True),
         "projected_gravity_b": _upright_g()}, {}, meta,
    )
    inphase = spec_g1_floss(
        {"joint_pos": _floss_joints(antiphase=False),
         "projected_gravity_b": _upright_g()}, {}, meta,
    )
    assert good["structure_checked"] == 1.0
    assert good["spec_score"] > 0.5
    assert good["opposition"] > 0.8
    # Same rhythm without opposition earns only the freq-match half.
    assert inphase["spec_score"] < good["spec_score"] * 0.75


def test_floss_rejects_single_joint_vibrator_with_names() -> None:
    """Audit H4: vibrating one wrist while standing is not flossing —
    the hip/arm structure gate must zero it when names are known."""
    jp = _static_joints()
    t = np.arange(T)
    jp[:, :, 11] = 0.5 * np.sin(2 * np.pi * t / 15)[:, None]  # neck only
    out = spec_g1_floss(
        {"joint_pos": jp, "projected_gravity_b": _upright_g()},
        {}, {"joint_names": _NAMES_12},
    )
    assert out["structure_checked"] == 1.0
    assert out["spec_score"] < 0.05, out


def test_kick_rejects_arm_flail_with_names() -> None:
    """Audit H1: violent ARM transients from a stable stance must not
    score as kicking when leg joints are identifiable."""
    jv = RNG.normal(0, 0.05, (T, E, J)).astype(np.float32)
    for t0 in range(25, T, 50):
        jv[t0:t0 + 3, :, 7] = 8.0          # right_shoulder bursts
    flail = spec_g1_kick(
        {"joint_vel": jv, "projected_gravity_b": _upright_g()},
        {}, {"joint_names": _NAMES_12},
    )
    leg_kick = RNG.normal(0, 0.05, (T, E, J)).astype(np.float32)
    for t0 in range(25, T, 50):
        leg_kick[t0:t0 + 3, :, 2] = 8.0    # left_knee bursts
    kick = spec_g1_kick(
        {"joint_vel": leg_kick, "projected_gravity_b": _upright_g()},
        {}, {"joint_names": _NAMES_12},
    )
    assert flail["leg_subset"] == 1.0
    assert flail["spec_score"] < 0.05, flail
    assert kick["spec_score"] > 0.5, kick


def test_kick_rejects_fall_cycling() -> None:
    """Audit H2: dramatic falls every ~2s are sustained high-velocity
    transients but must not count — bursts only qualify when launched
    from an upright window."""
    jv = RNG.normal(0, 0.05, (T, E, J)).astype(np.float32)
    g = _upright_g()
    for t0 in range(25, T, 50):
        jv[t0:t0 + 6, :, 2] = 7.0          # the "fall" transient
        g[t0 - 4:t0 + 12, :, 0] = 1.0      # tipping over around it
        g[t0 - 4:t0 + 12, :, 2] = -0.3
    out = spec_g1_kick(
        {"joint_vel": jv, "projected_gravity_b": g},
        {}, {"joint_names": _NAMES_12},
    )
    assert out["spec_score"] < 0.05, out


def test_kick_clean_kicker_scores_high_not_zero() -> None:
    """§Ship 34 regression: a CLEAN kicker — discrete strong leg bursts
    from a perfectly still upright stance (median speed ~0) — must score
    well. The pre-floor ratio guard (`med>1e-9 else 0.0`) scored the
    *ideal* kicker 0.0 (Ship-33 audit). With the rest-noise floor it
    reads as a real, repeated kick."""
    jv = np.zeros((T, E, J), dtype=np.float32)   # dead-still between kicks
    for t0 in range(25, T, 50):                  # a strong leg kick every 50
        jv[t0:t0 + 3, :, 2] = 8.0                # left_knee
    out = spec_g1_kick(
        {"joint_vel": jv, "projected_gravity_b": _upright_g()},
        {}, {"joint_names": _NAMES_12},
    )
    assert out["spec_score"] > 0.5, out
    assert out["burst_ratio_p95"] > 2.0, out     # floor → real ratio, not 0


def test_kick_penalizes_forward_travel() -> None:
    """§Ship 47: identical leg bursts score much LOWER from a TRAVELLING base
    than a stationary one — the stationarity gate stops a forward walker from
    earning kick credit off gait swings (the g1-kick-v3 Goodhart). The
    stationary kicker stays high; the guard keeps callers that omit
    root_link_pos_w unchanged."""
    jv = np.zeros((T, E, J), dtype=np.float32)
    for t0 in range(25, T, 50):
        jv[t0:t0 + 3, :, 2] = 8.0                # left_knee bursts
    g = _upright_g()
    still_root = np.zeros((T, E, 3), dtype=np.float32); still_root[:, :, 2] = 0.70
    travel_root = still_root.copy()
    travel_root[:, :, 0] = (np.arange(T) * 0.04)[:, None]
    meta = {"joint_names": _NAMES_12}
    stationary = spec_g1_kick(
        {"joint_vel": jv, "projected_gravity_b": g, "root_link_pos_w": still_root}, {}, meta)
    walking = spec_g1_kick(
        {"joint_vel": jv, "projected_gravity_b": g, "root_link_pos_w": travel_root}, {}, meta)
    no_root = spec_g1_kick({"joint_vel": jv, "projected_gravity_b": g}, {}, meta)
    assert stationary["spec_score"] > 0.5, stationary
    assert walking["spec_score"] < 0.5 * stationary["spec_score"], (walking, stationary)
    assert stationary["stationarity"] > 0.9 and walking["stationarity"] < 0.2
    # Guard: omitting root_link_pos_w must leave the score unchanged (==1.0 gate).
    assert no_root["spec_score"] == pytest.approx(stationary["spec_score"])


def test_kick_events_monotone_and_robust_to_baseline_motion() -> None:
    """§Ship 36: the discrete kick-EVENT count is monotone in genuine kicks
    and INVARIANT to competence-neutral sub-threshold baseline motion — the
    extremal-Goodhart failure the ratio score has (audit
    audit_spec_metric_monotonicity.py)."""
    g = _upright_g()

    def kicks(leg: int = 2, baseline: float = 0.0) -> np.ndarray:
        jv = np.zeros((T, E, J), dtype=np.float32)
        t = np.arange(T)
        if baseline:                              # sub-threshold whole-body motion
            jv += (baseline * np.sin(2 * np.pi * t / 17.0)).astype(np.float32)[:, None, None]
        for t0 in range(40, T, 50):               # sustained 6-frame leg kicks
            jv[t0:t0 + 6, :, leg] = 8.0
        return jv

    base = kick_events_score(kicks(baseline=0.0), g, joint_indices=[2])
    moving = kick_events_score(kicks(baseline=2.0), g, joint_indices=[2])
    assert base["kick_events"] > 0.5, base
    # adding 2 rad/s baseline (below the 5 rad/s gate) must NOT change the
    # event count — a competent walker that ALSO kicks isn't penalized.
    assert abs(moving["kick_events"] - base["kick_events"]) < 0.05, (base, moving)


def test_kick_events_rejects_arm_flail_with_leg_subset() -> None:
    """§Ship 36: with legs isolated, standing-and-flailing-arms scores ZERO
    kick events — the exact reward-hack Sam's G1 run fell into."""
    g = _upright_g()
    arm = np.zeros((T, E, J), dtype=np.float32)
    for t0 in range(40, T, 50):
        arm[t0:t0 + 6, :, 7] = 9.0                # right_shoulder flail
    flail = kick_events_score(arm, g, joint_indices=[2, 3, 4, 5])  # legs only
    assert flail["kick_events"] == 0.0, flail


def test_kick_spec_reports_kick_events_diagnostic() -> None:
    """§Ship 36: g1_kick now reports the monotone kick_events sub-component
    alongside the legacy (confounded) spec_score, for the diagnoser (F2)."""
    out = spec_g1_kick(
        {"joint_vel": _bursty_vel(), "projected_gravity_b": _upright_g()},
        {}, {"joint_names": _NAMES_12},
    )
    assert "kick_events" in out and "kick_events_per_env" in out
    assert 0.0 <= out["kick_events"] <= 1.0


def test_make_spec_fitness_fn_reads_rollout_score(tmp_path) -> None:
    """§Ship 34: make_spec_fitness_fn(name)(iter_dir) returns the named
    spec's score on iter_dir/rollout, 0.0 on missing artifacts, and
    raises on an unknown metric name (fail-fast before any GPU work)."""
    from sculptor.eval.spec_metrics import make_spec_fitness_fn

    fn = make_spec_fitness_fn("cartpole_balance")
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "behavior.json").write_text(
        json.dumps({"mean_episode_length": 450, "max_episode_steps": 500}),
        encoding="utf-8",
    )
    assert fn(tmp_path) == pytest.approx(0.9)
    # §Ship 36 (F2): the fitness fn also exposes a `.detail` accessor that
    # returns the FULL component dict for the diagnoser (not just the score).
    assert hasattr(fn, "detail")
    d = fn.detail(tmp_path)
    assert isinstance(d, dict) and d.get("spec_score") == pytest.approx(0.9)
    # missing rollout artifacts → honest 0.0, never a raise
    assert make_spec_fitness_fn("go1_trot")(tmp_path / "nope") == 0.0
    with pytest.raises(KeyError):
        make_spec_fitness_fn("not_a_metric")


def test_spec_metric_names_lists_all() -> None:
    from sculptor.eval.spec_metrics import spec_metric_names

    assert spec_metric_names() == [
        "cartpole_balance", "g1_floss", "g1_jump", "g1_kick", "go1_trot",
    ]


def test_spec_metric_robot_warning() -> None:
    """§Ship 34: warn on a metric/robot mismatch, stay silent on a match
    or when info is missing (never blocks)."""
    from sculptor.eval.spec_metrics import spec_metric_robot_warning

    # match → no warning
    assert spec_metric_robot_warning("go1_trot", "Mjlab-Velocity-Flat-Unitree-Go1") is None
    assert spec_metric_robot_warning("g1_kick", "Mjlab-Velocity-Flat-Unitree-G1") is None
    # mismatch → warning mentioning both
    w = spec_metric_robot_warning("go1_trot", "Mjlab-Velocity-Flat-Unitree-G1")
    assert w and "go1_trot" in w and "G1" in w
    # missing info → silent
    assert spec_metric_robot_warning(None, "x") is None
    assert spec_metric_robot_warning("go1_trot", None) is None


def test_cartpole_spec_from_behavior() -> None:
    assert spec_cartpole_balance({}, {"mean_episode_length": 450,
                                      "max_episode_length": 500})["spec_score"] == pytest.approx(0.9)
    assert spec_cartpole_balance({}, {})["spec_score"] == 0.0


# ── loader / dispatch ────────────────────────────────────────────────


def test_compute_spec_metrics_end_to_end(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "trajectory.npz",
        joint_pos=_periodic_joints(),
        projected_gravity_b=_upright_g(),
    )
    (tmp_path / "behavior.json").write_text(json.dumps({"mean_return": 1.0}))
    out = compute_spec_metrics("g1_floss", tmp_path)
    assert out["spec_name"] == "g1_floss"
    assert out["spec_score"] > 0.5
    assert "error" not in out


def test_compute_spec_metrics_missing_npz_is_observable_zero(tmp_path: Path) -> None:
    out = compute_spec_metrics("g1_floss", tmp_path)
    assert out["spec_score"] == 0.0
    assert "trajectory.npz missing" in out["error"]


def test_compute_spec_metrics_missing_array_is_observable_zero(tmp_path: Path) -> None:
    np.savez(tmp_path / "trajectory.npz", joint_pos=_periodic_joints())
    out = compute_spec_metrics("g1_floss", tmp_path)
    assert out["spec_score"] == 0.0
    assert "projected_gravity_b" in out["error"]


def test_compute_spec_metrics_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown spec metric"):
        compute_spec_metrics("nope", tmp_path)


def test_cartpole_spec_needs_no_npz(tmp_path: Path) -> None:
    (tmp_path / "behavior.json").write_text(
        json.dumps({"mean_episode_length": 500, "max_episode_length": 500}))
    out = compute_spec_metrics("cartpole_balance", tmp_path)
    assert out["spec_score"] == 1.0


# ── benchmark table ──────────────────────────────────────────────────


def test_benchmark_table_is_consistent() -> None:
    assert set(BENCHMARKS) == {
        "cartpole_balance", "g1_floss", "g1_kick", "go1_trot",
    }
    from sculptor.eval.spec_metrics import _SPEC_FNS

    for b in BENCHMARKS.values():
        assert b.spec_metric in _SPEC_FNS, b.name
        assert b.task_id.startswith("Mjlab-")
        assert b.behavior_goal
    assert get_benchmark("g1_floss").task_id == "Mjlab-Velocity-Flat-Unitree-G1"
    with pytest.raises(KeyError):
        get_benchmark("nonexistent")
