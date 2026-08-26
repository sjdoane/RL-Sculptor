"""Hand-authored task-spec metrics (§Ship 26 / E1).

Each metric is OBJECTIVE ground truth for one benchmark task, computed
purely from rollout artifacts (`trajectory.npz` + `behavior.json` +
`mjcf_limits.json` joint names) — deliberately independent of the
LLM-authored success criteria, which must never grade themselves.

Available signals (what the runner actually persists):

  joint_pos / joint_vel    (T, E, J)  full articulation kinematics
  projected_gravity_b      (T, E, 3)  gravity in body frame → uprightness
                                       (yaw-INVARIANT — spin is not
                                       measurable; that's why the
                                       quadruped benchmark is gait, not
                                       spin)
  root_link_pos_w          (T, E, 3)  world base position → speed/height
  behavior.json            episode stats + capture settings (step_dt,
                           max_episode_steps, rollout_num_envs)
  mjcf_limits.json         joint_names aligned with the (T, E, J) arrays

Design rules (several are audit findings from real recordings):
  * Spectral quantities are computed PER ENV and aggregated
    incoherently (power averaging) — envs oscillate out of phase, and
    averaging positions before the FFT cancels the very signal being
    measured (audit C1: perfect flossing scored 0.22 instead of 0.99).
  * Burst quantities use a SIGNED moving average before |·| — rsl_rl
    standing policies tremble at the control rate and raw |vel| spikes
    read like kicks (measured: standing G1 ≈ 6 rad/frame raw).
  * Bursts only count when launched from an UPRIGHT window — a policy
    that falls dramatically every few seconds is not "kicking"
    (audit H2: fall-cycling scored 0.5 without the gate).
  * Joint subsets (legs / hips / arms) come from persisted joint names
    when available; metrics DEGRADE (flagged, not silently) to
    all-joints when names are absent (pre-Ship-26 recordings).
  * Frequency bands are in CYCLES PER FRAME; behavior.json's capture
    settings are echoed into every result so the E2 harness can assert
    capture parity across conditions.
  * Metrics must degrade, not crash: missing arrays yield
    `{"spec_score": 0.0, "error": ...}` — a run that produced no
    usable rollout earns an honest, observable zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from sculptor.eval.joint_resolver import (
    arm_indices,
    hip_indices,
    leg_sagittal_indices,
)

# ── joint-name group matchers ────────────────────────────────────────
# §Ship 49: the ad-hoc substring `_match_joints` (which grabbed hip
# roll/yaw alongside the forward-kick hip pitch) is retired in favour of the
# canonical, direction-aware resolver in joint_resolver.py. The built-ins
# select via `leg_sagittal_indices` / `hip_indices` / `arm_indices`.


# ── shared signal helpers ────────────────────────────────────────────


def _check_te(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"expected (T, E, ...) array, got shape {arr.shape}")
    return arr


def _normalized_gravity(projected_gravity_b: np.ndarray) -> np.ndarray:
    """Unit-norm the gravity vectors defensively — a future adapter
    persisting 9.81-scaled vectors would otherwise silently disable
    every uprightness gate."""
    g = _check_te(projected_gravity_b)
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    return g / np.maximum(norm, 1e-9)


def upright_mask(
    projected_gravity_b: np.ndarray, *, z_thresh: float = -0.85,
) -> np.ndarray:
    """(T, E) bool — gravity points down in the body frame."""
    return _normalized_gravity(projected_gravity_b)[..., 2] < z_thresh


def uprightness(projected_gravity_b: np.ndarray, *, z_thresh: float = -0.85) -> float:
    """Fraction of (step, env) samples upright. 1.0 = upright whole
    rollout; 0.0 = fallen/inverted throughout."""
    return float(np.mean(upright_mask(projected_gravity_b, z_thresh=z_thresh)))


def _sliding_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Signed boxcar mean along axis 0 via cumsum: (T, ...) →
    (T - w + 1, ...), float64 accumulation."""
    cs = np.cumsum(x, axis=0, dtype=np.float64)
    return (
        cs[w - 1:] - np.concatenate(
            [np.zeros((1, *x.shape[1:])), cs[:-w]], axis=0,
        )
    ) / w


def periodicity(
    joint_pos: np.ndarray,
    *,
    min_period_frames: int = 8,
    max_period_frames: int = 80,
    min_amplitude_rad: float = 0.05,
    joint_indices: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    """Rhythmic-motion score: per joint, how concentrated the movement
    energy is at one dominant period in the band.

    PER-ENV spectra averaged incoherently (audit C1): envs reset and
    re-randomize out of phase, so averaging positions across envs
    before the FFT cancels real oscillation (~1/sqrt(E)) and lets
    correlated transients win the mover slots. Power spectra average
    constructively regardless of phase.

    Amplitude is the robust per-env half-range ((p97.5 − p2.5)/2,
    audit L2: one glitch frame must not promote a static joint), and
    gates joints below `min_amplitude_rad` to zero. Score = mean over
    the top-quartile most-moving joints.
    """
    jp = _check_te(joint_pos)
    if joint_indices is not None and len(joint_indices) > 0:
        jp = jp[:, :, list(joint_indices)]
    T = jp.shape[0]
    zero = {"periodicity": 0.0, "dominant_period_frames": 0.0,
            "moving_joint_amplitude": 0.0}
    if T < 2 * min_period_frames:
        return zero

    x = jp.astype(np.float64) - jp.mean(axis=0, keepdims=True)  # per-env detrend
    spec = (np.abs(np.fft.rfft(x, axis=0)) ** 2).mean(axis=1)   # (F, J) incoherent
    amp = (
        (np.quantile(x, 0.975, axis=0) - np.quantile(x, 0.025, axis=0)) / 2.0
    ).mean(axis=0)                                              # (J,)

    freqs = np.fft.rfftfreq(T)                                  # cycles/frame
    band = (freqs >= 1.0 / max_period_frames) & (freqs <= 1.0 / min_period_frames)
    if not band.any():
        return zero
    total = spec[1:].sum(axis=0)
    total = np.where(total <= 0, 1e-12, total)
    band_spec = spec[band]
    peak = band_spec.max(axis=0)
    peak_idx = band_spec.argmax(axis=0)
    score_per_joint = np.where(amp >= min_amplitude_rad, peak / total, 0.0)

    k = max(1, x.shape[2] // 4)
    movers = np.argsort(amp)[-k:]
    band_freqs = freqs[band]
    # Dominant period reported over scoring movers only (audit L4).
    scoring = movers[score_per_joint[movers] > 0]
    if scoring.size:
        dom_freq = float(np.mean(band_freqs[peak_idx[scoring]]))
        dom_period = float(1.0 / dom_freq) if dom_freq > 0 else 0.0
    else:
        dom_period = 0.0
    return {
        "periodicity": float(np.clip(score_per_joint[movers].mean(), 0.0, 1.0)),
        "dominant_period_frames": dom_period,
        "moving_joint_amplitude": float(amp[movers].mean()),
    }


def burstiness(
    joint_vel: np.ndarray,
    *,
    top_fraction: float = 0.05,
    smooth_frames: int = 5,
    joint_indices: Optional[Sequence[int]] = None,
    valid_mask: Optional[np.ndarray] = None,
    ratio_floor: float = 0.5,
) -> dict[str, float]:
    """Kick-like transients: per-step peak SUSTAINED joint speed over
    the selected joints, summarized by p95/p99 and their ratios to the
    median.

    * SIGNED boxcar over `smooth_frames` before |·| — control-rate
      tremor alternates sign and cancels; a real kick sustains one
      direction and survives (calibrated on real recordings where raw
      |vel| read ~6 rad/frame on standing robots).
    * `valid_mask` (T, E) — burst samples count only where the mask
      holds across the whole smoothing window (used to require bursts
      LAUNCHED FROM UPRIGHT; falling is not kicking — audit H2).
    * p99 reported alongside p95 (audit M3): a genuine kick every ~4 s
      occupies <5% of frames and p95 alone misses it.
    """
    jv = _check_te(joint_vel)
    if joint_indices is not None and len(joint_indices) > 0:
        jv = jv[:, :, list(joint_indices)]
    T = jv.shape[0]
    w = max(1, min(smooth_frames, T))
    smoothed = _sliding_mean(jv, w)            # (T', E, J)
    speed = np.abs(smoothed).max(axis=2)       # (T', E)
    if valid_mask is not None:
        m = _check_te(valid_mask).astype(np.float64)
        m_s = _sliding_mean(m, w)              # (T', E)
        speed = np.where(m_s > 0.999, speed, 0.0)
    flat = speed.reshape(-1)
    if flat.size == 0:
        return {"burst_p95": 0.0, "burst_p99": 0.0,
                "burst_ratio_p95": 0.0, "burst_ratio_p99": 0.0}
    p95 = float(np.quantile(flat, 1.0 - top_fraction))
    p99 = float(np.quantile(flat, 0.99))
    med = float(np.median(flat))
    # §Ship 34: floor the denominator at a rest-noise level. A CLEAN
    # kicker (still between kicks) has median ~0, and the old
    # `med>1e-9 else 0.0` guard then scored the *ideal* kicker's ratio as
    # 0.0 — i.e. read a perfect kick as "no kick" (Ship-33 audit). The
    # floor gives a genuine kicker a large, real ratio; a continuously-
    # moving policy (med > floor) is unchanged, preserving the ratio's
    # scale-invariance for active policies (and every existing test).
    denom = max(med, ratio_floor)
    return {
        "burst_p95": p95,
        "burst_p99": p99,
        "burst_ratio_p95": p95 / denom,
        "burst_ratio_p99": p99 / denom,
    }


def kick_events_score(
    joint_vel: np.ndarray,
    projected_gravity_b: np.ndarray,
    *,
    joint_indices: Optional[Sequence[int]] = None,
    thresh: float = 5.0,
    smooth_frames: int = 5,
    saturate_events: float = 3.0,
) -> dict[str, float]:
    """§Ship 36: monotone discrete kick-EVENT count — a saturating count of
    sustained leg-speed transients crossing an ABSOLUTE threshold from an
    upright window, refractory-gated (one event per contiguous hot run).

    This is the audit-prescribed replacement signal for `burstiness`'s
    peak/median RATIO, which is extremal-Goodhart: competence-neutral
    baseline motion raises the median and SUPPRESSES the ratio, so a clean
    stationary flailer outscores a competent walking kicker (see
    scripts/audit_spec_metric_monotonicity.py). The event count is invariant
    to sub-threshold baseline motion and monotone in the number of genuine
    kicks. Reported as a DIAGNOSTIC sub-component of g1_kick today; promoting
    it to the spec_score awaits real-rollout threshold calibration (the
    Ship-33/34 deferral) so the absolute `thresh`/`smooth_frames` are tuned
    to measured kick durations rather than guessed."""
    jv = _check_te(joint_vel)
    if joint_indices is not None and len(joint_indices) > 0:
        jv = jv[:, :, list(joint_indices)]
    T = jv.shape[0]
    w = max(1, min(smooth_frames, T))
    sm = _sliding_mean(jv, w)                          # (T', E, J)
    speed = np.abs(sm).max(axis=2)                     # (T', E)
    up = _sliding_mean(
        upright_mask(projected_gravity_b).astype(np.float64), w,
    ) > 0.999                                          # (T', E)
    hot = (speed >= thresh) & up                       # (T', E)
    if hot.shape[0] == 0:
        return {"kick_events": 0.0, "kick_events_per_env": 0.0}
    # Rising edges per env, vectorized across envs (refractory: only a
    # False→True transition starts a new event).
    prev = np.zeros((hot.shape[1],), dtype=bool)
    events = 0
    for t in range(hot.shape[0]):
        cur = hot[t]
        events += int(np.count_nonzero(cur & ~prev))
        prev = cur
    per_env = events / max(1, hot.shape[1])
    return {
        "kick_events": 1.0 - float(np.exp(-per_env / max(1e-9, saturate_events))),
        "kick_events_per_env": float(per_env),
    }


def horizontal_speed(root_link_pos_w: np.ndarray) -> dict[str, float]:
    """Mean horizontal speed (units/frame) + straightness (net/path),
    teleport-aware (audit M2): mid-capture auto-resets warp envs back
    to spawn; those steps are excluded from the path and the net is
    summed per contiguous segment, so a terminating policy isn't
    double-punished beyond its uprightness/height gates."""
    p = _check_te(root_link_pos_w)
    T, E = p.shape[0], p.shape[1]
    if T < 2:
        return {"speed_per_frame": 0.0, "straightness": 0.0,
                "net_displacement": 0.0, "teleport_steps": 0.0}
    xy = p[..., :2].astype(np.float64)
    steps = np.diff(xy, axis=0)                # (T-1, E, 2)
    step_len = np.linalg.norm(steps, axis=2)   # (T-1, E)
    med = np.median(step_len, axis=0, keepdims=True)
    tele = step_len > np.maximum(10.0 * med, 1e-6)

    path = np.where(tele, 0.0, step_len).sum(axis=0)   # (E,)
    net = np.zeros(E)
    for e in range(E):                          # E ≤ ~64; loop is fine
        acc = np.zeros(2)
        for t in range(steps.shape[0]):
            if tele[t, e]:
                net[e] += np.linalg.norm(acc)   # close the segment
                acc = np.zeros(2)
            else:
                acc += steps[t, e]
        net[e] += np.linalg.norm(acc)
    speed = net / (T - 1)
    straight = np.divide(net, path, out=np.zeros_like(net), where=path > 1e-9)
    return {
        "speed_per_frame": float(speed.mean()),
        "straightness": float(np.clip(straight.mean(), 0.0, 1.0)),
        "net_displacement": float(net.mean()),
        "teleport_steps": float(tele.sum()),
    }


def opposition_score(
    joint_pos: np.ndarray,
    group_a: Sequence[int],
    group_b: Sequence[int],
    *,
    min_period_frames: int = 8,
    max_period_frames: int = 80,
    min_amplitude_rad: float = 0.05,
) -> dict[str, float]:
    """Anti-phase coordination between two joint groups (audit H4: the
    floss goal demands hips swinging WITH arms in opposition; generic
    periodicity alone scores arm-waving or torso-rocking as flossing).

    Per env: group-mean signals → dominant band bin of group A; gates:
    both groups' amplitudes ≥ `min_amplitude_rad`, dominant frequencies
    within one bin; score = how close the cross-phase at A's dominant
    bin is to π (1 = perfect opposition, 0 = in-phase / unrelated),
    averaged over qualifying envs (non-qualifying envs score 0).
    """
    jp = _check_te(joint_pos).astype(np.float64)
    T = jp.shape[0]
    if T < 2 * min_period_frames or not group_a or not group_b:
        return {"opposition": 0.0, "freq_match": 0.0}
    a = jp[:, :, list(group_a)].mean(axis=2)    # (T, E)
    b = jp[:, :, list(group_b)].mean(axis=2)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    amp_a = (np.quantile(a, 0.975, axis=0) - np.quantile(a, 0.025, axis=0)) / 2
    amp_b = (np.quantile(b, 0.975, axis=0) - np.quantile(b, 0.025, axis=0)) / 2

    A = np.fft.rfft(a, axis=0)
    B = np.fft.rfft(b, axis=0)
    freqs = np.fft.rfftfreq(T)
    band = (freqs >= 1.0 / max_period_frames) & (freqs <= 1.0 / min_period_frames)
    if not band.any():
        return {"opposition": 0.0, "freq_match": 0.0}
    band_idx = np.where(band)[0]
    ka = band_idx[np.abs(A[band]).argmax(axis=0)]   # (E,) dominant bin of A
    kb = band_idx[np.abs(B[band]).argmax(axis=0)]
    E = a.shape[1]
    env_scores = np.zeros(E)
    matches = np.zeros(E)
    for e in range(E):
        if amp_a[e] < min_amplitude_rad or amp_b[e] < min_amplitude_rad:
            continue
        if abs(int(ka[e]) - int(kb[e])) > 1:
            continue
        matches[e] = 1.0
        phase = np.angle(B[ka[e], e]) - np.angle(A[ka[e], e])
        d = abs((phase + np.pi) % (2 * np.pi) - np.pi)   # |wrapped| ∈ [0, π]
        # 1 at exactly π, ramps to 0 at π/2 off.
        env_scores[e] = float(np.clip(1.0 - abs(np.pi - d) / (np.pi / 2), 0, 1))
    return {
        "opposition": float(env_scores.mean()),
        "freq_match": float(matches.mean()),
    }


# ── per-benchmark specs ──────────────────────────────────────────────
# Each takes (arrays, behavior, meta) and returns a flat dict with
# `spec_score` ∈ [0, 1]. `meta` carries joint_names when persisted.


def spec_cartpole_balance(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Sanity task: the pole stayed up. Spec = mean episode length
    normalized by the PERSISTED episode cap (falls back to the runner
    default 500 for pre-Ship-26 recordings — audit M4)."""
    mean_len = float(behavior.get("mean_episode_length", 0.0) or 0.0)
    cap = float(behavior.get("max_episode_steps", 0.0) or 0.0)
    if cap <= 0:
        cap = max(float(behavior.get("max_episode_length", 0.0) or 0.0), 500.0)
    score = mean_len / cap if cap > 0 else 0.0
    return {
        "mean_episode_length": mean_len,
        "episode_cap": cap,
        "spec_score": float(np.clip(score, 0.0, 1.0)),
    }


def spec_g1_floss(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Flossing = rhythmic motion while upright, with hip↔arm
    anti-phase structure when joint names are available. spec_score =
    periodicity × uprightness × structure. Without names (old
    recordings) structure degrades to 1.0 and `structure_checked` is
    False — flagged, never silent."""
    jp = arrays["joint_pos"]
    per = periodicity(jp)
    up = uprightness(arrays["projected_gravity_b"])
    names = list((meta or {}).get("joint_names") or [])
    structure = 1.0
    checked = 0.0
    extras: dict[str, float] = {}
    if len(names) == jp.shape[2]:
        hips = hip_indices(names)
        arms = arm_indices(names)
        if hips and arms:
            opp = opposition_score(jp, hips, arms)
            extras = opp
            # Opposition is demanding; freq-matched same-band motion
            # earns partial credit so near-misses aren't crushed to 0.
            structure = float(np.clip(
                0.5 * opp["freq_match"] + 0.5 * opp["opposition"], 0.0, 1.0,
            ))
            checked = 1.0
    return {
        **per,
        **extras,
        "uprightness": up,
        "structure": structure,
        "structure_checked": checked,
        "spec_score": float(np.clip(per["periodicity"] * up * structure, 0.0, 1.0)),
    }


# §Ship 47: characteristic horizontal speed (units/frame) above which a
# base is "travelling", not holding a stance. exp(-speed/scale) → ~0.49 for
# the real g1-kick-v3 forward walker (0.0072/frame), ~0.02 for the fast
# synthetic walker (0.04/frame), ~0.90 for a near-stationary kicker
# (~0.001/frame). Tuned against the on-disk rollouts (see Ship 47 notes).
_KICK_STATIONARY_SCALE = 0.01
# §Metric-quality laws — kick-metric constants. Tuned so a real, repeated,
# FORWARD kick from an upright, stationary stance scores high while every
# documented g1-kick-v5 hack (one-leg balance, partial twitch, rear/sideways
# kick, whip-and-fall, forward walker) floors near 0. See the kick-failure
# trace in docs/internal/LAWS_OBJECTIVE_METRIC.md.
_KICK_BURST_FLOOR = 3.0          # sustained sagittal-leg speed that counts as a launch
_KICK_BURST_WIDTH = 0.8          # transition width of the completion gate on burst speed
_KICK_UPRIGHT_FLOOR = 0.6        # sustained-uprightness fraction the kick must hold
_KICK_UPRIGHT_WIDTH = 0.08
_KICK_INTENSITY_SCALE = 4.0      # saturating scale for kick speed (channel, rad/frame)
_KICK_AMPLITUDE_SCALE = 0.5      # saturating scale for sagittal-leg range-of-motion (rad)
_KICK_FOOT_SCALE = 0.12          # saturating scale for forward foot excursion (m, pelvis frame)
# §kick-fix (Sam 2026-06-20): the completion gate requires a GENUINE forward-foot
# excursion. On g1-kick-v6 a real kick swung the foot ~0.46 m forward while a
# standing balance-jiggle moved it only ~0.13 m, yet the latter scored 0.22-0.27 —
# the launch floor fired on the leg-joint burst alone. This sharp gate (G1 foot
# geometry) floors the jiggle to ~0. (Absolute metres → G1-specific; an amplitude-
# relative center is the noted follow-up for smaller robots / low-slow kicks.)
_KICK_EXCURSION_FLOOR = 0.20     # forward-foot excursion (m) a real kick must clear
_KICK_EXCURSION_WIDTH = 0.04


def _sharp_gate(x: float, center: float, width: float) -> float:
    """A steep logistic gate in [0,1]: ~0 below `center`, ~1 above, transition
    over `width`. Used to build a competence GATE that OWNS THE FLOOR (LAW 1) —
    a degenerate sub-behavior falls to ~0, not to "a little"."""
    return float(1.0 / (1.0 + np.exp(-(x - center) / max(width, 1e-6))))


def _leg_range_of_motion(
    joint_pos: np.ndarray, joint_indices: Optional[Sequence[int]] = None,
) -> float:
    """Range of motion (rad) of the MOST-moving selected leg joint: the robust
    per-(env,joint) spread (p97.5−p2.5), max over joints, mean over envs. The
    AMPLITUDE-FLOOR signal (LAW 2) — a micro-twitch has ~0 ROM, a real kick
    swings through a large arc, so this floors a tiny but correctly-shaped
    motion that the completion gate alone would let pass."""
    jp = _check_te(joint_pos)
    if joint_indices is not None and len(joint_indices) > 0:
        jp = jp[:, :, list(joint_indices)]
    if jp.shape[0] < 2 or jp.shape[2] == 0:
        return 0.0
    rng = np.quantile(jp, 0.975, axis=0) - np.quantile(jp, 0.025, axis=0)  # (E, J')
    return float(rng.max(axis=-1).mean())


def _foot_anterior_peaks(
    fp: Optional[np.ndarray],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Per-env (E,) peak FORWARD (+x) and BACKWARD (−x) anterior deviation of one
    foot from its resting median, pelvis frame, in METRES. None when the channel is
    absent / too short. The single raw kernel shared by the signed-direction channel
    (LAW 4) and the completion-gate forward-excursion gate (§kick-fix), so the metre
    excursion and the [0,1] direction score derive from one source."""
    if fp is None:
        return None
    a = _check_te(fp)[..., 0].astype(np.float64)             # (T, E) anterior
    if a.shape[0] < 2:
        return None
    dev = a - np.median(a, axis=0, keepdims=True)            # vs resting anterior
    fwd = np.clip(dev, 0.0, None).max(axis=0)                # (E,) peak forward
    back = np.clip(-dev, 0.0, None).max(axis=0)              # (E,) peak backward
    return fwd, back


def _swing_foot_forward_excursion(
    left_foot_pos_b: Optional[np.ndarray],
    right_foot_pos_b: Optional[np.ndarray],
) -> Optional[float]:
    """The SWING foot's forward (+x) peak excursion from its resting median, in
    METRES (max over feet of the per-env mean forward peak). None when NEITHER foot
    channel is present. A genuine kick swings the foot far forward; a standing
    balance-jiggle barely moves it — so this floors a non-kick whose leg-joint burst
    happens to clear the launch floor (§kick-fix)."""
    peaks = []
    for fp in (left_foot_pos_b, right_foot_pos_b):
        p = _foot_anterior_peaks(fp)
        if p is not None:
            peaks.append(p[0])                              # forward peaks (E,)
    if not peaks:
        return None
    swing = np.maximum.reduce(peaks) if len(peaks) > 1 else peaks[0]
    return float(swing.mean())


def _forward_kick_direction(
    left_foot_pos_b: Optional[np.ndarray],
    right_foot_pos_b: Optional[np.ndarray],
) -> Optional[float]:
    """Signed FORWARD-ness of the kick from foot position in the PELVIS frame
    (LAW 4). Per foot, the anterior (x) displacement from its own resting median
    splits into forward (+) and backward (−) peaks; the per-foot score is
    (forward fraction) × (saturating forward magnitude). The SWING foot (more
    motion) wins. A forward kick → ~1; a rear/mule kick → ~0 (backward peak
    dominates); a purely sideways kick → ~0 (no anterior motion). Returns None
    when NEITHER foot channel is present, so the metric ABSTAINS (LAW 6) rather
    than falling back to a direction-free magnitude that re-opens the hole."""
    def _one(fp: Optional[np.ndarray]) -> Optional[np.ndarray]:
        p = _foot_anterior_peaks(fp)
        if p is None:
            return None
        fwd, back = p
        frac = fwd / (fwd + back + 1e-6)                      # forward fraction
        mag = 1.0 - np.exp(-fwd / _KICK_FOOT_SCALE)           # forward magnitude
        return frac * mag                                     # (E,)
    feet = [s for s in (_one(left_foot_pos_b), _one(right_foot_pos_b)) if s is not None]
    if not feet:
        return None
    swing = np.maximum.reduce(feet) if len(feet) > 1 else feet[0]   # swing foot wins
    return float(np.clip(swing.mean(), 0.0, 1.0))


def spec_g1_kick(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """A kick = a real, repeated, FORWARD leg swing of meaningful amplitude,
    launched from an UPRIGHT, roughly STATIONARY stance.

    §Metric-quality-laws rebuild. The prior form was a partial-credit product
    (`intensity × ratio_gate × up × stationarity`) that scored degenerate
    sub-motions 0.13–0.38 instead of 0.0 and was reward-hacked over 21
    g1-kick-v5 iterations (balance-on-one-leg; kick-behind; whip-and-fall).
    Now a composed gate (LAW 1):

        spec_score = completion_gate · min(quality_channels)

    `completion_gate` ∈ {0,1}-SHARP owns the floor — a sustained sagittal-leg
    burst (a launch) AND a sustained-upright AND non-travelling stance; a
    one-leg balance (no burst), a whip-and-fall (not upright) and a forward
    walker (travelling) all gate to ~0. Quality channels (each saturating
    [0,1], combined by MIN so none can be traded away):
      * intensity — sagittal-leg burst speed (how forceful).
      * amplitude — sagittal-leg range of motion (LAW 2; a micro-twitch ~0);
                    ABSTAINS when joint_pos is absent.
      * direction — signed FORWARD foot displacement in the pelvis frame
                    (LAW 4; a rear/sideways kick ~0); ABSTAINS (LAW 6) when the
                    left/right_foot_pos_b channels are absent — never a
                    direction-free magnitude proxy.
    Sagittal-plane legs only (hip pitch + knee + ankle pitch — §Ship 49), and
    stationarity is a VETO inside the gate, NEVER positive credit (a frozen
    one-leg pose is maximally stationary — the §Ship 47 lesson, hardened)."""
    names = list((meta or {}).get("joint_names") or [])
    jv = arrays["joint_vel"]
    legs = leg_sagittal_indices(names) if len(names) == jv.shape[2] else []
    g = arrays["projected_gravity_b"]
    mask = upright_mask(g)
    b = burstiness(jv, joint_indices=legs or None, valid_mask=mask)
    up = uprightness(g)
    ev = kick_events_score(jv, g, joint_indices=legs or None)

    # ── completion gate (sharp; owns the floor) ──────────────────────────
    # A real launch: sustained sagittal-leg burst above a floor, within upright
    # windows (burstiness is already masked to upright). kick_events is a
    # refractory-counted confirmation, reported as a diagnostic alongside.
    launch = _sharp_gate(b["burst_p95"], _KICK_BURST_FLOOR, _KICK_BURST_WIDTH)
    upright_gate = _sharp_gate(up, _KICK_UPRIGHT_FLOOR, _KICK_UPRIGHT_WIDTH)
    # §Ship 47: stationarity VETO. Degrades to 1.0 (no veto) when
    # root_link_pos_w is absent so leg-only callers/ladders are unchanged.
    root = arrays.get("root_link_pos_w")
    if root is not None:
        speed = float(horizontal_speed(root)["speed_per_frame"])
        stationarity = float(np.clip(np.exp(-speed / _KICK_STATIONARY_SCALE), 0.0, 1.0))
    else:
        speed, stationarity = 0.0, 1.0
    # §kick-fix: require a GENUINE forward-foot excursion — a standing leg-jiggle
    # clears the burst launch floor but barely moves the foot forward. Degrades to
    # 1.0 (no veto) when NEITHER foot channel is present, so the footless
    # calibration ladder + any leg-only caller stay byte-identical (and monotone).
    fwd_exc = _swing_foot_forward_excursion(
        arrays.get("left_foot_pos_b"), arrays.get("right_foot_pos_b"))
    excursion_gate = (1.0 if fwd_exc is None
                      else _sharp_gate(fwd_exc, _KICK_EXCURSION_FLOOR, _KICK_EXCURSION_WIDTH))
    completion_gate = float(launch * upright_gate * stationarity * excursion_gate)

    # ── quality channels (min; abstain on absent data) ───────────────────
    intensity = 1.0 - float(np.exp(-b["burst_p95"] / _KICK_INTENSITY_SCALE))
    channels = [intensity]

    jp = arrays.get("joint_pos")
    amplitude: Optional[float] = None
    if jp is not None and getattr(jp, "ndim", 0) >= 3 and jp.shape[2] == jv.shape[2]:
        rom = _leg_range_of_motion(jp, legs or None)
        amplitude = 1.0 - float(np.exp(-rom / _KICK_AMPLITUDE_SCALE))
        channels.append(amplitude)

    direction = _forward_kick_direction(
        arrays.get("left_foot_pos_b"), arrays.get("right_foot_pos_b"))
    if direction is not None:
        channels.append(direction)

    quality = float(min(channels))
    score = float(np.clip(completion_gate * quality, 0.0, 1.0))

    out = {
        **b,
        **ev,
        "uprightness": up,
        "leg_subset": 1.0 if legs else 0.0,
        "stationarity": stationarity,
        "horizontal_speed": speed,
        "completion_gate": completion_gate,
        "kick_intensity": intensity,
        "spec_score": score,
    }
    if amplitude is not None:
        out["kick_amplitude"] = amplitude
    if direction is not None:
        out["kick_direction"] = direction
    else:
        out["direction_abstained"] = 1.0
    return out


def spec_g1_jump(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """§Ship 41: a jump = repeated VERTICAL launch-and-LAND cycles of the base
    from an upright stance. spec_score = saturating apex height x completed-hop
    count x uprightness. Mirrors spec_g1_kick's noise-robustness (the §Ship 41
    review found a raw version rewarded sensor VIBRATION above real jumps):
      * base height and its velocity are SIGNED-smoothed (5-frame) before edge
        detection — control-rate jitter / sensor noise cancels;
      * apex is a robust per-env half-range (p97.5 above the resting median),
        so a single glitch frame cannot inflate height (audit L2 pattern);
      * a hop counts only as a COMPLETED cycle = an upward launch matched by a
        later descent, both within an UPRIGHT window — a monotonic climb
        (elevator) has launches but no descents and scores 0; a fall is not
        upright so its descent does not count.
    Forward travel scores 0 (vertical-only; yaw-/horizontal-invariant)."""
    g = arrays["projected_gravity_b"]
    rp = _check_te(arrays["root_link_pos_w"])
    z = rp[..., 2].astype(np.float64)                 # (T, E)
    up_frac = uprightness(g)
    T = z.shape[0]
    zero = {"apex_height": 0.0, "height_score": 0.0, "launches_per_env": 0.0,
            "repeat_score": 0.0, "uprightness": up_frac, "spec_score": 0.0}
    if T < 8:
        return zero
    w = max(1, min(5, T))
    zs = _sliding_mean(z, w)                           # (T', E) smoothed height
    base = np.median(zs, axis=0, keepdims=True)
    apex = float(np.clip(
        (np.quantile(zs, 0.975, axis=0) - base[0]).mean(), 0.0, None))
    height_score = 1.0 - float(np.exp(-apex / 0.3))
    vel = _sliding_mean(np.diff(z, axis=0), w)         # (T'', E) smoothed vel
    up_win = _sliding_mean(upright_mask(g).astype(np.float64), w) > 0.999
    up_win = up_win[:vel.shape[0]]                     # align to vel frames
    thr = 0.01                                         # ≈0.5 m/s vertical @dt=0.02

    def _edges(cond: np.ndarray) -> np.ndarray:
        """Per-env rising-edge counts of a (T, E) boolean."""
        counts = np.zeros(cond.shape[1], dtype=np.int64)
        prev = np.zeros(cond.shape[1], dtype=bool)
        for tt in range(cond.shape[0]):
            cur = cond[tt]
            counts += (cur & ~prev)
            prev = cur
        return counts

    ups = _edges((vel >= thr) & up_win)
    downs = _edges((vel <= -thr) & up_win)
    completed = np.minimum(ups, downs)                 # launch matched by a land
    per_env = float(completed.mean())
    repeat_score = 1.0 - float(np.exp(-per_env / 3.0))
    return {
        "apex_height": apex,
        "height_score": height_score,
        "launches_per_env": per_env,
        "repeat_score": repeat_score,
        "uprightness": up_frac,
        "spec_score": float(np.clip(
            height_score * repeat_score * up_frac, 0.0, 1.0)),
    }


def spec_go1_trot(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Forward gait = sustained straight horizontal travel, upright AND
    at stance height (audit H3: uprightness checks orientation only —
    a belly-crawl with level torso passed; root z gates it: go1 stance
    ≈ 0.30 m, gate ramps 0.18→0.28 m). Speed saturates via
    1 − exp(−v/0.02 units/frame)."""
    sp = horizontal_speed(arrays["root_link_pos_w"])
    up = uprightness(arrays["projected_gravity_b"])
    mean_z = float(np.mean(arrays["root_link_pos_w"][..., 2]))
    height_gate = float(np.clip((mean_z - 0.18) / 0.10, 0.0, 1.0))
    speed_score = 1.0 - float(np.exp(-sp["speed_per_frame"] / 0.02))
    return {
        **sp,
        "uprightness": up,
        "mean_root_height": mean_z,
        "height_gate": height_gate,
        "spec_score": float(np.clip(
            speed_score * sp["straightness"] * up * height_gate, 0.0, 1.0,
        )),
    }


def _prefix_length(mask: np.ndarray) -> int:
    """Length of the initial true run in a one-dimensional boolean mask."""
    false = np.flatnonzero(~mask)
    return int(false[0]) if false.size else int(mask.size)


def _longest_true_run(mask: np.ndarray) -> int:
    """Longest contiguous true run in a one-dimensional boolean mask."""
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def spec_object_lift_hold(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """Target-aware lift-clear-and-hold completion for grasping robots.

    This metric is deliberately capability- and artifact-driven: there are no
    robot, task, or object-name branches. A completion requires one target to
    begin below its world-space goal, rise at least 8 cm, remain within the
    task's (capped) 5 cm goal tolerance for 0.5 s, remain mechanically held by
    at least two independent contact groups, and stay dynamically quiet. For
    multi-object tasks every non-target object must remain within 5 cm of its
    initial pose through completion.

    Episode and command boundaries are hard boundaries. Invalid target IDs,
    post-reset samples, command changes, implausible state jumps, and
    excessive velocity turn the affected environment into an honest zero;
    non-finite telemetry and forged grasp/contact disagreement fail the
    whole artifact closed. ``spec_score`` is the binary
    completion rate across the originally requested vector environments; the
    continuous fields are diagnostics only and never substitute for completion.
    """
    contract = dict((meta or {}).get("manipulation_telemetry") or {})
    object_names = tuple(contract.get("object_names") or ())
    finger_groups = tuple(sorted((contract.get("finger_groups") or {}).keys()))
    if not object_names:
        raise ValueError("manipulation telemetry declares no objects")
    if not bool(contract.get("grasp_capable")) or len(finger_groups) < 2:
        raise ValueError(
            "object_lift_hold requires two independent grasp-contact groups")

    step_dt = float(behavior.get("step_dt") or 0.0)
    if not np.isfinite(step_dt) or not 0.001 <= step_dt <= 0.2:
        raise ValueError(f"invalid behavior.step_dt {step_dt!r}")

    target_idx = np.asarray(arrays["target_object_index"])
    target_pos = np.asarray(arrays["target__pos_w"], dtype=np.float64)
    valid = np.asarray(arrays["rollout_valid"], dtype=bool)
    terminal = np.asarray(arrays["rollout_terminal"], dtype=bool)
    if target_idx.ndim != 2:
        raise ValueError(
            f"target_object_index must be (T, E), got {target_idx.shape}")
    T, E = target_idx.shape
    if target_pos.shape != (T, E, 3):
        raise ValueError(f"target__pos_w must be {(T, E, 3)}")
    if valid.shape != (T, E) or terminal.shape != (T, E):
        raise ValueError("rollout masks must match target_object_index")

    positions: list[np.ndarray] = []
    lin_velocities: list[np.ndarray] = []
    ang_velocities: list[np.ndarray] = []
    grasps: list[np.ndarray] = []
    for name in object_names:
        pos = np.asarray(arrays[f"object__{name}__pos_w"], dtype=np.float64)
        lin = np.asarray(
            arrays[f"object__{name}__lin_vel_w"], dtype=np.float64)
        ang = np.asarray(
            arrays[f"object__{name}__ang_vel_w"], dtype=np.float64)
        grasp = np.asarray(arrays[f"grasp__{name}"]) > 0.5
        if pos.shape != (T, E, 3) or lin.shape != (T, E, 3) \
                or ang.shape != (T, E, 3) or grasp.shape != (T, E):
            raise ValueError(f"object {name!r} telemetry shapes are inconsistent")
        contacts = [
            np.asarray(arrays[f"contact__{group}__{name}"]) > 0.5
            for group in finger_groups
        ]
        if any(contact.shape != (T, E) for contact in contacts):
            raise ValueError(f"object {name!r} contact shapes are inconsistent")
        derived = np.sum(np.stack(contacts, axis=0), axis=0) >= 2
        if not np.array_equal(grasp, derived):
            raise ValueError(
                f"grasp__{name} disagrees with independent contact channels")
        positions.append(pos)
        lin_velocities.append(lin)
        ang_velocities.append(ang)
        grasps.append(grasp)

    pos_all = np.stack(positions, axis=0)       # (O, T, E, 3)
    lin_all = np.stack(lin_velocities, axis=0)
    ang_all = np.stack(ang_velocities, axis=0)
    grasp_all = np.stack(grasps, axis=0)        # (O, T, E)
    if not all(np.all(np.isfinite(value)) for value in (
        target_pos, pos_all, lin_all, ang_all,
    )):
        raise ValueError("manipulation telemetry contains non-finite values")

    declared_tolerance = (
        (contract.get("target_contract") or {})
        .get("declared_success_threshold_m", 0.05)
    )
    try:
        target_tolerance = min(0.05, float(declared_tolerance))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid target success threshold") from exc
    if not np.isfinite(target_tolerance) or target_tolerance <= 0:
        raise ValueError("invalid target success threshold")

    baseline_frames = max(2, int(np.ceil(0.10 / step_dt)))
    pickup_frames = max(baseline_frames, int(np.ceil(0.20 / step_dt)))
    hold_frames = max(2, int(np.ceil(0.50 / step_dt)))
    max_grasp_gap = max(1, int(np.floor(0.10 / step_dt)))
    min_clearance = 0.08
    max_distractor_displacement = 0.05

    completions = np.zeros(E, dtype=bool)
    structurally_valid = np.zeros(E, dtype=bool)
    physically_valid = np.zeros(E, dtype=bool)
    peak_lifts = np.zeros(E, dtype=np.float64)
    min_goal_errors = np.full(E, np.inf, dtype=np.float64)
    best_grasp_fractions = np.zeros(E, dtype=np.float64)
    longest_holds = np.zeros(E, dtype=np.float64)
    distractor_displacements = np.zeros(E, dtype=np.float64)

    for env_index in range(E):
        env_valid = valid[:, env_index]
        valid_len = _prefix_length(env_valid)
        # The validity mask must be one contiguous prefix. A terminal marker is
        # either absent (capture ended) or exactly the first invalid sample.
        if np.any(env_valid[valid_len:]):
            continue
        terminal_at = np.flatnonzero(terminal[:, env_index])
        if terminal_at.size > 1:
            continue
        if terminal_at.size and int(terminal_at[0]) != valid_len:
            continue
        if valid_len < baseline_frames + hold_frames:
            continue

        first_index = int(target_idx[0, env_index])
        if not 0 <= first_index < len(object_names):
            continue
        first_target = target_pos[0, env_index]
        # Treat a command/identity resample as a new attempt. Only the first
        # uninterrupted command segment can score, preventing mid-rollout
        # object resets from masquerading as transport.
        same_identity = target_idx[:valid_len, env_index] == first_index
        same_command = np.linalg.norm(
            target_pos[:valid_len, env_index] - first_target, axis=-1,
        ) <= 1e-6
        segment_len = _prefix_length(same_identity & same_command)
        if segment_len < baseline_frames + hold_frames:
            continue
        structurally_valid[env_index] = True

        obj_pos = pos_all[first_index, :segment_len, env_index]
        obj_lin = lin_all[first_index, :segment_len, env_index]
        obj_ang = ang_all[first_index, :segment_len, env_index]
        obj_grasp = grasp_all[first_index, :segment_len, env_index]
        baseline = np.median(obj_pos[:baseline_frames], axis=0)
        initial_goal_error = float(np.linalg.norm(first_target - baseline))
        vertical_goal_gap = float(first_target[2] - baseline[2])
        if vertical_goal_gap < min_clearance or initial_goal_error < min_clearance:
            continue
        if float(np.mean(obj_grasp[:baseline_frames])) > 0.2:
            continue
        # Spawned rigid bodies can settle for a few frames. A maximum-speed
        # gate made a single normal contact impulse veto the whole attempt on
        # real YAM rollouts; the median rejects sustained initial motion while
        # the jump/global-speed gates below still reject resets and explosions.
        if float(np.median(np.linalg.norm(
            obj_lin[:baseline_frames], axis=-1,
        ))) > 0.5:
            continue

        # Global plausibility gates are relative (vectorized world origins may
        # be far from zero). The absolute displacement floor below tolerates
        # contact impulses, so a sub-floor crawl with forged zero velocities
        # is NOT caught here: honest simulator provenance is established by
        # the spec-audit evidence hashes, not by these physics gates alone.
        implausible = False
        for object_index in range(len(object_names)):
            p = pos_all[object_index, :segment_len, env_index]
            lv = lin_all[object_index, :segment_len, env_index]
            av = ang_all[object_index, :segment_len, env_index]
            speed = np.linalg.norm(lv, axis=-1)
            if np.max(speed) > 10.0 or np.max(np.linalg.norm(av, axis=-1)) > 100.0:
                implausible = True
                break
            if segment_len > 1:
                step_displacement = np.linalg.norm(np.diff(p, axis=0), axis=-1)
                speed_bound = (
                    3.0 * np.maximum(speed[:-1], speed[1:]) * step_dt + 0.02)
                if np.any(step_displacement > np.maximum(0.12, speed_bound)):
                    implausible = True
                    break
            object_baseline = np.median(p[:baseline_frames], axis=0)
            if np.max(np.linalg.norm(p - object_baseline, axis=-1)) > 5.0:
                implausible = True
                break
        if implausible:
            continue
        physically_valid[env_index] = True

        lift = obj_pos[:, 2] - baseline[2]
        error = np.linalg.norm(obj_pos - first_target, axis=-1)
        lin_speed = np.linalg.norm(obj_lin, axis=-1)
        ang_speed = np.linalg.norm(obj_ang, axis=-1)
        goal_stable = (
            (lift >= min_clearance)
            & (error <= target_tolerance)
            & (lin_speed <= 0.15)
            & (ang_speed <= 1.0)
        )
        peak_lifts[env_index] = max(0.0, float(np.max(lift)))
        min_goal_errors[env_index] = float(np.min(error))
        longest_holds[env_index] = (
            _longest_true_run(goal_stable) * step_dt)

        for start in range(pickup_frames, segment_len - hold_frames + 1):
            stop = start + hold_frames
            if not np.all(goal_stable[start:stop]):
                continue
            held = obj_grasp[start:stop]
            grasp_fraction = float(np.mean(held))
            best_grasp_fractions[env_index] = max(
                best_grasp_fractions[env_index], grasp_fraction)
            if grasp_fraction < 0.8:
                continue
            if _longest_true_run(~held) > max_grasp_gap:
                continue

            disturbed = 0.0
            for other_index in range(len(object_names)):
                if other_index == first_index:
                    continue
                other = pos_all[other_index, :stop, env_index]
                other_base = np.median(other[:baseline_frames], axis=0)
                disturbed = max(
                    disturbed,
                    float(np.max(np.linalg.norm(other - other_base, axis=-1))),
                )
            distractor_displacements[env_index] = disturbed
            if disturbed > max_distractor_displacement:
                continue
            completions[env_index] = True
            break

    finite_goal_errors = min_goal_errors[np.isfinite(min_goal_errors)]
    return {
        "completion_rate": float(np.mean(completions)) if E else 0.0,
        "structurally_valid_rate": float(np.mean(structurally_valid)) if E else 0.0,
        "physically_valid_rate": float(np.mean(physically_valid)) if E else 0.0,
        "mean_peak_lift_m": float(np.mean(peak_lifts)) if E else 0.0,
        "mean_min_goal_error_m": (
            float(np.mean(finite_goal_errors))
            if finite_goal_errors.size else 0.0),
        "mean_best_grasp_fraction": (
            float(np.mean(best_grasp_fractions)) if E else 0.0),
        "mean_longest_goal_hold_s": (
            float(np.mean(longest_holds)) if E else 0.0),
        "max_distractor_displacement_m": (
            float(np.max(distractor_displacements)) if E else 0.0),
        "target_tolerance_m": target_tolerance,
        "required_hold_s": hold_frames * step_dt,
        "spec_score": float(np.mean(completions)) if E else 0.0,
    }


_SPEC_FNS: dict[str, Callable[..., dict[str, float]]] = {
    "cartpole_balance": spec_cartpole_balance,
    "g1_floss": spec_g1_floss,
    "g1_jump": spec_g1_jump,
    "g1_kick": spec_g1_kick,
    "go1_trot": spec_go1_trot,
    "object_lift_hold": spec_object_lift_hold,
}


def spec_metric_names() -> list[str]:
    """Public, sorted list of available spec-metric names. Source of
    truth for the UI's objective-fitness dropdown and the backend
    Literal — keep those in sync with this set (§Ship 34)."""
    return sorted(_SPEC_FNS)


#: §Ship 34: the robot family each spec metric is CALIBRATED for. The
#: metrics read arrays generically (so go1_trot's speed/straightness is
#: usable for any forward-locomotor), but several bake in robot-specific
#: constants (go1 stance-height gate ≈ 0.30 m; g1 hip/arm joint tokens).
#: Used for a soft mismatch warning — NOT a hard gate, since cross-robot
#: locomotion use is legitimate.
_METRIC_ROBOT_HINTS: dict[str, tuple[str, ...]] = {
    "cartpole_balance": ("cartpole",),
    "go1_trot": ("go1", "go2"),
    "g1_floss": ("g1",),
    "g1_jump": ("g1",),
    "g1_kick": ("g1",),
}


def spec_metric_robot_warning(
    spec_name: Optional[str], task_id: Optional[str],
) -> Optional[str]:
    """Return a human-readable warning if `spec_name` looks mismatched to
    the env `task_id` (e.g. fitness_metric='go1_trot' on a G1 task), else
    None. Soft check: the score is still computable, but its calibration
    (stance height, joint-name groups) may be wrong, so the loop could
    optimize the wrong thing. Callers WARN, they do not block."""
    if not spec_name or not task_id:
        return None
    hints = _METRIC_ROBOT_HINTS.get(spec_name)
    if not hints:
        return None
    tid = str(task_id).lower()
    if any(h in tid for h in hints):
        return None
    return (
        f"fitness_metric={spec_name!r} is calibrated for "
        f"{'/'.join(hints)} but the env task is {task_id!r} — the "
        f"objective fitness may be semantically wrong for this robot "
        f"(check the spec's stance-height / joint-group assumptions)."
    )

#: Arrays each spec needs from trajectory.npz (cartpole needs none —
#: it works off behavior.json alone).
_REQUIRED_ARRAYS: dict[str, tuple[str, ...]] = {
    "cartpole_balance": (),
    "g1_floss": ("joint_pos", "projected_gravity_b"),
    "g1_jump": ("root_link_pos_w", "projected_gravity_b"),
    # §Ship 47: g1_kick now also loads root_link_pos_w for the stationarity
    # gate (mjlab always writes it to trajectory.npz). The in-fn guard keeps
    # direct callers that omit it (synthetic ladders, leg-only unit tests)
    # working with stationarity=1.0.
    "g1_kick": ("joint_vel", "projected_gravity_b", "root_link_pos_w"),
    "go1_trot": ("root_link_pos_w", "projected_gravity_b"),
    # Object-specific state/contact keys are expanded from the hashable
    # manipulation sidecar by `_manipulation_required_arrays`.
    "object_lift_hold": (
        "target_object_index", "target__pos_w",
        "rollout_valid", "rollout_terminal",
    ),
}

#: §Ship 54-pre (#12): the FULL set of physical observables each spec metric MAY
#: read — `_REQUIRED_ARRAYS` plus the arrays the fn reads opportunistically via
#: `arrays.get(...)`. This is the metric's "held-out test surface": the
#: shaping↔metric partition gate (`partition_gate`) warns when a reward edit
#: touches one of these and hard-rejects a reward that lowers a completion gate.
#: g1_kick additionally reads joint_pos (amplitude), left/right_foot_pos_b
#: (signed forward direction) when present (spec_metrics.py:575-585).
_METRIC_OBSERVABLES: dict[str, tuple[str, ...]] = {
    "cartpole_balance": (),
    "g1_floss": ("joint_pos", "projected_gravity_b"),
    "g1_jump": ("root_link_pos_w", "projected_gravity_b"),
    "g1_kick": (
        "joint_vel", "projected_gravity_b", "root_link_pos_w",
        "joint_pos", "left_foot_pos_b", "right_foot_pos_b",
    ),
    "go1_trot": ("root_link_pos_w", "projected_gravity_b"),
    "object_lift_hold": (
        "target_object_index", "target__pos_w",
        "rollout_valid", "rollout_terminal",
        "object_state", "object_contact", "grasp_evidence",
    ),
}


def metric_observables(spec_name: str) -> frozenset[str]:
    """The physical observables `spec_name` scores (its held-out surface for the
    shaping↔metric partition gate). Falls back to `_REQUIRED_ARRAYS` for any
    spec not in `_METRIC_OBSERVABLES`; empty set for an unknown name (the gate
    then no-ops, never raises)."""
    obs = _METRIC_OBSERVABLES.get(spec_name)
    if obs is None:
        obs = _REQUIRED_ARRAYS.get(spec_name, ())
    return frozenset(obs)


#: Capture settings echoed into every result for E2 parity assertions.
_CAPTURE_KEYS = ("step_dt", "max_episode_steps", "rollout_num_envs")


def _load_manipulation_manifest(rollout_dir: Path) -> dict[str, Any]:
    """Load and strictly validate the dynamic manipulation array contract."""
    path = rollout_dir / "manipulation_telemetry.json"
    if not path.is_file():
        raise ValueError(f"manipulation_telemetry.json missing in {rollout_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manipulation telemetry manifest must be an object")
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "object_lift_hold requires manipulation telemetry schema 2 "
            "with first-episode masks")
    names = manifest.get("object_names")
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name for name in names
    ) or len(set(names)) != len(names):
        raise ValueError("manipulation telemetry object_names are invalid")
    groups = manifest.get("finger_groups")
    if not isinstance(groups, dict) or len(groups) < 2 or any(
        not isinstance(group, str) or not group
        or not isinstance(bodies, list) or not bodies
        for group, bodies in groups.items()
    ):
        raise ValueError(
            "object_lift_hold requires at least two contact-evidence groups")
    if manifest.get("grasp_capable") is not True:
        raise ValueError("manipulation telemetry is not grasp-capable")
    target_contract = manifest.get("target_contract")
    if not isinstance(target_contract, dict) \
            or target_contract.get("position_frame") != "world":
        raise ValueError("target position must declare the world frame")
    channels = manifest.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("manipulation telemetry channels are missing")
    return manifest


def _manipulation_required_arrays(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    needed = list(_REQUIRED_ARRAYS["object_lift_hold"])
    groups = sorted(manifest["finger_groups"])
    for name in manifest["object_names"]:
        needed.extend((
            f"object__{name}__pos_w",
            f"object__{name}__lin_vel_w",
            f"object__{name}__ang_vel_w",
            f"grasp__{name}",
        ))
        needed.extend(f"contact__{group}__{name}" for group in groups)
    return tuple(needed)


def compute_spec_metrics(
    spec_name: str,
    rollout_dir: Path | str,
    *,
    behavior: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load a rollout dir's artifacts and compute the named spec.

    Never raises on bad/missing artifacts: returns
    `{"spec_score": 0.0, "error": ...}` so the harness can aggregate a
    failed run as an honest zero instead of dropping it (silent
    dropping inflates means)."""
    if spec_name not in _SPEC_FNS:
        raise KeyError(
            f"unknown spec metric {spec_name!r}; known: {sorted(_SPEC_FNS)}"
        )
    rollout_dir = Path(rollout_dir)
    try:
        if behavior is None:
            bpath = rollout_dir / "behavior.json"
            behavior = (
                json.loads(bpath.read_text(encoding="utf-8"))
                if bpath.is_file() else {}
            )
        meta: dict[str, Any] = {}
        limits_path = rollout_dir / "mjcf_limits.json"
        if limits_path.is_file():
            try:
                limits = json.loads(limits_path.read_text(encoding="utf-8"))
                names = limits.get("joint_names") or []
                if names:
                    meta["joint_names"] = [str(n) for n in names]
            except Exception:  # noqa: BLE001 — names are an upgrade, not a dep
                pass
        arrays: dict[str, np.ndarray] = {}
        needed = _REQUIRED_ARRAYS[spec_name]
        if spec_name == "object_lift_hold":
            manipulation_manifest = _load_manipulation_manifest(rollout_dir)
            meta["manipulation_telemetry"] = manipulation_manifest
            needed = _manipulation_required_arrays(manipulation_manifest)
        if needed:
            npz_path = rollout_dir / "trajectory.npz"
            if not npz_path.is_file():
                return {
                    "spec_name": spec_name, "spec_score": 0.0,
                    "error": f"trajectory.npz missing in {rollout_dir}",
                }
            with np.load(npz_path) as z:
                missing = [k for k in needed if k not in z.files]
                if missing:
                    return {
                        "spec_name": spec_name, "spec_score": 0.0,
                        "error": f"trajectory.npz lacks arrays {missing}",
                    }
                for k in needed:
                    arrays[k] = z[k]
                    if spec_name == "object_lift_hold":
                        declared = manipulation_manifest["channels"].get(k)
                        if not isinstance(declared, dict):
                            raise ValueError(
                                f"manipulation manifest omits channel {k!r}")
                        if declared.get("shape") != list(arrays[k].shape):
                            raise ValueError(
                                f"manifest shape disagrees for channel {k!r}")
                        if declared.get("dtype") != str(arrays[k].dtype):
                            raise ValueError(
                                f"manifest dtype disagrees for channel {k!r}")
        out = _SPEC_FNS[spec_name](arrays, behavior, meta)
        capture = {k: behavior.get(k) for k in _CAPTURE_KEYS if k in behavior}
        return {"spec_name": spec_name, **out, "capture": capture}
    except Exception as e:  # noqa: BLE001 — zero, observably
        return {
            "spec_name": spec_name,
            "spec_score": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }


def make_spec_fitness_fn(spec_name: str) -> Callable[[Any], float]:
    """§Ship 34: build a `fitness_fn(iter_dir) -> float` from a spec
    metric name, for sculpt_run/mission_run's fitness-in-loop. Given an
    iteration dir, scores its `rollout/` with the named spec and returns
    `spec_score` (0.0 on any missing/failed artifact — never raises, so a
    bad iter is an honest zero rather than a crashed run).

    Used by both the eval harness (--fitness-in-loop) and the UI's
    objective-fitness dropdown (sculpt run --fitness-metric)."""
    if spec_name not in _SPEC_FNS:
        raise KeyError(
            f"unknown spec metric {spec_name!r}; known: {sorted(_SPEC_FNS)}"
        )

    def _fitness(iter_dir: Any) -> float:
        result = compute_spec_metrics(spec_name, Path(iter_dir) / "rollout")
        return float(result.get("spec_score", 0.0) or 0.0)

    def _detail(iter_dir: Any) -> dict:
        # §Ship 36 (F2): the FULL component breakdown (not just spec_score)
        # for the diagnoser's OBJECTIVE_TASK_PROGRESS block — lets the LLM
        # localize WHAT is wrong (e.g. high burst but low uprightness =
        # violent-but-falling). Rides on the fitness fn so no new param is
        # threaded through sculpt_run/mission_run. Never raises.
        try:
            return compute_spec_metrics(spec_name, Path(iter_dir) / "rollout")
        except Exception:  # noqa: BLE001 — breakdown is advisory, never fatal
            return {}

    def _detail_dir(rollout_dir: Any) -> dict:
        # §Selection statistics: score an ARBITRARY rollout dir (multi-seed
        # evaluation rolls into `rollout_eval_<k>/` beside the primary
        # `rollout/`; fresh-seed re-eval into `rollout_fresh_<j>/`). Same
        # computation as `_detail` minus the hardcoded subdir. Never raises.
        try:
            return compute_spec_metrics(spec_name, Path(rollout_dir))
        except Exception:  # noqa: BLE001 — advisory, never fatal
            return {}

    _fitness.detail = _detail  # type: ignore[attr-defined]
    _fitness.detail_dir = _detail_dir  # type: ignore[attr-defined]
    # §Ship 54-pre (#12): expose the metric's held-out observable surface so the
    # sculpt loop can hand it to the shaping↔metric partition gate at the
    # reward-edit commit point (sculpt.py passes
    # `metric_observables=getattr(fitness_fn, "metric_observables", None)`).
    _fitness.metric_observables = metric_observables(spec_name)  # type: ignore[attr-defined]
    _fitness.spec_name = spec_name  # type: ignore[attr-defined]
    _fitness.metric_id = spec_name  # type: ignore[attr-defined]
    _fitness.metric_version = None  # type: ignore[attr-defined]
    _fitness.metric_source = "built_in"  # type: ignore[attr-defined]
    try:
        _fitness.metric_sha256 = hashlib.sha256(  # type: ignore[attr-defined]
            Path(__file__).read_bytes()
        ).hexdigest()
    except OSError:
        _fitness.metric_sha256 = None  # type: ignore[attr-defined]
    return _fitness
