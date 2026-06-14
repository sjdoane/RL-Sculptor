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

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

# ── joint-name group matchers ────────────────────────────────────────

_LEG_TOKENS = ("hip", "knee", "ankle")
_HIP_TOKENS = ("hip",)
_ARM_TOKENS = ("shoulder", "elbow")


def _match_joints(names: Sequence[str], tokens: Sequence[str]) -> list[int]:
    return [
        i for i, n in enumerate(names)
        if any(t in str(n).lower() for t in tokens)
    ]


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
        hips = _match_joints(names, _HIP_TOKENS)
        arms = _match_joints(names, _ARM_TOKENS)
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


def spec_g1_kick(
    arrays: Mapping[str, np.ndarray],
    behavior: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, float]:
    """A kick = repeated high-speed LEG transients launched from an
    upright stance. spec_score = saturating burst intensity × ratio
    gate × uprightness. Bursts are leg-only when joint names exist
    (arm-flailing is not kicking — audit H1) and count only within
    fully-upright windows (falling is not kicking — audit H2). Ratio
    gate ramps 2→5, calibrated on real recordings (standing ≈ 2.2–2.4,
    real kicks ≈ 4.9–5.4); the p99 ratio also qualifies (rare kicks —
    audit M3)."""
    names = list((meta or {}).get("joint_names") or [])
    jv = arrays["joint_vel"]
    legs = _match_joints(names, _LEG_TOKENS) if len(names) == jv.shape[2] else []
    mask = upright_mask(arrays["projected_gravity_b"])
    b = burstiness(jv, joint_indices=legs or None, valid_mask=mask)
    up = uprightness(arrays["projected_gravity_b"])
    intensity = 1.0 - float(np.exp(-b["burst_p95"] / 5.0))
    ratio = max(b["burst_ratio_p95"], b["burst_ratio_p99"])
    ratio_gate = float(np.clip((ratio - 2.0) / 3.0, 0.0, 1.0))
    # §Ship 36: monotone discrete kick-event diagnostic, reported alongside
    # the (confounded) ratio score so the diagnoser sees an extremal-Goodhart-
    # robust signal. spec_score is unchanged pending real-rollout calibration.
    ev = kick_events_score(jv, arrays["projected_gravity_b"],
                           joint_indices=legs or None)
    return {
        **b,
        **ev,
        "uprightness": up,
        "leg_subset": 1.0 if legs else 0.0,
        "spec_score": float(np.clip(intensity * ratio_gate * up, 0.0, 1.0)),
    }


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


_SPEC_FNS: dict[str, Callable[..., dict[str, float]]] = {
    "cartpole_balance": spec_cartpole_balance,
    "g1_floss": spec_g1_floss,
    "g1_jump": spec_g1_jump,
    "g1_kick": spec_g1_kick,
    "go1_trot": spec_go1_trot,
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
    "g1_kick": ("joint_vel", "projected_gravity_b"),
    "go1_trot": ("root_link_pos_w", "projected_gravity_b"),
}

#: Capture settings echoed into every result for E2 parity assertions.
_CAPTURE_KEYS = ("step_dt", "max_episode_steps", "rollout_num_envs")


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

    _fitness.detail = _detail  # type: ignore[attr-defined]
    return _fitness
