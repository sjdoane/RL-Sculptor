"""Reference-clip perturbations: hard negatives + calibration-ladder rungs
derived FROM a reference clip (§REFERENCE_TRAJECTORY_PLAN §5.2, §6).

Every function takes a validated `sculptor.reference` clip dict and returns
a NEW valid clip dict (`validate_clip`-clean), operating on every `(T, ...)`
array key consistently (`root_pos_z`, `root_vel_z`, `root_pos_xy`,
`root_quat_wxyz`, `joint_pos`, `joint_vel`, `contact_left_foot`,
`contact_right_foot`). Non-time keys (`fps`, `joint_names`, `meta`) pass
through unchanged unless documented otherwise (`speed` scales `fps`-implied
duration, not the `fps` field itself — see its docstring).

These are consumed by `metric_validate.py`'s reference-anchored gates: a
metric must NOT rank a reversed/frozen/shuffled clip above the real
reference (hard negatives), and must rank truncations monotonically between
the degenerate anchors and the full reference (the calibration ladder).
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: Time-indexed array keys every perturbation must handle consistently.
_TIME_KEYS = (
    "root_pos_z", "root_vel_z", "root_pos_xy", "root_quat_wxyz",
    "joint_pos", "joint_vel", "contact_left_foot", "contact_right_foot",
)
#: Keys that pass through unchanged (not indexed by frame).
_PASSTHROUGH_KEYS = ("fps", "joint_names", "meta")


def _time_len(clip: dict) -> int:
    return int(np.asarray(clip["root_pos_z"]).shape[0])


def _rebuild(clip: dict, index_or_fn) -> dict:
    """Apply `index_or_fn` (an integer-array of source frame indices, OR a
    callable `(array) -> array` for keys that need custom resampling) to
    every present time-key, passing through the rest unchanged."""
    out: dict[str, Any] = {}
    for k in _PASSTHROUGH_KEYS:
        if k in clip:
            out[k] = clip[k]
    for k in _TIME_KEYS:
        v = clip.get(k)
        if v is None:
            continue
        if callable(index_or_fn):
            out[k] = index_or_fn(v)
        else:
            out[k] = np.asarray(v)[index_or_fn]
    return out


def _validate_or_raise(clip: dict) -> dict:
    from sculptor.reference import validate_clip

    errors = validate_clip(clip)
    if errors:
        raise AssertionError(
            "perturbation produced an invalid clip:\n  - " + "\n  - ".join(errors))
    return clip


# ── negatives (§5.2) ───────────────────────────────────────────────────────
def time_reverse(clip: dict) -> dict:
    """Play the clip backwards — a get-up reversed IS a fall."""
    n = _time_len(clip)
    idx = np.arange(n - 1, -1, -1)
    out = _rebuild(clip, idx)
    # Reversing velocities in TIME also flips their SIGN (d/dt of a
    # time-reversed signal is the negated, time-reversed derivative).
    for vk in ("root_vel_z", "joint_vel"):
        if vk in out:
            out[vk] = -out[vk]
    return _validate_or_raise(out)


def freeze_start(clip: dict) -> dict:
    """Hold the first frame for the full clip duration — lying still
    forever (no transition ever happens)."""
    n = _time_len(clip)
    out = _rebuild(clip, np.zeros(n, dtype=int))
    for vk in ("root_vel_z", "joint_vel"):
        if vk in out:
            out[vk] = np.zeros_like(out[vk])
    return _validate_or_raise(out)


def freeze_end(clip: dict) -> dict:
    """Hold the last frame for the full clip duration — standing still
    without ever having performed the transition."""
    n = _time_len(clip)
    out = _rebuild(clip, np.full(n, n - 1, dtype=int))
    for vk in ("root_vel_z", "joint_vel"):
        if vk in out:
            out[vk] = np.zeros_like(out[vk])
    return _validate_or_raise(out)


def complete_then_hold(clip: dict, hold_frac: float = 1.0) -> dict:
    """The FULL clip followed by `max(2, int(round(T*hold_frac)))` appended
    frames holding the LAST frame — a completed motion whose terminal state
    is then HELD (velocities zero throughout the hold; every posture/root/
    contact channel stays pinned at the clip's final values, same as
    `freeze_end` holds its frame).

    This is a POSITIVE exemplar, not a negative: the goal WAS achieved (the
    clip's transition happened in full) and the terminal state is simply
    sustained afterward — exactly the shape of a live rollout that reaches
    the goal early and holds it for the remainder of the episode (D23: the
    real `torso_righting` rollout that scored spec 0.0 despite a perfect
    sit-up looked exactly like this). A metric MUST score this AT LEAST AS
    well as the reference itself (gated in `metric_validate.py`'s
    `reference_complete_then_hold`); a metric that penalizes stillness after
    completion would wrongly punish an early, successful finish.

    This is the exact OPPOSITE of `freeze_end`, which never performs the
    transition at all (it holds the clip's LAST frame for the clip's FULL
    original duration, with no preceding motion) and stays a hard NEGATIVE —
    the two must coexist: `complete_then_hold` scores high, `freeze_end`
    scores degenerate, on the same honest metric.

    NOT added to `perturbation_suite()` (that dict is consumed as negatives
    / the calibration ladder); call this directly where a positive-hold
    exemplar is needed."""
    if hold_frac < 0.0:
        raise ValueError(f"hold_frac must be >= 0: {hold_frac}")
    n = _time_len(clip)
    hold_n = max(2, int(round(n * hold_frac)))
    idx = np.concatenate([np.arange(n), np.full(hold_n, n - 1, dtype=int)])
    out = _rebuild(clip, idx)
    for vk in ("root_vel_z", "joint_vel"):
        if vk in out:
            v = np.array(out[vk], copy=True)
            v[n:] = 0
            out[vk] = v
    return _validate_or_raise(out)


def segment_shuffle(clip: dict, n_segments: int = 6, seed: int = 0) -> dict:
    """Chop the clip into `n_segments` contiguous chunks and shuffle their
    ORDER (deterministic given `seed`) — right poses, wrong order."""
    n = _time_len(clip)
    n_segments = max(1, min(n_segments, n))
    bounds = np.linspace(0, n, n_segments + 1).astype(int)
    chunks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_segments)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_segments)
    idx = np.concatenate([chunks[i] for i in order]) if chunks else np.arange(n)
    out = _rebuild(clip, idx)
    return _validate_or_raise(out)


def speed(clip: dict, factor: float) -> dict:
    """Resample along T to play at `factor`x speed (0.25 = quarter speed /
    4x longer, 4.0 = 4x speed / quarter as long). `fps` is preserved — the
    clip's PLAYBACK RATE changes via frame count, not the fps field,
    matching how a resampled mocap clip is normally represented."""
    if factor <= 0:
        raise ValueError(f"speed factor must be > 0: {factor}")
    n = _time_len(clip)
    n_new = max(10, int(round(n / factor)))
    src_idx = np.linspace(0, n - 1, n_new)

    def _resample(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        if v.ndim == 1:
            return np.interp(src_idx, np.arange(n), v)
        flat = v.reshape(n, -1)
        cols = [np.interp(src_idx, np.arange(n), flat[:, j])
                for j in range(flat.shape[1])]
        return np.stack(cols, axis=1).reshape((n_new,) + v.shape[1:])

    out = _rebuild(clip, _resample)
    fps = float(clip["fps"])
    # Quaternions cannot be component-wise lerped raw: retargeted clips
    # carry hemisphere sign-flips (q and -q encode the same rotation), and
    # interpolating across a flip passes near zero norm — validate_clip's
    # unit-norm check then kills the ENTIRE perturbation suite (found live
    # 2026-07-09 on a fleaven clip: every truncation scored nan and no
    # metric could ever pass monotonicity). Hemisphere-align the source
    # first, lerp, then renormalize (nlerp — fine at mocap frame spacing).
    if "root_quat_wxyz" in clip:
        q = np.asarray(clip["root_quat_wxyz"], dtype=np.float64).copy()
        for i in range(1, q.shape[0]):
            if np.dot(q[i], q[i - 1]) < 0.0:
                q[i] = -q[i]
        q_new = _resample(q)
        norms = np.linalg.norm(q_new, axis=-1, keepdims=True)
        out["root_quat_wxyz"] = q_new / np.maximum(norms, 1e-9)
    # Recompute velocities from the resampled positions. The resampled
    # sequence covers the SAME displacement in n/factor frames at the same
    # fps, so gradient(pos)*fps ALREADY yields factor-scaled velocities —
    # an extra *factor here double-counts and produced factor^2 speeds
    # (audit finding 2026-07-09: speed(clip,4) gave 16.9 m/s for a 1 m/s
    # rise). gradient*fps is the whole story.
    if "root_vel_z" in out and "root_pos_z" in out:
        out["root_vel_z"] = np.gradient(out["root_pos_z"]) * fps
    if "joint_vel" in out and "joint_pos" in out:
        out["joint_vel"] = np.gradient(out["joint_pos"], axis=0) * fps
    return _validate_or_raise(out)


def truncate(clip: dict, frac: float) -> dict:
    """First `frac` fraction of frames — a partial completion."""
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1]: {frac}")
    n = _time_len(clip)
    n_keep = max(2, int(round(n * frac)))
    idx = np.arange(n_keep)
    out = _rebuild(clip, idx)
    return _validate_or_raise(out)


def degrade(
    clip: dict, *, joint_noise_rad: float = 0.0, root_damp: float = 1.0,
    seed: int = 0,
) -> dict:
    """Calibration-ladder degradation: additive gaussian joint noise
    (`joint_noise_rad` std, radians) and/or damping of root motion TOWARD
    the start pose (`root_damp` in `[0, 1]`, 1.0 = no damping, 0.0 = frozen
    at the start pose) — graded competence rungs for §6 ladders."""
    if not (0.0 <= root_damp <= 1.0):
        raise ValueError(f"root_damp must be in [0, 1]: {root_damp}")
    if joint_noise_rad < 0.0:
        raise ValueError(f"joint_noise_rad must be >= 0: {joint_noise_rad}")
    out = _rebuild(clip, lambda v: np.asarray(v).copy())
    rng = np.random.default_rng(seed)
    fps = float(clip["fps"])

    if joint_noise_rad > 0.0 and "joint_pos" in out:
        out["joint_pos"] = out["joint_pos"] + rng.normal(
            0.0, joint_noise_rad, size=out["joint_pos"].shape)
        out["joint_vel"] = np.gradient(out["joint_pos"], axis=0) * fps

    if root_damp < 1.0:
        z0 = float(out["root_pos_z"][0])
        out["root_pos_z"] = z0 + root_damp * (out["root_pos_z"] - z0)
        if "root_vel_z" in out:
            out["root_vel_z"] = np.gradient(out["root_pos_z"]) * fps
        if "root_pos_xy" in out:
            xy0 = out["root_pos_xy"][0]
            out["root_pos_xy"] = xy0 + root_damp * (out["root_pos_xy"] - xy0)

    return _validate_or_raise(out)


def root_motion_only(clip: dict) -> dict:
    """The clip's ROOT trajectory with every other channel frozen at
    frame 0 — locomotion-by-levitation. An Opus audit (2026-07-09, D18)
    PROVED a pelvis-rise-only metric passed all three reference gates:
    it rewarded base height climbing over time with zero dependence on
    posture, so a policy that scoots its pelvis upward while staying
    supine would earn full reward. This negative is the counterexample
    made flesh: right root motion, zero posture change. Any metric that
    scores it near the full reference is rewarding displacement, not the
    motion — gated in `reference_negatives`. Universal, not
    get-up-specific (a gliding statue must not pass a gait metric; a
    root-only pop must not pass a jump metric)."""
    n = _time_len(clip)
    out = _rebuild(clip, lambda v: np.repeat(
        np.asarray(v)[:1].copy(), n, axis=0))
    for key in ("root_pos_z", "root_pos_xy"):
        if key in clip:
            out[key] = np.asarray(clip[key]).copy()
    fps = float(clip["fps"])
    if "root_vel_z" in out and "root_pos_z" in out:
        out["root_vel_z"] = np.gradient(out["root_pos_z"]) * fps
    if "joint_vel" in out and "joint_pos" in out:
        out["joint_vel"] = np.zeros_like(out["joint_pos"])
    return _validate_or_raise(out)


def perturbation_suite(clip: dict) -> dict[str, dict]:
    """The standard named perturbation set consumed by
    `metric_validate.py`'s reference-anchored gates."""
    suite = {
        "reversal": time_reverse(clip),
        "freeze_start": freeze_start(clip),
        "freeze_end": freeze_end(clip),
        "shuffle": segment_shuffle(clip, n_segments=6, seed=0),
        "speed_slow": speed(clip, 0.25),
        "speed_fast": speed(clip, 4.0),
        "trunc_25": truncate(clip, 0.25),
        "trunc_50": truncate(clip, 0.50),
        "trunc_75": truncate(clip, 0.75),
    }
    # root_only (D18) exists ONLY when the clip carries posture channels
    # (joint_pos or root_quat_wxyz). For a root-channels-only clip the
    # perturbation would equal the original clip — every metric would
    # score it identically to the full reference and be convicted, which
    # is a false reject, not rigor.
    if "joint_pos" in clip or "root_quat_wxyz" in clip:
        suite["root_only"] = root_motion_only(clip)
    return suite
