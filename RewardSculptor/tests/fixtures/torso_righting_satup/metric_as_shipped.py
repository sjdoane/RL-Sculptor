import numpy as np

# We aggregate over ALL joints (no specific column is indexed), so no specific
# roles are required.  Frame checks that need unavailable signals ABSTAIN.
REQUIRED_JOINT_ROLES = []


def _roll_mean_t(x, k):
    """Edge-padded rolling mean along axis 0 for (T, ...) arrays. No zero-padding."""
    k = int(max(1, k))
    if x.shape[0] < 1:
        return x
    if k == 1:
        return x.copy()
    pad_lo = k // 2
    pad_hi = k - 1 - pad_lo
    pad_width = [(pad_lo, pad_hi)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad_width, mode="edge")
    c = np.cumsum(xp, axis=0, dtype=np.float64)
    zero = np.zeros((1,) + xp.shape[1:], dtype=np.float64)
    c = np.concatenate([zero, c], axis=0)
    out = (c[k:] - c[:-k]) / float(k)
    return out


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def compute_spec(arrays, behavior, meta):
    out = {
        "spec_score": 0.0,
        "progress_score": 0.0,
        # ---- resolved goal frame ----
        "frame_goal_axis": "world+z_root_rise",
        "frame_support_mode_abstained": 1.0,   # get-up transition: support schedule unspecified -> abstain
        "frame_torso_target": "upright_sitting(g_z_end<=-0.25)",
    }

    root = arrays.get("root_link_pos_w")
    if root is None or root.ndim != 3 or root.shape[0] < 8:
        return out
    root = np.asarray(root, dtype=np.float64)
    T, E = root.shape[0], root.shape[1]

    dt = float((behavior or {}).get("step_dt", 1.0 / 30.0))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / 30.0

    z = root[:, :, 2]                                # (T, E)
    z = np.where(np.isfinite(z), z, 0.0)

    # smoothing / window sizes (spike-robust: everything below reads window MEANS)
    k_sm = max(1, int(round(0.30 / dt)))             # ~0.3 s smoothing
    k_start = max(2, int(round(0.50 / dt)))          # first 0.5 s
    k_end = max(2, int(round(0.70 / dt)))            # last 0.7 s
    k_start = min(k_start, T)
    k_end = min(k_end, T)

    z_s = _roll_mean_t(z, k_sm)                      # (T, E) smoothed height

    z_start = np.mean(z[:k_start, :], axis=0)        # (E,)
    z_end = np.mean(z[T - k_end:, :], axis=0)        # (E,)
    z_min = np.min(z_s, axis=0)                      # (E,) smoothed minimum (no single-frame dips)
    rise = z_end - z_min                             # (E,) SIGNED rise along world +z (goal_axis)

    # ---------------- orientation (torso turning upright) ----------------
    grav = arrays.get("projected_gravity_b")
    grav_ok = grav is not None and grav.ndim == 3 and grav.shape[0] == T and grav.shape[2] >= 3
    if grav_ok:
        g_z = np.asarray(grav[:, :, 2], dtype=np.float64)
        g_z = np.where(np.isfinite(g_z), g_z, 0.0)
        g_end = np.mean(g_z[T - k_end:, :], axis=0)  # (E,) window mean, boundary-safe
    else:
        g_end = None

    # NOTE: the reference clip starts at gravity_z_b = -0.649 (NOT near 0), so a
    # "must start supine by gravity" gate would false-reject the real exemplar.
    # The started-low condition is carried by root height instead; the initial-
    # orientation gate ABSTAINS.
    out["initial_orientation_gate_abstained"] = 1.0

    # ---------------- joint articulation (anti root-only replay) ----------------
    # A rollout whose root rises while every joint is frozen is a replay/hack,
    # not a get-up: the body cannot push itself off the ground without moving
    # its joints.  Reference mean |joint_vel| during activity ~0.35-1.1 rad/s.
    jv = arrays.get("joint_vel")
    jp = arrays.get("joint_pos")
    mav = None  # per-env mean |joint_vel| over the episode
    if jv is not None and jv.ndim == 3 and jv.shape[0] == T:
        jv = np.asarray(jv, dtype=np.float64)
        jv = np.where(np.isfinite(jv), jv, 0.0)
        speed = np.mean(np.abs(jv), axis=2)          # (T, E)
        speed = _roll_mean_t(speed, k_sm)            # smooth: single-frame spikes barely count
        mav = np.mean(speed, axis=0)                 # (E,)
    elif jp is not None and jp.ndim == 3 and jp.shape[0] >= 3:
        jp = np.asarray(jp, dtype=np.float64)
        jp = np.where(np.isfinite(jp), jp, 0.0)
        djp = np.abs(jp[1:] - jp[:-1]) / dt          # finite-difference velocity
        speed = np.mean(djp, axis=2)                 # (T-1, E)
        speed = _roll_mean_t(speed, k_sm)
        mav = np.mean(speed, axis=0)                 # (E,)
    joint_abstain = mav is None
    out["joint_motion_abstained"] = 1.0 if joint_abstain else 0.0

    # ---------------- completion gate (per env, 0/1) ----------------
    # started low (lying: reference start 0.13-0.16 m)
    g_low = (z_start <= 0.25).astype(np.float64)
    # reached the goal height band and HELD it at the end (sitting/crouched >= 0.35 m)
    g_high = (z_end >= 0.35).astype(np.float64)
    # signed rise floor along +z (amplitude floor; reference rise ~0.49 m)
    g_rise = (rise >= 0.20).astype(np.float64)
    # chest turned upright at the end (reference end-window g_z ~ -0.46 .. -0.55)
    if g_end is not None:
        g_up = (g_end <= -0.25).astype(np.float64)
        up_abstain = False
    else:
        g_up = np.ones(E, dtype=np.float64)
        up_abstain = True
    out["upright_abstained"] = 1.0 if up_abstain else 0.0
    # joints actually articulated (veto root-only replay / frozen-joint hover)
    if not joint_abstain:
        g_joint = (mav >= 0.08).astype(np.float64)
    else:
        g_joint = np.ones(E, dtype=np.float64)

    gate = g_low * g_high * g_rise * g_up * g_joint  # (E,) each in {0,1}

    # ---------------- saturating channels (each [0,1], independent) ----------------
    # height held at the end: ramps 0.30 -> 0.50 m (reference end-window ~0.55)
    ch_height = np.clip((z_end - 0.30) / 0.20, 0.0, 1.0)
    # signed rise amplitude: ramps 0.15 -> 0.35 m (reference ~0.49)
    ch_rise = np.clip((rise - 0.15) / 0.20, 0.0, 1.0)
    # uprightness at the end: ramps g_z -0.25 -> -0.45 (reference ~ -0.46 => 1.0)
    if g_end is not None:
        ch_up = np.clip((-g_end - 0.25) / 0.20, 0.0, 1.0)
    else:
        ch_up = np.ones(E, dtype=np.float64)         # abstain (flagged above)
    # joint articulation: ramps 0 -> 0.30 rad/s mean (reference ~0.45)
    if not joint_abstain:
        ch_joint = np.clip(mav / 0.30, 0.0, 1.0)
    else:
        ch_joint = np.ones(E, dtype=np.float64)      # abstain (flagged above)

    ch_min = np.minimum(np.minimum(ch_height, ch_rise), np.minimum(ch_up, ch_joint))
    per_env = gate * ch_min
    spec = float(np.clip(np.mean(per_env), 0.0, 1.0))

    # ---------------- dense progress (ranking only; never feeds spec) ----------------
    # same physical quantities, smooth from the noise floor, no gate / hard floors
    d_rise = np.clip(rise / 0.35, 0.0, 1.0)                       # any rise counts
    d_height = np.clip((z_end - 0.12) / 0.38, 0.0, 1.0)           # from lying baseline
    if g_end is not None:
        d_up = np.clip((-g_end + 0.10) / 0.60, 0.0, 1.0)          # supine ~0 -> ~0.17, upright -> 1
    else:
        d_up = np.ones(E, dtype=np.float64)
    if not joint_abstain:
        d_joint = 1.0 - np.exp(-mav / 0.20)
    else:
        d_joint = np.ones(E, dtype=np.float64)
    prog = np.minimum(np.minimum(d_rise, d_height), np.minimum(d_up, d_joint))
    progress = float(np.clip(np.mean(prog), 0.0, 1.0))

    if not np.isfinite(spec):
        spec = 0.0
    if not np.isfinite(progress):
        progress = 0.0

    out.update({
        "spec_score": spec,
        "progress_score": progress,
        "completion_gate_mean": float(np.mean(gate)),
        "gate_started_low_frac": float(np.mean(g_low)),
        "gate_reached_035_frac": float(np.mean(g_high)),
        "gate_rise_floor_frac": float(np.mean(g_rise)),
        "gate_upright_frac": float(np.mean(g_up)),
        "gate_joint_motion_frac": float(np.mean(g_joint)),
        "ch_height_end": float(np.mean(ch_height)),
        "ch_rise_signed": float(np.mean(ch_rise)),
        "ch_upright_end": float(np.mean(ch_up)),
        "ch_joint_motion": float(np.mean(ch_joint)),
        "z_start_mean": float(np.mean(z_start)),
        "z_end_mean": float(np.mean(z_end)),
        "z_min_mean": float(np.mean(z_min)),
        "rise_mean": float(np.mean(rise)),
        "gravity_z_end_mean": float(np.mean(g_end)) if g_end is not None else 0.0,
        "mean_abs_joint_vel": float(np.mean(mav)) if mav is not None else 0.0,
    })
    return out
