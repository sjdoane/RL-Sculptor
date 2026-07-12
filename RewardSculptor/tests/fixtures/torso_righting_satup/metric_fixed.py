"""
Objective task-success metric: supine -> upright-torso floor sit for Unitree G1.

Goal frame resolution:
  * goal_axis    = None      (no horizontal travel direction is part of the goal;
                              the signed-direction channel therefore ABSTAINS)
  * support_mode = None      (floor-sitting: neither a foot-stance nor flight
                              schedule defines this skill; support check ABSTAINS)
  * torso_target = upright   (chest rotates upright: projected_gravity_b z must
                              go from the lying start (~ -0.28 at the eval reset,
                              ~ -0.65 at the reference clip start) toward the
                              reference terminal value of ~ -0.856)

Success signature (grounded in the reference clip + eval start state):
  * starts NOT upright         : first frames g_z > -0.72   (eval reset ~ -0.28,
                                 reference clip frame-0     ~ -0.649)
  * ends torso-upright         : end-window mean g_z < -0.76 (reference ~ -0.84..-0.856)
  * uprighting arc (amplitude) : g_z drop >= 0.10            (reference ~ 0.19-0.21)
  * pelvis stays near ground   : end root z < 0.35 m and smoothed max root z
                                 < 0.55 m for the whole episode (reference
                                 root z stays in 0.098-0.164 m; a crouch/stand
                                 belongs to LATER stages and is vetoed)
  * no travel hack             : horizontal root displacement < 1.5 m

Composition (hard rule 1): spec_score = completion_gate * min(channels).
All window reads are relative (earliest frames / last ~0.5 s) so a policy that
finishes the transition 16x faster than the reference and then HOLDS the sit
scores identically to the reference.
"""

import numpy as np


def _smooth_time(x, k):
    """Edge-padded moving average along axis 0. x: (T, E). Never zero-pads."""
    t = x.shape[0]
    if k <= 1 or t < 2:
        return x
    k = int(min(k, t))
    pad = np.pad(x, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    c = np.cumsum(
        np.concatenate([np.zeros((1, x.shape[1]), dtype=x.dtype), pad], axis=0),
        axis=0,
    )
    return (c[k:] - c[:-k]) / float(k)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_spec(arrays, behavior, meta):
    out = {
        "spec_score": 0.0,
        "progress_score": 0.0,
        "goal_axis": "none_abstained",
        "support_mode": "none_abstained",
        "torso_target": "upright",
        "direction_check_abstained": 1.0,
        "support_check_abstained": 1.0,
    }

    grav = arrays.get("projected_gravity_b")
    root = arrays.get("root_link_pos_w")
    if grav is None or root is None:
        out["missing_inputs"] = 1.0
        return out
    grav = np.asarray(grav, dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    if grav.ndim != 3 or root.ndim != 3 or grav.shape[0] < 5 or root.shape[0] < 5:
        out["short_or_malformed"] = 1.0
        return out
    if not (np.all(np.isfinite(grav)) and np.all(np.isfinite(root))):
        out["nonfinite_inputs"] = 1.0
        return out

    T = min(grav.shape[0], root.shape[0])
    E = min(grav.shape[1], root.shape[1])
    g = grav[:T, :E, 2]          # gravity z in body frame: -1 upright, ~0 lying
    z = root[:T, :E, 2]          # root height
    xy = root[:T, :E, :2]        # horizontal root position

    bh = behavior or {}
    dt = float(bh.get("step_dt") or 0.02)
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.02

    # spike-robust smoothing (~0.1 s window)
    k = max(1, int(round(0.1 / dt)))
    gs = _smooth_time(g, k)
    zs = _smooth_time(z, k)

    # ---- window reads (relative time; earliest-frames start per FAST-COMPLETION rule)
    n0 = min(3, T)                                   # earliest frames only
    n_end = min(T, max(3, int(round(0.5 / dt))))     # last ~0.5 s

    g_start = np.mean(g[:n0], axis=0)                # (E,)
    g_end = np.mean(gs[-n_end:], axis=0)
    z_end = np.mean(zs[-n_end:], axis=0)
    z_max = np.max(zs, axis=0)
    hold_frac = np.mean(gs[-n_end:] < -0.72, axis=0)
    delta_g = g_start - g_end                        # >0 means torso became MORE upright

    xy_disp = np.linalg.norm(
        np.mean(xy[-n0:], axis=0) - np.mean(xy[:n0], axis=0), axis=-1
    )

    # ---- completion gate (binary, owns the floor) ----------------------------
    started_away = g_start > -0.72        # eval reset ~ -0.28; reference clip start ~ -0.649
    ended_upright = g_end < -0.76         # reference terminal ~ -0.84 .. -0.856
    pelvis_low_end = z_end < 0.35         # reference end root z ~ 0.153 m
    amplitude_ok = delta_g >= 0.10        # reference arc ~ 0.19-0.21
    no_travel = xy_disp < 1.5             # sit-ups do not walk away
    never_rose = z_max < 0.55             # crouch/stand belongs to later stages

    gate = (
        started_away
        & ended_upright
        & pelvis_low_end
        & amplitude_ok
        & no_travel
        & never_rose
    ).astype(np.float64)

    # ---- saturating channels in [0,1] (min composition) ----------------------
    # torso_target = upright -> monotone in end-uprightness (reference ~ -0.856 -> 1.0)
    c_upright = _clip01((-g_end - 0.70) / 0.10)
    # pelvis stays near ground (reference 0.153 m -> 1.0; crouch ~0.4 -> 0)
    c_pelvis = _clip01((0.35 - z_end) / 0.10)
    # uprighting arc amplitude floor (reference ~0.19 -> 1.0; micro-twitch -> 0)
    c_amp = _clip01((delta_g - 0.08) / 0.10)
    # sustained hold of the upright-torso sit over the end window
    c_hold = _clip01((hold_frac - 0.6) / 0.3)
    # never rose toward crouch/stand anywhere in the episode (reference max 0.164 m)
    c_lowmax = _clip01((0.55 - z_max) / 0.15)

    channels = np.stack([c_upright, c_pelvis, c_amp, c_hold, c_lowmax], axis=0)
    per_env = gate * np.min(channels, axis=0)
    spec = float(np.clip(np.mean(per_env), 0.0, 1.0))

    # ---- dense progress (ranking only; NEVER feeds spec_score) ---------------
    # same physical quantities, ramping from the sensor-noise floor.
    p_upright = _clip01((-g_end - 0.30) / 0.50)   # eval-reset dead-still (~-0.28) -> ~0
    p_delta = _clip01((delta_g - 0.02) / 0.16)    # reference ~0.19 -> 1; frozen -> 0
    p_pelvis = _clip01((0.55 - z_end) / 0.20)     # standing up ranks down
    progress = float(
        np.clip(np.mean(np.min(np.stack([p_upright, p_delta, p_pelvis], axis=0), axis=0)), 0.0, 1.0)
    )

    out.update(
        {
            "spec_score": spec,
            "progress_score": progress,
            "completion_gate": float(np.mean(gate)),
            "gate_started_away": float(np.mean(started_away)),
            "gate_ended_upright": float(np.mean(ended_upright)),
            "gate_pelvis_low_end": float(np.mean(pelvis_low_end)),
            "gate_amplitude": float(np.mean(amplitude_ok)),
            "gate_no_travel": float(np.mean(no_travel)),
            "gate_never_rose": float(np.mean(never_rose)),
            "chan_upright_end": float(np.mean(c_upright)),
            "chan_pelvis_low": float(np.mean(c_pelvis)),
            "chan_amplitude": float(np.mean(c_amp)),
            "chan_hold": float(np.mean(c_hold)),
            "chan_height_cap": float(np.mean(c_lowmax)),
            "mean_g_start": float(np.mean(g_start)),
            "mean_g_end": float(np.mean(g_end)),
            "mean_delta_g": float(np.mean(delta_g)),
            "mean_z_end": float(np.mean(z_end)),
            "mean_z_max": float(np.mean(z_max)),
            "mean_xy_disp": float(np.mean(xy_disp)),
        }
    )
    return out
