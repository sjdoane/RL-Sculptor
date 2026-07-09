"""Reference trajectories for hard-exploration skills.

§RESEARCH_GAP_ANALYSIS §4.5/§7.6: every published G1 jump rides a
reference motion (ASAP, the tracking line), and the reference-free
existence proofs (Humanoid Parkour arXiv:2406.10759, HoST
arXiv:2502.08378) all needed explicit curriculum machinery — the reward
function alone cannot supply the states PPO never visits. DeepMimic
(arXiv:1804.02717) showed the cheapest transfer of a reference is
*Reference State Initialization*: start a fraction of episodes INSIDE
the hard-to-reach phase so the policy experiences apex → descent →
landing long before it can produce a launch.

This module makes that machinery available WITHOUT touching evaluation:

  * a small, validated clip format (root height + vertical velocity over
    time, optional proxy joint channels) — `save_clip`/`load_clip`;
  * `make_procedural_jump_clip` — an analytic crouch→extend→flight→
    land→recover profile when no mocap is available (fully offline);
  * `phase_keyframes` — a compact quantitative summary (crouch depth,
    takeoff velocity, apex, flight time) fit for diagnoser/editor
    prompts, replacing guessed thresholds with measured ones;
  * `derive_rsi_train_keys` — maps the clip onto the EXISTING env-spec
    TRAIN-scope surface (`reset_height_offset_m`,
    `reset_vertical_velocity_mps`, paired
    `min_base_height_termination_m`), clamped to the validator's own
    bounds tables so the emitted values can never fail the gate;
  * `apply_reference_rsi` — persists those values as the next validated
    env/v<N>.json via `env_spec.write_env_spec_version` (train-only by
    construction: rollout evaluation NEVER sees RSI — the schema-level
    shared/train split guarantees metric comparability).

Real retargeted mocap: Unitree publishes a LAFAN1 dataset retargeted to
G1/H1 (HuggingFace `unitreerobotics/LAFAN1_Retargeting_Dataset`,
auth-gated — `huggingface-cli login` then download the `g1/` jump
clips), and ASAP (LeCAR-Lab/ASAP) ships retargeted G1 jump motions. A
converted clip only needs `root_pos_z` (+`fps`) to drive everything
here; richer joint channels are carried through untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sculptor.env_spec import (
    ENV_SPEC_VERSION,
    _TRAIN_RANGES,
    _TRAIN_SCALARS,
    read_current_env_spec,
    write_env_spec_version,
)

_G = 9.81
# Sunk-height termination as a fraction of standing height. Measured on
# the tuck-jump E2E runs (G1 stand ≈ 0.78 m): 0.5 m (≈0.64·stand) works;
# 0.6 m killed training outright (iter 32, mean return −7.7). RSI without
# this termination REGRESSES (iters 19-20) — the validator enforces the
# pairing, and `derive_rsi_train_keys` always emits both.
_SUNK_FRAC_OF_STAND = 0.64


# ── Clip format ─────────────────────────────────────────────────────────
def validate_clip(clip: dict) -> list[str]:
    """All violations at once (mirrors `validate_env_spec` style)."""
    errors: list[str] = []
    z = clip.get("root_pos_z")
    if not isinstance(z, np.ndarray) or z.ndim != 1 or z.shape[0] < 10:
        errors.append("root_pos_z must be a 1-D array with >= 10 frames")
        return errors
    if not np.isfinite(z).all():
        errors.append("root_pos_z contains non-finite values")
    if (z <= 0).any():
        errors.append("root_pos_z must be strictly positive (metres)")
    fps = clip.get("fps")
    try:
        fps = float(fps)
        if not (1.0 <= fps <= 240.0):
            errors.append(f"fps must be in [1, 240], got {fps}")
    except (TypeError, ValueError):
        errors.append(f"fps must be numeric, got {fps!r}")
    vz = clip.get("root_vel_z")
    if vz is not None and (
            not isinstance(vz, np.ndarray) or vz.shape != z.shape
            or not np.isfinite(vz).all()):
        errors.append("root_vel_z must match root_pos_z shape and be finite")
    jp = clip.get("joint_pos")
    if jp is not None:
        if (not isinstance(jp, np.ndarray) or jp.ndim != 2
                or jp.shape[0] != z.shape[0] or not np.isfinite(jp).all()):
            errors.append("joint_pos must be finite with shape [T, J]")
        names = clip.get("joint_names") or []
        if jp.ndim == 2 and len(names) != jp.shape[1]:
            errors.append("joint_names length must equal joint_pos J")
    # ── R1 optional keys (schema-reserved / library ingest) ─────────────
    xy = clip.get("root_pos_xy")
    if xy is not None and (
            not isinstance(xy, np.ndarray) or xy.shape != (z.shape[0], 2)
            or not np.isfinite(xy).all()):
        errors.append("root_pos_xy must be finite with shape [T, 2]")
    quat = clip.get("root_quat_wxyz")
    if quat is not None:
        if (not isinstance(quat, np.ndarray) or quat.shape != (z.shape[0], 4)
                or not np.isfinite(quat).all()):
            errors.append("root_quat_wxyz must be finite with shape [T, 4]")
        else:
            norms = np.linalg.norm(quat, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                errors.append(
                    "root_quat_wxyz rows must be unit-norm (within 1e-3)")
    jv = clip.get("joint_vel")
    if jv is not None:
        if (not isinstance(jv, np.ndarray) or jv.ndim != 2
                or jv.shape[0] != z.shape[0] or not np.isfinite(jv).all()):
            errors.append("joint_vel must be finite with shape [T, J]")
        elif jp is not None and isinstance(jp, np.ndarray) and jv.shape != jp.shape:
            errors.append("joint_vel shape must match joint_pos shape [T, J]")
    for key in ("contact_left_foot", "contact_right_foot"):
        c = clip.get(key)
        if c is not None and (
                not isinstance(c, np.ndarray) or c.shape != (z.shape[0],)
                or not np.isfinite(c).all()):
            errors.append(f"{key} must be finite with shape [T]")
    return errors


def _with_velocity(clip: dict) -> dict:
    """Return the clip with `root_vel_z` present (finite-difference when
    the source didn't carry velocities), and likewise `joint_vel` backfilled
    by finite difference when `joint_pos` is present but `joint_vel` isn't."""
    if clip.get("root_vel_z") is None:
        clip = dict(clip)
        clip["root_vel_z"] = np.gradient(
            clip["root_pos_z"]) * float(clip["fps"])
    if clip.get("joint_pos") is not None and clip.get("joint_vel") is None:
        clip = dict(clip)
        clip["joint_vel"] = np.gradient(
            clip["joint_pos"], axis=0) * float(clip["fps"])
    return clip


def save_clip(path: Path | str, clip: dict) -> Path:
    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            "refusing to persist invalid reference clip:\n  - "
            + "\n  - ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "root_pos_z": clip["root_pos_z"].astype(np.float32),
        "fps": np.float32(clip["fps"]),
        "meta_json": np.bytes_(json.dumps(clip.get("meta") or {}).encode()),
    }
    for opt in (
            "root_vel_z", "joint_pos", "root_pos_xy", "root_quat_wxyz",
            "joint_vel", "contact_left_foot", "contact_right_foot"):
        if clip.get(opt) is not None:
            payload[opt] = clip[opt].astype(np.float32)
    if clip.get("joint_names"):
        payload["joint_names"] = np.array(
            [str(n) for n in clip["joint_names"]])
    np.savez_compressed(path, **payload)
    return path


def load_clip(path: Path | str) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        clip: dict[str, Any] = {
            "root_pos_z": z["root_pos_z"].astype(np.float64),
            "fps": float(z["fps"]),
        }
        if "root_vel_z" in z.files:
            clip["root_vel_z"] = z["root_vel_z"].astype(np.float64)
        if "joint_pos" in z.files:
            clip["joint_pos"] = z["joint_pos"].astype(np.float64)
        for opt in (
                "root_pos_xy", "root_quat_wxyz", "joint_vel",
                "contact_left_foot", "contact_right_foot"):
            if opt in z.files:
                clip[opt] = z[opt].astype(np.float64)
        if "joint_names" in z.files:
            clip["joint_names"] = [str(n) for n in z["joint_names"]]
        if "meta_json" in z.files:
            try:
                clip["meta"] = json.loads(bytes(z["meta_json"]).decode())
            except Exception:  # noqa: BLE001 — meta is advisory
                clip["meta"] = {}
    errors = validate_clip(clip)
    if errors:
        raise ValueError(
            f"invalid reference clip at {path}:\n  - " + "\n  - ".join(errors))
    return _with_velocity(clip)


# ── Procedural jump ─────────────────────────────────────────────────────
def make_procedural_jump_clip(
    *,
    stand_height_m: float = 0.78,
    crouch_frac: float = 0.62,
    apex_gain_m: float = 0.35,
    tuck_rad: float = 0.7,
    fps: float = 50.0,
) -> dict:
    """Analytic crouch→extend→flight→land→recover root-height profile.

    Defaults are Unitree-G1-shaped (stand 0.78 m, the audit's measured
    crouch ≈ 0.45-0.5 m, apex gains in the 0.1-0.5 m band the E2E runs
    produced). This is NOT a dynamically-consistent motion — it is a
    kinematic reference whose value is (a) RSI state ranges and (b)
    quantitative phase targets for prompts; both are robust to the
    approximation. Joint channels are proxy flexion profiles (named
    `*_proxy`), carried for completeness, never for tracking rewards.
    """
    if not (0.2 < crouch_frac < 1.0):
        raise ValueError(f"crouch_frac must be in (0.2, 1.0): {crouch_frac}")
    if not (0.02 <= apex_gain_m <= 1.0):
        raise ValueError(f"apex_gain_m must be in [0.02, 1.0]: {apex_gain_m}")
    z0 = float(stand_height_m)
    zc = crouch_frac * z0
    vz_takeoff = math.sqrt(2.0 * _G * apex_gain_m)
    t_flight = 2.0 * vz_takeoff / _G

    def seg(duration: float, fn) -> np.ndarray:
        n = max(2, int(round(duration * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    smooth = lambda s: (1 - np.cos(np.pi * s)) / 2.0          # noqa: E731
    stand = seg(0.30, lambda s: np.full_like(s, z0))
    crouch = seg(0.35, lambda s: z0 + (zc - z0) * smooth(s))
    # Extension is constant-acceleration from rest at the crouch to
    # exactly `vz_takeoff` at liftoff: displacement d = v·t/2 fixes the
    # duration, so the velocity profile is continuous into the ballistic
    # flight (no seam spike in the finite-difference vz).
    t_extend = 2.0 * (z0 - zc) / vz_takeoff
    extend = seg(t_extend, lambda s: zc + (z0 - zc) * s**2)
    flight = seg(t_flight, lambda s: z0
                 + vz_takeoff * (s * t_flight)
                 - 0.5 * _G * (s * t_flight) ** 2)
    dip = 0.85 * z0
    land = seg(0.15, lambda s: z0 + (dip - z0) * smooth(s))
    recover = seg(0.40, lambda s: dip + (z0 - dip) * smooth(s))
    z = np.concatenate([stand, crouch, extend, flight, land, recover])
    vz = np.gradient(z) * fps

    # Proxy flexion: ground-phase crouch depth + a flight-phase tuck bump.
    flex_ground = np.clip((z0 - z) / max(z0 - zc, 1e-6), 0.0, 1.0)
    tuck = np.zeros_like(z)
    f0 = stand.size + crouch.size + extend.size
    f1 = f0 + flight.size
    tuck[f0:f1] = np.sin(np.linspace(0.0, np.pi, flight.size)) * tuck_rad
    joint_pos = np.stack(
        [flex_ground * 0.9 + tuck, flex_ground * 1.2 + tuck * 0.6], axis=1)

    clip = {
        "root_pos_z": z,
        "root_vel_z": vz,
        "joint_pos": joint_pos,
        "joint_names": ["knee_flexion_proxy", "hip_flexion_proxy"],
        "fps": float(fps),
        "meta": {
            "source": "procedural:jump",
            "stand_height_m": z0,
            "crouch_frac": crouch_frac,
            "apex_gain_m": apex_gain_m,
            "takeoff_vz_mps": vz_takeoff,
            "flight_time_s": t_flight,
        },
    }
    errors = validate_clip(clip)
    if errors:  # pragma: no cover — generator invariant
        raise AssertionError("; ".join(errors))
    return clip


# ── Analysis ────────────────────────────────────────────────────────────
def phase_keyframes(clip: dict, n: int = 8) -> dict:
    """Quantitative phase summary for prompts: measured targets instead
    of guessed thresholds (the audit's 0.12 m-launch/0.55 rad-tuck class
    of numbers, but derived from an actual motion)."""
    clip = _with_velocity(clip)
    z = clip["root_pos_z"]
    vz = clip["root_vel_z"]
    fps = float(clip["fps"])
    z0 = float(np.median(z[: max(2, int(0.2 * fps))]))
    apex_i = int(np.argmax(z))
    crouch_i = int(np.argmin(z[:apex_i])) if apex_i > 0 else 0
    airborne = z > z0 + 0.01
    t_flight = float(airborne.sum()) / fps
    idx = np.linspace(0, z.size - 1, max(2, n)).astype(int)
    return {
        "stand_height_m": round(z0, 4),
        "crouch_depth_m": round(z0 - float(z[crouch_i]), 4),
        "apex_gain_m": round(float(z[apex_i]) - z0, 4),
        "takeoff_vz_mps": round(float(vz.max()), 4),
        "landing_vz_mps": round(float(vz.min()), 4),
        "flight_time_s": round(t_flight, 4),
        "duration_s": round(z.size / fps, 4),
        "keyframes": [
            {"t": round(int(i) / fps, 3),
             "z": round(float(z[i]), 4),
             "vz": round(float(vz[i]), 4)}
            for i in idx
        ],
    }


def _clamp_range(key: str, lo: float, hi: float) -> list[float]:
    b_lo, b_hi = _TRAIN_RANGES[key]
    lo = min(max(lo, b_lo), b_hi)
    hi = min(max(hi, b_lo), b_hi)
    if hi < lo:
        lo, hi = hi, lo
    return [round(lo, 4), round(hi, 4)]


def derive_rsi_train_keys(clip: dict) -> dict:
    """Map a reference clip onto the env-spec TRAIN surface.

    Emits DeepMimic-style RSI ranges covering the clip's airborne +
    landing states (offsets are relative to the robot's default reset,
    i.e. to standing — mjlab `reset_root_state_uniform` semantics), and
    ALWAYS pairs them with the sunk-height termination the validator's
    RSI↔ET invariant requires. Values are clamped into the validator's
    own bounds tables, so the emitted fragment can never fail the gate.
    """
    clip = _with_velocity(clip)
    z = clip["root_pos_z"]
    vz = clip["root_vel_z"]
    fps = float(clip["fps"])
    z0 = float(np.median(z[: max(2, int(0.2 * fps))]))
    window = z > z0 + 0.01                     # airborne frames
    if not window.any():
        raise ValueError(
            "reference clip never rises above its standing height — "
            "no airborne states to initialize from")
    height_hi = float((z[window] - z0).max())
    vz_lo = float(vz[window].min())
    vz_hi = float(vz[window].max())
    sunk_lo, sunk_hi = _TRAIN_SCALARS["min_base_height_termination_m"]
    sunk = round(min(max(_SUNK_FRAC_OF_STAND * z0, sunk_lo), sunk_hi), 2)
    return {
        "reset_height_offset_m": _clamp_range(
            "reset_height_offset_m", 0.0, height_hi),
        "reset_vertical_velocity_mps": _clamp_range(
            "reset_vertical_velocity_mps", vz_lo, vz_hi),
        "min_base_height_termination_m": sunk,
    }


def apply_reference_rsi(env_dir: Path | str, clip: dict) -> Path:
    """Persist the clip-derived RSI curriculum as the next validated
    env-spec version (train scope only; the frozen shared/eval section
    is untouched, so metric comparability is preserved by construction).
    Builds on the project's current spec when one exists."""
    env_dir = Path(env_dir)
    spec = read_current_env_spec(env_dir) or {
        "env_spec_version": ENV_SPEC_VERSION,
        "meta": {},
        "shared": {},
        "train": {},
    }
    spec = json.loads(json.dumps(spec))        # deep copy
    train = dict(spec.get("train") or {})
    derived = derive_rsi_train_keys(clip)
    train.update(derived)
    spec["train"] = train
    meta = dict(spec.get("meta") or {})
    src = (clip.get("meta") or {}).get("source", "clip")
    meta["source"] = f"reference:{src}"
    meta["rationale"] = (
        "RSI ranges derived from a reference trajectory "
        f"({src}): airborne heights up to "
        f"+{derived['reset_height_offset_m'][1]} m, vz "
        f"{derived['reset_vertical_velocity_mps']} m/s, paired sunk "
        f"termination {derived['min_base_height_termination_m']} m "
        "(DeepMimic RSI; validator RSI↔ET invariant)."
    )
    spec["meta"] = meta
    return write_env_spec_version(env_dir, spec)
