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
  * `derive_reference_reset` — `derive_rsi_train_keys` plus root-
    orientation (`reset_pitch_offset_rad`/`reset_roll_offset_rad`) and
    per-joint posture (`reset_joint_pos_target`/
    `reset_joint_pos_noise_rad`) ranges when the clip carries
    `root_quat_wxyz`/`joint_pos` and is get-up-shaped;
  * `apply_reference_rsi` — persists EVERY key `derive_reference_reset`
    returns as the next validated env/v<N>.json via
    `env_spec.write_env_spec_version` (train-only by construction:
    rollout evaluation NEVER sees RSI — the schema-level shared/train
    split guarantees metric comparability).

§REFERENCE_TRAJECTORY_PLAN §8 (2026-07-09): generalized beyond the
procedural jump clip to GET-UP-shaped motions (rises from a low/lying
start to standing). `derive_rsi_train_keys`/`derive_reference_reset`
classify a clip's ARCHETYPE from its own height trajectory
(`_archetype`: "getup" / "airborne" / "other") and dispatch accordingly:
airborne (jump-like) derivation is byte-identical to the original
jump-only implementation; get-up derivation emits a NEGATIVE
`reset_height_offset_m` (lying is below the robot's standing default)
and a sunk-termination threshold placed BELOW the clip's own minimum
observed height (rather than the airborne path's fraction-of-standing
rule) so early termination can never fire on the reference's own lying
reset.

§8 part 2 (2026-07-09, second increment): BOTH capabilities that were
previously computed-but-not-persistable are now fully wired end to end:
  * root orientation — `env_spec.py`'s `_TRAIN_RANGES` now carries
    `reset_pitch_offset_rad`/`reset_roll_offset_rad` (matching
    `env_gen.py::_TrainModel` fields keep the drift guard green), and
    `sculptor/adapters/_mjlab_runner.py::_apply_env_spec` writes them
    into the reset event's `pose_range["pitch"/"roll"]` — the mjlab
    MECHANISM (`reset_root_state_uniform`) already natively supported
    this;
  * per-joint lying POSTURE — mjlab's shipped `reset_joints_by_offset`
    genuinely has no per-joint target mechanism (single scalar range
    applied uniformly to every joint from the STANDING default), so
    `_mjlab_runner.py` now injects a NEW event term
    (`reset_joints_to_reference`, mirroring the shipped function's own
    clamp/write contract) driven by the new `train.reset_joint_pos_target`
    (list[float], clip joint order) + `train.reset_joint_pos_noise_rad`
    schema keys. mjlab's `events`/`terminations` cfg fields are plain
    dicts the adapter is already free to add entries to (see the
    pre-existing `sunk` termination injection) — no mjlab fork needed.
    Joint-count mismatches between the persisted target and the live
    robot are caught with a clear `ValueError` at cfg-apply time
    (`sculptor.eval.robot_manifest.robot_joint_names`), never silently
    misassigned.
See `derive_reference_reset`'s docstring for the full derivation detail
and `_mjlab_runner.reset_joints_to_reference`'s docstring for the event
mechanism.

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
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sculptor.env_spec import (
    _JOINT_TARGET_ELEMENT_BOUNDS,
    _JOINT_TARGET_MAX_LEN,
    _TRAIN_RANGES,
    _TRAIN_SCALARS,
    ENV_SPEC_VERSION,
    read_current_env_spec,
    write_env_spec_version,
)

_G = 9.81
# Sunk-height termination as a fraction of standing height. Measured on
# the tuck-jump E2E runs (G1 stand ≈ 0.78 m): 0.5 m (≈0.64·stand) works;
# 0.6 m killed training outright (iter 32, mean return −7.7). RSI without
# this termination REGRESSES (iters 19-20) — the validator enforces the
# pairing, and `derive_rsi_train_keys` always emits both (airborne
# clips). Get-up clips use a DIFFERENT guard — see
# `_GETUP_SUNK_MARGIN_M` below.
_SUNK_FRAC_OF_STAND = 0.64

# ── §REFERENCE_TRAJECTORY_PLAN §8: archetype thresholds ────────────────
# A clip's initial window (first `_ARCHETYPE_WINDOW_S` seconds) is
# compared against these to classify the motion. "Low" means the root is
# on/near the ground (lying); "high" means upright/standing. Values are
# G1-shaped (stand ≈0.78 m) but expressed as robot-relative *fractions
# are not available at this layer — reference.py never sees stand
# height directly, so we use absolute metre thresholds consistent with
# typical humanoid geometry (a lying human/humanoid root is well under
# 0.35 m; a standing one is well over 0.6 m for G1-class robots).
_ARCHETYPE_WINDOW_S = 0.5
_GETUP_START_MAX_M = 0.35
_GETUP_END_MIN_M = 0.55  # aligned with refs/segment.py QC_END_MIN_MEAN_Z (D15)
# Get-up sunk-termination guard: the derived threshold sits this far
# BELOW the clip's own observed minimum height, so the reference's own
# lying start (and any natural settling below it) never trips early
# termination at reset. §8: "sunk-height termination must NOT fire at
# reset."
_GETUP_SUNK_MARGIN_M = 0.05
# D19: absolute anchor for low-start reset offsets. Clips are same-robot
# retargets, so their metre heights transfer directly; the env-spec
# offset is relative to the task's default (standing) reset, for which
# this G1-class constant is the module's working assumption (same basis
# as the archetype thresholds). _MIN_RESET_Z_M floors the requested
# reset height: ground-clamped source data (root z = 0.00 while lying)
# must not put the pelvis inside the floor at reset.
_G1_CLASS_STAND_M = 0.74
_MIN_RESET_Z_M = 0.10
# §D21 Fix 3: public alias — the mechanical start-state gate in
# sculpt.py (`_evaluate_start_state_gate`) needs this SAME anchor to
# recompute an eval_reset.json's expected absolute frame-0 root z
# (`G1_CLASS_STAND_M + reset_height_offset_m`), a module boundary away.
# Kept as a plain re-export (not a rename) so every existing in-module
# reference to `_G1_CLASS_STAND_M` stays untouched.
G1_CLASS_STAND_M = _G1_CLASS_STAND_M


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
    # Round-then-RE-CLAMP (fourth occurrence of the rounding class,
    # 2026-07-13): a PRONE clip's derived roll offset is exactly pi, and
    # round(pi, 4) = 3.1416 > pi — the persisted value then failed the
    # env-spec validator's own [-pi, pi] hard bound and killed the first
    # prone mission at scaffold (reference_scaffold_failed). Readability
    # rounding must never push a value back OUT of the bound it was just
    # clamped into.
    b_lo, b_hi = _TRAIN_RANGES[key]
    lo = min(max(lo, b_lo), b_hi)
    hi = min(max(hi, b_lo), b_hi)
    if hi < lo:
        lo, hi = hi, lo
    lo = min(max(round(lo, 4), b_lo), b_hi)
    hi = min(max(round(hi, 4), b_lo), b_hi)
    return [lo, hi]


def _clamp_scalar(key: str, v: float) -> float:
    lo, hi = _TRAIN_SCALARS[key]
    # Same round-then-re-clamp as _clamp_range (round(pi, 4) > pi).
    return min(max(round(min(max(v, lo), hi), 4), lo), hi)


_DATASET_MID_START_TOKENS = frozenset({
    "crouch", "crouched", "crouching", "kneel", "kneeling", "sit",
    "sits", "sitting", "squat", "squatting",
})
_DATASET_AIRBORNE_TOKENS = frozenset({
    "bound", "bounding", "hop", "hops", "hopping", "jump", "jumps",
    "jumping", "leap", "leaps", "leaping", "parkour", "vault",
    "vaulting",
})
_ORIGIN_RELATIVE_UPRIGHT_GZ_MAX = -0.75


def _dataset_motion_tokens(clip: dict) -> set[str]:
    """Normalized advisory motion words for converted dataset clips.

    The ingester persists ``meta.source == "dataset"`` and its filename
    tokens inside ``clip.npz``.  These words may refine an otherwise
    representation-ambiguous *upright* clip, but never override measured
    orientation by themselves.
    """
    meta = clip.get("meta")
    if not isinstance(meta, dict) or meta.get("source") != "dataset":
        return set()
    raw: list[str] = []
    for key in ("tokens", "text", "motion", "name"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)):
            raw.extend(str(item) for item in value)
        elif value is not None:
            raw.append(str(value))
    return set(re.findall(r"[a-z]+", " ".join(raw).lower()))


def _origin_relative_upright_dataset_clip(clip: dict) -> bool:
    """Whether a low numeric root-z is an origin-relative upright pose.

    Large retargeted-motion corpora commonly zero root translation while
    preserving the body's world orientation.  Absolute height thresholds
    therefore cannot distinguish their standing walk/hop clips from a true
    lying start.  Require all three independent signals before interpreting
    the height as relative: explicit dataset metadata, a numerically low
    start, and measured start-window gravity that is strongly upright.

    Missing/corrupt orientation returns ``False`` and preserves the historical
    fail-closed absolute-height classification.  No robot or task identifier
    participates in this representation check.
    """
    if not _dataset_motion_tokens(clip):
        return False
    z = np.asarray(clip.get("root_pos_z"), dtype=np.float64)
    quat = clip.get("root_quat_wxyz")
    if z.ndim != 1 or z.size < 2 or quat is None:
        return False
    fps = float(clip["fps"])
    nw = max(2, int(_ARCHETYPE_WINDOW_S * fps), int(0.1 * len(z)))
    if float(np.mean(z[:nw])) >= _GETUP_START_MAX_M:
        return False
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] < nw:
        return False
    q = q[:nw]
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norms > 1e-9)
    if not valid.any():
        return False
    q = q[valid] / norms[valid, None]
    g = _quat_wxyz_to_gravity_b(q).mean(axis=0)
    g_norm = float(np.linalg.norm(g))
    return bool(
        g_norm > 1e-9
        and float(g[2] / g_norm) <= _ORIGIN_RELATIVE_UPRIGHT_GZ_MAX
    )


def _archetype(clip: dict) -> str:
    """Classify a reference clip's shape (§8).

    Checked in this order:

    1. ``"getup"`` — the clip STARTS low (near-ground root height).
       D19: the classification is START-STATE ONLY. It originally also
       required a standing END, which misrouted a real lie-to-crouch
       reference (z 0.00→0.28) to the airborne branch mid-mission: the
       stage got jump-style RSI, fell_over stayed armed, and NO eval
       reset was derived — eval rollouts reset standing, so the stage's
       certified started-low metric could never score above zero.
       Reset derivation only cares where the motion BEGINS: every
       low-start clip (get-up, lie-to-crouch, crawl, lying-hold) wants
       the same low reset semantics — initial-window pose, orientation,
       posture target, fell_over disabled, sunk guard below the clip's
       own minimum. A real jump clip starts STANDING, so the start
       condition alone still cleanly separates airborne clips.
    2. ``"mid_start"`` — the clip starts at a MID height (neither lying
       nor clearly upright): `_GETUP_START_MAX_M` (0.35 m) <= start <
       `_GETUP_END_MIN_M` (0.55 m, the SAME "clearly upright" line the
       get-up segment QC uses for its END threshold — reused
       deliberately rather than a third independent constant, per the
       one-classifier-per-shape-assumption meta-rule below). §start_pose
       CLOSES the D19-documented gap: a crouch-to-stand reference (or
       any stage whose `start_pose` is "sitting"/"crouched") starts here
       — not lying flat, but not upright either. Handled identically to
       `"getup"` by `derive_rsi_train_keys`/`derive_reference_reset`
       (same absolute-anchored offset math — see D19's note there — and
       the same sunk-guard/orientation/posture derivation), EXCEPT
       `fell_over_termination` stays disabled for the SAME conservative
       reason get-up gets it (D16: even a bent crouch pose can trip
       orientation-based fall termination at reset) and
       `derive_eval_reset` still returns a non-None override (D17: a
       non-standing start IS the task, whether it's flat-on-the-ground
       or mid-crouch).
    3. ``"airborne"`` — starts high (>= `_GETUP_END_MIN_M`) and rises
       above its own EARLY-window baseline (the pre-existing jump-clip
       rule, byte-identical for clips that reach this branch).
    4. ``"other"`` — starts high but never rises. (mid_start above closes
       the D19-documented "crouch-to-stand starts at z≈0.4" gap; there is
       no more OPEN gap in this classifier as of §start_pose.)

    Meta-rule (thrice-earned, D19): every assumption about clip SHAPE
    must live in exactly one classifier with QC enforcement — see
    `check_start_pose_compatibility`, the QC gate that gets a
    stage-authored `start_pose` to agree with what THIS function
    measures.
    """
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    # Start window: the LARGER of 0.5 s and 10% of the clip, MEAN
    # aggregation — aligned with refs/segment.py's per-segment QC windows
    # (a 0.5 s end-MEDIAN once misclassified a QC-passing segment, D15).
    nw = max(2, int(_ARCHETYPE_WINDOW_S * fps), int(0.1 * len(z)))
    start = float(np.mean(z[:nw]))
    if _origin_relative_upright_dataset_clip(clip):
        # Translation is relative, so use measured orientation plus the
        # dataset's advisory motion words.  Ordinary locomotion remains
        # ``other`` even when gait bounce exceeds the old 1 cm jump threshold;
        # crouch/sit and airborne intent require explicit matching words.
        tokens = _dataset_motion_tokens(clip)
        if tokens & _DATASET_MID_START_TOKENS:
            return "mid_start"
        n0 = max(2, int(0.2 * fps))
        z0 = float(np.median(z[:n0]))
        if tokens & _DATASET_AIRBORNE_TOKENS and (z > z0 + 0.01).any():
            return "airborne"
        return "other"
    if start < _GETUP_START_MAX_M:
        return "getup"
    if start < _GETUP_END_MIN_M:
        return "mid_start"
    n0 = max(2, int(0.2 * fps))
    z0 = float(np.median(z[:n0]))
    if (z > z0 + 0.01).any():
        return "airborne"
    return "other"


def derive_rsi_train_keys(clip: dict) -> dict:
    """Map a reference clip onto the env-spec TRAIN surface.

    Dispatches on the clip's shape (§REFERENCE_TRAJECTORY_PLAN §8):

    - **Airborne** (jump-like — rises above its own standing baseline):
      UNCHANGED, byte-identical behavior to the original jump-only
      implementation. Emits DeepMimic-style RSI ranges covering the
      clip's airborne + landing states (offsets relative to the robot's
      default reset, i.e. to standing — mjlab
      `reset_root_state_uniform` semantics), ALWAYS paired with the
      sunk-height termination the validator's RSI↔ET invariant
      requires.
    - **Get-up-like** (starts low, ends standing) OR **mid_start**
      (starts at a mid height — neither lying nor upright, e.g. a
      crouch-to-stand reference, or any "sitting"/"crouched"
      `start_pose` stage's attached clip): emits a NEGATIVE
      `reset_height_offset_m` derived from the clip's own initial-window
      height (anchored on the ABSOLUTE G1-class standing height, D19 —
      same math for both archetypes, they only differ in how far below
      standing the start window sits), near-zero
      `reset_vertical_velocity_mps` (a resting start, not a falling
      one), and a sunk-termination threshold derived BELOW the clip's
      own observed minimum height (never above it) so the reference's
      own start cannot trip early termination at reset — the
      start-pose-aware guard §8 requires.
    - **Other** (never rises, never low→high): unchanged — raises
      `ValueError` with an actionable message.

    Values are clamped into the validator's own bounds tables, so the
    emitted fragment can never fail the gate.
    """
    clip = _with_velocity(clip)
    z = clip["root_pos_z"]
    vz = clip["root_vel_z"]
    fps = float(clip["fps"])
    archetype = _archetype(clip)

    if archetype in ("getup", "mid_start"):
        nw = max(2, int(_ARCHETYPE_WINDOW_S * fps))
        z_start_lo = float(z[:nw].min())
        z_start_hi = float(z[:nw].max())
        vz_start = vz[:nw]
        z_min = float(z.min())
        sunk_lo, sunk_hi = _TRAIN_SCALARS["min_base_height_termination_m"]
        sunk = _clamp_scalar(
            "min_base_height_termination_m",
            min(max(z_min - _GETUP_SUNK_MARGIN_M, sunk_lo), sunk_hi))
        # D19 follow-through: the offset anchor was `z_stand = clip's own
        # END-window median` — valid ONLY while every low-start clip
        # ended standing. A real lie-to-crouch reference (end 0.22 m)
        # made the "offset" -0.22, i.e. a reset at ~0.56 m: a CROUCH,
        # not the clip's lying start. Clips in this library are
        # same-robot retargets, so absolute heights transfer: anchor on
        # the G1-class standing height constant (same geometry basis as
        # the archetype thresholds above), with a physical floor so
        # ground-clamped source data (z=0.00 lying) can't request a
        # reset inside the floor.
        lo = max(z_start_lo - _G1_CLASS_STAND_M,
                 _MIN_RESET_Z_M - _G1_CLASS_STAND_M)
        hi = max(z_start_hi - _G1_CLASS_STAND_M,
                 _MIN_RESET_Z_M - _G1_CLASS_STAND_M)
        return {
            "reset_height_offset_m": _clamp_range(
                "reset_height_offset_m", lo, hi),
            "reset_vertical_velocity_mps": _clamp_range(
                "reset_vertical_velocity_mps",
                float(vz_start.min()), float(vz_start.max())),
            "min_base_height_termination_m": sunk,
        }

    if archetype == "other":
        raise ValueError(
            "reference clip never rises above its standing height and "
            "never transitions low→high — no RSI states to initialize "
            "from (not a recognized jump- or get-up-shaped motion)")

    # archetype == "airborne" — original jump-only derivation, unchanged.
    z0 = float(np.median(z[: max(2, int(0.2 * fps))]))
    window = z > z0 + 0.01                     # airborne frames
    if not window.any():
        # Unreachable in practice — `_archetype` already classified this
        # clip as "airborne" using the identical rule, so `window` can't
        # be empty here. Kept as defense-in-depth (mirrors the original
        # jump-only guard byte-for-byte) rather than trusting the
        # dispatch alone.
        raise ValueError(
            "reference clip never rises above its standing height — "
            "no airborne states to initialize from")
    height_hi = float((z[window] - z0).max())
    vz_lo = float(vz[window].min())
    vz_hi = float(vz[window].max())
    sunk_lo, sunk_hi = _TRAIN_SCALARS["min_base_height_termination_m"]
    # Origin-relative retargets correctly express the jump's height DELTA but
    # not an absolute standing height.  Use the same physical default-height
    # anchor as reset derivation for the termination threshold; otherwise z0~0
    # clamps the guard near the floor and silently removes its protective value.
    sunk_anchor = (
        _G1_CLASS_STAND_M
        if _origin_relative_upright_dataset_clip(clip)
        else z0
    )
    sunk = round(
        min(max(_SUNK_FRAC_OF_STAND * sunk_anchor, sunk_lo), sunk_hi), 2)
    return {
        "reset_height_offset_m": _clamp_range(
            "reset_height_offset_m", 0.0, height_hi),
        "reset_vertical_velocity_mps": _clamp_range(
            "reset_vertical_velocity_mps", vz_lo, vz_hi),
        "min_base_height_termination_m": sunk,
    }


def _quat_wxyz_to_pitch_rad(q: np.ndarray) -> np.ndarray:
    """Standard aerospace-sequence pitch angle from a batch of unit
    quaternions `(N, 4)` in `[w, x, y, z]` order. Pitch alone is the
    axis that carries "lying face-down/face-up vs. standing upright"
    for a humanoid falling/rising in the sagittal plane."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    return np.arcsin(sinp)


def _quat_wxyz_to_roll_rad(q: np.ndarray) -> np.ndarray:
    """Standard aerospace-sequence roll angle from a batch of unit
    quaternions `(N, 4)` in `[w, x, y, z]` order — the axis that carries
    "lying on the left/right side vs. standing upright," complementary
    to pitch's "face-up/face-down." A get-up clip may need either or
    both depending on how the motion begins."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return np.arctan2(sinr_cosp, cosr_cosp)


# ── §D29-1: analytic pitch/roll derivation FROM body-frame gravity ─────
# D29 root cause: the prior derivation extracted RAW per-axis Euler
# angles at the clip's start window AND its own end/"standing" window,
# then SUBTRACTED them (`offset = start - stand`) to get a delta applied
# at reset. That subtraction silently assumed the clip's own end window
# equals the ROBOT's actual default (identity) orientation — untrue for
# real mocap (residual yaw/lean/noise) — and Euler-angle subtraction is
# not equivalent to relative-rotation composition at large angles /
# branch boundaries. A true-prone clip (start body-frame gravity
# (0.96, -0.12, 0.24)) derived (pitch=-0.05, roll=pi), whose body gravity
# is (0, 0, +1) — UPSIDE-DOWN (angle error 1.38 rad from the measured
# vector) — the robot spawned interpenetrating the floor and
# contact-exploded to z 2.34 m.
#
# The fix derives directly from MEASURED body-frame gravity instead:
# `reset_root_state_uniform` (mjlab `envs/mdp/events.py`, confirmed by
# reading the installed package) composes
# `orientations = quat_mul(default_quat, quat_from_euler_xyz(roll, pitch,
# yaw=0))`, and the G1 pelvis carries no MJCF `quat` attribute — i.e.
# `default_quat` is IDENTITY (already established by `_quat_from_pitch_
# roll`'s docstring/tests below). With an identity default, world gravity
# `g_world=(0,0,-1)` rotated into the body frame by the resulting
# orientation is, for `yaw=0` (yaw is unobservable from gravity — a
# rotation about world +z leaves `(0,0,-1)` unchanged before the
# body-frame projection, so it is legitimately dropped, never guessed):
#     gx = sin(pitch)
#     gy = -sin(roll) * cos(pitch)
#     gz = -cos(roll) * cos(pitch)
# which inverts in closed form (pitch in [-pi/2, pi/2], cos(pitch) >= 0
# so the shared positive scale factor drops out of atan2):
#     pitch = arcsin(gx)
#     roll  = atan2(-gy, -gz)
# Numerically verified against mjlab's OWN `quat_mul`/`quat_from_euler_
# xyz`/`quat_apply_inverse` (the exact functions `reset_root_state_
# uniform` and `EntityData.projected_gravity_b` use) over 2000 random
# (pitch, roll) pairs: max reconstruction error ~5e-7 (float32-scale
# torch noise). On the live D29 gravity vector this closed form derives
# pitch=1.2982 rad, roll=2.6779 rad, reconstructing (0.9631, -0.1204,
# 0.2408) — the measured vector exactly (angle error 0.0), not upside
# down. `_quat_wxyz_to_gravity_b`'s gx/gy component formulas are
# algebraically identical to `_quat_wxyz_to_pitch_rad`/`_quat_wxyz_to_
# roll_rad` above (gravity is exactly what those two Euler angles
# encode) — the bug was never the per-axis trig, only the cross-window
# subtraction this replaces.
_ORIENTATION_GATE_MAX_RAD = 0.35


def _quat_wxyz_to_gravity_b(q: np.ndarray) -> np.ndarray:
    """Body-frame gravity DIRECTION for a batch of unit quaternions
    `(..., 4)` in `[w, x, y, z]` order — world gravity `(0, 0, -1)`
    rotated into the body frame (`R(q)^T @ (0, 0, -1)`), matching
    mjlab's `EntityData.projected_gravity_b` convention exactly
    (`quat_apply_inverse(root_link_quat_w, gravity_vec_w)` with
    `gravity_vec_w = (0, 0, -1)` — confirmed by reading `mjlab/entity/
    entity.py` and `mjlab/entity/data.py`). Returns `(..., 3)`, matching
    `q`'s leading shape. Pure function of `q` — no default-orientation
    assumption is baked in here (that only enters via `_gravity_to_
    pitch_roll`'s use at yaw=0 for the RESET composition, below)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    gx = 2.0 * (w * y - x * z)
    gy = -2.0 * (y * z + w * x)
    gz = 2.0 * (x * x + y * y) - 1.0
    return np.stack([gx, gy, gz], axis=-1)


def _gravity_to_pitch_roll(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the reset composition's forward direction (`_quat_
    from_pitch_roll`, defined below): the `(pitch, roll)` pair (yaw
    dropped — unobservable from gravity) whose `_quat_from_pitch_roll`
    quaternion reproduces body-frame gravity `g` (shape `(..., 3)`,
    NOT required to be exactly unit-norm — only its direction is used).
    Gimbal-locked at `|gx| == 1` (pitch = +-pi/2): `roll` is genuinely
    unobservable there (both `gy` and `gz` collapse to 0 regardless of
    roll) and `np.arctan2(0, 0)` returns `0.0`, a harmless, self-
    consistent choice (the self-consistency gate below still passes,
    since roll=0 reconstructs the SAME degenerate gravity)."""
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    pitch = np.arcsin(np.clip(gx, -1.0, 1.0))
    roll = np.arctan2(-gy, -gz)
    return pitch, roll


def derive_reference_reset(clip: dict) -> dict:
    """`derive_rsi_train_keys` plus, when the clip carries
    `root_quat_wxyz` AND is get-up-shaped (getup OR mid_start — §D19
    closure), ADDITIONAL `reset_pitch_offset_rad` / `reset_roll_offset_rad`
    ranges derived ANALYTICALLY from the clip's initial-window MEASURED
    body-frame gravity (§D29-1 — see `_quat_wxyz_to_gravity_b`/
    `_gravity_to_pitch_roll`'s module-level comment for the full closed-
    form derivation and its mjlab-composition proof; NOT relative to the
    clip's own end window any more — that cross-window subtraction was
    the D29 upside-down-prone-reset root cause), gated by a mechanical
    self-consistency check (reconstruct the reset orientation from the
    derived midpoint and require its body gravity to land back within
    `_ORIENTATION_GATE_MAX_RAD` of the measured vector — raises
    `ValueError` naming both vectors on failure, fail-closed rather than
    scaffold a wrong pose), PLUS (when the clip carries `joint_pos`) a
    `reset_joint_pos_target` vector — the root-
    orientation and per-joint-posture halves of a low/mid start
    (§REFERENCE_TRAJECTORY_PLAN §8's "PLUS ... initial joint posture and
    root orientation" extension, now fully wired through
    `env_spec.py`'s `_TRAIN_RANGES`/`_TRAIN_SCALARS` and
    `sculptor/adapters/_mjlab_runner.py`'s `reset_joints_to_reference`
    event — see that module for the mjlab event-injection mechanism).

    `reset_joint_pos_target` is the clip's OWN initial-window mean
    `joint_pos` (the clip's lying/crouched posture), in the clip's
    `joint_names` order — the adapter resolves/validates that order
    against the robot's canonical joint order at apply time (a mismatch
    is a clear error there, never a silent misassignment here; this
    function has no robot context to check against). Clamped to
    `env_spec`'s own `_JOINT_TARGET_ELEMENT_BOUNDS`/`_JOINT_TARGET_MAX_LEN`
    so the emitted vector can never fail `validate_env_spec`. Absent
    when the clip carries no `joint_pos` channel (procedural clips, or
    real mocap without a retargeted joint trace) — the adapter's
    existing scalar `reset_joint_position_offset_rad` remains the
    fallback in that case.
    """
    derived = dict(derive_rsi_train_keys(clip))
    is_low_start = _archetype(clip) in ("getup", "mid_start")
    if is_low_start:
        # §get-up RSI fix (2026-07-09): a lying/crouched start (large
        # reset-height offset below standing, and/or a large pitch/roll
        # offset when orientation is derived below) IS exactly what the
        # task's standard fell-over/bad-orientation termination is
        # designed to catch — observed live, it fires on every env at
        # reset and the episode never runs. The sunk-height guard above
        # (derived BELOW the clip's own minimum height) plus episode
        # time_out remain the episode enders; a get-up/crouch policy
        # falling again mid-episode after standing up is legitimate
        # retry experience, not a failure state that needs early
        # termination. mid_start gets the SAME conservative treatment
        # (D16 lesson): even a bent crouch pose can trip orientation-
        # based fall termination at reset.
        derived["fell_over_termination"] = False
    quat = clip.get("root_quat_wxyz")
    if quat is not None and is_low_start:
        fps = float(clip["fps"])
        nw = max(2, int(_ARCHETYPE_WINDOW_S * fps))
        quat_arr = np.asarray(quat, dtype=np.float64)
        start_quat = quat_arr[:nw]
        # §D29-1: per-frame MEASURED body-frame gravity over the start
        # window — no cross-window subtraction, no "standing" reference
        # window at all (the reset's actual base is the ROBOT's own
        # identity default, not this clip's own end pose).
        g_frames = _quat_wxyz_to_gravity_b(start_quat)   # (nw, 3)
        g_mean = g_frames.mean(axis=0)
        g_mean_norm = float(np.linalg.norm(g_mean))
        if g_mean_norm < 1e-6:
            raise ValueError(
                "reference clip start window produced a degenerate "
                "(near-zero-norm) mean body-frame gravity vector — "
                "cannot derive a reset orientation from it (orientation "
                "frames may be corrupt or cancel out over the window)"
            )
        g_mean_unit = g_mean / g_mean_norm
        pitch_mid, roll_mid = _gravity_to_pitch_roll(g_mean_unit)
        pitch_mid = float(pitch_mid)
        roll_mid = float(roll_mid)

        # §D29-1 THE GATE: reconstruct the reset orientation from the
        # derived midpoints using the EXACT composition `reset_root_
        # state_uniform` applies (`_quat_from_pitch_roll` — proven
        # against mjlab's own quat_mul/quat_from_euler_xyz below), then
        # require its body gravity to land back within
        # `_ORIENTATION_GATE_MAX_RAD` of the clip's measured gravity.
        # The live D29 upside-down derivation ((pitch=-0.05, roll=pi) vs
        # measured (0.96,-0.12,0.24)) fails this by ~1.38 rad — fails
        # the stage CLOSED (via the caller's reference_scaffold_failed
        # path) rather than spawn a wrong pose again.
        recon_quat = _quat_from_pitch_roll(pitch_mid, roll_mid)
        recon_g = _quat_wxyz_to_gravity_b(recon_quat[None, :])[0]
        recon_norm = float(np.linalg.norm(recon_g))
        cos_ang = float(np.clip(
            float(np.dot(recon_g, g_mean_unit)) / max(recon_norm, 1e-12),
            -1.0, 1.0))
        gate_angle = math.acos(cos_ang)
        if gate_angle > _ORIENTATION_GATE_MAX_RAD:
            raise ValueError(
                "reference orientation derivation failed its self-"
                "consistency gate: reconstructed body-frame gravity "
                f"{tuple(round(float(v), 4) for v in recon_g)} from "
                f"derived pitch={pitch_mid:.4f} rad roll={roll_mid:.4f} "
                f"rad is {gate_angle:.3f} rad from the clip's own "
                f"measured start-window gravity "
                f"{tuple(round(float(v), 4) for v in g_mean_unit)} "
                f"(gate: {_ORIENTATION_GATE_MAX_RAD} rad) — refusing to "
                f"scaffold a reset that would reproduce the wrong pose "
                f"(§D29: an upside-down prone reset from exactly this "
                f"failure class caused a floor-interpenetration contact "
                f"explosion)."
            )

        # Range: the SAME analytic derivation applied per-frame across
        # the start window, min/max'd — structurally identical to the
        # prior per-axis-then-minmax range derivation, just correctly
        # targeted (no subtraction against a clip-relative "standing"
        # window).
        pitch_frames, roll_frames = _gravity_to_pitch_roll(g_frames)
        for key, frames in (
                ("reset_pitch_offset_rad", pitch_frames),
                ("reset_roll_offset_rad", roll_frames)):
            # Round-then-RE-CLAMP: a prone clip's roll offset can land
            # exactly at pi and round(pi, 4) = 3.1416 > pi — this exact
            # class produced the live reference_scaffold_failed (env-spec
            # hard-bound reject) on the first prone mission, 2026-07-13.
            derived[key] = _clamp_range(
                key, float(frames.min()), float(frames.max()))
    joint_pos = clip.get("joint_pos")
    if joint_pos is not None and is_low_start:
        fps = float(clip["fps"])
        nw = max(2, int(_ARCHETYPE_WINDOW_S * fps))
        jp_arr = np.asarray(joint_pos, dtype=np.float64)
        target = np.median(jp_arr[:nw], axis=0)
        n = min(target.shape[0], _JOINT_TARGET_MAX_LEN)
        lo, hi = _JOINT_TARGET_ELEMENT_BOUNDS
        clamped = np.clip(target[:n], lo, hi)
        # Same round-then-re-clamp as the orientation ranges above.
        derived["reset_joint_pos_target"] = [
            min(max(round(float(x), 4), lo), hi) for x in clamped]
        # A conservative fixed noise magnitude — enough per-episode
        # variation to avoid the policy overfitting to one exact pose,
        # small enough to stay a recognizable get-up posture (mirrors
        # the scale of `reset_joint_position_offset_rad`'s own typical
        # tuning, not clip-derived — the clip's own joint variance
        # within its initial window is a plausible alternative but
        # would need a second empirical validation pass; deferred).
        derived["reset_joint_pos_noise_rad"] = 0.05
    return derived


def derive_eval_reset(clip: dict) -> dict | None:
    """Stage-FIXED, deterministic eval-rollout reset override for
    get-up/mid_start-archetype clips (§REFERENCE_TRAJECTORY_PLAN §8, D17;
    mid_start closes the D19-documented gap — a non-standing start is
    the task whether it's flat-on-the-ground or mid-crouch).

    `derive_reference_reset` (above) produces TRAIN-only RSI *ranges* —
    curriculum the diagnoser may iterate between iterations, applied via
    `_apply_env_spec(..., train=True, ...)` and deliberately NEVER seen
    by rollout evaluation (`_cmd_rollout` calls `_apply_env_spec(...,
    train=False, ...)` on purpose — diagnoser-iterable train knobs must
    not leak into eval or per-iteration fitness becomes incomparable
    across iterations).

    For a get-up/mid_start stage this creates a real gap: eval rollouts
    fall back to the task's default (standing) reset, so the certified
    lying/crouched-start metric ends up scoring rollouts that never
    actually start there — the reset the metric was written to measure
    recovery FROM never happens in evaluation. For such a task the
    non-standing start IS the task definition, not curriculum, so it
    belongs in eval too — but as a single FIXED reset, decoupled from
    the diagnoser-iterable train ranges, so per-iteration fitness stays
    comparable (nothing here is something the diagnoser can move).

    Returns a deterministic payload — the MIDPOINT of each derived reset
    range (height offset, pitch, roll), zero vertical velocity (a
    resting start, not a falling one), the clip's own
    `reset_joint_pos_target` unchanged when present, zero joint-pos
    noise (no per-episode variation — eval must be reproducible), and
    `fell_over_termination: False` (the low/mid start would otherwise
    trip it, exactly as it does in train). `None` for non-low-start
    archetypes (airborne/other) — jump and other eval stays
    standing-start, unchanged behavior; a jump task's evaluation start
    was never in question.
    """
    if _archetype(clip) not in ("getup", "mid_start"):
        return None
    full = derive_reference_reset(clip)

    def _mid(key: str) -> float:
        lo, hi = full[key]
        # Round-then-re-clamp into the SOURCE range: the midpoint of a
        # range whose ends sit at a hard bound (prone roll: [pi, pi])
        # rounds to 3.1416 > pi — same class as the scaffold failure.
        mid = round((float(lo) + float(hi)) / 2.0, 4)
        return min(max(mid, float(lo)), float(hi))

    payload: dict[str, Any] = {
        "reset_height_offset_m": _mid("reset_height_offset_m"),
        "reset_vertical_velocity_mps": 0.0,
        "fell_over_termination": False,
    }
    if "reset_pitch_offset_rad" in full:
        payload["reset_pitch_offset_rad"] = _mid("reset_pitch_offset_rad")
    if "reset_roll_offset_rad" in full:
        payload["reset_roll_offset_rad"] = _mid("reset_roll_offset_rad")
    if "reset_joint_pos_target" in full:
        payload["reset_joint_pos_target"] = list(full["reset_joint_pos_target"])
        payload["reset_joint_pos_noise_rad"] = 0.0
    return payload


# ── §start_pose: clip-shape ↔ authored-start_pose consistency QC ────────
# D19's meta-rule, thrice-earned: every assumption about clip SHAPE must
# be enforced by exactly one classifier, or the next mismatch silently
# breaks a consumer. `start_pose` is a NEW field authored by Claude
# (decompose/redecompose) that makes an assumption about clip shape —
# this is that classifier's enforcement point.
_SUPINE_PRONE_POSES: frozenset[str] = frozenset({"supine", "prone"})
_SIT_CROUCH_POSES: frozenset[str] = frozenset({"sitting", "crouched"})


def _start_window_quat(clip: dict) -> Optional[np.ndarray]:
    """Approximate mean orientation over the clip's start window, as a
    unit `[w, x, y, z]` quaternion, or `None` when the clip carries no
    `root_quat_wxyz`. Component-wise mean + renormalize — a cheap
    approximation that is only ever used to classify a START POSE from a
    SHORT (`_ARCHETYPE_WINDOW_S`), largely-static window (the robot is
    lying/sitting/crouched at rest, not mid-rotation), so it is not the
    general-purpose quaternion-averaging problem (which needs an
    eigenvector solve for well-separated rotations)."""
    quat = clip.get("root_quat_wxyz")
    if quat is None:
        return None
    fps = float(clip["fps"])
    nw = max(2, int(_ARCHETYPE_WINDOW_S * fps))
    q = np.asarray(quat[:nw], dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return None
    return q / norm


def _body_frame_gravity_x(q: np.ndarray) -> float:
    """Horizontal (local +X / "forward") component of gravity expressed
    in the body frame, for a single unit quaternion `(4,)` in
    `[w, x, y, z]` order.

    Standing (identity quat): gravity is purely local -Z (down), so this
    is ~0. Convention used throughout this codebase and the G1 MJCF
    (`sculptor/refs/preview.py`'s `resolve_g1_mjcf`: the pelvis body
    carries no `quat` attribute, i.e. its LOCAL frame equals WORLD at
    the identity/standing pose) is local +X = forward/face direction.

    A humanoid pitched flat onto its FRONT (chest/face toward the
    ground — PRONE) has its local +X axis tipped toward world -Z, which
    rotates world gravity `(0, 0, -1)` into a POSITIVE local-X
    component; pitched flat onto its BACK (SUPINE) is the mirror image
    (NEGATIVE local-X component). Hand-verified numerically: a pure
    pitch=+pi/2 quat (which rotates local +X to world -Z, i.e. face-down
    = prone) yields gravity_body_x=+1.0; pitch=-pi/2 (local +X to world
    +Z, face-up = supine) yields gravity_body_x=-1.0. See
    `test_body_frame_gravity_x_prone_vs_supine` for the pinned check.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return 0.0
    w, x, y, z = (q / norm).tolist()
    qc = np.array([w, -x, -y, -z])
    g_world = np.array([0.0, 0.0, 0.0, -1.0])   # pure quat: world gravity dir

    def _ham(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    g_body = _ham(_ham(qc, g_world), np.array([w, x, y, z]))
    return float(g_body[1])


def check_start_pose_compatibility(
    clip: dict, start_pose: Optional[str], *, clip_id: str = "<clip>",
) -> None:
    """QC gate: does `clip`'s MEASURED start-state shape agree with a
    stage-authored `start_pose`?

    Rules (archetype from `_archetype`; mid_start closes the D19 gap):
      * `start_pose` in {"supine", "prone"} requires archetype ==
        "getup" — a lying-flat pose is a LOW start, not a mid-height
        crouch/sit.
      * `start_pose` in {"sitting", "crouched"} requires archetype in
        {"getup", "mid_start"} — real clips vary in exactly how low a
        "sit"/"crouch" reads at frame 0, so either low-start class is
        plausible.
      * `start_pose == "standing"` requires archetype NOT IN
        {"getup", "mid_start"} — a standing stage must not be pointed at
        a clip that measurably starts low or mid-height.
      * `start_pose is None`: no-op — nothing authored to check.

    When the clip carries `root_quat_wxyz` AND `start_pose` is "supine"
    or "prone", the start-window MEAN orientation is ALSO checked
    against the requested pose (on-back vs on-front) via the sign of
    `_body_frame_gravity_x` — see that function's docstring for the
    derivation/sign convention. A clip with no `root_quat_wxyz` accepts
    either supine or prone (orientation is simply unknown; the archetype
    check above still applies) — same "accept when we can't tell"
    posture the rest of this module takes (e.g. `derive_reference_reset`
    only derives orientation when a quat is present at all).

    Raises `ValueError` with an ACTIONABLE message (states the measured
    mismatch AND both plausible causes — wrong clip vs wrong
    `start_pose`) on any violation. Raises `ValueError` for an
    unrecognized `start_pose` string too (defense in depth — callers
    should already be passing a `mission.START_POSE_VALUES` member).
    Never raises for `start_pose=None`.
    """
    if start_pose is None:
        return
    archetype = _archetype(clip)
    z = clip["root_pos_z"]
    fps = float(clip["fps"])
    nw = max(2, int(_ARCHETYPE_WINDOW_S * fps), int(0.1 * len(z)))
    start_z = float(np.mean(z[:nw]))

    if start_pose in _SUPINE_PRONE_POSES:
        if archetype != "getup":
            raise ValueError(
                f"stage start_pose={start_pose!r} but clip {clip_id!r} "
                f"does not measure as a lying start (archetype="
                f"{archetype!r}, start-window mean z={start_z:.3f} m; a "
                f"lying start needs < {_GETUP_START_MAX_M} m): wrong "
                f"clip attached or wrong start_pose."
            )
        q = _start_window_quat(clip)
        if q is not None:
            gx = _body_frame_gravity_x(q)
            measured = "prone" if gx > 0.0 else (
                "supine" if gx < 0.0 else None)
            if measured is not None and measured != start_pose:
                raise ValueError(
                    f"stage start_pose={start_pose!r} but clip "
                    f"{clip_id!r}'s start-window orientation measures as "
                    f"{measured!r} (body-frame gravity x={gx:.3f}): "
                    f"wrong clip attached or wrong start_pose."
                )
    elif start_pose in _SIT_CROUCH_POSES:
        if archetype not in ("getup", "mid_start"):
            raise ValueError(
                f"stage start_pose={start_pose!r} but clip {clip_id!r} "
                f"measures archetype={archetype!r} (start-window mean "
                f"z={start_z:.3f} m) — a sitting/crouched start needs a "
                f"low or mid-height clip start (< {_GETUP_END_MIN_M} m): "
                f"wrong clip attached or wrong start_pose."
            )
    elif start_pose == "standing":
        if archetype in ("getup", "mid_start"):
            raise ValueError(
                f"stage start_pose='standing' but clip {clip_id!r} "
                f"measures a lying/crouched start (archetype="
                f"{archetype!r}, start-window mean z={start_z:.3f} m): "
                f"wrong clip attached or wrong start_pose."
            )
    else:
        raise ValueError(
            f"start_pose={start_pose!r} is not a recognized value "
            f"(expected one of 'supine', 'prone', 'sitting', 'crouched', "
            f"'standing', or None)"
        )


def _recenter_train_ranges_on_settled(full: dict, settled: dict) -> dict:
    """§D20a settle-then-rederive: re-center TRAIN-side `[lo, hi]` ranges
    on SETTLED eval scalars, keeping each range's ORIGINAL WIDTH (never
    widened/narrowed here — settling only tells us where physical rest
    actually is, not how much curriculum spread the stage should have),
    then re-clamp through the SAME validator bounds tables every other
    derived range goes through. `reset_joint_pos_target` has no width
    concept (a point target, not a range) — the settled joint angles
    simply REPLACE the derived ones, re-clamped to the same element
    bounds. Any key present in `full` but absent from `settled` (e.g.
    `settled` came from a clip with no `joint_pos`/`root_quat_wxyz`) is
    left untouched."""
    out = dict(full)
    for key in ("reset_height_offset_m", "reset_pitch_offset_rad",
                "reset_roll_offset_rad"):
        if key not in out or key not in settled:
            continue
        lo, hi = out[key]
        width = float(hi) - float(lo)
        center = float(settled[key])
        out[key] = _clamp_range(key, center - width / 2.0, center + width / 2.0)
    if "reset_joint_pos_target" in out and "reset_joint_pos_target" in settled:
        lo, hi = _JOINT_TARGET_ELEMENT_BOUNDS
        out["reset_joint_pos_target"] = [
            round(min(max(float(x), lo), hi), 4)
            for x in settled["reset_joint_pos_target"]]
    return out


def apply_reference_rsi(
    env_dir: Path | str, clip: dict, *, settled_centers: Optional[dict] = None,
) -> Path:
    """Persist the clip-derived RSI curriculum as the next validated
    env-spec version (train scope only; the frozen shared/eval section
    is untouched, so metric comparability is preserved by construction).
    Builds on the project's current spec when one exists. Uses
    `derive_reference_reset` for the derivation and now persists EVERY
    key it returns — `reset_pitch_offset_rad`/`reset_roll_offset_rad`/
    `reset_joint_pos_target`/`reset_joint_pos_noise_rad` are all live
    `env_spec.py` schema keys as of §REFERENCE_TRAJECTORY_PLAN §8 part
    2 (previously computed but not persistable; see git history for the
    prior fence-limited workaround, now removed).

    `settled_centers` (§D20a, optional): the `scalars` dict a
    `settle_reset` call returned for this SAME clip's derived EVAL
    reset. When provided, the derived TRAIN ranges are re-centered onto
    it via `_recenter_train_ranges_on_settled` before being persisted —
    see that function's docstring. `None` (default) is byte-identical to
    pre-§D20a behavior."""
    env_dir = Path(env_dir)
    spec = read_current_env_spec(env_dir) or {
        "env_spec_version": ENV_SPEC_VERSION,
        "meta": {},
        "shared": {},
        "train": {},
    }
    spec = json.loads(json.dumps(spec))        # deep copy
    train = dict(spec.get("train") or {})
    full = derive_reference_reset(clip)
    if settled_centers:
        full = _recenter_train_ranges_on_settled(full, settled_centers)
    train.update(full)
    spec["train"] = train
    meta = dict(spec.get("meta") or {})
    src = (clip.get("meta") or {}).get("source", "clip")
    meta["source"] = f"reference:{src}"
    orient_note = ""
    if "reset_pitch_offset_rad" in full or "reset_roll_offset_rad" in full:
        orient_note = (
            f", orientation offsets pitch="
            f"{full.get('reset_pitch_offset_rad')} rad roll="
            f"{full.get('reset_roll_offset_rad')} rad"
        )
    posture_note = ""
    if "reset_joint_pos_target" in full:
        posture_note = (
            f", per-joint reference posture "
            f"({len(full['reset_joint_pos_target'])} joints, noise "
            f"{full.get('reset_joint_pos_noise_rad')} rad)"
        )
    fell_over_note = ""
    if full.get("fell_over_termination") is False:
        fell_over_note = (
            ", fell_over termination disabled (lying start would "
            "otherwise trip it at reset)"
        )
    settle_note = ""
    if settled_centers:
        settle_note = (
            f", TRAIN ranges re-centered on a physically-settled pose "
            f"(§D20a settle-then-rederive; settled height offset "
            f"{settled_centers.get('reset_height_offset_m')} m)"
        )
    meta["rationale"] = (
        "RSI ranges derived from a reference trajectory "
        f"({src}): height offset "
        f"{full['reset_height_offset_m']} m, vz "
        f"{full['reset_vertical_velocity_mps']} m/s, paired sunk "
        f"termination {full['min_base_height_termination_m']} m"
        f"{orient_note}{posture_note}{fell_over_note}{settle_note} "
        "(DeepMimic RSI; validator RSI↔ET invariant)."
    )
    spec["meta"] = meta
    return write_env_spec_version(env_dir, spec)


# ── §D20a: settle-then-rederive (physically-resting reset poses) ────────
class SettleUnavailable(Exception):
    """Physical settling could not run in this environment: MJCF
    unresolvable, `mujoco` import/model-load failure, or a physics step
    itself raised. Callers (the `sculpt.py` scaffold path) catch this and
    proceed with the UNSETTLED reset scalars — settling is a REFINEMENT
    of a derived reset, never a stage-blocking dependency (mirrors
    `sculptor.refs.preview.PreviewUnavailable`'s "never blocks the
    caller" contract)."""


class SettleExplosion(SettleUnavailable):
    """§D29-2: the SPECIFIC settle-failure subclass that means the
    DERIVED POSE ITSELF is physically invalid (a contact-force explosion
    from joint/orientation floor-interpenetration, caught by the
    post-settle plausibility-bound check below) — as opposed to mere
    settle-infrastructure unavailability (MJCF/model/mujoco-import
    failure, a generic `mj_step` exception) or ordinary non-convergence
    (`converged: False`, which never raises at all). A plain `except
    SettleUnavailable` still catches this (subclass), so every existing
    caller/test is unaffected; `sculpt.py`'s scaffold path (§D29-2)
    catches this DISTINCTLY and fails a reference-derived stage closed
    rather than silently proceeding with the exploding unsettled reset —
    the exact live D29 disaster (a prone reset settled to z 2.34 m,
    logged as a warning, and trained on anyway)."""


#: Consecutive settle steps that must ALL measure max|qvel| under this
#: threshold before the pose is declared at rest. A single-step check is
#: NOT sufficient: starting from qvel=0, one gravity-only mj_step only
#: accelerates each DOF by ~g*dt (~0.02 m/s at dt=0.002s) — comfortably
#: under 0.1 on the VERY FIRST step regardless of how far the pose
#: actually is from physical rest, so a 1-step check would report
#: "converged" immediately, before anything has had time to fall/settle.
#: Requiring a SUSTAINED low-velocity window (not just a momentary dip
#: mid-fall) is what makes this a genuine rest detector.
_SETTLE_QVEL_STOP_THRESH = 0.1
_SETTLE_CONSECUTIVE_STEPS = 25
#: DOF damping floor applied for the duration of the settle only (§5):
#: the raw G1 MJCF (as loaded by `resolve_mjcf_for_robot`, the same
#: unconstrained asset `refs/preview.py` renders with) carries `nu=0`
#: actuators and zero `dof_damping` everywhere — verified directly
#: against the installed asset — so without an elevated floor here a
#: released pose has NOTHING damping its motion except contacts, and
#: oscillates/bounces rather than settling cleanly.
_SETTLE_MIN_DOF_DAMPING = 2.0
#: Largest PHYSICALLY PLAUSIBLE height change during a settle. A
#: get-up/mid_start pose's own root-height derivation is anchored on
#: `_G1_CLASS_STAND_M` (0.74 m), so no legitimate settle can move the
#: pelvis by more than about one standing-height's worth. Found
#: empirically while building this function: a get-up clip with NO
#: `joint_pos` channel (so every joint defaults to its straight-leg
#: zero angle) combined with a large lying pitch/height offset can
#: interpenetrate the floor badly enough to trigger a MuJoCo contact-
#: force explosion — a FINITE but wildly implausible result (a real
#: observed case: settled 5.08 m ABOVE the requested height in under a
#: second), not a `mj_step` exception the ordinary except-block would
#: catch. This guard converts that silent-garbage failure mode into an
#: explicit `SettleUnavailable`.
_SETTLE_MAX_PLAUSIBLE_DELTA_M = 1.5


def _resolve_settle_model(mjcf_path: Path):
    """Compile the robot MJCF for settling — WITH a ground plane.

    `sculptor/refs/preview.py`'s resolvers (reused by `settle_reset` for
    MJCF path lookup) point at the RAW per-robot asset, which — verified
    directly against the installed G1 MJCF — carries `ngeom` robot
    geoms and ZERO ground-plane geoms (fine for `preview.py`'s use, a
    single `mj_forward` pose with no dynamics stepping; fatal for
    physics settling, where an ungrounded model free-falls forever
    under gravity with nothing to catch it — caught empirically while
    building this function: an unmodified raw-asset settle measured a
    ~4.8 m drop in 1 s of sim time, matching free-fall, not a settle).
    Uses `mujoco.MjSpec` to inject a `mjGEOM_PLANE` at the world origin
    ONLY when the compiled model doesn't already have one (a future
    robot asset that DOES ship its own floor is left untouched)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    model = spec.compile()
    has_floor = bool(np.any(
        np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)))
    if not has_floor:
        floor = spec.worldbody.add_geom()
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.pos = [0.0, 0.0, 0.0]
        floor.contype = 1
        floor.conaffinity = 1
        model = spec.compile()
    return model


def _quat_from_pitch_roll(pitch: float, roll: float) -> np.ndarray:
    """Forward direction of `_quat_wxyz_to_pitch_rad`/
    `_quat_wxyz_to_roll_rad`: build the wxyz unit quaternion mjlab's
    `reset_root_state_uniform` produces when it applies a
    `pose_range["pitch"/"roll"]` OFFSET (yaw=0) against the G1 entity's
    DEFAULT orientation (verified against `mjlab.envs.mdp.events
    .reset_root_state_uniform` and `.utils.lab_api.math
    .quat_from_euler_xyz`/`.quat_mul`: `orientations = quat_mul(
    default_quat, quat_from_euler_xyz(roll, pitch, yaw))`). The G1 MJCF's
    pelvis body carries no `quat` attribute — i.e. an IDENTITY default
    orientation — so `quat_mul(identity, delta)` reduces to `delta`
    itself; this function returns exactly that `delta` (yaw=0).

    Round-trips exactly through `_quat_wxyz_to_pitch_rad`/
    `_quat_wxyz_to_roll_rad` for the (roll, pitch) domain those two
    functions are valid over — hand-verified numerically for several
    (roll, pitch) pairs spanning +-pi; see
    `test_quat_from_pitch_roll_roundtrips_through_extraction`.
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])


def settle_reset(
    reset_scalars: dict,
    *,
    joint_names: Optional[list] = None,
    robot: str = "g1",
    settle_time_s: float = 0.75,
) -> dict:
    """Run a derived EVAL reset pose to physical rest on CPU MuJoCo and
    re-derive its scalars from the settled state (§D20a).

    D20a's GPU probe found that derived lying resets can be PROPPED —
    the derivation copies the clip's start-window pose verbatim, and
    mocap start frames can be mid-roll/unsettled even after segment QC
    (z-stillness ≠ whole-body rest); two of four real stage resets
    collapsed 0.13-0.23 m during a 0.5 s zero-action settle. This
    function makes every derived reset physically resting BY
    CONSTRUCTION instead: build the exact qpos the derived scalars
    describe, drop it onto the real robot MJCF with elevated joint
    damping (so limbs sag to rest instead of oscillating — the raw asset
    has zero damping, see `_SETTLE_MIN_DOF_DAMPING`), step physics with
    zero control until velocity stays low for a sustained window (or
    `settle_time_s` elapses), and read the new (z, pitch, roll, joint
    angles) back off the settled state.

    This intentionally mimics "find the resting configuration for this
    posture" — it does NOT reproduce the training runner's exact reset
    mechanics (no domain randomization, no PD control, and the
    `reset_joints_to_reference` event's own soft-limit clamp is not
    replicated here since the raw MJCF has no actuators to clamp
    against).

    Parameters
    ----------
    reset_scalars : a flat SCALAR reset payload — the shape
        `derive_eval_reset` returns (`reset_height_offset_m` required;
        `reset_pitch_offset_rad`/`reset_roll_offset_rad`/
        `reset_joint_pos_target` optional). Any OTHER key (e.g.
        `reset_vertical_velocity_mps`, `reset_joint_pos_noise_rad`,
        `fell_over_termination`) is copied through UNCHANGED into the
        returned scalars — settling only touches pose, never velocity/
        noise/termination knobs.
    joint_names : the clip's `joint_names`, same order as
        `reset_scalars["reset_joint_pos_target"]` — REQUIRED to place a
        joint target (each one is resolved BY NAME via `model.joint
        (name)`, never by index; an unresolvable name is skipped, not
        fatal — mirrors `refs/preview.py`'s per-frame posing
        discipline). Ignored (with no target applied) if `None` or if
        `reset_scalars` carries no `reset_joint_pos_target`.
    robot : resolved to an MJCF via `sculptor.refs.preview
        .resolve_mjcf_for_robot` — reused rather than re-implemented, the
        SAME resolver `refs/preview.py`'s clip-preview rendering uses.
    settle_time_s : maximum sim time to step before giving up on
        convergence (the settled-but-not-yet-`_SETTLE_CONSECUTIVE_STEPS`
        -confirmed state is still returned — see `converged` below).

    Returns
    -------
    dict with keys:
      "scalars": a COPY of `reset_scalars` with `reset_height_offset_m`
        (always), and `reset_pitch_offset_rad`/`reset_roll_offset_rad`/
        `reset_joint_pos_target` (only the ones that were present in the
        input) replaced by their settled-state re-derivation.
      "delta_z_m": settled z minus the REQUESTED z (negative = sagged
        down during settling — the D20a "propped" signature).
      "steps": number of `mj_step` calls actually taken.
      "converged": whether `_SETTLE_CONSECUTIVE_STEPS` consecutive steps
        measured max|qvel| under `_SETTLE_QVEL_STOP_THRESH` before
        `settle_time_s` ran out.
      "duration_s": `steps * model.opt.timestep`.

    Raises
    ------
    `SettleUnavailable` on ANY failure to resolve the MJCF, load the
    model, or step physics. Never raises for a merely non-converged
    settle (that's `converged: False` in a normal return) — only for a
    genuine failure to run the simulation at all.
    """
    from sculptor.refs.preview import resolve_mjcf_for_robot

    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        raise SettleUnavailable(
            f"mujoco import failed: {type(e).__name__}: {e}") from e

    try:
        mjcf_path = resolve_mjcf_for_robot(robot)
    except Exception as e:  # noqa: BLE001 — PreviewUnavailable or anything else
        raise SettleUnavailable(
            f"MJCF resolution failed for robot={robot!r}: "
            f"{type(e).__name__}: {e}") from e

    try:
        model = _resolve_settle_model(mjcf_path)
        data = mujoco.MjData(model)
    except Exception as e:  # noqa: BLE001
        raise SettleUnavailable(
            f"MuJoCo failed to load {mjcf_path}: {type(e).__name__}: "
            f"{e}") from e

    height_offset = float(reset_scalars.get("reset_height_offset_m", 0.0))
    pitch = float(reset_scalars.get("reset_pitch_offset_rad") or 0.0)
    roll = float(reset_scalars.get("reset_roll_offset_rad") or 0.0)
    joint_target = reset_scalars.get("reset_joint_pos_target")
    requested_z = _G1_CLASS_STAND_M + height_offset

    qpos0 = np.zeros(model.nq, dtype=np.float64)
    qpos0[2] = requested_z
    qpos0[3:7] = _quat_from_pitch_roll(pitch, roll)
    if joint_target and joint_names:
        for name, angle in zip(joint_names, joint_target):
            try:
                adr = int(model.joint(str(name)).qposadr[0])
            except Exception:  # noqa: BLE001 — unknown joint name for this MJCF
                continue
            qpos0[adr] = float(angle)

    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    original_damping = model.dof_damping.copy()
    model.dof_damping[:] = np.maximum(
        original_damping, _SETTLE_MIN_DOF_DAMPING)

    dt = float(model.opt.timestep)
    max_steps = max(1, int(round(settle_time_s / dt)))
    steps = 0
    converged = False
    try:
        mujoco.mj_forward(model, data)
        consecutive = 0
        for i in range(1, max_steps + 1):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            steps = i
            if not (np.all(np.isfinite(data.qpos))
                    and np.all(np.isfinite(data.qvel))):
                raise SettleUnavailable(
                    f"physics settle diverged (non-finite state) at "
                    f"step {steps}")
            if float(np.max(np.abs(data.qvel))) < _SETTLE_QVEL_STOP_THRESH:
                consecutive += 1
                if consecutive >= _SETTLE_CONSECUTIVE_STEPS:
                    converged = True
                    break
            else:
                consecutive = 0
    except SettleUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 — non-convergence / step failure
        raise SettleUnavailable(
            f"physics settle step failed after {steps} steps: "
            f"{type(e).__name__}: {e}") from e
    finally:
        model.dof_damping[:] = original_damping

    settled_z = float(data.qpos[2])
    settled_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    if abs(settled_z - requested_z) > _SETTLE_MAX_PLAUSIBLE_DELTA_M:
        # §D29-2: this specific branch is THE explosion-class failure —
        # the derived pose itself is invalid, not settle infrastructure
        # being unavailable. `SettleExplosion` (a `SettleUnavailable`
        # subclass) lets `sculpt.py`'s scaffold path fail the stage
        # closed for a reference-derived reset instead of silently
        # training from the exploding unsettled pose (§D29 live).
        raise SettleExplosion(
            f"physics settle produced an implausible height change "
            f"({settled_z - requested_z:+.3f} m over {steps} steps, "
            f"exceeds the {_SETTLE_MAX_PLAUSIBLE_DELTA_M} m plausibility "
            f"bound) — likely a contact-force explosion from joint/"
            f"orientation interpenetration (e.g. a lying pitch/height "
            f"offset with no reset_joint_pos_target, so every joint sits "
            f"at its straight-leg default), not a genuine settle."
        )

    new_scalars = dict(reset_scalars)
    new_scalars["reset_height_offset_m"] = round(
        settled_z - _G1_CLASS_STAND_M, 4)
    if "reset_pitch_offset_rad" in reset_scalars:
        new_scalars["reset_pitch_offset_rad"] = round(
            float(_quat_wxyz_to_pitch_rad(settled_quat[None, :])[0]), 4)
    if "reset_roll_offset_rad" in reset_scalars:
        new_scalars["reset_roll_offset_rad"] = round(
            float(_quat_wxyz_to_roll_rad(settled_quat[None, :])[0]), 4)
    if joint_target and joint_names:
        new_target = []
        for name, orig in zip(joint_names, joint_target):
            try:
                adr = int(model.joint(str(name)).qposadr[0])
                new_target.append(round(float(data.qpos[adr]), 4))
            except Exception:  # noqa: BLE001 — unresolved name: pass through
                new_target.append(float(orig))
        new_scalars["reset_joint_pos_target"] = new_target

    return {
        "scalars": new_scalars,
        "delta_z_m": round(settled_z - requested_z, 4),
        "steps": steps,
        "converged": converged,
        "duration_s": round(steps * dt, 4),
    }
