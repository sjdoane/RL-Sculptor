"""§REFERENCE_TRAJECTORY_PLAN §8: generalized reference RSI beyond the
procedural jump clip — get-up-shaped (rises-from-low-start) motions.

Covers:
  * archetype detection (getup / airborne / other) via `derive_rsi_train_keys`
    and `derive_reference_reset`'s observable outputs;
  * get-up RSI ranges: lying reset height, near-zero reset velocity, and a
    sunk-termination guard that sits BELOW the clip's own minimum height so
    it can never fire at reset (§8's "sunk-height termination must NOT fire
    at reset" requirement);
  * orientation (`reset_pitch_offset_rad`) derived from a lying
    `root_quat_wxyz`, when the clip carries one;
  * the jump-clip regression pin lives in `test_reference.py`
    (`test_derive_reference_reset_byte_identical_to_rsi_keys_for_jump_clip`);
  * a load-failure-must-not-crash smoke test mirroring the exact
    load→apply call pattern the orchestrator hook (`sculpt.py`'s
    `needs_reference_rsi` branch, §JUMP_SCAFFOLD) uses.

Fence note (see final task report): the actual stage-scaffold orchestrator
that consumes `stage.needs_reference_rsi` / `stage.reference_clip_id` lives
in `sculptor/sculpt.py::_run_one_stage` (around the "§JUMP_SCAFFOLD"
comment), NOT in `sculptor/mission_runtime.py` — `mission_runtime.py` only
holds criterion-evaluation/trajectory-loading helpers and imports nothing
from `sculptor.reference` or `sculptor.mission` today. `sculpt.py` is
outside this task's file fence, so the tests below exercise the
`reference.py` primitives the orchestrator would call, at the same call
sites and with the same failure-handling shape, rather than driving
`sculpt.mission_run` end-to-end for the library-clip-by-id path (the
existing `test_stage_reference_rsi_applied_when_flagged` in
`tests/test_mission_run.py` already covers the procedural-fallback half of
that orchestration and is unaffected by this change).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.env_spec import validate_env_spec
from sculptor.reference import (
    G1_CLASS_STAND_M,
    _archetype,
    apply_reference_rsi,
    derive_eval_reset,
    derive_reference_reset,
    derive_rsi_train_keys,
    load_clip,
    save_clip,
    validate_clip,
)


# ── Synthetic clip builders ─────────────────────────────────────────────
def _make_getup_clip(
    *, fps: float = 50.0, with_quat: bool = False, with_joints: bool = False,
) -> dict:
    """z: 0.10 flat 0.5s -> ramp 1.0s -> 0.75 flat 0.5s. Optionally carries
    a `root_quat_wxyz` pitching from lying (pi/2) to upright (0) over the
    same three phases, and/or a `joint_pos` trace (3 synthetic joints,
    bent during the lying phase, ~0 once standing)."""
    def seg(dur: float, fn):
        n = max(2, int(round(dur * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    lying = seg(0.5, lambda s: np.full_like(s, 0.10))
    ramp = seg(1.0, lambda s: 0.10 + (0.75 - 0.10) * s)
    stand = seg(0.5, lambda s: np.full_like(s, 0.75))
    z = np.concatenate([lying, ramp, stand])
    clip: dict = {"root_pos_z": z, "fps": fps,
                  "meta": {"source": "synthetic:getup"}}
    if with_quat:
        def quat_pitch(theta: float) -> np.ndarray:
            return np.array(
                [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])

        lying_q = np.tile(quat_pitch(np.pi / 2), (lying.shape[0], 1))
        ramp_s = np.linspace(0.0, 1.0, ramp.shape[0], endpoint=False)
        ramp_q = np.stack(
            [quat_pitch(np.pi / 2 * (1 - s)) for s in ramp_s])
        stand_q = np.tile(quat_pitch(0.0), (stand.shape[0], 1))
        clip["root_quat_wxyz"] = np.concatenate(
            [lying_q, ramp_q, stand_q], axis=0)
    if with_joints:
        n = z.shape[0]
        # 3 synthetic joints: bent ~1.0/-0.8/0.5 rad while lying, ~0 once
        # standing (a crude knee/elbow-style "curled up" get-up posture).
        lying_j = np.tile([1.0, -0.8, 0.5], (lying.shape[0], 1))
        ramp_j = np.linspace(
            lying_j[0], [0.0, 0.0, 0.0], ramp.shape[0])
        stand_j = np.tile([0.0, 0.0, 0.0], (stand.shape[0], 1))
        clip["joint_pos"] = np.concatenate(
            [lying_j, ramp_j, stand_j], axis=0)
        clip["joint_names"] = ["j0", "j1", "j2"]
    assert validate_clip(clip) == []
    return clip


def _make_never_rising_never_low_clip(*, fps: float = 50.0) -> dict:
    """Flat at standing height the whole time — garbage input for RSI
    (neither jump- nor get-up-shaped)."""
    return {"root_pos_z": np.full(120, 0.78), "fps": fps}


def _make_origin_relative_hop_clip(*, fps: float = 60.0) -> dict:
    n = 120
    u = np.linspace(0.0, 1.0, n)
    return {
        "root_pos_z": 0.02 + 0.28 * np.sin(np.pi * u) ** 2,
        "fps": fps,
        "root_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "meta": {"source": "dataset", "tokens": ["run", "to", "hop"]},
    }


def test_origin_relative_dataset_hop_is_airborne_with_physical_sunk_guard() -> None:
    clip = _make_origin_relative_hop_clip()
    assert _archetype(clip) == "airborne"
    keys = derive_rsi_train_keys(clip)
    assert keys["reset_height_offset_m"][0] == pytest.approx(0.0)
    assert keys["reset_height_offset_m"][1] > 0.2
    assert keys["min_base_height_termination_m"] == pytest.approx(
        round(0.64 * G1_CLASS_STAND_M, 2))


def _make_mid_start_clip(
    *, fps: float = 50.0, with_quat: bool = False, with_joints: bool = False,
) -> dict:
    """z: 0.42 flat 0.5s (mid-height — a crouch-to-stand start, D19's
    documented gap) -> ramp 1.0s -> 0.75 flat 0.5s. Same optional
    quat/joint_pos shape as `_make_getup_clip`."""
    def seg(dur: float, fn):
        n = max(2, int(round(dur * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    crouch = seg(0.5, lambda s: np.full_like(s, 0.42))
    ramp = seg(1.0, lambda s: 0.42 + (0.75 - 0.42) * s)
    stand = seg(0.5, lambda s: np.full_like(s, 0.75))
    z = np.concatenate([crouch, ramp, stand])
    clip: dict = {"root_pos_z": z, "fps": fps,
                  "meta": {"source": "synthetic:mid_start"}}
    if with_quat:
        def quat_pitch(theta: float) -> np.ndarray:
            return np.array(
                [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])

        crouch_q = np.tile(quat_pitch(np.pi / 6), (crouch.shape[0], 1))
        ramp_s = np.linspace(0.0, 1.0, ramp.shape[0], endpoint=False)
        ramp_q = np.stack(
            [quat_pitch(np.pi / 6 * (1 - s)) for s in ramp_s])
        stand_q = np.tile(quat_pitch(0.0), (stand.shape[0], 1))
        clip["root_quat_wxyz"] = np.concatenate(
            [crouch_q, ramp_q, stand_q], axis=0)
    if with_joints:
        n = z.shape[0]
        crouch_j = np.tile([0.6, -0.4], (crouch.shape[0], 1))
        ramp_j = np.linspace(crouch_j[0], [0.0, 0.0], ramp.shape[0])
        stand_j = np.tile([0.0, 0.0], (stand.shape[0], 1))
        clip["joint_pos"] = np.concatenate(
            [crouch_j, ramp_j, stand_j], axis=0)
        clip["joint_names"] = ["knee", "hip"]
    assert validate_clip(clip) == []
    return clip


# ── §start_pose mid_start archetype (D19 closure) ────────────────────────
def test_mid_start_archetype_classified_between_getup_and_airborne() -> None:
    from sculptor.reference import _archetype

    clip = _make_mid_start_clip()
    assert _archetype(clip) == "mid_start"


def test_mid_start_derives_negative_height_offset_anchored_absolute() -> None:
    """Same absolute-anchor math as getup (D19): a 0.42 m mid-height
    start derives an offset well below zero but not as deep as a true
    lying (getup) start."""
    clip = _make_mid_start_clip()
    keys = derive_rsi_train_keys(clip)
    lo, hi = keys["reset_height_offset_m"]
    assert lo < 0.0 and hi < 0.0
    assert lo <= hi
    # Sanity: 0.74 (G1_CLASS_STAND_M) + offset ~= the clip's own
    # start-window height (0.42), well above a true lying floor.
    from sculptor.reference import G1_CLASS_STAND_M

    assert 0.30 <= G1_CLASS_STAND_M + hi <= 0.50


def test_mid_start_sunk_termination_guard_below_clip_minimum() -> None:
    clip = _make_mid_start_clip()
    keys = derive_rsi_train_keys(clip)
    z_min = float(clip["root_pos_z"].min())
    sunk = keys["min_base_height_termination_m"]
    assert sunk < z_min


def test_mid_start_disables_fell_over_termination() -> None:
    """§D16 conservative lesson, extended to mid_start: even a bent
    crouch pose can trip orientation-based fall termination at reset."""
    clip = _make_mid_start_clip()
    derived = derive_reference_reset(clip)
    assert derived["fell_over_termination"] is False


def test_mid_start_derives_orientation_and_joint_target_when_present() -> None:
    clip = _make_mid_start_clip(with_quat=True, with_joints=True)
    derived = derive_reference_reset(clip)
    assert "reset_pitch_offset_rad" in derived
    assert "reset_joint_pos_target" in derived
    assert len(derived["reset_joint_pos_target"]) == 2


def test_mid_start_eval_reset_is_non_none() -> None:
    """§D17 extended to mid_start: a non-standing start IS the task,
    whether it's flat-on-the-ground or mid-crouch."""
    clip = _make_mid_start_clip()
    eval_reset = derive_eval_reset(clip)
    assert eval_reset is not None
    assert eval_reset["fell_over_termination"] is False
    assert eval_reset["reset_vertical_velocity_mps"] == 0.0


def test_mid_start_apply_reference_rsi_passes_env_spec_gate(
    tmp_path: Path,
) -> None:
    clip = _make_mid_start_clip()
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert spec["train"]["reset_height_offset_m"][1] <= 0.0
    assert spec["train"]["fell_over_termination"] is False


def test_archetype_boundary_getup_vs_mid_start() -> None:
    """The 0.35 m boundary: just under is getup, just at/over is
    mid_start (aligned with `_GETUP_START_MAX_M`)."""
    from sculptor.reference import _archetype

    fps = 50.0
    below = {"root_pos_z": np.full(60, 0.30), "fps": fps}
    at_or_above = {"root_pos_z": np.full(60, 0.40), "fps": fps}
    assert _archetype(below) == "getup"
    assert _archetype(at_or_above) == "mid_start"


def test_archetype_boundary_mid_start_vs_airborne() -> None:
    """The 0.55 m boundary (`_GETUP_END_MIN_M`, reused deliberately):
    a flat clip just under never counts as airborne/other since it
    never rises — but a RISING clip starting just at/over 0.55 m
    classifies airborne, matching pre-mid_start behavior exactly."""
    from sculptor.reference import _archetype

    fps = 50.0
    n = 60
    # Starts at 0.60 (>= 0.55) and rises above its own baseline -> airborne.
    z = np.concatenate([np.full(30, 0.60), np.full(30, 0.90)])
    clip = {"root_pos_z": z, "fps": fps}
    assert _archetype(clip) == "airborne"


# ── Get-up archetype: RSI ranges ────────────────────────────────────────
def test_getup_clip_derives_lying_reset_height() -> None:
    clip = _make_getup_clip()
    keys = derive_rsi_train_keys(clip)
    lo, hi = keys["reset_height_offset_m"]
    # Offset relative to the clip's own "standing" (end-window) height:
    # lying (~0.10) - stand (~0.75) ≈ -0.65, well below zero.
    assert lo < -0.3 and hi < -0.3
    assert lo <= hi


def test_getup_clip_derives_near_zero_reset_velocity() -> None:
    """A resting lying start, not a falling one: reset vertical velocity
    must be near zero (the synthetic clip's initial window is flat)."""
    clip = _make_getup_clip()
    keys = derive_rsi_train_keys(clip)
    vlo, vhi = keys["reset_vertical_velocity_mps"]
    assert abs(vlo) < 0.5 and abs(vhi) < 0.5


def test_getup_sunk_termination_guard_below_clip_minimum() -> None:
    """§8: sunk-height termination must NOT fire at reset. The derived
    threshold must sit BELOW the clip's own observed minimum z, with a
    positive margin — never above it (which would trip immediately on a
    lying reset)."""
    clip = _make_getup_clip()
    keys = derive_rsi_train_keys(clip)
    z_min = float(clip["root_pos_z"].min())
    sunk = keys["min_base_height_termination_m"]
    assert sunk < z_min
    assert z_min - sunk >= 0.01   # non-trivial margin, not float noise


def test_getup_derivation_passes_the_real_env_spec_gate(tmp_path: Path) -> None:
    """The persisted spec validates clean, and — because the reset height
    offset is non-positive (a lying start, not airborne) — the RSI<->ET
    cross-field invariant does NOT force the sunk key the way it does for
    jump clips; §8's guard is a DIFFERENT mechanism (derived-below-min),
    not the airborne-pairing rule, and both must coexist without
    conflict."""
    clip = _make_getup_clip()
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert spec["train"]["reset_height_offset_m"][1] <= 0.0
    assert "min_base_height_termination_m" in spec["train"]
    assert spec["meta"]["source"] == "reference:synthetic:getup"


# ── Orientation derivation ───────────────────────────────────────────────
def test_getup_orientation_derived_from_lying_quat() -> None:
    clip = _make_getup_clip(with_quat=True)
    derived = derive_reference_reset(clip)
    assert "reset_pitch_offset_rad" in derived
    lo, hi = derived["reset_pitch_offset_rad"]
    # Lying ~ pi/2 rad pitched from the clip's own upright (end-window)
    # orientation baseline.
    assert lo == pytest.approx(np.pi / 2, abs=0.05)
    assert hi == pytest.approx(np.pi / 2, abs=0.05)


def test_orientation_not_derived_without_quat() -> None:
    clip = _make_getup_clip(with_quat=False)
    derived = derive_reference_reset(clip)
    assert "reset_pitch_offset_rad" not in derived


def test_orientation_not_derived_for_airborne_clips_even_with_quat() -> None:
    """Jump clips don't need a lying-orientation reset; a quat carried on
    an airborne clip (e.g. from a richer mocap source) must not spuriously
    produce a pitch offset."""
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    n = clip["root_pos_z"].shape[0]
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    clip["root_quat_wxyz"] = quat
    derived = derive_reference_reset(clip)
    assert "reset_pitch_offset_rad" not in derived


def test_reset_pitch_offset_rad_within_hard_envelope() -> None:
    """`reset_pitch_offset_rad` is now a real `env_spec._TRAIN_RANGES`
    key (§8 part 2 — the prior fence gap is closed), so the derivation's
    output must fall inside the VALIDATOR's own envelope, the single
    source of truth."""
    from sculptor.env_spec import _TRAIN_RANGES

    clip = _make_getup_clip(with_quat=True)
    derived = derive_reference_reset(clip)
    lo, hi = derived["reset_pitch_offset_rad"]
    b_lo, b_hi = _TRAIN_RANGES["reset_pitch_offset_rad"]
    assert b_lo <= lo <= b_hi
    assert b_lo <= hi <= b_hi


def test_apply_reference_rsi_persists_orientation_key(
    tmp_path: Path,
) -> None:
    """§8 part 2: `apply_reference_rsi` now WRITES
    `reset_pitch_offset_rad`/`reset_roll_offset_rad` into the env spec —
    both are live `env_spec.validate_env_spec` keys as of the second
    increment (env_gen.py's `_TrainModel` gained matching fields, so the
    drift guard stays green)."""
    clip = _make_getup_clip(with_quat=True)
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert "reset_pitch_offset_rad" in spec["train"]
    assert "reset_roll_offset_rad" in spec["train"]
    assert "orientation offsets" in spec["meta"]["rationale"]


# ── Per-joint reference posture derivation (§8 part 2) ───────────────────
def test_getup_joint_target_derived_from_lying_joint_pos() -> None:
    clip = _make_getup_clip(with_joints=True)
    derived = derive_reference_reset(clip)
    assert "reset_joint_pos_target" in derived
    target = derived["reset_joint_pos_target"]
    assert len(target) == 3
    # Initial-window mean of the lying-phase joint_pos ([1.0, -0.8, 0.5]).
    assert target[0] == pytest.approx(1.0, abs=0.05)
    assert target[1] == pytest.approx(-0.8, abs=0.05)
    assert target[2] == pytest.approx(0.5, abs=0.05)
    assert "reset_joint_pos_noise_rad" in derived
    assert derived["reset_joint_pos_noise_rad"] > 0.0


def test_joint_target_not_derived_without_joint_pos() -> None:
    clip = _make_getup_clip(with_joints=False)
    derived = derive_reference_reset(clip)
    assert "reset_joint_pos_target" not in derived
    assert "reset_joint_pos_noise_rad" not in derived


def test_joint_target_not_derived_for_airborne_clips_even_with_joint_pos() -> None:
    """Same "only get-up-shaped clips get this" guard as orientation."""
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    n = clip["root_pos_z"].shape[0]
    clip["joint_pos"] = np.zeros((n, 2))
    clip["joint_names"] = ["j0", "j1"]
    derived = derive_reference_reset(clip)
    assert "reset_joint_pos_target" not in derived


def test_apply_reference_rsi_persists_joint_target(tmp_path: Path) -> None:
    """§8 part 2: `apply_reference_rsi` writes `reset_joint_pos_target` +
    `reset_joint_pos_noise_rad` into the env spec, and the persisted spec
    validates clean end to end."""
    clip = _make_getup_clip(with_joints=True)
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert spec["train"]["reset_joint_pos_target"] == pytest.approx(
        [1.0, -0.8, 0.5], abs=0.05)
    assert spec["train"]["reset_joint_pos_noise_rad"] > 0.0
    assert "per-joint reference posture" in spec["meta"]["rationale"]


def test_joint_target_clamped_to_validator_element_bounds() -> None:
    """A pathological clip with an out-of-envelope joint value must be
    CLAMPED (never fail the gate) — same discipline as every other
    derived key in this module."""
    from sculptor.env_spec import _JOINT_TARGET_ELEMENT_BOUNDS

    clip = _make_getup_clip(with_joints=True)
    clip["joint_pos"] = clip["joint_pos"].copy()
    clip["joint_pos"][:, 0] = 99.0   # wildly out of bounds throughout
    derived = derive_reference_reset(clip)
    lo, hi = _JOINT_TARGET_ELEMENT_BOUNDS
    assert lo <= derived["reset_joint_pos_target"][0] <= hi


# ── fell_over_termination (§get-up RSI fix, 2026-07-09) ──────────────────
def test_getup_derivation_disables_fell_over_termination() -> None:
    """A get-up clip's lying start would trip the task's own
    fell-over/bad-orientation termination on the reset itself — the
    get-up branch must emit `fell_over_termination: False` regardless of
    whether the clip also carries orientation/joint_pos channels (the
    reset-height-offset alone is already lying-shaped)."""
    clip = _make_getup_clip()   # no quat, no joints — minimal getup clip
    derived = derive_reference_reset(clip)
    assert derived["fell_over_termination"] is False


def test_getup_derivation_disables_fell_over_termination_with_full_channels() -> None:
    clip = _make_getup_clip(with_quat=True, with_joints=True)
    derived = derive_reference_reset(clip)
    assert derived["fell_over_termination"] is False


def test_airborne_derivation_does_not_touch_fell_over_termination() -> None:
    """Jump/airborne behavior must be byte-unchanged — no
    `fell_over_termination` key at all (not even `True`), since the
    validator's default (term stays active) is exactly what jump
    training already relies on."""
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    derived = derive_reference_reset(clip)
    assert "fell_over_termination" not in derived
    keys = derive_rsi_train_keys(clip)
    assert "fell_over_termination" not in keys


def test_apply_reference_rsi_persists_fell_over_termination_for_getup(
    tmp_path: Path,
) -> None:
    clip = _make_getup_clip()
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert spec["train"]["fell_over_termination"] is False
    assert "fell_over termination disabled" in spec["meta"]["rationale"]


def test_apply_reference_rsi_airborne_omits_fell_over_termination(
    tmp_path: Path,
) -> None:
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    env_dir = tmp_path / "env"
    path = apply_reference_rsi(env_dir, clip)
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert validate_env_spec(spec) == []
    assert "fell_over_termination" not in spec["train"]


# ── §D17: derive_eval_reset — stage-fixed, deterministic eval override ──
def test_derive_eval_reset_getup_returns_midpoint_payload() -> None:
    """A minimal (no quat/joints) get-up clip: height offset is the
    MIDPOINT of the train-side range, vertical velocity is exactly
    zero (not just near-zero — eval must be deterministic), and
    fell_over_termination is disabled (the fixed lying start would
    otherwise trip it, exactly as it does in train)."""
    clip = _make_getup_clip()
    full = derive_reference_reset(clip)
    eval_reset = derive_eval_reset(clip)
    assert eval_reset is not None
    lo, hi = full["reset_height_offset_m"]
    assert eval_reset["reset_height_offset_m"] == round((lo + hi) / 2.0, 4)
    assert eval_reset["reset_vertical_velocity_mps"] == 0.0
    assert eval_reset["fell_over_termination"] is False
    # No orientation/joint keys — the clip carries no quat/joint_pos.
    assert "reset_pitch_offset_rad" not in eval_reset
    assert "reset_joint_pos_target" not in eval_reset


def test_derive_eval_reset_getup_with_quat_and_joints() -> None:
    """Full-channel get-up clip: pitch/roll collapse to their train-side
    midpoints, the joint target is passed through UNCHANGED (the clip's
    own reference posture — nothing to average, it's already a single
    vector), and joint-pos noise is forced to exactly zero (eval must be
    reproducible across resets, no per-episode variation)."""
    clip = _make_getup_clip(with_quat=True, with_joints=True)
    full = derive_reference_reset(clip)
    eval_reset = derive_eval_reset(clip)
    assert eval_reset is not None
    p_lo, p_hi = full["reset_pitch_offset_rad"]
    assert eval_reset["reset_pitch_offset_rad"] == round(
        (p_lo + p_hi) / 2.0, 4)
    r_lo, r_hi = full["reset_roll_offset_rad"]
    assert eval_reset["reset_roll_offset_rad"] == round(
        (r_lo + r_hi) / 2.0, 4)
    assert eval_reset["reset_joint_pos_target"] == full["reset_joint_pos_target"]
    assert eval_reset["reset_joint_pos_noise_rad"] == 0.0


def test_derive_eval_reset_airborne_clip_returns_none() -> None:
    """Jump eval must stay standing-start — unchanged behavior. A jump
    (or any non-get-up) clip's task definition is not "start airborne";
    the lying-start override is get-up-specific."""
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    assert derive_eval_reset(clip) is None


def test_derive_eval_reset_other_clip_returns_none() -> None:
    """A clip that is neither jump- nor get-up-shaped: no eval override
    (mirrors derive_rsi_train_keys's "other" archetype — no RSI states
    to initialize from at all, so certainly no eval override either)."""
    clip = _make_never_rising_never_low_clip()
    assert derive_eval_reset(clip) is None


def test_derive_eval_reset_deterministic_across_calls() -> None:
    """Same clip in, same payload out, every call — the whole point of a
    stage-FIXED override (§D17: "same payload every iteration of the
    stage by construction")."""
    clip = _make_getup_clip(with_quat=True, with_joints=True)
    a = derive_eval_reset(clip)
    b = derive_eval_reset(clip)
    assert a == b


# ── Never-rising, never-low clips: unchanged garbage-input behavior ──────
def test_never_rising_never_low_clip_raises_actionable_error() -> None:
    clip = _make_never_rising_never_low_clip()
    with pytest.raises(ValueError, match="never rises|never transitions"):
        derive_rsi_train_keys(clip)
    with pytest.raises(ValueError, match="never rises|never transitions"):
        derive_reference_reset(clip)


# ── Load-failure fallback path (mirrors the orchestrator's try/except
#    shape — see module docstring's fence note) ─────────────────────────
def test_missing_clip_load_failure_does_not_crash_the_caller(
    tmp_path: Path,
) -> None:
    """`load_clip` on a missing/corrupt path raises; a caller wrapping it
    the way `sculpt.py`'s `needs_reference_rsi` branch does (broad
    except, proceed without RSI, never crash the mission) must be able to
    catch it and continue. This pins the CONTRACT the orchestrator relies
    on, since the orchestrator itself is out of this task's file fence."""
    missing = tmp_path / "does_not_exist.npz"
    applied = False
    error: Exception | None = None
    try:
        clip = load_clip(missing)
        apply_reference_rsi(tmp_path / "env", clip)
        applied = True
    except Exception as e:  # noqa: BLE001 — mirrors the orchestrator's guard
        error = e
    assert not applied
    assert isinstance(error, (OSError, ValueError, FileNotFoundError))
    # And the env dir must be untouched — no partial/half-written spec.
    assert not (tmp_path / "env").exists()


def test_corrupt_clip_load_failure_does_not_crash_the_caller(
    tmp_path: Path,
) -> None:
    """Same contract, but for a clip file that exists yet fails
    `validate_clip` on load (e.g. truncated/hand-edited .npz) — `load_clip`
    raises ValueError rather than silently returning a bad clip."""
    from sculptor.reference import make_procedural_jump_clip

    clip = make_procedural_jump_clip()
    bad = dict(clip)
    bad["root_pos_z"] = bad["root_pos_z"].copy()
    bad["root_pos_z"][0] = -1.0   # violates "strictly positive"
    path = tmp_path / "bad.npz"
    # Bypass save_clip's own validation gate to simulate a corrupted file
    # already on disk (e.g. hand-edited, or written by an older/buggy
    # ingest path) rather than testing save_clip's guard again.
    payload = {k: v for k, v in bad.items()
               if k in ("root_pos_z", "fps") and isinstance(v, np.ndarray)}
    np.savez_compressed(
        path, root_pos_z=bad["root_pos_z"].astype(np.float32),
        fps=np.float32(bad["fps"]),
        meta_json=np.bytes_(b"{}"))

    applied = False
    error: Exception | None = None
    try:
        loaded = load_clip(path)
        apply_reference_rsi(tmp_path / "env", loaded)
        applied = True
    except Exception as e:  # noqa: BLE001 — mirrors the orchestrator's guard
        error = e
    assert not applied
    assert isinstance(error, ValueError)
    assert not (tmp_path / "env").exists()


def test_getup_clip_saved_and_loaded_via_library_roundtrip_still_derives(
    tmp_path: Path,
) -> None:
    """A get-up clip saved through the SAME `save_clip`/`load_clip` pair
    the reference library (`sculptor.refs.library`) uses for
    `reference_clip_id`-attached clips still classifies + derives
    correctly after a save->load round trip (guards against the
    archetype detector accidentally depending on float64-vs-float32
    precision or key ordering that `save_clip`'s npz round trip could
    perturb)."""
    clip = _make_getup_clip(with_quat=True)
    path = save_clip(tmp_path / "clip.npz", clip)
    loaded = load_clip(path)
    derived = derive_reference_reset(loaded)
    assert derived["reset_height_offset_m"][1] <= 0.0
    assert "reset_pitch_offset_rad" in derived
    z_min = float(loaded["root_pos_z"].min())
    assert derived["min_base_height_termination_m"] < z_min


def test_lie_to_crouch_clip_derives_lying_reset_d19():
    """D19 regression pin: a low-start clip that ends in a CROUCH (never
    stands) must classify getup and derive a LYING reset anchored on the
    absolute G1-class standing height — the old clip-end anchor produced
    a ~0.56 m reset (a crouch) for exactly this shape, and no eval reset
    at all (airborne misroute) while a real mission was training."""
    import numpy as np
    from sculptor.reference import (
        _archetype, derive_eval_reset, derive_reference_reset)
    T, fps = 240, 30.0
    t = np.linspace(0.0, 1.0, T)
    z = 0.02 + 0.24 * np.clip((t - 0.3) / 0.5, 0.0, 1.0)  # 0.02 -> 0.26 crouch
    clip = {"root_pos_z": z, "fps": fps}
    assert _archetype(clip) == "getup"
    r = derive_reference_reset(clip)
    lo, hi = r["reset_height_offset_m"]
    # floor clamp: ground-clamped 0.02 m start must not go below 0.10 m
    assert 0.74 + lo >= 0.10 - 1e-9
    assert 0.74 + hi <= 0.35  # genuinely lying, never a crouch-height reset
    assert r["fell_over_termination"] is False
    e = derive_eval_reset(clip)
    assert e is not None and e["fell_over_termination"] is False


def test_prone_pi_roll_offset_survives_rounding_and_validates() -> None:
    """LIVE SCAFFOLD FAILURE (first prone mission, 2026-07-13): a
    face-down clip's derived roll offset is exactly pi, and every
    persist site rounded to 4 decimals — round(pi, 4) = 3.1416 > pi —
    so the env-spec validator's own [-pi, pi] hard bound rejected the
    spec and the stage died reference_scaffold_failed at launch.
    Readability rounding must never push a value back out of the bound
    it was just clamped into (fourth occurrence of the rounding class:
    span persistence, QC bounds, span_boundaries, now reset offsets)."""
    fps = 50.0

    def quat_roll(theta: float) -> np.ndarray:
        return np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])

    def seg(dur: float, fn):
        n = max(2, int(round(dur * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    lying = seg(0.5, lambda s: np.full_like(s, 0.10))
    ramp = seg(1.0, lambda s: 0.10 + (0.75 - 0.10) * s)
    stand = seg(0.5, lambda s: np.full_like(s, 0.75))
    z = np.concatenate([lying, ramp, stand])
    # Face-down start: roll exactly pi, unrolling to 0 while rising.
    lying_q = np.tile(quat_roll(np.pi), (lying.shape[0], 1))
    ramp_s = np.linspace(0.0, 1.0, ramp.shape[0], endpoint=False)
    ramp_q = np.stack([quat_roll(np.pi * (1 - s)) for s in ramp_s])
    stand_q = np.tile(quat_roll(0.0), (stand.shape[0], 1))
    clip = {
        "root_pos_z": z, "fps": fps,
        "root_quat_wxyz": np.concatenate([lying_q, ramp_q, stand_q], axis=0),
        "meta": {"source": "synthetic:prone-getup"},
    }
    assert validate_clip(clip) == []

    full = derive_reference_reset(clip)
    lo_b, hi_b = -np.pi, np.pi
    for key in ("reset_roll_offset_rad", "reset_pitch_offset_rad"):
        if key in full:
            a_lo, a_hi = full[key]
            assert lo_b <= a_lo <= hi_b, (key, a_lo)
            assert lo_b <= a_hi <= hi_b, (key, a_hi)

    ev = derive_eval_reset(clip)
    assert ev is not None
    for key in ("reset_roll_offset_rad", "reset_pitch_offset_rad"):
        if key in ev:
            assert lo_b <= ev[key] <= hi_b, (key, ev[key])

    # The end-to-end contract that failed live: the derived TRAIN keys
    # must survive the env-spec validator's hard bounds.
    train = derive_rsi_train_keys(clip)
    spec = {"schema_version": 1, "shared": {}, "train": train,
            "meta": {"source": "reference:test"}}
    problems = validate_env_spec(spec)
    bound_problems = [p for p in problems if "outside hard bounds" in p]
    assert bound_problems == [], bound_problems


# ── §D29-1: analytic pitch/roll derivation FROM body-frame gravity ─────
# D29 live disaster: the PRIOR derivation (raw per-axis Euler angles at
# the clip's start window MINUS the clip's own end/"standing" window)
# derived (pitch=-0.05, roll=pi) for a clip whose true start body-frame
# gravity was (0.96, -0.12, 0.24) — reconstructing as (0, 0, +1),
# UPSIDE-DOWN. The fix derives (pitch, roll) directly from the MEASURED
# start-window gravity (no cross-window subtraction, no dependence on
# the clip's own end pose) and mechanically gates the result. These
# tests use an INDEPENDENT oracle (`_oracle_gravity_from_quat`, a
# Hamilton-product quaternion rotation — the same style as reference.py's
# own `_body_frame_gravity_x`, pinned by
# `test_body_frame_gravity_x_prone_vs_supine` in
# tests/test_start_pose_qc_settle.py) rather than importing the module's
# new `_quat_wxyz_to_gravity_b` closed form, so the checks below don't
# just re-assert the implementation against itself.
def _oracle_gravity_from_quat(q: np.ndarray) -> np.ndarray:
    """Independent reference implementation of body-frame gravity
    (`R(q)^T @ (0, 0, -1)`) from a `[w, x, y, z]` unit quaternion, via
    the raw Hamilton product — NOT the closed-form `_quat_wxyz_to_
    gravity_b` under test."""
    w, x, y, z = (float(v) for v in q)
    qc = np.array([w, -x, -y, -z])
    g_world = np.array([0.0, 0.0, 0.0, -1.0])

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
    return g_body[1:]


def test_prone_gravity_case_derives_not_upside_down() -> None:
    """(a) The LIVE D29 prone case: a clip whose start-window gravity
    measures (0.96, -0.12, 0.24) — true prone — must derive orientation
    offsets whose reconstructed gravity is within the 0.35 rad gate of
    the measured vector, and must NOT reconstruct as upside-down
    (0, 0, 1) (the live failure: (pitch=-0.05, roll=pi) -> (0, 0, +1),
    1.38 rad off)."""
    from sculptor.reference import _quat_from_pitch_roll

    target_g = np.array([0.96, -0.12, 0.24])
    target_g = target_g / np.linalg.norm(target_g)
    # Independently invert (not via the code under test) to build a
    # synthetic mocap quat measuring this gravity.
    pitch = float(np.arcsin(np.clip(target_g[0], -1.0, 1.0)))
    roll = float(np.arctan2(-target_g[1], -target_g[2]))
    start_quat = _quat_from_pitch_roll(pitch, roll)
    assert _oracle_gravity_from_quat(start_quat) == pytest.approx(
        target_g, abs=1e-6)

    clip = _make_getup_clip(with_quat=False)
    fps = float(clip["fps"])
    n = clip["root_pos_z"].shape[0]
    nw = max(2, int(round(0.5 * fps)))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    quat[:nw] = start_quat
    clip["root_quat_wxyz"] = quat
    assert validate_clip(clip) == []

    derived = derive_reference_reset(clip)   # must NOT raise
    lo_p, hi_p = derived["reset_pitch_offset_rad"]
    lo_r, hi_r = derived["reset_roll_offset_rad"]
    mid_pitch = (lo_p + hi_p) / 2.0
    mid_roll = (lo_r + hi_r) / 2.0

    recon_g = _oracle_gravity_from_quat(
        _quat_from_pitch_roll(mid_pitch, mid_roll))
    cos_ang = float(np.clip(np.dot(recon_g, target_g), -1.0, 1.0))
    angle = float(np.arccos(cos_ang))
    assert angle < 0.35, (recon_g, target_g, angle)
    assert not np.allclose(recon_g, [0.0, 0.0, 1.0], atol=0.3), recon_g


def test_supine_fixture_clip_derives_reconstructable_offsets() -> None:
    """(b) Supine regression guard: fixing prone must not break supine.
    The live sit-up fixture clip
    (tests/fixtures/torso_righting_satup/reference_clip.npz) must still
    derive an orientation reconstructing its own measured start-window
    gravity within the gate tolerance."""
    fixture = (Path(__file__).parent / "fixtures" / "torso_righting_satup"
               / "reference_clip.npz")
    clip = load_clip(fixture)
    assert clip.get("root_quat_wxyz") is not None

    quat = np.asarray(clip["root_quat_wxyz"], dtype=np.float64)
    fps = float(clip["fps"])
    nw = max(2, int(0.5 * fps))
    g_frames = np.stack(
        [_oracle_gravity_from_quat(q) for q in quat[:nw]])
    g_mean = g_frames.mean(axis=0)
    g_mean = g_mean / np.linalg.norm(g_mean)
    # Sanity: this fixture really is supine-shaped (measurably NOT
    # upright — a nonzero angle from (0, 0, -1)).
    assert g_mean[2] > -0.95

    from sculptor.reference import _quat_from_pitch_roll

    derived = derive_reference_reset(clip)   # must NOT raise
    lo_p, hi_p = derived["reset_pitch_offset_rad"]
    lo_r, hi_r = derived["reset_roll_offset_rad"]
    mid_pitch = (lo_p + hi_p) / 2.0
    mid_roll = (lo_r + hi_r) / 2.0
    recon_g = _oracle_gravity_from_quat(
        _quat_from_pitch_roll(mid_pitch, mid_roll))
    cos_ang = float(np.clip(np.dot(recon_g, g_mean), -1.0, 1.0))
    angle = float(np.arccos(cos_ang))
    assert angle < 0.35, (recon_g, g_mean, angle)


def test_low_start_upright_torso_derives_near_zero_orientation() -> None:
    """(c) A low-start (getup-archetype BY HEIGHT) clip whose torso is
    already upright at the start window (identity quat throughout) must
    derive near-zero pitch/roll offsets — height alone triggers a low
    start; there is no orientation correction to make."""
    clip = _make_getup_clip(with_quat=False)
    n = clip["root_pos_z"].shape[0]
    clip["root_quat_wxyz"] = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    assert validate_clip(clip) == []

    derived = derive_reference_reset(clip)
    lo_p, hi_p = derived["reset_pitch_offset_rad"]
    lo_r, hi_r = derived["reset_roll_offset_rad"]
    assert abs(lo_p) < 1e-6 and abs(hi_p) < 1e-6
    assert abs(lo_r) < 1e-6 and abs(hi_r) < 1e-6


def test_orientation_gate_raises_on_forced_inconsistent_reconstruction(
    monkeypatch,
) -> None:
    """(e) THE GATE: forced-inconsistent reconstruction (monkeypatched)
    must raise — the honest math never hits this in practice; this
    proves the safety net actually fires when the reconstruction step
    disagrees with the measured gravity."""
    import sculptor.reference as reference_mod

    monkeypatch.setattr(
        reference_mod, "_quat_from_pitch_roll",
        lambda pitch, roll: np.array([1.0, 0.0, 0.0, 0.0]),  # always identity
    )
    clip = _make_getup_clip(with_quat=True)   # lying ~ pi/2 pitch start
    with pytest.raises(ValueError, match="self-consistency gate"):
        derive_reference_reset(clip)
