"""§start_pose items 4 + 5: `check_start_pose_compatibility` (clip-shape
<-> authored-start_pose QC) and `settle_reset` (§D20a settle-then-
rederive).

Covers:
  * archetype-vs-start_pose compatibility rules (supine/prone -> getup;
    sitting/crouched -> getup or mid_start; standing -> neither);
  * supine-vs-prone orientation discrimination from `root_quat_wxyz`,
    including the "no quat -> accept either" fallback;
  * actionable mismatch messages;
  * `_quat_from_pitch_roll` round-tripping through the existing
    pitch/roll EXTRACTION functions (pure math, no physics);
  * `_recenter_train_ranges_on_settled` (pure math, no physics);
  * `settle_reset` against the REAL G1 MJCF (CPU mujoco) — a genuine
    physics settle, kept under the ~10s budget.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sculptor.reference import (
    G1_CLASS_STAND_M,
    SettleUnavailable,
    _body_frame_gravity_x,
    _quat_from_pitch_roll,
    _quat_wxyz_to_pitch_rad,
    _quat_wxyz_to_roll_rad,
    _recenter_train_ranges_on_settled,
    check_start_pose_compatibility,
    derive_eval_reset,
    settle_reset,
    validate_clip,
)


# ── Synthetic clip builders (mirrors test_reference_rsi_general.py's
#    conventions, kept local — this file focuses on QC/settle, not
#    derivation coverage, which already lives there) ─────────────────
def _flat_clip(z: float, *, fps: float = 50.0, quat: "np.ndarray | None" = None) -> dict:
    n = 60
    clip: dict = {"root_pos_z": np.full(n, z), "fps": fps}
    if quat is not None:
        clip["root_quat_wxyz"] = np.tile(quat, (n, 1))
    assert validate_clip(clip) == []
    return clip


def _getup_clip(*, quat: "np.ndarray | None" = None) -> dict:
    """Low start (getup archetype) -> ramp -> standing, mirrors
    test_reference_rsi_general.py's `_make_getup_clip` shape."""
    fps = 50.0

    def seg(dur, fn):
        n = max(2, int(round(dur * fps)))
        return fn(np.linspace(0.0, 1.0, n, endpoint=False))

    lying = seg(0.5, lambda s: np.full_like(s, 0.10))
    ramp = seg(1.0, lambda s: 0.10 + (0.75 - 0.10) * s)
    stand = seg(0.5, lambda s: np.full_like(s, 0.75))
    z = np.concatenate([lying, ramp, stand])
    clip: dict = {"root_pos_z": z, "fps": fps}
    if quat is not None:
        clip["root_quat_wxyz"] = np.tile(quat, (z.shape[0], 1))
    assert validate_clip(clip) == []
    return clip


def _mid_start_clip() -> dict:
    return _flat_clip(0.42)


def _standing_flat_clip() -> dict:
    return _flat_clip(0.78)


def _airborne_clip() -> dict:
    from sculptor.reference import make_procedural_jump_clip

    return make_procedural_jump_clip()


# Pitch offsets that produce a clean prone / supine lying pose — same
# convention `_quat_from_pitch_roll` and `_body_frame_gravity_x` use
# (verified numerically in this file's own pinned test below).
_PRONE_QUAT = _quat_from_pitch_roll(math.pi / 2, 0.0)
_SUPINE_QUAT = _quat_from_pitch_roll(-math.pi / 2, 0.0)


# ── _body_frame_gravity_x sign convention (pinned) ────────────────────
def test_body_frame_gravity_x_prone_vs_supine() -> None:
    assert _body_frame_gravity_x(_PRONE_QUAT) == pytest.approx(1.0, abs=1e-6)
    assert _body_frame_gravity_x(_SUPINE_QUAT) == pytest.approx(-1.0, abs=1e-6)


def test_body_frame_gravity_x_standing_is_near_zero() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert _body_frame_gravity_x(identity) == pytest.approx(0.0, abs=1e-9)


# ── archetype <-> start_pose compatibility ─────────────────────────────
def test_none_start_pose_is_a_noop() -> None:
    check_start_pose_compatibility(_standing_flat_clip(), None)  # no raise


@pytest.mark.parametrize("pose", ["supine", "prone"])
def test_supine_prone_require_getup_archetype(pose: str) -> None:
    check_start_pose_compatibility(_getup_clip(), pose)  # no raise
    for bad_clip in (_mid_start_clip(), _standing_flat_clip(), _airborne_clip()):
        with pytest.raises(ValueError, match="wrong clip attached or wrong start_pose"):
            check_start_pose_compatibility(bad_clip, pose)


@pytest.mark.parametrize("pose", ["sitting", "crouched"])
def test_sitting_crouched_accept_getup_or_mid_start(pose: str) -> None:
    check_start_pose_compatibility(_getup_clip(), pose)        # no raise
    check_start_pose_compatibility(_mid_start_clip(), pose)    # no raise
    with pytest.raises(ValueError, match="wrong clip attached or wrong start_pose"):
        check_start_pose_compatibility(_standing_flat_clip(), pose)


def test_standing_rejects_getup_and_mid_start() -> None:
    with pytest.raises(ValueError, match="wrong clip attached or wrong start_pose"):
        check_start_pose_compatibility(_getup_clip(), "standing")
    with pytest.raises(ValueError, match="wrong clip attached or wrong start_pose"):
        check_start_pose_compatibility(_mid_start_clip(), "standing")


def test_standing_accepts_non_low_archetypes() -> None:
    check_start_pose_compatibility(_standing_flat_clip(), "standing")
    check_start_pose_compatibility(_airborne_clip(), "standing")


def test_unrecognized_start_pose_raises() -> None:
    with pytest.raises(ValueError, match="not a recognized value"):
        check_start_pose_compatibility(_standing_flat_clip(), "levitating")


def test_mismatch_message_includes_clip_id() -> None:
    with pytest.raises(ValueError, match="my_clip_123"):
        check_start_pose_compatibility(
            _standing_flat_clip(), "supine", clip_id="my_clip_123")


# ── supine/prone orientation discrimination via root_quat_wxyz ────────
def test_supine_prone_quat_discrimination_matches() -> None:
    check_start_pose_compatibility(_getup_clip(quat=_PRONE_QUAT), "prone")
    check_start_pose_compatibility(_getup_clip(quat=_SUPINE_QUAT), "supine")


def test_supine_prone_quat_discrimination_mismatch_raises() -> None:
    """The archetype passes (it's a getup clip) but the ORIENTATION
    contradicts the authored start_pose — must still raise."""
    with pytest.raises(ValueError, match="measures as 'prone'"):
        check_start_pose_compatibility(_getup_clip(quat=_PRONE_QUAT), "supine")
    with pytest.raises(ValueError, match="measures as 'supine'"):
        check_start_pose_compatibility(_getup_clip(quat=_SUPINE_QUAT), "prone")


def test_supine_prone_no_quat_accepts_either() -> None:
    """Orientation is simply unknown without root_quat_wxyz — accept
    either supine or prone (archetype check alone still applies)."""
    clip = _getup_clip()
    assert clip.get("root_quat_wxyz") is None
    check_start_pose_compatibility(clip, "supine")
    check_start_pose_compatibility(clip, "prone")


# ── _quat_from_pitch_roll <-> extraction round-trip (pure math) ──────
@pytest.mark.parametrize(
    "roll,pitch",
    [
        (0.0, 0.0),
        (0.3, 0.5),
        (-0.7, 1.2),
        (1.5, -0.4),
        (0.0, math.pi / 2),
        (0.0, -math.pi / 2),
        (math.pi / 2, 0.0),
        (-math.pi / 2, 0.0),
        (2.0, 0.0),
    ],
)
def test_quat_from_pitch_roll_roundtrips_through_extraction(
    roll: float, pitch: float,
) -> None:
    q = _quat_from_pitch_roll(pitch, roll)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
    recovered_pitch = float(_quat_wxyz_to_pitch_rad(q[None, :])[0])
    recovered_roll = float(_quat_wxyz_to_roll_rad(q[None, :])[0])
    assert recovered_pitch == pytest.approx(pitch, abs=1e-6)
    assert recovered_roll == pytest.approx(roll, abs=1e-6)


# ── _recenter_train_ranges_on_settled (pure math, no physics) ─────────
def test_recenter_keeps_width_and_moves_center() -> None:
    full = {
        "reset_height_offset_m": [-0.5, -0.3],  # width 0.2, center -0.4
        "reset_pitch_offset_rad": [-2.2, -1.8],  # width 0.4, center -2.0
        "min_base_height_termination_m": 0.05,  # untouched key
    }
    settled = {
        "reset_height_offset_m": -0.6,   # new center
        "reset_pitch_offset_rad": -1.0,  # new center
    }
    out = _recenter_train_ranges_on_settled(full, settled)
    lo, hi = out["reset_height_offset_m"]
    assert hi - lo == pytest.approx(0.2, abs=1e-6)
    assert (lo + hi) / 2.0 == pytest.approx(-0.6, abs=1e-4)
    p_lo, p_hi = out["reset_pitch_offset_rad"]
    assert p_hi - p_lo == pytest.approx(0.4, abs=1e-6)
    assert (p_lo + p_hi) / 2.0 == pytest.approx(-1.0, abs=1e-4)
    # Untouched key passes through.
    assert out["min_base_height_termination_m"] == 0.05


def test_recenter_clamps_through_validator_bounds() -> None:
    from sculptor.env_spec import _TRAIN_RANGES

    lo_bound, hi_bound = _TRAIN_RANGES["reset_height_offset_m"]
    full = {"reset_height_offset_m": [lo_bound + 0.1, lo_bound + 0.3]}
    # A pathological settled center way outside the envelope must clamp,
    # never propagate an out-of-bounds value.
    settled = {"reset_height_offset_m": lo_bound - 100.0}
    out = _recenter_train_ranges_on_settled(full, settled)
    lo, hi = out["reset_height_offset_m"]
    assert lo_bound <= lo <= hi_bound
    assert lo_bound <= hi <= hi_bound


def test_recenter_replaces_joint_target_and_clamps() -> None:
    from sculptor.env_spec import _JOINT_TARGET_ELEMENT_BOUNDS

    full = {"reset_joint_pos_target": [1.0, -0.8, 0.5]}
    settled = {"reset_joint_pos_target": [1.1, -0.9, 99.0]}
    out = _recenter_train_ranges_on_settled(full, settled)
    lo, hi = _JOINT_TARGET_ELEMENT_BOUNDS
    assert out["reset_joint_pos_target"][0] == pytest.approx(1.1, abs=1e-4)
    assert out["reset_joint_pos_target"][1] == pytest.approx(-0.9, abs=1e-4)
    assert out["reset_joint_pos_target"][2] == pytest.approx(hi, abs=1e-4)


def test_recenter_leaves_keys_absent_from_settled_untouched() -> None:
    full = {"reset_height_offset_m": [-0.5, -0.3],
            "reset_pitch_offset_rad": [-1.0, -0.5]}
    settled = {"reset_height_offset_m": -0.6}  # no pitch key
    out = _recenter_train_ranges_on_settled(full, settled)
    assert out["reset_pitch_offset_rad"] == [-1.0, -0.5]


# ── settle_reset: cheap failure path (no physics) ──────────────────────
def test_settle_reset_unavailable_for_unknown_robot() -> None:
    with pytest.raises(SettleUnavailable):
        settle_reset(
            {"reset_height_offset_m": -0.5}, robot="no_such_robot_xyz")


# ── settle_reset: real physics against the G1 MJCF (CPU mujoco) ───────
#: Real G1 canonical joint order (`sculptor.eval.robot_manifest`), used
#: to build a PHYSICALLY PLAUSIBLE crouch-forward posture for the
#: physics tests below — a bent-knee/bent-hip target (real joint names,
#: resolvable by `model.joint(name)`) is what every real reference clip
#: in the library actually carries; a straight-leg all-zero default
#: (the "no reset_joint_pos_target at all" case) is what
#: `_SETTLE_MAX_PLAUSIBLE_DELTA_M` exists to CATCH, exercised by the
#: dedicated divergence test further down.
def _g1_joint_names() -> list:
    from sculptor.eval.robot_manifest import robot_joint_names

    return robot_joint_names("Mjlab-Velocity-Flat-Unitree-G1")


def _crouch_forward_joint_target(names: list) -> list:
    target = {n: 0.0 for n in names}
    target["left_hip_pitch_joint"] = -0.5
    target["right_hip_pitch_joint"] = -0.5
    target["left_knee_joint"] = 1.0
    target["right_knee_joint"] = 1.0
    target["left_ankle_pitch_joint"] = -0.4
    target["right_ankle_pitch_joint"] = -0.4
    return [target[n] for n in names]


def test_settle_reset_real_g1_model_realistic_crouch() -> None:
    """A physically plausible crouch-forward reset (real 29-joint
    target, matching what every real reference clip in the library
    carries) settled on the REAL G1 MJCF: must return the documented
    dict shape without raising, well under the ~10s budget (a few
    hundred ms of actual mj_step wall time once mujoco is already
    imported), and land within the plausibility bound."""
    names = _g1_joint_names()
    reset_scalars = {
        "reset_height_offset_m": -0.3,
        "reset_pitch_offset_rad": 0.5,
        "reset_vertical_velocity_mps": 0.0,
        "reset_joint_pos_target": _crouch_forward_joint_target(names),
    }

    result = settle_reset(
        reset_scalars, joint_names=names, robot="g1", settle_time_s=0.5)
    assert set(result) == {"scalars", "delta_z_m", "steps", "converged", "duration_s"}
    assert result["steps"] > 0
    assert isinstance(result["converged"], bool)
    assert abs(result["delta_z_m"]) < 1.0    # sags/rises, but plausibly

    scalars = result["scalars"]
    settled_z = G1_CLASS_STAND_M + scalars["reset_height_offset_m"]
    assert np.isfinite(settled_z)
    assert -0.5 < settled_z < 1.5
    assert "reset_pitch_offset_rad" in scalars
    assert len(scalars["reset_joint_pos_target"]) == len(names)


def test_settle_reset_unresolvable_joint_names_are_skipped_not_fatal() -> None:
    """Joint names that don't exist on the real G1 model (a synthetic
    clip's `joint_names` like "knee"/"hip" rather than the robot's real
    `left_knee_joint` etc.) must be silently skipped at the qpos-write
    step, never raise a KeyError — mirrors `refs/preview.py`'s
    per-frame posing discipline. Uses a MILD offset (not the
    crouch-forward posture above) since unresolved names mean every
    joint falls back to its straight-leg default, which is only
    plausible for a mild pose (a large offset with no real joint
    target legitimately trips `_SETTLE_MAX_PLAUSIBLE_DELTA_M` — see the
    dedicated divergence test)."""
    reset_scalars = {
        "reset_height_offset_m": -0.1,
        "reset_pitch_offset_rad": 0.2,
        "reset_joint_pos_target": [0.5, -0.3],
    }
    result = settle_reset(
        reset_scalars, joint_names=["knee", "hip"], robot="g1",
        settle_time_s=0.3)
    # Values pass through unresolved (best-effort), never crash.
    assert len(result["scalars"]["reset_joint_pos_target"]) == 2


def test_settle_reset_raises_on_implausible_explosion() -> None:
    """§robustness fix: a lying pitch/height offset with NO
    `reset_joint_pos_target` (every joint at its straight-leg default)
    interpenetrates the floor badly enough to trigger a MuJoCo
    contact-force explosion — a FINITE but wildly implausible result,
    not a `mj_step` exception. Found empirically while building this
    function (a real observed case launched the pelvis ~5 m into the
    air in well under a second). `settle_reset` must convert this into
    an explicit `SettleUnavailable`, never silently return the exploded
    scalars as if they were a genuine settle."""
    with pytest.raises(SettleUnavailable, match="implausible"):
        settle_reset(
            {"reset_height_offset_m": -0.3, "reset_pitch_offset_rad": 0.0},
            robot="g1", settle_time_s=0.5)
