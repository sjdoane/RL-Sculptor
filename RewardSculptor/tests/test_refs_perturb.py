"""§REFERENCE_TRAJECTORY_PLAN §5.2/§6: `sculptor/refs/perturb.py` — hard
negatives + calibration-ladder rungs derived from a reference clip.
Offline, synthetic clip fixtures only."""
from __future__ import annotations

import numpy as np
import pytest

from sculptor.reference import validate_clip
from sculptor.refs.perturb import (
    complete_then_hold,
    degrade,
    freeze_end,
    freeze_start,
    perturbation_suite,
    root_motion_only,
    segment_shuffle,
    settled_start_hold,
    speed,
    time_reverse,
    truncate,
)


def _clip(T: int = 60, fps: float = 30.0) -> dict:
    t = np.linspace(0.0, 1.0, T)
    z = 0.1 + 0.65 * (1 - np.cos(np.pi * t)) / 2.0
    jp = np.stack([np.linspace(0, 1.0, T), np.linspace(0, -0.5, T)], axis=1)
    xy = np.stack([np.linspace(0, 2.0, T), np.zeros(T)], axis=1)
    quat = np.zeros((T, 4)); quat[:, 0] = 1.0
    return {
        "root_pos_z": z, "fps": fps, "joint_pos": jp,
        "joint_names": ["a", "b"], "root_pos_xy": xy, "root_quat_wxyz": quat,
    }


# ── every perturbation individually ───────────────────────────────────────
def test_time_reverse_reverses_and_validates():
    clip = _clip()
    out = time_reverse(clip)
    assert validate_clip(out) == []
    np.testing.assert_allclose(out["root_pos_z"], clip["root_pos_z"][::-1])
    np.testing.assert_allclose(out["joint_pos"], clip["joint_pos"][::-1])
    assert out["fps"] == clip["fps"]
    assert out["joint_names"] == clip["joint_names"]


def test_freeze_start_holds_first_frame():
    clip = _clip()
    out = freeze_start(clip)
    assert validate_clip(out) == []
    assert np.all(out["root_pos_z"] == clip["root_pos_z"][0])
    assert np.all(out["joint_pos"] == clip["joint_pos"][0])
    assert out["root_pos_z"].shape == clip["root_pos_z"].shape


def test_freeze_end_holds_last_frame():
    clip = _clip()
    out = freeze_end(clip)
    assert validate_clip(out) == []
    assert np.all(out["root_pos_z"] == clip["root_pos_z"][-1])
    assert np.all(out["joint_pos"] == clip["joint_pos"][-1])


def test_complete_then_hold_length_and_hold_frames():
    clip = _clip(T=60)
    out = complete_then_hold(clip)  # hold_frac=1.0 -> hold_n == T
    assert validate_clip(out) == []
    assert out["root_pos_z"].shape[0] == 60 + 60
    # The first T frames are the original clip, verbatim.
    np.testing.assert_allclose(out["root_pos_z"][:60], clip["root_pos_z"])
    np.testing.assert_allclose(out["joint_pos"][:60], clip["joint_pos"])
    # The appended hold frames all equal the clip's LAST frame.
    assert np.all(out["root_pos_z"][60:] == clip["root_pos_z"][-1])
    assert np.all(out["joint_pos"][60:] == clip["joint_pos"][-1])
    assert np.all(out["root_quat_wxyz"][60:] == clip["root_quat_wxyz"][-1])


def test_complete_then_hold_velocities_zero_in_hold():
    clip = _clip(T=60)
    clip["root_vel_z"] = np.gradient(clip["root_pos_z"]) * clip["fps"]
    clip["joint_vel"] = np.gradient(clip["joint_pos"], axis=0) * clip["fps"]
    out = complete_then_hold(clip)
    assert validate_clip(out) == []
    # Original portion is untouched.
    np.testing.assert_allclose(out["root_vel_z"][:60], clip["root_vel_z"])
    np.testing.assert_allclose(out["joint_vel"][:60], clip["joint_vel"])
    # Hold portion is exactly zero, not the last frame's (possibly nonzero)
    # velocity — a held terminal state is not still moving.
    assert np.all(out["root_vel_z"][60:] == 0.0)
    assert np.all(out["joint_vel"][60:] == 0.0)


def test_complete_then_hold_frac_half():
    clip = _clip(T=60)
    out = complete_then_hold(clip, hold_frac=0.5)
    assert validate_clip(out) == []
    assert out["root_pos_z"].shape[0] == 60 + 30
    assert np.all(out["root_pos_z"][60:] == clip["root_pos_z"][-1])


def test_complete_then_hold_distinct_from_freeze_end():
    clip = _clip(T=60)
    cth = complete_then_hold(clip)
    fe = freeze_end(clip)
    # Different lengths (cth = 2T, freeze_end = T) and different content:
    # complete_then_hold performs the transition first, freeze_end never
    # transitions at all.
    assert cth["root_pos_z"].shape[0] != fe["root_pos_z"].shape[0]
    assert not np.allclose(cth["root_pos_z"][0], fe["root_pos_z"][0]) or (
        clip["root_pos_z"][0] == clip["root_pos_z"][-1])
    # cth starts at the clip's FIRST frame value; freeze_end starts at the
    # clip's LAST frame value (assuming the clip actually transitions).
    assert cth["root_pos_z"][0] == pytest.approx(clip["root_pos_z"][0])
    assert fe["root_pos_z"][0] == pytest.approx(clip["root_pos_z"][-1])


def test_complete_then_hold_not_in_perturbation_suite():
    clip = _clip()
    suite = perturbation_suite(clip)
    assert "complete_then_hold" not in suite


# ── §D24 F2: settled_start_hold ─────────────────────────────────────────
def test_settled_start_hold_prepends_settled_z_and_preserves_the_rest():
    clip = _clip(T=60)
    out, adjusted = settled_start_hold(clip, z=0.05, hold_s=0.5)  # 15 frames @30fps
    assert validate_clip(out) == []
    assert adjusted is False  # no pitch/roll given -> z-only
    hold_n = 15
    assert out["root_pos_z"].shape[0] == 60 + hold_n
    assert np.all(out["root_pos_z"][:hold_n] == pytest.approx(0.05))
    # Everything AFTER the prepend is the clip UNCHANGED.
    np.testing.assert_allclose(out["root_pos_z"][hold_n:], clip["root_pos_z"])
    np.testing.assert_allclose(out["joint_pos"][hold_n:], clip["joint_pos"])


def test_settled_start_hold_velocities_zero_in_prepend():
    clip = _clip(T=60)
    clip["root_vel_z"] = np.gradient(clip["root_pos_z"]) * clip["fps"]
    clip["joint_vel"] = np.gradient(clip["joint_pos"], axis=0) * clip["fps"]
    out, _ = settled_start_hold(clip, z=0.05, hold_s=0.5)
    hold_n = 15
    assert np.all(out["root_vel_z"][:hold_n] == 0.0)
    assert np.all(out["joint_vel"][:hold_n] == 0.0)
    # Original portion untouched.
    np.testing.assert_allclose(out["root_vel_z"][hold_n:], clip["root_vel_z"])


def test_settled_start_hold_z_only_leaves_orientation_at_frame_zero():
    """No pitch/roll given -> the prepend's orientation stays frame 0's
    own value (still a valid, if less complete, variant)."""
    clip = _clip(T=60)
    out, adjusted = settled_start_hold(clip, z=0.05)
    assert adjusted is False
    hold_n = 15
    assert np.all(out["root_quat_wxyz"][:hold_n] == clip["root_quat_wxyz"][0])


def test_settled_start_hold_with_pitch_roll_adjusts_orientation():
    from sculptor.reference import _quat_from_pitch_roll

    clip = _clip(T=60)
    out, adjusted = settled_start_hold(clip, z=0.05, pitch=0.3, roll=0.1)
    assert adjusted is True
    hold_n = 15
    expected_q = _quat_from_pitch_roll(0.3, 0.1)
    np.testing.assert_allclose(out["root_quat_wxyz"][0], expected_q)
    # NOT the clip's own frame-0 quat (identity here) — genuinely adjusted.
    assert not np.allclose(out["root_quat_wxyz"][0], clip["root_quat_wxyz"][0])
    # Original portion (after the prepend) is untouched.
    np.testing.assert_allclose(
        out["root_quat_wxyz"][hold_n:], clip["root_quat_wxyz"])


def test_settled_start_hold_missing_pitch_or_roll_is_z_only():
    """BOTH pitch and roll must be given — a partial pair also falls
    back to z-only (never guesses the missing half)."""
    clip = _clip(T=60)
    out, adjusted = settled_start_hold(clip, z=0.05, pitch=0.3, roll=None)
    assert adjusted is False
    hold_n = 15
    assert np.all(out["root_quat_wxyz"][:hold_n] == clip["root_quat_wxyz"][0])


def test_settled_start_hold_no_quat_channel_is_z_only():
    clip = {"root_pos_z": np.linspace(0.1, 0.7, 60), "fps": 30.0}
    out, adjusted = settled_start_hold(clip, z=0.05, pitch=0.3, roll=0.1)
    assert adjusted is False
    assert "root_quat_wxyz" not in out


def test_settled_start_hold_is_a_positive_exemplar_not_a_perturbation():
    """Not added to `perturbation_suite()` — it's a caller-constructed
    positive exemplar, same convention as `complete_then_hold`."""
    clip = _clip()
    suite = perturbation_suite(clip)
    assert "settled_start" not in suite
    assert "settled_start_hold" not in suite


def test_segment_shuffle_is_deterministic_per_seed_and_preserves_length():
    clip = _clip()
    a = segment_shuffle(clip, n_segments=6, seed=0)
    b = segment_shuffle(clip, n_segments=6, seed=0)
    c = segment_shuffle(clip, n_segments=6, seed=1)
    assert validate_clip(a) == []
    np.testing.assert_array_equal(a["root_pos_z"], b["root_pos_z"])
    assert a["root_pos_z"].shape == clip["root_pos_z"].shape
    # A different seed almost certainly gives a different order.
    assert not np.array_equal(a["root_pos_z"], c["root_pos_z"])
    # It's a REORDERING, not new data: same multiset of frame values.
    np.testing.assert_allclose(
        np.sort(a["root_pos_z"]), np.sort(clip["root_pos_z"]))


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_truncate_lengths(frac):
    clip = _clip(T=80)
    out = truncate(clip, frac)
    assert validate_clip(out) == []
    expected = max(2, round(80 * frac))
    assert out["root_pos_z"].shape[0] == expected
    np.testing.assert_allclose(out["root_pos_z"], clip["root_pos_z"][:expected])


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_speed_resampling_lengths(factor):
    clip = _clip(T=80)
    # Give the source a velocity channel so the value assertion below
    # actually exercises the recompute path (fixture omits it).
    clip["root_vel_z"] = np.gradient(clip["root_pos_z"]) * clip["fps"]
    out = speed(clip, factor)
    assert validate_clip(out) == []
    expected = max(10, round(80 / factor))
    assert out["root_pos_z"].shape[0] == expected
    assert out["fps"] == clip["fps"]
    # Endpoints preserved (interpolation is exact at the boundaries).
    assert out["root_pos_z"][0] == pytest.approx(clip["root_pos_z"][0])
    assert out["root_pos_z"][-1] == pytest.approx(clip["root_pos_z"][-1])
    # Velocity VALUES scale by `factor` (audit 2026-07-09: a dt_scale
    # double-count once produced factor^2 speeds — 16.9 m/s at 4x).
    if "root_vel_z" in out:
        src_peak = float(np.abs(clip["root_vel_z"]).max())
        out_peak = float(np.abs(out["root_vel_z"]).max())
        assert out_peak == pytest.approx(src_peak * factor, rel=0.15)


def test_speed_rejects_nonpositive_factor():
    with pytest.raises(ValueError):
        speed(_clip(), 0.0)


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_speed_survives_quat_hemisphere_flips(factor):
    # Retargeted clips carry q/-q sign flips between frames (same rotation,
    # opposite hemisphere). Raw component-wise lerp across a flip passes
    # near zero norm and failed validate_clip, which blanked the ENTIRE
    # perturbation suite on a real fleaven clip (D14). speed() must
    # hemisphere-align + renormalize.
    clip = _clip(T=80)
    q = clip["root_quat_wxyz"].copy()
    q[1::2] *= -1.0  # alternate hemispheres every other frame
    clip["root_quat_wxyz"] = q
    out = speed(clip, factor)
    assert validate_clip(out) == []
    norms = np.linalg.norm(out["root_quat_wxyz"], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_degrade_adds_joint_noise_and_damps_root():
    clip = _clip()
    out = degrade(clip, joint_noise_rad=0.1, root_damp=0.0, seed=1)
    assert validate_clip(out) == []
    # root_damp=0 freezes root motion at the start value.
    np.testing.assert_allclose(out["root_pos_z"], clip["root_pos_z"][0], atol=1e-9)
    np.testing.assert_allclose(
        out["root_pos_xy"],
        np.broadcast_to(clip["root_pos_xy"][0], out["root_pos_xy"].shape),
        atol=1e-9)
    # Joint noise perturbs joint_pos (not identical to the source).
    assert not np.allclose(out["joint_pos"], clip["joint_pos"])


def test_degrade_no_noise_no_damp_is_near_identity():
    clip = _clip()
    out = degrade(clip, joint_noise_rad=0.0, root_damp=1.0, seed=0)
    assert validate_clip(out) == []
    np.testing.assert_allclose(out["root_pos_z"], clip["root_pos_z"])
    np.testing.assert_allclose(out["joint_pos"], clip["joint_pos"])


def test_degrade_rejects_bad_params():
    with pytest.raises(ValueError):
        degrade(_clip(), root_damp=1.5)
    with pytest.raises(ValueError):
        degrade(_clip(), joint_noise_rad=-1.0)


# ── the standard named suite ──────────────────────────────────────────────
def test_perturbation_suite_has_the_standard_names_and_all_validate():
    clip = _clip()  # carries joint_pos + root_quat_wxyz -> root_only present
    suite = perturbation_suite(clip)
    assert set(suite) == {
        "reversal", "freeze_start", "freeze_end", "shuffle", "root_only",
        "speed_slow", "speed_fast", "trunc_25", "trunc_50", "trunc_75",
    }
    for name, pclip in suite.items():
        assert validate_clip(pclip) == [], (name, validate_clip(pclip))


def test_perturbation_suite_omits_root_only_without_posture_channels():
    # D18 guard: for a root-channels-only clip root_only would EQUAL the
    # original clip and falsely convict every metric.
    clip = {"root_pos_z": _clip()["root_pos_z"], "fps": 30.0}
    suite = perturbation_suite(clip)
    assert "root_only" not in suite


def test_root_motion_only_freezes_posture_keeps_root():
    clip = _clip(T=60)
    out = root_motion_only(clip)
    assert validate_clip(out) == []
    np.testing.assert_allclose(out["root_pos_z"], clip["root_pos_z"])
    # posture frozen at frame 0
    assert np.allclose(out["joint_pos"], clip["joint_pos"][0])
    assert np.allclose(out["root_quat_wxyz"], clip["root_quat_wxyz"][0])


def test_perturbation_suite_is_deterministic():
    clip = _clip()
    a = perturbation_suite(clip)
    b = perturbation_suite(clip)
    for name in a:
        np.testing.assert_array_equal(a[name]["root_pos_z"], b[name]["root_pos_z"])


def test_fast_completion_shape_and_defaults():
    """16x speed + 19x hold: motion ~5% of the padded trajectory — the
    exact live D24 rollout profile (0.5 s righting in a 10 s episode)."""
    from sculptor.refs.perturb import fast_completion, speed

    clip = _clip()
    out = fast_completion(clip)
    sped = speed(clip, 16.0)
    n_motion = len(sped["root_pos_z"])
    assert len(out["root_pos_z"]) == n_motion + max(2, int(round(n_motion * 19.0)))
    # hold frames pinned at the sped clip's final value, velocities zero
    import numpy as np
    assert np.allclose(out["root_pos_z"][n_motion:], sped["root_pos_z"][-1])
    if "root_vel_z" in out:
        assert np.allclose(out["root_vel_z"][n_motion:], 0.0)


def test_ends_settled_classifier():
    import numpy as np
    from sculptor.refs.perturb import ends_settled

    fps = 30.0
    still_tail = {"root_pos_z": np.concatenate(
        [np.linspace(0.1, 0.7, 60), np.full(30, 0.7)]), "fps": fps}
    moving_tail = {"root_pos_z": np.linspace(0.1, 0.7, 90), "fps": fps}
    assert ends_settled(still_tail) is True
    assert ends_settled(moving_tail) is False
