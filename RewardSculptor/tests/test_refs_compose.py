"""tests/test_refs_compose.py — composing novel motions from solved clips.

`sculptor.refs.compose` is the module that lifts the library from
"retrieve one clip" to "build a motion nobody recorded out of spans that were
each solved separately". Its failure modes are quiet ones — a teleporting
root, silently swapped limbs, velocities inherited from the wrong source —
so these tests pin the continuity guarantees rather than just the happy path.
"""
from __future__ import annotations

import numpy as np
import pytest

from sculptor.reference import validate_clip
from sculptor.refs.compose import (
    ComposeError,
    _quat_yaw,
    compose_reference,
    seam_report,
)

FPS = 60.0
J = 4
JOINTS = [f"joint_{i}" for i in range(J)]


def _clip(
    n: int = 120,
    *,
    fps: float = FPS,
    z0: float = 0.70,
    joint_offset: float = 0.0,
    yaw: float = 0.0,
    xy0: tuple[float, float] = (0.0, 0.0),
    names: list[str] | None = None,
) -> dict:
    """A valid, smooth synthetic clip: gentle sinusoidal joints, steady
    forward travel, constant heading."""
    t = np.arange(n, dtype=np.float64) / fps
    z = z0 + 0.02 * np.sin(2 * np.pi * 0.5 * t)
    jp = (joint_offset
          + 0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
          + 0.01 * np.arange(J)[None, :])
    xy = np.stack([xy0[0] + 0.5 * t * np.cos(yaw),
                   xy0[1] + 0.5 * t * np.sin(yaw)], axis=1)
    quat = np.tile(
        np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), (n, 1))
    return {
        "fps": fps,
        "joint_names": list(names or JOINTS),
        "root_pos_z": z,
        "root_vel_z": np.gradient(z, 1.0 / fps),
        "root_pos_xy": xy,
        "root_quat_wxyz": quat,
        "joint_pos": jp,
        "joint_vel": np.gradient(jp, 1.0 / fps, axis=0),
        "contact_left_foot": (np.sin(2 * np.pi * t) > 0).astype(np.float64),
        "contact_right_foot": (np.sin(2 * np.pi * t) <= 0).astype(np.float64),
        "meta": {"clip_id": "synthetic"},
    }


# ── core composition ────────────────────────────────────────────────────
def test_composes_two_clips_into_a_valid_clip():
    out = compose_reference([
        {"clip": _clip(), "label": "a", "source_id": "clip_a"},
        {"clip": _clip(joint_offset=0.05), "label": "b", "source_id": "clip_b"},
    ])
    assert validate_clip(out) == []
    comp = out["meta"]["composition"]
    assert [s["label"] for s in comp["segments"]] == ["a", "b"]
    assert [s["source_id"] for s in comp["segments"]] == ["clip_a", "clip_b"]
    # Blend consumes frames from both sides, so the result is shorter than
    # the naive concatenation.
    assert out["root_pos_z"].shape[0] == 240 - comp["blend_frames"]
    # A composite is a candidate, never a certified motion.
    assert comp["certified"] is False


def test_requires_at_least_two_segments():
    with pytest.raises(ComposeError, match=">= 2 segments"):
        compose_reference([{"clip": _clip()}])


def test_rejects_invalid_source_clip():
    bad = _clip()
    bad["root_pos_z"] = bad["root_pos_z"] * -1.0  # violates strictly-positive
    with pytest.raises(ComposeError, match="source clip is invalid"):
        compose_reference([{"clip": bad}, {"clip": _clip()}])


# ── continuity: the guarantees that make a composite trackable ──────────
def test_se2_alignment_removes_the_root_teleport():
    """Segment 2 starts 10 m away and 90 deg off. Without SE(2) alignment the
    root jumps at the seam; with it, position and heading stay continuous."""
    out = compose_reference([
        {"clip": _clip(yaw=0.0, xy0=(0.0, 0.0))},
        {"clip": _clip(yaw=np.pi / 2, xy0=(10.0, -4.0))},
    ], blend_s=0.0)
    xy = out["root_pos_xy"]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # No frame-to-frame jump anywhere near the 10 m raw offset.
    assert step.max() < 0.1, f"root teleported {step.max():.3f} m at a seam"
    yaw = np.unwrap(_quat_yaw(out["root_quat_wxyz"]))
    assert np.abs(np.diff(yaw)).max() < 0.1


def test_velocities_are_recomputed_from_composed_positions():
    """Inherited velocities are wrong after blending/resampling. The output's
    joint_vel must be the finite difference of its OWN joint_pos."""
    out = compose_reference([
        {"clip": _clip()}, {"clip": _clip(joint_offset=0.05)},
    ])
    expected = np.gradient(out["joint_pos"], 1.0 / float(out["fps"]), axis=0)
    assert np.allclose(out["joint_vel"], expected)
    expected_z = np.gradient(out["root_pos_z"], 1.0 / float(out["fps"]))
    assert np.allclose(out["root_vel_z"], expected_z)


def test_blend_does_not_inject_a_velocity_spike():
    """The smoothstep window exists so the blend is C1. A linear ramp would
    step the velocity at both ends of the window; assert we do not."""
    out = compose_reference([
        {"clip": _clip()}, {"clip": _clip(joint_offset=0.6)},
    ], blend_s=0.5)
    jv = np.abs(out["joint_vel"])
    # No isolated frame carries a wildly larger velocity than its neighbours.
    assert jv.max() < 12.0, f"velocity spike {jv.max():.2f} rad/s at the blend"


def test_contacts_are_never_averaged():
    """A 0.5 foot contact is not a physical state."""
    out = compose_reference([{"clip": _clip()}, {"clip": _clip()}])
    for key in ("contact_left_foot", "contact_right_foot"):
        assert set(np.unique(out[key])) <= {0.0, 1.0}


# ── joint-set handling ──────────────────────────────────────────────────
def test_reorders_joints_by_name():
    """Same joint SET, different ORDER. Composing positionally would swap
    limbs — a corruption that still validates cleanly."""
    shuffled = list(reversed(JOINTS))
    b = _clip(names=shuffled)
    b["joint_pos"] = b["joint_pos"][:, ::-1].copy()
    b["joint_vel"] = b["joint_vel"][:, ::-1].copy()
    out = compose_reference([{"clip": _clip()}, {"clip": b}], blend_s=0.0)
    assert out["joint_names"] == JOINTS
    # Segment 1's per-joint offsets now match segment 0's ordering, so the
    # seam is smooth rather than a limb-swap discontinuity.
    seam = out["meta"]["composition"]["seam_frames"][0]
    jump = np.abs(out["joint_pos"][seam] - out["joint_pos"][seam - 1]).max()
    assert jump < 0.05, f"limb swap at seam: {jump:.3f} rad"


def test_mismatched_joint_sets_are_rejected_loudly():
    other = _clip(names=["a", "b", "c", "d"])
    with pytest.raises(ComposeError, match="joint set does not match"):
        compose_reference([{"clip": _clip()}, {"clip": other}])


# ── resampling ──────────────────────────────────────────────────────────
def test_harmonizes_mixed_frame_rates():
    """Real library clips are a mix of 60 and 120 fps."""
    out = compose_reference([
        {"clip": _clip(n=120, fps=60.0)},
        {"clip": _clip(n=240, fps=120.0)},
    ])
    assert float(out["fps"]) == 120.0
    assert validate_clip(out) == []
    # Both segments contribute ~2 s at the common rate.
    assert out["root_pos_z"].shape[0] == pytest.approx(480, abs=40)


def test_explicit_target_fps_is_honored():
    out = compose_reference(
        [{"clip": _clip()}, {"clip": _clip()}], target_fps=30.0)
    assert float(out["fps"]) == 30.0
    assert validate_clip(out) == []


# ── strict QC: refuse composites that would waste a Tier-D run ──────────
def test_rejects_a_seam_that_does_not_meet():
    """Spans whose boundary poses are far apart cannot be blended into a
    plausible motion; failing here is far cheaper than failing in physics."""
    with pytest.raises(ComposeError, match="seam discontinuity"):
        compose_reference([
            {"clip": _clip(joint_offset=0.0)},
            {"clip": _clip(joint_offset=5.0)},
        ], blend_s=0.0, max_seam_joint_jump_rad=0.35)


def test_strict_false_returns_the_composite_with_its_measurements():
    """Non-strict still MEASURES the damage — it just does not refuse."""
    out = compose_reference([
        {"clip": _clip(joint_offset=0.0)},
        {"clip": _clip(joint_offset=5.0)},
    ], blend_s=0.0, strict=False)
    worst = max(s["max_joint_jump_rad"]
                for s in out["meta"]["composition"]["seam_report"]["seams"])
    assert worst > 0.35


def test_rejects_excessive_joint_velocity():
    with pytest.raises(ComposeError, match="peak joint velocity"):
        compose_reference([
            {"clip": _clip()}, {"clip": _clip(joint_offset=1.0)},
        ], blend_s=0.05, max_seam_joint_jump_rad=99.0,
            max_joint_vel_rad_s=1.0)


# ── provenance ──────────────────────────────────────────────────────────
def test_provenance_traces_every_frame_to_real_source_data():
    out = compose_reference([
        {"clip": _clip(), "t_start_s": 0.5, "t_end_s": 1.5,
         "label": "approach", "source_id": "clip_a"},
        {"clip": _clip(), "t_start_s": 0.2, "t_end_s": 1.2,
         "label": "strike", "source_id": "clip_b"},
    ])
    segments = out["meta"]["composition"]["segments"]
    assert segments[0]["source_span_s"] == [0.5, 1.5]
    assert segments[1]["source_span_s"] == [0.2, 1.2]
    assert segments[0]["source_frames"] == [30, 90]
    for seg in segments:
        assert "se2" in seg and "d_yaw_rad" in seg["se2"]
    # The honesty note must survive into the artifact.
    assert "UNVERIFIED" in out["meta"]["composition"]["note"]


def test_seam_report_measures_rather_than_asserts():
    out = compose_reference([{"clip": _clip()}, {"clip": _clip()}])
    report = seam_report(out, out["meta"]["composition"]["seam_frames"])
    assert report["peak_joint_vel_rad_s"] is not None
    assert report["duration_s"] > 0
    assert len(report["seams"]) == 1
