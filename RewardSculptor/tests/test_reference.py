"""§Reference trajectories (sculptor/reference.py): clip format,
procedural jump generator, phase keyframes, and the RSI derivation onto
the env-spec TRAIN surface. Offline only — no GPU, no LLM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sculptor.env_spec import validate_env_spec
from sculptor.reference import (
    apply_reference_rsi,
    derive_rsi_train_keys,
    load_clip,
    make_procedural_jump_clip,
    phase_keyframes,
    save_clip,
    validate_clip,
)


# ── Procedural generator ────────────────────────────────────────────────
def test_procedural_clip_validates_and_hits_targets() -> None:
    clip = make_procedural_jump_clip(apex_gain_m=0.35)
    assert validate_clip(clip) == []
    kf = phase_keyframes(clip)
    # Discretization tolerance: one 50 fps frame of ballistic drift.
    assert kf["apex_gain_m"] == pytest.approx(0.35, abs=0.04)
    assert kf["takeoff_vz_mps"] == pytest.approx(
        math.sqrt(2 * 9.81 * 0.35), rel=0.10)
    assert kf["crouch_depth_m"] == pytest.approx(0.78 * (1 - 0.62), abs=0.02)
    assert kf["flight_time_s"] > 0.3
    assert len(kf["keyframes"]) == 8
    # Keyframes are (t, z, vz) samples spanning the whole clip.
    assert kf["keyframes"][0]["t"] == 0.0
    assert kf["keyframes"][-1]["t"] == pytest.approx(
        kf["duration_s"], abs=0.05)


def test_procedural_generator_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="crouch_frac"):
        make_procedural_jump_clip(crouch_frac=1.2)
    with pytest.raises(ValueError, match="apex_gain_m"):
        make_procedural_jump_clip(apex_gain_m=2.0)


# ── Clip format ─────────────────────────────────────────────────────────
def test_clip_roundtrip_preserves_arrays_and_meta(tmp_path: Path) -> None:
    clip = make_procedural_jump_clip()
    p = save_clip(tmp_path / "ref" / "jump.npz", clip)
    loaded = load_clip(p)
    np.testing.assert_allclose(
        loaded["root_pos_z"], clip["root_pos_z"], rtol=1e-6)
    np.testing.assert_allclose(
        loaded["root_vel_z"], clip["root_vel_z"], rtol=1e-5, atol=1e-5)
    assert loaded["joint_names"] == clip["joint_names"]
    assert loaded["meta"]["source"] == "procedural:jump"
    assert loaded["fps"] == pytest.approx(50.0)


def test_validate_clip_reports_all_violations() -> None:
    bad = {
        "root_pos_z": np.array([0.7, -0.1] + [0.7] * 10),
        "fps": 999.0,
        "joint_pos": np.zeros((3, 2)),
        "joint_names": ["a"],
    }
    errors = validate_clip(bad)
    joined = " | ".join(errors)
    assert "strictly positive" in joined
    assert "fps" in joined
    assert "joint_pos" in joined or "joint_names" in joined


def test_save_refuses_invalid_and_load_derives_velocity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing to persist"):
        save_clip(tmp_path / "bad.npz", {"root_pos_z": np.zeros(3), "fps": 50})
    clip = make_procedural_jump_clip()
    clip = {k: v for k, v in clip.items() if k != "root_vel_z"}
    p = save_clip(tmp_path / "novel.npz", clip)
    loaded = load_clip(p)
    # Velocity derived by finite difference on load.
    assert "root_vel_z" in loaded
    assert loaded["root_vel_z"].shape == loaded["root_pos_z"].shape
    assert float(loaded["root_vel_z"].max()) > 1.0    # real takeoff spike


# ── §R1_BUILD_SPEC decision 2: optional library-clip keys ────────────────
def test_optional_library_keys_absent_by_default() -> None:
    """A clip with none of the R1 keys still validates clean — they are
    schema-RESERVED, not required (§decision 2)."""
    clip = make_procedural_jump_clip()
    assert validate_clip(clip) == []
    for key in ("root_pos_xy", "root_quat_wxyz", "joint_vel",
                "contact_left_foot", "contact_right_foot"):
        assert key not in clip


def test_joint_vel_backfilled_alongside_root_vel_z(tmp_path: Path) -> None:
    """`_with_velocity`'s finite-diff backfill pattern extended to
    `joint_vel`: present after load whenever `joint_pos` is present,
    even if the saved clip never carried it."""
    clip = make_procedural_jump_clip()
    clip = {k: v for k, v in clip.items()
            if k not in ("root_vel_z", "joint_vel")}
    p = save_clip(tmp_path / "no_vel.npz", clip)
    loaded = load_clip(p)
    assert "joint_vel" in loaded
    assert loaded["joint_vel"].shape == loaded["joint_pos"].shape
    assert np.isfinite(loaded["joint_vel"]).all()


def test_root_quat_wxyz_unit_norm_gate() -> None:
    n = 20
    ok_quat = np.zeros((n, 4))
    ok_quat[:, 0] = 1.0
    clip_ok = {"root_pos_z": np.full(n, 0.78), "fps": 30.0,
               "root_quat_wxyz": ok_quat}
    assert validate_clip(clip_ok) == []

    bad_quat = ok_quat.copy()
    bad_quat[0, 0] = 5.0  # not unit-norm
    clip_bad = {"root_pos_z": np.full(n, 0.78), "fps": 30.0,
                "root_quat_wxyz": bad_quat}
    errors = validate_clip(clip_bad)
    assert any("unit-norm" in e for e in errors)


# ── RSI derivation ──────────────────────────────────────────────────────
def test_derive_rsi_keys_within_validator_bounds() -> None:
    clip = make_procedural_jump_clip(apex_gain_m=0.35)
    keys = derive_rsi_train_keys(clip)
    lo, hi = keys["reset_height_offset_m"]
    assert lo == 0.0 and 0.25 <= hi <= 0.40          # ≈ apex gain
    vlo, vhi = keys["reset_vertical_velocity_mps"]
    assert -3.0 <= vlo < 0 < vhi <= 4.0              # descent + ascent both covered
    # The measured-good G1 sunk threshold falls out of 0.64·stand ≈ 0.50.
    assert keys["min_base_height_termination_m"] == pytest.approx(0.5, abs=0.01)


def test_derive_rsi_rejects_grounded_clip() -> None:
    flat = {
        "root_pos_z": np.full(120, 0.78),
        "fps": 50.0,
    }
    with pytest.raises(ValueError, match="never rises"):
        derive_rsi_train_keys(flat)


def test_apply_reference_rsi_writes_valid_versioned_spec(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    clip = make_procedural_jump_clip()
    path = apply_reference_rsi(env_dir, clip)
    assert path.is_file()
    spec = json.loads(path.read_text(encoding="utf-8"))
    # The persisted spec passes the real gate (incl. the RSI↔ET invariant:
    # airborne resets REQUIRE the sunk termination — both were emitted).
    assert validate_env_spec(json.loads(path.read_text())) == []
    assert spec["train"]["reset_height_offset_m"][1] > 0.2
    assert spec["train"]["min_base_height_termination_m"] == pytest.approx(0.5, abs=0.01)
    assert spec["meta"]["source"] == "reference:procedural:jump"
    # current.json repointed at the new version.
    cur = json.loads((env_dir / "current.json").read_text(encoding="utf-8"))
    assert cur["meta"]["version"] == spec["meta"]["version"]


def test_apply_reference_rsi_builds_on_existing_spec(tmp_path: Path) -> None:
    from sculptor.env_spec import write_env_spec_version

    env_dir = tmp_path / "env"
    write_env_spec_version(env_dir, {
        "env_spec_version": 1,
        "meta": {"source": "generated"},
        "shared": {"episode_length_s": 10.0},
        "train": {"entropy_coef_scale": 2.0},
    })
    path = apply_reference_rsi(env_dir, make_procedural_jump_clip())
    spec = json.loads(path.read_text(encoding="utf-8"))
    # Existing shared/eval scope and unrelated train keys survive.
    assert spec["shared"]["episode_length_s"] == 10.0
    assert spec["train"]["entropy_coef_scale"] == 2.0
    assert "reset_height_offset_m" in spec["train"]
    # New version number, not an overwrite.
    assert spec["meta"]["version"] != "v0"
