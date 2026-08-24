"""§R1_BUILD_SPEC W1: reference library data layer.

Covers `sculptor/refs/{library,ingest,segment}.py` and the schema
extension in `sculptor/reference.py` (root_pos_xy, root_quat_wxyz,
joint_vel, contact_*). Offline only: every fixture is constructed
in-test (no downloads, no network, no API key) per the Hard rules.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.robot_manifest import G1_29
from sculptor.reference import load_clip, save_clip, validate_clip
from sculptor.refs import library
from sculptor.refs.ingest import (
    ClipRejected,
    DatasetFormatError,
    N_COLS,
    ResegmentError,
    _hf_resolve_url,
    find_derived_segments,
    ingest_clip_bytes,
    ingest_source,
    parse_fleaven_npy,
    parse_lafan1_csv,
    resegment_clip,
    tokenize_label,
    xyzw_to_wxyz,
)
from sculptor.refs.segment import (
    DOWN_Z,
    Segment,
    STANDING_Z,
    segment_by_root_z,
    segment_by_root_z_full,
    segment_clip,
    segment_clip_full,
)

N_JOINTS = len(G1_29)


# ── synthetic dataset-row fixtures (36 cols: xyz + quat xyzw + 29 joints) ──
def _identity_quat_xyzw(n: int) -> np.ndarray:
    q = np.zeros((n, 4))
    q[:, 3] = 1.0  # w=1 at index 3 (xyzw)
    return q


def _neutral_rows(n: int = 50, *, z: np.ndarray | None = None) -> np.ndarray:
    """A generic (non fall/getup, non locomotion) synthetic clip: mild
    joint sinusoids, standing height, no meaningful horizontal travel."""
    t = np.linspace(0.0, 1.0, n)
    xy = np.stack([np.zeros(n), np.zeros(n)], axis=1)
    zcol = z if z is not None else np.full(n, 0.78)
    root_pos = np.concatenate([xy, zcol[:, None]], axis=1)
    quat = _identity_quat_xyzw(n)
    joints = 0.1 * np.sin(np.outer(t, np.linspace(1, 2, N_JOINTS)) * np.pi)
    return np.concatenate([root_pos, quat, joints], axis=1)


def _fall_getup_rows(n: int = 50) -> np.ndarray:
    """root z dips below 0.35 and rises above 0.6 — a valid fall/getup
    motion-class profile."""
    t = np.linspace(0.0, 1.0, n)
    z = 0.78 - 0.6 * np.sin(np.pi * t)  # dips to ~0.18, ends near 0.78
    z = np.clip(z, 0.05, 0.85)
    return _neutral_rows(n, z=z)


def _walk_rows(n: int = 50) -> np.ndarray:
    """Horizontal displacement > 1 m — a valid locomotion profile."""
    rows = _neutral_rows(n)
    rows[:, 0] = np.linspace(0.0, 2.0, n)  # x travels 2 m
    return rows


def _rows_to_csv_bytes(rows: np.ndarray) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in rows:
        w.writerow([f"{v:.6f}" for v in row])
    return buf.getvalue().encode("utf-8")


def _rows_to_npy_bytes(rows: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, rows.astype(np.float64))
    return buf.getvalue()


# ── quaternion conversion (§decision 1) ──────────────────────────────────
def test_xyzw_to_wxyz_known_90deg_about_z() -> None:
    # 90 degrees about z: xyzw (0, 0, 0.7071, 0.7071) -> wxyz (0.7071, 0, 0, 0.7071)
    xyzw = np.array([[0.0, 0.0, 0.70710678, 0.70710678]])
    wxyz = xyzw_to_wxyz(xyzw)
    np.testing.assert_allclose(
        wxyz, [[0.70710678, 0.0, 0.0, 0.70710678]], atol=1e-6)


def test_xyzw_to_wxyz_identity() -> None:
    xyzw = np.array([[0.0, 0.0, 0.0, 1.0]])
    wxyz = xyzw_to_wxyz(xyzw)
    np.testing.assert_allclose(wxyz, [[1.0, 0.0, 0.0, 0.0]], atol=1e-9)


def test_xyzw_to_wxyz_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="xyzw"):
        xyzw_to_wxyz(np.zeros((5, 3)))


# ── URL encoding (regression: real fleaven filenames contain spaces) ─────
def test_hf_resolve_url_percent_encodes_spaces_and_preserves_slashes() -> None:
    # Real fleaven-g1 path, e.g. "A10 - lie to crouch_poses_120_jpos.npy" —
    # an unencoded space made urllib/http.client raise InvalidURL
    # ("can't contain control characters") when spot-checked against the
    # live dataset during this build.
    path = "g1/ACCAD/Female1General_c3d/A10 - lie to crouch_poses_120_jpos.npy"
    url = _hf_resolve_url("fleaven/Retargeted_AMASS_for_robotics", path)
    assert " " not in url
    assert "%20" in url
    # Directory separators in the path are preserved, not encoded.
    assert "g1/ACCAD/Female1General_c3d/" in url
    assert url.startswith(
        "https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics/"
        "resolve/main/g1/ACCAD/Female1General_c3d/A10%20-%20lie%20to%20crouch")


# ── reference.py schema extension round-trip ─────────────────────────────
def test_clip_schema_roundtrip_with_new_optional_keys(tmp_path: Path) -> None:
    n = 40
    fps = 30.0
    z = np.full(n, 0.78)
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0  # wxyz identity
    joint_pos = np.zeros((n, 3))
    clip = {
        "root_pos_z": z,
        "fps": fps,
        "root_pos_xy": np.zeros((n, 2)),
        "root_quat_wxyz": quat,
        "joint_pos": joint_pos,
        "joint_names": ["a", "b", "c"],
        "contact_left_foot": np.zeros(n),
        "contact_right_foot": np.ones(n),
        "meta": {"source": "test"},
    }
    assert validate_clip(clip) == []
    p = save_clip(tmp_path / "clip.npz", clip)
    loaded = load_clip(p)
    np.testing.assert_allclose(loaded["root_pos_xy"], clip["root_pos_xy"])
    np.testing.assert_allclose(loaded["root_quat_wxyz"], clip["root_quat_wxyz"])
    np.testing.assert_allclose(loaded["contact_left_foot"], clip["contact_left_foot"])
    np.testing.assert_allclose(loaded["contact_right_foot"], clip["contact_right_foot"])
    # joint_vel backfilled by finite difference (joint_pos present, joint_vel absent).
    assert "joint_vel" in loaded
    assert loaded["joint_vel"].shape == joint_pos.shape


def test_validate_clip_rejects_bad_quat_norm() -> None:
    n = 20
    quat = np.zeros((n, 4))
    quat[:, 0] = 2.0  # norm 2, not unit
    clip = {"root_pos_z": np.full(n, 0.78), "fps": 30.0, "root_quat_wxyz": quat}
    errors = validate_clip(clip)
    assert any("unit-norm" in e for e in errors)


def test_validate_clip_rejects_bad_shapes_for_new_keys() -> None:
    n = 20
    clip = {
        "root_pos_z": np.full(n, 0.78),
        "fps": 30.0,
        "root_pos_xy": np.zeros((n, 3)),   # wrong width
        "joint_vel": np.zeros((n - 1, 2)),  # wrong length
        "contact_left_foot": np.zeros((n, 2)),  # wrong shape
    }
    errors = validate_clip(clip)
    joined = " | ".join(errors)
    assert "root_pos_xy" in joined
    assert "joint_vel" in joined
    assert "contact_left_foot" in joined


def test_joint_vel_matches_joint_pos_shape_gate() -> None:
    n = 20
    clip = {
        "root_pos_z": np.full(n, 0.78),
        "fps": 30.0,
        "joint_pos": np.zeros((n, 5)),
        "joint_vel": np.zeros((n, 3)),  # mismatched J
    }
    errors = validate_clip(clip)
    assert any("joint_vel shape must match joint_pos" in e for e in errors)


# ── LAFAN1 CSV ingest ─────────────────────────────────────────────────────
def test_parse_lafan1_csv_neutral_accepts() -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="dance1-2_subject3")
    assert validate_clip(clip) == []
    assert clip["joint_names"] == list(G1_29)
    assert clip["fps"] == pytest.approx(30.0)
    assert qc["n_frames"] == 50
    np.testing.assert_allclose(clip["root_quat_wxyz"][:, 0], 1.0, atol=1e-9)


def test_parse_lafan1_csv_wrong_column_count_raises() -> None:
    rows = _neutral_rows(50)[:, :-1]  # drop one column -> 35 cols
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(DatasetFormatError):
        parse_lafan1_csv(raw, stem="badwidth1")


def test_parse_lafan1_csv_too_few_frames_rejected() -> None:
    rows = _neutral_rows(10)  # below the 30-frame minimum
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="frames"):
        parse_lafan1_csv(raw, stem="tooshort1")


def test_parse_lafan1_csv_nonnumeric_row_raises_format_error() -> None:
    raw = b"a,b,c\n" + b",".join([b"0.0"] * N_COLS) + b"\n"
    with pytest.raises(DatasetFormatError):
        parse_lafan1_csv(raw, stem="garbage1")


# ── fleaven npy ingest ─────────────────────────────────────────────────────
def test_parse_fleaven_npy_accepts_and_reads_fps_from_filename() -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    clip, qc = parse_fleaven_npy(
        raw, stem="ACCAD_subject1_lie_to_crouch_poses_120_jpos")
    assert clip["fps"] == pytest.approx(120.0)
    assert validate_clip(clip) == []
    # Known-good "poses" pattern: fps_provenance records it as not-recovered.
    assert qc["fps_provenance"] == {
        "pattern": "poses", "recovered": False,
        "note": "fps encoded via the known-good "
                "'..._poses_{fps}_jpos' filename pattern",
    }


def test_parse_fleaven_npy_missing_fps_suffix_raises() -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    with pytest.raises(DatasetFormatError, match="fps"):
        parse_fleaven_npy(raw, stem="no_fps_suffix_here")


# ── fleaven npy: "stageii" fps recovery (§Problem 1, 2026-07-11) ─────────
def test_parse_fleaven_npy_stageii_known_subset_recovers_fps() -> None:
    """SMPL-X stageii filenames (GRAB/CNRS/SOMA/WEIZMANN/MOYO subsets)
    encode fps as the digits before '_jpos', not '_poses_{fps}_jpos' —
    verified against the dataset's own g1/visualize.py:read_rtj()."""
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    clip, qc = parse_fleaven_npy(
        raw, stem="airplane_fly_1_stageii_120_jpos")
    assert clip["fps"] == pytest.approx(120.0)
    assert validate_clip(clip) == []
    prov = qc["fps_provenance"]
    assert prov["pattern"] == "stageii"
    assert prov["recovered"] is True
    assert prov["source"] == (
        "https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics/"
        "blob/main/g1/visualize.py")


def test_parse_fleaven_npy_stageii_two_digit_fps_recovers() -> None:
    """MOYO subset uses fps=60 (2 digits, not 3) — the recovered regex
    must not assume a fixed digit width."""
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    clip, qc = parse_fleaven_npy(
        raw,
        stem="220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_"
             "Navasana_-a_stageii_60_jpos")
    assert clip["fps"] == pytest.approx(60.0)
    assert qc["fps_provenance"]["recovered"] is True


def test_parse_fleaven_npy_unknown_marker_still_rejected_with_reason() -> None:
    """A marker word that isn't 'poses' or the verified 'stageii' must
    still hard-reject (never silently guess an unverified fps) — with a
    reason that distinguishes it from a plain missing-fps filename."""
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    with pytest.raises(DatasetFormatError, match="recognized pattern"):
        parse_fleaven_npy(raw, stem="some_clip_unknownmarker_120_jpos")


# ── QC gates: motion-class content checks (§decision 5) ─────────────────
def test_qc_accepts_valid_fall_getup_clip() -> None:
    rows = _fall_getup_rows(50)
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="fallAndGetUp1_subject1")
    assert qc["root_z_range"][0] < 0.35
    assert qc["root_z_range"][1] > 0.6


def test_qc_rejects_mislabeled_fall_getup_clip() -> None:
    # Labeled fallAndGetUp but root_z never goes below 0.35 (mislabeled data).
    rows = _neutral_rows(50)  # flat z=0.78 throughout
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="fall/getup"):
        parse_lafan1_csv(raw, stem="fallAndGetUp2_subject9")


def test_qc_accepts_valid_walk_clip() -> None:
    rows = _walk_rows(50)
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="walk3_subject2")
    assert validate_clip(clip) == []


def test_qc_rejects_mislabeled_walk_clip() -> None:
    rows = _neutral_rows(50)  # no horizontal displacement
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="horizontal displacement"):
        parse_lafan1_csv(raw, stem="walk9_subject9")


def test_qc_rejects_nonfinite_values() -> None:
    rows = _neutral_rows(50)
    rows[5, 2] = np.nan
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="non-finite"):
        parse_lafan1_csv(raw, stem="dance2-1_subject1")


def test_qc_rejects_root_z_out_of_range() -> None:
    rows = _neutral_rows(50)
    rows[10, 2] = 3.0  # exceeds 2.5 m bound
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="root z"):
        parse_lafan1_csv(raw, stem="dance1-1_subject2")


def test_qc_rejects_joint_angle_out_of_range() -> None:
    rows = _neutral_rows(50)
    rows[3, 7] = 10.0  # exceeds 2*pi bound (first joint col)
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="joint angle"):
        parse_lafan1_csv(raw, stem="fight1_subject1")


def test_qc_flags_large_joint_delta_without_rejecting() -> None:
    rows = _neutral_rows(50)
    rows[25, 7] += 1.5  # big single-frame jump, still within +/-2pi
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="ground1_subject1")
    assert any("delta" in f for f in qc["flags"])


# ── root_z retargeting-noise clamp (§R1_BUILD_SPEC W2 item 0) ─────────────
def test_qc_clamps_small_negative_root_z_noise_and_records_qc() -> None:
    """A real fleaven-style floor-contact clip with root_z dipping to
    -3.85 mm (retargeting float noise, within the [-0.05, 0] m band) is
    clamped to a small positive floor instead of hard-rejecting, and the
    clamp is recorded in the qc block."""
    n = 50
    z = np.full(n, 0.78)
    z[10] = -0.00385
    z[11] = -0.001
    rows = _neutral_rows(n, z=z)
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="ground2_subject1")
    assert validate_clip(clip) == []
    assert (clip["root_pos_z"] > 0).all()
    assert "root_z_clamped" in qc
    assert qc["root_z_clamped"]["n_frames"] == 2
    assert qc["root_z_clamped"]["min_before"] == pytest.approx(-0.00385, abs=1e-9)
    assert "root_z_clamp" in qc["checks"]


def test_qc_rejects_root_z_below_noise_floor() -> None:
    """A min(root_z) below -0.05 m is a real below-ground pose, not
    retargeting noise — hard-reject with the `root_z_below_ground`
    reason, distinct from the general out-of-range plausibility fail."""
    n = 50
    z = np.full(n, 0.78)
    z[10] = -0.2  # well past the -0.05 m noise floor
    rows = _neutral_rows(n, z=z)
    raw = _rows_to_csv_bytes(rows)
    with pytest.raises(ClipRejected, match="root_z_below_ground"):
        parse_lafan1_csv(raw, stem="ground3_subject1")


def test_qc_leaves_all_positive_root_z_untouched_no_qc_entry() -> None:
    """An all-positive clip is not touched by the clamp path and carries
    no `root_z_clamped` qc entry."""
    rows = _neutral_rows(50)  # z is a flat 0.78, all positive
    raw = _rows_to_csv_bytes(rows)
    clip, qc = parse_lafan1_csv(raw, stem="ground4_subject1")
    assert "root_z_clamped" not in qc
    assert "root_z_clamp" not in qc["checks"]


# ── tokenize_label ────────────────────────────────────────────────────────
def test_tokenize_label_splits_camel_digits_underscore() -> None:
    assert tokenize_label("fallAndGetUp1_subject1") == [
        "fall", "and", "get", "up", "1", "subject", "1"]


# ── library.py: clip_id, hashing, provenance, index ──────────────────────
def test_slugify_and_clip_id_validation() -> None:
    assert library.CLIP_ID_RE.match(library.slugify("fallAndGetUp1_subject1"))
    library.validate_clip_id(library.slugify("Weird Name!! 123"))
    with pytest.raises(library.ClipIdError):
        library.validate_clip_id("Not_Valid_CAPS")


def test_make_provenance_requires_license() -> None:
    with pytest.raises(library.LicenseGuardError):
        library.make_provenance(
            clip_id="clip1", robot="g1", source={"kind": "hf_dataset"},
            license="", attribution="x", content_sha256_="a" * 64)


def test_write_provenance_refuses_incomplete_record(tmp_path: Path) -> None:
    with pytest.raises(library.LicenseGuardError):
        library.write_provenance(
            "g1", "clip1", {"schema": 1, "clip_id": "clip1"}, root=tmp_path)
    # Nothing written to disk on refusal.
    assert not (tmp_path / "g1" / "clip1" / "provenance.json").exists()


def test_provenance_write_read_roundtrip(tmp_path: Path) -> None:
    clip_path = (
        library.clip_dir("g1", "dance1_2", root=tmp_path)
        / library.CLIP_FILENAME
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"exact retained test artifact")
    artifact_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id="dance1_2", robot="g1",
        source={"kind": "hf_dataset", "repo": "r", "path": "p", "url": "u"},
        license="CC BY-NC-ND 4.0", attribution="attrib",
        content_sha256_=artifact_sha, source_content_sha256_="c" * 64,
        labels=["dance"], text="dance")
    path = library.write_provenance("g1", "dance1_2", prov, root=tmp_path)
    assert path.is_file()
    loaded = library.read_provenance("g1", "dance1_2", root=tmp_path)
    assert loaded == prov
    assert loaded["schema"] == 2
    assert loaded["content_sha256"] == artifact_sha
    assert loaded["source_content_sha256"] == "c" * 64


def test_rebuild_index_from_provenance_files(tmp_path: Path) -> None:
    for i in range(3):
        cid = f"clip{i}"
        clip_path = (
            library.clip_dir("g1", cid, root=tmp_path)
            / library.CLIP_FILENAME
        )
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"artifact-{i}".encode())
        prov = library.make_provenance(
            clip_id=cid, robot="g1",
            source={"kind": "hf_dataset", "repo": "r", "path": f"p{i}", "url": "u"},
            license="cc-by-4.0", attribution="a",
            content_sha256_=library.content_sha256(clip_path.read_bytes()),
            labels=["walk"], text="walk",
            qc={"duration_s": 1.0, "root_z_range": [0.1, 0.8], "n_frames": 50})
        library.write_provenance("g1", cid, prov, root=tmp_path)
    rows = library.rebuild_index(root=tmp_path)
    assert len(rows) == 3
    assert {r["clip_id"] for r in rows} == {"clip0", "clip1", "clip2"}
    # index.jsonl actually persisted and re-readable.
    reread = library.read_index(root=tmp_path)
    assert len(reread) == 3
    for row in reread:
        assert set(row) == set(library.INDEX_COLUMNS)


def test_rebuild_index_skips_missing_or_invalid_provenance(tmp_path: Path) -> None:
    # A clip dir with no provenance.json at all.
    (tmp_path / "g1" / "no_prov").mkdir(parents=True)
    # A clip dir with a provenance.json missing required fields.
    bad_dir = tmp_path / "g1" / "bad_prov"
    bad_dir.mkdir(parents=True)
    (bad_dir / "provenance.json").write_text(json.dumps({"schema": 1}))
    rows = library.rebuild_index(root=tmp_path)
    assert rows == []


def test_rebuild_index_is_robot_symmetric(tmp_path: Path) -> None:
    """§Problem 2: `rebuild_index` must scan every `<robot>/` dir
    identically — a g1 clip and a t1 clip in the SAME library must both
    survive the rebuild, each tagged with its own `robot`, and per-robot
    disk lookups (`clip_dir`) must resolve each to its own directory."""
    g1_clip = (
        library.clip_dir("g1", "walk1", root=tmp_path)
        / library.CLIP_FILENAME
    )
    g1_clip.parent.mkdir(parents=True, exist_ok=True)
    g1_clip.write_bytes(b"g1 artifact")
    prov_g1 = library.make_provenance(
        clip_id="walk1", robot="g1",
        source={"kind": "hf_dataset", "repo": "r", "path": "p", "url": "u"},
        license="cc-by-4.0", attribution="a",
        content_sha256_=library.content_sha256(g1_clip.read_bytes()),
        labels=["walk"], text="walk",
        qc={"duration_s": 1.0, "root_z_range": [0.1, 0.8], "n_frames": 50})
    library.write_provenance("g1", "walk1", prov_g1, root=tmp_path)

    t1_clip = (
        library.clip_dir("t1", "walk1", root=tmp_path)
        / library.CLIP_FILENAME
    )
    t1_clip.parent.mkdir(parents=True, exist_ok=True)
    t1_clip.write_bytes(b"t1 artifact")
    prov_t1 = library.make_provenance(
        clip_id="walk1", robot="t1",
        source={"kind": "retarget", "repo": "gmr", "path": "p", "url": "u"},
        license="cc-by-4.0", attribution="a",
        content_sha256_=library.content_sha256(t1_clip.read_bytes()),
        labels=["walk"], text="walk",
        qc={"duration_s": 1.0, "root_z_range": [0.1, 0.8], "n_frames": 50})
    library.write_provenance("t1", "walk1", prov_t1, root=tmp_path)

    rows = library.rebuild_index(root=tmp_path)
    assert len(rows) == 2
    by_robot = {r["robot"]: r for r in rows}
    assert set(by_robot) == {"g1", "t1"}
    assert by_robot["g1"]["clip_id"] == by_robot["t1"]["clip_id"] == "walk1"

    # Same clip_id, different robot -> distinct on-disk directories, no
    # collision (both sides of §Problem 2's "no shared g1 literal" ask).
    g1_dir = library.clip_dir("g1", "walk1", root=tmp_path)
    t1_dir = library.clip_dir("t1", "walk1", root=tmp_path)
    assert g1_dir != t1_dir
    assert library.read_provenance("g1", "walk1", root=tmp_path)["robot"] == "g1"
    assert library.read_provenance("t1", "walk1", root=tmp_path)["robot"] == "t1"


def test_indexed_content_hashes_reads_from_provenance(tmp_path: Path) -> None:
    clip_path = (
        library.clip_dir("g1", "c1", root=tmp_path)
        / library.CLIP_FILENAME
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"indexed artifact")
    artifact_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id="c1", robot="g1", source={"kind": "hf_dataset"},
        license="cc-by-4.0", attribution="a",
        content_sha256_=artifact_sha,
        source_content_sha256_="cafebabe" * 8)
    library.write_provenance("g1", "c1", prov, root=tmp_path)
    hashes = library.indexed_content_hashes(root=tmp_path)
    assert artifact_sha in hashes
    assert "cafebabe" * 8 not in hashes
    assert library.indexed_source_hashes(root=tmp_path) == {"cafebabe" * 8}


def test_schema2_provenance_rejects_missing_or_mismatched_artifact(
    tmp_path: Path,
) -> None:
    prov = library.make_provenance(
        clip_id="exact1", robot="g1", source={"kind": "test"},
        license="cc0", attribution="test", content_sha256_="a" * 64,
    )
    with pytest.raises(
        library.LicenseGuardError,
        match="requires the retained clip.npz artifact",
    ):
        library.write_provenance("g1", "exact1", prov, root=tmp_path)

    clip_path = (
        library.clip_dir("g1", "exact1", root=tmp_path)
        / library.CLIP_FILENAME
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"different bytes")
    with pytest.raises(
        library.LicenseGuardError,
        match="does not match the exact retained clip.npz bytes",
    ):
        library.write_provenance("g1", "exact1", prov, root=tmp_path)


def test_rebuild_index_drops_a_mutated_schema2_artifact(tmp_path: Path) -> None:
    clip_path = (
        library.clip_dir("g1", "exact2", root=tmp_path)
        / library.CLIP_FILENAME
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"admitted bytes")
    admitted_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id="exact2", robot="g1", source={"kind": "test"},
        license="cc0", attribution="test", content_sha256_=admitted_sha,
        source_content_sha256_="d" * 64,
    )
    library.write_provenance("g1", "exact2", prov, root=tmp_path)
    assert [row["clip_id"] for row in library.rebuild_index(root=tmp_path)] == [
        "exact2"
    ]

    clip_path.write_bytes(b"mutated after admission")
    assert library.rebuild_index(root=tmp_path) == []
    assert admitted_sha not in library.indexed_content_hashes(root=tmp_path)
    assert "d" * 64 not in library.indexed_source_hashes(root=tmp_path)

def test_migrate_artifact_identities_preserves_source_and_downgrades_tierd(
    tmp_path: Path,
) -> None:
    from sculptor.reference import save_clip

    clip_id = "legacy_hop"
    clip = {
        "root_pos_z": np.linspace(0.72, 0.82, 229),
        "fps": 60.0,
    }
    d = library.clip_dir("g1", clip_id, root=tmp_path)
    clip_path = save_clip(d / library.CLIP_FILENAME, clip)
    raw_source_sha = "1" * 64
    legacy = {
        "schema": 1,
        "clip_id": clip_id,
        "robot": "g1",
        "source": {"kind": "hf_dataset", "repo": "r", "path": "p"},
        "license": "cc-by-4.0",
        "attribution": "dataset",
        "content_sha256": raw_source_sha,
        "tier": "D",
        "tierD": {"feasible": True, "rollout_sha256": "2" * 64},
        "fps_source": 60.0,
        "qc": {"n_frames": 229, "duration_s": round(229 / 60, 4)},
    }
    library.write_provenance("g1", clip_id, legacy, root=tmp_path)

    dry = library.migrate_artifact_identities(root=tmp_path, dry_run=True)
    assert len(dry) == 1
    assert library.read_provenance("g1", clip_id, root=tmp_path)["schema"] == 1

    receipts = library.migrate_artifact_identities(root=tmp_path)
    assert len(receipts) == 1
    prov = library.read_provenance("g1", clip_id, root=tmp_path)
    artifact_sha = library.content_sha256(clip_path.read_bytes())
    assert prov["schema"] == 2
    assert prov["content_sha256"] == artifact_sha
    assert prov["source_content_sha256"] == raw_source_sha
    assert prov["legacy_content_sha256"] == raw_source_sha
    assert prov["qc"]["duration_s"] == pytest.approx(3.8)
    assert prov["tier"] == "K"
    assert prov["tierD"]["feasible"] is False
    assert "fresh Tier-D certification required" in (
        prov["tierD"]["identity_migration_invalidated"]["reason"])
    assert library.migrate_artifact_identities(root=tmp_path) == []


# ── segmentation (§decision 6) ────────────────────────────────────────────
def _synthetic_z_profile(fps: float, n_cycles: int = 3) -> np.ndarray:
    """Builds a z-trace with `n_cycles` clean down-up repetitions:
    standing (>0.6, 2.5s) -> down (<0.35, 1.0s) -> standing (>0.6, 2.0s)
    -> single-frame wobble (0.58) -> standing (>0.6, 1.0s), repeated.

    2026-07-09 fix: the original fixture used a flat 1.5s standing hold
    on both sides of each fall with no gap between one cycle's recovery
    and the next cycle's pre-fall stand. Two adjacent cycles' standing
    holds then merge into ONE continuous sustained-standing run (this
    cycle's recovery + the next cycle's pre-fall stand, back to back),
    so `nxt[1]` (the end of that merged run, per `segment_by_root_z`) sits
    EXACTLY at the next down-run's start — the trailing +0.5s pad then
    always rolls into the next fall, regardless of how long the standing
    hold is (lengthening the hold just shifts the merge point, it never
    creates a gap). The new hard start/end QC gate (build spec item 2)
    correctly rejects a segment whose padded tail lands back below the
    end-standing bar — this was a latent bug in the fixture (unrelated
    to the get-up-start fix), just never previously asserted on since
    the old tests only checked count/ordering/min-length, not segment
    content.

    Fix: insert a single-frame dip to 0.58 (below STANDING_Z=0.60, so it
    breaks the `_sustained_runs` mask and ends the standing run there;
    above DOWN_Z=0.35 and above the end-QC bar 0.55, so it's not a real
    down transition and doesn't fail QC on its own) partway through each
    recovery hold, with >= PAD_S of genuine >0.6 standing on both sides
    of it. This gives the trailing pad real "still standing" frames to
    land in, matching what an actual recovered-standing hold with a
    brief mocap wobble looks like, without changing what the fixture is
    testing (three clean, independent down->recover cycles).
    """
    segs = []
    for _ in range(n_cycles):
        segs.append(np.full(int(2.5 * fps), 0.75))   # standing (pre-fall)
        segs.append(np.full(int(1.0 * fps), 0.10))   # down
        segs.append(np.full(int(2.0 * fps), 0.75))   # standing (recovered)
        segs.append(np.full(1, 0.58))                 # brief sub-threshold wobble
        segs.append(np.full(int(1.0 * fps), 0.75))   # standing (still recovered)
    return np.concatenate(segs)


def test_segment_by_root_z_finds_three_cycles() -> None:
    fps = 30.0
    z = _synthetic_z_profile(fps, n_cycles=3)
    segs = segment_by_root_z(z, fps)
    assert len(segs) == 3
    for s in segs:
        assert isinstance(s, Segment)
        assert s.start < s.end
        assert s.n_frames >= int(2.0 * fps)  # min length


def test_segment_by_root_z_frame_ranges_are_ordered_and_within_bounds() -> None:
    fps = 30.0
    z = _synthetic_z_profile(fps, n_cycles=3)
    segs = segment_by_root_z(z, fps)
    starts = [s.start for s in segs]
    assert starts == sorted(starts)
    for s in segs:
        assert 0 <= s.start < s.end <= z.shape[0]


def test_segment_by_root_z_no_cycles_returns_empty() -> None:
    z = np.full(200, 0.78)  # always standing, never down
    assert segment_by_root_z(z, 30.0) == []


def test_segment_by_root_z_ignores_brief_noise_dip() -> None:
    fps = 30.0
    z = np.full(int(3.0 * fps), 0.75)
    # A single-frame noise dip below 0.35, far shorter than the 0.5s sustain.
    z[45] = 0.10
    assert segment_by_root_z(z, fps) == []


def test_segment_clip_slices_all_arrays(tmp_path: Path) -> None:
    fps = 30.0
    z = _synthetic_z_profile(fps, n_cycles=3)
    n = z.shape[0]
    clip = {
        "root_pos_z": z,
        "fps": fps,
        "root_pos_xy": np.zeros((n, 2)),
        "joint_pos": np.zeros((n, 4)),
        "joint_names": ["a", "b", "c", "d"],
        "meta": {"source": "test"},
    }
    slices = segment_clip(clip)
    assert len(slices) == 3
    for s in slices:
        frange = s["_segment_frame_range"]
        length = frange[1] - frange[0]
        assert s["root_pos_z"].shape[0] == length
        assert s["root_pos_xy"].shape[0] == length
        assert s["joint_pos"].shape[0] == length
        assert s["joint_names"] == ["a", "b", "c", "d"]
        assert s["fps"] == fps


def test_segment_clip_rejects_invalid_clip() -> None:
    with pytest.raises(ValueError, match="invalid"):
        segment_clip({"root_pos_z": np.array([-1.0] * 40), "fps": 30.0})


# ── settled-start fix (2026-07-09) ────────────────────────────────────────
def _stand_fall_settle_rise_stand_profile(fps: float) -> np.ndarray:
    """Explicit stand -> fall -> settle -> rise -> stand shape, built to
    make the bug this fix targets unmissable: a real fall is NOT a step
    function. The subject stands, then DESCENDS gradually through the
    down threshold (a genuine fall, several tenths of a second), settles
    at a fully-prone height and holds it, then rises back up to standing
    and holds that.

    The old rule took `down_run_start - 0.5s pad` as the segment start —
    landing back up in the pre-fall standing plateau, INCLUDING the
    falling motion in the segment. The fix must start the segment at the
    settled-lying frame instead.
    """
    stand1 = np.full(int(1.0 * fps), 0.75)                       # standing
    n_fall = int(0.6 * fps)
    fall = np.linspace(0.75, 0.05, n_fall)                        # gradual fall
    settle = np.full(int(1.0 * fps), 0.05)                        # settled, at rest
    n_rise = int(0.6 * fps)
    rise = np.linspace(0.05, 0.75, n_rise)                        # gradual rise
    stand2 = np.full(int(1.5 * fps), 0.75)                        # standing (recovered)
    return np.concatenate([stand1, fall, settle, rise, stand2])


def test_settled_start_excludes_leading_fall_and_prefall_standing() -> None:
    """The core regression this fix targets: the segment must start at
    the settled lying point, NOT at the pre-fall standing plateau and
    NOT partway down the fall."""
    fps = 30.0
    z = _stand_fall_settle_rise_stand_profile(fps)
    segs = segment_by_root_z(z, fps)
    assert len(segs) == 1
    seg = segs[0]
    # The settle plateau is constant 0.05 — the start frame's z must be
    # in that plateau (not the 0.75 pre-fall stand, not a mid-fall value
    # somewhere on the linspace between 0.75 and 0.05).
    assert z[seg.start] == pytest.approx(0.05, abs=1e-9)
    # The whole leading pre-fall stand (constant 0.75, first int(1.0*fps)
    # frames) must be excluded from the segment.
    assert seg.start >= int(1.0 * fps)
    # The fall itself (the linspace descent) must also be excluded —
    # every frame from seg.start onward should already be at/near the
    # settled floor, not still descending.
    fall_end = int(1.0 * fps) + int(0.6 * fps)
    assert seg.start >= fall_end - 1


def test_settled_start_no_leading_pad_before_settled_frame() -> None:
    """No standing padding before the start (build spec item 1): the
    pre-fall standing plateau must be entirely excluded, unlike the old
    `d_start - PAD_S` rule which pulled 0.5s of standing into the
    segment."""
    fps = 30.0
    z = _stand_fall_settle_rise_stand_profile(fps)
    segs = segment_by_root_z(z, fps)
    assert len(segs) == 1
    seg = segs[0]
    n_prefall_stand = int(1.0 * fps)
    # None of the pre-fall standing plateau's frame indices are included
    # in the segment — the old `d_start - PAD_S` rule would have pulled
    # the last ~15 of these 30 frames (0.5s pad) into the segment.
    assert seg.start >= n_prefall_stand
    # And the segment's own start frame is at the settled floor value,
    # not some intermediate/standing value.
    assert z[seg.start] == pytest.approx(0.05, abs=1e-9)


def test_find_settled_start_falls_back_to_argmin_when_never_settled() -> None:
    """If the subject bounces straight back up without ever holding
    still (no settled frame), the fallback is the down-interval's
    z-argmin frame — still a far better "start lying down" proxy than
    the old pre-fall standing pad."""
    fps = 30.0
    stand1 = np.full(int(1.0 * fps), 0.75)
    # A "V"-shaped bounce: continuously moving the whole time below
    # DOWN_Z, never still for a full SETTLE_WINDOW_S — down_min is 0.5s
    # (15 frames @30fps) so this interval must sustain z < DOWN_Z for at
    # least that long to even register as a down-run.
    n_down = int(1.0 * fps)
    down = np.concatenate([
        np.linspace(0.30, 0.05, n_down // 2),
        np.linspace(0.05, 0.30, n_down - n_down // 2),
    ])
    stand2 = np.full(int(1.5 * fps), 0.75)
    z = np.concatenate([stand1, down, stand2])
    segs = segment_by_root_z(z, fps)
    # This profile may or may not pass QC (its "settled" point is a
    # single instantaneous minimum, not a real hold) — assert on the
    # underlying start selection directly via the full-result API
    # regardless of accept/reject, since that's what this test targets.
    result = segment_by_root_z_full(z, fps)
    candidates = result.segments + [
        Segment(r.start, r.end) for r in result.rejected]
    assert len(candidates) == 1
    seg = candidates[0]
    down_start = int(1.0 * fps)
    down_end = down_start + n_down
    expected_start = down_start + int(np.argmin(z[down_start:down_end]))
    assert seg.start == expected_start


# ── QC gate (build spec item 2) ───────────────────────────────────────────
def test_qc_rejects_fall_only_interval_that_never_recovers_to_end_standing() -> None:
    """A down-interval whose padded tail rolls back into ANOTHER fall
    (rather than a genuine standing recovery) must be rejected with a
    reason, not silently accepted. Reuses the exact latent-bug shape
    documented on `_synthetic_z_profile` above (short standing hold with
    no real gap before the next fall) as a direct, minimal repro."""
    fps = 30.0
    z = np.concatenate([
        np.full(int(1.5 * fps), 0.75),   # standing
        np.full(int(1.0 * fps), 0.10),   # down (fall 1, settled the whole time)
        np.full(int(1.5 * fps), 0.75),   # standing recovery -- too short: recovery
                                          # (1.0s) + PAD_S (0.5s) == 1.5s exactly, so
                                          # the pad rolls straight into fall 2.
        np.full(int(1.0 * fps), 0.10),   # down (fall 2)
        np.full(int(1.5 * fps), 0.75),   # standing
    ])
    result = segment_by_root_z_full(z, fps)
    assert len(result.rejected) >= 1
    rej = result.rejected[0]
    assert "end standing" in rej.reason
    assert rej.start < rej.end


def test_qc_segment_rejects_start_window_that_is_not_lying() -> None:
    """Direct unit test of the start-side QC gate (`_qc_segment`, the
    engine behind `segment_by_root_z_full`'s item-2 gate). Architectural
    note: given a valid down-run (every frame strictly < DOWN_Z for the
    sustain duration) and the settled-start selection (which always
    picks a frame with z < DOWN_Z, settled or argmin-fallback), a
    start-side rejection is very hard to trigger through the public
    `segment_by_root_z_full` pipeline end-to-end — the settled-start fix
    makes it a defensive backstop rather than a commonly-hit path. This
    test instead exercises the gate function directly (it's the
    documented contract of build spec item 2) on a hand-built z-array
    whose first-10%-window mean sits above QC_START_MAX_MEAN_Z."""
    from sculptor.refs.segment import _qc_segment

    n = 100
    z = np.full(n, 0.75)
    z[:10] = 0.50   # start window (first 10%) mean = 0.50 >= 0.35: fails "start lying"
    z[-10:] = 0.75  # end window mean = 0.75 > 0.55: would pass "end standing"
    reason = _qc_segment(z, 0, n)
    assert reason is not None
    assert "start" in reason and "lying" in reason


def test_qc_segment_accepts_a_genuine_start_lying_end_standing_segment() -> None:
    from sculptor.refs.segment import _qc_segment

    n = 100
    z = np.linspace(0.10, 0.75, n)
    z[:10] = 0.10   # start window mean well under 0.35
    z[-10:] = 0.75  # end window mean well over 0.55
    assert _qc_segment(z, 0, n) is None


# ── frame_range / provenance preserved through the new start rule ────────
def test_segment_frame_range_and_provenance_shape_intact(tmp_path: Path) -> None:
    fps = 30.0
    z = _stand_fall_settle_rise_stand_profile(fps)
    n = z.shape[0]
    clip = {
        "root_pos_z": z,
        "fps": fps,
        "root_pos_xy": np.zeros((n, 2)),
        "joint_pos": np.zeros((n, 4)),
        "joint_names": ["a", "b", "c", "d"],
        "meta": {"source": "test"},
    }
    result = segment_clip_full(clip)
    assert len(result.segments) == 1
    s = result.segments[0]
    frange = s["_segment_frame_range"]
    assert isinstance(frange, list) and len(frange) == 2
    length = frange[1] - frange[0]
    assert s["root_pos_z"].shape[0] == length
    assert s["root_pos_xy"].shape[0] == length
    assert s["joint_pos"].shape[0] == length
    assert s["joint_names"] == ["a", "b", "c", "d"]
    assert s["fps"] == fps


# ── ingest_clip_bytes: end-to-end persist + provenance + segmentation ────
def test_ingest_clip_bytes_persists_clip_and_provenance(tmp_path: Path) -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_csv_bytes(rows)
    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/dance1-2_subject3.csv", stem="dance1-2_subject3",
        robot="g1", root=tmp_path, no_preview=True)
    assert result.clip_path.is_file()
    assert result.provenance_path.is_file()
    loaded = load_clip(result.clip_path)
    assert validate_clip(loaded) == []
    prov = json.loads(result.provenance_path.read_text())
    assert prov["license"] == "CC BY-NC-ND 4.0"
    assert prov["content_sha256"] == library.content_sha256(
        result.clip_path.read_bytes())
    assert prov["source_content_sha256"] == library.content_sha256(raw)
    assert prov["content_sha256"] != prov["source_content_sha256"]
    assert "dance" in prov["labels"]


def test_ingest_clip_bytes_fall_getup_produces_segments(tmp_path: Path) -> None:
    fps = 30.0
    z = _synthetic_z_profile(fps, n_cycles=3)
    n = z.shape[0]
    xy = np.zeros((n, 2))
    quat = _identity_quat_xyzw(n)
    joints = np.zeros((n, N_JOINTS))
    rows = np.concatenate([xy, z[:, None], quat, joints], axis=1)
    raw = _rows_to_csv_bytes(rows)
    ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/fallAndGetUp1_subject1.csv", stem="fallAndGetUp1_subject1",
        robot="g1", root=tmp_path, no_preview=True)
    rows_idx = library.rebuild_index(root=tmp_path)
    seg_rows = [r for r in rows_idx if r["clip_id"].startswith("fallandgetup1_subject1--seg")]
    assert len(seg_rows) >= 2   # >= 2 segments per the spec's data-run acceptance bar
    for r in seg_rows:
        prov = library.read_provenance("g1", r["clip_id"], root=tmp_path)
        assert prov["parent_clip_id"] == "fallandgetup1_subject1"
        assert prov["frame_range"] is not None
        assert "segment" in prov["labels"]
        clip_path = library.clip_dir("g1", r["clip_id"], root=tmp_path) / "clip.npz"
        assert prov["content_sha256"] == library.content_sha256(
            clip_path.read_bytes())
        assert prov["source_content_sha256"] is not None


def test_ingest_clip_bytes_rejects_bad_source() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        ingest_clip_bytes(
            b"", source="bogus", repo="r", rel_path="p", stem="s")


# ── `sculpt refs resegment` CLI / resegment_clip (build spec item 4) ─────
def _ingest_fake_fall_getup_parent(
    tmp_path: Path, *, stem: str = "fallAndGetUp1_subject1",
) -> str:
    """Ingests one synthetic fall/getup parent clip (using the OLD
    `_synthetic_z_profile` multi-cycle shape, which now produces 3
    settled-start segments under the current rules) under `tmp_path`,
    returning its clip_id. Used as the "stale old segments" fixture for
    resegment tests."""
    fps = 30.0
    z = _synthetic_z_profile(fps, n_cycles=3)
    n = z.shape[0]
    xy = np.zeros((n, 2))
    quat = _identity_quat_xyzw(n)
    joints = np.zeros((n, N_JOINTS))
    rows = np.concatenate([xy, z[:, None], quat, joints], axis=1)
    raw = _rows_to_csv_bytes(rows)
    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path=f"g1/{stem}.csv", stem=stem,
        robot="g1", root=tmp_path, no_preview=True)
    return result.clip_id


def test_find_derived_segments_matches_only_this_parent(tmp_path: Path) -> None:
    parent_id = _ingest_fake_fall_getup_parent(tmp_path, stem="fallAndGetUp1_subject1")
    other_id = _ingest_fake_fall_getup_parent(tmp_path, stem="fallAndGetUp2_subject2")
    library.rebuild_index(root=tmp_path)

    segs = find_derived_segments("g1", parent_id, root=tmp_path)
    assert len(segs) == 3
    assert all(s.startswith(f"{parent_id}--seg") for s in segs)
    other_segs = find_derived_segments("g1", other_id, root=tmp_path)
    assert len(other_segs) == 3
    assert set(segs).isdisjoint(other_segs)


def test_resegment_dry_run_reports_without_writing(tmp_path: Path) -> None:
    parent_id = _ingest_fake_fall_getup_parent(tmp_path)
    library.rebuild_index(root=tmp_path)
    before = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert len(before) == 3

    summary = resegment_clip(parent_id, robot="g1", root=tmp_path, dry_run=True)
    assert summary.dry_run is True
    assert set(summary.removed) == before
    assert len(summary.added) == 3  # same synthetic shape -> same 3 segments

    # Nothing on disk changed.
    after = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert after == before
    for seg_id in before:
        assert (library.clip_dir("g1", seg_id, root=tmp_path)
                / library.CLIP_FILENAME).is_file()


def test_resegment_replaces_old_segments_and_rebuilds_index(tmp_path: Path) -> None:
    parent_id = _ingest_fake_fall_getup_parent(tmp_path, stem="fallAndGetUp1_subject1")
    unrelated_id = _ingest_fake_fall_getup_parent(tmp_path, stem="fallAndGetUp2_subject2")
    library.rebuild_index(root=tmp_path)

    old_segs = set(find_derived_segments("g1", parent_id, root=tmp_path))
    unrelated_segs_before = set(find_derived_segments("g1", unrelated_id, root=tmp_path))
    assert len(old_segs) == 3

    # Simulate "old rule" staleness: hand-corrupt one old segment's
    # frame_range so it's unambiguously distinguishable from a freshly
    # regenerated one, then resegment and confirm it's gone.
    stale_seg_id = sorted(old_segs)[0]
    stale_prov = library.read_provenance("g1", stale_seg_id, root=tmp_path)
    stale_prov["frame_range"] = [0, 1]  # obviously-wrong marker value
    library.write_provenance("g1", stale_seg_id, stale_prov, root=tmp_path)

    summary = resegment_clip(parent_id, robot="g1", root=tmp_path, no_preview=True)
    assert summary.dry_run is False
    assert set(summary.removed) == old_segs
    assert len(summary.added) == 3

    new_segs = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert len(new_segs) == 3
    # The stale marker is gone — `stale_seg_id`'s clip_id is reused by a
    # freshly regenerated segment (deterministic segNN naming), but its
    # provenance no longer carries the hand-corrupted frame_range.
    assert stale_seg_id in new_segs
    fresh_prov = library.read_provenance("g1", stale_seg_id, root=tmp_path)
    assert fresh_prov["frame_range"] != [0, 1]
    for seg_id in new_segs:
        prov = library.read_provenance("g1", seg_id, root=tmp_path)
        assert prov["frame_range"] != [0, 1]
        assert prov["parent_clip_id"] == parent_id

    # Unrelated parent's segments are completely untouched.
    unrelated_segs_after = set(find_derived_segments("g1", unrelated_id, root=tmp_path))
    assert unrelated_segs_after == unrelated_segs_before

    # index.jsonl reflects the new (fresh, non-stale) segments, rebuilt
    # automatically by `resegment_clip`.
    index_rows = library.read_index(root=tmp_path)
    index_ids = {r["clip_id"] for r in index_rows}
    assert new_segs <= index_ids


def test_resegment_logs_qc_rejects(tmp_path: Path) -> None:
    """A parent whose current segmentation includes a QC-rejected
    candidate must have that rejection logged via
    `library.append_reject` (same mechanism ingest uses), not silently
    dropped."""
    fps = 30.0
    # Latent-bug shape from test_qc_rejects_fall_only_interval_...: the
    # first cycle's padded tail rolls into the second fall and gets
    # rejected; the final cycle recovers cleanly and is accepted.
    z = np.concatenate([
        np.full(int(1.5 * fps), 0.75),
        np.full(int(1.0 * fps), 0.10),
        np.full(int(1.5 * fps), 0.75),   # too-short recovery -> rejected
        np.full(int(1.0 * fps), 0.10),
        np.full(int(1.5 * fps), 0.75),
    ])
    n = z.shape[0]
    xy = np.zeros((n, 2))
    quat = _identity_quat_xyzw(n)
    joints = np.zeros((n, N_JOINTS))
    rows = np.concatenate([xy, z[:, None], quat, joints], axis=1)
    raw = _rows_to_csv_bytes(rows)
    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/fallAndGetUp3_subject3.csv", stem="fallAndGetUp3_subject3",
        robot="g1", root=tmp_path, no_preview=True)

    rejects_path = library.rejects_path(root=tmp_path)
    assert rejects_path.is_file()
    reject_lines = [
        json.loads(line) for line in rejects_path.read_text().splitlines() if line]
    assert any(r["reason"] == "segment_qc_failed" for r in reject_lines)

    # Now call resegment directly on the parent: it must log the SAME
    # rejection again (current rules re-evaluated from scratch), and
    # still leave the accepted segment intact.
    n_reject_lines_before = len(reject_lines)
    resegment_clip(result.clip_id, robot="g1", root=tmp_path, no_preview=True)
    reject_lines_after = [
        json.loads(line) for line in rejects_path.read_text().splitlines() if line]
    assert len(reject_lines_after) > n_reject_lines_before
    assert any(
        r["reason"] == "segment_qc_failed" and r.get("parent_clip_id") == result.clip_id
        for r in reject_lines_after)


def test_resegment_raises_for_unknown_parent(tmp_path: Path) -> None:
    with pytest.raises(ResegmentError, match="no such parent clip"):
        resegment_clip("does-not-exist", robot="g1", root=tmp_path)


def test_resegment_cli_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the actual `sculpt refs resegment` typer command (build
    spec item 4), via typer's CliRunner, against a tmp RS_REFERENCE_ROOT
    library — not just the underlying `resegment_clip` function."""
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    parent_id = _ingest_fake_fall_getup_parent(tmp_path)
    library.rebuild_index(root=tmp_path)
    before = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert len(before) == 3

    runner = CliRunner()
    result = runner.invoke(app, ["refs", "resegment", "--parent", parent_id, "--no-preview"])
    assert result.exit_code == 0, result.output
    assert "removed=3" in result.output
    assert "added=3" in result.output

    after = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert len(after) == 3


def test_resegment_cli_dry_run_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path))
    parent_id = _ingest_fake_fall_getup_parent(tmp_path)
    library.rebuild_index(root=tmp_path)
    before = set(find_derived_segments("g1", parent_id, root=tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        app, ["refs", "resegment", "--parent", parent_id, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would remove=3" in result.output
    assert "would add=3" in result.output

    after = set(find_derived_segments("g1", parent_id, root=tmp_path))
    assert after == before


# ── ingest_source idempotency (monkeypatched network) ─────────────────────
def test_ingest_source_idempotent_on_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_csv_bytes(rows)

    def fake_list_files(repo: str, path: str, **kwargs):
        return ["g1/dance1-2_subject3.csv"]

    def fake_get_bytes(url: str, **kwargs):
        return raw

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files", fake_list_files)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    summary1 = ingest_source("lafan1-g1", root=tmp_path, no_preview=True)
    assert summary1.accepted == ["dance1_2_subject3"]
    assert summary1.skipped_existing == []

    summary2 = ingest_source("lafan1-g1", root=tmp_path, no_preview=True)
    assert summary2.accepted == []
    assert summary2.skipped_existing == ["dance1_2_subject3"]


def test_ingest_source_rejects_are_logged_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = _rows_to_csv_bytes(_neutral_rows(50))
    bad = _rows_to_csv_bytes(_neutral_rows(10))  # too few frames -> rejected

    def fake_list_files(repo: str, path: str, **kwargs):
        return ["g1/dance1-1_subject1.csv", "g1/dance2-1_subject1.csv"]

    def fake_get_bytes(url: str, **kwargs):
        return bad if "dance2" in url else good

    monkeypatch.setattr("sculptor.refs.ingest._hf_list_files", fake_list_files)
    monkeypatch.setattr("sculptor.refs.ingest._http_get_bytes", fake_get_bytes)

    summary = ingest_source("lafan1-g1", root=tmp_path, no_preview=True)
    assert summary.accepted == ["dance1_1_subject1"]
    assert len(summary.rejected) == 1
    assert summary.rejected[0][0] == "dance2-1_subject1"
    rejects_file = library.rejects_path(root=tmp_path)
    assert rejects_file.is_file()
    lines = rejects_file.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["clip_id"] == "dance2-1_subject1"


def test_ingest_source_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        ingest_source("not-a-real-source")


def test_ingest_source_rejects_robot_mismatch(tmp_path: Path) -> None:
    """§Problem 2: fleaven-g1/lafan1-g1 rows are hardwired to G1_29 joint
    order regardless of the `robot` kwarg — registering them under a
    different robot slug would silently mislabel G1-schema data.
    `ingest_source` must refuse rather than allow that mislabeling."""
    with pytest.raises(ValueError, match="inherently robot='g1'"):
        ingest_source("lafan1-g1", root=tmp_path, robot="t1")
    with pytest.raises(ValueError, match="inherently robot='g1'"):
        ingest_source("fleaven-g1", root=tmp_path, robot="t1")
