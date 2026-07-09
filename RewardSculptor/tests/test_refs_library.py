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
    _hf_resolve_url,
    ingest_clip_bytes,
    ingest_source,
    parse_fleaven_npy,
    parse_lafan1_csv,
    tokenize_label,
    xyzw_to_wxyz,
)
from sculptor.refs.segment import Segment, segment_by_root_z, segment_clip

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


def test_parse_fleaven_npy_missing_fps_suffix_raises() -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_npy_bytes(rows)
    with pytest.raises(DatasetFormatError, match="fps"):
        parse_fleaven_npy(raw, stem="no_fps_suffix_here")


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
    prov = library.make_provenance(
        clip_id="dance1_2", robot="g1",
        source={"kind": "hf_dataset", "repo": "r", "path": "p", "url": "u"},
        license="CC BY-NC-ND 4.0", attribution="attrib",
        content_sha256_="b" * 64, labels=["dance"], text="dance")
    path = library.write_provenance("g1", "dance1_2", prov, root=tmp_path)
    assert path.is_file()
    loaded = library.read_provenance("g1", "dance1_2", root=tmp_path)
    assert loaded == prov


def test_rebuild_index_from_provenance_files(tmp_path: Path) -> None:
    for i in range(3):
        cid = f"clip{i}"
        prov = library.make_provenance(
            clip_id=cid, robot="g1",
            source={"kind": "hf_dataset", "repo": "r", "path": f"p{i}", "url": "u"},
            license="cc-by-4.0", attribution="a",
            content_sha256_=f"{i}" * 64, labels=["walk"], text="walk",
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


def test_indexed_content_hashes_reads_from_provenance(tmp_path: Path) -> None:
    prov = library.make_provenance(
        clip_id="c1", robot="g1", source={"kind": "hf_dataset"},
        license="cc-by-4.0", attribution="a", content_sha256_="deadbeef" * 8)
    library.write_provenance("g1", "c1", prov, root=tmp_path)
    hashes = library.indexed_content_hashes(root=tmp_path)
    assert "deadbeef" * 8 in hashes


# ── segmentation (§decision 6) ────────────────────────────────────────────
def _synthetic_z_profile(fps: float, n_cycles: int = 3) -> np.ndarray:
    """Builds a z-trace with `n_cycles` clean down-up repetitions:
    standing (>0.6, 1.5s) -> down (<0.35, 1.0s) -> standing (>0.6, 1.5s), repeated."""
    segs = []
    for _ in range(n_cycles):
        segs.append(np.full(int(1.5 * fps), 0.75))   # standing
        segs.append(np.full(int(1.0 * fps), 0.10))   # down
        segs.append(np.full(int(1.5 * fps), 0.75))   # standing (recovered)
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


# ── ingest_clip_bytes: end-to-end persist + provenance + segmentation ────
def test_ingest_clip_bytes_persists_clip_and_provenance(tmp_path: Path) -> None:
    rows = _neutral_rows(50)
    raw = _rows_to_csv_bytes(rows)
    result = ingest_clip_bytes(
        raw, source="lafan1-g1", repo="lvhaidong/LAFAN1_Retargeting_Dataset",
        rel_path="g1/dance1-2_subject3.csv", stem="dance1-2_subject3",
        robot="g1", root=tmp_path)
    assert result.clip_path.is_file()
    assert result.provenance_path.is_file()
    loaded = load_clip(result.clip_path)
    assert validate_clip(loaded) == []
    prov = json.loads(result.provenance_path.read_text())
    assert prov["license"] == "CC BY-NC-ND 4.0"
    assert prov["content_sha256"] == library.content_sha256(raw)
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
        robot="g1", root=tmp_path)
    rows_idx = library.rebuild_index(root=tmp_path)
    seg_rows = [r for r in rows_idx if r["clip_id"].startswith("fallandgetup1_subject1--seg")]
    assert len(seg_rows) >= 2   # >= 2 segments per the spec's data-run acceptance bar
    for r in seg_rows:
        prov = library.read_provenance("g1", r["clip_id"], root=tmp_path)
        assert prov["parent_clip_id"] == "fallandgetup1_subject1"
        assert prov["frame_range"] is not None
        assert "segment" in prov["labels"]


def test_ingest_clip_bytes_rejects_bad_source() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        ingest_clip_bytes(
            b"", source="bogus", repo="r", rel_path="p", stem="s")


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

    summary1 = ingest_source("lafan1-g1", root=tmp_path)
    assert summary1.accepted == ["dance1_2_subject3"]
    assert summary1.skipped_existing == []

    summary2 = ingest_source("lafan1-g1", root=tmp_path)
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

    summary = ingest_source("lafan1-g1", root=tmp_path)
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
