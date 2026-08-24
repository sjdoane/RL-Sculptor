"""§REFERENCE_TRAJECTORY_PLAN §11 R3: `sculptor/refs/retarget.py` —
cross-robot retargeting wrapper around GMR. Offline only: every test
either fabricates a GMR-shaped npz payload directly or monkeypatches
`subprocess.run` — nothing here shells out to GMR's real venv or touches
the network (the REAL end-to-end proof against GMR's actual venv is a
manual run, documented in the mission report, not part of this suite).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sculptor.reference import load_clip
from sculptor.refs import library
from sculptor.refs.retarget import (
    BVH_LAFAN1_CAPABLE_ROBOTS,
    GMR_ROBOT_IDS,
    RetargetError,
    attach_role_resolution_qc,
    default_gmr_python,
    gmr_quat_wxyz_to_container,
    gmr_result_to_clip,
    gmr_robot_id,
    retarget_and_register,
    retarget_clip,
)


# ── quaternion pass-through helper ──────────────────────────────────────
def test_quat_passthrough_identity():
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    out = gmr_quat_wxyz_to_container(q)
    np.testing.assert_allclose(out, q)


def test_quat_passthrough_normalizes_near_unit_input():
    # GMR's IK solve can leave a tiny numerical residual off unit norm.
    q = np.array([[0.70710678, 0.0, 0.70710678, 0.0]]) * 1.0001
    out = gmr_quat_wxyz_to_container(q)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-9)


def test_quat_passthrough_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected"):
        gmr_quat_wxyz_to_container(np.zeros((5, 3)))


def test_quat_passthrough_rejects_non_unit_norm():
    q = np.array([[2.0, 0.0, 0.0, 0.0]])  # norm 2, far outside the atol
    with pytest.raises(ValueError, match="not unit-norm"):
        gmr_quat_wxyz_to_container(q)


# ── robot-id mapping ─────────────────────────────────────────────────────
def test_gmr_robot_id_known_slug():
    assert gmr_robot_id("g1") == "unitree_g1"
    assert gmr_robot_id("t1") == "booster_t1_29dof"


def test_gmr_robot_id_unknown_slug_raises():
    with pytest.raises(ValueError, match="no GMR robot id mapped"):
        gmr_robot_id("nonexistent_robot")


def test_bvh_lafan1_capable_robots_are_all_mapped():
    # Every robot claimed BVH-lafan1-capable must also have a GMR id —
    # a mismatch here would mean the CLI could accept a slug it can't
    # actually resolve.
    assert BVH_LAFAN1_CAPABLE_ROBOTS <= set(GMR_ROBOT_IDS)


def test_default_gmr_python_path_shape():
    p = default_gmr_python(Path("/home/user/tools/GMR"))
    assert str(p).endswith(("bin/python", "Scripts/python.exe"))


# ── fabricated GMR-shaped npz -> our container ──────────────────────────
def _write_fake_gmr_npz(path: Path, *, n=20, n_joints=4) -> None:
    rng = np.random.default_rng(0)
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 2] = np.linspace(0.1, 0.8, n)
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0  # identity orientation throughout
    joint_pos = rng.uniform(-0.5, 0.5, size=(n, n_joints)).astype(np.float32)
    joint_names = np.array(
        ["left_hip_pitch_joint", "right_hip_pitch_joint",
         "left_knee_joint", "right_knee_joint"][:n_joints])
    np.savez_compressed(
        path, root_pos=root_pos, root_quat_wxyz=quat, joint_pos=joint_pos,
        joint_names=joint_names, fps=np.float32(30.0))


def test_gmr_result_to_clip_converts_and_validates(tmp_path: Path):
    npz_path = tmp_path / "raw.npz"
    _write_fake_gmr_npz(npz_path)
    clip = gmr_result_to_clip(npz_path, source_meta={"gmr_robot": "unitree_g1"})
    assert clip["root_pos_z"].shape == (20,)
    assert clip["root_pos_xy"].shape == (20, 2)
    assert clip["root_quat_wxyz"].shape == (20, 4)
    assert clip["joint_pos"].shape == (20, 4)
    assert clip["joint_names"] == [
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_knee_joint", "right_knee_joint"]
    assert clip["fps"] == 30.0
    assert clip["meta"]["source"] == "retarget:gmr"
    assert clip["meta"]["gmr_robot"] == "unitree_g1"


def test_gmr_result_to_clip_clamps_floor_contact_noise(tmp_path: Path):
    npz_path = tmp_path / "raw.npz"
    n = 15
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 2] = np.linspace(-0.02, 0.5, n)   # dips just below ground
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    joint_pos = np.zeros((n, 2), dtype=np.float32)
    np.savez_compressed(
        npz_path, root_pos=root_pos, root_quat_wxyz=quat, joint_pos=joint_pos,
        joint_names=np.array(["a_joint", "b_joint"]), fps=np.float32(30.0))

    clip = gmr_result_to_clip(npz_path, source_meta={})
    assert (clip["root_pos_z"] > 0).all()          # container invariant holds
    assert clip["meta"]["root_z_clamped"] is not None
    assert clip["meta"]["root_z_clamped"]["n_frames"] > 0


def test_gmr_result_to_clip_rejects_real_below_ground(tmp_path: Path):
    npz_path = tmp_path / "raw.npz"
    n = 10
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 2] = np.linspace(-1.0, 0.5, n)     # genuinely below ground
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    joint_pos = np.zeros((n, 1), dtype=np.float32)
    np.savez_compressed(
        npz_path, root_pos=root_pos, root_quat_wxyz=quat, joint_pos=joint_pos,
        joint_names=np.array(["a_joint"]), fps=np.float32(30.0))

    with pytest.raises(RetargetError, match="schema validation"):
        gmr_result_to_clip(npz_path, source_meta={})


# ── driver invocation, monkeypatched subprocess ──────────────────────────
def test_retarget_clip_invokes_subprocess_and_parses_result(tmp_path: Path, monkeypatch):
    src = tmp_path / "source.bvh"
    src.write_text("HIERARCHY\n")  # existence is all retarget_clip checks
    gmr_python = tmp_path / "fake_venv" / "bin" / "python"
    gmr_python.parent.mkdir(parents=True)
    gmr_python.write_text("")
    out_dir = tmp_path / "out"

    captured = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        args_path = Path(cmd[2])
        result_path = Path(cmd[3])
        captured["args"] = json.loads(args_path.read_text())
        captured["cwd"] = cwd
        # Simulate the driver: write a plausible npz at the requested
        # out_npz path, then a success result.json — mirrors
        # `_gmr_driver.py`'s real contract without importing GMR.
        out_npz = Path(captured["args"]["out_npz"])
        _write_fake_gmr_npz(out_npz)
        result_path.write_text(json.dumps({
            "ok": True, "robot": captured["args"]["gmr_robot"],
            "out_npz": str(out_npz),
            "stats": {"n_frames": 20, "fps": 30.0, "n_joints": 4,
                       "joint_names": ["a", "b", "c", "d"],
                       "root_z_range": [0.1, 0.8]},
            "tool_version": "0.2.0", "error": None,
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = retarget_clip(src, "bvh", "g1", out_dir, gmr_python=gmr_python)
    assert result["ok"] is True
    assert result["robot"] == "unitree_g1"
    assert captured["args"]["source_format"] == "bvh"
    assert captured["args"]["bvh_format"] == "lafan1"
    assert captured["args"]["gmr_robot"] == "unitree_g1"
    assert Path(result["out_npz"]).is_file()


def test_retarget_clip_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        retarget_clip(tmp_path / "nope.bvh", "bvh", "g1", tmp_path)


def test_retarget_clip_bad_source_format_raises(tmp_path: Path):
    src = tmp_path / "s.bvh"
    src.write_text("x")
    with pytest.raises(ValueError, match="source_format"):
        retarget_clip(src, "csv", "g1", tmp_path)


def test_retarget_clip_missing_gmr_python_raises(tmp_path: Path):
    src = tmp_path / "s.bvh"
    src.write_text("x")
    with pytest.raises(RetargetError, match="interpreter not found"):
        retarget_clip(src, "bvh", "g1", tmp_path, gmr_python=tmp_path / "nope" / "python")


def test_retarget_clip_driver_reports_failure_is_not_raised(tmp_path: Path, monkeypatch):
    src = tmp_path / "source.bvh"
    src.write_text("HIERARCHY\n")
    gmr_python = tmp_path / "fake_venv" / "bin" / "python"
    gmr_python.parent.mkdir(parents=True)
    gmr_python.write_text("")

    def fake_run(cmd, cwd, capture_output, text, timeout):
        result_path = Path(cmd[3])
        result_path.write_text(json.dumps({
            "ok": False, "robot": "unitree_g1", "out_npz": None,
            "stats": None, "tool_version": None,
            "error": "RuntimeError: IK config not found",
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = retarget_clip(src, "bvh", "g1", tmp_path, gmr_python=gmr_python)
    assert result["ok"] is False
    assert "IK config not found" in result["error"]


def test_retarget_clip_no_result_file_raises(tmp_path: Path, monkeypatch):
    src = tmp_path / "source.bvh"
    src.write_text("HIERARCHY\n")
    gmr_python = tmp_path / "fake_venv" / "bin" / "python"
    gmr_python.parent.mkdir(parents=True)
    gmr_python.write_text("")

    def fake_run(cmd, cwd, capture_output, text, timeout):
        # Simulate a hard crash: driver never wrote result.json.
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Traceback...")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RetargetError, match="no result file"):
        retarget_clip(src, "bvh", "g1", tmp_path, gmr_python=gmr_python)


# ── full pipeline: retarget_and_register + role resolution, tmp root ────
def test_retarget_and_register_writes_clip_and_provenance(tmp_path: Path, monkeypatch):
    src = tmp_path / "source.bvh"
    src.write_text("HIERARCHY\n")
    gmr_python = tmp_path / "fake_venv" / "bin" / "python"
    gmr_python.parent.mkdir(parents=True)
    gmr_python.write_text("")
    lib_root = tmp_path / "lib"

    def fake_run(cmd, cwd, capture_output, text, timeout):
        args = json.loads(Path(cmd[2]).read_text())
        out_npz = Path(args["out_npz"])
        _write_fake_gmr_npz(out_npz)
        Path(cmd[3]).write_text(json.dumps({
            "ok": True, "robot": args["gmr_robot"], "out_npz": str(out_npz),
            "stats": {"n_frames": 20, "fps": 30.0, "n_joints": 4,
                       "joint_names": ["left_hip_pitch_joint",
                                       "right_hip_pitch_joint",
                                       "left_knee_joint", "right_knee_joint"],
                       "root_z_range": [0.1, 0.8]},
            "tool_version": "0.2.0", "error": None,
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    lc = retarget_and_register(
        src, "bvh", "g1", license_="MIT", attribution="test clip",
        text="get up", labels=["getup"], gmr_python=gmr_python, root=lib_root)

    assert lc.clip_id == "source--g1"
    assert lc.clip_path.is_file()
    assert lc.provenance_path.is_file()

    clip = load_clip(lc.clip_path)
    assert clip["joint_names"] == [
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_knee_joint", "right_knee_joint"]

    prov = library.read_provenance("g1", "source--g1", root=lib_root)
    assert prov["retarget"]["tool"] == "GMR"
    assert prov["retarget"]["version"] == "0.2.0"
    assert prov["retarget"]["gmr_robot"] == "unitree_g1"
    assert prov["source"]["kind"] == "retarget"
    assert prov["license"] == "MIT"
    assert prov["tier"] == "K"
    assert prov["content_sha256"] == library.content_sha256(
        lc.clip_path.read_bytes())
    assert prov["source_content_sha256"] == library.content_sha256(
        src.read_bytes())
    assert prov["qc"]["duration_s"] == pytest.approx(19 / 30, abs=1e-4)

    rows = library.rebuild_index(root=lib_root)
    assert len(rows) == 1
    assert rows[0]["clip_id"] == "source--g1"
    assert rows[0]["robot"] == "g1"


def test_retarget_and_register_raises_on_driver_failure(tmp_path: Path, monkeypatch):
    src = tmp_path / "source.bvh"
    src.write_text("HIERARCHY\n")
    gmr_python = tmp_path / "fake_venv" / "bin" / "python"
    gmr_python.parent.mkdir(parents=True)
    gmr_python.write_text("")

    def fake_run(cmd, cwd, capture_output, text, timeout):
        Path(cmd[3]).write_text(json.dumps({
            "ok": False, "robot": "unitree_g1", "out_npz": None,
            "stats": None, "tool_version": None, "error": "boom",
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RetargetError, match="boom"):
        retarget_and_register(
            src, "bvh", "g1", license_="MIT", attribution="x",
            gmr_python=gmr_python, root=tmp_path / "lib")


def test_attach_role_resolution_qc_persists_summary(tmp_path: Path):
    # Register a clip directly via library/reference (no subprocess
    # needed — this test targets the QC-attachment seam in isolation).
    from sculptor.reference import save_clip

    root = tmp_path / "lib"
    clip = {
        "root_pos_z": np.linspace(0.1, 0.8, 10),
        "fps": 30.0,
        "joint_pos": np.zeros((10, 2)),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }
    d = library.clip_dir("g1", "clip1", root=root)
    save_clip(d / library.CLIP_FILENAME, clip)
    prov = library.make_provenance(
        clip_id="clip1", robot="g1", source={"kind": "test"},
        license="MIT", attribution="x", content_sha256_="0" * 64)
    library.write_provenance("g1", "clip1", prov, root=root)

    summary = attach_role_resolution_qc(
        "g1", "clip1", ["left_hip_pitch", "right_hip_pitch"], root=root)
    assert summary["ok"] is True
    assert summary["resolved"] == {
        "left_hip_pitch": "left_hip_pitch_joint",
        "right_hip_pitch": "right_hip_pitch_joint"}

    reloaded = library.read_provenance("g1", "clip1", root=root)
    assert reloaded["qc"]["joint_role_resolution"]["ok"] is True


def test_attach_role_resolution_qc_reports_missing_role(tmp_path: Path):
    from sculptor.reference import save_clip

    root = tmp_path / "lib"
    clip = {
        "root_pos_z": np.linspace(0.1, 0.8, 10),
        "fps": 30.0,
        "joint_pos": np.zeros((10, 1)),
        "joint_names": ["left_hip_pitch_joint"],
    }
    d = library.clip_dir("g1", "clip1", root=root)
    save_clip(d / library.CLIP_FILENAME, clip)
    prov = library.make_provenance(
        clip_id="clip1", robot="g1", source={"kind": "test"},
        license="MIT", attribution="x", content_sha256_="0" * 64)
    library.write_provenance("g1", "clip1", prov, root=root)

    summary = attach_role_resolution_qc(
        "g1", "clip1", ["left_hip_pitch", "right_ankle_roll"], root=root)
    assert summary["ok"] is False
    assert "right_ankle_roll" in summary["missing"]
