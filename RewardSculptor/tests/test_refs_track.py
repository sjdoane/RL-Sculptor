"""§REFERENCE_TRAJECTORY_PLAN §2.3 Tier D, §11 R4: `sculptor/refs/track.py`
— Tier-D certification (physics-tracking a Tier-K clip in mjlab).

All tests here are CPU/offline: no GPU training, no mjlab import, no
network. The pure-Python pieces (phase downsampling, generated tracking-
reward source, error metrics, donor-config templating, provenance
K->D/infeasible transitions, the `--dry-run` CLI path) are exercised
directly; the real GPU tracking pass is the orchestrator's job later
(see the mission's GATE note), not this suite's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.reference import save_clip
from sculptor.refs import library
from sculptor.refs.track import (
    DEFAULT_ITERATIONS,
    MEAN_JOINT_ERR_THRESHOLD_RAD,
    ROOT_Z_RMSE_THRESHOLD_M,
    TrackError,
    TrackingErrors,
    build_track_project,
    compute_tracking_errors,
    downsample_phase_targets,
    generate_tracking_reward_source,
    read_donor_adapter_config,
    track_clip,
    update_provenance_tier_d,
    write_project_config_toml,
)


# ── fixtures ─────────────────────────────────────────────────────────────
def _make_getup_clip(n: int = 40) -> dict:
    z = np.concatenate([
        np.full(15, 0.2), np.linspace(0.2, 0.75, 15), np.full(10, 0.75)])
    joint_pos = np.zeros((n, 2))
    joint_pos[:, 0] = np.linspace(0.0, 0.3, n)
    joint_pos[:, 1] = np.linspace(0.0, -0.2, n)
    return {
        "root_pos_z": z,
        "fps": 30.0,
        "joint_pos": joint_pos,
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }


def _register_clip(root: Path, clip: dict, *, clip_id: str = "getup1",
                    robot: str = "g1", tier: str = "K") -> None:
    d = library.clip_dir(robot, clip_id, root=root)
    save_clip(d / library.CLIP_FILENAME, clip)
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot, source={"kind": "test"},
        license="MIT", attribution="x", content_sha256_="0" * 64, tier=tier)
    library.write_provenance(robot, clip_id, prov, root=root)
    library.rebuild_index(root=root)


def _write_donor_project(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.toml").write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-G1", '
        'num_envs = 64, device = "cuda:0" }\n'
    )
    return path


# ── phase downsampling ───────────────────────────────────────────────────
def test_downsample_phase_targets_exact_count():
    arr = np.arange(10.0).reshape(10, 1)
    out = downsample_phase_targets(arr, n=5)
    assert out.shape == (5, 1)
    # Deterministic nearest-frame lookup at evenly spaced phases.
    assert out.ravel().tolist() == [0.0, 2.0, 4.0, 5.0, 7.0]


def test_downsample_phase_targets_n_equals_source_length_is_identity_order():
    arr = np.linspace(0, 1, 8).reshape(8, 1)
    out = downsample_phase_targets(arr, n=8)
    assert out.shape == (8, 1)
    # Every original frame should appear, monotonically.
    assert (np.diff(out.ravel()) >= 0).all()


def test_downsample_phase_targets_rejects_empty_array():
    with pytest.raises(ValueError, match="at least 1 frame"):
        downsample_phase_targets(np.zeros((0, 2)), n=4)


def test_downsample_phase_targets_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be"):
        downsample_phase_targets(np.zeros((5, 2)), n=0)


def test_downsample_phase_targets_single_frame_source():
    arr = np.array([[1.0, 2.0]])
    out = downsample_phase_targets(arr, n=4)
    assert out.shape == (4, 2)
    assert (out == 1.0).any() or (out == 2.0).any()  # sanity: values echoed
    np.testing.assert_allclose(out, np.tile(arr, (4, 1)))


# ── tracking-reward source generation ────────────────────────────────────
def test_generate_tracking_reward_source_compiles_and_is_callable():
    target_jp = np.zeros((4, 2))
    target_jp[:, 0] = [0.1, 0.2, 0.3, 0.4]
    target_root_z = np.array([0.7, 0.72, 0.74, 0.76])
    src = generate_tracking_reward_source(
        clip_id="testclip",
        joint_names=["left_knee_joint", "right_knee_joint"],
        target_joint_pos=target_jp, target_root_z=target_root_z,
        episode_len_steps=100)

    ns: dict = {}
    exec(compile(src, "gen_reward", "exec"), ns)  # noqa: S102 — controlled test source
    assert "compute_reward" in ns
    assert "REWARD_SPEC" in ns
    assert isinstance(ns["REWARD_SPEC"], dict)

    compute_reward = ns["compute_reward"]
    next_state = {"qpos": np.zeros(7 + 2)}
    next_state["qpos"][2] = 0.7   # root z matches phase-0 target
    next_state["qpos"][7] = 0.1   # joint 0 matches phase-0 target
    next_state["qpos"][8] = 0.0   # joint 1: target is 0 by default fill
    info = {"episode_length": 0}
    reward, components = compute_reward({}, None, next_state, info)
    assert isinstance(reward, float)
    assert set(components) == {"joint_tracking", "root_tracking"}
    # Near-perfect match at phase 0 -> both kernels near 1.0.
    assert reward > 1.9


def test_generate_tracking_reward_source_phase_clocks_through_episode():
    n_phase = 4
    target_jp = np.zeros((n_phase, 1))
    target_jp[:, 0] = [0.0, 1.0, 2.0, 3.0]
    target_root_z = np.zeros(n_phase)
    src = generate_tracking_reward_source(
        clip_id="c", joint_names=["knee_joint"],
        target_joint_pos=target_jp, target_root_z=target_root_z,
        episode_len_steps=8)
    ns: dict = {}
    exec(compile(src, "gen_reward", "exec"), ns)  # noqa: S102
    compute_reward = ns["compute_reward"]

    # At step 6/8 -> phase 0.75 -> index 3 (last phase target = 3.0).
    next_state = {"qpos": np.zeros(7 + 1)}
    next_state["qpos"][7] = 3.0
    info = {"episode_length": 6}
    reward, _ = compute_reward({}, None, next_state, info)
    assert reward > 1.9  # matches the phase-3 target exactly


def test_generate_tracking_reward_source_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="phase-count"):
        generate_tracking_reward_source(
            clip_id="c", joint_names=["a"],
            target_joint_pos=np.zeros((4, 1)), target_root_z=np.zeros(3),
            episode_len_steps=10)


def test_generate_tracking_reward_source_rejects_joint_name_mismatch():
    with pytest.raises(ValueError, match="joint_names"):
        generate_tracking_reward_source(
            clip_id="c", joint_names=["a", "b"],
            target_joint_pos=np.zeros((4, 1)), target_root_z=np.zeros(4),
            episode_len_steps=10)


def test_generate_tracking_reward_source_raises_on_qpos_too_short():
    src = generate_tracking_reward_source(
        clip_id="c", joint_names=["a", "b", "c"],
        target_joint_pos=np.zeros((2, 3)), target_root_z=np.zeros(2),
        episode_len_steps=10)
    ns: dict = {}
    exec(compile(src, "gen_reward", "exec"), ns)  # noqa: S102
    compute_reward = ns["compute_reward"]
    next_state = {"qpos": np.zeros(5)}  # too short for 3 tracked joints
    with pytest.raises(ValueError, match="too short"):
        compute_reward({}, None, next_state, {"episode_length": 0})


# ── error-metric computation ─────────────────────────────────────────────
def test_compute_tracking_errors_perfect_match_is_near_zero():
    clip = _make_getup_clip(20)
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    assert errs.mean_joint_err_rad < 1e-9
    assert errs.max_joint_err_rad < 1e-9
    assert errs.root_z_rmse_m < 1e-9
    assert errs.feasible
    assert errs.n_common_joints == 2


def test_compute_tracking_errors_known_offset_matches_expected_value():
    clip = _make_getup_clip(20)
    offset = 0.5
    rollout_jp = clip["joint_pos"] + offset
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=rollout_jp,
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    assert errs.mean_joint_err_rad == pytest.approx(offset, abs=1e-9)
    assert errs.max_joint_err_rad == pytest.approx(offset, abs=1e-9)
    assert errs.root_z_rmse_m < 1e-9
    assert not errs.feasible  # 0.5 rad >> MEAN_JOINT_ERR_THRESHOLD_RAD


def test_compute_tracking_errors_root_offset_matches_expected_rmse():
    clip = _make_getup_clip(20)
    z_offset = 0.2
    rollout_z = clip["root_pos_z"] + z_offset
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=rollout_z, rollout_joint_names=clip["joint_names"])
    assert errs.root_z_rmse_m == pytest.approx(z_offset, abs=1e-9)
    assert not errs.feasible  # 0.2 m >> ROOT_Z_RMSE_THRESHOLD_M


def test_compute_tracking_errors_feasibility_boundary_constants():
    # Sanity: the module's documented thresholds match the spec.
    assert MEAN_JOINT_ERR_THRESHOLD_RAD == 0.35
    assert ROOT_Z_RMSE_THRESHOLD_M == 0.12


def test_compute_tracking_errors_no_common_joints_degrades_to_root_only():
    clip = _make_getup_clip(20)
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((20, 3)),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=["some_unrelated_joint", "another", "third"])
    assert errs.n_common_joints == 0
    assert errs.mean_joint_err_rad == 0.0
    assert errs.max_joint_err_rad == 0.0
    assert errs.root_z_rmse_m < 1e-9  # root still scores


def test_compute_tracking_errors_partial_joint_overlap_uses_only_common():
    clip = _make_getup_clip(20)
    # Rollout only exposes one of the clip's two joints, plus an unrelated one.
    rollout_names = ["left_hip_pitch_joint", "unrelated_joint"]
    rollout_jp = np.zeros((20, 2))
    rollout_jp[:, 0] = clip["joint_pos"][:, 0]  # exact match on the common joint
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=rollout_jp,
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=rollout_names)
    assert errs.n_common_joints == 1
    assert errs.common_joint_names == ["left_hip_pitch_joint"]
    assert errs.mean_joint_err_rad < 1e-9


def test_compute_tracking_errors_duration_coverage_short_rollout():
    clip = _make_getup_clip(40)
    short_len = 10
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"][:short_len],
        rollout_root_z=clip["root_pos_z"][:short_len],
        rollout_joint_names=clip["joint_names"])
    assert errs.duration_coverage == pytest.approx(short_len / 40)


def test_tracking_errors_to_dict_shape():
    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0, common_joint_names=["a"], n_common_joints=1)
    d = errs.to_dict()
    assert d["feasible"] is True
    assert d["thresholds"]["mean_joint_err_rad"] == MEAN_JOINT_ERR_THRESHOLD_RAD
    assert d["thresholds"]["root_z_rmse_m"] == ROOT_Z_RMSE_THRESHOLD_M


# ── donor-config templating ───────────────────────────────────────────────
def test_read_donor_adapter_config_reads_class_and_config(tmp_path: Path):
    donor = _write_donor_project(tmp_path / "donor")
    cfg = read_donor_adapter_config(donor)
    assert cfg["class"] == "sculptor.adapters.mjlab.MjlabAdapter"
    assert cfg["config"]["task_id"] == "Mjlab-Velocity-Flat-Unitree-G1"
    assert cfg["config"]["num_envs"] == 64


def test_read_donor_adapter_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(TrackError, match="no config.toml"):
        read_donor_adapter_config(tmp_path / "nope")


def test_read_donor_adapter_config_missing_adapter_section_raises(tmp_path: Path):
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "config.toml").write_text("[kg]\nseeds_path = 'x.yml'\n")
    with pytest.raises(TrackError, match="adapter"):
        read_donor_adapter_config(donor)


def test_write_project_config_toml_roundtrips_via_load_adapter(tmp_path: Path):
    donor = _write_donor_project(tmp_path / "donor")
    cfg = read_donor_adapter_config(donor)
    out_dir = tmp_path / "throwaway"
    path = write_project_config_toml(out_dir, cfg)
    assert path.is_file()

    # Parse it back with the SAME tomllib load_adapter uses, confirming
    # the hand-serialized TOML is valid and round-trips the values.
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["adapter"]["class"] == "sculptor.adapters.mjlab.MjlabAdapter"
    assert parsed["adapter"]["config"]["task_id"] == "Mjlab-Velocity-Flat-Unitree-G1"
    assert parsed["adapter"]["config"]["num_envs"] == 64
    assert parsed["adapter"]["config"]["device"] == "cuda:0"


def test_write_project_config_toml_rejects_unsupported_value_type(tmp_path: Path):
    with pytest.raises(TrackError, match="unsupported TOML value"):
        write_project_config_toml(
            tmp_path / "proj",
            {"class": "x.Y", "config": {"bad": object()}})


# ── build_track_project ───────────────────────────────────────────────────
def test_build_track_project_writes_reward_config_and_env_spec(tmp_path: Path):
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    plan = build_track_project(
        clip=clip, clip_id="getup1", robot="g1", donor_project=donor,
        project_dir=tmp_path / "work")

    assert plan.reward_path.is_file()
    assert plan.config_path.is_file()
    assert plan.joint_names == clip["joint_names"]
    assert plan.iterations == DEFAULT_ITERATIONS

    src = plan.reward_path.read_text()
    assert "compute_reward" in src
    assert "getup1" in src

    assert (plan.env_dir / "current.json").is_file()
    # get-up clip -> derive_eval_reset returns a payload -> eval_reset.json written.
    assert (plan.env_dir / "eval_reset.json").is_file()
    eval_reset = json.loads((plan.env_dir / "eval_reset.json").read_text())
    assert "reset_height_offset_m" in eval_reset


def test_build_track_project_rejects_clip_without_joint_pos(tmp_path: Path):
    donor = _write_donor_project(tmp_path / "donor")
    clip = {"root_pos_z": np.linspace(0.1, 0.8, 20), "fps": 30.0}
    with pytest.raises(TrackError, match="joint_pos/joint_names"):
        build_track_project(
            clip=clip, clip_id="c1", robot="g1", donor_project=donor,
            project_dir=tmp_path / "work")


# ── provenance update: K->D and infeasible paths ─────────────────────────
def test_update_provenance_tier_d_feasible_upgrades_tier(tmp_path: Path):
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0, common_joint_names=["left_hip_pitch_joint"],
        n_common_joints=1)
    rollout_path = tmp_path / "fake_rollout.npz"
    prov = update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2,
        rollout_path=rollout_path, root=root)

    assert prov["tier"] == "D"
    assert prov["tierD"]["iterations"] == 2
    assert prov["tierD"]["rollout_path"] == str(rollout_path)
    assert prov["tierD"]["errors"]["feasible"] is True

    # Persisted to disk, index rebuilt.
    reloaded = library.read_provenance("g1", "getup1", root=root)
    assert reloaded["tier"] == "D"
    rows = library.read_index(root=root)
    assert any(r["clip_id"] == "getup1" and r["tier"] == "D" for r in rows)


def test_update_provenance_tier_d_infeasible_keeps_tier_k(tmp_path: Path):
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    errs = TrackingErrors(
        mean_joint_err_rad=1.0, max_joint_err_rad=2.0, root_z_rmse_m=0.5,
        duration_coverage=1.0, common_joint_names=[], n_common_joints=0)
    prov = update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2, root=root)

    assert prov["tier"] == "K"
    assert prov["tierD"]["feasible"] is False
    assert "rollout_path" not in prov["tierD"]

    reloaded = library.read_provenance("g1", "getup1", root=root)
    assert reloaded["tier"] == "K"
    assert reloaded["tierD"]["feasible"] is False


# ── dry-run pipeline (track_clip) ─────────────────────────────────────────
def test_track_clip_dry_run_builds_project_without_training(tmp_path: Path):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    result = track_clip(
        clip_id="getup1", robot="g1", donor_project=donor,
        dry_run=True, library_root=root, project_dir=tmp_path / "work")

    assert result.dry_run is True
    assert result.errors is None
    assert result.plan.reward_path.is_file()
    assert result.plan.config_path.is_file()
    # Dry-run must not touch provenance's tierD state.
    assert "tierD" not in result.provenance


def test_track_clip_missing_clip_raises(tmp_path: Path):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    with pytest.raises(TrackError, match="no such clip"):
        track_clip(
            clip_id="nonexistent", robot="g1", donor_project=donor,
            dry_run=True, library_root=root)


def test_track_clip_default_project_dir_is_under_clip_dir(tmp_path: Path):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    result = track_clip(
        clip_id="getup1", robot="g1", donor_project=donor,
        dry_run=True, library_root=root)  # no project_dir override

    expected = library.clip_dir("g1", "getup1", root=root) / "tierD_work"
    assert result.plan.project_dir == expected


# ── CLI: refs track --dry-run ─────────────────────────────────────────────
def test_cli_refs_track_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "lib"))
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    runner = CliRunner()
    result = runner.invoke(app, [
        "refs", "track", "--clip-id", "getup1", "--robot", "g1",
        "--donor-project", str(donor), "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run plan" in result.output
    assert "getup1" not in "".join(
        line for line in result.output.splitlines() if "FAILED" in line)


def test_cli_refs_track_missing_clip_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "lib"))
    donor = _write_donor_project(tmp_path / "donor")

    runner = CliRunner()
    result = runner.invoke(app, [
        "refs", "track", "--clip-id", "nope", "--robot", "g1",
        "--donor-project", str(donor), "--dry-run",
    ])
    assert result.exit_code != 0
    assert "FAILED" in result.output


def test_cli_refs_track_help_lists_key_options():
    from typer.testing import CliRunner

    from sculptor.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["refs", "track", "--help"])
    assert result.exit_code == 0
    for opt in ("--clip-id", "--donor-project", "--iterations", "--dry-run"):
        assert opt in result.output
