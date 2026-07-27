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
import re
from pathlib import Path

import numpy as np
import pytest

from sculptor.reference import save_clip
from sculptor.refs import library
from sculptor.refs.track import (
    DEFAULT_ITERATIONS,
    MEAN_JOINT_ERR_THRESHOLD_RAD,
    MIN_REFERENCE_MOTION_RAD,
    ORIGIN_RELATIVE_MAX_ROOT_Z_M,
    ROOT_Z_RMSE_THRESHOLD_M,
    STATIC_BASELINE_RATIO_MAX,
    TierDCertificate,
    TrackError,
    TrackingErrors,
    build_track_project,
    clip_root_frame,
    compute_tracking_errors,
    downsample_phase_targets,
    generate_tracking_residual_reward_source,
    generate_tracking_reward_source,
    read_donor_adapter_config,
    select_tracking_phase_window,
    track_clip,
    update_provenance_tier_d,
    verify_tierd_certificate,
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
                    robot: str = "g1", tier: str = "K") -> str:
    """Write clip.npz + provenance whose `content_sha256` is the REAL
    sha256 of the on-disk clip.npz bytes. §F7: `verify_tierd_certificate`
    now recomputes this hash from disk, so a placeholder fixture value
    (the historical `"0" * 64`) would make every certificate in this
    suite deny for the wrong reason. Returns the computed hash so
    callers can assert against it."""
    d = library.clip_dir(robot, clip_id, root=root)
    clip_path = d / library.CLIP_FILENAME
    save_clip(clip_path, clip)
    content_sha = library.content_sha256(clip_path.read_bytes())
    prov = library.make_provenance(
        clip_id=clip_id, robot=robot, source={"kind": "test"},
        license="MIT", attribution="x", content_sha256_=content_sha, tier=tier)
    library.write_provenance(robot, clip_id, prov, root=root)
    library.rebuild_index(root=root)
    return content_sha


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


def test_tracking_phase_window_loops_repeatable_translation_suffix():
    fps = 60.0
    t = np.arange(180, dtype=np.float64) / fps
    phase = 2.0 * np.pi * t / 0.75
    joints = np.stack([np.sin(phase), np.cos(phase)], axis=1)
    root = np.stack([
        0.8 * t, np.zeros_like(t), 0.72 + 0.03 * np.sin(phase),
    ], axis=1)
    gravity = np.tile(np.array([0.0, 0.0, -1.0]), (len(t), 1))

    window, mode = select_tracking_phase_window(
        joint_pos=joints, root_pos=root, gravity=gravity, fps=fps)

    assert mode == "loop"
    assert window.start is not None and window.start > 0
    assert 0.3 <= (len(t) - 1 - window.start) / fps <= 1.5


def test_tracking_phase_window_keeps_large_vertical_motion_one_shot():
    fps = 30.0
    t = np.arange(90, dtype=np.float64) / fps
    joints = np.stack([np.sin(t), np.cos(t)], axis=1)
    root = np.stack([
        t, np.zeros_like(t), np.linspace(0.15, 0.80, len(t)),
    ], axis=1)

    window, mode = select_tracking_phase_window(
        joint_pos=joints, root_pos=root, gravity=None, fps=fps)

    assert mode == "hold"
    assert window == slice(None)


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


def test_tracking_first_reward_scores_reference_and_supports_cpu_batch():
    torch = pytest.importorskip("torch")
    src = generate_tracking_residual_reward_source(
        clip=_make_getup_clip(), clip_id="getup1",
    )
    # The immutable prior is deliberately compact enough for complete-module
    # editor responses, even on robots with many joints.
    assert len(src) < 30_000
    ns: dict = {}
    exec(compile(src, "tracking_first_reward", "exec"), ns)  # noqa: S102
    assert ns["REFERENCE_N_PHASES"] == 16
    composition = ns["REWARD_SPEC"]["composition"]
    assert composition["type"] == "reference_tracking_residual"
    assert composition["residual_max"] <= 0.35 * composition["tracking_weight"]

    next_state = {
        "qpos": ns["REFERENCE_JOINT_POS"][0].copy(),
        "qvel": ns["REFERENCE_JOINT_VEL"][0].copy(),
    }
    info = {
        "episode_length": 0.0,
        "step_dt": 1.0 / 30.0,
        # Simulator and reference origins may differ; displacement does not.
        "base_height": 0.74,
        "base_height_delta": 0.0,
        "fallen": 0.0,
    }
    matched, components = ns["compute_reward"]({}, None, next_state, info)
    perturbed_state = {
        "qpos": next_state["qpos"] + 1.0,
        "qvel": next_state["qvel"] + 3.0,
    }
    perturbed_info = dict(
        info, base_height=info["base_height"] + 0.4,
        base_height_delta=0.4)
    perturbed, _ = ns["compute_reward"](
        {}, None, perturbed_state, perturbed_info,
    )
    assert matched > perturbed
    assert components["reference_tracking"] > 0.99
    assert components["residual_task"] == 0.0

    batch_state = {
        key: torch.as_tensor(value).repeat(2, 1)
        for key, value in next_state.items()
    }
    batch_info = {
        "episode_length": torch.zeros(2),
        "step_dt": torch.full((2,), 1.0 / 30.0),
        "base_height": torch.full((2,), info["base_height"]),
        "base_height_delta": torch.zeros(2),
        "fallen": torch.zeros(2),
    }
    rewards, batch_components = ns["compute_reward_batched"](
        batch_state, torch.zeros((2, 1)), batch_state, batch_info,
    )
    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all()
    assert torch.all(batch_components["residual_task"] == 0.0)


def test_tracking_first_locomotion_prior_wraps_repeatable_gait():
    n = 180
    fps = 60.0
    t = np.arange(n, dtype=np.float64) / fps
    phase = 2.0 * np.pi * t / 0.75
    clip = {
        "root_pos_z": 0.05 + 0.02 * np.sin(phase),
        "root_pos_xy": np.stack([0.8 * t, np.zeros_like(t)], axis=1),
        "fps": fps,
        "joint_pos": np.stack([np.sin(phase), np.cos(phase)], axis=1),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }
    src = generate_tracking_residual_reward_source(
        clip=clip, clip_id="translation-gait")
    ns: dict = {}
    exec(compile(src, "looping_tracking", "exec"), ns)  # noqa: S102

    assert ns["REFERENCE_PHASE_MODE"] == "loop"
    assert ns["REWARD_SPEC"]["composition"]["root_height_frame"] == (
        "episode_relative")
    start = ns["_phase_index_scalar"]({
        "episode_length": 0.0, "step_dt": 0.02})
    wrapped = ns["_phase_index_scalar"]({
        "episode_length": ns["REFERENCE_DURATION_S"] / 0.02,
        "step_dt": 0.02,
    })
    assert start == wrapped == 0


def test_tracking_first_validator_allows_hooks_but_freezes_composition():
    from types import SimpleNamespace

    from sculptor.edit import (
        EditValidationError,
        _reference_kernel_hash,
        _reference_tracking_contract,
        _validate_reference_tracking_contract,
    )

    parent_source = generate_tracking_residual_reward_source(
        clip=_make_getup_clip(), clip_id="getup1",
    )
    parent_ns: dict = {}
    exec(compile(parent_source, "parent_tracking", "exec"), parent_ns)  # noqa: S102
    parent_mod = SimpleNamespace(**parent_ns)
    parent = _reference_tracking_contract(parent_mod)
    parent_hash = _reference_kernel_hash(parent_source)
    assert parent is not None and parent_hash is not None

    # Residual hooks are the only intentionally editable surface.
    child_source = parent_source.replace(
        "    return 0.0\n\n\ndef compute_reward",
        "    return 0.1\n\n\ndef compute_reward",
    ).replace(
        "    return torch.zeros_like(like)\n\n\ndef compute_reward_batched",
        "    return torch.full_like(like, 0.1)\n\n\ndef compute_reward_batched",
    )
    child_ns: dict = {}
    exec(compile(child_source, "child_tracking", "exec"), child_ns)  # noqa: S102
    next_state = {
        "qpos": child_ns["REFERENCE_JOINT_POS"][0].copy(),
        "qvel": child_ns["REFERENCE_JOINT_VEL"][0].copy(),
    }
    info = {
        "episode_length": 0.0, "step_dt": 1.0 / 30.0,
        "base_height": 0.74, "base_height_delta": 0.0,
        "fallen": 0.0,
    }
    _, components = child_ns["compute_reward"]({}, None, next_state, info)
    _validate_reference_tracking_contract(
        mod=SimpleNamespace(**child_ns), source=child_source,
        components=components, parent=parent, parent_kernel_hash=parent_hash,
    )

    weakened = child_source.replace("_TRACKING_WEIGHT = 1.0", "_TRACKING_WEIGHT = 0.1")
    weakened_ns: dict = {}
    exec(compile(weakened, "weakened_tracking", "exec"), weakened_ns)  # noqa: S102
    _, weakened_components = weakened_ns["compute_reward"](
        {}, None, next_state, info,
    )
    with pytest.raises(EditValidationError, match="immutable reference tracking"):
        _validate_reference_tracking_contract(
            mod=SimpleNamespace(**weakened_ns), source=weakened,
            components=weakened_components, parent=parent,
            parent_kernel_hash=parent_hash,
        )

    # Temporal semantics are part of the motion prior too: an edit cannot
    # relabel a one-shot reference as looping (or hide its height frame) while
    # leaving the target arrays untouched.
    phase_drift = child_source.replace(
        '"phase_mode": \'hold\'', '"phase_mode": \'loop\'', 1)
    phase_drift_ns: dict = {}
    exec(compile(
        phase_drift, "phase_drift_tracking", "exec"), phase_drift_ns)  # noqa: S102
    _, phase_drift_components = phase_drift_ns["compute_reward"](
        {}, None, next_state, info)
    with pytest.raises(EditValidationError, match="changed phase_mode"):
        _validate_reference_tracking_contract(
            mod=SimpleNamespace(**phase_drift_ns), source=phase_drift,
            components=phase_drift_components, parent=parent,
            parent_kernel_hash=parent_hash,
        )


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
    # A certificate must say which convention it measured root height in.
    assert d["root_frame"] == "absolute"
    assert d["root_z_offset_m"] == 0.0


# ── root-height frame convention ──────────────────────────────────────────
#
# Retargeted AMASS zeroes root translation, so a clip's `root_pos_z` is an
# excursion near 0 while the rollout reports a ~0.74 m world height. Scoring
# them against each other measured the standing height of the robot, not its
# tracking, and made ROOT_Z_RMSE_THRESHOLD_M unreachable for 96% of the
# library. These pin the frame resolution and — importantly — that dividing
# the offset out does NOT let a vertically-wrong rollout through.
STANDING_G1_BASE_M = 0.7624   # measured from the real g1 rollout


def _make_hop_clip(n: int = 30, *, absolute: bool = False) -> dict:
    """A clip that rises 0.25 m, holds, and returns. Origin-relative unless
    `absolute`, in which case the same excursion sits on a standing base."""
    z = np.concatenate([np.zeros(5), np.full(n - 10, 0.25), np.zeros(5)])
    if absolute:
        z = z + STANDING_G1_BASE_M
    joint_pos = np.zeros((n, 2))
    joint_pos[:, 0] = np.linspace(0.0, 0.1, n)
    return {
        "root_pos_z": z, "fps": 30.0, "joint_pos": joint_pos,
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }


def test_clip_root_frame_detects_the_retargeted_amass_convention():
    assert clip_root_frame(_make_hop_clip()) == "origin_relative"
    assert clip_root_frame(_make_hop_clip(absolute=True)) == "absolute"
    # The getup fixture spans 0.2 -> 0.75 m: a real world height.
    assert clip_root_frame(_make_getup_clip(20)) == "absolute"


def test_clip_root_frame_prefers_an_explicit_declaration():
    """A clip that states its convention is believed over the height band."""
    clip = _make_hop_clip()                      # would sniff origin_relative
    clip["root_frame"] = "absolute"
    assert clip_root_frame(clip) == "absolute"


def test_clip_root_frame_falls_back_when_the_declaration_is_junk():
    """A metadata typo must not fail scoring closed."""
    clip = _make_hop_clip()
    clip["root_frame"] = "world-ish"
    assert clip_root_frame(clip) == "origin_relative"


def test_origin_relative_clip_scores_the_excursion_not_the_standing_height():
    """The real bug: a rollout that tracks the motion perfectly, offset by the
    robot's standing height, must certify — the offset is a frame convention,
    not tracking error."""
    clip = _make_hop_clip()
    rollout_z = clip["root_pos_z"] + STANDING_G1_BASE_M
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=rollout_z, rollout_joint_names=clip["joint_names"])
    assert errs.root_frame == "origin_relative"
    assert errs.root_z_rmse_m < 1e-9
    assert errs.feasible
    # The offset is divided out, but recorded — never silently dropped.
    assert errs.root_z_offset_m == pytest.approx(STANDING_G1_BASE_M, abs=1e-9)


def test_origin_relative_scoring_still_fails_a_flat_rollout():
    """The anti-gaming case. Dividing out the offset must not wave through a
    robot that never leaves the ground while the reference hops."""
    clip = _make_hop_clip()
    flat = np.full(clip["root_pos_z"].shape, STANDING_G1_BASE_M)
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=flat, rollout_joint_names=clip["joint_names"])
    assert errs.root_frame == "origin_relative"
    assert errs.mean_joint_err_rad < 1e-9        # joints track perfectly...
    assert errs.root_z_rmse_m > ROOT_Z_RMSE_THRESHOLD_M   # ...height does not
    assert not errs.feasible


def test_origin_relative_scoring_still_fails_an_inverted_rollout():
    """A rollout that crouches exactly when the reference rises has zero
    constant offset and must fail on shape alone."""
    clip = _make_hop_clip()
    inverted = STANDING_G1_BASE_M - clip["root_pos_z"]
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=inverted, rollout_joint_names=clip["joint_names"])
    assert errs.root_z_rmse_m > ROOT_Z_RMSE_THRESHOLD_M
    assert not errs.feasible


def test_absolute_clip_keeps_charging_a_constant_offset_as_error():
    """Unchanged behavior where the clip really is in world coordinates: a
    robot standing 20 cm too high is not tracking, and the frame fix must not
    quietly excuse it."""
    clip = _make_getup_clip(20)
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=clip["root_pos_z"] + 0.2,
        rollout_joint_names=clip["joint_names"])
    assert errs.root_frame == "absolute"
    assert errs.root_z_rmse_m == pytest.approx(0.2, abs=1e-9)
    assert not errs.feasible


# ── the static-pose control ───────────────────────────────────────────────
#
# The first real certification exposed that `mean_joint_err_rad < 0.35` alone
# is not a tracking test: on novel-running-jump-kick--g1 the trained policy
# scored 0.1685 rad, the SAME rollout played backwards scored 0.1691, and
# holding the rollout's time-averaged pose scored 0.1624 — better than the
# policy. Mean absolute error is blind to temporal structure, so a clip whose
# joint excursions are small next to the threshold certifies by standing
# still. These pin the control that closes that hole.
def _moving_clip(n: int = 60, *, amp: float = 0.5) -> dict:
    """A clip whose two joints genuinely swing, so 'hold one pose' is a
    meaningfully worse strategy than tracking."""
    t = np.linspace(0.0, 2 * np.pi, n)
    joint_pos = np.stack([amp * np.sin(t), amp * np.cos(t)], axis=1)
    return {
        "root_pos_z": np.full(n, 0.74), "fps": 30.0, "joint_pos": joint_pos,
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }


def test_a_perfect_tracker_beats_the_static_baseline():
    clip = _moving_clip()
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    assert errs.mean_joint_err_rad < 1e-9
    assert errs.static_baseline_err_rad > 0.2   # standing still would be bad
    assert errs.beats_static_baseline
    assert errs.feasible
    assert errs.motion_ratio == pytest.approx(1.0, abs=1e-6)


def test_standing_still_no_longer_certifies():
    """The headline hole: a rollout that holds one pose has a small mean
    absolute error against a modest-amplitude reference, and used to pass."""
    clip = _moving_clip(amp=0.30)
    frozen = np.tile(clip["joint_pos"].mean(axis=0), (len(clip["joint_pos"]), 1))
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=frozen,
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    # It still clears the absolute joint threshold — that is exactly the hole.
    assert errs.mean_joint_err_rad < MEAN_JOINT_ERR_THRESHOLD_RAD
    # ...but it cannot beat the static control, because it IS the control.
    assert not errs.beats_static_baseline
    assert not errs.feasible
    assert errs.motion_ratio == pytest.approx(0.0, abs=1e-9)


def test_a_time_reversed_rollout_does_not_certify():
    """Playing the reference backwards has the same mean absolute error, so
    only a control that sees temporal structure can reject it."""
    clip = _moving_clip()
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"][::-1].copy(),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    assert not errs.feasible


def test_a_motionless_reference_skips_the_control_rather_than_failing():
    """A constant reference IS tracked by a constant pose; failing it would
    be a false negative, so the vacuous comparison is skipped."""
    n = 40
    clip = {
        "root_pos_z": np.full(n, 0.74), "fps": 30.0,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    assert errs.static_baseline_err_rad < MIN_REFERENCE_MOTION_RAD
    assert errs.beats_static_baseline
    assert errs.feasible


def test_root_only_scoring_is_not_failed_by_the_control():
    """No common joints -> no joint trace -> the control cannot apply."""
    clip = _moving_clip()
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((60, 2)),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=["unrelated_a", "unrelated_b"])
    assert errs.n_common_joints == 0
    assert errs.beats_static_baseline
    assert errs.feasible


def test_phase_clock_tracks_wall_time_not_the_training_budget(tmp_path: Path):
    """`episode_len_steps` used to be `steps_per_iteration` — a count of PPO
    updates. At the 2000 default against ~500-step episodes the reference
    played at quarter speed and the policy never saw past phase 0.25."""
    from sculptor.refs.track import DEFAULT_CONTROL_HZ

    n, fps = 444, 120.0                      # the real composite: 3.70 s
    clip = {
        "root_pos_z": np.full(n, 0.05), "fps": fps,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "config.toml").write_text(
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        '[adapter.config]\ntask_id = "t"\n', encoding="utf-8")
    plan = build_track_project(
        clip=clip, clip_id="c", robot="g1", donor_project=donor,
        project_dir=tmp_path / "proj", steps_per_iteration=2000)

    src = (plan.reward_path).read_text(encoding="utf-8")
    match = re.search(r"^EPISODE_LEN_STEPS = (\d+)", src, re.M)
    assert match, "reward must declare EPISODE_LEN_STEPS"
    got = int(match.group(1))
    # 3.70 s at the task's 50 Hz control rate (200 Hz physics / decimation 4)
    # = 185 steps -- NOT the 2000 training budget.
    assert got == round((n / fps) * DEFAULT_CONTROL_HZ) == 185
    assert got != 2000
    # ...and the reward carries the real duration so it can clock off step_dt
    # rather than trusting the build-time rate at all.
    assert re.search(r"^REFERENCE_DURATION_S = 3\.7", src, re.M)


def test_phase_clock_prefers_step_dt_over_the_assumed_rate():
    """Two build-time rate assumptions have already been wrong (the training
    budget, then 50 Hz for a 25 Hz task). The reward must clock off the
    `step_dt` mjlab publishes, so a wrong build-time rate cannot mistime it."""
    n_phase = 32
    src = generate_tracking_reward_source(
        clip_id="c",
        joint_names=["left_hip_pitch_joint"],
        target_joint_pos=np.zeros((n_phase, 1)),
        target_root_z=np.zeros(n_phase),
        episode_len_steps=9999,        # deliberately absurd build-time value
        duration_s=4.0,
    )
    ns: dict = {}
    exec(compile(src, "tracking_reward", "exec"), ns)  # noqa: S102
    phase_index = ns["_phase_index"]
    # Half the reference elapsed (2.0 s of 4.0 s at 40 Hz) -> half the phases,
    # despite EPISODE_LEN_STEPS claiming the episode is 9999 steps long.
    assert phase_index({"episode_length": 80, "step_dt": 0.025}) == n_phase // 2
    assert phase_index({"episode_length": 0, "step_dt": 0.025}) == 0
    # Past the end clamps to the last phase rather than indexing out of range.
    assert phase_index({"episode_length": 500, "step_dt": 0.025}) == n_phase - 1
    # Without step_dt it falls back to the build-time count.
    assert phase_index({"episode_length": 0}) == 0


def _donor(tmp_path: Path) -> Path:
    d = tmp_path / "donor"
    d.mkdir(exist_ok=True)
    (d / "config.toml").write_text(
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        '[adapter.config]\ntask_id = "t"\n', encoding="utf-8")
    return d


def _clip_of_duration(seconds: float, fps: float = 120.0) -> dict:
    n = max(2, int(round(seconds * fps)))
    return {
        "root_pos_z": np.full(n, 0.05), "fps": fps,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }


def test_episode_is_capped_to_the_reference_duration(tmp_path: Path):
    """Otherwise the episode outruns the phase clock and its tail is the robot
    holding the last frame — and the scorer index-aligns a rollout far longer
    than the reference, comparing mismatched phases."""
    plan = build_track_project(
        clip=_clip_of_duration(3.70), clip_id="c", robot="g1",
        donor_project=_donor(tmp_path), project_dir=tmp_path / "proj")
    spec = json.loads((plan.env_dir / "current.json").read_text())
    assert spec["shared"]["episode_length_s"] == pytest.approx(3.70, abs=1e-3)


def test_capped_episode_still_validates_as_an_env_spec(tmp_path: Path):
    from sculptor.env_spec import validate_env_spec

    plan = build_track_project(
        clip=_clip_of_duration(4.0), clip_id="c", robot="g1",
        donor_project=_donor(tmp_path), project_dir=tmp_path / "proj")
    assert validate_env_spec(
        json.loads((plan.env_dir / "current.json").read_text())) == []


def test_a_clip_too_short_to_cap_is_left_alone(tmp_path: Path):
    """env_spec floors episode_length_s at 2.0 s; a 1 s clip must not write an
    invalid spec, it must simply not cap."""
    plan = build_track_project(
        clip=_clip_of_duration(1.0), clip_id="c", robot="g1",
        donor_project=_donor(tmp_path), project_dir=tmp_path / "proj")
    spec = json.loads((plan.env_dir / "current.json").read_text())
    assert "episode_length_s" not in spec.get("shared", {})


def test_static_baseline_constants_are_documented():
    assert STATIC_BASELINE_RATIO_MAX == 0.80
    assert MIN_REFERENCE_MOTION_RAD == 0.02


def test_certificate_reports_the_control_it_ran():
    """An auditor must be able to see the control's number, not just the
    verdict."""
    clip = _moving_clip()
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"])
    d = errs.to_dict()
    assert d["beats_static_baseline"] is True
    assert d["static_baseline_err_rad"] > 0.2
    assert d["thresholds"]["static_baseline_ratio_max"] == STATIC_BASELINE_RATIO_MAX


def test_origin_relative_threshold_constant_is_physically_separated():
    """The band must sit well below a standing humanoid base and well above
    the retargeted clips' peak excursion, or the sniff is a coin flip."""
    assert ORIGIN_RELATIVE_MAX_ROOT_Z_M == 0.30
    assert ORIGIN_RELATIVE_MAX_ROOT_Z_M < STANDING_G1_BASE_M / 2


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
    content_sha = _register_clip(root, clip, tier="K")

    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0, common_joint_names=["left_hip_pitch_joint"],
        n_common_joints=1)
    rollout_path = tmp_path / "fake_rollout.npz"
    rollout_bytes = b"fake npz payload for hashing"
    rollout_path.write_bytes(rollout_bytes)
    prov = update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2,
        rollout_path=rollout_path, root=root)

    assert prov["tier"] == "D"
    assert prov["tierD"]["iterations"] == 2
    assert prov["tierD"]["rollout_path"] == str(rollout_path)
    assert prov["tierD"]["errors"]["feasible"] is True
    # §audit-finding close: the tierD block now also records the rollout's
    # sha256 and a copy of the clip's content_sha256 at tracking time.
    assert prov["tierD"]["rollout_sha256"] == library.content_sha256(rollout_bytes)
    assert prov["tierD"]["clip_content_sha256"] == content_sha  # _register_clip's real hash

    # Persisted to disk, index rebuilt.
    reloaded = library.read_provenance("g1", "getup1", root=root)
    assert reloaded["tier"] == "D"
    rows = library.read_index(root=root)
    assert any(r["clip_id"] == "getup1" and r["tier"] == "D" for r in rows)


def test_update_provenance_tier_d_feasible_missing_rollout_file_omits_hash(
    tmp_path: Path,
):
    """A `rollout_path` that doesn't actually exist on disk (e.g. a caller
    error) must not crash provenance bookkeeping — `rollout_sha256` is
    just omitted, best-effort, and `verify_tierd_certificate` denies
    cleanly later."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0, common_joint_names=["left_hip_pitch_joint"],
        n_common_joints=1)
    rollout_path = tmp_path / "never_written.npz"
    prov = update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2,
        rollout_path=rollout_path, root=root)

    assert prov["tier"] == "D"
    assert "rollout_sha256" not in prov["tierD"]


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


# ── §audit-finding close: verify_tierd_certificate ─────────────────────────
#
# REFERENCE_BUILD_LOG.md "Audit findings deferred" (Tier-D spoofing): a
# caller-claimed `tier="D"` string must never be trusted; only a
# certificate this module verifies from disk counts. Every check in
# `verify_tierd_certificate`'s docstring gets its own tamper test below —
# each mutates exactly ONE thing off the otherwise-valid fixture.
_ROLLOUT_BYTES = b"legit tracking rollout npz bytes"


def _certify_valid_tier_d(tmp_path: Path, root: Path, *,
                           clip_id: str = "getup1", robot: str = "g1") -> dict:
    """Build a clip whose provenance carries a REAL, internally-consistent
    Tier-D certificate: a rollout file actually on disk INSIDE the
    library root (§F7 containment check — mirrors `track_clip`'s own
    `clip_d / "tierD_rollout.npz"` convention), its sha256 recorded
    correctly, and the clip content hash matching the real clip.npz
    bytes (§F7). Returns the written provenance dict."""
    clip = _make_getup_clip()
    _register_clip(root, clip, clip_id=clip_id, robot=robot, tier="K")
    rollout_path = library.clip_dir(robot, clip_id, root=root) / "tierD_rollout.npz"
    rollout_path.write_bytes(_ROLLOUT_BYTES)
    errs = TrackingErrors(
        mean_joint_err_rad=0.05, max_joint_err_rad=0.1, root_z_rmse_m=0.02,
        duration_coverage=1.0, common_joint_names=["left_hip_pitch_joint"],
        n_common_joints=1)
    assert errs.feasible  # sanity: fixture stats are within threshold
    return update_provenance_tier_d(
        robot=robot, clip_id=clip_id, errors=errs, iterations=2,
        rollout_path=rollout_path, root=root)


def test_verify_tierd_certificate_valid_fixture_returns_certificate(tmp_path: Path):
    root = tmp_path / "lib"
    prov = _certify_valid_tier_d(tmp_path, root)

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)

    assert reason is None
    assert isinstance(cert, TierDCertificate)
    assert cert.robot == "g1"
    assert cert.clip_id == "getup1"
    assert cert.mean_joint_err_rad == pytest.approx(0.05)
    assert cert.root_z_rmse_m == pytest.approx(0.02)
    assert cert.rollout_sha256 == library.content_sha256(_ROLLOUT_BYTES)
    assert cert.clip_content_sha256 == prov["content_sha256"]
    assert cert.rollout_path.is_file()


def test_verify_tierd_certificate_frozen_dataclass(tmp_path: Path):
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert reason is None
    assert cert is not None
    with pytest.raises(Exception):  # noqa: PT011 — frozen dataclass raises FrozenInstanceError
        cert.robot = "go1"  # type: ignore[misc]


def test_verify_tierd_certificate_no_tierd_block_denied(tmp_path: Path):
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")  # never tracked — no tierD block at all

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "no tierD block" in reason


def test_verify_tierd_certificate_missing_clip_denied(tmp_path: Path):
    root = tmp_path / "lib"
    cert, reason = verify_tierd_certificate("g1", "nonexistent", root=root)
    assert cert is None
    assert reason is not None


def test_verify_tierd_certificate_infeasible_run_denied(tmp_path: Path):
    """A clip that was tracked but stayed infeasible (tier never left K)
    must not verify, even though a tierD block exists."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    errs = TrackingErrors(
        mean_joint_err_rad=1.0, max_joint_err_rad=2.0, root_z_rmse_m=0.5,
        duration_coverage=1.0, common_joint_names=[], n_common_joints=0)
    update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2, root=root)

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "out of tolerance" in reason


def _mutate_provenance(root: Path, robot: str, clip_id: str, fn) -> None:
    prov = library.read_provenance(robot, clip_id, root=root)
    fn(prov)
    library.write_provenance(robot, clip_id, prov, root=root)


def test_verify_tierd_certificate_tamper_edited_tier_without_stats_denied(
    tmp_path: Path,
):
    """The 'edited tier' tamper: someone hand-sets provenance.tier="D" and
    tierD.errors.feasible=True but the RAW stats are still bad. The
    recomputed-feasibility check (never trust the stored bool) must catch
    this."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    # A clip that was never actually tracked: hand-fabricate a tierD block
    # with a "feasible": True LIE over bad stats.
    _mutate_provenance(root, "g1", "getup1", lambda p: p.update({
        "tier": "D",
        "tierD": {
            "tracked_at": "2026-01-01T00:00:00Z", "iterations": 1,
            "errors": {"mean_joint_err_rad": 1.5, "root_z_rmse_m": 0.9,
                       "feasible": True},
            "clip_content_sha256": p["content_sha256"],
            "rollout_path": str(tmp_path / "nope.npz"),
            "rollout_sha256": "0" * 64,
        },
    }))

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "out of tolerance" in reason


def test_verify_tierd_certificate_tamper_missing_rollout_file_denied(tmp_path: Path):
    root = tmp_path / "lib"
    prov = _certify_valid_tier_d(tmp_path, root)
    rollout_path = Path(prov["tierD"]["rollout_path"])
    rollout_path.unlink()  # the artifact vanishes after certification

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "missing on disk" in reason


def test_verify_tierd_certificate_tamper_wrong_rollout_hash_denied(tmp_path: Path):
    root = tmp_path / "lib"
    prov = _certify_valid_tier_d(tmp_path, root)
    rollout_path = Path(prov["tierD"]["rollout_path"])
    rollout_path.write_bytes(b"a different payload entirely")  # swapped file

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "sha256 mismatch" in reason


def test_verify_tierd_certificate_tamper_wrong_clip_hash_denied(tmp_path: Path):
    """Simulates the clip being re-ingested/edited after certification: the
    top-level content_sha256 drifts away from what tierD recorded."""
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(root, "g1", "getup1",
                        lambda p: p.update({"content_sha256": "f" * 64}))

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "hash drift" in reason


def test_verify_tierd_certificate_tamper_swapped_clip_bytes_denied(tmp_path: Path):
    """§F7: the pre-existing check 5 only compares
    `tierD.clip_content_sha256` against `provenance.content_sha256` —
    two fields of the SAME provenance.json. A hand-edited file could
    keep both mutually consistent (both wrong) without ever touching
    clip.npz. Swap the clip.npz bytes WITHOUT touching either hash
    field: the two-field comparison still agrees with itself, but the
    hash recomputed from the real file on disk must not."""
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    clip_path = library.clip_dir("g1", "getup1", root=root) / library.CLIP_FILENAME
    clip_path.write_bytes(b"swapped clip.npz payload, not the certified clip")

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "on-disk bytes do not match" in reason


def test_verify_tierd_certificate_missing_clip_file_denied(tmp_path: Path):
    """§F7: a vanished clip.npz (e.g. a partially-deleted library entry)
    must deny with a distinct reason, never crash."""
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    clip_path = library.clip_dir("g1", "getup1", root=root) / library.CLIP_FILENAME
    clip_path.unlink()

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "cannot read clip.npz" in reason


def test_verify_tierd_certificate_rollout_path_escapes_root_denied(tmp_path: Path):
    """§F7 containment: a `tierD.rollout_path` pointing OUTSIDE the
    library root must be rejected even when its bytes hash correctly —
    a hand-edited provenance.json must not be able to point this at an
    arbitrary file elsewhere on disk."""
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)

    escaped = tmp_path / "outside_root_rollout.npz"
    escaped.write_bytes(_ROLLOUT_BYTES)  # same bytes/hash as the real one
    _mutate_provenance(root, "g1", "getup1", lambda p: p["tierD"].update({
        "rollout_path": str(escaped),
    }))

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "resolves outside the library root" in reason


def test_verify_tierd_certificate_tamper_missing_rollout_sha_denied(tmp_path: Path):
    """A tierD block with a rollout_path but no recorded hash at all (e.g.
    the best-effort hashing in `update_provenance_tier_d` failed) must
    deny cleanly rather than silently trusting an unhashed artifact."""
    root = tmp_path / "lib"
    prov = _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(root, "g1", "getup1", lambda p: p["tierD"].pop("rollout_sha256"))

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "no rollout_sha256" in reason


def test_verify_tierd_certificate_out_of_tolerance_boundary(tmp_path: Path):
    """Threshold is a strict `<`, matching `TrackingErrors.feasible` — a
    stat sitting exactly AT the threshold is out of tolerance."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    errs = TrackingErrors(
        mean_joint_err_rad=MEAN_JOINT_ERR_THRESHOLD_RAD, max_joint_err_rad=0.4,
        root_z_rmse_m=0.02, duration_coverage=1.0,
        common_joint_names=["left_hip_pitch_joint"], n_common_joints=1)
    assert not errs.feasible
    update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=1, root=root)

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert "out of tolerance" in reason


def test_verify_tierd_certificate_never_raises_on_corrupt_provenance(tmp_path: Path):
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    prov_path = library.clip_dir("g1", "getup1", root=root) / library.PROVENANCE_FILENAME
    prov_path.write_text("{not valid json", encoding="utf-8")

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert cert is None
    assert reason is not None
