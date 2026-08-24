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

import copy
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sculptor.reference import save_clip
from sculptor.reference_clock import (
    reference_clock_from_reward_source,
    reference_playback_duration_s,
)
from sculptor.runtime_inputs import environment_artifacts_for_phase
from sculptor.refs import library
from sculptor.refs.track import (
    DEFAULT_ITERATIONS,
    MEAN_JOINT_ERR_THRESHOLD_RAD,
    MIN_REFERENCE_MOTION_RAD,
    N_PHASE_TARGETS,
    ORIGIN_RELATIVE_MAX_ROOT_Z_M,
    ROOT_Z_RMSE_THRESHOLD_M,
    STATIC_BASELINE_RATIO_MAX,
    TierDAdmissionError,
    TierDCertificate,
    TrackError,
    TrackingErrors,
    build_tierd_reference_clock,
    build_tierd_execution_contract,
    bind_tierd_runtime_artifacts,
    build_track_project,
    clip_root_frame,
    compare_tierd_target_contract,
    compute_tracking_errors,
    downsample_phase_targets,
    generate_tracking_residual_reward_source,
    generate_tracking_reward_source,
    projected_gravity_from_quat,
    read_donor_adapter_config,
    require_stage_tierd_admission,
    require_tierd_admission,
    require_tierd_target_compatibility,
    select_tracking_phase_window,
    track_clip,
    update_provenance_tier_d,
    verify_tierd_certificate,
    write_project_config_toml,
    _build_generated_tracker_policy_contract,
    _content_addressed_rollout_name,
    _materialize_tierd_rollout_artifact,
    _prepare_tierd_tracking_preflight,
    _publish_and_verify_tierd_verdict,
    _score_tierd_rollout_artifact,
    _verify_checkpoint_policy_contract_sidecar,
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
        "root_frame": "absolute",
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


def _write_donor_project(
    path: Path,
    *,
    policy_contract: dict | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.toml"
    config_path.write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-G1", '
        'num_envs = 64, device = "cuda:0" }\n'
    )
    contract = copy.deepcopy(policy_contract or _policy_contract())
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    receipt = {
        "schema": "reward-sculptor-tier-d-donor-interface-v1",
        "donor_config_sha256": config_sha,
        "certification_config_sha256": config_sha,
        "policy_contract": contract,
        "policy_contract_sha256": _canonical_sha256(contract),
    }
    (path / "tier_d_interface_contract.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _policy_contract(
    *,
    robot_joints: list[str] | None = None,
    task_id: str = "Mjlab-Velocity-Flat-Unitree-G1",
    sim_timestep_s: float = 0.005,
    decimation: int = 4,
) -> dict:
    joints = robot_joints or [
        "left_hip_pitch_joint", "right_hip_pitch_joint",
    ]
    return {
        "schema": 2,
        "identity": {
            "adapter_class": "sculptor.adapters.mjlab.MjlabAdapter",
            "task_id": task_id,
        },
        "joints": {"ordered_names": joints},
        "actions": {
            "ordered_names": joints,
            "term_names": ["joint_position"],
            "shape": [len(joints)],
        },
        "observations": {
            "ordered_terms": [
                {"name": "base", "source": "base", "shape": [3]},
            ],
            "shape": [3],
            "critic_ordered_terms": [
                {"name": "base", "source": "base", "shape": [3]},
            ],
            "critic_shape": [3],
        },
        # Retained in the full policy digest but deliberately outside the
        # Tier-D physical boundary. This gives tests a way to prove that
        # legitimate network/optimizer changes do not invalidate dynamics
        # evidence for an otherwise identical execution interface.
        "policy": {
            "actor": {
                "hidden_dims": [128, 128],
                "recurrent": {"type": None},
            },
            "critic": {
                "hidden_dims": [128, 128],
                "recurrent": {"type": None},
            },
            "normalizer": {
                "present": False,
                "actor_present": False,
                "critic_present": False,
                "actor_shape": None,
                "critic_shape": None,
            },
        },
        "timing": {
            "sim_timestep_s": sim_timestep_s,
            "decimation": decimation,
            "control_dt_s": sim_timestep_s * decimation,
        },
        "versions": {
            "torch": "2.7",
            "mjlab": "0.3.1",
            "rsl_rl": "3.1.0",
            "adapter": "0.7.0",
        },
    }


def _patch_cpu_preflight_policy_contract(
    monkeypatch: pytest.MonkeyPatch,
    *,
    donor_contract: dict | None = None,
    generated_contract: dict | None = None,
) -> list[tuple[Path, dict]]:
    """Keep dry-run tests CPU/offline while exercising the full boundary."""
    donor = copy.deepcopy(donor_contract or _policy_contract())
    calls: list[tuple[Path, dict]] = []

    def fake_read(project: Path, *, robot: str):
        project = Path(project).resolve()
        calls.append((project, {"robot": robot}))
        config_sha = hashlib.sha256(
            (project / "config.toml").read_bytes()
        ).hexdigest()
        return SimpleNamespace(
            donor_project=project,
            policy_contract=copy.deepcopy(donor),
            donor_config_sha256=config_sha,
            certification_config_sha256=config_sha,
            receipt_sha256="f" * 64,
        )

    monkeypatch.setattr(
        "sculptor.refs.track._read_tierd_donor_interface", fake_read,
    )
    if generated_contract is not None:
        monkeypatch.setattr(
            "sculptor.refs.track._build_generated_tracker_policy_contract",
            lambda *_args, **_kwargs: copy.deepcopy(generated_contract),
        )
    return calls


def _reference_clock_for_clip(
    clip: dict,
    *,
    clip_id: str = "getup1",
    robot: str = "g1",
    n_phase_targets: int = 32,
) -> dict:
    joint_pos = downsample_phase_targets(
        np.asarray(clip["joint_pos"], dtype=np.float64),
        n=n_phase_targets,
    )
    root_z = downsample_phase_targets(
        np.asarray(clip["root_pos_z"], dtype=np.float64),
        n=n_phase_targets,
    )
    gravity = None
    if clip.get("root_quat_wxyz") is not None:
        gravity = downsample_phase_targets(
            projected_gravity_from_quat(clip["root_quat_wxyz"]),
            n=n_phase_targets,
        )
    duration_s = reference_playback_duration_s(
        frame_count=int(np.asarray(clip["joint_pos"]).shape[0]),
        fps=float(clip["fps"]),
    )
    source = generate_tracking_reward_source(
        clip_id=clip_id,
        robot=robot,
        joint_names=list(clip["joint_names"]),
        target_joint_pos=joint_pos,
        target_root_z=root_z,
        target_gravity=gravity,
        root_frame=str(clip["root_frame"]),
        episode_len_steps=max(1, round(duration_s * 50.0)),
        duration_s=duration_s,
    )
    clock = reference_clock_from_reward_source(source)
    assert clock is not None
    return clock


def _clock_conditioned_policy_contract(base: dict, clock: dict) -> dict:
    contract = copy.deepcopy(base)
    term = {
        "name": clock["term_name"],
        "source": clock["source"],
        "shape": list(clock["shape"]),
    }
    base_term = {"name": "base", "source": "base", "shape": [3]}
    contract["schema"] = 4
    contract["reference_clock"] = copy.deepcopy(clock)
    contract["observations"] = {
        "ordered_terms": [base_term, copy.deepcopy(term)],
        "shape": [4],
        "critic_ordered_terms": [copy.deepcopy(base_term), copy.deepcopy(term)],
        "critic_shape": [4],
    }
    return contract


def _execution_contract(
    tmp_path: Path,
    clip: dict,
    *,
    robot: str = "g1",
    policy_contract: dict | None = None,
) -> dict:
    donor = _write_donor_project(tmp_path / "tier_d_donor")
    certification = tmp_path / "tier_d_certification.toml"
    certification.write_text(
        (donor / "config.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    clock = _reference_clock_for_clip(clip, robot=robot)
    tracker_contract = _clock_conditioned_policy_contract(
        policy_contract or _policy_contract(), clock,
    )
    contract = build_tierd_execution_contract(
        donor_project=donor,
        certification_config_path=certification,
        clip_id="getup1",
        robot=robot,
        clip=clip,
        policy_contract=tracker_contract,
        reference_clock=clock,
    )
    reward_sha = "a" * 64
    final_checkpoint_sha = "c" * 64
    train_environment = environment_artifacts_for_phase(
        contract["environment_artifacts"], "train",
    )
    seed_application = {
        "schema": "reward-sculptor-seed-application-v1",
        "applied_seed": 0,
        "python_random": True,
        "numpy_global": True,
        "torch_global": True,
        "env_cfg": True,
        "rl_cfg": True,
    }
    policy_contract_sha = contract["donor"]["policy_contract_sha256"]
    return bind_tierd_runtime_artifacts(
        contract,
        requested_reward_module_sha256=reward_sha,
        train_receipts=[
            {
                "iteration": 1,
                "schema": "reward-sculptor-runner-artifacts-v2",
                "phase": "train",
                "reward_module_sha256": reward_sha,
                "requested_max_iterations": 2000,
                "requested_seed": 0,
                "requested_num_envs": 64,
                "seed_application": seed_application,
                "environment_artifacts": train_environment,
                "env_spec_application": {
                    "schema": "reward-sculptor-env-spec-application-v1",
                    "phase": "train", "requested": [], "applied": [],
                    "dead": [], "errors": [],
                },
                "input_checkpoint_requested_sha256": None,
                "input_checkpoint_loaded_sha256": None,
                "input_checkpoint_load_completed": False,
                "output_checkpoint_sha256": "b" * 64,
                "output_policy_contract_sha256": policy_contract_sha,
                "output_policy_contract_sidecar_sha256": "d" * 64,
            },
            {
                "iteration": 2,
                "schema": "reward-sculptor-runner-artifacts-v2",
                "phase": "train",
                "reward_module_sha256": reward_sha,
                "requested_max_iterations": 2000,
                "requested_seed": 0,
                "requested_num_envs": 64,
                "seed_application": seed_application,
                "environment_artifacts": train_environment,
                "env_spec_application": {
                    "schema": "reward-sculptor-env-spec-application-v1",
                    "phase": "train", "requested": [], "applied": [],
                    "dead": [], "errors": [],
                },
                "input_checkpoint_requested_sha256": "b" * 64,
                "input_checkpoint_loaded_sha256": "b" * 64,
                "input_checkpoint_load_completed": True,
                "output_checkpoint_sha256": final_checkpoint_sha,
                "output_policy_contract_sha256": policy_contract_sha,
                "output_policy_contract_sidecar_sha256": "e" * 64,
            },
        ],
        final_checkpoint_sha256=final_checkpoint_sha,
        requested_steps_per_iteration=2000,
        requested_seed=0,
        requested_num_envs=64,
    )


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


# ── phase downsampling ───────────────────────────────────────────────────
def test_downsample_phase_targets_exact_count():
    arr = np.arange(10.0).reshape(10, 1)
    out = downsample_phase_targets(arr, n=5)
    assert out.shape == (5, 1)
    # Deterministic nearest-frame lookup at evenly spaced phases.
    assert out.ravel().tolist() == [0.0, 2.0, 4.0, 7.0, 9.0]
    np.testing.assert_array_equal(out[-1], arr[-1])


def test_downsample_phase_targets_one_row_is_exact_terminal_sample():
    arr = np.arange(12.0).reshape(6, 2)

    out = downsample_phase_targets(arr, n=1)

    assert out.shape == (1, 2)
    np.testing.assert_array_equal(out[0], arr[-1])


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


def test_tierd_reference_identity_changes_when_only_final_sample_changes():
    base = {
        "joint_names": ["left", "right"],
        "joint_pos": np.zeros((9, 2), dtype=np.float64),
        "root_pos_z": np.full(9, 0.72, dtype=np.float64),
        "root_frame": "absolute",
        "fps": 50.0,
    }
    changed = copy.deepcopy(base)
    changed["joint_pos"][-1] = np.asarray([0.4, -0.3])
    changed["root_pos_z"][-1] = 0.81

    original_clock = build_tierd_reference_clock(
        base, clip_id="terminal-identity", robot="g1", n_phase_targets=4,
    )
    changed_clock = build_tierd_reference_clock(
        changed, clip_id="terminal-identity", robot="g1", n_phase_targets=4,
    )

    assert (
        original_clock["reference_target_sha256"]
        != changed_clock["reference_target_sha256"]
    )
    np.testing.assert_array_equal(
        downsample_phase_targets(changed["joint_pos"], n=4)[-1],
        changed["joint_pos"][-1],
    )


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
        robot="g1",
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
        clip_id="c", robot="g1", joint_names=["knee_joint"],
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
            clip_id="c", robot="g1", joint_names=["a"],
            target_joint_pos=np.zeros((4, 1)), target_root_z=np.zeros(3),
            episode_len_steps=10)


def test_generate_tracking_reward_source_rejects_joint_name_mismatch():
    with pytest.raises(ValueError, match="joint_names"):
        generate_tracking_reward_source(
            clip_id="c", robot="g1", joint_names=["a", "b"],
            target_joint_pos=np.zeros((4, 1)), target_root_z=np.zeros(4),
            episode_len_steps=10)


def test_generate_tracking_reward_source_raises_on_qpos_too_short():
    src = generate_tracking_reward_source(
        clip_id="c", robot="g1", joint_names=["a", "b", "c"],
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
        clip=_make_getup_clip(), clip_id="getup1", robot="g1",
    )
    # The immutable prior is deliberately compact enough for complete-module
    # editor responses, even on robots with many joints.
    assert len(src) < 30_000
    ns: dict = {}
    exec(compile(src, "tracking_first_reward", "exec"), ns)  # noqa: S102
    assert ns["REFERENCE_N_PHASES"] == N_PHASE_TARGETS
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
        # This fixture explicitly declares an absolute world-height frame.
        "base_height": float(ns["REFERENCE_ROOT_Z"][0]),
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


def test_tracking_first_locomotion_prior_preserves_certified_full_schedule():
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
        clip=clip, clip_id="translation-gait", robot="g1")
    ns: dict = {}
    exec(compile(src, "looping_tracking", "exec"), ns)  # noqa: S102

    assert ns["REFERENCE_PHASE_MODE"] == "hold"
    assert ns["REWARD_SPEC"]["composition"]["root_height_frame"] == (
        "origin_relative")
    start = ns["_phase_index_scalar"]({
        "episode_length": 0.0, "step_dt": 0.02})
    completed = ns["_phase_index_scalar"]({
        "episode_length": ns["REFERENCE_DURATION_S"] / 0.02,
        "step_dt": 0.02,
    })
    assert start == 0
    assert completed == ns["REFERENCE_N_PHASES"] - 1


def test_tracking_first_validator_allows_hooks_but_freezes_composition():
    from types import SimpleNamespace

    from sculptor.edit import (
        EditValidationError,
        _reference_kernel_hash,
        _reference_tracking_contract,
        _validate_reference_tracking_contract,
    )

    parent_source = generate_tracking_residual_reward_source(
        clip=_make_getup_clip(), clip_id="getup1", robot="g1",
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
        "base_height": float(child_ns["REFERENCE_ROOT_Z"][0]),
        "base_height_delta": 0.0,
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
        rollout_joint_names=clip["joint_names"], control_hz=clip["fps"])
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
        rollout_joint_names=clip["joint_names"],
        # This "rollout" is a prefix of the clip's OWN frames, so it is sampled
        # at the clip's rate, not the G1 task's 50 Hz. Coverage is a wall-time
        # ratio now, so the rate has to be stated truthfully for the frame
        # ratio and the time ratio to coincide.
        control_hz=clip["fps"])
    assert errs.duration_coverage == pytest.approx((short_len - 1) / (40 - 1))


def test_capped_post_step_rollout_meets_coverage_without_reset_state():
    """A horizon-capped mjlab episode must not need the auto-reset state.

    A 191-frame, 50 Hz reference spans 190 control intervals (3.8 s).  The
    runner records post-step states, excludes the done/reset transition, and
    therefore retains 189 valid transition states.  Those states prove 3.78 s
    of control, which clears the 99% Tier-D coverage gate honestly.
    """
    n_clip = 191
    n_valid = 189
    phase = np.linspace(0.0, 1.0, n_clip)
    clip = {
        "joint_pos": np.stack((phase, -phase), axis=1),
        "joint_names": ["a", "b"],
        "root_pos_z": np.full(n_clip, 0.75),
        "root_frame": "absolute",
        "fps": 50.0,
    }
    rollout_joint_pos = downsample_phase_targets(
        clip["joint_pos"], n=n_valid,
    )
    errs = compute_tracking_errors(
        clip=clip,
        rollout_joint_pos=rollout_joint_pos,
        rollout_root_z=np.full(n_valid, 0.75),
        rollout_joint_names=clip["joint_names"],
        control_hz=50.0,
        rollout_samples_are_post_step=True,
    )
    assert errs.duration_coverage == pytest.approx(189 / 190)
    assert errs.duration_coverage >= 0.99


def test_tracking_errors_to_dict_shape():
    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0, common_joint_names=["a"], n_common_joints=1,
        static_baseline_err_rad=0.2)
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
    robot's standing height, must score zero root error — the offset is a
    frame convention, not tracking error. This tiny fixture deliberately has
    too little joint motion for the independent Tier-D temporal gate."""
    clip = _make_hop_clip()
    rollout_z = clip["root_pos_z"] + STANDING_G1_BASE_M
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=clip["joint_pos"].copy(),
        rollout_root_z=rollout_z, rollout_joint_names=clip["joint_names"],
        control_hz=clip["fps"])
    assert errs.root_frame == "origin_relative"
    assert errs.root_z_rmse_m < 1e-9
    assert not errs.feasible
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
        rollout_joint_names=clip["joint_names"], control_hz=clip["fps"])
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


def test_a_motionless_reference_cannot_claim_temporal_tier_d_tracking():
    """A constant pose has no non-vacuous temporal baseline evidence."""
    n = 40
    clip = {
        "root_pos_z": np.full(n, 0.74), "fps": 30.0,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    }
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=clip["joint_names"], control_hz=clip["fps"])
    assert errs.static_baseline_err_rad < MIN_REFERENCE_MOTION_RAD
    assert not errs.beats_static_baseline
    assert not errs.feasible


def test_root_only_scoring_cannot_claim_full_joint_tier_d_tracking():
    """No common joints means the Tier-D temporal control is unproven."""
    clip = _moving_clip()
    errs = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((60, 2)),
        rollout_root_z=clip["root_pos_z"].copy(),
        rollout_joint_names=["unrelated_a", "unrelated_b"],
        control_hz=clip["fps"])
    assert errs.n_common_joints == 0
    assert not errs.beats_static_baseline
    assert not errs.feasible


def test_phase_clock_tracks_wall_time_not_the_training_budget(tmp_path: Path):
    """`episode_len_steps` used to be `steps_per_iteration` — a count of PPO
    updates. At the 2000 default against ~500-step episodes the reference
    played at quarter speed and the policy never saw past phase 0.25."""
    from sculptor.refs.track import DEFAULT_CONTROL_HZ

    n, fps = 444, 120.0  # 443 sampled intervals: 3.6917 s
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
    # 3.6917 s at the task's 50 Hz control rate (200 Hz / decimation 4)
    # = 185 steps -- NOT the 2000 training budget.
    exact_duration_s = (n - 1) / fps
    assert got == round(exact_duration_s * DEFAULT_CONTROL_HZ) == 185
    assert got != 2000
    # ...and the reward carries the real duration so it can clock off step_dt
    # rather than trusting the build-time rate at all.
    assert re.search(r"^REFERENCE_DURATION_S = 3\.691666", src, re.M)


def test_phase_clock_prefers_step_dt_over_the_assumed_rate():
    """Two build-time rate assumptions have already been wrong (the training
    budget, then 50 Hz for a 25 Hz task). The reward must clock off the
    `step_dt` mjlab publishes, so a wrong build-time rate cannot mistime it."""
    n_phase = 32
    src = generate_tracking_reward_source(
        clip_id="c",
        robot="g1",
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
    assert spec["shared"]["episode_length_s"] == pytest.approx(
        (444 - 1) / 120.0, abs=1e-3,
    )


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


def test_exported_tierd_interface_is_data_only_and_config_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sculptor.refs.track as track_module

    donor = _write_donor_project(tmp_path / "donor")
    base_contract = _policy_contract()
    configured = read_donor_adapter_config(donor)
    monkeypatch.setattr(
        track_module,
        "_resolved_adapter_signature",
        lambda _path: (configured, set(), False),
    )
    monkeypatch.setattr(
        "sculptor.policy_contract.build_project_policy_contract",
        lambda _project, **_kwargs: copy.deepcopy(base_contract),
    )

    receipt_path = track_module.export_tierd_donor_interface(donor)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "reward-sculptor-tier-d-donor-interface-v1"
    assert payload["policy_contract"] == base_contract
    assert payload["policy_contract_sha256"] == _canonical_sha256(base_contract)
    admitted = track_module._read_tierd_donor_interface(donor, robot="g1")
    assert admitted.policy_contract == base_contract
    assert admitted.receipt_sha256 == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()

    (donor / "config.toml").write_text(
        (donor / "config.toml").read_text(encoding="utf-8").replace(
            "num_envs = 64", "num_envs = 32",
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackError, match="stale for config.toml"):
        track_module._read_tierd_donor_interface(donor, robot="g1")


def test_tracking_project_replaces_donor_env_and_eval_reset_paths(
    tmp_path: Path,
) -> None:
    donor = _write_donor_project(tmp_path / "donor")
    (donor / "config.toml").write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-G1", '
        'num_envs = 64, device = "cuda:0", '
        'env_spec_path = "/donor/env/current.json", '
        'eval_reset_path = "/donor/env/eval_reset.json" }\n',
        encoding="utf-8",
    )

    plan = build_track_project(
        clip=_make_getup_clip(),
        clip_id="getup1",
        robot="g1",
        donor_project=donor,
        project_dir=tmp_path / "tracker",
    )

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with plan.config_path.open("rb") as handle:
        generated = tomllib.load(handle)["adapter"]["config"]
    assert "env_spec_path" not in generated
    assert "eval_reset_path" not in generated
    assert (plan.env_dir / "current.json").is_file()
    assert (plan.env_dir / "eval_reset.json").is_file()


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


def test_build_tierd_execution_contract_rejects_donor_task_mismatch(
    tmp_path: Path,
) -> None:
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    clock = _reference_clock_for_clip(clip)
    policy_contract = _clock_conditioned_policy_contract(
        _policy_contract(task_id="Mjlab-Other-Task"), clock,
    )

    with pytest.raises(TrackError, match="donor config task id does not match"):
        build_tierd_execution_contract(
            donor_project=donor,
            certification_config_path=donor / "config.toml",
            clip_id="getup1",
            robot="g1",
            clip=clip,
            policy_contract=policy_contract,
            reference_clock=clock,
        )


def test_build_tierd_execution_contract_requires_generated_tracker_contract(
    tmp_path: Path,
) -> None:
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    clock = _reference_clock_for_clip(clip)

    with pytest.raises(
        TrackError, match="explicit generated tracker policy contract",
    ):
        build_tierd_execution_contract(
            donor_project=donor,
            certification_config_path=donor / "config.toml",
            clip_id="getup1",
            robot="g1",
            clip=clip,
            reference_clock=clock,
        )


def test_tierd_execution_contract_versions_endpoint_inclusive_target_sampling(
    tmp_path: Path,
) -> None:
    contract = _execution_contract(tmp_path, _make_getup_clip())

    assert contract["reference"]["cadence"] == {
        "schema": "generated-target-control-phase-clock-v3",
        "target_table_sampling": "nearest_frame_endpoint_inclusive",
        "target_selection": "floor(phase * n_phase_targets)",
        "phase_interval": "[0,1)",
        "clock": "per_environment_episode_elapsed_control_time",
    }


# ── provenance update: K->D and infeasible paths ─────────────────────────
def test_update_provenance_tier_d_feasible_upgrades_tier(tmp_path: Path):
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    content_sha = _register_clip(root, clip, tier="K")

    clip_dir = library.clip_dir("g1", "getup1", root=root)
    rollout_path = clip_dir / "rollout-candidate.npz"
    execution_contract = _execution_contract(tmp_path, clip)
    errs = _write_valid_tierd_rollout(
        rollout_path, clip=clip, execution_contract=execution_contract,
    )
    rollout_digest = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    retained_path = clip_dir / f"tierD_rollout_{rollout_digest}.npz"
    rollout_path.replace(retained_path)
    rollout_path = retained_path
    rollout_bytes = rollout_path.read_bytes()
    prov = update_provenance_tier_d(
        robot="g1", clip_id="getup1", errors=errs, iterations=2,
        rollout_path=rollout_path,
        execution_contract=execution_contract,
        root=root)

    assert prov["tier"] == "D"
    assert prov["tierD"]["iterations"] == 2
    assert prov["tierD"]["rollout_path"] == str(rollout_path.resolve())
    assert prov["tierD"]["errors"]["feasible"] is True
    # §audit-finding close: the tierD block now also records the rollout's
    # sha256 and a copy of the clip's content_sha256 at tracking time.
    assert prov["tierD"]["rollout_sha256"] == library.content_sha256(rollout_bytes)
    assert prov["tierD"]["clip_content_sha256"] == content_sha  # _register_clip's real hash
    assert len(prov["tierD"]["execution_contract_sha256"]) == 64
    assert len(prov["tierD"]["execution_boundary_sha256"]) == 64

    # Persisted to disk, index rebuilt.
    reloaded = library.read_provenance("g1", "getup1", root=root)
    assert reloaded["tier"] == "D"
    rows = library.read_index(root=root)
    assert any(r["clip_id"] == "getup1" and r["tier"] == "D" for r in rows)


def test_update_provenance_tier_d_feasible_missing_rollout_file_fails_closed(
    tmp_path: Path,
):
    """A feasible verdict is never persisted without retained evidence."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")

    errs = TrackingErrors(
        mean_joint_err_rad=0.1, max_joint_err_rad=0.2, root_z_rmse_m=0.05,
        duration_coverage=1.0,
        common_joint_names=list(clip["joint_names"]),
        n_common_joints=len(clip["joint_names"]),
        static_baseline_err_rad=0.2)
    rollout_path = (
        library.clip_dir("g1", "getup1", root=root) / "tierD_rollout.npz"
    )
    with pytest.raises(TrackError, match="unreadable"):
        update_provenance_tier_d(
            robot="g1", clip_id="getup1", errors=errs, iterations=2,
            rollout_path=rollout_path,
            execution_contract=_execution_contract(tmp_path, clip),
            root=root)


def test_fresh_v4_certificate_rejects_mutable_fixed_rollout_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    contract = _execution_contract(tmp_path, clip)
    fixed = library.clip_dir(
        "g1", "getup1", root=root,
    ) / "tierD_rollout.npz"
    errors = _write_valid_tierd_rollout(
        fixed, clip=clip, execution_contract=contract,
    )
    with pytest.raises(TrackError, match="content-addressed path"):
        update_provenance_tier_d(
            robot="g1", clip_id="getup1", errors=errors, iterations=2,
            rollout_path=fixed, execution_contract=contract, root=root,
        )


def test_failed_provenance_write_preserves_prior_content_addressed_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    prior = _certify_valid_tier_d(tmp_path, root)
    prior_bytes = (
        library.clip_dir("g1", "getup1", root=root)
        / library.PROVENANCE_FILENAME
    ).read_bytes()
    prior_rollout = Path(prior["tierD"]["rollout_path"])
    clip = _make_getup_clip()
    contract = prior["tierD"]["execution_contract"]
    with np.load(prior_rollout, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(payload["trajectory_contract_json"].item()))
    payload["trajectory_contract_json"] = np.asarray(json.dumps(
        metadata, indent=2, sort_keys=True,
    ))
    candidate = tmp_path / "replacement-candidate.npz"
    np.savez_compressed(candidate, **payload)
    errors = _score_tierd_rollout_artifact(
        candidate, clip=clip, execution_contract=contract,
    )
    retained = _materialize_tierd_rollout_artifact(
        candidate,
        clip_dir=library.clip_dir("g1", "getup1", root=root),
        clip=clip,
        execution_contract=contract,
        lane=0,
        expected_errors=errors,
        library_root=root,
    )
    assert retained != prior_rollout

    def fail_write(*args, **kwargs):
        raise OSError("simulated provenance replacement failure")

    monkeypatch.setattr(library, "write_provenance", fail_write)
    with pytest.raises(OSError, match="simulated provenance"):
        update_provenance_tier_d(
            robot="g1", clip_id="getup1", errors=errors, iterations=2,
            rollout_path=retained, execution_contract=contract, root=root,
        )
    assert prior_rollout.is_file()
    assert (
        library.clip_dir("g1", "getup1", root=root)
        / library.PROVENANCE_FILENAME
    ).read_bytes() == prior_bytes
    certificate, reason = verify_tierd_certificate(
        "g1", "getup1", root=root,
    )
    assert reason is None
    assert certificate is not None
    assert certificate.rollout_path == prior_rollout


def test_tierd_promotion_persists_rollout_link_before_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    contract = _execution_contract(tmp_path, clip)
    candidate = tmp_path / "candidate.npz"
    errors = _write_valid_tierd_rollout(
        candidate, clip=clip, execution_contract=contract,
    )
    events: list[str] = []

    def record_fsync(path, *, label):
        del path
        events.append(f"fsync:{label}")

    def record_fsync_descriptor(descriptor, *, label):
        del descriptor
        events.append(f"fsync:{label}")

    original_write = library.write_provenance

    def record_write(*args, **kwargs):
        events.append("write:provenance")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        "sculptor.refs.track._fsync_directory", record_fsync,
    )
    monkeypatch.setattr(
        "sculptor.refs.track._fsync_directory_descriptor",
        record_fsync_descriptor,
    )
    monkeypatch.setattr(library, "write_provenance", record_write)
    retained = _materialize_tierd_rollout_artifact(
        candidate,
        clip_dir=library.clip_dir("g1", "getup1", root=root),
        clip=clip,
        execution_contract=contract,
        lane=0,
        expected_errors=errors,
        library_root=root,
    )
    update_provenance_tier_d(
        robot="g1",
        clip_id="getup1",
        errors=errors,
        iterations=2,
        rollout_path=retained,
        execution_contract=contract,
        root=root,
    )

    assert events[:3] == [
        "fsync:Tier-D rollout",
        "write:provenance",
        "fsync:Tier-D provenance",
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX dirfd confinement")
def test_tierd_rollout_rename_symlink_swap_cannot_escape_or_certify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    contract = _execution_contract(tmp_path, clip)
    candidate = tmp_path / "candidate.npz"
    errors = _write_valid_tierd_rollout(
        candidate, clip=clip, execution_contract=contract,
    )
    clip_dir = library.clip_dir("g1", "getup1", root=root)
    moved = clip_dir.with_name("getup1-moved")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_link = os.link
    exchanged = False

    def exchange_then_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        nonlocal exchanged
        if not exchanged:
            exchanged = True
            clip_dir.rename(moved)
            clip_dir.symlink_to(outside, target_is_directory=True)
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr("sculptor.refs.track.os.link", exchange_then_link)
    with pytest.raises(
        TrackError,
        match="publication coordinate changed during write",
    ):
        _materialize_tierd_rollout_artifact(
            candidate,
            clip_dir=clip_dir,
            clip=clip,
            execution_contract=contract,
            lane=0,
            expected_errors=errors,
            library_root=root,
        )

    retained_name = _content_addressed_rollout_name(
        library.content_sha256(candidate.read_bytes())
    )
    assert not (outside / retained_name).exists()
    assert (moved / retained_name).is_file()
    assert json.loads(
        (moved / library.PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )["tier"] == "K"
    # The coordinate no longer names a confined clip, so no Tier-D authority
    # can be read or minted after the failed retention.
    certificate, reason = verify_tierd_certificate("g1", "getup1", root=root)
    assert certificate is None
    assert reason is not None


def test_update_provenance_tier_d_feasible_requires_execution_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    errs = TrackingErrors(
        mean_joint_err_rad=0.1,
        max_joint_err_rad=0.2,
        root_z_rmse_m=0.05,
        duration_coverage=1.0,
        common_joint_names=list(clip["joint_names"]),
        n_common_joints=len(clip["joint_names"]),
        static_baseline_err_rad=0.2,
    )

    with pytest.raises(TrackError, match="requires an execution contract"):
        update_provenance_tier_d(
            robot="g1",
            clip_id="getup1",
            errors=errs,
            iterations=2,
            root=root,
        )
    assert library.read_provenance("g1", "getup1", root=root)["tier"] == "K"


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


def test_tierd_updates_serialize_global_index_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different clips still share one index transaction domain.

    Without the root-scoped lock, two rebuilds can overlap: a slower scan taken
    before the other provenance commit can then replace ``index.jsonl`` last
    and silently drop the newer row.
    """
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    for clip_id in ("getup_a", "getup_b"):
        _register_clip(root, clip, clip_id=clip_id, tier="K")

    errors = TrackingErrors(
        mean_joint_err_rad=1.0,
        max_joint_err_rad=2.0,
        root_z_rmse_m=0.5,
        duration_coverage=1.0,
        common_joint_names=[],
        n_common_joints=0,
    )
    original_rebuild = library._rebuild_index_unlocked
    state_lock = threading.Lock()
    active_rebuilds = 0
    max_active_rebuilds = 0

    def observed_rebuild(*, root: Path | None = None):
        nonlocal active_rebuilds, max_active_rebuilds
        with state_lock:
            active_rebuilds += 1
            max_active_rebuilds = max(max_active_rebuilds, active_rebuilds)
        try:
            # Make an unlocked implementation overlap reliably.
            time.sleep(0.05)
            return original_rebuild(root=root)
        finally:
            with state_lock:
                active_rebuilds -= 1

    monkeypatch.setattr(library, "_rebuild_index_unlocked", observed_rebuild)
    start = threading.Barrier(3)

    def publish(clip_id: str) -> dict:
        start.wait()
        return update_provenance_tier_d(
            robot="g1",
            clip_id=clip_id,
            errors=errors,
            iterations=2,
            root=root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, clip_id)
            for clip_id in ("getup_a", "getup_b")
        ]
        start.wait()
        results = [future.result(timeout=5.0) for future in futures]

    assert max_active_rebuilds == 1
    assert all(result["tierD"]["feasible"] is False for result in results)
    assert {
        (row["robot"], row["clip_id"])
        for row in library.read_index(root=root)
    } >= {("g1", "getup_a"), ("g1", "getup_b")}


def test_direct_rebuild_and_tierd_mutation_share_one_transaction_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manual rebuild cannot publish a scan from inside a Tier-D mutation."""
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    failed = TrackingErrors(
        mean_joint_err_rad=1.0,
        max_joint_err_rad=2.0,
        root_z_rmse_m=0.5,
        duration_coverage=1.0,
        common_joint_names=[],
        n_common_joints=0,
    )
    original_rebuild = library._rebuild_index_unlocked
    state_lock = threading.Lock()
    active_rebuilds = 0
    max_active_rebuilds = 0

    def observed_rebuild(*, root: Path | None = None):
        nonlocal active_rebuilds, max_active_rebuilds
        with state_lock:
            active_rebuilds += 1
            max_active_rebuilds = max(max_active_rebuilds, active_rebuilds)
        try:
            time.sleep(0.05)
            return original_rebuild(root=root)
        finally:
            with state_lock:
                active_rebuilds -= 1

    monkeypatch.setattr(library, "_rebuild_index_unlocked", observed_rebuild)
    start = threading.Barrier(3)

    def manual_rebuild() -> list[dict]:
        start.wait()
        return library.rebuild_index(root=root)

    def failed_recertification() -> dict:
        start.wait()
        return update_provenance_tier_d(
            robot="g1",
            clip_id="getup1",
            errors=failed,
            iterations=7,
            root=root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rebuild_future = executor.submit(manual_rebuild)
        update_future = executor.submit(failed_recertification)
        start.wait()
        rebuild_future.result(timeout=5.0)
        updated = update_future.result(timeout=5.0)

    assert max_active_rebuilds == 1
    assert updated["tier"] == "K"
    assert updated["tierD"]["iterations"] == 7
    indexed = {
        (row["robot"], row["clip_id"]): row
        for row in library.read_index(root=root)
    }
    assert indexed[("g1", "getup1")]["tier"] == "K"


def test_same_clip_failed_self_check_cannot_clobber_later_recertification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification/invalidation stays ordered with same-clip recertification."""
    root = tmp_path / "lib"
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    contract = _execution_contract(tmp_path, clip)
    candidate = library.clip_dir(
        "g1", "getup1", root=root,
    ) / "rollout-candidate.npz"
    passed = _write_valid_tierd_rollout(
        candidate, clip=clip, execution_contract=contract,
    )
    rollout_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    retained = candidate.with_name(f"tierD_rollout_{rollout_sha}.npz")
    candidate.replace(retained)
    failed = TrackingErrors(
        mean_joint_err_rad=1.0,
        max_joint_err_rad=2.0,
        root_z_rmse_m=0.5,
        duration_coverage=1.0,
        common_joint_names=[],
        n_common_joints=0,
    )
    verification_entered = threading.Event()
    release_verification = threading.Event()

    def blocked_denial(*args, **kwargs):
        del args, kwargs
        verification_entered.set()
        assert release_verification.wait(timeout=5.0)
        return None, "forced self-verification denial"

    monkeypatch.setattr(
        "sculptor.refs.track.verify_tierd_certificate", blocked_denial,
    )

    def publish_then_fail() -> dict:
        return _publish_and_verify_tierd_verdict(
            robot="g1",
            clip_id="getup1",
            errors=passed,
            iterations=2,
            rollout_path=retained,
            execution_contract=contract,
            root=root,
        )

    def publish_later_failure() -> dict:
        return update_provenance_tier_d(
            robot="g1",
            clip_id="getup1",
            errors=failed,
            iterations=7,
            root=root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(publish_then_fail)
        assert verification_entered.wait(timeout=5.0)
        second = executor.submit(publish_later_failure)
        try:
            time.sleep(0.05)
            assert not second.done()
        finally:
            release_verification.set()
        with pytest.raises(TrackError, match="forced self-verification denial"):
            first.result(timeout=5.0)
        later = second.result(timeout=5.0)

    persisted = library.read_provenance("g1", "getup1", root=root)
    assert later["tierD"]["iterations"] == 7
    assert persisted["tierD"]["iterations"] == 7
    assert "verification_error" not in persisted["tierD"]
    assert persisted["tier"] == "K"


# ── dry-run pipeline (track_clip) ─────────────────────────────────────────
def test_track_clip_dry_run_completes_cpu_preflight_without_weights_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    calls = _patch_cpu_preflight_policy_contract(monkeypatch)

    def forbid_adapter_load(_config_path: Path):
        raise AssertionError("dry-run must not construct the GPU-aware adapter")

    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter", forbid_adapter_load,
    )

    result = track_clip(
        clip_id="getup1", robot="g1", donor_project=donor,
        dry_run=True, library_root=root, project_dir=tmp_path / "work")

    assert result.dry_run is True
    assert result.errors is None
    assert result.plan.reward_path.is_file()
    assert result.plan.config_path.is_file()
    assert len(calls) == 1
    receipt = result.preflight_receipt
    assert receipt["schema"] == "reward-sculptor-tier-d-preflight-v1"
    assert receipt["status"] == "ready"
    assert receipt["initialization"] == {
        "donor_project_role": "adapter_interface_and_config_only",
        "first_tracker_training": "fresh_random_policy",
        "donor_policy_weights_loaded": False,
    }
    contract = receipt["unbound_execution_contract"]
    assert contract["donor"]["policy_contract"]["schema"] == 4
    assert "runtime_artifacts" not in contract
    asserted_digest = receipt.pop("receipt_sha256")
    try:
        encoded = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == asserted_digest
    finally:
        receipt["receipt_sha256"] = asserted_digest
    # Dry-run must not touch provenance's tierD state.
    assert "tierD" not in result.provenance


def test_track_clip_missing_clip_raises(tmp_path: Path):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    with pytest.raises(TrackError, match="no such clip"):
        track_clip(
            clip_id="nonexistent", robot="g1", donor_project=donor,
            dry_run=True, library_root=root)


def test_track_clip_requires_explicit_donor_interface_export(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    (donor / "tier_d_interface_contract.json").unlink()
    _register_clip(root, _make_getup_clip(), tier="K")

    with pytest.raises(
        TrackError,
        match=r"sculpt refs export-tierd-interface --donor-project",
    ):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            project_dir=tmp_path / "work",
            dry_run=True,
            library_root=root,
        )


def test_track_clip_default_project_dir_is_unique_and_outside_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    result = track_clip(
        clip_id="getup1", robot="g1", donor_project=donor,
        dry_run=True, library_root=root)  # no project_dir override

    assert result.plan.project_dir.parent == root.parent / "tierD_work"
    assert result.plan.project_dir.name.startswith("g1-getup1-")
    assert result.plan.project_dir.is_dir()
    assert not result.plan.project_dir.is_relative_to(root.resolve())


def test_dry_run_receipt_matches_shared_live_preflight_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    clip_sha = _register_clip(root, clip, tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    direct = _prepare_tierd_tracking_preflight(
        clip=clip,
        clip_content_sha256=clip_sha,
        clip_id="getup1",
        robot="g1",
        donor_project=donor,
        project_dir=tmp_path / "direct",
        iterations=3,
        steps_per_iteration=2000,
        n_episodes=1,
        seed=0,
    )
    dry = track_clip(
        clip_id="getup1",
        robot="g1",
        donor_project=donor,
        dry_run=True,
        library_root=root,
        project_dir=tmp_path / "dry",
    )

    assert dry.preflight_receipt["artifacts"] == direct.receipt["artifacts"]
    assert (
        dry.preflight_receipt["unbound_execution_contract"]
        == direct.execution_contract
    )


def test_live_path_uses_shared_preflight_before_adapter_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import sculptor.refs.track as track_module

    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)
    real_prepare = track_module._prepare_tierd_tracking_preflight
    receipts: list[dict] = []

    def observed_prepare(**kwargs):
        prepared = real_prepare(**kwargs)
        receipts.append(prepared.receipt)
        return prepared

    def stop_at_adapter(_config_path: Path):
        raise RuntimeError("test stop after shared preflight")

    monkeypatch.setattr(
        track_module, "_prepare_tierd_tracking_preflight", observed_prepare,
    )
    monkeypatch.setattr("sculptor.adapters.base.load_adapter", stop_at_adapter)

    with pytest.raises(TrackError, match="test stop after shared preflight"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            dry_run=False,
            library_root=root,
            project_dir=tmp_path / "work",
        )
    assert len(receipts) == 1
    assert receipts[0]["status"] == "ready"
    assert "tierD" not in library.read_provenance(
        "g1", "getup1", root=root,
    )


def test_track_clip_dry_run_rejects_non_authoritative_donor_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    (donor / "config.toml").write_text(
        (donor / "config.toml").read_text(encoding="utf-8").replace(
            "sculptor.adapters.mjlab.MjlabAdapter", "missing.Adapter",
        ),
        encoding="utf-8",
    )
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    with pytest.raises(TrackError, match="trusted local adapter"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            dry_run=True,
            library_root=root,
            project_dir=tmp_path / "work",
        )


def test_track_clip_dry_run_rejects_bad_donor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(
        monkeypatch,
        donor_contract=_policy_contract(task_id="Mjlab-Other-Task"),
    )

    with pytest.raises(TrackError, match="donor config task id does not match"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            dry_run=True,
            library_root=root,
            project_dir=tmp_path / "work",
        )


def test_track_clip_dry_run_rejects_bad_generated_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(
        monkeypatch,
        generated_contract=_policy_contract(),
    )

    with pytest.raises(TrackError, match="must use schema 4"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            dry_run=True,
            library_root=root,
            project_dir=tmp_path / "work",
        )


def test_track_clip_dry_run_rejects_invalid_environment_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import sculptor.refs.track as track_module

    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)
    real_build = track_module.build_track_project

    def build_with_corrupt_env(*args, **kwargs):
        plan = real_build(*args, **kwargs)
        (plan.env_dir / "current.json").write_text(
            "{not-json", encoding="utf-8",
        )
        return plan

    monkeypatch.setattr(track_module, "build_track_project", build_with_corrupt_env)

    with pytest.raises(TrackError, match="cannot capture Tier-D environment"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            dry_run=True,
            library_root=root,
            project_dir=tmp_path / "work",
        )


def test_track_clip_requires_fresh_nonoverlapping_work_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    existing = tmp_path / "existing-work"
    existing.mkdir()
    marker = existing / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(TrackError, match="fresh and non-existing"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            project_dir=existing,
            dry_run=True,
            library_root=root,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (existing / "config.toml").exists()

    nested_in_donor = donor / "tier-d-work"
    donor_config = (donor / "config.toml").read_bytes()
    with pytest.raises(TrackError, match="distinct from donor/library/source"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            project_dir=nested_in_donor,
            dry_run=True,
            library_root=root,
        )
    assert (donor / "config.toml").read_bytes() == donor_config
    assert not nested_in_donor.exists()


def test_track_clip_rejects_untrusted_or_remote_certification_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    _register_clip(root, _make_getup_clip(), tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    untrusted = _write_donor_project(tmp_path / "untrusted")
    (untrusted / "config.toml").write_text(
        (untrusted / "config.toml").read_text(encoding="utf-8").replace(
            "sculptor.adapters.mjlab.MjlabAdapter",
            "sculptor.adapters.gym_sb3.GymSB3Adapter",
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackError, match="trusted local adapter"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=untrusted,
            project_dir=tmp_path / "untrusted-work",
            dry_run=True,
            library_root=root,
        )

    remote = _write_donor_project(tmp_path / "remote")
    with (remote / "config.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[remote]\nhost = "worker.example"\n')
    with pytest.raises(TrackError, match="remote execution is refused"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=remote,
            project_dir=tmp_path / "remote-work",
            dry_run=True,
            library_root=root,
        )

    local = _write_donor_project(tmp_path / "local")
    monkeypatch.setenv("SCULPTOR_REMOTE_HOST", "worker.example")
    with pytest.raises(TrackError, match="remote execution environment"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=local,
            project_dir=tmp_path / "remote-env-work",
            dry_run=True,
            library_root=root,
        )


def test_track_clip_rejects_robot_traversal_and_symlinked_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _write_donor_project(tmp_path / "donor")
    _patch_cpu_preflight_policy_contract(monkeypatch)
    with pytest.raises(TrackError, match="invalid robot namespace"):
        track_clip(
            clip_id="getup1",
            robot="../escape",
            donor_project=donor,
            project_dir=tmp_path / "work",
            dry_run=True,
            library_root=tmp_path / "lib",
        )

    outside = tmp_path / "outside"
    _register_clip(outside, _make_getup_clip(), tier="K")
    root = tmp_path / "linked-lib"
    root.mkdir()
    (root / "g1").symlink_to(outside / "g1", target_is_directory=True)
    with pytest.raises(TrackError, match="must not be a symlink"):
        track_clip(
            clip_id="getup1",
            robot="g1",
            donor_project=donor,
            project_dir=tmp_path / "linked-work",
            dry_run=True,
            library_root=root,
        )


def test_track_clip_binds_structured_root_frame_declaration_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    parent_clip = _make_getup_clip()
    parent_clip.pop("root_frame")
    _register_clip(root, parent_clip, clip_id="legacy", tier="K")

    free_text = library.materialize_root_frame_declaration(
        robot="g1",
        source_clip_id="legacy",
        output_clip_id="free_text_only",
        root_frame="absolute",
        rationale="The fixture values look like world heights.",
        root=root,
    )
    with pytest.raises(TrackError, match="lacks structured evidence"):
        track_clip(
            clip_id=free_text.clip_id,
            robot="g1",
            donor_project=donor,
            project_dir=tmp_path / "free-text-work",
            dry_run=True,
            library_root=root,
        )

    reviewed = library.materialize_root_frame_declaration(
        robot="g1",
        source_clip_id="legacy",
        output_clip_id="reviewed",
        root_frame="absolute",
        rationale="The exporter contract defines world-space root height.",
        evidence_method="deterministic_export_contract",
        reviewer="test-exporter-contract-v1",
        root=root,
    )
    _patch_cpu_preflight_policy_contract(monkeypatch)
    result = track_clip(
        clip_id=reviewed.clip_id,
        robot="g1",
        donor_project=donor,
        project_dir=tmp_path / "reviewed-work",
        dry_run=True,
        library_root=root,
    )
    assert result.preflight_receipt["unbound_execution_contract"][
        "reference"
    ]["root_frame_declaration_evidence"] == reviewed.provenance[
        "source"
    ]["evidence"]


def _register_root_frame_composite(root: Path, *, clip_id: str):
    from sculptor.refs.compose import compose_and_register

    _register_clip(root, _make_getup_clip(), clip_id=f"{clip_id}_a")
    second = _make_getup_clip()
    second["joint_pos"] = second["joint_pos"] + 0.02
    _register_clip(root, second, clip_id=f"{clip_id}_b")
    return compose_and_register(
        "g1",
        [
            {"clip_id": f"{clip_id}_a"},
            {"clip_id": f"{clip_id}_b"},
        ],
        clip_id=clip_id,
        root=root,
    )


def test_track_clip_rejects_missing_composite_root_frame_receipt_before_gpu(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    composite = _register_root_frame_composite(root, clip_id="missing_receipt")
    provenance = copy.deepcopy(composite.provenance)
    provenance["source"].pop("root_frame_inheritance")
    library.write_provenance(
        "g1", composite.clip_id, provenance, root=root,
    )

    with pytest.raises(TrackError, match="root-frame inheritance is invalid"):
        track_clip(
            clip_id=composite.clip_id,
            robot="g1",
            donor_project=donor,
            project_dir=tmp_path / "work",
            dry_run=True,
            library_root=root,
        )
    assert not (tmp_path / "work").exists()


def test_track_clip_rehashes_composite_parents_before_gpu(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    composite = _register_root_frame_composite(root, clip_id="stale_parent")
    changed = _make_getup_clip()
    changed["joint_pos"] = changed["joint_pos"] + 0.10
    save_clip(
        library.clip_dir("g1", "stale_parent_b", root=root)
        / library.CLIP_FILENAME,
        changed,
    )

    with pytest.raises(TrackError, match="root-frame inheritance is invalid"):
        track_clip(
            clip_id=composite.clip_id,
            robot="g1",
            donor_project=donor,
            project_dir=tmp_path / "work",
            dry_run=True,
            library_root=root,
        )
    assert not (tmp_path / "work").exists()


def test_real_cpu_preflight_never_calls_cuda_or_subprocess(
    tmp_path: Path,
) -> None:
    """A cold dry-run cannot import mjlab, construct an adapter, or execute."""
    import subprocess
    import sys

    from sculptor.eval.robot_manifest import robot_joint_names

    root = tmp_path / "lib"
    clip = _make_getup_clip()
    joints = robot_joint_names("Mjlab-Velocity-Flat-Unitree-G1")
    donor = _write_donor_project(
        tmp_path / "donor",
        policy_contract=_policy_contract(robot_joints=joints),
    )
    clip["joint_names"] = joints
    clip["joint_pos"] = np.zeros((len(clip["root_pos_z"]), len(joints)))
    _register_clip(root, clip, tier="K")
    script = f"""
import subprocess
import sys
import torch

assert not any(name == 'mjlab' or name.startswith('mjlab.') for name in sys.modules)

def forbidden(*args, **kwargs):
    del args, kwargs
    raise AssertionError('cold CPU preflight crossed an execution boundary')

subprocess.run = forbidden
subprocess.Popen.__init__ = forbidden
torch.cuda.is_available = forbidden
torch.cuda.mem_get_info = forbidden

from sculptor.adapters import base as adapter_base
adapter_base.load_adapter = forbidden
from sculptor.refs.track import track_clip

assert not any(name == 'mjlab' or name.startswith('mjlab.') for name in sys.modules)
result = track_clip(
    clip_id='getup1',
    robot='g1',
    donor_project={str(donor)!r},
    project_dir={str(tmp_path / 'real-cpu-work')!r},
    dry_run=True,
    library_root={str(root)!r},
)
assert result.dry_run is True
assert result.preflight_receipt['status'] == 'ready'
assert not any(name == 'mjlab' or name.startswith('mjlab.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


# ── CLI: refs track --dry-run ─────────────────────────────────────────────
def test_cli_refs_track_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner

    from sculptor.cli import app

    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "lib"))
    root = tmp_path / "lib"
    donor = _write_donor_project(tmp_path / "donor")
    clip = _make_getup_clip()
    _register_clip(root, clip, tier="K")
    _patch_cpu_preflight_policy_contract(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(app, [
        "refs", "track", "--clip-id", "getup1", "--robot", "g1",
        "--donor-project", str(donor), "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run plan" in result.output
    assert "reward-sculptor-tier-d-preflight-v1" in result.output
    assert '"donor_policy_weights_loaded": false' in result.output
    assert "getup1" not in "".join(
        line for line in result.output.splitlines() if "FAILED" in line)


def test_cli_exports_researcher_visible_tierd_interface_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from sculptor.cli import app

    donor = tmp_path / "donor"
    donor.mkdir()
    receipt = donor / "tier_d_interface_contract.json"
    receipt.write_bytes(b"exact donor interface receipt\n")
    observed: list[Path] = []

    def fake_export(path: Path) -> Path:
        observed.append(Path(path))
        return receipt

    monkeypatch.setattr(
        "sculptor.refs.track.export_tierd_donor_interface",
        fake_export,
    )
    result = CliRunner().invoke(app, [
        "refs",
        "export-tierd-interface",
        "--donor-project",
        str(donor),
    ])
    assert result.exit_code == 0, result.output
    assert observed == [donor]
    assert str(receipt) in result.output
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() in result.output
    assert "policy weights were not loaded" in result.output


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
    compact_help = " ".join(result.output.replace("│", " ").split())
    assert "donor policy weights are never loaded" in compact_help.lower()
    assert "first tracker training starts fresh" in compact_help
    assert "tierD_rollout_<sha256>.npz" in compact_help


# ── §audit-finding close: verify_tierd_certificate ─────────────────────────
#
# REFERENCE_BUILD_LOG.md "Audit findings deferred" (Tier-D spoofing): a
# caller-claimed `tier="D"` string must never be trusted; only a
# certificate this module verifies from disk counts. Every check in
# `verify_tierd_certificate`'s docstring gets its own tamper test below —
# each mutates exactly ONE thing off the otherwise-valid fixture.
def _write_valid_tierd_rollout(
    path: Path,
    *,
    clip: dict,
    execution_contract: dict,
) -> TrackingErrors:
    dt = float(execution_contract["execution_boundary"]["timing"][
        "control_dt_s"
    ])
    duration_s = reference_playback_duration_s(
        frame_count=len(clip["joint_pos"]), fps=float(clip["fps"]),
    )
    n_steps = int(round(duration_s / dt)) + 1
    n_targets = int(execution_contract["reference"]["phase_target_count"])
    target_joint_pos = np.round(
        downsample_phase_targets(clip["joint_pos"], n=n_targets), 5,
    )
    target_root_z = np.round(
        downsample_phase_targets(clip["root_pos_z"], n=n_targets), 5,
    )
    phase = np.clip(
        (np.arange(n_steps, dtype=np.float64) + 1.0) * dt / duration_s,
        0.0,
        0.999999,
    )
    indices = np.floor(phase * n_targets).astype(np.int64)
    joint_pos = target_joint_pos[indices]
    root_z = target_root_z[indices]
    root_pos = np.zeros((n_steps, 1, 3), dtype=np.float32)
    root_pos[:, 0, 2] = root_z
    metadata = {
        "schema": "reward-sculptor-trajectory-v1",
        "layout": ["time", "environment", "feature"],
        "ordered_joint_names": list(execution_contract[
            "execution_boundary"
        ]["joints"]["ordered_names"]),
        "control_dt_s": dt,
        "root_link_pos_w_frame": "world",
        "first_episode_lane": 0,
        "valid_mask": {
            "key": "first_episode_valid_mask",
            "semantics": "true_prefix_before_first_done",
            "invalid_state": "frozen_last_valid_sample",
            "state_samples": "post_step_after_valid_transition",
        },
        "runtime_artifacts": {
            "schema": "reward-sculptor-runner-artifacts-v2",
            "phase": "rollout",
            "reward_module_sha256": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["reward_module_sha256"],
            "checkpoint_sha256": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["checkpoint_sha256"],
            "checkpoint_load_completed": True,
            "environment_artifacts": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["environment_artifacts"],
            "requested_seed": 0,
            "applied_seed": 0,
            "seed_application": {
                "schema": "reward-sculptor-seed-application-v1",
                "applied_seed": 0,
                "python_random": True,
                "numpy_global": True,
                "torch_global": True,
                "env_cfg": True,
                "rl_cfg": False,
            },
            "env_spec_application": {
                "schema": "reward-sculptor-env-spec-application-v1",
                "phase": "rollout", "requested": [], "applied": [],
                "dead": [], "errors": [],
            },
            "eval_reset_application": {
                "schema": "reward-sculptor-eval-reset-application-v1",
                "requested": [], "applied": [], "dead": [], "errors": [],
            },
            "requested_n_episodes": 1,
            "configured_n_episodes": 1,
            "requested_max_episode_steps": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["requested_max_episode_steps"],
            "configured_max_episode_steps": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["requested_max_episode_steps"],
            "requested_task_id": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["requested_task_id"],
            "configured_task_id": execution_contract[
                "runtime_artifacts"
            ]["rollout_requirements"]["requested_task_id"],
            "configured_num_envs": 1,
            "completed_first_episodes": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_pos=joint_pos[:, None, :].astype(np.float32),
        root_link_pos_w=root_pos,
        first_episode_valid_mask=np.ones((n_steps, 1), dtype=bool),
        trajectory_contract_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"),
        )),
    )
    return _score_tierd_rollout_artifact(
        path, clip=clip, execution_contract=execution_contract,
    )


def test_tierd_rollout_rejects_time_dilated_replay(tmp_path: Path) -> None:
    """Repeating a good motion slowly cannot manufacture Tier-D evidence."""
    clip = _make_getup_clip()
    execution_contract = _execution_contract(tmp_path, clip)
    path = tmp_path / "tierD_rollout.npz"
    _write_valid_tierd_rollout(
        path, clip=clip, execution_contract=execution_contract,
    )
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("joint_pos", "root_link_pos_w", "first_episode_valid_mask"):
        payload[key] = np.repeat(payload[key], 2, axis=0)
    np.savez_compressed(path, **payload)

    with pytest.raises(TrackError, match="exceeds the certified reference"):
        _score_tierd_rollout_artifact(
            path, clip=clip, execution_contract=execution_contract,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reward_module_sha256", "f" * 64),
        ("checkpoint_sha256", "e" * 64),
        ("checkpoint_load_completed", False),
    ],
)
def test_tierd_rollout_rejects_runtime_artifact_mismatch(
    tmp_path: Path,
    field: str,
    value: str | bool,
) -> None:
    clip = _make_getup_clip()
    execution_contract = _execution_contract(tmp_path, clip)
    path = tmp_path / "tierD_rollout.npz"
    _write_valid_tierd_rollout(
        path, clip=clip, execution_contract=execution_contract,
    )
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(payload["trajectory_contract_json"].item()))
    metadata["runtime_artifacts"][field] = value
    payload["trajectory_contract_json"] = np.asarray(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
    ))
    np.savez_compressed(path, **payload)

    with pytest.raises(TrackError, match="trajectory contract differs"):
        _score_tierd_rollout_artifact(
            path, clip=clip, execution_contract=execution_contract,
        )


def test_tierd_scores_exact_rounded_phase_table_not_native_clip_frames(
    tmp_path: Path,
) -> None:
    """A 65-frame intra-bin alternation cannot change the 32-table target."""
    frame = np.arange(65, dtype=np.float64)
    clip = {
        "root_pos_z": np.full(65, 0.75),
        "root_frame": "absolute",
        "fps": 64.0,
        "joint_pos": (
            (frame / 64.0) + 0.35 * np.where(frame % 2 == 0, 1.0, -1.0)
        )[:, None],
        "joint_names": ["left_hip_pitch_joint"],
    }
    execution_contract = _execution_contract(
        tmp_path,
        clip,
        policy_contract=_policy_contract(
            robot_joints=["left_hip_pitch_joint"],
        ),
    )
    exact_path = tmp_path / "exact-table.npz"
    exact = _write_valid_tierd_rollout(
        exact_path, clip=clip, execution_contract=execution_contract,
    )
    assert exact.mean_joint_err_rad == pytest.approx(0.0, abs=1e-7)

    with np.load(exact_path, allow_pickle=False) as archive:
        native_payload = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    steps = native_payload["joint_pos"].shape[0]
    dt = execution_contract["execution_boundary"]["timing"]["control_dt_s"]
    native_indices = np.minimum(
        np.floor(
            (np.arange(steps, dtype=np.float64) + 1.0)
            * dt * float(clip["fps"])
            + 1e-12
        ).astype(np.int64),
        64,
    )
    native_payload["joint_pos"] = np.asarray(
        clip["joint_pos"], dtype=np.float32,
    )[native_indices, None, :]
    native_path = tmp_path / "native-frame-replay.npz"
    np.savez_compressed(native_path, **native_payload)
    native = _score_tierd_rollout_artifact(
        native_path, clip=clip, execution_contract=execution_contract,
    )
    assert native.mean_joint_err_rad > 0.1


def test_short_origin_relative_schedule_uses_rounded_table_zero_anchor() -> None:
    clip = {
        "root_pos_z": np.linspace(0.0, 0.16, 17),
        "root_frame": "origin_relative",
        "fps": 40.0,
        "joint_pos": np.linspace(0.0, 0.4, 17)[:, None],
        "joint_names": ["joint"],
    }
    table_root = np.round(
        downsample_phase_targets(clip["root_pos_z"], n=32), 5,
    )
    # Exercise a short rollout whose first observed phase is already above
    # the immutable table-zero anchor.  The scorer must not silently re-anchor
    # the reference at this truncated rollout's first scheduled sample.
    scheduled = table_root[np.array([2, 3, 4], dtype=np.int64)]
    result = compute_tracking_errors(
        clip=clip,
        rollout_joint_pos=np.zeros((3, 1)),
        rollout_root_z=scheduled.copy(),
        rollout_joint_names=["joint"],
        control_hz=50.0,
        rollout_samples_are_post_step=True,
        scheduled_target_joint_pos=np.zeros((3, 1)),
        scheduled_target_root_z=scheduled,
        scheduled_target_root_anchor=float(table_root[0]),
    )
    expected = float(abs(table_root[2] - table_root[0]))
    assert result.root_z_rmse_m == pytest.approx(expected, abs=1e-12)
    assert result.root_z_rmse_m > 0.0


def test_generated_tracker_contract_is_pure_clock_conditioning() -> None:
    base = _policy_contract()
    clock = _reference_clock_for_clip(_make_getup_clip())
    observed = _build_generated_tracker_policy_contract(
        base,
        reference_clock=clock,
    )
    assert observed == _clock_conditioned_policy_contract(base, clock)
    assert base["schema"] == 2
    assert "reference_clock" not in base


def test_checkpoint_sidecar_rejects_runner_contract_digest_mismatch(
    tmp_path: Path,
) -> None:
    from sculptor.policy_contract import contract_fingerprint

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = {"schema": 4, "observations": {"shape": [99]}}
    observed = {"schema": 4, "observations": {"shape": [171]}}
    sidecar = Path(str(checkpoint) + ".policy_contract.json")
    sidecar.write_text(json.dumps({
        "schema": 1,
        "checkpoint_sha256": checkpoint_sha,
        "policy_contract": observed,
        "policy_contract_sha256": contract_fingerprint(observed),
    }, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrackError, match="differs from the generated tracker"):
        _verify_checkpoint_policy_contract_sidecar(
            checkpoint,
            checkpoint_sha256=checkpoint_sha,
            expected_policy_contract=expected,
            expected_policy_contract_sha256=contract_fingerprint(expected),
            expected_sidecar_sha256=hashlib.sha256(
                sidecar.read_bytes()
            ).hexdigest(),
        )


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
    clip_dir = library.clip_dir(robot, clip_id, root=root)
    rollout_path = clip_dir / "rollout-candidate.npz"
    execution_contract = _execution_contract(
        tmp_path, clip, robot=robot,
        policy_contract=_policy_contract(),
    )
    errs = _write_valid_tierd_rollout(
        rollout_path, clip=clip, execution_contract=execution_contract,
    )
    rollout_sha = hashlib.sha256(rollout_path.read_bytes()).hexdigest()
    retained_rollout = clip_dir / f"tierD_rollout_{rollout_sha}.npz"
    rollout_path.replace(retained_rollout)
    rollout_path = retained_rollout
    assert errs.feasible  # sanity: fixture stats are within threshold
    return update_provenance_tier_d(
        robot=robot, clip_id=clip_id, errors=errs, iterations=2,
        rollout_path=rollout_path,
        execution_contract=execution_contract,
        root=root)


def test_verify_tierd_certificate_valid_fixture_returns_certificate(tmp_path: Path):
    root = tmp_path / "lib"
    prov = _certify_valid_tier_d(tmp_path, root)

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)

    assert reason is None
    assert isinstance(cert, TierDCertificate)
    assert cert.robot == "g1"
    assert cert.clip_id == "getup1"
    assert cert.mean_joint_err_rad == pytest.approx(
        prov["tierD"]["errors"]["mean_joint_err_rad"], abs=5e-7,
    )
    assert cert.root_z_rmse_m == pytest.approx(
        prov["tierD"]["errors"]["root_z_rmse_m"], abs=5e-7,
    )
    assert cert.rollout_sha256 == library.content_sha256(
        cert.rollout_path.read_bytes()
    )
    assert cert.clip_content_sha256 == prov["content_sha256"]
    assert cert.execution_contract_sha256 == prov["tierD"][
        "execution_contract_sha256"
    ]
    assert cert.execution_boundary_sha256 == prov["tierD"][
        "execution_boundary_sha256"
    ]
    assert len(cert.certificate_sha256) == 64
    assert cert.rollout_path.is_file()


def test_require_tierd_admission_rejects_stale_certificate_pin(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    admitted = require_tierd_admission("g1", "getup1", root=root)
    provenance = library.read_provenance("g1", "getup1", root=root)
    provenance["tierD"]["tracked_at"] = "2026-08-17T00:00:00Z"
    library.write_provenance("g1", "getup1", provenance, root=root)

    with pytest.raises(TierDAdmissionError, match="certificate sha256"):
        require_tierd_admission(
            "g1",
            "getup1",
            expected_clip_sha256=admitted.clip_content_sha256,
            expected_certificate_sha256=admitted.certificate_sha256,
            root=root,
        )


def test_require_tierd_admission_rejects_stale_execution_contract_pin(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)

    with pytest.raises(TierDAdmissionError, match="execution contract sha256"):
        require_tierd_admission(
            "g1",
            "getup1",
            expected_execution_contract_sha256="0" * 64,
            root=root,
        )
    with pytest.raises(TierDAdmissionError, match="execution boundary sha256"):
        require_tierd_admission(
            "g1",
            "getup1",
            expected_execution_boundary_sha256="0" * 64,
            root=root,
        )


def test_verify_tierd_certificate_legacy_record_without_execution_evidence_denied(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(
        root,
        "g1",
        "getup1",
        lambda p: [
            p["tierD"].pop(key)
            for key in (
                "execution_contract",
                "execution_contract_sha256",
                "execution_boundary_sha256",
            )
        ],
    )

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)

    assert cert is None
    assert "execution contract is missing" in reason


def test_verify_tierd_certificate_tampered_execution_contract_digest_denied(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(
        root,
        "g1",
        "getup1",
        lambda p: p["tierD"]["execution_contract"]["donor"].update(
            {"config_sha256": "f" * 64}
        ),
    )

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)

    assert cert is None
    assert "execution contract sha256 mismatch" in reason


def test_verify_tierd_certificate_reference_cadence_receipt_cannot_go_stale(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)

    def mutate(provenance: dict) -> None:
        tier_d = provenance["tierD"]
        contract = tier_d["execution_contract"]
        reference = contract["reference"]
        reference["fps"] = 25.0
        reference["playback_duration_s"] = (
            (reference["frame_count"] - 1) / reference["fps"]
        )
        reference["clock_contract"]["phase_duration_s"] = reference[
            "playback_duration_s"
        ]
        contract["donor"]["policy_contract"]["reference_clock"] = copy.deepcopy(
            reference["clock_contract"]
        )
        contract["donor"]["policy_contract_sha256"] = _canonical_sha256(
            contract["donor"]["policy_contract"]
        )
        for observation in contract["runtime_artifacts"]["train_observations"]:
            observation["output_policy_contract_sha256"] = contract["donor"][
                "policy_contract_sha256"
            ]
        unsigned = dict(contract)
        unsigned.pop("contract_sha256", None)
        contract["contract_sha256"] = _canonical_sha256(unsigned)
        tier_d["execution_contract_sha256"] = contract["contract_sha256"]

    _mutate_provenance(root, "g1", "getup1", mutate)

    cert, reason = verify_tierd_certificate("g1", "getup1", root=root)

    assert cert is None
    assert "clip fps/cadence differs" in reason


def test_tierd_target_accepts_changed_policy_details_on_same_execution_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert = require_tierd_admission("g1", "getup1", root=root)
    target = _policy_contract()
    target["policy"]["actor"]["hidden_dims"] = [512, 256, 128]
    target["optimizer"] = {"learning_rate": 1e-4}

    assert compare_tierd_target_contract(
        cert.execution_contract,
        target,
        target_robot="g1",
    ) == []
    assert require_tierd_target_compatibility(
        cert,
        tmp_path / "unused-because-contract-injected",
        target_robot="g1",
        target_policy_contract=target,
    ) is cert


def test_tierd_target_rejects_mismatched_donor_task(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert = require_tierd_admission("g1", "getup1", root=root)
    target = _policy_contract(task_id="Mjlab-Other-Task")

    with pytest.raises(TierDAdmissionError, match="identity.task_id differs"):
        require_tierd_target_compatibility(
            cert,
            tmp_path / "target",
            target_robot="g1",
            target_policy_contract=target,
        )


def test_tierd_target_requires_explicit_target_robot(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert = require_tierd_admission("g1", "getup1", root=root)

    with pytest.raises(TierDAdmissionError, match="target robot identity is required"):
        require_tierd_target_compatibility(
            cert,
            tmp_path / "target",
            target_robot="",
            target_policy_contract=_policy_contract(),
        )


def test_tierd_target_rejects_mismatched_control_cadence(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert = require_tierd_admission("g1", "getup1", root=root)
    target = _policy_contract(sim_timestep_s=0.005, decimation=2)

    reasons = compare_tierd_target_contract(
        cert.execution_contract,
        target,
        target_robot="g1",
    )

    assert any("timing.decimation differs" in reason for reason in reasons)
    assert any("timing.control_dt_s differs" in reason for reason in reasons)


def test_tierd_target_rejects_robot_interface_and_software_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    cert = require_tierd_admission("g1", "getup1", root=root)
    target = _policy_contract(robot_joints=["right_hip_pitch_joint", "other"])
    target["versions"]["mjlab"] = "0.4.0"

    reasons = compare_tierd_target_contract(
        cert.execution_contract,
        target,
        target_robot="h1",
    )

    assert any(reason.startswith("robot differs") for reason in reasons)
    assert any("joints.ordered_names differs" in reason for reason in reasons)
    assert any("versions.mjlab differs" in reason for reason in reasons)


def test_require_stage_tierd_admission_needs_exact_robot_clip_and_certificate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    admitted = require_tierd_admission("g1", "getup1", root=root)
    stage = SimpleNamespace(
        name="rise",
        reference_clip_id="getup1",
        reference_tier="D",
        reference_robot="g1",
        reference_clip_sha256=admitted.clip_content_sha256,
        reference_certificate_sha256=admitted.certificate_sha256,
        reference_execution_contract_sha256=(
            admitted.execution_contract_sha256
        ),
        reference_execution_boundary_sha256=(
            admitted.execution_boundary_sha256
        ),
    )

    assert require_stage_tierd_admission(
        stage, expected_robot="g1", root=root,
    ) == admitted
    stage.reference_execution_boundary_sha256 = "0" * 64
    with pytest.raises(TierDAdmissionError, match="execution boundary sha256"):
        require_stage_tierd_admission(
            stage, expected_robot="g1", root=root,
        )
    stage.reference_execution_boundary_sha256 = admitted.execution_boundary_sha256
    stage.reference_robot = "go1"
    with pytest.raises(TierDAdmissionError, match="active training robot"):
        require_stage_tierd_admission(
            stage, expected_robot="g1", root=root,
        )


def test_require_stage_tierd_admission_rejects_legacy_missing_execution_pins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    admitted = require_tierd_admission("g1", "getup1", root=root)
    legacy_stage = SimpleNamespace(
        name="rise",
        reference_clip_id="getup1",
        reference_tier="D",
        reference_robot="g1",
        reference_clip_sha256=admitted.clip_content_sha256,
        reference_certificate_sha256=admitted.certificate_sha256,
    )

    with pytest.raises(
        TierDAdmissionError,
        match="reference_execution_contract_sha256",
    ):
        require_stage_tierd_admission(
            legacy_stage, expected_robot="g1", root=root,
        )


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
    # Deliberately bypass the write-time guard: these tests model an
    # adversary editing provenance.json after a valid certificate was issued.
    path = (
        library.clip_dir(robot, clip_id, root=root)
        / library.PROVENANCE_FILENAME
    )
    path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")


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
    prov = _certify_valid_tier_d(tmp_path, root)

    escaped = tmp_path / "outside_root_rollout.npz"
    retained = Path(prov["tierD"]["rollout_path"])
    escaped.write_bytes(retained.read_bytes())  # exact same bytes/hash
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
    _certify_valid_tier_d(tmp_path, root)
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


def test_verify_tierd_certificate_never_raises_on_non_object_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    _register_clip(root, _make_getup_clip(), tier="K")
    prov_path = (
        library.clip_dir("g1", "getup1", root=root)
        / library.PROVENANCE_FILENAME
    )
    prov_path.write_text("[]", encoding="utf-8")

    certificate, reason = verify_tierd_certificate(
        "g1", "getup1", root=root,
    )
    assert certificate is None
    assert "JSON object" in reason


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        (lambda provenance: provenance.__setitem__("schema", 1), "schema"),
        (
            lambda provenance: provenance.__setitem__("robot", "t1"),
            "provenance.robot",
        ),
        (
            lambda provenance: provenance.__setitem__("clip_id", "other"),
            "provenance.clip_id",
        ),
    ],
)
def test_verify_tierd_certificate_rejects_legacy_or_misscoped_provenance(
    tmp_path: Path,
    mutation,
    reason_fragment: str,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(root, "g1", "getup1", mutation)

    certificate, reason = verify_tierd_certificate(
        "g1", "getup1", root=root,
    )
    assert certificate is None
    assert reason_fragment in reason


@pytest.mark.parametrize("iterations", ["oops", 0, 3])
def test_verify_tierd_certificate_rejects_invalid_or_stale_iterations(
    tmp_path: Path,
    iterations,
) -> None:
    root = tmp_path / "lib"
    _certify_valid_tier_d(tmp_path, root)
    _mutate_provenance(
        root,
        "g1",
        "getup1",
        lambda provenance: provenance["tierD"].__setitem__(
            "iterations", iterations,
        ),
    )

    certificate, reason = verify_tierd_certificate(
        "g1", "getup1", root=root,
    )
    assert certificate is None
    assert "tierD.iterations" in reason


def test_update_provenance_rechecks_schema_after_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lib"
    provenance = _certify_valid_tier_d(tmp_path, root)
    rollout = Path(provenance["tierD"]["rollout_path"])
    contract = provenance["tierD"]["execution_contract"]
    clip = _make_getup_clip()
    errors = _score_tierd_rollout_artifact(
        rollout, clip=clip, execution_contract=contract,
    )
    _mutate_provenance(
        root, "g1", "getup1",
        lambda current: current.__setitem__("schema", 1),
    )

    with pytest.raises(TrackError, match="invalid or mis-scoped provenance"):
        update_provenance_tier_d(
            robot="g1",
            clip_id="getup1",
            errors=errors,
            iterations=2,
            rollout_path=rollout,
            execution_contract=contract,
            root=root,
        )


# ── orientation tracking (OGMP Eq. 8) ───────────────────────────────────
def _load_src(src, tmp_path, name="orient_mod"):
    import importlib.util
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _orient_src(target_gravity, n=8, j=4):
    from sculptor.refs.track import generate_tracking_reward_source
    return generate_tracking_reward_source(
        clip_id="t", robot="g1", joint_names=[f"j{i}" for i in range(j)],
        target_joint_pos=np.zeros((n, j)), target_root_z=np.zeros(n),
        episode_len_steps=100, duration_s=2.0, target_gravity=target_gravity)


def test_projected_gravity_matches_the_rotation_matrix_definition():
    """R^T @ [0,0,-1], checked against an independent construction rather than
    trusting the sign conventions in the closed form."""
    from sculptor.refs.track import projected_gravity_from_quat

    def rot(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    rng = np.random.default_rng(0)
    qs = rng.normal(size=(64, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    want = np.stack([rot(q).T @ np.array([0.0, 0.0, -1.0]) for q in qs])
    assert np.allclose(projected_gravity_from_quat(qs), want, atol=1e-12)


def test_an_upright_quaternion_points_gravity_straight_down():
    from sculptor.refs.track import projected_gravity_from_quat

    got = projected_gravity_from_quat(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(got, [[0.0, 0.0, -1.0]])


def test_unnormalized_quaternions_are_normalized_not_rejected():
    from sculptor.refs.track import projected_gravity_from_quat

    got = projected_gravity_from_quat(np.array([[2.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(got, [[0.0, 0.0, -1.0]])


def test_orientation_term_rewards_matching_attitude(tmp_path):
    mod = _load_src(_orient_src(np.tile([0.0, 0.0, -1.0], (8, 1))), tmp_path)
    info = {"episode_length": 0, "step_dt": 0.02}
    upright = {"qpos": np.zeros(11), "projected_gravity_b": np.array([0.0, 0.0, -1.0])}
    tipped = {"qpos": np.zeros(11), "projected_gravity_b": np.array([1.0, 0.0, 0.0])}
    _, c_up = mod.compute_reward(None, None, upright, info)
    _, c_tip = mod.compute_reward(None, None, tipped, info)
    assert c_up["orientation_tracking"] == pytest.approx(1.0)
    assert c_tip["orientation_tracking"] < 0.2


def test_scalar_and_batched_orientation_agree(tmp_path):
    torch = pytest.importorskip("torch")
    mod = _load_src(_orient_src(np.tile([0.0, 0.0, -1.0], (8, 1))), tmp_path)
    grav = np.array([0.30, -0.20, -0.93])
    grav = grav / np.linalg.norm(grav)
    _, c = mod.compute_reward(
        None, None, {"qpos": np.zeros(11), "projected_gravity_b": grav},
        {"episode_length": 0, "step_dt": 0.02})
    _, cb = mod.compute_reward_batched(
        None, None,
        {"qpos": torch.zeros(3, 4),
         "projected_gravity_b": torch.tensor([grav] * 3, dtype=torch.float32)},
        {"episode_length": torch.zeros(3), "step_dt": torch.full((3,), 0.02)})
    assert float(cb["orientation_tracking"].mean()) == pytest.approx(
        c["orientation_tracking"], abs=1e-5)


def test_a_clip_without_orientation_is_unchanged(tmp_path):
    """Backward compatibility: no quaternion means no orientation term at all,
    not a fabricated upright target that would penalize a legitimate lean."""
    mod = _load_src(_orient_src(None), tmp_path, name="no_orient")
    assert mod.TARGET_GRAVITY is None
    assert mod.ORIENTATION_ERR_WEIGHT == 0.0
    _, c = mod.compute_reward(
        None, None,
        {"qpos": np.zeros(11), "projected_gravity_b": np.array([1.0, 0.0, 0.0])},
        {"episode_length": 0, "step_dt": 0.02})
    assert "orientation_tracking" not in c


def test_a_misshaped_gravity_target_is_rejected_at_build_time():
    with pytest.raises(ValueError, match="n_phase"):
        _orient_src(np.zeros((3, 3)))          # 3 rows vs 8 phase targets
    with pytest.raises(ValueError, match="n_phase"):
        _orient_src(np.zeros((8, 4)))          # not a 3-vector


def test_gravity_targets_stay_unit_through_downsampling(tmp_path):
    """mjlab's observed `projected_gravity_b` is a unit vector, so the target
    has to be one too — a shrunken target charges a standing error against a
    perfectly upright robot. `downsample_phase_targets` selects nearest frames
    rather than interpolating, so this holds; pin it, because an interpolating
    resampler would silently break it."""
    from sculptor.refs.track import (
        downsample_phase_targets, projected_gravity_from_quat)

    rng = np.random.default_rng(3)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = downsample_phase_targets(projected_gravity_from_quat(q), n=16)
    assert g.shape == (16, 3)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)


def test_orientation_error_is_measured_when_both_channels_exist():
    from sculptor.refs.track import compute_tracking_errors

    n = 40
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))   # upright clip
    clip = {"joint_pos": np.zeros((n, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(n), "root_quat_wxyz": quat, "fps": 30.0}
    upright = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    tipped = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))

    matched = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=np.zeros(n), rollout_joint_names=["a", "b"],
        rollout_gravity=upright)
    wrong = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=np.zeros(n), rollout_joint_names=["a", "b"],
        rollout_gravity=tipped)
    assert matched.orientation_err == pytest.approx(0.0, abs=1e-9)
    assert wrong.orientation_err == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_orientation_does_not_gate_certification():
    """Deliberate: nothing has ever passed Tier-D, so there is no evidence for
    an achievable orientation threshold. A completely inverted rollout must
    still be `feasible` if joints and root height pass — the number is
    reported so a threshold can be set from data later."""
    from sculptor.refs.track import TrackingErrors

    e = TrackingErrors(
        mean_joint_err_rad=0.01, max_joint_err_rad=0.02, root_z_rmse_m=0.01,
        duration_coverage=1.0, static_baseline_err_rad=0.5,
        common_joint_names=["joint"], n_common_joints=1,
        orientation_err=2.0)               # fully inverted
    assert e.feasible is True
    assert e.to_dict()["orientation_err"] == 2.0


def test_orientation_is_skipped_without_a_rollout_gravity_channel():
    """Older trajectories predate the channel; absence must not fail a run."""
    from sculptor.refs.track import compute_tracking_errors

    n = 20
    clip = {"joint_pos": np.zeros((n, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(n),
            "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            "fps": 30.0}
    e = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=np.zeros(n), rollout_joint_names=["a", "b"],
        rollout_gravity=None)
    assert e.orientation_err == 0.0


def test_orientation_is_skipped_when_the_clip_has_no_quaternion():
    from sculptor.refs.track import compute_tracking_errors

    n = 20
    clip = {"joint_pos": np.zeros((n, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(n), "fps": 30.0}
    e = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n, 2)),
        rollout_root_z=np.zeros(n), rollout_joint_names=["a", "b"],
        rollout_gravity=np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)))
    assert e.orientation_err == 0.0


def test_duration_coverage_is_wall_time_not_frame_count():
    """A 361-sample 120 fps clip and 151-sample 50 Hz rollout both contain
    3.0 s of sample intervals. Dividing frame counts reports about 41.8% for a
    rollout that ran the entire motion."""
    from sculptor.refs.track import compute_tracking_errors

    n_clip, n_roll = 361, 151
    clip = {"joint_pos": np.zeros((n_clip, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(n_clip), "fps": 120.0}
    e = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((n_roll, 2)),
        rollout_root_z=np.zeros(n_roll), rollout_joint_names=["a", "b"],
        control_hz=50.0)
    assert e.duration_coverage == pytest.approx(1.0)


def test_a_rollout_that_really_stops_early_still_reports_partial_coverage():
    """The metric must not become vacuous — half the wall time is half."""
    from sculptor.refs.track import compute_tracking_errors

    clip = {"joint_pos": np.zeros((444, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(444), "fps": 120.0}
    e = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((92, 2)),
        rollout_root_z=np.zeros(92), rollout_joint_names=["a", "b"],
        control_hz=50.0)
    assert e.duration_coverage == pytest.approx(
        ((92 - 1) / 50) / ((444 - 1) / 120), rel=1e-3,
    )


def test_duration_coverage_falls_back_to_frames_without_fps():
    """A clip with no fps has no wall-clock interpretation; comparing frames is
    the only thing left, and must not divide by zero."""
    from sculptor.refs.track import compute_tracking_errors

    clip = {"joint_pos": np.zeros((100, 2)), "joint_names": ["a", "b"],
            "root_pos_z": np.zeros(100)}
    e = compute_tracking_errors(
        clip=clip, rollout_joint_pos=np.zeros((50, 2)),
        rollout_root_z=np.zeros(50), rollout_joint_names=["a", "b"],
        control_hz=0.0)
    assert e.duration_coverage == pytest.approx(0.5)
