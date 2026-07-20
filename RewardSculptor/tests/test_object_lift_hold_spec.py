"""Adversarial contract tests for the generic object lift/hold metric.

These fixtures are executable counterexamples, not reporting evidence. They
exercise all ten A4 attack classes deterministically; real simulator rollouts
are still required before the metric may receive reporting authority.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from sculptor.eval.metric_calibration import calibrate_metric
from sculptor.eval.spec_metrics import compute_spec_metrics


DT = 0.02
T = 70


def _competent_arrays(*, multi: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    object_names = ["parcel", "spare"] if multi else ["parcel"]
    envs = 2 if multi else 1
    arrays: dict[str, np.ndarray] = {
        "target_object_index": np.zeros((T, envs), dtype=np.int32),
        "target__pos_w": np.zeros((T, envs, 3), dtype=np.float32),
        "rollout_valid": np.ones((T, envs), dtype=np.bool_),
        "rollout_terminal": np.zeros((T, envs), dtype=np.bool_),
    }
    if multi:
        arrays["target_object_index"][:, 1] = 1
    arrays["target__pos_w"][..., 0] = 0.30
    arrays["target__pos_w"][..., 2] = 0.25

    for object_index, name in enumerate(object_names):
        pos = np.zeros((T, envs, 3), dtype=np.float32)
        pos[..., 0] = 0.30
        pos[..., 2] = 0.03
        grasp = np.zeros((T, envs), dtype=np.float32)
        for env_index in range(envs):
            if object_index != int(arrays["target_object_index"][0, env_index]):
                continue
            pos[10:25, env_index, 2] = np.linspace(
                0.03, 0.25, 15, dtype=np.float32)
            pos[25:, env_index, 2] = 0.25
            grasp[12:, env_index] = 1.0
        lin = np.gradient(pos, DT, axis=0).astype(np.float32)
        arrays[f"object__{name}__pos_w"] = pos
        arrays[f"object__{name}__lin_vel_w"] = lin
        arrays[f"object__{name}__ang_vel_w"] = np.zeros_like(pos)
        arrays[f"contact__left__{name}"] = grasp.copy()
        arrays[f"contact__right__{name}"] = grasp.copy()
        arrays[f"grasp__{name}"] = grasp

    manifest = {
        "schema_version": 2,
        "robot_entity": "any_arm",
        "object_names": object_names,
        "capability_id": "example:parallel_gripper",
        "finger_groups": {"left": ["left_pad"], "right": ["right_pad"]},
        "grasp_capable": True,
        "contact_sensors": [],
        "target_contract": {
            "position_frame": "world",
            "declared_success_threshold_m": 0.05,
        },
        "derivations": {},
        "channels": {},
    }
    return arrays, manifest


def _write_rollout(
    path: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict,
) -> None:
    path.mkdir(parents=True)
    manifest = copy.deepcopy(manifest)
    manifest["channels"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }
    np.savez_compressed(path / "trajectory.npz", **arrays)
    (path / "behavior.json").write_text(json.dumps({
        "step_dt": DT,
        "max_episode_steps": T,
        "rollout_num_envs": arrays["target_object_index"].shape[1],
    }), encoding="utf-8")
    (path / "manipulation_telemetry.json").write_text(
        json.dumps(manifest), encoding="utf-8")


def _score(
    tmp_path: Path,
    transform: Callable[[dict[str, np.ndarray], dict], None] | None = None,
    *,
    multi: bool = False,
) -> dict:
    arrays, manifest = _competent_arrays(multi=multi)
    if transform is not None:
        transform(arrays, manifest)
    _write_rollout(tmp_path / "rollout", arrays, manifest)
    return compute_spec_metrics("object_lift_hold", tmp_path / "rollout")


def test_competent_target_aware_multi_object_lift_scores_one(tmp_path: Path) -> None:
    result = _score(tmp_path, multi=True)
    assert result.get("error") is None
    assert result["completion_rate"] == pytest.approx(1.0)
    assert result["spec_score"] == pytest.approx(1.0)
    assert result["max_distractor_displacement_m"] == pytest.approx(0.0)


def test_single_spawn_settling_impulse_does_not_veto_competence(
    tmp_path: Path,
) -> None:
    def settling(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
        arrays["object__parcel__lin_vel_w"][0, 0, 2] = -0.8

    result = _score(tmp_path, settling)
    assert result["physically_valid_rate"] == pytest.approx(1.0)
    assert result["spec_score"] == pytest.approx(1.0)


def _stillness(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    pos = arrays["object__parcel__pos_w"]
    pos[..., 2] = 0.03
    arrays["object__parcel__lin_vel_w"][:] = 0
    arrays["grasp__parcel"][:] = 0
    arrays["contact__left__parcel"][:] = 0
    arrays["contact__right__parcel"][:] = 0


def _falling(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    pos = arrays["object__parcel__pos_w"]
    pos[25:, 0, 2] = np.linspace(0.25, -0.05, T - 25)
    arrays["object__parcel__lin_vel_w"] = np.gradient(pos, DT, axis=0)


def _oscillation(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    # Position remains inside the goal tolerance but persisted motion is too
    # energetic to qualify as a hold.
    arrays["object__parcel__lin_vel_w"][25:, 0, 0] = 0.5


def _explosion(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    arrays["object__parcel__pos_w"][30, 0, 2] = np.inf


def _early_termination(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    arrays["rollout_valid"][20:, 0] = False
    arrays["rollout_terminal"][20, 0] = True


def _threshold_flicker(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    pos = arrays["object__parcel__pos_w"]
    pos[25::5, 0, 2] = 0.19  # 6 cm error: just outside the capped 5 cm gate.
    arrays["object__parcel__lin_vel_w"] = np.gradient(pos, DT, axis=0)


def _reset_artifact(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    arrays["object__parcel__pos_w"][:] = arrays["target__pos_w"]
    arrays["object__parcel__lin_vel_w"][:] = 0
    arrays["grasp__parcel"][:] = 1
    arrays["contact__left__parcel"][:] = 1
    arrays["contact__right__parcel"][:] = 1


def _time_truncation(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
    arrays["rollout_valid"][38:, 0] = False


def _proxy_only(arrays: dict[str, np.ndarray], manifest: dict) -> None:
    _stillness(arrays, manifest)
    arrays["grasp__parcel"][:] = 1
    arrays["contact__left__parcel"][:] = 1
    arrays["contact__right__parcel"][:] = 1


@pytest.mark.parametrize(
    ("attack_class", "transform"),
    [
        ("stillness", _stillness),
        ("falling", _falling),
        ("oscillation", _oscillation),
        ("explosion", _explosion),
        ("early_termination", _early_termination),
        ("threshold_flicker", _threshold_flicker),
        ("reset_artifact", _reset_artifact),
        ("time_truncation", _time_truncation),
        ("proxy_only", _proxy_only),
    ],
)
def test_a4_attack_classes_score_zero(
    tmp_path: Path,
    attack_class: str,
    transform: Callable[[dict[str, np.ndarray], dict], None],
) -> None:
    result = _score(tmp_path / attack_class, transform)
    assert result["spec_score"] == pytest.approx(0.0), result


def test_target_identity_flicker_cannot_assemble_a_hold(tmp_path: Path) -> None:
    def attack(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
        arrays["target_object_index"][20::2, 0] = 1

    # Index 1 is invalid in the single-object manifest and cuts the first
    # target segment before a hold can complete.
    result = _score(tmp_path, attack)
    assert result["spec_score"] == pytest.approx(0.0)


def test_moving_a_non_target_object_vetoes_multi_object_success(
    tmp_path: Path,
) -> None:
    def attack(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
        spare = arrays["object__spare__pos_w"]
        spare[20:, 0, 1] = 0.08
        arrays["object__spare__lin_vel_w"] = np.gradient(spare, DT, axis=0)

    result = _score(tmp_path, attack, multi=True)
    assert result["spec_score"] == pytest.approx(0.5)
    assert result["max_distractor_displacement_m"] > 0.05


def test_telekinesis_lift_without_grasp_evidence_scores_zero(
    tmp_path: Path,
) -> None:
    """A perfect transport with zero contact evidence must never complete.

    Direct counterpart of proxy_only: outcome without mechanism. The zeroed
    channels are self-consistent, so the forgery cross-check cannot fire and
    only the grasp-evidence gates stand between this artifact and a score.
    The grasp-fraction and grasp-gap gates are individually redundant on
    dense evidence; this test is the one that fails if BOTH are removed.
    """
    def attack(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
        for key in list(arrays):
            if key.startswith("grasp__") or key.startswith("contact__"):
                arrays[key] = np.zeros_like(arrays[key])

    result = _score(tmp_path, attack)
    assert result.get("error") is None
    assert result["spec_score"] == pytest.approx(0.0)
    assert result["mean_best_grasp_fraction"] == pytest.approx(0.0)


def test_forged_grasp_channel_fails_closed(tmp_path: Path) -> None:
    def attack(arrays: dict[str, np.ndarray], _manifest: dict) -> None:
        arrays["contact__right__parcel"][:] = 0

    result = _score(tmp_path, attack)
    assert result["spec_score"] == pytest.approx(0.0)
    assert "disagrees" in result["error"]


def test_legacy_sidecar_without_episode_masks_fails_closed(tmp_path: Path) -> None:
    def attack(arrays: dict[str, np.ndarray], manifest: dict) -> None:
        manifest["schema_version"] = 1
        arrays.pop("rollout_valid")
        arrays.pop("rollout_terminal")

    result = _score(tmp_path, attack)
    assert result["spec_score"] == pytest.approx(0.0)
    assert "schema 2" in result["error"]


def test_generic_synthetic_calibration_cannot_grant_manipulation_rights(
    tmp_path: Path,
) -> None:
    result = calibrate_metric(tmp_path / "unused.py", "object_lift_hold")
    assert result["ok"] is False
    assert "task-derived frozen rollouts" in result["error"]
