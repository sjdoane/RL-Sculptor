from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.same_lane_acceptance import (
    CONTRACT_SCHEMA,
    PhysicalAcceptanceError,
    evaluate_iteration_same_lane_acceptance,
    load_or_evaluate_iteration_acceptance,
    reduce_same_lane_acceptance,
    validate_physical_acceptance_contract,
    write_iteration_acceptance_receipt,
    write_precommitted_contract,
)
from backend.routes.policies import _objective_proof_decision


_REF_HASHES = {
    "world": "1" * 64,
    "task": "2" * 64,
    "reward": "3" * 64,
    "env_spec": "4" * 64,
    "resolved_eval": "5" * 64,
    "channel_catalog": "6" * 64,
}
_REFERENCE = {
    "robot": "g1",
    "clip_id": "four-hop-reference",
    "clip_sha256": "a" * 64,
    "certificate_sha256": "b" * 64,
    "rollout_sha256": "c" * 64,
    "execution_contract_sha256": "d" * 64,
    "execution_boundary_sha256": "e" * 64,
}


def _contract(*, requested_lane: int = 0) -> dict:
    regions = [
        ("landing_1", 0.65, 0.30),
        ("landing_2", 1.15, 0.30),
        ("landing_3", 1.65, 0.30),
        ("landing_4", 2.15, 0.30),
        ("finish", 2.55, 0.45),
    ]
    rails = [0.35, 0.85, 1.35, 1.85]
    return {
        "schema": CONTRACT_SCHEMA,
        "created_at": 100.0,
        "precommit_id": "job_example:four-hop",
        "identity": {
            "selection_tuple_sha256": "f" * 64,
            "selection_refs": dict(_REF_HASHES),
            "reference": dict(_REFERENCE),
        },
        "lane": {
            "requested_index": requested_lane,
            "selection": "precommitted",
        },
        "validity": {"mask_channel": "first_episode_valid_mask"},
        "route": {
            "waypoint_index_channel": "goal__course__waypoint_index",
            "waypoint_count": 5,
            "ordered_regions": [
                {
                    "id": name,
                    "relative_channel": f"region__{name}__relative",
                    "radius_m": radius,
                }
                for name, _center, radius in regions
            ],
        },
        "support_cycles": {
            "root_position_channel": "root_link_pos_w",
            "left_contact_channel": "left_foot_contact",
            "right_contact_channel": "right_foot_contact",
            "minimum_flight_frames": 3,
            "maximum_touchdown_gap_frames": 2,
            "maximum_waypoint_advance_lag_frames": 2,
            "mappings": [
                {
                    "phase_id": f"hop_{index}",
                    "obstacle_id": f"rail_{index}",
                    "obstacle_position_channel": f"object__rail_{index}__pos_w",
                    "crossing_axis": 0,
                    "crossing_direction": 1,
                    "crossing_half_extent_m": 0.05,
                    "landing_region_id": f"landing_{index}",
                    "landing_completion_index": index,
                }
                for index, _rail in enumerate(rails, start=1)
            ],
        },
        "forbidden_contact_channels": [
            f"contact__forbidden__{index}" for index in range(4)
        ],
        "safety": {
            "projected_gravity_channel": "projected_gravity_b",
            "projected_gravity_z_index": 2,
            "fall_gravity_z_above": -0.2,
            "maximum_consecutive_fall_frames": 10,
        },
        "terminal_hold": {
            "frames": 100,
            "finish_region_id": "finish",
            "root_linear_velocity_channel": "root_link_lin_vel_b",
            "horizontal_velocity_indices": [0, 1],
            "horizontal_speed_below_m_s": 0.12,
            "root_angular_velocity_channel": "root_link_ang_vel_b",
            "angular_speed_below_rad_s": 0.5,
            "joint_velocity_channel": "joint_vel",
            "joint_speed_rms_below_rad_s": 1.0,
            "projected_gravity_z_at_most": -0.7,
            "default_pose_rms_channel": "default_pose_rms",
            "default_pose_rms_below_rad": 0.6,
        },
    }


def _arrays(*, lanes: int = 2) -> dict[str, np.ndarray]:
    steps = 200
    root_x = np.interp(
        np.arange(steps),
        [0, 9, 15, 29, 35, 49, 55, 69, 75, 89, 90, 199],
        [-0.10, 0.20, 0.65, 0.70, 1.15, 1.20,
         1.65, 1.70, 2.15, 2.40, 2.55, 2.55],
    ).astype(np.float32)
    root = np.zeros((steps, lanes, 3), dtype=np.float32)
    root[:, :, 0] = root_x[:, None]
    root[:, :, 2] = 0.78
    valid = np.ones((steps, lanes), dtype=bool)
    waypoint = np.zeros((steps, lanes), dtype=np.int32)
    for frame, index in ((15, 1), (35, 2), (55, 3), (75, 4), (90, 5)):
        waypoint[frame:] = index
    contact = np.ones((steps, lanes), dtype=bool)
    for start, end in ((10, 14), (30, 34), (50, 54), (70, 74)):
        contact[start:end + 1] = False

    result: dict[str, np.ndarray] = {
        "first_episode_valid_mask": valid,
        "goal__course__waypoint_index": waypoint,
        "root_link_pos_w": root,
        "left_foot_contact": contact.copy(),
        "right_foot_contact": contact.copy(),
        "root_link_lin_vel_b": np.zeros((steps, lanes, 3), dtype=np.float32),
        "root_link_ang_vel_b": np.zeros((steps, lanes, 3), dtype=np.float32),
        "joint_vel": np.zeros((steps, lanes, 29), dtype=np.float32),
        "default_pose_rms": np.zeros((steps, lanes), dtype=np.float32),
        "projected_gravity_b": np.broadcast_to(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            (steps, lanes, 3),
        ).copy(),
    }
    for index in range(4):
        result[f"contact__forbidden__{index}"] = np.zeros(
            (steps, lanes), dtype=bool
        )
    for index, x in enumerate((0.35, 0.85, 1.35, 1.85), start=1):
        position = np.zeros((steps, lanes, 3), dtype=np.float32)
        position[:, :, 0] = x
        position[:, :, 2] = 0.03
        result[f"object__rail_{index}__pos_w"] = position
    for name, center, _radius in (
        ("landing_1", 0.65, 0.30),
        ("landing_2", 1.15, 0.30),
        ("landing_3", 1.65, 0.30),
        ("landing_4", 2.15, 0.30),
        ("finish", 2.55, 0.45),
    ):
        relative = np.zeros((steps, lanes, 3), dtype=np.float32)
        relative[:, :, 0] = center - root_x[:, None]
        result[f"region__{name}__relative"] = relative
    return result


def _iteration(tmp_path: Path) -> tuple[Path, dict, dict[str, np.ndarray]]:
    iter_dir = tmp_path / "runs" / "iter_0"
    rollout = iter_dir / "rollout"
    rollout.mkdir(parents=True)
    contract = _contract()
    write_precommitted_contract(
        iter_dir / "physical_acceptance_contract.json", contract
    )
    refs = {
        key: {
            "kind": key,
            "version": "v1",
            "path": f"env/{key}_v1.json",
            "sha256": digest,
        }
        for key, digest in _REF_HASHES.items()
    }
    artifact_tuple = {
        "created_at": 200.0,
        "evaluation_lineage": "test",
        "refs": refs,
        "selection_version": 1,
        "tuple_hash": "f" * 64,
    }
    (iter_dir / "artifact_tuple.json").write_text(
        json.dumps(artifact_tuple, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    behavior = {
        "rendered_env_index_requested": 0,
        "rendered_env_index": 0,
        "rendered_env_selection": "precommitted",
        "rendered_episode_percentile": 0.5,
    }
    (rollout / "behavior.json").write_text(
        json.dumps(behavior, sort_keys=True), encoding="utf-8"
    )
    arrays = _arrays()
    world_refs = {
        key: {
            "kind": key,
            "version": "v1",
            "path": f"env/{key}_v1.json",
            "sha256": digest,
        }
        for key, digest in _REF_HASHES.items()
    }
    trajectory_contract = {
        "schema": "reward-sculptor-trajectory-v1",
        "layout": ["time", "environment", "feature"],
        "runtime_artifacts": {
            "schema": "reward-sculptor-runner-artifacts-v2",
            "phase": "rollout",
            "reward_module_sha256": _REF_HASHES["reward"],
            "environment_artifacts": {
                "schema": "reward-sculptor-environment-artifacts-v1",
                "env_spec": {"present": True, "sha256": _REF_HASHES["env_spec"]},
                "eval_reset": {"present": False, "sha256": None},
                "world_selection": {
                    "present": True,
                    "sha256": "0" * 64,
                    "tuple_hash": "f" * 64,
                    "refs": world_refs,
                },
            },
        },
    }
    arrays["trajectory_contract_json"] = np.asarray(json.dumps(
        trajectory_contract, sort_keys=True, separators=(",", ":")
    ))
    np.savez_compressed(rollout / "trajectory.npz", **arrays)
    return iter_dir, contract, arrays


def _rewrite_trajectory(iter_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(iter_dir / "rollout" / "trajectory.npz", **arrays)


def test_positive_reducer_emits_conjunctive_same_lane_mask() -> None:
    result = reduce_same_lane_acceptance(_arrays(), _contract())
    assert result["full_pass_mask"] == [True, True]
    assert result["full_pass_count"] == 2
    assert result["requested_lane_full_pass"] is True
    assert all(len(row["support_cycles"]) == 4 for row in result["lanes"])
    assert all(
        all(cycle["rail_crossed"] for cycle in row["support_cycles"])
        for row in result["lanes"]
    )


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda a: a["goal__course__waypoint_index"].__setitem__(
                (slice(35, None), 0), 1
            ),
            "raw route never reaches index 2",
        ),
        (
            lambda a: a["region__landing_2__relative"].__setitem__(
                (35, 0, 0), 0.31
            ),
            "raw index 2 is not an entry",
        ),
        (
            lambda a: a["left_foot_contact"].__setitem__(
                (slice(50, 55), 0), True
            ),
            "observed 3 bilateral flight bouts",
        ),
        (
          lambda a: a["right_foot_contact"].__setitem__(
              (slice(75, 77), 0), False
          ),
            "hop_4 has no bilateral touchdown",
        ),
        (
            lambda a: a["object__rail_2__pos_w"].__setitem__(
                (slice(None), 0, 0), 4.0
            ),
            "hop_2 does not cross its obstacle",
        ),
        (
            lambda a: a["goal__course__waypoint_index"].__setitem__(
                (slice(15, 18), 0), 0
            ),
            "hop_1 is not one-to-one",
        ),
        (
            lambda a: a["contact__forbidden__2"].__setitem__((52, 0), True),
            "forbidden contact channel",
        ),
        (
            lambda a: a["projected_gravity_b"].__setitem__(
                (slice(20, 31), 0, 2), 0.0
            ),
            "sustained fall",
        ),
        (
            lambda a: a["region__finish__relative"].__setitem__(
                (120, 0, 0), 0.46
            ),
            "terminal hold fails inside_finish",
        ),
        (
            lambda a: a["root_link_lin_vel_b"].__setitem__((120, 0, 0), 0.12),
            "terminal hold fails horizontal_speed",
        ),
        (
            lambda a: a["root_link_ang_vel_b"].__setitem__((120, 0, 2), 0.5),
            "terminal hold fails angular_speed",
        ),
        (
            lambda a: a["joint_vel"].__setitem__((120, 0, slice(None)), 1.0),
            "terminal hold fails joint_speed_rms",
        ),
        (
            lambda a: a["projected_gravity_b"].__setitem__((120, 0, 2), -0.69),
            "terminal hold fails upright",
        ),
        (
            lambda a: a["default_pose_rms"].__setitem__((120, 0), 0.6),
            "terminal hold fails default_pose",
        ),
        (
          lambda a: a["first_episode_valid_mask"].__setitem__(
              (slice(150, None), 0), False
          ),
            "fewer than the required hold frames",
        ),
    ],
)
def test_each_physical_failure_blocks_only_that_lane(mutate, blocker: str) -> None:
    arrays = _arrays()
    mutate(arrays)
    result = reduce_same_lane_acceptance(arrays, _contract())
    assert result["full_pass_mask"] == [False, True]
    assert result["full_pass_count"] == 1
    assert any(blocker in item for item in result["lanes"][0]["blockers"])


def test_invalid_valid_mask_prefix_fails_closed_per_lane() -> None:
    arrays = _arrays()
    arrays["first_episode_valid_mask"][100, 0] = False
    result = reduce_same_lane_acceptance(arrays, _contract())
    assert result["full_pass_mask"] == [False, True]
    assert result["lanes"][0]["blockers"] == [
        "first_episode_valid_mask is not a true prefix"
    ]


def test_aggregate_success_cannot_mix_route_contact_and_hold_lanes() -> None:
    arrays = _arrays()
    # Lane 0 owns route/contact/cycles but fails hold. Lane 1 owns route/hold
    # but contacts a rail. Independent aggregate components would all look
    # successful; the conjunctive reducer correctly proves zero full lanes.
    arrays["root_link_lin_vel_b"][120, 0, 0] = 0.2
    arrays["contact__forbidden__0"][20, 1] = True
    result = reduce_same_lane_acceptance(arrays, _contract())
    assert result["full_pass_mask"] == [False, False]
    assert result["full_pass_count"] == 0


def test_missing_or_malformed_channel_is_global_unavailable(tmp_path: Path) -> None:
    iter_dir, _contract_value, arrays = _iteration(tmp_path)
    del arrays["right_foot_contact"]
    _rewrite_trajectory(iter_dir, arrays)
    result = evaluate_iteration_same_lane_acceptance(iter_dir)
    assert result["status"] == "unavailable"
    assert result["full_pass_mask"] == []
    assert "right_foot_contact" in result["reason"]


def test_iteration_receipt_binds_every_identity_and_exact_files(tmp_path: Path) -> None:
    iter_dir, contract, _arrays_value = _iteration(tmp_path)
    result = evaluate_iteration_same_lane_acceptance(iter_dir)
    assert result["status"] == "passed"
    assert result["full_pass_count"] == 2
    assert result["requested_lane_full_pass"] is True
    assert result["identity"] == {
        "selection_tuple_sha256": "f" * 64,
        "selection_refs": _REF_HASHES,
        "reference": _REFERENCE,
        "requested_lane": 0,
        "resolved_lane": 0,
    }
    expected_contract_sha = hashlib.sha256(
        json.dumps(
            validate_physical_acceptance_contract(contract),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert result["contract_sha256"] == expected_contract_sha
    assert len(result["trajectory_sha256"]) == 64
    assert len(result["receipt_sha256"]) == 64


def test_requested_video_lane_cannot_be_replaced_by_another_passing_lane(
    tmp_path: Path,
) -> None:
    iter_dir, _contract_value, arrays = _iteration(tmp_path)
    arrays["root_link_lin_vel_b"][120, 0, 0] = 0.2
    _rewrite_trajectory(iter_dir, arrays)
    result = evaluate_iteration_same_lane_acceptance(iter_dir)
    assert result["status"] == "failed"
    assert result["full_pass_mask"] == [False, True]
    assert result["full_pass_count"] == 1
    assert result["requested_lane_full_pass"] is False


@pytest.mark.parametrize(
    ("physical", "expected_status", "expected_blocker"),
    [
        (
            {
                "status": "passed",
                "requested_lane_full_pass": True,
                "reason": None,
            },
            "passed",
            None,
        ),
        (
            {
                "status": "failed",
                "requested_lane_full_pass": False,
                "reason": None,
            },
            "failed",
            "precommitted evidence lane failed",
        ),
        (
            {
                "status": "unavailable",
                "requested_lane_full_pass": False,
                "reason": "trajectory channel is missing",
            },
            "incomplete",
            "same-lane physical acceptance evidence is unavailable",
        ),
    ],
)
def test_policy_proof_uses_same_lane_receipt_instead_of_aggregate_components(
    physical: dict,
    expected_status: str,
    expected_blocker: str | None,
) -> None:
    # Deliberately contradictory independent aggregates prove that a physical
    # contract never falls back to, or mixes, their per-batch fractions.
    status, blockers = _objective_proof_decision(
        route={"passed": False},
        contact={"passed": True},
        hold={"passed": True},
        criterion_status="passed",
        lane_receipt={"lane_evidence_status": "verified"},
        rollout_available=True,
        metric_identity_complete=True,
        same_lane_acceptance=physical,
    )
    assert status == expected_status
    if expected_blocker is None:
        assert blockers == []
    else:
        assert any(expected_blocker in blocker for blocker in blockers)


@pytest.mark.parametrize(
    "mutation",
    [
        "tuple",
        "world_ref",
        "runtime_reward",
        "runtime_world",
        "requested_lane",
        "resolved_lane",
        "posthoc_contract",
    ],
)
def test_iteration_identity_mismatch_is_unavailable(
    tmp_path: Path, mutation: str
) -> None:
    iter_dir, _contract_value, arrays = _iteration(tmp_path)
    tuple_path = iter_dir / "artifact_tuple.json"
    behavior_path = iter_dir / "rollout" / "behavior.json"
    contract_path = iter_dir / "physical_acceptance_contract.json"
    if mutation in {"tuple", "world_ref"}:
        doc = json.loads(tuple_path.read_text(encoding="utf-8"))
        if mutation == "tuple":
            doc["tuple_hash"] = "0" * 64
        else:
            doc["refs"]["world"]["sha256"] = "0" * 64
        tuple_path.write_text(json.dumps(doc), encoding="utf-8")
    elif mutation in {"runtime_reward", "runtime_world"}:
        metadata = json.loads(str(arrays["trajectory_contract_json"].item()))
        runtime = metadata["runtime_artifacts"]
        if mutation == "runtime_reward":
            runtime["reward_module_sha256"] = "0" * 64
        else:
            runtime["environment_artifacts"]["world_selection"][
                "tuple_hash"
            ] = "0" * 64
        arrays["trajectory_contract_json"] = np.asarray(json.dumps(metadata))
        _rewrite_trajectory(iter_dir, arrays)
    elif mutation in {"requested_lane", "resolved_lane"}:
        behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
        behavior[
            "rendered_env_index_requested"
            if mutation == "requested_lane"
            else "rendered_env_index"
        ] = 1
        behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    else:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["created_at"] = 300.0
        contract_path.unlink()
        write_precommitted_contract(contract_path, contract)
    result = evaluate_iteration_same_lane_acceptance(iter_dir)
    assert result["status"] == "unavailable"
    assert result["full_pass_count"] == 0
    assert result["reason"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(extra=True),
        lambda c: c["identity"]["reference"].pop("rollout_sha256"),
        lambda c: c["route"]["ordered_regions"].pop(),
        lambda c: c["support_cycles"]["mappings"].reverse(),
        lambda c: c["support_cycles"]["mappings"][0].update(
            landing_completion_index=2
        ),
        lambda c: c["forbidden_contact_channels"].append(
            c["forbidden_contact_channels"][0]
        ),
        lambda c: c["terminal_hold"].update(frames=0),
    ],
)
def test_contract_validation_fails_closed_on_ambiguous_data(mutation) -> None:
    contract = copy.deepcopy(_contract())
    mutation(contract)
    with pytest.raises(PhysicalAcceptanceError):
        validate_physical_acceptance_contract(contract)


def test_requested_lane_outside_rollout_fails_closed() -> None:
    with pytest.raises(PhysicalAcceptanceError, match="outside the trajectory"):
        reduce_same_lane_acceptance(_arrays(), _contract(requested_lane=2))


def test_corrupt_trajectory_archive_is_unavailable_not_a_crash(
    tmp_path: Path,
) -> None:
    """2026-08-25 audit repro: a truncated trajectory.npz — a legitimate crash
    artifact — raised zipfile.BadZipFile through the evaluator and turned the
    project-wide policy listing into an unhandled 500. It must reduce to an
    'unavailable' receipt instead."""
    iter_dir, _contract_value, _arrays_value = _iteration(tmp_path)
    trajectory = iter_dir / "rollout" / "trajectory.npz"
    intact = trajectory.read_bytes()
    trajectory.write_bytes(intact[: len(intact) // 2])
    result = load_or_evaluate_iteration_acceptance(iter_dir)
    assert result["status"] == "unavailable"
    assert result["reason"]
    assert result["receipt_sha256"]
    status, blockers = _objective_proof_decision(
        route={"passed": True},
        contact={"passed": True},
        hold={"passed": True},
        criterion_status="passed",
        lane_receipt={"lane_evidence_status": "verified"},
        rollout_available=True,
        metric_identity_complete=True,
        same_lane_acceptance=result,
    )
    assert status == "incomplete"
    assert any("unavailable" in blocker for blocker in blockers)


def test_persisted_failed_verdict_survives_contract_deletion(
    tmp_path: Path,
) -> None:
    """Deleting the precommitted contract after a failed verdict must not
    silently revert the pass authority to aggregate route/contact/hold
    components (2026-08-25 audit: deletable pass authority)."""
    iter_dir, _contract_value, arrays = _iteration(tmp_path)
    arrays["root_link_lin_vel_b"][120, 0, 0] = 0.2
    _rewrite_trajectory(iter_dir, arrays)
    first = load_or_evaluate_iteration_acceptance(iter_dir)
    assert first["status"] == "failed"
    assert (iter_dir / "physical_acceptance_receipt.json").is_file()
    (iter_dir / "physical_acceptance_contract.json").unlink()
    second = load_or_evaluate_iteration_acceptance(iter_dir)
    assert second["status"] == "failed"
    assert second["receipt_sha256"] == first["receipt_sha256"]
    status, _blockers = _objective_proof_decision(
        route={"passed": True},
        contact={"passed": True},
        hold={"passed": True},
        criterion_status="passed",
        lane_receipt={"lane_evidence_status": "verified"},
        rollout_available=True,
        metric_identity_complete=True,
        same_lane_acceptance=second,
    )
    assert status == "failed"


def test_tampered_receipt_with_missing_contract_stays_unavailable(
    tmp_path: Path,
) -> None:
    """A receipt that fails self-verification while its contract is also gone
    must stay 'unavailable' — never a pass, never aggregate components."""
    iter_dir, _contract_value, _arrays_value = _iteration(tmp_path)
    receipt = write_iteration_acceptance_receipt(iter_dir)
    assert receipt["status"] == "passed"
    receipt_path = iter_dir / "physical_acceptance_receipt.json"
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    doc["full_pass_count"] = int(doc["full_pass_count"]) + 1
    receipt_path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    (iter_dir / "physical_acceptance_contract.json").unlink()
    result = load_or_evaluate_iteration_acceptance(iter_dir)
    assert result["status"] == "unavailable"
    assert "failed verification" in result["reason"]


def test_verified_terminal_receipt_is_served_without_reevaluation(
    tmp_path: Path,
) -> None:
    """Once a terminal verdict is persisted and self-verifies, listings serve
    it byte-for-byte instead of re-reducing the trajectory; the receipt's
    recorded artifact hashes remain the tamper record."""
    iter_dir, _contract_value, _arrays_value = _iteration(tmp_path)
    first = load_or_evaluate_iteration_acceptance(iter_dir)
    assert first["status"] == "passed"
    (iter_dir / "rollout" / "trajectory.npz").write_bytes(b"not a zip")
    second = load_or_evaluate_iteration_acceptance(iter_dir)
    assert second == first
