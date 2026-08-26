"""Immutable, same-lane physical acceptance for authored rollouts.

Generated fitness components are useful search signals, but their aggregate
fractions are not a physical proof: route, contact, and terminal-hold values
can each be satisfied by a different evaluation lane.  This module reduces
the official trajectory arrays conjunctively *within each lane* under a
precommitted, data-only contract.

The reducer is intentionally task/robot agnostic.  A contract names every
channel, ordered region, obstacle/landing association, and threshold.  The
four-rail G1 experiment is therefore data supplied to this module, not a
special case hidden in Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTRACT_SCHEMA = "reward-sculptor-physical-acceptance-v1"
RECEIPT_SCHEMA = "reward-sculptor-physical-acceptance-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_SELECTION_REFS = (
    "world",
    "task",
    "reward",
    "env_spec",
    "resolved_eval",
    "channel_catalog",
)
_REQUIRED_REFERENCE_HASHES = (
    "clip_sha256",
    "certificate_sha256",
    "rollout_sha256",
    "execution_contract_sha256",
    "execution_boundary_sha256",
)


class PhysicalAcceptanceError(ValueError):
    """Raised when the precommit or official evidence is not admissible."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _finite_number(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalAcceptanceError("threshold must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PhysicalAcceptanceError("threshold is outside its admitted range")
    return result


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PhysicalAcceptanceError(f"{label} must be a positive integer")
    return int(value)


def _channel_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalAcceptanceError(f"{label} must name a trajectory channel")
    return value.strip()


def validate_physical_acceptance_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a canonical data-only acceptance contract.

    Unknown top-level keys fail closed.  This prevents a newer producer from
    silently relying on semantics an older reducer does not implement.
    """

    if not isinstance(value, Mapping):
        raise PhysicalAcceptanceError("physical acceptance contract is missing")
    expected_top = {
        "schema",
        "created_at",
        "precommit_id",
        "identity",
        "lane",
        "validity",
        "route",
        "support_cycles",
        "forbidden_contact_channels",
        "safety",
        "terminal_hold",
    }
    if set(value) != expected_top:
        raise PhysicalAcceptanceError(
            "physical acceptance contract has non-canonical keys"
        )
    if value.get("schema") != CONTRACT_SCHEMA:
        raise PhysicalAcceptanceError("physical acceptance schema is unsupported")
    created_at = _finite_number(value.get("created_at"), minimum=0.0)
    precommit_id = value.get("precommit_id")
    if not isinstance(precommit_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", precommit_id
    ):
        raise PhysicalAcceptanceError("precommit_id is invalid")

    identity = value.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "selection_tuple_sha256", "selection_refs", "reference"
    }:
        raise PhysicalAcceptanceError("identity receipt is non-canonical")
    tuple_sha = identity.get("selection_tuple_sha256")
    if not _is_sha256(tuple_sha):
        raise PhysicalAcceptanceError("selection tuple SHA-256 is invalid")
    refs = identity.get("selection_refs")
    if not isinstance(refs, Mapping) or set(refs) != set(
        _REQUIRED_SELECTION_REFS
    ):
        raise PhysicalAcceptanceError("selection ref identity is incomplete")
    canonical_refs: dict[str, str] = {}
    for key in _REQUIRED_SELECTION_REFS:
        digest = refs.get(key)
        if not _is_sha256(digest):
            raise PhysicalAcceptanceError(f"selection ref {key!r} is invalid")
        canonical_refs[key] = str(digest)
    reference = identity.get("reference")
    expected_reference_keys = {
        "robot", "clip_id", *_REQUIRED_REFERENCE_HASHES,
    }
    if not isinstance(reference, Mapping) or set(reference) != (
        expected_reference_keys
    ):
        raise PhysicalAcceptanceError("reference identity is incomplete")
    for key in ("robot", "clip_id"):
        if not isinstance(reference.get(key), str) or not reference.get(key):
            raise PhysicalAcceptanceError(f"reference {key} is invalid")
    canonical_reference = {
        "robot": str(reference["robot"]),
        "clip_id": str(reference["clip_id"]),
    }
    for key in _REQUIRED_REFERENCE_HASHES:
        digest = reference.get(key)
        if not _is_sha256(digest):
            raise PhysicalAcceptanceError(f"reference {key} is invalid")
        canonical_reference[key] = str(digest)

    lane = value.get("lane")
    if not isinstance(lane, Mapping) or set(lane) != {
        "requested_index", "selection"
    }:
        raise PhysicalAcceptanceError("lane precommit is non-canonical")
    requested_index = lane.get("requested_index")
    if (
        isinstance(requested_index, bool)
        or not isinstance(requested_index, int)
        or requested_index < 0
    ):
        raise PhysicalAcceptanceError("requested lane index is invalid")
    if lane.get("selection") != "precommitted":
        raise PhysicalAcceptanceError("lane selection must be precommitted")

    validity = value.get("validity")
    if not isinstance(validity, Mapping) or set(validity) != {"mask_channel"}:
        raise PhysicalAcceptanceError("validity contract is non-canonical")
    mask_channel = _channel_name(
        validity.get("mask_channel"), label="validity.mask_channel"
    )

    route = value.get("route")
    if not isinstance(route, Mapping) or set(route) != {
        "waypoint_index_channel", "waypoint_count", "ordered_regions"
    }:
        raise PhysicalAcceptanceError("route contract is non-canonical")
    waypoint_channel = _channel_name(
        route.get("waypoint_index_channel"),
        label="route.waypoint_index_channel",
    )
    waypoint_count = _positive_int(
        route.get("waypoint_count"), label="route.waypoint_count"
    )
    raw_regions = route.get("ordered_regions")
    if not isinstance(raw_regions, list) or len(raw_regions) != waypoint_count:
        raise PhysicalAcceptanceError(
            "ordered_regions must contain exactly waypoint_count entries"
        )
    regions: list[dict[str, Any]] = []
    region_ids: set[str] = set()
    region_channels: set[str] = set()
    for index, region in enumerate(raw_regions, start=1):
        if not isinstance(region, Mapping) or set(region) != {
            "id", "relative_channel", "radius_m"
        }:
            raise PhysicalAcceptanceError("ordered region is non-canonical")
        region_id = region.get("id")
        if (
            not isinstance(region_id, str)
            or not region_id
            or region_id in region_ids
        ):
            raise PhysicalAcceptanceError("ordered region id is invalid/duplicate")
        channel = _channel_name(
            region.get("relative_channel"), label="ordered region channel"
        )
        if channel in region_channels:
            raise PhysicalAcceptanceError("ordered region channel is duplicated")
        regions.append({
            "id": region_id,
            "relative_channel": channel,
            "radius_m": _finite_number(region.get("radius_m"), minimum=1e-9),
            "completion_index": index,
        })
        region_ids.add(region_id)
        region_channels.add(channel)

    support = value.get("support_cycles")
    if not isinstance(support, Mapping) or set(support) != {
        "root_position_channel",
        "left_contact_channel",
        "right_contact_channel",
        "minimum_flight_frames",
        "maximum_touchdown_gap_frames",
        "maximum_waypoint_advance_lag_frames",
        "mappings",
    }:
        raise PhysicalAcceptanceError("support-cycle contract is non-canonical")
    root_position = _channel_name(
        support.get("root_position_channel"), label="root position"
    )
    left_contact = _channel_name(
        support.get("left_contact_channel"), label="left contact"
    )
    right_contact = _channel_name(
        support.get("right_contact_channel"), label="right contact"
    )
    if left_contact == right_contact:
        raise PhysicalAcceptanceError("left/right contact channels must differ")
    min_flight = _positive_int(
        support.get("minimum_flight_frames"),
        label="minimum_flight_frames",
    )
    touchdown_gap = _positive_int(
        support.get("maximum_touchdown_gap_frames"),
        label="maximum_touchdown_gap_frames",
    )
    advance_lag = _positive_int(
        support.get("maximum_waypoint_advance_lag_frames"),
        label="maximum_waypoint_advance_lag_frames",
    )
    raw_mappings = support.get("mappings")
    if not isinstance(raw_mappings, list) or len(raw_mappings) != (
        waypoint_count - 1
    ):
        raise PhysicalAcceptanceError(
            "support mappings must cover every non-terminal landing"
        )
    mappings: list[dict[str, Any]] = []
    seen_obstacles: set[str] = set()
    for index, mapping in enumerate(raw_mappings, start=1):
        if not isinstance(mapping, Mapping) or set(mapping) != {
            "phase_id",
            "obstacle_id",
            "obstacle_position_channel",
            "crossing_axis",
            "crossing_direction",
            "crossing_half_extent_m",
            "landing_region_id",
            "landing_completion_index",
        }:
            raise PhysicalAcceptanceError("support mapping is non-canonical")
        phase_id = mapping.get("phase_id")
        obstacle_id = mapping.get("obstacle_id")
        landing_id = mapping.get("landing_region_id")
        if not all(
            isinstance(item, str) and item
            for item in (phase_id, obstacle_id, landing_id)
        ):
            raise PhysicalAcceptanceError("mapping identity is invalid")
        if obstacle_id in seen_obstacles:
            raise PhysicalAcceptanceError("obstacle mapping is duplicated")
        if landing_id != regions[index - 1]["id"]:
            raise PhysicalAcceptanceError(
                "support mappings must follow ordered landing regions"
            )
        completion_index = mapping.get("landing_completion_index")
        if completion_index != index:
            raise PhysicalAcceptanceError(
                "landing_completion_index must match its ordered mapping"
            )
        axis = mapping.get("crossing_axis")
        direction = mapping.get("crossing_direction")
        if axis not in (0, 1) or direction not in (-1, 1):
            raise PhysicalAcceptanceError("crossing axis/direction is invalid")
        mappings.append({
            "phase_id": phase_id,
            "obstacle_id": obstacle_id,
            "obstacle_position_channel": _channel_name(
                mapping.get("obstacle_position_channel"),
                label="obstacle position channel",
            ),
            "crossing_axis": int(axis),
            "crossing_direction": int(direction),
            "crossing_half_extent_m": _finite_number(
                mapping.get("crossing_half_extent_m"), minimum=0.0
            ),
            "landing_region_id": landing_id,
            "landing_completion_index": index,
        })
        seen_obstacles.add(str(obstacle_id))

    forbidden = value.get("forbidden_contact_channels")
    if (
        not isinstance(forbidden, list)
        or not forbidden
        or len(set(forbidden)) != len(forbidden)
    ):
        raise PhysicalAcceptanceError(
            "forbidden_contact_channels must be a non-empty unique list"
        )
    forbidden_channels = [
        _channel_name(channel, label="forbidden contact channel")
        for channel in forbidden
    ]

    safety = value.get("safety")
    if not isinstance(safety, Mapping) or set(safety) != {
        "projected_gravity_channel",
        "projected_gravity_z_index",
        "fall_gravity_z_above",
        "maximum_consecutive_fall_frames",
    }:
        raise PhysicalAcceptanceError("safety contract is non-canonical")
    gravity_channel = _channel_name(
        safety.get("projected_gravity_channel"),
        label="safety projected gravity channel",
    )
    gravity_index = safety.get("projected_gravity_z_index")
    if gravity_index not in (0, 1, 2):
        raise PhysicalAcceptanceError("projected gravity z index is invalid")
    fall_frames = _positive_int(
        safety.get("maximum_consecutive_fall_frames"),
        label="maximum_consecutive_fall_frames",
    )
    fall_threshold = _finite_number(safety.get("fall_gravity_z_above"))

    terminal = value.get("terminal_hold")
    expected_terminal_keys = {
        "frames",
        "finish_region_id",
        "root_linear_velocity_channel",
        "horizontal_velocity_indices",
        "horizontal_speed_below_m_s",
        "root_angular_velocity_channel",
        "angular_speed_below_rad_s",
        "joint_velocity_channel",
        "joint_speed_rms_below_rad_s",
        "projected_gravity_z_at_most",
        "default_pose_rms_channel",
        "default_pose_rms_below_rad",
    }
    if not isinstance(terminal, Mapping) or set(terminal) != expected_terminal_keys:
        raise PhysicalAcceptanceError("terminal-hold contract is non-canonical")
    finish_id = terminal.get("finish_region_id")
    if finish_id != regions[-1]["id"]:
        raise PhysicalAcceptanceError("terminal finish must be the last region")
    velocity_indices = terminal.get("horizontal_velocity_indices")
    if (
        not isinstance(velocity_indices, list)
        or len(velocity_indices) != 2
        or len(set(velocity_indices)) != 2
        or any(index not in (0, 1, 2) for index in velocity_indices)
    ):
        raise PhysicalAcceptanceError("horizontal velocity indices are invalid")

    return {
        "schema": CONTRACT_SCHEMA,
        "created_at": created_at,
        "precommit_id": precommit_id,
        "identity": {
            "selection_tuple_sha256": tuple_sha,
            "selection_refs": canonical_refs,
            "reference": canonical_reference,
        },
        "lane": {
            "requested_index": int(requested_index),
            "selection": "precommitted",
        },
        "validity": {"mask_channel": mask_channel},
        "route": {
            "waypoint_index_channel": waypoint_channel,
            "waypoint_count": waypoint_count,
            "ordered_regions": [
                {key: region[key] for key in ("id", "relative_channel", "radius_m")}
                for region in regions
            ],
        },
        "support_cycles": {
            "root_position_channel": root_position,
            "left_contact_channel": left_contact,
            "right_contact_channel": right_contact,
            "minimum_flight_frames": min_flight,
            "maximum_touchdown_gap_frames": touchdown_gap,
            "maximum_waypoint_advance_lag_frames": advance_lag,
            "mappings": mappings,
        },
        "forbidden_contact_channels": forbidden_channels,
        "safety": {
            "projected_gravity_channel": gravity_channel,
            "projected_gravity_z_index": int(gravity_index),
            "fall_gravity_z_above": fall_threshold,
            "maximum_consecutive_fall_frames": fall_frames,
        },
        "terminal_hold": {
            "frames": _positive_int(terminal.get("frames"), label="hold frames"),
            "finish_region_id": finish_id,
            "root_linear_velocity_channel": _channel_name(
                terminal.get("root_linear_velocity_channel"),
                label="root linear velocity channel",
            ),
            "horizontal_velocity_indices": [int(i) for i in velocity_indices],
            "horizontal_speed_below_m_s": _finite_number(
                terminal.get("horizontal_speed_below_m_s"), minimum=0.0
            ),
            "root_angular_velocity_channel": _channel_name(
                terminal.get("root_angular_velocity_channel"),
                label="root angular velocity channel",
            ),
            "angular_speed_below_rad_s": _finite_number(
                terminal.get("angular_speed_below_rad_s"), minimum=0.0
            ),
            "joint_velocity_channel": _channel_name(
                terminal.get("joint_velocity_channel"),
                label="joint velocity channel",
            ),
            "joint_speed_rms_below_rad_s": _finite_number(
                terminal.get("joint_speed_rms_below_rad_s"), minimum=0.0
            ),
            "projected_gravity_z_at_most": _finite_number(
                terminal.get("projected_gravity_z_at_most")
            ),
            "default_pose_rms_channel": _channel_name(
                terminal.get("default_pose_rms_channel"),
                label="default pose RMS channel",
            ),
            "default_pose_rms_below_rad": _finite_number(
                terminal.get("default_pose_rms_below_rad"), minimum=0.0
            ),
        },
    }


def physical_acceptance_contract_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(validate_physical_acceptance_contract(value)))


def write_precommitted_contract(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically create an immutable precommit; existing bytes never change."""

    canonical = validate_physical_acceptance_contract(value)
    data = _canonical_bytes(canonical)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return _sha256_bytes(data)


def _require_array(
    arrays: Mapping[str, Any],
    name: str,
    *,
    time_steps: int | None = None,
    lanes: int | None = None,
    rank: int | None = None,
) -> np.ndarray:
    if name not in arrays:
        raise PhysicalAcceptanceError(f"trajectory is missing {name!r}")
    result = np.asarray(arrays[name])
    if rank is not None and result.ndim != rank:
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} has wrong rank")
    if result.ndim < 2:
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} has no lane axis")
    if time_steps is not None and result.shape[0] != time_steps:
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} time axis differs")
    if lanes is not None and result.shape[1] != lanes:
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} lane axis differs")
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} is non-finite")
    return result


def _bool_channel(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.dtype.kind == "b":
        return array.astype(bool, copy=False)
    if array.dtype.kind not in "iuf" or not np.isin(array, (0, 1)).all():
        raise PhysicalAcceptanceError(f"trajectory channel {name!r} is not boolean")
    return array.astype(bool)


def _true_prefix_length(mask: np.ndarray) -> int | None:
    if mask.ndim != 1 or mask.size == 0 or not bool(mask[0]):
        return None
    false = np.flatnonzero(~mask)
    length = int(false[0]) if false.size else int(mask.size)
    if np.any(mask[length:]):
        return None
    return length


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for item in values.astype(bool, copy=False):
        if bool(item):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _runs(values: np.ndarray, *, minimum: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], values.astype(bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if int(end - start + 1) >= minimum
    ]


def _first_at_least(values: np.ndarray, threshold: int) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if indices.size else None


def _lane_result(
    arrays: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    lane: int,
    prefix: int,
) -> dict[str, Any]:
    route = contract["route"]
    support = contract["support_cycles"]
    safety = contract["safety"]
    terminal = contract["terminal_hold"]
    blockers: list[str] = []

    waypoint = np.asarray(arrays[route["waypoint_index_channel"]])[:prefix, lane]
    if waypoint.dtype.kind not in "iu" or not np.equal(
        waypoint, waypoint.astype(np.int64)
    ).all():
        blockers.append("waypoint index is not integral")
        waypoint_i = waypoint.astype(np.int64, copy=False)
    else:
        waypoint_i = waypoint.astype(np.int64, copy=False)
    count = int(route["waypoint_count"])
    if waypoint_i.size == 0 or waypoint_i[0] != 0:
        blockers.append("route does not start at raw index 0")
    if np.any((waypoint_i < 0) | (waypoint_i > count)):
        blockers.append("waypoint index is outside the admitted range")
    delta = np.diff(waypoint_i)
    if np.any((delta < 0) | (delta > 1)):
        blockers.append("waypoint index is not monotonic one-step authority")

    completion_steps: list[int | None] = []
    ordered_entry_ok = True
    for completion_index, region in enumerate(
        route["ordered_regions"], start=1
    ):
        step = _first_at_least(waypoint_i, completion_index)
        completion_steps.append(step)
        if step is None:
            ordered_entry_ok = False
            blockers.append(f"raw route never reaches index {completion_index}")
            continue
        relative = np.asarray(arrays[region["relative_channel"]])[step, lane]
        if relative.shape[-1] < 2 or float(np.linalg.norm(relative[:2])) > float(
            region["radius_m"]
        ):
            ordered_entry_ok = False
            blockers.append(
                f"raw index {completion_index} is not an entry into "
                f"region {region['id']}"
            )
    route_ok = bool(ordered_entry_ok and waypoint_i[-1] == count)

    left = _bool_channel(
        np.asarray(arrays[support["left_contact_channel"]])[:prefix, lane],
        name=support["left_contact_channel"],
    )
    right = _bool_channel(
        np.asarray(arrays[support["right_contact_channel"]])[:prefix, lane],
        name=support["right_contact_channel"],
    )
    bouts = _runs(
        ~(left | right), minimum=int(support["minimum_flight_frames"])
    )
    if len(bouts) != len(support["mappings"]):
        blockers.append(
            f"observed {len(bouts)} bilateral flight bouts, expected "
            f"{len(support['mappings'])}"
        )
    cycle_details: list[dict[str, Any]] = []
    support_ok = len(bouts) == len(support["mappings"])
    root_position = np.asarray(arrays[support["root_position_channel"]])
    for index, mapping in enumerate(support["mappings"]):
        detail: dict[str, Any] = {
            "phase_id": mapping["phase_id"],
            "obstacle_id": mapping["obstacle_id"],
            "landing_region_id": mapping["landing_region_id"],
            "flight_start_frame": None,
            "flight_end_frame": None,
            "touchdown_frame": None,
            "rail_crossed": False,
            "waypoint_correlated": False,
            "passed": False,
        }
        if index >= len(bouts):
            cycle_details.append(detail)
            support_ok = False
            continue
        start, end = bouts[index]
        detail["flight_start_frame"] = start
        detail["flight_end_frame"] = end
        touchdown_limit = min(
            prefix - 1,
            end + int(support["maximum_touchdown_gap_frames"]),
        )
        touchdown = next(
            (
                frame
                for frame in range(end + 1, touchdown_limit + 1)
                if bool(left[frame] and right[frame])
            ),
            None,
        )
        detail["touchdown_frame"] = touchdown
        if touchdown is None:
            blockers.append(f"{mapping['phase_id']} has no bilateral touchdown")
            cycle_details.append(detail)
            support_ok = False
            continue

        obstacle = np.asarray(arrays[mapping["obstacle_position_channel"]])
        axis = int(mapping["crossing_axis"])
        direction = int(mapping["crossing_direction"])
        window_start = max(0, start - 1)
        signed = direction * (
            root_position[window_start:touchdown + 1, lane, axis]
            - obstacle[window_start:touchdown + 1, lane, axis]
        )
        half_extent = float(mapping["crossing_half_extent_m"])
        crossed = bool(
            signed.size
            and float(signed[0]) <= -half_extent
            and float(np.max(signed)) >= half_extent
        )
        detail["rail_crossed"] = crossed
        if not crossed:
            blockers.append(f"{mapping['phase_id']} does not cross its obstacle")

        completion_index = int(mapping["landing_completion_index"])
        advance = completion_steps[completion_index - 1]
        advance_limit = min(
            prefix - 1,
            touchdown + int(support["maximum_waypoint_advance_lag_frames"]),
        )
        correlated = bool(
            advance is not None and start <= advance <= advance_limit
        )
        detail["waypoint_correlated"] = correlated
        if not correlated:
            blockers.append(
                f"{mapping['phase_id']} is not one-to-one with waypoint "
                f"advance {completion_index}"
            )
        detail["passed"] = bool(crossed and correlated)
        support_ok = bool(support_ok and detail["passed"])
        cycle_details.append(detail)

    contact_free = True
    contact_hits: dict[str, int] = {}
    for channel in contract["forbidden_contact_channels"]:
        values = _bool_channel(
            np.asarray(arrays[channel])[:prefix, lane], name=channel
        )
        hits = int(np.count_nonzero(values))
        contact_hits[channel] = hits
        if hits:
            contact_free = False
            blockers.append(f"forbidden contact channel {channel!r} fired")

    gravity = np.asarray(arrays[safety["projected_gravity_channel"]])[
        :prefix, lane, int(safety["projected_gravity_z_index"])
    ]
    longest_fall = _longest_true_run(
        gravity > float(safety["fall_gravity_z_above"])
    )
    no_sustained_fall = longest_fall <= int(
        safety["maximum_consecutive_fall_frames"]
    )
    if not no_sustained_fall:
        blockers.append("projected gravity proves a sustained fall")

    finish_step = completion_steps[-1] if completion_steps else None
    hold_frames = int(terminal["frames"])
    hold_end = None if finish_step is None else finish_step + hold_frames
    terminal_components = {
        "inside_finish": False,
        "horizontal_speed": False,
        "angular_speed": False,
        "joint_speed_rms": False,
        "upright": False,
        "default_pose": False,
        "valid_prefix": False,
    }
    if finish_step is None or hold_end is None or hold_end > prefix:
        blockers.append("raw index completion has fewer than the required hold frames")
    else:
        sl = slice(finish_step, hold_end)
        finish = route["ordered_regions"][-1]
        finish_rel = np.asarray(arrays[finish["relative_channel"]])[sl, lane]
        terminal_components["inside_finish"] = bool(np.all(
            np.linalg.norm(finish_rel[:, :2], axis=-1) <= float(finish["radius_m"])
        ))
        linear = np.asarray(arrays[terminal["root_linear_velocity_channel"]])[
            sl, lane
        ]
        horizontal = linear[:, terminal["horizontal_velocity_indices"]]
        terminal_components["horizontal_speed"] = bool(np.all(
            np.linalg.norm(horizontal, axis=-1)
            < float(terminal["horizontal_speed_below_m_s"])
        ))
        angular = np.asarray(arrays[terminal["root_angular_velocity_channel"]])[
            sl, lane
        ]
        terminal_components["angular_speed"] = bool(np.all(
            np.linalg.norm(angular, axis=-1)
            < float(terminal["angular_speed_below_rad_s"])
        ))
        joint = np.asarray(arrays[terminal["joint_velocity_channel"]])[sl, lane]
        terminal_components["joint_speed_rms"] = bool(np.all(
            np.sqrt(np.mean(np.square(joint), axis=-1))
            < float(terminal["joint_speed_rms_below_rad_s"])
        ))
        terminal_components["upright"] = bool(np.all(
            gravity[sl] <= float(terminal["projected_gravity_z_at_most"])
        ))
        default_pose = np.asarray(arrays[terminal["default_pose_rms_channel"]])[
            sl, lane
        ]
        terminal_components["default_pose"] = bool(np.all(
            default_pose < float(terminal["default_pose_rms_below_rad"])
        ))
        terminal_components["valid_prefix"] = True
        for component, passed in terminal_components.items():
            if not passed:
                blockers.append(f"terminal hold fails {component}")
    terminal_ok = all(terminal_components.values())

    full_pass = bool(
        route_ok
        and support_ok
        and contact_free
        and no_sustained_fall
        and terminal_ok
        and not blockers
    )
    return {
        "lane": lane,
        "valid_prefix_frames": prefix,
        "route_complete": route_ok,
        "ordered_region_entry_frames": completion_steps,
        "support_cycle_count": len(bouts),
        "support_cycles": cycle_details,
        "support_cycles_passed": support_ok,
        "forbidden_contact_free": contact_free,
        "forbidden_contact_hits": contact_hits,
        "longest_fall_frames": longest_fall,
        "no_sustained_fall": no_sustained_fall,
        "finish_entry_frame": finish_step,
        "terminal_hold_frames_required": hold_frames,
        "terminal_hold": terminal_components,
        "terminal_hold_passed": terminal_ok,
        "full_pass": full_pass,
        "blockers": blockers,
    }


def reduce_same_lane_acceptance(
    arrays: Mapping[str, Any],
    contract_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce official trajectory arrays to a conjunctive per-lane mask."""

    contract = validate_physical_acceptance_contract(contract_value)
    mask_name = contract["validity"]["mask_channel"]
    valid = _bool_channel(_require_array(arrays, mask_name, rank=2), name=mask_name)
    time_steps, lane_count = valid.shape
    if contract["lane"]["requested_index"] >= lane_count:
        raise PhysicalAcceptanceError("requested lane is outside the trajectory")

    required: dict[str, int] = {
        contract["route"]["waypoint_index_channel"]: 2,
        contract["support_cycles"]["root_position_channel"]: 3,
        contract["support_cycles"]["left_contact_channel"]: 2,
        contract["support_cycles"]["right_contact_channel"]: 2,
        contract["safety"]["projected_gravity_channel"]: 3,
        contract["terminal_hold"]["root_linear_velocity_channel"]: 3,
        contract["terminal_hold"]["root_angular_velocity_channel"]: 3,
        contract["terminal_hold"]["joint_velocity_channel"]: 3,
        contract["terminal_hold"]["default_pose_rms_channel"]: 2,
    }
    for region in contract["route"]["ordered_regions"]:
        required[region["relative_channel"]] = 3
    for mapping in contract["support_cycles"]["mappings"]:
        required[mapping["obstacle_position_channel"]] = 3
    for channel in contract["forbidden_contact_channels"]:
        required[channel] = 2
    for name, rank in required.items():
        _require_array(
            arrays, name, time_steps=time_steps, lanes=lane_count, rank=rank
        )

    lane_rows: list[dict[str, Any]] = []
    for lane in range(lane_count):
        prefix = _true_prefix_length(valid[:, lane])
        if prefix is None:
            lane_rows.append({
                "lane": lane,
                "valid_prefix_frames": 0,
                "route_complete": False,
                "support_cycles_passed": False,
                "forbidden_contact_free": False,
                "no_sustained_fall": False,
                "terminal_hold_passed": False,
                "full_pass": False,
                "blockers": ["first_episode_valid_mask is not a true prefix"],
            })
            continue
        lane_rows.append(
            _lane_result(arrays, contract, lane=lane, prefix=prefix)
        )
    mask = [bool(row["full_pass"]) for row in lane_rows]
    requested = int(contract["lane"]["requested_index"])
    return {
        "lane_count": lane_count,
        "full_pass_mask": mask,
        "full_pass_count": int(sum(mask)),
        "requested_lane": requested,
        "requested_lane_full_pass": mask[requested],
        "lanes": lane_rows,
    }


def _parse_json_scalar(value: Any, *, label: str) -> dict[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise PhysicalAcceptanceError(f"{label} is not a Unicode JSON scalar")
    try:
        parsed = json.loads(str(array.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PhysicalAcceptanceError(f"{label} is unreadable") from exc
    if not isinstance(parsed, dict):
        raise PhysicalAcceptanceError(f"{label} is not an object")
    return parsed


def _verify_iteration_identity(
    *,
    contract: Mapping[str, Any],
    artifact_tuple: Mapping[str, Any],
    behavior: Mapping[str, Any],
    trajectory_contract: Mapping[str, Any],
) -> dict[str, Any]:
    identity = contract["identity"]
    tuple_hash = artifact_tuple.get("tuple_hash")
    if tuple_hash != identity["selection_tuple_sha256"]:
        raise PhysicalAcceptanceError("artifact tuple differs from precommit")
    refs = artifact_tuple.get("refs")
    if not isinstance(refs, Mapping):
        raise PhysicalAcceptanceError("artifact tuple refs are missing")
    for key, expected in identity["selection_refs"].items():
        ref = refs.get(key)
        if not isinstance(ref, Mapping) or ref.get("sha256") != expected:
            raise PhysicalAcceptanceError(
                f"artifact tuple {key!r} differs from precommit"
            )
    artifact_created = artifact_tuple.get("created_at")
    if (
        isinstance(artifact_created, bool)
        or not isinstance(artifact_created, (int, float))
        or not math.isfinite(float(artifact_created))
        or float(contract["created_at"]) > float(artifact_created)
    ):
        raise PhysicalAcceptanceError(
            "acceptance contract was not precommitted before the iteration"
        )

    requested = behavior.get("rendered_env_index_requested")
    resolved = behavior.get("rendered_env_index")
    selection = behavior.get("rendered_env_selection")
    expected_lane = contract["lane"]["requested_index"]
    if requested != expected_lane or resolved != expected_lane:
        raise PhysicalAcceptanceError(
            "worker rendered a different lane from the acceptance precommit"
        )
    if selection != "precommitted":
        raise PhysicalAcceptanceError("worker lane selection was not precommitted")

    runtime = trajectory_contract.get("runtime_artifacts")
    if not isinstance(runtime, Mapping):
        raise PhysicalAcceptanceError("trajectory runtime receipt is missing")
    if runtime.get("phase") != "rollout" or runtime.get(
        "reward_module_sha256"
    ) != identity["selection_refs"]["reward"]:
        raise PhysicalAcceptanceError("runtime reward differs from precommit")
    environment = runtime.get("environment_artifacts")
    if not isinstance(environment, Mapping):
        raise PhysicalAcceptanceError("runtime environment receipt is missing")
    world_selection = environment.get("world_selection")
    if (
        not isinstance(world_selection, Mapping)
        or world_selection.get("present") is not True
        or world_selection.get("tuple_hash")
        != identity["selection_tuple_sha256"]
    ):
        raise PhysicalAcceptanceError("runtime world tuple differs from precommit")
    observed_refs = world_selection.get("refs")
    if not isinstance(observed_refs, Mapping):
        raise PhysicalAcceptanceError("runtime world refs are missing")
    for key, expected in identity["selection_refs"].items():
        ref = observed_refs.get(key)
        if not isinstance(ref, Mapping) or ref.get("sha256") != expected:
            raise PhysicalAcceptanceError(
                f"runtime world ref {key!r} differs from precommit"
            )
    return {
        "selection_tuple_sha256": identity["selection_tuple_sha256"],
        "selection_refs": dict(identity["selection_refs"]),
        "reference": dict(identity["reference"]),
        "requested_lane": expected_lane,
        "resolved_lane": resolved,
    }


def _receipt_skeleton() -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "unavailable",
        "contract_sha256": None,
        "artifact_tuple_sha256": None,
        "trajectory_sha256": None,
        "behavior_sha256": None,
        "identity": None,
        "lane_count": 0,
        "full_pass_mask": [],
        "full_pass_count": 0,
        "requested_lane": None,
        "requested_lane_full_pass": False,
        "lanes": [],
        "reason": None,
    }


def _unavailable_receipt(reason: str) -> dict[str, Any]:
    result = _receipt_skeleton()
    result["reason"] = reason
    result["receipt_sha256"] = _sha256_bytes(_canonical_bytes(dict(result)))
    return result


def evaluate_iteration_same_lane_acceptance(
    iter_dir: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate retained official artifacts and return a durable receipt.

    Missing/malformed evidence returns ``status='unavailable'`` with an empty
    pass mask.  It never falls back to independently aggregated fitness
    components.
    """

    iter_dir = Path(iter_dir).expanduser().resolve()
    contract_path = (
        iter_dir / "physical_acceptance_contract.json"
        if contract_path is None
        else Path(contract_path).expanduser().resolve()
    )
    trajectory_path = iter_dir / "rollout" / "trajectory.npz"
    behavior_path = iter_dir / "rollout" / "behavior.json"
    tuple_path = iter_dir / "artifact_tuple.json"
    result = _receipt_skeleton()
    try:
        for path, label in (
            (contract_path, "precommitted acceptance contract"),
            (tuple_path, "artifact tuple"),
            (trajectory_path, "official trajectory"),
            (behavior_path, "worker behavior receipt"),
        ):
            if not path.is_file() or path.is_symlink():
                raise PhysicalAcceptanceError(f"{label} is missing or not plain")
        contract_bytes = contract_path.read_bytes()
        parsed_contract = json.loads(contract_bytes.decode("utf-8"))
        contract = validate_physical_acceptance_contract(parsed_contract)
        if contract_bytes != _canonical_bytes(contract):
            raise PhysicalAcceptanceError(
                "acceptance contract bytes are not canonical/immutable"
            )
        artifact_tuple = json.loads(tuple_path.read_text(encoding="utf-8"))
        behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
        if not isinstance(artifact_tuple, dict) or not isinstance(behavior, dict):
            raise PhysicalAcceptanceError("iteration JSON evidence is malformed")
        with np.load(trajectory_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        trajectory_contract = _parse_json_scalar(
            arrays.get("trajectory_contract_json"),
            label="trajectory_contract_json",
        )
        identity = _verify_iteration_identity(
            contract=contract,
            artifact_tuple=artifact_tuple,
            behavior=behavior,
            trajectory_contract=trajectory_contract,
        )
        reduced = reduce_same_lane_acceptance(arrays, contract)
        result.update({
            # The precommitted rendered lane is the human-review authority.
            # Passing on a different lane is useful batch evidence, but it
            # cannot silently substitute for the exact lane named before the
            # rollout and shown in the retained video.
            "status": (
                "passed" if reduced["requested_lane_full_pass"] else "failed"
            ),
            "contract_sha256": _sha256_bytes(contract_bytes),
            "artifact_tuple_sha256": _sha256_file(tuple_path),
            "trajectory_sha256": _sha256_file(trajectory_path),
            "behavior_sha256": _sha256_file(behavior_path),
            "identity": identity,
            **reduced,
            "reason": None,
        })
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        # A truncated/corrupt trajectory.npz — a legitimate crash artifact —
        # raises BadZipFile from np.load, which subclasses none of the above.
        # Without it here, one such iteration turned the whole project's
        # policy listing into an unhandled 500 (2026-08-25 audit repro).
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        PhysicalAcceptanceError,
    ) as exc:
        result["reason"] = str(exc)

    receipt_payload = dict(result)
    result["receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt_payload))
    return result


def _persist_receipt(iter_dir: Path, result: Mapping[str, Any]) -> None:
    target = iter_dir / "physical_acceptance_receipt.json"
    temporary = iter_dir / ".physical_acceptance_receipt.json.tmp"
    data = json.dumps(
        dict(result),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary.write_bytes(data)
    os.replace(temporary, target)


def write_iteration_acceptance_receipt(
    iter_dir: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate once and atomically persist the observed, content-bound receipt."""

    iter_dir = Path(iter_dir).expanduser().resolve()
    result = evaluate_iteration_same_lane_acceptance(
        iter_dir, contract_path=contract_path
    )
    _persist_receipt(iter_dir, result)
    return result


def _load_verified_receipt(receipt_path: Path) -> dict[str, Any] | None:
    """Return the persisted receipt only when its bytes still self-verify."""
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != RECEIPT_SCHEMA:
        return None
    stored = raw.get("receipt_sha256")
    payload = {key: value for key, value in raw.items() if key != "receipt_sha256"}
    try:
        recomputed = _sha256_bytes(_canonical_bytes(payload))
    except (TypeError, ValueError):
        return None
    if not isinstance(stored, str) or stored != recomputed:
        return None
    return raw


def load_or_evaluate_iteration_acceptance(
    iter_dir: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Receipt-first acceptance authority for a retained iteration.

    A persisted, self-verifying receipt with a terminal verdict is returned
    as-is, so deleting the precommitted contract after a failed verdict can
    never silently revert the pass authority to independently aggregated
    route/contact/hold components (2026-08-25 audit finding).  With no usable
    receipt, one fresh evaluation runs and terminal verdicts are persisted;
    a receipt that exists but fails verification while its contract is also
    gone stays ``unavailable`` with an explicit reason — never aggregates.

    Residual boundary, documented rather than closed here: deleting *both*
    files before any evaluation leaves no trace inside ``iter_dir``.  Closing
    that requires the launch-side producer pinning the contract digest into
    worker-written evidence, which does not exist yet.
    """
    iter_dir = Path(iter_dir).expanduser().resolve()
    receipt_path = iter_dir / "physical_acceptance_receipt.json"
    resolved_contract = (
        iter_dir / "physical_acceptance_contract.json"
        if contract_path is None
        else Path(contract_path).expanduser().resolve()
    )
    if receipt_path.is_file() and not receipt_path.is_symlink():
        persisted = _load_verified_receipt(receipt_path)
        if persisted is not None and persisted.get("status") in (
            "passed",
            "failed",
        ):
            return persisted
        if persisted is None and not (
            resolved_contract.is_file()
            and not resolved_contract.is_symlink()
        ):
            return _unavailable_receipt(
                "persisted acceptance receipt failed verification and the "
                "precommitted contract is missing"
            )
    result = evaluate_iteration_same_lane_acceptance(
        iter_dir, contract_path=contract_path
    )
    if result.get("status") in ("passed", "failed"):
        try:
            _persist_receipt(iter_dir, result)
        except OSError:
            # Persistence is defense-in-depth; the just-computed verdict is
            # already bound to artifact hashes, so a read-only store must not
            # turn a valid listing into an error.
            pass
    return result


__all__ = [
    "CONTRACT_SCHEMA",
    "RECEIPT_SCHEMA",
    "PhysicalAcceptanceError",
    "evaluate_iteration_same_lane_acceptance",
    "load_or_evaluate_iteration_acceptance",
    "physical_acceptance_contract_sha256",
    "reduce_same_lane_acceptance",
    "validate_physical_acceptance_contract",
    "write_iteration_acceptance_receipt",
    "write_precommitted_contract",
]
