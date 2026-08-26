"""Fail-closed audit for physical authored-world alignment in rollout evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_MAX_OVERLAP_EXAMPLES = 12


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "max_error_m": None,
        "threshold_m": 0.15,
        "objects_checked": [],
        "reason": reason,
    }


def _cross_env_geometry_receipt(
    *,
    root: np.ndarray,
    origins: np.ndarray,
    fixed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prove replicated authored geometry is disjoint across environments.

    Local object-position alignment alone cannot see a course from environment
    ``j`` covering the robot or course in environment ``i``. The runtime shares
    one MuJoCo model, so every fixed object is repeated at every exact
    ``env_origin_w``. Reconstruct those world-axis AABBs from the frozen
    manifest and reject any cross-environment intersection.
    """
    if root.ndim != 3 or root.shape[-1] != 3:
        return {"status": "unavailable", "reason": "root trajectory is not T×E×3"}
    if origins.shape != root.shape:
        return {"status": "unavailable", "reason": "environment origins do not match root shape"}
    n_envs = int(root.shape[1])
    if n_envs < 2:
        return {
            "status": "not_applicable",
            "environments_checked": n_envs,
            "overlap_pair_count": 0,
            "neighbor_root_projection_intrusion_lanes": 0,
            "reason": "one rollout environment has no replicated neighbor",
        }
    origin_drift = np.nanmax(np.abs(origins - origins[:1]), axis=(0, 2))
    # Older float32 trajectories derive origins from two recorded channels;
    # allow only their sub-micrometre round-off, never physical motion.
    if not np.all(np.isfinite(origin_drift)) or np.any(origin_drift > 1e-5):
        return {
            "status": "unavailable",
            "reason": "per-environment origins change during the rollout",
        }

    try:
        from sculptor.world.compiler import authored_object_half_extents_xy
    except ImportError as exc:
        return {
            "status": "unavailable",
            "reason": f"world geometry authority is unavailable: {exc}",
        }

    objects: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, record in sorted(fixed.items()):
        nominal_position = np.asarray(record.get("position_m"), dtype=np.float64)
        if nominal_position.shape != (3,) or not np.all(np.isfinite(nominal_position)):
            return {
                "status": "unavailable",
                "reason": f"fixed object {name!r} has no finite 3D nominal position",
            }
        half = np.asarray(authored_object_half_extents_xy(record), dtype=np.float64)
        if half.shape != (2,) or not np.all(np.isfinite(half)) or np.any(half <= 0.0):
            return {
                "status": "unavailable",
                "reason": f"fixed object {name!r} has no provable XY extent",
            }
        objects.append((name, nominal_position[:2], half))

    exact_origins = np.asarray(origins[0, :, :2], dtype=np.float64)
    overlap_count = 0
    overlap_examples: list[dict[str, Any]] = []
    for left_env in range(n_envs):
        for right_env in range(left_env + 1, n_envs):
            for left_name, left_local, left_half in objects:
                left_center = exact_origins[left_env] + left_local
                for right_name, right_local, right_half in objects:
                    right_center = exact_origins[right_env] + right_local
                    separation = np.abs(left_center - right_center)
                    if np.all(separation < (left_half + right_half) - 1e-12):
                        overlap_count += 1
                        if len(overlap_examples) < _MAX_OVERLAP_EXAMPLES:
                            overlap_examples.append({
                                "left_env": left_env,
                                "left_object": left_name,
                                "right_env": right_env,
                                "right_object": right_name,
                            })

    intrusion_lanes: set[int] = set()
    for lane in range(n_envs):
        lane_xy = root[:, lane, :2]
        for neighbor in range(n_envs):
            if neighbor == lane:
                continue
            for _, local_position, half in objects:
                center = exact_origins[neighbor] + local_position
                if np.any(np.all(np.abs(lane_xy - center) <= half, axis=-1)):
                    intrusion_lanes.add(lane)
                    break
            if lane in intrusion_lanes:
                break

    clear = overlap_count == 0 and not intrusion_lanes
    return {
        "status": "clear" if clear else "overlap",
        "environments_checked": n_envs,
        "overlap_pair_count": overlap_count,
        "overlap_examples": overlap_examples,
        "neighbor_root_projection_intrusion_lanes": len(intrusion_lanes),
        "reason": (
            "replicated fixed-object AABBs and neighboring robot projections are disjoint"
            if clear
            else "replicated authored geometry overlaps another environment"
        ),
    }


def audit_physical_scene_alignment(
    project_dir: Path,
    iter_dir: Path,
    *,
    threshold_m: float = 0.15,
) -> dict[str, Any]:
    """Compare fixed-object rollout poses with the frozen local manifest.

    Region-relative channels and the robot's world pose provide the simulator's
    per-environment origin for older trajectories.  Newer trajectories may
    provide ``env_origin_w`` directly.  Any missing input yields ``unavailable``
    rather than silently treating the evidence as aligned.
    """
    tuple_path = iter_dir / "artifact_tuple.json"
    trajectory_path = iter_dir / "rollout" / "trajectory.npz"
    if not tuple_path.is_file() or not trajectory_path.is_file():
        return _unavailable("artifact tuple or trajectory is missing")
    try:
        artifact_tuple = json.loads(tuple_path.read_text(encoding="utf-8"))
        manifest_ref = artifact_tuple["refs"]["resolved_eval"]["path"]
        manifest_path = (project_dir / str(manifest_ref)).resolve()
        if project_dir.resolve() not in manifest_path.parents:
            return _unavailable("resolved-evaluation path escapes the project")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _unavailable(f"could not load frozen manifest: {exc}")

    fixed = {
        str(name): record
        for name, record in (manifest.get("objects") or {}).items()
        if isinstance(record, dict) and bool(record.get("fixed", False))
    }
    if not fixed:
        return {
            "status": "not_applicable",
            "max_error_m": 0.0,
            "threshold_m": float(threshold_m),
            "objects_checked": [],
            "reason": "frozen manifest declares no fixed authored objects",
        }

    try:
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            root = np.asarray(trajectory["root_link_pos_w"], dtype=np.float64)
            if "env_origin_w" in trajectory:
                origins = np.asarray(
                    trajectory["env_origin_w"], dtype=np.float64)
                if origins.ndim == 2:
                    origins = np.broadcast_to(origins, root.shape)
            else:
                origins = _derive_env_origins(
                    trajectory, root, manifest.get("zones") or {})
            if origins is None or origins.shape != root.shape:
                return _unavailable(
                    "trajectory has no derivable per-environment origin")

            errors: dict[str, float] = {}
            for name, record in fixed.items():
                channel = f"object__{name}__pos_w"
                if channel not in trajectory:
                    return _unavailable(
                        f"trajectory is missing fixed-object channel {channel!r}")
                position = np.asarray(trajectory[channel], dtype=np.float64)
                if position.shape != root.shape:
                    return _unavailable(
                        f"fixed-object channel {channel!r} has shape "
                        f"{position.shape}, expected {root.shape}")
                nominal = np.asarray(record.get("position_m"), dtype=np.float64)
                if nominal.shape != (3,):
                    return _unavailable(
                        f"fixed object {name!r} has no 3D nominal position")
                error = np.linalg.norm(position - origins - nominal, axis=-1)
                errors[name] = float(np.nanmax(error))
            cross_env = _cross_env_geometry_receipt(
                root=root,
                origins=origins,
                fixed=fixed,
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _unavailable(f"could not audit trajectory: {exc}")

    if cross_env.get("status") == "unavailable":
        result = _unavailable(str(cross_env.get("reason") or "cross-environment geometry is unavailable"))
        result["cross_env_geometry"] = cross_env
        return result
    max_error = max(errors.values(), default=0.0)
    local_aligned = bool(np.isfinite(max_error) and max_error <= threshold_m)
    cross_env_clear = cross_env.get("status") in {"clear", "not_applicable"}
    aligned = local_aligned and cross_env_clear
    return {
        "status": "aligned" if aligned else "misaligned",
        "max_error_m": round(max_error, 6),
        "threshold_m": float(threshold_m),
        "objects_checked": sorted(errors),
        "object_errors_m": {
            name: round(error, 6) for name, error in sorted(errors.items())
        },
        "cross_env_geometry": cross_env,
        "reason": (
            "fixed authored objects match their local frame and replicated geometry is disjoint"
            if aligned
            else (
                "replicated authored geometry overlaps another environment"
                if not cross_env_clear
                else "fixed authored objects occupy a different frame from the "
                     "robot, commands, and validators"
            )
        ),
    }


def _derive_env_origins(
    trajectory: Any,
    root: np.ndarray,
    zones: dict[str, Any],
) -> np.ndarray | None:
    for name, zone in sorted(zones.items()):
        channel = f"region__{name}__relative"
        if channel not in trajectory or not isinstance(zone, dict):
            continue
        relative = np.asarray(trajectory[channel], dtype=np.float64)
        if relative.shape != root.shape:
            continue
        center = np.asarray(zone.get("center_m"), dtype=np.float64)
        if center.shape == (2,):
            center = np.concatenate((center, np.zeros(1, dtype=np.float64)))
        if center.shape != (3,):
            continue
        # relative = center_local - robot_local
        #          = center - (robot_world - env_origin)
        return relative - center + root
    return None
