"""Fail-closed audit for physical authored-world alignment in rollout evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "max_error_m": None,
        "threshold_m": 0.15,
        "objects_checked": [],
        "reason": reason,
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
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _unavailable(f"could not audit trajectory: {exc}")

    max_error = max(errors.values(), default=0.0)
    aligned = bool(np.isfinite(max_error) and max_error <= threshold_m)
    return {
        "status": "aligned" if aligned else "misaligned",
        "max_error_m": round(max_error, 6),
        "threshold_m": float(threshold_m),
        "objects_checked": sorted(errors),
        "object_errors_m": {
            name: round(error, 6) for name, error in sorted(errors.items())
        },
        "reason": (
            "fixed authored objects match nominal local pose + env origin"
            if aligned
            else "fixed authored objects occupy a different frame from the "
                 "robot, commands, and validators"
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
