from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.services.physical_scene_audit import (
    audit_physical_scene_alignment,
)


def _write_case(
    tmp_path: Path,
    *,
    aligned: bool,
    origins: np.ndarray | None = None,
    object_size_m: tuple[float, float, float] = (0.45, 0.45, 0.75),
) -> tuple[Path, Path]:
    project = tmp_path / "project"
    iter_dir = project / "runs" / "iter_1"
    rollout = iter_dir / "rollout"
    env_dir = project / "env"
    rollout.mkdir(parents=True)
    env_dir.mkdir()
    manifest = {
        "zones": {
            "waypoint_01": {
                "kind": "disk",
                "center_m": [2.0, 0.85],
                "radius_m": 0.35,
            },
        },
        "objects": {
            "box_01": {
                "fixed": True,
                "shape": "box",
                "position_m": [2.0, 0.0, 0.375],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "nominal": {"size_m": list(object_size_m)},
            },
        },
    }
    (env_dir / "resolved_eval_v1.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    (iter_dir / "artifact_tuple.json").write_text(json.dumps({
        "refs": {
            "resolved_eval": {"path": "env/resolved_eval_v1.json"},
        },
    }), encoding="utf-8")

    if origins is None:
        origins = np.asarray([[7.0, -7.0, 0.0], [0.0, 0.0, 0.0]])
    local_robot = np.asarray([2.0, 0.85, 0.8])
    root = np.broadcast_to(
        origins + local_robot, (3, 2, 3)).copy()
    relative = np.broadcast_to(
        np.asarray([0.0, 0.0, -0.8]), root.shape).copy()
    nominal = np.asarray([2.0, 0.0, 0.375])
    if aligned:
        objects = np.broadcast_to(origins + nominal, root.shape).copy()
    else:
        objects = np.broadcast_to(nominal, root.shape).copy()
    np.savez(
        rollout / "trajectory.npz",
        root_link_pos_w=root,
        region__waypoint_01__relative=relative,
        object__box_01__pos_w=objects,
    )
    return project, iter_dir


def test_physical_scene_audit_accepts_env_local_objects(tmp_path: Path) -> None:
    project, iter_dir = _write_case(tmp_path, aligned=True)
    result = audit_physical_scene_alignment(project, iter_dir)
    assert result["status"] == "aligned"
    assert result["max_error_m"] == 0.0


def test_physical_scene_audit_rejects_global_object_reuse(tmp_path: Path) -> None:
    project, iter_dir = _write_case(tmp_path, aligned=False)
    result = audit_physical_scene_alignment(project, iter_dir)
    assert result["status"] == "misaligned"
    assert result["max_error_m"] > 9.0
    assert result["objects_checked"] == ["box_01"]


def test_physical_scene_audit_rejects_cross_environment_geometry_overlap(
    tmp_path: Path,
) -> None:
    project, iter_dir = _write_case(
        tmp_path,
        aligned=True,
        origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        object_size_m=(3.0, 1.0, 0.75),
    )
    result = audit_physical_scene_alignment(project, iter_dir)
    assert result["status"] == "misaligned"
    assert result["max_error_m"] == 0.0
    assert result["cross_env_geometry"]["status"] == "overlap"
    assert result["cross_env_geometry"]["overlap_pair_count"] == 1
    assert "overlaps another environment" in result["reason"]


def test_physical_scene_audit_requires_object_extent_for_parallel_worlds(
    tmp_path: Path,
) -> None:
    project, iter_dir = _write_case(tmp_path, aligned=True)
    manifest_path = project / "env" / "resolved_eval_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"]["box_01"].pop("shape")
    manifest["objects"]["box_01"].pop("nominal")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_physical_scene_alignment(project, iter_dir)
    assert result["status"] == "unavailable"
    assert "no provable XY extent" in result["reason"]
