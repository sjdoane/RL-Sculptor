"""Tests for mjlab project creation validation (MJLAB_PIVOT_DESIGN §9
+ M2 verification #3).

The `adapter="mjlab"` path:
  - 412 /problems/mjlab-task-required when `task_id` is missing.
  - 412 /problems/mjlab-missing when mjlab is not importable.
  - 412 /problems/gpu-required when no CUDA.
  - 412 /problems/cuda-version when CUDA < 12.4.
  - 412 /problems/insufficient-vram when num_envs > VRAM headroom.
  - 201 + adapter_class=MjlabAdapter + task_id in adapter_config on success.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_mjlab_rejects_missing_task_id(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post(
        "/projects",
        json={"name": "Go1 Test", "adapter": "mjlab"},
    )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/mjlab-task-required"
    assert "task_id" in (body["detail"] or "")


def test_mjlab_rejects_when_mjlab_unavailable(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Patch find_spec probe to simulate uninstalled mjlab."""
    from backend.services import sculptor_bridge
    with patch.object(sculptor_bridge, "mjlab_available", return_value=False):
        r = client.post(
            "/projects",
            json={
                "name": "Go1 Test No Mjlab",
                "adapter": "mjlab",
                "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
            },
        )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/mjlab-missing"


def test_mjlab_rejects_when_no_cuda(
    client: TestClient, tmp_projects_root: Path
) -> None:
    from backend.services import sculptor_bridge
    fake_info = {
        "torch_available": True, "torch_version": "2.11.0",
        "cuda_available": False, "cuda_version": None,
        "device_count": 0, "devices": [],
        "mjlab_available": True, "mujoco_warp_available": True,
        "rsl_rl_available": True,
    }
    with patch.object(sculptor_bridge, "gpu_info", return_value=fake_info):
        r = client.post(
            "/projects",
            json={
                "name": "No GPU Test",
                "adapter": "mjlab",
                "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
            },
        )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/gpu-required"


def test_mjlab_rejects_stale_cuda_version(
    client: TestClient, tmp_projects_root: Path
) -> None:
    from backend.services import sculptor_bridge
    fake_info = {
        "torch_available": True, "torch_version": "2.11.0",
        "cuda_available": True, "cuda_version": "11.8",
        "device_count": 1,
        "devices": [{
            "index": 0, "name": "GTX 1080", "total_memory_bytes": 8 * 1024 ** 3,
            "free_memory_bytes": 6 * 1024 ** 3, "multi_processor_count": 20,
            "major": 6, "minor": 1,
        }],
        "mjlab_available": True, "mujoco_warp_available": True,
        "rsl_rl_available": True,
    }
    with patch.object(sculptor_bridge, "gpu_info", return_value=fake_info):
        r = client.post(
            "/projects",
            json={
                "name": "Old CUDA Test",
                "adapter": "mjlab",
                "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
            },
        )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/cuda-version"


def test_mjlab_rejects_insufficient_vram(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """num_envs=32768 on an 8GB GPU exceeds the 85% headroom — should
    412 with a suggested_num_envs in the body."""
    from backend.services import sculptor_bridge
    fake_info = {
        "torch_available": True, "torch_version": "2.11.0",
        "cuda_available": True, "cuda_version": "13.0",
        "device_count": 1,
        "devices": [{
            "index": 0, "name": "RTX 5070 Laptop", "total_memory_bytes": 8 * 1024 ** 3,
            "free_memory_bytes": 7 * 1024 ** 3, "multi_processor_count": 36,
            "major": 12, "minor": 0,
        }],
        "mjlab_available": True, "mujoco_warp_available": True,
        "rsl_rl_available": True,
    }
    with patch.object(sculptor_bridge, "gpu_info", return_value=fake_info):
        r = client.post(
            "/projects",
            json={
                "name": "Too Many Envs",
                "adapter": "mjlab",
                "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
                "num_envs": 32768,
            },
        )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/insufficient-vram"
    assert "suggested_num_envs" in body
    assert isinstance(body["suggested_num_envs"], int)
    assert body["suggested_num_envs"] < 32768


def test_mjlab_success_writes_mjlab_adapter_to_config(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Happy path: mjlab installed + CUDA 13.0 + 8GB + num_envs=1024 fits.
    Expect 201 and adapter_class == MjlabAdapter with task_id in config."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")

    r = client.post(
        "/projects",
        json={
            "name": "Mjlab Go1 Happy",
            "adapter": "mjlab",
            "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
            "num_envs": 1024,
            "gpu_device": "cuda:0",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_class"] == "sculptor.adapters.mjlab.MjlabAdapter"
    cfg = body["adapter_config"]
    assert cfg["task_id"] == "Mjlab-Velocity-Flat-Unitree-Go1"
    assert cfg["num_envs"] == 1024
    assert cfg["device"] == "cuda:0"

    project_dir = tmp_projects_root / body["slug"]
    toml_text = (project_dir / "config.toml").read_text()
    assert "sculptor.adapters.mjlab.MjlabAdapter" in toml_text
    assert "Mjlab-Velocity-Flat-Unitree-Go1" in toml_text


def test_gym_sb3_project_creation_unchanged(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Zero-regression check: existing gym_sb3 path still works
    without the new mjlab fields."""
    r = client.post(
        "/projects",
        json={"name": "Hopper Classic", "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["adapter_class"] == "sculptor.adapters.gym_sb3.GymSB3Adapter"
