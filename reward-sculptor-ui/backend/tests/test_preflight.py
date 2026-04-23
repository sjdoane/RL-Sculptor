"""Tests for backend/services/preflight.check_mjlab_preflight (M4 §6)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.services import preflight as preflight_mod


_BASE_SNAPSHOT = {
    "pynvml_available": True,
    "driver_version": "592.00",
    "nvml_version": "13.0",
    "device_count": 1,
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "total_memory_bytes": 8 * 1024 ** 3,
            "free_memory_bytes": 7 * 1024 ** 3,
            "used_memory_bytes": 1 * 1024 ** 3,
            "utilization_percent": 0.0,
            "temperature_c": 53.0,
            "multi_processor_count": 36,
            "major": 12,
            "minor": 0,
        }
    ],
}


def _with_snapshot(snapshot):  # noqa: ANN001
    from backend.services import gpu_monitor
    return patch.object(
        gpu_monitor, "get_live_snapshot", return_value=snapshot
    )


def test_preflight_ok_when_num_envs_fits() -> None:
    with _with_snapshot(_BASE_SNAPSHOT):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Velocity-Flat-Unitree-Go1",
            num_envs=1024,
            device="cuda:0",
        )
    assert r.ok
    assert r.suggested_num_envs == 1024
    assert r.free_vram_gb == pytest.approx(7.0, abs=0.05)


def test_preflight_fails_and_suggests_smaller_num_envs() -> None:
    with _with_snapshot(_BASE_SNAPSHOT):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            num_envs=32768,
            device="cuda:0",
        )
    assert r.ok is False
    assert r.problem_type == "/problems/insufficient-vram"
    assert r.suggested_num_envs is not None
    assert 128 <= r.suggested_num_envs <= 4096
    assert r.suggested_num_envs < 32768
    # Error reason includes the free-VRAM figure so the UI can render it.
    assert "GiB free" in (r.reason or "")


def test_preflight_rejects_missing_device() -> None:
    snap = dict(_BASE_SNAPSHOT, devices=[])
    with _with_snapshot(snap):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Cartpole-Balance",
            num_envs=128,
            device="cuda:0",
        )
    assert r.ok is False
    assert r.problem_type == "/problems/gpu-required"


def test_preflight_rejects_out_of_range_device_index() -> None:
    with _with_snapshot(_BASE_SNAPSHOT):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Cartpole-Balance",
            num_envs=128,
            device="cuda:7",
        )
    assert r.ok is False
    assert r.problem_type == "/problems/device-unavailable"
    assert "only 1 CUDA" in (r.reason or "")


def test_preflight_uses_cached_coefficient_when_available(tmp_path: Path) -> None:
    """When a per-project VRAM-probe cache exists with the current mjlab
    version, preflight uses the measured coefficient instead of the
    static 0.5 MB/env fallback."""
    from importlib.metadata import version

    cache_dir = tmp_path / ".sculptor_cache"
    cache_dir.mkdir()
    # Force a very cheap coefficient so 4096 envs fit trivially — proves
    # the cached value is being honored (static path would fail).
    (cache_dir / "vram_coefficients.json").write_text(
        json.dumps({
            "_cache_key": f"Mjlab-Cartpole-Balance|{version('mjlab')}",
            "coefficient_bytes_per_env": 100_000,  # 0.1 MB/env
        })
    )
    with _with_snapshot(_BASE_SNAPSHOT):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Cartpole-Balance",
            num_envs=4096,
            device="cuda:0",
            project_dir=tmp_path,
        )
    assert r.ok
    # 4096 × 100 000 = ~0.4 GiB + 1.5 GiB policy = ~1.9 GiB << 6 GiB budget.
    assert r.estimated_required_gb < 2.0


def test_preflight_ignores_cache_for_wrong_task_id(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".sculptor_cache"
    cache_dir.mkdir()
    (cache_dir / "vram_coefficients.json").write_text(
        json.dumps({"_cache_key": "some-other-task|1.0", "coefficient_bytes_per_env": 100_000})
    )
    with _with_snapshot(_BASE_SNAPSHOT):
        r = preflight_mod.check_mjlab_preflight(
            task_id="Mjlab-Velocity-Flat-Unitree-G1",
            num_envs=32768,
            device="cuda:0",
            project_dir=tmp_path,
        )
    # Cache key didn't match — static path applies and the huge num_envs fails.
    assert r.ok is False


# ── End-to-end project creation with live-preflight 412 shape ──────────────
def test_create_mjlab_project_412_body_shape(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """M4 verification #2: 412 body carries suggested_num_envs +
    free_vram_gb so the UI can render a meaningful retry dialog."""
    from backend.services import robot_library as lib_mod
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for live-VRAM preflight")

    lib_mod.reset_library_singleton()
    with _with_snapshot(_BASE_SNAPSHOT):
        r = client.post(
            "/projects",
            json={
                "name": "G1 OOM Test",
                "adapter": "mjlab",
                "task_id": "Mjlab-Velocity-Flat-Unitree-G1",
                "num_envs": 32768,
            },
        )
    assert r.status_code == 412, r.text
    body = r.json()
    assert body["type"] == "/problems/insufficient-vram"
    assert "suggested_num_envs" in body
    assert body["suggested_num_envs"] < 32768
    assert "free_vram_gb" in body
    assert "estimated_required_gb" in body
    assert "device_name" in body
