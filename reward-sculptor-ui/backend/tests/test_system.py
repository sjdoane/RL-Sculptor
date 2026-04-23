"""Tests for GET /system/gpu (MJLAB_PIVOT_DESIGN §3.4).

Runs against the real machine — the response values depend on whether
the test host has CUDA + mjlab installed. On the RTX 5070 Laptop dev
machine the fields below are asserted concretely; elsewhere the test
degrades to shape-only assertions.
"""

from __future__ import annotations

import importlib.util

from fastapi.testclient import TestClient


def test_system_gpu_returns_shape(client: TestClient) -> None:
    r = client.get("/system/gpu")
    assert r.status_code == 200, r.text
    body = r.json()

    # Required fields — shape-level, always present.
    for key in (
        "torch_available", "cuda_available", "device_count", "devices",
        "mjlab_available", "mujoco_warp_available", "rsl_rl_available",
    ):
        assert key in body, f"GET /system/gpu missing key {key!r}"

    assert isinstance(body["devices"], list)
    assert isinstance(body["device_count"], int)


def test_system_gpu_lazy_import_discipline(client: TestClient) -> None:
    """find_spec-only probes return True iff the package is installed,
    regardless of whether something elsewhere has imported it. Verifies
    the lazy-import discipline from MJLAB_PIVOT_DESIGN §7."""
    r = client.get("/system/gpu")
    body = r.json()

    expected_mjlab = importlib.util.find_spec("mjlab") is not None
    expected_warp = importlib.util.find_spec("mujoco_warp") is not None
    expected_rsl = importlib.util.find_spec("rsl_rl") is not None
    assert body["mjlab_available"] is expected_mjlab
    assert body["mujoco_warp_available"] is expected_warp
    assert body["rsl_rl_available"] is expected_rsl


def test_cuda_version_ok_uses_numeric_comparison(client: TestClient) -> None:
    """Pre-M3 gate (C): `packaging.version.parse` (not string compare).
    String comparison would say "9.0" > "12.4" because "9" > "1"
    lexically, which breaks the "CUDA >= 12.4 required" gate."""
    from unittest.mock import patch

    from backend.services import sculptor_bridge

    cases = [
        # (cuda_version string, min_major, min_minor, expected_ok)
        ("13.0", 12, 4, True),   # future CUDA versions
        ("12.8", 12, 4, True),
        ("12.4", 12, 4, True),   # exact match
        ("12.3", 12, 4, False),  # just under
        ("12.0", 12, 4, False),
        ("11.8", 12, 4, False),
        ("9.0",  12, 4, False),  # <- the lexical-compare trap
        ("8.0",  12, 4, False),
        (None,   12, 4, False),
        ("",     12, 4, False),
        ("not.a.version", 12, 4, False),
    ]
    for cuda_v, major, minor, expected in cases:
        fake_info = {"cuda_version": cuda_v}
        with patch.object(sculptor_bridge, "gpu_info", return_value=fake_info):
            got = sculptor_bridge.cuda_version_ok(min_major=major, min_minor=minor)
        assert got is expected, (
            f"cuda_version_ok({cuda_v!r}, >={major}.{minor}) returned {got}, "
            f"expected {expected}"
        )


def test_system_gpu_exposes_m4_pynvml_fields(client: TestClient) -> None:
    """M4 verification #1: /system/gpu returns pynvml-backed live fields
    when nvml is available on the host (RTX 5070 Laptop default path)."""
    r = client.get("/system/gpu")
    body = r.json()
    # Always present.
    for key in ("pynvml_available", "driver_version"):
        assert key in body, f"missing key {key!r}"

    import torch
    if torch.cuda.is_available():
        # On the dev machine with NVIDIA driver, pynvml should work.
        # But mock-friendly: allow False if nvml init fails for any reason.
        assert isinstance(body["pynvml_available"], bool)
        if body["pynvml_available"]:
            dev = body["devices"][0]
            # Live fields populated.
            assert "utilization_percent" in dev
            assert "temperature_c" in dev
            assert "used_memory_bytes" in dev
            # Memory sanity.
            assert dev["total_memory_bytes"] > 0
            assert (
                0 <= (dev["used_memory_bytes"] or 0) <= dev["total_memory_bytes"]
            )


def test_gpu_monitor_cache_is_respected() -> None:
    """Two calls inside the 2s TTL return the same cached payload."""
    from backend.services import gpu_monitor

    gpu_monitor.invalidate_cache()
    a = gpu_monitor.get_live_snapshot()
    b = gpu_monitor.get_live_snapshot()
    assert a is b, "gpu_monitor cache not returning same dict within TTL"


def test_system_gpu_reports_cuda_when_available(client: TestClient) -> None:
    """Shape check the CUDA-available path — only asserts concrete values
    when torch reports CUDA; otherwise asserts the None/0 fallback."""
    r = client.get("/system/gpu")
    body = r.json()

    import torch
    if torch.cuda.is_available():
        assert body["cuda_available"] is True
        assert body["torch_available"] is True
        assert body["device_count"] >= 1
        assert len(body["devices"]) == body["device_count"]
        dev = body["devices"][0]
        for key in (
            "index", "name", "total_memory_bytes", "free_memory_bytes",
            "major", "minor",
        ):
            assert key in dev
        assert dev["total_memory_bytes"] > 0
    else:
        assert body["cuda_available"] is False
        assert body["device_count"] == 0
        assert body["devices"] == []
