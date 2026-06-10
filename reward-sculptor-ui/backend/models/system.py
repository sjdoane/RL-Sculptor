"""Pydantic shapes for GET /system/gpu and related system-info endpoints.

MJLAB_PIVOT_DESIGN §3.4 — Settings page consumes this to render a GPU
panel with CUDA-version / VRAM / mjlab-import health.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class GpuDevice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    used_memory_bytes: Optional[int] = None
    utilization_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    multi_processor_count: int = 0
    major: int = 0
    minor: int = 0


class SystemGpuResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    torch_available: bool
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cuda_version_ok: bool = False
    device_count: int = 0
    devices: list[GpuDevice] = []
    driver_version: Optional[str] = None
    pynvml_available: bool = False
    mjlab_available: bool = False
    mujoco_warp_available: bool = False
    rsl_rl_available: bool = False
    import_error: Optional[str] = None


class RemoteDoctorCheck(BaseModel):
    """One row of `sculpt remote doctor` output (§Ship 23d)."""

    model_config = ConfigDict(extra="allow")

    name: str
    ok: bool
    detail: str = ""


class RemoteDoctorResponse(BaseModel):
    """POST /system/remote/doctor — connectivity report for the
    Settings page's Test-connection button."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    host: str = ""
    port: int = 22
    checks: list[RemoteDoctorCheck] = []


class SystemKgStatsResponse(BaseModel):
    """GET /system/kg/stats — aggregate counts over the shared KG."""

    model_config = ConfigDict(extra="allow")

    db_path: str
    db_exists: bool
    db_size_bytes: int = 0
    last_modified: Optional[str] = None  # ISO8601 UTC, or None if absent
    papers: int = 0
    techniques: int = 0
    failure_modes: int = 0
    reward_components: int = 0
    environments: int = 0
    results: int = 0
    edges: int = 0
    embeddings: int = 0
