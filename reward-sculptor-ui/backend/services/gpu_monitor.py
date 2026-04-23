"""Live GPU telemetry via pynvml with a torch-based fallback.

Extends the M2 stub `sculptor_bridge.gpu_info()` with:
  - per-device utilization percent (via nvmlDeviceGetUtilizationRates)
  - per-device temperature C (via nvmlDeviceGetTemperature, when supported)
  - driver version (via nvmlSystemGetDriverVersion)

Cached for 2 s — polling endpoints (Settings GPU panel, Dashboard widget)
can query every 2-5 s without thrashing nvml. Cache is process-local.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional


log = logging.getLogger("reward-sculptor-ui.gpu_monitor")


_CACHE_TTL_S = 2.0
_pynvml_init_error: Optional[str] = None
_pynvml_initialized = False
_init_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_ts: float = 0.0
_cache_lock = threading.Lock()


def _ensure_pynvml() -> bool:
    """Best-effort pynvml init. Safe to call repeatedly."""
    global _pynvml_init_error, _pynvml_initialized
    if _pynvml_initialized:
        return True
    if _pynvml_init_error is not None:
        return False
    with _init_lock:
        if _pynvml_initialized:
            return True
        if _pynvml_init_error is not None:
            return False
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            _pynvml_initialized = True
            return True
        except Exception as e:  # noqa: BLE001
            _pynvml_init_error = f"{type(e).__name__}: {e}"
            log.info("pynvml init failed: %s", _pynvml_init_error)
            return False


def _pynvml_snapshot() -> dict[str, Any]:
    """Collect a live GPU snapshot via pynvml. Raises on any nvml error
    — caller should fall back to torch-only info."""
    import pynvml  # type: ignore

    driver_version: Optional[str] = None
    try:
        v = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        driver_version = str(v)
    except Exception:  # noqa: BLE001
        pass

    try:
        nvml_version = pynvml.nvmlSystemGetNVMLVersion()
        if isinstance(nvml_version, bytes):
            nvml_version = nvml_version.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        nvml_version = None

    device_count = int(pynvml.nvmlDeviceGetCount())
    devices: list[dict[str, Any]] = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        util_percent: Optional[float] = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_percent = float(util.gpu)
        except Exception:  # noqa: BLE001
            pass

        temp_c: Optional[float] = None
        try:
            temp_c = float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception:  # noqa: BLE001
            pass

        # Compute capability (major.minor) — torch exposes these too,
        # but we prefer a single-source-of-truth from nvml when possible.
        major: Optional[int] = None
        minor: Optional[int] = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            major, minor = int(major), int(minor)
        except Exception:  # noqa: BLE001
            pass

        devices.append({
            "index": i,
            "name": str(name),
            "total_memory_bytes": int(mem.total),
            "free_memory_bytes": int(mem.free),
            "used_memory_bytes": int(mem.used),
            "utilization_percent": util_percent,
            "temperature_c": temp_c,
            "multi_processor_count": 0,  # populated below via torch if available
            "major": major or 0,
            "minor": minor or 0,
        })

    return {
        "pynvml_available": True,
        "driver_version": driver_version,
        "nvml_version": nvml_version,
        "device_count": device_count,
        "devices": devices,
    }


def _torch_augment(info: dict[str, Any]) -> None:
    """Fill in multi_processor_count from torch (nvml doesn't expose it
    directly). Mutates `info["devices"]` in place."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    for d in info.get("devices") or []:
        try:
            props = torch.cuda.get_device_properties(d["index"])
            d["multi_processor_count"] = int(props.multi_processor_count)
            if not d.get("major"):
                d["major"] = int(props.major)
            if not d.get("minor"):
                d["minor"] = int(props.minor)
        except Exception:  # noqa: BLE001
            pass


def get_live_snapshot() -> dict[str, Any]:
    """2 s-cached GPU snapshot. Returns a dict with keys:
    pynvml_available, driver_version, nvml_version (may be null),
    device_count, devices[{index, name, total/free/used_memory_bytes,
    utilization_percent, temperature_c, multi_processor_count, major,
    minor}].
    """
    global _cache, _cache_ts

    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and now - _cache_ts < _CACHE_TTL_S:
            return _cache

    info: dict[str, Any]
    if _ensure_pynvml():
        try:
            info = _pynvml_snapshot()
        except Exception as e:  # noqa: BLE001
            log.warning("pynvml snapshot failed: %s", e)
            info = _torch_fallback_snapshot()
    else:
        info = _torch_fallback_snapshot()

    _torch_augment(info)

    with _cache_lock:
        _cache = info
        _cache_ts = now
    return info


def _torch_fallback_snapshot() -> dict[str, Any]:
    """Minimal snapshot using torch.cuda only — no live utilization, no
    temperature. Matches the M2 sculptor_bridge.gpu_info() device shape."""
    out: dict[str, Any] = {
        "pynvml_available": False,
        "driver_version": None,
        "nvml_version": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch
    except ImportError:
        return out
    if not torch.cuda.is_available():
        return out
    out["device_count"] = int(torch.cuda.device_count())
    for i in range(out["device_count"]):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            free, total = 0, int(props.total_memory)
        out["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "free_memory_bytes": int(free),
            "used_memory_bytes": int(props.total_memory - free),
            "utilization_percent": None,
            "temperature_c": None,
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
        })
    return out


def invalidate_cache() -> None:
    """Test utility — forces next `get_live_snapshot` to re-query nvml."""
    global _cache, _cache_ts
    with _cache_lock:
        _cache = None
        _cache_ts = 0.0
