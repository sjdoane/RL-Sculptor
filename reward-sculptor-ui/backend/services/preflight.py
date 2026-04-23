"""Live device preflight for mjlab project creation + run launch.

Enhancement over the M2 static-estimate preflight: uses `gpu_monitor.
get_live_snapshot()` (pynvml-backed, 2 s cache) for *actual* free VRAM
at request time, and the post-M2 per-env coefficient cache at
`<project>/.sculptor_cache/vram_coefficients.json` when the project
has already run a VRAM probe.

Returns a `PreflightResult` with a suggested `num_envs` that fits the
current free VRAM — the UI wires a "Retry with suggested num_envs"
button to this value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


log = logging.getLogger("reward-sculptor-ui.preflight")

# Static fallback when no per-env coefficient has been measured yet.
# Matches MJLAB_PIVOT_DESIGN §9 / M2's formula: 1.5 GiB policy + 0.5 MB per env.
_STATIC_POLICY_GIB = 1.5
_STATIC_PER_ENV_BYTES = 0.5 * 1024 * 1024  # 512 KiB
_SAFETY_MULT = 1.2  # 20% headroom on top of measured/estimated.
_VRAM_FREE_BUDGET = 0.85  # fraction of free VRAM we're willing to use.


@dataclass
class PreflightResult:
    ok: bool
    device_index: int
    device_name: str
    free_vram_gb: float
    total_vram_gb: float
    estimated_required_gb: float
    suggested_num_envs: Optional[int] = None
    reason: Optional[str] = None  # non-None iff !ok
    problem_type: Optional[str] = None  # "/problems/..." when !ok


def _load_cached_coefficient(
    project_dir: Optional[Path], task_id: str
) -> Optional[float]:
    """Look up the per-env bytes coefficient in the project's probe
    cache. Returns None if the cache is missing or keyed to a
    different (task_id, mjlab_version)."""
    if project_dir is None:
        return None
    cache = project_dir / ".sculptor_cache" / "vram_coefficients.json"
    if not cache.is_file():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    key = payload.get("_cache_key", "")
    if not isinstance(key, str) or not key.startswith(f"{task_id}|"):
        return None
    coeff = payload.get("coefficient_bytes_per_env")
    try:
        return float(coeff) if coeff is not None else None
    except (TypeError, ValueError):
        return None


def check_mjlab_preflight(
    *,
    task_id: str,
    num_envs: int,
    device: str = "cuda:0",
    project_dir: Optional[Path] = None,
    min_gpu_memory_gb: float = 0.0,
) -> PreflightResult:
    """Check whether (task_id, num_envs, device) can be launched now.

    Uses live free VRAM from pynvml (via gpu_monitor). Returns a
    PreflightResult carrying a suggested_num_envs when the requested
    config doesn't fit.
    """
    # Lazy import — avoid circular import with sculptor_bridge.
    from backend.services import gpu_monitor

    snap = gpu_monitor.get_live_snapshot()
    devices = snap.get("devices") or []
    if not devices:
        return PreflightResult(
            ok=False,
            device_index=0,
            device_name="",
            free_vram_gb=0.0,
            total_vram_gb=0.0,
            estimated_required_gb=0.0,
            reason="No CUDA device detected.",
            problem_type="/problems/gpu-required",
        )

    # Parse device index from "cuda:N" form.
    device_idx = 0
    if device.startswith("cuda") and ":" in device:
        try:
            device_idx = int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            device_idx = 0

    if device_idx >= len(devices):
        return PreflightResult(
            ok=False,
            device_index=device_idx,
            device_name="",
            free_vram_gb=0.0,
            total_vram_gb=0.0,
            estimated_required_gb=0.0,
            reason=(
                f"Requested device cuda:{device_idx} is unavailable — "
                f"only {len(devices)} CUDA device(s) detected."
            ),
            problem_type="/problems/device-unavailable",
        )

    dev = devices[device_idx]
    free_bytes = int(dev.get("free_memory_bytes", 0))
    total_bytes = int(dev.get("total_memory_bytes", 0))
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    # Estimate required VRAM. Prefer the project's cached coefficient
    # (from the VRAM probe at training time); fall back to the static
    # 0.5 MB/env formula.
    cached = _load_cached_coefficient(project_dir, task_id)
    if cached is not None and cached > 0:
        per_env_bytes = cached  # cache already includes the 20% buffer
        estimated_bytes = per_env_bytes * num_envs
        # Add the policy overhead separately (~1.5 GiB).
        estimated_bytes += _STATIC_POLICY_GIB * (1024 ** 3)
    else:
        per_env_bytes = _STATIC_PER_ENV_BYTES
        estimated_bytes = (
            (_STATIC_POLICY_GIB * (1024 ** 3))
            + per_env_bytes * num_envs
        ) * _SAFETY_MULT

    estimated_gb = estimated_bytes / (1024 ** 3)

    # Hard floor from the reward contract.
    min_required_gb = max(estimated_gb, float(min_gpu_memory_gb or 0.0))
    budget_gb = free_gb * _VRAM_FREE_BUDGET

    if min_required_gb <= budget_gb:
        return PreflightResult(
            ok=True,
            device_index=device_idx,
            device_name=dev.get("name", ""),
            free_vram_gb=free_gb,
            total_vram_gb=total_gb,
            estimated_required_gb=min_required_gb,
            suggested_num_envs=num_envs,
        )

    # Doesn't fit — compute suggested num_envs that does.
    headroom_bytes = max(0.0, budget_gb * (1024 ** 3) - _STATIC_POLICY_GIB * (1024 ** 3))
    if per_env_bytes > 0:
        suggested_envs = int(headroom_bytes / per_env_bytes)
        # Snap to nearest lower power-of-two for clean recommendation.
        suggested = 128
        while suggested * 2 <= suggested_envs:
            suggested *= 2
        suggested = max(128, min(4096, suggested))
    else:
        suggested = 128
    if suggested >= num_envs:
        suggested = max(128, num_envs // 2)

    return PreflightResult(
        ok=False,
        device_index=device_idx,
        device_name=dev.get("name", ""),
        free_vram_gb=free_gb,
        total_vram_gb=total_gb,
        estimated_required_gb=min_required_gb,
        suggested_num_envs=suggested,
        reason=(
            f"Requested {num_envs} envs would need ~{min_required_gb:.1f} GiB; "
            f"only {budget_gb:.1f} GiB free ({free_gb:.1f} GiB × 85%) on "
            f"{dev.get('name', 'GPU')}."
        ),
        problem_type="/problems/insufficient-vram",
    )
