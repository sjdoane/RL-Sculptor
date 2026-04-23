"""System / diagnostic endpoints.

GET /system/gpu — torch.cuda status + mjlab / mujoco_warp / rsl_rl
import health. Consumed by the Settings page (MJLAB_PIVOT_DESIGN §3.4).

GET /system/kg/stats — aggregate counts over the shared user-wide KG
(M7 Phase 1). Consumed by the Settings page's "Knowledge graph" card.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.models.system import SystemGpuResponse, SystemKgStatsResponse
from backend.services import kg_store, sculptor_bridge


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/gpu", response_model=SystemGpuResponse)
def get_gpu_info() -> SystemGpuResponse:
    info = sculptor_bridge.gpu_info()
    return SystemGpuResponse(
        torch_available=bool(info.get("torch_available", False)),
        torch_version=info.get("torch_version"),
        cuda_available=bool(info.get("cuda_available", False)),
        cuda_version=info.get("cuda_version"),
        cuda_version_ok=sculptor_bridge.cuda_version_ok(),
        device_count=int(info.get("device_count", 0)),
        devices=info.get("devices", []),
        driver_version=info.get("driver_version"),
        pynvml_available=bool(info.get("pynvml_available", False)),
        mjlab_available=bool(info.get("mjlab_available", False)),
        mujoco_warp_available=bool(info.get("mujoco_warp_available", False)),
        rsl_rl_available=bool(info.get("rsl_rl_available", False)),
        import_error=info.get("import_error"),
    )


@router.get("/kg/stats", response_model=SystemKgStatsResponse)
def get_shared_kg_stats() -> SystemKgStatsResponse:
    """Summary of the user-wide shared KG.

    Returns a zero-filled response when the shared DB does not exist
    yet — callers render "0 papers" rather than surfacing an error.
    """
    shared = kg_store.shared_kg_db_path()
    if not shared.is_file():
        return SystemKgStatsResponse(
            db_path=str(shared),
            db_exists=False,
        )

    last_modified = datetime.fromtimestamp(
        shared.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    try:
        from sculptor.kg.store import SculptorKG

        with SculptorKG(shared) as kg:
            raw = kg.stats()
    except Exception:  # noqa: BLE001 — shared DB may be mid-write
        return SystemKgStatsResponse(
            db_path=str(shared),
            db_exists=True,
            db_size_bytes=shared.stat().st_size,
            last_modified=last_modified,
        )

    nodes = raw.get("nodes_by_kind", {})
    return SystemKgStatsResponse(
        db_path=str(shared),
        db_exists=True,
        db_size_bytes=shared.stat().st_size,
        last_modified=last_modified,
        papers=int(nodes.get("Paper", 0)),
        techniques=int(nodes.get("Technique", 0)),
        failure_modes=int(nodes.get("FailureMode", 0)),
        reward_components=int(nodes.get("RewardComponent", 0)),
        environments=int(nodes.get("Environment", 0)),
        results=int(nodes.get("Result", 0)),
        edges=int(raw.get("total_edges", 0)),
        embeddings=int(raw.get("total_embeddings", 0)),
    )
