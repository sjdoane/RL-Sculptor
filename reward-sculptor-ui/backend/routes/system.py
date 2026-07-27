"""System / diagnostic endpoints.

GET /system/gpu — torch.cuda status + mjlab / mujoco_warp / rsl_rl
import health. Consumed by the Settings page (MJLAB_PIVOT_DESIGN §3.4).

GET /system/kg/stats — aggregate counts over the shared user-wide KG
(M7 Phase 1). Consumed by the Settings page's "Knowledge graph" card.

GET/PUT /system/remote + POST /system/remote/doctor (§Ship 23d) —
persisted remote-GPU dispatch settings (`<projects_root>/_settings/
remote.json`) and the Test-connection report for the Settings page.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from backend.models.system import (
    RemoteDoctorResponse,
    SystemGpuResponse,
    SystemKgStatsResponse,
)
from backend.services import kg_store, sculptor_bridge
from backend.services.api_key_store import mask_key, save_key
from backend.services.remote_settings import (
    RemoteSettings,
    load_remote_settings,
    run_doctor,
    save_remote_settings,
)


router = APIRouter(prefix="/system", tags=["system"])


class ApiKeyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr = Field(min_length=20, max_length=512)


class ApiKeyStatus(BaseModel):
    configured: bool
    masked: str | None = None
    persisted: bool = False


@router.put("/api-key", response_model=ApiKeyStatus)
def put_api_key(request: Request, body: ApiKeyUpdate) -> ApiKeyStatus:
    """Save a localhost-only API key without ever echoing it back."""
    root = request.app.state.settings.resolved_projects_root
    value = body.api_key.get_secret_value()
    try:
        save_key(root, value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return ApiKeyStatus(
        configured=True,
        masked=mask_key(value.strip()),
        persisted=True,
    )


@router.get("/remote", response_model=RemoteSettings)
def get_remote_settings(request: Request) -> RemoteSettings:
    root = request.app.state.settings.resolved_projects_root
    return load_remote_settings(root)


@router.put("/remote", response_model=RemoteSettings)
def put_remote_settings(request: Request, body: RemoteSettings) -> RemoteSettings:
    root = request.app.state.settings.resolved_projects_root
    save_remote_settings(root, body)
    return load_remote_settings(root)


@router.post("/remote/doctor", response_model=RemoteDoctorResponse)
async def post_remote_doctor(request: Request) -> RemoteDoctorResponse:
    """Run connectivity checks against the SAVED settings (PUT first).
    Blocking ssh round-trips (~30 s worst case on an unreachable host)
    run in a worker thread so the event loop stays responsive."""
    root = request.app.state.settings.resolved_projects_root
    settings = load_remote_settings(root)
    report = await asyncio.to_thread(run_doctor, settings)
    return RemoteDoctorResponse.model_validate(report)


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
        run_cases=int(nodes.get("RunCase", 0)),
        edges=int(raw.get("total_edges", 0)),
        embeddings=int(raw.get("total_embeddings", 0)),
    )
