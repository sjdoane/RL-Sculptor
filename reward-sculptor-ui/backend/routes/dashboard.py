"""Aggregate dashboard endpoint.

`GET /api/dashboard` returns everything the `/` landing page needs in a
single round-trip, so the cold-load budget (Prompt 10 R3: < 500 ms) is
spent on one request rather than the ~5 endpoints the page would
otherwise fan out to.

Also exposes `GET /api/system/info` for the Settings page:
API-key status (masked), torch/CUDA detection, resolved paths.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.models.kg import JobSummary
from backend.services import kg_store, sculptor_bridge
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["dashboard"])


# ── shapes ────────────────────────────────────────────────────────────
class RecentRunRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    project_slug: str
    project_display_name: str
    status: str
    behavior_goal: str
    iterations_completed: int
    iterations_requested: int
    primary_metric_history: list[Optional[float]]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]


class KGAdditionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_slug: str
    project_display_name: str
    arxiv_id: str
    title: str
    extracted: bool
    ingested_at: Optional[datetime] = None


class DashboardSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_count: int
    has_any_projects: bool
    active_jobs: list[JobSummary]
    recent_runs: list[RecentRunRow]
    recent_kg_additions: list[KGAdditionRow]


class SystemInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sculptor_ok: bool
    sculptor_error: Optional[str] = None
    projects_root: str
    sculptor_path: Optional[str] = None
    anthropic_api_key_set: bool
    anthropic_api_key_masked: Optional[str] = None
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: list[str] = Field(default_factory=list)


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── /dashboard ────────────────────────────────────────────────────────
@router.get("/dashboard", response_model=DashboardSummary)
def get_dashboard(
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> DashboardSummary:
    projects = store.list()
    display_by_slug = {p.slug: p.display_name for p in projects}

    # Active jobs across all projects (any running / queued).
    active = [
        j.to_summary()
        for j in jobs.list()
        if j.status in ("running", "queued")
    ]

    # Recent runs: up to 10 most-recent sculpt_run jobs. Include both
    # active and terminal states so the dashboard reflects current
    # history accurately.
    sculpt_jobs = jobs.list(kind="sculpt_run")
    recent: list[RecentRunRow] = []
    for job in sculpt_jobs[:10]:
        params = job.params or {}
        slug = job.project_slug or ""
        history_raw = params.get("primary_metric_history") or []
        history: list[Optional[float]] = []
        if isinstance(history_raw, list):
            for v in history_raw:
                if isinstance(v, (int, float)):
                    history.append(float(v))
                else:
                    history.append(None)
        # Derive iterations_completed from iter_completed events to
        # stay consistent with the Runs-tab view.
        iters_completed = sum(
            1 for ev in job.events if ev.get("type") == "iter_completed"
        )
        recent.append(
            RecentRunRow(
                run_id=job.job_id,
                project_slug=slug,
                project_display_name=display_by_slug.get(slug, slug),
                status=job.status,
                behavior_goal=str(params.get("behavior_goal") or ""),
                iterations_completed=iters_completed,
                iterations_requested=int(params.get("iterations") or 0),
                primary_metric_history=history,
                started_at=job.started_at,
                ended_at=job.ended_at,
            )
        )

    # Recent KG additions: per-project, grab the newest-ingested Papers
    # and flatten across the top 3 projects by created_at. This is a
    # best-effort view — if a project's KG DB is missing we skip it.
    kg_additions: list[KGAdditionRow] = []
    for p in projects[:5]:
        try:
            from pathlib import Path
            papers = kg_store.list_papers(Path(p.project_dir))
        except Exception:  # noqa: BLE001
            continue
        for paper in papers[:3]:
            kg_additions.append(
                KGAdditionRow(
                    project_slug=p.slug,
                    project_display_name=p.display_name,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    extracted=paper.extracted,
                    ingested_at=None,
                )
            )
    # Cap to 10 most-recent-looking entries. We don't have per-paper
    # timestamps readily so we stop at 10 in iteration order.
    kg_additions = kg_additions[:10]

    return DashboardSummary(
        project_count=len(projects),
        has_any_projects=len(projects) > 0,
        active_jobs=active,
        recent_runs=recent,
        recent_kg_additions=kg_additions,
    )


# ── /system/info ──────────────────────────────────────────────────────
@router.get("/system/info", response_model=SystemInfo)
def get_system_info(request: Request) -> SystemInfo:
    settings = request.app.state.settings
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    api_key_masked = _mask_key(api_key) if api_key else None

    sculptor_path: Optional[str] = None
    if sculptor_bridge.sculptor_ok():
        try:
            import sculptor as _s  # type: ignore[import-untyped]
            sculptor_path = str(_s.__file__)
        except Exception:  # noqa: BLE001
            sculptor_path = None

    torch_available = False
    cuda_available = False
    cuda_device_count = 0
    cuda_device_names: list[str] = []
    try:
        import torch  # type: ignore[import-untyped]

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_count = int(torch.cuda.device_count())
            cuda_device_names = [
                str(torch.cuda.get_device_name(i))
                for i in range(cuda_device_count)
            ]
    except Exception:  # noqa: BLE001
        pass

    return SystemInfo(
        sculptor_ok=sculptor_bridge.sculptor_ok(),
        sculptor_error=sculptor_bridge.sculptor_error(),
        projects_root=str(settings.resolved_projects_root),
        sculptor_path=sculptor_path,
        anthropic_api_key_set=bool(api_key),
        anthropic_api_key_masked=api_key_masked,
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
    )


def _mask_key(key: str) -> str:
    """Last 4 chars visible, rest redacted. `sk-ant-...abcd`."""
    if len(key) <= 4:
        return "*" * len(key)
    visible = key[-4:]
    return f"{'*' * min(len(key) - 4, 12)}{visible}"
