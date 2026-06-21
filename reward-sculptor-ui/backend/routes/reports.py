"""Report endpoints.

  GET  /projects/{slug}/reports/final_report.md   — markdown text
  GET  /projects/{slug}/reports/final.mp4         — timelapse mp4
  GET  /projects/{slug}/reports/mission-quality   — §Ship 25b telemetry
  POST /projects/{slug}/reports/build             — run `sculptor.timelapse.build_report`

The build endpoint is synchronous today — `build_report` completes in a
few seconds for typical 3-10 iter runs. If it grows to minute-scale we
should move it behind JobManager. For now we keep the flow simple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from backend.models.project import ProblemDetail
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["reports"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def _problem(code: int, title: str, **extra: Any) -> JSONResponse:
    body = ProblemDetail(title=title, status=code, **{
        k: v for k, v in extra.items() if k in {"detail", "type", "instance"}
    }).model_dump()
    body.update({k: v for k, v in extra.items() if k not in body})
    return JSONResponse(status_code=code, content=body, media_type="application/problem+json")


def _project_dir(store: ProjectStore, slug: str) -> Path | None:
    d = store.get(slug)
    return Path(d.project_dir) if d else None


# ── GET final_report.md ───────────────────────────────────────────────
@router.get(
    "/projects/{slug}/reports/final_report.md",
    response_class=PlainTextResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_final_report(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")
    path = pd / "reports" / "final_report.md"
    if not path.is_file():
        # §7.6: sharpen the "no report" message so Sam can tell apart
        # (a) project with zero completed iters vs (b) project with
        # iters but report never built. Both return 404 but the detail
        # tells the frontend what to surface.
        runs_dir = pd / "runs"
        n_completed = 0
        if runs_dir.is_dir():
            for d in runs_dir.iterdir():
                if d.is_dir() and (d / "diagnosis.json").is_file():
                    n_completed += 1
        if n_completed == 0:
            detail = (
                "No completed sculpt iters yet. Run one before building a "
                "report."
            )
        else:
            detail = (
                f"No final_report.md (project has {n_completed} completed "
                f"iter{'s' if n_completed != 1 else ''} on disk). POST to "
                f"/projects/{{slug}}/reports/build to build it."
            )
        return _problem(
            404, "report not built",
            detail=detail,
            type="/problems/not-found",
            n_completed_iters=n_completed,
        )
    return PlainTextResponse(
        path.read_text(encoding="utf-8"), media_type="text/markdown"
    )


# ── GET final.mp4 ─────────────────────────────────────────────────────
@router.get(
    "/projects/{slug}/reports/final.mp4",
    response_class=FileResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_final_mp4(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")
    path = pd / "reports" / "final.mp4"
    if not path.is_file():
        return _problem(
            404, "timelapse not built",
            detail="No final.mp4 yet. Build the report first.",
            type="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")


# ── GET mission-quality (§Ship 25b / H2) ──────────────────────────────
@router.get(
    "/projects/{slug}/reports/mission-quality",
    responses={404: {"model": ProblemDetail}},
)
def get_mission_quality(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    """Decomposition-quality telemetry written by sculpt's mission
    orchestrator (`reports/mission_quality.json`): per-mission stage
    counts, stage-success rate, redecompositions, iteration spend.
    Empty list (not 404) when no mission has run — the card renders an
    empty state."""
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")
    path = pd / "reports" / "mission_quality.json"
    if not path.is_file():
        return {"schema": 1, "missions": []}
    try:
        import json

        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("missions"), list):
            raise ValueError("bad shape")
    except Exception:  # noqa: BLE001 — corrupt file → empty, never 500
        return {"schema": 1, "missions": []}
    return doc


# ── GET actuator-limits (§reports: per-motor torque/speed vs limits) ──
@router.get(
    "/projects/{slug}/reports/actuator-limits",
    responses={404: {"model": ProblemDetail}},
)
def get_actuator_limits(
    slug: str, iter: int | None = None, store: ProjectStore = Depends(get_store)
) -> Any:
    """Per-motor SPEED-vs-no-load-speed and TORQUE-vs-effort-limit utilization for
    one rollout iteration — the charts that visually confirm a policy respects the
    real actuator envelope. `iter` selects the iteration (default: the latest with
    rollout data). Empty state (not 404) when no rollout has trajectory data yet."""
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")
    runs = pd / "runs"
    avail: list[int] = []
    if runs.is_dir():
        for d in sorted(runs.glob("iter_*")):
            if (d / "rollout" / "trajectory.npz").is_file():
                try:
                    avail.append(int(d.name.split("_")[1]))
                except (ValueError, IndexError):
                    continue
    avail.sort()
    if not avail:
        return {"schema": 1, "available_iters": [], "iter": None, "ok": False,
                "has_torque": False,
                "reason": "no rollouts with trajectory data yet", "motors": []}
    sel = iter if (iter is not None and iter in avail) else avail[-1]
    rd = runs / f"iter_{sel}" / "rollout"
    try:
        from backend.services import sculptor_bridge

        rep = sculptor_bridge.actuator_limits_report(
            rd / "trajectory.npz", rd / "mjcf_limits.json")
    except Exception as e:  # noqa: BLE001 — never 500; surface an empty state
        rep = {"ok": False, "has_torque": False,
               "reason": f"{type(e).__name__}: {e}", "motors": []}
    rep.update({"schema": 1, "available_iters": avail, "iter": sel})
    return rep


# ── POST build ────────────────────────────────────────────────────────
@router.post(
    "/projects/{slug}/reports/build",
    responses={
        200: {"description": "Report built, see `final_report_md_path` / `final_mp4_path`."},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
    },
)
def build_report(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")

    config_path = pd / "config.toml"
    if not config_path.is_file():
        return _problem(
            409, "project config missing",
            detail=f"{config_path} does not exist — project may be corrupt.",
            type="/problems/state-conflict",
        )

    try:
        from sculptor.timelapse import build_report as _build_report  # type: ignore[import-untyped]
    except Exception as e:  # noqa: BLE001
        return _problem(
            500, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable",
        )

    try:
        result = _build_report(
            config_path=config_path,
            out_mp4=pd / "reports" / "final.mp4",
        )
    except Exception as e:  # noqa: BLE001
        return _problem(
            500, "report build failed",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/preview-failed",
        )

    return {
        "ok": True,
        "final_report_md_path": str(getattr(result, "final_report_md_path", "")),
        "final_mp4_path": str(getattr(result, "final_mp4_path", "")),
        "final_mp4_ok": bool(getattr(result, "final_mp4_ok", False)),
        "selected_iter_indices": list(
            getattr(result, "selected_iter_indices", []) or []
        ),
    }
