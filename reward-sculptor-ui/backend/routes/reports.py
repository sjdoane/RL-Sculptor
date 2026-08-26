"""Report endpoints.

  GET  /projects/{slug}/reports/final_report.md   — markdown text
  GET  /projects/{slug}/reports/final.mp4         — timelapse mp4
  GET  /projects/{slug}/reports/mission-quality   — §Ship 25b telemetry
  GET  /projects/{slug}/reports/sources           — §chunk C1: what can be
                                                     built into a report
                                                     (project runs + every
                                                     mission), for the
                                                     frontend's source picker
  POST /projects/{slug}/reports/build             — run `sculptor.timelapse.
                                                     build_report` (legacy,
                                                     no body / empty body) or
                                                     `build_mission_report`
                                                     (body `{"mission_slug"}`)

  GET  /projects/{slug}/missions/{mission_slug}/report/final_report.md
  GET  /projects/{slug}/missions/{mission_slug}/report/final.mp4
                                                   — §chunk C1: mission-level
                                                     report artifacts, mirror
                                                     of the two routes above.

The build endpoint is synchronous today — `build_report` completes in a
few seconds for typical 3-10 iter runs. If it grows to minute-scale we
should move it behind JobManager. For now we keep the flow simple.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from backend.models.project import ProblemDetail
from backend.routes.policies import report_selection_authority
from backend.services import mission_store
from backend.services.iteration_completion import iteration_completion_authority
from backend.services.project_store import ProjectStore


router = APIRouter(tags=["reports"])

# §chunk C1: mirrors routes/runs.py's `_SAFE_PATH_SEGMENT` — mission_slug
# (and any other segment we splice into a filesystem path) must be a
# snake_case-ish identifier before it's used to build a FileResponse path.
_SAFE_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BuildReportRequest(BaseModel):
    """POST /projects/{slug}/reports/build body (optional).

    Absent (or an empty `{}` body) → legacy project-runs report via
    `sculptor.timelapse.build_report`, unchanged. `mission_slug` set →
    mission-aware report via `sculptor.timelapse.build_mission_report`.
    """

    model_config = ConfigDict(extra="forbid")

    mission_slug: Optional[str] = None


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


def _count_attested_iterations(runs_dir: Path) -> int:
    if runs_dir.is_symlink() or not runs_dir.is_dir():
        return 0
    return sum(
        1 for iter_dir in runs_dir.iterdir()
        if iter_dir.is_dir() and not iter_dir.is_symlink()
        and iteration_completion_authority(iter_dir) == "attested"
    )


def _report_state(source_root: Path, *, source_kind: str) -> dict[str, Any]:
    try:
        from sculptor.timelapse import inspect_report_state

        return inspect_report_state(source_root, source_kind=source_kind)
    except Exception as exc:  # noqa: BLE001 - report authority fails closed
        return {
            "state": "stale",
            "reason": f"report receipt could not be verified: {type(exc).__name__}",
            "claim_status": "unavailable",
            "selected_iter_index": None,
        }


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
        n_completed = _count_attested_iterations(runs_dir)
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
    report_state = _report_state(pd, source_kind="project")
    markdown = path.read_text(encoding="utf-8")
    if report_state["state"] == "stale":
        markdown = (
            "> **STALE RETAINED REPORT — not current evidence.** "
            f"{report_state.get('reason') or 'Inputs changed.'}\n\n"
            + markdown
        )
    return PlainTextResponse(
        markdown,
        media_type="text/markdown",
        headers={
            "X-RewardSculptor-Report-State": str(report_state["state"]),
            "X-RewardSculptor-Claim-Status": str(
                report_state.get("claim_status", "unavailable")
            ),
        },
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
    report_state = _report_state(pd, source_kind="project")
    if report_state["state"] != "current":
        return _problem(
            409,
            "timelapse report is stale",
            detail=str(report_state.get("reason") or "report inputs changed"),
            type="/problems/stale-report",
        )
    return FileResponse(path, media_type="video/mp4")


# ── §chunk C1: mission report artifacts ────────────────────────────────
def _mission_dir_checked(
    store: ProjectStore, slug: str, mission_slug: str,
) -> tuple[Optional[Path], Optional[JSONResponse]]:
    """Resolve `<project_dir>/.missions/<mission_slug>/`, validating both
    the project slug and the mission slug. Returns `(mission_dir, None)`
    on success or `(None, problem_response)` on any failure — the 404
    shapes mirror `get_final_report`/`get_final_mp4`'s "project not
    found" / "not found" pattern, plus a traversal guard (mirrors
    routes/runs.py's `_SAFE_PATH_SEGMENT` discipline) before the slug
    is used to build a filesystem path."""
    pd = _project_dir(store, slug)
    if pd is None:
        return None, _problem(404, "project not found", type="/problems/not-found")
    if not _SAFE_PATH_SEGMENT.match(mission_slug):
        return None, _problem(
            404, "mission not found",
            detail=f"mission_slug={mission_slug!r} is not a valid slug.",
            type="/problems/not-found",
        )
    if mission_slug not in mission_store.list_mission_slugs(pd):
        return None, _problem(
            404, "mission not found",
            detail=f"no mission {mission_slug!r} under project {slug!r}.",
            type="/problems/not-found",
        )
    return mission_store.mission_dir(pd, mission_slug), None


@router.get(
    "/projects/{slug}/missions/{mission_slug}/report/final_report.md",
    response_class=PlainTextResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_mission_final_report(
    slug: str, mission_slug: str, store: ProjectStore = Depends(get_store),
) -> Any:
    md, problem = _mission_dir_checked(store, slug, mission_slug)
    if problem is not None:
        return problem
    path = md / "reports" / "final_report.md"
    if not path.is_file():
        # Mirror get_final_report's n_completed_iters signal, scoped to
        # however many stages this mission has scaffolded runs for.
        n_completed = 0
        stages_dir = md / "stages"
        if stages_dir.is_dir():
            for stage_dir in stages_dir.iterdir():
                runs_dir = stage_dir / "runs"
                if not runs_dir.is_dir():
                    continue
                n_completed += _count_attested_iterations(runs_dir)
        if n_completed == 0:
            detail = (
                "No completed stage iters yet. Run the mission before "
                "building a report."
            )
        else:
            detail = (
                f"No final_report.md (mission has {n_completed} completed "
                f"stage iter{'s' if n_completed != 1 else ''} on disk). "
                f"POST {{'mission_slug': {mission_slug!r}}} to "
                f"/projects/{{slug}}/reports/build to build it."
            )
        return _problem(
            404, "report not built",
            detail=detail,
            type="/problems/not-found",
            n_completed_iters=n_completed,
        )
    report_state = _report_state(md, source_kind="mission")
    markdown = path.read_text(encoding="utf-8")
    if report_state["state"] == "stale":
        markdown = (
            "> **STALE RETAINED REPORT — not current evidence.** "
            f"{report_state.get('reason') or 'Inputs changed.'}\n\n"
            + markdown
        )
    return PlainTextResponse(
        markdown,
        media_type="text/markdown",
        headers={
            "X-RewardSculptor-Report-State": str(report_state["state"]),
            "X-RewardSculptor-Claim-Status": str(
                report_state.get("claim_status", "unavailable")
            ),
        },
    )


@router.get(
    "/projects/{slug}/missions/{mission_slug}/report/final.mp4",
    response_class=FileResponse,
    responses={404: {"model": ProblemDetail}},
)
def get_mission_final_mp4(
    slug: str, mission_slug: str, store: ProjectStore = Depends(get_store),
) -> Any:
    md, problem = _mission_dir_checked(store, slug, mission_slug)
    if problem is not None:
        return problem
    path = md / "reports" / "final.mp4"
    if not path.is_file():
        return _problem(
            404, "timelapse not built",
            detail="No final.mp4 yet. Build the mission report first.",
            type="/problems/not-found",
        )
    report_state = _report_state(md, source_kind="mission")
    if report_state["state"] != "current":
        return _problem(
            409,
            "timelapse report is stale",
            detail=str(report_state.get("reason") or "report inputs changed"),
            type="/problems/stale-report",
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


def _actuator_limits_from_runs(runs: Path, iter: int | None) -> Any:
    """Shared computation: given a `runs/` dir (either a project's own
    `<project_dir>/runs` or a mission stage's `<mission_dir>/stages/
    <stage>/runs`), find iterations with rollout trajectory data and
    build the per-motor speed/torque utilization report for the
    selected (or latest) one. Never raises — degrades to an empty/ok:
    False state on any error, same contract as the project-scoped path
    had before this helper was factored out."""
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


# ── GET actuator-limits (§reports: per-motor torque/speed vs limits) ──
@router.get(
    "/projects/{slug}/reports/actuator-limits",
    responses={404: {"model": ProblemDetail}},
)
def get_actuator_limits(
    slug: str,
    iter: int | None = None,
    mission_slug: str | None = None,
    stage: str | None = None,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Per-motor SPEED-vs-no-load-speed and TORQUE-vs-effort-limit utilization for
    one rollout iteration — the charts that visually confirm a policy respects the
    real actuator envelope. `iter` selects the iteration (default: the latest with
    rollout data). Empty state (not 404) when no rollout has trajectory data yet.

    When `mission_slug` AND `stage` are BOTH given, the report is built from
    that stage's own runs tree (`<mission_dir>/stages/<stage>/runs`) instead
    of the project-level `runs/` — same response shape, mission-scoped data.
    Project-scoped behavior (both params omitted) is unchanged. Either param
    alone is ignored (falls back to project scope) since a lone `stage` name
    is ambiguous across missions."""
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")

    if mission_slug is not None and stage is not None:
        if not _SAFE_PATH_SEGMENT.match(mission_slug) or not _SAFE_PATH_SEGMENT.match(stage):
            return _problem(
                404, "invalid path segment",
                detail=(
                    f"mission_slug={mission_slug!r} / stage={stage!r} must "
                    "each match a plain slug component"
                ),
                type="/problems/not-found",
            )
        if mission_slug not in mission_store.list_mission_slugs(pd):
            return _problem(
                404, "mission not found",
                detail=f"no mission {mission_slug!r} under project {slug!r}",
                type="/problems/not-found",
            )
        stage_dir = mission_store.mission_dir(pd, mission_slug) / "stages" / stage
        if not stage_dir.is_dir():
            return _problem(
                404, "stage not found",
                detail=f"no stage {stage!r} under mission {mission_slug!r}",
                type="/problems/not-found",
            )
        return _actuator_limits_from_runs(stage_dir / "runs", iter)

    return _actuator_limits_from_runs(pd / "runs", iter)


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
def build_report(
    slug: str,
    store: ProjectStore = Depends(get_store),
    body: Optional[BuildReportRequest] = None,
) -> Any:
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")

    mission_slug = body.mission_slug if body is not None else None
    if mission_slug is not None:
        return _build_mission_report_response(pd, slug, mission_slug)
    return _build_project_report_response(pd)


def _build_project_report_response(pd: Path) -> Any:
    """Build a byte-receipted project report with fail-closed claims."""
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
        selection_authority = report_selection_authority(pd)
        result = _build_report(
            config_path=config_path,
            out_mp4=pd / "reports" / "final.mp4",
            selection_authority=selection_authority,
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
        "report_state": _report_state(pd, source_kind="project"),
        "claim_status": getattr(
            result, "report_claim_status", "descriptive_only"
        ),
    }


def _build_mission_report_response(pd: Path, slug: str, mission_slug: str) -> Any:
    """§chunk C1: mission-aware report path. Validates `mission_slug`
    against both the traversal guard AND the on-disk mission list
    before touching the filesystem, mirroring `routes/missions.py`'s
    404 shape for an unknown slug."""
    if not _SAFE_PATH_SEGMENT.match(mission_slug):
        return _problem(
            404, "mission not found",
            detail=f"mission_slug={mission_slug!r} is not a valid slug.",
            type="/problems/not-found",
        )
    if mission_slug not in mission_store.list_mission_slugs(pd):
        return _problem(
            404, "mission not found",
            detail=(
                f"no mission {mission_slug!r} under this project. "
                f"Available: {mission_store.list_mission_slugs(pd)}"
            ),
            type="/problems/not-found",
        )

    mission_dir_path = mission_store.mission_dir(pd, mission_slug)
    if not (mission_dir_path / "mission.json").is_file():
        return _problem(
            409, "mission not decomposed yet",
            detail=(
                f"mission {mission_slug!r} has no mission.json (decompose "
                f"still in flight or never completed) — nothing to report."
            ),
            type="/problems/state-conflict",
        )

    try:
        from sculptor.timelapse import (  # type: ignore[import-untyped]
            build_mission_report as _build_mission_report,
        )
    except Exception as e:  # noqa: BLE001
        return _problem(
            500, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable",
        )

    try:
        result = _build_mission_report(
            mission_dir=mission_dir_path,
            out_mp4=mission_dir_path / "reports" / "final.mp4",
        )
    except Exception as e:  # noqa: BLE001
        return _problem(
            500, "report build failed",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/preview-failed",
        )

    return {
        "ok": True,
        "mission_slug": mission_slug,
        "final_report_md_path": str(getattr(result, "final_report_md_path", "")),
        "final_mp4_path": str(getattr(result, "final_mp4_path", "")),
        "final_mp4_ok": bool(getattr(result, "final_mp4_ok", False)),
        "selected_iter_indices": list(
            getattr(result, "selected_iter_indices", []) or []
        ),
        "report_state": _report_state(
            mission_dir_path, source_kind="mission"
        ),
        "claim_status": getattr(
            result, "report_claim_status", "descriptive_only"
        ),
    }


# ── GET sources (§chunk C1: report-source picker) ──────────────────────
@router.get(
    "/projects/{slug}/reports/sources",
    responses={404: {"model": ProblemDetail}},
)
def get_report_sources(slug: str, store: ProjectStore = Depends(get_store)) -> Any:
    """Everything the frontend's Results-tab source picker needs in one
    call: the project-level runs report state, plus one entry per
    mission (goal, lifecycle, whether it already has a built report).
    Never 404s beyond "project not found" — an empty/fresh project
    still returns a valid (empty) shape."""
    pd = _project_dir(store, slug)
    if pd is None:
        return _problem(404, "project not found", type="/problems/not-found")

    runs_dir = pd / "runs"
    n_iters = _count_attested_iterations(runs_dir)
    project_report_state = _report_state(pd, source_kind="project")
    project_has_report = project_report_state["state"] != "missing"

    missions: list[dict[str, Any]] = []
    for mission_slug in mission_store.list_mission_slugs(pd):
        mdir = mission_store.mission_dir(pd, mission_slug)
        summary = mission_store.load_mission_summary(pd, slug, mission_slug)
        if summary is None:
            continue
        mission_report_state = _report_state(mdir, source_kind="mission")
        missions.append({
            "mission_slug": mission_slug,
            "goal": summary.goal,
            "lifecycle": summary.lifecycle,
            "has_report": mission_report_state["state"] != "missing",
            "report_state": mission_report_state,
        })

    return {
        "project_runs": {
            "n_iters": n_iters,
            "has_report": project_has_report,
            "report_state": project_report_state,
        },
        "missions": missions,
    }
