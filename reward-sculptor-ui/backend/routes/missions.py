"""Mission orchestrator REST + WebSocket routes (Ship 18a).

Endpoints (all under `/projects/{slug}/missions`):

  POST   /                              — start a `mission_decompose` job
                                          → 202 + JobSummary
  GET    /                              — list missions in this project
                                          → 200 + list[MissionSummary]
  GET    /{mission_slug}                — full mission detail
                                          → 200 + MissionDetail | 404
  POST   /{mission_slug}/run            — start a `mission_execute` job
                                          → 202 + JobSummary
  DELETE /{mission_slug}                — hard-delete the mission dir
                                          → 200 + DeleteMissionResponse
                                          | 409 if active job

  GET    /{mission_slug}/stages/{stage}/iterations
                                        — disk-truth per-iter rows for
                                          one stage (no job required)
                                          → 200 + list[StageIterationSummary]
  GET    /{mission_slug}/stages/{stage}/iterations/{i}/rollout
                                        — disk-truth rollout.mp4 for one
                                          stage iteration (no job required)
                                          → 200 video/mp4 | 404
  GET    /{mission_slug}/stages/{stage}/env-spec
                                        — the stage's env-spec + version
                                          list (no job required)
                                          → 200 + StageEnvSpecInfo

  WS     /ws/projects/{slug}/missions/{mission_slug}/events
                                        — replay + tee the active
                                          decompose / execute job's events

Source-of-truth invariant (per Ship 18a plan-review):
  `mission.json` is canonical; GET handlers read it on every call via
  `mission_store.load_mission_*`. JobManager events overlay during in-
  flight jobs (active_job_id / active_job_kind on the response).

§C2 stage de-siloing (this ship): the three `.../stages/{stage}/...`
routes above read ONLY the filesystem — no JobManager lookup at all.
`routes/runs.py`'s run/rollout endpoints require a live
`mission_stage_run` Job (`_find_run`), so a completed stage becomes
unreachable after a backend restart or once the mission has moved on
to the next stage even though its `runs/iter_*/` artifacts are still
on disk. These routes close that gap: any stage of any mission is
viewable regardless of job liveness.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Any, Annotated, Awaitable, Callable, Optional

from fastapi import (
    APIRouter,
    Depends,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.models.kg import JobDetail, JobSummary
from backend.models.mission import (
    CreateMissionRequest,
    DeleteMissionResponse,
    MissionDetail,
    MissionSummary,
    RunMissionRequest,
    StageEnvSpecInfo,
    StageIterationSummary,
)
from backend.models.project import ProblemDetail
from backend.services import mission_store
from backend.services.job_manager import JobManager
from backend.services.mission_jobs import (
    run_mission_decompose_job,
    run_mission_execute_job,
)
from backend.services.project_store import ProjectStore


# §C2: same lowercase snake/kebab-case-ish segment guard routes/runs.py
# applies to mission_slug / stage_name before they hit the filesystem.
# Rejects "..", "/", "\\", uppercase, spaces — anything that isn't a
# plain slug component. Mission slugs + stage names are both generated
# from `_slugify` (models/mission.py's `mission_slug` pattern is
# `^[a-z][a-z0-9_-]{0,63}$`), so this is not just defense-in-depth —
# it's the actual production shape.
_SAFE_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


router = APIRouter(tags=["missions"])
ws_router = APIRouter()


# ── DI ───────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ──────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: Optional[str] = None,
    type_: str = "about:blank",
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail,
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _project_dir(store: ProjectStore, slug: str) -> Optional[Path]:
    detail = store.get(slug)
    if detail is None:
        return None
    return Path(detail.project_dir)


# ── POST /projects/{slug}/missions  (decompose) ──────────────────────
# §Ship-19c response_model fix: was JobSummary (no `params`), so the
# frontend couldn't tell what mission_slug got assigned and the auto-
# open detail dialog + optimistic cache insert silently no-op'd. Use
# JobDetail (which extends JobSummary with `params` + `result`) so
# `params.mission_slug` is on the wire.
@router.post(
    "/projects/{slug}/missions",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        422: {"model": ProblemDetail},
    },
)
def create_mission(
    slug: str,
    body: CreateMissionRequest,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    # Audit fix #E (CRITICAL): two concurrent POSTs without a
    # `mission_slug` override would BOTH read the same `existing_slugs`
    # snapshot and derive the SAME slug, then both jobs would write
    # to the same `<slug>/mission.json`. Acquire the per-project lock
    # to serialize slug-derivation + job-submission. The lock is held
    # ONLY across the cheap reservation step (creating the empty
    # mission_dir to claim the slug); the actual decompose runs
    # without holding the lock so other endpoints aren't blocked.
    try:
        project_lock = store.acquire_lock(slug, timeout=5.0)
    except Exception as e:  # noqa: BLE001 — BusyError or similar
        return _problem(
            status.HTTP_409_CONFLICT,
            "project is busy",
            detail=(
                f"another write is in progress on project {slug!r}: "
                f"{type(e).__name__}: {e}"
            ),
            type_="/problems/state-conflict",
        )

    try:
        existing_slugs = set(mission_store.list_mission_slugs(project_dir))
        if body.mission_slug is not None:
            if body.mission_slug in existing_slugs:
                return _problem(
                    status.HTTP_409_CONFLICT,
                    "mission slug already exists",
                    detail=(
                        f"mission_slug={body.mission_slug!r} is already in "
                        f"use under project {slug!r}. Pick another or omit "
                        f"`mission_slug` to auto-derive."
                    ),
                    type_="/problems/state-conflict",
                )
            mission_slug = body.mission_slug
        else:
            mission_slug = mission_store.derive_unique_mission_slug(
                body.goal, existing_slugs,
            )
        # Reserve the slug atomically by mkdir before releasing the
        # lock — `list_mission_slugs` filters to dirs containing
        # mission.json, but our slug-collision set was JUST `existing
        # _slugs` (dir names). Add an `.in_progress` marker file so a
        # racing list call sees the dir but treats it as "claimed not
        # yet decomposed."
        reserved_dir = mission_store.mission_dir(project_dir, mission_slug)
        reserved_dir.mkdir(parents=True, exist_ok=True)
        # Empty marker — decompose will write mission.json here.
        (reserved_dir / ".decompose_pending").touch()

        # §Ship 21a: thread run_defaults from the NewMissionDialog
        # Advanced tab through to the decompose job so the resulting
        # mission.json carries them; RunMissionDialog pre-fills from
        # them on first open. body.run_defaults is already validated
        # via the RunMissionRequest pydantic shape.
        run_defaults_dict = (
            body.run_defaults.model_dump(exclude_none=False)
            if body.run_defaults is not None else None
        )
        # §mission-persistence increment 2: stage_metric_required lives
        # on CreateMissionRequest (a mission-creation-time policy, not
        # a per-launch RunMissionRequest knob), but `run_mission`'s 409
        # guard reads it off `mission.run_defaults` — the one place
        # already plumbed through to the persisted mission.json. Stash
        # it there even when the caller didn't otherwise set any
        # run_defaults.
        if body.stage_metric_required:
            run_defaults_dict = dict(run_defaults_dict or {})
            run_defaults_dict["stage_metric_required"] = True
        job = jobs.submit(
            kind="mission_decompose",
            project_slug=slug,
            fn=run_mission_decompose_job(
                project_dir=project_dir,
                project_slug=slug,
                goal=body.goal,
                mission_slug=mission_slug,
                no_kg=body.no_kg,
                run_defaults=run_defaults_dict,
                gen_stage_metrics=body.gen_stage_metrics,
                stage_metric_candidates=body.stage_metric_candidates,
            ),
            params={
                "mission_slug": mission_slug,
                "goal": body.goal,
                "no_kg": body.no_kg,
                # Surface in params so the frontend optimistic-cache
                # write can show the user the Advanced settings round-
                # tripped successfully.
                "run_defaults": run_defaults_dict,
            },
        )
    finally:
        project_lock.release()
    # §Ship-19c: return JobDetail (has `params`) so the frontend's
    # auto-open detail dialog can read mission_slug from the response.
    return job.to_detail()


# ── GET /projects/{slug}/missions  (list) ────────────────────────────
@router.get(
    "/projects/{slug}/missions",
    response_model=list[MissionSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_missions(
    slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    out: list[MissionSummary] = []
    for mission_slug in mission_store.list_mission_slugs(project_dir):
        active = jobs.active_mission_job(slug, mission_slug)
        summary = mission_store.load_mission_summary(
            project_dir, slug, mission_slug,
            active_job_kind=active.kind if active else None,
            active_job_id=active.job_id if active else None,
        )
        if summary is not None:
            out.append(summary)
    # Newest-first (mirror project list ordering).
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


# ── GET /projects/{slug}/missions/{mission_slug}  (detail) ───────────
@router.get(
    "/projects/{slug}/missions/{mission_slug}",
    response_model=MissionDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    active = jobs.active_mission_job(slug, mission_slug)
    detail = mission_store.load_mission_detail(
        project_dir, slug, mission_slug,
        active_job_kind=active.kind if active else None,
        active_job_id=active.job_id if active else None,
    )
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            detail=(
                f"no mission {mission_slug!r} under project {slug!r}. "
                f"Available: {mission_store.list_mission_slugs(project_dir)}"
            ),
            type_="/problems/not-found",
        )
    return detail


# ── POST /projects/{slug}/missions/{mission_slug}/run  (execute) ─────
@router.post(
    "/projects/{slug}/missions/{mission_slug}/run",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def run_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    body: Optional[RunMissionRequest] = None,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if not mission_store.mission_json_path(
        project_dir, mission_slug,
    ).is_file():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            detail=(
                f"no mission {mission_slug!r} under project {slug!r}"
            ),
            type_="/problems/not-found",
        )

    # Concurrent-mission guard: at most one active mission-scoped job
    # (decompose, execute, a per-stage run, or a metric regenerate)
    # PER (project, mission) pair. Widened from decompose/execute-only
    # to also catch a live `mission_stage_metric_regen` for this
    # mission — both it and `mission_execute` end in a non-atomic
    # `save_mission` write to the same mission.json, so letting
    # `mission_execute` start mid-regenerate risks a last-write-wins
    # clobber of whichever one finishes first.
    active_job = jobs.active_mission_scoped_job(slug, mission_slug)
    if active_job is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                f"mission {mission_slug!r} already has an active "
                f"{active_job.kind} job. Wait for it to finish "
                f"or call POST /jobs/{{id}}/stop to cancel."
            ),
            type_="/problems/state-conflict",
        )

    # GPU contention guard: refuse if any sculpt_run / mission_execute
    # is in flight cross-project (per Ship 18a plan-review).
    if jobs.has_any_active_gpu_job():
        return _problem(
            status.HTTP_409_CONFLICT,
            "GPU is busy",
            detail=(
                "Another sculpt_run or mission_execute is in flight. "
                "Mission execution requires exclusive GPU access "
                "(spawns per-stage sculpt_run subprocesses)."
            ),
            type_="/problems/state-conflict",
        )

    # §Ship-19d: optional run-time knobs from the RunMissionDialog.
    # An empty / missing body is the Ship 16 default (per-stage budget
    # from mission.json, no Goal A/B). The backend forwards these as
    # CLI flags to `sculpt mission-run` (see run_mission_execute_job).
    run_kwargs: dict[str, Any] = {}
    if body is not None:
        # Pydantic model_dump excludes unset fields with exclude_none.
        run_kwargs = body.model_dump(exclude_none=False)

    # §mission-persistence increment 2: stage_metric_required guard.
    # Persisted at creation time onto mission.run_defaults (see
    # create_mission). Blocks the run while any RUNNABLE stage
    # (pending/training — the ones that will actually execute) has no
    # steering_metric, unless this launch sets proceed_blind. A
    # succeeded/failed/skipped/superseded stage is never going to run
    # again, so it's exempt regardless of its metric.
    proceed_blind = bool(body.proceed_blind) if body is not None else False
    if not proceed_blind:
        from sculptor.mission import load_mission, MissionValidationError

        try:
            mission = load_mission(
                mission_store.mission_json_path(project_dir, mission_slug)
            )
        except (MissionValidationError, Exception):  # noqa: BLE001
            mission = None
        if mission is not None:
            run_defaults = mission.run_defaults or {}
            if run_defaults.get("stage_metric_required"):
                missing = [
                    s.name for s in mission.stages
                    if s.status in ("pending", "training")
                    and not getattr(s, "steering_metric", None)
                ]
                if missing:
                    return _problem(
                        status.HTTP_409_CONFLICT,
                        "stage metrics missing",
                        detail=(
                            f"mission {mission_slug!r} was created with "
                            f"stage_metric_required=True and the "
                            f"following runnable stage(s) have no "
                            f"steering metric: {missing}. Regenerate "
                            f"their metrics, or resend this request "
                            f"with proceed_blind=true to run without "
                            f"per-stage metrics."
                        ),
                        type_="/problems/stage-metrics-missing",
                        missing_stages=missing,
                    )

    job = jobs.submit(
        kind="mission_execute",
        project_slug=slug,
        fn=run_mission_execute_job(
            project_dir=project_dir,
            project_slug=slug,
            mission_slug=mission_slug,
            run_kwargs=run_kwargs,
            # §Ship 21: pass JobManager so the streamer can register
            # per-stage child Jobs (mission_stage_run kind) on the fly.
            job_manager=jobs,
        ),
        params={"mission_slug": mission_slug, **run_kwargs},
    )
    return job.to_detail()


# ── DELETE /projects/{slug}/missions/{mission_slug} ──────────────────
@router.delete(
    "/projects/{slug}/missions/{mission_slug}",
    response_model=DeleteMissionResponse,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def delete_mission(
    slug: str,
    mission_slug: str,
    request: Request,
    store: ProjectStore = Depends(get_store),
) -> Any:
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            type_="/problems/not-found",
        )

    if not mission_store.mission_dir(project_dir, mission_slug).is_dir():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            type_="/problems/not-found",
        )

    if jobs.active_mission_job(slug, mission_slug) is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                "Stop the active decompose / execute job before "
                "deleting the mission."
            ),
            type_="/problems/state-conflict",
        )

    freed = mission_store.delete_mission(project_dir, mission_slug)
    return DeleteMissionResponse(
        mission_slug=mission_slug, freed_bytes=freed,
    )


# ── §C2 stage de-siloing: disk-truth stage-iteration endpoints ────────
# Mission-stage iterations live at `<mission_dir>/stages/<stage>/runs/
# iter_<N>/` regardless of whether a `mission_stage_run` JobManager
# entry is still alive for them — the sculpt subprocess writes them to
# disk exactly like a top-level run. Unlike `routes/runs.py`'s
# `_find_run`-gated endpoints, everything below reads disk only, so a
# completed stage stays viewable after a backend restart or once the
# mission has moved on to a later stage.
_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


def _stage_dir_or_404(
    store: ProjectStore, slug: str, mission_slug: str, stage: str,
) -> "Path | JSONResponse":
    """Resolve + validate `<mission_dir>/stages/<stage>/`, or a 404
    JSONResponse explaining why. Shared by all three C2 routes so the
    traversal guard + existence checks live in exactly one place."""
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if not _SAFE_PATH_SEGMENT.match(mission_slug) or not _SAFE_PATH_SEGMENT.match(stage):
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "invalid path segment",
            detail=(
                f"mission_slug={mission_slug!r} / stage={stage!r} must "
                "each match a plain slug component"
            ),
            type_="/problems/not-found",
        )
    if mission_slug not in mission_store.list_mission_slugs(project_dir):
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "mission not found",
            detail=f"no mission {mission_slug!r} under project {slug!r}",
            type_="/problems/not-found",
        )
    stage_dir = mission_store.mission_dir(project_dir, mission_slug) / "stages" / stage
    if not stage_dir.is_dir():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "stage not found",
            detail=(
                f"no stage {stage!r} under mission {mission_slug!r} "
                f"(project {slug!r})"
            ),
            type_="/problems/not-found",
        )
    return stage_dir


# ── GET .../stages/{stage}/iterations ─────────────────────────────────
@router.get(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/iterations",
    response_model=list[StageIterationSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_stage_iterations(
    slug: str,
    mission_slug: str,
    stage: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Disk-truth iteration list for one stage — no JobManager entry
    required. Mirrors `sculptor.export.list_exportable_iters` but does
    NOT require a checkpoint (a rollout can exist without a saved
    checkpoint and vice versa; the UI wants both rows)."""
    resolved = _stage_dir_or_404(store, slug, mission_slug, stage)
    if isinstance(resolved, JSONResponse):
        return resolved
    stage_dir = resolved

    runs_root = stage_dir / "runs"
    out: list[StageIterationSummary] = []
    if not runs_root.is_dir():
        return out

    # metric_history.json (per-index list) is the stage's canonical
    # primary-metric series when present; falls back to each iter's own
    # metrics.json for older stages / gaps in the history file.
    metric_history: Optional[list[Any]] = None
    history_path = stage_dir / "reports" / "metric_history.json"
    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                metric_history = loaded
        except (OSError, ValueError):
            metric_history = None

    for d in sorted(runs_root.iterdir()):
        m = _ITER_DIR_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        iter_index = int(m.group(1))

        primary_metric: Optional[float] = None
        if metric_history is not None and 0 <= iter_index < len(metric_history):
            v = metric_history[iter_index]
            if isinstance(v, (int, float)):
                primary_metric = float(v)
        if primary_metric is None:
            metrics = _load_json_dict(d / "metrics.json") or {}
            mm = metrics.get("metrics")
            if isinstance(mm, dict):
                v = mm.get("mean_return")
                if isinstance(v, (int, float)):
                    primary_metric = float(v)

        behavior = _load_json_dict(d / "rollout" / "behavior.json") or {}
        fitness = behavior.get("fitness")

        spec = _load_json_dict(d / "reward_spec.json") or {}
        reward_version = spec.get("version")

        rollout_path = d / "rollout" / "rollout.mp4"
        checkpoint_path = _find_checkpoint(d)

        out.append(StageIterationSummary(
            iter_index=iter_index,
            primary_metric=primary_metric,
            fitness=fitness if isinstance(fitness, (int, float)) else None,
            has_rollout=rollout_path.is_file() and rollout_path.stat().st_size > 0,
            has_checkpoint=checkpoint_path is not None,
            reward_version=reward_version if isinstance(reward_version, str) else None,
        ))
    out.sort(key=lambda r: r.iter_index)
    return out


def _load_json_dict(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _find_checkpoint(iter_dir: Path) -> Optional[Path]:
    for name in ("checkpoint.pt", "checkpoint.zip"):
        p = iter_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ── GET .../stages/{stage}/iterations/{i}/rollout ─────────────────────
@router.get(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/iterations/{i}/rollout",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_stage_iter_rollout(
    slug: str,
    mission_slug: str,
    stage: str,
    i: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Disk-truth rollout.mp4 for one stage iteration — no JobManager
    entry required. Mirrors `routes/runs.py::get_iter_rollout`'s
    >2048-byte guard (a truncated / still-rendering file must not be
    served as if it were a finished clip)."""
    resolved = _stage_dir_or_404(store, slug, mission_slug, stage)
    if isinstance(resolved, JSONResponse):
        return resolved
    stage_dir = resolved

    path = stage_dir / "runs" / f"iter_{i}" / "rollout" / "rollout.mp4"
    if not path.is_file() or path.stat().st_size < 2048:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "rollout not available",
            detail=(
                f"iter {i} of stage {stage!r} has no rollout.mp4 yet "
                "(either still rendering or the iter errored before "
                "rollout capture)"
            ),
            type_="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")


# ── GET .../stages/{stage}/env-spec ────────────────────────────────────
@router.get(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/env-spec",
    response_model=StageEnvSpecInfo,
    responses={404: {"model": ProblemDetail}},
)
def get_stage_env_spec(
    slug: str,
    mission_slug: str,
    stage: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """The stage's environment-adaptation spec — mirrors
    `routes/projects.py::get_project_env_spec` but scoped to
    `<mission_dir>/stages/<stage>/env/`. A grounded stage (never ran
    env-spec adaptation) has no `env/` dir; that's `{current: null,
    versions: []}`, NOT a 404 — the stage itself is real, it just has
    no spec to show.

    `current.meta.source` starting `"reference:"` is the RSI
    (reference-state-initialization) tell the UI surfaces as a chip.
    """
    resolved = _stage_dir_or_404(store, slug, mission_slug, stage)
    if isinstance(resolved, JSONResponse):
        return resolved
    stage_dir = resolved

    env_dir = stage_dir / "env"
    current: Optional[dict] = None
    cur_path = env_dir / "current.json"
    if cur_path.is_file():
        loaded = _load_json_dict(cur_path)
        if loaded is not None:
            current = loaded
    versions: list[str] = []
    if env_dir.is_dir():
        for p in env_dir.glob("v*.json"):
            if p.stem[1:].isdigit():
                versions.append(p.stem)
        versions.sort(key=lambda s: int(s[1:]))
    return StageEnvSpecInfo(
        active=current is not None, current=current, versions=versions,
    )


# ── GET .../stages/{stage}/iterations/{i}/checkpoint ───────────────────
# §mission-persistence increment 2: disk-truth checkpoint download,
# same traversal guard as the rollout endpoint above (_stage_dir_or_404
# validates slug/mission_slug/stage; `i` is a path int so FastAPI
# already rejects non-integer segments before we see it).
@router.get(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/iterations/{i}/checkpoint",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_stage_iter_checkpoint(
    slug: str,
    mission_slug: str,
    stage: str,
    i: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Disk-truth checkpoint download for one stage iteration — no
    JobManager entry required. Mirrors `get_stage_iter_rollout`."""
    resolved = _stage_dir_or_404(store, slug, mission_slug, stage)
    if isinstance(resolved, JSONResponse):
        return resolved
    stage_dir = resolved

    iter_dir = stage_dir / "runs" / f"iter_{i}"
    checkpoint_path = _find_checkpoint(iter_dir)
    if checkpoint_path is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "checkpoint not available",
            detail=(
                f"iter {i} of stage {stage!r} has no checkpoint file "
                "(training may not have reached a save point, or the "
                "iter errored before saving)"
            ),
            type_="/problems/not-found",
        )
    download_name = (
        f"{mission_slug}_{stage}_iter{i}_checkpoint{checkpoint_path.suffix}"
    )
    return FileResponse(
        checkpoint_path,
        media_type="application/octet-stream",
        filename=download_name,
    )


# ── POST .../stages/{stage}/metric/regenerate ───────────────────────────
class _RegenerateMetricBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_candidates: Annotated[int, Field(ge=1, le=4)] = 1


@router.post(
    "/projects/{slug}/missions/{mission_slug}/stages/{stage}/metric/regenerate",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def regenerate_stage_metric(
    slug: str,
    mission_slug: str,
    stage: str,
    request: Request,
    body: Optional[_RegenerateMetricBody] = None,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """User-triggered regeneration of ONE stage's steering metric.

    409s if the mission has an active mission_execute / mission_decompose
    job, OR if another mission_stage_metric_regen for this SAME mission
    is already in flight (two overlapping regenerates both end in a
    non-atomic `save_mission` write — the second clobbers the first),
    OR if the named stage is currently the one training (a live
    mission_stage_run child job for this (mission_slug, stage) pair) —
    regenerating the metric out from under an in-flight stage would
    race with the orchestrator's own read of `steering_metric`.

    Runs in-process via `generate_stage_metrics(only_stages=[stage])`
    (same call `_do_stage_metrics` makes at decompose time), which
    BYPASSES the "skip if steering_metric already set" guard for this
    one named stage — that's the whole point of a user-requested
    regenerate. Emits the same `mission_stage_metrics_*` /
    `stage_metric_gen_*` event types as the decompose-time pass so any
    UI already listening on the mission WS for those types picks this
    up too (job kind differs — mission_stage_metric_regen — so the UI
    can also target this job specifically if it wants a dedicated
    progress surface).
    """
    jobs: JobManager = request.app.state.job_manager
    project_dir = _project_dir(store, slug)
    if project_dir is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    resolved = _stage_dir_or_404(store, slug, mission_slug, stage)
    if isinstance(resolved, JSONResponse):
        return resolved

    if jobs.active_mission_job(slug, mission_slug) is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                f"mission {mission_slug!r} has an active decompose or "
                f"execute job; wait for it to finish before "
                f"regenerating a stage metric."
            ),
            type_="/problems/state-conflict",
        )
    # Widened (previously unchecked): also 409 against another LIVE
    # `mission_stage_metric_regen` for this same mission (even a
    # different stage). Two concurrent regenerate calls both read+
    # mutate the same in-memory `Mission` from `load_mission` and both
    # end in a non-atomic `save_mission` write — the second writer
    # silently clobbers the first's result.
    other_regen = jobs.active_mission_metric_regen_job(slug, mission_slug)
    if other_regen is not None:
        return _problem(
            status.HTTP_409_CONFLICT,
            "mission has active job",
            detail=(
                f"mission {mission_slug!r} already has an active "
                f"metric regenerate job (stage "
                f"{other_regen.params.get('stage_name')!r}); wait for "
                f"it to finish before starting another."
            ),
            type_="/problems/state-conflict",
        )
    # A live mission_stage_run child job for THIS stage means it's
    # currently training — regenerating its metric mid-flight would
    # race the orchestrator's read of steering_metric.
    for j in jobs.list(kind="mission_stage_run", project_slug=slug):
        if (
            j.status in ("running", "queued")
            and j.params.get("mission_slug") == mission_slug
            and j.params.get("stage_name") == stage
        ):
            return _problem(
                status.HTTP_409_CONFLICT,
                "stage is training",
                detail=(
                    f"stage {stage!r} of mission {mission_slug!r} is "
                    f"currently training; wait for it to finish before "
                    f"regenerating its metric."
                ),
                type_="/problems/state-conflict",
            )

    n_candidates = body.n_candidates if body is not None else 1

    def _do_regen() -> Callable[[Any, asyncio.Event], Awaitable[dict[str, Any]]]:
        async def _runner(job: Any, cancel: asyncio.Event) -> dict[str, Any]:
            from sculptor.adapters.base import load_adapter
            from sculptor.llm import set_llm_log_dir
            from sculptor.mission import load_mission, save_mission
            from sculptor.mission_metrics import generate_stage_metrics

            md = mission_store.mission_dir(project_dir, mission_slug)

            def _emit_metric_ev(ev: dict[str, Any]) -> None:
                try:
                    if isinstance(ev, dict) and ev.get("type"):
                        job.emit({**ev, "mission_slug": mission_slug})
                except Exception:  # noqa: BLE001 — progress is advisory
                    pass

            def _do() -> dict[str, Any]:
                mission = load_mission(md)
                adapter = load_adapter(project_dir / "config.toml")
                set_llm_log_dir(md)
                report = generate_stage_metrics(
                    mission,
                    robot_hint=getattr(adapter, "task_id", None),
                    n_candidates=n_candidates,
                    on_event=_emit_metric_ev,
                    only_stages=[stage],
                )
                # §step 4b: does generate_stage_metrics persist
                # mission.json itself? No — it mutates `mission` in
                # place and documents "the caller re-saves" (see its
                # docstring). Mirror _do_stage_metrics's atomic-enough
                # save (Mission.save_mission — same call the decompose
                # path uses; not a tmp+rename atomic write, but this
                # matches EVERY existing mission.json writer in this
                # codebase, decompose included).
                save_mission(mission, md)
                return report

            job.emit({
                "type": "mission_stage_metrics_started",
                "mission_slug": mission_slug,
                "n_stages": 1,
                "only_stages": [stage],
            })
            report = await asyncio.to_thread(_do)
            job.emit({
                "type": "mission_stage_metrics_completed",
                "mission_slug": mission_slug,
                **{k: report.get(k) for k in
                   ("generated", "rejected", "skipped")},
            })
            return report

        return _runner

    job = jobs.submit(
        kind="mission_stage_metric_regen",
        project_slug=slug,
        fn=_do_regen(),
        params={
            "mission_slug": mission_slug,
            "stage_name": stage,
            "n_candidates": n_candidates,
        },
    )
    return job.to_detail()


# ── WS /ws/projects/{slug}/missions/{mission_slug}/events ────────────
@ws_router.websocket(
    "/ws/projects/{slug}/missions/{mission_slug}/events",
)
async def mission_events_ws(
    websocket: WebSocket,
    slug: str,
    mission_slug: str,
) -> None:
    """Replay + live-tee the active mission_decompose or
    mission_execute job's events for a (project, mission) pair.

    Mirrors the runs WebSocket pattern at runs.py — same envelope:
    `connected` snapshot, replay (last 200 log_line + all typed
    events), live forward, `terminal` on job end.
    """
    await websocket.accept()
    app = websocket.app
    store: ProjectStore = app.state.project_store
    jobs: JobManager = app.state.job_manager

    if store.get(slug) is None:
        await websocket.send_json({
            "type": "error",
            "error": f"project {slug!r} not found",
        })
        await websocket.close(code=4404)
        return

    job = jobs.active_mission_job(slug, mission_slug)
    if job is None:
        # Allow connecting even when no active job — the UI may want
        # to subscribe in advance. Send a pseudo-connected with an
        # empty event list and close cleanly.
        await websocket.send_json({
            "type": "connected",
            "status": "no_active_job",
            "total_event_count": 0,
            "replay_count": 0,
        })
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    # Replay: last 200 log_line + ALL typed events.
    replay_log_count = 0
    replay_events: list[dict[str, Any]] = []
    log_budget = 200
    for ev in reversed(job.events):
        if ev.get("type") == "log_line":
            if replay_log_count < log_budget:
                replay_events.append(ev)
                replay_log_count += 1
        else:
            replay_events.append(ev)
    replay_events.reverse()

    try:
        await websocket.send_json({
            "type": "connected",
            "status": job.status,
            "kind": job.kind,
            "job_id": job.job_id,
            "total_event_count": len(job.events),
            "replay_count": len(replay_events),
        })
        for ev in replay_events:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    queue = job.subscribe()
    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json(
                {"type": "terminal", "status": job.status},
            )
        finally:
            job.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if recv_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                break
            ev = get_task.result()
            try:
                await websocket.send_json(ev)
            except (WebSocketDisconnect, RuntimeError):
                break
            terminal_types = (
                "mission_decompose_completed", "mission_decompose_errored",
                "mission_execute_completed", "mission_execute_errored",
                "mission_completed", "mission_halted_terminal",
            )
            if ev.get("type") in terminal_types:
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "terminal", "status": job.status},
                    )
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _drain_client(ws: WebSocket) -> None:
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return
