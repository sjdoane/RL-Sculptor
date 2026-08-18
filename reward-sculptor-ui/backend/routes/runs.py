"""Sculpt-run REST routes.

Runs are JobManager jobs of kind `sculpt_run`. This router is a thin
view over `JobManager` — there is no parallel state tracker per
Prompt 8 R3. The WebSocket route is in this same module for locality.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from backend.models.mission import StageIterationSummary
from backend.models.project import ProblemDetail
from backend.models.run import (
    IterEventSummary,
    RunControl,
    RunControlState,
    RunDetail,
    RunParams,
    RunSummary,
)
# §list/detail fitness parity (UI item 8): the SAME disk-scrape helper
# `routes/missions.py::get_stage_iter_detail` and
# `list_stage_iterations` use — shared rather than re-implemented here,
# so this project-level list can never disagree with the stage-level
# one about whether a given iteration "has" a fitness value. No import
# cycle: `routes/missions.py` never imports from `routes/runs.py`.
from backend.routes.missions import (
    _extract_objective_fitness,
    _read_fitness_and_naturalness,
)
from backend.services import mission_store, world_store
from backend.services.job_manager import Job, JobManager
from backend.services.physical_scene_audit import (
    audit_physical_scene_alignment,
)
from backend.services.project_store import ProjectStore
from backend.services.project_robot import resolve_project_reference_robot
from backend.services.run_manager import (
    build_iterations_summary,
    control_file_path,
    read_control_file,
    resolve_starting_skill_target,
    resolve_warm_start_checkpoint,
    run_sculpt_job,
    write_control_file,
)
from sculptor.skill_bundle import ImportTarget, compatibility_for
from sculptor.skill_library import SkillLibrary, SkillLibraryError
from sculptor.reference_authority import (
    ActiveReferenceAuthorityError,
    resolve_active_reference_authority,
)

# Snake-case-ish segment guard for mission_slug / stage_name before they are
# used to build filesystem paths. Rejects traversal and path separators.
_SAFE_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


router = APIRouter(tags=["runs"])


# ── DI ─────────────────────────────────────────────────────────────────
def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


# ── helpers ────────────────────────────────────────────────────────────
def _problem(
    status_code: int,
    title: str,
    *,
    detail: Optional[str] = None,
    type_: str = "about:blank",
    **extra: Any,
) -> JSONResponse:
    body = ProblemDetail(
        type=type_, title=title, status=status_code, detail=detail
    ).model_dump()
    body.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=body,
        media_type="application/problem+json",
    )


def _sha256_path(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_run(jobs: JobManager, slug: str, run_id: str) -> Optional[Job]:
    job = jobs.get(run_id)
    if job is None:
        return None
    # §Ship 21: also accept mission_stage_run kind so per-stage rows
    # in /runs are addressable by the standard run-detail / clip /
    # rollout / WS routes.
    if job.kind not in ("sculpt_run", "mission_stage_run"):
        return None
    if job.project_slug != slug:
        return None
    return job


def _resolve_run_root(job: Job, project_dir: Path) -> Path:
    """§Ship 21: where this run's `runs/iter_*/` artifacts live.

    For top-level sculpt_run jobs: `<project_dir>/runs/`.
    For mission_stage_run jobs: `<project_dir>/.missions/<mission_slug>/
    stages/<stage_name>/runs/` — each stage scaffolds its own mini-
    project with its own runs/ tree.

    §Ship 21e (review fix, HIGH): traversal guard. The mission_slug /
    stage_name come from job.params, which mission_jobs sets from the
    subprocess's stage_started event. Defense-in-depth: validate both
    against the snake_case pattern before building a filesystem path
    that feeds a FileResponse (get_iter_rollout). A malformed name
    (corrupt mission.json, hand-edited) could otherwise traverse out
    of the project dir. Mirrors the guards in routes/rewards.py.
    On any invalid component, fall back to the project runs dir.
    """
    if job.kind == "mission_stage_run":
        mission_slug = job.params.get("mission_slug")
        stage_name = job.params.get("stage_name")
        if (
            isinstance(mission_slug, str)
            and isinstance(stage_name, str)
            and _SAFE_PATH_SEGMENT.match(mission_slug)
            and _SAFE_PATH_SEGMENT.match(stage_name)
        ):
            return (
                project_dir / ".missions" / mission_slug
                / "stages" / stage_name / "runs"
            )
    return project_dir / "runs"


def _run_summary(job: Job) -> RunSummary:
    params = job.params or {}
    iters = build_iterations_summary(job)
    completed = sum(1 for it in iters if it.get("status") == "completed")
    current = next(
        (it["iter_index"] for it in iters if it.get("status") == "running"),
        None,
    )
    history_from_params = params.get("primary_metric_history")
    if isinstance(history_from_params, list):
        metric_history = [
            (float(v) if isinstance(v, (int, float)) else None)
            for v in history_from_params
        ]
    else:
        metric_history = [
            (it.get("primary_metric") if isinstance(it.get("primary_metric"), (int, float)) else None)
            for it in iters
        ]
    # §Ship 35 review (CRITICAL): build the parallel objective-fitness
    # history so the Runs tab can foreground fitness (job_to_run_summary
    # does this too; this REST builder must match it).
    fitness_history = [
        (it.get("fitness") if isinstance(it.get("fitness"), (int, float)) else None)
        for it in iters
    ]
    classification_raw = params.get("error_classification")
    classification = None
    if isinstance(classification_raw, dict):
        from backend.models.run import ErrorClassification

        try:
            classification = ErrorClassification(**classification_raw)
        except Exception:  # noqa: BLE001
            classification = None

    # §Ship 21: surface mission/stage context for mission_stage_run
    # rows so the Runs sidebar can group them under their parent
    # mission, and the detail pane can route per-stage rewards.
    return RunSummary(
        run_id=job.job_id,
        project_slug=job.project_slug or "",
        status=job.status,
        behavior_goal=str(params.get("behavior_goal") or ""),
        iterations_requested=int(params.get("iterations_requested") or params.get("iterations") or 0),
        iterations_completed=completed,
        current_iter_index=current,
        primary_metric_history=metric_history,
        fitness_history=fitness_history,
        started_at=job.started_at,
        ended_at=job.ended_at,
        error=job.error,
        error_classification=classification,
        kind=job.kind,  # type: ignore[arg-type]
        parent_id=job.parent_id,
        mission_slug=params.get("mission_slug"),
        stage_name=params.get("stage_name"),
        stage_index=params.get("stage_index"),
        mode=params.get("mode"),
    )


# ── §mission-persistence increment 2: disk-reconstructed run rows ─────
def _find_stage_checkpoint(iter_dir: Path) -> Optional[Path]:
    for name in ("checkpoint.pt", "checkpoint.zip"):
        p = iter_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _parse_iso_dt(s: Any) -> Optional[Any]:
    if not isinstance(s, str) or not s:
        return None
    from datetime import datetime as _dt, timezone as _tz

    try:
        dt = _dt.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        return dt
    except ValueError:
        return None


def _synthesize_disk_run_rows(
    project_dir: Path, project_slug: str, jobs: JobManager,
) -> list[RunSummary]:
    """Reconstruct `RunSummary` rows for mission stages that have
    trained iterations on disk but no LIVE `mission_stage_run` job —
    the gap `routes/missions.py`'s module docstring calls out: a
    completed/failed stage becomes unreachable via the in-memory job
    endpoints after a backend restart (JobManager is purely in-memory;
    `list_runs` previously returned NOTHING for such rows once the
    process restarted, silently dropping the Training tab's history).

    Reuses `mission_store.iter_unioned_stages_for_project` — the exact
    same disk-union stage list `GET .../missions/{slug}` returns — so
    a stage's position / display_label / failure_reason here always
    agrees with the Missions tab.

    Dedup: a stage with ANY resident `mission_stage_run` job — running,
    queued, OR terminal (completed/errored/stopped) — is skipped. A
    terminal job stays resident in JobManager until backend restart,
    and `list_runs` already emits a row for it via the live-job branch
    above (`stage_runs` / `_run_summary`); including the stage again
    here via `_synthesize_disk_run_rows` would double it. Only a stage
    with NO resident job at all for that (mission_slug, stage_name)
    pair, but at least one `runs/iter_*` dir on disk, gets a synthetic
    row with `run_id="disk:<mission_slug>/<stage_name>"`.
    """
    live_stage_keys: set[tuple[str, str]] = set()
    for j in jobs.list(kind="mission_stage_run", project_slug=project_slug):
        ms = j.params.get("mission_slug")
        sn = j.params.get("stage_name")
        if isinstance(ms, str) and isinstance(sn, str):
            live_stage_keys.add((ms, sn))

    out: list[RunSummary] = []
    for mission_slug, mission, stages in mission_store.iter_unioned_stages_for_project(
        project_dir,
    ):
        for idx, s in enumerate(stages):
            if (mission_slug, s.name) in live_stage_keys:
                continue
            stage_dir = mission_store.mission_dir(project_dir, mission_slug) / "stages" / s.name
            runs_root = stage_dir / "runs"
            if not runs_root.is_dir():
                continue
            iter_dirs = sorted(
                (d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith("iter_")),
                key=lambda d: d.name,
            )
            if not iter_dirs:
                continue
            completed = sum(1 for d in iter_dirs if _find_stage_checkpoint(d) is not None)

            if s.status == "succeeded":
                run_status = "completed"
                error = None
            elif s.status in ("failed", "superseded"):
                run_status = "errored"
                error = s.failure_reason
            elif s.status == "training":
                # Stage record claims "training" but there's no LIVE
                # job for it — the backend restarted (or crashed) mid-
                # stage. Honest answer: interrupted, not still running.
                run_status = "errored"
                error = "interrupted"
            else:
                # pending / skipped with disk artifacts (e.g. a
                # redecomposed-away stage that got a few iters in
                # before being superseded but somehow kept "pending" —
                # defensive fallback, shouldn't normally happen).
                run_status = "errored"
                error = s.failure_reason or "interrupted"

            out.append(RunSummary(
                run_id=f"disk:{mission_slug}/{s.name}",
                project_slug=project_slug,
                status=run_status,  # type: ignore[arg-type]
                behavior_goal=s.goal_text,
                iterations_requested=(
                    s.effective_max_iterations or s.max_iterations or 0
                ),
                iterations_completed=completed,
                current_iter_index=None,
                primary_metric_history=[],
                fitness_history=[],
                started_at=_parse_iso_dt(s.started_at),
                ended_at=_parse_iso_dt(s.finished_at),
                error=error,
                kind="mission_stage_run",
                parent_id=None,
                mission_slug=mission_slug,
                stage_name=s.name,
                stage_index=idx,
                mode=None,
            ))
    return out


# ── §mission-persistence increment 3: project-level disk run row ──────
# Plain PROJECT-level runs (artifacts at `<project_dir>/runs/iter_*/`)
# had the same restart gap the stage rows above closed: JobManager is
# in-memory, so after a backend restart the Training sidebar's "Single
# runs" section went empty even with dozens of trained iterations on
# disk. Unlike mission stages there is no per-run record on disk —
# iter dirs ACCUMULATE across every sculpt run of the project into one
# shared tree — so we synthesize ONE row covering that whole tree.
_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")

# The one well-known id for the synthetic project-level row. The
# frontend routes it to a disk-truth pane; every _find_run-gated
# endpoint 404s for it by construction (never a resident job id).
PROJECT_DISK_RUN_ID = "disk:project"


def _project_iter_dirs(runs_root: Path) -> list[tuple[int, Path]]:
    """Sorted `(iter_index, dir)` pairs under a `runs/` tree."""
    if not runs_root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for d in runs_root.iterdir():
        m = _ITER_DIR_RE.match(d.name)
        if m and d.is_dir():
            out.append((int(m.group(1)), d))
    out.sort(key=lambda t: t[0])
    return out


def _load_json_dict(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _read_project_metric_history(project_dir: Path) -> list[Optional[float]]:
    """`reports/metric_history.json`, per-iter-index primary metrics.

    The PROJECT-level file is a dict `{"primary_metric": <name>,
    "history": [...]}` (the stage-level analog in routes/missions.py is
    a bare list); accept both shapes so this helper stays correct if
    the writer ever unifies them."""
    path = project_dir / "reports" / "metric_history.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(loaded, dict):
        loaded = loaded.get("history")
    if not isinstance(loaded, list):
        return []
    return [
        float(v) if isinstance(v, (int, float)) else None for v in loaded
    ]


def _iter_looks_completed(iter_dir: Path) -> bool:
    """An iteration that finished training leaves `metrics.json` (and
    usually a checkpoint). Either marker counts — older iters may have
    one without the other."""
    if (iter_dir / "metrics.json").is_file():
        return True
    return _find_stage_checkpoint(iter_dir) is not None


def _synthesize_project_disk_run_row(
    project_dir: Path, project_slug: str, jobs: JobManager,
) -> Optional[RunSummary]:
    """One synthetic `sculpt_run` row when `<project_dir>/runs/iter_*`
    exist but NO resident sculpt_run job covers them (backend restarted
    since the run(s) ended). Mirrors `_synthesize_disk_run_rows` for
    mission stages.

    Dedup: ANY resident sculpt_run job for this project — running,
    queued, or terminal — suppresses the synthetic row. Iter dirs are
    one shared per-project tree accumulated across runs, so a resident
    job's row already reaches these artifacts, and there is no per-run
    partition on disk that would let us row-split honestly.

    Status comes from the LAST iteration's disk state: a finished
    marker (metrics.json / checkpoint) reads "completed"; a bare iter
    dir means the process died mid-iteration → errored/"interrupted".
    A run that ended between iterations is indistinguishable from a
    clean finish on disk, so this is best-effort truth, not history.
    """
    if jobs.list(kind="sculpt_run", project_slug=project_slug):
        return None
    pairs = _project_iter_dirs(project_dir / "runs")
    if not pairs:
        return None

    completed = sum(1 for _, d in pairs if _iter_looks_completed(d))
    _, last_dir = pairs[-1]
    if _iter_looks_completed(last_dir):
        run_status = "completed"
        error = None
    else:
        run_status = "errored"
        error = "interrupted"

    meta = _load_json_dict(project_dir / "metadata.json") or {}
    goal = str(meta.get("behavior_goal") or "")

    # Best-effort timestamps from iter-dir mtimes. NOTE: the row spans
    # every past run of the project, so this reads as "first to last
    # trained iteration", not one run's wall-clock duration.
    from datetime import datetime as _dt, timezone as _tz

    started_at = ended_at = None
    try:
        mtimes = [d.stat().st_mtime for _, d in pairs]
        started_at = _dt.fromtimestamp(min(mtimes), tz=_tz.utc)
        ended_at = _dt.fromtimestamp(max(mtimes), tz=_tz.utc)
    except OSError:
        pass

    return RunSummary(
        run_id=PROJECT_DISK_RUN_ID,
        project_slug=project_slug,
        status=run_status,  # type: ignore[arg-type]
        behavior_goal=goal,
        iterations_requested=0,  # unknown across runs → UI shows "?"
        iterations_completed=completed,
        current_iter_index=None,
        primary_metric_history=_read_project_metric_history(project_dir),
        fitness_history=[],
        started_at=started_at,
        ended_at=ended_at,
        error=error,
        kind="sculpt_run",
        parent_id=None,
        mission_slug=None,
        stage_name=None,
        stage_index=None,
        mode=None,
    )


# ── POST /projects/{slug}/runs ────────────────────────────────────────
@router.post(
    "/projects/{slug}/runs",
    response_model=RunSummary,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        412: {"model": ProblemDetail},
    },
)
def launch_run(
    slug: str,
    body: RunParams,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )

    if jobs.has_active_sculpt_run(slug):
        return _problem(
            status.HTTP_409_CONFLICT,
            "sculpt run already active",
            detail=(
                "This project already has a sculpt run in the running "
                "/ queued state. Stop it before launching another."
            ),
            type_="/problems/job-busy",
        )

    # Live runs require ANTHROPIC_API_KEY unless --dry-run is set.
    if not body.dry_run:
        import os

        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "ANTHROPIC_API_KEY not set",
                detail=(
                    "Live sculpt runs need ANTHROPIC_API_KEY to diagnose + "
                    "edit. Set it in the environment or pass dry_run=true."
                ),
                type_="/problems/no-api-key",
            )

        if not (body.fitness_metric or "").strip() and not (
            body.acknowledge_blind_fitness
        ):
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "objective fitness is required for live training",
                detail=(
                    "choose an objective fitness metric, or explicitly set "
                    "acknowledge_blind_fitness=true for a deliberate blind "
                    "ablation"
                ),
                type_="/problems/objective-fitness-required",
            )

    project_dir = Path(detail.project_dir)
    if body.warm_start_iteration is not None and body.starting_skill_id is not None:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "choose one starting policy source",
            detail=(
                "warm_start_iteration recovers this project's checkpoint; "
                "starting_skill_id selects a shared/imported skill. They cannot "
                "both initialize the same run."
            ),
            type_="/problems/starting-skill",
        )
    if body.initialization_mode is not None and body.starting_skill_id is None:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "initialization mode has no starting skill",
            detail="select a starting_skill_id or clear initialization_mode",
            type_="/problems/starting-skill",
        )
    if (
        body.expected_starting_skill_manifest_digest is not None
        and body.starting_skill_id is None
    ):
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "starting-skill manifest pin has no skill",
            detail=(
                "expected_starting_skill_manifest_digest is only valid with "
                "starting_skill_id"
            ),
            type_="/problems/starting-skill-stale",
        )
    if (
        body.starting_skill_id is not None
        and body.expected_starting_skill_manifest_digest is None
    ):
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "starting-skill manifest pin is required",
            detail=(
                "refresh the starting-point picker and submit the receipt's "
                "manifest_digest with the selected skill"
            ),
            type_="/problems/starting-skill-stale",
        )
    if bool(body.reference_clip_id) != bool(body.reference_robot):
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "reference motion is incomplete",
            detail=(
                "reference_clip_id and reference_robot must be supplied together"
            ),
            type_="/problems/reference-motion",
        )

    selected_skill = None
    starting_skill_target_receipt: Optional[dict[str, Any]] = None
    selected_mode = body.initialization_mode or "actor_only"
    resolved_reference_clip_id = body.reference_clip_id
    resolved_reference_robot = body.reference_robot
    reference_feasibility: Optional[dict[str, Any]] = None
    try:
        active_reference = resolve_active_reference_authority(
            project_dir / "rewards",
        )
    except ActiveReferenceAuthorityError as exc:
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "active reward reference authority is invalid",
            detail=str(exc),
            type_="/problems/active-reference-authority",
        )
    active_reference_receipt = (
        active_reference.to_dict() if active_reference is not None else None
    )
    if body.starting_skill_id is not None:
        try:
            resolve_project_reference_robot(project_dir)
        except (OSError, TypeError, ValueError) as exc:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "project robot identity is unresolved",
                detail=str(exc),
                type_="/problems/starting-skill-project-contract",
                code="project_robot_unresolved",
            )
        library = SkillLibrary()
        selected_skill = library.load(body.starting_skill_id)
        if selected_skill is None:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "starting skill is unavailable",
                detail=f"no skill record {body.starting_skill_id!r}",
                type_="/problems/starting-skill",
            )
        if (
            selected_skill.manifest_digest
            != body.expected_starting_skill_manifest_digest
        ):
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "starting skill changed after selection",
                detail=(
                    "the selected skill's immutable manifest digest no longer "
                    "matches the launch request; refresh the starting-point "
                    "picker before launching"
                ),
                type_="/problems/starting-skill-stale",
                expected_manifest_digest=(
                    body.expected_starting_skill_manifest_digest
                ),
                actual_manifest_digest=selected_skill.manifest_digest,
            )
        try:
            target_payload, starting_skill_target_receipt = (
                resolve_starting_skill_target(
                    project_dir,
                    require_policy_contract=(selected_mode != "reference_only"),
                )
            )
        except Exception as exc:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "project policy contract is unavailable",
                detail=f"{type(exc).__name__}: {exc}",
                type_="/problems/starting-skill-project-contract",
            )
        target = ImportTarget(
            adapter_class=target_payload["adapter_class"],
            task_id=target_payload["task_id"],
            robot_slug=target_payload["robot_slug"],
            compatibility_contract=target_payload["compatibility_contract"],
        )
        compatibility = compatibility_for(selected_skill, target)
        if compatibility["reasons"]:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "starting skill is incompatible",
                detail="; ".join(compatibility["reasons"]),
                type_="/problems/starting-skill-incompatible",
                compatibility=compatibility,
            )
        if selected_mode not in compatibility["allowed_initialization_modes"]:
            selected_reasons = compatibility.get("mode_reasons", {}).get(
                selected_mode, [],
            )
            reason_suffix = (
                "; " + "; ".join(str(reason) for reason in selected_reasons)
                if selected_reasons
                else ""
            )
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "initialization mode is not supported",
                detail=(
                    f"{selected_mode!r} is unavailable for this skill; choose "
                    + ", ".join(compatibility["allowed_initialization_modes"])
                    + reason_suffix
                ),
                type_="/problems/starting-skill-mode",
                compatibility=compatibility,
            )
        if selected_mode != "reference_only":
            try:
                library.checkpoint_path_for(selected_skill)
            except SkillLibraryError as exc:
                return _problem(
                    status.HTTP_412_PRECONDITION_FAILED,
                    "starting skill failed its integrity check",
                    detail=str(exc),
                    type_="/problems/starting-skill-integrity",
                )
        if (
            selected_mode == "reference_only"
            and selected_skill.reference_clip_id
            and selected_skill.reference_robot
        ):
            bundled_pair = (
                selected_skill.reference_clip_id,
                selected_skill.reference_robot,
            )
            explicit_pair = (body.reference_clip_id, body.reference_robot)
            if all(explicit_pair) and explicit_pair != bundled_pair:
                return _problem(
                    status.HTTP_412_PRECONDITION_FAILED,
                    "starting skill and motion selection disagree",
                    detail=(
                        "clear the separately selected motion or select the "
                        "same motion carried by the starting-skill manifest"
                    ),
                    type_="/problems/starting-skill-reference",
                )
            resolved_reference_clip_id, resolved_reference_robot = bundled_pair
        elif selected_mode == "reference_only":
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "starting skill has no reference motion",
                detail="reference_only requires a bundled reference clip",
                type_="/problems/starting-skill-reference",
            )
    if active_reference is not None:
        active_pair = (
            active_reference.reference_clip_id,
            active_reference.reference_robot,
        )
        selected_pair = (
            resolved_reference_clip_id,
            resolved_reference_robot,
        )
        if any(selected_pair) and selected_pair != active_pair:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "motion selection disagrees with the active reward",
                detail=(
                    "the exact reward selected by rewards/current.py embeds "
                    f"{active_pair[1]}/{active_pair[0]}; select that motion "
                    "or promote a reward built for the requested motion"
                ),
                type_="/problems/active-reference-authority",
                active_reference=active_reference_receipt,
            )
        resolved_reference_clip_id, resolved_reference_robot = active_pair
    # A promoted authored world is part of the run's scientific input, not a
    # cosmetic preview. Verify the atomic tuple at the last responsible
    # moment so a stale browser or direct API client cannot start training on
    # tampered/drifted artifacts. Legacy projects without a selection keep
    # their existing default-scene behavior.
    selection_path = project_dir / "env" / "selection_current.json"
    authored_world_receipt: dict[str, Any] | None = None
    if selection_path.is_file():
        try:
            world_report = world_store.training_preflight(project_dir)
        except Exception as exc:  # noqa: BLE001 — fail closed before GPU work
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "authored world could not be verified",
                detail=f"{type(exc).__name__}: {exc}",
                type_="/problems/world-integrity",
            )
        if not isinstance(world_report, dict):
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "authored world integrity check failed",
                detail="authored world selection disappeared during preflight",
                type_="/problems/world-integrity",
            )
        if not bool(world_report.get("ok")):
            errors = world_report.get("errors") or ["unknown integrity error"]
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "authored world integrity check failed",
                detail="; ".join(str(error) for error in errors),
                type_="/problems/world-integrity",
            )
        if world_report.get("robot_matches_project") is False:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "training environment targets another robot",
                detail=(
                    f"the authored world targets "
                    f"{world_report.get('world_robot')!r}, but the project "
                    f"targets {world_report.get('project_robot')!r}; "
                    "re-author the world for the project robot before launching"
                ),
                type_="/problems/world-robot-mismatch",
            )
        authored_world_receipt = {
            "selection_version": world_report.get("selection_version"),
            "tuple_hash": world_report.get("tuple_hash"),
            "world_robot": world_report.get("world_robot"),
            "project_robot": world_report.get("project_robot"),
        }
    if body.warm_start_iteration is not None:
        try:
            resolve_warm_start_checkpoint(
                project_dir, body.warm_start_iteration,
            )
        except Exception as exc:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "warm-start checkpoint is unavailable",
                detail=f"{type(exc).__name__}: {exc}",
                type_="/problems/warm-start",
            )
    if bool(resolved_reference_clip_id) != bool(resolved_reference_robot):
        return _problem(
            status.HTTP_412_PRECONDITION_FAILED,
            "reference motion is incomplete",
            detail=(
                "reference_clip_id and reference_robot must be supplied "
                "together"
            ),
            type_="/problems/reference-motion",
        )
    if resolved_reference_clip_id and resolved_reference_robot:
        # Resolve the exact (embodiment, clip) pair before queuing GPU work.
        # The index is a cache, so provenance + clip bytes remain the
        # authoritative existence check.
        from sculptor.refs import library as reference_library

        ref_dir = reference_library.clip_dir(
            resolved_reference_robot, resolved_reference_clip_id,
        )
        provenance_path = ref_dir / reference_library.PROVENANCE_FILENAME
        clip_path = ref_dir / reference_library.CLIP_FILENAME
        if not provenance_path.is_file() or not clip_path.is_file():
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "reference motion is unavailable",
                detail=(
                    f"no complete reference clip "
                    f"{resolved_reference_robot}/{resolved_reference_clip_id}"
                ),
                type_="/problems/reference-motion",
            )
        # `_prepare_reference_guided_run` (sculptor/sculpt.py) hard-raises
        # without an authoritative world selection, because the tracking
        # reward has to be bound to the world atomically. That raise happens
        # at subprocess start, so the run appeared to launch and then died
        # with a ValueError in the log. Say it here, before the GPU is
        # touched, and say what to do about it.
        if not selection_path.is_file():
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "a reference motion needs an authored world",
                detail=(
                    "attaching a reference clip makes this a tracking-first "
                    "run, and the generated tracking reward is bound to the "
                    "authored world atomically — so the project needs a "
                    "promoted world first. Author one from the World tab "
                    "(the default flat scene still needs promoting), or "
                    "launch without a motion prior."
                ),
                type_="/problems/reference-motion",
            )
        # Lokesh feasibility admission: a reference-backed TRAINING run is
        # only scientifically meaningful after the exact clip has been
        # physics-tracked successfully.  Never trust provenance.tier alone;
        # the verifier re-hashes both clip and rollout artifacts.  Tier-K is
        # still useful for an explicit inspect-only check, where the worker
        # re-verifies the immutable reference contract and then returns before
        # creating a sculpt subprocess.  It never trains, rolls out, or
        # publishes a checkpoint; that weaker admission is recorded plainly.
        from sculptor.refs.track import (
            TierDAdmissionError,
            require_tierd_admission,
            require_tierd_target_compatibility,
        )

        try:
            target_robot = resolve_project_reference_robot(project_dir)
        except ValueError as exc:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "project robot identity is unresolved",
                detail=str(exc),
                type_="/problems/reference-feasibility",
            )
        if resolved_reference_robot != target_robot:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "reference motion is for a different robot",
                detail=(
                    f"selected reference robot {resolved_reference_robot!r} "
                    f"does not match project robot {target_robot!r}"
                ),
                type_="/problems/reference-feasibility",
            )
        try:
            certificate = require_tierd_admission(
                resolved_reference_robot, resolved_reference_clip_id,
            )
            certificate = require_tierd_target_compatibility(
                certificate,
                project_dir,
                target_robot=target_robot,
            )
            certificate_reason = None
        except TierDAdmissionError as exc:
            certificate = None
            certificate_reason = str(exc)
        if certificate is None and not body.dry_run:
            return _problem(
                status.HTTP_412_PRECONDITION_FAILED,
                "reference motion is kinematic-only",
                detail=(
                    "live reference-backed training requires a verified "
                    "Tier-D physics-tracking certificate for the exact clip "
                    f"and rollout: {certificate_reason or 'not certified'}"
                ),
                type_="/problems/reference-feasibility",
                reference_robot=resolved_reference_robot,
                reference_clip_id=resolved_reference_clip_id,
                required_tier="D",
                admitted_tier="K",
            )
        if certificate is not None:
            reference_feasibility = {
                "status": "tierd_verified",
                "tier": "D",
                "clip_sha256": certificate.clip_content_sha256,
                "rollout_sha256": certificate.rollout_sha256,
                "certificate_sha256": certificate.certificate_sha256,
                "execution_contract_sha256": (
                    certificate.execution_contract_sha256
                ),
                "execution_boundary_sha256": (
                    certificate.execution_boundary_sha256
                ),
                "target_robot": target_robot,
                "tracked_at": certificate.tracked_at,
            }
        else:
            reference_feasibility = {
                "status": "kinematic_reference_inspection_only",
                "tier": "K",
                "clip_sha256": _sha256_path(clip_path),
                "rollout_sha256": None,
                "reason": certificate_reason,
                "scope": "contract_and_reference_resolution",
                "inspection_only": True,
                "training_authorized": False,
                "training_invoked": False,
                "checkpoint_published": False,
                "target_robot": target_robot,
            }
    run_params: dict[str, Any] = body.model_dump()
    run_params["initialization_mode"] = selected_mode if selected_skill else None
    run_params["reference_clip_id"] = resolved_reference_clip_id
    run_params["reference_robot"] = resolved_reference_robot
    run_params["reference_feasibility"] = reference_feasibility
    run_params["active_reference_authority"] = active_reference_receipt
    run_params["starting_skill_target_receipt"] = (
        starting_skill_target_receipt
    )
    # Pin the admitted world tuple separately from the request. The worker
    # re-attests this receipt immediately before subprocess creation so a
    # queued run cannot inherit a re-authored or deleted world silently.
    run_params["authored_world_receipt"] = authored_world_receipt
    run_params["objective_fitness_receipt"] = {
        "requested_metric": body.fitness_metric,
        "objective_requested": bool((body.fitness_metric or "").strip()),
        "blind_ablation_acknowledged": bool(body.acknowledge_blind_fitness),
        "dry_run": bool(body.dry_run),
        "authorization": (
            "dry_run"
            if body.dry_run
            else (
                "objective_requested"
                if (body.fitness_metric or "").strip()
                else "blind_ablation_acknowledged"
            )
        ),
    }
    job = jobs.submit(
        kind="sculpt_run",
        project_slug=slug,
        fn=run_sculpt_job(project_dir=project_dir, run_params=run_params),
        params=run_params,
    )
    return _run_summary(job)


# ── GET /projects/{slug}/runs ─────────────────────────────────────────
@router.get(
    "/projects/{slug}/runs",
    response_model=list[RunSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_runs(
    slug: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    # §Ship 21: include `mission_stage_run` rows (per-stage child jobs
    # registered by mission_jobs._stream_stdout). Sidebar groups them
    # under their parent mission.
    sculpt_runs = jobs.list(kind="sculpt_run", project_slug=slug)
    stage_runs = jobs.list(kind="mission_stage_run", project_slug=slug)
    rows = [_run_summary(j) for j in (*sculpt_runs, *stage_runs)]
    # §mission-persistence increment 2: append disk-reconstructed rows
    # for stages with trained iterations but no live JobManager entry
    # (backend restart, or the mission moved past that stage). Purely
    # additive — never replaces a live row (see the dedup inside).
    rows.extend(_synthesize_disk_run_rows(
        Path(detail.project_dir), slug, jobs,
    ))
    # §increment 3: same treatment for plain project-level runs — one
    # synthetic sculpt_run row when iter dirs exist on disk but no
    # resident sculpt_run job covers them (see the dedup inside).
    project_row = _synthesize_project_disk_run_row(
        Path(detail.project_dir), slug, jobs,
    )
    if project_row is not None:
        rows.append(project_row)
    return rows


# ── GET /projects/{slug}/runs/{run_id} ────────────────────────────────
@router.get(
    "/projects/{slug}/runs/{run_id}",
    response_model=RunDetail,
    responses={404: {"model": ProblemDetail}},
)
def get_run(
    slug: str,
    run_id: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "run not found",
            detail=f"no sculpt_run with id {run_id!r} in project {slug!r}",
            type_="/problems/not-found",
        )
    summary = _run_summary(job)
    # Rehydrate every recorded launch field for standalone runs.  The old
    # minimal reconstruction kept only behavior/iterations/no_kg/dry_run, so
    # the detail API claimed advanced UI values were null even while the
    # durable subprocess log proved that training_iterations, num_envs,
    # device, rollout, seed, and metric overrides were active.  Filter by the
    # Pydantic model fields because job.params also contains runtime-only
    # values such as cmd/control_file.
    #
    # §Ship 21: stage runs don't have a full top-level RunParams (the parent
    # mission_execute owns those); missing fields simply retain model defaults
    # and behavior_goal becomes the stage's goal_text.
    iters_for_params = int(
        job.params.get("iterations_requested")
        or job.params.get("iterations")
        or 1
    )
    params_payload = {
        field_name: job.params[field_name]
        for field_name in RunParams.model_fields
        if field_name in job.params
    }
    params_payload.update({
        "behavior_goal": str(job.params.get("behavior_goal") or ""),
        "iterations": iters_for_params,
        "no_kg": bool(job.params.get("no_kg") or False),
        "dry_run": bool(job.params.get("dry_run") or False),
    })
    params = RunParams.model_validate(params_payload)
    iterations = [
        IterEventSummary(**it) for it in build_iterations_summary(job)
    ]
    return RunDetail(
        **summary.model_dump(),
        params=params,
        reference_feasibility=(
            dict(job.params["reference_feasibility"])
            if isinstance(job.params.get("reference_feasibility"), dict)
            else None
        ),
        objective_fitness_receipt=(
            dict(job.params["objective_fitness_receipt"])
            if isinstance(job.params.get("objective_fitness_receipt"), dict)
            else None
        ),
        starting_skill_target_receipt=(
            dict(job.params["starting_skill_target_receipt"])
            if isinstance(
                job.params.get("starting_skill_target_receipt"), dict,
            )
            else None
        ),
        iterations=iterations,
        stdout_tail=list(job.log_ring),
        total_event_count=len(job.events),
    )


# ── DELETE /projects/{slug}/runs/{run_id} ─────────────────────────────
@router.delete(
    "/projects/{slug}/runs/{run_id}",
    response_model=RunSummary,
    responses={404: {"model": ProblemDetail}},
)
def kill_run(
    slug: str,
    run_id: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    if store.get(slug) is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "run not found",
            type_="/problems/not-found",
        )
    jobs.stop(run_id)
    job.emit({"type": "run_stopped", "source": "user"})
    return _run_summary(job)


# ── PATCH /projects/{slug}/runs/{run_id}/control ──────────────────────
@router.patch(
    "/projects/{slug}/runs/{run_id}/control",
    response_model=RunControlState,
    responses={404: {"model": ProblemDetail}},
)
def control_run(
    slug: str,
    run_id: str,
    body: RunControl,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    """§Ship 39 (H1): interactive control for a live run — flip Auto/Manual
    at any point, resume a pause (optionally with human feedback for the next
    iteration's diagnose), or stop cleanly. Merges into the run's control
    sidecar that the sculpt subprocess polls at each iteration boundary."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "run not found",
            detail=f"no sculpt_run with id {run_id!r} in project {slug!r}",
            type_="/problems/not-found",
        )
    path = control_file_path(Path(detail.project_dir), run_id)
    ctrl = read_control_file(path)
    if body.mode is not None:
        ctrl["mode"] = body.mode
    if body.stop:
        ctrl["stop"] = True
    if body.resume:
        ctrl["resume_token"] = int(ctrl.get("resume_token", 0) or 0) + 1
        ctrl["feedback"] = body.feedback
    if body.gen_retry or body.gen_continue:
        # §Ship 45: deliver the launch-time-generation retry decision the
        # pre-phase is polling for (retry → regenerate; continue → run blind).
        ctrl["gen_decision"] = "retry" if body.gen_retry else "blind"
        ctrl["gen_decision_seq"] = int(ctrl.get("gen_decision_seq", 0) or 0) + 1
    write_control_file(path, ctrl)
    # Reflect the new mode on the job so a reconnect / REST summary sees it,
    # and tee a control event so other connected clients stay in sync.
    job.params["mode"] = ctrl.get("mode")
    job.emit({
        "type": "run_control_updated",
        "mode": ctrl.get("mode"),
        "resume": bool(body.resume),
        "stop": bool(body.stop),
    })
    return RunControlState(**ctrl)


# ── WS /projects/{slug}/runs/{run_id}/events ──────────────────────────
@router.websocket("/ws/projects/{slug}/runs/{run_id}/events")
async def run_events_ws(
    websocket: WebSocket, slug: str, run_id: str,
) -> None:
    jobs: JobManager = websocket.app.state.job_manager
    store: ProjectStore = websocket.app.state.project_store

    await websocket.accept()
    if store.get(slug) is None:
        await websocket.send_json({
            "type": "error",
            "error": f"project {slug!r} not found",
        })
        await websocket.close(code=4404)
        return
    job = _find_run(jobs, slug, run_id)
    if job is None:
        await websocket.send_json({
            "type": "error",
            "error": f"run {run_id!r} not found in project {slug!r}",
        })
        await websocket.close(code=4404)
        return

    # Replay: send a `connected` snapshot + the buffered events. The
    # frontend uses `seq` to dedupe on reconnect. We cap the replay at
    # the last 200 log lines + ALL typed events (typed events are small).
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
            "total_event_count": len(job.events),
            "replay_count": len(replay_events),
        })
        for ev in replay_events:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    queue = job.subscribe()
    # If the job already terminated we still want the client to receive
    # any final events that landed between replay and subscribe — the
    # job.events append is atomic, so replaying once covers it.
    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json(
                {"type": "terminal", "status": job.status}
            )
        finally:
            job.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await websocket.close()
        return

    # Live loop.
    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
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
            if ev.get("type") in ("run_completed", "run_errored", "run_stopped"):
                # Give the client a moment to receive, then close.
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "terminal", "status": job.status}
                    )
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


async def _drain_client(ws: WebSocket) -> None:
    """Keep the receive side pumping so the client can disconnect cleanly.
    We don't expect any inbound messages — the WS is one-directional —
    but the server must still await `receive` to notice close frames."""
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return


# ── WS /frames (live clip push) ───────────────────────────────────────
_FRAME_EVENT_TYPES = {"new_clip", "clip_skipped"}


@router.websocket("/ws/projects/{slug}/runs/{run_id}/frames")
async def run_frames_ws(
    websocket: WebSocket, slug: str, run_id: str,
) -> None:
    """Thin filtered view of the job's event stream. Only forwards
    `new_clip` and `clip_skipped` events. Replay mirrors /events — on
    connect the client gets `connected` + every prior clip event so
    they can populate the replay strip without an extra REST round."""
    jobs: JobManager = websocket.app.state.job_manager
    store: ProjectStore = websocket.app.state.project_store

    await websocket.accept()
    if store.get(slug) is None:
        await websocket.send_json({"type": "error", "error": f"project {slug!r} not found"})
        await websocket.close(code=4404)
        return
    job = _find_run(jobs, slug, run_id)
    if job is None:
        await websocket.send_json({"type": "error", "error": f"run {run_id!r} not found"})
        await websocket.close(code=4404)
        return

    clips = [ev for ev in job.events if ev.get("type") in _FRAME_EVENT_TYPES]
    try:
        await websocket.send_json({
            "type": "connected",
            "status": job.status,
            "replay_count": len(clips),
        })
        for ev in clips:
            await websocket.send_json(ev)
    except (WebSocketDisconnect, RuntimeError):
        return

    if job.status in ("completed", "errored", "stopped"):
        try:
            await websocket.send_json({"type": "terminal", "status": job.status})
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()
        return

    queue = job.subscribe()
    recv_task = asyncio.create_task(_drain_client(websocket))
    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                break
            ev = get_task.result()
            etype = ev.get("type")
            if etype in _FRAME_EVENT_TYPES:
                try:
                    await websocket.send_json(ev)
                except (WebSocketDisconnect, RuntimeError):
                    break
            if etype in ("run_completed", "run_errored", "run_stopped"):
                with contextlib.suppress(Exception):
                    await websocket.send_json({"type": "terminal", "status": job.status})
                break
    finally:
        job.unsubscribe(queue)
        recv_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


# ── clip + rollout file serving (Range-aware) ─────────────────────────
@router.get(
    "/projects/{slug}/runs/{run_id}/clips/{file}",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_clip_file(
    slug: str, run_id: str, file: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(404, "project not found", type_="/problems/not-found")
    if _find_run(jobs, slug, run_id) is None:
        return _problem(404, "run not found", type_="/problems/not-found")

    # Harden: only allow "iter_<N>.mp4" filenames, no traversal.
    if not _CLIP_NAME_RE.match(file):
        return _problem(
            404, "clip not found",
            detail=f"{file!r} is not a valid clip filename",
            type_="/problems/not-found",
        )
    project_dir = Path(detail.project_dir)
    path = project_dir / "uploads" / "live_clips" / run_id / file
    try:
        resolved = path.resolve()
        resolved.relative_to(
            (project_dir / "uploads" / "live_clips" / run_id).resolve()
        )
    except (ValueError, OSError):
        return _problem(
            404, "clip not found", type_="/problems/not-found"
        )
    if not resolved.is_file():
        return _problem(
            404, "clip not found",
            detail=f"no clip at {resolved.name}",
            type_="/problems/not-found",
        )
    return FileResponse(resolved, media_type="video/mp4")


@router.get(
    "/projects/{slug}/runs/{run_id}/iterations/{iter_index}/rollout",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_iter_rollout(
    slug: str, run_id: str, iter_index: int,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    detail = store.get(slug)
    if detail is None:
        return _problem(404, "project not found", type_="/problems/not-found")
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return _problem(404, "run not found", type_="/problems/not-found")
    project_dir = Path(detail.project_dir)
    # §Ship 21: stage runs live under .missions/<m>/stages/<s>/runs/
    runs_root = _resolve_run_root(job, project_dir)
    path = runs_root / f"iter_{iter_index}" / "rollout" / "rollout.mp4"
    if not path.is_file() or path.stat().st_size < 2048:
        return _problem(
            404, "rollout not available",
            detail=(
                f"iter {iter_index} has no rollout.mp4 yet (either "
                "still rendering or the run errored before this iter)"
            ),
            type_="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")


_CLIP_NAME_RE = re.compile(r"^iter_\d+\.mp4$")


# ── §increment 3: project-level disk-truth iteration endpoints ────────
# Mirrors routes/missions.py's §C2 stage endpoints (list_stage_iterations
# / get_stage_iter_rollout), but for the PROJECT runs tree — no
# JobManager entry required, so the synthetic "disk:project" row's
# detail pane keeps working after a backend restart.
@router.get(
    "/projects/{slug}/iterations",
    response_model=list[StageIterationSummary],
    responses={404: {"model": ProblemDetail}},
)
def list_project_iterations(
    slug: str,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Disk-truth iteration list for the project-level runs tree."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    project_dir = Path(detail.project_dir)
    metric_history = _read_project_metric_history(project_dir)

    out: list[StageIterationSummary] = []
    for iter_index, d in _project_iter_dirs(project_dir / "runs"):
        primary_metric: Optional[float] = None
        if 0 <= iter_index < len(metric_history):
            primary_metric = metric_history[iter_index]
        if primary_metric is None:
            metrics = _load_json_dict(d / "metrics.json") or {}
            mm = metrics.get("metrics")
            if isinstance(mm, dict):
                v = mm.get("mean_return")
                if isinstance(v, (int, float)):
                    primary_metric = float(v)

        spec = _load_json_dict(d / "reward_spec.json") or {}
        reward_version = spec.get("version")

        # §list/detail fitness parity (UI item 8): was a bare
        # `behavior.json['fitness']` read (null for any iteration whose
        # objective fitness only landed in reward_spec.json /
        # fitness.json / the diagnoser's evidence prose) — now shares
        # `routes/missions.py`'s `_extract_objective_fitness`, the same
        # helper `get_stage_iter_detail` uses.
        diagnosis = _load_json_dict(d / "diagnosis.json") or {}
        evidence = (
            diagnosis.get("evidence")
            if isinstance(diagnosis.get("evidence"), str) else None
        )
        fitness = _extract_objective_fitness(d, spec, evidence)

        rollout_path = d / "rollout" / "rollout.mp4"
        fresh_rollout_count = sum(
            1
            for path in d.glob("rollout_fresh_*/rollout.mp4")
            if path.is_file() and path.stat().st_size >= 2048
        )
        # §D24 (F4): plain project-level runs never go through mission
        # stage selection, so this file won't exist here today — read it
        # anyway for symmetry with `list_stage_iterations` (same model,
        # same on-disk convention) rather than hardcoding False.
        contradiction = _load_json_dict(d / "fitness_contradiction.json")

        # §fitness.json/selection.json backend increment (commit f1c339d
        # follow-up): same disk-truth fitness.json-first read the mission
        # C2 endpoint uses (`routes/missions.py::_read_fitness_and_
        # naturalness`) — kept in ONE place so the project-level and
        # stage-level iteration lists never disagree about steer_fitness/
        # progress/naturalness for the same on-disk iter.
        steer_fitness, progress, naturalness_flag, naturalness_hard_reject, \
            fitness_source = _read_fitness_and_naturalness(d)
        if fitness_source is not None:
            fitness_doc = _load_json_dict(d / "fitness.json") or {}
            v = fitness_doc.get("fitness")
            if isinstance(v, (int, float)):
                fitness = float(v)

        out.append(StageIterationSummary(
            iter_index=iter_index,
            primary_metric=primary_metric,
            fitness=fitness,
            has_rollout=rollout_path.is_file() and rollout_path.stat().st_size > 0,
            has_checkpoint=_find_stage_checkpoint(d) is not None,
            reward_version=reward_version if isinstance(reward_version, str) else None,
            fitness_contradiction=contradiction is not None,
            fitness_components=(
                contradiction.get("components")
                if isinstance(contradiction, dict) else None
            ),
            steer_fitness=steer_fitness,
            progress=progress,
            naturalness_flag=naturalness_flag,
            naturalness_hard_reject=naturalness_hard_reject,
            fitness_source=fitness_source,
            fresh_rollout_count=fresh_rollout_count,
        ))
    return out


@router.get(
    "/projects/{slug}/iterations/{iter_index}/physical-scene-audit",
    responses={404: {"model": ProblemDetail}},
)
def get_project_iter_physical_scene_audit(
    slug: str,
    iter_index: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Fail-closed evidence check that physical objects share the task frame."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    project_dir = Path(detail.project_dir)
    iter_dir = project_dir / "runs" / f"iter_{iter_index}"
    if not iter_dir.is_dir():
        return _problem(
            status.HTTP_404_NOT_FOUND,
            "iteration not found",
            detail=f"iter {iter_index} does not exist",
            type_="/problems/not-found",
        )
    return audit_physical_scene_alignment(project_dir, iter_dir)


@router.get(
    "/projects/{slug}/iterations/{iter_index}/rollout",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_project_iter_rollout(
    slug: str,
    iter_index: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Disk-truth rollout.mp4 for one project-level iteration — no
    JobManager entry required. Same >2048-byte truncation guard as
    `get_iter_rollout` / `get_stage_iter_rollout`."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}", type_="/problems/not-found",
        )
    path = (
        Path(detail.project_dir) / "runs" / f"iter_{iter_index}"
        / "rollout" / "rollout.mp4"
    )
    if not path.is_file() or path.stat().st_size < 2048:
        return _problem(
            status.HTTP_404_NOT_FOUND, "rollout not available",
            detail=(
                f"iter {iter_index} has no rollout.mp4 in the project "
                "runs tree (either it never rendered or the iter "
                "errored before rollout capture)"
            ),
            type_="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")


@router.get(
    "/projects/{slug}/iterations/{iter_index}/fresh-rollouts/{fresh_index}",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"model": ProblemDetail},
    },
)
def get_project_iter_fresh_rollout(
    slug: str,
    iter_index: int,
    fresh_index: int,
    store: ProjectStore = Depends(get_store),
) -> Any:
    """Fresh-seed best-policy replay from durable project artifacts."""
    detail = store.get(slug)
    if detail is None:
        return _problem(
            status.HTTP_404_NOT_FOUND, "project not found",
            detail=f"no project with slug {slug!r}",
            type_="/problems/not-found",
        )
    if fresh_index < 0:
        return _problem(
            status.HTTP_404_NOT_FOUND, "fresh rollout not available",
            detail="fresh rollout index must be non-negative",
            type_="/problems/not-found",
        )
    path = (
        Path(detail.project_dir) / "runs" / f"iter_{iter_index}"
        / f"rollout_fresh_{fresh_index}" / "rollout.mp4"
    )
    if not path.is_file() or path.stat().st_size < 2048:
        return _problem(
            status.HTTP_404_NOT_FOUND, "fresh rollout not available",
            detail=(
                f"iter {iter_index} has no complete fresh rollout "
                f"at index {fresh_index}"
            ),
            type_="/problems/not-found",
        )
    return FileResponse(path, media_type="video/mp4")
