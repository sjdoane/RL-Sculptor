"""Disk-side helpers for missions (Ship 18a).

Single invariant per Ship 18a plan-review: `mission.json` is the source
of truth. Every read goes through `sculptor.mission.load_mission` so
relocations are safe (mission_dir reconstructed from file location).
Every write goes through `sculptor.mission.save_mission` (atomic via
tmp+rename, already implemented in sculpt.py:_atomic_save_mission for
the orchestrator path).

Layout:
    <project_dir>/.missions/<mission_slug>/mission.json
    <project_dir>/.missions/<mission_slug>/stages/<stage_name>/...

This file does NOT manipulate stage subdirs — those are managed by
`mission_run` (Ship 16).
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.models.mission import (
    MissionDetail,
    MissionLifecycleStatus,
    MissionSummary,
    StageSchema,
)


MISSIONS_DIRNAME = ".missions"  # hidden subdir under each project_dir
MISSION_JSON_NAME = "mission.json"


# ── Path helpers ─────────────────────────────────────────────────────
def missions_root(project_dir: Path) -> Path:
    """Return `<project_dir>/.missions/` (created if missing).

    Hidden by convention so the dot-prefix avoids collision with
    sculpt's existing visible subdirs (`runs/`, `rewards/`,
    `reports/`, `kg/`, `uploads/`).
    """
    p = project_dir / MISSIONS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def mission_dir(project_dir: Path, mission_slug: str) -> Path:
    return missions_root(project_dir) / mission_slug


def mission_json_path(project_dir: Path, mission_slug: str) -> Path:
    return mission_dir(project_dir, mission_slug) / MISSION_JSON_NAME


def list_mission_slugs(project_dir: Path) -> list[str]:
    """List existing mission slugs under `<project_dir>/.missions/`.

    Filters to directories that contain EITHER a `mission.json` file
    OR a `.decompose_pending` reservation marker (audit fix #E:
    in-flight decompose claims the slug atomically before mission.json
    is written). A directory with neither file is treated as orphaned
    debris and skipped.
    """
    root = missions_root(project_dir)
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (
            (p / MISSION_JSON_NAME).is_file()
            or (p / ".decompose_pending").is_file()
        ):
            out.append(p.name)
    return out


# ── Slug derivation (mirrors project_store._ensure_unique_slug) ──────
_SLUG_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM_RE = re.compile(r"^-+|-+$")


def _slugify(goal: str) -> str:
    """Best-effort URL-safe ASCII slug from a free-form goal string.

    Handles: emoji / non-ASCII (dropped), all-stopwords (returns
    `mission` fallback), aggressive whitespace collapse. Capped at 32
    chars to leave headroom for a `-NN` collision suffix while
    keeping under any reasonable PATH_MAX.

    Audit cross-reference (#C): KEEP IN SYNC with
    `sculptor.cli._derive_mission_slug` — both must produce the same
    slug for the same goal so a mission created via CLI and one
    created via the REST API don't collide unexpectedly.
    """
    cleaned = _SLUG_NON_WORD_RE.sub("-", goal.lower())
    cleaned = _SLUG_TRIM_RE.sub("", cleaned)
    if not cleaned:
        return "mission"
    return cleaned[:32].rstrip("-") or "mission"


def derive_unique_mission_slug(
    goal: str, existing_slugs: set[str],
) -> str:
    """Return a slug guaranteed not to collide with `existing_slugs`.

    Mirrors `project_store._ensure_unique_slug`'s pattern: generate a
    base via `_slugify`, append `-2`/`-3`/... on collision, cap the
    counter at 99 so a pathological mission set doesn't loop forever.
    """
    base = _slugify(goal)
    slug = base
    n = 2
    while slug in existing_slugs:
        slug = f"{base}-{n}"
        n += 1
        if n > 99:
            raise RuntimeError(
                f"could not derive unique mission slug from {goal!r}: "
                f"99 candidates collided. Pass an explicit slug."
            )
    return slug


# ── Read paths ───────────────────────────────────────────────────────
def _stages_to_schema(stages: list) -> list[StageSchema]:
    """Convert sculptor.mission.Stage dataclass list → pydantic
    StageSchema list. Fields are 1:1 by name."""
    out: list[StageSchema] = []
    for s in stages:
        out.append(StageSchema(
            name=s.name,
            goal_text=s.goal_text,
            success_criterion=s.success_criterion,
            max_iterations=s.max_iterations,
            parent_stage=s.parent_stage,
            reward_seed_prompt=s.reward_seed_prompt,
            kg_seed_papers=list(s.kg_seed_papers or []),
            status=s.status,
            final_policy_path=s.final_policy_path,
            final_reward_path=s.final_reward_path,
            best_metric=s.best_metric,
            iterations_used=s.iterations_used,
            started_at=s.started_at,
            finished_at=s.finished_at,
            redecomposition_attempts=s.redecomposition_attempts,
        ))
    return out


def _derive_lifecycle(
    stages: list, active_job_kind: Optional[str], current_idx: int,
) -> MissionLifecycleStatus:
    """Compute the mission's lifecycle from on-disk stage statuses
    and any in-flight job kind.

    This is the source-of-truth contract: lifecycle is DERIVED, not
    stored, so stale UI state can never disagree with disk reality.
    """
    if active_job_kind == "mission_decompose":
        # Decompose-in-flight is the only state where the mission
        # exists but has no stages on disk yet — but the file does
        # exist by the time we GET here, so this branch fires only
        # if the caller passed `active_job_kind` for an in-flight
        # decompose targeting a soon-to-exist slug.
        return "running"
    if active_job_kind == "mission_execute":
        return "running"

    # No active job — derive from stage statuses.
    if not stages:
        # No stages at all: shouldn't normally happen post-decompose;
        # treat as errored so it surfaces to the user.
        return "errored"
    statuses = [s.status for s in stages]
    if all(s == "succeeded" for s in statuses):
        return "completed"
    if any(s == "failed" for s in statuses):
        return "halted"
    # Some pending / training / skipped → ready to run / resume.
    return "ready"


def load_mission_summary(
    project_dir: Path,
    project_slug: str,
    mission_slug: str,
    *,
    active_job_kind: Optional[str] = None,
    active_job_id: Optional[str] = None,
) -> Optional[MissionSummary]:
    """Read mission.json and return the slim summary view, or None
    if the mission doesn't exist on disk.

    Audit fix #F: a CORRUPT mission.json (parse error, missing fields)
    no longer silently disappears from the list. It surfaces with
    `lifecycle="errored"` and stub fields so the user can DELETE it.
    """
    from sculptor.mission import load_mission, MissionValidationError

    json_path = mission_json_path(project_dir, mission_slug)
    if not json_path.is_file():
        # §Ship 21a fix: distinguish "missing" from "decomposing".
        # Ship 18a's atomic slug-reservation creates `.decompose_pending`
        # before mission.json exists. While decompose runs (30-90 s) the
        # dir exists, the marker is set, but mission.json isn't written
        # yet. Pre-fix this returned None so the slug DISAPPEARED from
        # GET /missions during the in-flight window — the frontend's
        # auto-open dialog hit a 404 and rendered a destructive error
        # banner instead of the "Planning" spinner. Surface as a
        # decomposing stub so the dialog renders correctly.
        pending_marker = mission_dir(project_dir, mission_slug) / ".decompose_pending"
        if pending_marker.is_file():
            return MissionSummary(
                mission_slug=mission_slug,
                project_slug=project_slug,
                goal="(decomposing — Claude is planning the curriculum)",
                n_stages=0,
                current_stage_idx=0,
                decomposition_model="",
                created_at=_parse_iso_or_now(None),
                lifecycle="running",
                active_job_id=active_job_id,
                active_job_kind=active_job_kind,  # type: ignore[arg-type]
            )
        return None
    try:
        mission = load_mission(json_path)
    except (MissionValidationError, Exception):  # noqa: BLE001
        # Audit fix #F: corrupt mission.json → surface as errored
        # rather than disappearing. Stub the fields we can't read.
        return MissionSummary(
            mission_slug=mission_slug,
            project_slug=project_slug,
            goal="(unreadable mission.json)",
            n_stages=0,
            current_stage_idx=0,
            decomposition_model="unknown",
            created_at=_parse_iso_or_now(None),
            lifecycle="errored",
            active_job_id=active_job_id,
            active_job_kind=active_job_kind,  # type: ignore[arg-type]
        )

    return MissionSummary(
        mission_slug=mission_slug,
        project_slug=project_slug,
        goal=mission.goal,
        n_stages=len(mission.stages),
        current_stage_idx=mission.current_stage_idx,
        decomposition_model=mission.decomposition_model,
        created_at=_parse_iso_or_now(mission.created_at),
        lifecycle=_derive_lifecycle(
            mission.stages, active_job_kind, mission.current_stage_idx,
        ),
        active_job_id=active_job_id,
        active_job_kind=active_job_kind,  # type: ignore[arg-type]
    )


def load_mission_detail(
    project_dir: Path,
    project_slug: str,
    mission_slug: str,
    *,
    active_job_kind: Optional[str] = None,
    active_job_id: Optional[str] = None,
) -> Optional[MissionDetail]:
    """Read mission.json and return the full detail view (stages
    included), or None if missing.

    §Ship 21a: when the mission dir exists with a `.decompose_pending`
    marker but mission.json hasn't been written yet (decompose still
    running), return a stub MissionDetail with empty stages and
    lifecycle="running". This keeps GET /missions/{slug} from 404'ing
    during the in-flight decompose window — the frontend's auto-open
    dialog needs a 200 response with active_job_kind=mission_decompose
    to render the "Planning" spinner instead of an error banner.
    """
    from sculptor.mission import load_mission, MissionValidationError

    json_path = mission_json_path(project_dir, mission_slug)
    if not json_path.is_file():
        # In-flight decompose: slug reserved, mission.json not yet written.
        pending_marker = mission_dir(project_dir, mission_slug) / ".decompose_pending"
        if pending_marker.is_file():
            return MissionDetail(
                mission_slug=mission_slug,
                project_slug=project_slug,
                goal="(decomposing — Claude is planning the curriculum)",
                n_stages=0,
                current_stage_idx=0,
                decomposition_model="",
                created_at=_parse_iso_or_now(None),
                lifecycle="running",
                active_job_id=active_job_id,
                active_job_kind=active_job_kind,  # type: ignore[arg-type]
                stages=[],
                decomposition_rationale="",
                schema_version=1,
            )
        return None
    try:
        mission = load_mission(json_path)
    except (MissionValidationError, Exception):  # noqa: BLE001
        return None

    return MissionDetail(
        mission_slug=mission_slug,
        project_slug=project_slug,
        goal=mission.goal,
        n_stages=len(mission.stages),
        current_stage_idx=mission.current_stage_idx,
        decomposition_model=mission.decomposition_model,
        created_at=_parse_iso_or_now(mission.created_at),
        lifecycle=_derive_lifecycle(
            mission.stages, active_job_kind, mission.current_stage_idx,
        ),
        active_job_id=active_job_id,
        active_job_kind=active_job_kind,  # type: ignore[arg-type]
        stages=_stages_to_schema(mission.stages),
        decomposition_rationale=mission.decomposition_rationale,
        schema_version=mission.schema_version,
        # §Ship 21a: surface persisted run_defaults so RunMissionDialog
        # can pre-fill its inputs.
        run_defaults=mission.run_defaults,
    )


def _parse_iso_or_now(s: Optional[str]) -> datetime:
    """Pydantic 2 wants a real datetime for the `created_at` field
    rather than a str. The on-disk `created_at` is an ISO-8601 string;
    parse it, defaulting to now() on any error so the endpoint never
    crashes on legacy mission.json files."""
    if not s:
        return datetime.now(timezone.utc)
    try:
        # `fromisoformat` handles `+00:00` suffix as well as naive.
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc)


# ── Delete ───────────────────────────────────────────────────────────
def _dir_size_bytes(p: Path) -> int:
    """Total bytes of all files under `p`. Used by DELETE to report
    `freed_bytes` to the caller (Ship 18a plan-review UX win)."""
    total = 0
    if not p.is_dir():
        return 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def delete_mission(
    project_dir: Path, mission_slug: str,
) -> int:
    """Hard-delete the mission directory. Returns freed bytes.

    Caller MUST verify there's no active mission_decompose /
    mission_execute job for this slug before calling — this function
    doesn't introspect the JobManager (separation of concerns; routes
    layer enforces).
    """
    target = mission_dir(project_dir, mission_slug)
    if not target.is_dir():
        return 0
    freed = _dir_size_bytes(target)
    shutil.rmtree(target, ignore_errors=False)
    return freed
