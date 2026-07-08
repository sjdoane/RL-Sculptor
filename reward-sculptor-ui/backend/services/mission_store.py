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
    <project_dir>/.missions/<mission_slug>/provenance.json
    <project_dir>/.missions/<mission_slug>/stage_metrics/<stage_name>/meta.json

This file does NOT manipulate stage subdirs — those are managed by
`mission_run` (Ship 16).

§mission-persistence increment 2: `load_mission_detail` no longer
trusts `mission.stages` alone. A pre-increment-1 redecompose spliced
the superseded parent OUT of `mission.stages` while leaving its
`stages/<name>/` directory (with real trained iterations) on disk —
orphaning it from every UI view. `_disk_union_stages` retroactively
restores those orphans as synthetic, `on_disk_only=True` stages so old
missions render their full history too, not just missions created
after increment 1 landed.
"""

from __future__ import annotations

import json
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
            # §keep-best (B1): which iter the stage kept + why.
            selected_iter_index=getattr(s, "selected_iter_index", None),
            selection_source=getattr(s, "selection_source", None),
            # §mission-persistence increment 2: mirrors increment 1's
            # Stage.failure_reason/failure_detail/steering_metric.
            failure_reason=getattr(s, "failure_reason", None),
            failure_detail=getattr(s, "failure_detail", None),
            steering_metric=getattr(s, "steering_metric", None),
        ))
    return out


def _derive_lifecycle(
    stages: list, active_job_kind: Optional[str], current_idx: int,
) -> MissionLifecycleStatus:
    """Compute the mission's lifecycle from on-disk stage statuses
    and any in-flight job kind.

    This is the source-of-truth contract: lifecycle is DERIVED, not
    stored, so stale UI state can never disagree with disk reality.

    §mission-persistence increment 2: "superseded" is terminal-but-
    benign — a stage that was replaced by re-decomposition children.
    It must NOT trip "halted" (that's reserved for a stage that
    actually FAILED and has no successor picking up the work) and
    must NOT block "completed" — a mission is complete when every
    NON-superseded stage succeeded. `stages` here may be either the
    raw dataclass list (has `.status`) or the disk-union StageSchema
    list (also has `.status`) — both work identically.
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
    live_statuses = [s for s in statuses if s != "superseded"]
    if live_statuses and all(s == "succeeded" for s in live_statuses):
        return "completed"
    if any(s == "failed" for s in live_statuses):
        return "halted"
    # Some pending / training / skipped → ready to run / resume. (All-
    # superseded with nothing else is a degenerate case — falls through
    # to "ready" too, which is honest: there's nothing left to run but
    # nothing outright failed either.)
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

    §mission-persistence increment 2: deliberately CHEAP — no disk-
    union scan (`_disk_union_stages` walks `stages/*` + reads
    provenance.json, an O(stages) filesystem pass). `n_stages` /
    `current_stage_idx` are based on `mission.stages` as-is, exactly
    as before this increment. This can under-count relative to what
    `load_mission_detail` returns for a mission with orphaned on-disk
    stages (old destructive-splice data) — the list view's stage count
    is a cheap approximation, the detail view is the source of truth.
    `lifecycle`, however, uses the SAME `_derive_lifecycle` as detail;
    since `mission.stages` never contains "superseded"-only orphans
    (those only appear via disk-union), lifecycle here is unaffected
    by the union and stays correct without the extra scan.
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


# ── §mission-persistence increment 2: disk-union + enrichment ────────
_REPLAN_CHILD_RE = re.compile(r"^(?P<parent>.+)__r(?P<attempt>\d+)_(?P<child>\d+)$")


def _find_replan_parent_index(
    extracted_parent: str, candidates: list[str],
) -> Optional[int]:
    """Resolve a replan child's extracted parent token `Q` (the
    `{Q}` in `f"{Q}__r<k>_<i>"`) against a list of preceding stage
    names, honoring truncation.

    `sculptor.decompose._truncate_for_name_budget` shortens the parent
    name when embedding it in a child name would blow the 32-char
    stage-name cap (mission._STAGE_NAME_RE), e.g. parent
    "crouch_load_absorb_and_transfer" -> child
    "crouch_load_absorb__r1_0" (Q = "crouch_load_absorb", a prefix of
    the real parent name, not the parent name itself). A naive
    `P.name == Q` match therefore misses truncated parents entirely.

    Match rule: a candidate P matches iff `P == Q` OR `P.startswith(Q)`
    (Q is a truncation of P). When multiple candidates match, prefer
    an EXACT match; among ties (multiple exact, or no exact and
    multiple prefix matches), prefer the NEAREST preceding one (i.e.
    the last / highest-index match in `candidates`, since `candidates`
    is in walk order and only entries preceding the child are passed
    in).

    The prefix branch is restricted to candidates that do NOT
    themselves look like a replan-child name (`_REPLAN_CHILD_RE`).
    Without this, two SIBLING children of the same truncated parent —
    e.g. "crouch_load_absorb__r1_0" and "crouch_load_absorb__r1_1",
    both truncated from "crouch_load_absorb_and_transfer" — would
    wrongly prefix-match each other (`"..._r1_0".startswith(Q)` is
    true for `Q = "crouch_load_absorb"`), resolving the second sibling
    as a CHILD of the first instead of both nesting under the real
    parent. A truncated Q can only ever be a prefix of a genuine
    (non-replan-pattern) base stage name.

    Returns the matching index into `candidates`, or None if nothing
    matches.
    """
    exact_idx: Optional[int] = None
    prefix_idx: Optional[int] = None
    for i, name in enumerate(candidates):
        if name == extracted_parent:
            exact_idx = i  # keep scanning — later exact match wins (nearest)
        elif name.startswith(extracted_parent) and not _REPLAN_CHILD_RE.match(name):
            prefix_idx = i  # keep scanning — later prefix match wins (nearest)
    return exact_idx if exact_idx is not None else prefix_idx


def _iter_dir_count(stage_dir: Path) -> int:
    """Count `runs/iter_*` subdirectories under a stage dir. Used as
    the synthetic stage's `iterations_used` when no mission.json
    record exists to read it from."""
    runs_root = stage_dir / "runs"
    if not runs_root.is_dir():
        return 0
    n = 0
    for d in runs_root.iterdir():
        if d.is_dir() and d.name.startswith("iter_"):
            n += 1
    return n


def _load_json_dict(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _load_provenance_stage_records(mission_dir_path: Path) -> dict[str, dict]:
    """Read `<mission_dir>/provenance.json` and return the LATEST
    stage record per stage name (`stages` is append-only — a stage
    that redecomposed / re-ran has multiple records; the most recent
    `recorded_at` / list-position wins). Returns {} on any missing /
    malformed file — provenance is best-effort enrichment, never a
    hard dependency."""
    doc = _load_json_dict(mission_dir_path / "provenance.json")
    if doc is None:
        return {}
    raw_stages = doc.get("stages")
    if not isinstance(raw_stages, list):
        return {}
    out: dict[str, dict] = {}
    for rec in raw_stages:
        if isinstance(rec, dict) and isinstance(rec.get("stage_name"), str):
            # Later entries overwrite earlier ones — list order IS
            # append order (record_stage_in_provenance appends).
            out[rec["stage_name"]] = rec
    return out


def _disk_union_stages(
    mission_dir_path: Path, schemas: list[StageSchema],
) -> list[StageSchema]:
    """Insert synthetic `on_disk_only=True` StageSchemas for any
    `<mission_dir>/stages/<name>/` directory that has NO entry among
    `schemas` — orphans left behind by the pre-increment-1 destructive
    splice (a redecomposed parent's directory stayed on disk with its
    trained iterations, but its `mission.stages` entry was removed).

    Insert position: immediately before the first stage in `schemas`
    whose name matches `f"{Q}__r<k>_<i>"` where `Q == dirname` OR
    `Q` is a truncation of `dirname` (`dirname.startswith(Q)`) — i.e.
    right before its own replan children, so the union reads as
    "parent, then its children" exactly like a mission that never had
    the old splice bug. `sculptor.decompose._truncate_for_name_budget`
    shortens long parent names before embedding them in a child name
    (32-char stage-name cap), so a truncated `Q` must still resolve
    back to this orphan. If no such child exists in `schemas`, append
    at the end.
    """
    stages_root = mission_dir_path / "stages"
    if not stages_root.is_dir():
        return list(schemas)

    known_names = {s.name for s in schemas}
    provenance = _load_provenance_stage_records(mission_dir_path)

    orphans: list[StageSchema] = []
    for d in sorted(stages_root.iterdir()):
        if not d.is_dir() or d.name in known_names:
            continue
        prov = provenance.get(d.name, {})
        goal_text = ""
        success_criterion = ""
        reward_seed_prompt = ""
        final_policy_path = prov.get("final_policy_path")
        final_reward_path = prov.get("final_reward_path")
        best_metric = prov.get("last_iter_metric")
        failure_reason = prov.get("failure_reason")
        failure_detail = prov.get("failure_detail")
        iterations_used = prov.get("iterations_used")
        if not isinstance(iterations_used, int):
            iterations_used = _iter_dir_count(d)
        orphans.append(StageSchema(
            name=d.name,
            goal_text=goal_text,
            success_criterion=success_criterion or "True",
            max_iterations=max(1, iterations_used),
            parent_stage=None,
            reward_seed_prompt=reward_seed_prompt or "(recovered from disk; "
            "original prompt not persisted in mission.json)",
            kg_seed_papers=[],
            status="superseded",
            final_policy_path=(
                final_policy_path if isinstance(final_policy_path, str) else None
            ),
            final_reward_path=(
                final_reward_path if isinstance(final_reward_path, str) else None
            ),
            best_metric=(
                float(best_metric) if isinstance(best_metric, (int, float)) else None
            ),
            iterations_used=iterations_used,
            failure_reason=(
                failure_reason if isinstance(failure_reason, str) else None
            ),
            failure_detail=(
                failure_detail if isinstance(failure_detail, str) else None
            ),
            on_disk_only=True,
        ))

    if not orphans:
        return list(schemas)

    out = list(schemas)
    for orphan in orphans:
        insert_at = len(out)
        for i, s in enumerate(out):
            m = _REPLAN_CHILD_RE.match(s.name)
            if m is None:
                continue
            extracted_parent = m.group("parent")
            # Q (extracted_parent) matches this orphan iff Q equals the
            # orphan's full name, or Q is a truncation of it (the
            # orphan's name starts with Q) — see _find_replan_parent_
            # index's docstring for the truncation rationale.
            if extracted_parent == orphan.name or orphan.name.startswith(
                extracted_parent,
            ):
                insert_at = i
                break
        out.insert(insert_at, orphan)
    return out


def _enrich_from_provenance(
    mission_dir_path: Path, stages: list[StageSchema],
) -> None:
    """Mutate `stages` in place: for every schema with
    `failure_reason is None`, look up its provenance record and fill
    `failure_reason` / `failure_detail` (+ `iterations_used` for
    synthetic on-disk-only stages that had no record at union time —
    a no-op in practice since `_disk_union_stages` already tried, but
    kept here so a REAL mission.stages entry with no failure_reason
    persisted — e.g. pre-increment-1 mission.json — also gets it).
    Tolerates missing/malformed provenance silently (already true of
    `_load_provenance_stage_records`).
    """
    provenance = _load_provenance_stage_records(mission_dir_path)
    if not provenance:
        return
    for s in stages:
        if s.failure_reason is not None:
            continue
        prov = provenance.get(s.name)
        if not prov:
            continue
        fr = prov.get("failure_reason")
        fd = prov.get("failure_detail")
        if isinstance(fr, str):
            s.failure_reason = fr
        if isinstance(fd, str):
            s.failure_detail = fd
        if s.on_disk_only and not s.iterations_used:
            iu = prov.get("iterations_used")
            if isinstance(iu, int):
                s.iterations_used = iu


def _assign_display_labels(stages: list[StageSchema]) -> None:
    """Mutate `stages` in place, setting `display_label` per the
    walk-in-order scheme: a stage matching `f"{Q}__r<k>_<i>"` for some
    EARLIER stage P in the (already fully unioned + ordered) list gets
    `f"{label(P)}.{i+1}"`, where P matches Q by exact name OR by Q
    being a truncation of P's name (see `_find_replan_parent_index`);
    every other stage increments a top-level counter `t` and gets
    `str(t)`.

    Every stage's OWN name -> label is registered as it's visited —
    including replan children — so a grandchild
    (`f"{child_name}__r<k>_<i>"`) resolves its parent recursively and
    nests correctly (e.g. "1.2.1"), not just direct children of a
    top-level stage.

    Orphan replan children with no parent present in `stages` (old
    data where even the __r0 sibling's parent vanished) fall back to
    grouping consecutive same-prefix orphans under one incremented
    `t` as `t.1`, `t.2`, ... — this only matters for pathological
    hand-edited / doubly-corrupted mission.json; normal union output
    always has the parent present (either real or synthetic).
    """
    label_by_name: dict[str, str] = {}
    seen_names: list[str] = []
    t = 0
    pending_orphan_prefix: Optional[str] = None
    pending_orphan_label = 0

    for s in stages:
        m = _REPLAN_CHILD_RE.match(s.name)
        parent_name = m.group("parent") if m else None
        child_idx = int(m.group("child")) if m else None

        parent_idx = (
            _find_replan_parent_index(parent_name, seen_names)
            if parent_name is not None else None
        )

        if parent_idx is not None:
            resolved_parent_name = seen_names[parent_idx]
            s.display_label = (
                f"{label_by_name[resolved_parent_name]}.{child_idx + 1}"
            )
            pending_orphan_prefix = None
            seen_names.append(s.name)
            label_by_name[s.name] = s.display_label
            continue

        if parent_name is not None:
            # Orphan replan child: no matching parent in the list.
            if pending_orphan_prefix == parent_name:
                pending_orphan_label += 1
            else:
                t += 1
                pending_orphan_prefix = parent_name
                pending_orphan_label = 1
            s.display_label = f"{t}.{pending_orphan_label}"
            seen_names.append(s.name)
            label_by_name[s.name] = s.display_label
            continue

        t += 1
        pending_orphan_prefix = None
        s.display_label = str(t)
        seen_names.append(s.name)
        label_by_name[s.name] = s.display_label


def _stage_metric_status(
    mission_dir_path: Path, stage: StageSchema,
) -> Optional[str]:
    """Derive `metric_status` for one stage:
      "accepted"  — stage_metrics/<name>/meta.json exists and
                    `accepted` is true.
      "rejected"  — the dir/meta.json exists but `accepted` is false
                    (or missing/malformed — a dir with no readable
                    verdict is treated as a rejected attempt, not a
                    silent "none").
      "inherited" — no stage_metrics dir at all, but the stage carries
                    a `steering_metric` (set some other way, e.g. hand-
                    authored or inherited from a mission-level default
                    that got baked in).
      "none"      — no dir, no steering_metric.
    """
    meta_path = mission_dir_path / "stage_metrics" / stage.name / "meta.json"
    if meta_path.is_file():
        meta = _load_json_dict(meta_path)
        if meta is not None and meta.get("accepted") is True:
            return "accepted"
        return "rejected"
    if stage.steering_metric:
        return "inherited"
    return "none"


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

    stages = build_unioned_stage_schemas(project_dir, mission_slug, mission)

    return MissionDetail(
        mission_slug=mission_slug,
        project_slug=project_slug,
        goal=mission.goal,
        # §mission-persistence increment 2: n_stages / current_stage_idx
        # stay based on `mission.stages` (the orchestrator's own
        # bookkeeping), NOT the disk-union count — see docstring note
        # on `load_mission_summary` for why the union is detail-only.
        n_stages=len(mission.stages),
        current_stage_idx=mission.current_stage_idx,
        decomposition_model=mission.decomposition_model,
        created_at=_parse_iso_or_now(mission.created_at),
        lifecycle=_derive_lifecycle(
            stages, active_job_kind, mission.current_stage_idx,
        ),
        active_job_id=active_job_id,
        active_job_kind=active_job_kind,  # type: ignore[arg-type]
        stages=stages,
        decomposition_rationale=mission.decomposition_rationale,
        schema_version=mission.schema_version,
        # §Ship 21a: surface persisted run_defaults so RunMissionDialog
        # can pre-fill its inputs.
        run_defaults=mission.run_defaults,
    )


def build_unioned_stage_schemas(
    project_dir: Path, mission_slug: str, mission: Any,
) -> list[StageSchema]:
    """The full disk-union + enrichment + display-label + metric-status
    pipeline, factored out of `load_mission_detail` so
    `routes/runs.py`'s disk-reconstructed run rows (§step 3) can reuse
    the SAME unioned/ordered stage list without duplicating the union
    algorithm. `mission` is a loaded `sculptor.mission.Mission`.
    """
    mission_dir_path = mission_dir(project_dir, mission_slug)
    stages = _stages_to_schema(mission.stages)
    stages = _disk_union_stages(mission_dir_path, stages)
    _enrich_from_provenance(mission_dir_path, stages)
    _assign_display_labels(stages)
    for s in stages:
        s.metric_status = _stage_metric_status(mission_dir_path, s)  # type: ignore[assignment]
    return stages


def iter_unioned_stages_for_project(
    project_dir: Path,
) -> list[tuple[str, Any, list[StageSchema]]]:
    """`[(mission_slug, mission, unioned_stages), ...]` for every
    mission under `project_dir` that has a readable `mission.json`.
    Used by `routes/runs.py::list_runs` (§mission-persistence
    increment 2 step 3) to synthesize disk-reconstructed run rows
    without duplicating `build_unioned_stage_schemas`'s union
    algorithm. Missions with a corrupt/unreadable mission.json (or
    still mid-decompose, `.decompose_pending` only) are silently
    skipped — `list_runs` degrading to "fewer synthetic rows" beats
    a 500 on an unrelated corrupt mission.
    """
    from sculptor.mission import load_mission, MissionValidationError

    out: list[tuple[str, Any, list[StageSchema]]] = []
    for mission_slug in list_mission_slugs(project_dir):
        json_path = mission_json_path(project_dir, mission_slug)
        if not json_path.is_file():
            continue
        try:
            mission = load_mission(json_path)
        except (MissionValidationError, Exception):  # noqa: BLE001
            continue
        stages = build_unioned_stage_schemas(project_dir, mission_slug, mission)
        out.append((mission_slug, mission, stages))
    return out


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
    """Move the mission directory into the trash (chunk A1 — no
    longer a hard delete; see `backend.services.trash`). Returns freed
    bytes (measured before the move, same meaning as before).

    Caller MUST verify there's no active mission_decompose /
    mission_execute job for this slug before calling — this function
    doesn't introspect the JobManager (separation of concerns; routes
    layer enforces).
    """
    target = mission_dir(project_dir, mission_slug)
    if not target.is_dir():
        return 0
    freed = _dir_size_bytes(target)
    from backend.services import trash

    trash.move_to_trash(
        target, kind="mission", slug=mission_slug, origin=target,
    )
    return freed
