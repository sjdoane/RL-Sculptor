"""sculptor/mission.py — multi-stage curriculum data model (Ship 14).

A `Mission` is a complex behavior goal decomposed by Claude into an ordered
sequence of `Stage`s. Each stage is a self-contained mini-project with its
own reward seed prompt, success criterion, and optional warm-start pointer
to a parent stage's final policy.

Ship 14 ONLY lands the data model + serialization + validation. The actual
decomposition call (Claude → Mission) lives in `sculptor.decompose`. The
orchestrator that trains through stages lives in a later ship.

Design references:
  - CurricuLLM (arXiv:2409.18382) — LLM-generated subtask curricula with
    per-stage reward+goal+warm-start, validated on Berkeley Humanoid.
  - Voyager (arXiv:2305.16291) — skill-library pattern that Ship 19 will
    extend Mission to participate in.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


SCHEMA_VERSION = 1

# §start_pose: the physical configuration the robot is in at a stage's
# episode start. None = legacy/unspecified (older mission.json files, or
# a decomposer that omitted the field) — treated as "no opinion", NOT as
# "standing"; the force-rule below only fires on an EXPLICIT non-standing
# value, never on None.
START_POSE_VALUES: frozenset[str] = frozenset(
    {"supine", "prone", "sitting", "crouched", "standing"})

# Valid Stage names: snake_case identifier, ≤ 32 chars. Used as the
# reward-component-namespace prefix, on-disk subdirectory name, and
# parent-reference key — must be safe for all of those.
_STAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Stage lifecycle. "pending" = awaits orchestrator; "training" = sculpt_run
# in-flight; "succeeded"/"failed" = terminal. "skipped" is reserved for
# Ship 17's fail-through behavior (a failing stage whose successors still
# try to run from scratch). "superseded" (mission-persistence increment 1):
# terminal, never runnable — the stage was replaced by re-decomposition
# children but is RETAINED in `mission.stages` (rather than spliced out)
# so its trained iterations stay visible in every UI view / report. A
# superseded stage is permanently excluded from the runnable chain: the
# orchestrator's while-loop never executes it, and downstream parent
# references are always re-pointed past it to the last child.
StageStatus = Literal[
    "pending", "training", "succeeded", "failed", "skipped", "superseded",
]

# Matches the existing reward-prompt endpoint's char bounds so a stage's
# seed prompt drops into `apply_prompt_edit` without adjustment.
MIN_SEED_PROMPT_CHARS = 3
MAX_SEED_PROMPT_CHARS = 2000


class MissionValidationError(ValueError):
    """Raised when a Mission / Stage payload fails structural validation.

    Separate from `EditValidationError` (which is about reward-edit pre-
    flight) so callers can distinguish the two failure surfaces.
    """


@dataclass
class Stage:
    """One step of a mission curriculum.

    The tuple (name, goal_text, success_criterion, reward_seed_prompt)
    is what Claude produces at decomposition time. The remaining fields
    (status, final_policy_path, ...) are runtime bookkeeping populated
    by the mission orchestrator during / after training.
    """

    # ── Claude-authored at decompose time ────────────────────────────
    name: str
    goal_text: str
    success_criterion: str
    max_iterations: int
    parent_stage: Optional[str]
    reward_seed_prompt: str
    kg_seed_papers: list[str] = field(default_factory=list)
    # Ship 19: optional cross-mission skill-library reference. When
    # set by Claude in `decompose_task` (and validated against the
    # available skill ids in the rendered context block), the
    # orchestrator threads the resolved checkpoint into iter-0's
    # `init_policy_path` for warm-start. Backward-compatible: older
    # mission.json files load with init_skill_id=None via the
    # filter-unknown-keys path in `from_dict`.
    init_skill_id: Optional[str] = None
    # §Ship 38: optional PER-STAGE objective fitness metric — a built-in
    # spec-metric name (e.g. "g1_kick") or a resolved generated-metric .py
    # path. When set it OVERRIDES the mission-level fitness_metric for THIS
    # stage; this is what makes a true multi-phase curriculum sound (a
    # "balance on one leg" stage and a "kick" stage want DIFFERENT
    # objectives — Ship 34's uniform-metric-per-mission could not). None →
    # the uniform mission metric. Backward-compatible: older mission.json
    # without it load with steering_metric=None via from_dict's filter.
    steering_metric: Optional[str] = None
    # §JUMP_SCAFFOLD: DeepMimic-style reference-state initialization for
    # hard-exploration stages (jump launch/flight/landing). When true the
    # orchestrator derives a validated train-only RSI curriculum from a
    # reference clip (project clip if present, procedural jump otherwise)
    # and applies it to this stage's env spec before training. Backward-
    # compatible: older mission.json load with False via from_dict.
    needs_reference_rsi: bool = False
    # §R1_BUILD_SPEC decision 10: the reference-library clip (if any)
    # ATTACHED to this stage via `POST .../stages/{stage}/reference` —
    # distinct from `needs_reference_rsi` (which only says the stage
    # WANTS an RSI curriculum; the orchestrator falls back to the
    # procedural jump clip when no library clip is attached).
    # `reference_clip_id` is the library clip id (`sculptor.refs.library`
    # clip_id charset); `reference_tier` mirrors that clip's provenance
    # `tier` at attach time (cheap display without a second lookup);
    # `reference_match_confidence` carries the retrieval match_confidence
    # from `refs.retrieve.search` when the clip was attached via a
    # search result (None for a manual/direct attach, or when the
    # deterministic-only layer produced the match). None on stages with
    # no reference attached. Backward-compatible: older mission.json
    # without these keys load with None via from_dict's filter-unknown-
    # keys path, same guarantee `needs_reference_rsi` already relies on.
    reference_clip_id: Optional[str] = None
    reference_tier: Optional[str] = None
    reference_match_confidence: Optional[float] = None
    # §D24 F1 (docs/internal/REFERENCE_BUILD_LOG.md D23/D24): the goal-
    # aligned SUB-SPAN of `reference_clip_id`, selected by
    # `sculptor.refs.spans.select_reference_span`. D23 diagnosed a live
    # zero-fitness regression when a stage's goal is a strict sub-phase
    # of a longer attached clip (e.g. "sit up" is a sub-phase of a full
    # lying-to-standing get-up) and certification/RSI/eval-reset all ran
    # against the FULL clip — a physically correct sit-up scored zero
    # because passing certification against the full clip's own
    # truncation negatives REQUIRED zeroing exactly that motion.
    # `reference_span_start_s`/`_end_s` are the snapped-and-QC'd crop
    # window (seconds, clip-relative); `reference_span_confidence` is
    # the LLM's reported confidence in [0, 1]; `reference_span_method`
    # is `"llm+snap+qc"` (see `select_reference_span`'s docstring) or
    # None. All four are None together whenever no span applies: no
    # `reference_clip_id` attached, the goal covers the whole clip, or
    # selection was declined (low confidence / failed mechanical or
    # end-state QC / LLM unavailable) — a None span means "use the FULL
    # clip", never a partial/garbage crop. Every consumer of
    # `reference_clip_id` MUST resolve it through
    # `sculptor.mission_metrics.load_stage_reference_clip` (the one
    # loader) rather than cropping independently — §D19's "every
    # clip-shape assumption in ONE place" rule. Redecompose sub-stages
    # unconditionally inherit `reference_clip_id`/`_tier`/
    # `_match_confidence` from the failed stage (D21) but NEVER these
    # four fields — a new sub-goal needs its own span, freshly
    # (re-)selected against ITS OWN goal text. Backward-compatible:
    # older mission.json files load with all four None via
    # `from_dict`'s filter-unknown-keys path.
    reference_span_start_s: Optional[float] = None
    reference_span_end_s: Optional[float] = None
    reference_span_confidence: Optional[float] = None
    reference_span_method: Optional[str] = None
    # §start_pose: the physical configuration the robot is in at THIS
    # stage's episode start. One of `START_POSE_VALUES` (supine, prone,
    # sitting, crouched, standing), or None (unspecified — legacy
    # missions / a decomposer that omitted the field; NOT the same as
    # "standing", just "no opinion recorded"). Claude sets this in
    # `decompose_task`/`redecompose_stage` from the mission goal + this
    # stage's semantics; sub-stages of the SAME mission may carry
    # DIFFERENT start poses as a get-up motion progresses (e.g.
    # supine -> crouched -> standing across stages). `validate_mission`
    # enforces the DETERMINISTIC FORCE RULE: any non-"standing"
    # start_pose forces `needs_reference_rsi=True` regardless of what
    # was authored — a non-standing episode start is untrainable
    # without a reference-derived reset, and prompt compliance on
    # `needs_reference_rsi` is not trusted on its own (§sculpt.py's
    # scaffold additionally QCs the ATTACHED CLIP's measured shape
    # against this value via `sculptor.reference
    # .check_start_pose_compatibility` — this field only says what the
    # stage WANTS, not that a compatible clip is actually attached).
    # Backward-compatible: older mission.json files without this key
    # load with start_pose=None via `from_dict`'s filter-unknown-keys
    # path.
    start_pose: Optional[str] = None

    # ── Runtime-populated by orchestrator ────────────────────────────
    status: StageStatus = "pending"
    final_policy_path: Optional[str] = None
    final_reward_path: Optional[str] = None
    best_metric: Optional[float] = None
    iterations_used: int = 0
    # ISO-8601 timestamps; None when not yet reached that transition.
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Ship 17: re-decomposition budget. 0 = stage may be re-decomposed
    # once on criterion failure; ≥1 = stage was BORN from a re-
    # decomposition (or already used its budget) and cannot be split
    # further. Bounded at one level per CurricuLLM guidance — deeper
    # trees would invite combinatorial fanout on pathological tasks.
    redecomposition_attempts: int = 0
    # §Ship 20a: persisted cap actually enforced for this stage's last
    # (or current) run — `iterations_override or max_iterations`.
    # Persisted so the UI's `rounds X/Y` display stays correct AFTER
    # the WS event window slides past `stage_started`. Ship 20 derived
    # this from events alone; long missions evict those events and the
    # dialog fell back to authored `max_iterations`, showing nonsense
    # like `rounds 4/3` when the user actually capped at 5. None for
    # stages that haven't run yet; backward-compatible via Stage.from_
    # dict's filter-unknown-keys path.
    effective_max_iterations: Optional[int] = None
    # §keep-best finalization (B1): which iteration this stage actually
    # KEPT as its final policy, and why. The stage no longer finalizes on
    # the LAST iter — it selects the best iter whose rollout satisfies the
    # criterion (highest fitness), so a late regression (e.g. a jump stage
    # that collapses to standing) can't discard the good policy. None on
    # stages that predate this / haven't run. Backward-compatible via
    # from_dict's filter-unknown-keys path.
    selected_iter_index: Optional[int] = None
    #: "criterion+fitness" | "criterion_newest" | "fitness_fallback" | "last"
    selection_source: Optional[str] = None
    # §mission-persistence increment 1: the failure's short reason code
    # (e.g. "criterion_not_met", "no_checkpoint", "training_errored" —
    # same vocabulary as `StageResult.failure_reason` /
    # `_REDECOMPOSABLE_REASONS` in sculpt.py) and its free-form detail
    # message, persisted onto the stage itself. Previously this only
    # lived in the ephemeral `StageResult` (mission_runtime.py) and in
    # provenance.json — never in mission.json — so it was lost the
    # moment a stage was superseded/spliced-out or the process exited.
    # None on stages that never failed. Backward-compatible: older
    # mission.json files without these keys load with None via
    # `from_dict`'s filter-unknown-keys path.
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stage":
        known = {f.name for f in dataclasses.fields(cls)}
        # Drop unknown keys so forward-compat schema tweaks don't hard-
        # crash older readers; log via the caller if they care.
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class Mission:
    """An ordered curriculum toward a complex behavior goal.

    Stages are topologically ordered — `stages[i].parent_stage`, if set,
    must name a `stages[j]` with `j < i`. `_validate` enforces this.
    """

    goal: str                          # the user's original complex goal
    stages: list[Stage]
    decomposition_model: str           # e.g. "claude-opus-4-7"
    decomposition_rationale: str       # why Claude chose this decomposition
    created_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION
    current_stage_idx: int = 0
    # Optional on-disk anchor — set by the orchestrator when persisting.
    mission_dir: Optional[str] = None
    # §Ship 21a: persisted run-time defaults set at mission-creation
    # time via the NewMissionDialog Advanced tab. Stored as a free-form
    # dict to keep Mission decoupled from the backend's
    # RunMissionRequest pydantic shape — valid keys are the same as
    # that shape (iterations_override, steps_per_iter, seed,
    # early_stop_on_criterion, criterion_stability_window,
    # extend_on_improvement, max_extensions_per_stage,
    # extension_factor, extension_improvement_threshold). RunMission
    # Dialog pre-fills from these when the user later clicks Run
    # mission. Backward-compatible: older mission.json without this
    # field loads with run_defaults=None via from_dict's filter-
    # unknown-keys path.
    run_defaults: Optional[dict[str, Any]] = None

    # ── Serialization ────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        # Ship 18a path-relocation fix: do NOT persist `mission_dir`.
        # It's an absolute path that becomes stale the moment the
        # project (or its parent dir) is moved. Callers reconstruct
        # `mission_dir` from the JSON file's location at load time
        # (see `load_mission`). In memory, `mission_dir` is set so
        # downstream code (`stage_dir`, `parent_checkpoint_of`) keeps
        # working without changes.
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "decomposition_model": self.decomposition_model,
            "decomposition_rationale": self.decomposition_rationale,
            "created_at": self.created_at,
            "current_stage_idx": self.current_stage_idx,
            "stages": [s.to_dict() for s in self.stages],
        }
        # §Ship 21a: emit run_defaults only when set, so older mission
        # readers (and the UI's MissionDetail expecting Optional[dict])
        # see a clean None.
        if self.run_defaults is not None:
            out["run_defaults"] = dict(self.run_defaults)
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mission":
        ver = int(data.get("schema_version", 1))
        if ver > SCHEMA_VERSION:
            raise MissionValidationError(
                f"mission schema_version={ver} is newer than this "
                f"sculptor build's SCHEMA_VERSION={SCHEMA_VERSION}; "
                f"upgrade sculptor or downgrade the mission file."
            )
        stages = [Stage.from_dict(s) for s in data.get("stages", [])]
        run_defaults_raw = data.get("run_defaults")
        run_defaults = (
            dict(run_defaults_raw) if isinstance(run_defaults_raw, dict) else None
        )
        return cls(
            goal=data["goal"],
            stages=stages,
            decomposition_model=data["decomposition_model"],
            decomposition_rationale=data["decomposition_rationale"],
            created_at=data.get(
                "created_at",
                _dt.datetime.now(_dt.timezone.utc).isoformat(),
            ),
            schema_version=ver,
            current_stage_idx=int(data.get("current_stage_idx", 0)),
            mission_dir=data.get("mission_dir"),
            run_defaults=run_defaults,
        )

    @classmethod
    def from_json(cls, text: str) -> "Mission":
        return cls.from_dict(json.loads(text))

    # ── Lookup helpers ───────────────────────────────────────────────
    def stage_by_name(self, name: str) -> Optional[Stage]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def stage_index(self, name: str) -> int:
        for i, s in enumerate(self.stages):
            if s.name == name:
                return i
        raise KeyError(f"no stage named {name!r} in mission")

    # ── Ship 16 disk-layout helpers ──────────────────────────────────
    def stage_dir(self, stage_name: str) -> Path:
        """Return the on-disk project directory for a stage.

        Requires `self.mission_dir` to be set (via `save_mission` or an
        explicit assignment). Layout:
            <mission_dir>/stages/<stage_name>/

        Raises if mission_dir is None or the named stage doesn't exist
        on this mission.
        """
        if self.mission_dir is None:
            raise RuntimeError(
                "Mission.mission_dir is None — call save_mission first "
                "or set mission_dir manually before resolving stage paths."
            )
        if self.stage_by_name(stage_name) is None:
            raise KeyError(f"no stage named {stage_name!r} in mission")
        return Path(self.mission_dir) / "stages" / stage_name

    def parent_checkpoint_of(self, stage_name: str) -> Optional[Path]:
        """Return the parent stage's `final_policy_path` (as Path) if
        the parent exists AND has a persisted final policy; None
        otherwise. THIS METHOD COLLAPSES TWO CASES — use
        `parent_checkpoint_status_of` if you need to distinguish
        "no parent / parent untrained" from "parent's checkpoint
        was deleted externally" (the latter silently degrades to
        cold-start, which Ship 16's orchestrator surfaces as a
        warm_start_skipped event).
        """
        path, _ = self.parent_checkpoint_status_of(stage_name)
        return path

    def parent_checkpoint_status_of(
        self, stage_name: str,
    ) -> tuple[Optional[Path], str]:
        """Like `parent_checkpoint_of` but returns (path, status_tag).

        status_tag ∈ {
            "no_parent"          → stage.parent_stage is None,
            "parent_untrained"   → parent exists but has no final_policy_path,
            "parent_ckpt_missing" → parent had a path but the file is gone,
            "ok"                 → path resolved to a real file.
        }
        Used by the Ship 16 orchestrator to emit a clear warm-start
        skip event when a previously-trained parent's checkpoint has
        been deleted (audit finding: silent cold-start regression).
        """
        stage = self.stage_by_name(stage_name)
        if stage is None:
            raise KeyError(f"no stage named {stage_name!r} in mission")
        if stage.parent_stage is None:
            return None, "no_parent"
        parent = self.stage_by_name(stage.parent_stage)
        if parent is None or parent.final_policy_path is None:
            return None, "parent_untrained"
        p = Path(parent.final_policy_path)
        if not p.is_file():
            return None, "parent_ckpt_missing"
        return p, "ok"


# ── Persistence helpers ──────────────────────────────────────────────
def save_mission(mission: Mission, path: Path | str) -> Path:
    """Write a Mission to `<path>/mission.json` (if path is a dir) or
    directly to `path` (if it ends with .json). Creates parents as needed."""
    p = Path(path)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "mission.json"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(mission.to_json(), encoding="utf-8")
    return p


def load_mission(path: Path | str) -> Mission:
    """Load a Mission from `<path>/mission.json` or `<path>.json`.

    Ship 18a: reconstructs `mission.mission_dir` from the FILE's
    location (parent directory of the .json file). This makes the
    mission relocation-safe — copying / moving the project tree
    doesn't break stage-dir / parent-checkpoint resolution.

    Audit caveat (#D): `Path.resolve()` follows symlinks. If
    `mission.json` is itself a symlink to a file outside the
    `.missions/<slug>/` directory, `mission_dir` resolves to the
    LINK TARGET's parent — not the conceptual mission directory.
    Don't symlink mission.json. Tested workflow: file lives at
    `<project_dir>/.missions/<slug>/mission.json`, no symlinks.
    """
    p = Path(path)
    if p.is_dir():
        json_path = p / "mission.json"
    else:
        json_path = p
    mission = Mission.from_json(json_path.read_text(encoding="utf-8"))
    # The mission lives at `<mission_dir>/mission.json` per Ship 16
    # disk layout. Reconstruct the directory from the file's parent.
    mission.mission_dir = str(json_path.resolve().parent)
    return mission


# ── Validation ───────────────────────────────────────────────────────
def validate_mission(
    mission: Mission,
    *,
    info_keys: set[str],
) -> None:
    """Raise MissionValidationError on any structural issue.

    Checks:
      * ≥ 1 stage.
      * Every stage name is a valid snake_case identifier.
      * Stage names are unique.
      * First stage has parent_stage=None.
      * Every non-None parent_stage names an EARLIER stage (topological
        order; no forward / cycle references).
      * Every reward_seed_prompt is within the existing prompt-edit
        char limits.
      * Every success_criterion parses as a Python expression.
      * Every `info['<key>']` inside a success_criterion references a
        key in `info_keys`. (Bare identifier grounding is deferred to
        Ship 16 runtime — we don't know component names at decompose
        time since the reward module hasn't been materialized yet.)
      * max_iterations ∈ [1, 50].

    `info_keys` is the adapter's `expected_info_keys` set.

    §mission-persistence increment 1: a "superseded" stage (retained in
    `mission.stages` after being replaced by re-decomposition children —
    see `sculpt._maybe_redecompose_and_splice`) is validated exactly
    like any other stage. None of the checks above branch on
    `stage.status`; a superseded stage is structurally unchanged from
    when it was authored (same name/prompt/criterion/parent), it is
    simply excluded from the RUNNABLE chain at orchestration time. The
    one topological rule that matters — no OTHER stage may declare a
    superseded stage as its `parent_stage` — is enforced not here but
    by construction: `_repoint_downstream_children` always re-points
    downstream parents from the superseded name to its last child
    before the splice is validated, so a lingering reference to a
    superseded stage's name would already fail
    `_validate_parent_reference`'s ordinary "must name an earlier
    stage" check (the name still exists, just structurally orphaned —
    no consumer is expected to reference it going forward).
    """
    if not mission.stages:
        raise MissionValidationError("mission has no stages")

    seen_names: set[str] = set()
    for i, stage in enumerate(mission.stages):
        _validate_stage_structure(stage, i, seen_names)
        _validate_parent_reference(stage, i, mission.stages)
        _validate_success_criterion(stage, info_keys)
        seen_names.add(stage.name)


def _validate_stage_structure(
    stage: Stage, idx: int, seen_names: set[str]
) -> None:
    if not _STAGE_NAME_RE.match(stage.name):
        raise MissionValidationError(
            f"stage[{idx}].name={stage.name!r} must match "
            f"^[a-z][a-z0-9_]{{0,31}}$ (snake_case, letter-first, ≤32 chars)"
        )
    if stage.name in seen_names:
        raise MissionValidationError(
            f"stage[{idx}].name={stage.name!r} duplicates an earlier stage"
        )
    if not (MIN_SEED_PROMPT_CHARS <= len(stage.reward_seed_prompt)
            <= MAX_SEED_PROMPT_CHARS):
        raise MissionValidationError(
            f"stage[{idx}].reward_seed_prompt length "
            f"{len(stage.reward_seed_prompt)} not in "
            f"[{MIN_SEED_PROMPT_CHARS}, {MAX_SEED_PROMPT_CHARS}]"
        )
    if not (1 <= stage.max_iterations <= 50):
        raise MissionValidationError(
            f"stage[{idx}].max_iterations={stage.max_iterations} not in [1, 50]"
        )
    if not stage.goal_text.strip():
        raise MissionValidationError(
            f"stage[{idx}].goal_text is empty"
        )
    # §Ship 38: per-stage steering metric, when present, must be a sane
    # non-empty string (a spec-metric name or a generated-metric path). The
    # KNOWN-name check happens at decompose time (which has spec_metric_names);
    # here we only guard against empty / pathologically-long hand edits.
    if stage.steering_metric is not None:
        sm = str(stage.steering_metric).strip()
        if not sm or len(sm) > 128:
            raise MissionValidationError(
                f"stage[{idx}].steering_metric must be a non-empty string "
                f"≤128 chars (a spec-metric name or generated-metric path); "
                f"got {stage.steering_metric!r}"
            )
    # §start_pose: unknown values are a hard validation error (typos /
    # LLM drift must not silently pass through as "no opinion" — that
    # would be indistinguishable from None). A valid, non-"standing"
    # value that arrives with `needs_reference_rsi=False` is NOT an
    # error — it is FORCED true here (mutating the stage in place) since
    # prompt compliance on that separate boolean field is not trusted;
    # the force is disclosed via `warnings.warn` (this module has no
    # existing structured-notice channel — `warnings` is the standard
    # library's own "log/warn without changing the return contract"
    # mechanism, and `validate_mission` must keep returning None / raise
    # for every OTHER violation).
    if stage.start_pose is not None:
        if stage.start_pose not in START_POSE_VALUES:
            raise MissionValidationError(
                f"stage[{idx}].start_pose={stage.start_pose!r} must be "
                f"one of {sorted(START_POSE_VALUES)} or None"
            )
        if stage.start_pose != "standing" and not stage.needs_reference_rsi:
            warnings.warn(
                f"stage {stage.name!r} has start_pose="
                f"{stage.start_pose!r} (non-standing) but "
                f"needs_reference_rsi=False — forcing "
                f"needs_reference_rsi=True at validation. A non-standing "
                f"episode start is untrainable without a reference-"
                f"derived reset; prompt compliance on needs_reference_rsi "
                f"is not trusted on its own.",
                stacklevel=2,
            )
            stage.needs_reference_rsi = True


def _validate_parent_reference(
    stage: Stage, idx: int, stages: list[Stage]
) -> None:
    if stage.parent_stage is None:
        # First stage must have no parent; later stages MAY omit a
        # parent if they legitimately cold-start (e.g., an exploration
        # branch unrelated to prior skills).
        return
    if idx == 0:
        raise MissionValidationError(
            f"stage[0] (first) has parent_stage={stage.parent_stage!r}; "
            f"the first stage must cold-start (parent_stage=None)"
        )
    earlier_names = {s.name for s in stages[:idx]}
    if stage.parent_stage not in earlier_names:
        raise MissionValidationError(
            f"stage[{idx}] ({stage.name!r}) parent_stage="
            f"{stage.parent_stage!r} does not name an earlier stage. "
            f"Known earlier: {sorted(earlier_names)}"
        )


def _validate_success_criterion(stage: Stage, info_keys: set[str]) -> None:
    """Validate stage.success_criterion against the mission-runtime
    namespace (sculptor.mission_runtime).

    Ship 14 originally validated `info[<key>]` against the adapter's
    `expected_info_keys`. Ship 16 revealed a mismatch: those keys
    describe the per-step `info` dict at TRAINING time, but success
    criteria execute at ROLLOUT time against persisted trajectory.npz
    + behavior.json artifacts — a different (hardcoded) key set.
    `info_keys` is preserved as a parameter for API compatibility but
    ignored; validation now uses the persisted keys the runtime
    actually exposes.

    Checked:
      * `info['<key>']` / `trajectory['<key>']` subscripts use keys in
        `PERSISTED_TRAJECTORY_KEYS` (runtime-persisted array names).
      * `behavior['<key>']` subscripts use keys in `BEHAVIOR_KEYS`
        (stable behavior.json schema).
      * `components['<name>']` subscripts are NOT statically checked —
        names depend on the stage's reward_seed_prompt which hasn't
        materialized at decompose time (deferred to runtime).

    Bare identifiers (beyond `metric`, `min`, `max`, etc.) are also
    deferred — same rationale as Ship 14's component-name allowance.
    """
    import ast

    # Imported here (not at module top) to avoid a circular import:
    # mission.py ← mission_runtime.py ← … (later).
    from sculptor.mission_runtime import (
        BEHAVIOR_KEYS,
        PERSISTED_TRAJECTORY_KEYS,
    )

    crit = stage.success_criterion.strip()
    if not crit:
        raise MissionValidationError(
            f"stage {stage.name!r} has empty success_criterion"
        )
    try:
        tree = ast.parse(crit, mode="eval")
    except SyntaxError as e:
        raise MissionValidationError(
            f"stage {stage.name!r} success_criterion is not a valid "
            f"Python expression: {e}"
        ) from e

    # §Ship 21c: torch-idiom guard. The criterion namespace is built
    # from numpy arrays (trajectory.npz) + plain dicts (behavior.json).
    # numpy bool/int arrays do NOT have torch tensor methods like
    # `.float()`, `.long()`, `.cpu()`, `.detach()` — using those crashes
    # AT EVAL TIME after a stage has already burned its training
    # budget (10+ hours on G1). Catch at decompose time instead.
    # Sam's robot-flossing run: stage failed at the very last criterion
    # evaluation with `'numpy.ndarray' object has no attribute 'float'`
    # after iter 5 had nudged the metric to its peak — would have
    # succeeded if the criterion didn't use `.float()`.
    _FORBIDDEN_TORCH_METHODS: tuple[str, ...] = (
        "float", "long", "double", "int", "bool", "byte", "short",
        "half", "to", "cpu", "cuda", "detach", "item", "numpy",
        "requires_grad", "requires_grad_", "grad",
    )
    method_violations: list[str] = []
    for node in ast.walk(tree):
        # x.method(...) → Call(func=Attribute(value=x, attr='method'))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FORBIDDEN_TORCH_METHODS
        ):
            method_violations.append(node.func.attr)
        # x.requires_grad (no parens) — Attribute access without Call
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in ("requires_grad", "grad")
        ):
            method_violations.append(node.attr)
    if method_violations:
        unique = sorted(set(method_violations))
        raise MissionValidationError(
            f"stage {stage.name!r} success_criterion uses torch tensor "
            f"method(s) {unique!r} — namespace is numpy, NOT torch. "
            f"For numpy arrays use `.astype(float)` / `.mean()` / "
            f"`.any()` / `.all()` directly. A bool array's `.mean()` "
            f"already returns the fraction-True; no cast needed.\n"
            f"  bad:  (trajectory['root_height'] > 0.65).float().mean()\n"
            f"  good: (trajectory['root_height'] > 0.65).mean()"
        )

    # Map `container_name` → allowed-key set. Subscripts against
    # containers NOT in this map (e.g., `components['foo']`) are not
    # statically validated — see docstring.
    container_key_sets: dict[str, frozenset[str]] = {
        "info": PERSISTED_TRAJECTORY_KEYS,
        "trajectory": PERSISTED_TRAJECTORY_KEYS,
        "behavior": BEHAVIOR_KEYS,
    }

    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in container_key_sets
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            container = node.value.id
            key = node.slice.value
            allowed = container_key_sets[container]
            if key not in allowed:
                violations.append(
                    f"{container}[{key!r}] not in the persisted "
                    f"{container} keys ({sorted(allowed)})"
                )
    if violations:
        raise MissionValidationError(
            f"stage {stage.name!r} success_criterion references "
            f"keys the rollout artifacts don't persist:\n  - "
            + "\n  - ".join(violations)
            + "\nSee sculptor.mission_runtime for the full namespace."
        )
