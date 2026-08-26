"""Pydantic shapes for Ship 18a — mission CRUD + execution endpoints.

Source-of-truth invariant (per Ship 18a plan-review):
  `mission.json` on disk is the canonical Mission state. JobManager
  events are an EPHEMERAL overlay that the UI reads alongside the
  filesystem snapshot. `MissionDetail` is reconstructed every GET via
  `sculptor.mission.load_mission`, NOT cached or kept in memory.

Event payloads on the WebSocket wire are intentionally `dict[str, Any]`
(only the discriminator `type` is validated server-side) — mirrors the
runs.py WS pattern where typed views are server-built, not client-
typed. Ship 18b's frontend can narrow on `type` per-event.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Stage shape (mirrors sculptor.mission.Stage) ─────────────────────
# §mission-persistence increment 1/2: "superseded" mirrors
# sculptor.mission.StageStatus — a stage retained in mission.stages
# after being replaced by re-decomposition children. Terminal, never
# runnable.
StageStatus = Literal[
    "pending", "training", "succeeded", "failed", "skipped", "superseded",
]


class StageSchema(BaseModel):
    """One stage of a mission curriculum.

    Mirrors `sculptor.mission.Stage` field-for-field. Re-defining as a
    pydantic model (rather than wrapping the dataclass) keeps the
    backend's HTTP contract independent of internal sculptor changes —
    a sculptor-side rename to `Stage` would otherwise break the API.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    goal_text: str
    success_criterion: str
    max_iterations: Annotated[int, Field(ge=1, le=50)]
    parent_stage: Optional[str] = None
    reward_seed_prompt: str
    kg_seed_papers: list[str] = Field(default_factory=list)

    status: StageStatus = "pending"
    final_policy_path: Optional[str] = None
    final_reward_path: Optional[str] = None
    best_metric: Optional[float] = None
    iterations_used: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    redecomposition_attempts: int = 0
    # §keep-best (B1): the iteration the stage kept as its final policy +
    # why ("criterion+fitness" | "criterion_newest" | "fitness_fallback" |
    # "last"). None on legacy missions / stages that haven't finalized.
    selected_iter_index: Optional[int] = None
    selection_source: Optional[str] = None
    # §Ship 20a: the iteration cap actually enforced for the stage's last
    # (or current) run — `iterations_override or max_iterations`. The
    # frontend has typed this field since Ship 20a but the mirror never
    # carried it; disk-run synthesis and `rounds X/Y` displays need it.
    effective_max_iterations: Optional[int] = None
    # §mission-persistence increment 2: mirrors sculptor.mission.Stage's
    # failure_reason/failure_detail (increment 1) — the short reason
    # code + free-form detail for a failed/superseded stage. Filled
    # either from mission.json directly, or (for stages that predate
    # increment 1, or synthetic on-disk-only stages) enriched from
    # provenance.json by mission_store.load_mission_detail.
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None
    # Mirrors sculptor.mission.Stage.steering_metric — per-stage
    # objective fitness metric override (a spec-metric name or a
    # resolved generated-metric path). None = mission-level fallback.
    steering_metric: Optional[str] = None
    # §mission-persistence increment 2: human-facing position label
    # computed by mission_store over the FINAL disk-union stage order —
    # "1", "2", "2.1", "2.2", ... for replan children. Always set by
    # load_mission_detail; "" only as the pydantic default before that
    # pass runs (should never be visible on the wire).
    display_label: str = ""
    # True when this StageSchema was synthesized from an orphaned
    # `<mission_dir>/stages/<name>/` directory that has no entry in
    # mission.stages (old destructive-splice data). Such a stage is
    # always status="superseded" and its Claude-authored fields
    # (goal_text/success_criterion/reward_seed_prompt) are best-effort
    # recovered from provenance.json, defaulting to "".
    on_disk_only: bool = False
    # §mission-persistence increment 2: whether this stage has a
    # trust-gated steering metric on disk, derived from
    # `<mission_dir>/stage_metrics/<name>/meta.json` +
    # `steering_metric`. See mission_store._stage_metric_status.
    metric_status: Optional[Literal["accepted", "rejected", "inherited", "none"]] = None
    # §D28 F-SYNTH: NOT mirrored from sculptor.mission.Stage — a DERIVED
    # field (same convention as `metric_status` above), computed fresh
    # each request from `<mission_dir>/stage_metrics/<name>/meta.json`'s
    # `exemplar.kind` (stamped there by `sculptor.mission_metrics.
    # _attempt_synthetic_certification` on a synthetic-exemplar
    # acceptance). "synthetic" when the accepted metric was certified
    # against a last-resort synthesized exemplar (never steer-grade on
    # its own — observe until task-derived calibration); "reference"
    # when a real reference_clip_id is attached (regardless of whether
    # its metric is calibrated); `None` for a plain/unreferenced metric
    # or no metric at all. See mission_store._stage_exemplar_kind.
    exemplar_kind: Optional[Literal["reference", "synthetic"]] = None
    # §R1_BUILD_SPEC decision 10/11: mirrors sculptor.mission.Stage's
    # reference_clip_id/reference_tier/reference_match_confidence — the
    # reference-library clip (if any) attached to this stage via
    # `POST/DELETE .../stages/{stage}/reference`. None on stages with no
    # reference attached (the overwhelming majority; back-compat with
    # every mission.json that predates R1 is automatic via
    # `Stage.from_dict`'s filter-unknown-keys path — same guarantee
    # `steering_metric` already relies on).
    reference_clip_id: Optional[str] = None
    reference_tier: Optional[str] = None
    reference_match_confidence: Optional[float] = None
    # Exact dynamics-admission receipt persisted on the core Stage.  These are
    # intentionally nullable for legacy reads; runnable attached stages fail
    # closed unless all three are present and re-verify at launch.
    reference_robot: Optional[str] = None
    reference_clip_sha256: Optional[str] = None
    reference_certificate_sha256: Optional[str] = None
    reference_execution_contract_sha256: Optional[str] = None
    reference_execution_boundary_sha256: Optional[str] = None
    # §D24 F1: mirrors sculptor.mission.Stage's reference_span_start_s/
    # _end_s/_confidence/_method — the goal-aligned SUB-SPAN of
    # `reference_clip_id` selected by `sculptor.refs.spans.
    # select_reference_span` (docs/internal/REFERENCE_BUILD_LOG.md
    # D23/D24). All four None when no span applies (no clip attached,
    # goal covers the whole clip, or selection was declined) — the
    # stage's certification/RSI/eval-reset then use the FULL clip.
    # Back-compat with every mission.json that predates this field is
    # automatic via `Stage.from_dict`'s filter-unknown-keys path, same
    # guarantee `reference_clip_id` already relies on.
    reference_span_start_s: Optional[float] = None
    reference_span_end_s: Optional[float] = None
    reference_span_confidence: Optional[float] = None
    # §D24 W5 hardening: `reference_span_method` may also be
    # `"declined:<reason>"` (start/end/confidence stay None) — a
    # SEMANTIC decline (whole_clip/low_confidence/qc_reject) that was
    # already attempted, so the backfill/attach path never re-fires a
    # real LLM call for this stage. Plain `Optional[str]`, no length/
    # enum constraint, so this passes through unchanged.
    reference_span_method: Optional[str] = None
    # §start_pose: mirrors sculptor.mission.Stage.start_pose — the
    # physical configuration the robot is in at THIS stage's episode
    # start ("supine" | "prone" | "sitting" | "crouched" | "standing"),
    # or None (unspecified). Back-compat with every mission.json that
    # predates this field is automatic via `Stage.from_dict`'s
    # filter-unknown-keys path — same guarantee `reference_clip_id`
    # already relies on.
    start_pose: Optional[str] = None


# ── Mission summary / detail / create ────────────────────────────────
MissionLifecycleStatus = Literal[
    "ready",       # decompose succeeded; not yet run
    "running",     # mission_execute job in flight
    "completed",   # all stages succeeded
    "halted",      # mission_execute halted at a stage
    "errored",     # decompose or execute hit an unrecoverable error
]


class MissionSummary(BaseModel):
    """Slim list view — what the UI shows in a mission table row."""

    model_config = ConfigDict(extra="forbid")

    mission_slug: str
    project_slug: str
    goal: str
    n_stages: int
    current_stage_idx: int
    decomposition_model: str
    created_at: datetime
    # Lifecycle is DERIVED from on-disk stage statuses + active job
    # (per source-of-truth invariant). Server computes it; clients
    # read it.
    lifecycle: MissionLifecycleStatus
    # Set when a related job is in flight; null otherwise.
    active_job_id: Optional[str] = None
    active_job_kind: Optional[
        Literal["mission_decompose", "mission_execute"]
    ] = None


class MissionDetail(MissionSummary):
    """Full view — stage list + decomposition rationale.

    Adds `stages` and `decomposition_rationale` to the slim summary.
    Always reconstructed from `mission.json` on each GET; never cached.
    """

    model_config = ConfigDict(extra="forbid")

    stages: list[StageSchema]
    decomposition_rationale: str
    schema_version: int = 1
    # §Ship 21a: persisted run-time defaults set at mission-creation
    # time via NewMissionDialog's Advanced tab. RunMissionDialog
    # pre-fills from these on first open. Null when none were set
    # (mission was created with the basic-tab-only flow).
    run_defaults: Optional[dict[str, Any]] = None


class CreateMissionRequest(BaseModel):
    """POST /projects/{slug}/missions body."""

    model_config = ConfigDict(extra="forbid")

    goal: Annotated[str, Field(min_length=8, max_length=2000)]
    """Behavior goal. Min length 8 catches obvious typos; max 2000
    matches `Mission.goal` and `apply_prompt_edit`'s prompt cap."""

    mission_slug: Optional[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    ] = None
    """Optional explicit slug. When None, server derives one from the
    goal via `mission_store._derive_mission_slug`. Pattern matches
    URL-safe characters; collisions resolve via `-2`/`-3` suffix."""

    no_kg: bool = False
    """Skip KG context to Claude during decompose. Faster (~10s) but
    Claude can't cite KG papers in stage seed prompts."""

    gen_stage_metrics: bool = True
    """§MISSION_METRIC_GRANULARITY: after decompose, generate one
    trust-gated objective metric per stage from the stage's own goal
    text. Rejected generations leave the stage on the mission-level
    metric fallback. Default ON."""

    stage_metric_candidates: Annotated[int, Field(ge=1, le=4)] = 1
    """§MISSION_RUN_PARITY: best-of-N candidates sampled per stage metric
    (1 = single-shot-with-retry). Only meaningful when
    `gen_stage_metrics` is True; forwarded to
    `generate_stage_metrics(n_candidates=...)`. Each candidate is a
    ~1-2 min LLM call, so higher N trades wall-clock for a more-
    discriminating metric."""

    stage_metric_required: bool = False
    """§mission-persistence increment 2: when true, running this
    mission is BLOCKED (409) while any runnable stage (status pending
    or training) lacks a steering metric — unless the run request sets
    `proceed_blind`. Persisted into the mission's `run_defaults` at
    creation time; enforced by `POST .../run`. Default False preserves
    the pre-existing behavior (metrics are best-effort; a rejected /
    skipped generation silently falls back to the mission-level
    metric)."""

    # §Ship 21a: optional run-time defaults set up front via the
    # NewMissionDialog Advanced tab. These are persisted on the
    # mission and pre-fill RunMissionDialog when the user later
    # clicks Run mission. Validated against the same shape as
    # RunMissionRequest so the round-trip is type-safe.
    run_defaults: Optional["RunMissionRequest"] = None


class RunMissionRequest(BaseModel):
    """POST /projects/{slug}/missions/{ms}/run body — Ship 19d.

    All fields optional; an empty body `{}` preserves the Ship 16
    default (per-stage budget from mission.json, no Goal A/B). The
    UI's RunMissionDialog sets these per-launch.
    """

    model_config = ConfigDict(extra="forbid")

    iterations_override: Optional[
        Annotated[int, Field(ge=1, le=200)]
    ] = None
    """Override every stage's max_iterations (e.g., 2 to clamp a
    Claude-authored 6-iter-per-stage mission to a quick smoke)."""

    steps_per_iter: Optional[
        Annotated[int, Field(ge=100, le=200_000)]
    ] = None
    """Override [iteration].steps_per_iter for every stage. mjlab:
    rsl_rl iters per cycle. gym_sb3: env-step budget."""

    seed: Optional[Annotated[int, Field(ge=0, le=2_000_000_000)]] = None
    """Per-iter base seed override; iter i uses seed+i."""

    early_stop_on_criterion: bool = False
    """Goal A: exit a stage early on consecutive criterion-pass."""

    criterion_stability_window: Annotated[int, Field(ge=1, le=10)] = 1
    """Consecutive iters the criterion must hold before Goal A fires.
    1 = immediate exit on first pass; bump to 2-3 for noisy metrics."""

    extend_on_improvement: bool = False
    """Goal B: extend a stage if the metric is still improving at the
    end of its allocated budget."""

    max_extensions_per_stage: Annotated[int, Field(ge=0, le=3)] = 1
    """Hard cap on Goal B extensions per stage (max 3)."""

    extension_factor: Annotated[float, Field(ge=0.1, le=1.5)] = 0.5
    """Fraction of original max_iterations to add per Goal B extension."""

    extension_improvement_threshold: Annotated[
        float, Field(ge=0.0, le=1.0),
    ] = 0.05
    """Goal B trend threshold (5% relative improvement floor)."""

    # §Ship 34/35: objective fitness-in-the-loop, applied uniformly to
    # every stage (sound for single-skill missions). A built-in spec
    # name or a generated-metric id ("gen:<id>"); None = the blind loop.
    fitness_metric: Optional[str] = None
    """Spec metric (built-in name or 'gen:<id>') used as in-loop fitness
    for every stage. None keeps the blind loop."""

    # §Ship 35: observe vs steer (see RunParams.fitness_mode).
    fitness_mode: Literal["observe", "steer"] = "steer"
    """How the fitness signal is used per stage. observe = display only."""

    # §MISSION_RUN_PARITY: per-launch knobs mirrored from NewRunDialog's
    # stage-applicable set. All Optional; None = defer to the stage's
    # inherited config default. Forwarded as `sculpt mission-run` flags by
    # _build_mission_run_flags and applied uniformly to every stage.
    edit_candidates: Optional[Annotated[int, Field(ge=1, le=5)]] = None
    """Best-of-K framed reward-edit candidates per diagnosis, per stage
    (offline-screened; only the winner trains). None = 1."""

    rollout_episodes: Optional[Annotated[int, Field(ge=1, le=32)]] = None
    """Rollout episodes captured per iter for behavior metrics, per
    stage. None = the inherited [iteration] value (default 6)."""

    max_episode_steps: Optional[
        Annotated[int, Field(ge=50, le=5000)]
    ] = None
    """Rollout env steps per episode, per stage. None = default 500."""

    playback_speed: Optional[
        Annotated[float, Field(ge=0.1, le=10.0)]
    ] = None
    """Rollout video speed multiplier, per stage; 1.0 = real-time."""

    render_width: Optional[Annotated[int, Field(ge=1)]] = None
    """Rollout video width px, per stage (default 1280). Paired with
    render_height by the UI's resolution select."""

    render_height: Optional[Annotated[int, Field(ge=1)]] = None
    """Rollout video height px, per stage (default 720)."""

    fitness_patience: Optional[Annotated[int, Field(ge=1, le=50)]] = None
    """Iters with no new best fitness before a steered stage stops. Only
    meaningful with a fitness_metric set. None = sculpt default (2)."""

    num_envs_override: Optional[Annotated[int, Field(ge=1, le=8192)]] = None
    """Override [adapter].config.num_envs for every stage (mjlab). Drop
    if a stage OOMs. None = inherited value."""

    device_override: Optional[str] = None
    """Override [adapter].config.device for every stage (mjlab), e.g.
    cuda:0 / cpu. None = inherited value."""

    proceed_blind: bool = False
    """§mission-persistence increment 2: bypass the
    `stage_metric_required` 409 guard for this launch — run every
    runnable stage on whatever fallback metric it has (mission-level
    fitness_metric, or the blind loop), even though the mission was
    created with `stage_metric_required=True`. One-shot per request;
    does not mutate the persisted `run_defaults`."""


# §Ship 21a: resolve the forward reference now that RunMissionRequest
# is defined. Required because CreateMissionRequest declares
# `run_defaults: Optional["RunMissionRequest"]` higher up in the file.
CreateMissionRequest.model_rebuild()


class DeleteMissionResponse(BaseModel):
    """200 response for DELETE — includes freed_bytes so the UI can
    render "Deleted, freed 4.2 GB" (per Ship 18a plan-review's UX
    win)."""

    model_config = ConfigDict(extra="forbid")

    mission_slug: str
    freed_bytes: int


class BackfillFitnessResponse(BaseModel):
    """200 response for `POST .../missions/{mission_slug}/backfill-fitness`
    — `backend.services.fitness_backfill.backfill_mission_fitness`'s
    summary verbatim. `stages` maps stage name to the count of
    `fitness.json` files written for that stage this call (stages with
    nothing to backfill are simply absent, not zero-valued)."""

    model_config = ConfigDict(extra="forbid")

    written: int
    skipped_existing: int
    no_iter_dir: int
    stages: dict[str, int]


# ── WebSocket event payloads ─────────────────────────────────────────
# Per Ship 18a plan-review: mirror runs.py WS pattern. Validate the
# discriminator + envelope; leave per-event payload as `dict[str, Any]`.
class MissionEvent(BaseModel):
    """Common envelope for all mission-execute events on the WS wire.

    The 19 event types Ship 14-17 emit (`mission_started`,
    `stage_started`, ..., `feedback_read_degraded`) all carry their
    own type-specific fields. We validate the envelope and pass the
    payload through verbatim so frontend can narrow on `type`.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    # Optional fields most events share; use Any so the model doesn't
    # reject events that don't have them.
    stage_name: Optional[str] = None
    stage_index: Optional[int] = None
    ts: Optional[float] = None


# ── §C2 stage de-siloing: disk-truth stage-iteration endpoints ────────
# `mission_stage_run` artifacts (`<mission_dir>/stages/<stage>/runs/
# iter_<N>/`) are unreachable via the in-memory job endpoints once the
# backend restarts or training moves past that stage — `routes/runs.py`'s
# `_find_run` requires a live JobManager entry. These models back
# DISK-TRUTH endpoints in `routes/missions.py` that need no job at all,
# so any stage of any mission stays viewable.
class StageIterationSummary(BaseModel):
    """One row of `GET .../stages/{stage}/iterations` — mirrors
    `sculptor.export.list_exportable_iters` but does NOT require a
    checkpoint (a stage iter can have a rollout with no saved
    checkpoint, or vice versa; the UI wants to see both)."""

    model_config = ConfigDict(extra="forbid")

    iter_index: int
    primary_metric: Optional[float] = None
    fitness: Optional[float] = None
    has_rollout: bool
    has_checkpoint: bool
    reward_version: Optional[str] = None
    # §D24 (F4): true when `<iter_dir>/fitness_contradiction.json` exists —
    # the stage's success criterion evaluated True on this iter while the
    # objective fitness was at/near zero (the D20 hollow-success / D23
    # exemplar-scope-mismatch pattern). `fitness_components` passes
    # through the flag file's per-channel breakdown so the UI can show
    # WHICH channel zeroed the fitness without a second request.
    fitness_contradiction: bool = False
    fitness_components: Optional[dict] = None
    # §fitness.json/selection.json backend increment (commit f1c339d
    # follow-up): the naturalness-GATED steer value + dense progress
    # channel the loop actually used for keep-best selection, sourced from
    # `<iter_dir>/fitness.json` when present (LIVE or `log_backfill`); None
    # for iters that predate that file and haven't been backfilled.
    steer_fitness: Optional[float] = None
    progress: Optional[float] = None
    # naturalness_flag/naturalness_hard_reject: fitness.json's own values
    # when present, else a best-effort read of `realism_audit.json`'s
    # `naturalness` dict — never a crash on a missing/malformed source,
    # just null/False.
    naturalness_flag: Optional[str] = None
    naturalness_hard_reject: bool = False
    # "live" (sculpt.py wrote it in-loop) | "log_backfill" (recovered from
    # an `_execute_*.log`'s `iter_fitness` SCULPT-EVENT after the fact) |
    # None when `fitness` came from the legacy `_extract_objective_fitness`
    # scrape (no fitness.json at all for this iter).
    fitness_source: Optional[str] = None
    # Project-level best-policy evaluation may leave one or more fresh-seed
    # replay directories next to the ordinary rollout. Mission stages default
    # to zero until they adopt the same artifact convention.
    fresh_rollout_count: int = 0


class StageIterPaperRef(BaseModel):
    """One citation/grounding entry — shared shape for
    `StageIterDetail.reward_references` (sourced from `reward_spec.json`'s
    `references`/`grounding`) and `.literature_context` (sourced verbatim
    from `diagnosis.json`'s `literature_context`)."""

    model_config = ConfigDict(extra="forbid")

    arxiv_id: Optional[str] = None
    citation: Optional[str] = None
    description: Optional[str] = None
    # True: retrieved from the KG this iteration. False: the diagnoser
    # cited it from recall (it exists in the KG but wasn't fetched this
    # iter). None: unknown — either the diagnoser didn't tag it, or this
    # is an old diagnosis.json written before grounding tags existed.
    grounded: Optional[bool] = None


class StageIterDetail(BaseModel):
    """`GET .../stages/{stage}/iterations/{i}/detail` — the reasoning
    behind one finished stage iteration (diagnosis, cited papers, reward
    summary, component means), assembled disk-truth from the iter dir.
    Every field is best-effort: a missing/unreadable source file yields
    null/[] for the fields it would have populated, never a 500. 404 is
    reserved for an unknown stage or iter dir (see `_stage_dir_or_404`
    and the iter-dir existence check in the route)."""

    model_config = ConfigDict(extra="forbid")

    iter_index: int
    reward_version: Optional[str] = None
    reward_description: Optional[str] = None
    reward_references: list[StageIterPaperRef] = []
    primary_metric: Optional[float] = None
    objective_fitness: Optional[float] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None
    failure_modes: list[str] = []
    literature_context: list[StageIterPaperRef] = []
    components: Optional[dict[str, float]] = None


class StageMetricReference(BaseModel):
    """One entry of `meta["validation"]["references"]` — a reference clip
    the metric was certified against (§REFERENCE_TRAJECTORY_PLAN §5/§9).
    `gates` carries the three per-reference gate verdicts (nondegeneracy /
    monotonicity / negatives); a reference is "passed" iff all three are
    true. Mirrors `sculptor.eval.metric_validate._validate_references`'s
    per-entry dict shape — extra keys (`reasons`, `scores`,
    `full_components`) are dropped rather than modeled since the UI only
    needs the pass/fail summary."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    gates: dict[str, bool] = {}


class StageObjectiveMetric(BaseModel):
    """`GET .../stages/{stage}/metric` — the stage's objective (steering)
    metric record: what was generated under `stage_metrics/<stage>/` and
    whether the adversarial panel accepted it. `status` reuses the same
    accepted/rejected/inherited/none derivation as
    `StageSchema.metric_status` (see `mission_store._stage_metric_status`,
    which this route's helper mirrors so the two surfaces never
    disagree)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted", "rejected", "inherited", "none"]
    behavior_goal: Optional[str] = None
    metric_source: Optional[str] = None
    validation_passed: Optional[bool] = None
    review_summary: Optional[str] = None
    n_candidates: Optional[int] = None
    calibrated: Optional[bool] = None
    validator_basis: Optional[str] = None
    abstract_objective_program: list[str] = []
    channel_catalog_hash: Optional[str] = None
    # §R1 remainder (plan §9): the reference clips this metric was
    # certified against, mirrored from meta["validation"]["references"].
    # [] when no references were attached (pre-R1 records and non-
    # reference-anchored metrics alike) — never null, so the UI can
    # unconditionally .length-check.
    references: list[StageMetricReference] = []


class StageSelectionInfo(BaseModel):
    """`GET .../stages/{stage}/selection` — the stage's keep-best decision,
    disk-truth. `<stage_dir>/reports/selection.json` (written live by
    `sculpt.py` as of commit f1c339d) is returned VERBATIM plus
    `synthesized: false` when present. Every stage that predates that write
    gets an equivalent document SYNTHESIZED from mission.json's stage
    record (`selected_iter_index`/`selection_source`/`success_criterion`/
    `failure_reason`/`failure_detail`) plus a best-effort per-iter
    candidate list built off `runs/iter_N/` — `synthesized: true,
    schema: 0`. `candidates[].criterion_pass` is always null in the
    synthesized form: only the LIVE selection.json writer knows what the
    criterion actually evaluated to per-candidate at selection time.

    Deliberately loose (`extra="allow"`): `selection.json`'s shape is
    free to grow on the sculptor side, and passing an unrecognized key
    through untouched is the whole point of a disk-truth passthrough
    endpoint — only `synthesized` is pinned down as a real field."""

    model_config = ConfigDict(extra="allow")

    synthesized: bool


class StageEnvSpecInfo(BaseModel):
    """`GET .../stages/{stage}/env-spec` — mirrors `EnvSpecInfo`
    (routes/projects.py's project-level env-spec surface) but scoped to
    one stage's `<mission_dir>/stages/<stage>/env/` dir. A grounded
    stage (no env-spec adaptation) has no `env/` dir at all, which is
    NOT a 404 here — `current=None, versions=[]` is the honest answer.

    `current.meta.source` is the RSI (reference-state-initialization)
    tell: a value starting `"reference:"` means the spec's train-side
    reset ranges were seeded from a reference clip rather than a
    generator/diagnoser, which the UI surfaces as a chip.
    """

    model_config = ConfigDict(extra="forbid")

    active: bool
    current: Optional[dict[str, Any]] = None
    versions: list[str] = []
