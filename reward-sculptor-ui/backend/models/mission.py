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
StageStatus = Literal[
    "pending", "training", "succeeded", "failed", "skipped",
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
