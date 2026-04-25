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
