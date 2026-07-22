"""Pydantic shapes for sculpt-run launch + status + WS event replay.

Runs are JobManager jobs of kind `sculpt_run` (per Prompt 8 R3 — no
parallel state tracker). These models are lightweight views onto the
underlying Job.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.models.kg import JobDetail, JobStatus


class RunParams(BaseModel):
    """POST /projects/{slug}/runs body.

    Phase 4 adds GPU-flavored overrides (`training_iterations`,
    `num_envs_override`, `device_override`) + an `expand_kg` opt-in for
    Phase 2 prompt-time research. All new fields are optional so the
    existing Hopper/Ant gym flow keeps working unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    # Basic (gym_sb3 + mjlab).
    behavior_goal: Annotated[str, Field(min_length=4, max_length=500)]
    iterations: Annotated[int, Field(ge=1, le=100)] = 20
    no_kg: bool = False
    dry_run: bool = False

    # Advanced — Phase 4.
    training_iterations: Optional[
        Annotated[int, Field(ge=100, le=200_000)]
    ] = None
    """Inner-loop budget per sculpt cycle. Semantics depend on adapter:
    for gym_sb3 this overrides `steps_per_iter` (env steps); for mjlab
    it becomes `rsl_rl.max_iterations` (policy-update iterations).
    None = use the project's config.toml default."""

    num_envs_override: Optional[
        Annotated[int, Field(ge=1, le=8192)]
    ] = None
    """Per-run override for parallel env count. None = project default."""

    device_override: Optional[
        Annotated[str, Field(pattern=r"^(cpu|cuda(:\d+)?)$")]
    ] = None
    """Per-run device override (e.g. `cpu`, `cuda:0`). None = adapter
    default (cuda:0 for mjlab, irrelevant for gym_sb3)."""

    expand_kg: bool = False
    """When true, Phase 2 auto-research fires before Stage-2 diagnose
    so thin topics get fresh paper coverage. Requires ANTHROPIC_API_KEY."""

    # §Ship-7: rollout-video + RL knobs. Every standard RL knob users
    # might tune in a normal training run is surfaced here so the UI
    # New Run dialog can expose them all. None = runner default.
    max_episode_steps: Optional[
        Annotated[int, Field(ge=50, le=5000)]
    ] = None
    """Env steps per rollout episode. Default: 500 (runner)."""

    playback_speed: Optional[
        Annotated[float, Field(gt=0.1, le=10.0)]
    ] = None
    """Video playback speed multiplier. 1.0 = real-time, 2.0 = 2x faster,
    0.5 = slow motion. Default 1.0."""

    render_every: Optional[
        Annotated[int, Field(ge=1, le=100)]
    ] = None
    """Capture every N-th step. Advanced — default auto-caps at 500 frames."""

    rollout_fps: Optional[
        Annotated[float, Field(gt=0, le=240)]
    ] = None
    """Hard override on playback fps. 0/None = derive from env.step_dt."""

    render_width: Optional[
        Annotated[int, Field(ge=64, le=3840)]
    ] = None
    """Rollout video width in px. None = runner default (1280).
    Render cost is resolution-independent on this stack."""

    render_height: Optional[
        Annotated[int, Field(ge=64, le=2160)]
    ] = None
    """Rollout video height in px. None = runner default (720)."""

    rollout_episodes: Optional[
        Annotated[int, Field(ge=1, le=32)]
    ] = None
    """Number of rollout episodes to record each iter. Default: 6."""

    seed: Optional[
        Annotated[int, Field(ge=0, le=2**31 - 1)]
    ] = None
    """Base RNG seed. Iter i uses seed + i. Default: 42 or config.toml."""

    auto_adjust_physics: Optional[bool] = None
    """§7.4: emit a physics-edit suggestion event on severe realism
    audits. None = project's config.toml default (ships as true for new
    projects via CONFIG_TEMPLATE; older projects default to false)."""

    early_stop_enabled: Optional[bool] = None
    """Legacy compatibility no-op. Metric-plateau auto-kill is disabled."""

    early_stop_patience: Optional[Annotated[int, Field(ge=1, le=100)]] = None
    """Legacy compatibility no-op; accepted for older API clients/configs."""

    # §Ship 34/35: objective fitness-in-the-loop. A built-in spec-metric
    # name (cartpole_balance / g1_floss / g1_kick / go1_trot) OR a
    # generated-metric id ("gen:<id>", resolved server-side to the
    # project's metric .py). None = the blind loop. Kept as a free str
    # (not a Literal) so auto-generated metrics are selectable; the
    # sculpt CLI fail-fasts on an unresolvable value.
    fitness_metric: Optional[str] = None
    """Spec metric (built-in name or 'gen:<id>') used as in-loop fitness.
    None keeps the blind loop (criterion / metric-history only)."""

    # §Ship 35: observe vs steer. "steer" (default): fitness drives
    # best-selection + early-stop + the diagnoser. "observe": compute +
    # display only, zero influence (fair A/B; safe default for
    # not-yet-calibrated generated metrics).
    fitness_mode: Literal["observe", "steer"] = "steer"
    """How the fitness signal is used. observe = passive display only."""

    # §best-of-N: when fitness_metric == "generate-at-launch", how many candidate
    # metrics to sample at launch and select the most-discriminating valid one from
    # (1 → single-shot-with-retry). Bounded so a typo can't fan out unbounded LLM
    # calls; ignored for any other fitness_metric value.
    metric_n_candidates: Annotated[int, Field(ge=1, le=8)] = 1
    """Best-of-N candidates for a generate-at-launch metric. 1 = single-shot."""

    # §Ship 48: patience for the FITNESS-plateau early-stop — the live early
    # stop on a steered run (stop after this many iters with no NEW BEST
    # fitness). This is the knob that actually governs truncation; the
    # legacy early_stop_* fields above are a no-op for it. The sculpt-lib
    # default is 2, which truncated the g1-kick-v3 run at iter 4 before the
    # reward could escape the standing basin. None → sculpt-lib default.
    fitness_patience: Optional[Annotated[int, Field(ge=1, le=50)]] = None
    """Iters with no new best fitness before a steered run stops. None →
    sculpt default (2). The UI sends 4 for exploratory hard skills."""

    # §Ship 39 (H1): interactive human-in-the-loop start mode. "manual" (the
    # UI default) pauses for human feedback at each iteration boundary so the
    # user can steer from what they see in the rollout video; "auto" runs
    # straight through. The Auto/Manual toggle flips this at ANY point mid-run
    # via PATCH /runs/{id}/control. Model default "auto" keeps non-UI / test
    # launches non-interactive (no pause → no hang).
    start_mode: Literal["manual", "auto"] = "auto"
    """Interactive start mode. manual = pause-for-feedback each iteration."""

    resume_exact_tuple: bool = False
    """Before resuming, restore reward/env inputs from the authoritative
    promoted atomic selection.  This is an explicit recovery control for
    rejecting unpromoted diagnosis drafts; normal iterative resumes keep using
    the newly generated drafts."""


class RunControl(BaseModel):
    """PATCH /projects/{slug}/runs/{run_id}/control body — interactive
    human-in-the-loop control (§Ship 39). All fields optional; the route
    merges them into the run's control sidecar that the sculpt subprocess
    polls at each iteration boundary."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[Literal["manual", "auto"]] = None
    """Flip the run between pause-for-feedback (manual) and run-through (auto)."""
    resume: bool = False
    """Release the current pause (bumps the resume token), with optional feedback."""
    feedback: Optional[Annotated[str, Field(max_length=4000)]] = None
    """Free-text human observation to inject into the NEXT iteration's diagnose."""
    stop: bool = False
    """End the run cleanly after the current iteration."""
    gen_retry: bool = False
    """§Ship 45: retry the launch-time metric generation after a rejection."""
    gen_continue: bool = False
    """§Ship 45: stop retrying launch-time generation and continue blind."""


class RunControlState(BaseModel):
    """The control sidecar's current state, returned by the PATCH route."""

    model_config = ConfigDict(extra="allow")

    mode: str = "auto"
    resume_token: int = 0
    feedback: Optional[str] = None
    stop: bool = False


class IterEventSummary(BaseModel):
    """One row of the iteration timeline rendered in the Runs tab.
    Derived from filesystem watchers + the `iter_completed` stdout
    marker, with the filesystem winning on any disagreement."""

    model_config = ConfigDict(extra="forbid")

    iter_index: int
    status: Literal["running", "completed", "errored", "stopped"]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    reward_version_before: Optional[int] = None
    reward_version_after: Optional[int] = None
    primary_metric: Optional[float] = None
    metric_delta: Optional[float] = None
    failure_modes: list[str] = []
    edit_count: Optional[int] = None
    paper_refs: list[str] = []
    rollout_ready: bool = False
    diagnosed: bool = False
    # §7.3: realism-audit payload. None until `iter_<N>/realism_audit.json`
    # appears; populated via the `realism_audited` event emitted by
    # sculpt.py + forwarded by run_manager._check_iter_artifacts.
    realism_audit: Optional[dict] = None
    # §7.4: ready-to-apply MJCF-edit prompt when the realism audit hit
    # `severe` AND the project's config has `auto_adjust_physics = true`.
    # UI surfaces this as an "apply physics fix" chip that opens the
    # Physics tab with the prompt pre-filled. None otherwise.
    physics_edit_suggestion: Optional[dict] = None
    # §Ship 48: edits the diagnoser wanted but couldn't ground because the
    # adapter doesn't expose the needed field (requires_env_extension).
    # {"terms": [...], "rationales": [...]}. Surfaced as an informational
    # "needs adapter channels" chip — an env extension is a code change,
    # never auto-applied. None until an iter defers ≥1 such edit.
    env_extension_suggestion: Optional[dict] = None
    # §Ship 34: per-iter objective fitness (spec_score) and best-so-far
    # when the run was launched with a --fitness-metric. None for blind
    # runs. Populated from iter_fitness / best_reward_selected events.
    fitness: Optional[float] = None
    best_fitness: Optional[float] = None
    # §Convergence loop 1: dense sub-success progress (the metric's
    # progress_score — ranks iterations below the completion gate). None
    # when the metric doesn't emit it.
    progress: Optional[float] = None
    # §env generalization: diagnoser env-curriculum change applied at
    # this iter's boundary (env_spec_updated event): {"new_version":
    # "v2" | None, "applied": ["entropy_coef_scale=1.5", ...],
    # "rejected": [{"parameter": ..., "reason": ...}]}. Takes effect the
    # NEXT iteration's training. None until an iter proposes env edits.
    env_spec_update: Optional[dict] = None


class ErrorClassification(BaseModel):
    """Structured failure metadata surfaced by `run_manager._classify_run_failure`
    when a sculpt subprocess exits non-zero (M7 Phase 6). The UI uses this
    to render one-click remediation (e.g. "Regenerate reward template")
    alongside the raw error string."""

    model_config = ConfigDict(extra="allow")

    kind: str  # "oom" | "reward_contract_mismatch" | "driver_version" | "no_cuda" | "unknown"
    title: Optional[str] = None
    detail: Optional[str] = None
    suggestions: list[str] = []
    problem_type: str = "about:blank"
    action: Optional[dict] = None


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project_slug: str
    status: JobStatus
    behavior_goal: str
    iterations_requested: int
    iterations_completed: int
    current_iter_index: Optional[int]
    primary_metric_history: list[Optional[float]]
    # §Ship 35: per-iter objective fitness history (parallel to
    # primary_metric_history), for the Runs-tab/dashboard sparkline + the
    # detail chart when a fitness metric is in play. Empty for blind runs.
    fitness_history: list[Optional[float]] = []
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    error: Optional[str]
    error_classification: Optional[ErrorClassification] = None
    # §Ship 21: distinguish standalone sculpt_run from per-stage
    # mission_stage_run rows in the Runs tab. Optional with default
    # "sculpt_run" for backward compat with older clients.
    kind: str = "sculpt_run"
    # §Ship 21: parent mission_execute job_id for stage runs; None
    # for standalone sculpt_run.
    parent_id: Optional[str] = None
    mission_slug: Optional[str] = None
    stage_name: Optional[str] = None
    stage_index: Optional[int] = None
    # §Ship 39 (H1): current interactive mode ("manual" | "auto") so a
    # reconnect restores the Auto/Manual toggle. None for older / non-UI runs.
    mode: Optional[str] = None


class RunDetail(RunSummary):
    params: RunParams
    iterations: list[IterEventSummary]
    stdout_tail: list[str]                # last 200 in-memory lines
    total_event_count: int                # events list size


class PolicySummary(BaseModel):
    """One exportable trained iteration (GET /projects/{slug}/policies).
    Mirrors sculptor.export.list_exportable_iters — disk-backed, so the
    list survives backend restarts."""

    iter_index: int
    checkpoint: str                       # "checkpoint.pt" | "checkpoint.zip"
    checkpoint_bytes: int
    primary_metric: Optional[float] = None
    fitness: Optional[float] = None
    reward_version: Optional[str] = None


class RunWSMessage(BaseModel):
    """Envelope shape for `/ws/projects/{slug}/runs/{run_id}/events`.
    Frontend switches on `type`. See routes/ws_runs.py for the
    discriminated union the client sees."""

    model_config = ConfigDict(extra="allow")

    type: str
    seq: int
    ts: str
    # payload fields live as top-level extras


def job_to_run_summary(job: JobDetail, *, iterations_requested: int) -> RunSummary:
    params = job.params or {}
    completed = 0
    current_iter: Optional[int] = None
    metric_history: list[Optional[float]] = []
    fitness_history: list[Optional[float]] = []
    # These fields are populated by the run manager as events are seen.
    result = job.result or {}
    iters_info = result.get("iterations") or []
    if isinstance(iters_info, list):
        completed = sum(1 for it in iters_info if it.get("status") == "completed")
        metric_history = [
            (it.get("primary_metric") if isinstance(it, dict) else None)
            for it in iters_info
        ]
        # §Ship 35: parallel objective-fitness history (None where absent).
        fitness_history = [
            (it.get("fitness") if isinstance(it, dict) else None)
            for it in iters_info
        ]
        running = [it for it in iters_info if it.get("status") == "running"]
        if running:
            current_iter = int(running[0].get("iter_index") or 0)
    return RunSummary(
        run_id=job.job_id,
        project_slug=job.project_slug or "",
        status=job.status,
        behavior_goal=str(params.get("behavior_goal") or ""),
        iterations_requested=iterations_requested,
        iterations_completed=completed,
        current_iter_index=current_iter,
        primary_metric_history=metric_history,
        fitness_history=fitness_history,
        started_at=job.started_at,
        ended_at=job.ended_at,
        error=job.error,
    )
