"""Pydantic models for the project-lifecycle endpoints.

Mirrors the shapes from the API contract §2. Only the subset used by
Prompt 3 endpoints is defined here — full contract shapes land in later
prompts as their endpoints are implemented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# Immutable directory name; used as project primary key in URLs.
SlugStr = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$",
        min_length=1,
        max_length=64,
    ),
]
DisplayNameStr = Annotated[str, Field(min_length=1, max_length=120)]

ProjectStatus = Literal[
    "draft", "configured", "ready", "running", "completed", "errored"
]


class CreateProjectRequest(BaseModel):
    """POST /projects body.

    `display_name` also accepts the alias `name` so the simple smoke-test
    curl from Prompt 3 works without modification.

    `iteration_budget` and `behavior_goal` capture the form's intent at
    creation time. They are persisted in the sidecar and used as defaults
    when the user starts the first sculpt run. Neither affects
    `config.toml` directly in v1.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    display_name: DisplayNameStr = Field(
        validation_alias=AliasChoices("display_name", "name")
    )
    adapter: str = "gym_sb3"
    description: str = ""
    iteration_budget: int = Field(default=20, ge=1, le=1000)
    behavior_goal: str = Field(default="", max_length=2000)
    # mjlab-specific fields (ignored for other adapters). Set when
    # adapter == "mjlab". MJLAB_PIVOT_DESIGN §9.
    task_id: Optional[str] = None
    num_envs: Optional[int] = Field(default=None, ge=1, le=65536)
    gpu_device: Optional[str] = None  # e.g. "cuda:0"
    # M3: library-driven scaffolding. When set, backend resolves the
    # library entry and derives adapter / task_id / num_envs from it,
    # overriding any same-named fields in this request. Also triggers
    # auto-KG-seeding from the library's references list.
    library_slug: Optional[str] = None


class ProjectSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: SlugStr
    display_name: DisplayNameStr
    description: str = ""
    status: ProjectStatus
    created_at: datetime
    env_id: Optional[str] = None
    n_iterations_completed: int = 0
    # M6: non-null when the project's config.toml references an adapter
    # that no longer exists in `sculptor.adapters.ADAPTER_REGISTRY`.
    # UI shows a banner; "upgrade" is a fork (create parallel project).
    migration_warning: Optional[str] = None


class ProjectDetail(ProjectSummary):
    project_dir: str
    adapter_class: str
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    # True when the project's adapter + robot combination can actually
    # kick off training. False for preview_only library entries (where
    # the UI should disable the Train button). Populated by the store
    # after cross-referencing metadata.json's robot_source.
    ready_to_train: bool = True
    library_slug: Optional[str] = None
    # M5: true when the user created a project with a coming-soon
    # adapter (Isaac Lab / MJX / RLlib). UI disables the Train button
    # with a tooltip pointing at the adoption guide.
    adapter_unavailable: bool = False


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    sculptor_ok: bool
    sculptor_error: Optional[str] = None
    projects_root: str


class IterationSettings(BaseModel):
    """§Ship-8: persisted `[iteration]` config for a project.

    All fields are optional — PATCH merges them onto the existing
    config.toml [iteration] table. Unset fields are left untouched.
    Returned shape on GET is the literal current values, so the UI
    can show what's on disk rather than re-reading defaults.
    """

    model_config = ConfigDict(extra="forbid")

    steps_per_iter: Optional[Annotated[int, Field(ge=100, le=500_000)]] = None
    primary_metric: Optional[str] = None
    behavior_metrics: Optional[list[str]] = None
    rollout_episodes: Optional[Annotated[int, Field(ge=1, le=32)]] = None
    auto_adjust_physics: Optional[bool] = None
    # Rollout video + RL knobs (mirror RunParams per-run overrides but
    # persisted at the project level).
    max_episode_steps: Optional[Annotated[int, Field(ge=50, le=5000)]] = None
    playback_speed: Optional[Annotated[float, Field(gt=0.1, le=10.0)]] = None
    render_every: Optional[Annotated[int, Field(ge=1, le=100)]] = None
    rollout_fps: Optional[Annotated[float, Field(gt=0, le=240)]] = None
    seed: Optional[Annotated[int, Field(ge=0, le=2**31 - 1)]] = None
    # §Ship-9a: early-stop knobs (project defaults — per-run in RunParams).
    early_stop_enabled: Optional[bool] = None
    early_stop_patience: Optional[Annotated[int, Field(ge=1, le=100)]] = None


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: IterationSettings


class ProblemDetail(BaseModel):
    """RFC 7807 problem-details shape. Served with
    `application/problem+json` media type."""

    model_config = ConfigDict(extra="allow")

    type: str = "about:blank"
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None
