"""Pydantic shapes for KG-browser endpoints + job handles."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ArxivIdStr = Annotated[
    str,
    Field(
        pattern=r"^\d{4}\.\d{4,5}(v\d+)?$",
        description="arxiv id in the modern YYMM.NNNNN[vK] form",
    ),
]


class SeedPaperInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arxiv_id: ArxivIdStr
    title: Optional[str] = None
    rationale: Optional[str] = None


class AddSeedsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seeds: Annotated[
        list[SeedPaperInput],
        Field(min_length=1, max_length=50),
    ]
    auto_extract: bool = True


class ResearchTopicRequest(BaseModel):
    """POST /projects/{slug}/kg/research — Claude picks relevant arxiv
    IDs for the topic, dedupes against the shared KG, ingests + (by
    default) extracts them.
    """

    model_config = ConfigDict(extra="forbid")
    topic: Annotated[
        str,
        Field(
            min_length=3, max_length=500,
            description=(
                "Natural-language topic (e.g. 'SEA physics parameters', "
                "'quadruped jumping curriculum'). Fed to the librarian "
                "prompt verbatim."
            ),
        ),
    ]
    max_papers: int = Field(default=10, ge=1, le=20)
    auto_extract: bool = True


class PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    arxiv_id: str
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    extracted: bool
    has_pdf: bool


class KGEntitySummary(BaseModel):
    """One of the entities attached to a paper — technique / failure mode /
    reward component / environment."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal[
        "Technique", "FailureMode", "RewardComponent", "Environment"
    ]
    name: str
    description: str = ""


class PaperEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    techniques: list[KGEntitySummary] = []
    failure_modes: list[KGEntitySummary] = []
    reward_components: list[KGEntitySummary] = []
    environments: list[KGEntitySummary] = []


class PaperDetail(PaperSummary):
    abstract: str = ""
    conclusion_text: str = ""
    pdf_available: bool
    ingested_at: Optional[datetime] = None
    entities: PaperEntities


class TechniqueSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    description: str = ""
    tags: list[str] = []
    n_addresses: int = 0
    introduced_by: list[str] = []


# ── Jobs ──────────────────────────────────────────────────────────────
JobKind = Literal[
    "kg_ingest", "kg_extract", "kg_ingest_extract",
    "kg_research", "kg_viz_render", "sculpt_run",
    "reward_prompt_edit", "physics_prompt_edit",
    # §Ship 18a — mission orchestrator job kinds.
    "mission_decompose", "mission_execute",
    # §Ship 21 — per-stage child job created on `stage_started` so
    # mission stage runs are first-class entries in /runs.
    "mission_stage_run",
]
JobStatus = Literal["queued", "running", "completed", "errored", "stopped"]


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    kind: JobKind
    project_slug: Optional[str]
    status: JobStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error: Optional[str] = None
    # §Ship 21 — for mission_stage_run jobs, points at the parent
    # mission_execute job. Null for top-level jobs.
    parent_id: Optional[str] = None


class JobDetail(JobSummary):
    params: dict[str, Any] = {}
    result: Optional[dict[str, Any]] = None


# ── KG graph dump (placeholder; fuller shape lands with the viz endpoint) ──
class KGStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    papers: int = 0
    techniques: int = 0
    failure_modes: int = 0
    reward_components: int = 0
    environments: int = 0
    results: int = 0
    edges: int = 0
    embeddings: int = 0
