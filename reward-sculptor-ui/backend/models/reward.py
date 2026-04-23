"""Pydantic shapes for reward-editing endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RewardRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    arxiv_id: str
    citation: str = ""


class RewardSpec(BaseModel):
    """Mirror of the REWARD_SPEC dict literal written into v<n>.py."""

    model_config = ConfigDict(extra="allow")
    version: str
    parent_hash: str = ""
    author: str = "unknown"
    description: str = ""
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    references: list[RewardRef] = Field(default_factory=list)
    # Map of `hparam_name → citation-or-justification` for every
    # hyperparameter change. Added by the 2026-04-22 05:30 KG-grounding
    # mandate — every new/changed hp must be accompanied by a specific
    # arxiv_id (one that appears in `references`) or a physics-first-
    # principles justification. Older pre-mandate rewards have an empty
    # dict; the UI renders `(pre-grounding-mandate)` for them.
    grounding: dict[str, str] = Field(default_factory=dict)


class ComponentProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    components: Optional[dict[str, float]] = None
    total: Optional[float] = None
    error: Optional[str] = None


class RewardVersionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    file_name: str
    author: Literal["sculptor", "human", "unknown"]
    parent_hash: str
    parent_version: Optional[int]
    description: str
    iter_introduced: Optional[int]
    n_references: int
    primary_metric: Optional[float]
    metric_delta: Optional[float]
    created_at: datetime


class RewardVersionDetail(RewardVersionSummary):
    source: str
    spec: RewardSpec
    components_probe: ComponentProbe


class ManualEditRequest(BaseModel):
    """PUT /rewards/{v} body — user submits a replacement reward module.

    `expected_parent_version` must equal the current latest version's
    integer. If they diverge the server returns 409 and the UI discards
    the draft (no auto-merge in v1, per Prompt 7 precondition R1).
    """

    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=10, max_length=200_000)
    expected_parent_version: int = Field(ge=0)
    note: str = Field(default="", max_length=500)
