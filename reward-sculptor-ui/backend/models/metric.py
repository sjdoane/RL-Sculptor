"""§Ship 35: auto-generated objective-metric API shapes.

A generated metric is created from a behavior goal, validated + reviewed,
and stored per-project at `<project>/metrics/<id>/`. It runs OBSERVE-ONLY
until calibrated against a hand-authored ground-truth metric (Spearman),
which earns it steer-rights. The UI references it in a run as
`fitness_metric = "gen:<id>"`.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerateMetricRequest(BaseModel):
    """POST /projects/{slug}/metrics/generate body."""

    model_config = ConfigDict(extra="forbid")

    behavior_goal: Annotated[str, Field(min_length=4, max_length=500)]
    review: bool = True
    #: optional built-in metric to immediately calibrate against
    #: (earns steer-rights if Spearman >= threshold).
    calibrate_against: Optional[str] = None


class MetricSummary(BaseModel):
    """One generated metric — what the UI shows for review + selection."""

    model_config = ConfigDict(extra="allow")

    id: str
    behavior_goal: Optional[str] = None
    accepted: bool = False
    validation_passed: bool = False
    calibrated: bool = False
    review: Optional[dict] = None
    gates: Optional[dict] = None
    reasons: Optional[list] = None
    archetype_scores: Optional[dict] = None
    calibration: Optional[dict] = None
    source: Optional[str] = None
    recorded_at: Optional[str] = None
