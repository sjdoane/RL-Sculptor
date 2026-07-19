"""Environment-authoring request/response shapes (env-authoring item 5).

Responses are intentionally loose (``extra="allow"``): the authoritative
shapes are the sculptor world dataclasses' ``to_dict()`` output, and the
backend must not silently truncate fields the frontend may want.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorWorldRequest(BaseModel):
    prompt: str = Field(min_length=1, description=(
        "Natural-language environment and task description."))
    robot_capability_id: str | None = Field(
        default=None, description="Pin a robot; auto-selected when omitted.")
    kg_grounding: bool = Field(
        default=True, description=(
            "Ground authoring in the shared knowledge graph "
            "(best-effort retrieval; never blocks authoring)."))


class ClarificationAnswerBody(BaseModel):
    question_id: str
    choice_id: str = Field(description=(
        "A listed choice_id, or 'system_default' to hand the decision "
        "back to the system."))


class ApplyWorldRequest(BaseModel):
    session_id: str
    answers: list[ClarificationAnswerBody] = Field(
        default_factory=list, description=(
            "Explicit answers; every unanswered question takes its "
            "disclosed system default."))


class WorldDraftSummary(BaseModel, extra="allow"):
    session_id: str
    draft_hash: str
    capability_id: str


class WorldApplySummary(BaseModel, extra="allow"):
    ok: bool
    session_id: str
    evaluation_lineage: str
