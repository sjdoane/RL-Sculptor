"""Pydantic response shapes for the /library/* endpoints."""

from __future__ import annotations

from typing import Literal, Optional  # noqa: F401

from pydantic import BaseModel, ConfigDict


class LibraryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["paper", "repo"]
    url: str
    citation: str = ""


class LibraryPreconfiguredTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    display_name: str
    recommended_num_envs: int


class LibraryRobotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    display_name: str
    category: str
    description: str = ""
    source: str
    menagerie_package: Optional[str] = None
    training_support: str
    is_smoke_test_target: bool = False
    preconfigured_tasks: list[LibraryPreconfiguredTask] = []
    references: list[LibraryReference] = []
    thumbnail_path: str
    demote_note: Optional[str] = None


class LibraryListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robots: list[LibraryRobotResponse]
    total: int
    categories: list[str]


class AdapterInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    class_path: str
    status: Literal["ready", "coming_soon"]
    supported_robot_categories: list[str]
    adoption_guide_url: str
    estimated_effort: str = ""
