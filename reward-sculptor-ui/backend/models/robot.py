"""Pydantic models for robot-source and preview endpoints.

Library-picked robots carry an `env_id` derived from a fixed mapping;
uploaded MJCF/URDF carry a file path relative to the project dir.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


# Fixed library mapping. Keys are the stable wire-format names; values
# are the Gymnasium env ids the sculpt loop trains against. Extending
# this requires bumping the schema version in metadata.json.
LIBRARY_ROBOT_MAP: dict[str, str] = {
    "hopper":       "Hopper-v4",
    "half_cheetah": "HalfCheetah-v4",
    "ant":          "Ant-v4",
    "humanoid":     "Humanoid-v4",
    "walker2d":     "Walker2d-v4",
}

LibraryRobotName = Literal[
    "hopper", "half_cheetah", "ant", "humanoid", "walker2d"
]

RobotKind = Literal["library", "mjcf", "urdf", "none"]


class LibraryPickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    robot_name: LibraryRobotName


class RobotStateResponse(BaseModel):
    """Current robot source for a project.

    `env_id` is always populated for library picks; for uploads it is
    the project's `config.toml:[adapter].config.env_id` which typically
    remains unset / `CHANGE_ME` (the adapter needs custom wiring).
    """

    model_config = ConfigDict(extra="forbid")

    kind: RobotKind
    library_name: Optional[str] = None
    env_id: Optional[str] = None
    model_file: Optional[str] = None          # relative path under <project>/
    original_filename: Optional[str] = None
    size_bytes: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    mesh_paths: list[str] = []                # relative to <project>/
