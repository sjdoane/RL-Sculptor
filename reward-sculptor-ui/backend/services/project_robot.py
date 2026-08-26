"""Backend import boundary for the shared project robot resolver."""

from sculptor.project_robot import (
    LEGACY_REFERENCE_ROBOT_BY_LIBRARY_SLUG,
    resolve_project_reference_robot,
)

__all__ = [
    "LEGACY_REFERENCE_ROBOT_BY_LIBRARY_SLUG",
    "resolve_project_reference_robot",
]
