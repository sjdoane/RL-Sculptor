"""Pydantic shapes for chunk A1 — non-destructive delete / trash.

`TrashEntry` mirrors the dict shape `backend.services.trash` reads/
writes on disk (`trash_meta.json` + the `entry_id` derived from the
directory name) — see that module's docstring for the on-disk layout.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrashEntry(BaseModel):
    """One row of `GET /trash`. `extra="allow"` so a future
    `extra_meta` field written by a caller of `move_to_trash` doesn't
    need a schema bump to round-trip through this response."""

    model_config = ConfigDict(extra="allow")

    entry_id: str
    kind: Optional[str] = None  # "project" | "mission" | None (unreadable)
    slug: str
    origin_path: Optional[str] = None
    deleted_at: Optional[str] = None
    display_name: str
    size_bytes: Optional[int] = None
    # True when trash_meta.json was missing/corrupt — the entry is
    # still listed (never hidden) but restore() will 409 on it.
    unreadable: bool = False


class RestoreTrashResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    restored_path: str


class PurgeTrashRequest(BaseModel):
    """Body for `DELETE /trash/{entry_id}` — the confirm field must
    echo the entry_id being purged so a stray/automated DELETE can't
    silently nuke the wrong entry."""

    model_config = ConfigDict(extra="forbid")

    confirm: Annotated[str, Field(min_length=1)]
