"""Settings + cloud-sync guard.

The only source of config for the backend. Reads environment variables
prefixed with `RS_` and an optional `.env` file at the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Path segments that indicate a cloud-synced directory. Matched
# case-insensitively against each part of the resolved project root.
CLOUD_SYNC_SEGMENTS: tuple[str, ...] = (
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "icloud drive",
    "icloudrive",
    "icloud",
    "box",
    "pclouddrive",
    "pcloud drive",
)


def _platform_default_projects_root() -> Path:
    """Platform-appropriate default for $RS_PROJECTS_ROOT.

    Windows: `%LOCALAPPDATA%\\reward-sculptor\\projects\\`
    POSIX:   `~/.local/share/reward-sculptor/projects/`
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "reward-sculptor" / "projects"
        return Path.home() / "AppData" / "Local" / "reward-sculptor" / "projects"
    return Path.home() / ".local" / "share" / "reward-sculptor" / "projects"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    projects_root: Path = Field(default_factory=_platform_default_projects_root)
    allow_cloud_sync: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # Per Prompt 9 R1: live-rollout clip rendering defaults to CPU so
    # it can't steal VRAM from the active PPO training job. Today the
    # live-clip pipeline is pure ffmpeg truncation over an existing
    # rollout.mp4 (already rendered by the adapter, which uses
    # Gymnasium's rgb_array CPU backend), so this knob is effectively
    # documentation — but we honor it when the streamer ever needs to
    # re-render from a mid-training checkpoint.
    live_rollout_device: Literal["cpu", "gpu"] = "cpu"

    @property
    def resolved_projects_root(self) -> Path:
        return self.projects_root.expanduser().resolve()


def check_cloud_sync(path: Path) -> Optional[str]:
    """Return the offending segment name if `path` lives in a cloud-synced
    directory, else None. Matching is case-insensitive, per-path-segment."""
    parts_lower = [p.lower() for p in path.parts]
    for seg in CLOUD_SYNC_SEGMENTS:
        if seg in parts_lower:
            return seg
    return None


def abort_on_cloud_sync(settings: Settings) -> None:
    """Refuse to proceed if the projects root is cloud-synced, unless
    `RS_ALLOW_CLOUD_SYNC` was set. Raises SystemExit(2) on refusal."""
    root = settings.resolved_projects_root
    hit = check_cloud_sync(root)
    if hit is None:
        return
    if settings.allow_cloud_sync:
        sys.stderr.write(
            f"[reward-sculptor-ui] WARNING: projects root {root} is inside "
            f"a cloud-synced directory ({hit!r}); proceeding because "
            "RS_ALLOW_CLOUD_SYNC=true. sqlite/ffmpeg conflicts are possible.\n"
        )
        return
    sys.stderr.write(
        "\nFATAL: RS_PROJECTS_ROOT resolves to a cloud-synced directory:\n"
        f"  {root}\n"
        f"Matched segment: {hit!r}\n"
        "Cloud-sync causes file-lock conflicts with sqlite + ffmpeg and can\n"
        "silently corrupt runs. Move RS_PROJECTS_ROOT outside your cloud\n"
        "folder, or set RS_ALLOW_CLOUD_SYNC=true to override.\n\n"
    )
    raise SystemExit(2)
