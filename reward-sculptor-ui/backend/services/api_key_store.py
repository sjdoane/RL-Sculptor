"""Local persistence for the Anthropic API key configured from the UI.

The control panel is localhost-only, and stores the key beside the other
user-scoped RewardSculptor settings rather than inside either source tree.
The file is written atomically with owner-only permissions.  Callers only
ever receive a masked status; the saved value is never returned by an API.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def key_file(projects_root: Path) -> Path:
    return projects_root / "_settings" / "anthropic_api_key"


def mask_key(key: str) -> str:
    if len(key) <= 4:
        return "*" * len(key)
    return f"{'*' * min(len(key) - 4, 12)}{key[-4:]}"


def load_saved_key(projects_root: Path) -> bool:
    """Load the UI-saved key when the process has no environment key."""
    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return False
    path = key_file(projects_root)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    if not value:
        return False
    os.environ["ANTHROPIC_API_KEY"] = value
    return True


def save_key(projects_root: Path, value: str) -> Path:
    """Persist *value* atomically and activate it for this backend."""
    cleaned = value.strip()
    if len(cleaned) < 20 or any(ch.isspace() for ch in cleaned):
        raise ValueError(
            "API key must be at least 20 characters with no whitespace."
        )

    path = key_file(projects_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".anthropic_api_key.", dir=str(path.parent), text=True
    )
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cleaned)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        path.chmod(0o600)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    os.environ["ANTHROPIC_API_KEY"] = cleaned
    return path
