"""sculptor/prompts — loaded-at-runtime system prompts.

All Claude API system prompts used by the sculptor pipeline live in this
package as `.md` files so lab adopters can tune tone, constraints, or
domain phrasing for their task without touching Python.

Usage:
    from sculptor.prompts import load_prompt
    SYSTEM = load_prompt("diagnose_preliminary")

Adopters who want to ship domain-specific variants can either:
  * drop replacement `.md` files in this directory, or
  * set the `SCULPTOR_PROMPTS_DIR` environment variable to an alternative
    directory containing the same filenames (a per-project override).
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _prompts_dir() -> Path:
    override = os.environ.get("SCULPTOR_PROMPTS_DIR")
    if override:
        d = Path(override).expanduser().resolve()
        if d.is_dir():
            return d
    return _PACKAGE_DIR


def load_prompt(name: str) -> str:
    """Read `<prompts_dir>/<name>.md` as a string.

    `name` is the filename without extension, e.g. `"diagnose_preliminary"`.
    """
    path = _prompts_dir() / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"prompt {name!r} not found at {path} "
            f"(override via SCULPTOR_PROMPTS_DIR={os.environ.get('SCULPTOR_PROMPTS_DIR') or '(unset)'})"
        )
    return path.read_text(encoding="utf-8")


def available_prompts() -> list[str]:
    """Return the stems of every .md file in the prompts dir."""
    return sorted(p.stem for p in _prompts_dir().glob("*.md"))
