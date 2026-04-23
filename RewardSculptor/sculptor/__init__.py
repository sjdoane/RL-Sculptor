"""Reward Sculptor — autonomous PPO reward iteration grounded in a KG."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def _load_dotenv_best_effort() -> None:
    """Populate os.environ from a sibling `.env`. Treat present-but-empty
    environment variables as unset (Claude Code exports an empty
    `ANTHROPIC_API_KEY` which would otherwise block the .env value). Silent
    on missing file or missing `python-dotenv`."""
    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError:
        return
    env_file = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            env_file = candidate
            break
    if env_file is None:
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file():
            env_file = cwd_env
    if env_file is None:
        return
    for key, value in dotenv_values(env_file).items():
        if value is None:
            continue
        existing = os.environ.get(key)
        if existing is None or existing == "":
            os.environ[key] = value


_load_dotenv_best_effort()
del _load_dotenv_best_effort, os, Path
