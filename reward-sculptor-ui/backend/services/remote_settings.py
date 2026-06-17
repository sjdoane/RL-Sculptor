"""Persisted remote-GPU dispatch settings (§Ship 23d).

One JSON file at `<projects_root>/_settings/remote.json` (global, not
per-project — there is one rented pod). The UI edits it via
GET/PUT /system/remote; run_manager / mission_jobs read it at subprocess
spawn time and inject `SCULPTOR_REMOTE_*` env vars, which win over any
`[remote]` table in a project's config.toml (see
sculptor/adapters/_remote.py::RemoteConfig.from_sources).

Doctor runs in-process: `sculptor.adapters._remote` is stdlib-only at
import time (no mjlab/torch), and its checks are ssh subprocesses — so
no heavy imports leak into the backend (the lazy-import rule).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

_SETTINGS_DIR = "_settings"
_SETTINGS_FILE = "remote.json"


class RemoteSettings(BaseModel):
    """Mirrors the connection fields of sculptor's RemoteConfig (the
    tuning knobs poll_interval_s etc. stay TOML-only by design)."""

    enabled: bool = False
    host: str = ""
    port: int = Field(default=22, ge=1, le=65535)
    user: str = ""
    key_path: str = ""
    remote_workdir: str = "~/.sculptor_remote"
    # Venv lives on pod-LOCAL disk (network-fs imports cost ~60s per
    # runner subprocess); provision_remote.sh creates it there.
    remote_python: str = "~/.sculptor_venv/bin/python"
    device: str = ""
    rollout_remote: bool = False

    @field_validator("host", "user")
    @classmethod
    def _no_leading_dash(cls, v: str) -> str:
        # A host/user starting with "-" would be parsed as an ssh OPTION
        # (e.g. -oProxyCommand=...) when the executor builds its argv.
        v = v.strip()
        if v.startswith("-"):
            raise ValueError("must not start with '-'")
        return v


def settings_path(projects_root: Path) -> Path:
    return Path(projects_root) / _SETTINGS_DIR / _SETTINGS_FILE


def load_remote_settings(projects_root: Path) -> RemoteSettings:
    """Read settings; defaults when the file is missing or corrupt
    (corrupt → defaults is deliberate: a half-written file must never
    brick the Settings page or block run launches)."""
    path = settings_path(projects_root)
    if not path.is_file():
        return RemoteSettings()
    try:
        return RemoteSettings.model_validate(json.loads(path.read_text("utf-8")))
    except Exception:  # noqa: BLE001
        return RemoteSettings()


def save_remote_settings(projects_root: Path, settings: RemoteSettings) -> None:
    """Atomic write (tmp + replace) so a crash mid-save can't leave a
    truncated JSON that load() would then silently default away."""
    path = settings_path(projects_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.model_dump(), indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remote_env(projects_root: Path) -> dict[str, str]:
    """`SCULPTOR_REMOTE_*` env vars for a sculpt subprocess.

    Three states (env wins over the project's `[remote]` TOML table in
    RemoteConfig.from_sources, so the UI toggle is authoritative once
    settings have ever been saved):
      * no settings file        → {}            (TOML/local fallthrough)
      * saved but off / no host → {ENABLED: 0}  (explicit off — a TOML
        `enabled=true` must not contradict the UI showing "Off")
      * saved + enabled + host  → full mapping
    """
    if not settings_path(projects_root).is_file():
        return {}
    s = load_remote_settings(projects_root)
    if not s.enabled or not s.host.strip():
        return {"SCULPTOR_REMOTE_ENABLED": "0"}
    env = {
        "SCULPTOR_REMOTE_ENABLED": "1",
        "SCULPTOR_REMOTE_HOST": s.host.strip(),
        "SCULPTOR_REMOTE_PORT": str(s.port),
        "SCULPTOR_REMOTE_WORKDIR": s.remote_workdir,
        "SCULPTOR_REMOTE_PYTHON": s.remote_python,
        "SCULPTOR_REMOTE_ROLLOUT": "1" if s.rollout_remote else "0",
    }
    if s.user.strip():
        env["SCULPTOR_REMOTE_USER"] = s.user.strip()
    if s.key_path.strip():
        env["SCULPTOR_REMOTE_KEY_PATH"] = s.key_path.strip()
    if s.device.strip():
        env["SCULPTOR_REMOTE_DEVICE"] = s.device.strip()
    return env


def run_doctor(settings: Optional[RemoteSettings]) -> dict[str, Any]:
    """Connectivity report for the UI's Test-connection button. Never
    raises; shape matches `sculpt remote doctor --json`:
    {ok, host, port, checks: [{name, ok, detail}]}. Blocking (ssh round
    trips, up to ~30 s on an unreachable host) — call via
    asyncio.to_thread from routes."""
    if settings is None or not settings.host.strip():
        return {
            "ok": False,
            "host": settings.host if settings else "",
            "port": settings.port if settings else 22,
            "checks": [{
                "name": "configuration",
                "ok": False,
                "detail": "no remote host configured",
            }],
        }
    try:
        import dataclasses

        from sculptor.adapters._remote import RemoteConfig, RemoteExecutor

        cfg = RemoteConfig.from_sources(
            {
                "enabled": True,  # doctor must work before enabling
                "host": settings.host.strip(),
                "port": settings.port,
                "user": settings.user.strip(),
                "key_path": settings.key_path.strip() or None,
                "remote_workdir": settings.remote_workdir,
                "remote_python": settings.remote_python,
                "device": settings.device.strip() or None,
            },
            {},  # deliberately NOT os.environ: doctor checks the saved settings
        )
        if cfg is None:  # unreachable with a non-empty host, but never 500
            raise ValueError("remote config did not resolve")
        if not cfg.enabled:
            cfg = dataclasses.replace(cfg, enabled=True)
        return RemoteExecutor(cfg).doctor()
    except Exception as e:  # noqa: BLE001 — doctor must never 500
        return {
            "ok": False,
            "host": settings.host,
            "port": settings.port,
            "checks": [{
                "name": "doctor",
                "ok": False,
                "detail": f"{type(e).__name__}: {e}",
            }],
        }
