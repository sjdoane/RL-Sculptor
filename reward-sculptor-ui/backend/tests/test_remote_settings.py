"""GET/PUT /system/remote + POST /system/remote/doctor + env injection
(§Ship 23d)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.services.remote_settings import (
    RemoteSettings,
    load_remote_settings,
    remote_env,
    save_remote_settings,
    settings_path,
)

OK_REPORT = {
    "ok": True,
    "host": "1.2.3.4",
    "port": 22,
    "checks": [
        {"name": "ssh reachable", "ok": True, "detail": "reachable"},
    ],
}


# ── service layer ────────────────────────────────────────────────────


def test_load_defaults_when_missing(tmp_path: Path) -> None:
    s = load_remote_settings(tmp_path)
    assert s.enabled is False
    assert s.host == ""
    assert s.port == 22
    assert s.rollout_remote is False


def test_save_load_round_trip(tmp_path: Path) -> None:
    s = RemoteSettings(
        enabled=True, host="1.2.3.4", port=41234, user="root",
        key_path="~/.ssh/id_ed25519", device="cuda:0",
    )
    save_remote_settings(tmp_path, s)
    assert settings_path(tmp_path).is_file()
    loaded = load_remote_settings(tmp_path)
    assert loaded == s


def test_corrupt_file_loads_defaults(tmp_path: Path) -> None:
    p = settings_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    s = load_remote_settings(tmp_path)
    assert s == RemoteSettings()


def test_remote_env_three_states(tmp_path: Path) -> None:
    # 1. Never saved → {} (a project's [remote] TOML table may apply).
    assert remote_env(tmp_path) == {}
    # 2. Saved but off (or hostless) → explicit ENABLED=0: the UI toggle
    #    showing "Off" must override a TOML `enabled = true` (env wins
    #    in RemoteConfig.from_sources).
    save_remote_settings(tmp_path, RemoteSettings(enabled=False, host="1.2.3.4"))
    assert remote_env(tmp_path) == {"SCULPTOR_REMOTE_ENABLED": "0"}
    save_remote_settings(tmp_path, RemoteSettings(enabled=True, host=""))
    assert remote_env(tmp_path) == {"SCULPTOR_REMOTE_ENABLED": "0"}


def test_remote_env_mapping(tmp_path: Path) -> None:
    save_remote_settings(tmp_path, RemoteSettings(
        enabled=True, host="1.2.3.4", port=41234, user="root",
        key_path="/keys/k", device="cuda:1", rollout_remote=True,
    ))
    env = remote_env(tmp_path)
    assert env["SCULPTOR_REMOTE_ENABLED"] == "1"
    assert env["SCULPTOR_REMOTE_HOST"] == "1.2.3.4"
    assert env["SCULPTOR_REMOTE_PORT"] == "41234"
    assert env["SCULPTOR_REMOTE_USER"] == "root"
    assert env["SCULPTOR_REMOTE_KEY_PATH"] == "/keys/k"
    assert env["SCULPTOR_REMOTE_DEVICE"] == "cuda:1"
    assert env["SCULPTOR_REMOTE_ROLLOUT"] == "1"
    assert env["SCULPTOR_REMOTE_WORKDIR"] == "~/.sculptor_remote"
    assert env["SCULPTOR_REMOTE_PYTHON"] == "~/.sculptor_remote/venv/bin/python"


def test_remote_env_omits_blank_optionals(tmp_path: Path) -> None:
    save_remote_settings(tmp_path, RemoteSettings(enabled=True, host="h"))
    env = remote_env(tmp_path)
    assert "SCULPTOR_REMOTE_USER" not in env
    assert "SCULPTOR_REMOTE_KEY_PATH" not in env
    assert "SCULPTOR_REMOTE_DEVICE" not in env
    assert env["SCULPTOR_REMOTE_ROLLOUT"] == "0"


# ── routes ───────────────────────────────────────────────────────────


def test_get_remote_defaults(client: TestClient) -> None:
    r = client.get("/system/remote")
    assert r.status_code == 200
    data = r.json()
    assert data["enabled"] is False
    assert data["host"] == ""
    assert data["port"] == 22


def test_put_get_round_trip_and_persistence(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    body = {
        "enabled": True,
        "host": "1.2.3.4",
        "port": 41234,
        "user": "root",
        "key_path": "~/.ssh/id_ed25519",
        "remote_workdir": "/workspace/sculptor_remote",
        "remote_python": "/workspace/sculptor_remote/venv/bin/python",
        "device": "cuda:0",
        "rollout_remote": False,
    }
    r = client.put("/system/remote", json=body)
    assert r.status_code == 200
    assert r.json() == body

    r2 = client.get("/system/remote")
    assert r2.json() == body

    # Persisted at <projects_root>/_settings/remote.json.
    on_disk = json.loads(
        (tmp_projects_root / "_settings" / "remote.json").read_text("utf-8")
    )
    assert on_disk == body


def test_put_invalid_port_is_422(client: TestClient) -> None:
    r = client.put("/system/remote", json={"host": "h", "port": 99999})
    assert r.status_code == 422


def test_put_leading_dash_host_is_422(client: TestClient) -> None:
    """A host like '-oProxyCommand=…' would become an ssh OPTION."""
    r = client.put("/system/remote", json={"host": "-oProxyCommand=evil"})
    assert r.status_code == 422
    r2 = client.put("/system/remote", json={"host": "h", "user": "-x"})
    assert r2.status_code == 422


def test_doctor_unconfigured_reports_not_ok(client: TestClient) -> None:
    r = client.post("/system/remote/doctor")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert any("no remote host configured" in c["detail"] for c in data["checks"])


def test_doctor_uses_saved_settings(client: TestClient) -> None:
    client.put("/system/remote", json={"enabled": False, "host": "1.2.3.4"})
    captured: dict = {}

    def fake_doctor(settings):
        captured["settings"] = settings
        return OK_REPORT

    with patch("backend.routes.system.run_doctor", side_effect=fake_doctor):
        r = client.post("/system/remote/doctor")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["checks"][0]["name"] == "ssh reachable"
    # Doctor runs against the SAVED settings, even before enabled=true.
    assert captured["settings"].host == "1.2.3.4"


# ── subprocess env injection (run_manager / mission_jobs read this) ──


def test_remote_env_reads_projects_root_layout(tmp_path: Path) -> None:
    """run_manager passes project_dir.parent — verify the layout
    assumption end to end: settings at <root>/_settings/remote.json
    apply to a project at <root>/<slug>."""
    root = tmp_path / "projects"
    project_dir = root / "my-proj"
    project_dir.mkdir(parents=True)
    save_remote_settings(root, RemoteSettings(enabled=True, host="9.9.9.9"))
    env = remote_env(project_dir.parent)
    assert env["SCULPTOR_REMOTE_HOST"] == "9.9.9.9"
