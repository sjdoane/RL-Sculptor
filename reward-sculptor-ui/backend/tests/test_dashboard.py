from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_dashboard_empty_install(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.get("/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert body["project_count"] == 0
    assert body["has_any_projects"] is False
    assert body["active_jobs"] == []
    assert body["recent_runs"] == []
    assert body["recent_kg_additions"] == []


def test_dashboard_populated(
    client: TestClient, tmp_projects_root: Path
) -> None:
    client.post(
        "/projects",
        json={"name": "Alpha", "iteration_budget": 5, "behavior_goal": "hop"},
    )
    client.post(
        "/projects",
        json={"name": "Beta", "iteration_budget": 10, "behavior_goal": "run"},
    )
    r = client.get("/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert body["project_count"] == 2
    assert body["has_any_projects"] is True


def test_system_info_reports_sculptor_and_api_key(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1234567890abcdef")
    r = client.get("/system/info")
    assert r.status_code == 200
    body = r.json()
    assert body["sculptor_ok"] is True
    assert body["anthropic_api_key_set"] is True
    # Masked — last 4 visible.
    assert body["anthropic_api_key_masked"].endswith("cdef")
    assert "sk-ant-1234567890" not in body["anthropic_api_key_masked"]
    assert "projects_root" in body


def test_system_info_no_api_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.get("/system/info")
    body = r.json()
    assert body["anthropic_api_key_set"] is False
    assert body["anthropic_api_key_masked"] is None
