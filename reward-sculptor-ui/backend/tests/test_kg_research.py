"""Tests for POST /projects/{slug}/kg/research (M7 Phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project(client: TestClient, name: str = "ResearchProj") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def test_research_validation_rejects_empty_topic(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    r = client.post(
        f"/projects/{slug}/kg/research",
        json={"topic": "", "max_papers": 5},
    )
    assert r.status_code == 422, r.text


def test_research_validation_rejects_max_papers_out_of_range(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    r = client.post(
        f"/projects/{slug}/kg/research",
        json={"topic": "SEA dynamics", "max_papers": 50},
    )
    assert r.status_code == 422


def test_research_503_when_api_key_missing(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    slug = _make_project(client)

    r = client.post(
        f"/projects/{slug}/kg/research",
        json={"topic": "SEA dynamics", "max_papers": 5},
    )
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["type"] == "/problems/anthropic-key-missing"


def test_research_404_on_unknown_slug(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    r = client.post(
        "/projects/nope/kg/research",
        json={"topic": "SEA dynamics"},
    )
    assert r.status_code == 404


def test_research_409_when_kg_job_already_running(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager

    from backend.services.job_manager import Job
    jm._jobs["blocking"] = Job(
        job_id="blocking",
        kind="kg_ingest_extract",
        project_slug=slug,
        status="running",
    )

    r = client.post(
        f"/projects/{slug}/kg/research",
        json={"topic": "SEA dynamics"},
    )
    assert r.status_code == 409
    body = r.json()
    assert body["type"] == "/problems/job-busy"


def test_research_submits_job_and_returns_202(
    client: TestClient,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: endpoint fires a job and returns a 202 with its
    handle. The job's Claude call is stubbed out via module-level
    patch of `research_topic` so no real API call happens."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    # Stub research_topic to return a single fake paper. Import here
    # so we patch the already-imported module (routes/kg_jobs).
    from sculptor.kg import research as research_mod

    def fake_research_topic(topic, *, max_papers, store, dedupe_against_kg, **_):
        assert topic == "SEA dynamics"
        return research_mod.ResearchResponse(
            papers=[
                research_mod.ResearchPaper(
                    arxiv_id="2401.16337",
                    title="SEA Dynamics Paper",
                    relevance_score=0.9,
                    justification="directly on-topic",
                )
            ],
            coverage_note="",
        )

    monkeypatch.setattr(
        research_mod, "research_topic", fake_research_topic
    )

    r = client.post(
        f"/projects/{slug}/kg/research",
        json={"topic": "SEA dynamics", "max_papers": 5, "auto_extract": False},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "kg_research"
    assert body["project_slug"] == slug
    assert body["status"] in ("queued", "running")


# ── unit tests over the research module itself ──────────────────────
def test_research_normalize_arxiv_id_accepts_raw() -> None:
    from sculptor.kg.research import _normalize_arxiv_id

    assert _normalize_arxiv_id("2401.16337") == "2401.16337"
    assert _normalize_arxiv_id("1707.06347") == "1707.06347"


def test_research_normalize_arxiv_id_strips_version_and_prefix() -> None:
    from sculptor.kg.research import _normalize_arxiv_id

    assert _normalize_arxiv_id("2401.16337v2") == "2401.16337"
    assert _normalize_arxiv_id("arxiv:1707.06347") == "1707.06347"
    assert _normalize_arxiv_id("arXiv:1707.06347V3") == "1707.06347"
    assert _normalize_arxiv_id("https://arxiv.org/abs/2107.04034") == "2107.04034"


def test_research_normalize_arxiv_id_rejects_garbage() -> None:
    from sculptor.kg.research import _normalize_arxiv_id

    assert _normalize_arxiv_id("") is None
    assert _normalize_arxiv_id("not an id") is None
    assert _normalize_arxiv_id("20AA.16337") is None
    assert _normalize_arxiv_id("10.1109/whatever") is None


def test_research_topic_rejects_empty_or_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.kg.research import research_topic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    with pytest.raises(ValueError, match="non-empty"):
        research_topic("")
    with pytest.raises(ValueError, match="≤ 500"):
        research_topic("x" * 501)
    with pytest.raises(ValueError, match="max_papers"):
        research_topic("ok", max_papers=100)
    with pytest.raises(ValueError, match="max_papers"):
        research_topic("ok", max_papers=0)
