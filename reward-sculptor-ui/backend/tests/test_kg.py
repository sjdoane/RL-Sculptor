"""Tests for KG routes + JobManager plumbing.

Avoids live arxiv/Anthropic calls by monkey-patching
`sculptor.kg.ingest.ingest_from_seeds` + `sculptor.kg.extract.extract_all`
to synthesize Paper nodes directly in the project's SculptorKG store.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project_with_library(
    client: TestClient, name: str = "KGTest"
) -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 5, "behavior_goal": "hop"},
    )
    slug = r.json()["slug"]
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    assert r.status_code == 200, r.text
    return slug


# ── pending seeds / empty KG ──────────────────────────────────────────
def test_library_pick_seeds_kg_yaml(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client, "SeedLib")
    yaml_path = tmp_projects_root / slug / "kg_seeds.yml"
    assert yaml_path.is_file()
    content = yaml_path.read_text("utf-8")
    # At least the PPO seed lands.
    assert "1707.06347" in content


def test_pending_seeds_endpoint(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client)
    r = client.get(f"/projects/{slug}/kg/pending-seeds")
    assert r.status_code == 200
    body = r.json()
    assert body["ingested_count"] == 0
    assert len(body["pending"]) >= 4


def test_list_papers_empty_kg(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client)
    r = client.get(f"/projects/{slug}/kg/papers")
    assert r.status_code == 200
    assert r.json() == []


# ── POST /kg/seeds runs the job ───────────────────────────────────────
@pytest.fixture
def mock_sculptor_ingest(monkeypatch: pytest.MonkeyPatch):
    """Replace sculptor.kg.ingest.ingest_from_seeds with a stub that
    writes Paper nodes directly into the store, skipping arxiv.org +
    the PDF fetch.
    """
    import yaml
    from sculptor.kg.schema import Paper, make_paper_id
    from sculptor.kg import ingest as _ingest_mod

    def _fake(seeds_path, *, store=None, force=False):
        seeds_path = Path(seeds_path)
        doc = yaml.safe_load(seeds_path.read_text("utf-8")) or {}
        results = {}
        for entry in doc.get("papers") or []:
            if isinstance(entry, str):
                aid = entry
                title = "Synthetic paper"
            else:
                aid = entry.get("arxiv_id")
                title = entry.get("title") or "Synthetic paper"
            if not aid:
                continue
            pid = make_paper_id(aid)
            if store.has_node(pid):
                results[aid] = "already_present"
                continue
            store.add_node(
                Paper(
                    id=pid,
                    arxiv_id=aid,
                    title=title,
                    authors=["Test"],
                    year=2024,
                    abstract="Stubbed abstract",
                    conclusion_text="",
                    extracted=False,
                )
            )
            results[aid] = "ingested"
        return results

    monkeypatch.setattr(_ingest_mod, "ingest_from_seeds", _fake)


def test_add_seeds_runs_job_and_paper_appears(
    client: TestClient,
    tmp_projects_root: Path,
    mock_sculptor_ingest,
) -> None:
    slug = _make_project_with_library(client, "AddSeeds")
    r = client.post(
        f"/projects/{slug}/kg/seeds",
        json={
            "seeds": [
                {
                    "arxiv_id": "2010.04159",
                    "title": "Test Paper",
                    "rationale": "test",
                }
            ],
            "auto_extract": False,
        },
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert job_id.startswith("job_")

    # Poll until terminal (usually <1s with the stub).
    deadline = time.time() + 15
    while time.time() < deadline:
        jr = client.get(f"/jobs/{job_id}")
        assert jr.status_code == 200
        body = jr.json()
        if body["status"] in ("completed", "errored", "stopped"):
            break
        time.sleep(0.05)
    assert body["status"] == "completed", body

    # Papers list now shows the new entry.
    r = client.get(f"/projects/{slug}/kg/papers")
    assert r.status_code == 200
    papers = r.json()
    aids = {p["arxiv_id"] for p in papers}
    assert "2010.04159" in aids


def test_add_seeds_409_when_job_already_active(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project_with_library(client, "SeedsBusy")
    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["job_fake_ingest"] = Job(
        job_id="job_fake_ingest",
        kind="kg_ingest",
        project_slug=slug,
        status="running",
    )
    r = client.post(
        f"/projects/{slug}/kg/seeds",
        json={"seeds": [{"arxiv_id": "1234.56789"}]},
    )
    assert r.status_code == 409
    assert r.json()["type"] == "/problems/job-busy"


# ── graph.html ─────────────────────────────────────────────────────────
def test_kg_graph_html_renders(
    client: TestClient, tmp_projects_root: Path, mock_sculptor_ingest
) -> None:
    slug = _make_project_with_library(client, "GraphViz")
    # Kick off an ingest so the KG has at least one paper.
    r = client.post(
        f"/projects/{slug}/kg/seeds",
        json={
            "seeds": [{"arxiv_id": "2010.04159", "title": "Test"}],
            "auto_extract": False,
        },
    )
    job_id = r.json()["job_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)

    r = client.get(f"/projects/{slug}/kg/graph.html")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    assert len(r.content) > 500  # pyvis output is ~hundreds of KB
    # Sanity: must contain the pyvis network boilerplate.
    assert b"vis" in r.content.lower()


# ── job 404 ────────────────────────────────────────────────────────────
def test_job_404(client: TestClient) -> None:
    r = client.get("/jobs/job_nope")
    assert r.status_code == 404


# ── §7.7 / §Ship-7: heal-stubs route ────────────────────────────────────
def test_heal_stubs_on_clean_kg_returns_empty_summary(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Empty KG → route succeeds (200) with summary.total == 0."""
    slug = _make_project_with_library(client, "HealClean")
    from sculptor.kg import ingest

    monkeypatch.setattr(ingest, "heal_stub_titles", lambda *a, **kw: {})

    r = client.post(f"/projects/{slug}/kg/heal-stubs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["total"] == 0
    assert body["summary"]["healed"] == 0
    assert body["results"] == {}


def test_heal_stubs_counts_results_correctly(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Heal route's summary counters must match per-paper result strings."""
    slug = _make_project_with_library(client, "HealCounts")
    from sculptor.kg import ingest

    monkeypatch.setattr(ingest, "heal_stub_titles", lambda *a, **kw: {
        "9999.00001": "healed",
        "9999.00002": "healed",
        "9999.00003": "still_stubbed",
        "9999.00004": "error: arxiv 429",
    })

    r = client.post(f"/projects/{slug}/kg/heal-stubs")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 4
    assert body["summary"]["healed"] == 2
    assert body["summary"]["still_stubbed"] == 1
    assert body["summary"]["errored"] == 1


def test_heal_stubs_unknown_project_404(client: TestClient) -> None:
    r = client.post("/projects/does-not-exist/kg/heal-stubs")
    assert r.status_code == 404
