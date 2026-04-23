"""Tests for M7 Phase 6 polish:
  - POST /jobs/{id}/stop (smoke + 404).
  - `RunSummary.error_classification` populated from
    `job.params["error_classification"]`.
  - `extract_all(progress_cb=...)` callback contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── POST /jobs/{id}/stop ─────────────────────────────────────────────
def test_stop_endpoint_returns_summary_for_known_job(
    client: TestClient, tmp_projects_root: Path
) -> None:
    import asyncio

    from backend.services.job_manager import Job

    jm = client.app.state.job_manager  # type: ignore[attr-defined]
    # Jobs created via JobManager.submit get an asyncio.Event for
    # cooperative cancel. Mirror that here — stop() returns None
    # (→ 404) without it.
    job = Job(
        job_id="test_job",
        kind="kg_ingest",
        project_slug="anything",
        status="running",
    )
    job._cancel = asyncio.Event()  # type: ignore[attr-defined]
    jm._jobs["test_job"] = job

    r = client.post("/jobs/test_job/stop")
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "test_job"


def test_stop_endpoint_404_on_unknown_job(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.post("/jobs/nope_nope/stop")
    assert r.status_code == 404


# ── RunSummary.error_classification surfacing ────────────────────────
def test_run_summary_surfaces_error_classification(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """When a sculpt_run job errors with a classification stashed in
    `params['error_classification']`, `GET /projects/{slug}/runs/
    {run_id}` returns it alongside the raw error string."""
    from backend.services.job_manager import Job

    # Create a project so the slug resolves.
    r = client.post("/projects", json={"name": "ClassProj", "adapter": "gym_sb3"})
    assert r.status_code == 201
    slug = r.json()["slug"]

    jm = client.app.state.job_manager  # type: ignore[attr-defined]
    jm._jobs["errored_run"] = Job(
        job_id="errored_run",
        kind="sculpt_run",
        project_slug=slug,
        status="errored",
        error="Reward template missing batched entry point",
        params={
            "behavior_goal": "jump",
            "iterations": 20,
            "error_classification": {
                "kind": "reward_contract_mismatch",
                "title": "Reward template missing batched entry point",
                "detail": "mjlab needs compute_reward_batched.",
                "suggestions": ["Click Regenerate on the Rewards tab."],
                "problem_type": "/problems/reward-contract-mismatch",
                "action": {
                    "kind": "regenerate_reward_template",
                    "label": "Regenerate reward template",
                },
            },
        },
    )

    r = client.get(f"/projects/{slug}/runs/errored_run")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["error"] == "Reward template missing batched entry point"
    cls = body["error_classification"]
    assert cls is not None
    assert cls["kind"] == "reward_contract_mismatch"
    assert cls["action"]["kind"] == "regenerate_reward_template"
    assert any("Regenerate" in s for s in cls["suggestions"])


def test_run_summary_error_classification_is_none_when_absent(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Runs that errored for unknown reasons (no classification in
    params) surface `error_classification=None` — not a crash."""
    from backend.services.job_manager import Job

    r = client.post("/projects", json={"name": "NoClass", "adapter": "gym_sb3"})
    slug = r.json()["slug"]
    jm = client.app.state.job_manager  # type: ignore[attr-defined]
    jm._jobs["plain_errored"] = Job(
        job_id="plain_errored",
        kind="sculpt_run",
        project_slug=slug,
        status="errored",
        error="sculpt exited with code 1",
        params={"behavior_goal": "jump", "iterations": 5},
    )

    r = client.get(f"/projects/{slug}/runs/plain_errored")
    assert r.status_code == 200
    assert r.json()["error_classification"] is None


# ── extract_all progress callback ────────────────────────────────────
def test_extract_all_invokes_progress_cb_per_unextracted_paper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`extract_all(progress_cb=fn)` calls `fn(done, total, title)`
    once per paper that will be processed, ordered by the queue."""
    from sculptor.kg.extract import extract_all
    from sculptor.kg.schema import Paper, make_paper_id
    from sculptor.kg.store import SculptorKG

    # Fake anthropic client so no network call happens.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    import anthropic

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    # Stub extract_entities so we don't actually call Claude.
    import sculptor.kg.extract as extract_mod

    def fake_extract_entities(paper, *, store, client, force):
        from sculptor.kg.extract import ExtractionResult

        # Mark as extracted so a second call with force=False skips.
        paper.extracted = True
        store.add_node(paper)
        return ExtractionResult(paper_id=paper.id, elapsed_s=0.1)

    monkeypatch.setattr(extract_mod, "extract_entities", fake_extract_entities)

    # Materialize 3 papers in a fresh KG.
    db_path = tmp_path / "kg.db"
    with SculptorKG(db_path) as kg:
        for i, aid in enumerate(("1111.11111", "2222.22222", "3333.33333")):
            kg.add_node(Paper(
                id=make_paper_id(aid),
                arxiv_id=aid,
                title=f"Paper {i}",
                authors=[],
                year=2020 + i,
                extracted=False,
            ))

    events: list[tuple[int, int, str]] = []

    def progress_cb(done: int, total: int, title: str) -> None:
        events.append((done, total, title))

    with SculptorKG(db_path) as kg:
        results = extract_all(store=kg, progress_cb=progress_cb)

    assert len(results) == 3
    # Three progress events, 1-indexed against total=3, titles match.
    assert events == [(1, 3, "Paper 0"), (2, 3, "Paper 1"), (3, 3, "Paper 2")]


def test_extract_all_swallows_progress_cb_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken progress callback must NOT abort extraction."""
    from sculptor.kg.extract import extract_all
    from sculptor.kg.schema import Paper, make_paper_id
    from sculptor.kg.store import SculptorKG

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    import anthropic
    import sculptor.kg.extract as extract_mod

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    def fake_extract_entities(paper, *, store, client, force):
        from sculptor.kg.extract import ExtractionResult

        paper.extracted = True
        store.add_node(paper)
        return ExtractionResult(paper_id=paper.id, elapsed_s=0.1)

    monkeypatch.setattr(extract_mod, "extract_entities", fake_extract_entities)

    db_path = tmp_path / "kg.db"
    with SculptorKG(db_path) as kg:
        kg.add_node(Paper(
            id=make_paper_id("1234.56789"),
            arxiv_id="1234.56789",
            title="T",
            authors=[],
            year=2020,
            extracted=False,
        ))

    def broken_cb(done, total, title):
        raise RuntimeError("callback explodes")

    with SculptorKG(db_path) as kg:
        results = extract_all(store=kg, progress_cb=broken_cb)
    assert len(results) == 1
    assert results[0].error is None
