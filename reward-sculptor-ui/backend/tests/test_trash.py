"""Tests for chunk A1 — non-destructive delete / trash.

Covers:
  - DELETE /projects/{slug} moves the project into .trash/ (entry
    appears with meta, absent from GET /projects) instead of
    hard-deleting it.
  - GET /trash lists the entry.
  - POST /trash/{entry_id}/restore moves it back; the project is
    listable again via GET /projects.
  - DELETE /trash/{entry_id} requires ?confirm=<entry_id> and is the
    only permanent-delete path (verified via absence after purge).
  - DELETE /projects/{slug} 409s when the project has an active job
    (job_manager.active_jobs_for_project guard), and the project dir
    is untouched.
  - Mission delete also lands in trash (mirrors project delete).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


# ── helpers ────────────────────────────────────────────────────────────
def _make_project(client: TestClient, name: str = "Trash Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _seed_mission_on_disk(project_dir: Path, mission_slug: str) -> Path:
    """Same minimal mission.json fixture as test_missions.py's helper —
    duplicated locally to keep this file self-contained."""
    md = project_dir / ".missions" / mission_slug
    md.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "goal": "test mission",
        "decomposition_model": "claude-opus-4-7",
        "decomposition_rationale": "test",
        "created_at": "2026-04-24T00:00:00+00:00",
        "current_stage_idx": 0,
        "stages": [{
            "name": "stage_0",
            "goal_text": "step 0",
            "success_criterion": "metric > 0.5",
            "max_iterations": 2,
            "parent_stage": None,
            "reward_seed_prompt": "seed",
            "kg_seed_papers": [],
            "status": "pending",
            "final_policy_path": None,
            "final_reward_path": None,
            "best_metric": None,
            "iterations_used": 0,
            "started_at": None,
            "finished_at": None,
            "redecomposition_attempts": 0,
        }],
    }
    (md / "mission.json").write_text(json.dumps(payload))
    return md


def _trash_root(tmp_projects_root: Path) -> Path:
    return tmp_projects_root.parent / ".trash"


# ── project delete -> trash ──────────────────────────────────────────
def test_delete_project_lands_in_trash(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    assert project_dir.is_dir()

    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 204
    assert not project_dir.exists()

    # Absent from the live project listing.
    r = client.get("/projects")
    assert r.status_code == 200
    assert slug not in {p["slug"] for p in r.json()}

    # Present in GET /trash with a readable meta.
    r = client.get("/trash")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["kind"] == "project"
    assert entry["slug"] == slug
    assert entry["origin_path"] == str(project_dir)
    assert entry["unreadable"] is False
    assert entry["size_bytes"] is not None and entry["size_bytes"] > 0

    # On-disk shape: entry dir contains trash_meta.json + the moved
    # tree with its own metadata.json intact.
    on_disk = [d for d in _trash_root(tmp_projects_root).iterdir() if d.is_dir()]
    assert len(on_disk) == 1
    assert (on_disk[0] / "trash_meta.json").is_file()
    assert (on_disk[0] / slug / "metadata.json").is_file()


# ── restore ───────────────────────────────────────────────────────────
def test_restore_project_makes_it_listable_again(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug

    client.delete(f"/projects/{slug}")
    assert not project_dir.exists()

    entry_id = client.get("/trash").json()[0]["entry_id"]
    r = client.post(f"/trash/{entry_id}/restore")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entry_id"] == entry_id
    assert body["restored_path"] == str(project_dir)

    # Back on disk at the original location.
    assert project_dir.is_dir()
    assert (project_dir / "metadata.json").is_file()

    # Listable again via the normal project endpoints.
    r = client.get(f"/projects/{slug}")
    assert r.status_code == 200
    r = client.get("/projects")
    assert slug in {p["slug"] for p in r.json()}

    # Trash entry is consumed.
    assert client.get("/trash").json() == []


def test_restore_unknown_entry_404s(client: TestClient) -> None:
    r = client.post("/trash/does-not-exist--20260101T000000Z/restore")
    assert r.status_code == 404


def test_restore_malformed_entry_id_400s(client: TestClient) -> None:
    r = client.post("/trash/..%2F..%2Fetc/restore")
    # Either FastAPI's own routing normalizes this away from our
    # handler, or our handler rejects it — either way it must never
    # be treated as a valid entry_id (200/500 would both be bugs).
    assert r.status_code in (400, 404)


# ── purge ─────────────────────────────────────────────────────────────
def test_purge_requires_confirm_echo(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    client.delete(f"/projects/{slug}")
    entry_id = client.get("/trash").json()[0]["entry_id"]

    # No confirm at all.
    r = client.delete(f"/trash/{entry_id}")
    assert r.status_code == 400

    # Wrong confirm value.
    r = client.delete(f"/trash/{entry_id}?confirm=nope")
    assert r.status_code == 400

    # Entry must still be there — confirm mismatch must not purge.
    assert len(client.get("/trash").json()) == 1


def test_purge_with_confirm_permanently_removes_entry(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    client.delete(f"/projects/{slug}")
    entry_id = client.get("/trash").json()[0]["entry_id"]
    entry_dir = _trash_root(tmp_projects_root) / entry_id
    assert entry_dir.is_dir()

    r = client.delete(f"/trash/{entry_id}?confirm={entry_id}")
    assert r.status_code == 204

    assert not entry_dir.exists()
    assert client.get("/trash").json() == []

    # Purged entries are gone for good — restore now 404s.
    r = client.post(f"/trash/{entry_id}/restore")
    assert r.status_code == 404


def test_purge_unknown_entry_404s(client: TestClient) -> None:
    entry_id = "no-such-slug--20260101T000000Z"
    r = client.delete(f"/trash/{entry_id}?confirm={entry_id}")
    assert r.status_code == 404


# ── 409 guard: active jobs block delete ──────────────────────────────
def test_delete_project_409_when_active_job(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job

    jm._jobs["fake_active"] = Job(
        job_id="fake_active", kind="sculpt_run",
        project_slug=slug, status="running",
    )

    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 409
    body = r.json()
    assert body["type"] == "/problems/state-conflict"

    # Untouched — still on disk, not in trash.
    assert project_dir.is_dir()
    assert client.get("/trash").json() == []

    # Clearing the job unblocks delete.
    jm._jobs["fake_active"].status = "completed"
    r = client.delete(f"/projects/{slug}")
    assert r.status_code == 204


# ── mission delete -> trash ──────────────────────────────────────────
def test_delete_mission_lands_in_trash(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")

    r = client.delete(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 200
    assert not md.exists()

    r = client.get("/trash")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 1
    assert entries[0]["kind"] == "mission"
    assert entries[0]["slug"] == "alpha"
    assert entries[0]["origin_path"] == str(md)

    on_disk = [d for d in _trash_root(tmp_projects_root).iterdir() if d.is_dir()]
    assert (on_disk[0] / "alpha" / "mission.json").is_file()
