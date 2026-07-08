"""Tests for chunk A4 — saved-missions library backend.

Covers:
  - GET /saved lists archived entries, newest first, in the slim
    per-stage-status projection.
  - GET /saved/{entry_id} returns the full manifest + mission.json.
  - GET /saved/{entry_id}/file/{relpath} serves an archived file
    (mp4 media type) and 404s on a `..` traversal attempt.
  - DELETE /saved/{entry_id} moves the entry into /trash (kind="saved").
  - POST /projects/{slug}/missions/{mission_slug}/save schedules a
    `mission_save` job that lands a NEW entry under the saved root.

`RS_SAVED_ROOT` / `RS_TRASH_ROOT` are monkeypatched to tmp dirs so
these tests never touch the real `~/.local/share/reward-sculptor/`
tree.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def saved_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "saved"
    monkeypatch.setenv("RS_SAVED_ROOT", str(root))
    return root


@pytest.fixture
def trash_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "trash"
    monkeypatch.setenv("RS_TRASH_ROOT", str(root))
    return root


# ── helpers ────────────────────────────────────────────────────────────
def _make_project(client: TestClient, name: str = "Saved Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _seed_mission_on_disk(project_dir: Path, mission_slug: str) -> Path:
    """Minimal mission.json — same shape test_trash.py's helper uses.
    No stages/ dir needed: archive_mission tolerates a mission with no
    stages/ subdir on disk (_find_stage_dirs returns [] for a missing
    dir)."""
    md = project_dir / ".missions" / mission_slug
    md.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "goal": "jump over the box",
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
            "status": "succeeded",
            "final_policy_path": None,
            "final_reward_path": None,
            "best_metric": 0.9,
            "iterations_used": 1,
            "started_at": None,
            "finished_at": None,
            "redecomposition_attempts": 0,
        }],
    }
    (md / "mission.json").write_text(json.dumps(payload))
    return md


def _archive_fake_mission(
    saved_root_dir: Path, mission_dir: Path, *,
    project_slug: str = "proj", mission_slug: str = "alpha",
) -> str:
    """Archive a real mission tree via sculptor.archive.archive_mission
    (the same function the mission_save job calls), including a rollout
    video so the file-serving test has something real to fetch. Returns
    the entry_id."""
    from sculptor.archive import archive_mission

    # Give the mission one stage/iteration with a rollout video so
    # GET .../file/... has a real artifact to serve.
    stage_dir = mission_dir / "stages" / "stage_0"
    iter_dir = stage_dir / "runs" / "iter_0" / "rollout"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "rollout.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    result = archive_mission(
        mission_dir, saved_root_dir,
        project_slug=project_slug, incremental=False,
    )
    return result.entry_dir.name


# ── GET /saved (list) ────────────────────────────────────────────────
def test_list_saved_returns_slim_rows(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")

    r = client.get("/saved")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["entry_id"] == entry_id
    assert row["project_slug"] == slug
    assert row["mission_slug"] == "alpha"
    assert row["goal"] == "jump over the box"
    assert row["total_bytes"] is not None and row["total_bytes"] > 0
    assert isinstance(row["stages"], list) and len(row["stages"]) == 1
    assert row["stages"][0]["name"] == "stage_0"
    assert row["stages"][0]["status"] == "succeeded"
    # Thumbnail hint: first video relpath across stages.
    assert row["thumbnail_video"] is not None
    assert row["thumbnail_video"].endswith("rollout.mp4")


def test_list_saved_empty_when_nothing_archived(
    client: TestClient, saved_root: Path,
) -> None:
    r = client.get("/saved")
    assert r.status_code == 200
    assert r.json() == []


def test_list_saved_newest_first(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md_a = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    md_b = _seed_mission_on_disk(tmp_projects_root / slug, "beta")
    entry_a = _archive_fake_mission(
        saved_root, md_a, project_slug=slug, mission_slug="alpha")
    entry_b = _archive_fake_mission(
        saved_root, md_b, project_slug=slug, mission_slug="beta")

    # Force distinguishable created_at ordering regardless of clock
    # resolution — rewrite entry_a's manifest as strictly older.
    manifest_a_path = saved_root / entry_a / "manifest.json"
    manifest_a = json.loads(manifest_a_path.read_text())
    manifest_a["created_at"] = "2020-01-01T00:00:00+00:00"
    manifest_a_path.write_text(json.dumps(manifest_a))

    r = client.get("/saved")
    assert r.status_code == 200
    ids = [row["entry_id"] for row in r.json()]
    assert ids == [entry_b, entry_a]


# ── GET /saved/{entry_id} (detail) ───────────────────────────────────
def test_get_saved_entry_returns_manifest_and_mission_json(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")

    r = client.get(f"/saved/{entry_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entry_id"] == entry_id
    assert body["mission_slug"] == "alpha"
    assert body["schema"] == 1
    assert isinstance(body["stages"], list) and len(body["stages"]) == 1
    # Full manifest carries the un-slimmed kept_checkpoints list.
    assert "kept_checkpoints" in body["stages"][0]
    # mission.json was archived verbatim as a mission-root file.
    assert body["mission_json"] is not None
    assert body["mission_json"]["goal"] == "jump over the box"


def test_get_saved_entry_unknown_id_404s(client: TestClient, saved_root: Path) -> None:
    r = client.get("/saved/does-not-exist--alpha--20260101T000000Z")
    assert r.status_code == 404


def test_get_saved_entry_malformed_id_404s(client: TestClient, saved_root: Path) -> None:
    r = client.get("/saved/..%2F..%2Fetc")
    # Either FastAPI's routing normalizes this away, or our handler
    # rejects it via the anchored regex — either way, never a 200/500.
    assert r.status_code in (404, 400)


# ── GET /saved/{entry_id}/file/{relpath} ─────────────────────────────
def test_get_saved_file_serves_video(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")

    relpath = "stages/stage_0/runs/iter_0/rollout/rollout.mp4"
    r = client.get(f"/saved/{entry_id}/file/{relpath}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert r.content.startswith(b"\x00\x00\x00\x18ftypmp42")


def test_get_saved_file_traversal_blocked(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")

    r = client.get(f"/saved/{entry_id}/file/../../../etc/passwd")
    assert r.status_code == 404

    # A `..`-shaped relpath that still resolves inside the entry dir
    # boundary check (double-encoded / nested) must also 404, not 200.
    r = client.get(
        f"/saved/{entry_id}/file/stages/../../../../../../etc/passwd")
    assert r.status_code == 404


def test_get_saved_file_missing_file_404s(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")

    r = client.get(f"/saved/{entry_id}/file/does/not/exist.json")
    assert r.status_code == 404


# ── DELETE /saved/{entry_id} ─────────────────────────────────────────
def test_delete_saved_entry_moves_to_trash(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
    trash_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    entry_id = _archive_fake_mission(
        saved_root, md, project_slug=slug, mission_slug="alpha")
    entry_dir = saved_root / entry_id
    assert entry_dir.is_dir()

    r = client.delete(f"/saved/{entry_id}")
    assert r.status_code == 204
    assert not entry_dir.exists()

    # Absent from the saved listing.
    assert client.get("/saved").json() == []

    # Present in /trash with kind="saved".
    r = client.get("/trash")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 1
    assert entries[0]["kind"] == "saved"
    assert entries[0]["slug"] == "alpha"
    assert entries[0]["origin_path"] == str(entry_dir)


def test_delete_saved_entry_unknown_id_404s(
    client: TestClient, saved_root: Path,
) -> None:
    r = client.delete("/saved/nope--alpha--20260101T000000Z")
    assert r.status_code == 404


# ── POST /projects/{slug}/missions/{mission_slug}/save ───────────────
def test_save_mission_schedules_job(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    """The route returns a 202 + JobDetail for a `mission_save` job.
    The route layer's own event loop plumbing (JobManager.submit +
    the app's bound loop) is exercised end-to-end here; the runner's
    OWN archiving behavior is covered directly below (mirrors how
    test_missions.py / test_runs.py test decompose/execute runners:
    via asyncio.run(runner(job, event)), not by polling the live job
    through TestClient, which has no running loop to drive it)."""
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")

    r = client.post(f"/projects/{slug}/missions/alpha/save")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "mission_save"
    assert body["project_slug"] == slug
    assert body["params"]["mission_slug"] == "alpha"
    assert body["status"] in ("queued", "running", "completed")


def test_save_mission_unknown_project_404s(
    client: TestClient, saved_root: Path,
) -> None:
    r = client.post("/projects/no-such-project/missions/alpha/save")
    assert r.status_code == 404


def test_save_mission_unknown_mission_404s(
    client: TestClient, tmp_projects_root: Path, saved_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.post(f"/projects/{slug}/missions/does-not-exist/save")
    assert r.status_code == 404


# ── run_mission_save_job runner (direct, per test_missions.py idiom) ──
def test_run_mission_save_job_archives_and_reports_entry(
    tmp_projects_root: Path, saved_root: Path,
) -> None:
    from backend.services.job_manager import Job
    from backend.services.saved_jobs import run_mission_save_job

    project_dir = tmp_projects_root / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    _seed_mission_on_disk(project_dir, "alpha")

    runner = run_mission_save_job(
        project_dir=project_dir, project_slug="proj",
        mission_slug="alpha", pinned=None,
    )
    job = Job(job_id="j1", kind="mission_save", project_slug="proj")
    result = asyncio.run(runner(job, asyncio.Event()))

    assert result["mission_slug"] == "alpha"
    assert result["project_slug"] == "proj"
    assert result["total_bytes"] > 0
    entry_dir = Path(result["entry_dir"])
    assert entry_dir.is_dir()
    assert (entry_dir / "manifest.json").is_file()
    assert result["entry_id"] == entry_dir.name

    # job.emit'd a mission_saved event with the same payload shape.
    saved_events = [e for e in job.events if e.get("type") == "mission_saved"]
    assert len(saved_events) == 1
    assert saved_events[0]["entry_id"] == result["entry_id"]

    # GET /saved reflects it (disk is the source of truth).
    from sculptor.archive import list_saved

    manifests = list_saved(saved_root)
    assert len(manifests) == 1
    assert manifests[0]["mission_slug"] == "alpha"


def test_run_mission_save_job_twice_creates_two_entries(
    tmp_projects_root: Path, saved_root: Path,
) -> None:
    """Manual save always mints a fresh entry (incremental=False) —
    unlike the auto-archive hooks, a second explicit save must not
    silently merge into the first entry.

    `sculptor.archive`'s entry-id stamp is second-resolution with no
    same-second collision bump (unlike `backend.services.trash`'s
    `_unique_entry_id`), so back-to-back calls in the same wall-clock
    second mint the SAME id and the second archive_mission call just
    merges into entry 1's directory (dirs_exist_ok=True copies).
    Monkeypatch the clock function `archive_mission` uses so the two
    calls land in different seconds — this test is about `save`
    requesting `incremental=False` on every call, not about the
    archive module's stamp granularity."""
    import sculptor.archive as archive_mod
    from datetime import datetime, timedelta, timezone

    from backend.services.job_manager import Job
    from backend.services.saved_jobs import run_mission_save_job

    project_dir = tmp_projects_root / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    _seed_mission_on_disk(project_dir, "alpha")

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    calls = {"n": 0}
    real_utc_stamp = archive_mod._utc_stamp

    def _fake_utc_stamp(now=None):
        calls["n"] += 1
        return real_utc_stamp(base + timedelta(seconds=calls["n"]))

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(archive_mod, "_utc_stamp", _fake_utc_stamp)
    try:
        def _run_once() -> dict:
            runner = run_mission_save_job(
                project_dir=project_dir, project_slug="proj",
                mission_slug="alpha", pinned=None,
            )
            job = Job(job_id="j", kind="mission_save", project_slug="proj")
            return asyncio.run(runner(job, asyncio.Event()))

        result1 = _run_once()
        result2 = _run_once()
    finally:
        mp.undo()

    assert result1["entry_id"] != result2["entry_id"]

    entries = [d for d in saved_root.iterdir() if d.is_dir()]
    assert len(entries) == 2


def test_run_mission_save_job_missing_mission_fails_loud(
    tmp_projects_root: Path, saved_root: Path,
) -> None:
    from backend.services.job_manager import Job
    from backend.services.saved_jobs import run_mission_save_job

    project_dir = tmp_projects_root / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)

    runner = run_mission_save_job(
        project_dir=project_dir, project_slug="proj",
        mission_slug="does-not-exist", pinned=None,
    )
    job = Job(job_id="j1", kind="mission_save", project_slug="proj")
    with pytest.raises(FileNotFoundError):
        asyncio.run(runner(job, asyncio.Event()))
