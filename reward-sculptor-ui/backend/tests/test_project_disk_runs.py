"""Tests for mission-persistence increment 3 (project-level disk rows).

Covers:
  - GET /runs synthesizes ONE "disk:project" sculpt_run row when
    <project_dir>/runs/iter_* exist and no resident sculpt_run job
    covers them; dedups against resident sculpt_run jobs (live AND
    terminal); absent when there are no iter dirs.
  - status derives from the LAST iteration's disk state (finished
    marker → completed; bare dir → errored/"interrupted").
  - GET /projects/{slug}/iterations disk-truth list (metric fallback,
    dict-shaped metric_history.json, rollout/checkpoint flags).
  - GET /projects/{slug}/iterations/{i}/rollout 200 + 404 (missing and
    truncated file).
  - /runs/{disk:project} 404s cleanly on the _find_run-gated routes.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


# ── helpers ─────────────────────────────────────────────────────────
def _make_project(client: TestClient, name: str = "Disk Runs Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _seed_project_iter(
    project_dir: Path,
    i: int,
    *,
    with_metrics: bool = True,
    with_checkpoint: bool = True,
    with_rollout: bool = False,
    mean_return: float | None = None,
    fitness: float | None = None,
    reward_version: str | None = None,
    fitness_contradiction: dict | None = None,
) -> Path:
    iter_dir = project_dir / "runs" / f"iter_{i}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    if with_metrics:
        payload: dict = {"metrics": {}}
        if mean_return is not None:
            payload["metrics"]["mean_return"] = mean_return
        (iter_dir / "metrics.json").write_text(json.dumps(payload))
    if with_checkpoint:
        (iter_dir / "checkpoint.pt").write_bytes(b"x" * 16)
    if with_rollout:
        rd = iter_dir / "rollout"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "rollout.mp4").write_bytes(b"v" * 4096)
    if fitness is not None:
        rd = iter_dir / "rollout"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "behavior.json").write_text(json.dumps({"fitness": fitness}))
    if reward_version is not None:
        (iter_dir / "reward_spec.json").write_text(
            json.dumps({"version": reward_version})
        )
    if fitness_contradiction is not None:
        # §D24 (F4): the durable flag `_maybe_emit_fitness_contradiction`
        # writes next to an iter's other artifacts.
        (iter_dir / "fitness_contradiction.json").write_text(
            json.dumps(fitness_contradiction)
        )
    return iter_dir


# ── GET /runs project-level disk row ──────────────────────────────────
def test_list_runs_synthesizes_project_disk_row(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)
    _seed_project_iter(project_dir, 1)

    r = client.get(f"/projects/{slug}/runs")
    assert r.status_code == 200
    rows = [row for row in r.json() if row["run_id"] == "disk:project"]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "sculpt_run"
    assert row["status"] == "completed"
    assert row["error"] is None
    assert row["iterations_completed"] == 2
    assert row["iterations_requested"] == 0
    assert row["mission_slug"] is None
    assert row["stage_name"] is None


def test_project_disk_row_interrupted_last_iter(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A bare last iter dir (no metrics.json, no checkpoint) means the
    process died mid-iteration → errored/interrupted."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)
    _seed_project_iter(project_dir, 1, with_metrics=False, with_checkpoint=False)

    r = client.get(f"/projects/{slug}/runs")
    rows = [row for row in r.json() if row["run_id"] == "disk:project"]
    assert len(rows) == 1
    assert rows[0]["status"] == "errored"
    assert rows[0]["error"] == "interrupted"
    assert rows[0]["iterations_completed"] == 1


def test_project_disk_row_metric_history_from_reports(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """The project-level reports/metric_history.json is a DICT with a
    "history" key (unlike the stage-level bare list) — the sparkline
    history must come from it."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)
    _seed_project_iter(project_dir, 1)
    reports = project_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metric_history.json").write_text(
        json.dumps({"primary_metric": "mean_return", "history": [1.5, None, 3.0]})
    )

    r = client.get(f"/projects/{slug}/runs")
    rows = [row for row in r.json() if row["run_id"] == "disk:project"]
    assert rows[0]["primary_metric_history"] == [1.5, None, 3.0]


def test_project_disk_row_dedups_against_resident_sculpt_run(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """ANY resident sculpt_run job — live or terminal — suppresses the
    synthetic row (iter dirs are one shared tree; the resident job's
    row already covers them)."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    job = jobs.register_passive_job(
        kind="sculpt_run", project_slug=slug, params={},
    )

    r = client.get(f"/projects/{slug}/runs")
    rows = r.json()
    assert [row for row in rows if row["run_id"] == "disk:project"] == []

    # Terminal resident job: still deduped.
    job.status = "completed"
    r = client.get(f"/projects/{slug}/runs")
    rows = r.json()
    assert [row for row in rows if row["run_id"] == "disk:project"] == []
    assert [row for row in rows if row["run_id"] == job.job_id]


def test_project_disk_row_absent_without_iters(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/runs")
    assert r.status_code == 200
    assert [row for row in r.json() if row["run_id"] == "disk:project"] == []


def test_project_disk_row_not_suppressed_by_other_project_job(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """The dedup is per-project — a sculpt_run resident for a DIFFERENT
    project must not suppress this project's synthetic row."""
    slug = _make_project(client)
    other = _make_project(client, name="Other Project")
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)

    app = client.app  # type: ignore[attr-defined]
    app.state.job_manager.register_passive_job(
        kind="sculpt_run", project_slug=other, params={},
    )

    r = client.get(f"/projects/{slug}/runs")
    assert len([row for row in r.json() if row["run_id"] == "disk:project"]) == 1


# ── GET /projects/{slug}/iterations ───────────────────────────────────
def test_list_project_iterations(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(
        project_dir, 0,
        with_rollout=True, mean_return=12.5, fitness=0.8, reward_version="v3",
    )
    _seed_project_iter(project_dir, 1, with_checkpoint=False)

    r = client.get(f"/projects/{slug}/iterations")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert [row["iter_index"] for row in rows] == [0, 1]
    assert rows[0]["primary_metric"] == 12.5
    assert rows[0]["fitness"] == 0.8
    assert rows[0]["has_rollout"] is True
    assert rows[0]["has_checkpoint"] is True
    assert rows[0]["reward_version"] == "v3"
    assert rows[1]["has_rollout"] is False
    assert rows[1]["has_checkpoint"] is False
    # §D24 (F4): no flag file seeded for either row → default False/None.
    assert rows[0]["fitness_contradiction"] is False
    assert rows[0]["fitness_components"] is None


def test_list_project_iterations_reports_fitness_contradiction_flag(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """§D24 (F4): a run-dir fixture with `fitness_contradiction.json` ->
    payload flag true + components passthrough; absent -> false/None."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(
        project_dir, 0, with_rollout=True, fitness=0.0,
        fitness_contradiction={
            "type": "fitness_contradiction",
            "stage_name": None,
            "iter": 0,
            "fitness": 0.0,
            "criterion": "metric > 0.5",
            "components": {"gate_upright_frac": 1.0},
        },
    )
    _seed_project_iter(project_dir, 1, with_rollout=True, fitness=0.8)

    r = client.get(f"/projects/{slug}/iterations")
    assert r.status_code == 200, r.text
    rows = r.json()
    row0 = next(row for row in rows if row["iter_index"] == 0)
    assert row0["fitness_contradiction"] is True
    assert row0["fitness_components"] == {"gate_upright_frac": 1.0}

    row1 = next(row for row in rows if row["iter_index"] == 1)
    assert row1["fitness_contradiction"] is False
    assert row1["fitness_components"] is None


def test_list_project_iterations_prefers_metric_history(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0, mean_return=99.0)
    reports = project_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metric_history.json").write_text(
        json.dumps({"primary_metric": "mean_return", "history": [7.25]})
    )

    r = client.get(f"/projects/{slug}/iterations")
    assert r.json()[0]["primary_metric"] == 7.25


def test_list_project_iterations_empty_and_unknown_project(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/iterations")
    assert r.status_code == 200
    assert r.json() == []

    r = client.get("/projects/no-such-project/iterations")
    assert r.status_code == 404


# ── GET /projects/{slug}/iterations/{i}/rollout ───────────────────────
def test_project_iter_rollout_200(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0, with_rollout=True)

    r = client.get(f"/projects/{slug}/iterations/0/rollout")
    assert r.status_code == 200
    assert r.content == b"v" * 4096


def test_project_iter_rollout_404_missing_and_truncated(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)  # no rollout at all
    it1 = _seed_project_iter(project_dir, 1)
    rd = it1 / "rollout"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "rollout.mp4").write_bytes(b"v" * 100)  # < 2048 → still rendering

    for i in (0, 1, 99):
        r = client.get(f"/projects/{slug}/iterations/{i}/rollout")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/problem+json")


# ── /runs/{disk:project} 404s cleanly on job-gated routes ─────────────
def test_disk_project_run_id_404s_on_job_gated_routes(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _seed_project_iter(project_dir, 0)

    run_id = "disk:project"
    assert client.get(f"/projects/{slug}/runs/{run_id}").status_code == 404
    assert client.delete(f"/projects/{slug}/runs/{run_id}").status_code == 404
    assert client.get(
        f"/projects/{slug}/runs/{run_id}/iterations/0/rollout"
    ).status_code == 404
    assert client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"mode": "auto"},
    ).status_code == 404
