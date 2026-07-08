"""Tests for mission-persistence increment 2 (FastAPI backend).

Covers:
  - disk-union restores an orphan stage dir before its __r1_0 sibling
    (the pre-increment-1 destructive-splice recovery path).
  - provenance.json enrichment fills failure_reason/failure_detail.
  - display-label assignment for a parent + two replan children.
  - lifecycle stays non-halted/non-errored with a superseded stage.
  - metric_status covers all four values.
  - GET /runs synthesizes disk rows once JobManager has no live job,
    and dedups against a live mission_stage_run job.
  - GET /runs/{disk:...} and friends 404 cleanly (never 500).
  - checkpoint endpoint 200 + 404.
  - metric regenerate endpoint 409 while the mission/stage is busy.
  - run guard 409 when stage_metric_required + missing metric, and the
    proceed_blind bypass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── helpers ─────────────────────────────────────────────────────────
def _make_project(client: TestClient, name: str = "Persist Test") -> str:
    r = client.post("/projects", json={"name": name, "adapter": "gym_sb3"})
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _stage_dict(
    name: str,
    *,
    status: str = "pending",
    parent_stage: str | None = None,
    steering_metric: str | None = None,
    failure_reason: str | None = None,
    failure_detail: str | None = None,
) -> dict:
    return {
        "name": name,
        "goal_text": f"goal for {name}",
        "success_criterion": "metric > 0.5",
        "max_iterations": 4,
        "parent_stage": parent_stage,
        "reward_seed_prompt": f"seed for {name}",
        "kg_seed_papers": [],
        "status": status,
        "final_policy_path": None,
        "final_reward_path": None,
        "best_metric": None,
        "iterations_used": 0,
        "started_at": None,
        "finished_at": None,
        "redecomposition_attempts": 0,
        "steering_metric": steering_metric,
        "failure_reason": failure_reason,
        "failure_detail": failure_detail,
    }


def _write_mission(
    project_dir: Path, mission_slug: str, stages: list[dict],
    *, run_defaults: dict | None = None,
) -> Path:
    md = project_dir / ".missions" / mission_slug
    md.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "goal": "test mission",
        "decomposition_model": "claude-opus-4-7",
        "decomposition_rationale": "test",
        "created_at": "2026-07-01T00:00:00+00:00",
        "current_stage_idx": 0,
        "stages": stages,
    }
    if run_defaults is not None:
        payload["run_defaults"] = run_defaults
    (md / "mission.json").write_text(json.dumps(payload))
    return md


def _seed_stage_dir(
    mission_dir_path: Path, stage_name: str, *, n_iters: int = 2,
    with_checkpoint: bool = True,
) -> Path:
    stage_dir = mission_dir_path / "stages" / stage_name
    for i in range(n_iters):
        iter_dir = stage_dir / "runs" / f"iter_{i}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        if with_checkpoint:
            (iter_dir / "checkpoint.pt").write_bytes(b"x" * 16)
    return stage_dir


def _write_provenance(mission_dir_path: Path, stage_records: list[dict]) -> None:
    doc = {
        "contexts": [],
        "created_at": "2026-07-01T00:00:00+00:00",
        "goal": "test mission",
        "schema": 1,
        "stages": stage_records,
    }
    (mission_dir_path / "provenance.json").write_text(json.dumps(doc))


def _write_stage_metric_meta(
    mission_dir_path: Path, stage_name: str, *, accepted: bool,
) -> None:
    d = mission_dir_path / "stage_metrics" / stage_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"accepted": accepted}))


# ── mission_store unit-level coverage (via GET /missions/{slug}) ─────
def test_disk_union_restores_orphan_before_replan_sibling(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A stage dir with no mission.stages entry (old destructive
    splice) is inserted immediately before its __r1_0 child."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a__r1_0", status="succeeded", parent_stage=None),
        _stage_dict("b", status="pending", parent_stage="a__r1_0"),
    ])
    # "a" itself has no mission.stages entry — orphaned — but its dir
    # is still on disk with trained iterations.
    _seed_stage_dir(md, "a", n_iters=3)

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.status_code == 200, r.text
    names = [s["name"] for s in r.json()["stages"]]
    assert names == ["a", "a__r1_0", "b"]
    orphan = r.json()["stages"][0]
    assert orphan["on_disk_only"] is True
    assert orphan["status"] == "superseded"
    assert orphan["iterations_used"] == 3


def test_disk_union_appends_when_no_sibling(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a", status="succeeded"),
    ])
    _seed_stage_dir(md, "orphan_no_sibling", n_iters=1)

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()["stages"]]
    assert names == ["a", "orphan_no_sibling"]


def test_disk_union_restores_orphan_before_truncated_replan_sibling(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A 32-char (near-cap) parent name gets TRUNCATED by
    `sculptor.decompose._truncate_for_name_budget` when redecompose
    embeds it in a child name: "crouch_load_absorb_and_transfer"
    (31 chars) -> children prefixed "crouch_load_absorb__r1_<i>" (Q =
    "crouch_load_absorb" is a truncation of the parent's real name,
    not an exact match). The orphan parent dir must still be inserted
    immediately before its truncated-prefix children, not appended at
    the end."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    parent_name = "crouch_load_absorb_and_transfer"
    truncated_q = "crouch_load_absorb"
    md = _write_mission(project_dir, "m1", [
        _stage_dict(f"{truncated_q}__r1_0", status="succeeded", parent_stage=None),
        _stage_dict("b", status="pending", parent_stage=f"{truncated_q}__r1_0"),
    ])
    # The full-length parent dir is orphaned (no mission.stages entry)
    # but still on disk with trained iterations.
    _seed_stage_dir(md, parent_name, n_iters=3)

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.status_code == 200, r.text
    names = [s["name"] for s in r.json()["stages"]]
    assert names == [parent_name, f"{truncated_q}__r1_0", "b"]
    orphan = r.json()["stages"][0]
    assert orphan["on_disk_only"] is True
    assert orphan["status"] == "superseded"
    assert orphan["iterations_used"] == 3


def test_provenance_enrichment_fills_failure_reason(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        # No failure_reason persisted on the stage itself (pre-
        # increment-1 mission.json shape).
        _stage_dict("a", status="failed"),
    ])
    _write_provenance(md, [
        {
            "stage_name": "a",
            "status": "failed",
            "failure_reason": "criterion_not_met",
            "failure_detail": "metric never crossed threshold",
            "recorded_at": "2026-07-01T00:00:00+00:00",
        },
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.status_code == 200
    stage = r.json()["stages"][0]
    assert stage["failure_reason"] == "criterion_not_met"
    assert stage["failure_detail"] == "metric never crossed threshold"


def test_provenance_enrichment_does_not_override_persisted_value(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict(
            "a", status="failed",
            failure_reason="no_checkpoint", failure_detail="already set",
        ),
    ])
    _write_provenance(md, [
        {
            "stage_name": "a",
            "status": "failed",
            "failure_reason": "criterion_not_met",
            "failure_detail": "should not be used",
        },
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    stage = r.json()["stages"][0]
    assert stage["failure_reason"] == "no_checkpoint"
    assert stage["failure_detail"] == "already set"


def test_display_labels_parent_and_replan_children(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [
        _stage_dict("a", status="failed"),
        _stage_dict("a__r1_0", status="succeeded", parent_stage="a"),
        _stage_dict("a__r1_1", status="pending", parent_stage="a__r1_0"),
        _stage_dict("b", status="pending", parent_stage="a__r1_0"),
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    labels = {s["name"]: s["display_label"] for s in r.json()["stages"]}
    assert labels == {
        "a": "1",
        "a__r1_0": "1.1",
        "a__r1_1": "1.2",
        "b": "2",
    }


def test_display_labels_truncated_parent_name_nests_children(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Same truncation scenario as the union-ordering test, but for
    `_assign_display_labels`: children whose extracted parent token Q
    is a TRUNCATION of the real (32-char-near-cap) parent name must
    still nest under that parent ("1.1", "1.2"), not get relabeled as
    fresh top-level stages."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    parent_name = "crouch_load_absorb_and_transfer"
    truncated_q = "crouch_load_absorb"
    _write_mission(project_dir, "m1", [
        _stage_dict(parent_name, status="failed"),
        _stage_dict(f"{truncated_q}__r1_0", status="succeeded", parent_stage=parent_name),
        _stage_dict(f"{truncated_q}__r1_1", status="pending", parent_stage=f"{truncated_q}__r1_0"),
        _stage_dict("b", status="pending", parent_stage=f"{truncated_q}__r1_0"),
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    labels = {s["name"]: s["display_label"] for s in r.json()["stages"]}
    assert labels == {
        parent_name: "1",
        f"{truncated_q}__r1_0": "1.1",
        f"{truncated_q}__r1_1": "1.2",
        "b": "2",
    }


def test_display_labels_grandchild_resolves_replan_child_parent(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A replan child's OWN name must be registered in label_by_name
    as it's visited, so a grandchild (`f"{child}__r1_0"`, i.e. a
    replan of a replan) resolves its immediate parent recursively
    ("1.1.1"), not just direct children of a top-level stage."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [
        _stage_dict("a", status="failed"),
        _stage_dict("a__r1_0", status="failed", parent_stage="a"),
        _stage_dict("a__r1_0__r1_0", status="pending", parent_stage="a__r1_0"),
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    labels = {s["name"]: s["display_label"] for s in r.json()["stages"]}
    assert labels == {
        "a": "1",
        "a__r1_0": "1.1",
        "a__r1_0__r1_0": "1.1.1",
    }


def test_lifecycle_with_superseded_stage_stays_ready_not_halted(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A superseded stage must not trip 'halted' / block 'ready'."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [
        _stage_dict("a", status="superseded"),
        _stage_dict("a__r1_0", status="succeeded", parent_stage="a"),
        _stage_dict("b", status="pending", parent_stage="a__r1_0"),
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.status_code == 200
    assert r.json()["lifecycle"] == "ready"


def test_lifecycle_completed_ignores_superseded_stage(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [
        _stage_dict("a", status="superseded"),
        _stage_dict("a__r1_0", status="succeeded", parent_stage="a"),
    ])

    r = client.get(f"/projects/{slug}/missions/m1")
    assert r.json()["lifecycle"] == "completed"


def test_metric_status_all_four_values(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("accepted_stage", status="pending"),
        _stage_dict("rejected_stage", status="pending"),
        _stage_dict("inherited_stage", status="pending", steering_metric="g1_kick"),
        _stage_dict("none_stage", status="pending"),
    ])
    _write_stage_metric_meta(md, "accepted_stage", accepted=True)
    _write_stage_metric_meta(md, "rejected_stage", accepted=False)

    r = client.get(f"/projects/{slug}/missions/m1")
    status_by_name = {s["name"]: s["metric_status"] for s in r.json()["stages"]}
    assert status_by_name == {
        "accepted_stage": "accepted",
        "rejected_stage": "rejected",
        "inherited_stage": "inherited",
        "none_stage": "none",
    }


# ── GET /runs disk-reconstruction ─────────────────────────────────────
def test_list_runs_synthesizes_disk_row_after_restart(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """No live JobManager entry for the stage (simulated backend
    restart) — /runs still returns a synthetic row from disk."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a", status="succeeded"),
    ])
    _seed_stage_dir(md, "a", n_iters=2)

    r = client.get(f"/projects/{slug}/runs")
    assert r.status_code == 200
    rows = r.json()
    disk_rows = [row for row in rows if row["run_id"].startswith("disk:")]
    assert len(disk_rows) == 1
    row = disk_rows[0]
    assert row["run_id"] == "disk:m1/a"
    assert row["status"] == "completed"
    assert row["mission_slug"] == "m1"
    assert row["stage_name"] == "a"
    assert row["iterations_completed"] == 2
    assert row["kind"] == "mission_stage_run"


def test_list_runs_dedups_against_live_job(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage with a LIVE mission_stage_run job must NOT also get a
    synthetic disk row."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a", status="training"),
    ])
    _seed_stage_dir(md, "a", n_iters=1)

    from backend.main import create_app
    from backend.config import Settings

    # Reuse the same app instance the `client` fixture built so we can
    # reach its job_manager directly.
    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_run",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )

    r = client.get(f"/projects/{slug}/runs")
    assert r.status_code == 200
    rows = r.json()
    disk_rows = [row for row in rows if row["run_id"].startswith("disk:")]
    assert disk_rows == []
    live_rows = [row for row in rows if row.get("stage_name") == "a"]
    assert len(live_rows) == 1
    assert not live_rows[0]["run_id"].startswith("disk:")


def test_list_runs_dedups_against_terminal_resident_job(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A mission_stage_run job that already finished (completed /
    errored / stopped) but is still resident in JobManager (backend
    hasn't restarted since the run) must ALSO be deduped against — the
    real job's row already covers the stage; a terminal status is not
    an excuse to double it with a synthetic disk: row too."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a", status="succeeded"),
    ])
    _seed_stage_dir(md, "a", n_iters=1)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    job = jobs.register_passive_job(
        kind="mission_stage_run",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )
    job.status = "completed"

    r = client.get(f"/projects/{slug}/runs")
    assert r.status_code == 200
    rows = r.json()
    disk_rows = [row for row in rows if row["run_id"].startswith("disk:")]
    assert disk_rows == []
    live_rows = [row for row in rows if row.get("stage_name") == "a"]
    assert len(live_rows) == 1
    assert live_rows[0]["run_id"] == job.job_id


def test_list_runs_interrupted_training_stage(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A stage stuck at status='training' with NO live job (crashed
    backend) reports errored/interrupted, not still running."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [
        _stage_dict("a", status="training"),
    ])
    _seed_stage_dir(md, "a", n_iters=1)

    r = client.get(f"/projects/{slug}/runs")
    rows = [row for row in r.json() if row["run_id"] == "disk:m1/a"]
    assert len(rows) == 1
    assert rows[0]["status"] == "errored"
    assert rows[0]["error"] == "interrupted"


# ── /runs/{disk:...} routes 404 cleanly ────────────────────────────────
@pytest.mark.parametrize("method,path_suffix", [
    ("GET", ""),
    ("DELETE", ""),
])
def test_disk_run_id_404s_on_run_detail_and_kill(
    client: TestClient, tmp_projects_root: Path, method: str, path_suffix: str,
) -> None:
    slug = _make_project(client)
    run_id = "disk:some-mission/some-stage"
    r = client.request(method, f"/projects/{slug}/runs/{run_id}{path_suffix}")
    assert r.status_code == 404


def test_disk_run_id_404s_on_control(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    run_id = "disk:some-mission/some-stage"
    r = client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"mode": "auto"},
    )
    assert r.status_code == 404


def test_disk_run_id_404s_on_rollout(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    run_id = "disk:some-mission/some-stage"
    r = client.get(
        f"/projects/{slug}/runs/{run_id}/iterations/0/rollout",
    )
    assert r.status_code == 404


# ── checkpoint download endpoint ───────────────────────────────────────
def test_checkpoint_endpoint_200(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [_stage_dict("a", status="succeeded")])
    _seed_stage_dir(md, "a", n_iters=1)

    r = client.get(
        f"/projects/{slug}/missions/m1/stages/a/iterations/0/checkpoint",
    )
    assert r.status_code == 200
    assert r.content == b"x" * 16


def test_checkpoint_endpoint_404_missing_checkpoint(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    md = _write_mission(project_dir, "m1", [_stage_dict("a", status="succeeded")])
    _seed_stage_dir(md, "a", n_iters=1, with_checkpoint=False)

    r = client.get(
        f"/projects/{slug}/missions/m1/stages/a/iterations/0/checkpoint",
    )
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_checkpoint_endpoint_404_unknown_stage(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="pending")])

    r = client.get(
        f"/projects/{slug}/missions/m1/stages/no-such-stage/iterations/0/checkpoint",
    )
    assert r.status_code == 404


# ── metric regenerate endpoint ─────────────────────────────────────────
def test_regenerate_metric_409_when_mission_has_active_job(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="pending")])
    (project_dir / ".missions" / "m1" / "stages" / "a").mkdir(parents=True, exist_ok=True)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_execute",
        project_slug=slug,
        params={"mission_slug": "m1"},
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/metric/regenerate",
    )
    assert r.status_code == 409


def test_regenerate_metric_409_when_stage_training(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="training")])
    (project_dir / ".missions" / "m1" / "stages" / "a").mkdir(parents=True, exist_ok=True)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_run",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/metric/regenerate",
    )
    assert r.status_code == 409


def test_regenerate_metric_404_unknown_stage(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="pending")])

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/no-such-stage/metric/regenerate",
    )
    assert r.status_code == 404


def test_regenerate_metric_409_when_another_regen_live_same_mission(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Two concurrent regenerate requests for the SAME mission (even
    different stages) must not both proceed — both end in a
    non-atomic `save_mission` write to the same mission.json, so the
    second one silently clobbers the first's result. Regression test
    for the fix: `active_mission_job` didn't recognize
    `mission_stage_metric_regen` at all, so this was previously
    completely unguarded."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [
        _stage_dict("a", status="pending"),
        _stage_dict("b", status="pending"),
    ])
    (project_dir / ".missions" / "m1" / "stages" / "a").mkdir(parents=True, exist_ok=True)
    (project_dir / ".missions" / "m1" / "stages" / "b").mkdir(parents=True, exist_ok=True)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_metric_regen",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )

    # A regenerate for a DIFFERENT stage of the SAME mission must also
    # be blocked — the hazard is the shared mission.json, not the stage.
    r = client.post(
        f"/projects/{slug}/missions/m1/stages/b/metric/regenerate",
    )
    assert r.status_code == 409


def test_regenerate_metric_allowed_when_regen_live_different_mission(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live regen job for a DIFFERENT mission must not block this
    one — the guard is mission-scoped, not project-wide."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="pending")])
    _write_mission(project_dir, "m2", [_stage_dict("a", status="pending")])
    (project_dir / ".missions" / "m1" / "stages" / "a").mkdir(parents=True, exist_ok=True)
    (project_dir / ".missions" / "m2" / "stages" / "a").mkdir(parents=True, exist_ok=True)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_metric_regen",
        project_slug=slug,
        params={"mission_slug": "m2", "stage_name": "a"},
    )

    import backend.routes.missions as missions_route

    def _fake_generate_stage_metrics(*args, **kwargs):
        return {"generated": [], "rejected": [], "skipped": []}

    monkeypatch.setattr(
        "sculptor.mission_metrics.generate_stage_metrics",
        _fake_generate_stage_metrics,
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/stages/a/metric/regenerate",
    )
    assert r.status_code == 202, r.text


def test_run_mission_409_when_metric_regen_live_same_mission(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """POST .../run must not start mission_execute while a
    mission_stage_metric_regen for the SAME mission is in flight —
    both end in a non-atomic `save_mission` write to the same
    mission.json. Regression test: previously `run_mission`'s guard
    only recognized mission_decompose/mission_execute, so a live
    regen job did not block a concurrent run at all."""
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(project_dir, "m1", [_stage_dict("a", status="pending")])
    (project_dir / ".missions" / "m1" / "stages" / "a").mkdir(parents=True, exist_ok=True)

    app = client.app  # type: ignore[attr-defined]
    jobs = app.state.job_manager
    jobs.register_passive_job(
        kind="mission_stage_metric_regen",
        project_slug=slug,
        params={"mission_slug": "m1", "stage_name": "a"},
    )

    r = client.post(f"/projects/{slug}/missions/m1/run", json={})
    assert r.status_code == 409


# ── run guard: stage_metric_required + proceed_blind ───────────────────
def test_run_mission_409_when_stage_metric_required_and_missing(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(
        project_dir, "m1",
        [_stage_dict("a", status="pending")],
        run_defaults={"stage_metric_required": True},
    )

    r = client.post(f"/projects/{slug}/missions/m1/run", json={})
    assert r.status_code == 409
    body = r.json()
    assert body["type"] == "/problems/stage-metrics-missing"
    assert "a" in body.get("missing_stages", body.get("detail", ""))


def test_run_mission_proceed_blind_bypasses_guard(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(
        project_dir, "m1",
        [_stage_dict("a", status="pending")],
        run_defaults={"stage_metric_required": True},
    )

    # Stub the execute job so this doesn't try to spawn a real subprocess.
    import backend.routes.missions as missions_route

    def _fake_execute_job(**kwargs):
        async def _runner(job, cancel):
            return {"return_code": 0}
        return _runner

    monkeypatch.setattr(
        missions_route, "run_mission_execute_job", _fake_execute_job,
    )

    r = client.post(
        f"/projects/{slug}/missions/m1/run", json={"proceed_blind": True},
    )
    assert r.status_code == 202, r.text


def test_run_mission_no_guard_when_stage_has_metric(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _write_mission(
        project_dir, "m1",
        [_stage_dict("a", status="pending", steering_metric="g1_kick")],
        run_defaults={"stage_metric_required": True},
    )

    import backend.routes.missions as missions_route

    def _fake_execute_job(**kwargs):
        async def _runner(job, cancel):
            return {"return_code": 0}
        return _runner

    monkeypatch.setattr(
        missions_route, "run_mission_execute_job", _fake_execute_job,
    )

    r = client.post(f"/projects/{slug}/missions/m1/run", json={})
    assert r.status_code == 202, r.text
