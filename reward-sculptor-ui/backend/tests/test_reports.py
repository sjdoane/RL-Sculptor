"""§7.6: reports endpoints — sharpened 404 for missing final_report.md
with explicit n_completed_iters counter so the UI can distinguish
"zero iters" from "iters done but report never built"."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sculptor.run_manifests import (
    build_completion_manifest,
    build_rollout_input_manifest,
    build_train_input_manifest,
    manifest_sha256,
    write_json_atomic,
)
from sculptor.sculpt import _write_iteration_completion_marker


class _ReceiptAdapter:
    env_id = "report-route-test"
    robot = "unit"
    control_dt = 0.02


def _attest_iteration(project_dir: Path, iteration: int) -> None:
    iter_dir = project_dir / "runs" / f"iter_{iteration}"
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = iter_dir / "checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint:{iteration}".encode())
    for path, content in (
        (rollout_dir / "rollout.mp4", b"immutable-rollout"),
        (rollout_dir / "trajectory.npz", b"immutable-trajectory"),
    ):
        if not path.is_file():
            path.write_bytes(content)
    behavior = rollout_dir / "behavior.json"
    if not behavior.is_file():
        behavior.write_text("{}", encoding="utf-8")
    reward = project_dir / "rewards" / "v0.py"
    adapter = _ReceiptAdapter()
    request = build_train_input_manifest(
        adapter=adapter,
        iteration=iteration,
        reward_module_path=reward,
        steps=100,
        seed=iteration,
        init_policy_path=None,
        init_policy_mode="actor_critic",
    )
    train_input = {
        **request,
        "request_manifest_sha256": manifest_sha256(request),
        "effective_initialization": {
            "mode": None,
            "policy": None,
            "forwarded_to_adapter": False,
        },
    }
    rollout_input = build_rollout_input_manifest(
        adapter=adapter,
        iteration=iteration,
        checkpoint_path=checkpoint,
        reward_module_path=reward,
        n_episodes=1,
        seed=iteration,
        max_episode_steps=None,
        playback_speed=None,
        render_every=None,
        fps=None,
        render_width=None,
        render_height=None,
        render_env_index=None,
    )
    write_json_atomic(iter_dir / "train_request_manifest.json", request)
    write_json_atomic(iter_dir / "train_input_manifest.json", train_input)
    write_json_atomic(
        iter_dir / "train_completion_manifest.json",
        build_completion_manifest(train_input, [checkpoint]),
    )
    write_json_atomic(
        rollout_dir / "rollout_input_manifest.json", rollout_input,
    )
    write_json_atomic(
        rollout_dir / "rollout_completion_manifest.json",
        build_completion_manifest(
            rollout_input,
            [
                rollout_dir / "rollout.mp4",
                rollout_dir / "trajectory.npz",
                behavior,
            ],
        ),
    )
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=iteration,
        checkpoint_path=checkpoint,
        reward_version_before=0,
        reward_version_after=None,
        world_selection_hash=None,
    )


def _make_project_with_library(client: TestClient, name: str = "Reports") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 3, "behavior_goal": "hop"},
    )
    slug = r.json()["slug"]
    client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    return slug


# ── §Ship 25b: mission-quality telemetry route ────────────────────────


def test_mission_quality_empty_when_absent(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MqEmpty")
    r = client.get(f"/projects/{slug}/reports/mission-quality")
    assert r.status_code == 200
    assert r.json() == {"schema": 1, "missions": []}


def test_mission_quality_returns_doc(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MqDoc")
    reports = tmp_projects_root / slug / "reports"
    reports.mkdir(exist_ok=True)
    doc = {
        "schema": 1,
        "missions": [{
            "mission_slug": "flip-mission",
            "goal": "do a flip",
            "n_stages_at_start": 3,
            "n_stages_final": 4,
            "stages_executed": 4,
            "stages_succeeded": 3,
            "stage_success_rate": 0.75,
            "redecompositions": 1,
            "iterations_total": 18,
            "completed": False,
            "halted_reason": "criterion_not_met",
            "recorded_at": "2026-06-10T00:00:00+00:00",
        }],
    }
    (reports / "mission_quality.json").write_text(json.dumps(doc))
    r = client.get(f"/projects/{slug}/reports/mission-quality")
    assert r.status_code == 200
    assert r.json() == doc


def test_mission_quality_corrupt_file_is_empty_not_500(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MqCorrupt")
    reports = tmp_projects_root / slug / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "mission_quality.json").write_text("{broken")
    r = client.get(f"/projects/{slug}/reports/mission-quality")
    assert r.status_code == 200
    assert r.json() == {"schema": 1, "missions": []}


# ── §reports: actuator-limits route ───────────────────────────────────


def test_actuator_limits_empty_when_no_rollouts(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "ActEmpty")
    r = client.get(f"/projects/{slug}/reports/actuator-limits")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["available_iters"] == [] and body["motors"] == []


def test_actuator_limits_with_rollout(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "ActData")
    rd = tmp_projects_root / slug / "runs" / "iter_0" / "rollout"
    rd.mkdir(parents=True, exist_ok=True)
    T, E = 12, 4
    jv = np.zeros((T, E, 2))
    jt = np.zeros((T, E, 2))
    jv[..., 0] = 10.0
    jt[..., 0] = 100.0  # knee 10/20=50%, 100/139≈72%
    jv[..., 1] = 5.0
    jt[..., 1] = 20.0
    np.savez(rd / "trajectory.npz", joint_vel=jv, joint_torque=jt,
             projected_gravity_b=np.zeros((T, E, 3)))
    (rd / "mjcf_limits.json").write_text(
        json.dumps({"joint_names": ["left_knee_joint", "left_ankle_pitch_joint"]}))

    r = client.get(f"/projects/{slug}/reports/actuator-limits")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["has_torque"] and body["iter"] == 0
    assert body["available_iters"] == [0] and len(body["motors"]) == 2
    knee = next(m for m in body["motors"] if m["name"] == "left_knee_joint")
    assert knee["velocity_limit"] == 20.0 and abs(knee["speed_util_p99"] - 0.5) < 0.02


def test_actuator_limits_unknown_project_404(client: TestClient) -> None:
    r = client.get("/projects/nope-nope/reports/actuator-limits")
    assert r.status_code == 404


def test_mission_quality_unknown_project_404(client: TestClient) -> None:
    r = client.get("/projects/nope-nope/reports/mission-quality")
    assert r.status_code == 404


def test_final_report_404_with_zero_iters_cites_zero(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Fresh project, no runs yet — 404 with `n_completed_iters=0` and a
    detail nudging the user toward running first."""
    slug = _make_project_with_library(client, "ZeroIters")
    r = client.get(f"/projects/{slug}/reports/final_report.md")
    assert r.status_code == 404
    body = r.json()
    assert body.get("n_completed_iters") == 0
    # Detail mentions "no completed sculpt iters" so the frontend can
    # surface a "run first" CTA instead of a "build report" CTA.
    assert "sculpt iter" in body.get("detail", "").lower()


def test_final_report_404_with_some_iters_cites_count(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Project has iters on disk but no final_report.md yet — 404 with
    accurate count so the UI can say 'you have N iters; click build'."""
    slug = _make_project_with_library(client, "SomeIters")
    project_dir = tmp_projects_root / slug
    # Plant fully phase-attested schema-3 completions.
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    for i in range(3):
        iter_dir = runs_dir / f"iter_{i}"
        iter_dir.mkdir()
        (iter_dir / "diagnosis.json").write_text(
            json.dumps({"failure_modes": [], "confidence": 0.5})
        )
        _attest_iteration(project_dir, i)

    r = client.get(f"/projects/{slug}/reports/final_report.md")
    assert r.status_code == 404
    body = r.json()
    assert body.get("n_completed_iters") == 3
    detail = body.get("detail", "")
    assert "3 completed iter" in detail
    assert "POST" in detail or "/reports/build" in detail


def test_final_report_does_not_count_diagnosis_only_directories(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "DiagnosisIsNotCompletion")
    iter_dir = tmp_projects_root / slug / "runs" / "iter_0"
    iter_dir.mkdir(parents=True)
    (iter_dir / "diagnosis.json").write_text("{}", encoding="utf-8")

    response = client.get(f"/projects/{slug}/reports/final_report.md")

    assert response.status_code == 404
    assert response.json()["n_completed_iters"] == 0


def test_final_report_404_unchanged_when_project_missing(
    client: TestClient,
) -> None:
    """Unknown slug still returns the ProjectStore's 404 (`project not
    found`), not our new `n_completed_iters` variant."""
    r = client.get("/projects/does-not-exist/reports/final_report.md")
    assert r.status_code == 404
    body = r.json()
    # This route reaches `_project_dir` which returns None; the 404
    # uses `project not found` without the n_completed_iters field.
    assert "n_completed_iters" not in body


def test_final_report_200_when_file_exists(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Happy-path regression: existing final_report.md serves with 200."""
    slug = _make_project_with_library(client, "HasReport")
    report_path = (
        tmp_projects_root / slug / "reports" / "final_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Final Report\n\nAll good.")

    r = client.get(f"/projects/{slug}/reports/final_report.md")
    assert r.status_code == 200
    assert "final report" in r.text.lower()
    assert r.headers["x-rewardsculptor-report-state"] == "stale"
    assert "stale retained report" in r.text.lower()


# ── §chunk C1: mission-aware report build / sources / mission routes ──
def _write_mission_on_disk(
    project_dir: Path, mission_slug: str, *,
    stage_specs: list[tuple[str, int, list[float]]],
) -> Path:
    """Hand-build a mission tree on disk — same shape the mission
    orchestrator would leave behind, without going through decompose
    (which is an LLM call). `stage_specs` is `[(name, n_iters,
    metric_history), ...]` in stage order. Returns the mission dir."""
    from sculptor.mission import Mission, Stage, save_mission
    from sculptor.sculpt import sculpt_init

    from backend.services import mission_store

    mdir = mission_store.mission_dir(project_dir, mission_slug)
    stages: list[Stage] = []
    for i, (name, n_iters, metric_history) in enumerate(stage_specs):
        stage_dir = mdir / "stages" / name
        sculpt_init(stage_dir, "gym_sb3")
        for it in range(n_iters):
            iter_dir = stage_dir / "runs" / f"iter_{it}"
            (iter_dir / "rollout").mkdir(parents=True, exist_ok=True)
            (iter_dir / "metrics.json").write_text(json.dumps(
                {"metrics": {"mean_return": metric_history[it]}}))
            (iter_dir / "rollout" / "behavior.json").write_text(json.dumps(
                {"mean_return": metric_history[it]}))
            (iter_dir / "diagnosis.json").write_text(json.dumps({
                "failure_modes": [], "proposed_edits": [],
                "confidence": 0.5, "behavior_goal": name,
            }))
            _attest_iteration(stage_dir, it)
        (stage_dir / "reports").mkdir(exist_ok=True)
        (stage_dir / "reports" / "metric_history.json").write_text(json.dumps({
            "primary_metric": "mean_return", "history": metric_history,
        }))
        stages.append(Stage(
            name=name, goal_text=f"reach {name}",
            success_criterion="behavior['mean_return'] > 0",
            max_iterations=10, parent_stage=stage_specs[i - 1][0] if i > 0 else None,
            reward_seed_prompt=f"reward {name}",
            status="succeeded" if metric_history else "pending",
            best_metric=metric_history[-1] if metric_history else None,
            iterations_used=n_iters,
        ))
    mission = Mission(
        goal="do a complex multi-stage behavior", stages=stages,
        decomposition_model="claude-test",
        decomposition_rationale="stand then walk",
    )
    save_mission(mission, mdir)
    return mdir


def test_build_report_with_mission_slug_writes_mission_report(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MissionBuild")
    project_dir = tmp_projects_root / slug
    _write_mission_on_disk(
        project_dir, "walk-mission",
        stage_specs=[("stand", 2, [3.0, 9.0]), ("walk", 2, [1.0, 15.0])],
    )

    r = client.post(
        f"/projects/{slug}/reports/build",
        json={"mission_slug": "walk-mission"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["mission_slug"] == "walk-mission"
    assert body["final_report_md_path"]

    md_path = project_dir / ".missions" / "walk-mission" / "reports" / "final_report.md"
    assert md_path.is_file()
    md = md_path.read_text(encoding="utf-8")
    assert "# Sculpt Mission Report" in md
    assert "## Stage: `stand`" in md
    assert "## Stage: `walk`" in md


def test_build_report_unknown_mission_slug_404(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MissionBuildBadSlug")
    r = client.post(
        f"/projects/{slug}/reports/build",
        json={"mission_slug": "does-not-exist"},
    )
    assert r.status_code == 404
    assert "mission not found" in r.json().get("title", "").lower()


def test_build_report_no_body_still_uses_legacy_path(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """§chunk C1 regression: POST with no body at all (not even `{}`)
    must still hit the legacy project-runs build path unchanged."""
    slug = _make_project_with_library(client, "MissionBuildLegacyNoBody")
    project_dir = tmp_projects_root / slug
    runs_dir = project_dir / "runs"
    for i in range(2):
        iter_dir = runs_dir / f"iter_{i}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "diagnosis.json").write_text(json.dumps(
            {"failure_modes": [], "confidence": 0.5}))
        (iter_dir / "rollout").mkdir(exist_ok=True)
    (project_dir / "reports").mkdir(exist_ok=True)
    (project_dir / "reports" / "metric_history.json").write_text(json.dumps(
        {"primary_metric": "mean_return", "history": [1.0, 2.0]}))

    r = client.post(f"/projects/{slug}/reports/build")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert "mission_slug" not in body
    assert (project_dir / "reports" / "final_report.md").is_file()


def test_mission_report_md_and_mp4_routes(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MissionReportRoutes")
    project_dir = tmp_projects_root / slug
    _write_mission_on_disk(
        project_dir, "jump-mission",
        stage_specs=[("crouch", 1, [2.0])],
    )
    r = client.post(
        f"/projects/{slug}/reports/build", json={"mission_slug": "jump-mission"})
    assert r.status_code == 200, r.text

    r_md = client.get(f"/projects/{slug}/missions/jump-mission/report/final_report.md")
    assert r_md.status_code == 200
    assert "Sculpt Mission Report" in r_md.text

    r_mp4 = client.get(f"/projects/{slug}/missions/jump-mission/report/final.mp4")
    # mp4 build may fail if ffmpeg can't stitch our placeholder rollout
    # videos, but the route itself must resolve — 200 when final.mp4 was
    # written, 404 "timelapse not built" otherwise. Never 500.
    assert r_mp4.status_code in (200, 404)
    if r_mp4.status_code == 404:
        assert "timelapse not built" in r_mp4.json().get("title", "").lower()


def test_mission_report_md_404_before_build(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MissionReportNoBuild")
    project_dir = tmp_projects_root / slug
    _write_mission_on_disk(
        project_dir, "unbuilt-mission", stage_specs=[("crouch", 1, [2.0])])

    r = client.get(f"/projects/{slug}/missions/unbuilt-mission/report/final_report.md")
    assert r.status_code == 404
    body = r.json()
    assert body.get("n_completed_iters") == 1


def test_mission_report_unknown_mission_slug_404(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "MissionReportBadSlug")
    r = client.get(f"/projects/{slug}/missions/nope-mission/report/final_report.md")
    assert r.status_code == 404
    r2 = client.get(f"/projects/{slug}/missions/nope-mission/report/final.mp4")
    assert r2.status_code == 404


def test_mission_report_path_traversal_guard(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """A mission_slug with path-traversal characters is rejected before
    it reaches the filesystem — mirrors routes/runs.py's segment guard."""
    slug = _make_project_with_library(client, "MissionReportTraversal")
    r = client.get(
        f"/projects/{slug}/missions/..%2f..%2f..%2fetc/report/final_report.md")
    assert r.status_code in (404, 422)


def test_report_sources_lists_project_runs_and_missions(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project_with_library(client, "SourcesPicker")
    project_dir = tmp_projects_root / slug

    r = client.get(f"/projects/{slug}/reports/sources")
    assert r.status_code == 200
    body = r.json()
    assert body["project_runs"]["n_iters"] == 0
    assert body["project_runs"]["has_report"] is False
    assert body["project_runs"]["report_state"]["state"] == "missing"
    assert body["missions"] == []

    # Add two project-level iters + a final_report.md.
    runs_dir = project_dir / "runs"
    for i in range(2):
        (runs_dir / f"iter_{i}").mkdir(parents=True)
        _attest_iteration(project_dir, i)
    (project_dir / "reports").mkdir(exist_ok=True)
    (project_dir / "reports" / "final_report.md").write_text("# done")

    # Add one mission with a built report + one without.
    _write_mission_on_disk(
        project_dir, "done-mission", stage_specs=[("crouch", 1, [2.0])])
    client.post(
        f"/projects/{slug}/reports/build", json={"mission_slug": "done-mission"})
    _write_mission_on_disk(
        project_dir, "pending-mission", stage_specs=[("crouch", 1, [2.0])])

    r = client.get(f"/projects/{slug}/reports/sources")
    assert r.status_code == 200
    body = r.json()
    assert body["project_runs"]["n_iters"] == 2
    assert body["project_runs"]["has_report"] is True
    assert body["project_runs"]["report_state"]["state"] == "stale"
    by_slug = {m["mission_slug"]: m for m in body["missions"]}
    assert by_slug["done-mission"]["has_report"] is True
    assert by_slug["done-mission"]["report_state"]["state"] == "current"
    assert by_slug["done-mission"]["goal"] == "do a complex multi-stage behavior"
    assert by_slug["pending-mission"]["has_report"] is False
    assert by_slug["pending-mission"]["report_state"]["state"] == "missing"
    assert "lifecycle" in by_slug["done-mission"]


def test_report_sources_unknown_project_404(client: TestClient) -> None:
    r = client.get("/projects/nope-nope/reports/sources")
    assert r.status_code == 404
