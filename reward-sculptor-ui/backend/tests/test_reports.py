"""§7.6: reports endpoints — sharpened 404 for missing final_report.md
with explicit n_completed_iters counter so the UI can distinguish
"zero iters" from "iters done but report never built"."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


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
    # Fake a few completed iters by dropping `diagnosis.json` files.
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    for i in range(3):
        iter_dir = runs_dir / f"iter_{i}"
        iter_dir.mkdir()
        (iter_dir / "diagnosis.json").write_text(
            json.dumps({"failure_modes": [], "confidence": 0.5})
        )

    r = client.get(f"/projects/{slug}/reports/final_report.md")
    assert r.status_code == 404
    body = r.json()
    assert body.get("n_completed_iters") == 3
    detail = body.get("detail", "")
    assert "3 completed iter" in detail
    assert "POST" in detail or "/reports/build" in detail


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
