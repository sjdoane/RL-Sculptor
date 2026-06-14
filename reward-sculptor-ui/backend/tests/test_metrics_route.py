"""§Ship 35: the auto-generated objective-metric route. The LLM
generation is mocked (no live calls); this verifies storage, listing,
calibration persistence, and the 404/contract behavior."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _make_project(client: TestClient, name: str = "Metrics") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 10, "behavior_goal": "trot"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _fake_generate(behavior_goal, out_dir, *, robot_hint=None, review=True,
                   on_event=None):
    if on_event:  # §Ship 40: exercise the progress channel
        on_event({"stage": "generating", "attempt": 1, "max": 3,
                  "message": "Generating candidate metric (attempt 1/3)…"})
        on_event({"stage": "validating", "attempt": 1, "max": 3,
                  "message": "Validating…"})
        on_event({"stage": "done", "accepted": True, "message": "Metric accepted."})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metric.py").write_text(
        "import numpy as np\ndef compute_spec(arrays, behavior, meta):\n"
        "    return {'spec_score': 0.5}\n", encoding="utf-8")
    return {
        "accepted": True, "validation_passed": True, "calibrated": False,
        "behavior_goal": behavior_goal,
        "validation": {"gates": {"ast_safety": True}, "reasons": [],
                       "archetype_scores": {"active": 0.9, "still": 0.0}},
        "review": {"approved": True, "concerns": [], "summary": "ok"},
        "source": "def compute_spec(...): ...", "recorded_at": "now",
    }


def test_generate_list_and_calibrate(client: TestClient, monkeypatch):
    from backend.services import metric_store, sculptor_bridge

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric", _fake_generate)
    monkeypatch.setattr(sculptor_bridge, "calibrate_objective_metric",
                        lambda metric_path, builtin, threshold=0.7:
                        {"ok": True, "spearman": 0.93, "builtin": builtin,
                         "threshold": threshold})

    slug = _make_project(client)

    # generate
    r = client.post(f"/projects/{slug}/metrics/generate",
                    json={"behavior_goal": "trot forward fast"})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["id"] == "gen_001" and rec["accepted"] and not rec["calibrated"]

    # list
    r = client.get(f"/projects/{slug}/metrics")
    assert r.status_code == 200
    metrics = r.json()
    assert len(metrics) == 1 and metrics[0]["id"] == "gen_001"

    # calibrate → earns steer-rights
    r = client.post(f"/projects/{slug}/metrics/gen_001/calibrate?against=go1_trot")
    assert r.status_code == 200, r.text
    assert r.json()["calibrated"] is True

    # list now reflects calibrated
    r = client.get(f"/projects/{slug}/metrics")
    assert r.json()[0]["calibrated"] is True

    # a second generate gets gen_002
    r = client.post(f"/projects/{slug}/metrics/generate",
                    json={"behavior_goal": "kick with one leg"})
    assert r.json()["id"] == "gen_002"


def test_generate_writes_live_progress_then_clears(client: TestClient, monkeypatch):
    """§Ship 40: while a generate is in flight the progress sidecar reflects
    the live pipeline stage; it is cleared (active=false) on completion."""
    from backend.services import metric_store, sculptor_bridge

    seen: dict = {}

    def fake_gen(behavior_goal, out_dir, *, robot_hint=None, review=True, on_event=None):
        if on_event:
            on_event({"stage": "generating", "attempt": 1, "max": 3,
                      "message": "Generating candidate metric (attempt 1/3)…"})
        # the route's on_event has now written the sidecar — read it back.
        proj = Path(out_dir).parent.parent
        seen["mid"] = metric_store.read_progress(proj)
        return _fake_generate(behavior_goal, out_dir, robot_hint=robot_hint, review=review)

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric", fake_gen)
    slug = _make_project(client, "MetricsProg")

    r = client.post(f"/projects/{slug}/metrics/generate",
                    json={"behavior_goal": "kick with one leg"})
    assert r.status_code == 200, r.text

    assert seen["mid"]["active"] is True
    assert seen["mid"]["stage"] == "generating"
    assert "Generating" in (seen["mid"].get("message") or "")
    # cleared after completion.
    prog = client.get(f"/projects/{slug}/metrics/generate/progress")
    assert prog.status_code == 200
    assert prog.json()["active"] is False


def test_generate_progress_idle_default(client: TestClient):
    slug = _make_project(client, "MetricsIdle")
    r = client.get(f"/projects/{slug}/metrics/generate/progress")
    assert r.status_code == 200 and r.json()["active"] is False


def test_generate_unknown_project_404(client: TestClient):
    r = client.post("/projects/nope/metrics/generate",
                    json={"behavior_goal": "trot forward"})
    assert r.status_code == 404


def test_list_metrics_empty(client: TestClient):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/metrics")
    assert r.status_code == 200 and r.json() == []
