"""Tests for M7 Phase 7 polish surfaces.

Covers:
  - GET /projects/{slug}/rewards/{version}/diagnosis (7a).
  - sculptor.kg.viz `_inject_click_forwarder` round-trips cleanly (7e).

The frontend pieces (diff view, settings drawer, run-GPU card,
PaperDetailModal side-pane) are exercised via the existing
tsc --noEmit gate + manual smoke. No new backend endpoints for
7b/7c/7d, so this file is focused on 7a + 7e only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project(client: TestClient, name: str = "DiagProj") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


# ── GET /rewards/{version}/diagnosis ─────────────────────────────────
def test_diagnosis_endpoint_404_on_unknown_slug(
    client: TestClient, tmp_projects_root: Path
) -> None:
    r = client.get("/projects/nope/rewards/3/diagnosis")
    assert r.status_code == 404


def test_diagnosis_endpoint_404_on_v0(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """v0 is the scaffolded starter; there's no triggering diagnosis.
    Endpoint returns 404 with a friendly detail instead of producing
    a confusing empty response."""
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards/0/diagnosis")
    assert r.status_code == 404
    assert "v0" in r.json()["detail"].lower()


def test_diagnosis_endpoint_404_when_file_missing(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """A version request with no diagnosis.json on disk (e.g.
    human-authored v1 or a run that never reached diagnose) returns
    a 404 pointing at the expected iter_<n>/ path."""
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/rewards/5/diagnosis")
    assert r.status_code == 404
    assert "iter_4" in r.json()["detail"]


def test_diagnosis_endpoint_returns_payload(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Materialize a minimal diagnosis.json under runs/iter_0/ and
    confirm it round-trips through the endpoint."""
    slug = _make_project(client)
    iter_dir = tmp_projects_root / slug / "runs" / "iter_0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "failure_modes": ["component_imbalance"],
        "evidence": "alive_bonus saturates at 1.0",
        "proposed_edits": [
            {
                "target_term": "alive_bonus",
                "operation": "decrease",
                "rationale": "cap to prevent saturation",
                "suggested_value": "0.5",
                "paper_refs": ["1707.06347"],
            }
        ],
        "confidence": 0.8,
        "behavior_goal": "jump",
    }
    (iter_dir / "diagnosis.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    r = client.get(f"/projects/{slug}/rewards/1/diagnosis")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["failure_modes"] == ["component_imbalance"]
    assert body["proposed_edits"][0]["target_term"] == "alive_bonus"
    assert body["proposed_edits"][0]["paper_refs"] == ["1707.06347"]
    assert body["confidence"] == 0.8


def test_diagnosis_endpoint_500_on_malformed_json(
    client: TestClient, tmp_projects_root: Path
) -> None:
    slug = _make_project(client)
    iter_dir = tmp_projects_root / slug / "runs" / "iter_2"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "diagnosis.json").write_text("{not valid json", encoding="utf-8")

    r = client.get(f"/projects/{slug}/rewards/3/diagnosis")
    assert r.status_code == 500
    assert r.json()["type"] == "/problems/internal-error"


# ── sculptor.kg.viz click-forwarder injection (7e) ───────────────────
def test_click_forwarder_appends_before_body_close() -> None:
    from sculptor.kg.viz import _inject_click_forwarder

    orig = "<html><body><div id='net'></div></body></html>"
    out = _inject_click_forwarder(orig)
    assert "kg_node_click" in out
    assert out.index("kg_node_click") < out.rindex("</body>")


def test_click_forwarder_appends_when_body_close_missing() -> None:
    """Defensive: pyvis body HTML normally ends with </body></html>,
    but if a future pyvis change drops the closing tag we still emit
    the script (just appended)."""
    from sculptor.kg.viz import _inject_click_forwarder

    orig = "<div id='net'></div>"
    out = _inject_click_forwarder(orig)
    assert "kg_node_click" in out
    assert out.startswith(orig)
