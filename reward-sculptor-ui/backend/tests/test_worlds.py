"""Environment-authoring endpoints (env-authoring item 5).

The offline author is deterministic and needs no LLM, and the admission
gate chain runs a real (CPU) MuJoCo compile — so the full
author → clarify → admit → promote flow is exercised for real. The
shared KG is redirected to an empty tmp store by conftest, so grounding
resolves to no items without loading the embedder.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

PROMPT = "stay stable and walk on uneven rough terrain"


def _make_project(client: TestClient, name: str = "Worlds") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 10,
              "behavior_goal": "walk on rough ground"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _author(client: TestClient, slug: str) -> dict:
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt": PROMPT, "robot_capability_id": "unitree_g1:base"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_author_returns_draft_with_disclosed_defaults(client: TestClient):
    slug = _make_project(client)
    draft = _author(client, slug)
    assert draft["session_id"] and draft["draft_hash"]
    assert draft["capability_id"] == "unitree_g1:base"
    pages = draft["clarification_plan"]["pages"]
    assert pages, "load-bearing defaults must queue clarification questions"
    for page in pages:
        for question in page["questions"]:
            assert question["question_id"] and question["prompt"]
            assert question["choices"], "every question needs choices"
            # the clarifier contract: every question discloses a
            # system-default hand-back option with its value + reason
            system_default = question["system_default"]
            assert system_default["choice_id"] == "system_default"
            assert system_default["label"].startswith("System decides")
            assert system_default["reason"]
    report = draft["underspecification_report"]
    assert report["defaulted_load_bearing_paths"]


def test_apply_promotes_selection_and_lineage(client: TestClient):
    slug = _make_project(client)
    draft = _author(client, slug)
    first_q = draft["clarification_plan"]["pages"][0]["questions"][0]
    explicit = {
        "question_id": first_q["question_id"],
        "choice_id": first_q["choices"][0]["choice_id"],
    }

    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": [explicit]},
    )
    assert r.status_code == 200, r.text
    applied = r.json()
    assert applied["ok"] is True
    assert applied["admission"]["ok"] is True
    assert applied["evaluation_lineage"].startswith("world-")
    ledger = applied["clarification_answers"]
    assert ledger > 0

    # the promoted tuple is now the project's authoritative selection
    r = client.get(f"/projects/{slug}/worlds/selection")
    assert r.status_code == 200, r.text
    selection = r.json()
    assert selection["shared_summary"]["robot"] == "unitree_g1:base"
    assert selection["shared_summary"]["terrain_kind"] == "generator"
    assert selection["world_meta"]["prompt"] == PROMPT
    assert selection["train_variations"], "uneven author registers variations"

    # lineage lists the immutable selection history
    r = client.get(f"/projects/{slug}/worlds/lineage")
    assert r.status_code == 200
    lineage = r.json()
    assert len(lineage) >= 1
    assert lineage[-1]["tuple_hash"] == selection["selection"]["tuple_hash"]
    assert lineage[-1]["refs"]["world"]["version"]


def test_selection_404_before_any_authoring(client: TestClient):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/worlds/selection")
    assert r.status_code == 404
    r = client.get(f"/projects/{slug}/worlds/lineage")
    assert r.status_code == 200 and r.json() == []


def test_apply_unknown_session_and_question(client: TestClient):
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": "nonexistent-session", "answers": []},
    )
    assert r.status_code == 404, r.text

    draft = _author(client, slug)
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"],
              "answers": [{"question_id": "not_a_real_question",
                           "choice_id": "system_default"}]},
    )
    assert r.status_code == 422, r.text
    assert "unknown clarification question" in (r.json()["detail"] or "")


def test_unknown_project_404s_everywhere(client: TestClient):
    for method, url, body in (
        ("post", "/projects/nope/worlds/author", {"prompt": PROMPT}),
        ("post", "/projects/nope/worlds/author/apply",
         {"session_id": "x", "answers": []}),
        ("get", "/projects/nope/worlds/selection", None),
        ("get", "/projects/nope/worlds/lineage", None),
    ):
        r = (client.post(url, json=body) if method == "post"
             else client.get(url))
        assert r.status_code == 404, f"{url}: {r.text}"


def test_ungroundable_prompt_is_422(client: TestClient):
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt": "do fifty backflips through a wormhole"},
    )
    assert r.status_code == 422, r.text


def test_world_preview_renders_materialized_scene(client: TestClient):
    """The preview endpoint renders the promoted selection's materialized
    evaluation MJB (real offscreen MuJoCo render) and caches per
    selection version + angle."""
    slug = _make_project(client)
    assert client.get(f"/projects/{slug}/worlds/preview").status_code == 404

    draft = _author(client, slug)
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/projects/{slug}/worlds/preview")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    # cached: second call byte-identical
    again = client.get(f"/projects/{slug}/worlds/preview")
    assert again.content == r.content

    bad = client.get(f"/projects/{slug}/worlds/preview?angle=bogus")
    assert bad.status_code == 422
