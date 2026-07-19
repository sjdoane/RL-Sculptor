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


def test_validate_clean_then_detects_tamper(client: TestClient):
    """The integrity endpoint passes on a fresh promotion and pinpoints a
    hand-tampered immutable artifact by kind."""
    import json as _json
    from pathlib import Path as _Path

    slug = _make_project(client)
    assert client.get(f"/projects/{slug}/worlds/validate").status_code == 404

    draft = _author(client, slug)
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text

    clean = client.get(f"/projects/{slug}/worlds/validate")
    assert clean.status_code == 200, clean.text
    assert clean.json()["ok"] is True and clean.json()["errors"] == []

    # Tamper: flip a value inside the immutable world artifact.
    project_dir = _Path(client.get(f"/projects/{slug}").json()["project_dir"])
    world_path = next((project_dir / "env").glob("world_v*.json"))
    payload = _json.loads(world_path.read_text())
    payload["shared"]["eval_seed"] = 31337
    world_path.write_text(_json.dumps(payload))

    tampered = client.get(f"/projects/{slug}/worlds/validate")
    assert tampered.status_code == 200
    body = tampered.json()
    assert body["ok"] is False
    assert any("world" in e and "hash" in e for e in body["errors"])


def test_selection_carries_clarification_provenance(client: TestClient):
    """The selection view discloses who decided each parameter — explicit
    user answers vs disclosed system defaults."""
    slug = _make_project(client)
    draft = _author(client, slug)
    first_q = draft["clarification_plan"]["pages"][0]["questions"][0]
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"],
              "answers": [{"question_id": first_q["question_id"],
                           "choice_id": first_q["choices"][0]["choice_id"]}]},
    )
    assert r.status_code == 200, r.text
    selection = client.get(f"/projects/{slug}/worlds/selection").json()
    sources = selection["clarifications"]["answer_sources"]
    assert sources.get("user") == 1
    assert sources.get("default", 0) >= 1
    assert selection["clarifications"]["answers"]


def test_lineage_carries_eval_model_hash(client: TestClient):
    slug = _make_project(client)
    draft = _author(client, slug)
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    lineage = client.get(f"/projects/{slug}/worlds/lineage").json()
    assert lineage and lineage[-1]["eval_model_hash"]


def test_curriculum_progression_from_run_stats(client: TestClient):
    import json as _json
    from pathlib import Path as _Path

    slug = _make_project(client)
    empty = client.get(f"/projects/{slug}/worlds/curriculum").json()
    assert empty == {"run": None, "iterations": []}

    project_dir = _Path(client.get(f"/projects/{slug}").json()["project_dir"])
    run_dir = project_dir / "runs" / "run_001"
    for i, mean in enumerate([0.4, 1.1, 2.3]):
        iter_dir = run_dir / f"iter_{i}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "world_curriculum_stats.json").write_text(_json.dumps({
            "mean_level": mean, "max_level": 6, "num_envs": 2048,
            "histogram": {"0": 1024, "1": 1024},
        }))
    body = client.get(f"/projects/{slug}/worlds/curriculum").json()
    assert body["run"] == "run_001"
    assert [e["iter"] for e in body["iterations"]] == [0, 1, 2]
    assert body["iterations"][-1]["mean_level"] == 2.3


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


def _set_project_robot(client: TestClient, slug: str, library_slug: str) -> None:
    """Stamp a library robot into the project sidecar the way the
    library-driven create flow does — the world author reads it as the
    robot default."""
    import json as _json
    from pathlib import Path as _Path

    project_dir = _Path(client.get(f"/projects/{slug}").json()["project_dir"])
    meta_path = project_dir / "metadata.json"
    meta = _json.loads(meta_path.read_text())
    meta["robot_source"] = {"kind": "library", "library_slug": library_slug}
    meta_path.write_text(_json.dumps(meta))


def test_author_blank_robot_defaults_to_project_robot(client: TestClient):
    """A blank robot field must resolve to the PROJECT's configured robot,
    not a prompt-based guess — the live trap: a parkour prompt on a Go1
    project auto-selected unitree_g1:base."""
    slug = _make_project(client)
    _set_project_robot(client, slug, "unitree_go1")
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["capability_id"] == "unitree_go1:base"
    assert draft["project_capability_id"] == "unitree_go1:base"
    assert draft["robot_matches_project"] is True


def test_author_explicit_robot_mismatch_is_disclosed(client: TestClient):
    """An explicit robot that differs from the project robot is allowed
    but disclosed in the draft AND in the promoted selection."""
    slug = _make_project(client)
    _set_project_robot(client, slug, "unitree_go1")
    draft = _author(client, slug)  # pins unitree_g1:base explicitly
    assert draft["capability_id"] == "unitree_g1:base"
    assert draft["project_capability_id"] == "unitree_go1:base"
    assert draft["robot_matches_project"] is False

    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    summary = client.get(
        f"/projects/{slug}/worlds/selection").json()["shared_summary"]
    assert summary["robot"] == "unitree_g1:base"
    assert summary["project_capability_id"] == "unitree_go1:base"
    assert summary["robot_matches_project"] is False


def test_author_without_project_robot_reports_unknown_match(
        client: TestClient):
    """No library robot on the project → no capability mapping → the
    match state is None (unknown), not a false mismatch alarm."""
    slug = _make_project(client)
    draft = _author(client, slug)
    assert draft["project_capability_id"] is None
    assert draft["robot_matches_project"] is None


def test_selection_breaks_down_course_element_types(client: TestClient):
    """'5 course elements' next to a 3-box render looks wrong; the
    summary must expose the per-type breakdown (gaps have no geometry)."""
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": r.json()["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    summary = client.get(
        f"/projects/{slug}/worlds/selection").json()["shared_summary"]
    breakdown = summary["course_breakdown"]
    assert sum(breakdown.values()) == summary["course_elements"]
    assert breakdown.get("platform", 0) > 0
    assert breakdown.get("gap", 0) > 0


def test_preview_pose_uses_model_keyframe():
    """The preview must render keyframe 0 (the compiled scene's init
    pose), not qpos0 — a Go1 at qpos0 has fully extended legs and reads
    as a humanoid at thumbnail size."""
    import mujoco
    import numpy as np

    from backend.services.preview_renderer import _posed_data

    xml = """
    <mujoco>
      <worldbody>
        <body name=\"box\" pos=\"0 0 1\">
          <freejoint/>
          <geom type=\"box\" size=\"0.1 0.1 0.1\"/>
        </body>
      </worldbody>
      <keyframe>
        <key name=\"init_state\" qpos=\"0.5 0.25 0.4 1 0 0 0\"/>
      </keyframe>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = _posed_data(model)
    assert np.allclose(data.qpos, [0.5, 0.25, 0.4, 1, 0, 0, 0])

    # No keyframe → model default pose, unchanged behavior.
    plain = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body pos=\"0 0 1\"><freejoint/>"
        "<geom type=\"box\" size=\"0.1 0.1 0.1\"/></body>"
        "</worldbody></mujoco>")
    assert _posed_data(plain).qpos[2] == 1.0


def test_scene_endpoint_serves_semantic_scene_graph(client: TestClient):
    """The 3D-viewer scene graph is exported from the materialized
    evaluation model with semantic entity classification, and the
    authoritative env/ tuple is byte-identical afterwards."""
    import hashlib as _hashlib
    from pathlib import Path as _Path

    slug = _make_project(client)
    assert client.get(f"/projects/{slug}/worlds/scene").status_code == 404

    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": r.json()["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text

    project_dir = _Path(client.get(f"/projects/{slug}").json()["project_dir"])
    env_before = {
        p.name: _hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (project_dir / "env").glob("*.json")}

    r = client.get(f"/projects/{slug}/worlds/scene")
    assert r.status_code == 200, r.text
    scene = r.json()
    assert scene["selection_version"] == 1
    assert scene["geoms"], "compiled scene must expose geoms"

    course_geoms = [g for g in scene["geoms"]
                    if g["entity_kind"] == "course_element"]
    assert course_geoms, "platform geoms must classify as course elements"
    # Exact-set robot classification: EVERY geom on a robot/ body is
    # robot, and nothing else is — partial mislabeling must fail.
    robot_body_geoms = {g["name"] for g in scene["geoms"]
                       if g["body"] == "robot"
                       or g["body"].startswith("robot/")}
    robot_classified = {g["name"] for g in scene["geoms"]
                       if g["entity_kind"] == "robot"}
    assert robot_classified == robot_body_geoms and robot_classified
    mesh_names = {g["mesh"] for g in scene["geoms"] if g.get("mesh")}
    assert mesh_names and mesh_names <= set(scene["meshes"].keys())

    by_kind = {e["kind"]: e for e in scene["entities"]}
    assert "robot" in by_kind and "terrain" in by_kind
    course_entities = [e for e in scene["entities"]
                       if e["kind"] == "course_element"]
    platforms = [e for e in course_entities if e["element"] == "platform"]
    gaps = [e for e in course_entities if e["element"] == "gap"]
    assert platforms and all(e["geoms"] for e in platforms)
    assert gaps and all(not e["geoms"] for e in gaps)
    assert all(e.get("marker") for e in gaps), (
        "interior gaps need a spacing marker for the viewer")
    # Marker interval correctness: each gap marker spans exactly the
    # space between its neighbouring platforms' facing edges.
    course_order = [e["id"] for e in course_entities]
    geom_by_entity = {
        g["entity_id"]: g for g in course_geoms if g["type"] == "box"}
    for gap in gaps:
        idx = course_order.index(gap["id"])
        prev_geom = geom_by_entity[course_order[idx - 1]]
        next_geom = geom_by_entity[course_order[idx + 1]]
        marker = gap["marker"]
        span_lo = marker["pos"][0] - marker["size"][0]
        span_hi = marker["pos"][0] + marker["size"][0]
        assert abs(span_lo - (prev_geom["pos"][0] + prev_geom["size"][0])) < 1e-3
        assert abs(span_hi - (next_geom["pos"][0] - next_geom["size"][0])) < 1e-3
        assert span_hi > span_lo
    assert all(e["params"] for e in course_entities)

    # scene export is read-only over the authoritative tuple
    env_after = {
        p.name: _hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (project_dir / "env").glob("*.json")
        if not p.name.startswith("world_scene_v")}
    assert env_after == env_before

    # cached: a second call returns the identical payload
    assert client.get(f"/projects/{slug}/worlds/scene").json() == scene


def test_preview_draft_dry_run_compiles_without_promoting(
        client: TestClient):
    """The build-loop dry-run returns the full gate report + scene while
    leaving the project without any promoted selection; different answers
    produce a different compiled scene."""
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    draft = r.json()

    r = client.post(
        f"/projects/{slug}/worlds/author/preview",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["ok"] is True
    gates = {g["gate"]: g["ok"] for g in preview["admission"]["gates"]}
    assert gates.get("build") is True and gates.get("settle") is True
    assert preview["scene"]["geoms"]
    assert preview["summary"]["course_breakdown"].get("platform", 0) > 0

    # dry-run must NOT promote
    assert client.get(f"/projects/{slug}/worlds/selection").status_code == 404
    assert client.get(f"/projects/{slug}/worlds/lineage").json() == []

    # an explicit answer changes the compiled scene deterministically
    first_q = draft["clarification_plan"]["pages"][0]["questions"][0]
    r = client.post(
        f"/projects/{slug}/worlds/author/preview",
        json={"session_id": draft["session_id"],
              "answers": [{"question_id": first_q["question_id"],
                           "choice_id": first_q["choices"][0]["choice_id"]}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["result_hash"] != preview["result_hash"]

    # unknown session and unknown question keep their contracts
    r = client.post(
        f"/projects/{slug}/worlds/author/preview",
        json={"session_id": "nonexistent", "answers": []})
    assert r.status_code == 404
    r = client.post(
        f"/projects/{slug}/worlds/author/preview",
        json={"session_id": draft["session_id"],
              "answers": [{"question_id": "bogus",
                           "choice_id": "system_default"}]})
    assert r.status_code == 422


def test_variation_edit_promotes_new_version_same_eval_lineage(
        client: TestClient):
    """UI train-variation edits ride the §10.2 primitive: a valid edit
    promotes a NEW selection version under the SAME evaluation lineage
    with a byte-identical evaluation model; unknown ids and no-ops are
    rejected without touching the selection."""
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": r.json()["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    before = client.get(f"/projects/{slug}/worlds/selection").json()
    lineage_before = client.get(f"/projects/{slug}/worlds/lineage").json()
    variation = before["train_variations"][0]
    dist = dict(variation["distribution"])
    assert dist.get("kind") == "uniform"

    # no-op + unknown id → rejected, selection untouched
    r = client.post(
        f"/projects/{slug}/worlds/variations",
        json={"edits": [
            {"variation_id": variation["id"], "distribution": dist},
            {"variation_id": "nonexistent", "distribution": dist},
        ]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] == [] and len(body["rejected"]) == 2
    assert body["selection"] is None
    after_noop = client.get(f"/projects/{slug}/worlds/selection").json()
    assert (after_noop["selection"]["tuple_hash"]
            == before["selection"]["tuple_hash"])

    # real edit → new version, same lineage, evaluation provably frozen
    widened = {**dist, "high": float(dist["high"]) * 1.1}
    r = client.post(
        f"/projects/{slug}/worlds/variations",
        json={"edits": [
            {"variation_id": variation["id"], "distribution": widened,
             "rationale": "widen for the build loop test"},
        ]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["variation_id"] for e in body["applied"]] == [variation["id"]]
    assert body["selection"]["selection_version"] == 2

    after = client.get(f"/projects/{slug}/worlds/selection").json()
    assert after["selection"]["selection_version"] == 2
    assert (after["selection"]["evaluation_lineage"]
            == before["selection"]["evaluation_lineage"])
    edited = next(v for v in after["train_variations"]
                  if v["id"] == variation["id"])
    assert edited["distribution"]["high"] == widened["high"]

    lineage_after = client.get(f"/projects/{slug}/worlds/lineage").json()
    assert len(lineage_after) == len(lineage_before) + 1
    # §6.1 the Act-III proof: identical compiled evaluation across entries
    assert (lineage_after[-1]["eval_model_hash"]
            == lineage_after[0]["eval_model_hash"])
    ok = client.get(f"/projects/{slug}/worlds/validate").json()
    assert ok["ok"] is True, ok


def test_parkour_course_world_admits_and_promotes(client: TestClient):
    """Course-bearing worlds exercise with_admission -> to_dict on a
    non-empty course — the exact path a live UI parkour authoring crashed
    on while every empty-course test stayed green."""
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/worlds/author",
        json={"prompt":
              "traverse a parkour course of ascending boxes with gaps"},
    )
    assert r.status_code == 200, r.text
    draft = r.json()
    r = client.post(
        f"/projects/{slug}/worlds/author/apply",
        json={"session_id": draft["session_id"], "answers": []},
    )
    assert r.status_code == 200, r.text
    selection = client.get(f"/projects/{slug}/worlds/selection").json()
    assert selection["shared_summary"]["course_elements"] > 0
