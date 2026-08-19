"""KG grounding for the world author (plan item 3).

Contract under test:
- retrieval surfaces Techniques, FailureModes, and Papers for a prompt;
- paper retrieval honors the author-intent tag filter;
- grounding is strictly fail-soft (broken embedder / empty store =>
  ungrounded authoring, never an exception);
- the meta.grounding ledger carries node IDs through both the offline
  and model authoring paths, even when the model omits it;
- the authoring-model request receives the rich kg_grounding evidence;
- the CLI flag wires retrieval in by default and --no-kg-grounding
  keeps authoring fully offline.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sculptor.kg import query as qmod
from sculptor.kg.schema import (
    Edge,
    FailureMode,
    Paper,
    Relation,
    Technique,
    make_failure_mode_id,
    make_paper_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG
from sculptor.world.author import author_environment
from sculptor.world.grounding import (
    ExplicitGroundingError,
    GroundingItem,
    gather_grounding,
    grounding_context,
    grounding_ids,
)

PROMPT = "stay stable and walk on uneven rough terrain"


class _ConstantEmbedder:
    """Every text -> the same unit vector, so cosine similarity is 1.0
    everywhere and retrieval floors cannot filter the test fixtures."""

    def _vec(self) -> np.ndarray:
        v = np.zeros(8, dtype=np.float64)
        v[0] = 1.0
        return v

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._vec()
        return np.stack([self._vec() for _ in texts])


@pytest.fixture
def const_embedder(monkeypatch):
    emb = _ConstantEmbedder()
    monkeypatch.setattr(qmod, "_get_embedder", lambda *a, **k: emb)
    return emb


@pytest.fixture
def seeded_kg(tmp_path: Path):
    store = SculptorKG(tmp_path / "kg.db")
    aid = "2109.11978"
    paper = Paper(
        id=make_paper_id(aid), arxiv_id=aid,
        title="Rough-terrain curriculum grids", year=2021,
        abstract="A grid of sub-terrains with difficulty rows.",
        rationale="Terrain-curriculum reference for uneven-ground prompts.",
        tags=["terrain"], tier="S",
    )
    tech = Technique(
        id=make_technique_id("terrain difficulty curriculum"),
        name="terrain difficulty curriculum",
        description="Grid of sub-terrains ordered by difficulty; promote "
                    "environments as the policy clears easier rows.",
    )
    fail = FailureMode(
        id=make_failure_mode_id("foot trap on box edges"),
        name="foot trap on box edges",
        description="Feet catch on obstacle edges and the robot trips.",
    )
    for node in (paper, tech, fail):
        store.add_node(node)
    store.add_edge(Edge(src=paper.id, dst=tech.id,
                        relation=Relation.INTRODUCES))
    yield store
    store.close()


def test_gather_returns_all_three_kinds(seeded_kg, const_embedder):
    items = gather_grounding(PROMPT, store=seeded_kg)
    assert {item.kind for item in items} == {
        "technique", "failure_mode", "paper"}
    ids = grounding_ids(items)
    assert len(ids) == len(set(ids)) == 3
    by_kind = {item.kind: item for item in items}
    assert by_kind["technique"].guidance.startswith("Grid of sub-terrains")
    assert by_kind["technique"].score == pytest.approx(1.0)
    assert by_kind["paper"].node_id == "paper:2109.11978"
    assert "Terrain-curriculum reference" in by_kind["paper"].guidance
    assert by_kind["failure_mode"].name == "foot trap on box edges"


def test_paper_retrieval_honors_intent_tag_filter(seeded_kg, const_embedder):
    off_domain = Paper(
        id=make_paper_id("2400.99999"), arxiv_id="2400.99999",
        title="Object rearrangement", abstract="Tabletop objects.",
        tags=["objects"], tier="A",
    )
    seeded_kg.add_node(off_domain)
    ids = grounding_ids(gather_grounding(PROMPT, store=seeded_kg))
    assert "paper:2109.11978" in ids
    assert off_domain.id not in ids  # terrain intent filters objects-only


def test_explicit_paper_pins_are_resolved_before_semantic_retrieval(
    seeded_kg,
):
    items = gather_grounding(
        "Use paper:2109.11978 for this rough terrain world.",
        store=seeded_kg,
        top_k_techniques=0,
        top_k_failure_modes=0,
        top_k_papers=0,
    )
    assert grounding_ids(items) == ["paper:2109.11978"]
    assert items[0].name == "Rough-terrain curriculum grids"


def test_unresolved_explicit_paper_pin_fails_clearly(seeded_kg):
    with pytest.raises(
        ExplicitGroundingError,
        match=r"paper:9999\.99999.*shared/project KG",
    ):
        gather_grounding(
            "Ground this in paper:9999.99999.",
            store=seeded_kg,
            top_k_techniques=0,
            top_k_failure_modes=0,
            top_k_papers=0,
        )


def test_explicit_paper_pins_dedupe_in_first_mentioned_order(seeded_kg):
    second = Paper(
        id=make_paper_id("2501.00001"),
        arxiv_id="2501.00001",
        title="Second explicit reference",
        abstract="Another grounded method.",
        tags=["terrain"],
        tier="A",
    )
    seeded_kg.add_node(second)
    prompt = (
        "Use paper:2501.00001, then paper:2109.11978; "
        "paper:2501.00001 is intentionally repeated."
    )
    items = gather_grounding(
        prompt,
        store=seeded_kg,
        top_k_techniques=0,
        top_k_failure_modes=0,
        top_k_papers=0,
    )
    assert grounding_ids(items) == [
        "paper:2501.00001",
        "paper:2109.11978",
    ]


def test_no_explicit_pins_preserve_existing_semantic_behavior(
    seeded_kg, const_embedder,
):
    items = gather_grounding(PROMPT, store=seeded_kg)
    assert {item.kind for item in items} == {
        "technique", "failure_mode", "paper",
    }
    assert items[0].kind == "technique"


def test_fail_soft_when_embedder_is_broken(seeded_kg, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("embedding model unavailable")

    monkeypatch.setattr(qmod, "_get_embedder", _boom)
    assert gather_grounding(PROMPT, store=seeded_kg) == ()


def test_empty_store_grounds_nothing_without_embedder(tmp_path, monkeypatch):
    def _boom(*a, **k):  # empty pools must short-circuit before encoding
        raise AssertionError("embedder must not load for an empty store")

    monkeypatch.setattr(qmod, "_get_embedder", _boom)
    store = SculptorKG(tmp_path / "empty.db")
    try:
        assert gather_grounding(PROMPT, store=store) == ()
    finally:
        store.close()


def test_offline_author_records_grounding_ledger():
    ids = ["technique:terrain-difficulty-curriculum", "paper:2109.11978"]
    draft = author_environment(
        PROMPT, robot_capability_id="unitree_g1:base", grounding=ids)
    assert draft.world_spec["meta"]["grounding"] == ids
    assert draft.task_spec["meta"]["grounding"] == ids


def test_model_request_gets_evidence_and_ledger_is_injected():
    items = (
        GroundingItem(node_id="technique:terrain-difficulty-curriculum",
                      kind="technique", name="terrain difficulty curriculum",
                      guidance="promote rows by difficulty", score=0.9),
        GroundingItem(node_id="paper:2109.11978", kind="paper",
                      name="Rough-terrain curriculum grids",
                      guidance="terrain grid reference", score=0.8),
    )
    baseline = author_environment(
        PROMPT, robot_capability_id="unitree_g1:base")

    captured: dict = {}

    class _Model:
        def generate_authoring(self, request):
            captured.update(request)
            world = copy.deepcopy(baseline.world_spec)
            task = copy.deepcopy(baseline.task_spec)
            # A model that drops the ledger entirely must not lose it.
            world["meta"]["grounding"] = []
            task["meta"]["grounding"] = []
            return {
                "world_spec": world,
                "task_spec": task,
                "parameter_provenance":
                    world["meta"]["parameter_provenance"],
            }

    draft = author_environment(
        PROMPT, model=_Model(), robot_capability_id="unitree_g1:base",
        grounding=grounding_ids(items),
        grounding_context=grounding_context(items),
    )
    assert captured["kg_grounding"] == [item.to_dict() for item in items]
    assert draft.world_spec["meta"]["grounding"] == grounding_ids(items)
    assert draft.task_spec["meta"]["grounding"] == grounding_ids(items)


def test_local_grounding_overrides_nonempty_model_grounding_receipts():
    explicit = "paper:2109.11978"
    baseline = author_environment(
        PROMPT, robot_capability_id="unitree_g1:base")

    class _Model:
        def generate_authoring(self, request):
            world = copy.deepcopy(baseline.world_spec)
            task = copy.deepcopy(baseline.task_spec)
            world["meta"]["grounding"] = ["paper:model-substitute"]
            task["meta"]["grounding"] = ["paper:model-substitute"]
            return {
                "world_spec": world,
                "task_spec": task,
                "parameter_provenance": (
                    world["meta"]["parameter_provenance"]
                ),
            }

    draft = author_environment(
        f"{PROMPT}; use {explicit}",
        model=_Model(),
        robot_capability_id="unitree_g1:base",
        grounding=[explicit],
    )
    assert draft.world_spec["meta"]["grounding"] == [explicit]
    assert draft.task_spec["meta"]["grounding"] == [explicit]


def test_cli_grounds_by_default_and_flag_disables(tmp_path, monkeypatch):
    from sculptor.cli import app
    from sculptor.world import grounding as gmod
    from tests.test_world_project import _project

    runner = CliRunner()

    fixed = (GroundingItem(node_id="technique:terrain-difficulty-curriculum",
                           kind="technique",
                           name="terrain difficulty curriculum",
                           guidance="promote rows by difficulty"),)
    calls: list[str] = []

    def _fake_gather(prompt, **kwargs):
        calls.append(prompt)
        return fixed

    monkeypatch.setattr(gmod, "gather_grounding", _fake_gather)
    grounded_project = _project(tmp_path / "grounded")
    grounded = runner.invoke(app, [
        "world", "author", PROMPT,
        "--project", str(grounded_project),
        "--robot", "unitree_g1:base", "--yes", "--json",
    ])
    assert grounded.exit_code == 0, grounded.output
    payload = json.loads(grounded.stdout)
    assert payload["kg_grounding"] == [
        "technique:terrain-difficulty-curriculum"]
    assert calls == [PROMPT]

    def _must_not_run(prompt, **kwargs):
        raise AssertionError("--no-kg-grounding must skip KG retrieval")

    monkeypatch.setattr(gmod, "gather_grounding", _must_not_run)
    offline_project = _project(tmp_path / "offline")
    offline = runner.invoke(app, [
        "world", "author", PROMPT,
        "--project", str(offline_project),
        "--robot", "unitree_g1:base", "--yes", "--json",
        "--no-kg-grounding",
    ])
    assert offline.exit_code == 0, offline.output
    assert json.loads(offline.stdout)["kg_grounding"] == []
