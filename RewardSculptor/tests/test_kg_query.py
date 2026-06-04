"""tests/test_kg_query.py — offline tests for extract materialization and query.

No LLM calls, no network, no sentence-transformers download. We hand-build
nodes and edges to exercise the graph-walk query; the LLM path is tested
separately via mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.kg.extract import (
    EnvironmentExtract,
    ExtractionPayload,
    FailureModeExtract,
    PaperToEnvironmentRel,
    PaperToTechniqueRel,
    RewardComponentExtract,
    TechniqueExtract,
    TechniqueToFailureModeRel,
    TechniqueToRewardComponentRel,
    _materialize,
    extract_entities,
)
from sculptor.kg.query import (
    TechniqueMatch,
    cite,
    query_techniques,
)
from sculptor.kg.schema import (
    Edge,
    Environment,
    FailureMode,
    Paper,
    Relation,
    RewardComponent,
    Technique,
    make_environment_id,
    make_failure_mode_id,
    make_paper_id,
    make_reward_component_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG


@pytest.fixture
def kg(tmp_path: Path):
    store = SculptorKG(tmp_path / "kg.db")
    yield store
    store.close()


# ── Materialization ─────────────────────────────────────────────────────────
def _good_ev(label: str) -> str:
    # padded above MIN_EVIDENCE_CHARS (20)
    return f"Verbatim snippet about {label} from the source paper excerpt."


def test_materialize_builds_all_node_kinds_and_edges(kg):
    paper = Paper(
        id=make_paper_id("2024.00001"), arxiv_id="2024.00001",
        title="Test Paper", authors=["Alice"], year=2024,
    )
    kg.add_node(paper)

    payload = ExtractionPayload(
        techniques=[
            TechniqueExtract(
                name="reference_state_initialization",
                description="Start episodes from states sampled along a reference.",
                tags=["curriculum", "exploration"],
                evidence=_good_ev("RSI"),
            ),
            TechniqueExtract(
                name="ctrl_cost_penalty",
                description="Penalize action magnitude to discourage bang-bang.",
                evidence=_good_ev("ctrl penalty"),
            ),
        ],
        failure_modes=[
            FailureModeExtract(
                name="sparse_reward_plateau",
                description="Agent plateaus when rewards are sparse.",
                symptoms=["flat training curve"],
                environment_tag="continuous_locomotion",
                evidence=_good_ev("sparse reward"),
            ),
        ],
        reward_components=[
            RewardComponentExtract(
                name="ctrl_cost",
                description="Sum of squared actions",
                formula="-w * ||a||^2",
                hyperparameters={"w": 0.001},
                evidence=_good_ev("ctrl cost"),
            ),
        ],
        environments=[
            EnvironmentExtract(
                name="Hopper-v4",
                description="MuJoCo Hopper via Gymnasium",
                tags=["mujoco", "continuous_locomotion"],
                evidence=_good_ev("Hopper"),
            ),
        ],
        paper_to_technique=[
            PaperToTechniqueRel(technique="reference_state_initialization"),
        ],
        technique_to_failure_mode=[
            TechniqueToFailureModeRel(
                technique="reference_state_initialization",
                failure_mode="sparse_reward_plateau"),
        ],
        technique_to_reward_component=[
            TechniqueToRewardComponentRel(
                technique="ctrl_cost_penalty",
                reward_component="ctrl_cost"),
        ],
        paper_to_environment=[
            PaperToEnvironmentRel(environment="Hopper-v4"),
        ],
    )
    node_ids, edge_count = _materialize(kg, paper, payload)

    # nodes
    assert make_technique_id("reference_state_initialization") in node_ids
    assert make_failure_mode_id("sparse_reward_plateau") in node_ids
    assert make_reward_component_id("ctrl_cost") in node_ids
    assert make_environment_id("Hopper-v4") in node_ids
    assert kg.count_nodes(Technique.kind) == 2
    assert kg.count_nodes(FailureMode.kind) == 1
    assert kg.count_nodes(RewardComponent.kind) == 1
    assert kg.count_nodes(Environment.kind) == 1

    # edges
    assert edge_count == 4
    assert kg.count_edges(Relation.INTRODUCES) == 1
    assert kg.count_edges(Relation.ADDRESSES) == 1
    assert kg.count_edges(Relation.USES) == 1
    assert kg.count_edges(Relation.EVALUATES_ON) == 1

    # edge evidence was carried through
    addresses_edges = kg.neighbors(
        make_technique_id("reference_state_initialization"),
        relation=Relation.ADDRESSES, direction="out")
    assert len(addresses_edges) == 1
    edge, _ = addresses_edges[0]
    assert "sparse reward" in edge.data.get("evidence", "")


def test_materialize_drops_entities_with_thin_evidence(kg):
    paper = Paper(id=make_paper_id("2024.00002"), arxiv_id="2024.00002", title="Thin")
    kg.add_node(paper)
    payload = ExtractionPayload(
        techniques=[
            TechniqueExtract(name="good_tech", description="d", evidence=_good_ev("good")),
            TechniqueExtract(name="empty_tech", description="d", evidence=""),
            TechniqueExtract(name="short_tech", description="d", evidence="too short"),
        ],
    )
    node_ids, edge_count = _materialize(kg, paper, payload)
    assert len(node_ids) == 1
    assert kg.count_nodes(Technique.kind) == 1
    assert kg.get_node(make_technique_id("good_tech")) is not None
    assert kg.get_node(make_technique_id("empty_tech")) is None
    assert kg.get_node(make_technique_id("short_tech")) is None


def test_extract_entities_is_idempotent_on_extracted_flag(kg):
    paper = Paper(
        id=make_paper_id("2024.00003"), arxiv_id="2024.00003", title="Extracted already",
        extracted=True,
    )
    kg.add_node(paper)
    res = extract_entities(paper, store=kg)
    assert res.skipped is True
    # No LLM call was made (we didn't pass a client)


# ── query_techniques ────────────────────────────────────────────────────────
def _seed_graph(kg):
    """Build a small graph: 2 papers, 2 techniques, 2 failures, 2 envs."""
    p1 = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347",
               title="PPO", authors=["Schulman"], year=2017)
    p2 = Paper(id=make_paper_id("2020.00001"), arxiv_id="2020.00001",
               title="Reward Shaping Survey", authors=["Author"], year=2020)
    kg.add_node(p1); kg.add_node(p2)

    t_rsi = Technique(id=make_technique_id("reference_state_initialization"),
                      name="reference_state_initialization",
                      description="Reference-State Initialization — start episodes "
                      "sampled from a reference trajectory to ease exploration in "
                      "sparse-reward locomotion tasks.")
    t_shaping = Technique(id=make_technique_id("potential_based_shaping"),
                          name="potential_based_shaping",
                          description="Potential-based reward shaping that preserves "
                          "optimal policy while densifying feedback.")
    kg.add_node(t_rsi); kg.add_node(t_shaping)

    f_sparse = FailureMode(id=make_failure_mode_id("sparse_reward"),
                           name="sparse_reward",
                           description="Training plateaus when rewards are sparse.")
    f_hack = FailureMode(id=make_failure_mode_id("reward_hacking"),
                         name="reward_hacking",
                         description="Agent exploits a poorly specified reward.")
    kg.add_node(f_sparse); kg.add_node(f_hack)

    e_hop = Environment(id=make_environment_id("Hopper-v4"), name="Hopper-v4",
                        description="MuJoCo Hopper.", tags=["continuous_locomotion", "mujoco"])
    e_atari = Environment(id=make_environment_id("Atari"), name="Atari",
                          description="Atari 2600.", tags=["discrete_control"])
    kg.add_node(e_hop); kg.add_node(e_atari)

    # Relations
    kg.add_edge(Edge(src=p1.id, dst=t_rsi.id, relation=Relation.INTRODUCES,
                     data={"evidence": "p1 introduces RSI"}))
    kg.add_edge(Edge(src=p2.id, dst=t_shaping.id, relation=Relation.INTRODUCES,
                     data={"evidence": "p2 introduces shaping"}))
    kg.add_edge(Edge(src=t_rsi.id, dst=f_sparse.id, relation=Relation.ADDRESSES,
                     data={"evidence": "RSI helps when rewards are sparse.",
                           "source_paper_id": p1.id}))
    kg.add_edge(Edge(src=t_shaping.id, dst=f_sparse.id, relation=Relation.ADDRESSES,
                     data={"evidence": "Potential-based shaping densifies sparse signal.",
                           "source_paper_id": p2.id}))
    kg.add_edge(Edge(src=t_shaping.id, dst=f_hack.id, relation=Relation.ADDRESSES,
                     data={"evidence": "Potential-based shaping cannot introduce reward hacking.",
                           "source_paper_id": p2.id}))
    kg.add_edge(Edge(src=p1.id, dst=e_hop.id, relation=Relation.EVALUATES_ON,
                     data={"evidence": "p1 on Hopper"}))
    kg.add_edge(Edge(src=p2.id, dst=e_atari.id, relation=Relation.EVALUATES_ON,
                     data={"evidence": "p2 on Atari"}))
    return p1, p2


def test_query_techniques_multi_failure_ranking(kg):
    _seed_graph(kg)
    results = query_techniques(["sparse_reward", "reward_hacking"], store=kg, top_k=5)
    assert len(results) == 2
    # potential_based_shaping matches BOTH failures → ranks first.
    assert results[0].technique.name == "potential_based_shaping"
    assert set(results[0].matched_on) == {"sparse_reward", "reward_hacking"}
    assert results[1].technique.name == "reference_state_initialization"
    assert results[1].matched_on == ["sparse_reward"]
    assert "PPO" in results[1].paper_citation or "Schulman" in results[1].paper_citation


def test_query_techniques_domain_filter(kg):
    _seed_graph(kg)
    # Only RSI's paper EVALUATES_ON Hopper-v4 → only it survives the locomotion filter.
    loc = query_techniques(
        ["sparse_reward"], domain_filter="continuous_locomotion", store=kg)
    assert [m.technique.name for m in loc] == ["reference_state_initialization"]

    # Atari tag → only shaping's paper.
    dc = query_techniques(
        ["sparse_reward"], domain_filter="discrete_control", store=kg)
    assert [m.technique.name for m in dc] == ["potential_based_shaping"]


def test_query_techniques_unknown_failure_returns_empty(kg):
    _seed_graph(kg)
    assert query_techniques(["nonexistent_failure_xyz"], store=kg) == []


def test_query_techniques_resolves_fuzzy_failure_slug(kg):
    _seed_graph(kg)
    # Underscore/space/case tolerance — the caller might pass human prose.
    r = query_techniques(["Sparse Reward"], store=kg)
    names = {m.technique.name for m in r}
    assert "reference_state_initialization" in names


def test_cite_formats_paper_with_and_without_authors(kg):
    _seed_graph(kg)
    s = cite("1707.06347", store=kg)
    assert "Schulman" in s
    assert "2017" in s
    assert "arXiv:1707.06347" in s

    # Paper without authors
    orphan = Paper(id=make_paper_id("9999.99999"), arxiv_id="9999.99999",
                   title="Orphan", authors=[], year=None)
    kg.add_node(orphan)
    s2 = cite("9999.99999", store=kg)
    assert "Unknown" in s2
    assert "arXiv:9999.99999" in s2


def test_query_semantic_uses_cached_embeddings(monkeypatch, kg):
    """Semantic search should use the cached embeddings and NEVER hit the real
    SentenceTransformer model when all embeddings are pre-cached."""
    import numpy as np

    _seed_graph(kg)

    # Pre-cache deterministic embeddings so we don't need to load the model.
    model_name = "test-model"
    rsi_vec = np.array([1.0, 0.0], dtype=np.float32)
    shaping_vec = np.array([0.0, 1.0], dtype=np.float32)
    kg.set_embedding(make_technique_id("reference_state_initialization"), model_name, rsi_vec)
    kg.set_embedding(make_technique_id("potential_based_shaping"), model_name, shaping_vec)

    # Fail loudly if the real embedder is called.
    def _should_not_load(*args, **kwargs):  # pragma: no cover - failure path
        raise RuntimeError("SentenceTransformer should not have been loaded")

    import sculptor.kg.query as qmod
    monkeypatch.setattr(qmod, "_get_embedder", _should_not_load)
    monkeypatch.setattr(qmod, "_embed_text", lambda text, mn=model_name: rsi_vec)

    from sculptor.kg.query import query_semantic
    results = query_semantic("anything", top_k=2, store=kg, model_name=model_name)
    assert len(results) == 2
    assert results[0].technique.name == "reference_state_initialization"
    assert results[0].relevance_score == pytest.approx(1.0, abs=1e-5)
    assert results[1].technique.name == "potential_based_shaping"
    assert results[1].relevance_score == pytest.approx(0.0, abs=1e-5)


def test_get_embedder_is_thread_safe_under_concurrent_load(monkeypatch):
    """Pins the fix for the 2026-04-23 KG-preview hang regression.

    Symptom: Ship-10's startup-time `_prewarm_embedding_model` hook
    fires `asyncio.to_thread(_load_embedder)` concurrently with the
    reward-prompt worker's own `_get_embedder` call. Pre-fix,
    `_get_embedder` had no lock, so two threads entered
    `SentenceTransformer(...)` simultaneously. On WSL2 this deadlocked
    on the huggingface_hub cache FileLock + torch CUDA init, producing
    a 5-minute hang at the KG preview step.

    Post-fix, `_get_embedder` uses double-checked locking so the second
    caller waits for the first to populate the cache, then reads it.

    Test construction: swap `SentenceTransformer` for a stub that sleeps
    briefly on init (simulates the cold load). Fire N threads at once,
    assert only ONE init happened.
    """
    import threading
    import time

    import sculptor.kg.query as qmod

    # Reset cache — previous tests may have populated it.
    monkeypatch.setattr(qmod, "_EMBEDDER_CACHE", {})

    init_count = [0]
    init_lock = threading.Lock()

    class _FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            # Accept `local_files_only=True` (added for the 2026-04-23
            # WSL2 httpx-bypass fix) and any other forwarded kwargs.
            with init_lock:
                init_count[0] += 1
            # Simulate the cold-load window where the race condition
            # used to trigger. 200 ms is plenty for 8 threads to race.
            time.sleep(0.2)
            self.model_name = model_name

        def encode(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("not used in this test")

    # Intercept the lazy import inside `_get_embedder`.
    import sys
    fake_mod = type(sys)("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    results: list = [None] * 8
    threads: list[threading.Thread] = []

    def _target(i: int) -> None:
        results[i] = qmod._get_embedder("thread-safety-test-model")

    for i in range(8):
        t = threading.Thread(target=_target, args=(i,))
        threads.append(t)
    for t in threads:
        t.start()
    # Bounded join — if the lock deadlocks we'd hang forever; 5 s is
    # way more than 8 × 200 ms worth of serialized loads.
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "thread deadlocked — _get_embedder lock broken"

    assert init_count[0] == 1, (
        f"expected exactly 1 SentenceTransformer init under concurrent "
        f"load, got {init_count[0]} — lock is not serializing"
    )
    # All threads see the same cached instance.
    assert all(r is results[0] for r in results)
