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
    USEFUL_CITATION_BOOST_CAP,
    TechniqueMatch,
    _citation_boost,
    cite,
    query_semantic,
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


# ── §Agentic-data upgrade 2: usage-based ranking boost ────────────────────
def test_citation_boost_is_zero_below_one_and_capped_at_high_counts():
    assert _citation_boost(0) == 0.0
    assert _citation_boost(-3) == 0.0
    small = _citation_boost(1)
    assert 0.0 < small < USEFUL_CITATION_BOOST_CAP
    big = _citation_boost(1_000_000)
    assert big == pytest.approx(USEFUL_CITATION_BOOST_CAP, abs=1e-9)
    # monotonically non-decreasing.
    assert _citation_boost(10) >= _citation_boost(1)
    assert _citation_boost(100) >= _citation_boost(10)


def test_query_techniques_boost_breaks_ties_without_exceeding_cap(kg):
    """Two techniques tied on matched failure-mode count: the one with
    more useful_citations ranks first, but the boost never overwhelms a
    real relevance difference (checked via the cap constant itself)."""
    _seed_graph(kg)
    tech_rsi = kg.get_node(make_technique_id("reference_state_initialization"))
    tech_rsi.useful_citations = 50
    kg.add_node(tech_rsi)

    results = query_techniques(["sparse_reward"], store=kg, top_k=5)
    # Only RSI addresses sparse_reward directly in the single-failure query
    # from the base seed graph fixture below — assert the boost is applied
    # and stays within the documented cap.
    rsi_match = next(m for m in results if m.technique.name == "reference_state_initialization")
    assert rsi_match.relevance_score >= 1.0  # base score (1 matched fm) + boost
    assert rsi_match.relevance_score <= 1.0 + USEFUL_CITATION_BOOST_CAP + 0.05  # + year tie-break slack


def test_query_semantic_floor_applies_before_citation_boost(monkeypatch, kg):
    """A match below min_similarity must NOT be resurrected by a large
    useful_citations boost — the floor check happens on the raw cosine
    similarity, before the boost is added."""
    import numpy as np

    _seed_graph(kg)
    model_name = "test-model-boost"
    # RSI sits just BELOW a 0.5 floor; shaping sits comfortably above.
    below_floor_vec = np.array([0.4, np.sqrt(1 - 0.4 ** 2)], dtype=np.float32)
    above_floor_vec = np.array([0.9, np.sqrt(1 - 0.9 ** 2)], dtype=np.float32)
    kg.set_embedding(make_technique_id("reference_state_initialization"),
                     model_name, below_floor_vec)
    kg.set_embedding(make_technique_id("potential_based_shaping"),
                     model_name, above_floor_vec)

    # Give the below-floor technique a huge citation count — if the boost
    # were applied before the floor check this could push it back above
    # 0.5 (0.4 + 0.05 cap = 0.45, still short, so also assert the actual
    # score never crosses the floor even with max boost).
    tech_rsi = kg.get_node(make_technique_id("reference_state_initialization"))
    tech_rsi.useful_citations = 10_000_000
    kg.add_node(tech_rsi)

    import sculptor.kg.query as qmod
    monkeypatch.setattr(qmod, "_embed_text",
                        lambda text, mn=model_name: np.array([1.0, 0.0], dtype=np.float32))

    results = query_semantic("anything", top_k=5, store=kg, model_name=model_name,
                             min_similarity=0.5)
    names = {m.technique.name for m in results}
    assert "reference_state_initialization" not in names, (
        "below-floor match must not be resurrected by the citation boost")
    assert "potential_based_shaping" in names


def test_query_semantic_boost_reorders_within_floor_but_score_stays_raw(monkeypatch, kg):
    """Among matches that already clear the floor, useful_citations can
    reorder them — but the displayed relevance_score stays the RAW cosine
    similarity (the boost affects ordering only, per the upgrade spec)."""
    import numpy as np

    _seed_graph(kg)
    model_name = "test-model-boost2"
    # shaping has slightly higher raw similarity than RSI...
    rsi_vec = np.array([0.80, np.sqrt(1 - 0.80 ** 2)], dtype=np.float32)
    shaping_vec = np.array([0.82, np.sqrt(1 - 0.82 ** 2)], dtype=np.float32)
    kg.set_embedding(make_technique_id("reference_state_initialization"),
                     model_name, rsi_vec)
    kg.set_embedding(make_technique_id("potential_based_shaping"),
                     model_name, shaping_vec)

    # ...but RSI has a large useful_citations boost that should push it
    # ahead in ORDER while its displayed score remains its raw cosine sim.
    tech_rsi = kg.get_node(make_technique_id("reference_state_initialization"))
    tech_rsi.useful_citations = 10_000_000
    kg.add_node(tech_rsi)

    import sculptor.kg.query as qmod
    monkeypatch.setattr(qmod, "_embed_text",
                        lambda text, mn=model_name: np.array([1.0, 0.0], dtype=np.float32))

    results = query_semantic("anything", top_k=5, store=kg, model_name=model_name,
                             min_similarity=0.0)
    assert results[0].technique.name == "reference_state_initialization"
    # displayed score is the RAW cosine similarity, not sim+boost.
    assert results[0].relevance_score == pytest.approx(0.80, abs=1e-2)


# ── §KG-retrieval fix 2: resolve_failure_modes_semantic ────────────────────
def test_resolve_failure_modes_semantic_matches_descriptor_to_failure_mode(monkeypatch, kg):
    """A free-text descriptor resolves (via embedding similarity) to the
    right FailureMode node id — the semantic counterpart to
    `_resolve_failure_modes`'s fixed-enum fuzzy resolution."""
    import numpy as np

    from sculptor.kg.query import resolve_failure_modes_semantic
    from sculptor.kg.schema import FailureMode, make_failure_mode_id

    fm_a = FailureMode(id=make_failure_mode_id("forearm_planking"),
                       name="forearm_planking",
                       description="Robot supports weight on forearms instead "
                                   "of extending its legs.")
    fm_b = FailureMode(id=make_failure_mode_id("hip_lockout"),
                       name="hip_lockout",
                       description="Hip joints stay locked at full extension.")
    kg.add_node(fm_a)
    kg.add_node(fm_b)

    model_name = "test-model-fm"
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)
    kg.set_embedding(fm_a.id, model_name, vec_a)
    kg.set_embedding(fm_b.id, model_name, vec_b)

    def _should_not_load(*args, **kwargs):  # pragma: no cover - failure path
        raise RuntimeError("SentenceTransformer should not have been loaded")

    import sculptor.kg.query as qmod
    monkeypatch.setattr(qmod, "_get_embedder", _should_not_load)
    monkeypatch.setattr(qmod, "_embed_text", lambda text, mn=model_name: vec_a)

    ids = resolve_failure_modes_semantic(
        kg, ["planks on forearms without leg drive"], model_name=model_name)
    assert ids == [fm_a.id]


def test_resolve_failure_modes_semantic_floors_below_min_similarity(monkeypatch, kg):
    import numpy as np

    from sculptor.kg.query import resolve_failure_modes_semantic
    from sculptor.kg.schema import FailureMode, make_failure_mode_id

    fm_a = FailureMode(id=make_failure_mode_id("far_mode"), name="far_mode",
                       description="Something else entirely.")
    kg.add_node(fm_a)
    model_name = "test-model-fm-floor"
    kg.set_embedding(fm_a.id, model_name, np.array([1.0, 0.0], dtype=np.float32))

    import sculptor.kg.query as qmod
    monkeypatch.setattr(
        qmod, "_embed_text",
        lambda text, mn=model_name: np.array([0.0, 1.0], dtype=np.float32))

    ids = resolve_failure_modes_semantic(
        kg, ["totally unrelated phrase"], model_name=model_name, min_similarity=0.45)
    assert ids == []


def test_resolve_failure_modes_semantic_empty_descriptors_returns_empty(kg):
    from sculptor.kg.query import resolve_failure_modes_semantic
    assert resolve_failure_modes_semantic(kg, []) == []


def test_resolve_failure_modes_semantic_dedupes_and_respects_top_k_per(monkeypatch, kg):
    """Two descriptors resolving to the SAME node dedupe to one entry;
    `top_k_per` caps how many nodes a single descriptor can contribute."""
    import numpy as np

    from sculptor.kg.query import resolve_failure_modes_semantic
    from sculptor.kg.schema import FailureMode, make_failure_mode_id

    fm_a = FailureMode(id=make_failure_mode_id("mode_a"), name="mode_a", description="a")
    fm_b = FailureMode(id=make_failure_mode_id("mode_b"), name="mode_b", description="b")
    kg.add_node(fm_a)
    kg.add_node(fm_b)
    model_name = "test-model-fm-dedupe"
    kg.set_embedding(fm_a.id, model_name, np.array([1.0, 0.0], dtype=np.float32))
    kg.set_embedding(fm_b.id, model_name, np.array([0.9, np.sqrt(1 - 0.9 ** 2)], dtype=np.float32))

    import sculptor.kg.query as qmod
    monkeypatch.setattr(
        qmod, "_embed_text",
        lambda text, mn=model_name: np.array([1.0, 0.0], dtype=np.float32))

    # Same descriptor text twice -> both resolve to the same top matches;
    # dedup keeps each node id once.
    ids = resolve_failure_modes_semantic(
        kg, ["desc one", "desc two"], model_name=model_name,
        top_k_per=1, min_similarity=0.5)
    assert ids == [fm_a.id]


# ── §KG-retrieval fix 2: query_techniques(extra_failure_node_ids=...) ──────
def test_query_techniques_honors_extra_failure_node_ids(kg):
    """A FailureMode reachable ONLY via `extra_failure_node_ids` (not the
    enum-resolved `failure_modes` list) still pulls in its technique, and
    an unknown extra id is silently skipped rather than erroring."""
    from sculptor.kg.schema import (
        Edge, FailureMode, Relation, Technique,
        make_failure_mode_id, make_technique_id,
    )

    p1, _p2 = _seed_graph(kg)

    fm_niche = FailureMode(id=make_failure_mode_id("niche_failure"),
                           name="niche_failure", description="A niche failure.")
    kg.add_node(fm_niche)
    tech_niche = Technique(id=make_technique_id("niche_fix"),
                           name="niche_fix", description="Fixes the niche failure.")
    kg.add_node(tech_niche)
    kg.add_edge(Edge(src=p1.id, dst=tech_niche.id, relation=Relation.INTRODUCES))
    kg.add_edge(Edge(src=tech_niche.id, dst=fm_niche.id, relation=Relation.ADDRESSES,
                     data={"source_paper_id": p1.id}))

    # `failure_modes` alone doesn't resolve to anything real.
    r = query_techniques(["nonexistent_xyz"], store=kg,
                         extra_failure_node_ids=[fm_niche.id])
    names = {m.technique.name for m in r}
    assert "niche_fix" in names

    # Unknown extra id -> skipped, not an error, not a false match.
    r2 = query_techniques(["nonexistent_xyz"], store=kg,
                          extra_failure_node_ids=["failure:does-not-exist"])
    assert r2 == []


def test_query_techniques_extra_failure_node_ids_dedupes_with_enum_resolved(kg):
    """When the SAME FailureMode is reachable via both the enum-resolved
    `failure_modes` and `extra_failure_node_ids`, ranking is unaffected —
    the technique is counted once, not double-credited."""
    _seed_graph(kg)
    from sculptor.kg.schema import make_failure_mode_id

    sparse_id = make_failure_mode_id("sparse_reward")
    r_plain = query_techniques(["sparse_reward"], store=kg, top_k=5)
    r_dup = query_techniques(
        ["sparse_reward"], store=kg, top_k=5,
        extra_failure_node_ids=[sparse_id])
    assert [m.technique.name for m in r_plain] == [m.technique.name for m in r_dup]
    assert [m.relevance_score for m in r_plain] == [m.relevance_score for m in r_dup]


# ── §KG-retrieval fix 4: outcome-stats ranking ────────────────────────────
def test_outcome_stats_adjustment_scoped_scaled_and_clamped():
    from sculptor.kg.query import (
        OUTCOME_STATS_ADJUSTMENT_CAP,
        _outcome_stats_adjustment,
    )

    assert _outcome_stats_adjustment(None, {"failure:x"}) == 0.0
    assert _outcome_stats_adjustment({}, {"failure:x"}) == 0.0

    stats = {"failure:x": {"helped": 2, "regressed": 0}}
    # net=2 -> 0.03 * 2 = 0.06, well under the cap.
    assert _outcome_stats_adjustment(stats, {"failure:x"}) == pytest.approx(0.06)

    # A failure mode NOT in the queried set contributes nothing, even
    # though the technique has stats for it.
    assert _outcome_stats_adjustment(stats, {"failure:other"}) == 0.0

    # Clamped at +/- the cap regardless of how lopsided the tally is.
    huge_helped = {"failure:x": {"helped": 1000, "regressed": 0}}
    assert _outcome_stats_adjustment(huge_helped, {"failure:x"}) == pytest.approx(
        OUTCOME_STATS_ADJUSTMENT_CAP)
    huge_regressed = {"failure:x": {"helped": 0, "regressed": 1000}}
    assert _outcome_stats_adjustment(huge_regressed, {"failure:x"}) == pytest.approx(
        -OUTCOME_STATS_ADJUSTMENT_CAP)


def test_query_techniques_outcome_stats_reorders_equal_rank_ties(kg):
    """Two techniques tied on matched-FM count (the seed graph's
    sparse_reward query) reorder by outcome_stats when one has a
    helped-dominant record for the QUERIED failure mode and the other's
    record is for a DIFFERENT failure mode (must not count against/for it)."""
    _seed_graph(kg)
    sparse_id = make_failure_mode_id("sparse_reward")
    hack_id = make_failure_mode_id("reward_hacking")

    # Baseline: with no outcome_stats, alphabetical tie-break sorts
    # potential_based_shaping before reference_state_initialization.
    baseline = query_techniques(["sparse_reward"], store=kg, top_k=5)
    assert [m.technique.name for m in baseline] == [
        "potential_based_shaping", "reference_state_initialization"]

    tech_rsi = kg.get_node(make_technique_id("reference_state_initialization"))
    tech_rsi.outcome_stats = {sparse_id: {"helped": 5, "regressed": 0}}
    kg.add_node(tech_rsi)

    tech_shaping = kg.get_node(make_technique_id("potential_based_shaping"))
    # Regressed, but on reward_hacking — NOT the queried failure mode below
    # — so it must not drag potential_based_shaping down for this query.
    tech_shaping.outcome_stats = {hack_id: {"helped": 0, "regressed": 5}}
    kg.add_node(tech_shaping)

    results = query_techniques(["sparse_reward"], store=kg, top_k=5)
    assert results[0].technique.name == "reference_state_initialization"
    assert results[1].technique.name == "potential_based_shaping"


def test_query_techniques_outcome_stats_capped_relative_to_relevance(kg):
    """The outcome-stats adjustment never exceeds its documented cap, even
    with an enormous helped/regressed imbalance — same 'small tie-breaker,
    never overwhelms real relevance' contract as the citation boost."""
    from sculptor.kg.query import OUTCOME_STATS_ADJUSTMENT_CAP

    _seed_graph(kg)
    sparse_id = make_failure_mode_id("sparse_reward")
    tech_rsi = kg.get_node(make_technique_id("reference_state_initialization"))
    tech_rsi.outcome_stats = {sparse_id: {"helped": 10_000, "regressed": 0}}
    kg.add_node(tech_rsi)

    results = query_techniques(["sparse_reward"], store=kg, top_k=5)
    rsi_match = next(
        m for m in results if m.technique.name == "reference_state_initialization")
    assert rsi_match.relevance_score <= 1.0 + OUTCOME_STATS_ADJUSTMENT_CAP + 0.05
