"""tests/test_kg.py — offline tests for schema + store.

No network. The ingest path is exercised by
`uv run python -m sculptor.kg.ingest ...`, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.kg.schema import (
    Edge,
    Environment,
    FailureMode,
    Paper,
    Relation,
    RewardComponent,
    Result,
    Technique,
    make_environment_id,
    make_failure_mode_id,
    make_paper_id,
    make_reward_component_id,
    make_result_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG


@pytest.fixture
def kg(tmp_path: Path):
    store = SculptorKG(tmp_path / "kg.db")
    yield store
    store.close()


def test_node_roundtrip_all_kinds(kg):
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347",
                  title="PPO", authors=["Schulman"], year=2017,
                  abstract="abs", conclusion_text="we showed that...")
    tech = Technique(id=make_technique_id("RAMIEL r_keep"),
                     name="RAMIEL r_keep",
                     description="horizontal velocity penalty",
                     tags=["locomotion", "regularization"])
    fm = FailureMode(id=make_failure_mode_id("walking not jumping"),
                     name="walking_not_jumping",
                     description="agent walks instead of jumping",
                     symptoms=["nonzero xy velocity during launch"],
                     environment_tag="continuous_locomotion")
    rc = RewardComponent(id=make_reward_component_id("ctrl_cost"),
                          name="ctrl_cost",
                          description="sum of squared actions",
                          formula="-w * ||a||^2",
                          hyperparameters={"w": 0.001})
    env = Environment(id=make_environment_id("Hopper-v4"), name="Hopper-v4",
                      description="gymnasium hopper", tags=["mujoco", "locomotion"])
    res = Result(id=make_result_id("paper:1707.06347", "mean_return", "environment:hopper-v4"),
                 paper_id="paper:1707.06347", metric_name="mean_return",
                 value=3200.0, environment_id="environment:hopper-v4",
                 notes="reference number")

    for node in (paper, tech, fm, rc, env, res):
        kg.add_node(node)
        fetched = kg.get_node(node.id)
        assert fetched is not None
        assert type(fetched) is type(node)
        assert fetched == node


def test_add_node_is_upsert(kg):
    a = Paper(id=make_paper_id("x1"), arxiv_id="x1", title="Old")
    b = Paper(id=make_paper_id("x1"), arxiv_id="x1", title="New")
    kg.add_node(a)
    kg.add_node(b)
    got = kg.get_node(a.id)
    assert got.title == "New"
    assert kg.count_nodes(Paper.kind) == 1


def test_find_nodes_by_kind(kg):
    kg.add_node(Paper(id=make_paper_id("a"), arxiv_id="a", title="A"))
    kg.add_node(Paper(id=make_paper_id("b"), arxiv_id="b", title="B"))
    kg.add_node(Technique(id=make_technique_id("t"), name="t"))
    papers = kg.find_nodes(kind=Paper.kind)
    techs = kg.find_nodes(kind=Technique.kind)
    assert {p.arxiv_id for p in papers} == {"a", "b"}
    assert [t.name for t in techs] == ["t"]


def test_find_nodes_filter(kg):
    kg.add_node(Paper(id=make_paper_id("a"), arxiv_id="a", title="A", year=2017))
    kg.add_node(Paper(id=make_paper_id("b"), arxiv_id="b", title="B", year=2020))
    kg.add_node(Paper(id=make_paper_id("c"), arxiv_id="c", title="C",
                      year=2020, extracted=True))
    y2020 = kg.find_nodes(kind=Paper.kind, year=2020)
    assert {p.arxiv_id for p in y2020} == {"b", "c"}
    extracted = kg.find_nodes(kind=Paper.kind, extracted=True)
    assert [p.arxiv_id for p in extracted] == ["c"]


def test_edges_roundtrip_and_neighbors(kg):
    kg.add_node(Paper(id=make_paper_id("a"), arxiv_id="a", title="A"))
    kg.add_node(Paper(id=make_paper_id("b"), arxiv_id="b", title="B"))
    kg.add_node(Technique(id=make_technique_id("t1"), name="t1"))

    e1 = Edge(src=make_paper_id("a"), dst=make_paper_id("b"),
              relation=Relation.CITES, data={"snippet": "as shown in [b]"})
    e2 = Edge(src=make_paper_id("a"), dst=make_technique_id("t1"),
              relation=Relation.INTRODUCES)
    kg.add_edge(e1)
    kg.add_edge(e2)

    out_all = kg.neighbors(make_paper_id("a"))
    assert len(out_all) == 2
    citations = kg.neighbors(make_paper_id("a"), relation=Relation.CITES)
    assert len(citations) == 1
    edge, other = citations[0]
    assert edge.data["snippet"].startswith("as shown")
    assert other == make_paper_id("b")

    in_b = kg.neighbors(make_paper_id("b"), direction="in")
    assert len(in_b) == 1
    assert in_b[0][1] == make_paper_id("a")


def test_delete_node_cascades_edges(kg):
    kg.add_node(Paper(id=make_paper_id("a"), arxiv_id="a", title="A"))
    kg.add_node(Paper(id=make_paper_id("b"), arxiv_id="b", title="B"))
    kg.add_edge(Edge(src=make_paper_id("a"), dst=make_paper_id("b"),
                     relation=Relation.CITES))
    assert kg.count_edges() == 1
    assert kg.delete_node(make_paper_id("a"))
    assert kg.get_node(make_paper_id("a")) is None
    assert kg.count_edges() == 0  # cascade


def test_stats_shape(kg):
    kg.add_node(Paper(id=make_paper_id("a"), arxiv_id="a", title="A"))
    kg.add_node(Technique(id=make_technique_id("t"), name="t"))
    kg.add_edge(Edge(src=make_paper_id("a"), dst=make_technique_id("t"),
                     relation=Relation.INTRODUCES))
    s = kg.stats()
    assert s["total_nodes"] == 2
    assert s["total_edges"] == 1
    assert s["nodes_by_kind"] == {"Paper": 1, "Technique": 1}
    assert s["edges_by_relation"] == {"INTRODUCES": 1}


def test_ingest_conclusion_extraction():
    """Unit-test the heuristic: no network, no arxiv."""
    from sculptor.kg.ingest import _extract_conclusion

    text = (
        "1 Introduction\nLots of text.\n\n"
        "2 Method\nDetails.\n\n"
        "3 Results\nTables.\n\n"
        "4 Conclusion\n"
        "In this paper we presented PPO, an on-policy algorithm...\n"
        "We demonstrated strong performance on Mujoco.\n\n"
        "References\n[1] ..."
    )
    conc = _extract_conclusion(text)
    assert "presented PPO" in conc
    assert "References" not in conc


def test_ingest_normalize_arxiv_id():
    from sculptor.kg.ingest import _normalize_arxiv_id

    assert _normalize_arxiv_id("1707.06347") == "1707.06347"
    assert _normalize_arxiv_id("1707.06347v2") == "1707.06347"
    assert _normalize_arxiv_id("arxiv:1707.06347") == "1707.06347"
    assert _normalize_arxiv_id("https://arxiv.org/abs/1707.06347v3") == "1707.06347"


# ── M7 Phase 1: shared default DB path ────────────────────────────────────
def test_default_db_path_respects_sculptor_env(tmp_path, monkeypatch):
    """SCULPTOR_KG_PATH overrides the default resolution."""
    from sculptor.kg.store import default_db_path

    target = tmp_path / "override.db"
    monkeypatch.setenv("SCULPTOR_KG_PATH", str(target))
    monkeypatch.delenv("RS_KG_PATH", raising=False)

    assert default_db_path() == target.resolve()


def test_default_db_path_respects_backend_alias(tmp_path, monkeypatch):
    """RS_KG_PATH (backend-facing alias) overrides the default when set."""
    from sculptor.kg.store import default_db_path

    target = tmp_path / "rs_override.db"
    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)
    monkeypatch.setenv("RS_KG_PATH", str(target))

    assert default_db_path() == target.resolve()


def test_default_db_path_ignores_legacy_cwd_db(tmp_path, monkeypatch, capsys):
    """2026-07-03: a cwd-relative kg/graph.db is NO LONGER honored — the
    back-compat preference fragmented the graph by launch directory (the
    tuck-jump E2E diagnosed against a 6-technique stub while the shared
    graph held 493 techniques). The shared path always wins; the stray
    file just triggers a stderr pointer at `sculpt kg merge`."""
    from sculptor.kg.store import default_db_path, shared_db_path

    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)
    monkeypatch.delenv("RS_KG_PATH", raising=False)
    legacy_dir = tmp_path / "kg"
    legacy_dir.mkdir()
    legacy_db = legacy_dir / "graph.db"
    legacy_db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    monkeypatch.chdir(tmp_path)

    assert default_db_path() == shared_db_path()
    assert "sculpt kg merge" in capsys.readouterr().err


def test_merge_stores_is_additive_and_never_clobbers(tmp_path):
    """merge_stores copies missing nodes/edges/embeddings and NEVER
    overwrites an existing destination node (a legacy stub must not
    clobber the shared graph's richer node of the same id)."""
    import numpy as np

    from sculptor.kg.schema import Edge, FailureMode, Relation, RunCase
    from sculptor.kg.store import SculptorKG, merge_stores

    src = SculptorKG(tmp_path / "stray.db")
    dst = SculptorKG(tmp_path / "shared.db")
    try:
        # dst holds the rich node; src holds a stub with the SAME id.
        dst.add_node(FailureMode(
            id="failure:static_equilibrium", name="static_equilibrium",
            description="rich paper-derived description"))
        src.add_node(FailureMode(
            id="failure:static_equilibrium", name="static_equilibrium",
            description="(diagnoser-flagged stub)"))
        # src-only content.
        case = RunCase(id="case:x", task="hop", symptom="stand-still")
        src.add_node(case)
        src.add_edge(Edge(src="case:x", dst="failure:static_equilibrium",
                          relation=Relation.INSTANTIATES))
        src.set_embedding("case:x", "test-model",
                          np.ones(4, dtype=np.float32))
        src.close()

        counts = merge_stores(tmp_path / "stray.db", dst)
        assert counts["nodes"] == 1                # only the case copied
        assert counts["nodes_skipped"] == 1        # stub did NOT clobber
        assert counts["edges"] == 1
        assert counts["embeddings"] == 1
        kept = dst.get_node("failure:static_equilibrium")
        assert kept.description == "rich paper-derived description"
        assert dst.has_node("case:x")
        assert dst.has_embedding("case:x", "test-model")
        # Idempotent: merging again adds nothing.
        counts2 = merge_stores(tmp_path / "stray.db", dst)
        assert counts2["nodes"] == 0 and counts2["edges"] == 0
    finally:
        dst.close()


def test_merge_stores_rejects_self_merge(tmp_path):
    import pytest as _pytest

    from sculptor.kg.store import SculptorKG, merge_stores

    dst = SculptorKG(tmp_path / "one.db")
    try:
        with _pytest.raises(ValueError, match="same database"):
            merge_stores(tmp_path / "one.db", dst)
    finally:
        dst.close()


def test_default_db_path_falls_back_to_shared(monkeypatch, tmp_path):
    """With no env var and no legacy DB, returns the shared path."""
    from sculptor.kg.store import default_db_path, shared_db_path

    monkeypatch.delenv("SCULPTOR_KG_PATH", raising=False)
    monkeypatch.delenv("RS_KG_PATH", raising=False)
    empty = tmp_path / "empty_cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)

    assert default_db_path() == shared_db_path()


def test_shared_db_path_is_under_local_share():
    """`shared_db_path()` points at ~/.local/share/sculptor/kg/graph.db
    so all UI backends + CLI usages converge on one location."""
    from sculptor.kg.store import shared_db_path

    p = shared_db_path()
    assert p.name == "graph.db"
    assert p.parent.name == "kg"
    assert p.parent.parent.name == "sculptor"
    assert p.parent.parent.parent.name == "share"


# ── §7.7: arxiv retry-with-backoff + stub-title heal ──────────────────────
def test_fetch_arxiv_metadata_retries_on_transient_failure(monkeypatch):
    """`_fetch_arxiv_metadata` must retry up to len(delays) times before
    returning None. Uses a stub arxiv module that succeeds on the 3rd
    attempt to exercise the retry path without network."""
    from sculptor.kg import ingest

    call_count = {"n": 0}
    slept: list[float] = []

    class _FakeResult:
        title = "Stub Title"
        authors: list = []
        published = None
        summary = "abs"

    class _FakeClient:
        def __init__(self, **_kw):
            pass

        def results(self, _search):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("arxiv rate-limit (simulated)")

            class _Iter:
                def __iter__(self_inner):
                    return self_inner

                def __next__(self_inner):
                    return _FakeResult()

            return _Iter()

    class _FakeArxivModule:
        Client = _FakeClient

        class Search:
            def __init__(self, **_kw):
                pass

    monkeypatch.setitem(__import__("sys").modules, "arxiv", _FakeArxivModule)

    out = ingest._fetch_arxiv_metadata(
        "1234.5678",
        retry_delays_s=(0.0, 0.01, 0.01, 0.01),
        sleep_fn=lambda s: slept.append(s),
    )
    assert out is not None, "should have succeeded on 3rd attempt"
    assert out["title"] == "Stub Title"
    assert call_count["n"] == 3
    # Sleep called twice (between attempts 1→2 and 2→3).
    assert len(slept) == 2


def test_fetch_arxiv_metadata_returns_none_after_all_retries(monkeypatch):
    """All attempts fail → returns None (not raises). Exercises the
    `return None` branch after the last retry."""
    from sculptor.kg import ingest

    class _FailingClient:
        def __init__(self, **_kw):
            pass

        def results(self, _search):
            raise RuntimeError("permanent failure")

    class _FakeArxivModule:
        Client = _FailingClient

        class Search:
            def __init__(self, **_kw):
                pass

    monkeypatch.setitem(__import__("sys").modules, "arxiv", _FakeArxivModule)

    out = ingest._fetch_arxiv_metadata(
        "1234.5678",
        retry_delays_s=(0.0, 0.01, 0.01),
        sleep_fn=lambda s: None,
    )
    assert out is None


def test_heal_stub_titles_no_stubs_returns_empty(kg):
    """Happy-path KG (all titles populated) → no-op, returns empty dict."""
    kg.add_node(Paper(
        id=make_paper_id("2020.00001"), arxiv_id="2020.00001",
        title="A Real Title", authors=["Author"], year=2020,
    ))

    from sculptor.kg.ingest import heal_stub_titles
    assert heal_stub_titles(store=kg) == {}


def test_heal_stub_titles_identifies_stub_papers(kg, monkeypatch):
    """KG has two stub-titled papers + one healthy paper. Heal must
    attempt re-ingest on both stubs and leave the healthy paper alone."""
    kg.add_node(Paper(
        id=make_paper_id("9999.00001"), arxiv_id="9999.00001",
        title="arxiv:9999.00001",
        authors=[], year=None,
    ))
    kg.add_node(Paper(
        id=make_paper_id("9999.00002"), arxiv_id="9999.00002",
        title="arxiv:9999.00002",
        authors=[], year=None,
    ))
    kg.add_node(Paper(
        id=make_paper_id("2020.00001"), arxiv_id="2020.00001",
        title="Real Paper",
        authors=[], year=2020,
    ))

    attempted: list[str] = []

    def _fake_ingest(arxiv_id, *, store=None, force=False, fallback_metadata=None):
        attempted.append(arxiv_id)
        p = store.get_node(make_paper_id(arxiv_id))
        if p is not None:
            p.title = f"Healed Title for {arxiv_id}"
            store.add_node(p)
        return p

    from sculptor.kg import ingest
    monkeypatch.setattr(ingest, "ingest_arxiv", _fake_ingest)

    results = ingest.heal_stub_titles(store=kg)
    assert sorted(attempted) == ["9999.00001", "9999.00002"]
    assert results["9999.00001"] == "healed"
    assert results["9999.00002"] == "healed"
    assert "2020.00001" not in results


def test_heal_stub_titles_reports_still_stubbed_when_retry_fails(
    kg, monkeypatch,
):
    """Retry attempt lands during a rate-limit window → title stays as
    `arxiv:XXXX.XXXXX`. Heal result marks it `still_stubbed` so the
    caller knows to retry later."""
    kg.add_node(Paper(
        id=make_paper_id("9999.00099"), arxiv_id="9999.00099",
        title="arxiv:9999.00099",
        authors=[], year=None,
    ))

    def _fake_ingest(arxiv_id, *, store=None, force=False, fallback_metadata=None):
        return store.get_node(make_paper_id(arxiv_id))

    from sculptor.kg import ingest
    monkeypatch.setattr(ingest, "ingest_arxiv", _fake_ingest)

    results = ingest.heal_stub_titles(store=kg)
    assert results == {"9999.00099": "still_stubbed"}
