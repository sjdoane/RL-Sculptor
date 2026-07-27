"""tests/test_kg_fulltext.py — lexical recall over paper BODIES.

Paper retrieval embedded `title + abstract + rationale` only
(`query.paper_embed_text`), and nothing in the query path ever opened
`full_text_path`. So a paper that answers a question in section IV but never
says so in its abstract was unretrievable — the concrete case was OGMP, whose
Eq. 8 contains "non-toe contact" verbatim while its abstract does not, and
which abstract-only search missed entirely.

These pin the fix AND its safety contract: with an unindexed graph the ranking
must be bit-identical to before, or enabling this silently reshuffles every
existing campaign's citations.
"""
from __future__ import annotations

import pytest

from sculptor.kg.ingest import backfill_full_text_index
from sculptor.kg.schema import Paper, make_paper_id
from sculptor.kg.store import SculptorKG


def _paper(kg: SculptorKG, aid: str, *, abstract: str = "", body: str = "",
           tags: list[str] | None = None, tier: str | None = None,
           tmp_path=None) -> Paper:
    p = Paper(id=make_paper_id(aid), arxiv_id=aid, title=f"Paper {aid}",
              abstract=abstract, tags=tags or [], tier=tier)
    if body and tmp_path is not None:
        f = tmp_path / f"{aid}.txt"
        f.write_text(body, encoding="utf-8")
        p.full_text_path = str(f)
    kg.add_node(p)
    return p


@pytest.fixture()
def kg(tmp_path):
    store = SculptorKG(db_path=tmp_path / "g.db")
    yield store
    store.close()


# ── the index itself ────────────────────────────────────────────────────
def test_fts5_is_available_in_this_sqlite_build(kg):
    """Everything below degrades to a no-op without it — so state the
    assumption rather than letting the suite pass vacuously."""
    assert kg._fts_ok is True


def test_indexing_is_idempotent(kg):
    assert kg.index_paper_fulltext("n1", "1234.5678", "alpha beta") is True
    assert kg.index_paper_fulltext("n1", "1234.5678", "alpha beta") is True
    assert kg.count_paper_fulltext() == 1


def test_reindexing_replaces_rather_than_appends(kg):
    kg.index_paper_fulltext("n1", "1234.5678", "quadruped locomotion")
    kg.index_paper_fulltext("n1", "1234.5678", "bipedal parkour")
    assert kg.count_paper_fulltext() == 1
    assert [n for n, _ in kg.search_paper_fulltext("parkour")] == ["n1"]
    assert kg.search_paper_fulltext("quadruped") == []


def test_search_ranks_best_first_with_positive_scores(kg):
    kg.index_paper_fulltext("hit", "1", "parkour parkour parkour leaping")
    kg.index_paper_fulltext("weak", "2", "a single mention of parkour here")
    hits = kg.search_paper_fulltext("parkour")
    assert [n for n, _ in hits][0] == "hit"
    assert all(score > 0 for _, score in hits)


def test_a_blank_or_punctuation_only_query_returns_nothing(kg):
    kg.index_paper_fulltext("n1", "1", "body text")
    assert kg.search_paper_fulltext("") == []
    assert kg.search_paper_fulltext("   ") == []
    assert kg.search_paper_fulltext("?? -- ??") == []


def test_fts_operators_in_user_text_do_not_raise(kg):
    """A free-text question is not guaranteed to be valid FTS syntax. Returning
    nothing is acceptable; raising into a campaign is not."""
    kg.index_paper_fulltext("n1", "1", "reward shaping for locomotion")
    for q in ['reward AND', 'NEAR(', '"unbalanced quote', 'a OR OR b', '*(']:
        assert isinstance(kg.search_paper_fulltext(q), list)


# ── backfill ────────────────────────────────────────────────────────────
def test_backfill_indexes_bodies_and_counts_missing(kg, tmp_path):
    _paper(kg, "1111.1111", body="oracle guided policy", tmp_path=tmp_path)
    _paper(kg, "2222.2222", body="domain randomization", tmp_path=tmp_path)
    _paper(kg, "3333.3333")                      # no body on disk
    assert backfill_full_text_index(kg) == {
        "indexed": 2, "missing": 1, "skipped": 0}
    assert kg.count_paper_fulltext() == 2


def test_backfill_survives_a_dead_full_text_path(kg, tmp_path):
    p = _paper(kg, "4444.4444", body="x", tmp_path=tmp_path)
    (tmp_path / "4444.4444.txt").unlink()
    kg.add_node(p)
    assert backfill_full_text_index(kg)["missing"] == 1


# ── retrieval fusion ────────────────────────────────────────────────────
def test_an_unindexed_graph_ranks_exactly_as_before(kg, tmp_path):
    """The safety contract. Bodies exist on disk but were never indexed, so
    fusion must be a no-op — not a reshuffle."""
    from sculptor.kg.query import query_papers

    for i, topic in enumerate(["locomotion", "manipulation", "navigation"]):
        _paper(kg, f"900{i}.0001", abstract=f"a paper about {topic}",
               body="non-toe contact penalty", tmp_path=tmp_path)
    q = "non-toe contact penalty"
    off = [m.paper.arxiv_id for m in query_papers(q, store=kg, full_text=False)]
    on = [m.paper.arxiv_id for m in query_papers(q, store=kg, full_text=True)]
    assert on == off


def test_a_body_only_match_becomes_retrievable(kg, tmp_path):
    """The actual bug. The abstract never mentions the term; the body does."""
    from sculptor.kg.query import query_papers

    for i in range(6):
        _paper(kg, f"800{i}.0001",
               abstract="reinforcement learning for robotic manipulation",
               tmp_path=tmp_path)
    _paper(kg, "8099.0001",
           abstract="reinforcement learning for robotic manipulation",
           body="the reward penalizes any non-toe contact with the ground",
           tmp_path=tmp_path)
    backfill_full_text_index(kg)

    q = "non-toe contact"
    off = [m.paper.arxiv_id for m in query_papers(q, top_k=3, store=kg,
                                                  full_text=False)]
    on = [m.paper.arxiv_id for m in query_papers(q, top_k=3, store=kg,
                                                 full_text=True)]
    assert "8099.0001" not in off      # abstracts are all identical — invisible
    assert "8099.0001" in on           # ...and the body finds it


def test_full_text_cannot_smuggle_a_paper_past_a_tag_filter(kg, tmp_path):
    """Structured filters are exact by contract. A body match must not be a
    back door around a tag the caller explicitly required."""
    from sculptor.kg.query import query_papers

    _paper(kg, "7001.0001", abstract="quadruped work", tags=["locomotion"],
           tmp_path=tmp_path)
    _paper(kg, "7002.0001", abstract="unrelated", tags=["manipulation"],
           body="non-toe contact penalty everywhere", tmp_path=tmp_path)
    backfill_full_text_index(kg)

    got = [m.paper.arxiv_id for m in query_papers(
        "non-toe contact penalty", store=kg, tags=["locomotion"],
        full_text=True)]
    assert "7002.0001" not in got


def test_full_text_cannot_smuggle_a_paper_past_a_tier_filter(kg, tmp_path):
    from sculptor.kg.query import query_papers

    _paper(kg, "7101.0001", abstract="anything", tier="A", tmp_path=tmp_path)
    _paper(kg, "7102.0001", abstract="anything", tier="B",
           body="non-toe contact penalty", tmp_path=tmp_path)
    backfill_full_text_index(kg)

    got = [m.paper.arxiv_id for m in query_papers(
        "non-toe contact penalty", store=kg, tier="A", full_text=True)]
    assert "7102.0001" not in got


def test_semantic_stays_primary_when_it_already_has_the_answer(kg, tmp_path):
    """Lexical is a recall supplement at half weight — a strong semantic top
    hit must not be displaced by a paper that merely shares a common word."""
    from sculptor.kg.query import query_papers

    _paper(kg, "6001.0001",
           abstract="oracle guided policy optimization with an ansatz",
           tmp_path=tmp_path)
    for i in range(5):
        _paper(kg, f"600{i + 2}.0001", abstract="an unrelated survey",
               body="the state of the art " * 60, tmp_path=tmp_path)
    backfill_full_text_index(kg)

    top = query_papers("oracle ansatz policy optimization", top_k=3, store=kg,
                       full_text=True)
    assert top[0].paper.arxiv_id == "6001.0001"
