"""tests/test_kg_hardening.py — §Phase-0 hardening 2026-07-18.

Covers the audit's confirmed gaps:
  1. extract._materialize's MERGE path preserves run-learned fields
     (useful_citations / outcome_stats) and upgrades provenance by trust
     tier instead of resetting nodes on re-extraction.
  2. Embedding staleness: text_hash tracking, stale detection after a
     description change, trust-once backfill for pre-hash rows, and the
     shared ensure_embeddings pool re-embedding stale vectors.
  3. row_to_node forward-compat: unknown data keys are dropped, not
     TypeError.
  4. Unknown relation strings: neighbors()/all_edges() skip the bad row
     instead of aborting the whole scan.
  5. Schema migration: a pre-hash DB gains the text_hash column on open.
  6. kg doctor: detects + repairs referential slack; heal_dead_text_paths.

Offline: no LLM, no network, no sentence-transformers — the embedder is
stubbed exactly like tests/test_kg_query.py does.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sculptor.kg.query as qmod
from sculptor.kg.cases import _case_text, _ensure_case_embeddings
from sculptor.kg.doctor import format_report, run_doctor
from sculptor.kg.extract import (
    ExtractionPayload,
    FailureModeExtract,
    TechniqueExtract,
    _materialize,
)
from sculptor.kg.ingest import heal_dead_text_paths
from sculptor.kg.query import (
    ensure_embeddings,
    technique_embed_text,
)
from sculptor.kg.schema import (
    Edge,
    FailureMode,
    Paper,
    PROVENANCE_LLM_EXTRACTION,
    PROVENANCE_OBSERVED_RUN,
    PROVENANCE_PAPER_CLAIM,
    PROVENANCE_SEED,
    Relation,
    RunCase,
    Technique,
    make_failure_mode_id,
    make_paper_id,
    make_technique_id,
    merge_provenance,
    row_to_node,
)
from sculptor.kg.store import SculptorKG, embedding_text_hash


@pytest.fixture
def kg(tmp_path: Path):
    store = SculptorKG(tmp_path / "kg.db")
    yield store
    store.close()


class _FakeEmbedder:
    """Deterministic text->vector stub: different text => different vector,
    same text => same vector. Records what it was asked to encode."""

    def __init__(self):
        self.encoded: list[str] = []

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(
            int(embedding_text_hash(text)[:8], 16))
        v = rng.standard_normal(8).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            self.encoded.append(texts)
            return self._vec(texts)
        self.encoded.extend(texts)
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def fake_embedder(monkeypatch):
    emb = _FakeEmbedder()
    monkeypatch.setattr(qmod, "_get_embedder", lambda *a, **k: emb)
    return emb


def _paper(kg: SculptorKG, aid: str = "2400.00001") -> Paper:
    p = Paper(id=make_paper_id(aid), arxiv_id=aid, title="T", abstract="A")
    kg.add_node(p)
    return p


def _payload_with_technique(name="reference_state_initialization",
                            description="short desc") -> ExtractionPayload:
    return ExtractionPayload(techniques=[TechniqueExtract(
        name=name, description=description, tags=["rsi"],
        evidence="a long enough verbatim snippet from the paper text")])


# ── 1. merge preserves run-learned fields ───────────────────────────────────
def test_reextraction_preserves_useful_citations_and_outcome_stats(kg):
    tid = make_technique_id("reference_state_initialization")
    fm_id = make_failure_mode_id("floor_sit")
    kg.add_node(Technique(
        id=tid, name="reference_state_initialization",
        description="existing description that is longer than the new one",
        tags=["init"], useful_citations=7,
        outcome_stats={fm_id: {"helped": 3, "regressed": 1}}))

    _materialize(kg, _paper(kg), _payload_with_technique())

    after = kg.get_node(tid)
    assert after.useful_citations == 7
    assert after.outcome_stats == {fm_id: {"helped": 3, "regressed": 1}}
    # richer existing description kept, tags unioned
    assert after.description.startswith("existing description")
    assert set(after.tags) == {"init", "rsi"}


def test_reextraction_upgrades_stub_failure_mode_provenance(kg):
    fm_id = make_failure_mode_id("floor_sit")
    kg.add_node(FailureMode(
        id=fm_id, name="floor_sit",
        description="(diagnoser-flagged failure mode; not paper-derived)",
        provenance=PROVENANCE_LLM_EXTRACTION))

    payload = ExtractionPayload(failure_modes=[FailureModeExtract(
        name="floor_sit", description="policy sits on the floor to farm",
        symptoms=["low base"],
        evidence="a long enough verbatim snippet from the paper text")])
    _materialize(kg, _paper(kg), payload)

    after = kg.get_node(fm_id)
    assert after.provenance == PROVENANCE_PAPER_CLAIM
    # the placeholder stub description is replaced even though it is longer
    assert after.description == "policy sits on the floor to farm"
    assert "low base" in after.symptoms


def test_merge_provenance_trust_ordering():
    assert merge_provenance(
        PROVENANCE_LLM_EXTRACTION, PROVENANCE_PAPER_CLAIM) \
        == PROVENANCE_PAPER_CLAIM
    assert merge_provenance(
        PROVENANCE_OBSERVED_RUN, PROVENANCE_PAPER_CLAIM) \
        == PROVENANCE_OBSERVED_RUN
    assert merge_provenance(PROVENANCE_SEED, None) == PROVENANCE_SEED
    assert merge_provenance(None, "future_unknown_tier") \
        == PROVENANCE_LLM_EXTRACTION  # unknown ranks least-trusted; tie -> existing


# ── 2. embedding staleness ──────────────────────────────────────────────────
def test_embedding_status_lifecycle(kg):
    vec = np.ones(4, dtype=np.float32)
    assert kg.embedding_status("n1", "m", "text v1") == "missing"
    kg.set_embedding("n1", "m", vec, text="text v1")
    assert kg.embedding_status("n1", "m", "text v1") == "fresh"
    assert kg.embedding_status("n1", "m", "text v2") == "stale"
    # legacy write without text -> unhashed; backfill starts tracking
    kg.set_embedding("n2", "m", vec)
    assert kg.embedding_status("n2", "m", "whatever") == "unhashed"
    kg.backfill_embedding_hash("n2", "m", "whatever")
    assert kg.embedding_status("n2", "m", "whatever") == "fresh"
    assert kg.embedding_status("n2", "m", "changed") == "stale"
    # backfill never overwrites an existing hash
    kg.backfill_embedding_hash("n2", "m", "changed")
    assert kg.embedding_status("n2", "m", "whatever") == "fresh"


def test_ensure_embeddings_reembeds_after_description_change(
        kg, fake_embedder):
    tid = make_technique_id("rsi")
    kg.add_node(Technique(id=tid, name="rsi", description="old description"))
    pool = ensure_embeddings(
        kg, kg.find_nodes(kind="Technique"), technique_embed_text)
    v_old = dict((n.id, v) for n, v in pool)[tid]

    # description changes (an extraction merge would do this)
    kg.add_node(Technique(id=tid, name="rsi", description="new richer description"))
    pool = ensure_embeddings(
        kg, kg.find_nodes(kind="Technique"), technique_embed_text)
    v_new = dict((n.id, v) for n, v in pool)[tid]

    assert not np.allclose(v_old, v_new), "stale vector was served"
    assert kg.embedding_status(
        tid, qmod.EMBEDDING_MODEL,
        technique_embed_text(kg.get_node(tid))) == "fresh"


def test_ensure_embeddings_trusts_prehash_rows_once(kg, fake_embedder):
    tid = make_technique_id("rsi")
    kg.add_node(Technique(id=tid, name="rsi", description="desc"))
    seeded = np.ones(8, dtype=np.float32) / np.sqrt(8)
    kg.set_embedding(tid, qmod.EMBEDDING_MODEL, seeded)  # no text: unhashed

    pool = ensure_embeddings(
        kg, kg.find_nodes(kind="Technique"), technique_embed_text)
    v = dict((n.id, v) for n, v in pool)[tid]
    assert np.allclose(v, seeded), "trust-once row was re-embedded"
    assert fake_embedder.encoded == []
    # ...but the hash is now stamped, so a text change IS tracked
    kg.add_node(Technique(id=tid, name="rsi", description="changed desc"))
    pool = ensure_embeddings(
        kg, kg.find_nodes(kind="Technique"), technique_embed_text)
    v2 = dict((n.id, v) for n, v in pool)[tid]
    assert not np.allclose(v2, seeded)


def test_case_reembedding_after_verdict_reattribution(kg, fake_embedder):
    case = RunCase(id="case:t:0:abc", task="jump", symptom="floor_sit",
                   verdict="unknown")
    kg.add_node(case)
    _ensure_case_embeddings(kg)
    v_old = kg.get_embedding(case.id, qmod.EMBEDDING_MODEL)

    # resume re-records the same case id with a measured verdict
    kg.add_node(RunCase(id=case.id, task="jump", symptom="floor_sit",
                        verdict="regressed"))
    _ensure_case_embeddings(kg)
    v_new = kg.get_embedding(case.id, qmod.EMBEDDING_MODEL)
    assert not np.allclose(v_old, v_new), (
        "case embedding not refreshed after verdict changed "
        f"({_case_text(case)!r})")


# ── 3. row_to_node forward-compat ───────────────────────────────────────────
def test_row_to_node_drops_unknown_keys():
    node = row_to_node(
        "technique:x", "Technique",
        {"name": "x", "description": "d", "from_the_future": 42})
    assert isinstance(node, Technique)
    assert not hasattr(node, "from_the_future")


def test_get_node_survives_future_field_on_disk(kg):
    tid = make_technique_id("x")
    kg.add_node(Technique(id=tid, name="x", description="d"))
    # simulate a newer schema having written an extra field
    import json
    row = kg._conn.execute(
        "SELECT data FROM nodes WHERE id = ?", (tid,)).fetchone()
    data = json.loads(row["data"])
    data["future_field"] = {"nested": True}
    with kg._tx() as cx:
        cx.execute("UPDATE nodes SET data = ? WHERE id = ?",
                   (json.dumps(data), tid))
    node = kg.get_node(tid)
    assert isinstance(node, Technique) and node.name == "x"


def test_row_to_node_unknown_kind_still_raises():
    with pytest.raises(ValueError, match="unknown node kind"):
        row_to_node("z:1", "FutureKind", {"name": "z"})


# ── 4. unknown relation tolerance ───────────────────────────────────────────
def _insert_raw_edge(kg, src, dst, relation):
    with kg._tx() as cx:
        cx.execute(
            "INSERT OR REPLACE INTO edges (src, dst, relation, data) "
            "VALUES (?, ?, ?, ?)", (src, dst, relation, "{}"))


def test_neighbors_skips_unknown_relation_rows(kg):
    tid = make_technique_id("t")
    fid = make_failure_mode_id("f")
    kg.add_node(Technique(id=tid, name="t"))
    kg.add_node(FailureMode(id=fid, name="f"))
    kg.add_edge(Edge(src=tid, dst=fid, relation=Relation.ADDRESSES))
    _insert_raw_edge(kg, tid, fid, "FUTURE_RELATION")

    out = kg.neighbors(tid, direction="out")
    assert len(out) == 1 and out[0][0].relation == Relation.ADDRESSES
    assert sum(1 for _ in kg.all_edges()) == 1


# ── 5. migration ────────────────────────────────────────────────────────────
def test_prehash_db_gains_text_hash_column(tmp_path):
    db = tmp_path / "old.db"
    cx = sqlite3.connect(str(db))
    cx.executescript("""
        CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                            data TEXT NOT NULL);
        CREATE TABLE edges (src TEXT NOT NULL, dst TEXT NOT NULL,
                            relation TEXT NOT NULL, data TEXT NOT NULL,
                            PRIMARY KEY (src, dst, relation));
        CREATE TABLE node_embeddings (
            node_id TEXT NOT NULL, model TEXT NOT NULL,
            dim INTEGER NOT NULL, vector BLOB NOT NULL,
            updated_at REAL NOT NULL, PRIMARY KEY (node_id, model));
        INSERT INTO node_embeddings VALUES
            ('n1', 'm', 2, x'0000803f0000803f', 1.0);
    """)
    cx.commit()
    cx.close()

    store = SculptorKG(db)
    try:
        # legacy row visible + reported unhashed; vector intact
        assert store.embedding_status("n1", "m", "any") == "unhashed"
        v = store.get_embedding("n1", "m")
        assert v is not None and v.shape == (2,)
    finally:
        store.close()


# ── 6. doctor ───────────────────────────────────────────────────────────────
def test_doctor_detects_and_fixes_referential_slack(kg, fake_embedder):
    tid = make_technique_id("t")
    kg.add_node(Technique(id=tid, name="t", description="d"))
    # orphan embedding (node never existed)
    kg.set_embedding("technique:ghost", qmod.EMBEDDING_MODEL,
                     np.ones(4, dtype=np.float32))
    # dangling edge
    _insert_raw_edge(kg, tid, "failure:ghost", Relation.ADDRESSES.value)
    # unknown relation edge
    _insert_raw_edge(kg, tid, tid, "FUTURE_RELATION")
    # stale technique embedding
    kg.set_embedding(tid, qmod.EMBEDDING_MODEL,
                     np.ones(8, dtype=np.float32),
                     text="an old text that no longer matches")

    report = run_doctor(kg, fix=False)
    assert report["orphan_embeddings"] == ["technique:ghost"]
    assert (tid, "ADDRESSES", "failure:ghost") in report["dangling_edges"]
    assert (tid, "FUTURE_RELATION", tid) in report["unknown_relation_edges"]
    assert report["embedding_pools"]["Technique"]["stale"] == 1
    # read-only: nothing repaired
    assert kg.get_embedding("technique:ghost", qmod.EMBEDDING_MODEL) is not None

    report = run_doctor(kg, fix=True, network=False)
    assert report["fixes"]["orphan_embeddings_deleted"] == 1
    assert report["fixes"]["dangling_edges_deleted"] == 1
    assert kg.get_embedding("technique:ghost", qmod.EMBEDDING_MODEL) is None

    clean = run_doctor(kg, fix=False)
    assert clean["dangling_edges"] == []
    assert clean["orphan_embeddings"] == []
    assert clean["embedding_pools"]["Technique"]["stale"] == 0
    # unknown-relation edge is REPORTED but never auto-deleted (a newer
    # schema's data is not slack)
    assert clean["unknown_relation_edges"] == [
        (tid, "FUTURE_RELATION", tid)]
    assert "KG doctor" in format_report(clean)


def test_doctor_flags_dead_text_paths_and_heal_repairs(kg, monkeypatch,
                                                       tmp_path):
    aid = "2400.00042"
    dead = tmp_path / "gone" / "x.txt"
    kg.add_node(Paper(id=make_paper_id(aid), arxiv_id=aid, title="T",
                      full_text_path=str(dead), extracted=True))
    report = run_doctor(kg, fix=False)
    assert report["dead_text_paths"] == [aid]

    def _fake_ingest(arxiv_id, *, store=None, force=False, **kw):
        alive = tmp_path / f"{arxiv_id}.txt"
        alive.write_text("full text", encoding="utf-8")
        p = store.get_node(make_paper_id(arxiv_id))
        import dataclasses as dc
        p = dc.replace(p, full_text_path=str(alive))
        store.add_node(p)
        return p

    import sculptor.kg.ingest as imod
    monkeypatch.setattr(imod, "ingest_arxiv", _fake_ingest)
    results = heal_dead_text_paths(store=kg)
    assert results == {aid: "healed"}
    assert run_doctor(kg, fix=False)["dead_text_paths"] == []
    # extracted flag untouched by the heal
    assert kg.get_node(make_paper_id(aid)).extracted is True
