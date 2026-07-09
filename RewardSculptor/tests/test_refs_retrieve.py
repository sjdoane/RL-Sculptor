"""§R1_BUILD_SPEC W2: reference-clip retrieval (`sculptor/refs/retrieve.py`).

Deterministic layer is zero-API and unit-tested directly; the LLM rerank
layer is tested ONLY via monkeypatched/injected clients — no test here
ever requires an `anthropic` package import to succeed or an API key to
be set (§Hard rules).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.refs import library
from sculptor.refs.retrieve import (
    RefMatch,
    RerankUnavailable,
    _rerank_with_llm,
    deterministic_rank,
    expand_synonyms,
    search,
    search_rows,
    tokenize_query,
)


# ── fixture index rows: fallAndGetUp segments + decoys ───────────────────
def _row(clip_id: str, labels: list[str], **overrides) -> dict:
    row = {
        "clip_id": clip_id,
        "robot": "g1",
        "text": " ".join(labels),
        "labels": labels,
        "tier": "K",
        "license": "CC BY-NC-ND 4.0",
        "n_frames": 200,
        "fps": 30.0,
        "duration_s": 6.6,
        "root_z_range": [0.1, 0.8],
        "has_preview": False,
    }
    row.update(overrides)
    return row


FIXTURE_ROWS = [
    _row("fallandgetup1_subject1--seg00",
         ["fall", "and", "get", "up", "1", "subject", "1", "segment"]),
    _row("fallandgetup2_subject2--seg00",
         ["fall", "and", "get", "up", "2", "subject", "2", "segment"]),
    _row("walk1_subject3", ["walk", "1", "subject", "3"]),
    _row("run1_subject4", ["run", "1", "subject", "4"]),
    _row("dance1_subject5", ["dance", "1", "subject", "5"]),
    _row("jumps1_subject6", ["jumps", "1", "subject", "6"]),
]


# ── tokenizer / synonym expansion ────────────────────────────────────────
def test_tokenize_query_matches_ingest_tokenize_label_on_filename_stems() -> None:
    """For filename-SHAPED input (no whitespace — what ingest actually
    tokenizes), the two tokenizers must agree byte-for-byte."""
    from sculptor.refs.ingest import tokenize_label

    for text in ("fallAndGetUp1_subject1", "dance1-2_subject3",
                 "ACCAD_subject1_lie_to_crouch_poses_120_jpos"):
        assert tokenize_query(text) == tokenize_label(text)


def test_tokenize_query_splits_on_whitespace_unlike_tokenize_label() -> None:
    """Deliberate divergence: `tokenize_query` handles free-text search
    queries (spaces are token boundaries); `tokenize_label` only ever
    sees filename stems and does not split on whitespace at all."""
    from sculptor.refs.ingest import tokenize_label

    assert tokenize_query("get up off the ground") == [
        "get", "up", "off", "the", "ground"]
    # tokenize_label treats the whole space-containing string as one
    # token — it was never designed to see free text.
    assert tokenize_label("get up off the ground") == ["get up off the ground"]


def test_expand_synonyms_pulls_in_getup_group() -> None:
    expanded = expand_synonyms(["get", "up"])
    assert "getup" in expanded
    assert "stand" in expanded
    assert "rise" in expanded
    assert "fall" in expanded  # get up off the ground <-> fall and get up


def test_expand_synonyms_leaves_unknown_tokens_alone() -> None:
    assert expand_synonyms(["subject", "42"]) == {"subject", "42"}


# ── deterministic ranking — THE acceptance query ─────────────────────────
def test_acceptance_query_ranks_fallandgetup_segment_first_no_llm() -> None:
    """§decision 7's literal acceptance test: "get up off the ground"
    with no LLM must rank a fallAndGetUp segment top against walk/run/
    dance/jump decoys."""
    results = search_rows(
        "get up off the ground", FIXTURE_ROWS, k=10, use_llm=False)
    assert results, "expected at least one match"
    assert results[0].clip_id.startswith("fallandgetup")
    assert results[0].match_confidence is None
    assert results[0].rerank == "deterministic-only"


def test_search_public_api_reads_disk_index(tmp_path: Path) -> None:
    """`search()` (not `search_rows()`) reads a real on-disk index via
    `library.read_index` — exercise the full disk path once."""
    for row in FIXTURE_ROWS:
        prov = library.make_provenance(
            clip_id=row["clip_id"], robot="g1",
            source={"kind": "hf_dataset", "repo": "r",
                    "path": row["clip_id"], "url": "u"},
            license=row["license"], attribution="a",
            content_sha256_=f"{hash(row['clip_id']) & 0xffffffffffffffff:016x}" * 4,
            labels=row["labels"], text=row["text"],
            qc={"duration_s": row["duration_s"], "n_frames": row["n_frames"],
                "root_z_range": row["root_z_range"]})
        library.write_provenance("g1", row["clip_id"], prov, root=tmp_path)
    library.rebuild_index(root=tmp_path)

    results = search(
        "get up off the ground", robot="g1", k=5, use_llm=False,
        root=tmp_path)
    assert results
    assert results[0].clip_id.startswith("fallandgetup")


def test_search_unknown_robot_returns_empty(tmp_path: Path) -> None:
    library.rebuild_index(root=tmp_path)  # empty library
    assert search("anything", robot="g1", root=tmp_path, use_llm=False) == []


def test_deterministic_rank_walk_query_prefers_walk_over_getup() -> None:
    results = deterministic_rank("walking gait", FIXTURE_ROWS, k=10)
    assert results
    assert results[0].clip_id == "walk1_subject3"


def test_deterministic_rank_no_overlap_returns_empty() -> None:
    assert deterministic_rank("xyzzy plugh", FIXTURE_ROWS) == []


def test_deterministic_rank_respects_k() -> None:
    results = deterministic_rank("subject", FIXTURE_ROWS, k=2)
    assert len(results) == 2


def test_deterministic_rank_ties_broken_by_clip_id() -> None:
    """Two rows with identical token sets score identically; the
    tiebreak must be deterministic (sorted clip_id), not dict/set
    iteration order, so results are stable across runs/platforms."""
    rows = [
        _row("bbb_clip", ["dance"]),
        _row("aaa_clip", ["dance"]),
    ]
    results = deterministic_rank("dance", rows, k=10)
    assert [m.clip_id for m in results] == ["aaa_clip", "bbb_clip"]


# ── LLM rerank layer — success + fallback paths ──────────────────────────
class _FakeParsedResponse:
    def __init__(self, ranked):
        self.parsed_output = _FakeParsed(ranked)
        self.usage = None


class _FakeParsed:
    def __init__(self, ranked):
        self.ranked = ranked


class _FakeRankedItem:
    def __init__(self, clip_id, match_confidence, reason):
        self.clip_id = clip_id
        self.match_confidence = match_confidence
        self.reason = reason


class _FakeMessages:
    def __init__(self, response):
        self._response = response

    def parse(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_rerank_with_fake_client_reorders_and_tags_llm() -> None:
    candidates = deterministic_rank(
        "get up off the ground", FIXTURE_ROWS, k=10)
    # Reverse the deterministic order to prove the LLM ranking actually
    # took effect (not just passed through).
    fake_ranked = [
        _FakeRankedItem(m.clip_id, 0.9 - 0.1 * i, f"reason {i}")
        for i, m in enumerate(reversed(candidates))
    ]
    fake_client = _FakeClient(_FakeParsedResponse(fake_ranked))

    reranked = _rerank_with_llm(
        "get up off the ground", candidates, client=fake_client)

    assert [m.clip_id for m in reranked] == [m.clip_id for m in reversed(candidates)]
    assert all(m.rerank == "llm" for m in reranked)
    assert reranked[0].match_confidence == pytest.approx(0.9)
    assert reranked[0].reason == "reason 0"


def test_rerank_with_fake_client_rejects_unknown_clip_id() -> None:
    candidates = deterministic_rank(
        "get up off the ground", FIXTURE_ROWS, k=10)
    fake_ranked = [_FakeRankedItem("not-a-real-clip-id", 0.5, "hallucinated")]
    fake_client = _FakeClient(_FakeParsedResponse(fake_ranked))
    with pytest.raises(RerankUnavailable):
        _rerank_with_llm("get up off the ground", candidates, client=fake_client)


def test_search_rows_use_llm_true_with_working_fake_client() -> None:
    # "subject" overlaps every fixture row -> a full-width candidate set,
    # so k=3 actually exercises the truncation path (unlike the
    # narrower "get up off the ground" query, which only 2 rows match).
    candidates = deterministic_rank("subject", FIXTURE_ROWS, k=10)
    assert len(candidates) >= 3
    fake_ranked = [
        _FakeRankedItem(m.clip_id, 0.8, "ok") for m in candidates
    ]
    fake_client = _FakeClient(_FakeParsedResponse(fake_ranked))
    results = search_rows(
        "subject", FIXTURE_ROWS, k=3, use_llm=True, client=fake_client)
    assert len(results) == 3
    assert all(m.rerank == "llm" for m in results)


# ── LLM rerank layer — the NEVER RAISES fallback contract ───────────────
def test_search_falls_back_to_deterministic_when_client_ctor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch `anthropic.Anthropic` (client CONSTRUCTION) to raise —
    §decision 7 requires this to degrade to the deterministic ranking,
    never propagate. No API key, no network, no real `anthropic` import
    needed beyond the module existing (or not) in the environment."""
    import sys
    import types

    fake_anthropic = types.ModuleType("anthropic")

    def _raising_ctor(*args, **kwargs):
        raise RuntimeError("no API key configured")

    fake_anthropic.Anthropic = _raising_ctor
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    results = search_rows(
        "get up off the ground", FIXTURE_ROWS, k=5, use_llm=True,
        client=None)
    assert results
    assert results[0].clip_id.startswith("fallandgetup")
    assert results[0].match_confidence is None
    assert results[0].rerank == "deterministic-fallback"


def test_search_falls_back_when_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client constructs fine but `.messages.parse(...)` raises (network
    blip / malformed response) — same fallback contract."""
    class _RaisingMessages:
        def parse(self, **kwargs):
            raise ConnectionError("network blip")

    class _RaisingClient:
        def __init__(self):
            self.messages = _RaisingMessages()

    results = search_rows(
        "get up off the ground", FIXTURE_ROWS, k=5, use_llm=True,
        client=_RaisingClient())
    assert results
    assert results[0].clip_id.startswith("fallandgetup")
    assert results[0].rerank == "deterministic-fallback"


def test_search_rows_never_raises_on_empty_rows() -> None:
    assert search_rows("anything", [], use_llm=True) == []
    assert search_rows("anything", [], use_llm=False) == []


def test_ref_match_is_the_documented_dataclass_shape() -> None:
    m = RefMatch(
        clip_id="c", text="t", score=1.0, match_confidence=None,
        reason=None, tier="K", license="cc-by-4.0", n_frames=10,
        fps=30.0, duration_s=1.0)
    assert m.rerank == "deterministic-only"  # default
