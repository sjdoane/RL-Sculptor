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


#: §2026-07-09 concept-boost fixture: reproduces the two failure
#: patterns an adversarial audit found against the real 301-clip
#: index — a rare MODIFIER token (forward/high) out-weighing the
#: motion-CONCEPT token it's attached to. Each block below mirrors one
#: real clip family so the ordering assertions below exercise the same
#: shape of index-wide IDF skew as production (many walk/run/lie/hop
#: rows sharing the concept token so its IDF is modest, one-or-two rows
#: carrying the rare modifier token so ITS IDF is large).
CONCEPT_BOOST_ROWS = [
    # walk family (concept token "walk" appears in 4 rows -> modest IDF)
    _row("walk_a_subject1", ["walk", "subject", "1"]),
    _row("walk_b_subject2", ["walk", "subject", "2"]),
    _row("walk_c_subject3", ["walk", "turn", "left", "subject", "3"]),
    _row("walk_d_subject4", ["walk", "with", "box", "subject", "4"]),
    # the decoy: literal "forward" (appears in exactly ONE row -> very
    # rare / high IDF) attached to a NON-walk-group concept ("crawl").
    _row("crawl_forward_subject5", ["crawl", "forward", "subject", "5"]),
    # jump/hop family (concept tokens "jump"/"hop" synonym-linked) —
    # same df (2) as the "high" decoy below, so the two tokens' raw IDF
    # weights are equal and the win comes purely from `_CONCEPT_BOOST`.
    _row("hop_a_subject6", ["hop", "subject", "6"]),
    _row("hop_b_subject7", ["hop", "subject", "7"]),
    # the decoy: literal "high" attached to a non-jump concept.
    _row("block_high_subject8", ["block", "high", "subject", "8"]),
    _row("block_high2_subject9", ["block", "high", "subject", "9"]),
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
    # §2026-07-11 (build-log D2 follow-up): "fall" was deliberately moved
    # OUT of the get-up group into its own group (see `SYNONYM_GROUPS`'s
    # docstring) — folding it into "up"'s group diluted "fall"'s corpus
    # rarity down to "up"'s (very common), which let unrelated up/down
    # clips outrank real fallAndGetUp clips for queries like "fall down".
    # "get up off the ground" still ranks fallAndGetUp top via literal
    # "get"/"up" overlap alone (see `test_deterministic_rank_fall_down_...
    # ranks_fallandgetup_first` below) — it never depended on this
    # cross-group link.
    assert "fall" not in expanded


def test_expand_synonyms_pulls_in_fall_group() -> None:
    """"fall" now has its own group, separate from get-up (§2026-07-11)."""
    expanded = expand_synonyms(["fall"])
    assert "falling" in expanded
    assert "collapse" in expanded
    assert "tumble" in expanded
    assert "get" not in expanded
    assert "up" not in expanded


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
        clip_path = (
            library.clip_dir("g1", row["clip_id"], root=tmp_path)
            / library.CLIP_FILENAME
        )
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"artifact:{row['clip_id']}".encode())
        prov = library.make_provenance(
            clip_id=row["clip_id"], robot="g1",
            source={"kind": "hf_dataset", "repo": "r",
                    "path": row["clip_id"], "url": "u"},
            license=row["license"], attribution="a",
            content_sha256_=library.content_sha256(clip_path.read_bytes()),
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


def test_search_filters_by_robot_symmetrically(tmp_path: Path) -> None:
    """§Problem 2: a g1 clip and a t1 clip carrying the IDENTICAL text/
    labels must each be found ONLY under their own robot — `search()`'s
    robot filter is not a g1-specific special case, it treats every
    robot slug the same way."""
    for robot in ("g1", "t1"):
        clip_id = f"fallandgetup1_subject1--{robot}"
        clip_path = (
            library.clip_dir(robot, clip_id, root=tmp_path)
            / library.CLIP_FILENAME
        )
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"artifact:{robot}".encode())
        prov = library.make_provenance(
            clip_id=clip_id, robot=robot,
            source={"kind": "hf_dataset", "repo": "r",
                    "path": "p", "url": "u"},
            license="cc-by-4.0", attribution="a",
                content_sha256_=library.content_sha256(
                    clip_path.read_bytes()),
            labels=["fall", "and", "get", "up", "1", "subject", "1"],
            text="fall and get up 1 subject 1",
            qc={"duration_s": 5.0, "n_frames": 150, "root_z_range": [0.1, 0.8]})
        library.write_provenance(robot, prov["clip_id"], prov, root=tmp_path)
    library.rebuild_index(root=tmp_path)

    g1_results = search(
        "get up off the ground", robot="g1", k=5, use_llm=False, root=tmp_path)
    t1_results = search(
        "get up off the ground", robot="t1", k=5, use_llm=False, root=tmp_path)

    assert len(g1_results) == 1
    assert g1_results[0].clip_id == "fallandgetup1_subject1--g1"
    assert len(t1_results) == 1
    assert t1_results[0].clip_id == "fallandgetup1_subject1--t1"


def test_deterministic_rank_walk_query_prefers_walk_over_getup() -> None:
    results = deterministic_rank("walking gait", FIXTURE_ROWS, k=10)
    assert results
    assert results[0].clip_id == "walk1_subject3"


def test_deterministic_rank_no_overlap_returns_empty() -> None:
    assert deterministic_rank("xyzzy plugh", FIXTURE_ROWS) == []


def test_deterministic_rank_score_tie_prefers_denser_match() -> None:
    """§2026-07-11 regression (found against the real index after the
    stageii ingest recovered the MOYO yoga clips): a verbose yoga-pose
    label containing the literal token "dance" ("lord of the dance pose
    or natarajasana") EXACTLY ties a real dance clip on IDF score for
    the query "dance", and the old clip_id-only tie-break put the yoga
    clip first purely because its id starts with a digit. Equal-score
    ties must now prefer the row with FEWER distinct tokens (tighter
    match density) — and the tie-break must not disturb any non-tied
    ordering."""
    rows = [
        _row("220926_yogi_lord_of_the_dance_pose_natarajasana",
             ["220926", "yogi", "body", "hands", "lord", "of", "the",
              "dance", "pose", "or", "natarajasana", "stageii", "60",
              "jpos"]),
        _row("dance1_subject5", ["dance", "1", "subject", "5"]),
    ]
    results = deterministic_rank("dance", rows, k=5)
    assert [m.clip_id for m in results][0] == "dance1_subject5"
    # Both still returned (the yoga clip is a legitimate weaker hit).
    assert {m.clip_id for m in results} == {
        "dance1_subject5", "220926_yogi_lord_of_the_dance_pose_natarajasana"}
    # Sanity: scores actually tied — this test exercises the tie-break,
    # not some scoring difference that could silently vanish later.
    assert results[0].score == pytest.approx(results[1].score)


# ── concept-boost regression: motion CONCEPT beats rare MODIFIER ─────────
# §2026-07-09: an adversarial audit against the real 301-clip library
# found that `_SYNONYM_MATCH_WEIGHT` alone (round 1) fixed "get up off
# the ground" but broke other queries — a rare literal MODIFIER token
# ("forward", "high") could out-weigh the motion-CONCEPT token it rides
# with, because IDF only measures corpus rarity, not conceptual
# relevance. `_CONCEPT_BOOST` fixes this by scaling every
# `SYNONYM_GROUPS`-credited point (concept matches) above plain-literal
# (modifier) matches. `CONCEPT_BOOST_ROWS` reproduces both failure
# shapes at fixture scale: a walk-family sharing "walk" plus one decoy
# row with the rare literal "forward" glued to an unrelated concept
# ("crawl"); a hop-family sharing "hop" (synonym-linked to "jump") plus
# decoy rows with the rare literal "high" glued to an unrelated concept
# ("block").
def test_deterministic_rank_walk_forward_prefers_walk_over_rare_modifier() -> None:
    results = deterministic_rank("walk forward", CONCEPT_BOOST_ROWS, k=10)
    assert results
    assert results[0].clip_id.startswith("walk_")
    walk_in_top5 = sum(
        1 for m in results[:5] if m.clip_id.startswith("walk_"))
    assert walk_in_top5 >= 3


def test_deterministic_rank_jump_high_prefers_hop_over_rare_modifier() -> None:
    results = deterministic_rank("jump high", CONCEPT_BOOST_ROWS, k=10)
    assert results
    top3 = results[:3]
    hop_in_top3 = sum(1 for m in top3 if m.clip_id.startswith("hop_"))
    assert hop_in_top3 >= 2


def test_deterministic_rank_get_up_off_ground_still_locked_with_concept_boost() -> None:
    """The original (round-1) acceptance query must keep passing under
    the round-2 `_CONCEPT_BOOST` scoring — this is the LOCKED
    requirement, not just a regression nice-to-have."""
    results = deterministic_rank(
        "get up off the ground", FIXTURE_ROWS, k=10)
    assert results
    assert all(m.clip_id.startswith("fallandgetup") for m in results[:2])


def test_deterministic_rank_fall_down_ranks_fallandgetup_first() -> None:
    """§2026-07-11 defect regression (deferred audit finding, build-log D2
    follow-up): "fall" no longer shares a synonym group with "up" (see
    `SYNONYM_GROUPS`'s docstring for the mechanism) — folding it into the
    get-up group diluted "fall"'s corpus rarity down to "up"'s (common
    across the real 5040-clip library: any *_up/*up_down/direction-style
    clip), which let an unrelated clip that only shared the common word
    "up" (bridged) plus the rare literal modifier "down" outrank the
    actual fallAndGetUp clips for the query "fall down". Full
    before/after top-5 against the real library is in
    `tests/test_refs_golden_queries.py` / `tests/data/golden_queries.yml`
    ("fall down" entry); this is the fixture-scale version of the same
    regression."""
    results = deterministic_rank("fall down", FIXTURE_ROWS, k=10)
    assert results
    assert results[0].clip_id.startswith("fallandgetup")


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
