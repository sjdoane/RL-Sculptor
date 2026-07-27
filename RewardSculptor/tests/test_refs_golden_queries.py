"""Golden-query acceptance suite for `sculptor.refs.retrieve` (deterministic
layer only — `use_llm=False`, zero API, never flaky).

Queries + expected-class regexes live in `tests/data/golden_queries.yml`
(edit that file, not this one, to add/adjust a query). Two independent
layers, both driving the exact same `search_rows`/`search` code path the
CLI and production callers use:

  1. **Fixture layer** (`test_golden_query_against_fixture`) — always on,
     zero disk I/O. `GOLDEN_FIXTURE_ROWS` is a small, hand-built index with
     exactly one clean representative clip per motion category (plus a
     couple of intentional cross-category decoys reused from other
     categories' rows — e.g. "sitdown"/"standup" rows double as decoys for
     the fall/get-up queries). This is the suite's floor: it must pass on
     every checkout, with or without the real reference library installed.

  2. **Real-index layer** (`test_golden_query_against_real_index`) —
     skipped when `~/.local/share/reward-sculptor/references/index.jsonl`
     (or `RS_REFERENCE_ROOT`'s equivalent) doesn't exist. Exercises
     `search()` (disk I/O, `library.read_index`) against the actual g1
     reference library, so it's the suite's real-world signal — but only
     for entries marked `real_index: true` in the YAML (a couple of
     categories, e.g. "sit", currently have zero matching clips in the
     real library and are honestly excluded rather than pinned to a
     misleading proxy match; see the YAML file's per-entry notes).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from sculptor.refs import library
from sculptor.refs.retrieve import search, search_rows

DATA_PATH = Path(__file__).parent / "data" / "golden_queries.yml"


def _load_queries() -> list[dict[str, Any]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    entries = doc["queries"]
    for e in entries:
        e.setdefault("top_n", 1)
        e.setdefault("real_index", True)
    return entries


GOLDEN_QUERIES: list[dict[str, Any]] = _load_queries()


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


#: One clean representative clip per golden-query category, plus a couple
#: of rows that exist ONLY to prove a query doesn't accidentally prefer a
#: plausible-looking decoy (sitdown/standup share literal tokens with the
#: get-up group's "up"/"stand"; a bare "down" appears on both the sitdown
#: decoy and would-be fall/crouch/lie modifiers). Every clip_id here is
#: chosen to also satisfy its category's `expected` regex from
#: `golden_queries.yml`, so the same YAML drives both test layers.
GOLDEN_FIXTURE_ROWS = [
    _row("fallandgetup1_subject1",
         ["fall", "and", "get", "up", "1", "subject", "1"]),
    _row("walk1_subject2", ["walk", "1", "subject", "2"]),
    _row("run1_subject3", ["run", "1", "subject", "3"]),
    _row("jump1_subject5", ["jump", "1", "subject", "5"]),
    _row("sitdown1_subject6", ["sit", "down", "1", "subject", "6"]),
    _row("standup1_subject7", ["stand", "up", "1", "subject", "7"]),
    _row("crouch1_subject8", ["crouch", "1", "subject", "8"]),
    _row("lie1_subject9", ["lie", "1", "subject", "9"]),
    _row("dance1_subject10", ["dance", "1", "subject", "10"]),
    _row("fight1_subject11", ["fight", "1", "subject", "11"]),
    _row("push1_subject12", ["push", "1", "subject", "12"]),
]


def _real_index_path() -> Path:
    override_root = library.references_root()
    return override_root / library.INDEX_FILENAME


def _matches(expected: str, top_n: int, clip_ids: list[str]) -> bool:
    pat = re.compile(expected, re.IGNORECASE)
    return any(pat.search(cid) for cid in clip_ids[:top_n])


# ── layer 1: always-on fixture index ─────────────────────────────────────
@pytest.mark.parametrize(
    "entry", GOLDEN_QUERIES, ids=[e["query"] for e in GOLDEN_QUERIES])
def test_golden_query_against_fixture(entry: dict[str, Any]) -> None:
    results = search_rows(
        entry["query"], GOLDEN_FIXTURE_ROWS, k=max(entry["top_n"], 3),
        use_llm=False)
    assert results, f"no matches for {entry['query']!r} against fixture index"
    got = [m.clip_id for m in results]
    assert _matches(entry["expected"], entry["top_n"], got), (
        f"query {entry['query']!r} (category={entry['category']}): "
        f"expected top-{entry['top_n']} to match /{entry['expected']}/, "
        f"got {got[:5]}")


# ── layer 2: real on-disk g1 reference library ───────────────────────────
_REAL_INDEX_ENTRIES = [e for e in GOLDEN_QUERIES if e["real_index"]]


@pytest.mark.skipif(
    not _real_index_path().is_file(),
    reason=f"real reference index not found at {_real_index_path()} "
           "(run `sculpt refs ingest`/`sculpt refs index`, or this is a "
           "checkout without the downloaded reference library)")
@pytest.mark.parametrize(
    "entry", _REAL_INDEX_ENTRIES, ids=[e["query"] for e in _REAL_INDEX_ENTRIES])
def test_golden_query_against_real_index(entry: dict[str, Any]) -> None:
    results = search(entry["query"], robot="g1", k=max(entry["top_n"], 5),
                      use_llm=False)
    assert results, f"no matches for {entry['query']!r} against real index"
    got = [m.clip_id for m in results]
    assert _matches(entry["expected"], entry["top_n"], got), (
        f"query {entry['query']!r} (category={entry['category']}): "
        f"expected top-{entry['top_n']} to match /{entry['expected']}/, "
        f"got {got[:5]}")


def test_golden_queries_cover_every_advertised_category() -> None:
    """Guards the YAML itself: the task's 12 named categories must all be
    present at least once, so this suite can't silently shrink."""
    required = {
        "get-up", "fall", "walk", "run-sprint", "jump", "sit",
        "stand-from-sit", "crouch", "ground-lying", "dance", "fight",
        "push",
    }
    present = {e["category"] for e in GOLDEN_QUERIES}
    missing = required - present
    assert not missing, f"golden_queries.yml is missing categories: {missing}"


def test_golden_queries_real_index_gap_is_exactly_sit_categories() -> None:
    """Locks the CURRENT real-index content gap so a future re-ingest that
    adds sit-content is a visible prompt to flip `real_index: true` rather
    than a silent no-op. If this starts failing because the real index
    now HAS sit-adjacent clips, update `golden_queries.yml` (flip the
    flag(s)) rather than this test."""
    excluded = {e["category"] for e in GOLDEN_QUERIES if not e["real_index"]}
    assert excluded == {"sit", "stand-from-sit"}
