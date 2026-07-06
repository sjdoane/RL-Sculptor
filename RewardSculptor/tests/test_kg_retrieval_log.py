"""tests/test_kg_retrieval_log.py — §Agentic-data upgrade 3: retrieval
trajectory logging.

Mirrors tests/test_llm_registry.py's provenance-sink coverage: round-trip
when configured, no-op when unconfigured, NEVER raises on an unwritable
sink, and the `out_dir=None` -> `sculptor.llm.llm_log_dir()` fallback.
"""
from __future__ import annotations

import json

import pytest

from sculptor.kg.retrieval_log import LOG_BASENAME, log_retrieval
from sculptor.kg.query import TechniqueMatch
from sculptor.kg.cases import CaseMatch
from sculptor.kg.schema import RunCase, Technique
from sculptor.llm import llm_log_dir, set_llm_log_dir


@pytest.fixture(autouse=True)
def _isolated_log_dir():
    """Restore the llm.py contextvar so tests don't leak a sink into
    each other (same pattern as test_llm_registry.py's fixture)."""
    prev = llm_log_dir()
    yield
    set_llm_log_dir(prev)


def _tech_match(tech_id: str, score: float) -> TechniqueMatch:
    return TechniqueMatch(
        technique=Technique(id=tech_id, name=tech_id),
        description="", paper_citation="", evidence="",
        relevance_score=score,
    )


def _case_match(case_id: str, score: float) -> CaseMatch:
    return CaseMatch(case=RunCase(id=case_id, task="kick"), relevance_score=score)


# ── explicit out_dir ───────────────────────────────────────────────────────
def test_log_retrieval_round_trip_with_explicit_out_dir(tmp_path):
    matches = [_tech_match("technique:rsi", 0.81), _tech_match("technique:shaping", 0.42)]
    log_retrieval("diagnose", "kick forward jump", matches, out_dir=tmp_path / "iter_3")

    lines = (tmp_path / "iter_3" / LOG_BASENAME).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["decision"] == "diagnose"
    assert rec["query"] == "kick forward jump"
    assert rec["node_ids"] == ["technique:rsi", "technique:shaping"]
    assert rec["scores"] == [0.81, 0.42]
    assert rec["ts"] > 0


def test_log_retrieval_appends_one_line_per_call(tmp_path):
    out = tmp_path / "iter_0"
    log_retrieval("diagnose", "q1", [_tech_match("technique:a", 0.5)], out_dir=out)
    log_retrieval("decompose", "q2", [], out_dir=out)
    lines = (out / LOG_BASENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "diagnose"
    assert json.loads(lines[1])["decision"] == "decompose"
    assert json.loads(lines[1])["node_ids"] == []


def test_log_retrieval_extracts_case_matches():
    """CaseMatch (not TechniqueMatch) node ids are pulled via `.case.id`."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "iter_1"
        matches = [_case_match("case:x", 0.9), _case_match("case:y", 0.3)]
        log_retrieval("diagnose", "q", matches, out_dir=out)
        lines = (out / LOG_BASENAME).read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        assert rec["node_ids"] == ["case:x", "case:y"]
        assert rec["scores"] == [0.9, 0.3]


# ── llm_log_dir() fallback ───────────────────────────────────────────────
def test_log_retrieval_falls_back_to_llm_log_dir_when_out_dir_none(tmp_path):
    set_llm_log_dir(tmp_path / "mission_dir")
    log_retrieval("decompose", "goal text", [_tech_match("technique:a", 0.7)], out_dir=None)
    lines = (tmp_path / "mission_dir" / LOG_BASENAME).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["decision"] == "decompose"
    assert rec["query"] == "goal text"


def test_log_retrieval_noop_when_out_dir_none_and_no_llm_log_dir_configured(tmp_path):
    set_llm_log_dir(None)
    log_retrieval("decompose", "goal text", [_tech_match("technique:a", 0.7)], out_dir=None)
    assert not list(tmp_path.rglob(LOG_BASENAME))


# ── never raises ─────────────────────────────────────────────────────────
def test_log_retrieval_never_raises_on_unwritable_dir(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file", encoding="utf-8")
    # Pointing out_dir at a FILE path makes mkdir/open fail — must swallow.
    log_retrieval("diagnose", "q", [_tech_match("technique:a", 0.5)], out_dir=blocker)
    assert blocker.read_text(encoding="utf-8") == "i am a file"


def test_log_retrieval_never_raises_on_malformed_matches(tmp_path):
    """`matches` accepts anything duck-typed close to a Match — an object
    with neither `.technique` nor `.case` nor `.id` must not raise; it
    just yields a None node_id."""
    class _Weird:
        relevance_score = 0.5

    out = tmp_path / "iter_weird"
    log_retrieval("diagnose", "q", [_Weird(), None, 42], out_dir=out)
    lines = (out / LOG_BASENAME).read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert rec["node_ids"] == [None, None, None]
    assert rec["scores"][0] == 0.5


def test_log_retrieval_never_raises_when_matches_is_none(tmp_path):
    out = tmp_path / "iter_none"
    log_retrieval("diagnose", "q", None, out_dir=out)
    lines = (out / LOG_BASENAME).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["node_ids"] == []
