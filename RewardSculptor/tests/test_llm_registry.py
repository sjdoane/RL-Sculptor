"""tests/test_llm_registry.py — sculptor.llm role registry + provenance sink.

Covers the two contracts every converted call site relies on:
  1. `model_for` resolution precedence (RS_MODEL_<ROLE> > RS_MODEL_ALL >
     table default) and the author-disjoint review panel.
  2. `log_llm_call` — no-op when unconfigured, full JSONL round-trip when
     configured, and NEVER raising on an unwritable sink (provenance is
     advisory; the loop is not).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from sculptor.llm import (
    LOG_BASENAME,
    REVIEW_PANEL_DEFAULTS,
    ROLE_DEFAULTS,
    llm_log_dir,
    log_llm_call,
    model_for,
    response_text_blocks,
    review_panel_models,
    set_llm_log_dir,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Clear every RS_MODEL_* override and restore the log-dir contextvar,
    so tests neither see the developer's env nor leak a sink."""
    monkeypatch.delenv("RS_MODEL_ALL", raising=False)
    monkeypatch.delenv("RS_MODEL_REVIEW_PANEL", raising=False)
    for role in ROLE_DEFAULTS:
        monkeypatch.delenv(f"RS_MODEL_{role.upper()}", raising=False)
    prev = llm_log_dir()
    yield
    set_llm_log_dir(prev)


# ── model_for resolution ─────────────────────────────────────────────────
def test_model_for_role_defaults():
    for role, default in ROLE_DEFAULTS.items():
        assert model_for(role) == default


def test_model_for_per_role_override(monkeypatch):
    monkeypatch.setenv("RS_MODEL_DIAGNOSE", "claude-test-diagnose")
    assert model_for("diagnose") == "claude-test-diagnose"
    # Sibling roles are untouched by a per-role override.
    assert model_for("edit") == ROLE_DEFAULTS["edit"]


def test_model_for_all_override_and_precedence(monkeypatch):
    monkeypatch.setenv("RS_MODEL_ALL", "claude-test-all")
    assert model_for("edit") == "claude-test-all"
    assert model_for("kg_extract") == "claude-test-all"
    # RS_MODEL_<ROLE> beats RS_MODEL_ALL.
    monkeypatch.setenv("RS_MODEL_EDIT", "claude-test-edit")
    assert model_for("edit") == "claude-test-edit"
    assert model_for("kg_extract") == "claude-test-all"


def test_model_for_unknown_role_never_crashes(monkeypatch):
    # Unknown role → RS_MODEL_ALL, else the edit default (docstring contract).
    assert model_for("not_a_role") == ROLE_DEFAULTS["edit"]
    monkeypatch.setenv("RS_MODEL_ALL", "claude-test-all")
    assert model_for("not_a_role") == "claude-test-all"


def test_calibration_is_model_disjoint_from_metric_gen():
    """§RESEARCH_GAP_ANALYSIS §3.5: the competence-ladder / gaming-archetype
    author must NOT be the same model as the metric author. If they share a
    blind spot, L2 agreement passes without catching it, so the ladder stops
    being independent evidence. This is a real scientific-validity invariant,
    not a style rule — it is the reason `calibration` is pinned off whatever
    the all-in-system default currently is (opus-5 as of 2026-07-25).
    """
    assert model_for("calibration") != model_for("metric_gen")


# ── review panel ─────────────────────────────────────────────────────────
def test_review_panel_default_is_author_disjoint():
    """§LAW 9 blinding by construction: the default panel must never
    contain the metric author's model."""
    panel = review_panel_models()
    assert panel == list(REVIEW_PANEL_DEFAULTS)
    assert model_for("metric_gen") not in panel


def test_review_panel_env_override(monkeypatch):
    monkeypatch.setenv("RS_MODEL_REVIEW_PANEL", "claude-a, claude-b ,claude-c")
    assert review_panel_models() == ["claude-a", "claude-b", "claude-c"]
    # Empty / whitespace-only override falls back to the defaults.
    monkeypatch.setenv("RS_MODEL_REVIEW_PANEL", " , ")
    assert review_panel_models() == list(REVIEW_PANEL_DEFAULTS)


# ── provenance sink ──────────────────────────────────────────────────────
def test_log_llm_call_noop_when_unset(tmp_path):
    set_llm_log_dir(None)
    log_llm_call("edit", "claude-x", system="s", user="u", response_text="r")
    assert not list(tmp_path.rglob(LOG_BASENAME))
    assert llm_log_dir() is None


def test_log_llm_call_round_trip(tmp_path):
    set_llm_log_dir(tmp_path / "iter_0")   # not pre-created — sink mkdirs
    log_llm_call(
        "diagnose", "claude-x",
        system="the system prompt", user="the user prompt",
        response_text="the response",
        meta={"stage": "preliminary", "attempt": 1},
    )
    lines = (tmp_path / "iter_0" / LOG_BASENAME).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["role"] == "diagnose" and rec["model"] == "claude-x"
    assert rec["system"] == "the system prompt"
    assert rec["user"] == "the user prompt"
    assert rec["response"] == "the response"
    for field, text in (("system_sha256", "the system prompt"),
                        ("user_sha256", "the user prompt"),
                        ("response_sha256", "the response")):
        assert rec[field] == hashlib.sha256(text.encode()).hexdigest()
    assert rec["meta"] == {"stage": "preliminary", "attempt": 1}
    assert rec["ts"] > 0
    # A second call APPENDS (one JSON line per call).
    log_llm_call("edit", "claude-y", system="s2", user="u2",
                 response_text="r2")
    lines = (tmp_path / "iter_0" / LOG_BASENAME).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["role"] == "edit"


def test_log_llm_call_never_raises_on_unwritable_sink(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file", encoding="utf-8")
    # Pointing the sink at a FILE path makes mkdir/open fail — the call
    # must swallow it (advisory-by-contract), not raise into the pipeline.
    set_llm_log_dir(blocker)
    log_llm_call("edit", "claude-x", system="s", user="u", response_text="r")
    assert blocker.read_text(encoding="utf-8") == "i am a file"


def test_response_text_blocks_joins_text_and_skips_thinking():
    class _B:
        def __init__(self, type_, text=""):
            self.type, self.text = type_, text

    class _Resp:
        content = [_B("thinking", "hidden"), _B("text", "a"), _B("text", "b")]

    assert response_text_blocks(_Resp()) == "ab"
    assert response_text_blocks(object()) == ""   # no .content → empty
