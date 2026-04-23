"""Prompt-time KG research.

User types a topic in the UI ("SEA physics parameters", "quadruped
jumping curriculum"); Claude Opus 4.7 returns 5-10 arXiv IDs directly
relevant to that topic. The IDs are then ingested + extracted into the
shared KG so subsequent sculpt runs can cite the new papers.

This module is the "find papers" half only — ingest + extract happen
in the existing `sculptor.kg.ingest.ingest_arxiv` + `extract_all`
pipelines. A deduplication step against the shared KG's `has_paper`
set means repeated research on the same topic never re-fetches.

Entry point
-----------

    research_topic("series-elastic actuator dynamics", max_papers=8)
        -> ResearchResponse(papers=[...], coverage_note="...")

`research_topic` is a blocking Claude call; wrap it in
`asyncio.to_thread` at the backend-job layer.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from sculptor.kg.store import SculptorKG
from sculptor.prompts import load_prompt


log = logging.getLogger(__name__)

_MODEL = "claude-opus-4-7"
_MAX_TOKENS = 2048

# Bare arxiv IDs in YYMM.NNNNN form. No `arXiv:` prefix, no version
# suffix, no URL. The system prompt enforces this but we also hard-
# validate the model's output to avoid polluting the KG with garbage.
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")


class ResearchPaper(BaseModel):
    """One paper returned by the librarian."""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(description="Bare arXiv ID, e.g. '2401.16337'.")
    title: str = Field(description="Paper title.")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = unrelated, 1.0 = directly on-topic.",
    )
    justification: str = Field(
        description="One-sentence reason this paper is relevant.",
    )


class ResearchResponse(BaseModel):
    """Structured output of a single research_topic call."""

    model_config = ConfigDict(extra="ignore")

    papers: list[ResearchPaper] = Field(default_factory=list)
    coverage_note: str = Field(
        default="",
        description=(
            "Free-form note: if fewer than ~5 solid matches, what's "
            "missing and what the user would have to search manually."
        ),
    )
    # Test 1 / Issue D (2026-04-22) — surface the pipeline intermediate
    # counts so the UI can distinguish "Claude returned 0" from "Claude
    # returned 3, all already in KG". Both land at `added: 0` in the
    # current UI toast; without these counters Sam has no signal which
    # stage ate the papers.
    papers_returned_by_claude: int = Field(
        default=0,
        description="Count of papers Claude proposed before normalization.",
    )
    papers_deduped_against_kg: int = Field(
        default=0,
        description="Count dropped because make_paper_id(id) already in KG.",
    )


def _normalize_arxiv_id(s: str) -> Optional[str]:
    """Coerce Claude output to the bare `YYMM.NNNNN` form. Strip
    `arXiv:` prefix + `vN` version suffix (case-insensitive). Return
    None if the residual doesn't match the arxiv ID regex."""
    s = s.strip()
    if not s:
        return None
    s = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"v\d+$", "", s, flags=re.IGNORECASE)
    return s if _ARXIV_RE.match(s) else None


def research_topic(
    topic: str,
    *,
    max_papers: int = 10,
    client=None,
    store: Optional[SculptorKG] = None,
    dedupe_against_kg: bool = True,
) -> ResearchResponse:
    """Ask Claude for arXiv papers relevant to `topic`.

    Parameters
    ----------
    topic : str
        Natural-language topic (1-200 chars).
    max_papers : int
        Soft cap on the number of papers Claude returns. Forwarded to
        the prompt as context; enforced as a hard truncation on the
        validated output.
    client : anthropic.Anthropic | None
        Pre-built Anthropic client. If None, a fresh client is
        created. Requires `ANTHROPIC_API_KEY`.
    store : SculptorKG | None
        KG used for dedup. If None, a default `SculptorKG()` is opened.
    dedupe_against_kg : bool
        When True (default), any paper whose arxiv_id already has a
        node in `store` is filtered out — the caller gets only NEW
        papers to ingest.

    Returns
    -------
    ResearchResponse
        Validated, normalized, and (optionally) deduplicated. Papers
        that fail arxiv-ID normalization are silently dropped with a
        warning.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic must be a non-empty string")
    if len(topic) > 500:
        raise ValueError(f"topic must be ≤ 500 chars (got {len(topic)})")
    if max_papers < 1 or max_papers > 20:
        raise ValueError("max_papers must be in [1, 20]")

    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise RuntimeError(
                "research_topic requires ANTHROPIC_API_KEY to be set."
            )
        import anthropic
        client = anthropic.Anthropic()

    system = load_prompt("research_topic")
    user_msg = (
        f"Topic: {topic}\n"
        f"Return up to {max_papers} papers."
    )

    # `messages.parse` enforces the ResearchResponse schema. Any parse
    # failure raises; we let it bubble so the job handler surfaces it.
    resp = client.messages.parse(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        output_format=ResearchResponse,
    )
    # `messages.parse` returns a ParsedMessage whose parsed payload lives
    # under `.parsed_output` (scans content blocks for the first parsed
    # text block). Previous `.output` was wrong — sibling call sites in
    # extract.py + diagnose.py both use `.parsed_output`. Fixing this
    # unblocks the KG Research-a-topic flow (2026-04-22 Cartpole test #2).
    result = resp.parsed_output  # type: ignore[union-attr]

    # Pipeline counters (Issue D): track how many papers each stage
    # produced so the UI can explain "added 0" vs "Claude returned 0".
    claude_count = len(result.papers)
    log.info(
        "research_topic: Claude returned %d paper(s) for %r",
        claude_count, topic,
    )

    # Post-validation: normalize arxiv IDs, drop malformed entries,
    # optionally filter against the existing KG.
    kept: list[ResearchPaper] = []
    seen_ids: set[str] = set()
    for p in result.papers:
        norm = _normalize_arxiv_id(p.arxiv_id)
        if norm is None:
            log.warning(
                "research_topic: dropping malformed arxiv_id %r (title=%r)",
                p.arxiv_id, p.title,
            )
            continue
        if norm in seen_ids:
            continue
        seen_ids.add(norm)
        kept.append(
            ResearchPaper(
                arxiv_id=norm,
                title=p.title.strip(),
                relevance_score=float(p.relevance_score),
                justification=p.justification.strip(),
            )
        )

    dedup_count = 0
    if dedupe_against_kg:
        owns_store = store is None
        store = store or SculptorKG()
        try:
            from sculptor.kg.schema import make_paper_id

            pre_dedup = len(kept)
            kept = [
                p for p in kept
                if not store.has_node(make_paper_id(p.arxiv_id))
            ]
            dedup_count = pre_dedup - len(kept)
            if dedup_count:
                log.info(
                    "research_topic: %d paper(s) already in KG (after dedupe: %d new)",
                    dedup_count, len(kept),
                )
        finally:
            if owns_store:
                store.close()

    # Honor the soft cap.
    kept.sort(key=lambda p: -p.relevance_score)
    kept = kept[:max_papers]

    return ResearchResponse(
        papers=kept,
        coverage_note=result.coverage_note,
        papers_returned_by_claude=claude_count,
        papers_deduped_against_kg=dedup_count,
    )


def arxiv_ids(papers: Iterable[ResearchPaper]) -> list[str]:
    """Flat list of arxiv IDs, in the order given."""
    return [p.arxiv_id for p in papers]
