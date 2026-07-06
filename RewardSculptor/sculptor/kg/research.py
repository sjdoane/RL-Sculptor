"""Prompt-time KG research.

User types a topic in the UI ("SEA physics parameters", "quadruped
jumping curriculum"); Claude (the registry's "kg_research" role,
`sculptor.llm.model_for`) returns 5-10 arXiv IDs directly
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
from sculptor.llm import log_llm_call, model_for, response_text_blocks
from sculptor.prompts import load_prompt


log = logging.getLogger(__name__)

_MODEL = model_for("kg_research")
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
    # 2026-04-23: Claude was returning real-but-wrong arxiv IDs —
    # e.g. `2407.14795` for "bipedal robot kicking" which is actually
    # a Persian-text error-correction paper. Claude recalls that SOME
    # arxiv ID exists and fabricates a justification matching the
    # topic, but the paper at that ID is unrelated. Post-fix we fetch
    # each ID's real metadata and drop papers whose title+abstract
    # don't semantically match the topic.
    papers_rejected_off_topic: int = Field(
        default=0,
        description=(
            "Count dropped because arxiv metadata showed the paper is "
            "unrelated to the topic (Claude returned a valid ID for "
            "the wrong paper)."
        ),
    )


# Minimum cosine similarity between the topic and a paper's
# (real) title+abstract for the paper to pass the verification
# filter. Calibrated against Sam's 2026-04-23 "bipedal robot
# kicking" session (using all-MiniLM-L6-v2):
#
#   Persian spelling correction (off)      sim=0.05  DROP
#   SEFL educational feedback (off)        sim=0.13  DROP
#   DeepMimic (on, foundational imitation) sim=0.18  KEEP
#   Actuator-Constrained RL (adj)          sim=0.35  KEEP
#   RMA Rapid Motor Adaptation (adj)       sim=0.38  KEEP
#   Expressive WBC for Humanoids (on)      sim=0.42  KEEP
#   BeyondMimic (on)                       sim=0.42  KEEP
#   Humanoid Parkour (on)                  sim=0.48  KEEP
#
# Threshold 0.15 cleanly separates the two confirmed hallucinations
# from the DeepMimic-style low-overlap-vocabulary but on-topic
# papers. Tighter thresholds false-positive DeepMimic; looser ones
# keep SEFL. The signal isn't perfect — if a clearer test emerges,
# re-tune. Users can also raise max_papers to widen the net.
_MIN_TOPIC_SIMILARITY = 0.15


def _fetch_arxiv_metadata_batch(
    arxiv_ids: list[str], *, timeout_s: float = 30.0,
) -> dict[str, Optional[dict]]:
    """Batch-fetch title + abstract for a list of arxiv IDs.

    Returns `{arxiv_id: {"title": ..., "abstract": ...} | None}`.
    None values mean the API didn't return that paper (rate-limited,
    ID doesn't exist, or transient failure); caller decides whether
    to keep or drop the entry.
    """
    out: dict[str, Optional[dict]] = {aid: None for aid in arxiv_ids}
    if not arxiv_ids:
        return out
    try:
        import arxiv
        import socket

        socket.setdefaulttimeout(timeout_s)
        try:
            client = arxiv.Client(
                page_size=max(len(arxiv_ids), 1),
                delay_seconds=3.0, num_retries=1,
            )
            search = arxiv.Search(id_list=list(arxiv_ids))
            for result in client.results(search):
                entry_id = str(result.entry_id or "")
                # entry_id looks like `http://arxiv.org/abs/2407.14795v1`
                aid = entry_id.rsplit("/", 1)[-1]
                aid = re.sub(r"v\d+$", "", aid, flags=re.IGNORECASE)
                if aid in out:
                    out[aid] = {
                        "title": str(result.title or "").strip(),
                        "abstract": str(result.summary or "").strip(),
                    }
        finally:
            socket.setdefaulttimeout(None)
    except Exception as e:  # noqa: BLE001 — arxiv rate-limits are common; don't hard-fail research
        log.warning(
            "research_topic: arxiv batch metadata fetch failed "
            "(%s); skipping off-topic verification",
            f"{type(e).__name__}: {e}",
        )
    return out


def _verify_topic_match(
    topic: str, papers: list["ResearchPaper"],
    *, min_similarity: float = _MIN_TOPIC_SIMILARITY,
    metadata_fn=None,
) -> tuple[list["ResearchPaper"], int]:
    """Drop papers whose real arxiv title+abstract is semantically
    distant from `topic`. Returns (kept, dropped_count).

    `metadata_fn` is injected for tests; production default is the
    batch arxiv fetch defined above. When metadata fetch fails or
    returns None for a given paper, we KEEP it (fail-open) — an
    arxiv outage shouldn't wipe out the user's research query.
    """
    if not papers:
        return papers, 0

    fetch = metadata_fn or _fetch_arxiv_metadata_batch
    metadata_by_id = fetch([p.arxiv_id for p in papers])

    try:
        from sculptor.kg.query import _embed_text
        topic_vec = _embed_text(topic)
    except Exception as e:  # noqa: BLE001 — embedder unavailable, skip verification
        log.warning(
            "research_topic: embedder unavailable (%s); skipping "
            "off-topic verification",
            f"{type(e).__name__}: {e}",
        )
        return papers, 0

    import numpy as np

    kept: list[ResearchPaper] = []
    dropped = 0
    for p in papers:
        meta = metadata_by_id.get(p.arxiv_id)
        if meta is None:
            # Fail-open: keep the paper, log so user sees the gap.
            log.info(
                "research_topic: %s arxiv metadata unavailable; "
                "keeping without topic-match verification",
                p.arxiv_id,
            )
            kept.append(p)
            continue
        real_title = meta.get("title", "") or ""
        real_abstract = meta.get("abstract", "") or ""
        # Embed title + first ~400 chars of abstract — long abstracts
        # dilute the signal and cost no-op encoding time.
        text_for_embed = f"{real_title}. {real_abstract[:400]}".strip(". ")
        if not text_for_embed:
            kept.append(p)
            continue
        paper_vec = _embed_text(text_for_embed)
        sim = float(np.dot(topic_vec, paper_vec))
        if sim < min_similarity:
            log.warning(
                "research_topic: dropping %s off-topic (sim=%.2f < %.2f): "
                "real_title=%r; Claude claimed=%r",
                p.arxiv_id, sim, min_similarity,
                real_title[:80], p.title[:80],
            )
            dropped += 1
            continue
        # Replace Claude's (possibly hallucinated) title with the real one.
        p.title = real_title or p.title
        kept.append(p)
    return kept, dropped


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
    log_llm_call(
        "kg_research", _MODEL, system=system, user=user_msg,
        response_text=response_text_blocks(resp),
        usage=getattr(resp, "usage", None))
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

    # 2026-04-23 off-topic-hallucination guard. Claude recalls valid
    # arxiv IDs but sometimes attaches the wrong paper to a topic
    # ("2407.14795 is about bipedal robot kicking" — actually Persian
    # spell-correction). Fetch the real title+abstract from arxiv,
    # embed vs. topic, drop below-threshold hits. Fail-open on arxiv
    # outage or embedder failure.
    kept, off_topic_dropped = _verify_topic_match(topic, kept)
    if off_topic_dropped:
        log.info(
            "research_topic: %d paper(s) dropped as off-topic after "
            "arxiv metadata verification",
            off_topic_dropped,
        )

    # Honor the soft cap.
    kept.sort(key=lambda p: -p.relevance_score)
    kept = kept[:max_papers]

    return ResearchResponse(
        papers=kept,
        coverage_note=result.coverage_note,
        papers_returned_by_claude=claude_count,
        papers_deduped_against_kg=dedup_count,
        papers_rejected_off_topic=off_topic_dropped,
    )


def arxiv_ids(papers: Iterable[ResearchPaper]) -> list[str]:
    """Flat list of arxiv IDs, in the order given."""
    return [p.arxiv_id for p in papers]
