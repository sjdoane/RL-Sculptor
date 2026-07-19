"""Prompt-time knowledge-graph grounding for the world author.

Retrieval-only bridge between the shared knowledge graph and
environment authoring (plan item 3, ENV_AUTHORING_ARCHITECTURE §KG
integration): given the user's mission prompt, surface the most
relevant Techniques, FailureModes, and Papers so that

- the authoring model receives a compact evidence block
  (``request["kg_grounding"]``) it can cite while choosing terrain,
  obstacle, object, and curriculum parameters, and
- the authored artifacts carry a durable provenance trail —
  ``meta.grounding`` in both WorldSpec and TaskSpec holds the matched
  KG node IDs (the validator's contract), so any later reader can ask
  the graph exactly which prior art shaped this world.

Failure policy: grounding is strictly best-effort. A missing store, an
unavailable embedder, or any retrieval error degrades to *ungrounded
authoring* with one warning — it must never block or fail the author.
The deterministic offline author and all validation gates are upstream
of and independent from anything returned here.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Sequence

log = logging.getLogger(__name__)

#: Retrieval floors follow sculptor.kg.query guidance: prompt-time
#: callers pass ~0.35 so a corpus dominated by one domain cannot
#: surface tangential matches for an out-of-domain prompt.
DEFAULT_MIN_SIMILARITY = 0.35

#: Author intent -> Paper campaign-tag filter (tags as curated in the
#: env-authoring seeds corpus). Unknown/unmatched intents apply no tag
#: filter rather than guessing.
_INTENT_PAPER_TAGS: dict[str, list[str]] = {
    "uneven_terrain": ["terrain"],
    "parkour": ["parkour"],
    "object_to_region": ["objects"],
    "robot_to_region": [],
}

_GUIDANCE_MAX_CHARS = 280


@dataclasses.dataclass(frozen=True)
class GroundingItem:
    """One retrieved KG node, ready for the model block and the ledger."""

    node_id: str
    kind: str  # "technique" | "failure_mode" | "paper"
    name: str
    guidance: str
    citation: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _trim(text: str) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= _GUIDANCE_MAX_CHARS:
        return text
    return text[: _GUIDANCE_MAX_CHARS - 1].rstrip() + "…"


def _paper_tags_for_prompt(prompt: str) -> list[str] | None:
    """Map the author's intent classifier onto corpus tag filters."""
    from sculptor.world.author import AuthoringError, _intent

    try:
        return _INTENT_PAPER_TAGS.get(_intent(prompt))
    except AuthoringError:
        return None


def gather_grounding(
    prompt: str,
    *,
    store: Any = None,
    top_k_techniques: int = 4,
    top_k_failure_modes: int = 3,
    top_k_papers: int = 3,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> tuple[GroundingItem, ...]:
    """Retrieve prompt-relevant KG nodes; empty tuple on any failure.

    Never raises and never creates state beyond the store's own lazy
    embedding maintenance. Authoring must behave identically (minus the
    evidence block) when this returns ``()``.
    """
    try:
        return _gather(
            prompt,
            store,
            top_k_techniques=top_k_techniques,
            top_k_failure_modes=top_k_failure_modes,
            top_k_papers=top_k_papers,
            min_similarity=min_similarity,
        )
    except Exception as exc:  # noqa: BLE001 — grounding is best-effort by contract
        log.warning(
            "world.grounding: KG grounding unavailable (%s: %s); "
            "authoring proceeds ungrounded",
            type(exc).__name__, exc,
        )
        return ()


def _gather(
    prompt: str,
    store: Any,
    *,
    top_k_techniques: int,
    top_k_failure_modes: int,
    top_k_papers: int,
    min_similarity: float,
) -> tuple[GroundingItem, ...]:
    from sculptor.kg import query as kg_query
    from sculptor.kg.schema import FailureMode
    from sculptor.kg.store import SculptorKG

    prompt = str(prompt or "").strip()
    if not prompt:
        return ()

    owns_store = store is None
    store = store or SculptorKG()
    items: list[GroundingItem] = []
    seen: set[str] = set()

    def _add(item: GroundingItem) -> None:
        if item.node_id not in seen:
            seen.add(item.node_id)
            items.append(item)

    try:
        if top_k_techniques > 0:
            for match in kg_query.query_semantic(
                prompt, top_k_techniques, store=store,
                min_similarity=min_similarity,
            ):
                _add(GroundingItem(
                    node_id=match.technique.id,
                    kind="technique",
                    name=match.technique.name,
                    guidance=_trim(match.description),
                    citation=match.paper_citation,
                    score=round(float(match.relevance_score), 4),
                ))

        if top_k_failure_modes > 0:
            for node_id in kg_query.resolve_failure_modes_semantic(
                store, [prompt], top_k_per=top_k_failure_modes,
            ):
                node = store.get_node(node_id)
                if isinstance(node, FailureMode):
                    _add(GroundingItem(
                        node_id=node.id,
                        kind="failure_mode",
                        name=node.name,
                        guidance=_trim(node.description),
                    ))

        if top_k_papers > 0:
            for match in kg_query.query_papers(
                prompt, top_k_papers, store=store,
                tags=_paper_tags_for_prompt(prompt) or None,
                min_similarity=min_similarity,
            ):
                paper = match.paper
                year = f" ({paper.year})" if paper.year else ""
                _add(GroundingItem(
                    node_id=paper.id,
                    kind="paper",
                    name=paper.title,
                    guidance=_trim(paper.rationale or paper.abstract),
                    citation=f"{paper.title}{year}",
                    score=round(float(match.relevance_score), 4),
                ))
    finally:
        if owns_store:
            store.close()
    return tuple(items)


def grounding_ids(items: Sequence[GroundingItem]) -> list[str]:
    """Node IDs for the WorldSpec/TaskSpec ``meta.grounding`` ledger."""
    out: list[str] = []
    for item in items:
        if item.node_id not in out:
            out.append(item.node_id)
    return out


def grounding_context(items: Sequence[GroundingItem]) -> list[dict[str, Any]]:
    """Evidence block for the authoring model request (rich, non-ledger)."""
    return [item.to_dict() for item in items]
