"""sculptor/kg/query.py — the KG read surface the diagnoser/editor consumes.

Three entry points:
  * query_techniques(failure_modes, domain_filter=None, top_k=5)
      Graph walk. For each named failure mode, list techniques that ADDRESS it
      (optionally restricted to techniques whose papers EVALUATE_ON an env
      whose tags include `domain_filter`).

  * query_semantic(text, top_k=5)
      Sentence-transformer cosine similarity against cached Technique
      description embeddings. Embeddings are populated lazily and persisted
      via `SculptorKG.set_embedding`.

  * cite(arxiv_id) -> str
      Short human-facing citation string.

All three return / use `TechniqueMatch` dataclasses so the downstream editor
sees a uniform shape regardless of which path produced the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sculptor.kg.schema import (
    Environment,
    FailureMode,
    Paper,
    Relation,
    Technique,
    make_failure_mode_id,
)
from sculptor.kg.store import SculptorKG


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Result shape ────────────────────────────────────────────────────────────
@dataclass
class TechniqueMatch:
    technique: Technique
    description: str
    paper_citation: str
    evidence: str
    relevance_score: float
    matched_on: list[str] = field(default_factory=list)  # diag: why it matched
    source_paper_ids: list[str] = field(default_factory=list)


# ── cite() ──────────────────────────────────────────────────────────────────
def cite(arxiv_id: str, *, store: SculptorKG | None = None) -> str:
    """Short human-facing citation string for a paper by arxiv_id.

    Format: 'Author et al. (Year). Title. arXiv:<id>'
    """
    owns_store = store is None
    store = store or SculptorKG()
    try:
        arxiv_id = arxiv_id.strip()
        paper = store.get_node(f"paper:{arxiv_id}")
        if not isinstance(paper, Paper):
            return f"arXiv:{arxiv_id} (unknown)"
        if paper.authors:
            first = paper.authors[0].split(",")[0].split(" ")[-1]  # surname best-effort
            author_part = f"{first} et al." if len(paper.authors) > 1 else first
        else:
            author_part = "Unknown"
        year_part = f" ({paper.year})" if paper.year else ""
        title = paper.title.rstrip(".")
        return f"{author_part}{year_part}. {title}. arXiv:{paper.arxiv_id}"
    finally:
        if owns_store:
            store.close()


# ── query_techniques ────────────────────────────────────────────────────────
def _resolve_failure_modes(
    store: SculptorKG, names: list[str]
) -> dict[str, FailureMode]:
    """Map the caller's strings onto FailureMode nodes (slug-first, then
    substring in name/description) so we're tolerant to synonyms."""
    all_fm: list[FailureMode] = store.find_nodes(kind=FailureMode.kind)  # type: ignore[assignment]
    out: dict[str, FailureMode] = {}
    for raw in names:
        slug_id = make_failure_mode_id(raw)
        hit = store.get_node(slug_id)
        if isinstance(hit, FailureMode):
            out[raw] = hit
            continue
        target = raw.lower().replace("_", " ").replace("-", " ").strip()
        best: Optional[FailureMode] = None
        for fm in all_fm:
            hay = f"{fm.name} {fm.description}".lower()
            if target in hay or any(tok and tok in hay for tok in target.split()):
                best = fm
                break
        if best is not None:
            out[raw] = best
    return out


def _paper_touches_domain(
    store: SculptorKG, paper_id: str, domain: str
) -> bool:
    """True if this paper has an EVALUATES_ON edge to an Environment whose tags
    include the domain string (case-insensitive)."""
    if not domain:
        return True
    target = domain.strip().lower()
    for _, env_id in store.neighbors(paper_id, relation=Relation.EVALUATES_ON, direction="out"):
        env = store.get_node(env_id)
        if isinstance(env, Environment):
            tags = [t.lower() for t in env.tags]
            name = env.name.lower()
            if target in tags or target in name:
                return True
    return False


def query_techniques(
    failure_modes: list[str],
    domain_filter: str | None = None,
    top_k: int = 5,
    *,
    store: SculptorKG | None = None,
) -> list[TechniqueMatch]:
    """Techniques that ADDRESS any of `failure_modes`, optionally filtered by
    the domain tag of the introducing paper's evaluation environment.

    Ranking: more matched failure modes first, then introducing paper's year
    (newer first), then technique name.
    """
    owns_store = store is None
    store = store or SculptorKG()
    try:
        fm_map = _resolve_failure_modes(store, failure_modes)
        if not fm_map:
            return []

        # For each FailureMode, find all Techniques with ADDRESSES edges into it.
        # (Edge direction: Technique -> FailureMode.)
        tech_hits: dict[str, dict[str, Any]] = {}
        for raw, fm in fm_map.items():
            for edge, tech_id in store.neighbors(
                fm.id, relation=Relation.ADDRESSES, direction="in"
            ):
                tech = store.get_node(tech_id)
                if not isinstance(tech, Technique):
                    continue
                bucket = tech_hits.setdefault(tech_id, {
                    "technique": tech,
                    "matched_raw": set(),
                    "matched_fms": set(),
                    "evidence": "",
                    "source_paper_ids": set(),
                })
                bucket["matched_raw"].add(raw)
                bucket["matched_fms"].add(fm.name)
                if not bucket["evidence"] and edge.data.get("evidence"):
                    bucket["evidence"] = edge.data["evidence"]
                src = edge.data.get("source_paper_id")
                if src:
                    bucket["source_paper_ids"].add(src)

        # Resolve introducing paper(s) via INTRODUCES inbound edges; fall back
        # to whichever paper provided the ADDRESSES edge.
        results: list[TechniqueMatch] = []
        for tech_id, bucket in tech_hits.items():
            tech: Technique = bucket["technique"]
            intro_papers: list[Paper] = []
            for _, paper_id in store.neighbors(
                tech_id, relation=Relation.INTRODUCES, direction="in"
            ):
                p = store.get_node(paper_id)
                if isinstance(p, Paper):
                    intro_papers.append(p)
            if not intro_papers:
                for pid in bucket["source_paper_ids"]:
                    p = store.get_node(pid)
                    if isinstance(p, Paper):
                        intro_papers.append(p)

            if domain_filter:
                keep = any(
                    _paper_touches_domain(store, p.id, domain_filter)
                    for p in intro_papers
                )
                if not keep:
                    continue

            newest_paper = max(
                intro_papers, key=lambda p: (p.year or 0, p.arxiv_id), default=None)
            citation = cite(newest_paper.arxiv_id, store=store) if newest_paper else "(unknown paper)"

            score = float(len(bucket["matched_fms"]))
            if newest_paper and newest_paper.year:
                # Tiny tie-breaker so newer papers rank higher at the same score.
                score += min((newest_paper.year - 2000) / 1000.0, 0.05)

            results.append(TechniqueMatch(
                technique=tech,
                description=tech.description,
                paper_citation=citation,
                evidence=bucket["evidence"],
                relevance_score=score,
                matched_on=sorted(bucket["matched_fms"]),
                source_paper_ids=sorted(bucket["source_paper_ids"]),
            ))

        results.sort(
            key=lambda m: (-m.relevance_score, m.technique.name.lower()))
        return results[:top_k]
    finally:
        if owns_store:
            store.close()


# ── Semantic query ──────────────────────────────────────────────────────────
_EMBEDDER_CACHE: dict[str, Any] = {}


def _get_embedder(model_name: str = EMBEDDING_MODEL):
    """Lazy-load and memoize the SentenceTransformer model."""
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def _embed_text(text: str, model_name: str = EMBEDDING_MODEL):
    import numpy as np

    embedder = _get_embedder(model_name)
    vec = embedder.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _ensure_technique_embeddings(
    store: SculptorKG, model_name: str = EMBEDDING_MODEL
) -> list[tuple[Technique, Any]]:
    """Embed every Technique description that doesn't have a cached embedding
    yet, return (technique, vector) for all techniques."""
    import numpy as np

    techniques: list[Technique] = store.find_nodes(kind=Technique.kind)  # type: ignore[assignment]
    need: list[Technique] = [
        t for t in techniques
        if not store.has_embedding(t.id, model_name) and (t.description or t.name)
    ]
    if need:
        embedder = _get_embedder(model_name)
        texts = [f"{t.name}. {t.description}".strip(". ") for t in need]
        vecs = embedder.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        for t, v in zip(need, vecs):
            store.set_embedding(t.id, model_name, v)

    out: list[tuple[Technique, Any]] = []
    for t in techniques:
        v = store.get_embedding(t.id, model_name)
        if v is not None:
            out.append((t, v))
    return out


def query_semantic(
    text: str,
    top_k: int = 5,
    *,
    store: SculptorKG | None = None,
    model_name: str = EMBEDDING_MODEL,
    min_similarity: float = 0.0,
) -> list[TechniqueMatch]:
    """Rank Techniques by cosine similarity between `text` and their descriptions.

    `min_similarity`: floor for the cosine score (0.0 by default →
    return top-k regardless). Prompt-edit callers should pass ~0.35 to
    filter out tangentially-related matches — the KG's 46-paper seed is
    locomotion-heavy, so a Cartpole query otherwise returns the 5
    closest humanoid papers at sim=0.1-0.3 and Claude dutifully cites
    them. See CONTEXT.md 2026-04-22 Test 1 Issue G.
    """
    import numpy as np

    owns_store = store is None
    store = store or SculptorKG()
    try:
        pool = _ensure_technique_embeddings(store, model_name)
        if not pool:
            return []

        qv = _embed_text(text, model_name)
        # Vectors are L2-normalized, so dot product == cosine similarity.
        scored = []
        for tech, v in pool:
            sim = float(np.dot(qv, v))
            if sim < min_similarity:
                continue
            scored.append((sim, tech))
        scored.sort(key=lambda x: -x[0])

        results: list[TechniqueMatch] = []
        for sim, tech in scored[:top_k]:
            intro_papers: list[Paper] = []
            evidence = ""
            for edge, paper_id in store.neighbors(
                tech.id, relation=Relation.INTRODUCES, direction="in"
            ):
                p = store.get_node(paper_id)
                if isinstance(p, Paper):
                    intro_papers.append(p)
                    if not evidence and edge.data.get("evidence"):
                        evidence = edge.data["evidence"]
            newest = max(
                intro_papers, key=lambda p: (p.year or 0, p.arxiv_id), default=None)
            citation = cite(newest.arxiv_id, store=store) if newest else "(unknown paper)"

            results.append(TechniqueMatch(
                technique=tech,
                description=tech.description,
                paper_citation=citation,
                evidence=evidence,
                relevance_score=sim,
                matched_on=["semantic"],
                source_paper_ids=[p.id for p in intro_papers],
            ))
        return results
    finally:
        if owns_store:
            store.close()
