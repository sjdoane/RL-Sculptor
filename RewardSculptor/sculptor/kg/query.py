"""sculptor/kg/query.py — the KG read surface the diagnoser/editor consumes.

Entry points:
  * query_techniques(failure_modes, domain_filter=None, top_k=5,
                      extra_failure_node_ids=None)
      Graph walk. For each named failure mode (plus any already-resolved
      FailureMode node ids passed via `extra_failure_node_ids` — e.g. from
      `resolve_failure_modes_semantic`), list techniques that ADDRESS it
      (optionally restricted to techniques whose papers EVALUATE_ON an env
      whose tags include `domain_filter`).

  * query_semantic(text, top_k=5)
      Sentence-transformer cosine similarity against cached Technique
      description embeddings. Embeddings are populated lazily and persisted
      via `SculptorKG.set_embedding`.

  * resolve_failure_modes_semantic(store, descriptors, top_k_per=2,
                                    min_similarity=0.45)
      Sentence-transformer cosine similarity between free-text failure
      descriptors and cached FailureMode name+description embeddings —
      the semantic counterpart to `_resolve_failure_modes`'s fixed-enum
      fuzzy resolution. Returns FailureMode node ids for use as
      `query_techniques`'s `extra_failure_node_ids`.

  * cite(arxiv_id) -> str
      Short human-facing citation string.

`query_techniques` / `query_semantic` return `TechniqueMatch` dataclasses so
the downstream editor sees a uniform shape regardless of which path
produced the candidate.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

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


#: Default cosine floor for PROMPT-FEEDING semantic queries (§Ship 31).
#: Below ~0.35 the match is tangential and Claude dutifully cites it
#: anyway (CONTEXT 2026-04-22 Issue G). edit.py adopted this in Ship 8;
#: diagnose/decompose now share the same constant — an unfloored
#: query_semantic is for exploration tools only, never for prompts.
DEFAULT_MIN_PROMPT_SIMILARITY = 0.35

#: §Agentic-data upgrade 2 (usage-based enrichment): cap on the
#: useful_citations ranking boost. Retrieval must stay RELEVANCE-dominated
#: — a technique cited in ten past "helped" edits is a mild tie-breaker
#: signal, not grounds to outrank something the current failure/behavior
#: text actually matches. 0.05 is small next to the ~0.35-1.0 range of a
#: real relevance/similarity score, and log1p makes the boost diminish
#: fast (citation count 1 -> ~0.007, 10 -> ~0.024, 100 -> ~0.046,
#: asymptoting at the 0.05 cap) so no amount of accumulated usage can ever
#: let a barely-relevant technique leapfrog a strongly-relevant one.
USEFUL_CITATION_BOOST_CAP = 0.05
_USEFUL_CITATION_BOOST_SCALE = 0.01


def _citation_boost(useful_citations: int) -> float:
    """CAPPED ranking boost from usage history. Applied to the score used
    for ORDERING only — never to gate/floor thresholds, which must be
    checked on the raw relevance/similarity score before this is added
    (otherwise a below-floor match could be boosted back above it)."""
    if not useful_citations or useful_citations <= 0:
        return 0.0
    return min(
        USEFUL_CITATION_BOOST_CAP,
        _USEFUL_CITATION_BOOST_SCALE * math.log1p(useful_citations),
    )


#: §KG-retrieval fix 4 (outcome-stats ranking): cap on the per-FailureMode
#: helped/regressed ranking adjustment — same "small, ORDERING-only tie-
#: breaker" contract as `USEFUL_CITATION_BOOST_CAP` above, deliberately
#: scaled a bit larger (0.15 vs 0.05) because this signal is SCOPED to the
#: queried failure mode(s) rather than a global usage count, so it is a
#: more targeted (and thus more trustworthy) tie-breaker.
OUTCOME_STATS_ADJUSTMENT_CAP = 0.15
_OUTCOME_STATS_ADJUSTMENT_SCALE = 0.03


def _outcome_stats_adjustment(
    outcome_stats: dict | None, queried_failure_mode_ids: set[str],
) -> float:
    """Net helped-minus-regressed signal from THIS project's own RunCase
    verdicts, restricted to the failure mode(s) actually being queried
    (`query_techniques`'s enum-resolved + `extra_failure_node_ids`).
    Applied to the ORDERING score only, alongside `_citation_boost` — never
    to a similarity floor. Deterministic: a technique with no recorded
    stats for the queried failure modes returns 0.0 (no behavior change)."""
    if not outcome_stats or not queried_failure_mode_ids:
        return 0.0
    net = 0
    for fm_id in queried_failure_mode_ids:
        entry = outcome_stats.get(fm_id)
        if not entry:
            continue
        net += int(entry.get("helped", 0) or 0) - int(entry.get("regressed", 0) or 0)
    return max(
        -OUTCOME_STATS_ADJUSTMENT_CAP,
        min(OUTCOME_STATS_ADJUSTMENT_CAP, _OUTCOME_STATS_ADJUSTMENT_SCALE * net),
    )


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
    """Map the caller's strings onto FailureMode nodes: exact slug,
    then full-phrase containment, then a SCORED token fallback.

    §Ship 31 grounding fix: the old fallback accepted the FIRST node
    matching ANY single token — with 325 FailureModes,
    "reward_saturation" resolved to an effectively arbitrary node
    containing the word "reward", silently mis-grounding
    query_techniques. Now: candidates are ranked by matched-token
    count (full-phrase hits rank above all token hits), a token
    fallback requires a MAJORITY of the target's tokens, and ties
    break deterministically by (name length, name) — tightest match
    wins, ordering is stable across runs.
    """
    all_fm: list[FailureMode] = store.find_nodes(kind=FailureMode.kind)  # type: ignore[assignment]
    all_fm = sorted(all_fm, key=lambda f: ((f.name or ""), f.id))
    out: dict[str, FailureMode] = {}
    for raw in names:
        slug_id = make_failure_mode_id(raw)
        hit = store.get_node(slug_id)
        if isinstance(hit, FailureMode):
            out[raw] = hit
            continue
        target = raw.lower().replace("_", " ").replace("-", " ").strip()
        tokens = [t for t in target.split() if t]
        if not tokens:
            continue
        need = max(1, (len(tokens) + 1) // 2)   # majority of tokens
        best: Optional[FailureMode] = None
        best_key: tuple = ()
        for fm in all_fm:
            hay = f"{fm.name} {fm.description}".lower()
            phrase = target in hay
            n_tok = sum(1 for t in tokens if t in hay)
            if not phrase and n_tok < need:
                continue
            # Rank: phrase match beats token matches; more tokens beat
            # fewer; shorter (tighter) names beat longer. Remaining
            # ties go to the alphabetically-first node — all_fm is
            # name-sorted and `>` is strict, so the first seen wins.
            key = (1 if phrase else 0, n_tok, -len(fm.name or ""))
            if best is None or key > best_key:
                best, best_key = fm, key
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
    extra_failure_node_ids: list[str] | None = None,
) -> list[TechniqueMatch]:
    """Techniques that ADDRESS any of `failure_modes`, optionally filtered by
    the domain tag of the introducing paper's evaluation environment.

    `extra_failure_node_ids`: additional FailureMode node ids (already
    resolved — e.g. via `resolve_failure_modes_semantic` matching free-text
    failure descriptors) merged in alongside the enum-resolved nodes from
    `failure_modes`, before the ADDRESSES walk below. Deduped by node id;
    unknown / non-FailureMode ids are silently skipped. Ranking is
    otherwise unchanged — a technique simply gets credit for however many
    FailureMode nodes it addresses, enum- or descriptor-resolved alike.

    Ranking: more matched failure modes first, then introducing paper's year
    (newer first), then technique name.
    """
    owns_store = store is None
    store = store or SculptorKG()
    try:
        fm_map = _resolve_failure_modes(store, failure_modes)
        if extra_failure_node_ids:
            existing_ids = {fm.id for fm in fm_map.values()}
            for node_id in extra_failure_node_ids:
                if not node_id or node_id in existing_ids:
                    continue
                node = store.get_node(node_id)
                if isinstance(node, FailureMode):
                    fm_map[node_id] = node
                    existing_ids.add(node_id)
        if not fm_map:
            return []
        # §KG-retrieval fix 4: the full set of FailureMode ids this call is
        # querying for (enum-resolved ∪ extra_failure_node_ids) — used
        # below to scope the outcome-stats ranking adjustment to what was
        # actually asked about, not a technique's stats against every
        # failure mode it has ever touched.
        queried_fm_ids = {fm.id for fm in fm_map.values()}

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
            # §Agentic-data upgrade 2: capped usage-based boost, ORDERING only.
            # query_techniques has no similarity floor to protect (the graph
            # walk is a hard ADDRESSES-edge match, not a threshold), so the
            # boost is simply added to the tie-break score here.
            score += _citation_boost(tech.useful_citations)
            # §KG-retrieval fix 4: capped outcome-stats adjustment, ORDERING
            # only — same rationale as the citation boost above.
            score += _outcome_stats_adjustment(
                getattr(tech, "outcome_stats", None), queried_fm_ids)

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
# Serializes concurrent first-time loads of the SentenceTransformer model.
# The reward-sculptor-ui's Ship-10 prewarm hook fires a background
# `_load_embedder` at uvicorn boot AND the reward-prompt worker thread
# calls `_get_embedder` when processing a user edit. Without this lock,
# two threads enter `SentenceTransformer(...)` concurrently on a cold
# cache; on WSL2 the huggingface_hub cache `FileLock` + torch CUDA init
# combination deadlocks under that race (observed symptom: reward-prompt
# hangs indefinitely at the KG preview step, times out at 300s). The
# lock lets the prewarm hold exclusive access while loading; the worker
# waits, then reads the cached embedder on the second pass.
_EMBEDDER_LOAD_LOCK = threading.Lock()


def _get_embedder(model_name: str = EMBEDDING_MODEL):
    """Lazy-load and memoize the SentenceTransformer model.

    Thread-safe: concurrent callers (prewarm + reward-prompt worker)
    serialize on `_EMBEDDER_LOAD_LOCK` so only one thread invokes
    `SentenceTransformer(...)`. Subsequent callers read the cached
    embedder without taking the lock (fast path).

    Uses `local_files_only=True` when the model is already in the HF
    cache. This bypasses huggingface_hub's httpx-based HEAD request
    for cache freshness, which started hanging indefinitely on WSL2
    after a recent `huggingface_hub` update (observed as a 5-minute
    reward-prompt hang, 2026-04-23). Offline load takes ~4 s on warm
    cache vs. >5 min hang with the network check. Falls back to
    online mode if the cache is empty (fresh install).
    """
    # Fast path — module-level dict reads are atomic under the GIL.
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    # Slow path — serialize first-time loads.
    with _EMBEDDER_LOAD_LOCK:
        cached = _EMBEDDER_CACHE.get(model_name)
        if cached is not None:
            return cached
        from sentence_transformers import SentenceTransformer

        _t0 = time.time()
        try:
            log.info(
                "loading sentence-transformer %r (local_files_only=True)",
                model_name,
            )
            embedder = SentenceTransformer(model_name, local_files_only=True)
            log.info(
                "loaded sentence-transformer %r from cache in %.1fs",
                model_name, time.time() - _t0,
            )
        except (OSError, ValueError) as e:
            # Cache miss — fall back to online load. This pays the
            # download cost (one-time per model) but also re-exposes
            # the WSL2 httpx-hang risk. `SCULPTOR_HF_NO_NETWORK=1`
            # forces offline-only and surfaces the cache-miss as an
            # error so CI / repeatable environments fail fast instead
            # of hanging.
            import os as _os
            if _os.environ.get("SCULPTOR_HF_NO_NETWORK") == "1":
                raise
            log.warning(
                "local-only load failed (%s); falling back to online "
                "download for %r", type(e).__name__, model_name,
            )
            _t0 = time.time()
            embedder = SentenceTransformer(model_name)
            log.info(
                "downloaded + loaded sentence-transformer %r in %.1fs",
                model_name, time.time() - _t0,
            )
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

    _t0 = time.time()
    techniques: list[Technique] = store.find_nodes(kind=Technique.kind)  # type: ignore[assignment]
    log.info(
        "kg.query: fetched %d Technique nodes in %.2fs",
        len(techniques), time.time() - _t0,
    )
    _t0 = time.time()
    need: list[Technique] = [
        t for t in techniques
        if not store.has_embedding(t.id, model_name) and (t.description or t.name)
    ]
    log.info(
        "kg.query: has_embedding scan: %d of %d missing in %.2fs",
        len(need), len(techniques), time.time() - _t0,
    )
    if need:
        log.info(
            "kg.query: backfilling %d technique embeddings (first query after ingest)",
            len(need),
        )
        embedder = _get_embedder(model_name)
        texts = [f"{t.name}. {t.description}".strip(". ") for t in need]
        _t0 = time.time()
        vecs = embedder.encode(texts, normalize_embeddings=True)
        log.info(
            "kg.query: embedder.encode(%d texts) took %.2fs",
            len(texts), time.time() - _t0,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        for t, v in zip(need, vecs):
            store.set_embedding(t.id, model_name, v)

    _t0 = time.time()
    out: list[tuple[Technique, Any]] = []
    for t in techniques:
        v = store.get_embedding(t.id, model_name)
        if v is not None:
            out.append((t, v))
    log.info(
        "kg.query: loaded %d embeddings in %.2fs",
        len(out), time.time() - _t0,
    )
    return out


def _ensure_failure_mode_embeddings(
    store: SculptorKG, model_name: str = EMBEDDING_MODEL
) -> list[tuple[FailureMode, Any]]:
    """Embed every FailureMode name+description that doesn't have a cached
    embedding yet, return (failure_mode, vector) for all failure modes.

    Mirrors `_ensure_technique_embeddings` exactly, over FailureMode nodes
    instead of Technique nodes — §KG-retrieval fix 2's embedding-matched
    free-text failure descriptors (`resolve_failure_modes_semantic`) needs
    the same lazy-embed-then-cache pool, just against a different node kind.
    """
    import numpy as np

    _t0 = time.time()
    modes: list[FailureMode] = store.find_nodes(kind=FailureMode.kind)  # type: ignore[assignment]
    log.info(
        "kg.query: fetched %d FailureMode nodes in %.2fs",
        len(modes), time.time() - _t0,
    )
    _t0 = time.time()
    need: list[FailureMode] = [
        m for m in modes
        if not store.has_embedding(m.id, model_name) and (m.description or m.name)
    ]
    log.info(
        "kg.query: has_embedding scan (FailureMode): %d of %d missing in %.2fs",
        len(need), len(modes), time.time() - _t0,
    )
    if need:
        log.info(
            "kg.query: backfilling %d failure-mode embeddings (first "
            "descriptor-resolution call after ingest)",
            len(need),
        )
        embedder = _get_embedder(model_name)
        texts = [f"{m.name}. {m.description}".strip(". ") for m in need]
        _t0 = time.time()
        vecs = embedder.encode(texts, normalize_embeddings=True)
        log.info(
            "kg.query: embedder.encode(%d texts) took %.2fs",
            len(texts), time.time() - _t0,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        for m, v in zip(need, vecs):
            store.set_embedding(m.id, model_name, v)

    _t0 = time.time()
    out: list[tuple[FailureMode, Any]] = []
    for m in modes:
        v = store.get_embedding(m.id, model_name)
        if v is not None:
            out.append((m, v))
    log.info(
        "kg.query: loaded %d FailureMode embeddings in %.2fs",
        len(out), time.time() - _t0,
    )
    return out


#: Default cosine floor for descriptor->FailureMode resolution
#: (§KG-retrieval fix 2). Higher than DEFAULT_MIN_PROMPT_SIMILARITY
#: (0.35) because a false-positive FailureMode match feeds
#: query_techniques's ADDRESSES walk and can pull in unrelated
#: techniques — descriptor resolution should stay conservative.
DEFAULT_MIN_DESCRIPTOR_SIMILARITY = 0.45


def resolve_failure_modes_semantic(
    store: SculptorKG,
    descriptors: list[str],
    *,
    top_k_per: int = 2,
    min_similarity: float = DEFAULT_MIN_DESCRIPTOR_SIMILARITY,
    model_name: str = EMBEDDING_MODEL,
) -> list[str]:
    """Embedding-match free-text failure descriptors onto FailureMode node
    ids — the semantic counterpart to `_resolve_failure_modes`'s fixed-enum
    fuzzy resolution.

    For each descriptor, ranks all FailureModes by cosine similarity
    between the descriptor text and `"{name}. {description}"`, keeping up
    to `top_k_per` above `min_similarity`. Results across all descriptors
    are deduped by node id (first-seen order). Returns `[]` when
    `descriptors` is empty, the KG has no FailureMode embeddings, or
    nothing clears the floor — the same best-effort, never-internally-
    guarded shape as `query_semantic` (a broken/missing embedder raises
    through to the caller, which is expected to guard it exactly like it
    guards `query_semantic`).
    """
    import numpy as np

    if not descriptors:
        return []
    pool = _ensure_failure_mode_embeddings(store, model_name)
    if not pool:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for desc in descriptors:
        text = str(desc or "").strip()
        if not text:
            continue
        qv = _embed_text(text, model_name)
        scored = []
        for fm, v in pool:
            sim = float(np.dot(qv, v))
            if sim < min_similarity:
                continue
            scored.append((sim, fm))
        scored.sort(key=lambda x: -x[0])
        for _, fm in scored[:top_k_per]:
            if fm.id in seen:
                continue
            seen.add(fm.id)
            out.append(fm.id)
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
        log.info(
            "kg.query.query_semantic: start (text_len=%d, top_k=%d, "
            "min_sim=%.2f, db=%s)",
            len(text or ""), top_k, min_similarity,
            getattr(store, "db_path", "<unknown>"),
        )
        pool = _ensure_technique_embeddings(store, model_name)
        if not pool:
            log.info("kg.query.query_semantic: empty pool, returning []")
            return []

        _t0 = time.time()
        qv = _embed_text(text, model_name)
        log.info(
            "kg.query.query_semantic: encoded query in %.2fs",
            time.time() - _t0,
        )
        # Vectors are L2-normalized, so dot product == cosine similarity.
        # §Agentic-data upgrade 2: the min_similarity FLOOR is checked on the
        # raw cosine `sim` — BEFORE the usage-based boost is added — so a
        # tangential match can never be resurrected above the floor by
        # citation count. The boost only reorders among matches that
        # already cleared the floor on relevance alone.
        scored = []
        for tech, v in pool:
            sim = float(np.dot(qv, v))
            if sim < min_similarity:
                continue
            ranked_score = sim + _citation_boost(tech.useful_citations)
            scored.append((ranked_score, sim, tech))
        scored.sort(key=lambda x: -x[0])

        results: list[TechniqueMatch] = []
        for ranked_score, sim, tech in scored[:top_k]:
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
