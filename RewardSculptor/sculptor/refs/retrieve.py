"""Reference-clip retrieval: deterministic token-overlap search + an
optional LLM rerank pass (§R1_BUILD_SPEC decision 7).

Two layers, always composed in this order:

  1. **Deterministic** (always on, zero API, never fails): tokenize the
     query the same way `sculptor.refs.ingest.tokenize_label` tokenizes
     source stems, expand both query and index-row tokens through a
     small SYNONYM map, and score by rarity-weighted token overlap (an
     IDF-ish weighting — tokens that appear in few index rows count for
     more than tokens every clip shares, e.g. "g1"/"subject"). This
     layer is the acceptance-tested floor: `search("get up off the
     ground", ..., use_llm=False)` MUST rank a fallAndGetUp segment
     first with NO network access.
  2. **LLM rerank** (optional, best-effort): re-orders the top-20
     deterministic hits using the goal text, returning a match
     confidence + one-line reason per clip. Both client construction
     and the call itself are wrapped — ANY failure (no `anthropic`
     package, no API key, network error, unparseable response) falls
     back to the deterministic ranking with `match_confidence=None` and
     `rerank="deterministic-fallback"` on every returned `RefMatch`.
     `search()` NEVER raises because of the LLM layer.

`search()` reads `sculptor.refs.library.read_index()` — the on-disk
index.jsonl cache (§decision 4) — so a fresh process, a rebuilt index, or
a test fixture built entirely in-memory via `search_rows()` all go
through the identical scoring path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from sculptor.llm import log_llm_call, model_for, response_text_blocks
from sculptor.refs import library

MODEL_ID = model_for("reference_rerank")
MAX_TOKENS = 2048

#: Same camelCase/digit-boundary + lowercasing rule as ingest's
#: `tokenize_label` — re-implemented here (not imported) so
#: `refs/retrieve.py` has no import-time dependency on `refs/ingest.py`'s
#: network-adjacent module (ingest.py's docstring promises test isolation
#: from any network path; retrieve.py keeps that same guarantee trivially
#: true by not importing it at all). ONE deliberate difference: this
#: tokenizer splits on ANY non-alnum run (including whitespace), while
#: `tokenize_label` only splits on `_`/`-` — `tokenize_label` tokenizes
#: filename STEMS (no spaces expected), this one tokenizes free-text
#: SEARCH QUERIES ("get up off the ground" must become 5 tokens, not
#: one). Equivalence is covered only for filename-shaped inputs by
#: `test_tokenize_query_matches_ingest_tokenize_label`; the divergence on
#: space-containing text is deliberate and covered by
#: `test_tokenize_query_splits_on_whitespace_unlike_tokenize_label`.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")


def tokenize_query(text: str) -> list[str]:
    """Split free text into lowercase tokens: camelCase/digit boundaries
    split, non-alnum runs treated as separators, empties dropped."""
    s = _CAMEL_BOUNDARY.sub("_", str(text))
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    parts = [p for p in s.split("_") if p]
    return [p.lower() for p in parts]


#: §decision 7's extensible synonym dict. Keys and values are both
#: tokenizer-normalized (lowercase, already-split words); membership in
#: the same set is symmetric — every token in a group expands to every
#: other token in that group.
#: §2026-07-11 (deferred audit finding, build-log D2 follow-up): "fall"
#: used to live INSIDE the get-up group (`get≈up≈getup≈stand≈rise≈fall
#: ≈and`). That's correct for the PHRASE "fall and get up" (both
#: concepts co-occur literally on every fallAndGetUp clip, so it always
#: scored a full literal in-group hit regardless of grouping) but wrong
#: for the single token "fall" in isolation: `_idf_weights` computes a
#: token's rarity from its EXPANDED per-row token set (§`_idf_weights`
#: docstring), so every group member shares one corpus-wide weight. "up"
#: is an extremely common word across this library (direction/pose
#: modifiers like "wrists up down", "look up", any *_up clip) — folding
#: "fall" into the same group as "up" dragged "fall"'s own rarity down
#: to "up"'s near-zero IDF, so a query like "fall down" let an unrelated
#: clip that only shares the common word "up" (bridging through the
#: group) and the rare literal modifier "down" outscore the actual
#: fallAndGetUp clips — the literal "fall" match on the correct clips
#: was scored at that same diluted group weight. Splitting "fall" into
#: its own group restores its true (much higher) corpus rarity, and
#: since "fall and get up" clips still contain LITERAL "get"/"up" tokens
#: in the get-up group (and now also a literal "fall" in its own group,
#: each scored via `in_group_literal` — a full, undiscounted hit per
#: §D2), the "get up off the ground" LOCKED acceptance behavior is
#: unaffected: it never depended on "fall" being a member of the get-up
#: group, only on literal "get"/"up" overlap.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("get", "up", "getup", "stand", "rise", "and"),
    ("fall", "falling", "collapse", "tumble"),
    ("jump", "leap", "hop"),
    ("walk", "locomotion", "gait"),
    ("run", "sprint", "jog"),
    ("lie", "lying", "supine", "ground"),
    ("crouch", "squat"),
    ("kick",),
    ("dance",),
    ("fight", "boxing", "sports"),
    ("push", "shove"),
)


def _build_synonym_map(groups: tuple[tuple[str, ...], ...]) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for group in groups:
        gs = frozenset(group)
        for tok in group:
            out[tok] = out.get(tok, frozenset()) | gs
    return out


_SYNONYM_MAP = _build_synonym_map(SYNONYM_GROUPS)


def expand_synonyms(tokens: list[str]) -> set[str]:
    """Token list -> set of tokens ∪ every synonym-group member for any
    token that has one. Order-independent (used purely as a scoring
    bag), duplicates collapse naturally via `set`."""
    out: set[str] = set(tokens)
    for tok in tokens:
        out |= _SYNONYM_MAP.get(tok, frozenset())
    return out


# ── deterministic scoring ────────────────────────────────────────────────
def _row_tokens(row: dict[str, Any]) -> list[str]:
    """Tokens for one index row: its stored `labels` (already tokenized
    at ingest time) plus a fresh tokenization of `text` (covers any
    words in `text` that ingest didn't also put in `labels`, e.g. a
    later hand-edited description)."""
    toks: list[str] = list(row.get("labels") or [])
    text = row.get("text") or ""
    toks.extend(tokenize_query(text))
    return toks


def _idf_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Rarity weight per token across the corpus: `log(1 + N/df)`, so a
    token every row shares (near-zero signal) counts far less than one
    that appears in a handful of rows. `math.log` is monotonic in df so
    ties don't matter for ranking purposes; the +1 keeps a token with
    df==N from hitting exactly zero (it should still count a little)."""
    import math

    n = max(1, len(rows))
    df: dict[str, int] = {}
    for row in rows:
        for tok in set(expand_synonyms(_row_tokens(row))):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1.0 + n / count) for tok, count in df.items()}


@dataclass
class RefMatch:
    """One retrieval result (§decision 7). `match_confidence`/`reason`
    are populated only by the LLM rerank layer; deterministic-only
    results carry `match_confidence=None`.

    §spec deviation (reported, not improvised — §Hard rules): decision
    7's dataclass field list doesn't name a `rerank` field, but the same
    decision's prose requires "a `rerank: 'deterministic-fallback'` flag
    in the response" whenever the LLM layer didn't run/succeed. Adding
    `rerank` as an 11th field is the smallest change that satisfies both
    sentences — every result on a given `search()` call carries the same
    `rerank` tag (`"llm"` when the rerank succeeded, else
    `"deterministic-fallback"` or `"deterministic-only"`), so a caller
    can filter/display without re-deriving it from `match_confidence`.
    """

    clip_id: str
    text: str
    score: float
    match_confidence: Optional[float]
    reason: Optional[str]
    tier: Optional[str]
    license: Optional[str]
    n_frames: Optional[int]
    fps: Optional[float]
    duration_s: Optional[float]
    rerank: str = "deterministic-only"


def _row_to_match(row: dict[str, Any], score: float) -> RefMatch:
    return RefMatch(
        clip_id=row["clip_id"],
        text=row.get("text", ""),
        score=score,
        match_confidence=None,
        reason=None,
        tier=row.get("tier"),
        license=row.get("license"),
        n_frames=row.get("n_frames"),
        fps=row.get("fps"),
        duration_s=row.get("duration_s"),
    )


def _is_exact_clip_query(query: str, row: Mapping[str, Any]) -> bool:
    """Whether ``query`` names this exact robot-scoped library artifact.

    The HTTP surface scopes search with a separate ``robot`` parameter, while
    the UI renders the copyable identity as ``robot/clip_id``.  Accept both
    forms so pasting the identity shown on screen has the same authority as
    typing the bare id.  This is an identity comparison, not fuzzy retrieval;
    no token expansion or filename-prefix inference participates.
    """
    clip_id = str(row.get("clip_id") or "").casefold()
    if not clip_id:
        return False
    normalized = str(query).strip().casefold()
    if normalized == clip_id:
        return True
    robot = str(row.get("robot") or "").casefold()
    return bool(robot) and normalized == f"{robot}/{clip_id}"


#: Synonym-bridged matches count HALF a literal match (within the
#: concept component — see `_CONCEPT_BOOST` below). Rationale
#: (2026-07-09, found against the real 301-row library): expanding BOTH
#: sides and summing every overlapping token let one conceptual bridge
#: ("ground"≈"lie") count as 4 rare tokens' worth of weight, so `lie*`
#: clips outranked the fallAndGetUp segments for the literal acceptance
#: query "get up off the ground". A synonym hit is one concept — count
#: it once per group, discounted below a literal in-group match.
_SYNONYM_MATCH_WEIGHT = 0.5

#: Multiplier applied to the whole "concept component" of the score —
#: i.e. everything credited via a `SYNONYM_GROUPS` match (literal
#: in-group or synonym-bridged). Rationale (2026-07-09, found against
#: the real 301-row library, round 2): `SYNONYM_GROUPS` IS the
#: hand-curated motion-concept vocabulary (walk/run/jump/lie/crouch/...).
#: Once the first fix (`_SYNONYM_MATCH_WEIGHT`) fixed "get up off the
#: ground", a second failure mode showed up: a rare MODIFIER token that
#: isn't a motion concept at all (e.g. "forward", "high", "down") can
#: still carry a bigger raw IDF weight than the concept token itself,
#: because it appears in fewer rows. "walk forward" then let
#: crawl_forward/lie_forward clips (which share the literal "forward"
#: token but no walk-group token) outrank actual walk clips; "jump
#: high" let block_*_high clips (literal "high", no jump/hop token at
#: all) outrank hop clips. Motion concept > modifier is the intended
#: retrieval semantics, so every point earned through a
#: `SYNONYM_GROUPS` hit — literal-in-group or synonym-bridged — is
#: scaled up by this factor before non-group literal (modifier) tokens
#: are added at their plain IDF weight. 2.0 was picked empirically: it
#: is enough to make a single-concept-token match beat a single rare
#: modifier token in every one of the 6 tuning queries, without being
#: so large that a purely-modifier query (no concept token in the
#: SYNONYM_GROUPS vocabulary at all) loses all discriminating power.
_CONCEPT_BOOST = 2.2


def deterministic_rank(
    query: str, rows: list[dict[str, Any]], *, k: int = 10,
) -> list[RefMatch]:
    """Score every row by rarity-weighted token overlap and return the
    top `k`, best first.

    The score has two components:

    - **Concept component** (`_CONCEPT_BOOST`x): for each
      `SYNONYM_GROUPS` group, if the query and row share a LITERAL
      token in that group, credit the sum of the shared literal
      members' IDF weights; otherwise, if the group only bridges the
      two sides via synonym expansion (no literal overlap), credit
      `_SYNONYM_MATCH_WEIGHT` x the strongest present member's weight.
      The whole component is then scaled by `_CONCEPT_BOOST` — motion
      CONCEPT tokens (walk/run/jump/lie/crouch/...) are the vocabulary
      the caller actually cares about matching, and must outrank a
      coincidental rare-modifier hit (see `_CONCEPT_BOOST`'s docstring).
    - **Modifier component** (1x): literal query∩row tokens that are
      NOT in any `SYNONYM_GROUPS` group, at plain IDF weight — e.g.
      "forward"/"high"/"down"/"subject" style tokens that still add
      useful discriminating signal between same-concept clips.

    Zero-scoring rows are excluded unless the query is the row's exact bare
    or robot-scoped clip identity.  Exact identity always ranks before fuzzy
    relevance; every non-exact result retains the prior score/density/id
    ordering. Deterministic, offline, never raises on well-formed input —
    this is the §decision-7 "always on, zero API" floor."""
    q_lit = set(tokenize_query(query))
    if not q_lit or not rows:
        return []
    q_all = expand_synonyms(sorted(q_lit))
    weights = _idf_weights(rows)
    grouped_tokens: set[str] = set()
    for group in SYNONYM_GROUPS:
        grouped_tokens |= frozenset(group)
    scored: list[tuple[bool, float, int, dict[str, Any]]] = []
    for row in rows:
        r_lit = set(_row_tokens(row))
        r_all = expand_synonyms(sorted(r_lit))
        literal = q_lit & r_lit

        concept = 0.0
        for group in SYNONYM_GROUPS:
            gs = frozenset(group)
            in_group_literal = literal & gs
            if in_group_literal:
                concept += sum(weights.get(tok, 0.0) for tok in in_group_literal)
                continue
            if (q_all & gs) and (r_all & gs):
                present = (q_all | r_all) & gs
                concept += _SYNONYM_MATCH_WEIGHT * max(
                    weights.get(tok, 0.0) for tok in present
                )
        modifier = sum(
            weights.get(tok, 0.0) for tok in literal if tok not in grouped_tokens
        )
        score = _CONCEPT_BOOST * concept + modifier
        exact_id = _is_exact_clip_query(query, row)
        if score <= 0.0 and not exact_id:
            continue
        scored.append((exact_id, score, len(r_lit), row))
    # Tie-break (2026-07-11, found against the real index after the
    # stageii ingest recovered MOYO yoga clips): rows that EXACTLY tie on
    # score are ordered by match DENSITY — fewer distinct row tokens
    # first — before the final clip_id key. Rationale: a 1-token overlap
    # against a 4-token label ("dance 1 subject 5") is a tighter match
    # than the same 1-token overlap against a 15-token label ("220926
    # yogi body hands ... lord of the dance pose or natarajasana ..."),
    # but both carry the identical IDF score, and the old clip_id-only
    # tie-break put the yoga clip first purely because "2..." sorts
    # before "d...". Density only ever reorders EXACT score ties, so no
    # non-tied ranking (including every locked acceptance ordering) can
    # change. clip_id stays as the last key so ties remain deterministic
    # across runs/platforms (dict/set iteration order is not a ranking
    # input).
    scored.sort(key=lambda sr: (not sr[0], -sr[1], sr[2], sr[3]["clip_id"]))
    return [
        _row_to_match(row, score)
        for _exact, score, _n_tokens, row in scored[:k]
    ]


# ── LLM rerank (§decision 7, optional) ──────────────────────────────────
class RerankUnavailable(Exception):
    """Internal signal: the LLM layer could not run (any reason). Always
    caught inside `search()` — never escapes to the caller."""


def _rerank_with_llm(
    query: str, candidates: list[RefMatch], *, client=None,
) -> list[RefMatch]:
    """Rerank `candidates` (already the deterministic top-20) via the
    `reference_rerank` role model. Raises `RerankUnavailable` on ANY
    failure — `search()` catches it and falls back. Never imports
    `anthropic` at module scope (§Hard rules: no test may require an API
    key; importing only inside this function keeps that true even if a
    caller imports `retrieve` at collection time)."""
    from pydantic import BaseModel, Field

    class _RerankItemModel(BaseModel):
        clip_id: str
        match_confidence: float = Field(ge=0.0, le=1.0)
        reason: str

    class _RerankModel(BaseModel):
        ranked: list[_RerankItemModel]

    try:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=2, timeout=60.0)

        candidate_block = "\n".join(
            f"- clip_id={m.clip_id!r} text={m.text!r} "
            f"tier={m.tier!r} duration_s={m.duration_s!r}"
            for m in candidates
        )
        system_prompt = (
            "You rank motion-capture reference clips by how well they "
            "match a natural-language goal. You will be given a goal "
            "and a list of candidate clips (clip_id, a short text "
            "label, a quality tier, duration). Return every candidate "
            "clip_id, ordered best-match first, each with a "
            "match_confidence in [0, 1] and a one-line reason. Do not "
            "invent clip_ids that aren't in the candidate list."
        )
        user_content = f"GOAL: {query}\n\nCANDIDATES:\n{candidate_block}"

        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_format=_RerankModel,
        )
        log_llm_call(
            "reference_rerank", MODEL_ID, system=system_prompt,
            user=user_content, response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None))
        parsed = resp.parsed_output
    except Exception as e:  # noqa: BLE001 — any failure -> fallback
        raise RerankUnavailable(f"{type(e).__name__}: {e}") from e

    by_id = {m.clip_id: m for m in candidates}
    out: list[RefMatch] = []
    seen: set[str] = set()
    for item in parsed.ranked:
        base = by_id.get(item.clip_id)
        if base is None or item.clip_id in seen:
            # LLM invented or duplicated a clip_id — not trustworthy
            # enough to special-case; treat the whole response as
            # unusable rather than silently drop/reorder around it.
            raise RerankUnavailable(
                f"rerank response referenced unknown/duplicate "
                f"clip_id {item.clip_id!r}")
        seen.add(item.clip_id)
        out.append(RefMatch(
            clip_id=base.clip_id, text=base.text, score=base.score,
            match_confidence=float(item.match_confidence),
            reason=str(item.reason)[:280],
            tier=base.tier, license=base.license, n_frames=base.n_frames,
            fps=base.fps, duration_s=base.duration_s, rerank="llm",
        ))
    # Any candidate the LLM silently dropped is appended at the tail in
    # its original deterministic order — never lose a result to a lazy
    # LLM response. Still tagged "llm" since the call itself succeeded.
    for m in candidates:
        if m.clip_id not in seen:
            out.append(RefMatch(
                clip_id=m.clip_id, text=m.text, score=m.score,
                match_confidence=None, reason=None, tier=m.tier,
                license=m.license, n_frames=m.n_frames, fps=m.fps,
                duration_s=m.duration_s, rerank="llm",
            ))
    return out


# ── public API ────────────────────────────────────────────────────────
_RERANK_TOP_N = 20


def search_rows(
    query: str, rows: list[dict[str, Any]], *, k: int = 10,
    use_llm: bool = True, client=None,
) -> list[RefMatch]:
    """Core search over an already-loaded list of index rows (no disk
    I/O) — what `search()` calls after filtering by robot, and what
    tests use directly against an in-memory fixture index."""
    deterministic = deterministic_rank(query, rows, k=max(k, _RERANK_TOP_N))
    # A pasted artifact identity is already a complete selection instruction,
    # not a semantic prompt.  Letting an optional LLM reshuffle it would turn
    # exact lookup back into an ambiguous recommendation list.
    exact_id_query = any(_is_exact_clip_query(query, row) for row in rows)
    if not use_llm or not deterministic or exact_id_query:
        return deterministic[:k]
    try:
        reranked = _rerank_with_llm(query, deterministic[:_RERANK_TOP_N], client=client)
        return reranked[:k]
    except RerankUnavailable:
        return [
            RefMatch(
                clip_id=m.clip_id, text=m.text, score=m.score,
                match_confidence=None, reason=None, tier=m.tier,
                license=m.license, n_frames=m.n_frames, fps=m.fps,
                duration_s=m.duration_s, rerank="deterministic-fallback",
            )
            for m in deterministic[:k]
        ]


def search(
    query: str, robot: str = "g1", k: int = 10, use_llm: bool = True,
    *, root: Optional[Any] = None, client=None,
) -> list[RefMatch]:
    """§decision 7 public API. Reads `library.read_index()` (optionally
    scoped to a custom `root` for tests), filters to `robot`, and
    returns up to `k` ranked `RefMatch` results. Never raises: a
    missing/empty index returns `[]`; an LLM failure silently falls
    back to the deterministic ranking."""
    rows = [r for r in library.read_index(root=root) if r.get("robot") == robot]
    return search_rows(query, rows, k=k, use_llm=use_llm, client=client)
