"""sculptor/kg/cases.py — §Ship 37: run-learning "case memory".

The KG was read-only research literature: papers in, nothing about the
system's OWN runs ever came back out. So every diagnosis started blind to
what the previous run already tried — the loop could (and did) re-propose an
edit that had already failed.

A `RunCase` records one sculpt iteration's experience: the task, the failure
modes observed, the edit made in response, and whether the OBJECTIVE fitness
then improved or regressed. Cases are:

  * WRITTEN at the end of a fitness-tracked sculpt_run (`record_run_cases`),
  * RETRIEVED by the diagnoser via semantic similarity (`query_cases`,
    mirroring `query_semantic` over Technique nodes — same MiniLM model, same
    lazy-backfill embedding cache), and
  * RENDERED into the grounded-diagnosis prompt (`_render_case_context`) as a
    "CASE MEMORY" block above the literature context.

Design notes (from the Ship-35/36 research + red-team): cases live in a
SEPARATE silo from the literature graph (their own node kind + the
INSTANTIATES relation), retrieved independently and merged only at prompt
time — so fast-moving, run-scoped artifacts never pollute the stable
published-knowledge graph. No store schema change is needed: the node blob is
generic JSON and the embedding table already exists.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sculptor.kg.query import EMBEDDING_MODEL, _embed_text, _get_embedder
from sculptor.kg.schema import (
    Edge,
    FailureMode,
    Relation,
    RunCase,
    Technique,
    evidence_tag,
    make_failure_mode_id,
    make_paper_id,
    make_run_case_id,
)
from sculptor.kg.store import SculptorKG


# ── verdict + text helpers ───────────────────────────────────────────────
def _verdict(delta: float | None, eps: float = 1e-4) -> str:
    """Classify a fitness delta into a coarse, retrieval-friendly verdict."""
    if delta is None:
        return "unknown"
    if delta > eps:
        return "helped"
    if delta < -eps:
        return "regressed"
    return "neutral"


def _case_text(case: RunCase) -> str:
    """The text a case is embedded + matched on — task + symptom dominate so
    retrieval keys on 'similar task with this failure'; the edit identities
    are included so 'similar edit under consideration' also matches."""
    edits = f" edits: {', '.join(case.edits)}." if case.edits else ""
    return (
        f"{case.task}. symptom: {case.symptom}.{edits} "
        f"verdict: {case.verdict}"
    ).strip()


# Behavior-signature keys worth carrying into a case, in priority order.
# Metric-agnostic fallback: if none of these appear, the first few numeric
# components are taken as-is.
_SIGNATURE_PRIORITY = (
    "progress_score", "apex_gain_m_mean", "frac_launched", "frac_returned",
    "frac_upright_end", "tuck_excess_rad_mean", "max_lateral_drift_m_mean",
)
_SIGNATURE_MAX_KEYS = 6


def _behavior_signature(components: dict | None) -> dict[str, float]:
    """Distill the metric's component breakdown into ≤6 floats that
    characterize WHAT the policy did (stand-farm vs tumble vs sit-bob),
    priority keys first, then any other numerics up to the cap."""
    if not isinstance(components, dict):
        return {}
    out: dict[str, float] = {}
    for k in _SIGNATURE_PRIORITY:
        v = components.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v), 4)
    for k, v in components.items():
        if len(out) >= _SIGNATURE_MAX_KEYS:
            break
        if k in out:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v), 4)
    return out


# ── write-back ────────────────────────────────────────────────────────────
def record_run_cases(
    store: SculptorKG,
    *,
    task: str,
    robot: str = "",
    result,  # duck-typed SculptRunResult (avoids a sculpt<->kg import cycle)
    nonce: str | None = None,
    eps: float = 1e-4,
    iter_references: dict[int, list[str]] | None = None,
) -> int:
    """Materialize one `RunCase` per iteration of a run and link it to its
    failure-mode nodes (INSTANTIATES). Forward-attribution: the edit made at
    iter N is judged by the LEXICOGRAPHIC (fitness, dense-progress) change
    measured at iter N+1 — the same key the loop selects on, so a run whose
    completion-gated fitness is 0.0 throughout still yields real verdicts.
    Blind runs (no fitness) record too: symptom + edit identities + behavior
    signature, verdict 'unknown' — what was tried is itself memory.

    `iter_references` (§Agentic-data upgrade 2, optional): iter_index ->
    cited arxiv_ids for the reward version KEPT at that iteration.
    `sculpt.py`'s `IterOutcome`/`SculptRunResult` do not themselves carry a
    reward file's `REWARD_SPEC.references` (that list lives in the reward
    module's source, not the in-memory run record) — reading it would mean
    this function opening and importing the kept reward file, which is out
    of scope for a KG-side helper. Wiring a real value through is a
    follow-up at the `sculpt.py` call site (untouched here, per scope);
    until then this stays None and no useful_citations increment happens.
    When present: for each iteration whose verdict is "helped", the cited
    papers' INTRODUCES-d Techniques get `useful_citations` incremented by
    1 — a coarse, best-effort signal that a technique's citation actually
    paid off in THIS project's own run, feeding the capped ranking boost
    in `query.py`.

    Best-effort and additive — callers wrap this so a logging failure can
    never affect a run. Returns the number of cases written."""
    nonce = nonce or uuid.uuid4().hex[:8]
    iter_references = iter_references or {}
    iters = list(getattr(result, "completed_iters", []) or [])
    fits = list(getattr(result, "fitness_history", []) or [])
    progs = list(getattr(result, "progress_history", []) or [])
    written = 0
    for idx, outcome in enumerate(iters):
        cur = fits[idx] if idx < len(fits) else None
        cur_p = progs[idx] if idx < len(progs) else None
        nxt_outcome = iters[idx + 1] if (idx + 1) < len(iters) else None
        # §Ship 37 review (HIGH): forward attribution only holds if iter N+1
        # actually TRAINED iter N's edit. With §Ship 36 revert-on-regression a
        # regressing edit is discarded and N+1 RE-MEASURES the best-so-far
        # reward; crediting that rebound to N's edit would record a regressing
        # edit as "helped" and recommend it to future runs. When N+1 reverted,
        # the edit's effect was never measured → leave the verdict 'unknown'.
        measured_next = (
            nxt_outcome is not None
            and not bool(getattr(nxt_outcome, "reverted_to_best", False))
        )
        nxt = fits[idx + 1] if measured_next and (idx + 1) < len(fits) else None
        nxt_p = (progs[idx + 1]
                 if measured_next and (idx + 1) < len(progs) else None)
        delta = (nxt - cur) if (nxt is not None and cur is not None) else None
        delta_p = ((nxt_p - cur_p)
                   if (nxt_p is not None and cur_p is not None) else None)
        fms = [
            str(fm) for fm in (getattr(outcome, "failure_modes", []) or [])
            if fm and str(fm) != "none"
        ]
        edits = [str(e) for e in (getattr(outcome, "applied_edits", []) or [])]
        # Skip iterations that carry no learning at all (no failure flagged,
        # no edit identity, AND no measurable change to attribute).
        if not fms and not edits and delta is None and delta_p is None:
            continue
        # §2026-07-03: LEXICOGRAPHIC attribution — mirror the loop's own
        # selection key (spec decides; dense progress breaks ties). The
        # tuck-jump E2E recorded 12 straight 'neutral'/'unknown' cases
        # because spec stayed 0.0 while the progress channel was doing
        # ALL the ranking — a 0.0196 → 0.0 progress regression was
        # invisible to the memory.
        if delta is not None and abs(delta) > eps:
            verdict = _verdict(delta, eps)
            attributed = f"fitness {delta:+.4f}"
        elif delta_p is not None and abs(delta_p) > eps:
            verdict = _verdict(delta_p, eps)
            attributed = f"progress {delta_p:+.4f}"
        elif delta is not None or delta_p is not None:
            verdict = "neutral"
            attributed = "no measurable change"
        else:
            verdict = "unknown"
            attributed = "effect never measured (reverted next / last iter)"
        symptom = ", ".join(fms) if fms else "no failure modes flagged"
        iter_index = int(getattr(outcome, "iter_index", idx))
        behavior = _behavior_signature(
            getattr(outcome, "fitness_components", None))
        behavior_s = (
            " behavior: " + ", ".join(
                f"{k}={v:g}" for k, v in behavior.items()) + "."
            if behavior else ""
        )
        edits_s = "; ".join(edits) if edits else (
            f"{int(getattr(outcome, 'edit_count', 0) or 0)} edit(s)")
        edit_summary = (
            f"iter {iter_index}: [{symptom}] → applied: {edits_s} → "
            f"{verdict} ({attributed})." + behavior_s
        )
        # §Agentic-data upgrade 4 (freshness metadata): the reward/env-spec
        # version ACTIVE while this iter trained. `IterOutcome` already
        # carries these (no sculpt.py call-site change needed) — reward as
        # a Path to the v<n>.py file, env spec as a plain version string.
        reward_path_trained = getattr(outcome, "reward_path_trained", None)
        reward_version = (
            str(reward_path_trained.stem) if reward_path_trained else None)
        env_spec_version = getattr(outcome, "env_spec_trained", None)
        case = RunCase(
            id=make_run_case_id(task, iter_index, nonce),
            task=task, robot=robot, symptom=symptom, failure_modes=fms,
            edit_summary=edit_summary,
            fitness_before=cur, fitness_after=nxt, fitness_delta=delta,
            verdict=verdict,
            edits=edits,
            progress_before=cur_p, progress_after=nxt_p,
            progress_delta=delta_p,
            behavior=behavior,
            reward_version=reward_version,
            env_spec_version=env_spec_version,
        )
        store.add_node(case)
        # §Agentic-data upgrade 2: usage-based enrichment. A "helped" iter
        # with a known reference list credits each cited paper's
        # INTRODUCES-d techniques — this project's own run just showed the
        # citation paid off, not merely that a paper claims it works.
        if verdict == "helped":
            for arxiv_id in iter_references.get(iter_index, []) or []:
                paper_id = make_paper_id(str(arxiv_id))
                for _, tech_id in store.neighbors(
                    paper_id, relation=Relation.INTRODUCES, direction="out"
                ):
                    tech = store.get_node(tech_id)
                    if not isinstance(tech, Technique):
                        continue
                    tech.useful_citations = int(tech.useful_citations or 0) + 1
                    store.add_node(tech)
        for fm in fms:
            fm_id = make_failure_mode_id(fm)
            # §KG integrity: the diagnoser can flag a failure mode that was never
            # extracted from a paper, so no FailureMode node exists. Ensure the node
            # exists before linking, else the INSTANTIATES edge DANGLES — the
            # case→failure provenance is dead and the graph viz must tolerate a broken
            # edge (sculptor/kg/viz.py). A stub node keeps the silo self-consistent.
            if not store.has_node(fm_id):
                store.add_node(FailureMode(
                    id=fm_id, name=str(fm),
                    description="(diagnoser-flagged failure mode; not paper-derived)",
                ))
            store.add_edge(Edge(
                src=case.id, dst=fm_id,
                relation=Relation.INSTANTIATES,
                data={"verdict": verdict, "delta": delta},
            ))
        written += 1
    return written


# ── retrieval ─────────────────────────────────────────────────────────────
@dataclass
class CaseMatch:
    case: RunCase
    relevance_score: float


def _ensure_case_embeddings(store: SculptorKG, model_name: str = EMBEDDING_MODEL):
    """Embed every RunCase that lacks a cached vector (lazy backfill, same
    pattern as Technique embeddings), return [(case, vector), ...]."""
    import numpy as np

    cases = store.find_nodes(kind=RunCase.kind)
    need = [c for c in cases if not store.has_embedding(c.id, model_name)]
    if need:
        embedder = _get_embedder(model_name)
        vecs = np.asarray(
            embedder.encode([_case_text(c) for c in need], normalize_embeddings=True),
            dtype=np.float32,
        )
        for c, v in zip(need, vecs):
            store.set_embedding(c.id, model_name, v)
    out = []
    for c in cases:
        v = store.get_embedding(c.id, model_name)
        if v is not None:
            out.append((c, v))
    return out


#: §Agentic-data upgrade 4: similarity gap under which two cases are
#: considered "tied" for ranking purposes — recency then breaks the tie.
#: This is a SOFT preference (stable sort), never a hard filter: an old
#: case that is genuinely more similar still outranks a fresher, weaker
#: match by more than this gap.
_RECENCY_TIEBREAK_GAP = 0.02


def query_cases(
    text: str,
    top_k: int = 3,
    *,
    store: SculptorKG | None = None,
    model_name: str = EMBEDDING_MODEL,
    min_similarity: float = 0.0,
) -> list[CaseMatch]:
    """Rank RunCase nodes by cosine similarity between `text` (the current
    failure context) and each case's task+symptom. Mirrors `query_semantic`.
    Returns [] when there are no cases yet (the common early state).

    §Agentic-data upgrade 4: when two matches' similarities are within
    `_RECENCY_TIEBREAK_GAP`, the more recently-created case sorts first —
    a stable-sort nudge, not a hard filter (old-but-closer cases are never
    dropped)."""
    import numpy as np

    owns = store is None
    store = store or SculptorKG()
    try:
        pool = _ensure_case_embeddings(store, model_name)
        if not pool:
            return []
        qv = _embed_text(text, model_name)
        scored = []
        for case, v in pool:
            sim = float(np.dot(qv, v))
            if sim < min_similarity:
                continue
            scored.append((sim, case))
        # Bucket similarity into _RECENCY_TIEBREAK_GAP-wide bands so ties
        # within the band sort by recency, while a genuinely closer match
        # outside the band still wins outright. Primary key: the band
        # (coarser -> preserves "more similar wins" at the band level).
        # Secondary key: created_at (recency breaks within-band ties).
        scored.sort(
            key=lambda x: (
                -round(x[0] / _RECENCY_TIEBREAK_GAP),
                -x[1].created_at,
            )
        )
        return [CaseMatch(case=c, relevance_score=s) for s, c in scored[:top_k]]
    finally:
        if owns:
            store.close()


_VERDICT_MARK = {"helped": "[+]", "regressed": "[-]", "neutral": "[=]", "unknown": "[?]"}


def _render_case_context(matches: list[CaseMatch]) -> str:
    """Render retrieved cases as a prompt block. Empty string when none, so
    the caller can prepend it unconditionally.

    §Agentic-data upgrade 1: each case carries a compact `[evidence: ...]`
    tag (cases are RunCase nodes, so this is always "observed in THIS
    project's own runs") and the block's header states the trust-tier
    rule: observations from this system's own runs outrank paper claims
    when they conflict."""
    if not matches:
        return ""
    lines = [
        "# CASE MEMORY",
        "# This system's OWN past runs on similar tasks/failures. Learn from "
        "them: prefer directions marked [+] (helped), and do NOT repeat ones "
        "marked [-] (regressed the objective).",
        "# Observations from this system's own runs outrank paper claims "
        "when they conflict.",
    ]
    for m in matches:
        c = m.case
        lines.append(
            f"- {_VERDICT_MARK.get(c.verdict, '[?]')} {c.edit_summary} "
            f"(task: {c.task[:60]}; sim {m.relevance_score:.2f}) "
            f"{evidence_tag(c.provenance)}"
        )
    return "\n".join(lines)
