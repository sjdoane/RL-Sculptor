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
    make_failure_mode_id,
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
    retrieval keys on 'similar task with this failure', not on the verdict."""
    return f"{case.task}. symptom: {case.symptom}. verdict: {case.verdict}".strip()


# ── write-back ────────────────────────────────────────────────────────────
def record_run_cases(
    store: SculptorKG,
    *,
    task: str,
    robot: str = "",
    result,  # duck-typed SculptRunResult (avoids a sculpt<->kg import cycle)
    nonce: str | None = None,
    eps: float = 1e-4,
) -> int:
    """Materialize one `RunCase` per iteration of a fitness-tracked run and
    link it to its failure-mode nodes (INSTANTIATES). Forward-attribution: the
    edit made at iter N is judged by the fitness CHANGE measured at iter N+1.

    Best-effort and additive — callers wrap this so a logging failure can
    never affect a run. Returns the number of cases written."""
    nonce = nonce or uuid.uuid4().hex[:8]
    iters = list(getattr(result, "completed_iters", []) or [])
    fits = list(getattr(result, "fitness_history", []) or [])
    written = 0
    for idx, outcome in enumerate(iters):
        if idx >= len(fits):
            break
        cur = fits[idx]
        nxt_outcome = iters[idx + 1] if (idx + 1) < len(iters) else None
        # §Ship 37 review (HIGH): forward attribution only holds if iter N+1
        # actually TRAINED iter N's edit. With §Ship 36 revert-on-regression a
        # regressing edit is discarded and N+1 RE-MEASURES the best-so-far
        # reward; crediting that rebound to N's edit would record a regressing
        # edit as "helped" and recommend it to future runs. When N+1 reverted,
        # the edit's effect was never measured → leave the verdict 'unknown'.
        nxt = (
            fits[idx + 1]
            if (idx + 1) < len(fits)
            and not bool(getattr(nxt_outcome, "reverted_to_best", False))
            else None
        )
        delta = (nxt - cur) if nxt is not None else None
        fms = [
            str(fm) for fm in (getattr(outcome, "failure_modes", []) or [])
            if fm and str(fm) != "none"
        ]
        # Skip iterations that carry no learning at all (no failure flagged AND
        # no measurable fitness change to attribute).
        if not fms and delta is None:
            continue
        verdict = _verdict(delta, eps)
        symptom = ", ".join(fms) if fms else "no failure modes flagged"
        edit_count = int(getattr(outcome, "edit_count", 0) or 0)
        iter_index = int(getattr(outcome, "iter_index", idx))
        edit_summary = (
            f"iter {iter_index}: responded to [{symptom}] with "
            f"{edit_count} edit(s); objective fitness then {verdict}"
            + (f" ({delta:+.4f})" if delta is not None else "")
        )
        case = RunCase(
            id=make_run_case_id(task, iter_index, nonce),
            task=task, robot=robot, symptom=symptom, failure_modes=fms,
            edit_summary=edit_summary,
            fitness_before=cur, fitness_after=nxt, fitness_delta=delta,
            verdict=verdict,
        )
        store.add_node(case)
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
    Returns [] when there are no cases yet (the common early state)."""
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
        scored.sort(key=lambda x: -x[0])
        return [CaseMatch(case=c, relevance_score=s) for s, c in scored[:top_k]]
    finally:
        if owns:
            store.close()


_VERDICT_MARK = {"helped": "[+]", "regressed": "[-]", "neutral": "[=]", "unknown": "[?]"}


def _render_case_context(matches: list[CaseMatch]) -> str:
    """Render retrieved cases as a prompt block. Empty string when none, so
    the caller can prepend it unconditionally."""
    if not matches:
        return ""
    lines = [
        "# CASE MEMORY",
        "# This system's OWN past runs on similar tasks/failures. Learn from "
        "them: prefer directions marked [+] (helped), and do NOT repeat ones "
        "marked [-] (regressed the objective).",
    ]
    for m in matches:
        c = m.case
        lines.append(
            f"- {_VERDICT_MARK.get(c.verdict, '[?]')} {c.edit_summary} "
            f"(task: {c.task[:60]}; sim {m.relevance_score:.2f})"
        )
    return "\n".join(lines)
