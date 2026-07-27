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

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sculptor.kg.query import EMBEDDING_MODEL, _embed_text, _get_embedder
from sculptor.kg.schema import (
    Edge,
    FailureMode,
    PROVENANCE_LLM_EXTRACTION,
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
    scope = " ".join(
        part for part in (
            f"project: {case.project}." if case.project else "",
            f"robot: {case.robot}." if case.robot else "",
        ) if part)
    return (
        f"{case.task}. {scope} symptom: {case.symptom}.{edits} "
        f"verdict: {case.verdict}"
    ).strip()


def _artifact_version_int(raw) -> int | None:
    """Parse an ArtifactRef version — the store serializes them as
    ``"v<N>"`` strings (artifacts.ArtifactRef), bare ints tolerated."""
    match = re.fullmatch(r"v?([0-9]+)", str(raw).strip())
    return int(match.group(1)) if match else None


def _world_identity(outcome) -> tuple[str | None, int | None, int | None]:
    """(tuple_hash, world_version, task_version) of the atomic selection
    this iteration trained under (env-authoring §10). ``IterOutcome``
    already carries the pinned selection hash/path, so no sculpt.py
    call-site change is needed. Fail-soft by contract: legacy runs and
    any read/parse failure return (None, None, None) — world identity is
    freshness evidence, never a record blocker."""
    tuple_hash = str(getattr(outcome, "world_selection_hash", "") or "") or None
    world_v: int | None = None
    task_v: int | None = None
    path = getattr(outcome, "world_selection_path", None)
    if path:
        try:
            refs = json.loads(Path(path).read_text())["refs"]
            world_v = _artifact_version_int(refs["world"]["version"])
            task_v = _artifact_version_int(refs["task"]["version"])
        except Exception:  # noqa: BLE001 — fail-soft freshness metadata
            pass
    return tuple_hash, world_v, task_v


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
    project: str = "",
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
    iter_references = iter_references or {}
    iters = list(getattr(result, "completed_iters", []) or [])
    # §KG-retrieval fix 4 idempotency: a resumed/re-run stage re-reports the
    # SAME physical iter dirs through this tail; a random nonce would mint a
    # new case id each time and the cumulative counters below would
    # double-count. Derive the nonce from the run's on-disk identity (the
    # runs root) so the same physical iteration always maps to the same
    # case id, and gate the counter increments on that id not existing yet.
    if nonce is None:
        _first_dir = getattr(iters[0], "iter_dir", None) if iters else None
        if _first_dir is not None:
            nonce = hashlib.sha1(
                str(Path(_first_dir).resolve().parent).encode("utf-8")
            ).hexdigest()[:8]
        else:
            nonce = uuid.uuid4().hex[:8]
    fits = list(getattr(result, "fitness_history", []) or [])
    progs = list(getattr(result, "progress_history", []) or [])
    def _technique_ids(references: list[str]) -> set[str]:
        out: set[str] = set()
        for arxiv_id in references:
            for _, tech_id in store.neighbors(
                    make_paper_id(str(arxiv_id)),
                    relation=Relation.INTRODUCES, direction="out"):
                if isinstance(store.get_node(tech_id), Technique):
                    out.add(tech_id)
        return out

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
        world_tuple_hash, world_version, task_version = _world_identity(outcome)
        case_id = make_run_case_id(task, iter_index, nonce)
        old_case = store.get_node(case_id)
        old_case = old_case if isinstance(old_case, RunCase) else None
        incoming_refs = sorted(set(
            str(r) for r in (iter_references.get(iter_index, []) or []) if r))
        # A transient unreadable reward artifact on resume must not erase
        # references that were already persisted with this physical case.
        case_refs = incoming_refs or (
            list(old_case.references) if old_case is not None else [])
        case = RunCase(
            id=case_id,
            task=task, robot=robot, project=project,
            symptom=symptom, failure_modes=fms,
            edit_summary=edit_summary,
            fitness_before=cur, fitness_after=nxt, fitness_delta=delta,
            verdict=verdict,
            edits=edits, references=case_refs, attribution_version=1,
            progress_before=cur_p, progress_after=nxt_p,
            progress_delta=delta_p,
            behavior=behavior,
            created_at=(old_case.created_at if old_case is not None
                        else time.time()),
            reward_version=reward_version,
            env_spec_version=env_spec_version,
            world_tuple_hash=world_tuple_hash,
            world_version=world_version,
            task_version=task_version,
        )
        store.add_node(case)
        # §Agentic-data upgrade 2: usage-based enrichment. A "helped" iter
        # with a known reference list credits each cited paper's
        # INTRODUCES-d techniques — this project's own run just showed the
        # citation paid off, not merely that a paper claims it works.
        #
        # §KG-retrieval fix 4 (outcome-stats ranking): additionally, for
        # BOTH "helped" and "regressed" iterations, tally a per-FailureMode
        # helped/regressed count on each cited technique — a real, scoped
        # learning signal `query_techniques` can rank on (a technique that
        # helped THIS failure mode outranks one that only ever regressed
        # it, independent of the unscoped `useful_citations` count above).
        # "unknown"/"neutral" iterations contribute zero. Re-recording is
        # delta-based: unknown→helped adds once; helped→helped is a no-op;
        # helped→regressed reverses the old contribution then adds the new.
        old_verdict = old_case.verdict if old_case is not None else "unknown"
        old_fm_ids = {
            make_failure_mode_id(fm)
            for fm in (old_case.failure_modes if old_case is not None else [])
        }
        new_fm_ids = {make_failure_mode_id(fm) for fm in fms}
        old_references = (
            list(old_case.references) if old_case is not None else [])
        if (old_case is not None
                and int(old_case.attribution_version or 0) < 1
                and not old_references):
            # Pre-reference rows may already have credited their papers, but
            # did not persist which ones. On the first post-upgrade re-record,
            # conservatively treat the supplied references as the old basis.
            # This prevents live helped cases from being credited twice while
            # still allowing unknown→helped to add its first contribution.
            old_references = list(case_refs)
        old_tech_ids = _technique_ids(old_references)
        new_tech_ids = _technique_ids(case_refs)
        for tech_id in old_tech_ids | new_tech_ids:
            tech = store.get_node(tech_id)
            if not isinstance(tech, Technique):
                continue
            old_active = tech_id in old_tech_ids
            new_active = tech_id in new_tech_ids
            useful_delta = (
                int(new_active and verdict == "helped")
                - int(old_active and old_verdict == "helped"))
            if useful_delta:
                tech.useful_citations = max(
                    0, int(tech.useful_citations or 0) + useful_delta)
            stats = {k: dict(v) for k, v in (tech.outcome_stats or {}).items()}
            stats_changed = False
            for fm_id in old_fm_ids | new_fm_ids:
                entry = dict(stats.get(fm_id) or {})
                entry["helped"] = int(entry.get("helped", 0))
                entry["regressed"] = int(entry.get("regressed", 0))
                for label in ("helped", "regressed"):
                    delta_count = (
                        int(new_active and fm_id in new_fm_ids
                            and verdict == label)
                        - int(old_active and fm_id in old_fm_ids
                              and old_verdict == label))
                    if delta_count:
                        entry[label] = max(0, entry[label] + delta_count)
                        stats_changed = True
                if entry["helped"] or entry["regressed"]:
                    stats[fm_id] = entry
                else:
                    stats.pop(fm_id, None)
            if useful_delta or stats_changed:
                tech.outcome_stats = stats
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
                    # Not paper-derived: the type default (paper_claim) would
                    # over-trust a diagnoser inference. A later paper
                    # extraction that attests this failure mode upgrades it
                    # via schema.merge_provenance (extract._materialize).
                    provenance=PROVENANCE_LLM_EXTRACTION,
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
    """Staleness-aware RunCase embedding pool (kg.query.ensure_embeddings).
    Matters here specifically: a resumed run re-records the same case id
    with a possibly UPDATED verdict/edit summary (`record_run_cases`
    upserts), and `_case_text` embeds the verdict — the old
    has_embedding-only check kept serving the pre-resume vector."""
    from sculptor.kg.query import ensure_embeddings

    cases = store.find_nodes(kind=RunCase.kind)
    return ensure_embeddings(
        store, cases, _case_text, model_name, label="RunCase")


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
    tag (cases are RunCase nodes, so this is always an observed run) and
    the block's header states the trust-tier
    rule: observations from this system's own runs outrank paper claims
    when they conflict."""
    if not matches:
        return ""
    lines = [
        "# CASE MEMORY",
        "# This system's observed past runs on similar tasks/failures. Learn from "
        "them: prefer directions marked [+] (helped), and do NOT repeat ones "
        "marked [-] (regressed the objective).",
        "# Observed runs outrank paper claims "
        "when they conflict.",
    ]
    for m in matches:
        c = m.case
        lines.append(
            f"- {_VERDICT_MARK.get(c.verdict, '[?]')} {c.edit_summary} "
            f"(task: {c.task[:60]}; project: {c.project or 'unknown'}; "
            f"robot: {c.robot or 'unknown'}; sim {m.relevance_score:.2f}) "
            # `provenance` is a NEW field (agentic-data upgrade 1) — same
            # defensive getattr as diagnose._render_kg_context so a case
            # object lacking it degrades to the least-trusted tag
            # (evidence_tag(None) → llm-inferred tier) instead of crashing.
            f"{evidence_tag(getattr(c, 'provenance', None))}"
        )
    return "\n".join(lines)
