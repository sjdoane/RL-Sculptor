"""§Ship 37: KG case-memory — record run-learnings + retrieve them by
similarity. GPU/model-free: the embedding encoder is avoided by pre-seeding
embeddings and monkeypatching the query encoder."""
from __future__ import annotations

import types

import numpy as np

from sculptor.kg import cases as C
from sculptor.kg.schema import (
    Edge,
    Paper,
    Relation,
    RunCase,
    Technique,
    make_failure_mode_id,
    make_paper_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG


def _outcome(i: int, fms: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(iter_index=i, failure_modes=fms, edit_count=1)


def test_verdict_thresholds() -> None:
    assert C._verdict(None) == "unknown"
    assert C._verdict(0.3) == "helped"
    assert C._verdict(-0.3) == "regressed"
    assert C._verdict(0.0) == "neutral"


def test_record_run_cases_writes_nodes_and_edges(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    # 3 iters, fitness 0.2 -> 0.5 -> 0.3. Forward attribution:
    #   iter0 delta=+0.3 helped; iter1 delta=-0.2 regressed; iter2 no-next unknown.
    result = types.SimpleNamespace(
        completed_iters=[
            _outcome(0, ["reward_hacking"]),
            _outcome(1, []),
            _outcome(2, ["flat_reward_plateau"]),
        ],
        fitness_history=[0.2, 0.5, 0.3],
    )
    n = C.record_run_cases(store, task="kick forward", result=result, nonce="t1")
    assert n == 3

    nodes = store.find_nodes(kind=RunCase.kind)
    assert len(nodes) == 3
    assert sorted(c.verdict for c in nodes) == ["helped", "regressed", "unknown"]
    # only iters with failure modes get INSTANTIATES edges (iter0, iter2).
    assert store.count_edges(Relation.INSTANTIATES) == 2
    # the edge targets the failure-mode node id.
    hack = [c for c in nodes if c.verdict == "helped"][0]
    nbrs = store.neighbors(hack.id, relation=Relation.INSTANTIATES, direction="out")
    assert nbrs and nbrs[0][1] == make_failure_mode_id("reward_hacking")
    store.close()


def test_record_creates_missing_failure_mode_node_no_dangling_edge(tmp_path) -> None:
    """§KG integrity: a diagnoser-flagged failure mode with no paper-derived FailureMode
    node must still get a (stub) node, so the case→failure INSTANTIATES edge does not
    DANGLE. (Live KG had 48 such dangling edges to 4 absent failure ids; the viz had to
    tolerate broken edges. This keeps the case silo self-consistent.)"""
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["premature_termination"])],
        fitness_history=[0.2, 0.5],
    )
    C.record_run_cases(store, task="balance on one leg", result=result, nonce="t1")
    fm_id = make_failure_mode_id("premature_termination")
    assert store.has_node(fm_id)                       # stub FailureMode node created
    node_ids = {n.id for n in store.find_nodes()}
    for case in store.find_nodes(kind=RunCase.kind):
        for nbr in store.neighbors(case.id, relation=Relation.INSTANTIATES, direction="out"):
            assert nbr[1] in node_ids, f"dangling edge to {nbr[1]}"   # no orphan edge
    store.close()


def test_record_does_not_credit_a_reverted_edit(tmp_path) -> None:
    """§Ship 37 review (HIGH): under Ship-36 revert-on-regression, a regressing
    edit at iter N is discarded and iter N+1 RE-MEASURES the best-so-far reward.
    The +delta rebound must NOT be attributed to iter N's edit as 'helped' —
    the verdict must be 'unknown' (the edit's effect was never measured)."""
    store = SculptorKG(tmp_path / "kg.db")

    def rev_outcome(i, fms, reverted=False):
        return types.SimpleNamespace(iter_index=i, failure_modes=fms,
                                     edit_count=1, reverted_to_best=reverted)

    # iter0 regresses; iter1 reverts → re-measures high fitness 0.5.
    result = types.SimpleNamespace(
        completed_iters=[rev_outcome(0, ["reward_hacking"]),
                         rev_outcome(1, [], reverted=True)],
        fitness_history=[0.2, 0.5],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="r")
    nodes = store.find_nodes(kind=RunCase.kind)
    assert len(nodes) == 1                       # iter1 (no fms, no fwd delta) skipped
    assert nodes[0].verdict == "unknown"         # NOT "helped"
    assert nodes[0].fitness_delta is None
    store.close()


def test_record_skips_iterations_with_no_learning(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    # single iter, no failure modes, no next-iter delta → nothing to learn.
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, [])], fitness_history=[0.2],
    )
    assert C.record_run_cases(store, task="trot", result=result, nonce="t2") == 0
    assert store.find_nodes(kind=RunCase.kind) == []
    store.close()


def test_cases_accumulate_across_runs(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["x"]), _outcome(1, ["y"])],
        fitness_history=[0.2, 0.4],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="run1")
    C.record_run_cases(store, task="kick", result=result, nonce="run2")
    # distinct nonce → distinct ids → cases accumulate (not overwrite).
    assert len(store.find_nodes(kind=RunCase.kind)) == 4
    store.close()


def test_query_cases_ranks_by_cosine(tmp_path, monkeypatch) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    a = RunCase(id="case:a", task="kick", symptom="reward_hacking",
                verdict="regressed", edit_summary="iter 2: tried X; regressed")
    b = RunCase(id="case:b", task="trot", symptom="falls",
                verdict="helped", edit_summary="iter 1: tried Y; helped")
    store.add_node(a)
    store.add_node(b)
    # pre-seed embeddings so _ensure_case_embeddings never calls the model.
    store.set_embedding("case:a", C.EMBEDDING_MODEL, np.array([1.0, 0.0], np.float32))
    store.set_embedding("case:b", C.EMBEDDING_MODEL, np.array([0.0, 1.0], np.float32))
    monkeypatch.setattr(C, "_embed_text",
                        lambda text, model_name=None: np.array([1.0, 0.0], np.float32))

    matches = C.query_cases("kick", top_k=2, store=store)
    assert [m.case.id for m in matches] == ["case:a", "case:b"]
    assert matches[0].relevance_score > matches[1].relevance_score
    # similarity floor drops the orthogonal case.
    floored = C.query_cases("kick", top_k=2, store=store, min_similarity=0.5)
    assert [m.case.id for m in floored] == ["case:a"]
    store.close()


def test_query_cases_empty_store_returns_empty(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    assert C.query_cases("anything", store=store) == []
    store.close()


def test_render_case_context_marks_verdicts() -> None:
    matches = [C.CaseMatch(
        case=RunCase(id="c", task="kick forward", verdict="regressed",
                     edit_summary="iter 2: did X; fitness then regressed (-0.20)"),
        relevance_score=0.81,
    )]
    out = C._render_case_context(matches)
    assert "CASE MEMORY" in out and "[-]" in out and "iter 2" in out
    assert C._render_case_context([]) == ""


# ── §2026-07-03 case-content upgrade ───────────────────────────────────────
def _rich_outcome(i, fms, edits=(), components=None, reverted=False):
    return types.SimpleNamespace(
        iter_index=i, failure_modes=fms, edit_count=len(edits),
        applied_edits=list(edits), fitness_components=components,
        reverted_to_best=reverted,
    )


def test_progress_breaks_fitness_ties_in_attribution(tmp_path) -> None:
    """Below the completion gate the spec fitness is 0.0 everywhere — the
    dense progress channel must supply the verdict (the tuck-jump E2E
    recorded 12 straight neutral/unknown cases without this)."""
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[
            _rich_outcome(10, ["reward_hacking"], ["decrease stance_weight"]),
            _rich_outcome(11, ["component_imbalance"], ["increase launch_weight"]),
            _rich_outcome(12, ["static_equilibrium"], ["add flight_bonus"]),
        ],
        fitness_history=[0.0, 0.0, 0.0],
        progress_history=[0.0196, 0.0, 0.008],
    )
    C.record_run_cases(store, task="tuck jump", result=result, nonce="p1")
    by_iter = {c.edit_summary.split(":")[0]: c
               for c in store.find_nodes(kind=RunCase.kind)}
    assert by_iter["iter 10"].verdict == "regressed"     # progress 0.0196→0.0
    assert by_iter["iter 10"].progress_delta < 0
    assert "progress" in by_iter["iter 10"].edit_summary
    assert by_iter["iter 11"].verdict == "helped"        # progress 0.0→0.008
    assert by_iter["iter 12"].verdict == "unknown"       # last iter
    store.close()


def test_fitness_still_dominates_progress_when_it_moves(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_rich_outcome(0, ["x"], ["add a"]),
                         _rich_outcome(1, [], ["add b"])],
        fitness_history=[0.2, 0.5],
        progress_history=[0.9, 0.1],   # progress fell, but fitness ROSE
    )
    C.record_run_cases(store, task="kick", result=result, nonce="p2")
    cases = sorted(store.find_nodes(kind=RunCase.kind),
                   key=lambda c: c.edit_summary)
    assert cases[0].verdict == "helped"                  # fitness wins
    assert "fitness" in cases[0].edit_summary
    store.close()


def test_case_records_edit_identities_and_behavior_signature(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    comps = {"apex_gain_m_mean": 0.0381, "frac_launched": 0.0,
             "c_returned": 0.9721, "note": "not-a-number",
             "progress_score": 0.0196}
    result = types.SimpleNamespace(
        completed_iters=[
            _rich_outcome(10, ["static_equilibrium"],
                          ["decrease stance_weight", "increase launch_weight"],
                          components=comps),
        ],
        fitness_history=[0.0],
        progress_history=[0.0196],
    )
    C.record_run_cases(store, task="tuck jump", result=result, nonce="p3")
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.edits == ["decrease stance_weight", "increase launch_weight"]
    assert "decrease stance_weight" in case.edit_summary
    assert case.behavior["apex_gain_m_mean"] == 0.0381
    assert "note" not in case.behavior                   # non-numeric dropped
    assert "apex_gain_m_mean=0.0381" in case.edit_summary
    # the embedded/matched text carries the edit identities.
    assert "decrease stance_weight" in C._case_text(case)
    store.close()


def test_blind_run_records_what_was_tried(tmp_path) -> None:
    """No fitness_fn (blind loop): cases still record symptom + edits with
    verdict 'unknown' — what was tried is itself memory."""
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_rich_outcome(0, ["reward_hacking"], ["clip tuck_reward"])],
        fitness_history=[],
        progress_history=[],
    )
    n = C.record_run_cases(store, task="hop", result=result, nonce="b1")
    assert n == 1
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.verdict == "unknown"
    assert case.edits == ["clip tuck_reward"]
    store.close()


def test_pre_upgrade_rows_still_load(tmp_path) -> None:
    """Old RunCase rows (no edits/progress/behavior keys) must round-trip
    through the store with the new dataclass defaults."""
    import json as _json
    import sqlite3 as _sqlite3

    store = SculptorKG(tmp_path / "kg.db")
    old_blob = {"task": "kick", "robot": "", "symptom": "falls",
                "failure_modes": ["x"], "edit_summary": "iter 1: old shape",
                "fitness_before": 0.1, "fitness_after": 0.2,
                "fitness_delta": 0.1, "verdict": "helped",
                "created_at": 0.0}
    conn = _sqlite3.connect(store.db_path)
    conn.execute("INSERT INTO nodes (id, kind, data) VALUES (?, ?, ?)",
                 ("case:old", "RunCase", _json.dumps(old_blob)))
    conn.commit(); conn.close()
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.verdict == "helped"
    assert case.edits == [] and case.behavior == {}
    assert case.progress_delta is None
    store.close()


# ── §Agentic-data upgrade 1: provenance + evidence tag on cases ───────────
def test_recorded_case_has_observed_run_provenance(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"])],
        fitness_history=[0.2, 0.5],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="prov1")
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.provenance == "observed_run"
    store.close()


def test_render_case_context_includes_evidence_tag_and_header_rule() -> None:
    matches = [C.CaseMatch(
        case=RunCase(id="c", task="kick forward", verdict="helped",
                     edit_summary="iter 2: did X; fitness then helped (+0.20)"),
        relevance_score=0.81,
    )]
    out = C._render_case_context(matches)
    assert "[evidence: observed run]" in out
    assert "outrank paper claims" in out


# ── §Agentic-data upgrade 2: usage-based enrichment ───────────────────────
def _outcome_with_paths(i, fms, reward_path=None, env_spec=None):
    return types.SimpleNamespace(
        iter_index=i, failure_modes=fms, edit_count=0,
        reward_path_trained=reward_path, env_spec_trained=env_spec,
    )


def test_helped_iter_with_references_increments_useful_citations(tmp_path) -> None:
    """A 'helped' iteration whose kept reward cites an arxiv_id must
    increment useful_citations on that paper's INTRODUCES-d techniques —
    the run just showed the citation paid off, not merely that a paper
    claims it works."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    # Forward attribution needs a NEXT iteration to measure iter0's edit
    # against (cases.py: verdict is judged by iter N+1's fitness) — a
    # single-iter run always attributes 'unknown'.
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.2, 0.5],  # iter0 delta +0.3 -> helped
    )
    n = C.record_run_cases(
        store, task="kick", result=result, nonce="cite1",
        iter_references={0: ["1707.06347"]},
    )
    assert n == 1
    fetched = store.get_node(tech.id)
    assert fetched.useful_citations == 1


def test_regressed_iter_with_references_does_not_increment(tmp_path) -> None:
    """Only 'helped' verdicts credit citations — a regressed iteration's
    references must NOT bump useful_citations (it didn't pay off)."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.5, 0.2],  # iter0 delta -0.3 -> regressed
    )
    n = C.record_run_cases(
        store, task="kick", result=result, nonce="cite2",
        iter_references={0: ["1707.06347"]},
    )
    assert n == 1
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.verdict == "regressed"  # sanity: the gate condition actually applies
    fetched = store.get_node(tech.id)
    assert fetched.useful_citations == 0


def test_no_iter_references_leaves_useful_citations_untouched(tmp_path) -> None:
    """Default call (no iter_references kwarg) must not touch
    useful_citations at all — the sculpt.py call site is unchanged and a
    follow-up wires the real value through."""
    store = SculptorKG(tmp_path / "kg.db")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(tech)
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"])],
        fitness_history=[0.2, 0.5],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="cite3")
    fetched = store.get_node(tech.id)
    assert fetched.useful_citations == 0


def test_useful_citations_increments_across_multiple_helped_iters(tmp_path) -> None:
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("2020.00001"), arxiv_id="2020.00001", title="Shaping")
    tech = Technique(id=make_technique_id("shaping"), name="shaping")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    # 3 iters so BOTH iter0 and iter1 have a next-iter measurement to be
    # forward-attributed against (iter2 itself attributes 'unknown', which
    # is fine — it isn't referenced in iter_references below).
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["x"]), _outcome(1, ["y"]), _outcome(2, [])],
        fitness_history=[0.1, 0.4, 0.7, 1.0],  # iter0, iter1 both helped
    )
    C.record_run_cases(
        store, task="kick", result=result, nonce="cite4",
        iter_references={0: ["2020.00001"], 1: ["2020.00001"]},
    )
    fetched = store.get_node(tech.id)
    assert fetched.useful_citations == 2


# ── §Agentic-data upgrade 4: freshness metadata ───────────────────────────
def test_reward_version_and_env_spec_version_populate_from_outcome(tmp_path) -> None:
    from pathlib import Path as _Path

    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[
            _outcome_with_paths(
                0, ["x"],
                reward_path=_Path("/proj/rewards/v3.py"),
                env_spec="v2",
            ),
        ],
        fitness_history=[0.1, 0.4],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="fresh1")
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.reward_version == "v3"
    assert case.env_spec_version == "v2"
    store.close()


def test_reward_version_none_when_outcome_lacks_trained_path(tmp_path) -> None:
    """Old-shape outcomes (no reward_path_trained/env_spec_trained
    attrs, or None values) must leave the fields None — no call-site
    changes required."""
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["x"])],  # plain _outcome has neither attr
        fitness_history=[0.1, 0.4],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="fresh2")
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.reward_version is None
    assert case.env_spec_version is None
    store.close()


def test_pre_upgrade_case_rows_load_without_freshness_fields(tmp_path) -> None:
    """Old RunCase rows (no reward_version/env_spec_version keys) must
    round-trip with None defaults."""
    import json as _json
    import sqlite3 as _sqlite3

    store = SculptorKG(tmp_path / "kg.db")
    old_blob = {"task": "kick", "symptom": "falls", "verdict": "helped"}
    conn = _sqlite3.connect(store.db_path)
    conn.execute("INSERT INTO nodes (id, kind, data) VALUES (?, ?, ?)",
                 ("case:old2", "RunCase", _json.dumps(old_blob)))
    conn.commit(); conn.close()
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.reward_version is None
    assert case.env_spec_version is None
    assert case.provenance == "observed_run"
    store.close()


# ── §Agentic-data upgrade 4: recency tiebreak in query_cases ──────────────
def test_query_cases_recency_tiebreak_within_gap(tmp_path, monkeypatch) -> None:
    """Two cases within the 0.02 similarity gap: the more recently-created
    one sorts first (stable-sort nudge, not a hard filter)."""
    store = SculptorKG(tmp_path / "kg.db")
    older = RunCase(id="case:older", task="kick", symptom="falls",
                    verdict="helped", edit_summary="old case",
                    created_at=100.0)
    newer = RunCase(id="case:newer", task="kick", symptom="falls",
                    verdict="helped", edit_summary="new case",
                    created_at=200.0)
    store.add_node(older)
    store.add_node(newer)
    # Nearly identical vectors -> similarities within the 0.02 gap.
    store.set_embedding("case:older", C.EMBEDDING_MODEL,
                       np.array([1.0, 0.0], np.float32))
    store.set_embedding("case:newer", C.EMBEDDING_MODEL,
                       np.array([0.9999, 0.0141], np.float32))
    monkeypatch.setattr(C, "_embed_text",
                        lambda text, model_name=None: np.array([1.0, 0.0], np.float32))

    matches = C.query_cases("kick", top_k=2, store=store)
    assert [m.case.id for m in matches] == ["case:newer", "case:older"]
    store.close()


def test_query_cases_recency_does_not_override_a_real_similarity_gap(tmp_path, monkeypatch) -> None:
    """A genuinely closer OLD case still outranks a fresher but weaker
    match — recency is a tiebreak, never a hard override."""
    store = SculptorKG(tmp_path / "kg.db")
    older_close = RunCase(id="case:older_close", task="kick", symptom="falls",
                         verdict="helped", edit_summary="old but close",
                         created_at=1.0)
    newer_far = RunCase(id="case:newer_far", task="kick", symptom="falls",
                       verdict="helped", edit_summary="new but far",
                       created_at=999.0)
    store.add_node(older_close)
    store.add_node(newer_far)
    store.set_embedding("case:older_close", C.EMBEDDING_MODEL,
                       np.array([1.0, 0.0], np.float32))
    store.set_embedding("case:newer_far", C.EMBEDDING_MODEL,
                       np.array([0.0, 1.0], np.float32))
    monkeypatch.setattr(C, "_embed_text",
                        lambda text, model_name=None: np.array([1.0, 0.0], np.float32))

    matches = C.query_cases("kick", top_k=2, store=store)
    assert [m.case.id for m in matches] == ["case:older_close", "case:newer_far"]


# ── §KG-retrieval fix 4: outcome-stats ranking ────────────────────────────
def test_helped_iter_increments_outcome_stats_for_its_failure_mode(tmp_path) -> None:
    """A 'helped' iteration that cites a paper via iter_references must tally
    a per-FailureMode helped count on that paper's INTRODUCES-d technique,
    scoped to the failure mode(s) THIS iteration actually flagged."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.2, 0.5],  # iter0 delta +0.3 -> helped
    )
    C.record_run_cases(
        store, task="kick", result=result, nonce="stats1",
        iter_references={0: ["1707.06347"]},
    )
    fetched = store.get_node(tech.id)
    fm_id = make_failure_mode_id("reward_hacking")
    assert fetched.outcome_stats == {fm_id: {"helped": 1, "regressed": 0}}
    # useful_citations still increments too — outcome_stats is additive,
    # not a replacement for the existing signal.
    assert fetched.useful_citations == 1


def test_record_run_cases_rerun_does_not_double_count(tmp_path) -> None:
    """§KG-retrieval fix 4 idempotency: a resumed/re-run stage re-reports the
    same physical iterations through record_run_cases. The case node is
    upserted, but the cumulative counters (outcome_stats, useful_citations)
    must fire exactly once per case id — both with an explicit nonce and
    with the iter_dir-derived deterministic nonce."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    runs_root = tmp_path / "runs"
    o0 = _outcome(0, ["reward_hacking"])
    o0.iter_dir = runs_root / "iter_0"
    o1 = _outcome(1, [])
    o1.iter_dir = runs_root / "iter_1"
    result = types.SimpleNamespace(
        completed_iters=[o0, o1],
        fitness_history=[0.2, 0.5],  # iter0 delta +0.3 -> helped
    )
    refs = {0: ["1707.06347"]}
    fm_id = make_failure_mode_id("reward_hacking")

    # No explicit nonce: derived from the runs root, so both calls agree.
    C.record_run_cases(store, task="kick", result=result, iter_references=refs)
    C.record_run_cases(store, task="kick", result=result, iter_references=refs)
    fetched = store.get_node(tech.id)
    assert fetched.outcome_stats == {fm_id: {"helped": 1, "regressed": 0}}
    assert fetched.useful_citations == 1

    # Explicit nonce re-run over the same ids: still counted once.
    C.record_run_cases(
        store, task="kick2", result=result, nonce="fixed", iter_references=refs)
    C.record_run_cases(
        store, task="kick2", result=result, nonce="fixed", iter_references=refs)
    fetched = store.get_node(tech.id)
    assert fetched.outcome_stats == {fm_id: {"helped": 2, "regressed": 0}}
    assert fetched.useful_citations == 2


def test_rerecord_unknown_to_helped_updates_counters_once(tmp_path) -> None:
    """A last iteration can be unknown on the first write, then acquire a
    measured next iteration on resume. Its persisted case and counters must
    move together, exactly once."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"),
                  arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id,
                        relation=Relation.INTRODUCES))
    refs = {0: ["1707.06347"]}
    first = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"])],
        fitness_history=[0.2])
    C.record_run_cases(
        store, task="kick", result=first, nonce="resume", iter_references=refs)
    assert store.get_node(tech.id).useful_citations == 0

    resumed = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.2, 0.5])
    C.record_run_cases(
        store, task="kick", result=resumed, nonce="resume",
        iter_references=refs)
    C.record_run_cases(
        store, task="kick", result=resumed, nonce="resume",
        iter_references=refs)
    fetched = store.get_node(tech.id)
    fm_id = make_failure_mode_id("reward_hacking")
    assert fetched.useful_citations == 1
    assert fetched.outcome_stats == {
        fm_id: {"helped": 1, "regressed": 0}}


def test_legacy_helped_case_does_not_double_count_on_reference_migration(
    tmp_path,
) -> None:
    """Pre-reference live rows already affected aggregate counters. Their
    first upgraded re-record persists references without crediting twice."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"),
                  arxiv_id="1707.06347", title="PPO")
    fm_id = make_failure_mode_id("reward_hacking")
    tech = Technique(
        id=make_technique_id("rsi"), name="rsi", useful_citations=1,
        outcome_stats={fm_id: {"helped": 1, "regressed": 0}})
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id,
                        relation=Relation.INTRODUCES))
    store.add_node(RunCase(
        id=C.make_run_case_id("kick", 0, "legacy"),
        task="kick", symptom="reward_hacking",
        failure_modes=["reward_hacking"], verdict="helped"))

    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.2, 0.5])
    C.record_run_cases(
        store, task="kick", result=result, nonce="legacy",
        iter_references={0: ["1707.06347"]})

    fetched = store.get_node(tech.id)
    assert fetched.useful_citations == 1
    assert fetched.outcome_stats == {
        fm_id: {"helped": 1, "regressed": 0}}
    migrated = store.get_node(C.make_run_case_id("kick", 0, "legacy"))
    assert migrated.references == ["1707.06347"]
    assert migrated.attribution_version == 1


def test_regressed_iter_increments_outcome_stats_but_not_useful_citations(
    tmp_path,
) -> None:
    """A 'regressed' iteration must bump the regressed counter on
    outcome_stats (a real, negative learning signal) while leaving
    useful_citations untouched (that field is 'helped'-only)."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, ["reward_hacking"]), _outcome(1, [])],
        fitness_history=[0.5, 0.2],  # iter0 delta -0.3 -> regressed
    )
    C.record_run_cases(
        store, task="kick", result=result, nonce="stats2",
        iter_references={0: ["1707.06347"]},
    )
    fetched = store.get_node(tech.id)
    fm_id = make_failure_mode_id("reward_hacking")
    assert fetched.outcome_stats == {fm_id: {"helped": 0, "regressed": 1}}
    assert fetched.useful_citations == 0


def test_outcome_stats_accumulate_across_helped_and_regressed_iters(
    tmp_path,
) -> None:
    """Repeated (helped, regressed) verdicts on the same (technique,
    failure-mode) pair both accumulate in the same counter dict, and a
    verdict on a DIFFERENT failure mode gets its own entry."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("2020.00001"), arxiv_id="2020.00001", title="Shaping")
    tech = Technique(id=make_technique_id("shaping"), name="shaping")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    result = types.SimpleNamespace(
        completed_iters=[
            _outcome(0, ["reward_hacking"]),
            _outcome(1, ["reward_hacking"]),
            _outcome(2, ["sparse_reward"]),
            _outcome(3, []),
        ],
        # iter0 delta +0.3 helped, iter1 delta -0.2 regressed,
        # iter2 delta +0.1 helped.
        fitness_history=[0.1, 0.4, 0.2, 0.3, 1.0],
    )
    C.record_run_cases(
        store, task="kick", result=result, nonce="stats3",
        iter_references={0: ["2020.00001"], 1: ["2020.00001"], 2: ["2020.00001"]},
    )
    fetched = store.get_node(tech.id)
    hack_id = make_failure_mode_id("reward_hacking")
    sparse_id = make_failure_mode_id("sparse_reward")
    assert fetched.outcome_stats[hack_id] == {"helped": 1, "regressed": 1}
    assert fetched.outcome_stats[sparse_id] == {"helped": 1, "regressed": 0}


def test_no_failure_modes_flagged_leaves_outcome_stats_untouched(tmp_path) -> None:
    """A 'helped' iteration with no failure_modes flagged has nothing to
    scope a per-failure-mode tally to — outcome_stats stays empty even
    though useful_citations still increments."""
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(id=make_paper_id("1707.06347"), arxiv_id="1707.06347", title="PPO")
    tech = Technique(id=make_technique_id("rsi"), name="rsi")
    store.add_node(paper)
    store.add_node(tech)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))

    result = types.SimpleNamespace(
        completed_iters=[_outcome(0, []), _outcome(1, [])],
        # delta +0.3 with no failure modes still passes record_run_cases's
        # skip gate (a measurable fitness change alone is enough to record
        # the row) — see the `if not fms and not edits and delta is None
        # and delta_p is None: continue` condition in cases.py.
        fitness_history=[0.2, 0.5],
    )
    C.record_run_cases(
        store, task="kick", result=result, nonce="stats4",
        iter_references={0: ["1707.06347"]},
    )
    fetched = store.get_node(tech.id)
    assert fetched.outcome_stats == {}
    assert fetched.useful_citations == 1
    store.close()


# ── §Env-authoring §10: world-tuple identity ──────────────────────────────
def _outcome_with_world(i, fms, *, sel_hash=None, sel_path=None):
    return types.SimpleNamespace(
        iter_index=i, failure_modes=fms, edit_count=0,
        world_selection_hash=sel_hash, world_selection_path=sel_path,
    )


def test_world_identity_populates_from_pinned_selection(tmp_path) -> None:
    import json

    sel = tmp_path / "selection_v5.json"
    # The REAL store serializes ArtifactRef versions as "v<N>" strings
    # (verifier finding: an integer-version fixture passed while
    # production silently recorded None).
    sel.write_text(json.dumps({
        "selection_version": 5,
        "refs": {"world": {"version": "v3"}, "task": {"version": "v1"}},
        "tuple_hash": "abc123",
    }))
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[_outcome_with_world(
            0, ["x"], sel_hash="abc123", sel_path=str(sel))],
        fitness_history=[0.1, 0.4],
    )
    C.record_run_cases(store, task="kick", result=result, nonce="w1")
    (case,) = store.find_nodes(kind=RunCase.kind)
    assert case.world_tuple_hash == "abc123"
    assert case.world_version == 3
    assert case.task_version == 1
    store.close()


def test_world_identity_fail_soft_on_legacy_and_dead_path(tmp_path) -> None:
    """Legacy outcomes (no world attrs) and a dead selection path must
    record None identity fields, never raise — world identity is
    freshness evidence, not a record blocker."""
    store = SculptorKG(tmp_path / "kg.db")
    result = types.SimpleNamespace(
        completed_iters=[
            _outcome(0, ["x"]),
            _outcome_with_world(
                1, ["y"], sel_hash="livehash",
                sel_path=str(tmp_path / "missing.json")),
            _outcome(2, []),
        ],
        fitness_history=[0.1, 0.4, 0.5, 0.6],
    )
    n = C.record_run_cases(store, task="kick", result=result, nonce="w2")
    assert n == 2  # the trailing no-failure iter records nothing
    cases = list(store.find_nodes(kind=RunCase.kind))
    legacy = [c for c in cases if c.world_tuple_hash is None]
    dead = [c for c in cases if c.world_tuple_hash == "livehash"]
    assert len(legacy) == 1 and len(dead) == 1
    assert dead[0].world_version is None
    assert dead[0].task_version is None
    store.close()
