"""§Ship 37: KG case-memory — record run-learnings + retrieve them by
similarity. GPU/model-free: the embedding encoder is avoided by pre-seeding
embeddings and monkeypatching the query encoder."""
from __future__ import annotations

import types

import numpy as np

from sculptor.kg import cases as C
from sculptor.kg.schema import Relation, RunCase, make_failure_mode_id
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
