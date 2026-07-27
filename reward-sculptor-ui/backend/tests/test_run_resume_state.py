"""Regression: a RESUMED run must not surface the PRIOR run's iterations as
RUNNING (the long-standing "all previous iters show running at once" bug), and
a stranded mid-loop iter must reconcile to completed.

Root cause: the fs watcher emitted `iter_started` for every on-disk iter_<n>
dir, but `iter_completed` only ever comes from the live subprocess stdout for
the iters it actually runs — so a prior run's iters got `started` with no
matching `completed` and hung "running" forever.
"""
from __future__ import annotations

import json

from backend.services.job_manager import Job
from backend.services.run_manager import (
    _preseed_seen,
    build_iterations_summary,
)


def _mk_iter(runs, n, *, complete: bool) -> None:
    d = runs / f"iter_{n}"
    (d / "rollout").mkdir(parents=True)
    if complete:
        (d / "diagnosis.json").write_text(
            json.dumps({"failure_modes": ["x"], "confidence": 0.5,
                        "proposed_edits": []}), encoding="utf-8")
        (d / "realism_audit.json").write_text(
            json.dumps({"verdict": "ok"}), encoding="utf-8")
        (d / "rollout" / "rollout.mp4").write_bytes(b"\0" * 5000)


def test_preseed_seeds_prior_state_without_emitting(tmp_path):
    """A project with a PRIOR run on disk: the pre-seed populates the dedup
    sets (so the watcher won't re-emit) and emits ZERO events."""
    runs = tmp_path / "runs"; runs.mkdir()
    _mk_iter(runs, 0, complete=True)
    _mk_iter(runs, 1, complete=True)
    _mk_iter(runs, 2, complete=False)          # a half-finished prior iter
    rewards = tmp_path / "rewards"; rewards.mkdir()
    for v in (0, 1, 2, 3):
        (rewards / f"v{v}.py").write_text("x = 1\n", encoding="utf-8")

    job = Job(job_id="t", kind="sculpt_run", project_slug="p", status="running")
    sets = dict(seen_iters=set(), seen_iter_done=set(), seen_rollouts=set(),
                seen_rewards=set(), seen_citations=set(), seen_realism=set())
    _preseed_seen(tmp_path, **sets)

    assert sets["seen_iters"] == {0, 1, 2}             # every dir
    # artifact-keyed sets only include the VALID ones (iter_2 is half-finished):
    assert sets["seen_iter_done"] == {0, 1}
    assert sets["seen_rollouts"] == {0, 1}
    assert sets["seen_realism"] == {0, 1}
    assert sets["seen_rewards"] == {1, 2, 3}           # v>0
    # the pre-seed NEVER emits — no phantom iter_started/edit_applied:
    assert job.events == []


def test_iter_events_reconciles_stranded_running_iter(tmp_path):
    """If a lower iter's `iter_completed` was lost (dropped stdout / crash),
    the sequential loop having advanced past it means it is done — reconcile to
    completed; only the highest-started iter stays running."""
    job = Job(job_id="t2", kind="sculpt_run", project_slug="p", status="running")
    job.emit({"type": "iter_started", "iter": 9, "ts": "t9a"})
    job.emit({"type": "iter_completed", "iter": 9, "ts": "t9b"})
    job.emit({"type": "iter_started", "iter": 10, "ts": "t10"})
    # iter 10's iter_completed is MISSING …
    job.emit({"type": "iter_started", "iter": 11, "ts": "t11"})  # … but the loop moved on

    by = {it["iter_index"]: it for it in build_iterations_summary(job)}
    assert by[9]["status"] == "completed"
    assert by[10]["status"] == "completed"   # reconciled — loop advanced past it
    assert by[11]["status"] == "running"     # current iter, genuinely running


def test_iter_events_keeps_sole_running_iter(tmp_path):
    """The only started iter (no higher one) stays running — no false complete."""
    job = Job(job_id="t3", kind="sculpt_run", project_slug="p", status="running")
    job.emit({"type": "iter_started", "iter": 9, "ts": "t9"})
    by = {it["iter_index"]: it for it in build_iterations_summary(job)}
    assert by[9]["status"] == "running"
