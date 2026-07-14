"""§Ship 33: fitness-in-the-loop (sculpt_run best-by-fitness selection,
plateau / target early-stop, and the diagnoser objective-progress block).

GPU-free: `_run_one_iter` is faked so no training/adapter is needed; we
test only the loop-level fitness logic and the prompt block.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sculptor.sculpt as S
from sculptor.diagnose import _build_preliminary_user_content


def _make_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "rewards").mkdir(parents=True)
    (proj / "runs").mkdir()
    (proj / "reports").mkdir()
    (proj / "rewards" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "rewards" / "v0.py").write_text(
        "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
        encoding="utf-8",
    )
    (proj / "config.toml").write_text(
        "[iteration]\nsteps_per_iter = 10\nseed = 42\n", encoding="utf-8",
    )
    return proj / "config.toml"


def _install_fakes(monkeypatch, fitness_by_iter: dict[int, float]):
    """Patch the heavy collaborators so sculpt_run exercises only its
    own loop logic. The fake _run_one_iter writes a real v<n+1>.py (so
    best-by-fitness current.py repoint has a file to point at) and stamps
    the preset fitness onto the returned IterOutcome."""
    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr(
        "sculptor.run_context.capture_run_context",
        lambda *a, **k: {}, raising=True,
    )
    monkeypatch.setattr(
        "sculptor.run_context.write_run_context",
        lambda *a, **k: Path("run_context.json"), raising=True,
    )

    def fake_iter(**kw):
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
        trained = rewards_dir / f"v{i}.py"      # the reward trained this iter
        edit = rewards_dir / f"v{i + 1}.py"     # the (untested) edit produced
        edit.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
            encoding="utf-8",
        )
        # Mirror the real iter: current.py re-exports the newest reward.
        S._write_current_reexport(rewards_dir, edit)
        return S.IterOutcome(
            iter_index=i,
            iter_dir=kw["runs_dir"] / f"iter_{i}",
            reward_path_before=rewards_dir / "current.py",
            reward_path_after=edit,
            primary_metric=0.0,
            behavior={},
            failure_modes=[],
            edit_count=1,
            fitness=fitness_by_iter[i],
            reward_path_trained=trained,
        )

    monkeypatch.setattr(S, "_run_one_iter", fake_iter)


def test_best_by_fitness_selection_and_plateau_stop(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    # best is iter 1 (0.5); iters 2,3 don't beat it → plateau at patience=2.
    _install_fakes(monkeypatch, {0: 0.2, 1: 0.5, 2: 0.4, 3: 0.45, 4: 0.41})
    res = S.sculpt_run(
        cfg_path, "goal", iterations=6, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=2,
    )
    assert res.fitness_history == [0.2, 0.5, 0.4, 0.45]
    assert res.best_fitness == 0.5
    assert res.best_fitness_iter == 1
    assert res.early_stopped and "plateau" in res.early_stop_reason
    assert res.iterations_run == 4          # stopped early, not all 6
    # current.py must re-export the BEST iter's TRAINED reward (iter 1
    # trained v1), NOT the last edit (v4) nor the best iter's untested
    # edit (v2).
    current = (cfg_path.parent / "rewards" / "current.py").read_text()
    assert "v1" in current
    assert "v2" not in current and "v4" not in current


def test_fitness_target_early_stop(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    _install_fakes(monkeypatch, {0: 0.3, 1: 0.6, 2: 0.9})
    res = S.sculpt_run(
        cfg_path, "goal", iterations=6, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_target=0.5,
    )
    assert res.iterations_run == 2          # hit target at iter 1
    assert res.early_stopped and "target" in res.early_stop_reason
    assert res.best_fitness == 0.6


def test_no_fitness_fn_is_unchanged_blind_default(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    _install_fakes(monkeypatch, {0: 0.2, 1: 0.9})
    res = S.sculpt_run(cfg_path, "goal", iterations=2, no_kg=True)
    assert res.fitness_history == []        # nothing tracked
    assert res.best_fitness is None and res.best_fitness_iter is None
    assert not res.early_stopped
    assert res.iterations_run == 2
    # blind default: current.py points at the LAST reward (v2.py).
    current = (cfg_path.parent / "rewards" / "current.py").read_text()
    assert "v2" in current


def test_diagnose_objective_progress_block_present_and_absent():
    base = dict(
        behavior_goal="trot forward", reward_spec={}, metrics={},
        behavior={}, behavior_metric_names=[], contract_text="", keyframes=[],
    )
    with_block = _build_preliminary_user_content(
        **base,
        objective_progress={"current": 0.31, "best_so_far": 0.31,
                            "last": 0.12, "delta": 0.19},
    )
    text = with_block[0]["text"]
    assert "OBJECTIVE_TASK_PROGRESS" in text
    assert "current=0.31" in text and "delta_vs_previous=0.19" in text

    without = _build_preliminary_user_content(**base)
    assert "OBJECTIVE_TASK_PROGRESS" not in without[0]["text"]


# ── §Ship 36: F1 revert-on-regression + F2 component breakdown ───────────


def _install_recording_fake(monkeypatch, fitness_by_iter, seen_revert):
    """Like `_install_fakes` but records the `revert_base` each iter was
    called with, so the loop's best-first revert logic is observable."""
    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr("sculptor.run_context.capture_run_context",
                        lambda *a, **k: {}, raising=True)
    monkeypatch.setattr("sculptor.run_context.write_run_context",
                        lambda *a, **k: Path("run_context.json"), raising=True)

    def fake_iter(**kw):
        seen_revert.append(kw.get("revert_base"))
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
        trained = rewards_dir / f"v{i}.py"
        edit = rewards_dir / f"v{i + 1}.py"
        edit.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
            encoding="utf-8",
        )
        S._write_current_reexport(rewards_dir, edit)
        return S.IterOutcome(
            iter_index=i, iter_dir=kw["runs_dir"] / f"iter_{i}",
            reward_path_before=rewards_dir / "current.py",
            reward_path_after=edit, primary_metric=0.0, behavior={},
            failure_modes=[], edit_count=1, fitness=fitness_by_iter[i],
            reward_path_trained=trained,
        )

    monkeypatch.setattr(S, "_run_one_iter", fake_iter)


def test_fitness_revert_targets_best_reward_on_regression(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    seen: list = []
    # best is iter 1; iter 2 regresses → iter 3 must edit from iter 1's reward.
    _install_recording_fake(monkeypatch, {0: 0.2, 1: 0.5, 2: 0.4, 3: 0.45}, seen)
    res = S.sculpt_run(
        cfg_path, "goal", iterations=6, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=2,
    )
    # iters 0,1 set new bests → no revert; iter 2 regressed (best=iter1) →
    # iter 3 is handed iter 1's TRAINED reward (v1.py) as its edit base.
    assert seen[0] is None and seen[1] is None and seen[2] is None
    assert seen[3] is not None and seen[3].name == "v1.py"
    assert res.iterations_run == 4          # plateau still stops at iter 3


def test_observe_only_never_reverts(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_recording_fake(monkeypatch, {0: 0.2, 1: 0.5, 2: 0.4, 3: 0.45}, seen)
    S.sculpt_run(
        cfg_path, "goal", iterations=4, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=2,
        fitness_observe_only=True,
    )
    # observe mode is purely passive: the edit base is NEVER reverted.
    assert all(s is None for s in seen), seen


def test_fitness_revert_disabled_keeps_forward_edit_base(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_recording_fake(monkeypatch, {0: 0.2, 1: 0.5, 2: 0.4, 3: 0.45}, seen)
    S.sculpt_run(
        cfg_path, "goal", iterations=4, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=2,
        fitness_revert=False,
    )
    assert all(s is None for s in seen), seen


def test_fitness_components_for_prompt_filters():
    from sculptor.sculpt import _fitness_components_for_prompt
    assert _fitness_components_for_prompt(None) is None
    assert _fitness_components_for_prompt({"spec_score": 0.5}) is None
    out = _fitness_components_for_prompt({
        "spec_score": 0.5, "spec_name": "g1_kick", "error": None,
        "capture": {"step_dt": 0.02}, "uprightness": 0.99,
        "burst_p95": 4.2, "leg_subset": 1.0, "bad": float("nan"),
    })
    assert out == {"uprightness": 0.99, "burst_p95": 4.2, "leg_subset": 1.0}


# ── §D24 (F4): runtime fitness/criterion contradiction detector ──────────


def test_is_fitness_contradiction_true_false_none_and_eps_edge():
    from sculptor.sculpt import (
        FITNESS_CONTRADICTION_EPS,
        _is_fitness_contradiction,
    )

    assert FITNESS_CONTRADICTION_EPS == 0.05
    # criterion passed + fitness at/near zero → contradiction.
    assert _is_fitness_contradiction(True, 0.0) is True
    # eps boundary is inclusive (<=), not a strict <.
    assert _is_fitness_contradiction(True, FITNESS_CONTRADICTION_EPS) is True
    # just above eps → no longer a contradiction.
    assert _is_fitness_contradiction(
        True, FITNESS_CONTRADICTION_EPS + 1e-6) is False
    # a genuinely good fitness never contradicts.
    assert _is_fitness_contradiction(True, 0.9) is False
    # criterion did not pass → never a contradiction, whatever the fitness.
    assert _is_fitness_contradiction(False, 0.0) is False
    assert _is_fitness_contradiction(False, None) is False
    # no fitness_fn wired (None) → nothing to contradict against.
    assert _is_fitness_contradiction(True, None) is False


def _make_iter_outcome_with_rollout(
    tmp_path, *, iter_index, fitness, components=None, mean_return=0.9,
):
    """Minimal on-disk iter dir (checkpoint + rollout/behavior.json) plus
    the matching in-memory IterOutcome, so `_select_stage_final_iter` can
    both evaluate the criterion (reads behavior.json) and see `.fitness`/
    `.fitness_components` (carried on the outcome, exactly like the real
    loop populates them at ~1392)."""
    from sculptor.sculpt import IterOutcome

    iter_dir = tmp_path / f"iter_{iter_index}"
    (iter_dir / "rollout").mkdir(parents=True)
    (iter_dir / "checkpoint.pt").write_bytes(b"fake-ckpt")
    (iter_dir / "rollout" / "behavior.json").write_text(
        json.dumps({"mean_return": mean_return}))
    return IterOutcome(
        iter_index=iter_index,
        iter_dir=iter_dir,
        reward_path_before=tmp_path / "rewards" / "current.py",
        reward_path_after=None,
        primary_metric=mean_return,
        behavior={"mean_return": mean_return},
        failure_modes=[],
        edit_count=0,
        fitness=fitness,
        fitness_components=components,
    )


def test_select_stage_final_iter_emits_fitness_contradiction(tmp_path):
    """§D24 (F4) / D20 hollow-success: a candidate whose criterion passes
    (mean_return > 0.5) but whose objective fitness is ~0 must (a) emit a
    `fitness_contradiction` SCULPT-EVENT via the caller's `emit` sink and
    (b) drop a durable `fitness_contradiction.json` flag in THAT iter's
    own directory — not the stage dir, not the winner's dir if a
    different iter is selected."""
    from sculptor.sculpt import _select_stage_final_iter

    class _Stage:
        name = "torso_righting"
        success_criterion = "metric > 0.5"

    hollow = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=0, fitness=0.0,
        components={"gate_upright_frac": 1.0, "gate_reached_035": 0.0},
        mean_return=0.9,
    )
    seen_events: list = []
    selected, criterion_ok, source, _err, _missing, _mismatch = (
        _select_stage_final_iter(
            [hollow], _Stage(), stage_dir=None, emit=seen_events.append,
        )
    )
    assert selected is hollow
    assert criterion_ok is True  # the hollow "success" the detector traps

    contradiction_events = [
        e for e in seen_events if e.get("type") == "fitness_contradiction"
    ]
    assert len(contradiction_events) == 1, seen_events
    ev = contradiction_events[0]
    assert ev["stage_name"] == "torso_righting"
    assert ev["iter"] == 0
    assert ev["fitness"] == 0.0
    assert ev["criterion"] == "metric > 0.5"
    assert ev["components"] == {
        "gate_upright_frac": 1.0, "gate_reached_035": 0.0,
    }

    flag_path = hollow.iter_dir / "fitness_contradiction.json"
    assert flag_path.is_file()
    assert json.loads(flag_path.read_text()) == ev


def test_select_stage_final_iter_no_contradiction_on_healthy_fitness(
        tmp_path):
    """The same criterion-passing iter, but with a healthy (non-zero)
    fitness, must NOT be flagged — the detector traps the D20 pattern
    specifically, not every criterion pass."""
    from sculptor.sculpt import _select_stage_final_iter

    class _Stage:
        name = "torso_righting"
        success_criterion = "metric > 0.5"

    healthy = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=0, fitness=0.8, mean_return=0.9,
    )
    seen_events: list = []
    _select_stage_final_iter(
        [healthy], _Stage(), stage_dir=None, emit=seen_events.append,
    )
    assert not any(
        e.get("type") == "fitness_contradiction" for e in seen_events)
    assert not (healthy.iter_dir / "fitness_contradiction.json").is_file()


def test_select_stage_final_iter_no_contradiction_when_fitness_none(
        tmp_path):
    """No fitness_fn wired for the stage (`fitness=None`) — nothing to
    contradict the criterion against, so the detector stays silent."""
    from sculptor.sculpt import _select_stage_final_iter

    class _Stage:
        name = "torso_righting"
        success_criterion = "metric > 0.5"

    blind = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=0, fitness=None, mean_return=0.9,
    )
    seen_events: list = []
    _select_stage_final_iter(
        [blind], _Stage(), stage_dir=None, emit=seen_events.append,
    )
    assert not any(
        e.get("type") == "fitness_contradiction" for e in seen_events)
    assert not (blind.iter_dir / "fitness_contradiction.json").is_file()


def test_select_stage_final_iter_writes_selection_json(tmp_path):
    """§Ship-56 (persistence increment): the per-candidate scan + final
    keep-decision must be recorded to `<stage_dir>/reports/selection.json`
    (previously computed then discarded — only selected_iter_index/
    selection_source landed in mission.json), byte-identical returned
    tuple."""
    from sculptor.sculpt import _select_stage_final_iter

    class _Stage:
        name = "torso_righting"
        success_criterion = "metric > 0.5"

    stage_dir = tmp_path / "stage_dir"
    stage_dir.mkdir()

    weak_pass = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=0, fitness=0.3, mean_return=0.9)
    strong_pass = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=1, fitness=0.8, mean_return=0.9)
    fails = _make_iter_outcome_with_rollout(
        tmp_path, iter_index=2, fitness=None, mean_return=0.1)

    result = _select_stage_final_iter(
        [weak_pass, strong_pass, fails], _Stage(), stage_dir=stage_dir,
    )
    selected, criterion_ok, source, err, missing, mismatch = result
    assert selected is strong_pass       # higher fitness among passers
    assert criterion_ok is True
    assert source == "criterion+fitness"
    assert err is None and missing is False and mismatch is None

    record = json.loads(
        (stage_dir / "reports" / "selection.json").read_text())
    assert record["stage"] == "torso_righting"
    assert record["selected_iter_index"] == 1
    assert record["selection_source"] == "criterion+fitness"
    assert record["criterion_ok"] is True
    assert record["criterion"] == "metric > 0.5"
    assert record["criterion_error"] is None
    assert record["start_state_mismatch"] is None
    assert record["gate"] == {
        "skipped": True, "checked": 0, "mismatched_count": 0,
    }
    rows_by_iter = {r["iter_index"]: r for r in record["candidates"]}
    assert set(rows_by_iter) == {0, 1, 2}
    assert rows_by_iter[0]["criterion_pass"] is True
    assert rows_by_iter[0]["fitness"] == 0.3
    assert rows_by_iter[0]["selected"] is False
    assert rows_by_iter[1]["criterion_pass"] is True
    assert rows_by_iter[1]["fitness"] == 0.8
    assert rows_by_iter[1]["selected"] is True
    assert rows_by_iter[2]["criterion_pass"] is False
    assert rows_by_iter[2]["fitness"] is None
    assert rows_by_iter[2]["selected"] is False


def test_diagnose_objective_progress_renders_components_and_revert():
    base = dict(
        behavior_goal="kick", reward_spec={}, metrics={}, behavior={},
        behavior_metric_names=[], contract_text="", keyframes=[],
    )
    out = _build_preliminary_user_content(
        **base,
        objective_progress={
            "current": 0.1, "best_so_far": 0.3, "last": 0.3, "delta": -0.2,
            "components": {"uprightness": 0.99, "kick_events": 0.0},
            "reverted_to_best": True,
        },
    )
    text = out[0]["text"]
    assert "component breakdown" in text
    assert "kick_events" in text and "uprightness" in text
    assert "REGRESSED fitness" in text and "DIFFERENT direction" in text


# ── §LAW 11: Goodhart-onset detector ─────────────────────────────────


def test_detect_goodhart_onset_fires_on_sustained_unnatural_climb():
    """The gaming signature: the metric keeps climbing while the policy SUSTAINS
    a loss of naturalness (>=2 of the recent window non-pass, AND a decline vs
    earlier). Naturalness uses the REAL discrete channel values — §kick-fix added
    the 'mild' tier so the set is now {1.0, 0.75, 0.5, 0.0} (steer factors)."""
    reason = S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 1.0, 0.5, 0.5])
    assert reason is not None and "goodhart" in reason.lower()


def test_detect_goodhart_onset_fires_on_sustained_mild():
    """§kick-fix: now that 'mild' down-weights (steer_factor 0.75 < 1.0), two
    SUSTAINED mild iters during a rising/declining window fire onset — the g1-kick-v6
    catch (the violent kicks read mild/severe at 2-3.7× the real motor limit)."""
    reason = S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 1.0, 0.75, 0.75])
    assert reason is not None and "goodhart" in reason.lower()
    # ...but a persistently-mild run (always 0.75) is NOT *becoming* less natural →
    # the decline guard still protects a legitimately-always-aggressive skill.
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [0.75, 0.75, 0.75, 0.75]) is None
    # and a single transient mild iter must not trip it (needs >=2 sustained)
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 1.0, 1.0, 0.75]) is None


def test_detect_goodhart_onset_silent_on_single_transient_severe():
    """THE false-positive guard: a SINGLE 'severe' iter (a legit hard kick
    transiently >3x nominal joint speed) must NOT lock the run — LAW 7 already
    down-weights it. Needs >=2 sustained unnatural iters."""
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 1.0, 1.0, 0.5]) is None
    # a dip that RECOVERED is not a sustained decline either
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 0.5, 1.0, 1.0]) is None


def test_detect_goodhart_onset_silent_on_persistently_aggressive():
    """A run that was ALWAYS aggressive (severe from the start) is not *becoming*
    less natural — the decline guard (c) lets a legit always-aggressive skill
    through rather than killing it mid-improvement."""
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [0.5, 0.5, 0.5, 0.5]) is None


def test_detect_goodhart_onset_silent_on_honest_progress():
    """Honest improvement (naturalness FLAT at 1.0 — the no-audit default) must
    NEVER trip the onset stop."""
    assert S.detect_goodhart_onset(
        [0.20, 0.30, 0.40, 0.50], [1.0, 1.0, 1.0, 1.0]) is None


def test_detect_goodhart_onset_silent_when_metric_not_rising():
    """A REGRESSING metric is a plateau/regression (handled by patience/revert),
    not gaming — onset requires the metric to be RISING."""
    assert S.detect_goodhart_onset(
        [0.50, 0.40, 0.30, 0.20], [1.0, 1.0, 0.5, 0.5]) is None


def test_detect_goodhart_onset_needs_enough_points():
    assert S.detect_goodhart_onset([0.2, 0.3], [1.0, 0.5]) is None
    assert S.detect_goodhart_onset([], []) is None


# ── §Convergence (RL_SCULPTOR_AUDIT §4.1): dense progress channel ─────────
# The tuck-jump deadlock: an all-or-nothing metric read 0.0 on every iter,
# strict-> selection made every 0.0 TIE count as "no new best", the revert
# fired every iter, and every corrective edit was generated but never
# trained. These tests pin the fix: ties build forward; a dense
# `progress_score` ranks sub-success iters; spec fitness still dominates.


def _install_progress_fake(monkeypatch, fitness_by_iter, seen_revert,
                           progress_by_iter=None):
    """Recording fake that also stamps the dense `progress` channel."""
    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr("sculptor.run_context.capture_run_context",
                        lambda *a, **k: {}, raising=True)
    monkeypatch.setattr("sculptor.run_context.write_run_context",
                        lambda *a, **k: Path("run_context.json"), raising=True)

    def fake_iter(**kw):
        seen_revert.append(kw.get("revert_base"))
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
        trained = rewards_dir / f"v{i}.py"
        edit = rewards_dir / f"v{i + 1}.py"
        edit.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
            encoding="utf-8",
        )
        S._write_current_reexport(rewards_dir, edit)
        return S.IterOutcome(
            iter_index=i, iter_dir=kw["runs_dir"] / f"iter_{i}",
            reward_path_before=rewards_dir / "current.py",
            reward_path_after=edit, primary_metric=0.0, behavior={},
            failure_modes=[], edit_count=1, fitness=fitness_by_iter[i],
            reward_path_trained=trained,
            progress=(progress_by_iter or {}).get(i),
        )

    monkeypatch.setattr(S, "_run_one_iter", fake_iter)


def test_fitness_tie_does_not_revert(tmp_path, monkeypatch):
    """A tie with best (the all-zero-metric case) must NOT trigger a revert —
    reverting on ties is the deadlock that pinned tuck-jump to v0."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, seen)
    res = S.sculpt_run(
        cfg_path, "goal", iterations=4, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=3,
    )
    # Every iter ties best (0.0, 0.0): the loop keeps building FORWARD so
    # each new edit actually gets trained next iter.
    assert all(s is None for s in seen), seen
    # Patience still counts ties, so the plateau stop is unaffected.
    assert res.early_stopped and "plateau" in res.early_stop_reason


def test_progress_breaks_ties_in_best_selection(tmp_path, monkeypatch):
    """Below the completion gate (spec 0.0 everywhere) the dense progress
    channel is the ranking signal: best = highest progress."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, seen,
        progress_by_iter={0: 0.0, 1: 0.1, 2: 0.3, 3: 0.2},
    )
    res = S.sculpt_run(
        cfg_path, "goal", iterations=4, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=3,
    )
    assert res.best_fitness_iter == 2
    assert res.best_progress == pytest.approx(0.3)
    assert res.progress_history == [0.0, 0.1, 0.3, 0.2]
    # current.py re-exports the best-progress iter's TRAINED reward.
    current = (cfg_path.parent / "rewards" / "current.py").read_text()
    assert "v2" in current


def test_progress_regression_reverts_to_best(tmp_path, monkeypatch):
    """A STRICT drop on the tuple (spec tie, progress down) is a real
    regression and must revert the edit base to the best-so-far reward."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0}, seen,
        progress_by_iter={0: 0.2, 1: 0.05, 2: 0.1},
    )
    S.sculpt_run(
        cfg_path, "goal", iterations=3, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=5,
    )
    # iter 0 sets best; iter 1 strictly regresses on progress → iter 2 is
    # handed iter 0's TRAINED reward (v0.py) as its edit base.
    assert seen[0] is None and seen[1] is None
    assert seen[2] is not None and seen[2].name == "v0.py"


def test_spec_fitness_dominates_progress(tmp_path, monkeypatch):
    """Lexicographic: any completion-gate success outranks ANY amount of
    dense progress — progress can never outbid the real task score."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.5}, seen,
        progress_by_iter={0: 0.9, 1: 0.0},
    )
    res = S.sculpt_run(
        cfg_path, "goal", iterations=2, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=5,
    )
    assert res.best_fitness_iter == 1
    assert res.best_fitness == pytest.approx(0.5)


# ── §Selection statistics: noise-band epsilon + fresh-seed re-eval ───────


def test_progress_noise_tick_is_tie_not_new_best(tmp_path, monkeypatch):
    """A progress uptick INSIDE the measured noise band (default epsilon
    1e-5, audit §6: seed noise spans 1e-7..4e-6) must NOT mint a new best —
    noise-floor bests reset patience and made plateau-stop unreachable."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0}, seen,
        progress_by_iter={0: 1e-6, 1: 3e-6, 2: 2e-6},
    )
    res = S.sculpt_run(
        cfg_path, "goal", iterations=3, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=2,
    )
    assert res.best_fitness_iter == 0          # 3e-6 did NOT dethrone 1e-6
    assert all(s is None for s in seen), seen  # noise dips never revert
    assert res.early_stopped                   # patience got to count ties


def test_progress_above_epsilon_mints_best(tmp_path, monkeypatch):
    """Real progress signal (≫ noise band) still mints a new best."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0}, seen,
        progress_by_iter={0: 1e-6, 1: 1e-3},
    )
    res = S.sculpt_run(
        cfg_path, "goal", iterations=2, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=5,
    )
    assert res.best_fitness_iter == 1


def test_progress_epsilon_zero_restores_strict_compare(tmp_path, monkeypatch):
    """progress_epsilon=0.0 is the exact pre-epsilon strict-`>` behavior."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0}, seen,
        progress_by_iter={0: 1e-6, 1: 3e-6},
    )
    res = S.sculpt_run(
        cfg_path, "goal", iterations=2, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=5,
        progress_epsilon=0.0,
    )
    assert res.best_fitness_iter == 1


def test_progress_big_dip_still_regresses(tmp_path, monkeypatch):
    """The epsilon must not swallow REAL regressions: a progress drop far
    outside the noise band still arms the best-first revert."""
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(
        monkeypatch, {0: 0.0, 1: 0.0, 2: 0.0}, seen,
        progress_by_iter={0: 1e-2, 1: 1e-6, 2: 5e-3},
    )
    S.sculpt_run(
        cfg_path, "goal", iterations=3, no_kg=True,
        fitness_fn=lambda d: 0.0, fitness_patience=5,
    )
    assert seen[2] is not None and seen[2].name == "v0.py"


def test_median_helper():
    assert S._median([]) is None
    assert S._median([0.4]) == pytest.approx(0.4)
    assert S._median([0.0, 0.4, 0.9]) == pytest.approx(0.4)
    assert S._median([0.0, 1.0]) == pytest.approx(0.5)


def test_rollout_or_resume_threads_seed_only_when_declared(tmp_path):
    """`seed` reaches adapters that declare it; legacy signatures
    (gym_sb3-shaped) are called without it — no TypeError."""
    calls: list = []

    class _SeedAdapter:
        def rollout(self, checkpoint_path, output_dir, n_episodes, *,
                    seed=None, **kw):
            calls.append(seed)

    class _LegacyAdapter:
        def rollout(self, checkpoint_path, output_dir, n_episodes):
            calls.append("legacy")

    d1 = tmp_path / "r1"
    d1.mkdir()
    S._rollout_or_resume(
        adapter=_SeedAdapter(), iter_index=0, rollout_dir=d1,
        checkpoint_path=tmp_path / "ck.pt", n_episodes=2, seed=123)
    d2 = tmp_path / "r2"
    d2.mkdir()
    S._rollout_or_resume(
        adapter=_LegacyAdapter(), iter_index=0, rollout_dir=d2,
        checkpoint_path=tmp_path / "ck.pt", n_episodes=2, seed=123)
    assert calls == [123, "legacy"]


def _fitness_fn_with_detail_dir(scores: list):
    """Stub fitness fn exposing the `detail_dir` accessor (multi-seed /
    fresh-eval contract); pops one queued spec score per call."""
    def _fn(iter_dir):
        return 0.0

    def _detail_dir(rollout_dir):
        return {"spec_score": scores.pop(0)} if scores else {}

    _fn.detail_dir = _detail_dir  # type: ignore[attr-defined]
    return _fn


def test_fresh_seed_reeval_of_kept_best(tmp_path, monkeypatch):
    """End-of-run: the kept best is re-rolled on held-out seeds and the
    unbiased median lands in `best_fitness_fresh` beside the selected
    (max-statistic) value. Selection itself is untouched."""
    cfg_path = _make_project(tmp_path)
    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr("sculptor.run_context.capture_run_context",
                        lambda *a, **k: {}, raising=True)
    monkeypatch.setattr("sculptor.run_context.write_run_context",
                        lambda *a, **k: Path("run_context.json"), raising=True)

    def fake_iter(**kw):
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
        trained = rewards_dir / f"v{i}.py"
        edit = rewards_dir / f"v{i + 1}.py"
        edit.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
            encoding="utf-8")
        S._write_current_reexport(rewards_dir, edit)
        iter_dir = kw["runs_dir"] / f"iter_{i}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        ckpt = iter_dir / "checkpoint.pt"
        ckpt.write_text("x", encoding="utf-8")
        # Pre-bake the fresh-eval rollout artifacts so _rollout_or_resume
        # SKIPS the adapter call (the adapter here is a bare object()).
        fresh = iter_dir / "rollout_fresh_0"
        fresh.mkdir()
        for name in ("rollout.mp4", "trajectory.npz", "behavior.json"):
            (fresh / name).write_text("x", encoding="utf-8")
        return S.IterOutcome(
            iter_index=i, iter_dir=iter_dir,
            reward_path_before=rewards_dir / "current.py",
            reward_path_after=edit, primary_metric=0.0, behavior={},
            failure_modes=[], edit_count=1, fitness={0: 0.2, 1: 0.7}[i],
            reward_path_trained=trained, checkpoint_path=ckpt,
        )

    monkeypatch.setattr(S, "_run_one_iter", fake_iter)
    fn = _fitness_fn_with_detail_dir([0.55])
    res = S.sculpt_run(
        cfg_path, "goal", iterations=2, no_kg=True,
        fitness_fn=fn, fitness_patience=5,
    )
    assert res.best_fitness_iter == 1
    assert res.best_fitness == pytest.approx(0.7)   # selection unchanged
    assert res.best_fitness_fresh == pytest.approx(0.55)
    assert res.fresh_fitness_per_seed == [pytest.approx(0.55)]


def test_fresh_eval_disabled_by_zero(tmp_path, monkeypatch):
    cfg_path = _make_project(tmp_path)
    seen: list = []
    _install_progress_fake(monkeypatch, {0: 0.2}, seen)
    fn = _fitness_fn_with_detail_dir([0.99])
    res = S.sculpt_run(
        cfg_path, "goal", iterations=1, no_kg=True,
        fitness_fn=fn, fitness_patience=5, fresh_eval_seeds=0,
    )
    assert res.best_fitness_fresh is None
    assert res.fresh_fitness_per_seed == []
