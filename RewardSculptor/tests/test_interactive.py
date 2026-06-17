"""§Ship 39 (H1): interactive human-in-the-loop — control-file parsing, the
iteration-boundary pause/resume/auto/stop/timeout branches, human_note
threading through sculpt_run, and the USER OBSERVATION diagnose block.
GPU/LLM-free: `_run_one_iter` and `_pause_for_feedback` are faked."""
from __future__ import annotations

from pathlib import Path

import sculptor.sculpt as S
from sculptor.diagnose import _build_preliminary_user_content


# ── control-file parsing ─────────────────────────────────────────────────
def test_read_control_file_missing_or_bad_is_empty(tmp_path) -> None:
    assert S._read_control_file(None) == {}
    assert S._read_control_file(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert S._read_control_file(bad) == {}
    good = tmp_path / "ok.json"
    good.write_text('{"mode": "manual", "resume_token": 3}', encoding="utf-8")
    assert S._read_control_file(good) == {"mode": "manual", "resume_token": 3}


# ── pause branches (monkeypatch the file reader for deterministic polls) ──
def test_pause_auto_mode_does_not_block(monkeypatch) -> None:
    monkeypatch.setattr(S, "_read_control_file", lambda _p: {"mode": "auto"})
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=5, poll_interval=0.01)
    assert note is None and stop is False


def test_pause_stop_returns_should_stop(monkeypatch) -> None:
    monkeypatch.setattr(S, "_read_control_file", lambda _p: {"stop": True})
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=5, poll_interval=0.01)
    assert note is None and stop is True


def test_pause_resumes_on_token_bump_with_feedback(monkeypatch) -> None:
    seq = iter([
        {"mode": "manual", "resume_token": 0},                       # initial
        {"mode": "manual", "resume_token": 0},                       # still waiting
        {"mode": "manual", "resume_token": 1, "feedback": "kick higher"},
    ])
    monkeypatch.setattr(S, "_read_control_file", lambda _p: next(seq))
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=10, poll_interval=0.01)
    assert note == "kick higher" and stop is False


def test_pause_bare_auto_flip_carries_no_stale_feedback(monkeypatch) -> None:
    """§Ship 39 review (MEDIUM): flipping the toggle to Auto (mode change with
    NO resume-token bump) must NOT re-inject a stale note left in the sidecar
    by a prior iteration's resume."""
    seq = iter([
        {"mode": "manual", "resume_token": 1, "feedback": "old note"},  # initial
        {"mode": "auto", "resume_token": 1, "feedback": "old note"},    # toggle only
    ])
    monkeypatch.setattr(S, "_read_control_file", lambda _p: next(seq))
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=10, poll_interval=0.01)
    assert note is None and stop is False


def test_pause_continue_and_go_auto_carries_feedback(monkeypatch) -> None:
    """"Continue + go Auto": mode=auto AND an explicit resume (token bumped)
    with feedback → the feedback IS carried to the next diagnose."""
    seq = iter([
        {"mode": "manual", "resume_token": 0},
        {"mode": "auto", "resume_token": 1, "feedback": "go full-auto with this"},
    ])
    monkeypatch.setattr(S, "_read_control_file", lambda _p: next(seq))
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=10, poll_interval=0.01)
    assert note == "go full-auto with this" and stop is False


def test_pause_times_out_and_auto_resumes(monkeypatch) -> None:
    monkeypatch.setattr(S, "_read_control_file",
                        lambda _p: {"mode": "manual", "resume_token": 0})
    note, stop = S._pause_for_feedback(Path("c"), 0, timeout=0.05, poll_interval=0.01)
    assert note is None and stop is False   # never pins the GPU forever


# ── integration: human_note threads to the NEXT iter's diagnose ───────────
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
        "[iteration]\nsteps_per_iter = 10\nseed = 42\n", encoding="utf-8")
    return proj / "config.toml"


def test_sculpt_run_threads_human_note_to_next_iter(tmp_path, monkeypatch) -> None:
    cfg_path = _make_project(tmp_path)
    seen_notes: list = []

    def fake_iter(**kw):
        seen_notes.append(kw.get("human_note"))
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
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
            failure_modes=[], edit_count=1,
        )

    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr("sculptor.run_context.capture_run_context",
                        lambda *a, **k: {}, raising=True)
    monkeypatch.setattr("sculptor.run_context.write_run_context",
                        lambda *a, **k: Path("rc.json"), raising=True)
    monkeypatch.setattr(S, "_run_one_iter", fake_iter)
    # fake the pause: return a note tagged with the iter it followed.
    monkeypatch.setattr(
        S, "_pause_for_feedback",
        lambda _cf, idx, **kw: (f"note-after-{idx}", False))

    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text('{"mode": "manual", "resume_token": 0}', encoding="utf-8")
    S.sculpt_run(cfg_path, "goal", iterations=3, no_kg=True, control_file=ctrl)

    # iter0: no note yet; iter1: feedback after iter0; iter2: feedback after
    # iter1. No pause after the last iter (nothing left to steer).
    assert seen_notes == [None, "note-after-0", "note-after-1"]


def test_sculpt_run_user_stop_ends_early(tmp_path, monkeypatch) -> None:
    cfg_path = _make_project(tmp_path)

    def fake_iter(**kw):
        i = kw["iter_index"]
        rewards_dir = kw["rewards_dir"]
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
            failure_modes=[], edit_count=1)

    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr("sculptor.run_context.capture_run_context",
                        lambda *a, **k: {}, raising=True)
    monkeypatch.setattr("sculptor.run_context.write_run_context",
                        lambda *a, **k: Path("rc.json"), raising=True)
    monkeypatch.setattr(S, "_run_one_iter", fake_iter)
    monkeypatch.setattr(S, "_pause_for_feedback", lambda _cf, idx, **kw: (None, True))

    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text('{"mode": "manual", "resume_token": 0}', encoding="utf-8")
    res = S.sculpt_run(cfg_path, "goal", iterations=5, no_kg=True, control_file=ctrl)
    assert res.early_stopped and "user" in res.early_stop_reason
    assert res.iterations_run == 1          # stopped right after iter 0


# ── diagnose USER OBSERVATION block ──────────────────────────────────────
def test_diagnose_renders_user_observation() -> None:
    base = dict(behavior_goal="kick", reward_spec={}, metrics={}, behavior={},
                behavior_metric_names=[], contract_text="", keyframes=[])
    with_note = _build_preliminary_user_content(
        **base, human_note="it's standing still and flailing its arms")
    text = with_note[0]["text"]
    assert "USER OBSERVATION" in text
    assert "flailing its arms" in text
    # absent / blank note → no block.
    assert "USER OBSERVATION" not in _build_preliminary_user_content(**base)[0]["text"]
    assert "USER OBSERVATION" not in _build_preliminary_user_content(
        **base, human_note="   ")[0]["text"]
