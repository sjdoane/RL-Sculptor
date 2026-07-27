"""tests/test_mode_metrics.py — the objective-metric gauntlet, per mode and
per transition.

The claim under test is narrow and checkable: an episode-level score can be
healthy while one mode of the behavior is completely degenerate, and scoring
per mode makes that visible with an address. The rest of the file pins the
things that would quietly break that claim — a clock that guesses, a mode the
rollout never reached being reported as a zero, a guard that abstains being
reported as a verdict, and a mode goal that inherits the episode goal's
behavior family.
"""
from __future__ import annotations

import numpy as np
import pytest

from sculptor.eval.mode_metrics import (
    ModeMetricError,
    calibrate_mode_metrics,
    check_transitions,
    generate_mode_metrics,
    mode_gauntlet_report,
    mode_goal_text,
    mode_prompt_context,
    mode_reference_clip,
    mode_slices,
    render_mode_report,
    resolve_step_dt,
    score_modes,
    slice_behavior,
    validate_mode_metrics,
)
from sculptor.modes import Guard, Mode, ModeGraph, Transition

FPS = 60.0
DT = 0.02
T, E, J = 100, 4, 12
_NAMES_12 = [
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_elbow", "right_elbow", "torso", "neck",
]


def _graph(fps: float = FPS) -> ModeGraph:
    """Two modes, 60 frames each at `fps` — 1.0 s apiece at 60 fps."""
    return ModeGraph(
        modes=(Mode("approach", (0, 60)), Mode("strike", (60, 120))),
        transitions=(Transition("approach", "strike",
                                Guard("phase", at_phase=1.0)),),
        fps=fps)


def _behavior(steps: int = T) -> dict:
    return {"max_episode_steps": steps, "rollout_num_envs": E, "step_dt": DT}


def _half_active_rollout() -> dict[str, np.ndarray]:
    """A rollout whose FIRST half moves and whose SECOND half is dead still.

    This is the shape the whole module exists for: an episode-level average
    reports a middling score and says nothing about where the motion stopped.
    """
    t = np.arange(T)
    jp = np.zeros((T, E, J))
    jp[:50] = np.sin(2 * np.pi * t[:50] / 20.0)[:, None, None] * 0.5
    jp[50:] = jp[49]                       # frozen at the last active pose
    jv = np.gradient(jp, axis=0) / DT
    grav = np.zeros((T, E, 3))
    grav[..., 2] = -1.0                    # upright throughout
    root = np.zeros((T, E, 3))
    root[..., 2] = 0.7
    return {"joint_pos": jp, "joint_vel": jv,
            "projected_gravity_b": grav, "root_link_pos_w": root}


def _motion_metric(arrays, behavior, meta):
    """Mean joint speed, saturating — high on a moving slice, 0 on a still one."""
    jv = arrays.get("joint_vel")
    if jv is None:
        return {"spec_score": 0.0}
    motion = float(np.mean(np.abs(jv)))
    return {"spec_score": float(np.clip(motion / 5.0, 0.0, 1.0)),
            "motion": motion}


def _clip(n: int = 120, *, fps: float = FPS) -> dict:
    t = np.arange(n, dtype=np.float64) / fps
    return {
        "fps": fps,
        "joint_names": [f"joint_{i}" for i in range(4)],
        "root_pos_z": 0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        "root_pos_xy": np.stack([0.5 * t, np.zeros(n)], axis=1),
        "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "joint_pos": (0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
                      + 0.01 * np.arange(4)[None, :]),
        "meta": {"clip_id": "synthetic"},
    }


# ── the headline claim ──────────────────────────────────────────────────
def test_a_degenerate_mode_scores_zero_where_an_episode_score_hides_it():
    result = score_modes(_motion_metric, _half_active_rollout(), _behavior(),
                         {}, _graph())
    assert result["episode"] > 0.2, "the episode score should look survivable"
    assert result["modes"]["approach"]["score"] > 0.2
    assert result["modes"]["strike"]["score"] == pytest.approx(0.0)
    assert result["worst_mode"] == "strike"
    # The gap is exactly what an episode-level score averages away.
    assert result["worst_mode_gap"] == pytest.approx(result["episode"])


def test_the_worst_mode_gap_is_reported_and_not_turned_into_a_verdict():
    result = score_modes(_motion_metric, _half_active_rollout(), _behavior(),
                         {}, _graph())
    assert isinstance(result["worst_mode_gap"], float)
    # No pass/fail derived from it anywhere in the record — measuring a gap we
    # have no calibration for must not become an invented threshold.
    assert "ok" not in result and "passed" not in result


# ── the clock ───────────────────────────────────────────────────────────
def test_step_dt_is_never_silently_defaulted():
    with pytest.raises(ModeMetricError, match="cannot resolve"):
        resolve_step_dt({"max_episode_steps": 100})
    with pytest.raises(ModeMetricError, match="positive finite"):
        resolve_step_dt({"step_dt": 0.0})
    with pytest.raises(ModeMetricError, match="not a number"):
        resolve_step_dt({"step_dt": "fast"})


def test_an_explicit_step_dt_outranks_the_behavior_dict():
    assert resolve_step_dt({"step_dt": 0.02}, step_dt=0.005) == 0.005


def test_mode_windows_map_to_rollout_frames_by_wall_time_not_frame_count():
    """A 120 fps reference and a 50 Hz rollout describe the same 2 seconds.

    Matching by frame COUNT would put the mode boundary at frame 120 of a
    100-frame rollout; matching by wall time puts it at frame 50, which is what
    1.0 s actually is at 50 Hz.
    """
    graph = ModeGraph(
        modes=(Mode("a", (0, 120)), Mode("b", (120, 240))),
        transitions=(Transition("a", "b", Guard("phase", at_phase=1.0)),),
        fps=120.0)
    a, b = mode_slices(graph, rollout_frames=100, step_dt=DT)
    assert (a.lo, a.hi) == (0, 50)
    assert (b.lo, b.hi) == (50, 100)
    assert a.coverage == 1.0 and b.coverage == 1.0


def test_slice_boundaries_round_rather_than_truncate():
    """`mode_phase_windows` rounds seconds to 4 decimals; flooring the frame
    conversion drops a boundary frame about half the time depending on fps."""
    graph = ModeGraph(
        modes=(Mode("a", (0, 50)), Mode("b", (50, 100))),
        transitions=(Transition("a", "b", Guard("phase", at_phase=1.0)),),
        fps=30.0)                                    # 50/30 = 1.6667 s
    a, b = mode_slices(graph, rollout_frames=100, step_dt=DT)
    assert a.hi == 83 and b.lo == 83                 # no hole, no overlap


def test_mode_slices_tile_the_rollout_without_gaps_or_overlap():
    slices = mode_slices(_graph(), rollout_frames=T, step_dt=DT)
    assert slices[0].lo == 0
    assert slices[-1].hi == T
    for earlier, later in zip(slices, slices[1:]):
        assert earlier.hi == later.lo


# ── unentered vs degenerate ─────────────────────────────────────────────
def test_a_mode_the_rollout_never_reached_is_unscored_not_scored_zero():
    """'never performed' and 'performed degenerately' are opposite diagnoses.

    Reporting the first as 0.0 makes them indistinguishable, and the fixes
    point in different directions: one says the policy stalls earlier, the
    other says this mode's reward is wrong.
    """
    arrays = {k: v[:40] for k, v in _half_active_rollout().items()}
    result = score_modes(_motion_metric, arrays, _behavior(40), {}, _graph())
    strike = result["modes"]["strike"]
    assert strike["score"] is None and strike["scored"] is False
    assert strike["slice"]["entered"] is False
    assert "never entered" in strike["error"]
    assert result["unentered_modes"] == ["strike"]
    assert result["worst_mode"] == "approach"    # not silently outranked by a 0.0


def test_partial_mode_coverage_is_measured_and_left_ungated():
    arrays = {k: v[:75] for k, v in _half_active_rollout().items()}
    result = score_modes(_motion_metric, arrays, _behavior(75), {}, _graph())
    strike = result["modes"]["strike"]["slice"]
    assert strike["entered"] is True
    assert strike["coverage"] == pytest.approx(0.5)
    assert strike["requested_hi"] == 100 and strike["hi"] == 75
    assert result["modes"]["strike"]["scored"] is True   # scored, not rejected


# ── what a sliced metric sees ───────────────────────────────────────────
def test_a_slice_rebases_the_episode_length_a_metric_normalizes_by():
    """A metric reading 'the last 20% of the episode' must get 20% of the
    SLICE. Leaving max_episode_steps at the episode value points it at the
    wrong frames entirely."""
    window = mode_slices(_graph(), rollout_frames=T, step_dt=DT)[1]
    behavior = slice_behavior(
        {**_behavior(), "mean_episode_length": 100.0, "mean_return": 12.5},
        window)
    assert behavior["max_episode_steps"] == 50
    assert behavior["mean_episode_length"] == 50
    assert behavior["step_dt"] == DT and behavior["rollout_num_envs"] == E
    # An episode-level aggregate that cannot be sliced is left alone rather
    # than scaled into a fiction.
    assert behavior["mean_return"] == 12.5


def test_a_mode_metric_that_crashes_records_an_error_without_blanking_the_others():
    def _explodes(arrays, behavior, meta):
        raise RuntimeError("boom")

    result = score_modes(
        _motion_metric, _half_active_rollout(), _behavior(), {}, _graph(),
        metrics_by_mode={"strike": _explodes})
    assert result["modes"]["approach"]["scored"] is True
    assert result["modes"]["strike"]["scored"] is False
    assert "RuntimeError: boom" in result["modes"]["strike"]["error"]


def test_a_non_finite_score_is_recorded_as_a_reason_not_carried_as_a_nan():
    """A nan is not valid JSON and says nothing; the record must carry the
    finding instead of a hole."""
    def _nan(arrays, behavior, meta):
        return {"spec_score": float("nan")}

    result = score_modes(
        _motion_metric, _half_active_rollout(), _behavior(), {}, _graph(),
        metrics_by_mode={"strike": _nan})
    strike = result["modes"]["strike"]
    assert strike["score"] is None and strike["scored"] is False
    assert strike["error"] == "spec_score is not finite"


def test_each_mode_can_be_scored_by_its_own_metric():
    def _always_one(arrays, behavior, meta):
        return {"spec_score": 1.0}

    result = score_modes(
        _motion_metric, _half_active_rollout(), _behavior(), {}, _graph(),
        metrics_by_mode={"strike": _always_one})
    assert result["modes"]["strike"]["score"] == 1.0
    assert result["modes"]["approach"]["score"] < 1.0


def test_ragged_rollout_arrays_are_rejected_rather_than_silently_shortened():
    arrays = _half_active_rollout()
    arrays["joint_vel"] = arrays["joint_vel"][:80]
    with pytest.raises(ModeMetricError, match="disagree on their time axis"):
        score_modes(_motion_metric, arrays, _behavior(), {}, _graph())


def test_a_mode_finer_than_the_control_rate_says_so_rather_than_never_entered():
    """A 1-frame mode of a 120 fps reference is 0.0083 s — at a 50 Hz control
    rate that is a decomposition the controller cannot resolve, which is a
    different problem from a policy that stalled before reaching it."""
    graph = ModeGraph(
        modes=(Mode("wind_up", (0, 1)), Mode("strike", (1, 240))),
        transitions=(Transition("wind_up", "strike",
                                Guard("phase", at_phase=1.0)),),
        fps=120.0)
    result = score_modes(_motion_metric, _half_active_rollout(), _behavior(),
                         {}, graph)
    assert result["modes"]["strike"]["scored"] is True     # the rest still scores
    wind_up = result["modes"]["wind_up"]
    assert wind_up["slice"]["shorter_than_one_step"] is True
    assert "shorter than one control step" in wind_up["error"]
    rendered = render_mode_report(mode_gauntlet_report(graph, scores=result))
    assert "wind_up: SHORTER THAN ONE CONTROL STEP" in rendered


def test_transition_checking_refuses_an_invalid_mode_graph():
    graph = ModeGraph(modes=(Mode("a", (0, 60)), Mode("orphan", (60, 120))),
                      transitions=(), fps=FPS)
    with pytest.raises(ModeMetricError, match="unreachable"):
        check_transitions(graph, rollout_frames=T, step_dt=DT)


def test_scoring_refuses_an_invalid_mode_graph():
    graph = ModeGraph(modes=(Mode("a", (0, 60)), Mode("orphan", (60, 120))),
                      transitions=(), fps=FPS)      # 'orphan' unreachable
    with pytest.raises(ModeMetricError, match="unreachable"):
        score_modes(_motion_metric, _half_active_rollout(), _behavior(), {},
                    graph)


# ── per-transition guard firing ─────────────────────────────────────────
def test_a_phase_guard_that_is_never_reached_names_the_mode_the_policy_never_left():
    [transition] = check_transitions(_graph(), rollout_frames=40, step_dt=DT)
    assert transition["fired"] is False
    assert transition["fire_time_s"] == pytest.approx(1.0)
    assert transition["rollout_duration_s"] == pytest.approx(0.8)
    assert "never left 'approach'" in transition["reason"]
    assert transition["to_mode_entered"] is False


def test_a_phase_guard_fires_once_the_rollout_reaches_its_time():
    [transition] = check_transitions(_graph(), rollout_frames=T, step_dt=DT)
    assert transition["fired"] is True
    assert transition["to_mode_entered"] is True
    assert "reached" in transition["reason"]


def test_a_guard_firing_early_is_distinguished_from_its_target_being_entered():
    """`at_phase < 1.0` hands over mid-window, so 'the guard fired' and 'the
    next mode's window began' are genuinely different instants."""
    graph = ModeGraph(
        modes=(Mode("approach", (0, 60)), Mode("strike", (60, 120))),
        transitions=(Transition("approach", "strike",
                                Guard("phase", at_phase=0.5)),),
        fps=FPS)
    [transition] = check_transitions(graph, rollout_frames=30, step_dt=DT)
    assert transition["fire_time_s"] == pytest.approx(0.5)
    assert transition["fired"] is True
    assert transition["to_mode_entered"] is False


def test_a_predicate_guard_is_evaluated_by_the_shared_isolated_evaluator():
    graph = ModeGraph(
        modes=(Mode("approach", (0, 60)), Mode("strike", (60, 120))),
        transitions=(Transition("approach", "strike",
                                Guard("predicate", expression="metric > 0.5")),),
        fps=FPS)
    [fired] = check_transitions(graph, rollout_frames=T, step_dt=DT,
                                namespace={"metric": 0.9})
    assert fired["fired"] is True
    [not_fired] = check_transitions(graph, rollout_frames=T, step_dt=DT,
                                    namespace={"metric": 0.1})
    assert not_fired["fired"] is False
    assert "never held" in not_fired["reason"]


def test_a_predicate_guard_without_a_namespace_abstains_rather_than_deciding():
    graph = ModeGraph(
        modes=(Mode("a", (0, 60)), Mode("b", (60, 120))),
        transitions=(Transition("a", "b",
                                Guard("predicate", expression="metric > 0.5")),),
        fps=FPS)
    [transition] = check_transitions(graph, rollout_frames=T, step_dt=DT)
    assert transition["fired"] is None            # a THIRD state, not False
    assert "abstained, not failed" in transition["reason"]


def test_a_predicate_guard_that_cannot_be_evaluated_abstains_rather_than_reporting_false():
    """An evaluator failure is absence of evidence. Reporting it as `False`
    would manufacture a 'the policy never transitioned' diagnosis out of a
    broken expression."""
    graph = ModeGraph(
        modes=(Mode("a", (0, 60)), Mode("b", (60, 120))),
        transitions=(Transition("a", "b",
                                Guard("predicate", expression="nope(((")),),
        fps=FPS)
    [transition] = check_transitions(graph, rollout_frames=T, step_dt=DT,
                                     namespace={"metric": 0.9})
    assert transition["fired"] is None
    assert "could not be evaluated" in transition["reason"]


def test_guard_firing_is_reported_but_never_fails_a_metric():
    """Whether a guard fired is a fact about the ROLLOUT; this package's gates
    judge METRICS. A stalled policy must not be recorded as a bad metric."""
    transitions = check_transitions(_graph(), rollout_frames=10, step_dt=DT)
    assert all(t["fired"] is False for t in transitions)
    for key in ("ok", "passed", "gate"):
        assert all(key not in t for t in transitions)
    report = mode_gauntlet_report(
        _graph(), transitions=transitions,
        validation={"ok": True, "modes": {}, "failed_modes": []})
    assert report["unfired_guards"] == ["approach->strike"]
    assert report["failed_modes"] == []           # not contaminated by the guard


# ── per-mode goals ──────────────────────────────────────────────────────
def test_a_mode_goal_does_not_inherit_the_episode_goals_behavior_family():
    """`resolve_behavior_family` word-matches the WHOLE goal string, so
    splicing the episode goal into a mode goal resolves every mode to the
    episode's family — anchoring an approach metric against a kick positive."""
    from sculptor.eval.metric_validate import resolve_behavior_family

    episode_goal = "run up and kick the ball"
    approach = mode_goal_text(Mode("approach", (0, 60)))
    assert resolve_behavior_family(episode_goal) == "kick"
    assert resolve_behavior_family(approach) != "kick"
    assert "kick" not in approach


def test_a_supplied_mode_goal_wins_over_the_mode_name():
    mode = Mode("mode_2", (0, 60))
    assert mode_goal_text(mode) == "mode 2"
    assert mode_goal_text(mode, mode_goals={"mode_2": "plant the stance foot"}) \
        == "plant the stance foot"


def test_the_episode_goal_reaches_the_author_as_context_not_as_the_goal():
    graph = _graph()
    context = mode_prompt_context(graph, graph.modes[0],
                                  episode_goal="run up and kick the ball")
    assert "episode_goal_for_context_only" in context
    assert "run up and kick the ball" in context
    assert '"mode": "approach"' in context
    assert '"followed_by": "strike"' in context


# ── per-mode references ─────────────────────────────────────────────────
def test_a_mode_reference_is_cropped_to_that_modes_own_frames():
    clip = _clip(120)
    cropped = mode_reference_clip(clip, Mode("strike", (60, 120)))
    assert cropped["root_pos_z"].shape[0] == 60
    assert cropped["root_pos_z"][0] == pytest.approx(clip["root_pos_z"][60])
    assert cropped["joint_pos"].shape == (60, 4)
    assert cropped["fps"] == clip["fps"]


def test_a_cropped_reference_records_that_it_was_cropped():
    clip = _clip(120)
    cropped = mode_reference_clip(clip, Mode("strike", (60, 120)),
                                  clip_id="composite-1")
    crop = cropped["meta"]["mode_crop"]
    assert crop["mode"] == "strike"
    assert crop["frame_range"] == [60, 120]
    assert crop["source_clip_id"] == "composite-1"
    assert crop["source_frames"] == 120
    # The caller's clip is untouched — meta is copied, not mutated.
    assert "mode_crop" not in clip["meta"]


def test_cropping_rejects_a_frame_range_the_clip_cannot_support():
    clip = _clip(80)
    with pytest.raises(ModeMetricError, match="but the clip has 80 frames"):
        mode_reference_clip(clip, Mode("strike", (60, 120)))
    with pytest.raises(ModeMetricError, match="at least 2"):
        mode_reference_clip(clip, Mode("blink", (10, 11)))


# ── the gauntlet, per mode ──────────────────────────────────────────────
GOOD_MOTION_METRIC = '''import numpy as np

ABSTRACT_OBJECTIVE = {"phases": ["move_forward"]}


def compute_spec(arrays, behavior, meta):
    root = arrays.get("root_link_pos_w")
    grav = arrays.get("projected_gravity_b")
    if root is None or grav is None:
        return {"spec_score": 0.0}
    disp = float(np.linalg.norm(root[-1, :, :2].mean(0) - root[0, :, :2].mean(0)))
    up = float(np.mean(grav[..., 2] < -0.85))
    speed = 1.0 - float(np.exp(-disp / 2.0))
    return {"spec_score": float(np.clip(speed * up, 0.0, 1.0)),
            "disp": disp, "up": up}
'''


MOTION_METRIC_SRC = '''import numpy as np


def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel")
    if jv is None:
        return {"spec_score": 0.0}
    motion = float(np.mean(np.abs(jv)))
    return {"spec_score": float(np.clip(motion / 5.0, 0.0, 1.0)),
            "motion": motion}
'''


def _write(tmp_path, name, src):
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    return path


def test_a_real_sandboxed_generated_metric_scores_a_mode_slice(tmp_path):
    """The per-mode path has to work across the ACTUAL untrusted-code boundary.

    A generated metric is not a plain callable — it is a proxy onto a seccomp/
    rlimit worker process with a bounded array-IPC wire format. Slices go over
    that wire like any other arrays; this pins that they do.
    """
    from sculptor.eval.generated_metric import load_generated_metric

    fn = load_generated_metric(_write(tmp_path, "m.py", MOTION_METRIC_SRC))
    result = score_modes(fn, _half_active_rollout(), _behavior(), {}, _graph())
    assert result["modes"]["approach"]["score"] > 0.2
    assert result["modes"]["strike"]["score"] == pytest.approx(0.0)
    assert result["episode"] > result["modes"]["strike"]["score"]


def test_per_mode_validation_runs_each_modes_metric_against_its_own_goal(tmp_path):
    path = _write(tmp_path, "m.py", GOOD_MOTION_METRIC)
    sources = {"approach": (GOOD_MOTION_METRIC, path),
               "strike": (GOOD_MOTION_METRIC, path)}
    result = validate_mode_metrics(
        sources, _graph(),
        mode_goals={"approach": "walk forward", "strike": "kick the ball"})
    assert result["modes"]["approach"]["goal"] == "walk forward"
    assert result["modes"]["strike"]["goal"] == "kick the ball"
    assert result["modes"]["approach"]["family"] == "locomotion"
    assert result["modes"]["strike"]["family"] == "kick"


def test_per_mode_validation_adds_no_synthetic_positive_to_the_fixed_battery(tmp_path):
    """An unconditionally-added fixed-battery positive has already, once,
    masked the still/flail negatives and rescued a gameable metric. Running a
    gate 'per mode' must therefore change WHICH GOAL is passed and nothing
    else — the battery must be byte-for-byte what the goal alone produces.
    """
    from sculptor.eval.metric_validate import validate_generated_metric

    path = _write(tmp_path, "m.py", GOOD_MOTION_METRIC)
    direct = validate_generated_metric(
        GOOD_MOTION_METRIC, path, behavior_goal="walk forward")
    through_modes = validate_mode_metrics(
        {"approach": (GOOD_MOTION_METRIC, path)},
        ModeGraph(modes=(Mode("approach", (0, 60)),), transitions=(), fps=FPS),
        mode_goals={"approach": "walk forward"})
    scored = through_modes["modes"]["approach"]["validation"]
    assert set(scored["archetype_scores"]) == set(direct["archetype_scores"])
    assert scored["archetype_scores"] == direct["archetype_scores"]
    assert scored["gates"] == direct["gates"]


def test_a_mode_with_no_metric_is_a_failure_not_an_omission(tmp_path):
    path = _write(tmp_path, "m.py", GOOD_MOTION_METRIC)
    result = validate_mode_metrics(
        {"approach": (GOOD_MOTION_METRIC, path)}, _graph(),
        mode_goals={"approach": "walk forward"})
    assert result["ok"] is False
    assert result["failed_modes"] == ["strike"]
    assert any("no metric supplied" in r
               for r in result["modes"]["strike"]["reasons"])


def test_per_mode_validation_anchors_on_the_modes_own_cropped_reference(tmp_path):
    path = _write(tmp_path, "m.py", GOOD_MOTION_METRIC)
    result = validate_mode_metrics(
        {"approach": (GOOD_MOTION_METRIC, path),
         "strike": (GOOD_MOTION_METRIC, path)},
        _graph(), mode_goals={"approach": "walk forward",
                              "strike": "walk forward"},
        reference_clip=_clip(120), reference_clip_id="composite-1")
    for name in ("approach", "strike"):
        references = result["modes"][name]["validation"]["references"]
        assert [r["clip_id"] for r in references] == [f"composite-1#{name}"]


# ── per-mode gaming archetypes + competence ladder ──────────────────────
def test_gaming_archetypes_and_the_ladder_are_authored_per_mode(monkeypatch):
    """A metric hard to game across a whole episode can be trivially gameable
    inside one mode, so the blind gaming/ladder authors must be pointed at the
    sub-behavior, not the behavior."""
    from sculptor.eval import metric_calibration

    seen: list[dict] = []

    def _fake(path, behavior_goal, robot_hint=None, **kw):
        seen.append({"goal": behavior_goal, "adversarial": kw.get("adversarial")})
        return {"ok": True, "reason": None, "spearman": 0.9,
                "adversarial": {"gameable": False}}

    monkeypatch.setattr(metric_calibration, "calibrate_task_derived", _fake)
    result = calibrate_mode_metrics(
        {"approach": "a.py", "strike": "b.py"}, _graph(),
        mode_goals={"approach": "walk forward", "strike": "kick the ball"})
    assert [s["goal"] for s in seen] == ["walk forward", "kick the ball"]
    assert result["ok"] is True
    assert result["granted_modes"] == ["approach", "strike"]


def test_the_per_mode_gate_starts_at_the_adversarial_setting(monkeypatch):
    from sculptor.eval import metric_calibration

    seen: list[bool] = []

    def _fake(path, behavior_goal, robot_hint=None, **kw):
        seen.append(kw.get("adversarial"))
        return {"ok": True}

    monkeypatch.setattr(metric_calibration, "calibrate_task_derived", _fake)
    calibrate_mode_metrics({"approach": "a.py", "strike": "b.py"}, _graph())
    assert seen == [True, True]


def test_a_mode_gameable_within_its_window_is_named(monkeypatch):
    from sculptor.eval import metric_calibration

    def _fake(path, behavior_goal, robot_hint=None, **kw):
        gameable = behavior_goal == "strike"
        return {"ok": not gameable, "reason": "gamed" if gameable else None,
                "adversarial": {"gameable": gameable}}

    monkeypatch.setattr(metric_calibration, "calibrate_task_derived", _fake)
    result = calibrate_mode_metrics(
        {"approach": "a.py", "strike": "b.py"}, _graph())
    assert result["gameable_modes"] == ["strike"]
    assert result["ok"] is False


def test_a_mode_with_no_metric_path_is_not_silently_granted(monkeypatch):
    from sculptor.eval import metric_calibration

    monkeypatch.setattr(metric_calibration, "calibrate_task_derived",
                        lambda *a, **k: {"ok": True})
    result = calibrate_mode_metrics({"approach": "a.py"}, _graph())
    assert result["ok"] is False
    assert "nothing to calibrate" in result["modes"]["strike"]["reason"]


# ── per-mode generation ─────────────────────────────────────────────────
def test_generated_mode_metrics_land_in_one_directory_per_mode(tmp_path, monkeypatch):
    from sculptor.eval import metric_gen

    calls: list[dict] = []

    def _fake(goal, out_dir, **kw):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        calls.append({"goal": goal, "out_dir": str(out_dir),
                      "appendix": kw.get("prompt_appendix"),
                      "review_appendix": kw.get("review_appendix"),
                      "references": kw.get("references")})
        return {"accepted": True, "metric_path": str(Path(out_dir) / "metric.py")}

    monkeypatch.setattr(metric_gen, "generate_objective_metric", _fake)
    result = generate_mode_metrics(
        _graph(), tmp_path, episode_goal="run up and kick the ball",
        mode_goals={"approach": "walk forward", "strike": "kick the ball"},
        reference_clip=_clip(120), reference_clip_id="composite-1")
    assert result["n_accepted"] == 2
    assert [c["goal"] for c in calls] == ["walk forward", "kick the ball"]
    assert calls[0]["out_dir"].endswith("mode_approach")
    assert calls[1]["out_dir"].endswith("mode_strike")
    # Each mode is anchored on its OWN cropped segment of the composite.
    assert calls[1]["references"][0][0] == "composite-1#strike"
    assert calls[1]["references"][0][1]["root_pos_z"].shape[0] == 60


def test_the_mode_scope_contract_reaches_the_author_and_the_reviewer(tmp_path,
                                                                     monkeypatch):
    from sculptor.eval import metric_gen

    captured: dict = {}

    def _fake(goal, out_dir, **kw):
        captured.update(kw)
        return {"accepted": True}

    monkeypatch.setattr(metric_gen, "generate_objective_metric", _fake)
    generate_mode_metrics(_graph(), tmp_path, episode_goal="kick the ball")
    assert "MODE SCOPE" in captured["prompt_appendix"]
    assert "already sliced to this mode's" in captured["prompt_appendix"]
    assert "THIS MODE (data" in captured["prompt_appendix"]
    assert "MODE-SCOPE REVIEW" in captured["review_appendix"]


def test_one_modes_generation_failure_does_not_abort_the_others(tmp_path,
                                                               monkeypatch):
    from sculptor.eval import metric_gen

    def _fake(goal, out_dir, **kw):
        if goal == "approach":
            raise RuntimeError("api down")
        return {"accepted": True}

    monkeypatch.setattr(metric_gen, "generate_objective_metric", _fake)
    result = generate_mode_metrics(_graph(), tmp_path)
    assert "RuntimeError: api down" in result["modes"]["approach"]["error"]
    assert result["modes"]["strike"]["accepted"] is True
    assert result["n_accepted"] == 1


def test_a_goal_derived_from_a_mode_name_is_recorded_as_derived(tmp_path,
                                                                monkeypatch):
    """A mode named `mode_2` carries no semantics; a report must not imply
    more grounding than there was."""
    from sculptor.eval import metric_gen

    monkeypatch.setattr(metric_gen, "generate_objective_metric",
                        lambda goal, out_dir, **kw: {"accepted": True})
    result = generate_mode_metrics(
        _graph(), tmp_path, mode_goals={"approach": "walk forward"})
    assert result["modes"]["approach"]["goal_source"] == "supplied"
    assert result["modes"]["strike"]["goal_source"] == "derived_from_mode_name"


# ── the prompt appendix is additive ─────────────────────────────────────
def test_the_prompt_appendix_only_adds_to_the_shared_rubric(tmp_path):
    """The mode contract layers on `gen_objective_metric.md` rather than
    forking it — a forked rubric drifts, and the rules it would carry are the
    hard-won rejection causes."""
    from sculptor.prompts import load_prompt

    seen: list[str] = []

    class _Block:
        def __init__(self, text):
            self.type, self.text = "text", text

    class _Messages:
        def create(self, **kw):
            seen.append(kw["system"])
            return type("R", (), {"content": [_Block("```python\n" + GOOD_MOTION_METRIC + "\n```")],
                                  "stop_reason": "end_turn", "usage": None})()

        def parse(self, **kw):
            from sculptor.eval.metric_gen import MetricReview
            return type("P", (), {"parsed_output": MetricReview(approved=True),
                                  "content": [], "usage": None})()

    class _Client:
        messages = _Messages()

    from sculptor.eval.metric_gen import generate_objective_metric

    generate_objective_metric("walk forward", tmp_path / "base",
                              client=_Client(), max_attempts=1)
    generate_objective_metric("walk forward", tmp_path / "moded",
                              client=_Client(), max_attempts=1,
                              prompt_appendix="# EXTRA CONTRACT")
    base_prompt = load_prompt("gen_objective_metric")
    assert seen[0] == base_prompt                      # unchanged by default
    assert seen[1].startswith(base_prompt)             # additive, never a fork
    assert seen[1].endswith("# EXTRA CONTRACT")


# ── the report ──────────────────────────────────────────────────────────
def test_the_report_names_which_mode_failed_and_why():
    scores = score_modes(_motion_metric, _half_active_rollout(), _behavior(),
                         {}, _graph())
    report = mode_gauntlet_report(
        _graph(), episode_goal="run up and kick the ball", scores=scores,
        transitions=check_transitions(_graph(), rollout_frames=T, step_dt=DT),
        validation={"ok": False, "failed_modes": ["strike"], "modes": {
            "approach": {"ok": True, "reasons": []},
            "strike": {"ok": False,
                       "reasons": ["[nondegeneracy] 'still' scores >= the "
                                   "best positive"]}}},
        calibration={"ok": False, "gameable_modes": ["strike"], "modes": {
            "approach": {"ok": True}, "strike": {"ok": False, "gameable": True,
                                                 "reason": "gamed in-window"}}})
    assert report["failed_modes"] == ["strike"]
    assert report["gameable_modes"] == ["strike"]
    assert report["worst_mode"] == "strike"

    rendered = render_mode_report(report)
    assert "strike" in rendered
    assert "gates FAIL" in rendered
    assert "GAMEABLE within this mode" in rendered
    assert "nondegeneracy" in rendered
    # The episode score comes LAST — leading with it is what hid the failure.
    assert rendered.index("strike:") < rendered.index("episode score")


def test_the_report_records_a_missing_piece_as_absent_not_as_a_pass():
    report = mode_gauntlet_report(_graph())
    assert report["have"] == {"scores": False, "transitions": False,
                              "validation": False, "calibration": False}
    assert report["failed_modes"] == [] and report["gameable_modes"] == []
    for entry in report["modes"].values():
        assert "validation_ok" not in entry and "score" not in entry


def test_the_report_says_a_mode_was_never_entered_rather_than_scoring_it():
    arrays = {k: v[:40] for k, v in _half_active_rollout().items()}
    scores = score_modes(_motion_metric, arrays, _behavior(40), {}, _graph())
    rendered = render_mode_report(mode_gauntlet_report(
        _graph(), scores=scores,
        transitions=check_transitions(_graph(), rollout_frames=40, step_dt=DT)))
    assert "strike: NEVER ENTERED" in rendered
    assert "NEVER FIRED" in rendered
    assert "never left 'approach'" in rendered
