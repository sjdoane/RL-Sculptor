"""tests/test_mode_rewards.py — per-mode reward authoring.

`sculptor.modes` writes down the automaton; this is the half that turns it into
reward code where each mode's terms are paid only inside its own window
(docs/RESEARCH_DIRECTION.md §4, OGMP arXiv 2403.04205).

The load-bearing property is the gating, so most of these EXECUTE the generated
module rather than pattern-matching its text — a scaffold that looks right and
dispatches wrong is exactly the failure the module exists to prevent, and both
real Tier-D failures in this repo were clock bugs that a text assertion would
have sailed past.
"""
from __future__ import annotations

import importlib.util
import re

import numpy as np
import pytest

from sculptor.mode_rewards import (
    BATCHED_FN_SUFFIX,
    MAX_PROMPT_CHARS,
    MODE_COMPONENT_PREFIX,
    MODE_FN_PREFIX,
    authored_modes,
    authoring_twin_source,
    generate_mode_reward_scaffold,
    graft_mode_bodies,
    mode_authoring_prompt,
    mode_ident,
    mode_windows_s,
    validate_mode_reward_source,
)
from sculptor.modes import Guard, Mode, ModeError, ModeGraph, Transition


def _graph(names=("approach", "launch", "land"), fps=30.0, span=30) -> ModeGraph:
    modes = tuple(
        Mode(name=n, frame_range=(i * span, (i + 1) * span))
        for i, n in enumerate(names)
    )
    trans = tuple(
        Transition(from_mode=names[i], to_mode=names[i + 1],
                   guard=Guard(kind="phase", at_phase=1.0))
        for i in range(len(names) - 1)
    )
    return ModeGraph(modes=modes, transitions=trans, fps=fps)


def _load(source, tmp_path, name="mode_reward_mod"):
    p = tmp_path / f"{name}.py"
    p.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _info(t_s, step_dt=0.02):
    return {"episode_length": t_s / step_dt, "step_dt": step_dt}


# ── identifiers ─────────────────────────────────────────────────────────
def test_free_text_mode_names_become_safe_identifiers():
    """Mode names come from a composed clip's provenance and are free text."""
    assert mode_ident("running approach") == "running_approach"
    assert mode_ident("One-Leg Kick!") == "one_leg_kick"
    assert mode_ident("3rd phase").startswith("m_")


def test_a_name_with_no_usable_characters_is_rejected():
    with pytest.raises(ModeError):
        mode_ident("!!!")


def test_names_that_collide_after_sanitizing_are_rejected():
    """Two modes sharing one function body is silent — the second would simply
    never be paid, and the scaffold would look complete."""
    with pytest.raises(ModeError, match="sanitize"):
        generate_mode_reward_scaffold(_graph(("push off", "push-off")))


# ── the gating, executed ────────────────────────────────────────────────
def test_only_the_active_mode_is_paid(tmp_path):
    """The whole point. A term authored for 'land' must not be paid during
    'launch' — episode-level summing is what makes a single scalar fight
    itself."""
    src = generate_mode_reward_scaffold(_graph())
    # Author each mode with a distinguishable constant.
    for i, name in enumerate(("approach", "launch", "land"), start=1):
        src = src.replace(
            f"    del state, action, next_state, info\n    return 0.0, {{}}\n",
            f"    del state, action, next_state, info\n"
            f"    return {float(i)}, {{'k': {float(i)}}}\n", 1)
    mod = _load(src, tmp_path)

    # windows at 30 fps, 30 frames each: [0,1), [1,2), [2,3) seconds
    assert mod.compute_reward(None, None, None, _info(0.5))[0] == 1.0
    assert mod.compute_reward(None, None, None, _info(1.5))[0] == 2.0
    assert mod.compute_reward(None, None, None, _info(2.5))[0] == 3.0


def test_components_are_namespaced_by_mode(tmp_path):
    """Per-mode metrics slice a rollout by these keys, so the naming is part of
    the contract rather than cosmetic."""
    src = generate_mode_reward_scaffold(_graph()).replace(
        "    del state, action, next_state, info\n    return 0.0, {}\n",
        "    del state, action, next_state, info\n"
        "    return 1.0, {'upright': 0.5}\n", 1)
    mod = _load(src, tmp_path)
    _, comp = mod.compute_reward(None, None, None, _info(0.5))
    assert comp["mode_approach"] == 1.0
    assert comp["approach.upright"] == 0.5
    assert comp["active_mode_index"] == 0.0


def test_time_past_the_last_window_stays_in_the_terminal_mode(tmp_path):
    """An episode running long is still IN the last mode, not outside the
    automaton — matches `modes.mode_at_frame`."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path)
    assert mod.active_mode(_info(99.0)) == "land"


def test_there_is_never_an_instant_with_no_owner(tmp_path):
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path)
    for t in (0.0, 0.999, 1.0, 2.999, 3.0, 50.0):
        assert mod.active_mode(_info(t)) in ("approach", "launch", "land")


def test_the_clock_reads_step_dt_rather_than_assuming_a_rate(tmp_path):
    """Both Tier-D failures here were clock bugs. At 100 Hz the same step count
    is half the wall time, so it must land in an earlier mode."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path)
    assert mod.active_mode({"episode_length": 75, "step_dt": 0.02}) == "launch"
    assert mod.active_mode({"episode_length": 75, "step_dt": 0.01}) == "approach"


def test_a_missing_step_dt_falls_back_to_the_real_g1_rate(tmp_path):
    from sculptor.refs.timing import MJLAB_G1_VELOCITY

    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path)
    assert MJLAB_G1_VELOCITY.control_dt == 0.02
    assert mod.active_mode({"episode_length": 75}) == "launch"


def test_windows_are_seconds_derived_from_fps(tmp_path):
    g = _graph(fps=120.0, span=60)          # 0.5 s per mode at 120 fps
    assert mode_windows_s(g)["launch"] == (0.5, 1.0)
    mod = _load(generate_mode_reward_scaffold(g), tmp_path)
    assert mod.active_mode(_info(0.75)) == "launch"


# ── stubs are visible ───────────────────────────────────────────────────
def test_an_unauthored_mode_pays_nothing_and_says_so(tmp_path):
    """A stub must be visibly unauthored — a plausible-looking default would
    let a half-authored graph reach training looking complete."""
    src = generate_mode_reward_scaffold(_graph())
    mod = _load(src, tmp_path)
    assert mod.compute_reward(None, None, None, _info(0.5))[0] == 0.0
    assert authored_modes(src) == {
        "approach": False, "launch": False, "land": False}


def _author(src, mode, *, scalar=True, batched=True):
    """Replace `mode`'s stub bodies the way a real authoring pass would — the
    whole function, docstring included, so the stub marker goes with it.

    Spans come from the library's own AST helper rather than "up to the next
    `def`". A naive scan is wrong for the LAST mode in the module: the text
    between its scalar body and the next `def` includes `_MODE_FNS` and
    `compute_reward`, and splicing over that deletes the dispatch.
    """
    from sculptor.mode_rewards import _fn_span

    out = src
    if scalar:
        fn = f"{MODE_FN_PREFIX}{mode}"
        start, end = _fn_span(out, fn)
        out = out[:start] + (
            f"def {fn}(state, action, next_state, info):\n"
            f'    """{mode}: authored."""\n'
            "    del state, action, next_state, info\n"
            "    return 1.0, {}\n") + out[end:]
    if batched:
        fn = f"{MODE_FN_PREFIX}{mode}{BATCHED_FN_SUFFIX}"
        start, end = _fn_span(out, fn)
        out = out[:start] + (
            f"def {fn}(state, action, next_state, info, like):\n"
            f'    """{mode}: authored."""\n'
            "    del state, action, next_state, info\n"
            "    return like + 1.0, {}\n") + out[end:]
    return out


def test_authoring_one_mode_is_detected(tmp_path):
    """A real authoring pass replaces the whole body, docstring included — so
    the stub marker goes with it. Keeping the marker while changing the return
    still reads as a stub, which is the safe direction to be wrong in."""
    src = _author(generate_mode_reward_scaffold(_graph()), "approach")
    assert authored_modes(src) == {
        "approach": True, "launch": False, "land": False}


def test_a_mode_authored_only_in_the_scalar_half_reads_as_unauthored():
    """The silent one: mjlab dispatches to `compute_reward_batched` and never
    calls the scalar path, so a mode written only in the scalar half evaluates
    correctly in replay and pays exactly zero in training. That looks like a bad
    reward, not a missing one — which is much harder to notice."""
    src = _author(generate_mode_reward_scaffold(_graph()), "approach",
                  batched=False)
    assert authored_modes(src)["approach"] is False
    assert authored_modes(src, require_batched=False)["approach"] is True


def test_a_mode_authored_only_in_the_batched_half_reads_as_unauthored():
    """The mirror case: training would be right and every replay-based score
    would read the mode as contributing nothing."""
    src = _author(generate_mode_reward_scaffold(_graph()), "approach",
                  scalar=False)
    assert authored_modes(src)["approach"] is False


def test_keeping_the_stub_marker_still_reads_as_unauthored(tmp_path):
    """Fail toward 'not yet authored': a body that still carries the marker is
    treated as a stub even if it returns credit, so a half-finished edit cannot
    look complete."""
    src = generate_mode_reward_scaffold(_graph()).replace(
        "    del state, action, next_state, info\n    return 0.0, {}\n",
        "    del state, action, next_state, info\n    return 1.0, {}\n", 1)
    assert authored_modes(src)["approach"] is False


# ── validation ──────────────────────────────────────────────────────────
def test_a_valid_scaffold_validates_clean():
    g = _graph()
    assert validate_mode_reward_source(generate_mode_reward_scaffold(g), g) == []


def test_a_scaffold_stale_against_a_renamed_mode_is_caught():
    """The silent dead end: the graph gained a mode after the scaffold was
    written, so that mode's terms could never be paid."""
    src = generate_mode_reward_scaffold(_graph(("approach", "launch")))
    errors = validate_mode_reward_source(src, _graph(("approach", "launch", "land")))
    assert any("land" in e for e in errors)


def test_a_scaffold_stale_against_shifted_windows_is_caught():
    src = generate_mode_reward_scaffold(_graph(span=30))
    errors = validate_mode_reward_source(src, _graph(span=45))
    assert any("stale" in e for e in errors)


def test_validation_reports_every_problem_at_once():
    """Mirrors validate_mode_graph, so a generator retry gets complete
    feedback instead of one error per round trip."""
    errors = validate_mode_reward_source("# empty", _graph())
    assert len(errors) >= 4          # compute_reward, MODE_WINDOWS_S, 3 modes
    assert any("compute_reward" in e for e in errors)
    assert any("MODE_WINDOWS_S" in e for e in errors)


def test_an_invalid_graph_is_refused_rather_than_scaffolded():
    bad = ModeGraph(modes=(Mode(name="a", frame_range=(10, 10)),),
                    transitions=(), fps=30.0)
    with pytest.raises(ModeError):
        generate_mode_reward_scaffold(bad)


# ── the authoring prompt ────────────────────────────────────────────────
def test_the_prompt_states_the_window_and_the_neighbours():
    """The main per-mode authoring failure is a term that is right for its mode
    but written as a global constraint, so scope is stated explicitly."""
    p = mode_authoring_prompt(_graph(), "launch",
                              behavior_goal="running jump kick",
                              mode_goal="drive off the back foot")
    assert f"{MODE_FN_PREFIX}launch" in p
    assert "1s-2s" in p
    assert "approach" in p and "land" in p
    assert "running jump kick" in p and "drive off the back foot" in p


def test_the_prompt_tells_the_author_not_to_re_detect_phase():
    """Phase detection inside an authored body would duplicate — and could
    contradict — the derived dispatch."""
    p = mode_authoring_prompt(_graph(), "approach")
    assert "do not re-detect the phase" in p.lower()


def test_the_first_and_last_modes_have_only_one_neighbour():
    first = mode_authoring_prompt(_graph(), "approach")
    last = mode_authoring_prompt(_graph(), "land")
    assert "after 'approach'" not in first and "before 'launch'" in first
    assert "after 'launch'" in last and "before " not in last


def test_a_single_mode_graph_reads_as_the_only_mode():
    assert "the only mode" in mode_authoring_prompt(_graph(("solo",)), "solo")


def test_asking_for_an_unknown_mode_is_a_caller_bug():
    with pytest.raises(KeyError):
        mode_authoring_prompt(_graph(), "nope")


# ── the spec the metric layer reads ─────────────────────────────────────
def test_reward_spec_publishes_the_windows_for_the_metric_layer(tmp_path):
    """A per-mode metric scores each mode's own slice; it reads the windows
    from here rather than re-deriving the automaton."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path)
    assert mod.REWARD_SPEC["mode_windows_s"]["launch"] == [1.0, 2.0]
    assert mod.MODE_ORDER == ["approach", "launch", "land"]


# ── the real path: composed clip -> automaton -> scaffold ───────────────
def test_a_composed_clip_becomes_an_authorable_scaffold(tmp_path):
    """End-to-end on the shape `refs.compose` actually writes. This is the
    composition Lokesh's §4 names — one composed segment is one mode, and the
    seam between segments is the transition — so it must work on real
    provenance, not only on hand-built graphs.
    """
    from sculptor.modes import modes_from_composition

    fps = 120.0
    clip = {
        "root_pos_z": np.zeros(444),
        "fps": fps,
        "meta": {"composition": {
            # One seam fewer than segments — the seams ARE the transitions.
            "seam_frames": [150, 300],
            "segments": [
                {"index": 0, "label": "approach", "source_id": "run",
                 "source_fps": 60.0, "source_frames": [60, 150]},
                {"index": 1, "label": "launch", "source_id": "jump",
                 "source_fps": 60.0, "source_frames": [10, 85]},
                {"index": 2, "label": "strike", "source_id": "kick",
                 "source_fps": 60.0, "source_frames": [0, 72]},
            ]}},
    }
    g = modes_from_composition(clip)
    assert [m.name for m in g.modes] == ["approach", "launch", "strike"]

    src = generate_mode_reward_scaffold(
        g, behavior_goal="running approach into a one-leg jumping kick",
        goal_by_mode={"launch": "drive off the plant foot"})
    assert validate_mode_reward_source(src, g) == []

    mod = _load(src, tmp_path, name="composed_mode_reward")
    # Each segment's window must own its own slice of the timeline.
    seen = {mod.active_mode(_info(t)) for t in (0.1, 1.5, 3.0)}
    assert seen == {"approach", "launch", "strike"}
    # ...and the per-mode goal reaches the prompt for the mode it was given for.
    assert "drive off the plant foot" in mode_authoring_prompt(
        g, "launch", mode_goal="drive off the plant foot")


def test_a_single_segment_composition_is_refused_upstream():
    """Deriving an automaton from a one-segment composition is refused by
    `modes_from_composition` — there is no seam, so there is no transition to
    read. Pinned here because the reward layer would happily scaffold it, and
    the two layers must not disagree about what a mode graph is."""
    from sculptor.modes import modes_from_composition

    clip = {"root_pos_z": np.zeros(120), "fps": 60.0,
            "meta": {"composition": {"segments": [
                {"index": 0, "label": "solo", "source_id": "s",
                 "source_fps": 60.0, "source_frames": [0, 120]}]}}}
    with pytest.raises(ModeError, match="at least 2"):
        modes_from_composition(clip)


def test_a_hand_built_one_mode_graph_still_scaffolds(tmp_path):
    """A one-mode automaton is degenerate but legal at the reward layer — a
    single-clip stage is exactly that, and it must not be a special case."""
    mod = _load(generate_mode_reward_scaffold(_graph(("solo",))),
                tmp_path, name="solo_mode")
    assert mod.active_mode(_info(0.0)) == "solo"
    assert mod.active_mode(_info(99.0)) == "solo"


# ── the batched path (the one mjlab actually trains on) ─────────────────
torch = pytest.importorskip("torch")


def _steps(n=260, step_dt=0.02):
    s = torch.arange(0, n, dtype=torch.float32)
    return s, {"episode_length": s, "step_dt": torch.full_like(s, step_dt)}


def test_the_scaffold_declares_and_defines_the_batched_path(tmp_path):
    """mjlab dispatches to `compute_reward_batched` and treats its absence as a
    reward-contract violation (`adapters/mjlab.py:670`), falling back to a
    per-env Python loop. Without this the whole module could not train."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b0")
    assert mod.REWARD_SPEC["supports_batched"] is True
    assert callable(mod.compute_reward_batched)


def test_batched_masks_agree_with_the_scalar_dispatch_step_for_step(tmp_path):
    """The load-bearing invariant. A rollout is SCORED through the scalar path
    and TRAINED through the batched one; if they disagreed about which mode owns
    an instant, the metric would be grading terms the policy was never paid for.
    Swept past the terminal window so the overrun fallback is covered too."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b1")
    steps, info = _steps()
    like = mod._batch_like(torch.zeros(len(steps), 3), {}, info)
    masks = torch.stack(mod._mode_masks(like, info))

    assert masks.sum(0).eq(1).all(), "modes must partition every instant"
    batched = [mod.MODE_ORDER[int(i)] for i in masks.float().argmax(0)]
    scalar = [mod.active_mode({"episode_length": float(s), "step_dt": 0.02})
              for s in steps]
    assert batched == scalar


def test_envs_at_different_episode_times_get_different_modes(tmp_path):
    """Why this is a tensor and not a scalar: mjlab's envs reset independently,
    so at any given step they sit at different points in the automaton. A scalar
    clock would put the whole batch in one mode and pay most of it the wrong
    terms."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b2")
    # 0.5 s, 1.5 s, 2.5 s into their episodes — one env per mode.
    steps = torch.tensor([25.0, 75.0, 125.0])
    info = {"episode_length": steps, "step_dt": torch.full_like(steps, 0.02)}
    _, comps = mod.compute_reward_batched({}, torch.zeros(3, 3), {}, info)
    assert comps["active_mode_index"].tolist() == [0.0, 1.0, 2.0]


def test_an_unauthored_batched_scaffold_pays_nothing_finitely(tmp_path):
    """The pre-flight probe in `sculptor.edit` runs this on zero tensors and
    rejects non-finite rewards, so a fresh scaffold has to survive it."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b3")
    steps, info = _steps()
    r, comps = mod.compute_reward_batched({}, torch.zeros(len(steps), 3), {}, info)
    assert tuple(r.shape) == (len(steps),)
    assert torch.isfinite(r).all() and r.abs().sum() == 0
    assert comps, "components must be non-empty — edit.py rejects an empty dict"
    for name in mod.MODE_ORDER:
        assert f"{MODE_COMPONENT_PREFIX}{name}" in comps


def test_a_mode_is_paid_only_inside_its_own_window_batched(tmp_path):
    mod = _load(_author(generate_mode_reward_scaffold(_graph()), "launch"),
                tmp_path, name="b4")
    steps, info = _steps()
    t = steps * 0.02
    r, comps = mod.compute_reward_batched({}, torch.zeros(len(steps), 3), {}, info)
    inside = (t >= 1.0) & (t < 2.0)          # launch's window at 30 fps / 30 frames
    assert (r[inside] == 1.0).all()
    assert (r[~inside] == 0.0).all()
    assert (comps[f"{MODE_COMPONENT_PREFIX}approach"] == 0.0).all()


def test_out_of_window_nan_cannot_poison_the_batch(tmp_path):
    """Why the dispatch masks with `torch.where` and not `mask * value`.

    A mode's terms are only defined inside its own window, but every mode's
    function is evaluated for every env before masking — that is what makes it
    vectorizable. `0.0 * nan` is `nan`, so a multiply would let one out-of-window
    env take down the entire batch's reward. `where` discards the unselected
    branch, while a nan produced INSIDE the window still surfaces, which is the
    direction a numerical bug should fail in."""
    src = generate_mode_reward_scaffold(_graph())
    start = src.index(f"def {MODE_FN_PREFIX}launch{BATCHED_FN_SUFFIX}(")
    tail = src.index("\n\ndef ", start)
    src = src[:start] + (
        f"def {MODE_FN_PREFIX}launch{BATCHED_FN_SUFFIX}"
        "(state, action, next_state, info, like):\n"
        '    """launch: nan before this window opens."""\n'
        "    del state, action, next_state\n"
        "    import torch\n"
        "    return torch.sqrt(_elapsed_s_batched(like, info) - 1.0), {}\n"
    ) + src[tail:]

    mod = _load(src, tmp_path, name="b5")
    steps, info = _steps()
    t = steps * 0.02
    raw = torch.sqrt(t - 1.0)
    assert raw.isnan().any(), "the fixture must actually produce nan"
    assert (raw * (t < 1.0).float()).isnan().any(), "a multiply would spread it"

    r, comps = mod.compute_reward_batched({}, torch.zeros(len(steps), 3), {}, info)
    assert torch.isfinite(r).all()
    inside = (t >= 1.0) & (t < 2.0)
    assert torch.allclose(r[inside], raw[inside]), "in-window value is untouched"


def test_the_batched_clock_reads_step_dt_rather_than_assuming_a_rate(tmp_path):
    """Same assumption that broke Tier-D twice, now on the training path."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b6")
    steps = torch.tensor([50.0])
    slow = {"episode_length": steps, "step_dt": torch.tensor([0.02])}   # 1.0 s
    fast = {"episode_length": steps, "step_dt": torch.tensor([0.005])}  # 0.25 s
    like = mod._batch_like(torch.zeros(1, 3), {}, slow)
    assert float(mod._elapsed_s_batched(like, slow)) == pytest.approx(1.0)
    assert float(mod._elapsed_s_batched(like, fast)) == pytest.approx(0.25)
    _, a = mod.compute_reward_batched({}, torch.zeros(1, 3), {}, slow)
    _, b = mod.compute_reward_batched({}, torch.zeros(1, 3), {}, fast)
    assert a["active_mode_index"].tolist() == [1.0]   # 'launch'
    assert b["active_mode_index"].tolist() == [0.0]   # still 'approach'


def test_a_missing_step_dt_falls_back_to_the_real_g1_rate_batched(tmp_path):
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b7")
    steps = torch.tensor([75.0])
    like = mod._batch_like(torch.zeros(1, 3), {}, {})
    bare = float(mod._elapsed_s_batched(like, {"episode_length": steps}))
    assert bare == pytest.approx(75.0 * 0.02)
    # A published-but-zero step_dt must take the fallback too, matching the
    # scalar path's `or DEFAULT_STEP_DT` rather than collapsing every env to t=0.
    zeroed = float(mod._elapsed_s_batched(
        like, {"episode_length": steps, "step_dt": torch.zeros(1)}))
    assert zeroed == bare


def test_the_batch_size_is_derived_rather_than_assumed(tmp_path):
    """`edit.py`'s probe builds state/next_state from the contract's schema, so
    no single key is guaranteed present."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b8")
    n = 7
    from_info = mod._batch_like(None, None, {"episode_length": torch.zeros(n)})
    from_action = mod._batch_like(torch.zeros(n, 3), {}, {})
    from_state = mod._batch_like(None, {"qpos": torch.zeros(n, 35)}, {})
    for like in (from_info, from_action, from_state):
        assert tuple(like.shape) == (n,) and float(like.sum()) == 0.0
    with pytest.raises(ValueError, match="batch size"):
        mod._batch_like(None, {}, {})


def test_a_long_behavior_goal_cannot_blow_the_prompt_budget():
    """`apply_prompt_edit` hard-rejects a prompt over 2000 chars (edit.py:2042),
    and a behavior goal is free text a user typed. Budgeting here means a long
    goal is truncated visibly in `--print-prompt`; without it the authoring call
    fails at the very end, after the KG query, with an error about a character
    count rather than about the goal."""
    p = mode_authoring_prompt(_graph(), "launch",
                              behavior_goal="sprint and " * 400,
                              mode_goal="take off from one leg " * 400)
    assert len(p) <= MAX_PROMPT_CHARS
    assert "…" in p, "truncation must be visible, not silent"
    # The window and the scope rules survive — they are what the prompt is for.
    assert "Window:" in p and "_mode_masks" in p


def test_a_realistic_goal_leaves_the_prompt_well_inside_the_budget():
    p = mode_authoring_prompt(
        _graph(), "launch",
        behavior_goal="run in, launch off one leg, strike at the apex",
        mode_goal="convert horizontal speed into a single-leg takeoff")
    assert len(p) <= MAX_PROMPT_CHARS
    assert "…" not in p, "a normal goal must not be truncated"


def test_the_prompt_asks_for_both_halves():
    """A prompt that asks for one body would produce exactly the half-authored
    mode `authored_modes` refuses to call done."""
    p = mode_authoring_prompt(_graph(), "launch")
    assert f"{MODE_FN_PREFIX}launch{BATCHED_FN_SUFFIX}" in p
    assert "num_envs" in p and "torch.zeros_like(like)" in p


# ── the reference-tracking backbone ─────────────────────────────────────
def _tracking_clip(n=240, fps=120.0, j=6):
    t = np.arange(n, dtype=np.float64) / fps
    return {
        "fps": fps,
        "joint_names": [f"joint_{i}" for i in range(j)],
        "joint_pos": (0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
                      + 0.01 * np.arange(j)[None, :]),
        "root_pos_z": 0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
    }


def test_without_a_clip_the_scaffold_pays_exactly_zero(tmp_path):
    """Which is the problem the backbone exists to solve: a stubs-only module
    is not trainable until every mode has been authored, and even then nothing
    tells the policy to follow the reference."""
    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="t0")
    assert not hasattr(mod, "TARGET_JOINT_POS")
    v, _ = mod.compute_reward({}, None, {}, _info(1.5))
    assert v == 0.0


def test_the_backbone_makes_a_fresh_scaffold_trainable(tmp_path):
    """The whole point: scaffold, then train, then author on top — rather than
    author three modes before finding out whether any of it works."""
    clip = _tracking_clip()
    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=clip),
                tmp_path, name="t1")
    assert mod.N_JOINTS == 6 and mod.N_PHASE == 32
    assert mod.REFERENCE_DURATION_S == pytest.approx(2.0)

    qpos = np.zeros(7 + mod.N_JOINTS)
    qpos[2] = 0.70
    v, comps = mod.compute_reward(
        {}, None, {"qpos": qpos, "projected_gravity_b": np.array([0., 0., -1.])},
        _info(1.0))
    assert v > 0.0, "an unauthored scaffold with a backbone still pays"
    assert {"joint_tracking", "root_tracking", "orientation_tracking"} <= set(comps)
    # The mode's own contribution is still zero — the backbone is not a mode.
    assert comps[f"{MODE_COMPONENT_PREFIX}launch"] == 0.0


def test_the_scalar_path_handles_a_joints_only_qpos(tmp_path):
    """Caught by a real authoring run, not by reasoning about it.

    The G1 task's reward contract declares `qpos: (29,)` — the ACTUATED joints
    only, no free-joint DOFs — and `sculptor.edit` probes the scalar path with
    exactly that. Slicing `qpos[7:7+N]` raised `qpos too short for 29 tracked
    joints: (29,)` and the whole authoring call was rejected after the model had
    already been called. Both layouts are real, so both are handled."""
    clip = _tracking_clip()
    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=clip),
                tmp_path, name="t2a")
    joints = np.linspace(-0.2, 0.2, mod.N_JOINTS)
    info = {"episode_length": 50, "step_dt": 0.02, "base_height": 0.70}

    joints_only, _ = mod.compute_reward(
        {}, None, {"qpos": joints,
                   "projected_gravity_b": np.array([0., 0., -1.])}, info)
    full = np.zeros(7 + mod.N_JOINTS)
    full[2] = 0.70
    full[7:] = joints
    full_mujoco, _ = mod.compute_reward(
        {}, None, {"qpos": full,
                   "projected_gravity_b": np.array([0., 0., -1.])}, info)
    assert joints_only == pytest.approx(full_mujoco), (
        "the trailing N_JOINTS are the same joints in both layouts")


def test_the_two_paths_agree_on_the_terms_they_can_agree_on(tmp_path):
    """`root_tracking` differs by construction — the scalar path compares
    absolute root z while the batched path compares a delta from the
    reference's first frame, because retargeting zeroes root translation and an
    absolute comparison would saturate the kernel at zero for every frame. The
    joint and orientation terms have no such excuse and must match."""
    clip = _tracking_clip()
    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=clip),
                tmp_path, name="t2")
    rng = np.random.default_rng(0)
    for step in range(0, 120, 11):
        j = rng.normal(0.0, 0.3, mod.N_JOINTS)
        grav = np.array([0.05, -0.02, -0.998])
        grav = grav / np.linalg.norm(grav)

        qpos = np.zeros(7 + mod.N_JOINTS)
        qpos[2] = 0.70
        qpos[7:] = j
        _, s = mod.compute_reward(
            {}, None, {"qpos": qpos, "projected_gravity_b": grav},
            {"episode_length": step, "step_dt": 0.02})

        qb = torch.zeros(1, 7 + mod.N_JOINTS)
        qb[0, -mod.N_JOINTS:] = torch.tensor(j, dtype=torch.float32)
        _, b = mod.compute_reward_batched(
            {}, torch.zeros(1, mod.N_JOINTS),
            {"qpos": qb,
             "projected_gravity_b": torch.tensor(grav, dtype=torch.float32).reshape(1, 3)},
            {"episode_length": torch.tensor([float(step)]),
             "step_dt": torch.tensor([0.02]),
             "base_height": torch.tensor([0.70])})
        assert s["joint_tracking"] == pytest.approx(
            float(b["joint_tracking"]), abs=1e-6)
        assert s["orientation_tracking"] == pytest.approx(
            float(b["orientation_tracking"]), abs=1e-6)


def test_the_tracking_clock_spans_the_composite_not_the_mode(tmp_path):
    """The automaton decides which TASK terms apply; it does not change what
    the robot is supposed to be tracking. Re-anchoring phase per mode is a
    defensible design but a different one, and it would mean the backbone is no
    longer the version that has actually been measured."""
    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=_tracking_clip()),
                tmp_path, name="t3")
    # Three modes over 2.0 s. Phase must advance monotonically ACROSS mode
    # boundaries rather than resetting at each one.
    idx = [mod._phase_index(_info(t)) for t in (0.0, 0.6, 0.7, 1.3, 1.4, 1.99)]
    assert idx == sorted(idx)
    assert idx[0] == 0 and idx[-1] == mod.N_PHASE - 1


def test_a_clip_without_orientation_zeroes_the_term_rather_than_guessing(tmp_path):
    """An upright target would be a fabricated one, and it would charge an
    error against a robot that is correctly pitched over."""
    clip = _tracking_clip()
    clip.pop("root_quat_wxyz")
    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=clip),
                tmp_path, name="t4")
    assert mod.TARGET_GRAVITY is None
    assert mod.ORIENTATION_ERR_WEIGHT == 0.0
    qpos = np.zeros(7 + mod.N_JOINTS)
    _, comps = mod.compute_reward({}, None, {"qpos": qpos}, _info(1.0))
    assert "orientation_tracking" not in comps


def test_a_clip_with_no_joint_target_is_refused_rather_than_scaffolded():
    clip = _tracking_clip()
    clip.pop("joint_pos")
    with pytest.raises(ModeError, match="joint_pos"):
        generate_mode_reward_scaffold(_graph(fps=120.0, span=80), clip=clip)


def test_the_backbone_survives_the_pre_flight_probe(tmp_path):
    import types

    from sculptor.edit import _call_compute_reward_batched

    mod = _load(generate_mode_reward_scaffold(_graph(fps=120.0, span=80),
                                              clip=_tracking_clip()),
                tmp_path, name="t5")
    contract = types.SimpleNamespace(
        supports_batched=True,
        state_schema={"qpos": (13,), "projected_gravity_b": (3,),
                      "actuator_force": (6,)},
        info_schema={"episode_length": (), "step_dt": (), "base_height": ()},
        expected_info_keys=["episode_length", "step_dt", "base_height"])
    _call_compute_reward_batched(mod, contract)


# ── grafting an authored mode into a backbone-carrying module ───────────
def test_grafting_moves_a_mode_without_touching_the_tables(tmp_path):
    """The reason grafting exists. `apply_prompt_edit` rewrites the WHOLE
    module, and a scaffold with the tracking backbone is ~27 KB of which most
    is a table of float literals. The first real authoring run against one came
    back `SyntaxError: '[' was never closed` — the model had mangled the table
    while reproducing it. So authoring runs against a stubs-only twin and the
    result is transplanted."""
    clip = _tracking_clip()
    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=clip)
    twin = _author(generate_mode_reward_scaffold(g), "launch")

    grafted = graft_mode_bodies(full, twin, ["launch"])
    assert validate_mode_reward_source(grafted, g) == []
    assert authored_modes(grafted) == {
        "approach": False, "launch": True, "land": False}
    # The tables came from the deterministic side and are untouched.
    for marker in ("TARGET_JOINT_POS", "TARGET_ROOT_Z", "TARGET_GRAVITY"):
        i, j = full.index(marker), grafted.index(marker)
        assert full[i:i + 400] == grafted[j:j + 400]

    mod = _load(grafted, tmp_path, name="gr1")
    qpos = np.zeros(mod.N_JOINTS)
    v, comps = mod.compute_reward(
        {}, None, {"qpos": qpos, "projected_gravity_b": np.array([0., 0., -1.])},
        {"episode_length": 50, "step_dt": 0.02, "base_height": 0.70})
    assert comps[f"{MODE_COMPONENT_PREFIX}launch"] == 1.0, "authored term is live"
    assert v > 1.0, "and the backbone is still paid alongside it"


def test_grafting_drops_edits_the_model_had_no_business_making(tmp_path):
    """A model that rewrites the dispatch, the windows or another mode's body
    gets that edit dropped rather than having to be caught downstream — the
    graft only ever moves the functions it was asked for."""
    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=_tracking_clip())
    twin = _author(_author(generate_mode_reward_scaffold(g), "launch"), "land")
    twin = twin.replace("MODE_ORDER: list = ['approach', 'launch', 'land']",
                        "MODE_ORDER: list = ['launch']")

    grafted = graft_mode_bodies(full, twin, ["launch"])
    assert "MODE_ORDER: list = ['approach', 'launch', 'land']" in grafted
    # 'land' was authored in the twin but not requested, so it stays a stub.
    assert authored_modes(grafted)["land"] is False


def test_grafting_a_mode_that_is_not_in_the_authored_module_is_refused():
    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=_tracking_clip())
    with pytest.raises(ModeError, match="nothing to graft"):
        graft_mode_bodies(full, "def unrelated():\n    pass\n", ["launch"])


def test_a_grafted_module_still_clears_the_pre_flight_probes(tmp_path):
    """`apply_prompt_edit` validated the TWIN; the grafted module is what
    trains. Re-probing is what makes the graft safe rather than convenient."""
    import types

    from sculptor.edit import _call_compute_reward, _call_compute_reward_batched

    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=_tracking_clip())
    grafted = graft_mode_bodies(full, _author(
        generate_mode_reward_scaffold(g), "launch"), ["launch"])
    mod = _load(grafted, tmp_path, name="gr2")
    contract = types.SimpleNamespace(
        supports_batched=True,
        state_schema={"qpos": (6,), "projected_gravity_b": (3,),
                      "actuator_force": (6,)},
        info_schema={"episode_length": (), "step_dt": (), "base_height": ()},
        expected_info_keys=["episode_length", "step_dt", "base_height"])
    _call_compute_reward(mod, contract)
    _call_compute_reward_batched(mod, contract)


def test_the_authoring_twin_is_small_and_carries_no_tables(tmp_path):
    """Both real authoring runs died on the model's output budget: the first
    truncated inside the 32x29 float table (`SyntaxError: '[' was never
    closed`), the second only after the tables were gone. `apply_prompt_edit`
    rewrites the WHOLE module, so every byte in it is a byte the model has to
    reproduce."""
    clip = _tracking_clip(j=29)          # G1's real actuated-joint count
    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=clip)
    twin = authoring_twin_source(g, clip=clip)

    assert len(twin) < len(full) / 2
    assert "np.zeros((N_PHASE, N_JOINTS)" in twin
    assert "TARGET_JOINT_POS" in twin, "the shape stays; only the numbers go"
    assert validate_mode_reward_source(twin, g) == []
    # The per-mode docstrings survive — they are the part being acted on.
    assert authored_modes(twin) == {
        "approach": False, "launch": False, "land": False}
    # The real joint count and duration do, too: a body that indexes joints
    # must see the same shape it will see after grafting.
    mod = _load(twin, tmp_path, name="tw1")
    assert mod.N_JOINTS == 29
    assert mod.REFERENCE_DURATION_S == pytest.approx(2.0)


def test_the_twin_is_state_dependent_so_edits_own_gate_accepts_it(tmp_path):
    """The second real run was rejected with `reward is state-independent:
    compute_reward returned the IDENTICAL total (0.0)`. That gate is right —
    a constant reward gives PPO no gradient — but a per-mode module is
    DELIBERATELY zero outside the active mode, so a stubs-only twin reads as
    constant-0 and is rejected for a property it is supposed to have. The
    placeholder backbone reads qpos, height and gravity, so the twin clears the
    gate on the same grounds the real module does."""
    mod = _load(authoring_twin_source(_graph(fps=120.0, span=80),
                                      clip=_tracking_clip()),
                tmp_path, name="tw2")
    info = {"episode_length": 20, "step_dt": 0.02, "base_height": 0.70}
    a, _ = mod.compute_reward(
        {}, None, {"qpos": np.linspace(-0.3, 0.3, mod.N_JOINTS),
                   "projected_gravity_b": np.array([0., 0., -1.])}, info)
    b, _ = mod.compute_reward(
        {}, None, {"qpos": np.linspace(-0.9, 0.9, mod.N_JOINTS),
                   "projected_gravity_b": np.array([0.2, 0., -0.98])},
        {**info, "base_height": 0.90})
    assert a != b, "a stubs-only twin would return 0.0 for both"


def test_a_twin_without_a_clip_still_has_no_backbone(tmp_path):
    """`clip=None` is the stubs-only case, which stays available — it is what
    scalar-only adapters and the `--no-tracking` path want."""
    twin = authoring_twin_source(_graph(), clip=None)
    assert "TARGET_JOINT_POS" not in twin
    assert validate_mode_reward_source(twin, _graph()) == []


def test_function_spans_are_parsed_rather_than_pattern_matched():
    """The graft is how an authored mode reaches the trainable module, and a
    regex has to guess where a function ends from blank lines — a property of
    whatever formatting the model happened to emit. Two blank lines, one, or
    none must all graft the same."""
    g = _graph()
    base = generate_mode_reward_scaffold(g)
    authored = _author(base, "launch")
    tight = re.sub(r"\n\n+(?=def )", "\n", authored)
    assert "\n\ndef " not in tight, "the fixture must actually be tight"

    loose = graft_mode_bodies(base, authored, ["launch"])
    packed = graft_mode_bodies(base, tight, ["launch"])
    assert authored_modes(loose)["launch"] is True
    assert authored_modes(packed)["launch"] is True
    assert validate_mode_reward_source(packed, g) == []


def test_a_fresh_scaffold_passes_edits_real_pre_flight_probe(tmp_path):
    """The actual admission gate, not a stand-in for it.

    `_call_compute_reward_batched` is what `apply_prompt_edit` runs before a
    reward is allowed near a GPU — it executes the batched path on N=2 zero
    tensors and rejects wrong shapes, empty component dicts and non-finite
    rewards. Its docstring records why it exists: a bool-tensor arithmetic bug
    that got caught live AFTER training started and burned the stage. A scaffold
    that cannot clear this probe cannot be authored against at all.
    """
    import types

    from sculptor.edit import _call_compute_reward_batched

    mod = _load(generate_mode_reward_scaffold(_graph()), tmp_path, name="b9")
    mjlab = types.SimpleNamespace(
        supports_batched=True,
        state_schema={"qpos": (36,), "qvel": (35,), "actuator_force": (29,),
                      "projected_gravity_b": (3,)},
        info_schema={"episode_length": (), "step_dt": (), "base_height": ()},
        expected_info_keys=["episode_length", "step_dt", "base_height"])
    _call_compute_reward_batched(mod, mjlab)          # raises on failure

    # A contract that declares no info keys at all: the clock finds nothing to
    # read and must still produce a correctly shaped, finite reward.
    bare = types.SimpleNamespace(
        supports_batched=True, state_schema={"actuator_force": (29,)},
        info_schema={}, expected_info_keys=[])
    _call_compute_reward_batched(mod, bare)


def test_validation_catches_a_scaffold_with_the_batched_half_stripped():
    g = _graph()
    src = generate_mode_reward_scaffold(g)
    start = src.index(f"def {MODE_FN_PREFIX}launch{BATCHED_FN_SUFFIX}(")
    stripped = src[:start] + src[src.index("\n\ndef ", start) + 2:]
    errors = validate_mode_reward_source(stripped, g)
    assert any("launch" in e and BATCHED_FN_SUFFIX in e for e in errors)


# ── promotion into the reward version chain ──────────────────────────────
def _rewards_dir(tmp_path, *, tracking=True, author_all=True):
    """A project-shaped rewards/ holding a scaffold and a v0 starter."""
    from sculptor.mode_rewards import generate_mode_reward_scaffold

    d = tmp_path / "rewards"
    d.mkdir()
    (d / "v0.py").write_text(
        "REWARD_SPEC: dict = {'version': 'v0', 'parent_hash': ''}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 1.0, {'alive': 1.0}\n", encoding="utf-8")
    src = generate_mode_reward_scaffold(
        _graph(), clip=_tracking_clip() if tracking else None,
        behavior_goal="a test behavior", clip_id="c--g1")
    if author_all:
        for m in _graph().modes:
            src = _author(src, m.name)
    (d / "mode_reward_v3.py").write_text(src, encoding="utf-8")
    return d


def test_promoting_puts_the_module_in_the_version_chain(tmp_path):
    """Without this the feature does nothing: only `v<n>.py` is a version, and
    `current.py` — what every adapter imports — points at one."""
    from sculptor.mode_rewards import promote_mode_reward

    d = _rewards_dir(tmp_path)
    out = promote_mode_reward(d / "mode_reward_v3.py")
    assert out["version"] == 1
    assert (d / "v1.py").is_file()
    assert "v1.py" in (d / "current.py").read_text(encoding="utf-8")
    assert out["parent_hash"] and len(out["parent_hash"]) == 16


def test_the_promoted_spec_reads_as_the_version_it_now_is(tmp_path):
    """`reward_store` literal-evals the REWARD_SPEC dict and never executes
    the module, so the version has to be rewritten IN the literal — a
    `REWARD_SPEC[...] = ...` line at the bottom would be invisible to it."""
    import ast

    from sculptor.mode_rewards import promote_mode_reward

    d = _rewards_dir(tmp_path)
    promote_mode_reward(d / "mode_reward_v3.py")
    tree = ast.parse((d / "v1.py").read_text(encoding="utf-8"))
    spec = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                and "REWARD_SPEC" in ast.dump(n))
    assert spec["version"] == "v1"
    assert spec["author"] == "sculptor"
    assert spec["parent_hash"]
    # and the rest of the spec survived the rewrite — the windows in
    # particular, which are how a per-mode metric finds its slice.
    assert sorted(spec["mode_windows_s"]) == sorted(
        m.name for m in _graph().modes)


def test_promoting_a_half_authored_module_is_refused(tmp_path):
    """A stub pays nothing. Promoting one trains a reward that is blank across
    part of the episode — silently, since everything else about it validates."""
    from sculptor.mode_rewards import ModeAuthorError, promote_mode_reward

    d = _rewards_dir(tmp_path, author_all=False)
    with pytest.raises(ModeAuthorError) as e:
        promote_mode_reward(d / "mode_reward_v3.py")
    assert "unauthored stub" in str(e.value)
    for m in _graph().modes:
        assert m.name in str(e.value)
    assert not (d / "v1.py").exists(), "refused means nothing was written"


def test_a_scaffold_can_be_promoted_on_purpose(tmp_path):
    """The tracking backbone alone IS trainable — that is the whole Tier-D
    path — so this is a flag, not a hard refusal."""
    from sculptor.mode_rewards import promote_mode_reward

    d = _rewards_dir(tmp_path, author_all=False)
    out = promote_mode_reward(d / "mode_reward_v3.py", allow_unauthored=True)
    assert out["version"] == 1
    assert sorted(out["unauthored"]) == sorted(m.name for m in _graph().modes)


def test_promoting_chains_rather_than_overwriting(tmp_path):
    from sculptor.mode_rewards import promote_mode_reward

    d = _rewards_dir(tmp_path)
    assert promote_mode_reward(d / "mode_reward_v3.py")["version"] == 1
    assert promote_mode_reward(d / "mode_reward_v3.py")["version"] == 2
    assert (d / "v1.py").is_file() and (d / "v2.py").is_file()
    assert "v2.py" in (d / "current.py").read_text(encoding="utf-8")


def test_the_promoted_module_still_imports_and_pays_per_mode(tmp_path):
    """The point of promotion is that current.py loads it — so the promoted
    file has to survive the spec rewrite as working code."""
    import importlib.util

    from sculptor.mode_rewards import promote_mode_reward

    d = _rewards_dir(tmp_path)
    promote_mode_reward(d / "mode_reward_v3.py")
    spec = importlib.util.spec_from_file_location("promoted", d / "current.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compute_reward")
    assert hasattr(mod, "compute_reward_batched"), "mjlab looks this up on current"
    total, comps = mod.compute_reward(
        {}, None,
        {"qpos": [0.0] * 6, "projected_gravity_b": [0.0, 0.0, -1.0]},
        _info(0.5))
    assert isinstance(total, float)
    assert any(k.startswith(MODE_COMPONENT_PREFIX) for k in comps)


def test_grafting_carries_a_helper_bound_by_tuple_unpacking(tmp_path):
    """The shape that killed a real authoring run.

    A model asked to reward "come to rest between 4.92s and 6.92s" writes its
    window bounds as one statement — `_SETTLE_LO, _SETTLE_HI = 4.92, 6.92` —
    and `_module_bindings` only collected bare-`Name` assignment targets. The
    graft therefore did not believe the donor defined those names, dropped the
    statement, and the module died at the contract probe with
    `NameError: name '_SETTLE_HI' is not defined` after a three-minute call.
    """
    from sculptor.mode_rewards import _fn_span

    g = _graph(fps=120.0, span=80)
    full = generate_mode_reward_scaffold(g, clip=_tracking_clip())
    twin = generate_mode_reward_scaffold(g)

    fn = f"{MODE_FN_PREFIX}launch"
    start, end = _fn_span(twin, fn)
    twin = twin[:start] + (
        f"def {fn}(state, action, next_state, info):\n"
        '    """launch: authored."""\n'
        "    del state, action, next_state, info\n"
        "    return _LAUNCH_HI - _LAUNCH_LO, {}\n") + twin[end:]
    start, end = _fn_span(twin, fn + BATCHED_FN_SUFFIX)
    twin = twin[:start] + (
        f"def {fn}{BATCHED_FN_SUFFIX}(state, action, next_state, info, like):\n"
        '    """launch: authored."""\n'
        "    del state, action, next_state, info\n"
        "    return like + (_LAUNCH_HI - _LAUNCH_LO), {}\n") + twin[end:]
    twin = "_LAUNCH_LO, _LAUNCH_HI = 0.25, 0.75\n\n" + twin

    grafted = graft_mode_bodies(full, twin, ["launch"])

    assert grafted.count("_LAUNCH_LO, _LAUNCH_HI = 0.25, 0.75") == 1, (
        "one statement binding two wanted names must be emitted once")
    assert validate_mode_reward_source(grafted, g) == []
    mod = _load(grafted, tmp_path, name="gr_tuple")
    assert mod._LAUNCH_HI == 0.75 and mod._LAUNCH_LO == 0.25
