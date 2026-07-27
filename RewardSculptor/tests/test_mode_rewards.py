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

import numpy as np
import pytest

from sculptor.mode_rewards import (
    BATCHED_FN_SUFFIX,
    MODE_COMPONENT_PREFIX,
    MODE_FN_PREFIX,
    authored_modes,
    generate_mode_reward_scaffold,
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
    whole function, docstring included, so the stub marker goes with it."""
    out = src
    if scalar:
        start = out.index(f"def {MODE_FN_PREFIX}{mode}(")
        end = out.index(f"def {MODE_FN_PREFIX}{mode}{BATCHED_FN_SUFFIX}(")
        out = out[:start] + (
            f"def {MODE_FN_PREFIX}{mode}(state, action, next_state, info):\n"
            f'    """{mode}: authored."""\n'
            "    del state, action, next_state, info\n"
            "    return 1.0, {}\n\n\n") + out[end:]
    if batched:
        start = out.index(f"def {MODE_FN_PREFIX}{mode}{BATCHED_FN_SUFFIX}(")
        tail = out.index("\n\ndef ", start)
        out = out[:start] + (
            f"def {MODE_FN_PREFIX}{mode}{BATCHED_FN_SUFFIX}"
            "(state, action, next_state, info, like):\n"
            f'    """{mode}: authored."""\n'
            "    del state, action, next_state, info\n"
            "    return like + 1.0, {}\n") + out[tail:]
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
    assert "1.0s to 2.0s" in p
    assert "approach" in p and "land" in p
    assert "running jump kick" in p and "drive off the back foot" in p


def test_the_prompt_tells_the_author_not_to_re_detect_phase():
    """Phase detection inside an authored body would duplicate — and could
    contradict — the derived dispatch."""
    p = mode_authoring_prompt(_graph(), "approach")
    assert "do not try to detect the phase" in p.lower()


def test_the_first_and_last_modes_have_only_one_neighbour():
    first = mode_authoring_prompt(_graph(), "approach")
    last = mode_authoring_prompt(_graph(), "land")
    assert "preceded by" not in first and "followed by" in first
    assert "preceded by" in last and "followed by" not in last


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


def test_the_prompt_asks_for_both_halves():
    """A prompt that asks for one body would produce exactly the half-authored
    mode `authored_modes` refuses to call done."""
    p = mode_authoring_prompt(_graph(), "launch")
    assert f"{MODE_FN_PREFIX}launch{BATCHED_FN_SUFFIX}" in p
    assert "num_envs" in p and "torch.zeros_like(like)" in p


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
