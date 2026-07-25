"""tests/test_modes.py — stages as OGMP modes with transition guards.

The value of writing the automaton down is that its failure modes become
*statable*: an unreachable mode, a guard that can never fire, a transition to
a mode that does not exist. These tests pin those, and pin that a mode graph
derived from a composed reference lines up frame-for-frame with the seams the
composition actually produced.
"""
from __future__ import annotations

import numpy as np
import pytest

from sculptor.modes import (
    Guard,
    Mode,
    ModeError,
    ModeGraph,
    Transition,
    mode_at_frame,
    mode_phase_windows,
    modes_from_composition,
    validate_mode_graph,
)
from sculptor.refs.compose import compose_reference

FPS = 60.0
J = 4


def _clip(n: int = 120, *, joint_offset: float = 0.0) -> dict:
    t = np.arange(n, dtype=np.float64) / FPS
    jp = (joint_offset + 0.10 * np.sin(2 * np.pi * 0.5 * t)[:, None]
          + 0.01 * np.arange(J)[None, :])
    return {
        "fps": FPS,
        "joint_names": [f"joint_{i}" for i in range(J)],
        "root_pos_z": 0.70 + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        "root_pos_xy": np.stack([0.5 * t, np.zeros(n)], axis=1),
        "root_quat_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        "joint_pos": jp,
        "meta": {"clip_id": "synthetic"},
    }


def _graph(**kw) -> ModeGraph:
    modes = kw.pop("modes", (
        Mode("approach", (0, 100)),
        Mode("strike", (100, 200)),
    ))
    transitions = kw.pop("transitions", (
        Transition("approach", "strike", Guard("phase", at_phase=1.0)),
    ))
    return ModeGraph(modes=modes, transitions=transitions,
                     fps=kw.pop("fps", FPS), source=kw.pop("source", {}))


# ── validation ──────────────────────────────────────────────────────────
def test_a_well_formed_graph_validates():
    assert validate_mode_graph(_graph()) == []


def test_unreachable_mode_is_rejected():
    """The load-bearing check: a mode nobody can enter has its reward
    authored and gated but never paid — a silent dead end."""
    g = _graph(
        modes=(Mode("a", (0, 50)), Mode("b", (50, 100)), Mode("orphan", (100, 150))),
        transitions=(Transition("a", "b", Guard("phase", at_phase=1.0)),),
    )
    errors = validate_mode_graph(g)
    assert any("orphan" in e and "unreachable" in e for e in errors), errors


def test_transition_to_a_nonexistent_mode_is_rejected():
    g = _graph(transitions=(
        Transition("approach", "nowhere", Guard("phase", at_phase=1.0)),))
    assert any("nowhere" in e for e in validate_mode_graph(g))


def test_self_transition_is_rejected():
    g = _graph(transitions=(
        Transition("approach", "approach", Guard("phase", at_phase=1.0)),))
    assert any("self-transition" in e for e in validate_mode_graph(g))


def test_duplicate_mode_names_are_rejected():
    g = _graph(modes=(Mode("a", (0, 50)), Mode("a", (50, 100))))
    assert any("duplicate" in e for e in validate_mode_graph(g))


def test_empty_or_inverted_frame_range_is_rejected():
    for rng in ((50, 50), (80, 20)):
        g = _graph(modes=(Mode("a", (0, 50)), Mode("b", rng)))
        assert any("frame_range" in e for e in validate_mode_graph(g)), rng


def test_phase_guard_bounds_are_enforced():
    """at_phase=0 would fire before the mode does anything; >1 can never
    fire. Both make the transition meaningless."""
    for bad in (0.0, 1.5, -0.2):
        g = _graph(transitions=(
            Transition("approach", "strike", Guard("phase", at_phase=bad)),))
        assert any("at_phase" in e for e in validate_mode_graph(g)), bad


def test_phase_guard_without_a_phase_is_rejected():
    g = _graph(transitions=(
        Transition("approach", "strike", Guard("phase")),))
    assert any("requires 'at_phase'" in e for e in validate_mode_graph(g))


def test_predicate_guard_requires_an_expression():
    g = _graph(transitions=(
        Transition("approach", "strike", Guard("predicate", expression="  ")),))
    assert any("expression" in e for e in validate_mode_graph(g))


def test_unknown_guard_kind_is_rejected_not_ignored():
    """A guard nobody evaluates is worse than a missing one."""
    g = _graph(transitions=(
        Transition("approach", "strike", Guard("vibes", at_phase=1.0)),))
    assert any("unknown guard kind" in e for e in validate_mode_graph(g))


def test_validation_reports_every_violation_at_once():
    g = _graph(
        modes=(Mode("a", (0, 50)), Mode("a", (80, 20)), Mode("orphan", (100, 150))),
        transitions=(Transition("a", "ghost", Guard("phase", at_phase=9.0)),),
    )
    errors = validate_mode_graph(g)
    assert len(errors) >= 4, errors


# ── derivation from a composed reference ────────────────────────────────
def test_modes_derive_from_a_composition_one_per_segment():
    composed = compose_reference([
        {"clip": _clip(), "label": "approach", "source_id": "clip_a"},
        {"clip": _clip(joint_offset=0.05), "label": "launch", "source_id": "clip_b"},
        {"clip": _clip(joint_offset=0.02), "label": "strike", "source_id": "clip_c"},
    ])
    g = modes_from_composition(composed, clip_id="novel--g1")

    assert [m.name for m in g.modes] == ["approach", "launch", "strike"]
    assert [m.source_clip_id for m in g.modes] == ["clip_a", "clip_b", "clip_c"]
    assert [(t.from_mode, t.to_mode) for t in g.transitions] == [
        ("approach", "launch"), ("launch", "strike")]
    assert validate_mode_graph(g) == []


def test_mode_windows_tile_the_composed_clip_exactly():
    """No frame may be unowned or double-owned — a per-mode reward gated on
    these windows would otherwise pay twice or not at all."""
    composed = compose_reference([
        {"clip": _clip(), "label": "a"},
        {"clip": _clip(joint_offset=0.05), "label": "b"},
        {"clip": _clip(joint_offset=0.02), "label": "c"},
    ])
    g = modes_from_composition(composed)
    n = len(composed["root_pos_z"])

    assert g.modes[0].frame_range[0] == 0
    assert g.modes[-1].frame_range[1] == n
    for prev, nxt in zip(g.modes, g.modes[1:]):
        assert prev.frame_range[1] == nxt.frame_range[0]
    assert sum(m.n_frames for m in g.modes) == n


def test_mode_boundaries_are_the_composition_seams():
    composed = compose_reference([
        {"clip": _clip(), "label": "a"},
        {"clip": _clip(joint_offset=0.05), "label": "b"},
    ])
    seams = composed["meta"]["composition"]["seam_frames"]
    g = modes_from_composition(composed)
    assert g.modes[0].frame_range[1] == seams[0]
    assert g.modes[1].frame_range[0] == seams[0]


def test_duplicate_segment_labels_are_disambiguated():
    """Labels are free text from a user; modes must stay uniquely named."""
    composed = compose_reference([
        {"clip": _clip(), "label": "kick"},
        {"clip": _clip(joint_offset=0.05), "label": "kick"},
    ])
    g = modes_from_composition(composed)
    assert [m.name for m in g.modes] == ["kick", "kick_2"]
    assert validate_mode_graph(g) == []


def test_unlabeled_segments_get_positional_names():
    composed = compose_reference([
        {"clip": _clip()}, {"clip": _clip(joint_offset=0.05)},
    ])
    g = modes_from_composition(composed)
    assert all(m.name for m in g.modes)
    assert len(set(m.name for m in g.modes)) == 2


def test_single_clip_reference_is_refused_with_a_useful_message():
    with pytest.raises(ModeError, match="COMPOSED reference"):
        modes_from_composition(_clip())


def test_guard_phase_is_configurable():
    composed = compose_reference([
        {"clip": _clip(), "label": "a"},
        {"clip": _clip(joint_offset=0.05), "label": "b"},
    ])
    g = modes_from_composition(composed, guard_at_phase=0.8)
    assert g.transitions[0].guard.at_phase == pytest.approx(0.8)


# ── helpers ─────────────────────────────────────────────────────────────
def test_mode_at_frame_maps_frames_to_owners():
    g = _graph()
    assert mode_at_frame(g, 0).name == "approach"
    assert mode_at_frame(g, 99).name == "approach"
    assert mode_at_frame(g, 100).name == "strike"


def test_frames_past_the_end_stay_in_the_terminal_mode():
    """An episode running long is still IN the last mode, not outside the
    automaton."""
    assert mode_at_frame(_graph(), 10_000).name == "strike"


def test_phase_windows_are_seconds():
    w = mode_phase_windows(_graph())
    assert w["approach"] == (0.0, pytest.approx(100 / FPS, abs=1e-4))
    assert w["strike"] == (pytest.approx(100 / FPS, abs=1e-4),
                           pytest.approx(200 / FPS, abs=1e-4))


def test_graph_round_trips_to_a_serializable_dict():
    d = _graph().to_dict()
    import json
    assert json.loads(json.dumps(d))["modes"][0]["name"] == "approach"
    assert d["transitions"][0]["guard"]["kind"] == "phase"
    assert d["schema_version"] == 1
