"""§2026-07-19 Prompt2Policy-informed guardrails: single-term reward
dominance + training-plateau screens over the per-component training
series, surfaced as GUARDRAIL lines in both diagnose prompts.

Advisory prompt context only — the tests pin that they render into the
TRAINING_FEEDBACK block and never gate anything.
"""
from __future__ import annotations

from sculptor.diagnose import (
    _PreliminaryModel,
    _build_grounded_user_content,
    _build_preliminary_user_content,
    _training_guardrails,
)


def test_dominance_triggers_above_ninety_percent() -> None:
    warnings = _training_guardrails({
        "forward_velocity": [10.0, 11.0, 12.0],
        "alive_bonus": [0.1, 0.1, 0.1],
        "posture": [0.2, 0.2, 0.2],
    })
    assert any("reward-term dominance" in w for w in warnings)
    assert any("'forward_velocity'" in w for w in warnings)


def test_no_dominance_when_balanced() -> None:
    warnings = _training_guardrails({
        "a": [1.0, 2.0, 3.0],
        "b": [1.5, 2.5, 4.0],
    })
    assert not any("dominance" in w for w in warnings)


def test_aux_signals_excluded_from_dominance() -> None:
    """__episode_length et al. are not reward terms — a huge aux series
    must not trigger (or mask) the dominance screen."""
    warnings = _training_guardrails({
        "__episode_length": [500.0, 500.0, 500.0],
        "a": [1.0, 1.0, 1.0],
        "b": [1.1, 1.2, 1.0],
    })
    assert not any("dominance" in w for w in warnings)


def test_plateau_triggers_on_flat_totals() -> None:
    warnings = _training_guardrails({
        "a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "b": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    })
    assert any("plateau" in w for w in warnings)


def test_no_plateau_when_improving() -> None:
    warnings = _training_guardrails({
        "a": [1.0, 2.0, 4.0, 8.0],
        "b": [1.0, 2.0, 4.0, 8.0],
    })
    assert not any("plateau" in w for w in warnings)


def test_short_series_and_junk_are_silent() -> None:
    assert _training_guardrails({}) == []
    assert _training_guardrails({"a": [1.0, 2.0]}) == []  # <4 windows, 1 term
    assert _training_guardrails({"a": "junk", "b": None}) == []


def test_guardrails_render_in_both_prompt_builders() -> None:
    feedback = {
        "forward_velocity": [10.0, 10.0, 10.0, 10.0],
        "alive_bonus": [0.1, 0.1, 0.1, 0.1],
    }
    prelim_kwargs = dict(
        behavior_goal="walk", reward_spec={}, metrics={}, behavior={},
        behavior_metric_names=[], contract_text="c", keyframes=[],
        training_feedback=feedback,
    )
    prelim_prompt = _build_preliminary_user_content(**prelim_kwargs)
    prelim_text = (
        "".join(part.get("text", "") for part in prelim_prompt
                if isinstance(part, dict))
        if isinstance(prelim_prompt, list) else prelim_prompt)
    assert "GUARDRAIL reward-term dominance" in prelim_text
    assert "GUARDRAIL training plateau" in prelim_text

    grounded_prompt = _build_grounded_user_content(
        behavior_goal="walk", reward_spec={}, metrics={}, behavior={},
        contract_text="c",
        preliminary=_PreliminaryModel(
            failure_modes=["none"], evidence="e", confidence=0.5),
        kg_context="",
        training_feedback=feedback,
    )
    assert "GUARDRAIL reward-term dominance" in grounded_prompt
    assert "GUARDRAIL training plateau" in grounded_prompt


def test_rendered_episode_note_states_percentile_or_arbitrary() -> None:
    from sculptor.diagnose import _rendered_episode_note

    note = _rendered_episode_note({"rendered_episode_percentile": 0.37})
    assert "37%" in note and "best" not in note
    assert "arbitrary draw" in _rendered_episode_note({})
    assert _rendered_episode_note({"rendered_episode_percentile": "junk"}) == ""
