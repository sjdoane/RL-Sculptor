"""tests/test_auto_physics.py — §7.4 auto-adjust physics prompt synthesis.

Unit tests for `sculptor.adapters.auto_physics`. No Claude calls; the
synthesizer is pure string formatting over the audit dict that §7.3
produces.
"""

from __future__ import annotations

import pytest

from sculptor.adapters.auto_physics import (
    _CANONICAL_PHYSICS_PAPERS,
    should_auto_adjust_physics,
    synthesize_auto_physics_prompt,
)


# ── Happy path ────────────────────────────────────────────────────────────
def test_synthesize_prompt_includes_all_audit_metrics() -> None:
    """Every headline metric from the audit must appear in the prompt."""
    audit = {
        "verdict": "severe",
        "torque_saturation_frac": 0.47,
        "any_joint_saturation_max": 0.92,
        "joint_vel_p99_max": 85.0,
        "joint_vel_multiplier_vs_nominal": 2.83,
        "joint_limit_violation_frac": 0.01,
        "top_joints_saturation": [
            {"name": "knee_pitch_left", "value": 0.92},
            {"name": "ankle_roll_right", "value": 0.74},
        ],
        "top_joints_vel": [{"name": "knee_pitch_left", "value": 85.0}],
    }
    out = synthesize_auto_physics_prompt(audit)
    # Verdict up-cased in header.
    assert "SEVERE" in out
    # Percentage metrics surfaced.
    assert "47.0%" in out or "47%" in out
    assert "92.0%" in out or "92%" in out
    # Joint names quoted.
    assert "knee_pitch_left" in out
    assert "ankle_roll_right" in out
    # Velocity number.
    assert "85" in out
    assert "2.83" in out or "2.8" in out
    # Mitigation actions mentioned.
    assert "forcerange" in out
    assert "armature" in out
    assert "damping" in out
    # Canonical citations.
    for arxiv_id, _desc in _CANONICAL_PHYSICS_PAPERS:
        assert f"arXiv:{arxiv_id}" in out


def test_synthesize_prompt_stays_within_physics_prompt_length_limit() -> None:
    """Physics-editor endpoint caps prompts at 2000 chars — the auto-
    synthesized prompt must leave headroom for user tweaks."""
    audit = {
        "verdict": "severe",
        "torque_saturation_frac": 0.5,
        "any_joint_saturation_max": 0.9,
        "joint_vel_p99_max": 100.0,
        "joint_vel_multiplier_vs_nominal": 3.3,
        "joint_limit_violation_frac": 0.1,
        "top_joints_saturation": [
            {"name": f"joint_{i}_very_long_name", "value": 0.9 - i * 0.05}
            for i in range(10)
        ],
        "top_joints_vel": [
            {"name": f"j{i}", "value": 100.0 - i} for i in range(10)
        ],
    }
    out = synthesize_auto_physics_prompt(audit)
    assert len(out) <= 1800, f"prompt too long: {len(out)} chars"


def test_synthesize_prompt_handles_missing_top_joints() -> None:
    """When joint lists are empty (pre-§7.1 iters, positional fallbacks),
    the prompt must still be coherent."""
    audit = {
        "verdict": "severe",
        "torque_saturation_frac": 0.6,
        "any_joint_saturation_max": 0.6,
        "joint_vel_p99_max": 50.0,
        "joint_limit_violation_frac": 0.0,
    }
    out = synthesize_auto_physics_prompt(audit)
    # Still produces output with the scalar metrics.
    assert len(out) > 200
    assert "60.0%" in out or "60%" in out
    # Graceful fallback for missing joint-level data.
    assert "no joint-level data" in out or "top saturated joints" in out


def test_synthesize_prompt_handles_n_a_metrics() -> None:
    """Audit dicts with missing numeric fields render as 'n/a' inline
    instead of crashing."""
    audit = {"verdict": "severe"}  # all metrics missing
    out = synthesize_auto_physics_prompt(audit)
    assert "n/a" in out
    # The scaffolding (mitigation actions, citations) must still land.
    assert "forcerange" in out
    assert "arXiv:" in out


# ── Feature-flag gate ─────────────────────────────────────────────────────
def test_should_auto_adjust_fires_on_severe_with_flag() -> None:
    assert should_auto_adjust_physics(
        {"verdict": "severe"}, auto_adjust_enabled=True
    ) is True


def test_should_auto_adjust_blocked_by_flag() -> None:
    """verdict=severe but flag=off → never fires. This is the default
    config behavior — users opt in per-project."""
    assert should_auto_adjust_physics(
        {"verdict": "severe"}, auto_adjust_enabled=False
    ) is False


def test_should_auto_adjust_does_not_fire_on_mild() -> None:
    """Mild is SUGGESTIVE only — the diagnoser already sees it via the
    §7.3 audit block. Auto-adjust is for clear-cut severe cases."""
    assert should_auto_adjust_physics(
        {"verdict": "mild"}, auto_adjust_enabled=True
    ) is False


def test_should_auto_adjust_does_not_fire_on_ok_or_unknown() -> None:
    for verdict in ("ok", "unknown", "", None):
        assert should_auto_adjust_physics(
            {"verdict": verdict}, auto_adjust_enabled=True
        ) is False, f"should not fire on verdict={verdict!r}"


def test_should_auto_adjust_handles_none_audit() -> None:
    """Audit failed (returned None or non-dict) → no auto-adjust."""
    assert should_auto_adjust_physics(None, auto_adjust_enabled=True) is False
    assert should_auto_adjust_physics(
        "not a dict",  # type: ignore[arg-type]
        auto_adjust_enabled=True,
    ) is False


# ── Verdict casing (audit may have uppercase from UI forwarding) ──────────
def test_should_auto_adjust_case_insensitive_verdict() -> None:
    """Some event forwarding paths upper-case the verdict string before
    it re-enters the audit dict. Stay robust to casing."""
    assert should_auto_adjust_physics(
        {"verdict": "SEVERE"}, auto_adjust_enabled=True
    ) is True
    assert should_auto_adjust_physics(
        {"verdict": "Severe"}, auto_adjust_enabled=True
    ) is True
