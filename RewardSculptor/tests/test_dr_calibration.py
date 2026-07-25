"""tests/test_dr_calibration.py — task-matched domain randomization.

The point of `dr_calibration` is that a robustness push should be sized to the
behavior rather than to a constant: the same push that barely perturbs a run
knocks over a crouch. These pin the scaling relationship, the clamps that keep
a degenerate clip from producing an unsurvivable push, and the task-typing.
"""
from __future__ import annotations

import numpy as np
import pytest

from sculptor.refs.dr_calibration import (
    DEFAULT_PUSH_MOMENTUM_FRACTION,
    MAX_PUSH_MPS,
    MIN_PUSH_MPS,
    characteristic_speed,
    dr_profile_for_task,
    push_events_for_behavior,
)

FPS = 60.0


def _clip(speed_mps: float, *, n: int = 120, with_xy: bool = True) -> dict:
    """A clip translating at a constant `speed_mps` along +X."""
    t = np.arange(n, dtype=np.float64) / FPS
    clip = {
        "fps": FPS,
        "root_pos_z": np.full(n, 0.70),
        "joint_names": ["j0"],
        "joint_pos": np.zeros((n, 1)),
    }
    if with_xy:
        clip["root_pos_xy"] = np.stack([speed_mps * t, np.zeros(n)], axis=1)
    else:
        clip["root_pos_z"] = 0.70 + speed_mps * t
    return clip


# ── characteristic speed ────────────────────────────────────────────────
def test_characteristic_speed_recovers_a_known_translation():
    assert characteristic_speed(_clip(1.8)) == pytest.approx(1.8, abs=0.05)


def test_characteristic_speed_uses_height_when_there_is_no_xy():
    """A squat/jump clip may carry only root height; it must not read as
    motionless."""
    assert characteristic_speed(_clip(0.9, with_xy=False)) == pytest.approx(
        0.9, abs=0.05)


def test_characteristic_speed_ignores_a_single_spike():
    """A retargeting spike must not set the push magnitude for the whole
    behavior — that is why this is a percentile, not a max."""
    clip = _clip(1.0)
    clip["root_pos_xy"][60, 0] += 5.0     # one-frame glitch
    assert characteristic_speed(clip) < 2.0


# ── push sizing ─────────────────────────────────────────────────────────
def test_push_is_a_fraction_of_the_behavior_speed():
    """Mass cancels (push and behavior act on the same body), so the momentum
    ratio is a pure velocity ratio."""
    out = push_events_for_behavior(_clip(2.0), fraction=0.5)
    assert out["linear_mps"] == pytest.approx(1.0, abs=0.05)
    assert out["enabled"] is True


def test_push_scales_with_the_behavior_not_a_constant():
    """The whole point: a fast behavior gets a bigger push than a slow one."""
    slow = push_events_for_behavior(_clip(0.6))["linear_mps"]
    fast = push_events_for_behavior(_clip(2.4))["linear_mps"]
    assert fast > slow * 3


def test_push_floors_for_a_near_stationary_behavior():
    """Balance/hold behaviors would otherwise get a zero push and no
    robustness signal at all."""
    out = push_events_for_behavior(_clip(0.0))
    assert out["linear_mps"] == pytest.approx(MIN_PUSH_MPS)
    assert out["provenance"]["clamped"] is True


def test_push_is_railed_against_a_corrupt_clip():
    out = push_events_for_behavior(_clip(50.0))
    assert out["linear_mps"] == pytest.approx(MAX_PUSH_MPS)
    assert out["provenance"]["clamped"] is True


def test_push_records_the_fraction_it_used():
    """0.5 is a practitioner heuristic, not a published constant, so a run has
    to be able to say which value produced its numbers."""
    out = push_events_for_behavior(_clip(2.0), fraction=0.25)
    prov = out["provenance"]
    assert prov["fraction"] == 0.25
    assert prov["characteristic_speed_mps"] == pytest.approx(2.0, abs=0.05)
    assert "not a published constant" in prov["caveat"]
    assert DEFAULT_PUSH_MOMENTUM_FRACTION == 0.5


def test_zero_fraction_is_refused_rather_than_silently_disabling():
    with pytest.raises(ValueError, match="enabled = false"):
        push_events_for_behavior(_clip(1.0), fraction=0.0)


def test_push_block_matches_the_env_spec_schema():
    """The result must validate as a real env_spec push_events block, or it is
    just a dict that looks right."""
    from sculptor.env_spec import validate_env_spec

    push = push_events_for_behavior(_clip(1.5))
    push.pop("provenance")          # provenance is ours, not the schema's
    spec = {
        "env_spec_version": 1,
        "shared": {},
        "train": {"push_events": push},
    }
    assert validate_env_spec(spec) == []


# ── task typing ─────────────────────────────────────────────────────────
def test_locomotion_randomizes_the_surface():
    p = dr_profile_for_task("locomotion")
    assert "friction_range" in p
    assert "body_friction_range" in p


def test_carry_randomizes_inertia_not_terrain():
    p = dr_profile_for_task("carry")
    assert "body_mass_scale_range" in p
    assert "friction_range" not in p


def test_every_task_gets_the_core_axes():
    """Mass and motor strength are uncertain on real hardware whatever the
    task is."""
    for task in ("locomotion", "carry", "jump", "getup", "balance",
                 "manipulation"):
        p = dr_profile_for_task(task)
        assert "body_mass_scale_range" in p, task
        assert "motor_strength_scale_range" in p, task


def test_unknown_task_gets_conservative_axes_not_none_and_not_everything():
    p = dr_profile_for_task("interpretive dance")
    assert p["dr_profile"]["recognized"] is False
    assert "body_mass_scale_range" in p
    assert "friction_range" not in p


def test_mass_range_follows_the_recommended_window():
    assert dr_profile_for_task("locomotion")["body_mass_scale_range"] == [0.75, 1.5]


def test_profile_includes_a_matched_push_when_a_clip_is_given():
    p = dr_profile_for_task("locomotion", clip=_clip(2.0))
    assert p["push_events"]["linear_mps"] == pytest.approx(1.0, abs=0.05)
    assert p["dr_profile"]["task_type"] == "locomotion"


def test_profile_omits_pushes_without_a_clip():
    """No reference means no measured behavior speed, so there is nothing to
    size a push against — better absent than guessed."""
    assert "push_events" not in dr_profile_for_task("locomotion")


def test_profile_axes_are_all_real_env_spec_keys():
    """A profile key that the env spec does not recognize would be silently
    dropped at apply time."""
    from sculptor.env_spec import validate_env_spec

    p = dr_profile_for_task("locomotion", clip=_clip(1.2))
    p["push_events"].pop("provenance")
    p.pop("dr_profile")
    spec = {"env_spec_version": 1, "shared": {}, "train": p}
    assert validate_env_spec(spec) == []
