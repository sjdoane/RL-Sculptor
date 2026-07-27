"""tests/test_timing.py — physics vs control timestep policy.

These two rates were implicit until a Tier-D certification went wrong because
of them twice. The point of `sculptor.refs.timing` is that they are now stated
once, derived one way, and checkable — so these pin the derivation, the
literature band, and the checks that would have caught the real failures.
"""
from __future__ import annotations

from sculptor.refs.timing import (
    CONVENTIONAL_CONTROL_HZ,
    MJLAB_G1_VELOCITY,
    SEA_MIN_PHYSICS_HZ,
    SimTiming,
    validate_timing,
)


# ── derivation ──────────────────────────────────────────────────────────
def test_control_rate_is_physics_rate_divided_by_decimation():
    t = SimTiming(physics_dt=0.005, decimation=4)
    assert t.physics_hz == 200.0
    assert t.control_dt == 0.02
    assert t.control_hz == 50.0


def test_the_g1_task_constant_matches_mjlab_and_the_convention():
    """Read from mjlab/tasks/velocity/velocity_env_cfg.py, not inferred from a
    training statistic — inferring it is what produced a spurious 25 Hz."""
    assert MJLAB_G1_VELOCITY.physics_dt == 0.005
    assert MJLAB_G1_VELOCITY.decimation == 4
    assert MJLAB_G1_VELOCITY.control_hz == CONVENTIONAL_CONTROL_HZ == 50.0


def test_steps_for_is_the_number_a_phase_clock_must_agree_with():
    """3.70 s of reference at 50 Hz is 185 control steps — the value the
    tracking reward's clock and the episode cap both have to use."""
    assert MJLAB_G1_VELOCITY.steps_for(3.70) == 185
    assert MJLAB_G1_VELOCITY.steps_for(20.0) == 1000
    assert MJLAB_G1_VELOCITY.steps_for(0.0) == 1      # never zero-length


# ── the checks that would have caught the real failures ─────────────────
def test_the_shipped_g1_timing_is_clean():
    assert validate_timing(MJLAB_G1_VELOCITY) == []


def test_too_few_control_steps_for_the_phase_targets_is_flagged():
    """The concrete tracking failure mode: more keyframes than actions, so
    some targets are never visited."""
    findings = validate_timing(
        MJLAB_G1_VELOCITY, reference_duration_s=0.2, n_phase_targets=32)
    assert any("never" in f for f in findings)


def test_the_real_composite_has_enough_control_steps():
    assert validate_timing(
        MJLAB_G1_VELOCITY, reference_duration_s=3.70, n_phase_targets=32) == []


def test_a_reference_faster_than_nyquist_is_flagged():
    slow = SimTiming(physics_dt=0.005, decimation=20)      # 10 Hz control
    findings = validate_timing(slow, reference_fps=120.0)
    assert any("Nyquist" in f for f in findings)


def test_a_coarse_physics_step_is_flagged():
    coarse = SimTiming(physics_dt=0.02, decimation=1)      # 50 Hz physics
    assert any("contact resolution" in f for f in validate_timing(coarse))


def test_control_rate_outside_the_deployable_band_is_flagged():
    too_fast = SimTiming(physics_dt=0.001, decimation=1)   # 1 kHz control
    assert any("deployable" in f for f in validate_timing(too_fast))


# ── series elasticity moves the floor ───────────────────────────────────
def test_series_elastic_actuators_demand_a_finer_physics_step():
    """A rigid-actuator G1 at 200 Hz says nothing about an SEA model of the
    same robot — the spring adds a fast mode."""
    findings = validate_timing(MJLAB_G1_VELOCITY, series_elastic=True)
    assert any("series-elastic" in f for f in findings)
    # ...and the same rate is fine without series elasticity.
    assert validate_timing(MJLAB_G1_VELOCITY, series_elastic=False) == []


def test_a_fine_enough_step_satisfies_the_sea_check():
    sea_ok = SimTiming(physics_dt=1.0 / SEA_MIN_PHYSICS_HZ, decimation=20)
    assert not any(
        "series-elastic" in f for f in validate_timing(sea_ok, series_elastic=True))


# ── degenerate input ────────────────────────────────────────────────────
def test_invalid_timing_is_reported_not_raised():
    assert validate_timing(SimTiming(physics_dt=0.0, decimation=4))
    assert validate_timing(SimTiming(physics_dt=0.005, decimation=0))


def test_to_dict_is_serialisable_for_a_run_record():
    d = MJLAB_G1_VELOCITY.to_dict()
    assert d["control_hz"] == 50.0 and d["physics_hz"] == 200.0
    assert d["decimation"] == 4
