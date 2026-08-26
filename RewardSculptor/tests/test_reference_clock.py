from types import SimpleNamespace

import pytest

from sculptor.reference_clock import (
    build_reference_clock,
    reference_clock_from_module,
)


def _clock() -> dict:
    return build_reference_clock(
        clip_id="motion",
        robot="g1",
        target_sha256="a" * 64,
        phase_mode="hold",
        phase_duration_s=2.0,
        n_phase_targets=32,
    )


def test_legacy_module_without_reward_spec_has_no_reference_clock() -> None:
    assert reference_clock_from_module(SimpleNamespace()) is None


def test_declared_reference_requires_executable_clock_surface() -> None:
    module = SimpleNamespace(REWARD_SPEC={
        "reference_tracking": True,
        "reference_clock": _clock(),
    })
    with pytest.raises(ValueError, match="reference_clock_batched"):
        reference_clock_from_module(module)


def test_non_reference_reward_spec_remains_supported() -> None:
    assert reference_clock_from_module(
        SimpleNamespace(REWARD_SPEC={"supports_batched": True})
    ) is None
