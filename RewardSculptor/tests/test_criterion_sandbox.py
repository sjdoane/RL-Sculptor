"""Independent process-boundary tests for authored success criteria."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest

from sculptor.eval import generated_metric
from sculptor import mission_runtime
from sculptor.mission_runtime import (
    CriterionEvalError,
    _evaluate_success_criterion,
)


def _namespace() -> dict:
    trajectory = {"height": np.asarray([0.6, 0.8], dtype=np.float64)}
    return {
        "metric": 0.75,
        "behavior": {"mean_return": 1.25},
        "components": {"hold": 0.9},
        "trajectory": trajectory,
        "info": trajectory,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "float": float,
        "int": int,
        "bool": bool,
    }


def _bypass_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the OS boundary independently of the semantic AST gate."""
    monkeypatch.setattr(
        mission_runtime, "_validate_criterion_ast",
        lambda _tree, *, namespace_keys: None,
    )


def test_isolated_criterion_preserves_numerical_namespace_semantics() -> None:
    criterion = (
        "metric > 0.7 and behavior['mean_return'] > 1.0 "
        "and components.get('hold', 0.0) > 0.8 "
        "and trajectory['height'].mean() > 0.65 "
        "and info['height'].max() == 0.8"
    )
    assert _evaluate_success_criterion(criterion, _namespace()) is True


def test_distinct_info_and_trajectory_dicts_remain_distinct() -> None:
    namespace = _namespace()
    namespace["info"] = {"height": np.asarray([0.1, 0.2])}
    assert _evaluate_success_criterion(
        "trajectory['height'].mean() > info['height'].mean()", namespace,
    ) is True


def test_present_but_unapproved_callable_is_rejected() -> None:
    namespace = _namespace()
    namespace["danger"] = lambda value: value
    with pytest.raises(CriterionEvalError, match="disallowed function 'danger'"):
        _evaluate_success_criterion("danger(metric)", namespace)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux seccomp test")
def test_ast_bypass_cannot_write_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_ast(monkeypatch)
    marker = tmp_path / "must_not_exist"
    with pytest.raises(CriterionEvalError, match="PermissionError"):
        _evaluate_success_criterion(
            f"open({str(marker)!r}, 'w') is not None", _namespace())
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux seccomp test")
def test_ast_bypass_cannot_create_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    _bypass_ast(monkeypatch)
    with pytest.raises(CriterionEvalError, match="PermissionError"):
        _evaluate_success_criterion(
            "__import__('socket').socket() is not None", _namespace())


def test_ast_bypass_cannot_read_parent_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_ast(monkeypatch)
    monkeypatch.setenv("REWARDSCULPTOR_CRITERION_SECRET", "parent-only")
    assert _evaluate_success_criterion(
        "__import__('os').environ.get('REWARDSCULPTOR_CRITERION_SECRET') is None",
        _namespace(),
    ) is True


def test_ast_bypass_infinite_expression_is_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_ast(monkeypatch)
    monkeypatch.setattr(generated_metric, "METRIC_CALL_TIMEOUT_SECONDS", 0.2)
    started = time.monotonic()
    with pytest.raises(CriterionEvalError, match="MetricSandboxTimeout"):
        _evaluate_success_criterion("sum(iter(int, 1))", _namespace())
    assert time.monotonic() - started < 2.0


def test_each_criterion_gets_fresh_interpreter_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_ast(monkeypatch)
    # Deliberately poison this worker's builtins. The expression itself is
    # false; what matters is that no later decision inherits the mutation.
    assert _evaluate_success_criterion(
        "setattr(__import__('builtins'), 'abs', lambda value: 0) or False",
        _namespace(),
    ) is False
    assert _evaluate_success_criterion("abs(-2) == 2", _namespace()) is True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux crash test")
def test_native_criterion_crash_isolated_and_next_decision_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_ast(monkeypatch)
    with pytest.raises(CriterionEvalError, match="MetricSandboxError"):
        _evaluate_success_criterion(
            "__import__('ctypes').string_at(0) is not None", _namespace())

    assert _evaluate_success_criterion("metric > 0.7", _namespace()) is True
