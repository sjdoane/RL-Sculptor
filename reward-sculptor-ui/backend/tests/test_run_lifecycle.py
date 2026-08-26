from __future__ import annotations

import json

import pytest

from backend.services.run_lifecycle import (
    RunLifecycleError,
    RunLifecycleSession,
    build_terminal_run_receipt,
    verified_lifecycle_completed_iterations,
    verify_terminal_run_receipt,
    write_terminal_run_receipt,
)


def _run_started(start: int, count: int) -> dict:
    return {
        "type": "run_started",
        "iterations": count,
        "start_iter": start,
        "end_iter": start + count,
    }


def _complete(session: RunLifecycleSession, index: int) -> None:
    session.observe_event({"type": "iter_started", "iter": index})
    session.observe_event({
        "type": "iter_completed",
        "iter": index,
        "primary_metric": 1.0,
    })


def test_full_nonreference_lifecycle_proves_exact_requested_plan():
    session = RunLifecycleSession(
        run_id="run-1", expected_iterations=(7, 8, 9)
    )
    session.observe_event(_run_started(7, 3))
    for index in (7, 8, 9):
        _complete(session, index)
    proof = session.finalize_proof()
    assert proof["authority"] == "worker_iteration_lifecycle_verified"
    assert proof["iteration_plan"]["requested"] == [7, 8, 9]
    assert proof["iteration_plan"]["completed"] == [7, 8, 9]
    assert len(proof["proof_sha256"]) == 64
    assert verified_lifecycle_completed_iterations(
        proof, run_id="run-1",
    ) == (7, 8, 9)
    assert verified_lifecycle_completed_iterations(
        proof, run_id="a-different-run",
    ) is None

    tampered = json.loads(json.dumps(proof))
    tampered["iteration_plan"]["completed"] = [7, 8]
    assert verified_lifecycle_completed_iterations(
        tampered, run_id="run-1",
    ) is None


def test_rc_zero_equivalent_without_completed_plan_is_rejected():
    session = RunLifecycleSession(
        run_id="run-2", expected_iterations=(4, 5)
    )
    session.observe_event(_run_started(4, 2))
    _complete(session, 4)
    with pytest.raises(RunLifecycleError, match="requested plan"):
        session.finalize_proof()


def test_authorized_early_stop_proves_prefix_only():
    session = RunLifecycleSession(
        run_id="run-3",
        expected_iterations=(10, 11, 12),
        allowed_early_stop_sources=("fitness",),
    )
    session.observe_event(_run_started(10, 3))
    _complete(session, 10)
    session.observe_event({
        "type": "early_stop",
        "at_iter": 10,
        "source": "fitness",
        "reason": "target reached",
    })
    proof = session.finalize_proof()
    assert proof["iteration_plan"]["completed"] == [10]
    assert proof["iteration_plan"]["early_stop"]["source"] == "fitness"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "iter_started", "iter": 8},
        {"type": "iter_completed", "iter": 7},
        {
            "type": "early_stop",
            "at_iter": 7,
            "source": "user",
            "reason": "stop",
        },
    ],
)
def test_out_of_order_or_unauthorized_events_are_fatal(event):
    session = RunLifecycleSession(
        run_id="run-4",
        expected_iterations=(7, 8),
        allowed_early_stop_sources=("fitness",),
    )
    session.observe_event(_run_started(7, 2))
    if event["type"] == "early_stop":
        _complete(session, 7)
    with pytest.raises(RunLifecycleError):
        session.observe_event(event)
    with pytest.raises(RunLifecycleError, match="rejected evidence"):
        session.finalize_proof()


def test_worker_cannot_redefine_prespawn_plan():
    session = RunLifecycleSession(
        run_id="run-5", expected_iterations=(3, 4)
    )
    with pytest.raises(RunLifecycleError, match="pre-spawn"):
        session.observe_event(_run_started(4, 2))


def test_boolean_iteration_fields_are_not_integers():
    session = RunLifecycleSession(
        run_id="run-6", expected_iterations=(0,)
    )
    with pytest.raises(RunLifecycleError, match="pre-spawn"):
        session.observe_event({
            "type": "run_started",
            "iterations": True,
            "start_iter": 0,
            "end_iter": 1,
        })


def test_user_stop_requires_exact_sidecar_authorization():
    session = RunLifecycleSession(
        run_id="job_0123456789abcdef", expected_iterations=(0, 1)
    )
    session.observe_event(_run_started(0, 2))
    _complete(session, 0)
    session.authorize_user_stop({
        "schema": 1,
        "authority": "server_control_sidecar_stop",
        "run_id": "job_0123456789abcdef",
        "control_file": "_control_job_0123456789abcdef.json",
        "control_sha256": "a" * 64,
        "control_bytes": 123,
        "resume_token": 2,
        "stop": True,
    })
    session.observe_event({
        "type": "early_stop",
        "at_iter": 0,
        "source": "user",
        "reason": "stopped by user (interactive)",
    })

    proof = session.finalize_proof()
    plan = proof["iteration_plan"]
    assert plan["completed"] == [0]
    assert plan["allowed_early_stop_sources"] == ["user"]
    assert plan["user_stop_authorization"]["stop"] is True


def test_user_stop_authorization_cannot_be_forged_with_stop_false():
    session = RunLifecycleSession(
        run_id="job_0123456789abcdef", expected_iterations=(0, 1)
    )
    with pytest.raises(RunLifecycleError, match="control-sidecar"):
        session.authorize_user_stop({
            "schema": 1,
            "authority": "server_control_sidecar_stop",
            "run_id": "job_0123456789abcdef",
            "control_file": "_control_job_0123456789abcdef.json",
            "control_sha256": "a" * 64,
            "control_bytes": 123,
            "resume_token": 0,
            "stop": False,
        })


def test_terminal_receipt_tampering_fails_closed(tmp_path, monkeypatch):
    from sculptor import run_manifests

    session = RunLifecycleSession(
        run_id="job_0123456789abcdef", expected_iterations=(3,)
    )
    session.observe_event(_run_started(3, 1))
    _complete(session, 3)
    iteration_receipt = {
        "schema": 3,
        "iter_index": 3,
        "marker_sha256": "b" * 64,
    }
    terminal = build_terminal_run_receipt(
        project_slug="project",
        lifecycle_proof=session.finalize_proof(),
        iteration_receipts=[iteration_receipt],
        started_at="2026-08-25T10:00:00+00:00",
        completed_at="2026-08-25T10:01:00+00:00",
    )
    path = write_terminal_run_receipt(tmp_path, terminal)
    monkeypatch.setattr(
        run_manifests,
        "verify_iteration_completion_marker",
        lambda _path: dict(iteration_receipt),
    )
    assert verify_terminal_run_receipt(
        tmp_path, path, project_slug="project",
    ) == terminal

    tampered = dict(terminal)
    tampered["status"] = "errored"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert verify_terminal_run_receipt(
        tmp_path, path, project_slug="project",
    ) is None
