from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def _event_line(event: dict) -> bytes:
    return (
        "[SCULPT-EVENT] " + json.dumps(event, sort_keys=True) + "\n"
    ).encode()


class _Process:
    pid = 42001
    returncode: int | None = None

    def __init__(self, events: list[dict]) -> None:
        self.stdout = asyncio.StreamReader()
        for event in events:
            self.stdout.feed_data(_event_line(event))
        self.stdout.feed_eof()

    async def wait(self) -> int:
        await asyncio.sleep(0.02)
        self.returncode = 0
        return 0

    def send_signal(self, _signal) -> None:
        return None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    rewards = project / "rewards"
    rewards.mkdir(parents=True)
    (rewards / "v0.py").write_text("reward = 1\n", encoding="utf-8")
    return project


def _job(project: Path):
    from backend.services.job_manager import Job

    job = Job(
        job_id="lifecycle-test",
        kind="sculpt_run",
        project_slug=project.name,
        status="running",
    )
    job._cancel = asyncio.Event()
    return job


def _stub_iteration_and_terminal_receipts(monkeypatch) -> None:
    """Keep manager tests CPU-only while exercising the durable boundary."""
    from backend.services import iteration_completion, run_lifecycle

    monkeypatch.setattr(
        iteration_completion,
        "attested_completion_receipt",
        lambda path: {
            "schema": 3,
            "iter_index": int(Path(path).name.removeprefix("iter_")),
            "marker_sha256": (
                f"{int(Path(path).name.removeprefix('iter_')) + 1:064x}"
            ),
        },
    )

    def _verify_written_receipt(
        _project_dir, receipt_path, *, project_slug,
    ):
        assert project_slug
        return json.loads(Path(receipt_path).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        run_lifecycle,
        "verify_terminal_run_receipt",
        _verify_written_receipt,
    )


def test_run_manager_completes_from_events_not_metric_history(
    tmp_path, monkeypatch,
):
    from backend.services import run_manager

    project = _project(tmp_path)
    events = [
        {
            "type": "run_started",
            "iterations": 2,
            "start_iter": 0,
            "end_iter": 2,
        },
        {"type": "iter_started", "iter": 0},
        {"type": "iter_completed", "iter": 0},
        {"type": "iter_started", "iter": 1},
        {"type": "iter_completed", "iter": 1},
    ]

    async def _fake_exec(*_args, **_kwargs):
        return _Process(events)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    _stub_iteration_and_terminal_receipts(monkeypatch)
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "unit lifecycle",
            "iterations": 2,
            "dry_run": True,
            "no_kg": True,
        },
    )
    job = _job(project)
    result = asyncio.run(runner(job, job._cancel))

    assert result["iterations_run"] == 2
    assert result["primary_metric_history"] == []
    assert job.params["run_lifecycle_proof"]["iteration_plan"] == {
        "requested": [0, 1],
        "completed": [0, 1],
        "allowed_early_stop_sources": [],
        "early_stop": None,
    }
    assert any(
        event.get("type") == "run_completed" for event in job.events
    )
    terminal_path = Path(job.params["run_terminal_receipt_path"])
    assert terminal_path.is_file()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "completed"
    assert terminal["lifecycle_proof"]["iteration_plan"]["completed"] == [
        0, 1,
    ]


def test_run_manager_rejects_rc_zero_with_truncated_events(
    tmp_path, monkeypatch,
):
    from backend.services import run_manager

    project = _project(tmp_path)
    events = [
        {
            "type": "run_started",
            "iterations": 2,
            "start_iter": 0,
            "end_iter": 2,
        },
        {"type": "iter_started", "iter": 0},
        {"type": "iter_completed", "iter": 0},
    ]

    async def _fake_exec(*_args, **_kwargs):
        return _Process(events)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "truncated lifecycle",
            "iterations": 2,
            "dry_run": True,
            "no_kg": True,
        },
    )
    job = _job(project)
    result = asyncio.run(runner(job, job._cancel))

    assert result["return_code"] == 0
    assert job.status == "errored"
    assert job.params["error_classification"]["kind"] == (
        "run_lifecycle_unproven"
    )
    assert not any(
        event.get("type") == "run_completed" for event in job.events
    )


def test_run_manager_defense_rejects_reference_without_lineage_before_spawn(
    tmp_path, monkeypatch,
):
    from backend.services import run_manager

    project = _project(tmp_path)
    spawned = False

    async def _fake_exec(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "invalid reference ablation",
            "iterations": 1,
            "dry_run": True,
            "no_kg": True,
            "reference_clip_id": "clip",
            "reference_robot": "g1",
        },
    )
    job = _job(project)
    with pytest.raises(RuntimeError, match="cannot disable lineage"):
        asyncio.run(runner(job, job._cancel))
    assert spawned is False


def test_run_manager_accepts_user_stop_only_from_reread_control_sidecar(
    tmp_path, monkeypatch,
):
    from backend.services import run_manager

    project = _project(tmp_path)
    events = [
        {
            "type": "run_started",
            "iterations": 2,
            "start_iter": 0,
            "end_iter": 2,
        },
        {"type": "iter_started", "iter": 0},
        {"type": "iter_completed", "iter": 0},
        {
            "type": "early_stop",
            "at_iter": 0,
            "source": "user",
            "reason": "stopped by user (interactive)",
        },
    ]

    async def _fake_exec(*_args, **_kwargs):
        return _Process(events)

    original_write = run_manager.write_control_file

    def _write_authorized_stop(path, data):
        original_write(path, {**data, "stop": True, "resume_token": 1})

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        run_manager, "write_control_file", _write_authorized_stop,
    )
    _stub_iteration_and_terminal_receipts(monkeypatch)
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "authorized interactive stop",
            "iterations": 2,
            "dry_run": True,
            "no_kg": True,
        },
    )
    job = _job(project)
    result = asyncio.run(runner(job, job._cancel))

    assert result["iterations_run"] == 1
    assert job.status == "running"
    proof = job.params["run_lifecycle_proof"]
    assert proof["iteration_plan"]["early_stop"]["source"] == "user"
    assert proof["iteration_plan"]["user_stop_authorization"]["stop"] is True
    assert any(
        event.get("type") == "user_stop_authorization_verified"
        for event in job.events
    )


def test_run_manager_rejects_forged_user_stop_when_sidecar_is_false(
    tmp_path, monkeypatch,
):
    from backend.services import run_manager

    project = _project(tmp_path)
    events = [
        {
            "type": "run_started",
            "iterations": 2,
            "start_iter": 0,
            "end_iter": 2,
        },
        {"type": "iter_started", "iter": 0},
        {"type": "iter_completed", "iter": 0},
        {
            "type": "early_stop",
            "at_iter": 0,
            "source": "user",
            "reason": "forged worker event",
        },
    ]

    async def _fake_exec(*_args, **_kwargs):
        return _Process(events)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "reject an unauthorized stop",
            "iterations": 2,
            "dry_run": True,
            "no_kg": True,
        },
    )
    job = _job(project)
    result = asyncio.run(runner(job, job._cancel))

    assert result["return_code"] == 0
    assert job.status == "errored"
    assert job.params["error_classification"]["kind"] == (
        "run_lifecycle_unproven"
    )
    assert any(
        event.get("type") == "user_stop_authorization_rejected"
        for event in job.events
    )


@pytest.mark.parametrize(
    "failure_stage", ["attestation", "write", "reverification"],
)
def test_run_manager_requires_durable_verified_terminal_receipt(
    tmp_path, monkeypatch, failure_stage,
):
    from backend.services import (
        iteration_completion,
        run_lifecycle,
        run_manager,
    )

    project = _project(tmp_path)
    events = [
        {
            "type": "run_started",
            "iterations": 1,
            "start_iter": 0,
            "end_iter": 1,
        },
        {"type": "iter_started", "iter": 0},
        {"type": "iter_completed", "iter": 0},
    ]

    async def _fake_exec(*_args, **_kwargs):
        return _Process(events)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    _stub_iteration_and_terminal_receipts(monkeypatch)
    if failure_stage == "attestation":
        monkeypatch.setattr(
            iteration_completion,
            "attested_completion_receipt",
            lambda _path: None,
        )
    elif failure_stage == "write":
        monkeypatch.setattr(
            run_lifecycle,
            "write_terminal_run_receipt",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("terminal receipt disk unavailable")
            ),
        )
    else:
        monkeypatch.setattr(
            run_lifecycle,
            "verify_terminal_run_receipt",
            lambda *_args, **_kwargs: None,
        )

    runner = run_manager.run_sculpt_job(
        project_dir=project,
        run_params={
            "behavior_goal": "durable terminal receipt required",
            "iterations": 1,
            "dry_run": True,
            "no_kg": True,
        },
    )
    job = _job(project)
    result = asyncio.run(runner(job, job._cancel))

    assert result["return_code"] == 0
    assert job.status == "errored"
    assert job.params["error_classification"]["kind"] == (
        "run_terminal_receipt_unproven"
    )
    assert any(
        event.get("type") == "run_terminal_receipt_unavailable"
        for event in job.events
    )
    assert not any(
        event.get("type") == "run_completed" for event in job.events
    )
