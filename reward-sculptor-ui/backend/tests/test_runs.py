"""Tests for run lifecycle + WS stream.

The live sculpt subprocess is fast (dry-run) but still takes ~30s for 3
iters and hits the actual adapter code. These tests instead monkey-
patch `run_sculpt_job` with a fake that:
  - prints a few `[SCULPT-EVENT]` markers to stdout (mirrors the real
    sculpt CLI's additive format);
  - writes a `reports/metric_history.json` so the filesystem watcher's
    heartbeat populates job.params;
  - emits `run_completed`.

That's enough to cover the event plumbing end-to-end without ~50s
subprocess startup.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project_with_library(
    client: TestClient, name: str = "Runs"
) -> str:
    r = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 3, "behavior_goal": "hop"},
    )
    slug = r.json()["slug"]
    r = client.post(
        f"/projects/{slug}/robot/library",
        json={"robot_name": "hopper"},
    )
    assert r.status_code == 200
    return slug


# ── fake sculpt job factory ──────────────────────────────────────────
def _fake_sculpt_factory(project_dir: Path):
    """Returns a `JobManager.submit`-compatible callable that emits a
    realistic event sequence without spawning a subprocess."""
    from backend.services.job_manager import Job

    async def _runner(job: Job, cancel: asyncio.Event):
        job.emit({
            "type": "log_line",
            "text": "[sculpt] project=" + str(project_dir),
        })
        job.emit({
            "type": "run_started",
            "source": "stdout",
            "project": str(project_dir),
            "iterations": int(job.params.get("iterations") or 2),
            "behavior_goal": job.params.get("behavior_goal") or "",
        })
        for i in range(int(job.params.get("iterations") or 2)):
            if cancel.is_set():
                break
            job.emit({
                "type": "iter_started",
                "source": "fs",
                "iter": i,
                "reward_version_before": i,
            })
            # Simulate rollout + diagnosis + edit_applied.
            job.emit({
                "type": "rollout_done",
                "source": "fs",
                "iter": i,
                "size_bytes": 12345,
            })
            job.emit({
                "type": "diagnosed",
                "source": "fs",
                "iter": i,
                "failure_modes": ["component_imbalance"],
                "confidence": 0.6,
                "n_edits": 1,
            })
            job.emit({
                "type": "edit_applied",
                "source": "fs",
                "iter": i,
                "reward_version_before": i,
                "reward_version_after": i + 1,
                "paper_refs": ["1707.06347"] if i == 0 else [],
            })
            if i == 0:
                job.emit({
                    "type": "citation_added",
                    "source": "fs",
                    "iter": 0,
                    "reward_version": 1,
                    "arxiv_id": "1707.06347",
                })
            metric = 10.0 + i * 2.5
            job.emit({
                "type": "iter_completed",
                "source": "stdout",
                "iter": i,
                "primary_metric": metric,
                "metric_delta": 2.5 if i > 0 else None,
                "failure_modes": ["component_imbalance"],
                "edit_count": 1,
                "reward_version_after": i + 1,
            })
            job.emit({"type": "log_line", "text": f"[sculpt] iter {i} done"})
            await asyncio.sleep(0.01)

        job.emit({
            "type": "run_completed",
            "return_code": 0,
            "iterations_run": int(job.params.get("iterations") or 2),
            "primary_metric_history": [
                10.0 + i * 2.5
                for i in range(int(job.params.get("iterations") or 2))
            ],
        })
        return {"return_code": 0, "iterations_run": int(job.params.get("iterations") or 2)}

    return _runner


@pytest.fixture
def fake_sculpt(monkeypatch: pytest.MonkeyPatch):
    from backend.services import run_manager

    def _factory(*, project_dir: Path, run_params: dict):
        return _fake_sculpt_factory(project_dir)

    monkeypatch.setattr(run_manager, "run_sculpt_job", _factory)
    # The route imports the symbol at import time; re-patch the bound
    # reference on the routes module too.
    from backend.routes import runs as runs_routes
    monkeypatch.setattr(runs_routes, "run_sculpt_job", _factory)


# ── POST + GET ────────────────────────────────────────────────────────
def test_launch_run_returns_summary(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    # Simulate ANTHROPIC_API_KEY is set so live runs aren't blocked.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    slug = _make_project_with_library(client, "RunLaunch")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 2, "dry_run": False},
    )
    assert r.status_code == 202, r.text
    summary = r.json()
    assert summary["project_slug"] == slug
    assert summary["iterations_requested"] == 2

    # Wait for completion.
    run_id = summary["run_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/projects/{slug}/runs/{run_id}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)
    assert body["status"] == "completed", body
    assert body["iterations_completed"] == 2
    assert body["iterations"][0]["reward_version_after"] == 1
    assert body["iterations"][1]["primary_metric"] == 12.5
    # Citations attached to iter 0's edit.
    assert "1707.06347" in body["iterations"][0]["paper_refs"]


def test_cannot_launch_two_concurrent_runs(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunConcurrent")
    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    # Inject a synthetic "running" sculpt_run so the 2nd POST is refused.
    jm._jobs["job_fake_run"] = Job(
        job_id="job_fake_run",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 2, "dry_run": False},
    )
    assert r.status_code == 409
    assert r.json()["type"] == "/problems/job-busy"


def test_live_run_blocked_without_api_key(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    slug = _make_project_with_library(client, "RunNoKey")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 2, "dry_run": False},
    )
    assert r.status_code == 412
    assert r.json()["type"] == "/problems/no-api-key"


def test_dry_run_does_not_require_api_key(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    slug = _make_project_with_library(client, "RunDry")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 1, "dry_run": True},
    )
    assert r.status_code == 202


def test_kill_run_transitions_to_stopped(
    client: TestClient, tmp_projects_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    # Use a slower fake so we have time to kill it.
    async def _slow_runner(job, cancel):
        job.emit({"type": "log_line", "text": "started"})
        for _ in range(20):
            if cancel.is_set():
                job.emit({"type": "run_stopped", "source": "cancel"})
                return {"stopped": True}
            await asyncio.sleep(0.1)
        job.emit({"type": "run_completed", "return_code": 0})
        return {"return_code": 0}

    from backend.services import run_manager
    from backend.routes import runs as runs_routes
    factory = lambda *, project_dir, run_params: _slow_runner  # noqa: E731
    monkeypatch.setattr(run_manager, "run_sculpt_job", factory)
    monkeypatch.setattr(runs_routes, "run_sculpt_job", factory)

    slug = _make_project_with_library(client, "RunKill")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 2, "dry_run": False},
    )
    run_id = r.json()["run_id"]

    # Give it a tick to start.
    time.sleep(0.2)
    rk = client.delete(f"/projects/{slug}/runs/{run_id}")
    assert rk.status_code == 200, rk.text

    # Poll for terminal state.
    deadline = time.time() + 5
    while time.time() < deadline:
        body = client.get(f"/projects/{slug}/runs/{run_id}").json()
        if body["status"] in ("stopped", "errored", "completed"):
            break
        time.sleep(0.05)
    assert body["status"] == "stopped"


# ── WS replay ─────────────────────────────────────────────────────────
def test_ws_replays_events_after_completion(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunWs")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "run forward", "iterations": 2, "dry_run": False},
    )
    run_id = r.json()["run_id"]

    # Wait for completion.
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/projects/{slug}/runs/{run_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)

    # Open WS post-completion — should get `connected` + replay + `terminal`.
    with client.websocket_connect(f"/ws/projects/{slug}/runs/{run_id}/events") as ws:
        seen_types: list[str] = []
        while True:
            msg = ws.receive_json()
            seen_types.append(msg["type"])
            if msg["type"] == "terminal":
                break
    assert "connected" in seen_types
    assert "iter_completed" in seen_types
    assert "run_completed" in seen_types
    assert "terminal" in seen_types


# ── env var regression: SCULPTOR_KG_PATH resolves via shared-first ─────
def test_run_sculpt_job_exports_shared_kg_path_for_new_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `run_sculpt_job` must set `SCULPTOR_KG_PATH` via
    `project_kg_db_path` (shared-first) rather than hardcoding the
    legacy `<project>/kg/graph.db`. The hardcoded path created an empty
    legacy DB on every first run, which then shadowed the shared DB
    (papers=46) and left the UI's KG tab showing "No papers" for that
    project."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job
    from backend.services.kg_store import shared_kg_db_path

    # Fresh project with no legacy kg/graph.db → shared path should win.
    project_dir = tmp_path / "fresh-proj"
    project_dir.mkdir()

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    # RS_KG_PATH redirect from conftest applies here — shared_kg_db_path()
    # respects it. We assert the runner picks up the SAME resolved path.
    expected_kg_path = str(shared_kg_db_path())

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={"behavior_goal": "hop", "iterations": 1},
    )
    job = Job(job_id="t", kind="sculpt_run", project_slug="fresh-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert captured["env"]["SCULPTOR_KG_PATH"] == expected_kg_path, (
        f"run_manager must export the shared-first resolved KG path; "
        f"got {captured['env']['SCULPTOR_KG_PATH']!r}, "
        f"expected {expected_kg_path!r}"
    )
    legacy = str(project_dir / "kg" / "graph.db")
    assert captured["env"]["SCULPTOR_KG_PATH"] != legacy, (
        "falling back to the legacy per-project path would re-introduce "
        "the empty-DB-shadows-shared bug"
    )


def test_run_sculpt_job_honors_existing_legacy_kg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opposite case: if a project DOES have a pre-existing legacy
    `<project>/kg/graph.db`, `run_sculpt_job` keeps pointing at it (no
    silent migration to shared, per `project_kg_db_path`'s contract)."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "legacy-proj"
    (project_dir / "kg").mkdir(parents=True)
    legacy_db = project_dir / "kg" / "graph.db"
    legacy_db.write_bytes(b"")  # empty but present

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={"behavior_goal": "hop", "iterations": 1},
    )
    job = Job(job_id="t", kind="sculpt_run", project_slug="legacy-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert captured["env"]["SCULPTOR_KG_PATH"] == str(legacy_db)


# ── Test 1 follow-up (Issue C): training_iterations plumbing ─────────
def test_run_sculpt_job_forwards_training_iterations_as_cli_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the 'UI set rsl_rl iters/cycle=100 but subprocess
    used 1500' bug (2026-04-22 Test 1): the UI's `training_iterations`
    run-param must surface as `--steps-per-iter N` on the sculpt CLI.
    Pre-fix, only 4 of 8 NewRunRequest fields reached the CLI — the
    rest (training_iterations, num_envs_override, device_override,
    expand_kg) were silently dropped.
    """
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "ti-proj"
    project_dir.mkdir()
    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "balance the pole",
            "iterations": 3,
            "training_iterations": 100,
        },
    )
    job = Job(job_id="t_ti", kind="sculpt_run", project_slug="ti-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert "--steps-per-iter" in cmd, (
        f"--steps-per-iter must be in CLI args when training_iterations "
        f"is set; got {cmd!r}"
    )
    sp_idx = cmd.index("--steps-per-iter")
    assert cmd[sp_idx + 1] == "100", (
        f"expected --steps-per-iter 100, got {cmd[sp_idx + 1]!r}"
    )


def test_run_sculpt_job_omits_steps_per_iter_flag_when_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default case: no training_iterations → no --steps-per-iter flag
    (config.toml's value wins). Prevents accidental override-by-None."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "no-ti-proj"
    project_dir.mkdir()
    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={"behavior_goal": "x", "iterations": 1},
    )
    job = Job(job_id="t_no", kind="sculpt_run", project_slug="no-ti-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert "--steps-per-iter" not in captured["cmd"], captured["cmd"]


# ── §Ship-7: rollout-video + RL-knob CLI forwarding ────────────────────
def test_run_sculpt_job_forwards_ship7_params_as_cli_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every §Ship-7 param (max_episode_steps, playback_speed, seed,
    etc.) must land as a matching `sculpt run` CLI flag so Sam's UI
    edits actually reach `_mjlab_runner`."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "proj-ship7"
    project_dir.mkdir()
    (project_dir / "runs").mkdir()

    captured: dict[str, list[str]] = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "x",
            "iterations": 1,
            "max_episode_steps": 750,
            "playback_speed": 2.0,
            "render_every": 3,
            "rollout_fps": 30,
            "rollout_episodes": 10,
            "seed": 1337,
            "auto_adjust_physics": True,
        },
    )
    job = Job(job_id="ship7_fwd", kind="sculpt_run", project_slug="ship7-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert "--max-episode-steps" in cmd
    assert cmd[cmd.index("--max-episode-steps") + 1] == "750"
    assert "--playback-speed" in cmd
    assert cmd[cmd.index("--playback-speed") + 1] == "2.0"
    assert "--render-every" in cmd
    assert cmd[cmd.index("--render-every") + 1] == "3"
    assert "--rollout-fps" in cmd
    assert cmd[cmd.index("--rollout-fps") + 1] == "30.0"
    assert "--rollout-episodes" in cmd
    assert cmd[cmd.index("--rollout-episodes") + 1] == "10"
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "1337"
    assert "--auto-adjust-physics" in cmd
    assert "--no-auto-adjust-physics" not in cmd


def test_run_sculpt_job_forwards_early_stop_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship-9a: --early-stop/--no-early-stop + --early-stop-patience
    must land on the CLI."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "proj-es"
    project_dir.mkdir()
    (project_dir / "runs").mkdir()
    captured: dict[str, list[str]] = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "x", "iterations": 1,
            "early_stop_enabled": False,
            "early_stop_patience": 7,
        },
    )
    job = Job(job_id="es_test", kind="sculpt_run", project_slug="es-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))
    cmd = captured["cmd"]
    assert "--no-early-stop" in cmd
    assert "--early-stop" not in cmd  # mutex flag
    assert "--early-stop-patience" in cmd
    assert cmd[cmd.index("--early-stop-patience") + 1] == "7"


def test_run_sculpt_job_forwards_auto_adjust_physics_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auto_adjust_physics=false → --no-auto-adjust-physics flag."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "proj-off"
    project_dir.mkdir()
    (project_dir / "runs").mkdir()
    captured: dict[str, list[str]] = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "x", "iterations": 1,
            "auto_adjust_physics": False,
        },
    )
    job = Job(job_id="ship7_off", kind="sculpt_run", project_slug="ship7-off", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert "--no-auto-adjust-physics" in captured["cmd"]
    assert "--auto-adjust-physics" not in captured["cmd"]


def test_run_sculpt_job_omits_ship7_flags_when_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None values → no CLI flags. Preserves backward compat for the
    existing M7-phase-4 `run_params` shapes."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "proj-noparams"
    project_dir.mkdir()
    (project_dir / "runs").mkdir()
    captured: dict[str, list[str]] = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={"behavior_goal": "x", "iterations": 1},
    )
    job = Job(job_id="ship7_none", kind="sculpt_run", project_slug="ship7-none", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    for flag in (
        "--max-episode-steps", "--playback-speed", "--render-every",
        "--rollout-fps", "--rollout-episodes", "--seed",
        "--auto-adjust-physics", "--no-auto-adjust-physics",
    ):
        assert flag not in cmd, f"{flag} leaked when not set: {cmd}"


# ── §7.3: realism audit surfacing ──────────────────────────────────────
def _fake_sculpt_factory_with_realism(project_dir: Path):
    """Fake sculpt runner that emits a `realism_audited` event alongside
    the usual iter lifecycle so we can verify the backend plumbing picks
    it up and surfaces it on `IterEventSummary.realism_audit`."""
    from backend.services.job_manager import Job

    async def _runner(job: Job, cancel: asyncio.Event):
        job.emit({
            "type": "run_started", "source": "stdout",
            "project": str(project_dir), "iterations": 1,
            "behavior_goal": job.params.get("behavior_goal") or "",
        })
        job.emit({
            "type": "iter_started", "source": "fs", "iter": 0,
            "reward_version_before": 0,
        })
        job.emit({"type": "rollout_done", "source": "fs", "iter": 0, "size_bytes": 1024})
        # §7.3 event carrying the full audit payload.
        job.emit({
            "type": "realism_audited",
            "source": "stdout",
            "iter": 0,
            "verdict": "severe",
            "audit": {
                "verdict": "severe",
                "torque_saturation_frac": 0.47,
                "any_joint_saturation_max": 0.92,
                "joint_vel_p99_max": 85.0,
                "joint_vel_multiplier_vs_nominal": 2.83,
                "joint_limit_violation_frac": 0.01,
                "top_joints_saturation": [
                    {"name": "knee_pitch_left", "value": 0.92},
                ],
                "top_joints_vel": [],
                "top_joints_limit_violation": [],
                "n_actuators": 23, "n_joints": 29, "n_steps": 500,
            },
        })
        job.emit({
            "type": "diagnosed", "source": "fs", "iter": 0,
            "failure_modes": ["reward_hacking"],
            "confidence": 0.9, "n_edits": 1,
        })
        job.emit({
            "type": "iter_completed", "source": "stdout", "iter": 0,
            "primary_metric": 1.5, "failure_modes": ["reward_hacking"],
            "edit_count": 1, "reward_version_after": 1,
        })
        job.emit({
            "type": "run_completed", "return_code": 0,
            "iterations_run": 1, "primary_metric_history": [1.5],
        })
        return {"return_code": 0, "iterations_run": 1}

    return _runner


@pytest.fixture
def fake_sculpt_with_realism(monkeypatch: pytest.MonkeyPatch):
    from backend.services import run_manager

    def _factory(*, project_dir: Path, run_params: dict):
        return _fake_sculpt_factory_with_realism(project_dir)

    monkeypatch.setattr(run_manager, "run_sculpt_job", _factory)
    from backend.routes import runs as runs_routes
    monkeypatch.setattr(runs_routes, "run_sculpt_job", _factory)


def test_realism_audit_surfaced_in_iter_detail(
    client: TestClient, tmp_projects_root: Path,
    fake_sculpt_with_realism, monkeypatch,
) -> None:
    """When a `realism_audited` event arrives with a full `audit` dict,
    `GET /projects/{slug}/runs/{run_id}` must expose it on the matching
    iteration's `realism_audit` field so the UI can render the verdict
    chip. Uses the fake sculpt runner so no subprocess is spawned."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunRealism")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "test realism", "iterations": 1, "dry_run": False},
    )
    run_id = r.json()["run_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/projects/{slug}/runs/{run_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)

    body = client.get(f"/projects/{slug}/runs/{run_id}").json()
    iters = body["iterations"]
    assert iters, "iter list empty"
    iter0 = next(it for it in iters if it["iter_index"] == 0)
    ra = iter0.get("realism_audit")
    assert ra is not None, f"realism_audit not on iter0: {iter0}"
    assert ra["verdict"] == "severe"
    assert ra["torque_saturation_frac"] == pytest.approx(0.47)
    assert ra["top_joints_saturation"][0]["name"] == "knee_pitch_left"


def _fake_sculpt_factory_with_physics_suggestion(project_dir: Path):
    """§7.4 fake runner — emits `physics_edit_suggested` event after the
    realism audit so we can verify the slot-field plumbing."""
    from backend.services.job_manager import Job

    async def _runner(job: Job, cancel: asyncio.Event):
        job.emit({"type": "iter_started", "source": "fs", "iter": 0})
        job.emit({"type": "rollout_done", "source": "fs", "iter": 0, "size_bytes": 1024})
        job.emit({
            "type": "realism_audited", "source": "stdout", "iter": 0,
            "verdict": "severe",
        })
        job.emit({
            "type": "physics_edit_suggested",
            "source": "stdout",
            "iter": 0,
            "prompt": (
                "The last RL rollout exploited unrealistic actuator "
                "response (verdict=SEVERE). Tighten the MJCF so the "
                "policy can't continue doing this."
            ),
            "verdict": "severe",
            "top_joints_saturation": [
                {"name": "knee_pitch_left", "value": 0.92},
            ],
        })
        job.emit({
            "type": "diagnosed", "source": "fs", "iter": 0,
            "failure_modes": ["reward_hacking"], "confidence": 0.9, "n_edits": 1,
        })
        job.emit({
            "type": "iter_completed", "source": "stdout", "iter": 0,
            "primary_metric": 1.0, "failure_modes": ["reward_hacking"],
            "edit_count": 1, "reward_version_after": 1,
        })
        job.emit({
            "type": "run_completed", "return_code": 0,
            "iterations_run": 1, "primary_metric_history": [1.0],
        })
        return {"return_code": 0, "iterations_run": 1}

    return _runner


@pytest.fixture
def fake_sculpt_with_physics_suggestion(monkeypatch: pytest.MonkeyPatch):
    from backend.services import run_manager

    def _factory(*, project_dir: Path, run_params: dict):
        return _fake_sculpt_factory_with_physics_suggestion(project_dir)

    monkeypatch.setattr(run_manager, "run_sculpt_job", _factory)
    from backend.routes import runs as runs_routes
    monkeypatch.setattr(runs_routes, "run_sculpt_job", _factory)


def test_physics_edit_suggestion_surfaced_in_iter_detail(
    client: TestClient, tmp_projects_root: Path,
    fake_sculpt_with_physics_suggestion, monkeypatch,
) -> None:
    """§7.4: `physics_edit_suggested` events land in the iter slot's
    `physics_edit_suggestion` field so the UI can render the 'apply
    physics fix' chip."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunPhysSug")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "test phys sugg", "iterations": 1, "dry_run": False},
    )
    run_id = r.json()["run_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/projects/{slug}/runs/{run_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)

    body = client.get(f"/projects/{slug}/runs/{run_id}").json()
    iter0 = next(it for it in body["iterations"] if it["iter_index"] == 0)
    sug = iter0.get("physics_edit_suggestion")
    assert sug is not None, iter0
    assert "SEVERE" in sug["prompt"]
    assert sug["verdict"] == "severe"
    assert sug["top_joints_saturation"][0]["name"] == "knee_pitch_left"


def test_physics_edit_suggestion_none_without_event(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch,
) -> None:
    """Baseline: fake runs that do NOT emit the event keep the field None."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunNoPhysSug")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "test baseline", "iterations": 1, "dry_run": False},
    )
    run_id = r.json()["run_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/projects/{slug}/runs/{run_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)
    body = client.get(f"/projects/{slug}/runs/{run_id}").json()
    for it in body["iterations"]:
        assert it.get("physics_edit_suggestion") is None


def test_iter_detail_has_realism_audit_none_when_no_event(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch,
) -> None:
    """Baseline: when the sculpt run emits no `realism_audited` events
    (old-style runs, or rollout didn't finish), the field must be None
    rather than crashing the pydantic serialization."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "RunNoRealism")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "test baseline", "iterations": 1, "dry_run": False},
    )
    run_id = r.json()["run_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/projects/{slug}/runs/{run_id}").json()["status"] == "completed":
            break
        time.sleep(0.05)

    body = client.get(f"/projects/{slug}/runs/{run_id}").json()
    for it in body["iterations"]:
        assert it.get("realism_audit") is None, (
            f"expected None realism_audit; got {it.get('realism_audit')}"
        )
