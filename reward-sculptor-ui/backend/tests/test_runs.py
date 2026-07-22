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


def test_get_run_preserves_advanced_launch_params(
    client: TestClient, tmp_projects_root: Path, fake_sculpt, monkeypatch
) -> None:
    """Run history must report the exact controls the UI actually launched."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project_with_library(client, "AdvancedRunHistory")
    launch_params = {
        "behavior_goal": "cross the authored platform course",
        "iterations": 4,
        "no_kg": False,
        "dry_run": False,
        "training_iterations": 750,
        "num_envs_override": 1024,
        "device_override": "cuda:0",
        "max_episode_steps": 500,
        "playback_speed": 1.0,
        "rollout_episodes": 2,
        "render_width": 960,
        "render_height": 540,
        "seed": 42,
        "fitness_metric": "go1_trot",
        "fitness_mode": "steer",
        "fitness_patience": 4,
        "start_mode": "auto",
    }
    response = client.post(f"/projects/{slug}/runs", json=launch_params)
    assert response.status_code == 202, response.text

    run_id = response.json()["run_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        detail_response = client.get(f"/projects/{slug}/runs/{run_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()
        if detail["status"] == "completed":
            break
        time.sleep(0.05)
    assert detail["status"] == "completed", detail

    for field, expected in launch_params.items():
        assert detail["params"][field] == expected, field


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


def test_run_sculpt_job_ignores_existing_legacy_kg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy project-local graph must never fragment spawned training
    away from the shared graph; the resolver warns and exports shared."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job
    from backend.services.kg_store import shared_kg_db_path

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

    assert captured["env"]["SCULPTOR_KG_PATH"] == str(shared_kg_db_path())
    assert captured["env"]["SCULPTOR_KG_PATH"] != str(legacy_db)


def _promoted_recovery_project(project_dir: Path) -> tuple[Path, Path]:
    """Build the smallest valid atomic tuple needed by recovery tests."""
    from sculptor.edit import _write_current_reexport
    from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

    env_dir = project_dir / "env"
    rewards_dir = project_dir / "rewards"
    env_dir.mkdir(parents=True)
    rewards_dir.mkdir(parents=True)

    promoted_reward = rewards_dir / "v1.py"
    draft_reward = rewards_dir / "v2.py"
    promoted_reward.write_text("REWARD_SPEC = {}\ndef compute_reward(_obs): return 1.0\n")
    draft_reward.write_text("REWARD_SPEC = {}\ndef compute_reward(_obs): return 2.0\n")
    _write_current_reexport(rewards_dir, draft_reward)

    promoted_env = env_dir / "v1.json"
    draft_env = env_dir / "v2.json"
    promoted_env.write_text(json.dumps({
        "env_spec_version": 1,
        "meta": {"version": "v1", "parent": "v0", "source": "test",
                 "rationale": "promoted test input"},
        "shared": {}, "train": {},
    }))
    draft_env.write_text(json.dumps({
        "env_spec_version": 1,
        "meta": {"version": "v2", "parent": "v1", "source": "test",
                 "rationale": "unpromoted test draft"},
        "shared": {}, "train": {"entropy_coef_scale": 2.0},
    }))
    (env_dir / "current.json").write_text(draft_env.read_text())

    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v1", promoted_reward, base=project_dir,
        ),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v1", promoted_env, base=project_dir,
        ),
    }
    for kind in (
        "world", "task", "resolved_eval", "channel_catalog", "clarifications",
    ):
        artifact = env_dir / f"{kind}_v1.json"
        artifact.write_text("{}")
        refs[kind] = ArtifactRef.from_path(
            kind, "v1", artifact, base=project_dir,
        )
    WorldArtifactStore(project_dir).promote(
        refs, evaluation_lineage="recovery-test",
    )
    return promoted_reward, promoted_env


def test_restore_promoted_training_inputs_uses_hash_verified_tuple(
    tmp_path: Path,
) -> None:
    from backend.services import run_manager

    project_dir = tmp_path / "promoted-recovery"
    promoted_reward, promoted_env = _promoted_recovery_project(project_dir)

    result = run_manager._restore_promoted_training_inputs(project_dir)

    assert result["selection_version"] == 1
    assert result["reward_version"] == "v1"
    assert result["env_spec_version"] == "v1"
    assert promoted_reward.name in (
        project_dir / "rewards" / "current.py"
    ).read_text()
    assert json.loads((project_dir / "env" / "current.json").read_text()) == (
        json.loads(promoted_env.read_text())
    )


def test_restore_promoted_training_inputs_rejects_hash_drift_before_repoint(
    tmp_path: Path,
) -> None:
    from backend.services import run_manager

    project_dir = tmp_path / "tampered-recovery"
    promoted_reward, _promoted_env = _promoted_recovery_project(project_dir)
    reward_current = project_dir / "rewards" / "current.py"
    env_current = project_dir / "env" / "current.json"
    reward_before = reward_current.read_bytes()
    env_before = env_current.read_bytes()

    promoted_reward.write_text("REWARD_SPEC = {}\ndef compute_reward(_obs): return 99.0\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_manager._restore_promoted_training_inputs(project_dir)

    assert reward_current.read_bytes() == reward_before
    assert env_current.read_bytes() == env_before


def test_run_sculpt_job_restores_promoted_tuple_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "ui-recovery"
    project_dir.mkdir()
    order: list[str] = []

    def _fake_restore(path: Path) -> dict:
        assert path == project_dir
        order.append("restore")
        return {
            "selection_version": 7,
            "tuple_hash": "f" * 64,
            "reward_version": "v4",
            "reward_sha256": "a" * 64,
            "env_spec_version": "v2",
            "env_spec_sha256": "b" * 64,
        }

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        order.append("subprocess")
        raise _Sentinel()

    monkeypatch.setattr(
        run_manager, "_restore_promoted_training_inputs", _fake_restore,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "recover the last accepted behavior",
            "iterations": 1,
            "resume_exact_tuple": True,
        },
    )
    job = Job(
        job_id="t_recovery", kind="sculpt_run",
        project_slug="ui-recovery", status="running",
    )
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert order == ["restore", "subprocess"]
    restored = next(
        event for event in job.events
        if event.get("type") == "promoted_tuple_restored"
    )
    assert restored["selection_version"] == 7
    assert restored["reward_version"] == "v4"


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


def test_run_sculpt_job_forwards_hardware_overrides_as_cli_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch-scoped env/device controls shown in the UI must be real."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "hardware-override"
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
            "behavior_goal": "traverse rough terrain",
            "iterations": 2,
            "num_envs_override": 512,
            "device_override": "cuda:0",
        },
    )
    job = Job(
        job_id="t_hw", kind="sculpt_run",
        project_slug="hardware-override", status="running",
    )
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert cmd[cmd.index("--num-envs") + 1] == "512"
    assert cmd[cmd.index("--device") + 1] == "cuda:0"


def test_launch_rejects_tampered_authored_world_before_job_submission(
    client: TestClient, fake_sculpt, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services import world_store

    slug = _make_project_with_library(client, "WorldIntegrity")
    project = client.get(f"/projects/{slug}").json()
    selection = Path(project["project_dir"]) / "env" / "selection_current.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        world_store, "validate",
        lambda _project: {"ok": False, "errors": ["task artifact hash mismatch"]},
    )

    response = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "traverse rough terrain",
            "iterations": 1,
            "dry_run": True,
        },
    )
    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/world-integrity"
    assert "hash mismatch" in response.json()["detail"]
    assert client.app.state.job_manager.list(  # type: ignore[attr-defined]
        kind="sculpt_run", project_slug=slug,
    ) == []


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


# ── §Ship 34: objective fitness-in-the-loop CLI forwarding ─────────────
def test_run_sculpt_job_forwards_fitness_metric_as_cli_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI's 'Objective fitness metric' dropdown sets run_params
    ['fitness_metric']; it must surface as `--fitness-metric <name>` on
    the sculpt CLI so the loop is fitness-guided."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "fit-proj"
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
            "behavior_goal": "gallop forward fast",
            "iterations": 3,
            "fitness_metric": "go1_trot",
        },
    )
    job = Job(job_id="t_fit", kind="sculpt_run", project_slug="fit-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert "--fitness-metric" in cmd, cmd
    assert cmd[cmd.index("--fitness-metric") + 1] == "go1_trot", cmd


def test_run_sculpt_job_forwards_fitness_patience_as_cli_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 48: the New Run 'Fitness patience' field sets run_params
    ['fitness_patience']; it must surface as `--fitness-patience N` so the
    LIVE (fitness-plateau) early stop honors the user's value instead of the
    sculpt-lib default of 2 (which truncated the g1-kick-v3 run at iter 4).
    Only emitted alongside a resolved fitness metric."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "fitp-proj"
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
            "behavior_goal": "kick forward repeatedly",
            "iterations": 8,
            "fitness_metric": "g1_kick",
            "fitness_patience": 4,
        },
    )
    job = Job(job_id="t_fitp", kind="sculpt_run", project_slug="fitp-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert "--fitness-patience" in cmd, cmd
    assert cmd[cmd.index("--fitness-patience") + 1] == "4", cmd


def test_run_sculpt_job_omits_fitness_patience_without_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 48: fitness_patience is only meaningful with a metric — no
    metric set → no flag (it would be inert in the blind loop)."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "fitp-nop"
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
            "behavior_goal": "walk forward",
            "iterations": 3,
            "fitness_patience": 4,   # set, but no fitness_metric
        },
    )
    job = Job(job_id="t_fitp0", kind="sculpt_run", project_slug="fitp-nop", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert "--fitness-patience" not in captured["cmd"], captured["cmd"]


def test_run_sculpt_job_gen_metric_uncalibrated_forces_observe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 35 review (CRITICAL): the BACKEND must downgrade steer→observe
    for an uncalibrated generated metric, even if the API request says
    steer — the UI's client-side lock is not the only guard."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "gp"
    mdir = project_dir / "metrics" / "gen_001"
    mdir.mkdir(parents=True)
    (mdir / "metric.py").write_text("def compute_spec(a,b,m): return {}\n", encoding="utf-8")
    (mdir / "meta.json").write_text('{"calibrated": false}', encoding="utf-8")
    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={"behavior_goal": "trot", "iterations": 2,
                    "fitness_metric": "gen:gen_001", "fitness_mode": "steer"},
    )
    job = Job(job_id="t_g", kind="sculpt_run", project_slug="gp", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))
    cmd = captured["cmd"]
    assert "--fitness-metric" in cmd
    assert cmd[cmd.index("--fitness-metric") + 1].endswith("metrics/gen_001/metric.py")
    assert cmd[cmd.index("--fitness-mode") + 1] == "observe"  # steer downgraded


def test_run_sculpt_job_omits_fitness_metric_when_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default: no fitness_metric → no flag → the blind loop (unchanged)."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "no-fit-proj"
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
    job = Job(job_id="t_nofit", kind="sculpt_run", project_slug="no-fit-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert "--fitness-metric" not in captured["cmd"], captured["cmd"]


def test_run_sculpt_job_launch_gen_sentinel_disabled_runs_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 42/43: with the launch-gen kill-switch off (SCULPTOR_LAUNCH_GEN=0),
    the sentinel "generate-at-launch" must not reach the CLI as a metric name —
    it degrades to a blind loop (no --fitness-metric flag, no crash)."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    monkeypatch.setenv("SCULPTOR_LAUNCH_GEN", "0")
    project_dir = tmp_path / "launchgen-proj"
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
        run_params={"behavior_goal": "kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id="t_lg", kind="sculpt_run", project_slug="launchgen-proj",
              status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    assert "--fitness-metric" not in captured["cmd"], captured["cmd"]


def _fake_bridge_gen(*, accept: bool):
    """A mocked sculptor_bridge.generate_objective_metric: writes metric.py,
    fires the Ship-40 stage events, returns an accept/reject rec (no LLM)."""
    def _gen(
        behavior_goal, out_dir, *, robot_hint=None, review=True,
        n_candidates=1, on_event=None, channel_catalog=None,
    ):
        if on_event:
            on_event({"stage": "generating", "attempt": 1, "max": 3,
                      "message": "Generating candidate metric (attempt 1/3)…"})
            on_event({"stage": "validating", "attempt": 1, "max": 3,
                      "message": "Validating…"})
            if accept:
                on_event({"stage": "reviewing", "message": "Reviewing…"})
            on_event({"stage": "done", "accepted": accept,
                      "message": "Metric accepted." if accept else "Rejected."})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metric.py").write_text(
            "def compute_spec(a, b, m):\n    return {'spec_score': 0.5}\n",
            encoding="utf-8")
        return {
            "accepted": accept, "validation_passed": accept, "calibrated": False,
            "behavior_goal": behavior_goal,
            "validation": {
                "gates": {"nondegeneracy": accept},
                "reasons": [] if accept else ["[nondegeneracy] near-constant metric"],
                "archetype_scores": {},
            },
            "review": ({"approved": True, "concerns": [], "summary": "ok"} if accept
                       else None),
            "source": "def compute_spec(...): ...", "recorded_at": "now",
        }
    return _gen


def test_run_sculpt_job_launch_gen_accepts_and_steers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 43: the launch-time generation pre-phase streams metric_generation_*
    events into the run stream and, on acceptance, rewrites the cmd to point at
    the generated metric (observe-only — uncalibrated)."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen(accept=True))
    project_dir = tmp_path / "lg-accept"
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
        run_params={"behavior_goal": "repeatedly kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "steer"},
    )
    job = Job(job_id="t_lga", kind="sculpt_run", project_slug="lg-accept", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    stages = [e["type"] for e in job.events if e.get("type", "").startswith("metric_generation")
              or e.get("type") == "metric_generated"]
    assert stages[0] == "metric_generation_started", stages
    assert "metric_generation_progress" in stages, stages
    assert stages[-1] == "metric_generated", stages
    cmd = captured["cmd"]
    assert "--fitness-metric" in cmd
    assert cmd[cmd.index("--fitness-metric") + 1].endswith("metrics/gen_001/metric.py")
    # uncalibrated → steer downgraded to observe (firewall)
    assert cmd[cmd.index("--fitness-mode") + 1] == "observe"


def test_run_sculpt_job_launch_gen_spec_audit_emits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Metric-quality laws (LAW 9): with the adversarial flag ON, a launch-
    generated KICK metric calibrating against g1_kick ALSO runs the AUDIT-ONLY
    adversarial probe of the hand-authored spec_g1_kick (the gate that never ran on
    the metric that scored g1-kick-v5) and streams a metric_spec_audit event —
    record-only, never revoking the ground-truth fence."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    monkeypatch.setattr(run_manager, "_ADVERSARIAL_ENABLED", True)
    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen(accept=True))
    monkeypatch.setattr(
        sculptor_bridge, "calibrate_objective_metric",
        lambda metric_path, builtin, threshold=0.7: {
            "ok": True, "spearman": 0.95, "builtin": builtin})
    captured_audit: dict = {}

    def _fake_audit(builtin, goal, robot_hint=None, *, client=None):
        captured_audit["args"] = (builtin, goal)
        return {"ran": True, "gameable": False, "worst_name": "active_kick_behind",
                "worst_gaming": 0.01, "coverage_gaps": [], "reason": None}

    monkeypatch.setattr(sculptor_bridge, "audit_builtin_spec_metric", _fake_audit)
    project_dir = tmp_path / "lg-audit"
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
        run_params={"behavior_goal": "repeatedly kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id="t_lgaudit", kind="sculpt_run", project_slug="lg-audit", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    audits = [e for e in job.events if e.get("type") == "metric_spec_audit"]
    assert audits, [e.get("type") for e in job.events]
    assert audits[0]["audit_only"] is True and audits[0]["ran"] is True
    assert audits[0]["gameable"] is False        # audit never revokes the fence
    assert captured_audit["args"][0] == "g1_kick"


def test_run_sculpt_job_launch_gen_spec_audit_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag OFF (the unit-test default) → no spec audit fires, no metric_spec_audit
    event, and no bridge audit call (byte-identical to before the seam)."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    # _disable_network_adversarial autouse already forces the flag OFF.
    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen(accept=True))
    monkeypatch.setattr(
        sculptor_bridge, "calibrate_objective_metric",
        lambda metric_path, builtin, threshold=0.7: {
            "ok": True, "spearman": 0.95, "builtin": builtin})

    def _boom_audit(*a, **k):
        raise AssertionError("audit must not run when the flag is off")

    monkeypatch.setattr(sculptor_bridge, "audit_builtin_spec_metric", _boom_audit)
    project_dir = tmp_path / "lg-audit-off"
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
        run_params={"behavior_goal": "repeatedly kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id="t_lgaudit_off", kind="sculpt_run", project_slug="lg-audit-off",
              status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))
    assert not [e for e in job.events if e.get("type") == "metric_spec_audit"]


async def _decision_blind(control_path, cancel, *, timeout_s=1800.0):
    return "blind"


async def _decision_retry(control_path, cancel, *, timeout_s=1800.0):
    return "retry"


def _fake_bridge_gen_seq(*accepts):
    """Stateful mock: accept/reject per successive call (one per launch-gen
    attempt), writing metric.py each time."""
    state = {"i": 0}

    def _gen(
        behavior_goal, out_dir, *, robot_hint=None, review=True,
        n_candidates=1, on_event=None, channel_catalog=None,
    ):
        accept = accepts[min(state["i"], len(accepts) - 1)]
        state["i"] += 1
        if on_event:
            on_event({"stage": "generating", "attempt": 1, "max": 3, "message": "Generating…"})
            on_event({"stage": "done", "accepted": accept, "message": "x"})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metric.py").write_text(
            "def compute_spec(a, b, m):\n    return {'spec_score': 0.5}\n", encoding="utf-8")
        return {
            "accepted": accept, "validation_passed": accept, "calibrated": False,
            "behavior_goal": behavior_goal,
            "validation": {"gates": {"nondegeneracy": accept},
                           "reasons": [] if accept else ["[nondegeneracy] near-constant metric"],
                           "archetype_scores": {}},
            "review": ({"approved": True, "concerns": [], "summary": "ok"} if accept else None),
            "source": "...", "recorded_at": "now",
        }
    return _gen


def test_run_sculpt_job_launch_gen_rejected_runs_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 43/45: a rejected launch-time generation surfaces a
    metric_generation_rejected event (never silent) WITH reasons; on the
    "continue blind" decision the run proceeds blind (no --fitness-metric)."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen(accept=False))
    monkeypatch.setattr(run_manager, "_await_gen_decision", _decision_blind)
    project_dir = tmp_path / "lg-reject"
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
        run_params={"behavior_goal": "do a spin", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id="t_lgr", kind="sculpt_run", project_slug="lg-reject", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    rejected = [e for e in job.events if e.get("type") == "metric_generation_rejected"]
    assert rejected, [e.get("type") for e in job.events]
    assert rejected[0]["reasons"], rejected[0]
    assert rejected[0]["can_retry"] is True, rejected[0]
    assert "--fitness-metric" not in captured["cmd"], captured["cmd"]


def test_run_sculpt_job_launch_gen_retry_then_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 45: a rejection PAUSES for a one-click decision; on "retry" the
    pre-phase regenerates, and an accepted second attempt steers the run."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    # reject the first attempt, accept the second; no real calibration needed.
    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen_seq(False, True))
    monkeypatch.setattr(sculptor_bridge, "resolve_calibration_builtin",
                        lambda goal, robot_hint=None: None)
    monkeypatch.setattr(run_manager, "_await_gen_decision", _decision_retry)
    project_dir = tmp_path / "lg-retry"
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
        run_params={"behavior_goal": "kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id="t_lgretry", kind="sculpt_run", project_slug="lg-retry", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    starts = [e for e in job.events if e.get("type") == "metric_generation_started"]
    assert len(starts) == 2, [e.get("type") for e in job.events]   # initial + 1 retry
    assert any(e.get("type") == "metric_generated" for e in job.events)
    cmd = captured["cmd"]
    assert "--fitness-metric" in cmd
    assert cmd[cmd.index("--fitness-metric") + 1].endswith("metrics/gen_002/metric.py")


def test_launch_gen_clears_progress_sidecar_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 45 review (MEDIUM): a Stop during in-flight launch-gen must clear
    the Ship-40 progress sidecar — CancelledError (a BaseException) must not
    leave it stuck at {active:true} (phantom spinner in the standalone UI)."""
    import asyncio

    from backend.services import metric_store, run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    def _gen_then_cancel(
        behavior_goal, out_dir, *, robot_hint=None, review=True,
        n_candidates=1, on_event=None, channel_catalog=None,
    ):
        if on_event:  # write an active-progress sidecar, then get cancelled
            on_event({"stage": "generating", "attempt": 1, "max": 4, "message": "working"})
        raise asyncio.CancelledError()

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric", _gen_then_cancel)
    project_dir = tmp_path / "lg-cancel"
    project_dir.mkdir()
    control_path = run_manager.control_file_path(project_dir, "jc")
    run_manager.write_control_file(
        control_path, {"mode": "auto", "resume_token": 0, "feedback": None, "stop": False})
    job = Job(job_id="jc", kind="sculpt_run", project_slug="lg-cancel", status="running")
    cancel = asyncio.Event()

    async def _go():
        return await run_manager._generate_at_launch(
            job, project_dir, "kick with one leg", control_path, cancel)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_go())
    assert metric_store.read_progress(project_dir).get("active") is False


def _run_launch_gen_with_calibration(
    tmp_path, monkeypatch, *, calibrates: bool, slug: str,
):
    """§Ship 44 helper: run the launch-gen pre-phase with a mocked generation
    (accept) + a mocked calibration outcome; return (job, captured cmd)."""
    import asyncio

    from backend.services import run_manager, sculptor_bridge
    from backend.services.job_manager import Job

    monkeypatch.setattr(sculptor_bridge, "generate_objective_metric",
                        _fake_bridge_gen(accept=True))
    monkeypatch.setattr(
        sculptor_bridge, "calibrate_objective_metric",
        lambda metric_path, builtin, threshold=0.7: {
            "ok": calibrates, "spearman": 0.95 if calibrates else 0.2,
            "builtin": builtin, "threshold": threshold})
    project_dir = tmp_path / slug
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
        run_params={"behavior_goal": "repeatedly kick with one leg", "iterations": 1,
                    "fitness_metric": "generate-at-launch", "fitness_mode": "observe"},
    )
    job = Job(job_id=f"t_{slug}", kind="sculpt_run", project_slug=slug, status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))
    return job, captured["cmd"]


def test_run_sculpt_job_launch_gen_calibrates_and_steers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 44: a launch-generated kick metric that PASSES calibration vs
    g1_kick earns steer-rights — the cmd gets --fitness-mode steer even though
    the launch request was observe (uncalibrated at selection time)."""
    job, cmd = _run_launch_gen_with_calibration(
        tmp_path, monkeypatch, calibrates=True, slug="lg-cal-ok")
    assert "--fitness-metric" in cmd
    assert cmd[cmd.index("--fitness-mode") + 1] == "steer", cmd
    done = [e for e in job.events if e.get("type") == "metric_calibration_done"]
    assert done and done[0]["calibrated"] is True, [e.get("type") for e in job.events]
    assert done[0]["builtin"] == "g1_kick"


def test_run_sculpt_job_launch_gen_uncalibrated_stays_observe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§Ship 44: a launch-generated metric that FAILS calibration stays
    observe-only — the firewall holds (steer downgraded)."""
    job, cmd = _run_launch_gen_with_calibration(
        tmp_path, monkeypatch, calibrates=False, slug="lg-cal-no")
    assert "--fitness-metric" in cmd
    assert cmd[cmd.index("--fitness-mode") + 1] == "observe", cmd
    done = [e for e in job.events if e.get("type") == "metric_calibration_done"]
    assert done and done[0]["calibrated"] is False


def test_build_mission_run_flags_includes_fitness_metric(tmp_path) -> None:
    """RunMissionRequest.fitness_metric must become `--fitness-metric`
    on the `sculpt mission-run` CLI; absent when unset. A built-in name
    passes through resolution unchanged (§Ship 35)."""
    from backend.services.mission_jobs import _build_mission_run_flags

    flags = _build_mission_run_flags(
        {"fitness_metric": "g1_kick", "fitness_mode": "steer", "seed": 1000},
        tmp_path)
    assert "--fitness-metric" in flags
    assert flags[flags.index("--fitness-metric") + 1] == "g1_kick"
    assert flags[flags.index("--fitness-mode") + 1] == "steer"
    assert "--fitness-metric" not in _build_mission_run_flags({"seed": 1000}, tmp_path)


def test_build_mission_run_flags_gen_metric_resolution(tmp_path) -> None:
    """§Ship 35 review: a gen:<id> ref is RESOLVED to the metric.py path
    (not passed raw, which would crash the CLI), and an UNCALIBRATED gen
    metric is downgraded steer→observe even if the request says steer."""
    from backend.services.mission_jobs import _build_mission_run_flags

    mdir = tmp_path / "metrics" / "gen_001"
    mdir.mkdir(parents=True)
    (mdir / "metric.py").write_text("def compute_spec(a,b,m): return {}\n", encoding="utf-8")
    (mdir / "meta.json").write_text('{"calibrated": false}', encoding="utf-8")

    flags = _build_mission_run_flags(
        {"fitness_metric": "gen:gen_001", "fitness_mode": "steer"}, tmp_path)
    mp = flags[flags.index("--fitness-metric") + 1]
    assert mp.endswith("metrics/gen_001/metric.py")          # resolved to a path
    assert flags[flags.index("--fitness-mode") + 1] == "observe"  # steer downgraded

    # calibrated → steer allowed
    (mdir / "meta.json").write_text('{"calibrated": true}', encoding="utf-8")
    flags2 = _build_mission_run_flags(
        {"fitness_metric": "gen:gen_001", "fitness_mode": "steer"}, tmp_path)
    assert flags2[flags2.index("--fitness-mode") + 1] == "steer"


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
            "render_width": 1920,
            "render_height": 1080,
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
    assert "--render-width" in cmd
    assert cmd[cmd.index("--render-width") + 1] == "1920"
    assert "--render-height" in cmd
    assert cmd[cmd.index("--render-height") + 1] == "1080"
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
        "--rollout-fps", "--render-width", "--render-height",
        "--rollout-episodes", "--seed",
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


def test_env_extension_suggestion_surfaced_in_iter_summary() -> None:
    """§Ship 48: a `requires_env_extension` event lands in the iter slot's
    `env_extension_suggestion` field (the never-silent deferred-edit signal
    that was missing when every g1-kick-v3 kick term was silently deferred);
    an iter without the event keeps the field None."""
    from backend.services.job_manager import Job
    from backend.services.run_manager import build_iterations_summary

    job = Job(job_id="t_env", kind="sculpt_run", project_slug="p", status="completed")
    job.events = [
        {"type": "iter_started", "iter": 0},
        {"type": "requires_env_extension", "iter": 0,
         "terms": ["swing_leg_forward_kick", "single_leg_stance"],
         "rationales": ["needs per-foot velocity", "needs per-foot contact"]},
        {"type": "iter_completed", "iter": 0, "failure_modes": [], "edit_count": 0},
        {"type": "iter_started", "iter": 1},
        {"type": "iter_completed", "iter": 1, "failure_modes": [], "edit_count": 1},
    ]
    iters = build_iterations_summary(job)
    s0 = next(it for it in iters if it["iter_index"] == 0)
    s1 = next(it for it in iters if it["iter_index"] == 1)
    assert s0["env_extension_suggestion"] is not None, s0
    assert s0["env_extension_suggestion"]["terms"] == [
        "swing_leg_forward_kick", "single_leg_stance"]
    assert s1["env_extension_suggestion"] is None


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


# ── §Ship 21e: _resolve_run_root traversal guard (review fix) ─────────
def test_resolve_run_root_rejects_traversal_stage_name() -> None:
    """A mission_stage_run job whose stage_name (from a corrupt
    mission.json / subprocess event) contains traversal must NOT
    build a path outside the project — _resolve_run_root falls back
    to the project runs dir. Without the guard, get_iter_rollout
    would serve a FileResponse from `<project>/.missions/m/stages/
    ../../../etc/runs/...`."""
    from pathlib import Path

    from backend.routes.runs import _resolve_run_root
    from backend.services.job_manager import Job

    project_dir = Path("/tmp/proj-xyz")
    safe_fallback = project_dir / "runs"

    bad_names = ["../../etc", "a/../b", "..", "with space", "Caps", "m/s"]
    for bad in bad_names:
        job = Job(
            job_id="job_x",
            kind="mission_stage_run",
            project_slug="p",
            params={"mission_slug": "ok_mission", "stage_name": bad},
        )
        assert _resolve_run_root(job, project_dir) == safe_fallback, (
            f"stage_name={bad!r} should fall back to project runs"
        )

    # A bad mission_slug is also rejected.
    job_bad_mission = Job(
        job_id="job_y", kind="mission_stage_run", project_slug="p",
        params={"mission_slug": "../escape", "stage_name": "stand"},
    )
    assert _resolve_run_root(job_bad_mission, project_dir) == safe_fallback

    # A well-formed pair resolves to the stage runs dir.
    job_ok = Job(
        job_id="job_z", kind="mission_stage_run", project_slug="p",
        params={"mission_slug": "my_mission", "stage_name": "stand_stable"},
    )
    assert _resolve_run_root(job_ok, project_dir) == (
        project_dir / ".missions" / "my_mission" / "stages"
        / "stand_stable" / "runs"
    )

    # A top-level sculpt_run always uses the project runs dir.
    job_sculpt = Job(
        job_id="job_w", kind="sculpt_run", project_slug="p", params={},
    )
    assert _resolve_run_root(job_sculpt, project_dir) == safe_fallback


# ── §Ship 39 (H1): interactive control sidecar + endpoint ─────────────
def test_run_sculpt_job_writes_control_file_and_passes_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner ALWAYS writes the control sidecar (deterministic path) and
    passes `--control-file` so the Auto/Manual toggle works mid-run; the
    initial mode comes from start_mode."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "ctrl-proj"
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
        run_params={"behavior_goal": "kick forward",
                    "iterations": 3, "start_mode": "manual"},
    )
    job = Job(job_id="t_ctrl", kind="sculpt_run",
              project_slug="ctrl-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    assert "--control-file" in cmd, cmd
    cf = Path(cmd[cmd.index("--control-file") + 1])
    assert cf == run_manager.control_file_path(project_dir, "t_ctrl")
    assert cf.is_file()
    data = run_manager.read_control_file(cf)
    assert data["mode"] == "manual" and data["resume_token"] == 0
    assert job.params["mode"] == "manual"


def test_run_sculpt_job_control_defaults_to_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No start_mode → control file present but mode='auto' (non-UI / test
    launches never pause)."""
    import asyncio

    from backend.services import run_manager
    from backend.services.job_manager import Job

    project_dir = tmp_path / "auto-proj"
    project_dir.mkdir()
    captured: dict = {}

    class _Sentinel(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        raise _Sentinel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir, run_params={"behavior_goal": "x", "iterations": 1})
    job = Job(job_id="t_auto", kind="sculpt_run",
              project_slug="auto-proj", status="running")
    job._cancel = asyncio.Event()
    with pytest.raises(_Sentinel):
        asyncio.run(runner(job, job._cancel))

    cf = run_manager.control_file_path(project_dir, "t_auto")
    assert run_manager.read_control_file(cf)["mode"] == "auto"


def test_control_endpoint_merges_mode_resume_and_stop(
    client: TestClient, tmp_projects_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    async def _slow_runner(job, cancel):
        job.emit({"type": "log_line", "text": "started"})
        for _ in range(60):
            if cancel.is_set():
                return {"stopped": True}
            await asyncio.sleep(0.05)
        return {"return_code": 0}

    from backend.routes import runs as runs_routes
    from backend.services import run_manager

    factory = lambda *, project_dir, run_params: _slow_runner  # noqa: E731
    monkeypatch.setattr(run_manager, "run_sculpt_job", factory)
    monkeypatch.setattr(runs_routes, "run_sculpt_job", factory)

    slug = _make_project_with_library(client, "RunCtl")
    r = client.post(
        f"/projects/{slug}/runs",
        json={"behavior_goal": "kick forward", "iterations": 3,
              "start_mode": "manual"},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    time.sleep(0.1)

    # resume with feedback bumps the token + stores the note.
    rc = client.patch(
        f"/projects/{slug}/runs/{run_id}/control",
        json={"mode": "manual", "resume": True, "feedback": "lift the leg"},
    )
    assert rc.status_code == 200, rc.text
    body = rc.json()
    assert body["mode"] == "manual"
    assert body["resume_token"] == 1
    assert body["feedback"] == "lift the leg"

    # flip to auto.
    rc2 = client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"mode": "auto"})
    assert rc2.json()["mode"] == "auto"

    # stop sets the flag.
    rc3 = client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"stop": True})
    assert rc3.json()["stop"] is True

    # §Ship 45: a launch-gen retry decision is recorded in the sidecar for the
    # pre-phase to poll; continue-blind bumps the seq with a "blind" decision.
    rc4 = client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"gen_retry": True})
    assert rc4.status_code == 200, rc4.text
    assert rc4.json()["gen_decision"] == "retry"
    assert rc4.json()["gen_decision_seq"] == 1
    rc5 = client.patch(
        f"/projects/{slug}/runs/{run_id}/control", json={"gen_continue": True})
    assert rc5.json()["gen_decision"] == "blind"
    assert rc5.json()["gen_decision_seq"] == 2

    # unknown run → 404.
    assert client.patch(
        f"/projects/{slug}/runs/does_not_exist/control",
        json={"mode": "auto"}).status_code == 404

    client.delete(f"/projects/{slug}/runs/{run_id}")


def test_env_spec_update_surfaced_in_iter_summary() -> None:
    """§env generalization: an `env_spec_updated` event lands in the iter
    slot's `env_spec_update` field (applied + rejected with reasons — the
    diagnoser's env-curriculum change); an iter without the event keeps
    the field None."""
    from backend.services.job_manager import Job
    from backend.services.run_manager import build_iterations_summary

    job = Job(job_id="t_envspec", kind="sculpt_run", project_slug="p",
              status="completed")
    job.events = [
        {"type": "iter_started", "iter": 0},
        {"type": "env_spec_updated", "iter": 0,
         "new_version": "v1",
         "applied": ["entropy_coef_scale=1.5"],
         "rejected": [{"parameter": "entropy_coef_scale",
                       "reason": "99.0 outside hard bounds [0.25, 4.0]"}]},
        {"type": "iter_completed", "iter": 0, "failure_modes": [],
         "edit_count": 0},
        {"type": "iter_started", "iter": 1},
        {"type": "iter_completed", "iter": 1, "failure_modes": [],
         "edit_count": 1},
    ]
    iters = build_iterations_summary(job)
    s0 = next(it for it in iters if it["iter_index"] == 0)
    s1 = next(it for it in iters if it["iter_index"] == 1)
    assert s0["env_spec_update"] == {
        "new_version": "v1",
        "applied": ["entropy_coef_scale=1.5"],
        "rejected": [{"parameter": "entropy_coef_scale",
                      "reason": "99.0 outside hard bounds [0.25, 4.0]"}],
    }
    assert s1["env_spec_update"] is None
