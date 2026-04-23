"""Tests for POST /projects/{slug}/rewards/prompt (M7 Phase 3).

The endpoint fires a background `reward_prompt_edit` job. Tests
exercise the request-validation + lock/busy paths; the Claude call
itself is not executed here — integration with real Claude would
need `ANTHROPIC_API_KEY` and a full sculptor install.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_project(client: TestClient, name: str = "PromptProj") -> str:
    r = client.post(
        "/projects",
        json={"name": name, "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def test_reward_prompt_validation_rejects_short_prompt(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={"prompt": "go", "expected_parent_version": 0},
    )
    assert r.status_code == 422


def test_reward_prompt_validation_rejects_oversized_prompt(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={"prompt": "x" * 2001, "expected_parent_version": 0},
    )
    assert r.status_code == 422


def test_reward_prompt_503_without_api_key(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={
            "prompt": "reward the robot for jumping",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 503
    body = r.json()
    assert body["type"] == "/problems/anthropic-key-missing"


def test_reward_prompt_404_on_unknown_slug(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    r = client.post(
        "/projects/nope/rewards/prompt",
        json={
            "prompt": "reward the robot for jumping",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 404


def test_reward_prompt_409_when_sculpt_run_active(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["sculpt_blocking"] = Job(
        job_id="sculpt_blocking",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )

    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={
            "prompt": "reward the robot for jumping",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 409
    assert r.json()["type"] == "/problems/state-conflict"


def test_reward_prompt_409_when_another_prompt_edit_active(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["prompt_blocking"] = Job(
        job_id="prompt_blocking",
        kind="reward_prompt_edit",
        project_slug=slug,
        status="running",
    )

    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={
            "prompt": "reward the robot for jumping",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 409
    assert r.json()["type"] == "/problems/job-busy"


def test_reward_prompt_submits_job_and_returns_202(
    client: TestClient, tmp_projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug = _make_project(client)

    # Stub the actual edit call so the job doesn't try to hit Anthropic.
    from sculptor import edit as edit_mod

    def fake_apply_prompt_edit(
        current_reward_path, user_prompt, new_iter_id,
        reward_contract, *, kg_store=None, client=None, on_event=None,
    ):
        target = Path(current_reward_path).parent / f"{new_iter_id}.py"
        target.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(s,a,n,i): return 0.0, {}\n",
            encoding="utf-8",
        )
        if on_event is not None:
            on_event({"type": "log_line", "text": "[fake] committed"})
        return target

    monkeypatch.setattr(edit_mod, "apply_prompt_edit", fake_apply_prompt_edit)

    r = client.post(
        f"/projects/{slug}/rewards/prompt",
        json={
            "prompt": "reward the robot for jumping 30 cm",
            "expected_parent_version": 0,
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "reward_prompt_edit"
    assert body["project_slug"] == slug
    assert body["status"] in ("queued", "running")


# ── S3 (bug #6.1 / T2): progress events + timeout ──────────────────
#
# These tests exercise `run_reward_prompt_edit_job`'s runner directly
# rather than going through the HTTP route, because the scaffold-level
# adapter load (gym_sb3 with env_id=CHANGE_ME) errors before reaching
# apply_prompt_edit. Stubbing both load_adapter and apply_prompt_edit
# at the sculptor layer keeps the test on the plumbing we care about:
# event emission + timeout wrapping in reward_jobs.py.

def _stub_sculptor_for_reward_prompt(monkeypatch: pytest.MonkeyPatch):
    """Install no-op stubs for `sculptor.adapters.base.load_adapter`
    and seed the project with a bare v0.py so `_find_latest` finds it.
    Returns a holder the caller fills with `apply_prompt_edit`."""
    from sculptor.adapters import base as base_mod

    class _FakeContract:
        pass

    class _FakeAdapter:
        def reward_contract(self):
            return _FakeContract()

    monkeypatch.setattr(base_mod, "load_adapter", lambda _p: _FakeAdapter())


def _seed_project_rewards(project_dir: Path) -> None:
    """Write the minimal rewards/v0.py `_find_latest` needs."""
    rewards = project_dir / "rewards"
    rewards.mkdir(parents=True, exist_ok=True)
    (rewards / "v0.py").write_text(
        "REWARD_SPEC = {'version': 'v0', 'hyperparameters': {}, 'references': []}\n"
        "def compute_reward(s,a,n,i): return 0.0, {}\n",
        encoding="utf-8",
    )


def _run_runner_sync(runner, project_slug: str = "stub"):
    """Execute an async runner inside a fresh loop bound to a
    JobManager so `job.emit` calls from the to_thread worker thread
    marshal via call_soon_threadsafe into the same loop."""
    import asyncio

    from backend.services.job_manager import Job, JobManager

    async def _go():
        jm = JobManager()
        jm.bind_loop(asyncio.get_running_loop())
        cancel = asyncio.Event()
        job = Job(job_id="stub_job", kind="reward_prompt_edit",
                  project_slug=project_slug)
        job._cancel = cancel
        try:
            await runner(job, cancel)
            job.status = "completed" if job.status == "queued" else job.status
        except Exception as e:  # noqa: BLE001
            job.status = "errored"
            job.error = f"{type(e).__name__}: {e}"
        return job

    return asyncio.run(_go())


def test_reward_prompt_edit_emits_log_line_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reward_prompt_edit runner must emit log_line events via
    `job.emit()` so the UI's Activity panel can surface progress.
    Pre-S3 the runner only set `job.progress` / `job.message`; no
    events = silent hang from the user's perspective. Bug #6.1.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    project_dir = tmp_path / "proj"
    _seed_project_rewards(project_dir)
    _stub_sculptor_for_reward_prompt(monkeypatch)

    from sculptor import edit as edit_mod

    def fake_apply_prompt_edit(
        current_reward_path, user_prompt, new_iter_id,
        reward_contract, *, kg_store=None, client=None, on_event=None,
    ):
        target = Path(current_reward_path).parent / f"{new_iter_id}.py"
        target.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(s,a,n,i): return 0.0, {}\n",
            encoding="utf-8",
        )
        # Simulate the two transitions sculptor's real _call_llm fires.
        if on_event is not None:
            on_event({"type": "log_line", "text": "[edit] LLM request start"})
            on_event({"type": "log_line", "text": "[edit] LLM response received"})
        return target

    monkeypatch.setattr(edit_mod, "apply_prompt_edit", fake_apply_prompt_edit)

    from backend.services.reward_jobs import run_reward_prompt_edit_job

    runner = run_reward_prompt_edit_job(
        project_dir=project_dir,
        user_prompt="reward the robot for jumping 30 cm",
        expected_parent_version=0,
    )
    job = _run_runner_sync(runner)

    assert job.status == "completed", job.error
    log_texts = [
        ev.get("text", "")
        for ev in job.events
        if ev.get("type") == "log_line"
    ]
    # Minimum: runner start, dispatch-to-Claude, the two forwarded
    # events from the fake, completion marker. Exact count may grow if
    # the runner adds more transitions later.
    assert len(log_texts) >= 4, f"expected ≥4 log lines, got {log_texts!r}"
    joined = " ".join(log_texts).lower()
    assert "start" in joined
    assert "llm" in joined or "claude" in joined
    assert any("completed" in t.lower() for t in log_texts), log_texts


def test_reward_prompt_edit_timeout_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Claude call that wedges past `RS_REWARD_PROMPT_TIMEOUT_S` must
    error the job (releasing the per-project 409 lock) rather than
    block forever. Bug #6.1 symptom: Sam saw 'Another prompt-driven
    reward edit is in progress' on retry because the first job never
    reached a terminal state.
    """
    import time

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    project_dir = tmp_path / "proj"
    _seed_project_rewards(project_dir)
    _stub_sculptor_for_reward_prompt(monkeypatch)

    from sculptor import edit as edit_mod

    def slow_apply_prompt_edit(*args, **kwargs):
        time.sleep(2.0)  # intentionally > timeout
        return Path("unreachable")

    monkeypatch.setattr(edit_mod, "apply_prompt_edit", slow_apply_prompt_edit)

    from backend.services.reward_jobs import run_reward_prompt_edit_job

    runner = run_reward_prompt_edit_job(
        project_dir=project_dir,
        user_prompt="reward the robot for jumping 30 cm",
        expected_parent_version=0,
        timeout_s=0.25,
    )
    job = _run_runner_sync(runner)

    assert job.status == "errored", (
        f"expected timeout to error the job; got status={job.status!r}"
    )
    err = (job.error or "").lower()
    assert "time" in err or "wedged" in err or "timeout" in err, job.error


def test_job_events_ws_replays_buffered_events(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """The /ws/jobs/{job_id}/events endpoint replays the job's full
    events list on connect then closes with `terminal` when the job is
    already in a terminal state. This is what RewardsTab's Activity
    panel subscribes to."""
    from backend.services.job_manager import Job

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager

    job = Job(
        job_id="fixture_job",
        kind="reward_prompt_edit",
        project_slug="some-slug",
        status="completed",
    )
    job.emit({"type": "log_line", "text": "hello from unit test"})
    job.emit({"type": "log_line", "text": "second line"})
    jm._jobs[job.job_id] = job

    received: list[dict] = []
    with client.websocket_connect(f"/ws/jobs/{job.job_id}/events") as ws:
        # Drain everything until the WS closes (terminal marker).
        try:
            while True:
                received.append(ws.receive_json())
        except Exception:  # noqa: BLE001 — closed
            pass

    types = [ev.get("type") for ev in received]
    assert "connected" in types
    assert types.count("log_line") == 2
    assert any(ev.get("text") == "hello from unit test"
               for ev in received if ev.get("type") == "log_line")
    assert "terminal" in types


def test_job_events_ws_404_for_missing_job(client: TestClient) -> None:
    """Unknown job_id produces an error frame + close. Must not 500."""
    received: list[dict] = []
    with client.websocket_connect("/ws/jobs/nonexistent/events") as ws:
        try:
            while True:
                received.append(ws.receive_json())
        except Exception:  # noqa: BLE001 — closed
            pass
    types = [ev.get("type") for ev in received]
    assert "error" in types
    assert any("not found" in (ev.get("error") or "") for ev in received)
