"""Tests for Ship 18a — mission CRUD + execution endpoints.

Covers:
  - POST /projects/{slug}/missions submits a mission_decompose job (202).
  - GET /projects/{slug}/missions lists missions from .missions/.
  - GET /projects/{slug}/missions/{mission_slug} returns full detail
    or 404.
  - POST /projects/{slug}/missions/{mission_slug}/run requires the
    mission to exist + no active job + GPU not busy.
  - DELETE removes the mission directory + reports freed_bytes.
  - Slug auto-derivation with collision resolution.
  - Path-relocation safety: mission_dir is reconstructed from file
    location (not persisted).

Decompose calls real Claude in production; tests stub
`run_mission_decompose_job` so they don't hit Anthropic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Helpers ─────────────────────────────────────────────────────────
def _make_project(client: TestClient) -> str:
    r = client.post(
        "/projects", json={"name": "Mission Test", "adapter": "gym_sb3"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _seed_mission_on_disk(
    project_dir: Path, mission_slug: str, *, n_stages: int = 2,
    statuses: list[str] | None = None,
) -> Path:
    """Materialize a `.missions/<slug>/mission.json` directly on disk
    (bypassing decompose) so GET / DELETE / RUN tests can exercise
    the routes without a real Claude call."""
    md = project_dir / ".missions" / mission_slug
    md.mkdir(parents=True, exist_ok=True)
    statuses = statuses or ["pending"] * n_stages
    stages = []
    for i in range(n_stages):
        stages.append({
            "name": f"stage_{i}",
            "goal_text": f"step {i}",
            "success_criterion": "metric > 0.5",
            "max_iterations": 2,
            "parent_stage": f"stage_{i-1}" if i > 0 else None,
            "reward_seed_prompt": f"seed for {i}",
            "kg_seed_papers": [],
            "status": statuses[i],
            "final_policy_path": None,
            "final_reward_path": None,
            "best_metric": None,
            "iterations_used": 0,
            "started_at": None,
            "finished_at": None,
            "redecomposition_attempts": 0,
        })
    payload = {
        "schema_version": 1,
        "goal": "test mission",
        "decomposition_model": "claude-opus-4-7",
        "decomposition_rationale": "test",
        "created_at": "2026-04-24T00:00:00+00:00",
        "current_stage_idx": 0,
        "stages": stages,
    }
    (md / "mission.json").write_text(json.dumps(payload))
    return md


# ── 404 paths ───────────────────────────────────────────────────────
def test_create_mission_404_unknown_project(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.post(
        "/projects/no-such-slug/missions",
        json={"goal": "Stand on one leg without falling"},
    )
    assert r.status_code == 404


def test_list_missions_404_unknown_project(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.get("/projects/no-such-slug/missions")
    assert r.status_code == 404


def test_get_mission_404_unknown_project(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    r = client.get("/projects/no-such-slug/missions/foo")
    assert r.status_code == 404


def test_get_mission_404_unknown_mission(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/missions/no-such-mission")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


# ── List paths ──────────────────────────────────────────────────────
def test_list_missions_empty_when_none_exist(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/missions")
    assert r.status_code == 200
    assert r.json() == []


def test_list_missions_returns_pre_seeded(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    _seed_mission_on_disk(tmp_projects_root / slug, "beta")

    r = client.get(f"/projects/{slug}/missions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    slugs = {m["mission_slug"] for m in body}
    assert slugs == {"alpha", "beta"}
    # All seeded as 'pending' → lifecycle=ready (no active job).
    for m in body:
        assert m["lifecycle"] == "ready"
        assert m["active_job_id"] is None


# ── Detail paths ────────────────────────────────────────────────────
def test_get_mission_detail_returns_stages(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha", n_stages=3)

    r = client.get(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 200
    body = r.json()
    assert body["mission_slug"] == "alpha"
    assert body["project_slug"] == slug
    assert body["n_stages"] == 3
    assert len(body["stages"]) == 3
    assert body["stages"][0]["name"] == "stage_0"
    assert body["stages"][1]["parent_stage"] == "stage_0"
    assert body["lifecycle"] == "ready"


def test_get_mission_detail_lifecycle_completed_when_all_succeeded(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(
        tmp_projects_root / slug, "alpha",
        n_stages=2, statuses=["succeeded", "succeeded"],
    )
    r = client.get(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 200
    assert r.json()["lifecycle"] == "completed"


def test_get_mission_detail_lifecycle_halted_when_any_failed(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(
        tmp_projects_root / slug, "alpha",
        n_stages=2, statuses=["succeeded", "failed"],
    )
    r = client.get(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 200
    assert r.json()["lifecycle"] == "halted"


# ── Create paths ────────────────────────────────────────────────────
def test_create_mission_validates_short_goal(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "x"},  # < 8 chars
    )
    assert r.status_code == 422


def test_create_mission_rejects_extra_fields(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Stand on one leg", "evil_field": "x"},
    )
    assert r.status_code == 422


def test_create_mission_explicit_slug_collision_returns_409(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Stand on one leg", "mission_slug": "alpha"},
    )
    assert r.status_code == 409


def test_create_mission_returns_202_with_jobsummary(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """POST submits a `mission_decompose` job. Stub the runner so
    we don't actually call Claude; just verify the route plumbing."""
    slug = _make_project(client)

    # Stub the decompose runner so no Anthropic call fires.
    import asyncio as _asyncio
    from backend.services import mission_jobs as mj_mod

    async def _stub_runner(job, cancel):
        # Simulate immediate completion without writing anything.
        return {"mission_slug": "stubbed", "n_stages": 0}

    def _factory(**_kwargs):
        return _stub_runner
    monkeypatch.setattr(
        mj_mod, "run_mission_decompose_job", _factory,
    )

    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Stand on one leg without falling"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "mission_decompose"
    assert body["project_slug"] == slug
    assert body["status"] in ("queued", "running", "completed")


def test_create_mission_accepts_run_defaults_and_round_trips(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """§Ship 21a regression: NewMissionDialog Advanced tab POSTs
    `run_defaults` alongside `goal`. Backend persists them on the
    Mission so RunMissionDialog can pre-fill on first open.

    Validates:
      - Pydantic CreateMissionRequest accepts the nested
        `run_defaults` shape (same fields as RunMissionRequest).
      - Route forwards run_defaults to the decompose runner via
        `params["run_defaults"]` (so the optimistic-cache write also
        sees them).
      - JobDetail.params on the wire includes the round-tripped
        run_defaults dict.

    Doesn't run an actual decompose (stubbed); the full
    persist-then-load round-trip is exercised by the sculptor-side
    mission.py round-trip tests.
    """
    captured: dict[str, object] = {}

    async def _stub_runner(job, cancel):
        return {"mission_slug": "stubbed", "n_stages": 0}

    def _factory(**kwargs):
        # Capture the run_defaults kwarg so we can assert the route
        # forwarded it to the runner factory correctly. Patch in the
        # routes module namespace (where it's bound at import) rather
        # than the services module — `from backend.services... import
        # run_mission_decompose_job` made a local reference.
        captured["run_defaults"] = kwargs.get("run_defaults")
        return _stub_runner

    monkeypatch.setattr(
        "backend.routes.missions.run_mission_decompose_job", _factory,
    )

    slug = _make_project(client)
    payload = {
        "goal": "Hold cartpole upright for 200 steps",
        "run_defaults": {
            "iterations_override": 5,
            "early_stop_on_criterion": True,
            "criterion_stability_window": 2,
            "extend_on_improvement": True,
            "max_extensions_per_stage": 2,
            "extension_factor": 0.75,
        },
    }
    r = client.post(f"/projects/{slug}/missions", json=payload)
    assert r.status_code == 202, r.text
    body = r.json()
    # Wire format: params.run_defaults round-trips so the optimistic-
    # cache write in useCreateMission has access to them.
    assert "run_defaults" in body["params"], (
        "params.run_defaults should be on the wire so the frontend "
        "can read what defaults were stored"
    )
    rd = body["params"]["run_defaults"]
    assert rd["iterations_override"] == 5
    assert rd["early_stop_on_criterion"] is True
    assert rd["criterion_stability_window"] == 2
    assert rd["extend_on_improvement"] is True
    assert rd["max_extensions_per_stage"] == 2
    assert rd["extension_factor"] == 0.75
    # Defaults filled in for unspecified fields by pydantic.
    assert "extension_improvement_threshold" in rd
    # The route also forwarded run_defaults to the decompose runner
    # factory so it can persist on the Mission once decompose completes.
    assert captured["run_defaults"] is not None
    assert captured["run_defaults"]["iterations_override"] == 5


def test_create_mission_run_defaults_optional(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """§Ship 21a: omitting run_defaults preserves the Ship 18a/19d
    behavior (Basic-tab-only flow). params.run_defaults is None on
    the wire — the frontend's RunMissionDialog falls back to
    Claude-authored max_iterations as before."""

    async def _stub_runner(job, cancel):
        return {"mission_slug": "stubbed", "n_stages": 0}

    def _factory(**kwargs):
        return _stub_runner

    monkeypatch.setattr(
        "backend.routes.missions.run_mission_decompose_job", _factory,
    )

    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Hold cartpole upright"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["params"]["run_defaults"] is None


def test_create_mission_response_includes_params_mission_slug(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """§Ship 20 Goal #3 regression: the auto-open MissionDetailDialog
    relies on `params.mission_slug` being on the wire of POST
    /projects/{slug}/missions's response. Ship 19c flipped the
    response_model from JobSummary → JobDetail to include `params`;
    this test pins that contract so a future refactor can't silently
    revert it (which is exactly what would re-break the auto-open
    dialog Sam reported in the Ship 20 handoff).

    NewMissionDialog.tsx reads response.params.mission_slug at the
    onSuccess of useCreateMission and calls onCreated(slug) which
    triggers MissionsTab to open the detail dialog. If params drops
    off the wire, the cast to {params?: {mission_slug?: string}}
    yields undefined and the dialog never opens.
    """
    from backend.services import mission_jobs as mj_mod

    async def _stub_runner(job, cancel):
        return {"mission_slug": "stubbed", "n_stages": 0}

    def _factory(**_kwargs):
        return _stub_runner

    monkeypatch.setattr(mj_mod, "run_mission_decompose_job", _factory)

    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Hold cartpole upright"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    # The load-bearing fields the frontend type-casts off (see
    # NewMissionDialog.tsx onSuccess + useMissions.ts useCreateMission
    # onSuccess) — both call sites read `params.mission_slug`.
    assert "params" in body, (
        "JobDetail.params must be on the wire so the auto-open dialog "
        "can route to the new mission_slug"
    )
    assert body["params"].get("mission_slug"), (
        "params.mission_slug is the routing key for the auto-open "
        "MissionDetailDialog; missing or empty breaks Ship 19c UX"
    )
    # Verify the slug is the route's derived value (kebab-case from
    # the goal), not whatever the stub returned via job.result.
    assert body["params"]["mission_slug"] == "hold-cartpole-upright"
    # And `goal` round-trips so the optimistic-cache placeholder shows
    # the right text while the list refetches.
    assert body["params"].get("goal") == "Hold cartpole upright"


# ── §MISSION_RUN_PARITY: per-stage metric best-of-N ──────────────────
def test_create_mission_forwards_stage_metric_candidates(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """§MISSION_RUN_PARITY: NewMissionDialog's Basic-tab best-of-N select
    POSTs `stage_metric_candidates` alongside `gen_stage_metrics`. The
    route must forward BOTH to the decompose runner factory (which passes
    n_candidates into generate_stage_metrics as its second phase).
    """
    captured: dict[str, object] = {}

    async def _stub_runner(job, cancel):
        return {"mission_slug": "stubbed", "n_stages": 0}

    def _factory(**kwargs):
        captured["gen_stage_metrics"] = kwargs.get("gen_stage_metrics")
        captured["stage_metric_candidates"] = kwargs.get(
            "stage_metric_candidates")
        return _stub_runner

    monkeypatch.setattr(
        "backend.routes.missions.run_mission_decompose_job", _factory,
    )

    slug = _make_project(client)
    r = client.post(
        f"/projects/{slug}/missions",
        json={
            "goal": "Squat then launch into a vertical jump and land",
            "gen_stage_metrics": True,
            "stage_metric_candidates": 3,
        },
    )
    assert r.status_code == 202, r.text
    assert captured["gen_stage_metrics"] is True
    assert captured["stage_metric_candidates"] == 3


def test_create_mission_stage_metric_candidates_defaults_to_one() -> None:
    """CreateMissionRequest carries gen_stage_metrics=True +
    stage_metric_candidates=1 by default (no body change needed for the
    single-shot path), and validates the 1..4 bound."""
    import pytest as _pytest
    from pydantic import ValidationError

    from backend.models.mission import CreateMissionRequest

    req = CreateMissionRequest(goal="Stand on one leg without falling")
    assert req.gen_stage_metrics is True
    assert req.stage_metric_candidates == 1

    req2 = CreateMissionRequest(
        goal="Stand on one leg", gen_stage_metrics=False,
        stage_metric_candidates=4,
    )
    assert req2.gen_stage_metrics is False
    assert req2.stage_metric_candidates == 4

    # Out-of-bound N is a 422 (Field ge=1, le=4).
    with _pytest.raises(ValidationError):
        CreateMissionRequest(goal="x" * 8, stage_metric_candidates=5)


def test_decompose_job_passes_n_candidates_to_generate_stage_metrics(
    tmp_projects_root: Path, monkeypatch,
) -> None:
    """§MISSION_RUN_PARITY: run_mission_decompose_job's stage-metrics
    phase calls generate_stage_metrics(..., n_candidates=N). Mock the
    generator + decompose so no Anthropic/GPU call fires; assert the
    n_candidates kwarg threads all the way through.
    """
    import asyncio
    import json

    from backend.services import mission_jobs as mj_mod
    from backend.services.job_manager import Job

    # Build a real sculpt project so load_adapter + config.toml resolve.
    project_dir = tmp_projects_root / "cand_proj"
    from sculptor.sculpt import sculpt_init
    sculpt_init(project_dir, "gym_sb3")
    mission_slug = "cand-mission"

    captured: dict[str, object] = {}

    # Stub decompose_task so it writes a minimal 1-stage mission + returns.
    def _fake_decompose_task(goal, reward_contract, *, kg_store=None):
        from sculptor.mission import Mission, Stage
        return Mission(
            goal=goal,
            stages=[Stage(
                name="stage_0", goal_text="do the thing",
                success_criterion="metric > 0.5", max_iterations=2,
                parent_stage=None, reward_seed_prompt="seed",
            )],
            decomposition_model="stub",
            decomposition_rationale="stub",
        )

    def _fake_generate_stage_metrics(mission, *, robot_hint=None,
                                     n_candidates=1, **_kw):
        captured["n_candidates"] = n_candidates
        return {"generated": [], "rejected": [], "skipped": []}

    # The freshly-scaffolded gym_sb3 config has env_id="CHANGE_ME", which a
    # real load_adapter can't gym.make — stub it (the decompose + metrics
    # phases only need reward_contract() + a task_id hint).
    class _FakeContract:
        expected_info_keys: list = []
        expected_components = None

    class _FakeAdapter:
        task_id = None

        def reward_contract(self):
            return _FakeContract()

    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter", lambda _p: _FakeAdapter(),
    )
    monkeypatch.setattr(
        "sculptor.decompose.decompose_task", _fake_decompose_task,
    )
    monkeypatch.setattr(
        "sculptor.mission_metrics.generate_stage_metrics",
        _fake_generate_stage_metrics,
    )

    runner = mj_mod.run_mission_decompose_job(
        project_dir=project_dir,
        project_slug="cand_proj",
        goal="Squat then jump",
        mission_slug=mission_slug,
        no_kg=True,  # skip the shared KG DB in the test
        gen_stage_metrics=True,
        stage_metric_candidates=3,
    )
    job = Job(job_id="j1", kind="mission_decompose", project_slug="cand_proj")
    result = asyncio.run(runner(job, asyncio.Event()))

    assert result["n_stages"] == 1
    assert captured["n_candidates"] == 3


# ── §MISSION_RUN_PARITY: run-time knob → CLI flag translation ─────────
def test_build_mission_run_flags_emits_parity_knobs(
    tmp_path: Path,
) -> None:
    """§MISSION_RUN_PARITY: RunMissionRequest's per-launch knobs translate
    into the matching `sculpt mission-run` flags. Names MUST match
    sculptor/cli.py::mission_run_cli's typer Options. None-valued fields
    are skipped (defer to the stage's inherited config).
    """
    from backend.models.mission import RunMissionRequest
    from backend.services.mission_jobs import _build_mission_run_flags

    body = RunMissionRequest(
        edit_candidates=3,
        rollout_episodes=8,
        max_episode_steps=750,
        playback_speed=0.5,
        render_width=960,
        render_height=540,
        fitness_patience=4,
        num_envs_override=1024,
        device_override="cuda:1",
    )
    flags = _build_mission_run_flags(
        body.model_dump(exclude_none=False), tmp_path,
    )

    # Adjacent flag+value pairs so we assert the exact value too.
    def _val(flag: str) -> str:
        i = flags.index(flag)
        return flags[i + 1]

    assert "--edit-candidates" in flags and _val("--edit-candidates") == "3"
    assert "--render-width" in flags and _val("--render-width") == "960"
    assert "--render-height" in flags and _val("--render-height") == "540"
    assert "--rollout-episodes" in flags and _val("--rollout-episodes") == "8"
    assert "--max-episode-steps" in flags
    assert "--playback-speed" in flags and _val("--playback-speed") == "0.5"
    assert "--fitness-patience" in flags and _val("--fitness-patience") == "4"
    assert "--num-envs" in flags and _val("--num-envs") == "1024"
    assert "--device" in flags and _val("--device") == "cuda:1"


def test_build_mission_run_flags_skips_unset_parity_knobs(
    tmp_path: Path,
) -> None:
    """§MISSION_RUN_PARITY: an empty RunMissionRequest emits none of the
    parity flags, and a blank device_override string is skipped (must not
    shadow the stage's inherited device with '')."""
    from backend.models.mission import RunMissionRequest
    from backend.services.mission_jobs import _build_mission_run_flags

    flags = _build_mission_run_flags(
        RunMissionRequest().model_dump(exclude_none=False), tmp_path,
    )
    for flag in (
        "--edit-candidates", "--rollout-episodes", "--max-episode-steps",
        "--playback-speed", "--render-width", "--render-height",
        "--fitness-patience", "--num-envs", "--device",
    ):
        assert flag not in flags

    # Blank device string is dropped even though it's not None.
    flags2 = _build_mission_run_flags({"device_override": ""}, tmp_path)
    assert "--device" not in flags2


# ── Slug derivation ──────────────────────────────────────────────────
def test_derive_unique_mission_slug_basic():
    from backend.services.mission_store import derive_unique_mission_slug

    slug = derive_unique_mission_slug(
        "Stand on one leg and kick", existing_slugs=set(),
    )
    assert slug == "stand-on-one-leg-and-kick"


def test_derive_unique_mission_slug_collision_resolves():
    from backend.services.mission_store import derive_unique_mission_slug

    slug = derive_unique_mission_slug(
        "Stand on one leg",
        existing_slugs={"stand-on-one-leg", "stand-on-one-leg-2"},
    )
    assert slug == "stand-on-one-leg-3"


def test_derive_unique_mission_slug_empty_goal_falls_back():
    """All-stopwords / non-ASCII goal returns 'mission' fallback."""
    from backend.services.mission_store import derive_unique_mission_slug

    slug = derive_unique_mission_slug("!!!", existing_slugs=set())
    assert slug == "mission"

    slug = derive_unique_mission_slug(
        "🚀🎯", existing_slugs={"mission"},
    )
    assert slug == "mission-2"


# ── Run paths ───────────────────────────────────────────────────────
def test_run_mission_404_when_mission_missing(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.post(f"/projects/{slug}/missions/no-such/run")
    assert r.status_code == 404


def test_run_mission_409_when_active_decompose_running(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Per Ship 18a plan-review concurrency matrix: at most one
    active mission_decompose / mission_execute per (project, mission)
    pair."""
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")

    # Inject a running mission_decompose job for the alpha mission.
    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["fake_decompose"] = Job(
        job_id="fake_decompose",
        kind="mission_decompose",
        project_slug=slug,
        status="running",
        params={"mission_slug": "alpha"},
    )
    r = client.post(f"/projects/{slug}/missions/alpha/run")
    assert r.status_code == 409
    assert r.json()["type"] == "/problems/state-conflict"


def test_run_mission_409_when_other_gpu_job_active(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """GPU contention: a sculpt_run on ANY project blocks new mission
    executions cross-project."""
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["other_run"] = Job(
        job_id="other_run", kind="sculpt_run",
        project_slug="some-other-project", status="running",
    )
    r = client.post(f"/projects/{slug}/missions/alpha/run")
    assert r.status_code == 409


# ── Delete paths ────────────────────────────────────────────────────
def test_delete_mission_returns_freed_bytes(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    md = _seed_mission_on_disk(tmp_projects_root / slug, "alpha")
    # Drop a non-trivial file in the mission so freed_bytes > 0.
    (md / "extra.bin").write_bytes(b"x" * 1024)

    r = client.delete(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 200
    body = r.json()
    assert body["mission_slug"] == "alpha"
    assert body["freed_bytes"] >= 1024
    assert not md.exists()


def test_delete_mission_409_when_active_job(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    _seed_mission_on_disk(tmp_projects_root / slug, "alpha")

    app = client.app  # type: ignore[attr-defined]
    jm = app.state.job_manager
    from backend.services.job_manager import Job
    jm._jobs["active"] = Job(
        job_id="active", kind="mission_execute",
        project_slug=slug, status="running",
        params={"mission_slug": "alpha"},
    )
    r = client.delete(f"/projects/{slug}/missions/alpha")
    assert r.status_code == 409


def test_delete_mission_404_when_missing(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    r = client.delete(f"/projects/{slug}/missions/never-was")
    assert r.status_code == 404


# ── Path-relocation safety ──────────────────────────────────────────
# ── Audit-driven regression tests ──────────────────────────────────
def test_create_mission_reserves_slug_dir_under_lock(
    client: TestClient, tmp_projects_root: Path, monkeypatch,
) -> None:
    """Audit fix #E (CRITICAL): the slug-derivation + slug-reservation
    sequence runs UNDER the project's filelock. Pre-fix, two
    concurrent POSTs without a `mission_slug` override could derive
    the same slug. Post-fix, the second POST sees the first POST's
    reserved dir in `list_mission_slugs` (filtered to dirs containing
    `.decompose_pending` OR `mission.json`) and gets the next slug.

    We can't easily do real concurrency in a TestClient suite, but we
    CAN verify: after the first POST, `<project_dir>/.missions/
    <slug>/.decompose_pending` exists, claiming the slug.
    """
    slug = _make_project(client)

    # Stub the decompose runner so it doesn't actually decompose.
    import asyncio as _asyncio
    from backend.services import mission_jobs as mj_mod

    async def _stub_runner(job, cancel):
        # Sleep so the test can verify the marker file exists DURING.
        await _asyncio.sleep(0)
        return {"mission_slug": "stubbed", "n_stages": 0}

    monkeypatch.setattr(
        mj_mod, "run_mission_decompose_job",
        lambda **_kw: _stub_runner,
    )

    r = client.post(
        f"/projects/{slug}/missions",
        json={"goal": "Stand on one leg without falling"},
    )
    assert r.status_code == 202, r.text

    # The reservation marker was created.
    project_dir = tmp_projects_root / slug
    missions_root = project_dir / ".missions"
    reserved = list(missions_root.iterdir())
    assert len(reserved) == 1
    assert (reserved[0] / ".decompose_pending").is_file()


def test_corrupt_mission_json_surfaces_as_errored_lifecycle(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    """Audit fix #F: a corrupt mission.json no longer silently
    disappears from `GET /missions`. It surfaces with
    `lifecycle="errored"` so the user can see + DELETE it."""
    slug = _make_project(client)
    md = (tmp_projects_root / slug / ".missions" / "broken")
    md.mkdir(parents=True)
    (md / "mission.json").write_text("not valid JSON {{{")

    r = client.get(f"/projects/{slug}/missions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["mission_slug"] == "broken"
    assert body[0]["lifecycle"] == "errored"
    assert body[0]["goal"] == "(unreadable mission.json)"

    # User can DELETE the errored mission.
    r = client.delete(f"/projects/{slug}/missions/broken")
    assert r.status_code == 200
    assert not md.exists()


def test_cli_and_backend_slug_derivation_agree():
    """Audit cross-ref #C: the CLI's `_derive_mission_slug` and the
    backend's `_slugify` + `derive_unique_mission_slug` MUST produce
    the same slug for the same goal. If they diverge, a CLI-created
    mission and a REST-created mission could collide unexpectedly."""
    from sculptor.cli import _derive_mission_slug as cli_slug
    from backend.services.mission_store import derive_unique_mission_slug as backend_slug

    test_cases = [
        "Stand on one leg",
        "Walk forward at 1.5 m/s",
        "🚀 jump high",
        "x",
        "ALL CAPS GOAL",
        "with-some_punctuation!!!",
    ]
    for goal in test_cases:
        cli_result = cli_slug(goal, set())
        backend_result = backend_slug(goal, set())
        assert cli_result == backend_result, (
            f"slug divergence for goal={goal!r}: cli={cli_result!r} "
            f"vs backend={backend_result!r}"
        )


def test_load_mission_reconstructs_mission_dir_from_file_location(
    tmp_path: Path,
) -> None:
    """Ship 18a path-reloc fix: the on-disk mission.json no longer
    persists `mission_dir` (an absolute path). On load, the dir is
    derived from the JSON file's parent. So copying the project
    tree to a new location should keep stage paths correct."""
    from sculptor.mission import load_mission, save_mission, Mission, Stage

    # Build + save a mission in dir A.
    dir_a = tmp_path / "A"
    dir_a.mkdir()
    m = Mission(
        goal="x",
        stages=[Stage(
            name="s", goal_text="x", success_criterion="True",
            max_iterations=1, parent_stage=None,
            reward_seed_prompt="alive",
        )],
        decomposition_model="x", decomposition_rationale="",
    )
    save_mission(m, dir_a)

    # Confirm `mission_dir` is NOT persisted in the JSON.
    raw = json.loads((dir_a / "mission.json").read_text())
    assert "mission_dir" not in raw

    # Move the file to dir B; load should reconstruct mission_dir.
    dir_b = tmp_path / "B"
    dir_b.mkdir()
    (dir_a / "mission.json").rename(dir_b / "mission.json")
    loaded = load_mission(dir_b)
    assert loaded.mission_dir == str(dir_b.resolve())
