"""§Ship 27 (E2): eval-harness tests — stats, campaign orchestration
(stub adapter, no GPU/LLM), resumability, honest-zero failures,
aggregation + capture parity, report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)
from sculptor.eval import CampaignConfig, aggregate, iqm, run_campaign
from sculptor.eval.stats import stratified_bootstrap_ci

# ── stats ────────────────────────────────────────────────────────────


def test_iqm_small_n_is_mean() -> None:
    assert iqm([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_iqm_trims_outliers() -> None:
    # middle 50% of [0,1,2,3,4,5,6,100] = [2,3,4,5] → 3.5 (mean = 15.1)
    assert iqm([0, 1, 2, 3, 4, 5, 6, 100]) == pytest.approx(3.5)


def test_bootstrap_ci_deterministic_and_brackets_point() -> None:
    vals = [0.4, 0.5, 0.55, 0.6, 0.7]
    a = stratified_bootstrap_ci(vals, rng_seed=1)
    b = stratified_bootstrap_ci(vals, rng_seed=1)
    assert a == b
    assert a["ci_low"] <= a["point"] <= a["ci_high"]
    assert a["n"] == 5


def test_bootstrap_ci_degenerate_n1() -> None:
    out = stratified_bootstrap_ci([0.42])
    assert out["point"] == out["ci_low"] == out["ci_high"] == 0.42


# ── stub adapter (dotted-path consumable by load_adapter) ────────────


class _EvalStubAdapter(SculptorAdapter):
    """Deterministic fake: episode length depends on the seed so
    aggregation has real variation; capture settings configurable for
    parity tests; can be armed to fail for specific seeds."""

    train_calls: list[dict[str, Any]] = []
    fail_seeds: set[int] = set()
    step_dt: float = 0.02

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self._last_seed = 0

    def train(self, reward_module_path, output_dir, steps, seed, *,
              init_policy_path=None) -> TrainResult:
        type(self).train_calls.append({
            "reward": str(reward_module_path) if reward_module_path else None,
            "steps": steps, "seed": seed,
        })
        if seed in type(self).fail_seeds:
            raise RuntimeError(f"synthetic training failure for seed {seed}")
        self._last_seed = seed
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.pt").write_bytes(b"CKPT")
        (out / "metrics.json").write_text(json.dumps({"mean_return": 1.0}))
        return TrainResult(
            checkpoint_path=out / "checkpoint.pt",
            metrics_dict={"mean_return": 1.0},
            component_means={},
            logs_path=out / "logs",
        )

    def rollout(self, checkpoint_path, output_dir, n_episodes, **kw) -> RolloutResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Seeded, deterministic quality: seed 1000 → 250, 1017 → 420, …
        mean_len = 250.0 + (self._last_seed % 100) * 10.0
        (out / "behavior.json").write_text(json.dumps({
            "n_episodes": n_episodes,
            "mean_return": mean_len / 100.0,
            "mean_episode_length": mean_len,
            "max_episode_length": 500,
            "step_dt": type(self).step_dt,
            "max_episode_steps": 500,
            "rollout_num_envs": 64,
        }))
        (out / "rollout.mp4").write_bytes(b"MP4")
        np.savez(out / "trajectory.npz", rewards=np.zeros(4, dtype=np.float32))
        return RolloutResult(
            video_path=out / "rollout.mp4",
            keyframes_dir=out / "keyframes",
            trajectory_path=out / "trajectory.npz",
            n_episodes=n_episodes,
        )

    def compute_behavior_metrics(self, rollout) -> dict[str, Any]:
        return {}

    def reward_contract(self) -> RewardContract:
        return RewardContract(observation_space_spec=None, action_space_spec=None)

    def probe_component(self, reward_module_path) -> ComponentProbe:
        return ComponentProbe(ok=True, components={"alive_bonus": 1.0}, total=1.0)


_STUB = "tests.test_eval_harness._EvalStubAdapter"


@pytest.fixture(autouse=True)
def _reset_stub():
    _EvalStubAdapter.train_calls = []
    _EvalStubAdapter.fail_seeds = set()
    _EvalStubAdapter.step_dt = 0.02
    yield


def _cfg(tmp_path: Path, **kw: Any) -> CampaignConfig:
    defaults: dict[str, Any] = dict(
        name="smoke",
        out_dir=tmp_path / "campaign",
        benchmarks=["cartpole_balance"],
        conditions=["plain_ppo", "seed_only"],
        seeds=[1000, 1017],
        steps_per_iter=10,
        spec_threshold=0.5,
        adapter_class_override=_STUB,
    )
    defaults.update(kw)
    return CampaignConfig(**defaults)


# ── config validation ────────────────────────────────────────────────


def test_validate_rejects_unknowns_and_dup_seeds(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown benchmark"):
        _cfg(tmp_path, benchmarks=["nope"]).validate()
    with pytest.raises(KeyError, match="unknown condition"):
        _cfg(tmp_path, conditions=["nope"]).validate()
    with pytest.raises(ValueError, match="duplicate seeds"):
        _cfg(tmp_path, seeds=[1, 1]).validate()


# ── train_only end-to-end (real load_adapter, stub class) ────────────


def test_campaign_train_only_end_to_end(tmp_path: Path, capsys) -> None:
    report = run_campaign(_cfg(tmp_path))
    # 2 conditions × 2 seeds = 4 jobs.
    assert len(report["jobs"]) == 4
    # plain_ppo trains with NO reward module; seed_only with the shared v0.
    rewards = {(c["reward"] is None) for c in _EvalStubAdapter.train_calls}
    assert rewards == {True, False}
    v0s = {
        c["reward"] for c in _EvalStubAdapter.train_calls if c["reward"]
    }
    assert all(r.endswith("v0.py") for r in v0s)

    agg = report["aggregates"]["benchmarks"]["cartpole_balance"]
    assert set(agg) == {"plain_ppo", "seed_only"}
    for cond in agg.values():
        assert cond["seeds"] == [1000, 1017]
        # spec = mean_episode_length / 500: seed 1000 → 250/500 = 0.5;
        # seed 1017 → (250+170)/500 = 0.84.
        assert cond["final_spec_scores"] == pytest.approx([0.5, 0.84])
        assert cond["final"]["ci_low"] <= cond["final"]["point"] <= cond["final"]["ci_high"]
        assert cond["threshold_hit_rate"] == 1.0
    assert report["aggregates"]["capture_parity_warnings"] == []

    out = tmp_path / "campaign"
    assert (out / "campaign_report.json").is_file()
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "cartpole_balance" in html and "plain_ppo" in html
    # Per-job result files exist (the resume keys).
    assert (out / "cartpole_balance" / "plain_ppo" / "seed_1000"
            / "result.json").is_file()
    # Events emitted for observability.
    out_text = capsys.readouterr().out
    assert "eval_campaign_started" in out_text
    assert "eval_job_finished" in out_text
    assert "eval_campaign_finished" in out_text


def test_campaign_resumes_from_result_json(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, conditions=["plain_ppo"])
    run_campaign(cfg)
    first_calls = len(_EvalStubAdapter.train_calls)
    assert first_calls == 2
    report2 = run_campaign(cfg)
    assert len(_EvalStubAdapter.train_calls) == first_calls, "must not retrain"
    assert all(j.get("cached") for j in report2["jobs"])


def test_failed_job_is_honest_zero(tmp_path: Path) -> None:
    _EvalStubAdapter.fail_seeds = {1017}
    report = run_campaign(_cfg(tmp_path, conditions=["plain_ppo"]))
    by_seed = {j["seed"]: j for j in report["jobs"]}
    assert by_seed[1017]["error"] and "synthetic training failure" in by_seed[1017]["error"]
    assert by_seed[1017]["final_spec_score"] == 0.0
    assert by_seed[1000]["error"] is None
    agg = report["aggregates"]["benchmarks"]["cartpole_balance"]["plain_ppo"]
    assert agg["final_spec_scores"] == pytest.approx([0.5, 0.0])
    assert agg["errors"]


def test_sculpt_mode_passes_condition_knobs(tmp_path: Path, monkeypatch) -> None:
    """sculpt-mode jobs route through sculpt_run with the campaign's
    iterations/seed/no_kg — and the harness computes specs from
    whatever iter dirs the run leaves behind."""
    calls: list[dict[str, Any]] = []

    def fake_sculpt_run(*, config_path, behavior_goal, iterations, resume,
                        no_kg, steps_per_iter, rollout_episodes, seed, **kw):
        calls.append({
            "no_kg": no_kg, "iterations": iterations, "seed": seed,
            "resume": resume, "goal": behavior_goal,
        })
        proj = Path(config_path).parent
        for i in range(iterations):
            ro = proj / "runs" / f"iter_{i}" / "rollout"
            ro.mkdir(parents=True, exist_ok=True)
            (ro / "behavior.json").write_text(json.dumps({
                "mean_episode_length": 100.0 + 200.0 * i,
                "max_episode_length": 500,
                "max_episode_steps": 500,
                "step_dt": 0.02,
                "rollout_num_envs": 64,
            }))
        from types import SimpleNamespace
        return SimpleNamespace(iterations_run=iterations)

    import sculptor.sculpt as sculpt_mod
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake_sculpt_run)

    cfg = _cfg(
        tmp_path, conditions=["full", "no_kg"], seeds=[1000], iterations=2,
    )
    report = run_campaign(cfg)
    assert {c["no_kg"] for c in calls} == {True, False}
    assert all(c["iterations"] == 2 and c["seed"] == 1000 for c in calls)
    assert all(c["resume"] is True for c in calls)
    assert all("pole" in c["goal"] for c in calls)
    job = report["jobs"][0]
    # spec series across iters: 100/500=0.2 then 300/500=0.6 → final 0.6,
    # threshold (0.5) first hit at iter 2.
    assert job["final_spec_score"] == pytest.approx(0.6)
    assert job["iters_to_threshold"] == 2
    assert job["iterations_completed"] == 2


def test_compute_matched_condition_scales_steps(tmp_path: Path) -> None:
    """plain_ppo_matched must train ONE run at iterations × steps —
    the equal-GPU baseline (a sculpt job retrains from scratch each
    LLM iter, so its total budget is iterations × steps)."""
    cfg = _cfg(
        tmp_path, conditions=["plain_ppo", "plain_ppo_matched"],
        seeds=[1000], iterations=3, steps_per_iter=10,
    )
    report = run_campaign(cfg)
    steps_by_cond = {
        j["condition"]: j["total_rl_iterations"] for j in report["jobs"]
    }
    assert steps_by_cond == {"plain_ppo": 10, "plain_ppo_matched": 30}
    train_steps = sorted(c["steps"] for c in _EvalStubAdapter.train_calls)
    assert train_steps == [10, 30]


def test_pairwise_paired_differences(tmp_path: Path) -> None:
    """The headline statistic: per-seed condition differences with
    their own bootstrap CI — not two overlapping independent CIs."""
    report = run_campaign(_cfg(tmp_path))
    pw = report["aggregates"]["pairwise"]["cartpole_balance"]
    assert "plain_ppo - seed_only" in pw
    entry = pw["plain_ppo - seed_only"]
    assert entry["seeds"] == [1000, 1017]
    # Stub adapter is condition-blind → identical scores → diffs all 0.
    assert entry["diffs"] == pytest.approx([0.0, 0.0])
    assert entry["diff"]["point"] == pytest.approx(0.0)
    assert entry["diff"]["ci_low"] <= 0.0 <= entry["diff"]["ci_high"]


def test_parity_warns_on_missing_vs_real_capture() -> None:
    """A job with NO capture info must not hide inside the parity
    check — mixed missing/real is itself a warning."""
    base = {"final_spec_score": 0.5, "best_spec_score": 0.5,
            "iters_to_threshold": 1, "wall_seconds": 1.0, "error": None,
            "total_rl_iterations": 10}
    results = [
        {**base, "benchmark": "b", "condition": "x", "seed": 1,
         "capture": {"step_dt": 0.02}},
        {**base, "benchmark": "b", "condition": "y", "seed": 1,
         "capture": None},
    ]
    agg = aggregate(results)
    assert len(agg["capture_parity_warnings"]) == 1


# ── mission mode (§Ship 29 / E4 curriculum condition) ────────────────


def _canned_mission(goal: str):
    from sculptor.mission import Mission, Stage

    return Mission(
        goal=goal,
        decomposition_rationale="two-step canned curriculum",
        decomposition_model="canned-test-model",
        stages=[
            Stage(name="stage_a", goal_text="a", success_criterion="metric > 0",
                  max_iterations=3, parent_stage=None, reward_seed_prompt="ra"),
            Stage(name="stage_b", goal_text="b", success_criterion="metric > 0",
                  max_iterations=3, parent_stage="stage_a",
                  reward_seed_prompt="rb"),
        ],
    )


def _fake_mission_run_factory(calls: list[dict[str, Any]]):
    """Writes per-stage iter layouts + iterations_used into
    mission.json — the artifacts the harness reads for the series and
    the GPU bill."""

    def fake_mission_run(mission, *, adapter_short_name, kg_store,
                         iterations_override, steps_per_iter, seed,
                         early_stop_on_criterion, **kw):
        calls.append({
            "kg": kg_store is not None,
            "iterations_override": iterations_override,
            "steps_per_iter": steps_per_iter,
            "seed": seed,
            "early_stop": early_stop_on_criterion,
        })
        from types import SimpleNamespace

        from sculptor.mission import save_mission

        md = Path(mission.mission_dir)
        lengths = {"stage_a": (2, 100.0), "stage_b": (1, 400.0)}
        for stage in mission.stages:
            n_iters, mean_len = lengths[stage.name]
            stage.iterations_used = n_iters
            stage.status = "succeeded"
            for i in range(n_iters):
                ro = md / "stages" / stage.name / "runs" / f"iter_{i}" / "rollout"
                ro.mkdir(parents=True, exist_ok=True)
                (ro / "behavior.json").write_text(json.dumps({
                    "mean_episode_length": mean_len + 50.0 * i,
                    "max_episode_length": 500,
                    "max_episode_steps": 500,
                    "step_dt": 0.02,
                    "rollout_num_envs": 64,
                }))
        save_mission(mission, md)
        return SimpleNamespace(completed=True, halted_reason=None)

    return fake_mission_run


def test_mission_mode_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """mission condition: decompose once, mission_run with the
    campaign budget, spec series spans stages in curriculum order with
    a job-global index, GPU bill from mission.json iterations_used."""
    calls: list[dict[str, Any]] = []
    decompose_calls: list[str] = []

    def fake_decompose(goal, contract, *, kg_store=None, **kw):
        decompose_calls.append(goal)
        return _canned_mission(goal)

    monkeypatch.setattr("sculptor.decompose.decompose_task", fake_decompose)
    monkeypatch.setattr(
        "sculptor.sculpt.mission_run", _fake_mission_run_factory(calls),
    )

    cfg = _cfg(tmp_path, conditions=["mission_no_kg"], seeds=[1000],
               iterations=3, steps_per_iter=10)
    report = run_campaign(cfg)
    job = report["jobs"][0]

    assert decompose_calls == [
        "balance the pole upright and keep the cart centered",
    ]
    assert calls == [{
        "kg": False, "iterations_override": 3, "steps_per_iter": 10,
        "seed": 1000, "early_stop": True,
    }]
    # Series: stage_a iters 0,1 then stage_b iter 0 — global idx 0..2.
    rj = json.loads(
        (tmp_path / "campaign" / "cartpole_balance" / "mission_no_kg"
         / "seed_1000" / "result.json").read_text(encoding="utf-8")
    )
    series = rj["spec_series"]
    assert [(s["iter"], s["stage"], s["stage_iter"]) for s in series] == [
        (0, "stage_a", 0), (1, "stage_a", 1), (2, "stage_b", 0),
    ]
    # Final = last stage's last iter: 400/500 = 0.8.
    assert job["final_spec_score"] == pytest.approx(0.8)
    assert job["final_rule"] == "last_iteration"
    # GPU bill: (2 + 1) iterations_used × 10 steps.
    assert job["total_rl_iterations"] == 30
    assert job["mission"] == {
        "n_stages": 2, "completed": True, "halted_reason": None,
    }
    assert job["error"] is None


def test_mission_mode_resume_skips_decompose(tmp_path: Path, monkeypatch) -> None:
    """The decomposition is part of the seed's experiment state — a
    resumed job must reuse it, not re-roll the curriculum."""
    calls: list[dict[str, Any]] = []
    decompose_count = {"n": 0}

    def fake_decompose(goal, contract, *, kg_store=None, **kw):
        decompose_count["n"] += 1
        return _canned_mission(goal)

    monkeypatch.setattr("sculptor.decompose.decompose_task", fake_decompose)
    monkeypatch.setattr(
        "sculptor.sculpt.mission_run", _fake_mission_run_factory(calls),
    )
    cfg = _cfg(tmp_path, conditions=["mission_no_kg"], seeds=[1000],
               iterations=3, steps_per_iter=10)
    run_campaign(cfg)
    # Drop the result.json so the job re-RUNS, but keep .missions/.
    (tmp_path / "campaign" / "cartpole_balance" / "mission_no_kg"
     / "seed_1000" / "result.json").unlink()
    run_campaign(cfg)
    assert decompose_count["n"] == 1, "resume must not re-decompose"
    assert len(calls) == 2


def test_capture_parity_warning(tmp_path: Path) -> None:
    """Two jobs of one benchmark with different capture settings must
    surface a parity warning — frame-domain specs are incomparable."""
    results = [
        {"benchmark": "b", "condition": "x", "seed": 1,
         "final_spec_score": 0.5, "best_spec_score": 0.5,
         "iters_to_threshold": 1, "wall_seconds": 1.0, "error": None,
         "capture": {"step_dt": 0.02, "max_episode_steps": 500}},
        {"benchmark": "b", "condition": "y", "seed": 1,
         "final_spec_score": 0.6, "best_spec_score": 0.6,
         "iters_to_threshold": 1, "wall_seconds": 1.0, "error": None,
         "capture": {"step_dt": 0.04, "max_episode_steps": 500}},
    ]
    agg = aggregate(results)
    assert len(agg["capture_parity_warnings"]) == 1
    assert "mixes capture settings" in agg["capture_parity_warnings"][0]
