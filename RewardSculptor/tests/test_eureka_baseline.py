"""§Ship 28 (E3): Eureka-style baseline tests — candidate sampling,
fitness selection, reward reflection, invalid-candidate handling, and
harness integration. No network, no GPU (stub client + stub adapter)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sculptor.eval.eureka import (
    _reward_reflection,
    _strip_code_fences,
    run_eureka_job,
)

# ── code-fence extraction ────────────────────────────────────────────


def test_strip_fences_prefers_largest_python_block() -> None:
    text = (
        "Here is the reward:\n```python\nshort = 1\n```\n"
        "```python\nREWARD_SPEC = {}\ndef compute_reward(): ...\n```\n"
    )
    out = _strip_code_fences(text)
    assert "REWARD_SPEC" in out and "short" not in out


def test_strip_fences_falls_back_to_raw_text() -> None:
    assert _strip_code_fences("REWARD_SPEC = {}").startswith("REWARD_SPEC")


# ── stub client (anthropic messages.create shape) ────────────────────


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _CreateMessages:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("stub ran out of candidate responses")
        return _Resp(self._responses.pop(0))


class _StubLLM:
    def __init__(self, *responses: str) -> None:
        self.messages = _CreateMessages(list(responses))


def _candidate(marker: str) -> str:
    return (
        f"```python\n# candidate {marker}\nREWARD_SPEC = "
        f"{{'version': '{marker}'}}\n"
        f"def compute_reward(state, action, next_state, info):\n"
        f"    return 0.0, {{}}\n"
        f"def compute_reward_batched(state, action, next_state, info):\n"
        f"    return None, {{}}\n```"
    )


def _broken_candidate(marker: str) -> str:
    """Statically valid (all required defs) but probe-failing."""
    return _candidate(marker).replace(
        f"# candidate {marker}", f"# candidate {marker}\nBROKEN = True",
    )


# ── stub adapter with controllable per-candidate quality ─────────────


from sculptor.adapters.base import SculptorAdapter


class _EurekaStubAdapter(SculptorAdapter):
    """Quality is read from the candidate source's marker comment so
    tests can script which candidate 'wins'. Probe validity likewise:
    a candidate containing BROKEN fails the probe."""

    #: marker -> mean_episode_length the rollout reports (spec = len/500)
    quality: dict[str, float] = {}
    instances: list["_EurekaStubAdapter"] = []

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self._marker = ""
        type(self).instances.append(self)

    def reward_contract(self):
        from sculptor.adapters.base import RewardContract

        return RewardContract(
            observation_space_spec=None, action_space_spec=None,
            expected_info_keys=["fallen", "episode_length"],
            state_schema={"qpos": (2,), "qvel": (2,)},
        )

    def probe_component(self, reward_module_path: Path):
        from sculptor.adapters.base import ComponentProbe

        src = Path(reward_module_path).read_text(encoding="utf-8")
        if "BROKEN" in src:
            return ComponentProbe(ok=False, error="synthetic probe failure")
        return ComponentProbe(ok=True, components={"c": 0.0}, total=0.0)

    def train(self, reward_module_path, output_dir, steps, seed, **kw):
        from sculptor.adapters.base import TrainResult

        src = Path(reward_module_path).read_text(encoding="utf-8")
        self._marker = src.split("candidate ", 1)[1].split("\n", 1)[0].strip()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.pt").write_bytes(b"CKPT")
        (out / "metrics.json").write_text("{}")
        (out / "reward_trajectory.json").write_text(json.dumps({
            "task_term": [0.1, 0.5], "alive": [1.0, 1.0],
        }))
        return TrainResult(
            checkpoint_path=out / "checkpoint.pt", metrics_dict={},
            component_means={}, logs_path=out / "logs",
        )

    def rollout(self, checkpoint_path, output_dir, n_episodes, **kw):
        from sculptor.adapters.base import RolloutResult

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mean_len = type(self).quality.get(self._marker, 50.0)
        (out / "behavior.json").write_text(json.dumps({
            "n_episodes": n_episodes,
            "mean_episode_length": mean_len,
            "max_episode_length": 500,
            "max_episode_steps": 500,
            "step_dt": 0.02,
            "rollout_num_envs": 64,
        }))
        return RolloutResult(
            video_path=out / "rollout.mp4", keyframes_dir=out / "kf",
            trajectory_path=out / "trajectory.npz", n_episodes=n_episodes,
        )

    def compute_behavior_metrics(self, rollout):
        return {}


@pytest.fixture(autouse=True)
def _reset_stub():
    _EurekaStubAdapter.quality = {}
    _EurekaStubAdapter.instances = []
    yield


def _job_setup(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.toml").write_text(
        '[adapter]\n'
        'class = "tests.test_eureka_baseline._EurekaStubAdapter"\n'
        'config = { task_id = "T" }\n'
    )
    return job


# ── core algorithm ───────────────────────────────────────────────────


def test_eureka_selects_best_candidate_and_reflects(tmp_path: Path) -> None:
    """Gen 0: candidates A (weak) and B (strong) → B selected, mirrored
    to runs/iter_0, and gen 1's prompt carries B's source + reflection."""
    _EurekaStubAdapter.quality = {"A": 100.0, "B": 400.0,
                                  "C": 150.0, "D": 200.0}
    client = _StubLLM(
        _candidate("A"), _candidate("B"),   # gen 0
        _candidate("C"), _candidate("D"),   # gen 1
    )
    job = _job_setup(tmp_path)
    summary = run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="balance the pole", spec_metric="cartpole_balance",
        generations=2, k=2, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    assert summary == {
        "generations": 2, "candidates_sampled": 4, "candidates_trained": 4,
    }
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    g0, g1 = log["generations"]
    assert g0["best_candidate"] == 1                 # B
    assert g0["best_fitness"] == pytest.approx(0.8)  # 400/500
    assert g1["best_fitness"] == pytest.approx(0.4)  # D = 200/500

    # Mirrors: iter_0 = B's artifacts, iter_1 = D's (per-gen best).
    m0 = json.loads((job / "runs" / "iter_0" / "eureka_candidate.json").read_text())
    assert m0["candidate"] == 1 and m0["fitness"] == pytest.approx(0.8)
    assert (job / "runs" / "iter_0" / "rollout" / "behavior.json").is_file()

    # Gen 1 prompt carried gen 0's best source + reflection.
    gen1_user = client.messages.calls[2]["messages"][0]["content"]
    payload = json.loads(gen1_user)
    assert "candidate B" in payload["previous_best_reward_source"]
    refl = payload["reward_reflection"]
    assert refl["fitness_spec_score"] == pytest.approx(0.8)
    # H1: the reflection carries the SERIES (Appendix-F shape), not
    # just means — saturation is undetectable from a mean.
    tt = refl["reward_components_over_training"]["task_term"]
    assert tt["points"] == [0.1, 0.5]
    assert tt["mean"] == pytest.approx(0.3)
    assert tt["max"] == pytest.approx(0.5)
    # Gen 0 prompt had no prior.
    gen0_payload = json.loads(client.messages.calls[0]["messages"][0]["content"])
    assert gen0_payload["previous_best_reward_source"] is None
    assert gen0_payload["state_schema"] == {"qpos": [2], "qvel": [2]}
    # Eureka sampling temperature pinned at 1.0.
    assert all(c["temperature"] == 1.0 for c in client.messages.calls)


def test_eureka_invalid_candidate_resampled_then_zero(tmp_path: Path) -> None:
    """A probe-failing candidate is resampled ONCE; two failures score
    an honest zero without training."""
    _EurekaStubAdapter.quality = {"OK": 250.0}
    client = _StubLLM(
        _broken_candidate("X"),                        # cand 0, attempt 1
        _broken_candidate("Y"),                        # cand 0, attempt 2
        _candidate("OK"),                              # cand 1
    )
    job = _job_setup(tmp_path)
    summary = run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=1, k=2, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    assert summary["candidates_sampled"] == 3
    assert summary["candidates_trained"] == 1
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    cands = log["generations"][0]["candidates"]
    assert cands[0]["valid"] is False
    assert cands[0]["fitness"] == 0.0
    # L1: per-attempt audit records, no stale single error field.
    assert len(cands[0]["attempts"]) == 2
    assert all("probe failure" in a["probe_error"]
               for a in cands[0]["attempts"])
    assert cands[1]["fitness"] == pytest.approx(0.5)
    assert log["generations"][0]["best_candidate"] == 1
    # Audit-trail property: candidate sources never land in the log
    # (they live on disk at gen_*/cand_*/reward.py).
    assert "source" not in json.dumps(log)


def test_eureka_best_overall_survives_weaker_generation(tmp_path: Path) -> None:
    """The reflection target is the best ACROSS generations: a weaker
    gen 1 must still reflect gen 0's champion in gen 2's prompt."""
    _EurekaStubAdapter.quality = {"A": 450.0, "B": 100.0, "C": 120.0}
    client = _StubLLM(_candidate("A"), _candidate("B"), _candidate("C"))
    job = _job_setup(tmp_path)
    run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=3, k=1, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    gen2_payload = json.loads(client.messages.calls[2]["messages"][0]["content"])
    assert "candidate A" in gen2_payload["previous_best_reward_source"]
    assert gen2_payload["reward_reflection"]["fitness_spec_score"] == pytest.approx(0.9)


def test_reflection_first_generation_shape() -> None:
    assert "first generation" in _reward_reflection(None, None)["note"]


def test_reflection_fitness_only_when_artifacts_missing() -> None:
    """L2: a champion whose artifacts are gone gets a fitness-only
    reflection — never 'no prior candidates' next to a source."""
    refl = _reward_reflection({"fitness": 0.7, "spec": {"x": 1.0}}, None)
    assert refl["fitness_spec_score"] == pytest.approx(0.7)
    assert "no prior candidates" not in refl.get("note", "")


def test_invalid_then_valid_attempt_records(tmp_path: Path) -> None:
    """L1: invalid first attempt + valid resample → attempts show the
    failure then the success; no stale error beside valid=true."""
    _EurekaStubAdapter.quality = {"OK": 300.0}
    client = _StubLLM(
        _broken_candidate("Z"),
        _candidate("OK"),
    )
    job = _job_setup(tmp_path)
    run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=1, k=1, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    cand = log["generations"][0]["candidates"][0]
    assert cand["valid"] is True
    assert cand["fitness"] == pytest.approx(0.6)
    assert "probe_error" in cand["attempts"][0]
    assert cand["attempts"][1] == {"attempt": 1, "ok": True}


def test_api_failure_is_invalid_candidate_not_dead_job(tmp_path: Path) -> None:
    """H2: an LLM API exception mid-generation must not crash the job —
    the candidate scores zero with the error recorded, the log is
    flushed, and trained generations keep their GPU accounting."""

    class _FlakyMessages(_CreateMessages):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            nxt = self._responses.pop(0)
            if nxt == "RAISE":
                raise RuntimeError("api exploded")
            return _Resp(nxt)

    client = _StubLLM()
    client.messages = _FlakyMessages(["RAISE", "RAISE", _candidate("OK")])
    _EurekaStubAdapter.quality = {"OK": 300.0}
    job = _job_setup(tmp_path)
    summary = run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=1, k=2, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    assert summary["candidates_trained"] == 1
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    cands = log["generations"][0]["candidates"]
    assert cands[0]["valid"] is False
    assert all("api exploded" in a["api_error"] for a in cands[0]["attempts"])
    assert cands[1]["fitness"] == pytest.approx(0.6)


def test_train_crash_counts_untrained_with_error(tmp_path: Path) -> None:
    """M5/accounting: probe passes but train raises → trained=False,
    error recorded, fitness 0, excluded from the GPU bill."""

    class _TrainCrashAdapter(_EurekaStubAdapter):
        def train(self, reward_module_path, output_dir, steps, seed, **kw):
            raise RuntimeError("OOM on real shapes")

    job = tmp_path / "job"
    job.mkdir()
    (job / "config.toml").write_text(
        '[adapter]\n'
        'class = "tests.test_eureka_baseline._TrainCrashAdapter"\n'
        'config = { task_id = "T" }\n'
    )
    globals()["_TrainCrashAdapter"] = _TrainCrashAdapter
    client = _StubLLM(_candidate("A"))
    summary = run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=1, k=1, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    assert summary["candidates_trained"] == 0
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    cand = log["generations"][0]["candidates"][0]
    assert cand["trained"] is False
    assert "OOM" in cand["error"]
    assert cand["fitness"] == 0.0


def test_missing_batched_fn_caught_statically(tmp_path: Path) -> None:
    """M5: the probe only exercises the scalar path — a candidate
    without compute_reward_batched must fail the STATIC gate, not burn
    a training run."""
    no_batched = (
        "```python\n# candidate NB\nREWARD_SPEC = {}\n"
        "def compute_reward(s, a, n, i):\n    return 0.0, {}\n```"
    )
    client = _StubLLM(no_batched, no_batched)
    job = _job_setup(tmp_path)
    summary = run_eureka_job(
        config_path=job / "config.toml", job_dir=job,
        behavior_goal="g", spec_metric="cartpole_balance",
        generations=1, k=1, steps_per_iter=10, rollout_episodes=2,
        seed=7, client=client,
    )
    assert summary["candidates_trained"] == 0
    log = json.loads((job / "eureka_log.json").read_text(encoding="utf-8"))
    attempts = log["generations"][0]["candidates"][0]["attempts"]
    assert all("compute_reward_batched" in a["static_error"] for a in attempts)


# ── harness integration ──────────────────────────────────────────────


def test_harness_runs_eureka_condition(tmp_path: Path, monkeypatch) -> None:
    """mode='eureka' dispatches through _run_job with the campaign's
    generations=iterations and k=eureka_k, and total_rl_iterations
    reflects actually-trained candidates."""
    from sculptor.eval import CampaignConfig, run_campaign

    _EurekaStubAdapter.quality = {"A": 400.0, "B": 100.0}
    samples = iter([_candidate("A"), _candidate("B")] * 10)
    real_run_eureka_job = run_eureka_job   # capture BEFORE patching

    def fake_run_eureka_job(**kw):
        # Run the REAL function with a stub client built per call.
        client = _StubLLM(*[next(samples) for _ in range(kw["generations"] * kw["k"])])
        return real_run_eureka_job(**{**kw, "client": client})

    # _run_job imports inside the function — patch the source module.
    monkeypatch.setattr(
        "sculptor.eval.eureka.run_eureka_job", fake_run_eureka_job,
    )

    cfg = CampaignConfig(
        name="e3", out_dir=tmp_path / "c",
        benchmarks=["cartpole_balance"], conditions=["eureka"],
        seeds=[1000], iterations=1, steps_per_iter=10, eureka_k=2,
        adapter_class_override="tests.test_eureka_baseline._EurekaStubAdapter",
    )
    report = run_campaign(cfg)
    job = report["jobs"][0]
    assert job["condition"] == "eureka"
    assert job["final_spec_score"] == pytest.approx(0.8)   # best of A/B
    assert job["eureka"]["candidates_trained"] == 2
    assert job["total_rl_iterations"] == 20                # 2 trained × 10
    assert job["error"] is None


def test_harness_eureka_final_is_best_across_generations(
    tmp_path: Path, monkeypatch,
) -> None:
    """C1 regression: a regressing last generation must NOT drag the
    headline down — Eureka's defined output (Algorithm 1) is the best
    across generations."""
    from sculptor.eval import CampaignConfig, run_campaign

    _EurekaStubAdapter.quality = {"A": 400.0, "B": 100.0}
    samples = iter([_candidate("A"), _candidate("B")])
    real_run = run_eureka_job

    def fake(**kw):
        client = _StubLLM(*[next(samples)
                            for _ in range(kw["generations"] * kw["k"])])
        return real_run(**{**kw, "client": client})

    monkeypatch.setattr("sculptor.eval.eureka.run_eureka_job", fake)
    cfg = CampaignConfig(
        name="e3", out_dir=tmp_path / "c",
        benchmarks=["cartpole_balance"], conditions=["eureka"],
        seeds=[1000], iterations=2, steps_per_iter=10, eureka_k=1,
        adapter_class_override="tests.test_eureka_baseline._EurekaStubAdapter",
    )
    report = run_campaign(cfg)
    job = report["jobs"][0]
    # Series: gen0 = 0.8, gen1 = 0.2 (regression) → final = best = 0.8.
    scores = [round(s["spec_score"], 3) for s in
              json.loads((tmp_path / "c" / "cartpole_balance" / "eureka"
                          / "seed_1000" / "result.json"
                          ).read_text())["spec_series"]]
    assert scores == [0.8, 0.2]
    assert job["final_spec_score"] == pytest.approx(0.8)
    assert job["final_rule"] == "best_across_generations"
    assert job["best_spec_score"] == pytest.approx(0.8)


def test_harness_eureka_crash_recovers_gpu_accounting(
    tmp_path: Path, monkeypatch,
) -> None:
    """H2 regression: a job that crashes AFTER training generations
    must still bill the trained candidates (recovered from the
    per-generation log flush), not read total_rl_iterations=0."""
    from sculptor.eval import CampaignConfig, run_campaign

    _EurekaStubAdapter.quality = {"A": 400.0}
    real_run = run_eureka_job

    def crash_after_first_gen(**kw):
        client = _StubLLM(_candidate("A"))
        real_run(**{**kw, "generations": 1, "client": client})
        raise RuntimeError("synthetic crash between generations")

    monkeypatch.setattr(
        "sculptor.eval.eureka.run_eureka_job", crash_after_first_gen,
    )
    cfg = CampaignConfig(
        name="e3", out_dir=tmp_path / "c",
        benchmarks=["cartpole_balance"], conditions=["eureka"],
        seeds=[1000], iterations=2, steps_per_iter=10, eureka_k=1,
        adapter_class_override="tests.test_eureka_baseline._EurekaStubAdapter",
    )
    report = run_campaign(cfg)
    job = report["jobs"][0]
    assert "synthetic crash" in job["error"]
    assert job["total_rl_iterations"] == 10   # 1 trained × 10, from the log
    assert job["final_spec_score"] == pytest.approx(0.8)  # gen 0 still counts


def test_all_candidates_invalid_generation_leaves_indexed_hole(
    tmp_path: Path,
) -> None:
    """M3: a generation whose candidates all fail produces no mirror;
    iterations-to-threshold must use the REAL iter index of the later
    success, not its position in the shortened series."""
    from sculptor.eval import CampaignConfig, run_campaign
    import sculptor.eval.eureka as eureka_mod

    _EurekaStubAdapter.quality = {"OK": 400.0}
    real_run = run_eureka_job

    def fake(**kw):
        client = _StubLLM(
            _broken_candidate("X"), _broken_candidate("Y"),  # gen 0 dead
            _candidate("OK"),                                 # gen 1
        )
        return real_run(**{**kw, "client": client})

    import unittest.mock as mock
    with mock.patch.object(eureka_mod, "run_eureka_job", fake):
        cfg = CampaignConfig(
            name="e3", out_dir=tmp_path / "c",
            benchmarks=["cartpole_balance"], conditions=["eureka"],
            seeds=[1000], iterations=2, steps_per_iter=10, eureka_k=1,
            adapter_class_override=_STUB_PATH,
        )
        report = run_campaign(cfg)
    job = report["jobs"][0]
    # Only iter_1 exists; the threshold hit is at REAL iter index 1 → 2.
    assert job["iterations_completed"] == 1
    assert job["iters_to_threshold"] == 2
    assert job["final_spec_score"] == pytest.approx(0.8)


_STUB_PATH = "tests.test_eureka_baseline._EurekaStubAdapter"
