"""tests/test_mission_run.py — Ship 16 mission orchestrator.

Two distinct layers tested here:

1. **mission_runtime** (pure functions, no sculptor state):
   - `_evaluate_success_criterion` — safe-eval contract: namespace grounding,
     __builtins__ isolation, attribute-access allow-list, unsafe node types.
   - `_build_criterion_namespace` — loading behavior.json + trajectory.npz
     into the namespace shape the evaluator consumes.

2. **mission_run orchestrator** (integration flow):
   - Happy path: 2-stage mission, both succeed.
   - Failure path: stage 1 fails criterion → mission halts at stage 1.
   - Warm-start: stage 1 receives parent's final_policy_path.
   - Resume: pre-marked succeeded stages skip.
   - File lock contention raises clearly.
   - mission.json written atomically after each stage.

Stubs `sculpt_run` and `apply_prompt_edit` so no GPU / Anthropic calls
are made. Tests complete in <5 s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sculptor.mission import (
    Mission,
    Stage,
    save_mission,
)
from sculptor.mission_runtime import (
    BARE_IDENTIFIERS,
    BEHAVIOR_KEYS,
    CriterionEvalError,
    MissionResult,
    PERSISTED_TRAJECTORY_KEYS,
    StageResult,
    _build_criterion_namespace,
    _evaluate_success_criterion,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _fabricate_rollout_artifacts(
    iter_dir: Path,
    *,
    behavior: dict[str, Any] | None = None,
    trajectory: dict[str, np.ndarray] | None = None,
    components: dict[str, np.ndarray] | None = None,
) -> None:
    """Write minimal valid behavior.json + trajectory.npz to the iter's
    rollout/ subdir. Mirrors the mjlab runner's output shape.

    Omitted `behavior` falls back to a default set of the 4 stable keys.
    Omitted `trajectory` skips the npz file entirely (evaluator handles
    missing npz gracefully — see _load_trajectory_arrays).
    """
    rollout = iter_dir / "rollout"
    rollout.mkdir(parents=True, exist_ok=True)
    beh = behavior if behavior is not None else {
        "n_episodes": 4,
        "mean_return": 0.5,
        "mean_episode_length": 300.0,
        "max_episode_length": 500,
    }
    (rollout / "behavior.json").write_text(json.dumps(beh))

    if trajectory is None and components is None:
        return
    payload: dict[str, np.ndarray] = {}
    if trajectory:
        for k, v in trajectory.items():
            payload[k] = v
    if components:
        for name, arr in components.items():
            payload[f"reward_term__{name}"] = arr
    np.savez(rollout / "trajectory.npz", **payload)


# ── 1. Criterion evaluator: happy paths ──────────────────────────────
def test_criterion_true_on_simple_metric_threshold():
    ns = {"metric": 0.8}
    # Math helpers must be in namespace for Name-node validation to pass.
    ns.update({"abs": abs, "min": min, "max": max})
    assert _evaluate_success_criterion("metric > 0.5", ns) is True


def test_criterion_false_on_threshold_miss():
    ns = {"metric": 0.3, "abs": abs, "min": min, "max": max}
    assert _evaluate_success_criterion("metric > 0.5", ns) is False


def test_criterion_accesses_behavior_subscript():
    ns = {
        "metric": None,
        "behavior": {"mean_return": 0.9, "mean_episode_length": 520.0},
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    }
    assert _evaluate_success_criterion(
        "behavior['mean_return'] > 0.7 and "
        "behavior['mean_episode_length'] > 500",
        ns,
    ) is True


def test_criterion_accesses_components_subscript():
    ns = {
        "metric": None, "components": {"support_phase": 0.42},
        "abs": abs, "min": min, "max": max,
    }
    assert _evaluate_success_criterion(
        "components['support_phase'] > 0.4", ns,
    ) is True


def test_criterion_accesses_trajectory_array_methods():
    """Ship-16 design: allow `.mean()` / `.float()` / `.any()` etc.
    on persisted numpy arrays via the attribute allow-list."""
    ns = {
        "metric": None,
        "trajectory": {
            "root_link_pos_w": np.array([[0.7, 0.7, 0.7], [0.5, 0.5, 0.5]]),
        },
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "float": float,
    }
    # Mean over the array — 0.6 > 0.5
    assert _evaluate_success_criterion(
        "trajectory['root_link_pos_w'].mean() > 0.5", ns,
    ) is True


def test_criterion_info_alias_resolves_to_trajectory():
    """`info[<key>]` is an alias for `trajectory[<key>]` (Ship-14 prompt
    compatibility). Same underlying dict."""
    traj = {"rewards": np.array([1.0, 2.0, 3.0])}
    ns = {
        "metric": None, "trajectory": traj, "info": traj,
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    }
    assert _evaluate_success_criterion(
        "info['rewards'].mean() > 1.0", ns,
    ) is True


# ── 2. Criterion evaluator: safety / rejection paths ─────────────────
def test_criterion_rejects_unparseable_expression():
    with pytest.raises(CriterionEvalError, match="not a valid Python"):
        _evaluate_success_criterion("metric > (", {"metric": 0.5})


def test_criterion_rejects_unknown_bare_name():
    ns = {"metric": 0.5, "abs": abs}
    with pytest.raises(CriterionEvalError, match="unknown identifier"):
        _evaluate_success_criterion("fnord > 0.5", ns)


def test_criterion_rejects_dangerous_attribute():
    """`.tobytes()` / `.view()` / `.__reduce__()` would let a malicious
    criterion exfiltrate or execute code. The allow-list blocks them."""
    ns = {
        "metric": None,
        "trajectory": {"rewards": np.array([1.0])},
        "abs": abs, "min": min, "max": max,
    }
    with pytest.raises(CriterionEvalError, match="disallowed attribute"):
        _evaluate_success_criterion(
            "trajectory['rewards'].tobytes()", ns,
        )


def test_criterion_rejects_builtins_access():
    """`__builtins__` must be empty so `__import__`, `open`, `eval`,
    etc. are unreachable even via name resolution."""
    ns = {"metric": None, "abs": abs}
    # Even if Claude tried `__import__`, the Name check fires first.
    with pytest.raises(CriterionEvalError, match="unknown identifier"):
        _evaluate_success_criterion("__import__('os')", ns)


def test_criterion_rejects_lambda_node():
    """Lambdas could define arbitrary code paths; disallow at AST level.
    Post-audit: the rejection message uses "explicitly-rejected" prefix
    (Lambda is on REJECTED_NODES). Test pinned to that prefix."""
    ns = {"metric": 0.5}
    with pytest.raises(
        CriterionEvalError, match=r"(disallowed|explicitly-rejected) AST node",
    ):
        _evaluate_success_criterion("(lambda x: x)(metric)", ns)


def test_criterion_rejects_non_bool_coercion_failure():
    """A criterion that returns something boolean-incoercible should
    raise clearly (e.g., numpy.ndarray with multiple truthy values)."""
    ns = {
        "metric": None,
        "trajectory": {"rewards": np.array([1.0, 2.0, 3.0])},
        "abs": abs, "min": min, "max": max,
    }
    # numpy raises "truth value of an array is ambiguous" on this.
    with pytest.raises(CriterionEvalError):
        _evaluate_success_criterion("trajectory['rewards']", ns)


def test_criterion_rejects_call_to_unknown_function():
    """Function calls to names not in the namespace are blocked at
    the ast.Name check (before the call would execute)."""
    ns = {"metric": 0.5}
    with pytest.raises(CriterionEvalError, match="unknown identifier"):
        _evaluate_success_criterion("sketchy_func(metric)", ns)


# ── 3. Namespace construction from on-disk artifacts ─────────────────
def test_build_namespace_loads_behavior_json(tmp_path: Path):
    iter_dir = tmp_path / "iter_0"
    _fabricate_rollout_artifacts(iter_dir)
    ns = _build_criterion_namespace(iter_dir, primary_metric=0.42)
    assert ns["metric"] == 0.42
    assert ns["behavior"]["mean_return"] == 0.5
    assert ns["behavior"]["mean_episode_length"] == 300.0


def test_build_namespace_exposes_components(tmp_path: Path):
    iter_dir = tmp_path / "iter_0"
    _fabricate_rollout_artifacts(
        iter_dir,
        components={
            "support_phase": np.array([0.3, 0.4, 0.5]),  # mean=0.4
            "kick_swing": np.array([1.0, 2.0]),  # mean=1.5
        },
    )
    ns = _build_criterion_namespace(iter_dir, primary_metric=0.0)
    assert ns["components"]["support_phase"] == pytest.approx(0.4)
    assert ns["components"]["kick_swing"] == pytest.approx(1.5)


def test_build_namespace_info_is_alias_for_trajectory(tmp_path: Path):
    iter_dir = tmp_path / "iter_0"
    _fabricate_rollout_artifacts(
        iter_dir,
        trajectory={"rewards": np.array([1.0, 2.0])},
    )
    ns = _build_criterion_namespace(iter_dir, primary_metric=0.0)
    assert ns["info"] is ns["trajectory"]
    assert list(ns["info"].keys()) == ["rewards"]


def test_build_namespace_raises_on_missing_behavior_json(tmp_path: Path):
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    with pytest.raises(CriterionEvalError, match="no behavior.json"):
        _build_criterion_namespace(iter_dir, primary_metric=0.0)


def test_build_namespace_drops_unexpected_trajectory_keys(tmp_path: Path):
    """Keys that aren't in PERSISTED_TRAJECTORY_KEYS + don't match
    `reward_term__<name>` pattern are silently ignored — a future
    adapter adding a new trajectory key doesn't crash the evaluator."""
    iter_dir = tmp_path / "iter_0"
    _fabricate_rollout_artifacts(
        iter_dir,
        trajectory={
            "rewards": np.array([1.0]),       # persisted-allowed
            "future_feature": np.array([9.9]),  # not in PERSISTED set
        },
    )
    ns = _build_criterion_namespace(iter_dir, primary_metric=0.0)
    assert "rewards" in ns["trajectory"]
    assert "future_feature" not in ns["trajectory"]


# ── 4. Ship-16 mission_run orchestrator ─────────────────────────────
def _make_mission(tmp_path: Path, n_stages: int = 2) -> Mission:
    """Build a Mission with N stages pre-saved to disk PLUS pre-
    scaffolded stage dirs so `sculpt_init` is skipped via
    `_is_stage_scaffolded`. Stage names: stage_0, stage_1, ..."""
    stages = []
    for i in range(n_stages):
        stages.append(Stage(
            name=f"stage_{i}",
            goal_text=f"do step {i}",
            success_criterion="metric > 0.5",
            max_iterations=2,
            parent_stage=f"stage_{i-1}" if i > 0 else None,
            reward_seed_prompt=f"seed for stage {i}",
        ))
    m = Mission(
        goal="multi-stage test",
        stages=stages,
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="test",
    )
    mission_dir = tmp_path / "mission"
    save_mission(m, mission_dir)
    m.mission_dir = str(mission_dir.resolve())

    # Pre-scaffold each stage dir so `mission_run` skips sculpt_init.
    # The config.toml here is intentionally minimal; tests that need
    # real adapter behavior stub `load_adapter` separately.
    for stage in stages:
        stage_dir = mission_dir / "stages" / stage.name
        (stage_dir / "rewards").mkdir(parents=True, exist_ok=True)
        (stage_dir / "runs").mkdir(exist_ok=True)
        (stage_dir / "reports").mkdir(exist_ok=True)
        (stage_dir / "config.toml").write_text(
            '[target]\nname = "x"\n'
            '[adapter]\nclass = "stubbed"\nconfig = {}\n'
            '[iteration]\n'
        )
        (stage_dir / "rewards" / "__init__.py").write_text("")
        (stage_dir / "rewards" / "v0.py").write_text(
            "REWARD_SPEC={'version':'v0','hyperparameters':{},'references':[]}\n"
            "def compute_reward(s,a,n,i): return 0.0, {}\n"
        )
    return m


class _FakeContract:
    """Stand-in for RewardContract. `apply_prompt_edit` only reads
    `expected_info_keys` / `expected_components` from it at validation
    time — we don't need a real one for stubbed tests."""
    expected_info_keys = []
    expected_components = None
    supports_batched = False
    training_device = "any"
    min_gpu_memory_gb = None
    state_schema = None


class _FakeAdapter:
    def reward_contract(self):
        return _FakeContract()


def _stub_load_adapter(_config_path):
    return _FakeAdapter()


@pytest.fixture
def stub_adapter(monkeypatch):
    """Auto-applied to mission_run tests — the orchestrator's
    `load_adapter(stage_dir / 'config.toml')` call would otherwise fail
    on the minimal test config that doesn't reference a real adapter.
    """
    monkeypatch.setattr(
        "sculptor.adapters.base.load_adapter", _stub_load_adapter,
    )


def _fake_sculpt_run_factory(
    *, metric: float, write_ckpt: bool = True,
    write_trajectory: bool = True,
):
    """Build a fake `sculpt_run` that writes the artifacts
    `_run_one_stage` looks for post-training (checkpoint, behavior.json,
    trajectory.npz) to a per-call iter_dir.
    """
    def fake(*, config_path, behavior_goal, iterations=3, steps_per_iter=None,
             seed=None, init_policy_path=None, **_kw):
        project = Path(config_path).parent
        iter_dir = project / "runs" / f"iter_{iterations}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        if write_ckpt:
            (iter_dir / "checkpoint.pt").write_bytes(b"fake-ckpt")
        if write_trajectory:
            _fabricate_rollout_artifacts(
                iter_dir,
                behavior={"n_episodes": 1, "mean_return": metric,
                          "mean_episode_length": 400.0,
                          "max_episode_length": 500},
            )
        # Return a SculptRunResult-shaped object.
        from sculptor.sculpt import IterOutcome, SculptRunResult
        outcome = IterOutcome(
            iter_index=iterations,
            iter_dir=iter_dir,
            reward_path_before=project / "rewards" / "v1.py",
            reward_path_after=project / "rewards" / f"v{iterations}.py",
            primary_metric=metric,
            behavior={"mean_return": metric},
            failure_modes=[],
            edit_count=0,
        )
        return SculptRunResult(
            iterations_run=iterations,
            completed_iters=[outcome],
            primary_metric_history=[metric],
        )
    return fake


def _stub_apply_prompt_edit(*_a, **kw):
    """Fake `apply_prompt_edit` that just writes v1.py to the caller's
    rewards_dir without hitting Claude."""
    current = Path(kw["current_reward_path"])
    new_iter = kw["new_iter_id"]
    new_path = current.parent / f"{new_iter}.py"
    new_path.write_text(
        "REWARD_SPEC = {'version':'v1','hyperparameters':{},'references':[]}\n"
        "def compute_reward(s,a,n,i): return 0.0, {}\n"
    )
    return new_path


def test_mission_run_happy_path_two_stages_both_succeed(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """2-stage mission, metric > criterion threshold on both → both
    succeed, mission_completed fires, final mission.json has both
    stages status='succeeded'."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)

    fake = _fake_sculpt_run_factory(metric=0.9)  # > 0.5 criterion
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", kg_store=None, on_event=events.append,
    )

    assert result.completed is True
    assert result.halted_at_stage is None
    assert len(result.stage_results) == 2
    assert all(sr.status == "succeeded" for sr in result.stage_results)
    # Mission events in order.
    types = [e["type"] for e in events]
    assert types[0] == "mission_started"
    assert types[-1] == "mission_completed"
    assert "stage_succeeded" in types


def test_mission_run_halts_when_stage_criterion_fails(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Metric = 0.3 on both stages → stage_0 fails criterion (0.3 !> 0.5)
    → mission halts. Ship 17 update: pre-set `redecomposition_attempts=1`
    on stage_0 so the orchestrator skips Ship 17's re-decomposition path
    (budget already exhausted) and tests the bare halt behavior. The
    Ship 17 redecomposition flow has its own dedicated tests below."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    # Ship 17: budget pre-exhausted → no redecomposition attempted.
    m.stages[0].redecomposition_attempts = 1
    fake = _fake_sculpt_run_factory(metric=0.3)  # < 0.5 criterion
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    assert result.completed is False
    assert result.halted_at_stage == "stage_0"
    assert result.halted_reason == "criterion_not_met"
    # Only stage 0 ran.
    assert len(result.stage_results) == 1
    assert result.stage_results[0].status == "failed"
    types = [e["type"] for e in events]
    assert "stage_failed" in types
    assert "mission_halted" in types
    # Ship 17: redecomposition_skipped emitted (budget_exhausted).
    skipped = [
        e for e in events
        if e.get("type") == "redecomposition_skipped"
        and e.get("reason") == "budget_exhausted"
    ]
    assert len(skipped) == 1
    # stage_1 did not start.
    started = [
        e for e in events
        if e.get("type") == "stage_started" and e.get("stage_name") == "stage_1"
    ]
    assert started == []


def test_extract_toml_section_returns_section_body():
    """Unit test for the helper underpinning _inherit_parent_adapter_config."""
    from sculptor.sculpt import _extract_toml_section
    text = (
        '[target]\nname = "x"\n\n'
        '[adapter]\nclass = "A"\nconfig = { task_id = "T" }\n\n'
        '[kg]\nseeds_path = "y"\n'
    )
    body = _extract_toml_section(text, "adapter")
    assert body is not None
    assert 'class = "A"' in body
    assert 'task_id = "T"' in body
    # Body excludes the section header AND the next header.
    assert "[adapter]" not in body
    assert "[kg]" not in body


def test_replace_toml_section_substitutes_body():
    from sculptor.sculpt import _replace_toml_section
    text = (
        '[target]\nname = "x"\n\n'
        '[adapter]\nclass = "OLD"\nconfig = { env_id = "CHANGE_ME" }\n\n'
        '[kg]\nseeds_path = "y"\n'
    )
    new_body = 'class = "NEW"\nconfig = { task_id = "T" }\n\n'
    out = _replace_toml_section(text, "adapter", new_body)
    assert 'class = "NEW"' in out
    assert 'class = "OLD"' not in out
    assert 'task_id = "T"' in out
    assert 'env_id = "CHANGE_ME"' not in out
    # Sibling sections are untouched.
    assert 'name = "x"' in out
    assert 'seeds_path = "y"' in out


def test_inherit_parent_adapter_config_replaces_stage_config(tmp_path: Path):
    """Integration: a freshly sculpt_init-scaffolded stage gets the
    parent project's [adapter] section so MjlabAdapter doesn't see
    the gym_sb3-flavored env_id="CHANGE_ME" template (Ship 19
    hotfix; verified-against-real-bug regression)."""
    from sculptor.sculpt import _inherit_parent_adapter_config

    project_dir = tmp_path / "proj"
    stage_dir = tmp_path / "proj" / ".missions" / "m" / "stages" / "s0"
    project_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)

    # Parent's config: mjlab with task_id (the correct shape).
    (project_dir / "config.toml").write_text(
        '[target]\nname = "p"\n\n'
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Cartpole-Balance", num_envs = 1024, device = "cuda:0" }\n\n'
        '[kg]\nseeds_path = "kg_seeds.yml"\n'
    )
    # Stage's config: the gym_sb3-flavored sculpt_init template.
    (stage_dir / "config.toml").write_text(
        '[target]\nname = "s0"\n\n'
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { env_id = "CHANGE_ME", n_envs = 4, ppo_kwargs = { learning_rate = 3e-4 } }\n\n'
        '[kg]\nseeds_path = "kg_seeds.yml"\n'
    )

    changed = _inherit_parent_adapter_config(
        stage_dir=stage_dir, project_dir=project_dir,
    )
    assert changed is True
    final = (stage_dir / "config.toml").read_text()
    # The parent's adapter config wins.
    assert 'task_id = "Mjlab-Cartpole-Balance"' in final
    assert 'num_envs = 1024' in final
    # The broken gym_sb3 keys are gone.
    assert 'env_id = "CHANGE_ME"' not in final
    assert 'n_envs = 4' not in final
    # Stage-specific bits are preserved.
    assert 'name = "s0"' in final
    assert 'seeds_path = "kg_seeds.yml"' in final


def test_inherit_parent_adapter_config_tolerates_missing_files(tmp_path: Path):
    """No project config / no stage config → return False, do not raise."""
    from sculptor.sculpt import _inherit_parent_adapter_config
    assert _inherit_parent_adapter_config(
        stage_dir=tmp_path / "missing_stage",
        project_dir=tmp_path / "missing_project",
    ) is False


def test_mission_run_fails_stage_when_no_checkpoint_produced(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """If sculpt_run completes but no checkpoint.pt/.zip lands in the
    last iter, the stage can't warm-start a successor → mark failed."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    fake = _fake_sculpt_run_factory(metric=0.9, write_ckpt=False)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    result = sculpt_mod.mission_run(m, adapter_short_name="mjlab")
    assert result.completed is False
    assert result.stage_results[0].status == "failed"
    assert result.stage_results[0].failure_reason == "no_checkpoint"


def test_mission_run_warm_starts_stage_1_from_stage_0_final_policy(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Stage 1's sculpt_run call MUST receive `init_policy_path=<stage_0
    final ckpt>`. This is the mission's raison d'être."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    seen_init_paths: list[Any] = []

    def fake_run(**kw):
        seen_init_paths.append(kw.get("init_policy_path"))
        return _fake_sculpt_run_factory(metric=0.9)(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake_run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    sculpt_mod.mission_run(m, adapter_short_name="mjlab")

    assert len(seen_init_paths) == 2
    # First stage: no parent → None.
    assert seen_init_paths[0] is None
    # Second stage: path to stage_0's checkpoint.
    assert seen_init_paths[1] is not None
    assert Path(seen_init_paths[1]).name == "checkpoint.pt"


def test_mission_run_skips_already_succeeded_stages(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Resume: if a stage is pre-marked succeeded (from a prior partial
    run), mission_run skips it and moves to the next."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    # Pre-mark stage_0 succeeded.
    m.stages[0].status = "succeeded"
    m.stages[0].final_policy_path = str(tmp_path / "prior.pt")
    (tmp_path / "prior.pt").write_bytes(b"stub")

    seen_goals: list[str] = []

    def fake_run(*, behavior_goal, **kw):
        seen_goals.append(behavior_goal)
        return _fake_sculpt_run_factory(metric=0.9)(behavior_goal=behavior_goal, **kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake_run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    # sculpt_run called once — only for stage_1.
    assert seen_goals == ["do step 1"]
    types = [e.get("type") for e in events]
    assert "stage_skipped" in types


def test_mission_run_rejects_mission_dir_none():
    """Mission without mission_dir raises before any side effects."""
    from sculptor import sculpt as sculpt_mod

    m = Mission(
        goal="x",
        stages=[Stage(
            name="s", goal_text="x", success_criterion="True",
            max_iterations=1, parent_stage=None,
            reward_seed_prompt="alive",
        )],
        decomposition_model="x", decomposition_rationale="",
    )
    # mission_dir stays None (never called save_mission).
    with pytest.raises(RuntimeError, match="mission_dir is None"):
        sculpt_mod.mission_run(m, adapter_short_name="mjlab")


def test_mission_run_persists_mission_json_after_each_stage(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Atomic save: mission.json on disk reflects stage_0 outcome BEFORE
    stage_1 starts, so a mid-mission crash + resume picks up correctly."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    mission_dir = Path(m.mission_dir)

    mission_json_states: list[list[str]] = []

    def fake_run(**kw):
        # Called AFTER stage_0's save → read current mission.json.
        data = json.loads((mission_dir / "mission.json").read_text())
        mission_json_states.append(
            [s["status"] for s in data["stages"]]
        )
        return _fake_sculpt_run_factory(metric=0.9)(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake_run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    sculpt_mod.mission_run(m, adapter_short_name="mjlab")

    # Two sculpt_run calls — stage_0 and stage_1.
    assert len(mission_json_states) == 2
    # State visible when stage_0 is running: stage_0 = training/pending,
    # stage_1 = pending. We don't strictly check stage_0's transient
    # state since the save happens AFTER stage_0 completes.

    # Final on-disk state: both succeeded.
    final = json.loads((mission_dir / "mission.json").read_text())
    assert [s["status"] for s in final["stages"]] == ["succeeded", "succeeded"]


# ── 5. mission.py helper tests ───────────────────────────────────────
def test_mission_stage_dir_resolves_under_mission_dir(tmp_path: Path):
    m = _make_mission(tmp_path, n_stages=2)
    expected = Path(m.mission_dir) / "stages" / "stage_0"
    assert m.stage_dir("stage_0") == expected


def test_mission_stage_dir_raises_without_mission_dir():
    m = Mission(
        goal="x",
        stages=[Stage(
            name="s", goal_text="x", success_criterion="True",
            max_iterations=1, parent_stage=None, reward_seed_prompt="alive",
        )],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(RuntimeError, match="mission_dir is None"):
        m.stage_dir("s")


def test_mission_parent_checkpoint_returns_none_for_first_stage(
    tmp_path: Path,
):
    m = _make_mission(tmp_path, n_stages=2)
    # parent_stage=None → no ckpt to resolve.
    assert m.parent_checkpoint_of("stage_0") is None


def test_mission_parent_checkpoint_returns_none_when_parent_not_trained(
    tmp_path: Path,
):
    """Parent exists but has no final_policy_path set yet → None
    (first stage of a mission that hasn't run)."""
    m = _make_mission(tmp_path, n_stages=2)
    # stage_0's final_policy_path still None (default).
    assert m.parent_checkpoint_of("stage_1") is None


def test_mission_parent_checkpoint_resolves_after_parent_trained(
    tmp_path: Path,
):
    m = _make_mission(tmp_path, n_stages=2)
    ckpt = tmp_path / "stage_0_ckpt.pt"
    ckpt.write_bytes(b"stub")
    m.stages[0].final_policy_path = str(ckpt)
    resolved = m.parent_checkpoint_of("stage_1")
    assert resolved == ckpt


# ── 6. Audit-driven regression tests ────────────────────────────────
def test_criterion_allows_ellipsis_in_subscript():
    """Audit finding #1 (CRITICAL): `trajectory[..., 2]` requires
    ast.Tuple in the AST allow-list. Pre-fix this raised
    'disallowed AST node Tuple'."""
    ns = {
        "metric": None,
        "trajectory": {
            "root_link_pos_w": np.array([
                [0.5, 0.5, 0.7],
                [0.5, 0.5, 0.8],
            ]),
        },
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "float": float,
    }
    # Z-component of root_link_pos_w averages 0.75 > 0.65.
    assert _evaluate_success_criterion(
        "trajectory['root_link_pos_w'][..., 2].mean() > 0.65", ns,
    ) is True


def test_criterion_explicitly_rejects_comprehension():
    """Audit finding #1: comprehensions are explicitly rejected via
    REJECTED_NODES. Pre-fix would have fallen through to the generic
    'disallowed' message; now the error names the construct."""
    ns = {
        "metric": None,
        "trajectory": {"rewards": np.array([1.0, 2.0])},
        "abs": abs,
    }
    with pytest.raises(
        CriterionEvalError, match="explicitly-rejected.*ListComp",
    ):
        _evaluate_success_criterion(
            "[x for x in trajectory['rewards']]", ns,
        )


def test_criterion_explicitly_rejects_walrus():
    """Audit finding #1: walrus operator `:=` would let a criterion
    bind a variable for use later in the same expression. Block it
    even though current ALLOWED_NODES would also catch ast.NamedExpr."""
    ns = {"metric": 0.5, "abs": abs}
    with pytest.raises(
        CriterionEvalError, match="explicitly-rejected.*NamedExpr",
    ):
        _evaluate_success_criterion("(x := metric) > 0.4", ns)


def test_criterion_multi_element_array_yields_friendly_hint():
    """Audit finding #2 (HIGH): pre-fix, a criterion that returned
    a multi-element bool array got numpy's raw error
    ('truth value ambiguous'). Post-fix, the wrapper hints at
    .all() / .any() / .mean() reductions."""
    ns = {
        "metric": None,
        "trajectory": {"rewards": np.array([0.5, 0.5, 0.5])},
        "abs": abs, "min": min, "max": max,
    }
    with pytest.raises(
        CriterionEvalError, match=r"multi-element array.*\.all\(\).*\.any\(\)",
    ):
        _evaluate_success_criterion(
            "trajectory['rewards'] > 0.0", ns,
        )


def test_mission_run_emits_warm_start_skipped_when_parent_ckpt_deleted(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding #3 (HIGH): if a user deletes the parent's
    checkpoint between mission runs, the child stage previously
    silently degraded to cold-start. Post-fix, a `warm_start_skipped`
    event with reason='parent_ckpt_missing' fires so the user sees
    the curriculum break."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    # Pre-mark stage_0 succeeded with a checkpoint that DOESN'T exist
    # on disk (simulating user-cleanup between runs).
    m.stages[0].status = "succeeded"
    m.stages[0].final_policy_path = str(tmp_path / "DELETED_ckpt.pt")
    # Note: do NOT write the file.

    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.9),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    skipped = [
        e for e in events
        if e.get("type") == "warm_start_skipped"
        and e.get("reason") == "parent_ckpt_missing"
    ]
    assert len(skipped) == 1, (
        f"expected exactly one parent_ckpt_missing event, got {skipped!r}"
    )
    assert "stage_1" == skipped[0]["stage_name"]


def test_mission_run_detects_adapter_mismatch_on_resume(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding (Audit 1 #7): on resume with a different
    adapter_short_name than the stage was scaffolded under, the stage
    fails fast with reason='adapter_mismatch' rather than silently
    training under the on-disk adapter."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    # Replace the stub config with one that names a real adapter dotted-
    # path, simulating a previous scaffold under a different adapter.
    stage_dir = m.stage_dir("stage_0")
    (stage_dir / "config.toml").write_text(
        '[target]\nname = "x"\n'
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
        'config = {}\n[iteration]\n'
    )

    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.9),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",  # different from on-disk
        on_event=events.append,
    )
    assert result.completed is False
    assert result.stage_results[0].failure_reason == "adapter_mismatch"


def test_mission_run_stage_started_event_includes_stage_dir(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding (Audit 2 #G): UI symmetry — `stage_started` now
    carries `stage_dir` so Ship 18 doesn't have to wait for
    `stage_scaffolded` to link to the project page."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.9),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    started = [e for e in events if e.get("type") == "stage_started"]
    assert started and "stage_dir" in started[0]
    assert started[0]["stage_dir"].endswith("/stages/stage_0")


def test_mission_run_lock_release_allows_reentry(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding (Audit 2 lead): the post-fix cleanup MUST allow a
    second mission_run to acquire the lock cleanly. Pre-fix code tried
    to `unlink` the lock file inside the `finally`, which on Windows/WSL
    can silently fail (filelock holds the file handle past release).
    Post-fix drops the unlink — filelock's release() handles cleanup
    cleanly across platforms, so re-entry just works.

    What we test: two back-to-back mission_run calls succeed. Whether
    the .lock FILE persists between them is implementation-detail of
    filelock and varies by OS — what matters is the *advisory lock*
    is released.
    """
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.9),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    # First call.
    r1 = sculpt_mod.mission_run(m, adapter_short_name="mjlab")
    assert r1.completed
    # Second call — would fail with FileLockTimeout if the lock leaked.
    r2 = sculpt_mod.mission_run(m, adapter_short_name="mjlab")
    assert r2.completed


# ── 7. Ship 17: re-decomposition on stage failure ───────────────────
def _make_redecompose_response(failed_stage_name: str, n_sub: int = 3):
    """Build a canned `_RedecompositionModel` instance with N
    well-formed sub-stages. The LAST sub-stage's success_criterion is
    byte-equal to the failed stage's so `redecompose_stage`'s
    validator is satisfied."""
    from sculptor.decompose import _RedecompositionModel, _StageModel

    stages = []
    for i in range(n_sub):
        is_last = (i == n_sub - 1)
        # Sub-stage 0 has the failed stage's parent (will be
        # overridden by the orchestrator anyway, but we set what
        # Claude WOULD emit). Subsequent sub-stages reference the
        # PRIOR sub-stage's name as parent.
        if i == 0:
            parent = None  # default: failed stage was top-level
        else:
            parent = f"{failed_stage_name}__r1_{i - 1}"
        stages.append(_StageModel(
            name=f"{failed_stage_name}__r1_{i}",
            goal_text=f"sub-stage {i} of {failed_stage_name}",
            success_criterion=(
                "metric > 0.5"  # MUST match failed stage's criterion on LAST
                if is_last else "metric > 0.0"
            ),
            max_iterations=2,
            parent_stage=parent,
            reward_seed_prompt=f"simpler reward for sub-stage {i}",
            kg_seed_papers=[],
        ))
    return _RedecompositionModel(
        decomposition_rationale="test redecomp",
        stages=stages,
    )


def _stub_claude_redecompose(monkeypatch, response):
    """Patch `_parse_with_retry` so redecompose_stage doesn't hit
    Anthropic. Pass the canned `_RedecompositionModel` to return."""
    def fake_parse(client, system_prompt, user_content, *,
                   output_format=None, model=None, max_tokens=None):
        return response
    import sculptor.decompose as dmod
    monkeypatch.setattr(dmod, "_parse_with_retry", fake_parse)


def test_redecompose_replaces_failed_stage_with_substages(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Stage 0 fails criterion → Ship 17 splices 3 sub-stages in.
    Verify `mission.stages` mutated correctly and the FIRST sub-stage
    inherits the failed stage's parent."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    # Stage 0 will fail; sub-stages will succeed (metric=0.9 > 0.5).
    # We need `sculpt_run` to RETURN different metrics depending on
    # which stage is being trained. Use a simple counter — first call
    # returns metric=0.3 (fail), all subsequent return 0.9 (succeed).
    call_count = {"n": 0}

    def varied_run(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _fake_sculpt_run_factory(metric=0.3)(**kw)
        return _fake_sculpt_run_factory(metric=0.9)(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", varied_run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=3),
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    # Mission completed: sub-stages 0,1,2 + original stage_1.
    assert result.completed is True, (
        f"expected mission to complete via sub-stages; halted_reason="
        f"{result.halted_reason!r}, halted_at={result.halted_at_stage!r}"
    )
    # Mission.stages now has 3 sub-stages + stage_1 = 4.
    assert [s.name for s in m.stages] == [
        "stage_0__r1_0", "stage_0__r1_1", "stage_0__r1_2", "stage_1",
    ]
    # First sub-stage inherits stage_0's parent (None).
    assert m.stages[0].parent_stage is None
    # Linear parent chain inside the sub-stage block.
    assert m.stages[1].parent_stage == "stage_0__r1_0"
    assert m.stages[2].parent_stage == "stage_0__r1_1"
    # Downstream child (stage_1, formerly parent="stage_0") now points
    # at the LAST sub-stage.
    assert m.stages[3].parent_stage == "stage_0__r1_2"


def test_redecompose_substages_have_attempts_set_to_1(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Budget enforcement: the splice MUST set
    redecomposition_attempts=1 on each sub-stage so they can't be
    re-decomposed again (Ship 17 caps at one level)."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    # First call fails; we don't care about subsequent calls' status.
    call_count = {"n": 0}

    def run(**kw):
        call_count["n"] += 1
        return _fake_sculpt_run_factory(
            metric=0.3 if call_count["n"] == 1 else 0.9,
        )(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=2),
    )

    sculpt_mod.mission_run(m, adapter_short_name="mjlab")

    # All sub-stages have attempts=1.
    sub_stages = [s for s in m.stages if "__r1_" in s.name]
    assert len(sub_stages) == 2
    for s in sub_stages:
        assert s.redecomposition_attempts == 1, (
            f"sub-stage {s.name!r} has attempts={s.redecomposition_attempts}; "
            "Ship 17 must set 1 to prevent recursive re-decomposition."
        )


def test_redecompose_only_fires_once_per_stage(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """If a sub-stage ALSO fails its criterion, Ship 17 must NOT
    re-decompose it (budget=1 already used). Mission halts cleanly."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    # ALL calls fail (metric=0.3 < 0.5).
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.3),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    redecompose_calls = {"n": 0}
    _orig_response = _make_redecompose_response("stage_0", n_sub=2)

    def fake_parse(client, system_prompt, user_content, *,
                   output_format=None, model=None, max_tokens=None):
        redecompose_calls["n"] += 1
        return _orig_response
    import sculptor.decompose as dmod
    monkeypatch.setattr(dmod, "_parse_with_retry", fake_parse)

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    # Redecomposition called EXACTLY ONCE (for the original stage_0,
    # not for sub-stages whose attempts=1 already).
    assert redecompose_calls["n"] == 1
    assert result.completed is False
    # Halt event references the LAST sub-stage that failed.
    assert result.halted_at_stage and "__r1_" in result.halted_at_stage
    # And a "redecomposition_skipped" with reason=budget_exhausted
    # was emitted on a sub-stage.
    skipped = [
        e for e in events
        if e.get("type") == "redecomposition_skipped"
        and e.get("reason") == "budget_exhausted"
    ]
    assert len(skipped) >= 1


def test_redecompose_does_not_fire_for_infra_failures(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Ship 17 trigger condition: only `criterion_not_met` is
    re-decomposable. Other failure_reasons (no_checkpoint,
    training_errored, ...) signal env/code issues and should halt
    the mission directly without invoking Claude."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    # Make sculpt_run produce no checkpoint → fails with reason
    # "no_checkpoint", which is NOT in _REDECOMPOSABLE_REASONS.
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run",
        _fake_sculpt_run_factory(metric=0.9, write_ckpt=False),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    # If redecompose were attempted, this would fire — assert NOT.
    parse_calls = {"n": 0}

    def fake_parse(*a, **kw):
        parse_calls["n"] += 1
        raise AssertionError(
            "redecompose_stage must NOT fire for infrastructure failures"
        )
    import sculptor.decompose as dmod
    monkeypatch.setattr(dmod, "_parse_with_retry", fake_parse)

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    assert parse_calls["n"] == 0
    assert result.halted_reason == "no_checkpoint"
    skipped = [
        e for e in events
        if e.get("type") == "redecomposition_skipped"
        and e.get("reason") == "non_curriculum_failure"
    ]
    assert len(skipped) == 1


def test_redecompose_invalid_claude_response_halts_cleanly(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """If Claude returns a Mission whose validation fails (e.g., last
    sub-stage's success_criterion doesn't match the original's), the
    splice rolls back and the mission halts with a distinct reason."""
    from sculptor.decompose import _RedecompositionModel, _StageModel
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.3),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    # Claude emits sub-stages but the LAST one's criterion is NOT
    # byte-equal to the original (Ship 17 hard rule violation).
    bad_response = _RedecompositionModel(
        decomposition_rationale="bad — soft criterion",
        stages=[
            _StageModel(
                name="stage_0__r1_0",
                goal_text="precursor",
                success_criterion="metric > 0.0",
                max_iterations=2,
                parent_stage=None,
                reward_seed_prompt="alive",
                kg_seed_papers=[],
            ),
            _StageModel(
                name="stage_0__r1_1",
                goal_text="final — but wrong criterion",
                success_criterion="metric > 0.1",  # NOT == original "metric > 0.5"
                max_iterations=2,
                parent_stage="stage_0__r1_0",
                reward_seed_prompt="alive",
                kg_seed_papers=[],
            ),
        ],
    )
    _stub_claude_redecompose(monkeypatch, bad_response)

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    assert result.completed is False
    failed_events = [
        e for e in events
        if e.get("type") == "stage_redecomposition_failed"
    ]
    assert len(failed_events) == 1
    assert failed_events[0]["reason"] == "validation_failed"
    # Original stage list NOT mutated by failed splice.
    assert [s.name for s in m.stages] == ["stage_0"]


def test_redecompose_emits_stage_redecomposed_event(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """UI surface: `stage_redecomposed` event carries
    original_stage_name, sub_stage_names, and downstream-child
    repointing count so Ship 18 can update the DAG visualization."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)  # stage_1 is downstream child
    call_count = {"n": 0}

    def run(**kw):
        call_count["n"] += 1
        return _fake_sculpt_run_factory(
            metric=0.3 if call_count["n"] == 1 else 0.9,
        )(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=2),
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    redec = [e for e in events if e.get("type") == "stage_redecomposed"]
    assert len(redec) == 1
    assert redec[0]["original_stage_name"] == "stage_0"
    assert redec[0]["sub_stage_names"] == [
        "stage_0__r1_0", "stage_0__r1_1",
    ]
    # stage_1 had parent=stage_0 → re-pointed to last sub-stage.
    assert redec[0]["downstream_children_repointed"] == 1


def test_redecompose_substages_warm_start_in_chain(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Ship 17 + Ship 15 integration: each sub-stage's sculpt_run
    receives `init_policy_path` resolved to the prior sub-stage's
    final checkpoint. The first sub-stage inherits the failed stage's
    parent (here None → no warm-start). The downstream stage_1 picks
    up from the LAST sub-stage."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    # Track init_policy_path passed to each sculpt_run call.
    init_paths: list = []
    call_count = {"n": 0}

    def run(*, init_policy_path=None, **kw):
        init_paths.append(init_policy_path)
        call_count["n"] += 1
        return _fake_sculpt_run_factory(
            metric=0.3 if call_count["n"] == 1 else 0.9,
        )(init_policy_path=init_policy_path, **kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=3),
    )

    sculpt_mod.mission_run(m, adapter_short_name="mjlab")

    # Calls in order: [original stage_0 (fails), sub_0, sub_1, sub_2, stage_1].
    assert len(init_paths) == 5
    # stage_0 (original): no parent → None.
    assert init_paths[0] is None
    # sub_0: inherits stage_0's parent (None).
    assert init_paths[1] is None
    # sub_1, sub_2: warm-start from previous sub-stage's checkpoint.
    assert init_paths[2] is not None and Path(init_paths[2]).name == "checkpoint.pt"
    assert init_paths[3] is not None and Path(init_paths[3]).name == "checkpoint.pt"
    # stage_1 (downstream child): re-pointed to last sub-stage,
    # warm-starts from sub_2's checkpoint.
    assert init_paths[4] is not None and Path(init_paths[4]).name == "checkpoint.pt"


# ── 7b. Ship 17 audit-driven regression tests ──────────────────────
def test_redecompose_rolls_back_in_memory_on_save_failure(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding #A (CRITICAL): if `_atomic_save_mission` fails
    after a successful splice, the in-memory mission has the new
    sub-stages but on-disk has the OLD failed stage. Pre-fix this
    silently diverges in-memory ≠ on-disk; post-fix the in-memory
    splice is rolled back AND a `stage_redecomposition_failed` event
    with reason='save_failed' fires."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=2)
    monkeypatch.setattr(
        sculpt_mod, "sculpt_run", _fake_sculpt_run_factory(metric=0.3),
    )
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=2),
    )

    # Make `_atomic_save_mission` fail on the SECOND call (the
    # post-splice save). The first call (pre-stage save) succeeds.
    save_calls = {"n": 0}
    real_save = sculpt_mod._atomic_save_mission

    def flaky_save(mission, mdir):
        save_calls["n"] += 1
        if save_calls["n"] == 2:
            raise OSError(28, "no space left on device")
        return real_save(mission, mdir)
    monkeypatch.setattr(sculpt_mod, "_atomic_save_mission", flaky_save)

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    failed_events = [
        e for e in events
        if e.get("type") == "stage_redecomposition_failed"
        and e.get("reason") == "save_failed"
    ]
    assert len(failed_events) == 1
    # In-memory mission rolled back: stage list is still the original.
    assert [s.name for s in m.stages] == ["stage_0", "stage_1"]
    # Original downstream child still points at the failed stage's name.
    assert m.stages[1].parent_stage == "stage_0"


def test_resolve_unique_name_caps_collision_loop():
    """Audit finding #C: a pathological mission with 100+ pre-existing
    collisions on the proposed name must raise rather than spin
    forever or produce an over-length name."""
    from sculptor.decompose import (
        _MAX_COLLISION_ATTEMPTS, _resolve_unique_name,
    )
    from sculptor.mission import MissionValidationError

    base = "stage_0__r1_0"
    # Pre-populate the collision set with 200 candidates so the loop
    # MUST exhaust its cap.
    existing = {base} | {f"{base}_v{n}" for n in range(2, 200)}
    with pytest.raises(MissionValidationError, match="consecutive collisions"):
        _resolve_unique_name(base, existing)


def test_resolve_unique_name_rejects_overlong_resolved_name():
    """Audit finding #C: a base name leaving insufficient headroom
    for a `_vN` disambiguator must raise so the orchestrator never
    emits a stage name violating the 32-char regex cap."""
    from sculptor.decompose import _resolve_unique_name
    from sculptor.mission import MissionValidationError

    # Exactly 32-char base — the regex max. Adding `_v2` (3 chars)
    # would yield 35 chars, violating the cap.
    base = "a" * 32
    existing = {base}
    with pytest.raises(MissionValidationError, match="exceeds.*char.*limit"):
        _resolve_unique_name(base, existing)


def test_scan_iter_metric_history_pushes_corrupt_dirs_to_end(
    tmp_path: Path,
):
    """Audit finding #B: corrupt `iter_X_Y` dirs must sort to the END
    of the iter list, not BEFORE iter_0 (which would shadow legit
    metric history). Pre-fix `else -1` ranked them first."""
    from sculptor.sculpt import _scan_iter_metric_history
    from sculptor.mission import Mission, Stage

    # Build a fake mission + stage_dir layout with one corrupt + two
    # well-named iter dirs. Each has a behavior.json with a unique
    # mean_return so we can assert order.
    mdir = tmp_path / "mission"
    stage_dir = mdir / "stages" / "s1"
    runs = stage_dir / "runs"
    runs.mkdir(parents=True)
    for i, mean_return in [(0, 1.1), (1, 2.2)]:
        d = runs / f"iter_{i}"
        rollout = d / "rollout"
        rollout.mkdir(parents=True)
        (rollout / "behavior.json").write_text(
            json.dumps({"mean_return": mean_return}),
        )
    # Corrupt dir — non-numeric suffix.
    corrupt = runs / "iter_99_backup"
    rollout = corrupt / "rollout"
    rollout.mkdir(parents=True)
    (rollout / "behavior.json").write_text(
        json.dumps({"mean_return": 99.9}),
    )

    m = Mission(
        goal="x",
        stages=[Stage(
            name="s1", goal_text="x", success_criterion="True",
            max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
        )],
        decomposition_model="x", decomposition_rationale="",
        mission_dir=str(mdir),
    )

    history = _scan_iter_metric_history(m, m.stages[0])
    # iter_0=1.1, iter_1=2.2, then corrupt at end (99.9) — NOT before iter_0.
    assert history[0] == 1.1
    assert history[1] == 2.2
    # 99.9 is at the END (index 2), not BEFORE iter_0.
    assert history[-1] == 99.9


def test_redecompose_emits_feedback_read_degraded_when_data_missing(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Audit finding #E: when Claude is going to see partial training
    feedback (missing diagnosis.json / metric history / etc.), an
    explicit `feedback_read_degraded` event fires so the user knows
    the redecomposition's quality is degraded."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)

    # Custom fake sculpt_run that DOESN'T write trajectory.npz or
    # diagnosis.json — only a minimal behavior.json. Forces the
    # feedback to be heavily degraded.
    def fake(*, config_path, behavior_goal, iterations=3, seed=None,
             init_policy_path=None, **_kw):
        from sculptor.sculpt import IterOutcome, SculptRunResult
        project = Path(config_path).parent
        iter_dir = project / "runs" / f"iter_{iterations}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "checkpoint.pt").write_bytes(b"stub")
        # Bare-minimum behavior.json (no trajectory, no diagnosis).
        (iter_dir / "rollout").mkdir(parents=True, exist_ok=True)
        (iter_dir / "rollout" / "behavior.json").write_text(
            json.dumps({"n_episodes": 1, "mean_return": 0.3,
                        "mean_episode_length": 1.0,
                        "max_episode_length": 1}),
        )
        outcome = IterOutcome(
            iter_index=iterations, iter_dir=iter_dir,
            reward_path_before=project / "rewards" / "v1.py",
            reward_path_after=None,  # ← forces final_reward_source = ""
            primary_metric=0.3,
            behavior={"mean_return": 0.3},
            failure_modes=[], edit_count=0,
        )
        return SculptRunResult(
            iterations_run=iterations, completed_iters=[outcome],
            primary_metric_history=[0.3],
        )

    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=2),
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab", on_event=events.append,
    )

    degraded = [
        e for e in events if e.get("type") == "feedback_read_degraded"
    ]
    assert len(degraded) == 1
    # At least final_reward_source missing (reward_path_after=None
    # in the stub).
    assert "final_reward_source" in degraded[0]["missing_signals"]


def test_redecompose_persists_to_mission_json_atomically(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """After splice, mission.json on disk reflects the new stage list
    with current_stage_idx pointing AT the first sub-stage (so a
    crash-resume picks up correctly)."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    mission_dir = Path(m.mission_dir)

    # Capture mission.json state right after the splice. Simplest way:
    # have sub-stage 0's sculpt_run read the on-disk mission.json.
    json_after_splice: list[dict] = []
    call_count = {"n": 0}

    def run(**kw):
        call_count["n"] += 1
        if call_count["n"] == 2:  # right after splice, first sub-stage call
            data = json.loads(
                (mission_dir / "mission.json").read_text(encoding="utf-8"),
            )
            json_after_splice.append(data)
        return _fake_sculpt_run_factory(
            metric=0.3 if call_count["n"] == 1 else 0.9,
        )(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", run)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )
    _stub_claude_redecompose(
        monkeypatch, _make_redecompose_response("stage_0", n_sub=2),
    )

    sculpt_mod.mission_run(m, adapter_short_name="mjlab")

    assert len(json_after_splice) == 1
    snapshot = json_after_splice[0]
    # Stage list reflects the splice.
    names = [s["name"] for s in snapshot["stages"]]
    assert names == ["stage_0__r1_0", "stage_0__r1_1"]
    # current_stage_idx points at the first sub-stage (resume safety).
    assert snapshot["current_stage_idx"] == 0


# ── §Ship-19d Goal A + Goal B ────────────────────────────────────────

def _is_metric_still_improving():
    """Wrapper to keep `from sculptor.sculpt import _is_metric_still_improving`
    available to test bodies even if the module gets re-imported."""
    from sculptor.sculpt import _is_metric_still_improving as fn
    return fn


def test_is_metric_still_improving_detects_positive_trend():
    fn = _is_metric_still_improving()
    # Steady climb: recent_best=0.4, prior_best=0.2 → 0.4 > 0.2+max(0.01,0.05) = True
    assert fn([0.1, 0.2, 0.3, 0.4]) is True


def test_is_metric_still_improving_rejects_plateau():
    fn = _is_metric_still_improving()
    # No improvement: prior_best=0.5, recent_best=0.52 → not > 0.5+0.05 → False
    assert fn([0.5, 0.5, 0.51, 0.52]) is False


def test_is_metric_still_improving_rejects_regression():
    fn = _is_metric_still_improving()
    assert fn([0.9, 0.8, 0.7, 0.6]) is False


def test_is_metric_still_improving_handles_negative_metrics():
    fn = _is_metric_still_improving()
    # Recovering from negative: prior_best=-1.5, recent_best=-0.5
    #   abs(-1.5)*0.05=0.075, max(0.075, 0.05)=0.075
    #   -0.5 > -1.5 + 0.075 = -1.425 → True
    assert fn([-2.0, -1.5, -1.0, -0.5]) is True


def test_is_metric_still_improving_short_history_returns_false():
    fn = _is_metric_still_improving()
    # Too short for a trend test — extension should NOT fire.
    assert fn([0.4, 0.5]) is False
    assert fn([0.4, 0.5, 0.6]) is False
    # 4+ iters required.
    assert fn([0.1, 0.2, 0.3, 0.4]) is True


def _criterion_aware_sculpt_run_factory(
    *,
    iter_metrics: list[float],
    write_behavior_pass_at: list[bool],
):
    """A stub that runs `len(iter_metrics)` iters, generates synthetic
    IterOutcomes, and FIRES the `per_iter_callback` after each. If the
    callback returns a non-empty string, the loop short-circuits with
    `early_stopped=True` (matching the real sculpt_run contract).

    `write_behavior_pass_at[i]` controls whether iter `i`'s
    behavior.json's `mean_return` is over the test's criterion
    threshold (≥ 0.5). Lets us simulate "criterion satisfied at iter
    N" without spinning a real subprocess.
    """
    assert len(iter_metrics) == len(write_behavior_pass_at)

    def fake(
        *, config_path, behavior_goal, iterations=3,
        steps_per_iter=None, seed=None, init_policy_path=None,
        resume=False, per_iter_callback=None, **_kw,
    ):
        from sculptor.sculpt import IterOutcome, SculptRunResult
        project = Path(config_path).parent
        result = SculptRunResult(iterations_run=0)
        for i in range(iterations):
            metric = iter_metrics[i] if i < len(iter_metrics) else 0.0
            iter_dir = project / "runs" / f"iter_{i + 1}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "checkpoint.pt").write_bytes(b"fake-ckpt")
            # Behavior signal varies per iter so a criterion like
            # `metric > 0.5` would pass / fail depending on
            # `write_behavior_pass_at[i]`.
            beh_metric = (
                metric if write_behavior_pass_at[i] else min(metric, 0.3)
            )
            _fabricate_rollout_artifacts(
                iter_dir,
                behavior={
                    "n_episodes": 1,
                    "mean_return": beh_metric,
                    "mean_episode_length": 400.0,
                    "max_episode_length": 500,
                },
            )
            outcome = IterOutcome(
                iter_index=i + 1, iter_dir=iter_dir,
                reward_path_before=project / "rewards" / "v1.py",
                reward_path_after=project / "rewards" / f"v{i + 2}.py",
                primary_metric=metric,
                behavior={"mean_return": beh_metric},
                failure_modes=[], edit_count=0,
            )
            result.completed_iters.append(outcome)
            result.primary_metric_history.append(metric)
            result.iterations_run += 1
            if per_iter_callback is not None:
                reason = per_iter_callback(outcome)
                if reason:
                    result.early_stopped = True
                    result.early_stop_reason = str(reason)
                    return result
        return result

    return fake


def test_goal_a_early_stops_stage_when_criterion_holds(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Goal A: with `early_stop_on_criterion=True`, sculpt_run is told
    to break the loop the moment the stage's criterion holds. The
    stage's iterations_used should reflect the EARLY iter, not the
    full max_iterations budget."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 5
    m.stages[0].success_criterion = "metric > 0.5"

    # Iter 0 fails (0.3), iter 1 passes (0.9). Behavior.json mean_return
    # at iter 1 is 0.9 too — criterion `metric > 0.5` evaluated against
    # `metric` (primary_metric) is True. Loop should break at iter 1.
    fake = _criterion_aware_sculpt_run_factory(
        iter_metrics=[0.3, 0.9, 0.95, 0.96, 0.97],
        write_behavior_pass_at=[False, True, True, True, True],
    )
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append,
        early_stop_on_criterion=True,
        criterion_stability_window=1,
    )
    assert result.completed is True
    assert len(result.stage_results) == 1
    sr = result.stage_results[0]
    assert sr.status == "succeeded"
    # Loop broke at iter 2 (1-indexed) = 2 iters consumed, NOT all 5.
    assert sr.iterations_used == 2


def test_goal_a_does_not_fire_below_stability_window(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Goal A respects `criterion_stability_window` — a single-iter
    pass should NOT exit when window=2, only after TWO consecutive
    iters satisfy the criterion."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 5
    m.stages[0].success_criterion = "metric > 0.5"

    # Pass at iter 0 (0.9), fail iter 1 (0.3, regressed), pass iter 2,
    # pass iter 3. With window=2, must hold for 2 consecutive iters.
    # Stability count: iter0 (0.9 pass) → 1, iter1 (regressed) → 0,
    # iter2 (pass) → 1, iter3 (pass) → 2 → BREAK at iter 4 (1-indexed).
    fake = _criterion_aware_sculpt_run_factory(
        iter_metrics=[0.9, 0.3, 0.9, 0.95, 0.97],
        write_behavior_pass_at=[True, False, True, True, True],
    )
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        early_stop_on_criterion=True,
        criterion_stability_window=2,
    )
    assert result.stage_results[0].iterations_used == 4


def test_goal_a_off_by_default_runs_full_budget(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Backward-compat: with both flags off, mission_run preserves
    Ship 16 behavior — full max_iterations budget regardless of when
    the criterion would have passed."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 4
    m.stages[0].success_criterion = "metric > 0.5"

    # Criterion holds from iter 0; without Goal A, runs all 4 iters.
    fake = _criterion_aware_sculpt_run_factory(
        iter_metrics=[0.9, 0.95, 0.96, 0.97],
        write_behavior_pass_at=[True, True, True, True],
    )
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    result = sculpt_mod.mission_run(m, adapter_short_name="mjlab")
    assert result.stage_results[0].iterations_used == 4


def test_goal_b_extends_when_metric_still_improving(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Goal B: when the budget exhausts WITHOUT criterion satisfaction
    AND the metric is still trending up, mission_run invokes
    sculpt_run again (resume mode). Verify by counting fake-call
    invocations + watching for `stage_extended` events."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 4
    m.stages[0].success_criterion = "metric > 1.0"
    # Pre-exhaust Ship 17 redecomp so a failed stage HALTS instead
    # of triggering a Claude redecompose call (which the test stub
    # doesn't mock). Goal B tests focus on the extension path; they
    # don't exercise re-decomposition.
    m.stages[0].redecomposition_attempts = 1  # never satisfied

    # First call: monotonic climb 0.1 → 0.6. Recent half (0.4, 0.6)
    # vs prior half (0.1, 0.2) → 0.6 > 0.2 + max(0.01, 0.05) = True.
    # Extension fires.
    # Second call: similar climb but still under 1.0 → criterion fails.
    # max_extensions_per_stage=1 → bail.
    call_count = {"n": 0}
    def factory(metrics):
        return _criterion_aware_sculpt_run_factory(
            iter_metrics=metrics,
            write_behavior_pass_at=[False] * len(metrics),
        )
    first = factory([0.1, 0.2, 0.4, 0.6])
    second = factory([0.6, 0.65, 0.7, 0.72])  # stalls — no further ext

    def dispatcher(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return first(**kw)
        return second(**kw)
    monkeypatch.setattr(sculpt_mod, "sculpt_run", dispatcher)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    result = sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append,
        extend_on_improvement=True,
        max_extensions_per_stage=1,
        extension_factor=0.5,
    )
    # Two sculpt_run calls: original + 1 extension.
    assert call_count["n"] == 2
    # `stage_extended` event recorded.
    extensions = [e for e in events if e.get("type") == "stage_extended"]
    assert len(extensions) == 1
    assert extensions[0]["extension_count"] == 1


def test_goal_b_skips_extension_on_metric_plateau(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Goal B's trend test prevents wasteful extensions when the
    metric has flat-lined. Sculpt_run is called ONCE; we observe
    `stage_extension_skipped(reason="no_improvement_trend")`."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 4
    m.stages[0].success_criterion = "metric > 1.0"
    # Pre-exhaust Ship 17 redecomp so a failed stage HALTS instead
    # of triggering a Claude redecompose call (which the test stub
    # doesn't mock). Goal B tests focus on the extension path; they
    # don't exercise re-decomposition.
    m.stages[0].redecomposition_attempts = 1

    # Plateau: prior_best=0.51, recent_best=0.52 → 0.52 > 0.51 + 0.05 → False
    fake = _criterion_aware_sculpt_run_factory(
        iter_metrics=[0.5, 0.51, 0.51, 0.52],
        write_behavior_pass_at=[False, False, False, False],
    )
    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append,
        extend_on_improvement=True,
    )
    skipped = [
        e for e in events
        if e.get("type") == "stage_extension_skipped"
        and e.get("reason") == "no_improvement_trend"
    ]
    assert len(skipped) == 1


def test_goal_b_caps_at_max_extensions(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """Goal B's hard cap. With max_extensions=2, even a perpetually-
    improving stage stops after the cap is hit."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 4
    m.stages[0].success_criterion = "metric > 100.0"  # impossibly high
    m.stages[0].redecomposition_attempts = 1  # halt on fail (no redecomp)

    # Each call: monotonic climb from prior_best → prior_best + delta.
    call_count = {"n": 0}

    def dispatcher(**kw):
        call_count["n"] += 1
        # Always-improving stub. Each call adds +1.0 to the base.
        base = call_count["n"] * 1.0
        sub = _criterion_aware_sculpt_run_factory(
            iter_metrics=[base, base + 0.5, base + 1.0, base + 1.5],
            write_behavior_pass_at=[False] * 4,
        )
        return sub(**kw)

    monkeypatch.setattr(sculpt_mod, "sculpt_run", dispatcher)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append,
        extend_on_improvement=True,
        max_extensions_per_stage=2,
        # extension_factor=1.0 so each extension call gets a full
        # 4-iter budget. With the default 0.5 factor a 4-iter base
        # would yield 2-iter extensions, whose primary_metric_history
        # is shorter than `_is_metric_still_improving`'s 4-iter floor
        # — so the trend test would correctly bail BEFORE hitting the
        # cap. That safety-fallback is good behavior; this test
        # specifically exercises the CAP path, so we use a 1.0
        # factor to keep extensions long enough to chain.
        extension_factor=1.0,
    )
    # Original + 2 extensions = 3 calls. NOT 4 (cap honored).
    assert call_count["n"] == 3
    exhausted = [
        e for e in events
        if e.get("type") == "stage_extension_exhausted"
    ]
    assert len(exhausted) == 1
    assert exhausted[0]["extensions_used"] == 2


def test_goal_b_does_not_extend_when_patience_early_stop_fired(
    tmp_path: Path, monkeypatch, stub_adapter,
):
    """If the existing metric-plateau early-stop (Ship 9a) fired
    inside sculpt_run, the metric ISN'T improving by definition.
    Goal B's extension check should bail with reason=metric_plateau_
    early_stop instead of attempting a futile extension."""
    from sculptor import sculpt as sculpt_mod

    m = _make_mission(tmp_path, n_stages=1)
    m.stages[0].max_iterations = 4
    m.stages[0].success_criterion = "metric > 1.0"
    # Pre-exhaust Ship 17 redecomp so a failed stage HALTS instead
    # of triggering a Claude redecompose call (which the test stub
    # doesn't mock). Goal B tests focus on the extension path; they
    # don't exercise re-decomposition.
    m.stages[0].redecomposition_attempts = 1

    def fake(
        *, config_path, behavior_goal, iterations=3,
        steps_per_iter=None, seed=None, init_policy_path=None,
        resume=False, per_iter_callback=None, **_kw,
    ):
        # Hand-craft an early-stopped result.
        from sculptor.sculpt import IterOutcome, SculptRunResult
        project = Path(config_path).parent
        iter_dir = project / "runs" / "iter_3"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "checkpoint.pt").write_bytes(b"fake")
        _fabricate_rollout_artifacts(
            iter_dir,
            behavior={"n_episodes": 1, "mean_return": 0.4,
                      "mean_episode_length": 400.0,
                      "max_episode_length": 500},
        )
        outcome = IterOutcome(
            iter_index=3, iter_dir=iter_dir,
            reward_path_before=project / "rewards" / "v1.py",
            reward_path_after=project / "rewards" / "v3.py",
            primary_metric=0.4,
            behavior={"mean_return": 0.4},
            failure_modes=[], edit_count=0,
        )
        result = SculptRunResult(
            iterations_run=3,
            completed_iters=[outcome],
            primary_metric_history=[0.4, 0.4, 0.4],
            early_stopped=True,
            early_stop_reason="no improvement (test stub)",
        )
        return result

    monkeypatch.setattr(sculpt_mod, "sculpt_run", fake)
    monkeypatch.setattr(
        "sculptor.edit.apply_prompt_edit", _stub_apply_prompt_edit,
    )

    events: list[dict] = []
    sculpt_mod.mission_run(
        m, adapter_short_name="mjlab",
        on_event=events.append,
        extend_on_improvement=True,
    )
    skipped = [
        e for e in events
        if e.get("type") == "stage_extension_skipped"
        and e.get("reason") == "metric_plateau_early_stop"
    ]
    assert len(skipped) == 1


def test_inherit_parent_adapter_config_also_inherits_iteration(tmp_path: Path):
    """§Ship-19c (extended): _inherit_parent_adapter_config now also
    copies [iteration]. Without this, sculpt_init's generic
    steps_per_iter=50000 was ALSO leaking into stages, blowing
    wall-clock on Cartpole. Verify [iteration] inherits too."""
    from sculptor.sculpt import _inherit_parent_adapter_config

    project_dir = tmp_path / "proj"
    stage_dir = tmp_path / "proj" / ".missions" / "m" / "stages" / "s0"
    project_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)

    (project_dir / "config.toml").write_text(
        '[target]\nname = "p"\n\n'
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Cartpole-Balance", num_envs = 1024, device = "cuda:0" }\n\n'
        '[iteration]\nsteps_per_iter = 1500\nprimary_metric = "mean_return"\n\n'
        '[kg]\nseeds_path = "kg_seeds.yml"\n'
    )
    (stage_dir / "config.toml").write_text(
        '[target]\nname = "s0"\n\n'
        '[adapter]\nclass = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { env_id = "CHANGE_ME" }\n\n'
        '[iteration]\nsteps_per_iter = 50000\n\n'
        '[kg]\nseeds_path = "kg_seeds.yml"\n'
    )
    changed = _inherit_parent_adapter_config(
        stage_dir=stage_dir, project_dir=project_dir,
    )
    assert changed is True
    final = (stage_dir / "config.toml").read_text()
    assert "steps_per_iter = 1500" in final
    assert "steps_per_iter = 50000" not in final
    # Adapter inheritance still works (Ship 19c original case).
    assert 'task_id = "Mjlab-Cartpole-Balance"' in final


def test_mission_run_source_does_not_unlink_lock():
    """Audit-fix regression guard: verify the orchestrator's `finally`
    block does NOT call `lock_path.unlink`. Pre-fix unlink could
    silently fail on Windows/WSL; post-fix we let the `.lock` file
    persist (or filelock auto-cleans it — backend-specific). Either
    way, the explicit unlink is gone."""
    import inspect
    from sculptor import sculpt as sculpt_mod

    src = inspect.getsource(sculpt_mod.mission_run)
    # The audit-fix removed the explicit unlink. If a future refactor
    # re-introduces it, this guard fires.
    assert "lock_path.unlink" not in src, (
        "mission_run must not unlink the lock file in its finally — "
        "filelock release() handles cleanup, and unlink can fail "
        "silently on Windows/WSL. See Ship-16 audit fix."
    )
