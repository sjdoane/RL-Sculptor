"""§RL_SCULPTOR_AUDIT (env generalization 3/4): diagnoser-iterable env
config — apply_env_edits validation gates, the diagnose() surface
(# ENV_SPEC block, proposed_env_edits packing, the requires_env_extension
regression), and the sculpt-loop keep-best/revert threading for the
(reward, env) pair."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sculptor.sculpt as S
from sculptor import env_spec as es
from sculptor.diagnose import (
    ProposedEnvEdit,
    _GroundedModel,
    _ProposedEditModel,
    _ProposedEnvEditModel,
    _render_env_spec_block,
)


def _seed_env(tmp_path: Path) -> Path:
    env_dir = tmp_path / "env"
    es.write_env_spec_version(env_dir, es.jump_preset_spec())
    return env_dir


# ── apply_env_edits ────────────────────────────────────────────────────────
def test_apply_env_edits_happy_scalar_and_range(tmp_path) -> None:
    env_dir = _seed_env(tmp_path)
    res = es.apply_env_edits(env_dir, [
        ProposedEnvEdit("min_base_height_termination_m", "0.25", "sit basin"),
        ProposedEnvEdit("reset_height_offset_m", "[0.0, 0.6]", "higher RSI"),
    ])
    assert res["new_version"] == "v1"
    assert res["rejected"] == []
    assert sorted(res["applied"]) == [
        "min_base_height_termination_m=0.25",
        "reset_height_offset_m=[0.0, 0.6]",
    ]
    cur = es.read_current_env_spec(env_dir)
    assert cur["meta"]["version"] == "v1"
    assert cur["meta"]["source"] == "diagnoser"
    assert cur["meta"]["parent"] == "v0"
    assert cur["train"]["min_base_height_termination_m"] == 0.25
    assert cur["train"]["reset_height_offset_m"] == [0.0, 0.6]
    # Untouched keys survive.
    assert cur["shared"]["episode_length_s"] == 10.0
    assert cur["train"]["entropy_coef_scale"] == 2.0


def test_apply_env_edits_rejects_shared_and_unknown_params(tmp_path) -> None:
    """The eval surface is structurally unreachable — a shared/eval key
    is rejected exactly like a made-up one."""
    env_dir = _seed_env(tmp_path)
    res = es.apply_env_edits(env_dir, [
        ProposedEnvEdit("episode_length_s", "5.0", "shorter"),
        ProposedEnvEdit("orientation_termination_deg", "90", "tighter"),
        ProposedEnvEdit("gravity", "-3.0", "moon"),
    ])
    assert res["applied"] == [] and res["new_version"] is None
    assert len(res["rejected"]) == 3
    assert es.read_current_env_spec(env_dir)["meta"]["version"] == "v0"


def test_apply_env_edits_rejects_bad_json_and_bounds(tmp_path) -> None:
    env_dir = _seed_env(tmp_path)
    res = es.apply_env_edits(env_dir, [
        ProposedEnvEdit("entropy_coef_scale", "very high", "junk value"),
        ProposedEnvEdit("entropy_coef_scale", "99.0", "out of bounds"),
        ProposedEnvEdit("friction_range", "[1.5, 0.2]", "inverted"),
    ])
    assert res["applied"] == [] and res["new_version"] is None
    reasons = dict((p, r) for p, r in res["rejected"] if p)
    assert "not valid JSON" in reasons["entropy_coef_scale"] \
        or "hard bounds" in reasons["entropy_coef_scale"]
    assert any("lo 1.5 > hi 0.2" in r for _, r in res["rejected"])


def test_apply_env_edits_mixed_applies_good_rejects_bad(tmp_path) -> None:
    env_dir = _seed_env(tmp_path)
    res = es.apply_env_edits(env_dir, [
        ProposedEnvEdit("entropy_coef_scale", "99.0", "bad"),
        ProposedEnvEdit("entropy_coef_scale", "1.5", "good"),
    ])
    assert res["applied"] == ["entropy_coef_scale=1.5"]
    assert len(res["rejected"]) == 1
    cur = es.read_current_env_spec(env_dir)
    assert cur["train"]["entropy_coef_scale"] == 1.5
    assert cur["meta"]["version"] == "v1"


def test_apply_env_edits_no_active_spec(tmp_path) -> None:
    res = es.apply_env_edits(tmp_path / "env", [
        ProposedEnvEdit("entropy_coef_scale", "1.5", "r")])
    assert res["applied"] == [] and res["new_version"] is None
    assert res["rejected"] == [("entropy_coef_scale", "no active env spec")]


def test_apply_env_edits_accepts_dict_edits(tmp_path) -> None:
    env_dir = _seed_env(tmp_path)
    res = es.apply_env_edits(env_dir, [
        {"parameter": "friction_range", "new_value": "[0.2, 1.6]",
         "rationale": "robustness"}])
    assert res["applied"] == ["friction_range=[0.2, 1.6]"]


# ── diagnose() surface ─────────────────────────────────────────────────────
def test_env_param_literal_rejects_shared_keys() -> None:
    with pytest.raises(Exception):
        _ProposedEnvEditModel(
            parameter="episode_length_s", new_value="5", rationale="r")
    m = _ProposedEnvEditModel(
        parameter="entropy_coef_scale", new_value="1.5", rationale="r")
    assert m.parameter == "entropy_coef_scale"


def test_render_env_spec_block_contents() -> None:
    block = _render_env_spec_block(es.jump_preset_spec())
    assert block.startswith("# ENV_SPEC")
    assert "train (editable)" in block
    assert "shared (frozen)" in block
    assert "entropy_coef_scale" in block          # bounds listed
    assert "reset_height_offset_m" in block
    assert _render_env_spec_block(None) == ""


def test_diagnose_packs_env_edits_and_env_extension_flag(
        tmp_path, monkeypatch) -> None:
    """Through the REAL diagnose() path (stub client): env edits reach
    Diagnosis only when a spec is active, and — REGRESSION (bug found
    2026-07-04, present since Ship 48) — requires_env_extension survives
    the pydantic→dataclass packing."""
    import sys as _sys
    sys_path_added = str(Path(__file__).parent)
    if sys_path_added not in _sys.path:
        _sys.path.insert(0, sys_path_added)
    from test_diagnose import (   # reuse the shipped fixtures' builders
        _StubClient as _DiagStubClient,
    )
    from sculptor.diagnose import _PreliminaryModel, diagnose

    # Project skeleton the diagnose() artifact loader expects.
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    (iter_dir / "metrics.json").write_text(json.dumps({"metrics": {}}))
    (iter_dir / "behavior.json").write_text(json.dumps({"mean_return": 0.0}))
    (iter_dir / "reward_spec.json").write_text(json.dumps({"version": "v0"}))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
        'config = { env_id = "Hopper-v4", n_envs = 1 }\n')

    prelim = _PreliminaryModel(
        failure_modes=["reward_hacking"], evidence="e", confidence=0.5)
    grounded = _GroundedModel(
        proposed_edits=[
            _ProposedEditModel(
                target_term="needs_new_channel", operation="add",
                rationale="needs foot contact field the adapter lacks",
                suggested_value=None, requires_env_extension=True),
        ],
        proposed_env_edits=[
            _ProposedEnvEditModel(
                parameter="entropy_coef_scale", new_value="1.5",
                rationale="exploration collapse"),
        ],
        confidence=0.5)

    # Case 1: NO env spec (gym adapter has no env_spec_path) → env edits
    # dropped; the deferred-edit flag still survives.
    d = diagnose(
        iter_dir=iter_dir, behavior_goal="hop", config=cfg_path,
        client=_DiagStubClient(prelim, grounded), skip_kg=True)
    assert d.proposed_env_edits == []
    assert d.proposed_edits[0].requires_env_extension is True   # REGRESSION
    dumped = d.to_dict()
    assert dumped["proposed_env_edits"] == []
    assert dumped["proposed_edits"][0]["requires_env_extension"] is True


def test_diagnose_env_block_and_edits_with_active_spec(
        tmp_path, monkeypatch) -> None:
    """With an env spec active (adapter.env_spec_path set), the grounded
    prompt carries the # ENV_SPEC block and env edits are packed."""
    import sys as _sys
    sys_path_added = str(Path(__file__).parent)
    if sys_path_added not in _sys.path:
        _sys.path.insert(0, sys_path_added)
    from test_diagnose import _StubClient as _DiagStubClient
    from sculptor.diagnose import _PreliminaryModel, diagnose

    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    (iter_dir / "metrics.json").write_text(json.dumps({"metrics": {}}))
    (iter_dir / "behavior.json").write_text(json.dumps({"mean_return": 0.0}))
    (iter_dir / "reward_spec.json").write_text(json.dumps({"version": "v0"}))
    env_dir = _seed_env(tmp_path)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
        'config = { env_id = "Hopper-v4", n_envs = 1 }\n')

    # gym_sb3 has no env_spec_path field; fake the injected attribute the
    # way load_adapter does for spec-aware adapters.
    import sculptor.diagnose as D
    real_load = D.load_adapter

    def load_with_spec(p):
        a = real_load(p)
        a.env_spec_path = str(env_dir / "current.json")
        return a

    monkeypatch.setattr(D, "load_adapter", load_with_spec)

    prelim = _PreliminaryModel(
        failure_modes=["none"], evidence="e", confidence=0.5)
    grounded = _GroundedModel(
        proposed_env_edits=[
            _ProposedEnvEditModel(
                parameter="min_base_height_termination_m", new_value="0.25",
                rationale="floor data dominates"),
        ],
        confidence=0.5)
    client = _DiagStubClient(prelim, grounded)
    d = diagnose(
        iter_dir=iter_dir, behavior_goal="jump", config=cfg_path,
        client=client, skip_kg=True)
    assert len(d.proposed_env_edits) == 1
    assert d.proposed_env_edits[0].parameter == "min_base_height_termination_m"
    # The grounded (2nd) prompt carried the # ENV_SPEC block.
    grounded_prompt = client.messages.captured_prompts[1]["messages"][0]["content"]
    assert "# ENV_SPEC" in grounded_prompt
    assert "shared (frozen)" in grounded_prompt


# ── sculpt-loop threading (fake _run_one_iter, real sculpt_run) ────────────
def _make_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "rewards").mkdir(parents=True)
    (proj / "runs").mkdir()
    (proj / "reports").mkdir()
    (proj / "rewards" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "rewards" / "v0.py").write_text(
        "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
        encoding="utf-8")
    (proj / "config.toml").write_text(
        "[iteration]\nsteps_per_iter = 10\nseed = 42\n", encoding="utf-8")
    return proj / "config.toml"


def _install_env_fake(monkeypatch, fitness_by_iter, env_by_iter, seen):
    monkeypatch.setattr(S, "load_adapter", lambda _p: object())
    monkeypatch.setattr(
        "sculptor.run_context.capture_run_context", lambda *a, **k: {})
    monkeypatch.setattr(
        "sculptor.run_context.write_run_context",
        lambda *a, **k: Path("run_context.json"))

    def fake_iter(**kw):
        i = kw["iter_index"]
        seen.setdefault("env_revert", {})[i] = kw.get("env_revert_version")
        rewards_dir = kw["rewards_dir"]
        edit = rewards_dir / f"v{i + 1}.py"
        edit.write_text(
            "REWARD_SPEC = {}\ndef compute_reward(*a, **k):\n    return 0.0, {}\n",
            encoding="utf-8")
        S._write_current_reexport(rewards_dir, edit)
        return S.IterOutcome(
            iter_index=i, iter_dir=kw["runs_dir"] / f"iter_{i}",
            reward_path_before=rewards_dir / "current.py",
            reward_path_after=edit, primary_metric=0.0, behavior={},
            failure_modes=[], edit_count=1,
            fitness=fitness_by_iter[i],
            reward_path_trained=rewards_dir / f"v{i}.py",
            env_spec_trained=env_by_iter.get(i))

    monkeypatch.setattr(S, "_run_one_iter", fake_iter)


def test_env_revert_version_threads_best_env_on_regression(
        tmp_path, monkeypatch) -> None:
    cfg_path = _make_project(tmp_path)
    seen: dict = {}
    # iter0 best (env v0); iter1 regresses strictly (env v1) → iter2 must
    # be told to revert the env to v0 alongside the reward.
    _install_env_fake(
        monkeypatch, {0: 0.5, 1: 0.1, 2: 0.2}, {0: "v0", 1: "v1", 2: "v0"},
        seen)
    res = S.sculpt_run(
        cfg_path, "goal", iterations=3, no_kg=True,
        fitness_fn=lambda p: 0.0, fitness_patience=10)
    assert seen["env_revert"][0] is None
    assert seen["env_revert"][1] is None          # iter0 was best; no revert
    assert seen["env_revert"][2] == "v0"          # regression → best env
    assert res.best_env_spec == "v0"


def test_best_env_spec_repointed_at_run_end(tmp_path, monkeypatch) -> None:
    cfg_path = _make_project(tmp_path)
    proj = cfg_path.parent
    env_dir = proj / "env"
    es.write_env_spec_version(env_dir, es.jump_preset_spec())   # v0
    spec2 = es.jump_preset_spec()
    spec2["train"]["entropy_coef_scale"] = 1.0
    es.write_env_spec_version(env_dir, spec2)                   # v1 = current
    assert es.read_current_env_spec(env_dir)["meta"]["version"] == "v1"

    seen: dict = {}
    # Best iter is 0, which trained under env v0; run ends with current at
    # v1 → sculpt_run must repoint current.json back to v0.
    _install_env_fake(
        monkeypatch, {0: 0.7, 1: 0.2}, {0: "v0", 1: "v1"}, seen)
    res = S.sculpt_run(
        cfg_path, "goal", iterations=2, no_kg=True,
        fitness_fn=lambda p: 0.0, fitness_patience=10)
    assert res.best_env_spec == "v0"
    assert es.read_current_env_spec(env_dir)["meta"]["version"] == "v0"


# ── remote-dispatch threading (increment-1 verifier finding) ───────────────
def test_remote_train_and_rollout_sync_env_spec(tmp_path) -> None:
    """The env spec file must ride RunnerJob.input_paths (synced to the
    pod's mirror path) and --env-profile must be excluded when the spec
    wins — for BOTH train and rollout remote dispatch."""
    pytest.importorskip("mjlab")
    from unittest.mock import patch

    from sculptor.adapters import mjlab as mjlab_mod

    spec_path = tmp_path / "env" / "current.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps(es.jump_preset_spec()))
    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=64,
        env_spec_path=str(spec_path), env_profile="jump",
        remote={"enabled": True, "host": "1.2.3.4", "user": "u",
                "rollout_remote": True})
    jobs: list = []

    class _FakeExec:
        cfg = SimpleNamespace(device="cuda:0", rollout_remote=True)

        def execute(self, job):
            jobs.append(job)
            # Satisfy the post-run artifact checks.
            (tmp_path / "checkpoint.pt").write_bytes(b"stub")
            (tmp_path / "metrics.json").write_text("{}")
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    with patch.object(mjlab_mod.MjlabAdapter, "_remote_executor",
                      return_value=_FakeExec()):
        adapter.train(reward_module_path=None, output_dir=tmp_path,
                      steps=1, seed=1)
        adapter.rollout(checkpoint_path=tmp_path / "checkpoint.pt",
                        output_dir=tmp_path, n_episodes=1)

    assert len(jobs) == 2
    for job in jobs:
        assert job.input_paths.get("--env-spec") == spec_path.resolve()
        assert "--env-profile" not in job.options
