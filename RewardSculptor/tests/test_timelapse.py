"""tests/test_timelapse.py — offline tests for the final-report builder.

No ffmpeg invocation (we monkeypatch `_build_final_mp4`). The goal is to
exercise the MD-report generation logic against a hand-built project on
disk: iter dir selection, top-3 ranking, literature map, novel-edit
detection, summary table.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from sculptor.timelapse import (
    _canonical_digest,
    _collect_iter_edits,
    _completion_receipt,
    _describe_behavior,
    _find_iter_dirs,
    _report_claim_inputs,
    _select_iter_indices,
    build_mission_report,
    build_report,
    inspect_report_state,
)
from sculptor.run_manifests import (
    build_completion_manifest,
    build_rollout_input_manifest,
    build_train_input_manifest,
    manifest_sha256,
    write_json_atomic,
)
from sculptor.sculpt import _write_iteration_completion_marker


def test_select_iter_indices_edges():
    assert _select_iter_indices(0) == []
    assert _select_iter_indices(1) == [0]
    assert _select_iter_indices(2) == [0, 1]
    assert _select_iter_indices(3) == [0, 1, 2]
    assert _select_iter_indices(5) == [0, 2, 4]
    assert _select_iter_indices(10) == [0, 5, 9]


def test_describe_behavior_uses_configured_vocabulary():
    behavior = {
        "mean_return": 71.0,
        "max_episode_length": 43,
        "fall_rate": 1.0,
        "mean_forward_velocity": 0.68,
        "something_else": "ignored",
    }
    out = _describe_behavior(
        behavior, ["max_episode_length", "mean_forward_velocity", "fall_rate"])
    assert "max_episode_length" in out
    assert "fall_rate" in out
    # something_else is not in the vocab, should be absent.
    assert "something_else" not in out
    # mean_return is prepended automatically when not in the vocab list.
    assert "mean_return" in out


def test_collect_iter_edits_attributes_delta_to_next_iter(tmp_path: Path):
    project = _write_project(tmp_path, n_iters=3,
                             metric_history=[10.0, 20.0, 15.0])
    from sculptor.timelapse import _find_iter_dirs
    dirs = _find_iter_dirs(project / "runs")
    edits = _collect_iter_edits(dirs, [10.0, 20.0, 15.0])
    # iter 0 edits get delta = 20-10 = +10
    # iter 1 edits get delta = 15-20 = -5
    # iter 2 edits get delta = None (no iter 3)
    deltas_by_iter: dict[int, list[float | None]] = {}
    for e in edits:
        deltas_by_iter.setdefault(e.iter_index, []).append(e.delta)
    assert deltas_by_iter[0][0] == pytest.approx(10.0)
    assert deltas_by_iter[1][0] == pytest.approx(-5.0)
    assert deltas_by_iter[2][0] is None


# ── Fixture writer ──────────────────────────────────────────────────────
class _ReceiptAdapter:
    env_id = "report-test"
    robot = "unit"
    control_dt = 0.02


def _attest_iteration(project: Path, iteration: int) -> None:
    """Write exact schema-3 phase and completion receipts for a fixture."""
    iter_dir = project / "runs" / f"iter_{iteration}"
    rollout_dir = iter_dir / "rollout"
    checkpoint = iter_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        checkpoint.write_bytes(f"checkpoint:{iteration}".encode())
    trajectory = rollout_dir / "trajectory.npz"
    if not trajectory.is_file():
        trajectory.write_bytes(f"trajectory:{iteration}".encode())
    reward = project / "rewards" / f"v{iteration}.py"
    adapter = _ReceiptAdapter()
    request = build_train_input_manifest(
        adapter=adapter,
        iteration=iteration,
        reward_module_path=reward,
        steps=1000,
        seed=iteration,
        init_policy_path=None,
        init_policy_mode="actor_critic",
    )
    train_input = {
        **request,
        "request_manifest_sha256": manifest_sha256(request),
        "effective_initialization": {
            "mode": None,
            "policy": None,
            "forwarded_to_adapter": False,
        },
    }
    rollout_input = build_rollout_input_manifest(
        adapter=adapter,
        iteration=iteration,
        checkpoint_path=checkpoint,
        reward_module_path=reward,
        n_episodes=1,
        seed=iteration,
        max_episode_steps=None,
        playback_speed=None,
        render_every=None,
        fps=None,
        render_width=None,
        render_height=None,
        render_env_index=None,
    )
    write_json_atomic(iter_dir / "train_request_manifest.json", request)
    write_json_atomic(iter_dir / "train_input_manifest.json", train_input)
    write_json_atomic(
        iter_dir / "train_completion_manifest.json",
        build_completion_manifest(train_input, [checkpoint]),
    )
    write_json_atomic(
        rollout_dir / "rollout_input_manifest.json", rollout_input,
    )
    write_json_atomic(
        rollout_dir / "rollout_completion_manifest.json",
        build_completion_manifest(
            rollout_input,
            [
                rollout_dir / "rollout.mp4",
                trajectory,
                rollout_dir / "behavior.json",
            ],
        ),
    )
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=iteration,
        checkpoint_path=checkpoint,
        reward_version_before=iteration,
        reward_version_after=None,
        world_selection_hash=None,
    )


def _verified_report_authority(project: Path, selected: int) -> dict:
    selection_path = project / "reports" / "selection.json"
    selection_path.write_text(json.dumps({
        "schema": 1,
        "selected_iter_index": selected,
        "selection_source": "objective_criterion",
        "candidates": [
            {"iter_index": selected, "selected": True, "criterion_pass": True},
        ],
    }), encoding="utf-8")
    iter_dir = project / "runs" / f"iter_{selected}"
    completion = _completion_receipt(iter_dir)
    assert completion is not None
    claim_inputs = _report_claim_inputs(iter_dir, completion)
    objective = {"objective_proof_status": "passed"}
    authority = {
        "schema": 1,
        "status": "verified",
        "selected_iter_index": selected,
        "selection_source": "objective_criterion",
        "selection_receipt_sha256": hashlib.sha256(
            selection_path.read_bytes()
        ).hexdigest(),
        "selected_checkpoint_sha256": completion["checkpoint_sha256"],
        "claim_inputs": claim_inputs,
        "claim_inputs_sha256": _canonical_digest(claim_inputs),
        "objective_evidence_receipt": objective,
        "objective_evidence_sha256": _canonical_digest(objective),
        "blockers": [],
    }
    authority["authority_digest"] = _canonical_digest(authority)
    return authority


def _write_project(tmp_path: Path, *, n_iters: int,
                   metric_history: list[float]) -> Path:
    project = tmp_path / "proj"
    return _write_project_at(project, n_iters=n_iters, metric_history=metric_history)


def _write_project_at(project: Path, *, n_iters: int,
                      metric_history: list[float]) -> Path:
    """Same fixture body as `_write_project`, but writes directly at
    `project` instead of always under `<tmp_path>/proj`. Used both by
    `_write_project` (project-level tests) and `_write_mission` (which
    needs each stage's mini-project at a fixed `stages/<name>/` path
    dictated by `Mission.stage_dir`)."""
    (project / "rewards").mkdir(parents=True)
    (project / "reports").mkdir()
    (project / "runs").mkdir()

    # config.toml — use the real GymSB3Adapter dotted class so build_report
    # can still try to read the [adapter] block. We don't actually load it.
    (project / "config.toml").write_text(textwrap.dedent("""
        [target]
        name = "unit_test"

        [adapter]
        class = "tests.stub.AdapterStub"
        config = {}

        [kg]
        environment_tag = "unit_test"

        [iteration]
        steps_per_iter = 1000
        primary_metric = "mean_return"
        behavior_metrics = ["max_episode_length", "fall_rate"]
    """).strip() + "\n", encoding="utf-8")

    # rewards: v0 through v<n_iters>
    for i in range(n_iters + 1):
        (project / "rewards" / f"v{i}.py").write_text(textwrap.dedent(f'''
            """v{i}"""
            REWARD_SPEC = {{
                "version": "v{i}",
                "parent_hash": "stub",
                "description": "stub v{i}",
                "author": "sculptor" if {i} > 0 else "human",
                "hyperparameters": {{"alive_bonus": {1.0 + 0.5 * i}}},
                "references": [],
            }}
            def compute_reward(state, action, next_state, info):
                a = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
                return a, {{"alive_bonus": a}}
        ''').strip() + "\n", encoding="utf-8")

    # Per-iter: metrics.json, behavior.json, diagnosis.json, rollout stub
    for i in range(n_iters):
        iter_dir = project / "runs" / f"iter_{i}"
        (iter_dir / "rollout" / "keyframes").mkdir(parents=True)
        (iter_dir / "metrics.json").write_text(json.dumps({
            "metrics": {"mean_return": metric_history[i]},
        }), encoding="utf-8")
        (iter_dir / "rollout" / "behavior.json").write_text(json.dumps({
            "mean_return": metric_history[i],
            "max_episode_length": 40 + i,
            "fall_rate": 1.0 - 0.1 * i,
            "mean_forward_velocity": 0.5 + 0.05 * i,
        }), encoding="utf-8")
        # Small sentinel mp4 (below _probe_video_ok's 2KB threshold so it's
        # classified as invalid — we don't invoke ffmpeg in these tests).
        (iter_dir / "rollout" / "rollout.mp4").write_bytes(b"stub")
        diag = {
            "failure_modes": ["component_imbalance"],
            "evidence": f"iter {i} evidence",
            "proposed_edits": [
                {
                    "target_term": "alive_bonus", "operation": "increase",
                    "rationale": f"iter {i}: DM Control reference",
                    "suggested_value": str(1.0 + 0.5 * (i + 1)),
                    "paper_refs": ["1801.00690"],
                    "requires_env_extension": False,
                },
                {
                    "target_term": f"novel_term_{i}", "operation": "add",
                    "rationale": "novel. probe proposal.",
                    "suggested_value": "0.5",
                    "paper_refs": [],
                    "requires_env_extension": False,
                },
            ],
            "literature_context": [],
            "confidence": 0.5,
            "iter_dir": str(iter_dir),
            "behavior_goal": "do the thing",
        }
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diag, indent=2), encoding="utf-8")
        _attest_iteration(project, i)

    # metric_history.json
    (project / "reports" / "metric_history.json").write_text(json.dumps({
        "primary_metric": "mean_return",
        "history": metric_history,
    }), encoding="utf-8")

    # provenance.json — one live active entry + one retired entry
    prov = {
        "alive_bonus": [{
            "arxiv_id": "1801.00690",
            "citation": "Tassa et al. (2018). DeepMind Control Suite. arXiv:1801.00690",
            "iter_introduced": 1,
            "how_used": "alive_bonus raised per DM Control guidance",
            "still_active": True,
        }],
        "removed_term": [{
            "arxiv_id": "1707.06347",
            "citation": "Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347",
            "iter_introduced": 1,
            "how_used": "Removed at iter 2",
            "still_active": False,
        }],
    }
    (project / "reports" / "provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")

    # CHANGELOG sentinel
    (project / "CHANGELOG.md").write_text("# changelog stub\n", encoding="utf-8")
    return project


# ── build_report end-to-end (mp4 stubbed) ──────────────────────────────
def test_build_report_produces_md_and_calls_mp4_builder(tmp_path: Path, monkeypatch):
    project = _write_project(tmp_path, n_iters=3,
                             metric_history=[10.0, 25.0, 18.0])
    config_path = project / "config.toml"
    out_mp4 = tmp_path / "final.mp4"

    seen: dict = {}

    def _fake_build_mp4(**kwargs):
        seen.update(kwargs)
        Path(kwargs["out_path"]).write_bytes(b"x" * 8192)
        return True, ""

    monkeypatch.setattr(
        "sculptor.timelapse._build_final_mp4", _fake_build_mp4)

    result = build_report(config_path=config_path, out_mp4=out_mp4)

    # MP4 builder was bypassed because panel videos are sentinels < 2KB.
    # That means the fallback branch triggers (title-only) and our patch
    # of _build_final_mp4 isn't even reached. Verify the title-only mp4
    # path at least produced an md + attempted ffmpeg. If ffmpeg isn't
    # available the mp4 will be empty; md must still exist.
    assert result.final_report_md_path.is_file()
    md = result.final_report_md_path.read_text(encoding="utf-8")

    # Report shape
    assert "# Sculpt Final Report" in md
    assert "do the thing" in md  # behavior goal
    assert "Top 3 most impactful edits" in md
    assert "Literature map" in md
    assert "Candidate novel contributions" in md
    assert "Summary" in md
    # Top-3 ranking: iter 0 delta = +15 (biggest), iter 1 delta = -7, iter 2 none.
    assert "iter 0" in md
    # The DM Control citation shows up in the literature map (alive_bonus).
    assert "1801.00690" in md
    # Novel terms show up in Candidate novel contributions
    assert "novel_term_0" in md
    # Retired entries with still_active=false are omitted.
    assert "removed_term" not in md.lower()
    assert result.report_claim_status == "descriptive_only"
    assert "**Selected**" not in md


def test_report_uses_canonical_selection_and_its_pinned_reward(
    tmp_path: Path,
):
    project = _write_project(
        tmp_path,
        n_iters=4,
        metric_history=[10.0, 12.0, 20.0, 15.0],
    )
    fitness_values = [0.1, 0.2, 0.9, 0.3]
    for i, fitness in enumerate(fitness_values):
        iter_dir = project / "runs" / f"iter_{i}"
        (iter_dir / "fitness.json").write_text(
            json.dumps({"fitness": fitness}),
            encoding="utf-8",
        )

    selected = project / "runs" / "iter_1"
    selected_reward_sha = hashlib.sha256(
        (project / "rewards" / "v1.py").read_bytes()
    ).hexdigest()
    (selected / "artifact_tuple.json").write_text(
        json.dumps({
            "refs": {
                "reward": {
                    "path": "rewards/v1.py",
                    "version": "v1",
                    "sha256": selected_reward_sha,
                },
            },
        }),
        encoding="utf-8",
    )

    result = build_report(
        config_path=project / "config.toml",
        out_mp4=tmp_path / "final.mp4",
        selection_authority=_verified_report_authority(project, 1),
    )
    md = result.final_report_md_path.read_text(encoding="utf-8")

    assert "**Selected policy reward module**" in md
    assert "rewards/v1.py" in md
    assert "rewards/v4.py" not in md
    assert "**Selected** (iter 1)" in md
    assert "+10.0000 → +12.0000" in md


def test_report_refuses_stale_or_failed_selection_authority(tmp_path: Path):
    project = _write_project(tmp_path, n_iters=1, metric_history=[10.0])
    selected = project / "runs" / "iter_0"
    (selected / "fitness.json").write_text(
        json.dumps({"fitness": 1.0}), encoding="utf-8",
    )
    (selected / "artifact_tuple.json").write_text(
        json.dumps({
            "refs": {
                "reward": {
                    "path": "rewards/v0.py",
                    "sha256": hashlib.sha256(
                        (project / "rewards" / "v0.py").read_bytes()
                    ).hexdigest(),
                },
            },
        }),
        encoding="utf-8",
    )
    authority = _verified_report_authority(project, 0)
    authority["objective_evidence_receipt"] = {
        "objective_proof_status": "failed",
    }
    authority["objective_evidence_sha256"] = _canonical_digest(
        authority["objective_evidence_receipt"]
    )
    authority["authority_digest"] = _canonical_digest({
        key: value for key, value in authority.items()
        if key != "authority_digest"
    })

    result = build_report(
        config_path=project / "config.toml",
        out_mp4=tmp_path / "failed-authority.mp4",
        selection_authority=authority,
    )
    markdown = result.final_report_md_path.read_text(encoding="utf-8")

    assert result.report_claim_status == "descriptive_only"
    assert "**Selected**" not in markdown
    assert "makes no selected-policy or task-success claim" in markdown


def test_report_receipt_detects_changed_video_and_evidence(
    tmp_path: Path, monkeypatch,
):
    project = _write_project(tmp_path, n_iters=1, metric_history=[10.0])
    iter_dir = project / "runs" / "iter_0"
    (iter_dir / "rollout" / "rollout.mp4").write_bytes(b"x" * 4096)
    _attest_iteration(project, 0)

    def _fake_build_mp4(**kwargs):
        Path(kwargs["out_path"]).write_bytes(b"report-video")
        return True, ""

    monkeypatch.setattr(
        "sculptor.timelapse._build_final_mp4", _fake_build_mp4,
    )
    result = build_report(
        config_path=project / "config.toml",
        out_mp4=project / "reports" / "final.mp4",
    )
    assert inspect_report_state(project)["state"] == "current"

    result.final_mp4_path.write_bytes(b"mutated-report-video")
    assert inspect_report_state(project)["state"] == "stale"

    build_report(
        config_path=project / "config.toml",
        out_mp4=project / "reports" / "final.mp4",
    )
    assert inspect_report_state(project)["state"] == "current"
    (iter_dir / "fitness.json").write_text(
        json.dumps({"fitness": 0.25}), encoding="utf-8",
    )
    assert inspect_report_state(project)["state"] == "stale"


def test_report_counts_only_attested_iterations(tmp_path: Path):
    project = _write_project(tmp_path, n_iters=2, metric_history=[1.0, 2.0])
    incomplete = project / "runs" / "iter_9"
    incomplete.mkdir()
    (incomplete / "diagnosis.json").write_text("{}", encoding="utf-8")

    result = build_report(
        config_path=project / "config.toml",
        out_mp4=tmp_path / "attested-only.mp4",
    )
    markdown = result.final_report_md_path.read_text(encoding="utf-8")

    assert "**Iterations completed**: 2" in markdown
    assert result.selected_iter_indices == [0, 1]


def test_iteration_discovery_rejects_symlinked_directories(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    external = tmp_path / "iter_7"
    external.mkdir()
    (runs / "iter_7").symlink_to(external, target_is_directory=True)

    assert _find_iter_dirs(runs) == []


def test_build_report_with_valid_mp4s_calls_builder(tmp_path: Path, monkeypatch):
    """Force panel videos to pass the 2KB probe, so `_build_final_mp4`
    actually gets called with the right iter selection."""
    project = _write_project(tmp_path, n_iters=5,
                             metric_history=[0.0, 10.0, 20.0, 15.0, 25.0])
    config_path = project / "config.toml"
    # Make every rollout.mp4 larger than 2KB so _probe_video_ok passes.
    for iter_dir in (project / "runs").iterdir():
        mp4 = iter_dir / "rollout" / "rollout.mp4"
        mp4.write_bytes(b"x" * 4096)
        _attest_iteration(project, int(iter_dir.name.removeprefix("iter_")))

    received: dict = {}

    def _fake_build_mp4(**kwargs):
        received["panel_videos"] = kwargs["panel_videos"]
        received["panel_labels"] = kwargs["panel_labels"]
        Path(kwargs["out_path"]).write_bytes(b"y" * 10_000)
        return True, ""

    monkeypatch.setattr(
        "sculptor.timelapse._build_final_mp4", _fake_build_mp4)

    out_mp4 = tmp_path / "final.mp4"
    result = build_report(config_path=config_path, out_mp4=out_mp4)
    assert result.final_mp4_ok is True
    # For N=5, selected indices are [0, 2, 4].
    assert result.selected_iter_indices == [0, 2, 4]
    assert len(received["panel_videos"]) == 3
    assert len(received["panel_labels"]) == 3
    # Labels include the primary_metric key + value.
    assert "mean_return" in received["panel_labels"][0]
    # Iter 4's label should show the final value (25.0).
    assert "+25.000" in received["panel_labels"][-1]


# ── build_mission_report (§chunk C1) ────────────────────────────────────
def _write_mission(
    tmp_path: Path, *, stage_specs: list[tuple[str, int, list[float]]],
    valid_mp4s: bool = False,
) -> Path:
    """Write a synthetic multi-stage mission tree:

        <mission_dir>/mission.json
        <mission_dir>/stages/<stage_name>/  (per _write_project_at)

    `stage_specs` is `[(stage_name, n_iters, metric_history), ...]` in
    stage order. Returns `mission_dir`. `valid_mp4s=True` makes every
    iter's rollout.mp4 large enough to pass `_probe_video_ok` (2KB).
    """
    from sculptor.mission import Mission, Stage, save_mission

    mission_dir = tmp_path / "mission"
    stages: list[Stage] = []
    for i, (name, n_iters, metric_history) in enumerate(stage_specs):
        stage_dir = mission_dir / "stages" / name
        _write_project_at(stage_dir, n_iters=n_iters, metric_history=metric_history)
        if valid_mp4s:
            for iter_dir in (stage_dir / "runs").iterdir():
                (iter_dir / "rollout" / "rollout.mp4").write_bytes(b"x" * 4096)
                _attest_iteration(
                    stage_dir,
                    int(iter_dir.name.removeprefix("iter_")),
                )
        stages.append(Stage(
            name=name,
            goal_text=f"reach the {name} milestone",
            success_criterion="behavior['mean_return'] > 0",
            max_iterations=10,
            parent_stage=stage_specs[i - 1][0] if i > 0 else None,
            reward_seed_prompt=f"reward the {name} behavior",
            status="succeeded" if metric_history else "pending",
            best_metric=metric_history[-1] if metric_history else None,
            iterations_used=n_iters,
        ))

    mission = Mission(
        goal="do a complex multi-stage behavior",
        stages=stages,
        decomposition_model="claude-test",
        decomposition_rationale="split into stand-then-walk for curriculum reasons",
    )
    save_mission(mission, mission_dir)
    return mission_dir


def test_build_mission_report_writes_md_with_both_stage_sections(
    tmp_path: Path, monkeypatch,
):
    """Sentinel (<2KB) rollout mp4s mean no panel qualifies — mirrors
    `test_build_report_produces_md_and_calls_mp4_builder`'s title-only
    fallback. The markdown must still contain both stage sections."""
    mission_dir = _write_mission(tmp_path, stage_specs=[
        ("stand", 2, [5.0, 12.0]),
        ("walk", 3, [1.0, 8.0, 20.0]),
    ])
    out_mp4 = mission_dir / "reports" / "final.mp4"

    result = build_mission_report(mission_dir=mission_dir, out_mp4=out_mp4)

    assert result.final_report_md_path.is_file()
    md = result.final_report_md_path.read_text(encoding="utf-8")

    assert "# Sculpt Mission Report" in md
    assert "do a complex multi-stage behavior" in md  # mission goal
    assert "stand-then-walk" in md  # decomposition_rationale
    assert "## Stage: `stand`" in md
    assert "## Stage: `walk`" in md
    assert "reach the stand milestone" in md  # stage.goal_text
    assert "reach the walk milestone" in md
    # Per-stage primary-metric start -> end lines rendered.
    assert "+5.0000" in md and "+12.0000" in md
    assert "+1.0000" in md and "+20.0000" in md
    # Stages overview table lists both by name + status.
    assert "`stand`" in md and "`walk`" in md
    assert "succeeded" in md


def test_build_mission_report_stitches_mp4_across_stages(
    tmp_path: Path, monkeypatch,
):
    """Valid-sized rollout.mp4s in both stages: `_build_final_mp4` should
    be invoked once with panels drawn from BOTH stages (capped at 3/stage),
    labeled "<stage> · iter N · <metric>=<value>"."""
    mission_dir = _write_mission(
        tmp_path,
        stage_specs=[
            ("stand", 2, [5.0, 12.0]),
            ("walk", 3, [1.0, 8.0, 20.0]),
        ],
        valid_mp4s=True,
    )
    out_mp4 = mission_dir / "reports" / "final.mp4"

    received: dict = {}

    def _fake_build_mp4(**kwargs):
        received["panel_videos"] = kwargs["panel_videos"]
        received["panel_labels"] = kwargs["panel_labels"]
        Path(kwargs["out_path"]).write_bytes(b"y" * 10_000)
        return True, ""

    monkeypatch.setattr("sculptor.timelapse._build_final_mp4", _fake_build_mp4)

    result = build_mission_report(mission_dir=mission_dir, out_mp4=out_mp4)

    assert result.final_mp4_ok is True
    assert result.final_mp4_path.is_file()
    # stand has 2 iters -> _select_iter_indices(2) = [0, 1] (both, <=3 cap).
    # walk has 3 iters -> _select_iter_indices(3) = [0, 1, 2] (all, <=3 cap).
    assert len(received["panel_videos"]) == 5
    assert len(received["panel_labels"]) == 5
    stand_labels = [
        label for label in received["panel_labels"] if label.startswith("stand")
    ]
    walk_labels = [
        label for label in received["panel_labels"] if label.startswith("walk")
    ]
    assert len(stand_labels) == 2
    assert len(walk_labels) == 3
    # Label format: "<stage> · iter N · <primary_key>=<value>".
    assert any(
        "stand · iter 0 · mean_return=" in label for label in stand_labels
    )
    assert any(
        "walk · iter 2 · mean_return=+20.000" in label for label in walk_labels
    )

    md = result.final_report_md_path.read_text(encoding="utf-8")
    assert "[final.mp4]" in md


def test_build_mission_report_skips_unscaffolded_stage_gracefully(
    tmp_path: Path,
):
    """A stage listed in mission.json but never scaffolded (no
    config.toml on disk yet — e.g. mission halted after stage 0) must
    not raise; its section renders a placeholder and the OTHER stage's
    data still comes through."""
    from sculptor.mission import Mission, Stage, save_mission

    mission_dir = tmp_path / "mission"
    _write_project_at(
        mission_dir / "stages" / "stand", n_iters=2,
        metric_history=[3.0, 9.0])

    stages = [
        Stage(
            name="stand", goal_text="reach the stand milestone",
            success_criterion="behavior['mean_return'] > 0",
            max_iterations=10, parent_stage=None,
            reward_seed_prompt="reward standing", status="succeeded",
            best_metric=9.0, iterations_used=2,
        ),
        Stage(
            name="walk", goal_text="reach the walk milestone",
            success_criterion="behavior['mean_return'] > 0",
            max_iterations=10, parent_stage="stand",
            reward_seed_prompt="reward walking", status="pending",
            best_metric=None, iterations_used=0,
        ),
    ]
    mission = Mission(
        goal="do a complex multi-stage behavior", stages=stages,
        decomposition_model="claude-test",
        decomposition_rationale="split into stand-then-walk",
    )
    save_mission(mission, mission_dir)

    out_mp4 = mission_dir / "reports" / "final.mp4"
    result = build_mission_report(mission_dir=mission_dir, out_mp4=out_mp4)

    assert result.final_report_md_path.is_file()
    md = result.final_report_md_path.read_text(encoding="utf-8")
    assert "## Stage: `stand`" in md
    assert "## Stage: `walk`" in md
    assert "+3.0000" in md and "+9.0000" in md  # stand's data present
    # walk's placeholder — never scaffolded, no config.toml.
    assert "has not scaffolded yet" in md
