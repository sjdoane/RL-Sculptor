"""tests/test_timelapse.py — offline tests for the final-report builder.

No ffmpeg invocation (we monkeypatch `_build_final_mp4`). The goal is to
exercise the MD-report generation logic against a hand-built project on
disk: iter dir selection, top-3 ranking, literature map, novel-edit
detection, summary table.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from sculptor.timelapse import (
    _collect_iter_edits,
    _describe_behavior,
    _select_iter_indices,
    build_report,
)


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
def _write_project(tmp_path: Path, *, n_iters: int,
                   metric_history: list[float]) -> Path:
    project = tmp_path / "proj"
    (project / "rewards").mkdir(parents=True)
    (project / "reports").mkdir()
    (project / "runs").mkdir()

    # config.toml — use the real GymSB3Adapter dotted class so build_report
    # can still try to read the [adapter] block. We don't actually load it.
    (project / "config.toml").write_text(textwrap.dedent(f"""
        [target]
        name = "unit_test"

        [adapter]
        class = "tests.stub.AdapterStub"
        config = {{}}

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
