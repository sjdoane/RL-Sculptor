"""tests/test_heldout.py — §7.4 held-out evaluation battery (offline)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor.eval.heldout import (
    DEFAULT_HELDOUT_SEEDS,
    _perturbed_spec,
    run_heldout_battery,
)


class _FakeAdapter:
    """Rollout stub with the mjlab-shaped env_spec_path lever."""

    def __init__(self):
        self.env_spec_path = ""
        self.calls: list[dict] = []

    def rollout(self, *, checkpoint_path, output_dir, n_episodes, seed=None):
        self.calls.append({
            "spec": self.env_spec_path, "seed": seed,
            "out": Path(output_dir),
        })
        (Path(output_dir) / "behavior.json").write_text("{}")


class _NoLeverAdapter:
    def rollout(self, *, checkpoint_path, output_dir, n_episodes, seed=None):
        (Path(output_dir) / "behavior.json").write_text("{}")


@pytest.fixture
def fake_scoring(monkeypatch):
    """Score = 1.0 minus the push level baked into the rollout dir name —
    a deterministic degradation curve."""
    def fake_compute(metric, rollout_dir):
        name = Path(rollout_dir).parent.name  # push_<L>_seed_<S>
        level = float(name.split("_")[1])
        return {"spec_score": max(0.0, 1.0 - 0.4 * level)}

    monkeypatch.setattr(
        "sculptor.eval.spec_metrics.compute_spec_metrics", fake_compute)
    monkeypatch.setattr(
        "sculptor.eval.spec_metrics.spec_metric_names",
        lambda: ["fake_spec"])


def test_perturbed_spec_zero_level_has_no_pushes():
    spec = _perturbed_spec(None, 0.0)
    assert "push_events" not in spec["shared"]
    assert spec["meta"]["source"] == "heldout:push=0"


def test_perturbed_spec_carries_push_and_validates():
    spec = _perturbed_spec(
        {"env_spec_version": 1, "meta": {}, "shared": {}, "train": {}}, 1.5)
    pe = spec["shared"]["push_events"]
    assert pe["enabled"] is True and pe["linear_mps"] == 1.5


def test_battery_grid_scores_and_degradation(tmp_path, fake_scoring):
    ad = _FakeAdapter()
    report = run_heldout_battery(
        adapter=ad, checkpoint_path=tmp_path / "ckpt.pt",
        out_dir=tmp_path / "battery", metric="fake_spec",
        push_levels=[0.0, 1.0], seeds=[70_001, 70_002], n_episodes=2,
    )
    # 2 levels × 2 seeds = 4 rollouts, each with a per-level spec file.
    assert len(ad.calls) == 4
    specs_used = {c["spec"] for c in ad.calls}
    assert len(specs_used) == 2
    base, pushed = report["levels"]
    assert base["median_score"] == pytest.approx(1.0)
    assert pushed["median_score"] == pytest.approx(0.6)
    assert pushed["degradation_vs_base"] == pytest.approx(0.4)
    assert base["degradation_vs_base"] == pytest.approx(0.0)
    on_disk = json.loads(
        (tmp_path / "battery" / "heldout_report.json").read_text())
    assert on_disk["aggregator"] == "median"
    assert on_disk["seeds"] == [70_001, 70_002]


def test_battery_rejects_non_hand_metric(tmp_path, fake_scoring):
    with pytest.raises(KeyError, match="HAND specs only"):
        run_heldout_battery(
            adapter=_FakeAdapter(), checkpoint_path=tmp_path / "c.pt",
            out_dir=tmp_path / "b", metric="gen:abc123",
        )


def test_cell_failure_is_honest_zero_not_crash(tmp_path, fake_scoring):
    class _Flaky(_FakeAdapter):
        def rollout(self, **kw):
            if kw.get("seed") == 70_002:
                raise RuntimeError("CUDA OOM")
            super().rollout(**kw)

    report = run_heldout_battery(
        adapter=_Flaky(), checkpoint_path=tmp_path / "c.pt",
        out_dir=tmp_path / "b", metric="fake_spec",
        push_levels=[0.0], seeds=[70_001, 70_002],
    )
    errs = [c for c in report["cells"] if "error" in c]
    assert len(errs) == 1 and "CUDA OOM" in errs[0]["error"]
    assert errs[0]["score"] == 0.0
    # aggregate excludes errored cells but reports how many scored
    assert report["levels"][0]["n_scored"] == 1


def test_adapter_without_lever_marks_perturbed_cells(tmp_path, fake_scoring):
    report = run_heldout_battery(
        adapter=_NoLeverAdapter(), checkpoint_path=tmp_path / "c.pt",
        out_dir=tmp_path / "b", metric="fake_spec",
        push_levels=[0.0, 1.0], seeds=[70_001],
    )
    assert report["env_spec_lever_available"] is False
    flags = {c["push_mps"]: c["env_spec_applied"] for c in report["cells"]}
    assert flags[0.0] is True and flags[1.0] is False


def test_default_seed_band_disjoint_from_loop_bands():
    # Loop eval seeds live at 10_000+ and fresh re-eval at 90_001+
    # (sculpt.py) — the held-out band must not collide with either.
    for s in DEFAULT_HELDOUT_SEEDS:
        assert 70_000 <= s < 80_000
