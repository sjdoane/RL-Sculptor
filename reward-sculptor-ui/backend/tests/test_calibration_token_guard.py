"""§round-5 review: the calibration token guard.

The live launch path runs task-derived calibration under `asyncio.wait_for(to_thread(...),
timeout=300)`. wait_for CANNOT cancel the worker thread, so a calibration that genuinely
succeeds AFTER the timeout would (pre-fix) write `calibrated=true` to meta.json behind the
"observe-only" verdict already shown to the user — and a later run/resume would then steer
on it. `stamp_cal_token` + `expect_token` make the persist conditional: a timed-out launch
re-stamps the token, so the orphaned thread's late write becomes a no-op.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.services import metric_store, sculptor_bridge


def _make_metric(tmp_path: Path, gid: str = "gen_001") -> Path:
    d = tmp_path / "metrics" / gid
    d.mkdir(parents=True)
    (d / "metric.py").write_text("def compute_spec(a,b,m):\n    return {'spec_score':0.0}\n")
    (d / "meta.json").write_text(json.dumps(
        {"id": gid, "accepted": True, "validation": {"ok": True, "axioms": {"ok": True}}}))
    return d


def _grant_cal(*a, **k):
    return {"ok": True, "method": "task_derived", "rho_min": 0.9, "agreement_fraction": 1.0}


def _patch_grant(monkeypatch):
    monkeypatch.setattr(sculptor_bridge, "calibrate_task_derived_metric", _grant_cal)
    monkeypatch.setattr(sculptor_bridge, "finalize_calibration",
                        lambda cal, val: {"calibrated": True, "trust": {"trust": 0.9}})


def test_stale_token_does_not_persist_grant(tmp_path, monkeypatch):
    """An orphaned thread whose token was superseded (timeout re-stamp) must NOT write
    calibrated=true — the observe-only persisted state is preserved."""
    _make_metric(tmp_path)
    _patch_grant(monkeypatch)
    token = metric_store.stamp_cal_token(tmp_path, "gen_001")   # the launch's token
    metric_store.stamp_cal_token(tmp_path, "gen_001")           # timeout re-stamp (supersede)
    out = metric_store.calibrate_task_derived(
        tmp_path, "gen_001", "touch your toes", expect_token=token)  # the orphan, late
    assert out["calibrated"] is not True
    meta = json.loads((tmp_path / "metrics/gen_001/meta.json").read_text())
    assert meta.get("calibrated") is not True                  # NOT resurrected


def test_matching_token_persists_grant(tmp_path, monkeypatch):
    """The normal (in-time) case: the token still matches → the genuine grant persists."""
    _make_metric(tmp_path)
    _patch_grant(monkeypatch)
    token = metric_store.stamp_cal_token(tmp_path, "gen_001")
    out = metric_store.calibrate_task_derived(
        tmp_path, "gen_001", "touch your toes", expect_token=token)
    assert out["calibrated"] is True
    meta = json.loads((tmp_path / "metrics/gen_001/meta.json").read_text())
    assert meta.get("calibrated") is True


def test_no_token_persists_unconditionally(tmp_path, monkeypatch):
    """The explicit, synchronous calibrate-route path passes no token → unconditional
    persist (there is no orphan to guard against) — byte-identical to before."""
    _make_metric(tmp_path)
    _patch_grant(monkeypatch)
    out = metric_store.calibrate_task_derived(tmp_path, "gen_001", "touch your toes")
    assert out["calibrated"] is True
    meta = json.loads((tmp_path / "metrics/gen_001/meta.json").read_text())
    assert meta.get("calibrated") is True
