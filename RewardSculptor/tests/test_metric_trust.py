"""§Ship 52: the standardized trust score + unified grant decision.

trust is a DISPLAY confidence (built-in + task-derived on one scale); the
GRANT stays each path's own gate ANDed with validate ∧ axioms — byte-identical
to today for the 5 built-ins.
"""
from __future__ import annotations

import pytest

from sculptor.eval.metric_calibration import compute_trust, grant_decision


# ── compute_trust ─────────────────────────────────────────────────────


def test_trust_builtin_at_threshold():
    """Built-in rho=0.7 (the grant boundary), validate+axioms pass →
    CAL=clip((0.7-0.5)/0.5)=0.4, EVID=1, trust=0.6·0.4+0.4·1=0.64."""
    cal = {"ok": True, "spearman": 0.7, "builtin": "g1_kick"}
    val = {"ok": True, "axioms": {"ok": True}}
    t = compute_trust(cal, val)
    assert t["cal"] == pytest.approx(0.4) and t["evid"] == pytest.approx(1.0)
    assert t["trust"] == pytest.approx(0.64)
    assert t["method"] == "builtin" and t["agreement_fraction"] == 1.0


def test_trust_perfect_builtin_is_one():
    t = compute_trust({"ok": True, "spearman": 1.0}, {"ok": True, "axioms": {"ok": True}})
    assert t["trust"] == pytest.approx(1.0)


def test_trust_task_derived_floor_is_low():
    """A floor task-derived grant (rho_min=0.5, agreement=2/3) is a LOW-
    confidence grant — trust≈0.27 — which is exactly why trust≥0.7 cannot be
    the gate (it would deny a legitimate task-derived grant)."""
    cal = {"ok": True, "method": "task_derived", "rho_min": 0.5,
           "agreement_fraction": 2 / 3}
    t = compute_trust(cal, {"ok": True, "axioms": {"ok": True}})
    assert t["cal"] == pytest.approx(0.0)
    assert t["evid"] == pytest.approx(2 / 3, abs=1e-3)        # = agreement_fraction
    assert t["trust"] == pytest.approx(0.4 * (2 / 3), abs=1e-3)
    assert t["trust"] < 0.7


def test_trust_strong_task_derived():
    cal = {"ok": True, "method": "task_derived", "rho_min": 1.0,
           "agreement_fraction": 1.0}
    t = compute_trust(cal, {"ok": True, "axioms": {"ok": True}})
    assert t["trust"] == pytest.approx(1.0)


def test_trust_axiom_failure_sinks_evid():
    cal = {"ok": True, "spearman": 1.0}
    t = compute_trust(cal, {"ok": True, "axioms": {"ok": False}})
    assert t["evid"] == 0.0
    assert t["trust"] == pytest.approx(0.6)            # CAL only


# ── grant_decision: byte-identical for built-ins + defense-in-depth ───


def test_grant_byte_identical_builtin():
    val = {"ok": True, "axioms": {"ok": True}}         # an accepted metric
    assert grant_decision({"ok": True, "spearman": 0.7}, val) is True
    assert grant_decision({"ok": False, "spearman": 0.5}, val) is False
    # No validation record (legacy) → falls back to the calibration ok.
    assert grant_decision({"ok": True}, None) is True


def test_grant_blocks_when_a_gate_failed():
    """Defense in depth: even if calibration says ok, a persisted validate/
    axiom failure denies the grant (can't happen for an accepted metric, but
    the firewall must never silently steer a record that shows a failed gate)."""
    cal = {"ok": True, "spearman": 1.0}
    assert grant_decision(cal, {"ok": False, "axioms": {"ok": True}}) is False
    assert grant_decision(cal, {"ok": True, "axioms": {"ok": False}}) is False


def test_grant_task_derived():
    val = {"ok": True, "axioms": {"ok": True}}
    assert grant_decision({"ok": True, "method": "task_derived", "rho_min": 0.6}, val) is True
    assert grant_decision({"ok": False, "method": "task_derived", "rho_min": 0.4}, val) is False
