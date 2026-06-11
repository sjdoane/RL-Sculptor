"""§Ship 25a (H1): criterion ↔ reward-component contract hardening.

Covers: hard-key extraction (`components['x']` vs soft `.get()`),
`reconcile_criterion`'s LLM rewrite + validation gates (stub client —
no network), and the `_reconcile_stage_criterion_if_needed` wiring in
sculpt.py (events, stage mutation, persistence, never-fatal contract).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.adapters.base import ComponentProbe
from sculptor.decompose import _ReconcileModel, reconcile_criterion
from sculptor.mission import MissionValidationError, Stage
from sculptor.mission_runtime import extract_components_keys
from tests.test_decompose import _StubClient


# ── extract_components_keys ──────────────────────────────────────────


def test_extract_hard_subscripts() -> None:
    crit = ("components['hip_sway_osc'] > 0.5 and "
            'components["upright"] >= 0.2')
    assert extract_components_keys(crit) == {"hip_sway_osc", "upright"}


def test_extract_ignores_soft_get() -> None:
    """`components.get('x', 0.0)` cannot KeyError — by design NOT
    collected (it's the documented soft-reference idiom)."""
    crit = ("components.get('future_term', 0.0) > 0.1 and "
            "components['real_term'] > 0.2")
    assert extract_components_keys(crit) == {"real_term"}


def test_extract_ignores_other_namespaces_and_dynamic_keys() -> None:
    crit = ("behavior['mean_return'] > 0.3 and "
            "info['fallen'].mean() < 0.1 and "
            "components[some_var] > 0")
    assert extract_components_keys(crit) == set()


def test_extract_unparseable_returns_empty() -> None:
    assert extract_components_keys("components['x' >") == set()
    assert extract_components_keys("") == set()


# ── reconcile_criterion (stub LLM) ───────────────────────────────────


def _stage(criterion: str = "components['hip_sway_osc'] > 0.5") -> Stage:
    return Stage(
        name="hip_sway",
        goal_text="oscillate the hips laterally",
        success_criterion=criterion,
        max_iterations=5,
        parent_stage=None,
        reward_seed_prompt="hip_sway_reward (lateral oscillation), alive_bonus",
    )


def test_reconcile_happy_path() -> None:
    client = _StubClient(_ReconcileModel(
        rationale="hip_sway_osc was authored as hip_sway_reward",
        success_criterion="components['hip_sway_reward'] > 0.5",
    ))
    new, rationale = reconcile_criterion(
        _stage(),
        missing_keys=["hip_sway_osc"],
        available_components=["hip_sway_reward", "alive_bonus"],
        client=client,
    )
    assert new == "components['hip_sway_reward'] > 0.5"
    assert "hip_sway_reward" in rationale
    # The stub got the full grounding payload.
    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "hip_sway_osc" in user_content
    assert "hip_sway_reward" in user_content
    assert "mean_return" in user_content   # behavior keys offered
    assert "available_trajectory_keys" in user_content
    assert "prior_attempt_error" not in user_content


def _expect_two_attempt_failure(bad: _ReconcileModel, match: str) -> None:
    """Gate failures get ONE feedback retry — feed two bad responses and
    expect the final error; also assert the retry carried the gate error."""
    client = _StubClient(bad, bad)
    with pytest.raises(MissionValidationError, match="failed validation twice"):
        reconcile_criterion(
            _stage(), missing_keys=["hip_sway_osc"],
            available_components=["alive_bonus"], client=client,
        )
    assert len(client.messages.calls) == 2
    retry_content = client.messages.calls[1]["messages"][0]["content"]
    assert "prior_attempt_error" in retry_content
    assert match in retry_content


def test_reconcile_rejects_identical_rewrite() -> None:
    _expect_two_attempt_failure(
        _ReconcileModel(
            rationale="no change",
            success_criterion="components['hip_sway_osc'] > 0.5",
        ),
        match="identical",
    )


def test_reconcile_rejects_still_missing_keys() -> None:
    _expect_two_attempt_failure(
        _ReconcileModel(
            rationale="oops, invented another name",
            success_criterion="components['made_up_term'] > 0.5",
        ),
        match="absent component",
    )


def test_reconcile_rejects_unsafe_rewrite() -> None:
    _expect_two_attempt_failure(
        _ReconcileModel(
            rationale="sneaky",
            success_criterion="(lambda: True)()",
        ),
        match="",
    )


def test_reconcile_rejects_item_method() -> None:
    """`.item()` passes the RUNTIME gate but the mission gate forbids it
    (torch-idiom list) — the prompt must not advertise it, and the gate
    must reject it (Ship 25a audit H-1)."""
    _expect_two_attempt_failure(
        _ReconcileModel(
            rationale="scalar extract",
            success_criterion="components['alive_bonus'].item() > 0.2",
        ),
        match="",
    )


def test_reconcile_retry_recovers() -> None:
    """First rewrite fails a gate; the retry (carrying the gate error)
    succeeds."""
    client = _StubClient(
        _ReconcileModel(
            rationale="bad", success_criterion="components['nope'] > 1",
        ),
        _ReconcileModel(
            rationale="fixed", success_criterion="components['alive_bonus'] > 1",
        ),
    )
    new, rationale = reconcile_criterion(
        _stage(), missing_keys=["hip_sway_osc"],
        available_components=["alive_bonus"], client=client,
    )
    assert new == "components['alive_bonus'] > 1"
    assert rationale == "fixed"
    assert len(client.messages.calls) == 2


def test_reconcile_allows_soft_get_fallback() -> None:
    """A rewrite that hard-refs an available key and soft-refs a future
    one passes the gates."""
    client = _StubClient(_ReconcileModel(
        rationale="alive now, sway later",
        success_criterion=(
            "components['alive_bonus'] > 0.2 and "
            "components.get('hip_sway_osc', 0.0) >= 0.0"
        ),
    ))
    new, _ = reconcile_criterion(
        _stage(), missing_keys=["hip_sway_osc"],
        available_components=["alive_bonus"], client=client,
    )
    assert "components.get('hip_sway_osc', 0.0)" in new


# ── sculpt.py wiring ─────────────────────────────────────────────────


class _ProbeAdapter:
    def __init__(self, probe: ComponentProbe):
        self._probe = probe
        self.probed_paths: list[Path] = []

    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        self.probed_paths.append(Path(reward_module_path))
        return self._probe


def _stage_dir_with_reward(tmp_path: Path) -> Path:
    sd = tmp_path / "stage"
    (sd / "rewards").mkdir(parents=True)
    (sd / "rewards" / "v0.py").write_text("# v0", encoding="utf-8")
    (sd / "rewards" / "v1.py").write_text("# v1", encoding="utf-8")
    return sd


def _run_wiring(tmp_path, monkeypatch, *, stage, probe, reconcile=None,
                api_key: bool = False):
    """Drive _reconcile_stage_criterion_if_needed with everything faked;
    returns (events, save_calls, adapter)."""
    import sculptor.sculpt as sculpt_mod

    events: list[dict] = []
    save_calls: list = []
    monkeypatch.setattr(
        sculpt_mod, "_atomic_save_mission",
        lambda m, d: save_calls.append((m, d)),
    )
    if api_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if reconcile is not None:
        import sculptor.decompose as decompose_mod
        monkeypatch.setattr(decompose_mod, "reconcile_criterion", reconcile)
    adapter = _ProbeAdapter(probe)
    sculpt_mod._reconcile_stage_criterion_if_needed(
        stage=stage,
        stage_dir=_stage_dir_with_reward(tmp_path),
        adapter=adapter,
        mission=object(),
        mission_dir=tmp_path,
        emit=events.append,
    )
    return events, save_calls, adapter


def test_wiring_no_component_refs_is_silent(tmp_path, monkeypatch) -> None:
    stage = _stage("behavior['mean_return'] > 0.3")
    events, saves, adapter = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=True, components={"alive_bonus": 1.0}),
    )
    assert events == []
    assert adapter.probed_paths == []   # probe not even run
    assert saves == []


def test_wiring_validated_when_keys_exist(tmp_path, monkeypatch) -> None:
    stage = _stage("components['alive_bonus'] > 0.2")
    events, saves, adapter = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=True, components={"alive_bonus": 1.0}),
    )
    assert [e["type"] for e in events] == ["criterion_keys_validated"]
    assert adapter.probed_paths and adapter.probed_paths[0].name == "v1.py"
    assert saves == []


def test_wiring_mismatch_without_key_skips_llm(tmp_path, monkeypatch) -> None:
    stage = _stage()  # references hip_sway_osc
    events, saves, _ = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=True, components={"alive_bonus": 1.0}),
        api_key=False,
    )
    types = [e["type"] for e in events]
    assert types == ["criterion_keys_mismatch", "criterion_reconcile_skipped"]
    mismatch = events[0]
    assert mismatch["missing"] == ["hip_sway_osc"]
    assert mismatch["available"] == ["alive_bonus"]
    assert stage.success_criterion == "components['hip_sway_osc'] > 0.5"
    assert saves == []


def test_wiring_reconciles_and_persists(tmp_path, monkeypatch) -> None:
    stage = _stage()

    def fake_reconcile(s, *, missing_keys, available_components):
        assert missing_keys == ["hip_sway_osc"]
        assert available_components == ["alive_bonus"]
        return "components['alive_bonus'] > 0.5", "mapped osc to bonus"

    events, saves, _ = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=True, components={"alive_bonus": 1.0}),
        reconcile=fake_reconcile, api_key=True,
    )
    types = [e["type"] for e in events]
    assert types == ["criterion_keys_mismatch", "criterion_reconciled"]
    assert stage.success_criterion == "components['alive_bonus'] > 0.5"
    assert events[1]["old_criterion"] == "components['hip_sway_osc'] > 0.5"
    assert events[1]["rationale"] == "mapped osc to bonus"
    assert len(saves) == 1, "reconciled criterion must be persisted"


def test_wiring_save_failure_reverts_mutation(tmp_path, monkeypatch) -> None:
    """M-1 regression: if persisting the reconciled criterion fails, the
    in-memory stage must NOT keep the rewrite (memory≠disk would make
    this run evaluate a criterion a resume never sees)."""
    import sculptor.sculpt as sculpt_mod

    stage = _stage()
    events: list[dict] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import sculptor.decompose as decompose_mod
    monkeypatch.setattr(
        decompose_mod, "reconcile_criterion",
        lambda s, **kw: ("components['alive_bonus'] > 0.5", "r"),
    )

    def exploding_save(m, d):
        raise OSError("disk full")

    monkeypatch.setattr(sculpt_mod, "_atomic_save_mission", exploding_save)
    adapter = _ProbeAdapter(
        ComponentProbe(ok=True, components={"alive_bonus": 1.0}))
    sculpt_mod._reconcile_stage_criterion_if_needed(
        stage=stage, stage_dir=_stage_dir_with_reward(tmp_path),
        adapter=adapter, mission=object(), mission_dir=tmp_path,
        emit=events.append,
    )
    types = [e["type"] for e in events]
    assert types == ["criterion_keys_mismatch", "criterion_reconcile_failed"]
    assert stage.success_criterion == "components['hip_sway_osc'] > 0.5"


def test_wiring_probe_raise_is_unverified(tmp_path, monkeypatch) -> None:
    """L-1 regression: a probe that RAISES (vs ok=False) is an
    unverified outcome, not a reconcile failure."""
    import sculptor.sculpt as sculpt_mod

    class _ExplodingAdapter:
        def probe_component(self, p):
            raise RuntimeError("subprocess died")

    stage = _stage()
    events: list[dict] = []
    monkeypatch.setattr(sculpt_mod, "_atomic_save_mission", lambda m, d: None)
    sculpt_mod._reconcile_stage_criterion_if_needed(
        stage=stage, stage_dir=_stage_dir_with_reward(tmp_path),
        adapter=_ExplodingAdapter(), mission=object(), mission_dir=tmp_path,
        emit=events.append,
    )
    assert [e["type"] for e in events] == ["criterion_keys_unverified"]
    assert "subprocess died" in events[0]["reason"]


def test_parent_env_terms_union(tmp_path, monkeypatch) -> None:
    """M-3: env intrinsic terms observed in the parent stage's rollout
    count as available — a redecomposed sub-stage criterion referencing
    them must NOT be flagged."""
    import numpy as np
    import sculptor.sculpt as sculpt_mod

    parent_dir = tmp_path / "stages" / "parent"
    roll = parent_dir / "runs" / "iter_2" / "rollout"
    roll.mkdir(parents=True)
    np.savez(
        roll / "trajectory.npz",
        reward_term__track_velocity=np.zeros(3),
        base_height=np.zeros(3),
    )

    class _FakeMission:
        def stage_dir(self, name):
            return tmp_path / "stages" / name

    stage = _stage("components['track_velocity'] > 0.1")
    stage.parent_stage = "parent"
    events: list[dict] = []
    monkeypatch.setattr(sculpt_mod, "_atomic_save_mission", lambda m, d: None)
    adapter = _ProbeAdapter(
        ComponentProbe(ok=True, components={"alive_bonus": 1.0}))
    sculpt_mod._reconcile_stage_criterion_if_needed(
        stage=stage, stage_dir=_stage_dir_with_reward(tmp_path),
        adapter=adapter, mission=_FakeMission(), mission_dir=tmp_path,
        emit=events.append,
    )
    assert [e["type"] for e in events] == ["criterion_keys_validated"]
    assert events[0]["env_terms_considered"] == ["track_velocity"]


def test_criterion_eval_strips_leading_whitespace() -> None:
    """L-6 drive-by: a leading-whitespace criterion must evaluate, not
    die with 'unexpected indent' (mission gate strips; runtime didn't)."""
    from sculptor.mission_runtime import _evaluate_success_criterion

    assert _evaluate_success_criterion("  metric > 0.5", {"metric": 1.0})


def test_wiring_reconcile_failure_is_recoverable(tmp_path, monkeypatch) -> None:
    stage = _stage()

    def exploding_reconcile(s, **kw):
        raise MissionValidationError("rewrite was bad")

    events, saves, _ = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=True, components={"alive_bonus": 1.0}),
        reconcile=exploding_reconcile, api_key=True,
    )
    types = [e["type"] for e in events]
    assert types == ["criterion_keys_mismatch", "criterion_reconcile_failed"]
    assert stage.success_criterion == "components['hip_sway_osc'] > 0.5"
    assert saves == []


def test_wiring_probe_failure_is_unverified(tmp_path, monkeypatch) -> None:
    stage = _stage()
    events, saves, _ = _run_wiring(
        tmp_path, monkeypatch, stage=stage,
        probe=ComponentProbe(ok=False, error="import crashed"),
    )
    assert [e["type"] for e in events] == ["criterion_keys_unverified"]
    assert "import crashed" in events[0]["reason"]
    assert saves == []
