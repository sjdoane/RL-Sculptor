"""Strict, capability-described external evaluation benchmark manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor.eval import CampaignConfig
from sculptor.eval.benchmarks import (
    BENCHMARKS,
    BenchmarkManifestError,
    benchmark_registry,
    load_benchmark_manifest,
)
from sculptor.eval.spec_audit import ATTACK_CLASSES, run_spec_audit


def _manifest_doc(
    *, name: str = "external_cartpole", certificate: str = "missing.json",
) -> dict:
    return {
        "schema_version": 1,
        "suite_id": "test-suite",
        "suite_version": "1.0.0",
        "benchmarks": [{
            "name": name,
            "task_id": "Mjlab-Cartpole-Balance",
            "behavior_goal": "balance the pole under a held-out reset split",
            "spec_metric": "cartpole_balance",
            "adapter": "mjlab",
            "adapter_config": {"num_envs": 64},
            "robot_id": "cartpole_variant",
            "embodiment_family": "underactuated_system",
            "task_family": "control_sanity",
            "required_capabilities": ["balance"],
            "evaluation_tier": "rollout_artifact",
            "campaign_ready": True,
            "spec_authority": "A4_reporting",
            "spec_audit_certificate": certificate,
            "known_limitations": [],
            "notes": "test-only external entry",
        }],
    }


def _write(tmp_path: Path, doc: dict, name: str = "suite.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _certificate(tmp_path: Path) -> Path:
    cases = []
    for index, attack_class in enumerate(sorted(ATTACK_CLASSES)):
        rollout = tmp_path / "audit_evidence" / str(index)
        rollout.mkdir(parents=True)
        positive = attack_class == "competent_positive"
        (rollout / "behavior.json").write_text(json.dumps({
            "mean_episode_length": 500 if positive else 0,
            "max_episode_steps": 500,
        }))
        cases.append({
            "case_id": f"case_{index}",
            "attack_class": attack_class,
            "rollout_dir": str(rollout),
            "expectation": {"min_score": 0.9} if positive else {"max_score": 0.1},
        })
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": 1,
        "audit_id": "external-cartpole-a4",
        "spec_name": "cartpole_balance",
        "authority_target": "A4_reporting",
        "cases": cases,
    }))
    run_spec_audit(audit, tmp_path / "audit_output")
    return tmp_path / "audit_output" / "spec_audit_certificate.json"


def test_frontier_manifest_exposes_real_arm_tasks_without_false_readiness() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs" / "benchmarks" / "cross_embodiment_frontier_v1.json"
    )
    loaded = load_benchmark_manifest(path)
    assert set(loaded) == {"yam_lift_cube", "yam_multi_cube_lift"}
    lift = loaded["yam_lift_cube"]
    assert lift.embodiment_family == "robot_arm_with_parallel_gripper"
    assert "grasp" in lift.required_capabilities
    assert lift.evaluation_tier == "compile_only"
    assert lift.spec_metric is None
    assert not lift.campaign_ready
    assert lift.spec_authority == "A0_rejected"
    assert any("object pose" in limitation for limitation in lift.known_limitations)


def test_compile_only_frontier_cannot_enter_campaign() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs" / "benchmarks" / "cross_embodiment_frontier_v1.json"
    )
    registry = benchmark_registry([path])
    cfg = CampaignConfig(
        name="reject-frontier",
        out_dir=Path("unused"),
        benchmarks=["yam_lift_cube"],
        conditions=["plain_ppo"],
        seeds=[1],
        benchmark_manifests=[path],
    )
    with pytest.raises(ValueError, match="compile_only.*not campaign-ready"):
        cfg.validate(registry)


def test_valid_external_campaign_entry_and_non_override(tmp_path: Path) -> None:
    certificate = _certificate(tmp_path)
    path = _write(tmp_path, _manifest_doc(certificate=str(certificate)))
    registry = benchmark_registry([path])
    assert set(BENCHMARKS).issubset(registry)
    assert registry["external_cartpole"].robot_id == "cartpole_variant"
    cfg = CampaignConfig(
        name="external",
        out_dir=tmp_path / "out",
        benchmarks=["external_cartpole"],
        conditions=["plain_ppo"],
        seeds=[1],
        benchmark_manifests=[path],
    )
    cfg.validate(registry)

    conflict = _write(
        tmp_path,
        _manifest_doc(name="cartpole_balance", certificate=str(certificate)),
        "conflict.json",
    )
    with pytest.raises(BenchmarkManifestError, match="overrides are forbidden"):
        benchmark_registry([conflict])


def test_manifest_rejects_placeholder_authority_and_unknown_fields(
    tmp_path: Path,
) -> None:
    doc = _manifest_doc()
    item = doc["benchmarks"][0]
    item["evaluation_tier"] = "compile_only"
    item["campaign_ready"] = False
    # A compile-only task may not carry a plausible-looking metric name.
    with pytest.raises(BenchmarkManifestError, match="spec_metric=null"):
        load_benchmark_manifest(_write(tmp_path, doc, "placeholder.json"))

    typo = _manifest_doc()
    typo["benchmarks"][0]["campain_ready"] = True
    with pytest.raises(BenchmarkManifestError, match="unknown fields"):
        load_benchmark_manifest(_write(tmp_path, typo, "typo.json"))
