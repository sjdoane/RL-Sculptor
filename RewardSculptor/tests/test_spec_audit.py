"""Adversarial evidence certificates for objective success specs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sculptor.eval.spec_audit import (
    ATTACK_CLASSES,
    SpecAuditError,
    load_and_verify_spec_certificate,
    load_spec_audit_manifest,
    run_spec_audit,
)


def _audit_manifest(
    tmp_path: Path, *, failing_class: str | None = None,
    classes: set[str] | None = None,
) -> Path:
    selected = sorted(classes or ATTACK_CLASSES)
    cases = []
    for index, attack_class in enumerate(selected):
        rollout = tmp_path / "evidence" / f"{index:02d}_{attack_class}"
        rollout.mkdir(parents=True)
        is_positive = attack_class == "competent_positive"
        mean_length = 500 if is_positive else 0
        if attack_class == failing_class:
            mean_length = 500
        (rollout / "behavior.json").write_text(json.dumps({
            "mean_episode_length": mean_length,
            "max_episode_steps": 500,
            "step_dt": 0.02,
            "rollout_num_envs": 64,
        }), encoding="utf-8")
        cases.append({
            "case_id": f"case_{attack_class}",
            "attack_class": attack_class,
            "rollout_dir": str(rollout),
            "expectation": (
                {"min_score": 0.9} if is_positive else {"max_score": 0.1}
            ),
            "notes": "synthetic contract test",
        })
    manifest = {
        "schema_version": 1,
        "audit_id": "cartpole-adversarial-v1",
        "spec_name": "cartpole_balance",
        "authority_target": "A4_reporting",
        "cases": cases,
    }
    path = tmp_path / f"audit_{failing_class or 'pass'}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_reporting_certificate_requires_and_passes_full_attack_battery(
    tmp_path: Path,
) -> None:
    manifest = _audit_manifest(tmp_path)
    normalized = load_spec_audit_manifest(manifest)
    assert {case["attack_class"] for case in normalized["cases"]} == ATTACK_CLASSES
    assert all(len(case["evidence_sha256"]) == 64 for case in normalized["cases"])

    certificate = run_spec_audit(manifest, tmp_path / "certificate")
    assert certificate["passed"] is True
    assert certificate["authority_decision"] == "A4_reporting"
    assert certificate["coverage"]["complete"] is True
    assert certificate["summary"] == {
        "n_cases": 10, "n_passed": 10, "n_failed": 0,
    }
    loaded = load_and_verify_spec_certificate(
        tmp_path / "certificate" / "spec_audit_certificate.json"
    )
    assert loaded["certificate_sha256"] == certificate["certificate_sha256"]
    assert (tmp_path / "certificate" / "spec_audit_report.md").is_file()


def test_proxy_gaming_failure_rejects_requested_authority(tmp_path: Path) -> None:
    manifest = _audit_manifest(tmp_path, failing_class="proxy_only")
    certificate = run_spec_audit(manifest, tmp_path / "failed_certificate")
    assert certificate["passed"] is False
    assert certificate["authority_decision"] == "A0_rejected"
    failed = [case for case in certificate["cases"] if not case["passed"]]
    assert [case["attack_class"] for case in failed] == ["proxy_only"]
    assert "exceeds max_score" in failed[0]["failures"][0]


def test_missing_attack_class_is_visible_and_cannot_grant_reporting(
    tmp_path: Path,
) -> None:
    classes = set(ATTACK_CLASSES) - {"threshold_flicker", "reset_artifact"}
    manifest = _audit_manifest(tmp_path, classes=classes)
    certificate = run_spec_audit(manifest, tmp_path / "incomplete_certificate")
    assert not certificate["passed"]
    assert certificate["summary"]["n_failed"] == 0
    assert certificate["coverage"]["missing_classes"] == [
        "reset_artifact", "threshold_flicker",
    ]


def test_tampering_unknown_fields_and_overwrite_are_rejected(tmp_path: Path) -> None:
    manifest = _audit_manifest(tmp_path)
    out = tmp_path / "certificate"
    run_spec_audit(manifest, out)
    with pytest.raises(SpecAuditError, match="not empty"):
        run_spec_audit(manifest, out)

    certificate_path = out / "spec_audit_certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["passed"] = False
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(SpecAuditError, match="hash mismatch"):
        load_and_verify_spec_certificate(tampered)

    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["cases"][0]["max_socre"] = 0.1
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SpecAuditError, match="unknown fields"):
        load_spec_audit_manifest(typo)
