"""Adversarial evidence certificates for objective success specifications.

A metric implementation existing in ``spec_metrics.py`` is not sufficient
authority for a research claim. This module evaluates frozen rollout artifacts
against predeclared positive and exploit expectations and records exactly which
attack classes support a requested authority level.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from sculptor.eval.spec_metrics import compute_spec_metrics, spec_metric_names
from sculptor.run_context import write_json_atomic


SPEC_AUDIT_SCHEMA_VERSION = 1
AUTHORITY_LEVELS = ("A1_descriptive", "A2_advisory", "A3_steering", "A4_reporting")
ATTACK_CLASSES = frozenset({
    "competent_positive",
    "stillness",
    "falling",
    "oscillation",
    "explosion",
    "early_termination",
    "threshold_flicker",
    "reset_artifact",
    "time_truncation",
    "proxy_only",
})
AUTHORITY_COVERAGE: dict[str, frozenset[str]] = {
    "A1_descriptive": frozenset({"competent_positive"}),
    "A2_advisory": frozenset({
        "competent_positive", "stillness", "falling", "explosion",
    }),
    "A3_steering": frozenset({
        "competent_positive", "stillness", "falling", "oscillation",
        "explosion", "early_termination", "proxy_only",
    }),
    "A4_reporting": ATTACK_CLASSES,
}
_INPUT_FILENAMES = ("behavior.json", "trajectory.npz", "mjcf_limits.json")
_TOP_FIELDS = {
    "schema_version", "audit_id", "spec_name", "authority_target", "cases",
    "notes",
}
_CASE_FIELDS = {
    "case_id", "attack_class", "rollout_dir", "expectation", "notes",
}
_EXPECTATION_FIELDS = {"min_score", "max_score"}


class SpecAuditError(ValueError):
    """The audit design or its frozen evidence is malformed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(raw: Mapping[str, Any], field: str, context: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SpecAuditError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise SpecAuditError(
            f"{context} has unknown fields {unknown}; refusing possible typos"
        )


def _score_threshold(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SpecAuditError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecAuditError(f"{context} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SpecAuditError(f"{context} must be finite and in [0, 1]")
    return result


def _evidence_hash(rollout_dir: Path) -> tuple[str, dict[str, str]]:
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for name in _INPUT_FILENAMES:
        path = rollout_dir / name
        if not path.is_file():
            continue
        file_hash = _hash_file(path)
        files[name] = file_hash
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
    if not files:
        raise SpecAuditError(
            f"audit evidence has none of the evaluator inputs {_INPUT_FILENAMES}: "
            f"{rollout_dir}"
        )
    return digest.hexdigest(), files


def load_spec_audit_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecAuditError(f"cannot read spec audit manifest {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise SpecAuditError("spec audit manifest must be a JSON object")
    _reject_unknown(doc, _TOP_FIELDS, "manifest")
    if doc.get("schema_version") != SPEC_AUDIT_SCHEMA_VERSION:
        raise SpecAuditError(
            f"manifest.schema_version must be {SPEC_AUDIT_SCHEMA_VERSION}"
        )
    _text(doc, "audit_id", "manifest")
    spec_name = _text(doc, "spec_name", "manifest")
    if spec_name not in spec_metric_names():
        raise SpecAuditError(
            f"unknown spec_name {spec_name!r}; known: {spec_metric_names()}"
        )
    authority = _text(doc, "authority_target", "manifest")
    if authority not in AUTHORITY_LEVELS:
        raise SpecAuditError(
            f"authority_target must be one of {list(AUTHORITY_LEVELS)}"
        )
    raw_cases = doc.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SpecAuditError("manifest.cases must be a non-empty list")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        context = f"cases[{index}]"
        if not isinstance(raw, dict):
            raise SpecAuditError(f"{context} must be an object")
        _reject_unknown(raw, _CASE_FIELDS, context)
        case_id = _text(raw, "case_id", context)
        if case_id in seen_ids:
            raise SpecAuditError(f"duplicate case_id {case_id!r}")
        seen_ids.add(case_id)
        attack_class = _text(raw, "attack_class", context)
        if attack_class not in ATTACK_CLASSES:
            raise SpecAuditError(
                f"{context}.attack_class must be one of {sorted(ATTACK_CLASSES)}"
            )
        rollout_raw = _text(raw, "rollout_dir", context)
        rollout_dir = Path(rollout_raw)
        if not rollout_dir.is_absolute():
            rollout_dir = path.parent / rollout_dir
        rollout_dir = rollout_dir.resolve()
        if not rollout_dir.is_dir():
            raise SpecAuditError(
                f"{context}.rollout_dir is not a directory: {rollout_dir}"
            )
        expectation = raw.get("expectation")
        if not isinstance(expectation, dict) or not expectation:
            raise SpecAuditError(f"{context}.expectation must be a non-empty object")
        _reject_unknown(expectation, _EXPECTATION_FIELDS, f"{context}.expectation")
        normalized_expectation = {
            key: _score_threshold(value, f"{context}.expectation.{key}")
            for key, value in expectation.items()
        }
        if attack_class == "competent_positive":
            if "min_score" not in normalized_expectation:
                raise SpecAuditError(
                    f"{context}: competent_positive requires min_score"
                )
        elif "max_score" not in normalized_expectation:
            raise SpecAuditError(
                f"{context}: adversarial cases require max_score"
            )
        if (
            "min_score" in normalized_expectation
            and "max_score" in normalized_expectation
            and normalized_expectation["min_score"]
            > normalized_expectation["max_score"]
        ):
            raise SpecAuditError(
                f"{context}: min_score cannot exceed max_score"
            )
        evidence_hash, input_hashes = _evidence_hash(rollout_dir)
        cases.append({
            "case_id": case_id,
            "attack_class": attack_class,
            "rollout_dir": str(rollout_dir),
            "expectation": normalized_expectation,
            "evidence_sha256": evidence_hash,
            "evaluator_input_sha256": input_hashes,
            "notes": str(raw.get("notes") or ""),
        })
    return {
        "schema_version": SPEC_AUDIT_SCHEMA_VERSION,
        "audit_id": str(doc["audit_id"]),
        "spec_name": spec_name,
        "authority_target": authority,
        "manifest_path": str(path),
        "manifest_sha256": _hash_file(path),
        "notes": str(doc.get("notes") or ""),
        "cases": cases,
    }


def run_spec_audit(manifest_path: Path | str, out_dir: Path | str) -> dict[str, Any]:
    """Evaluate all frozen cases and emit a tamper-evident certificate."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SpecAuditError(
            f"audit output directory is not empty: {out_dir}; use a fresh "
            "directory so an earlier certificate cannot be silently replaced"
        )
    manifest = load_spec_audit_manifest(manifest_path)
    observed_classes = Counter(case["attack_class"] for case in manifest["cases"])
    required = AUTHORITY_COVERAGE[manifest["authority_target"]]
    missing_classes = sorted(required - set(observed_classes))
    case_results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        observed = compute_spec_metrics(
            manifest["spec_name"], Path(case["rollout_dir"]),
        )
        score_raw = observed.get("spec_score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = float("nan")
        failures: list[str] = []
        if observed.get("error"):
            failures.append(f"metric error: {observed['error']}")
        finite_score = math.isfinite(score)
        if not finite_score:
            failures.append(f"non-finite spec_score: {score_raw!r}")
        expected = case["expectation"]
        if "min_score" in expected and not score >= expected["min_score"]:
            failures.append(
                f"score {score:.6g} is below min_score {expected['min_score']:.6g}"
            )
        if "max_score" in expected and not score <= expected["max_score"]:
            failures.append(
                f"score {score:.6g} exceeds max_score {expected['max_score']:.6g}"
            )
        serializable_observed = dict(observed)
        if not finite_score:
            serializable_observed["spec_score"] = None
        case_results.append({
            **case,
            "observed": serializable_observed,
            "passed": not failures,
            "failures": failures,
        })

    all_cases_passed = all(case["passed"] for case in case_results)
    passed = all_cases_passed and not missing_classes
    certificate: dict[str, Any] = {
        "schema_version": SPEC_AUDIT_SCHEMA_VERSION,
        "audit_id": manifest["audit_id"],
        "spec_name": manifest["spec_name"],
        "authority_target": manifest["authority_target"],
        "authority_decision": (
            manifest["authority_target"] if passed else "A0_rejected"
        ),
        "passed": passed,
        "source_manifest": {
            "path": manifest["manifest_path"],
            "sha256": manifest["manifest_sha256"],
        },
        "coverage": {
            "required_classes": sorted(required),
            "observed_case_counts": dict(sorted(observed_classes.items())),
            "missing_classes": missing_classes,
            "complete": not missing_classes,
        },
        "summary": {
            "n_cases": len(case_results),
            "n_passed": sum(case["passed"] for case in case_results),
            "n_failed": sum(not case["passed"] for case in case_results),
        },
        "cases": case_results,
        "limitations": (
            "This certificate grants authority only for the recorded task, "
            "robot, sensor/artifact contract, evidence hashes, attack classes, "
            "and thresholds. New counterexamples or dependency changes require "
            "a new certificate."
        ),
    }
    certificate["certificate_sha256"] = _hash_json(certificate)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "spec_audit_certificate.json", certificate)
    (out_dir / "spec_audit_report.md").write_text(
        _certificate_markdown(certificate), encoding="utf-8",
    )
    return certificate


def load_and_verify_spec_certificate(path: Path | str) -> dict[str, Any]:
    """Verify a persisted certificate before treating it as evidence."""
    try:
        certificate = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecAuditError(f"cannot read spec audit certificate {path}: {exc}") from exc
    if not isinstance(certificate, dict):
        raise SpecAuditError("spec audit certificate must be a JSON object")
    stored = certificate.get("certificate_sha256")
    unhashed = {
        key: value for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    if stored != _hash_json(unhashed):
        raise SpecAuditError(
            "spec audit certificate hash mismatch; the evidence record was altered"
        )
    return certificate


def _certificate_markdown(certificate: Mapping[str, Any]) -> str:
    mark = "PASS" if certificate["passed"] else "FAIL"
    lines = [
        f"# Spec audit: {certificate['audit_id']}",
        "",
        f"**Decision: {mark} — {certificate['authority_decision']}**",
        "",
        f"Metric: `{certificate['spec_name']}`  ",
        f"Certificate SHA-256: `{certificate['certificate_sha256']}`",
        "",
        "## Coverage",
        "",
        f"Missing required classes: `{certificate['coverage']['missing_classes']}`",
        "",
        "## Cases",
        "",
        "| Case | Class | Score | Expected | Result |",
        "|---|---|---:|---|---|",
    ]
    for case in certificate["cases"]:
        observed = case["observed"]
        raw_score = observed.get("spec_score")
        score_text = "non-finite" if raw_score is None else f"{float(raw_score):.4f}"
        lines.append(
            f"| {case['case_id']} | {case['attack_class']} | "
            f"{score_text} | "
            f"`{case['expectation']}` | "
            f"{'pass' if case['passed'] else '; '.join(case['failures'])} |"
        )
    lines.extend(["", certificate["limitations"], ""])
    return "\n".join(lines)
