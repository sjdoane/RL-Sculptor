"""Immutable campaign charters for defensible evaluation runs.

An eval output directory is an experimental identity, not merely a cache.
The charter freezes that identity *before* the first training job and rejects
resumes whose design or executable source has changed.  This prevents a
particularly dangerous failure mode: combining old ``result.json`` files with
new jobs under a silently modified benchmark, condition, or analysis plan.

The charter is tamper-evident rather than access-control based.  It carries a
hash of its scientific design and a second hash of the complete document.
Those hashes are verified on every resume and report regeneration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from sculptor.run_context import write_json_atomic


CHARTER_SCHEMA_VERSION = 1
CHARTER_FILENAME = "campaign_charter.json"


class CharterError(RuntimeError):
    """Base class for campaign-integrity failures."""


class CharterMismatchError(CharterError):
    """The requested experiment differs from the frozen experiment."""


class CharterIntegrityError(CharterError):
    """A stored charter or result is malformed or has been altered."""


class UncharteredCampaignError(CharterError):
    """Artifacts already exist, so a charter cannot honestly be backdated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _source_tree_hash() -> tuple[str, list[str]]:
    """Hash behavior-bearing package and dependency-lock files.

    This deliberately does not use ``git status``: a dirty or untracked source
    file still changes the experiment and must be pinned.  Documentation,
    output artifacts, caches, and tests are outside the runtime identity.
    """
    package_dir = Path(__file__).resolve().parents[1]
    project_dir = package_dir.parent
    files = [
        p for p in package_dir.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".pyo"}
    ]
    files.extend(
        p for p in (project_dir / "pyproject.toml", project_dir / "uv.lock")
        if p.is_file()
    )
    files = sorted(files, key=lambda p: p.relative_to(project_dir).as_posix())
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in files:
        rel = path.relative_to(project_dir).as_posix()
        relative_paths.append(rel)
        raw_name = rel.encode("utf-8")
        data = path.read_bytes()
        # Length framing prevents ambiguous path/content concatenations.
        digest.update(len(raw_name).to_bytes(8, "big"))
        digest.update(raw_name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest(), relative_paths


def _snapshot(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot snapshot {type(value).__name__}")


def build_campaign_design(
    cfg: Any,
    *,
    benchmarks: Mapping[str, Any],
    conditions: Mapping[str, Any],
    external_inputs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the scientific portion of a campaign charter.

    ``out_dir`` is excluded because moving an otherwise byte-identical
    experiment does not change its design.  The ordered benchmark, condition,
    and seed lists remain included because they define the execution plan.
    """
    config = _snapshot(cfg)
    config.pop("out_dir", None)
    source_hash, source_files = _source_tree_hash()
    requested_benchmarks = [
        _snapshot(benchmarks[name]) for name in cfg.benchmarks
    ]
    requested_conditions = [
        _snapshot(conditions[name]) for name in cfg.conditions
    ]
    return {
        "campaign": config,
        "benchmarks": requested_benchmarks,
        "conditions": requested_conditions,
        "analysis_protocol": {
            "primary_outcome": "final_spec_score",
            "secondary_outcomes": [
                "best_spec_score",
                "iters_to_threshold",
                "threshold_hit_rate",
                "total_rl_iterations",
                "wall_seconds",
            ],
            "aggregate": "interquartile_mean",
            "confidence_interval": {
                "method": "percentile_bootstrap_over_seed_units",
                "n_boot": 2000,
                "alpha": 0.05,
                "rng_seed": 0,
            },
            "comparison": "paired_per_seed_condition_differences",
            "failed_job_policy": "retain_and_score_zero",
            "capture_parity_policy": "warn_on_mixed_or_missing_settings",
            "final_output_rules": {
                "eureka": "best_across_generations",
                "fitness_guided_sculpt_or_mission": "best_across_iterations",
                "other_conditions": "last_iteration",
            },
            "spec_threshold": float(cfg.spec_threshold),
        },
        "runtime_identity": {
            "method": "sha256_length_framed_relative_path_and_bytes",
            "source_tree_sha256": source_hash,
            "source_file_count": len(source_files),
            "source_roots": ["sculptor/", "pyproject.toml", "uv.lock"],
        },
        "external_inputs": dict(external_inputs or {}),
    }


def _document_with_hashes(design: Mapping[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": CHARTER_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "design": dict(design),
        "design_sha256": _sha256(design),
    }
    doc["document_sha256"] = _sha256(doc)
    return doc


def load_and_verify_charter(path: Path) -> dict[str, Any]:
    """Load a charter and verify its schema plus both integrity hashes."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CharterIntegrityError(f"campaign charter is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise CharterIntegrityError(
            f"campaign charter is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise CharterIntegrityError("campaign charter must be a JSON object")
    if doc.get("schema_version") != CHARTER_SCHEMA_VERSION:
        raise CharterIntegrityError(
            "unsupported campaign charter schema: "
            f"{doc.get('schema_version')!r} (expected {CHARTER_SCHEMA_VERSION})"
        )
    if doc.get("status") != "frozen" or not isinstance(doc.get("design"), dict):
        raise CharterIntegrityError("campaign charter is not a frozen design")

    stored_document_hash = doc.get("document_sha256")
    unhashed = {k: v for k, v in doc.items() if k != "document_sha256"}
    actual_document_hash = _sha256(unhashed)
    if stored_document_hash != actual_document_hash:
        raise CharterIntegrityError(
            "campaign charter document hash mismatch; the charter was altered "
            "or incompletely written"
        )
    actual_design_hash = _sha256(doc["design"])
    if doc.get("design_sha256") != actual_design_hash:
        raise CharterIntegrityError(
            "campaign charter design hash mismatch; the frozen design was altered"
        )
    return doc


def _differences(stored: Any, current: Any, path: str = "design") -> list[str]:
    """Return compact, deterministic leaf-level differences."""
    if isinstance(stored, dict) and isinstance(current, dict):
        out: list[str] = []
        for key in sorted(set(stored) | set(current)):
            child = f"{path}.{key}"
            if key not in stored:
                out.append(f"{child}: added {current[key]!r}")
            elif key not in current:
                out.append(f"{child}: removed (was {stored[key]!r})")
            else:
                out.extend(_differences(stored[key], current[key], child))
        return out
    if isinstance(stored, list) and isinstance(current, list):
        if stored == current:
            return []
        # Lists encode ordered plans; report them atomically so index shifts do
        # not produce dozens of misleading leaf differences.
        return [f"{path}: {stored!r} -> {current!r}"]
    if stored != current:
        return [f"{path}: {stored!r} -> {current!r}"]
    return []


def _has_prior_results(out_dir: Path) -> bool:
    return (
        (out_dir / "campaign_report.json").exists()
        or any(out_dir.glob("*/*/seed_*/result.json"))
    )


def freeze_or_verify_campaign_charter(
    cfg: Any,
    *,
    benchmarks: Mapping[str, Any],
    conditions: Mapping[str, Any],
    external_inputs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Freeze a new campaign or verify an exact, honest resume.

    Creation is refused when results predate the charter.  A retroactive
    charter would document the current code, not necessarily the code that
    produced those results, and would therefore provide false assurance.
    """
    out_dir = Path(cfg.out_dir)
    path = out_dir / CHARTER_FILENAME
    design = build_campaign_design(
        cfg,
        benchmarks=benchmarks,
        conditions=conditions,
        external_inputs=external_inputs,
    )
    if not path.exists():
        if _has_prior_results(out_dir):
            raise UncharteredCampaignError(
                f"{out_dir} already contains eval results but has no "
                f"{CHARTER_FILENAME}; choose a fresh --out directory. "
                "Refusing to retroactively pre-register an experiment."
            )
        doc = _document_with_hashes(design)
        write_json_atomic(path, doc)
        # Verify the bytes that actually landed on disk, not just memory.
        return load_and_verify_charter(path)

    stored = load_and_verify_charter(path)
    current_hash = _sha256(design)
    if stored["design_sha256"] != current_hash:
        diffs = _differences(stored["design"], design)
        preview = "\n  - ".join(diffs[:8])
        remainder = len(diffs) - min(8, len(diffs))
        suffix = f"\n  - ... and {remainder} more" if remainder else ""
        raise CharterMismatchError(
            "requested campaign does not match its frozen charter:\n  - "
            f"{preview}{suffix}\nUse a new --out directory for a new design."
        )
    return stored


def verify_result_lineage(
    results: Sequence[Mapping[str, Any]], charter: Mapping[str, Any],
) -> None:
    """Require every result to belong to exactly this frozen campaign."""
    expected = charter.get("design_sha256")
    bad: list[str] = []
    for result in results:
        if result.get("charter_design_sha256") != expected:
            identity = "/".join(str(result.get(k, "?")) for k in (
                "benchmark", "condition", "seed",
            ))
            bad.append(identity)
    if bad:
        raise CharterIntegrityError(
            f"{len(bad)} result(s) are missing or have the wrong campaign "
            "charter hash: " + ", ".join(bad[:8])
        )
