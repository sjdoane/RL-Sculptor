"""Charter-aware distributed campaign coordination (CPU-only stubs)."""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import shutil
from pathlib import Path

import pytest

from sculptor.eval import CampaignConfig
from sculptor.eval.sharding import (
    DuplicateShardResultError,
    ShardDesignMismatchError,
    ShardIntegrityError,
    _verify_charter_reference,
    merge_sharded_campaign,
    prepare_sharded_campaign,
    run_campaign_shard,
)


_STUB = "tests.test_eval_harness._EvalStubAdapter"


def _cfg(root: Path, **overrides) -> CampaignConfig:
    values = {
        "name": "sharded-smoke",
        "out_dir": root,
        "benchmarks": ["cartpole_balance"],
        "conditions": ["plain_ppo"],
        "seeds": [1000, 1017],
        "iterations": 1,
        "steps_per_iter": 2,
        "rollout_episodes": 1,
        "adapter_class_override": _STUB,
    }
    values.update(overrides)
    return CampaignConfig(**values)


def _shard_dirs(root: Path) -> list[Path]:
    return sorted((root / "shards").glob("shard-*"))


def _run_worker(manifest: str) -> None:
    run_campaign_shard(Path(manifest))


def _reseal(path: Path, hash_field: str) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.pop(hash_field)
    raw = json.dumps(
        doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")
    doc[hash_field] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_merge_rejects_shard_from_another_charter(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_sharded_campaign(_cfg(first), shard_count=1)
    prepare_sharded_campaign(_cfg(second, seeds=[1000]), shard_count=1)

    with pytest.raises(ShardDesignMismatchError, match="belongs to charter"):
        merge_sharded_campaign(first, [_shard_dirs(second)[0]])


def test_merge_rejects_duplicate_job_from_different_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "duplicate"
    prepare_sharded_campaign(_cfg(root), shard_count=2)
    shard_a, shard_b = _shard_dirs(root)
    run_campaign_shard(shard_a / "shard_manifest.json")
    run_campaign_shard(shard_b / "shard_manifest.json")

    result_a = next(shard_a.glob("*/*/seed_*/result.json"))
    duplicate_dir = (
        shard_b / result_a.parents[2].name / result_a.parents[1].name
        / result_a.parent.name
    )
    shutil.copytree(result_a.parent, duplicate_dir)
    # Forge shard B's attestation of the copied job (the report seal is
    # tamper-evidence, not a secret) so the merge reaches the cross-shard
    # duplicate check rather than stopping at the unattested-result gate.
    report_a = json.loads(
        (shard_a / "shard_report.json").read_text(encoding="utf-8")
    )
    report_b_path = shard_b / "shard_report.json"
    report_b = json.loads(report_b_path.read_text(encoding="utf-8"))
    report_b["jobs"].extend(report_a["jobs"])
    report_b_path.write_text(json.dumps(report_b), encoding="utf-8")
    _reseal(report_b_path, "shard_report_sha256")

    with pytest.raises(DuplicateShardResultError, match="appears in both"):
        merge_sharded_campaign(root, [shard_a, shard_b])


def test_merge_reports_missing_shard_as_incomplete_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial"
    prepare_sharded_campaign(_cfg(root), shard_count=2)
    first_shard = _shard_dirs(root)[0]
    run_campaign_shard(first_shard / "shard_manifest.json")

    report = merge_sharded_campaign(root, [first_shard])

    assert report["status"] == "incomplete_coverage"
    assert report["coverage"] == {
        "complete": False,
        "n_expected": 2,
        "n_completed": 1,
        "n_missing": 1,
        "missing_jobs": [{
            "benchmark": "cartpole_balance",
            "condition": "plain_ppo",
            "seed": 1017,
        }],
    }
    assert report["sharding"]["missing_shards"] == ["shard-001"]
    on_disk = json.loads(
        (root / "campaign_report.json").read_text(encoding="utf-8")
    )
    assert on_disk["coverage"]["complete"] is False
    assert "Coverage INCOMPLETE" in (root / "report.html").read_text(
        encoding="utf-8"
    )


def test_merge_rejects_shard_with_altered_kg_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_kg = tmp_path / "shared" / "kg.db"
    monkeypatch.setenv("RS_KG_PATH", str(shared_kg))
    root = tmp_path / "kg"
    prepare_sharded_campaign(
        _cfg(root, conditions=["full"], seeds=[1000]), shard_count=1,
    )
    shard = _shard_dirs(root)[0]
    with (shard / "campaign_inputs" / "kg_base.db").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ShardIntegrityError, match="KG base hash"):
        merge_sharded_campaign(root, [shard])


def test_merge_rejects_worker_runtime_hash_drift(tmp_path: Path) -> None:
    root = tmp_path / "runtime-drift"
    prepare_sharded_campaign(
        _cfg(root, seeds=[1000]), shard_count=1,
    )
    shard = _shard_dirs(root)[0]
    run_campaign_shard(shard / "shard_manifest.json")
    identity_path = shard / "shard_run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["runtime_identity"]["source_tree_sha256"] = "0" * 64
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    _reseal(identity_path, "run_identity_sha256")

    with pytest.raises(ShardDesignMismatchError, match="runtime_identity"):
        merge_sharded_campaign(root, [shard])


def test_cpu_stub_matrix_runs_in_process_shards_and_merges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "multiprocess"
    prepare_sharded_campaign(
        _cfg(root, conditions=["plain_ppo", "seed_only"]), shard_count=2,
    )
    shards = _shard_dirs(root)
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_run_worker,
            args=(str(shard / "shard_manifest.json"),),
        )
        for shard in shards
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    report = merge_sharded_campaign(root)

    assert report["status"] == "complete"
    assert report["coverage"] == {
        "complete": True,
        "n_expected": 4,
        "n_completed": 4,
        "n_missing": 0,
        "missing_jobs": [],
    }
    assert len(report["jobs"]) == 4
    assert set(report["aggregates"]["benchmarks"]["cartpole_balance"]) == {
        "plain_ppo", "seed_only",
    }
    for condition in ("plain_ppo", "seed_only"):
        for seed in (1000, 1017):
            assert (
                root / "cartpole_balance" / condition / f"seed_{seed}"
                / "result.json"
            ).is_file()
    repeated = merge_sharded_campaign(root)
    assert repeated["coverage"]["complete"] is True


def test_charter_reference_verification_is_pinned() -> None:
    """Direct pin on `_verify_charter_reference`.

    This layer is redundant with the byte-level charter checks on every
    end-to-end path, so no integration test fails if it is silently
    removed (an adversarial mutation proved exactly that). Redundant
    defenses stay only if they are pinned.
    """
    charter = {"design_sha256": "a" * 64, "document_sha256": "b" * 64}
    good = {"charter": dict(charter)}
    _verify_charter_reference(good, charter, context="pin")

    for field in ("design_sha256", "document_sha256"):
        bad = {"charter": {**charter, field: "f" * 64}}
        with pytest.raises(ShardDesignMismatchError):
            _verify_charter_reference(bad, charter, context="pin")
    with pytest.raises(ShardIntegrityError, match="no charter reference"):
        _verify_charter_reference({}, charter, context="pin")


def test_merge_rejects_result_rewritten_after_shard_seal(
    tmp_path: Path,
) -> None:
    """A fabricated metric injected into result.json after the worker sealed
    its shard report must not merge: merge trusts only attested content."""
    root = tmp_path / "fabricated"
    prepare_sharded_campaign(_cfg(root, seeds=[1000]), shard_count=1)
    shard = _shard_dirs(root)[0]
    run_campaign_shard(shard / "shard_manifest.json")

    result_path = next(shard.glob("*/*/seed_*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["final_spec_score"] = 999.0
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ShardIntegrityError, match="sealed shard report"):
        merge_sharded_campaign(root, [shard])


def test_merge_rejects_results_without_sealed_shard_report(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unattested"
    prepare_sharded_campaign(_cfg(root, seeds=[1000]), shard_count=1)
    shard = _shard_dirs(root)[0]
    run_campaign_shard(shard / "shard_manifest.json")
    (shard / "shard_report.json").unlink()

    with pytest.raises(ShardIntegrityError, match="no sealed shard report"):
        merge_sharded_campaign(root, [shard])


def test_shard_resume_reuses_exact_matching_results(tmp_path: Path) -> None:
    from tests.test_eval_harness import _EvalStubAdapter

    _EvalStubAdapter.train_calls = []
    root = tmp_path / "resume"
    prepare_sharded_campaign(
        _cfg(root, seeds=[1000]), shard_count=1,
    )
    manifest = _shard_dirs(root)[0] / "shard_manifest.json"
    first = run_campaign_shard(manifest)
    calls = len(_EvalStubAdapter.train_calls)
    second = run_campaign_shard(manifest)

    assert calls == 1
    assert len(_EvalStubAdapter.train_calls) == calls
    assert first["coverage"]["complete"] is True
    assert second["jobs"][0]["cached"] is True
