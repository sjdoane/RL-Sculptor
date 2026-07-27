"""Charter-aware coordination for one distributed campaign matrix.

The scientific design is always the full benchmark x condition x seed
matrix in one campaign charter.  Shard manifests are operational job
assignments under that design; they never create subset charters.  Workers
verify the complete design and runtime identity before doing work, and the
merger verifies every contributing worker before accepting any result.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from sculptor.eval.benchmarks import BenchmarkTask, benchmark_registry
from sculptor.eval.charter import (
    CHARTER_FILENAME,
    CharterError,
    freeze_or_verify_campaign_charter,
    load_and_verify_charter,
    verify_result_lineage,
)
from sculptor.eval.harness import (
    CONDITIONS,
    CampaignConfig,
    _file_sha256,
    _prepare_campaign_kg_snapshot,
    _report_html,
    _run_job,
    aggregate,
)
from sculptor.run_context import capture_run_context, write_json_atomic


COORDINATOR_SCHEMA_VERSION = 1
SHARD_MANIFEST_SCHEMA_VERSION = 1
SHARD_RUN_SCHEMA_VERSION = 1
SHARD_REPORT_SCHEMA_VERSION = 1
MERGE_SCHEMA_VERSION = 1

COORDINATOR_FILENAME = "campaign_shards.json"
SHARD_MANIFEST_FILENAME = "shard_manifest.json"
SHARD_RUN_FILENAME = "shard_run_identity.json"
SHARD_REPORT_FILENAME = "shard_report.json"
MERGE_FILENAME = "campaign_merge.json"


class ShardError(RuntimeError):
    """Base class for distributed-campaign failures."""


class ShardIntegrityError(ShardError):
    """A coordinator, shard, input, or result artifact is inconsistent."""


class ShardDesignMismatchError(ShardError):
    """A shard belongs to a different frozen campaign design."""


class DuplicateShardResultError(ShardIntegrityError):
    """More than one shard claims the same atomic campaign job."""


JobKey = tuple[str, str, int]


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


def _seal(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    doc = dict(payload)
    if hash_field in doc:
        raise ValueError(f"payload already contains {hash_field!r}")
    doc[hash_field] = _sha256(doc)
    return doc


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ShardIntegrityError(f"{description} is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardIntegrityError(
            f"{description} is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ShardIntegrityError(f"{description} must be a JSON object: {path}")
    return value


def _load_sealed(
    path: Path,
    *,
    description: str,
    schema_version: int,
    hash_field: str,
) -> dict[str, Any]:
    doc = _load_json_object(path, description)
    if doc.get("schema_version") != schema_version:
        raise ShardIntegrityError(
            f"unsupported {description} schema {doc.get('schema_version')!r}; "
            f"expected {schema_version}"
        )
    stored = doc.get(hash_field)
    unhashed = {key: value for key, value in doc.items() if key != hash_field}
    if not isinstance(stored, str) or stored != _sha256(unhashed):
        raise ShardIntegrityError(
            f"{description} hash mismatch; artifact was altered or "
            f"incompletely written: {path}"
        )
    return doc


def _write_or_verify_sealed(
    path: Path,
    payload: Mapping[str, Any],
    *,
    description: str,
    schema_version: int,
    hash_field: str,
) -> dict[str, Any]:
    expected = _seal(payload, hash_field)
    if Path(path).exists():
        stored = _load_sealed(
            path,
            description=description,
            schema_version=schema_version,
            hash_field=hash_field,
        )
        if _canonical_bytes(stored) != _canonical_bytes(expected):
            raise ShardIntegrityError(
                f"existing {description} does not exactly match the frozen "
                f"coordinator plan: {path}"
            )
        return stored
    write_json_atomic(path, expected)
    return _load_sealed(
        path,
        description=description,
        schema_version=schema_version,
        hash_field=hash_field,
    )


def _job_payload(key: JobKey) -> dict[str, Any]:
    return {"benchmark": key[0], "condition": key[1], "seed": key[2]}


def _job_key(value: Mapping[str, Any], *, context: str) -> JobKey:
    if set(value) != {"benchmark", "condition", "seed"}:
        raise ShardIntegrityError(
            f"{context} must contain exactly benchmark, condition, and seed"
        )
    benchmark = value.get("benchmark")
    condition = value.get("condition")
    seed = value.get("seed")
    if not isinstance(benchmark, str) or not benchmark:
        raise ShardIntegrityError(f"{context}.benchmark must be non-empty text")
    if not isinstance(condition, str) or not condition:
        raise ShardIntegrityError(f"{context}.condition must be non-empty text")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ShardIntegrityError(f"{context}.seed must be an integer")
    return benchmark, condition, seed


def _matrix(
    benchmarks: Sequence[str], conditions: Sequence[str], seeds: Sequence[int],
) -> list[JobKey]:
    return [
        (benchmark, condition, seed)
        for benchmark in benchmarks
        for condition in conditions
        for seed in seeds
    ]


def _validate_partition(partition: Mapping[str, Any]) -> list[JobKey]:
    if partition.get("method") != "round_robin_by_global_job_index":
        raise ShardIntegrityError("unsupported or missing shard partition method")
    matrix = partition.get("matrix")
    assignments = partition.get("assignments")
    if not isinstance(matrix, dict) or not isinstance(assignments, dict):
        raise ShardIntegrityError("partition needs matrix and assignments objects")
    benchmarks = matrix.get("benchmarks")
    conditions = matrix.get("conditions")
    seeds = matrix.get("seeds")
    jobs = matrix.get("jobs")
    if not isinstance(benchmarks, list) or not all(
        isinstance(value, str) for value in benchmarks
    ):
        raise ShardIntegrityError("partition matrix benchmarks are malformed")
    if not isinstance(conditions, list) or not all(
        isinstance(value, str) for value in conditions
    ):
        raise ShardIntegrityError("partition matrix conditions are malformed")
    if not isinstance(seeds, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in seeds
    ):
        raise ShardIntegrityError("partition matrix seeds are malformed")
    if not isinstance(jobs, list):
        raise ShardIntegrityError("partition matrix jobs must be a list")
    expected = _matrix(benchmarks, conditions, seeds)
    declared = [
        _job_key(value, context=f"partition.matrix.jobs[{index}]")
        for index, value in enumerate(jobs)
        if isinstance(value, dict)
    ]
    if len(declared) != len(jobs) or declared != expected:
        raise ShardIntegrityError(
            "partition matrix jobs are not the exact ordered Cartesian product"
        )
    assigned: list[JobKey] = []
    if not assignments:
        raise ShardIntegrityError("partition needs at least one shard assignment")
    for shard_id, shard_jobs in assignments.items():
        if not isinstance(shard_id, str) or not shard_id:
            raise ShardIntegrityError("partition shard ids must be non-empty text")
        if not isinstance(shard_jobs, list):
            raise ShardIntegrityError(f"assignment {shard_id!r} must be a list")
        assigned.extend(
            _job_key(value, context=f"partition.assignments.{shard_id}[{index}]")
            for index, value in enumerate(shard_jobs)
            if isinstance(value, dict)
        )
        if any(not isinstance(value, dict) for value in shard_jobs):
            raise ShardIntegrityError(
                f"assignment {shard_id!r} contains a non-object job"
            )
    if len(assigned) != len(expected) or set(assigned) != set(expected):
        raise ShardIntegrityError(
            "shard assignments must cover every global-matrix job exactly once"
        )
    if len(set(assigned)) != len(assigned):
        raise ShardIntegrityError("shard assignments contain duplicate jobs")
    return expected


def _runtime_references(charter: Mapping[str, Any]) -> dict[str, Any]:
    try:
        runtime = charter["design"]["runtime_identity"]
        dependency_hash = runtime["dependency_identity_sha256"]
    except (KeyError, TypeError):
        raise ShardIntegrityError(
            "campaign charter has no distributed runtime/dependency identity"
        ) from None
    if not isinstance(runtime, dict) or not isinstance(dependency_hash, str):
        raise ShardIntegrityError("campaign runtime identity is malformed")
    return {
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": _sha256(runtime),
        "dependency_identity_sha256": dependency_hash,
    }


def _verify_runtime_references(
    artifact: Mapping[str, Any], charter: Mapping[str, Any], *, context: str,
) -> None:
    expected = _runtime_references(charter)
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ShardDesignMismatchError(
                f"{context} {key} does not exactly match the campaign design"
            )


def _config_snapshot(cfg: CampaignConfig) -> dict[str, Any]:
    snapshot = asdict(cfg)
    snapshot.pop("out_dir", None)
    snapshot["benchmark_manifests"] = [
        str(path) for path in cfg.benchmark_manifests
    ]
    return snapshot


def _config_from_charter(
    charter: Mapping[str, Any], out_dir: Path,
) -> CampaignConfig:
    raw = charter.get("design", {}).get("campaign")
    if not isinstance(raw, dict):
        raise ShardIntegrityError("charter campaign config is malformed")
    expected_fields = {field.name for field in fields(CampaignConfig)} - {"out_dir"}
    if set(raw) != expected_fields:
        raise ShardIntegrityError(
            "charter campaign config fields do not match this runtime: "
            f"expected {sorted(expected_fields)}, got {sorted(raw)}"
        )
    values = dict(raw)
    manifests = values.get("benchmark_manifests")
    if not isinstance(manifests, list) or not all(
        isinstance(path, str) for path in manifests
    ):
        raise ShardIntegrityError("charter benchmark_manifests are malformed")
    values["benchmark_manifests"] = [Path(path) for path in manifests]
    return CampaignConfig(out_dir=Path(out_dir), **values)


def _registry_from_charter(
    charter: Mapping[str, Any],
) -> dict[str, BenchmarkTask]:
    raw_benchmarks = charter.get("design", {}).get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        raise ShardIntegrityError("charter benchmark definitions are malformed")
    expected_fields = {field.name for field in fields(BenchmarkTask)}
    registry: dict[str, BenchmarkTask] = {}
    for index, raw in enumerate(raw_benchmarks):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ShardIntegrityError(
                f"charter benchmark definition {index} has unexpected fields"
            )
        values = dict(raw)
        values["required_capabilities"] = tuple(values["required_capabilities"])
        values["known_limitations"] = tuple(values["known_limitations"])
        benchmark = BenchmarkTask(**values)
        if benchmark.name in registry:
            raise ShardIntegrityError(
                f"charter contains duplicate benchmark {benchmark.name!r}"
            )
        registry[benchmark.name] = benchmark
    return registry


def _campaign_external_inputs(
    cfg: CampaignConfig, kg_input: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    external_inputs: dict[str, Any] = {}
    if kg_input is not None:
        external_inputs["knowledge_graph"] = dict(kg_input)
    if cfg.benchmark_manifests:
        external_inputs["benchmark_manifests"] = [
            {
                "sha256": _file_sha256(Path(path).expanduser().resolve()),
                "size_bytes": Path(path).expanduser().resolve().stat().st_size,
            }
            for path in cfg.benchmark_manifests
        ]
    return external_inputs


def _copy_exact(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    expected_hash = _file_sha256(source)
    expected_size = source.stat().st_size
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != expected_size
            or _file_sha256(destination) != expected_hash
        ):
            raise ShardIntegrityError(
                f"existing replicated input differs from frozen source: "
                f"{destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp)
        if tmp.stat().st_size != expected_size or _file_sha256(tmp) != expected_hash:
            raise ShardIntegrityError(
                f"replicated input failed post-copy verification: {destination}"
            )
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def _relative_input(path: Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def _stage_benchmark_inputs(
    cfg: CampaignConfig, root: Path, external_inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = external_inputs.get("benchmark_manifests", [])
    if len(expected) != len(cfg.benchmark_manifests):
        raise ShardIntegrityError("benchmark-manifest input count drifted")
    records: list[dict[str, Any]] = []
    for index, source_value in enumerate(cfg.benchmark_manifests):
        source = Path(source_value).expanduser().resolve()
        destination = (
            root / "campaign_inputs" / "benchmark_manifests"
            / f"{index:03d}_{source.name}"
        )
        _copy_exact(source, destination)
        record = {
            "relative_path": _relative_input(destination, root),
            "sha256": _file_sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
        if record["sha256"] != expected[index].get("sha256") or (
            record["size_bytes"] != expected[index].get("size_bytes")
        ):
            raise ShardIntegrityError(
                f"staged benchmark manifest {index} does not match charter input"
            )
        records.append(record)
    return records


def _input_records(
    root: Path,
    *,
    kg_snapshot: Optional[Path],
    benchmark_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kg_record: Optional[dict[str, Any]] = None
    if kg_snapshot is not None:
        record_path = kg_snapshot.with_name("kg_base.json")
        kg_record = {
            "db_relative_path": _relative_input(kg_snapshot, root),
            "db_sha256": _file_sha256(kg_snapshot),
            "db_size_bytes": kg_snapshot.stat().st_size,
            "record_relative_path": _relative_input(record_path, root),
            "record_sha256": _file_sha256(record_path),
        }
    return {
        "kg_base": kg_record,
        "benchmark_manifests": [dict(record) for record in benchmark_inputs],
    }


def _replicate_inputs(
    source_root: Path,
    destination_root: Path,
    records: Mapping[str, Any],
) -> None:
    kg = records.get("kg_base")
    if isinstance(kg, dict):
        for field_name in ("db_relative_path", "record_relative_path"):
            relative = Path(str(kg[field_name]))
            _copy_exact(source_root / relative, destination_root / relative)
    benchmark_records = records.get("benchmark_manifests", [])
    if not isinstance(benchmark_records, list):
        raise ShardIntegrityError("benchmark input records are malformed")
    for record in benchmark_records:
        relative = Path(str(record["relative_path"]))
        _copy_exact(source_root / relative, destination_root / relative)


def _resolve_input(root: Path, relative_value: Any) -> Path:
    if not isinstance(relative_value, str):
        raise ShardIntegrityError("input relative path must be text")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ShardIntegrityError(f"unsafe input relative path: {relative_value!r}")
    root_resolved = Path(root).resolve()
    resolved = (root_resolved / relative).resolve()
    if root_resolved not in resolved.parents:
        raise ShardIntegrityError(f"input path escapes shard root: {relative_value!r}")
    return resolved


def _verify_input_records(
    root: Path, records: Mapping[str, Any], charter: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_external = charter.get("design", {}).get("external_inputs")
    if not isinstance(frozen_external, dict):
        raise ShardIntegrityError("charter external inputs are malformed")
    actual_external: dict[str, Any] = {}
    kg = records.get("kg_base")
    expected_kg = frozen_external.get("knowledge_graph")
    if expected_kg is None:
        if kg is not None:
            raise ShardIntegrityError("shard unexpectedly carries a KG base")
    else:
        if not isinstance(kg, dict):
            raise ShardIntegrityError("chartered KG base is missing from shard")
        db_path = _resolve_input(root, kg.get("db_relative_path"))
        record_path = _resolve_input(root, kg.get("record_relative_path"))
        if not db_path.is_file() or not record_path.is_file():
            raise ShardIntegrityError("shard KG base snapshot is incomplete")
        db_hash = _file_sha256(db_path)
        if (
            db_hash != kg.get("db_sha256")
            or db_hash != expected_kg.get("sha256")
            or db_path.stat().st_size != kg.get("db_size_bytes")
            or _file_sha256(record_path) != kg.get("record_sha256")
        ):
            raise ShardIntegrityError(
                "shard KG base hash does not match the chartered single source"
            )
        actual_external["knowledge_graph"] = {
            "kind": "sqlite_kg_snapshot",
            "sha256": db_hash,
            "size_bytes": db_path.stat().st_size,
        }

    benchmark_records = records.get("benchmark_manifests")
    expected_benchmarks = frozen_external.get("benchmark_manifests", [])
    if not isinstance(benchmark_records, list) or (
        len(benchmark_records) != len(expected_benchmarks)
    ):
        raise ShardIntegrityError("shard benchmark-manifest inputs are incomplete")
    if benchmark_records:
        actual_external["benchmark_manifests"] = []
    for index, record in enumerate(benchmark_records):
        if not isinstance(record, dict):
            raise ShardIntegrityError("benchmark input record must be an object")
        path = _resolve_input(root, record.get("relative_path"))
        if not path.is_file():
            raise ShardIntegrityError(f"shard benchmark input is missing: {path}")
        actual = {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if actual != expected_benchmarks[index] or (
            actual["sha256"] != record.get("sha256")
            or actual["size_bytes"] != record.get("size_bytes")
        ):
            raise ShardIntegrityError(
                f"shard benchmark input {index} does not match the charter"
            )
        actual_external["benchmark_manifests"].append(actual)
    if actual_external != frozen_external:
        raise ShardIntegrityError(
            "physical shard inputs do not exactly reproduce charter external inputs"
        )
    return actual_external


def _verify_charter_reference(
    artifact: Mapping[str, Any], charter: Mapping[str, Any], *, context: str,
) -> None:
    reference = artifact.get("charter")
    if not isinstance(reference, dict):
        raise ShardIntegrityError(f"{context} has no charter reference")
    if reference.get("design_sha256") != charter.get("design_sha256"):
        raise ShardDesignMismatchError(
            f"{context} charter design hash does not match the campaign"
        )
    if reference.get("document_sha256") != charter.get("document_sha256"):
        raise ShardDesignMismatchError(
            f"{context} charter document hash does not match the campaign"
        )


def _validate_manifest(
    manifest: Mapping[str, Any], charter: Mapping[str, Any], *, context: str,
) -> list[JobKey]:
    if manifest.get("status") != "assigned":
        raise ShardIntegrityError(f"{context} is not a frozen assignment")
    _verify_charter_reference(manifest, charter, context=context)
    _verify_runtime_references(manifest, charter, context=context)
    if manifest.get("external_inputs") != charter["design"]["external_inputs"]:
        raise ShardDesignMismatchError(
            f"{context} external inputs do not match the campaign design"
        )
    partition = manifest.get("partition")
    if not isinstance(partition, dict):
        raise ShardIntegrityError(f"{context} has no frozen partition")
    if manifest.get("partition_sha256") != _sha256(partition):
        raise ShardIntegrityError(f"{context} partition hash mismatch")
    expected = _validate_partition(partition)
    shard_id = manifest.get("shard_id")
    assignments = partition["assignments"]
    if not isinstance(shard_id, str) or shard_id not in assignments:
        raise ShardIntegrityError(f"{context} shard id is not in the partition")
    if manifest.get("assignment") != assignments[shard_id]:
        raise ShardIntegrityError(
            f"{context} assignment differs from the frozen partition"
        )
    return expected


def prepare_sharded_campaign(
    cfg: CampaignConfig, *, shard_count: int,
) -> dict[str, Any]:
    """Freeze one full design and deterministically assign all jobs to shards.

    The returned shard directories are self-contained transport units.  A
    byte-identical charter replica and frozen input replicas are copied into
    each directory, but the charter is created only once at the campaign root.
    """
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or (
        shard_count < 1
    ):
        raise ValueError("shard_count must be a positive integer")
    root = Path(cfg.out_dir)
    index_path = root / COORDINATOR_FILENAME
    if not index_path.exists() and (
        (root / "campaign_report.json").exists()
        or any(root.glob("*/*/seed_*/result.json"))
    ):
        raise ShardIntegrityError(
            "cannot retroactively partition a campaign root that already has "
            "results; choose a fresh output directory"
        )

    registry = benchmark_registry(cfg.benchmark_manifests)
    cfg.validate(registry)
    root.mkdir(parents=True, exist_ok=True)
    kg_snapshot, kg_input = _prepare_campaign_kg_snapshot(cfg)
    external_inputs = _campaign_external_inputs(cfg, kg_input)
    charter = freeze_or_verify_campaign_charter(
        cfg,
        benchmarks=registry,
        conditions=CONDITIONS,
        external_inputs=external_inputs,
    )
    charter_path = root / CHARTER_FILENAME
    charter_file_hash = _file_sha256(charter_path)
    runtime = _runtime_references(charter)
    benchmark_inputs = _stage_benchmark_inputs(cfg, root, external_inputs)
    inputs = _input_records(
        root,
        kg_snapshot=kg_snapshot,
        benchmark_inputs=benchmark_inputs,
    )

    matrix = _matrix(cfg.benchmarks, cfg.conditions, cfg.seeds)
    shard_ids = [f"shard-{index:03d}" for index in range(shard_count)]
    assignments: dict[str, list[dict[str, Any]]] = {
        shard_id: [] for shard_id in shard_ids
    }
    for index, key in enumerate(matrix):
        assignments[shard_ids[index % shard_count]].append(_job_payload(key))
    partition = {
        "method": "round_robin_by_global_job_index",
        "matrix": {
            "benchmarks": list(cfg.benchmarks),
            "conditions": list(cfg.conditions),
            "seeds": list(cfg.seeds),
            "jobs": [_job_payload(key) for key in matrix],
        },
        "assignments": assignments,
    }
    _validate_partition(partition)
    partition_hash = _sha256(partition)
    charter_reference = {
        "design_sha256": charter["design_sha256"],
        "document_sha256": charter["document_sha256"],
        "file_sha256": charter_file_hash,
    }
    manifest_records: list[dict[str, Any]] = []
    for shard_id in shard_ids:
        shard_dir = root / "shards" / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        _copy_exact(charter_path, shard_dir / CHARTER_FILENAME)
        _replicate_inputs(root, shard_dir, inputs)
        manifest_payload = {
            "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
            "status": "assigned",
            "created_at": charter["created_at"],
            "shard_id": shard_id,
            "charter": charter_reference,
            **runtime,
            "campaign_config": _config_snapshot(cfg),
            "external_inputs": external_inputs,
            "input_records": inputs,
            "partition": partition,
            "partition_sha256": partition_hash,
            "assignment": assignments[shard_id],
        }
        manifest = _write_or_verify_sealed(
            shard_dir / SHARD_MANIFEST_FILENAME,
            manifest_payload,
            description=f"shard manifest {shard_id}",
            schema_version=SHARD_MANIFEST_SCHEMA_VERSION,
            hash_field="manifest_sha256",
        )
        manifest_records.append({
            "shard_id": shard_id,
            "relative_dir": shard_dir.relative_to(root).as_posix(),
            "manifest_sha256": manifest["manifest_sha256"],
            "n_jobs": len(assignments[shard_id]),
        })

    coordinator_payload = {
        "schema_version": COORDINATOR_SCHEMA_VERSION,
        "status": "frozen",
        "created_at": charter["created_at"],
        "charter": charter_reference,
        **runtime,
        "external_inputs": external_inputs,
        "input_records": inputs,
        "partition": partition,
        "partition_sha256": partition_hash,
        "shards": manifest_records,
    }
    coordinator = _write_or_verify_sealed(
        index_path,
        coordinator_payload,
        description="campaign shard coordinator",
        schema_version=COORDINATOR_SCHEMA_VERSION,
        hash_field="coordinator_sha256",
    )
    return coordinator


def _load_manifest(path: Path) -> dict[str, Any]:
    return _load_sealed(
        path,
        description="shard manifest",
        schema_version=SHARD_MANIFEST_SCHEMA_VERSION,
        hash_field="manifest_sha256",
    )


def _verify_replica_charter(
    shard_root: Path, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    charter_path = Path(shard_root) / CHARTER_FILENAME
    charter = load_and_verify_charter(charter_path)
    _verify_charter_reference(manifest, charter, context="shard manifest")
    reference = manifest["charter"]
    if _file_sha256(charter_path) != reference.get("file_sha256"):
        raise ShardIntegrityError(
            "shard charter replica is not byte-identical to the frozen charter"
        )
    return charter


def _verify_worker_design(
    shard_root: Path,
    manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
) -> tuple[
    CampaignConfig,
    dict[str, BenchmarkTask],
    Optional[Path],
    Optional[dict[str, Any]],
]:
    cfg = _config_from_charter(charter, shard_root)
    if _canonical_bytes(_config_snapshot(cfg)) != _canonical_bytes(
        manifest.get("campaign_config")
    ):
        raise ShardDesignMismatchError(
            "shard campaign config does not match its charter"
        )
    registry = _registry_from_charter(charter)
    cfg.validate(registry)
    records = manifest.get("input_records")
    if not isinstance(records, dict):
        raise ShardIntegrityError("shard manifest input records are malformed")
    physical_external = _verify_input_records(shard_root, records, charter)
    kg_snapshot, kg_input = _prepare_campaign_kg_snapshot(cfg)
    if physical_external.get("knowledge_graph") != kg_input:
        raise ShardIntegrityError(
            "worker KG verification disagrees with the chartered KG input"
        )
    verified = freeze_or_verify_campaign_charter(
        cfg,
        benchmarks=registry,
        conditions=CONDITIONS,
        external_inputs=physical_external,
    )
    if verified["design_sha256"] != charter["design_sha256"]:
        raise ShardDesignMismatchError("worker reconstructed a different design")
    _verify_runtime_references(manifest, verified, context="shard manifest")
    return cfg, registry, kg_snapshot, kg_input


def _initialize_shard_run(
    shard_root: Path,
    manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SHARD_RUN_SCHEMA_VERSION,
        "status": "runtime_verified",
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "shard_id": manifest["shard_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "charter": dict(manifest["charter"]),
        **_runtime_references(charter),
        "partition_sha256": manifest["partition_sha256"],
    }
    path = Path(shard_root) / SHARD_RUN_FILENAME
    if path.exists():
        stored = _load_sealed(
            path,
            description="shard run identity",
            schema_version=SHARD_RUN_SCHEMA_VERSION,
            hash_field="run_identity_sha256",
        )
        for key, value in payload.items():
            if key != "initialized_at" and stored.get(key) != value:
                raise ShardDesignMismatchError(
                    f"shard resume identity changed at {key!r}"
                )
        _verify_runtime_references(stored, charter, context="shard run identity")
        return stored
    return _write_or_verify_sealed(
        path,
        payload,
        description="shard run identity",
        schema_version=SHARD_RUN_SCHEMA_VERSION,
        hash_field="run_identity_sha256",
    )


def _validate_result_identity(result: Mapping[str, Any], expected: JobKey) -> None:
    actual = (
        result.get("benchmark"), result.get("condition"), result.get("seed")
    )
    if actual != expected:
        raise ShardIntegrityError(
            f"job result identity {actual!r} does not match assignment {expected!r}"
        )


def _coverage(expected: Sequence[JobKey], completed: Iterable[JobKey]) -> dict[str, Any]:
    completed_set = set(completed)
    missing = [key for key in expected if key not in completed_set]
    return {
        "complete": not missing,
        "n_expected": len(expected),
        "n_completed": len(expected) - len(missing),
        "n_missing": len(missing),
        "missing_jobs": [_job_payload(key) for key in missing],
    }


def run_campaign_shard(
    manifest_path: Path,
    *,
    job_hook: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run one assigned shard after exact full-design/runtime verification."""
    manifest_path = Path(manifest_path).resolve()
    shard_root = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    charter = _verify_replica_charter(shard_root, manifest)
    _validate_manifest(manifest, charter, context="shard manifest")
    cfg, registry, kg_snapshot, kg_input = _verify_worker_design(
        shard_root, manifest, charter,
    )
    run_identity = _initialize_shard_run(shard_root, manifest, charter)
    assigned_raw = manifest.get("assignment")
    if not isinstance(assigned_raw, list):
        raise ShardIntegrityError("shard assignment must be a list")
    assigned = [
        _job_key(value, context=f"assignment[{index}]")
        for index, value in enumerate(assigned_raw)
        if isinstance(value, dict)
    ]
    partition_assignment = manifest["partition"]["assignments"][
        manifest["shard_id"]
    ]
    if assigned_raw != partition_assignment:
        raise ShardIntegrityError(
            "shard assignment was redefined outside the frozen partition"
        )

    results: list[dict[str, Any]] = []
    charter_hash = str(charter["design_sha256"])
    for key in assigned:
        benchmark, condition, seed = key
        result = _run_job(
            cfg,
            registry[benchmark],
            CONDITIONS[condition],
            seed,
            charter_hash,
            kg_snapshot,
            kg_input["sha256"] if kg_input is not None else None,
        )
        _validate_result_identity(result, key)
        results.append(result)
        if job_hook is not None:
            job_hook(result)

    verify_result_lineage(results, charter)
    coverage = _coverage(assigned, [
        (result["benchmark"], result["condition"], result["seed"])
        for result in results
    ])
    report_payload = {
        "schema_version": SHARD_REPORT_SCHEMA_VERSION,
        "status": "complete" if coverage["complete"] else "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard_id": manifest["shard_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "run_identity_sha256": run_identity["run_identity_sha256"],
        "charter": dict(manifest["charter"]),
        **_runtime_references(charter),
        "coverage": coverage,
        "aggregates": aggregate(results),
        "jobs": [
            {key: value for key, value in result.items() if key != "spec_series"}
            for result in results
        ],
    }
    report = _seal(report_payload, "shard_report_sha256")
    write_json_atomic(shard_root / SHARD_REPORT_FILENAME, report)
    return report


def _load_coordinator(root: Path) -> dict[str, Any]:
    return _load_sealed(
        Path(root) / COORDINATOR_FILENAME,
        description="campaign shard coordinator",
        schema_version=COORDINATOR_SCHEMA_VERSION,
        hash_field="coordinator_sha256",
    )


def _validate_coordinator(
    coordinator: Mapping[str, Any], charter: Mapping[str, Any],
) -> list[JobKey]:
    if coordinator.get("status") != "frozen":
        raise ShardIntegrityError("campaign shard coordinator is not frozen")
    _verify_charter_reference(coordinator, charter, context="coordinator")
    _verify_runtime_references(coordinator, charter, context="coordinator")
    if coordinator.get("external_inputs") != charter["design"]["external_inputs"]:
        raise ShardDesignMismatchError(
            "coordinator external inputs do not match the campaign design"
        )
    partition = coordinator.get("partition")
    if not isinstance(partition, dict) or (
        coordinator.get("partition_sha256") != _sha256(partition)
    ):
        raise ShardIntegrityError("coordinator partition hash mismatch")
    expected = _validate_partition(partition)
    shards = coordinator.get("shards")
    if not isinstance(shards, list):
        raise ShardIntegrityError("coordinator shards must be a list")
    declared_ids = [record.get("shard_id") for record in shards if isinstance(record, dict)]
    if len(declared_ids) != len(shards) or set(declared_ids) != set(
        partition["assignments"]
    ):
        raise ShardIntegrityError(
            "coordinator shard records do not match frozen assignments"
        )
    if len(set(declared_ids)) != len(declared_ids):
        raise ShardIntegrityError("coordinator contains duplicate shard ids")
    for record in shards:
        shard_id = record["shard_id"]
        if record.get("n_jobs") != len(partition["assignments"][shard_id]):
            raise ShardIntegrityError(
                f"coordinator job count drifted for shard {shard_id!r}"
            )
        if not isinstance(record.get("manifest_sha256"), str):
            raise ShardIntegrityError(
                f"coordinator manifest hash is missing for shard {shard_id!r}"
            )
        if not isinstance(record.get("relative_dir"), str):
            raise ShardIntegrityError(
                f"coordinator relative dir is missing for shard {shard_id!r}"
            )
    return expected


def _verify_merge_runtime(
    root: Path,
    coordinator: Mapping[str, Any],
    charter: Mapping[str, Any],
) -> None:
    records = coordinator.get("input_records")
    if not isinstance(records, dict):
        raise ShardIntegrityError("coordinator input records are malformed")
    physical_external = _verify_input_records(root, records, charter)
    cfg = _config_from_charter(charter, root)
    registry = _registry_from_charter(charter)
    cfg.validate(registry)
    _, kg_input = _prepare_campaign_kg_snapshot(cfg)
    if physical_external.get("knowledge_graph") != kg_input:
        raise ShardIntegrityError("campaign-root KG verification disagrees")
    freeze_or_verify_campaign_charter(
        cfg,
        benchmarks=registry,
        conditions=CONDITIONS,
        external_inputs=physical_external,
    )


def _validate_shard_run_identity(
    shard_root: Path,
    manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    path = Path(shard_root) / SHARD_RUN_FILENAME
    if not path.exists():
        return None
    identity = _load_sealed(
        path,
        description="shard run identity",
        schema_version=SHARD_RUN_SCHEMA_VERSION,
        hash_field="run_identity_sha256",
    )
    if identity.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ShardDesignMismatchError(
            "shard run identity references a different manifest"
        )
    _verify_charter_reference(identity, charter, context="shard run identity")
    _verify_runtime_references(identity, charter, context="shard run identity")
    return identity


def _result_path_key(path: Path) -> JobKey:
    seed_name = path.parent.name
    try:
        seed = int(seed_name.removeprefix("seed_"))
    except ValueError:
        raise ShardIntegrityError(f"invalid result seed directory: {path}") from None
    if not seed_name.startswith("seed_") or seed_name != f"seed_{seed}":
        raise ShardIntegrityError(f"non-canonical result seed directory: {path}")
    return path.parents[2].name, path.parents[1].name, seed


#: Runtime annotations excluded from result-vs-report attestation equality:
#: `spec_series` is stripped from sealed report job records by design, and
#: `cached` is added in memory on the resume path without rewriting the
#: result file. Neither is evidence; everything else must match exactly.
_ATTESTATION_EXCLUDED_FIELDS = frozenset({"spec_series", "cached"})


def _attested_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items()
        if key not in _ATTESTATION_EXCLUDED_FIELDS
    }


def _load_shard_attestations(
    shard_root: Path,
    manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> dict[JobKey, dict[str, Any]]:
    """Load the sealed shard report and index its attested job records.

    Merge must never trust bare ``result.json`` content: the worker seals a
    ``shard_report.json`` over every job outcome at run time, and a result
    file rewritten after that seal (e.g. a fabricated score) has no
    attestation. The report must exist, verify, and be bound to this exact
    manifest and run identity.
    """
    report_path = Path(shard_root) / SHARD_REPORT_FILENAME
    if not report_path.is_file():
        raise ShardIntegrityError(
            f"shard {manifest.get('shard_id')!r} has results but no sealed "
            "shard report attesting them; resume the shard to completion "
            "before merging"
        )
    report = _load_sealed(
        report_path,
        description="shard report",
        schema_version=SHARD_REPORT_SCHEMA_VERSION,
        hash_field="shard_report_sha256",
    )
    _verify_charter_reference(report, charter, context="shard report")
    _verify_runtime_references(report, charter, context="shard report")
    if report.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ShardIntegrityError(
            "shard report is bound to a different shard manifest"
        )
    if report.get("run_identity_sha256") != run_identity.get(
        "run_identity_sha256"
    ):
        raise ShardIntegrityError(
            "shard report is bound to a different shard run identity"
        )
    jobs = report.get("jobs")
    if not isinstance(jobs, list):
        raise ShardIntegrityError("shard report jobs are malformed")
    attested: dict[JobKey, dict[str, Any]] = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ShardIntegrityError(f"shard report job {index} is malformed")
        key = (job.get("benchmark"), job.get("condition"), job.get("seed"))
        if not (
            isinstance(key[0], str)
            and isinstance(key[1], str)
            and isinstance(key[2], int)
            and not isinstance(key[2], bool)
        ):
            raise ShardIntegrityError(
                f"shard report job {index} has a malformed job tuple"
            )
        if key in attested:
            raise ShardIntegrityError(
                f"shard report attests job {key!r} more than once"
            )
        attested[key] = _attested_view(job)
    return attested


def _scan_shard_results(
    shard_root: Path,
    manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    run_identity: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    paths = sorted(Path(shard_root).glob("*/*/seed_*/result.json"))
    if paths and run_identity is None:
        raise ShardIntegrityError(
            f"shard {manifest.get('shard_id')!r} has results but no verified "
            "runtime identity"
        )
    attested: dict[JobKey, dict[str, Any]] = {}
    if paths:
        attested = _load_shard_attestations(
            shard_root, manifest, charter, run_identity,
        )
    records: list[dict[str, Any]] = []
    for path in paths:
        result = _load_json_object(path, "shard result")
        path_key = _result_path_key(path)
        result_key = (
            result.get("benchmark"), result.get("condition"), result.get("seed")
        )
        if result_key != path_key:
            raise ShardIntegrityError(
                f"result tuple does not match its path: {path}"
            )
        sealed = attested.get(path_key)
        if sealed is None:
            raise ShardIntegrityError(
                f"result has no attestation in the sealed shard report: {path}"
            )
        if _attested_view(result) != sealed:
            raise ShardIntegrityError(
                f"result content does not match its sealed shard report "
                f"attestation: {path}"
            )
        try:
            verify_result_lineage([result], charter)
        except CharterError as exc:
            raise ShardDesignMismatchError(
                f"result has wrong charter lineage: {path}: {exc}"
            ) from exc
        records.append({
            "key": path_key,
            "path": path,
            "result": result,
            "result_sha256": _file_sha256(path),
            "source_shard": manifest["shard_id"],
            "shard_manifest_sha256": manifest["manifest_sha256"],
            "shard_run_identity_sha256": (
                run_identity["run_identity_sha256"] if run_identity else None
            ),
        })
    return records


def _load_prior_merge(
    root: Path,
    coordinator: Mapping[str, Any],
    charter: Mapping[str, Any],
) -> dict[JobKey, dict[str, Any]]:
    path = Path(root) / MERGE_FILENAME
    root_paths = sorted(Path(root).glob("*/*/seed_*/result.json"))
    if not path.exists():
        if root_paths:
            raise ShardIntegrityError(
                "campaign root has results with no verified merge provenance"
            )
        return {}
    merge = _load_sealed(
        path,
        description="campaign merge provenance",
        schema_version=MERGE_SCHEMA_VERSION,
        hash_field="merge_sha256",
    )
    _verify_charter_reference(merge, charter, context="campaign merge")
    _verify_runtime_references(merge, charter, context="campaign merge")
    if merge.get("coordinator_sha256") != coordinator.get("coordinator_sha256"):
        raise ShardDesignMismatchError(
            "prior merge belongs to a different coordinator plan"
        )
    entries = merge.get("results")
    if not isinstance(entries, list):
        raise ShardIntegrityError("campaign merge results must be a list")
    prior: dict[JobKey, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("job"), dict):
            raise ShardIntegrityError(f"campaign merge result {index} is malformed")
        key = _job_key(entry["job"], context=f"campaign_merge.results[{index}].job")
        if key in prior:
            raise DuplicateShardResultError(
                f"prior merge provenance duplicates job {key!r}"
            )
        result_path = (
            Path(root) / key[0] / key[1] / f"seed_{key[2]}" / "result.json"
        )
        if not result_path.is_file() or (
            _file_sha256(result_path) != entry.get("result_sha256")
        ):
            raise ShardIntegrityError(
                f"merged result no longer matches its provenance: {result_path}"
            )
        result = _load_json_object(result_path, "merged result")
        _validate_result_identity(result, key)
        try:
            verify_result_lineage([result], charter)
        except CharterError as exc:
            raise ShardDesignMismatchError(str(exc)) from exc
        prior[key] = {**entry, "result": result, "path": result_path}
    documented = {entry["path"].resolve() for entry in prior.values()}
    if documented != {path.resolve() for path in root_paths}:
        raise ShardIntegrityError(
            "campaign root result files do not exactly match merge provenance"
        )
    return prior


def _copy_job_tree(source_result: Path, destination_result: Path) -> None:
    destination_dir = destination_result.parent
    if destination_result.exists():
        if _file_sha256(destination_result) != _file_sha256(source_result):
            raise ShardIntegrityError(
                f"merged destination already has a different result: "
                f"{destination_result}"
            )
        return
    if destination_dir.exists():
        raise ShardIntegrityError(
            f"merged destination has an untracked partial job directory: "
            f"{destination_dir}"
        )
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=".campaign-merge-", dir=destination_dir.parent,
    ))
    staged = staging_root / destination_dir.name
    try:
        shutil.copytree(source_result.parent, staged)
        staged_result = staged / "result.json"
        if _file_sha256(staged_result) != _file_sha256(source_result):
            raise ShardIntegrityError(
                f"job tree failed post-copy verification: {source_result.parent}"
            )
        os.replace(staged, destination_dir)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _default_shard_dirs(
    root: Path, coordinator: Mapping[str, Any],
) -> list[Path]:
    directories: list[Path] = []
    for record in coordinator["shards"]:
        relative = record.get("relative_dir")
        if not isinstance(relative, str):
            raise ShardIntegrityError("coordinator shard relative_dir is malformed")
        directories.append(_resolve_input(root, relative))
    return directories


def merge_sharded_campaign(
    campaign_root: Path,
    shard_dirs: Optional[Sequence[Path]] = None,
) -> dict[str, Any]:
    """Verify and merge available shard results into one honest campaign.

    Missing shards or jobs are allowed and explicitly reported as incomplete
    coverage.  Any present result must have a verified shard runtime identity,
    exact charter lineage, unique tuple, and frozen assignment.
    """
    root = Path(campaign_root).resolve()
    charter = load_and_verify_charter(root / CHARTER_FILENAME)
    coordinator = _load_coordinator(root)
    expected = _validate_coordinator(coordinator, charter)
    if _file_sha256(root / CHARTER_FILENAME) != coordinator["charter"].get(
        "file_sha256"
    ):
        raise ShardIntegrityError(
            "campaign-root charter bytes differ from the coordinator source"
        )
    _verify_merge_runtime(root, coordinator, charter)
    prior = _load_prior_merge(root, coordinator, charter)
    expected_set = set(expected)
    expected_shards = {
        record["shard_id"]: record for record in coordinator["shards"]
    }
    directories = (
        _default_shard_dirs(root, coordinator)
        if shard_dirs is None
        else [Path(path).resolve() for path in shard_dirs]
    )
    if len(set(directories)) != len(directories):
        raise ShardIntegrityError("the same shard directory was supplied twice")

    incoming: dict[JobKey, dict[str, Any]] = {}
    provided_shards: set[str] = set()
    for shard_root in directories:
        manifest = _load_manifest(shard_root / SHARD_MANIFEST_FILENAME)
        shard_charter = _verify_replica_charter(shard_root, manifest)
        if shard_charter["design_sha256"] != charter["design_sha256"]:
            raise ShardDesignMismatchError(
                f"shard {manifest.get('shard_id')!r} belongs to charter "
                f"{shard_charter['design_sha256']}, expected "
                f"{charter['design_sha256']}"
            )
        _validate_manifest(manifest, charter, context="merge shard manifest")
        shard_id = manifest["shard_id"]
        if shard_id in provided_shards:
            raise ShardIntegrityError(f"shard id {shard_id!r} was supplied twice")
        provided_shards.add(shard_id)
        expected_record = expected_shards.get(shard_id)
        if expected_record is None or expected_record.get(
            "manifest_sha256"
        ) != manifest.get("manifest_sha256"):
            raise ShardDesignMismatchError(
                f"shard {shard_id!r} does not match the frozen coordinator "
                "assignment"
            )
        records = manifest.get("input_records")
        if not isinstance(records, dict):
            raise ShardIntegrityError("shard input records are malformed")
        _verify_input_records(shard_root, records, charter)
        run_identity = _validate_shard_run_identity(
            shard_root, manifest, charter,
        )
        shard_results = _scan_shard_results(
            shard_root, manifest, charter, run_identity,
        )
        assigned = {
            _job_key(value, context=f"assignment for {shard_id}")
            for value in manifest["assignment"]
        }
        for record in shard_results:
            key = record["key"]
            if key not in expected_set:
                raise ShardIntegrityError(
                    f"shard result is outside the frozen global matrix: {key!r}"
                )
            if key in incoming:
                other = incoming[key]["source_shard"]
                raise DuplicateShardResultError(
                    f"job {key!r} appears in both shard {other!r} and "
                    f"{shard_id!r}"
                )
            incoming[key] = record
        for record in shard_results:
            if record["key"] not in assigned:
                raise ShardIntegrityError(
                    f"shard {shard_id!r} produced unassigned job "
                    f"{record['key']!r}"
                )

    merged = dict(prior)
    for key, record in incoming.items():
        previous = prior.get(key)
        if previous is not None:
            if previous.get("source_shard") != record["source_shard"]:
                raise DuplicateShardResultError(
                    f"job {key!r} was previously merged from shard "
                    f"{previous.get('source_shard')!r} and is now claimed by "
                    f"{record['source_shard']!r}"
                )
            if previous.get("result_sha256") != record["result_sha256"]:
                raise ShardIntegrityError(
                    f"shard result drifted after a prior merge: {key!r}"
                )
            continue
        destination = root / key[0] / key[1] / f"seed_{key[2]}" / "result.json"
        _copy_job_tree(record["path"], destination)
        merged[key] = {
            **record,
            "path": destination,
        }

    merge_entries = []
    results: list[dict[str, Any]] = []
    for key in expected:
        record = merged.get(key)
        if record is None:
            continue
        results.append(dict(record["result"]))
        merge_entries.append({
            "job": _job_payload(key),
            "result_sha256": record["result_sha256"],
            "source_shard": record["source_shard"],
            "shard_manifest_sha256": record["shard_manifest_sha256"],
            "shard_run_identity_sha256": record[
                "shard_run_identity_sha256"
            ],
        })
    verify_result_lineage(results, charter)
    coverage = _coverage(expected, merged)
    contributing_shards = provided_shards | {
        str(record["source_shard"]) for record in merged.values()
    }
    missing_shards = sorted(set(expected_shards) - contributing_shards)
    merge_payload = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "status": "complete" if coverage["complete"] else "incomplete_coverage",
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "charter": dict(coordinator["charter"]),
        "coordinator_sha256": coordinator["coordinator_sha256"],
        **_runtime_references(charter),
        "coverage": coverage,
        "provided_shards": sorted(contributing_shards),
        "missing_shards": missing_shards,
        "results": merge_entries,
    }
    merge_doc = _seal(merge_payload, "merge_sha256")
    write_json_atomic(root / MERGE_FILENAME, merge_doc)

    authority_warnings = [
        f"benchmark {benchmark['name']!r} uses spec "
        f"{benchmark['spec_metric']!r} with authority "
        f"{benchmark['spec_authority']!r}, not a verified A4_reporting "
        "certificate; treat its campaign result as provisional"
        for benchmark in charter["design"]["benchmarks"]
        if benchmark["spec_authority"] != "A4_reporting"
    ]
    report = {
        "name": charter["design"]["campaign"]["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": merge_payload["status"],
        "coverage": coverage,
        "charter": {
            "path": str(root / CHARTER_FILENAME),
            "schema_version": charter["schema_version"],
            "design_sha256": charter["design_sha256"],
            "document_sha256": charter["document_sha256"],
        },
        "sharding": {
            "coordinator_path": str(root / COORDINATOR_FILENAME),
            "coordinator_sha256": coordinator["coordinator_sha256"],
            "merge_path": str(root / MERGE_FILENAME),
            "merge_sha256": merge_doc["merge_sha256"],
            "provided_shards": sorted(contributing_shards),
            "missing_shards": missing_shards,
        },
        "campaign_inputs": charter["design"]["external_inputs"],
        "authority_warnings": authority_warnings,
        "config": {
            **charter["design"]["campaign"],
            "out_dir": str(root),
        },
        "aggregates": aggregate(results),
        "jobs": [
            {key: value for key, value in result.items() if key != "spec_series"}
            for result in results
        ],
        "run_context": capture_run_context(
            root, None, behavior_goal=charter["design"]["campaign"]["name"],
        ),
    }
    write_json_atomic(root / "campaign_report.json", report)
    (root / "report.html").write_text(_report_html(report), encoding="utf-8")
    return report


__all__ = [
    "COORDINATOR_FILENAME",
    "SHARD_MANIFEST_FILENAME",
    "SHARD_RUN_FILENAME",
    "SHARD_REPORT_FILENAME",
    "MERGE_FILENAME",
    "ShardError",
    "ShardIntegrityError",
    "ShardDesignMismatchError",
    "DuplicateShardResultError",
    "prepare_sharded_campaign",
    "run_campaign_shard",
    "merge_sharded_campaign",
]
