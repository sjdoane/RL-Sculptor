"""Fail-closed discovery and resolution of interrupted PPO snapshots.

Periodic ``model_N.pt`` files are useful research inputs, but they are not
completed policies and their original path is mutable when an interrupted
outer iteration is retried.  This module reconstructs only evidence-backed
legacy receipts, copies the selected bytes into a server-owned immutable
cache, and exposes opaque ids plus content digests to the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from sculptor.policy_contract import (
    build_project_policy_contract,
    contract_fingerprint,
    recovery_snapshot_receipt_fingerprint,
)
from sculptor.world.artifacts import WorldArtifactStore, file_sha256

from backend.services.iteration_completion import is_completed_iteration


_MODEL_RE = re.compile(r"^model_(?P<step>[1-9][0-9]*)\.pt$")
_ITER_RE = re.compile(r"^iter_(?P<iteration>[0-9]+)$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^snap_[a-f0-9]{24}$")
_EVENT_PREFIX = "[SCULPT-EVENT] "


class RecoverySnapshotError(ValueError):
    """A recovery candidate failed provenance or content attestation."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoverySnapshotError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RecoverySnapshotError(f"{label} must be a JSON object: {path}")
    return value


def _assert_plain_file(path: Path, *, parent: Path, label: str) -> Path:
    """Return an exact regular file while rejecting every symlink component."""
    parent = parent.resolve(strict=True)
    candidate = Path(path)
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise RecoverySnapshotError(f"{label} escapes its authority root") from exc
    current = parent
    for part in candidate.relative_to(parent).parts:
        current = current / part
        if current.is_symlink():
            raise RecoverySnapshotError(f"{label} contains a symlink: {current}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise RecoverySnapshotError(
            f"{label} resolves outside its authority root"
        ) from exc
    if resolved.parent != candidate.parent.resolve(strict=True):
        raise RecoverySnapshotError(f"{label} parent changed during resolution")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RecoverySnapshotError(f"{label} is absent or empty: {candidate}")
    return resolved


def _events_from_log(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RecoverySnapshotError(f"worker log is unreadable: {path}") from exc
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        marker = raw_line.find(_EVENT_PREFIX)
        if marker < 0:
            continue
        try:
            event = json.loads(raw_line[marker + len(_EVENT_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, text


def _successful_train_checkpoints(text: str) -> list[str]:
    """Return exact checkpoint paths from successful train subprocess output.

    ``iter_progress`` is an outer orchestration signal and may be advanced to
    the requested total while a failed train subprocess is being torn down.
    The train runner's own compact JSON result is therefore the only legacy
    evidence that a final ``model_(steps - 1).pt`` save was handed back to the
    orchestrator before post-training evaluation began.
    """
    checkpoints: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("status") == "ok"
            and isinstance(value.get("checkpoint"), str)
        ):
            checkpoints.append(value["checkpoint"])
    return checkpoints


def _attested_run_context(
    project_dir: Path,
    *,
    events: list[dict[str, Any]],
    iteration: int,
) -> dict[str, Any] | None:
    """Return a content-bound run context, or ``None`` when it is ambiguous.

    Scratch training has no inbound policy-load receipt.  Its own canonical
    run context and immutable world selection are therefore the authorities
    from which the produced policy contract can be reconstructed.  The report
    is mutable until admitted, so every event field is checked against its
    bytes and the exact config bytes before it may be copied into the immutable
    recovery cache.
    """
    context_events = [
        event for event in events
        if event.get("type") == "run_context_captured"
    ]
    if len(context_events) != 1:
        return None
    event = context_events[0]
    try:
        context_path = _assert_plain_file(
            Path(str(event["path"])).expanduser(),
            parent=project_dir / "reports",
            label="run context",
        )
    except (KeyError, OSError, RecoverySnapshotError, RuntimeError):
        return None
    if context_path != project_dir / "reports" / "run_context.json":
        return None
    try:
        context = _read_json_object(context_path, label="run context")
    except RecoverySnapshotError:
        return None
    start_iter = context.get("start_iter")
    iterations = context.get("iterations")
    if (
        type(start_iter) is not int
        or type(iterations) is not int
        or iterations <= 0
        or not start_iter <= iteration < start_iter + iterations
    ):
        return None
    code = context.get("code_git")
    config = context.get("config")
    seeds = context.get("seeds")
    if not all(isinstance(value, dict) for value in (code, config, seeds)):
        return None
    assert isinstance(code, dict)
    assert isinstance(config, dict)
    assert isinstance(seeds, dict)
    code_sha = code.get("sha")
    code_dirty = code.get("dirty")
    config_sha = config.get("sha256")
    base_seed = seeds.get("base_seed")
    if (
        not isinstance(code_sha, str)
        or not re.fullmatch(r"[a-f0-9]{40}", code_sha)
        or type(code_dirty) is not bool
        or not isinstance(config_sha, str)
        or not _SHA_RE.fullmatch(config_sha)
        or type(base_seed) is not int
        or code_dirty is not False
        or event.get("code_sha") != code_sha
        or event.get("code_dirty") is not code_dirty
        or event.get("config_sha256") != config_sha
        or event.get("base_seed") != base_seed
    ):
        return None
    try:
        config_path = _assert_plain_file(
            Path(str(config["path"])).expanduser(),
            parent=project_dir,
            label="run config",
        )
    except (KeyError, OSError, RecoverySnapshotError, RuntimeError):
        return None
    if config_path != project_dir / "config.toml":
        return None
    if file_sha256(config_path) != config_sha:
        return None
    return {
        "path": context_path,
        "sha256": file_sha256(context_path),
        "config_path": config_path,
        "config_sha256": config_sha,
        "context": context,
    }


def _assert_contract_matches_run_context(
    contract: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """Bind a scratch contract to the runtime versions that produced it."""
    packages = context.get("packages")
    versions = contract.get("versions")
    identity = contract.get("identity")
    config = context.get("config")
    effective = config.get("effective") if isinstance(config, dict) else None
    adapter = effective.get("adapter") if isinstance(effective, dict) else None
    adapter_cfg = adapter.get("config") if isinstance(adapter, dict) else None
    if not all(isinstance(value, dict) for value in (
        packages, versions, identity, adapter, adapter_cfg,
    )):
        raise RecoverySnapshotError(
            "scratch run context cannot attest the effective policy contract"
        )
    assert isinstance(packages, dict)
    assert isinstance(versions, dict)
    assert isinstance(identity, dict)
    assert isinstance(adapter, dict)
    assert isinstance(adapter_cfg, dict)
    torch_version = packages.get("torch")
    if not isinstance(torch_version, str):
        raise RecoverySnapshotError("scratch run context has no torch version")
    torch_match = re.match(r"^(\d+\.\d+)", torch_version)
    expected = {
        "torch": torch_match.group(1) if torch_match else torch_version,
        "mjlab": packages.get("mjlab"),
        "rsl_rl": packages.get("rsl-rl-lib"),
        "adapter": packages.get("reward-sculptor"),
    }
    if any(not isinstance(value, str) for value in expected.values()):
        raise RecoverySnapshotError(
            "scratch run context has incomplete policy runtime versions"
        )
    if any(versions.get(key) != value for key, value in expected.items()):
        raise RecoverySnapshotError(
            "scratch effective policy contract differs from run-context versions"
        )
    task_id = adapter_cfg.get("task_id") or adapter_cfg.get("env_id")
    if (
        identity.get("adapter_class") != adapter.get("class")
        or identity.get("task_id") != task_id
    ):
        raise RecoverySnapshotError(
            "scratch effective policy contract differs from run-context adapter"
        )


def _matching_origin_log(
    project_dir: Path,
    *,
    iteration: int,
    ppo_step: int,
) -> dict[str, Any]:
    """Return the strongest terminal worker-log evidence for one iteration."""
    candidates: list[dict[str, Any]] = []
    runs_dir = project_dir / "runs"
    for log_path in runs_dir.glob("_run_job_*.log"):
        if log_path.is_symlink() or not log_path.is_file():
            continue
        events, text = _events_from_log(log_path)
        started = [
            event for event in events
            if event.get("type") == "iter_started"
            and event.get("iter") == iteration
        ]
        if len(started) != 1:
            continue
        pins = [
            event for event in events
            if event.get("type") == "artifact_tuple_pinned"
            and event.get("iter") == iteration
        ]
        effective = [
            event for event in events
            if event.get("type") in {
                "warm_start_observation_extended",
                "warm_start_observation_contract_verified",
            }
            and event.get("effective_policy_contract")
        ]
        loaded_events = [
            event for event in events
            if event.get("type") == "warm_start_loaded"
        ]
        loaded = [
            event for event in loaded_events
            if event.get("load_cfg_keys") in (
                ["actor"], ["actor", "critic"],
            )
        ]
        if len(pins) != 1:
            continue
        warm_start_source = started[0].get("warm_start_source")
        run_context = _attested_run_context(
            project_dir, events=events, iteration=iteration,
        )
        effective_event: dict[str, Any] | None = None
        effective_contract_path: Path | None = None
        if warm_start_source:
            # The inbound transfer role does not define the produced PPO
            # snapshot.  Both actor-only and actor+critic loads create fresh
            # actor+critic training saves, but the load itself must have
            # succeeded exactly once and its effective contract must be the
            # iteration-owned worker artifact.
            if (
                len(effective) != 1
                or len(loaded_events) != 1
                or len(loaded) != 1
            ):
                continue
            effective_event = effective[0]
            try:
                effective_contract_path = _assert_plain_file(
                    Path(str(effective_event["effective_policy_contract"])),
                    parent=project_dir / "runs" / f"iter_{iteration}",
                    label="effective policy contract",
                )
            except (KeyError, OSError, RecoverySnapshotError, RuntimeError):
                continue
            if effective_contract_path != (
                project_dir / "runs" / f"iter_{iteration}"
                / "warm_start_effective_policy_contract.json"
            ):
                continue
        else:
            # A scratch run has no inbound load receipt.  It is admissible
            # only when no warm-start evidence is mixed in and the canonical
            # run context/config bytes are still available for exact contract
            # reconstruction from this iteration's pinned selection.
            if effective or loaded_events or run_context is None:
                continue
        last_learned = max(
            (
                int(event["rl_iter"])
                for event in events
                if event.get("type") == "learning_vitals"
                and type(event.get("rl_iter")) is int
            ),
            default=-1,
        )
        last_observed = max(
            (
                int(event["rl_iter"])
                for event in events
                if event.get("type") in {"learning_vitals", "iter_progress"}
                and type(event.get("rl_iter")) is int
            ),
            default=-1,
        )
        terminal_error = (
            "RuntimeError: mjlab runner exited" in text
            or "RuntimeError: mjlab train runner exited" in text
            or "RuntimeError: mjlab rollout runner exited" in text
            or "sculpt exited with code" in text
            or any(event.get("type") == "run_errored" for event in events)
        )
        if (
            not terminal_error
            or last_observed < ppo_step
            or last_learned < ppo_step
        ):
            continue
        requested_steps = started[0].get("steps")
        if type(requested_steps) is not int or requested_steps <= 0:
            continue
        canonical_checkpoint = (
            project_dir / "runs" / f"iter_{iteration}" / "checkpoint.pt"
        )
        successful_train_checkpoints = _successful_train_checkpoints(text)
        exact_train_handoffs = [
            checkpoint
            for checkpoint in successful_train_checkpoints
            if Path(checkpoint).expanduser() == canonical_checkpoint
        ]
        training_completed = len(exact_train_handoffs) == 1
        # A final RSL-RL save is model_(steps - 1).pt.  Outer progress alone
        # cannot attest it because failure cleanup may still publish the
        # requested total.  Partial saves remain admissible from interrupted
        # training, but final saves require the train subprocess handoff.
        if ppo_step >= requested_steps - 1 and not training_completed:
            continue
        job_id = log_path.stem.removeprefix("_run_")
        pin = pins[0]
        candidates.append({
            "job_id": job_id,
            "status": "errored",
            "log_path": log_path,
            "log_sha256": file_sha256(log_path),
            "last_observed_ppo_step": last_observed,
            "last_learned_ppo_step": last_learned,
            "requested_steps": requested_steps,
            "training_completed": training_completed,
            "training_checkpoint_path": (
                canonical_checkpoint if training_completed else None
            ),
            "selection": pin.get("selection"),
            "tuple_hash": pin.get("tuple_hash"),
            "contract_authority": (
                "warm_start_effective_contract_event"
                if effective_event is not None
                else "scratch_run_context_selection"
            ),
            "effective_policy_contract_path": effective_contract_path,
            "effective_policy_contract_sha256": (
                effective_event.get("effective_policy_contract_sha256")
                if effective_event is not None
                else None
            ),
            "warm_start_loaded": loaded[0] if loaded else None,
            "run_context": run_context,
            "modified_at": log_path.stat().st_mtime,
        })
    if not candidates:
        raise RecoverySnapshotError(
            f"no terminal worker log attests interrupted iteration {iteration}"
        )
    if len(candidates) != 1:
        raise RecoverySnapshotError(
            "multiple terminal worker logs could have produced the same "
            f"iter_{iteration}/model_{ppo_step}.pt; no exact producer receipt "
            "exists"
        )
    return candidates[0]


def _cache_evidence_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    """Copy one authority byte-string once and reject later conflicts."""
    if destination.exists() or destination.is_symlink():
        cached = _assert_plain_file(
            destination, parent=destination.parent, label=label,
        )
        if file_sha256(cached) != expected_sha256:
            raise RecoverySnapshotError(
                f"immutable {label} conflicts with its content identity"
            )
        return cached
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        shutil.copyfile(source, tmp_path)
        with tmp_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        if file_sha256(tmp_path) != expected_sha256:
            raise RecoverySnapshotError(f"{label} changed while copying")
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    if file_sha256(source) != expected_sha256:
        raise RecoverySnapshotError(f"origin {label} changed while copying")
    return _assert_plain_file(
        destination, parent=destination.parent, label=label,
    )


def _cache_evidence_bytes(
    value: bytes,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    """Materialize deterministic derived evidence into the immutable cache."""
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise RecoverySnapshotError(f"derived {label} digest is inconsistent")
    if destination.exists() or destination.is_symlink():
        cached = _assert_plain_file(
            destination, parent=destination.parent, label=label,
        )
        if file_sha256(cached) != expected_sha256:
            raise RecoverySnapshotError(
                f"immutable {label} conflicts with its content identity"
            )
        return cached
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    try:
        if file_sha256(tmp_path) != expected_sha256:
            raise RecoverySnapshotError(f"derived {label} changed while writing")
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _assert_plain_file(
        destination, parent=destination.parent, label=label,
    )


def _materialize_legacy_receipt(
    project_dir: Path,
    *,
    iteration: int,
    ppo_step: int,
    origin_checkpoint: Path,
) -> dict[str, Any]:
    project_dir = project_dir.expanduser().resolve(strict=True)
    runs_dir = (project_dir / "runs").resolve(strict=True)
    iter_dir = runs_dir / f"iter_{iteration}"
    logs_dir = iter_dir / "logs"
    origin_checkpoint = _assert_plain_file(
        origin_checkpoint, parent=logs_dir, label="origin PPO snapshot",
    )
    expected_name = f"model_{ppo_step}.pt"
    if origin_checkpoint.name != expected_name or origin_checkpoint.parent != logs_dir:
        raise RecoverySnapshotError(
            "origin PPO snapshot is not the exact iter_N/logs/model_S.pt path"
        )

    log_evidence = _matching_origin_log(
        project_dir,
        iteration=iteration,
        ppo_step=ppo_step,
    )
    if ppo_step > int(log_evidence["last_observed_ppo_step"]):
        raise RecoverySnapshotError(
            "PPO snapshot is newer than the worker's truthful observed progress"
        )
    if (
        log_evidence.get("training_completed")
        and ppo_step >= int(log_evidence["requested_steps"]) - 1
    ):
        training_checkpoint = _assert_plain_file(
            Path(str(log_evidence["training_checkpoint_path"])),
            parent=iter_dir,
            label="successful training checkpoint",
        )
        if (
            training_checkpoint != iter_dir / "checkpoint.pt"
            or file_sha256(training_checkpoint) != file_sha256(origin_checkpoint)
        ):
            raise RecoverySnapshotError(
                "final PPO snapshot differs from the successful training checkpoint"
            )

    raw_selection = log_evidence.get("selection")
    if not isinstance(raw_selection, str) or not re.fullmatch(
        r"selection_v[1-9][0-9]*\.json", raw_selection
    ):
        raise RecoverySnapshotError("worker log has no immutable selection name")
    selection_path = _assert_plain_file(
        project_dir / "env" / raw_selection,
        parent=(project_dir / "env"),
        label="source selection",
    )
    selection = WorldArtifactStore(project_dir).read_selection(selection_path)
    if selection is None:
        raise RecoverySnapshotError("source selection failed component verification")
    if selection.tuple_hash != log_evidence.get("tuple_hash"):
        raise RecoverySnapshotError(
            "source selection tuple differs from the worker's pinned tuple"
        )

    contract_authority = str(log_evidence["contract_authority"])
    effective_contract_path = log_evidence.get(
        "effective_policy_contract_path"
    )
    run_context_evidence = log_evidence.get("run_context")
    if contract_authority == "warm_start_effective_contract_event":
        if not isinstance(effective_contract_path, Path):
            raise RecoverySnapshotError(
                "worker event has no effective policy contract artifact"
            )
        effective_contract = _read_json_object(
            effective_contract_path, label="effective policy contract",
        )
        effective_contract_bytes: bytes | None = None
        effective_contract_file_sha = file_sha256(effective_contract_path)
        effective_contract_origin_path: str | None = str(
            effective_contract_path
        )
        # Worker events pin the canonical contract fingerprint, not the raw
        # pretty-printed JSON bytes.  The receipt embeds the source object and
        # retains the exact sidecar bytes, binding both representations.
        effective_sha = contract_fingerprint(effective_contract)
        if effective_sha != log_evidence["effective_policy_contract_sha256"]:
            raise RecoverySnapshotError(
                "effective policy contract digest differs from the worker event"
            )
    elif contract_authority == "scratch_run_context_selection":
        if not isinstance(run_context_evidence, dict):
            raise RecoverySnapshotError(
                "scratch snapshot has no attested run context"
            )
        try:
            effective_contract = build_project_policy_contract(
                project_dir, world_selection_path=selection_path,
            )
        except Exception as exc:
            raise RecoverySnapshotError(
                "scratch effective policy contract could not be reconstructed"
            ) from exc
        _assert_contract_matches_run_context(
            effective_contract, run_context_evidence["context"],
        )
        effective_sha = contract_fingerprint(effective_contract)
        effective_contract_bytes = _canonical_json_bytes(effective_contract)
        effective_contract_file_sha = hashlib.sha256(
            effective_contract_bytes
        ).hexdigest()
        effective_contract_origin_path = None
    else:
        raise RecoverySnapshotError("unsupported effective contract authority")

    artifact_tuple_path = _assert_plain_file(
        iter_dir / "artifact_tuple.json",
        parent=iter_dir,
        label="iteration artifact tuple",
    )
    artifact_tuple = _read_json_object(
        artifact_tuple_path, label="iteration artifact tuple",
    )
    matches_pinned_selection = (
        _canonical_json_bytes(artifact_tuple)
        == _canonical_json_bytes(selection.to_dict())
    )

    checkpoint_sha = file_sha256(origin_checkpoint)
    checkpoint_bytes = origin_checkpoint.stat().st_size
    identity = hashlib.sha256(
        (
            "evidence-cache-v2:"
            f"{project_dir.name}:{iteration}:{ppo_step}:{checkpoint_sha}:"
            f"{log_evidence['log_sha256']}:{effective_sha}:"
            f"{effective_contract_file_sha}:"
            f"{(run_context_evidence or {}).get('sha256', '')}:"
            f"{(run_context_evidence or {}).get('config_sha256', '')}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    snapshot_id = f"snap_{identity}"
    recovery_root = runs_dir / "_recovery"
    if recovery_root.is_symlink():
        raise RecoverySnapshotError("recovery cache root is a symlink")
    recovery_root.mkdir(parents=True, exist_ok=True)
    if recovery_root.resolve(strict=True).parent != runs_dir:
        raise RecoverySnapshotError("recovery cache root escapes project runs")
    recovery_dir = recovery_root / snapshot_id
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if (
        recovery_dir.is_symlink()
        or recovery_dir.resolve(strict=True).parent
        != recovery_root.resolve(strict=True)
    ):
        raise RecoverySnapshotError(
            "recovery cache directory is a symlink or escapes its root"
        )
    selection_sha = file_sha256(selection_path)
    artifact_tuple_sha = file_sha256(artifact_tuple_path)
    cached_checkpoint = _cache_evidence_file(
        origin_checkpoint,
        recovery_dir / "checkpoint.pt",
        expected_sha256=checkpoint_sha,
        label="cached PPO snapshot",
    )
    if effective_contract_bytes is None:
        assert isinstance(effective_contract_path, Path)
        cached_effective_contract = _cache_evidence_file(
            effective_contract_path,
            recovery_dir / "effective_policy_contract.json",
            expected_sha256=effective_contract_file_sha,
            label="cached effective policy contract",
        )
    else:
        cached_effective_contract = _cache_evidence_bytes(
            effective_contract_bytes,
            recovery_dir / "effective_policy_contract.json",
            expected_sha256=effective_contract_file_sha,
            label="cached effective policy contract",
        )
    cached_selection = _cache_evidence_file(
        selection_path,
        recovery_dir / "source_selection.json",
        expected_sha256=selection_sha,
        label="cached source selection",
    )
    cached_artifact_tuple = _cache_evidence_file(
        artifact_tuple_path,
        recovery_dir / "artifact_tuple.json",
        expected_sha256=artifact_tuple_sha,
        label="cached artifact tuple",
    )
    cached_log = _cache_evidence_file(
        Path(log_evidence["log_path"]),
        recovery_dir / "worker.log",
        expected_sha256=str(log_evidence["log_sha256"]),
        label="cached worker log",
    )
    cached_run_context: Path | None = None
    cached_run_config: Path | None = None
    if isinstance(run_context_evidence, dict):
        cached_run_context = _cache_evidence_file(
            Path(run_context_evidence["path"]),
            recovery_dir / "run_context.json",
            expected_sha256=str(run_context_evidence["sha256"]),
            label="cached run context",
        )
        cached_run_config = _cache_evidence_file(
            Path(run_context_evidence["config_path"]),
            recovery_dir / "config.toml",
            expected_sha256=str(run_context_evidence["config_sha256"]),
            label="cached run config",
        )

    receipt: dict[str, Any] = {
        "schema": 1,
        "kind": "interrupted_ppo_snapshot",
        "snapshot_id": snapshot_id,
        "checkpoint": {
            "path": str(cached_checkpoint),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
            "ppo_step": ppo_step,
            "origin_path": str(origin_checkpoint),
            "origin_sha256": checkpoint_sha,
        },
        "source": {
            "effective_policy_contract_authority": contract_authority,
            "producer_initialization_roles": (
                log_evidence["warm_start_loaded"].get("load_cfg_keys")
                if isinstance(log_evidence.get("warm_start_loaded"), dict)
                else []
            ),
            "effective_policy_contract": effective_contract,
            "effective_policy_contract_sha256": effective_sha,
            "effective_policy_contract_path": str(cached_effective_contract),
            "effective_policy_contract_file_sha256": (
                effective_contract_file_sha
            ),
            "effective_policy_contract_origin_path": (
                effective_contract_origin_path
            ),
            "selection_path": str(cached_selection),
            "selection_sha256": selection_sha,
            "selection_origin_path": str(selection_path),
            "selection_version": selection.selection_version,
            "tuple_hash": selection.tuple_hash,
            "artifact_tuple_path": str(cached_artifact_tuple),
            "artifact_tuple_sha256": artifact_tuple_sha,
            "artifact_tuple_origin_path": str(artifact_tuple_path),
            "matches_pinned_selection": matches_pinned_selection,
            "job_id": log_evidence["job_id"],
            "status": log_evidence["status"],
            "log_path": str(cached_log),
            "log_sha256": log_evidence["log_sha256"],
            "log_origin_path": str(log_evidence["log_path"]),
            "last_observed_ppo_step": log_evidence["last_observed_ppo_step"],
            "iteration": iteration,
            **(
                {
                    "run_context_path": str(cached_run_context),
                    "run_context_sha256": run_context_evidence["sha256"],
                    "run_context_origin_path": str(
                        run_context_evidence["path"]
                    ),
                    "run_config_path": str(cached_run_config),
                    "run_config_sha256": run_context_evidence[
                        "config_sha256"
                    ],
                    "run_config_origin_path": str(
                        run_context_evidence["config_path"]
                    ),
                }
                if isinstance(run_context_evidence, dict)
                else {}
            ),
        },
        "provenance_status": "legacy_reconstructed",
    }
    receipt["receipt_digest"] = recovery_snapshot_receipt_fingerprint(receipt)
    receipt_path = recovery_dir / "receipt.json"
    encoded = _canonical_json_bytes(receipt)
    if receipt_path.exists():
        existing = _read_json_object(receipt_path, label="recovery receipt")
        if _canonical_json_bytes(existing) != encoded:
            raise RecoverySnapshotError(
                "immutable recovery receipt conflicts with reconstructed evidence"
            )
    else:
        with tempfile.NamedTemporaryFile(
            dir=recovery_dir, prefix=".receipt-", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_receipt = Path(handle.name)
        try:
            os.replace(tmp_receipt, receipt_path)
        finally:
            tmp_receipt.unlink(missing_ok=True)
    return receipt


def _public_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = dict(receipt["checkpoint"])
    source = dict(receipt["source"])
    return {
        "snapshot_id": receipt["snapshot_id"],
        "iteration": source["iteration"],
        "ppo_step": checkpoint["ppo_step"],
        "source_job_id": source["job_id"],
        "source_job_status": source["status"],
        "last_observed_ppo_iteration": source["last_observed_ppo_step"],
        "checkpoint_bytes": checkpoint["bytes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "receipt_digest": receipt["receipt_digest"],
        "provenance_status": receipt["provenance_status"],
        "selectable": True,
        "blocker": None,
    }


def discover_recovery_snapshots(project_dir: Path) -> list[dict[str, Any]]:
    """Discover immutable cached receipts, then admit new legacy candidates.

    A valid cached receipt is the authority after admission.  In particular,
    origin iteration completion, deletion, or mutation must not hide it or
    cause the same ``(iteration, ppo_step)`` to be silently reconstructed as a
    different selectable snapshot.
    """
    project_dir = Path(project_dir).expanduser().resolve(strict=True)
    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return []
    receipts_by_iteration: dict[int, dict[str, Any]] = {}
    admitted_progress: set[tuple[int, int]] = set()
    recovery_root = runs_dir / "_recovery"
    if recovery_root.is_dir() and not recovery_root.is_symlink():
        for recovery_dir in sorted(recovery_root.glob("snap_*")):
            if (
                not _SNAPSHOT_ID_RE.fullmatch(recovery_dir.name)
                or recovery_dir.is_symlink()
                or not recovery_dir.is_dir()
            ):
                continue
            try:
                receipt_path = _assert_plain_file(
                    recovery_dir / "receipt.json",
                    parent=recovery_dir,
                    label="recovery receipt",
                )
                disclosed = _read_json_object(
                    receipt_path, label="recovery receipt",
                )
                checkpoint = disclosed.get("checkpoint")
                if not isinstance(checkpoint, dict):
                    continue
                receipt = resolve_recovery_snapshot(
                    project_dir,
                    snapshot_id=recovery_dir.name,
                    checkpoint_sha256=str(checkpoint.get("sha256", "")),
                    receipt_digest=str(disclosed.get("receipt_digest", "")),
                )[1]
                source = receipt["source"]
                progress = (
                    int(source["iteration"]),
                    int(receipt["checkpoint"]["ppo_step"]),
                )
            except (OSError, RecoverySnapshotError, TypeError, ValueError):
                # Invalid or pre-cache receipts are never selectable authority.
                # They also do not suppress a fresh, fully cached admission.
                continue
            admitted_progress.add(progress)
            incumbent = receipts_by_iteration.get(progress[0])
            if (
                incumbent is None
                or int(receipt["checkpoint"]["ppo_step"])
                > int(incumbent["checkpoint"]["ppo_step"])
            ):
                receipts_by_iteration[progress[0]] = receipt

    for iter_dir in runs_dir.glob("iter_*"):
        match = _ITER_RE.fullmatch(iter_dir.name)
        if match is None or iter_dir.is_symlink() or not iter_dir.is_dir():
            continue
        # A final checkpoint is not a completion receipt.  Training can
        # preserve it immediately before rollout/objective evaluation fails.
        # Only the shared marker/full-legacy authority suppresses recovery.
        if is_completed_iteration(iter_dir):
            continue
        iteration = int(match.group("iteration"))
        logs_dir = iter_dir / "logs"
        if not logs_dir.is_dir() or logs_dir.is_symlink():
            continue
        incumbent = receipts_by_iteration.get(iteration)
        incumbent_step = (
            int(incumbent["checkpoint"]["ppo_step"])
            if incumbent is not None
            else -1
        )
        candidates: list[tuple[int, Path]] = []
        for model_path in logs_dir.glob("model_*.pt"):
            model_match = _MODEL_RE.fullmatch(model_path.name)
            if model_match is None:
                continue
            step = int(model_match.group("step"))
            if step <= incumbent_step:
                continue
            candidates.append((step, model_path))
        # Periodic models are implementation detail, not a policy menu. Admit
        # the newest uniquely attested candidate and stop; lower saves remain
        # available only if every newer candidate fails provenance checks.
        for step, model_path in sorted(
            candidates, key=lambda item: item[0], reverse=True,
        ):
            progress = (iteration, step)
            if progress in admitted_progress:
                continue
            try:
                receipt = _materialize_legacy_receipt(
                    project_dir,
                    iteration=iteration,
                    ppo_step=step,
                    origin_checkpoint=model_path,
                )
            except (OSError, RecoverySnapshotError, ValueError):
                continue
            admitted_progress.add(progress)
            receipts_by_iteration[iteration] = receipt
            break
    rows = [_public_summary(receipt) for receipt in receipts_by_iteration.values()]
    return sorted(
        rows,
        key=lambda row: (row["iteration"], row["ppo_step"]),
        reverse=True,
    )


def resolve_recovery_snapshot(
    project_dir: Path,
    *,
    snapshot_id: str,
    checkpoint_sha256: str,
    receipt_digest: str,
) -> tuple[Path, dict[str, Any]]:
    """Re-attest an opaque recovery receipt and return its immutable copy."""
    if not _SNAPSHOT_ID_RE.fullmatch(str(snapshot_id)):
        raise RecoverySnapshotError("recovery snapshot id is not canonical")
    if not _SHA_RE.fullmatch(str(checkpoint_sha256)):
        raise RecoverySnapshotError("recovery checkpoint digest is not canonical")
    if not _SHA_RE.fullmatch(str(receipt_digest)):
        raise RecoverySnapshotError("recovery receipt digest is not canonical")
    project_dir = Path(project_dir).expanduser().resolve(strict=True)
    recovery_root = project_dir / "runs" / "_recovery"
    recovery_dir = recovery_root / snapshot_id
    receipt_path = _assert_plain_file(
        recovery_dir / "receipt.json",
        parent=recovery_dir,
        label="recovery receipt",
    )
    receipt = _read_json_object(receipt_path, label="recovery receipt")
    actual_receipt_digest = recovery_snapshot_receipt_fingerprint(receipt)
    if (
        receipt.get("receipt_digest") != actual_receipt_digest
        or receipt_digest != actual_receipt_digest
    ):
        raise RecoverySnapshotError("recovery receipt changed after selection")
    if receipt.get("snapshot_id") != snapshot_id:
        raise RecoverySnapshotError("recovery receipt has another snapshot id")
    if (
        receipt.get("schema") != 1
        or receipt.get("kind") != "interrupted_ppo_snapshot"
    ):
        raise RecoverySnapshotError("recovery receipt kind/schema changed")
    if receipt.get("provenance_status") not in {
        "origin_persisted", "legacy_reconstructed",
    }:
        raise RecoverySnapshotError("recovery receipt provenance is unsupported")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RecoverySnapshotError("recovery receipt has no checkpoint object")
    cached_checkpoint = _assert_plain_file(
        recovery_dir / "checkpoint.pt",
        parent=recovery_dir,
        label="cached PPO snapshot",
    )
    if Path(str(checkpoint.get("path"))).resolve(strict=True) != cached_checkpoint:
        raise RecoverySnapshotError("recovery receipt checkpoint path changed")
    actual_sha = file_sha256(cached_checkpoint)
    if (
        checkpoint.get("sha256") != checkpoint_sha256
        or actual_sha != checkpoint_sha256
        or checkpoint.get("bytes") != cached_checkpoint.stat().st_size
    ):
        raise RecoverySnapshotError("recovery checkpoint changed after selection")
    if checkpoint.get("origin_sha256") != checkpoint_sha256:
        raise RecoverySnapshotError("origin PPO snapshot disclosure changed")
    source = receipt.get("source")
    if not isinstance(source, dict):
        raise RecoverySnapshotError("recovery receipt has no source evidence")
    iteration = source.get("iteration")
    ppo_step = checkpoint.get("ppo_step")
    last_observed = source.get("last_observed_ppo_step")
    if (
        type(iteration) is not int
        or iteration < 0
        or type(ppo_step) is not int
        or ppo_step <= 0
        or type(last_observed) is not int
        or last_observed < ppo_step
    ):
        raise RecoverySnapshotError("recovery source progress is invalid")
    if source.get("status") not in {"errored", "stopped"}:
        raise RecoverySnapshotError("recovery source job is not interrupted")
    expected_origin = (
        project_dir / "runs" / f"iter_{iteration}" / "logs"
        / f"model_{ppo_step}.pt"
    )
    origin_disclosure = Path(str(checkpoint.get("origin_path", "")))
    if not origin_disclosure.is_absolute() or origin_disclosure != expected_origin:
        raise RecoverySnapshotError("recovery origin path is not canonical")

    iter_dir = project_dir / "runs" / f"iter_{iteration}"
    effective_path = _assert_plain_file(
        Path(str(source.get("effective_policy_contract_path"))),
        parent=recovery_dir,
        label="cached effective policy contract",
    )
    if effective_path != recovery_dir / "effective_policy_contract.json":
        raise RecoverySnapshotError(
            "cached effective policy contract path changed"
        )
    if file_sha256(effective_path) != source.get(
        "effective_policy_contract_file_sha256"
    ):
        raise RecoverySnapshotError(
            "cached effective policy contract changed after admission"
        )
    effective_contract = _read_json_object(
        effective_path, label="cached effective policy contract",
    )
    if (
        source.get("effective_policy_contract") != effective_contract
        or source.get("effective_policy_contract_sha256")
        != contract_fingerprint(effective_contract)
    ):
        raise RecoverySnapshotError("effective policy contract evidence changed")
    contract_authority = source.get(
        "effective_policy_contract_authority",
        "warm_start_effective_contract_event",
    )
    producer_roles = source.get("producer_initialization_roles")
    if producer_roles is None:
        # Backwards-compatible validation for already cached schema-1
        # actor+critic receipts created before this disclosure was added.
        producer_roles = ["actor", "critic"]
    effective_origin_value = source.get(
        "effective_policy_contract_origin_path"
    )
    if contract_authority == "warm_start_effective_contract_event":
        if producer_roles not in (["actor"], ["actor", "critic"]):
            raise RecoverySnapshotError(
                "warm-start producer roles changed after admission"
            )
        effective_origin = Path(str(effective_origin_value or ""))
        if effective_origin != (
            iter_dir / "warm_start_effective_policy_contract.json"
        ):
            raise RecoverySnapshotError(
                "effective policy contract origin disclosure changed"
            )
    elif contract_authority == "scratch_run_context_selection":
        if effective_origin_value is not None or producer_roles != []:
            raise RecoverySnapshotError(
                "scratch effective contract claims an origin sidecar"
            )
    else:
        raise RecoverySnapshotError(
            "effective policy contract authority changed after admission"
        )
    selection_path = _assert_plain_file(
        Path(str(source.get("selection_path"))),
        parent=recovery_dir,
        label="cached source selection",
    )
    if selection_path != recovery_dir / "source_selection.json":
        raise RecoverySnapshotError("cached source selection path changed")
    if file_sha256(selection_path) != source.get("selection_sha256"):
        raise RecoverySnapshotError(
            "cached source selection changed after admission"
        )
    selection = _read_json_object(
        selection_path, label="cached source selection",
    )
    if (
        selection.get("selection_version") != source.get("selection_version")
        or selection.get("tuple_hash") != source.get("tuple_hash")
    ):
        raise RecoverySnapshotError("cached source selection no longer verifies")
    selection_origin = Path(str(source.get("selection_origin_path", "")))
    expected_selection_origin = (
        project_dir / "env"
        / f"selection_v{source.get('selection_version')}.json"
    )
    if selection_origin != expected_selection_origin:
        raise RecoverySnapshotError("source selection origin disclosure changed")
    artifact_tuple_path = _assert_plain_file(
        Path(str(source.get("artifact_tuple_path"))),
        parent=recovery_dir,
        label="cached artifact tuple",
    )
    if artifact_tuple_path != recovery_dir / "artifact_tuple.json":
        raise RecoverySnapshotError("cached artifact tuple path changed")
    if file_sha256(artifact_tuple_path) != source.get("artifact_tuple_sha256"):
        raise RecoverySnapshotError("cached artifact tuple changed")
    artifact_tuple = _read_json_object(
        artifact_tuple_path, label="cached artifact tuple",
    )
    actual_tuple_match = (
        _canonical_json_bytes(artifact_tuple)
        == _canonical_json_bytes(selection)
    )
    if actual_tuple_match is not source.get("matches_pinned_selection"):
        raise RecoverySnapshotError(
            "tuple/selection disclosure changed after admission"
        )
    artifact_origin = Path(str(source.get("artifact_tuple_origin_path", "")))
    if artifact_origin != iter_dir / "artifact_tuple.json":
        raise RecoverySnapshotError("artifact tuple origin disclosure changed")
    log_path = _assert_plain_file(
        Path(str(source.get("log_path"))),
        parent=recovery_dir,
        label="cached worker log",
    )
    if log_path != recovery_dir / "worker.log":
        raise RecoverySnapshotError("cached worker log path changed")
    if file_sha256(log_path) != source.get("log_sha256"):
        raise RecoverySnapshotError("cached worker log changed after admission")
    source_job_id = source.get("job_id")
    log_origin = Path(str(source.get("log_origin_path", "")))
    if (
        not isinstance(source_job_id, str)
        or not source_job_id.startswith("job_")
        or log_origin != project_dir / "runs" / f"_run_{source_job_id}.log"
    ):
        raise RecoverySnapshotError("worker log origin disclosure changed")
    if contract_authority == "scratch_run_context_selection":
        run_context_path = _assert_plain_file(
            Path(str(source.get("run_context_path"))),
            parent=recovery_dir,
            label="cached run context",
        )
        run_config_path = _assert_plain_file(
            Path(str(source.get("run_config_path"))),
            parent=recovery_dir,
            label="cached run config",
        )
        if (
            run_context_path != recovery_dir / "run_context.json"
            or run_config_path != recovery_dir / "config.toml"
            or file_sha256(run_context_path) != source.get(
                "run_context_sha256"
            )
            or file_sha256(run_config_path) != source.get("run_config_sha256")
            or Path(str(source.get("run_context_origin_path")))
            != project_dir / "reports" / "run_context.json"
            or Path(str(source.get("run_config_origin_path")))
            != project_dir / "config.toml"
        ):
            raise RecoverySnapshotError(
                "scratch run-context authority changed after admission"
            )
        run_context = _read_json_object(
            run_context_path, label="cached run context",
        )
        config = run_context.get("config")
        if (
            not isinstance(config, dict)
            or config.get("sha256") != source.get("run_config_sha256")
        ):
            raise RecoverySnapshotError(
                "cached run context no longer binds the run config"
            )
        _assert_contract_matches_run_context(effective_contract, run_context)
        cached_events, _ = _events_from_log(log_path)
        context_events = [
            event for event in cached_events
            if event.get("type") == "run_context_captured"
        ]
        if len(context_events) != 1:
            raise RecoverySnapshotError(
                "cached worker log no longer uniquely binds the run context"
            )
        context_event = context_events[0]
        code = run_context.get("code_git")
        seeds = run_context.get("seeds")
        if (
            not isinstance(code, dict)
            or not isinstance(seeds, dict)
            or context_event.get("code_sha") != code.get("sha")
            or context_event.get("code_dirty") is not code.get("dirty")
            or context_event.get("config_sha256")
            != source.get("run_config_sha256")
            or context_event.get("base_seed") != seeds.get("base_seed")
        ):
            raise RecoverySnapshotError(
                "cached worker log/run-context receipt no longer agrees"
            )
    return cached_checkpoint, receipt


__all__ = [
    "RecoverySnapshotError",
    "discover_recovery_snapshots",
    "resolve_recovery_snapshot",
]
