"""Immutable input and completion receipts for expensive run phases.

The orchestration layer may resume training or rollout only when the exact
inputs that produced the retained bytes still match.  This module deliberately
contains no adapter or simulator imports: it only canonicalizes JSON-compatible
facts and hashes files.  That keeps resume admission CPU-only and independently
testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1

_ITERATION_PHASE_MANIFESTS = (
    "train_request_manifest.json",
    "train_input_manifest.json",
    "train_completion_manifest.json",
    "rollout/rollout_input_manifest.json",
    "rollout/rollout_completion_manifest.json",
)
_OPTIONAL_ITERATION_PHASE_MANIFESTS = (
    "evaluation_plan.json",
    "evaluation_results.json",
)
_ITERATION_CHECKPOINT_NAMES = {"checkpoint.pt", "checkpoint.zip"}


class RunManifestError(ValueError):
    """Raised when a claim-bearing manifest cannot be made canonical."""


def canonical_bytes(value: Any) -> bytes:
    """Return the one accepted JSON encoding for manifest evidence."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunManifestError(
            f"run manifest is not canonical JSON: {exc}"
        ) from exc


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(manifest))).hexdigest()


def file_identity(path: Path | str, *, required: bool = True) -> dict[str, Any]:
    """Describe exact bytes without interpreting or executing them."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise RunManifestError(f"required manifest input is missing: {resolved}")
        return {"path": str(resolved), "exists": False}
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _simple_adapter_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, (list, tuple)):
        converted = [_simple_adapter_value(item) for item in value]
        return converted if all(item is not _UNSUPPORTED for item in converted) else _UNSUPPORTED
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        converted = {key: _simple_adapter_value(item) for key, item in value.items()}
        return converted if all(item is not _UNSUPPORTED for item in converted.values()) else _UNSUPPORTED
    return _UNSUPPORTED


_UNSUPPORTED = object()


_ADAPTER_CONTRACT_FIELDS = (
    "adapter_name",
    "env_id",
    "robot",
    "robot_slug",
    "task",
    "device",
    "num_envs",
    "env_spec_path",
    "world_selection_path",
    "reference_clip_id",
    "reference_robot",
    "physics_dt",
    "sim_dt",
    "step_dt",
    "control_dt",
    "decimation",
)


def adapter_identity(adapter: Any) -> dict[str, Any]:
    """Capture stable adapter/runtime facts that can affect phase outputs."""
    identity: dict[str, Any] = {
        "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
    }
    facts: dict[str, Any] = {}
    for name in _ADAPTER_CONTRACT_FIELDS:
        if not hasattr(adapter, name):
            continue
        converted = _simple_adapter_value(getattr(adapter, name))
        if converted is not _UNSUPPORTED:
            facts[name] = converted
    identity["facts"] = facts
    return identity


def software_identity() -> dict[str, Any]:
    """Bind the runtime family without importing the simulator stack."""
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "sculptor_contract": "run-input-manifest-v1",
    }


def build_train_input_manifest(
    *,
    adapter: Any,
    iteration: int,
    reward_module_path: Path,
    steps: int,
    seed: int,
    init_policy_path: Path | None,
    init_policy_mode: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_VERSION,
        "phase": "train",
        "iteration": int(iteration),
        "reward": file_identity(reward_module_path),
        "training_plan": {"steps": int(steps), "seed": int(seed)},
        "initialization": {
            "mode": str(init_policy_mode),
            "policy": (
                file_identity(init_policy_path)
                if init_policy_path is not None
                else None
            ),
        },
        "adapter": adapter_identity(adapter),
        "software": software_identity(),
        "context": dict(context or {}),
    }
    canonical_bytes(manifest)
    return manifest


def build_rollout_input_manifest(
    *,
    adapter: Any,
    iteration: int,
    checkpoint_path: Path,
    reward_module_path: Path | None,
    n_episodes: int,
    seed: int | None,
    max_episode_steps: int | None,
    playback_speed: float | None,
    render_every: int | None,
    fps: float | None,
    render_width: int | None,
    render_height: int | None,
    render_env_index: int | None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_VERSION,
        "phase": "rollout",
        "iteration": int(iteration),
        "checkpoint": file_identity(checkpoint_path),
        "reward": (
            file_identity(reward_module_path)
            if reward_module_path is not None
            else None
        ),
        "evaluation_plan": {
            "n_episodes": int(n_episodes),
            "seed": int(seed) if seed is not None else None,
            "max_episode_steps": (
                int(max_episode_steps) if max_episode_steps is not None else None
            ),
            "playback_speed": (
                float(playback_speed) if playback_speed is not None else None
            ),
            "render_every": int(render_every) if render_every is not None else None,
            "fps": float(fps) if fps is not None else None,
            "render_width": int(render_width) if render_width is not None else None,
            "render_height": int(render_height) if render_height is not None else None,
            "render_env_index": (
                int(render_env_index) if render_env_index is not None else None
            ),
        },
        "adapter": adapter_identity(adapter),
        "software": software_identity(),
        "context": dict(context or {}),
    }
    canonical_bytes(manifest)
    return manifest


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read an object receipt without accepting scalar/list JSON."""
    return _read_json_object(path)


def file_identity_matches(identity: Mapping[str, Any] | None) -> bool:
    """Re-hash a recorded file identity and require exact equality."""
    if not isinstance(identity, Mapping):
        return False
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        observed = file_identity(raw_path, required=bool(identity.get("exists")))
    except RunManifestError:
        return False
    return canonical_bytes(observed) == canonical_bytes(dict(identity))


def input_manifest_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    observed = _read_json_object(path)
    return observed is not None and canonical_bytes(observed) == canonical_bytes(expected)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a small receipt after canonical validation."""
    data = canonical_bytes(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def build_completion_manifest(
    input_manifest: Mapping[str, Any],
    outputs: Iterable[Path],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "phase": str(input_manifest.get("phase") or ""),
        "input_manifest_sha256": manifest_sha256(input_manifest),
        "outputs": [file_identity(path) for path in outputs],
    }


def completion_manifest_matches(
    path: Path,
    input_manifest: Mapping[str, Any],
    outputs: Iterable[Path],
) -> bool:
    observed = _read_json_object(path)
    if observed is None:
        return False
    try:
        expected = build_completion_manifest(input_manifest, outputs)
    except RunManifestError:
        return False
    return canonical_bytes(observed) == canonical_bytes(expected)


def _plain_file_receipt(path: Path) -> dict[str, Any] | None:
    """Return the marker-style identity of one non-linked regular file."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        if size <= 0:
            return None
        return {"sha256": digest.hexdigest(), "bytes": size}
    except OSError:
        return None


def _identity_matches_exact_path(
    identity: Any,
    expected_path: Path,
) -> bool:
    """Verify a run-manifest file identity against one required exact path."""
    if not isinstance(identity, Mapping):
        return False
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        disclosed = Path(raw_path).expanduser()
        if disclosed.is_symlink() or expected_path.is_symlink():
            return False
        if disclosed.resolve(strict=True) != expected_path.resolve(strict=True):
            return False
    except (OSError, RuntimeError):
        return False
    return file_identity_matches(identity)


def _manifest_identity_is_current(identity: Any) -> bool:
    if identity is None:
        return True
    if not isinstance(identity, Mapping):
        return False
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        if Path(raw_path).expanduser().is_symlink():
            return False
    except OSError:
        return False
    return file_identity_matches(identity)


def verify_iteration_completion_marker(
    iter_dir: Path | str,
) -> dict[str, Any] | None:
    """Re-verify a schema-3 iteration and every phase receipt it binds.

    This is the CPU/data-only completion authority shared by report and API
    adapters.  It verifies more than the outer marker's filenames: the train
    request/effective-input relationship, completion-to-input digests, exact
    checkpoint and rollout output identities, and current reward/initialization
    inputs must all remain byte-identical.  Any missing, linked, malformed, or
    contradictory evidence fails closed.
    """
    directory = Path(iter_dir)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or not directory.name.startswith("iter_")
    ):
        return None
    raw_index = directory.name.removeprefix("iter_")
    if not raw_index.isdigit():
        return None
    iteration = int(raw_index)
    marker_path = directory / "iteration_complete.json"
    marker_receipt = _plain_file_receipt(marker_path)
    marker = _read_json_object(marker_path)
    if (
        marker_receipt is None
        or marker is None
        or marker.get("schema") != 3
        or marker.get("state") != "completed"
        or type(marker.get("iter")) is not int
        or marker.get("iter") != iteration
    ):
        return None

    disclosed = marker.get("checkpoint")
    expected_checkpoint_sha = marker.get("checkpoint_sha256")
    expected_checkpoint_bytes = marker.get("checkpoint_bytes")
    if (
        not isinstance(disclosed, str)
        or not disclosed.strip()
        or not isinstance(expected_checkpoint_sha, str)
        or len(expected_checkpoint_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_checkpoint_sha)
        or type(expected_checkpoint_bytes) is not int
        or expected_checkpoint_bytes <= 0
    ):
        return None
    checkpoint = Path(disclosed).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    try:
        checkpoint_resolved = checkpoint.resolve(strict=True)
        if (
            checkpoint.is_symlink()
            or checkpoint_resolved.parent != directory.resolve(strict=True)
            or checkpoint_resolved.name not in _ITERATION_CHECKPOINT_NAMES
        ):
            return None
    except (OSError, RuntimeError):
        return None
    checkpoint_receipt = _plain_file_receipt(checkpoint_resolved)
    if checkpoint_receipt != {
        "sha256": expected_checkpoint_sha,
        "bytes": expected_checkpoint_bytes,
    }:
        return None

    phase_receipts = marker.get("phase_manifests")
    observed_phase_names = (
        set(phase_receipts) if isinstance(phase_receipts, dict) else set()
    )
    allowed_phase_names = set(
        _ITERATION_PHASE_MANIFESTS + _OPTIONAL_ITERATION_PHASE_MANIFESTS
    )
    if (
        not isinstance(phase_receipts, dict)
        or not set(_ITERATION_PHASE_MANIFESTS).issubset(observed_phase_names)
        or not observed_phase_names.issubset(allowed_phase_names)
    ):
        return None
    manifests: dict[str, dict[str, Any]] = {}
    for relative in sorted(observed_phase_names):
        path = directory / relative
        try:
            if path.is_symlink():
                return None
            resolved = path.resolve(strict=True)
            resolved.relative_to(directory.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        receipt = phase_receipts.get(relative)
        if (
            not isinstance(receipt, dict)
            or _plain_file_receipt(resolved) != receipt
        ):
            return None
        manifest = _read_json_object(resolved)
        if manifest is None:
            return None
        manifests[relative] = manifest

    optional_present = observed_phase_names.intersection(
        _OPTIONAL_ITERATION_PHASE_MANIFESTS
    )
    if optional_present and optional_present != set(
        _OPTIONAL_ITERATION_PHASE_MANIFESTS
    ):
        return None
    if optional_present:
        evaluation_plan = manifests["evaluation_plan.json"]
        evaluation_results = manifests["evaluation_results.json"]
        requested_count = evaluation_plan.get("requested_count")
        requested_seeds = evaluation_plan.get("requested_seeds")
        result_rows = evaluation_results.get("results")
        observed_seeds = (
            [row.get("seed") for row in result_rows]
            if isinstance(result_rows, list)
            and all(isinstance(row, dict) for row in result_rows)
            else []
        )
        if (
            evaluation_plan.get("schema") != SCHEMA_VERSION
            or evaluation_plan.get("iteration") != iteration
            or evaluation_plan.get("authority")
            != "precommitted_objective_evaluation_seeds"
            or type(requested_count) is not int
            or requested_count <= 0
            or not isinstance(requested_seeds, list)
            or len(requested_seeds) != requested_count
            or any(type(seed) is not int for seed in requested_seeds)
            or len(set(requested_seeds)) != requested_count
            or evaluation_results.get("schema") != SCHEMA_VERSION
            or evaluation_results.get("iteration") != iteration
            or evaluation_results.get("plan") != evaluation_plan
            or evaluation_results.get("requested_count") != requested_count
            or not isinstance(result_rows, list)
            or len(result_rows) != requested_count
            or any(type(seed) is not int for seed in observed_seeds)
            or set(observed_seeds) != set(requested_seeds)
        ):
            return None
        succeeded = sum(
            isinstance(row, dict) and row.get("status") == "succeeded"
            for row in result_rows
        )
        if (
            succeeded != requested_count
            or evaluation_results.get("completed_count") != succeeded
            or evaluation_results.get("complete") is not True
            or any(
                not isinstance(row, dict)
                or row.get("status") != "succeeded"
                for row in result_rows
            )
        ):
            return None

    train_request = manifests["train_request_manifest.json"]
    train_input = manifests["train_input_manifest.json"]
    train_completion = manifests["train_completion_manifest.json"]
    rollout_input = manifests["rollout/rollout_input_manifest.json"]
    rollout_completion = manifests["rollout/rollout_completion_manifest.json"]
    if any(
        manifest.get("schema") != SCHEMA_VERSION
        or manifest.get("phase") != phase
        for manifest, phase in (
            (train_request, "train"),
            (train_input, "train"),
            (train_completion, "train"),
            (rollout_input, "rollout"),
            (rollout_completion, "rollout"),
        )
    ):
        return None
    if any(
        manifest.get("iteration") != iteration
        for manifest in (train_request, train_input, rollout_input)
    ):
        return None
    try:
        request_fields_match = all(
            key in train_input
            and canonical_bytes(train_input[key]) == canonical_bytes(value)
            for key, value in train_request.items()
        )
    except RunManifestError:
        return None
    try:
        digest_relationships_match = (
            train_input.get("request_manifest_sha256")
            == manifest_sha256(train_request)
            and train_completion.get("input_manifest_sha256")
            == manifest_sha256(train_input)
            and rollout_completion.get("input_manifest_sha256")
            == manifest_sha256(rollout_input)
        )
    except RunManifestError:
        return None
    if not request_fields_match or not digest_relationships_match:
        return None

    train_outputs = train_completion.get("outputs")
    rollout_outputs = rollout_completion.get("outputs")
    rollout_paths = [
        directory / "rollout" / "rollout.mp4",
        directory / "rollout" / "trajectory.npz",
        directory / "rollout" / "behavior.json",
    ]
    if (
        not isinstance(train_outputs, list)
        or len(train_outputs) != 1
        or not _identity_matches_exact_path(
            train_outputs[0], checkpoint_resolved,
        )
        or not isinstance(rollout_outputs, list)
        or len(rollout_outputs) != len(rollout_paths)
        or any(
            not _identity_matches_exact_path(identity, expected)
            for identity, expected in zip(rollout_outputs, rollout_paths)
        )
        or not _identity_matches_exact_path(
            rollout_input.get("checkpoint"), checkpoint_resolved,
        )
    ):
        return None

    initialization = train_request.get("initialization")
    effective_initialization = train_input.get("effective_initialization")
    identities = [train_request.get("reward"), rollout_input.get("reward")]
    if isinstance(initialization, dict):
        identities.append(initialization.get("policy"))
    else:
        return None
    if isinstance(effective_initialization, dict):
        identities.append(effective_initialization.get("policy"))
    else:
        return None
    if not all(_manifest_identity_is_current(identity) for identity in identities):
        return None

    return {
        "schema": 3,
        "iter_index": iteration,
        "marker_sha256": marker_receipt["sha256"],
        "checkpoint": checkpoint_resolved.name,
        "checkpoint_sha256": expected_checkpoint_sha,
        "checkpoint_bytes": expected_checkpoint_bytes,
        "phase_manifests": phase_receipts,
        "phase_manifests_sha256": manifest_sha256(phase_receipts),
    }


__all__ = [
    "RunManifestError",
    "adapter_identity",
    "build_completion_manifest",
    "build_rollout_input_manifest",
    "build_train_input_manifest",
    "completion_manifest_matches",
    "file_identity",
    "file_identity_matches",
    "input_manifest_matches",
    "manifest_sha256",
    "read_json_object",
    "software_identity",
    "verify_iteration_completion_marker",
    "write_json_atomic",
]
