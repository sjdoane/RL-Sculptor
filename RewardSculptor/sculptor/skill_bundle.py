"""Safe import of data-only RewardSculptor starting-skill bundles.

The archive is an interchange envelope, never an executable package.  In
particular, uploaded ``.pt``/TorchScript files are not deserialized.  A
trainable import must carry an allowlisted safetensors state-dict export; the
importer reconstructs a new server-owned native checkpoint from those tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from sculptor.skill_library import SkillLibrary, SkillRecord
from sculptor.tierd_tracker_policy import (
    TIERD_TRACKER_POLICY_ORIGIN_KIND,
    TIERD_TRACKER_POLICY_ORIGIN_MEMBER,
    TierDTrackerPolicyError,
    validate_tierd_tracker_policy_origin,
)


# Portable starting skills and deployment packages are deliberately different
# artifact types.  A deployment ZIP contains runnable Python/TorchScript and a
# raw recovery checkpoint; none of those bytes belongs at this hostile-upload
# boundary.  Keeping a distinct kind prevents a renamed deployment ZIP from
# becoming an importable `.rskill` by accident.
BUNDLE_KIND = "reward-sculptor-starting-skill"
DEPLOYMENT_BUNDLE_KIND = "reward-sculptor-deployment-bundle"
LEGACY_DEPLOYMENT_BUNDLE_KIND = "reward-sculptor-policy-bundle"
SUPPORTED_BUNDLE_SCHEMAS = {2, 3}
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_MEMBERS = 256
MAX_EXPANDED_BYTES = 4 * 1024**3
MAX_MANIFEST_BYTES = 2 * 1024**2
MAX_TENSOR_ELEMENTS = 400_000_000
MAX_TENSORS = 4096
MAX_TENSOR_KEY_CHARS = 512
MAX_SINGLE_FILE_BYTES = 2 * 1024**3
MAX_REFERENCE_ARCHIVE_BYTES = 512 * 1024**2
MAX_REFERENCE_EXPANDED_BYTES = 1024**3
MAX_REFERENCE_MEMBERS = 128
_REFERENCE_TRANSACTION_LOCK_TIMEOUT_S = 10.0
_REFERENCE_TRANSACTION_LOCK_FILENAME = ".rskill-import.lock"
_SAFE_TENSOR_GROUPS = {
    "actor_state_dict",
    "critic_state_dict",
    "actor_obs_normalizer_state_dict",
    "critic_obs_normalizer_state_dict",
}
_NORMALIZER_TENSOR_SPECS = {
    "_mean": ("torch.float32", "vector"),
    "_var": ("torch.float32", "vector"),
    "_std": ("torch.float32", "vector"),
    "count": ("torch.int64", "scalar"),
}
_STOCHASTIC_POLICY_TENSOR = "distribution.std_param"
_CONTROLLER_KINDS = {
    "reference_tracker",
    "residual_policy",
    "hierarchical_mode_controller",
}
_SAFE_ROBOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# The portable envelope is a closed, data-only format.  Do not broaden this
# list merely because a new file is useful in a deployment package: every new
# member needs its own parser, bounds, and negative tests here first.
_PORTABLE_MEMBER_LIMITS = {
    "manifest.json": MAX_MANIFEST_BYTES,
    "policy/weights.safetensors": MAX_SINGLE_FILE_BYTES,
    "motion/clip.npz": MAX_REFERENCE_ARCHIVE_BYTES,
    "motion/provenance.json": MAX_MANIFEST_BYTES,
    # `reference/` is the schema-v2 compatibility spelling used by early
    # producers.  New exports use `motion/`.
    "reference/clip.npz": MAX_REFERENCE_ARCHIVE_BYTES,
    "reference/provenance.json": MAX_MANIFEST_BYTES,
    "controller/controller.json": 256 * 1024,
    "world/manifest.json": MAX_MANIFEST_BYTES,
    "provenance/origin_policy_contract.json": MAX_MANIFEST_BYTES,
    "provenance/origin_job.log": 256 * 1024**2,
    "provenance/source_config.toml": MAX_MANIFEST_BYTES,
    "provenance/selection_source.json": MAX_MANIFEST_BYTES,
    "provenance/selection_observed.json": MAX_MANIFEST_BYTES,
    TIERD_TRACKER_POLICY_ORIGIN_MEMBER: 64 * 1024**2,
}
_PORTABLE_DIRECTORIES = {
    "policy", "motion", "reference", "controller", "world", "provenance",
}


class SkillBundleError(ValueError):
    """A user-actionable bundle admission failure."""

    def __init__(self, message: str, *, code: str = "invalid_bundle") -> None:
        super().__init__(message)
        self.code = code


class _StrictJSONError(ValueError):
    """Internal parse failure normalized to :class:`SkillBundleError`."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise _StrictJSONError(f"non-finite JSON number {value!r} is forbidden")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _StrictJSONError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _load_strict_json(payload: bytes | str, *, label: str) -> Any:
    """Parse authoritative upload JSON without Python's permissive extensions.

    Duplicate keys make a signed/hashed document ambiguous across parsers, and
    ``NaN``/``Infinity`` are not JSON values.  Reject both recursively for every
    JSON member admitted by the portable envelope.
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJSONError) as exc:
        raise SkillBundleError(
            f"{label} is not valid UTF-8 JSON: {exc}",
        ) from exc


def _validated_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SkillBundleError(f"{label} must be a 64-character hexadecimal SHA-256")
    return value.lower()


@dataclass(frozen=True)
class ImportTarget:
    adapter_class: str
    task_id: str
    robot_slug: Optional[str] = None
    compatibility_contract: Optional[dict[str, Any]] = None
    policy_contract_error: Optional[str] = None
    # Listing/import remains useful while a project is still being authored,
    # but an unknown embodiment can never make a policy or motion structurally
    # selectable.  Keep this separate from ``policy_contract_error`` because
    # it also blocks reference-only initialization modes.
    robot_contract_error: Optional[str] = None


@dataclass(frozen=True)
class ImportedStartingSkill:
    record: SkillRecord
    receipt: dict[str, Any]


def reference_source_provenance_sha256(provenance: dict[str, Any]) -> str:
    """Identity of immutable source provenance, excluding local certification.

    ``tier`` and ``tierD`` are derived, locally mutable admission overlays: a
    newly imported Tier-K candidate necessarily changes both when it earns a
    target-specific Tier-D certificate.  They therefore cannot be part of the
    bundle/source identity used for collision checks or launch revalidation.
    Clip bytes and Tier-D artifacts retain their own exact digest pins.
    """
    if not isinstance(provenance, dict):
        raise SkillBundleError("reference provenance must be an object")
    immutable = dict(provenance)
    immutable.pop("tier", None)
    immutable.pop("tierD", None)
    payload = {
        "schema": "rewardsculptor.reference-source-provenance.v1",
        "provenance": immutable,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise SkillBundleError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SkillBundleError(f"unsafe archive member path: {name!r}")
    if ":" in path.parts[0]:
        raise SkillBundleError(f"drive-qualified archive path is forbidden: {name!r}")
    return path.as_posix()


def _validate_member_type(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise SkillBundleError(f"encrypted archive member is forbidden: {info.filename}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode:
        kind = stat.S_IFMT(unix_mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise SkillBundleError(
                f"symlink or special archive member is forbidden: {info.filename}"
            )


def _validate_portable_member_name(name: str, *, is_dir: bool) -> None:
    """Enforce the closed data-only `.rskill` member vocabulary.

    ZIP filename extensions are not a security boundary.  An exact allowlist
    is: it rejects raw checkpoints, pickle, Python, TorchScript, ONNX, native
    libraries, and future unknown formats before any component parser runs.
    """
    if is_dir:
        if name not in _PORTABLE_DIRECTORIES:
            raise SkillBundleError(
                f"unsupported portable bundle directory: {name!r}",
                code="unsupported_member",
            )
        return
    if name not in _PORTABLE_MEMBER_LIMITS:
        raise SkillBundleError(
            "unsupported portable bundle member "
            f"{name!r}; .rskill accepts only bounded declarative data",
            code="unsupported_member",
        )


def _validate_portable_member_set(member_names: set[str]) -> None:
    motion = {
        "motion/clip.npz", "motion/provenance.json",
    } & member_names
    legacy_reference = {
        "reference/clip.npz", "reference/provenance.json",
    } & member_names
    if motion and legacy_reference:
        raise SkillBundleError(
            "bundle cannot mix motion/ and reference/ trajectory spellings",
            code="unsupported_member",
        )


def _read_member_limited(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int,
) -> bytes:
    if info.file_size > limit:
        raise SkillBundleError(
            f"archive member {info.filename!r} exceeds {limit} bytes"
        )
    out = bytearray()
    with archive.open(info, "r") as source:
        while True:
            chunk = source.read(min(1 << 20, limit + 1 - len(out)))
            if not chunk:
                break
            out.extend(chunk)
            if len(out) > limit:
                raise SkillBundleError(
                    f"archive member {info.filename!r} exceeds {limit} bytes"
                )
    return bytes(out)


def _copy_member_verified(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with archive.open(info, "r") as source, destination.open("wb") as output:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            if written > expected_size:
                raise SkillBundleError(
                    f"descriptor size mismatch for {info.filename!r}",
                    code="descriptor_mismatch",
                )
        output.flush()
        os.fsync(output.fileno())
    if written != expected_size or digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise SkillBundleError(
            f"descriptor digest/size mismatch for {info.filename!r}",
            code="descriptor_mismatch",
        )


def _verify_member_descriptor(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    """Stream-attest an archive member without extracting or evaluating it."""
    digest = hashlib.sha256()
    read = 0
    with archive.open(info, "r") as source:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            read += len(chunk)
            if read > expected_size:
                raise SkillBundleError(
                    f"descriptor size mismatch for {info.filename!r}",
                    code="descriptor_mismatch",
                )
            digest.update(chunk)
    if read != expected_size or digest.hexdigest() != expected_sha256:
        raise SkillBundleError(
            f"descriptor digest/size mismatch for {info.filename!r}",
            code="descriptor_mismatch",
        )


def _parse_descriptors(
    manifest: dict[str, Any], member_names: set[str],
) -> dict[str, tuple[str, int]]:
    raw = manifest.get("files")
    if not isinstance(raw, list) or not raw:
        raise SkillBundleError("manifest.files must be a non-empty descriptor list")
    descriptors: dict[str, tuple[str, int]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise SkillBundleError("every manifest.files entry must be an object")
        path = _safe_member_name(entry.get("path"))
        sha = entry.get("sha256")
        size = entry.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise SkillBundleError(f"invalid descriptor for {path!r}")
        sha = _validated_sha256(
            sha, label=f"manifest descriptor sha256 for {path!r}",
        )
        if path in descriptors:
            raise SkillBundleError(f"duplicate manifest descriptor: {path!r}")
        descriptors[path] = (sha, size)
    described = set(descriptors)
    actual = member_names - {"manifest.json"}
    if described != actual:
        missing = sorted(actual - described)
        extra = sorted(described - actual)
        raise SkillBundleError(
            "manifest descriptor set does not match archive members "
            f"(missing={missing}, extra={extra})",
            code="descriptor_mismatch",
        )
    return descriptors


def _safetensors_spec(
    manifest: dict[str, Any], member_names: set[str],
) -> tuple[str, list[str]]:
    starting = manifest.get("starting_skill") or {}
    if not isinstance(starting, dict):
        raise SkillBundleError("manifest.starting_skill must be an object")
    file_name = starting.get("weights_file")
    if file_name is None:
        network = manifest.get("network") or {}
        exported = network.get("trainable_checkpoint") if isinstance(network, dict) else None
        if isinstance(exported, dict):
            file_name = exported.get("file")
            if "policy_roles" not in starting:
                starting = {**starting, "policy_roles": exported.get("policy_roles")}
    if file_name is None and "policy/weights.safetensors" in member_names:
        file_name = "policy/weights.safetensors"
    if not isinstance(file_name, str) or not file_name.endswith(".safetensors"):
        raise SkillBundleError(
            "trainable imports require starting_skill.weights_file pointing "
            "to a .safetensors member; uploaded .pt pickle checkpoints are "
            "never deserialized",
            code="safetensors_required",
        )
    file_name = _safe_member_name(file_name)
    if file_name != "policy/weights.safetensors":
        raise SkillBundleError(
            "trainable weights must use the canonical "
            "policy/weights.safetensors member path",
            code="safetensors_required",
        )
    if file_name not in member_names:
        raise SkillBundleError(f"declared safetensors file is missing: {file_name}")
    roles = starting.get("policy_roles") or []
    if not isinstance(roles, list) or any(r not in ("actor", "critic") for r in roles):
        raise SkillBundleError("starting_skill.policy_roles must contain actor/critic")
    return file_name, list(dict.fromkeys(str(role) for role in roles))


def _declares_trainable_policy(
    manifest: dict[str, Any], member_names: set[str],
) -> bool:
    starting = manifest.get("starting_skill") or {}
    if isinstance(starting, dict) and starting.get("weights_file") is not None:
        return True
    network = manifest.get("network") or {}
    if isinstance(network, dict) and network.get("trainable_checkpoint") is not None:
        return True
    return "policy/weights.safetensors" in member_names


def _native_checkpoint_from_safetensors(
    source: Path, destination: Path, compatibility_contract: dict[str, Any],
) -> tuple[list[str], str]:
    """Convert allowlisted tensor maps without evaluating uploaded code."""
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - packaging regression
        raise SkillBundleError(
            "safetensors support is unavailable in this installation"
        ) from exc

    grouped: dict[str, dict[str, Any]] = {}
    signature_rows: list[dict[str, Any]] = []
    total_elements = 0
    tensor_count = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "reward-sculptor-rsl-rl-v1":
            raise SkillBundleError(
                "unsupported safetensors metadata format; expected "
                "reward-sculptor-rsl-rl-v1"
            )
        for full_key in handle.keys():
            tensor_count += 1
            if tensor_count > MAX_TENSORS:
                raise SkillBundleError(
                    f"bundle has more than {MAX_TENSORS} policy tensors"
                )
            if len(full_key) > MAX_TENSOR_KEY_CHARS:
                raise SkillBundleError("starting-skill tensor key is too long")
            if "::" not in full_key:
                raise SkillBundleError(f"invalid starting-skill tensor key: {full_key!r}")
            group, key = full_key.split("::", 1)
            if group not in _SAFE_TENSOR_GROUPS or not key:
                raise SkillBundleError(f"unallowlisted starting-skill tensor: {full_key!r}")
            tensor = handle.get_tensor(full_key)
            if int(tensor.numel()) <= 0:
                raise SkillBundleError(f"tensor {full_key!r} is empty")
            if not tensor.is_floating_point() and "count" not in key.lower():
                raise SkillBundleError(
                    f"policy tensor {full_key!r} must use a floating dtype"
                )
            total_elements += int(tensor.numel())
            if total_elements > MAX_TENSOR_ELEMENTS:
                raise SkillBundleError(
                    f"bundle tensors exceed {MAX_TENSOR_ELEMENTS} elements"
                )
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise SkillBundleError(f"tensor {full_key!r} contains NaN/Inf")
            grouped.setdefault(group, {})[key] = tensor.contiguous().cpu()
            signature_rows.append({
                "key": full_key,
                "dtype": str(tensor.dtype),
                "shape": [int(v) for v in tensor.shape],
            })

    actor = grouped.get("actor_state_dict")
    if not actor:
        raise SkillBundleError("safetensors payload has no actor_state_dict tensors")
    payload: dict[str, Any] = {"actor_state_dict": actor}
    for optional in sorted(_SAFE_TENSOR_GROUPS - {"actor_state_dict"}):
        if grouped.get(optional):
            payload[optional] = grouped[optional]
    payload["infos"] = {
        "source": "sanitized_safetensors",
        "format": "reward-sculptor-rsl-rl-v1",
    }
    _validate_tensor_contract(payload, compatibility_contract)
    # Safetensors keeps normalizers in explicit allowlisted groups so their
    # inventory can be validated independently.  rsl_rl's server-owned
    # MLPModel owns those buffers under ``obs_normalizer.*``; materialize that
    # exact native state only after the untrusted inventory has passed.
    for role in ("actor", "critic"):
        normalizer_group = f"{role}_obs_normalizer_state_dict"
        normalizer = payload.pop(normalizer_group, None)
        if not normalizer:
            continue
        state = payload[f"{role}_state_dict"]
        state.update({f"obs_normalizer.{key}": value for key, value in normalizer.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    roles = [
        role
        for role, group in (
            ("actor", "actor_state_dict"),
            ("critic", "critic_state_dict"),
        )
        if grouped.get(group)
    ]
    signature = hashlib.sha256(
        json.dumps(
            sorted(signature_rows, key=lambda row: row["key"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return roles, signature


def _validate_tensor_contract(
    payload: dict[str, Any], contract: dict[str, Any],
) -> None:
    """Prove an exact rsl_rl-v5 MLP state inventory.

    Shape-only matching is insufficient for a trainable warm start: rsl_rl
    loads these maps strictly, and the actor's learned Gaussian standard
    deviation is part of the policy.  This validator therefore recognizes one
    complete server-owned architecture, rejects every surplus key, and checks
    exact dtypes/ranks/shapes before a native checkpoint can be written.
    """

    observations = contract.get("observations") or {}
    actions = contract.get("actions") or {}
    policy = contract.get("policy") or {}
    try:
        obs_dim = int(observations["shape"][0])
        critic_obs_dim = int(
            (observations.get("critic_shape") or observations["shape"])[0]
        )
        action_dim = int(actions["shape"][0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SkillBundleError("compatibility contract has invalid tensor shapes") from exc

    def mlp_specs(
        widths: list[int], *, stochastic: bool,
    ) -> dict[str, tuple[str, tuple[int, ...]]]:
        specs: dict[str, tuple[str, tuple[int, ...]]] = {}
        for layer, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
            index = layer * 2
            specs[f"mlp.{index}.weight"] = (
                "torch.float32", (width_out, width_in),
            )
            specs[f"mlp.{index}.bias"] = ("torch.float32", (width_out,))
        if stochastic:
            specs[_STOCHASTIC_POLICY_TENSOR] = (
                "torch.float32", (action_dim,),
            )
        return specs

    def normalizer_specs(width: int) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {
            key: (
                dtype,
                (1, width) if shape_kind == "vector" else (),
            )
            for key, (dtype, shape_kind) in _NORMALIZER_TENSOR_SPECS.items()
        }

    def embedded_normalizer_specs(
        width: int,
    ) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {
            f"obs_normalizer.{key}": spec
            for key, spec in normalizer_specs(width).items()
        }

    def validate_inventory(
        group: str,
        state: Any,
        expected: dict[str, tuple[str, tuple[int, ...]]],
    ) -> None:
        if not isinstance(state, dict):
            raise SkillBundleError(f"safetensors payload has no {group} tensors")
        actual = set(state)
        required = set(expected)
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        if missing or extra:
            raise SkillBundleError(
                f"{group} tensor inventory mismatch "
                f"(missing={missing}, extra={extra})"
            )
        for key, (expected_dtype, expected_shape) in expected.items():
            tensor = state[key]
            actual_dtype = str(getattr(tensor, "dtype", None))
            actual_shape = tuple(int(value) for value in tensor.shape)
            if actual_dtype != expected_dtype:
                raise SkillBundleError(
                    f"{group}::{key} dtype {actual_dtype} does not match "
                    f"supported dtype {expected_dtype}"
                )
            if actual_shape != expected_shape:
                raise SkillBundleError(
                    f"{group}::{key} shape {actual_shape} does not match "
                    f"supported shape {expected_shape}"
                )

    def validate_role(
        role: str,
        *,
        widths: list[int],
        stochastic: bool,
        expects_normalizer: bool,
    ) -> None:
        state_group = f"{role}_state_dict"
        normalizer_group = f"{role}_obs_normalizer_state_dict"
        state = payload.get(state_group)
        separate_normalizer = payload.get(normalizer_group)
        if state is None:
            if separate_normalizer is not None:
                raise SkillBundleError(
                    f"orphan {normalizer_group} without {state_group}"
                )
            return

        base_specs = mlp_specs(widths, stochastic=stochastic)
        embedded_specs = embedded_normalizer_specs(widths[0])
        has_embedded = any(
            str(key).startswith("obs_normalizer.") for key in state
        )
        has_separate = separate_normalizer is not None
        if expects_normalizer:
            if has_embedded == has_separate:
                raise SkillBundleError(
                    f"{role} observation normalizer must use exactly one "
                    "complete representation"
                )
            if has_embedded:
                base_specs.update(embedded_specs)
            else:
                validate_inventory(
                    normalizer_group,
                    separate_normalizer,
                    normalizer_specs(widths[0]),
                )
        elif has_embedded or has_separate:
            raise SkillBundleError(
                f"unexpected {role} observation normalizer tensors"
            )
        validate_inventory(state_group, state, base_specs)

    actor_cfg = policy.get("actor") or {}
    recurrent = actor_cfg.get("recurrent") or {}
    if recurrent.get("type") not in (None, "", "none"):
        raise SkillBundleError(
            "recurrent starting policies are not admitted by the v1 "
            "safetensors converter"
        )
    expected_hidden = [int(v) for v in (actor_cfg.get("hidden_dims") or [])]
    critic_state = payload.get("critic_state_dict")
    critic_cfg = policy.get("critic") or {}
    critic_recurrent = critic_cfg.get("recurrent") or {}
    if critic_state and critic_recurrent.get("type") not in (None, "", "none"):
        raise SkillBundleError(
            "recurrent starting critics are not admitted by the v1 "
            "safetensors converter"
        )
    critic_hidden = [int(v) for v in (critic_cfg.get("hidden_dims") or [])]
    normalizer_cfg = policy.get("normalizer") or {}
    expects_actor_normalizer = bool(
        normalizer_cfg.get("actor_present", normalizer_cfg.get("present"))
    )
    expects_critic_normalizer = bool(normalizer_cfg.get("critic_present", False))
    validate_role(
        "actor",
        widths=[obs_dim, *expected_hidden, action_dim],
        stochastic=True,
        expects_normalizer=expects_actor_normalizer,
    )
    validate_role(
        "critic",
        widths=[critic_obs_dim, *critic_hidden, 1],
        stochastic=False,
        expects_normalizer=expects_critic_normalizer,
    )

    actor_std = payload["actor_state_dict"][_STOCHASTIC_POLICY_TENSOR]
    if not bool((actor_std > 0).all()):
        raise SkillBundleError(
            "actor stochastic distribution.std_param must be strictly positive"
        )
    for role in ("actor", "critic"):
        state = payload.get(f"{role}_state_dict") or {}
        normalizer = payload.get(f"{role}_obs_normalizer_state_dict") or {}
        combined = {
            **{
                key.removeprefix("obs_normalizer."): value
                for key, value in state.items()
                if key.startswith("obs_normalizer.")
            },
            **normalizer,
        }
        if combined:
            if not bool((combined["_std"] > 0).all()):
                raise SkillBundleError(
                    f"{role} observation normalizer _std must be positive"
                )
            if not bool((combined["_var"] >= 0).all()):
                raise SkillBundleError(
                    f"{role} observation normalizer _var must be nonnegative"
                )
            if int(combined["count"].item()) < 0:
                raise SkillBundleError(
                    f"{role} observation normalizer count must be nonnegative"
                )


def _validate_reference_npz(path: Path) -> None:
    """Bound the one intentional nested ZIP container before NumPy opens it."""
    if path.stat().st_size > MAX_REFERENCE_ARCHIVE_BYTES:
        raise SkillBundleError("reference clip exceeds the admission size limit")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > MAX_REFERENCE_MEMBERS:
                raise SkillBundleError("reference clip has too many arrays")
            expanded = 0
            folded: set[str] = set()
            for info in infos:
                _validate_member_type(info)
                name = _safe_member_name(info.filename.rstrip("/"))
                if info.is_dir():
                    continue
                if not name.endswith(".npy"):
                    raise SkillBundleError(
                        f"reference clip contains a non-array member: {name!r}"
                    )
                if name.casefold() in folded:
                    raise SkillBundleError(
                        f"reference clip contains colliding members: {name!r}"
                    )
                folded.add(name.casefold())
                expanded += int(info.file_size)
                if expanded > MAX_REFERENCE_EXPANDED_BYTES:
                    raise SkillBundleError(
                        "expanded reference clip exceeds the admission limit"
                    )
    except zipfile.BadZipFile as exc:
        raise SkillBundleError("reference clip is not a valid NPZ container") from exc


@dataclass(frozen=True)
class _PreparedReference:
    clip_id: str
    robot: str
    content_sha256: str
    source_provenance_sha256: str
    clip_path: Path
    provenance_path: Path


@dataclass(frozen=True)
class _ReferenceRegistration:
    prepared: _PreparedReference
    created: bool
    index_path: Optional[Path] = None
    previous_index_bytes: Optional[bytes] = None
    index_previously_existed: bool = False


def _prepare_reference(
    archive: zipfile.ZipFile,
    infos: dict[str, zipfile.ZipInfo],
    descriptors: dict[str, tuple[str, int]],
    workdir: Path,
    *,
    expected_robot: Optional[str] = None,
) -> Optional[_PreparedReference]:
    """Validate and stage a bundled trajectory without mutating libraries."""
    pairs = (
        ("motion/clip.npz", "motion/provenance.json"),
        ("reference/clip.npz", "reference/provenance.json"),
    )
    selected = next(
        ((clip, prov) for clip, prov in pairs if clip in infos or prov in infos),
        None,
    )
    if selected is None:
        return None
    clip_name, prov_name = selected
    if clip_name not in infos or prov_name not in infos:
        raise SkillBundleError("a bundled reference requires clip.npz and provenance.json")
    clip_tmp = workdir / "reference" / "clip.npz"
    prov_tmp = workdir / "reference" / "provenance.json"
    for name, dest in ((clip_name, clip_tmp), (prov_name, prov_tmp)):
        sha, size = descriptors[name]
        _copy_member_verified(
            archive, infos[name], dest, expected_sha256=sha, expected_size=size
        )

    from sculptor import reference
    from sculptor.refs import library as refs

    try:
        _validate_reference_npz(clip_tmp)
        reference.load_clip(clip_tmp)
        provenance = _load_strict_json(
            prov_tmp.read_bytes(), label=f"{prov_name}",
        )
    except Exception as exc:
        raise SkillBundleError(f"invalid bundled reference: {exc}") from exc
    if not isinstance(provenance, dict):
        raise SkillBundleError("invalid reference provenance: root must be an object")
    clip_id = provenance.get("clip_id")
    robot = provenance.get("robot")
    if not isinstance(clip_id, str):
        raise SkillBundleError("reference provenance clip_id must be a string")
    try:
        refs.validate_clip_id(clip_id)
    except ValueError as exc:
        raise SkillBundleError(f"invalid reference provenance: {exc}") from exc
    if not isinstance(robot, str):
        raise SkillBundleError("reference provenance robot must be a string")
    if not _SAFE_ROBOT_RE.fullmatch(robot):
        raise SkillBundleError(
            "reference provenance robot must be a safe stable identifier"
        )
    errors = refs.validate_provenance(provenance)
    if errors:
        raise SkillBundleError("invalid reference provenance: " + "; ".join(errors))
    if expected_robot and robot != expected_robot:
        raise SkillBundleError(
            "bundle robot_slug conflicts with reference provenance "
            f"({expected_robot!r} != {robot!r})",
            code="robot_identity_mismatch",
        )
    expected_content = _validated_sha256(
        provenance["content_sha256"],
        label="reference provenance content_sha256",
    )
    actual_content = _sha256_file(clip_tmp)
    if expected_content != actual_content:
        raise SkillBundleError("reference provenance content_sha256 mismatch")
    canonical_provenance_sha256 = reference_source_provenance_sha256(
        provenance,
    )
    return _PreparedReference(
        clip_id=clip_id,
        robot=robot,
        content_sha256=actual_content,
        source_provenance_sha256=canonical_provenance_sha256,
        clip_path=clip_tmp,
        provenance_path=prov_tmp,
    )


def _restore_reference_index(registration: _ReferenceRegistration) -> None:
    index_path = registration.index_path
    if index_path is None:
        return
    if not registration.index_previously_existed:
        index_path.unlink(missing_ok=True)
        return
    assert registration.previous_index_bytes is not None
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.rollback.", dir=index_path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(registration.previous_index_bytes)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, index_path)
    finally:
        tmp.unlink(missing_ok=True)


def _rollback_reference_registration(
    registration: Optional[_ReferenceRegistration],
) -> None:
    if registration is None or not registration.created:
        return
    from sculptor.refs import library as refs

    prepared = registration.prepared
    destination = refs.clip_dir(prepared.robot, prepared.clip_id)
    if destination.exists():
        shutil.rmtree(destination)
    _restore_reference_index(registration)
    # Avoid leaving empty robot/root directories after a rejected transaction.
    for candidate in (destination.parent, destination.parent.parent):
        try:
            candidate.rmdir()
        except OSError:
            break


@contextmanager
def _reference_import_transaction_lock() -> Iterator[None]:
    """Serialize an ``.rskill`` reference install through commit or rollback.

    The reference index is a rebuildable, library-wide cache.  A per-clip lock
    is therefore insufficient: without this transaction lock, import A can
    snapshot the index, import B can commit a different clip, and A can then
    restore its stale snapshot while rolling back.  Keep this lock outside the
    skill-library publish lock and always acquire it first; no code in the
    skill library acquires this reference lock, so the ordering cannot cycle.
    """
    from filelock import FileLock, Timeout as _FileLockTimeout
    from sculptor.refs import library as refs

    root = refs.references_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SkillBundleError(
            f"could not prepare reference library transaction lock: {exc}",
            code="reference_library_unavailable",
        ) from exc
    lock_path = root / _REFERENCE_TRANSACTION_LOCK_FILENAME
    lock = FileLock(
        str(lock_path), timeout=_REFERENCE_TRANSACTION_LOCK_TIMEOUT_S,
    )
    try:
        lock.acquire()
    except _FileLockTimeout as exc:
        raise SkillBundleError(
            "reference library is busy with another starting-skill import; "
            "retry after that import completes",
            code="reference_library_busy",
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _install_reference(prepared: _PreparedReference) -> _ReferenceRegistration:
    """Install staged reference bytes and retain exact rollback state."""
    from sculptor.refs import library as refs

    destination = refs.clip_dir(prepared.robot, prepared.clip_id)
    existing_clip = destination / refs.CLIP_FILENAME
    if destination.exists():
        existing_provenance = destination / refs.PROVENANCE_FILENAME
        try:
            existing_provenance_data = _load_strict_json(
                existing_provenance.read_bytes(),
                label="existing reference provenance",
            )
            existing_provenance_sha256 = reference_source_provenance_sha256(
                existing_provenance_data,
            )
        except (OSError, SkillBundleError, TypeError, ValueError):
            existing_provenance_sha256 = None
        if (
            existing_clip.is_file()
            and _sha256_file(existing_clip) == prepared.content_sha256
            and existing_provenance_sha256
            == prepared.source_provenance_sha256
        ):
            return _ReferenceRegistration(prepared=prepared, created=False)
        raise SkillBundleError(
            f"reference id collision for {prepared.robot}/{prepared.clip_id}; "
            "existing clip or "
            "canonical provenance identity differs"
        )
    index_path = refs.index_path()
    index_existed = index_path.is_file()
    previous_index = index_path.read_bytes() if index_existed else None
    registration = _ReferenceRegistration(
        prepared=prepared,
        created=True,
        index_path=index_path,
        previous_index_bytes=previous_index,
        index_previously_existed=index_existed,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{prepared.clip_id}.", dir=destination.parent,
    ))
    installed = False
    try:
        shutil.copy2(prepared.clip_path, stage / refs.CLIP_FILENAME)
        shutil.copy2(prepared.provenance_path, stage / refs.PROVENANCE_FILENAME)
        os.replace(stage, destination)
        installed = True
        # The caller holds the same library-wide transaction lock used by
        # certification and manual rebuilds.  Use the lock-owned primitive to
        # avoid recursively acquiring a second FileLock instance.
        refs._rebuild_index_unlocked()
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        if installed:
            _rollback_reference_registration(registration)
        if isinstance(exc, SkillBundleError):
            raise
        raise SkillBundleError(
            f"could not atomically register reference "
            f"{prepared.robot}/{prepared.clip_id}: {exc}",
        ) from exc
    return registration


def _policy_migration_observation_label(
    migration: dict[str, Any] | None,
) -> str:
    """Plain-language name for the exact admitted interface transform."""
    migration_type = migration.get("type") if migration else None
    return {
        "zero_initialized_event_phase_observation": (
            "event-phase observation extension"
        ),
        "zero_initialized_reference_clock_observation": (
            "reference-clock observation extension"
        ),
        "zero_initialized_observation_extensions": (
            "event-phase and reference-clock observation extensions"
        ),
    }.get(migration_type, "observation interface extension")


def compatibility_for(record: SkillRecord, target: ImportTarget) -> dict[str, Any]:
    policy_reasons: list[str] = []
    policy_migration: dict[str, Any] | None = None
    reason_codes: list[str] = []
    if target.robot_contract_error:
        policy_reasons.append(target.robot_contract_error)
        reason_codes.append("project_robot_unresolved")
    if record.policy_roles:
        if record.adapter_class != target.adapter_class:
            policy_reasons.append(
                f"adapter differs ({record.adapter_class} != {target.adapter_class})"
            )
        if record.task_id != target.task_id:
            policy_reasons.append(
                f"task differs ({record.task_id} != {target.task_id})"
            )
        if (
            target.robot_slug
            and record.robot_slug
            and record.robot_slug != target.robot_slug
        ):
            policy_reasons.append(
                f"robot differs ({record.robot_slug} != {target.robot_slug})"
            )
        if record.source == "imported_bundle" and (
            not record.tensor_contract_verified
            or not record.tensor_signature_sha256
        ):
            policy_reasons.append(
                "imported policy tensors were not verified against the "
                "compatibility contract"
            )
        if target.compatibility_contract is None:
            policy_reasons.append(
                target.policy_contract_error
                or "project_contract_missing: project target compatibility "
                "contract is unavailable"
            )
        else:
            from sculptor.policy_contract import (
                compare_policy_contracts,
                policy_contract_migration,
            )

            policy_migration = policy_contract_migration(
                record.compatibility_contract,
                target.compatibility_contract,
            )
            policy_reasons.extend(compare_policy_contracts(
                record.compatibility_contract, target.compatibility_contract,
            ))
    else:
        policy_reasons.append("skill has no admitted trainable policy")

    reference_reasons: list[str] = []
    if target.robot_contract_error:
        reference_reasons.append(target.robot_contract_error)
    if not (record.reference_clip_id and record.reference_robot):
        reference_reasons.append("skill has no complete reference trajectory")
    elif (
        target.robot_slug
        and record.reference_robot
        and record.reference_robot != target.robot_slug
    ):
        reference_reasons.append(
            f"reference robot differs ({record.reference_robot} != "
            f"{target.robot_slug})"
        )

    mode_reasons: dict[str, list[str]] = {}
    allowed: list[str] = []
    for mode in record.initialization_modes:
        failures = list(
            reference_reasons if mode == "reference_only" else policy_reasons
        )
        if (
            policy_migration is not None
            and mode not in {"actor_only", "actor_critic", "reference_only"}
        ):
            failures.append(
                "migration for "
                f"{_policy_migration_observation_label(policy_migration)} "
                "supports actor or "
                "actor+critic initialization only, not full resume"
            )
        mode_reasons[mode] = list(dict.fromkeys(failures))
        if not failures:
            allowed.append(mode)
    all_reasons = list(dict.fromkeys(
        reason
        for failures in mode_reasons.values()
        for reason in failures
    ))
    reasons = [] if allowed else all_reasons
    if not allowed:
        status = "incompatible"
    elif len(allowed) != len(record.initialization_modes):
        status = "partially_compatible"
    else:
        status = record.compatibility_status
    return {
        "status": status,
        "allowed_initialization_modes": allowed,
        "reasons": reasons,
        "reason_codes": list(dict.fromkeys(reason_codes)) if reasons else [],
        "mode_reasons": mode_reasons,
        "policy_contract_migration": policy_migration,
    }


def _authorization_for(
    record: SkillRecord,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    """Describe remaining runtime proof without granting it at import time."""
    mode_gates: dict[str, list[str]] = {}
    migration_label = _policy_migration_observation_label(
        compatibility.get("policy_contract_migration")
    )
    for mode in compatibility["allowed_initialization_modes"]:
        if mode == "reference_only":
            mode_gates[mode] = [
                (
                    "complete a separate Tier-D exact-schedule tracking evidence job "
                    "for the exact clip and target execution boundary before live "
                    "launch"
                ),
                (
                    "re-attest the exact clip, immutable provenance, rollout, "
                    "certificate, execution contract, and boundary at launch"
                ),
                "observe the worker executing the exact admitted reference",
            ]
        elif mode == "actor_only":
            mode_gates[mode] = [
                "revalidate the current project policy contract at launch",
                "re-attest the server-owned checkpoint full SHA-256",
                "observe warm_start_loaded with the exact digest and actor role",
            ]
            if compatibility.get("policy_contract_migration"):
                mode_gates[mode].insert(
                    1,
                    "materialize the declared zero-initialized "
                    f"{migration_label} and record its digest",
                )
        elif mode == "actor_critic":
            mode_gates[mode] = [
                "revalidate the current project policy contract at launch",
                "re-attest the server-owned checkpoint full SHA-256",
                "observe warm_start_loaded with exact actor and critic roles",
            ]
            if compatibility.get("policy_contract_migration"):
                mode_gates[mode].insert(
                    1,
                    "materialize the declared zero-initialized "
                    f"{migration_label} and record its digest",
                )
        elif mode == "full_resume":
            mode_gates[mode] = [
                "verify an exact optimizer and complete run-state resume contract",
                "observe the worker restoring every admitted resume component",
            ]
    selectable = not compatibility["reasons"]
    detail = "No initialization mode currently passes structural admission."
    if selectable:
        detail = (
            "Import/list admission makes this starting point selectable, not "
            "trainable."
        )
        if "reference_only" in compatibility["allowed_initialization_modes"]:
            detail += (
                " A reference requires a separate Tier-D exact-schedule tracking "
                "evidence job before live launch; launch only re-verifies "
                "that existing evidence."
            )
        if any(
            mode != "reference_only"
            for mode in compatibility["allowed_initialization_modes"]
        ):
            detail += (
                " Policy modes still require their listed launch and worker proofs."
            )
    return {
        "status": "candidate" if selectable else "blocked",
        "receipt_scope": "structural_selectability_only",
        "training_authorized": False,
        "mode_gates": mode_gates,
        "detail": detail,
        "policy_present": bool(record.policy_roles),
    }


def receipt_for(record: SkillRecord, target: ImportTarget) -> dict[str, Any]:
    compatibility = compatibility_for(record, target)
    selectable = not compatibility["reasons"]
    excluded = [
        "raw checkpoints, pickle, Python, TorchScript, native binaries, "
        "and unknown members are rejected before admission"
    ]
    return {
        "skill": {
            "skill_id": record.skill_id,
            "alias": record.alias,
            "created_at": record.created_at,
            "adapter_class": record.adapter_class,
            "task_id": record.task_id,
            "robot_slug": record.robot_slug,
            "source": record.source,
            "checkpoint_sha256": record.checkpoint_sha256,
            "checkpoint_size_bytes": record.checkpoint_size_bytes,
            "manifest_digest": record.manifest_digest,
            "identity_digest": record.identity_digest,
            "source_weights_sha256": record.source_weights_sha256,
            "source_training_provenance": record.source_training_provenance,
            "source_training_provenance_sha256": (
                record.source_training_provenance_sha256
            ),
            "reference_clip_id": record.reference_clip_id,
            "reference_robot": record.reference_robot,
            "reference_sha256": record.reference_sha256,
            "reference_provenance_sha256": (
                record.reference_provenance_sha256
            ),
            "reference_dynamics_certificate_sha256": (
                record.reference_dynamics_certificate_sha256
            ),
            "reference_rollout_sha256": record.reference_rollout_sha256,
            "reference_execution_contract_sha256": (
                record.reference_execution_contract_sha256
            ),
            "reference_execution_boundary_sha256": (
                record.reference_execution_boundary_sha256
            ),
            "world_bundle_sha256": record.world_bundle_sha256,
            "active_reward_sha256": record.active_reward_sha256,
            "mode_execution_manifest_digest": (
                record.mode_execution_manifest_digest
            ),
            "world_tuple_hash": record.world_tuple_hash,
            "world_selection_sha256": record.world_selection_sha256,
            "world_artifact_sha256": dict(record.world_artifact_sha256),
            "execution_model": record.execution_model,
            "mode_reuse_supported": record.mode_reuse_supported,
            "controller_sha256": record.controller_sha256,
            "compatibility_contract": record.compatibility_contract,
            "compatibility_contract_digest": record.compatibility_contract_digest,
            "compatibility_contract_provenance": (
                record.compatibility_contract_provenance
            ),
            "compatibility_contract_provenance_digest": (
                record.compatibility_contract_provenance_digest
            ),
            "compatibility_contract_provenance_status": (
                record.compatibility_contract_provenance_status
            ),
            "tensor_contract_verified": record.tensor_contract_verified,
            "tensor_signature_sha256": record.tensor_signature_sha256,
            "initialization_modes": list(record.initialization_modes),
            "policy_roles": list(record.policy_roles),
            "trust_status": record.trust_status,
            "checkpoint_format": record.checkpoint_format,
            "source_format": record.source_format,
        },
        # Legacy structural/selectability alias. It does not authorize a run.
        "compatible": selectable,
        "selectable": selectable,
        "training_authorized": False,
        "authorization": _authorization_for(record, compatibility),
        "compatibility": compatibility,
        "trust": {
            "status": record.trust_status,
            "detail": (
                "Imported tensors were read from safetensors and reserialized "
                "into a new server-owned native checkpoint; uploaded pickle "
                "artifacts were never deserialized."
                if record.source_format == "safetensors"
                else (
                    "Reference bytes and provenance fields were validated; "
                    "the bundle contains no executable policy checkpoint."
                    if not record.policy_roles and record.reference_clip_id
                    else "Checkpoint was produced locally by RewardSculptor."
                )
            ),
            "source_format": record.source_format,
            "checkpoint_format": record.checkpoint_format,
            "manifest_digest": record.manifest_digest,
            "compatibility_contract_digest": record.compatibility_contract_digest,
            "compatibility_contract_provenance_digest": (
                record.compatibility_contract_provenance_digest
            ),
            "compatibility_contract_provenance_status": (
                record.compatibility_contract_provenance_status
            ),
            "tensor_contract_verified": record.tensor_contract_verified,
            "tensor_signature_sha256": record.tensor_signature_sha256,
            "checkpoint_sha256": record.checkpoint_sha256,
            "source_training_provenance_sha256": (
                record.source_training_provenance_sha256
            ),
        },
        "components": {
            "policy_roles": list(record.policy_roles),
            "source_training": (
                {
                    "kind": record.source_training_provenance.get("kind"),
                    "status": "retained_unverified_origin_claim",
                    "authenticity_verified": False,
                    "bytes_retained": True,
                    "activatable": False,
                    "sha256": record.source_training_provenance_sha256,
                    "summary": dict(record.source_training_provenance),
                    "detail": (
                        "The uploaded bundle claims these actor weights came "
                        "from the described Tier-D tracker, and its retained "
                        "hash chain is internally consistent. No trusted "
                        "issuer or local Tier-D re-admission authenticates "
                        "that claim. Its reference, reward, world, controller, "
                        "optimizer, and mode executor remain independently "
                        "selectable and are not activated by this skill."
                    ),
                }
                if record.source_training_provenance is not None
                else None
            ),
            "reference": (
                {
                    "clip_id": record.reference_clip_id,
                    "robot": record.reference_robot,
                    "admission": {
                        "status": "registered_candidate",
                        "structural_checks": [
                            "bounded archive and descriptor digests",
                            "finite trajectory arrays and canonical provenance",
                            "exact source robot identity",
                        ],
                        "training_authorized": False,
                        "next_gate": (
                            "run a separate target-specific Tier-D exact-schedule "
                            "tracking evidence job before live launch; launch only "
                            "re-verifies the resulting exact evidence"
                        ),
                    },
                }
                if record.reference_clip_id and record.reference_robot
                else None
            ),
            "world": (
                {
                    "included": True,
                    "status": "training_provenance_retained",
                    "bytes_retained": True,
                    "activatable": False,
                    "tuple_hash": record.world_tuple_hash,
                    "selection_sha256": record.world_selection_sha256,
                    "artifact_sha256": dict(record.world_artifact_sha256),
                }
                if record.world_tuple_hash
                else {
                    "included": record.bundled_world,
                    "status": (
                        "digest_recorded_bytes_discarded"
                        if record.bundled_world
                        else "absent"
                    ),
                    "bytes_retained": False,
                    "activatable": False,
                    "sha256": record.world_bundle_sha256,
                }
            ),
            "execution_boundary": {
                "model": record.execution_model,
                "policy_warm_start": bool(record.policy_roles),
                "mode_reuse_supported": record.mode_reuse_supported,
                "detail": (
                    "Mission publication retains phase provenance when present, "
                    "but only the policy checkpoint is reusable; reward, world, "
                    "reference, and mode executors are not silently activated."
                ),
            },
            "controller": (
                {
                    "kind": record.controller_kind,
                    "status": "digest_recorded_bytes_discarded",
                    "bytes_retained": False,
                    "activatable": False,
                    "sha256": record.controller_sha256,
                }
                if record.controller_kind
                else None
            ),
            "excluded": excluded,
        },
        "warnings": list(record.import_warnings),
    }


class StartingSkillBundleImporter:
    def __init__(self, library: Optional[SkillLibrary] = None) -> None:
        self.library = library or SkillLibrary()

    def import_archive(
        self,
        archive_path: Path,
        *,
        target: ImportTarget,
        admission_callback: Optional[
            Callable[[ImportedStartingSkill], None]
        ] = None,
    ) -> ImportedStartingSkill:
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise SkillBundleError("uploaded bundle is missing")
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise SkillBundleError(
                f"bundle exceeds the {MAX_ARCHIVE_BYTES}-byte upload limit",
                code="bundle_too_large",
            )
        try:
            archive = zipfile.ZipFile(archive_path, "r")
        except zipfile.BadZipFile as exc:
            raise SkillBundleError("bundle is not a valid ZIP/.rskill archive") from exc

        with archive, tempfile.TemporaryDirectory(prefix="rs_skill_import_") as td:
            infos_list = archive.infolist()
            if len(infos_list) > MAX_MEMBERS:
                raise SkillBundleError(
                    f"bundle has more than {MAX_MEMBERS} members",
                    code="bundle_too_large",
                )
            infos: dict[str, zipfile.ZipInfo] = {}
            folded: set[str] = set()
            expanded = 0
            for info in infos_list:
                _validate_member_type(info)
                name = _safe_member_name(info.filename.rstrip("/"))
                if info.is_dir():
                    _validate_portable_member_name(name, is_dir=True)
                    continue
                if name in infos or name.casefold() in folded:
                    raise SkillBundleError(
                        f"duplicate or case-colliding archive member: {name!r}"
                    )
                _validate_portable_member_name(name, is_dir=False)
                member_limit = _PORTABLE_MEMBER_LIMITS[name]
                if info.file_size > member_limit:
                    raise SkillBundleError(
                        f"archive member {name!r} exceeds {member_limit} bytes",
                        code="bundle_too_large",
                    )
                expanded += info.file_size
                if expanded > MAX_EXPANDED_BYTES:
                    raise SkillBundleError(
                        f"expanded bundle exceeds {MAX_EXPANDED_BYTES} bytes",
                        code="bundle_too_large",
                    )
                infos[name] = info
                folded.add(name.casefold())
            _validate_portable_member_set(set(infos))
            manifest_info = infos.get("manifest.json")
            if manifest_info is None:
                raise SkillBundleError("bundle has no manifest.json")
            manifest_bytes = _read_member_limited(
                archive, manifest_info, MAX_MANIFEST_BYTES
            )
            manifest = _load_strict_json(
                manifest_bytes, label="manifest.json",
            )
            if not isinstance(manifest, dict):
                raise SkillBundleError("manifest.json must contain an object")
            schema = manifest.get("schema_version")
            kind = manifest.get("kind")
            if kind in {
                DEPLOYMENT_BUNDLE_KIND,
                LEGACY_DEPLOYMENT_BUNDLE_KIND,
            }:
                raise SkillBundleError(
                    "deployment ZIPs are not portable starting skills; "
                    "export a data-only .rskill instead",
                    code="deployment_bundle_not_portable",
                )
            if (
                kind != BUNDLE_KIND
                or isinstance(schema, bool)
                or not isinstance(schema, int)
                or schema not in SUPPORTED_BUNDLE_SCHEMAS
            ):
                raise SkillBundleError(
                    f"unsupported bundle kind/schema: {kind!r}/{schema!r}"
                )
            source_training_claim = manifest.get("source_training_provenance")
            if schema == 2:
                if (
                    TIERD_TRACKER_POLICY_ORIGIN_MEMBER in infos
                    or source_training_claim is not None
                ):
                    raise SkillBundleError(
                        "schema-v2 bundles cannot claim Tier-D tracker provenance",
                        code="unsupported_member",
                    )
            else:
                expected_members = {
                    "manifest.json",
                    "policy/weights.safetensors",
                    "provenance/origin_policy_contract.json",
                    TIERD_TRACKER_POLICY_ORIGIN_MEMBER,
                }
                if set(infos) != expected_members:
                    raise SkillBundleError(
                        "schema-v3 tracker bundles must contain exactly actor "
                        "safetensors, the origin policy contract, and Tier-D "
                        "tracker provenance",
                        code="unsupported_member",
                    )
                if (
                    not isinstance(source_training_claim, dict)
                    or set(source_training_claim) != {"kind", "member", "sha256"}
                    or source_training_claim.get("kind")
                    != TIERD_TRACKER_POLICY_ORIGIN_KIND
                    or source_training_claim.get("member")
                    != TIERD_TRACKER_POLICY_ORIGIN_MEMBER
                ):
                    raise SkillBundleError(
                        "schema-v3 bundle has no canonical Tier-D tracker "
                        "provenance descriptor",
                        code="source_training_provenance_invalid",
                    )
            descriptors = _parse_descriptors(manifest, set(infos))
            if schema == 3:
                claimed_sha = _validated_sha256(
                    source_training_claim.get("sha256"),
                    label="source_training_provenance.sha256",
                )
                if claimed_sha != descriptors[
                    TIERD_TRACKER_POLICY_ORIGIN_MEMBER
                ][0]:
                    raise SkillBundleError(
                        "source training provenance descriptor digest mismatch",
                        code="descriptor_mismatch",
                    )
            # Attest every described data member before any library mutation.
            # Executable/serialized/unknown formats have already failed the
            # closed member allowlist above.
            for member_name, (expected_sha, expected_size) in descriptors.items():
                _verify_member_descriptor(
                    archive,
                    infos[member_name],
                    expected_sha256=expected_sha,
                    expected_size=expected_size,
                )
            if "world/manifest.json" in infos:
                world_manifest_bytes = _read_member_limited(
                    archive,
                    infos["world/manifest.json"],
                    _PORTABLE_MEMBER_LIMITS["world/manifest.json"],
                )
                world_manifest = _load_strict_json(
                    world_manifest_bytes, label="world/manifest.json",
                )
                if not isinstance(world_manifest, dict):
                    raise SkillBundleError(
                        "world/manifest.json must contain an object"
                    )

            from sculptor.policy_contract import contract_fingerprint

            source_contract = manifest.get("compatibility_contract")
            source_contract_digest = manifest.get("compatibility_contract_digest")
            source_contract_provenance = manifest.get(
                "compatibility_contract_provenance"
            )
            source_contract_provenance_digest = manifest.get(
                "compatibility_contract_provenance_digest"
            )
            contract_provenance_status: Optional[str] = None
            compatibility_provenance_sources: dict[str, Path] = {}
            workdir = Path(td)
            member_names = set(infos)
            has_trainable_policy = _declares_trainable_policy(
                manifest, member_names,
            )
            has_reference = any(
                name in member_names
                for name in (
                    "motion/clip.npz", "motion/provenance.json",
                    "reference/clip.npz", "reference/provenance.json",
                )
            )
            if not has_trainable_policy and not has_reference:
                raise SkillBundleError(
                    "bundle has neither canonical safetensors policy weights "
                    "nor a complete reference trajectory; uploaded .pt files "
                    "are inert and cannot initialize training",
                    code="safetensors_required",
                )
            if has_trainable_policy and target.compatibility_contract is None:
                raise SkillBundleError(
                    target.policy_contract_error
                    or "project_contract_missing: the project policy interface "
                    "contract is unavailable",
                    code="project_contract_missing",
                )

            actual_contract_digest: Optional[str] = None
            source_weights_sha256: Optional[str] = None
            native: Optional[Path] = None
            actual_roles: list[str] = []
            tensor_signature_sha256: Optional[str] = None
            if has_trainable_policy:
                if not isinstance(source_contract, dict):
                    raise SkillBundleError(
                        "trainable bundle is missing compatibility_contract; "
                        "policy initialization is blocked",
                        code="compatibility_contract_required",
                    )
                actual_contract_digest = contract_fingerprint(source_contract)
                if source_contract_digest != actual_contract_digest:
                    raise SkillBundleError(
                        "compatibility_contract_digest mismatch",
                        code="descriptor_mismatch",
                    )
                weights_name, declared_roles = _safetensors_spec(
                    manifest, member_names,
                )
                weights_path = workdir / "policy" / "weights.safetensors"
                sha, size = descriptors[weights_name]
                source_weights_sha256 = sha
                _copy_member_verified(
                    archive,
                    infos[weights_name],
                    weights_path,
                    expected_sha256=sha,
                    expected_size=size,
                )
                native = workdir / "sanitized" / "checkpoint.pt"
                actual_roles, tensor_signature_sha256 = (
                    _native_checkpoint_from_safetensors(
                        weights_path, native, source_contract,
                    )
                )
                if declared_roles and set(declared_roles) != set(actual_roles):
                    raise SkillBundleError(
                        "declared policy_roles do not match safetensors contents"
                    )
                raw_provenance_iter = manifest.get("iter_index")
                if (
                    isinstance(raw_provenance_iter, bool)
                    or not isinstance(raw_provenance_iter, int)
                    or raw_provenance_iter < 0
                ):
                    raise SkillBundleError(
                        "trainable bundle requires a non-negative source iter_index"
                    )
                from sculptor.compatibility_provenance import (
                    CompatibilityProvenanceError,
                    ORIGIN_PERSISTED,
                    provenance_fingerprint,
                    validate_compatibility_contract_provenance,
                )

                if not isinstance(source_contract_provenance, dict):
                    raise SkillBundleError(
                        "trainable bundle is missing "
                        "compatibility_contract_provenance; a generated contract "
                        "without attributable origin evidence cannot initialize "
                        "training",
                        code="compatibility_contract_provenance_required",
                    )
                actual_provenance_digest = provenance_fingerprint(
                    source_contract_provenance
                )
                if source_contract_provenance_digest != actual_provenance_digest:
                    raise SkillBundleError(
                        "compatibility_contract_provenance_digest mismatch",
                        code="descriptor_mismatch",
                    )

                def _read_provenance_member(member_name: str) -> bytes:
                    info = infos.get(member_name)
                    if info is None:
                        raise KeyError(member_name)
                    return _read_member_limited(
                        archive,
                        info,
                        _PORTABLE_MEMBER_LIMITS[member_name],
                    )

                try:
                    contract_provenance_status = (
                        validate_compatibility_contract_provenance(
                            source_contract_provenance,
                            contract=source_contract,
                            policy_roles=list(actual_roles),
                            iter_index=raw_provenance_iter,
                            read_member=_read_provenance_member,
                        )
                    )
                except (CompatibilityProvenanceError, KeyError) as exc:
                    raise SkillBundleError(
                        "compatibility contract provenance failed verification: "
                        f"{exc}",
                        code="compatibility_contract_provenance_invalid",
                    ) from exc
                for evidence_role, row in (
                    source_contract_provenance.get("evidence") or {}
                ).items():
                    member_name = row["path"]
                    source_path = (
                        workdir
                        / "retained-contract-provenance"
                        / Path(member_name).name
                    )
                    sha, size = descriptors[member_name]
                    _copy_member_verified(
                        archive,
                        infos[member_name],
                        source_path,
                        expected_sha256=sha,
                        expected_size=size,
                    )
                    compatibility_provenance_sources[evidence_role] = source_path
                if schema == 3 and (
                    actual_roles != ["actor"]
                    or contract_provenance_status != ORIGIN_PERSISTED
                ):
                    raise SkillBundleError(
                        "schema-v3 Tier-D tracker transfer permits only an "
                        "origin-persisted actor and actor_only initialization",
                        code="source_training_provenance_invalid",
                    )
            elif source_contract is not None or source_contract_digest is not None:
                if not isinstance(source_contract, dict):
                    raise SkillBundleError(
                        "compatibility_contract must be an object when supplied"
                    )
                actual_contract_digest = contract_fingerprint(source_contract)
                if source_contract_digest != actual_contract_digest:
                    raise SkillBundleError(
                        "compatibility_contract_digest mismatch",
                        code="descriptor_mismatch",
                    )
                if (
                    source_contract_provenance is not None
                    or source_contract_provenance_digest is not None
                ):
                    raise SkillBundleError(
                        "reference-only bundles cannot claim policy-contract provenance"
                    )

            deployment = manifest.get("deployment") or {}
            starting = manifest.get("starting_skill") or {}
            if not isinstance(deployment, dict):
                raise SkillBundleError("manifest.deployment must be an object")
            if not isinstance(starting, dict):
                raise SkillBundleError("manifest.starting_skill must be an object")
            task_id = starting.get("task_id") or deployment.get("task_id")
            adapter_class = starting.get("adapter_class")
            if has_trainable_policy:
                if not isinstance(task_id, str) or not task_id.strip():
                    raise SkillBundleError(
                        "trainable bundle does not declare its source task_id; "
                        "target-project identity cannot substitute for provenance",
                        code="task_identity_required",
                    )
                if not isinstance(adapter_class, str) or not adapter_class.strip():
                    raise SkillBundleError(
                        "trainable bundle does not declare its source "
                        "adapter_class; target-project identity cannot "
                        "substitute for provenance",
                        code="adapter_identity_required",
                    )
            else:
                # Reference-only bundles have no executable policy interface.
                # Keep their record fields honest rather than copying the
                # target project's task/adapter and implying source provenance
                # that the archive never declared.
                if not isinstance(task_id, str) or not task_id.strip():
                    task_id = "reference_trajectory"
                if not isinstance(adapter_class, str) or not adapter_class.strip():
                    adapter_class = "reference_trajectory"
            task_id = task_id.strip()
            adapter_class = adapter_class.strip()
            declared_robot_slug = starting.get("robot_slug")
            if declared_robot_slug is None:
                declared_robot_slug = deployment.get("robot_slug")
            if declared_robot_slug == "":
                declared_robot_slug = None
            elif declared_robot_slug is not None and (
                not isinstance(declared_robot_slug, str)
                or _SAFE_ROBOT_RE.fullmatch(declared_robot_slug) is None
            ):
                raise SkillBundleError(
                    "manifest robot_slug must be a safe stable string identifier",
                    code="robot_identity_invalid",
                )
            if has_trainable_policy and not declared_robot_slug:
                raise SkillBundleError(
                    "trainable bundle does not declare its source robot_slug; "
                    "target-project identity cannot substitute for provenance",
                    code="robot_identity_required",
                )

            controller_kind: Optional[str] = None
            controller_sha256: Optional[str] = None
            controller_path = "controller/controller.json"
            if controller_path in infos:
                controller_bytes = _read_member_limited(
                    archive, infos[controller_path], 256 * 1024
                )
                expected_sha, expected_size = descriptors[controller_path]
                if (
                    len(controller_bytes) != expected_size
                    or hashlib.sha256(controller_bytes).hexdigest() != expected_sha
                ):
                    raise SkillBundleError(
                        "controller descriptor mismatch",
                        code="descriptor_mismatch",
                    )
                controller = _load_strict_json(
                    controller_bytes, label="controller/controller.json",
                )
                controller_kind = controller.get("kind") if isinstance(controller, dict) else None
                if controller_kind not in _CONTROLLER_KINDS:
                    raise SkillBundleError(
                        f"controller kind must be one of {sorted(_CONTROLLER_KINDS)}"
                    )
                controller_sha256 = expected_sha

            prepared_reference = _prepare_reference(
                archive,
                infos,
                descriptors,
                workdir,
                expected_robot=declared_robot_slug,
            )
            reference_clip_id = (
                prepared_reference.clip_id if prepared_reference else None
            )
            reference_robot = (
                prepared_reference.robot if prepared_reference else None
            )
            reference_provenance_sha256 = (
                prepared_reference.source_provenance_sha256
                if prepared_reference else None
            )
            robot_slug = declared_robot_slug or reference_robot
            if not robot_slug:
                raise SkillBundleError(
                    "bundle does not carry an exact source robot identity",
                    code="robot_identity_required",
                )

            if actual_roles:
                initialization_modes = ["actor_only"]
                status = "transfer_actor"
                if "critic" in actual_roles:
                    initialization_modes.append("actor_critic")
                    status = "transfer_actor_critic"
                if reference_clip_id:
                    initialization_modes.append("reference_only")
            else:
                initialization_modes = ["reference_only"]
                status = "reference_only"
            bundled_world = any(name.startswith("world/") for name in infos)
            world_bundle_sha256 = None
            if bundled_world:
                world_rows = [
                    {"path": name, "sha256": descriptors[name][0], "bytes": descriptors[name][1]}
                    for name in sorted(infos)
                    if name.startswith("world/")
                ]
                world_bundle_sha256 = hashlib.sha256(
                    json.dumps(
                        world_rows, sort_keys=True, separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            original_checkpoint_sha: Optional[str] = None
            source_training_provenance: dict[str, Any] | None = None
            source_training_provenance_sha256: str | None = None
            source_training_provenance_source: Path | None = None
            checkpoint_meta = manifest.get("checkpoint")
            if checkpoint_meta is not None and not isinstance(checkpoint_meta, dict):
                raise SkillBundleError("manifest.checkpoint must be an object")
            if isinstance(checkpoint_meta, dict):
                raw_sha = checkpoint_meta.get("sha256")
                if raw_sha is not None:
                    original_checkpoint_sha = _validated_sha256(
                        raw_sha, label="manifest.checkpoint.sha256",
                    )
            if schema == 3:
                if original_checkpoint_sha is None:
                    raise SkillBundleError(
                        "schema-v3 tracker bundle must pin its excluded raw "
                        "source checkpoint",
                        code="source_training_provenance_invalid",
                    )
                origin_member_sha, origin_member_size = descriptors[
                    TIERD_TRACKER_POLICY_ORIGIN_MEMBER
                ]
                source_training_provenance_source = (
                    workdir / "retained-source-training" / "tierd_tracker_origin.json"
                )
                _copy_member_verified(
                    archive,
                    infos[TIERD_TRACKER_POLICY_ORIGIN_MEMBER],
                    source_training_provenance_source,
                    expected_sha256=origin_member_sha,
                    expected_size=origin_member_size,
                )
                origin_payload = _load_strict_json(
                    source_training_provenance_source.read_bytes(),
                    label=TIERD_TRACKER_POLICY_ORIGIN_MEMBER,
                )
                try:
                    source_training_provenance = (
                        validate_tierd_tracker_policy_origin(
                            origin_payload,
                            source_contract=source_contract,
                            source_checkpoint_sha256=original_checkpoint_sha,
                            portable_actor_safetensors_sha256=str(
                                source_weights_sha256
                            ),
                        )
                    )
                except TierDTrackerPolicyError as exc:
                    raise SkillBundleError(
                        f"Tier-D tracker source provenance failed verification: {exc}",
                        code="source_training_provenance_invalid",
                    ) from exc
                if source_training_provenance.get("robot") != robot_slug:
                    raise SkillBundleError(
                        "Tier-D tracker provenance robot differs from the "
                        "portable policy robot",
                        code="robot_identity_invalid",
                    )
                source_training_provenance_sha256 = origin_member_sha
            raw_warnings = manifest.get("warnings")
            if raw_warnings is None:
                warnings: list[str] = []
            elif not isinstance(raw_warnings, list) or any(
                not isinstance(item, str) for item in raw_warnings
            ):
                raise SkillBundleError("manifest.warnings must be a list of strings")
            else:
                warnings = list(raw_warnings)
            if schema == 3:
                warnings.append(
                    "The bundle's Tier-D tracker origin is an unauthenticated "
                    "source claim with an internally consistent hash chain. "
                    "It is retained as provenance only; no source reference, "
                    "reward, world, controller, or mode executor is selected "
                    "by this import."
                )
            raw_source_iter_index = manifest.get("iter_index", -1)
            if raw_source_iter_index is None:
                source_iter_index = -1
            elif (
                isinstance(raw_source_iter_index, bool)
                or not isinstance(raw_source_iter_index, int)
                or raw_source_iter_index < -1
            ):
                raise SkillBundleError(
                    "manifest.iter_index must be an integer greater than or equal "
                    "to -1"
                )
            else:
                source_iter_index = raw_source_iter_index
            if bundled_world:
                warnings.append(
                    "Bundled world bytes were verified and then discarded; "
                    "only their aggregate digest was recorded. This import is "
                    "not an activatable world."
                )
            if controller_kind:
                warnings.append(
                    "Bundled controller JSON was verified and then discarded; "
                    "only its kind and digest were recorded. It is neither "
                    "executable nor activatable."
                )
            transaction = (
                _reference_import_transaction_lock()
                if prepared_reference is not None
                else nullcontext()
            )
            with transaction:
                registration: Optional[_ReferenceRegistration] = None
                publication_state: dict[str, bool] = {"created": False}
                record: Optional[SkillRecord] = None
                try:
                    if prepared_reference is not None:
                        registration = _install_reference(prepared_reference)
                    record = self.library.publish_imported_checkpoint(
                        checkpoint_path=native,
                        adapter_class=adapter_class,
                        task_id=task_id,
                        robot_slug=robot_slug,
                        alias=(
                            str(
                                starting.get("name")
                                or manifest.get("project")
                                or "Imported skill"
                            )
                        ),
                        manifest_digest=hashlib.sha256(
                            manifest_bytes,
                        ).hexdigest(),
                        manifest_schema_version=int(schema),
                        original_checkpoint_sha256=original_checkpoint_sha,
                        source_weights_sha256=source_weights_sha256,
                        reference_clip_id=reference_clip_id,
                        reference_robot=reference_robot,
                        reference_sha256=(
                            prepared_reference.content_sha256
                            if prepared_reference is not None
                            else None
                        ),
                        reference_provenance_sha256=(
                            reference_provenance_sha256
                        ),
                        world_bundle_sha256=world_bundle_sha256,
                        compatibility_contract=source_contract,
                        compatibility_contract_digest=actual_contract_digest,
                        compatibility_contract_provenance=(
                            source_contract_provenance
                            if has_trainable_policy
                            else None
                        ),
                        compatibility_contract_provenance_digest=(
                            source_contract_provenance_digest
                            if has_trainable_policy
                            else None
                        ),
                        compatibility_contract_provenance_status=(
                            contract_provenance_status
                        ),
                        compatibility_provenance_sources=(
                            compatibility_provenance_sources
                        ),
                        source_training_provenance=(
                            source_training_provenance
                        ),
                        source_training_provenance_sha256=(
                            source_training_provenance_sha256
                        ),
                        source_training_provenance_source=(
                            source_training_provenance_source
                        ),
                        tensor_contract_verified=bool(actual_roles),
                        tensor_signature_sha256=tensor_signature_sha256,
                        compatibility_status=status,
                        initialization_modes=initialization_modes,
                        policy_roles=actual_roles,
                        controller_kind=controller_kind,
                        controller_sha256=controller_sha256,
                        bundled_world=bundled_world,
                        warnings=warnings,
                        source_project=str(
                            manifest.get("project") or "imported bundle"
                        ),
                        source_iter_index=source_iter_index,
                        publication_state=publication_state,
                    )
                    receipt = receipt_for(record, target)
                    imported = ImportedStartingSkill(
                        record=record, receipt=receipt,
                    )
                    if admission_callback is not None:
                        try:
                            admission_callback(imported)
                        except Exception as exc:
                            raise SkillBundleError(
                                "bundle components were valid, but immutable "
                                "admission lineage could not be published",
                                code="lineage_publication_failed",
                            ) from exc
                except BaseException:
                    try:
                        if (
                            record is not None
                            and publication_state.get("created") is True
                        ):
                            self.library.rollback_imported_publication(record)
                    finally:
                        _rollback_reference_registration(registration)
                    raise
            return imported


__all__ = [
    "BUNDLE_KIND",
    "DEPLOYMENT_BUNDLE_KIND",
    "LEGACY_DEPLOYMENT_BUNDLE_KIND",
    "ImportTarget",
    "ImportedStartingSkill",
    "MAX_ARCHIVE_BYTES",
    "SkillBundleError",
    "StartingSkillBundleImporter",
    "compatibility_for",
    "reference_source_provenance_sha256",
    "receipt_for",
]
