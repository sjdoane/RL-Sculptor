"""sculptor/skill_library.py — Ship 19: cross-mission skill library.

Public surface: a content-addressed, filesystem-backed registry of
trained policies. Each `SkillRecord` carries a checkpoint + metadata
(adapter class, task identity, source mission/stage, training metric)
that lets a future mission's `decompose_task` pick a compatible prior
policy and the orchestrator warm-start a stage from it.

Architectural references:
  - **CurricuLLM** (arXiv:2409.18382) — curriculum stages warm-start
    from prior stages.
  - **Voyager** (arXiv:2305.16291) — the "skill library" pattern: a
    growing repository of learned skills retrievable by description.
  - **Ship 15** (sculptor.sculpt._train_or_resume) — the warm-start
    plumbing this module hands `init_policy_path` to. mjlab is the
    only adapter currently implementing the kwarg; other adapters
    drop it silently. We GATE publish on `adapter.train` accepting
    `init_policy_path` so the library never grows entries that no
    future mission could load.

Ship 19 scope (this file):
  - SkillRecord dataclass + JSON roundtrip.
  - SkillLibrary: filesystem registry with publish / list / load.
  - SkillLibraryHandle: per-(adapter, task_id) call-site context that
    consolidates the kwargs `mission_run` and `decompose_task` would
    otherwise have to pipe through individually.
  - derive_skill_id, default_library_root, adapter_supports_warm_start.

Out of scope (future ships):
  - Cross-adapter skills (gym_sb3 → mjlab).
  - Cross-task / cross-env compatibility checks.
  - Auto-pruning / quotas / cleanup of stale skills.
  - Skill deletion API (manual `rm -rf` on the directory works for v1).
  - LLM-evaluator second-opinion before publish — Ship 20+.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# Import only the type we need. Stage's `init_skill_id` is the user-
# facing surface; `redecomposition_attempts` gates publish (Ship 17
# sub-stages are specialized and should not enter the general library).
from sculptor.mission import Mission, Stage


SCHEMA_VERSION = 4

# ── Defaults / constants ─────────────────────────────────────────────

#: Env var override for the library root. Tests use this OR the
#: explicit `SkillLibrary(root=...)` constructor — see test fixtures.
ENV_LIBRARY_ROOT = "SCULPTOR_SKILL_LIBRARY_ROOT"

#: Filename inside each `<root>/<skill_id>/` directory.
METADATA_FILENAME = "metadata.json"

#: Filelock timeout (s). Mirrors the mission filelock at sculpt.py.
_LOCK_TIMEOUT_S = 10.0

#: Skill-id length. 12 hex chars = 48 bits of entropy ≈ 1-in-280
#: trillion collision rate, well above the size of any plausible
#: library. We hash deterministically, so identical (adapter, task,
#: ckpt-bytes) re-publishes idempotently into the same id.
_SKILL_ID_HEX_LEN = 12

_SKILL_ID_RE = re.compile(rf"^[0-9a-f]{{{_SKILL_ID_HEX_LEN}}}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_BASENAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,250}\.(?:pt|zip)$"
)
_WINDOWS_RESERVED_STEMS = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})
_ALLOWED_SOURCES = frozenset({"trained", "imported_bundle"})
_ALLOWED_TRUST_STATUSES = frozenset({
    "verified_local", "sanitized", "validated",
})
_ALLOWED_CHECKPOINT_FORMATS = frozenset({
    "native_pt", "server_native_pt", "none",
})
_ALLOWED_SOURCE_FORMATS = frozenset({None, "safetensors"})
_ALLOWED_COMPATIBILITY_STATUSES = frozenset({
    "transfer_actor", "transfer_actor_critic", "reference_only",
})
_ALLOWED_INITIALIZATION_MODES = frozenset({
    "actor_only", "actor_critic", "reference_only",
})
_ALLOWED_POLICY_ROLES = frozenset({"actor", "critic"})
_ALLOWED_EXECUTION_MODELS = frozenset({
    "policy_checkpoint",
    "reward_sculpting",
    "flat_reference_tracking_residual",
    "phase_window_automaton",
})
_PROVENANCE_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,250}$")


# ── Default library root ─────────────────────────────────────────────

def default_library_root() -> Path:
    """Return the default on-disk library root.

    Resolution: `$SCULPTOR_SKILL_LIBRARY_ROOT` if set, else
    `~/.local/share/sculptor/skills/`. The directory is NOT created
    eagerly — `SkillLibrary(root).publish_from_stage(...)` creates it
    lazily on first write.
    """
    env = os.environ.get(ENV_LIBRARY_ROOT)
    if env:
        return Path(env).expanduser()
    return Path("~/.local/share/sculptor/skills").expanduser()


# ── Skill ID derivation ──────────────────────────────────────────────

def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Full-file SHA-256 hex digest. ~1.5 s per 500 MB on NVMe.
    Single-pass; no mmap (mjlab checkpoints can exceed available RAM
    on consumer laptops). Used for skill-id derivation AND a future
    integrity check (the digest is stored in metadata)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def derive_skill_id(
    adapter_class: str,
    task_id: str,
    checkpoint_sha256: str,
) -> str:
    """Deterministic skill id.

    Inputs intentionally exclude reward_seed_prompt and
    success_criterion: those are metadata used for ranking + Claude
    context, not identity. Identity is "this policy on this task" so
    re-publishing the same policy file under a different criterion
    description does NOT create a duplicate (audit fix C3).

    Returns 12-hex-char string (48 bits).
    """
    payload = "\n".join([
        adapter_class.strip(),
        task_id.strip(),
        checkpoint_sha256.strip(),
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_SKILL_ID_HEX_LEN]


# ── Adapter capability check ─────────────────────────────────────────

def adapter_supports_warm_start(adapter: Any) -> bool:
    """True iff `adapter.train(...)` accepts `init_policy_path`.

    Mirrors the `_train_or_resume` introspection at sculpt.py:732
    (Ship 15 audit fix): accept an explicit `init_policy_path`
    parameter OR a `**kwargs` catch-all (because `**kwargs` adapters
    might forward the kwarg to a downstream call). When neither is
    present, the adapter would silently drop `init_policy_path` —
    publishing a skill from such an adapter would mean creating a
    library entry no future mission could ever load.
    """
    train = getattr(adapter, "train", None)
    if train is None:
        return False
    try:
        sig = inspect.signature(train)
    except (TypeError, ValueError):
        return False
    if "init_policy_path" in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


# ── SkillRecord ──────────────────────────────────────────────────────

@dataclass
class SkillRecord:
    """Metadata for one library entry.

    Stored on disk as `<root>/<skill_id>/metadata.json`. The actual
    policy file lives in the same directory under
    `<checkpoint_filename>` so the record is fully relocatable: move
    `<root>` to a new machine, the records still resolve. Only the
    file basename is persisted (Ship 18a path-relocation precedent).
    """

    schema_version: int
    skill_id: str
    adapter_class: str          # full dotted path, e.g. sculptor.adapters.mjlab.MjlabAdapter
    task_id: str                # mjlab convention, e.g. Mjlab-Velocity-Flat-Unitree-Go1
    robot_slug: Optional[str]   # optional UI-side stable id (None when called from CLI without ui context)
    reward_seed_prompt: str
    success_criterion: str
    final_metric: float          # BEST primary_metric across iters (audit fix C2)
    source_iter_index: int       # iter that produced the published checkpoint
    iterations_used: int
    source_mission_goal: str
    source_stage_name: str
    created_at: str              # ISO-8601 UTC
    checkpoint_filename: str     # basename only; resolve via `record_dir / checkpoint_filename`
    checkpoint_sha256: str       # full-file digest for future integrity checks
    checkpoint_size_bytes: int
    alias: Optional[str] = None  # human-friendly tag (currently unused; reserved for Ship 19c)
    # Schema v2: imported-starting-skill provenance.  Every field is
    # defaulted so v1 records remain readable without a migration pass.
    source: str = "trained"
    trust_status: str = "verified_local"
    checkpoint_format: str = "native_pt"
    source_format: Optional[str] = None
    identity_digest: Optional[str] = None
    manifest_digest: Optional[str] = None
    manifest_schema_version: Optional[int] = None
    original_checkpoint_sha256: Optional[str] = None
    source_weights_sha256: Optional[str] = None
    # Schema v4: an actor may come from a trusted-local Tier-D tracker export.
    # This compact summary and its retained full origin document describe how
    # the weights were trained; they never select the source reference/world/
    # controller/reward or imply optimizer/exact resume.
    source_training_provenance: Optional[dict[str, Any]] = None
    source_training_provenance_sha256: Optional[str] = None
    reference_clip_id: Optional[str] = None
    reference_robot: Optional[str] = None
    reference_sha256: Optional[str] = None
    reference_provenance_sha256: Optional[str] = None
    world_bundle_sha256: Optional[str] = None
    # Schema v3: exact execution provenance for locally trained mission
    # stages.  These fields describe the bytes that produced the policy; they
    # do not imply that a future warm-start will reuse the old reward, world,
    # reference controller, or phase executor.
    reference_dynamics_certificate_sha256: Optional[str] = None
    reference_rollout_sha256: Optional[str] = None
    reference_execution_contract_sha256: Optional[str] = None
    reference_execution_boundary_sha256: Optional[str] = None
    active_reward_sha256: Optional[str] = None
    mode_execution_manifest_digest: Optional[str] = None
    world_tuple_hash: Optional[str] = None
    world_selection_sha256: Optional[str] = None
    world_artifact_sha256: dict[str, str] = field(default_factory=dict)
    # Server-owned immutable copies. Keys are semantic roles such as
    # ``active_reward``, ``world_selection``, and ``world:task``; values are
    # safe basenames within the skill directory plus their exact digest.
    provenance_files: dict[str, dict[str, str]] = field(default_factory=dict)
    execution_model: str = "policy_checkpoint"
    mode_reuse_supported: bool = False
    compatibility_contract: Optional[dict[str, Any]] = None
    compatibility_contract_digest: Optional[str] = None
    compatibility_contract_provenance: Optional[dict[str, Any]] = None
    compatibility_contract_provenance_digest: Optional[str] = None
    compatibility_contract_provenance_status: Optional[str] = None
    # A compatibility contract describes the claimed interface. Imported
    # trainable skills additionally record that the actual safetensors keys
    # and shapes were checked against that contract before the server-owned
    # checkpoint was created.  The signature is a canonical digest of the
    # admitted tensor names, dtypes, and shapes (never tensor values).
    tensor_contract_verified: bool = False
    tensor_signature_sha256: Optional[str] = None
    compatibility_status: str = "transfer_actor_critic"
    initialization_modes: list[str] = field(
        default_factory=lambda: ["actor_only", "actor_critic"]
    )
    policy_roles: list[str] = field(
        default_factory=lambda: ["actor", "critic"]
    )
    controller_kind: Optional[str] = None
    controller_sha256: Optional[str] = None
    bundled_world: bool = False
    import_warnings: list[str] = field(default_factory=list)
    imported_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        expected_skill_id: Optional[str] = None,
    ) -> "SkillRecord":
        if not isinstance(data, dict):
            raise SkillLibraryError("skill metadata must be a JSON object")
        ver = int(data.get("schema_version", SCHEMA_VERSION))
        if ver < 1 or ver > SCHEMA_VERSION:
            raise SkillLibraryError(
                f"skill metadata.json schema_version={ver} is outside the "
                f"supported range 1..{SCHEMA_VERSION}"
            )
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        record = cls(**filtered)
        _validate_skill_record(record, expected_skill_id=expected_skill_id)
        return record


class SkillLibraryError(RuntimeError):
    """Raised on unrecoverable library failure (corrupt root, lock
    timeout, etc.). Per-record JSON parse errors are NOT raised —
    `list_compatible` skips bad records to keep the library
    queryable when one entry rots."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _import_identity_digest(record: SkillRecord) -> str:
    payload = {
        "manifest_digest": record.manifest_digest,
        "manifest_schema_version": record.manifest_schema_version,
        "source_weights_sha256": record.source_weights_sha256,
        "reference_clip_id": record.reference_clip_id,
        "reference_robot": record.reference_robot,
        "reference_sha256": record.reference_sha256,
        "reference_provenance_sha256": record.reference_provenance_sha256,
        "world_bundle_sha256": record.world_bundle_sha256,
        "controller_sha256": record.controller_sha256,
        "compatibility_contract_digest": record.compatibility_contract_digest,
        "compatibility_contract_provenance_digest": (
            record.compatibility_contract_provenance_digest
        ),
        "compatibility_contract_provenance_status": (
            record.compatibility_contract_provenance_status
        ),
        "tensor_signature_sha256": record.tensor_signature_sha256,
        "compatibility_status": record.compatibility_status,
        "initialization_modes": record.initialization_modes,
        "policy_roles": record.policy_roles,
    }
    # Preserve schema-v2 import identities exactly.  New tracker-origin fields
    # enter identity only when present, so existing immutable records remain
    # readable without changing their derived skill ids.
    if (
        record.source_training_provenance is not None
        or record.source_training_provenance_sha256 is not None
    ):
        payload["source_training_provenance"] = (
            record.source_training_provenance
        )
        payload["source_training_provenance_sha256"] = (
            record.source_training_provenance_sha256
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_string_list(
    value: Any,
    *,
    field_name: str,
    allowed: frozenset[str],
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise SkillLibraryError(f"{field_name} must be a list of strings")
    if len(value) != len(set(value)):
        raise SkillLibraryError(f"{field_name} must not contain duplicates")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SkillLibraryError(
            f"{field_name} contains unsupported values: {', '.join(unknown)}"
        )
    return value


def _validate_checkpoint_filename(filename: Any, *, allow_empty: bool) -> str:
    if not isinstance(filename, str):
        raise SkillLibraryError("checkpoint_filename must be a string")
    if not filename:
        if allow_empty:
            return filename
        raise SkillLibraryError("trainable skill has an empty checkpoint_filename")
    if (
        filename in {".", "..", METADATA_FILENAME}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
        or _CHECKPOINT_BASENAME_RE.fullmatch(filename) is None
        or Path(filename).stem.upper() in _WINDOWS_RESERVED_STEMS
    ):
        raise SkillLibraryError(
            "checkpoint_filename must be a safe .pt/.zip basename"
        )
    return filename


def _validate_provenance_files(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise SkillLibraryError("provenance_files must be an object")
    normalized: dict[str, dict[str, str]] = {}
    seen_filenames: set[str] = set()
    for role, descriptor in value.items():
        if not isinstance(role, str) or not role or len(role) > 120:
            raise SkillLibraryError("provenance file roles must be non-empty strings")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "filename", "sha256",
        }:
            raise SkillLibraryError(
                f"provenance file {role!r} must contain filename and sha256"
            )
        filename = descriptor.get("filename")
        digest = descriptor.get("sha256")
        if (
            not isinstance(filename, str)
            or _PROVENANCE_BASENAME_RE.fullmatch(filename) is None
            or Path(filename).name != filename
            or filename in {METADATA_FILENAME, ".", ".."}
            or filename.upper().split(".", 1)[0] in _WINDOWS_RESERVED_STEMS
        ):
            raise SkillLibraryError(
                f"provenance file {role!r} has an unsafe basename"
            )
        if not _is_sha256(digest):
            raise SkillLibraryError(
                f"provenance file {role!r} requires a lowercase SHA-256"
            )
        # A single retained byte artifact may serve several semantic roles
        # (the active reward is also the world's reward ref), but a filename
        # cannot carry contradictory digests.
        if filename in seen_filenames:
            prior = next(
                item for item in normalized.values()
                if item["filename"] == filename
            )
            if prior["sha256"] != digest:
                raise SkillLibraryError(
                    f"provenance basename {filename!r} has conflicting digests"
                )
        seen_filenames.add(filename)
        normalized[role] = {"filename": filename, "sha256": digest}
    return normalized


def _validate_skill_record(
    record: SkillRecord,
    *,
    expected_skill_id: Optional[str] = None,
) -> None:
    """Validate persisted metadata before it can influence a launch.

    The library is writable state, not a trust boundary.  Recompute identity
    and constrain every execution-facing enum so editing metadata cannot turn
    a reference candidate into a trainable policy or redirect a checkpoint.
    """
    if not isinstance(record.skill_id, str) or not _SKILL_ID_RE.fullmatch(
        record.skill_id
    ):
        raise SkillLibraryError("skill_id must be exactly 12 lowercase hex chars")
    if expected_skill_id is not None:
        if not isinstance(expected_skill_id, str) or not _SKILL_ID_RE.fullmatch(
            expected_skill_id
        ):
            raise SkillLibraryError("requested skill_id is not canonical")
        if record.skill_id != expected_skill_id:
            raise SkillLibraryError(
                "metadata skill_id does not match its requested library directory"
            )
    if not isinstance(record.adapter_class, str) or not record.adapter_class.strip():
        raise SkillLibraryError("adapter_class must be a non-empty string")
    if not isinstance(record.task_id, str) or not record.task_id.strip():
        raise SkillLibraryError("task_id must be a non-empty string")
    if not isinstance(record.source, str) or record.source not in _ALLOWED_SOURCES:
        raise SkillLibraryError(f"unsupported skill source: {record.source!r}")
    if (
        not isinstance(record.trust_status, str)
        or record.trust_status not in _ALLOWED_TRUST_STATUSES
    ):
        raise SkillLibraryError(
            f"unsupported skill trust_status: {record.trust_status!r}"
        )
    if (
        not isinstance(record.checkpoint_format, str)
        or record.checkpoint_format not in _ALLOWED_CHECKPOINT_FORMATS
    ):
        raise SkillLibraryError(
            f"unsupported checkpoint_format: {record.checkpoint_format!r}"
        )
    if (
        record.source_format is not None
        and (
            not isinstance(record.source_format, str)
            or record.source_format not in _ALLOWED_SOURCE_FORMATS
        )
    ):
        raise SkillLibraryError(
            f"unsupported source_format: {record.source_format!r}"
        )
    if (
        not isinstance(record.compatibility_status, str)
        or record.compatibility_status not in _ALLOWED_COMPATIBILITY_STATUSES
    ):
        raise SkillLibraryError(
            f"unsupported compatibility_status: {record.compatibility_status!r}"
        )

    modes = _validate_string_list(
        record.initialization_modes,
        field_name="initialization_modes",
        allowed=_ALLOWED_INITIALIZATION_MODES,
    )
    roles = _validate_string_list(
        record.policy_roles,
        field_name="policy_roles",
        allowed=_ALLOWED_POLICY_ROLES,
    )
    has_checkpoint = record.checkpoint_format != "none"
    _validate_checkpoint_filename(
        record.checkpoint_filename, allow_empty=not has_checkpoint,
    )
    if has_checkpoint:
        if not _is_sha256(record.checkpoint_sha256):
            raise SkillLibraryError("checkpoint_sha256 must be lowercase SHA-256")
        if (
            not isinstance(record.checkpoint_size_bytes, int)
            or isinstance(record.checkpoint_size_bytes, bool)
            or record.checkpoint_size_bytes <= 0
        ):
            raise SkillLibraryError("checkpoint_size_bytes must be positive")
        if "actor" not in roles or "actor_only" not in modes:
            raise SkillLibraryError(
                "trainable checkpoint requires the actor role and actor_only mode"
            )
    elif (
        record.checkpoint_filename
        or record.checkpoint_sha256
        or record.checkpoint_size_bytes != 0
        or roles
    ):
        raise SkillLibraryError(
            "checkpoint_format=none cannot carry checkpoint or policy metadata"
        )

    expected_modes: list[str]
    expected_status: str
    if roles == ["actor"]:
        expected_modes = ["actor_only"]
        expected_status = "transfer_actor"
    elif roles == ["actor", "critic"]:
        expected_modes = ["actor_only", "actor_critic"]
        expected_status = "transfer_actor_critic"
    elif not roles:
        expected_modes = ["reference_only"]
        expected_status = "reference_only"
    else:
        raise SkillLibraryError(
            "policy_roles must be [], ['actor'], or ['actor', 'critic']"
        )
    # An imported bundle may expose its reference as an independent starting
    # point.  A locally trained policy merely RECORDS the reference that
    # shaped it; mission publication must not silently advertise reference or
    # phase-mode reuse.
    if record.source == "imported_bundle" and roles and record.reference_clip_id:
        expected_modes.append("reference_only")
    if modes != expected_modes or record.compatibility_status != expected_status:
        raise SkillLibraryError(
            "initialization modes/status do not match the admitted policy roles"
        )
    if "reference_only" in modes and not (
        isinstance(record.reference_clip_id, str)
        and record.reference_clip_id
        and isinstance(record.reference_robot, str)
        and record.reference_robot
    ):
        raise SkillLibraryError(
            "reference_only mode requires an exact reference clip and robot"
        )

    if record.compatibility_contract is None:
        if record.compatibility_contract_digest is not None:
            raise SkillLibraryError(
                "compatibility contract digest has no corresponding contract"
            )
    else:
        if not isinstance(record.compatibility_contract, dict):
            raise SkillLibraryError("compatibility_contract must be an object")
        from sculptor.policy_contract import contract_fingerprint

        actual_contract_digest = contract_fingerprint(record.compatibility_contract)
        if record.compatibility_contract_digest != actual_contract_digest:
            raise SkillLibraryError("compatibility contract digest mismatch")

    if record.compatibility_contract_provenance is None:
        if (
            record.compatibility_contract_provenance_digest is not None
            or record.compatibility_contract_provenance_status is not None
        ):
            raise SkillLibraryError(
                "compatibility contract provenance metadata is incomplete"
            )
    else:
        if not isinstance(record.compatibility_contract_provenance, dict):
            raise SkillLibraryError(
                "compatibility_contract_provenance must be an object"
            )
        from sculptor.compatibility_provenance import (
            LEGACY_RECONSTRUCTED,
            ORIGIN_PERSISTED,
            provenance_fingerprint,
        )

        if (
            record.compatibility_contract_provenance_digest
            != provenance_fingerprint(record.compatibility_contract_provenance)
        ):
            raise SkillLibraryError(
                "compatibility contract provenance digest mismatch"
            )
        status = record.compatibility_contract_provenance.get("status")
        if (
            status not in {ORIGIN_PERSISTED, LEGACY_RECONSTRUCTED}
            or record.compatibility_contract_provenance_status != status
        ):
            raise SkillLibraryError(
                "compatibility contract provenance status is invalid"
            )

    if (
        not isinstance(record.execution_model, str)
        or record.execution_model not in _ALLOWED_EXECUTION_MODELS
    ):
        raise SkillLibraryError(
            f"unsupported execution_model: {record.execution_model!r}"
        )
    if not isinstance(record.mode_reuse_supported, bool):
        raise SkillLibraryError("mode_reuse_supported must be boolean")
    if record.mode_reuse_supported:
        raise SkillLibraryError(
            "published skills do not support phase-mode executor reuse; "
            "only the policy checkpoint may initialize a new run"
        )
    provenance_files = _validate_provenance_files(record.provenance_files)
    if record.compatibility_contract_provenance is not None:
        evidence = record.compatibility_contract_provenance.get("evidence") or {}
        expected_roles = {
            f"compatibility_contract:{role}": row.get("sha256")
            for role, row in evidence.items()
            if isinstance(row, dict)
        }
        retained_roles = {
            role: descriptor["sha256"]
            for role, descriptor in provenance_files.items()
            if role.startswith("compatibility_contract:")
        }
        if retained_roles != expected_roles:
            raise SkillLibraryError(
                "compatibility contract evidence has no exact retained byte set"
            )
    source_training_file = provenance_files.get("source_training")
    if record.source_training_provenance is None:
        if (
            record.source_training_provenance_sha256 is not None
            or source_training_file is not None
        ):
            raise SkillLibraryError(
                "source training provenance metadata is incomplete"
            )
    else:
        if not isinstance(record.source_training_provenance, dict):
            raise SkillLibraryError(
                "source_training_provenance must be an object"
            )
        if (
            not _is_sha256(record.source_training_provenance_sha256)
            or source_training_file is None
            or source_training_file["sha256"]
            != record.source_training_provenance_sha256
        ):
            raise SkillLibraryError(
                "source training provenance has no exact retained byte artifact"
            )
        if (
            record.source != "imported_bundle"
            or roles != ["actor"]
            or modes != ["actor_only"]
            or record.compatibility_status != "transfer_actor"
            or record.compatibility_contract is None
            or record.compatibility_contract_provenance_status
            != "origin_persisted"
            or not _is_sha256(record.original_checkpoint_sha256)
            or record.execution_model != "policy_checkpoint"
            or record.mode_reuse_supported
        ):
            raise SkillLibraryError(
                "Tier-D tracker provenance permits imported actor-only "
                "initialization, not resume or executor reuse"
            )
        forbidden_source_components = (
            record.reference_clip_id,
            record.reference_robot,
            record.reference_sha256,
            record.reference_provenance_sha256,
            record.reference_dynamics_certificate_sha256,
            record.reference_rollout_sha256,
            record.reference_execution_contract_sha256,
            record.reference_execution_boundary_sha256,
            record.world_bundle_sha256,
            record.world_tuple_hash,
            record.world_selection_sha256,
            record.active_reward_sha256,
            record.mode_execution_manifest_digest,
            record.controller_kind,
            record.controller_sha256,
        )
        if (
            any(value is not None for value in forbidden_source_components)
            or record.world_artifact_sha256
            or record.bundled_world
        ):
            raise SkillLibraryError(
                "Tier-D tracker provenance cannot activate or bundle its "
                "source reference, world, or controller"
            )
        try:
            from sculptor.tierd_tracker_policy import (
                TierDTrackerPolicyError,
                validate_tierd_tracker_policy_summary,
            )

            summary = validate_tierd_tracker_policy_summary(
                record.source_training_provenance,
                source_contract=record.compatibility_contract,
                source_checkpoint_sha256=str(record.original_checkpoint_sha256),
                portable_actor_safetensors_sha256=str(
                    record.source_weights_sha256
                ),
            )
        except TierDTrackerPolicyError as exc:
            raise SkillLibraryError(
                f"source training provenance summary is invalid: {exc}"
            ) from exc
        if (
            summary != record.source_training_provenance
            or summary.get("robot") != record.robot_slug
            or summary.get("tracker_iterations") != record.source_iter_index
        ):
            raise SkillLibraryError(
                "source training provenance summary differs from skill metadata"
            )
    for field_name in (
        "reference_dynamics_certificate_sha256",
        "reference_rollout_sha256",
        "reference_execution_contract_sha256",
        "reference_execution_boundary_sha256",
        "active_reward_sha256",
        "mode_execution_manifest_digest",
        "world_tuple_hash",
        "world_selection_sha256",
    ):
        value = getattr(record, field_name)
        if value is not None and not _is_sha256(value):
            raise SkillLibraryError(f"{field_name} must be lowercase SHA-256")
    if not isinstance(record.world_artifact_sha256, dict):
        raise SkillLibraryError("world_artifact_sha256 must be an object")
    for kind, digest in record.world_artifact_sha256.items():
        if not isinstance(kind, str) or not kind or not _is_sha256(digest):
            raise SkillLibraryError(
                "world_artifact_sha256 requires non-empty names and SHA-256 values"
            )

    if record.active_reward_sha256 is not None:
        active = provenance_files.get("active_reward")
        if active is None or active["sha256"] != record.active_reward_sha256:
            raise SkillLibraryError(
                "active reward digest has no matching retained reward bytes"
            )
    elif record.mode_execution_manifest_digest is not None:
        raise SkillLibraryError(
            "mode execution manifest cannot exist without an active reward"
        )
    if record.execution_model == "phase_window_automaton":
        if record.mode_execution_manifest_digest is None:
            raise SkillLibraryError(
                "phase_window_automaton requires a manifest digest"
            )
    elif record.mode_execution_manifest_digest is not None:
        raise SkillLibraryError(
            "mode manifest digest requires phase_window_automaton execution_model"
        )

    world_fields = (
        record.world_tuple_hash,
        record.world_selection_sha256,
        record.world_artifact_sha256,
    )
    if any(bool(value) for value in world_fields):
        if (
            record.world_tuple_hash is None
            or record.world_selection_sha256 is None
            or not record.world_artifact_sha256
        ):
            raise SkillLibraryError(
                "world provenance requires tuple, selection, and artifact digests"
            )
        selection = provenance_files.get("world_selection")
        if (
            selection is None
            or selection["sha256"] != record.world_selection_sha256
        ):
            raise SkillLibraryError(
                "world selection digest has no matching retained selection bytes"
            )
        missing_world = sorted(
            kind for kind in record.world_artifact_sha256
            if f"world:{kind}" not in provenance_files
        )
        if missing_world:
            raise SkillLibraryError(
                "world provenance is missing retained artifacts: "
                + ", ".join(missing_world)
            )

    trained_reference_hashes = (
        record.reference_sha256,
        record.reference_dynamics_certificate_sha256,
        record.reference_rollout_sha256,
        record.reference_execution_contract_sha256,
        record.reference_execution_boundary_sha256,
    )
    if record.source == "trained" and record.reference_clip_id:
        if not record.reference_robot or any(
            not _is_sha256(value) for value in trained_reference_hashes
        ):
            raise SkillLibraryError(
                "trained reference-guided skill lacks exact clip, dynamics, "
                "rollout, execution-contract, or boundary provenance"
            )
        if record.execution_model not in {
            "flat_reference_tracking_residual", "phase_window_automaton",
        }:
            raise SkillLibraryError(
                "reference-guided trained skill must disclose its execution model"
            )
    elif record.source == "trained" and any(
        value is not None for value in trained_reference_hashes
    ):
        raise SkillLibraryError(
            "trained reference hashes require reference_robot and reference_clip_id"
        )

    if record.source == "trained":
        if (
            record.trust_status != "verified_local"
            or record.checkpoint_format != "native_pt"
            or record.source_format is not None
            or roles != ["actor", "critic"]
            or modes != ["actor_only", "actor_critic"]
            or record.compatibility_status != "transfer_actor_critic"
        ):
            raise SkillLibraryError("trained skill provenance fields are inconsistent")
        identity_material = record.checkpoint_sha256
    else:
        if not _is_sha256(record.manifest_digest):
            raise SkillLibraryError("imported skill requires a manifest SHA-256")
        if (
            not isinstance(record.manifest_schema_version, int)
            or isinstance(record.manifest_schema_version, bool)
            or record.manifest_schema_version < 1
        ):
            raise SkillLibraryError(
                "imported skill requires a positive manifest_schema_version"
            )
        actual_identity = _import_identity_digest(record)
        if record.identity_digest != actual_identity:
            raise SkillLibraryError("imported skill identity digest mismatch")
        identity_material = actual_identity
        if has_checkpoint:
            if (
                record.trust_status != "sanitized"
                or record.checkpoint_format != "server_native_pt"
                or record.source_format != "safetensors"
                or not record.tensor_contract_verified
                or record.compatibility_contract is None
                or record.compatibility_contract_provenance is None
                or not _is_sha256(record.tensor_signature_sha256)
            ):
                raise SkillLibraryError(
                    "imported trainable policy lacks sanitized tensor/contract proof"
                )
        elif (
            record.trust_status != "validated"
            or record.source_format is not None
            or modes != ["reference_only"]
        ):
            raise SkillLibraryError(
                "imported reference-only provenance fields are inconsistent"
            )

    expected_identity = derive_skill_id(
        record.adapter_class, record.task_id, identity_material,
    )
    if record.skill_id != expected_identity:
        raise SkillLibraryError(
            "skill_id does not match the record's immutable identity"
        )


# ── Filesystem helpers (atomic writes, locked publish) ───────────────

def _fsync_dir(d: Path) -> None:
    """fsync a directory's inode so a freshly-renamed dirent is
    durable across power loss / kernel crash. POSIX `os.replace` is
    atomic for the FILE rename but the parent directory's metadata
    isn't persisted until the dir inode itself is fsync'd — without
    this the post-publish state could lose entries on crash recovery
    (code-audit BIGGEST BUG fix). Best-effort: silently skip on
    Windows (where opening a directory isn't supported)."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except (OSError, NotImplementedError):  # pragma: no cover - non-POSIX
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some FS reject dir fsync
        pass
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Tmp-file + rename + parent fsync. POSIX `os.replace` is atomic
    at the filesystem level so concurrent readers either see the OLD
    content or the NEW content, never a half-written file (audit fix
    H2). The parent fsync makes the dirent durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` via tmp + rename + parent fsync. Uses
    `shutil.copy2` to preserve mtime so future provenance audits see
    the original training time, not the publish time. No hardlinks
    (audit fix M3): a hardlinked source can be GC'd by a future
    cleanup pass, leaving the library record pointing at a dangling
    inode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    # fsync the destination file's content before the rename so a
    # post-rename crash can't surface a zero-byte file.
    try:
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
    except OSError:  # pragma: no cover
        pass
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)


# ── SkillLibrary ─────────────────────────────────────────────────────

class SkillLibrary:
    """Filesystem-backed registry of trained policies."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root if root is not None else default_library_root()).resolve()

    # ── Publish ──────────────────────────────────────────────────────
    def publish_from_stage(
        self,
        *,
        stage: Stage,
        mission: Mission,
        adapter_class: str,
        task_id: str,
        robot_slug: Optional[str],
        checkpoint_path: Path,
        final_metric: float,
        source_iter_index: int,
        compatibility_contract: Optional[dict[str, Any]] = None,
        source_reward_path: Optional[Path] = None,
        world_selection_path: Optional[Path] = None,
        world_selection_hash: Optional[str] = None,
        reference_certificate: Optional[Any] = None,
    ) -> SkillRecord:
        """Persist a skill to the library.

        Caller (typically `SkillLibraryHandle.maybe_publish`) is
        responsible for gating: stage must be succeeded, not a
        re-decomposition sub-stage, adapter must support warm-start,
        and the checkpoint file must exist. This method validates the
        ckpt path and writes; gates are caller-side so the handle can
        emit precise `stage_skill_published_skipped(reason=...)`
        events.

        Raises SkillLibraryError on lock timeout / IO failure. Atomic
        on success (tmp+rename for both metadata + checkpoint).
        """
        from filelock import FileLock, Timeout as _FileLockTimeout

        if not checkpoint_path.is_file():
            raise SkillLibraryError(
                f"checkpoint not found: {checkpoint_path}"
            )

        # Resolve all execution-facing provenance BEFORE taking the library
        # lock. The copied bytes are re-hashed after copy below, so a source
        # racing this preparation cannot publish a mixed receipt.
        reference_values: dict[str, Optional[str]] = {
            "reference_clip_id": None,
            "reference_robot": None,
            "reference_sha256": None,
            "reference_provenance_sha256": None,
            "reference_dynamics_certificate_sha256": None,
            "reference_rollout_sha256": None,
            "reference_execution_contract_sha256": None,
            "reference_execution_boundary_sha256": None,
        }
        if stage.reference_clip_id:
            if reference_certificate is None:
                raise SkillLibraryError(
                    "reference-guided stage publication requires a fresh "
                    "Tier-D admission certificate"
                )
            expected_reference = {
                "reference_clip_id": stage.reference_clip_id,
                "reference_robot": stage.reference_robot,
                "reference_sha256": stage.reference_clip_sha256,
                "reference_dynamics_certificate_sha256": (
                    stage.reference_certificate_sha256
                ),
                "reference_execution_contract_sha256": (
                    stage.reference_execution_contract_sha256
                ),
                "reference_execution_boundary_sha256": (
                    stage.reference_execution_boundary_sha256
                ),
            }
            actual_reference = {
                "reference_clip_id": getattr(reference_certificate, "clip_id", None),
                "reference_robot": getattr(reference_certificate, "robot", None),
                "reference_sha256": getattr(
                    reference_certificate, "clip_content_sha256", None,
                ),
                "reference_dynamics_certificate_sha256": getattr(
                    reference_certificate, "certificate_sha256", None,
                ),
                "reference_execution_contract_sha256": getattr(
                    reference_certificate, "execution_contract_sha256", None,
                ),
                "reference_execution_boundary_sha256": getattr(
                    reference_certificate, "execution_boundary_sha256", None,
                ),
            }
            if expected_reference != actual_reference:
                raise SkillLibraryError(
                    "fresh Tier-D certificate disagrees with the stage's exact "
                    "reference admission pins"
                )
            reference_values.update(actual_reference)
            reference_values["reference_rollout_sha256"] = getattr(
                reference_certificate, "rollout_sha256", None,
            )

        provenance_sources: dict[str, tuple[Path, str, str]] = {}
        active_reward_sha256: Optional[str] = None
        mode_manifest_digest: Optional[str] = None
        execution_model = "policy_checkpoint"
        reward_resolved: Optional[Path] = None
        if source_reward_path is not None:
            reward_resolved = Path(source_reward_path).resolve()
            if not reward_resolved.is_file():
                raise SkillLibraryError(
                    f"active reward not found: {reward_resolved}"
                )
            active_reward_sha256 = _file_sha256(reward_resolved)
            reward_source = reward_resolved.read_text(encoding="utf-8")
            try:
                from sculptor.mode_rewards import (
                    mode_execution_manifest_digest,
                    reward_spec_from_source,
                )

                reward_spec = reward_spec_from_source(reward_source)
                manifest = reward_spec.get("mode_execution_manifest")
                if isinstance(manifest, dict):
                    mode_manifest_digest = mode_execution_manifest_digest(manifest)
            except Exception as exc:
                raise SkillLibraryError(
                    "active reward mode provenance is unreadable: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            execution_model = (
                "phase_window_automaton"
                if mode_manifest_digest is not None
                else (
                    "flat_reference_tracking_residual"
                    if stage.reference_clip_id
                    else "reward_sculpting"
                )
            )
            reward_filename = "active_reward" + (reward_resolved.suffix or ".py")
            provenance_sources["active_reward"] = (
                reward_resolved, reward_filename, active_reward_sha256,
            )
        elif stage.reference_clip_id:
            raise SkillLibraryError(
                "reference-guided stage publication requires the exact active "
                "reward bytes"
            )

        world_tuple_hash: Optional[str] = None
        world_selection_sha256: Optional[str] = None
        world_artifact_sha256: dict[str, str] = {}
        if world_selection_path is not None:
            from sculptor.world.artifacts import WorldArtifactStore, file_sha256

            selection_path = Path(world_selection_path).resolve()
            if not selection_path.is_file():
                raise SkillLibraryError(
                    f"pinned world selection not found: {selection_path}"
                )
            store = WorldArtifactStore(selection_path.parent.parent)
            try:
                selection = store.read_selection(selection_path)
            except Exception as exc:
                raise SkillLibraryError(
                    f"pinned world selection is invalid: {exc}"
                ) from exc
            if selection is None:
                raise SkillLibraryError(
                    f"pinned world selection is missing: {selection_path}"
                )
            if world_selection_hash and selection.tuple_hash != world_selection_hash:
                raise SkillLibraryError(
                    "iteration world tuple hash disagrees with its pinned selection"
                )
            world_tuple_hash = selection.tuple_hash
            world_selection_sha256 = file_sha256(selection_path)
            provenance_sources["world_selection"] = (
                selection_path, "world_selection.json", world_selection_sha256,
            )
            for kind, ref in sorted(selection.refs.items()):
                try:
                    artifact_path = store.resolve_ref(ref)
                except Exception as exc:
                    raise SkillLibraryError(
                        f"pinned world artifact {kind!r} is invalid: {exc}"
                    ) from exc
                digest = file_sha256(artifact_path)
                world_artifact_sha256[kind] = digest
                if kind == "reward" and reward_resolved is not None:
                    if (
                        digest != active_reward_sha256
                        or artifact_path.resolve() != reward_resolved
                    ):
                        raise SkillLibraryError(
                            "active reward differs from the reward selected by "
                            "the pinned world tuple"
                        )
                    provenance_sources[f"world:{kind}"] = (
                        reward_resolved,
                        provenance_sources["active_reward"][1],
                        digest,
                    )
                    continue
                suffix = artifact_path.suffix or ".bin"
                filename = f"world_{re.sub(r'[^A-Za-z0-9_.-]+', '_', kind)}{suffix}"
                provenance_sources[f"world:{kind}"] = (
                    artifact_path, filename, digest,
                )
        elif world_selection_hash is not None:
            raise SkillLibraryError(
                "world tuple hash cannot be published without its selection bytes"
            )

        ckpt_sha = _file_sha256(checkpoint_path)
        skill_id = derive_skill_id(adapter_class, task_id, ckpt_sha)
        record_dir = self.root / skill_id
        ckpt_filename = checkpoint_path.name
        _validate_checkpoint_filename(ckpt_filename, allow_empty=False)

        # Per-(adapter, task_id) lock prevents the global library root
        # from being a single-writer choke point (audit Risk 3 fix).
        # Sanitize task_id for use as a path component.
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id) or "untagged"
        safe_adapter = re.sub(r"[^A-Za-z0-9_.-]+", "_", adapter_class) or "noadapter"
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{safe_adapter}__{safe_task}.lock"
        lock = FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S)

        try:
            with lock:
                record_dir.mkdir(parents=True, exist_ok=True)
                ckpt_dst = record_dir / ckpt_filename
                _atomic_copy(checkpoint_path, ckpt_dst)

                provenance_files: dict[str, dict[str, str]] = {}
                copied_by_filename: dict[str, str] = {}
                for role, (source_path, filename, expected_sha) in sorted(
                    provenance_sources.items()
                ):
                    prior = copied_by_filename.get(filename)
                    if prior is None:
                        destination = record_dir / filename
                        _atomic_copy(source_path, destination)
                        copied_sha = _file_sha256(destination)
                        if copied_sha != expected_sha:
                            raise SkillLibraryError(
                                f"provenance source {role!r} changed during "
                                "publication"
                            )
                        copied_by_filename[filename] = copied_sha
                    elif prior != expected_sha:
                        raise SkillLibraryError(
                            f"provenance basename collision for {filename!r}"
                        )
                    provenance_files[role] = {
                        "filename": filename, "sha256": expected_sha,
                    }

                size = ckpt_dst.stat().st_size
                contract_digest = None
                if compatibility_contract is not None:
                    from sculptor.policy_contract import contract_fingerprint

                    contract_digest = contract_fingerprint(
                        compatibility_contract,
                    )

                rec = SkillRecord(
                    schema_version=SCHEMA_VERSION,
                    skill_id=skill_id,
                    adapter_class=adapter_class,
                    task_id=task_id,
                    robot_slug=robot_slug,
                    reward_seed_prompt=stage.reward_seed_prompt,
                    success_criterion=stage.success_criterion,
                    final_metric=float(final_metric),
                    source_iter_index=int(source_iter_index),
                    iterations_used=int(stage.iterations_used),
                    source_mission_goal=mission.goal,
                    source_stage_name=stage.name,
                    created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    checkpoint_filename=ckpt_filename,
                    checkpoint_sha256=ckpt_sha,
                    checkpoint_size_bytes=size,
                    alias=None,
                    compatibility_contract=(
                        dict(compatibility_contract)
                        if compatibility_contract is not None
                        else None
                    ),
                    compatibility_contract_digest=(
                        contract_digest
                    ),
                    reference_clip_id=reference_values["reference_clip_id"],
                    reference_robot=reference_values["reference_robot"],
                    reference_sha256=reference_values["reference_sha256"],
                    reference_provenance_sha256=(
                        reference_values["reference_provenance_sha256"]
                    ),
                    reference_dynamics_certificate_sha256=(
                        reference_values[
                            "reference_dynamics_certificate_sha256"
                        ]
                    ),
                    reference_rollout_sha256=(
                        reference_values["reference_rollout_sha256"]
                    ),
                    reference_execution_contract_sha256=(
                        reference_values[
                            "reference_execution_contract_sha256"
                        ]
                    ),
                    reference_execution_boundary_sha256=(
                        reference_values[
                            "reference_execution_boundary_sha256"
                        ]
                    ),
                    active_reward_sha256=active_reward_sha256,
                    mode_execution_manifest_digest=mode_manifest_digest,
                    world_tuple_hash=world_tuple_hash,
                    world_selection_sha256=world_selection_sha256,
                    world_artifact_sha256=world_artifact_sha256,
                    provenance_files=provenance_files,
                    execution_model=execution_model,
                    mode_reuse_supported=False,
                )
                _validate_skill_record(rec, expected_skill_id=skill_id)
                _atomic_write_text(
                    record_dir / METADATA_FILENAME,
                    json.dumps(rec.to_dict(), indent=2, default=str),
                )
                return rec
        except _FileLockTimeout as e:  # pragma: no cover - timing-sensitive
            raise SkillLibraryError(
                f"timed out acquiring skill-library lock at {lock_path}: {e}"
            ) from e

    # ── Read ─────────────────────────────────────────────────────────
    def load(self, skill_id: str) -> Optional[SkillRecord]:
        """Return the record by id, or None if missing / unreadable.
        A corrupt metadata.json is returned as None (NOT raised) so a
        single rotted entry doesn't break callers iterating the
        library."""
        if not isinstance(skill_id, str) or not _SKILL_ID_RE.fullmatch(skill_id):
            return None
        record_dir = self.root / skill_id
        path = record_dir / METADATA_FILENAME
        if record_dir.is_symlink() or path.is_symlink() or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        try:
            record = SkillRecord.from_dict(data, expected_skill_id=skill_id)
            if record.checkpoint_format != "none":
                self._checkpoint_candidate_for(record, expected_skill_id=skill_id)
            return record
        except (KeyError, TypeError, ValueError, SkillLibraryError):
            return None

    def _checkpoint_candidate_for(
        self,
        record: SkillRecord,
        *,
        expected_skill_id: str,
    ) -> Path:
        _validate_skill_record(record, expected_skill_id=expected_skill_id)
        record_dir = self.root / expected_skill_id
        if record_dir.is_symlink():
            raise SkillLibraryError(
                f"skill directory is a symlink for {expected_skill_id}"
            )
        resolved_dir = record_dir.resolve(strict=False)
        if resolved_dir.parent != self.root:
            raise SkillLibraryError(
                f"skill directory escapes library root for {expected_skill_id}"
            )
        candidate = record_dir / record.checkpoint_filename
        if candidate.is_symlink():
            raise SkillLibraryError(
                f"checkpoint is a symlink for skill {expected_skill_id}"
            )
        resolved_candidate = candidate.resolve(strict=False)
        if resolved_candidate.parent != resolved_dir:
            raise SkillLibraryError(
                f"checkpoint path escapes skill directory for {expected_skill_id}"
            )
        return resolved_candidate

    def checkpoint_path_for(self, record: SkillRecord) -> Path:
        """Resolve and re-attest a checkpoint immediately before use.

        Metadata hashes are not decorative provenance: a library file may be
        edited after import/publish, so every launch-time resolution recomputes
        SHA-256 and refuses a missing, resized, or modified checkpoint. Schema
        v3 execution provenance is re-attested here too, making direct API and
        worker callers as strict as ``SkillLibraryHandle`` mission callers.
        """
        self.verify_execution_provenance(record)
        if not record.checkpoint_filename or not record.checkpoint_sha256:
            raise SkillLibraryError(
                f"skill {record.skill_id} has no trainable checkpoint"
            )
        path = self._checkpoint_candidate_for(
            record, expected_skill_id=record.skill_id,
        )
        if not path.is_file():
            raise SkillLibraryError(f"checkpoint missing for skill {record.skill_id}")
        size = path.stat().st_size
        if size != int(record.checkpoint_size_bytes):
            raise SkillLibraryError(
                f"checkpoint size mismatch for skill {record.skill_id}: "
                f"expected {record.checkpoint_size_bytes}, got {size}"
            )
        actual = _file_sha256(path)
        if actual != record.checkpoint_sha256:
            raise SkillLibraryError(
                f"checkpoint digest mismatch for skill {record.skill_id}: "
                f"expected {record.checkpoint_sha256}, got {actual}"
            )
        return path

    def verify_execution_provenance(self, record: SkillRecord) -> Optional[Any]:
        """Re-attest retained and external training inputs before reuse.

        A warm-start consumes only the checkpoint, but the library must not
        present that checkpoint with stale scientific claims. Locally retained
        reward/world bytes are hashed from the skill directory, and an attached
        Tier-D reference is re-admitted from its canonical library so missing or
        changed clip, rollout, certificate, contract, or boundary bytes reject
        resolution.

        Returns the fresh Tier-D certificate when one is attached.
        """
        _validate_skill_record(record, expected_skill_id=record.skill_id)
        record_dir = self.root / record.skill_id
        resolved_dir = record_dir.resolve(strict=False)
        if record_dir.is_symlink() or resolved_dir.parent != self.root:
            raise SkillLibraryError(
                f"skill provenance directory escapes library root for "
                f"{record.skill_id}"
            )

        for role, descriptor in record.provenance_files.items():
            path = record_dir / descriptor["filename"]
            if path.is_symlink():
                raise SkillLibraryError(
                    f"provenance file {role!r} is a symlink"
                )
            resolved = path.resolve(strict=False)
            if resolved.parent != resolved_dir or not resolved.is_file():
                raise SkillLibraryError(
                    f"provenance file {role!r} is missing or escapes the skill"
                )
            actual = _file_sha256(resolved)
            if actual != descriptor["sha256"]:
                raise SkillLibraryError(
                    f"provenance file {role!r} digest mismatch: expected "
                    f"{descriptor['sha256']}, got {actual}"
                )

        if record.source_training_provenance is not None:
            source_training_file = record.provenance_files["source_training"]
            source_training_path = (
                record_dir / source_training_file["filename"]
            )
            if source_training_path.stat().st_size > 64 * 1024**2:
                raise SkillLibraryError(
                    "retained source training provenance exceeds 64 MiB"
                )
            try:
                from sculptor.tierd_tracker_policy import (
                    TierDTrackerPolicyError,
                    parse_tierd_tracker_policy_origin,
                    validate_tierd_tracker_policy_origin,
                )

                origin = parse_tierd_tracker_policy_origin(
                    source_training_path.read_bytes()
                )
                summary = validate_tierd_tracker_policy_origin(
                    origin,
                    source_contract=record.compatibility_contract,
                    source_checkpoint_sha256=str(
                        record.original_checkpoint_sha256
                    ),
                    portable_actor_safetensors_sha256=str(
                        record.source_weights_sha256
                    ),
                )
            except (OSError, TierDTrackerPolicyError) as exc:
                raise SkillLibraryError(
                    f"retained source training provenance is invalid: {exc}"
                ) from exc
            if summary != record.source_training_provenance:
                raise SkillLibraryError(
                    "retained source training provenance differs from metadata"
                )

        if record.active_reward_sha256 is not None:
            reward_file = record.provenance_files["active_reward"]
            reward_path = record_dir / reward_file["filename"]
            source = reward_path.read_text(encoding="utf-8")
            from sculptor.mode_rewards import (
                mode_execution_manifest_digest,
                reward_spec_from_source,
            )

            spec = reward_spec_from_source(source)
            manifest = spec.get("mode_execution_manifest")
            actual_manifest = (
                mode_execution_manifest_digest(manifest)
                if isinstance(manifest, dict)
                else None
            )
            if actual_manifest != record.mode_execution_manifest_digest:
                raise SkillLibraryError(
                    "retained active reward mode-manifest digest mismatch"
                )

        if record.world_tuple_hash is not None:
            selection_desc = record.provenance_files["world_selection"]
            selection_path = record_dir / selection_desc["filename"]
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                refs = selection["refs"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SkillLibraryError(
                    f"retained world selection is unreadable: {exc}"
                ) from exc
            if not isinstance(refs, dict):
                raise SkillLibraryError("retained world selection refs are invalid")
            from sculptor.world.artifacts import canonical_json_bytes, sha256_bytes

            tuple_payload: dict[str, Any] = {}
            for kind, value in sorted(refs.items()):
                if not isinstance(value, dict):
                    raise SkillLibraryError(
                        f"retained world ref {kind!r} is invalid"
                    )
                tuple_payload[kind] = value
                expected = record.world_artifact_sha256.get(kind)
                if expected is None or value.get("sha256") != expected:
                    raise SkillLibraryError(
                        f"retained world ref {kind!r} differs from metadata"
                    )
                retained = record.provenance_files.get(f"world:{kind}")
                if retained is None or retained["sha256"] != expected:
                    raise SkillLibraryError(
                        f"retained world bytes for {kind!r} are missing"
                    )
            actual_tuple = sha256_bytes(canonical_json_bytes(tuple_payload))
            if (
                selection.get("tuple_hash") != record.world_tuple_hash
                or actual_tuple != record.world_tuple_hash
                or set(refs) != set(record.world_artifact_sha256)
            ):
                raise SkillLibraryError("retained world tuple identity mismatch")

        certificate = None
        if record.source == "trained" and record.reference_clip_id:
            from sculptor.refs.track import require_tierd_admission

            try:
                certificate = require_tierd_admission(
                    str(record.reference_robot),
                    str(record.reference_clip_id),
                    expected_clip_sha256=record.reference_sha256,
                    expected_certificate_sha256=(
                        record.reference_dynamics_certificate_sha256
                    ),
                    expected_rollout_sha256=record.reference_rollout_sha256,
                    expected_execution_contract_sha256=(
                        record.reference_execution_contract_sha256
                    ),
                    expected_execution_boundary_sha256=(
                        record.reference_execution_boundary_sha256
                    ),
                )
            except Exception as exc:
                raise SkillLibraryError(
                    "reference-guided skill provenance is no longer valid: "
                    f"{exc}"
                ) from exc
        return certificate

    def publish_imported_checkpoint(
        self,
        *,
        checkpoint_path: Optional[Path],
        adapter_class: str,
        task_id: str,
        robot_slug: Optional[str],
        alias: Optional[str],
        manifest_digest: str,
        manifest_schema_version: int,
        original_checkpoint_sha256: Optional[str],
        source_weights_sha256: Optional[str],
        reference_clip_id: Optional[str],
        reference_robot: Optional[str],
        reference_sha256: Optional[str],
        reference_provenance_sha256: Optional[str],
        world_bundle_sha256: Optional[str],
        compatibility_contract: Optional[dict[str, Any]],
        compatibility_contract_digest: Optional[str],
        compatibility_contract_provenance: Optional[dict[str, Any]] = None,
        compatibility_contract_provenance_digest: Optional[str] = None,
        compatibility_contract_provenance_status: Optional[str] = None,
        compatibility_provenance_sources: Optional[dict[str, Path]] = None,
        source_training_provenance: Optional[dict[str, Any]] = None,
        source_training_provenance_sha256: Optional[str] = None,
        source_training_provenance_source: Optional[Path] = None,
        tensor_contract_verified: bool = False,
        tensor_signature_sha256: Optional[str] = None,
        compatibility_status: str,
        initialization_modes: list[str],
        policy_roles: list[str],
        controller_kind: Optional[str],
        controller_sha256: Optional[str] = None,
        bundled_world: bool,
        warnings: list[str],
        source_project: str = "imported bundle",
        source_iter_index: int = -1,
        publication_state: Optional[dict[str, bool]] = None,
    ) -> SkillRecord:
        """Atomically publish a server-produced checkpoint from an import.

        The caller must already have converted a data-only format into the
        native checkpoint.  This method deliberately accepts a filesystem
        path, not uploaded bytes, so untrusted deserialization cannot leak
        into the registry layer.
        """
        from filelock import FileLock, Timeout as _FileLockTimeout

        if publication_state is not None:
            publication_state["created"] = False

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                raise SkillLibraryError(
                    f"sanitized checkpoint not found: {checkpoint_path}"
                )
            ckpt_sha = _file_sha256(checkpoint_path)
        else:
            ckpt_sha = ""
        source_training_values = (
            source_training_provenance,
            source_training_provenance_sha256,
            source_training_provenance_source,
        )
        if any(value is not None for value in source_training_values):
            if not all(value is not None for value in source_training_values):
                raise SkillLibraryError(
                    "source training provenance object, digest, and bytes are "
                    "required together"
                )
            assert source_training_provenance is not None
            assert source_training_provenance_sha256 is not None
            assert source_training_provenance_source is not None
            source_training_provenance_source = Path(
                source_training_provenance_source
            )
            if (
                not source_training_provenance_source.is_file()
                or source_training_provenance_source.is_symlink()
                or not _is_sha256(source_training_provenance_sha256)
                or _file_sha256(source_training_provenance_source)
                != source_training_provenance_sha256
                or compatibility_contract is None
                or original_checkpoint_sha256 is None
            ):
                raise SkillLibraryError(
                    "source training provenance bytes or binding are invalid"
                )
            try:
                from sculptor.tierd_tracker_policy import (
                    TierDTrackerPolicyError,
                    validate_tierd_tracker_policy_summary,
                )

                source_training_provenance = (
                    validate_tierd_tracker_policy_summary(
                        source_training_provenance,
                        source_contract=compatibility_contract,
                        source_checkpoint_sha256=original_checkpoint_sha256,
                        portable_actor_safetensors_sha256=str(
                            source_weights_sha256
                        ),
                    )
                )
            except TierDTrackerPolicyError as exc:
                raise SkillLibraryError(
                    f"source training provenance summary is invalid: {exc}"
                ) from exc
        identity_payload = {
            "manifest_digest": manifest_digest,
            "manifest_schema_version": int(manifest_schema_version),
            "source_weights_sha256": source_weights_sha256,
            "reference_clip_id": reference_clip_id,
            "reference_robot": reference_robot,
            "reference_sha256": reference_sha256,
            "reference_provenance_sha256": reference_provenance_sha256,
            "world_bundle_sha256": world_bundle_sha256,
            "controller_sha256": controller_sha256,
            "compatibility_contract_digest": compatibility_contract_digest,
            "compatibility_contract_provenance_digest": (
                compatibility_contract_provenance_digest
            ),
            "compatibility_contract_provenance_status": (
                compatibility_contract_provenance_status
            ),
            "tensor_signature_sha256": tensor_signature_sha256,
            "compatibility_status": compatibility_status,
            "initialization_modes": initialization_modes,
            "policy_roles": policy_roles,
        }
        if source_training_provenance is not None:
            identity_payload["source_training_provenance"] = (
                source_training_provenance
            )
            identity_payload["source_training_provenance_sha256"] = (
                source_training_provenance_sha256
            )
        identity_digest = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        skill_id = derive_skill_id(adapter_class, task_id, identity_digest)
        record_dir = self.root / skill_id
        ckpt_filename = "checkpoint.pt"
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id) or "untagged"
        safe_adapter = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", adapter_class) or "noadapter"
        )
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{safe_adapter}__{safe_task}.lock"
        lock = FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        try:
            with lock:
                metadata_path = record_dir / METADATA_FILENAME
                if metadata_path.is_file():
                    try:
                        existing = SkillRecord.from_dict(
                            json.loads(metadata_path.read_text(encoding="utf-8")),
                            expected_skill_id=skill_id,
                        )
                    except Exception as exc:
                        raise SkillLibraryError(
                            f"skill-id collision with unreadable metadata at "
                            f"{metadata_path}"
                        ) from exc
                    if existing.identity_digest != identity_digest:
                        raise SkillLibraryError(
                            f"immutable skill-id collision for {skill_id}: "
                            "the admitted manifest/components differ"
                        )
                    if checkpoint_path is not None:
                        self.checkpoint_path_for(existing)
                    elif existing.checkpoint_filename or existing.checkpoint_sha256:
                        raise SkillLibraryError(
                            f"immutable skill-id collision for {skill_id}: "
                            "checkpoint presence differs"
                        )
                    return existing
                if record_dir.exists():
                    raise SkillLibraryError(
                        f"refusing to overwrite incomplete/colliding skill "
                        f"directory {record_dir}"
                    )
                stage_dir = Path(tempfile.mkdtemp(
                    prefix=f".{skill_id}.", dir=self.root,
                ))
                try:
                    dst = stage_dir / ckpt_filename
                    if checkpoint_path is not None:
                        _atomic_copy(checkpoint_path, dst)
                    retained_provenance: dict[str, dict[str, str]] = {}
                    for role, source_path in sorted(
                        (compatibility_provenance_sources or {}).items()
                    ):
                        safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)
                        suffix = source_path.suffix or ".bin"
                        filename = f"contract_provenance_{safe_role}{suffix}"
                        _atomic_copy(source_path, stage_dir / filename)
                        retained_provenance[
                            f"compatibility_contract:{role}"
                        ] = {
                            "filename": filename,
                            "sha256": _file_sha256(stage_dir / filename),
                        }
                    if source_training_provenance_source is not None:
                        source_training_filename = "tierd_tracker_origin.json"
                        source_training_destination = (
                            stage_dir / source_training_filename
                        )
                        _atomic_copy(
                            source_training_provenance_source,
                            source_training_destination,
                        )
                        retained_source_sha = _file_sha256(
                            source_training_destination
                        )
                        if retained_source_sha != source_training_provenance_sha256:
                            raise SkillLibraryError(
                                "retained source training provenance digest mismatch"
                            )
                        retained_provenance["source_training"] = {
                            "filename": source_training_filename,
                            "sha256": retained_source_sha,
                        }
                    rec = SkillRecord(
                        schema_version=SCHEMA_VERSION,
                        skill_id=skill_id,
                        adapter_class=adapter_class,
                        task_id=task_id,
                        robot_slug=robot_slug,
                        reward_seed_prompt="",
                        success_criterion="",
                        final_metric=0.0,
                        source_iter_index=int(source_iter_index),
                        iterations_used=0,
                        source_mission_goal=source_project,
                        source_stage_name="import",
                        created_at=now,
                        checkpoint_filename=(
                            ckpt_filename if checkpoint_path is not None else ""
                        ),
                        checkpoint_sha256=ckpt_sha,
                        checkpoint_size_bytes=(
                            dst.stat().st_size if checkpoint_path is not None else 0
                        ),
                        alias=alias,
                        source="imported_bundle",
                        trust_status=(
                            "sanitized" if checkpoint_path is not None else "validated"
                        ),
                        checkpoint_format=(
                            "server_native_pt" if checkpoint_path is not None else "none"
                        ),
                        source_format=(
                            "safetensors" if checkpoint_path is not None else None
                        ),
                        identity_digest=identity_digest,
                        manifest_digest=manifest_digest,
                        manifest_schema_version=int(manifest_schema_version),
                        original_checkpoint_sha256=original_checkpoint_sha256,
                        source_weights_sha256=source_weights_sha256,
                        source_training_provenance=(
                            dict(source_training_provenance)
                            if source_training_provenance is not None
                            else None
                        ),
                        source_training_provenance_sha256=(
                            source_training_provenance_sha256
                        ),
                        reference_clip_id=reference_clip_id,
                        reference_robot=reference_robot,
                        reference_sha256=reference_sha256,
                        reference_provenance_sha256=reference_provenance_sha256,
                        world_bundle_sha256=world_bundle_sha256,
                        compatibility_contract=(
                            dict(compatibility_contract)
                            if compatibility_contract is not None
                            else None
                        ),
                        compatibility_contract_digest=compatibility_contract_digest,
                        compatibility_contract_provenance=(
                            dict(compatibility_contract_provenance)
                            if compatibility_contract_provenance is not None
                            else None
                        ),
                        compatibility_contract_provenance_digest=(
                            compatibility_contract_provenance_digest
                        ),
                        compatibility_contract_provenance_status=(
                            compatibility_contract_provenance_status
                        ),
                        provenance_files=retained_provenance,
                        tensor_contract_verified=bool(tensor_contract_verified),
                        tensor_signature_sha256=tensor_signature_sha256,
                        compatibility_status=compatibility_status,
                        initialization_modes=list(initialization_modes),
                        policy_roles=list(policy_roles),
                        controller_kind=controller_kind,
                        controller_sha256=controller_sha256,
                        bundled_world=bool(bundled_world),
                        import_warnings=list(warnings),
                        imported_at=now,
                    )
                    _validate_skill_record(rec, expected_skill_id=skill_id)
                    _atomic_write_text(
                        stage_dir / METADATA_FILENAME,
                        json.dumps(rec.to_dict(), indent=2, default=str),
                    )
                    os.replace(stage_dir, record_dir)
                    if publication_state is not None:
                        publication_state["created"] = True
                except BaseException:
                    shutil.rmtree(stage_dir, ignore_errors=True)
                    raise
                return rec
        except _FileLockTimeout as exc:  # pragma: no cover - timing-sensitive
            raise SkillLibraryError(
                f"timed out acquiring skill-library lock at {lock_path}: {exc}"
            ) from exc

    def rollback_imported_publication(self, record: SkillRecord) -> None:
        """Remove only a record created by the current failed admission.

        The caller tracks whether publication created a new directory; an
        idempotently reused record must never be removed.  Revalidation under
        the same per-target lock prevents deleting a changed or colliding
        record if another process touched it meanwhile.
        """
        from filelock import FileLock, Timeout as _FileLockTimeout

        if record.source != "imported_bundle":
            raise SkillLibraryError("only imported records can be rolled back")
        _validate_skill_record(record, expected_skill_id=record.skill_id)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.task_id) or "untagged"
        safe_adapter = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", record.adapter_class)
            or "noadapter"
        )
        lock_path = (
            self.root / ".locks" / f"{safe_adapter}__{safe_task}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_S):
                record_dir = self.root / record.skill_id
                if not record_dir.exists():
                    return
                current = self.load(record.skill_id)
                if (
                    current is None
                    or current.identity_digest != record.identity_digest
                    or current.manifest_digest != record.manifest_digest
                ):
                    raise SkillLibraryError(
                        "refusing to roll back a changed imported skill record"
                    )
                shutil.rmtree(record_dir)
                _fsync_dir(self.root)
        except _FileLockTimeout as exc:  # pragma: no cover - timing-sensitive
            raise SkillLibraryError(
                f"timed out acquiring rollback lock at {lock_path}: {exc}"
            ) from exc

    def __iter__(self) -> Iterator[SkillRecord]:
        return self._iter_records()

    def _iter_records(self) -> Iterator[SkillRecord]:
        # Guard at the iterator level (not just `__iter__`) — this
        # method is also called directly from `list_compatible`. A
        # missing root is the normal pre-publish state, NOT an error:
        # an empty library should be iterable and yield zero records.
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name.startswith("."):
                # `.locks/` and any future hidden dirs.
                continue
            if not _SKILL_ID_RE.fullmatch(child.name):
                continue
            meta = child / METADATA_FILENAME
            if meta.is_symlink() or not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                yield SkillRecord.from_dict(
                    data, expected_skill_id=child.name,
                )
            except (
                json.JSONDecodeError,
                OSError,
                KeyError,
                TypeError,
                ValueError,
                SkillLibraryError,
            ):
                # audit H2: skip corrupt entries instead of erroring
                # the whole listing.
                continue

    def list_compatible(
        self,
        *,
        adapter_class: str,
        task_id: str,
        robot_slug: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SkillRecord]:
        """Return up to `top_k` records matching `(adapter_class,
        task_id)`, ordered by `final_metric` descending. When
        `robot_slug` is provided, also requires equality (None on
        the record matches anything to ease pre-Ship-19 records).
        """
        out: list[SkillRecord] = []
        for r in self._iter_records():
            if r.adapter_class != adapter_class:
                continue
            if r.task_id != task_id:
                continue
            if robot_slug is not None and r.robot_slug not in (None, robot_slug):
                continue
            out.append(r)
        out.sort(key=lambda r: r.final_metric, reverse=True)
        return out[:top_k]


# ── SkillLibraryHandle ───────────────────────────────────────────────

@dataclass
class SkillLibraryHandle:
    """Per-call-site context that bundles the (library, identity, knobs)
    tuple a caller would otherwise pipe through `mission_run` /
    `decompose_task` as five separate kwargs (audit fix H3).

    Construction:
        handle = SkillLibraryHandle(
            library=SkillLibrary(),
            adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
            task_id="Mjlab-Velocity-Flat-Unitree-Go1",
            robot_slug="g1_humanoid",  # optional
            publish=True,              # default: yes, contribute to library
        )
        mission = decompose_task(goal, contract, skill_library_handle=handle)
        result = mission_run(mission, adapter_short_name="mjlab",
                             skill_library_handle=handle)
    """

    library: SkillLibrary
    adapter_class: str
    task_id: str
    robot_slug: Optional[str] = None
    publish: bool = True
    compatibility_contract: Optional[dict[str, Any]] = None
    strict_runtime_admission: bool = False

    def _exact_import_target(self):
        from sculptor.skill_bundle import ImportTarget

        return ImportTarget(
            adapter_class=self.adapter_class,
            task_id=self.task_id,
            robot_slug=self.robot_slug,
            compatibility_contract=self.compatibility_contract,
            policy_contract_error=(
                None
                if self.compatibility_contract is not None
                else "project policy contract is unavailable"
            ),
        )

    # ── decompose-time helpers ───────────────────────────────────────
    def list_for_decompose(self, *, top_k: int = 5) -> list[SkillRecord]:
        records = self.library.list_compatible(
            adapter_class=self.adapter_class,
            task_id=self.task_id,
            robot_slug=self.robot_slug,
            top_k=top_k,
        )
        if not self.strict_runtime_admission:
            return records
        from sculptor.skill_bundle import compatibility_for

        target = self._exact_import_target()
        return [
            record
            for record in records
            if record.robot_slug == self.robot_slug
            and "actor_only" in compatibility_for(
                record, target,
            )["allowed_initialization_modes"]
        ]

    # ── runtime helpers ──────────────────────────────────────────────
    def maybe_load_for_stage(
        self,
        stage: Stage,
        emit: Callable[[dict[str, Any]], None],
    ) -> Optional[Path]:
        """Resolve `stage.init_skill_id` to a checkpoint path, or
        emit a `skill_warm_start_skipped` event with a reason and
        return None.

        Reason taxonomy:
          - `"no_init_skill_id"`       — stage didn't set one (silent;
            handled by caller, not emitted)
          - `"library_unavailable"`    — handle has no library (silent;
            same reason)
          - `"skill_not_found"`        — id was set but missing from lib
          - `"adapter_class_mismatch"` — id resolves but recorded
            adapter doesn't match the handle's
          - `"task_id_mismatch"`       — same idea for task_id
          - `"checkpoint_missing"`     — record exists but the .pt /
            .zip file is gone
        """
        if stage.init_skill_id is None:
            return None
        record = self.library.load(stage.init_skill_id)
        if record is None:
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "skill_not_found",
            })
            if self.strict_runtime_admission:
                raise SkillLibraryError(
                    f"explicit mission skill {stage.init_skill_id!r} is missing"
                )
            return None
        if record.adapter_class != self.adapter_class:
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "adapter_class_mismatch",
                "expected": self.adapter_class,
                "actual": record.adapter_class,
            })
            if self.strict_runtime_admission:
                raise SkillLibraryError("explicit mission skill adapter differs")
            return None
        if record.task_id != self.task_id:
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "task_id_mismatch",
                "expected": self.task_id,
                "actual": record.task_id,
            })
            if self.strict_runtime_admission:
                raise SkillLibraryError("explicit mission skill task differs")
            return None
        try:
            reference_certificate = self.library.verify_execution_provenance(
                record,
            )
            if reference_certificate is not None and self.strict_runtime_admission:
                from sculptor.refs.track import require_tierd_target_compatibility

                require_tierd_target_compatibility(
                    reference_certificate,
                    Path("."),
                    target_robot=str(self.robot_slug or ""),
                    target_policy_contract=self.compatibility_contract,
                )
        except Exception as exc:
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "execution_provenance_invalid",
                "detail": str(exc),
            })
            if self.strict_runtime_admission:
                if isinstance(exc, SkillLibraryError):
                    raise
                raise SkillLibraryError(str(exc)) from exc
            return None
        if self.strict_runtime_admission:
            from sculptor.skill_bundle import compatibility_for

            compatibility = compatibility_for(
                record, self._exact_import_target(),
            )
            if (
                record.robot_slug != self.robot_slug
                or "actor_only"
                not in compatibility["allowed_initialization_modes"]
            ):
                emit({
                    "type": "skill_warm_start_skipped",
                    "stage_name": stage.name,
                    "skill_id": stage.init_skill_id,
                    "reason": "exact_contract_mismatch",
                    "compatibility": compatibility,
                })
                raise SkillLibraryError(
                    "explicit mission skill failed exact robot, observation, "
                    "action, network, timing, or software compatibility"
                )
        try:
            ckpt = self.library.checkpoint_path_for(record)
        except SkillLibraryError as exc:
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "checkpoint_integrity_failed",
                "detail": str(exc),
            })
            if self.strict_runtime_admission:
                raise
            return None
        if self.strict_runtime_admission:
            emit({
                "type": "skill_warm_start_admitted",
                "stage_name": stage.name,
                "skill_id": record.skill_id,
                "initialization_mode": "actor_only",
                "checkpoint_sha256": record.checkpoint_sha256,
                "compatibility_contract_digest": (
                    record.compatibility_contract_digest
                ),
                "execution_model": record.execution_model,
                "mode_reuse_supported": record.mode_reuse_supported,
                "reference_robot": record.reference_robot,
                "reference_clip_id": record.reference_clip_id,
                "reference_sha256": record.reference_sha256,
                "reference_dynamics_certificate_sha256": (
                    record.reference_dynamics_certificate_sha256
                ),
                "reference_execution_contract_sha256": (
                    record.reference_execution_contract_sha256
                ),
                "reference_execution_boundary_sha256": (
                    record.reference_execution_boundary_sha256
                ),
                "active_reward_sha256": record.active_reward_sha256,
                "mode_execution_manifest_digest": (
                    record.mode_execution_manifest_digest
                ),
                "world_tuple_hash": record.world_tuple_hash,
            })
        return ckpt

    def maybe_publish(
        self,
        *,
        stage: Stage,
        mission: Mission,
        adapter: Any,
        sculpt_result: Any,
        emit: Callable[[dict[str, Any]], None],
    ) -> Optional[SkillRecord]:
        """Publish `stage` to the library if all gates pass.

        Gates (each emits a precise skip reason on failure):
          - `publish=False` on this handle
          - stage.status != "succeeded"
          - stage.redecomposition_attempts > 0  (Ship 17 sub-stages
            are by construction specialized to slices that the parent
            couldn't handle in one stage; their reward shape is a
            poor seed for OTHER missions — audit fix H1)
          - adapter doesn't accept `init_policy_path` (no future
            mission could load this skill anyway — audit BIGGEST HOLE
            mitigation)
          - sculpt_result.primary_metric_history is empty (training
            never produced a metric)
        Emits `stage_skill_publish_skipped(reason=...)` on failure.

        On success, copies the BEST-iter checkpoint (audit fix C2)
        and emits `stage_skill_published`. Returns the record.
        """
        if not self.publish:
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "handle_publish_disabled",
            })
            return None
        if stage.status != "succeeded":
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "stage_not_succeeded",
                "status": stage.status,
            })
            return None
        if stage.redecomposition_attempts > 0:
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "redecomposition_artifact",
                "redecomposition_attempts": stage.redecomposition_attempts,
            })
            return None
        if not adapter_supports_warm_start(adapter):
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "adapter_does_not_support_warm_start",
                "adapter_class": self.adapter_class,
            })
            return None

        history = list(getattr(sculpt_result, "primary_metric_history", []) or [])
        # Filter out None / non-finite for argmax; keep index alignment.
        finite = [
            (i, v) for i, v in enumerate(history)
            if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf"))
        ]
        if not finite:
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "no_metric_history",
            })
            return None
        best_idx, best_metric = max(finite, key=lambda t: t[1])

        completed = list(getattr(sculpt_result, "completed_iters", []) or [])
        if best_idx >= len(completed):
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "best_iter_not_in_completed",
                "best_idx": best_idx,
                "completed_iters": len(completed),
            })
            return None
        best_iter = completed[best_idx]
        best_iter_dir = Path(getattr(best_iter, "iter_dir"))
        ckpt: Optional[Path] = None
        for name in ("checkpoint.pt", "checkpoint.zip"):
            candidate = best_iter_dir / name
            if candidate.is_file():
                ckpt = candidate
                break
        if ckpt is None:
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "best_iter_checkpoint_missing",
                "best_iter_dir": str(best_iter_dir),
            })
            return None

        reference_certificate = None
        if stage.reference_clip_id:
            try:
                from sculptor.refs.track import (
                    require_stage_tierd_admission,
                    require_tierd_target_compatibility,
                )

                reference_certificate = require_stage_tierd_admission(
                    stage,
                    expected_robot=(
                        str(self.robot_slug) if self.robot_slug else None
                    ),
                )
                if reference_certificate is None:  # pragma: no cover - guard
                    raise SkillLibraryError(
                        "attached reference produced no Tier-D certificate"
                    )
                if self.strict_runtime_admission:
                    if not mission.mission_dir:
                        raise SkillLibraryError(
                            "reference-guided publication requires mission_dir"
                        )
                    project_root = Path(mission.mission_dir).parent.parent
                    require_tierd_target_compatibility(
                        reference_certificate,
                        project_root,
                        target_robot=str(self.robot_slug or ""),
                        target_policy_contract=self.compatibility_contract,
                    )
            except Exception as exc:
                emit({
                    "type": "stage_skill_publish_skipped",
                    "stage_name": stage.name,
                    "reason": "reference_provenance_invalid",
                    "detail": str(exc),
                })
                return None

        source_reward_path = getattr(best_iter, "reward_path_trained", None)
        if source_reward_path is None:
            source_reward_path = getattr(best_iter, "reward_path_before", None)
        world_selection_path = getattr(best_iter, "world_selection_path", None)
        world_selection_hash = getattr(best_iter, "world_selection_hash", None)

        try:
            rec = self.library.publish_from_stage(
                stage=stage,
                mission=mission,
                adapter_class=self.adapter_class,
                task_id=self.task_id,
                robot_slug=self.robot_slug,
                checkpoint_path=ckpt,
                final_metric=best_metric,
                source_iter_index=best_idx,
                compatibility_contract=self.compatibility_contract,
                source_reward_path=(
                    Path(source_reward_path)
                    if source_reward_path is not None else None
                ),
                world_selection_path=(
                    Path(world_selection_path)
                    if world_selection_path is not None else None
                ),
                world_selection_hash=world_selection_hash,
                reference_certificate=reference_certificate,
            )
        except SkillLibraryError as e:
            emit({
                "type": "stage_skill_publish_skipped",
                "stage_name": stage.name,
                "reason": "library_error",
                "detail": str(e),
            })
            return None

        emit({
            "type": "stage_skill_published",
            "stage_name": stage.name,
            "skill_id": rec.skill_id,
            "adapter_class": rec.adapter_class,
            "task_id": rec.task_id,
            "final_metric": rec.final_metric,
            "source_iter_index": rec.source_iter_index,
            "checkpoint_size_bytes": rec.checkpoint_size_bytes,
            "execution_model": rec.execution_model,
            "mode_reuse_supported": rec.mode_reuse_supported,
            "reference_robot": rec.reference_robot,
            "reference_clip_id": rec.reference_clip_id,
            "reference_sha256": rec.reference_sha256,
            "reference_dynamics_certificate_sha256": (
                rec.reference_dynamics_certificate_sha256
            ),
            "reference_rollout_sha256": rec.reference_rollout_sha256,
            "reference_execution_contract_sha256": (
                rec.reference_execution_contract_sha256
            ),
            "reference_execution_boundary_sha256": (
                rec.reference_execution_boundary_sha256
            ),
            "active_reward_sha256": rec.active_reward_sha256,
            "mode_execution_manifest_digest": (
                rec.mode_execution_manifest_digest
            ),
            "world_tuple_hash": rec.world_tuple_hash,
            "world_selection_sha256": rec.world_selection_sha256,
        })
        return rec


__all__ = [
    "SCHEMA_VERSION",
    "ENV_LIBRARY_ROOT",
    "METADATA_FILENAME",
    "SkillLibrary",
    "SkillLibraryError",
    "SkillLibraryHandle",
    "SkillRecord",
    "adapter_supports_warm_start",
    "default_library_root",
    "derive_skill_id",
]
