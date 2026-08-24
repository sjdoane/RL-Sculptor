"""On-disk reference library: root location, provenance schema,
`index.jsonl` cache, clip_id rules, content hashing, license guard.

§R1_BUILD_SPEC decisions 3-4. The index is a REBUILDABLE cache —
`provenance.json` beside each clip is the single source of truth
(`rebuild_index` rescans disk and never trusts a stale index row).
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from sculptor.project_robot import validate_robot_namespace

#: Provenance schema version.  Schema 2 makes ``content_sha256`` the
#: immutable identity of the bytes that execution actually consumes
#: (``clip.npz``).  Schema 1 overloaded that field with a downloaded source
#: hash, a partial array hash, or a retarget input hash depending on the
#: producer.  ``source_content_sha256`` now carries raw-source lineage
#: independently.
PROVENANCE_SCHEMA = 2
SUPPORTED_PROVENANCE_SCHEMAS = frozenset({1, PROVENANCE_SCHEMA})

#: clip_id charset (§decision 4): lowercase alnum + `_-`, must start
#: alnum, max 96 chars total.
CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")

INDEX_FILENAME = "index.jsonl"
REJECTS_FILENAME = "index_rejects.jsonl"
CLIP_FILENAME = "clip.npz"
PROVENANCE_FILENAME = "provenance.json"
PREVIEW_FILENAME = "preview.png"

ROOT_FRAME_DECLARATION_EVIDENCE_SCHEMA = (
    "reward-sculptor-root-frame-declaration-evidence-v1"
)
ROOT_FRAME_DECLARATION_ASSERTION_VERSION = 1
ROOT_FRAME_DECLARATION_EVIDENCE_METHODS = frozenset(
    {
        "visual_inspection",
        "source_documentation",
        "deterministic_export_contract",
    }
)
ROOT_FRAME_INHERITANCE_SCHEMA = (
    "reward-sculptor-root-frame-inheritance-v1"
)
ROOT_FRAME_INHERITANCE_METHOD = (
    "unanimous_parent_contract_preserved_by_se2_compose_v1"
)
ROOT_FRAME_EMBEDDED_EVIDENCE_SCHEMA = (
    "reward-sculptor-embedded-root-frame-evidence-v1"
)

# Keep the existing transfer-import lock filename so Tier-D certification and
# portable-reference installation coordinate across processes.  The name is a
# compatibility detail; the authority is now the whole reference library, not
# only ``.rskill`` imports.
REFERENCE_LIBRARY_MUTATION_LOCK_FILENAME = ".rskill-import.lock"
REFERENCE_LIBRARY_MUTATION_LOCK_TIMEOUT_S = 10.0

#: Slim index row columns (§decision 4), in this fixed order.
INDEX_COLUMNS = (
    "clip_id", "robot", "text", "labels", "tier", "license",
    "n_frames", "fps", "duration_s", "root_z_range", "has_preview",
)


class LicenseGuardError(ValueError):
    """Raised when a caller tries to index/persist a clip without a
    provenance record — library clips MUST carry attribution+license."""


class ClipIdError(ValueError):
    """Raised when a clip_id fails the §decision-4 charset/length rule."""


class ArtifactMaterializationError(ValueError):
    """Raised when an immutable derived reference cannot be materialized."""


@contextmanager
def reference_library_mutation_lock(
    *, root: Optional[Path] = None,
) -> Iterator[None]:
    """Serialize a reference-library mutation across processes.

    ``index.jsonl`` is one library-wide cache, so a per-clip lock is not
    sufficient: a slow rebuild for clip A can otherwise publish a scan taken
    before clip B committed.  Tier-D promotion/demotion holds this lock across
    its provenance read-modify-write *and* the subsequent global rebuild.

    The lock path intentionally matches the historical portable-import lock,
    which makes certification and transfer installation participate in the
    same transaction domain without changing retained artifact identities.
    """
    from filelock import FileLock

    library_root = Path(root or references_root()).expanduser()
    library_root.mkdir(parents=True, exist_ok=True)
    library_root = library_root.resolve()
    lock = FileLock(
        str(library_root / REFERENCE_LIBRARY_MUTATION_LOCK_FILENAME),
        timeout=REFERENCE_LIBRARY_MUTATION_LOCK_TIMEOUT_S,
    )
    with lock:
        yield


def _materialization_parent_metadata_issues(prov: Any) -> list[str]:
    """Return strict type errors for metadata copied into a derived record."""
    if not isinstance(prov, dict):
        return ["source provenance must be a JSON object"]
    issues: list[str] = []
    for field_name in ("license", "attribution"):
        value = prov.get(field_name)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"source provenance.{field_name} must be non-empty text")
    source = prov.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("kind"), str)
        or not source["kind"].strip()
    ):
        issues.append("source provenance.source must name a non-empty kind")
    for field_name in ("retarget", "qc", "joint_mapping"):
        value = prov.get(field_name)
        if value is not None and not isinstance(value, dict):
            issues.append(f"source provenance.{field_name} must be an object")
    frame_range = prov.get("frame_range")
    if frame_range is not None and (
        not isinstance(frame_range, list)
        or len(frame_range) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in frame_range
        )
        or frame_range[0] < 0
        or frame_range[1] < frame_range[0]
    ):
        issues.append(
            "source provenance.frame_range must be [nonnegative_start, end]"
        )
    labels = prov.get("labels")
    if labels is not None and (
        not isinstance(labels, list)
        or any(not isinstance(item, str) for item in labels)
    ):
        issues.append("source provenance.labels must be a list of text values")
    text_value = prov.get("text")
    if text_value is not None and not isinstance(text_value, str):
        issues.append("source provenance.text must be text")
    fps_source = prov.get("fps_source")
    if fps_source is not None and (
        not isinstance(fps_source, (int, float))
        or isinstance(fps_source, bool)
        or not np.isfinite(float(fps_source))
        or float(fps_source) <= 0.0
    ):
        issues.append("source provenance.fps_source must be positive")
    parent_clip_id = prov.get("parent_clip_id")
    if parent_clip_id is not None:
        try:
            validate_clip_id(parent_clip_id)
        except (ClipIdError, TypeError):
            issues.append("source provenance.parent_clip_id is invalid")
    source_sha = prov.get("source_content_sha256")
    if source_sha is not None and (
        not isinstance(source_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None
    ):
        issues.append("source provenance.source_content_sha256 is invalid")
    return issues


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    required: bool,
) -> Optional[bytes]:
    """Read one non-link regular file relative to an already-pinned dirfd."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:  # pragma: no cover - production executes on Linux
        raise OSError(
            errno.ENOTSUP,
            "no-follow file access is unavailable on this platform",
        )
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | nofollow,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        if not required:
            return None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, f"{name} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _capture_source_artifact_snapshot(
    library_root: Path,
    *,
    robot: str,
    source_clip_id: str,
) -> tuple[bytes, bytes, Optional[bytes]]:
    """Capture provenance, clip, and optional preview from pinned directories.

    Path-level ``is_symlink`` checks are useful diagnostics but are not an
    authority: an attacker can exchange a checked directory before its files
    are opened.  Pin every directory with ``O_NOFOLLOW`` and read members
    relative to those handles so a rename/symlink swap cannot redirect any
    part of the admitted snapshot outside the configured library root.
    """
    if os.name != "posix":  # pragma: no cover - production executes on Linux
        raise OSError(
            errno.ENOTSUP,
            "secure reference snapshot capture requires POSIX dirfd support",
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:  # pragma: no cover - old POSIX runtime
        raise OSError(
            errno.ENOTSUP,
            "secure no-follow directory access is unavailable",
        )
    flags = os.O_RDONLY | directory | nofollow
    root_fd = os.open(str(library_root), flags)
    try:
        robot_fd = os.open(robot, flags, dir_fd=root_fd)
        try:
            source_fd = os.open(source_clip_id, flags, dir_fd=robot_fd)
            try:
                provenance_bytes = _read_regular_file_at(
                    source_fd, PROVENANCE_FILENAME, required=True,
                )
                clip_bytes = _read_regular_file_at(
                    source_fd, CLIP_FILENAME, required=True,
                )
                preview_bytes = _read_regular_file_at(
                    source_fd, PREVIEW_FILENAME, required=False,
                )
            finally:
                os.close(source_fd)
        finally:
            os.close(robot_fd)
    finally:
        os.close(root_fd)
    assert provenance_bytes is not None and clip_bytes is not None
    return provenance_bytes, clip_bytes, preview_bytes


@contextmanager
def _pinned_confined_clip_dir(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path] = None,
) -> Iterator[tuple[Path, int]]:
    """Pin one existing robot/clip directory without following links.

    Returning a checked :class:`Path` is not sufficient for a mutation: a
    concurrent rename plus symlink can exchange that pathname after the check
    and redirect ``NamedTemporaryFile`` or ``os.replace`` outside the library.
    Keep the root, robot, and clip directories open by descriptor and require
    every mutating member operation to be relative to the final descriptor.

    Production reference mutation is Linux/POSIX-only.  Platforms without
    directory-fd and no-follow support fail closed instead of silently falling
    back to a check-then-open sequence.
    """
    normalized_robot = validate_robot_namespace(robot)
    normalized_clip = validate_clip_id(clip_id)
    library_root = Path(root or references_root()).expanduser().resolve(
        strict=True
    )
    candidate = require_confined_clip_dir(
        normalized_robot,
        normalized_clip,
        root=library_root,
        require_existing=True,
    )
    if os.name != "posix":  # pragma: no cover - production executes on Linux
        raise OSError(
            errno.ENOTSUP,
            "secure reference mutation requires POSIX dirfd support",
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:  # pragma: no cover - old POSIX runtime
        raise OSError(
            errno.ENOTSUP,
            "secure no-follow directory mutation is unavailable",
        )
    flags = os.O_RDONLY | directory | nofollow
    root_fd = os.open(str(library_root), flags)
    try:
        robot_fd = os.open(normalized_robot, flags, dir_fd=root_fd)
        try:
            clip_fd = os.open(normalized_clip, flags, dir_fd=robot_fd)
            try:
                yield candidate, clip_fd
            finally:
                os.close(clip_fd)
        finally:
            os.close(robot_fd)
    finally:
        os.close(root_fd)


def _confined_clip_coordinate_matches_fd(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path],
    expected_fd: int,
) -> bool:
    """Return whether the public coordinate still names a pinned directory."""
    expected = os.fstat(expected_fd)
    try:
        with _pinned_confined_clip_dir(
            robot, clip_id, root=root,
        ) as (_path, current_fd):
            current = os.fstat(current_fd)
    except (OSError, TypeError, ValueError):
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _atomic_replace_regular_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    """Fsync and atomically replace one simple filename under a pinned dirfd."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("atomic member name must be one simple filename")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if os.name != "posix" or not nofollow:  # pragma: no cover - Linux contract
        raise OSError(
            errno.ENOTSUP,
            "secure no-follow file replacement is unavailable",
        )
    temporary_name: Optional[str] = None
    descriptor = -1
    for _attempt in range(32):
        candidate = f".{name}.{os.urandom(12).hex()}.tmp"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:  # pragma: no cover - random collision
            continue
        temporary_name = candidate
        break
    if temporary_name is None or descriptor < 0:  # pragma: no cover - entropy
        raise OSError(errno.EEXIST, "cannot allocate atomic temporary member")
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry updates using the repo's POSIX convention."""
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except (OSError, NotImplementedError):  # pragma: no cover - non-POSIX
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - filesystems without directory fsync
        pass
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing name.

    Linux ``renameat2(RENAME_NOREPLACE)`` is the execution platform's kernel
    primitive for this contract.  Windows ``os.rename`` already fails when the
    destination exists.  Other POSIX platforms fail closed instead of using a
    check-then-rename sequence that could replace a racing empty directory.
    """
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - old non-glibc runtime
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory publication is unavailable",
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_parent = os.open(str(source.parent), directory_flags)
        try:
            destination_parent = os.open(
                str(destination.parent), directory_flags,
            )
        except BaseException:
            os.close(source_parent)
            raise
        try:
            result = renameat2(
                source_parent,
                os.fsencode(source.name),
                destination_parent,
                os.fsencode(destination.name),
                1,  # RENAME_NOREPLACE
            )
        finally:
            os.close(destination_parent)
            os.close(source_parent)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination
            )
        raise OSError(error_number, os.strerror(error_number), destination)
    if os.name == "nt":  # pragma: no cover - production executes on Linux
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory publication is unavailable",
    )


# ── location ─────────────────────────────────────────────────────────────
def references_root() -> Path:
    """Root directory the reference library lives under.

    `RS_REFERENCE_ROOT` overrides; default `~/.local/share/reward-sculptor/
    references/`, sibling to `saved_root()` / `trash_root()`. Mirrors
    `sculptor.archive.saved_root()` exactly.
    """
    override = os.environ.get("RS_REFERENCE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / "reward-sculptor" / "references"


def robot_dir(robot: str, *, root: Optional[Path] = None) -> Path:
    return (root or references_root()) / validate_robot_namespace(robot)


def clip_dir(robot: str, clip_id: str, *, root: Optional[Path] = None) -> Path:
    validate_clip_id(clip_id)
    return robot_dir(robot, root=root) / clip_id


def require_confined_clip_dir(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path] = None,
    require_existing: bool = True,
) -> Path:
    """Resolve one robot-scoped clip directory without following links.

    The configured library root is the caller's authority.  Robot and clip
    coordinates are data, not path fragments: both are validated centrally,
    and neither directory may redirect reads or publication through a symlink.
    """
    robot = validate_robot_namespace(robot)
    validate_clip_id(clip_id)
    library_root = Path(root or references_root()).expanduser().resolve()
    robot_path = library_root / robot
    candidate = robot_path / clip_id
    if robot_path.is_symlink():
        raise ValueError("robot reference directory must not be a symlink")
    if candidate.is_symlink():
        raise ValueError("clip reference directory must not be a symlink")
    try:
        candidate.resolve(strict=False).relative_to(library_root)
    except ValueError as exc:
        raise ValueError(
            "reference artifact path escapes the configured library root"
        ) from exc
    if require_existing:
        if not robot_path.is_dir():
            raise FileNotFoundError(f"robot reference directory not found: {robot}")
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"reference clip directory not found: {robot}/{clip_id}"
            )
    return candidate


def capture_reference_artifact_snapshot(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path] = None,
) -> tuple[bytes, bytes, Optional[bytes]]:
    """Read provenance, clip, and preview from one pinned confined directory."""
    library_root = Path(root or references_root()).expanduser().resolve()
    require_confined_clip_dir(robot, clip_id, root=library_root)
    return _capture_source_artifact_snapshot(
        library_root,
        robot=validate_robot_namespace(robot),
        source_clip_id=validate_clip_id(clip_id),
    )


# ── clip_id / hashing ───────────────────────────────────────────────────
def validate_clip_id(clip_id: str) -> str:
    if not isinstance(clip_id, str) or not CLIP_ID_RE.match(clip_id):
        raise ClipIdError(
            f"clip_id {clip_id!r} must match {CLIP_ID_RE.pattern!r}")
    return clip_id


def slugify(name: str) -> str:
    """Slugify a source filename/stem into a valid clip_id: lowercase,
    non-alnum runs -> single `_`, strip leading/trailing separators,
    clamp to the 96-char budget. Never returns an empty string (falls
    back to a short hash) so a pathological input still produces a
    usable id."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_-")
    if not s:
        s = "clip_" + hashlib.sha256(str(name).encode()).hexdigest()[:8]
    if not s[0].isalnum():
        s = "c" + s
    return s[:96]


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_root_frame_declaration_evidence(
    evidence: Any,
    *,
    expected_root_frame: Optional[str] = None,
    expected_source_artifact_sha256: Optional[str] = None,
) -> list[str]:
    """Validate the small, exact receipt that can support a Tier-D claim.

    A rationale remains useful human context, but it is deliberately not
    certification authority.  Authority requires an allow-listed inspection
    method, the exact inspected parent bytes, and a versioned reviewer
    assertion bound to the declared frame convention.
    """
    expected_keys = {
        "schema",
        "method",
        "inspected_source_artifact_sha256",
        "reviewer",
        "assertion_version",
        "asserted_root_frame",
    }
    if not isinstance(evidence, dict):
        return ["root-frame declaration evidence is missing"]
    issues: list[str] = []
    if set(evidence) != expected_keys:
        issues.append("root-frame declaration evidence is non-canonical")
    if evidence.get("schema") != ROOT_FRAME_DECLARATION_EVIDENCE_SCHEMA:
        issues.append("root-frame declaration evidence schema is unsupported")
    method = evidence.get("method")
    if method not in ROOT_FRAME_DECLARATION_EVIDENCE_METHODS:
        issues.append("root-frame declaration evidence method is unsupported")
    inspected_sha = evidence.get("inspected_source_artifact_sha256")
    if (
        not isinstance(inspected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", inspected_sha) is None
    ):
        issues.append("root-frame inspected source artifact sha256 is invalid")
    reviewer = evidence.get("reviewer")
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or reviewer != reviewer.strip()
        or len(reviewer) > 200
    ):
        issues.append("root-frame declaration reviewer is invalid")
    if (
        evidence.get("assertion_version")
        != ROOT_FRAME_DECLARATION_ASSERTION_VERSION
    ):
        issues.append("root-frame declaration assertion version is unsupported")
    asserted_root_frame = evidence.get("asserted_root_frame")
    if asserted_root_frame not in {"absolute", "origin_relative"}:
        issues.append("root-frame declaration asserted value is unsupported")
    if (
        expected_root_frame is not None
        and asserted_root_frame != expected_root_frame
    ):
        issues.append("root-frame declaration evidence value is stale")
    if (
        expected_source_artifact_sha256 is not None
        and inspected_sha != expected_source_artifact_sha256
    ):
        issues.append("root-frame declaration inspected artifact is stale")
    return issues


def root_frame_declaration_evidence_from_provenance(
    provenance: Any,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Extract and bind declaration evidence to its exact parent artifact.

    Native artifacts that already carry a root-frame field are unaffected.
    Only a metadata declaration needs this receipt before it can authorize a
    Tier-D certification run.
    """
    if not isinstance(provenance, dict):
        return None, ["reference provenance must be an object"]
    source = provenance.get("source")
    if not isinstance(source, dict) or not (
        source.get("kind") == "metadata_declaration"
        and source.get("field") == "root_frame"
    ):
        return None, []
    root_frame = source.get("value")
    parent_artifact = source.get("parent_artifact")
    parent_sha = (
        parent_artifact.get("content_sha256")
        if isinstance(parent_artifact, dict)
        else None
    )
    issues: list[str] = []
    if not isinstance(parent_artifact, dict):
        issues.append("root-frame declaration parent artifact is missing")
    if (
        not isinstance(parent_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", parent_sha) is None
    ):
        issues.append("root-frame declaration parent artifact sha256 is invalid")
    top_level_parent = provenance.get("parent_artifact")
    if top_level_parent != parent_artifact:
        issues.append("root-frame declaration parent artifact receipt is stale")
    evidence = source.get("evidence")
    issues.extend(
        validate_root_frame_declaration_evidence(
            evidence,
            expected_root_frame=(
                root_frame if isinstance(root_frame, str) else None
            ),
            expected_source_artifact_sha256=(
                parent_sha if isinstance(parent_sha, str) else None
            ),
        )
    )
    if issues or not isinstance(evidence, dict):
        return None, issues
    return json.loads(json.dumps(evidence, allow_nan=False)), []


def _canonical_evidence_sha256(value: Any) -> str:
    """Hash one evidence object under the library's strict JSON contract."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_root_frame_inheritance_receipt(
    receipt: Any,
    *,
    expected_root_frame: Optional[str] = None,
    expected_parent_artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[str]:
    """Validate the canonical, ordered authority retained by a composite.

    The receipt is intentionally small. Every immediate parent is named by
    robot-scoped artifact identity, its explicit frame convention, and a
    digest of the evidence that made that convention authoritative. Nested
    composites use the digest of their own fully validated inheritance
    receipt; metadata declarations use the digest of their reviewer evidence.
    """
    expected_keys = {"schema", "method", "asserted_root_frame", "parents"}
    parent_keys = {
        "robot",
        "clip_id",
        "content_sha256",
        "root_frame",
        "root_frame_evidence_sha256",
    }
    if not isinstance(receipt, dict):
        return ["root-frame inheritance receipt is missing"]
    issues: list[str] = []
    if set(receipt) != expected_keys:
        issues.append("root-frame inheritance receipt is non-canonical")
    if receipt.get("schema") != ROOT_FRAME_INHERITANCE_SCHEMA:
        issues.append("root-frame inheritance schema is unsupported")
    if receipt.get("method") != ROOT_FRAME_INHERITANCE_METHOD:
        issues.append("root-frame inheritance method is unsupported")
    asserted = receipt.get("asserted_root_frame")
    if asserted not in {"absolute", "origin_relative"}:
        issues.append("root-frame inheritance asserted value is unsupported")
    if expected_root_frame is not None and asserted != expected_root_frame:
        issues.append("root-frame inheritance asserted value is stale")
    parents = receipt.get("parents")
    if not isinstance(parents, list) or len(parents) < 2:
        issues.append("root-frame inheritance requires at least two parents")
        parents = []
    for index, parent in enumerate(parents):
        prefix = f"root-frame inheritance parent {index}"
        if not isinstance(parent, dict):
            issues.append(f"{prefix} is not an object")
            continue
        if set(parent) != parent_keys:
            issues.append(f"{prefix} is non-canonical")
        try:
            validate_robot_namespace(parent.get("robot"))
        except (TypeError, ValueError):
            issues.append(f"{prefix} robot is invalid")
        try:
            validate_clip_id(parent.get("clip_id"))
        except (TypeError, ValueError):
            issues.append(f"{prefix} clip id is invalid")
        for digest_key in (
            "content_sha256",
            "root_frame_evidence_sha256",
        ):
            digest = parent.get(digest_key)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                issues.append(f"{prefix} {digest_key} is invalid")
        if parent.get("root_frame") not in {"absolute", "origin_relative"}:
            issues.append(f"{prefix} root frame is unsupported")
        elif asserted in {"absolute", "origin_relative"} and (
            parent.get("root_frame") != asserted
        ):
            issues.append(f"{prefix} root frame is mixed")
    if expected_parent_artifacts is not None:
        expected: list[dict[str, Any]] = []
        for index, value in enumerate(expected_parent_artifacts):
            if not isinstance(value, Mapping):
                issues.append(
                    f"ordered parent artifact {index} is not an object"
                )
                continue
            expected.append(dict(value))
        observed = [
            {
                "robot": parent.get("robot"),
                "clip_id": parent.get("clip_id"),
                "content_sha256": parent.get("content_sha256"),
            }
            for parent in parents
            if isinstance(parent, dict)
        ]
        if observed != expected:
            issues.append(
                "root-frame inheritance parents differ from ordered parent "
                "artifact provenance"
            )
    return issues


def _root_frame_from_clip_bytes(clip_bytes: bytes) -> Optional[str]:
    """Read only the data-only root-frame scalar from an exact NPZ snapshot."""
    try:
        with np.load(io.BytesIO(clip_bytes), allow_pickle=False) as archive:
            if "root_frame" not in archive.files:
                return None
            raw = np.asarray(archive["root_frame"])
            if raw.size != 1:
                return None
            value = raw.reshape(()).item()
    except (OSError, TypeError, ValueError):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return str(value) if value in {"absolute", "origin_relative"} else None


def _root_frame_parent_receipt_from_snapshot(
    *,
    robot: str,
    clip_id: str,
    provenance_bytes: bytes,
    clip_bytes: bytes,
    root: Path,
    ancestry: frozenset[tuple[str, str]],
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Rebuild one parent's frame authority from exact retained bytes."""
    coordinate = (robot, clip_id)
    if coordinate in ancestry:
        return None, [
            f"root-frame inheritance cycle reaches {robot}/{clip_id}"
        ]
    try:
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [
            f"cannot decode parent provenance {robot}/{clip_id}: {exc}"
        ]
    issues = validate_provenance(provenance)
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        issues.append(
            f"parent {robot}/{clip_id} requires provenance schema "
            f"{PROVENANCE_SCHEMA}"
        )
    if provenance.get("robot") != robot or provenance.get("clip_id") != clip_id:
        issues.append(f"parent {robot}/{clip_id} provenance identity is stale")
    artifact_sha = content_sha256(clip_bytes)
    if provenance.get("content_sha256") != artifact_sha:
        issues.append(
            f"parent {robot}/{clip_id} content hash differs from retained bytes"
        )
    root_frame = _root_frame_from_clip_bytes(clip_bytes)
    if root_frame not in {"absolute", "origin_relative"}:
        issues.append(
            f"parent {robot}/{clip_id} has no explicit supported root_frame"
        )
    if issues:
        return None, issues

    source = provenance.get("source")
    source_kind = source.get("kind") if isinstance(source, dict) else None
    if source_kind == "metadata_declaration":
        evidence, evidence_issues = (
            root_frame_declaration_evidence_from_provenance(provenance)
        )
        if isinstance(source, dict) and source.get("value") != root_frame:
            evidence_issues.append(
                f"parent {robot}/{clip_id} declaration value differs from clip"
            )
        if evidence_issues or evidence is None:
            return None, [
                f"parent {robot}/{clip_id} has weak root-frame evidence: {issue}"
                for issue in evidence_issues
            ]
        evidence_sha = _canonical_evidence_sha256(evidence)
    elif source_kind == "compose":
        inheritance, inheritance_issues = root_frame_inheritance_from_provenance(
            provenance,
            root=root,
            expected_root_frame=root_frame,
            _ancestry=ancestry,
        )
        if inheritance_issues or inheritance is None:
            return None, [
                f"parent {robot}/{clip_id} has invalid inherited root-frame "
                f"evidence: {issue}"
                for issue in inheritance_issues
            ]
        evidence_sha = _canonical_evidence_sha256(inheritance)
    else:
        # Native/server-produced artifacts bind the declaration directly to
        # their immutable bytes. No human assertion is being inferred.
        evidence_sha = _canonical_evidence_sha256({
            "schema": ROOT_FRAME_EMBEDDED_EVIDENCE_SCHEMA,
            "content_sha256": artifact_sha,
            "root_frame": root_frame,
        })
    return {
        "robot": robot,
        "clip_id": clip_id,
        "content_sha256": artifact_sha,
        "root_frame": root_frame,
        "root_frame_evidence_sha256": evidence_sha,
    }, []


def root_frame_parent_receipt(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path] = None,
    _ancestry: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Capture and rebuild one robot-scoped parent's frame authority."""
    try:
        normalized_robot = validate_robot_namespace(robot)
        normalized_clip = validate_clip_id(clip_id)
        effective_root = Path(root or references_root()).expanduser().resolve()
        provenance_bytes, clip_bytes, _ = capture_reference_artifact_snapshot(
            normalized_robot, normalized_clip, root=effective_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        return None, [
            f"cannot capture root-frame parent {robot}/{clip_id}: "
            f"{type(exc).__name__}: {exc}"
        ]
    return _root_frame_parent_receipt_from_snapshot(
        robot=normalized_robot,
        clip_id=normalized_clip,
        provenance_bytes=provenance_bytes,
        clip_bytes=clip_bytes,
        root=effective_root,
        ancestry=_ancestry,
    )


def root_frame_inheritance_from_provenance(
    provenance: Any,
    *,
    root: Optional[Path] = None,
    expected_root_frame: Optional[str] = None,
    _ancestry: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Re-derive a composite's ordered inheritance from current parents.

    A non-composite has no inheritance receipt. A composite must have one, and
    every call reopens the exact retained parents under the configured library
    root. This is the Tier-D anti-laundering boundary: editing a receipt without
    the parent bytes/evidence to support it cannot authorize certification.
    """
    if not isinstance(provenance, dict):
        return None, ["reference provenance must be an object"]
    source = provenance.get("source")
    if not isinstance(source, dict) or source.get("kind") != "compose":
        return None, []
    receipt = source.get("root_frame_inheritance")
    parent_artifacts = source.get("parent_artifacts")
    issues = validate_root_frame_inheritance_receipt(
        receipt,
        expected_root_frame=expected_root_frame,
        expected_parent_artifacts=(
            parent_artifacts if isinstance(parent_artifacts, list) else None
        ),
    )
    if not isinstance(parent_artifacts, list):
        issues.append("compose provenance parent_artifacts is missing")
    parent_ids = source.get("parent_clip_ids")
    if not isinstance(parent_ids, list):
        issues.append("compose provenance parent_clip_ids is missing")
    elif isinstance(receipt, dict) and isinstance(receipt.get("parents"), list):
        if parent_ids != [
            parent.get("clip_id")
            for parent in receipt["parents"]
            if isinstance(parent, dict)
        ]:
            issues.append("compose parent clip ids differ from inheritance order")
    current_robot = provenance.get("robot")
    current_clip = provenance.get("clip_id")
    coordinate = (current_robot, current_clip)
    if not all(isinstance(value, str) for value in coordinate):
        issues.append("compose provenance identity is missing")
    elif coordinate in _ancestry:
        issues.append(
            f"root-frame inheritance cycle reaches {current_robot}/{current_clip}"
        )
    if issues or not isinstance(receipt, dict):
        return None, issues

    next_ancestry = frozenset((*_ancestry, coordinate))
    expected_parents = receipt["parents"]
    for index, expected_parent in enumerate(expected_parents):
        actual_parent, parent_issues = root_frame_parent_receipt(
            expected_parent["robot"],
            expected_parent["clip_id"],
            root=root,
            _ancestry=next_ancestry,
        )
        if parent_issues:
            issues.extend(
                f"root-frame inheritance parent {index}: {issue}"
                for issue in parent_issues
            )
        elif actual_parent != expected_parent:
            issues.append(
                f"root-frame inheritance parent {index} receipt is stale"
            )
    if issues:
        return None, issues
    return json.loads(json.dumps(receipt, allow_nan=False)), []


# ── provenance ───────────────────────────────────────────────────────────
def make_provenance(
    *,
    clip_id: str,
    robot: str,
    source: dict[str, Any],
    license: str,
    attribution: str,
    content_sha256_: str,
    source_content_sha256_: Optional[str] = None,
    retarget: Optional[dict[str, Any]] = None,
    tier: str = "K",
    fps_source: Optional[float] = None,
    parent_clip_id: Optional[str] = None,
    frame_range: Optional[list[int]] = None,
    joint_mapping: Optional[dict[str, Any]] = None,
    labels: Optional[list[str]] = None,
    text: str = "",
    qc: Optional[dict[str, Any]] = None,
    ingested_at: Optional[str] = None,
) -> dict[str, Any]:
    """Build a provenance record matching §decision-3's schema exactly."""
    validate_clip_id(clip_id)
    robot = validate_robot_namespace(robot)
    if not license:
        raise LicenseGuardError(
            f"clip {clip_id!r}: license is required and must be non-empty")
    return {
        "schema": PROVENANCE_SCHEMA,
        "clip_id": clip_id,
        "robot": robot,
        "source": source,
        "license": license,
        "attribution": attribution,
        "retarget": retarget or {"tool": "dataset-provided", "notes": ""},
        "tier": tier,
        "fps_source": fps_source,
        "parent_clip_id": parent_clip_id,
        "frame_range": frame_range,
        "joint_mapping": joint_mapping or {"identity": True},
        "content_sha256": content_sha256_,
        "source_content_sha256": source_content_sha256_,
        "labels": list(labels or []),
        "text": text,
        "qc": qc or {},
        "ingested_at": ingested_at or _utc_now_iso(),
    }


def validate_provenance(prov: dict[str, Any]) -> list[str]:
    """All violations at once (mirrors `validate_clip` / `validate_env_spec`
    style). Does not check disk state — pure schema validation."""
    errors: list[str] = []
    required = (
        "schema", "clip_id", "robot", "source", "license", "attribution",
        "content_sha256",
    )
    for key in required:
        if key not in prov or prov[key] in (None, ""):
            errors.append(f"provenance missing required field: {key}")
    schema = prov.get("schema")
    if schema not in SUPPORTED_PROVENANCE_SCHEMAS:
        errors.append(
            "provenance.schema must be one of "
            f"{sorted(SUPPORTED_PROVENANCE_SCHEMAS)}, got {schema!r}")
    clip_id = prov.get("clip_id")
    if isinstance(clip_id, str):
        try:
            validate_clip_id(clip_id)
        except ClipIdError as e:
            errors.append(str(e))
    robot = prov.get("robot")
    if isinstance(robot, str):
        try:
            validate_robot_namespace(robot)
        except ValueError as exc:
            errors.append(str(exc))
    src = prov.get("source")
    if src is not None and not isinstance(src, dict):
        errors.append("provenance.source must be an object")
    artifact_sha = prov.get("content_sha256")
    if artifact_sha not in (None, "") and (
            not isinstance(artifact_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha) is None):
        errors.append(
            "provenance.content_sha256 must be a lowercase SHA-256 digest")
    source_sha = prov.get("source_content_sha256")
    if source_sha is not None and (
            not isinstance(source_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None):
        errors.append(
            "provenance.source_content_sha256 must be null or a lowercase "
            "SHA-256 digest")
    return errors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _schema2_artifact_identity_errors(
    robot: str, clip_id: str, prov: dict[str, Any], *, root: Optional[Path],
) -> list[str]:
    """Verify that schema-2 provenance names the retained artifact bytes.

    Schema 2 exists specifically to make ``content_sha256`` the identity of
    the server-owned ``clip.npz`` consumed by training.  Validating only the
    digest's syntax would let a caller write a perfectly shaped lie at the
    library's single persistence choke point.
    """
    if prov.get("schema") != PROVENANCE_SCHEMA:
        return []
    path = clip_dir(robot, clip_id, root=root) / CLIP_FILENAME
    if not path.is_file():
        return [
            "schema-2 provenance requires the retained clip.npz artifact "
            "to exist before provenance is written"
        ]
    try:
        actual_sha = content_sha256(path.read_bytes())
    except OSError as exc:
        return [f"cannot read retained clip.npz artifact: {exc}"]
    declared_sha = prov.get("content_sha256")
    if declared_sha != actual_sha:
        return [
            "provenance.content_sha256 does not match the exact retained "
            f"clip.npz bytes ({declared_sha!r} != {actual_sha!r})"
        ]
    return []


def write_provenance(
    robot: str, clip_id: str, prov: dict[str, Any], *, root: Optional[Path] = None,
) -> Path:
    """Persist provenance.json beside the clip (§decision-3(b)). Refuses
    to write an invalid record (license guard lives here — this is the
    single choke point every ingest/segment path must go through)."""
    errors = validate_provenance(prov)
    if prov.get("robot") != robot:
        errors.append(
            "provenance.robot does not match the robot-scoped destination "
            f"({prov.get('robot')!r} != {robot!r})")
    if prov.get("clip_id") != clip_id:
        errors.append(
            "provenance.clip_id does not match the destination "
            f"({prov.get('clip_id')!r} != {clip_id!r})")
    # Preserve the actionable missing/mismatch diagnostic before opening the
    # pinned mutation boundary.  A successful path-level check is never
    # trusted on its own: the exact bytes are read again through the dirfd
    # below, closing the check/open race.
    if not errors:
        errors.extend(
            _schema2_artifact_identity_errors(
                robot, clip_id, prov, root=root,
            )
        )
    if errors:
        raise LicenseGuardError(
            "refusing to write provenance for clip "
            f"{clip_id!r}:\n  - " + "\n  - ".join(errors))

    payload = json.dumps(prov, indent=2, sort_keys=True).encode("utf-8")
    try:
        with _pinned_confined_clip_dir(
            robot, clip_id, root=root,
        ) as (path_dir, directory_fd):
            if prov.get("schema") == PROVENANCE_SCHEMA:
                clip_bytes = _read_regular_file_at(
                    directory_fd, CLIP_FILENAME, required=True,
                )
                assert clip_bytes is not None
                actual_sha = content_sha256(clip_bytes)
                declared_sha = prov.get("content_sha256")
                if declared_sha != actual_sha:
                    raise LicenseGuardError(
                        "refusing to write provenance for clip "
                        f"{clip_id!r}:\n  - provenance.content_sha256 does not "
                        "match the exact retained clip.npz bytes "
                        f"({declared_sha!r} != {actual_sha!r})"
                    )
            _atomic_replace_regular_file_at(
                directory_fd,
                PROVENANCE_FILENAME,
                payload,
            )
            if not _confined_clip_coordinate_matches_fd(
                robot,
                clip_id,
                root=root,
                expected_fd=directory_fd,
            ):
                raise LicenseGuardError(
                    "refusing stale provenance publication: the robot-scoped "
                    "clip directory changed during atomic replacement"
                )
            path = path_dir / PROVENANCE_FILENAME
    except LicenseGuardError:
        raise
    except OSError:
        # Preserve the historical filesystem-error contract while keeping the
        # operation confined through the pinned descriptor.
        raise
    except (TypeError, ValueError) as exc:
        raise LicenseGuardError(
            f"refusing unsafe provenance destination: {exc}"
        ) from exc
    return path


def read_provenance(robot: str, clip_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    path = require_confined_clip_dir(
        robot, clip_id, root=root,
    ) / PROVENANCE_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


# ── index ────────────────────────────────────────────────────────────────
def _index_row_from_provenance(prov: dict[str, Any], *, has_preview: bool) -> dict[str, Any]:
    qc = prov.get("qc") or {}
    return {
        "clip_id": prov["clip_id"],
        "robot": prov["robot"],
        "text": prov.get("text", ""),
        "labels": prov.get("labels") or [],
        "tier": prov.get("tier", "K"),
        "license": prov.get("license", ""),
        "n_frames": qc.get("n_frames"),
        "fps": prov.get("fps_source"),
        "duration_s": qc.get("duration_s"),
        "root_z_range": qc.get("root_z_range"),
        "has_preview": has_preview,
    }


def index_path(*, root: Optional[Path] = None) -> Path:
    return (root or references_root()) / INDEX_FILENAME


def rejects_path(*, root: Optional[Path] = None) -> Path:
    return (root or references_root()) / REJECTS_FILENAME


def read_index(*, root: Optional[Path] = None) -> list[dict[str, Any]]:
    path = index_path(root=root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_index_unlocked(
    rows: list[dict[str, Any]], *, root: Optional[Path] = None,
) -> Path:
    """Atomically replace the index while the caller holds the library lock."""
    path = index_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{INDEX_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def write_index(rows: list[dict[str, Any]], *, root: Optional[Path] = None) -> Path:
    """Atomically publish explicit index rows in the global mutation domain."""
    with reference_library_mutation_lock(root=root):
        return _write_index_unlocked(rows, root=root)


def append_reject(reason: str, detail: dict[str, Any], *, root: Optional[Path] = None) -> Path:
    path = rejects_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"reason": reason, "ingested_at": _utc_now_iso(), **detail}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _rebuild_index_unlocked(*, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Rebuild the index while the caller holds the library mutation lock."""
    r = Path(root or references_root())
    rows: list[dict[str, Any]] = []
    if r.is_dir():
        for robot_d in sorted(p for p in r.iterdir() if p.is_dir()):
            for clip_d in sorted(p for p in robot_d.iterdir() if p.is_dir()):
                prov_path = clip_d / PROVENANCE_FILENAME
                if not prov_path.is_file():
                    continue
                try:
                    prov = json.loads(prov_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                errors = validate_provenance(prov)
                if prov.get("robot") != robot_d.name:
                    errors.append(
                        "provenance.robot does not match its robot directory")
                if prov.get("clip_id") != clip_d.name:
                    errors.append(
                        "provenance.clip_id does not match its clip directory")
                errors.extend(
                    _schema2_artifact_identity_errors(
                        robot_d.name, clip_d.name, prov, root=r,
                    )
                )
                if errors:
                    continue
                has_preview = (clip_d / PREVIEW_FILENAME).is_file()
                rows.append(_index_row_from_provenance(prov, has_preview=has_preview))
    rows.sort(key=lambda row: (row["robot"], row["clip_id"]))
    _write_index_unlocked(rows, root=r)
    return rows


def rebuild_index(*, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Rescan every `<robot>/<clip_id>/provenance.json` on disk and
    rewrite `index.jsonl` from scratch. The index is a cache; provenance
    is truth (§decision 4) — this is the only correct way to recover
    from a stale/corrupt/missing index."""
    with reference_library_mutation_lock(root=root):
        return _rebuild_index_unlocked(root=root)


def indexed_content_hashes(*, root: Optional[Path] = None) -> set[str]:
    """Exact ``clip.npz`` artifact identities present in provenance.

    This is deliberately *not* the raw-source deduplication index.  Call
    :func:`indexed_source_hashes` when deciding whether downloaded source
    bytes have already been admitted.
    """
    r = root or references_root()
    hashes: set[str] = set()
    if not r.is_dir():
        return hashes
    for robot_d in r.iterdir():
        if not robot_d.is_dir():
            continue
        for clip_d in robot_d.iterdir():
            prov_path = clip_d / PROVENANCE_FILENAME
            if prov_path.is_file():
                try:
                    prov = json.loads(prov_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                sha = prov.get("content_sha256")
                if (
                    prov.get("schema") == PROVENANCE_SCHEMA
                    and not _schema2_artifact_identity_errors(
                        robot_d.name, clip_d.name, prov, root=r,
                    )
                    and sha
                ):
                    hashes.add(sha)
    return hashes


def indexed_source_hashes(*, root: Optional[Path] = None) -> set[str]:
    """Raw-source SHA-256 values already admitted to the library.

    Schema-2 records expose this as ``source_content_sha256``.  For an
    unmigrated schema-1 *root* dataset/retarget record only, the legacy
    ``content_sha256`` field is the raw source digest and is accepted as a
    compatibility fallback.  Derived segments and composites are never
    treated as independent source admissions.
    """
    r = root or references_root()
    hashes: set[str] = set()
    if not r.is_dir():
        return hashes
    for robot_d in r.iterdir():
        if not robot_d.is_dir():
            continue
        for clip_d in robot_d.iterdir():
            prov_path = clip_d / PROVENANCE_FILENAME
            if not prov_path.is_file():
                continue
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if (
                prov.get("schema") == PROVENANCE_SCHEMA
                and _schema2_artifact_identity_errors(
                    robot_d.name, clip_d.name, prov, root=r,
                )
            ):
                continue
            sha = prov.get("source_content_sha256")
            if not sha and prov.get("schema") == 1:
                source = prov.get("source") or {}
                if (
                    not prov.get("parent_clip_id")
                    and isinstance(source, dict)
                    and source.get("kind") in {"hf_dataset", "retarget"}
                ):
                    sha = prov.get("content_sha256")
            if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
                hashes.add(sha)
    return hashes


@dataclass(frozen=True)
class ArtifactIdentityMigration:
    """One schema-1 -> schema-2 reference identity migration receipt."""

    robot: str
    clip_id: str
    previous_content_sha256: str
    artifact_content_sha256: str
    source_content_sha256: Optional[str]
    previous_tier: str
    resulting_tier: str


def migrate_artifact_identities(
    *, root: Optional[Path] = None, robot: Optional[str] = None,
    clip_ids: Optional[set[str]] = None, dry_run: bool = False,
) -> list[ArtifactIdentityMigration]:
    """Migrate artifact identities as one library-wide transaction.

    A migration can rewrite several provenance records before publishing one
    shared index.  Keep that entire write set under the same lock as imports,
    Tier-D certification, materialization, and manual index rebuilds so no
    concurrent writer can be lost or downgraded by a stale migration snapshot.
    """
    if dry_run:
        return _migrate_artifact_identities_unlocked(
            root=root,
            robot=robot,
            clip_ids=clip_ids,
            dry_run=True,
        )
    with reference_library_mutation_lock(root=root):
        return _migrate_artifact_identities_unlocked(
            root=root,
            robot=robot,
            clip_ids=clip_ids,
            dry_run=False,
        )


def _migrate_artifact_identities_unlocked(
    *, root: Optional[Path] = None, robot: Optional[str] = None,
    clip_ids: Optional[set[str]] = None, dry_run: bool = False,
) -> list[ArtifactIdentityMigration]:
    """Make every retained reference identity name its exact clip bytes.

    The migration is explicit and idempotent.  It preserves the overloaded
    schema-1 value under ``legacy_content_sha256``, carries true raw-source
    identity in ``source_content_sha256`` where it can be established, and
    recalculates sampled duration from the shared ``(N - 1) / fps`` clock.
    Any Tier-D claim touched by the migration is downgraded to Tier K because
    its old certificate did not bind the new identity contract; its evidence
    remains on disk and is marked invalidated for audit rather than deleted.
    """
    from sculptor.reference import load_clip
    from sculptor.reference_clock import reference_playback_duration_s

    r = root or references_root()
    if not r.is_dir():
        return []

    records: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    target_keys: set[tuple[str, str]] = set()
    for robot_d in sorted(p for p in r.iterdir() if p.is_dir()):
        for clip_d in sorted(p for p in robot_d.iterdir() if p.is_dir()):
            prov_path = clip_d / PROVENANCE_FILENAME
            clip_path = clip_d / CLIP_FILENAME
            if not prov_path.is_file() or not clip_path.is_file():
                continue
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            key = (robot_d.name, clip_d.name)
            records[key] = (clip_path, prov)
            if (
                (robot is None or robot_d.name == robot)
                and (clip_ids is None or clip_d.name in clip_ids)
            ):
                target_keys.add(key)

    def resolve_source_sha(
        key: tuple[str, str], seen: frozenset[tuple[str, str]] = frozenset(),
    ) -> Optional[str]:
        if key in seen or key not in records:
            return None
        _clip_path, prov = records[key]
        direct = prov.get("source_content_sha256")
        if isinstance(direct, str) and re.fullmatch(r"[0-9a-f]{64}", direct):
            return direct
        parent = prov.get("parent_clip_id")
        source = prov.get("source") or {}
        if parent and isinstance(source, dict) and source.get("kind") != "compose":
            inherited = resolve_source_sha(
                (key[0], str(parent)), seen | {key})
            if inherited:
                return inherited
        if (
            prov.get("schema") == 1
            and not parent
            and isinstance(source, dict)
            and source.get("kind") in {"hf_dataset", "retarget"}
        ):
            legacy = prov.get("content_sha256")
            if isinstance(legacy, str) and re.fullmatch(r"[0-9a-f]{64}", legacy):
                return legacy
        return None

    migrated: list[ArtifactIdentityMigration] = []
    for robot_name, clip_id in sorted(target_keys):
        clip_path, original = records[(robot_name, clip_id)]
        artifact_sha = content_sha256(clip_path.read_bytes())
        previous_sha = str(original.get("content_sha256") or "")
        source_sha = resolve_source_sha((robot_name, clip_id))
        needs_migration = (
            original.get("schema") != PROVENANCE_SCHEMA
            or previous_sha != artifact_sha
            or original.get("source_content_sha256") != source_sha
        )
        if not needs_migration:
            continue

        prov = dict(original)
        prov["schema"] = PROVENANCE_SCHEMA
        prov["content_sha256"] = artifact_sha
        prov["source_content_sha256"] = source_sha
        if previous_sha and previous_sha != artifact_sha:
            prov.setdefault("legacy_content_sha256", previous_sha)

        clip = load_clip(clip_path)
        qc = dict(prov.get("qc") or {})
        n_frames = int(clip["root_pos_z"].shape[0])
        fps = float(clip["fps"])
        qc["n_frames"] = n_frames
        qc["duration_s"] = round(
            reference_playback_duration_s(frame_count=n_frames, fps=fps), 6)
        prov["qc"] = qc

        previous_tier = str(original.get("tier") or "K")
        resulting_tier = previous_tier
        if previous_tier == "D" or isinstance(original.get("tierD"), dict):
            resulting_tier = "K"
            prov["tier"] = "K"
            tier_d = dict(original.get("tierD") or {})
            tier_d["feasible"] = False
            tier_d["identity_migration_invalidated"] = {
                "reason": (
                    "schema-1 certificate did not bind the exact clip.npz "
                    "artifact identity; fresh Tier-D certification required"),
                "previous_content_sha256": previous_sha,
                "artifact_content_sha256": artifact_sha,
            }
            prov["tierD"] = tier_d

        receipt = ArtifactIdentityMigration(
            robot=robot_name,
            clip_id=clip_id,
            previous_content_sha256=previous_sha,
            artifact_content_sha256=artifact_sha,
            source_content_sha256=source_sha,
            previous_tier=previous_tier,
            resulting_tier=resulting_tier,
        )
        migrated.append(receipt)
        if not dry_run:
            write_provenance(robot_name, clip_id, prov, root=r)

    if migrated and not dry_run:
        _rebuild_index_unlocked(root=r)
    return migrated


@dataclass
class LibraryClip:
    """A single ingested/segmented clip's on-disk identity, returned by
    ingest/segment so callers don't have to re-derive paths."""

    robot: str
    clip_id: str
    clip_path: Path
    provenance_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)
    index_refresh_error: Optional[str] = None


def materialize_root_frame_declaration(
    *,
    robot: str,
    source_clip_id: str,
    output_clip_id: str,
    root_frame: str,
    rationale: str,
    evidence_method: Optional[str] = None,
    reviewer: Optional[str] = None,
    root: Optional[Path] = None,
) -> LibraryClip:
    """Materialize an explicit root-frame declaration under a new identity.

    A missing root-frame declaration is not safe to infer at execution time:
    identical numeric root trajectories can mean either world-space positions
    or offsets from the episode origin.  This operation preserves every NPZ
    member from a validated server-owned parent, adds only ``root_frame``, and
    emits schema-2 provenance bound to both the new bytes and the exact parent
    artifact bytes.  The result is always Tier K; any Tier-D claim must be
    earned again for the newly materialized identity.

    The source is never edited and an existing destination is never replaced.
    Admission happens by a single same-filesystem directory rename after the
    staged clip and provenance have both passed their normal validators.
    """
    from sculptor.reference import load_clip
    from sculptor.reference_clock import reference_playback_duration_s

    validate_clip_id(source_clip_id)
    validate_clip_id(output_clip_id)
    if source_clip_id == output_clip_id:
        raise ArtifactMaterializationError(
            "output_clip_id must differ from source_clip_id; declarations "
            "are immutable derived artifacts"
        )
    if (
        not isinstance(root_frame, str)
        or root_frame not in {"absolute", "origin_relative"}
    ):
        raise ArtifactMaterializationError(
            "root_frame must be 'absolute' or 'origin_relative'"
        )
    declaration_rationale = (
        rationale.strip() if isinstance(rationale, str) else ""
    )
    if not declaration_rationale:
        raise ArtifactMaterializationError(
            "rationale must explain the evidence for the root_frame "
            "declaration"
        )
    if (evidence_method is None) != (reviewer is None):
        raise ArtifactMaterializationError(
            "evidence_method and reviewer must be supplied together"
        )
    if evidence_method is not None:
        if evidence_method not in ROOT_FRAME_DECLARATION_EVIDENCE_METHODS:
            raise ArtifactMaterializationError(
                "evidence_method must be one of: "
                + ", ".join(sorted(ROOT_FRAME_DECLARATION_EVIDENCE_METHODS))
            )
        if (
            not isinstance(reviewer, str)
            or not reviewer.strip()
            or reviewer != reviewer.strip()
            or len(reviewer) > 200
        ):
            raise ArtifactMaterializationError(
                "reviewer must be non-empty, trimmed text of at most 200 "
                "characters"
            )

    try:
        robot = validate_robot_namespace(robot)
    except ValueError as exc:
        raise ArtifactMaterializationError(
            f"invalid robot namespace: {exc}"
        ) from exc

    library_root = Path(root or references_root()).expanduser()
    library_root.mkdir(parents=True, exist_ok=True)
    library_root = library_root.resolve()
    robot_path = robot_dir(robot, root=library_root)
    if robot_path.is_symlink():
        raise ArtifactMaterializationError(
            "robot reference directory must not be a symlink"
        )
    source_dir = clip_dir(robot, source_clip_id, root=library_root)
    source_clip_path = source_dir / CLIP_FILENAME
    source_prov_path = source_dir / PROVENANCE_FILENAME
    source_preview_path = source_dir / PREVIEW_FILENAME
    destination_dir = clip_dir(robot, output_clip_id, root=library_root)
    if source_dir.is_symlink():
        raise ArtifactMaterializationError(
            "source reference directory must not be a symlink"
        )
    for candidate in (source_dir, destination_dir):
        try:
            candidate.resolve(strict=False).relative_to(library_root)
        except ValueError as exc:
            raise ArtifactMaterializationError(
                "reference artifact path escapes the configured library root"
            ) from exc
    if destination_dir.exists() or destination_dir.is_symlink():
        raise ArtifactMaterializationError(
            f"destination already exists: {destination_dir}"
        )
    if not source_clip_path.is_file() or not source_prov_path.is_file():
        raise ArtifactMaterializationError(
            "source requires both retained clip.npz and provenance.json"
        )
    if source_clip_path.is_symlink() or source_prov_path.is_symlink():
        raise ArtifactMaterializationError(
            "source clip and provenance must be retained regular files, not "
            "symlinks"
        )
    if source_preview_path.is_symlink():
        raise ArtifactMaterializationError(
            "source preview must not be a symlink"
        )

    try:
        (
            source_provenance_bytes,
            source_bytes,
            source_preview_bytes,
        ) = _capture_source_artifact_snapshot(
            library_root,
            robot=robot,
            source_clip_id=source_clip_id,
        )
        source_prov = json.loads(source_provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise ArtifactMaterializationError(
            f"cannot capture source artifact snapshot: {exc}"
        ) from exc
    source_errors = validate_provenance(source_prov)
    source_errors.extend(_materialization_parent_metadata_issues(source_prov))
    if source_prov.get("robot") != robot:
        source_errors.append("source provenance.robot does not match robot")
    if source_prov.get("clip_id") != source_clip_id:
        source_errors.append(
            "source provenance.clip_id does not match source_clip_id"
        )
    parent_artifact_sha = content_sha256(source_bytes)
    if (
        source_prov.get("schema") == PROVENANCE_SCHEMA
        and source_prov.get("content_sha256") != parent_artifact_sha
    ):
        source_errors.append(
            "source schema-2 content_sha256 does not match retained clip.npz"
        )
    if source_errors:
        raise ArtifactMaterializationError(
            "invalid source artifact:\n  - " + "\n  - ".join(source_errors)
        )

    source_sha = source_prov.get("source_content_sha256")
    source_kind = (source_prov.get("source") or {}).get("kind")
    if (
        source_sha is None
        and source_prov.get("schema") == 1
        and not source_prov.get("parent_clip_id")
        and source_kind in {"hf_dataset", "retarget"}
    ):
        legacy_sha = source_prov.get("content_sha256")
        if isinstance(legacy_sha, str) and re.fullmatch(
            r"[0-9a-f]{64}", legacy_sha
        ):
            source_sha = legacy_sha

    parent_receipt = {
        "robot": robot,
        "clip_id": source_clip_id,
        "content_sha256": parent_artifact_sha,
    }
    declaration_evidence = None
    if evidence_method is not None:
        declaration_evidence = {
            "schema": ROOT_FRAME_DECLARATION_EVIDENCE_SCHEMA,
            "method": evidence_method,
            "inspected_source_artifact_sha256": parent_artifact_sha,
            "reviewer": reviewer,
            "assertion_version": ROOT_FRAME_DECLARATION_ASSERTION_VERSION,
            "asserted_root_frame": root_frame,
        }
        evidence_issues = validate_root_frame_declaration_evidence(
            declaration_evidence,
            expected_root_frame=root_frame,
            expected_source_artifact_sha256=parent_artifact_sha,
        )
        if evidence_issues:  # pragma: no cover - guarded input construction
            raise ArtifactMaterializationError(
                "invalid root-frame declaration evidence: "
                + "; ".join(evidence_issues)
            )
    parent_preview_sha = (
        content_sha256(source_preview_bytes)
        if source_preview_bytes is not None
        else None
    )

    provenance: dict[str, Any]
    with tempfile.TemporaryDirectory(
        prefix=".reference-materialization-", dir=library_root
    ) as staging_name:
        staging_root = Path(staging_name)
        snapshot_path = staging_root / ".parent.clip.npz"
        with snapshot_path.open("wb") as stream:
            stream.write(source_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        # load_clip is the canonical semantic validator.  Both it and the raw
        # member extraction consume the same captured byte snapshot, never a
        # second read of the mutable parent path.
        try:
            source_clip = load_clip(snapshot_path)
            with np.load(snapshot_path, allow_pickle=False) as archive:
                source_member_names = list(archive.files)
                if len(source_member_names) != len(set(source_member_names)):
                    raise ArtifactMaterializationError(
                        "source clip contains duplicate NPZ member names"
                    )
                raw_payload = {
                    name: archive[name].copy()
                    for name in source_member_names
                }
        except ArtifactMaterializationError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactMaterializationError(
                f"invalid source clip snapshot: {exc}"
            ) from exc
        if "root_frame" in source_clip:
            raise ArtifactMaterializationError(
                "source already declares root_frame; no metadata declaration "
                "is needed"
            )
        raw_payload["root_frame"] = np.asarray(root_frame)

        fps = float(source_clip["fps"])
        n_frames = int(source_clip["root_pos_z"].shape[0])
        qc = dict(source_prov.get("qc") or {})
        qc.update(
            {
                "n_frames": n_frames,
                "duration_s": round(
                    reference_playback_duration_s(
                        frame_count=n_frames, fps=fps
                    ),
                    6,
                ),
                "root_frame": root_frame,
                "metadata_declaration": {
                    "field": "root_frame",
                    "value": root_frame,
                    "rationale": declaration_rationale,
                    "arrays_preserved": True,
                    "preview_preserved": source_preview_bytes is not None,
                    "parent_preview_sha256": parent_preview_sha,
                    "evidence": declaration_evidence,
                },
            }
        )

        staged_dir = clip_dir(robot, output_clip_id, root=staging_root)
        staged_dir.mkdir(parents=True, exist_ok=False)
        staged_clip_path = staged_dir / CLIP_FILENAME
        with staged_clip_path.open("wb") as stream:
            np.savez_compressed(stream, **raw_payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            materialized = load_clip(staged_clip_path)
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactMaterializationError(
                f"invalid materialized reference: {exc}"
            ) from exc
        if materialized.get("root_frame") != root_frame:
            raise ArtifactMaterializationError(
                "materialized artifact did not retain the declared root_frame"
            )
        with np.load(staged_clip_path, allow_pickle=False) as derived_archive:
            derived_member_names = list(derived_archive.files)
            if len(derived_member_names) != len(set(derived_member_names)):
                raise ArtifactMaterializationError(
                    "materialized artifact has duplicate NPZ member names"
                )
            if set(derived_member_names) != set(raw_payload):
                raise ArtifactMaterializationError(
                    "materialized artifact member set changed unexpectedly"
                )
            for name, expected in raw_payload.items():
                observed = derived_archive[name]
                if (
                    observed.dtype != expected.dtype
                    or observed.shape != expected.shape
                    or not np.array_equal(observed, expected)
                ):
                    raise ArtifactMaterializationError(
                        f"materialized artifact member {name!r} changed"
                    )

        if source_preview_bytes is not None:
            staged_preview_path = staged_dir / PREVIEW_FILENAME
            with staged_preview_path.open("wb") as stream:
                stream.write(source_preview_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            if content_sha256(staged_preview_path.read_bytes()) != parent_preview_sha:
                raise ArtifactMaterializationError(
                    "materialized preview differs from the parent preview"
                )

        artifact_sha = content_sha256(staged_clip_path.read_bytes())
        source_receipt = {
            "kind": "metadata_declaration",
            "field": "root_frame",
            "value": root_frame,
            "rationale": declaration_rationale,
            "parent_artifact": parent_receipt,
        }
        if declaration_evidence is not None:
            source_receipt["evidence"] = declaration_evidence
        provenance = make_provenance(
            clip_id=output_clip_id,
            robot=robot,
            source=source_receipt,
            license=str(source_prov["license"]),
            attribution=str(source_prov.get("attribution") or ""),
            content_sha256_=artifact_sha,
            source_content_sha256_=source_sha,
            retarget={
                "tool": "rewardsculptor:materialize-root-frame",
                "notes": (
                    "All parent NPZ members preserved exactly; only the "
                    f"root_frame={root_frame!r} declaration was added."
                ),
            },
            tier="K",
            fps_source=fps,
            parent_clip_id=source_clip_id,
            frame_range=source_prov.get("frame_range"),
            joint_mapping=dict(source_prov.get("joint_mapping") or {}),
            labels=list(source_prov.get("labels") or []),
            text=str(source_prov.get("text") or ""),
            qc=qc,
        )
        provenance["parent_artifact"] = parent_receipt
        if parent_preview_sha is not None:
            provenance["parent_preview_sha256"] = parent_preview_sha
        write_provenance(
            robot, output_clip_id, provenance, root=staging_root
        )

        index_refresh_error: Optional[str] = None
        with reference_library_mutation_lock(root=library_root):
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            if destination_dir.parent.is_symlink():
                raise ArtifactMaterializationError(
                    "destination robot directory must not be a symlink"
                )
            try:
                destination_dir.parent.resolve().relative_to(library_root)
            except ValueError as exc:
                raise ArtifactMaterializationError(
                    "destination robot directory escapes the library root"
                ) from exc
            _fsync_directory(staged_dir)
            _fsync_directory(destination_dir.parent)
            if destination_dir.exists() or destination_dir.is_symlink():
                raise ArtifactMaterializationError(
                    "destination appeared during materialization: "
                    f"{destination_dir}"
                )
            try:
                _rename_directory_noreplace(staged_dir, destination_dir)
            except FileExistsError as exc:
                raise ArtifactMaterializationError(
                    "destination appeared during materialization: "
                    f"{destination_dir}"
                ) from exc
            _fsync_directory(destination_dir.parent)
            try:
                _rebuild_index_unlocked(root=library_root)
            except Exception as exc:  # noqa: BLE001 - cache is non-authoritative
                # The provenance beside the immutable artifact is authoritative;
                # the index is explicitly a rebuildable cache.  Report a
                # recoverable cache warning without misrepresenting the committed
                # publication as a failed materialization.
                index_refresh_error = f"{type(exc).__name__}: {exc}"
    return LibraryClip(
        robot=robot,
        clip_id=output_clip_id,
        clip_path=destination_dir / CLIP_FILENAME,
        provenance_path=destination_dir / PROVENANCE_FILENAME,
        provenance=provenance,
        index_refresh_error=index_refresh_error,
    )
