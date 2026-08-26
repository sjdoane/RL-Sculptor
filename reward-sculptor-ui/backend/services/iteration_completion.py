"""Fail-closed completion authority for trained policy iterations.

``checkpoint.pt`` proves that policy bytes were preserved; it does not prove
that rollout, objective evaluation, or the outer sculpt iteration completed.
Modern runs write an atomic completion marker after those stages.  A narrow
legacy fallback admits older iterations only when both rollout and objective
artifacts are present and readable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Literal


_ITER_RE = re.compile(r"^iter_(?P<iteration>[0-9]+)$")
_CHECKPOINT_NAMES = ("checkpoint.pt", "checkpoint.zip")


def _plain_nonempty_file(path: Path) -> bool:
    try:
        return (
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size > 0
        )
    except OSError:
        return False


def _json_object(path: Path) -> dict[str, Any] | None:
    if not _plain_nonempty_file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iteration_index(iter_dir: Path) -> int | None:
    match = _ITER_RE.fullmatch(iter_dir.name)
    return int(match.group("iteration")) if match is not None else None


def _checkpoint(iter_dir: Path) -> Path | None:
    for name in _CHECKPOINT_NAMES:
        candidate = iter_dir / name
        if _plain_nonempty_file(candidate):
            return candidate
    return None


def _readable_npy_member(stream: BinaryIO) -> bool:
    """Validate an NPY header without loading or allocating array data."""
    try:
        if stream.read(6) != b"\x93NUMPY":
            return False
        version = stream.read(2)
        if version == b"\x01\x00":
            raw_header_size = stream.read(2)
            if len(raw_header_size) != 2:
                return False
            header_size = struct.unpack("<H", raw_header_size)[0]
            encoding = "latin1"
        elif version in {b"\x02\x00", b"\x03\x00"}:
            raw_header_size = stream.read(4)
            if len(raw_header_size) != 4:
                return False
            header_size = struct.unpack("<I", raw_header_size)[0]
            encoding = "utf-8" if version == b"\x03\x00" else "latin1"
        else:
            return False
        # Real NumPy headers are small. Bound parsing so a hostile legacy
        # archive cannot make policy listing allocate arbitrary memory.
        if header_size <= 0 or header_size > 1024 * 1024:
            return False
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size or not raw_header.endswith(b"\n"):
            return False
        header = ast.literal_eval(raw_header.decode(encoding).strip())
        return (
            isinstance(header, dict)
            and isinstance(header.get("descr"), (str, list))
            and type(header.get("fortran_order")) is bool
            and isinstance(header.get("shape"), tuple)
            and all(type(size) is int and size >= 0 for size in header["shape"])
        )
    except (OSError, UnicodeError, ValueError, SyntaxError, struct.error):
        return False


def _readable_npz(path: Path) -> bool:
    if not _plain_nonempty_file(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.endswith(".npy")
            ]
            if not members or any(info.flag_bits & 0x1 for info in members):
                return False
            for info in members:
                with archive.open(info, "r") as member:
                    if not _readable_npy_member(member):
                        return False
            return True
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False


def _mp4_boxes(
    handle: BinaryIO, start: int, end: int,
) -> list[tuple[bytes, int, int]] | None:
    """Parse one bounded MP4 box level without reading media payloads."""
    boxes: list[tuple[bytes, int, int]] = []
    cursor = start
    try:
        while cursor < end:
            if end - cursor < 8:
                return None
            handle.seek(cursor)
            header = handle.read(8)
            if len(header) != 8:
                return None
            size32, kind = struct.unpack(">I4s", header)
            header_size = 8
            if size32 == 1:
                raw_size = handle.read(8)
                if len(raw_size) != 8:
                    return None
                size = struct.unpack(">Q", raw_size)[0]
                header_size = 16
            elif size32 == 0:
                size = end - cursor
            else:
                size = size32
            if size < header_size or cursor + size > end:
                return None
            boxes.append((kind, cursor + header_size, cursor + size))
            cursor += size
        return boxes if cursor == end else None
    except (OSError, struct.error):
        return None


def _readable_mp4(path: Path) -> bool:
    """Require the structural boxes of a playable MP4, without decoding it."""
    if not _plain_nonempty_file(path):
        return False
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            top = _mp4_boxes(handle, 0, file_size)
            if top is None:
                return False
            by_type = {kind: (start, end) for kind, start, end in top}
            if not {b"ftyp", b"moov", b"mdat"}.issubset(by_type):
                return False
            ftyp_start, ftyp_end = by_type[b"ftyp"]
            if ftyp_end - ftyp_start < 8:
                return False
            moov_start, moov_end = by_type[b"moov"]
            moov = _mp4_boxes(handle, moov_start, moov_end)
            if moov is None or not any(kind == b"mvhd" for kind, _, _ in moov):
                return False
            tracks = [(start, end) for kind, start, end in moov if kind == b"trak"]
            if not tracks:
                return False
            for track_start, track_end in tracks:
                track = _mp4_boxes(handle, track_start, track_end)
                if track is None:
                    continue
                track_types = {kind for kind, _, _ in track}
                if {b"tkhd", b"mdia"}.issubset(track_types):
                    return True
            return False
    except OSError:
        return False


def _has_valid_schema2_completion_marker(iter_dir: Path) -> bool:
    """Return whether a pre-phase-manifest marker pins this checkpoint.

    Schema 2 remains readable only for recovery/backward compatibility.  It is
    not enough to earn the modern ``attested`` authority because it does not
    bind train/rollout request, effective input, and output receipts.
    """
    iter_dir = Path(iter_dir)
    iteration = _iteration_index(iter_dir)
    marker = _json_object(iter_dir / "iteration_complete.json")
    checkpoint = _checkpoint(iter_dir)
    if iteration is None or marker is None or checkpoint is None:
        return False
    if (
        marker.get("schema") != 2
        or marker.get("state") != "completed"
        or type(marker.get("iter")) is not int
        or marker.get("iter") != iteration
    ):
        return False
    disclosed = marker.get("checkpoint")
    if not isinstance(disclosed, str) or not disclosed.strip():
        return False
    expected_sha256 = marker.get("checkpoint_sha256")
    expected_bytes = marker.get("checkpoint_bytes")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or type(expected_bytes) is not int
        or expected_bytes <= 0
    ):
        return False
    candidate = Path(disclosed).expanduser()
    if not candidate.is_absolute():
        candidate = iter_dir / candidate
    try:
        if (
            candidate.is_symlink()
            or candidate.resolve(strict=True) != checkpoint.resolve(strict=True)
        ):
            return False
        digest = hashlib.sha256()
        actual_bytes = 0
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                actual_bytes += len(chunk)
        return (
            actual_bytes == expected_bytes
            and digest.hexdigest() == expected_sha256
        )
    except (OSError, RuntimeError):
        return False


def attested_completion_receipt(
    iter_dir: Path,
) -> dict[str, Any] | None:
    """Return a fully reverified schema-3 completion receipt, or ``None``.

    Importing this pure core helper is CPU/data-only.  It does not import an
    adapter, simulator, CUDA, or launch a subprocess.
    """
    try:
        from sculptor.run_manifests import verify_iteration_completion_marker

        return verify_iteration_completion_marker(Path(iter_dir))
    except Exception:  # noqa: BLE001 - API authority always fails closed
        return None


def has_valid_completion_marker(iter_dir: Path) -> bool:
    """Return whether any recognized checkpoint-bound marker is valid.

    Callers making scientific or deployment claims must use
    :func:`iteration_completion_authority`; this compatibility predicate also
    recognizes schema 2 so older retained runs remain recoverable.
    """
    return (
        attested_completion_receipt(iter_dir) is not None
        or _has_valid_schema2_completion_marker(iter_dir)
    )


def has_full_legacy_completion_evidence(iter_dir: Path) -> bool:
    """Admit a pre-marker iteration only with a complete evidence quartet.

    A video alone is presentation evidence, and a fitness scalar alone may be
    left by a partial evaluator.  The canonical behavior JSON, trajectory,
    video, and finite objective result together prove that the legacy rollout
    and fitness stages returned after preserving a checkpoint.
    """
    iter_dir = Path(iter_dir)
    if _iteration_index(iter_dir) is None or _checkpoint(iter_dir) is None:
        return False
    rollout_dir = iter_dir / "rollout"
    if rollout_dir.is_symlink() or not rollout_dir.is_dir():
        return False
    behavior = _json_object(rollout_dir / "behavior.json")
    fitness = _json_object(iter_dir / "fitness.json")
    if behavior is None or fitness is None:
        return False
    raw_fitness = fitness.get("fitness")
    if (
        not isinstance(raw_fitness, (int, float))
        or isinstance(raw_fitness, bool)
        or not math.isfinite(float(raw_fitness))
    ):
        return False
    return (
        _readable_npz(rollout_dir / "trajectory.npz")
        and _readable_mp4(rollout_dir / "rollout.mp4")
    )


def is_completed_iteration(iter_dir: Path) -> bool:
    """Return the single completion decision used by policy/recovery APIs."""
    return (
        has_valid_completion_marker(iter_dir)
        or has_full_legacy_completion_evidence(iter_dir)
    )


CompletionAuthority = Literal["attested", "legacy_recovery"]


def iteration_completion_authority(
    iter_dir: Path,
) -> CompletionAuthority | None:
    """Classify why an iteration is admitted at the API boundary.

    Only a fully revalidated schema-3 marker that binds the exact train and
    rollout request, input, completion-manifest, and output bytes earns the
    modern ``attested`` authority.  Schema-2 markers and the legacy evidence
    quartet remain discoverable for recovery and reproducibility, but they
    must never inherit a deployment or scientific-success claim.
    """
    if attested_completion_receipt(iter_dir) is not None:
        return "attested"
    if (
        _has_valid_schema2_completion_marker(iter_dir)
        or has_full_legacy_completion_evidence(iter_dir)
    ):
        return "legacy_recovery"
    return None


__all__ = [
    "attested_completion_receipt",
    "has_full_legacy_completion_evidence",
    "has_valid_completion_marker",
    "is_completed_iteration",
    "iteration_completion_authority",
]
