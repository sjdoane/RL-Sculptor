"""Immutable world artifacts and one authoritative atomic selection pointer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

_VERSION_RE = re.compile(
    r"^(world|task|resolved_eval|channel_catalog|clarifications|selection)_v(\d+)\.json$")
_KINDS = {
    "world", "task", "resolved_eval", "channel_catalog", "clarifications",
}
_REQUIRED_SELECTION_REFS = frozenset({
    "reward", "env_spec", "world", "task", "resolved_eval",
    "channel_catalog", "clarifications",
})


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    version: str
    path: str
    sha256: str

    @classmethod
    def from_path(
        cls, kind: str, version: str, path: Path, *, base: Path,
    ) -> "ArtifactRef":
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            relative = str(resolved)
        return cls(kind=kind, version=version, path=relative,
                   sha256=file_sha256(resolved))


@dataclass(frozen=True)
class ArtifactSelection:
    selection_version: int
    created_at: float
    refs: dict[str, ArtifactRef]
    tuple_hash: str
    evaluation_lineage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_version": self.selection_version,
            "created_at": self.created_at,
            "refs": {k: asdict(v) for k, v in sorted(self.refs.items())},
            "tuple_hash": self.tuple_hash,
            "evaluation_lineage": self.evaluation_lineage,
        }


class WorldArtifactStore:
    """Project-local immutable store rooted at ``<project>/env``.

    ``selection_current.json`` is the sole authoritative pointer. Individual
    current files are intentionally not maintained, preventing a crash between
    several replacements from exposing a mixed reward/world/task tuple.
    """

    def __init__(self, project_dir: Path | str):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.env_dir = self.project_dir / "env"
        self.env_dir.mkdir(parents=True, exist_ok=True)

    @property
    def selection_path(self) -> Path:
        return self.env_dir / "selection_current.json"

    def _next_version(self, kind: str) -> int:
        if kind not in _KINDS | {"selection"}:
            raise ValueError(f"unsupported artifact kind {kind!r}")
        best = 0
        prefix = f"{kind}_v"
        for path in self.env_dir.glob(f"{prefix}*.json"):
            match = _VERSION_RE.match(path.name)
            if match and match.group(1) == kind:
                best = max(best, int(match.group(2)))
        return best + 1

    def write_json(self, kind: str, payload: Mapping[str, Any]) -> ArtifactRef:
        """Write a new immutable canonical JSON artifact."""
        version_number = self._next_version(kind)
        version = f"v{version_number}"
        path = self.env_dir / f"{kind}_{version}.json"
        data = canonical_json_bytes(dict(payload))
        _atomic_write(path, data)
        return ArtifactRef.from_path(
            kind, version, path, base=self.project_dir)

    def write_rejected_draft(
        self, session_id: str, payload: Mapping[str, Any],
    ) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", session_id):
            raise ValueError("invalid author session id")
        path = self.env_dir / "rejected" / f"{session_id}.json"
        _atomic_write(path, canonical_json_bytes(dict(payload)))
        return path

    def resolve_ref(self, ref: ArtifactRef) -> Path:
        path = Path(ref.path)
        if not path.is_absolute():
            path = self.project_dir / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"selected {ref.kind} artifact is absent: {resolved}")
        actual = file_sha256(resolved)
        if actual != ref.sha256:
            raise ValueError(
                f"selected {ref.kind} hash mismatch: "
                f"expected {ref.sha256}, got {actual}")
        return resolved

    def _coerce_ref(self, key: str, value: Any) -> ArtifactRef:
        if isinstance(value, ArtifactRef):
            return value
        if isinstance(value, Mapping):
            return ArtifactRef(**value)
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve()
            return ArtifactRef.from_path(
                key, path.stem, path, base=self.project_dir)
        raise TypeError(f"invalid artifact reference for {key}: {value!r}")

    def promote(
        self, refs: Mapping[str, ArtifactRef | Mapping[str, Any] | Path | str],
        *, evaluation_lineage: str,
    ) -> ArtifactSelection:
        """Atomically promote a complete, verified artifact tuple."""
        missing = _REQUIRED_SELECTION_REFS - set(refs)
        if missing:
            raise ValueError(f"selection missing required refs {sorted(missing)}")
        normalized = {key: self._coerce_ref(key, value)
                      for key, value in refs.items()}
        for ref in normalized.values():
            self.resolve_ref(ref)
        tuple_payload = {k: asdict(v) for k, v in sorted(normalized.items())}
        tuple_hash = sha256_bytes(canonical_json_bytes(tuple_payload))
        number = self._next_version("selection")
        selection = ArtifactSelection(
            selection_version=number, created_at=time.time(),
            refs=normalized, tuple_hash=tuple_hash,
            evaluation_lineage=evaluation_lineage,
        )
        data = canonical_json_bytes(selection.to_dict())
        immutable = self.env_dir / f"selection_v{number}.json"
        _atomic_write(immutable, data)
        # This single replace is the commit point. Either the previous tuple
        # or this fully verified tuple is authoritative after interruption.
        _atomic_write(self.selection_path, data)
        return selection

    def read_selection(self) -> ArtifactSelection | None:
        if not self.selection_path.is_file():
            return None
        data = json.loads(self.selection_path.read_text(encoding="utf-8"))
        refs = {key: ArtifactRef(**value)
                for key, value in data["refs"].items()}
        missing = _REQUIRED_SELECTION_REFS - set(refs)
        if missing:
            raise ValueError(
                f"selection missing required refs {sorted(missing)}")
        selection = ArtifactSelection(
            selection_version=int(data["selection_version"]),
            created_at=float(data["created_at"]), refs=refs,
            tuple_hash=str(data["tuple_hash"]),
            evaluation_lineage=str(data["evaluation_lineage"]),
        )
        expected = sha256_bytes(canonical_json_bytes(
            {k: asdict(v) for k, v in sorted(refs.items())}))
        if expected != selection.tuple_hash:
            raise ValueError("selection tuple hash mismatch")
        for ref in refs.values():
            self.resolve_ref(ref)
        return selection

    def load_json_ref(self, ref: ArtifactRef) -> dict[str, Any]:
        return json.loads(self.resolve_ref(ref).read_text(encoding="utf-8"))

    def current_bundle(self) -> dict[str, Any] | None:
        selection = self.read_selection()
        if selection is None:
            return None
        return {
            key: (self.load_json_ref(ref)
                  if self.resolve_ref(ref).suffix == ".json"
                  else str(self.resolve_ref(ref)))
            for key, ref in selection.refs.items()
        } | {"selection": selection.to_dict()}
