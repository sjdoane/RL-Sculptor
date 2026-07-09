"""On-disk reference library: root location, provenance schema,
`index.jsonl` cache, clip_id rules, content hashing, license guard.

§R1_BUILD_SPEC decisions 3-4. The index is a REBUILDABLE cache —
`provenance.json` beside each clip is the single source of truth
(`rebuild_index` rescans disk and never trusts a stale index row).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Provenance schema version. Bump + document on any breaking field change.
PROVENANCE_SCHEMA = 1

#: clip_id charset (§decision 4): lowercase alnum + `_-`, must start
#: alnum, max 96 chars total.
CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")

INDEX_FILENAME = "index.jsonl"
REJECTS_FILENAME = "index_rejects.jsonl"
CLIP_FILENAME = "clip.npz"
PROVENANCE_FILENAME = "provenance.json"
PREVIEW_FILENAME = "preview.png"

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
    return (root or references_root()) / robot


def clip_dir(robot: str, clip_id: str, *, root: Optional[Path] = None) -> Path:
    validate_clip_id(clip_id)
    return robot_dir(robot, root=root) / clip_id


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


# ── provenance ───────────────────────────────────────────────────────────
def make_provenance(
    *,
    clip_id: str,
    robot: str,
    source: dict[str, Any],
    license: str,
    attribution: str,
    content_sha256_: str,
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
    clip_id = prov.get("clip_id")
    if isinstance(clip_id, str):
        try:
            validate_clip_id(clip_id)
        except ClipIdError as e:
            errors.append(str(e))
    src = prov.get("source")
    if src is not None and not isinstance(src, dict):
        errors.append("provenance.source must be an object")
    return errors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_provenance(
    robot: str, clip_id: str, prov: dict[str, Any], *, root: Optional[Path] = None,
) -> Path:
    """Persist provenance.json beside the clip (§decision-3(b)). Refuses
    to write an invalid record (license guard lives here — this is the
    single choke point every ingest/segment path must go through)."""
    errors = validate_provenance(prov)
    if errors:
        raise LicenseGuardError(
            "refusing to write provenance for clip "
            f"{clip_id!r}:\n  - " + "\n  - ".join(errors))
    d = clip_dir(robot, clip_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / PROVENANCE_FILENAME
    path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_provenance(robot: str, clip_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    path = clip_dir(robot, clip_id, root=root) / PROVENANCE_FILENAME
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


def write_index(rows: list[dict[str, Any]], *, root: Optional[Path] = None) -> Path:
    path = index_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def append_reject(reason: str, detail: dict[str, Any], *, root: Optional[Path] = None) -> Path:
    path = rejects_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"reason": reason, "ingested_at": _utc_now_iso(), **detail}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def rebuild_index(*, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Rescan every `<robot>/<clip_id>/provenance.json` on disk and
    rewrite `index.jsonl` from scratch. The index is a cache; provenance
    is truth (§decision 4) — this is the only correct way to recover
    from a stale/corrupt/missing index."""
    r = root or references_root()
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
                if errors:
                    continue
                has_preview = (clip_d / PREVIEW_FILENAME).is_file()
                rows.append(_index_row_from_provenance(prov, has_preview=has_preview))
    rows.sort(key=lambda r: (r["robot"], r["clip_id"]))
    write_index(rows, root=root)
    return rows


def indexed_content_hashes(*, root: Optional[Path] = None) -> set[str]:
    """content_sha256 values already present in the library (via
    provenance, not the possibly-stale index) — used by ingest for
    idempotency (§decision 9: re-ingest skips already-indexed content)."""
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
                if sha:
                    hashes.add(sha)
    return hashes


@dataclass
class LibraryClip:
    """A single ingested/segmented clip's on-disk identity, returned by
    ingest/segment so callers don't have to re-derive paths."""

    robot: str
    clip_id: str
    clip_path: Path
    provenance_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)
