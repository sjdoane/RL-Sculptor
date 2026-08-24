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
    errors.extend(
        _schema2_artifact_identity_errors(
            robot, clip_id, prov, root=root,
        )
    )
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
    rows.sort(key=lambda r: (r["robot"], r["clip_id"]))
    write_index(rows, root=root)
    return rows


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
        rebuild_index(root=r)
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
