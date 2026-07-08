"""Non-destructive delete — trash for projects + missions (chunk A1).

Motivation: `DELETE /projects/{slug}` used to `shutil.rmtree` straight
through, so a permanently-destroyed project (including any trained
mission inside it) was unrecoverable by design. This module makes
delete a MOVE into `<trash_root>/<slug>--<timestamp>/` instead; the
only function in the codebase allowed to call `shutil.rmtree` on a
trashed entry is `purge()` below.

Trash root defaults to a sibling of `settings.resolved_projects_root`
(`~/.local/share/reward-sculptor/.trash/` on POSIX), overridable via
`RS_TRASH_ROOT`. There is no auto-expiry — `RS_TRASH_RETENTION_DAYS`
is reserved for a future sweep job but is not enforced here.

Entry layout:
    <trash_root>/<slug>--<UTC YYYYMMDDTHHMMSSZ>/
        trash_meta.json          # schema, kind, slug, origin_path, ...
        <...moved tree...>       # the project/mission dir, unmodified

`trash_meta.json` lives INSIDE the entry (not a separate index) so the
trash directory is self-describing even if this module's index logic
changes later — `list_trash()` just globs for the sidecar file.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


META_FILE = "trash_meta.json"
META_SCHEMA_VERSION = 1

# `<slug>--<timestamp>` — timestamp is `strftime("%Y%m%dT%H%M%SZ")`, so
# the whole entry_id is safe on both POSIX and Windows filesystems
# (no colons). Anchored full-match; rejects `/`, `..`, and anything
# else that could escape `trash_root` when joined onto it.
_ENTRY_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?--\d{8}T\d{6}Z$")


class TrashError(RuntimeError):
    """Base class for trash-module failures the routes layer maps to
    problem-detail responses."""


class InvalidEntryIdError(TrashError):
    """`entry_id` failed the traversal-safety pattern."""


class EntryNotFoundError(TrashError):
    """No trash entry with the given id (or its meta is unreadable in
    a way that prevents locating the moved tree)."""


class RestoreConflictError(TrashError):
    """Restore target's parent (the origin project) no longer exists —
    e.g. restoring a mission whose parent project was itself deleted
    (and possibly purged) since."""


def _resolved_trash_root(trash_root: Optional[Path] = None) -> Path:
    """Resolve the trash root. `trash_root` lets callers (routes,
    tests) skip re-reading `Settings()`; when omitted this reads the
    current environment via `backend.config.Settings`, mirroring how
    `conftest.py`'s `client` fixture constructs a fresh `Settings()`
    after `monkeypatch.setenv` — there is no long-lived settings
    singleton in this codebase for plain service functions to reuse."""
    if trash_root is not None:
        return trash_root
    from backend.config import Settings

    return Settings().resolved_trash_root


def _utc_stamp(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _dir_size_bytes(p: Path) -> Optional[int]:
    """Total bytes of all files under `p`. `None` on any walk failure
    rather than a partial (misleading) total."""
    if not p.is_dir():
        return None
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        return None
    return total


def _rmtree_forced(path: Path) -> None:
    """Cross-platform rmtree that clears Windows read-only bits on
    failure and retries. git marks `.git/objects/*` pack files
    read-only, which shutil.rmtree can't remove on Windows without
    this dance.

    This is the ONE place in the codebase allowed to hold this logic —
    `project_store.py` used to have its own copy; `purge()` below (the
    only permanent-delete entry point) is now the sole caller."""

    def _onexc(func: Any, p: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        if isinstance(exc, PermissionError) or (
            isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EACCES
        ):
            try:
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            try:
                func(p)
                return
            except Exception:
                pass
        raise exc

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onexc)
    else:  # pragma: no cover
        def _legacy(func, p, exc_info):
            _onexc(func, p, exc_info[1])
        shutil.rmtree(path, onerror=_legacy)


# ── entry_id helpers ─────────────────────────────────────────────────
def _validate_entry_id(entry_id: str) -> None:
    if not _ENTRY_ID_RE.fullmatch(entry_id):
        raise InvalidEntryIdError(f"malformed trash entry_id: {entry_id!r}")


def _entry_dir(trash_root: Path, entry_id: str) -> Path:
    _validate_entry_id(entry_id)
    return trash_root / entry_id


def _unique_entry_id(trash_root: Path, slug: str, now: Optional[datetime] = None) -> str:
    stamp = _utc_stamp(now)
    entry_id = f"{slug}--{stamp}"
    # Collision is astronomically unlikely (same slug deleted twice in
    # the same second) but cheap to guard — bump the seconds field
    # rather than silently overwriting an existing entry's meta.
    i = 1
    base_dt = now or datetime.now(tz=timezone.utc)
    while (trash_root / entry_id).exists():
        from datetime import timedelta

        entry_id = f"{slug}--{_utc_stamp(base_dt + timedelta(seconds=i))}"
        i += 1
        if i > 999:
            raise TrashError(f"could not allocate a unique trash entry_id for {slug!r}")
    return entry_id


# ── public API ───────────────────────────────────────────────────────
def move_to_trash(
    src_dir: Path,
    *,
    kind: str,
    slug: str,
    origin: Path,
    extra_meta: Optional[dict[str, Any]] = None,
    trash_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Move `src_dir` into the trash. Returns the meta dict (with
    `entry_id` added).

    `origin` is recorded so `restore()` knows where to move the tree
    back to; it is usually `src_dir` itself (the caller deletes in
    place) but is a separate parameter in case a future caller ever
    trashes from a staging copy.

    Caller is responsible for releasing any lock it holds on `src_dir`
    BEFORE calling this — `shutil.move` can't relocate a directory
    with an open file handle on Windows.
    """
    root = _resolved_trash_root(trash_root)
    root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=timezone.utc)
    entry_id = _unique_entry_id(root, slug, now)
    entry_dir = root / entry_id

    size_bytes = _dir_size_bytes(src_dir)

    # Move the tree itself INTO a subdir of entry_dir, not entry_dir's
    # root — trash_meta.json lives alongside it, not merged into it.
    # `shutil.move` creates the destination's parent chain if needed
    # but not the leaf; make the leaf ourselves first.
    entry_dir.mkdir(parents=True)
    payload_dir = entry_dir / src_dir.name
    shutil.move(str(src_dir), str(payload_dir))

    meta: dict[str, Any] = {
        "schema": META_SCHEMA_VERSION,
        "kind": kind,
        "slug": slug,
        "origin_path": str(origin),
        "deleted_at": now.isoformat(),
        "display_name": slug,
        "size_bytes": size_bytes,
    }
    if extra_meta:
        meta.update(extra_meta)

    (entry_dir / META_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    result = dict(meta)
    result["entry_id"] = entry_id
    return result


def _read_meta(entry_dir: Path) -> Optional[dict[str, Any]]:
    path = entry_dir / META_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _payload_dir(entry_dir: Path, meta: Optional[dict[str, Any]]) -> Optional[Path]:
    """The moved tree lives at `entry_dir/<original dir name>` — the
    original name is the basename of `origin_path` in meta. Falls back
    to "the one subdirectory in entry_dir" if meta is unreadable/
    malformed, so a corrupt sidecar never hides an otherwise-intact
    payload from restore/purge."""
    if meta is not None:
        origin = meta.get("origin_path")
        if isinstance(origin, str) and origin:
            candidate = entry_dir / Path(origin).name
            if candidate.is_dir():
                return candidate
    subdirs = [d for d in entry_dir.iterdir() if d.is_dir()] if entry_dir.is_dir() else []
    return subdirs[0] if len(subdirs) == 1 else None


def list_trash(*, trash_root: Optional[Path] = None) -> list[dict[str, Any]]:
    """List every trash entry, newest first. An entry with unreadable
    meta is still listed (never silently hidden) — its dict carries
    `"unreadable": True` plus whatever we could infer from the
    directory name so the user can still find/purge it."""
    root = _resolved_trash_root(trash_root)
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for entry_dir in root.iterdir():
        if not entry_dir.is_dir():
            continue
        meta = _read_meta(entry_dir)
        if meta is None:
            # Best-effort reconstruction from the dir name so the
            # entry is still actionable (restore will fail cleanly;
            # purge always works).
            entry_id = entry_dir.name
            slug = entry_id.rsplit("--", 1)[0] if "--" in entry_id else entry_id
            out.append({
                "entry_id": entry_id,
                "schema": None,
                "kind": None,
                "slug": slug,
                "origin_path": None,
                "deleted_at": None,
                "display_name": slug,
                "size_bytes": _dir_size_bytes(entry_dir),
                "unreadable": True,
            })
            continue
        row = dict(meta)
        row["entry_id"] = entry_dir.name
        row.setdefault("unreadable", False)
        out.append(row)

    out.sort(key=lambda r: r.get("deleted_at") or "", reverse=True)
    return out


def restore(entry_id: str, *, trash_root: Optional[Path] = None) -> Path:
    """Move a trash entry back to its `origin_path`. On a name
    collision at the destination, restores to `<slug>-restored-N`
    instead of overwriting whatever now occupies `origin_path`.

    Raises `EntryNotFoundError` if the entry doesn't exist, or
    `RestoreConflictError` if the origin's parent directory is gone
    (e.g. restoring a mission whose project was itself deleted).
    """
    root = _resolved_trash_root(trash_root)
    entry_dir = _entry_dir(root, entry_id)
    if not entry_dir.is_dir():
        raise EntryNotFoundError(f"no trash entry {entry_id!r}")

    meta = _read_meta(entry_dir)
    payload = _payload_dir(entry_dir, meta)
    if payload is None or not payload.is_dir():
        raise EntryNotFoundError(
            f"trash entry {entry_id!r} has no recoverable payload directory"
        )
    if meta is None or not isinstance(meta.get("origin_path"), str):
        raise EntryNotFoundError(
            f"trash entry {entry_id!r} has no usable trash_meta.json — "
            "origin_path is unknown, cannot restore automatically"
        )

    origin = Path(meta["origin_path"])
    slug = str(meta.get("slug") or payload.name)

    if not origin.parent.is_dir():
        raise RestoreConflictError(
            f"cannot restore {entry_id!r}: parent directory {origin.parent} "
            "no longer exists (its project may have been deleted/purged)"
        )

    dest = origin
    if dest.exists():
        i = 2
        while True:
            candidate = origin.parent / f"{slug}-restored-{i}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
            if i > 999:
                raise TrashError(
                    f"could not allocate a restore destination for {entry_id!r}"
                )

    shutil.move(str(payload), str(dest))
    # Clean up the now-empty entry dir (meta + the emptied payload
    # parent). Best-effort — a leftover meta file is harmless clutter,
    # not a correctness issue, so don't fail the restore over it.
    try:
        _rmtree_forced(entry_dir)
    except OSError:
        pass
    return dest


def purge(entry_id: str, *, trash_root: Optional[Path] = None) -> None:
    """Permanently delete a trash entry. THE ONLY function in this
    codebase allowed to call `shutil.rmtree` on a trashed tree —
    everything else moves."""
    root = _resolved_trash_root(trash_root)
    entry_dir = _entry_dir(root, entry_id)
    if not entry_dir.is_dir():
        raise EntryNotFoundError(f"no trash entry {entry_id!r}")
    _rmtree_forced(entry_dir)
