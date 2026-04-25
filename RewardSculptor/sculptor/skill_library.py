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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# Import only the type we need. Stage's `init_skill_id` is the user-
# facing surface; `redecomposition_attempts` gates publish (Ship 17
# sub-stages are specialized and should not enter the general library).
from sculptor.mission import Mission, Stage


SCHEMA_VERSION = 1

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

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillRecord":
        ver = int(data.get("schema_version", SCHEMA_VERSION))
        if ver > SCHEMA_VERSION:
            raise SkillLibraryError(
                f"skill metadata.json schema_version={ver} is newer than "
                f"this sculptor build's SCHEMA_VERSION={SCHEMA_VERSION}; "
                "upgrade sculptor."
            )
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class SkillLibraryError(RuntimeError):
    """Raised on unrecoverable library failure (corrupt root, lock
    timeout, etc.). Per-record JSON parse errors are NOT raised —
    `list_compatible` skips bad records to keep the library
    queryable when one entry rots."""


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

        ckpt_sha = _file_sha256(checkpoint_path)
        skill_id = derive_skill_id(adapter_class, task_id, ckpt_sha)
        record_dir = self.root / skill_id
        ckpt_filename = checkpoint_path.name

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

                size = ckpt_dst.stat().st_size

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
                )
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
        path = self.root / skill_id / METADATA_FILENAME
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        try:
            return SkillRecord.from_dict(data)
        except (KeyError, TypeError, SkillLibraryError):
            return None

    def checkpoint_path_for(self, record: SkillRecord) -> Path:
        return self.root / record.skill_id / record.checkpoint_filename

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
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                # `.locks/` and any future hidden dirs.
                continue
            meta = child / METADATA_FILENAME
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                yield SkillRecord.from_dict(data)
            except (json.JSONDecodeError, OSError, KeyError, TypeError, SkillLibraryError):
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

    # ── decompose-time helpers ───────────────────────────────────────
    def list_for_decompose(self, *, top_k: int = 5) -> list[SkillRecord]:
        return self.library.list_compatible(
            adapter_class=self.adapter_class,
            task_id=self.task_id,
            robot_slug=self.robot_slug,
            top_k=top_k,
        )

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
            return None
        ckpt = self.library.checkpoint_path_for(record)
        if not ckpt.is_file():
            emit({
                "type": "skill_warm_start_skipped",
                "stage_name": stage.name,
                "skill_id": stage.init_skill_id,
                "reason": "checkpoint_missing",
                "expected_path": str(ckpt),
            })
            return None
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
