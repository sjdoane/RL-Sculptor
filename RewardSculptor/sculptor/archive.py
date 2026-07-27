"""sculptor/archive.py — durable mission archive (chunk A2).

A mission's live tree (`<project>/.missions/<slug>/`) is meant to be
disposable: stages are scratch dirs the orchestrator can re-scaffold,
and iteration dirs carry gigabytes of `logs/` (tensorboard events +
rsl_rl periodic checkpoints) and `edit_candidates/` (raw LLM-sample
sources) that are pure working set, never needed again once an
iteration is diagnosed. Deleting a mission today throws all of that
away — including the parts worth keeping (rollout videos, the winning
checkpoint, the reward/env spec that produced it).

`archive_mission` copies the KEEP-worthy subset of a mission (or one
of its stages, called incrementally per-stage as training progresses)
into a durable, project-independent location:

    <saved_root>/<project_slug>--<mission_slug>--<UTCstamp>/
        manifest.json
        mission.json / telemetry.json / provenance.json / llm_calls.jsonl
        stage_metrics/**
        reports/**
        stages/<name>/
            config.toml / CHANGELOG.md / kg_seeds.yml
            rewards/** / env/** / reports/**
            runs/iter_N/
                metrics.json / diagnosis.json / reward_spec.json / ...
                checkpoint.pt|zip        (only kept iters — see below)
                rollout/rollout.mp4, behavior.json, mjcf_limits.json,
                        reward_trajectory.json
                rollout/trajectory.npz, rollout/keyframes/  (kept iters only)

Checkpoints are the expensive part (10s-100s of MB each), so only a
bounded set survives per stage: the FINAL iter, the BEST-scoring iter,
and any caller-PINNED iters. Everything else keeps its lightweight
metrics/diagnosis/rollout-video trail but drops the checkpoint (and
the checkpoint-adjacent trajectory.npz / keyframes/, which are only
useful alongside a loadable checkpoint).

`archive_mission` is called once per stage-completion event (see the
mission orchestrator), each call passing the SAME `mission_dir`. The
first call mints a stamped entry id and records it in
`<mission_dir>/.archive_id`; later calls read that file back so every
stage of one mission run lands in one entry directory instead of
fragmenting across N timestamped dirs. `incremental=False` forces a
fresh entry (stamped now) regardless of any existing `.archive_id` —
useful for an explicit "save a NEW copy" action distinct from the
default "keep appending to this run's entry" behavior.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

_ARCHIVE_ID_FILENAME = ".archive_id"
_STAMP_FMT = "%Y%m%dT%H%M%SZ"

# Mission-root files copied verbatim when present.
_MISSION_FILES = (
    "mission.json", "telemetry.json", "provenance.json", "llm_calls.jsonl",
)
_MISSION_DIRS = ("stage_metrics", "reports")

# Per-stage files/dirs copied verbatim when present.
_STAGE_FILES = ("config.toml", "CHANGELOG.md", "kg_seeds.yml")
_STAGE_DIRS = ("rewards", "env", "reports")

# Per-iteration files ALWAYS copied (never gated on checkpoint status).
_ITER_ROOT_FILES = (
    "metrics.json", "diagnosis.json", "reward_spec.json",
    "realism_audit.json", "partition_gate.json", "env_spec.json",
    "edit_candidates.json", "reward_trajectory.json",
)
_ITER_ROLLOUT_FILES = (
    "rollout.mp4", "behavior.json", "mjcf_limits.json",
    "reward_trajectory.json",
)
# Checkpoint-adjacent — copied ONLY beside a kept checkpoint.
_ITER_CHECKPOINT_SIBLINGS = ("trajectory.npz",)
_CHECKPOINT_NAMES = ("checkpoint.pt", "checkpoint.zip")

# Directories intentionally never copied (working-set only): tensorboard
# events + periodic rsl_rl checkpoints, and raw LLM edit-candidate
# sources. `rollout_fresh_*/` iteration dirs are handled specially —
# every file inside them is skipped EXCEPT behavior.json.
_NEVER_COPY_DIR_NAMES = frozenset({"logs", "edit_candidates", "__pycache__"})


class ArchiveError(RuntimeError):
    """Raised when a mission tree can't be archived at all (bad mission_dir,
    unreadable mission.json, etc.). Call sites are expected to fail-open —
    archiving must never abort a training run — but the function itself
    raises normally so tests and callers can tell success from failure."""


@dataclass
class ArchiveResult:
    entry_dir: Path
    total_bytes: int
    dropped_bytes: int
    kept_checkpoints: list[dict[str, Any]] = field(default_factory=list)


# ── location ─────────────────────────────────────────────────────────────
def saved_root() -> Path:
    """Root directory archived missions live under.

    `RS_SAVED_ROOT` overrides; default `~/.local/share/reward-sculptor/
    saved/`, sibling to the existing `.../reward-sculptor/projects/` tree.
    """
    override = os.environ.get("RS_SAVED_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / "reward-sculptor" / "saved"


# ── entry id / .archive_id bookkeeping ──────────────────────────────────
def _utc_stamp(now: Optional[datetime] = None) -> str:
    dt = now if now is not None else datetime.now(timezone.utc)
    return dt.strftime(_STAMP_FMT)


def _entry_id(project_slug: str, mission_slug: str, stamp: str) -> str:
    return f"{project_slug}--{mission_slug}--{stamp}"


def _read_archive_id(mission_dir: Path) -> Optional[str]:
    p = mission_dir / _ARCHIVE_ID_FILENAME
    if not p.is_file():
        return None
    try:
        entry_id = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return entry_id or None


def _write_archive_id(mission_dir: Path, entry_id: str) -> None:
    (mission_dir / _ARCHIVE_ID_FILENAME).write_text(entry_id, encoding="utf-8")


def _resolve_entry_id(
    mission_dir: Path, *, project_slug: str, mission_slug: str,
    incremental: bool,
) -> str:
    """Pick the entry id this call writes into.

    `incremental=True` (the common case — one call per stage-completion
    during a mission run) reuses the id recorded by this run's FIRST
    call, so every stage lands in the same entry directory. A fresh id
    is minted (and recorded) the first time, or whenever no
    `.archive_id` exists yet, or when the caller opts out via
    `incremental=False`.
    """
    if incremental:
        existing = _read_archive_id(mission_dir)
        if existing:
            return existing
    entry_id = _entry_id(project_slug, mission_slug, _utc_stamp())
    if incremental:
        _write_archive_id(mission_dir, entry_id)
    return entry_id


# ── small IO helpers ─────────────────────────────────────────────────────
def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _dir_size(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _copy_file(src: Path, dst: Path) -> int:
    """Copy one file, creating parents. Returns bytes copied (0 if the
    source doesn't exist — callers already gate on existence, but the
    check is cheap insurance against a TOCTOU delete mid-archive)."""
    if not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        return dst.stat().st_size
    except OSError:
        return 0


def _copy_dir(src: Path, dst: Path) -> int:
    """Copy a directory tree, skipping `_NEVER_COPY_DIR_NAMES` subdirs
    wherever they appear (e.g. `rewards/__pycache__`). Returns total
    bytes copied. `dirs_exist_ok=True` so re-invocation of an already-
    archived stage (e.g. re-running `archive_mission` for the SAME
    stage after more iterations landed) merges rather than errors."""
    if not src.is_dir():
        return 0

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in _NEVER_COPY_DIR_NAMES}

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore)
    return _dir_size(dst)


# ── checkpoint retention ────────────────────────────────────────────────
def _find_checkpoint(iter_dir: Path) -> Optional[Path]:
    for name in _CHECKPOINT_NAMES:
        p = iter_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _iter_index(iter_dir: Path) -> Optional[int]:
    name = iter_dir.name
    if not name.startswith("iter_"):
        return None
    try:
        return int(name[len("iter_"):])
    except ValueError:
        return None


def _iter_fitness(
    iter_dir: Path, iter_idx: int, primary_history: list[float],
) -> Optional[float]:
    """Per-iter score used to pick the "best" checkpoint.

    Prefers `rollout/behavior.json["fitness"]`; falls back to the
    stage's `reports/metric_history.json` primary-metric value at this
    iter's index when fitness is absent (older runs predate the
    fitness field — see `sculptor/export.py`'s identical fallback).
    """
    behavior = _load_json(iter_dir / "rollout" / "behavior.json", {}) or {}
    fitness = behavior.get("fitness")
    if isinstance(fitness, (int, float)):
        return float(fitness)
    if 0 <= iter_idx < len(primary_history):
        try:
            return float(primary_history[iter_idx])
        except (TypeError, ValueError):
            return None
    return None


def _select_kept_checkpoint_iters(
    stage_dir: Path, iter_dirs: list[Path], pinned: set[int],
) -> dict[int, list[str]]:
    """Which iter indices keep their checkpoint, and why.

    Returns `{iter_idx: [reasons]}` — an iter can be kept for more than
    one reason (e.g. the final iter also happens to be the best one).
    Only iters that actually HAVE a checkpoint on disk are eligible;
    a pinned iter with no checkpoint is silently dropped from the
    result (nothing to keep).
    """
    with_ckpt: list[tuple[int, Path]] = []
    for d in iter_dirs:
        idx = _iter_index(d)
        if idx is None:
            continue
        if _find_checkpoint(d) is not None:
            with_ckpt.append((idx, d))
    if not with_ckpt:
        return {}
    with_ckpt.sort(key=lambda t: t[0])

    reasons: dict[int, list[str]] = {}

    final_idx = with_ckpt[-1][0]
    reasons.setdefault(final_idx, []).append("final")

    history_obj = _load_json(stage_dir / "reports" / "metric_history.json", {}) or {}
    primary_history = list(history_obj.get("history", []) or [])

    best_idx: Optional[int] = None
    best_score: Optional[float] = None
    for idx, d in with_ckpt:
        score = _iter_fitness(d, idx, primary_history)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None:
        reasons.setdefault(best_idx, []).append("best")

    ckpt_idx_set = {idx for idx, _ in with_ckpt}
    for idx in sorted(pinned & ckpt_idx_set):
        reasons.setdefault(idx, []).append("pinned")

    return reasons


# ── per-iteration copy ───────────────────────────────────────────────────
def _archive_iteration(
    *, src_iter_dir: Path, dst_iter_dir: Path, keep_checkpoint: bool,
) -> tuple[int, int]:
    """Copy one iteration's artifacts. Returns (bytes_copied, bytes_dropped)
    — `bytes_dropped` only accounts for the checkpoint + its trajectory/
    keyframes siblings when `keep_checkpoint=False` (the always-dropped
    `logs/`/`edit_candidates/` dirs aren't sized — they're pure working
    set, never a retention DECISION, so counting them would conflate
    "we chose not to keep this" with "this was never eligible")."""
    copied = 0
    dropped = 0

    for name in _ITER_ROOT_FILES:
        copied += _copy_file(src_iter_dir / name, dst_iter_dir / name)

    src_rollout = src_iter_dir / "rollout"
    dst_rollout = dst_iter_dir / "rollout"
    for name in _ITER_ROLLOUT_FILES:
        copied += _copy_file(src_rollout / name, dst_rollout / name)

    # `rollout_fresh_*/` sibling dirs — behavior.json only.
    if src_iter_dir.is_dir():
        for child in src_iter_dir.iterdir():
            if child.is_dir() and child.name.startswith("rollout_fresh_"):
                copied += _copy_file(
                    child / "behavior.json",
                    dst_iter_dir / child.name / "behavior.json")

    ckpt = _find_checkpoint(src_iter_dir)
    if ckpt is not None:
        if keep_checkpoint:
            copied += _copy_file(ckpt, dst_iter_dir / ckpt.name)
            for name in _ITER_CHECKPOINT_SIBLINGS:
                copied += _copy_file(
                    src_rollout / name, dst_rollout / name)
            keyframes_src = src_rollout / "keyframes"
            if keyframes_src.is_dir():
                copied += _copy_dir(
                    keyframes_src, dst_rollout / "keyframes")
        else:
            dropped += ckpt.stat().st_size
            for name in _ITER_CHECKPOINT_SIBLINGS:
                p = src_rollout / name
                if p.is_file():
                    dropped += p.stat().st_size
            keyframes_src = src_rollout / "keyframes"
            if keyframes_src.is_dir():
                dropped += _dir_size(keyframes_src)

    return copied, dropped


# ── manifest ─────────────────────────────────────────────────────────────
def _stage_status(mission_doc: dict, stage_name: str) -> str:
    for s in mission_doc.get("stages", []) or []:
        if isinstance(s, dict) and s.get("name") == stage_name:
            return str(s.get("status", "pending"))
    return "unknown"


def _relpath(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def _find_stage_dirs(stages_root: Path) -> list[Path]:
    if not stages_root.is_dir():
        return []
    return sorted(
        (d for d in stages_root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )


def _find_iter_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    out = [d for d in runs_dir.iterdir()
           if d.is_dir() and _iter_index(d) is not None]
    out.sort(key=lambda d: _iter_index(d) or 0)
    return out


# ── public entry ─────────────────────────────────────────────────────────
def archive_mission(
    mission_dir: Path,
    dest_root: Path,
    *,
    project_slug: str,
    pinned: Optional[dict[str, set[int]]] = None,
    incremental: bool = True,
) -> ArchiveResult:
    """Copy the durable subset of a mission into `dest_root`.

    `mission_dir` is the LIVE mission directory (`<project>/.missions/
    <slug>/`) — must contain `mission.json`. `dest_root` is the archive
    root (typically `saved_root()`, but callers pass a tmp dir in
    tests). `project_slug` identifies the parent project; `mission_slug`
    is always `mission_dir.name`.

    `pinned` maps stage name → set of iter indices whose checkpoint
    must be kept regardless of final/best status. `incremental=True`
    (default) reuses the entry this mission run already started (see
    `.archive_id` in the module docstring); pass `False` to force a
    brand-new timestamped entry.

    Copies EVERY stage under `<mission_dir>/stages/`, re-running the
    retention logic per stage, so a second call after more iterations
    have landed simply adds/updates rather than duplicating — safe to
    call once per stage-completion event.

    Raises `ArchiveError` if `mission_dir` doesn't look like a mission
    (no `mission.json`). Never silently swallows other IO errors —
    fail-open belongs at the call site, not here.
    """
    mission_dir = Path(mission_dir)
    dest_root = Path(dest_root)
    mission_json = mission_dir / "mission.json"
    if not mission_json.is_file():
        raise ArchiveError(
            f"{mission_dir} has no mission.json — not a mission directory")

    mission_doc = _load_json(mission_json, None)
    if not isinstance(mission_doc, dict):
        raise ArchiveError(f"{mission_json} did not parse as a JSON object")

    mission_slug = mission_dir.name
    pinned = pinned or {}

    entry_id = _resolve_entry_id(
        mission_dir, project_slug=project_slug, mission_slug=mission_slug,
        incremental=incremental)
    entry_dir = dest_root / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    dropped_bytes = 0

    # ── mission-root artifacts ──────────────────────────────────────
    for name in _MISSION_FILES:
        total_bytes += _copy_file(mission_dir / name, entry_dir / name)
    for name in _MISSION_DIRS:
        total_bytes += _copy_dir(mission_dir / name, entry_dir / name)

    # ── stages ───────────────────────────────────────────────────────
    stage_manifests: list[dict[str, Any]] = []
    stages_root = mission_dir / "stages"
    for stage_src in _find_stage_dirs(stages_root):
        stage_name = stage_src.name
        stage_dst = entry_dir / "stages" / stage_name

        for name in _STAGE_FILES:
            total_bytes += _copy_file(stage_src / name, stage_dst / name)
        for name in _STAGE_DIRS:
            total_bytes += _copy_dir(stage_src / name, stage_dst / name)

        iter_dirs = _find_iter_dirs(stage_src / "runs")
        stage_pinned = set(pinned.get(stage_name, set()) or set())
        kept_reasons = _select_kept_checkpoint_iters(
            stage_src, iter_dirs, stage_pinned)

        kept_checkpoints: list[dict[str, Any]] = []
        video_relpaths: list[str] = []
        for src_iter_dir in iter_dirs:
            idx = _iter_index(src_iter_dir)
            if idx is None:
                continue
            dst_iter_dir = stage_dst / "runs" / src_iter_dir.name
            reasons = kept_reasons.get(idx)
            copied, dropped = _archive_iteration(
                src_iter_dir=src_iter_dir, dst_iter_dir=dst_iter_dir,
                keep_checkpoint=bool(reasons))
            total_bytes += copied
            dropped_bytes += dropped

            video_path = dst_iter_dir / "rollout" / "rollout.mp4"
            if video_path.is_file():
                video_relpaths.append(_relpath(video_path, entry_dir))

            if reasons:
                ckpt = _find_checkpoint(dst_iter_dir)
                # "final" > "best" > "pinned" as the reported PRIMARY
                # reason when an iter qualifies more than one way; the
                # manifest's `reason` field is singular for readability,
                # but nothing about the copy decision itself is lossy.
                primary = next(
                    (r for r in ("final", "best", "pinned") if r in reasons),
                    reasons[0])
                kept_checkpoints.append({
                    "iter": idx,
                    "reason": primary,
                    "bytes": ckpt.stat().st_size if ckpt else 0,
                })

        best_metric = None
        history_obj = _load_json(
            stage_src / "reports" / "metric_history.json", {}) or {}
        hist = history_obj.get("history") or []
        if hist:
            try:
                best_metric = max(float(v) for v in hist)
            except (TypeError, ValueError):
                best_metric = None

        stage_manifests.append({
            "name": stage_name,
            "status": _stage_status(mission_doc, stage_name),
            "best_metric": best_metric,
            "kept_checkpoints": sorted(
                kept_checkpoints, key=lambda k: k["iter"]),
            "n_iters": len(iter_dirs),
            "videos": video_relpaths,
        })

    manifest = {
        "schema": SCHEMA_VERSION,
        "entry_id": entry_id,
        "project_slug": project_slug,
        "mission_slug": mission_slug,
        "goal": mission_doc.get("goal", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mission_dir": str(mission_dir),
        "stages": stage_manifests,
        "total_bytes": total_bytes,
        "dropped_bytes": dropped_bytes,
    }
    (entry_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8")

    return ArchiveResult(
        entry_dir=entry_dir,
        total_bytes=total_bytes,
        dropped_bytes=dropped_bytes,
        kept_checkpoints=[
            {"stage": sm["name"], **kc}
            for sm in stage_manifests for kc in sm["kept_checkpoints"]
        ],
    )


# ── read-side ────────────────────────────────────────────────────────────
def read_manifest(entry_dir: Path) -> dict[str, Any]:
    """Read `<entry_dir>/manifest.json`. Raises (does not swallow) on a
    missing or corrupt file — callers that want fail-open should catch
    at the call site, same convention as `archive_mission`."""
    path = Path(entry_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved(dest_root: Path) -> list[dict[str, Any]]:
    """Every archived entry under `dest_root`, newest first.

    Skips entries whose `manifest.json` is missing/corrupt rather than
    raising — a browse/list view must not go blank because one entry
    got interrupted mid-write."""
    dest_root = Path(dest_root)
    if not dest_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(dest_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            out.append(read_manifest(d))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out
