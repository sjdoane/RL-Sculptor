"""Fitness backfill from `_execute_*.log` `[SCULPT-EVENT]` markers.

Missions run BEFORE commit f1c339d never got a `runs/iter_N/fitness.json`
written live — `sculpt.py`'s `iter_fitness` `[SCULPT-EVENT]` line survived
in the mission's `_execute_<job_id>.log`, but nothing replayed it back onto
disk. This module scans those logs and reconstructs `fitness.json` for
every `(stage, iter)` that has a `runs/iter_N/` directory but no
`fitness.json` of its own — it never overwrites an existing `fitness.json`
(a live-written record always wins over a backfilled one).

Steer values (`steer_fitness`/`steer_progress`/`naturalness_*`) are
recomputed under the CURRENT (post-D31) naturalness rules whenever a fresh
audit can run (`rollout/trajectory.npz` + `rollout/mjcf_limits.json` still
present) — sculptor's naturalness policy has moved since some of these
iters trained, so a byte-for-byte replay of what the live loop computed at
the time would be actively wrong. When the rollout files are gone, this
falls back to the HISTORICAL `realism_audit.json`'s recorded `naturalness`
dict (never rewriting that file — it stays the historical record) and
computes steer values from CURRENT fitness/progress against that recorded
naturalness. If neither source is available the steer fields are null.

Called synchronously from `POST .../missions/{mission_slug}/backfill-
fitness` (`backend/routes/missions.py`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_EVENT_TAG = "[SCULPT-EVENT] "


def _iter_log_files(mission_dir: Path) -> list[Path]:
    """`_execute_*.log` files directly under `mission_dir`
    (`backend.services.mission_jobs.run_mission_execute_job` names each
    one `_execute_<job_id>.log`), oldest→newest by mtime — a mission
    re-run after a crash produces a second log, and its events must win
    over the first's for the same `(stage, iter)` key."""
    candidates = [p for p in mission_dir.glob("_execute_*.log") if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates


def _iter_sculpt_events(log_path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed `[SCULPT-EVENT]` JSON dicts from one log file, in file
    order. The marker is looked up with `str.find` (not `str.startswith`)
    so it's tolerant of anything that might precede it on the line;
    malformed/partial JSON on a given line is skipped rather than fatal —
    an in-flight mission's log can end mid-write."""
    try:
        fh = log_path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    try:
        for line in fh:
            idx = line.find(_EVENT_TAG)
            if idx < 0:
                continue
            payload_str = line[idx + len(_EVENT_TAG):].strip()
            if not payload_str:
                continue
            try:
                payload = json.loads(payload_str)
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("type"):
                yield payload
    finally:
        fh.close()


def _load_json_dict(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _recompute_steer(
    iter_dir: Path, fitness: Optional[float], progress: Optional[float],
) -> dict[str, Any]:
    """Best-effort steer-fitness/naturalness recompute for one backfilled
    iter. Never raises — any failure (missing sculptor, corrupt trajectory,
    shape drift) degrades to the all-null/1.0 no-op defaults, same posture
    as `sculpt.py`'s own realism-audit try/except."""
    out: dict[str, Any] = {
        "steer_fitness": None,
        "steer_progress": None,
        "naturalness_factor": 1.0,
        "naturalness_flag": None,
        "naturalness_hard_reject": False,
    }

    naturalness: Optional[dict[str, Any]] = None
    traj_path = iter_dir / "rollout" / "trajectory.npz"
    limits_path = iter_dir / "rollout" / "mjcf_limits.json"
    try:
        if traj_path.is_file() and limits_path.is_file():
            from sculptor.adapters.realism import audit_rollout, naturalness_channel

            audit = audit_rollout(
                trajectory_path=traj_path, limits_path=limits_path,
            )
            naturalness = naturalness_channel(audit)
        else:
            audit_doc = _load_json_dict(iter_dir / "realism_audit.json")
            if isinstance(audit_doc, dict):
                recorded = audit_doc.get("naturalness")
                if isinstance(recorded, dict):
                    naturalness = recorded
    except Exception:  # noqa: BLE001 — recompute is best-effort
        naturalness = None

    if naturalness is not None:
        try:
            from sculptor.adapters.realism import steer_fitness as _steer_fitness

            out["steer_fitness"] = _steer_fitness(fitness, naturalness)
            out["steer_progress"] = _steer_fitness(progress, naturalness)
            out["naturalness_factor"] = float(
                naturalness.get("steer_factor", 1.0) or 1.0
            )
            flag = naturalness.get("flag")
            out["naturalness_flag"] = flag if isinstance(flag, str) else None
            out["naturalness_hard_reject"] = bool(naturalness.get("hard_reject"))
        except Exception:  # noqa: BLE001 — recompute is best-effort
            pass

    return out


def backfill_mission_fitness(mission_dir: Path) -> dict[str, Any]:
    """Scan every `_execute_*.log` under `mission_dir` for `iter_fitness`
    `[SCULPT-EVENT]` markers and write a `fitness.json` for any
    `stages/<stage>/runs/iter_N/` dir that doesn't already have one.

    Returns `{"written": n, "skipped_existing": n, "no_iter_dir": n,
    "stages": {stage: written_count}}`.
    """
    recorded: dict[tuple[str, int], dict[str, Any]] = {}
    current_stage: Optional[str] = None

    for log_path in _iter_log_files(mission_dir):
        for ev in _iter_sculpt_events(log_path):
            ev_type = ev.get("type")
            if ev_type == "stage_started":
                name = ev.get("stage_name")
                if isinstance(name, str) and name:
                    current_stage = name
                continue
            if ev_type != "iter_fitness":
                continue
            iter_idx = ev.get("iter")
            if current_stage is None or not isinstance(iter_idx, int):
                continue
            # Oldest→newest log iteration + in-order line reads within a
            # log means this assignment already gives last-write-wins.
            recorded[(current_stage, iter_idx)] = ev

    written = 0
    skipped_existing = 0
    no_iter_dir = 0
    stages: dict[str, int] = {}

    for (stage, iter_idx), ev in recorded.items():
        iter_dir = mission_dir / "stages" / stage / "runs" / f"iter_{iter_idx}"
        if not iter_dir.is_dir():
            no_iter_dir += 1
            continue
        fitness_path = iter_dir / "fitness.json"
        if fitness_path.is_file():
            skipped_existing += 1
            continue

        raw_fitness = ev.get("fitness")
        raw_progress = ev.get("progress")
        fitness = float(raw_fitness) if isinstance(raw_fitness, (int, float)) else None
        progress = float(raw_progress) if isinstance(raw_progress, (int, float)) else None

        steer = _recompute_steer(iter_dir, fitness, progress)

        raw_eval_seeds = ev.get("eval_seeds")
        raw_fps = ev.get("fitness_per_seed")
        raw_components = ev.get("components")

        record = {
            "schema": 1,
            "iter": iter_idx,
            "fitness": fitness,
            "progress": progress,
            "steer_fitness": steer["steer_fitness"],
            "steer_progress": steer["steer_progress"],
            "naturalness_factor": steer["naturalness_factor"],
            "naturalness_flag": steer["naturalness_flag"],
            "naturalness_hard_reject": steer["naturalness_hard_reject"],
            "observe_only": bool(ev.get("observe_only", False)),
            "eval_seeds": raw_eval_seeds if isinstance(raw_eval_seeds, int) else None,
            "fitness_per_seed": raw_fps if isinstance(raw_fps, list) else None,
            "components": raw_components if isinstance(raw_components, dict) else None,
            "source": "log_backfill",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        fitness_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        written += 1
        stages[stage] = stages.get(stage, 0) + 1

    return {
        "written": written,
        "skipped_existing": skipped_existing,
        "no_iter_dir": no_iter_dir,
        "stages": stages,
    }
