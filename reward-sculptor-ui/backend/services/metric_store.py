"""§Ship 35: per-project storage for auto-generated objective metrics.

Metrics live at `<project>/metrics/<id>/{metric.py, meta.json}`. This
module allocates ids, drives generation/calibration via sculptor_bridge
(the only sculptor entry point), and shapes the summaries the UI shows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from backend.services import sculptor_bridge


def _metrics_root(project_dir: Path) -> Path:
    return Path(project_dir) / "metrics"


# ── §Ship 40: live generation-progress sidecar (atomic; polled by the UI) ──
def _progress_path(project_dir: Path) -> Path:
    return _metrics_root(project_dir) / ".gen_progress.json"


def write_progress(project_dir: Path, data: dict[str, Any]) -> None:
    """Atomically write the generation-progress sidecar (tmp + rename) so a
    poll never reads a half-written file. Best-effort — never raises."""
    try:
        p = _progress_path(project_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(p)
    except Exception:  # noqa: BLE001 — progress is advisory
        pass


def read_progress(project_dir: Path) -> dict[str, Any]:
    """Read the generation-progress sidecar; absent/unreadable → inactive."""
    try:
        d = json.loads(_progress_path(project_dir).read_text(encoding="utf-8"))
        if isinstance(d, dict):
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"active": False}


def clear_progress(project_dir: Path) -> None:
    write_progress(project_dir, {"active": False})


def _next_id(root: Path) -> str:
    """Allocate the next gen_NNN id, claiming its directory ATOMICALLY
    (mkdir exist_ok=False) so two concurrent generates can't collide on
    the same id (Ship 35 review)."""
    root.mkdir(parents=True, exist_ok=True)
    n = 1 + max(
        (int(p.name[4:]) for p in root.iterdir()
         if p.is_dir() and p.name.startswith("gen_") and p.name[4:].isdigit()),
        default=0,
    )
    while True:
        gid = f"gen_{n:03d}"
        try:
            (root / gid).mkdir(exist_ok=False)
            return gid
        except FileExistsError:
            n += 1


def _summary(gid: str, rec: dict) -> dict[str, Any]:
    validation = rec.get("validation") or {}
    return {
        "id": gid,
        "behavior_goal": rec.get("behavior_goal"),
        "accepted": bool(rec.get("accepted")),
        "validation_passed": bool(rec.get("validation_passed")),
        "calibrated": bool(rec.get("calibrated", False)),
        "review": rec.get("review"),
        "gates": validation.get("gates"),
        "reasons": validation.get("reasons"),
        "archetype_scores": validation.get("archetype_scores"),
        # §Ship 50: L1 axiom per-layer breakdown (for the UI evidence line +
        # the Ship-52 trust score). None for pre-Ship-50 records.
        "axioms": validation.get("axioms"),
        "calibration": rec.get("calibration"),
        # §Ship 51: "builtin" | "task_derived" so the UI can show the right card.
        "calibration_method": rec.get("calibration_method"),
        "source": rec.get("source"),
        "recorded_at": rec.get("recorded_at"),
    }


def generate(
    project_dir: Path, behavior_goal: str, *,
    robot_hint: Optional[str] = None, review: bool = True,
    on_event=None,
) -> dict[str, Any]:
    """Generate + validate + review a metric; persist under a fresh id.
    §Ship 40: `on_event` streams pipeline progress to the caller."""
    root = _metrics_root(project_dir)
    gid = _next_id(root)
    rec = sculptor_bridge.generate_objective_metric(
        behavior_goal, root / gid, robot_hint=robot_hint, review=review,
        on_event=on_event)
    rec["id"] = gid
    # Re-stamp meta.json with the id so list/calibrate can find it.
    meta = root / gid / "meta.json"
    try:
        meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — meta is best-effort; rec already returned
        pass
    return _summary(gid, rec)


def list_metrics(project_dir: Path) -> list[dict[str, Any]]:
    root = _metrics_root(project_dir)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(root.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.is_file():
            try:
                out.append(_summary(d.name, json.loads(meta.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001 — skip a corrupt record
                continue
    return out


def calibrate(project_dir: Path, gid: str, builtin_name: str) -> dict[str, Any]:
    """Calibrate a stored metric against a built-in ground truth and
    persist `calibrated` + the calibration record into its meta.json."""
    d = _metrics_root(project_dir) / gid
    metric_py = d / "metric.py"
    meta = d / "meta.json"
    if not metric_py.is_file() or not meta.is_file():
        raise FileNotFoundError(f"generated metric {gid!r} not found")
    cal = sculptor_bridge.calibrate_objective_metric(metric_py, builtin_name)
    rec = json.loads(meta.read_text(encoding="utf-8"))
    rec["calibrated"] = bool(cal.get("ok"))
    rec["calibration"] = cal
    rec["calibration_method"] = "builtin"
    meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return _summary(gid, rec)


def calibrate_task_derived(
    project_dir: Path, gid: str, behavior_goal: str,
    robot_hint: Optional[str] = None, *, client=None,
) -> dict[str, Any]:
    """§Ship 51: calibrate a stored metric against K independently-authored
    competence ladders (the novel-task path) and persist `calibrated` + the
    full per-source provenance into its meta.json. The grant flips at the SAME
    point as the built-in path; `steer_allowed` is untouched."""
    d = _metrics_root(project_dir) / gid
    metric_py = d / "metric.py"
    meta = d / "meta.json"
    if not metric_py.is_file() or not meta.is_file():
        raise FileNotFoundError(f"generated metric {gid!r} not found")
    cal = sculptor_bridge.calibrate_task_derived_metric(
        metric_py, behavior_goal, robot_hint, client=client)
    rec = json.loads(meta.read_text(encoding="utf-8"))
    rec["calibrated"] = bool(cal.get("ok"))
    rec["calibration"] = cal
    rec["calibration_method"] = "task_derived"
    meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return _summary(gid, rec)
