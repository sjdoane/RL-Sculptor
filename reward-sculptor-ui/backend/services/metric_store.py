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
        "calibration": rec.get("calibration"),
        "source": rec.get("source"),
        "recorded_at": rec.get("recorded_at"),
    }


def generate(
    project_dir: Path, behavior_goal: str, *,
    robot_hint: Optional[str] = None, review: bool = True,
) -> dict[str, Any]:
    """Generate + validate + review a metric; persist under a fresh id."""
    root = _metrics_root(project_dir)
    gid = _next_id(root)
    rec = sculptor_bridge.generate_objective_metric(
        behavior_goal, root / gid, robot_hint=robot_hint, review=review)
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
    meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return _summary(gid, rec)
