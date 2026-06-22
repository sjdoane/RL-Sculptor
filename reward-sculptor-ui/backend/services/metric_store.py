"""§Ship 35: per-project storage for auto-generated objective metrics.

Metrics live at `<project>/metrics/<id>/{metric.py, meta.json}`. This
module allocates ids, drives generation/calibration via sculptor_bridge
(the only sculptor entry point), and shapes the summaries the UI shows.
"""
from __future__ import annotations

import json
import time
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
        # §best-of-N: how many candidates were sampled + which one won (with its
        # offline discrimination), so the UI can show "selected 2/3 (disc 1.83)".
        # None / 1 for the single-shot path (the byte-identical default).
        "n_candidates": rec.get("n_candidates"),
        "selected_candidate": rec.get("selected_candidate"),
        "candidates": rec.get("candidates"),
        # §Ship 51: "builtin" | "task_derived" so the UI can show the right card.
        "calibration_method": rec.get("calibration_method"),
        # §Ship 52: standardized trust score + per-layer breakdown.
        "trust": rec.get("trust"),
        "source": rec.get("source"),
        "recorded_at": rec.get("recorded_at"),
    }


def generate(
    project_dir: Path, behavior_goal: str, *,
    robot_hint: Optional[str] = None, review: bool = True,
    n_candidates: int = 1, on_event=None,
) -> dict[str, Any]:
    """Generate + validate + review a metric; persist under a fresh id.
    §Ship 40: `on_event` streams pipeline progress to the caller.
    §best-of-N: `n_candidates` >1 samples N candidates and keeps the most
    discriminating valid one (default 1 → single-shot-with-retry)."""
    root = _metrics_root(project_dir)
    gid = _next_id(root)
    rec = sculptor_bridge.generate_objective_metric(
        behavior_goal, root / gid, robot_hint=robot_hint, review=review,
        n_candidates=n_candidates, on_event=on_event)
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
    # §Ship 52: unified grant (byte-identical here) + standardized trust score.
    fin = sculptor_bridge.finalize_calibration(cal, rec.get("validation"))
    rec["calibrated"] = fin["calibrated"]
    rec["calibration"] = cal
    rec["calibration_method"] = "builtin"
    rec["trust"] = fin["trust"]
    meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return _summary(gid, rec)


def stamp_cal_token(project_dir: Path, gid: str) -> str:
    """§round-5: write a fresh calibration token into the metric's meta.json and return
    it. The LIVE launch path stamps a token BEFORE its timeout-bounded, un-cancellable
    (asyncio.to_thread) task-derived calibration and passes it as `expect_token`; on
    TIMEOUT it re-stamps to INVALIDATE the orphaned worker thread, so a calibration that
    genuinely succeeds AFTER the 300 s timeout cannot silently resurrect `calibrated=true`
    behind the 'observe-only' verdict already surfaced to the user (it becomes a no-op
    write — observe-only persisted state is preserved). Never raises (best-effort)."""
    meta = _metrics_root(project_dir) / gid / "meta.json"
    try:
        rec = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    except Exception:  # noqa: BLE001 — unreadable meta → start fresh token map
        rec = {}
    token = f"{gid}:{time.time_ns()}"
    rec["cal_token"] = token
    try:
        meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — best-effort; a missing token just disables the guard
        pass
    return token


def calibrate_task_derived(
    project_dir: Path, gid: str, behavior_goal: str,
    robot_hint: Optional[str] = None, *, client=None, adversarial: bool = False,
    expect_token: Optional[str] = None,
) -> dict[str, Any]:
    """§Ship 51: calibrate a stored metric against K independently-authored
    competence ladders (the novel-task path) and persist `calibrated` + the
    full per-source provenance into its meta.json. The grant flips at the SAME
    point as the built-in path; `steer_allowed` is untouched.
    §Ship 53: `adversarial` adds the L3 gaming-archetype gate (flag-gated).
    §round-5: `expect_token` (set by the live launch path via `stamp_cal_token`) makes
    the grant persist ONLY if this attempt is still current — a timed-out launch
    re-stamps the token, so an orphaned thread that finishes late SKIPS its write
    (the verdict the user saw stays authoritative). None → unconditional (the explicit,
    synchronous calibrate-route path, where there is no orphan)."""
    d = _metrics_root(project_dir) / gid
    metric_py = d / "metric.py"
    meta = d / "meta.json"
    if not metric_py.is_file() or not meta.is_file():
        raise FileNotFoundError(f"generated metric {gid!r} not found")
    cal = sculptor_bridge.calibrate_task_derived_metric(
        metric_py, behavior_goal, robot_hint, client=client, adversarial=adversarial)
    rec = json.loads(meta.read_text(encoding="utf-8"))
    if expect_token is not None and rec.get("cal_token") != expect_token:
        # Superseded (this run timed out and re-stamped the token) — do NOT resurrect a
        # grant behind the already-surfaced 'observe-only' verdict. Leave meta as-is.
        return _summary(gid, rec)
    # §Ship 52: unified grant + standardized trust score (same shape as builtin).
    fin = sculptor_bridge.finalize_calibration(cal, rec.get("validation"))
    rec["calibrated"] = fin["calibrated"]
    rec["calibration"] = cal
    rec["calibration_method"] = "task_derived"
    rec["trust"] = fin["trust"]
    meta.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return _summary(gid, rec)
