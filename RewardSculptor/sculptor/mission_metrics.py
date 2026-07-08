"""Per-stage objective-metric generation for missions.

§MISSION_METRIC_GRANULARITY (decision record, 2026-07-06): each mission
stage steers on a FRESH, trust-gated objective metric generated from the
stage's own goal text at decomposition time. The mission-level
`fitness_metric` stays as decomposition context and as the fallback for
stages whose generated metric the pipeline rejects — the existing
`steering_metric or fitness_metric` resolution is unchanged.

Mechanically this populates the Ship-38 `Stage.steering_metric` slot
with a *mission-dir-relative* path
("stage_metrics/<name>/metric.py"). Relative refs keep `mission.json`
portable and inside the 128-char validator bound;
`resolve_stage_metric_ref` anchors them at the mission dir before
`resolve_fitness_fn`'s fail-fast resolution.

Metrics live under `<mission_dir>/stage_metrics/`, NOT inside
`<mission_dir>/stages/<name>/` — the orchestrator scaffolds each
`stages/<name>/` dir with `sculpt_init`, which (correctly) refuses a
non-empty target. Generation happens at decompose time, long before
scaffolding, so anything written inside a stage dir would brick the
stage (live-caught: g1-standing-jump halted `scaffold_errored` on
exactly this).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

from sculptor.mission import Mission


def resolve_stage_metric_ref(ref: str, mission_dir: Path | str) -> str:
    """Anchor a mission-dir-relative generated-metric ref at the mission
    dir. Spec-metric names ("g1_kick") and absolute paths pass through
    untouched — only a RELATIVE `*.py` ref is joined."""
    p = Path(ref)
    if p.suffix == ".py" and not p.is_absolute():
        return str(Path(mission_dir) / p)
    return ref


def generate_stage_metrics(
    mission: Mission,
    *,
    robot_hint: Optional[str] = None,
    client: Any = None,
    n_candidates: int = 1,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    only_stages: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Generate one objective metric per pending stage that has no
    `steering_metric` yet. Full trust pipeline per stage (L0 gates +
    review); an accepted metric sets `steering_metric` to the
    mission-dir-relative path, a rejected one leaves the stage on the
    mission-level fallback. Mutates `mission` in place; the caller
    re-saves. Never raises for a single stage's failure — the mission
    must stay runnable on fallbacks.

    `only_stages` (§mission-persistence increment 1): when given, only
    stages whose name is in the list are processed, and — for those
    named stages ONLY — the "skip if steering_metric already set" guard
    is BYPASSED, so a user can explicitly trigger regeneration of a
    stage that already has a metric. The "skip if already succeeded"
    guard still applies (never regenerate the metric a stage already
    won on). Stages not named in `only_stages` are left untouched (not
    even reported as "skipped" — they were never candidates).

    When `only_stages` is None (the default, whole-mission pass),
    "superseded" stages are also skipped — they are terminal and will
    never train again, so generating a metric for one wastes an LLM
    call. A user-targeted `only_stages` regenerate is still allowed to
    hit a superseded stage (e.g. to refresh what a report shows), which
    is why that check is IN the default branch, not a blanket guard.

    Returns `{"generated": [...], "rejected": [...], "skipped": [...]}`
    where each entry is `{stage, reason?}`.
    """
    from sculptor.eval import generate_objective_metric

    if mission.mission_dir is None:
        raise RuntimeError(
            "mission.mission_dir is None — save_mission before "
            "generate_stage_metrics so stage metric dirs resolve.")
    mission_dir = Path(mission.mission_dir)

    def _emit(ev: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001 — progress is advisory
                pass

    report: dict[str, list[dict[str, Any]]] = {
        "generated": [], "rejected": [], "skipped": [],
    }
    only_set = set(only_stages) if only_stages is not None else None
    for stage in mission.stages:
        if only_set is not None:
            if stage.name not in only_set:
                continue
        else:
            if getattr(stage, "steering_metric", None):
                report["skipped"].append(
                    {"stage": stage.name, "reason": "steering_metric already set"})
                continue
            if getattr(stage, "status", "pending") == "superseded":
                report["skipped"].append(
                    {"stage": stage.name, "reason": "stage superseded"})
                continue
        if getattr(stage, "status", "pending") == "succeeded":
            report["skipped"].append(
                {"stage": stage.name, "reason": "stage already succeeded"})
            continue
        out_dir = mission_dir / "stage_metrics" / stage.name
        _emit({
            "type": "stage_metric_gen_started",
            "stage": stage.name,
            "goal_text": stage.goal_text[:200],
        })
        try:
            rec = generate_objective_metric(
                stage.goal_text, out_dir,
                robot_hint=robot_hint, client=client,
                n_candidates=n_candidates,
                on_event=on_event,
            )
        except Exception as e:  # noqa: BLE001 — stage falls back, mission runs
            print(
                f"[mission-metrics] stage {stage.name!r}: generation "
                f"crashed ({type(e).__name__}: {e}) — stage falls back to "
                f"the mission-level metric.", file=sys.stderr, flush=True)
            report["rejected"].append(
                {"stage": stage.name,
                 "reason": f"{type(e).__name__}: {e}"})
            _emit({
                "type": "stage_metric_gen_failed",
                "stage": stage.name,
                "reason": f"{type(e).__name__}: {e}",
            })
            continue
        if rec.get("accepted"):
            rel = f"stage_metrics/{stage.name}/metric.py"
            stage.steering_metric = rel
            report["generated"].append({"stage": stage.name, "ref": rel})
            _emit({
                "type": "stage_metric_gen_accepted",
                "stage": stage.name,
                "ref": rel,
            })
        else:
            # generate_objective_metric never raises on a bad candidate —
            # rejection reasons live in the record it wrote to meta.json.
            reasons = rec.get("validation") or {}
            reason = "; ".join(
                (reasons.get("reasons") or ["review rejected"])[:3])
            report["rejected"].append(
                {"stage": stage.name, "reason": reason})
            _emit({
                "type": "stage_metric_gen_rejected",
                "stage": stage.name,
                "reason": reason,
            })
    return report
