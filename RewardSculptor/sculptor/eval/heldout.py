"""Held-out evaluation battery — final numbers the loop never optimized.

§RESEARCH_GAP_ANALYSIS §7.4 / Eurekaverse hygiene: in-loop selection may
use whatever proxy it wants, but REPORTED numbers must come from an
evaluation surface the optimizer never saw. This battery evaluates a
kept checkpoint on:

  * a PUSH-PERTURBATION grid over the frozen shared env — `push_events`
    is the one physics perturbation the ROLLOUT path honors by design
    (train-scope levers like RSI/friction-DR never touch evaluation, so
    they are not usable here — that is the point of the scope split);
  * FRESH rollout seeds never used for in-loop selection (Patterson
    max-statistic discipline: pass seeds disjoint from `eval_seeds` /
    `fresh_eval_seeds`);
  * HAND spec metrics only (`spec_metrics`), never the generated metric
    the loop steered on.

Each grid cell writes standard rollout artifacts into its own dir and
is scored with `compute_spec_metrics`; the report carries per-cell
scores, per-level aggregates (median across seeds — IQM needs >=5
seeds; the report states which was used), and the degradation curve
relative to the unperturbed level. GPU cost: one rollout per (level,
seed) cell — minutes per cell, run once per kept policy, never in-loop.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from sculptor.env_spec import validate_env_spec

#: Default push magnitudes (m/s, linear). 0.0 = the unperturbed base
#: level every degradation number is relative to. Envelope bound is
#: _PUSH_LINEAR_MAX (2.0) in env_spec.py — validated per cell anyway.
DEFAULT_PUSH_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
#: Fresh-seed defaults — disjoint by construction from the loop's
#: deterministic eval seeds (10_000+ band) and fresh re-eval band
#: (90_001+): a dedicated 70_000 band.
DEFAULT_HELDOUT_SEEDS: tuple[int, ...] = (70_001, 70_002, 70_003)


def _perturbed_spec(base_spec: Optional[dict], push_mps: float) -> dict:
    """A validated env spec whose SHARED scope carries the push level.
    Builds on the project's current spec (shared scope preserved) so the
    battery perturbs the env the policy actually trained under."""
    spec = json.loads(json.dumps(base_spec)) if base_spec else {
        "env_spec_version": 1, "meta": {}, "shared": {}, "train": {},
    }
    shared = dict(spec.get("shared") or {})
    if push_mps > 0:
        shared["push_events"] = {
            "enabled": True,
            # interval is a [lo, hi] pair (validator contract) — a fixed
            # 2 s cadence keeps cells comparable across levels.
            "interval_s": [2.0, 2.0],
            "linear_mps": float(push_mps),
        }
    else:
        shared.pop("push_events", None)
    spec["shared"] = shared
    meta = dict(spec.get("meta") or {})
    meta["source"] = f"heldout:push={push_mps:g}"
    meta["rationale"] = (
        "held-out battery perturbation cell — NEVER a project spec "
        "version; exists only for this evaluation rollout.")
    spec["meta"] = meta
    errors = validate_env_spec(spec)
    if errors:
        raise ValueError(
            f"held-out perturbation spec invalid at push={push_mps:g}: "
            + "; ".join(errors))
    return spec


def run_heldout_battery(
    *,
    adapter: Any,
    checkpoint_path: Path | str,
    out_dir: Path | str,
    metric: str,
    base_env_spec: Optional[dict] = None,
    push_levels: Sequence[float] = DEFAULT_PUSH_LEVELS,
    seeds: Sequence[int] = DEFAULT_HELDOUT_SEEDS,
    n_episodes: int = 3,
    on_event: Any = None,
) -> dict[str, Any]:
    """Evaluate `checkpoint_path` across the (push_level × seed) grid.

    `adapter` must expose `.rollout(checkpoint_path, output_dir,
    n_episodes, *, seed=…)` and (for perturbations to take effect) an
    `env_spec_path` attribute the rollout threads to the runner — the
    mjlab adapter's contract. Adapters without `env_spec_path` still
    run, but only the unperturbed level is meaningful; the report marks
    perturbed cells `env_spec_applied=False` so a missing lever can
    never masquerade as robustness.

    `metric` is a HAND spec-metric name (resolve via spec_metrics) —
    passing a generated metric defeats the battery's purpose and is
    rejected.

    Returns (and writes to `<out_dir>/heldout_report.json`) the report.
    Individual cell failures score 0.0 with the error recorded — a
    crashed rollout is an honest zero, not a crashed battery.
    """
    from sculptor.eval.spec_metrics import compute_spec_metrics, spec_metric_names

    if metric not in spec_metric_names():
        raise KeyError(
            f"held-out battery scores HAND specs only; {metric!r} is not "
            f"one of {sorted(spec_metric_names())}")
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _emit(ev: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001 — progress is advisory
                pass

    has_spec_lever = hasattr(adapter, "env_spec_path")
    cells: list[dict[str, Any]] = []
    for level in push_levels:
        spec = _perturbed_spec(base_env_spec, float(level))
        spec_path = out_dir / f"env_push_{level:g}.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
        if has_spec_lever:
            adapter.env_spec_path = str(spec_path)
        for seed in seeds:
            cell_dir = out_dir / f"push_{level:g}_seed_{seed}"
            rollout_dir = cell_dir / "rollout"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            _emit({"type": "heldout_cell_started",
                   "push_mps": float(level), "seed": int(seed)})
            cell: dict[str, Any] = {
                "push_mps": float(level), "seed": int(seed),
                "env_spec_applied": bool(has_spec_lever or level == 0.0),
                "score": 0.0,
            }
            t0 = time.time()
            try:
                adapter.rollout(
                    checkpoint_path=checkpoint_path,
                    output_dir=rollout_dir,
                    n_episodes=int(n_episodes),
                    seed=int(seed),
                )
                result = compute_spec_metrics(metric, rollout_dir)
                cell["score"] = float(result.get("spec_score", 0.0) or 0.0)
                cell["components"] = {
                    k: v for k, v in result.items()
                    if isinstance(v, (int, float))}
            except Exception as e:  # noqa: BLE001 — honest zero per cell
                cell["error"] = f"{type(e).__name__}: {e}"
                print(f"[heldout] cell push={level:g} seed={seed} failed: "
                      f"{cell['error']}", file=sys.stderr, flush=True)
            cell["wall_s"] = round(time.time() - t0, 1)
            cells.append(cell)
            _emit({"type": "heldout_cell_scored", **{
                k: cell[k] for k in ("push_mps", "seed", "score")}})

    # Per-level aggregate: median across seeds (honest for K=3; IQM
    # needs >=5 — the report names the aggregator so nobody misreads).
    levels_out: list[dict[str, Any]] = []
    base_score: Optional[float] = None
    for level in push_levels:
        scores = [c["score"] for c in cells
                  if c["push_mps"] == float(level) and "error" not in c]
        agg = float(statistics.median(scores)) if scores else 0.0
        if float(level) == 0.0:
            base_score = agg
        levels_out.append({
            "push_mps": float(level),
            "n_scored": len(scores),
            "median_score": agg,
        })
    for row in levels_out:
        row["degradation_vs_base"] = (
            round(base_score - row["median_score"], 6)
            if base_score is not None else None)

    report = {
        "metric": metric,
        "aggregator": "median",
        "checkpoint": str(checkpoint_path),
        "n_episodes": int(n_episodes),
        "seeds": [int(s) for s in seeds],
        "push_levels": [float(p) for p in push_levels],
        "env_spec_lever_available": bool(has_spec_lever),
        "levels": levels_out,
        "cells": cells,
    }
    (out_dir / "heldout_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
