"""Auto-generated, per-task objective metrics (§Ship 35).

A generated metric is a small Python module that MIRRORS the hand-authored
spec contract in `spec_metrics.py`:

    def compute_spec(arrays, behavior, meta) -> dict[str, float]
        # returns {"spec_score": float in [0,1], ...sub-components}

computed PURELY from the persisted physical rollout arrays (joint_pos,
joint_vel, projected_gravity_b, root_link_pos_w + behavior.json +
joint_names) — NEVER from LLM judgment. It is generated from the NL goal,
then put through a validation chain (safety/contract/determinism/bounds +
a monotonicity audit) and an independent LLM review, and must EARN
steer-rights via calibration (Spearman vs a hand-authored ground-truth
metric) before it is allowed to drive selection. Until then it runs
OBSERVE-ONLY (computed + displayed, no influence). This module is the
RUNTIME side (load + compute + resolve); generation lives in `metric_gen`,
validation in `metric_validate`, calibration in `metric_calibration`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from sculptor.eval.spec_metrics import _CAPTURE_KEYS, _SPEC_FNS, make_spec_fitness_fn

#: The function a generated metric module must define.
GENERATED_FN_NAME = "compute_spec"

#: Physical rollout arrays a generated metric MAY read (the full contract —
#: the validator enforces a metric references only these). Mirrors the
#: spec_metrics.py array contract; kept here as the single allow-list a
#: generated metric is constrained to.
ALLOWED_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "projected_gravity_b",
    "root_link_pos_w",
)


def load_generated_metric(module_path: Path | str) -> Callable[..., dict]:
    """Import a generated-metric module and return its `compute_spec`. Uses
    a unique module name so re-loading an edited metric never hits a stale
    sys.modules entry."""
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(
        f"_genmetric_{module_path.stem}_{abs(hash(str(module_path)))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generated metric at {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, GENERATED_FN_NAME, None)
    if not callable(fn):
        raise ValueError(
            f"generated metric {module_path} lacks a callable "
            f"{GENERATED_FN_NAME}()"
        )
    return fn


def compute_generated_metric(
    module_path: Path | str,
    rollout_dir: Path | str,
    *,
    behavior: Optional[dict] = None,
) -> dict[str, Any]:
    """Run a generated metric on a rollout dir. Mirrors
    `compute_spec_metrics`' defensive loading: NEVER raises — a bad/missing
    artifact or a crashing metric yields `{"spec_score": 0.0, "error": ...}`
    so the loop aggregates an honest zero instead of dying."""
    rollout_dir = Path(rollout_dir)
    try:
        if behavior is None:
            bpath = rollout_dir / "behavior.json"
            behavior = (
                json.loads(bpath.read_text(encoding="utf-8"))
                if bpath.is_file() else {}
            )
        meta: dict[str, Any] = {}
        limits_path = rollout_dir / "mjcf_limits.json"
        if limits_path.is_file():
            try:
                limits = json.loads(limits_path.read_text(encoding="utf-8"))
                names = limits.get("joint_names") or []
                if names:
                    meta["joint_names"] = [str(n) for n in names]
            except Exception:  # noqa: BLE001 — names are an upgrade, not a dep
                pass
        arrays: dict[str, np.ndarray] = {}
        npz_path = rollout_dir / "trajectory.npz"
        if npz_path.is_file():
            with np.load(npz_path) as z:
                # Load every ALLOWED array that's present — the generated
                # metric may use any subset; missing ones simply aren't
                # there (the validator forbids referencing absent arrays).
                for k in ALLOWED_ARRAYS:
                    if k in z.files:
                        arrays[k] = z[k]
        fn = load_generated_metric(module_path)
        out = fn(arrays, behavior, meta)
        if not isinstance(out, dict) or "spec_score" not in out:
            return {"spec_score": 0.0,
                    "error": "metric did not return a dict with spec_score"}
        score = float(out.get("spec_score", 0.0) or 0.0)
        if not np.isfinite(score):
            return {"spec_score": 0.0, "error": "spec_score not finite"}
        out["spec_score"] = float(np.clip(score, 0.0, 1.0))
        capture = {k: behavior.get(k) for k in _CAPTURE_KEYS if k in behavior}
        return {**out, "capture": capture}
    except Exception as e:  # noqa: BLE001 — zero, observably
        return {"spec_score": 0.0, "error": f"{type(e).__name__}: {e}"}


def make_generated_fitness_fn(module_path: Path | str) -> Callable[[Any], float]:
    """`fitness_fn(iter_dir) -> float` for a generated metric module —
    scores `iter_dir/rollout` (0.0 on any failure)."""
    module_path = Path(module_path)

    def _fitness(iter_dir: Any) -> float:
        result = compute_generated_metric(module_path, Path(iter_dir) / "rollout")
        return float(result.get("spec_score", 0.0) or 0.0)

    def _detail(iter_dir: Any) -> dict:
        # §Ship 36 (F2): full component breakdown for the diagnoser. Rides
        # on the fitness fn (no new threaded param). Never raises.
        try:
            return compute_generated_metric(module_path, Path(iter_dir) / "rollout")
        except Exception:  # noqa: BLE001 — breakdown is advisory, never fatal
            return {}

    _fitness.detail = _detail  # type: ignore[attr-defined]
    return _fitness


def resolve_fitness_fn(spec: str) -> Callable[[Any], float]:
    """Resolve a fitness spec to a `fitness_fn(iter_dir) -> float`. `spec`
    is either a built-in spec-metric name (e.g. "go1_trot") or a filesystem
    path to a generated-metric .py module. Raises on anything else (fail
    fast before GPU work)."""
    if spec in _SPEC_FNS:
        return make_spec_fitness_fn(spec)
    p = Path(spec)
    if p.suffix == ".py" and p.is_file():
        return make_generated_fitness_fn(p)
    raise KeyError(
        f"unknown fitness metric {spec!r}: not a built-in "
        f"{sorted(_SPEC_FNS)} and not a generated-metric .py path that exists"
    )
