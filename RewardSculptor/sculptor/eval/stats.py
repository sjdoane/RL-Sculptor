"""Aggregation statistics for eval campaigns (§Ship 27 / E2).

rliable-style robust aggregates (Agarwal et al. 2021, "Deep RL at the
Edge of the Statistical Precipice"): IQM instead of mean (mean is
outlier-hostage at small n; median throws away too much), stratified
bootstrap over SEEDS for confidence intervals (the seed is the unit of
replication — resampling steps or envs would fake independence).

Pure numpy, deterministic via explicit rng seeds.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% of sorted values.
    n < 4 degrades to the plain mean (nothing to trim)."""
    x = np.sort(np.asarray(list(values), dtype=np.float64))
    n = x.size
    if n == 0:
        return float("nan")
    lo = int(np.floor(0.25 * n))
    hi = n - lo
    return float(x[lo:hi].mean())


def stratified_bootstrap_ci(
    per_seed_values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[Sequence[float]], float] = iqm,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI of `statistic` over seeds.

    Resamples SEEDS with replacement (stratified at the replication
    unit), recomputes the statistic per resample, and reports the
    [alpha/2, 1-alpha/2] percentiles. Deterministic given `rng_seed`.

    Honest-small-n note: with 3 seeds the CI is wide and that is the
    point — it reports the uncertainty instead of hiding it.
    """
    x = np.asarray(list(per_seed_values), dtype=np.float64)
    point = statistic(x)
    if x.size == 0:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    if x.size == 1:
        return {"point": point, "ci_low": float(x[0]),
                "ci_high": float(x[0]), "n": 1}
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = np.apply_along_axis(statistic, 1, x[idx])
    return {
        "point": float(point),
        "ci_low": float(np.quantile(stats, alpha / 2)),
        "ci_high": float(np.quantile(stats, 1 - alpha / 2)),
        "n": int(x.size),
    }
