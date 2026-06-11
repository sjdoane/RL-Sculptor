"""sculptor.eval — the research-grade evaluation layer (§Phase 3).

E1 (Ship 26): benchmark task definitions + hand-authored spec metrics
that are OBJECTIVE ground truth, computed from rollout artifacts and
fully independent of the LLM-authored success criteria (which cannot
be allowed to grade themselves).
"""

from sculptor.eval.benchmarks import BENCHMARKS, BenchmarkTask, get_benchmark
from sculptor.eval.harness import (
    CONDITIONS,
    CampaignConfig,
    EvalCondition,
    aggregate,
    run_campaign,
)
from sculptor.eval.spec_metrics import compute_spec_metrics
from sculptor.eval.stats import iqm, stratified_bootstrap_ci

__all__ = [
    "BENCHMARKS",
    "BenchmarkTask",
    "get_benchmark",
    "compute_spec_metrics",
    "CONDITIONS",
    "CampaignConfig",
    "EvalCondition",
    "aggregate",
    "run_campaign",
    "iqm",
    "stratified_bootstrap_ci",
]
