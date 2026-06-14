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
from sculptor.eval.generated_metric import (
    compute_generated_metric,
    make_generated_fitness_fn,
    resolve_fitness_fn,
)
from sculptor.eval.metric_calibration import calibrate_metric
from sculptor.eval.metric_gen import generate_objective_metric
from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.eval.spec_metrics import (
    compute_spec_metrics,
    make_spec_fitness_fn,
    spec_metric_names,
)
from sculptor.eval.stats import iqm, stratified_bootstrap_ci

__all__ = [
    "BENCHMARKS",
    "BenchmarkTask",
    "get_benchmark",
    "compute_spec_metrics",
    "make_spec_fitness_fn",
    "spec_metric_names",
    "resolve_fitness_fn",
    "compute_generated_metric",
    "make_generated_fitness_fn",
    "validate_generated_metric",
    "generate_objective_metric",
    "calibrate_metric",
    "CONDITIONS",
    "CampaignConfig",
    "EvalCondition",
    "aggregate",
    "run_campaign",
    "iqm",
    "stratified_bootstrap_ci",
]
