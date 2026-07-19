"""sculptor.eval — the research-grade evaluation layer (§Phase 3).

E1 (Ship 26): benchmark task definitions + hand-authored spec metrics
that are OBJECTIVE ground truth, computed from rollout artifacts and
fully independent of the LLM-authored success criteria (which cannot
be allowed to grade themselves).
"""

from sculptor.eval.benchmarks import (
    BENCHMARKS,
    BenchmarkManifestError,
    BenchmarkTask,
    benchmark_registry,
    get_benchmark,
    load_benchmark_manifest,
)
from sculptor.eval.charter import (
    CHARTER_FILENAME,
    CharterError,
    CharterIntegrityError,
    CharterMismatchError,
    UncharteredCampaignError,
    load_and_verify_charter,
)
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
from sculptor.eval.gauntlet import (
    GauntletError,
    analyze_blind_study,
    build_blind_study,
    load_and_verify_study_key,
)
from sculptor.eval.metric_calibration import (
    adversarial_archetype_gate,
    adversarial_archetype_gate_spec,
    calibrate_metric,
    calibrate_metric_against_reference,
    calibrate_task_derived,
    compute_trust,
    grant_decision,
    kick_required_losers,
)
from sculptor.eval.metric_gen import generate_objective_metric
from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.eval.spec_metrics import (
    compute_spec_metrics,
    make_spec_fitness_fn,
    spec_metric_names,
)
from sculptor.eval.spec_audit import (
    ATTACK_CLASSES,
    AUTHORITY_COVERAGE,
    SpecAuditError,
    load_and_verify_spec_certificate,
    load_spec_audit_manifest,
    run_spec_audit,
)
from sculptor.eval.stats import iqm, stratified_bootstrap_ci

__all__ = [
    "BENCHMARKS",
    "BenchmarkTask",
    "BenchmarkManifestError",
    "benchmark_registry",
    "load_benchmark_manifest",
    "get_benchmark",
    "CHARTER_FILENAME",
    "CharterError",
    "CharterIntegrityError",
    "CharterMismatchError",
    "UncharteredCampaignError",
    "load_and_verify_charter",
    "compute_spec_metrics",
    "make_spec_fitness_fn",
    "spec_metric_names",
    "ATTACK_CLASSES",
    "AUTHORITY_COVERAGE",
    "SpecAuditError",
    "load_spec_audit_manifest",
    "load_and_verify_spec_certificate",
    "run_spec_audit",
    "resolve_fitness_fn",
    "compute_generated_metric",
    "make_generated_fitness_fn",
    "GauntletError",
    "build_blind_study",
    "analyze_blind_study",
    "load_and_verify_study_key",
    "validate_generated_metric",
    "generate_objective_metric",
    "adversarial_archetype_gate",
    "adversarial_archetype_gate_spec",
    "calibrate_metric",
    "calibrate_metric_against_reference",
    "calibrate_task_derived",
    "compute_trust",
    "grant_decision",
    "kick_required_losers",
    "CONDITIONS",
    "CampaignConfig",
    "EvalCondition",
    "aggregate",
    "run_campaign",
    "iqm",
    "stratified_bootstrap_ci",
]
