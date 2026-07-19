# Adversarial success-spec audits

A function in `spec_metrics.py` is an implementation, not evidence that the
function measures task success. `sculpt eval spec-audit` runs a frozen battery
of competent and adversarial rollout artifacts and emits a tamper-evident
authority certificate.

This protocol is task- and embodiment-agnostic. The attack classes describe
ways an evaluator can be wrong, not assumptions about legs, arms, or grippers.
The rollout artifacts themselves carry the task-specific observables.

## Authority coverage

| Requested level | Minimum evidence classes |
|---|---|
| `A1_descriptive` | competent positive |
| `A2_advisory` | competent, stillness, falling, explosion |
| `A3_steering` | A2 plus oscillation, early termination, proxy-only behavior |
| `A4_reporting` | all above plus threshold flicker, reset artifact, and time truncation |

Every recorded case must also meet its predeclared score bound. Missing
coverage or one failed case produces `A0_rejected`; passing a smaller battery
never silently grants a larger authority level.

## Manifest

```json
{
  "schema_version": 1,
  "audit_id": "yam-lift-spec-v1",
  "spec_name": "yam_lift_cube",
  "authority_target": "A4_reporting",
  "notes": "Held-out objects, worlds, and seeds; none used during metric authoring.",
  "cases": [
    {
      "case_id": "competent_seed_401",
      "attack_class": "competent_positive",
      "rollout_dir": "evidence/competent_seed_401/rollout",
      "expectation": {"min_score": 0.8},
      "notes": "Cube remains above the lift threshold for the full hold window."
    },
    {
      "case_id": "proxy_gripper_motion_017",
      "attack_class": "proxy_only",
      "rollout_dir": "evidence/proxy_gripper_motion_017/rollout",
      "expectation": {"max_score": 0.1},
      "notes": "Fast gripper motion without cube contact or displacement."
    }
  ]
}
```

Use one or more independent artifacts for every required class. Paths are
resolved relative to the manifest. The runner hashes the exact evaluator
inputs (`behavior.json`, `trajectory.npz`, and `mjcf_limits.json` when
present), so replacing evidence after an audit changes the certificate.
Thresholds are normalized to `[0, 1]` and frozen before results are inspected.

Run the audit into a fresh directory:

```bash
sculpt eval spec-audit audits/yam_lift_v1.json \
  --out audits/results/yam_lift_v1
```

Outputs:

- `spec_audit_certificate.json`: source/evidence hashes, coverage, observed
  score details, failures, decision, and certificate hash;
- `spec_audit_report.md`: readable coverage and per-case result table.

The command exits nonzero when the requested authority is rejected. It also
refuses a non-empty output directory, so a later run cannot overwrite an
earlier certificate under the same path.

## Evidence design rules

- Use unseen evaluation seeds, worlds, objects, and robot variants. Metric
  generation, calibration, reward selection, and early stopping must never see
  the audit artifacts.
- Make attacks task-specific. For manipulation, “proxy-only” should include
  end-effector or gripper motion without the required object outcome. For
  locomotion, it may include motion magnitude without commanded traversal.
- Early termination, time truncation, and reset artifacts need separate cases;
  they fail through different causal mechanisms.
- Include threshold-adjacent positives as well as threshold flicker attacks.
  A metric that rejects every near-success can be as unusable as one that
  accepts every exploit.
- Treat a certificate as scoped evidence, not proof of universal safety. A new
  task, robot, world lineage, sensor contract, metric revision, or discovered
  counterexample requires a new versioned audit.

The current YAM lift frontiers in
`docs/benchmarks/cross_embodiment_frontier_v1.json` remain compile-only because
the rollout contract does not yet persist object pose, end-effector pose,
gripper contact, and grasp state. Their path to campaign readiness is: add
those observables, implement the temporal lift-and-hold spec, assemble held-out
evidence, pass this audit, then change the benchmark manifest in a new version.
