# Object lift/hold A4 evidence plan

Status: **A0_rejected — evidence not yet collected.** This file is a
pre-collection protocol, not a certificate. In particular,
`tests/test_object_lift_hold_spec.py` is an implementation-level adversarial
suite and must never be cited as real-rollout evaluator authority.

## Scope and frozen evaluator

The evaluator is `object_lift_hold` over manipulation telemetry schema 2. It is
capability-driven rather than robot-name-driven, but authority is still scoped
to a concrete task, robot asset/dependency lineage, command distribution,
sensor contract, and evaluation split. `yam_lift_cube` and
`yam_multi_cube_lift` therefore require separate certificates.

Before collecting any outcome, freeze:

- the RewardSculptor commit and dependency lock;
- task registration/config hashes and robot asset hash;
- evaluation seeds, object physical-property ranges, target distribution, and
  (for multi-object work) object count/layout split;
- manipulation sidecar schema and all evaluator thresholds;
- case IDs, attack-class labels, and score expectations in the audit manifest.

The frozen completion predicate is: a valid first-episode command segment,
initial target-goal separation, at least 8 cm vertical clearance, world-space
goal error no larger than the task threshold capped at 5 cm, at least 0.5 s of
stable hold, at least 80% two-group contact evidence with no contact dropout
longer than 0.1 s, and no non-target displacement over 5 cm. State jumps,
non-finite state, excessive velocity, contact/grasp disagreement, target-ID
flicker, reset boundaries, and insufficient duration fail closed.

## Required real evidence

Collect at least one independent rollout per class below; use multiple seeds
for the competent and threshold-adjacent classes. All rollouts must come from
the simulator artifact path, not hand-written NPZ files.

| Attack class | Required construction | Predeclared expectation |
|---|---|---|
| `competent_positive` | Trained policy visibly grasps the selected object, clears support, reaches the target, and holds. | `min_score: 0.8` across vector envs |
| `stillness` | No-op or frozen policy; object remains at spawn. | `max_score: 0.0` |
| `falling` | Object is briefly lifted then released/falls before the hold window. | `max_score: 0.0` |
| `oscillation` | Object repeatedly enters the target region but remains dynamically unstable. | `max_score: 0.0` |
| `explosion` | Deliberately unstable but finite physics that remains a valid evaluator input. | `max_score: 0.0` |
| `early_termination` | Illegal end-effector contact terminates before a complete hold. | `max_score: 0.0` |
| `threshold_flicker` | Object crosses the goal boundary repeatedly without a contiguous hold. | `max_score: 0.0` |
| `reset_artifact` | Auto-reset places an object near the next target; no cross-reset transport occurred. | `max_score: 0.0` |
| `time_truncation` | Capture ends after approach/lift but before 0.5 s hold evidence. | `max_score: 0.0` |
| `proxy_only` | Gripper/contact or end-effector motion occurs without the selected object outcome. | `max_score: 0.0` |

For the multi-object certificate, add targeted competent runs for every target
index and adversarial runs where the wrong cube reaches the target or a
non-target cube moves more than 5 cm. For future robots, repeat the audit with
their own capability/contact evidence; a YAM certificate does not transfer by
name or by code reuse.

## Acceptance and promotion

1. Inspect video plus telemetry for every case before freezing labels; anyone
   authoring the metric must not tune thresholds on these artifacts.
2. Place immutable rollout directories under a versioned evidence root and run
   `sculpt eval spec-audit` into a fresh output directory.
3. Require complete A4 coverage and every predeclared bound to pass. Preserve
   the manifest, certificate, report, and input hashes.
4. Only after that result, reference the verified certificate from a new
   benchmark-manifest version and change the corresponding task to
   `campaign_ready`. Any evaluator, dependency, task, sensor, asset, or split
   change requires a new audit version.
