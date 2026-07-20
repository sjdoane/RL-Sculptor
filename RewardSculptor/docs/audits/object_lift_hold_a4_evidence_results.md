# Object lift/hold — real-rollout audit results (yam_lift_cube v1)

Status: **A1_descriptive certificate PASSED on real simulator rollouts.**
YAM remains `compile_only` / not `campaign_ready`; A4 reporting authority is
**not** yet granted (see the coverage gap below). This file records the
outcome of the first real-evidence pass against
[`object_lift_hold_a4_evidence_plan.md`](object_lift_hold_a4_evidence_plan.md).

## Provenance

- RewardSculptor commit at freeze: `44bf70f` (the `object_lift_hold` evaluator
  and schema-2 telemetry commit).
- Freeze manifest `freeze.json` sha256:
  `8f1ede1103654e1a96d58f9d2c1c59b727e01621f64c0e605e68b10f5d06d073`
  (pins commit, uv.lock/pyproject hashes, env-config + robot-asset hashes for
  both YAM tasks, all evaluator thresholds, disjoint evaluation seeds per
  class, and predeclared score expectations — sealed BEFORE any outcome
  rollout was collected).
- Certificate `spec_audit_certificate.json` sha256:
  `80ae54ae69a0493a946de38a648cbf430a6773eaf3cbafefa5a7ec8b8675a7a7`
  (verified via `load_and_verify_spec_certificate`).
- Policy: full upstream config, `Mjlab-Lift-Cube-Yam`, 1024 envs, 5000
  iterations, seed 42 (reserved as a policy input, excluded from every
  evaluation split). Final task lift-success ~0.75; the frozen evaluator is
  stricter (8 cm clearance + 0.5 s stable hold + bilateral grasp + quiet
  dynamics), and the competent evaluation rollouts scored 0.98–1.00.
- Evidence root (this machine): `~/rs_evidence/object_lift_hold_v1/`. All
  rollouts came from the `sculptor.adapters._mjlab_runner rollout` path; no
  hand-written NPZ. Every case's video and telemetry were inspected before
  its label was frozen. Survey/tuning rollouts used disjoint scratch seeds
  (900+), never the frozen evaluation seeds.

## Result

`authority_target: A1_descriptive` → `authority_decision: A1_descriptive`,
`passed: true`, 8/8 cases passed.

| Attack class | Case (seed) | spec_score | Expectation | Pass |
|---|---|---|---|---|
| competent_positive | s101 | 1.000 | ≥ 0.8 | ✓ |
| competent_positive | s102 | 0.984 | ≥ 0.8 | ✓ |
| competent_positive | s103 | 1.000 | ≥ 0.8 | ✓ |
| stillness | s201 | 0.000 | ≤ 0.0 | ✓ |
| proxy_only | s209 | 0.000 | ≤ 0.0 | ✓ |
| explosion | s204 | 0.000 | ≤ 0.0 | ✓ |
| reset_artifact | s207 | 0.000 | ≤ 0.0 | ✓ |
| time_truncation | s208 | 0.000 | ≤ 0.0 | ✓ |

A1 requires only `competent_positive`; the five adversarial classes above are
recorded as passing **bonus** coverage (they belong to the A2/A3/A4 sets).

## Coverage gap to A2–A4 (honest)

Not collected, so A2+ is **not** granted:

- `falling`, `oscillation`, `threshold_flicker`, `early_termination`.

These four are *dynamic near-miss* modes: the object is meaningfully engaged
but the outcome fails in a time-dependent way. They could not be sourced
cleanly from this policy family:

- The lift policy transitions **sharply** from "cannot lift" (peak lift ≈ 0)
  to "lifts and holds" across ~100–200 training iterations; there is no wide
  band of checkpoints that lift the object substantially in most envs yet hold
  in none (a partially-competent checkpoint legitimately completes a fraction
  of envs, so its whole-rollout score is > 0 and violates `max_score: 0.0`).
- The gripper is **position-controlled**: once it pinches the box it does not
  spontaneously release, so a grasp does not fall. Heavy-cube perturbation
  produces "lifts partway and holds below the goal" (scores 0, but the
  behaviour is not `falling`); weak-grip produces "cannot lift" (proxy-like);
  neither is a clean lift-then-drop. `ee_ground_collision` never fires because
  the policy grasps from above without striking the ground.

**Next step for A2–A4:** train (or hand-script) a *deliberate near-miss*
policy — e.g. a reward-shaped agent that lifts and carries toward the goal but
drops or jitters before the hold — or add a time-varying grip-release
perturbation to the rollout path. Repeat the audit into a fresh output
directory; grant a higher authority only on a passing certificate. Any
evaluator, dependency, task, sensor, asset, or split change requires a new
audit version. YAM's benchmark manifest stays `compile_only` / `A0_rejected`
until a passing A4 certificate for both single- and multi-cube tasks exists.
