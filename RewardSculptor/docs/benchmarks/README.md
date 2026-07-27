# External benchmark manifests

Use `--benchmark-manifest` to add versioned, capability-described tasks without
editing the built-in Python table:

```bash
sculpt eval list \
  --benchmark-manifest docs/benchmarks/cross_embodiment_frontier_v1.json
```

Manifests are strict JSON (`schema_version: 1`). Unknown fields, duplicate
names/capabilities, built-in overrides, placeholder spec names, and invalid
readiness claims are rejected. Every entry declares:

- task and natural-language behavior goal;
- adapter plus adapter config;
- robot ID, embodiment family, and task family;
- required capability labels;
- evaluation tier (`compile_only`, `rollout_artifact`, or
  `heldout_solution`);
- campaign readiness, known limitations, and evaluator authority.

`compile_only` is a useful and honest state: the task and adapter may exist,
but the rollout contract or objective evaluator is not sufficient for a
campaign. Such entries must use `spec_metric: null`,
`spec_authority: A0_rejected`, and `spec_audit_certificate: null`. The harness
lists them and explains the blockers but refuses GPU work.

A campaign-ready external entry must name a registered objective spec and cite
a tamper-evident certificate produced by
[`sculpt eval spec-audit`](../spec_audit.md). The certificate must pass
`A4_reporting` for that exact spec. Its certificate hash and the external
manifest file hash are frozen into the campaign charter. Merely naming an
existing metric is not enough.

The built-in four-task suite predates this certificate boundary. It remains
executable for compatibility, but reports now label its evaluator authority as
`legacy_provisional` and emit a visible warning. Do not use those provisional
results as a new headline claim until each spec has a held-out A4 audit.

The included `cross_embodiment_frontier_v1.json` adds the real registered YAM
lift and multi-cube manipulation tasks as compile-only arm/gripper frontiers.
Schema-2 generic rollouts and the `object_lift_hold` implementation now cover
their object, target, independent-contact, grasp, temporal-hold, and distractor
semantics. Their remaining blocker is evidence, not mechanism: no task-scoped
real-rollout A4 certificate or frozen evaluation split exists. This prevents
the suite from claiming cross-embodiment coverage merely because code and UI
support an arm.
