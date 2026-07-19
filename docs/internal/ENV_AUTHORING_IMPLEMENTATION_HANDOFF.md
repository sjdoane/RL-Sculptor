# Environment-authoring implementation handoff

Read this file, then read `ENV_AUTHORING_ARCHITECTURE.md` completely. Continue
implementing the approved P1–P5 plan in `/home/samjd/projects` on branch
`ship-20-ux-revamp`. Do not restart the design or weaken the metric firewall,
evaluation freeze, capability-driven generality, clarification/default
provenance, or atomic keep/revert invariants.

## Current state

The research/audit/architecture foundation is committed:

- `f31f56d` and `7e56a8a`: KG audit and hardening;
- `adc7279`: structured 89-paper environment-authoring corpus;
- `307c41a`: normative architecture and mjlab schema PoC.

The live shared KG is clean (1,962 nodes, 2,107 edges, 1,335 embeddings).
Baseline verification: RewardSculptor 1,994 passed / 1 optional-JAX skip;
UI backend 531 passed with `MUJOCO_GL=egl`; the PoC passes against mjlab 1.3.0
and MuJoCo 3.7.0.

The first implementation slice is complete in commit `f217753`:

- `sculptor/world/capabilities.py`: immutable, extensible robot/simulator
  descriptors. Semantic body and site roles are data-resolved. Stock G1 does
  not advertise `grasp`; the installed Yam arm/gripper does. New robots use
  the same descriptor contract rather than task-name branches.
- `world_spec.py` / `task_spec.py`: strict nested validation, stable-ID
  variation pointers, capability/reference checks, and closed goal/contact
  vocabularies.
- `artifacts.py`: immutable canonical JSON artifacts and one atomic,
  hash-verified `selection_current.json` commit point for the complete
  reward/env/world/task/eval/catalog/clarification tuple.
- `channels.py`: initial typed channel-catalog foundation.
- `tests/test_world_foundation.py`: 11 adversarial contract tests, all passing
  in 1.28 seconds.

The next coherent implementation slice is complete in the working tree and
awaiting its commit hash:

- robot-agnostic prompt author + fully paginated clarification/default ledger;
- deterministic mjlab compiler, complete admission gates, exact materialized
  evaluation terrain replay, and immutable selection loading;
- generic task runtime bindings for contacts, height scans, observations,
  reset/goal/termination semantics;
- ChannelCatalog threaded through metric generation, validation, calibration,
  runtime, and the metric-only/shared-shaping firewall;
- GPU-native reward-visible channel production plus strict rollout recording;
- atomic project admission/promotion, per-iteration tuple rebinding, remote
  world-selection dispatch, and `sculpt world author|show|validate`;
- capability-matrix acceptance tests: G1 rough terrain, Go1 box parkour, and
  Yam arm/gripper object-to-region all pass the same author/compiler/gate flow.

Verification for this slice: 569 affected regression tests passed in 18.16s;
the three acceptance compiles plus genericity assertion passed in 4.37s; the
CLI author/show/validate E2E passed in 3.37s. All tests are CPU/headless and
well below the one-hour cap.

The active plan is:

1. capability descriptors plus strict WorldSpec/TaskSpec and persistence (done);
2. real mjlab compiler, admission gates, eval manifest, ChannelCatalog (done);
3. prompt author/clarifier (done) and KG grounding (pending);
4. reward/metric/atomic-selection integration (mostly done), then finish
   diagnoser/curriculum/run-memory integration;
5. backend/UI authoring, clarification, preview, and lineage workflows;
6. bounded tests, adversarial review, docs, and an incremental commit per
   coherent slice.

The architecture/core/UI seam reviews are complete. Their key constraint is
that legacy `env/current.json` remains separate; authored runs consume one
atomic selection file, and evaluation must load materialized terrain rather
than replaying a seed. Keep test commands below one hour; prefer focused suites and headless
`MUJOCO_GL=egl` smoke tests. Preserve the unrelated untracked historical log
files under `RewardSculptor/`.

## Resume instruction

Inspect `git status`, recent commits, this handoff, and the architecture.
Continue the first incomplete numbered item above, update this file with exact
files/tests/commits after each increment, and do not claim the vision is
complete until all three acceptance flows (terrain, parkour, object-to-region
with a gripper-capable humanoid) pass their documented gates.
