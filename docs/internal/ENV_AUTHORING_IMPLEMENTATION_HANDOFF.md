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

The next coherent implementation slice is committed as `82bf032` (also
fixes the stale `fake_resolve` stub in `test_mission_run.py` for the new
`channel_catalog` kwarg; full suite re-verified at 2,043 passed / 1
optional-JAX skip):

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

The third slice — KG grounding for the author — is complete:

- `sculptor/world/grounding.py`: retrieval-only, fail-soft bridge from the
  shared knowledge graph to authoring. `gather_grounding(prompt)` returns
  Technique/FailureMode/Paper `GroundingItem`s (semantic retrieval at the
  0.35 prompt-time floor; papers additionally filtered by the author-intent
  tag map terrain/parkour/objects). Any retrieval failure degrades to
  ungrounded authoring with one warning — it can never block the author.
- `author.py`: `grounding_context` param; the model request gains a
  `kg_grounding` evidence block (input-only — the model's output key set
  stays closed), and the retrieved node IDs are re-injected into
  `meta.grounding` when a model omits or clears the ledger.
- `cli.py world author`: `--kg-grounding/--no-kg-grounding` (default on),
  node IDs surface in the JSON result and the human summary.
- `tests/test_world_grounding.py`: 7 tests — three-kind retrieval, intent
  tag filter, broken-embedder and empty-store fail-soft (embedder must not
  even load on an empty store), offline + model ledger paths, CLI default
  and disable flag. Live smoke against the shared graph: terrain prompt →
  fractal-terrain-training + risky-terrain papers at sim 0.56-0.67;
  ball-into-goal prompt → humanoid-soccer papers.

Also in the graph since the second slice: ENPIRE (`paper:2606.19980`) and
KRAFTON Prompt2Policy (`paper:krafton-p2p-2026`, first non-arxiv Paper
node), both LLM-extracted; four candidate roadmap items are recorded in
`RL_SCULPTOR_AUDIT.md` (VLM advisory judge, multi-seed training,
branch-parallel edits, behavioral milestones — none may weaken the
metric firewall).

Plan item 4 (diagnoser/curriculum/run-memory) is now partially complete —
three committed increments:

- **World-aware run cases** (`dc8c87a`): `RunCase` records
  `world_tuple_hash` / `world_version` / `task_version`, read fail-soft
  from the iteration's pinned selection file (`_world_identity` in
  `kg/cases.py`; `IterOutcome` already carried hash+path).
- **Within-run terrain curriculum** (`bdc0c94`):
  `apply_world_selection(train=True)` widens the compiled generator
  difficulty from the pinned nominal to
  `train.curriculum.difficulty_range` (curriculum_grid + generator only;
  `train_difficulty_span` / `expand_train_terrain_difficulty` in
  `world/compiler.py`). mjlab rows interpolate lo→hi and the base task's
  terrain-levels term promotes/demotes origins. Eval manifests, assets,
  and hashes stay computed from the pinned difficulty.
- **Train-variation edit application** (`world/project.py`):
  `WorldVariationEdit` + `apply_world_variation_edits(project_dir, edits)`
  — resolves registered `train.variations` by stable ID, per-edit
  rollback on validation failure, `meta.version` bump (parent-linked),
  full re-admission, and atomic promotion with the EXISTING evaluation
  lineage. `admit_and_promote` gained `require_eval_invariance_with`: the
  recompiled evaluation must match the prior manifest in every
  evaluation-defining field and materialized asset byte, so a train-only
  edit provably cannot move the baseline (§6.1). Rejections land in
  `env/rejected` with `eval_invariance_violations`.

The diagnoser world-edit surface and world keep/revert are also complete:

- `diagnose.py`: `# WORLD_VARIATIONS` block (`_render_world_variations_block`)
  renders the authored world's registered train variations, resolved
  best-effort from `adapter.world_selection_path` via
  `load_selected_world`; `_ProposedWorldEditModel` (variation_id +
  COMPLETE new_distribution — membership validated at packing, since IDs
  are per-world data, not a static enum); `Diagnosis.proposed_world_edits`
  persisted in diagnosis.json.
- `sculpt.py`: `_run_one_iter` applies world edits after the env edits via
  `apply_world_variation_edits` (advisory — failures emit
  `world_variations_updated` with rejections and the loop continues),
  repointing the adapter at the newly promoted pin so the NEXT iteration
  trains under it. `_promote_iteration_selection` gained
  `base_selection`: keep-best records
  `result.best_world_selection_path` and both the regression revert and
  the end-of-run best-tuple commit source the five immutable world-half
  refs from the best iteration's pinned selection — the tuple moves as
  one, and a post-best world edit can never pair with the winning
  reward. Events: `world_selection_reverted`, `world_variations_updated`;
  applied edits ride case memory as `world: <id>`.
- `tests/test_world_edits_loop.py`: 4 tests (render block; packing gated
  on surface + registered-ID membership through a stubbed diagnose;
  hallucinated edits dropped without a surface; base_selection revert
  restores the complete world half with preserved lineage).

Adversarial verification of the item-4 slices (subagent, per the
every-phase mandate): claims 2-4 CONFIRMED; claim 1 REFUTED and fixed —
`_world_identity` parsed ArtifactRef versions with `int("v1")` (always
swallowed → None in every authored run) and the original test masked it
with an integer-version fixture the store never emits. Fixed with
`_artifact_version_int` ("v<N>" + bare-int tolerated), the fixture now
uses the real string format, and the production path was empirically
re-probed (returns real versions). Also fixed the verifier's TOCTOU
finding: `apply_world_variation_edits` now holds the store lock across
read-edit-admit (FileLock is reentrant) so a concurrent UI promotion
cannot be clobbered by a train edit applied to a stale parent.

Item 4 is complete. Per-difficulty promotion statistics now flow to the
diagnoser: `_write_world_curriculum_stats` in `_mjlab_runner.py` exports
the end-of-training terrain-level histogram (mjlab promotes
`terrain_levels` on traversal success, so the level distribution IS the
per-difficulty success summary) to `<iter_dir>/world_curriculum_stats.json`,
fail-soft for plane/non-curriculum/legacy envs; `diagnose.py` loads it
best-effort and renders it inside the `# WORLD_VARIATIONS` block —
stats without registered variations create no edit surface.

Item 5 (backend) — the authoring API is complete in
`reward-sculptor-ui/backend`:

- `services/world_store.py`: author (gather_grounding → author_environment,
  session persisted under `<project>/worlds/<session>/` with the exact
  authoring inputs), apply (deterministic re-author verified against the
  persisted `draft_hash` → 409 on drift; unanswered questions take their
  disclosed system default with `default` provenance, explicit answers
  record `user`; unknown question ids are a 422, never ignored →
  `apply_clarifications` → `admit_and_promote` with
  `lineage=world-<result_hash[:24]>`), selection (shaped
  `load_selected_world` bundle), lineage (immutable `selection_v*.json`
  history).
- `routes/worlds.py`: `POST /projects/{slug}/worlds/author`,
  `POST .../author/apply` (per-project write lock, 409 busy/stale, 422
  carries the gate-violation summary), `GET .../selection` (404 before
  first authoring), `GET .../lineage`. Registered in `main.py`.
- `models/world.py` request/response shapes.
- `tests/test_worlds.py`: 6 tests running the REAL offline author and
  REAL admission gate chain (MuJoCo compile) — draft shape with
  disclosed system-default options, full author→apply→selection→lineage
  round-trip, 404/422 contracts, ungroundable-prompt 422.

The item-5 frontend slice is also complete: `lib/types.ts` + `lib/api.ts`
Worlds sections, `qk.worldSelection/worldLineage`, `hooks/useWorlds.ts`
(selection/lineage queries with `retry:false` for the 404-before-first-
authoring case; author/apply mutations invalidating both keys + the
project), `components/AuthorWorldDialog.tsx` (prompt → draft → paginated
clarification pages, every question rendered with its choices AND the
disclosed "System decides" default; unanswered = system default;
ApiError problem.detail surfaced via toasts), `components/WorldTab.tsx`
(authoritative-tuple card + train variations + immutable selection
lineage), and a `world` tab in `pages/ProjectDetail.tsx`. Verified:
`tsc -b --noEmit` clean + `vite build` clean.

The scene preview and the legacy status-key fix both landed
(`GET /projects/{slug}/worlds/preview` renders the materialized
evaluation MJB, cached per selection version + angle; `_compute_status`
now resolves the shared KG path). One recorded FUTURE direction (not
implemented, per Sam): a grounded, unbiased VLM video judge — full
design + Prompt2Policy source evidence in `RL_SCULPTOR_AUDIT.md`
(2026-07-19 entry). Advisory only; the metric firewall stays
authoritative.

The active plan is:

1. capability descriptors plus strict WorldSpec/TaskSpec and persistence (done);
2. real mjlab compiler, admission gates, eval manifest, ChannelCatalog (done);
3. prompt author/clarifier (done) and KG grounding (done);
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
