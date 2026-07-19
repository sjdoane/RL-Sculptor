# RewardSculptor: Roadmap to a Research-Grade Robot-Learning System

Status: strategic research and product guide, 2026-07-19.

Scope: this document describes what RewardSculptor should become and why. It
is deliberately broader than an implementation plan, but it is ordered by
scientific dependency so that ambition does not outrun evidence. It reflects
the repository through commit 9e2d646, including the reference-motion program,
metric-trust pipeline, capability-driven WorldSpec and TaskSpec compiler,
interactive World tab, immutable evaluation lineage, and train-only world
variation editing.

## Executive decision

RewardSculptor should become a **trustworthy closed-loop robot-learning
scientist**:

> From a natural-language intent, it should author a typed task and world,
> propose reward, curriculum, environment, hyperparameter, and eventually
> algorithm hypotheses, run controlled experiments, learn from causal
> evidence, and return both a trained policy and an auditable evidence package.

That is a stronger and more defensible goal than “an LLM that writes rewards”
or “a UI that builds robot worlds.” It combines three research programs:

1. **Evaluation authority:** determine when a generated metric, success
   predicate, reference, or multimodal judge deserves to influence training.
2. **Causal agentic search:** determine whether diagnosis- and
   literature-guided hypotheses improve robot learning more efficiently than
   blind or population-based search.
3. **Embodiment-general task authoring:** determine whether one typed,
   capability-driven language can safely create useful tasks and worlds across
   humanoids, quadrupeds, robot arms, grippers, and future robots.

The immediate bottleneck is no longer feature count. The system has many
important mechanisms, but few of its central claims have been demonstrated
with matched baselines, held-out evaluation, cross-robot studies, and
statistically honest replication. The highest-value next move is therefore to
turn the current system into a credible scientific instrument before greatly
expanding its surface area.

The desired progression is:

- **Instrument:** every result is reproducible, evaluation is calibrated, and
  the system knows when it does not know.
- **Scientist:** search is hypothesis-driven, branch-parallel, causal, and
  resource-aware.
- **Platform:** new labs can add a robot, task family, or simulator without
  weakening the scientific controls.
- **Physical autoresearch system:** the same reset, execute, verify, refine
  abstraction eventually extends to real hardware under a formal safety case.

## 1. Honest baseline: what exists now

The starting point is much stronger than the older planning documents imply.
The following should be treated as shipped foundations, not proposed future
features.

| Area | Current capability | Honest evidence status |
|---|---|---|
| Reward-learning loop | Staged train, rollout, diagnose, constrained reward/environment edit, keep/revert, provenance, and recovery paths | Mechanically extensive; not yet shown to beat strong matched baselines consistently |
| Evaluation trust | Generated metrics, contract and AST gates, synthetic degenerates, reference-grounded positives, calibration, realism gates, Goodhart checks, and authority labels | Strong engineering and rich failure evidence; human calibration and a complete base-rate study remain open |
| Reference program | Retrieval, segmentation, retarget-aware metadata, start-pose handling, synthetic fallback exemplars, reference certificates, and stage-scoped metrics | Demonstrated on live get-up missions; still has corpus quality, cross-robot, and systematic benchmark gaps |
| Knowledge graph | Paper, technique, failure-mode, and run-case memory; evidence-conditioned retrieval; staleness rotation; citation grounding; outcome statistics | Useful provenance and adaptation substrate; causal benefit over no-KG remains unproven |
| Environment authoring | Strict WorldSpec and TaskSpec, capability resolution, deterministic compilation, admission gates, materialized evaluation manifests, train-only variation surfaces, clarification provenance, and atomic tuple selection | Real author/compiler/gate path; compiling a task is not evidence that a policy can learn it |
| World product | Interactive compiled-scene viewer, click inspection, clarification-driven dry-run preview, robot mismatch guard, train-variation editing, and evaluation-lineage display | High-quality authoring interface; not itself a research result |
| Robot generality | Semantic capability descriptors and acceptance compiles for G1 terrain, Go1 parkour, and Yam arm/gripper object-to-region flows | Good architecture proof across three embodiment classes; not yet a cross-robot learning benchmark |
| Product and operations | Full local UI, run monitoring, reports, GPU awareness, robot library, KG tools, mission review, policy export, and extensive tests | Strong prototype; durable orchestration, packaging, and multi-lab deployment remain incomplete |

Three distinctions must remain explicit in every paper, demo, and UI:

1. **Authored** means a typed specification was produced.
2. **Admitted** means the scene compiled and passed mechanical gates.
3. **Solved** means a policy met a frozen held-out evaluation protocol with
   uncertainty reported.

The current system proves the first two on several flows. It must not imply the
third until the corresponding training evidence exists.

## 2. What the lab-shared systems change

### 2.1 Prompt-to-Policy

KRAFTON's [Prompt-to-Policy](https://github.com/krafton-ai/Prompt2Policy)
turns an intent into a reward, trains multiple seeds and hyperparameter
configurations, combines a code judge with a VLM judge, and revises from
experiment lineage. Its public description adds several particularly useful
ideas: criteria are committed before the rollout is shown; the revise agent
can branch from an earlier iteration; ordinary revisions are constrained to
one or two changes; accumulated lessons have trust tiers; and structural
rewrites are reserved for plateaus. The project also reports its own important
limitations: inconsistent VLM scores, orientation errors, difficulty judging
subjective motion, over-correction of near-successes, and early simulator
coverage. See the [KRAFTON technical
overview](https://www.krafton.ai/blog/posts/2026-04-03-prompt-to-policy/prompt-to-policy_en.html).

RewardSculptor should adopt:

- multi-seed and multi-configuration candidates as a standard search unit;
- lineage-wide branching instead of a forced single chain;
- small-change discipline with a separately justified structural-rewrite
  mode;
- precommitted visual criteria and independent physics cross-checks;
- explicit lesson confidence, retirement, and consolidation;
- guardrails for component dominance, plateaus, and repeated regressions.

RewardSculptor should go beyond it by:

- keeping generated judges outside the selection loop until their calibration
  is measured;
- grounding visual criteria in TaskSpec and the materialized world rather than
  relying on model priors;
- separating reward, world, curriculum, hyperparameter, and algorithm
  interventions so results are causally attributable;
- reporting held-out cross-robot results and statistical uncertainty, not
  merely successful examples;
- treating the evaluator as an untrusted artifact with a revocable authority
  level.

### 2.2 ENPIRE

[ENPIRE](https://arxiv.org/abs/2606.19980) argues that physical autoresearch
needs a repeatable reset, execute, verify, refine loop. Its Environment,
Policy Improvement, Rollout, and Evolution modules turn robot experiments into
an agent-operable process. Especially relevant ideas include immutable
environment APIs after construction, automatic reset and verification,
branch-per-hypothesis search, reuse of successful recipes across branches,
stage-wise progress, fixed evaluation cases, and explicit robot, GPU, token,
and time-to-success accounting. Its [project
site](https://research.nvidia.com/labs/gear/enpire/) makes the hypothesis tree
and cross-agent reuse unusually legible.

RewardSculptor should adopt:

- a first-class hypothesis tree rather than only a reward-version timeline;
- experiment recipes that can be compared, reused, rejected, and transferred;
- stage-wise behavioral milestones before final success;
- utilization and cost as research outcomes, not operational footnotes;
- automatic reset and verification as part of the environment contract;
- a clean distinction between one-time environment construction and repeated
  policy improvement.

RewardSculptor should go beyond it by:

- using pre-registered selection and held-out evaluation sets rather than a
  single rolling success target;
- separating one-shot precision, recovery after failure, and best-of-k retry
  success;
- measuring branch diversity and causal contribution, not only wall-clock
  acceleration;
- preventing peers from propagating a spurious winning recipe without fresh
  reproduction;
- making robot compatibility explicit through conformance levels rather than
  assuming a homogeneous fleet;
- retaining a frozen metric firewall even when agents can edit training
  infrastructure.

ENPIRE also supplies a caution: larger agent fleets reduced wall-clock time
but increased token cost super-linearly and lowered per-robot utilization.
Parallelism is therefore not an unconditional good; it should be allocated
when expected information gain exceeds the coordination cost.

### 2.3 Other load-bearing references

- [Eureka](https://arxiv.org/abs/2310.12931) establishes a strong
  population-based LLM reward-search baseline. RewardSculptor must compare
  against it under matched proposal count and training budget.
- [Eurekaverse](https://arxiv.org/abs/2411.01775) shows why environment
  curricula and held-out course diversity matter for parkour and transfer.
- [OMNI-EPIC](https://arxiv.org/abs/2405.15568) demonstrates the promise of
  open-ended task and environment generation, while explicitly observing
  that available VLMs were not accurate enough to serve as universal success
  detectors. Typed generation and evaluator skepticism are advantages, not
  restrictions.
- [RL-VLM-F](https://arxiv.org/abs/2402.03681) supports pairwise visual
  preferences over raw absolute VLM scores.
- [Adversarial Reward Auditing](https://arxiv.org/abs/2602.01750) motivates
  active hacker-versus-auditor evaluation rather than reliance on a fixed
  negative battery.
- [rliable](https://arxiv.org/abs/2108.13264) motivates IQM, interval
  estimates, performance profiles, and uncertainty-aware conclusions in
  small-sample RL.
- [DrEureka](https://arxiv.org/abs/2406.01967) motivates joint reward and
  domain-randomization research, but RewardSculptor should isolate those
  interventions when measuring causality.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
  provides public locomotion and manipulation tasks, expert rewards, and a
  practical comparison surface.
- [RoboCasa365](https://arxiv.org/abs/2603.04356) provides a longer-term
  manipulation and generalization surface with many tasks and diverse scenes.
- [Meta-Harness](https://arxiv.org/abs/2603.28052) suggests a later outer loop
  that optimizes the agent harness itself, but only after RewardSculptor has a
  trusted held-out objective for harness quality.

## 3. Research identity: three claims, not one kitchen-sink claim

### Claim A — generated evaluation can earn bounded authority

**Question:** Can a layered, reference- and adversary-grounded trust pipeline
predict when a generated robot-behavior metric agrees with blind human
judgment and resists known gaming behaviors?

This is the strongest near-term paper because most required artifacts already
exist, much of the work is offline, and both positive and negative outcomes
are informative. The contribution is not “our metric is correct.” It is a
measured authority protocol that can accept, restrict, or reject a metric.

### Claim B — grounded diagnosis improves the proposal distribution

**Question:** At matched compute and search topology, do diagnosis, retrieved
research, run-case memory, and typed world edits produce better hypotheses
than blind resampling or generic LLM revision?

The object of study is the **proposal distribution**, not whether one
particular single-chain run beats a population method. Both treatment and
control should receive the same candidate count, screening budget, seeds, and
selection rule.

### Claim C — typed prompt-to-world authoring transfers across embodiments

**Question:** Can one capability-driven WorldSpec and TaskSpec language author
valid, learnable, evaluation-safe tasks across distinct robot morphologies
with less task-specific code and fewer invalid generations than free-form code
generation?

This claim requires more than compilation. It needs cross-robot authoring
success, learnability probes, full training, held-out variations, and an
external-adopter study.

These claims can become separate papers or releases. Combining them in an
initial paper would make every weak link a reviewer target and obscure the
strongest result.

## 4. Non-negotiable design principles

1. **Evaluator independence.** Training reward, search fitness, final success,
   and human-facing qualitative judgment are separate artifacts with separate
   provenance.
2. **Revocable authority.** Every metric and judge has a measured authority
   level, calibration date, valid task domain, and abstention behavior.
3. **Frozen evaluation.** The optimizer never edits or silently regenerates
   the final evaluation world, seeds, success logic, or reference set.
4. **Fresh confirmation.** Selection seeds and final evaluation seeds differ;
   winners are re-evaluated after selection.
5. **Causal interventions.** Every experiment declares what changed and what
   stayed fixed. Multi-variable edits are exploratory evidence, not causal
   evidence.
6. **Capability-driven embodiment.** Core logic uses semantic roles and
   capabilities, never G1, Yam, gripper, or task-name conditionals.
7. **Typed generation.** Language models propose declarative, bounded
   artifacts or reviewed patches; they do not receive unrestricted authority
   over simulator or evaluation code.
8. **Uncertainty is a result.** Disagreement, low confidence, insufficient
   evidence, and abstention must be visible and actionable.
9. **Cost is part of performance.** GPU time, environment steps, LLM tokens,
   wall-clock time, retries, and human interventions are reported alongside
   success.
10. **Negative results remain first-class.** Failed hypotheses, rejected
    worlds, metric exploits, and non-transferable recipes are retained and
    queryable.
11. **A valid world is not necessarily learnable.** Mechanical admission and
    policy-learning evidence stay separate.
12. **Human control remains available.** Users can steer, freeze, compare,
    revert, and inspect any autonomous decision without corrupting lineage.

## 5. Priority 0: establish scientific truth

Nothing in later sections is more important than this block.

### P0.1 Freeze a benchmark charter

Define in advance the system versions, task families, robots, baselines,
budgets, selection rule, final metrics, exclusions, and analysis plan. The
charter should be immutable for a campaign. This prevents the system from
changing its question after seeing results.

### P0.2 Build a graded cross-embodiment task suite

The first suite should cover at least:

- a cheap control sanity task;
- flat and rough locomotion;
- a staged whole-body transition such as get-up;
- parkour or waypoint traversal;
- non-prehensile object interaction;
- arm reach, push, and object-to-region;
- one gripper-dependent manipulation task;
- one long-horizon composition task.

It should include at least one humanoid, one quadruped, and one robot arm.
Tasks should be graded from compile-only through full held-out solution so
failure location is visible.

### P0.3 Repair or replace every benchmark success specification

Every “ground-truth” success predicate should be attacked with stillness,
falling, oscillation, explosion, early termination, threshold flicker,
teleport-like reset artifacts, time truncation, and proxy-only behavior. A
specification that fails is valuable evidence, but it cannot remain the
campaign authority.

### P0.4 Run the metric-gaming and human-anchor study

Generate naive and gated metrics for representative tasks, score archived
honest and gamed rollouts, and obtain blind human pairwise rankings. Measure:

- false competence on known gaming classes;
- rejection rate and false rejection rate at each trust layer;
- correlation and calibration against human preference;
- agreement across tasks, robots, and motion families;
- whether trust scores predict actual evaluator reliability;
- which exploit classes remain undetected.

This should produce a public “metric gauntlet” dataset and evaluation tool,
not only a table in a paper.

### P0.5 Establish a held-out evaluation battery

Separate:

- authoring examples from unseen prompts;
- training worlds from unseen geometries and physics;
- selection seeds from fresh evaluation seeds;
- metric calibration clips from human-test clips;
- familiar robots from held-out robot variants;
- in-distribution reset states from structured OOD conditions.

The final claim must be based on artifacts the search loop never saw.

### P0.6 Make the statistical protocol mandatory

Report per-task distributions, IQM, stratified bootstrap confidence
intervals, paired differences, performance profiles, and sensitivity to seed
count. Predefine tie/noise bands and minimum practically important effects.
Never let a single best video or best seed stand in for a result.

### P0.7 Create matched baseline and ablation arms

At minimum:

- fixed human-expert reward;
- Eureka-style population search;
- generic LLM revision without diagnosis;
- blind or random bounded edits;
- current system without KG;
- current system without run-case memory;
- current system without curriculum/world edits;
- current system with reward-only versus world-only interventions;
- an oracle-metric upper-bound arm;
- a P2P-inspired reward plus hyperparameter search arm.

All arms should receive matched model, proposal, environment-step, seed, and
wall-clock budgets where possible.

### P0.8 Remove causal confounding from the iteration loop

Reward, world, reset curriculum, PPO hyperparameters, reference use, and
algorithm changes need independent treatment labels. Exploratory combined
edits can remain, but the system must reproduce a gain with isolated or
factorial interventions before promoting a causal lesson into memory.

### P0.9 Re-run the core campaign on the current system

The earlier E4 result predates major convergence, metric, reference,
curriculum, KG, and world-authoring changes. A new campaign should be
pre-registered and run only after P0.1-P0.8. A positive result supports the
closed-loop claim; a null result still leaves a strong metric-trust result and
identifies which mechanisms fail to add value.

### P0.10 Archive the complete experiment operator

Persist the exact model and provider, prompt, response, tool trace, retrieved
evidence, random seed, code hash, dependency lock, world tuple, metric
certificate, hardware, training config, and decision rationale. LLM outputs
cannot be reproduced exactly; their inputs and effects must be.

### P0.11 Close the known authority and reference-integrity debt

The existing audit already identifies several defects that should be closed
before a new headline campaign:

- replace filename-only reference clip IDs with collision-safe,
  provenance-preserving IDs;
- filter retrieval by robot, start pose, contact phase, and motion scope before
  semantic ranking, rather than expecting later QC to reject a bad binding;
- represent normalized versus absolute root heights explicitly and repair
  affected corpus metadata;
- propagate the project robot through reference search, attachment, preview,
  and conversion, and reject incompatible robot/reference pairs;
- provide a mission-wide re-certification sweep so grandfathered steering
  metrics cannot bypass newer gates;
- unify reference certificate authority with the actual live steer/observe
  decision path;
- feed synthetic-exemplar signatures into diagnosis and editing with their
  lower trust tier visible;
- extend positive batteries to overshoot-then-settle, intermediate-speed, and
  short-hold behaviors;
- add hard timeouts to every model call plus a job-level metric-generation
  watchdog;
- finish process/container isolation for generated metric and criterion code;
- select p10, median, and p90 episode videos for review rather than presenting
  only the best-return episode.

These are not miscellaneous cleanup items. Each one can invalidate a metric,
reference, or campaign conclusion, so they belong in the scientific
foundation rather than a later polish backlog.

## 6. Evaluation and metric authority

### E1. Formalize success as temporal task semantics

Extend TaskSpec from terminal predicates toward composable temporal semantics:
sequence, hold, repetition, forbidden intervals, recovery, “reach then
maintain,” and “never violate.” This prevents brittle free-text criteria and
bare any-event checks from becoming success authority. The same semantics
should compile into monitoring channels, human-readable rubrics, and test
generators.

### E2. Introduce an evaluator authority ladder

A proposed ladder:

- **A0 — rejected:** invalid, degenerate, or exploitable.
- **A1 — descriptive:** may appear in reports but cannot diagnose or select.
- **A2 — advisory:** may influence diagnosis, never keep/revert.
- **A3 — steering:** may select within a calibrated task and robot scope.
- **A4 — reporting:** independently validated for final benchmark reporting.
- **A5 — external:** human, hardware, or benchmark authority outside the
  optimizer's control.

Authority should expire when the task, world lineage, robot, sensor contract,
or model version changes.

### E3. Replace a fixed negative list with active adversarial auditing

Keep the existing archetypes as regression tests, then add a policy or search
process whose explicit objective is to maximize the candidate metric while
violating TaskSpec. The resulting counterexamples enlarge the gauntlet. A
metric is not “unhackable”; it is certified only against a recorded attack
budget and exploit family.

### E4. Build the grounded multimodal judge

The best judge should:

- derive its rubric from frozen TaskSpec semantics;
- commit criteria before seeing outcome evidence;
- receive multiple synchronized camera views;
- receive a telemetry strip with contacts, predicate state, object pose,
  traversal, and phase boundaries;
- judge p10, median, and p90 episodes rather than a cherry-picked rollout;
- use blinded, order-randomized pairwise comparisons against a reference or
  best-so-far behavior;
- cross-examine visual claims against trajectory data;
- report per-criterion evidence, uncertainty, and abstention;
- remain advisory until calibrated against blind human labels.

It should never receive reward code, iteration number, prior score, or edit
history. Judge disagreement with the objective metric is a diagnostic event,
not an automatic verdict.

### E5. Create a judge-calibration benchmark

Build a labeled set spanning robot types, camera orientations, occlusion,
speed, contact-rich behavior, subjective naturalness, and adversarial clips.
Measure order bias, self-agreement, human agreement, robot/motion domain
shift, calibration error, and sensitivity to camera/telemetry removal.
Without this dataset, changing VLMs only changes an unmeasured dependency.

### E6. Treat disagreement as information

Metric versus judge, judge versus human, reference versus TaskSpec, and
per-seed disagreement should each trigger a distinct investigation path.
Possible interpretations include metric gaming, judge bias, ambiguous intent,
bad reference scope, stochastic training, or a flawed world. The system should
not collapse all disagreement into one score.

### E7. Add evaluator lifecycle management

Track who or what authored the evaluator, the evidence it passed, known
counterexamples, revisions, applicable domains, and calibration drift. Re-run
certification when dependencies change. Retired evaluators remain visible so
the same exploit is not rediscovered.

## 7. Causal, branch-parallel agentic search

### S1. Make a hypothesis a first-class artifact

Every proposal should state:

- observed failure;
- causal hypothesis;
- proposed intervention;
- variables intentionally held fixed;
- expected directional effect;
- cheapest discriminating experiment;
- failure interpretation;
- transfer conditions.

This converts an edit log into a research record.

### S2. Move from a single chain to controlled branch search

Generate a small diverse portfolio of diagnosis-grounded candidates from the
same parent. Give the blind baseline the same number of candidates. Branch
from any prior strong state, not only the latest. Require fresh reproduction
before merging a recipe across branches.

The research question becomes: does grounded diagnosis produce a better
candidate set per unit cost?

### S3. Use multi-fidelity experimentation

Screen candidates with contract checks, replay tests, short training,
low-environment-count probes, and paired rollout seeds. Promote only
informative candidates to full training. Calibrate the screening fidelity so
it does not systematically discard slow-starting methods.

### S4. Search reward, world, curriculum, hyperparameters, and algorithms
without losing attribution

The eventual system should co-design all of these, but through explicit
experiment types:

- reward-only;
- world/curriculum-only;
- hyperparameter-only;
- algorithm/representation-only;
- pre-registered factorial combinations;
- exploratory combined interventions that require later confirmation.

This is more scientifically useful than unconstrained code editing.

### S5. Optimize branch diversity, not just branch count

Multiple agents proposing coefficient changes are not a useful population.
Candidate portfolios should span distinct mechanisms: reward decomposition,
reset distribution, exploration, reference/imitation, optimizer stability,
observation design, policy architecture, and task decomposition. Measure
hypothesis diversity and marginal information gain.

### S6. Turn memory into causal recipe learning

Run-case memory should distinguish:

- correlated observation from reproduced effect;
- robot-specific from morphology-level evidence;
- task-specific from reusable mechanism;
- strong, soft, contradicted, and retired lessons;
- direct evidence from model-recalled suggestions;
- positive transfer from negative transfer.

Recipes should specify preconditions and failure signatures, not universal
rules.

### S7. Add a resource-aware experiment planner

The planner should choose experiments by expected information gain per GPU
hour, environment step, token, and human minute. It should decide when to run
parallel branches, when to wait for a definitive full run, and when evidence
is already sufficient to stop. Report time-to-threshold and cost-to-threshold
as primary system metrics.

### S8. Optimize the harness only after the evaluator is trusted

A later Meta-Harness-style outer loop can modify prompts, memory policy,
candidate count, tool order, compression, or retry logic. It must optimize on
held-out tasks and models, with the benchmark and evaluator outside its edit
surface. Otherwise it will merely overfit the research harness.

## 8. World and task authoring beyond the current World tab

### W1. Expand the world grammar by physical category

Grow in controlled layers:

1. static terrain and primitive obstacles;
2. free rigid objects and non-prehensile interaction;
3. articulated mechanisms such as doors, drawers, hinges, and levers;
4. robot-actuated tools and constrained objects;
5. moving obstacles and scripted dynamic entities;
6. perception-dependent and partially observable tasks;
7. deformable or granular objects only after the evaluation and simulator
   contracts can support them;
8. multi-robot or human-robot scenes as a frontier layer.

Each category needs capability checks, channel semantics, admission gates,
benchmark tasks, and explicit simulator support. Visual expressiveness alone
is not sufficient.

### W2. Add a compositional long-horizon TaskSpec

Support task graphs with ordered, optional, parallel, and recovery phases;
state-machine transitions; subgoal-specific observations; reset checkpoints;
and success semantics at both phase and mission level. This should represent
“find, approach, grasp, carry, place, recover if dropped” without embedding a
task-specific program in the compiler.

### W3. Separate physical feasibility from learnability

The admission chain should eventually include:

- mechanical compilation and settle;
- kinematic reachability;
- observation sufficiency;
- success-predicate observability;
- reset reliability;
- sparse-success probe;
- reference or scripted feasibility where available;
- short policy-learning probe;
- uncertainty label when none is decisive.

The result should say “mechanically valid but unproven learnable” rather than
pretending a reachability check solves exploration.

### W4. Turn the World author into a benchmark generator

Generate controlled families that vary one scientific factor at a time:
roughness, gap, occlusion, object mass, friction, distractors, goal pose,
sensor noise, embodiment scale, and reset distribution. This enables
performance profiles and causal robustness studies rather than isolated
showcase scenes.

### W5. Add open-ended task generation behind a safety and usefulness filter

Once fixed benchmarks are credible, use learned and failed task archives to
propose tasks that are novel, useful, and near the policy's competence
frontier. Unlike free-form code systems, generated tasks should still compile
through WorldSpec and TaskSpec and use a separate novelty/learnability review.
Open-ended generation should never contaminate the fixed benchmark suite.

### W6. Make human world interaction scientifically meaningful

The interactive World tab should eventually record user changes as explicit
hypotheses: what was changed, why, whether evaluation lineage changed, and
what learning effect followed. Human edits can then serve as a baseline and a
source of preference data rather than disappearing into UI state.

### W7. Add world counterexample generation

Search for valid worlds that break a policy, metric, reset, or capability
assumption while respecting a bounded variation grammar. Preserve minimal
counterexamples and promote them to the regression suite. This turns world
authoring into active robustness testing.

### W8. Preserve a simulator-neutral semantic layer

Do not prioritize Isaac Sim integration now. First prove that RobotCapability,
WorldSpec, TaskSpec, ChannelCatalog, evaluation manifests, and the conformance
suite are independent of mjlab naming and storage details. A later simulator
adapter should translate the same semantic artifact rather than create a
second product.

## 9. Robot generality as a measurable contract

The user-facing promise must apply to humanoids, quadrupeds, arms, grippers,
and robots not yet in the library. “Supports a robot” should therefore have
levels:

| Level | Meaning |
|---|---|
| R0 Preview | Asset loads and renders; no learning claim |
| R1 Describe | Capability roles, geometry, actuators, sensors, limits, and variants resolve |
| R2 Compile | Representative worlds/tasks compile and pass admission gates |
| R3 Learn | A standard task trains and rolls out under the adapter |
| R4 Benchmark | Multiple seeds meet a frozen held-out benchmark with uncertainty |
| R5 Transfer | Skills, recipes, or authored tasks transfer across a robot family |
| R6 Hardware | Deployment passes a robot-specific safety and verification protocol |

### G1. Build a robot conformance suite

Every new robot should run the same semantic tests for root/body roles,
end-effectors, contact groups, observations, control modes, action limits,
reset poses, scene scale, and task compatibility. Missing capabilities should
fail precisely. A robot without a gripper should still support locomotion,
contact, or pushing tasks; a robot arm with a gripper should expose grasp
capability without humanoid assumptions.

### G2. Validate across morphology classes

Every central claim should include at least a humanoid, quadruped, and arm
task, then a held-out robot where practical. A mechanism that works only on G1
is a G1 result. A mechanism that works on G1, Go1, and Yam under the same
semantic contract begins to support a system claim.

### G3. Replace remaining morphology heuristics with data

Joint realism limits, start-pose logic, contact semantics, reference
retargeting, camera conventions, termination envelopes, and reachable world
dimensions should all come from capability data or certified adapters.
Robot-name substring rules should be treated as conformance failures.

### G4. Make task compatibility bidirectional

The system should answer both:

- “Which robots can perform this authored task, and why?”
- “Which task families are valid and learnable for this robot?”

This enables task migration, robot selection, and honest capability error
messages.

### G5. Evaluate cross-robot authoring transfer

Author a task once, compile it for several compatible robots, and measure how
many clarifications, repairs, parameter adaptations, and task-specific code
changes are required. This is the direct experiment for the capability-driven
generality claim.

## 10. Learning, references, transfer, and lifelong skill growth

### L1. Complete the reference pipeline across robots

The reference program should become:

prompt or video → retrieve/generate motion → scope to the task phase →
retarget to the selected embodiment → physics-track → certify → use as
training scaffold and evaluator anchor.

References should carry morphology, contact, phase, feasibility, and quality
metadata. Synthetic exemplars remain lower authority than tracked real motion.
[General Motion Retargeting](https://arxiv.org/abs/2510.02252) and related
physics-tracking work provide a path, but cross-robot validity must be
measured rather than assumed.

### L2. Build a reusable skill graph

Store learned policies and references as skills with:

- semantic preconditions and terminal state;
- supported robots and control interfaces;
- world and observation requirements;
- success distribution and known failures;
- composability and reset points;
- transfer and fine-tuning history.

Task decomposition should retrieve proven skills before training every stage
from scratch.

### L3. Study transfer as a first-class outcome

Measure:

- warm-start versus scratch;
- reference versus no-reference;
- same-robot cross-task transfer;
- same-task cross-robot transfer;
- world-to-world transfer;
- recipe transfer versus checkpoint transfer;
- positive and negative transfer detection.

Transfer should never silently replace a fair from-scratch baseline.

### L4. Expand beyond PPO without giving agents arbitrary code authority

Expose typed choices among behavior cloning, offline RL, online RL,
offline-to-online RL, residual control, heuristic/code policies, motion
tracking, and hybrid methods. The agent may select and tune a registered
algorithm recipe. New algorithm code enters through a reviewed extension
path, not an unrestricted mutation surface.

### L5. Add perception and representation co-design

For object tasks, failures may come from observations rather than rewards.
The system should reason about camera placement, privileged versus deployable
state, history, tactile/contact signals, object-relative frames, and
representation bottlenecks. Observation edits must be isolated and benchmarked
like reward edits.

### L6. Develop robustness and sim-to-real as an evidence program

Use domain randomization, dynamics identification, latency/noise models,
sensor corruption, actuator uncertainty, and visual variation. Measure which
randomizations improve held-out transfer and which merely make training
harder. Eventually compare authored randomization to human and DrEureka-style
baselines.

### L7. Pursue lifelong and open-ended learning only after task-level proof

An eventual system can propose new tasks near the competence frontier,
retrieve or compose existing skills, train the missing capability, and update
the skill graph. It must maintain a fixed retrospective benchmark so new
skills do not erase old ones and “interestingness” does not replace utility or
competence.

## 11. Knowledge graph and scientific memory

### K1. Separate literature claims from local empirical claims

Every memory item should state whether it is a source claim, a system design
inference, a single-run observation, a replicated local effect, or a
cross-robot result. Retrieval rank should reflect evidence class and
applicability, not only semantic similarity.

### K2. Measure KG value directly

Beyond the no-KG campaign arm, test:

- retrieved relevant evidence versus random papers;
- citations shown versus hidden;
- failure-conditioned retrieval versus static task retrieval;
- paper knowledge versus run-case memory;
- correct evidence versus adversarially irrelevant evidence;
- outcome-aware ranking versus similarity-only ranking.

If KG grounding does not improve outcomes, reposition it honestly as
provenance and interpretability infrastructure.

### K3. Build causal outcome memory

A technique should gain confidence only after paired or otherwise controlled
evidence. Store effect size, uncertainty, robot, task family, failure
signature, and interaction with other edits. Memory should support
contradictory evidence rather than averaging it into false certainty.

### K4. Add active literature review with evidence hygiene

The system may search for new techniques when existing memory is weak, but it
should archive exact sources, distinguish paper results from extrapolation,
track retractions/version changes, and prevent literature retrieval from
altering the frozen evaluation.

### K5. Support private, shared, and public memory layers

Labs need private run cases and unpublished data; the community benefits from
public technique and failure taxonomies. Design explicit provenance,
licensing, access, redaction, and export boundaries so adoption does not
require leaking lab data.

## 12. Lab adoption and artifact quality

### A1. Produce a narrow, reproducible reference release

The first serious release should include:

- one-command CPU smoke;
- one bounded GPU benchmark;
- exact containers and lockfiles;
- immutable configs, seeds, and evaluation manifests;
- baseline implementations;
- raw per-seed results and analysis notebooks;
- all generated prompts/responses allowed by provider terms;
- videos tied to run hashes;
- metric gauntlet and human-label protocol;
- a limitations and negative-results appendix.

### A2. Replace stale documentation with generated capability truth

The current README and older briefs understate or misstate the system. Public
docs should derive robot counts, adapter status, commands, test counts, schema
versions, and supported task categories from registries or verified release
manifests. A lab should not have to read internal audit logs to learn what is
real.

### A3. Create an extension SDK and conformance harness

Labs should be able to add:

- a robot capability descriptor;
- simulator/task bindings;
- observation and action adapters;
- world entities and channels;
- reward and evaluator plugins;
- algorithm recipes;
- import/export formats.

Each extension receives automatic schema, unit, compile, rollout, and
benchmark checks.

### A4. Make orchestration durable

Move from an in-process local job model toward resumable jobs with explicit
state, leases, cancellation, retry policy, artifact upload, and crash
recovery. Support a simple remote single-GPU worker before attempting cluster
scale. This is necessary for multi-hour campaigns and lab servers.

### A5. Build a complete experiment registry

Every project should expose a queryable graph:

intent → clarification → world/task tuple → metric/reference certificates →
hypothesis branches → training runs → selection → held-out evaluation →
policy export.

The UI should make any score traceable to its exact artifacts in one click.

### A6. Add observability for autonomous research

Track stage health, queue time, GPU/robot utilization, LLM latency, token use,
training throughput, branch status, evaluator disagreement, world rejection,
reset failure, and human intervention. Alerts should distinguish scientific
failure from infrastructure failure.

### A7. Add deliberate human collaboration modes

Support:

- autonomous;
- approve-each-hypothesis;
- intervene-on-plateau;
- compare-and-choose;
- critique behavior in natural language;
- lock selected artifacts;
- hand off between researchers or agents.

Human feedback becomes a versioned experimental input, not an invisible
override.

### A8. Harden security and isolation

Generated reward, metric, tool, and algorithm code should run in resource-
limited processes or containers with explicit filesystem, network, import,
time, and memory policies. Uploaded robot assets and external repositories
need validation and license tracking. Hardware commands require an even
stricter capability and safety boundary.

### A9. Create adoption-focused examples

Maintain small, complete tutorials for:

- adding a new robot without a gripper;
- adding an arm with a gripper;
- authoring a terrain task;
- authoring an object task;
- building a custom metric and gauntlet;
- running a controlled ablation;
- reproducing a paper table;
- exporting a policy and its evidence package.

### A10. Establish compatibility and governance policy

Version public schemas and artifact formats; publish migration tools; define
deprecation windows; record dependency compatibility; use a proposal process
for new semantic roles and task predicates; and publish benchmark governance
so results cannot be silently invalidated.

## 13. Long-term physical autoresearch

This is a frontier program, not the next implementation sprint.

### R1. Define a hardware Environment contract

Extend the ENPIRE abstraction with typed reset, readiness verification,
safety boundaries, observations, actions, outcome verification, and recovery.
The learning agent receives a stable interface; it cannot edit hard safety
limits or final verification during policy improvement.

### R2. Require a safety case per robot and task

Declare joint/workspace/force/velocity limits, collision zones, watchdogs,
emergency stop, human proximity rules, maximum unattended duration, allowed
controllers, and recovery behavior. Safety violations are first-class
failures and benchmark metrics.

### R3. Make reset reliability a research metric

Measure reset success, time, state distribution, accumulated drift, object
wear, and manual interventions. A system that learns quickly but requires a
human reset every third trial is not autonomous.

### R4. Build multimodal physical verification

Fuse calibrated cameras, proprioception, force/torque, tactile sensing, and
task-specific perception. Verification code should be trained and tested on
held-out success/failure examples and operate under latency constraints.
Ambiguity must trigger abstention or human review.

### R5. Use shadow and staged deployment

Progress from simulation, to replayed hardware data, to shadow decisions, to
low-energy or constrained trials, to supervised autonomy, and only then to
unattended loops. Sim-to-real evidence and safety evidence advance
independently.

### R6. Scale to fleets only after single-station validity

When multiple robots are available, assign hypotheses asynchronously, share
recipes through verified artifacts, reproduce wins on a second station, and
report robot utilization, token cost, wall-clock acceleration, and hardware
variance. Fleet size is an experimental factor, not a success metric.

## 14. Minimum benchmark and reporting design

### 14.1 Task matrix

| Family | Example capability under test | Embodiments |
|---|---|---|
| Control sanity | balance or simple locomotion | cheap Gym or mjlab task |
| Locomotion | velocity, rough terrain, recovery | quadruped and humanoid |
| Whole-body transition | get-up, crouch-to-stand, jump frontier | humanoid |
| Navigation/parkour | waypoint sequence, boxes, gaps, beam | quadruped and humanoid |
| Non-prehensile interaction | push/kick object into region | legged robot and arm |
| Prehensile manipulation | reach, grasp, carry, place | one or more arms/grippers |
| Long-horizon composition | multi-phase object or tool task | arm/gripper first |

Each task should have:

- a human-authored reference environment and reward where available;
- a frozen TaskSpec and held-out world family;
- a known-success or feasibility reference;
- a gaming/counterexample set;
- a fixed selection budget;
- fresh final seeds;
- an explicit unsupported/abstain path.

### 14.2 Reported outcomes

Report at least:

- task success and phase success;
- IQM and confidence interval;
- probability of improvement over each baseline;
- time and samples to threshold;
- GPU hours, environment steps, LLM tokens, and human interventions;
- authoring validity and admission-repair rate;
- reset and infrastructure failure rate;
- held-out world and robot-variant performance;
- metric/judge/human agreement;
- safety and realism violations;
- branch diversity and number of hypotheses tried;
- reproducibility across reruns and machines.

### 14.3 Prevent misleading aggregation

Do not average compile success, learnability, and final task success into one
number. Do not mix one-shot and retry success. Do not compare fitness across
evaluation hashes. Do not report the selected seed as the expected result.
Always retain per-task and per-robot results beside aggregate statistics.

## 15. Research hypotheses worth testing

| ID | Hypothesis | Informative null result |
|---|---|---|
| H1 | Trust-gated metrics produce fewer false-competence judgments than naive generated metrics | Existing gates do not predict human judgment; redesign authority rather than adding more gates |
| H2 | Diagnosis-guided candidates outperform blind candidates under matched best-of-k search | Search topology, not diagnosis quality, explains prior gains |
| H3 | KG-grounded diagnosis improves sample or compute efficiency | KG is primarily provenance infrastructure |
| H4 | Typed world/curriculum edits improve convergence without reducing held-out robustness | World co-design is unnecessary or overfits the train distribution |
| H5 | Branch-parallel search improves time-to-threshold enough to justify token and coordination cost | Single-agent search is more efficient at available scale |
| H6 | Reference scaffolds unlock transitions that reward shaping alone does not | Reference cost or mismatch outweighs benefit for the selected tasks |
| H7 | Capability-driven authoring transfers across robot classes with fewer invalid generations and less task-specific code | The semantic abstraction is incomplete or still morphology-specific |
| H8 | Grounded multimodal judging correlates better with humans than pixels-only or telemetry-only judging | One modality dominates; simplify the judge |
| H9 | Causal recipe memory improves transfer to new tasks | Retrieved recipes cause negative transfer or add no value |
| H10 | Open-ended authored curricula improve a fixed retrospective benchmark | Task novelty does not translate into reusable competence |

The roadmap should be judged by how quickly it produces credible answers to
these questions, not by how many features are added.

## 16. Sequencing and exit gates

### Horizon A — credible instrument

Complete P0.1-P0.10, evaluator authority levels, the metric gauntlet, human
anchor, fixed benchmark manifests, baseline arms, complete provenance, and
public reporting. Exit only when a third party can reproduce one benchmark
table and audit every selected result.

### Horizon B — effective agentic researcher

Add hypothesis artifacts, controlled best-of-k branches, multi-fidelity
screening, causal recipe memory, and resource-aware planning. Exit only when a
matched campaign determines whether grounded diagnosis improves the proposal
distribution.

### Horizon C — cross-embodiment authoring platform

Complete robot conformance levels, multi-stage tasks, expanded world grammar,
learnability probes, and cross-robot transfer studies. Exit only when at least
three embodiment classes pass author, compile, learn, and held-out benchmark
gates under the same semantic system.

### Horizon D — lab-grade release

Ship durable orchestration, extension SDK, containers, experiment registry,
security boundaries, tutorials, and a public reference release. Exit only
when an external lab can add a robot or task family without core-team code
changes and reproduce the expected conformance result.

### Horizon E — physical autoresearch

Add hardware contracts, reset/verification, safety cases, shadow mode, and
eventually verified fleet search. Exit only after unattended operation is
safe, reset reliability is measured, and improvements reproduce across
stations.

## 17. What not to prioritize next

- Do not make the VLM judge fitness or keep/revert authority because its
  output looks perceptually convincing.
- Do not run a large campaign before repairing benchmark specifications and
  freezing the analysis protocol.
- Do not claim object manipulation is solved because an object world compiles
  or renders.
- Do not broaden the World tab faster than its task families can be evaluated.
- Do not allow unrestricted environment or evaluator code generation; the
  typed specification is a core advantage.
- Do not combine reward, world, curriculum, seed, and algorithm changes and
  then store the result as a causal lesson.
- Do not optimize the agent harness against the same tasks used to report its
  quality.
- Do not build special G1 or gripper paths in core logic. Add capabilities,
  roles, and adapter conformance instead.
- Do not prioritize Isaac Sim compatibility before the semantic adapter
  boundary has been validated against a second backend.
- Do not treat more agents, robots, GPUs, or tokens as automatically better;
  measure their marginal value.
- Do not hide abstention, rejected metrics, failed worlds, negative transfer,
  or null campaign results.
- Do not let documentation continue to advertise stale counts or outdated
  limitations.

## 18. The next fifteen concrete decisions

1. Freeze the three-claim research framing in section 3.
2. Choose the first benchmark robots and tasks across humanoid, quadruped,
   arm, and gripper capability.
3. Write and lock the benchmark charter.
4. Repair the ground-truth task specifications and build their adversarial
   tests.
5. Assemble and label the metric-gauntlet rollout set.
6. Run the blind human pairwise study.
7. Implement complete LLM and experiment-operator provenance before new
   campaigns.
8. Define evaluator authority levels and show them in reports.
9. Make reward-only, world-only, and hyperparameter-only experiments explicit.
10. Add matched diagnosis-guided versus blind best-of-k search.
11. Re-run the current system against strong baselines under the frozen
    protocol.
12. Build the cross-robot conformance suite and publish support levels.
13. Calibrate the grounded multimodal judge as an advisory signal.
14. Replace stale public documentation with release-manifest truth.
15. Package the metric study and one cross-embodiment benchmark as the first
    research release.

## 19. Definition of success

A lab should want to adopt RewardSculptor when it can answer, with evidence:

- What exactly did the user ask for?
- What world, task, robot capability, reward, curriculum, reference, and
  algorithm were actually used?
- Which parts were authored by models, retrieved from research, selected by
  humans, or generated by deterministic compilers?
- Why was a candidate tried?
- What changed, what stayed fixed, and what effect was reproduced?
- How trustworthy was the evaluator, and on what domain was it calibrated?
- Did the result hold on fresh seeds, unseen worlds, and another compatible
  robot?
- What did it cost in compute, tokens, samples, retries, and human time?
- What failed, and can the failure be reproduced?
- Can another lab recreate the result from the released artifacts?

The long-term vision is not a system that always claims success. It is a
system that can autonomously improve robot policies while remaining honest
about what it knows, what it changed, why it believes the change helped, and
where its authority ends.
