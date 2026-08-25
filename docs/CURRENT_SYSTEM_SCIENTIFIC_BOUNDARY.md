# RewardSculptor before OGMP: the current scientific boundary

**Meeting-ready snapshot:** 2026-08-24

**Scope:** one fixed reference trajectory, before modes or an oracle are
treated as part of the method.

## The five-minute answer

RewardSculptor is an **auditable policy-training and evaluation harness built
around a fixed motion prior**. It can validate an exact reference, attach a
frozen tracking reward and playback clock, train PPO with a bounded task
residual, evaluate the result independently from the generated reward,
diagnose failure, and keep or revert a bounded training change.

The trained PPO actor is closed-loop to its declared proprioceptive and task
observations. The **reference generator is not closed-loop**: it keeps playing
the same motion on the same timeline. The system has not demonstrated that it
can react to a displaced object, regenerate or select a useful reference,
integrate SONIC, or solve a difficult G1 motion from a reference.

> A fixed reference says what should happen on the nominal timeline.
> RewardSculptor trains and audits a policy around it. Nothing in the scoped
> reference path decides what should happen when the world no longer matches
> that timeline.

## What exists, and where it stops

| Component | What is implemented | Scientific boundary |
|---|---|---|
| Reference data plane | The library accepts bounded robot-scoped motion data and verifies immutable bytes, provenance, and root-frame evidence. It can retarget, crop, segment, and compose clips. | Ingestion is not launch compatibility. Exact ordered joints, cadence, target robot/runtime, and dynamics evidence must be earned later at Tier D and launch. Composition remains kinematic. |
| Single-reference runtime | One exact target schedule is played from elapsed episode time and held at its final target. Actor and critic each receive one scalar `reference_phase`. | The policy does not receive target joints, a future reference horizon, a reference token, or state-dependent phase correction from this mechanism. |
| Training objective | A frozen backbone tracks ordered joint position and root height, plus projected gravity/orientation when available. PPO trains with a capped editable task residual. Native world command, contact, safety, and curriculum terms may also supervise the task. | The agent cannot edit the reference tables, clock, or tracking kernels. Joint velocity is audited but is not a backbone tracking term. |
| Starting point and world | Starting policy weights, reference bytes, and training world are selected independently and pinned. Compatible data-only actor/critic weights can initialize training. | This is parameter initialization, not arbitrary uploaded controller execution, SONIC execution, or exact optimizer resume. |
| Variation and reset hooks | Typed physics randomization, task observations, contacts, and reset curricula exist. Consumer plumbing exists for full random-frame reference state initialization. | The selected-reference producer does not populate the full joint-position/velocity reset trajectories, so that DeepMimic-style path is not active end to end. |
| Evaluation loop | Content-addressed lineage, fail-closed launch checks, independent objective metrics, trajectory/contact/fall/posture/scene evidence, video, diagnosis, and bounded keep/revert iteration exist. | These establish identity and expose failure. They do not prove that the learning method is effective. |

The components and contracts are regression-tested. The live G1
reference-guided task path has not crossed Tier D and cannot currently launch:
no local G1 reference has a valid Tier-D certificate.

Outside this deliberately pre-OGMP scope, the repository has a validated
**linear elapsed-time phase-window automaton**. It is not a learned mode
latent, predicate/branching executor, state-conditioned oracle, or paper-faithful
OGMP implementation.

The system also lacks a first-class behavior-set × variation sampler with an
immutable per-episode behavior identity; online vision; SONIC model/token
execution; online reference generation or selection; demonstrated
task-space recovery; held-out multi-seed generalization; and hardware proof.

## Why the fixed-reference system is scientifically limited

1. **The displaced task state is only partially observable.** Phase and
   proprioception cannot directly localize an independently moved box before
   interaction, although contacts may provide indirect evidence. Exact object
   or visual feedback is required for a clean adaptation claim.
2. **One clip covers one narrow state tube.** It contains neither the
   distribution of valid task solutions nor the bridge states created by
   contact errors, object motion, or falls. Strong tracking may actively
   penalize the departure needed after the reference becomes invalid.
3. **Reward editing cannot repair missing state coverage.** RL may discover an
   unshown transition, but only if resets, interventions, or exploration visit
   the relevant states and the observations and objective distinguish useful
   behavior there.
4. **Tier D is necessary but narrow.** It binds exact bytes, cadence, contract,
   joint/root-height tracking, and a static baseline. It does not certify
   root-XY task execution, contact schedule, collision safety, or general
   dynamics feasibility; orientation is reported but is not a gate.
5. **The historical evidence is not a controlled method comparison.** Many
   iterations changed reward, world, reset distribution, seed, commands, and
   controller logic together. Parallel rollout environments from one
   checkpoint are not independent training seeds.

### Specific failure evidence

| Evidence | What happened | Correct interpretation |
|---|---|---|
| Reward-only tuck jump, first five iterations | Objective success stayed `0.0`. An alive bonus produced standing; the next reward was farmed while supine with raised feet; selection by generated mean return reverted to standing. | Temporal specification, exploration coverage, and selection by an independent objective all matter. This does not prove that any one reference method solves jumping. |
| Historical reward-only box weave | Iter 36 reported `64/64` own-course routes and two strict holds, but it predated the world-grid fix. Neighboring copies of the course overlapped, while the audit checked only intended per-environment boxes. | It is pre-fix diagnostic history, not physical acceptance or method evidence. No post-fix rerun restored the claim. |
| First completed four-rail Tier-D attempt | Joint MAE was `0.114937 rad`, root-Z RMSE `0.060565 m`, and duration coverage `0.996491`, but its static-baseline ratio was `0.862787` against the required `<=0.80`. The prominent leg moved while much of the support leg stayed near a safer static pose. | Average error can hide failure to perform the defining motion. Do not relax the gate to manufacture admission. |
| Original matched campaign | On `go1_trot`, the only clean, discriminating early benchmark, final IQM favored Eureka (`0.955`) over mission (`0.113`). Major later loop fixes have not been rerun in that full comparison. | RewardSculptor's advantage over reward-search baselines is still a hypothesis. |

The local receipts are retained in
[`RL_SCULPTOR_AUDIT.md`](internal/RL_SCULPTOR_AUDIT.md),
[`RESEARCH_GAP_ANALYSIS.md`](internal/RESEARCH_GAP_ANALYSIS.md),
[`HANDOFF.md`](../HANDOFF.md), and
[`CODEX_TO_CLAUDE_HANDOFF.md`](../CODEX_TO_CLAUDE_HANDOFF.md).

Earlier root-frame, playback-clock, and metric-control defects were repaired
after backward, phase-shifted, constant-mean, or zero-joint controls looked
acceptable. Those tests now protect software invariants; they are not learned
competence evidence.

## What can be claimed in August 2026

**Supported:** RewardSculptor implements content-addressed experiment
contracts, fail-closed launch, independent objective gates, and an inspectable
train/evaluate/diagnose loop for reference-guided RL.

**Not supported:** a difficult reference-guided G1 success; improvement over
expert or automated baselines; adaptive object recovery; generalization;
SONIC, OGMP, or visual-policy integration; or sim-to-real safety.

The novelty bar is high. [DeepMimic](https://arxiv.org/abs/1804.02717) already
combined imitation and task rewards;
[Eureka](https://arxiv.org/abs/2310.12931) automated reward search; reusable
motion-prior systems include [ASE](https://arxiv.org/abs/2205.01906),
[PULSE](https://arxiv.org/abs/2310.04582), and
[BeyondMimic](https://arxiv.org/abs/2508.08241). [OGMP](https://arxiv.org/abs/2403.04205)
uses a closed-loop oracle and bounded exploration, while
[Preferenced OGMP](https://arxiv.org/abs/2410.01030) demonstrates
feedback-conditioned G1 box manipulation and recovery-capable transitions.
[SONIC v4](https://arxiv.org/html/2511.07820v4) scales universal G1 tracking;
[ULTRA](https://arxiv.org/abs/2603.03279) studies sparse task conditioning and
OOD robustness; and [StableMimic](https://arxiv.org/abs/2608.02385) studies
recovery beyond a tracker's nominal distribution.

A defensible candidate contribution is therefore narrower: **under matched
budgets, an auditable agentic training loop repairs held-out, task-specific
off-reference failures better than expert and automated reward-search
baselines while preserving nominal motion.** That remains to be tested.

## Recommended next step, pending Lokesh's agreement

Run a **one-phase privileged-state counterfactual object-recovery benchmark**
before adding pixels or OGMP modes:

> Starting from one nominal G1 motion prior, can RewardSculptor train a policy
> to depart from an invalid reference, reacquire a displaced object, and finish
> the same task under held-out interventions?

**Gate 0 — make the experiment executable.** Obtain and independently pin
(a) the reference data, (b) the starting policy/controller architecture and
weights, and (c) the training world. Admit exact compatibility and pass a G1
Tier-D reference certificate. Decide whether SONIC is actually available; do
not use its name for a different tracker.

**Task and observations.** Use one nominal box-face-reorientation or
push-to-pose task. Move/rotate the box after commitment or first contact. Give
all claim-bearing policies the same proprioception, exact relative object
pose/velocity, goal pose, and phase slots. Pixels come later. A phase-only arm
with masked object slots is an observability lower bound, not a matched method
baseline.

**Prior authority.** The current frozen tracking backbone and capped residual
may prevent needed departure. In a pilot, choose one constant tracking weight
and explicit deviation budget that lets an expert baseline recover; then
freeze both for every claim-bearing arm. Do not let RewardSculptor edit that
authority and do not add state-triggered mode switching in this study.

**Matched arms.** Use one architecture and, if new observation slots are
needed, one identical predeclared zero-initialized transfer map:

1. object state, no reference, expert fixed reward/curriculum;
2. object state + fixed reference, expert fixed reward/curriculum;
3. the same inputs/reference + RewardSculptor iteration;
4. the same inputs/reference + Eureka-style population reward search, matched
   on simulator steps and LLM/token budget;
5. Preferenced OGMP as the direct method comparator. A separately declared
   executing privileged oracle/replanner may serve as a ceiling.

**Generalization contract.** Pre-register separate training, fresh-seed
in-distribution test, and held-out test partitions across intervention timing,
direction, translation/rotation severity, mass, and friction. Report strict
recovery success, latency, final object error, falls/forbidden contacts,
nominal no-intervention success, reference deviation, compute, and curves
versus intervention severity. Use a three-seed pilot only for debugging;
choose the claim-bearing sample size from pilot variance/power analysis, with
ten paired training seeds as the provisional minimum target and paired
intervals plus failure-rate uncertainty.

If the expert arm cannot recover, fix the observation/prior/training contract
before judging RewardSculptor. If it can recover but RewardSculptor cannot beat
expert or Eureka under matched budgets, the agentic loop is not yet the
contribution. A held-out gain without nominal regression would justify adding
perception and true OGMP modes.

## Four questions for Lokesh

1. Which exact task, reference bytes, starting policy/controller, and world
   will the lab provide, and is SONIC actually available for this study?
2. Confirm two separate contracts: (a) what the deployed policy outputs—SONIC
   tokens, motion commands, residual joint targets, or direct PD targets; and
   (b) what RewardSculptor may edit during training—reward, curriculum,
   reference weight, and/or data.
3. Is privileged object state acceptable before vision, and should
   Preferenced OGMP be the direct baseline (or is there a newer internal one)?
4. Is the intended claim the policy-training method, automatic recovery-data
   generation, or the full behavior-generator-to-policy stack?

The immediate deliverable is this boundary plus agreement on the benchmark
contract. Lokesh did not request a new GPU or hardware result before the next
meeting.
