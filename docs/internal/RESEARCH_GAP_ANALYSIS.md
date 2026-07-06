# RL-Sculptor — Research Gap Analysis & Roadmap to a Shareable Artifact

2026-07-05. Full-system sweep + literature review. Purpose: identify what
this project is missing to be a legitimate research artifact worth
showing to a lab, excluding raw compute. Companion to
`RL_SCULPTOR_AUDIT.md` (convergence engineering) — this doc is about
*research credibility*.

---

## 1. TL;DR — the must-do list

1. **Run the metric-gaming base-rate study (§7.1).** No one has
   measured how often LLM-generated control metrics get gamed; the
   archived rollout classes + the trust pipeline + a small blind human
   rating make this a compute-free, publishable result. This is the
   paper; everything else supports it.
2. **Fix the loop's decision statistics (§7.2)** — K-seed evaluation +
   IQM per iteration, noise-band on ties, fresh-seed re-eval of the
   kept best. The core loop currently decides "better" at N=1.
3. **Repair the ground-truth specs (§7.3), add a held-out eval battery
   (§7.4)** — every claim inherits their defects.
4. **Then, and only then, spend the big run on E4-v2 (§7.5)** — the
   null/negative E4 verdict predates every convergence-loop mechanism;
   the current system has never been measured against its Eureka arm.
   Both outcomes are publishable once (1) exists.
5. **Decide the jump question in writing (§7.6)** — no published G1
   standing jump exists from pure reward shaping; rescope the headline
   to feasible tasks (kick/trot/stand-up), and treat the jump as the
   frontier case study or add the curriculum/reference machinery the
   literature requires.
6. **Land provenance + packaging (§7.8, §7.10)** — LLM call archive,
   MuJoCo Playground comparability, writeup + demo. The novelty is
   confirmed (§6); what's missing is evidence and packaging, not
   mechanism.

---

## 2. What the system is, in research terms

One paragraph per subsystem, with the honest research-facing reading.

**The loop** (`sculptor/sculpt.py:1804` `sculpt_run`, `:1022` one iter):
single-chain LLM reward-code iteration: train → rollout → objective
fitness → realism audit → two-stage diagnosis → constrained rewrite →
keep-best/revert. Selection key is lexicographic
`(steer_fitness, steer_progress)` where `steer_fitness =
naturalness_factor × spec_score` (LAW 7 gating, `adapters/realism.py:74`)
and `progress` is the dense sub-gate channel that only breaks ties
(`sculpt.py:2108-2125`). Revert restores the best *(reward, env)* PAIR
and invalidates the checkpoint. In literature terms: **greedy
(1+1)-evolution with elitism and an LLM mutation operator conditioned
on diagnosis**, vs Eureka's (1+λ) population search selected on human
fitness. The project deliberately keeps population search as the
*baseline arm* (`eval/eureka.py`), not merged into the treatment
(`RL_SCULPTOR_AUDIT.md` §4.3).

**Grounded diagnosis** (`sculptor/diagnose.py:676-943`): stage-1 failure
classification from metrics + keyframes + per-component reward
time-series (Eureka-style reflection) + realism audit + objective
progress; stage-2 edit proposal grounded in KG retrieval (techniques by
failure tag, semantic neighbors, past run cases) with hard
anti-hallucination citation gates (unknown arxiv_ids dropped
`diagnose.py:860-889`, hard-fail at edit `edit.py`). Case memory records
(edit identity, behavior signature, lexicographic verdict) per iteration
(`kg/cases.py`).

**Env co-iteration** (`sculptor/env_spec.py`): a validated, bounded env
spec with two scopes — `shared` (train+eval, frozen per run) and `train`
(diagnoser-iterable curriculum: RSI reset ranges, sunk termination,
entropy, friction, pushes). Schema-level scope split means eval
contamination is *structurally unrepresentable* (the pydantic Literal
over `ITERABLE_TRAIN_KEYS` can't name shared keys). Cross-field
invariant: airborne RSI requires min-base-height termination
(`env_spec.py:190-212`) — a measured lesson (iters 19-20) promoted into
the validator. In literature terms: **LLM-driven unsupervised
environment design (UED) restricted to a typed, bounded parameter
surface, with train/eval separation enforced by construction** — most
curriculum work enforces this by convention only.

**Metric trust pipeline** (`sculptor/eval/`): generated metrics must
earn steer-rights: L0 AST/contract/determinism/non-degeneracy gates
(`metric_validate.py`), independent review panel with lens
diversification and author-exclusion (`metric_gen.py:184-234`), L1
task-agnostic axioms (`metric_axioms.py:176-289`), L2 metric-blind
K=3 competence ladders with Spearman-midrank agreement
(`metric_calibration.py:427-531`), L3 adversarial gaming archetypes
(default OFF), L4 VLM grounding + L5 optimization-outcome audit
(cost-gated, not routine). Plus partition gate between reward edits and
metric observables (`partition_gate.py`), Goodhart-onset early stop
(`sculpt.py:1708-1744`). **This is the novel core**: every prior system
assumes a human-provided task fitness; this one tries to mint its own
and bound the trust you can place in it.

**Evidence state**: E4 campaign (frozen design, 5 paired seeds, matched
compute, KG/curriculum/Eureka ablations — `RewardSculptor/docs/campaign_plan.md`)
ran partially 2026-06-11 and paused: **null/negative** (go1_trot:
eureka 0.955 vs mission 0.113; g1_floss dead; g1_kick metric
non-monotonic). Root cause diagnosed (loop never saw objective fitness)
and fixed (Ship 33+), followed by the whole convergence-loop arc
(progress channel, seed reward, env-spec, RSI, case memory). **The
current system has never been re-measured against the baseline.** All
post-E4 validation is qualitative, single-task (G1 tuck-jump), single
training seed per iteration.

**What is already research-grade** (worth saying plainly):
- Campaign harness with IQM + stratified-bootstrap CIs over *paired*
  seeds and paired-difference reporting (`eval/harness.py:612-648`) —
  this is the rliable standard most papers skip.
- Structural train/eval separation in the env layer.
- The negative-results culture (`RL_SCULPTOR_AUDIT.md` records every
  failure with mechanism and evidence) — this reads as scientific
  maturity and should survive into any writeup.
- Offline pre-screens (variance probe, batched probe, replay
  anti-collapse screen) — a superset of Eureka's executability filter.
- Honest compute accounting baked into the campaign design.

---

## 3. Questioning the choices

Ordered by how much they matter for research credibility.

**3.1 The selection signal is statistically naked.** One rollout batch
per iteration, strict `>` on a noisy scalar, training seed varying per
iter (`seed 42+i`), no repeated-eval, no noise band. Documented
consequence: noise-floor progress values (1e-7..4e-6) mint "new bests"
and reset patience (`RL_SCULPTOR_AUDIT.md` §6); two best-selections in
E2E run 2 were decided below display precision. A reviewer will land on
this in minutes: *the loop's core decision — is this candidate better —
is made at N=1 with no significance criterion.* Fix is cheap relative
to training cost (§7.2).

**3.2 Reward-edit effect is confounded with seed and env-edit effects.**
Each iteration changes (a) the reward, (b) the env spec, and (c) the
training seed simultaneously; the audit itself flags that a regression
"cannot be attributed to one half" (E2E run 1, honest read #2). Case
memory then records causally-ambiguous lessons. Alternating or factorial
scheduling (reward-only iters vs env-only iters), or paired-seed
re-rolls for candidate comparisons, would make the loop's own learning
signal causal.

**3.3 Single-chain-by-decree may be handicapping the treatment arm.**
§4.3's decision not to merge population search protects the A/B framing
(diagnosis-guided vs population), but E4's only readable benchmark had
Eureka *winning 0.955 vs 0.113*. If the research question is "does
grounded diagnosis help," the cleaner design is: **both arms get
best-of-K selection; the treatment's K candidates come from diagnosis,
the baseline's from blind resampling.** Then the ablation isolates the
proposal distribution — the thing the KG story is actually about —
instead of bundling it with a search-topology handicap. A multi-fidelity
screen (train all K short, promote 1 to full budget — successive-halving
style) keeps GPU cost ≈ (1 + K·ε) per iteration.

**3.4 The demo task is the hardest possible setting for
reward-shaping-from-scratch.** Humanoid explosive skills in the
literature are trained with reference motions (DeepMimic/AMP/PHC line),
motion-prior discriminators, or heavy staged curricula — not pure
LLM-written shaping from a standing start (§5 for citations). The
composition gap (upright+flight+landing) that has held since loop 4 is
exactly what motion priors solve. Two coherent responses: (a) add a
train-scope reference/imitation channel (even one hand-authored or
trajectory-optimized jump clip) — env-spec already has the seam; or (b)
rescope the headline claims to a graded task ladder where from-scratch
shaping is viable (trot, kick, floss-class), and present the jump as the
honest frontier case study. Doing neither means the flagship demo keeps
stalling for a *known* literature reason.

**3.5 The trust pipeline's shipped teeth are L0-L2; the layers that
answer circularity are OFF.** L3 adversarial archetypes default OFF;
L4 VLM grounding and L5 optimization-outcome audit are cost-gated and
not routine. The in-run guards are Goodhart-onset (window=3, heuristic)
and the partition gate (non-blocking flags; hard-reject only on
same-name gate lowering — a renamed gate slips through,
`edit.py:1086-1110`). Meanwhile the loop trusts a calibrated metric for
the whole run with no re-calibration. And the deepest circularity is
documented but unresolved: if the metric author and the ladder author
share a blind spot (same model family, same goal parse), L2 agreement
passes and nothing downstream catches it. **The missing anchor is
external: human judgment.** No experiment currently ties trust scores to
human ratings of rollouts (§7.3 — this is also the paper's best
experiment).

**3.6 The ground-truth specs the whole methodology leans on are
themselves broken or gameable.** Measured: `spec_g1_jump` scored
sit-bobbing 0.215 (audit, loop 4d — "the hand-written ground truth is
itself gameable"); the kick metric is non-monotonic (extremal-Goodhart,
E4); cartpole's spec is degenerate (always 1.0); floss never trained.
Any campaign result — positive or negative — inherits these defects.
Benchmark repair is a precondition for every claim (§7.1).

**3.7 The KG is the system's identity but has no measured benefit.**
E4's KG ablation was ns (uninterpretable run, but still ns); case memory
landed after. The retrieval-grounded-diagnosis story is what makes the
project distinctive at demo time, and the anti-hallucination gates are
real engineering — but "grounded in a living knowledge graph" is
currently an *unvalidated feature*, not a result. Either E4-v2 shows
`mission − mission_no_kg > 0` somewhere, or the KG gets honestly
repositioned as provenance/explainability infrastructure (which is
still valuable and demo-able).

**3.8 Robot-agnostic claims vs G1-specific tables.** Joint-velocity
limits are substring-matched hardcoded tables (`realism.py:54-90`;
"knee" → 20 rad/s regardless of robot), goal-frame/behavior-family
resolution is keyword regex (`metric_validate.py:196-249`, known
unbounded), and the AST denylist is empirically unclosable (two new
escape classes in consecutive rounds; Fix B subprocess sandbox designed
but not landed). None of these is fatal; all of them are the kind of
thing a systems reviewer pokes. The honest framing is "G1-instantiated
with a documented generalization path," not "robot-agnostic."

**3.9 LLM provenance is not archived.** Structured outputs land in
`diagnosis.json`/changelogs, but raw prompts+responses per call are not
persisted (no `llm_calls`/api-log surface found). For a paper whose
mutation operator is a closed-source LLM, replayability of every
decision (prompt hash, model id, response) is the only reproducibility
story available. Cheap to add; impossible to retrofit onto past runs.

---

## 4. Literature landscape

Sources: web sweep 2026-07-05 (claims extracted with citations; the
harness's adversarial verify stage was cut short by a billing limit, so
each claim below was instead cross-checked against model knowledge —
items marked ⚠ are post-cutoff and taken on the source's word).

### 4.1 LLM reward generation, 2023 → 2026

| System | Search topology | Selection signal | Key numbers |
|---|---|---|---|
| **Eureka** (arXiv:2310.12931, ICLR'24) | population: K=16 i.i.d. candidates/iter × 5 iters (~80 full trainings/run) | **human-designed task fitness F** | beats human expert rewards on 83% of 29 envs, +52% avg; pen-spinning needed curriculum on top |
| **DrEureka** (RSS'24) | Eureka + safety instruction + reward-aware DR sampling | human F + real deploy | sim-to-real (yoga ball) |
| **Text2Reward** (ICLR'24) | single-chain, dense staged rewards | human feedback on rollouts | manipulation |
| **REvolve** (arXiv:2406.01309) | true evolutionary DB (islands, migration, crossover); argues Eureka is *actually greedy single-chain* (mutates only the best, discards rest) | **human Elo** from pairwise video judgments (10 evaluators × 20 pairs/gen) | driving |
| **ICPL** (arXiv:2410.17233) | candidate sets, in-context refinement | **human preferences** over videos, no programmatic fitness | ~30× fewer queries than PbRL baselines |
| **CARD** (arXiv:2410.14660) | single-chain, ~3 LLM queries | internal **Evaluator** (process/trajectory feedback) + **TPE gate**: candidate reward must rank success>failure trajectories *before any training is spent* | 14k vs Eureka's 663k tokens; matches/beats expert rewards 10/12 manipulation tasks |
| **ProgressCounts** (arXiv:2410.09187) | LLM progress functions + count-based intrinsic reward | task success of trained policy (assumes F) | SOTA Bi-DexHands; **4 policy samples/task vs Eureka's 48–80** |
| **RF-Agent** ⚠ (arXiv:2602.23876, 2026) | **MCTS over a tree of reward edits** (edit histories as sequential decisions) | human/env F + an ungated LLM self-verification score inside UCT | positions beyond population-vs-chain |

Cross-cutting reads for this project:

1. **Every system that works selects on an external fitness signal** —
   human-designed F (Eureka, ProgressCounts, RF-Agent), or literal
   humans (REvolve, ICPL). This simultaneously (a) confirms the novelty
   gap RL-Sculptor targets and (b) *explains the E4 null result
   mechanically*: the old loop navigated on LLM-authored criteria with
   no F while its Eureka arm selected on F directly. Ship 33 fixed
   this; the comparison has not been re-run since.
2. **The single-chain + cheap-gates design is directionally vindicated
   by CARD** — its TPE order-preservation gate is convergent evolution
   with RL-Sculptor's replay screens (`edit.py` variance probe /
   anti-collapse replay), and it beat Eureka's budget by 40× on
   manipulation. But note what CARD did *not* claim: legged/humanoid
   agile skills. Nothing single-chain has cracked hard-exploration
   skills.
3. **Eureka's own wins leaned on curriculum for the hard case** (pen
   spinning) — consistent with this project's env-spec turn.
4. RF-Agent's MCTS is the natural upgrade path for keep-best/revert:
   the loop currently maintains a *path* with reverts; a shallow *tree*
   over edit histories (revisit any ancestor, not just best) is the
   same machinery generalized. Also note RF-Agent's LLM
   self-verification score is exactly the kind of ungated judge signal
   RL-Sculptor's trust pipeline exists to discipline — a citable
   contrast.

### 4.2 LLM/automated environment + curriculum design

- **Eurekaverse** (arXiv:2411.01775, CoRL'24): LLM writes parkour env
  code; N=8 parallel agents, each with its own env library; evolve the
  envs that trained the best policy. **Evaluation hygiene**: fixed
  held-out benchmark of 20 human-designed tasks *never shown to the
  LLM*; intermediate keep-best selection uses a proxy (union of all
  generated envs), never the test set.
- **EnvGen** (arXiv:2403.12014): LLM-generated envs are train-only;
  every reported number comes from the original unmodified env.
- **OMNI-EPIC** (arXiv:2405.15568): LLM generates env AND reward as
  code (Gymnasium/PyBullet, ≤5 compile-repair iterations); deliberately
  separates the success-checking function from the training reward on
  the argument that a non-shaping checker is harder to game.
- **UED line** (PAIRED 2020, PLR 2021, ACCEL 2022): regret-based
  curriculum without LLMs — the principled ancestry to cite for env
  co-iteration.

Reads: RL-Sculptor's schema-level shared/train scope split is *stronger
engineering* than the convention-based separation in these papers (it's
unrepresentable, not just untested). What it's missing is the other
half of Eurekaverse's hygiene: **final evaluation on a battery the loop
never optimized against** — currently the frozen shared env + the
generated metric are both loop-visible. §7 makes the held-out eval
battery a deliverable.

### 4.3 Self-evaluation without human fitness (the novelty check)

- **OMNI-EPIC's success detectors agreed with humans only 72.7%** of
  the time (50-participant study) — the single best external number for
  "ungated generated metrics are unreliable," i.e., the problem this
  project's trust pipeline exists to solve. Cite it in any writeup.
- **CARD's Evaluator/TPE** is the nearest automated-evaluation
  precedent, but it is refinement feedback (which candidate to keep
  iterating), not a trust-gated *steering metric* with anti-gaming
  layers, and it presumes success/failure trajectory labels.
- **REvolve/ICPL** replace F with humans — they underline that the
  field considers evaluation the unautomatable part.
- **Machine-judge reliability, measured**:
  - VLM-RM (arXiv:2310.12921): CLIP-as-reward hits 100% human-judged
    success on 5/8 humanoid poses but 0% on arms-crossed / 64% on
    hands-on-hips — it fails *silently on fine-grained pose
    distinctions* (exactly the flight/landing compositions a jump
    metric needs), and the policy Goodharts the undiscriminating judge
    anyway. Discriminability scales sharply with VLM size.
  - RL-VLM-F (arXiv:2402.03681): raw VLM reward *scores* are "noisy and
    inconsistent"; pairwise preferences + analyze-then-label are the
    reliable query mode. **L4 should be pairwise, not a rating.**
  - SuccessVQA (arXiv:2303.07280): fine-tuned VLM success detectors hit
    83-85% balanced accuracy in-distribution and **collapse to ~59-62%
    out-of-distribution** — VLM verdicts on novel tasks are evidence,
    not ground truth. Supports the pipeline's VLM-as-one-layer stance.
  - Judges get gamed, with numbers (arXiv:2410.07137, ICLR'25): a
    **constant input-irrelevant response scores 86.5% LC win-rate on
    AlpacaEval 2.0** and tops Arena-Hard-Auto — GPT-4-as-judge, gamed
    by a null model. RL post-training raises verifier-exploit rates
    0.6%→13.9%, and *environment hardening cuts exploits 87.7%
    relative* (arXiv:2605.02964 ⚠) — the same shape as the layered-gate
    claim, in a different domain.
  - Pre-use validation of generated reward code exists piecemeal:
    CARD's TPE (preference-order check, no training needed), PCGRLLM's
    self-alignment probe (arXiv:2502.10906 ⚠). **No published
    end-to-end trust pipeline for generated evaluation metrics was
    found, and — critically — no paper measures the base rate at which
    LLM-designed control rewards/metrics get gamed. That measurement is
    itself publishable, and this project already owns the raw material
    (the archived sit-farm / sit-bob / tumble-bounce / dive-farm /
    stand-farm rollout classes plus honest attempts, all
    trajectory-verified).**

**Verdict on novelty**: as of this sweep, no published system grants a
self-generated metric the right to steer training through a layered
adversarial validation pipeline (axioms + blind ladders + gaming
archetypes + optimization audit). Nearest neighbors: CARD (TPE gate),
OMNI-EPIC (success/reward separation + human study), VLM-RM-line
(judge-as-reward, known-gameable). The claim is defensible — **but
novelty of mechanism is not evidence of efficacy**; the pipeline's own
headline experiment (does trust predict human judgment? does it block
gamed metrics that naive generation lets through?) has not been run.
That experiment is compute-light and is the paper (§8).

### 4.4 Methodology standards for credible small-scale RL

- **rliable** (arXiv:2108.13264, NeurIPS'21 Outstanding Paper): IQM as
  primary aggregate + stratified bootstrap CIs + performance profiles +
  probability-of-improvement. Percentile-bootstrap CIs are adequate
  from **N=10 runs** for IQM; 3-run point estimates flip conclusions.
  The campaign harness already implements IQM + stratified bootstrap
  over paired seeds (`eval/harness.py:612-648`) — ahead of most of the
  field; the E4 design's 5 paired seeds should go to 10 on the
  benchmarks that matter.
- **Empirical Design in RL** (JMLR 2024, Patterson et al.): 5 runs
  insufficient for strong claims; the seed is not a hyperparameter;
  **reporting a max-over-runs is an explicitly flagged pitfall** — and
  the loop's keep-best pair IS a max statistic. Final claims must
  re-evaluate the selected artifact on *fresh seeds never used for
  selection*. Tune baselines as hard as your method.
- **Eureka's own reporting** (verified from the paper): 5 runs,
  mean-of-MAX over 10 checkpoints, **no confidence intervals, no
  significance tests** — follow-up papers criticize exactly this.
  Matching Eureka's rigor is a reviewer target; exceeding it (10 seeds,
  IQM, CIs — machinery this repo already has) is a cheap
  differentiator.
- **Benchmarks a small project can adopt for comparability**:
  **MuJoCo Playground** (arXiv:2502.08844, DeepMind 2025) ships G1 and
  H1 tasks with *documented reward formulations*, RSL-RL support, and
  humanoid training in 15-30 min on one GPU — feasible on the 5070,
  and its shipped rewards double as honest "human expert" baselines
  for an Eureka-style comparison. LocoMuJoCo (arXiv:2311.02496) adds
  mocap tiers if a reference-motion channel ever lands. HumanoidBench
  (arXiv:2403.10506) is the documented-hard suite.
- **The credible-small-paper pattern** (CleanRL JMLR'22; "37
  Implementation Details of PPO" ICLR'22 blog): exhaustive verification
  of a *narrow* claim + full artifacts (code, configs, tracked runs,
  videos). No new algorithm required. This is precisely the shape the
  audit-doc culture here already has — it needs packaging, not
  reinvention.

### 4.5 Humanoid agile-skills SOTA (2024-2026)

**Every published G1 jump rides a retargeted reference motion.**
- **ASAP** (arXiv:2502.01143, RSS'25): G1 forward jump 0.85/1.5 m, side
  jump 1.3 m — motion-tracking policies from retargeted human video +
  delta-action sim2real model. Exploration comes free from the
  reference; the paper's problem is sim2real, not skill discovery.
- **Tracking line** (PHC 2305.06456, ExBody2 2412.13196, OmniH2O,
  HumanPlus, KungfuBot 2506.12851 / KungfuBot2, BeyondMimic): the whole
  high-capability humanoid stack is "reward = tracking error to a
  reference," precisely because it converts hard exploration into dense
  supervision. 2025 papers compete on *how well you track*, not on
  removing the reference.
- **DeepMimic ablations** (1804.02717): even WITH a tracking reward,
  removing RSI makes the backflip unlearnable (policy "cheats by
  hopping backwards"); early termination similarly load-bearing. The
  canonical citation for this project's RSI↔ET validator invariant —
  the loop re-derived a known result (iters 19-20), which is *good*
  (independent confirmation) and should be cited as such.
- **From-scratch existence proofs, and their machinery**:
  - *Humanoid Parkour Learning* (2406.10759, CoRL'24): H1 jumps onto
    0.42 m platforms / 0.8 m gaps with NO motion prior — but with
    fractal-noise terrain forcing foot-lift, a 10-obstacle
    auto-curriculum (promote >75%, demote <50%), two-stage training
    (walk first), teacher-student distillation. Jumps are
    *terrain-forced*, not standing vertical jumps.
  - *HoST* (2502.08378, RSS'25): G1 standing-up from scratch, no
    reference — multi-critic reward groups + terrain curriculum.
    Proves non-periodic whole-body G1 skills are learnable
    reward-only; stand-up is lower-explosiveness than jump.
- **HumanoidBench** (2403.10506): SOTA model-free/model-based RL below
  success threshold on most whole-body tasks; on *hurdle*, the policy
  "does not recognize the need to surpass the hurdle by jumping, which
  is a hard exploration problem" — the published statement that
  jumping IS the exploration failure mode for shaped-reward humanoid RL.
- **Quadruped contrast**: reference-free quadruped jumping exists
  (Atanassov 2401.16337: 90 cm real-hardware jumps; Rudin 2106.09357)
  but every one needed a staged curriculum, and quadrupeds have
  forgiving landing geometry. Humanoids add the crouch→extend→flight→
  catch composition — exactly the composition gap the E2E runs hit.

**Verdict**: an LLM-shaped from-scratch G1 standing jump is unpublished
territory. The literature predicts the reward function alone cannot
supply what's missing — you need curriculum + RSI/ET + (per HoST)
staged or multi-critic reward decomposition, or a reference motion.
Consequence for this project: the composition gap is not a bug in the
loop; it is the known frontier. §7 turns this into a decision rather
than a stall.

---

## 5. What the strong systems do that RL-Sculptor doesn't

| Practice | Who established it | Status here |
|---|---|---|
| Select candidates on an external fitness signal | Eureka, ProgressCounts, RF-Agent (F); REvolve, ICPL (humans) | fixed Ship 33 — **never re-measured vs baseline since** |
| ≥10 seeds, IQM + bootstrap CIs for claims | rliable | harness HAS it; **in-loop decisions are N=1, no noise band** |
| Selection seeds ≠ evaluation seeds (max-statistic discipline) | Patterson et al. | **missing** — kept-best pair is never re-evaluated on fresh seeds |
| Final eval on a battery the optimizer never saw | Eurekaverse (held-out 20 tasks), EnvGen | **missing** — frozen shared env + generated metric are both loop-visible |
| Cheap pre-training candidate gates | CARD (TPE) | HAVE — replay screens (`edit.py`); cite as convergent prior art |
| Curriculum forcing for explosive skills (staged training, terrain, auto-promote/demote, multi-critic) | Humanoid Parkour, HoST | partial (RSI/ET/entropy in env-spec); no staged training, no multi-critic, no forcing structure |
| Reference motions for humanoid agility | ASAP, tracking line (all published G1 jumps) | absent **by design** — must become an explicit, argued decision (§7.6) |
| Pairwise (not absolute) machine-judge queries | RL-VLM-F | L4 planned as rating — **switch to pairwise** |
| Comparable public benchmark + human-expert reward baseline | MuJoCo Playground G1 tasks | **missing** — adopt |
| Full-artifact release around a narrow verified claim | CleanRL, 37-Details | culture exists (audit doc); missing LLM provenance archive + packaging |

---

## 7. The roadmap

Ordered by research-value per unit effort. Compute-light first; the
one big spend (§7.5) comes last and everything above it is a
precondition. P0 = the paper depends on it; P1 = strengthens it;
WON'T = explicitly not.

**7.1 (P0, ~zero GPU) The metric-gaming base-rate study — this is the
paper.** Nobody has measured how often LLM-generated rewards/metrics
for control get gamed (§4.3). The raw material exists on disk:
trajectory-verified gamed-rollout classes (sit-farm, sit-bob,
tumble-bounce, dive-farm, stand-farm, termination-laundering) plus
honest attempts, across ~35 archived iterations. Design: (a) generate
N metrics per task naive (prompt-only, no gates) vs pipeline-gated;
(b) score every archived rollout class under both populations —
measure the fraction of naive metrics that score a gamed behavior
above a competent threshold, and which pipeline layer (L0/L1/L2/L3)
catches each; (c) **human anchor**: 2-4 blind raters rank ~100
rollouts (mix of classes); report correlation of naive metrics,
gated metrics, trust score, and the hand-written specs against human
judgment. OMNI-EPIC's 72.7% is the motivating prior; `spec_g1_jump`
scoring sit-bob 0.215 goes in as evidence that even *human-written*
metrics fail — which is the strongest honest argument for the
pipeline. Layer-ablation falls out for free (turn L1/L2/L3 off one at
a time). All offline: rollouts are archived, metrics are cheap LLM
calls, scoring is numpy.

**7.2 (P0, cheap) Fix the loop's decision statistics.**
**[SHIPPED 2026-07-05, commit 4184f7b — (a) `eval_seeds` multi-seed
median selection + min-naturalness gating, (b) `progress_epsilon`
noise band (default 1e-5), (c) paired eval seeds are deterministic per
iter, (d) `fresh_eval_seeds` end-of-run re-eval → `best_fitness_fresh`.
Training-seed pairing across iterations remains open.]**
(a) Evaluate each iteration's policy on K≥5 rollout seeds (rollout is
~2 min vs ~1 h training — the cost is trivial); select on IQM of the
lexicographic key. (b) Add the noise-band epsilon on progress
tie-breaks (audit §6 watch item — noise-floor bests decided two
selections in E2E run 2). (c) Pair training seeds across iterations so
candidate comparisons are seed-matched. (d) At end of run, re-evaluate
the kept-best pair on fresh seeds never used for selection (Patterson
max-statistic discipline) and report THAT number.

**7.3 (P0, cheap) Repair the ground-truth specs before any campaign.**
Known list (`project_e4_campaign_state`): monotone event-count kick
metric (prototype exists in the audit script); replace or drop floss;
cartpole needs a terminating env config; g1_jump needs contact-verified
launch + median-height artifact fix (the sit-bob 0.215 hole). Every
campaign claim inherits these.

**7.4 (P0, cheap) Held-out evaluation battery.** Final numbers come
from things the loop never optimized: repaired hand specs + a
perturbation suite (push/friction/mass deltas on the frozen shared
env) + pairwise VLM/human judgment. Mirrors Eurekaverse's hygiene
(select on proxy, report on held-out).

**7.5 (P0, the one big spend) E4-v2.** Re-run the frozen matrix on the
current loop — fitness-in-loop, progress channel, seed rewards,
env-spec co-iteration, case memory all postdate E4. Keep the honest
arms (mission / no_kg / no_curriculum / eureka / matched controls),
5→10 paired seeds where budget allows, report IQM + CIs + paired
differences. Two outcomes, both publishable given 7.1: win → headline;
lose → the paper is 7.1's metric-trust result plus an honest
"diagnosis-guided search does not yet beat population search at this
budget" — which CARD/ProgressCounts make respectable (single-chain
wins on manipulation, not yet on legged).

**7.6 (P0, decision not code) The jump question.**
**[PARTIAL 2026-07-05, commit 39a52a4 — option (c)-infrastructure:
`sculptor/reference.py` + `sculpt reference jump` derive validated
train-only RSI curricula from reference clips (procedural jump or
converted mocap); real retargeted G1 clips documented (Unitree LAFAN1
HF dataset, auth-gated; ASAP motions). The headline-rescope decision
(a) and multi-critic/staged training (b) remain open.]**
Pick one, in writing: (a) *rescope the headline* to published-feasible reward-only
tasks (kick, trot, HoST-style stand-up — G1 stand-up from scratch is
proven) and present the jump as the honest frontier case study; (b)
*add the curriculum machinery the literature demands* — staged
two-phase training (stand→jump), multi-critic reward groups (HoST),
terrain/goal forcing (Parkour) — as env-spec train-scope extensions
the diagnoser can iterate; (c) *add a reference-motion channel*
(LocoMuJoCo mocap tiers / one retargeted clip) as a train-only
imitation term, positioning against the ASAP line. Recommendation:
(a) for the campaign now, (b) as the jump arc afterward, (c) only if
the jump must be the demo. What is NOT viable is continuing to expect
pure shaping to compose the jump — §4.5 says that result doesn't
exist anywhere.

**7.7 (P1) Adopt MuJoCo Playground G1/Go1 tasks** as the comparable
benchmark surface; use its documented rewards as the "human expert"
baseline arm. Ships RSL-RL; trains in 15-30 min/policy on this class
of GPU.

**7.8 (P1) LLM provenance archive.** Persist every call's prompt,
response, model id, and hash per iteration dir. The mutation operator
is a closed-source model; replayability of decisions is the only
reproducibility story. Cannot be retrofitted onto past runs — land it
before E4-v2.

**7.9 (P1) Trust-pipeline hardening that the study (7.1) motivates:**
L3 adversarial archetypes ON by default (it's the layer that catches
what 7.1 measures); L4 switched to pairwise queries (RL-VLM-F);
Fix B subprocess sandbox lands as designed (closes the AST class
durably). Close the partition-gate rename bypass
(`edit.py:1086-1110`) with a value-similarity check.

**7.10 (P1) Packaging.** 6-8 page writeup (§8), project page with the
iteration-narrative videos (the E2E tables are a compelling story),
one-command repro, configs+seeds pinned. The audit doc, lightly
edited, is a genuinely unusual appendix: mechanism-level negative
results with evidence.

**COULD (post-paper):** MCTS/tree search over edit histories
(RF-Agent direction — keep-best/revert is already the degenerate
path case); opt-in best-of-K diagnosis-guided candidates with
short-train screening (fixes the §3.3 confound); KG-value measurement
stays inside E4-v2 (mission vs mission_no_kg).

**WON'T:** more AST-denylist whack-a-mole (Fix B supersedes; two
consecutive rounds proved the surface unbounded); reactive tuning of
`spec_g1_jump` (it's evidence now, 7.3 replaces it); new trust layers
before 7.1 anchors the existing ones to human judgment.

---

## 8. What "research success" looks like

**The claim structure** (one narrow, verified claim + honest context —
the CleanRL lesson):

- **Primary**: *Self-generated evaluation metrics are unreliable by
  default (OMNI-EPIC: 72.7% human agreement; ours: X% of naive metrics
  score gamed behavior as competent) and even human-written specs get
  gamed (spec_g1_jump 0.215 on sit-bobbing); a layered trust pipeline
  blocks Y% of gamed metrics at zero GPU cost, and its trust score
  correlates ρ=Z with blind human ranking.* Experiments: §7.1. No
  large training run required.
- **Secondary**: *With a trusted metric in the loop,
  diagnosis-grounded single-chain sculpting [does/does not] match
  population search at matched budget* (E4-v2, §7.5) and *validated
  (reward, env) pair co-iteration [does/does not] improve
  convergence* (matrix ablation). Honest either way; the loop's
  measured re-derivation of RSI↔ET (iters 19-20 vs DeepMimic) and the
  live env-lever demonstration (launch ×6, iter 24) are supporting
  evidence the mechanisms are real.
- **Explicitly out of scope**: solving the G1 standing jump (§4.5 —
  unpublished territory field-wide; state the decision from §7.6).

**Venue path**: arXiv + project page first; CoRL/ICRA workshop or RLC
as the realistic peer-review target; the full-system paper only after
E4-v2 has numbers.

**The USC-lab conversation** is a different artifact from the paper:
(1) the 15-minute live demo — UI, a running loop, the KG provenance
chain from a diagnosed failure to a cited technique to a reward diff;
(2) the gaming base-rate table (7.1) as the "here's my result" slide;
(3) the audit doc as the "here's how I work" evidence — mechanism-level
honesty about what failed and why is rarer than positive results in an
undergrad portfolio, and labs know it. The pitch is not "I solved
reward design"; it is "I built the instrument, found the open problem
everyone else routes around (evaluation trust), measured it, and my
negative results are load-bearing."

---

## 6. Is the novel claim actually novel?

Covered in §4.3 — yes, with CARD / OMNI-EPIC / VLM-judge lines as the
nearest neighbors to cite and differentiate. The differentiation
sentence for a writeup: *prior work either assumes a ground-truth
fitness (Eureka line), substitutes humans (REvolve, ICPL), or generates
success checkers without validating them (OMNI-EPIC, 72.7% human
agreement); we treat the evaluation metric itself as an untrusted
artifact that must earn steering authority through adversarial
validation.*

---

## 7. The roadmap

(filled after research workflow — ordered SHOULD / COULD / WON'T)

---

## 8. What "research success" looks like

(filled after research workflow — the paper skeleton, the experiments,
the artifact list)
