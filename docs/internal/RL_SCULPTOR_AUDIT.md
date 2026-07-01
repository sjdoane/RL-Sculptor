# RL Sculptor — Convergence Audit & Improvement Loop

Living design + audit document for making the autonomous sculpt loop
reliably drive a policy to the target behavior instead of stalling at
zero fitness. Keeps the gap analysis, the prioritized plan, and a
per-loop change log so the work stays resumable across sessions.

Origin: the `tuck-jump` failure (objective metric 0.0 on every
iteration; policy stood still / farmed a tuck term while lying down;
loop regressed to v0). Constraints: never weaken the anti-gaming
firewall (`metric_calibration.py` / `metric_validate.py`); no further
firewall/AST hardening rabbit-holes — the problem here is the opposite
one (sparse fitness, weak reward generation, non-converging loop).

---

## 1. Research foundations (what successful systems do)

### Eureka (arXiv:2310.12931)
- **Population search, not greedy editing.** Each generation samples
  K=16 independent reward candidates from the LLM (environment source
  code as context, no few-shot prompt engineering). All are trained;
  the best by *task fitness* F (human-designed sparse task score, NOT
  the learned reward's return) survives.
- **Keep-the-best across generations (evolutionary elitism).** The
  next generation mutates the best-so-far reward; a generation that
  produces nothing better changes nothing. Fitness is monotone
  non-decreasing by construction.
- **Reward reflection.** The feedback given to the LLM is the *scalar
  time-series of each reward component* during training plus the task
  fitness — which terms moved, which were flat/saturated, at what
  magnitude. This is what lets the LLM fix dead or dominating terms.
- **Executability filter.** Candidates that fail to compile/run are
  discarded before training (in their setting ~weak models produced
  many; the filter is essential when K is small).
- Result: >52% of tasks beat human expert rewards; improvement is
  roughly monotone in generations *because of elitism*.

### DrEureka (sim-to-real follow-up)
- Adds a **safety instruction** to reward generation (penalize action
  rates, joint limits) — unregularized Eureka rewards are physically
  violent.
- Success criterion separated from shaping: a sharp binary success
  gate exists *beside* the dense shaping terms, not multiplied into
  everything.

### Text2Reward
- Generates **dense staged rewards** (distance-to-subgoal terms,
  potential-style shaping) rather than sparse gates; explicitly
  documents that sparse rewards fail to train from scratch.
- Iterative refinement driven by *human/behavioral feedback on the
  rollout*, not on the return.

### Motion-prior lines (AMP / ASE, PHC)
- When the goal is a *style/skill* ("explosive jump with a tuck"), a
  discriminator or reference motion carries most of the shaping load;
  hand/LLM-written terms only carry the task objective. Relevant here
  as a later option, not the first increment.

### Reference-state initialization (RSI, from DeepMimic)
- For skills with a hard-to-reach "interesting" phase (mid-air tuck),
  initializing some episodes *in that phase* gives the policy
  gradient signal about the apex/landing long before it can reach
  them from standing. Cheap to implement if the env exposes qpos/qvel
  setting.

### Curriculum / task-env alignment
- A flat-velocity walking env fights a jump task: termination on
  torso-pitch or low root height kills tuck attempts; a standing
  initial pose plus short episodes may not even contain one full
  jump. The task env must at minimum not terminate the target
  behavior.

### PPO per behavior class
- Explosive-motion tasks want higher entropy coefficient early,
  larger batch, and often shorter horizon (GAE lambda/γ tuned to the
  ~1 s jump timescale). Walking defaults (γ=0.99, long horizon) wash
  out a single-jump reward spike.

### Distilled checklist (what a converging system has)
1. Fitness that *ranks* partial progress (dense base) while success
   stays a sharp gate — partial credit for selection, no credit
   inflation for "done".
2. Population/best-of-N reward candidates per generation; train the
   promising ones; **keep the best ever seen**.
3. Reward reflection = per-component training statistics fed back to
   the generator (dead / saturated / dominating detection).
4. Selection signal = spec fitness, never mean_return.
5. Offline pre-screen (compile, run on a cached rollout, component
   sanity) before spending GPU minutes.
6. Env/task alignment: termination, init pose, episode length,
   observation adequacy audited against the goal.
7. Never-regress memory: the best reward+policy pair is never
   overwritten by a worse child.

---

## 2. Codebase map (as-is, verified 2026-07-01)

The loop already implements most of the Eureka checklist — the failure
was a degenerate interaction, not missing machinery.

- `sculptor/sculpt.py:1530` `sculpt_run()`; `:953` `_run_one_iter()`
  (train → rollout → fitness → realism audit → diagnose → apply_edits
  → commit).
- **Best-by-fitness selection exists** (§Ship 33): selection on the
  naturalness-gated `steer_fitness`; `best_reward_path` returns the
  best iter's TRAINED reward; post-loop repoints `current.py` to it.
- **Revert-on-regression exists** (§Ship 36 F1): when an iter doesn't
  set a new best, the next iter's train+edit base is reverted to the
  best-so-far reward, checkpoint cleared, diagnoser told
  `reverted_to_best`.
- **Early stop**: `fitness_patience` (no new best in N iters),
  `fitness_target`, and Goodhart-onset (`detect_goodhart_onset`,
  LAW 11 — fitness rising while naturalness falls).
- **Reward reflection exists**: `edit.py` feeds per-component training
  time-series (`reward_trajectory.json`, downsampled + max/mean/min)
  into the rewrite prompt; diagnoser gets fitness sub-components
  (§Ship 36 F2) + KG context.
- **Partition gate** (§Ship 54-pre #12): edits touching the metric's
  held-out observables are flagged; completion-gate hyperparameters
  may only tighten.
- **Eureka baseline** exists standalone (`eval/eureka.py:202
  run_eureka_job`, K candidates/generation, select on spec metric)
  but is not wired into `sculpt_run`.
- Metric pipeline: `metric_gen.py` (best-of-N sample → validate →
  panel review) → `metric_validate.py` (MUST-HAVE gates incl. AST
  safety, non-degeneracy archetypes) → `metric_calibration.py`
  (steer-rights firewall). Generated metric contract:
  `compute_spec(arrays, behavior, meta) -> {"spec_score": [0,1], ...}`
  with `spec_score = completion_gate × min(saturating channels)`
  (LAW 1/8, review-lens enforced). Extra keys pass through to the
  loop via `fitness_fn.detail()`.

## 3. Gap analysis

### 3.0 Evidence: the tuck-jump post-mortem (verified 2026-07-01)

Project at `~/.local/share/reward-sculptor/projects/tuck-jump/`.
5 iterations (iter_0..iter_4), spec_score = 0.0 on every one.

| Iter | Reward | mean_return | spec_score | What happened |
|---|---|---|---|---|
| 0 | v0 | 39.32 | 0.0 | constant `alive_bonus=1.0`; stands still |
| 1 | v1 | 2.75 | 0.0 | HACKED: supine, feet up, tuck_reward saturated 2.34/2.4; launch 0.02; fall_penalty DEAD (`info["fallen"]` never fires) |
| 2 | v0 | 41.44 | 0.0 | REGRESSION: reverted to v0 |
| 3 | v0 | 42.12 | 0.0 | still v0 |
| 4 | v0 | 44.07 | 0.0 | still v0; never jumps |

Confirmed mechanisms:

1. **All-or-nothing metric.** `metric.py` gen_001 core:
   `gate_bits = launched & tucked & returned & upright_end & upright_start`;
   `score = mean(gate * min(c_height, c_tuck, c_vertical, c_landing))`.
   Zero gradient below full success; min-composition bottlenecks even
   inside the gate. (Note: gate*min is exactly what the firewall's
   min-composition law wants for GRANTING success — the gap is that
   *selection* needs a dense sub-gate signal, not that the gate is wrong.)
2. **Loop selects on mean_return.** `metric_history.json` says
   `"primary_metric": "mean_return"`, history `[39.32, 2.75, 41.44, 42.12, 44.07]`.
   A constant alive bonus outscores every real jumping attempt, so v0
   is a fixed-point attractor.
3. **Greedy revert, no keep-best.** After v1's mean_return dropped,
   the loop reverted to v0 and stayed there. Corrective versions
   v2–v5 (airborne-gated tuck, height_fall_penalty, dense shaping)
   exist on disk but were **never deployed into training**.
4. **Reward ⟂ metric channels.** No reward version encodes the
   metric's channels (apex-height saturation, knee-flexion excess,
   drift Gaussian, landing min) or thresholds (0.12 m launch,
   0.55 rad tuck, gz>0.90). Partition-gate flags confirmed the reward
   read `base_height` while the metric derives height from
   `root_link_pos_w`.
5. **Env/diagnostics extras:** episodes always hit the 500-step cap
   (`terminated≈0` — no fall termination at all); lateral drift ~4.2–4.5 m
   per episode; the diagnoser *did* catch the v1 hack in prose but the
   loop's selection rule threw the fix away.

### 3.1 Gap → seam mapping

The deadlock, precisely: spec_score = 0.0 on every iter ⇒ iter-0's v0
pinned as best ⇒ strict-`>` comparison means every 0.0 TIE counts as
"no new best" ⇒ `revert_base` fired every iter (`sculpt.py` old
`iters_since_best >= 1` condition) ⇒ each iter re-trained v0 and its
corrective edit (v2..v5) was generated but NEVER trained. The
machinery worked as designed; the signal had no gradient and ties were
treated as regressions.

| # | Gap | Checklist | Seam | Status |
|---|---|---|---|---|
| 1 | No sub-success gradient in selection signal | 1 | metric contract + `sculpt_run` selection key | **FIXED loop-side + prompt-side (loop 1)** |
| 2 | Tie treated as regression → revert deadlock | 7 | `sculpt.py` revert decision | **FIXED (loop 1)** |
| 3 | No population/best-of-N reward search in main loop | 2 | `_run_one_iter` / `eureka.py` | open (loop 3) |
| 4 | Starter v0 is a constant alive-bonus (no gradient, high mean_return attractor) | 2/4 | project init template | open |
| 5 | Env/task misalignment: walking task (velocity commands), 500-step cap, steps_per_iter=1500–2000 likely far too small to learn a jump | 6/8 | config + adapter task selection | open (verify units first) |
| 6 | No offline pre-screen of reward candidates beyond compile/probe | 5 | `edit.py` post-flight | partially exists (validators); no cached-rollout scoring |
| 7 | No RSI / curriculum for hard-to-reach phases | 6 | adapter/env | open, later |

## 4. Prioritized plan

### 4.1 Dense selection fitness + tie-deadlock fix (loop 1 — DONE)
- Metrics additionally emit `progress_score` = min over the SAME
  saturating channels WITHOUT the completion gate, ramping from the
  sensor-noise floor (no hard amplitude floors). `spec_score`
  composition (gate × min) unchanged ⇒ firewall untouched
  (`metric_calibration.py` / `metric_validate.py` not modified).
- Loop selection key becomes LEXICOGRAPHIC `(steer_spec,
  steer_progress)`: spec decides, progress only breaks ties; both are
  naturalness-gated (an unnatural rollout cannot rank up via progress).
- Revert fires only on STRICT tuple regression; ties build forward
  (patience still counts them).
- Hack-resistance argument: progress is min-composed over
  firewall-vetted channels (raising it requires raising the WEAKEST
  requirement, so single-channel farming scores 0), it is never
  granted as success, never displayed as fitness, and is
  naturalness-gated. Goodhart-onset detection still watches the true
  spec/naturalness pair. Residual risk documented in §6.

### 4.2 tuck-jump gen_002 metric + E2E re-run (loop 2)
- Hand-author `metrics/gen_002/metric.py` = gen_001 with dense
  channels + `progress_score` (spec_score identical); pass it through
  `validate_generated_metric` before use.
- Verify training-scale units (steps_per_iter semantics for
  MjlabAdapter) and re-run tuck-jump short; success = progress > 0
  and climbing, robot visibly attempting a jump.

### 4.3 Population / best-of-N + offline pre-screen (loop 3)
- `candidates_per_iter` knob; K edit candidates per iter, offline
  pre-screen (compile/probe/partition + cached-rollout sanity), train
  survivors, select on the same lexicographic key; keep-best already
  handles memory.

### 4.4 Starter reward + env alignment (loop 4)
- Goal-conditioned v0 generation at project init (not constant
  alive-bonus); audit termination/init-pose/episode-length vs goal.

## 6. Residual risks / watch items
- `progress_score` is a new steering surface. Mitigations in place:
  min-composition, naturalness gating, lexicographic subordination to
  spec, Goodhart-onset stop. NOT yet covered: a metric whose dense
  channels are individually gameable in a correlated way — revisit
  after first real runs; do NOT relax the review lenses.
- Metric calibration (steer-rights) still calibrates spec_score only;
  progress_score earns no separate rights (it inherits the metric's).
  Acceptable while operator-granted via `--fitness-metric`.

## 5. Per-loop change log

Format per entry: date — what changed (files:lines) — why — evidence
(tests/smoke) — commit.

### 2026-07-01 — loop 1: dense progress channel + tie-deadlock fix
- **What**: `sculptor/sculpt.py` — `IterOutcome.progress/steer_progress`,
  `SculptRunResult.progress_history/best_progress`, phase-3b extraction
  of the metric's optional `progress_score` (clipped, advisory),
  naturalness-gated steer progress, LEXICOGRAPHIC best selection
  `(steer_spec, steer_progress)`, revert only on STRICT tuple
  regression (ties build forward); `sculptor/prompts/gen_objective_metric.md`
  rule 10 (require `progress_score` = min of dense pre-gate channels);
  `sculptor/prompts/review_objective_metric.md` + `metric_gen.py`
  consistency lens (progress_score allowlisted with its own reject
  conditions); 4 new tests in `tests/test_fitness_in_loop.py`.
- **Why**: tuck-jump deadlock (§3.0/§3.1) — all-zero metric + tie
  ⇒ revert ⇒ corrective edits never trained.
- **Firewall**: `metric_calibration.py` / `metric_validate.py`
  untouched; spec_score composition unchanged.
- **Verified**: full suite 973 passed / 1 skipped (was 969/1 before
  the 4 new tests).
