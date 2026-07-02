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
| 3 | No population/best-of-N reward search in main loop | 2 | `_run_one_iter` / `eureka.py` | DECIDED: not merged (see §4.3) |
| 4 | Starter v0 is a constant alive-bonus (no gradient, high mean_return attractor) | 2/4 | project init template | **FIXED (loop 4c)**: goal-conditioned seed at first run |
| 5 | Env/task misalignment: walking task (velocity commands + fell_over termination + command curriculum fight the jump). NOTE: steps_per_iter for MjlabAdapter = PPO max_iterations (2000 ≈ full-scale, ~1 h/iter on the 5070) — training scale was NOT the problem | 6/8 | config + adapter task selection | **FIXED (loop 4a)**: `env_profile="jump"` |
| 6 | No offline pre-screen of reward candidates beyond compile/probe | 5 | `edit.py` post-flight | **FIXED (loop 3)**: dead-reward variance probe |
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

### 4.3 Offline pre-screen (loop 3 — DONE); population search NOT merged
- **Done**: dead-reward variance probe in `edit.py` post-flight
  (`_probe_reward_variance`): 6 diverse deterministic input batteries
  (incl. asymmetric state≠next_state so difference-only shaping isn't
  false-rejected); a candidate whose total AND every component are
  constant across all probes is rejected with actionable feedback —
  the existing retry loop regenerates. Kills the v0-class constant
  reward before it burns a GPU hour training "stand still".
- **Decision — do NOT merge Eureka-style K-candidate population into
  `sculpt_run`**: (a) `eval/eureka.py` deliberately exists as the
  BASELINE condition of the project's A/B methodology (§Ship 28 / E3
  campaign — sculptor navigates by diagnosis+KG, Eureka by
  population); merging it into the treatment arm blurs the research
  design; (b) at ~1 h per mjlab training, K>1 multiplies GPU cost
  linearly for uncertain marginal gain now that keep-best + strict
  revert + dense ranking exist; (c) revert-on-regression already
  provides the elitism half of the evolutionary loop. Revisit only if
  the fixed loop still fails to converge, as an explicit opt-in knob
  and as its own reviewed increment.

### 4.4 Starter reward + env alignment (loop 4)
- Goal-conditioned v0 generation at project init (not constant
  alive-bonus); audit termination/init-pose/episode-length vs goal
  (fell_over termination + velocity-command curriculum actively fight
  a jump on Mjlab-Velocity-Flat-Unitree-G1).

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

### 2026-07-01 — loop 2: tuck-jump gen_002 metric + E2E re-run launched
- **What**: hand-authored
  `~/.local/share/reward-sculptor/projects/tuck-jump/metrics/gen_002/metric.py`
  (outside the repo): spec_score logic byte-identical to gen_001 +
  dense `progress_score` (min of noise-floor-ramped variants of the
  same 4 channels, no gate); meta.json provenance. Repointed the
  project's `rewards/current.py` at v5 — the corrective reward the
  old deadlock generated but never trained.
- **Verified**: `validate_generated_metric` ALL gates pass
  (behavior_goal-anchored); spec parity + progress on all 5 archived
  rollouts: standing iters progress ≈ 1e-4 (noise), iter_1 supine
  tuck-farm hack progress = **exactly 0.0** (d_tuck and landing
  channels 0 under min-composition) — the dense channel cannot be
  climbed by the hack that fooled the reward.
- **E2E**: launched 13:33 local, iters 5–8, `--fitness-mode steer
  --fitness-patience 4 --steps-per-iter 1500`, gen_002 metric; log
  `runs/sculpt_loop2_1333.log`. ~45 min training per iter.
- **Iter 5 datapoint (v5 trained for the first time)**: spec 0.0,
  progress ~0 — but the BEHAVIOR moved decisively off the old
  attractor: apex_gain 5.9 cm (was 2 cm standing), tuck_excess
  0.72 rad (real knee flexion near apex), lateral drift 0.29 m (was
  4.2 m walking). Bottleneck channel is c_returned ≈ 0: the policy
  ends ~0.7 m below start height (kneels/sits after the dip-hop;
  keyframes confirm). The dense channel breakdown now hands the
  diagnoser the exact failing requirement instead of a blank 0.0.
  Naturalness verdict ok (steer_factor 1.0).

- **E2E COMPLETE (iters 5–8, ~4 h GPU) — loop pathologies fixed,
  tuck-jump gate not yet cleared.** Full arc:
  | iter | reward trained | behavior | spec/progress | loop action |
  |---|---|---|---|---|
  | 5 | v5 (1st time ever) | dip-hop + tuck 0.72 rad + kneel-end | 0.0 / 5.3e-23 | best; edit → v6 (diagnosed reward_hacking+static_eq) |
  | 6 | v6 | INSTANT FALL (ep 16 steps) | 0.0 / — | STRICT regression → revert armed; edit → v7 |
  | 7 | v5 (reverted) | reproduces iter 5 (hop 5.2 cm, tuck 0.74) | 0.0 / ~0 | edit → v8 (5 edits, knows v6 failed) |
  | 8 | v8 | early fall (ep 18 steps) | 0.0 / — | regression; edit → v9 (untrained) |
  Post-loop: `best_reward_selected` = iter 7 → current.py = v5. ✔
  Every new edit TRAINED before judgment (old loop: never). ✔ Bad
  edits could not compound (strict revert). ✔ Keep-best held v5. ✔
  No v0 regression. ✘ spec/progress stayed 0: the min-composed
  progress is bottlenecked by return-to-stance, and both diagnoser
  edits over-corrected into instant-fall rewards. REMAINING GAP =
  edit quality on hard skills + env misalignment (velocity task
  termination/commands fight a standing jump) — §4.4.

- **FITNESS-CLIMB PROOF (hand-authored ground truth).** Scoring every
  rollout on the repo's own `spec_g1_jump` (saturating apex ×
  completed launch-and-land cycles × uprightness — the Eureka-style
  task fitness F):
  | iter | 0 | 1 | 2 | 3 | 4 | **5** | 6 | **7** | 8 |
  |---|---|---|---|---|---|---|---|---|---|
  | spec_g1_jump | 0.0 | 0.020* | 0.0 | 0.0 | 0.0 | **0.258** | 0.0 | **0.250** | 0.0 |
  (*iter 1 = supine hack, crushed by uprightness 0.074.) The fixed
  loop's kept-best behavior scores 13× the old run's best and comes
  with upright=1.0 + a completed launch/descent cycle + a real 0.72
  rad knee tuck. Caveat, stated honestly: part of the apex half-range
  credit comes from the stand→kneel height drop, so 0.26 is a partial
  jump attempt, not a clean tuck-jump. The generated gen_002 metric
  (stricter: min-composition incl. return-to-stance) correctly still
  reads 0 — do NOT soften it; the missing piece is a landing/recovery
  shaping term (§4.4), not more metric credit.

- **Cartpole proxy (negative result, documented).** A 4-iter run on
  `Mjlab-Cartpole-Balance` + built-in `cartpole_balance` spec:
  fitness = 1.0 from iter 0 even under the constant v0 reward — the
  env never terminates episodes early, so the episode-length spec is
  degenerate (always cap). Useless as a convergence demo (it did
  exercise tie-no-revert + keep-best at saturation correctly). Note
  for future eval work: the cartpole benchmark's spec cannot show a
  climb unless the env config terminates on pole fall.

### 2026-07-01 — loop 3: offline dead-reward pre-screen
- **What**: `sculptor/edit.py` `_probe_reward_variance` + call in
  `_post_validate` after the batched probe; 5 new tests + 1 fixture
  fix in `tests/test_edit.py`.
- **Why**: gap #6 — v0-class constant rewards reached the GPU;
  4 of 5 tuck-jump iterations trained a constant alive-bonus.
- **Verified**: full suite 978 passed / 1 skipped.

### 2026-07-01 — loop 4a: jump-class env profile (gap #5, env alignment)
- **What**: `sculptor/adapters/_mjlab_runner.py` `_apply_env_profile`
  (applied in BOTH `_cmd_train` and `_cmd_rollout`, before env build;
  new `--env-profile` arg on both subcommands);
  `sculptor/adapters/mjlab.py` `MjlabAdapter.env_profile` config field
  (validated at __init__: only ''/'default'/'jump'), threaded into the
  local + remote train/rollout command constructions; tuck-jump project
  `config.toml` now sets `env_profile = "jump"`. New
  `tests/test_env_profile.py` (10 tests).
- **Audit findings** (mjlab `velocity_env_cfg.py` + G1 flat variant,
  verified against the installed package):
  * Commands: `twist` UniformVelocityCommand, lin ±1 m/s, resampled
    every 3-8 s, only 10% standing envs → observation noise + the 0.3×
    `track_linear_velocity` floor (effective weight 0.6) paid for
    walking away (the 4.2-4.5 m/episode drift in iters 0-4).
  * Curriculum `command_vel` re-widens ranges to (-2,3) m/s at
    5k/10k steps — would silently undo zeroed commands.
  * Event `push_robot`: random base kick every 1-3 s (±0.4 m/s
    vertical, ±0.52 rad/s pitch/roll) — locomotion robustness DR that
    destroys launch/landing attempts.
  * Termination `fell_over` = bad_orientation(70°). KEY COUPLING: the
    reward-contract `fallen` signal fires at projected-gravity flip
    (90°), so at a 70° termination the signal was STRUCTURALLY DEAD in
    training — every fall_penalty in v1..v9 never fired (explains the
    iter-1 "fall_penalty DEAD" post-mortem note). Worse, termination is
    an ESCAPE: under a penalty-bearing reward the optimal policy falls
    fast to reset away the pain — the v6/v8 16-18-step collapse
    mechanism (§4.4 edit-quality gap, same root).
  * `episode_length_s` 20 s vs rollout eval horizon 10 s (500 steps).
  * Init pose (standing + small z jitter) is already jump-appropriate —
    left untouched.
- **Profile "jump"**: zero all twist ranges + rel_standing_envs=1.0 +
  heading_command=False; pop `command_vel` curriculum; pop `push_robot`;
  `fell_over` 70°→120° (fall now accrues its penalty; only an
  unrecoverable inversion terminates); episode_length_s=10.0. Verified
  live against the real `Mjlab-Velocity-Flat-Unitree-G1` cfg. Default
  '' profile is byte-identical CLI + cfg (existing projects untouched).
- **Follow-up (UI)**: `env_profile` is editable only via config.toml for
  now; a project-settings dropdown is a UI-side follow-up.
- **Verified**: tests/test_env_profile.py 10 passed; adapter regression
  set (test_mjlab_adapter, test_ground_texture, test_load_adapter,
  test_adapter_contract) 49 passed.

### 2026-07-01 — loop 4b: edit anti-collapse screen + hard-skill prompt
laws + hand-authored v10 landing reward (gap: edit quality)
- **Post-mortem of the v6/v8 collapses, measured (not guessed).**
  Replaying the actual reward modules over the archived rollouts:
  * iter_5 (v5, kept-best): the "kneel" is a FLOOR-SIT — base z ends at
    0.145 m with BOTH feet off the ground 497/500 frames; the
    sit-with-feet-up pose farms tuck_reward (mean 3.43/step) all
    episode. Hop apex 0.92 m from a 0.78 m start.
  * v6 on iter_5 frames: mean +0.52/step (seated_penalty −1.33 but the
    kneel STILL earns tuck +1.73 — v6's both-feet-airborne gate did not
    close the exploit: sitting has feet off the ground). v8: +0.28.
    ⇒ the planned "reject if mean-negative on the best rollout" screen
    would NOT have caught these two specific edits. Stated honestly.
  * v6 on its OWN collapsed rollout (mean ep len 15.7): fallen_frac =
    0.00 — the 70° fell_over terminated every fall BEFORE the 90°
    `fallen` flip, and tuck_reward paid +1.4/step DURING the topple
    (feet leave the ground). The collapse class is TERMINATION-
    LAUNDERED FARMING: dive → collect airborne credit in the 70-90°
    corridor → reset before any penalty accrues → repeat. v8 on its own
    rollout: mean −0.19 while alive (drift −0.39) — the plain
    suicide-by-termination variant. The loop-4a env profile (fell_over
    → 120°) closes the corridor for BOTH: a fall now persists, `fallen`
    fires, fall_penalty accrues, no quick reset.
- **What shipped**:
  1. `SculptorAdapter.build_reward_replay(rollout_dir)` (base: None) +
     `MjlabAdapter` implementation — reconstructs a batched
     (state, action, next_state, info) from `trajectory.npz` (exact:
     qpos/qvel/gravity/action/contacts/base_height/fallen; approximated:
     foot heights + finite-difference speeds; ≤4096 evenly-spaced
     frames, deterministic).
  2. `edit.py` `_replay_reward_summary` + `_screen_reward_on_replay` in
     `_post_validate` (both attempts): reject when the candidate's mean
     per-step total over NON-FALLEN frames < −0.05 on the protected
     rollout; reject message carries per-component means + the parent's
     baseline on the same frames (retry feedback names the offending
     term). Guards the net-negative-living class (true suicide
     attractor); does NOT claim to catch credit-starvation edits —
     those are covered by the env fix + prompt laws below. <32
     surviving frames → no evidence → pass.
  3. `sculpt.py`: replay source = THIS iter's rollout, or the
     best-so-far iter's rollout when this iter strictly regressed the
     lexicographic key (a collapsed rollout is not the behavior to
     protect); `prior_fitness` now carries best_progress +
     best_iter_dir.
  4. Prompt laws: `edit_rewriter.md` #10 net-positive-living
     (machine-checked; penalties sized vs earned credit in EVERY
     recoverable pose; exploits made relatively unprofitable, not
     absolutely negative) and #11 progress-preservation (never gate
     credit above the achieved level; ramp from achieved toward
     target; add the missing phase, don't re-threshold working ones).
     `diagnose_preliminary.md`: reward-suicide pattern (episode-length
     crash after penalty-adding edit ⇒ rebalance, not more shaping).
     `diagnose_grounded.md`: hard-skill edit policy (bottleneck-channel
     targeting, minimal edits on real progress, penalty sizing must
     name its offsetting positive term).
  5. **v10 hand-authored** (project rewards/, outside repo;
     current.py repointed v5 → v10): v5 + (a) smooth elevation gate on
     tuck (ramp 0.55→0.65 m; sit at 0.145 m earns 0, every hop apex
     0.90-0.94 m keeps full credit), (b) NEW `stance_recovery` =
     upright × both-feet-contact × height-ramp(0.20→0.70 m) — the
     dense return-to-stance credit the metric's bottleneck channel
     (c_returned ≈ 0) demands. No new penalties. Verified pose
     ordering: fallen ≪ sit-farm 0.10 < crouch 0.35 < stand 0.60 ≪
     flight-tuck 4.45/step; scalar/batched parity ≤1e-5 on all four;
     variance + batched + zero probes pass; anti-collapse screen pass
     (mean +0.29 on iter_5 replay, sit-farm tuck credit 3.43 → 0.20).
- **Firewall**: metric_calibration.py / metric_validate.py / gen_002
  spec composition untouched.
- **Verified**: full offline suite 996 passed / 1 skipped (pre-change
  baseline re-measured at 974 collected — the loop-3 "978" note counted
  a different collection; +22 new tests here: 10 env-profile [4a] + 12
  replay-screen [4b]).

### 2026-07-01/02 — loop 4d: E2E re-run under the §4.4 fixes (iters 10-14)

Run 1 (iters 10-12, `runs/sculpt_loop4_1959.log`, code b2846e3, jump
env profile active, v10 base, gen_002 steer, seed 42+i):

| iter | trained | behavior (measured) | gen_002 spec/progress | g1_jump GT |
|---|---|---|---|---|
| 10 | v10 | STANDS: c_returned 0.97, upright_end 1.0, drift 0.49 m (was 4.2) — but apex 3.8 cm, no tuck | 0.0 / **0.0196** (first nonzero progress ever) | 0.0005 |
| 11 | v11 (LLM edit) | TUMBLE-BOUNCE: apex 0.56 m, frac_launched 1.0/env, uprightness 0.154 | 0.0 / 0.0 (upright/settle min→0) | 0.042 |
| 12 | v10 (strict-revert ✓) | tumble again @seed 44: apex 0.49 m + REAL 0.96 rad tuck, uprightness 0.078 | 0.0 / 0.0 | 0.021 |

- Loop mechanics all held: first-ever nonzero dense progress ranked
  iter 10 best; iter 11's strict regression armed the revert; iter 12
  trained the reverted best; `best_reward_selected` → v10. The
  keyframes for iter 10 visibly confirm an upright stance start-to-end
  (no sit, no fall) — the c_returned bottleneck from iter 5 is CLOSED.
- The v11 LLM edit was exactly the shape the new laws demand (stance
  0.5→0.1, launch 5→8, ZERO new penalties, honest env-extension
  deferral) — the v6/v8 penalty-stacking class did not recur. The v13
  edit (from the reverted iter-12 diagnosis) re-gated tuck above stand
  height (0.75→0.85 m) — correct anti-crouch-farm, but blind to the
  tumble lesson (its diagnosis iter never saw v11's rollout).
- **Root structure now measured**: v10-family rewards oscillate between
  stand-farm and tumble-bounce because flight credit paid regardless of
  ORIENTATION. Every tuck-jump ingredient exists across rollouts
  (0.5 m launches, 0.96 rad tucks, perfect return-to-stance) — never
  simultaneously.
- **v14 hand-authored** (parent v13): launch AND tuck multiplied by
  uprightness = clamp(−proj_gravity_z, 0, 1) — tumbling flight earns
  ~0.1×, upright flight ~1×, dense in orientation; keeps v13 tuck
  gates + v11 weights. Verified: pose ordering fallen ≪ sit 0.10 <
  crouch 0.15 < stand 0.20 < tumble-flight 0.62 ≪ upright-flight-tuck
  5.04/step; scalar/batched parity; probes + anti-collapse replay
  screens PASS on all three archived behavior classes (iters 10/11/12).
- Run 2 (extension, iters 14-15, `runs/sculpt_loop4d_*.log`) trains v14.

### 2026-07-01 — loop 4c: goal-conditioned starter seed (gap #4)
- **What**: `sculptor/sculpt.py` — `_is_pristine_starter_reward`
  (REWARD_SPEC-signature detection of the untouched `sculpt init`
  template), `_seed_reward_prompt` (goal + iteration-0 design rules,
  clipped under apply_prompt_edit's 2000-char ceiling),
  `_maybe_seed_goal_reward` (generates v1 from the behavior goal via
  `apply_prompt_edit` — full post-flight stack incl. the variance
  pre-screen, which rejects a still-constant generation, and the
  bounded 1-call+1-retry budget). Called at the top of `sculpt_run`
  when start_iter==0, not dry-run, and behavior_goal is non-empty.
  New `tests/test_seed_reward.py` (8 tests, stub LLM client).
- **Design decision — seed at FIRST RUN, not at `sculpt init`**:
  project creation must stay instant and API-key-free (the UI backend
  scaffolds synchronously); the first run already spends LLM calls, so
  the seed rides on it. Every failure path (no API key, network,
  validation twice) logs + emits `seed_reward_failed` and proceeds on
  the template — the loop never blocks on the seed. Emits
  `seed_reward_started/generated` events for the UI stream.
- **Cost bound**: exactly one LLM call (+1 internal retry) once per
  project lifetime (only when v<latest> is still the pristine v0).
- **Verified**: 8 new tests (template detection incl. both shipped
  variants; generation happy path writes v1 + repoints current.py;
  non-pristine and later-version skips never touch the LLM; failure
  keeps the template; constant generation rejected on both attempts →
  exactly 2 stub calls). Full offline suite 1004 passed / 1 skipped.
