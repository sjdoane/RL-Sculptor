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
| 7 | No RSI / curriculum for hard-to-reach phases | 6 | adapter/env (jump profile reset events) | **open — now THE blocker** (loop-4d E2E: all degenerate attractors priced out, but PPO never discovers the coordinated upright launch under live fall risk; warm-start from the tumble basin adapts into sit-bobbing instead) |

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
- **Noise-floor progress mints new bests (observed E2E run 1,
  2026-07-04)**: progress_score values at the sensor-noise ramp's
  bottom (1e-7..4e-6) differ seed-to-seed among behaviorally-identical
  re-rolls, so "new bests" at noise level reset fitness_patience and
  plateau early-stop may never fire. Harmless for fixed iteration
  budgets; consider an epsilon on the lexicographic tie-break (loop-1
  surface, NOT the env layer) if patience-based stopping matters.
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

### 2026-07-19 — FUTURE DIRECTION (not implemented): grounded, unbiased
video judge (Sam's ask, informed by Prompt2Policy's implementation)

Sam's framing after watching the Prompt2Policy demo: the video judge is
imperative to judging success; ours should "use the systems to best let
it fully and completely understand the actions of the robot and be able
to score in an unbiased way." Design direction, recorded for a later
increment. Evidence base: KRAFTON Prompt2Policy source
(`src/p2p/agents/judge_agent.py`, `prompts/judge_agent.py`,
`inference/vlm.py`, `analysis/guardrails.py` — MIT).

**What Prompt2Policy actually does (verified in source):**
- **Two-turn judging as agreement-bias mitigation**: Turn 1 makes the
  VLM pre-commit 3-5 observable visual success criteria BEFORE seeing
  any rollout (grounded on the intent + the episode's first frame +
  injected camera-orientation and joint-rotation conventions), then a
  self-review loop (≤3 rounds) refines them; criteria are cached per
  session and frozen for all Turn 2 scoring calls.
- **Physics cross-reference**: prompts explicitly warn that VLMs cannot
  reliably judge rotation direction from video; a final LLM synthesis
  stage merges the VLM critique with the code judge and per-term reward
  telemetry into pass/fail.
- **Distributional inputs**: judged videos are percentile-selected
  (p10 / median / p90) from parallel evals, not cherry-picked; a
  StreamingJudge overlaps VLM inference with PPO training.
- **Guardrails**: single-reward-term dominance (>90% of magnitude →
  reward-hacking warning) and plateau detection feed the revise loop.

**How OUR judge goes further — grounded in system assets:**
1. **Criteria from the authored TaskSpec, not VLM priors**: generate the
   Turn-1 rubric from the goal type, success predicates, zones, and
   contact spec of the promoted tuple; freeze the rubric per evaluation
   lineage (mirrors the metric-freeze discipline).
2. **Full action understanding**: render eval rollouts from the
   materialized scene at multiple camera angles with a synchronized
   telemetry strip (per-frame success-predicate state, contact events,
   waypoint/traversal progress from the world channel recorder) — the
   VLM sees ground-truth physics beside pixels instead of guessing at
   them, and the synthesis stage cross-references trajectory.npz +
   metric components rather than raw reward terms.
3. **Reference anchoring**: pairwise comparison against reference/demo
   keyframes and against the best-so-far rollout, order-randomized —
   relative judgments over absolute scores.
4. **Bias controls**: blind judging (no reward code, edit history,
   iteration index, or prior scores in context); ensemble over seeds and
   judge samples with the score DISTRIBUTION reported, not a point
   estimate; periodic calibration of judge scores against the calibrated
   objective metric (Spearman, reusing metric_calibration) — the judge
   is trusted only within its measured agreement band.
5. **Firewall (non-negotiable)**: the video judge is ADVISORY — a
   diagnoser input and a human-facing report, never fitness, never
   keep/revert. Judge-vs-metric disagreement is surfaced as a diagnoser
   signal (possible metric gaming OR judge bias), never auto-resolved.
   Verdicts land in run-case memory as observations so judge drift is
   itself measurable.

**Cheap near-term adoptions (independent of the VLM judge):** the
single-term-dominance guardrail over our component breakdowns; plateau
detection over training feedback; criteria-self-review as a pattern for
our metric generator's review stage.

### 2026-07-18 — external corpus additions from USC robotics contact
(Lokesh Krishna) + roadmap candidates

Two S-tier works shared by Lokesh Krishna (USC robotics lab) ingested
into the shared graph with full LLM extraction (doctor clean after:
2,006 nodes / 2,140 edges / 1,365 embeddings, all pools fresh):

- `paper:2606.19980` — **ENPIRE: Agentic Robot Policy Self-Improvement
  in the Real World** (NVIDIA GEAR/CMU/Berkeley). Coding agents run a
  reset→execute→verify→refine loop on physical robots; an Evolution
  module compares algorithmic branches, reuses successful recipes, and
  prunes failed hypotheses; 99% pass@8 on real manipulation. 24 nodes /
  17 edges extracted. This is the hardware analog of the sculpt loop —
  primary reference for a future hardware deployment path.
- `paper:krafton-p2p-2026` — **Prompt-to-Policy: Agentic Engineering
  for RL** (KRAFTON; blog + MIT-licensed repo, no arxiv — ingested from
  blog + README text sidecar, first non-arxiv Paper node). Pipeline:
  Intent Elicitor → Reward Author + Judge Author → Code Review →
  multi-seed×config PPO → Code Judge ∥ VLM video judge → Synthesizer.
  Independent convergence on the RewardSculptor design; their Intent
  Elicitor stage independently validates our clarifier protocol.
  20 nodes / 16 edges extracted.

Candidate roadmap items harvested (NOT implemented — recorded for
prioritization; none may weaken the metric firewall):

1. **VLM rollout-video judge as an advisory diagnoser input** (P2P):
   a video-level "does this look like the requested behavior" signal
   feeding the diagnoser's edit reasoning only — never fitness, never
   keep/revert, which stay with the calibrated metric firewall.
2. **Multi-seed × multi-config training per iteration** (P2P): variance
   over seeds separates reward-design failures from training noise
   before the diagnoser attributes blame.
3. **Branch-parallel edit exploration with recipe reuse** (ENPIRE
   Evolution): explore 2-3 competing edit hypotheses per iteration and
   prune, rather than the current linear keep-best; successful edit
   recipes are already partially captured by run-case memory —
   promotion into reusable "recipes" is the delta.
4. **Stage-wise behavioral milestones + pass@k** (ENPIRE): first-move /
   range-satisfaction / first-success milestones as diagnoser
   observables, and pass@k emergent-retry as an eval-only statistic.

Evidence: `.lokesh_ingest.py` run (transient, deleted after landing);
`sculpt kg doctor` clean. Commit: this docs commit.

### 2026-07-18 — KG Phase-0 audit + hardening (substrate for the
env-authoring milestone)

Full audit of sculptor/kg/* (ingestion, extraction, embeddings,
semantic + tag retrieval, run-case memory, shared-DB path resolution)
before the env-authoring research corpus lands. Live shared-graph
integrity measured first (1,662 nodes / 1,908 edges / 1,024 embeddings:
0 dangling edges, 0 orphan embeddings, 0 stubs, 100 % of the three
semantic pools embedded — but 20 papers with dead `/tmp/pdfs/*`
full_text_path sidecars). Confirmed defects fixed:

1. **Extraction merge data-loss (HIGH)**: `extract._materialize`
   rebuilt existing Technique/FailureMode/RewardComponent/Environment
   nodes from the payload subset — re-extraction reset
   `useful_citations`/`outcome_stats` (whole-GPU-run learning signals)
   to 0/{} and clobbered provenance. Now `dataclasses.replace` merges:
   unspoken fields survive; provenance upgrades by trust tier only
   (new `schema.merge_provenance`; diagnoser-flagged FailureMode stubs
   created as `llm_extraction`, upgraded to `paper_claim` when a paper
   attests them, never downgraded).
2. **Stale embeddings (MED)**: `has_embedding`-only pools served old
   vectors forever after text changed (extraction enriches
   descriptions; resumed runs re-attribute case verdicts, and
   `_case_text` embeds the verdict). `node_embeddings` gains
   `text_hash` (additive in-place migration); all four semantic pools
   (Paper, Technique, FailureMode, RunCase) share one staleness-aware
   `ensure_embeddings` (missing/stale
   re-embed; pre-hash rows trusted once + hash-stamped).
3. **Forward-compat (MED)**: `row_to_node` now drops unknown data
   keys (newer-schema rows no longer TypeError older readers; unknown
   KINDS still raise); `neighbors()`/`all_edges()` skip
   unknown-relation rows with a once-per-relation warning instead of
   aborting the whole scan.
4. **Global socket timeout (MED)**: ingest/research wrapped arxiv
   calls in `socket.setdefaulttimeout` — process-global; inside the
   live uvicorn backend any socket created by another thread in the
   window inherited a 30 s timeout. New `make_arxiv_client` binds the
   timeout to the arxiv client's own requests session.
5. **UI-side graph fragmentation (MED)**: backend
   `project_kg_db_path` still preferred a legacy `<project>/kg/graph.db`
   — the exact silo bug loop 5a removed sculptor-side, and run_manager
   exports the result as `SCULPTOR_KG_PATH` to training runs. Now
   always the shared path (once-per-file warning → `sculpt kg merge`);
   backend `pdfs_dir` follows the resolved DB (has_pdf flags were
   always False against the shared graph).
6. **`sculpt kg doctor`** (new `sculptor/kg/doctor.py`): pre/post-fix
   integrity report (dangling edges, orphan embeddings, unknown
   kinds/relations, stub titles, dead text paths, missing/stale/
   unhashed embeddings) + `--fix` mechanical repairs +
   `--reembed-all`; `heal_dead_text_paths` re-ingests the /tmp-era
   papers. Doc/help staleness fixed (CLI `--store` text, store.py
  header, docs/knowledge_graph.md legacy-path + research sections).
7. **Store consolidation safety**: merging now copies only text-hashed
   vectors whose nodes are safe to copy, skips unknowable unhashed legacy
   vectors, and unions paper-level `supports` when the same claim exists in
   multiple databases. Concurrent PDF downloads use unique same-directory
   partial files instead of one fixed temporary name.
8. **Evidence/citation alignment**: shared Technique→FailureMode and
   Technique→RewardComponent claims retain every corroborating paper. Both
   tag and semantic retrieval choose evidence from the paper they actually
   cite; free-form tags normalize across hierarchical campaign aliases.
9. **Case-memory accounting**: case retrieval includes project and robot
   scope; persisted references make re-recording delta-based and idempotent.
   A one-time `attribution_version` migration prevents the 194 legacy cases
   from double-crediting already-counted citations. Reference attribution
   now records the reward that trained the measured policy, not the edit
   written afterward.
10. **Campaign controls**: structured Paper metadata (`tier`, `tags`,
    `rationale`, `source_url`) survives idempotent ingest. Exact seed/tier/tag
    extraction normalizes arXiv versions and fails loudly when selected
    papers are absent. Paper nodes now have their own semantic retrieval API.
11. **Final live verification**: after exact S-tier extraction and a full
    rebuild, the shared graph has 1,962 nodes, 2,107 edges, and 1,335
    embeddings. Doctor reports zero dangling/orphan/unknown-schema issues,
    zero stubs or dead text paths, and zero missing/stale/unhashed vectors.
    The remaining 49 unextracted A/B papers are the deliberate metadata tier
    of the hybrid policy. Independent Phase-0 red-team: pass. Repository
    validation with `MUJOCO_GL=egl`: RewardSculptor 1,994 passed / 1 optional
    JAX test skipped; UI backend 531 passed.

### 2026-07-18 — environment-authoring research corpus

- Added `RewardSculptor/kg_seeds_env_authoring_2026-07.yml`: 89 verified
  papers with applicability rationales and structured domain tags; tiers are
  S=33, A=41, B=15. All 89 PDFs, nonempty text sidecars, metadata records,
  and source URLs are present in the shared graph.
- Hybrid extraction is complete by policy: all 33 S papers are extracted
  (40/89 overall including seven previously extracted A/B papers). Every S
  paper produced graph artifacts; 123 INTRODUCES and 111 EVALUATES_ON edges
  have nonempty evidence. The 49 A/B holdbacks remain searchable as embedded
  Paper nodes and can be promoted selectively without another corpus search.
- Real MiniLM smoke queries retrieved the intended literature across LLM
  terrain generation, object goals, simulator frameworks, and UED/regret.
  Exact campaign selection is idempotent and all title/metadata checks pass.

### 2026-07-18 — prompt-driven environment-authoring architecture

- Added `docs/internal/ENV_AUTHORING_ARCHITECTURE.md`, the normative design
  for mission prompting, capability negotiation, clarification/default
  provenance, versioned WorldSpec/TaskSpec artifacts, frozen evaluation
  manifests, simulator validation gates, channel-catalog threading, honest
  reward/metric access, diagnoser feedback, curriculum policy, and KG memory.
- The design explicitly covers uneven-terrain locomotion, ball-to-goal
  interaction, and gripper-capable humanoid tasks without task-name
  hardcoding. Train variation is separated into compiled generator
  parameters, expandable model fields, and reset state; structural mutations
  require a new compiled lineage.
- Added `RewardSculptor/scripts/world_schema_poc.py`. Against installed
  mjlab 1.3.0 / MuJoCo 3.7.0 it compiles the normative schema into a scene
  with 5 bodies, 17 geoms, and 3 heightfields, validates a clean settle, and
  rejects a deliberately overlapping spawn at 13.5 cm penetration. It is a
  schema/compiler smoke test, not the production implementation.
- Independent architecture red-team: all blockers closed. Implementation
  remains deliberately staged behind the documented P1–P5 gates.

### 2026-07-05 — research-driven upgrades (gap analysis → three landed
increments)

- **Gap analysis**: `docs/internal/RESEARCH_GAP_ANALYSIS.md` (68fc775)
  — full literature sweep (Eureka line successors, env/curriculum
  design, machine-judge reliability, methodology standards, humanoid
  agile SOTA) + roadmap §7. Key verdicts: trust-pipeline novelty
  confirmed (nearest: CARD TPE, OMNI-EPIC 72.7% human agreement); NO
  published from-scratch G1 standing jump exists (all ride retargeted
  references); the compute-free missing paper is the metric-gaming
  base-rate study (§7.1).
- **Selection statistics** (4184f7b, roadmap §7.2): `eval_seeds` K-seed
  median selection (per-seed dispersion to the diagnoser; naturalness =
  MIN over seeds), `progress_epsilon` noise band (default 1e-5 —
  noise-floor ticks are ties, not bests/regressions; the §6 watch item
  closed), `fresh_eval_seeds` end-of-run re-roll of the kept best on
  held-out seeds → `best_fitness_fresh` (report-of-max discipline);
  rollout `--seed` threaded local+remote (0 = legacy). +8 tests.
- **Hack-income regression screen** (787ce12; CARD arXiv:2410.14660
  TPE adapted): candidates are replayed on archived reward_hacking
  rollouts and rejected if they pay a caught exploit more than the
  parent (abs 0.05/rel 10% tol) — caught hacks monotonically lose
  income. `hack_income_screen=false` disables. +7 tests.
- **Reference trajectories** (39a52a4, roadmap §7.6 infra):
  `sculptor/reference.py` + `sculpt reference jump` — validated clip
  format, analytic procedural jump, measured phase keyframes for
  prompts, and DeepMimic-RSI derivation onto the EXISTING env-spec
  TRAIN surface (bounds-clamped, RSI↔ET pairing always emitted;
  0.64×stand reproduces the measured-good 0.5 m sunk on G1).
  Train-only by construction — evaluation untouched. +9 tests.
- **KG seeds**: `kg_seeds_research_2026-07.yml` — 17 sweep papers with
  applicability rationales (extraction deliberate, costs LLM+PDF).
- Suite: 1113 passed / 1 skipped after all three increments.

### 2026-07-04 — env generalization 1/4: general per-project env spec
(overnight loop, mandate: environment adapts itself to each prompt —
no hand-picked presets)

- **Plan for the arc (increments 1-4)**: (1) declarative validated
  env-spec schema + general runner applier, jump profile becomes a
  preset INSTANCE; (2) goal-conditioned spec generation at first run
  (mirrors the loop-4c seed-reward pattern: bounded LLM calls, full
  validation, safe fallback); (3) diagnoser proposes train-section
  spec deltas between iterations (rides the existing grounded-diagnose
  call; versioned env/v<N>.json + current.json, participates in
  keep-best/revert); (4) tuck-jump migrated to the general mechanism +
  E2E on GPU. Firm constraints honored throughout: firewall untouched
  (metric_calibration.py / metric_validate.py / gate×min composition);
  train-only curricula never touch rollout evaluation.
- **What (increment 1)**: new `sculptor/env_spec.py` — schema v1 with
  TWO SCOPES: `shared` (applied to train AND rollout, frozen per run:
  command zeroing, orientation-termination angle, episode length, push
  events) and `train` (train-only curricula, the diagnoser-iterable
  surface: RSI reset height/velocity offsets, joint-reset ranges,
  friction randomization, sunk-height termination, entropy scale, push
  overrides). Strict validation: unknown keys REJECTED (a typo fails
  loudly, not silently no-ops), hard per-field bounds, well-ordered
  ranges, all violations reported at once (complete generator
  feedback). `ITERABLE_TRAIN_KEYS` = the diagnoser's whole editable
  surface — shared keys are structurally not iterable, which IS the
  metric-comparability guarantee. `_mjlab_runner._apply_env_spec` +
  `_apply_rl_spec`: general appliers replacing the hardcoded jump
  body; `_apply_env_profile`/`_apply_rl_profile` remain as preset
  resolvers routing "jump" → `jump_preset_spec()` → the general path
  (byte-equivalent mutations, parity-tested). New `--env-spec <path>`
  on both runner subcommands (wins over `--env-profile`); adapter
  field `env_spec_path` (validated fail-fast at __init__, threaded
  local + remote via RunnerJob input_paths for pod sync);
  `load_adapter` injects `env/current.json` by convention when present
  (signature-introspected like the `[remote]` plumb). Values are
  robot-agnostic by construction: reset offsets are ADDED to the
  robot's default reset state (mjlab reset_root_state_uniform
  semantics); absolute thresholds (sunk height) are per-project DATA
  chosen by the generator, bounded by the validator.
- **Firewall**: metric_calibration.py / metric_validate.py untouched.
  Rollout applies the shared section ONLY — RSI/sunk/DR structurally
  cannot reach evaluation (schema-level section split + applier
  train=False gate + tests pinning both).
- **Verified**: new tests/test_env_spec.py (30) + existing
  test_env_profile.py (10, unmodified — pins jump parity through the
  general path). Full suite 1051 passed / 1 skipped (was 1021/1
  baseline this session). Commit 73c3adc.

### 2026-07-04 — env generalization 2/4: goal-conditioned env-spec
generation at first run

- **What**: `sculptor/env_gen.py` + `sculptor/prompts/gen_env_spec.md`
  — the behavior goal → validated env spec, mirroring the seed-reward
  discipline: pydantic-constrained structured output
  (`messages.parse`, same as diagnose), the REAL `validate_env_spec`
  gate on the result, exactly ONE retry carrying the complete
  violation list, hard bounds rendered into the prompt FROM THE
  VALIDATOR'S OWN TABLES (single source of truth — prompt and gate
  cannot drift). The prompt encodes the audit's measured task-env
  lessons as a decision checklist (commands vs in-place skills; the
  70°-termination dead-fallen-signal + termination-as-escape pair;
  episode length ~ skill timescale; RSI **always paired** with
  min-base-height termination — the iters-19-20 lesson; pushes vs
  single-burst skills; friction; entropy for explosive skills) with
  omit-means-default semantics, and states the metric-firewall
  separation explicitly. Project-side versioning in `env_spec.py`:
  `env/v<N>.json` + `current.json` (exact copy, identity in
  meta.version — no symlinks; survives WSL/Windows + pod sync),
  `write_env_spec_version` validates BEFORE persisting,
  `repoint_env_current` for keep-best/revert (wired in 3/4).
  `sculpt.py _maybe_seed_env_spec`: runs once per project at first
  run, ONLY when no env spec exists AND config.toml made no explicit
  env choice (env_profile / env_spec_path respected — tuck-jump's
  `jump` profile is NOT silently overridden; its migration is
  increment 4, deliberate); activates the spec for the current run by
  setting adapter.env_spec_path (later runs pick it up via the
  load_adapter convention). Every failure path (no key, network,
  double validation failure) emits `env_spec_failed` and proceeds on
  task defaults. Events: env_spec_started/generated/failed.
- **Cost bound**: ≤2 LLM calls once per project lifetime; zero on
  every subsequent run.
- **Firewall**: metric files untouched; generation writes ONLY
  `env/` — it cannot touch metrics, rewards, or the loop's selection
  machinery.
- **Verified**: tests/test_env_gen.py (15: happy path, omit-dropping,
  retry-with-full-violations, double-failure raise, parse-error-as-
  attempt, versioning stamps/repoints/refuses-invalid, seed wiring:
  skip-when-exists / respect-explicit-config / failure-keeps-defaults
  / stub-client E2E). Full suite 1066 passed / 1 skipped.
  Commit 907c92e. Increment-1 adversarial verification (subagent):
  pass-with-findings — EXECUTABLE byte-parity of retired profile vs
  general path over 27 cfg shapes; no CRITICAL/HIGH. Actionable
  findings folded into 3/4: remote threading test-pinned, dead-knob
  disclosure, velocity-reset/pose_range decouple, env_spec_path
  pinned absolute at adapter init.

### 2026-07-04 — env generalization 3/4: the diagnoser iterates the
environment between training iterations

- **What**: the env spec's TRAIN section is now a first-class
  iteration surface beside the reward, with the same loop mechanics:
  * `diagnose.py`: `_ProposedEnvEditModel` — `parameter` is a pydantic
    Literal over `ITERABLE_TRAIN_KEYS` (single-sourced from
    env_spec.py, so the shared/eval section is UNREPRESENTABLE in the
    model's output, enforced at parse time); grounded prompt gains an
    `# ENV_SPEC` block (current train values + bounds from the
    validator's own tables + frozen shared for context) only when a
    spec is active; `Diagnosis.proposed_env_edits`; packing drops env
    edits when no spec is active.
  * `diagnose_grounded.md`: env-adaptation rules — 0-2 edits, only
    for training-DISTRIBUTION pathologies (floor-data domination →
    sunk height; unexperienced target phase → RSI ranges, always
    paired; exploration collapse → entropy; surface overfit →
    friction), explicit "training only — cannot make scoring easier".
  * `env_spec.apply_env_edits`: per-edit gates (train-key allowlist →
    JSON parse → full-spec validation with single-edit rollback);
    valid edits persist as the next v<N>.json (meta: source=diagnoser,
    parent, rationale), rejects carry reasons back to the event
    stream. Applied AFTER this iter's reward edit in `_run_one_iter`;
    takes effect NEXT iteration — exactly the reward-edit lifecycle.
  * keep-best/revert now operates on the (reward, env) PAIR:
    `IterOutcome.env_spec_trained` + `SculptRunResult.best_env_spec`;
    on strict regression the next iter reverts env/current.json to the
    best iter's version alongside the reward (`env_spec_reverted`
    event); end-of-run best selection repoints both
    (`best_env_spec_selected`). Env edits ride `applied_edits`
    ("env: …") so KG case memory records environment lessons.
  * Events: `env_spec_updated` (applied + rejected with reasons),
    `env_spec_reverted`, `best_env_spec_selected`.
- **BUG FIX (pre-existing, found here)**: since Ship 48,
  `diagnose()`'s pydantic→dataclass packing DROPPED
  `requires_env_extension` — on the real path every deferred edit lost
  its flag, so apply_edits burned retries on ungrounded formulas and
  the never-silent `requires_env_extension` event could not fire.
  One-line fix + regression test through the real diagnose() path.
- **Cost**: zero extra LLM calls — env edits ride the existing
  grounded-diagnose call.
- **Firewall**: metric files untouched. Env edits are structurally
  train-only (Literal + allowlist + validator section split); eval env
  frozen per run ⇒ metric comparability preserved.
- **Verified**: tests/test_env_edits.py (13: apply gates incl.
  shared-key rejection + mixed batches + dict edits; # ENV_SPEC block;
  real-path packing with/without active spec; the
  requires_env_extension regression; loop threading — env revert
  version on regression, end-of-run repoint; remote-dispatch sync pin
  from the increment-1 verifier finding). Full suite 1079 passed / 1
  skipped. Commit f00d5f4.

### 2026-07-04 — env generalization 4/4: tuck-jump migrated to the
general mechanism + verifier-hardening + E2E

- **Increment-2 adversarial verification** (subagent, commit 907c92e):
  pass-with-findings, no CRITICAL/HIGH. Folded here:
  * MEDIUM → the RSI↔early-termination pairing is now a HARD
    cross-field invariant in `validate_env_spec` (airborne height
    offsets or upward spawn velocities REQUIRE
    min_base_height_termination_m; horizontal/downward-only jitter
    exempt) — the measured iters-19-20 lesson is enforced by the gate,
    not just the prompt, and covers diagnoser edits too.
  * LOW → v<N>.json written via tmp+replace (no truncated version
    files); retry preamble distinguishes API failure from validation
    failure; meta.behavior_goal clip aligned with the prompt clip
    (900); pushes-in-train-only prompt guidance now spells out the
    required shared disable; drift-guard test pins generator model
    fields == schema key sets.
- **Migration**: tuck-jump project (outside repo,
  ~/.local/share/reward-sculptor/projects/tuck-jump) now carries
  `env/v0.json` = the jump preset expressed as a per-project spec
  (meta.source "migrated:preset:jump"); `env_profile = "jump"` REMOVED
  from its config.toml — the project runs entirely on the general
  mechanism. LIVE parity verified on the real loaded
  Mjlab-Velocity-Flat-Unitree-G1 cfg (not the fake): all 10 mutation
  surfaces byte-identical between `--env-spec` (migrated file) and the
  retired `--env-profile jump`, incl. train-only RSI/sunk.
- **E2E**: resumed sculpt run on tuck-jump (v22 base, gen_002 steer,
  fitness-patience 4) under the migrated spec — the diagnoser now
  holds the env-curriculum surface (# ENV_SPEC block) for the first
  time. Runner log iter 23 confirms the general path live:
  `env-spec applied (train=True): [commands zeroing, curriculum pop,
  push pop, fell_over 120°, ep 10 s, RSI, sunk 0.3 m]`. Evidence
  appended below as iterations complete.
- **Increment-3 adversarial verification** (subagent, commit f00d5f4):
  pass-with-findings; BOTH firm constraints held against constructed
  bypasses (injected-key JSON, 3-element ranges, Infinity, shared-key
  edits — all rejected, nothing written; firewall diff empty). Fixed
  from its findings (this commit):
  * MEDIUM — dead-knob disclosure falsely marked RSI/joint-reset knobs
    applied when the cfg couldn't honor them (my velocity-decouple had
    loosened the gate): applied[] now tags per ACTUAL write
    (`RSI(z,vz,vxy)` / `randomized(pos,vel)`); dead knobs land on the
    NOT-APPLICABLE line the diagnoser can see.
  * MEDIUM — explicit config env_spec_path pinned OUTSIDE
    env/current.json desynced the diagnose surface from the apply
    target: the loop now iterates ONLY the managed per-project spec —
    unmanaged pins get no # ENV_SPEC block, no apply/revert/record,
    and any stray proposals are rejected observably.
  * LOW — no-op edits no longer burn a version (rejected with "no
    change"); meta.rationale aggregates APPLIED edits only; invalid
    current.json at apply now emits the rejected event rather than
    stderr-only; `new_value` coerces bare numbers/pairs (saves a parse
    retry); the real `_run_one_iter` env wiring (record + apply +
    events) is now driven by an actual sculpt_run in tests via a
    stubbed-LLM loop adapter.
- **Verified**: tests/test_env_edits.py grew 13 → 19; full suite 1083
  passed / 1 skipped / 4 gpu-marked deselected (they passed in the
  pre-launch run; deselected only to keep VRAM free for the live E2E).
  Commit 066c2e9.

### 2026-07-04 — env generalization 5: 4a+4b verification fixes + the
env-spec lifecycle visible in the UI

- **Increments 4a+4b adversarial verification** (subagent, commits
  9773563+066c2e9): pass-with-findings; migration confirmed live
  (spec == jump_preset_spec() by dict equality; live run applying it,
  0 tracebacks); invariant edge-probes green. Fixed here:
  * MEDIUM — `validate_env_spec` raised KeyError on a dict-valued
    range field (valid-JSON LLM edit like {"lo":0,"hi":0.4}), killing
    a whole edit batch: `_hi` now catches LookupError; regression test
    pins the innocent-bystander edit surviving.
  * LOW — RSI+sunk proposed in one batch was order-dependent (per-edit
    validation rejected RSI-before-sunk): apply_env_edits now applies
    min_base_height_termination_m first; order-independence tested.
  * LOW — symlinked env/ dir could split the diagnose surface from the
    apply target (resolve() asymmetry): both sides now resolve fully.
  * LOW — remaining false-applied branches gated per-write
    (commands zeroing, push retune); enabled-true-no-values push spec
    correctly not a dead knob; entropy_coef_scale (diagnoser-iterable)
    now discloses NOT APPLICABLE when the task cfg lacks a positive
    entropy_coef; bool-vs-number no-op comparison quirk closed.
- **UI (memory rule: every feature UI-reachable)**: the env-spec
  lifecycle is now visible end to end —
  * backend: `IterEventSummary.env_spec_update` populated from
    `env_spec_updated` events (REST timeline survives reload); new
    read-only `GET /projects/{slug}/env-spec` (active/current/
    versions; corrupt current.json degrades to inactive, versions
    still listed).
  * frontend: Runs-tab iteration card shows an `env → v<N>` chip
    (tooltip: applied + rejected-with-reasons; "training-only, takes
    effect next iteration"); iteration detail card lists applied
    (tags) and rejected (struck-through, reason on hover) env edits;
    live WS handler + slot merge + types extended.
  The generated/failed/reverted/best-selected env events already
  reach the UI through the generic typed-event stream + LogViewer.
- **Verified**: sculptor env-layer 78 passed; UI backend suite + new
  endpoint/timeline tests; `pnpm typecheck` clean (via WSL pnpm node —
  Windows node cannot run tsc over UNC). Sculptor full suite 1085
  passed / 1 skipped / 4 gpu deselected; UI backend 365 passed.
  Commit 0d334fe.

### 2026-07-04 — E2E run 1 under the general env layer (iters 23-28,
~4.5 GPU-h, code 9773563): every new mechanism exercised live

First run in project history where the DIAGNOSER holds the
environment surface. All events below verified from the run log
(runs/sculpt_envspec_1455.log) + full-precision gen_002 recompute of
every archived rollout.

| iter | trained (reward, env) | diagnoser env edit → version | outcome (gen_002 full-precision progress; behavior) |
|---|---|---|---|
| 23 | v22, v0 | vz→[-0.5,3.0] + entropy 3.0 → **v1** | BEST (4.3e-7); upright crouch (upright 1.0, tuck 0.79, apex 5 cm) |
| 24 | v24, v1 | min_base 0.45 → v2 | STRICT REGRESSION (0.0): highest apex of the run (0.108 m) + frac_launched 0.19 — the env lever visibly moved behavior toward launching — but upright_end 0.0 (paired reward edit lost the upright basin) |
| 25 | v22, v0 (PAIR revert ✓) | vz→[0,3] → v3 | NEW BEST (3.6e-6); upright crouch reproduced |
| 26 | v26, v3 | min_base 0.45 → v4 | STRICT REGRESSION (0.0): reward_hacking + premature_termination, mean_return −9.4 |
| 27 | v22, v0 (PAIR revert ✓) | min_base 0.5 + vz [0.5,3] → v5 | regression (3.8e-11 < best) |
| 28 | v22, v0 (PAIR revert ✓) | vz→[-0.5,3.0] → v6 | NEW BEST (4.1e-6) |

- **What this proves (all machinery, live)**: goal-directed env
  proposals generated + validated + applied every iteration (6/6
  applied, 0 rejected — all within bounds incl. the RSI↔sunk pairing);
  keep-best/revert moved the (reward, env) PAIR together on every
  strict regression (3/3); end-of-run selection restored BOTH halves
  (best_reward_selected → v22, best_env_spec_selected → v0;
  current.py + env/current.json verified on disk); env edits recorded
  into KG case memory (run_cases_recorded 6, edit identities incl.
  "env: …"); the UI event stream carried env_spec_updated/reverted
  end-to-end; zero tracebacks, zero policy collapses, naturalness ok
  on all 6 iters.
- **Honest reads**: (1) no candidate beat the incumbent this run — the
  best "improvements" are noise-floor progress ties among v22 re-rolls
  (1e-7..4e-6, sensor-noise-scale ramps; seed-to-seed noise, not real
  progress); the composition problem (upright + flight + landing in
  one policy) stands where loop 6 left it. (2) reward and env edits
  land between the same iterations, so a regression cannot be
  attributed to one half — the revert restores both (Eureka-style
  joint move; both histories reach the diagnoser + case memory).
  (3) env v5/v6 were written but never trained (superseded by
  reverts/end-selection) — same disk semantics as untrained reward
  edits. (4) the iter-24 evidence (launch fraction ×6 under the env
  change) is the first measured demonstration that the env surface
  has real behavioral leverage in this loop.
- **Increment-5 adversarial verification** (subagent, commit 0d334fe):
  pass-with-findings, all LOW/INFO; independently confirmed run-1 log
  coherence (every env revert paired 1:1 with a reward revert; at most
  one env_spec_updated per iter by construction; endpoint
  traversal-safe; frontend guards present; typecheck exit 0). Fixed
  (commit b349fd1): mixed-type dict keys made the validator's
  unknown-key `sorted()` raise — now key=str + regression test; the
  errors-never-raises contract holds over Any. Left documented, not
  patched: per-GROUP (not per-sub-knob) applied disclosure for
  commands/push (partially-writable cfg shapes don't occur on real
  mjlab tasks; RSI — where sub-knobs genuinely vary — already
  discloses per-knob); a pre-existing slug-shaped-traversal 500 in the
  UI project store (spawned as a follow-up task chip, out of tonight's
  scope). E2E run 2 (iters 29-34) launched 19:08 under the
  fully-hardened code; evidence below when complete.

### 2026-07-04/05 — E2E run 2 (iters 29-34, ~5 GPU-h): first
loop-discovered (reward, env) best pair

Run under the fully-hardened code (0b31141-era; runner subprocesses
picked up per-write disclosure — log shows `RSI(z,vz)` tags and the
diagnoser's live sunk values `+sunk(base<0.5m)`).

- **CUDA transient (not a code defect)**: iter 30's first attempt died
  14 min in with an async `torch.AcceleratorError: CUDA error: unknown
  error` (Warp error 999 on device-free) inside mjlab's OWN default
  reward term — WSL2 GPU passthrough flake; iter 29 and the retry
  trained fine under identical code. Resumed with `--resume`
  (sculpt_envspec2b_2006.log); the interrupted (v30, v7) pair trained
  exactly as the resume semantics intend. Watch item: long WSL2
  sessions can drop CUDA; the loop's resume machinery absorbed it.
- **Iteration table** (gen_002 full-precision recompute):
  | iter | trained (reward, env) | env edit → version | outcome |
  |---|---|---|---|
  | 29 | v22, v0 | sunk 0.5 → v7 | 1.9e-7; upright crouch reproduced |
  | 30 | v30, v7 | vz [0,0.5] → v8 | 0.0; apex 0.097, launched 0.09, upright lost |
  | 31 | v31, v8 | sunk 0.6 → v9 | 9e-14 (new best by sub-display margin) |
  | 32 | v32, v9 | sunk 0.4 + vz [0.5,2.5] → v10 | 0.0; sunk-0.6 killed everything (mean return −7.7, premature_termination) → PAIR revert |
  | 33 | v31, v8 (revert ✓) | entropy 3.5 + vz [0,2.5] → v11 | **0.00765 — best dense progress of the RSI era** (returned 1.0, upright_end 1.0, settle > 0; tuck 0.03, apex 0.033 — a STABLE STANDER, all channels weakly nonzero) |
  | 34 | v34, v11 | sunk 0.6 → v12 | 0.0; the most launch-oriented rollout of the run (launched 0.25, apex 0.109) but upright/returned 0 |
- **Selected**: best_reward_selected → v31, best_env_spec_selected →
  v8 (vz [0, 0.5], sunk 0.5, RSI heights [0, 0.4], entropy ×2) — the
  FIRST time both halves of the project's kept-best training config
  were discovered by the autonomous loop rather than hand-authored.
  current.py + env/current.json verified on disk.
- **Honest reads**: (1) iter 33's progress is real signal (3 orders
  above the noise floor) but rewards the STABILITY half of the task —
  min-composition scores a stable stander above a tumbling launcher
  (iter 34: real jump attempts, zero score). The upright+flight+
  landing composition problem remains THE open gap, unchanged from
  loop 6. (2) The diagnoser explored the env surface systematically —
  sunk 0.3→0.5→0.6→0.4→0.6 (found 0.6 kills training: measured, in
  case memory now), spawn vz narrowed and widened, entropy raised —
  12 env versions on disk, 5 trained, every change validated, applied,
  attributed, and reverted-on-regression correctly. (3) Progress
  values below display precision again decided two best-selections
  (§6 watch item stands).

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

Runs 2-4 (iters 14-18; `sculpt_loop4d_*` / `sculpt_loop4e_*` /
`sculpt_loop4f_warmstart_*` logs; ~9 GPU-h total this E2E):

| iter | trained | behavior (measured) | gen_002 progress | g1_jump GT |
|---|---|---|---|---|
| 14 | v14 | stands very still (jv_p99 1.6); launch credit rose then DECAYED (0.009→0.001) as early jump attempts hit the now-live fall penalty | 0.0081 | 0.0 |
| 15 | v15 (LLM: −stance_recovery, +threshold-free flight_bonus 0.3 — the dense on-ground→flight bridge; KEPT tuck gate citing the partition flag) | sit-and-lift farm: flight_bonus saturated 97 % of cap; launch/tuck credit GROWING (0.01→0.07 / 0→0.16) when run ended | 0.0 | 0.030 |
| 16 | v14 again (run-boundary repoint — see BUG below) | standing | 0.0011 | 0.0 |
| 17 | v17 (LLM: flight_bonus × base-height, drift 0.2→0.5) | SIT-BOB oscillator: median base 0.15 m, feet NEVER in contact, bobs to ~0.8 m, torso upright | 0.0 | **0.215 (GAMED)** |
| 18 | v18 (LLM: base-lift gate on airborne bonus) WARM-STARTED from iter-12 tumble ckpt | tumbler adapted into upright sit-bobbing (median 0.19 m), not upright hops | 0.0 | 0.165 (gamed) |

- **DISCOVERY — the hand-written ground truth is itself gameable.**
  `spec_g1_jump` scored the iter-17 sit-bob 0.215 and iter-18 0.165:
  its apex = p97.5−median half-range reads posture cycling as height,
  its velocity-edge "launches" count sit↔rise cycles, and uprightness
  is satisfied by an upright torso while sitting. Trajectory-verified:
  feet never touch the ground in either rollout. RETROACTIVE: iter 5's
  0.258 "kept-best" is the same family (sit-farm + height-drop
  artifact — the loop-2 caveat, now confirmed as a full gaming class).
  The generated gen_002 metric (contact-verified launch + min-
  composition incl. return-to-stance + settle) correctly scores every
  one of these 0. The E2E's success bar "beat 0.258 on spec_g1_jump"
  is therefore partly a bar on a GAMED baseline; gen_002 progress is
  the honest yardstick. Do NOT tune spec_g1_jump reactively — note it
  as a benchmark-quality issue for the eval harness.
- **What the E2E proved (all with event/trajectory evidence)**:
  1. Dense progress channel ranks below the gate — 0.0196 (iter 10)
     was the FIRST nonzero selection signal in project history; ties
     no longer revert; keep-best + strict-revert + partition gate +
     Goodhart machinery all exercised and correct.
  2. Landing/return-to-stance credit works: c_returned 0 → 0.97-0.99,
     upright_end 1.0, drift 4.2 m → 0.5 m; iter-10 keyframes visibly
     confirm an upright stance start-to-end.
  3. ZERO policy collapses in 8 trained iterations (was 2-of-2 edit
     iters before the fixes); no edit was penalty-stacked; every LLM
     edit was minimal, phase-directed, correctly diagnosed from the
     component traces, and correctly deferred to the partition guard.
     The v6/v8 failure class did not recur.
  4. The env profile closed the termination-laundering corridor
     (fallen now fires; falls accrue their penalty).
- **What remains open — re-scoped gap #7.** The loop now reliably
  prices out every degenerate attractor (stand-farm, sit-farm,
  tumble-bounce, sit-and-lift, sit-bob) but PPO-from-scratch (or from
  a tumble warm-start) does not DISCOVER the coordinated upright
  launch-land cycle under a live fall penalty: exploration retreats to
  the nearest safe/degenerate basin (measured: iter-14 launch credit
  decay 0.009→0.001). This is no longer a reward/metric/loop-mechanics
  gap. Next increment = reference-state initialization (DeepMimic RSI,
  §1): initialize a fraction of episodes mid-flight/at-apex via the
  env profile's reset events so the policy experiences
  apex→descent→landing→stance BEFORE it can produce a launch, plus
  optionally a crouch-impulse curriculum. The jump env profile is the
  natural seam (reset_base pose/velocity ranges are already mutable
  there).
- **BUG — FIXED 2026-07-04**: `_run_one_iter` trained `current.py`'s
  target but recorded `reward_path_trained = latest_reward_file` AND
  used the latest file as the EDIT BASE; the three diverge at run
  boundaries after best-selection repoints (iter 16 trained v14 while
  events said v16, its diagnosis was applied to v16's source, and
  keep-best then repointed current.py at v16 — a never-trained file).
  Fix: `_current_reward_target()` parses current.py's re-export line;
  the trained record, the edit base, and both iter events now follow
  it (fallback: latest, with auto-repair when a generated current.py
  dangles at a deleted file). Regression tests drive the real loop
  through a simulated boundary (v3 derives from v0's source, not v2's;
  events report v0). Same-run reverts were always correct
  (revert_base sets all three explicitly).
- **Final project state**: current.py → v14 (strongest validated
  reward design: upright-gated flight credit + honest gates + stance
  0.1); best-by-gen_002 behavior remains iter 10 (progress 0.0196);
  rewards v15-v19 on disk with v16/v19 never trained.

### 2026-07-04 — loop 6: gap #7 built — RSI + early termination +
explosive-motion PPO (commits c645094, b882dbe; run-boundary fix
23dd83f/cfd13c9 landed the same day)

- **What shipped**: jump profile now adds, TRAIN-ONLY (rollout keeps
  honest standing starts so the metric's view is unchanged):
  (a) DeepMimic-style RSI resets — episodes start uniformly stance→
  +0.40 m with vz ∈ [−0.5, +2.0] m/s (verified live: 66 % of envs
  spawn airborne); (b) `sunk` termination at base < 0.30 m — RSI's
  required other half: the floor-sit basin is orientation-UPRIGHT so
  no bad_orientation cut can touch it, and without it ~9 s of every
  crashed episode is floor data that dominates PPO's distribution
  (measured iters 19-20: all shaping terms opened strong under RSI
  then decayed to ~0 as the policy converged to the sit);
  (c) entropy_coef ×2 (0.01→0.02) per §1's explosive-motion PPO note.
  Plus `sculpt run --init-policy` (warm-start chaining on the CLI).
- **E2E evidence (iters 19-22, ~4 GPU-h, all under the fixed
  run-boundary bookkeeping — iter events now name the true trained
  version)**:
  | iter | trained | behavior | key numbers |
  |---|---|---|---|
  | 19 | v14 (RSI, no sunk) | floor-roll | upright 0.05, median z 0.14 |
  | 20 | v20 (LLM: dense height term, stance ramp from floor) | floor-sit again | shaping terms 0.33-0.50 first window → decayed ~0 |
  | 21 | v20 (RSI + sunk + entropy×2) | **FIRST CONTACT-VERIFIED FLIGHT in project history**: 3/6 envs, apexes 1.24-1.39 m, frac_launched 0.83, tuck 0.45 rad — but tumbling (upright 0.07) | flight frames [29,26,0,0,31,0]; GT g1_jump apex 0.60 m |
  | 22 | v22 (LLM: tuck gated on both-feet-airborne — the dive-farm read straight off the sunk-termination episode stats) | upright DEEP CROUCH — the launch posture: median z 0.45, upright 1.00, tuck 0.70 rad, ends upright at start height | no flight this iter; c_upright_end 0.98 |
- **Reading**: the degenerate attractors are being eliminated in
  sequence (sit → roll → tumble → dive-farm), and the surviving
  behaviors now live on the jump manifold (real 0.5 m flights at iter
  21; the upright pre-jump crouch at iter 22). The remaining composition
  problem — upright + flight + landing in ONE policy — is now a reward-
  balance question the loop is actively iterating (v22's known hole:
  height_progress pays tumbling flight; the case memory carries that
  lesson). Dense progress is still gated at 0 by the min-composition
  (correct: no single iter satisfies all requirements yet).
- **Honest negative**: RSI WITHOUT the early termination made things
  worse (iters 19-20) — recorded so the pairing is never split again.

### 2026-07-03 — loop 5: the KG actually learns from every run
(commits ed70d76, 4ca55ae, f1a20f8, 99d0a2e)

Full-system audit of the knowledge graph + run-learning memory, driven
by the question "does every run make the system smarter?" Answer
before this loop: NO, for three separately-measured reasons.

- **5a — ONE graph (ed70d76).** `default_db_path()` preferred a
  cwd-relative `kg/graph.db` when present. Measured harm: the entire
  loop-4 E2E (launched from the repo dir) diagnosed against a
  6-technique repo-local stub while the shared graph held 94 papers /
  493 techniques, and recorded its run cases into that silo. Removed
  the legacy preference (env overrides intact); new `sculpt kg merge`
  (additive-only — a legacy stub can never clobber a richer shared
  node; source renamed .merged); both strays on this box folded in →
  shared graph 1523 nodes / 1608 edges / 546 embeddings. tests/conftest
  now isolates every test onto a temp DB (previously a bare
  `SculptorKG()` in a test could write the developer's real graph).
- **5b — case memory with content (4ca55ae).** All 17 tuck-jump cases
  were verdict-neutral/unknown noise: attribution used only the
  completion-gated fitness (0.0 throughout) and recorded only "N
  edit(s)". Now: lexicographic (fitness, dense-progress) attribution —
  the same key the loop selects on, so "decrease stance_weight;
  increase launch_weight → regressed (progress −0.0196)" is exactly
  what the memory says; RunCase carries the applied-edit identities +
  a ≤6-float behavior signature (apex/launch/return/upright/tuck/
  drift) distinguishing stand-farm from tumble-bounce; the CASE MEMORY
  block now ALSO feeds the edit REWRITER prompt (where magnitudes are
  chosen); blind runs record too. The 17 thin cases were deleted and
  re-recorded rich from on-disk artifacts (diagnosis.json + recomputed
  gen_002 details, revert/run-boundary attribution respected).
- **5c — the graph is worth looking at (f1a20f8).** RunCase nodes were
  gray unlabeled blobs; now teal diamonds "iter N ✓/✗" with
  verdict-colored borders + full tooltips. Scale-adaptive rendering
  for the 1.5k-node unified graph: improvedLayout off (the quadratic
  Kamada-Kawai placement froze the tab — caught live), edge labels →
  hover tooltips, forceAtlas2, physics frozen after stabilization
  (measured: 36.6 ms full redraw, zero idle). New in-page controls
  (also in the UI GraphModal iframe): debounced search with animated
  focus + Enter-cycling, per-kind visibility filters with counts,
  re-run-layout button. Verified functionally in a live browser.
- **5d — visible in the UI (99d0a2e).** KGStats models predate the
  case memory; the Settings shared-KG card now shows "Run experience"
  (run_cases) alongside papers/techniques.
- **Verified**: sculptor suite 1013 passed / 1 skipped; UI backend 362
  passed; frontend typecheck clean. Live retrieval check: a tuck-jump
  query returns the rich cases at sim 0.72-0.86 with edit identities +
  behavior signatures in the rendered prompt block.

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
