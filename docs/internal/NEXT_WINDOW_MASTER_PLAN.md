# MASTER PLAN — everything the next window needs (2026-07-06, final)

Written at session end under a hard token budget. This supersedes
HANDOFF_RESEARCH_GRADE_SESSION.md where they differ. Read this FIRST,
then MISSION_METRIC_GRANULARITY.md and RESEARCH_GAP_ANALYSIS.md.
Branch: ship-20-ux-revamp, all gates green at last commit (adfba04):
sculptor 1211 passed/1 skip, backend 376 passed, pnpm build green.

## 1. LIVE RESULT — the G1 standing-jump mission (read this carefully)

A bounded smoke of the full autonomous chain ran TODAY on the 5070
(log: ~/rs_smoke/g1_mission_run2.log; mission dir:
~/.local/share/reward-sculptor/projects/tuck-jump/.missions/g1-standing-jump).
Budgets were deliberately tiny (--iterations-override 2
--steps-per-iter 600) — this was a MACHINERY test, not a convergence
attempt. What happened, in order:

1. fable-5 decomposed into crouch_load → explosive_launch →
   jump_and_land_stable (warm-start chained), flagged
   needs_reference_rsi=true on exactly the two airborne stages, and
   self-assigned the hand g1_jump metric to them; crouch_load got a
   FRESH generated metric that passed the full trust pipeline
   (stage_metrics/crouch_load/metric.py — steering source event
   confirmed "stage").
2. crouch_load SUCCEEDED in 2 iters (criterion: crouch_depth>0.35 +
   uprightness + episode length; last metric 42.7).
3. explosive_launch: reference-RSI applied (procedural clip → env/v0
   with paired sunk termination), warm-started from crouch_load, and
   SUCCEEDED in 2 iters — criterion included
   `(root_link_pos_w[...,2] > 0.85).any()` and flight_bonus>0.05, i.e.
   the policy left the ground. CAVEAT: its last_iter_metric was
   NEGATIVE (-5.52) while the criterion passed — criterion vs metric
   disagreement. Watch the rollout video before believing "flight":
   possible criterion gaming (a fall's root excursion can spike z? no —
   z>0.85 is ABOVE stand height 0.78, falls go down; but a bounce off
   RSI-initialized airborne resets could satisfy .any() without a
   self-produced launch. RSI resets start SOME episodes airborne —
   `.any()` over the whole trajectory can be satisfied BY THE RESET
   STATE ITSELF). **This is the #1 thing to check in the next window:
   does the criterion evaluate on rollout episodes that start airborne
   because of train-scope RSI? It must not — rollout is supposed to be
   RSI-free (shared/train split). If rollout is clean, the flight is
   real and remarkable at 600-step budgets; if trajectory.npz shows
   episodes STARTING above 0.85m, the eval path is leaking train RSI
   and that's a CRITICAL bug in the mjlab runner's spec handling.**
3b. POST-SESSION CHECK (token-limited, inconclusive — FINISH THIS
   FIRST): explosive_launch's rollout trajectory.npz has
   root_link_pos_w shape (500, 64, 3) against episode_id (3000,), and
   naive z-extraction gives values up to 7.4 m — impossible for a G1.
   Either the 64-axis is bodies/links (so the criterion's
   `[..., 2] > 0.85 .any()` scans EVERY body's z — a raised HAND could
   satisfy "flight"), or coordinates/episodes are laid out differently
   than the criterion evaluator assumes. Resolution path: read
   `_episodes_to_npz_dict` + the mjlab runner's trajectory writer to
   pin the layout, cross-check `mission_runtime._build_criterion_
   namespace`'s handling, and WATCH stages/explosive_launch/runs/
   iter_1/rollout/rollout.mp4 (30-second human check). Until then,
   treat BOTH airborne-stage criterion passes as UNVALIDATED, and
   treat "criterion references multi-body arrays without an explicit
   root index" as a decompose-prompt defect to fix (criteria should
   use a defined root channel, not `[..., 2]` over an ambiguous
   array). The stage-machinery verification (RSI applied, warm-start
   chain, redecompose) stands regardless.
4. jump_and_land_stable failed its (much harder, full-composition)
   criterion at the 2-iter budget — expected — and then AUTONOMOUSLY
   RE-DECOMPOSED itself (Ship 17) into sub-stages; the first sub-stage
   goal was "stop hopping, hold a sustained stable stand", which is a
   sensible diagnosis of the landing problem. The run was still going
   at session end; final state will be in the log +
   reports/mission_quality.json.

Known refinement found live: re-decomposed sub-stages INHERIT
needs_reference_rsi (decompose.py, `or failed_stage.needs_reference_rsi`)
— the "hold a stable stand" sub-stage got RSI it doesn't need
(airborne resets while learning to stand still). Not harmful (paired
termination keeps it safe) but wasteful; better rule: inherit ONLY if
the sub-stage's own flag is true OR its goal is airborne (let the
redecompose LLM decide — remove the `or failed_stage...` inheritance
and trust the per-sub-stage flag, since the redecompose prompt now
documents the field).

## 2. THE REAL JUMP ATTEMPT — exact recipe

```bash
cd ~/projects/RewardSculptor && source .venv/bin/activate
export ANTHROPIC_API_KEY=$(grep -oP 'ANTHROPIC_API_KEY\s*=\s*\K\S+' .env)
P=~/.local/share/reward-sculptor/projects/tuck-jump
# fresh mission, full budgets (config already has edit_candidates = 3):
python -m sculptor.cli mission-init $P --goal "<same jump goal>" --slug g1-jump-full
python -m sculptor.cli mission-run $P g1-jump-full   # NO overrides
```
- Full budget ≈ stages × max_iterations × ~22 min at steps_per_iter
  1500. Raise steps_per_iter (config or --steps-per-iter 3000+) for the
  launch/landing stages if crouch converges fast — undertrained
  warm-starts poison downstream stages.
- BEFORE the full run, fix §1's RSI-in-rollout question and the §1
  inheritance refinement.
- Better reference: convert a real retargeted clip to
  $P/reference/jump.npz (Unitree LAFAN1 HF dataset, auth-gated;
  only root_pos_z + fps required — see sculptor/reference.py
  docstring). Procedural is serviceable; real mocap gives honest
  crouch depth/timing.
- Success artifact for the lab demo: rollout.mp4 of
  jump_and_land_stable's kept iter + the mission event stream + the
  per-stage metric provenance chain (llm_calls.jsonl +
  kg_retrievals.jsonl) — "the system decided, cited, measured".

## 3. UNVERIFIED WORK — one focused review pass needed

TWO adversarial verifier agents were killed by the spend limit
mid-pass. Inline re-checks done by the main agent (ranking-key ties,
all-invalid fallback propagation, contextvar-in-thread, backend
phase-2 round-trip, KG boost cap/floor ordering, provenance defaults).
STILL UNREVIEWED by fresh eyes: commits f128ca8, a97a982, 021ae85,
0551a6d, 1ac8f46, e5afa7f, adfba04. Specific open questions to check:
- heldout.py `_perturbed_spec` REMOVES existing push_events at level 0
  — arguably the unperturbed base should be the project's shared env
  VERBATIM (if it already trains with pushes, level-0 should keep
  them). Judgment call; current behavior is documented but debatable.
- partition-gate rename close: an attacker adding a DECOY reject gate
  makes the pairing ambiguous → advisory only. Accepted limitation
  (advisory still fires + the hack-income screen still catches the
  exploit's income); documented here so nobody re-discovers it.
- RSI hook: if a stage's diagnoser later writes a non-reference env
  version and the stage RESUMES, the hook re-applies RSI on top
  (meta.source no longer starts "reference:"). Benign (validated,
  bounded, paired) but produces an extra env version; guard could
  check ANY prior version with reference source, not just current.

## 4. STRATEGIC PRIORITIES (unchanged, now unblocked)

P0 — the paper is §7.1 (metric-gaming base-rate study): all raw
material archived; naive-vs-gated metric populations, score archived
gamed/honest rollout classes, 2-4 blind human raters, layer ablation.
Compute-free. The trust pipeline + per-stage metrics + provenance
landed this session make the "instrument" story complete.
P0 — spec repair (§7.3) before ANY campaign: monotone kick metric
(prototype in audit script), g1_jump contact-verified launch fix (the
sit-bob 0.215 hole — NOTE §1's .any() concern is the same class).
P1 — E4-v2 (§7.5) after 7.1/7.3: arms are now model-matched (both
fable-5); add mission arm with per-stage metrics; 5→10 paired seeds.
P1 — KG follow-ups: retrieval logs now exist per decision — correlate
retrieved-node sets with edit acceptance (retrieval
precision-in-hindsight; free analysis over kg_retrievals.jsonl).
DEFERRED (designed, not built): Fix B subprocess sandbox
(FIX_B_SANDBOX_DESIGN.md); MCTS over edit histories (RF-Agent
direction); L4 pairwise VLM queries; heldout UI card (Sam launched the
chip task task_b923f338 in another session — check its result).

## 5. COST LEVERS (protect the budget)

- RS_MODEL_ALL=claude-opus-4-8 (or sonnet-5) downgrades every LLM role
  in one env var; RS_MODEL_EDIT etc. for per-role. Defaults are
  fable-5 on steering roles = the expensive-but-strongest setting.
- edit_candidates=3 triples ONLY the edit-call cost, not GPU. Set to 1
  for budget runs.
- Subagent discipline for the orchestrating model: this session lost
  three subagents to spend-limit deaths mid-task. Prefer inline work +
  ONE well-specified agent at a time; commit after every increment;
  never leave uncommitted work while an agent runs.

## 6. WHERE EVERYTHING IS

- Commits (this session, oldest first): c75d8f8 merge-recovery,
  f26e2c0 UI, f128ca8 registry+best-of-K, a97a982 per-stage metrics,
  021ae85 RSI stages+grounding, 0551a6d partition/provenance,
  1ac8f46 heldout, e5afa7f stage-metrics path fix, adfba04 KG.
- Live mission artifacts: tuck-jump/.missions/g1-standing-jump/
  (mission.json, stage_metrics/, stages/*/runs/*/rollout/rollout.mp4,
  llm_calls.jsonl, kg_retrievals.jsonl) + reports/mission_quality.json.
- Smoke #1 (hopper, negative-control evidence that trust gates reject
  out-of-manifest metrics): ~/rs_smoke/hopmission.
- NOT pushed to GitHub — push when ready:
  `git push origin ship-20-ux-revamp` (repo github.com/sjdoane/RL-Sculptor).
