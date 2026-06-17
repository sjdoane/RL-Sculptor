# HANDOFF — Making objective-metric generation rigorous (esp. joint identification)

**Status:** research + design complete (2026-06-15). NO code changed for this handoff — it is the
spec for a new window to implement. Author: prior window (after the g1-kick-v3 diagnosis + Ships 46–48).

**Why this matters (Sam's framing):** the single biggest lever in the project is *reliably generating an
objective metric, tested before each project, that accurately measures the target movement and then
steers the reward rewrites.* If that metric is trustworthy, the whole sculpt loop is trustworthy. Two
hard requirements: (1) it must **ALWAYS correctly identify joints**, and (2) it must **actually measure
success for the given movement** (not a Goodhart proxy).

---

## 0. Start here (new window)

Read, in order: this doc → [`DESIGN_autonomous_metric_eval.md`](DESIGN_autonomous_metric_eval.md) (the
existing L0–L5 trust-pipeline spec, Ships 50–55) → CONTEXT.md change-log Ships 35, 36, 40–47 → memory
`project-g1-kick-diagnosis` and `project-autonomous-metric-eval`. Everything below is grounded in
file:line from the current tree and a read-only experiment you can reproduce (§3A).

This handoff adds ONE thing the existing design doc does **not** cover — **joint-identification
correctness (§3A/§4A)** — and reorganizes the existing trust-pipeline work (§3B/§4B) around the same
goal, enriched with external research (§6).

---

## 1. What "done" looks like

At project launch (no GPU), for *any* movement goal + robot:
1. A metric is generated and **proven before the run starts** to (a) resolve the exact joints it needs
   on *this* robot, or be rejected; (b) be safe/deterministic/bounded; (c) discriminate competent vs
   degenerate/gaming behavior; (d) rank an *independently-authored* competence ladder (so it earns
   steer-rights even on a novel task, not just the 5 built-ins).
2. If it can't be proven, it runs **observe-only** with a *specific* reason — never silently steers a bad
   metric (the firewall already enforces this; we widen what can legitimately pass it).
3. Joint identity can NEVER silently drift: a wrong/shuffled/foreign joint list either resolves correctly
   or hard-fails loudly — never produces a plausible-looking wrong score.

---

## 2. Current state — the pipeline (verified, file:line)

Trust chain: **generate → validate → review → calibrate → steer-rights firewall → runtime**.

| Stage | Where | What it does |
|---|---|---|
| Generate | `metric_gen.py:90-208` (`generate_objective_metric`), prompt `prompts/gen_objective_metric.md`; launch-time `run_manager.py:143-247` (`_generate_at_launch`, ≤4 attempts) | LLM writes `compute_spec(arrays, behavior, meta) -> {spec_score∈[0,1], ...}`. Retries on gate failure. Never raises. |
| Validate (L0) | `metric_validate.py:317-461` | 5 MUST-HAVE gates: `_ast_safety` (110-135, numpy-only, no eval/exec/dunder), array-contract (`_referenced_array_keys` 138-164; only `joint_pos/joint_vel/projected_gravity_b/root_link_pos_w`), determinism (3× identical), bounded ([0,1] finite), non-degeneracy (399-456). Ship 47 added the stationary-skill walker ceiling. **All gates run on 9 synthetic archetypes** (176-300) using a fixed name list `_NAMES_12`. |
| Review | `metric_gen.py:62-87`, prompt `review_objective_metric.md` | Independent LLM context (never sees the metric source) judges: measures the goal, hard-to-game, physical-only, sane archetype scores. Accepted ⟺ validate ok AND review approved (`metric_gen.py:177-187`). |
| Calibrate (earns steer) | `metric_calibration.py:147-182` (`calibrate_metric`), ladders `61-144`, `spearman` `35-53` | Spearman ≥ 0.7 vs a **hand-authored built-in** over a **synthetic** competence ladder → `ok=True`. `metric_store.calibrate` (`run_manager`/`metric_store.py:127-140`) writes `calibrated` to `meta.json`. |
| Firewall | `run_manager.py:78-94` (`steer_allowed` reads `meta.calibrated`) | Only `calibrated=true` steers; else observe-only. Built-ins always steer. Never silent (`metric_calibration_skipped`/`_done` events). |
| Runtime | `generated_metric.py:46-118` | Per iter, loads the metric (unique module name per path), loads arrays from `trajectory.npz` + `joint_names` from `mjcf_limits.json` into `meta`, scores `iter_n/rollout`. Crash → `spec_score=0.0`. |

**Family resolution** (`metric_validate.py:66-107`, `resolve_behavior_family`): WORD-token match of the
goal/robot_hint → `kick/floss/jump/locomotion/cartpole/None`. Selects the calibration built-in and the
validation positive archetype.

**What Ship 47 (this session) already added** to L0 — context so you don't re-do it:
`spec_g1_kick` stationarity factor (`spec_metrics.py:445` `_KICK_STATIONARY_SCALE=0.01`, gate ~481-486; `root_link_pos_w` added to `_REQUIRED_ARRAYS`); realistic `walker` archetype (`metric_validate.py:274-295`); `_STATIONARY_FAMILIES={kick,floss,jump}` + `distractor_ceiling=0.3` family-scoped gate (309-314, 322, 449-455). Tests: `test_kick_penalizes_forward_travel`, `test_walker_archetype_present_and_caught_for_kick`, `test_good_kick_metric_clears_walker_ceiling`, `test_walker_ceiling_skipped_for_locomotion`.

**The CRITICAL gap, in one line:** calibration is **synthetic-only** and only exists for the **5 built-in
families** — a novel task gets `resolve_calibration_builtin → None → observe-only forever`
(`run_manager.py:208-226`). And **nothing anywhere verifies joints were correctly identified.**

---

## 2.5 Live signal — g1-kick-v4 (Sam, 2026-06-15, dev server still running)

Fresh evidence from Sam live-testing the Ship 46–48 stack on project **g1-kick-v4** (better than v3):
- It reached a **high objective metric while actually kicking** — BUT the kick was **SIDEWAYS and not
  realistic**. Strong signal that the metric rewards hip roll/abduction-style leg motion, not a *forward*
  hip-pitch kick → the **directionality / joint-precision gap** in §3A (greedy `_match_joints` grabs all leg
  joints with no pitch-vs-roll or left/right precision; nothing enforces "forward").
- As training developed, **fitness got stuck at 0 with heavy flailing**. Two hypotheses for the new window to
  disambiguate from the g1-kick-v4 artifacts (`~/.local/share/reward-sculptor/projects/g1-kick-v4/`): (a) the
  metric correctly zeroed flailing but the *sculpted reward* drove the policy into a flailing local optimum
  with no recovery gradient (a reward-shaping/curriculum problem), or (b) the metric **over-gates** and zeros
  almost everything (a metric-quality problem). Pull `runs/*/diagnosis.json`, `metric_history`, and re-score
  the metric's component breakdown per iter (as was done for v3 gen_005) to tell which.

Takeaway: the foot channels + Ship-47 hardening helped, but the metric still doesn't pin the kick's
**direction or realism** — this is concrete motivation for the §4A joint-precision work and the §4B
trust pipeline (esp. directionality-aware archetypes + L5 optimization-outcome audit).

## 3. Gap analysis

### 3A. Joint identification — currently UNGUARDED (the new finding; not in the design doc)

**How joint identity flows:** the mjlab runner captures the robot's `joint_names` and persists them to
`mjcf_limits.json` (`_mjlab_runner.py` ~920-926); `compute_spec_metrics`/`generated_metric` load them into
`meta["joint_names"]` (`spec_metrics.py:684-693`, `generated_metric.py:86-95`). Metrics then locate joints:
- **Built-ins:** `_match_joints(names, tokens)` (`spec_metrics.py:58-62`) — greedy lowercase **substring**
  match. `_LEG_TOKENS=("hip","knee","ankle")` grabs **all 12** G1 leg joints (both sides, incl. roll/yaw),
  with **no left/right or pitch/roll precision**. Falls back to **all joints** when `len(names)!=J`
  (`spec_metrics.py:466`).
- **Generated:** each metric **reinvents** brittle name matching (gen_005 searches `"hip"`+`"pitch"` +
  `left/right/l_/r_` tokens, returns indices, hard-fails to 0.0 if not found).

**Two unstated invariants that nothing checks:** (i) `joint_names` order MUST equal the `joint_pos`/
`joint_vel` buffer column order; (ii) the robot's naming must match the metric's tokens. Break either and
the metric **silently scores the wrong joints**.

**Reproducible evidence** (read-only; real G1 iter1 rollout — rerun the snippet from this session):

| joint_names passed | gen_005 (name-based) | builtin g1_kick (token-based) |
|---|---|---|
| correct | spec 0.591, kicks 3.22 | 0.139, leg_subset=1, 12 legs |
| **shuffled** (same names, wrong order) | **0.559, kicks 2.12 — silent mis-score** | **0.185 — silently HIGHER, leg_subset=1** |
| empty | 0.0 (hard fail, gen_005 `len==J` guard) | 0.137, leg_subset=0 (all joints) |
| wrong robot (FL_hip/thigh/calf…) | 0.0 (hard fail) | **0.212 on 4 wrong joints, leg_subset=1** |

**No validation gate checks joint resolution.** The non-degeneracy gate scores archetypes with a fixed
`meta={"joint_names": _NAMES_12}` (`metric_validate.py:372`) — a synthetic 12-name list that need not match
the real 29-joint G1 order/naming. A metric that hard-codes `jp[:, :, 0]`, or mis-resolves by name, passes
validation and then silently mis-scores on the real rollout.

### 3B. Metric trustworthiness for arbitrary movements (the existing L0–L5 design)

Already specified in `DESIGN_autonomous_metric_eval.md` (Ships 50–55) — status **design-only** except L0
(implemented) and L0's Ship-47 hardening:
- **L1 axioms (Ship-50):** universal-invariant gate (no reward for stillness/extremes; monotone-in-
  uprightness; responds to the goal's named quantities). Pure-numpy over archetypes. *Missing.*
- **L2 task-derived ladder (Ship-51) — THE UNBLOCKER:** K=3 *independently-authored* competence ladders
  (author sees goal+robot_hint only, never the metric); score the metric on each; `rho_min` (min Spearman,
  not mean) + `agreement_fraction ≥ (K-1)/K` earns steer-rights for **novel** tasks, no GPU.
  `gen_competence_ladder.md` prompt + `calibrate_task_derived()` *do not exist yet.*
- **L3 cross-metric consensus + adversarial archetypes (Ship-53):** fold author-proposed gaming policies
  into the negatives. *Missing.*
- **L4 VLM grounding (Ship-54, cost-gated):** numeric metric vs a general VLM "matches goal?" over existing
  keyframes. *Missing.*
- **L5 optimization-outcome audit (Ship-55, cost-gated):** train briefly against the metric → `realism.
  audit_rollout`; a hacked/`severe` outcome revokes the grant. *Missing.* (This is the only truly
  non-circular ground truth.)

Other gaps the research surfaced: family resolution is word-token only (paraphrases → `None`); the
`distractor_ceiling=0.3` is hardcoded; synthetic ladders are graded *rollouts*, not trained *policies*.

---

## 4. Proposed design

Two parallel thrusts. **4A (joint ID) is a prerequisite for everything** — a perfect trust pipeline is
worthless if the metric reads the wrong joints. **4B is the existing L0–L5 work, reprioritized.**

### 4A. Joint identification — make it ALWAYS correct or hard-fail

1. **Adapter order-contract test (root cause of the shuffle failure).** Add a sculptor test asserting the
   adapter persists `mjcf_limits.json["joint_names"]` in **exactly** the `joint_pos`/`joint_vel` buffer
   column order (`_mjlab_runner.py` ~920-926). This is the invariant every name-based metric silently
   relies on. Pin it so a future adapter change can't break every metric at once.
2. **One canonical joint resolver** — `sculptor/eval/joint_resolver.py`: `resolve_joint_roles(names,
   required_roles) -> {role: index}` returning canonical roles (`left_hip_pitch`, `right_knee`, …) or
   raising/flagging missing roles. Handles naming variants (`left_/L_/_l_`, `hip_pitch` vs `pitch_hip`) in
   ONE audited place. **Replace** `_match_joints` and every generated metric's ad-hoc loop with it.
3. **Required-roles declaration + validation gate.** The generator declares the roles it needs (structured
   field, e.g. `REQUIRED_JOINT_ROLES = ["swing_hip_pitch", "swing_knee", "stance_*"]`). New gate in
   `metric_validate.py` (after non-degeneracy): on the **actual robot's** `joint_names`, assert every
   required role resolves and the names at those indices contain the expected tokens — else **reject
   pre-project**. This is the gate that makes "always identify joints" enforceable.
4. **Cross-permutation robustness gate (cheap, catches hardcoded indices).** Re-run the non-degeneracy
   archetypes with **permuted** `joint_names` (same data). A name-resolving metric is stable; an
   index-hardcoding one swings — flag `[robustness] index-sensitive joint access`. (This is exactly the
   §3A shuffle experiment, turned into a gate.)
5. **Ban raw integer joint indexing** in generated metrics (extend `_ast_safety`): flag literal
   `arrays[...][:, :, <int>]` into joint axes; require name-resolved indices. Whitelist non-joint axes
   (e.g. gravity `[...,2]`).
6. **Persist resolution to `meta.json`** (`{resolved_joints, method}`) so `realism.audit_rollout` can verify
   the indices still match the live `joint_names` each run (catches drift across runs/robots).
7. **Prompt update** (`gen_objective_metric.md`): a "JOINT RESOLUTION SAFETY" section showing the canonical
   `resolve_joint_roles(...)` pattern + how to declare required roles; forbid integer joint indices.

### 4B. The trust pipeline (implement the design doc, in this order)

- **Ship-50 (L1 axioms)** — small, no-GPU, hardens *every* metric immediately. Add the universal-invariant
  gate beside `validate` reusing `_archetypes()`.
- **Ship-51 (L2 task-derived ladders) — the headline unblocker.** `gen_competence_ladder.md` (author sees
  goal+robot_hint ONLY) emits a *competence axis + rung descriptions*; a **deterministic** synthesizer (NOT
  the LLM) builds the numpy rollouts (reuse the `_ladder` builders' style, and **4A's resolver** so rungs map
  to the right joints). `calibrate_task_derived(metric_path, goal, robot_hint, k_sources=3)` → `rho_min` +
  per-source scores; replace the `metric_calibration_skipped` branch in `_generate_at_launch`. Gate:
  `rho_min ≥ 0.5` AND `agreement_fraction ≥ 2/3`. **Structural anti-collusion:** assert the metric source
  never appears in any ladder/grader payload; log model-id/timestamp/payload-hash to `meta.json`.
- **Ship-53 (adversarial archetypes)** — ask an independent LLM for 3 gaming policies per goal; synthesize
  them (resolver-backed) and require the metric to score them below the positive. Generalizes Ship-47's
  hardcoded walker/flail negatives to any task.
- **Ship-55 (L5 optimization-outcome audit, cost-gated)** — the empirical Goodhart catch: short train vs
  the metric → `realism.audit_rollout`; `severe` → `calibrated=False`. This is what would have caught the
  g1-kick-v3 0.59 metric *behaviorally*, not just by archetype.
- **Ship-54 (L4 VLM grounding, cost-gated, optional booster)** — Spearman(metric, VLM "matches goal?") over
  existing keyframes; a confidence signal in `meta.json`, not a gate.
- **Ship-52 (trust-score unification)** — one scalar `trust = w_cal·CAL + w_evi·EVID`
  (`CAL=clip((rho_min−0.5)/0.5,0,1)`, `EVID=gate_pass(validate)·gate_pass(axioms)·agreement_fraction`);
  built-ins must produce byte-identical decisions (regression test).

### 4C. UX (design doc §UX)

Replace the dead-end "No matching built-in — observe-only" card with "Calibrating vs 3 independently-
generated competence ladders…" → "Trust 0.81 (3/3 agree) — steering" or "Trust 0.42 — observe-only (ladders
disagree)". Every observe state names the failing layer + number. Add a joint-resolution line ("resolved
swing_hip_pitch→idx 0 ✓"). Extend the existing `calib` event + reducer; no new component, no new terminal step.

---

## 5. Recommended phased plan (no-cost first; matches "tested before each project")

1. **Phase J (joint ID, do FIRST — prerequisite, no GPU):** §4A items 1→7. The order-contract test (1) +
   resolver (2) + required-roles gate (3) + permutation gate (4) are the core. Ship as its own commit
   ("Ship 49: always-correct joint identification or reject"). Add tests mirroring the §3A table
   (shuffled/empty/wrong-robot must reject or be stable, never silently mis-score).
2. **Phase 1 (L1 axioms, Ship-50):** no GPU.
3. **Phase 2 (L2 ladders, Ship-51):** no GPU; API-mockable. **This is what makes novel-task metrics earn
   steer-rights** — the biggest single win after joint ID.
4. **Phase 3 (Ship-52 trust score, Ship-53 adversarial):** no GPU.
5. **Phase 4 (cost-gated boosters, confirm cost with Sam): Ship-55 optimization-outcome audit (GPU), then
   Ship-54 VLM grounding (API).**

Every phase: four gates green (sculptor `uv run pytest tests/ -q`; backend
`uv run pytest backend/tests/ -q -k 'not test_reward_prompt_edit_emits'`; frontend `pnpm build`; live
`./run.sh`). Append a CONTEXT.md entry + commit per ship (`Co-Authored-By` trailer). All of Phases J–3 are
GPU/API-free (API-mockable for tests).

---

## 6. External research (verified, with sources) — techniques to apply

- **Eureka** (arxiv 2310.12931) — LLM reward design + evolutionary reflection. Apply to **Ship-51**: mutate
  ladder *rung descriptions* under disagreement feedback so ladders are genuinely independent of the metric.
- **Text2Reward / Auto-MC** (2309.11489, 2312.09238) — code-level critic of generated reward; flag
  exploitable patterns (joint-limit/torque thresholds, ignoring gravity). Feeds **L1/L3**.
- **Defining/Characterizing Reward Hacking** (2209.13085) + Lil'Log survey — multi-layer Goodhart defense;
  validates the L0–L5 layering. Empirical audit (L5) is the decisive layer.
- **VLM-RM** (2310.12921), **SuccessVQA** (2303.07280) — VLM zero-shot success scoring from keyframes →
  **Ship-54**. Correlation signal, not a gate (VLMs fail on sim-unrealism).
- **Consensus Entropy / ensemble validation** (2504.11101, 2411.06535) — use **min** (not mean) Spearman
  across K sources so one colluding source can't carry the grant → **Ship-51** gate.
- **CurricuLLM / LgTS** (2409.18382, 2310.09454) — LLM emits ranked behavior rungs; a deterministic
  synthesizer builds the rollouts → **Ship-51** ladder synthesizer (keeps ground truth physical).
- **Self-reflection / iterative refinement** (2502.05605, 2411.00418) — after validate passes, have an
  independent LLM critique the metric's archetype ranking; 1–2 regen cycles → improves acceptance.
- **Joint-naming robustness (SMPL / URDF / MJCF)** — anchor on **functional role**, not names/indices →
  §4A resolver. Caveat: embeddings fail across differing kinematic-chain depths; prefer explicit canonical
  roles per robot.
- **Physics-constrained validation** (2308.12517, 2508.02194) — torque-saturation / joint-limit audits →
  **Ship-55** `realism.audit_rollout` verdict gates the grant.

---

## 7. Open decisions for the new window

1. **Required-roles declaration format** — structured module constant (`REQUIRED_JOINT_ROLES = [...]`) the
   generator must emit, vs. inferring roles by static analysis of the metric. (Recommend explicit
   declaration — auditable, and the validation gate becomes trivial.)
2. **Canonical role vocabulary** — define the role set per robot family (G1 biped, Go1 quadruped). Where does
   the per-robot manifest live (project `config.toml`? a `robot_manifest.py`?).
3. **Ship-51 ladder storage** — `metrics/<id>/.ladders/` vs a `meta.json` field; persist per-source `rho` +
   payload-hash for the audit trail.
4. **Gate thresholds** — `rho_min ≥ 0.5` + `agreement_fraction ≥ 2/3` (design rec); configurable vs fixed.
5. **Trust-score continuity** — keep `trust ≥ 0.7` so the 5 built-ins are byte-identical (makes Ship-52 a
   provable no-op for them).
6. **L2-alone-grants vs L3 booster** — confirm an uncalibrated task-derived metric (ladders disagree/time
   out) runs **observe-only** (not hard-rejected) — keep the run alive, just blind.
7. **Joint-resolution failure UX** — reject at launch with a specific message ("metric needs
   `swing_hip_pitch`; robot exposes no matching joint") vs. auto-retry generation with the missing role fed
   back.

---

## 8. Context pointers

- Existing design: [`DESIGN_autonomous_metric_eval.md`](DESIGN_autonomous_metric_eval.md) (L0–L5, Ships 50–55, UX, decisions).
- Source: `RewardSculptor/sculptor/eval/{metric_gen.py, metric_validate.py, metric_calibration.py, generated_metric.py, spec_metrics.py}`; prompts `gen_objective_metric.md`, `review_objective_metric.md`; adapter `sculptor/adapters/_mjlab_runner.py` (joint_names capture ~920-926). UI: `reward-sculptor-ui/backend/services/{run_manager.py, metric_store.py, sculptor_bridge.py}`, frontend `RunsTab.tsx`.
- This session: Ships 46 (`410bbdc`), 47 (`15fc9ea` — the metric hardening this builds on), 48 (`db3d20b`) on branch `ship-20-ux-revamp`. CONTEXT.md change-log entries 2026-06-15.
- The motivating case study: g1-kick-v3 `gen_005` scored a non-kicking walker ~0.59; re-running the (now Ship-47-hardened) validator rejects it. Memory: `project-g1-kick-diagnosis`, `reference-fitness-patience-vs-early-stop`, `project-autonomous-metric-eval`.
- Repro for §3A joint-ID evidence: load `~/.local/share/reward-sculptor/projects/g1-kick-v3/runs/iter_1/rollout/{trajectory.npz, mjcf_limits.json}`, score `metrics/gen_005/metric.py` and `spec_g1_kick` with correct vs shuffled vs empty vs foreign `joint_names`.

**Bottom line for the new window:** do **Phase J (joint identification)** first — it's the unaddressed
prerequisite and directly satisfies Sam's "ALWAYS correctly identify joints." Then **Ship-51 (task-derived
ladders)** is the highest-value step for "accurately measure any movement," because it lets a generated
metric earn steer-rights on novel tasks without hand-authored ground truth. Everything in Phases J–3 is
no-GPU/no-API (API-mockable); the GPU/API boosters (L5/L4) come last, with cost confirmed.
