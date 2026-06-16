# Design: Autonomous metric-quality evaluation — steer-rights for ANY novel task

**Status:** design only (2026-06-15). Not implemented — drafted while an overnight
sculpt run was live. Minimal first ship = **Ship-51** (below), no GPU needed.

## The problem (verified against source)

Today a generated objective metric can only earn the right to **steer** a run by
**calibrating** — Spearman ≥ 0.7 vs a **hand-authored** ground-truth built-in
(`g1_kick`/`g1_floss`/`g1_jump`/`go1_trot`/`cartpole`). The trust chain:

`validate` (smell test, `metric_validate.py`) → `review` (independent LLM,
`metric_gen.py`) → `calibrate` (Spearman vs built-in, `metric_calibration.py`) →
`steer_allowed` (`run_manager.py:78-94`, reads `meta.calibrated`).

For a **novel task** (anything not in the 5 families) `resolve_calibration_builtin`
(`sculptor_bridge.py`) returns `None` → calibration is **skipped** → `calibrated`
stays `False` → the metric runs **observe-only forever** (computed + charted, never
drives the loop). This is circular: you need a ground truth to trust a metric, but
the whole point is to not hand-author one per task.

The non-degeneracy validator only proves the metric isn't *degenerate* (doesn't
reward stillness/flail/falling; discriminates *some* synthetic archetype). It does
**not** prove the metric measures *your* task.

**Goal:** a standardized, autonomous way to earn steer-rights on arbitrary/complex
missions, without per-task hand authorship — while keeping the firewall (never
silently steer an unvalidated metric).

## The two risks any solution must beat (or it's worse than observe-only)

### Circularity — same LLM authors the metric *and* its grader → trivial self-agreement
Three defenses, **all required** for a steer-grant:
1. **Independent-context generation (structural):** the grader (ladder author) is a
   *separate* LLM call with a different system prompt, given **only**
   `behavior_goal` + `robot_hint` — never the metric source. (This already matches
   the existing reviewer: `review_objective_metric.md` says "You did NOT write it.")
   Persist call timestamps/model-ids/payload-hash to `meta.json`; assert the
   metric source never appears in the grader payload.
2. **Cross-source agreement (statistical):** require ≥ K (K=3) *independently
   authored* evidence sources to agree — score the metric on each, take the
   **min** Spearman (not mean). One colluding author can't carry the grant.
3. **Optimization-outcome audit (empirical):** train against the metric briefly and
   judge the *resulting behavior* (not the metric's self-description). The only
   truly non-circular ground truth. GPU-gated capstone.

### Goodhart — looks great offline, gameable when optimized
- **Reuse** the hard-won existing defenses verbatim: the `upright_flail`/`chaotic`
  non-degeneracy negatives (`metric_validate.py`, added Ship 36/41 after the G1
  stand-and-flail hack) and the `spearman()` round+std guard (`metric_calibration.py`,
  added after a `joint_pos`-magnitude metric spuriously hit rho≈1.0 at ~1e-7).
- **Adversarial archetype expansion (new):** ask the independent author to also
  propose 3 "gaming policies" for this goal — they become extra negatives the
  metric must score below. Generalizes the hard-coded negatives to any task.
- **Optimization-outcome audit (new):** a metric whose trained optimum is a
  torque-saturated / joint-limit-exploiting policy is caught by the **existing**
  `realism.audit_rollout` (`verdict ∈ {ok,mild,severe,unknown}`) → `severe` revokes
  the grant.

## The layered trust pipeline + one standardized trust score

A single scalar `trust ∈ [0,1]`, computed identically for every task (built-in or
novel) → one threshold, no per-task tuning:

```
trust = w_cal·CAL + w_evi·EVID
CAL  = clip((rho_min − 0.5) / 0.5, 0, 1)          # rank evidence
EVID = gate_pass(validate) · gate_pass(axioms) · agreement_fraction
steer-rights  ⟺  trust ≥ 0.7  AND  axioms pass  AND  agreement_fraction ≥ (K−1)/K
```
`rho_min` = min Spearman across all available rank sources (the built-in ladder for
the 5 families; else the K independent task-derived ladders). **For the 5 built-ins
this reduces to today's rho ≥ 0.7 — identical grant decisions (no regression).**

| Layer | Proves | Cost | Plugs into |
|---|---|---|---|
| **L0 Validate** (exists) | safe, physical-only, deterministic, bounded, non-degenerate | offline | `metric_validate.validate_generated_metric` |
| **L1 Task-agnostic axioms** (new) | universal invariants: no reward for stillness/extremes; monotone-in-uprightness; responds to the goal's named quantities | offline | new gate beside `validate`, reuses `_archetypes()` |
| **L2 Independent task-derived ladder** (new — **the unblocker**) | metric ranks an *independently-authored* competence ladder monotonically; K sources agree | K LLM calls (~20s ea), no GPU | new `metric_calibration.calibrate_task_derived()` + prompt `gen_competence_ladder.md` |
| **L3 Cross-metric consensus** (new, opt) | the ranking is corroborated by K independent metrics + adversarial archetypes | few LLM calls, no GPU | new `metric_consensus.py` |
| **L4 VLM grounding** (new, opt, cost-gated) | numeric metric correlates with a general VLM's "matches goal?" rating over rollout keyframes | VLM API; reuses *existing* keyframes (free) | reuses `diagnose._encode_image` + the keyframe sampler |
| **L5 Optimization-outcome audit** (new, opt, cost-gated) | training against the metric yields legit (not hacked) behavior | short GPU run | `realism.audit_rollout` + diagnoser |

**L0+L1+L2 are the steer-rights minimum for novel tasks — all offline, no GPU.**
L3 raises agreement; L4/L5 are budget-gated confidence boosters for high-stakes
metrics. Every layer writes its verdict to `meta.json` — nothing silent.

**Composition insight:** the grant still flips at `metric_store.calibrate()` →
`calibrated=True`, and `steer_allowed` is **untouched**. We don't change the
firewall gate — we widen the set of inputs that can legitimately set `calibrated`.

## Phased ships (smallest-valuable-first, flag-gated, GPU-free where possible)

All offline ships are unit-testable with a **mocked LLM client** (the codebase
already injects `client` into `generate_objective_metric`) + synthetic ladders.

- **Ship-50 — Task-agnostic axioms (L1).** Universal-invariant gate; pure-numpy over
  `_archetypes()`. Hardens *every* metric immediately. No GPU/API.
- **Ship-51 — Independent task-derived ladder (L2): THE MINIMAL UNBLOCKER.** New
  `gen_competence_ladder.md` (author sees the goal only); `calibrate_task_derived(
  metric_path, goal, robot_hint, *, k_sources=3)` generates K ladders in fresh
  contexts, scores the metric on each, returns `rho_min` + per-source scores (same
  record shape as `calibrate_metric`). In `run_manager._generate_at_launch`, replace
  the `metric_calibration_skipped` branch with this path. Flag default off → on
  after a manual audit. **After this ship, a novel task can earn steer-rights.** No GPU; API mockable.
- **Ship-52 — Trust-score unification.** `metric_store.calibrate` writes the
  standardized `trust` + per-layer breakdown; built-ins produce byte-identical
  decisions (regression test). No GPU.
- **Ship-53 — Cross-metric consensus + adversarial archetypes (L3).** Raises
  `agreement_fraction`; folds author-proposed gaming archetypes into the negatives.
- **Ship-54 — VLM grounding (L4). COST-GATED.** Reuses existing keyframes; general
  model only. API budget.
- **Ship-55 — Optimization-outcome audit (L5). COST-GATED.** Short train-against-
  metric → `realism.audit_rollout` + diagnoser; hack → revoke grant. GPU budget.

## UX

The Runs "Objective metric generation" card already renders generate→validate→
calibrate with three outcomes (`RunsTab.tsx`), incl. the dead-end muted "No matching
built-in ground truth — observe-only." Changes:
- Replace that dead-end: emit `metric_calibration_started/_done` for the
  task-derived path too (with `method:"task_derived"`). Card shows "Calibrating vs
  3 independently-generated competence ladders…" → green "Trust 0.81 (3/3 agree) —
  steering" or amber "Trust 0.42 — observe-only (ladders disagree)."
- **Observe-only is always specific, never a shrug** — every observe state names the
  layer that failed + the number.
- Accepted block gains an expandable "evidence" line (per-layer pass/fail, `rho_min`,
  `agreement_fraction`). Extend the existing `calib` event type + reducer — no new
  component. No new terminal/CLI step.

## Open decisions (recommendations)

1. **K (independent ladders):** K=3, single model (Opus), fresh contexts.
2. **Threshold continuity:** keep `trust ≥ 0.7` so the 5 built-ins are byte-identical
   (makes Ship-52 a provable no-op for them).
3. **Ship-51 default:** ship `off`; flip on only after a manual audit on 3–5 real
   novel goals confirms no known-bad metric is granted.
4. **L2 alone grants steer** (offline) — L3 is a booster, not a gate (keep the
   minimal path cheap; else novel-task steer needs ~8 LLM calls).
5. **L4/L5** flagged + cost-gated + reserved for high-stakes metrics — not on the
   default novel-task path.

## Minimal first ship — confirmed reuse points

**Ship-51** reuses, verbatim where possible:
- `spearman()` (goal-agnostic; keep its anti-spurious-correlation guards).
- `calibrate_metric()` as the template (load metric → score over ladder → spearman
  → return record); swap the hand-authored `_ladder(builtin)` for K
  LLM-authored ladders, return `rho_min`.
- The ladder data contract `list[tuple[arrays, behavior, meta]]`: the LLM author
  emits a *competence axis + rung descriptions*; a deterministic synthesizer (NOT
  the LLM) builds the numpy rollouts (graded `root/joint_vel/joint_pos/gravity`,
  like the existing `loco`/`g1_kick`/`g1_jump` builders) — keeping the ground truth
  physical and the author blind to metric internals.
- Grant write stays at `metric_store.calibrate()`; `steer_allowed` untouched.

### Critical files
- `RewardSculptor/sculptor/eval/metric_calibration.py`
- `RewardSculptor/sculptor/eval/metric_validate.py`
- `RewardSculptor/sculptor/prompts/gen_competence_ladder.md` (new)
- `reward-sculptor-ui/backend/services/run_manager.py`
- `reward-sculptor-ui/backend/services/metric_store.py`
- `reward-sculptor-ui/frontend/src/components/RunsTab.tsx`
