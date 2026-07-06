# Mission metric granularity — decision record

2026-07-06. Decides the objective-metric granularity for the mission
system: one metric per run, one per stage, or one overall metric
guiding decomposition. Companion to `RESEARCH_GAP_ANALYSIS.md` and
`LAWS_OBJECTIVE_METRIC.md`.

## Decision

**One fresh, trust-gated objective metric per mission stage, generated
at decomposition time from the stage's own goal text; the mission-level
metric (when provided) remains as (a) context the decomposer sees and
(b) the fallback for any stage whose generated metric fails the trust
pipeline.**

This is not a third option so much as the composition of two existing
mechanisms with one new one: Ship 34 gave missions a uniform metric,
Ship 38 gave stages an override slot (`Stage.steering_metric`); this
decision makes the override slot the *default-populated* path, filled
by `generate_objective_metric` per stage rather than left empty.

## Why per-stage (and not per-run)

1. **The composition-Goodhart record.** The measured failure mode of
   this project's flagship task is a run-level metric scoring behavior
   it wasn't written for: `spec_g1_jump` scored sit-bobbing 0.215
   (audit loop 4d); the E2E runs' best-pair was a stable stander
   because upright-credit dominated a jump-scoped score. A mission that
   chains "stand → crouch-launch → land" under one jump metric pays
   stage 1 nothing (it can't leave the ground yet, so early-stop and
   extension decisions run on noise) and pays stage 2 for regressions
   toward standing. Per-stage metrics keep the measurement aligned with
   what the stage is actually supposed to produce.

2. **The trust pipeline assumes goal-scoped calibration.** L2
   competence ladders are synthesized metric-blind *from the goal
   text* (`metric_calibration.py`); a stage-goal-scoped metric gets a
   stage-goal-scoped ladder, so agreement actually tests what the stage
   trains. Calibrating a whole-mission metric against a stage's ladder
   would structurally depress agreement for early stages (the metric
   scores an ability the stage doesn't teach) — i.e. per-run metrics
   are not just worse steering, they make the trust layer's verdicts
   incoherent at stage boundaries.

3. **Goal A / Goal B decisions are per-stage decisions.** Early-stop
   (criterion holds for N iters) and budget extension (metric improving
   despite criterion failure) both read the stage's metric trend
   (`sculpt.py::_scan_iter_metric_history`). A metric that can't move
   during this stage makes extension impossible and early-stop
   arbitrary.

4. **CurricuLLM precedent.** The curriculum line this system's mission
   design follows generates per-substep rewards *and* per-substep
   evaluation; the mission system already accepted that argument for
   rewards (each stage sculpts its own) — evaluation granularity should
   match the unit of selection, and the unit of selection is the stage.

## Why keep the mission-level metric at all

- **Decomposition guidance:** the decomposer already receives the
  available-metrics block (Ship 38); a mission-level metric is the
  clearest statement of "what the whole thing is for" and shapes stage
  boundaries. Generation-time context, not steering.
- **Fallback:** stage-metric generation is LLM work gated by L0–L2 —
  it can be rejected (`accepted=False`). A rejected stage metric must
  not blind the stage when a serviceable mission metric exists.
  Fallback order: stage metric → mission metric → blind (criterion
  only), which is exactly the existing `steering_metric or
  fitness_metric` resolution — unchanged.

## Costs and their answers

- **LLM cost:** ~1–2 min and a handful of calls per stage
  (generation + review panel), at decomposition time, once per
  mission. Negligible against a single training iteration; disable
  with `--no-stage-metrics` / `gen_stage_metrics=false`.
- **Cross-stage comparability:** per-stage metrics are NOT comparable
  across stages — `best_metric` on one stage means nothing next to
  another's. This was already true under Ship 38 overrides; the
  mission-level report should compare stages only on criterion
  satisfaction + iterations used, never raw metric values.
- **Trust surface multiplies:** each generated metric is a fresh
  gaming surface. That is what the trust pipeline is *for*; stage
  metrics go through the same L0 gates + review panel as any generated
  metric, and stage-scoped goals are narrower — narrower goals produce
  more checkable metrics, not less.

## Implementation (landed with this decision)

- `sculptor/mission_metrics.py` — `generate_stage_metrics(mission, …)`:
  per pending stage without a `steering_metric`, generate into
  `<mission_dir>/stages/<name>/metric/`, set `steering_metric` to the
  mission-dir-relative path on acceptance; record rejections in the
  return report and leave the stage on the fallback.
- Stage-metric refs may be *mission-dir-relative* paths;
  `mission_run`'s fail-fast resolution anchors relative refs at the
  mission dir before `resolve_fitness_fn`.
- CLI: `sculpt mission-init --stage-metrics/--no-stage-metrics`
  (default ON when the adapter is metric-capable); backend
  `POST /missions` gains `gen_stage_metrics` (default true) and the
  decompose job streams per-stage generation progress events.
