# Reference-Trajectory System: Architecture & Execution Plan

2026-07-09. Companion to `OBJECTIVE_METRIC_CEILING.md` (the possibility survey).
This is the build plan: exact components, integration points, gates, trust rules,
phases, and acceptance criteria.

## 0. Objective

Make the per-stage objective metric **certifiable for arbitrary motions** (get-up,
air-twist, flip, …) on **any supported robot**, with rigor **equal or better** than
today's fixed-battery validator — by giving the system a supply of reference
trajectories and making them first-class citizens of generation, validation,
calibration, and training.

**Non-goals:** replacing the AST sandbox, the axioms layer, the review panel, or the
trust-gating architecture. Those stay. We extend what the gates can *see*, not how
strict they are.

## 1. The one architectural insight everything hangs on

A reference clip is **one artifact with four uses**:

1. **Validation positive** — the competent exemplar the nondegeneracy gate lacks for
   novel motions (`metric_validate.py` `family is None` branch).
2. **Calibration ladder** — truncations/degradations of the clip form a graded
   competence ladder, plugging into the SAME `metric_calibration.py` machinery that
   today only exists for the five built-in families (`FAMILY_TO_BUILTIN`). This is
   what grants steer-rights for novel motions **through the existing trust gate**.
3. **Generation grounding** — a kinematic signature extracted from the clip (e.g.
   "root_pos_z rises 0.12→0.72 m over ~1.8 s; projected_gravity_b[...,2] goes
   −0.1→−0.95; both feet gain contact by 60% phase") is injected into
   `metric_gen.py`'s prompt AND into decompose's success-criterion authoring, so the
   LLM writes metrics/criteria against real numbers instead of guesses.
4. **Training enablement** — `reference.py` already derives RSI env-init from a clip
   (`derive_rsi_train_keys` / `apply_reference_rsi`, currently jump-only). Generalized,
   the same clip initializes training from reference states. For get-up this is not
   optional: the env resets STANDING today, so a get-up mission is untrainable
   without reference RSI regardless of metrics.

Design consequence: build ONE reference subsystem with one canonical clip format and
provenance, consumed by four existing subsystems. No per-consumer formats.

## 2. Canonical representation (decided, not open)

### 2.1 Clip container — extend `sculptor/reference.py`, don't invent
Existing clip dict: `root_pos_z (T,)`, `fps`, optional `root_vel_z`, optional
`joint_pos (T,J)` + `joint_names`, validated by `validate_clip`. Extend with optional:
- `root_pos_xy (T,2)`, `root_quat_w (T,4)` (world orientation — needed to derive
  `projected_gravity_b`),
- `joint_vel (T,J)` (finite-diff via existing `_with_velocity` if absent),
- `contacts: {left_foot, right_foot} (T,)` (measured or inferred),
- `provenance` block (§2.3), `source_fps`, `retarget` block (robot, tool, error stats).
All optional fields degrade gracefully; `validate_clip` gains shape checks only for
fields that are present.

### 2.2 Validator-facing conversion: `clip_to_arrays(clip, meta) -> dict`
Converts a clip into the SAME `(T, E=1, J)` array dict the synthetic battery uses
(`ALLOWED_ARRAYS` in `generated_metric.py`: `joint_pos`, `joint_vel`,
`projected_gravity_b`, `root_link_pos_w`, `left/right_foot_contact`,
`left/right_foot_pos_b`). This is THE key integration trick: a converted clip is just
another battery entry — `metric_validate._score(fn, arrays, meta)` works unchanged.
- `projected_gravity_b` from `root_quat_w` (rotate world gravity into body frame).
- Kinematic-tier contacts inferred: foot below height threshold AND |foot vel| below
  threshold (standard heuristic) — flagged `inferred=True` in meta.
- Missing arrays are simply absent (metrics already must `arrays.get(...)`-guard).

### 2.3 Two fidelity tiers + provenance (trust depends on both)
- **Tier K (kinematic)** — retargeted clip converted directly. Cheap (CPU, laptop).
  Approximate contacts, no dynamics guarantee. Good enough for: retrieval, generation
  grounding, nondegeneracy positives, RSI derivation, kinematic ladders.
- **Tier D (dynamic)** — the clip is TRACKED in our own sim (mjlab) by a DeepMimic-style
  tracking run (§R4); the resulting rollout is a native `trajectory.npz` with ALL
  arrays real (true contacts, dynamics-consistent velocities) AND is a feasibility
  certificate on the actual robot. GPU-costed. Required for: calibration-grade
  certification (steer-rights), and as the trusted exemplar for discriminator training.
- **Provenance record** (stored beside every clip): source dataset/clip id OR
  generator+prompt OR video URL, license tag (AMASS/LAFAN1 = research-only),
  retarget tool+version+error stats, tier, inferred-fields list, content hash.
  No clip without provenance is ever used by a gate.

### 2.4 Library layout
```
~/.local/share/reward-sculptor/references/
  <robot>/<clip_id>/clip.npz + provenance.json + preview.mp4 (optional)
  index.jsonl        # one row per clip: id, robot, text labels, tags, tier, license
```
Project-independent (shared across projects), env-var `RS_REFERENCE_ROOT` override,
same pattern as `RS_SAVED_ROOT`.

## 3. Component architecture (new module `sculptor/refs/`)

```
sculptor/refs/
  library.py     # index, lookup, provenance, license guard
  ingest.py      # HF/LAFAN1/AMASS importers -> canonical clips (Tier K)
  retrieve.py    # stage-goal -> ranked candidate clips (keyword shortlist + LLM rerank)
  convert.py     # clip_to_arrays; kinematic signature extraction; QC gates
  perturb.py     # hard negatives + ladders from a clip (§5.2, §6)
  retarget.py    # GMR wrapper (R3): human/SMPL/BVH -> per-robot clip + feasibility flags
  generate.py    # (R4) text2motion / video2motion adapters -> SMPL -> retarget
  track.py       # (R4) Tier-D certification: tracking run in mjlab, rollout capture
```
Consumers touched (no rewrites, additive):
- `eval/metric_validate.py` — reference-anchored nondegeneracy (§5).
- `eval/metric_calibration.py` — reference-derived ladders (§6).
- `eval/metric_gen.py` — kinematic signature in the authoring prompt (§7).
- `mission_metrics.py` — pass retrieved reference(s) through the pipeline per stage.
- `reference.py` — clip schema extension; `derive_rsi_train_keys` generalization (§8).
- `decompose.py` — stage authoring sees available references (needs_reference_rsi,
  criterion grounding).
- backend/UI — reference picker + approval surface (§9).

## 4. Retrieval design (R1)

1. **Shortlist** (no new ML infra): keyword/BM25 over the index's text labels
   (HumanML3D/BABEL labels come with the datasets; our ingest stores them).
2. **LLM rerank + verdict** (existing `llm.py` role registry, one cheap call): given
   stage `goal_text` + top-20 labels, return ranked ids + `match_confidence` + a
   one-line justification. Below-threshold confidence ⇒ "no mocap match" ⇒ R4 path
   (generation) or family fallback.
3. **Human confirmation (UI, one click)**: the metric card shows the chosen clip
   (preview video) with Accept/Swap. Retrieval-mismatch is a NEW error surface
   (wrong exemplar certifies a wrong metric); a human glance at a 3-second clip
   closes it cheaply. Auto-accept only above a high confidence bar, always logged.

## 5. Reference-anchored validation (R2) — the get-up unlock

### 5.1 Positive anchor
In `validate_generated_metric`, when a stage has attached reference(s):
- add each `clip_to_arrays(clip)` to the scored battery as `reference:<id>`;
- **nondegeneracy passes only if** `score(reference) ≥ max(score(negatives)) + spread_min`
  — same margin discipline as today, real exemplar instead of none.
- The synthetic battery (still/fallen/chaotic/flail/…) REMAINS as the negative anchors
  and as universal sanity checks. The generic probe remains as fallback when no
  reference is attached.

### 5.2 Hard negatives synthesized FROM the reference (rigor upgrade, cheap)
`perturb.py` derives adversarial near-misses no fixed battery can provide:
- **time-reversal** (a get-up played backwards = falling; must score LOW),
- **freeze at start / freeze at end** (lying still forever; standing still without the
  transition),
- **segment shuffle** (right poses, wrong order),
- **speed x0.25 / x4** (implausible dynamics),
- **truncation at 25/50/75%** (partial completion — must score BETWEEN degenerate and
  full, monotonic; doubles as the ladder §6).
Gate: metric must rank full reference > truncations (monotone) > degenerates, and
must NOT reward reversal/freeze/shuffle above the degenerate anchor + margin.
This is strictly MORE rigorous than today's gate, and it is motion-specific for free.

### 5.3 What does not change
AST sandbox, array contract, joint-role resolution, determinism/bounded/
index-robustness, L1 axioms, review panel — untouched. Reference entries go through
the same `_score` path; a metric that crashes on a reference fails `bounded` as usual.

## 6. Reference-derived calibration ladders (R2) — steer-rights for novel motions

Today steer-rights require calibration against a hand-authored builtin ladder
(`FAMILY_TO_BUILTIN` — only 5 families). Add `calibrate_metric_against_reference`:
- ladder = §5.2 truncations + degradations (noise-injected joints, damped root motion)
  of the Tier-D reference, labeled with intended competence order;
- acceptance = same statistic as today (Spearman monotonicity ≥ existing threshold,
  same trust bookkeeping, same firewall in `metric_calibration.py`);
- trust tier recorded as `reference:<tier>:<source>` (§10) — mission_runtime's
  existing trust unification consumes it unchanged.
Tier-K-only calibration grants **observe** rights; **steer** requires Tier-D
(dynamics-real exemplar) — mirroring the existing "generated metrics observe until
calibrated" rule.

## 7. Generation grounding (R2)

`convert.py::kinematic_signature(clip) -> dict` — compact, numeric, LLM-readable:
phase segmentation (via existing `phase_keyframes`), per-phase root height/velocity
ranges, orientation trajectory, contact schedule, duration. Injected into:
- `metric_gen.py` authoring prompt ("the competent motion looks like THIS — write a
  metric that scores THESE numbers high"), which should measurably raise the
  first-pass acceptance rate;
- decompose's success-criterion authoring (criteria grounded in reference numbers,
  not guessed thresholds — directly attacks the criterion_not_met→replan loop the
  jump mission suffered).

## 8. Training enablement (R2, small but load-bearing)

Generalize `reference.py::derive_rsi_train_keys` beyond the procedural jump clip:
it already consumes the clip container; verify/extend range derivation for clips
whose START is non-standing (get-up: reset lying, sunk-height termination must NOT
fire at reset — the sunk-termination pairing logic needs a start-pose-aware guard).
Wire: a stage with an attached reference and `needs_reference_rsi=true` gets
`apply_reference_rsi` at scaffold time (mission_runtime already carries the flag).
**Without this, get-up cannot train no matter how good the metric is.**

## 9. UI surfaces (every capability UI-reachable — house rule)

- Mission stage card: "Reference: <clip name> [preview] [Swap] [None]" + tier badge.
- Metric card gains "certified against reference:<id> (Tier D)" line.
- New Mission dialog: per-stage reference auto-retrieval on decompose; the metrics
  step shows retrieval verdicts; a stage with no match offers "generate reference
  (R4)" or "proceed with family/blind fallback" — same explicit-choice pattern as
  stage_metric_required.
- Library page (later, R5): browse/import clips, licenses visible.

## 10. Trust taxonomy (extends, does not replace)

| Certification path                                  | Rights                        |
|-----------------------------------------------------|-------------------------------|
| Builtin-family ladder (today)                       | steer (unchanged)             |
| Reference Tier-D (mocap source) + §5/§6 gates       | steer                         |
| Reference Tier-K (mocap source)                     | observe → steer after task-derived calibration (existing L2 flag) |
| Reference Tier-D (generated/video source)           | observe → steer after in-loop agreement window |
| VLM-exemplar only (R4 fallback)                     | advisory only, never steer    |
| No reference, no family (today's dead end)          | blind fallback (unchanged, but now explicit + rare) |

## 11. Phases, effort, compute gates

**R1 — Library + retrieval (days; CPU only).**
Ingest `fleaven/Retargeted_AMASS_for_robotics` + `openhe/g1-retargeted-motions`
(pre-retargeted G1; zero retarget work) + LAFAN1 fall-and-getup subset; canonical
clips + index + provenance; `retrieve.py`; UI picker.
*Acceptance:* "get up off the ground" retrieves a real G1 get-up clip with preview.

**R2 — Validation + calibration + grounding + RSI (1-2 weeks; CPU + existing LLM).**
§5 reference-anchored nondegeneracy + §5.2 perturbation negatives; §6 ladders
(Tier-K first); §7 signature injection; §8 RSI generalization.
*Acceptance:* the standing-test mission's 4 stages each certify a metric against a
retrieved reference (no blind fallback), and stage 1 trains from a lying reset.
*This closes the actual get-up demo gap end-to-end.*

**R3 — Cross-robot retargeting (1-2 weeks; CPU).**
`retarget.py` wrapping GMR (MIT license): SMPL-X/BVH → any of its 18 humanoids;
joint-name mapping into our adapter's joint-role resolver; feasibility flags stored
in provenance; per-robot library caching.
*Acceptance:* same source clip yields certified references for G1 AND one other
morphology (e.g. H1) with role-resolution passing.

**R4 — On-demand generation + Tier-D tracking (weeks; GPU + API — COST-GATED).**
- `generate.py`: text2motion (MoMask) and video2motion (GVHMR) adapters → SMPL →
  GMR → Tier-K clip. Every generated clip carries `source=generated` provenance.
- `track.py`: Tier-D certification = a bounded DeepMimic-style tracking run in mjlab
  using our OWN training loop (tracking reward = pose/root/contact error to clip);
  success within tolerance ⇒ feasibility certificate + native-array exemplar;
  failure ⇒ clip flagged infeasible-for-robot (a *useful* verdict, not an error).
- VLM-exemplar fallback (= existing cost-gated Ship-54 L4 concept): when neither
  mocap nor generation matches, a VLM panel picks the most goal-like rollout as a
  SOFT exemplar — advisory trust only.
*Compute rule (standing agreement): GPU tracking runs and VLM panels are spend —
confirm with Sam before enabling by default.* (Aligns with Ship-54/55 cost gates.)

**R5 — Robust-metric upgrades (research track; GPU — COST-GATED).**
- AMP-style discriminator channel trained on the stage's reference pool (+ FMD/FVMD
  distributional channel); exposed as a SECOND metric column, ensembled with the
  interpretable compute_spec; disagreement beyond threshold ⇒ flag, never silent.
- Adversarial auditor (= Ship-55 L5 concept): short RL "hacker" runs that try to
  maximize a candidate metric with degenerate behavior; success ⇒ reject. Replaces
  fixed negatives with generated ones; run cadence cost-gated.
*Acceptance:* on a held-out gaming suite (existing L3 archetypes + new perturbation
negatives), ensemble false-accept rate strictly below current single-metric rate.

## 12. New failure surfaces introduced (and their mitigations)

| Surface | Risk | Mitigation |
|---|---|---|
| Retrieval mismatch | wrong exemplar certifies wrong metric | LLM confidence + one-click human preview/approval (§4.3); provenance shows the clip on the metric card |
| Retarget artifacts | exemplar physically wrong for robot | GMR feasibility flags; Tier-D tracking required for steer; QC gates (foot-skate/penetration stats) in `convert.py` |
| Generated-motion bias | plausible-looking but wrong dynamics | NEVER trust raw generator output; Tier-D tracking mandatory; `source=generated` capped at observe until agreement window |
| License leakage | AMASS/LAFAN1 are research-only | license tag in provenance; library guard refuses untagged clips; fine for this research project, revisit if commercialized |
| Discriminator gaming (R5) | learned metric is itself hackable | ensemble-agreement law with interpretable metric; adversarial auditor; calibration regularization |
| Clip-format drift | silent schema divergence | single container in `reference.py` + `validate_clip` as the only entry point; content hashes in provenance |

## 13. Sequencing & first actions

R1 → R2 unlock the get-up demo and are laptop-only; R3 makes it multi-robot; R4/R5
are the generalization horizon and are cost-gated. Recommended immediate order:
1. R1 ingest + retrieval (get-up clip visible in UI),
2. R2 §8 RSI (get-up becomes trainable),
3. R2 §5 validation (get-up metric certifies),
4. R2 §6/§7 (steer-rights + better generation),
then reassess before spending on R4 GPU tracking.

## 14. Open questions (tracked, not blocking R1/R2)

- Retargeted-HF-dataset joint ordering vs our adapter's joint names — verify at
  ingest with the joint-role resolver; may need a per-dataset mapping table.
- E-dimension: battery uses E=4 envs; references are E=1 — confirm `_score`
  broadcasting or tile the clip ×4 (trivial either way).
- Tier-D tracking tolerance thresholds (what MPJPE/root error counts as "feasible") —
  calibrate against the known-good jump reference first.
- Whether decompose should retrieve references BEFORE writing stages (stage
  boundaries informed by clip phases) — attractive, deferred to after R2.
