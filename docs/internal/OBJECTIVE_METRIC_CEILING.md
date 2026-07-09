# Raising the objective-metric ceiling: reference trajectories for arbitrary motions

Design brief — 2026-07-09. Not an implementation plan; a map of the possibility space
grounded in current (2023-2026) research. Companion to
`RewardSculptor/sculptor/eval/metric_validate.py`.

## The ceiling, precisely

The metric validator (`metric_validate.py`) certifies a generated `compute_spec` by
scoring it against a **fixed battery of 7 hand-authored synthetic behaviors**
(`_archetypes()`: still, fallen, chaotic, active/walk, upright_flail, walker, +
family-specific jump/kick/floss positives) plus a generic goal-agnostic probe
(squat/bow/wave/twist). A metric is accepted only if it ranks a **positive exemplar of
the target motion** above the degenerate anchors (still/fallen/flail/chaos) with margin.

For a motion outside the five known families (kick, floss, jump, locomotion, cartpole),
there is **no positive exemplar of the target**, so the nondegeneracy gate finds no
signal and rejects every candidate. Get-up fails this way — and worse, its *start state*
(lying on the ground) IS the `fallen` degenerate anchor.

**So the entire problem reduces to one need: supply, generate, or stop-needing a positive
exemplar of the target motion — for any motion, on any robot — without losing the
non-gameability rigor.**

## The reframe that makes it tractable

A reference trajectory does not only *validate* a metric. A reference trajectory can *be*
the metric:
- **tracking error** — score = closeness of the rollout to the reference (DeepMimic).
- **motion-prior discriminator** — score = "does this look drawn from the reference
  distribution" (AMP).

So "reference-trajectory generator" and "objective-metric generator" are largely the same
problem. Solving the reference side solves most of the metric side. This is why building a
reference system is the high-leverage move.

## Three families of fix + two cross-cutting layers

- **Family I — supply the exemplar**: get real reference trajectories (databases,
  generation, video).
- **Family II — change the metric's form** so the exemplar plays a robust role
  (distributional / discriminator metrics instead of brittle per-frame tracking).
- **Family III — change validation** so it needs no per-motion exemplar, or generates its
  own adversaries (VLM judges + adversarial certification).
- **Cross-cut A — retargeting**: one motion → many robots (the cross-robot key).
- **Cross-cut B — physics tracking**: make generated/video references dynamically valid.

---

## Possibilities

### P1 — Reference-library retrieval (retrieve real mocap, don't invent) — FAMILY I
Pull curated motion clips from **AMASS** (~45h, 11k+ actions, SMPL) and **LAFAN1**
(BVH, dance/fight/jump/**fall-and-getup**) — and note **pre-retargeted-to-G1 datasets
already exist** (`fleaven/Retargeted_AMASS_for_robotics`, `openhe/g1-retargeted-motions`
on HuggingFace). For a stage goal, retrieve the k nearest clips by text label / embedding
(**HumanML3D / Motion-X** provide text↔motion pairs). The retrieved clip becomes the
**positive exemplar the nondegeneracy gate currently lacks**; the existing synthetic
battery stays as the negatives.
- **Plugs in**: add a retrieval step in `metric_validate.py`'s `family is None` branch —
  replace the generic probe with real retrieved references as the competent anchor.
- **Rigor preserved**: identical gate, just a real positive instead of a synthetic-absent
  one. Metric must still rank it above still/fallen/chaos with margin.
- **Cross-robot**: retarget via GMR (P-infra-A).
- **Effort**: days. **Unlocks get-up immediately** — it's literally in LAFAN1.
- **Limit**: only motions present in mocap corpora (huge, not universal).

### P2 — On-demand reference generation (the "reference generator" you imagined) — FAMILY I
For motions with no mocap match: **text→motion** (MoMask, OmniControl for joint-level
control) or **video→motion** (GVHMR, WHAM, TRAM — "here's a YouTube backflip") produces an
SMPL sequence; **retarget** (GMR) to the robot; **physics-track** (PHC / MaskedMimic) to
make it dynamically valid; use the tracked result as the exemplar. This is exactly a
text/video-to-certifiable-reference pipeline, buildable end-to-end from open parts.
- **Rigor**: generated motion is kinematically plausible but **NOT dynamically
  guaranteed** (foot-skate, ground penetration are the industry-wide weak point). It must
  pass through a physics tracker before being trusted as ground truth — never feed raw
  generator output in as the reference.
- **Cross-robot**: GMR + PHC are the backbone; one generated motion → 18 platforms.
- **Effort**: weeks. Covers the long tail P1 misses.

### P3 — Discriminator-form metric (AMP-style) instead of hand-authored `compute_spec` — FAMILY II
Instead of the LLM writing a scalar function, train a small discriminator on a **pool** of
reference snippets for the task; its output IS the objective metric. It scores "does this
transition look like the reference *distribution*", not "does it match one trajectory" — so
it generalizes across style/variation and **degrades gracefully with imperfect/retargeted
references**, which is the exact fragility that makes a single canonical reference brittle.
- **Rigor**: the discriminator is validated with the SAME battery philosophy (scores real
  references high, still/fallen/chaos low) plus adversarial audit (P6).
- This is the deepest but most robust shift; it collapses reference and metric into one
  object. ASE/CALM add skill-conditioning if you want to direct which skill it rewards.

### P4 — Distributional realism score (Fréchet Motion Distance / FVMD) — FAMILY II
A per-rollout distance to the reference distribution in a motion-feature space (velocity/
acceleration of keypoints). A "naturalness/reference-like" channel that complements the
goal-completion gate; **FVMD** correlates with human judgment of motion dynamics better
than raw FID and is directly computable on rendered rollouts.
- Best as a **secondary channel** ensembled with an interpretable completion metric, not a
  standalone (it measures realism, not goal-achievement).

### P5 — VLM-as-judge, no reference needed — FAMILY III
Render the rollout; ask a VLM to score / **prefer** it against the text goal (RL-VLM-F:
pairwise preferences are more robust than absolute scores). Zero-shot for arbitrary
motions.
- **Hard caveat**: VLM rewards are **demonstrably gameable** — false positives before task
  completion are a recurring (not edge-case) failure across 2025-26 papers. So NOT a
  standalone certifiable metric. Use it as (a) a **candidate/exemplar generator** (VLM
  picks which rollout most looks like the goal → a soft positive label when no mocap
  exists), or (b) a tie-breaker — always gated through your rigor.
- **Note**: your fixed-battery certifier is *better* than what this subfield ships;
  Eureka / Text2Reward / RL-VLM-F all report reward hacking as live and unsolved with no
  equivalent gate. Preserve the certifier as a differentiator.

### P6 — Adversarially-generated validator (upgrade the battery, keep the rigor) — FAMILY III
Replace the fixed 7 archetypes with an **adversary** (Adversarial Reward Auditing pattern):
train a "hacker" policy to maximize the candidate metric with degenerate behavior; reject
the metric if the hacker succeeds. This certifies non-gameability for **any** motion
without a hand-authored negative set per motion. Pair with a real positive (P1/P2).
- This is the direct answer to "tested with as much rigor as now" but generalizing beyond 7
  cases. Calibration regularization further reduces hackability.

### P7 — Hybrid ensemble (the pragmatic high-confidence design)
Retrieve-or-generate a reference (P1+P2) → use it BOTH as the positive to validate an
interpretable LLM-authored `compute_spec` (keeps human-readable metrics) AND to train a
discriminator (P3) as a second channel → certify via adversarial audit (P6) with the
existing synthetic battery as negatives → **require agreement** between the interpretable
metric and the discriminator (ensemble disagreement flags problems). Highest confidence,
most general; each piece independently shippable.

---

## Cross-cutting infrastructure

### Infra-A — Retargeting (the cross-robot key): GMR
**General Motion Retargeting** (ICRA 2026, MIT license) ingests SMPL-X / BVH / FBX / live
mocap / **GVHMR video output**, retargets in real time on CPU to **18 humanoid platforms**
(G1, H1, Booster, Fourier, …, 19-43 DoF), and flags infeasible motions rather than silently
failing. This is close to the "morphology-agnostic retargeter" you asked about — it's the
piece that makes any reference system work "not just for G1."

### Infra-B — Physics tracking (make references valid + a source of competent rollouts)
**PHC/PULSE** (universal AMASS tracker + 32-dim latent covering 99.8% of AMASS) and
**MaskedMimic** (NVIDIA — one controller that turns masked/partial goals into full-body
motion). Two uses: (1) convert kinematic/generated references into dynamically-valid ones;
(2) the tracked reference *on the actual robot* is itself a competent exemplar to validate
against. **BeyondMimic** (G1, sim-to-real, arbitrary minutes-long references incl. spins/
cartwheels) is the closest existing "any reference → deployable tracker" result — relevant
on the policy side.

---

## Staged roadmap (effort ↑, generality ↑)

- **Stage 0 (days) — mocap retrieval.** Ingest the pre-retargeted G1 datasets; add a
  retrieval step so any motion with a mocap match gets a real positive exemplar. Unlocks
  get-up, jumps, fights, dances **now**. (P1 + Infra-A.)
- **Stage 1 (weeks) — on-demand generator.** text/video→motion→GMR→physics-track pipeline
  for the long tail with no mocap match. (P2 + Infra-A/B.)
- **Stage 2 (research) — robust general metric.** Discriminator-form metric + adversarial
  validator + ensemble agreement — the durable, cross-robot, novel-motion ceiling-raiser.
  (P3 + P6 + P7.)

## Direct answers

- *Pull from research papers?* Yes — AMASS/LAFAN1/Motion-X ARE aggregated mocap from many
  labs; G1-retargeted versions exist on HuggingFace today.
- *Databases to pull from?* AMASS, LAFAN1, Motion-X, HumanML3D/KIT, 100STYLE (sizes/licenses
  above); GMR-retargeted G1 sets on HF.
- *How to get many reference trajectories?* Retrieve (P1) + generate text/video→motion (P2)
  + physics-track for validity (Infra-B).
- *Foolproof cross-robot reference generator?* The pipeline is buildable now
  (text/video → SMPL → GMR retarget → physics-track → per-robot reference), and GMR/PHC are
  the cross-robot backbone. "Foolproof" is a program, not a switch: generated refs aren't
  dynamically valid until tracked, and no single generator covers 100% of motions — hence
  the hybrid retrieve→generate→VLM-fallback stack.

## Honest caveats

- Physical plausibility of pure generative references is the industry-wide weak point —
  always route through a physics tracker.
- Coverage: retrieval covers the common case; generation covers the tail; neither is total —
  the ensemble is what makes it robust.
- Your existing certifier is ahead of the field's practice; the move is to *extend* it
  (real positives + adversarial negatives), not replace it.

## Key sources
- AMASS https://amass.is.tue.mpg.de · Motion-X https://arxiv.org/abs/2307.00818 · LAFAN1
  (Ubisoft) · HumanML3D/KIT
- G1-retargeted: https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics ,
  https://huggingface.co/datasets/openhe/g1-retargeted-motions
- Curated retarget recipe: https://arxiv.org/html/2601.23080
- Text→motion: MoMask, OmniControl · plausibility patch https://arxiv.org/pdf/2602.18199
- Video→motion: GVHMR https://github.com/zju3dv/GVHMR , WHAM, TRAM
- GMR https://github.com/YanjieZe/GMR , https://arxiv.org/pdf/2510.02252
- PHC https://github.com/ZhengyiLuo/PHC · PULSE https://github.com/ZhengyiLuo/PULSE ·
  MaskedMimic https://research.nvidia.com/labs/par/maskedmimic/
- BeyondMimic https://beyondmimic.github.io/ , https://github.com/HybridRobotics/whole_body_tracking
- AMP (adversarial motion priors); ASE/CALM https://arxiv.org/abs/2305.02195
- Fréchet Motion Distance https://arxiv.org/abs/2204.12318 · FVMD https://arxiv.org/html/2407.16124v1
- Eureka https://github.com/eureka-research/Eureka · Text2Reward · RL-VLM-F
  https://liralab.usc.edu/pdfs/publications/wang2024rlvlmf.pdf
- Reward hacking survey https://github.com/xhwang22/Awesome-Reward-Hacking · Adversarial
  Reward Auditing https://arxiv.org/pdf/2602.01750 · calibration
  https://arxiv.org/pdf/2510.03231
