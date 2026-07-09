# Reference-Trajectory Program — Build Log & Design Decisions

Autonomous execution of `REFERENCE_TRAJECTORY_PLAN.md` / `R1_BUILD_SPEC.md`,
started 2026-07-09. Orchestrator: Claude Fable 5. This file records every
design decision made BEYOND the written plan, with rationale — read it next
to the plan, not instead of it.

## Decisions (chronological)

### D1. LLM roles: all-fable-5 with ONE exception (commit be45576)
All ROLE_DEFAULTS roles moved to `claude-fable-5` per Sam's instruction,
EXCEPT `calibration`, kept on `claude-opus-4-8`. Reason: the model-disjoint
law (RESEARCH_GAP_ANALYSIS §3.5) — the competence-ladder/gaming-archetype
author must not share the metric author's blind spots, and metric_gen is now
fable-5. Making calibration fable-5 would have silently voided an
adversarial-verification invariant. The review panel (opus/sonnet/haiku)
remains author-disjoint from metric_gen unchanged.

### D2. Retrieval: synonym-bridged matches discounted 0.5, once per group (a9f0602)
Found against the REAL 301-clip index (not the fixture): double-sided synonym
expansion made query "ground" worth ~4 rare-token weights via the lie-group,
outranking fallAndGetUp for THE acceptance query. Literal overlap = full IDF
weight; a synonym group bridging query↔row with no literal overlap = one hit
at 0.5× its strongest present member. Lesson recorded: fixture-passing ≠
real-corpus-passing for ranking code.

### D3. speed×0.25/×4 perturbations recorded but NOT gated (65bcfdb)
Plan §5.2 lists speed variants among hard negatives, but its gate sentence
only binds reversal/freeze/shuffle + truncation monotonicity. A kinematic
metric may legitimately score a time-scaled completion high (a 4× faster
get-up still gets up); gating speed would false-reject honest metrics.
Scores are recorded per reference for later analysis; revisit in the R5
adversarial suite where dynamics-implausibility can be judged properly.

### D4. `reference_anchored` nondegeneracy branch (65bcfdb)
Plan said references "replace the vacuous/selectivity-probe fallback" for
family-None goals. In reality the probe fallback only covered the strict
near-zero-battery case; a get-up metric typically dies in the general
`spread < spread_min` branch instead — the reference gates would never have
fired for the exact scenario the plan targets. Added a third branch: when
references are attached AND family is None AND the fixed battery is
near-zero OR near-constant, nondegeneracy defers to the reference gates
(real positive vs degenerate anchors + margin). No-references behavior is
byte-identical to before.

### D5. Zero-run missions listed in Training sidebar (e2586ef)
A decomposed-but-never-trained mission was invisible in the UI (mission
groups derived only from runs), so the reference picker/curriculum dialog
was unreachable for exactly the mission that needs pre-training reference
attachment. partitionRuns now emits mission groups with `stages: []`.
House rule: every feature UI-reachable.

### D6. R1 acceptance interpretation
With the LLM rerank ON, top-1 for "get up off the ground" is
`a10_lie_to_crouch` (conf 0.95) with fallAndGetUp segments at ranks 2-5 —
defensible (lie→crouch IS rising off the ground). The LOCKED deterministic
unit-test requirement (fallAndGetUp top with no LLM) is what D2 enforces,
verified against the real index and in the browser.

### D7. Opus audit of the R1+R2-a batch (be038c5)
Verified clean: quaternion math (hand-checked vs mju_rotVecQuat), path
traversal (no bypass constructible), reference gates don't weaken the
no-reference path, RunsTab stages:[] has no crash path, 409 guard covers
all four save_mission-writing job kinds. Fixed: speed() velocity
factor²-scaling (now pinned by test); save_mission atomicity comment.
Open item delegated: the synonym discount (D2) regressed "walk forward"/
"jump high" style queries — concept-boost rework in flight with a locked
6-query acceptance suite against the real index.

### D8. Truncation gate: plateaus allowed, inversions and non-discrimination rejected (ae4a38f)
The FIRST real reference-anchored certification (fable-authored
torso-righting metric vs fallandgetup1_subject1--seg00) failed
reference_monotonicity on an honest metric: the segment lies still for
its first half, so trunc_25 == trunc_50 == 0.0 — a tie the strict > chain
rejected. Real mocap paces unevenly; partial-completion scores plateau.
New gate: non-inversion (t25 <= t50 <= t75 <= full, 1e-6 tolerance) AND
discrimination (full >= t25 + spread_min). Constant metrics still die
(no discrimination), inverted grading still dies (inversion). Also
dropped "truncations >= degenerate anchor" — a prefix with zero goal
progress legitimately scores 0. Follow-on watch item: R2-b's calibration
ladder Spearman can now see ties at 0 on the same clips; if acceptance
shows rho degradation from tied rungs, switch to spearman_midrank there.

### D9. RSI mechanism (5623d90 + 58e0abd)
mjlab needed no fork: pose_range natively supports pitch/roll (unwired
until now); per-joint posture reset added as a NEW event function
injected through env_cfg.events (same pattern as the existing sunk
termination injection). Verified by an actual CUDA env reset: z 0.0999 m,
|pitch| 1.5708 rad, joints near the reference initial window. Scaffold
precedence: stage reference clip > project jump.npz > procedural jump.

### D10. Segment quality: get-up segments must START settled-lying (in flight)
Second live-certification finding: a fable-authored metric scored 0.000 on
seg00 because the SEGMENT starts standing (z=0.637) and contains the fall —
the R1 segmentation pads ±0.5 s and opens at down-interval onset. Measured:
only 26/~90 fallAndGetUp segments are true lying→standing spans. The metric's
start-supine gate was RIGHT to fail it. Response: (a) stage references
swapped to shape-verified segments (start z<0.35 settled, end z>0.55);
(b) segmentation refined to open at the settled low-still point with a
hard per-segment QC (start-lying + end-standing or reject) + `sculpt refs
resegment` CLI. Meta-lesson: reference SHAPE (start/end state) is part of
the exemplar contract, not just motion class — codify as QC, don't rely
on retrieval labels.

### D11. R3 verification lesson
The R3 worker's report claimed library registration, but the real library
had no retargeted clips (its proof ran under a tmp root). Orchestrator
re-ran the real proof: fallAndGetUp1 BVH → g1 (29 joints) + t1 (27 joints,
2-DOF elbows) in 39 s CPU, role resolution 8/8 both, registered + indexed
(303 rows). Plain H1 is blocked upstream (no LAFAN1-BVH IK config in GMR;
SMPL-X path auth-gated) — Booster T1 substituted as the second morphology.
Lesson: worker "done" claims about durable side effects get disk-verified
before commit, every time.

### D12. Gate-failure diagnostics + time-series hygiene (1fc50e4)
Third live-certification finding: a fable-authored metric scored 0.000 on
a verified-good segment for 3 straight retries. Root cause (found by
orchestrator tracing, invisible to the retry loop): np.convolve(mode=
"same") zero-pads boundaries — the smoothed tail of the rising height
signal collapsed below the lying threshold, so the episode's FINAL frames
classified as lying and zeroed the completion gate. Fixes: (a) when a
reference gate fails, the metric's own named sub-components on the full
exemplar ride the failure reason into the retry context (completion_gate
0.0 vs progress 0.93 is instantly diagnostic); (b) authoring prompt gains
TIME-SERIES HYGIENE (no zero-padded smoothing; boundary artifacts corrupt
exactly the completion frames). The reference gate CAUGHT a metric that
would mis-score every real rollout at episode end — the machinery working
as designed; the gap was feedback quality.

### D13. Re-segmentation results (30addd5 + data run)
Settled-start + hard QC re-segmentation of the six fallAndGetUp parents:
~90 padded segments → 34 accepted (97% true lying→standing; 12 honest
rejects logged, all "does not end standing"). All 34 have previews.

### D15. Archetype windows aligned with segment QC (6e4089c)
First real get-up mission run scaffolded stage 1 with AIRBORNE RSI
(positive height offsets) despite a QC-verified get-up clip: _archetype's
0.5 s end-MEDIAN saw the subject's post-stand settle (0.52 m) and failed
the getup rule. Windows now max(0.5 s, 10% of clip), MEAN aggregation,
end threshold 0.55 — the exact contract the segment QC guarantees.
Meta-rule: every layer that classifies "is this a get-up" must share ONE
window/threshold convention or clips pass one layer and fail the next.
After the fix the runner applied the full lying reset:
RSI(z,pitch,roll,vz) + reset_joints_to_reference(29 joints) + sunk 0.0485.

### D16. fell_over termination vs lying reset (in flight)
With the correct lying reset live, ALL 2048 envs terminated 'fell_over'
at reset — mjlab's orientation-based fall termination fires on the task's
own start state. Fix: train-scope `fell_over_termination: false` emitted
by the get-up RSI derivation (sunk guard + time_out remain the enders;
falling after standing = legitimate retry experience). Also under
investigation: whether EVAL rollouts consume train-scope resets — for
get-up the lying start is the TASK, not curriculum; if eval resets
standing, the certified metric scores garbage on eval rollouts (scope
design question, evidence being gathered).

### D17. Stage-fixed eval reset for get-up stages (in flight)
Eval-scope recon (D16 agent, evidence _mjlab_runner.py:1541): fitness
rollouts apply shared scope ONLY — deliberate, so diagnoser-iterable
train knobs can't make per-iteration fitness incomparable. But for
get-up the lying start is the TASK, not curriculum: eval resetting
standing means the certified metric scores rollouts that never lie
down. Decision: stage-FIXED deterministic eval reset — derive_eval_reset
(range midpoints, vz 0, joint noise 0, fell_over off) written ONCE at
scaffold to <stage>/eval_reset.json, passed to the rollout path as an
explicit override applied after the shared-only env-spec. Never touches
the frozen shared scope; nothing the diagnoser can iterate; jump
missions byte-identical (None for airborne). Rejected alternatives:
reset keys in shared scope (breaks the frozen invariant with a
sub-invariant), reading train scope at eval (diagnoser edits would
shift eval conditions mid-mission).

### D18. root_only hard negative closes the displacement-only gaming class
Second Opus audit PROVED a pelvis-rise-only metric (time-height
correlation of the base, zero posture dependence) passed all three
reference gates — a levitating-supine policy would score 1.0. New
perturbation `root_motion_only`: the clip's root trajectory with every
posture channel frozen at frame 0. Gated in reference_negatives; present
ONLY when the clip carries posture channels (else it equals the original
and would falsely convict everything). Consequence, intended: height-only
metrics are now convicted on posture-carrying clips — get-up metrics MUST
read orientation. Universal rigor: gliding statues can't pass gait
metrics, root-pop can't pass jump metrics.

### Audit findings deferred (logged, not yet fixed)
- Tier-D spoofing (LOW, latent): calibrate_metric_against_reference's
  `tier` arg comes from caller/provenance (user-writable) — no production
  caller yet; MUST be wired to require a verified track.py feasibility
  certificate (tierD block + rollout hash) before §6 goes live in the
  mission pipeline.
- "fall down" retrieval quality (LOW): "fall" is concept-diluted inside
  the 7-member get-up synonym group; rare modifier "down" wins. Optional
  polish: own fall-group + regression test.

## Verified state after R1 + R2-a (2026-07-09)
- Library: 301 g1 clips (LAFAN1 40 + segments; fleaven ACCAD slice), all
  with preview.png, 0 unexplained rejects (199 rejects all QC-reasoned,
  e.g. root_z below ground noise floor).
- Browser E2E: picker on standing-test get-up mission attaches
  fallandgetup1_subject1--seg00 (Tier K), persisted to mission.json.
- Gates: sculptor 1375 passed/1 skip; backend 509 passed ×3; frontend
  typecheck+build green.
