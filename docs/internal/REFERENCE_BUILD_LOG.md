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

### D18 follow-through: certified-metric re-validation
All four standing-test stage metrics were re-validated offline against
the strengthened gate. torso_righting's certificate FAILED root_only
(score 1.0 — displacement-influenced, Goodhart-able by pelvis-scooting;
it was actively steering the running mission). Mission stopped, metric
regenerated under the new gate (accepted, root_only 0.0), mission
relaunched (job_975bda9d06226ced) — the definitive R2 acceptance run:
lying RSI + posture event + fell_over removed + eval reset + all four
metrics D18-certified. Ops rule while a mission runs: NO sculptor-package
edits (iterations re-import the package; mid-run code drift corrupts the
experiment) and no backend restarts (kills the job).

### D19. Archetype is START-STATE only; reset anchor is absolute (live find #6)
Stage 1 (torso_righting) PASSED its success criterion from the lying
reset — first stage of the get-up mission trained and certified end to
end. Stage 2 then exposed two coupled defects: (a) its lie-to-crouch
reference (z 0.00→0.28, never stands) failed the getup end-condition and
misrouted to airborne — jump RSI, fell_over armed, NO eval reset (eval
resets standing ⇒ the certified started-low metric could never score);
(b) the getup offset anchored on the CLIP'S OWN END as "standing", so
even correctly-routed lie-to-crouch clips derived a crouch-height reset.
Fixes: archetype = start-window only (jumps start standing — that alone
separates them; documented gap: mid-height starts like crouch-to-stand
need a future mid_start class); offset anchored on absolute G1-class
standing 0.74 m (same-robot retargets transfer metres) with a 0.10 m
floor for ground-clamped source data. All four stage clips now derive
lying resets 0.10-0.24 m. Meta-rule (now thrice-earned): every
assumption about clip SHAPE must live in exactly one classifier with QC
enforcement, or the next clip shape silently breaks a consumer.

### D20. Overnight mission post-mortem: redecomposition drops the reference binding (job_2b0265c81e5c83b0)

Observed: the get-up mission "completed" with stage_success_rate 1.0, but
late iterations show the robot STARTING STANDING (root_h 0.79 m, pg_z −1.0
at frame 0), sometimes hopping. Reconstruction from the 176 MB execute log
+ per-stage env artifacts + rollout trajectories:

1. **The D17/D19 fixes WORKED.** Job started 2026-07-09 20:11 (after
   a9cb73a 19:18). Original stages 2-4 each scaffolded correct lying RSI
   from their attached clips (v0.json: negative height offsets to
   0.10-0.24 m absolute, pitch/roll offsets, 29-joint posture, fell_over
   removed, low sunk) AND stage-fixed eval resets (eval_reset.json
   written; runner logged "eval reset: reference-derived lying start").
   Trajectory frame-0 root_h 0.13-0.31 m on train AND eval. Verified.
2. **Stages 2-4 honestly FAILED their criteria** (criterion_not_met after
   3-5 iters each) — getting up is hard; that is the real research signal.
3. **Each failure triggered redecomposition, and `redecompose_stage`
   builds sub-stages with NO reference binding** (decompose.py ~1130: the
   Stage() call sets steering_metric + LLM-chosen needs_reference_rsi but
   never reference_clip_id/tier/confidence, and redecompose runs no
   retrieval pass — unlike decompose_task, which attaches refs).
4. **The scaffold fallback silently substituted the wrong task class**:
   needs_reference_rsi=True + reference_clip_id=None fell through the
   precedence chain (stage clip → project jump.npz → procedural jump,
   sculpt.py ~4590) to `procedural:jump` — reset offsets [0, +0.35] m
   ABOVE standing, vz ±2.4 m/s, sunk 0.5 m. That IS the observed
   "starts standing, hops". derive_eval_reset(jump) = None → no
   eval_reset.json → eval also standing. feet_under_crouch__r1_* got
   needs_reference_rsi=False from the LLM → no env spec at all → default
   standing reset.
5. **The last line of defense saw it and was overruled**: the certified
   reference-anchored metrics scored fitness 0.0 on every standing
   rollout (start-low gates working as designed), but stage success used
   the LLM trajectory criterion (root_height/upright/episode-length — no
   start-state clause), trivially satisfied from standing.
   stage_final_selection: criterion_pass=true, fitness=0.0 → "succeeded"
   ×9. Hollow mission "success".

Root cause: stage↔reference binding is not preserved across
redecomposition, and two silent fallbacks (procedural-jump RSI, default
standing reset) plus start-state-blind criteria let the wrong start
state masquerade as success. Fixes (D21): (a) sub-stages inherit the
failed stage's reference binding; (b) needs_reference_rsi with no
resolvable clip must not silently fall back across task class; (c) a
MECHANICAL start-state gate — when a stage has eval_reset.json, success
requires the eval rollout's frame-0 state to match it within tolerance.

### D20a. GPU reset probe: two of four derived lying resets are PROPPED (scripts/probe_start_pose.py)
2-env CUDA probe, 25 zero-action settle steps, per-stage renders
(reports/start_pose_probe/). feet_under_crouch and drive_to_stand rest
stably from frame 0 (z drop ≤ 0.01 m). torso_righting resets in a
partial-inversion pose (legs cocked in the air) and collapses 0.234 m
while settling; getup_and_hold resets in a bridge pose and collapses
0.132 m. The derivation copies the clip's start-window pose verbatim;
mocap start frames can be mid-roll/unsettled even after segment QC
(z-stillness ≠ whole-body rest). Fix chosen (D21 batch): settle-then-
rederive at scaffold — run the derived midpoint pose ~0.5 s on CPU
MuJoCo, re-read (z, pitch, roll, joints) from the settled state, keep
range widths/noise; eval_reset gets the settled scalars. Every derived
reset becomes physically resting by construction.

### D22. Hardening sprint close-out (2026-07-11, commits ec38efc..c3e8e53)
Both prior deferred findings CLOSED: Tier-D steer rights now require a
verified track.py certificate (rollout+live-clip sha256 chain, containment
check, tier resolved internally — spoofed provenance downgrades to
K/observe; wired into stage-metric acceptance non-fatally); "fall" got its
own synonym group (root cause: IDF dilution by common "up" in the shared
group) + a 20-query golden acceptance suite (fixture + real-index layers).
Also landed: D21 fixes (redecompose binding inheritance, fatal clip-load,
mechanical start-state gate); first-class promptable start_pose with
mid_start archetype, clip↔pose QC, settle-then-rederive (CPU MuJoCo,
floor-injected, divergence-guarded); reference signatures threaded into
diagnose+edit prompts; stageii fps recovery (+3303 clips, index 6014
rows) + multi-robot preview/ingest symmetry (T1 MJCF via GMR checkout);
match-density tie-break (recovered "lord of the dance pose" clips tied
real dance clips and won on id sort). Fresh-context Opus audit over the
batch: core math/logic verified clean; 7 findings, 6 fixed same-day
(posture check closes the near-standing gate blind spot; gate fails
CLOSED on unverifiable trajectories; scaffold enforces non-standing
start_pose independent of the persisted RSI flag; clip text fenced+capped
in LLM prompts; tierd_cert bypass kwarg removed; live-bytes staleness +
rollout containment), 1 fixed by the density tie-break.

### Deferred findings (logged, not yet fixed)
- Steer ENFORCEMENT unification: reference-calibration trust is computed
  and event-recorded in mission_metrics, but the live steer/observe gate
  lives in backend run_manager.py on a different data model (gen:<id> +
  calibrated bool) — wiring reference:<tier>:<source> labels into the
  actual training steer decision is its own reviewed increment.
- clip_id collisions at ingest (MEDIUM data-quality): slugify(stem)
  ignores the directory; GRAB's 10 subject dirs share filenames → 3929
  stageii files yielded 1220 unique ids, last-write-wins. Follow-up:
  prefix subject/subset into clip_id (pre-existing design, not new).
- UI robot asymmetries (5, from the ingest worker's recon): useReferences
  defaults robot="g1" with no caller override; picker has no robot
  selector (needs project→robot plumbing); attach endpoint doesn't
  cross-check clip robot vs project robot; convert.py foot-contact
  inference is G1-hardcoded (t1 degrades gracefully); g1_hands preview
  reuses plain-G1 MJCF.
- Stopword IDF (LOW): common words ("on","the") can carry high IDF and
  swamp real matches ("lying on the ground" → running_on_the_spot).
- Sit-content gap: zero sit/chair clips in the index; golden queries for
  sit categories are fixture-only until such clips are ingested.
- QC fps backstop is one-sided: the per-frame delta check catches an fps
  assumed 2x too LOW but not 2x too HIGH (and it flags, not rejects) —
  the per-subset-constant + dataset-author-code cross-check is the real
  safeguard.
- settle_reset convergence flag: max|qvel| criterion never reports
  converged=True within the 0.75 s budget on the real stages (residual
  limb oscillation); diagnostic-only, z stabilizes.
- Hold-goal stages vs freeze_end negative (structural, found live on
  g1-standing feet_under_crouch 2026-07-12): a stage whose goal is
  "reach X and HOLD it" keeps authoring hold-rewarding metrics; the
  freeze_end perturbation (already-in-end-pose-from-frame-0) then
  correctly scores 1.0 and the gate rejects — twice in a row, with the
  failure reason in the retry context. The gate is RIGHT (the metric
  must also require the reach); the AUTHORING prompt needs an explicit
  hold-goal rule: "when the goal includes holding a terminal state, the
  metric MUST gate on the start state being below/away from it".
  Designed fallback (stage steers by mission-level metric; criterion +
  start-state gate still enforced) is acceptable meanwhile.
- Decompose-time job hang (ops, 2026-07-12): a mission_decompose job
  hung 9 h at "generating per-stage metrics" (last LLM call 23:39, no
  timeout fired); jobs/stop cleared it and per-stage regenerate
  recovered. Follow-up: per-call timeout + job-level watchdog in the
  metric-generation loop.
- Decompose auto-retrieval quality at 6k clips: stage 1 attached NO
  clip and stage 4 attached the 168 s standing-start PARENT clip
  (segment-preference + confidence gating needed); verbose goal-text
  queries rank push-recovery clips over get-up clips (stopword/common-
  token weakness, same family as the stopword IDF item).

## Verified state after R1 + R2-a (2026-07-09)
- Library: 301 g1 clips (LAFAN1 40 + segments; fleaven ACCAD slice), all
  with preview.png, 0 unexplained rejects (199 rejects all QC-reasoned,
  e.g. root_z below ground noise floor).
- Browser E2E: picker on standing-test get-up mission attaches
  fallandgetup1_subject1--seg00 (Tier K), persisted to mission.json.
- Gates: sculptor 1375 passed/1 skip; backend 509 passed ×3; frontend
  typecheck+build green.
