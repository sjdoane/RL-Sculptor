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

### D23. Live zero-fitness on a CORRECT sit-up: exemplar-scope mismatch, certified in (H1 confirmed)
Observed (g1-standing / starting-from-lying-flat-on-its, torso_righting iter 0,
job_c75bf9d56f091262): the policy rights the torso cleanly from the settled
supine reset; objective fitness 0.0. Offline recompute of the certified metric
on the rollout trajectory.npz reproduces the live event exactly (spec 0.0,
progress 0.0609). Per-channel: gate_upright_frac 1.0 with gravity_z_end −0.917
(more upright than the reference clip's own end window −0.46), started-low 1.0
(z_start 0.137), joint-motion 1.0 (0.33 rad/s). The zero comes ONLY from the two
height channels: gate_reached_035 = 0 (z_end 0.143 m < 0.35) and gate_rise_floor
= 0 (rise 0.035 m < 0.20), which also zero ch_height/ch_rise, so
per_env = gate x ch_min = 0 on every env.

Hypothesis outcomes: H1 CONFIRMED. H2 (settled-start mismatch) excluded — the
start gate passed from the settled reset (the authored metric abstains on
initial orientation and carries started-low via root height). H3 (time-locked
assumptions) excluded — this metric is window-based, no absolute-time terms.
H4 (plumbing) excluded — the metric computed cleanly offline; the runtime event
carried no per-channel components (iter_fitness has only scalar fitness), which
is a visibility gap (F4), not the cause of the zero.

Root-cause chain:
1. The stage goal (right the torso to sitting) is a SUB-PHASE of its attached
   clip fallandgetup2_subject3--seg00, a full lying-to-standing get-up
   (z 0.13 -> 0.55). In the clip itself the root stays ~0.15 m during the
   torso-righting phase; the rise to 0.5 m happens in the final ~1.6 s
   (drive-to-stand). A physically correct sit-up keeps the pelvis at ~0.14 m.
2. The mismatch entered UPSTREAM at decomposition: the stage's behavior_goal
   text already bakes in the full-clip number ("raising the root above
   ~0.35 m"), and the success_criterion demands (root_height > 0.35).mean()
   > 0.25 — both unmeetable by the goal behavior. The whole stage contract
   (goal text, criterion, metric) inherited full-clip scope.
3. Certification then LOCKED it in: reference gates score the FULL get-up as
   the positive and REQUIRE trunc_25/50 to score ~0 (reference_negatives) —
   and a sit-up is kinematically a truncation of the full clip. Under these
   gates, zeroing a perfect sit-up is not an authoring bug; it is the only
   way to PASS certification. The metric did what its exemplar told it.

Class statement: whenever stage_goal is a strict sub-phase of clip scope,
certifying against the full clip guarantees a correct stage rollout scores
like a punished truncation. Same family as the D22 deferred hold-goal /
freeze_end finding — both are "the stage's motion is not the clip's motion"
scope errors. Consequence for the live mission: stage 1 could never pass its
criterion honestly; every iteration steers blind (fitness pinned 0, dense
progress noise-floor) while the diagnoser chases the dead height kernel.

Fix plan (D24 batch): F1 phase-cropped stage references — select the
goal-aligned sub-span, certify the metric against the SUB-SPAN, derive RSI +
eval reset from the same sub-span (reset/cert/scoring agree on what the stage
motion IS); F2 certification scores the stage's ACTUAL settled eval start and
the authoring context carries eval_reset numbers; F3 completion-then-hold
synthetic positive (goal reached early, terminal state held — the exact live
rollout shape) gated MUST-SCORE-HIGH + no-absolute-time authoring rule;
F4 runtime contradiction detector (criterion-pass x near-zero fitness -> loud
event + UI flag) and fitness events always carry per-channel components (the
gate_upright 1.0 / gate_height 0.0 split makes this class visible at a glance).
Golden fixture: the live sat-up rollout is preserved at
tests/fixtures/torso_righting_satup/ and must score well above zero under the
fixed metric class (regression test).

### D24 W4: F1 wiring — reference sub-spans consumed everywhere
Built on the landed F1 core (spans.py's `crop_span`/`select_reference_span`,
commit b14708a) and its end-state-QC addendum (`_end_state_qc`, checking the
SNAPPED span's own end-window z-band + orientation-direction claim against
`expected_end` — new required LLM-response key; catches an over-extended span
that `_span_qc`'s start-only checks pass identically to a correct one).

Persistence: `Stage` gains `reference_span_start_s`/`_end_s`/`_confidence`/
`_method` (mission.py), auto-serialized like every other Stage field;
redecompose sub-stages inherit the clip (D21, unchanged) but never these four
— span re-selected per sub-goal.

Selection sites (all real LLM calls by default, `span_select` role):
decompose.py's `_attach_stage_references` (right after a clip attaches,
`_select_and_attach_span` helper) and `redecompose_stage` (per sub-stage,
new `robot_hint` param, wired from sculpt.py's redecompose call site); lazy
backfill in `mission_metrics.generate_stage_metrics`
(`_backfill_stage_reference_span`, runs once, reason recorded on the report
entry like `reference_load_error`) — this is how the existing g1-standing
mission gets a span without redecomposition.

One loader (D19 rule): `mission_metrics.load_stage_reference_clip(stage,
robot) -> (clip_id, clip, span_meta) | None` — crops via `crop_span` when
span fields are persisted, else returns the full clip unchanged.
`_load_stage_reference` now routes through it (fixes cert + calibration +
the metric-authoring prompt's REFERENCE MOTION SIGNATURE block in one
place); sculpt.py's `_resolve_stage_rsi_clip` (RSI/eval-reset derivation)
and the reference_signature.json write (block 2.6, gains an optional
`"span"` key) do too. Proven to matter on the real torso_righting_satup
fixture: `derive_reference_reset`'s pitch/roll offset is `start_window -
end_window`, and the [0, 8.5]s span's end window (mid-recovery) measurably
disagrees with the full clip's (true standing) — RSI/eval-reset derived
from the wrong one would silently diverge from what certification scored.

Live-mission repair (backfill + criterion re-authoring for the existing
g1-standing mission) and the free-text `Stage.success_criterion` field
itself are explicitly NOT re-grounded by this increment — the criterion
Python boolean is authored in the SAME LLM call as goal_text, before any
per-stage clip is known, so it can't be retroactively grounded without a
decompose architecture change; the trust-gated generated METRIC (which
`generate_stage_metrics` now grounds in the cropped signature) is the
channel this increment makes span-aware, per the D24 spec's explicit
"criterion re-authoring rides decompose only ... else record as manual
TODO for next mission."

Verification hazard found and fixed live: `decompose_task`'s default
`attach_references=True` resolves the REAL on-disk reference library when
a test doesn't override `RS_REFERENCE_ROOT` — pre-existing, but harmless
before this increment (no LLM call downstream of retrieval). Wiring span
selection in made it a genuine live-network hazard: an actual outbound
HTTPS connection was caught mid-test-run. Fixed with a module-scoped
autouse fixture in test_decompose.py isolating `RS_REFERENCE_ROOT` by
default (mirrors conftest.py's `_isolate_shared_kg` for the KG graph),
plus explicit `select_reference_span` stubs in the handful of
test_mission_run.py/test_mission_metrics.py tests that set
`reference_clip_id` directly.

Gates: sculptor 1798 passed / 2 skipped (both pre-existing/expected — jax
unavailable, the sibling F2/F3/F4 worker's not-yet-landed golden metric
snapshot); backend 519 passed; frontend typecheck+build green.

### D25. Live repair of torso_righting: four regen runs, three NEW classes closed, spec 0.0 -> 1.0
Running the D24 pipeline against the real g1-standing mission was itself
the decisive test — each regen run exposed a class no offline test had:

1. **Regen #1 — empty LLM response** (06b09c2): span_select's 1024-token
   budget was consumed ENTIRELY by fable-5's thinking block (usage showed
   output_tokens == 1024, zero text blocks) -> '' -> cryptic
   parse_error:JSONDecodeError char 0, silent full-clip fallback. Budget
   -> 4096 (sized like retrieve 2048 / decompose 8000); empty text is now
   its own retryable `empty_response` infra reason, pinned by test.
2. **Regen #2 — coherently-wrong span from contaminated goal_text**
   (d6c5cec): the selector picked the RISE phase (5.3-9.77 s, conf 0.72)
   for the sit-up goal because the stage's own goal_text contains the
   blind-decompose invention "raising the root above ~0.35 m" — the
   end-state self-consistency QC passed (the LLM honestly described the
   span it chose; it chose from a defective description). ALSO: the
   criterion re-grounding returned components.get('righting_progress',
   1.0) — a FAIL-OPEN default vacuously deleting a criterion leg; the
   mechanical check now evaluates every components conjunct with an
   EMPTY components dict and rejects if it passes. Systemic fixes:
   decompose rule 11 / redecompose rule 6 — goal_text is QUALITATIVE (no
   invented numbers; numbers only if copied from a reference signature;
   state what the stage does NOT do). Live mission's goal_text repaired
   in place (original in mission.json.pre_d24_repair.bak).
3. **Regen #3 — fast completion vs start-window reads** (15a27f3): with
   the CORRECT span (0-8.1 s, conf 0.85), grounded thresholds, and the
   eval-start numbers in context, the freshly certified metric STILL
   zeroed the live rollout (progress 0.96, spec 0.0): its started-away
   read was a 0.5 s window MEAN and the policy completes the 8.1 s human
   righting span in ~0.5 s (16x) — the completed state leaked into the
   "start" window. New positive `fast_completion` (speed x16 then hold
   x19: motion ~5% of trajectory, the exact live profile), REQUIRED gate
   `reference_fast_completion` ONLY for reach-and-hold-shaped references
   (`ends_settled`, one classifier per D19); pace-sensitive clips stay
   record-only with an explicit abstain (D3's false-reject protection
   preserved where it binds). FAST-COMPLETION authoring rule: start
   reads = earliest frames or the EVAL START STATE numbers.
4. **Regen #4 — CLOSED.** Metric accepted with ALL SIX reference gates
   green (nondegeneracy, monotonicity, negatives, complete_then_hold
   x1+x24, settled_start, fast_completion — every positive at 1.0);
   criterion re-grounded to achievable numbers (root_height > 0.12,
   fail-closed component default); span 0-8.1 s persisted. The metric
   scores the live sat-up rollouts **spec 1.0 / progress 1.0** (iter_0
   fixture AND iter_1 full 64-env) — 0.0 this morning. Golden regression
   active: tests/test_torso_righting_regression.py pins as-shipped 0.0
   AND fixed >= 0.5 (metric_fixed.py + meta_fixed.json +
   stage_record_fixed.json snapshotted in the fixture).

Meta-lesson (fourth-earned): synthetic batteries converge only when
their variants BRACKET the live operating envelope — hold length (x24 >
realistic 19x), completion speed (x16 = observed), start state (settled
scalars) — envelope-bounded gates, not arms races. And offline-green is
not live-green: every one of the three classes above shipped through a
fully green suite and was caught only by running the real pipeline
against the real mission.

### D26. Fresh-context Opus audit of the D24/D25 batch: 6 confirmed findings, all fixed same-session
The audit's headline: the individual gates were strong, but EVERY span-
selection failure mode funneled to full-clip certification — the exact
D23 configuration the batch exists to eliminate. Findings and fixes:
- **H1 (proven exploit)**: the end-state QC was vacuously satisfied by a
  loose z_band ([0.05,0.80] and [0,1] both ACCEPTED the over-extended
  0->11.2 s span). Fix: band width capped at max(0.15 m, 35% of the
  clip's z-range) — an uncommitted claim is a rejection; audit's exact
  table pinned (f0c5905).
- **H2**: unresolved span declines (low_confidence/qc_reject/crop_error)
  fell back to full-clip certification AND the declined marker pinned it
  permanently. Fix: fail CLOSED — reject the stage's metric generation
  loudly (stage uses the mission-level fallback; criterion + start-state
  gate verified unaffected); only the LLM's affirmative whole_clip
  verdict certifies full-clip; the explicit per-stage regen endpoint
  clears declined markers so a user retry re-runs selection.
- **H3 (proven live)**: scaffold idempotency guards reused STALE
  full-clip artifacts after the span changed — the repaired stage's
  on-disk eval_reset had a SIGN-FLIPPED roll offset vs the span-derived
  preview (+1.265 vs -0.370), and the stale reference_signature would
  have steered the diagnoser toward standing on resume. (The orchestra-
  tor's own "span starts at 0 so reset is equivalent" assumption was
  WRONG — end-window-dependent derivation.) Fix: span stamped into the
  RSI env-spec meta (derived_from_span) + signature; guards re-derive on
  mismatch with span_changed events; live stage's derived env lineage
  cleared with backups (.pre_d24_repair_backup/).
- **M1 (proven)**: _phase_segments' 3-decimal rounding emitted an end
  boundary ~0.4 ms past the true duration — legitimate reaches-the-end
  spans died as crop errors. Fix: half-frame clamp.
- **M2**: ends_settled's range-over-0.5s read flipped non-monotonically
  on mocap wiggle — the REQUIRED fast_completion gate could silently
  fail OPEN on a near-boundary span. Fix: least-squares trend over
  min(1.0 s, 25% of duration); live verdicts preserved.
- **M3**: an ATTACHED clip that failed to load silently downgraded to
  ungated no-reference acceptance. Fix: fail closed (reject with the
  load error); clipless stages unchanged.
- **L1 (accepted, documented)**: the golden regression pins the
  snapshotted ARTIFACT, not the pipeline — reframed honestly in the test
  docstring; the pipeline itself is pinned by the spans/gates/criterion
  test files.
- **Clean bills** (audit genuinely tried): criterion-grounding eval
  safety (allowlist AST gate ordering verified — materially stronger
  than the denylist history), F4 NaN/serialization, declined-marker
  consumer confusion, snap-collision handling.
- **L2 residual (open)**: positives don't bracket overshoot-then-settle
  or intermediate-speed short-hold shapes — the surface where a fifth
  class would live; no constructive exploit found.

### D26a. Stages 2-4 regenerated under the fixed pipeline; rejection paths now clear stale steering pointers
Full-mission repair status after regenerating every stage's metric:
- torso_righting: span 0-8.1 s, six gates green, spec 1.0 on the live
  rollouts (D25).
- supine_getup_and_hold: span 1.1-16.37 s (the full get-up + hold — the
  correct scope for the final chain-everything stage), six gates green,
  accepted.
- feet_under_crouch: span DECLINED twice (LLM proposes out-of-bounds
  spans, e.g. [2.408, 4.742] on a shorter clip) — fail-closed to the
  mission-level fallback. Root cause is upstream: its a10_lie_to_crouch
  binding never reaches the criterion's 0.4 m (D19 noted z 0.00->0.28)
  — the D22-deferred auto-retrieval-quality item, not a span bug.
- drive_to_stand: span 8.433-11.0 s selected consistently across two
  runs, but the AUTHORED metric failed certification both times
  (nondegeneracy 0 on its own exemplar) — fail-closed; needs an author
  retry with a larger candidate budget.
NEW GAP found while verifying fail-closed was real: the runtime resolves
`steering_metric or fitness_metric` WITHOUT re-checking acceptance, and
two live states bypassed the policy — a grandfathered pre-D24 full-clip
metric behind an unresolved decline (feet_under_crouch), and a regen
that overwrote an accepted metric.py with the REJECTED candidate while
the pointer survived (drive_to_stand). Every rejection path in
generate_stage_metrics now clears a stale steering_metric
(stage_steering_metric_cleared event; pinned by test); both live stages'
pointers cleared in mission.json. Note: the whole-mission pass's
"steering_metric already set" skip guard still grandfathers stale
metrics — only the per-stage regen path re-evaluates; deferred below.

### Deferred findings (logged, not yet fixed)
- D27 live findings (first user-driven decompose, g1-standing-up
  2026-07-12): (a) retrieval auto-attached a PRONE-start clip
  (a10_lie_to_crouch, start-window g_x +0.955) to a SUPINE stage — the
  clip-pose QC correctly refuses every span; the fix is a clip swap, not
  regeneration (same retrieval-quality family as above; consider a
  start-pose filter in retrieval itself). (b) GRAB "poses_120" clips
  carry NORMALIZED root heights (d13_crouch_to_ready: crouch start
  z=0.0, standing plateau 0.39 vs real 0.47->0.78) — span selection now
  handles it (no-headroom guard, 6a50cdb) and criterion re-grounding
  produced direction-safe numbers, but the metric author failed
  monotonicity+settled_start twice against the normalized signature;
  ingest should either denormalize (absolute-anchor like D19's offset
  math) or stamp a `heights_normalized` flag into the signature so
  authors are warned.
- D26a follow-ups: feet_under_crouch needs a better reference binding
  (its lie-to-crouch clip never reaches the stage's height band — same
  family as the auto-retrieval-quality item below); drive_to_stand needs
  a metric-author retry with n_candidates>1; the whole-mission
  "steering_metric already set" skip guard grandfathers pre-D24 metrics
  (only per-stage regen re-evaluates them) — consider a mission-wide
  re-certification sweep command.
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
