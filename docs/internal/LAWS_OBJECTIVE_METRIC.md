# Laws of a High-Quality Auto-Generated Objective Metric

**Status:** spec / implementation plan (2026-06-18). Authored by a 16-agent research workflow
(10 readers: internal forensics on g1-kick-v5 + external RL literature; high-effort synthesis;
4-lens adversarial review; code-verified reconcile). Forensic anchor: the g1-kick-v5 21-iteration
reward-hacking failure. This is the spec we implement the metric-quality bake-in + review panels from.

---

# THE LAWS OF A HIGH-QUALITY AUTO-GENERATED OBJECTIVE METRIC — FINAL SPEC

*Forensic anchor: **g1-kick-v5**. A Unitree G1 asked to kick learned to (a) balance on one leg doing a partial kick and (b) kick behind/sideways, surviving 21 reward-edit iterations because the hand-authored `spec_g1_kick` measured sagittal-leg burst + stationarity but not direction, completion, balance-during-strike, or naturalness. The metric was **partial-credit-blind**: it scored degenerate sub-motions 0.13–0.38 instead of 0.0, keeping every hack alive.*

**Ground-truth verified against the live tree (not the dossier):**
- `spec_metrics.py:471` computes `ev = kick_events_score(...)`, merges `**ev` into the dict, but `spec_metrics.py:497` builds `spec_score = intensity * ratio_gate * up * stationarity` — **`ev` is not a factor.** The central forensic claim is true in live code.
- `ratio_gate` (`spec_metrics.py:476`) is a peak/median ratio and is **load-bearing in the positive path** — yet `gen_objective_metric.md:64` forbids peak/median ratios. The live metric violates its own generator's law.
- `metric_validate.py:539-540`: `positive_keys=("active","active_kick","active_floss","active_jump")`, `negative_keys=("still","fallen","upright_flail","chaotic")` — **no `kick_behind`, `one_leg_balance`, or `partial_rep` negative exists.** `active_kick` (l.295-300) bursts one leg with the stance leg quiet → a one-leg pose is the *positive template*.
- `metric_axioms.py:27` `uprightness_monotone` is marked **"(universal)"** and l.9-12,38-41 *concede* every L0/L1 archetype is "origin, unit gravity, +x" so non-forward goals "pass vacuously."
- `realism.py:147` `audit_rollout` returns `{ok,mild,severe,unknown}` and is **advisory-only** — never wired into metric acceptance.
- **Data-plumbing (the correction that matters):** the metric's persisted `arrays` are exactly `joint_pos, joint_vel, actuator_force, projected_gravity_b, root_link_pos_w, action` (`_mjlab_runner.py:1045-1132`). The per-foot channels `left/right_foot_contact`, `..._swing_speed`, `..._height` (`mjlab.py:209-214`, computed in `_mjlab_runner.py:_foot_info`) are surfaced to the **reward `info` dict only** — they are NOT in the metric's arrays. Foot **Cartesian position** (pelvis frame) is not computed at all.

This refutes the dossier's binary "foot data absent." The truth is three tiers, and it changes the priority order decisively (see §1 priority and §3b).

---

## 1. THE LAWS (final, prioritized)

**Priority is by leverage-on-future-success (cheap × high-impact × already-plumbed), NOT by causal-blame.** The consistency review's headline defect — the draft's Part-A ranking (LAW 1 direction first) contradicting its own ROI sequence (gate first) — is resolved here: **the composed completion-gate is #1** because the unwired `kick_events_score` is sitting at `spec_metrics.py:471` and retroactively kills the partial-credit that *is* the forensic root cause; direction (a real plumbing dependency) is sequenced after the offline-free wins.

Every directional/postural law is wrapped in the **self-scoping guard of LAW 0**, which the over-restriction review demanded: a gate that lacks its task-declared frame field **abstains (observe-only), never penalizes**. This is what stops handstand/backflip/mule-kick false-rejects.

---

### LAW 0 — GOAL-FRAME DERIVATION FIRST; ABSTAIN ON UNRESOLVED. *(new — over-restriction)*
**Statement.** Before any directional, postural, or completion gate fires, the metric MUST resolve a task-declared frame: `{goal_axis: unit-vector|None, support_mode: double|single|flight|mixed|None, torso_target: upright|horizontal|any|None}`. **Any gate whose required field is `None` abstains — it contributes neither a penalty nor a pass, and the metric emits reduced steer-rights.** No fixed global negative archetype may encode a direction or posture; only task-invariant failures (frozen-no-motion, NaN/chaos, fell-relative-to-the-task's-own-support-mode) may be hard global negatives.
**Failure prevented.** False-rejection of handstand (`support=single/flight`, `torso=any`), backflip (`torso=horizontal` mid-flight), mule-kick (`goal_axis=−x`), crawl (`torso=horizontal`). Directly neutralizes the over-restriction review's defects 1-4 and the measurability review's "behind = bad is hard-coded." Without it, LAWs 1/3/5/13's asymmetry is forward-upright-biped overfit.
**Offline test.** A schema/lint check: every directional-or-postural sub-metric must read its frame field and branch to honest-abstain when `None`. Add a `frame_unresolved` archetype family (a valid backflip, a valid handstand, a valid mule-kick) and assert the metric does NOT score them below floor when their frame field is absent.
**Target.** `gen_objective_metric.md` (new "GOAL FRAME" section before HARD RULES); `metric_validate.py` (the `frame_unresolved` non-penalty assertion); task-spec schema in `metric_gen.py`.

---

### LAW 1 — COMPOSED COMPLETION GATE: `spec_score = completion_gate · min(saturated_channels)`. *(merges draft LAW 2+4+8; resolves consistency D1/D2/D5)*
**Statement.** The metric MUST be exactly one formula:
```
spec_score = completion_gate({0,1}-sharp) · min(saturated_independent_channels)
```
`completion_gate ∈ {0,1}`-like (sharp sigmoid on a *completed, returned* cycle) **owns the floor**; `min` over ≥2 **decorrelated** saturating channels **owns the aggregation of what survives the floor**. No free-floating product term, no weighted sum, no fractional partial credit, no peak/median ratio anywhere.
**Failure prevented.** The core forensic finding: `intensity·ratio_gate·up·stationarity` (`spec_metrics.py:497`) gave 0.2–0.4 for 3-of-4 weak satisfaction, so static-twitch and whip-and-fall were *lower-scoring kicks*, not *non-kicks* — 13 iterations of oscillation. **Skalse 2209.13085** (partial credit reverses true orderings); **Coste ICLR 2024** (min/worst-case aggregation eliminates overoptimization); **Abdolmaleki ICML 2020** (linear scalarization collapses onto a degenerate vertex of the concave humanoid Pareto front). The `ratio_gate` is also the **Extremal-Goodhart** term (Pan 2201.03544) that drove 18 rad/s whips.
**Resolved tensions (consistency review):**
- **D1 (LAW 2 vs 4 collision):** `min(0.4,0.9)=0.4` IS partial credit. Resolution: the `{0,1}` `completion_gate` multiplies *outside* the `min`. `min` only ranks survivors.
- **D2 (min vs Spearman ties):** a `min`-gated ladder collapses all below-floor rungs to ≈0 → tied zeros destroy rank resolution. Resolution: **LAW 7 splits the ladder** — completion-positive rungs ranked by `min`-quality (smooth, Spearman applies); completion-negative rungs asserted **below-floor as a pass/fail set**, NOT rank-correlated.
- **D5 (LAW 8 deletes the live positive term):** `ratio_gate` is the current burst signal. Resolution: replace it with a **saturated integrated-impulse over the kick window** (`1 − exp(−∫|signed swing-vel| dt / scale)`), specified in this same law — not "flag and leave a hole."
**Offline test (today, on current arrays):** (a) AST scan rejecting weighted sums, fractional products, and `p95/p99`-over-median forms; (b) the **behavioral perturbation suite** (replaces static AST as canonical, per measurability F5): inject a single-frame velocity spike → `Δscore < ε`; inject one extra valid rep → score monotone-non-decreasing; the partial-rep archetype must floor to 0.
**Concrete kick fix:** define a completed cycle = signed-forward swing crosses `+thr`, then returns below a stance band within window `W`, with double-support restored (the cycle predicate must state `W, thr, band, debounce` — measurability F2); fold the (recalibrated) `kick_events_score` in as the gate. **`spec_metrics.py:471 → :497`.**
**Target.** `gen_objective_metric.md` (HARD RULE: the single composition formula); `spec_metrics.py:471,476,497`; `metric_validate.py` (perturbation suite + disagreement test).

---

### LAW 2 — MINIMUM-AMPLITUDE / RANGE-OF-MOTION FLOOR. *(new — completeness D2)*
**Statement.** The goal-defining end-effector/joint MUST traverse ≥ a task-derived minimum arc within the action window; sub-threshold motions floor to 0. This is the missing counterpart to LAW 1's saturation *ceiling*.
**Failure prevented.** Both "partial kick" and "static twitch" are *small-amplitude* motions scoring 0.13–0.38. The draft law-set caps the top (LAW 1 saturation) but has **no floor on motion magnitude** — a correctly-signed, completed, *tiny* kick passes everything. This is the literal sub-threshold-amplitude exploit in the forensic record.
**Offline test (today):** swing-speed integral or peak joint excursion over the window must exceed a floor; the `partial_rep` and `static_twitch` archetypes assert below-floor. For G1, `*_foot_swing_speed` (once plumbed into arrays, §3b) gives this directly; the joint-angle excursion proxy (`joint_pos` range over the window) works on current arrays *today* with no plumbing.
**Target.** `gen_objective_metric.md` (amplitude-floor HARD RULE); `spec_metrics.py`; `metric_validate.py` (amplitude rungs).

---

### LAW 3 — VETO DEGENERATE SUPPORT/POSTURE — SCOPED BY `support_mode`. *(draft LAW 3 + over-restriction guard + completeness D1/D4/D7)*
**Statement.** When `support_mode=double`, the metric MUST veto (≈0) a one-leg-balance-with-twitch via an explicit **contact-schedule** check (sustained single-foot contact, sub-threshold leg motion, zero translation) AND a **support-polygon / COM-feasibility** check distinct from torso-uprightness. **When `support_mode ∈ {single,flight,mixed}` or `None`, the contact veto is disabled** (hop, flamingo, handstand, cartwheel are legitimately single-support). A stationarity term MUST NOT *certify* posture — a frozen one-leg pose is maximally stationary.
**Failure prevented.** "Balance on one leg doing a partial kick" (the #1 surviving hack). `_KICK_STATIONARY_SCALE=0.01` (`spec_metrics.py:444`) makes a non-translating one-leg balance score `stationarity≈1.0`, *amplifying* the hack; and `active_kick` (`metric_validate.py:295`) makes a one-leg pose the *positive* template. **Extremal Goodhart, model-insufficiency** (Manheim) + KG `reward_exploitation_local_optima`. Completeness D7: `uprightness=1.0, stationarity=0.86, fitness=0.0` proves torso-uprightness ≠ postural validity → the COM-over-support-polygon gate is required as a *separate* check.
**Completeness D4 resolved (veto vs legitimate share a contact schedule):** a kick *is* transiently single-support. The separating principle is **temporal**: the veto fires on *sustained* single-support with *sub-floor motion and no completed return* (LAW 1's cycle) — a real kick is single-support *briefly* then returns to double-support. The contact-schedule check is "single-support duration > X AND amplitude < floor AND no double-support return," not "any single-support instant."
**Offline test:** **Plumbing-gated for the contact half** — needs `left/right_foot_contact` in the metric arrays (one-buffer-append, §3b; mjlab already computes them). Until then, a COM-proxy from `root_link_pos_w` lateral excursion + a `one_leg_balance` archetype (sustained single-contact, sub-threshold wiggle, zero travel) asserted below floor runs *today* on current arrays. **This is the single highest-value archetype to add.**
**Target.** `metric_validate.py:295,539` (add `one_leg_balance` negative, scope `active_kick`); `gen_objective_metric.md`; `review_objective_metric.md` (add to gaming-vector list); `_mjlab_runner.py:1116-1132` (persist contact); `spec_metrics.py:444`.

---

### LAW 4 — SIGNED, FRAME-INVARIANT GOAL DIRECTION ALONG `goal_axis`. *(draft LAW 1 + LAW 0 guard; honestly re-tiered)*
**Statement.** When `goal_axis` is declared, the metric MUST read the **signed** projection of the goal-defining motion onto that axis (forward foot/end-effector velocity, hip-pitch flexion sign), never a direction-free magnitude; a motion of equal magnitude **opposite or orthogonal to `goal_axis`** MUST score lower. When `goal_axis=None`, this law **abstains** (LAW 0).
**Failure prevented.** The #1 enabler of "kick behind/sideways." `burstiness` uses `|smoothed joint_vel|` (magnitude only, `spec_metrics.py:200`), so a rear kick has an identical sagittal-leg burst signature to a forward kick. Provenance recorded the gap at iter 5 ("no directional component … sideways flailing earns kick_swing") and deferred it. **Causal Goodhart** (Manheim): rewarding a correlate the policy intervenes on via an OOD rear-kick. The generator prompt already mandates the sagittal *plane* (`gen_objective_metric.md:48-52`) — the gap is **sign, not plane**.
**HONEST TIER (measurability D1, corrected):** signed *hip-pitch velocity* is computable on current arrays *today* (cheap proxy). But **true signed foot direction requires foot Cartesian position in the pelvis frame, which is NOT computed at all** (not in arrays, not in reward-info). A hip-pitch-extension with an abducted hip can still travel the foot forward — pitch-sign alone cannot see it. So: **hip-pitch-sign = offline-today (proxy); signed foot displacement = BLOCKED on a new site-xpos persist (§3b).** The draft's "offline-cheap, complete" is false; mark LAW 4-foot **observe-only until foot xpos is plumbed.**
**Offline test:** `active_kick_behind` negative (sagittal burst, hip-pitch *extension* sign / foot posterior) asserted strictly below every forward positive, in `metric_validate.py` and regression-tested in `audit_spec_metric_monotonicity.py`. The hip-pitch-sign assertion runs today; the foot-anterior assertion activates when xpos lands.
**Target.** `gen_objective_metric.md:48-52` (strengthen plane→sign, axis from task); `metric_validate.py` (`active_kick_behind`); `audit_spec_metric_monotonicity.py`; `_mjlab_runner.py:1130` (persist foot site xpos).

---

### LAW 5 — POSITIVE STRICTLY BEATS EVERY NEGATIVE ON A SPLIT COMPETENCE LADDER (Spearman gate). *(draft LAW 7 + consistency D2 ladder split)*
**Statement.** The metric MUST rank a curated competence ladder — *including the known hack archetypes* — correctly, gated by **two separate mechanisms**: (a) completion-**positive** rungs ranked by `min`-quality, Spearman ρ ≥ 0.7 (smooth); (b) completion-**negative** rungs (one-leg-balance, kick-behind, partial-rep, static-twitch) asserted **below-floor as a pass/fail set, NOT Spearman-ranked**. Keep the existing `min`-Spearman (not mean) anti-collusion rule across K task-derived ladders.
**Failure prevented.** This is the single check that would have caught the kick hacks pre-deployment. The infrastructure is **already strong**: positive-beats-every-negative (`metric_validate.py:561-566`), Spearman≥0.7 (`metric_calibration.py:158`), tie-robust midrank (`:239`), K independent ladders (`:214-219,427`). The gap is **rung coverage** — negatives are `still/fallen/upright_flail/chaotic`; neither one-leg-balance nor kick-behind is a rung. **Consistency D2 resolved:** the ladder split is why min-aggregation (LAW 1) and Spearman (this law) stop colliding — you never rank across the floor.
**Offline test (today):** add the four hack rungs; assert (a) ρ≥0.7 on positives, (b) all four hacks < floor. The cheapest high-value change after LAW 3's archetype.
**Target.** `metric_calibration.py:154,427` (split ladder, hack rungs); `metric_validate.py:526,561`; `audit_spec_metric_monotonicity.py`.

---

### LAW 6 — DATA-SUFFICIENCY GATE: HONEST-ZERO ON ABSENT SIGNALS, NEVER A SILENT PROXY. *(new — measurability F1)*
**Statement.** A metric that references a signal absent from the metric's persisted arrays (foot xpos, foot contact, foot swing-speed) MUST emit honest-zero / observe-only for that channel, **never silently degrade to a magnitude proxy that re-introduces a blind spot.** This makes every data dependency a *falsifiable precondition*, not a hidden assumption.
**Failure prevented.** The exact iter-5 failure mode: the direction term was deferred, and the metric fell back to magnitude — re-creating the rear-kick blindness. Without this law, LAWs 3/4 *look* enforced but silently run on proxies.
**Offline test (today):** lint — for each declared channel, assert the referenced array key is in the adapter's persisted set OR the channel is marked observe-only. Maintains the existing strong honest-zero infrastructure (`generated_metric.py` runtime honest-zero, `metric_validate.py:179-203` raw-index ban).
**Target.** `gen_objective_metric.md` (data-sufficiency HARD RULE); `metric_validate.py` (referenced-key vs persisted-key lint); adapter `expected_info_keys` / arrays manifest.

---

### LAW 7 — NATURALNESS AS A SEPARATE, REFERENCE-CONDITIONAL CHANNEL. *(draft LAW 5 + consistency D3/LAW 15)*
**Statement.** Keep naturalness/realism a *separate* channel from task-correctness (never scalarized in). It is a **hard gate iff a task reference exists; else observe-only with explicitly reduced steer-rights** (not a silent pass, not a wrong-direction veto). Ground cheapest-first: (a) Tier-A kinematic-plausibility assertions, escalating to (b) a single retargeted reference clip's end-effector kernel, then (c) held-out AMP / VLM audit.
**Failure prevented.** The "unrealistic" half of both hacks; 18 rad/s hip-yaw flailing passed as "ok." Fusing correctness+naturalness into one gated scalar is exactly what got gamed. **DeepMimic** end-effector kernel `r_e=exp(−σ‖p̂−p‖²)` geometrically excludes kick-behind/one-leg. **Consistency D3 resolved (the triple-counted tension):** LAWs 5/9/11 in the draft all blocked on the *same* missing reference distribution — collapsed here into one conditional. **Completeness D3 corrected:** a *static* foot-anterior position check is satisfiable by the one-leg pose holding the foot forward — so the Tier-A assertion MUST be **dynamic** (foot velocity + swept trajectory), per LAW 4's plumbing. The draft's "catches both hacks at ~zero compute" is **false for hack (a)** until contact-schedule (LAW 3) + amplitude (LAW 2) are wired.
**Offline test:**
- **Tier-A (today, cheap):** **wire `realism.audit_rollout` (`realism.py:147`) into metric-acceptance** — it computes torque-saturation / vel-p99 / limit-violation on current arrays but is advisory-only; a `severe` verdict caps the naturalness channel. This catches the *whip* (vel-p99) today. Honest limit: a *slow, smooth, kinematically-legal* kick-behind trips none of these — that is caught by LAW 4 (direction), not naturalness.
- **Tier-B (one clip):** DeepMimic end-effector exp-kernel as a channel — **gated on Open Decision #1.**
- **Tier-C (GPU/VLM):** held-out AMP discriminator / VLM-as-judge, **audit-only, never optimized against** — cost-gated, Open Decisions #2/#3.
**Target.** `gen_objective_metric.md` (decouple style/goal); `realism.py:147` + acceptance hook in `metric_gen.py`; `metric_validate.py`; `review_objective_metric.md`.

---

### LAW 8 — BOUND AND SATURATE EVERY TERM; BEHAVIORAL (NOT AST) ANTI-EXTREMAL CHECK. *(draft LAW 8, made behavioral — measurability D9)*
**Statement.** Every term bounded in [0,1] via a saturating kernel (`1−exp(−x/scale)` or tolerance kernel); forbid unbounded magnitudes and peak/median ratios. The **canonical** check is **behavioral, not syntactic**: inject a single-frame velocity spike → assert `Δscore < ε`. (An AST scan for `p95/p99` is a *secondary* hint; an author can evade it via `max()−median()`.)
**Failure prevented.** Extremal Goodhart: `ratio_gate` (`spec_metrics.py:476`) + unbounded burst drove 18 rad/s whips. The generator already states the saturation rule (`gen_objective_metric.md:78`) and forbids ratios (`:64`) — the live metric *violates its own generator*. M2 chaos axiom (`metric_axioms.py:258`) already names the extremal failure — extend to **reject**, not just diagnose.
**Offline test (today):** single-frame spike perturbation (pure array op, evasion-proof); `mean−λ·std` as the softer `min` variant for noisy channels.
**Target.** `gen_objective_metric.md:64,78`; `metric_validate.py` (spike perturbation); `metric_axioms.py:258`.

---

### LAW 9 — ADVERSARIAL RED-TEAM PANEL: ≥3 GAMING POLICIES PER SCORED CHANNEL, CROSS-FAMILY, VETO. *(draft LAW 6 + completeness D5 coverage obligation)*
**Statement.** Before acceptance, a metric-blind cross-family red-team panel MUST propose ≥1 concrete gaming policy **per scored sub-metric channel** (not a fixed list keyed to known hacks); if any plausibly scores ≥ floor, the metric is GAMEABLE and rejected. An unrealizability claim from the panel requires a **constructive exploit rollout**, not an aesthetic "looks unnatural" (over-restriction guard).
**Failure prevented.** `spec_g1_kick` was hand-authored, never adversarially tested; the L3 `adversarial_archetype_gate` (`metric_calibration.py:314`) that synthesizes task-specific hacks is **flag-OFF (`RS_ADVERSARIAL_ARCHETYPES`) AND generated-metric-only** — it never ran on the hand-authored metric that scored v5. **Skalse Cor 1-2:** unhackability cannot be achieved by careful authoring alone. **Completeness D5:** a fixed three-archetype list is curve-fitting to the anchor; the *per-channel coverage obligation* forces archetypes for the *unenumerated* third hack.
**Offline test:** none — **this tier is VLM-API, cost-gated, not offline** (measurability D8 correction: the draft wrongly listed "flip L3 ON" in the all-offline ROI sequence). Belongs strictly in §4-A.
**Target.** `metric_calibration.py:314` (flip default, extend to `spec_*` hand-authored, per-channel obligation); `gen_gaming_archetypes.md`; `review_objective_metric.md`.

---

### LAW 10 — SHAPING (DENSE, TRADEABLE) vs METRIC (SPARSE, NON-TRADEABLE GATES). *(draft LAW 11)*
**Statement.** Style/smoothness regularizers (action-rate, jerk, energy, foot-clearance) belong in the *reward* and may be soft and tradeable; *directionality + posture-feasibility + transient-success + amplitude* belong in the *metric* as hard pass/fail gates. This is what lets LAWs 1/2/8's conjunctive-saturating metric *not* starve the reward loop: the reward stays dense.
**Failure prevented.** `spec_g1_kick` collapsed correctness + naturalness + stationarity into one soft gated scalar, letting the policy *trade* posture for burst credit — the soft-penalty regime CMDP literature (T-RO 2024, constraints-not-penalties) says gets gamed. `edit_rewriter.md` already keeps regularizers soft at 0.3× — the metric side needs the hard gates.
**Offline test:** lint — the metric must not contain action-rate/jerk/energy summands (those are reward-only); assert the metric is a gate, not a regularizer sum.
**Target.** `gen_objective_metric.md` (partition framing); `edit_rewriter.md`.

---

### LAW 11 — HELD-OUT RE-EVAL + GOODHART-ONSET EARLY-STOP. *(draft LAW 10, honestly tiered)*
**Statement.** The selected iteration's metric MUST be re-evaluated on fresh held-out seeds before it is believed (report a shrunken/held-out score, multi-seed + CIs); and a sustained metric rise with a *falling* naturalness/secondary channel MUST trigger early-stop, not "success."
**Failure prevented.** **Optimizer's curse** (Smith & Winkler 2006): selecting the best-metric edit upward-biases that edit. **Overoptimization hump** (Gao 2210.10760): the loop edited past iter-8's genuine 0.4806 success and walked out of the basin — a divergence-aware stop would have locked v8. Single-seed metric is noise (Henderson 1709.06560).
**HONEST TIER (measurability D7):** the **Goodhart-onset divergence check** (primary rises while secondary falls) is **offline-today** on logged `metric_history`. The **fresh-seed re-eval is a GPU rollout** for G1, not an array op — mark GPU-gated. The draft's "offline" label is wrong for the seed half.
**Target.** loop policy; `edit_rewriter.md` / `diagnose_grounded.md` (Goodhart-onset, offline); fresh-seed re-eval (GPU).

---

### LAW 12 — JOINT/ROLE SELECTION BY DIRECTION-AWARE NAME; HONEST-ZERO ON UNRESOLVED. *(draft LAW 12 — KEEP-DOING, already enforced)*
**Statement.** Select joints by functional direction-aware role (side+segment+axis), resolve each to exactly one joint or emit observable `spec_score=0.0`; never a hard-coded integer index.
**Status: already ENFORCED and strong** — `_raw_joint_index_violations` (`metric_validate.py:179-203`), permutation-robustness (`:494-524`), `leg_sagittal_indices` excludes hip roll/yaw (`joint_resolver.py:329`), runtime honest-zero (`generated_metric.py:174-191`), generator role-naming (`gen_objective_metric.md:34-52`). Listed to **lock won ground** so a refactor doesn't regress it.
**Offline test:** maintain existing index/permutation gates.
**Target.** keep `metric_validate.py:179,494`; `joint_resolver.py:329`; `gen_objective_metric.md:34`.

---

### LAW 13 — FRAME/UNIT/HEADING INVARIANCE; MONOTONE-IN-UPRIGHTNESS **CONDITIONED ON `torso_target`**; NO REWARD FOR CHAOS. *(draft LAW 13 + over-restriction fix to M1)*
**Statement.** The metric MUST be invariant to world-translation, gravity-magnitude, and heading; MUST NOT increase under whole-body velocity chaos; and MUST be monotone-non-increasing as the torso tilts toward horizontal **only when `torso_target=upright`**. For `torso_target ∈ {horizontal, any}` (backflip, dive, roll, crawl), **M1 is inverted or disabled.**
**Status: mostly ENFORCED** as the L1 axiom battery — I1 translation (`metric_axioms.py:225`), I2 gravity-scale (`:227`), I3 yaw (`:229`), M2 no-reward-for-chaos (`:258`), M3 stationary-no-travel (`:270`). **Over-restriction defect (verified):** `uprightness_monotone` is marked **"(universal)"** at `metric_axioms.py:27` — so **a backflip's correct horizontal-torso execution is scored DOWN.** This is the most dangerous overfit because it's buried in a "keep-doing" law. **Fix: condition M1 on `torso_target`.** The module already concedes (l.38-41) that non-+x metrics "pass vacuously" — which is acceptable for invariance axioms (they don't *false-reject*, they just don't *constrain*), but M1-monotone is an *active* penalty and must be gated.
**Offline test:** maintain I1/I2/I3/M2/M3; gate M1 on `torso_target`; add the `frame_unresolved` non-penalty assertion (LAW 0).
**Target.** `metric_axioms.py:27,242` (condition M1); keep the rest.

---

## 2. THE KICK-FAILURE TRACE (sufficiency proof)

| g1-kick-v5 hack | Why the live metric scored it >0 | Final law that catches it | Mechanism | Tier |
|---|---|---|---|---|
| **(a) One-leg balance + partial kick** | `stationarity≈1.0` at `_KICK_STATIONARY_SCALE=0.01` (no travel) × `active_kick` IS one-leg positive template; partial swing crosses burst threshold | **LAW 3** (one-leg veto, `support=double`) + **LAW 2** (amplitude floor) + **LAW 1** (completion gate: no double-support return → gate=0) | `one_leg_balance` negative below floor (today, COM-proxy); contact-schedule veto (after foot-contact plumb); amplitude floor on swing excursion | offline-today (proxy) → cheap-plumb |
| **(b) Kick behind / sideways** | `burstiness` = `|smoothed joint_vel|`, magnitude-only → rear kick == forward kick signature | **LAW 4** (signed `goal_axis` projection) + **LAW 5** (`active_kick_behind` rung below forward) | hip-pitch-sign assertion (today); signed foot-displacement (after xpos plumb); rung in `audit_spec_metric_monotonicity.py` | hip-sign offline-today; foot-dir blocked-on-plumb |
| **(b′) 18 rad/s whip** | unbounded burst + `ratio_gate` peak/median | **LAW 8** (single-frame spike → Δ<ε; saturate) + **LAW 1** (no ratio terms) | behavioral spike perturbation; replace ratio_gate with saturated integrated-impulse | offline-today |
| **(general) 21-iter oscillation** | fractional product gave 0.2–0.4 partial credit → no categorical floor | **LAW 1** (composed `completion_gate · min`) + **LAW 11** (Goodhart-onset stop locks iter-8) | fold `kick_events_score` into `spec_score`; lock prior best when secondary channel falls | offline-today (gate) / offline (stop) |
| **(unnatural, both)** | zero naturalness grounding | **LAW 7** (separate channel) + **LAW 9** (red-team) | `realism.audit_rollout` wired in (whip, today); ref-clip/AMP/VLM (cost-gated) | Tier-A today; rest cost-gated |

**Sufficiency claim (honest):** the **offline-today** subset (LAWs 1-gate, 2-amplitude-proxy, 3-COM-proxy-archetype, 5-rungs, 8-spike, 11-stop, plus wiring `realism.audit_rollout`) **denies every partial-credit score that kept the 21-iteration oscillation alive** — hack (a) via amplitude+completion, the whip via spike, the oscillation via the composed gate. **The residual gap is offline-incomplete-but-cheap-plumb:** the *clean* directional discrimination of hack (b) (signed foot displacement) and the contact-schedule precision of hack (a) need the one-buffer-append of foot site xpos + contact (§3b). Hip-pitch-sign (LAW 4 proxy) catches the *gross* rear-kick today; the abducted-hip edge case waits on xpos. No law requires GPU/VLM to kill the two documented hacks.

---

## 3. BAKE-IN PLAN

### 3a. `gen_objective_metric.md` — instructions to add
Insert a **GOAL FRAME** section before HARD RULES, and amend/add rules:
1. **GOAL FRAME (new, LAW 0):** "First resolve `{goal_axis, support_mode, torso_target}` from the task spec. Any directional/postural/completion check whose field is unresolved MUST abstain (contribute neither penalty nor pass) — never assume forward/upright/double-support."
2. **HARD RULE — single composition (LAW 1):** "`spec_score = completion_gate · min(saturated_channels)`. `completion_gate ∈ {0,1}` (sharp sigmoid on a completed, *returned* cycle) and multiplies OUTSIDE the min. No weighted sums, no fractional products, no peak/median ratios. Fold any 'events' counter INTO the score — never report it diagnostic-only." *(directly fixes `spec_metrics.py:471→497`)*
3. **HARD RULE — amplitude floor (LAW 2):** "The goal-defining motion must traverse ≥ a task-minimum arc within the window; sub-threshold motions floor to 0."
4. **Strengthen DIRECTION (LAW 4):** change l.48-52 from sagittal-*plane* to **signed projection on `goal_axis`**: "a motion of equal magnitude opposite/orthogonal to `goal_axis` MUST score lower. Use a SIGNED quantity, not `|·|`."
5. **HARD RULE — data sufficiency (LAW 6):** "If a needed signal (foot xpos, contact) is absent from the arrays, the channel is observe-only — never substitute a magnitude proxy."
6. **HARD RULE — shaping partition (LAW 10):** "The metric is a pass/fail competence gate, NOT a sum of style regularizers. Smoothness/energy/jerk live in the reward, not here."
7. **Condition M1 framing (LAW 13):** "Monotone-in-uprightness applies only when `torso_target=upright`; flip/dive/roll invert it."

### 3b. `metric_validate.py` — new/strengthened gates
- **Add three negative archetypes** (the highest-ROI change). At `_archetypes()` (l.233) add `one_leg_balance` (sustained single-foot contact, sub-threshold leg wiggle, zero travel — COM-proxy from `root_link_pos_w` today), `active_kick_behind` (sagittal burst, hip-pitch *extension* sign), `partial_rep`/`static_twitch` (one half-swing, no return; tiny amplitude). Add all three to `negative_keys` at **l.540**.
- **Scope `active_kick` (l.295)** so the one-leg positive only anchors `support=single` tasks (over-restriction).
- **Behavioral perturbation suite (LAWs 1/8, canonical):** single-frame spike → Δ<ε; one-extra-rep → monotone-up; mirror/flip → invariant. Pure array ops; replaces static AST as the primary gate.
- **Null-control test (measurability F7):** frozen-pose, pure-noise, time-reversed-real-kick → all below floor. Cheapest universal falsifier; catches one-leg-freeze and rear-kick (time-reversed ≈ rearward) in one shot.
- **Data-sufficiency lint (LAW 6):** referenced array key ∈ persisted set, else channel observe-only.
- **Independence gate (measurability F4):** pairwise |Spearman| between sub-metrics over the ladder < 0.8, else collapse — credits `min` only for decorrelated channels.
- **Persist foot channels (the plumbing unlock):** in `_mjlab_runner.py:1116-1132` append `left/right_foot_contact` and `left/right_foot` **site xpos** to the rollout-arrays buffers (mjlab already computes contact in `_foot_info`; site xpos is the same `_robot.data` access pattern as `root_link_pos_w` at l.1130). This converts LAW 3-contact and LAW 4-foot from observe-only to enforced.

### 3c. `metric_calibration.py` — calibration / adversarial tests
- **Split the ladder (consistency D2):** in `_ladder` (l.68) / `calibrate_task_derived` (l.427), rank completion-positive rungs by `min`-quality with Spearman≥0.7; assert completion-negative rungs (one-leg, kick-behind, partial-rep) below-floor as a **set**, NOT Spearman-ranked.
- **Add hack rungs** to the kick ladder.
- **Flip `RS_ADVERSARIAL_ARCHETYPES` default ON** for high-stakes metrics; **extend `adversarial_archetype_gate` (l.314) to score hand-authored `spec_*`**, not only generated; add the **per-channel coverage obligation** (≥1 archetype per scored channel).
- Keep `min`-Spearman anti-collusion (`spearman_midrank`, l.239).

### 3d. `review_objective_metric.md` — real rubric (currently 29 lines, walker-only)
Expand the closed gaming-vector list (l.16-21) and add law-derived criteria. New rubric items — REJECT if any holds:
1. **Direction:** scores a rear/lateral motion ≥ a goal-axis-aligned one (LAW 4).
2. **One-leg balance:** a sustained single-support sub-threshold pose scores ≥ floor for a `support=double` task (LAW 3).
3. **Partial/amplitude:** a sub-threshold or non-returned half-cycle scores ≥ floor (LAWs 1/2).
4. **Composition:** any weighted sum, fractional partial-credit product, or peak/median ratio (LAW 1/8).
5. **Naturalness fused:** correctness and naturalness scalarized into one gate (LAW 7).
6. **Frame overfit:** a directional/postural gate fires without a declared frame field (LAW 0) — would false-reject handstand/backflip/mule-kick.
Reviewer must **emit a concrete gaming policy per scored channel**, and an unrealizability claim needs a constructive exploit rollout, not an aesthetic judgment.

---

## 4. REVIEW-PANEL ARCHITECTURE (two high-stakes decisions)

**Shared-family-bias mitigation (both panels):** **Panickssery NeurIPS 2024** (self-preference is mechanistic) → the author's model family is **excluded** from its own review. **Verga PoLL 2404.18796** (disjoint juries beat single GPT-4, 7× cheaper, decorrelated error) → ≥3 distinct families. Provenance/author identity **blinded**. **Veto is asymmetric** (any reviewer can reject; approval requires quorum) — a false approval silently corrupts the whole optimization (the review prompt already states this at l.27-29).

### Panel A — Objective-Metric Generation (`metric_gen.py`)
- **Reviewers:** 3–5, **≥3 distinct model families**, author family excluded, provenance blinded.
- **Diversified lenses (distinct system prompts, one per reviewer):** (1) **completeness** — "name a gaming policy per scored channel that the metric rates ≥ floor"; (2) **measurability** — "is each channel computable on the declared arrays, or a silent proxy? (LAW 6)"; (3) **over-restriction** — "does any gate fire without its frame field, false-rejecting handstand/backflip/mule-kick? (LAW 0)"; (4) **consistency** — "is the composition exactly `completion_gate · min`, no free product/ratio? (LAW 1/8)"; (5) **naturalness** — "is naturalness a separate channel, reference-conditional? (LAW 7)".
- **Rubric:** §3d, one criterion-cluster per lens.
- **Aggregation/veto:** **gameability is a pass/fail VETO**, not an averaged score. Any lens raising a *constructive* exploit (rollout that scores ≥ floor) → REJECT → regenerate. Reserve **debate** (Du 2305.14325) for *disputed* exploits (one reviewer claims gameable, another disputes realizability).
- **Plug-in point:** `metric_gen.py` after `validate_generated_metric` passes and before steer-rights are granted — the panel is the gate between "passes offline axioms" and "is trusted to steer." Wire alongside `calibrate_task_derived` / `adversarial_archetype_gate` in `metric_calibration.py:314`.
- **Cost:** VLM-API, **cost-gated** (Open Decision #2). Offline gates (§3b/3c) run *first and free*; the panel runs only on metrics that clear them.

### Panel B — Reward-Function Edit Between Rounds (`edit.py` / `sculpt.py`)
- **Reviewers:** 2–3 (edits are higher-frequency than metric-gen → lighter panel), ≥2 families, the editing model's family excluded from approving its own edit.
- **Diversified lenses:** (1) **Goodhart-onset** — "did the objective metric rise while a naturalness/secondary channel fell? If so, prefer LOCKING the prior best over this edit (LAW 11)"; (2) **shaping-partition** — "did this edit smuggle a hard task-gate into the reward, or a style regularizer into the metric? (LAW 10)"; (3) **realism mandate** — "are naturalness regularizers kept soft (≤0.3×), per `edit_rewriter.md`?".
- **Rubric:** divergence (primary↑ + secondary↓ = flag), partition integrity, realism-soft.
- **Aggregation/veto:** divergence flag from any reviewer → the loop **locks the prior best** rather than applying the edit (this is the offline-today early-stop that would have saved iter-8). Non-divergent edits proceed.
- **Plug-in point:** `edit.py` (post-edit, pre-apply) reading logged `metric_history` for the Goodhart-onset check (**offline, no GPU**); `sculpt.py` orchestrates the lock-vs-edit decision.
- **Cost:** the Goodhart-onset check is **offline-today**; the LLM-panel review of the edit text is VLM-API (cost-gated, can run as a single-reviewer cheap pass and escalate to 3 only on a divergence flag).

---

## 5. OPEN DECISIONS FOR SAM (need your call before implementation)

1. **Reference kick clip — yes/no?** LAW 7-Tier-B (DeepMimic end-effector kernel) is the strongest *complete* naturalness catch and the 2026 soccer result (91.3% tracking vs 46.8% AMP) argues one retargeted clip beats AMP for a *specific* directional kick. **But** a clip exists for "kick," not for arbitrary novel tasks, so it makes the kick metric strong while leaving novel-task naturalness on Tier-A only. **Decision:** do we retarget one forward-kick clip for G1 (unblocks Tier-B for the anchor), or stay Tier-A-only (offline-free, incomplete on smooth-but-degenerate naturalness)? *My recommendation: defer — the offline-today subset kills both documented hacks without it; revisit only if a smooth-degenerate hack appears.*

2. **VLM grounding (Ship-54, L4) — enable for high-stakes metric-gen?** Panel A's adversarial veto and LAW 7-Tier-C are VLM-API. Per the campaign state this is **cost-gated, STOP-confirm-spend**. **Decision:** enable VLM Panel A only for *promoting a metric to steer-rights* on a flagged-high-stakes task (bounded call count), or keep fully offline for now? *Recommendation: enable bounded VLM Panel A for steer-rights promotion only; keep the edit-panel (B) offline (Goodhart-onset) until proven necessary.*

3. **Naturalness veto strictness.** When naturalness is observe-only (no reference) — does a `severe` `realism.audit_rollout` verdict **hard-reject** the iteration, or **down-weight + flag** (reduced steer-rights, LAW 7)? Hard-reject risks false-rejecting a genuinely-aggressive-but-valid motion (a hard kick legitimately has high vel-p99). **Decision needed.** *Recommendation: down-weight + flag (reduced steer-rights), hard-reject only on limit-violation (joints past their stops = exploit), not on vel-p99 alone.*

4. **Foot-data plumbing — approve the one-buffer-append?** Persisting `left/right_foot_contact` + `left/right_foot` **site xpos** into the metric arrays (`_mjlab_runner.py:1116-1132`) is the single change that converts LAW 3-contact and LAW 4-foot-direction from observe-only proxies to *enforced* — and it's cheap (mjlab already computes the data). **Decision:** approve the schema addition now (unblocks the *clean* directional + contact-schedule discrimination), or ship the offline-today proxy subset first and add xpos later? *Recommendation: approve now — it is the highest-leverage cheap change and the only thing standing between the proxy and the complete kill of hack (b).*

5. **`RS_ADVERSARIAL_ARCHETYPES` default + scope.** Flip default ON for high-stakes metrics AND extend the gate to score **hand-authored `spec_*`** metrics (today it's generated-only, which is why it never ran on the metric that scored v5)? This adds latency/API cost to every high-stakes acceptance. **Decision needed.** *Recommendation: ON for high-stakes + extend to `spec_*`; keep OFF for routine generated-metric iteration to bound cost.*

---

**Files referenced (all absolute):**
- `/home/samjd/projects/RewardSculptor/sculptor/eval/spec_metrics.py` (lines 200, 444, 471, 476, 497 — unwired `kick_events_score`, ratio_gate, `_KICK_STATIONARY_SCALE`)
- `/home/samjd/projects/RewardSculptor/sculptor/eval/metric_validate.py` (179-203, 295-300, 526-583, 539-540 — archetypes + gates)
- `/home/samjd/projects/RewardSculptor/sculptor/eval/metric_calibration.py` (42, 158, 214-219, 239, 314, 427 — Spearman, K-ladder, adversarial gate)
- `/home/samjd/projects/RewardSculptor/sculptor/eval/metric_axioms.py` (27, 225-270 — M1 "universal" bug, vacuous-pass concession)
- `/home/samjd/projects/RewardSculptor/sculptor/eval/joint_resolver.py` (329 — `leg_sagittal_indices`)
- `/home/samjd/projects/RewardSculptor/sculptor/eval/generated_metric.py` (174-191 — runtime honest-zero)
- `/home/samjd/projects/RewardSculptor/sculptor/adapters/realism.py` (147 — advisory-only `audit_rollout`)
- `/home/samjd/projects/RewardSculptor/sculptor/adapters/mjlab.py` (209-223 — `_G1_INFO_EXTRA` reward-only foot keys)
- `/home/samjd/projects/RewardSculptor/sculptor/adapters/_mjlab_runner.py` (333-433 `_foot_info`; 1045-1132 persisted metric arrays — the plumbing seam)
- `/home/samjd/projects/RewardSculptor/sculptor/prompts/gen_objective_metric.md` (34-52 direction-plane, 64 no-ratio, 78 saturate)
- `/home/samjd/projects/RewardSculptor/sculptor/prompts/review_objective_metric.md` (29 lines — walker-only gaming list)
- `/home/samjd/projects/RewardSculptor/sculptor/prompts/gen_gaming_archetypes.md`; `/home/samjd/projects/RewardSculptor/sculptor/prompts/edit_rewriter.md`; `/home/samjd/projects/RewardSculptor/sculptor/prompts/diagnose_grounded.md`
- `/home/samjd/projects/RewardSculptor/sculptor/eval/metric_gen.py`; `/home/samjd/projects/RewardSculptor/sculptor/sculpt.py`; `/home/samjd/projects/RewardSculptor/sculptor/edit.py` (panel plug-in points)
- `/home/samjd/projects/RewardSculptor/scripts/audit_spec_metric_monotonicity.py` (regression rungs)