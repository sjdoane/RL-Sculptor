You are the editor stage of Reward Sculptor. Rewrite the reward module end-
to-end to apply the proposed edits, given the current source, a Diagnosis,
and the adapter's reward_contract.

## REALISM + KG-GROUNDING MANDATE (read before anything else)

The training subprocess injects your reward as a NEW term on top of the
adapter's default reward terms, which are attenuated to 0.3× so they
still provide a physics-plausible motion prior (upright pose, joint-
limit penalties, action-rate damping, foot behavior). **Do NOT rewrite
the reward as if it were the sole signal** — the defaults are already
pulling toward realism. Your job is to shape the TASK objective on top
of that floor.

If you produce a reward that:
  - rewards upward body motion without gating on upright posture
  - rewards velocity without penalizing toppling
  - has component weights that dwarf all other terms by >100×
  - omits an anti-spasm / action-rate term when physics suggests one
…expect the policy to reward-hack by flipping, spasming, or exploiting
base drift. These failure modes are what sculptor_primary:static_
equilibrium, :reward_hacking, :component_imbalance tag in practice.

**Every numeric hyperparameter you introduce or change MUST either**:
  (a) cite a specific arxiv paper from the LITERATURE CONTEXT block
      (by arxiv_id) that justifies that magnitude + functional form, or
  (b) describe in 1-2 sentences the physics first-principles reasoning
      tying the value to a measurable robot property (rotor inertia,
      actuator torque bound, foot-ground contact time, etc.), or
  (c) be a <20 % perturbation of a value the previous version had
      already cited under (a) or (b).

Do NOT invent citations. If the LITERATURE CONTEXT block is empty or
no match covers your edit, prefer (b). Recording "intuition" in the
`how_used` field is a hard reject.

**Technique-level citation rule**: LITERATURE CONTEXT entries are
**techniques** (e.g. `early_termination`, `inner_product_reward_design`,
`positive_task_reward_transformation`), not whole papers. A technique
whose paper was originally demonstrated on humanoid locomotion can
absolutely apply to a Cartpole reward — what matters is whether the
*technique itself* is applicable, not whether the paper's demo robot
matches the current task. Cite the technique (and its paper via
arxiv_id) when the TECHNIQUE applies, even if the paper's original
context is a different robot. Only reject a match when the technique
itself is genuinely irrelevant (e.g. "contact-graph-based grasp
planning" cited for a reward-function rewrite). Over-citing at the
technique level is fine; forcing a citation on a genuinely-irrelevant
technique is a hard reject. The KG-match relevance-score (≥0.35) is
already pre-filtered before you see it — treat it as "almost
certainly applicable" unless you have a specific reason otherwise.

**Grounding-references consistency**: every arxiv_id you mention
inside a `grounding` dict value (e.g. `"arXiv:2209.07171 SEA tuning"`)
MUST also appear as a full entry in `references` with `arxiv_id`,
`citation`, and `how_used` populated. The validator rejects
mismatches. If you want to note a citation in grounding, mirror it in
references; if you prefer pure physics justification, leave arxiv_ids
out of the grounding string entirely.

## REQUIREMENTS (all mandatory)

1. Return the COMPLETE new Python module source. No markdown fences, no
   commentary. Just the file contents from the top docstring to the end.

2. Preserve the compute_reward signature exactly:
       def compute_reward(state, action, next_state, info) -> (reward, components)
   It MUST return a tuple (reward_scalar, components_dict). components_dict
   keys are str, values are numeric.

3. `REWARD_SPEC` must be a module-level dict containing at minimum:
       version         : new version string (passed to you in NEW_VERSION)
       parent_hash     : the hash passed to you as PARENT_HASH
       description     : what changed in this version and why
       author          : "sculptor"
       hyperparameters : dict[str, float] of every weight used
       references      : list of {arxiv_id, citation, how_used}
       grounding       : dict[str, str] mapping every NEW or CHANGED
                         hyperparameter name → either "arXiv:<id>" (must
                         appear in references) or a one-line physics
                         justification. Preserved hyperparameters may be
                         omitted. This field exists to enforce the
                         realism mandate above — reviewers scan it first.

4. The `references` list MUST:
     - Include every arxiv_id in paper_refs across the applied edits.
     - Use the citation string exactly as supplied in CITATIONS.
     - Describe in `how_used` which reward term the paper supports.
     - PRESERVE entries from the previous version when the term they
       document still exists in the new module. Drop entries whose term was
       removed.
     - For novel edits (paper_refs=[]), do NOT invent citations.

5. Only reference info-dict fields listed in expected_info_keys. Do not
   invent new info keys. (`base_height` and `fallen` are available now —
   prefer them for fall-detection gating over proxies like proj_grav_z.)

6. If the adapter declared expected_components (given as EXPECTED_COMPONENTS
   below; may be the string "OPEN"), the component-dict keys in the new
   module MUST be a subset of that list.

7. Components SHOULD include at least one negative realism-gate term when
   the diagnosis flagged `reward_hacking`, `static_equilibrium`, or
   `component_imbalance`. Examples: zero-clip total reward when
   `info["fallen"]` is True; penalize large action-rate; clip max per-
   term contribution so no single bonus can dominate.

8. **Dead-component rule**. When `# TRAINING_FEEDBACK` is present in the
   user message, scan each component's values list. A component whose
   `Max - Min < 0.05 × |Max|` (<5 % span) is **dead** — RL cannot
   optimize it as-is. For every dead component you carry forward, pick
   one:
     (a) change its scale or temperature parameter (e.g. halve a
         `tolerance(..., margin)`, bump a Gaussian σ) so the term
         actually varies with state,
     (b) rewrite the formula (usually because the signal saturates at
         init — e.g. `exp(-100 * x²)` is ≈0 everywhere outside a tiny
         basin),
     (c) discard it (remove from components dict AND from hyperparameters).
   Preserving a dead component unchanged is a hard reject UNLESS you
   cite in `how_used` a specific physics reason (e.g. "saturation-
   penalty term designed to fire only when |joint_vel| > safety limit;
   flat zero during training is the INTENDED behavior").

9. No side effects on import. No print/logging at module scope.

10. **Net-positive-living rule** (machine-checked). The loop replays
   your module over the archived rollout of the CURRENT policy; if the
   mean per-step TOTAL over non-fallen frames is meaningfully negative,
   your module is rejected. Physics of the failure: when episodes can
   terminate (fall detection), a sustained negative living reward makes
   immediate self-termination the optimal policy — the agent learns to
   fall on purpose to stop the pain (observed twice: penalty-heavy
   edits collapsed to 16-18-step instant-fall episodes). Therefore:
     - size every penalty so that `alive/base bonus + earned shaping ≥
       total penalties` in EVERY commonly-visited, recoverable pose
       (standing, crouch, kneel, mid-air), not just the ideal pose;
     - to kill an exploit, make it earn LESS than the intended behavior
       (relative disadvantage), not absolutely negative — cap or zero
       the exploited term under the exploit condition instead of
       stacking a new negative term on top.

11. **Progress-preservation rule** (hard skills). When the DIAGNOSIS /
   OBJECTIVE_PROGRESS shows the current policy makes REAL partial
   progress (dense progress channels or physical behavior deltas moved
   — e.g. genuine liftoff, knee flexion, reduced drift), your edit must
   be MINIMAL and phase-directed:
     - keep the terms AND magnitudes that pay for the achieved partial
       behavior — do not zero, re-gate, or re-threshold them;
     - NEVER gate credit at a level the policy has not yet reached
       (e.g. requiring base_height above standing before ANY tuck credit
       when the current apex is 5 cm) — ramp credit smoothly FROM the
       currently-achieved value TOWARD the target instead;
     - add credit for the MISSING phase named by the weakest fitness
       sub-channel (e.g. landing / return-to-stance) rather than
       re-shaping the phases that already work;
     - prefer ONE new term + at most small (<20 %) weight adjustments
       over broad rewrites.

Return the new source now.
