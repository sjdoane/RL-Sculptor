# G1 reference-hop evolution showcase

## Research claim

Demonstrate that RewardSculptor can consume one exact, physics-certified G1
motion reference and use it as a bounded motion prior while optimizing an
independent task residual. The evolved task is a sequence of forward/lateral
one-leg hops across low alternating obstacles into a finish pad, followed by a
stable two-second hold.

This is not a claim of full OGMP. The implemented path is an immutable linear
reference clock with a phase-indexed tracking reward and a bounded task
residual. A run may be called reference-conditioned only after the same pinned
clock is present in actor/critic observations and evaluation. It may not be
called an online oracle, latent-mode policy, predicate automaton, controller
warm start, or closed-loop OGMP planner.

## Exact starting motion

- Robot/artifact: `g1/50009_one_leg_jump_poses_60_jpos`
- Content SHA-256:
  `92fc2431f99969e0835a91657a040c55a0cfe5db9db536d2daed533a7b9deca0`
- Source: Retargeted AMASS, CC-BY-4.0
- Cadence: 229 frames at 60 Hz (3.8167 s)
- Kinematic structure: five vertical peaks at approximately frames 55, 87,
  120, 152, and 184; peak root-height excursions are 0.142-0.158 m.
- Current status: Tier K only. Live training is blocked until Tier-D tracking
  re-verifies exact clip bytes, embodiment, simulator boundary, cadence, and
  rollout evidence.

## Experimental sequence

1. **Certificate:** train the dedicated tracker against the exact clip and
   require every Tier-D gate. Preserve failure as a diagnostic; never promote
   by hand.
2. **Reference control:** in a fresh UI project, attach the certified clip and
   train on a flat, obstacle-free world. Verify that runtime receipts pin the
   clip/certificate/rollout/execution-boundary hashes and that the policy
   reproduces multiple airborne peaks.
3. **Task evolution:** retain the same reference prior and add alternating low
   obstacles plus an offset finish. Only the capped residual and authored
   command/safety supervision may optimize task progress.
4. **A/B control:** run a scratch policy under the exact same world, seed,
   horizon, PPO budget, and objective metric. Report both distributions; do
   not compare a cherry-picked video against a population metric.

## Objective evidence

Acceptance is conjunctive over an immutable all-lane trajectory:

- scene geometry and evaluation world match the admitted world tuple;
- ordered intermediate regions and finish are entered;
- at least four distinct bilateral-air windows occur after the start, with no
  double-counting of one long flight;
- the robot clears every forbidden obstacle contact and has no sustained fall;
- the first bilateral landing after the final flight places both feet inside
  the finish region;
- neither foot leaves the finish on any later valid frame;
- terminal horizontal speed is below 0.12 m/s; and
- at least 100 uninterrupted post-landing frames satisfy horizontal, angular,
  joint, uprightness, and default-pose quiet.

Use direct `left_foot_pos_w` and `right_foot_pos_w` site channels for the
landing/retention test. Root-in-region is not a substitute. Objective metrics
remain independent from the generated reward and may not be weakened after a
failure.

## UI proof

The visible launch receipt must independently show:

- starting policy choice (including `scratch` when no policy is loaded);
- selected reference identity and Tier-D status;
- selected world tuple;
- exact tracking-base target hash and residual cap;
- requested versus resolved evaluation lane; and
- runtime events proving reference admission, reference-guided reward
  preparation, policy observation contract, and checkpoint load facts.

The final review includes the full selected-lane video, keyframes at each hop
and landing, all-lane objective artifacts, behavior resolution, and the Results
physical-scene audit.
