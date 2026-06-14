You are an expert RL evaluation engineer. Write an OBJECTIVE TASK-SUCCESS
METRIC for a robot behavior goal — the ground-truth fitness used to judge
whether a learned policy achieves the goal. This is NOT a reward function:
it must be an HONEST, hard-to-game measurement of task competence.

Output a single self-contained Python module (and nothing else outside the
code fence) defining exactly:

    import numpy as np

    def compute_spec(arrays, behavior, meta):
        # ... compute from physical rollout quantities ...
        return {"spec_score": <float in [0,1]>, "<subcomponent>": <float>, ...}

INPUTS you may use (ALL OTHER inputs are forbidden):
  arrays["joint_pos"]            (T, E, J)  joint angles over time/envs
  arrays["joint_vel"]            (T, E, J)  joint velocities
  arrays["projected_gravity_b"]  (T, E, 3)  gravity in body frame; z < -0.85
                                            ≈ upright (yaw is NOT observable)
  arrays["root_link_pos_w"]      (T, E, 3)  base world position (x,y forward/
                                            lateral, z height)
  behavior                       dict: max_episode_steps, rollout_num_envs,
                                 step_dt, mean_episode_length
  meta["joint_names"]            list[str] aligned with axis J (may be absent;
                                 use arrays.get / len-check before relying on it)
Any array may be ABSENT — always `arrays.get(k)` + guard for None.

HARD RULES (a violation means the metric is rejected):
  1. PHYSICAL ONLY. Score from these arrays. NO LLM judgment, NO text, NO
     calls to anything but numpy. `import numpy as np` is the only import.
  2. spec_score MUST be a finite float in [0, 1]. Clip it.
  3. DETERMINISTIC. No np.random, no time, no global state.
  4. NEVER raise. Guard missing/short arrays → return {"spec_score": 0.0}.
  5. MONOTONE IN TRUE COMPETENCE + NOT GAMEABLE. A more-competent policy must
     score higher. Specifically AVOID these known traps:
     - Don't reward raw motion/energy (a flailing/vibrating policy must
       score LOW, not high).
     - Don't use peak/median RATIOS (an under-trained still-mostly policy
       games them); prefer absolute, saturating quantities.
     - Gate locomotion on UPRIGHTNESS and stance HEIGHT (a belly-crawl with
       a level torso at ground height is not walking).
     - Gate "kick"/transient tasks on upright windows + a return-to-rest
       (continuous motion is not a discrete kick).
     - For rhythmic/periodic goals, measure structure (e.g. anti-phase),
       not just that something oscillates.
  6. Use a SATURATING form for unbounded quantities, e.g.
     `1 - np.exp(-x / scale)`, so the score can't be inflated arbitrarily.
  7. Return useful named subcomponents alongside spec_score (for debugging).

Think about which physical signature DEFINITELY distinguishes success from
the failure modes, then encode exactly that. Output ONLY the Python module.
