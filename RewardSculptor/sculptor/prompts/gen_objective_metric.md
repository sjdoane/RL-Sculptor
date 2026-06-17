You are an expert RL evaluation engineer. Write an OBJECTIVE TASK-SUCCESS
METRIC for a robot behavior goal — the ground-truth fitness used to judge
whether a learned policy achieves the goal. This is NOT a reward function:
it must be an HONEST, hard-to-game measurement of task competence.

Output a single self-contained Python module (and nothing else outside the
code fence). It may declare an optional module constant naming the joint
ROLES it reads, and must define `compute_spec`:

    import numpy as np

    REQUIRED_JOINT_ROLES = ["left_hip_pitch", "left_knee"]   # only if you read joints

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
  meta["joint_roles"]            dict {role: column index} — the runtime
                                 resolves your REQUIRED_JOINT_ROLES against the
                                 robot's actual joints and hands you the indices
  meta["joint_names"]            list[str] aligned with axis J (raw names; you
                                 normally do NOT need these — use joint_roles)
Any array may be ABSENT — always `arrays.get(k)` + guard for None.

JOINT RESOLUTION SAFETY (a violation means the metric is rejected):
  * To read a SPECIFIC joint, declare it in REQUIRED_JOINT_ROLES and read its
    column from meta["joint_roles"] — NEVER a hard-coded integer column. A
    literal `joint_vel[:, :, 0]` reads a DIFFERENT joint on every robot and is
    rejected. (The `[..., 2]` form is fine for the 3-vector gravity/root axes.)
        roles = (meta or {}).get("joint_roles", {})
        idx = [roles[r] for r in ("left_knee", "right_knee") if r in roles]
        if not idx:               # role unavailable on this robot → honest 0
            return {"spec_score": 0.0}
        knee_speed = np.abs(arrays["joint_vel"][..., idx])
  * Role names are FUNCTIONAL and DIRECTION-AWARE: side + segment + axis, e.g.
    `left_hip_pitch`, `right_hip_roll`, `left_knee`, `left_ankle_pitch`,
    `left_shoulder_pitch`. A bare `left_hip` is AMBIGUOUS (pitch/roll/yaw) and
    will be rejected — always name the axis.
  * DIRECTION MATTERS. A FORWARD kick / forward gait lives in the SAGITTAL
    plane: hip PITCH, knee, ankle PITCH. Hip ROLL is sideways abduction, hip
    YAW is twist. A forward-kick metric must read hip_pitch + knee (NOT
    hip_roll), or it scores a sideways kick as highly as a forward one (a real
    failure we have seen). Pick the roles whose motion DEFINES the goal.

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
     - For a skill performed FROM A STANDING STANCE (kick, balance, floss,
       in-place jump), gate on a roughly STATIONARY base — multiply by
       `exp(-horizontal_speed / scale)` using `root_link_pos_w`. A forward
       WALKER must score LOW: its gait hip/knee swings look like bursts, so
       a metric that doesn't penalise base travel rewards walking, not the
       skill (this is the exact failure that stalled a real kick run).
     - For rhythmic/periodic goals, measure structure (e.g. anti-phase),
       not just that something oscillates.
  6. Use a SATURATING form for unbounded quantities, e.g.
     `1 - np.exp(-x / scale)`, so the score can't be inflated arbitrarily.
  7. Return useful named subcomponents alongside spec_score (for debugging).

Think about which physical signature DEFINITELY distinguishes success from
the failure modes, then encode exactly that. Output ONLY the Python module.
