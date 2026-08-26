# MODE SCOPE (this metric grades ONE MODE, not the episode)

Everything above still applies. This block narrows WHAT you are measuring.

The behavior is a hybrid automaton (OGMP, arXiv 2403.04205): an ordered set of
**modes**, each a bundled sub-behavior, joined by transition guards. You are
writing the objective metric for exactly ONE of them. The data block below
names which mode, its window, and what runs before and after it.

## What you actually receive

`compute_spec` is called on the rollout **already sliced to this mode's
window**. Concretely:

  * `arrays[...]` axis 0 covers THIS MODE ONLY. Frame 0 is the instant the mode
    begins; the last frame is the instant it ends.
  * `behavior["max_episode_steps"]` is the length of THIS SLICE, not the
    episode. So is `behavior["mean_episode_length"]`.
  * `behavior["step_dt"]` is unchanged — it is the real control timestep.
  * You cannot see any other mode. There is no way to read what happened
    before this window or after it, and no way to reward it.

## Rules that change inside a mode

1. **"Completed" means THIS MODE completed.** The base rubric's hard rule —
   `spec_score = completion_gate * min(channels...)` — is unchanged, but the
   gate is on this mode's own sub-goal. For an `approach` mode the gate is
   "arrived", not "the whole behavior succeeded". A gate that can only be
   satisfied by the terminal state of the WHOLE behavior scores every
   non-terminal mode zero, which makes the per-mode score useless exactly
   where it is most needed.

2. **Do not assume the mode starts from rest, or from a nominal pose.** A mode
   begins wherever the previous mode left the robot — mid-stride, airborne,
   already leaning. The base rubric's start-state rules still hold, but the
   reference point is this mode's own first frames, not a standing reset.

3. **Never key on absolute time or on a fraction of the whole behavior.** The
   RELATIVE-TIME rule above is stricter here: within a mode the slice length
   varies with the control rate AND with how much of the window the episode
   actually reached. A slice may arrive TRUNCATED (the policy stalled and the
   episode ended mid-mode). Score honestly what is present; do not require a
   minimum number of frames, and do not index a fixed absolute frame.

4. **Assume a shorter window is easier to game, and compensate.** An
   episode-wide gaming policy has to fool every phase at once; a mode-scoped
   one only has to fool a window of a second or two. A channel that is
   adequately hard to game over a whole episode — total travel, mean
   uprightness, peak joint speed — is often trivially satisfiable inside one
   mode by a policy doing nothing else. For each channel you score, name to
   yourself the degenerate two-second behavior that maximizes it, and add the
   requirement that separates it.

5. **Do not reach for a neighbouring mode's evidence.** If the only honest way
   to confirm this mode succeeded is to observe the NEXT mode (e.g. "the grasp
   worked because the lift later held"), that check belongs to that mode or to
   the transition guard, not here — and per the DATA-SUFFICIENCY rule you must
   ABSTAIN rather than substitute a proxy for it.

6. **A reference, when attached, is already cropped to this mode.** Its frame 0
   is the mode's first frame. Ground your thresholds in it as usual; it is real
   data for this phase, not for the whole behavior.
