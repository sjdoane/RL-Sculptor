# MODE-SCOPE REVIEW (this metric grades ONE MODE, not the episode)

The metric under review scores a single **mode** of a hybrid automaton (OGMP,
arXiv 2403.04205) — one bundled sub-behavior of a longer motion. It is called
on the rollout already sliced to that mode's window: axis 0 of every array
covers only this mode, and `behavior["max_episode_steps"]` is the slice length,
not the episode length.

Apply the full rubric above, plus these four checks. Any one failing is a
reject; a concrete degenerate policy that beats the metric inside the window is
a `gaming_exploit`.

1. **Window-local gameability — the sharpest question here.** An episode-wide
   gaming policy must fool every phase at once; a mode-scoped one only has to
   fool a second or two. For each scored channel, construct the SHORT
   degenerate behavior that maximizes it within this window alone and check
   whether the metric separates it. Channels that are adequately hard to game
   across a whole episode (accumulated travel, mean uprightness, peak joint
   speed, total energy) frequently are not, inside one mode.

2. **A completion gate scoped to the WRONG level.** The gate must fire on this
   mode's own sub-goal. A gate that can only be satisfied by the terminal state
   of the whole behavior scores every non-terminal mode zero — which destroys
   exactly the localization per-mode scoring exists to provide. Reject it.

3. **Hidden assumptions about slice length or start state.** Reject any
   absolute frame index, any required minimum number of frames, any "the
   transition happens at t=X", and any gate that assumes the mode begins from a
   nominal standing reset. A mode starts wherever the previous mode left the
   robot, and a slice may arrive truncated because the policy stalled and the
   episode ended mid-mode.

4. **Evidence borrowed from a neighbouring mode.** The metric cannot observe
   any other mode. If a check is really about what happens before or after this
   window, it must ABSTAIN — a magnitude proxy substituted for evidence that is
   structurally absent is the LAW 6 violation, and here the absence is
   guaranteed by construction rather than merely likely.
