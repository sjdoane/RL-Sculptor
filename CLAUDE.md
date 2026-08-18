# Claude project instructions

`AGENTS.md` is the single authoritative engineering contract for this
workspace. Read it before changing either `RewardSculptor/` or
`reward-sculptor-ui/`, then read the newest section of `HANDOFF.md` for current
state. Do not duplicate or weaken those rules here.

For work that changes researcher-facing behavior:

1. Trace the UI choice through its API model, immutable receipt, worker
   command, observed runtime event, result artifact, and knowledge-graph edge.
2. State the honest capability boundary in the UI and documentation.
3. Add the negative regression test that would have caught the prior failure.
4. Run the focused core/backend/frontend checks and browser QA described in
   `AGENTS.md` before committing.

Historical heartbeat instructions are evidence, not current authority. Never
launch or mutate a long-running GPU job unless the active user request asks for
that run.
