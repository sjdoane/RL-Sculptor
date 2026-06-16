# Handoff prompt — KG-preview hang regression

Paste this into a new Claude Code window. Then attach the full repo
(`~/projects/`) so the agent can read the referenced files.

---

You are taking over a Reward Sculptor session. Start by reading these
three files, top-to-bottom, in order:

1. `~/projects/CONTEXT.md` — the first entry (2026-04-23 03:30) frames
   the open regression.
2. `~/projects/NEXT_WINDOW_DIRECTIVE_v3.md` — your full directive,
   including file map, §5 regression details, and testing protocol.
3. `~/projects/HANDOFF_KG_PREVIEW_HANG.md` — this file.

## Sam's goals (restated)

1. **Autonomous RL reward iteration** — train → rollout → diagnose → edit
   → commit, grounded in a knowledge graph of RL-reward papers.
2. **UI-reachable workflow** — every feature must be reachable from
   `./run.sh`; the terminal is an escape hatch, not the primary path.
3. **Overnight ambitious runs** on G1 biped (cartwheel / backflip)
   without babysitting.
4. **Right now, the reward-prompt tab is broken**, so he can't edit
   reward functions via the UI. This is the one blocker for everything
   else.

## Your task

Diagnose and fix the following regression. **Do not ship code before
Sam greenlights your plan.**

### The bug

A reward-prompt edit job hangs for 5 minutes and times out. Sam's
activity log from the 03:22 run (verbatim):

```
03:22:51 [reward_prompt_edit] start — validating parent + loading adapter
03:22:51 [reward_prompt_edit] dispatching to Claude (timeout=300s)
03:22:51 [reward_prompt_edit] loading adapter + reward_contract
03:22:51 [reward_prompt_edit] opened KG at graph.db
03:22:51 [reward_prompt_edit] KG preview query (first call may take 60-120s on cold embedding model)
<<< HANG — no further events >>>
```

Sam's statement: **"this wasn't before your recent fixes so it was a
new issue made"** — meaning Ship 9 or Ship 10 (documented in CONTEXT.md
2026-04-23 entries) introduced this. The same prompt completed in 1-2
minutes before those ships.

### Where the hang lives

`reward-sculptor-ui/backend/services/reward_jobs.py` lines 139-142:

```python
kg_preview_matches = query_semantic(
    user_prompt, top_k=5, store=store,
    min_similarity=_MIN_SIM,
)
```

Neither the "KG preview done" success emit (line 153) nor the "KG
preview failed" except-branch (line 144) fires. So the call is
**blocked**, not erroring.

### Three candidates to investigate (see `NEXT_WINDOW_DIRECTIVE_v3.md` §5.3 for full detail)

1. **Prewarm/worker deadlock.** Ship 10 added `_prewarm_embedding_model`
   in `backend/main.py` that fires `_asyncio.to_thread(_load_embedder)`
   at startup. Reward-prompt worker thread loads the same embedder. If
   concurrent init isn't safe, they deadlock. **Fast test:** run
   `RS_SKIP_EMBEDDER_PREWARM=1 ./run.sh`; if the hang disappears, this
   is it.
2. **KG embedding-backfill on first query.** If `query_semantic` kicks
   off a lazy "embed every technique that's missing its embedding"
   backfill on a grown KG, 416+ techniques serialize through HF
   inference. **Fast test:** `sqlite3 ~/.local/share/sculptor/kg/graph.db
   "SELECT COUNT(*) FROM techniques WHERE embedding IS NULL"`.
3. **Ship 9c lock held by PDF extract.** Less likely. Check
   `backend/routes/physics.py:extract_datasheet_pdf` for any stale
   `asyncio.Lock` / `threading.Lock`.

### What you should do first

1. Run the two fast tests above and report which culprit matches.
2. Read `sculptor/kg/query.py:query_semantic` + `_get_embedder` +
   `_ensure_technique_embeddings` (or equivalent) to confirm the hang
   mechanism.
3. Add heartbeat emits inside `query_semantic` consistent with the
   Ship-10 `log_line` pattern so the next hang pins the exact sub-step.
4. **Write a plan — which culprit, what fix, what regression tests —
   and ask Sam to greenlight before shipping.**

### Non-goals (explicit)

- Don't rewrite `query_semantic` or the embedder broadly.
- Don't revert Ship 10 as a first step — the heartbeats already pinned
  the hang; they're doing their job.
- Don't touch the Ship 1-9 features without a direct reason.
- Don't run the pre-existing sluggish
  `test_reward_prompt_edit_emits_log_line_events` full-suite — skip it
  with `-k "not test_reward_prompt_edit"` for baseline runs.

### Sam's style

Terse, file:line over prose, confirm destructive ops, no emojis in
files. Shell is **Windows Git-Bash**, not WSL — use `wsl bash <<'EOF'`
heredoc for anything with `$variables`, globs, or `~` expansion. After
every ship, run an independent critique agent. One commit per sub-ship,
pushed to https://github.com/sjdoane/RL-Sculptor.

---

Start by reading the three doc files and running the two fast tests.
Report back with your diagnosis and a plan.
