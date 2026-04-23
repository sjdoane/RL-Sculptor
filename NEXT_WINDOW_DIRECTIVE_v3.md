# Reward Sculptor — next-window directive v3 (2026-04-23)

Read top-to-bottom before touching code. Ships 1-10 landed in the previous
window; **one regression is unresolved and is your main target.** Baselines
stay green for every sub-ship.

This file supersedes `NEXT_WINDOW_DIRECTIVE.md` (v1, 2026-04-22) and
`NEW_WINDOW_DIRECTIVE.md` (v0, 2026-04-22). The original v1 listed §7.1
through §7.7 of the P0/P1 plan — **all of those shipped as Ships 1-7.**
Ships 8-10 are additional asks from Sam that landed in the same window.

**DO NOT SHIP IMPLEMENTATION before Sam greenlights the diagnosis +
plan.** Same structure as v1: read, plan, confirm, then ship
test-by-test. After every ship, run an independent critique agent.

---

## 1. Who / what

Sam Doane — USC ME undergrad, AME456 quadruped-jumping SEA capstone.
RTX 5070 Laptop (8 GiB VRAM, sm_120), WSL2 Ubuntu 24.04, Python 3.13 via
uv. Shell is **Windows Git-Bash, not WSL** — use `wsl bash <<'EOF'`
heredoc for anything with `$variables`, globs, or `~` path expansion.

Style: terse, file:line over prose, no emojis, confirm destructive ops,
no scope creep. Every feature must be UI-reachable — the rule is
"**no terminal commands beyond `./run.sh`** except first-time
`uv sync` / `pnpm install`, `export ANTHROPIC_API_KEY=…`, and
`uv run pytest` for dev/CI." See memory file
`feedback_no_terminal_after_run_sh.md`.

## 2. Stack

```
~/projects/                              # RL-Sculptor monorepo root (git-init'd)
├── AME456/                              # quadruped capstone (gitignored, read-only)
├── RewardSculptor/                      # sculptor pkg
│   ├── sculptor/
│   │   ├── adapters/
│   │   │   ├── mjlab.py
│   │   │   ├── _mjlab_runner.py        # subprocess CLI — trajectory capture lives here
│   │   │   ├── mjcf_editor.py           # NEW (Ship 8b) — Claude MJCF editor + filelock
│   │   │   ├── realism.py               # NEW (Ship 3) — audit_rollout verdict
│   │   │   └── auto_physics.py          # NEW (Ship 4) — synthesize auto-physics prompt
│   │   ├── diagnose.py                  # Eureka reward-reflection block lives here
│   │   ├── edit.py                      # apply_edits + apply_prompt_edit
│   │   ├── kg/                          # SculptorKG + query + ingest + extract
│   │   ├── prompts/
│   │   ├── sculpt.py                    # outer loop + _should_early_stop + auto-physics
│   │   └── cli.py                       # `sculpt` entry point
│   └── tests/
└── reward-sculptor-ui/                  # FastAPI + React
    ├── backend/
    │   ├── main.py                      # create_app — prewarm embedder startup hook
    │   ├── services/
    │   │   ├── reward_jobs.py           # reward-prompt edit worker — HEARTBEAT SITE
    │   │   ├── physics.py               # wraps sculptor.adapters.mjcf_editor
    │   │   ├── run_manager.py           # CLI-flag forwarding for all 10+ run params
    │   │   └── kg_jobs.py               # ingest/extract jobs
    │   ├── routes/
    │   │   ├── physics.py               # motor-limits form + PDF datasheet upload
    │   │   ├── projects.py              # GET/PATCH /projects/{slug}/settings
    │   │   ├── kg.py                    # heal-stubs endpoint
    │   │   └── rewards.py
    │   └── tests/
    └── frontend/
        └── src/components/
            ├── RunsTab.tsx              # realism + physics-edit chips
            ├── PhysicsTab.tsx           # MotorLimitsCard + datasheet upload
            ├── NewRunDialog.tsx         # Advanced tab — every RL knob
            ├── ProjectSettingsDialog.tsx # IterationSettingsSection
            └── KnowledgeGraphTab.tsx    # heal-stubs button
```

Running: `cd ~/projects/reward-sculptor-ui && ./run.sh`.
Shared KG: `~/.local/share/sculptor/kg/graph.db` (71 papers, 416
techniques as of 2026-04-22). Auto-seeded from `examples/kg_preextracted.db`
on first launch — see `backend/main.py:_bootstrap_shared_kg`.

## 3. Reference documents (read in order)

1. **[CONTEXT.md](CONTEXT.md)** — newest-first change log. The top entry
   (2026-04-23 03:30) is the Ship 10 + regression summary that frames
   your task.
2. **This file** — the directive you're reading.
3. **[HANDOFF_KG_PREVIEW_HANG.md](HANDOFF_KG_PREVIEW_HANG.md)** — the
   short new-window prompt with Sam's evidence and the specific regression.
4. **[NEXT_WINDOW_DIRECTIVE.md](NEXT_WINDOW_DIRECTIVE.md)** — v1 directive.
   Section §7 is all shipped; Sections 1-4 are still current (robot setup,
   project flow, etc).
5. **[QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md)** — T11 overnight
   acceptance checklist. The ambitious G1 overnight run recipe
   (see §8 below) still pending live execution.
6. **[MJLAB_PIVOT_DESIGN.md](MJLAB_PIVOT_DESIGN.md)** — §1.3 reward
   injection + §1.4 state snapshot.
7. **Eureka paper** at [arxiv:2310.12931](https://arxiv.org/abs/2310.12931).
   Full text saved at `~/projects/_eureka_fulltext.txt` (gitignored).

## 4. Ships 1-10 summary (what landed in the previous window)

| # | Date | Scope | Sculptor files | UI files | Tests added |
| - | ---- | ----- | -------------- | -------- | ----------- |
| 1 | 04-23 05:20 | §7.1 expanded rollout trajectory + training-side `SculptorRewardTerm` component sink | `adapters/_mjlab_runner.py` | (none) | 6 |
| 2 | 04-23 05:27 | §7.2 Eureka reward-reflection block in `diagnose.py` + `edit.py` | `diagnose.py`, `edit.py` | (none) | 5 |
| 3 | 04-23 05:38 | §7.3 physics-realism audit (`adapters/realism.py`) + RunsTab chip | `adapters/realism.py`, `sculpt.py` | `RunsTab.tsx` | 8 |
| 4 | 04-23 05:51 | §7.4 MVP auto-physics suggestion on severe verdict | `adapters/auto_physics.py`, `sculpt.py` | `RunsTab.tsx` | 7 |
| 5 | 04-23 05:58 | §7.6 project-tab resilience under GPU contention | (none) | `DashboardTab.tsx` error-handling | 4 |
| 6 | 04-23 06:04 | §7.7 arxiv retry-with-backoff + `heal_stub_titles` + seed titles | `kg/ingest.py`, `cli.py` (`sculpt kg heal-stubs`) | (none) | 9 |
| 7 | 04-23 06:56 | Real-time video (playback-speed/render-every/rollout-fps) + every RL knob UI-exposed + heal-stubs button | `sculpt.py`, `cli.py`, `adapters/_mjlab_runner.py` | `NewRunDialog.tsx`, `KnowledgeGraphTab.tsx` | 12 |
| 8 | 04-23 07:45 | 8a: per-project settings UI + TOML upsert. 8b: `mjcf_editor.py` refactor + full physics auto-apply. 8c: `run.sh` orphan reclaim | `adapters/mjcf_editor.py`, `sculpt.py` | `ProjectSettingsDialog.tsx`, `RunsTab.tsx`, `run.sh` | 23 mjcf + 15 settings |
| 9 | 04-23 09:46 | 9a: configurable early-stop. 9b: motor-specs form on Physics tab. 9c: datasheet PDF upload + `asyncio.wait_for` bounds Claude call. | `sculpt.py`, `cli.py` | `routes/physics.py`, `PhysicsTab.tsx` | 8 + 10 + 9 |
| 10 | 04-23 03:30 | Heartbeat `log_line` events in `reward_jobs._do_edit` + `_prewarm_embedding_model` startup hook | (none) | `backend/main.py`, `backend/services/reward_jobs.py` | 2 |

Baselines after Ship 10:
- Sculptor: **191 passed, 1 skipped** (Ship-9 total).
- Backend per-file: all 20 test files pass individually; full-suite has a
  pre-existing `test_reward_prompt_edit_emits_log_line_events` hang that's
  unrelated to Ship 9/10 changes — see §6 open follow-ups.
- Frontend `tsc`: exit 0.

## 5. Your primary task — diagnose + fix the KG-preview-hang regression

### 5.1 The evidence (verbatim from Sam's 03:22 run)

```
03:22:51 AM [reward_prompt_edit] start — validating parent + loading adapter
03:22:51 AM [reward_prompt_edit] dispatching to Claude (timeout=300s)
03:22:51 AM [reward_prompt_edit] loading adapter + reward_contract
03:22:51 AM [reward_prompt_edit] opened KG at graph.db
03:22:51 AM [reward_prompt_edit] KG preview query (first call may take 60-120s on cold embedding model)
<<< HANG — no further events, times out 5 minutes later >>>
```

Sam's quote: **"this wasn't before your recent fixes so it was a new
issue made."** The same prompt worked in 1-2 minutes before Ships 9+10.

### 5.2 What you know about the failing call

The hang is at [reward_jobs.py:139-142](reward-sculptor-ui/backend/services/reward_jobs.py):

```python
kg_preview_matches = query_semantic(
    user_prompt, top_k=5, store=store,
    min_similarity=_MIN_SIM,
)
```

- The "KG preview done" emit at line 153 never fires.
- The "KG preview failed" except-branch at line 144 never fires either —
  so the call is **blocked**, not erroring.
- Sam's KG UI shows all checkmarks (papers and techniques presumed
  populated with embeddings).

### 5.3 Three most likely culprits (investigate in order)

1. **Prewarm-task deadlock with the reward-prompt worker thread.**
   Ship 10 added `_asyncio.to_thread(_load_embedder)` at uvicorn boot
   (`backend/main.py:_prewarm_embedding_model` line 139). The reward-
   prompt worker thread also loads the embedder via
   `query_semantic → _get_embedder`. If sentence-transformers / huggingface
   isn't thread-safe on first init (concurrent downloads, tokenizer load,
   `torch.load` lock), both threads block waiting for each other. Test:
   set `RS_SKIP_EMBEDDER_PREWARM=1` before starting `./run.sh`; if the
   hang disappears, the prewarm is the culprit and needs a lock-before-
   load pattern or an "already initialized" short-circuit.

2. **`_ensure_technique_embeddings` backfill on a grown KG.** Check
   `sculptor/kg/query.py` for an embedding-backfill path that runs on
   the first `query_semantic` call. If the KG grew (new techniques
   without `embedding BLOB` populated), backfill serializes 416+
   techniques × HF inference. Test: `sqlite3 ~/.local/share/sculptor/kg/graph.db
   "SELECT COUNT(*) FROM techniques WHERE embedding IS NULL"`. If non-
   zero, the backfill is the hang. Fix: surface progress via
   `_emit_from_worker` events, and/or run backfill as a separate
   background task at startup.

3. **Ship 9c's `asyncio.to_thread` in the datasheet PDF extract route
   holding a stale lock.** Less likely given Sam was on Rewards tab, but
   worth eliminating. Check `backend/routes/physics.py:extract_datasheet_pdf`
   for any `asyncio.Lock` / `threading.Lock` that could linger.

### 5.4 Instrumentation before you fix

Before touching any code, add diagnostics to `query_semantic` itself
(inside `sculptor/kg/query.py`) so the next hang pins the exact line.
Examples: emit a `log_line` before `_get_embedder()`, before
`embedder.encode(...)`, before the sqlite `cursor.execute`, and before
the cosine loop. This is consistent with the Ship-10 heartbeat pattern
and cheap. Report your instrumentation plan + diagnosis before writing
any fix code.

### 5.5 Explicit non-goals

- **Do NOT** broadly rewrite `query_semantic`, the embedder, or the
  prewarm hook. The goal is a tight, regression-pinned fix.
- **Do NOT** revert Ship 10 as a first step. The heartbeats are working
  — they pinned the hang to `query_semantic`. Only revert the prewarm
  hook if you prove it's culprit #1 above.
- **Do NOT** run the sluggish full-suite test that hits
  `test_reward_prompt_edit_emits_log_line_events` — use `-k "not
  test_reward_prompt_edit"` for baseline runs, and pin the specific
  new-regression test you add as a gate.

## 6. Open follow-ups (nice-to-haves, lower priority than §5)

- **§7.5 failing-mode KG query** — diagnose.py should also `query_by_failure_mode`
  based on the audit verdict. Currently only does semantic query. See
  QUALITY_PASS_PLAN.md §T3.
- **Pre-existing test hang** — `test_reward_prompt_edit_emits_log_line_events`
  and `test_library.py` pass individually but hang mid-sequence. Likely
  HF embedding-model cache / network issue during parallel test runs.
  Worth a dedicated diagnosis pass after §5 lands.
- **Overnight G1 biped test** — the ambitious cartwheel / backflip recipe
  Sam requested in CONTEXT 06:56. Not yet executed; blocked on §5 being
  fixed so the reward-prompt tab is usable again.
- **Frontend vitest setup** — no vitest config in this repo. All frontend
  validation is manual / `tsc`.

## 7. Testing protocol

After §5:
- Add 1-2 regression tests pinning the specific hang you fix.
- Re-baseline: `cd RewardSculptor && uv run pytest` should show 191 passed.
- Re-baseline: `cd reward-sculptor-ui && uv run pytest backend/tests/ -k
  "not test_reward_prompt_edit_emits_log_line_events"` should show the
  per-file totals green.
- Live smoke: `./run.sh`, open a project, submit a reward-prompt edit
  with the same text that failed (Sam's saved prompt). Timestamped
  activity log should show every Ship-10 heartbeat firing, AND a new
  heartbeat from your `query_semantic` instrumentation showing the
  sub-step timings, AND the edit completing within 2-3 min.

## 8. Ambitious overnight G1 recipe (still pending)

Details in `CONTEXT.md` 2026-04-23 06:56 (the Ship 7 entry's "ambitious
project" handoff). Short version: G1 humanoid + backflip goal, 15
iterations × 2500 steps, seed=42, rollout_episodes=3, auto_adjust_physics=
true, early_stop_patience=5. Runs ≈6-8 h. DO NOT start this until §5 is
fixed — it relies on the reward-prompt tab for mid-run edits if the
initial reward is stuck.

## 9. Commit discipline

`~/projects/` is git-init'd and pushed to
**https://github.com/sjdoane/RL-Sculptor**. Commit every meaningful
change:
- One commit per sub-ship.
- Message format: `Ship NN: <one-line summary>` then a paragraph with
  file:line references and test counts.
- Co-author footer: `Co-Authored-By: Claude Opus 4.7 (1M context)
  <noreply@anthropic.com>`.
- Never commit `.env`, `.venv/`, `node_modules/`, `*.db`, `runs/`,
  `AME456/`. `.gitignore` already excludes all of these.

## 10. Sam's durable rules (memory)

- Terminal-free workflow (see above).
- Rollout video fps must match sim dt (Ship 7 delivered this — the knob
  is `--rollout-fps` in CLI, "Rollout video fps" in NewRunDialog).
- Edit tool strips exec bit — `chmod +x` after every `.sh` edit.
- Don't run `grep` / `find` / `cat` — use the dedicated Grep/Glob/Read
  tools.
- Use `wsl bash <<'EOF'` for `$variables` / heredocs (Git-Bash quirks).
- No emojis in files unless user asks.
- Confirm before destructive ops.
- Run an independent critique agent after every ship.

---

**Your next message should be a short plan** — which culprit(s) you'll
instrument and in what order, what tests you'll add, and whether you
want Sam's greenlight before shipping. Do not write code yet.
