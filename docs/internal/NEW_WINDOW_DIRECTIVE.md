# Reward Sculptor — new-window directive

Read top to bottom before touching anything. You are picking up a half-
finished tool that partially works on one robot (Unitree Go1) and
partially doesn't work on any of the others. Sam is testing it
end-to-end from the browser UI and has hit a rat's nest of bugs that
this document catalogues. Your job (when Sam greenlights) is to plan
the path from "works on Go1 if you nurse it" to "Sam can pick any
robot, describe any behavior, and have the tool iterate a reward +
physics + policy with confidence". Do NOT ship fixes until Sam
explicitly asks — he wants a PLAN from you first, anchored in this
document.

---

## 1. Who / what

- **User**: Sam Doane — USC ME undergrad. AME 456 capstone = **quadruped
  jumping robot with series-elastic actuators (SEAs)**. Goal: hop, jump
  over obstacles, use SEAs to absorb landings + recover stored energy.
- **Hardware**: RTX 5070 Laptop (8 GiB VRAM, sm_120, CUDA 13.0). 16 GiB
  system RAM. WSL2 Ubuntu 24.04 on Windows.
- **Env**: Python 3.13.5 via `uv`. `ANTHROPIC_API_KEY` already set in
  `~/projects/RewardSculptor/.env` (auto-loaded).
- **Working style**: terse, file:line over prose, confirm before
  destructive actions, no scope creep, no emojis. Memory at
  `~/.claude/projects/.../memory/MEMORY.md` has more — read it.
- **Shell gotcha**: shell is Windows Git-Bash, not WSL bash. Use
  `wsl bash <<'EOF' ... EOF` heredoc for anything with `$variables`
  or globs.

## 2. Stack

```
~/projects/
├── AME456/                     # Sam's capstone repo (DO NOT modify)
├── RewardSculptor/             # library (sculptor pkg): diagnose, edit, adapters, KG
│   ├── sculptor/
│   │   ├── adapters/           # gym_sb3 + mjlab + stubs
│   │   │   ├── mjlab.py
│   │   │   └── _mjlab_runner.py   # THE subprocess mjlab invokes
│   │   ├── diagnose.py         # Claude-backed failure-mode analysis
│   │   ├── edit.py             # Claude-backed reward rewrite + pre/post-validate
│   │   ├── kg/                 # SculptorKG + query + research + extract
│   │   ├── prompts/            # edit_rewriter.md, physics_editor.md, ...
│   │   └── sculpt.py           # the outer loop (sculpt_run)
│   ├── tests/                  # pytest; 97 passing, 1 skipped (jax-gated)
│   └── examples/kg_preextracted.db    # 46 papers seed
└── reward-sculptor-ui/         # FastAPI backend + React/Vite frontend
    ├── backend/
    │   ├── routes/             # projects, rewards, runs, physics, kg, library, ...
    │   ├── services/           # physics, reward_jobs, run_manager, gpu_monitor, ...
    │   └── tests/              # 205 passing, 1 failing flake, 3 skipped
    ├── frontend/               # Vite + React + shadcn + Monaco + TanStack Query
    └── run.sh                  # one-shot start, exports MUJOCO_GL=egl
```

Running: `cd ~/projects/reward-sculptor-ui && ./run.sh` → uvicorn on
127.0.0.1:8000, Vite on 127.0.0.1:5173, browser opens automatically.

## 3. Reference documents (read in order of priority)

1. **[CONTEXT.md](CONTEXT.md)** — session history, newest-first. The
   last ~10 entries describe every fix shipped in the preceding 48 h.
2. **[QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md)** — the overnight-
   reliability pass; includes deferred items that are still deferred.
3. **[MJLAB_PIVOT_DESIGN.md](MJLAB_PIVOT_DESIGN.md)** — the original M0
   architecture doc for the mjlab adapter. Section §1.3 (reward
   injection) and §9 (preflight) are load-bearing.
4. **[M7_PLAN.md](M7_PLAN.md)** — M7 milestone which is complete, but
   §Phase 5 is the Physics tab design.
5. This file — the directive you're reading.

## 4. Baseline test counts (must not regress)

| suite | command | expected |
|---|---|---|
| sculptor | `cd ~/projects/RewardSculptor && uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py` | 97 passed, 1 skipped |
| backend | `cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q` | 205 passed, 3 skipped, **1 known flake** (`test_mjlab_rejects_insufficient_vram` — pynvml visibility in test pytest fixtures, unrelated to any real code path) |
| frontend typecheck | `cd ~/projects/reward-sculptor-ui/frontend && PATH=$HOME/.local/share/pnpm:$PATH node_modules/.bin/tsc --noEmit` | exit 0 |

## 5. Current state — what works

Tested directly on Sam's RTX 5070 Laptop:

- **Go1 training** works end-to-end: 22 min per sculpt iter at
  `steps_per_iter=1500`, 2048 envs, ~55k env-steps/sec. Ran 6 iters
  overnight.
- **Rollout video** is produced at `runs/iter_<i>/rollout/rollout.mp4`
  via `_cmd_rollout` with parallel envs (num_envs=64) and rendering
  throttled to ~120 frames. ~40 s per rollout. UI's `RobotViewer`
  embeds this at `/projects/{slug}/runs/{run_id}/iterations/{iter_index}/rollout`.
- **KG** has 46 papers, 269 techniques, 688 edges. Shared DB at
  `~/.local/share/sculptor/kg/graph.db` (847 KB). `project_kg_db_path`
  returns the shared path for fresh projects; legacy per-project DBs
  still honored.
- **Resume** works: `--resume` passed by default from `run_manager`.
  `_train_or_resume` + `_rollout_or_resume` skip phases when artifacts
  are on disk.
- **Atomic checkpoint write** (`.pt.tmp + os.replace`) prevents corrupt
  files from mid-copy SIGKILL.
- **NaN/Inf guard** in `_call_compute_reward` rejects non-finite
  reward / component values at pre-flight.
- **Anthropic SDK retries** at 6 (SDK default 2) in all hot-path sites.
- **KG grounding mandate** in `edit_rewriter.md` + `physics_editor.md`
  prompts: every new/changed hyperparameter must cite an arxiv_id or
  physics first-principles justification. `REWARD_SPEC.grounding`
  dict is required.
- **Realism floor**: mjlab's default reward terms scaled to 0.3×
  (pre-pass: zeroed) — anti-spasm / upright / foot-behavior priors
  preserved.
- **`base_height` + `fallen`** signals added to reward's info dict.

## 6. Current state — what's broken (Sam's Cartpole test, 2026-04-22)

Each bug numbered — reference these numbers in your plan. Sam created
a fresh Cartpole project and ran through the full UI test flow. Every
tab revealed something. Scene-keys bug (#5) was fixed in this session;
everything else is still broken.

### 6.1 Reward-tab prompt edit hangs with no feedback

- **Repro**: Rewards tab → type `keep cartpole upright` → click "Prompt
  Claude". UI shows no progress for minutes. Try to send another prompt
  → 409 error: *"Another prompt-driven reward edit is in progress.
  Wait for it to finish before firing another."*
- **Symptoms**: backend job exists + is queued/running; UI doesn't
  surface progress or error. No `log_line` events stream. WS may be
  connected but the job's `runner` isn't emitting.
- **Where to look**:
  - [backend/services/reward_jobs.py::run_reward_prompt_edit_job](reward-sculptor-ui/backend/services/reward_jobs.py)
  - [backend/routes/rewards.py](reward-sculptor-ui/backend/routes/rewards.py)
  - [frontend/src/hooks/useJob.ts](reward-sculptor-ui/frontend/src/hooks/useJob.ts) — the polling hook
  - Check: is `apply_prompt_edit` (sculptor.edit) blocking on an
    Anthropic call that never returns? `max_retries=6` means up to ~4
    min of backoff on transient failures — user-visible as "hung".
- **Must-fix**: emit `log_line` / progress events from the reward prompt
  edit job. Surface "Claude is thinking..." / "validating..." / "done"
  states to the UI. Add a timeout so the job can't hang forever.

### 6.2 Overview tab: doesn't show the selected robot

- **Repro**: create project with library robot (Cartpole) → Overview
  tab. Shows the robot library GRID (same as project-creation screen)
  instead of details about the already-selected robot. Clicking a
  robot in the grid offers *"create new project from here"* — wrong
  affordance for an existing project.
- **What Sam wants**:
  - Overview shows the SELECTED robot's info: name, category, paper
    references, thumbnail, DOF counts, MJCF path, adapter + task_id.
  - A **"Edit robot setup"** prompt area — Claude rewrites the MJCF
    based on natural-language asks ("increase the pole mass 50%", "add
    a second pendulum"), grounded in the KG.
  - Controls for the foundational choices: swap task_id within the
    robot's task family (e.g. Go1 velocity vs Go1 tracking), change
    num_envs, change device.
- **Where to look**:
  - [frontend/src/pages/ProjectDetail.tsx](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx) — tab routing
  - grep for `OverviewTab` — probably a stub rendering the library grid
  - `GET /projects/{slug}` returns `ProjectDetail` with `library_slug`,
    `adapter_class`, `adapter_config`, etc. — all the info is there,
    just not surfaced.
- **Design ref**: this overlaps with the Physics tab (which edits MJCF
  content). Overview should be the high-level robot dashboard;
  Physics is the XML-diff drill-down.

### 6.3 Physics tab says "Physics editor unavailable" on library robot

- **Repro**: Cartpole project → Physics tab. Error card: *"Pick a
  library robot OR upload a URDF/MJCF first. Library-only projects
  require `robot_descriptions` to resolve the Menagerie MJCF path."*
- **Root cause** (verified): Cartpole's library entry has
  `source: mjlab_builtin`, slug `cartpole_mjlab`.
  `RobotLibrary.resolve_menagerie_path('cartpole_mjlab')` returns
  `None` because Cartpole is not in `mujoco_menagerie` — it ships
  inside mjlab itself. The physics tab's resolver only handles
  menagerie-sourced library entries, not `mjlab_builtin`.
- **Must-fix**:
  - Extend `resolve_mjcf_path` / `_resolve_library_mjcf` in
    [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py)
    to look up mjlab_builtin MJCFs from the mjlab package (e.g.
    `mjlab.tasks.cartpole.scene` or whatever mjlab exposes).
  - Reword the error when the issue is legitimately "no MJCF" vs
    "wrong library source kind".

### 6.4 KG "Research a topic" errors with Anthropic API mismatch

- **Repro**: KG tab → "Research a topic" → type "reward functions to
  make a robot do a cartwheel" → error banner:
  *`TypeError: Messages.parse() got an unexpected keyword argument 'response_format'`*.
- **Root cause**: [sculptor/kg/research.py:149-154](RewardSculptor/sculptor/kg/research.py)
  calls `client.messages.parse(response_format=ResearchResponse, ...)`.
  `response_format` is OpenAI SDK syntax; Anthropic's `messages.parse`
  uses different kwargs. Likely wants `tools=` + `tool_choice=` or
  just `response_model=` depending on SDK version.
- **Also affects**: [sculptor/kg/extract.py:185,203](RewardSculptor/sculptor/kg/extract.py),
  [sculptor/diagnose.py:325](RewardSculptor/sculptor/diagnose.py) —
  same `messages.parse(response_format=...)` call pattern. These are
  working (diagnose succeeded overnight) so maybe `response_format` IS
  valid in some SDK versions and research.py has a subtle difference.
  `pip show anthropic` first; diff the call sites.
- **Must-fix**: unify the pattern. If the SDK supports
  `response_format`, fix research.py. If it doesn't, migrate all
  sites to the supported pattern. Add a regression test.

### 6.5 (FIXED in this session) mjlab Scene entity enumeration

- **Was**: `env.scene.keys()` → `AttributeError` → masked to empty list
  → "Scene keys: []" KeyError when looking for the articulated entity.
- **Fix**: use `env.scene.entities` (the attribute that actually
  exists). Shipped in [_mjlab_runner.py::_find_articulated_entity](RewardSculptor/sculptor/adapters/_mjlab_runner.py).
- **Verification**: direct probe confirmed `env.scene.entities =
  {'terrain': ..., 'cartpole': ...}` for Cartpole.

### 6.6 Cartpole-specific state schema is minimal but UNTESTED end-to-end

- Added `_CARTPOLE_STATE_SCHEMA = {qpos:(2,), qvel:(2,), actuator_force:(1,)}`
  in [mjlab.py::_schema_for_task](RewardSculptor/sculptor/adapters/mjlab.py).
- Snapshot now tolerates missing `root_link_*` attrs (Cartpole is
  fixed-base).
- v0.py (alive-bonus only) should work; Claude's v1.py onwards WILL
  only access `qpos`/`qvel`/`actuator_force` if he respects the schema
  — which the edit_rewriter prompt now requires, but this is untested.

## 7. Ambitious goal set — what the tool SHOULD look like

Sam's vision, phrased so you know what to optimize for:

1. **Any supported robot works out of the box.** Pick Go1, G1, ANYmal,
   T1, Yam, Cartpole, any gymnasium robot — the tool Just Works. No
   per-task schema hardcoded. Adapter introspects the env to build
   the state snapshot.

2. **Every tab is useful for every project.**
   - **Overview** = robot dashboard + high-level Claude-prompted edits
     ("add a second pendulum", "swap task to tracking").
   - **Rewards** = v0 starter visible + prompt-edit flow with live
     progress + KG grounding visible + `grounding` dict surfaced.
   - **Physics** = MJCF viewer + prompt-edit flow + `Re-materialize`
     button for healing + works for `menagerie`, `mjlab_builtin`,
     `gymnasium_builtin`, and uploaded MJCFs.
   - **KG** = view + seed + ingest + extract + `Research a topic`
     works end-to-end.
   - **Runs** = timeline with per-iter progress bar + rollout video
     embedded + primary-metric chart + event log.
   - **Reports** = training CHANGELOG, reward-diff history, citation
     provenance.

3. **Claude decisions are literature-grounded by default.** Every
   reward hyperparameter change carries an arxiv citation. Every
   physics parameter change carries an arxiv citation or manufacturer-
   spec reference. `grounding` dicts surface in the UI so Sam can
   audit why Claude picked a value.

4. **Every async operation surfaces progress.** No silent hangs. Each
   of: prompt edit, physics edit, research-a-topic, bulk-seed-library,
   KG extract, sculpt run, rollout — all emit `[SCULPT-EVENT]` or
   equivalent progress lines. Hanging ≥ 30 s with no output is a bug.

5. **Resume is transparent.** Sam clicks Run. If the previous run
   errored mid-iter, the iter's artifacts are reused. No retrain of a
   completed phase. Phase-skipped breadcrumbs show in the UI timeline.

6. **Errors surface inline, not via toast.** Every failure mode surfaces
   in the tab where it happened: Physics tab errors show in the Physics
   tab with a Re-materialize button; reward-validator rejections show
   in the RejectionCard; run errors show in the Runs tab's error panel
   with the Claude-classified error_kind + suggested actions.

7. **The UI understands what's expensive and warns.** 12 iters × 22 min
   = ~4.5 h. If Sam hits Run at 2 pm with `iterations=12`, show an
   estimated completion time. If he's about to retrain a completed
   iter (resume disabled), warn him.

8. **The robot's behavior is visible at each iter.** Not just a metric
   number. Rollout video auto-loads in the Runs tab. Keyframes from
   diagnose are shown alongside. If the robot is reward-hacking (e.g.
   flipping to exploit upward-motion rewards), the Runs tab surfaces
   the specific hacking failure-mode with Claude's evidence quote.

9. **Sam can edit the robot + reward + physics from ONE conversational
   surface per tab.** Not "go to Physics tab, type prompt, wait 90 s,
   go back to Overview". Each tab's prompt-edit takes ≤ 30 s to
   surface first progress (streaming Claude output if possible).

10. **The tool is robust to overnight.** Power blip, backend restart,
    laptop sleep — any of these can happen and the run picks up where
    it left off. No silent corruption. Every artifact is atomic.

## 8. Failure modes the new window will hit

Read these. Don't learn them the expensive way.

- **Shell is Git-Bash, not WSL bash.** Any `$var` in a `wsl bash -c
  '...'` call will expand on the WINDOWS side. Use `wsl bash <<'EOF'`
  heredocs. Memory entry `feedback_wsl_shell_gotcha.md`.
- **Write/Edit tool strips exec bit.** After editing any `*.sh`,
  follow up with `wsl bash -c 'chmod +x <path>'`. Memory
  entry `feedback_exec_bit.md`.
- **Glob over `\\wsl.localhost\...` UNC paths times out.** Use
  `wsl bash -c 'find ...'` for file enumeration.
- **Backend tests redirect `RS_KG_PATH` per-test.** Any new route that
  reads the KG must use `backend.services.kg_store.shared_kg_db_path()`
  / `project_kg_db_path()` so tests stay isolated.
- **torch.cuda.is_available() caches at import.** If backend starts
  before CUDA is warm post-boot, stays False forever → kill + restart.
  `gpu_monitor._ensure_pynvml` has the same sticky-failure bug.
- **uvicorn --reload is active.** Any edit to `backend/` or
  `../RewardSculptor/sculptor/` triggers a reload. Job-manager state
  resets (sculpt subprocess continues — it's detached — but UI loses
  tracking). Avoid edits during active runs.
- **`--resume` is on by default.** Fresh projects are unaffected; but
  a rerun of an already-iterated project picks up at the next iter.
  If you want to force a restart, `rm runs/iter_<n>/` first.
- **mjlab's Scene doesn't implement `.keys()`.** Use `.entities`.
- **`env_cfg.scene.num_envs`** assignment is the right way to set
  num_envs for mjlab; don't try to edit the scene dict.
- **MUJOCO_GL=egl** must be set before importing mujoco. `run.sh`
  exports it. Subprocess inherits.
- **WANDB_MODE=disabled** must be set before rsl_rl's runner is
  constructed — `mjlab.py::_run_with_cleanup` sets it via
  `env.setdefault`. Don't rip that out.
- **mjlab runs in a detached process group** (`start_new_session=True`).
  Killing the backend doesn't kill mjlab. Use `os.killpg` with the
  pgid on cleanup (the `_run_with_cleanup` wrapper does this).
- **Test fixtures monkey-patch `_resolve_library_mjcf`** (physics
  tests). Any new library-source-resolution path needs the same seam.
- **JobManager.submit populates `job._cancel`.** Tests that inject
  `Job(...)` directly must set `job._cancel = asyncio.Event()`.
- **FastAPI lifespan events** are used for `_bootstrap_shared_kg` and
  `job_manager.bind_loop`. The app-startup shared-KG copy can fail
  silently — check `backend/main.py:84` area.

## 9. Suggested approach when Sam greenlights

Don't try to fix everything in one pass. Sam is testing end-to-end
from the UI. Structure your plan document as a test matrix:

| test | robot | tab | verifies | priority |
|---|---|---|---|---|
| T1 | Cartpole | Run | adapter plumbing + default reward | P0 |
| T2 | Cartpole | Rewards | prompt-edit round-trip + KG + progress | P0 |
| T3 | Cartpole | Physics | mjlab_builtin MJCF resolution | P0 |
| T4 | Cartpole | Overview | dashboard render + robot-edit prompt | P1 |
| T5 | Cartpole | KG | Research-a-topic API fix | P1 |
| T6 | Go1 | Run | realism floor + fall-detect on an actual flipper | P1 |
| T7 | Go1 | Rewards | grounding dict visible + citation UI | P1 |
| T8 | Go1 | Physics | SEA edit flow with KG grounding | P1 |
| T9 | G1 | Run | humanoid schema works | P2 |
| T10 | ANYmal-C | Run | second quadruped works | P2 |
| T11 | Any | Run | overnight 12-iter run completes | P2 |

For each P0, identify the CURRENT failure mode (§6 above), propose
the fix with file:line, and estimate time. Propose fixes that don't
regress the working Go1 path.

Ask Sam to prioritize within P0/P1 before you ship anything.

## 10. Starter prompt for the new window

```
Read ~/projects/NEW_WINDOW_DIRECTIVE.md end-to-end. Then:

1. Run the three test suites (§4). Confirm baselines match.
2. Generate a plan that addresses every bug in §6 + every goal in §7,
   prioritized against the T1..T11 matrix in §9. For each item: file
   path / line range to touch, rough complexity (S/M/L), regression
   risk, and test you'd add.
3. Do NOT implement anything until I confirm the plan.
4. After I approve, ship test-by-test. Log each in CONTEXT.md. Keep
   the test baselines green after every ship.
```

---

*End of directive. Index this into the new session's prompt and let
Claude plan from it.*
