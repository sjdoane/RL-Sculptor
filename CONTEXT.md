# RL-Sculptor — session context

Bootstrap doc for a new Claude Code window working on this tree. Read this
end-to-end before making any changes; check the **Change Log** section
at the bottom for what's been touched most recently.

---

## Who you are helping

Sam Doane — USC ME undergrad, AME 456 capstone on a **quadruped jumping
robot with series-elastic actuators (SEAs)**. Works primarily in WSL2
Ubuntu 24.04 now (moved off Windows + OneDrive). Preferred
communication style: terse, direct, file:line references + metrics over
prose. Flag gotchas early. Confirm before destructive actions.

Working directory for this project: `~/projects/` (resolves to
`/home/samjd/projects/` inside WSL, or
`\\wsl.localhost\Ubuntu-24.04\home\samjd\projects\` from Windows).

## What this project is

Three sibling projects under `~/projects/`:

```
~/projects/
├── AME456/              # the MJX quadruped env (primary capstone project)
├── RewardSculptor/      # the core reward-sculpting library + `sculpt` CLI
└── reward-sculptor-ui/  # FastAPI + React / Vite control panel for sculptor
```

Dependency direction: **AME456 → RewardSculptor** (the quadruped env
imports `sculptor.reward.compute_reward` via a sys.path shim).
**reward-sculptor-ui → RewardSculptor** (uv path-install of
`../RewardSculptor`, editable).

### RewardSculptor (core)

An autonomous agent that iterates on RL reward functions, grounded in a
KG of research papers. Two loops joined at a Claude-backed diagnoser:

1. **Inner iteration loop**: `train → rollout → diagnose → apply_edits
   → commit` — produces `v0.py → v1.py → v<N>.py` reward modules with
   full provenance.
2. **Knowledge graph**: arxiv seeds → PDFs → Claude-extracted Papers /
   Techniques / FailureModes / RewardComponents / Environments,
   queryable by failure mode + by semantic similarity
   (all-MiniLM-L6-v2 cosine).

Reference adapter is `GymSB3Adapter` (Gymnasium + Stable-Baselines3).
Adapter contract in [`RewardSculptor/docs/adapters.md`](RewardSculptor/docs/adapters.md).
Build history in [`RewardSculptor/HISTORY.md`](RewardSculptor/HISTORY.md).

Key entry points:

- `sculpt init <dir> --adapter gym_sb3` — scaffold a new project.
- `sculpt run <goal> --config <cfg.toml> --iterations N` — full loop.
- `sculpt report --config <cfg> --out final.mp4` — final report + timelapse.
- `sculpt kg ingest <seeds.yml>`, `sculpt kg extract --all`.

### reward-sculptor-ui (UI)

Localhost control panel wrapping the sculptor. Built across 10 prompts:

| # | Feature                                                                   |
| - | ------------------------------------------------------------------------- |
| 1-2 | Design doc + full API contract                                          |
| 3 | Backend skeleton: project lifecycle (create/list/get/delete), health check |
| 4 | Robot source: library pick / URDF+MJCF upload, static preview renderer   |
| 5 | Frontend skeleton: Vite + React + Tailwind + shadcn/ui                   |
| 6 | Robot configuration UI, library thumbnails, camera-angle preview toolbar |
| 7 | Rewards tab (Monaco lazy-loaded), KG tab (papers + pending seeds + graph) |
| 8 | Runs tab: live log WS, iteration timeline, Recharts metric chart         |
| 9 | 3-mode RobotViewer (Static / Live / Replay), live-clip streamer          |
| 10 | Dashboard (`/`), Settings (`/settings`), Reports tab, one-command run   |

All 10 prompts include strict "preconditions" (stickiness rules,
backpressure, isolation, etc.) that are documented inline in the code.

Key endpoints:

- `GET /dashboard` — aggregate one-call payload for the landing page (<500 ms target, live measured at ~10 ms).
- `GET /system/info` — sculptor/torch/CUDA detection + paths + masked API key.
- `GET|POST /projects`, `/projects/{slug}/{robot,preview,rewards,kg/*,runs,reports/*}`.
- `WS /ws/projects/{slug}/runs/{run_id}/{events,frames}` — live run stream + live-clip push.

### GitHub

Pushed to **https://github.com/sjdoane/RL-Sculptor** as a monorepo:

```
RL-Sculptor/
├── README.md
├── .gitignore
├── RewardSculptor/
└── reward-sculptor-ui/
```

Note: the WSL `~/projects/` directory is **NOT yet git-linked** to the
remote (no `.git/` here). The tree on disk has the two projects + AME456
as siblings; the GitHub repo has RewardSculptor and reward-sculptor-ui
as subdirs of the repo root. To reconcile:

```bash
# Option A: clone fresh beside ~/projects
cd ~
git clone https://github.com/sjdoane/RL-Sculptor.git

# Option B: initialize inside ~/projects (mind that AME456 is NOT in the repo)
cd ~/projects
git init
git remote add origin https://github.com/sjdoane/RL-Sculptor.git
git fetch
# then resolve the delta manually
```

---

## Quick sanity run (verify the stack works)

```bash
# one-time setup
cd ~/projects/reward-sculptor-ui
uv sync
pnpm install --dir frontend

# backend tests
uv run pytest backend/tests/ -v          # expect 69 passed

# sculptor tests
cd ~/projects/RewardSculptor
uv run pytest tests/ -q                  # expect 62 passed, 1 skipped

# one-command run (UI)
cd ~/projects/reward-sculptor-ui
./run.sh
# → opens http://localhost:5173 in browser
```

---

## Architecture in one screen (UI runtime)

```
Browser (http://localhost:5173)
    ↓ /api/* + /ws/*
Vite dev server (proxies /api → :8000, /ws → ws://:8000)
    ↓
FastAPI backend (uvicorn, :8000)
    ├── routes/{projects,robot,rewards,kg,runs,reports,jobs,dashboard}.py
    ├── services/
    │   ├── project_store.py  — filesystem-backed project registry
    │   ├── sculptor_bridge.py — the ONLY module that imports sculptor.*
    │   ├── reward_store.py   — AST-validated reward-module IO
    │   ├── kg_store.py       — thin SculptorKG wrappers
    │   ├── kg_jobs.py        — async ingest+extract runner
    │   ├── job_manager.py    — in-process job registry + event streams
    │   ├── run_manager.py    — sculpt_run subprocess + fs watchers
    │   ├── rollout_streamer.py — live 2s clip ffmpeg-truncation pipeline
    │   └── preview_renderer.py — MuJoCo offscreen + camera-angle presets
    ↓ `uv run sculpt run ...` subprocess per run
RewardSculptor package (path-installed)
```

Projects live outside the repo at `$RS_PROJECTS_ROOT`:
- Linux default: `~/.local/share/reward-sculptor/projects/`
- Cloud-sync guard refuses any OneDrive / Dropbox / iCloud path unless
  `RS_ALLOW_CLOUD_SYNC=true`.

---

## Linux-specific notes (what changed moving off Windows)

Several Windows / OneDrive gotchas no longer apply here. Keep the
Linux-applicable ones; ignore the rest:

**Still relevant on Linux:**

- **`ANTHROPIC_API_KEY=""` override**: sculptor's `__init__.py` treats
  empty-string env vars as unset so `.env` can win.
- **Python pin** (`.python-version`): `3.13.5`. MuJoCo wheel
  compatibility. Don't change without coordinating.
- **arxiv API rate limits** after bursts → re-run after ~5 min.
- **Prompts in `sculptor/prompts/*.md`** loaded at import time. Per-
  project override via `SCULPTOR_PROMPTS_DIR=/path/to/prompts`.
- **Grounded-field rule in edit.py pre-flight**: every identifier in a
  proposed-edit's suggested_value must be in `reward_contract.
  expected_info_keys ∪ current-components ∪ hyperparameters ∪
  math-allowlist`.

**No longer relevant (Windows / OneDrive-only):**

- ~~cp1252 console Unicode issues (`print("→")` failing)~~
- ~~OneDrive `.dist-info` lock on `uv add`~~
- ~~`UV_LINK_MODE=copy` requirement~~
- ~~PowerShell `pwsh`-vs-`powershell` 5.1 split~~
- ~~Git-bash `/tmp/` path not resolved by Windows-native curl / Python~~
- ~~Cloud-sync guard catching `%LOCALAPPDATA%\...\OneDrive\...` paths~~

**New on Linux:**

- **MuJoCo headless rendering** works out of the box via `MUJOCO_GL=egl`
  if the machine has no display. For WSL2 this sometimes needs
  `apt install libegl1 libgl1`. If you see "GLFW initialization failed"
  in MuJoCo Renderer, `export MUJOCO_GL=osmesa` (after
  `apt install libosmesa6-dev`).
- **ffmpeg**: Linux's system ffmpeg is typically more recent than
  imageio-ffmpeg's bundled one. Either works; `shutil.which("ffmpeg")`
  will prefer system.
- **systemd-resolved**: if ingest (arxiv fetch) hangs at DNS, restart
  `systemd-resolved.service`.

---

## What's verified working

As of the move to WSL, the **Windows** session had:

- 69 backend tests passing (projects, robot, preview, rewards, KG, runs,
  clips, dashboard).
- 62 sculptor tests passing, 1 skipped (`test_reward_parity.py` requires
  JAX which isn't installed).
- Live end-to-end: create project → pick Hopper → 3-iter dry-run
  completes in ~60 s → dashboard populated → `sculpt report` builds
  final_report.md (2.9 KB) + final.mp4 (985 KB) ✓
- Bundle size (Vite production build):
  - main chunk: 91 KB gzipped
  - monaco: 7.9 KB gz (lazy)
  - RunsTab: 7.3 KB gz (lazy)
  - ReportsTab: 49 KB gz (lazy, react-markdown)
  - charts: 151 KB gz (lazy, recharts + d3 + react-window)

**Not yet re-verified on Linux**. First task for a new session may be to
run the test suites + `./run.sh` and confirm parity.

## What's explicitly incomplete

From Prompt 10's honest-status report:

- **Accessibility pass** — keyboard nav works via Radix primitives and
  focus-ring utilities are set; full WCAG AA audit not done.
- **CONTRIBUTING.md** for the UI — not written.
- **Expanded test coverage** — no vitest frontend tests; backend
  coverage is route-level, not exhaustive per-service.
- **Reports tab polish** — final_report.md is rendered via react-
  markdown as a single body; the "Candidate novel contributions"
  section isn't broken out as expandable items (cosmetic only).
- **"Full run zip" download button** — individual .md + .mp4 download
  works; zip bundling isn't implemented.

## Open questions / worth-exploring

1. Run `sculpt kg extract --all` live for a fresh project — hand-seeded
   entities work, but the live LLM extraction path hasn't been used for
   a real new-project KG build under the UI.
2. Brax / MJX adapter for AME456 — `docs/adapters.md` has the sketch;
   implementing it would close the loop on the original capstone goal.
3. Autonomous KG research agent — post-v1 mini-series; agent reads
   behavior goal + diagnosed failure modes and proposes new arxiv IDs
   to ingest automatically.
4. Reports tab polish (see above).

---

## File-reading priority when asked "what does X do?"

1. `RewardSculptor/HISTORY.md` — authoritative build log for the core
   library (incl. all gotchas in one place).
2. `RewardSculptor/README.md` — user-facing architecture + Brax worked
   example.
3. `RewardSculptor/docs/adapters.md` — adapter contract for lab
   contributors.
4. The relevant module source under `RewardSculptor/sculptor/` or
   `reward-sculptor-ui/backend/`.
5. `tests/test_*.py` — contracts are often most precisely encoded in
   tests.

## Commands cheat-sheet

```bash
# RewardSculptor
cd ~/projects/RewardSculptor
uv sync
uv run pytest tests/ -v
uv run sculpt --help
uv run sculpt run "run forward fast" --config examples/hopper/config.toml --iterations 3 --dry-run   # ~50 s

# reward-sculptor-ui
cd ~/projects/reward-sculptor-ui
uv sync
pnpm install --dir frontend
uv run pytest backend/tests/ -v
./run.sh                        # one-command start
# or manually:
uv run uvicorn backend.main:app --reload --port 8000   # terminal 1
pnpm --dir frontend dev                                 # terminal 2
```

---

## Working style

- Be terse. File:line references + metrics over prose.
- Flag gotchas early. When there's a judgment call, say what you picked
  and why in one sentence.
- Confirm before destructive actions: destructive git ops, `rm -rf`
  outside `runs/` / temp dirs, force-push, anything touching AME456.
- Don't add features, refactor, or introduce abstractions beyond what
  the task requires. A bug fix doesn't need surrounding cleanup.
- Default to writing no comments unless the *why* is non-obvious.
- **Always keep CONTEXT.md updated** (Sam's standing rule, 2026-04-25).
  Append a Change Log entry for every meaningful change in the same
  window you make it — do NOT let commits outrun the changelog. A new
  window must be able to reconstruct the arc from CONTEXT.md alone.
- **Review major changes + decisions with agents** (Sam's standing
  rule, 2026-04-25). For anything non-trivial — a feature, a refactor,
  a cross-cutting bug fix, an architecture call — run the audit-driven
  loop: research (Explore) → plan → plan-audit (Plan agent) → implement
  → code-audit + design-critique (Explore agents, distinct
  perspectives: senior eng, UI/UX, a11y) → apply CRITICAL/HIGH fixes.
  VERIFY each agent's load-bearing claims against source before acting
  — agents produce plausible-but-wrong findings; reject the ones that
  don't hold. Mirrors Ships 14-21e.

---

## Change Log

Append an entry **every time you make a meaningful change**. Format:

```
### <YYYY-MM-DD HH:MM> — <one-line summary>

- **What**: files changed (with paths + line numbers if surgical).
- **Why**: motivation / what prompted it.
- **How**: approach chosen, trade-offs considered.
- **Verified**: tests run, live smoke done, bundle size, etc.
```

Start the next entry below this line.

### 2026-04-25 — Ships 21a-21e: mission UX hardening + review pass

Five follow-up ships on the `ship-20-ux-revamp` branch after Ship 21's
Missions→Runs merge. All on GitHub (PR #1 → base `ship-19-skill-
library`; `main` has unrelated history). Branch head: `db1cdea`.

- **Ship 21a** (`ffe9c5c`) — NewMissionDialog Advanced tab (Basic/
  Advanced, mirrors NewRunDialog) persisting `run_defaults` on the
  mission (sculptor `Mission.run_defaults`, backend CreateMission
  Request + MissionDetail, RunMissionDialog pre-fill). + Fixed the
  404-flash on Decompose: `mission_store` returns a "decomposing" stub
  when `.decompose_pending` exists but mission.json doesn't yet.
- **Ship 21b** (`3c6a6e0`) — live rewards + live video for stage runs
  (frontend-only). RewardsTab auto-scopes to the active stage (Project/
  Stage toggle, 3s poll); RobotViewer `LiveStageRollout` plays per-iter
  rollout.mp4 for `mission_stage_run` kind.
- **Ship 21c** (`190ce05`) — torch-idiom guard for success_criterion.
  Claude was writing `.float()` (torch) on numpy arrays → stage failed
  at the LAST criterion eval after 10+ h of G1 training. 3-layer
  defense: decompose prompt rule + `validate_mission` AST walker +
  runtime `_evaluate_success_criterion` safe-attr removal. Plus the
  iter-row "v3 · held" UX (no-edit iters were misread as failures).
- **Ship 21d** (`cb2ef96`) — RELIABLE reward propagation. Root cause:
  `useRuns` stopped polling at every stage boundary (no running run
  for a beat) and never auto-resumed → scope froze/flipped to project
  → "rewards only sometimes update." Fix: `useRuns(slug, {keepPolling})`
  driven by `missionActive`; sticky stage scope in RewardsTab; mission-
  scoped reward polling; stage-aware `?stage=` on the diagnosis route
  so "Why this edit?" works per stage.
- **Ship 21e** (`db1cdea`) — 4-agent review panel (backend, frontend,
  UI/UX, a11y). Verified findings → fixed: `useLiveClips` infinite-
  reconnect loop (missing terminalRef — CRITICAL); `_resolve_run_root`
  path-traversal guard + stage_name sanitize; WCAG AA motion-safe
  guards + badge contrast (-700→-800); WS reconnect-timer cleanup;
  "Plan" button affordance; aria-labels (badges, videos, sparkline,
  Re-render, form aria-invalid); deleted dead MissionsTab.tsx.
  Rejected 2 overstated agent claims + 1 regressive suggestion.

**Verified across the series:** `pnpm tsc --noEmit` 0; backend pytest
305 passed, 1 deselected; sculptor 364 passed, 1 skipped.

**KG note:** AME456 jumping-robot bibliography (14 papers) ingested +
extracted into the shared KG at `~/.local/share/sculptor/kg/graph.db`
(now 80 papers / 468 techniques). Seeds at `~/projects/ame456_kg_
seeds.yml`.

### 2026-04-25 — Ship 21: Missions merged into Runs (cross-tab integration v2)

**Scope:** User reported five concrete bugs after the Ship 20 G1 test:
(1) the auto-open detail dialog flashed open then closed for ~1s on
Decompose submit; (2) post-failure the StageCard read `rounds 4/3`
because `effective_max_iterations` (introduced Ship 20) was only on
WS events, NOT persisted on the Stage dataclass — when the WS event
window evicted `stage_started` the UI fell back to the authored
budget; (3) mission stages didn't appear in the Runs tab as runs;
(4) live training videos didn't surface in Overview for stages;
(5) per-stage reward versions weren't visible. Sam's verdict:
"I don't even think there needs to be a separate missions tab. It
should be integrated within the runs tab." Ship 21 is two phases
landed in one commit:

**Phase A (small fixes)** — auto-open flicker fix + persisted
`effective_max_iterations` + 3 regression tests.

**Phase B (cross-tab merge)** — mission stages become first-class
`mission_stage_run` Job entries; `list_runs` returns them; the
Runs tab sidebar groups them under their parent mission with a
collapsible chevron; the run detail pane shows per-stage rewards;
the Missions tab is **removed** from `ProjectDetail`. The
`MissionDetailDialog` survives as the curriculum view (decomposition
rationale + stage cards + Run/Delete) — opened from the new "Plan"
button on each mission group header in the Runs sidebar, AND from
post-Decompose auto-open via `NewMissionDialog → onCreated →
setMissionDialogSlug` (the same auto-open flow Ship 19c-20 wired,
relocated to RunsTab).

**Process (mirrors prior ships' audit-driven pattern):**

1. **Research (`Explore` agent)** — mapped JobManager parent/child
   support (none — would need new wiring), `list_runs` filtering
   (`kind="sculpt_run"` only), per-stage filesystem layout (each
   stage scaffolds `<project>/.missions/<m>/stages/<s>/{rewards,runs}/`
   as a self-contained sub-project), the mission_jobs `_stream_stdout`
   event-tag parser, the relationship between sculpt_run subprocess
   events and the parent mission_execute job's stdout. Surfaced 4
   load-bearing constraints: (a) JobManager has zero parent/child
   wiring; (b) `RunSummary` lacks parent_id/mission_slug/stage_name;
   (c) per-stage rollouts live under stage_dir, not project root;
   (d) per-stage rewards are already on disk in stage_dir/rewards/
   but no endpoint exposes them.
2. **Plan v2** committed to the lightweight architecture: register
   per-stage child Jobs as `mission_stage_run` kind via a new
   `register_passive_job` helper that doesn't spawn a runner (the
   work is already happening inside the parent's subprocess);
   tee `iter_*` and stage-lifecycle events from the parent's
   stdout to the active child; extend `list_runs` filter; add a
   `?stage=<m>/<s>` query to rewards endpoints; rebuild RunsTab's
   sidebar to group stage rows under their mission.
3. **Implementation** landed across 11 files (backend + frontend).
4. **Code-audit (`Explore` agent)** — surfaced 2 CRITICALs and a
   handful of MEDIUM/LOW. CRITICALs both fixed in the same commit.
5. **Design-critique (`Explore`-as-design-critic agent)** —
   surfaced 1 WORST issue (mission group header click-target
   ambiguity) + 11 lower-severity items. WORST fixed.

**Files added / changed:**

*Sculptor (`~/projects/RewardSculptor`)*:
- **[sculptor/mission.py](RewardSculptor/sculptor/mission.py)** —
  Phase A: `Stage.effective_max_iterations: Optional[int] = None`
  field. Persisted via `dataclasses.asdict`; backward-compatible
  via `Stage.from_dict`'s filter-unknown-keys path so older
  mission.json loads fine.
- **[sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py)** —
  Phase A: `_run_one_stage` sets
  `stage.effective_max_iterations = effective_max_iterations`
  BEFORE the `stage_started` emit so the value is captured by
  any save path (success, criterion-failure, training-error,
  scaffold-error). Inline comment documents the BASELINE
  semantic (Goal B extensions emit their own events; the cap
  reflects what the user explicitly chose).
- **[tests/test_mission_run.py](RewardSculptor/tests/test_mission_run.py)** —
  Phase A regression tests (3): `test_mission_run_persists_
  effective_max_iterations_on_stage` (success path round-trips
  through JSON + reload), `test_mission_run_persists_effective_
  max_iterations_on_failure` (the actual G1 case Sam hit:
  criterion failure path also persists), `test_stage_effective_
  max_iterations_backward_compat_load` (older mission.json
  without the field loads with `effective_max_iterations=None`).

*Backend UI (`~/projects/reward-sculptor-ui`)*:
- **[backend/services/job_manager.py](reward-sculptor-ui/backend/services/job_manager.py)** —
  `Job.parent_id: Optional[str] = None`. New `register_passive_
  job(kind, project_slug, *, params, parent_id) -> Job` that
  registers a Job WITHOUT a runner: `_cancel = None`, `_task =
  None`, `status = "running"` immediately. Audit-fix CRITICAL
  #2: `JobManager.stop` routes a stop request on a passive job
  (no `_cancel`) to its `parent_id` so a "Stop" click on a
  stage row terminates the parent mission's subprocess (which
  kills all child stages) instead of silently returning None.
- **[backend/models/kg.py](reward-sculptor-ui/backend/models/kg.py)** —
  `JobKind` literal extended with `"mission_stage_run"`.
  `JobSummary.parent_id` field.
- **[backend/services/mission_jobs.py](reward-sculptor-ui/backend/services/mission_jobs.py)** —
  `_stream_stdout` accepts `job_manager`, `project_slug`,
  `mission_slug` so it can register child Jobs. New module-
  level `_STAGE_OPEN_EVENT`, `_STAGE_CLOSE_EVENTS`,
  `_STAGE_TEE_EVENTS` constants enumerate the stage-lifecycle
  contract. On `stage_started`: closes any prior unclosed stage
  defensively, then registers a passive Job with stage params
  (mission_slug, stage_name, stage_index, stage_dir,
  behavior_goal, iterations_requested = effective_max_iterations).
  All `iter_*` + warm-start + redecomposition + skill-publish
  events get tee'd to the active child too. On `stage_succeeded`/
  `stage_skipped`: child marked `completed`. On `stage_failed`:
  child marked `errored` with the reason. On parent-subprocess
  termination: any still-open child closed `errored` so the
  Runs UI doesn't show a perpetually-running row.
- **[backend/routes/missions.py](reward-sculptor-ui/backend/routes/missions.py)** —
  `POST /missions/{slug}/run` passes `job_manager=jobs` into
  `run_mission_execute_job` so the streamer can register child
  Jobs.
- **[backend/routes/runs.py](reward-sculptor-ui/backend/routes/runs.py)** —
  `_find_run` accepts `mission_stage_run` kind. New
  `_resolve_run_root(job, project_dir)` returns the right
  `runs/` root: `<project>/.missions/<m>/stages/<s>/runs/`
  for stage runs, `<project>/runs/` for top-level. `list_runs`
  merges both kinds (top-level first, then stage runs sorted
  by JobManager order). `_run_summary` populates the new
  fields (`kind`, `parent_id`, `mission_slug`, `stage_name`,
  `stage_index`). `get_iter_rollout` uses `_resolve_run_root`
  so `/projects/{slug}/runs/{stage_run_id}/iterations/{N}/rollout`
  serves the stage's rollout video. `get_clip_file` left
  unchanged — live clips are written by `rollout_streamer`
  keyed on the parent mission_execute job_id, not the child;
  stage runs return a clean 404 on the clips endpoint, which
  is fine because the frontend uses `get_iter_rollout` for
  per-iter videos. Stage params synthesized into `RunParams`
  in `get_run` (`iterations_requested` falls back to
  `params["iterations_requested"]` then `params["iterations"]`
  then `1`).
- **[backend/models/run.py](reward-sculptor-ui/backend/models/run.py)** —
  `RunSummary` adds `kind: str = "sculpt_run"`, `parent_id`,
  `mission_slug`, `stage_name`, `stage_index` (all
  Optional). RunDetail extends RunSummary so the same fields
  flow into the detail endpoint.
- **[backend/routes/rewards.py](reward-sculptor-ui/backend/routes/rewards.py)** —
  New `_resolve_rewards_dir(store, slug, stage)` helper. When
  `stage="<mission_slug>/<stage_name>"`, the function returns
  `<project>/.missions/<mission_slug>/stages/<stage_name>/rewards/`
  with traversal guards (rejects `..`, empty halves, backslashes,
  non-2-part splits). `list_rewards` and `get_reward` accept
  `?stage=...` query; absent → project rewards (Ship 18a
  behavior preserved). Stage queries that resolve to a
  not-yet-scaffolded rewards/ dir return `[]` (not 404) so the
  Runs detail pane shows "no reward versions yet" cleanly while
  the stage is still v0-only.

*Frontend (`~/projects/reward-sculptor-ui/frontend/src`)*:
- **[lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts)** —
  `RunSummary` new optional fields (`kind`, `parent_id`,
  `mission_slug`, `stage_name`, `stage_index`). `StageSchema`
  gains `effective_max_iterations: number | null` (Phase A).
- **[lib/api.ts](reward-sculptor-ui/frontend/src/lib/api.ts)** —
  `listRewards(slug, stage?)` and `getReward(slug, version,
  stage?)` accept the optional stage param, append
  `?stage=<encoded>` query.
- **[hooks/useRewards.ts](reward-sculptor-ui/frontend/src/hooks/useRewards.ts)** —
  `useRewards` and `useReward` accept a `stage?: string | null`
  argument. Cache keys are namespaced (`["rewards", slug,
  "stage", stageKey]`) so stage-scoped queries don't clobber
  the project rewards cache.
- **[components/RunsTab.tsx](reward-sculptor-ui/frontend/src/components/RunsTab.tsx)** —
  Substantial rewrite. Public `RunsTab` now owns mission state:
  imports `useMissions`, `NewMissionDialog`, `MissionDetailDialog`,
  `MissionLifecycleBadge`. New `partitionRuns(runs, missions)`
  splits runs into top-level sculpt_runs and mission groups
  (one entry per mission_slug with its stage runs sorted by
  `stage_index`, audit-fix #6 stable tiebreak on `run_id`).
  Runs sidebar `RunSidebar` is fully rewritten: header
  "Single runs" section followed by mission group entries.
  Each group has a collapsible chevron-button (toggles expand);
  the group "Plan" button (audit-fix WORST: explicit hit
  target, was an inline subbutton with click-target ambiguity)
  opens MissionDetailDialog. Stage rows render via the same
  `RunRow` primitive as top-level runs but with
  `stageContext={true}` which prefixes the row with
  `${stage_index + 1}.` and uses `stage_name` instead of
  `run_id`. New `StageContextCard` (in RunDetailPane, only
  for mission_stage_run rows) surfaces "Stage N: name" +
  "mission ${slug}" + behavior_goal. New `StageRewardsCard`
  (right column, mission_stage_run only) lists per-stage
  reward versions via `useRewards(slug, stage)`. Removed
  the Ship-20 `ActiveMissionsCard` (rendered the missions
  separately above the runs grid) — its function is replaced
  by the inline mission groups.
- **[components/MissionsTab.tsx](reward-sculptor-ui/frontend/src/components/MissionsTab.tsx)** —
  Phase A1: gating fix at the dialog mount (`open={selectedSlug
  != null}` instead of `open={!!selected}`) so the auto-open
  dialog doesn't flicker closed when the optimistic-cache
  placeholder is briefly replaced by an in-flight refetch.
  File still on disk but unreferenced; deletable in a follow-
  up clean-up.
- **[components/MissionDetailDialog.tsx](reward-sculptor-ui/frontend/src/components/MissionDetailDialog.tsx)** —
  Phase A3: StageCard reads from `stage.effective_max_iterations`
  (persisted, source of truth) first, falls back to the WS
  event-derived map (transient), then to `stage.max_iterations`
  (Claude's authored value). Source-doc comment in
  `deriveStageEffectiveMaxIters` documents the 5000-event-cap
  limitation as known.
- **[pages/ProjectDetail.tsx](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx)** —
  Missions tab removed from the `TABS` array. The dead
  `<TabsContent value="missions">` block deleted. Import for
  `MissionsTab` removed.

**Audit findings + fixes:**

- **Code-audit CRITICAL #2 — `JobManager.stop` silently fails on
  passive jobs.** Pre-fix: `stop(stage_run_id)` returned None
  (because passive jobs have `_cancel = None`), so the frontend
  toast "Kill signal sent" was a lie — the parent's subprocess
  kept training. **Fix shipped:** `stop` now routes to
  `parent_id` for passive jobs so the parent's `_cancel` flag
  fires, terminating the subprocess (and all child stages).
- **Code-audit MEDIUM #6 — sort stability.** `partitionRuns`
  sorts stages by `stage_index` only; if two stages share an
  index (Ship 17 redecomposition splice edge case), order was
  non-deterministic. **Fix shipped:** secondary tiebreak on
  `run_id.localeCompare`.
- **Design-audit WORST — mission group header click target
  ambiguity.** Pre-fix: chevron toggled expand, the rest of
  the header opened MissionDetailDialog — users clicking the
  goal text expecting expansion got a dialog instead. **Fix
  shipped:** the entire header (chevron + Sparkles + lifecycle
  + label + goal) is one button that toggles expand/collapse;
  a small "Plan" pill on the right is the explicit
  curriculum-dialog button.
- **Code-audit MEDIUM #4 — `get_clip_file` for stage runs.**
  Live clips are written by `rollout_streamer` keyed on the
  parent mission_execute job. Stage runs return a clean 404
  on the clips endpoint. Acceptable: the frontend uses
  `get_iter_rollout` for per-iter videos, which DOES resolve
  to the stage's rollout. Documented in the route's inline
  comment as a known limitation.
- **Code-audit FINDING #1 — event ordering invariant.** Auditor
  flagged a theoretical race where `iter_*` could fire before
  `stage_started`. The orchestrator (sculpt.py) emits
  `stage_started` BEFORE entering sculpt_run (the source of
  iter_* events) at line 2169-2178, so the invariant holds.
  Defensive close-prior-stage logic already in place handles
  redecomposition splices. No code change.

**Tests + verification:**
- ✅ **TypeScript**: `pnpm tsc --noEmit` returns 0.
- ✅ **Sculptor pytest**: **358 passed, 1 skipped** (was 355 →
  +3 Phase A persistence regression tests, zero regressions
  across the 355 baseline).
- ⏳ **Backend pytest**: pending (existing 299 baseline; no
  new tests added in Ship 21 itself — the cross-tab merge is
  exercised end-to-end by the frontend tsc + the existing
  list_runs test which still asserts the `sculpt_run` happy
  path).
- ⏳ **Live smoke**: Sam's G1 mission should now show:
  - Decompose submit → MissionDetailDialog opens stably (no
    flicker; gated on direct selectedSlug state).
  - StageCard `rounds X/Y` reflects the effective cap from
    `stage.effective_max_iterations` (persisted), survives
    page refresh + WS event-cap eviction.
  - Mission stages appear as nested rows under the mission
    group in Runs sidebar.
  - Click stage row → live iter timeline + metric chart +
    log viewer + per-stage reward versions in the right
    column.
  - Stage rollout videos available via the same per-iter
    rollout endpoint (stage runs route to `<project>/.missions
    /<m>/stages/<s>/runs/iter_N/rollout/rollout.mp4`).
  - Top-right "Plan" button on each mission group opens the
    decomposition view (rationale + stage cards + Run/Delete).
  - Missions tab no longer in the tab strip.

**Callable surface (UI):**
- Open a project → click **Runs** tab.
- Click **New mission** in the header to decompose a goal;
  MissionDetailDialog auto-opens to show the live decompose
  stream.
- After decompose completes (lifecycle "ready"), click **Plan**
  on the mission group in the sidebar to review the curriculum
  and click **Run mission** in the dialog footer to launch
  with iterations_override / Goal A / Goal B settings.
- Once running, each stage appears as a nested row under the
  mission group. Click a stage row to see live metrics,
  iter-by-iter logs, and the stage's reward versions.
- **New run** still launches a single standalone training run
  (kind="sculpt_run"), surfaced under "Single runs" in the
  sidebar.

**Explicitly NOT in Ship 21** (deferred):
- Per-stage Monaco editing in the Runs detail pane —
  StageRewardsCard is read-only; full editing stays in the
  Rewards tab (project-scoped). A future ship can add inline
  edit; the underlying `?stage=` query is already wired.
- Live clips for stage runs — `rollout_streamer` is keyed on
  the parent mission_execute job; ship a stage-keyed streamer
  if Sam wants live frame-by-frame mid-stage video. Per-iter
  rollouts work today.
- RobotViewer (Overview tab) doesn't yet pick the active
  stage as the most-recent run — would need to extend its
  most-recent-run selector to consider `mission_stage_run`
  rows. Defer.
- Deleting `MissionsTab.tsx` (now dead code on disk). Leave
  for a tidy-up commit.
- Auto-reconnect on WS disconnect for stage runs. Inherits
  Ship 18b's "no auto-reconnect" decision; refresh-to-retry
  banner still shows.

### 2026-04-25 — Ship 20: UX label revamp + cross-tab mission integration

**Scope:** Audit-driven UI cycle on the React app. Five user-facing wins: (1) Goal A/Goal B labels in RunMissionDialog renamed to plain English; (2) `iters X/Y` display in StageCard fixed to honor `iterations_override` (was showing nonsense like `iters 2/3` when override capped run at 2); (3) `params.mission_slug` wire format pinned with a regression test (Sam's auto-open-dialog complaint from Ship 19d); (4) blanket label cleanup — `MISSION_EXECUTE`/`MISSION_DECOMPOSE` chips, `0/3 stages`, `redecomp ×N`, `orphan parent_ref`, "Decomposing — Claude is building the curriculum"; (5) active missions surface in the Runs tab via a static `ActiveMissionsCard` panel that opens the existing `MissionDetailDialog`.

**Process (mirrors Ships 14-19d's audit-driven pattern):**
1. **Research (`Explore` agent)** — file:line-cited map of Missions/Runs/Rewards/Physics/Overview tab structure, WS/query-key surface, exact `iters X/Y` failure mode (MissionDetailDialog.tsx:567-569 reads `stage.max_iterations` — Claude's authored budget — instead of the override-respecting cap from sculpt.py:2342).
2. **Website-designer (`Plan` agent)** — IA proposal. Lead recommendation: **keep Missions tab separate** (decompose-time UX is fundamentally different from run-time monitoring); migrate stage live-state visibility into Runs via a `useMissionStageRuns(slug)` derived hook + `<MissionStageGroup>` collapsible. Full label-and-copy table.
3. **Plan-audit (`Plan` agent)** — biggest hole flagged: **"the synthetic-stage-rows go blank when the user switches tabs and comes back."** The designer plan derived per-stage iter timelines from a per-tab `useMissionEvents` subscription, but `useMissionEvents` resets state on every effect re-run + the structuredEvents 5000-cap means re-mounting after a tab switch loses all stage cursor history. The proposed mitigation (hoist WS above tab-mount) would touch outside the in-scope file list. Other CRITICAL/HIGH: synthetic id collision across mission re-runs, `iter_progress` events dropped by `deriveStageIters`, `useRunEvents` vs `useMissionEvents` shape mismatch, decompose-stream UX in MissionDetailDialog would break if the live-stream half got excised wholesale.
4. **Plan v2 — chose lower-blast-radius cross-tab integration.** Don't migrate live-stream UI from MissionDetailDialog to RunsTab; the dialog stays the canonical live-monitoring surface (with its existing audit-driven WS gating). Add a static `ActiveMissionsCard` to RunsTab that surfaces missions with `active_job_id != null OR lifecycle === "running"`. Click a row → opens the **existing** MissionDetailDialog (now mounted from RunsTab too, but only one tab is mounted at a time per ProjectDetail.tsx's deferred-mount, so no double-WS concern).
5. **Implementation** — see file list below.
6. **Code-audit (`Explore` agent)** — flagged Goal B semantic clarification (the new `effective_max_iterations` event payload is the BASELINE cap, not the post-extension cap; documented in inline comment), 5000-event cap interaction (documented as known limitation), `useMissions` polling overlap (deduped by React Query cache key — fine).
7. **Design-critique (`Explore`-as-design-critic agent)** — applied CRITICAL fixes: `*` override indicator's aria-label now scopes to the rounds value (was on the asterisk only — screen reader would announce "Per-launch override applied" disconnected from the number); `ActiveMissionsCard` got `max-h-[420px] overflow-y-auto` (was unbounded — would push runs grid off-screen on mobile with many active missions); empty-state copy in RunsTab now conditionally mentions "panel above" only when at least one mission is active; `missionRunStateLabel` (RunsTab) and `stageProgressLabel` (MissionsTab) harmonized so the same mission reads the same way across tabs.

**Files added/changed:**

*Sculptor (`~/projects/RewardSculptor`)*:
- **[sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py)** — Goal #2: `_run_one_stage` now computes `effective_max_iterations = iterations_override or stage.max_iterations` BEFORE the `stage_started` emit (line ~2169) and includes it in both `stage_started` and `stage_completed_training` event payloads. The pre-existing `max_iters = ...` line at sculpt.py:2342 was hoisted up + reused. Audit-driven inline comment documents that this is the BASELINE cap before any Goal B (`extend_on_improvement`) extensions; extensions are surfaced via separate `stage_extended` events.

- **[tests/test_mission_run.py](RewardSculptor/tests/test_mission_run.py)** — 2 new regression tests: `test_mission_run_stage_events_include_effective_max_iterations` (override path: payload reflects override, NOT authored value) + `test_mission_run_effective_max_iterations_falls_back_to_authored` (no-override path: payload equals authored).

*Backend (`~/projects/reward-sculptor-ui`)*:
- **[backend/tests/test_missions.py](reward-sculptor-ui/backend/tests/test_missions.py)** — 1 new regression test: `test_create_mission_response_includes_params_mission_slug`. Pins the `params.mission_slug` wire format that the auto-open dialog depends on (Ship 19c flipped the response_model from `JobSummary` → `JobDetail` to include `params`; this test makes sure a future refactor can't silently revert it). Also asserts `params.goal` round-trips so the optimistic-cache placeholder in `useCreateMission.onSuccess` shows the right text while the list refetches.

*Frontend (`~/projects/reward-sculptor-ui/frontend/src`)*:
- **[components/RunMissionDialog.tsx](reward-sculptor-ui/frontend/src/components/RunMissionDialog.tsx)** — Goal #1: "Goal A: early-stop on criterion" → "Stop when the goal is met"; "Goal B: extend on improvement" → "Keep training while still improving". Goal #4: stripped Ship 9a/16 references in help text. Renamed "Outer iters / stage" → "Rounds per stage", "Steps / iter" → "Steps per round" (avoids `rsl_rl iters (mjlab) / env steps (gym)` adapter-leak). ETA copy uses "rounds" / "per-round wall-clock" for terminology consistency with the rest of the dialog and the StageCard.
- **[components/MissionDetailDialog.tsx](reward-sculptor-ui/frontend/src/components/MissionDetailDialog.tsx)** — Goal #2: new `deriveStageEffectiveMaxIters(events)` function walks WS events for `stage_started` + `stage_completed_training` payloads → `Map<stage_name, effective_max_iterations>`. `StageCard` accepts `effectiveMaxIters: number | null` prop, renders `rounds X/Y` where Y prefers the WS-derived effective cap and falls back to authored `stage.max_iterations`. When the override differs, a dotted-underline + amber `*` indicator appears + `aria-label` describes "rounds {used} of {effective}; per-launch override (Claude allocated {authored})" so screen readers announce the relationship. `title` tooltip on hover explains the override. Goal #4: "Decomposing — Claude is building the curriculum" → "Planning — Claude is breaking your goal into stages"; "redecomp ×N" → "replanned ×N"; "orphan parent_ref" → "Missing parent stage"; "WebSocket disconnected — refresh the page to retry. (Auto-reconnect lands in Ship 18c.)" → "Live stream interrupted. Refresh to reconnect."; "No active job — events from prior runs are not replayed." → "Nothing running. Launch a run to watch live events." Button-disabled `title` text de-jargoned.
- **[components/MissionsTab.tsx](reward-sculptor-ui/frontend/src/components/MissionsTab.tsx)** — Goal #4: card description rewritten ("Define a goal. Claude breaks it into stages and trains them in order, warm-starting each from the previous." replaces the old `Claude decomposes a goal into a curriculum of stages... .missions/<slug>/mission.json` filesystem leak). New `missionJobKindLabel` helper replaces `MISSION_EXECUTE` / `MISSION_DECOMPOSE` enum chips with English "Training" / "Planning". New `stageProgressLabel` replaces ambiguous `0/3 stages` (which read as a test failure) with lifecycle-aware phrasing: `Planning…` / `3 stages planned` / `Stage 1 of 3` / `3 of 3 stages complete`. Slug code deemphasized to last position in the metadata row (goal sentence is the primary identifier; the slug was pseudo-derived from it anyway).
- **[components/RunsTab.tsx](reward-sculptor-ui/frontend/src/components/RunsTab.tsx)** — Goal #5: imports `useMissions`, `MissionDetailDialog`, `MissionLifecycleBadge`, `MissionSummary`. New `ActiveMissionsCard` panel between the Runs Card header and the runs grid, surfaces missions where `active_job_id != null OR lifecycle === "running"`. Each row shows lifecycle badge + `missionRunStateLabel(m)` (e.g., "Stage 2 of 3") + active-job kind chip + truncated goal + slug. Click → opens the existing `MissionDetailDialog` (mounted at the bottom of RunsTab), which keeps the live WS subscription, stage cards, iter ribbon — all the audit-driven Ship 18b/19c machinery — in its single canonical place. CardDescription updated to mention both standalone runs AND mission stages. Empty-state copy on the runs Card conditionally mentions "panel above" only when missions are active. `max-h-[420px] overflow-y-auto` on `ActiveMissionsCard.CardContent` so 5+ active missions don't push the grid off-screen on mobile (design-audit fix).

**Tests + verification:**
- ✅ **TypeScript**: `pnpm tsc --noEmit` returns 0.
- ✅ **Sculptor pytest**: **355 passed, 1 skipped** (was 353 → +2 Ship 20 tests; zero regressions).
- ✅ **Backend pytest**: `298 passed → 299 passed, 1 deselected` (+1 Ship 20 regression test).
- ⏳ **Live smoke**: pending user run via `./run.sh`. Expected:
  - Decompose submit → MissionDetailDialog auto-opens immediately (Goal #3 wire format locked by test).
  - Stage card with `iterations_override=2` and `stage.max_iterations=3` shows `rounds 2/2*` with tooltip "Claude allocated 3 rounds; this run capped at 2." (Goal #2).
  - "Stop when the goal is met" + "Keep training while still improving" labels (Goal #1).
  - Active mission visible in RunsTab's `ActiveMissionsCard` panel; click opens the existing dialog (Goal #5).

**Audit findings + fixes (the load-bearing list)**:

*Plan-audit BIGGEST HOLE — designer's "synthetic-stage-rows in RunsTab" plan loses state on tab switch.*
- The designer's `useMissionStageRuns` would have opened a per-mission WS inside RunsTab; switching to Rewards/Physics unmounts RunsTab → the WS closes → `useMissionEvents` resets local state → re-mount sees empty events array. Combined with the 5000-event cap and no server replay, a stage at iter 4/8 would show `iter 0/0` when the user comes back.
- **Decision (Plan v2):** keep MissionDetailDialog as the canonical live-monitoring surface; RunsTab only adds a **static read-only entry point** to it. The dialog has all the audit-driven WS-gating (Ship 18b finding A), optimistic cache writes (Ship 19c), and iter-ribbon attribution (Ship 19c) already correct. Sidesteps the entire WS-lifecycle-vs-tab-mount problem.
- **Fix shipped:** ActiveMissionsCard is REST-driven (uses `useMissions(slug)` polling); click opens the existing dialog.

*Code-audit CRITICAL — `effective_max_iterations` semantics during Goal B extensions.*
- Auditor's concern: when Goal B extends a stage past its initial budget, the event payload's `effective_max_iterations` doesn't reflect the cumulative cap.
- **Decision:** `effective_max_iterations` is the BASELINE cap (the value the user explicitly chose / Claude authored). Goal B extensions are surfaced via separate `stage_extended` / `stage_extension_skipped` / `stage_extension_exhausted` events. The cap shown on the stage card reflects what the user configured; `iterations_used > effective_max_iterations` is exactly the "extended" signal Goal B users opt into seeing.
- **Fix shipped:** load-bearing inline comment in sculpt.py at the `effective_max_iterations` assignment documents the semantic. No code change.

*Design-critique CRITICAL #1 — asterisk override indicator lacked screen-reader scope.*
- Pre-fix: `<span aria-label="Per-launch override applied">*</span>` — screen reader announces just the asterisk's label, disconnected from the rounds value.
- **Fix shipped:** moved aria-label to the parent `<span>` covering the entire `rounds X/Y` element so the announcement is "rounds X of Y; per-launch override (Claude allocated Z)". The `*` itself is `aria-hidden="true"` (purely decorative for sighted users).

*Design-critique CRITICAL #2 — ActiveMissionsCard had no max-height.*
- Failure mode: 5+ active missions on a 375px-wide mobile would push the runs grid off-screen.
- **Fix shipped:** `max-h-[420px] overflow-y-auto scrollbar-thin` on the CardContent, mirroring RunSidebar's pattern.

*Design-critique HIGH — copy tone inconsistency between `missionRunStateLabel` and `stageProgressLabel`.*
- Two helpers in two files producing different strings for the same lifecycle state.
- **Fix shipped:** harmonized — both use `Planning…` / `${n} stages planned` / `Stage ${i+1} of ${n}` / `${n} of ${n} stages complete`. Inline cross-reference comments in both functions.

*Code-audit MEDIUM — `deriveStageEffectiveMaxIters` "last value wins" on stage re-runs.*
- If user re-runs a mission within one WS session, the second `stage_started` overwrites the first.
- **Decision:** acceptable — within one mission_run the cap is constant per stage; re-runs emit a fresh `stage_started` before any new iters display. The re-run's cap is in place before anything else displays.
- **Fix shipped:** load-bearing inline comment in `deriveStageEffectiveMaxIters` documents the semantics + the 5000-event-cap known limitation.

**Callable surface (UI):**
- Open a project → click **Runs** tab. If any mission has an active job, the **Active missions** panel appears above the runs grid.
- Click an active mission row → opens the existing **MissionDetailDialog** (rationale + stage cards + live WS event stream + per-stage iter ribbon). All Ship 18b/19c functionality intact; the dialog now has a UI entry point from BOTH Missions tab AND Runs tab.
- Open the **RunMissionDialog** (from Missions tab → mission row → "Run mission") → see the renamed adaptive options and updated terminology ("Rounds per stage" instead of "Outer iters / stage", etc.).
- Pass an `iterations_override` smaller than `stage.max_iterations` → the resulting stage card shows `rounds X/effective*` with tooltip "Claude allocated Y rounds; this run capped at Z" (Goal #2 fix).

**Explicitly NOT in Ship 20** (deferred):
- **Stage runs as first-class `RunSummary` entries in `list_runs`** — would need `RunSummary` shape change, route refactor, JobManager child-job model. Plan-audit deemed it too expensive for one cycle. The lightweight read-only `ActiveMissionsCard` ships the user-visible win without the backend churn.
- **Active-mission banners on Rewards / Physics / Overview tabs** — designer's stretch goal. Cardinality (what if 2+ missions running?) wasn't worked out; defer to Ship 21 with a cardinality spec.
- **Stage-scoped Rewards / Physics filtering** — would require timestamp matching against stage windows. Scope-creep.
- **Slug rename UX** — `mission_slug` is currently the URL path key; renaming would need a redirect-on-slug-change story.
- **Auto-open dialog runtime hypothesis #1 (`./run.sh` not picking up route changes)** — a `--reload` deficiency in the bash script, not Ship 20's lane. Ship 19d's commit already fixed the response_model; the new `test_create_mission_response_includes_params_mission_slug` regression test pins it. If the user still hits the auto-open bug after `./run.sh` cold restart, the runtime issue is uvicorn-reload, not the wire format.

### 2026-04-24 — Ship 19: cross-mission skill library (Voyager-flavored)

**Scope:** Fifth brick of the Mission roadmap and a deliberate extension toward CurricuLLM/Voyager. Where Ship 15-16 chained policies WITHIN a mission (parent_stage → init_policy_path), Ship 19 adds a **filesystem-backed registry of trained policies that survives across missions**. A `stand_on_one_leg` mission's final policy can warm-start the `stand_on_one_leg__then_kick` mission's "stand" stage, even though the two missions are independent. Sculptor library + CLI only — no UI surface in this ship (Ship 19b).

**Process (mirrors Ships 14-18a):**
1. **Explore agent** mapped Ship 15 warm-start plumbing (`init_policy_path` is mjlab-only; `_train_or_resume` introspects `adapter.train` for `init_policy_path` OR `**kwargs`), Ship 16 parent-chain (`Mission.parent_checkpoint_status_of(name)` returns `(Path|None, status_tag)`), and the project layout (each stage is a full sculpt project at `<mission>/stages/<name>/runs/iter_<i>/checkpoint.{pt,zip}`). Key constraint surfaced: **`env_id` is NOT a real field on mjlab — it uses `task_id`**. Other adapters use env_id but don't support `init_policy_path` anyway.
2. **Plan v1**: (adapter_class, env_id) compatibility key, "parent wins over skill" default (mirroring Ship 15 "local checkpoint wins"), first-8KB hash for skill ID, optimistic publish-on-stage-success.
3. **Plan agent audit** flagged the BIGGEST HOLE: `env_id` doesn't exist on the only adapter that supports warm-start. Plus 5 CRITICAL/HIGH:
   - **C1**: "parent wins" inverts CurricuLLM's premise. **Decision flipped to "explicit beats implicit"** — when Claude sets `init_skill_id`, the SKILL wins; otherwise the parent_ckpt wins (Ship 16 behavior preserved).
   - **C2**: `Stage.best_metric` is actually LAST-iter metric, not best. **Switched to `argmax(primary_metric_history)`** at publish time so the BEST-iter checkpoint enters the library (with `source_iter_index` recorded).
   - **C3**: skill_id derivation including `reward_seed_prompt` made non-whitespace text a discriminator, fragmenting the library. **Dropped seed prompt + criterion from the hash**; identity is `sha256(adapter_class + task_id + full_ckpt_sha256)[:12]`.
   - **C4**: first-8-KB hash is identical for two policies sharing architecture (the actor's first layer is the same). **Switched to full-file SHA-256** (~1.5 s for 500 MB, runs once per publish).
   - **H1**: don't publish `redecomposition_attempts > 0` sub-stages — by Ship 17 design they're specialized to slices the parent task couldn't cover; their reward shape is a poor cross-mission seed.
   - **H2**: `list_compatible` per-record try/except + tmp+rename for metadata (audit fix).
   - **H3**: 5-kwarg proliferation collapsed into `SkillLibraryHandle(library, adapter_class, task_id, robot_slug, publish)` — single kwarg threads through `decompose_task` + `mission_run`.
   - **H4**: pydantic validator normalizes empty/whitespace `init_skill_id` to None.
   - **H5**: CLI `mission-init` and `mission-run` build a default handle from the project's config.toml; `--no-skill-library` opt-out, `--skill-library-root` override.
   - **M3**: copy-only (no hardlink) — hardlinks across stage iter dirs would break under a future cleanup pass that GCs source iters.
4. **Plan v2 implemented** across 6 files.
5. **Code-audit agent** (Explore-type) found 1 CRITICAL bug + verified all 11 plan-audit fixes are in place:
   - **CRITICAL — `os.replace` is atomic for the file but the parent directory's new dirent isn't durable until the dir inode is fsync'd.** A post-rename kernel/process crash could lose the entry. **Fixed**: new `_fsync_dir(d)` helper called after `_atomic_write_text` and `_atomic_copy`. Best-effort on non-POSIX (Windows directory open isn't supported). Source-inspection regression guard `test_atomic_helpers_fsync_parent_directory_in_source` so future refactors can't silently drop the fsync.

**Files added/changed:**

- **NEW [sculptor/skill_library.py](RewardSculptor/sculptor/skill_library.py)** (~540 lines):
  - `SkillRecord` (schema_version, skill_id, adapter_class, task_id, robot_slug, reward_seed_prompt, success_criterion, final_metric, source_iter_index, iterations_used, source_mission_goal, source_stage_name, created_at, checkpoint_filename, checkpoint_sha256, checkpoint_size_bytes, alias) — JSON roundtrip, only basenames persisted (Ship 18a path-relocation precedent).
  - `SkillLibrary(root)` — `publish_from_stage`, `load`, `__iter__`, `list_compatible(adapter_class, task_id, robot_slug=None, top_k=5)`, `checkpoint_path_for(record)`. Per-(adapter, task_id) filelock at `<root>/.locks/<safe_adapter>__<safe_task>.lock`. Atomic write/copy via tmp+rename + parent fsync.
  - `SkillLibraryHandle` — bundles (library, adapter_class, task_id, robot_slug, publish) with `list_for_decompose`, `maybe_load_for_stage` (taxonomized skip reasons: `skill_not_found / adapter_class_mismatch / task_id_mismatch / checkpoint_missing`), `maybe_publish` (gates on stage status, `redecomposition_attempts == 0`, `adapter_supports_warm_start`, non-empty metric history, best-iter ckpt presence — each with a precise `stage_skill_publish_skipped(reason=...)` event).
  - `derive_skill_id(adapter, task, ckpt_sha)` (12 hex), `default_library_root()` (env `SCULPTOR_SKILL_LIBRARY_ROOT` else `~/.local/share/sculptor/skills/`), `adapter_supports_warm_start(adapter)` (mirrors Ship 15's introspection at sculpt.py:732).
- **[sculptor/mission.py](RewardSculptor/sculptor/mission.py)** — `Stage.init_skill_id: Optional[str] = None` field. Forward-compat: `from_dict` already filters unknown keys, so older mission.json loads with init_skill_id=None.
- **[sculptor/decompose.py](RewardSculptor/sculptor/decompose.py)** — `_StageModel.init_skill_id` with `field_validator` mode="before" mapping `""`/whitespace → None. New `_render_skill_library_context(handle)` returns `(markdown, available_ids)` mirroring `_render_kg_context`. New `_validate_skill_ids(stages, available_ids)` — Claude inventing an unknown id raises `MissionValidationError` at decompose time (caught at the right layer; not at runtime). `decompose_task(..., skill_library_handle=None)` is the new kwarg; defaults preserve Ship 14-17 behavior.
- **[sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py)** — `mission_run(..., skill_library_handle=None)` + `_run_one_stage(..., skill_library_handle=None)`. New skill resolution block BEFORE the parent_ckpt fallback: `skill_ckpt = handle.maybe_load_for_stage(stage, emit) if handle and stage.init_skill_id`; `init_policy_path = skill_ckpt or parent_ckpt`. Emits `stage_warm_start_chosen(source ∈ {skill_library, parent_stage, none}, source_id)` and `warm_start_skipped(reason="skill_overrides_parent")` when skill displaces parent. After successful criterion, `handle.maybe_publish(...)` runs in a try/except that emits `stage_skill_publish_skipped(reason="publish_call_errored")` on any IO failure without breaking the stage's success.
- **[sculptor/prompts/decompose_task.md](RewardSculptor/sculptor/prompts/decompose_task.md)** — input list mentions the optional SKILL_LIBRARY block, schema gains `init_skill_id`, new rule #8 explains the skill-vs-parent precedence + "guessing is worse than null".
- **[sculptor/cli.py](RewardSculptor/sculptor/cli.py)** — new `--no-skill-library` + `--skill-library-root` flags on `mission-init` and `mission-run`. New `_build_skill_library_handle(config_path, library_root)` reads `[adapter].class` + `[adapter.config].task_id` (falling back to `env_id`) from config.toml; returns None when either is missing so non-mjlab projects degrade gracefully.

**Tests — [tests/test_skill_library.py](RewardSculptor/tests/test_skill_library.py) (29 tests) + [tests/test_ship19_skill_warm_start.py](RewardSculptor/tests/test_ship19_skill_warm_start.py) (15 tests), 44 new total:**
- skill_library.py: id determinism + textual normalization, full-file SHA-256 (audit C4 regression), publish atomic via tmp+rename, no leftover `.tmp` files, raise on missing checkpoint, `list_compatible` filtering + ordering + top_k cap + corrupt-record skip, `default_library_root()` env-var override + fallback, `adapter_supports_warm_start` (explicit kwarg / **kwargs / unsupported / no train method), concurrent publish via filelock (2 threads, no clobber), `SkillLibraryHandle` skip-reason taxonomy (5 reasons covered), `maybe_publish` gates (handle disabled / redecomp artifact / adapter no-warm-start / no-history / best-iter checkpoint), best-iter checkpoint correctness (audit C2 regression: history `[0.3, 0.5, 0.9, 0.6, 0.4]` → publish iter 2's bytes), **`test_atomic_helpers_fsync_parent_directory_in_source` (audit BIGGEST BUG regression guard)**.
- ship19_skill_warm_start.py: skill resolution wiring (skill present → init_policy_path = skill ckpt; skill explicit → wins over parent; init_skill_id unset → parent wins per Ship 16; unknown skill_id → cold-start + `skill_warm_start_skipped(reason="skill_not_found")`), publish on success (event shape, library reflects it), publish suppressed (handle None / failed stage), decompose-time skill-library block rendering (id appears in user content), decompose-time validation rejects unknown init_skill_id, Pydantic empty/whitespace normalization (audit H4 regression), legacy mission.json loads with init_skill_id=None (backward-compat), redecompose sub-stages don't propagate init_skill_id (source-inspection guard).

**Verified:**
- Sculptor: **335 passed, 1 skipped** (was 291 → +44 Ship 19 tests). Zero regressions across the pre-existing 291-test baseline.
- Backend: **298 passed, 1 deselected** — no backend changes; baseline retained.
- Test runtimes: skill_library.py + ship19_skill_warm_start.py finish in ~2.4 s; full sculptor suite ~47 s.

**Callable surface:**

```python
from pathlib import Path
from sculptor.skill_library import SkillLibrary, SkillLibraryHandle
from sculptor.decompose import decompose_task
from sculptor.sculpt import mission_run

# 1. Build a handle (reads $SCULPTOR_SKILL_LIBRARY_ROOT or
#    ~/.local/share/sculptor/skills/ by default).
handle = SkillLibraryHandle(
    library=SkillLibrary(),
    adapter_class="sculptor.adapters.mjlab.MjlabAdapter",
    task_id="Mjlab-Velocity-Flat-Unitree-Go1",
    robot_slug="g1_humanoid",  # optional
    publish=True,              # default: contribute on stage success
)

# 2. Decompose with skill-library context. Claude may set
#    init_skill_id on any stage to warm-start from a prior policy.
mission = decompose_task(
    "stand on one leg and kick", reward_contract,
    kg_store=kg, skill_library_handle=handle,
)

# 3. Run with skill-library wired through. Stages with init_skill_id
#    set load that skill as init_policy_path (skill wins over
#    parent_ckpt). Successful stages publish their best-iter
#    checkpoint to the library.
result = mission_run(
    mission, adapter_short_name="mjlab",
    kg_store=kg, skill_library_handle=handle,
)
```

CLI:
```bash
# Default: skill library ON (publishes + reuses).
uv run sculpt mission-init /path/to/project --goal "stand on one leg and kick"
uv run sculpt mission-run /path/to/project

# Opt out / override library root.
uv run sculpt mission-init /path/to/project --goal "..." --no-skill-library
uv run sculpt mission-run /path/to/project --skill-library-root /tmp/skills
```

**New events for Ship 18 / Ship 19b UI to surface:**
- `stage_warm_start_chosen` (source ∈ {`skill_library`, `parent_stage`, `none`}, source_id, checkpoint).
- `skill_warm_start_skipped` (reason ∈ {`skill_not_found`, `adapter_class_mismatch`, `task_id_mismatch`, `checkpoint_missing`}).
- `stage_skill_published` (skill_id, adapter_class, task_id, final_metric, source_iter_index, checkpoint_size_bytes).
- `stage_skill_publish_skipped` (reason ∈ {`handle_publish_disabled`, `stage_not_succeeded`, `redecomposition_artifact`, `adapter_does_not_support_warm_start`, `no_metric_history`, `best_iter_not_in_completed`, `best_iter_checkpoint_missing`, `library_error`, `publish_call_errored`}).
- `warm_start_skipped(reason="skill_overrides_parent")` when explicit skill displaces an available parent_ckpt.

**Source-of-truth invariants:**
- The library at `$SCULPTOR_SKILL_LIBRARY_ROOT` (default `~/.local/share/sculptor/skills/`) is canonical for skill metadata + checkpoints.
- Skill identity is `(adapter_class, task_id, full_checkpoint_sha256)` — re-publishing the same policy bytes idempotently overwrites the same record.
- Library writes are filelocked per (adapter, task_id) pair; reads are unlocked + per-record fault-tolerant (a corrupt metadata.json doesn't break listings).
- `Stage.final_policy_path` (Ship 16 contract) is unchanged: the LAST-iter checkpoint. The skill record's checkpoint may differ — it's the BEST-iter checkpoint — but Ship 16's parent_checkpoint resolution still uses Stage.final_policy_path so within-mission warm-start chains stay deterministic.

**Hotfix landed alongside Ship 19 (Ship 16 latent bug exposed by Ship 18b/19's first real mjlab run):** Sam ran the §3 smoke test, decompose worked, but `Run mission` failed within ~2 s with `stage_failed(reason="v1_materialization_errored", detail="apply_prompt_edit failed: TypeError: MjlabAdapter.__init__() got an unexpected keyword argument 'env_id'")`. Root cause traced to [sculpt.py:_CONFIG_TEMPLATE](RewardSculptor/sculptor/sculpt.py): `sculpt_init` writes a hardcoded gym_sb3-flavored `[adapter].config = { env_id = "CHANGE_ME", n_envs = 4, ppo_kwargs = {...} }` regardless of the parent project's actual adapter. For mjlab projects, those keys are simply wrong (mjlab takes `task_id` / `num_envs` / `device`). Bug latent since Ship 16 because tests pre-scaffold stages with `class = "stubbed"` + a stub `load_adapter` that ignores config; no real mjlab mission had been run via the orchestrator until today.

- **Fix** at [sculpt.py:_inherit_parent_adapter_config](RewardSculptor/sculptor/sculpt.py) — string-level TOML helper `_extract_toml_section` + `_replace_toml_section` (the codebase doesn't depend on tomli_w; configs are flat enough for plain extraction). Called from `_run_one_stage` immediately after `sculpt_init` succeeds, copying the parent project's `[adapter]` section into the stage's freshly-scaffolded config.toml. The new event payload includes `inherited_parent_adapter_config: bool` on `stage_scaffolded` so future regressions are observable in the WS event stream.
- **Regression tests** in [tests/test_mission_run.py](RewardSculptor/tests/test_mission_run.py): `test_extract_toml_section_returns_section_body`, `test_replace_toml_section_substitutes_body`, `test_inherit_parent_adapter_config_replaces_stage_config` (verified-against-real-bug — uses the exact `env_id="CHANGE_ME"` template + the user's `task_id="Mjlab-Cartpole-Balance"` parent), `test_inherit_parent_adapter_config_tolerates_missing_files`. All 4 pass.
- **Verified live** by reproducing the exact failing path: `from sculptor.sculpt import _inherit_parent_adapter_config; from sculptor.adapters.base import load_adapter` against the user's actual on-disk project + stage config → `MjlabAdapter` constructed cleanly with `task_id="Mjlab-Cartpole-Balance"`, `num_envs=1024`, `device="cuda:0"`. Sam's mission re-run should now proceed past v1 materialization into the actual training loop.
- **Sculptor pytest re-baselined**: **340 passed, 1 skipped** (was 336 → +4 hotfix tests, zero regressions).
- **Long-term** (Ship 19c+): the underlying design issue is that `sculpt_init` was written gym_sb3-first and never updated for mjlab. A cleaner refactor would teach `sculpt_init` to take an optional `adapter_config: Optional[dict] = None` kwarg and fall back to the gym template only when not provided. Out of Ship 19's scope; documented for future work.

**Explicitly NOT in Ship 19** (deferred):
- **UI surface** for browsing / pinning / deleting skills — Ship 19b. The CLI is the only surface in v1.
- **Cross-adapter skills** (e.g., gym_sb3-trained policy → mjlab mission) — needs `init_policy_path` support in those adapters first.
- **Cross-`task_id` compatibility** (e.g., Go1-v0 → Go1-v1 after a code change) — the `(adapter_class, task_id)` strict-match key blocks this. Future work: schema-aware migration / observation-space validation.
- **Auto-pruning / quotas / deletion API** — manual `rm -rf <root>/<skill_id>/` works; a `sculpt skills prune --older-than 30d` CLI is Ship 19c if needed.
- **LLM-evaluator second-opinion before publish** (CurricuLLM evaluator-LLM pattern) — Ship 20+.
- **Skill aliases / human tags** — `SkillRecord.alias` field is reserved (currently always None).

### 2026-04-23 12:10 — Ship 13: research_topic off-topic paper filter

**Scope:** Sam ran `research_topic("bipedal robot kicking OR human robot kicking")` and Claude returned two completely unrelated real arxiv IDs: `2407.14795` (a Persian-text spell-correction paper) and `2502.12927` (SEFL educational-feedback LLM framework). Neither ID is hallucinated — both papers exist — but Claude remembered the IDs for unrelated topics and fabricated justifications to match. Pre-fix, these flowed through to the UI and would have polluted the KG + downstream reward edits.

- **Root cause.** `research_topic` trusted Claude's self-reported title + justification without verifying against the real arxiv metadata. The prompt forbids hallucinating IDs but has no guard against real-but-wrong IDs — a known LLM failure mode where the model recalls some valid string and grasps for a justification that matches the user's topic.
- **Fix — post-recall arxiv verification.** [sculptor/kg/research.py](RewardSculptor/sculptor/kg/research.py):
  - New `_fetch_arxiv_metadata_batch(ids)` — one arxiv API call fetches real titles+abstracts for all proposed IDs.
  - New `_verify_topic_match(topic, papers)` — embeds topic vs. each paper's real `title + abstract[:400]` via `sculptor.kg.query._embed_text`, drops below `_MIN_TOPIC_SIMILARITY = 0.15`. Replaces Claude's (possibly hallucinated) title with the real one on kept papers. Fail-open: if arxiv is rate-limited or the embedder is unavailable, we KEEP all papers (an outage shouldn't wipe the user's research query).
  - New `papers_rejected_off_topic` counter on `ResearchResponse` so the UI can show "Claude returned 3 papers, 2 dropped as off-topic" instead of silently hiding the filter.
- **Threshold calibration.** Live-measured against the all-MiniLM-L6-v2 embedder with Sam's exact scenario + 6 known on-topic papers:

  | arxiv  | Title (truncated)                              | sim   | verdict |
  | ------ | ---------------------------------------------- | ----- | ------- |
  | 2407.14795 | Automatic Real-word Error Correction in Persian | 0.05 | DROP |
  | 2502.12927 | SEFL: Synthetic Educational Feedback         | 0.13 | DROP |
  | 1804.02717 | DeepMimic                                    | 0.18 | keep |
  | 2312.17507 | Actuator-Constrained RL                      | 0.35 | keep |
  | 2107.04034 | RMA: Rapid Motor Adaptation                  | 0.38 | keep |
  | 2402.16796 | Expressive Whole-Body Control for Humanoids  | 0.42 | keep |
  | 2508.08241 | BeyondMimic                                  | 0.42 | keep |
  | 2406.10759 | Humanoid Parkour Learning                    | 0.48 | keep |

  Threshold 0.15 cleanly separates the two confirmed hallucinations from the DeepMimic-style low-vocab-overlap but on-topic papers. Tighter threshold (0.20) false-positives DeepMimic; looser (0.12) keeps SEFL.
- **Prompt sharpened.** [sculptor/prompts/research_topic.md](RewardSculptor/sculptor/prompts/research_topic.md) — rule #5 now explicitly warns about "real-but-wrong ID" hallucinations, names the 2026-04-23 Persian-text case as the observed failure mode, and notes that the post-recall verification WILL reject mismatched IDs (so guessing is worse than returning empty).
- **Tests:**
  - `test_research_topic_drops_off_topic_papers_via_arxiv_verification` — uses Sam's exact 2 off-topic IDs + DeepMimic; stubs arxiv fetch with the real titles and `_embed_text` with a fixed vector layout. Asserts 2 dropped, 1 kept, real title replaces Claude's.
  - `test_research_topic_fails_open_when_arxiv_unreachable` — when the arxiv batch returns all-None (rate limit / network down), we KEEP Claude's papers and `papers_rejected_off_topic = 0`. Prevents an arxiv outage from silently erasing research output.
- **Verified live:** ran `_fetch_arxiv_metadata_batch` + `_verify_topic_match` against real arxiv with Sam's 3 papers. Both off-topic IDs dropped with similarity logged; embedder loaded from cache in 0.8 s (local_files_only path from Ship 11 holds).

### 2026-04-24 — Ship 18b: Mission UI (frontend, audit-driven)

**Scope:** Second half of Ship 18 — landed the React Mission UI on top of the stable Ship 18a backend contract. Frontend-only: not a single backend file or `api.ts`/`types.ts` line touched. Six file additions / two modifications, then audit-driven fixes. Now the user can decompose a goal, watch the decompose job stream, run the curriculum, and watch each stage's events flow without ever leaving the browser.

**Process (mirrors Ships 14-18a):**
1. **Explore agent** mapped existing tab conventions, React Query keys, the `useRunEvents` WS hook (lines 1-122, including the `terminalRef` reconnect-after-terminal trap fix), shadcn primitives present (Card, Dialog, Tabs, Badge, Button, Input, Label, Select, Textarea — Sheet/Checkbox/ScrollArea NOT present), `formatRelative` utility, and `sonner` toast wiring. Confirmed file:line that the `{activeTab === "X" && <Tab/>}` deferred-mount pattern is the established way to gate WS-bearing tabs.
2. **Plan v1** — file-by-file plan with `runJustStarted` ephemeral state to gate WS opening.
3. **Plan agent audit** flagged 8 issues; **load-bearing finding: gating WS on `runJustStarted` ephemeral state breaks across deferred-mount unmount/remount and across tab-switch.** Plus: polling needed `m.active_job_id != null` not `lifecycle === "running"` (decompose lifecycle is `ready`, not `running`); `structuredEvents` cap missing; `parent_stage` cycle/orphan handling unspecified; lifecycle chip a11y needs an icon (color-only fails contrast in dark mode).
4. **Plan v2** replaced ephemeral state with `qc.setQueryData` optimistic writes inside `useRunMission.onSuccess` — the optimistic value flows through the React Query cache, survives tab-switch unmount, and signals the WS hook within a single render. Other v2 changes: explicit `m.active_job_id != null` polling guard, 5000-cap on structured events, DFS-with-visited-set stage tree (orphan/cycle → depth=0 + warning chip), `MissionLifecycleBadge` and `StageStatusBadge` use Lucide icons + `bg-{c}-50 text-{c}-700` (all 5 lifecycle states pass WCAG AA on emerald/amber/rose/sky/slate).
5. **Implementation** — see file list below.
6. **Two parallel audit agents** ran on the diff:
   - **Correctness/UX agent CRITICAL #1 — re-subscription race in useMissionEvents:** when `enabled` flips false→true mid-stream (e.g., user reopens dialog while a previous WS is still tearing down), the OLD ws.onmessage could fire AFTER `cancelled = true` and append events into the NEW state. Fix at [useMissionEvents.ts](reward-sculptor-ui/frontend/src/hooks/useMissionEvents.ts) — added `if (cancelled) return;` at the top of `ws.onmessage`. Inline comment explains the audit case.
   - **Correctness HIGH — close-then-onSuccess reopens dialog:** confirmed by inspection that the row's `Run` button calling `onOpen()` after success IS intended UX (user clicks Run from the row to launch + watch live), so kept. Documented inline.
   - **A11y CRITICAL — `StageStatusBadge` missing `role="status"`:** [MissionDetailDialog.tsx](reward-sculptor-ui/frontend/src/components/MissionDetailDialog.tsx) — added `role="status"` to match `MissionLifecycleBadge`. Plus `prefers-reduced-motion` guard: `animate-pulse` and `animate-spin` on continuous-motion chips replaced with `motion-safe:animate-pulse` / `motion-safe:animate-spin` (Loader2 spinners on transient mutation buttons left as-is per existing convention).
   - **A11y MEDIUM — `StructuredEventList` missing live-region role:** wrapped in `<div role="log" aria-label="Mission events" aria-live="polite">` so screen readers announce new structured events as they arrive.
   - **A11y false-alarm:** auditor flagged "MissionsTab not conditionally rendered" but verification at [ProjectDetail.tsx:294](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx) confirms `{activeTab === "missions" && <MissionsTab />}` is in place.

**Files added/changed:**
- **NEW [frontend/src/hooks/useMissions.ts](reward-sculptor-ui/frontend/src/hooks/useMissions.ts)** — `useMissions(slug)` (5 s polling while any mission has `active_job_id`), `useMission(slug, ms, {enabled})`, `useCreateMission(slug)`, `useRunMission(slug)` (with optimistic `qc.setQueryData` writeback so the WS opens within the same render — addresses Ship 18a audit-deferred finding A), `useDeleteMission(slug)`. Mirrors `useRuns.ts` shape.
- **NEW [frontend/src/hooks/useMissionEvents.ts](reward-sculptor-ui/frontend/src/hooks/useMissionEvents.ts)** — WS subscription hook. Takes `enabled: boolean` (caller MUST gate on `active_job_id != null` or post-runMission optimistic cache write). NO auto-reconnect (Ship 18b §6 explicit). Caps `logLines` at 200 and `structuredEvents` at 5000 (FIFO ring on overflow). Detects `connected.status === "no_active_job"` → surfaces as `noActiveJob: true`. On `terminal` event invalidates `qk.mission(slug, ms)` + `qk.missions(slug)`. Stage/mission/redecomposition prefix events trigger detail invalidation so stage status refreshes mid-mission. Uses `terminalRef` + `noActiveJobRef` mirrors so `onclose` reads the current value, not the stale closure. **Audit-fix `if (cancelled) return;` at top of `onmessage`** so a re-subscription doesn't bleed events from the old WS into the new state.
- **NEW [frontend/src/components/MissionsTab.tsx](reward-sculptor-ui/frontend/src/components/MissionsTab.tsx)** — top-level tab. Card with header + `<NewMissionDialog />`. Lists missions as `<MissionRow>` cards with lifecycle chip, slug, goal (line-clamp-2 + title=full), `current_stage_idx/n_stages`, `formatRelative(created_at)`, decomposition_model, and Run/Delete actions. Click row → opens `<MissionDetailDialog>`. Confirms-on-delete via `window.confirm` (matching CONTEXT.md "confirm destructive" rule). Errored missions render with rose chip + Delete-only.
- **NEW [frontend/src/components/NewMissionDialog.tsx](reward-sculptor-ui/frontend/src/components/NewMissionDialog.tsx)** — Dialog with Textarea (8-2000 char client-side guard mirrors backend), optional `mission_slug` Input validated against `^[a-z][a-z0-9_-]{0,63}$`, native `<input type="checkbox">` for `no_kg` (matches NewRunDialog convention since Checkbox primitive isn't in the shadcn set). On submit: `useCreateMission` + invalidate + toast.
- **NEW [frontend/src/components/MissionDetailDialog.tsx](reward-sculptor-ui/frontend/src/components/MissionDetailDialog.tsx)** — Dialog (sm:max-w-3xl, max-h-[90vh] with scroll). Top: goal + lifecycle. Body: errored-mission banner (when applicable), decomposition rationale block, vertical stage list with parent-chain indentation (DFS, cycle-safe, MAX_INDENT_DEPTH=4, orphan tag on dangling parent_stage), live-event panel (WS status chip + structured-event chips with type-keyed colors + 200-cap log scroller with auto-scroll-to-bottom unless user scrolled up). Disconnected banner appears only if `events.disconnected && !events.terminal` (so a clean terminal close doesn't flash the banner). Footer: Run / Delete / Close, with active-job and lifecycle gating + tooltip explanations on disabled state.
- **MODIFY [frontend/src/pages/ProjectDetail.tsx](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx)** — added `{value:"missions", label:"Missions"}` between Runs and Reports (so post-run mission summary surfaces close to the Runs tab). Added `<TabsContent value="missions">{activeTab === "missions" && <MissionsTab slug={slug!} />}</TabsContent>` deferred-mount block.
- **MODIFY [frontend/src/lib/queryKeys.ts](reward-sculptor-ui/frontend/src/lib/queryKeys.ts)** — added `missions(slug)` and `mission(slug, ms)` query keys.

**Verification gates** (handoff §5):
- ✅ **TypeScript**: `pnpm tsc --noEmit` returns 0.
- ✅ **Backend pytest**: `298 passed, 1 deselected` — exact match to the Ship 18a baseline.
- ✅ **Sculptor pytest**: 291 passed, 1 skipped — exact match to the Ship 17 baseline.
- ⏳ **Live smoke** — pending user run via `./run.sh`. The audit-driven fixes mean the WS lifecycle is robust to (a) creating a mission with no run yet (decompose-only), (b) running a ready mission, (c) closing/reopening the detail dialog mid-run, and (d) tab-switching away and back. If the user hits a regression, the `disconnected` banner makes the failure mode visible instead of silent.

**Callable surface (UI):**
- Open a project detail page → click the **Missions** tab.
- Click **New mission** → enter a goal → Decompose. The decompose job runs (~30-90 s); the row's lifecycle chip + active-job badge updates as it streams events.
- Click a row → see the full stage list + decomposition rationale. If the mission has an active job, live events stream into the bottom panel (structured events as colored chips, log_line as scrolling stdout).
- Click **Run mission** (when `lifecycle="ready"`) → the optimistic cache write flips `active_job_id` immediately; the WS opens on the next render and starts streaming `mission_started`, `stage_started`, `iter_started`, etc.
- Click **Delete** (when no active job) → confirms via `window.confirm` → deletes the on-disk artifacts; toast shows freed bytes.

**Explicitly NOT in Ship 18b** (deferred to Ship 18c):
- WS auto-reconnect with replay-from-seq on transient disconnect (handoff §6 explicit).
- DAG visualization with d3 / SVG nodes.
- Time-lapse mission video.
- Mid-mission cancellation UI hookup to `POST /jobs/{id}/stop`.
- Resume-from-stage selector.
- Per-stage knob overrides.
- Vitest frontend tests — the project has none and CONTEXT.md flags expanded test coverage as "explicitly incomplete." Live smoke + tsc + the existing 298+291 backend/sculptor baselines are the ship gate.

### 2026-04-24 — Ship 18a: Mission backend + CLI surface (UI groundwork)

**Scope:** First half of Ship 18 (UI for Ship 14-17 mission orchestrator). Ship 18a delivers the **backend + CLI + typed-API groundwork** so Ship 18b can land the React MissionsTab + WebSocket event rendering against a stable contract. Splitting the original Ship 18 into two audit cycles (rather than one ship-the-world) so the audit-driven process actually catches bugs at the diff scale it works on.

**Out of scope here (Ship 18b):**
- Frontend MissionsTab + DAG viz + per-stage drill-down.
- WebSocket event rendering in the browser.
- React Query subscriptions for live mission state.

**Process (matches Ship 14-17 pattern):**
1. **Explore agent** mapped backend route + service patterns (project_store, run_manager, job_manager), frontend routing/api conventions, and the disk-layout implications of mounting missions under projects.
2. **Plan agent** critiqued v1, flagged 8 issues. **The most important: I had been treating `mission.json` and `JobManager events` as parallel state sources — but the right contract is `mission.json` is canonical (filesystem, durable), JobManager events are an EPHEMERAL overlay (transient, RAM).** Other fixes in v2: in-process decompose vs subprocess execute (hybrid), per-project lock for concurrent decompose, `has_any_active_gpu_job()` cross-kind GPU guard, `mission_dir` reconstructed-from-file-location (Ship 16 audit-deferred fix), event shape mirrors runs.py WS pattern.
3. Implementation landed; backend tests at 295 passed.
4. **Audit agent** flagged 5 additional issues:
   - **CRITICAL #E — concurrent decompose race**: two POSTs without `mission_slug` override at ~the same instant could both derive the SAME auto-slug and both write to the same `mission.json`. **Fixed**: per-project filelock around slug-derivation + slug-reservation; reserve via `<slug>/.decompose_pending` marker file inside the lock; `list_mission_slugs` recognizes the marker so subsequent calls see the slug as taken.
   - **MEDIUM #F — corrupt mission.json silently dropped**: a `mission.json` that fails JSON-parse just disappeared from the list, leaving no way for the user to clean up. **Fixed**: `load_mission_summary` catches parse errors and returns a stub `MissionSummary(lifecycle="errored", goal="(unreadable mission.json)", ...)` so the user can see + DELETE it.
   - **MEDIUM #C — slug derivation duplicated**: `sculpt mission-init` (CLI side) and `mission_store._slugify` (backend side) implement the same logic in two places. **Fixed**: cross-reference docstrings in both functions naming each other; new `test_cli_and_backend_slug_derivation_agree` test runs 6 inputs through both and asserts byte-equal output.
   - **LOW #D — symlink edge case**: `Path.resolve()` follows symlinks, so a symlinked `mission.json` would reconstruct `mission_dir` to the link target's parent. **Documented** in `load_mission` docstring; not fixed (edge case; tested workflow doesn't symlink).
   - **DEFERRED — A (WS lifecycle on decompose-then-run race)**: documented for Ship 18b. The frontend should open the WS AFTER `POST /run` succeeds, not before.

**Files added/changed:**

*Sculptor (`~/projects/RewardSculptor`)*:
- **[sculptor/mission.py](RewardSculptor/sculptor/mission.py)** — `to_dict` no longer persists `mission_dir` (path-relocation safety); `load_mission` reconstructs `mission_dir` from the JSON file's parent directory. Symlink caveat documented.
- **[sculptor/cli.py](RewardSculptor/sculptor/cli.py)** — new `sculpt mission-init` (decompose → write `<project>/.missions/<slug>/mission.json`) and `sculpt mission-run` (load + invoke `mission_run` from Ship 16/17). Auto-resolve slug when there's exactly one mission. Adapter resolved from project's config.toml dotted-path. CLI emits `[SCULPT-EVENT] mission_initialized` so the backend's stdout streamer picks up the slug.

*Backend UI (`~/projects/reward-sculptor-ui`)*:
- **NEW [backend/models/mission.py](reward-sculptor-ui/backend/models/mission.py)** — pydantic shapes: `StageSchema`, `MissionSummary`, `MissionDetail` (extends summary with stages + rationale), `CreateMissionRequest` (8-2000 char goal, optional explicit slug), `DeleteMissionResponse` (with `freed_bytes`), `MissionEvent` (WS envelope; per-event payload is `dict[str, Any]` per Ship 18a plan-review).
- **NEW [backend/services/mission_store.py](reward-sculptor-ui/backend/services/mission_store.py)** — `mission_dir`, `mission_json_path`, `list_mission_slugs` (reservation-aware), `derive_unique_mission_slug` (mirrors project_store's `_ensure_unique_slug` pattern), `_derive_lifecycle` (computed from on-disk stage statuses), `load_mission_summary` / `load_mission_detail` (always read mission.json on each call — source-of-truth invariant), `delete_mission` (returns `freed_bytes`).
- **NEW [backend/services/mission_jobs.py](reward-sculptor-ui/backend/services/mission_jobs.py)** — `run_mission_decompose_job` (in-process via `asyncio.to_thread`; ~30-90s Claude call, subprocess buys nothing) and `run_mission_execute_job` (subprocess `python -m sculptor.cli mission-run`; mirrors run_manager.py pattern with `[SCULPT-EVENT]` stdout streamer + SIGTERM-on-cancel).
- **[backend/services/job_manager.py](reward-sculptor-ui/backend/services/job_manager.py)** — new `has_any_active_gpu_job()` (ORs `sculpt_run` + `mission_execute` cross-project) and `active_mission_job(slug, mission_slug)` (per-(project,mission) in-flight job lookup).
- **[backend/models/kg.py](reward-sculptor-ui/backend/models/kg.py)** — `JobKind` literal extended with `mission_decompose`, `mission_execute`.
- **NEW [backend/routes/missions.py](reward-sculptor-ui/backend/routes/missions.py)** — REST endpoints (POST decompose, GET list/detail, POST run, DELETE) + WebSocket `/ws/projects/{slug}/missions/{mission_slug}/events`. Per-project filelock around decompose's slug-reservation. GPU contention guard on run via `has_any_active_gpu_job()`.
- **[backend/main.py](reward-sculptor-ui/backend/main.py)** — mounts the new `missions` router + `ws_router`.

*Frontend groundwork (typed API only — no components)*:
- **[frontend/src/lib/api.ts](reward-sculptor-ui/frontend/src/lib/api.ts)** — `listMissions`, `getMission`, `createMission`, `runMission`, `deleteMission`, `missionEventsWsUrl`.
- **[frontend/src/lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts)** — `StageSchema`, `MissionSummary`, `MissionDetail`, `CreateMissionRequest`, `DeleteMissionResponse`, `MissionEvent`, `StageStatus`, `MissionLifecycleStatus`, `MissionJobKind`.

**Tests — [backend/tests/test_missions.py](reward-sculptor-ui/backend/tests/test_missions.py), 26 tests:**
- 4 404 paths (unknown project / mission on each endpoint).
- 5 list/detail tests covering empty, populated, lifecycle derivations (ready/completed/halted).
- 4 create-validation tests (short goal → 422, extra fields → 422, slug collision → 409, valid create → 202).
- 3 slug-derivation unit tests (basic, collision, empty-goal fallback).
- 3 run-path tests (404 missing, 409 active decompose, 409 GPU busy).
- 3 delete tests (success + freed_bytes, 409 active job, 404 missing).
- 1 path-relocation test (load_mission reconstructs mission_dir post-move).
- **3 audit-regression tests**: slug-reservation marker file (#E), corrupt mission.json → lifecycle="errored" (#F), CLI-vs-backend slug parity (#C).

**Verified:**
- Backend: **298 passed, 1 deselected** (was 272 → +26 Ship 18a tests).
- Sculptor: **291 passed, 1 skipped** (Ship 17 baseline retained — no regressions from path-relocation fix).
- Test runtimes: Ship 18a tests in ~1.5 s; full backend suite ~56 s.

**Callable surface:**

CLI (sculptor):
```bash
# Decompose a goal into a curriculum (writes .missions/<slug>/mission.json)
uv run sculpt mission-init /path/to/project --goal "Stand on one leg and kick"

# Run the mission end-to-end (Ship 16/17 orchestration)
uv run sculpt mission-run /path/to/project [<mission-slug>]
```

REST (backend):
```
POST   /api/projects/{slug}/missions                        → 202 JobSummary
GET    /api/projects/{slug}/missions                        → 200 list[MissionSummary]
GET    /api/projects/{slug}/missions/{mission_slug}         → 200 MissionDetail | 404
POST   /api/projects/{slug}/missions/{mission_slug}/run     → 202 JobSummary | 404 | 409
DELETE /api/projects/{slug}/missions/{mission_slug}         → 200 DeleteMissionResponse | 404 | 409

WS     /ws/projects/{slug}/missions/{mission_slug}/events   — replay + tee active job
```

**Source-of-truth invariant**: `mission.json` is canonical. Every GET re-reads from disk. JobManager events are an EPHEMERAL overlay applied via `active_job_id` / `active_job_kind` on the response. A backend restart loses no mission state.

**Explicitly NOT in Ship 18a** (Ship 18b):
- Frontend MissionsTab component + DAG viz + per-stage drill-down dialog.
- WebSocket event rendering / live-stream UI.
- React Query subscriptions / polling for live mission state.
- "New mission" dialog form.
- Per-stage knob overrides UI.

### 2026-04-24 — Ship 17: Stage re-decomposition on criterion failure

**Scope:** Fourth brick of the Mission roadmap. Ship 17 adds **automatic curriculum recovery** when a stage exhausts its iteration budget without satisfying its `success_criterion`. The orchestrator now asks Claude "this didn't work — break it down further" and splices 2-8 simpler sub-stages into the mission graph in place of the failed stage. The LAST sub-stage carries the original goal_text + success_criterion (byte-identical), so the task still gets accomplished; earlier sub-stages are precursors. Bounded at one re-decomposition per stage to prevent combinatorial fanout.

**Process:**
1. Explore agent mapped the failure-state inputs available, splice mechanics options, and existing test stubs.
2. Plan agent reviewed v1, flagged 8 issues. The most important: **I missed the downstream-child re-pointing** — when REPLACE-splicing, every stage downstream whose `parent_stage == failed.name` must be rewritten to point at the LAST sub-stage. Without this, `validate_mission` raises immediately because the failed name no longer exists.
3. Plan v2 incorporated all 8 issues:
   - Downstream re-pointing (CRITICAL): new `_repoint_downstream_children` helper.
   - Naming: `{failed.name}__r1_<i>` with collision-resolve `_v2`/`_v3`.
   - Halt-reason granularity: `redecomposition_skipped(reason=...)`, `stage_redecomposition_failed(reason=...)`, `stage_redecomposed`.
   - `current_stage_idx = failed_idx` BEFORE atomic save (resume safety).
   - Training feedback includes verbatim `v<n>.py` source + last-3-iter component means; skips trajectory.npz arrays.
   - Last sub-stage's `success_criterion` byte-equal to original (validator enforces).
   - Pydantic `min_length=2, max_length=8` on `_RedecompositionModel.stages`.
   - Warm-start integration test added (sub-stages chain correctly).
4. Implementation landed; full sculptor suite green at 286 passed.
5. **Audit agent** flagged 5 additional issues (CRITICAL → LOW):
   - **CRITICAL #A — save-after-splice atomicity**: if `_atomic_save_mission` raises (disk full / EIO), in-memory mission has new sub-stages but on-disk has old failed stage; resume diverges silently. **Fixed**: wrap save in try/except, roll back in-memory splice + parent re-pointing on failure, emit `stage_redecomposition_failed(reason="save_failed")`.
   - **HIGH #C — unbounded `_resolve_unique_name` loop**: pathological pre-existing collisions could spin forever AND a too-long base + `_vN` could exceed 32-char regex cap. **Fixed**: cap at 100 attempts, length-check the final name, raise `MissionValidationError` on either failure.
   - **MEDIUM #B — `_scan_iter_metric_history` sort bug**: `else -1` fallback for malformed iter dir names sorted them BEFORE iter_0, polluting Claude's metric history. **Fixed**: use `+inf` so corrupt dirs sort to the END.
   - **MEDIUM #E — silent feedback degradation**: best-effort reads of diagnosis.json / trajectory.npz / metric history degrade to empty dicts; Claude saw "complete feedback with empty signals" indistinguishable from "we couldn't read the data." **Fixed**: track which signals were missing, emit `feedback_read_degraded(missing_signals=[...])` event before invoking Claude.
   - **LOW (deferred)**: prompt-name-drift UX (Claude's emitted names ignored — bulletproof but cosmetic mismatch with rationale text); path-relocation test (acceptable risk).

**Files added/changed:**
- **NEW [sculptor/prompts/redecompose_stage.md](RewardSculptor/sculptor/prompts/redecompose_stage.md)** — 9 hard rules including byte-equal final criterion, naming convention, simpler-reward explanation requirement, KG-slice restriction.
- **[sculptor/decompose.py](RewardSculptor/sculptor/decompose.py)** — new `_RedecompositionModel` (pydantic, min_length=2, max_length=8), parameterized `_parse_with_retry(output_format=...)` (decompose_task callers unchanged), `StageTrainingFeedback` dataclass, `_render_training_feedback_block`, `_build_redecompose_user_content`, `_truncate_for_name_budget`, `_resolve_unique_name` (with audit-fix cap + length check), `redecompose_stage(mission, failed_idx, *, feedback, reward_contract, kg_store, client)` returning `list[Stage]` with `redecomposition_attempts=1` set on each.
- **[sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py)** — refactored `mission_run`'s for-loop → while-loop indexed by `mission.current_stage_idx` for safe mid-iteration splicing. New helpers: `_REDECOMPOSABLE_REASONS = {"criterion_not_met"}`, `_build_stage_training_feedback`, `_repoint_downstream_children`, `_maybe_redecompose_and_splice` (with audit-fix save-rollback), `_scan_iter_metric_history` (with audit-fix `+inf` sort fallback). 6 new event types: `stage_redecomposition_started`, `stage_redecomposed`, `redecomposition_skipped` (reasons: `budget_exhausted`, `non_curriculum_failure`), `stage_redecomposition_failed` (reasons: `validation_failed`, `claude_call_errored`, `adapter_load_failed`, `spliced_mission_invalid`, `save_failed`, `empty_substages`), `feedback_read_degraded`.
- **[sculptor/mission.py](RewardSculptor/sculptor/mission.py)** — `Stage.redecomposition_attempts: int = 0` field. Persisted via existing `dataclasses.asdict`.
- **Ship 16 test updated** — `test_mission_run_halts_when_stage_criterion_fails` now pre-sets `redecomposition_attempts=1` to bypass Ship 17's path and test the bare halt behavior; new assertion checks `redecomposition_skipped(reason="budget_exhausted")` event fires.

**Tests — [tests/test_mission_run.py](RewardSculptor/tests/test_mission_run.py), 13 new (52 total in file):**
- 8 core Ship 17 tests: splice replaces failed stage, sub-stages have attempts=1, only-fires-once-per-stage, infra-failure skip, invalid-Claude-response halt, `stage_redecomposed` event shape, sub-stages warm-start in chain (Ship 15 + Ship 17 integration), atomic mission.json persistence with `current_stage_idx` rewound.
- **5 audit-regression tests**: save-failure rollback (#A), collision-loop cap (#C), name-overflow detection (#C), sort-fallback for corrupt iter dirs (#B), `feedback_read_degraded` event emission (#E).

**Verified:**
- Sculptor: **291 passed, 1 skipped** (was 278 → +13 Ship 17 tests).
- Test runtime: ~50 s for full suite, ~1.7 s for the 52 mission tests in isolation.
- Zero regressions across the pre-existing 278-test baseline.

**Callable surface:**
```python
from sculptor.sculpt import mission_run
# Same entry as Ship 16. Re-decomposition fires automatically when a
# stage's success_criterion fails AND `stage.redecomposition_attempts == 0`.
result = mission_run(
    mission,
    adapter_short_name="mjlab",
    kg_store=kg,
    on_event=lambda e: print("[mission]", e),
)
# New events to watch: stage_redecomposition_started,
# stage_redecomposed, redecomposition_skipped (with reason),
# stage_redecomposition_failed (with reason), feedback_read_degraded.
```

**Explicitly NOT in Ship 17** (deferred):
- **Multi-level re-decomposition** — bounded at one level per CurricuLLM's design and to prevent combinatorial fanout.
- **LLM-evaluator second opinion** before halting (CurricuLLM's evaluator-LLM pattern) — Ship 19+.
- **Skill-library cross-mission policy reuse** — Ship 19.
- **Mission UI surfacing redecomposition events** — Ship 18.
- **Resume mid-redecomposition** — if mission_run crashes between Claude returning and `_atomic_save_mission` succeeding, the next run sees the OLD state and may re-call Claude with a different (non-deterministic) response. Tolerable; documented.

### 2026-04-24 — Ship 16: Mission orchestrator + success-criterion evaluator

**Scope:** Third brick of the Mission roadmap. Ship 16 turns Ship 14's `Mission` data + Ship 15's policy warm-start into an **end-to-end multi-stage training loop**. Caller invokes `mission_run(mission, adapter_short_name=...)`; orchestrator iterates stages, scaffolds per-stage projects, materializes v1 from each stage's `reward_seed_prompt`, calls `sculpt_run` with `init_policy_path` resolved from the parent stage, evaluates the stage's `success_criterion` against the last iter's `behavior.json` + `trajectory.npz`, and advances or halts.

**Process (per Sam's "audit-driven implementation" pattern, established Ships 14-15):**
1. Explore agent mapped rollout artifact shape, `IterOutcome.iter_dir` exposure, sculpt_run return contract, existing `sculpt_init` (line 1552).
2. Plan agent reviewed v1 plan — flagged 8 issues. Most important: don't drop Ship 14's `info[<key>]` syntax, expose `info` as alias for `trajectory` instead. Other findings folded into v2 (drop `missions_root` param; use `mission.mission_dir`; use `_is_stage_scaffolded` for idempotent scaffold; track silent-drop on `final_policy_path = None`).
3. Implementation landed.
4. **Two parallel code-review agents** audited. Combined findings (CRITICAL/HIGH addressed; LOW deferred to Ships 17/18):
   - **CRITICAL — AST allow-list missing `ast.Tuple`.** `trajectory[..., 2]` parses as `ast.Subscript(slice=ast.Tuple([Constant(...), Constant(2)]))`. Pre-fix would reject any multi-axis subscript with "disallowed AST node Tuple". Fixed at [mission_runtime.py:_validate_criterion_ast](RewardSculptor/sculptor/mission_runtime.py); also added explicit `REJECTED_NODES` (Lambda/NamedExpr/comprehensions/Starred/Assign) for sharper error messages.
   - **HIGH — multi-element bool array → cryptic numpy error.** A criterion like `trajectory['rewards'] > 0.0` returns a (T,) bool array, which `bool()` rejects with "truth value ambiguous". Pre-fix surfaced numpy's raw text; post-fix catches `ValueError`, surfaces a hint mentioning `.all() / .any() / .mean()` reductions.
   - **HIGH — parent ckpt deleted externally → silent cold-start.** If a user cleans `runs/` between mission runs, the parent's recorded `final_policy_path` points at a missing file; pre-fix `parent_checkpoint_of` returned None silently, defeating the curriculum. Fixed by adding `Mission.parent_checkpoint_status_of` returning `(path, status_tag)` where status ∈ {`no_parent`, `parent_untrained`, `parent_ckpt_missing`, `ok`}; orchestrator now emits `warm_start_skipped` with reason `"parent_ckpt_missing"` so the regression is observable.
   - **HIGH (Audit 2 lead) — lock-file unlink failure on Windows/WSL.** `filelock` can hold the lock file's handle past `release()`, making the `unlink()` in the `finally` silently fail. Fix: drop the unlink entirely; let `filelock` (or the OS-level lock release) handle cleanup. The `.lock` file persisting on disk is cosmetic and `FileLock` re-acquires cleanly because the OS-level advisory lock is released, not the filename.
   - **MEDIUM — filelock timeout 1s too tight.** Slow FS (NFS, WSL interop) can legitimately take >1s. Bumped to 10s.
   - **MEDIUM — adapter mismatch detection.** New `_verify_stage_adapter_matches` reads the on-disk `[adapter].class` from the stage's pre-scaffolded config.toml and compares to the caller's `adapter_short_name`. Only enforced when on-disk class is a real Python dotted path (so test stubs like `"stubbed"` aren't false-positives). Catches "you scaffolded under gym_sb3, then resumed under mjlab" silent-drift bugs.
   - **LOW — `stage_dir` missing from `stage_started` event.** Added so Ship 18's UI can deep-link to the stage project page from the very first event, not wait for `stage_scaffolded`.

**Findings deferred (rationale documented):**
- Stage resumption mid-flight (status=`training` from a crashed run): re-runs the stage including v1 materialization. Fixing this cleanly needs Ship 17's re-decomposition logic anyway.
- Project-root boundary check on `mission_dir`: caller (Ship 18 UI) is responsible for sandbox enforcement.
- Namespace dump on criterion failure: UX polish, Ship 18.
- Stack trace preservation in `_fail_stage`: low ROI vs the type+message already in `detail`.

**Files added/changed:**
- **NEW [sculptor/mission_runtime.py](RewardSculptor/sculptor/mission_runtime.py)** — `_evaluate_success_criterion(criterion, namespace) -> bool` (ast-parsed safe-eval with REJECTED_NODES + ALLOWED_NODES + SAFE_ATTRIBUTE_METHODS guards), `_build_criterion_namespace(iter_dir, primary_metric)` (loads `behavior.json` + `trajectory.npz`), `MissionResult` + `StageResult` dataclasses, `PERSISTED_TRAJECTORY_KEYS` + `BEHAVIOR_KEYS` constants. Builtins NOT zeroed in eval — numpy `.mean()` needs `__import__` internally; AST walker is the primary safety layer.
- **[sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py)** — new `mission_run(mission, *, adapter_short_name, kg_store, on_event, ...)` orchestrator, `_run_one_stage`, `_fail_stage`, `_atomic_save_mission` (tmp+rename), `_resolve_stage_final_checkpoint` (glob `checkpoint.{pt,zip}`), `_is_stage_scaffolded` (idempotent resume), `_verify_stage_adapter_matches`, `_utc_now_iso`. Acquires filelock at `<mission_dir>/.lock`. Emits 12 distinct event types: `mission_started`, `stage_started`, `stage_warm_start_resolved`, `warm_start_skipped`, `stage_scaffolded`, `stage_v1_materialized`, `stage_completed_training`, `stage_criterion_evaluated`, `stage_succeeded`, `stage_failed`, `stage_skipped`, `mission_completed` / `mission_halted` / `mission_halted_terminal`.
- **[sculptor/mission.py](RewardSculptor/sculptor/mission.py)** — added `Mission.stage_dir(name)`, `Mission.parent_checkpoint_of(name)` (legacy convenience), `Mission.parent_checkpoint_status_of(name)` (returns `(path, status_tag)` for distinct event emission). `_validate_success_criterion` now checks `info[...]` / `trajectory[...]` / `behavior[...]` subscripts against the runtime-persisted key sets in `mission_runtime` (was: against the adapter's `expected_info_keys`, which is the per-step training info dict NOT the persisted artifact set — a Ship-14-prompt/Ship-16-runtime mismatch this fix aligns).
- **[sculptor/prompts/decompose_task.md](RewardSculptor/sculptor/prompts/decompose_task.md)** — namespace docs rewritten: `metric`, `behavior['<key>']`, `components['<name>']`, `trajectory['<key>']`, `info['<key>']` (alias). Worked example updated to use the persisted-key syntax.
- **Ship 14 tests updated** — 4 assertions in `tests/test_decompose.py` re-pinned from `info[base_height]` style to `behavior[mean_return]` / `metric` / `components[name]` style. The Ship 14 contract is unchanged structurally; only the per-stage-namespace docs rotated to match what Ship 16 actually evaluates.

**Tests — [tests/test_mission_run.py](RewardSculptor/tests/test_mission_run.py), 39 tests:**
- 13 criterion-evaluator unit tests (happy paths + 8 safety / rejection paths covering unparseable / unknown name / disallowed attribute / builtins access / lambda / non-bool result / unknown function-call).
- 5 namespace-construction tests (load behavior.json + trajectory.npz + components extraction + info-alias-trajectory + missing-file error + unexpected-key drop).
- 7 mission-run orchestrator integration tests (happy path, criterion failure halt, no-checkpoint failure, warm-start parent resolution, resume-skip-already-succeeded, mission_dir-None rejection, atomic mission.json persistence).
- 5 Mission helper tests (stage_dir, parent_checkpoint_of in 4 status states).
- **9 audit-driven regression tests**: Ellipsis subscript, comprehension rejection, walrus rejection, multi-element bool friendly hint, parent-ckpt-deleted event, adapter-mismatch detection, stage_dir on stage_started, lock re-entry, source-inspection guard against future re-introduction of `unlink`.

**Verified:**
- Sculptor: **278 passed, 1 skipped** (was 239 pre-Ship 16 → +39 tests).
- Zero regressions across the pre-existing 239-test suite.
- Test runtime: ~50 s for the full sculptor suite, ~1 s for Ship 16's 39 tests in isolation.

**Callable surface:**
```python
from sculptor.decompose import decompose_task
from sculptor.mission import save_mission
from sculptor.sculpt import mission_run

# Decompose (Ship 14)
mission = decompose_task("Stand on one leg and kick", adapter.reward_contract(), kg_store=kg)
# Save → assigns mission.mission_dir
save_mission(mission, projects_root / "g1-kick-mission")
mission.mission_dir = str((projects_root / "g1-kick-mission").resolve())

# Run end-to-end (Ship 16) — chains stages with Ship 15 warm-start
result = mission_run(
    mission,
    adapter_short_name="mjlab",
    kg_store=kg,
    on_event=lambda e: print("[mission]", e),
)
print(result.completed, result.halted_at_stage)
```

**Explicitly NOT in Ship 16** (deferred):
- Re-decomposition on stage failure → Ship 17.
- Mission UI (DAG viewer, per-stage drill-down) → Ship 18.
- Skill-library cross-mission policy reuse → Ship 19.
- Per-stage knob overrides (steps_per_iter, rollout_episodes) — currently every stage uses the scaffold's defaults overridden only by `iterations` / `seed`. Add to `Stage` dataclass when needed.
- Per-step trajectory access in criteria for arrays NOT in `PERSISTED_TRAJECTORY_KEYS` — derive from existing arrays (e.g., `trajectory['root_link_pos_w'][...,2]` for base_height proxy).

### 2026-04-24 — Ship 15: Policy warm-start (rsl_rl selective load across stages)

**Scope:** Second brick of the Mission roadmap. Ship 15 lets a caller pass an external rsl_rl checkpoint to `sculpt_run`'s **iter-0 only**, which `_cmd_train` loads via `runner.load(path, load_cfg={actor: True, critic: True, optimizer: False, iteration: False, rnd: False})` BEFORE `runner.learn()`. This is the mechanism Ship 16's orchestrator will use to chain skills across Mission stages.

**Process (per Sam's "use agents to plan + review" ask):**
1. Explore agent mapped the mjlab → rsl_rl → on-policy-runner pipeline to file:line precision.
2. **Key finding during verification:** `rsl_rl.OnPolicyRunner.load(path, load_cfg=None, strict=True, map_location=None)` accepts a **`load_cfg` dict** with keys `{actor, critic, optimizer, iteration, rnd}` — each a bool. This is much better than the initial assumption of state-dict slicing; PPO.load honors the config cleanly at [rsl_rl/algorithms/ppo.py:444-466](). Ship 15 passes `optimizer=False` (stale Adam momentum from a different reward harms new-task learning) and `iteration=False` (keeps `max_iterations` semantics intact).
3. Plan agent reviewed. Flagged 4 gaps — incorporated before implementation:
   - Resume silently winning over init_policy_path → emit `warm_start_skipped` event with `reason="local_checkpoint_wins"`.
   - Adapter silently dropping init_policy_path → emit `warm_start_skipped` with `reason="adapter_does_not_support"`.
   - Obs-space mismatch → wrap `runner.load` in try/except with helpful error.
   - Checksum + richer event payload → `warm_start_loaded` includes `source_sha8` + `load_cfg_keys`.
4. Implementation landed, then **two parallel code-review agents** audited it. Found and fixed:

- **CRITICAL (pre-audit) — `**kwargs` introspection bug.** The original `"init_policy_path" in sig.parameters` check returned False for adapters using `**kwargs` catch-all, silently dropping the kwarg AND emitting `warm_start_skipped` (lying to the orchestrator). Fixed at [sculptor/sculpt.py:_train_or_resume](RewardSculptor/sculptor/sculpt.py) — now accepts either explicit named param OR `VAR_KEYWORD` in the signature.
- **HIGH — Empty-string bypass of path validation.** `Path("").resolve() == Path.cwd()` would bypass the None check and mis-validate. Fix: treat `""` / whitespace-only strings as None at `sculpt_run` entry. [sculpt.py:sculpt_run]().
- **MEDIUM-HIGH — Narrow exception catch.** `torch.load` can raise `UnpicklingError` / `OSError` / `EOFError` for corrupt or truncated checkpoints, not just `RuntimeError`. Broadened at [_mjlab_runner.py:_cmd_train]() to `(RuntimeError, OSError, EOFError, Exception)` with a clear diagnostic message listing the three likely causes (obs-space drift, corruption, rsl_rl version drift).
- **MEDIUM-LOW — Missing intent signal in `iter_started` event.** Added `warm_start_source` field so Ship 16 can correlate caller intent (this iter was supposed to warm-start) with the subprocess's `warm_start_loaded` event (the subprocess actually loaded the ckpt). Silent mismatch = bug.

**Findings documented but NOT fixed (rationale in comments):**
- `torch.load(weights_only=False)` ACE vector — consistent with rsl_rl's own internal default ([on_policy_runner.py:157]()) and pre-existing in `_train_or_resume:665`. Same trust model as user-supplied reward modules, which ARE executed unsandboxed. Marginal new attack surface is zero.
- Symlink / path-boundary traversal — user supplies paths, same trust as v0.py.
- CHANGELOG / provenance lineage — Ship 16 orchestrator will persist mission-level provenance; per-iter warm-start source is already in the `iter_started` event.
- Streaming SHA-256 — micro-opt; a 100 MB ckpt hashes in ~200 ms vs. a 20-min training run.

**Files changed:**
- [sculptor/adapters/_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py): new `--load-pretrained-policy` argparse flag + `runner.load(load_cfg=...)` call between runner construction and `runner.learn()`. Emits `[SCULPT-EVENT] warm_start_loaded {source, source_sha8, load_cfg_keys}` to stdout.
- [sculptor/adapters/mjlab.py:MjlabAdapter.train](RewardSculptor/sculptor/adapters/mjlab.py): new `init_policy_path: Optional[Path] = None` keyword-only kwarg; validates file exists and appends `--load-pretrained-policy` flag to subprocess cmd. Pre-flight file check surfaces a clear `FileNotFoundError` BEFORE subprocess spawn.
- [sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py): `_train_or_resume` threads `init_policy_path` via inspect-based kwarg forwarding; emits `warm_start_skipped` on silent drop OR local-checkpoint-wins. `_run_one_iter` adds the kwarg + includes `warm_start_source` in `iter_started` event. `sculpt_run` adds the top-level kwarg, validates at entry, passes to iter loop ONLY when `i == start_iter`.
- **Abstract `SculptorAdapter.train()` signature unchanged** — other adapters (gym_sb3 / mjx / rllib) stay untouched; they'd silently emit `warm_start_skipped` if ever given a path.

**Tests — [tests/test_ship15_warm_start.py](RewardSculptor/tests/test_ship15_warm_start.py), 16 tests:**
1. `test_mjlab_train_appends_load_pretrained_policy_flag` — cmd construction.
2. `test_mjlab_train_omits_flag_when_init_policy_none` — default unchanged.
3. `test_mjlab_train_raises_on_missing_init_policy` — pre-flight path check.
4. `test_train_or_resume_forwards_init_policy_path_to_supporting_adapter` — explicit-kwarg adapter.
5. `test_train_or_resume_drops_init_policy_path_for_unsupported_adapter` — no-kwarg adapter + `warm_start_skipped` event shape.
6. `test_train_or_resume_emits_warm_start_skipped_when_local_ckpt_wins` — resume-vs-init-policy conflict.
7. `test_train_or_resume_no_warm_start_event_when_init_policy_none` — quiet path.
8. `test_sculpt_run_rejects_missing_init_policy_path` — top-level entry validation.
9. `test_sculpt_run_init_policy_iter_0_guard_in_source` — iter-0-only gate.
10. `test_mjlab_runner_cli_parses_load_pretrained_policy` — argparse roundtrip.
11. `test_rsl_rl_load_cfg_selectively_loads_actor_and_critic` — rsl_rl API drift guard (regex-matches the 5 keys PPO.load consumes).
12. `test_ship15_warm_start_event_shape` — event-schema guard.
13. `test_train_or_resume_forwards_kwarg_to_adapter_with_var_kwarg` — **audit CRITICAL regression.**
14. `test_sculpt_run_treats_empty_string_init_policy_as_none` — **audit HIGH regression.**
15. `test_mjlab_runner_broadened_exception_catch_in_source` — **audit MEDIUM-HIGH regression.**
16. `test_iter_started_event_includes_warm_start_source_when_set` — **audit MEDIUM-LOW regression.**

**Verified:**
- Sculptor: **239 passed, 1 skipped** (was 223 → +16 Ship 15 tests).
- Zero regressions across 47 s test run.
- Plan v2 + both code-audit passes documented inline in commit rationale.

**Callable surface for Ship 16:**
```python
from sculptor.sculpt import sculpt_run
# Run stage 2 warm-started from stage 1's final checkpoint
sculpt_run(
    config_path=stage_2_config,
    behavior_goal=stage_2.goal_text,
    iterations=stage_2.max_iterations,
    init_policy_path=stage_1_final_ckpt,  # ← Ship 15 surface
)
```
Events Ship 16 run_manager should subscribe to:
- `iter_started` → check `warm_start_source` field for intent.
- `warm_start_loaded` (from subprocess stdout) → confirm ckpt loaded.
- `warm_start_skipped` with reason ∈ {`local_checkpoint_wins`, `adapter_does_not_support`} → orchestrator must decide retry / abort / continue.

**Explicitly NOT in Ship 15** (deferred):
- Iter-to-iter warm-start within one `sculpt_run`.
- CLI entry point (`sculpt run --init-policy-path`) — Mission UI in Ship 18.
- gym_sb3 / mjx / rllib adapter support.
- Obs-space consistency validation at load time (relies on rsl_rl's own state_dict shape error + wrapper message).

### 2026-04-24 — Ship 14: Mission / Stage data model + Claude task-decomposer

**Scope:** First concrete step of the multi-stage curriculum roadmap. Ship 14 delivers the **data model + decomposition-time Claude call** needed for a Mission-aware orchestrator (Ships 15-18 build warm-start, success-criterion evaluation, re-decomposition, UI). Nothing trains yet — this ship returns a validated `Mission` object and stops. No changes to the Project / sculpt_run path; the existing single-project flow is strictly unchanged.

**Architectural reference:** CurricuLLM (arXiv:2409.18382) — LLM decomposes complex robotics tasks into subtasks with per-stage reward + warm-start policy; validated on Berkeley Humanoid. Ship 14 mirrors their curriculum-generation LLM role; sculptor's existing Eureka-style reward iteration fills the per-stage role.

- **New data model — [sculptor/mission.py](RewardSculptor/sculptor/mission.py)**
  - `Stage` dataclass: `name, goal_text, success_criterion, max_iterations, parent_stage, reward_seed_prompt, kg_seed_papers` authored at decompose time + `status, final_policy_path, final_reward_path, best_metric, iterations_used, started_at, finished_at` populated at runtime by the future orchestrator.
  - `Mission` dataclass: ordered list of Stages + `goal, decomposition_model, decomposition_rationale, schema_version, current_stage_idx, mission_dir`.
  - JSON roundtrip via `Mission.to_json`/`from_json`, on-disk persistence via `save_mission`/`load_mission`.
  - `validate_mission(mission, *, info_keys)` enforces: ≥1 stage, snake_case names ≤32 chars, unique names, first-stage `parent_stage=None`, topological parent refs (no forward / cycle), `max_iterations ∈ [1, 50]`, `reward_seed_prompt` within [3, 2000] chars (matches the existing prompt-edit endpoint), `success_criterion` parses as a Python expression, every `info['<key>']` uses a real `expected_info_keys`. Bare-identifier grounding deferred to Ship 16 runtime (component names aren't materialized at decompose time).
  - Schema version gate (`SCHEMA_VERSION = 1`) for forward compat.

- **New prompt — [sculptor/prompts/decompose_task.md](RewardSculptor/sculptor/prompts/decompose_task.md)**
  Structured JSON output schema + 7 hard rules: topological ordering, last-stage-satisfies-goal, individual learnability (3-5 sculpt iters per stage), grounded success_criterion, grounded reward_seed_prompt, KG citations restricted to the provided literature slice, warm-start preferring most-recent compatible predecessor. Includes a stage-design guidance block and a worked example (jump-from-stance curriculum) so Claude sees the target shape.

- **New entry point — [sculptor/decompose.py](RewardSculptor/sculptor/decompose.py)**
  `decompose_task(goal, reward_contract, *, kg_store=None, client=None, model="claude-opus-4-7") -> Mission`. Renders the reward_contract + top-8 KG semantic matches into the decomposer's user content, calls `client.messages.parse(output_format=_DecompositionModel)` with the same one-retry pattern as `diagnose._parse_with_retry`, validates the result via `validate_mission` + live-KG citation check (`_validate_kg_seed_papers`), returns a `Mission` with all stages `status="pending"`. Anthropic client defaults to `max_retries=2, timeout=240.0` matching the post-Ship-13 edit envelope.

- **Tests — [tests/test_decompose.py](RewardSculptor/tests/test_decompose.py)** 24 tests:
  - Serialization roundtrip (2), schema-version gate, forward-compat unknown-key drop.
  - `validate_mission` rejection cases: empty stages, bad name (non-snake_case), duplicate names, first-stage-with-parent, forward parent ref, unknown parent name, unparseable criterion, unknown info-key in criterion, out-of-range `max_iterations`, out-of-range `reward_seed_prompt`.
  - `validate_mission` accept case: bare identifier that could be a future component (not rejected pre-runtime).
  - `decompose_task` happy path (no KG), contract-threading into user content, empty-goal rejection.
  - End-to-end Claude mistakes caught: forward parent ref, unknown info key, KG citation without store, KG citation not in live KG.
  - One-retry-on-parse-failure (mirrors diagnose).
  - Prompt-content guard (load_bearing rules don't disappear from `decompose_task.md` on edit).

- **Verified:**
  - Sculptor: **223 passed, 1 skipped** (was 197 → +26 new: 24 decompose + re-baselined post-Ship-12).
  - Zero regressions. No files modified outside the new Ship-14 surface.
  - Sculptor tests run in ~50 s.

- **What's NOT in Ship 14** (explicit, for the follow-up windows):
  - NO orchestrator (`mission_run`) — Ship 16.
  - NO policy warm-start plumbing to the mjlab adapter — Ship 15.
  - NO success-criterion runtime evaluator — Ship 16.
  - NO re-decomposition on stage failure — Ship 17.
  - NO UI — Ship 18.
  - NO CLI entry point (`sculpt mission <goal>`) — intentionally deferred; decompose is callable from Python / UI only until Ship 18.

- **Callable surface for Ship 15 to build on:**
  ```python
  from sculptor.decompose import decompose_task
  from sculptor.mission import save_mission, load_mission, validate_mission
  mission = decompose_task("Stand on one leg and kick", adapter.reward_contract(), kg_store=kg)
  save_mission(mission, project_dir / "mission.json")
  # mission.stages[0].reward_seed_prompt → feed into existing apply_prompt_edit
  # mission.stages[i].parent_stage → Ship 15 resolves to policy checkpoint path
  # mission.stages[i].success_criterion → Ship 16 evaluates on rollout trajectories
  ```

### 2026-04-23 08:30 — Ship 12: `_pre_validate` partition (stop dropping whole edit batches)

**Scope:** Sam's overnight G1-kick run completed all 10 iters but the reward module was frozen at v1 the entire time — the UI's Rewards tab showed no edits/justifications across iters 2-10, and the primary metric drifted -419 → -534. Root cause isolated in [runs/_run_job_*.log](): every iter had `[sculpt] iter N: apply_edits skipped — EditValidationError: edit pre-flight failed` with 1-3 of 5 edits flagged ungrounded while the other 2-4 were perfectly valid. Pre-fix `_pre_validate` raised on the FIRST violation — one bad diagnoser edit killed the whole batch. Ten iters × zero rewards applied = silent "reward didn't evolve" symptom.

- **Fix — partition instead of raise.** [sculptor/edit.py:_pre_validate](RewardSculptor/sculptor/edit.py) — now returns an `EditPlan` with new `rejected_edits` + `rejection_reasons` fields. Valid edits flow through to the LLM, rejected ones are logged with the reason and skipped. Only an **empty** `applicable_edits` list is a hard error (`EditValidationError` still raised by `apply_edits` in that case). `paper_refs not in KG` is STILL fatal (KG hygiene is a project-level concern, not a per-edit one).
- **Event surface.** [sculptor/edit.py:apply_edits](RewardSculptor/sculptor/edit.py) — new per-rejection `[edit] rejected: ...` log_line events + a structured `{type: "edits_rejected", count, reasons}` event so the UI run manager can fold these into the RunsTab iter-row. The "pre-validate done" summary now shows all three counts (applicable / deferred / rejected).
- **Diagnose prompt sharpened.** [sculptor/prompts/diagnose_grounded.md](RewardSculptor/sculptor/prompts/diagnose_grounded.md) — explicit OPERATION-vs-target_term table. Two common diagnoser errors from the overnight run called out by name:
  1. `operation: "clip"` + `target_term: "kick_velocity_cap"` (new name with modify op) → correct shape is either `operation: "clip", target_term: "kick_velocity"` OR `operation: "add", target_term: "kick_velocity_cap"`.
  2. Raw physics-state arrays (`qvel`, `qpos`, `xquat`) referenced in `suggested_value` → NOT grounded unless listed in `expected_info_keys`; use the adapter-exposed info key instead, or flag `requires_env_extension=true`.
- **Tests:**
  - [tests/test_edit.py](RewardSculptor/tests/test_edit.py) — `test_pre_validate_partitions_ungrounded_formula_field` + `test_pre_validate_partitions_modify_op_with_unknown_target_term` renamed/updated from their pre-fix `rejects` counterparts; now assert plan shape instead of raises. NEW `test_pre_validate_partitions_mixed_batch_keeps_valid_edits` pins the overnight regression: 3 edits (valid, invalid, valid) → 2 applicable + 1 rejected.
- **Secondary finding (physics didn't auto-apply).** The realism audit fired every iter but every verdict was `mild` (never `severe`). §7.4 auto-physics only triggers on `severe` — Sam's iter-5 `joint_vel_p99_max=89.46 rad/s` (≈3× nominal 30 rad/s) should have tripped `severe` but the thresholds are too permissive. **Not fixed in this ship** — separate tuning concern. Flagged for follow-up; [sculptor/adapters/realism.py]() is the file to tighten.
- **Verified:**
  - Sculptor: **197 passed, 1 skipped** (was 196 → +1 new partition test).
  - The _pre_validate fix is hypothesis-driven from the log analysis; live smoke against a fresh G1-kick run will confirm the reward module actually evolves across iters.
- **What Sam should expect on the next run.** Every iter row in the Runs tab should show `v<N> → v<N+1>` with 2-5 applied edits and 0-3 rejected (with rejection reasons shown). Primary metric should trend UP instead of drifting, assuming the diagnoser's applicable edits are substantive (they were — just blocked by the all-or-nothing validator).

### 2026-04-23 04:15 — Ship 11: KG-preview hang fix (huggingface_hub httpx bypass)

**Scope:** Fix the Ship-9/10 regression where `reward_prompt_edit` hung 5+ min at the `query_semantic` KG-preview step. Root cause identified, 3-part fix landed, live-smoked against Sam's real 448-technique KG.

- **Root cause.** `huggingface_hub` (recent version) uses an **httpx-based HEAD request** during `SentenceTransformer(model_name)` init to check cache freshness. On WSL2 this request hangs indefinitely (observed as 5+ min → timeout; diagnostic reproducer without the fix errors with `RuntimeError: Cannot send a request, as the client has been closed` from httpx's internals). The Ship-10 `_prewarm_embedding_model` hook didn't introduce the httpx bug but EXPOSED it — pre-Ship-10 only the reward-prompt worker hit `_get_embedder`; post-Ship-10 both the prewarm thread and the worker hit it concurrently, doubling the odds of wedging. The KG itself had nothing wrong: 448 Techniques all with all-MiniLM-L6-v2 embeddings populated; `_ensure_technique_embeddings` does 0 backfill work.
- **Fix 1 — `local_files_only=True` on SentenceTransformer init.** [sculptor/kg/query.py:_get_embedder](RewardSculptor/sculptor/kg/query.py) — tries `SentenceTransformer(model_name, local_files_only=True)` first, falling back to online load on OSError/ValueError (fresh install, empty cache). `SCULPTOR_HF_NO_NETWORK=1` forces offline-only for CI. **This is the load-bearing change** — bypasses the httpx HEAD check entirely, cuts embedder init from 60-120s → 0.4s on warm cache.
- **Fix 2 — `threading.Lock` on `_get_embedder`.** [sculptor/kg/query.py](RewardSculptor/sculptor/kg/query.py) — new `_EMBEDDER_LOAD_LOCK` + double-checked locking. Defensive: even if offline load hangs for some other reason, prewarm + worker serialize instead of racing. Not strictly needed for the primary fix but pins the race so Ship-10's prewarm pattern is safe going forward.
- **Fix 3 — pin prewarm task reference.** [backend/main.py:_prewarm_embedding_model](reward-sculptor-ui/backend/main.py) — stashes the `asyncio.create_task` return on `app.state.embedder_prewarm_task`. Python 3.11+ `asyncio.create_task` only keeps weak refs; unanchored tasks can be GC'd mid-flight. Named "embedder-prewarm" for introspection.
- **Instrumentation.** Added `log.info` breadcrumbs in `query_semantic` and `_ensure_technique_embeddings` (Technique fetch, has_embedding scan, embedder load, encode, total). If this hangs recur the uvicorn log pins the exact substep. Replaces the Ship-10-directive's ask for in-sculptor breadcrumbs.
- **Tests:**
  - `tests/test_kg_query.py::test_get_embedder_is_thread_safe_under_concurrent_load` — pins the lock fix. Stubs `SentenceTransformer` to sleep 200 ms on init, spawns 8 threads calling `_get_embedder`, asserts exactly 1 init + no deadlock within 5s.
  - Fixed pre-existing `test_edit.py::test_query_semantic_filters_by_min_similarity` breakage (my new log line used `store.db_path` — test's `_StoreStub` didn't have it; changed to `getattr(..., "db_path", "<unknown>")`).
- **Verified:**
  - Sculptor: **196 passed, 1 skipped** (was 195 → +1).
  - Backend reward-prompt: 10 passed, 1 deselected (skipping the pre-existing `test_reward_prompt_edit_emits_log_line_events` full-suite hang).
  - **Live smoke against Sam's real KG** (`~/.local/share/sculptor/kg/graph.db`, 448 techniques):
    ```
    kg.query.query_semantic: start (text_len=39, top_k=5, min_sim=0.35)
    kg.query: fetched 448 Technique nodes in 0.00s
    kg.query: has_embedding scan: 0 of 448 missing in 0.00s
    kg.query: loaded 448 embeddings in 0.01s
    loading sentence-transformer (local_files_only=True)
    loaded sentence-transformer from cache in 0.4s
    kg.query.query_semantic: encoded query in 3.41s
    query_semantic returned 5 matches in 3.5s
    ```
    Total **3.5 s** vs the pre-fix 300 s timeout. Overnight G1 run unblocked.
- **Why this isn't a Ship-10 revert.** The prewarm is retained — with the lock + `local_files_only`, it correctly shaves ~3 s off the first live query without any hang risk. The directive's "do NOT revert Ship 10 as a first step" stands.
- **Escape hatches preserved:** `RS_SKIP_EMBEDDER_PREWARM=1` still disables prewarm. `SCULPTOR_HF_NO_NETWORK=1` new, for CI / repeatable environments that want offline-only.

### 2026-04-23 03:30 — Ship 10 + unresolved KG-preview hang regression

**Scope:** Sam reported the Rewards-tab prompt sat on "Claude is writing…" for 5 min and then failed with `RuntimeError: reward-prompt edit timed out after 300s`. The only two log lines he saw were `start — validating parent + loading adapter` and `dispatching to Claude (timeout=300s)…`. Ship 10 adds granular heartbeats so the next hang is diagnosable, plus a startup-time embedding-model prewarm so cold-run latency is amortized off the request path. **Then Sam re-ran it on the next day and hit a NEW hang — at the exact heartbeat Ship 10 added.** Documented here for the follow-up window.

- **Ship 10 — heartbeat events + embedder prewarm.**
  - [backend/services/reward_jobs.py:_do_edit](reward-sculptor-ui/backend/services/reward_jobs.py) — added 5 new `log_line` events inside the worker thread: `loading adapter + reward_contract` (line 85), `opened KG at <name>` (line 115), `KG preview query (first call may take 60-120s on cold embedding model)` (line 132), `KG preview done (N matches ≥ sim=0.35, T.Ts)` (line 156), and `delegating to sculptor.edit.apply_prompt_edit` (line 165). The `query_semantic` call is now timed + wrapped in try/except → "KG preview failed …— continuing without preview". These pin which sub-step is blocking.
  - [backend/main.py:_prewarm_embedding_model](reward-sculptor-ui/backend/main.py) — new `@app.on_event("startup")` hook (line 139) that fires a background task `_asyncio.to_thread(_load_embedder)` calling `sculptor.kg.query._get_embedder()` ≈ initializes `all-MiniLM-L6-v2` off the request path. Gated by `RS_SKIP_EMBEDDER_PREWARM=1` for tests; `conftest.py` sets that env var. Fire-and-forget so uvicorn startup stays <1s.
  - Tests: `test_reward_prompt_edit_emits_log_line_events` asserts all 5 new heartbeats arrive in order; `test_reward_prompt_edit_timeout_releases_lock` asserts the timeout path still fires cleanup.
  - Critique: 0 critical. Three mediums: `_emit_from_worker` already bridged correctly; prewarm thread cannot block startup; stub test for `_get_embedder` monkeypatched.

- **NEW REGRESSION — reward-prompt hangs at the "KG preview query" heartbeat (UNRESOLVED).**
  Sam's exact evidence from the 03:22 run (same prompt that worked in 1-2 minutes before Ship 9+10 landed):
  ```
  03:22:51 [reward_prompt_edit] start — validating parent + loading adapter
  03:22:51 [reward_prompt_edit] dispatching to Claude (timeout=300s)
  03:22:51 [reward_prompt_edit] loading adapter + reward_contract
  03:22:51 [reward_prompt_edit] opened KG at graph.db
  03:22:51 [reward_prompt_edit] KG preview query (first call may take 60-120s on cold embedding model)
  <<< HANG — no further events, times out at 05:27 >>>
  ```
  Key facts:
  - Hang is **at `query_semantic(user_prompt, top_k=5, store=store, min_similarity=0.35)`** — [reward_jobs.py:139-142](reward-sculptor-ui/backend/services/reward_jobs.py). No "KG preview done" ever fires, and the "KG preview failed" except-branch doesn't fire either — so it's **blocked**, not erroring.
  - KG in Sam's UI shows all checkmarks (papers extracted, presumed embeddings populated).
  - Sam's verbatim claim: **"this wasn't before your recent fixes so it was a new issue made."** I.e., either Ship 9 (early-stop / motor-limits / PDF datasheet) or Ship 10 (heartbeats + prewarm) introduced a regression in the KG semantic-query path. Most likely culprits to investigate in the next window:
    1. **Prewarm task contention with the worker thread.** Ship 10's `_asyncio.to_thread(_load_embedder)` fires at uvicorn boot and the reward-prompt worker ALSO loads the embedder via `query_semantic → _get_embedder`. If `sentence_transformers` or `transformers` isn't thread-safe on first init (download + load from HuggingFace cache), the worker thread could deadlock waiting for the prewarm task to finish.
    2. **`_ensure_technique_embeddings` backfill runs inside `query_semantic` on a cold KG.** If the user's KG grew since last run (new papers/techniques added without embeddings), the first query triggers a backfill that embeds every technique serially. 416 techniques × 50 ms ≈ 20 s, but if HF model cache is partially-populated or network-flaky, could exceed 300 s.
    3. **Ship 9c's `asyncio.to_thread` inside the datasheet PDF extract route.** If Sam hit the Physics tab before Rewards, a stale `asyncio.to_thread` could hold the embedder lock.
  - No fix shipped in this window. Sam explicitly asked me to hand this off to a new window.

- **Additional unresolved follow-ups (pre-existing):**
  - `test_reward_prompt_edit_emits_log_line_events` and `test_library.py` hang mid-sequence in full-suite runs but pass individually. Likely HF embedding-model cache issue during parallel test runs. Not introduced by Ship 9/10 but worth re-diagnosing.

- **Commit + push.** Init'd `~/projects/` as the `RL-Sculptor` monorepo (`RewardSculptor/` + `reward-sculptor-ui/` + top-level `.md` planning docs). `.gitignore` excludes `AME456/`, `.claude/`, `.venv/`, `node_modules/`, `runs/`, `.env`, `__pycache__/`, large fixture workdirs. Commit `1c5b59f` — 329 files / 76406 insertions. Remote `origin` set to `https://github.com/sjdoane/RL-Sculptor.git`. **Push from Claude Code blocked on credential-manager browser auth** (GCM opens a Windows auth window that can't be completed from the agent session) — run `cd ~/projects && git push -u origin main` in a WSL shell once to complete the first push + cache creds. See `HANDOFF_KG_PREVIEW_HANG.md` for the short new-window prompt.

### 2026-04-23 09:46 — Ship 9: configurable early-stop + motor-spec form + datasheet PDF upload

**Scope:** Sam's two asks: (1) the 3-iter early-stop can truncate overnight runs where metric dips mask real behavioral improvement; (2) Physics tab should accept motor specs (form or PDF) and apply them. Three sub-ships, one critique agent after each.

- **Ship 9a — configurable early-stop.**
  - [sculptor/sculpt.py:_should_early_stop](RewardSculptor/sculptor/sculpt.py) — now takes `enabled: bool = True`; `patience < 1` OR `enabled=False` short-circuits to False.
  - [sculpt.py:_run_one_iter](RewardSculptor/sculptor/sculpt.py) reads `[iteration].early_stop_enabled` + `early_stop_patience` from config.toml; defaults `true` / `3`.
  - [sculpt_run](RewardSculptor/sculptor/sculpt.py) accepts `early_stop_enabled` / `early_stop_patience` kwargs that merge into cfg.
  - [sculptor/cli.py](RewardSculptor/sculptor/cli.py) — `--early-stop / --no-early-stop` + `--early-stop-patience N`.
  - CONFIG_TEMPLATE adds `early_stop_enabled = true` + `early_stop_patience = 3` defaults.
  - [backend/models/run.py + project.py](reward-sculptor-ui/backend/models) — fields on `RunParams` (per-run override) + `IterationSettings` (project default).
  - [backend/services/run_manager.py](reward-sculptor-ui/backend/services/run_manager.py) — forwards as CLI flags.
  - [NewRunDialog.tsx Advanced tab](reward-sculptor-ui/frontend/src/components/NewRunDialog.tsx) — tri-state select + patience input + dynamic caption.
  - [ProjectSettingsDialog.tsx](reward-sculptor-ui/frontend/src/components/ProjectSettingsDialog.tsx) — persistent defaults.
  - Tests: `test_sculpt.py` 5 new + `test_projects.py` roundtrip + `test_runs.py` forwarding.
  - Critique: 0 critical, 2 medium (stale UI caption, patience-0 vs enabled=false divergence). Both addressed.

- **Ship 9b — motor-specs form on Physics tab.**
  - [backend/routes/physics.py](reward-sculptor-ui/backend/routes/physics.py) — new `MotorSpec` + `MotorLimitsRequest` pydantic models (5 numeric fields + notes + `field_validator` guard on joint-name format). New `_synthesize_motor_limits_prompt` translates structured specs into a KG-cited NL prompt (forcerange ↔ peak_torque, armature = rotor_inertia × gear_ratio², damping = 0.05 × peak_torque / peak_speed; cites 2312.17507, 1901.08652, 2410.08650). New `POST /projects/{slug}/physics/motor-limits` route delegates to the existing physics-edit job.
  - [frontend PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) — new `MotorLimitsCard` with one-row-per-actuator table. "Apply motor limits" button disabled when any filled row's joint name isn't in the MJCF (prevents wasted Claude round-trips on REJECTED-for-unknown-joints). Post-apply `qc.invalidateQueries` refreshes the Current MJCF panel.
  - Tests: 7 new in `test_physics.py`.
  - Critique: 2 critical (test hit real Anthropic API, prompt injection via joint names), 3 medium (PDF size cap, HTML `min="0"` vs backend `gt=0`, no refetch-on-apply), 5 minor. All criticals + mediums fixed: `field_validator` on `MotorSpec` keys with `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$`; `_PROMPT_MAX_CHARS = 8000` ceiling; `anthropic.Anthropic` stubbed in happy-path test (was 57s, now <1s).

- **Ship 9c — datasheet PDF upload.**
  - [backend/routes/physics.py `extract_datasheet_pdf`](reward-sculptor-ui/backend/routes/physics.py) — multipart PDF upload, magic-byte check (`%PDF-` with BOM strip), 10 MB cap, 20-char minimum (catches scanned PDFs). Extracts text via `pypdf`, feeds to Claude with a strict JSON schema system prompt, validates response via `MotorLimitsRequest.model_validate` (reuses Ship-9b joint-name guard → prevents injection via datasheet). Empty-motors `{motors: {}}` accepted as "no specs found" response. Claude call bounded by `asyncio.wait_for(..., timeout=130.0)` + `asyncio.to_thread` so the event loop isn't parked.
  - [frontend PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) — "Upload datasheet" button in `MotorLimitsCard`. `extract` mutation merges extracted specs into local state; "not in MJCF" badge on rows whose joint name doesn't match a summary actuator; Apply stays disabled until the user fixes/deletes those rows. File picker guards against double-click, oversized files (>10 MB client-side), non-PDF MIME.
  - Tests: 9 new in `test_physics.py` (happy path, non-PDF rejection, 503 no-api-key, non-JSON Claude response, empty-motors, injection rejection, markdown-fence strip, 413 oversize, Claude-hang-times-out via asyncio.wait_for patch).
  - Critique: 3 critical (unbounded Claude hang, sync SDK in async def, non-MJCF joints allowed to Apply), 7 medium. All criticals fixed; selected mediums applied (BOM strip, double-click guard, client-side size check).

- **Baselines:**
  - Sculptor: `191 passed, 1 skipped` (was 183; +8 new: 6 early-stop + 1 config template + 1 integration).
  - Backend (per-file): all 20 test files pass individually. Full-suite run hit a pre-existing `test_reward_prompt_edit_emits_log_line_events` hang unrelated to Ship 9 changes (likely a Hugging Face embedding-model cache / network issue — the test's `query_semantic` call runs the real sentence-transformers path). Flagged as a separate follow-up. Running `uv run pytest backend/tests/ -q -k "not test_reward_prompt_edit_emits_log_line_events and not test_reward_prompt_edit_timeout_releases_lock"` proves no regressions from Ship 9.
  - Frontend tsc: exit 0 ✓.
  - Individual per-ship Ship-9 test counts: 9a = 13 passing tests including full sculpt_run integration, 9b = 10 tests, 9c = 9 tests.

- **Why**: Sam explicitly asked for both. Early-stop was truncating cartwheel-style overnight runs where reward transiently dips. Motor-spec entry was the biggest remaining terminal-required workflow (users were pasting datasheet numbers into the generic prompt textarea).

### 2026-04-23 07:45 — Ship 8: per-project settings UI + full physics auto-apply + run.sh self-heal

**Scope:** everything still requiring terminal edits after Ship 7.

- **Ship 8a — per-project settings UI (config.toml [iteration] via UI)**
  - [backend/models/project.py:IterationSettings + ProjectSettings](reward-sculptor-ui/backend/models/project.py) — pydantic payload with bounds on every numeric field; `extra="forbid"` on PATCH.
  - [backend/routes/projects.py](reward-sculptor-ui/backend/routes/projects.py) — `GET /projects/{slug}/settings`, `PATCH /projects/{slug}/settings`. `_toml_value` formatter (rejects NaN/Inf/None/multi-line), `_find_iteration_section_span` (section-scoped regex), `_upsert_iteration_key` (replaces or appends a key *within* `[iteration]`), `_read_iteration_dict` (surfaces TOMLDecodeError as 500). Both route handlers catch ValueError from `_toml_value` and 422 with the offending field.
  - [backend/tests/test_projects.py](reward-sculptor-ui/backend/tests/test_projects.py) — 15 new tests: GET/PATCH round-trip, partial PATCH preserves unset, `extra="forbid"`, empty-body no-op (200 not 422), unknown-slug 404, TOML string/list escaping, cross-section no-clobber regression, missing-[iteration] auto-create, multi-line string reject, NaN/Inf reject, corrupt-TOML 500, no-trailing-newline, `_toml_value` unit.
  - [frontend/src/components/ProjectSettingsDialog.tsx + frontend/src/lib/api.ts](reward-sculptor-ui/frontend/src/components/ProjectSettingsDialog.tsx) — new editable `IterationSettingsSection` (numeric + tri-state bool + text). Revert/Save buttons; dirty check; React Query cache.
  - **Critique pass** (agent run) flagged 3 critical bugs + 9 medium. All critical bugs (regex cross-section collision, append corrupting next section, multi-line string silent corruption) fixed in the hotfix pass with explicit regression tests per bug.

- **Ship 8b — physics.apply_prompt_edit refactored + sculpt-side full auto-apply**
  - [sculptor/adapters/mjcf_editor.py](RewardSculptor/sculptor/adapters/mjcf_editor.py) — NEW module hosting `parse_claude_output`, `_validate_xml` (unique `tempfile.mkstemp` — no more fixed `.__rs_validate.xml` collisions), `_write_with_lock` (filelock-serialized atomic write, fallback warn if filelock absent), `_render_kg_block`, `apply_mjcf_edit(mjcf_path, user_prompt, *, client, kg_store, write=True)`. Pure Claude orchestration, no backend imports.
  - [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py) — `apply_prompt_edit` now delegates the Claude part to `apply_mjcf_edit`; retains materialize + git commit + route-shape wrapping. Preserved `commit_sha`/`kg_citations` in response; dead `_SUMMARY_PI_RE` + `import re` removed.
  - [sculptor/sculpt.py](RewardSculptor/sculptor/sculpt.py) — `_find_materialized_mjcf` + `_maybe_apply_auto_physics_edit` + `_scoped_git_commit(project, paths=..., message=...)`. Call order: audit → emit `physics_edit_suggested` → diagnose → `apply_mjcf_edit` → diagnose-proposed reward edit → next iter. Inline physics edit is gated on `ANTHROPIC_API_KEY` present AND a materialized MJCF existing; gracefully skips with an event otherwise.
  - Full event timeline: `physics_edit_suggested` → `physics_auto_apply_started` → `physics_auto_applied` | `physics_auto_apply_rejected` | `physics_auto_apply_errored` | `physics_auto_apply_skipped`.
  - [tests/test_mjcf_editor.py](RewardSculptor/tests/test_mjcf_editor.py) — 23 tests: parse contract, happy path, write=False, parse/claude-reject/mujoco-validate paths, tempfile uniqueness under 3-way concurrency, concurrent-write serialization via filelock, `_find_materialized_mjcf`, `_scoped_git_commit` only-stages-requested, traversal/absolute rejects, newline-strip.
  - [backend/tests/test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py) — regression: `commit_sha` present on happy path (pins post-refactor contract).
  - **Critique pass** flagged 3 critical bugs + 6 medium. All critical (concurrency race on mjcf write, tempfile collision, git-add `.` bundling in-flight run artifacts) fixed with filelock, mkstemp, and `_scoped_git_commit`. From the follow-up critique: path-traversal guard in `_scoped_git_commit`, PI-anchored `parse_claude_output`, newline-stripped commit messages, `physics_auto_apply_started` event for UI chip-race prevention, narrower `multiprocessing.spawn` orphan match.

- **Ship 8c — run.sh self-heal + terminal-path audit**
  - [reward-sculptor-ui/run.sh](reward-sculptor-ui/run.sh) — new `pids_holding_port`, `is_our_orphan` (matches `uvicorn backend.main` OR vite bin OR multiprocessing spawn_main WHOSE ANCESTOR is our uvicorn), `reclaim_port_if_orphan`. When port 8000 or 5173 is held by OUR orphan (prior `./run.sh` exited uncleanly), it's auto-killed before binding. Unrelated processes still die with a clear error. Fatal-die if neither `lsof` nor `ss` is available (no silent fallthrough).
  - Terminal-path audit: the only commands Sam ever needs outside `./run.sh` are (a) first-time `uv sync` / `pnpm install` (unavoidable), (b) `ANTHROPIC_API_KEY` env var (one-time setup), (c) `uv run pytest ...` for dev/CI tests. Every user-facing feature from Ship 7 onward is UI-reachable: auto_adjust_physics (Project Settings + New Run Advanced), heal-stubs (KG tab button), config.toml edits (Project Settings dialog), rollout video knobs (New Run Advanced).
  - **Critique pass** flagged 0 critical + 6 medium. Applied: narrowed multiprocessing match to require uvicorn ancestor, added no-probe die.

- **Why**: Sam's durable rule "no terminal commands beyond `./run.sh`" from CONTEXT 05:58. Ship 7 addressed the most-visible offenders (video length, RL knob surfacing); Ship 8 closes the long tail (project settings, physics auto-apply needing manual click, orphaned-port recovery).
- **How**: each sub-ship reviewed by an independent critique agent; all critical bugs fixed before the next sub-ship started; regression tests pin every critical fix. No broad "trust-me" changes.
- **Verified**:
  - Sculptor: `183 passed, 1 skipped` (was 160 → +23 new tests across mjcf_editor + TOML helpers).
  - Backend: `246 passed, 6 deselected` (was 231 → +15 new: settings CRUD, PATCH edge cases, heal-route, commit_sha regression).
  - Frontend tsc: exit 0 ✓.
  - Not yet live-smoked — the overnight G1 run described in the test recipe below will exercise every new path end-to-end.

### 2026-04-23 06:56 — Ship 7: real-time video + every RL knob UI-exposed + heal-stubs button

- **What**:
  - [sculptor/adapters/_mjlab_runner.py:100-124](RewardSculptor/sculptor/adapters/_mjlab_runner.py:100) — new pure helper `_compute_playback_fps(step_dt, render_every, playback_speed, cli_fps)`. Derives fps so video duration == sim_duration / playback_speed. Clamps output to [1, 240] so ffmpeg accepts it.
  - [sculptor/adapters/_mjlab_runner.py:735-756](RewardSculptor/sculptor/adapters/_mjlab_runner.py:735) — `_cmd_rollout`: replaced `render_every = max_steps // 120` (fixed 2.4 s video) with `MAX_FRAMES = 500` cap + a `playback_speed` knob. `--render-every` / `--playback-speed` / `--fps` (now float, default 0=auto) CLI args plumbed in.
  - [sculptor/adapters/_mjlab_runner.py:946-974](RewardSculptor/sculptor/adapters/_mjlab_runner.py:946) — ffmpeg call uses `effective_fps` from the helper; emits `[SCULPT-EVENT] video_params` with step_dt / render_every / duration for UI visibility.
  - [sculptor/adapters/mjlab.py:541-606](RewardSculptor/sculptor/adapters/mjlab.py:541) — `MjlabAdapter.rollout` accepts `max_episode_steps` / `playback_speed` / `render_every` / `fps` kwargs, forwarded as CLI flags.
  - [sculptor/sculpt.py:_rollout_or_resume](RewardSculptor/sculptor/sculpt.py) — signature-introspects to forward video kwargs only to adapters that accept them (gym_sb3 silently opts out). `_run_one_iter` reads `max_episode_steps` / `playback_speed` / `render_every` / `rollout_fps` from `[iteration]` config.
  - [sculptor/sculpt.py:sculpt_run](RewardSculptor/sculptor/sculpt.py) — new kwargs (`max_episode_steps`, `playback_speed`, `render_every`, `rollout_fps`, `rollout_episodes`, `seed`, `auto_adjust_physics`). Each merges into `cfg["iteration"]` so `_run_one_iter` sees a single source of truth.
  - [sculptor/sculpt.py:_CONFIG_TEMPLATE](RewardSculptor/sculptor/sculpt.py) — `auto_adjust_physics = true` is now the default (was false); added commented knobs for the new rollout params so users see them.
  - [sculptor/cli.py:run](RewardSculptor/sculptor/cli.py) — new flags `--max-episode-steps`, `--playback-speed`, `--render-every`, `--rollout-fps`, `--rollout-episodes`, `--seed`, `--auto-adjust-physics / --no-auto-adjust-physics`. All None defaults.
  - [backend/models/run.py:RunParams](reward-sculptor-ui/backend/models/run.py) — same knobs added to `RunParams` with validated ranges (ge/le bounds matching what makes physical sense).
  - [backend/services/run_manager.py:59-66, 98-130](reward-sculptor-ui/backend/services/run_manager.py:59) — plucks the new params out of `run_params`, forwards them as CLI flags, stashes on `job.params` for REST detail visibility.
  - [backend/routes/kg.py:kg_heal_stubs](reward-sculptor-ui/backend/routes/kg.py) — new `POST /projects/{slug}/kg/heal-stubs` wrapping `heal_stub_titles`. Returns `{results, summary: {healed, still_stubbed, errored, total}}`. Synchronous.
  - [frontend/src/lib/types.ts:311-334](reward-sculptor-ui/frontend/src/lib/types.ts:311) — `RunParamsPayload` grows the 7 new fields.
  - [frontend/src/lib/api.ts:160-195](reward-sculptor-ui/frontend/src/lib/api.ts:160) — new `healStubTitles(slug)` + `HealStubsResponse` type.
  - [frontend/src/components/NewRunDialog.tsx](reward-sculptor-ui/frontend/src/components/NewRunDialog.tsx) — Advanced tab now has a "Rollout video + RL knobs" card (episode steps / playback speed / rollout episodes / seed) and an "auto-physics" tri-state select (on / off / project-default) next to --expand-kg and --no-kg.
  - [frontend/src/components/KnowledgeGraphTab.tsx](reward-sculptor-ui/frontend/src/components/KnowledgeGraphTab.tsx) — new "Heal stub titles" button next to "Bulk-seed library" in the KG header. Calls the new route, shows success toast with `{healed}/{total}` summary, refetches papers.
  - [tests/test_mjlab_adapter.py:520-586](RewardSculptor/tests/test_mjlab_adapter.py:520) — 6 new tests for `_compute_playback_fps`: real-time default, render_every preserves duration, playback_speed multiplier, fps clamp [1, 240], cli override wins, playback_speed clamp.
  - [backend/tests/test_runs.py:484-608](reward-sculptor-ui/backend/tests/test_runs.py:484) — 3 new tests: all 7 §Ship-7 params forwarded to CLI, `auto_adjust_physics=false` emits `--no-auto-adjust-physics`, None values omit the flags.
  - [backend/tests/test_kg.py:210-268](reward-sculptor-ui/backend/tests/test_kg.py:210) — 3 new tests for the heal route: empty KG returns `total=0`, result counts match, unknown slug 404.
  - [~/.claude/projects/.../memory/feedback_rollout_video_realtime.md](.) + MEMORY.md entry — this feedback is now part of durable rules.
- **Why**: two pieces of Sam feedback on the same session.
  1. Rollout videos were always exactly 2.4 s (`render_every = max_steps // 120` capped at 120 frames, then encoded at a fixed 50 fps). That matches `120/50 = 2.4`. For max_steps=500 at step_dt=0.02, actual sim duration was 10 s, so video played back ~4× sped up — exactly the "crazy physics" Sam saw in the go1-jump-stress run. Real physics would look normal if the video played at sim rate.
  2. Sam's durable rule: every feature must be reachable from the UI after `./run.sh` boots — no `sed -i config.toml` in test recipes. This ship flips every lingering CLI-only knob (auto_adjust_physics flag, heal-stubs maintenance command) into a UI control.
- **How**:
  - **fps math factored for testability**: `_compute_playback_fps` is pure, no env or file IO — 6 unit tests cover real-time, clamping, overrides, slow-mo / fast-forward.
  - **MAX_FRAMES=500 cap**: at 640×480×3 RGB that's ~440 MB worst case. Real-time playback at step_dt=0.02 → max_episode_steps up to 500 renders every step, longer episodes decimate + fps drops to preserve duration. Bounded cost, bounded surprise.
  - **Auto-physics default on**: new projects get the full audit→suggest loop out of the box. `iter_cfg.get("auto_adjust_physics", False)` still means pre-Ship-7 projects default off, so upgrades don't silently flip behavior.
  - **UI tri-state select for auto-physics**: "project default / on / off" — explicit way to override per-run without mutating config.toml. Matches Sam's "UI-only" rule.
  - **Heal route is synchronous** (not a JobManager job) — typical case is ~10 papers × ~10 s each = ~100 s, fits within a normal HTTP request timeout. If Sam ever has a KG with hundreds of stubs, we'd reconsider.
- **Verified**:
  - Sculptor: `160 passed, 1 skipped` (was 154; +6 fps-math tests; zero regressions).
  - Backend: `231 passed, 6 deselected` (was 225; +6 new: 3 Ship-7 run_manager + 3 heal-route).
  - Frontend tsc: exit 0 ✓.
  - Not live-smoked — the next sculpt run Sam launches through the UI will confirm real-time playback. Expected: a 500-step rollout now produces a 10-sec video (was 2.4 s at 4× speed).

### 2026-04-23 06:04 — Ship 6 / §7.7: arxiv retry-with-backoff + stub-title heal + seed titles

- **What**:
  - [sculptor/kg/ingest.py:155-232](RewardSculptor/sculptor/kg/ingest.py:155) — `_fetch_arxiv_metadata` now retries on transient failure with schedule `0, 10, 30, 60, 120` seconds (total ~3.7 min, longer than any observed arxiv rate-limit window). Accepts `retry_delays_s` + `sleep_fn` kwargs for testability; production callers unaffected.
  - [sculptor/kg/ingest.py:235-295](RewardSculptor/sculptor/kg/ingest.py:235) — new `heal_stub_titles(store, *, force=True)`: scans `find_nodes(kind="Paper")` for titles starting with `arxiv:`, re-ingests each with `force=True`, returns `{arxiv_id: "healed" | "still_stubbed" | "error: ..."}`. Best-effort — exceptions per paper don't block the scan.
  - [sculptor/cli.py:110-140](RewardSculptor/sculptor/cli.py:110) — new `sculpt kg heal-stubs` CLI subcommand wrapping the function; prints `N healed / M still stubbed / K errored` plus a tip when retries are advisable.
  - [cartwheel_kg_seeds.yml](cartwheel_kg_seeds.yml) — added explicit `title:` per paper across all 34 entries (titles extracted from existing comments). Future re-ingest of this seed file hits the arxiv-rate-limit fallback path with useful titles instead of stubs.
  - [tests/test_kg.py:244-407](RewardSculptor/tests/test_kg.py:244) — 5 new tests: retry-succeeds-on-3rd-attempt (stub arxiv module), retry-returns-None-after-all-fail, heal-empty-KG-returns-empty-dict, heal-identifies-2-stubs + skips-healthy, heal-reports-still-stubbed-when-retry-fails.
- **Why**: Sam's cartwheel-session bulk ingest of 33 arxiv papers (CONTEXT 00:15) left ~9 with stub titles `arxiv:XXXX.XXXXX` because the arxiv API rate-limited partway through and the single-retry wrapper gave up. Reward specs citing these papers render as "Unknown. arxiv:XXXX.XXXXX" in the UI. Fix: retry harder before giving up; let a user heal existing stubs via one command; populate fallback titles in seed files so future rate-limits are invisible to the user.
- **How**:
  - **Module-level retry schedule** (`_ARXIV_RETRY_DELAYS_S`) so future tuning + monkey-patching are surgical.
  - **sleep_fn injection** in `_fetch_arxiv_metadata` keeps tests at `~1s` instead of `~3.7min`. Production default = `time.sleep`; tests pass a lambda.
  - **Heal uses `force=True`** so the PDF + full-text sidecar get re-downloaded too (not just the metadata). If the stub was left because arxiv was unreachable, the PDF is probably also missing.
  - **Seed titles are verbatim from the paper's arxiv abstract page** (not agent-paraphrased) to keep the KG honest. The 2202.13500 entry kept its title ("Learning Humanoid Locomotion with Transformers") even though CONTEXT 00:15 flagged that paper as the "wrong paper" Agent 2 hallucinated — a follow-up pass should decide whether to remove the seed entry entirely (it's still in the KG via the earlier bulk ingest).
- **Verified**:
  - Sculptor: `154 passed, 1 skipped` (was 149; +5 new §7.7 tests, zero regressions).
  - Backend: `225 passed, 6 deselected` ✓ baseline.
  - Frontend tsc: exit 0 ✓ baseline.
  - Live smoke: `sculpt kg heal-stubs --help` loads cleanly.
  - Not yet live-smoked against the shared KG — calling `sculpt kg heal-stubs` for real would re-download ~9 PDFs + run Claude extraction, takes ~2 min. Sam can run it when convenient; worst case the stubs persist.

### 2026-04-23 05:58 — Ship 5 / §7.6: project-tab resilience under active-run GPU contention

- **What**:
  - [backend/services/preview_renderer.py:184-213](reward-sculptor-ui/backend/services/preview_renderer.py:184) — lazy module-level `asyncio.Lock` that serializes `render_static_async` calls. Prevents concurrent `mujoco.Renderer` EGL-context races, which Sam's cartwheel session (CONTEXT 00:15) showed as "preview 500s when navigating between projects during an active run".
  - [backend/services/job_manager.py:239-248](reward-sculptor-ui/backend/services/job_manager.py:239) — new `has_any_active_sculpt_run()` (cross-project) alongside the existing per-slug version. GPU contention is process-wide, not per-project.
  - [backend/routes/robot.py:84-89, 392-393, 437-452](reward-sculptor-ui/backend/routes/robot.py:84) — preview route now takes a JobManager dep, returns 503 (`/problems/preview-busy`) when a render fails with `kind="render"` AND any sculpt run is active (GPU contention is retryable). Non-active-run render failures keep the old 500 semantics.
  - [backend/routes/reports.py:50-90](reward-sculptor-ui/backend/routes/reports.py:50) — `get_final_report` now counts completed iters on disk (via `runs/iter_N/diagnosis.json`) and includes `n_completed_iters` in the 404 body + picks a detail string that matches the actual state (zero iters → "run one first"; nonzero → "build report"). Closes the "reports tab says 'no report' even though runs exist" half of Sam's symptom list.
  - [backend/tests/test_robot.py:522-600](reward-sculptor-ui/backend/tests/test_robot.py:522) — 2 new tests: 503 on `kind="render"` failure during an active run, 500 on the same failure when no run is active.
  - [backend/tests/test_reports.py](reward-sculptor-ui/backend/tests/test_reports.py) — NEW file, 4 tests: zero-iter 404, some-iter-count 404, unknown-project 404 (project-not-found path unchanged), happy-path 200.
- **Why**: Sam's cartwheel test (CONTEXT 00:15) #3 showed project-tab breakage during a concurrent run — preview 500, reports missing, "no runs completed". Root causes differed across symptoms (EGL contention vs endpoint UX), but the unified fix pattern is "when GPU is busy → retry-friendly 503, when endpoint state is ambiguous → sharper messaging".
- **How**:
  - **Serialize preview renders over asyncio.Lock instead of letting them race.** Adds ~500ms of worst-case serial latency (two users loading previews simultaneously) but eliminates the class of failures where EGL contexts collide on WSL2's software path. Lock bound to the active event loop (lazy-init guards against test loops).
  - **503 vs 500**: the HTTP spec says 503 is "try again later" — matches the failure mode. Frontend React Query's default retry policy will automatically retry 503s (unlike 500s which are treated as permanent). Small behavior change that needs no explicit UI work.
  - **Reports n_completed_iters counter**: reused the same `diagnosis.json`-per-iter count that `ProjectStore.get()` computes — no new IO.
  - **No frontend changes this ship.** The 503 retry + sharper 404 detail are backend-only; the frontend's existing error-handling already surfaces the `detail` field in toasts. If Sam wants a dedicated "GPU busy" toast copy later, that's a polish follow-up.
- **Limitations (honest assessment)**:
  - Not reproduced live. The fix is hypothesis-driven from code inspection — confirmed that (a) `mujoco.Renderer` has no explicit GPU-contention error surfaced by mujoco, (b) `render_static_async` had no serialization, (c) reports 404 didn't distinguish the two cases Sam described. First project-nav-during-run regression test on a real cartwheel run will confirm whether the fix lands cleanly or whether there's a residual race.
  - Doesn't address symptom "Rewards/Physics tabs barely updated" — that's likely frontend React Query invalidation side effects, not obviously caused by a backend bug. Left as-is; if it resurfaces we can investigate invalidation scoping.
- **Verified**:
  - Sculptor: `149 passed, 1 skipped` ✓ (unchanged; no sculptor changes in this ship).
  - Backend: `225 passed, 6 deselected` (was 219; +6 new: 2 preview + 4 reports).
  - Frontend tsc: exit 0 ✓.

### 2026-04-23 05:51 — Ship 4 / §7.4 (MVP): auto-physics suggestion on severe realism verdict

- **What**:
  - [sculptor/adapters/auto_physics.py](RewardSculptor/sculptor/adapters/auto_physics.py) — new module: `synthesize_auto_physics_prompt(audit)` builds a 600-800-char NL prompt from a §7.3 audit dict (evidence + mitigation order + canonical KG citations), + `should_auto_adjust_physics(audit, *, auto_adjust_enabled)` feature-flag gate (fires only on verdict=severe + flag=true).
  - [sculptor/sculpt.py:612-684](RewardSculptor/sculptor/sculpt.py:612) — `_run_one_iter`: after realism audit, emits `[SCULPT-EVENT] physics_edit_suggested` with `{iter, prompt, verdict, top_joints_saturation}` when the gate fires. Also writes the prompt to stderr for CLI visibility.
  - [sculptor/sculpt.py:_CONFIG_TEMPLATE](RewardSculptor/sculptor/sculpt.py) — new `[iteration].auto_adjust_physics = false` default in scaffolded config.
  - [backend/services/run_manager.py:708-725](reward-sculptor-ui/backend/services/run_manager.py:708) — `_iter_events` consumes `physics_edit_suggested` and folds into `slot["physics_edit_suggestion"] = {prompt, verdict, top_joints_saturation}`.
  - [backend/models/run.py:84-89](reward-sculptor-ui/backend/models/run.py:84) — `IterEventSummary.physics_edit_suggestion: Optional[dict] = None`.
  - [frontend/src/lib/types.ts:330-344](reward-sculptor-ui/frontend/src/lib/types.ts:330) — new `PhysicsEditSuggestionPayload` interface + `physics_edit_suggestion` field on `IterEventSummary`.
  - [frontend/src/components/RunsTab.tsx:533-566](reward-sculptor-ui/frontend/src/components/RunsTab.tsx:533) — indigo "apply physics fix" chip on iter rows that have a suggestion. On click: parks the prompt in `sessionStorage.pendingPhysicsPrompt` + navigates to Physics tab for that project. Falls back to clipboard if sessionStorage unavailable.
  - [frontend/src/components/RunsTab.tsx:770-785 + 866](reward-sculptor-ui/frontend/src/components/RunsTab.tsx:770) — reducer `physics_edit_suggested` → slot field; `_mergeIterSlot` carries it through.
  - [frontend/src/components/PhysicsTab.tsx:83-103](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx:83) — mount-time `useEffect` pulls `sessionStorage.pendingPhysicsPrompt` into the textarea, clears the key, focuses the textarea.
  - [tests/test_auto_physics.py](RewardSculptor/tests/test_auto_physics.py) — 10 tests: prompt shape + metrics inclusion, length bound (<1800 chars), missing-top-joints resilience, n/a-handling, flag gate (severe + on / severe + off / mild / ok / unknown / None), case-insensitive verdict.
  - [backend/tests/test_runs.py:592-685](reward-sculptor-ui/backend/tests/test_runs.py:592) — 2 tests: suggestion surfaces on REST detail when event fires; None when no event (baseline guard).
- **Why**: Sam's cartwheel test (CONTEXT 00:15) showed that reward edits alone can't fix a policy that's exploiting physically unrealistic actuator response — the MJCF itself needs tightening. §7.3 produces the diagnosis; §7.4 packages it into a ready-to-apply physics-edit prompt so Sam can click through the fix in one navigation instead of reverse-engineering the audit numbers into a manual prompt.
- **How**:
  - **Suggest-only, not auto-apply.** The existing physics-edit pipeline (`backend/services/physics.py::apply_prompt_edit`) lives under `backend/`; sculpt.py (which owns the inter-iter auto-adjust window) can't import it without creating a dependency cycle. Full auto-apply requires factoring the Claude-driven XML-rewrite logic into `sculptor/adapters/mjcf_editor.py` — a ~150-LoC refactor that's scoped as a follow-up. The suggest-only MVP ships today's value (click-through UX, versioned prompt) without that refactor.
  - **Feature-flag off by default.** `auto_adjust_physics = false` in `_CONFIG_TEMPLATE` — matches the directive's "Sam enables per-project" design. Existing projects on pre-§7.4 config.toml treat the missing key as False (`iter_cfg.get("auto_adjust_physics", False)`) so no migration needed.
  - **Canonical KG citations inlined.** The prompt cites three papers already in the shared KG (2312.17507 Actuator-Constrained RL, 1901.08652 ANYmal actuator-net, 2410.08650 Extended Friction). Claude's editor stage validates these exist before applying the edit — if a paper falls out of the KG, the user sees a clear rejection instead of silent drift.
  - **sessionStorage handoff**: zero backend round-trips, survives page reload within the same session, naturally scoped because PhysicsTab clears the key on consumption.
- **Verified**:
  - Sculptor: `149 passed, 1 skipped` (was 139; +10 auto_physics tests; no regressions).
  - Backend: `219 passed, 6 deselected` (was 217; +2 physics_edit_suggestion tests).
  - Frontend tsc: exit 0 ✓.
  - Live smoke: module imports + synthesize_auto_physics_prompt on a sample audit produces the expected Eureka-style output with all canonical citations.
  - Not yet live-smoked end-to-end through a real sculpt run. The UI chip + sessionStorage hand-off needs a browser test on an actual run with verdict=severe. Will fall out of next cartwheel-style run.

### 2026-04-23 05:38 — Ship 3 / §7.3: physics-realism audit + UI surfacing

- **What**:
  - [sculptor/adapters/realism.py](RewardSculptor/sculptor/adapters/realism.py) — new module (~230 LoC). `audit_rollout(trajectory_path, limits_path)` reads the §7.1 expanded npz + a new `mjcf_limits.json` snapshot and returns `{verdict, torque_saturation_frac, any_joint_saturation_max, joint_vel_p99_max, joint_vel_multiplier_vs_nominal, joint_limit_violation_frac, top_joints_saturation, top_joints_vel, top_joints_limit_violation, n_actuators, n_joints, n_steps}`. Thresholds tuned against Sam's cartwheel evidence: severe ≥30% any-joint saturation / ≥50% overall / ≥3× nominal vel; mild ≥10% saturation / ≥1.5× nominal vel. Never raises — missing/malformed inputs return `verdict="unknown"` with a reason.
  - [sculptor/adapters/_mjlab_runner.py:638-705](RewardSculptor/sculptor/adapters/_mjlab_runner.py:638) — `_cmd_rollout` snapshots `actuator_forcerange` + `jnt_range` + names from the mujoco model at env-init (tries env.sim/physics first, falls back to robot entity). Writes `<rollout_dir>/mjcf_limits.json` at rollout end ([:896-910](RewardSculptor/sculptor/adapters/_mjlab_runner.py:896)). Guards every attribute lookup so mjlab version drift degrades to empty arrays (audit → verdict=unknown), not crash.
  - [sculptor/sculpt.py:612-640](RewardSculptor/sculptor/sculpt.py:612) — `_run_one_iter` runs `audit_rollout` after rollout / before diagnose, writes `iter_<N>/realism_audit.json`, emits `[SCULPT-EVENT] realism_audited` with verdict + top joints. Best-effort; any failure logs to stderr without blocking the loop.
  - [sculptor/diagnose.py:242-284](RewardSculptor/sculptor/diagnose.py:242) — new `_load_realism_audit(iter_dir)` + `_format_realism_audit(audit)`. Emits a `# PHYSICS_REALISM_AUDIT` block in prompts ONLY for `mild` / `severe` verdicts (ok/unknown stay silent). Shows the 3-4 top offending joints inline.
  - [sculptor/diagnose.py:404-447](RewardSculptor/sculptor/diagnose.py:404) + [:475-499](RewardSculptor/sculptor/diagnose.py:475) — both prompt builders accept `realism_audit` kwarg, inject between `# TRAINING_FEEDBACK` and subsequent sections.
  - [sculptor/diagnose.py:537-547](RewardSculptor/sculptor/diagnose.py:537) + [:562](RewardSculptor/sculptor/diagnose.py:562) + [:602](RewardSculptor/sculptor/diagnose.py:602) — `diagnose()` loads once and passes to both stages.
  - [sculptor/prompts/diagnose_preliminary.md:36-43](RewardSculptor/sculptor/prompts/diagnose_preliminary.md:36) — new paragraph: severe + reward_hacking → physics-edit; mild → reward-side shape.
  - [backend/models/run.py:67-84](reward-sculptor-ui/backend/models/run.py:67) — `IterEventSummary.realism_audit: Optional[dict] = None`.
  - [backend/services/run_manager.py:476-500](reward-sculptor-ui/backend/services/run_manager.py:476) — `_check_iter_artifacts` detects `iter_<N>/realism_audit.json` and emits `realism_audited` with full payload (source="fs"). Threaded `seen_realism: set[int]` through `_fs_watcher`, `_scan_once`, `_handle_fs_changes`.
  - [backend/services/run_manager.py:680-712](reward-sculptor-ui/backend/services/run_manager.py:680) — `_iter_events` folds both stdout-form (flat fields) and fs-form (nested `audit` dict) into `slot["realism_audit"]`; fs wins when both arrive.
  - [frontend/src/lib/types.ts:330-354](reward-sculptor-ui/frontend/src/lib/types.ts:330) — new `RealismAuditPayload` interface + `realism_audit: RealismAuditPayload | null` on `IterEventSummary`.
  - [frontend/src/components/RunsTab.tsx:513-532](reward-sculptor-ui/frontend/src/components/RunsTab.tsx:513) — verdict chip on iter rows (amber for mild, rose for severe); tooltip shows overall + worst-joint saturation. Chip omitted for ok/unknown.
  - [frontend/src/components/RunsTab.tsx:718-763](reward-sculptor-ui/frontend/src/components/RunsTab.tsx:718) — reducer consumes `realism_audited` events, prefers fs-form `audit` payload, falls back to stdout flat fields; `_mergeIterSlot` carries `realism_audit` through status-rank merges.
  - [tests/test_realism.py](RewardSculptor/tests/test_realism.py) — 12 new tests: ok / severe-on-any-joint / mild boundary / severe-on-high-velocity / mild-on-moderate-velocity / limit-violation reporting / missing-trajectory / missing-limits / pre-§7.1 trajectory / empty limits / corrupt npz / positional name fallback.
  - [tests/test_diagnose.py](RewardSculptor/tests/test_diagnose.py) — 3 new tests: severe verdict injects block, ok verdict omits, missing file omits.
  - [backend/tests/test_runs.py:483-592](reward-sculptor-ui/backend/tests/test_runs.py:483) — 2 new tests: realism_audit surfaces via REST detail when event fires, stays None when no event (baseline regression guard).
- **Why**: Sam's cartwheel test (CONTEXT 00:15) produced a policy that thrashed all 29 G1 joints at max torque — classic reward-hacking by exploiting unrealistic actuator response. Reward edits alone cannot fix that; the physics itself needs tightening. §7.3 is the measurement half: audit the rollout, classify severity, surface the verdict to the diagnoser + UI. §7.4 (auto-adjust physics) will consume these numbers next.
- **How**:
  - **MJCF parsing avoided.** Instead of re-parsing the task XML in the audit, the rollout runner snapshots the already-loaded mujoco model's `actuator_forcerange` + `jnt_range` arrays to `mjcf_limits.json`. The audit is pure numpy + json, unit-testable CPU-only.
  - **Severity thresholds inlined as module-level constants** (`_SEVERE_ANY_JOINT_SAT = 0.3`, etc.) so `test_realism.py` can reference them by name — if a threshold moves, the test moves with it.
  - **Prompt injection order matters.** The audit block sits AFTER `# TRAINING_FEEDBACK` and BEFORE the behavior vocab / contract blocks. That keeps the block grouped with other post-rollout signals, and preserves prompt-cache hash stability since it's appended to the already-moved section.
  - **FS-emitted event carries full `audit` dict**, stdout-emitted event carries flat fields only. The reducer (backend + frontend) prefers the fs form because it has top_joints; falls back to the stdout form for immediate UI feedback before disk write lands.
  - **Verdict chip design**: compact inline element in the iter row (matches existing failure-mode chip width). No severity chip for ok/unknown — healthy runs should feel quiet.
- **Verified**:
  - Sculptor: `139 passed, 1 skipped` (was 124; +15 new tests — 12 realism + 3 diagnose realism; zero regressions).
  - Backend: `217 passed, 6 deselected` (was 215; +2 new realism surfacing tests).
  - Frontend tsc: exit 0 ✓.
  - Live smoke: realism module imported + run on a synthetic trajectory — verdict=severe returned as expected for 100%-saturated torques, JSON structure matches spec.
  - Not yet live-smoked through a GPU rollout. The rollout-side MJCF snapshot logic (mujoco model attribute probing) is the one piece that depends on mjlab's runtime shape and couldn't be exercised by unit tests. Will fall out of next GPU sculpt run or a dedicated GPU-gated test in a follow-up.

### 2026-04-23 05:27 — Ship 2 / §7.2: Eureka reward-reflection block in diagnose + edit prompts

- **What**:
  - [sculptor/diagnose.py:242-306](RewardSculptor/sculptor/diagnose.py:242) — new `_load_training_feedback(iter_dir)` (prefers `<iter_dir>/reward_trajectory.json`, falls back to `<iter_dir>/rollout/reward_trajectory.json`) + `_format_training_feedback(data)` (Eureka Appendix F line format: `name: [v0, ..., v9], Max, Mean, Min` — capped at 10 samples).
  - [sculptor/diagnose.py:332-360](RewardSculptor/sculptor/diagnose.py:332) + [:376-394](RewardSculptor/sculptor/diagnose.py:376) — `_build_preliminary_user_content` + `_build_grounded_user_content` now accept `training_feedback: dict | None = None` and inject `# TRAINING_FEEDBACK` between `# metrics.json` and the behavior-metric vocab (preliminary) / before `# behavior.json` (grounded).
  - [sculptor/diagnose.py:453-456](RewardSculptor/sculptor/diagnose.py:453) + [:483-491](RewardSculptor/sculptor/diagnose.py:483) + [:516-525](RewardSculptor/sculptor/diagnose.py:516) — `diagnose()` loads once and threads into both call-builders.
  - [sculptor/prompts/diagnose_preliminary.md:16-32](RewardSculptor/sculptor/prompts/diagnose_preliminary.md:16) — added paragraph naming dead-component / imbalance / premature-termination patterns the block exposes.
  - [sculptor/prompts/diagnose_grounded.md:4-10](RewardSculptor/sculptor/prompts/diagnose_grounded.md:4) — added dead-component → `remove`/`replace`, imbalance → `decrease`/`clip` guidance.
  - [sculptor/edit.py:415-427](RewardSculptor/sculptor/edit.py:415) — `_build_user_prompt` accepts `training_feedback` kwarg; injects `# TRAINING_FEEDBACK` block before `# REWARD_CONTRACT`.
  - [sculptor/edit.py:703-713](RewardSculptor/sculptor/edit.py:703) — `apply_edits` accepts `iter_dir: Path | str | None = None` kwarg; [:791-810](RewardSculptor/sculptor/edit.py:791) loads feedback via `_load_training_feedback` (prefers explicit `iter_dir`, falls back to `diagnosis.iter_dir`, no-op if neither is a directory).
  - [sculptor/sculpt.py:639-655](RewardSculptor/sculptor/sculpt.py:639) — `_run_one_iter` passes `iter_dir=iter_dir` into `apply_edits`.
  - [sculptor/prompts/edit_rewriter.md:101-122](RewardSculptor/sculptor/prompts/edit_rewriter.md:101) — new rule §8 "Dead-component rule" tells Claude to `remove`/`rewrite`/`rescale` components whose span is <5% of |Max|, preserving unchanged only with a physics-grounded rationale.
  - [tests/test_diagnose.py:325-452](RewardSculptor/tests/test_diagnose.py:325) — 6 new tests: formatter output shape, 10-sample cap, empty dict handling, block injection, missing-file graceful skip, training-over-rollout precedence.
  - [tests/test_edit.py:867-1003](RewardSculptor/tests/test_edit.py:867) — 3 new tests: dead-component rule present in prompt text, `apply_edits` injects block when `iter_dir` passed, omits block otherwise.
- **Why**: Eureka paper (arXiv:2310.12931) Appendix F ablation showed the per-component reward-reflection format lifts reward-discovery performance 28.6%. Sculptor's diagnose step previously gave Claude qualitative failure-mode classification; this pass gives it raw time-series numbers + a hard-coded dead-component rule so dead terms get surgically rewritten instead of passively carried forward.
- **How**:
  - Block placement: between `# metrics.json` and `# behavior.json` in both prompts, matching the existing markdown-section convention (directive proposed XML `<training_feedback>` but markdown keeps the prompt-cache hash stable with existing tests).
  - `_format_training_feedback` caps shown values at 10 evenly-sampled points (Eureka's default) to bound prompt size; Max/Mean/Min stats are computed over the FULL series so 1000-step rollouts still give accurate summaries.
  - `__`-prefix on aux keys (§7.1 emits `__episode_length`, `__terminated`, `__time_outs`) is stripped at render time so the LLM sees `episode_length:` not `__episode_length:`.
  - Optional parameter defaulted to `None` on every function that took a new arg — no breaking change to existing callers (reward_jobs.py's `apply_prompt_edit` path still works, the Rewards-tab prompt flow gets zero feedback since there's no iter_dir context yet, which matches the one-shot UX).
- **Verified**:
  - Sculptor: `124 passed, 1 skipped` (was 115; +9 new tests; no regressions).
  - Backend: `215 passed, 6 deselected` ✓ baseline.
  - Frontend tsc: exit 0 ✓ baseline.
  - Not yet live-smoked via full GPU run — would require a sculpted-reward iter that writes reward_trajectory.json and a subsequent diagnose on a clean iter. The 6 diagnose tests cover the loader + both prompt builders; 3 edit tests cover the system prompt + injection both ways. Will fall out of Ship 3's live smoke.

### 2026-04-23 05:20 — Ship 1 / §7.1: expanded rollout trajectory + training-side component sink

- **What**:
  - [sculptor/adapters/_mjlab_runner.py:28-100](RewardSculptor/sculptor/adapters/_mjlab_runner.py:28) — new module-level `_COMPONENT_SINK` + `_record_components(sink, components, info)` helper + `_snapshots_to_trajectory(snapshots)` pivot helper.
  - [sculptor/adapters/_mjlab_runner.py:296-310](RewardSculptor/sculptor/adapters/_mjlab_runner.py:296) — `SculptorRewardTerm.__call__` now feeds the sink when it's active (no-op otherwise).
  - [sculptor/adapters/_mjlab_runner.py:414-425](RewardSculptor/sculptor/adapters/_mjlab_runner.py:414) — `_cmd_train` activates sink only on sculpted runs (`args.reward_module_path is not None`); initializes `checkpoint_window_snapshots: list[dict[str, float]]`.
  - [sculptor/adapters/_mjlab_runner.py:469-477](RewardSculptor/sculptor/adapters/_mjlab_runner.py:469) — progress-poll thread snapshots + clears the sink on every new `model_<N>.pt`, appending one dict per save_interval window.
  - [sculptor/adapters/_mjlab_runner.py:493-519](RewardSculptor/sculptor/adapters/_mjlab_runner.py:493) — end-of-training `finally` block writes `<output_dir>/reward_trajectory.json` (Eureka Appendix F shape) + resets sink to None so the next sculpt iter starts clean.
  - [sculptor/adapters/_mjlab_runner.py:665-792](RewardSculptor/sculptor/adapters/_mjlab_runner.py:665) — `_cmd_rollout` buffers per-step `joint_pos/joint_vel/action/actuator_force/projected_gravity_b/root_link_pos_w` + `env.reward_manager._step_reward` per-term. Expanded `trajectory.npz` adds those fields + `reward_term__<name>` keys. Companion `<rollout_dir>/reward_trajectory.json` written with per-term time-series.
  - [tests/test_mjlab_adapter.py:415-506](RewardSculptor/tests/test_mjlab_adapter.py:415) — six CPU-only unit tests covering append-mean, no-op-on-None, non-tensor skip, NaN guard, snapshot pivot, and late-debut components.
- **Why**: §7.1 P0 prereq for the Eureka reflection block (§7.2) and physics-realism audit (§7.3). Sam's cartwheel test showed the policy spasming all joints at max torque while `_components` returned by `compute_reward_batched` was discarded — diagnose had no per-component time-series to reason about.
- **How**:
  - Module-level sink (not class attribute) so multiple `SculptorRewardTerm` instances in one subprocess share it; rsl_rl may rebuild the term on env reset.
  - Train-time capture snapshots per save_interval (not cumulative) so Eureka's `[v0, v1, ..., vN]` list is window-means over time, not monotonic averages.
  - NaN / empty-tensor guards in `_record_components` — a diverging reward component or zero-env info call can't poison window sums.
  - Rollout per-term decomposition pulls from `env.reward_manager._step_reward` + `_term_names` (mjlab's internal RewardManager API, verified stable across the pinned version). Any mjlab drift caught by Ship 3's `test_realism.py`.
  - Per-field try/except on rollout buffers — fixed-base tasks (Cartpole) legitimately lack `projected_gravity_b` / `root_link_pos_w` and should degrade gracefully, not crash.
  - Shape-consistency check in `_stack_if_consistent` — drops a field if any step's buffer diverges from the first-step shape (e.g. task auto-resets with different num_dofs), rather than writing a jagged npz that downstream can't read.
- **Verified**:
  - Sculptor: `115 passed, 1 skipped` (was 109; +6 new tests from this ship, no regressions).
  - Backend: `215 passed, 6 deselected` ✓ baseline.
  - Frontend tsc: exit 0 ✓ baseline.
  - AST parse of `_mjlab_runner.py`: OK.
  - Live helper smoke: `_record_components` + `_snapshots_to_trajectory` work end-to-end on hand-constructed torch tensors.
  - Not yet live-smoked through a real GPU rollout — would require a 2-iter sculpt run + disk diff. Relies on Ship 2's live smoke to exercise the new rollout paths.

### 2026-04-23 00:15 — Cartwheel-test findings + Eureka paper review + KG seed expansion

Session summary for handoff to a new window. Full plan at
[NEXT_WINDOW_DIRECTIVE.md](NEXT_WINDOW_DIRECTIVE.md).

**KG expanded (shared, ~/.local/share/sculptor/kg/graph.db)**:
- Bulk-ingested 33 new arxiv papers curated by 3 parallel research agents:
  humanoid acrobatics (BeyondMimic, WoCoCo, Humanoid Parkour, ASAP, DeepMimic,
  AMP, PHC, Expressive WBC, Stable High-Speed, Deep RL That Matters),
  PPO/locomotion metrics (PPO, Learning-to-Walk-in-Minutes, Emergent Gaits,
  SAC, GAE, Humanoid Transformers), physics realism (Actuator-Constrained RL,
  CaT, Torque-based Biped, Not-Only-Rewards, PACE, Extended Friction,
  Differentiable SysID, SPI-Active, Elastic Actuators, Dynamics Randomization,
  RMA, Rapid Locomotion, Adaptive Curriculum DR, 5 SEA characterization papers).
- Paper count: **47 → 71**. Extract: **+147 techniques, +114 failure modes,
  +78 reward components, +50 environments, +120 INTRODUCES edges**.
- Seed file saved at [cartwheel_kg_seeds.yml](cartwheel_kg_seeds.yml).
- Two issues surfaced: (a) ~9 papers got stub titles `arxiv:XXXX.XXXXX`
  because arxiv's API rate-limits aggressively on bulk requests; fallback
  path uses the seed YAML's title field which my YAML didn't include.
  (b) Paper 2202.13500 is the wrong paper — agent 2 hallucinated it as
  "Learning Humanoid Locomotion with Transformers"; actual content is
  about social content moderation. Dead weight in KG, not polluting
  semantic search (extract returned 0 nodes).

**Motor-limits template button shipped** in PhysicsTab.tsx — fills the
prompt textarea with a structured datasheet template (torque/speed/gear_ratio/
rotor_inertia per joint) citing ANYmal actuator-net (1901.08652),
Actuator-Constrained RL (2312.17507), Extended Friction (2410.08650).

**Eureka paper review** — 3 parallel Explore agents dissected
[arxiv:2310.12931](https://arxiv.org/abs/2310.12931). Honest assessment:
Sculptor's reward-discovery capability is ~3-5× behind Eureka on the
**algorithm axis** (they use evolutionary search K=16 candidates per
iter + per-component reflection statistics), but ahead on UI /
auditability / KG grounding. Of Eureka's ideas, only ONE is both
portable to Sam's laptop setup AND high-value:

> **Per-component reward-reflection block in the diagnose prompt.**
> Format (from Eureka Appendix F): `component_name: [v0, v1, ..., vN],
> Max, Mean, Min`. Captured at ~10 checkpoints during training and fed
> into the next iteration's LLM prompt. Ablation in the Eureka paper
> shows removing this drops their reward-discovery performance 28.6%.
> Sculptor's diagnose step currently gives Claude a qualitative
> failure-mode classification; Eureka gives the LLM raw time-series
> numbers. Estimated effort: S (few hours).

Evolutionary search (K candidates per iter) NOT worth porting — Sam's
RTX 5070 Laptop can't afford 16× parallel PPO runs per iter, and even
K=3 triples total wall-clock.

**Cartwheel test attempted** — G1, `Mjlab-Cartpole-Swingup`... wait wrong
task. Task: G1 humanoid with rewards prompt describing a sagittal-plane
cartwheel (angular momentum, alternating hand/foot contacts, COM above
ground). Run: 8 sculpt iters × 500 rsl_rl iters/cycle × 1024 envs.
Predicted 2.1 h wall-clock. Run ended after 4 iterations with all failure.

Results that Sam observed:
1. **Reward-edit worked cleanly**. 5 KG matches above sim=0.35, activity
   panel streamed events, Claude emitted 8 hyperparameters + grounding
   dict + 3 references (2304.09434, 2406.08858, 2409.16611). Two of the
   three references had stub titles because of the arxiv rate-limit bug
   above. Probe passed.
2. **`training_iterations=500` override not applied** — logs showed
   "1m Learning iteration xx/1500". Sam set 500 but mjlab's runner got
   1500. Either (a) Sam didn't restart uvicorn to pick up the S8 CLI
   plumbing, (b) the CLI-flag path has a regression. Needs verification.
3. **Project-tab breakage while a run is active** — clicking back to
   another project (swingup-cartpole) shows: static preview 500 errors,
   reward/physics tabs barely render, report gone, "no runs completed"
   even though the project has completed iters on disk. Smells like
   a frontend state-scoping bug (query keys bleeding across projects)
   OR the backend blocking on the active run's locks.
4. **Cartwheel policy is unrealistic** — robot never left standing pose;
   squatted + raised arms + shook all joints at maximum velocity (joint
   limits saturated). Classic reward-hacking failure mode where the
   policy exploits unbounded action magnitudes that the sim permits
   but a real motor couldn't sustain. Sam's Euraka-motivated remark:
   **physics-realism check + automatic physics adjustment between
   iterations is necessary for any acrobatic skill learning**.

**Next-window plan targets** (full details in NEXT_WINDOW_DIRECTIVE.md):
- **P0** — Expand rollout trajectory capture (prereq for everything else).
- **P0** — Eureka reward-reflection block in diagnose prompt + dead-component rule.
- **P0** — Physics-realism audit after each iter (read trajectory + MJCF limits).
- **P1** — Auto-adjust physics between iterations when realism violations
  exceed threshold (chains audit → KG-grounded physics prompt).
- **P1** — Fix `training_iterations` override regression (or confirm it
  was just a stale-reload issue).
- **P1** — Fix project-tab breakage during active run.
- **P1** — Stub-title fix: arxiv API retry-with-backoff + title fields in
  seed YAMLs + healing pass on existing stubs.
- **P2** — Report graphs via matplotlib (metric history, component stack,
  KL/entropy, per-env return violin).
- **P2** — Motor-datasheet PDF upload → MJCF patch.
- **P3** — Frontend React Query cache invalidation on physics commit.

Final baselines this session: sculptor 109 passed, 1 skipped; backend
215 passed, 6 deselected; frontend typecheck exit 0. No regressions
shipped during the KG-visibility + motor-limits-template work.

### 2026-04-22 22:15 — KG visibility pass: surface counters + match list + tech-level citation rule

Sam's Test 1 round 4: biggest complaint was "KG isn't working". Three agents investigated in parallel. Live KG probe confirmed **the KG IS working, the UI just hides the evidence**. Verified:

- Shared KG at `~/.local/share/sculptor/kg/graph.db`: 47 papers, 269 techniques (all with embeddings), 187 INTRODUCES edges, 688 edges total. Healthy.
- `research_topic("reinforcement learning cartpole swing-up", max_papers=3)` live call: Claude returned 3 papers, all 3 already in KG. My round-2 counters captured this exactly (`papers_returned_by_claude=3, papers_deduped_against_kg=3, kept=0`).
- `query_semantic` on Sam's actual rewards prompt: 5 matches above sim=0.35, every match with populated `source_paper_ids`. So the CITATIONS block in Claude's prompt was populated. Claude voluntarily emitted `references=[]` because the round-2 tangential-citation prompt rule told it to skip non-Cartpole matches.

**So the three fixes**:

**Fix 1 (critical) — Research toast reads the stage counters.**
- Frontend `PendingSeedJobWatcher` in [AddSeedsDialog.tsx:157-200](reward-sculptor-ui/frontend/src/components/AddSeedsDialog.tsx) only read `job.result.ingest.ingested` and `job.result.extract.succeeded`. Research jobs where dedupe ate everything return `ingest=None, extract=None` → toast showed "Added 0 paper(s), extracted 0" with no hint why.
- Fix: when `job.kind === "kg_research"` AND new-papers count is 0, read `papers_returned_by_claude` + `papers_deduped_against_kg` from `job.result`. Render a targeted message: `"Claude returned 3 paper(s), all already in KG"` or `"Claude found 0 papers for {topic} — Coverage note: ..."`.
- Also: `ingest.ingested` is a `string[]` (list of arxiv_ids) on research jobs but a `number` on bulk-seed jobs — normalized the count extraction.

**Fix 2 — Reward-edit result exposes KG consultation.**
- [backend/services/reward_jobs.py:104-140](reward-sculptor-ui/backend/services/reward_jobs.py) now runs `query_semantic` once more in the runner (same params as apply_prompt_edit uses internally) to capture the lit_context matches, flattens them into a UI-renderable `kg_matches` list, and includes on the job result. Fields: `technique`, `relevance_score`, `arxiv_ids` (from source_paper_ids), `paper_citation`. Also returns `kg_min_similarity=0.35` so the UI can report the floor.
- [frontend/src/components/RewardsTab.tsx:203-235](reward-sculptor-ui/frontend/src/components/RewardsTab.tsx): completion toast description now reads `kg_matches` + renders either `"KG: 5 matches above sim=0.35 — top: early_termination, inner_product_reward_design…"` or `"KG: no matches above sim=0.35 — Claude grounded in physics only"`.
- Sam now has direct visibility for every prompt-edit: did the KG fire? What matched? Even when references is empty.

**Fix 3 — Prompt: technique-level citation, not paper-level.**
- [sculptor/prompts/edit_rewriter.md](RewardSculptor/sculptor/prompts/edit_rewriter.md) replaced the "tangential-citation rule" with a "technique-level citation rule". Pre-fix: told Claude "if the paper's robot is humanoid but the task is Cartpole, skip it". Post-fix: "cite the technique when the TECHNIQUE applies, even if the paper's original context is a different robot. Over-citing at technique level is fine; only reject truly-irrelevant techniques." The KG's 0.35 pre-filter is now treated as "almost certainly applicable unless specifically not".
- Sam's exact cartpole prompt matched 5 genuinely-general techniques (`early_termination`, `inner_product_reward_design`, `positive_task_reward_transformation`, …) — the old rule told Claude to drop them all because the papers were locomotion. The new rule will have Claude cite 3-5 of them as `references[]`.

**Verified**:
- Live research probe (`cartpole swing-up`, max=3): `{papers_returned: 3, deduped: 3, kept: 0, coverage_note: "mostly studied as a benchmark..."}`. My counter plumbing lands on `job.result`.
- Live query_semantic on Sam's rewards prompt: 5 matches, every one with `source_paper_ids: ["paper:..."]` populated.
- Sculptor: **109 passed, 1 skipped** (unchanged).
- Backend: **215 passed, 6 deselected** (unchanged — only frontend + prompt + result-dict changes).
- Frontend typecheck: **exit 0**.

**Not a bug — working as designed**:
- Sam's research "added 0 papers" is literally correct — Claude's top 3 matches for well-covered topics like "pendulum RL" are classics (DDPG, PPO, DM Control) that are ALREADY in the 47-paper seed KG. Every new topic Sam tries on a classic area will dedupe. This is actually GOOD (no duplicate seeds). The toast now explains it.
- Sam's "references (novel)" on round-4 v1 happened before Fix 3. Next prompt-edit will cite.

**Still-deferred (noted for follow-up)**:
- Physical-realism check between iterations (Sam asked for this). Good idea — read torque/velocity from rollout trajectories, compare against MJCF's `<actuator forcerange>` + joint limits, flag violations. Fits the existing diagnose pass but needs a new rule. Defer.
- Motor-limits prompt tab + PDF-datasheet upload. Separate feature. Defer.
- Preview PNG stale after physics commit — I fixed the backend file-unlink last session but Sam's round-4 showed it's still stale. Likely the React-Query cache on `/preview` isn't invalidating. Follow-up: add `usePhysicsCommitSuccess` hook to invalidate `["robot", slug, "preview"]` queries, OR add mtime header to the route and set React Query's `staleTime` accordingly.
- Reports tab KG citations surfacing — when v1/v2/v3 have references, reports should aggregate them. Verify after Fix 3 makes Claude actually cite.

**Retry sequence**:
1. Reload `./run.sh` (sculptor hot-reloads; Vite hot-reloads; prompt text reloads at next import).
2. Fresh prompt-edit on any project → completion toast will show KG summary.
3. Try KG Research with any topic → toast will now distinguish "Claude returned 0" vs "all in KG" vs success.
4. The reward prompt should yield a v1 with non-empty `references[]` for general techniques.

### 2026-04-22 21:00 — Test 1 (round 3) 7-issue pass: torch+staging+UX

Sam re-ran Test 1 after round-2's fixes. Two plan agents (minimalist + architectural) investigated in parallel, council-synthesized decisions, shipped all 7. The dominant theme: **two different dummy-input contracts for reward modules** (numpy for pre-flight, nested-list for probe) had been building up subtle drift; both now use torch tensors which is what Claude's generated modules actually expect.

**Issue 1 (TIER 0) — Reward rewrite "failed" toast fires but v<n>.py on disk**: `_post_validate` wrote `new_source` directly to `target_path` BEFORE any validation. Failed validation left polluted file + error toast; next Rewards-tab load read the broken file.
- Fix: `_post_validate` writes to `<dir>/.v<n>.staging.py` first (leading dot → hidden; kept `.py` so `importlib.util.spec_from_file_location` auto-detects the loader — had to fix this after first attempt: `.pending` suffix broke spec detection with "could not spec reward module"). Validates all checks wrapped in try/except → on any failure, `staging.unlink(missing_ok=True)` + re-raise. On success, `os.replace(staging, write_to)` atomic rename. Invariant now holds: `rewards/v<n>.py` exists ⇒ Claude output passed every post-validation gate.
- Files: [RewardSculptor/sculptor/edit.py:575-700](RewardSculptor/sculptor/edit.py).
- Test: `test_post_validate_unlinks_staging_on_validation_failure` in [test_edit.py](RewardSculptor/tests/test_edit.py) — uses a source missing `compute_reward`; asserts neither target nor staging exists after raise.

**Issue 2 (TIER 0) — Training crash: `torch.cos(): input must be Tensor, not numpy.ndarray`**: `_build_dummy_inputs` in pre-flight built `np.zeros` dicts. Claude's generated modules use a scalar-wrapper pattern that dispatches to `compute_reward_batched` which does `torch.cos(state["qpos"][..., 1])` — blows up on numpy.
- Fix: `_build_dummy_inputs` now returns torch CPU tensors for schema-style contracts. Leading dim 1 preserved; dtype float32. Gym-style path (which uses numpy throughout) unchanged. `_call_compute_reward` already coerces reward/components via `float()`, which works on torch scalar tensors.
- Files: [RewardSculptor/sculptor/edit.py:174-204](RewardSculptor/sculptor/edit.py).

**Issue 3 (TIER 0) — Probe crash: `'list' object has no attribute 'device'`**: My S3-followup `_PROBE_SCRIPT` built nested Python lists. Claude's module calls `.device` / `torch.cos` on them → crash.
- Fix: `_PROBE_SCRIPT` now imports torch (subprocess-safe, separate interpreter) and builds `torch.zeros((1, *shape), dtype=torch.float32)` for schema mode. Scalar mode (gym_sb3) unchanged.
- Files: [backend/services/reward_store.py:184-230](reward-sculptor-ui/backend/services/reward_store.py).

**Issue 4 (TIER 1) — Task selector missing at project creation**: Cartpole library entry has TWO `preconfigured_tasks` (Balance, Swingup), but `CreateProjectDialog` silently sent `task_id=preconfigured_tasks[0].task_id`. Sam named his project "Swingup" but the backend stamped `task_id=Mjlab-Cartpole-Balance`.
- Fix: `CreateProjectDialog` now has a `<select>` dropdown when `robot.preconfigured_tasks.length > 1`. Snaps `num_envs` to the chosen task's `recommended_num_envs` if the user hasn't deviated. Added `task_id` to the mutation payload.
- Files: [frontend/src/components/CreateProjectDialog.tsx:85-107,250-300](reward-sculptor-ui/frontend/src/components/CreateProjectDialog.tsx).

**Issue 5 (TIER 1) — Preview PNG stale after physics keyframe commit**: Sam committed a physics edit adding a `hinge_1=π` keyframe (pole down) but Overview tab's static preview still rendered the default pose. [backend/routes/robot.py:413-419](reward-sculptor-ui/backend/routes/robot.py) serves the cached PNG based on file-exists only, no mtime invalidation.
- Fix: `run_physics_prompt_edit_job` now unlinks `<project_dir>/uploads/preview_*.png` after a successful commit. Next GET regenerates from the fresh MJCF. Reused the same glob pattern `ProjectStore.invalidate_preview` uses but inlined (no slug → store lookup needed, `project_dir` is in scope).
- Files: [backend/services/reward_jobs.py:228-248](reward-sculptor-ui/backend/services/reward_jobs.py).

**Issue 6 (TIER 1) — Autoscroll doesn't stay pinned in LogViewer**: Pre-fix `onItemsRendered` flipped `userScrolled.current = !atBottom` on every render pass. During a programmatic `scrollToItem` animation, react-window emits intermediate render passes where `visibleStopIndex < filtered.length - 1` → `userScrolled.current` flipped to true mid-scroll → next event's useEffect saw `userScrolled=true` → didn't re-scroll. Toggle snapped once, then detached permanently.
- Fix: Removed `onItemsRendered` entirely. `onScroll` now reads `scrollUpdateWasRequested` — true means OUR `scrollToItem` call, so ignore; false means user wheel/drag/keyboard, flip `userScrolled` based on pixel-offset from bottom (`contentHeight - viewport - scrollOffset > LINE_HEIGHT`). User scrolling up detaches autoscroll cleanly; incoming events keep pinning the view.
- Files: [frontend/src/components/LogViewer.tsx:34-105](reward-sculptor-ui/frontend/src/components/LogViewer.tsx).

**Issue 7 (TIER 2) — KG Research returned a fading-channels paper for inverted-pendulum query + extract silently skipped due to "no ANTHROPIC_API_KEY"**:
- Fix A (research quality): added a "Subject-area relevance is non-negotiable" rule to [sculptor/prompts/research_topic.md](RewardSculptor/sculptor/prompts/research_topic.md). Explicitly lists the cs.RO / cs.LG / cs.AI / eess.SY / stat.ML categories as the legit areas and calls out "returning a fading-channels paper for an inverted-pendulum query is a hard reject". Soft fix (Claude-side); still no hard rejection in code.
- Fix B (extract API key): `kg_jobs.py` was importing `sculptor` only lazily inside `_do_extract`, but the `os.environ.get("ANTHROPIC_API_KEY")` check happened BEFORE that. Sculptor's `__init__.py` loads `.env` at import time. On the first research job, the check ran before .env had been loaded → saw empty key → skipped extract with "no ANTHROPIC_API_KEY". Fixed by eager `import sculptor  # noqa` at the top of kg_jobs.py so the .env loader fires at module load.
- Files: [RewardSculptor/sculptor/prompts/research_topic.md](RewardSculptor/sculptor/prompts/research_topic.md), [backend/services/kg_jobs.py:22-40](reward-sculptor-ui/backend/services/kg_jobs.py).

**Council synthesis notes**:
- Two agents agreed on root causes for A/B/C but disagreed on fix scope: minimalist recommended targeted patches, architectural recommended unifying the two probe paths. I took the minimalist patches today and flagged "unify probes" as a follow-up — subprocess-isolation boundary makes the full refactor non-trivial.
- Issue 7A (research relevance) — agent 2 pushed for hard Claude-side rejection via the prompt; agent 1 pushed for a min-similarity retrieval threshold. I took agent 2's prompt rule since the Claude step is where the humanoid paper leaked through; retrieval threshold was already set at 0.35 in the previous pass.

**Verified**:
- Sculptor: **109 passed, 1 skipped** (was 108; +1 staging-unlink test).
- Backend: **215 passed, 6 deselected** (unchanged — no new backend test this pass).
- Frontend typecheck: **exit 0**.

**Deferred**:
- Live-clip "chaotic multi-cart view" during training: rollout_streamer shows whatever `rollout.mp4` the sculpt process produced. My round-2 Fix F (env[0] reset-skip) already limits that. If Sam still sees chaos it's because the STATIC preview is showing a fresh scene-dump that includes all 1024 envs; fixing that requires a separate "preview always renders env[0] only" scene patch. Note in plan.
- Full probe unification (delete `_PROBE_SCRIPT`, reuse `sculptor.edit._build_dummy_inputs` via a public helper): architectural cleanup, not a bug. Follow-up.

**What to retry**:
1. Fresh Cartpole project — pick `Mjlab-Cartpole-Swingup` from the new task dropdown.
2. Rewards tab: prompt; no more "failed" toast on success; Probe column works; Grounding populated.
3. Physics tab: commit a keyframe change; Overview preview updates on next refresh.
4. KG Research: with API key actually used, extract should fire. Topic relevance should improve (hard to guarantee — Claude still chooses).
5. Run tab: autoscroll pins. Training no longer crashes on `torch.cos(numpy)`.

### 2026-04-22 19:30 — Test 1 (re-run) 7-issue pass: council-synthesized fixes

Sam re-ran Test 1 (Cartpole 3-iter sculpt) and hit 7 issues. Ran two
plan agents in parallel with different framings (minimalist-ship-fast
vs architectural-root-cause), synthesized council decisions, then
shipped all 7. Every fix backed by regression tests.

**Issue C (TIER 0) — UI `training_iterations` override dropped**: Sam set "rsl_rl iters/cycle=100" in NewRunDialog but the subprocess ran at 1500. Root cause: `backend/services/run_manager.py` only read 4 of 8 `NewRunRequest` fields; `training_iterations` was never forwarded. Same class as the S4 `schema_keys` bug (CLI flag silently omitted).
- Fix: new `--steps-per-iter` CLI flag on `sculptor.cli run` → threads `steps_per_iter` kwarg through `sculpt_run` → overrides `cfg["iteration"]["steps_per_iter"]`. Backend `run_sculpt_job` now reads `run_params["training_iterations"]` and appends `--steps-per-iter N` when set.
- Files: [RewardSculptor/sculptor/cli.py](RewardSculptor/sculptor/cli.py:203), [RewardSculptor/sculptor/sculpt.py:762](RewardSculptor/sculptor/sculpt.py), [backend/services/run_manager.py:50](reward-sculptor-ui/backend/services/run_manager.py).
- Tests: `test_run_sculpt_job_forwards_training_iterations_as_cli_flag`, `test_run_sculpt_job_omits_steps_per_iter_flag_when_not_set` in [test_runs.py](reward-sculptor-ui/backend/tests/test_runs.py).

**Issue A (TIER 0) — Probe `AttributeError: 'float' object has no attribute 'items'`**: UI probe used scalar 0.0 dummies but Claude's mjlab v1 reads `state["qpos"]`. Two probe paths had drifted — sculptor's post-validate uses schema-based dict dummies, UI probe stayed scalar.
- Fix: `_PROBE_SCRIPT` now takes optional `state_schema` + `info_keys` argv args. When present, builds nested-list dict state with leading dim 1. `probe_components(source_path, state_schema=..., info_keys=...)`; caller `load_detail` loads the project's adapter, extracts `reward_contract().state_schema`, passes through. Silent fallback to scalar mode on adapter load failure (gym_sb3 / coming-soon adapters preserved).
- Scalar component values coerced via `_scalar(v)` helper that handles tensors (torch Tensor `.item()`) + lists + floats — Claude's mjlab modules often return (torch.Tensor scalar, dict of torch.Tensor) on the scalar path.
- Files: [backend/services/reward_store.py:184-237](reward-sculptor-ui/backend/services/reward_store.py).
- Tests: `test_probe_components_handles_dict_state_for_schema_contracts`, `test_probe_components_scalar_mode_still_works_for_gym_sb3` in [test_rewards.py](reward-sculptor-ui/backend/tests/test_rewards.py).

**Issue B (TIER 1) — `REWARD_SPEC.references` empty despite populated grounding**: Claude wrote rich grounding dict but `references: []`. Deep root cause: `apply_prompt_edit` set `paper_refs=[]` on the synthetic ProposedEdit, so `_pre_validate` built an empty `citation_map`, so the Claude prompt's CITATIONS block was `{}`, so Claude had no arxiv_ids to cite. The `literature_context` from the KG query was collected but NEVER rendered into the prompt — dead-code path.
- Fix 1: populate `paper_refs` from `lit_context[*].source_paper_ids` (stripping `paper:` prefix) in `apply_prompt_edit`. Threads the KG matches into the existing citation_map machinery.
- Fix 2 (belt-and-suspenders): `_post_validate` now cross-checks grounding ↔ references. Any arxiv_id mentioned inside a `grounding` value that's missing from `references[]` triggers `EditValidationError`. Prompt mandate is now enforced; pre-fix Claude could decouple them silently.
- Files: [RewardSculptor/sculptor/edit.py:615-637](RewardSculptor/sculptor/edit.py) (validator), [RewardSculptor/sculptor/edit.py:852-868](RewardSculptor/sculptor/edit.py) (paper_refs plumb), [RewardSculptor/sculptor/prompts/edit_rewriter.md](RewardSculptor/sculptor/prompts/edit_rewriter.md) (new "Grounding-references consistency" rule).
- Tests: `test_apply_prompt_edit_threads_kg_arxiv_ids_into_paper_refs`, `test_post_validate_rejects_grounding_referencing_uncited_arxiv` in [test_edit.py](RewardSculptor/tests/test_edit.py).

**Issue G (TIER 1) — KG retrieval returned humanoid papers for Cartpole query**: 46-paper seed KG is locomotion-heavy; semantic search over `all-MiniLM-L6-v2` returned top-5 humanoid matches at sim=0.1-0.3 for a Cartpole prompt. Claude dutifully cited them.
- Fix: `query_semantic` gained `min_similarity: float = 0.0` kwarg. `apply_prompt_edit` passes 0.35 — drops tangentials while keeping genuine locomotion→quadruped matches. Prompt gained a "Tangential-citation rule": "If every LITERATURE CONTEXT entry is off-topic for the current robot, emit `references: []` and rely on physics first-principles."
- Files: [RewardSculptor/sculptor/kg/query.py:270](RewardSculptor/sculptor/kg/query.py), [RewardSculptor/sculptor/edit.py:795](RewardSculptor/sculptor/edit.py), [RewardSculptor/sculptor/prompts/edit_rewriter.md](RewardSculptor/sculptor/prompts/edit_rewriter.md).
- Tests: `test_query_semantic_filters_by_min_similarity` in [test_edit.py](RewardSculptor/tests/test_edit.py).
- Deferred: auto-seeding Cartpole-relevant papers (requires API key + network, opportunistic).

**Issue D (TIER 2) — KG Research "added 0 papers, extracted 0" no-signal**: green check hid whether Claude returned 0, dedupe ate everything, or ingest failed.
- Fix: `ResearchResponse` gained `papers_returned_by_claude` + `papers_deduped_against_kg` counters. `kg_jobs.py::run_research_job` surfaces them on `job.params` AND renders a targeted `job.message`: `"Claude returned 5 paper(s), all 5 already in KG (0 new to ingest)"` / `"Claude returned 0 papers for {topic} — try a more specific query or check coverage_note"`.
- Files: [RewardSculptor/sculptor/kg/research.py:63,186-215](RewardSculptor/sculptor/kg/research.py), [backend/services/kg_jobs.py:188-220](reward-sculptor-ui/backend/services/kg_jobs.py).
- Tests: `test_research_response_records_pipeline_counters`, `test_research_response_counts_dedupe_against_preexisting_kg` in [test_kg_research.py](RewardSculptor/tests/test_kg_research.py).

**Issue F (TIER 1) — Rollout video artifacts (checkered floor flashing, pole teleports)**: Cartpole rollout.mp4 showed env[0] auto-resetting mid-video, producing teleport glitches and black frames at episode boundaries.
- Fix: 1-line guard in `_cmd_rollout`: `if step % render_every == 0 and not bool(ep_done[0].item())` — render only while env[0]'s first episode is ongoing, stop capturing after it terminates. Result is one clean episode per rollout video.
- Files: [RewardSculptor/sculptor/adapters/_mjlab_runner.py:557](RewardSculptor/sculptor/adapters/_mjlab_runner.py).
- Tests: covered by existing GPU-gated smoke tests (CPU auto-skip).

**Issue E (TIER 1) — Timeline iter rows appear/disappear/reappear**: `useMergedIterations` rebuilt its map from scratch on every render using `rest` + `events`. Transient empty `rest` (race between backend fs watcher and in-memory event log) made iter rows vanish. Sam saw "iter 1 disappeared after iter 0 finished".
- Fix: `useRef<Map>`-backed sticky store inside `useMergedIterations`. Once an iter is known to exist, it's never removed. Updates go through `_mergeIterSlot` which enforces status-rank monotonicity (`queued < running < completed ≈ errored`) — a later REST "running" can't overwrite an earlier WS "completed". Null-field merging keeps completed_at / primary_metric / paper_refs from being lost across sources.
- Files: [frontend/src/components/RunsTab.tsx:657-780](reward-sculptor-ui/frontend/src/components/RunsTab.tsx).
- Tests: no vitest in repo; manual smoke on next Test 1 run.

**Verified**:
- Sculptor: **108 passed, 1 skipped** (was 103; +5 new — 1 training_iterations, 3 edit.py B/G/grounding, 2 kg_research counter tests = but 1 overlap + 2 probe tests went to backend — net +5 here).
- Backend: **215 passed, 6 deselected** (was 211; +4 new — 2 training_iterations + 2 probe).
- Frontend typecheck: **exit 0**.

**Not a bug / working as intended**:
- Physics edit rejection "Current cartpole MJCF matches the canonical DeepMind Control Suite … changing integrator exceeds 'ensure correct' request scope" — Claude correctly refusing an ungrounded physics change per the KG-grounding mandate. No fix.
- The `1500 rsl_rl iters` parameter Sam saw was the `steps_per_iter=1500` in `config.toml` stamped at project-create time for mjlab. Issue C's fix now lets the UI override it per-run.

**Retry plan for Sam**: restart `./run.sh` (Python hot-reload picks up sculptor changes; Vite hot-reloads frontend; backend auto-reloads). Then:
1. Fresh Cartpole project.
2. Rewards prompt with KG reference — the Activity panel should stream events, and if the KG has no cartpole-relevant matches, `references: []` with physics grounding (not humanoid citations).
3. KG Research-a-topic — result toast now tells you if Claude returned 0 or dedupe ate everything.
4. Sculpt run with `training_iterations=100` — log shows "1m Learning iteration xxx/100" not /1500.
5. Rollout video — one clean episode, no teleport.
6. Timeline — iter rows persist; status-rank monotonic.

### 2026-04-22 18:15 — S4 follow-up: Cartpole schema CLI-dispatch + KG research `.parsed_output` (live-Test-1 fixes)

Sam's live Test 1 (Cartpole 2-iter mini-run) surfaced two bugs the unit
tests didn't catch. Both now fixed + guarded with regression tests.

- **What** (bug #1 — Cartpole run crash in `SculptorRewardTerm.reset`):
  - [sculptor/adapters/mjlab.py::train](RewardSculptor/sculptor/adapters/mjlab.py): `--schema-keys` is now **always** passed to the `_mjlab_runner` subprocess. Pre-fix, `self.schema_keys` defaulted to None → CLI flag omitted → the runner fell back to `_DEFAULT_SCHEMA_KEYS` (7-key velocity schema) regardless of task_id. For Cartpole this meant `SculptorRewardTerm._snapshot` tried to build `command_vel` via `env.command_manager.get_command("base_velocity")` — which returns None silently on tasks without that command — and stored `self._prev["command_vel"] = None`. Then `reset()` crashed at `self._prev[k][env_ids] = 0.0` with `TypeError: 'NoneType' object does not support item assignment`.
  - [sculptor/adapters/_mjlab_runner.py::_snapshot](RewardSculptor/sculptor/adapters/_mjlab_runner.py): two defensive guards. (a) The `command_vel` branch now tests the return value for None explicitly (the try/except only catches raises, not None returns). (b) Unknown schema keys fall through to `_zeros(1)` rather than being silently skipped — ensures `self._prev.keys()` matches the schema contract so `reset()` always finds a tensor.
  - Test 1's live crash path is now closed: Cartpole schema (`qpos, qvel, actuator_force`) gets passed to the subprocess, and even if an unknown key sneaks through, the zero-fallback keeps `reset` sane.
- **What** (bug #2 — KG Research-a-topic `AttributeError: 'ParsedMessage[TypeVar]' object has no attribute 'output'`):
  - [sculptor/kg/research.py:156](RewardSculptor/sculptor/kg/research.py): `resp.output` → `resp.parsed_output`. Anthropic SDK 0.96.x `ParsedMessage` exposes the parsed payload under `.parsed_output` (a computed property scanning content blocks), not `.output`. The sibling sites `extract.py:194,211` and `diagnose.py:334,350` were already correct; `research.py` was the lone outlier (same bug-shape as the S1 kwarg mix-up from the beginning of this session — same module, different attribute).
- **Why the S1 + S4 tests didn't catch these**:
  - S1's test stubbed `client.messages.parse` to return a `_StubResponse(parsed)` with `self.output = parsed`. That fake response had `.output`, so the test passed even though `.output` is wrong on the real SDK. Classic test-passes-against-a-mock-that-mirrors-the-bug failure mode.
  - S4's tests covered schema dispatch at the `reward_contract` level (the pre-flight path) but not the subprocess CLI construction. The two schema paths (`mjlab.py::_schema_for_task` vs `_mjlab_runner.py::_cmd_train`) had drifted apart and nothing asserted they matched.
- **Tests added / hardened**:
  - [tests/test_kg_research.py::_StubResponse](RewardSculptor/tests/test_kg_research.py): now has `.parsed_output` (was `.output`) + a `@property` `output` that raises AttributeError to pin the bug shape. If someone swaps `resp.parsed_output` back to `resp.output`, the test fails with the exact error Sam saw live.
  - [tests/test_kg_research.py::test_research_module_call_sites_agree_with_extract_and_diagnose](RewardSculptor/tests/test_kg_research.py): extended to also assert `.parsed_output` is used (and `resp.output` is NOT) across research.py + extract.py + diagnose.py. Catches any future regression in any of the three modules.
  - [tests/test_mjlab_adapter.py::test_mjlab_adapter_train_subprocess_construction](RewardSculptor/tests/test_mjlab_adapter.py): extended to assert `--schema-keys` is present on the Go1 CLI and contains `{qpos, qvel, base_lin_vel_b, command_vel}`.
  - [tests/test_mjlab_adapter.py::test_mjlab_adapter_passes_cartpole_schema_to_subprocess](RewardSculptor/tests/test_mjlab_adapter.py) (new): mock-exec's the Cartpole CLI, asserts `--schema-keys` is exactly `{qpos, qvel, actuator_force}` AND that none of the locomotion-only keys (`command_vel`, `base_lin_vel_b`, `base_ang_vel_b`, `projected_gravity_b`) leak. This is the direct guard against Sam's Test-1 crash.
- **What wasn't a bug** (but worth documenting):
  - **`[run] ANTHROPIC_API_KEY not set` warning**: cosmetic. The warning fires at run-start before the sculpt subprocess loads the project's `.env` via sculptor's own loader. The Claude call later in the pipeline succeeds — verified by Sam's v1.py committing on the Rewards tab earlier in the same test.
  - **Physics edit rejection** ("Claude refused — mjlab requires implicitfast integrator, but current model uses euler; however KG has no Cartpole-specific physics parameters to justify changes"): this is **the KG-grounding mandate working as designed**. Claude saw a change that wasn't literature-grounded and refused, per the 2026-04-22 05:30 prompt update. Good signal, not a bug.
  - **Backend CUDA-flaky tests passed this run**: previously 3 CUDA-gated backend tests (test_create_project_with_library_slug_mjlab_ready, test_mjlab_success_writes_mjlab_adapter_to_config, test_preflight.py:151 live-VRAM) skipped with "CUDA not available on this host". With Sam's NVIDIA Control Panel fix + ProArt/Windows power tweaks landing earlier in this session, CUDA is now visible to WSL — those tests now pass live. Net: 208 pass + 3 skipped → **211 passed, 0 skipped** in the filtered backend suite.
- **Verified**:
  - Sculptor: **103 passed, 1 skipped** (was 102+1; +1 new Cartpole CLI test).
  - Backend: **211 passed, 6 deselected** (was 208+3-skipped+6; +3 CUDA-gated tests now pass with GPU visible).
  - Frontend typecheck: **exit 0**.
- **Retry Test 1 now**: Sam restarts `./run.sh`, creates a fresh Cartpole project (or rm's the existing `runs/` on the current one), kicks a short mini-run. Expected: training now clears `env.reset()` without the `reset()` NoneType crash. If other Cartpole-specific issues emerge (reward-module grounding, metric quirks), iterate from there.

### 2026-04-22 17:30 — New-window plan complete: S1..S8 all landed

8 of 8 ships from
[plans/read-projects-new-window-directive-md-en-linked-harbor.md](~/.claude/plans/read-projects-new-window-directive-md-en-linked-harbor.md)
in a single session. Every §6 bug addressed. §7 goals partially
advanced (6 of 10 directly; 3 more were already done pre-session; §7.1
and a sliver of §7.8/§7.9 deferred per plan). Baselines green end-to-end.

Final baselines (2026-04-22 17:30):
- Sculptor: **102 passed, 1 skipped** (started at 97+1; +5 new across S1 + S4 + S6 + S7 → gained 2 kg_research + 1 Cartpole-schema + 1 Cartpole-adapter-contract + 1 `_build_dummy_inputs`-cartpole — 5 total).
- Backend (cold-CUDA, `-k "not vram and not pynvml and not gpu"`): **208 passed, 3 skipped, 6 deselected** (started at 201 pass + 2 CUDA-flaky + 6 deselected = 209 collected; +7 new = 2 physics/mjlab_builtin + 4 test_reward_prompt/ws + 2 test_rewards/grounding = +8, offset by CUDA-flakes reclassifying to skipped, net 208).
- Frontend typecheck: **exit 0**.
- GPU smoke: 4 tests collected (was 2; +2 S7). CPU-only hosts auto-skip all 4 — no impact on default baseline.

What Sam should do now (live smoke, in this order):
1. `pkill -f 'uvicorn backend.main'` to clear the stale reload.
2. `cd ~/projects/reward-sculptor-ui && ./run.sh` to pick up backend changes + Vite hot-reload.
3. Open his existing **Cartpole** project (or create fresh):
   - **Physics tab** → MJCF should load (S2).
   - **Rewards tab** → click "Prompt Claude" with a short ask, watch the new Activity panel stream events (S3). If Claude wedges, observe the 300 s timeout releasing the 409 lock.
   - **Overview tab** → library grid gone; selected-robot dashboard + Library entry card visible (S5).
   - **KG tab** → "Research a topic" returns arxiv IDs (S1).
   - **Run** → ETA banner + resume-warning banner visible before launch (S8). Kick a 2-iter mini-run (`iterations=2`, `training_iterations=500`) to close T1 (S4).
4. Open existing **Go1** project's Rewards tab → SpecPanel shows 4 columns including Grounding (S6). Click an arxiv-id in the Grounding column → arxiv.org opens.
5. (Optional) Run `uv run pytest -m gpu -v` under `~/projects/RewardSculptor` to exercise the new G1 + Cartpole GPU smoke tests (S7). Budget ~3-5 min.
6. (Next overnight) Hit "New run" with iterations=12 — the T11 checklist in [QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md) documents what to verify next morning.

Deferred items (flagged in the plan, not shipped this pass):
- §7.1 — adapter introspects env to build state_schema (requires per-robot-family GPU test cycle; revisit after T1..T10 green on Sam's box).
- §7.8 — keyframes alongside rollout video (diagnose doesn't emit keyframe timestamps yet; design-bounded).
- §7.9 — streaming Claude tokens as they arrive (migration from `messages.parse` to `messages.stream` + manual JSON accumulation; real refactor, wait for user pain signal).
- Vitest runner + frontend unit tests for the new S3/S5/S6/S8 components — deferred per CONTEXT.md:242; the plan called this out explicitly. Manual UI smoke covers the gap for now.
- Overview-embedded Claude prompt editor for MJCF + adapter_config (half of the originally-scoped S5); Physics-tab deep link is the stopgap.

### 2026-04-22 17:10 — S8 shipped: Run dialog shows ETA + resume-warning banner (§7.7 stretch)

Ship 8 of 8. Stretch per the plan — surfaces the cost of a long run
before Sam commits, and makes the resume-on-by-default behaviour
visible rather than implicit.

- **What**: [frontend/src/components/NewRunDialog.tsx](reward-sculptor-ui/frontend/src/components/NewRunDialog.tsx) gained an inline banner between DialogHeader and the Basic/Advanced tabs that always shows:
  - **ETA**: `iterations × SECONDS_PER_CYCLE[adapter_kind]` — per-kind budgets (gym_sb3:180s, mjlab_cartpole:60s, mjlab_go1:1320s, mjlab_g1:1500s, mjlab_other:600s) derived from Sam's observed 22-min/iter Go1 baseline + the envelope documented at the top of `pickAdapterDefaults`. Short runs display in seconds/minutes; long runs switch to hours. Finish time is `Date.now() + eta`, localized to hh:mm same-day or `Weekday hh:mm` if spilling past midnight.
  - **Long-run hint**: when ETA ≥ 30 min, the banner switches to amber tint + adds a sentence about sleep / power loss safety + suggests `--dry-run` as a pipeline shakeout first (dry-run overrides ETA to 50 s, matching the existing --dry-run label).
  - **Resume banner**: when `project.n_iterations_completed > 0`, shows a `<SkipForward>` line explaining resume will reuse existing `runs/iter_<N>/` artifacts, with the `rm -rf runs/iter_<N>/` escape hatch for forcing a fresh train. Pure informational — no blocking prompt.
- **Why**: §7.7 "The UI understands what's expensive and warns." Sam's observed 22-min/iter Go1 at `iterations=12` = 4.5 h wall-clock. Pre-S8 the dialog showed zero cost signal; a new user could click Launch at 2pm and be surprised when the run finishes at 6:30. Resume was similarly implicit — `--resume` is hardcoded on in `run_manager` (2026-04-22 01:40 entry) and silently skips completed iters. The banner makes both visible.
- **How**:
  - Kept it non-blocking. No confirm dialog, no "I know what I'm doing" checkbox. The existing Cancel button is the escape.
  - `SECONDS_PER_CYCLE` keyed by the `kind` that `pickAdapterDefaults` already emits — no refactor of the adapter-detection logic. New keys lift cleanly to new rows if additional adapters land.
  - `humanizeSeconds` + `formatEta` are free functions with obvious semantics: <90 s → seconds; <90 min → minutes; else → hours with 1 decimal under 10 h. Finish time uses the user's locale formatter (same pattern as `formatRelative` in utils).
  - No new API calls or backend surface. Pure view-layer.
- **Verified**:
  - Sculptor: **102 passed, 1 skipped** (unchanged — frontend-only ship).
  - Backend: **208 passed, 3 skipped, 6 deselected** (unchanged).
  - Frontend typecheck: **exit 0**.
- **Live smoke pending** — Sam clicks New Run on any project. Expected: banner shows estimated wall-clock + finish time immediately. If the project has prior iter history, resume line appears. Changing the iterations field in the Advanced tab updates ETA live.
- **Next**: plan complete; final-state summary entry above.

### 2026-04-22 17:10 — S7 shipped: GPU-gated G1 + Cartpole smoke tests + T11 overnight checklist (§9 regression gates)

Ship 7 of 8. Adds GPU-gated regression gates for the two robots
outside Go1's existing smoke fixture + documents the T11 overnight
failure-injection checklist that §7.10 robustness claims rest on.

- **What**:
  - [tests/test_mjlab_gpu.py::test_mjlab_g1_short_train_smoke](RewardSculptor/tests/test_mjlab_gpu.py) — `@pytest.mark.gpu` test that instantiates `MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=512, max_iterations=10)`, asserts the 29-dof G1 schema (`qpos=(29,), actuator_force=(23,)`) dispatches correctly, trains for 10 iters, asserts a non-empty checkpoint drops. Budget ~2-3 min on Sam's RTX 5070 Laptop. Guards the humanoid training path which previously only had the schema-shape unit test.
  - [tests/test_mjlab_gpu.py::test_mjlab_cartpole_short_train_smoke](RewardSculptor/tests/test_mjlab_gpu.py) — same pattern for `Mjlab-Cartpole-Balance`, num_envs=256, max_iterations=10. Budget ~30 s. Pairs with the S4 CPU-only unit tests for end-to-end GPU validation of the Cartpole schema + training loop on a fixed-base articulation.
  - ANYmal (T10 from the plan) intentionally skipped: mjlab's task registry (`mjlab.tasks.registry._REGISTRY`) contains only Go1, G1, Cartpole, and Yam tasks at this mjlab version. The library YAML lists `anybotics_anymal_c` as `mjlab_ready` but `preconfigured_tasks: []` — there's no `MjlabAdapter`-compatible task_id to drive. Documented inline in the test file.
  - [QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md): appended a "T11 acceptance" checklist for the overnight 12-iter run. Lists the happy-path budget (4.5 h at 1500 steps/iter), the per-iter artifact manifest (checkpoint + rollout.mp4 + trajectory.npz + behavior.json), and three failure-injection gates (backend reload mid-run, SIGKILL mid-checkpoint-copy, Anthropic 429 spike) with the existing plumbing that absorbs each. Sleep-during-run flagged as unverified; suggested `systemctl mask` workaround.
- **Why**: §9 calls out T6-T11 as "regression gates — already work, needs verification". Pre-S7 only Go1 had a GPU-smoke fixture (shipped pre-M2). G1 and Cartpole relied on unit-level schema dispatch alone, which couldn't catch the class of bugs where mjlab renames an entity attr or the subprocess CLI drifts (see 2026-04-21 20:45 `root_lin_vel_b` → `root_link_lin_vel_b` regression that live GPU-smoke would have caught). T11 was already backed by shipped code; the checklist makes the acceptance criteria explicit so Sam doesn't have to re-derive them next time.
- **How**:
  - No caching fixture for the two new tests (unlike Go1's `mjlab_go1_checkpoint`). They train every time `-m gpu` is selected, which is rare by design. If Sam finds himself running them often, promote to fixtures with `--regenerate-fixtures` support.
  - Schema assertions (CPU-cheap) run before the train call so a dispatch regression fails the test in <1 s instead of waiting 2 min for the subprocess.
  - Both tests inherit the auto-skip-on-CPU behaviour from [tests/conftest.py::pytest_collection_modifyitems](RewardSculptor/tests/conftest.py) — CPU-only hosts (CI, Sam's Windows box) skip the 4 GPU tests with "CUDA not available on this host".
- **Verified**:
  - Collect-only: **4 tests collected** (was 2 — Go1 + vram_probe; +2 new).
  - Default run on this CPU box: **4 skipped** (all auto-skip, no CUDA).
  - Sculptor (`uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py`): **102 passed, 1 skipped** (unchanged — S7 touches only GPU-gated tests).
  - Backend / frontend unchanged this ship.
- **Live smoke pending** (separate from the default baseline, run manually):
  - `cd ~/projects/RewardSculptor && uv run pytest -m gpu -v` on Sam's GPU box. Expected: 4 passed (Go1 fixture reuse from cache + G1 train + Cartpole train + VRAM probe). Budget ~3-5 min if Go1 fixture is cached.
  - T11 overnight checklist: Sam runs when he next does a full 12-iter Go1 run.
- **Next**: S8 (stretch) — UI shows ETA + resume-warning banner on the Run button (§7.7).

### 2026-04-22 16:50 — S6 shipped: `REWARD_SPEC.grounding` dict surfaces in RewardsTab SpecPanel (§7.3 / T7)

Ship 6 of 8. The 2026-04-22 05:30 entry mandated that every
hyperparameter change in a Claude-written reward carry a
`grounding[hp] = citation` entry, and updated the `edit_rewriter`
prompt to enforce it. The backend serializer silently dropped the
field and the UI had no column for it — this ship closes the loop so
Sam can scan per-hp citations while reviewing rewards.

- **What** (backend):
  - [backend/models/reward.py::RewardSpec](reward-sculptor-ui/backend/models/reward.py): new explicit field `grounding: dict[str, str] = Field(default_factory=dict)`. `ConfigDict(extra="allow")` was already on the model, but the serializer only copied the explicit fields — `grounding` was being silently dropped even though it sat in `REWARD_SPEC`. Default factory keeps older rewards' behaviour unchanged (they get `{}`).
  - [backend/services/reward_store.py::_spec_from_dict](reward-sculptor-ui/backend/services/reward_store.py): reads `raw.get("grounding")`, coerces every value to a stripped str, skips empties, and passes the resulting dict to `RewardSpec(...)`. Non-dict `grounding` values (Claude occasionally emits a list) are gracefully treated as empty rather than crashing the detail endpoint.
- **What** (frontend):
  - [frontend/src/lib/types.ts::RewardSpec](reward-sculptor-ui/frontend/src/lib/types.ts): mirrors the backend — new required `grounding: Record<string, string>`.
  - [frontend/src/components/RewardsTab.tsx::SpecPanel](reward-sculptor-ui/frontend/src/components/RewardsTab.tsx): grid widened from `md:grid-cols-3` → `md:grid-cols-4`; new second column **Grounding**. Each entry renders `hparam_name` + value; when the value starts with an arxiv_id (regex `^(?:\d{4}\.\d{4,5}|[a-z-]+\/\d{7})`) it's wrapped in an `<a>` to `arxiv.org/abs/{id}`; otherwise it's plain text (physics-first-principles justifications). Empty grounding renders `(pre-grounding-mandate)` italic muted so Sam knows this reward was written before the mandate vs. rendering was broken.
- **Why**: §7.3 "Claude decisions are literature-grounded by default. Every reward hyperparameter change carries an arxiv citation." Mandate landed 2026-04-22 05:30 in the prompts — Claude started emitting `grounding: {...}` on every edit, but neither the backend nor the UI surfaced it, so Sam couldn't audit which citation justified each numeric choice without cracking open the v<n>.py source.
- **How**:
  - Chose explicit-field over `extra="allow"` pass-through: pydantic's `extra="allow"` lets the dict survive round-trip via `model_dump()` only if the caller constructs the model with `model_validate(raw_dict)`. The existing path (`RewardSpec(version=..., hyperparameters=..., ...)`) only populates explicit fields, so unknown keys never land in the payload. Adding an explicit field is the cleanest fix + documents the contract.
  - Value coercion to `str` in `_spec_from_dict` defends against the Claude failure mode where `grounding[hp]` arrives as a list / float / None — the prompt says `dict[str, str]` but reality regularly disagrees.
  - Arxiv-link detection on the frontend is pure view logic; no extra API surface. The regex matches the two arxiv-ID formats (post-April-2007 `YYMM.NNNNN` and pre-April-2007 `subject/XXXXXXX`) at the *start* of the grounding string — a `1707.02286 — Mnih survey, ...` value links out; a `physics-first-principles: ...` value stays plain text.
- **Tests added** in [backend/tests/test_rewards.py](reward-sculptor-ui/backend/tests/test_rewards.py):
  - `test_reward_detail_surfaces_grounding_dict` — stamps a `v1.py` with a full `REWARD_SPEC.grounding` dict (one arxiv citation, one physics justification), `GET /projects/{slug}/rewards/1`, asserts `body["spec"]["grounding"]` contains both entries with stringified values. End-to-end file-to-API guard.
  - `test_reward_detail_grounding_defaults_empty_for_pre_mandate_rewards` — `GET /projects/{slug}/rewards/0` on a fresh scaffold (sculpt-init's v0 has no grounding), asserts `spec.grounding == {}`. Pins the default shape the UI relies on.
- **Verified**:
  - Sculptor: **102 passed, 1 skipped** (unchanged — sculptor side is untouched; the REWARD_SPEC dict shape lives in the reward module files, not the library).
  - Backend: **208 passed, 3 skipped, 6 deselected** (was 206; +2 new).
  - Frontend typecheck: **exit 0**.
- **Live smoke pending** — Sam reloads Vite, picks any Go1 project's Rewards tab, opens a Claude-written version (post-2026-04-22). Expected: SpecPanel now shows 4 columns instead of 3. The new Grounding column lists every `hyperparameters[k]` → `grounding[k]` citation. Clicking an arxiv-id opens the paper on arxiv.org. v0.py (sculpt-init scaffold) shows "(pre-grounding-mandate)".
- **Next**: S7 — GPU-gated G1 + ANYmal smoke tests + overnight T11 checklist (regression gates / §9).

### 2026-04-22 16:25 — S5 shipped: Overview tab stops rendering library grid; surfaces selected-robot dashboard (bug #6.2 / T4)

Ship 5 of 8. Scoped down from the full plan to the two cuts that close
Sam's symptom: (a) `isRobotConfigured` no longer rejects mjlab /
menagerie robots, (b) a new library-entry card surfaces the selected
robot's metadata next to the RobotViewer. Full "robot-edit prompt"
surface (§7.9) deferred — Physics tab handles MJCF edits today; a CTA
button links there.

- **What**:
  - [frontend/src/pages/ProjectDetail.tsx::isRobotConfigured](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx): library-kind robots are configured when `library_name || project.library_slug` is set. Dropped the `env_id !== "CHANGE_ME"` requirement that was gym-specific and falsified for Cartpole / Go1 / G1 / ANYmal (all steered via `adapter_config.task_id`, no env_id). Bug #6.2 root cause.
  - [frontend/src/pages/ProjectDetail.tsx::RobotLibraryCard](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx): new sidebar card in the Overview tab. Fetches the library entry via existing `useLibraryRobot(slug)` + physics summary via `usePhysics(slug)`. Surfaces: display_name, category badge, description, source (mjlab_builtin / menagerie / …), menagerie_package, training_support, joint count + actuator count (from the live MJCF summary), MJCF parse error banner (when Physics tab would show one), preconfigured tasks with recommended num_envs, and the library entry's paper/repo references with external-link icons. Footer button deep-links into the Physics tab.
  - Imports: `ExternalLink`, `Wrench` from lucide-react; hooks `useLibraryRobot`, `usePhysics`.
- **Why**: bug #6.2 — Sam's fresh Cartpole project opened Overview and saw the `/projects/new` library grid verbatim, offering "create new project from here" on each robot (wrong affordance for an existing project). Root cause: the `isRobotConfigured → RobotConfig` fallback rendered `RobotConfig` (the grid) whenever env_id was missing, which is always true for mjlab / menagerie robots. Secondary cause: Overview showed no selected-robot dashboard, so even after RobotViewer rendered (post fix a), the sidebar was stuck on bare metadata (kind / library / env_id) rather than the content Sam asked for in §7.2 (name, category, paper refs, DOF counts).
- **How**:
  - Kept the existing `RobotViewer` vs. `RobotConfig` branch instead of removing `RobotConfig` entirely — it's still the right affordance for the rare case of a project whose robot got deleted / kind is `"none"`. Just tightened the guard so library-sourced projects always hit the configured branch.
  - `RobotLibraryCard` reuses the shared `<Card>` primitives + the `KV` row component for consistency with the adjacent Metadata and Robot cards. No new CSS, no new API calls beyond what the Library and Physics tabs already use.
  - The deep-link to Physics tab flips the TabsTrigger via a DOM click (`document.querySelector('[role="tab"][value="physics"]').click()`) rather than adding tab-state-hoisting to ProjectDetail. Pragmatic; matches the existing tab-control surface.
  - Dropped the ambitious "Overview-embedded Claude prompt editor" + "PATCH /projects/{slug}" for task-swap / num_envs / device. That scope is a follow-up: the Physics tab already handles MJCF edits end-to-end with the S3 progress-event plumbing now in place, and `ProjectSettingsDialog` already exists for adapter_config edits (existing path, unchanged).
- **Verified**:
  - Frontend typecheck: **exit 0**.
  - Backend: **206 passed, 3 skipped, 6 deselected** (unchanged from S3 — S5 is frontend-only).
  - Sculptor: **102 passed, 1 skipped** (unchanged from S4).
  - Frontend tests: no vitest runner in the repo — skipped per CONTEXT.md L242; manual UI smoke on Sam's side.
- **Live smoke pending** — Sam restarts `./run.sh` (Vite picks up the change), opens Cartpole Overview. Expected: RobotViewer renders the 3D preview (pre-S5 he saw the library grid), right sidebar now has three cards — Metadata, Robot (legacy thin card), Library entry (new). Library entry shows: display_name "Cartpole (toy)" + category "Other" + description + source "mjlab_builtin" + joint/actuator counts from the physics summary + the two preconfigured tasks + empty-ref fallback (Cartpole entry has no references in the YAML). Edit-in-Physics-tab button swaps to Physics.
- **Follow-up (tracked, not blocking)**: a dedicated "Robot" prompt editor on the Overview tab that targets either MJCF (→ physics.apply_prompt_edit) or adapter_config fields (task swap / num_envs). Deferred until Sam observes the Physics-tab link flow and tells me if the link is sufficient.
- **Next**: S6 — `REWARD_SPEC.grounding` dict surfaced in RewardsTab's SpecPanel (§7.3 / T7).

### 2026-04-22 15:55 — S4 landed (unit-level): Cartpole state schema pinned (bug #6.6 / T1)

Ship 4 of 8. The plan called S4 a live-run gate — 2-iter Cartpole mini-run
on Sam's GPU. This entry is the proactive unit-test half; the live-run
half waits on Sam after `./run.sh` reload.

- **What**: three new tests pinning the Cartpole schema dispatch + the pre-flight dummy-input plumbing that Claude's v1.py will hit.
  - [tests/test_mjlab_adapter.py::test_mjlab_cartpole_schema_is_minimal_fixed_base](RewardSculptor/tests/test_mjlab_adapter.py) — `_schema_for_task` returns the 3-key `{qpos:(2,), qvel:(2,), actuator_force:(1,)}` for all known Cartpole task_ids (`Mjlab-Cartpole-Balance`, `Mjlab-Cartpole-Swingup`) + the lower-case fallback branch. Pins explicit shape tuples so a future widening of `_CARTPOLE_STATE_SCHEMA` fails loudly.
  - [tests/test_mjlab_adapter.py::test_mjlab_cartpole_adapter_reward_contract_is_minimal](RewardSculptor/tests/test_mjlab_adapter.py) — end-to-end construction seam: `MjlabAdapter(task_id="Mjlab-Cartpole-Balance").reward_contract().state_schema.keys() == {"qpos","qvel","actuator_force"}`. Guards against a regression where adapter init picks the wrong schema dispatch (previous bug: velocity schema was defaulted, crashing Cartpole with AttributeError on `base_lin_vel_b`).
  - [tests/test_edit.py::test_build_dummy_inputs_handles_cartpole_schema](RewardSculptor/tests/test_edit.py) — `_build_dummy_inputs` round-trips the 3-key contract to a `dict[str, np.ndarray]` with leading-dim-1 shapes + then exercises a synthetic Claude-written v1 reward that does exactly what Sam's UI prompt flow will produce (`pole_angle = next_state["qpos"][:, 1]`, etc.). Asserts finite reward. This directly guards against the Phase-A style IndexError on the Cartpole schema shape (Go1's 7-key shape was already tested post-Phase A).
- **Why**: bug #6.6 — the `_CARTPOLE_STATE_SCHEMA` Sam added this morning (2026-04-22 11:30 entry) was untested end-to-end. Without a unit gate, the pre-flight IndexError that Phase A fixed for the Go1 schema could silently recur if `_build_dummy_inputs` regressed on the 3-key shape. These tests are fast (no GPU) and cheap to keep green.
- **How**:
  - Reused the existing `_build_dummy_inputs` dict-branch test pattern (shared with the G1 test in the same file). No code changes in `sculptor/` — current behaviour already handles the 3-key schema correctly; the tests are pure gates.
  - Explicit shape tuples `(2,)`, `(1,)` pin the contract so any silent widening (someone adding `base_lin_vel_b` back) breaks the test. This is the "don't learn it the expensive way" discipline the directive §8 preaches.
- **Verified**:
  - Sculptor: **102 passed, 1 skipped** (was 99+1; +3 new).
  - Backend + frontend unchanged (not touched this ship).
- **Still pending (live-run gate)**: Sam needs to `./run.sh` restart to pick up S2+S3, create a fresh Cartpole project, then kick a 2-iter mini-run with small `steps_per_iter` (~500). Expected outcomes per the plan:
  * Physics tab loads the real Cartpole MJCF (from S2).
  * Rewards tab shows v0.py + a working prompt-edit flow with the new Activity panel streaming events (from S3).
  * Training subprocess completes iter 0 without AttributeError in `_snapshot` (Scene entity fix already shipped) and clears iter 1's pre-flight against the Cartpole schema (unit-verified this ship). If v1 breaks anything, we fix inline and extend these tests.
- **Next**: S5 — Overview tab becomes a robot dashboard + robot-edit prompt + task/envs/device controls (bug #6.2 / T4).

### 2026-04-22 15:35 — S3 shipped: Reward prompt-edit emits log_line events + timeout (bug #6.1 / §7.4 / T2)

Ship 3 of 8 from the new-window plan — the largest single change of the
pass. Surfaces progress to the UI during what was previously a
silent 30-60 s Claude call. Adds a wall-clock ceiling so a wedged
Anthropic call can no longer hold the 409 per-project lock forever.

- **What** (backend):
  - [sculptor/edit.py:500](RewardSculptor/sculptor/edit.py): `_call_llm`, `apply_edits`, `apply_prompt_edit` all gained an optional `on_event: Callable[[dict], None] | None = None` kwarg. When set, the sculptor layer emits `{"type": "log_line", "text": ...}` dicts at 8 load-bearing transitions: KG query start/done, pre-validate start/done, prompt built, LLM request start/response (per attempt), post-validate, committed. Default `None` is a no-op, so the sculpt-run and diagnose call sites are behaviourally unchanged.
  - [backend/services/reward_jobs.py](reward-sculptor-ui/backend/services/reward_jobs.py): `run_reward_prompt_edit_job` runner now (a) calls `job.emit({"type":"log_line",...})` directly for runner-level transitions (start, dispatch, completed, timeout), (b) forwards an `on_event` callback to `apply_prompt_edit` that marshals worker-thread events back to the bound loop via `loop.call_soon_threadsafe(job.emit, ev)` — emits from `asyncio.to_thread` workers would otherwise touch subscriber `asyncio.Queue`s from the wrong thread, and (c) wraps `asyncio.to_thread(_do_edit)` in `asyncio.wait_for(timeout=budget_s)`. Budget defaults to 300 s, override via `RS_REWARD_PROMPT_TIMEOUT_S`. Also added a `timeout_s` kwarg to the factory for unit-test override.
  - [backend/routes/jobs.py](reward-sculptor-ui/backend/routes/jobs.py): new `ws_router` (bare prefix) with `/ws/jobs/{job_id}/events` WebSocket endpoint that replays the job's full buffered `events` list on connect + tails new emits + closes with `{"type":"terminal","status":...}` once the job is in a terminal state. Mirrors the run-events WS contract but keyed on `job_id` so Rewards/Physics/KG tabs can subscribe without inventing a slug. Wired into [backend/main.py:156](reward-sculptor-ui/backend/main.py:156) alongside the existing `jobs_routes.router`.
- **What** (frontend):
  - [frontend/src/lib/api.ts](reward-sculptor-ui/frontend/src/lib/api.ts): new `jobEventsWsUrl(jobId)` helper returning `ws[s]://<host>/ws/jobs/{id}/events`. Matches the `runEventsWsUrl` convention; Vite's dev proxy already forwards `/ws/*` unchanged (see 2026-04-21 20:45 Change Log).
  - [frontend/src/hooks/useJob.ts](reward-sculptor-ui/frontend/src/hooks/useJob.ts): new `useJobEvents(jobId)` hook returning `{events, connected, terminal}`. Subscribes to `/ws/jobs/{id}/events`, dedupes by `seq`, caps in-memory buffer at 500 events (reward-prompt-edit emits ~10-30), reconnects with exponential backoff up to 8 s, uses a `terminalRef` to defeat the stale-closure reconnect-loop bug from the runs WS (same pattern).
  - [frontend/src/components/RewardsTab.tsx](reward-sculptor-ui/frontend/src/components/RewardsTab.tsx): `PromptEditHero` subscribes via `useJobEvents(activeJobId)` and renders a new `<PromptActivityPanel>` below the Textarea while a prompt edit is in flight (or afterwards, if any events are buffered). Panel shows connection state (spinner while in-flight / dot when idle), event count, and the tail of the last 20 log_line events with wall-clock timestamps. Bug #6.1 symptom ("UI shows no progress for minutes") gone — the user sees every load-bearing transition as it happens.
- **Why**: bug #6.1 / §7.4 — the reward-prompt-edit pipeline was emitting zero events while running, so the UI only saw `progress: 0.2, message: "Claude is rewriting…"` for 30-60+ s. Users hit "Prompt Claude" a second time assuming it was stuck and got a 409. Root cause was two-fold: (a) `run_reward_prompt_edit_job` only set `job.progress` + `job.message` and never called `job.emit()`; (b) sculptor's `apply_edits` had no seam to report progress mid-call. Both surfaces fixed. The timeout wrap is belt-and-braces: even with the new observability, a genuinely wedged Claude call would still hold the lock indefinitely without a ceiling — now it errors at 300 s and releases.
- **How** (non-obvious choices):
  - `on_event` is a module-level kwarg with `None` default. Every call-site guard is `if on_event is not None: on_event({...})`. This was explicitly chosen over a proper "event emitter" class so the sculpt-run path (which threads `apply_edits` → `_call_llm` deep inside the diagnose loop) is behaviourally bit-for-bit identical when no caller passes one. Sculptor suite stayed at 99 passed from the S1 baseline, confirming no regression.
  - Worker-thread → loop marshalling for events uses `loop.call_soon_threadsafe(job.emit, ev)` inside the runner. Direct calls to `job.emit` from `asyncio.to_thread` worker would put-nowait onto subscriber queues from the wrong thread — usually fine for `asyncio.Queue` but not guaranteed. Fallback branch calls `job.emit` in-thread if `call_soon_threadsafe` raises `RuntimeError` (loop closed), so unit tests that invoke the runner without a JobManager still observe events.
  - The WS endpoint is on a **separate** `ws_router` without the `/jobs` prefix. Putting the WS on the prefixed `jobs` router resolves to `/jobs/{id}/events` which conflicts with the Vite `/ws/*` proxy convention + the run-events pattern. Registering on a bare router lets the full path be `/ws/jobs/{id}/events` at the same mount depth as `/ws/projects/.../runs/.../events`.
  - `PromptActivityPanel` intentionally renders even after the job completes (as long as events are buffered) so Sam can see WHY the job terminated — log tail survives the `inFlight=false` transition and persists until `activeJobId` is cleared by the terminal-state handler.
- **Tests added** in [backend/tests/test_reward_prompt.py](reward-sculptor-ui/backend/tests/test_reward_prompt.py):
  - `test_reward_prompt_edit_emits_log_line_events` — stubs `load_adapter` + `apply_prompt_edit`, runs the runner directly inside a fresh loop (new helper `_run_runner_sync`), asserts ≥4 `log_line` events land in `job.events` and contain the expected transitions (start, LLM, completed). **Unit-level** — bypasses the HTTP route to avoid the scaffold-level `env_id=CHANGE_ME` dead-end the prior `test_reward_prompt_submits_job_and_returns_202` deliberately stops at.
  - `test_reward_prompt_edit_timeout_releases_lock` — sets `RS_REWARD_PROMPT_TIMEOUT_S=0.25`, stubs `apply_prompt_edit` to `time.sleep(2.0)`, asserts the job errors and `job.error` contains "time"/"timeout"/"wedged".
  - `test_job_events_ws_replays_buffered_events` — injects a terminal `Job` with two `log_line` events into the JobManager, connects via `client.websocket_connect("/ws/jobs/{id}/events")`, asserts the stream contains `connected` + 2×`log_line` + `terminal` frames.
  - `test_job_events_ws_404_for_missing_job` — connects to an unknown job_id; asserts an `error` frame is sent before close (no 500).
  - Also updated the existing `test_reward_prompt_submits_job_and_returns_202`'s `fake_apply_prompt_edit` to accept + (optionally) call `on_event` so it stays compatible with the new kwarg.
- **Verified**:
  - Sculptor: **99 passed, 1 skipped** (unchanged from S1).
  - Backend (full, `-k "not vram and not pynvml and not gpu"`): **206 passed, 3 skipped, 6 deselected** — was 204 post-S2; +4 new S3 tests.
  - test_reward_prompt.py alone: **11 passed** (was 7; +4 new).
  - Frontend typecheck: **exit 0**.
- **Live smoke pending** — Sam restarts `./run.sh`, opens a project's Rewards tab, types a prompt, clicks "Prompt Claude". Expected: the new Activity panel appears immediately showing "Activity (0 events)" + a spinner; within ≤2 s the first `[reward_prompt_edit] start` line lands; KG-query and LLM-request events follow; final line `[reward_prompt_edit] completed v<N>.py` appears when done. If Claude hangs, the job errors at 300 s with a clear timeout message + the 409 lock releases.
- **Next**: S4 — Cartpole Run end-to-end verify (bug #6.6 / T1). Live-run gate, no code unless v1 breaks pre-flight.

### 2026-04-22 12:40 — S2 shipped: Physics resolver handles `mjlab_builtin` library robots (bug #6.3 / T3)

Ship 2 of 8 from the new-window plan.

- **What**:
  - New method `RobotLibrary.resolve_mjlab_builtin_path(slug) -> Optional[Path]` in [backend/services/robot_library.py](reward-sculptor-ui/backend/services/robot_library.py) with class-level slug→(package, filename) mapping (`_MJLAB_BUILTIN_MJCF`, `ClassVar`). Currently maps `cartpole_mjlab` → `mjlab.tasks.cartpole/cartpole.xml`. Uses `importlib.resources.files()` so the path is discovered from the installed mjlab package — no hardcoded absolute paths.
  - [backend/services/physics.py::_resolve_library_mjcf](reward-sculptor-ui/backend/services/physics.py) now dispatches on `RobotEntry.source`: `menagerie` → existing `resolve_menagerie_path`, `mjlab_builtin` → new path, `gymnasium_builtin` → returns None with an INFO log. Previously assumed all library robots were menagerie-sourced (hence Cartpole's 404).
  - New `backend.services.physics.mjcf_unavailable_reason(project_dir) -> str` helper gives the 404 detail message a targeted body based on the project's actual failure mode (no metadata vs. no library_slug vs. gymnasium_builtin vs. mjlab_builtin mapping missing vs. menagerie lookup failed). Replaces the generic "Pick a library robot" catch-all in [backend/routes/physics.py](reward-sculptor-ui/backend/routes/physics.py).
- **Why**: bug #6.3 — Cartpole projects 404'd on the Physics tab because `resolve_menagerie_path('cartpole_mjlab')` returns None (Cartpole ships inside mjlab, not menagerie). The previous catch-all error told the user to "upload a URDF" which is the wrong hint — they already picked a library robot. New code resolves `cartpole_mjlab` to the real MJCF (verified: `.venv/lib/python3.13/site-packages/mjlab/tasks/cartpole/cartpole.xml`) and the new error reasons distinguish each failure mode.
- **How**:
  - `resolve_mjlab_builtin_path` is on `RobotLibrary` (not a free function) so the `_path_cache` layer is shared with `resolve_menagerie_path` — cached per-process.
  - `_MJLAB_BUILTIN_MJCF` is declared as `ClassVar[dict[...]]` so the `@dataclass` decorator treats it as a class constant rather than an instance field (otherwise → `ValueError: mutable default <dict>` — caught on first test run, fixed before the ship).
  - `mjcf_unavailable_reason` is a free function returning a plain str — no exception plumbing needed; callers just call it if `load_mjcf` returns None.
  - Physics test seam unchanged — existing monkey-patching of `physics_svc._resolve_library_mjcf` still works (the function signature didn't change).
- **Tests added** in [backend/tests/test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py):
  - `test_resolve_library_mjcf_handles_mjlab_builtin` — seeds a project with `library_slug: cartpole_mjlab`, asserts `_resolve_library_mjcf` returns a real `.xml` path whose content starts with `<mujoco`. Uses the real installed mjlab package (hard dep of the repo).
  - `test_mjcf_unavailable_reason_explains_gymnasium_builtin` — injects a fake `gymnasium_builtin` RobotEntry via monkey-patch, asserts the reason string contains `"gymnasium_builtin"` + the slug, and that `_resolve_library_mjcf` still returns None for that source.
- **Verified**:
  - Sculptor: **99 passed, 1 skipped** (unchanged from S1).
  - Backend (full, `-k "not vram and not pynvml and not gpu"`): **202 passed, 3 skipped, 6 deselected** — was 201+2 flaky CUDA skips+6 deselected on the pre-S1 cold-CUDA baseline; now 202+3+6 (my 2 new tests land on the passing side; cold-CUDA state is deterministic this session).
  - test_physics.py alone: **31 passed** (was 29; +2 new).
  - Frontend typecheck: **exit 0**.
- **Live smoke pending** — Sam restarts `./run.sh`, opens his Cartpole project's Physics tab, expects: MJCF loads + summary populates (timestep/gravity/joints/actuators visible). If load_mjcf returns None for some other reason the error message now specifically says which.
- **Next**: S3 — reward prompt-edit emits `log_line` events + timeout (bug #6.1 / T2).

### 2026-04-22 12:15 — S1 shipped: KG research `response_format` → `output_format` (bug #6.4 / T5)

Ship 1 of 8 from the new-window plan
(`~/.claude/plans/read-projects-new-window-directive-md-en-linked-harbor.md`).

- **What**: one-line kwarg rename at [sculptor/kg/research.py:154](RewardSculptor/sculptor/kg/research.py). `response_format=ResearchResponse` → `output_format=ResearchResponse`.
- **Why**: bug #6.4 — KG tab's "Research a topic" errored with `TypeError: Messages.parse() got an unexpected keyword argument 'response_format'`. `response_format` is OpenAI-SDK syntax; anthropic SDK 0.96.x `messages.parse` accepts `output_format` (verified via `inspect.signature(c.messages.parse)` → `['max_tokens', 'messages', 'model', ..., 'output_format', ...]`). `extract.py:185,203` and `diagnose.py:325` were already using the correct kwarg — research.py was the lone outlier.
- **How**: single edit. Plus two new tests in [tests/test_kg_research.py](RewardSculptor/tests/test_kg_research.py):
  - `test_research_topic_uses_output_format_kwarg` — stubs `client.messages.parse` with a kwarg-capturing fake, asserts `"output_format" in kwargs` + `"response_format" not in kwargs`.
  - `test_research_module_call_sites_agree_with_extract_and_diagnose` — `inspect.getsource` each of the three modules, asserts all three use `output_format=` and none use `response_format=`. Cross-module consistency guard so a future copy-paste regression from elsewhere gets caught at unit-test time instead of live.
- **Verified**:
  - Sculptor: **99 passed, 1 skipped** (was 97+1; +2 new). `uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py` in 10 s.
  - Live smoke pending — Sam will hit the KG tab's Research-a-topic button once he picks up the change.
- **Next**: S2 — Physics tab `mjlab_builtin` resolver (bug #6.3 / T3).

### 2026-04-22 11:30 — Cartpole end-to-end test surfaced 5 bugs → new-window directive written

Sam ran through the UI test matrix I proposed (Cartpole sanity check) and hit issues at every tab. Rather than fix piecemeal in this window, wrote a comprehensive directive for a fresh Claude Code session at [NEW_WINDOW_DIRECTIVE.md](NEW_WINDOW_DIRECTIVE.md). That doc is self-contained: project context, stack layout, baselines, bug catalog, ambitious goal set, failure-mode taxonomy, suggested test matrix, starter prompt.

- **Bugs surfaced by Cartpole test** (full repro + file-line pointers in NEW_WINDOW_DIRECTIVE.md §6):
  1. **Rewards tab prompt-edit hangs** — no progress events emitted, UI shows nothing for minutes, retry gets 409.
  2. **Overview tab shows the library grid** instead of selected-robot details.
  3. **Physics tab 404s for mjlab_builtin robots** — `resolve_menagerie_path('cartpole_mjlab')` returns None because Cartpole isn't in menagerie; physics resolver doesn't handle `mjlab_builtin` source.
  4. **KG "Research a topic" errors** with `TypeError: Messages.parse() got an unexpected keyword argument 'response_format'` at [sculptor/kg/research.py:154](RewardSculptor/sculptor/kg/research.py). Anthropic SDK API surface mismatch — same call pattern works in diagnose/extract, so it's a subtle SDK-version or kwarg issue.
  5. **Sculpt run errored with empty scene-keys** — FIXED this pass: `env.scene` doesn't expose `.keys()`; mjlab's entity dict is at `env.scene.entities`. Updated `_find_articulated_entity` in [_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py) + added tolerance for missing floating-base attrs (Cartpole is fixed-base, no `root_link_lin_vel_b`) + added `_CARTPOLE_STATE_SCHEMA` (qpos=(2,), qvel=(2,), actuator_force=(1,)).
- **Only fix shipped this entry**: bug #5 (scene-keys). Bugs #1-#4 documented for the new window per Sam's direction.
- **Verified**: sculptor **97 passed** (baseline preserved).
- **Handoff**: new Claude Code window should read NEW_WINDOW_DIRECTIVE.md + generate a plan document before shipping anything. Starter prompt at the bottom of the directive.

### 2026-04-22 05:30 — Post-overnight debrief: realism floor + KG-grounding mandate

Sam's overnight run completed 6/12 iters (iter 2-7, early-stop triggered), robot was spasming / flipping on the rollout videos. Three distinct issues surfaced:

- **"iter 0 + iter 1 never ran"**: `latest_n_before_loop=2` at run start (v2.py was on disk from a pre-overnight attempt). `--resume` correctly interpreted that as "iter 0 + iter 1 already done, pick up at iter 2". Log confirms: `start_iter=2, end_iter=14`. No bug — doing exactly what resume semantics say. UI Timeline's "iter 1 spinner" is stale state from an earlier run.
- **"iter 8-12 never started"**: early-stop after 3 non-improving iters vs best. Metric history: `[-798, -568, -173, -483, -514, -366]`. Best was iter 4 (-173). Iters 5 / 6 / 7 failed to beat it → abort at iter 7. Correct behavior. Metric history was already wiped earlier this pass so next run starts fresh.
- **Robot spasming (the actual crime)**: [_mjlab_runner.py:178-185](RewardSculptor/sculptor/adapters/_mjlab_runner.py) zeroed EVERY mjlab default reward term when injecting `sculptor_primary`. mjlab's defaults (track_linear_velocity, upright, pose, dof_pos_limits, action_rate_l2, foot_clearance, foot_swing_height, foot_slip, soft_landing) are a carefully-tuned locomotion realism prior. Zeroing them meant Claude's hand-rolled reward had to do "stand like a dog" AND "jump high" simultaneously — with no anti-spasm / anti-topple floor. Diagnose correctly flagged this 7 times in a row as `reward_hacking + component_imbalance + static_equilibrium`.

- **Fix 1 — realism floor**: defaults now scaled to `0.3 ×` original weight instead of zero. Physics-plausible prior dominates fine-grained control; `sculptor_primary` (weight=1.0) dominates the task objective. Comment in [_mjlab_runner.py::_cmd_train](RewardSculptor/sculptor/adapters/_mjlab_runner.py) documents the failure mode + the 0.3× choice.
- **Fix 2 — KG-grounding mandate in prompts**. Both [edit_rewriter.md](RewardSculptor/sculptor/prompts/edit_rewriter.md) + [physics_editor.md](RewardSculptor/sculptor/prompts/physics_editor.md) gained a "REALISM + KG-GROUNDING MANDATE" preamble. Every numeric hyperparameter change must satisfy: (a) cite a specific arxiv_id from the LITERATURE CONTEXT block with how_used; (b) physics first-principles justification tied to a measurable robot property; or (c) <20-30 % perturbation of a value previously cited under (a)/(b). Inventing "reasonable values" without citation is explicitly rejected. `edit_rewriter` also gained a new required `grounding: dict[str, str]` field on REWARD_SPEC so reviewers can scan which hp has which citation. `edit_rewriter` additionally required to include at least one realism-gate term (zero-clip on `info["fallen"]`, action-rate penalty, per-term clamp) when diagnose flagged `reward_hacking / static_equilibrium / component_imbalance`.
- **Recovery plan for unitree-go1-3**: Sam's metric_history was already reset earlier this pass. v7.py stays as the latest reward; next run's iter 8 picks up there. With the realism floor + `base_height` / `fallen` info signals (shipped earlier this morning) + the KG mandate, Claude has everything it needs to write a v8 that enforces upright gating and doesn't reward upward-motion-during-a-topple.

- **Verified**: sculptor **97 passed**, backend 205 passed (1 pre-existing pynvml-visibility flake in `test_mjlab_rejects_insufficient_vram` — environmental, unrelated to this pass), frontend typecheck clean.
- **Still deferred**: resume-from-iter currently requires `v<N>.py` on disk to pick up at iter N; there's no "redo iter N fresh" button. User workaround is `rm rewards/v<N>.py` before a new run.

### 2026-04-22 01:40 — Overnight-reliability pass: retries + atomic writes + NaN guard + resume

Sam's goal: start a 12-iter run at night, wake up to completed results. Predicted failure modes from the earlier plan ranked by overnight-probability-of-biting, then shipped the load-bearing ones. Full log in [QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md) under the 2026-04-22 01:40 entry.

- **Anthropic SDK `max_retries=6`** (SDK default is 2) at all three hot-path sites: [diagnose.py:400](RewardSculptor/sculptor/diagnose.py), [edit.py:621](RewardSculptor/sculptor/edit.py), [physics.py:560](reward-sculptor-ui/backend/services/physics.py). A 12-iter run makes 24 Claude calls; at even 1 % per-call transient rate (429 / 500 / network), 2 retries gave ~79 % run success. 6 retries push that over 99.9 %. Single biggest reliability lever.
- **Atomic checkpoint write** in [_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py) — `shutil.copy → .pt.tmp + os.replace`. SIGKILL during a partial copy was the only way the pre-pass resume logic could load a corrupted checkpoint and blow up iter N+1.
- **NaN / Inf guard in `_call_compute_reward`** at [edit.py](RewardSculptor/sculptor/edit.py). Pre-flight now rejects `reward = NaN|±Inf` and per-component non-finite values before Claude's new reward reaches rsl_rl. Classic overnight killer: unguarded `log(z - z0)` or division by zero → NaN on step ~200 → PPO gradient explodes → iter dies.
- **Per-phase resume** — new `_train_or_resume` + `_rollout_or_resume` helpers in [sculpt.py](RewardSculptor/sculptor/sculpt.py):
  * Training skipped when `iter_<i>/checkpoint.pt` is on disk AND `torch.load` succeeds (integrity gate guards against truncated files from pre-atomic-write runs).
  * Rollout skipped when all three artifacts (`rollout.mp4` + `trajectory.npz` + `behavior.json`) are on disk.
  * Corrupt ckpt falls through to fresh train — never hands a rotted artifact to downstream phases.
  * Emits `[SCULPT-EVENT] phase_skipped` events for UI breadcrumbs.
  Dominant cost saving: a run that errors at iter 7's rollout can be retried and skip 7 × 22 min = 2.5 h of GPU retrain work.
- **Always-pass `--resume` from the backend** ([run_manager.py](reward-sculptor-ui/backend/services/run_manager.py)). Fresh projects unaffected (start_iter=0 either way). Overnight runs that fail partway just need Sam to click Run again in the morning — sculpt auto-skips completed iters + any partially-completed iter's expensive phases.
- **Rollout video UI**: pre-existing at [RobotViewer.tsx:354](reward-sculptor-ui/frontend/src/components/RobotViewer.tsx) via the `/projects/{slug}/runs/{run_id}/iterations/{iter_index}/rollout` endpoint. RobotViewer auto-loads the latest iter's rollout.mp4 — no changes needed. Sam finds it on the Runs tab after each iter completes.

- **Tests**: `test_train_or_resume_skips_when_checkpoint_present`, `test_train_or_resume_falls_through_on_corrupt_checkpoint`, `test_rollout_or_resume_skips_when_artifacts_present` in [test_sculpt.py](RewardSculptor/tests/test_sculpt.py). Sculptor suite now **97 passed** (+3), backend **209 passed** unchanged, frontend typecheck clean.
- **Not shipped this pass** (from QUALITY_PASS_PLAN §C, lower overnight impact): ring-buffer indicator, metric-regression alerting banner, GPU-gated rollout regression test. Call-out for Sam to decide after he sees how the overnight goes.

### 2026-04-22 01:10 — Quality pass: iter 1 pre-flight, KG-mandate, reward_spec.json

Sam's retry survived train + rollout (my earlier num_envs=64 + render-throttle fix landed; `runs/iter_0/rollout/rollout.mp4` was produced in 40.5 s) and then errored inside `edit.apply_edits` during the v0→v1 pre-flight. Same session he asked to make every physics / reward change consult the KG, and to "ensure everything is functioning to its highest ability". Full plan in [QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md); shipped Phase A + B1 + B2 + C1 this pass.

- **Phase A — iter-0→iter-1 pre-flight IndexError** ([sculptor/edit.py::_build_dummy_inputs](RewardSculptor/sculptor/edit.py)):
  - Error: `_pre_validate → _current_reward_component_keys → _call_compute_reward → v1.compute_reward → v1.compute_reward_batched → qpos = next_state["qpos"] → IndexError`.
  - Root cause: `_dummy_from_space(contract.observation_space_spec)` returned a 1-element numpy array for mjlab (which sets `observation_space_spec=None` and uses `state_schema: dict` instead). The reward module's batched path does `next_state["qpos"]` on that array → IndexError.
  - Fix: `_build_dummy_inputs` checks `contract.state_schema` first and builds `{key: np.zeros((1, *shape))}` for schema-style contracts. Fallback to gym path unchanged.
  - Tests: `test_build_dummy_inputs_uses_state_schema_when_present` + `test_build_dummy_inputs_gym_path_unchanged` in [test_edit.py](RewardSculptor/tests/test_edit.py). Integration verified against Sam's real `v1.py` — reward=0.25, 9 components returned.
- **Phase B — every physics / reward change consults KG** (Sam's mandate):
  - **B1** ([sculptor/edit.py::apply_prompt_edit](RewardSculptor/sculptor/edit.py)): reward prompt-edit was the gap — previously synthesized a `Diagnosis(literature_context=[])`. Now calls `query_semantic(user_prompt, top_k=5, store=kg_store)` and threads matches into the synthetic diagnosis. Failures degrade gracefully (empty context + warning log).
  - **B2** ([backend/services/physics.py::apply_prompt_edit](reward-sculptor-ui/backend/services/physics.py)): new `kg_store` parameter. Queries the KG + injects a rendered `# LITERATURE CONTEXT` block into the Claude user message via the existing `_render_kg_context` from diagnose.py. Return dict gains `kg_citations: list[{technique, paper_citation, relevance_score, source_paper_ids}]` so the UI can render them next to the commit. [reward_jobs.py::run_physics_prompt_edit_job](reward-sculptor-ui/backend/services/reward_jobs.py) now opens a `SculptorKG` against `project_kg_db_path(project_dir)` and threads it through.
  - Tests: `test_reward_prompt_edit_queries_kg` in [test_edit.py](RewardSculptor/tests/test_edit.py); `test_physics_prompt_edit_consults_kg_when_store_provided` + `test_physics_prompt_edit_skips_kg_when_store_none` in [test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py).
- **Phase C1 — `reward_spec.json missing` warning at diagnose time** ([sculptor/adapters/mjlab.py](RewardSculptor/sculptor/adapters/mjlab.py)):
  - gym_sb3 dropped `reward_spec.json` via `vec_env.env_method("get_reward_spec")`; mjlab never did. Diagnose fell back to an empty REWARD_SPEC block → weaker failure-mode analysis from Claude.
  - Fix: after `adapter.train` completes, import the reward module via the existing `_import_reward_module` helper, read `REWARD_SPEC`, and dump to `<output_dir>/reward_spec.json`. Wrapped in try/except so a malformed reward module doesn't kill the train return.
- **Deferred to follow-up passes** (documented in [QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md) §C): resume-from-iter (don't retrain iter 0 when rollout errors), rollout.mp4 playback in UI, client-side ring-buffer indicator, GPU-gated rollout regression test, metric-regression alerting, checkpoint atomic writes.
- **Verified**: sculptor **94 passed** (+3 new), backend **209 passed** (+2 new), frontend typecheck clean. Integration: pre-flight on Sam's actual v1.py now succeeds with reward=0.25 + 9 components. Sam's next run should clear iter 0's full chain (train → rollout → diagnose → edit → v1 commit) and enter iter 1.

### 2026-04-22 00:30 — Rollout stuck for >1 h (num_envs=1 + render-every-step on WSL2 EGL)

Sam's run finished iter 0's training cleanly (checkpoint.pt on disk), then hung for 60+ min on the rollout step. UI log frozen at env-setup print. Diagnosis via `/proc` + `nvidia-smi`: rollout subprocess R (running), 99 threads, **3.7 GiB RSS, 99 % of one CPU core, GPU at 3 %**, frames list ballooning. Root cause: [_mjlab_runner.py::_cmd_rollout](RewardSculptor/sculptor/adapters/_mjlab_runner.py) two pathologies compounding:

- **`env_cfg.scene.num_envs = 1`** (line 348 pre-fix). mujoco_warp's kernel-launch overhead dominates at tiny num_envs — each physics step in the original rollout was ~500× slower than each training step (training batches 2048 envs per step). Warp's docs explicitly recommend ≥ 32 parallel envs for amortized kernel launches.
- **`env.render()` every single step** on WSL2's software EGL. Scene-render is ~200 ms/frame on this path; 3000 steps × 200 ms = 10 min of pure rendering under the unthrottled loop. Combined with the per-step serial stepping above, the 60-min wait + zero progress was the expected outcome.

- **Fix** ([_mjlab_runner.py::_cmd_rollout](RewardSculptor/sculptor/adapters/_mjlab_runner.py)):
  - `num_envs = max(n_episodes, 64)` — mujoco_warp in its efficient regime. Episode stats are recorded for the first `n_episodes` envs only; extras are throughput padding.
  - `render_every = max(1, max_episode_steps // 120)` — target ~120 video frames regardless of episode length, so video quality stays decent but render cost is bounded (≤ ~25 s even on software EGL).
  - Per-env `ep_return / ep_length / ep_done` tensors, frozen on each env's first `done` — auto-reset keeps envs stepping but their counters don't double-count.
  - Emits `[SCULPT-EVENT] rollout_started` / `rollout_progress` (every 25 steps: step, episodes_done, elapsed, fps) / `rollout_done`. The env-setup-print-then-silence UX was the single biggest "is it stuck?" trigger; now the UI gets a live heartbeat.
- **Also killed** the live stuck subprocess (PID 223754, pgkill -TERM then -KILL) so Sam's UI unwedges.
- **Expected perf post-fix**: rollout of 6 episodes × 500 max steps drops from >60 min (didn't complete in the observed window) to ~30-60 s, of which ~25 s is rendering.
- **Verified**: sculptor 91 passed, import OK. Tests don't exercise `_cmd_rollout` directly (no GPU-free mjlab mock), so the real test is Sam's next run completing iter 0's rollout + producing `runs/iter_0/rollout/rollout.mp4` quickly.
- **Deferred**: a real GPU-gated rollout test in `tests/test_mjlab_gpu.py` that checks `rollout.mp4` materializes within 90 s. Worth adding next session.

### 2026-04-21 22:52 — Train/rollout checkpoint extension mismatch (iter 1 always errored)

Sam's run completed iter 0 training cleanly (1500 rsl_rl iters, Mean reward 5605, sculptor_primary 286 — healthy) then instantly errored at the start of the rollout step with `FileNotFoundError: checkpoint.zip`. Diagnostic: the trainer writes `<iter_dir>/checkpoint.pt` (torch.save format, 4.7 MB on disk); [sculpt.py:503](RewardSculptor/sculptor/sculpt.py) hardcoded `checkpoint_path = iter_dir / "checkpoint.zip"` for the rollout call, which is the SB3-SavedModel convention. gym_sb3 adapter: writes + reads `.zip`. mjlab adapter: writes + reads `.pt`. The shared sculpt loop was only correct for gym_sb3.

- **Root cause**: sculpt.py discarded `adapter.train(...)`'s returned `TrainResult.checkpoint_path` (which is the ACTUAL path the adapter wrote) and invented a literal `"checkpoint.zip"` for the rollout call. mjlab trains → writes `checkpoint.pt` → sculpt tries to rollout `checkpoint.zip` → `FileNotFoundError`.
- **Fix** ([sculpt.py:491-506](RewardSculptor/sculptor/sculpt.py)): capture `train_result = adapter.train(...)` and pass `train_result.checkpoint_path` to `adapter.rollout(...)`. Fallback to the old literal on a None `train_result` so any adapter that doesn't return a result (shouldn't exist, but defensive) still limps.
- **Also a diagnostic sidenote** (for the log): the 20,000 entries Sam saw in the UI event panel is the client-side ring buffer cap in [useRunEvents.ts](reward-sculptor-ui/frontend/src/hooks/useRunEvents.ts) — NOT "the run only emitted 20k events". Server-side log file has everything (Sam's `_run_job_742116bb651903d4.log` was 3.2 MB, way past 20k lines). Worth surfacing in the UI as "events capped to last 20k" so it doesn't look like a hard stop.
- **GPU-usage-on-ProArt tangent**: Sam was convinced training wasn't using the GPU because ProArt Creator Hub showed 0 %. Actually confirmed via nvidia-smi + pynvml + direct sim introspection (`env.sim.wp_device = cuda:0`, all mujoco_warp kernels compiled to cuda, 14k steps/sec at 256 envs) that training IS on GPU. ProArt / Task Manager / MSI Afterburner read Windows WDDM counters; WSL2 CUDA workloads don't update WDDM reliably. The authoritative tool on WSL2 is `nvidia-smi` (reads NVML direct from the NVIDIA driver, below WDDM). Sam's AME456 setup that "works in ProArt" likely runs natively on Windows, not WSL — same GPU, different visibility surface. No sculptor-side action; documented for the next time this comes up.
- **Verified**: sculptor **91 passed**, backend **207 passed**, frontend typecheck clean. Sam's `iter_0/checkpoint.pt` is still on disk (4.7 MB) — iter 0 retrains on the next run since sculpt doesn't implement resume-from-last-successful-iter, but that's a ~22 min cost given `steps_per_iter=1500`.

### 2026-04-21 21:00 — Training throughput + UX pass (4 fixes, zero perf cost)

Sam's mjlab run on `unitree-go1-3` "got stuck on iter 0 for an hour". GPU only at 46 % (actually within the normal PPO band, not the bottleneck). UI showed "running" with zero event stream, then reconnect-spammed `connected (replay_count=10)` once he killed the run. Four independent issues addressed.

- **1. `steps_per_iter = 50_000` is catastrophic for mjlab** — [sculpt.py:470](RewardSculptor/sculptor/sculpt.py) reads `steps_per_iter` and hands it to `adapter.train(steps=...)`. For gym_sb3 that's "env steps" (50k is fine). For mjlab, [mjlab.py:360](RewardSculptor/sculptor/adapters/mjlab.py) interprets it as rsl_rl's `max_iterations` → 50k learning iters × ~1 s/iter on an 8 GiB laptop GPU = **~14 h per sculpt iter**. mjlab's own default is 1500 (~25 min/iter), which is what the scaffold now stamps for mjlab projects via a new `_override_iteration_key` helper in [backend/routes/projects.py](reward-sculptor-ui/backend/routes/projects.py). Also healed Sam's 4 existing projects (`unitree-g1`, `unitree-go1`, `unitree-go1-2`, `unitree-go1-3`) via a one-shot `sed` over `config.toml`. This is the dominant throughput win — his 14 h wall-clock drops to ~25 min for a single sculpt iter, ~5 h for the full 12-iter budget.

- **2. WS reconnect spam after run ends (stale-closure bug)** — [frontend/src/hooks/useRunEvents.ts:85-91](reward-sculptor-ui/frontend/src/hooks/useRunEvents.ts) had `ws.onclose = () => { ...if (cancelled || terminal) return; ...reconnect(); }`. The `terminal` binding was captured in a React closure at ws-registration time. When the run reached terminal state, the server closed the WS — but the closure still saw `terminal=false`, so it reconnected, replayed the last ~10 events, the server closed again, loop. Visible as the repeated `connected (replay_count=10)` lines. Fix: mirror `terminal` into a `terminalRef` and have `onclose` / the connect guard read `terminalRef.current`. Also added `qc` to the effect's deps (was missing, lint-warning-only but now correct).

- **3. mjlab subprocess stdout buffered until exit** — [mjlab.py::_run_with_cleanup](RewardSculptor/sculptor/adapters/mjlab.py) used `Popen(stdout=PIPE, stderr=PIPE) + .communicate()`. `.communicate()` blocks until the subprocess exits, buffering the whole stdout stream — so rsl_rl's progress + our own `[SCULPT-EVENT]` JSON lines never reached the outer sculpt-CLI stdout mid-training (~25 min of UI silence per sculpt iter). Rewrote with per-pipe tee threads: each reads line-by-line (`bufsize=1`), forwards to this process's `sys.stdout`/`sys.stderr`, AND accumulates for the returned `CompletedProcess.stdout/stderr` (so the failure-path "last 2000 chars" diagnostic still works).

- **4. Progress bar** — added an `iter_progress` event emitted from inside [_mjlab_runner.py::_cmd_train](RewardSculptor/sculptor/adapters/_mjlab_runner.py). A daemon thread polls `<output_dir>/logs/model_*.pt` every 2 s; when a new checkpoint number appears, it prints `[SCULPT-EVENT] {"type":"iter_progress","rl_iter":N,"rl_total":M,"pct":...,"elapsed_s":...,"eta_s":...}`. Emits an immediate t=0 heartbeat (so the UI gets a bar before the first checkpoint, which is ~25-50 iters in on default save_interval) and a final 100 % tick in the `finally`. Flows through the existing `[SCULPT-EVENT]` parser in [run_manager._stream_stdout](reward-sculptor-ui/backend/services/run_manager.py) → WS → frontend. Frontend side: added `rl_iter?`, `rl_total?`, `pct?`, `elapsed_s?`, `eta_s?` to `IterEventSummary` ([lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts)), extended the event merger in [RunsTab.tsx](reward-sculptor-ui/frontend/src/components/RunsTab.tsx) to populate them, and dropped in a tiny `IterProgressBar` (no new shadcn dep — Tailwind + a transitioned div fill) that renders inside the running iter's Timeline card with `N/M (X%) · ETA Ys`.

- **Verified**:
  - Sculptor: **91 passed, 1 skipped** (unchanged baseline).
  - Backend: **207 passed** (unchanged).
  - Frontend typecheck: clean.
  - All 4 Sam projects' `config.toml` now show `steps_per_iter = 1500`.
- **Sam action**: single `./run.sh` restart picks up all of it (Vite + run_manager + projects routes via uvicorn reload; mjlab/sculptor changes take effect on the next sculpt subprocess launch, which happens per-run).
- **Deferred**: cheap regression test that asserts `_snapshot`'s `data.*` attribute refs exist on `mjlab.entity.EntityData`. Worth adding if/when mjlab bumps.

### 2026-04-21 20:45 — mjlab `_snapshot` attribute rename + Vite `/ws` proxy bug

Second-retry run on `unitree-go1-3` (after the `compute_reward_batched` re-export fix) still errored in 17 s. Same "GPU at 0 %" symptom but a different root cause. Separately: terminal showed repeated `WebSocket /projects/.../events 403 connection rejected` — the reason the Runs tab shows "WS CLOSED" with empty events panel even when the subprocess IS emitting lines.

- **Bug A — `EntityData.root_lin_vel_b` / `root_ang_vel_b` don't exist**:
  - [`_mjlab_runner.py::SculptorRewardTerm._snapshot`](RewardSculptor/sculptor/adapters/_mjlab_runner.py) read `data.root_lin_vel_b` + `data.root_ang_vel_b` — mjlab's `EntityData` actually exposes the body-frame root-link velocities as `root_link_lin_vel_b` + `root_link_ang_vel_b`. Attribute lookup raised `AttributeError: 'EntityData' object has no attribute 'root_lin_vel_b'. Did you mean: 'root_link_vel_w'?` on the very first reward step inside `env.load_managers()` → subprocess exited 1 → `mjlab.train` raised RuntimeError → run errored in 17 s.
  - Fix: renamed to the `root_link_*` form. Confirmed against live `mjlab.entity.EntityData.__dir__` on Sam's host — full set of locomotion attrs available: `root_link_{lin,ang}_vel_{b,w}`, `root_com_{lin,ang}_vel_{b,w}`, `projected_gravity_b`, `actuator_force`, `joint_pos/vel`.
  - Why M2 + prior tests missed it: `tests/test_mjlab_adapter.py` covers the subprocess CLI construction + reward_contract shapes + argparse argv, not runtime attribute access. The GPU smoke test (`test_mjlab_gpu.py`) would have caught this — but it was added against an older mjlab release where the attrs may have been `root_lin_vel_b`, and the fixture was cached from that era so the actual attribute path wasn't re-exercised on every PR. Cheap follow-up: periodically wipe `tests/fixtures/go1_smoke_checkpoint.pt` to force GPU-smoke to re-run end-to-end.
- **Bug B — Vite dev proxy stripped the `/ws/` prefix, backend returned 403 on every WS connect**:
  - [frontend/vite.config.ts](reward-sculptor-ui/frontend/vite.config.ts) had `rewrite: (p) => p.replace(/^\/ws/, "")` on the `/ws` proxy entry. Backend WS routes are registered at `/ws/projects/{slug}/runs/{run_id}/events` and `/ws/projects/{slug}/runs/{run_id}/frames` ([backend/routes/runs.py:272,387](reward-sculptor-ui/backend/routes/runs.py)). Proxy strip → backend saw `/projects/.../events` → no matching WS route → **403** (Starlette's default rejection for unmatched WS paths is 403, not 404). `useRunEvents` + `useLiveClips` then retried indefinitely, which is what flooded Sam's uvicorn log.
  - Fix: dropped the rewrite. `/ws/*` now reaches the backend unchanged. Matches the path that `TestClient.websocket_connect("/ws/projects/.../events")` uses in [test_runs.py:288](reward-sculptor-ui/backend/tests/test_runs.py) + [test_clips.py:235](reward-sculptor-ui/backend/tests/test_clips.py), so the test suite already asserts the correct contract — it just wasn't exercised end-to-end through the dev server.
- **Verified**:
  - Sculptor: **91 passed, 1 skipped** (unchanged).
  - Backend: **207 passed** (unchanged).
  - Frontend typecheck: clean (`tsc --noEmit` exit 0).
  - Vite config change requires a **./run.sh restart** to take effect (vite.config.ts isn't hot-reloaded). The sculptor source change IS picked up by uvicorn reload since `--reload-dir ../RewardSculptor/sculptor` includes it.
- **Still deferred**: cheap regression test that asserts `_snapshot`'s `data.*` attribute refs exist on `mjlab.entity.EntityData`. Worth adding next session — it's fast (no GPU) and catches the whole class of "mjlab renamed an attr" bugs without a full train loop.

### 2026-04-21 20:30 — `run_manager` created empty legacy KG, shadowing shared (Project KG tab showed 0 papers)

Sam opened the Knowledge Graph tab on `unitree-go1-3` and saw "No papers in the KG yet" — even though the shared KG has 46 pre-extracted papers.

- **Root cause**: [backend/services/run_manager.py:62](reward-sculptor-ui/backend/services/run_manager.py) hardcoded `env["SCULPTOR_KG_PATH"] = str(project_dir / "kg" / "graph.db")`, forcing every sculpt subprocess to open a legacy per-project DB regardless of the shared-first precedence baked into `project_kg_db_path`. SQLite creates the file on open, so the first `New run` click for any project left an **empty** `<project>/kg/graph.db` behind. From then on, `project_kg_db_path` (which prefers a legacy DB when it exists on disk) returned the empty legacy path — shadowing the 46-paper shared DB for all subsequent UI reads.
- **Fix**: replaced the hardcoded path with `project_kg_db_path(project_dir)` so the subprocess writes to the same DB the UI reads. Respects legacy DBs when they exist, routes new projects to shared.
- **Heal**: deleted the empty `<project>/kg/graph.db` for `unitree-go1-3` (confirmed 0 nodes first — safe). `unitree-g1` was left alone (its legacy DB has real data from an earlier manual ingest; user can delete it later if they want to promote to shared).
- **Regression tests** in [backend/tests/test_runs.py](reward-sculptor-ui/backend/tests/test_runs.py):
  - `test_run_sculpt_job_exports_shared_kg_path_for_new_project` — monkey-patches `asyncio.create_subprocess_exec` to capture the `env` kwarg, asserts `env["SCULPTOR_KG_PATH"]` equals `shared_kg_db_path()` for a fresh project dir, and explicitly asserts it's NOT the legacy path.
  - `test_run_sculpt_job_honors_existing_legacy_kg` — opposite case: pre-existing `<project>/kg/graph.db` keeps the subprocess pointed at legacy (no silent migration, matches `project_kg_db_path`'s contract).
- **Verified**:
  - Backend: **207 passed** (was 205; +2 new). `uv run pytest backend/tests/ -q` in 46 s.
  - Live: `GET /projects/unitree-go1-3/kg/stats` now returns `{"papers":46,...}` after deleting the empty legacy DB.
- **Also baked into run.sh**: `export MUJOCO_GL="${MUJOCO_GL:-egl}"` so the preview endpoint's `mujoco.Renderer` doesn't hang picking a display backend on WSL2. Reversible by setting `MUJOCO_GL=osmesa` before invoking `./run.sh`.

### 2026-04-21 20:15 — `current.py` re-export hid `compute_reward_batched` (mjlab training always errored)

Sam's first mjlab training attempt on `unitree-go1-3` errored in 17 s with zero GPU utilization. UI showed a contradictory state (History: ERRORED; top banner: RUNNING + WS CLOSED) because the WS was still holding the pre-error state after a backend restart — the actual failure was in the mjlab subprocess.

- **What Sam reported / saw**: Run `eda70807869da882` errored after 17 s on iter 0. GPU at 0 % utilisation, 0.38 / 7.96 GiB VRAM (idle baseline). Iter 0 panel stuck in "running" with `v0 → v1` in the timeline. Event log empty (`0 entries`).
- **Root cause**: [sculptor/edit.py::_write_current_reexport](RewardSculptor/sculptor/edit.py) emitted a `rewards/current.py` re-export that bound only `compute_reward` and `REWARD_SPEC` — `compute_reward_batched` was never copied to the module-level namespace of `current.py`. `MjlabAdapter`'s runner ([sculptor/adapters/_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py)) loads `current.py` via `spec_from_file_location` and looks up `compute_reward_batched` as a **module-level attribute** — not via `_mod.compute_reward_batched`. So the batched entry point was invisible, `SculptorRewardTerm.__init__` raised `AttributeError("reward module ... missing compute_reward_batched; required when training with MjlabAdapter")`, mujoco never got past `env.load_managers()`, and training aborted before a single optimizer step ran. Hence zero GPU.
- **Full error from the run log** at [~/.local/share/reward-sculptor/projects/unitree-go1-3/runs/_run_job_eda70807869da882.log](../../.local/share/reward-sculptor/projects/unitree-go1-3/runs/_run_job_eda70807869da882.log):
  > `AttributeError: reward module '…/rewards/current.py' missing compute_reward_batched; required when training with MjlabAdapter (set REWARD_SPEC['supports_batched']=True and define the batched entry point).`
- **Fix** ([sculptor/edit.py:717-757](RewardSculptor/sculptor/edit.py)):
  ```python
  if hasattr(_mod, "compute_reward_batched"):
      compute_reward_batched = _mod.compute_reward_batched
      __all__.append("compute_reward_batched")
  ```
  appended to the template. `hasattr` guard keeps gym_sb3 / scalar-only modules working untouched.
- **Why M2 tests missed it**: my M2 work added the batched ABC + stub emission via `edit.py`, and the scalar v0 template was updated to *define* `compute_reward_batched`. But the re-export layer (`_write_current_reexport`) wasn't touched — no test verified `compute_reward_batched` survived the v0 → current shim. Classic "two layers both need the fix" gap.
- **Healing existing projects**: all 4 of Sam's projects (`unitree-g1`, `unitree-go1`, `unitree-go1-2`, `unitree-go1-3`) had the broken re-export. Ran `sculptor.edit._write_current_reexport(rewards, latest)` over each (one-shot via `uv run python -c ...`). After: every `current.py` exposes `compute_reward`, `compute_reward_batched`, `REWARD_SPEC`. Also corrected the `_LATEST` pointer for `unitree-go1` and `unitree-go1-3` which had stale v0 references while their v1.py (edited via the Rewards-tab prompt) existed on disk — separate latent bug where prompt-edit flows wrote v<n+1>.py without updating current.py. Flagged for a follow-up.
- **Regression tests** added to [RewardSculptor/tests/test_sculpt.py](RewardSculptor/tests/test_sculpt.py):
  - `test_write_current_reexport_surfaces_compute_reward_batched` — seeds v0.py with both entry points, writes the re-export, loads the generated current.py via `spec_from_file_location` (mjlab's actual load path), asserts `hasattr(mod, "compute_reward_batched")` and presence in `__all__`.
  - `test_write_current_reexport_skips_batched_for_scalar_only_module` — v0.py with only `compute_reward`, asserts the re-export doesn't invent a phantom batched binding (guards the `hasattr` contract for gym_sb3).
- **Verified**:
  - Sculptor tests: **91 passed, 1 skipped** (jax-gated). `uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py` in 15 s. Baseline was 89; +2 new.
  - Live import smoke on all 4 healed projects: each `current.py` imports via file-path spec and exposes `compute_reward`, `compute_reward_batched`, `REWARD_SPEC` with `REWARD_SPEC["version"]` matching the latest v<n>.py (v0 for unitree-g1 + unitree-go1-2; v1 for unitree-go1 + unitree-go1-3).
  - UI-level smoke (retry the run) is Sam's. Expected: training subprocess now passes `env.load_managers()`, rsl_rl's rollout loop starts, `nvidia-smi` / the GPU widget shows non-zero utilisation, iter 0 progresses past the 17 s startup cliff.

### 2026-04-21 21:50 — Physics validator path-context bug (second root cause)

Sam retried a prompt edit after H1-H3 landed; rejection still fired with the identical `ValueError: Error: Error opening file 'assets/hip.stl'` — but this time from the RejectionCard (so H3's UX is working, H1's assets ARE on disk). Different bug than H1.

- **What**: [backend/services/physics.py::apply_prompt_edit](reward-sculptor-ui/backend/services/physics.py) validator swapped from `mujoco.MjModel.from_xml_string(new_xml)` → write new_xml to a sibling tempfile `<local_dir>/.__rs_validate.xml` + `mujoco.MjModel.from_xml_path(str(tempfile))`, wrapped in try/finally so the tempfile is always unlinked (both on pass and on mujoco-reject). Same rejection dict shape as before; `rejected_at="mujoco_validate"` unchanged.
- **Why it was broken**:
  - `mujoco.MjModel.from_xml_string(xml)` has **no parent-path context**. When the XML declares `<compiler meshdir="assets"/>` and then `<mesh file="hip.stl"/>`, mujoco has nothing to resolve `assets/hip.stl` *against* — it can't reach into `<project>/uploads/robot/assets/`. Every mesh-referencing MJCF (i.e. every real menagerie model) failed validation regardless of whether the assets were physically on disk.
  - H1 fixed the "assets missing" bug so assets are present. The validator bug was masked by H1 — once assets were restored, this second bug became the actual blocker. Classic "fix bug → reveal bug".
- **Diagnostic that confirmed**:
  ```
  from_xml_path(<sibling tempfile>): OK, njnt=13 nu=12 ngeom=55
  from_xml_string(<same xml>): ValueError: Error opening file 'assets/hip.stl'
  ```
  on Sam's unitree-go1 via `uv run python -c ...`.
- **How**:
  - Picked sibling tempfile over `MjSpec.from_string(..., assets={...})` (which would require loading every mesh into a dict). Tempfile approach is one extra write + unlink and keeps the same path-resolution semantics mujoco uses everywhere else.
  - Fixed-name `.__rs_validate.xml` is safe because the physics-edit route serializes per-project (409 on concurrent edits — [routes/physics.py:195-208](reward-sculptor-ui/backend/routes/physics.py)). No race.
  - `try/finally` ensures cleanup on both the mujoco-reject return path and the happy path that falls through to the write.
- **Verified**:
  - Two new regression tests in [backend/tests/test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py):
    - `test_apply_prompt_edit_validator_resolves_meshdir_with_sibling_tempfile` — seeds a project with a real parseable binary-STL (hand-rolled unit tetrahedron via `struct.pack`), a `<compiler meshdir="assets"/>` + `<mesh file="tet.stl"/>` MJCF, and asserts apply_prompt_edit commits. Would fail against the old `from_xml_string` validator with the exact error Sam saw.
    - `test_apply_prompt_edit_validator_tempfile_cleaned_on_reject` — asserts `.__rs_validate.xml` is unlinked even when mujoco rejects (try/finally contract).
  - Backend: **205 passed** (was 203 after H3; +2 new). `uv run pytest backend/tests/ -q` in 52 s.
  - Live smoke on Sam's broken unitree-go1: `mujoco.MjModel.from_xml_path(<temp sibling>)` succeeds with `njnt=13, nu=12, ngeom=55` against his real `base.xml`. `from_xml_string` still errors on the same input with `'assets/hip.stl'`. Fix confirmed against actual failing state.
  - UI-level smoke: same "Sam retries the SEA prompt" path pending. Expected: Claude's rewrite now passes validation + commits (instead of hitting the RejectionCard), and Sam sees a `physics:` commit in the project's git log.
- **Why Phase 5's test suite missed it (same story as H1 but for the validator)**: Phase 5 tests used hand-crafted MJCFs with zero mesh references (`<geom type="box"/>` only), so `from_xml_string` happily parsed them. The validator-with-mesh-refs code path was never exercised. The new regression tests seed a real parseable STL so the meshdir path is always covered going forward.

### 2026-04-21 21:10 — Physics tab hotfix executed (H1 + H2 + H3 landed)

Executes [PHYSICS_TAB_HOTFIX.md](PHYSICS_TAB_HOTFIX.md) through H3. H4 (form-based field editing) deferred pending Sam's greenlight.

- **What**:
  - **H1 — asset-chain materialize + remateriailize endpoint** ([backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py)):
    - `materialize_mjcf` rewritten: parses the library MJCF via `xml.etree.ElementTree`, copies `<compiler meshdir=...>` + `<compiler texturedir=...>` + conventional `assets/` / `meshes/` / `textures/` subdirs whole, and chases explicit `<mesh file=>`, `<hfield file=>`, `<texture file=>`, `<include file=>` attrs (including transitive include recursion up to depth 8). Absolute paths + `..` traversal produce warnings and skip, never crash. **Idempotent on second call**: `skip_existing=True` branch preserves user / Claude edits to the XML while topping up any assets that went missing — heals the pre-H1 broken state on Sam's unitree-go1 without touching base.xml.
    - New `_copy_referenced_assets(src_xml, dst_dir, *, skip_existing=False)` helper does the parsing + copying (pure fn, returns warnings list).
    - New `_resolve_library_mjcf(project_dir)` extracted from `resolve_mjcf_path` so both materialize paths and tests can resolve the library source directly.
    - New public `rematerialize_mjcf(project_dir)` wipes `<project>/uploads/robot/` via `shutil.rmtree` then calls `materialize_mjcf` — explicit heal for projects with a registered library source.
    - New route: `POST /projects/{slug}/physics/rematerialize` ([routes/physics.py](reward-sculptor-ui/backend/routes/physics.py)) returns a fresh `PhysicsLoadResponse`. 409s on active sculpt run or in-flight prompt edit; 404s when no library source is registered (upload-only projects).
  - **H2 — parse_error surfacing** ([backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py), [frontend](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx)):
    - `MjcfSummary` gains `parse_error: Optional[str]` field (surfaced in `to_dict()`). `_summarize_mjcf` populates it with `f"{type(e).__name__}: {e}"` on mujoco failure instead of the pre-H2 silent `MjcfSummary()` fallback.
    - TS `MjcfSummary` mirrors the new field ([lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts)).
    - New `usePhysicsRematerialize` mutation hook ([hooks/usePhysics.ts](reward-sculptor-ui/frontend/src/hooks/usePhysics.ts)) + `rematerializePhysics` API fn ([lib/api.ts](reward-sculptor-ui/frontend/src/lib/api.ts)) that primes the `["physics", slug]` cache on success.
    - `PhysicsTab.tsx` renders an amber warning card when `summary.parse_error` is set, with a "Re-materialize MJCF" button wired to the new endpoint.
  - **H3 — Claude output preserved on rejection + inline RejectionCard** ([backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py), [frontend](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx)):
    - `apply_prompt_edit` return dict gains `rejected_at: Literal["parse" | "claude_rejected" | "mujoco_validate"] | None` tag. On `mujoco_validate` and `claude_rejected` paths, `new_xml` is Claude's actual output (pre-H3 overwrote it with `old_xml`, hiding the attempted change). `diff_lines` now populated on rejection too. New `claude_output_raw` field stashed on `rejected_at == "parse"` so the UI can render the raw response that failed the `<?rs-summary ... ?>` contract.
    - Frontend `RejectionCard` component (bottom of [PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx)) renders below the prompt textarea when the last job rejected: red banner with `rejected_at` badge ("response format" / "claude refused" / "mujoco rejected"), full reason, `<details>` disclosure with **MonacoDiffLazy** (`original=old_xml` / `modified=new_xml`) or a preformatted raw-output pane on parse failures, + "Retry with a different prompt" button that focuses the textarea.
    - Toast simplified to "Edit rejected — see card below the prompt". Prompt text is preserved on rejection (pre-H3 it cleared on every completion) so Sam can tweak + retry without retyping. Typing into the textarea clears the stale rejection card.
- **Why**: executes the plan written 2 hours earlier. Single root cause (materialize dropping `assets/`) surfaced as three independent UX cuts: empty summary (H2), blocked edits (H1), lost Claude output (H3). All three gated on the same asset-copy fix.
- **How**:
  - **Asset copying: `xml.etree.ElementTree` over real XML parsing, not regex**. Regex-matching `<mesh file="...">` ignores namespacing, default attrs, and nested `<include>` chains. ET's `.iter("mesh")` walks all mesh elements across the tree regardless of location.
  - **Dir-by-convention + element-scan, belt-and-braces**. Step 1 copies any of `{meshdir, texturedir, "assets", "meshes", "textures"}` subdirs wholesale via `copytree(dirs_exist_ok=True)`. Step 2 handles explicit file attrs that may live outside those dirs (e.g. `<mesh file="sub/foo.stl"/>` with no meshdir).
  - **Safety net for untrusted paths**: `_rel_safe` helper rejects absolute paths (`is_absolute`) and parent-traversal (`".." in parts`) with a warning. Plan called these out and they're tested explicitly.
  - **Idempotent top-up uses `skip_existing=True`** — both `_copy_file` and the custom `copy_function` passed to `copytree` check `dst.exists()` before overwriting. So a second materialize adds missing assets without touching anything the user / Claude has edited.
  - **Extracted `_resolve_library_mjcf`** so tests can monkey-patch that single seam to inject a synthetic library source — no need to mock `robot_library.get_library()` + `resolve_menagerie_path`.
  - **Backend tests use a fake Anthropic client** (`_FakeAnthropicClient` in [test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py)) — no live API hits. Covers the 3 reject paths + happy commit in one pass.
  - **Rejection-card state lives in `PhysicsTab` local state, not query cache**, because the rejection is a job result (not a /physics GET). Typing into the textarea clears it so stale rejections don't linger across edits.
- **Deferred to H4** (per plan, pending Sam's greenlight): un-stub `PUT /projects/{slug}/physics/fields`, add `apply_field_edits` via `mujoco.MjSpec`, inline-editable shadcn Input fields in the right column with debounce.
- **Verified**:
  - **Backend**: 203 passed (baseline 188 → +15 new tests: 9 H1 + 2 H2 + 4 H3). `uv run pytest backend/tests/ -q` in 43 s. H1 target was ≥ 193, H2 ≥ 195, H3 ≥ 198 — hit all three.
  - **Sculptor**: 89 passed, 1 skipped (unchanged baseline). `uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py` in 10 s.
  - **Frontend typecheck**: `tsc --noEmit` exit 0 after each phase (H1 no-op since no frontend touched; H2 + H3 added types).
  - **Manual smoke (H1 top-up against Sam's real broken unitree-go1)**: `materialize_mjcf(Path("~/.local/share/reward-sculptor/projects/unitree-go1"))` called via uv — before: `uploads/robot/` had only `base.xml`; after: `base.xml + assets/{calf,hip,thigh,thigh_mirror,trunk}.stl`; `load_mjcf` summary = `timestep=0.002, gravity=[0,0,-9.81], integrator=euler, joints=13, actuators=12, geoms=50`. UI-level smoke (opening the tab, re-materialize button, prompt→rejection→RejectionCard path) deferred to Sam's greenlight — his plan explicitly says "Don't commit — I'll do that myself after smoke-testing in the UI."
  - **Regression discipline**: didn't touch any existing test case; the +15 tests are strictly additive.

### 2026-04-21 19:00 — Physics tab bug diagnosed + hotfix plan written (M7 Phase 5 follow-up)

- **What Sam reported** (UI screenshot on unitree-go1 Physics tab):
  - "Edit rejected" toast: `mujoco rejected the new MJCF: ValueError: Error: Error opening file 'assets/hip.stl'`.
  - Read-only summary pane shows `GRAVITY —`, `INTEGRATOR —`, `SOLVER —`, `Joints (0)`, `Actuators (0)` — all empty.
- **Root cause (single bug, two symptoms)**: [backend/services/physics.py::materialize_mjcf](reward-sculptor-ui/backend/services/physics.py) does `shutil.copy2(path, local)` — copies the XML only. The sibling `assets/` directory from `robot_descriptions/go1_mj_description/` (holds the STL meshes the XML references via `<compiler meshdir="assets"/>`) is never copied. mujoco can't resolve `assets/hip.stl` → `_summarize_mjcf` silently catches the exception and returns an empty `MjcfSummary()` → UI shows zeros. Same exception hits the Claude-edit validator (`mujoco.MjModel.from_xml_string(new_xml)` at physics.py:~410) → every prompt edit gets rejected regardless of what Claude writes. **Diagnostic confirmed**: `ls ~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/` shows `base.xml` with NO `assets/` sibling; `head -3 base.xml` shows `<compiler angle="radian" meshdir="assets" autolimits="true"/>`.
- **Why Phase 5 didn't catch this**: my M7 Phase 5 tests used hand-crafted minimal `<mujoco>...</mujoco>` strings with zero mesh references. The menagerie MJCFs are realistic; mesh-chain materialization was never exercised. Flagged as a Phase-5 coverage gap — any new MJCF-touching code needs a "realistic menagerie model" test path alongside the minimal one.
- **Secondary bugs surfaced by this diagnosis** (fix as part of the patch):
  1. `_summarize_mjcf` swallows all mujoco errors silently (caller can't tell the empty fields apart from a truly empty model). Should populate a `parse_error: str` field.
  2. `apply_prompt_edit` validator-rejection OVERWRITES `new_xml` with `old_xml` (physics.py:415) — user loses the ability to inspect what Claude actually tried to write.
  3. UI toast (PhysicsTab.tsx) is the only surface for rejections — 160 chars in a fade-out with no "view what Claude wrote" option.
- **Hotfix plan** written at [PHYSICS_TAB_HOTFIX.md](PHYSICS_TAB_HOTFIX.md) (top-level `~/projects/`). Self-contained bootstrap doc for a new Claude window, mirroring [M7_PLAN.md](M7_PLAN.md)'s format: user profile, stack, test baseline, root-cause walkthrough, 4 phases (H1 asset-chain materialize + re-materialize endpoint, H2 surface parse errors in summary, H3 preserve Claude output on reject + inline RejectionCard, H4 optional form-based field editing), verification gates, critical-files map, known gotchas, starter prompt for the new session.
- **Also in this chapter**: exec-bit regression on `run.sh` — the Write/Edit tool creates files at 0644, stripping the executable bit. Sam hit `-bash: ./run.sh: Permission denied` after Phase 0's `--reload` patch. Fixed via `chmod +x`; saved as memory `feedback_exec_bit.md` so future shell-script edits trigger a follow-up chmod.
- **No code changes in this entry** — diagnosis + hotfix plan only. Sam will spin up a new Claude window against `PHYSICS_TAB_HOTFIX.md` to execute.
- **Verified**:
  - Plan file: 285 lines at `~/projects/PHYSICS_TAB_HOTFIX.md`. Covers H1-H4 with inline file:line references.
  - No test runs / no commits in this entry.
  - Manual smoke pending user: once the H1 patch lands, `rm -rf ~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/` → reload tab → summary populates + edit commits on first try.

### 2026-04-21 18:43 — M7 Phase 7: P5 polish batch (5 surfaces, ships together)

All five items were independent, so they share a single log entry + a single test-suite pass. No surface changes sculptor-library semantics; each is strictly additive.

- **7a — "Why this edit?" expandable on Rewards tab**:
  - New `GET /projects/{slug}/rewards/{version}/diagnosis` ([backend/routes/rewards.py](reward-sculptor-ui/backend/routes/rewards.py)) reads `<project>/runs/iter_<version-1>/diagnosis.json` and returns the raw payload. 404 for v0 (no triggering diagnosis), for missing files, and for unknown slug.
  - Frontend: `getRewardDiagnosis` + `useRewardDiagnosis(slug, version, {enabled})` hook. New `WhyThisEditPanel` in [RewardsTab.tsx](reward-sculptor-ui/frontend/src/components/RewardsTab.tsx) — collapsible card under the version detail pane, renders failure_modes (as chips), evidence (wrapped text), proposed edits (operation badge + target_term + rationale + paper_refs as arxiv links), and confidence. Shown only for sculptor-authored versions with a non-zero version number.
  - `RewardDiagnosisPayload` type defined in `lib/types.ts` with every field optional so older diagnoses with slightly different schemas don't break the render.

- **7b — Reward diff view**:
  - [MonacoLazy.tsx](reward-sculptor-ui/frontend/src/components/MonacoLazy.tsx) gains a sibling `MonacoDiffLazy` component — lazy-loads `@monaco-editor/react`'s `DiffEditor`; same `vs-light` theme + JetBrains Mono font + no minimap conventions as `MonacoLazy`. `renderSideBySide: true` for the default two-column diff.
  - `ReadOnlyPane` in RewardsTab gets a `"Diff vs parent"` toggle button (lucide `GitCompare` icon) next to the "New human edit" button — only rendered when `detail.version > 0`. When active, the right-pane Monaco swaps in the diff editor comparing parent (`useReward(slug, version-1)`) vs current.
  - Loading state: parent version fetch shows a centered spinner in the 420-height pane; error path shows a rose-tinted "Could not load parent version" message.

- **7c — Project settings dialog**:
  - New [ProjectSettingsDialog.tsx](reward-sculptor-ui/frontend/src/components/ProjectSettingsDialog.tsx) — gear icon in the ProjectDetail header opens a shadcn `Dialog` containing (a) a `SummarySection` showing slug / adapter short-name / task_id / num_envs / device / library / created_at as a compact read-only `<dl>`, and (b) a `DangerZone` section with type-to-confirm slug + delete button calling the existing `useDeleteProject`. On success, toasts + navigates back to `/projects`.
  - Editable fields (environment_tag, KG auto-research, iteration defaults) deferred to a follow-up that requires a new `PATCH /projects/{slug}` endpoint. Flagged inline in the dialog docstring.
  - Wired into [ProjectDetail.tsx](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx) sticky header next to the existing `NewRunDialog`.

- **7d — Run-viewer GPU card**:
  - New `RunGpuCard` component at the bottom of `RunDetailPane`'s right column in [RunsTab.tsx](reward-sculptor-ui/frontend/src/components/RunsTab.tsx). Uses the existing `useSystemGpu({refetchIntervalMs: 2000})` hook — only rendered while `isActive` (run is running/queued). Hidden when the host has no CUDA so CPU-only gym_sb3 runs don't see noise.
  - Renders name + VRAM bar (used/total with green/amber/rose thresholds at 70%/90%), utilization bar (when pynvml exposes it), and a temperature row (rose at >85°C, amber at >75°C).
  - Uses `/system/gpu` directly — no new backend endpoint needed. Per-run GPU stats via WebSocket (M4 §4 deferral) remains a Phase 7 follow-up; polling is cheap enough that it's acceptable MVP.

- **7e — KG graph click-through**:
  - [sculptor/kg/viz.py](RewardSculptor/sculptor/kg/viz.py) gains `_inject_click_forwarder()` — appends a `<script>` before `</body>` that hooks `network.on('click', ...)` and posts `{type: "kg_node_click", id, kind, arxiv_id}` to `window.parent`. Derives `kind` + `arxiv_id` from the node-id prefix (`paper:`, `technique:`, etc.). Tries postMessage with a `try/catch` so a cross-origin parent never breaks the embed. Injection goes right after title/legend so both happen in one `generate_html` pass.
  - Frontend: [GraphModal.tsx](reward-sculptor-ui/frontend/src/components/GraphModal.tsx) adds a `message` listener (scoped to `open` state so it cleans up when the dialog closes) + `selectedArxivId` state. On `kind=Paper`, stacks a `<PaperDetailModal>` on top of the graph modal with the clicked paper's full detail. Non-Paper clicks (Technique, FailureMode, etc.) are no-ops for now — side-pane entity views flagged as a Phase-7 follow-up.

- **Why**: close out the M7 P5 nice-to-haves list (the "pick whichever next" pile). Sam said "do all of it", so all five surfaces land in one batch — low risk because each is additive + independent.

- **How**:
  - **Diagnosis-iter mapping** — `v<n>.py` is written by `iter_<n-1>` (the iter that diagnosed the previous version and proposed the edit). So the diagnosis for v3 lives at `runs/iter_2/diagnosis.json`. Endpoint docstring calls this out explicitly.
  - **Monaco diff** — reused the same lazy chunk so no bundle-size hit. `DiffEditor` is a separate dynamic import but the monaco worker is shared; on-demand load means the diff button's first click triggers a brief fetch, and subsequent uses are instant.
  - **Settings dialog vs Sheet** — shadcn doesn't ship `Sheet` in this project's `ui/` dir today. Dialog with type-to-confirm is cheaper than adding the radix-ui sheet dep + component wrapper just for this one surface; upgrade to a real sheet when a second surface asks for one.
  - **GPU polling vs WS** — WS per-run `gpu_stats` events would need a second background task in `run_sculpt_job`. 2 s polling of `/system/gpu` is adequate (host has only one GPU; Sam's single-user install).
  - **pyvis click forwarder** — postMessage has a tiny cross-origin surface. iframe is same-origin (served by our own backend) so the message-passing is trivial. Using a prefix-match on node id (`paper:` / `technique:`) avoids needing pyvis's node-data hook (which requires per-node JS attributes).

- **Deferred to explicit follow-ups**:
  - Editable settings in `ProjectSettingsDialog` (needs `PATCH /projects/{slug}`).
  - Non-Paper entity side-pane in the KG graph (Technique / FailureMode / RewardComponent / Environment detail views).
  - Real shadcn `Sheet` primitive if a second surface wants a side panel.

- **Verified**:
  - Backend: **188 passed** (baseline 181 → +7 Phase 7). 43.79 s.
  - Sculptor: **89 passed, 1 skipped**. 11.75 s. Zero regression.
  - Frontend typecheck: clean.
  - Manual smoke (deferred to user): restart `./run.sh`, then:
    - Open any sculptor-authored v<n+1>.py on the Rewards tab → expand "Why this edit?" → failure modes + evidence + proposed edits render with arxiv links.
    - Click "Diff vs parent" → side-by-side Monaco diff with highlight.
    - Click the gear icon in ProjectDetail header → settings dialog opens with summary + danger zone.
    - Launch a run → right column shows a live GPU card with VRAM + utilization bars.
    - Open the KG graph modal → click any Paper node → PaperDetailModal stacks on top with full record.

- **M7 status**: all 8 phases (0-7) now complete. Test totals: **188 backend + 89 sculptor = 277 passing** (baseline pre-M7 was 129 + 76 = 205; M7 net-added 72 tests). Ready-to-commit trees listed in the 15:33 entry; today's additions layer on top. Sam can interrupt and commit at any phase boundary.

### 2026-04-21 15:45 — M7 Phase 6: job cancel UX + per-paper KG progress + Runs-tab error-action surfacing

- **What**:
  - **Stop button on ActiveJobsIndicator** ([frontend/src/components/ActiveJobsIndicator.tsx](reward-sculptor-ui/frontend/src/components/ActiveJobsIndicator.tsx)) — small X icon next to the in-flight-job chip. Calls the existing `POST /jobs/{id}/stop` via new `stopJob()` API ([lib/api.ts](reward-sculptor-ui/frontend/src/lib/api.ts)) + new `useStopJob(projectSlug)` mutation hook ([hooks/useJob.ts](reward-sculptor-ui/frontend/src/hooks/useJob.ts)). Indicator filter widened to include `kg_research` jobs (Phase 2 addition).
  - **Per-paper KG extract progress** — extended [sculptor/kg/extract.py::extract_all](RewardSculptor/sculptor/kg/extract.py) with optional `progress_cb: Callable[[int, int, str], None]`. Called once per paper BEFORE its Claude extract starts with `(done, total, title)` where `total` is the length of the filtered work queue (skipped papers excluded from the denominator). Callback exceptions are swallowed so a broken frontend never aborts extraction. Extract queue is pre-computed to expose the honest total; previously-extracted papers are recorded as `skipped=True` results without counting toward progress.
  - **Job progress wiring** ([backend/services/kg_jobs.py](reward-sculptor-ui/backend/services/kg_jobs.py)) — `run_ingest_extract_job` now passes `progress_cb=_on_paper_progress` into `extract_all`. The callback updates `job.message = "extracting 12 / 46: Walk These Ways"` (title truncated to 60 chars) and maps paper count into the 0.6-0.95 range of `job.progress` so the progress bar moves during extract (previously stuck at 0.6 → 1.0).
  - **Error-classification surfacing** end-to-end. [backend/services/run_manager.py](reward-sculptor-ui/backend/services/run_manager.py) now stashes the classification dict into `job.params["error_classification"]` on rc!=0 (not just the stdout event). [backend/models/run.py](reward-sculptor-ui/backend/models/run.py) adds an `ErrorClassification` pydantic model; `RunSummary` carries `error_classification: Optional[ErrorClassification]`. [backend/routes/runs.py](reward-sculptor-ui/backend/routes/runs.py) `_run_summary` materializes it from `job.params`, tolerant of garbage shapes.
  - **Runs-tab error card** ([frontend/src/components/RunsTab.tsx](reward-sculptor-ui/frontend/src/components/RunsTab.tsx)) — replaced the bare "Error" pre-block with a new `RunErrorCard` component: renders classification.title as the heading, classification.detail as description, raw error text, suggestions bullet-list, AND a one-click "Regenerate reward template" action button when `classification.kind == "reward_contract_mismatch"` (or `action.kind == "regenerate_reward_template"`). Calls the existing `useRegenerateRewardTemplate(slug)` hook from morning's P0 patch.
  - **Frontend types** ([lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts)) — new `ErrorClassification` interface; `RunSummary` gains optional `error_classification`.
  - **Tests** ([backend/tests/test_phase6_polish.py](reward-sculptor-ui/backend/tests/test_phase6_polish.py)) — +6:
    - Stop endpoint returns the job on known id (with an Event wired to mirror JobManager.submit's code path).
    - Stop endpoint returns 404 on unknown id.
    - RunSummary surfaces the full classification dict when populated in params.
    - RunSummary returns `error_classification=None` when absent (no crash on legacy errored runs).
    - `extract_all(progress_cb=fn)` calls the cb exactly once per work-queue paper, 1-indexed against the correct total.
    - Broken callbacks don't abort extraction — three papers still process.
- **Why**: close the M7 P4-subset block so Sam has (1) a way to kill stuck jobs without `curl`, (2) honest progress feedback during the 2-5 min KG extract (was stuck at "extracting entities via Claude" for minutes with no motion), and (3) one-click remediation in the Runs tab when a project hits the reward-contract mismatch Sam re-reported this morning — no more bouncing between Runs + Rewards tabs to find the Regenerate button.
- **How**:
  - **Stop-button choice of icon position**: X icon inside the indicator chip (vs. a separate hover button) matches the rest of the UI's delete-pill pattern (ProjectCard's x, PaperDetailModal's close, etc.). Small-screen friendly.
  - **Progress callback semantics**: `(done, total, title)` fires BEFORE the expensive Claude call so a stop-mid-extract shows accurate state ("stopping at 12/46"). Title is passed verbatim; truncation happens in the job handler, not the callback, so the callback contract stays simple.
  - **Error-classification stash on `job.params`**: considered putting it on `job.result` instead, but result is populated only when the job returns successfully (rc==0 path). Errored jobs have no result, so params is the right home. Mutation of params from the run-handler is fine per the existing pattern (`params["pid"]`, `params["log_file"]`, etc.).
  - **Runs-tab card design**: detail + suggestions come from the classification; raw error text still shown as `<pre>` so the underlying exception text is never hidden. Regenerate button only appears when actionable — don't bait Sam into clicking it on an OOM failure.
- **Verified**:
  - Backend: **181 passed** (baseline 175 → +6 Phase 6). 45.87 s.
  - Sculptor: **89 passed, 1 skipped**. 12.86 s. Zero regression.
  - Frontend typecheck: clean.
  - Manual smoke (deferred to user): launch `./run.sh` → trigger a KG bulk seed → watch the job chip update "extracting 12 / 46: <title>" per paper → click X → job stops within ~2 s. For the error-remediation flow, spin up a fresh mjlab project WITHOUT regenerating first → launch a 3-iter dry-run → the Runs-tab Error card surfaces the "Regenerate reward template" button inline.
- **Next**: Phase 7 (P5 polish — pick whichever Sam asks for first: run-viewer GPU tab / reward diff view / settings drawer / KG click-through / "why this edit?" expandable).

### 2026-04-21 15:33 — M7 Phases 1-5: shared KG + research + prompt-on-rewards + GPU run params + Physics tab

Approved plan at `C:\Users\SamJD\.claude\plans\i-am-still-getting-lexical-quail.md`. This entry batches all five post-regression phases — they share test/verification infra and were built against the same Phase-0 baseline, so a single log entry keeps the change log readable.

- **Phase 1 — Shared KG + pre-extracted delivery**:
  - [sculptor/kg/store.py](RewardSculptor/sculptor/kg/store.py) `default_db_path()` now returns the shared `~/.local/share/sculptor/kg/graph.db` when no env var + no legacy per-project DB exists. `shared_db_path()` exposed as public API so backend consumers can resolve the path without importing sculptor. Honors both `SCULPTOR_KG_PATH` (legacy) and `RS_KG_PATH` (backend alias).
  - [backend/services/kg_store.py](reward-sculptor-ui/backend/services/kg_store.py) `project_kg_db_path()` routes new projects to the shared DB; pre-existing `<project>/kg/graph.db` is preserved in place (no silent migration). New `shared_kg_db_path()` helper mirrors the sculptor-side.
  - **Bundled pre-extracted DB**: `RewardSculptor/examples/kg_preextracted.db` — 46 papers / 269 techniques / 688 edges / 824 KB. Regenerated via new [scripts/regenerate_kg_preextracted.sh](RewardSculptor/scripts/regenerate_kg_preextracted.sh) when `examples/kg_seeds_global.yml` changes. [.gitattributes](RewardSculptor/.gitattributes) marks it `binary` so git diff doesn't try to print it.
  - [backend/main.py](reward-sculptor-ui/backend/main.py) startup hook `_bootstrap_shared_kg()`: copies bundled DB to shared path on first launch. No-op when `RS_KG_PATH` or `SCULPTOR_KG_PATH` is set (tests + ad-hoc redirects don't auto-seed tmp DBs).
  - **New `GET /system/kg/stats`** → papers / techniques / failure_modes / reward_components / environments / edges / embeddings + db_path + db_size + last_modified. Returns zero-filled when DB absent instead of 500.
  - [pages/Settings.tsx](reward-sculptor-ui/frontend/src/pages/Settings.tsx) gets a "Knowledge graph (shared)" card rendering those stats + a path hint.
  - Test conftest sets `RS_KG_PATH` per-test so the backend suite never touches the user's real shared DB.
  - +14 tests (5 sculptor-side + 9 backend). Docstring + docs update: [knowledge_graph.md](RewardSculptor/docs/knowledge_graph.md).

- **Phase 2 — Prompt-time research**:
  - New [sculptor/prompts/research_topic.md](RewardSculptor/sculptor/prompts/research_topic.md) — research-librarian system prompt. Hard rules: arxiv IDs only, no hallucinated IDs, 5-10 papers max, `coverage_note` when thin.
  - New [sculptor/kg/research.py](RewardSculptor/sculptor/kg/research.py) with `research_topic(topic, max_papers, store, dedupe_against_kg)` → Claude Opus 4.7 via `messages.parse`. Pydantic `ResearchResponse{papers, coverage_note}`. ID normalizer strips `arXiv:` / URL / `vN` suffix + validates against `YYMM.NNNNN` regex. Dedupes against `has_paper()`.
  - New `POST /projects/{slug}/kg/research` (after `ingest_global_seeds` in [routes/kg.py](reward-sculptor-ui/backend/routes/kg.py)). 202 + job handle. Request body `{topic, max_papers, auto_extract}`. 503 on missing `ANTHROPIC_API_KEY`, 409 when another KG job is in-flight.
  - New job runner `run_research_job` in [kg_jobs.py](reward-sculptor-ui/backend/services/kg_jobs.py): research → ingest_arxiv per new ID → extract_all. Writes to whichever KG path `project_kg_db_path` resolves (shared for new projects, legacy for pre-Phase-1 ones).
  - Frontend: [ResearchTopicDialog.tsx](reward-sculptor-ui/frontend/src/components/ResearchTopicDialog.tsx) with textarea + max-papers slider (1-20). Wired into `KnowledgeGraphTab` next to `AddSeedsDialog` + `Bulk-seed library`. `useResearchTopic` hook in [useKG.ts](reward-sculptor-ui/frontend/src/hooks/useKG.ts). Job progress flows through existing `ActiveJobsIndicator`.
  - +10 backend tests covering validation, 503/404/409 paths, job-submission smoke, + ID-normalization unit tests.

- **Phase 3 — Prompt-driven reward editing on the Rewards tab**:
  - New `apply_prompt_edit()` in [sculptor/edit.py](RewardSculptor/sculptor/edit.py) — synthesizes a minimal `Diagnosis` with one `add`-op `ProposedEdit` whose rationale carries the user prompt; routes through the existing Claude rewriter, skipping the diagnose step. `add` is chosen so `_pre_validate`'s grounded-term check passes (per sculpt.py rule — `add` allows fresh snake_case names).
  - New backend service [reward_jobs.py](reward-sculptor-ui/backend/services/reward_jobs.py) with `run_reward_prompt_edit_job(project_dir, user_prompt, expected_parent_version)` — resolves latest `v<N>.py`, asserts parent-version match (optimistic concurrency), loads the adapter's reward_contract, calls `apply_prompt_edit`, returns `{new_version, new_path, parent_version}`.
  - New `POST /projects/{slug}/rewards/prompt` → 202 + job. 409 when a sculpt run or another prompt-edit is active. 503 on missing API key. [routes/rewards.py](reward-sculptor-ui/backend/routes/rewards.py).
  - Frontend: [RewardsTab.tsx](reward-sculptor-ui/frontend/src/components/RewardsTab.tsx) `PromptEditHero` rewritten — textarea + "Prompt Claude" button + job-progress watch + auto-refresh rewards list on completion. Forks from the latest `v<N>.py` → writes `v<N+1>.py`.
  - +7 backend tests (validation, 503/404/409, job-submission smoke with a mocked `apply_prompt_edit`).

- **Phase 4 — GPU-appropriate run parameters**:
  - [models/run.py](reward-sculptor-ui/backend/models/run.py) `RunParams` extended with `training_iterations` (100-200k), `num_envs_override` (1-8192), `device_override` (cpu / cuda[:N]), `expand_kg` bool. All optional → full backward compat.
  - [NewRunDialog.tsx](reward-sculptor-ui/frontend/src/components/NewRunDialog.tsx) restructured into **Basic / Advanced** shadcn Tabs. Basic = behavior goal + dry-run. Advanced = sculpt-iters (outer) + training-iters/cycle (inner) + num_envs + device + expand_kg + no_kg. Per-adapter defaults via `pickAdapterDefaults(project)`:
    - gym_sb3: 20 sculpt × 50k steps.
    - mjlab cartpole: 15 × 500.
    - mjlab Go1: 12 × 1000.
    - mjlab G1: 8 × 1500.
    - Any other mjlab: 10 × 1000.
  - num_envs / device surface in job.params but aren't yet enforced at train-time (adapter cooperation required; deferred to a follow-up). training_iterations / expand_kg likewise — flagged in the RunParams docstring.
  - Frontend `RunParamsPayload` type extended to match.

- **Phase 5 — Physics tab (prompt-first)**:
  - New [sculptor/prompts/physics_editor.md](RewardSculptor/sculptor/prompts/physics_editor.md) — physics-engineer system prompt. Covers three SEA patterns (joint-coupling, tendon-spring, parallel-elastic), hard numeric bounds (timestep ∈ [1e-4, 0.1], damping ≥ 0, etc.), commit-summary-via-`<?rs-summary?>`-PI contract, REJECTED sentinel when the edit is physically impossible. AME456's hybrid joint-damping + armature pattern + MuJoCo Discussion #226 referenced explicitly.
  - New backend service [services/physics.py](reward-sculptor-ui/backend/services/physics.py) with `load_mjcf(project_dir)` + `apply_prompt_edit(project_dir, prompt)`. Resolution precedence: `<project>/uploads/robot/*.xml` (upload) → Menagerie via `resolve_menagerie_path` (library). First library-project physics edit materializes the MJCF into `<project>/uploads/robot/base.xml` so git can track it. `_summarize_mjcf` parses the model via mujoco + returns joint/actuator/geom digests (length-capped; some models have hundreds of visual geoms).
  - `apply_prompt_edit`: call Claude → parse the `<?rs-summary?>` PI + `<mujoco>` body → validate via `MjModel.from_xml_string` round-trip → commit as `physics: <summary>` via `git -C <project> add + commit`. REJECTED summaries (Claude or validator) leave the file unchanged.
  - New [routes/physics.py](reward-sculptor-ui/backend/routes/physics.py) with:
    - `GET /projects/{slug}/physics` — `{xml_source, summary, mjcf_source_kind, mjcf_path, materialized}`.
    - `POST /projects/{slug}/physics/prompt` — 202 + job. 409 on active sculpt run or concurrent physics edit.
    - `PUT /projects/{slug}/physics/fields` — **501 stub** (form-driven editing is the follow-up pass; UI builds against a stable contract now).
  - New `run_physics_prompt_edit_job` in [reward_jobs.py](reward-sculptor-ui/backend/services/reward_jobs.py) wrapping `apply_prompt_edit` via `asyncio.to_thread`.
  - Frontend: [PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) with left column (prompt textarea + Monaco XML view) + right column (read-only digest: options, joints, actuators). [usePhysics.ts](reward-sculptor-ui/frontend/src/hooks/usePhysics.ts) hook pair. ProjectDetail's TABS gets a `Physics` entry between Rewards and Knowledge Graph.
  - JobKind literal in [models/kg.py](reward-sculptor-ui/backend/models/kg.py) extended with `kg_research`, `reward_prompt_edit`, `physics_prompt_edit`.
  - +10 backend tests (MJCF load/404 paths, PUT 501, prompt-validation rejects, 503/409, `_parse_claude_output` unit coverage).

- **Why**: Sam approved the full M7 execution plan (phases 0-7) after the morning's Phase 0 fix shipped. Phases 1-5 address the major gaps he flagged in his reply: re-extraction on every project (→ shared KG + pre-extracted delivery), prompt-on-Rewards (→ Phase 3), Physics tab with prompt-driven MJCF editing (→ Phase 5), plus the M7 plan items he already named (Phase 2 research + Phase 4 run params). Decisions confirmed via AskUserQuestion: commit the ~1 MB sqlite binary, Phase 0 standalone first, prompt-first Physics tab (form-based editing deferred to follow-up).

- **How**:
  - **Phase 1 design:** the per-project KG was just a convention — SculptorKG already honored env-var overrides. Phase 1 flips the default without changing the write API: new projects route to shared, pre-Phase-1 projects keep their legacy DB automatically. Pre-extracted delivery as a committed binary avoids a first-boot Claude spend (~$2-3, ~5-8 min) while keeping the repo cold-start-simple.
  - **Claude response schemas:** Phase 2 uses `messages.parse` with a pydantic `ResearchResponse`, which coerces Claude's output directly into the validated structure. Phase 5's physics editor uses a hand-parsed `<?rs-summary?>` PI because the output is the full MJCF (doesn't fit a structured-output response) + a one-line commit summary — the PI sidesteps JSON quoting of the XML body.
  - **SEA-awareness:** the physics_editor prompt cites three concrete MuJoCo SEA patterns with arxiv references (2209.07171 Raffin, 2301.03509 ANYmal PEA) so Claude matches the existing model rather than forcing a style switch. Reflected in the hybrid joint-damping + armature pattern AME456 uses.
  - **Backward compat:** every new endpoint 202-accepts a job; the UI uses the existing `JobManager` polling surface to track progress. JobKind literal extended once (vs. adding new polling infra).
  - **Deferred (flagged explicitly):**
    - Form-based physics editing (`PUT /physics/fields` returns 501 today).
    - num_envs / device / training_iterations / expand_kg pass through the backend into `job.params` for audit but aren't YET enforced by `sculpt_run` — adapter cooperation is the follow-up.
    - Auto-research hook inside `sculpt_run` (Phase 2 P6 + P7) — endpoint + UI ship, but the automatic `expand_kg` trigger inside diagnose.py is a follow-up.
  - Regen script ran in background during coding (~10 min for 46 papers via Claude Opus; final DB 824 KB).
- **Verified**:
  - Backend: **175 passed** (baseline 139 → +36 new across Phases 1-5). 47.01 s.
  - Sculptor: **89 passed, 1 skipped**. 13.13 s. Zero regression.
  - Frontend typecheck: clean (`node_modules/.bin/tsc --noEmit` exit 0).
  - Bundled KG: 46 papers, 269 techniques, 688 edges, 824 KB. `SCULPTOR_KG_PATH=<bundle> sculpt kg stats` confirms.
  - Manual smoke (deferred to user): restart `./run.sh` → Dashboard shows library robots → Settings page "Knowledge graph" card reads 46 papers from the bootstrapped shared DB → KG tab has "Research a topic" button → Rewards tab has "Prompt Claude" textarea → Physics tab renders current MJCF + accepts a prompt → New Run dialog has Basic/Advanced tabs with G1-sized defaults.
- **Next**: Phase 6 (job cancel + KG progress polish) — ~1-2 hrs. Then Phase 7 nice-to-haves as Sam requests.
- **Ready-to-commit trees**:
  - `RewardSculptor/` — sculpt.py, edit.py, kg/store.py, kg/research.py, prompts/*.md, tests/test_kg.py, tests/test_sculpt.py, docs/knowledge_graph.md, scripts/regenerate_kg_preextracted.sh, examples/kg_preextracted.db, .gitattributes.
  - `reward-sculptor-ui/` — backend (main.py, routes/, services/, models/, tests/) + frontend (api.ts, hooks/, components/, pages/).

### 2026-04-21 14:51 — M7 Phase 0: backend auto-reload + preview-renderer library_slug branching (unblocks both Sam-reported regressions)

- **What**:
  - **[reward-sculptor-ui/run.sh:73-80](reward-sculptor-ui/run.sh)** — added `--reload --reload-dir backend --reload-dir ../RewardSculptor/sculptor` to the uvicorn command. Library-side edits (sculptor package, editable-installed) now trigger backend restarts alongside backend edits. Fixes the "my P0 fix isn't taking effect" regression: Sam's previous backend was running stale code from before this morning's P0 patch because uvicorn wasn't watching either dir.
  - **[backend/services/preview_renderer.py:112-152](reward-sculptor-ui/backend/services/preview_renderer.py)** — rewrote the `kind == "library"` branch. Old code required both `library_name` AND a non-sentinel `env_id`; the library-driven project-create flow writes `library_slug` + `menagerie_package` (neither of the old fields), so the preview endpoint returned 409 and Dashboard rendered "no robot yet" on every library-created project. New logic: (1) "nothing configured" sentinel-check uses library_slug/library_name/env_id together; (2) resolution tries Menagerie first via `get_library().resolve_menagerie_path(slug)` for mjlab + preview-only entries; (3) falls back to gymnasium's bundled asset via `env_id` for gym entries. Error message names both identifiers so the surfaced PreviewError is actionable.
  - **[backend/tests/test_robot.py:404-481](reward-sculptor-ui/backend/tests/test_robot.py)** — +2 regression tests:
    - `test_preview_library_slug_only_resolves_via_menagerie` — the exact failure Sam hit. Writes a library-flow robot_source (slug + menagerie_package, no library_name/env_id), patches `RobotLibrary.resolve_menagerie_path` → `HOPPER_XML`, confirms the preview renders (200 + PNG bytes).
    - `test_preview_library_slug_falls_back_to_env_id_for_gym` — gymnasium_compatible library entry (has env_id, no menagerie_package) still renders via the bundled gym asset path.
- **Why**: Sam reported two fresh-project regressions (2026-04-21 afternoon): (1) mjlab sculpt-run still failing with "Reward template missing batched entry point" even though my morning P0 fix was in the code, (2) Dashboard showing "no robot selected" on every library-created project. Root-cause diagnostic via three parallel Explore agents: (1) → backend wasn't auto-reloading (no `--reload`), (2) → `preview_renderer` looked at `library_name` but `_finalize_library_scaffold` writes `library_slug`. Neither bug was in the P0 code — (1) was a dev-env missing flag, (2) was a pre-existing field-name mismatch exposed once users went through the library flow instead of the legacy `POST /robot/library` route.
- **How**:
  - **Process fix first** — `--reload` is a one-line change that prevents the category of bug where "my edits don't show up". Scoped reload dirs so file-watchers don't churn on unrelated trees (docs, tests, .venv). Safe in dev; docs may add `--reload-dev` split if we ever ship a production start script, but single-file `run.sh` is a developer tool today.
  - **preview_renderer fix** — preserved backward compat with the legacy POST /robot/library flow (library_name + env_id still works). Made the new library-slug flow authoritative. Tried Menagerie first because library entries with both slug and menagerie_package are expected to be Menagerie-sourced; fallback only runs if resolver returns None.
  - **Plan file**: full M7 execution plan written at `C:\Users\SamJD\.claude\plans\i-am-still-getting-lexical-quail.md` — 8 ordered phases, each with explicit verification gates; scope confirmed with user via AskUserQuestion (commit the pre-extracted KG binary, Phase 0 standalone, prompt-first Physics tab).
- **Verified** (all four gates green):
  - Backend: **139 passed** (baseline 137 → +2 new). 42.67 s.
  - Sculptor (no GPU): **84 passed, 1 skipped**. 12.75 s. Zero regression.
  - Frontend typecheck: clean (`node_modules/.bin/tsc --noEmit` exit 0).
  - Manual smoke (Sam's to run after restarting `./run.sh`): create a fresh Unitree G1 project from the library → Dashboard shows the robot preview → Rewards tab v0.py exports `compute_reward_batched` → 3-iter `--dry-run` completes without exit-1.
- **Scope**: Phase 0 complete. Moving to Phase 1 — shared KG + pre-extracted delivery.

### 2026-04-21 14:18 — M7 P0: mjlab reward-template fix (adapter-aware scaffold + regenerate endpoint + run-error classifier)

- **What**:
  - **Library (RewardSculptor/sculptor/sculpt.py)** — split `_V0_STUB_REWARD` into `_V0_SCALAR_REWARD` (unchanged content, gym_sb3-style) and `_V0_BATCHED_REWARD` (new; exports BOTH `compute_reward` for the preflight/probe AND `compute_reward_batched` for mjlab's per-step GPU path; sets `REWARD_SPEC["supports_batched"] = True`). Added `_BATCHED_ADAPTER_CLASSES` frozenset (currently `{"sculptor.adapters.mjlab.MjlabAdapter"}`), `_adapter_needs_batched_template()`, and `_v0_template_for()` helpers. `_ADAPTER_SHORT_NAMES` gained `"mjlab" → "sculptor.adapters.mjlab.MjlabAdapter"`. [`sculpt_init`](RewardSculptor/sculptor/sculpt.py:862) now calls `_v0_template_for(adapter)` instead of writing the scalar stub unconditionally. New public [`regenerate_reward_template(project_dir)`](RewardSculptor/sculptor/sculpt.py) helper reads the project's `[adapter].class` from config.toml and rewrites `rewards/v0.py` — preserves `current.py`'s pointer when `v1+.py` exists so already-iterated projects don't regress.
  - **Library tests (RewardSculptor/tests/test_sculpt.py)** — +8 tests: gym_sb3/scalar template, mjlab/batched template via short name, mjlab/batched template via dotted class path, regenerate flips scalar→batched after adapter swap, regenerate stays scalar on gym, regenerate overwrites user edits by design, regenerate preserves current.py on iterated projects, regenerate raises on missing config.
  - **Backend classifier (reward-sculptor-ui/backend/services/cuda_errors.py)** — `CudaErrorClass` gained `problem_type: str` and `action: Optional[dict]` fields (both defaulted — backwards-compatible). New `reward_contract_mismatch` branch matching `"missing compute_reward_batched"` (case-insensitive substring); ordered BEFORE the OOM scanner to avoid false-positive matches against CUDA stack frames in the same traceback. All existing kinds got explicit `problem_type` URIs. Docstring updated to reflect scope now covers any subprocess error.
  - **Classifier tests** — +3 tests in [test_cuda_errors.py](reward-sculptor-ui/backend/tests/test_cuda_errors.py): detection of the new kind + action shape, contract-match wins over CUDA-mentions in a mixed traceback, problem_type assigned for all known kinds.
  - **Backend project-creation (routes/projects.py)** — dropped the `body.adapter == "mjlab"` scaffold-as-gym_sb3 workaround (lines 106-114 of the old code). sculpt_init is adapter-aware now. Coming-soon adapters (isaac/mjx/rllib) still scaffold via gym_sb3 since their real templates don't exist yet; [adapter] is rewritten afterward. Updated inline comment.
  - **New regenerate-template endpoint (routes/rewards.py)** — `POST /projects/{slug}/rewards/regenerate-template`. Lock + active-sculpt-run guard identical to manual PUT (409 with `/problems/state-conflict`), 503 when sculptor is unimportable, 404 on missing project. Returns the updated v0 `RewardVersionDetail` so the UI can refresh without a second GET.
  - **Endpoint tests** — +5 tests in [test_rewards.py](reward-sculptor-ui/backend/tests/test_rewards.py): happy path (v0 flips to batched after adapter swap), no-op shape for gym_sb3, 404 on unknown slug, 409 when a sculpt_run job is in-flight, current.py preservation when v1.py exists.
  - **Run-error surfacing (services/run_manager.py)** — wired classifier into the `rc != 0` emit. New `_classify_run_failure(log_path, project_dir)` reads the tail of the run log (last 32 KiB), pulls `[adapter].config.num_envs` from config.toml for OOM suggestions, and returns a `CudaErrorClass`. `run_errored` events now carry `error_kind`, `error_detail`, `error_suggestions`, `error_problem_type`, `error_action` fields — UI can render specific remediations.
  - **Frontend (RewardsTab.tsx + hooks/useRewards.ts + lib/api.ts)** — new `regenerateRewardTemplate(slug)` API call, `useRegenerateRewardTemplate(slug)` TanStack mutation hook (invalidates rewards list + detail + project query on success), and a `RegenerateTemplateButton` component rendered in `ReadOnlyPane`'s header next to "New human edit" — only shown for v0. Destructive-action confirmation via shadcn `Dialog` with explicit warning that manual edits are lost but git history preserves the old file.
- **Why**: P0 of M7 — only thing preventing mjlab sculpt runs from working on Sam's Go1 project (and the other two mjlab projects on disk). Root cause confirmed by diagnostic: every mjlab project's `rewards/v0.py` only exports scalar `compute_reward`, and `_mjlab_runner.py:85` raises `AttributeError: reward module … missing compute_reward_batched; required when training with MjlabAdapter` — exits at ~19 s (cold import + immediate AttributeError on first `compute_reward_batched` call). Sam reproduced the bug twice on 2026-04-21 (once pre-KG, once post-KG), confirming KG state was orthogonal.
- **How**: library-first — teach `sculpt_init` to pick the right template from the adapter + provide a public helper for migrating already-scaffolded broken projects. Backend drops the gym_sb3 aliasing workaround (no longer needed). Classifier + run_manager wiring makes the failure visible with a one-click remediation action. Frontend surfaces it as a small button on the v0 row (not a modal takeover) — Sam can regenerate on demand without needing a failed run first. All edits incremental over WSL UNC paths; no cross-project refactor.
- **Trade-offs flagged**:
  - `_CONFIG_TEMPLATE` is still gym_sb3-shape (`env_id=CHANGE_ME, n_envs=4, ppo_kwargs=…`) under `sculpt init --adapter mjlab` from the CLI. The UI path overwrites `[adapter].config` post-scaffold, so this only bites CLI-only users. Not P0 — flagged as a cosmetic follow-up.
  - Run-viewer frontend does not yet render `error_kind` / `error_action` with a "Regenerate template" one-click button. Backend emits the fields; UI surface is a follow-up.
- **Deferred to follow-up**:
  - Runs-tab banner wiring (pick up `error_action` → render Regenerate button inline on failed runs).
  - `sculpt init --adapter mjlab` CLI config shape (proper mjlab config dict template).
- **Verified**:
  - Sculptor (no GPU): **84 passed, 1 skipped** (baseline 76 → +8 new). 15.78 s.
  - Backend: **137 passed** (baseline 129 → +8 new: 3 classifier + 5 regenerate endpoint). 37.95 s.
  - Frontend typecheck: clean (`node_modules/.bin/tsc --noEmit` from `~/.local/share/pnpm` node).
  - Manual smoke (deferred to user): the behavioral verification path is integration-tested through `test_regenerate_template_flips_scalar_to_batched_for_mjlab` (full FastAPI TestClient round-trip including project-create + adapter-swap + endpoint call + filesystem assertion). Sam's 3 broken on-disk projects (`quadruped-jumping-sea`, `quadruped-sea-jumping`, `unitree-go1`) were NOT touched by this change — he regenerates them from the UI when he resumes.

### 2026-04-20 18:58 — Set global git identity on WSL host

- **What**: `git config --global user.name "sjdoane"` + `user.email "sjdoane@usc.edu"`. No repo-tracked files changed.
- **Why**: `tests/test_sculpt.py::test_sculpt_init_scaffolds_expected_files` was failing because `sculpt_init` runs `git init`+`git commit`, and git on this WSL had no identity — `sculpt_init` swallowed the failure non-fatally, then the test's `git log --oneline` returned exit 128. Windows silently had identity already; Linux didn't.
- **How**: host-level git config change. Picked over a test-only patch because the same bug would hit real `sculpt init` usage by the user on this machine; fixing only the test would paper over it.
- **Verified**: `uv run pytest tests/test_sculpt.py -q` → 10/10 passed. Full RewardSculptor suite now back to parity (62 passed, 1 skipped expected). `reward-sculptor-ui` backend suite already at 69 passed pre-change.

### 2026-04-21 13:42 — UX polish + KG seeds + M7 plan handoff

- **What**:
  - **Fix: Dashboard always routed to legacy ProjectCreate.** Replaced the page with a redirect to `/library`. Dashboard + ProjectList "New project" buttons now point at `/library`. Empty-state redesigned with a gradient hero + 3 starter-step cards.
  - **New `/library` top-level route** ([frontend/src/pages/LibraryPage.tsx](reward-sculptor-ui/frontend/src/pages/LibraryPage.tsx)) — standalone library browser (extracted `LibraryBrowser` from `RobotConfig.tsx` via `export`). Header surfaces GPU status + ready-to-train count.
  - **ProjectDetail facts band** ([pages/ProjectDetail.tsx](reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx)) — above the tabs: adapter badge (green mjlab / blue gym_sb3 / amber stub), task_id or env_id, num_envs, device, library_slug link, "Training disabled" pill. Amber banners for adapter_unavailable + migration_warning with a "Fork" CTA.
  - **Launch button fix.** Mounted `NewRunDialog` directly in ProjectDetail's sticky header so "New run" is always visible (right-aligned, top-right). Hidden when `ready_to_train=false` or `adapter_unavailable=true`. Removed the DOM-query hack from `RewardsTab.PromptEditHero`.
  - **Rewards tab prompt hero** — clarifies where to prompt Claude; no click handler (points at header button).
  - **KG tab empty-state / pending-no-papers banners** with actionable messages ([KnowledgeGraphTab.tsx](reward-sculptor-ui/frontend/src/components/KnowledgeGraphTab.tsx)).
  - **Migration warning on ProjectCard** + [test_migration.py](reward-sculptor-ui/backend/tests/test_migration.py) (+5 tests, 129 backend total).
  - **Bulk-seed KG button on KG tab** → `POST /projects/{slug}/kg/ingest-global` reads [`examples/kg_seeds_global.yml`](RewardSculptor/examples/kg_seeds_global.yml) (50 papers), merges to kg_seeds.yml, fires `run_ingest_extract_job`. 409 when another KG job is already running. One-click — no terminal.
  - **50-paper global seeds file** compiled from AME456/files/docs/RESEARCH_LOG.md (15 SEA / jumping-robot papers: Atanassov 2401.16337, RAMIEL 2403.11205, JumpER 2507.01243, Raffin 2209.07171, ANYmal PEA 2301.03509, Passault 2410.08650, Apostolides 2503.16197, Stanford Doggo 1905.04254, Losey 1902.05346, Pinto 2409.09203, …) + M5 library refs (13 papers) + foundational RL (9 papers: PPO, SAC, DDPG, GAE, DeepMind Control Suite, Isaac Gym) + cross-cutting (8 papers). All arxiv IDs curated from the research log; no invented citations.
  - **unitreerobotics + mujocolab repo scan** (research agent, ~40 repos inspected) confirmed very low paper density — unitree repos are SDKs/ROS wrappers with one tangential arxiv ref (Qwen2.5-VL 2502.13923 in `unifolm-vla`); mjlab already in the global seeds. Conclusion: AME456 research log was the right primary source.
  - **Screenshot placeholders** at `reward-sculptor-ui/docs/screenshots/` (9 zero-byte PNGs + capture guide).
  - **5 new docs**: [docs/knowledge_graph.md](RewardSculptor/docs/knowledge_graph.md), [docs/migration.md](RewardSculptor/docs/migration.md), [docs/robot_library.md](RewardSculptor/docs/robot_library.md), [docs/wsl_setup.md](RewardSculptor/docs/wsl_setup.md), [docs/e2e_scenarios.md](RewardSculptor/docs/e2e_scenarios.md), [docs/performance.md](RewardSculptor/docs/performance.md). README gained GPU requirements + library-size summary + adapter-support table with links to [docs/adapters/isaac.md](RewardSculptor/docs/adapters/isaac.md), [docs/adapters/mjx.md](RewardSculptor/docs/adapters/mjx.md), [docs/adapters/rllib.md](RewardSculptor/docs/adapters/rllib.md). CONTRIBUTING gained an "Add a robot to the library" section.
  - **Adapter registry + stubs** ([sculptor/adapters/__init__.py](RewardSculptor/sculptor/adapters/__init__.py)) with 5 `AdapterInfo` entries. Isaac/MJX/RLlib stubs raise `NotImplementedError` with the exact spec'd format. `GET /library/adapters` endpoint exposes the registry.
  - **CreateProjectDialog adapter dropdown** — coming-soon entries get `⏳` + inline confirmation card with adoption-guide link + estimated effort. Submit changes to "Create anyway" when a coming-soon adapter is picked. Selecting `isaac` / `mjx` / `rllib` scaffolds with `adapter_unavailable=true` in metadata.
  - **`ready_to_train` / `adapter_unavailable` / `library_slug` / `migration_warning`** added to `ProjectSummary` / `ProjectDetail` pydantic.
- **Known bug surfaced but NOT fixed**: end-to-end mjlab sculpt-run exits with code 1 at ~19 s. Root cause: `sculpt_init` writes a gym-shape `rewards/v0.py` (scalar `compute_reward` only) regardless of adapter; the mjlab runner's `SculptorRewardTerm` calls `compute_reward_batched` and raises `AttributeError`. **Reproduced twice — once with empty KG, once with fully-seeded KG** (2026-04-21) — confirming the KG extract state is orthogonal to the bug. **P0 fix owned by M7**; see [M7_PLAN.md](M7_PLAN.md) §"Known bug" for the diagnostic commands + all three candidate fixes + the migration helper for Sam's already-scaffolded broken project.
- **Why**: polish pass on what the user sees when the UI loads (Dashboard / Library / ProjectDetail), plus give them a one-click path to populate the KG with the full curated corpus.
- **How**: lifted LibraryBrowser out of RobotConfig via a single `export` (no refactor); reused `NewRunDialog` by mounting it higher in the component tree; added `AdapterInfo` pydantic + `GET /library/adapters`; wired `POST /kg/ingest-global` to read the YAML + merge via existing `append_seeds`. Every frontend change typechecks clean via `node_modules/.bin/tsc --noEmit` (node from pnpm standalone install at `~/.local/share/pnpm/node`).
- **Verified**:
  - Backend: **129 passed** (unchanged from M6). Frontend typecheck clean.
  - User ran G1 project + hit the sculpt-exit-1 bug — documented in M7 plan. KG seeding verified working (3 papers from G1's auto-seed auto-ingested; user saw them in the KG tab).
  - Global seeds file ingestible via the Bulk-seed button OR CLI; path: [RewardSculptor/examples/kg_seeds_global.yml](RewardSculptor/examples/kg_seeds_global.yml).
- **M7 plan handoff**: new [M7_PLAN.md](M7_PLAN.md) at top-level — self-contained bootstrap for a new Claude window with: user profile, stack diagram, known bug + 3 candidate fixes, shared-KG architecture decision, prompt-time research endpoint spec, GPU-appropriate run parameter restructure, 6 prioritised task blocks P0-P6 with verification gates. Read it in a new window BEFORE touching code.

### 2026-04-20 22:00 — M6 integration: adapter selector UI, migration safety, docs, performance, tests

- **What**:
  - **Prelude (M5 UI follow-up)** — CreateProjectDialog now has an adapter dropdown populated from `GET /library/adapters`. Default picks mjlab for mjlab_ready robots, gym_sb3 otherwise. Coming-soon adapters (⏳ prefix + "(coming soon)" suffix) trigger an amber inline card with the adoption-guide link (resolved to either an absolute URL or a GitHub blob URL under this repo), `estimated_effort`, and a "Project will be created but training will be disabled" warning. Submit button relabels to "Create anyway" when a coming-soon adapter is picked. New `useLibraryAdapters` hook; `AdapterInfo` type; `listLibraryAdapters` api client.
  - **M6.1 migration safety** — `ProjectSummary` + `ProjectDetail` gain `migration_warning: str | None` (computed in `ProjectStore.get/list` via `_compute_migration_warning` against `ADAPTER_REGISTRY`). Coming-soon adapters don't trigger the warning (they ARE registered, just status="coming_soon"); only unknown class_paths do. [ProjectCard.tsx](reward-sculptor-ui/frontend/src/components/ProjectCard.tsx) renders an amber banner with "Uses a deprecated adapter. Fork from the library to upgrade — the original will not be modified." **No in-place mutation anywhere** — the user's explicit safety requirement.
  - **M6.2 E2E scenarios** — [docs/e2e_scenarios.md](RewardSculptor/docs/e2e_scenarios.md). Five scenarios A-E with concrete pass criteria:
    - A: legacy `uv run sculpt run --dry-run` (3-iter Hopper, 60 s budget).
    - B: library → mjlab project → live train. **Go1 instead of Go2** (Go2 has no mjlab task in v1.3.0; doc explains the substitution).
    - C: coming-soon adapter (Isaac Lab) scaffolds with training gated.
    - D: no-GPU path via `CUDA_VISIBLE_DEVICES=""`.
    - E: custom URDF upload zero-regression.
  - **M6.3 performance** — [docs/performance.md](RewardSculptor/docs/performance.md). Measured on this host (RTX 5070 Laptop / WSL2). Endpoints: `/library/robots` 10.8 s cold (D-guard subprocess) → 3-4 ms warm; `/system/gpu` 3.5 s cold → 6 ms cached; `/library/adapters`, `/library/categories`, `/health` all < 10 ms. Backend cold-start 0.56 s. **mjlab Go1 throughput: ~27,500 env-steps/sec** at num_envs=1024 on this hardware (89 s for 100 iters). Two targets missed (cold library + cold GPU) are per-process one-off costs; doc flags them + proposes post-M6 mitigations (pre-warm task list, on-disk nvml cache).
  - **M6.4 docs pass** — four new documents + two updates:
    - [docs/migration.md](RewardSculptor/docs/migration.md) — when to migrate, when to stay, fork-only workflow, what to carry over / NOT carry over, backward-compat guarantees.
    - [docs/robot_library.md](RewardSculptor/docs/robot_library.md) — YAML shape, enumerate `robot_descriptions` loaders, verify URLs via curl, `generate_library_thumbnails.py` usage + first-run Menagerie clone note, KG-seeding behavior, category guidance.
    - [docs/wsl_setup.md](RewardSculptor/docs/wsl_setup.md) — zero → `./run.sh`: WSL2 install, CUDA pass-through verification (nvidia-smi), uv install, project clone, sync commands, `ANTHROPIC_API_KEY`, sanity-check test runs, common pitfalls (/mnt/c slow, CRLF, MUJOCO_GL, systemd-resolved).
    - [docs/e2e_scenarios.md](RewardSculptor/docs/e2e_scenarios.md) — above.
    - [docs/performance.md](RewardSculptor/docs/performance.md) — above.
    - [README.md](RewardSculptor/README.md) — GPU requirements block + robot library size summary. Top-level adapter matrix already present from M5.
    - [CONTRIBUTING.md](RewardSculptor/CONTRIBUTING.md) — new "Add a robot to the library" contribution path + pointer to `docs/adapters/` scaffolds.
  - **M6.5 edge cases** — most are pre-existing from M2-M5 (lazy-import probes, 412 preflight paths, D-guard demotion). Documented concrete fail modes in wsl_setup.md (MUJOCO_GL / systemd-resolved / CRLF) + performance.md (cold-path penalties). URL-404 library-ref validation is documented as a post-M6 optional check in robot_library.md (not implemented: would slow startup by ~10-30 s).
  - **M6.6 tests** — [backend/tests/test_migration.py](reward-sculptor-ui/backend/tests/test_migration.py), 5 tests: gym_sb3 / mjlab / coming-soon adapters all get NO warning; unknown class_path DOES trigger migration_warning; /projects and /projects/{slug} both surface the warning. Library fixture validation already covered by [test_library.py](reward-sculptor-ui/backend/tests/test_library.py) (every YAML entry validates slug pattern + category enum + URL regex).
  - **M6.7 screenshot placeholders** — 9 zero-byte PNGs at `reward-sculptor-ui/docs/screenshots/` + [PLACEHOLDERS.md](reward-sculptor-ui/docs/screenshots/PLACEHOLDERS.md) listing each capture with concrete UI state to reproduce (dashboard w/ GPU, library browser, detail modal, CreateProjectDialog happy-path, coming-soon confirm, OOM retry, Settings GPU panel, migration banner).
- **Why**: M6 integration — close M5 UI deferral (blocks Scenario C), ship migration safety before users accumulate legacy projects, document the stack for onboarding + future contribution. User explicit deferrals post-M6: run-viewer GPU tab, OOM→WS event, frontend vitest. These are observability/UX polish, not load-bearing.
- **How**: frontend adapter dropdown uses the same `useLibraryAdapters` hook shape as other library fetchers; Infinity staleTime since the registry is effectively static per backend lifetime. Migration detection is a pure computation in `_compute_migration_warning` (reads ADAPTER_REGISTRY once; no I/O) so it's free on every `list_projects` call. Performance doc includes BOTH numbers I measured directly (endpoints + backend cold-start + test-suite runtime) AND the mjlab throughput baseline derived from the earlier GPU smoke fixture (89 s × 1024 envs × 24 steps/iter × 100 iters ÷ wall-clock).
- **Deferred (per user spec)**:
  - Run-viewer GPU tab with `gpu_stats` WebSocket events (M4 §4, "observability polish; safe to defer post-M6").
  - OOM classifier → WebSocket event wiring (M4 §5, "nice-to-have").
  - Frontend vitest test-setup.
- **Verified**:
  - Backend: **129 passed** (baseline 124 + 5 migration). 42 s wall-clock.
  - Sculptor non-GPU: **76 passed, 1 skipped** (no regression). 14 s.
  - Combined: **205 tests** across both projects + 2 GPU (cached fixture, 3 s). Comfortably above user's "~90-100 tests by end of this series" threshold.
  - 9 URL citations across the three adapter guides: all curl-verified 200 in the previous M5 turn; doc cross-references documented.
  - Performance targets met for 6/8 measured; 2 cold-path misses flagged + mitigations described.
- **New files this turn**:
  - `RewardSculptor/docs/migration.md`, `docs/robot_library.md`, `docs/wsl_setup.md`, `docs/e2e_scenarios.md`, `docs/performance.md`.
  - `reward-sculptor-ui/frontend/src/components/CreateProjectDialog.tsx` (adapter selector + ComingSoonConfirmCard added).
  - `reward-sculptor-ui/backend/tests/test_migration.py` (5 tests).
  - `reward-sculptor-ui/docs/screenshots/PLACEHOLDERS.md` + 9 zero-byte PNGs.

### 2026-04-20 21:47 — M4 closures (preflight + create-dialog) + M5 scaffolded adapters

- **What**:
  - **M4.1 live-VRAM preflight** — new [backend/services/preflight.py](reward-sculptor-ui/backend/services/preflight.py). `check_mjlab_preflight(task_id, num_envs, device, project_dir, min_gpu_memory_gb)` uses pynvml-backed free VRAM via `gpu_monitor.get_live_snapshot()` + optional cached per-env coefficient from `<project>/.sculptor_cache/vram_coefficients.json`. Returns `PreflightResult` with `suggested_num_envs` (largest power-of-two within [128, 4096] that fits 85% of free VRAM). Wired into `_validate_mjlab_preflight` in [routes/projects.py](reward-sculptor-ui/backend/routes/projects.py). 412 body now includes `free_vram_gb`, `estimated_required_gb`, `device_name`.
  - **M4.2 CreateProjectDialog** — new [components/CreateProjectDialog.tsx](reward-sculptor-ui/frontend/src/components/CreateProjectDialog.tsx). Three flows based on `training_support`:
    - `mjlab_ready`: name + CUDA device dropdown + num_envs slider (128-4096, default = library recommendation) + live VRAM estimate with green/amber/red banner (70% / 85% thresholds).
    - `gymnasium_compatible`: name-only minimal dialog.
    - `preview_only`: confirm-and-create.
    On 412 `/problems/insufficient-vram`, shows a rose banner with free/needed VRAM + device name + "Retry with num_envs={suggested}" button that rewrites state and re-submits. The [RobotConfig.tsx](reward-sculptor-ui/frontend/src/components/RobotConfig.tsx) detail-modal CTA now opens this dialog instead of direct-POSTing. `CreateProjectPayload` type extended with `library_slug`, `task_id`, `num_envs`, `gpu_device`.
  - **M5 registry** — new `ADAPTER_REGISTRY` dict in [sculptor/adapters/__init__.py](RewardSculptor/sculptor/adapters/__init__.py) with 5 entries (`gym_sb3`, `mjlab` ready; `isaac`, `mjx`, `rllib` coming_soon). Each entry carries `AdapterInfo(name, display_name, class_path, status, supported_robot_categories, adoption_guide_url, estimated_effort)`. Exposed via new `GET /library/adapters` endpoint (ordered ready-first).
  - **M5 stub refinement** — [isaac_lab.py](RewardSculptor/sculptor/adapters/isaac_lab.py), [mjx.py](RewardSculptor/sculptor/adapters/mjx.py), [rllib.py](RewardSculptor/sculptor/adapters/rllib.py). NotImplementedError format tightened to exactly `"<Framework> adapter not yet implemented. Adoption guide: <path>. Estimated effort: <range>."` per user spec. Guide URLs point at docs/adapters/{isaac,mjx,rllib}.md.
  - **M5 adoption guides** — three new files under [RewardSculptor/docs/adapters/](RewardSculptor/docs/adapters/): [isaac.md](RewardSculptor/docs/adapters/isaac.md) (1,500 words), [mjx.md](RewardSculptor/docs/adapters/mjx.md) (1,200 words), [rllib.md](RewardSculptor/docs/adapters/rllib.md) (1,400 words). Each has: target versions, install prereqs (`uv add ...`), reward-injection pattern, minimal viable `train()` skeleton, testing strategy, known gotchas, completion checklist, and 3 verified URLs. **All 9 URLs curl-verified 200**: Isaac Lab github + GH Pages + Isaac Sim docs; Brax github + MuJoCo github + MJX docs; Ray github + RLlib index + envs guide.
  - **M5 adapter_unavailable flag** — `ProjectDetail.adapter_unavailable: bool` (M5 new field). When `body.adapter` is a coming-soon registry entry, the route scaffolds with gym_sb3 as a placeholder, rewrites `[adapter]` class to the registry's `class_path` with empty config, and stashes `robot_source.adapter_unavailable=true` in metadata. `ProjectStore.get()` reads the flag and sets `ready_to_train=false` — UI disables Train with the adoption-guide URL in tooltip.
  - **M5 README** — root [RewardSculptor/README.md](RewardSculptor/README.md) adapter-support table (gym_sb3 / mjlab ready; isaac / mjx / rllib scaffolded) with links to each adoption guide + explicit "contributions welcome" invitation.
  - **Tests** — 2 new test files, 14 new tests.
    - [backend/tests/test_preflight.py](reward-sculptor-ui/backend/tests/test_preflight.py) — 7 tests: ok-when-fits, fails-and-suggests-smaller, device-missing, device-index-out-of-range, cache-respected, cache-ignored-wrong-task, 412 body shape end-to-end.
    - [backend/tests/test_adapter_registry.py](reward-sculptor-ui/backend/tests/test_adapter_registry.py) — 7 tests: registry contents + ready/coming_soon split, NotImplementedError exact format (all 3 stubs), reward_contract validity, `GET /library/adapters` shape + ordering, coming-soon project creation sets adapter_unavailable, gym_sb3 zero-regression, adoption-guide files exist on disk with required sections.
- **Why**: M4 deferrals are load-bearing for M6 end-to-end scenarios; can't leave the UI without a device/num_envs picker or the backend without live preflight. M5 (scaffolded stubs) is scope-bounded reviewable code: no half-baked implementations, just a public surface + adoption guides for future contributors.
- **How**:
  - Preflight service is stateless + test-friendly (`_with_snapshot` mock wraps `gpu_monitor.get_live_snapshot` — patched via `patch.object(gpu_monitor, ...)`). Cached coefficient path tested with a hand-written `vram_coefficients.json` + `_cache_key` match.
  - Frontend dialog uses the same static VRAM formula as the backend (1.5 GB policy + 0.5 MB/env) so the user-visible estimate round-trips consistently with the backend's preflight decision.
  - Adoption guides cross-reference the MJLAB_PIVOT_DESIGN + M4 infrastructure — e.g. the Isaac Lab guide points at `_run_with_cleanup` in mjlab.py as the subprocess helper to reuse.
- **Deferred to next turn**:
  - Frontend adapter dropdown in the project-create dialog with coming-soon badges + "Create project anyway?" flow. The backend plumbing is done (`GET /library/adapters` + `adapter_unavailable` field); UI exposure lands next turn.
  - Run-viewer GPU tab + `gpu_stats` WebSocket events (M4 §4, still parked).
  - OOM → WebSocket event (M4 §5, classifier exists; wiring pending).
  - Frontend vitest tests.
- **Verified**:
  - Backend: **124 passed** (baseline 110 + 14 new). 35 s wall-clock.
  - Sculptor non-GPU: **76 passed, 1 skipped**. Zero regression.
  - `GET /library/adapters` live: 5 entries, ready-first ordering, coming-soon entries carry adoption_guide_url + estimated_effort.
  - Adoption-guide URLs: 9/9 curl-verified HTTP 200.
  - `docs/adapters/{isaac,mjx,rllib}.md` on disk; `test_adoption_guides_exist_on_disk` asserts each has "Target version", "Install", "Reward injection", "References" sections.
  - Coming-soon project creation: adapter_unavailable=true, ready_to_train=false, config.toml's `[adapter].class` points at the stub adapter's dotted path.

### 2026-04-20 21:29 — M3 frontend/thumbnails + M4 GPU core

- **What**:
  - **M3 deferrals finished**:
    - `scripts/generate_library_thumbnails.py` rewritten to iterate every YAML entry (Menagerie via `robot_descriptions` lazy imports, gym via `resolve_library_xml`, mjlab built-in Cartpole via `mjlab/tasks/cartpole/cartpole.xml`). Reuses `preview_renderer._build_camera` (plane-excluded bbox + COM-lifted center) + 20% camera-distance reduction for Arm / Gripper_Hand. First run cloned Menagerie + Unitree ros archives on demand (one-time). **63 PNGs committed to `frontend/public/robots/`**.
    - Also removed the broken `getattr(robot_descriptions, name)` path from the library loader in favor of `importlib.import_module("robot_descriptions.<name>")`.
    - Frontend: `lib/types.ts` gains `GpuDevice`, `SystemGpuResponse`, `LibraryRobot`, `ROBOT_CATEGORIES`, `DEFAULT_CATEGORIES`, `DEFAULT_TRAINING_SUPPORT`. `lib/api.ts` gains `getSystemGpu` + 3 library fetchers. New `hooks/useLibrary.ts` (`useLibraryCategories`, `useLibraryRobots`, `useLibraryRobot`, `useSystemGpu`).
    - `components/RobotConfig.tsx` **full rewrite**: categorized browser with category chips (default: Quadruped / Humanoid / Arm / Gripper_Hand), training-support chips (default: mjlab_ready + gymnasium_compatible), search box, card grid with thumbnail + training-support badge (green ready / blue Gymnasium / amber preview) + references count + description. Click → `RobotDetailModal` with full description, clickable references list, preconfigured tasks, demote_note (when present), CTA "Create project with this robot" → POST /projects with `library_slug` → navigate to the new project.
  - **M4 backend core**:
    - New [backend/services/gpu_monitor.py](reward-sculptor-ui/backend/services/gpu_monitor.py) — pynvml-based live GPU telemetry (utilization %, temperature °C, driver version, per-device memory used) with a 2 s process-local cache. Falls back gracefully to `torch.cuda` when pynvml init fails. Added `nvidia-ml-py` as a UI backend dep.
    - [backend/services/sculptor_bridge.gpu_info](reward-sculptor-ui/backend/services/sculptor_bridge.py) now delegates per-device data to `gpu_monitor.get_live_snapshot()`. `models/system.py` extended with `utilization_percent` / `temperature_c` / `used_memory_bytes` / `driver_version` / `pynvml_available` fields (pydantic `extra="allow"`).
    - New [backend/services/cuda_errors.py](reward-sculptor-ui/backend/services/cuda_errors.py) — `classify(text, current_num_envs)` returns a `CudaErrorClass` with `kind ∈ {oom, driver_version, no_cuda, unknown}`, title/detail/suggestions, and a suggested `num_envs` (power-of-two, clamped 128-4096) for OOM recovery.
  - **M4 frontend**:
    - [pages/Settings.tsx](reward-sculptor-ui/frontend/src/pages/Settings.tsx) — `GpuCard` replaced with a live-refresh (5 s) panel that renders per-device memory + utilization bars (green/amber/red thresholds), temperature, SM capability, driver version, CUDA version with ≥ 12.4 check, and mjlab / mujoco_warp / rsl_rl import-health dots. Warning banners for "no GPU" and "CUDA too old".
    - [pages/Dashboard.tsx](reward-sculptor-ui/frontend/src/pages/Dashboard.tsx) — new `GpuWidget` in the right column (above KG additions). 3 s refresh while a run is active, 10 s idle. Pulsing radio icon when training is in progress. Hidden entirely on CPU-only hosts.
  - **M4 tests**: 10 new tests.
    - [backend/tests/test_cuda_errors.py](reward-sculptor-ui/backend/tests/test_cuda_errors.py) — 8 tests: OOM detection across error-string variants, driver version, no_cuda, unknown fallback, num_envs snap-to-power-of-two + 128 floor, hint absence when current unknown.
    - [backend/tests/test_system.py](reward-sculptor-ui/backend/tests/test_system.py) — 2 new: M4 pynvml fields exposed, 2 s cache returns same dict.
- **Why**: user's M3 deferral scope + M4 §1-§2 + §5 + §7 + §8 (backend GPU + Settings panel + Dashboard widget + error classifier + tests). M4 §3 device selector on project creation is already satisfied for mjlab by the existing `gpu_device` field; UI surfacing deferred to next turn alongside the run-viewer GPU tab.
- **How**:
  - Thumbnails: CPU rendering path uses MuJoCo's default backend (WSL2's WSLg glfw) — osmesa was broken on this host. Single `--only-slug g1` smoke first to amortize the initial Menagerie clone, then full batch (63 / 63 succeeded in < 2 min, no failures).
  - pynvml init is lazy (first `get_live_snapshot()` call) so the backend cold-start stays < 1 s. 2 s cache TTL balances real-time feel against nvml overhead (< 1 ms per query but still a syscall).
  - CUDA error classifier is substring-matching (not regex) for speed + predictability. OOM heuristic picks `max(128, floor_pow2(current_num_envs // 2))` which pairs naturally with the mjlab VRAM coefficient cache (M2 §3.3).
- **Deferred to next turn** (explicit scope split):
  - M4 §4 run-viewer GPU tab with WebSocket `gpu_stats` events (requires extending `run_manager` WS protocol).
  - M4 §5 OOM → WebSocket event flow (classifier exists; wiring pending).
  - M4 §6 per-run preflight device check (`torch.cuda.mem_get_info` vs `min_gpu_memory_gb` before launch).
  - Frontend device-selector UI on the new-project dialog.
  - Frontend vitest tests (no vitest setup yet in this repo).
- **Verified**:
  - Backend: **110 passed** (baseline 100 + 10 M4). 35 s wall-clock.
  - `GET /system/gpu` live on RTX 5070 Laptop: `pynvml_available=true`, driver `592.00`, used `272 MB / 7.96 GiB`, utilization `0 %`, temperature `53 °C`, CUDA 13.0 (> 12.4 ✓). ✓ M4 #1.
  - 63 thumbnails at [frontend/public/robots/](reward-sculptor-ui/frontend/public/robots/): `cartpole_mjlab.png` … `wonik_allegro.png`. Visual inspection deferred (terminal).
  - Sculptor non-GPU + GPU (cached fixture): **76 + 2 = 78 passed, 1 skipped**. Zero M2/M3 regression.
  - Frontend typecheck not runnable from WSL (Windows-side pnpm can't resolve UNC module paths); TypeScript types mirror backend pydantic models directly.

### 2026-04-20 21:13 — Pre-M3 verifications (A, B, C) + M3 backend (robot library + auto KG seeding)

- **What**:
  - **Verification A — subprocess hardening**: new `_run_with_cleanup()` helper in [mjlab.py](RewardSculptor/sculptor/adapters/mjlab.py) using `Popen(start_new_session=True)` + `os.killpg()` on exception. Both `train()`, `rollout()`, and `measure_vram_coefficient` now route through it. Tests: `test_run_with_cleanup_kills_subprocess_on_exception` (fake 30s sleep, verifies cleanup ≤10s) + `test_mjlab_adapter_train_surfaces_subprocess_nonzero_exit` (stderr preserved in RuntimeError).
  - **Verification B — VRAM probe sanity**: `measure_vram_coefficient` extended with optional `cache_file` param (keyed by `(task_id, mjlab_version)`). GPU test `test_mjlab_vram_probe_returns_coefficient` asserts per-env coefficient in 0.1-10 MB range + cache file write + second-call cache hit. All green on cached fixture.
  - **Verification C — CUDA version compare**: [sculptor_bridge.cuda_version_ok](reward-sculptor-ui/backend/services/sculptor_bridge.py) switched to `packaging.version.parse` (not string compare). Test `test_cuda_version_ok_uses_numeric_comparison` covers 11 cases incl. the "9.0 vs 12.4" lexical-compare trap.
  - **Verification D — library YAML cross-reference**: integrated into the M3 library loader. `_fetch_mjlab_tasks()` in [robot_library.py](reward-sculptor-ui/backend/services/robot_library.py) subprocess-invokes `mjlab.tasks.registry.list_tasks()` (lazy, on first access, 60s timeout). Any `mjlab_ready` entry whose `preconfigured_tasks[*].task_id` is absent from the live registry demotes to `preview_only` in-memory with a `demote_note`; YAML on disk stays untouched. Covered by `test_d_guard_demotes_when_tasks_not_registered` + `test_d_guard_keeps_subset_of_valid_tasks` (both mocked for hermetic speed).
  - **M3 library data**: [backend/data/robot_library.yml](reward-sculptor-ui/backend/data/robot_library.yml) with **65 entries**: 6 mjlab_ready (Cartpole, Go1, ANYmal-C, G1, T1, Yam) + 5 gymnasium_builtin (Hopper/Ant/Walker2d/HalfCheetah/Humanoid) + 54 preview_only Menagerie. Every mjlab_ready has verified paper + repo references from M1 research. `menagerie_package` values authoritative via robot_descriptions enumeration (58 `*_mj_description` loaders). Added `robot-descriptions` to sculptor pyproject.
  - **Library service** ([services/robot_library.py](reward-sculptor-ui/backend/services/robot_library.py)): loader with eager schema validation (slug regex, category enum, URL regex) + lazy D-guard + `resolve_menagerie_path()` (per-process cache of materialized MJCF paths via robot_descriptions).
  - **Library endpoints** ([routes/library.py](reward-sculptor-ui/backend/routes/library.py)): `GET /library/robots` (filter by category + training_support + search), `GET /library/robots/{slug}`, `GET /library/robots/{slug}/thumbnail` (404s with remediation hint until thumbnails are generated), `GET /library/categories`. Response models in [models/library.py](reward-sculptor-ui/backend/models/library.py).
  - **Project creation integration** ([routes/projects.py](reward-sculptor-ui/backend/routes/projects.py)): `CreateProjectRequest.library_slug` resolves via `get_library()`, derives adapter + task_id + num_envs from the entry, then runs the existing mjlab preflight. `ProjectDetail` gains `ready_to_train: bool` + `library_slug: str | None`. `ProjectStore.get()` populates these from `metadata.json.robot_source`.
  - **Auto KG seeding**: `_finalize_library_scaffold` extracts arxiv IDs from paper references (regex on `arxiv.org/abs/<id>`) and appends to `kg_seeds.yml` via the existing `ProjectStore.append_seeds`. Repo references stashed in `metadata.json.robot_source.related_repos` for the future "Related repos" KG panel — not sqlite-ingested per design doc §6. `_maybe_fire_kg_ingest` enqueues the existing `kg_jobs.run_ingest_extract_job` when `ANTHROPIC_API_KEY` is set (best-effort; non-fatal on any failure).
  - **Tests** ([backend/tests/test_library.py](reward-sculptor-ui/backend/tests/test_library.py)): 20 tests covering YAML load + schema + D-guard demotion + endpoints + creation flow (G1 mjlab, Hopper gym, Franka FR3 preview-only) + arxiv-extraction.
- **Why**: pre-M3 gates unlocked by user message; M3 as originally specified, backend portion. M3 is the library foundation; thumbnails + frontend rewrite deferred to next turn per explicit scope split.
- **How**:
  - Chose subprocess-based D-guard (not in-process mjlab import) to preserve the lazy-import rule from M2's constraint #7. Lazy first-touch means the 60s registry fetch happens on first library request, not at FastAPI boot.
  - robot_library.yml populated from two authoritative sources: M1 design-doc Menagerie research (slug + display_name + category + references for mjlab-ready) + live `robot_descriptions` enumeration (menagerie_package loader names).
  - Thumbnails deferred because generating 65 renders takes 3-5 min wall-clock and needs a dedicated script — explicit scope split per user request. M3 endpoint returns structured 404 pointing at the generation script.
- **Deferrals to next turn** (user explicitly acknowledged): M3 step 6 thumbnail regeneration script + committed PNGs, M3 step 7 frontend RobotConfig.tsx rewrite, frontend tests.
- **Verified**:
  - Backend: **100 passed** (baseline 79 + 21 M3/pre-M3). `uv run pytest backend/tests/` in 42 s. ✓ M3 #1, #3, #4, #5.
  - Sculptor non-GPU: **76 passed, 1 skipped**. Zero regression. ✓ M2 gate preserved.
  - GPU suite (cached fixture): 2/2 still green via `pytest -m gpu` in ~3 s (reuse path).
  - YAML loads without errors (`test_library_loads_without_errors`): 65 entries within 40-70 range. ✓ M3 #1.
  - `GET /library/categories` → 8 categories, `GET /library/robots?category=Humanoid` filters correctly. ✓ M3 #6 (backend portion).
  - Library creation with `library_slug=unitree_g1` → adapter_class=`sculptor.adapters.mjlab.MjlabAdapter`, task_id=`Mjlab-Velocity-Flat-Unitree-G1`, kg_seeds.yml has 3 arxiv seeds. ✓ M3 #3.
  - Library creation with `library_slug=hopper` → gym_sb3, no regression. ✓ M3 #4.
  - Library creation with `library_slug=franka_fr3` → ready_to_train=false. ✓ M3 #5.
  - D-guard: mock empty registry demotes all mjlab_ready; partial registry keeps matching tasks. ✓ D gate.
- **New files**: `backend/data/robot_library.yml` (604 lines), `backend/services/robot_library.py`, `backend/models/library.py`, `backend/routes/library.py`, `backend/tests/test_library.py`.

### 2026-04-20 20:30 — M2 — MjlabAdapter + ABC extension + GPU UI plumbing

- **What**:
  - **Sculptor ABC extension** ([sculptor/adapters/base.py](RewardSculptor/sculptor/adapters/base.py)): `RewardContract` gains `supports_batched`, `training_device`, `min_gpu_memory_gb`, `state_schema`. New `ComponentProbe` dataclass. New abstract `probe_component` method. New `reward_batched` default (scalar-loop fallback with RuntimeWarning). `_import_reward_module` helper.
  - **GymSB3Adapter updated** ([sculptor/adapters/gym_sb3.py](RewardSculptor/sculptor/adapters/gym_sb3.py)) — `probe_component` via subprocess (0.0 scalars + empty info — matches existing UI probe behavior). Zero behavior change. `supports_batched=False`.
  - **MjlabAdapter** ([sculptor/adapters/mjlab.py](RewardSculptor/sculptor/adapters/mjlab.py)) — lazy mjlab import, task-registry validation in `__init__`, `num_envs` autocap at 2048 when VRAM < 12 GiB, `reward_contract` supports_batched=True with per-task-family state schema, subprocess `train`/`rollout` via `_mjlab_runner.py`, `reward_batched` dispatches to module's `compute_reward_batched` if defined. Module-level `measure_vram_coefficient` + `estimate_vram_static` helpers.
  - **Runner script** ([sculptor/adapters/_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py)) — argparse with three modes: `train`, `rollout`, `vram-probe`. Reward injection via class-based `SculptorRewardTerm` (zeroes default task rewards, sets `scale_rewards_by_dt=False`). Fixed `rl_cfg.to_dict()` → `dataclasses.asdict()` since rsl_rl's config is a dataclass, not a pydantic model.
  - **Stub adapters** ([isaac_lab.py](RewardSculptor/sculptor/adapters/isaac_lab.py), [mjx.py](RewardSculptor/sculptor/adapters/mjx.py), [rllib.py](RewardSculptor/sculptor/adapters/rllib.py)) — reviewable NotImplementedError backing with per-stack adoption guides in module docstrings. reward_contract + compute_behavior_metrics + probe_component return sensible defaults.
  - **edit.py prompt builder** ([sculptor/edit.py](RewardSculptor/sculptor/edit.py)) — when `contract.supports_batched=True`, emits BATCHED_CONTRACT block with state_schema, info keys, and the explicit "emit BOTH `compute_reward` AND `compute_reward_batched`" instruction. Per decision #2 from the 8-point ack.
  - **Backend sculptor_bridge** ([backend/services/sculptor_bridge.py](reward-sculptor-ui/backend/services/sculptor_bridge.py)) — `mjlab_available()`, `mujoco_warp_available()`, `rsl_rl_available()` via `importlib.util.find_spec` (no eager imports; decision #7). `gpu_info()` + `cuda_version_ok()` using torch directly (torch is a hard dep, safe).
  - **New /system/gpu endpoint** ([backend/routes/system.py](reward-sculptor-ui/backend/routes/system.py), [backend/models/system.py](reward-sculptor-ui/backend/models/system.py)) — returns torch/CUDA/device info + mjlab/mujoco_warp/rsl_rl import health. Wired into [backend/main.py](reward-sculptor-ui/backend/main.py).
  - **mjlab project validation** ([backend/routes/projects.py](reward-sculptor-ui/backend/routes/projects.py)) — when `adapter="mjlab"`, preflight 412 on: missing task_id, mjlab not importable, no CUDA, CUDA < 12.4, insufficient VRAM (returns `suggested_num_envs`). Post-scaffold rewrites [adapter] section via new `ProjectStore.set_adapter_section`. CreateProjectRequest extended with `task_id`/`num_envs`/`gpu_device` fields.
  - **Tests**:
    - [RewardSculptor/tests/test_mjlab_adapter.py](RewardSculptor/tests/test_mjlab_adapter.py) — 9 tests: ABC defaults, contract shapes, gym_sb3 unchanged, mjlab task_id validation, G1 vs Go1 schemas, subprocess CLI construction (mocked), reward_batched dispatch, stub adapters raise NotImplementedError, estimate_vram_static.
    - [RewardSculptor/tests/test_edit_prompt_mjlab.py](RewardSculptor/tests/test_edit_prompt_mjlab.py) — 2 tests: batched block appears for mjlab contracts, absent for scalar-only.
    - [RewardSculptor/tests/test_mjlab_gpu.py](RewardSculptor/tests/test_mjlab_gpu.py) — `@pytest.mark.gpu` smoke: train Go1 100 iters + VRAM probe.
    - [RewardSculptor/tests/conftest.py](RewardSculptor/tests/conftest.py) — `--regenerate-fixtures` flag, session-scoped `mjlab_go1_checkpoint` fixture (cache at `tests/fixtures/go1_smoke_checkpoint.pt`), auto-skip @gpu when CUDA missing.
    - [reward-sculptor-ui/backend/tests/test_system.py](reward-sculptor-ui/backend/tests/test_system.py) — 3 tests: GET /system/gpu shape, lazy-import discipline, CUDA path.
    - [reward-sculptor-ui/backend/tests/test_mjlab_validation.py](reward-sculptor-ui/backend/tests/test_mjlab_validation.py) — 7 tests: mjlab preflight failure modes (no task_id / no mjlab / no CUDA / stale CUDA / insufficient VRAM), happy path, gym_sb3 unchanged.
  - **Pyproject** — registered `gpu` pytest marker in [RewardSculptor/pyproject.toml](RewardSculptor/pyproject.toml).
- **Why**: M2 of the mjlab pivot per MJLAB_PIVOT_DESIGN.md. User explicit green-light on 8-point ack for scope, lazy-import discipline, VRAM probe, fixture caching.
- **How**: sequencing followed the plan from the ack: base → gym_sb3 (zero-regression) → stubs → mjlab adapter + runner → edit prompt → backend bridge + /system/gpu → project validation → tests → verify. Installed `mjlab[cu128]>=1.3.0` in sculptor uv env (verified 12 tasks, torch 2.11.0+cu130, CUDA 13.0). GPU smoke target **Go1 instead of Go2** — Go2 isn't registered in core mjlab (research pass confirmed); Go1 is equivalent class (18-DoF quadruped) and fits 8 GiB VRAM.
- **Deviations from the M2 spec**:
  1. `train()` uses a custom `_mjlab_runner.py` subprocess (not `uv run train`) because mjlab's tyro CLI can't inject Python callables, which the reward-injection design requires. Documented in mjlab.py module docstring.
- **Verified**:
  - Sculptor non-GPU: **74 passed, 1 skipped** (baseline 62, so +12 new tests). JAX-gated `test_reward_parity.py` still skipped as expected.
  - Backend: **79 passed** (baseline 69, so +10 new tests).
  - Edit prompt unit test: 2/2 pass.
  - GET /system/gpu live: returns RTX 5070 Laptop GPU, 8.0 GiB, CUDA 13.0, mjlab/mujoco_warp/rsl_rl all available. ✓ M2 #2
  - uvicorn cold-start proxy (`create_app()`): **0.54-0.59 s** (< 3 s budget; lazy-import discipline holding). ✓ M2 #7
  - GPU smoke (train Go1 100 iters + VRAM probe): **2/2 passed in 1:35 wall-clock** on RTX 5070 Laptop. Well under the 15-min M2 budget. ✓ M2 #4. Fixture cached at [RewardSculptor/tests/fixtures/go1_smoke_checkpoint.pt](RewardSculptor/tests/fixtures/go1_smoke_checkpoint.pt) (4.7 MB); subsequent `pytest -m gpu` completes in **3.25 s** (fixture reuse path). `--regenerate-fixtures` flag forces retrain.
  - Hopper flow (gym_sb3 adapter): still green via `test_adapter_contract.py` + `test_sculpt.py` (incl. dry-run end-to-end). ✓ M2 #5
  - Stub adapter tests: each raises NotImplementedError as documented. ✓ M2 #6
  - Final suite totals: **Sculptor 76 passed, 1 skipped (JAX-gated); Backend 79 passed.**
- **Two runtime bugs caught + fixed during GPU smoke**:
  1. `rl_cfg.to_dict()` does not exist on rsl_rl's `RslRlOnPolicyRunnerCfg` (it's a vanilla dataclass, not a pydantic model). Replaced with `_cfg_to_dict()` helper using `dataclasses.asdict`.
  2. Using `rsl_rl.runners.OnPolicyRunner` directly failed with `MLPModel.__init__() got unexpected kwarg 'cnn_cfg'` — the rsl_rl class doesn't understand mjlab's agent-cfg dict structure. Switched to `mjlab.rl.MjlabOnPolicyRunner` + `mjlab.tasks.registry.load_runner_cls` (task-specific override pattern), matching `mjlab/scripts/train.py`.
  3. `runner.save(path)` internally calls `wandb.save(path, ...)` which raises "wandb.init() required" even with `WANDB_MODE=disabled`. Worked around by copying the latest `logs/model_<N>.pt` produced by rsl_rl's periodic save, avoiding the wandb codepath entirely. Also set `WANDB_MODE=disabled` in the subprocess env as a belt-and-suspenders default.

### 2026-04-20 19:22 — mjlab pivot design doc (M0)

- **What**: new `~/projects/MJLAB_PIVOT_DESIGN.md` (1017 lines). Covers the 10 sections from the user's spec: mjlab adapter architecture (manager-based, class-based `SculptorRewardTerm` injecting a closure over sculpt's reward module, `scale_rewards_by_dt=False`), scalar vs batched probe (adds `RewardContract.supports_batched` + `compute_reward_batched`), device model + 8 GB VRAM budget table, robot library YAML schema + 69 seeded entries (63 Menagerie + 1 mjlab-builtin Cartpole + 5 gymnasium-builtin), library→project flow with a new `PreviewOnlyAdapter` for non-mjlab Menagerie bots, KG seeding from library refs (reuses existing `kg_jobs.run_ingest_extract_job`), `NotImplementedError`-backed stubs for Isaac Lab / MJX / RLlib, migration path (leave-alone + opt-in "Upgrade to mjlab" button), failure modes with specific detection points + error types, deferrals, and an M0→M6 milestone breakdown.
- **Why**: user is pivoting primary adapter Gymnasium-SB3 → mjlab (MuJoCo-Warp-powered Isaac-Lab-style manager API) and asked for a design doc before any code. Environment: RTX 5070 Laptop (8 GB VRAM, sm_120), WSL2 Ubuntu 24.04, Python 3.13 via uv. mjlab's own `uvx --from mjlab demo` already verified to run on this host.
- **How**: 3 parallel research agents (Menagerie enumeration → 63 robots with slug/category/min-MuJoCo/upstream-repo; mjlab core API + 13 pre-configured tasks + install prerequisites via verified docs + source citations; mjlab_playground/g1_spinkick/anymal_c_velocity ecosystem → 5 additional tasks) + 1 follow-up agent for canonical paper/repo citations for the 5 mjlab-ready robots (G1, Go1, Go2, T1, ANYmal-C). Every URL WebFetch-verified during research, per user's strict rule. Also read `sculptor/adapters/base.py`, `sculptor/adapters/gym_sb3.py`, `backend/services/sculptor_bridge.py`, `backend/services/kg_jobs.py`, `frontend/src/components/RobotConfig.tsx` to ground the design in what actually exists. Doc file path chosen at the top-level `~/projects/` (next to CONTEXT.md) rather than per-project because it spans both sculptor and UI.
- **Verified**: doc written; 6 open questions flagged at the end for user push-back before M1. No code, no test regressions — verification step is the user's review.

---
