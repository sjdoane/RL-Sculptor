# HANDOFF — start here

**You are picking up the RL-Sculptor project. When the user says "read handoff", read this whole file, then begin Task 1 in §4. Work autonomously through the big tasks in order; after you have implemented enough of a task (Task 1, and ideally Task 2), STOP and run the End-to-End Live UI Verification in §6 — drive the whole app through the browser, screenshot every page, confirm it matches expectations and looks good, then report with the screenshots before continuing.**

Repo root: `/home/samjd/projects`. Two subprojects in one git repo:
- `RewardSculptor/` — the core Python library + `sculpt` CLI (the train→rollout→diagnose→edit loop, the paper knowledge graph, the objective-metric trust pipeline, adapters for MJLab/MuJoCo + Isaac). Run things with `uv run …` from inside `RewardSculptor/`.
- `reward-sculptor-ui/` — FastAPI backend + React/Vite frontend that wraps the library (project lifecycle, robot picker, reward editor, KG browser, world builder, live run streaming, reports).

Current git branch: `ship-20-ux-revamp` (do NOT work on `main`). Keep committing to this branch.

---

## 1. What this project is (30-second version)

RL Sculptor turns a natural-language behavior goal into a trained legged-robot RL policy. An LLM agent **decomposes a mission into stages**, **writes and iterates a reward function per stage**, trains in sim, and — crucially — every generated reward/metric must pass an **objective "trust" metric gauntlet** (a hard-to-game, mostly-offline validation pipeline) before it's allowed to steer a run. It also grounds decisions in a **knowledge graph of research papers**, uses a **library of reference motions**, does **domain randomization + RSI**, and can **export a sim-to-real bundle**. The trust pipeline is the project's differentiator and its most security-sensitive surface.

## 2. Current state (already done + committed this session — verified)

Six commits landed on `ship-20-ux-revamp`, all tested (full library suite `2186 passed`; the only failures are 3 **pre-existing, unrelated** `tests/test_world_runtime.py` cases — a `_Scene.env_origins` attribute error present before any of this work; leave them):

| commit | what |
|---|---|
| `eb81baa` fix(metrics) | Unblocked abstract/compound objective metrics: fixed family mis-resolution (traversal goals no longer resolve to the stationary `jump` family), the synthetic probe fabricating always-in-contact feet, reviewer JSON truncation, and a regression from `4f1dfef` that broke 8 tests. |
| `785f3e5` feat(export) | `sculpt export` bundle now carries a real sim→real hardware contract (joint order, control rate, action→target formula + per-joint scale + default pose, obs layout) + a runnable `inference.py`. |
| `862c902` feat(world) | Authored `train.variations` are now consumed as per-episode mjlab reset events (object mass/friction, course platform heights). Multi-env-correct. |
| `43c970a` feat(world) | Prompt-driven course counts ("4 boxes" → 4) + ball (`sphere`) / soccer-goal (`frame`) objects. |
| `5e535ce` feat(world) | Hybrid LLM world author (`RS_WORLD_LLM_AUTHOR=1`): LLM proposes a full world spec, the existing validators gate it, offline templates are the fallback; deterministic capture/replay so the draft-hash contract holds. |
| `d2662f7` feat(train) | Full physics domain randomization (mass, CoM, PD gains, motor strength, joint damping/armature, friction) applied at the world-independent chokepoint (`_apply_env_spec`) so it reaches **every mission stage and run**, always-on by default; + DeepMimic phase-RSI reset with joint-velocity init. Actuator-type + entity-name guarded (cartpole GPU smoke-train passes). |

Two supporting docs:
- `docs/RESEARCH_DIRECTION.md` — the full, research-validated future-direction analysis (from a lab meeting with a robotics PhD, Lokesh). **Read it — it is the "why" behind Tasks 1–4 below.**
- `.claude/…/memory/*` — Claude's memory notes (physics-dr-rsi, metric-fixed-battery-scoping, abstract-metric-work-status). Optional context.

## 3. The direction / thesis (why the next tasks exist)

The lab's core critique, validated against the literature: **a single scalar reward cannot carry whole-body, time-extended intent** ("the symphony of state × time"), and hand-shaping it produces the oscillation you already see the agent do. The fix is **structure**: make the reward *track a reference motion* (DeepMimic/AMP paradigm) instead of inventing dense scalar terms from scratch, and formalize the stage decomposition as an explicit **modes / hybrid-automaton** with per-mode reward + per-mode trust-metric gating (this is the reviewer's OGMP framework, arXiv 2403.04205, and the thing he liked most about the system). The trust-metric gauntlet is the publishable differentiator — lean into it. See `docs/RESEARCH_DIRECTION.md` §1–§4 for the grounded details, real papers, and honest caveats.

## 4. Task queue — big steps, in priority order

Do these top-to-bottom. Each is a substantial chunk. Fully unit-test each before moving on. **After Task 1 (and, if you get there, Task 2) is implemented and unit-tested, run §6 (Live UI Verification) before continuing.**

### TASK 1 — Reward = track a reference motion + small residual (highest impact) 🎯
**Goal:** when a stage has an attached reference sub-motion, the generated reward's base should be a **tracking reward** (DeepMimic-style: joint pos/vel, body pos, orientation error against the reference), and the LLM should author only **small residual/task terms** (goal, direction, completion gate) + *select/segment* which reference to track — instead of inventing a whole dense scalar reward.
**Why:** converts the fragile, locally-valid scalar-shaping loop (the "graduate-student descent" oscillation) into a dense, well-posed objective that's valid everywhere. This is the single biggest lever and it reuses assets the system already has (reference library + reference-signature injection + RSI/tracking plumbing).
**Where to work:**
- Reward-generation prompt(s) under `RewardSculptor/sculptor/prompts/` (the reward edit/rewrite prompt) — add a tracking-first composition contract when a reference is present.
- `RewardSculptor/sculptor/reference_context.py` + `sculptor/refs/convert.py` (`kinematic_signature`, `clip_to_arrays`) + `sculptor/decompose.py` (already build "REFERENCE MOTION SIGNATURE" blocks) — the signature-injection machinery exists for the *metric*; extend the same pattern to *reward* generation.
- Keep it OPTIONAL/back-compat: goals with no reference keep today's behavior.
**Acceptance:** unit tests showing that (a) with a reference attached, the generated reward is composed as `tracking_terms(reference) + residual` and grounds its thresholds in the reference signature; (b) no-reference goals are unchanged; (c) the full metric-trust gauntlet still passes on the new reward shape. Do NOT weaken the objective-metric gates to make this pass.

### TASK 2 — Formalize stages as OGMP "modes" + gate each mode with the trust metric
**Goal:** make the stage decomposition a first-class schema: `mode = {reference, per-mode reward terms, success predicate}` and `transition = {from, to, guard predicate}` (a hybrid automaton). Run reward synthesis **per mode**, and run the existing nondegeneracy/calibration gauntlet **per mode and per transition guard** before composition.
**Why:** cheapest high-value win — builds on the existing stage decomposition + metric gauntlet, and it's exactly the reviewer's framework (strong research/collaboration story). Combined with Task 1 it *is* the paper.
**Where:** the mission/stage decomposition path (`sculptor/decompose.py`, `sculptor/sculpt.py`, `sculptor/mission_metrics.py`), the env_spec schema (`sculptor/env_spec.py`), and `sculptor/eval/metric_validate.py` (per-mode gating).
**Acceptance:** a modes/transitions schema that validates; per-mode reward generation; per-mode metric gating; a test decomposing a compound behavior (e.g. box-lift: approach → grasp → lift → carry → place) into modes with guards.

### TASK 3 — Privileged teacher → student adaptation (RMA-style) for real sim2real
**Goal:** turn the physics-DR knobs (friction, mass, CoM already in `env_spec`) into an **adaptation signal**: a teacher policy observes ground-truth env params; a student regresses them from proprioceptive history. This is what makes hardware transfer actually good (the user's #1 goal).
**Where:** `sculptor/adapters/_mjlab_runner.py` (obs construction, training loop), `sculptor/env_spec.py`. Medium-high effort — do after 1 & 2.
**Acceptance:** a teacher/student split wired into the mjlab adapter; a GPU-gated smoke test (mirror `tests/test_reference_getup_gpu_smoke.py`).

### TASK 4 — Make DR task-typed + auto-calibrated (not frozen constants)
**Goal:** a per-task-type DR library (force randomization for force tasks, payload for carry tasks, terrain for locomotion) and a DrEureka-style loop that *tunes* the DR ranges from rollouts rather than hardcoding them. Builds directly on the physics-DR substrate from `d2662f7`.
**Where:** `sculptor/env_spec.py`, `sculptor/adapters/_mjlab_runner.py::_apply_env_spec`, the diagnose/edit loop.
**Acceptance:** task-typed DR selection + a mechanism that adjusts ranges across iterations, unit-tested.

**Explicitly OUT of scope for now** (effort sinks / belong to the lab collaboration): vision / VLM "visual behavior adaptation"; a from-scratch motion generator or pretrained whole-body tracker (use the reference *library* as v0); rewriting the agent harness to a custom MCP framework. See `docs/RESEARCH_DIRECTION.md` "Skip for now."

## 5. How to work (conventions + landmines)

- **Branch:** stay on `ship-20-ux-revamp`. Commit in logical, conventional-commit chunks (`feat(...)`, `fix(...)`) with a clear body. End commit messages with a co-author trailer for whoever you are.
- **Run tests (library):** from `RewardSculptor/`, `uv run pytest -q`. Targeted: `uv run pytest tests/test_generated_metric.py tests/test_env_spec.py …`. **Baseline: `2186 passed`, 1 skipped (jax), 3 pre-existing `test_world_runtime` failures — those 3 are not yours; verify any new failure isn't pre-existing by stashing your changes and re-running.**
- **Run tests (backend):** from `reward-sculptor-ui/`, `uv run pytest backend/tests/ -q`.
- **The objective-metric trust pipeline is security-sensitive and deliberately hard to game.** NEVER relax a gate in `metric_validate.py` / `metric_axioms.py` / `metric_gen.py` just to make new code pass. If a legitimate metric is false-rejected, fix the *scoping*, and add a test — don't widen the acceptance surface.
- **Runtime-vs-cfg bugs:** DR/RSI/reset events fire at ENV STARTUP/RESET on the GPU and are invisible to CPU cfg-building tests. When you touch adapter event wiring, add a fake-env CPU test for the math AND run the cartpole GPU smoke (`uv run pytest tests/test_mjlab_gpu.py -q`) — a real short train catches entity-name/actuator-type crashes.
- **After any nontrivial change, adversarially re-verify** (a second independent read for correctness) before declaring done — several real bugs this session were only caught that way, not by the passing suite.
- **`RS_WORLD_LLM_AUTHOR=1`** enables the LLM world author (off by default). LLM env-spec generation and world authoring need Anthropic API access; guard/skip gracefully when absent.

## 6. End-to-End LIVE UI Verification (do this after Task 1 / Task 2) 📸

**Goal:** prove the change works through the real app, not just unit tests — drive the whole UI in a browser, screenshot every page, confirm each is what we expect, and confirm the frontend looks clean and polished. Report with the screenshots.

**Start the app** (from `reward-sculptor-ui/`): first-time setup `uv sync` and `corepack pnpm@9.12.0 install --dir frontend`, then `./run.sh` (backend on `http://localhost:8000`, Vite frontend on `http://localhost:5173`). If `run.sh` auto-opens a browser you don't want, start the two processes manually (backend: `uv run uvicorn backend.main:app --port 8000`; frontend: `corepack pnpm@9.12.0 --dir frontend dev`).

**Drive + screenshot with a headless browser** (Playwright/Chromium against `http://localhost:5173`). Take a full-page screenshot of **every** page/tab and every step of the core workflow, in both light and dark theme where the app supports it. Save them to `reward-sculptor-ui/.ui-verification/<date>/` and reference them in your report.

**Walk the full workflow through the UI (not the API):**
1. **Dashboard** (`/`) — landing page renders, nav works.
2. **Projects** (`/projects`) → **Create Project** (dialog) → open the new **Project Detail** (`/projects/:slug`).
3. **Robot picker** — select a robot capability.
4. **World builder** — open **Author World** dialog; author a world (e.g. "a parkour course with 4 boxes", and separately "push a ball into a soccer goal"); step through clarifications; **preview** → confirm the **3D World Viewer** renders the scene (4 boxes / ball + goal frame); apply/promote. Check the **World tab** (selection, variations, curriculum, lineage).
5. **Rewards tab** — view generated reward(s) + the per-iteration rationale (and, once Task 1 lands, that the reward is tracking-composed when a reference is attached).
6. **Reference picker** — attach a reference motion.
7. **Runs / Missions** — create a run or mission through the UI; launch it; screenshot the **live streaming view + the 3-mode robot viewer**. (Full training needs a GPU and time — it's fine to start it and screenshot the streaming/queued state; don't block on completion.)
8. **Physics tab** — confirm the domain-randomization / RSI settings surface (this is where the new physics-DR knobs should appear or be viewable).
9. **Knowledge Graph tab** — browse the paper graph; try a research-topic action.
10. **Reports tab** — open a report for an existing/finished iteration.

**For every screenshot, verify:**
- The page matches what the feature should produce (e.g. the world viewer shows the *number of boxes you asked for*; the goal frame appears; reward text is present and sensible).
- Layout is clean: no overlapping/cut-off elements, no horizontal body scroll, tables/3D canvases contained, readable in light AND dark theme, consistent spacing, buttons/inputs aligned.
- No broken components, no obviously placeholder/error states, and **no errors in the browser console** (capture the console log).

**Deliverable:** a short report listing each page, its screenshot path, PASS/FAIL vs. expectation, and any visual or functional defect found — then **fix the defects** (backend and/or frontend under `reward-sculptor-ui/frontend/src/`) and re-screenshot until every page passes and looks polished. Only then continue to the next task.

## 7. Key file map (orient fast)

- Reward/metric loop: `RewardSculptor/sculptor/sculpt.py`, `sculptor/decompose.py`, `sculptor/diagnose.py`, `sculptor/eval/metric_validate.py`, `sculptor/eval/metric_gen.py`, `sculptor/eval/metric_axioms.py`, `sculptor/prompts/*.md`.
- References/RSI: `sculptor/reference.py`, `sculptor/reference_context.py`, `sculptor/refs/` (`convert.py`, `track.py`, `perturb.py`).
- Env/DR/adapters: `sculptor/env_spec.py`, `sculptor/env_gen.py`, `sculptor/adapters/_mjlab_runner.py` (`_apply_env_spec`, `reset_joints_to_reference`, `_primary_robot_entity`, `_is_pd_actuated`), `sculptor/adapters/mjlab.py`.
- World builder: `sculptor/world/` (`author.py`, `world_spec.py`, `compiler.py`, `randomization.py`, `llm_author.py`, `capabilities.py`), prompt `sculptor/prompts/gen_world_author.md`.
- Sim2real export: `sculptor/export.py`.
- UI backend: `reward-sculptor-ui/backend/` (routes: `worlds.py`, `missions.py`, `runs`/`policies.py`; services: `world_store.py`, `mission_jobs.py`, `run_manager.py`, `metric_store.py`).
- UI frontend: `reward-sculptor-ui/frontend/src/` (pages under `components/`: `WorldTab`, `WorldViewer3D`, `RewardsTab`, `RunsTab`, `KnowledgeGraphTab`, `PhysicsTab`, `ReportsTab`, `RobotViewer`, dialogs `AuthorWorldDialog`/`NewMissionDialog`/`NewRunDialog`/`ReferencePickerDialog`; routing in `App.tsx`).

**First action on "read handoff": begin Task 1.**

---

## 8. Progress update 2026-07-21 — Task 1 implemented (Codex)

Task 1 is implemented and ready for the §6 live UI verification. An attached
stage reference now deterministically replaces the pristine v0 placeholder
with a compact, 16-phase DeepMimic-style base (joint position, joint velocity,
root height, and orientation when available). The LLM can edit only scalar and
batched residual hooks; reference targets, target hash, tracking kernels,
tracking weight, composition wrappers, fall gate, and residual cap are frozen
by AST/content validation. Manual UI edits enforce the same invariant. Stages
without an attached reference stay on the historical reward path.

The Rewards tab now renders the composition explicitly (reference tracking vs.
maximum residual, clip id, and target hash). The initial broad suite exposed
the three previously documented World-runtime failures: scenes without
`env_origins` had lost the correct zero-origin fallback. That compatibility
regression is fixed for NumPy and Torch runtimes.

Verification at this checkpoint:

- core focused reference suite: 51 passed;
- UI focused reward suite: 25 passed;
- full core suite: 2,178 passed / 1 optional-JAX skip;
- full UI backend suite: 562 passed;
- frontend TypeScript check and production build: passed (2,761 modules);
- scoped Ruff, compileall, and `git diff --check`: passed.

Next action: commit this slice, then execute §6 through the real browser before
starting Task 2. In particular, create/attach a reference and confirm the
Rewards composition card appears in both themes and the browser console stays
clean.
