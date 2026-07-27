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

**First action on "read handoff": read §9 for current state, then begin the top open task.**

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

---

## 9. Progress update 2026-07-25 (Claude Opus 5) — read this first

### What landed this session

| commit | what |
|---|---|
| `343d389` feat(llm) | **Every in-system role moved to `claude-opus-5`** (was fable-5). The `calibration` role stays on `opus-4-8` — RESEARCH_GAP_ANALYSIS §3.5 requires it model-DISJOINT from `metric_gen` or a shared blind spot passes L2 agreement unnoticed. That invariant had no test; it has one now. Also routed the UI actuator-datasheet extractor through the registry (it hardcoded `opus-4-7`, bypassing provenance). |
| `ad02825` fix(diagnose) | Diagnose now renders `held_out_metric_observables` (names only) beside `expected_info_keys`. See "the iter-29 finding" below. |
| `d4a9134` feat(refs) | **`sculptor/refs/compose.py` — compose novel motions from spans of several solved clips.** The headline capability. |
| `cfaa8f6` feat(refs) | `compose_and_register` — composites become first-class library clips (tier K, multi-parent provenance, license conjunction). |
| `870423a` feat(ui) | Compose UI in the reference picker + **a real nested-modal layout fix** (see below). |
| `bf641c0` feat(refs) | `sculptor/refs/dr_calibration.py` — behavior-matched push sizing + task-typed DR. |

**Test state: `2294 passed, 1 skipped (jax), 0 failed`** for the library
(`MUJOCO_GL=egl uv run pytest tests/ -q --ignore=tests/test_refs_preview.py`),
**579 passed** for the UI backend. Note the baseline in §5 is stale: the 3
`test_world_runtime` failures it says to expect are **fixed** — expect zero.

### The iter-29 finding (closes out the Codex showcase campaign)

The `g1-lab-showcase-weave-and-stop` campaign ran 29 iterations and **regressed**.
Peak objective fitness was **0.247 at iter 13**; iters 27/28/29 all scored
**0.0** with `order_ok_frac` ~0.0. Each "recovery" iteration added another
geometric constraint (clearance chord → through-disk → predicate depth) and
route completion collapsed. This is the scalar-shaping oscillation
`docs/RESEARCH_DIRECTION.md` predicts, and it is the evidence for the pivot —
**do not launch iter 30 of the same loop.**

Iter 29 also burned itself emitting `requires_env_extension` for a `box_contact`
field that **already existed** as `contact__forbidden__{0..3}`. Root cause: the
diagnoser was told held-out channels were "structurally excluded" from its
evidence but never that they EXIST or what they are named, so it inferred
contact from box VELOCITY — blind to a robot leaning on a stationary box, which
is exactly the failure it exhibited. `build_partition_prompt_block` already
names those channels but was wired only into `edit.py`; the diagnoser is the
stage that emits `requires_env_extension`. Fixed in `ad02825`.

### The new capability: compose (`sculptor/refs/compose.py`)

Every reference consumer was **single-clip**. A goal whose motion is not
substantially contained in some single clip had no reference and fell through
to `refs.synth` (an LLM sketch) or ran blind. Compose stitches spans of
*different* solved clips into one candidate, so every frame stays real solved
data. Verified on the real 6,013-clip library: `running_on_spot` +
`one_leg_jump` + `kicking1` (mixed 60/120 fps) → a 3.70 s jumping kick that
exists in no single clip, seams at 0.023 and 0.009 rad.

Continuity is the whole problem — SE(2)-align each seam (the kick clip needed
−1.52 rad or the root teleports), smoothstep cross-fade (C1, so the blend adds
no velocity step), recompute velocities from the composed positions, reorder
joints by NAME (positional composition silently swaps limbs).

**Honest boundary, recorded in the artifact:** kinematic only. Momentum is not
conserved across a seam. Output is `certified: false` at tier K, and
`refs.track`'s existing Tier-D physics run is the admission filter. **Compose
then track; never compose and trust.**

### A real UI bug worth knowing about

`.rs-modal` animates `transform` with fill-mode `both`, so it keeps a non-none
transform forever — which makes it a **containing block for `position: fixed`
descendants**. Every modal opened from inside another modal had its fixed scrim
resolve against the parent dialog's box instead of the viewport, clipping its
own header and footer. `Modal` now portals its scrim to `<body>`. Found only by
driving the browser — the automated overflow audit passed it.

### Demonstration (evidence in `reward-sculptor-ui/.ui-verification/`)

`2026-07-25-compose/` — all 12 pages, light + dark, zero console errors, no
horizontal overflow, plus the compose flow end to end.
`2026-07-25-demo/` — a real GPU run launched through the UI whose reference is
the composed `novel-running-jump-kick--g1`. Both composites rank **above** the
real kick clips for "jumping kick" (25.75 / 19.61 vs 18.51).

### Open, in priority order

1. **TASK 2 — OGMP modes** (§4). Still the best value/effort left and it is the
   reviewer's own framework. `compose`'s per-segment structure is a natural
   fit: one composed segment ≈ one mode, and `meta.composition.segments`
   already carries the boundaries a transition guard would key on.
2. **Wire `dr_calibration` into the live pipeline.** The module is pure and
   tested but nothing calls it yet — `sculpt.py` / `_mjlab_runner.py` should
   pull the task-typed profile and the momentum-matched push instead of the
   flat always-on set.
3. **Tier-D certify a composite.** `refs.track` exists and compose emits
   `certified: false`; nothing has closed that loop on a composed clip yet.
4. **TASK 3 (RMA teacher/student)** and **TASK 4** in §4 are untouched.

---

## 10. Progress update 2026-07-26 (Claude Opus 5) — the Tier-D root-frame bug

### Why no clip had ever been certified

Closing item 3 above surfaced a bug that had silently capped the whole
reference library at tier K. `compute_tracking_errors` compared the clip's
`root_pos_z` **directly** against the rollout's `root_link_pos_w[..., 2]` —
but retargeted AMASS zeroes root translation, so a clip's `root_pos_z` is an
*excursion* near 0 while the rollout reports a ~0.74 m **world** height.

The two are not in the same frame. The resulting `root_z_rmse_m` measured the
robot's standing height, not its tracking:

| | before | after | gate |
|---|---|---|---|
| `root_z_rmse_m` | 0.7439 | **0.0251** | < 0.12 |
| `mean_joint_err_rad` | 0.1685 | 0.1685 | < 0.35 |

Measured library-wide: **5798 of 6015 g1 clips are origin-relative** (217 carry
absolute heights), and **all 6015 sat at tier K** — the `root_z_rmse_m < 0.12`
gate was unsatisfiable for 96% of the library since it was written.

The reward was never wrong: `compute_reward_batched` compares mjlab's
`base_height_delta` (measured from each env's own reset anchor, supplied by
`_mjlab_runner.py:661`) against the reference's excursion from its first
frame. Its `root_tracking` term ran 0.776 → 0.953 over the real run — healthy,
never saturated. So **the scorer was certifying something different from what
the reward optimized**; the fix makes them agree.

### What the fix is (and is not)

`ROOT_Z_RMSE_THRESHOLD_M` is **unchanged at 0.12**. What changed is the frame:
`clip_root_frame()` resolves a clip as `absolute` or `origin_relative`
(explicit `clip["root_frame"]` wins, else the `ORIGIN_RELATIVE_MAX_ROOT_Z_M
= 0.30` band), and only origin-relative clips have the constant offset divided
out. That offset is recorded as `root_z_offset_m` on the certificate rather
than dropped, so an auditor can always see what was *not* measured.

Guardrails, in `tests/test_refs_track.py`:
- a flat rollout that never leaves the ground while the reference hops still
  **fails** (`test_origin_relative_scoring_still_fails_a_flat_rollout`);
- an inverted rollout (crouches as the reference rises) still **fails**;
- an absolute clip still charges a constant 20 cm offset as real error, so the
  pre-existing intent is untouched.

### …which then exposed that the gate behind it does not discriminate

With the frame bug fixed, the composite "passed" — `mean_joint_err_rad` 0.1685
against 0.35. **That pass was not real.** Scoring trivial baselines against the
same reference:

| baseline | mean\|err\| rad | passed 0.35? |
|---|---|---|
| trained policy | 0.1685 | ✓ |
| the same rollout played **backwards** | 0.1691 | ✓ |
| rollout **phase-shifted** half a clip | 0.1689 | ✓ |
| rollout's mean pose held **constant** | **0.1624** | ✓ (beats the policy) |
| clip's mean pose held constant | **0.1486** | ✓ |
| all joints at zero | 0.3042 | ✓ |

Mean absolute error is blind to temporal structure. On a clip whose joint
excursions (0.18 rad std) are small next to the threshold, **standing still
passes** — and the trained policy's own motion amplitude was 0.050 rad, 28% of
the reference's.

So certification now runs a control: `static_baseline_err_rad` is what the
rollout's own time-averaged pose would have scored, and `feasible` requires
beating it by `STATIC_BASELINE_RATIO_MAX = 0.80`. This makes the gate strictly
harder, never easier. `motion_ratio` is reported alongside. A motionless
reference skips the control (vacuous, not a failure), as does root-only
scoring.

**Current honest verdict on the composite: NOT certified.**
`beats_static_baseline: false`, `motion_ratio: 0.278`, `feasible: false`.

### Root cause: the phase clock ran on the training budget

`build_track_project` set `episode_len_steps = int(steps_per_iteration)`. But
`steps_per_iteration` is mjlab's `max_iterations` — a count of **PPO updates**,
not env steps (this module's own §DEFAULT_ITERATIONS docstring says so). At the
2000 default against ~500-step episodes, the phase clock only ever reached
`500/2000 = 0.25`: the policy never saw past the first quarter of the reference
— never reached the jump or the kick of a three-phase composite — and the
reference played at quarter speed.

Now derived from wall time: `round(duration_s * DEFAULT_CONTROL_HZ)`, i.e.
**185** steps for the 3.70 s composite at mjlab's 50 Hz, not 2000. A
re-certification with the corrected clock is running.

### Diagnosed root cause of the tracking failure — OPEN DECISION

After fixing the clock (wall-time), the episode cap, and the control rate, the
learning curve barely moved: v2 (phase reaching 0.5) and v3 (phase reaching
1.0) converged to the SAME `joint_tracking` 0.5447. A policy that is actually
tracking cannot be insensitive to how much of the reference it sees. So the
phase bugs were real but were not what is holding tracking back.

The mechanism is in `_mjlab_runner.py` §11 and its injection site (~2330):
reward injection adds `sculptor_primary` at **weight 1.0** and attenuates the
task-shipped terms to a **0.3x realism floor** — it does not remove them. The
G1 velocity task ships **14** such terms (visible as `reward_term__*` in any
rollout npz): `pose`, `upright`, `track_linear_velocity`,
`track_angular_velocity`, `foot_clearance`, `foot_swing_height`, `air_time`,
`foot_slip`, `soft_landing`, `action_rate_l2`, `angular_momentum`,
`body_ang_vel`, `dof_pos_limits`, `self_collisions`.

Several of those — `pose` above all, plus `upright` — are explicit
regularizers toward the robot's NOMINAL pose. Fourteen of them at 0.3x
against one tracking term at 1.0x is a direct explanation for a policy that
settles into a near-default posture: measured `motion_ratio` 0.278, i.e. the
rollout reproduced 28% of the reference's joint amplitude.

The floor exists for a good reason (§11: without it an early policy learns to
fall immediately to dodge realism penalties). But for a Tier-D *tracking*
certification it is competing with the thing being certified. This is a design
decision, not a bug to quietly patch — the options are roughly:

  a. raise `sculptor_primary`'s weight for tracking runs specifically;
  b. drop the floor to safety/termination terms only when a reference is
     attached, keeping `self_collisions`/`dof_pos_limits` but dropping `pose`
     and the command-tracking terms;
  c. leave it and accept that Tier-D certifies "tracks while remaining a
     plausible robot", with the static-pose control as the honest bar.

**Do not skip the static-pose control to make (c) pass.** That control is the
only reason this failure was visible at all.

### Landmine that cost 13 GPU-hours

`refs track` writes its throwaway project to `<clip_dir>/tierD_work/`, and
`train/logs/model_*.pt` **survive a crash**. After the 2026-07-26 reboot the
policy was recoverable and only needed rollout + scoring — do not restart a
certification from zero without checking `tierD_work/train/logs/` first.

### Timesteps: what the literature says, and what the UI was showing

New module `RewardSculptor/sculptor/refs/timing.py` — the literature snapshot,
with sources in its docstring. Short version:

* **Control at ~50 Hz is the sim2real convention** for legged/humanoid RL
  (MuJoCo Playground, Booster Gym). That is what our mjlab G1 task uses.
* **Physics 200–500 Hz via decimation**, not a slower integrator. mjlab: 0.005 s
  × decimation 4.
* **On hardware the PD loop underneath runs ~1 kHz**; the policy emits joint
  targets, not torques, so it does not need that rate.
* **Series elasticity moves the floor by an order of magnitude.** The spring
  adds a fast mode needing `dt ≪ 1/ω_n` — SEA force-control studies use ~1e-5 s.
  This is also why Brax needs finer steps than MuJoCo: MuJoCo's constraint
  solver is implicit in the velocity update and tolerates steps that blow up an
  explicit spring model. **A rigid-actuator G1 at 200 Hz is not evidence that
  200 Hz suffices for an SEA model of the same robot.** `validate_timing(...,
  series_elastic=True)` flags that rather than letting it pass.

Conclusion: 200/50 Hz is correct and intentional for the rigid-actuator G1 we
simulate. It should be revisited *per-task* only if an actuator model with
series elasticity is added — not calibrated per-clip.

`validate_timing` immediately found a real issue on the composite: control at
50 Hz cannot represent a 120 fps reference above 25 Hz (Nyquist), so fast
transients alias — and that composite's third phase is a kick.

**The UI was showing the wrong number.** The Physics tab reported the MJCF's
compiled timestep. The Unitree G1 XML declares no `<option timestep>`, so it
compiled at MuJoCo's 0.002 s default and the tab said **500 Hz**, while mjlab's
task config sets 0.005 s and training ran at **200 Hz**. The control rate — the
one that must match the hardware loop — was not displayed anywhere. There is
now a Timing card (physics / control / decimation + `validate_timing`
findings), and the MJCF row is marked "overridden" so editing the XML timestep
no longer looks like it changes what trains. Unresolvable tasks render
"unknown" rather than a default.

### Paper corpus: full text is stored, but only partly *read*

Audited all Paper nodes after OGMP (2403.04205) turned out to be missing from
the KG entirely. State now:

| check | result |
|---|---|
| Paper nodes | 167 |
| stub titles (`arxiv:…`, from a rate-limited ingest) | 0 |
| missing full text on disk | 0 |
| abstract-sized only | 0 |
| corpus | 11.9 MB, p50 63.7 KB/paper |

So the sidecars under `~/.local/share/sculptor/kg/pdfs/*.txt` are real bodies,
not abstracts. Two places consume them differently, and the difference matters:

* **`kg/extract.py`** (mines Techniques / FailureModes / RewardComponents) reads
  a bounded middle slice: skip 1000 chars, then `MAX_EXCERPT_CHARS = 28_000`.
  Measured coverage across the corpus is **p10 26% / p50 45% / p90 82%, and
  zero papers are read end-to-end.** The worst are the long ones —
  `2405.15568` at 9%, `2401.16889` at 12%. Anything the authors put after the
  ~28 K mark (most Results, Discussion, and appendix tables) never reached the
  extractor. Not a bug, but it bounds what the KG can possibly know.
* **`kg/query.py`** ranks papers by `paper_embed_text()` =
  `title + abstract + rationale` (`query.py:512`). Retrieval never touches
  `full_text_path` — grep it, there are no hits in `query.py`, `research.py`,
  or `retrieval_log.py`. A paper whose body answers a question but whose
  abstract does not mention the topic will rank poorly no matter how good the
  body is.

Neither is worth fixing blind. Re-extracting 167 papers at full length is a
large LLM spend, and chunk-embedding bodies changes retrieval ranking
everywhere — both want a deliberate decision, not a silent upgrade.

**OGMP was ingested this session** (35,231 B body; metadata healed by hand
because the arXiv API 429'd through all 5 retries — `heal_stub_titles()` in
`kg/ingest.py:272` is the supported path once the API is reachable).

Two things fall out of actually reading it that the abstract does not tell you:

1. **The paper gives no explicit transition-guard equations.** Mode transitions
   emerge implicitly from the learned policy. Our `sculptor/modes.py`
   `Guard`/`predicate` formalization is therefore an **extension beyond** OGMP,
   not a reproduction of it — describe it that way.
2. **Its tracking reward weights orientation.** Eq. 8 is
   `0.475·e^(−5‖er_p‖) + 0.475·e^(−5‖er_o‖)` — position and orientation at
   equal weight, plus `0.05·e^(−0.01‖u_t‖) − 0.3·𝟙(non-toe contact)`.

   **Closed, 2026-07-26.** The gap was narrower than first written: `refs/
   track.py` has *two* reward generators, and only one had orientation.
   `generate_tracking_residual_reward_source` (reference-run path) already
   tracked projected gravity; `generate_tracking_reward_source` — the one
   `build_track_project` actually uses for Tier-D — did not, so every
   certification attempt trained without it. Verified against the live
   `tierD_work/rewards/current.py`: no `REFERENCE_GRAVITY`, no
   `tracking_orientation`.

   Now wired: `projected_gravity_from_quat()` derives the target from the
   clip's `root_quat_wxyz` (mjlab publishes `projected_gravity_b` every step,
   so that is the common frame), and it feeds both the scalar and batched
   paths. Projected gravity rather than raw quaternion because it is
   yaw-invariant — retargeting zeroes root translation, so a heading offset is
   not an orientation error.

   `TrackingErrors.orientation_err` reports it, and **deliberately does not
   gate**. Nothing has ever passed Tier-D, so there is no evidence for an
   achievable threshold; inventing one would be a made-up number. Set it from
   data once a run certifies. `test_orientation_does_not_gate_certification`
   pins that on purpose — a fully inverted rollout is still `feasible` today.

---

## 11. Progress update 2026-07-26 — the gauntlet runs PER MODE and PER TRANSITION

Task 2 (§4), the gating half. `sculptor/modes.py` writes the hybrid automaton
down and its own docstring says it stops short of running the gauntlet against
it; this is that piece. Per-mode reward **authoring** is a separate, concurrent
workstream (`sculptor/mode_rewards.py`) — untouched here.

### What landed

`sculptor/eval/mode_metrics.py`, `prompts/gen_mode_metric.md`,
`prompts/review_mode_metric.md`, `tests/test_mode_metrics.py` (47 tests), plus
two additive parameters on `generate_objective_metric`.

1. **Per-mode scoring.** `score_modes(fn, arrays, behavior, meta, graph)` slices
   a rollout to each mode's window and scores each slice, with `metrics_by_mode`
   so a mode is graded by its OWN metric. On the test fixture — a rollout that
   moves for its first half and is frozen for its second — the episode score is
   **0.50 and looks survivable while `strike` scores 0.000**. That is the whole
   claim: the degenerate half gets an address instead of being averaged away.
   The iter-29 campaign in §9 is what this is for; nothing in that loop could
   say which part of the route had collapsed, because nothing scored a part.
   Slices cross the real untrusted-code boundary, not just a test callable: one
   test drives `score_modes` with a metric loaded through
   `load_generated_metric`, so the seccomp worker's array-IPC wire format is
   exercised on sliced arrays.

2. **Wall time, not frame counts.** Windows come from `mode_phase_windows`
   (SECONDS), and `resolve_step_dt` READS the rollout's own `step_dt` and
   **raises rather than defaulting to 0.02**. A 120 fps reference and a 50 Hz
   rollout describe the same 2 s; matching by frame count would put a boundary
   at frame 120 of a 100-frame rollout. This repo has shipped a silently-wrong
   phase clock twice (§10: `episode_len_steps` from `max_iterations`; the
   Physics tab's 500 vs 200 Hz), and a wrong clock here is worse than a crash —
   every mode still gets a plausible score, attributed to the wrong mode.

3. **An unentered mode is not a zero.** A mode the episode ended before reaching
   is `scored: False, score: None`. "Never performed" and "performed
   degenerately" are opposite diagnoses — the first says the policy stalls
   earlier, the second says this mode's reward is wrong — and collapsing them to
   0.0 destroys exactly what per-mode scoring adds.

4. **Per-transition guard firing.** `check_transitions` answers, per guard,
   whether it fired. Phase guard: *"the approach→strike guard fires at 1.000s
   but the rollout is 40 frames (0.800s) long — the policy never left
   approach."* Predicate guard: the verdict from `mission_runtime`'s isolated
   seccomp evaluator, the SAME one that runs mission success criteria and never
   an inline eval, as `modes.Guard` requires. No namespace → `fired: None`, an
   explicit abstain that is never collapsed into `False`.

5. **The existing quality machinery, re-pointed one mode at a time.**
   `validate_mode_metrics` runs the **unmodified** `validate_generated_metric`
   per mode with the mode's own goal and its own CROPPED slice of the reference
   (`mode_reference_clip`, cut at the composition's real seam frames), so the
   perturbation suite asks "can this metric tell THIS PHASE from this phase
   reversed / frozen / shuffled". `calibrate_mode_metrics` runs
   `calibrate_task_derived` per mode, which is what points the metric-blind
   ladder author AND the metric-blind gaming author at the sub-behavior. That
   last one is the least obvious and the sharpest test in the set: an
   episode-wide gaming policy has to fool every phase at once, a mode-scoped one
   only has to fool a two-second window. It defaults to `adversarial=True`
   here — new surface starts strict rather than being loosened into strictness
   later.

6. **A per-mode report.** `mode_gauntlet_report` + `render_mode_report` key every
   finding by mode and print the episode score **last** — leading with it is the
   thing that hid the failure in the first place.

### The per-mode goal must NOT be the episode goal

`resolve_behavior_family` word-matches the whole goal string. Splice the episode
goal into a mode goal and every mode of "run up and kick the ball" resolves to
family `kick` — so the approach mode's non-degeneracy is anchored against a kick
positive and an honest run-up metric is false-rejected. `mode_goal_text`
therefore returns the mode's OWN goal only (a caller-supplied `mode_goals` entry,
else the humanized mode name), and the episode goal reaches the author as
*context* through the prompt appendix, where it informs the prose without
steering family resolution. `goal_source` records which of the two was used, so a
report never implies more grounding than there was.

### What I deliberately did NOT gate, and why

* **Guard firing.** Reported; gates nothing. Whether a guard fired is a fact
  about the ROLLOUT (did the episode last long enough, did the predicate hold);
  this package's gates judge METRICS. A policy that stalls in mode 1 is a thing
  to diagnose, not evidence that mode 1's metric is bad.
  `test_guard_firing_is_reported_but_never_fails_a_metric` pins it.
* **Mode coverage.** The fraction of a mode's window a rollout reached is a
  number in the record and nothing more. There is no data behind any particular
  coverage floor — same reasoning as `TrackingErrors.orientation_err` in §10.
* **`worst_mode_gap`** (episode score − worst scored mode). This is the exact
  signature of "a degenerate sub-motion averaged away", and it is still only
  reported. Where to draw a line on it is unknown, so drawing one would be an
  invented number.
* **No new synthetic positive, anywhere.** Running the gate "per mode" changes
  WHICH GOAL is passed and nothing else.
  `test_per_mode_validation_adds_no_synthetic_positive_to_the_fixed_battery`
  asserts the `archetype_scores` and `gates` produced through the mode layer are
  identical to a direct `validate_generated_metric` call. Nothing in
  `metric_validate.py` / `metric_axioms.py` / `metric_gen.py` was relaxed; the
  only change to `metric_gen.py` is two optional `*_appendix` parameters that ADD
  system-prompt text (default `None` → byte-identical, pinned by a test) and can
  neither remove a rule from the shared rubric nor touch a gate.

### Found while doing this, and out of my scope

1. **`modes.py` accepts a mode finer than the control rate.** `Mode("wind_up",
   (0, 1))` on a 120 fps reference is 0.0083 s — below one step at 50 Hz.
   `validate_mode_graph` only checks `lo < hi` in FRAMES, so it validates, and
   nothing downstream can score it. `mode_metrics` reports it distinctly
   (`shorter_than_one_step`) rather than mislabelling it "never entered", but the
   real fix is for the automaton to know the control rate, or at minimum for
   `validate_mode_graph`'s docstring to say frame ranges are unchecked against
   it. Same family as the Nyquist finding in §10.
2. **A phase guard is a clock, and "fired" invites a misread.** `modes.py` calls
   a guard that never fires a diagnosable event, which is right — but "fired"
   means only that the episode lasted long enough to reach the handover time,
   NOT that the sub-behavior succeeded. What the mode achieved is what
   `score_modes` measures. Worth one sentence in `Guard`'s docstring; today a
   reader has to infer it.
3. **`Mode` carries no goal text.** `Mode(name, frame_range, reward_terms,
   success_predicate)` — a per-mode metric needs a goal, and
   `modes_from_composition` derives names from free-text composition segment
   labels (`mode_2` when unlabeled). A `goal_text` field would remove the
   caller-supplied `mode_goals` map entirely.
4. **The isolated criterion evaluator is private.**
   `mission_runtime._evaluate_success_criterion` / `._build_criterion_namespace`
   are the only isolated expression evaluator in the codebase, and `modes.Guard`
   explicitly delegates predicate guards to them. `mode_metrics` imports the
   private name knowingly — a second evaluator would be a second security
   boundary to audit — but they want a public alias.
5. **Nothing derives a predicate guard from data yet.** `modes_from_composition`
   emits phase guards only, so every derived automaton's handovers are pure
   clocks. Predicate guards work end to end here (tested against the real
   sandbox) but have no producer.

### Verification

`tests/test_mode_metrics.py`: **47 passed**. Full library suite
(`MUJOCO_GL=egl uv run pytest tests/ -q -m "not gpu"`): **2507 passed, 1 skipped
(jax), 7 deselected, 0 failed**.

GPU-marked tests were deselected on purpose — a ~6.5 h `sculpt refs track`
certification was live on the GPU and a smoke train would have contended with
it. Note that `conftest.py`'s auto-skip does not cover this case: it skips
`@gpu` only when CUDA is ABSENT, and CUDA is present precisely because the
certification is running, so a plain `pytest tests/` would have started a
second GPU job. Re-run those 7 once the GPU is free.

## 12. Progress update 2026-07-26 — per-mode reward AUTHORING

Task 2 (§4), the authoring half, concurrent with §11's gating half.
`sculptor/modes.py` writes the automaton down; §11 scores against it; this turns
it into reward code. File ownership was disjoint by design — nothing here
touches `sculptor/eval/**`, nothing in §11 touches `sculptor/mode_rewards.py`.

### What landed

`sculptor/mode_rewards.py`, `tests/test_mode_rewards.py` (50 tests),
`tests/test_modes_cli.py` (13), and a `sculpt modes` CLI (`show` / `scaffold` /
`author`).

1. **The gating is derived, not authored.** The obvious approach — prompt an LLM
   for one module handling all the modes — puts the gating inside generated
   code, where it is unverifiable and silently wrong when the phase clock is
   off. **Both real Tier-D failures in this repo were clock bugs, not reward
   bugs** (§10). So `generate_mode_reward_scaffold` emits the clock, the windows
   and the dispatch deterministically from the graph, and the LLM fills one
   function body per mode. `apply_prompt_edit`'s KG grounding, repair retries and
   pre-flight probes all keep working unchanged — the only new thing is what the
   prompt asks for.

2. **Two functions per mode, because mjlab only calls one of them.**
   `adapters/mjlab.py:670` dispatches to `compute_reward_batched` and treats its
   absence as a reward-contract violation. A scalar-only scaffold could never
   have trained. Each mode now has a `_batched` twin, and `authored_modes`
   requires BOTH — a mode written only in the scalar half evaluates correctly in
   replay and pays exactly zero in training, which reads as a *bad* reward
   rather than a missing one.

3. **Per-env masking, not a scalar clock.** mjlab's envs reset independently, so
   at any step they sit at different points in the automaton. `_mode_masks`
   mirrors `active_mode` term for term and a test sweeps 260 steps asserting the
   two paths never disagree about which mode owns an instant — a rollout is
   SCORED through the scalar path (§11) and TRAINED through the batched one, and
   disagreement would mean grading terms the policy was never paid for.

4. **`torch.where`, not `mask * value`.** Every mode's function runs for every
   env before masking (that is what makes it vectorizable), but a mode's terms
   are only defined inside its own window. `0.0 * nan == nan`, so a multiply
   lets one out-of-window env poison the whole batch. Measured on a mode whose
   term is `sqrt(t - 1.0)`: a multiply spreads nan to **63 of 260 steps**,
   `where` gives **0**, and the in-window value is untouched — a numerical bug
   inside the window still surfaces, which is the right direction to fail in.

5. **The tracking backbone, so a scaffold is trainable before it is authored.**
   Pass `clip=` and the module carries the same two-Gaussian-plus-orientation
   reward `refs/track.py` emits — the one that took a Tier-D rollout from 28% of
   the reference's joint amplitude to 85%. Without it a scaffold pays zero until
   every mode is authored, and even then nothing tells the policy to follow the
   reference. With it, `sculpt modes scaffold` produces a trainable reward
   immediately and authoring adds mode-specific task terms on top. That layering
   is OGMP's own shape: one oracle tracked throughout, a per-mode objective
   above it. The tracking clock stays GLOBAL over the composite (the automaton
   decides which TASK terms apply; it does not change what the robot should be
   tracking) — re-anchoring phase per mode is defensible but would mean the
   backbone is no longer the version that has been measured.

6. **The prompt budgets for its own limit.** `apply_prompt_edit` hard-rejects a
   prompt over 2000 chars (`edit.py:2042`) and a behavior goal is free text a
   user typed. Unbudgeted, a long goal failed at the *end* of the authoring
   call, after the KG query, with an error about a character count. The fixed
   part is now 1500 chars and the free text is truncated visibly, in
   `--print-prompt`, before any model is called.

### Verified live

`sculpt modes show --clip-id novel-running-jump-kick--g1` reads the real
composite's own provenance: 3 modes @ 120 fps — `approach` [0,150) = 0.000–1.250s
from `50002_running_on_spot_poses_60_jpos`, `launch` [150,300) = 1.250–2.500s
from `50002_one_leg_jump_poses_60_jpos`, `strike` [300,444) = 2.500–3.700s from
`0016_kicking1_poses_120_jpos`. `sculpt modes scaffold` on that clip emits a
386-line module: `N_JOINTS=29`, `N_PHASE=32`, `REFERENCE_DURATION_S=3.7`,
`ORIENTATION_ERR_WEIGHT=4.0`, `supports_batched: True`. Both paths run finite
over the whole episode, and the scalar/batched `joint_tracking` and
`orientation_tracking` agree to **4e-8** (`root_tracking` differs by
construction — scalar absolute z vs batched delta-from-frame-0, per §10's
origin-relative finding). The generated module clears `edit.py`'s real
`_call_compute_reward_batched` probe on an mjlab-shaped contract, on one
declaring no info keys, and with the backbone present.

### Still open in this lane

1. ~~**No end-to-end LLM authoring pass has run.**~~ **Closed — see §13.**
2. **The backbone duplicates `refs/track.py`'s emitter.** Both build the same
   phase tables and the same two-Gaussian formula. `mode_rewards` imports the
   pure helpers (`downsample_phase_targets`, `projected_gravity_from_quat`,
   the weights) but re-emits the source text. Factoring out one shared fragment
   is the right fix and was deliberately NOT done while a `refs track` job had
   `track.py` loaded — worth doing next, since two copies of a phase clock is
   exactly the shape of the bug that has bitten twice.
3. **`Mode` still carries no goal text** — same finding as §11's #3, reached
   independently. `generate_mode_reward_scaffold` takes a `goal_by_mode` map and
   `mode_authoring_prompt` takes `mode_goal` for the same reason. A `goal_text`
   field on `Mode` would delete both.

### Found while doing this — a latent bug in `refs/track.py`

**The Tier-D tracking reward's SCALAR `compute_reward` would crash on the real
mjlab contract.** It slices `qpos[7:7+N_JOINTS]` assuming a full MuJoCo vector
(7 free-joint DOFs + N actuated), but `MjlabAdapter.reward_contract()` for
`Mjlab-Velocity-Flat-Unitree-G1` declares **`qpos: (29,)`** — actuated joints
only. Feeding that layout in raises `qpos too short for 29 tracked joints`.

Found the hard way: the identical slice in my backbone was rejected by
`edit.py`'s scalar pre-flight probe during a real `sculpt modes author` run,
*after* the model had already been called. `mode_rewards` now takes the trailing
`N_JOINTS` (correct for both layouts, and the same slice the batched path uses)
and prefers `info["base_height_delta"]` for root height, falling back to
`qpos[2]` only when qpos really is a full MuJoCo vector.

`track.py` is not currently *broken* by this, for two reasons that are both
accidents: `build_track_project` writes `rewards/current.py` directly rather
than through `apply_edits`, so nothing ever probes it; and the mjlab runner only
ever calls `compute_reward_batched`. So the scalar half of the Tier-D reward is
effectively dead code that would raise if anything exercised it — including any
future replay-based scorer. **Not fixed here on purpose**: a ~6.5 h `sculpt refs
track` certification had `track.py` loaded. Fix it together with the emitter
dedup in #2 above.

---

## 13. Progress update 2026-07-27 — the authoring loop closes

All three modes of `novel-running-jump-kick--g1` are now authored by a model
through `sculpt modes author`, and the finished module clears every gate. This
was §12's #1 and it is closed. Four things had to be fixed to get there, each
found by running the thing rather than by reading it.

### 1. The twin grew as modes got authored (and blew the output budget)

Authoring is one mode per call and `apply_prompt_edit` regenerates the WHOLE
module, so the first design carried each finished mode's body into the next
call's twin — "so the model sees the real neighbours". By mode 3 that put the
twin back at ~17 KB and **both attempts came back truncated**.

What a model needs from a neighbour is which terms are already paid, not how.
`summarize_authored_modes` replaces a finished neighbour's body with one line
of component names. Twin size across the three calls: 10.9 KB → 11.4 KB →
11.4 KB, flat. The summary never leaves the twin.

### 2. A model asked for one function writes two

The first successful edit called an `_info_b` helper it had defined at module
level; `graft_mode_bodies` copied only the two mode functions, and the module
failed the batched probe with `NameError: name '_info_b' is not defined`.

`_carry_helpers` now transplants module-level definitions that the grafted
bodies read and the base does not define — transitively, so a helper calling a
helper comes along. It is scoped by construction: a name already defined in the
base is never overwritten, so the dispatch, the windows and the other modes
still cannot be touched. The prompt now also says a shared helper is fine and
names the convention, which is how the final run produced `_launch_scalar`,
`_launch_tensor`, `_launch_ramp01`, `_launch_ramp01_batched`.

### 3. Attempt 1 truncated on EVERY run — it was the token ceiling

Every real authoring call failed its first attempt with a truncation-shaped
`SyntaxError` (`'(' was never closed`, `unterminated triple-quoted string`).
`edit.MAX_TOKENS = 16000` shared with adaptive thinking is not enough for
"write a single-leg-takeoff reward" *plus* carrying the module back.

- `_call_llm` now takes an optional `max_tokens`, defaulting to `MAX_TOKENS`.
  Threaded through `apply_edits` and `apply_prompt_edit`. **The shared default
  is unchanged** — its 240 s HTTP timeout is calibrated against it for the
  training-mission path, and that path is not what needed more room.
- `sculpt modes author` passes 32000. The final mode authored on attempt 1
  with no retry.
- `_call_llm` also now checks `stop_reason == "max_tokens"` and raises
  `EditValidationError` saying the response was **cut off, not wrong**. Raised
  inside the repair-retry loop's `try`, so the next attempt is told to be
  concise instead of being handed a baffling parse error.

### 4. New gate: a mode may not read an `info` key the env never publishes

The first authored mode reached for ten info keys through a helper doing
`info.get(key, 0.0)`. All ten happened to be real. Had one not been, that term
would have paid a constant 0.0 for the whole of training while the module
imported, ran, and passed every existing probe — which is the exact shape of a
gameable reward.

`_probe_info_keys` runs the authored mode's two halves against a recording
`info` dict built from the contract and rejects any key not in
`expected_info_keys`. Recorded at runtime, so a key reached through a helper, a
loop or an f-string is caught the same as a literal. This is an ADDITIVE gate;
nothing was relaxed.

### The result

`sculpt modes author` x3 on `novel-running-jump-kick--g1`:

```
approach  0.00–1.25 s   frames [0,150)    from 50002_running_on_spot
launch    1.25–2.50 s   frames [150,300)  from 50002_one_leg_jump
strike    2.50–3.70 s   frames [300,444)  from 0016_kicking1
```

Batched dispatch walked across wall clock, `active_mode_index` and each mode's
component:

```
t (s)          0.20    0.80    1.40    2.00    2.60    3.50
active index   0       0       1       1       2       2
mode_approach  2.197   2.197   0       0       0       0
mode_launch    0       0       2.700   2.700   0       0
mode_strike    0       0       0       0       0.387   1.210
```

Each mode pays only inside its own window; the index steps at 1.25 s and 2.5 s,
which is seam frames 150 and 300 at 120 fps. Totals add the tracking backbone.

What the model wrote for `launch`, unedited: `run_in_speed` ramped on
`base_horizontal_speed`, `single_leg_support` paying most for exactly one foot
in contact and less for flight, `takeoff_rise` on `base_height_delta`,
`trail_leg_swing` as the max over the two legs of (height x swing speed) gated
on that foot being airborne, an `action_rate` penalty, and an alive bonus —
every term multiplied by `(1 - fallen)`. Scalar and batched halves compute the
same quantity through shared helpers. That is a genuinely reasonable
single-leg-takeoff reward and none of it is in the scaffold.

### Verification

- `2539 passed, 1 skipped` (jax) — up from 2526.
- `tests/test_modes_cli.py` 22 tests, including the whole authoring machinery
  with the model call stubbed: twin construction, graft, helper carry,
  re-probe, the info-key gate firing and NOT false-positiving on the backbone,
  and twin size staying flat across all three modes.
- `tests/test_edit.py` +3 for the token ceiling, asserting `MAX_TOKENS` is
  still 16000.
- Both gates re-run against the real `v3.py`: clean for all three modes.

### Still open

The `sculpt modes author` output is a reward module, not a trained policy — it
has not been through a training run yet. That is the natural next step and it
needs the GPU, which recert5 still has.

---

## 14. Progress update 2026-07-27 — per-mode authoring is reachable from the UI

`sculpt modes author` worked (§13) but only from a terminal. It now runs from
the Rewards tab, and — the part that matters more — the CLI and the UI run the
SAME implementation.

### One implementation, not two

`mode_rewards.author_mode` holds the whole sequence: stale-scaffold check,
prompt, stub twin, neighbour summaries, `apply_prompt_edit`, graft, helper
carry, re-validate, contract re-probe, info-key gate, silent-no-op check. The
CLI was rewritten to call it; `cli.py` now decides only where the file goes and
what to print. `probe_reward_module` / `probe_info_keys` moved into the library
with it.

This is not tidying. Duplicating that sequence into the backend would have put
a second copy of the phase-window contract in the tree, which is the exact
shape of the bug that has cost this repo two Tier-D certifications. The CLI's
22 tests pass unchanged across the move — that is what makes it a refactor.

### What was added

- `mode_author` job kind + `backend/services/mode_jobs.py`, mirroring
  `reward_jobs.py`'s `asyncio.to_thread` shape. 900 s ceiling.
- `POST /projects/{slug}/references/{clip_id}/mode-reward/author` → 202 + job.
  Refuses, each before any model call: unknown mode (naming the ones that
  exist), missing scaffold, in-place write, a filename that escapes the
  project, a second concurrent authoring job, a live sculpt run, and a missing
  `ANTHROPIC_API_KEY`.
- Authoring always chains to a NEW file (`mode_reward_v0` → `v1` → …), so the
  scaffold survives a rejected edit and the caller chains by passing the
  previous filename back.
- `ModeRewardPanel` on the Rewards tab: scaffold, then one row per mode with
  its window, a per-mode goal field, and an Author button. Shown when the
  reward's `composition.reference_clip_id` is set. A non-composite clip gets an
  explanatory banner rather than an error — one mode with nothing to transition
  to is a real answer.

### Verified live, not just in tests

Against the running backend, through the HTTP API, all three modes of
`novel-running-jump-kick--g1`:

```
scaffold  -> mode_reward_v0.py   0/3
launch    -> mode_reward_v1.py   1/3   3m42s
approach  -> mode_reward_v2.py   2/3
strike    -> mode_reward_v3.py   3/3
```

`mode_reward_v3.py` in the real project passes the contract probe and the
info-key gate for all three modes. Walked across wall clock with an upright
`projected_gravity_b`, a 2.5 m/s run-in and a rise through takeoff:

```
t (s)          0.20    0.80    1.40    2.00    2.60    3.50
active index   0       0       1       1       2       2
mode_approach  1.480   1.164   0       0       0       0
mode_launch    0       0       2.150   2.450   0       0
mode_strike    0       0       0       0       0.650   0.950
```

Note this is a DIFFERENT authoring of `launch` than §13's, and a better one: it
gates on `projected_gravity_b` for uprightness and projects `base_lin_vel_b`
onto the up axis for takeoff velocity, where the CLI sample used only scalar
info channels. Worth knowing when reading either — two samples of the same
prompt produce different, both-valid rewards, which is the point of keeping the
gating out of the model's hands.

**A trap for whoever measures these next.** A first pass at the walk above fed
`torch.randn * 0.05` as the state and `mode_launch` read 0.000 across its whole
window — which looks exactly like a dead term. It was not: random
`projected_gravity_b` normalizes to a random direction, the uprightness gate
reads ~0, and everything downstream of it is multiplied by zero. Zero-or-noise
state is fine for a shape probe and useless for a value probe. Feed physical
gravity.

### Verification

- `2539 passed, 1 skipped` (library) — unchanged across the refactor.
- `604 passed` (backend), `test_references.py` 42 → 50.
- Frontend typechecks and builds.

### Still open

Same as §13: this is a reward module, not a trained policy. Training with it
needs the GPU, which recert5 has (round 2 of 3 as of this writing).

---

## §15 — training with the authored per-mode reward, and the bug it exposed

§13/§14 ended with "this is a reward module, not a trained policy." This closes
that: a policy has now trained on the authored per-mode reward, driven entirely
from the UI. Getting there surfaced a physics bug that had been silently
corrupting **every run on an authored world**, not just this one.

### Promotion — the gap that made the feature a no-op

Authoring writes `mode_reward_v*.py`. `reward_store._V_RE` only recognizes
`v<n>.py`, so those files are not in the version chain: pressing **New run**
after authoring all three modes trained `v0.py`'s starter `alive_bonus` and
discarded the work, silently.

`promote_mode_reward` closes it — library, `sculpt modes promote`, a POST route,
and a **Use for training** button. It refuses a module with any unauthored stub
unless `allow_unauthored`, because a module where 2 of 3 modes still `return
0.0` looks like a working reward to every downstream consumer.

Live: `mode_reward_v3.py` → `v1.py`, `rewards/current.py` re-exports it, and the
Rewards tab shows `Reward v1 · SCULPTOR`.

### The bug: constraint buffers sized for the wrong scene

First UI run died 18 learning iterations in with
`ValueError: observation group 'actor' contains NaN`, behind **197,037** lines of
`nefc overflow - please increase njmax to 336` on stderr.

mjlab sizes `njmax`/`nconmax` per task against that task's **own** scene — the
G1 flat velocity config pins `njmax=300` for a bare plane. An authored world
replaces the scene, so the constant stops describing what the robot can touch.
Measured on the 7-element box course at the real `num_envs=1024` config, 120
steps of random actions:

| njmax | nconmax | overflow lines | peak nefc/world | peak GPU |
|------:|--------:|---------------:|----------------:|---------:|
| 300 | 64 | 112,130 | 496 | ~3.5 GiB |
| 768 | 256 | 0 | 625 | 3585 MiB |
| 1536 | 512 | 0 | 532 | 3767 MiB |
| 3072 | 1024 | 0 | 610 | 4289 MiB |

**The default overflows on ~100% of steps.** Not an edge case near the crash —
the physics was wrong from step 0 of every run that ever used an authored world.
Overflow is silent: mjwarp drops constraint rows, prints to stderr, and keeps
stepping with wrong contact forces until observations go NaN.

`_reconcile_constraint_budget` (in `world/compiler.py`) raises the buffers when a
world is applied, on both the train and eval paths. Two things worth keeping:

- **It only ever raises.** A task asking for more knows something about its
  scene that this function does not.
- **`nconmax=None` counts as unset, not as "big enough."** None means "use
  mjwarp's heuristic", and measured on this same scene the heuristic overflows
  **worse** than the pinned default (68,974 vs 9,124 lines). Handing sizing back
  to the simulator is not the fix here — that was the first thing tried.

Overridable via `RS_WORLD_NJMAX` / `RS_WORLD_NCONMAX`; garbage values fall back
to the measured floor.

### Result

Same run, after the fix — clean telemetry, zero overflow, event count 1,586
instead of 20,000+:

```
Learning iteration 28/300      Mean reward: 177.91
Total steps: 712704            Mean episode length: 131.71
Steps per second: 7146
Episode_Reward/sculptor_primary: 54.9698     <- the authored per-mode reward
Episode_Reward/sculptor_survival: 22.8000
Episode_Reward/sculptor_failure: -0.2500
```

Before the fix the same run reached mean reward 33.36 at learning iteration 17
and then NaN'd. 

### A note on what this says about the earlier runs

Every past run on an authored world was training against dropped contacts. Runs
that "just didn't converge" on a box course are now suspect — the physics they
saw was not the physics the world describes. Worth re-reading any conclusion
drawn from a course run before this commit.

### Still open

- The 197k-line stderr flood is fixed at the source, but nothing *diagnoses* an
  overflow if a future world exceeds even the raised floor. A dedupe + one
  `physics_edit_suggested`-style event would turn the next occurrence from a
  wall of text into a finding.
- recert5 finished: the composed clip is still **INFEASIBLE**, tier stays K.
  `mean_joint_err 0.190` and `root_z_rmse 0.014` both pass, but
  `motion_ratio 1.147` — the tracker is *worse than holding the mean pose*
  (static baseline `0.165` rad). Item #9 is a clip/tracker problem, not a
  threshold problem; do not touch `STATIC_BASELINE_RATIO_MAX` to make it pass.

  Measured the clip to find out why it loses to a static pose. It is **not** a
  low-motion clip — the right knee sweeps 1.708 rad (98°) and leg joints average
  0.206 rad of deviation. The problem is the *denominator*:

  ```
  whole-body mean |q - mean_q|   0.1489 rad   (8.5 deg)
    leg joints (12)              0.2058 rad   peak 1.098
    arm joints (14)              0.1178 rad
  6/29 joints move < 0.05 rad on average
  left/right wrist pitch + yaw:  0.0000 rad   <- frozen, all four
  ```

  `mean_joint_err_rad` averages uniformly over all 29 joints, so four
  identically-frozen wrists plus two near-frozen joints deflate the static
  baseline and the tracker's error together. The `beats_static_baseline` test
  ends up decided partly by DoF that carry no task information at all.

  Two separable questions for whoever picks this up, and they want different
  fixes:
  1. *Is the metric measuring the right thing?* A task-weighted or
     motion-weighted joint error (weight by each joint's variance in the
     reference) would stop dead wrists from voting. That is a metric change and
     goes through the `metric_validate` gates like any other — it is **not** a
     threshold relaxation, and must not be done by touching
     `STATIC_BASELINE_RATIO_MAX`.
  2. *Is the tracker actually learning the kick?* Independent of (1), 0.190 rad
     mean error with a 1.098 rad peak excursion says it is smoothing through the
     large leg swings. Look at per-joint error on `right_knee_joint` and
     `right_hip_pitch_joint` specifically before concluding anything from the
     whole-body average.

  The frozen wrists are worth a look on their own — four DoF at exactly 0.0000
  across all 444 frames is retargeting output, not human motion.

---

## §16 — what the first trained policy actually learned (and it isn't the kick)

The run in §15 completed: 300 learning iterations, mean reward 177 → **3239**,
`reward_spec.json` in `iter_1/` records `version: v1` with all three mode
windows, so the policy demonstrably trained on the authored per-mode reward.

It also learned to game it. `reward_trajectory.json`, first → last recorded
sample:

```
mode_strike                  0.8064 -> 1.2979      strike.landing_absorption  0.4322 -> 0.6900
mode_approach                0.0132 -> 0.0133      strike.landing_stability   0.3071 -> 0.5128
mode_launch                  0.0037 -> 0.0040      joint_tracking             0.2700 -> 0.0343
active_mode_index            1.4501 -> 1.8204      __episode_length          378    -> 535
```

Share of per-mode reward mass actually paid: **strike 98.6%, approach 1.0%,
launch 0.4%.** Meanwhile `joint_tracking` — the term that makes the policy
resemble the reference at all — fell by 8x.

### Why

Two effects compound, and neither is a bug in the generated code:

1. **The terminal mode absorbs the tail.** The dispatch clamps time past the
   last window into the last mode, matching `sculptor.modes.mode_at_frame`. The
   windows are authored against a **3.7 s** clip; the mjlab G1 episode ran
   **9.5–10.7 s** (cap 20 s). So `strike` owns 74–88% of every episode, not the
   32% its authored window describes.

   The model predicts `active_mode_index` 1.61 (at 477 steps) to 1.81 (at 1000);
   observed mean was **1.75**. The clock is not drifting — this is the dispatch
   working exactly as written.

2. **Strike's terms pay ~15x more per step** than approach's, on top of owning
   6x more time.

Net: the cheapest policy is to survive a long episode standing stably and farm
`landing_absorption` + `landing_stability` forever. Reward went up 18x. The
robot is not doing a running jump kick.

### The rollout confirms it visually and numerically

`iter_1/rollout/keyframes/frame_04.png` and `frame_11.png` are, to the eye, the
same image: same crouch, same arm position, same stance. Measured over the
500-step (10 s) rollout, 64 envs:

```
mean_episode_length            500.0 steps   <- all 6 episodes hit the cap, zero terminations
net horizontal displacement    0.297 m       -> 0.030 m/s   (the reference is a RUN)
total path length              0.895 m
root height                    0.798 -> 0.610 m, range 0.190 m over the whole episode
joint |dev| from own mean      0.0384 rad (2.20 deg), last 8 s
  reference clip, same measure 0.1489 rad (8.53 deg)   -> policy moves 4x LESS than the clip
double support                 75.9%
```

A running jump kick that travels 30 cm in ten seconds at 3 cm/s, never leaves
double support for long, and holds its joints four times stiller than the
reference is not a running jump kick. The policy crouched once and held.

### This is the reward-hacking the system exists to surface

Worth being precise about what did and didn't work. The authoring loop, the
graft, the info-key gate, promotion, and training all did their jobs — an LLM
wrote three mode rewards, they were scoped correctly, and a policy optimized
them. The failure is at the level *above* the per-mode terms: **nothing ties the
episode to the clip**, so the automaton's proportions are silently rescaled by
whatever episode length the base task happens to use.

### Candidate fixes, in the order I would try them

1. **End the episode when the automaton ends.** A reference-tracking episode
   that runs 3x the clip is not tracking the clip for 2/3 of its life. This is
   the honest fix and it makes every mode's authored window mean what it says.
2. **Do not clamp — pay nothing past the terminal window.** Cheaper, but leaves
   a long unrewarded tail the policy will still optimize against via the
   survival bonus.
3. **Normalize each mode's per-step pay by its window duration** so total mass
   per mode matches the authored proportion. Fixes the magnitude half only; the
   time-share half stays broken.

Do **not** "fix" this by shrinking strike's weights by hand. The imbalance is
structural — it would come back the moment the episode length changed.

### Do not read the reward curve as progress

Mean reward 177 → 3239 looks like a training success and is the opposite. Any
future comparison across per-mode reward versions has to control for episode
length, or it is comparing how long the robot stood up.

---

## §17 — the sculpt loop could train the per-mode reward but not evolve it

The iteration in §16 finished all four stages. The last one failed:

```
[sculpt] iter 1: apply_edits skipped — EditValidationError: response was cut off
at the 16000-token ceiling — the module is incomplete, not wrong.
```

The run completed and kept `v1`, so nothing was corrupted — but the whole point
of the loop is that stage 4 writes `v2`. **A per-mode reward could be authored,
promoted, and trained, and then the loop could not iterate on it.**

### The diagnosis was right — only the write failed

Worth being clear about what worked, because it is the strongest evidence so far
that the loop functions. Stage 3 found the §16 reward hack **on its own**, from
the same `reward_trajectory.json` I read by hand, at confidence 0.76:

```
failure_modes: reward_hacking, static_equilibrium, sparse_reward, reward_saturation
```

Its proposed edits, quoted:

> `mode_strike` is the dominant component (0.81 → 1.30) versus every launch term
> at ≤0.01 — a >100x imbalance that makes 'sit in the strike window and stay
> alive' the highest-value policy: terminated stays 0.00, episode_length climbs
> 378 → 535 (full horizon), and joint_tracking collapses 0.27 → 0.03 while
> return rises.

> `joint_tracking` degraded monotonically 0.27 → 0.03 while total return ROSE,
> proving the posture/landing channel can be harvested with the reference motion
> fully abandoned. Following PhysHOI's task-agnostic imitation reward, which
> MULTIPLIES kinematic rewards together so none of them may be small, gate the
> landing/posture credit …

It also found something my hand analysis missed. Several mode terms are not
merely outweighed — they are **identically dead**:

```
approach.run_speed              0.00 across all six windows
launch.takeoff_rise             0.00
launch.vertical_launch          0.00
launch.single_support           0.00
launch.trailing_leg_drive       0.00
strike.apex_leg_swing           0.00   <- the actual kick
strike.flight_foot_clearance    0.00
```

These are gated behind thresholds (speed, flight detection, apex height) that a
frozen policy never reaches, so they pay nothing and produce no gradient toward
ever reaching them. That is a cold-start problem in the authored terms, not a
weighting problem, and it needs a different fix from §16's: dense,
threshold-free terms that pay partial credit from the current state. §16's
episode-length fix and this one are both required — neither alone is enough.

So the pipeline diagnosed its own generated reward correctly and grounded the
remedy in cited literature. The only broken link was the token budget for
writing the result down.

### Why

`apply_edits` does a whole-module rewrite: the model emits the complete new
`reward.py`. `MAX_TOKENS = 16000` is a comfortable ceiling for a hand-written
reward and a wall for a generated one. The live module:

```
v0.py (starter alive_bonus)      57 lines    2.0 KB     ~0.5k tokens
v1.py (3-mode automaton)      1,038 lines   49.4 KB    ~12.4k tokens
  TARGET_JOINT_POS             8.3 KB   16.9%
  TARGET_* tables total        9.9 KB   20.0%   <- ~2.5k tokens of pure DATA
```

A fifth of the module is inlined reference tables (32x29 joint targets, root
heights, gravity vectors). The editor has to restate all of it verbatim to
change one weight, and 12.4k of the 16k budget is gone before it writes
anything new.

### Fix

`_rewrite_token_ceiling(source)` sizes the ceiling from the module being
rewritten — `max(MAX_TOKENS, len(source)/3.5 * 1.6)`. It is a **floor, not a
replacement**: every existing hand-written and gym reward keeps its calibrated
16K budget byte-for-byte (the 240s HTTP timeout is tuned against it). The live
`v1.py` gets 22,541.

Note the earlier `stop_reason == "max_tokens"` detection is what made this
diagnosable at all — without it this surfaces as
`SyntaxError: '(' was never closed`, which reads as a model failure rather than
a budget one, and sends you looking in the wrong place.

**The ceiling alone was not the fix, and only replaying the real call showed
that.** Raising `max_tokens` and leaving the HTTP timeout at its 240s wall just
relocates the failure: the first replay of the per-mode edit at a 22,541 ceiling
died on `anthropic.APITimeoutError` instead. The 240s figure was calibrated
against `MAX_TOKENS`, so the two have to move as a pair —
`_rewrite_http_timeout_s` scales the wall with the ceiling, floored at 240s so
every existing call site keeps its tuned budget exactly.

That cascaded once more. Authoring runs at `AUTHOR_MAX_TOKENS = 32000`, so one
attempt is ~480s and `apply_prompt_edit`'s two attempts are ~960s — past the
flat 900s `DEFAULT_MODE_AUTHOR_TIMEOUT_S`, which would have killed a legitimate
retry. That budget is now *derived* from `_rewrite_http_timeout_s(
AUTHOR_MAX_TOKENS)` (1440s) rather than hardcoded, so raising either ceiling
cannot silently outgrow it again.

Three coupled limits, one of which was only visible by making the real call.
Worth remembering that the unit tests for the ceiling all passed while the
end-to-end path was still broken.

**And the ceiling itself was still wrong, because it was estimated.** The second
replay got past the timeout and truncated anyway, twice. Counting with
`messages.count_tokens` instead of guessing:

```
v1.py whole module     49,310 B -> 20,173 tok   2.44 B/tok
  TARGET_JOINT_POS      8,326 B ->  5,540 tok   1.50 B/tok   <- 27% of the module
  TARGET_GRAVITY        1,194 B ->    675 tok   1.77 B/tok
```

The 3.5 B/tok figure is right for prose-like Python and 30% optimistic here,
because dense float literals tokenize about half as well as code. The "raised"
ceiling of 22,541 was therefore only **1.12x** the module — and adaptive
thinking is charged against the same budget, so there was never room. Now 2.2
B/tok with 2.0x headroom: 44,827, or 2.22x the module, hard-capped at 64,000
(`claude-opus-5` was probed and accepts at least 96,000).

**This is the argument for the sidecar (#17), not a footnote to it.** One float
table is 5,540 tokens — 27% of the module — that the editor must retype exactly
on every single rewrite, and nothing checks it came back unchanged. The ceiling
fix makes the loop work; moving the tables out is what makes it sane.

### The better fix, not taken here

Raising the ceiling treats the symptom. The real problem is that **~2.5k tokens
of the module are data the LLM has no business rewriting** and cannot improve.
Moving `TARGET_JOINT_POS` / `TARGET_ROOT_Z` / `TARGET_GRAVITY` into a sidecar
loaded at import would cut a fifth of the module and remove a whole class of
transcription risk — every rewrite currently re-types 928 floats and nothing
checks they came back unchanged.

That is a larger change: the scaffold generator, the validator, promotion, and
`current.py`'s by-path loader all have to agree on where the sidecar lives and
how it travels with a promoted version. `reward_store._extract_reward_spec` is
AST-only and never executes the module, so it is unaffected — but a promoted
`v<n>.py` that references a sidecar is no longer a single self-contained file,
and that is the design question to settle first.

Filed rather than done, because the ceiling fix unblocks the loop now and the
sidecar change deserves its own review.

### Verified end to end

Third replay of the same call, with all three limits corrected:

```
old ceiling: 16000   new ceiling: 44827
WROTE .../v2.py  (61,727 bytes, 1,235 lines)
  TARGET_JOINT_POS     IDENTICAL
  TARGET_ROOT_Z        IDENTICAL
  TARGET_GRAVITY       IDENTICAL
  _mode_approach / _mode_launch / _mode_strike        present
  compute_reward / compute_reward_batched / _MODE_FNS present
  MODE_WINDOWS_S                                      present
```

The loop can now evolve a per-mode reward. `v2` is a real edit, not a
reformatting: `hyperparameters` goes from `{}` to a named set,
`references` cites Bjelonic et al. (2025), a `grounding` map ties each
hyperparameter to a paper, and the dead gated terms are replaced with floored
variants (`launch_gate_floor`, `launch_support_floor`, `air_floor`,
`land_vz_on_ref`) — which is what the diagnosis asked for. `mode_windows_s` is
unchanged: the editor left the dispatch alone, as the scaffold instructs.

The 928 floats came back byte-identical on this rewrite. That is one sample and
nothing in the production path checks it — the replay harness did. #17 stands.

**Method note.** This bug was "fixed" three times and only the third was real:
raise the ceiling (unit tests green, end-to-end died on APITimeoutError) → scale
the timeout (unit tests green, truncated twice more) → actually count the tokens.
Every round the tests passed. The only thing that found the next layer was making
the real call — the same lesson as the `njmax` bug in §15, where the build was
green while the physics was wrong on 100% of steps.
