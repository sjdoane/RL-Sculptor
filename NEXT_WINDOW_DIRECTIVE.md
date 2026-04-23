# Reward Sculptor — next-window directive (2026-04-23)

Read top-to-bottom before touching code. Sam has tested the tool on a
biped cartwheel task and hit four distinct failure modes (listed below).
He also asked for a specific Eureka-paper-inspired upgrade to the
diagnose step. Your job is to land the P0/P1 items end-to-end while
keeping test baselines green.

**Do NOT ship implementation before Sam greenlights the plan.** Structure
matches `NEW_WINDOW_DIRECTIVE.md` from 2026-04-22: read, plan, confirm,
then ship test-by-test.

---

## 1. Who / what

Same as `NEW_WINDOW_DIRECTIVE.md` — USC ME undergrad, AME456 capstone,
RTX 5070 Laptop 8 GiB VRAM, WSL2 Ubuntu, Claude-backed autonomous RL
reward iterator. Memory file has the full style sheet — terse, file:line
over prose, confirm destructive ops, no emojis. **Shell is Git-Bash,
not WSL bash — use `wsl bash -lc '...'` heredoc for anything with
`$variables` or globs.**

## 2. Stack

```
~/projects/
├── AME456/                     # quadruped capstone (read-only)
├── RewardSculptor/             # sculptor pkg
│   ├── sculptor/
│   │   ├── adapters/           # gym_sb3 + mjlab + stubs
│   │   │   ├── mjlab.py
│   │   │   └── _mjlab_runner.py  # subprocess CLI — TRAJECTORY CAPTURE LIVES HERE
│   │   ├── diagnose.py         # failure-mode analysis; PROMPT LIVES HERE
│   │   ├── edit.py             # reward rewrite + validation
│   │   ├── kg/                 # SculptorKG + query + research + extract
│   │   ├── prompts/            # edit_rewriter.md, physics_editor.md, diagnose.md
│   │   └── sculpt.py           # outer loop
│   └── tests/                  # pytest
└── reward-sculptor-ui/         # FastAPI + React
    ├── backend/
    └── frontend/
```

Running: `cd ~/projects/reward-sculptor-ui && ./run.sh`.
Shared KG: `~/.local/share/sculptor/kg/graph.db` (71 papers, 416
techniques as of 2026-04-22).

## 3. Reference documents (read in order)

1. **[CONTEXT.md](CONTEXT.md)** — session log, newest-first. The last 4-5
   entries cover the S1..S8 pass + the KG-visibility fixes + this
   cartwheel-test session.
2. **[QUALITY_PASS_PLAN.md](QUALITY_PASS_PLAN.md)** — has the T11
   overnight acceptance checklist.
3. **[MJLAB_PIVOT_DESIGN.md](MJLAB_PIVOT_DESIGN.md)** — §1.3 reward
   injection + §1.4 state snapshot.
4. **This file** — the directive you're reading.
5. The Eureka paper at [arxiv:2310.12931](https://arxiv.org/abs/2310.12931)
   if you need to re-read. Full text saved at
   `~/projects/_eureka_fulltext.txt`. Most relevant sections are the
   reward-reflection template (§3.3, Appendix A Prompt 2) and the
   per-component statistics format (Appendix F.1).

## 4. Baseline test counts (must not regress)

| suite | command | expected |
|---|---|---|
| sculptor | `cd ~/projects/RewardSculptor && uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py` | 109 passed, 1 skipped |
| backend | `cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q -k "not vram and not pynvml and not gpu"` | 215 passed, 3 skipped, 6 deselected, 1 known CUDA-state flake (see NEW_WINDOW_DIRECTIVE §4) |
| frontend typecheck | `cd ~/projects/reward-sculptor-ui/frontend && ./node_modules/.bin/tsc --noEmit` (with `PATH=$HOME/.local/share/pnpm:$PATH`) | exit 0 |

## 5. What works today

All S1..S8 shipped 2026-04-22 plus the round-2/round-3/round-4 followups:
- Cartpole and Go1 mjlab training loops work end-to-end through the UI.
- KG-grounded Claude reward edits work with visible citations + grounding dict.
- Physics tab prompt-edit commits MJCF changes, triggers preview cache invalidate.
- Runs-tab timeline is monotonic; autoscroll is pinned.
- Live KG probe: 71 papers, 416 techniques, 307 INTRODUCES edges, 269 embeddings.
- Motor-limits template button on Physics tab.

## 6. Current state — what's broken (Sam's cartwheel test, 2026-04-22)

Reference these numbers in your plan.

### 6.1 `training_iterations=500` override ignored — logs show /1500

Sam set `rsl_rl iters/cycle = 500` in NewRunDialog → Advanced. mjlab
subprocess logged `1m Learning iteration xx/1500`. This was supposedly
fixed in S-round3 via the `--steps-per-iter` CLI flag (see
[RewardSculptor/sculptor/cli.py:213-225](RewardSculptor/sculptor/cli.py)
+ [backend/services/run_manager.py:92-102](reward-sculptor-ui/backend/services/run_manager.py)).

**First step before fixing: reproduce with a clean restart**. Sam may have
been running uvicorn against stale code (--reload doesn't always pick up
new CLI flags + new kwargs on sculpt_run). Sequence:
1. `pkill -KILL -f uvicorn` + `pkill -KILL -f run.sh`
2. Start fresh: `./run.sh`
3. Launch a 2-iter mini-run with training_iterations=100
4. Grep the sculpt log for "Learning iteration xx/100" vs "xx/1500"

If it shows /100, the bug was a stale reload — just document. If /1500,
trace: (a) is `--steps-per-iter 100` in the `ps -ef | grep sculptor.cli`
output? (b) does `sculpt_run(steps_per_iter=100)` actually override
cfg["iteration"]["steps_per_iter"]? (c) does `_run_one_iter` read the
override? Add a `[sculpt]` print at each step.

### 6.2 Project tab breakage when another project's run is active

Sam started a G1 cartwheel run. Clicked back to his `swingup-cartpole`
project (different project, completed prior runs on disk). Symptoms:
- static preview endpoint returned 500
- Rewards + Physics tabs "barely updated"
- Reports tab said "no report" even though `final_report.md` exists on disk
- History showed "no runs completed"

Hypothesis: frontend React Query keys scoped too coarsely. When the
active run updates `activeJobsIndicator`, its invalidation may blow away
OTHER projects' cached queries. Or the backend's ProjectStore has a
single in-memory lock that blocks reads on non-running projects.

**Diagnosis path**:
1. Reproduce: launch a run on project A, navigate to project B while A
   is running. Capture the exact endpoints returning errors in uvicorn
   logs.
2. Check `frontend/src/hooks/useProjects.ts` + `useRobot.ts` query keys
   — are they project-slug-scoped correctly?
3. Check `backend/services/project_store.py` for any global lock.
4. Check `backend/routes/reports.py` — does its "report exists" check
   read from the right project_dir?

### 6.3 Stubbed paper titles in KG — show up as "Unknown" in reward refs

Diagnosed this session. arxiv's API rate-limits aggressively on bulk
ingest. `sculptor/kg/ingest.py:_fetch_arxiv_metadata` returns None on
rate-limit, code falls through to `fallback_metadata` (the seed YAML's
`title` field), which was absent in my `cartwheel_kg_seeds.yml`. Stub
title `arxiv:XXXX.XXXXX` propagates.

Sam's cartwheel reward v1 cited `2406.08858` and `2409.16611` — both
have stub titles. Reward spec references column renders them as
"Unknown. arxiv:2406.08858". Cosmetic but embarrassing.

**Fix (three-part)**:
1. Healing pass: scan shared KG for Paper nodes where
   `title.startswith("arxiv:")`, re-call `ingest_arxiv(id, force=True)` on
   each. One-time script, runs in ~2 min.
2. Prevention: bump `_fetch_arxiv_metadata` retry count from 1 → 4 with
   exponential backoff (10s, 30s, 60s, 120s). Arxiv's rate-limit window
   is ~2 min; will recover.
3. Defense in depth: update `cartwheel_kg_seeds.yml` + the default
   seeds file to include `title:` per paper. Agent 1's deliverable from
   this session has the titles — reference them.

### 6.4 Physics is physically unrealistic for hard tasks (cartwheel case)

**This is the big one.** Sam's cartwheel reward produced a policy that
(a) never left standing, (b) thrashed all joints at maximum velocity,
(c) exploited action clip limits. Training completed 4/8 iters with no
meaningful progress. Diagnose classified it as reward-hacking; Claude's
reward edits addressed the REWARD side but couldn't fix the fact that
the underlying MJCF permits a motor to go from 0 to max torque every
timestep — unrealistic, not representative of real hardware.

**Required feature**: **physics-realism audit between iterations**, with
automatic MJCF adjustment when violations exceed threshold.

Two-layer fix:
- **Audit layer (P0)**: after each rollout, read `trajectory.npz` +
  project's MJCF. Compute torque-saturation %, joint-velocity tail
  heaviness, joint-limit violation rate. Write to
  `runs/iter_<N>/realism_audit.json`. Surface in UI + feed into the
  next iteration's diagnose prompt.
- **Auto-adjust layer (P1)**: if audit severity is high (e.g.
  >30% torque saturation across >50% of steps), kick off a
  KG-grounded physics-prompt-edit to tighten forcerange / add damping
  / reduce control frequency. Existing `apply_prompt_edit` in
  `physics.py` handles this; just need to synthesize the prompt +
  wire into the sculpt loop.

**This REQUIRES expanding trajectory capture** (see §7.1 below).

## 7. Plan — P0 → P3 ship order

### §7.1 P0 — Expand rollout trajectory capture (prereq for everything)

Current `_mjlab_runner.py::_cmd_rollout` saves only `rewards` +
`episode_id` to `trajectory.npz`. Expand to include:
- `joint_pos`: (T, num_envs, num_dofs)
- `joint_vel`: (T, num_envs, num_dofs)
- `action`: (T, num_envs, num_actuators)
- `actuator_force`: (T, num_envs, num_actuators) if available on entity
- `reward_components`: dict of (T, num_envs) arrays — one per term
- Optional: `root_link_pos_w`, `projected_gravity_b` (already read for
  fall detection)

Memory: at T=500, num_envs=64, num_dofs=29 (G1), this is 500×64×29×4 bytes ≈
3.7 MB per field. Negligible.

**Sculptor-side changes**:
- `_mjlab_runner.py::_cmd_rollout` — append to buffer lists at every
  step, `np.savez_compressed` at end.
- During training too: expose an optional checkpoint-level save for the
  reward-reflection feature (see §7.2). Don't save EVERY training step
  (too much data); save at the same cadence rsl_rl writes its checkpoint
  (~every 50 iters by default).

**Tests**: extend the GPU-gated `test_mjlab_cartpole_short_train_smoke`
to assert the new fields exist + shapes are right.

### §7.2 P0 — Eureka reward-reflection block in diagnose prompt

Implements the one Eureka idea worth porting. Format (verbatim from
Eureka Appendix F, lines 1482-1487 of `_eureka_fulltext.txt`):

```
rotation_reward: ['0.03', '0.31', '0.30', '0.32', ..., '0.32'], Max: 0.36, Mean: 0.32, Min: 0.03
success_rate:    ['0.00', '0.83', '1.85', '2.89', ..., '8.83'], Max: 9.29, Mean: 4.81, Min: 0.00
episode_lengths: ['7.07', '384.30', '378.22', ..., '434.24'], Max: 482.35, Mean: 396.02, Min: 7.07
```

**Where to capture**:
- During training, at each `save_interval` checkpoint (~every 50 rsl_rl
  iters), run a tiny eval pass: step the env once with the current
  policy, record each reward component's value. Accumulate a list of
  ~10 snapshots per training run.
- Save to `runs/iter_<N>/reward_trajectory.json` alongside other iter
  artifacts.
- Schema: `{"component_name": [v0, v1, ..., vN], "success_rate": [...], "episode_length": [...]}`

**Where to inject**:
- `sculptor/diagnose.py::_build_user_prompt` — add a `<training_feedback>`
  XML block right after REWARD_SPEC + before the failure-mode ask.
  Format EXACTLY as Eureka does: per-component values as list literals
  plus Max/Mean/Min.
- Also inject into `sculptor/edit.py::_build_user_prompt` so the
  reward-rewrite step sees the same diagnostic data (Eureka does this).

**Dead-component rule**: add a paragraph to the edit_rewriter.md prompt:
> If `<training_feedback>` shows a component's values are near-identical
> across training (max-min < 5% of max), RL can't optimize it. Choose
> one: (a) change its scale or temperature parameter, (b) rewrite the
> component, (c) discard it. Preserve the term only if you have a
> specific physics reason. This catches dead/unoptimizable components
> before they survive through iterations.

**Tests**:
- Unit test that `reward_trajectory.json` file gets created with the
  right schema.
- Unit test that diagnose prompt contains the `<training_feedback>`
  block with parsable data.

### §7.3 P0 — Physics realism audit + surfacing

After rollout, run an audit pass. New module
`sculptor/adapters/realism.py`:

```python
def audit_rollout(trajectory_path, mjcf_path) -> dict:
    # torque_saturation_frac: fraction of steps where |actuator_force| > 0.95 * forcerange
    # joint_vel_p99: 99th percentile of |joint_vel|, per joint
    # joint_limit_violation_frac: fraction where joint_pos outside soft-range
    # verdict: "ok" | "mild" | "severe"
```

Save to `iter_<N>/realism_audit.json`. Surface in:
- Backend: new field on iter detail response.
- Frontend: new collapsible card in the iter timeline entry.
- Diagnose prompt: if severity ≥ "mild", prepend a `<physics_realism>`
  block like "torque saturation at 47% on knee_pitch — reward shape
  may be exploiting unrealistic actuator response".

**Tests**:
- Unit: call audit on a synthetic trajectory with known saturation,
  assert frac matches.
- Integration (GPU-gated): audit after a real rollout.

### §7.4 P1 — Auto-adjust physics between iterations on severe violations

When `realism_audit.verdict == "severe"`, insert a physics-edit step
BEFORE the next reward edit. Existing `apply_prompt_edit` in
`backend/services/physics.py` handles the mechanics; just need to
synthesize the prompt automatically. Template:

> The previous iteration's policy exploited physically unrealistic
> actuator behavior: {audit summary}. Adjust the MJCF to tighten this:
> {specific joints that saturated}. Prefer: reduce forcerange on
> saturated joints, increase joint armature to match motor inertia,
> increase joint damping to model viscous friction. Cite the relevant
> KG papers (likely 2312.17507 Actuator-Constrained RL, 1901.08652
> ANYmal actuator-net, 2410.08650 Extended Friction).

Feature-flagged behind `[iteration].auto_adjust_physics = true` in
config.toml. Off by default; Sam enables per-project.

### §7.5 P1 — `training_iterations` override regression

Reproduce, then fix per §6.1.

### §7.6 P1 — Project-tab breakage during active run

Diagnose per §6.2. Likely-candidate fixes ranked:
1. Frontend React Query: ensure `qk.project(slug)` + `qk.robot(slug)` +
   `qk.rewards(slug)` never overlap with `["runs"]` keys from another
   project.
2. Backend: audit `routes/reports.py` + `routes/robot.py::get_preview`
   for any per-project path that might take a slug-global lock.
3. Add an integration test: launch project A run, issue GET requests
   for project B's preview/rewards/reports/runs, assert all return 200.

### §7.7 P1 — Stub-title fix (three parts per §6.3)

Ship the healing pass first (closes the immediate cosmetic issue), then
the retry-with-backoff + seeds-with-titles for prevention.

### §7.8 P2 — Report graphs

`sculptor/timelapse.py::_write_final_report_md` — add matplotlib
renderer producing 4 plots:
1. **Metric history**: line chart of primary_metric across iters, with
   marker at each edit commit.
2. **Reward component stack**: stacked area of each term's mean per
   iter, from the trajectory files (§7.1).
3. **Termination cause breakdown**: stacked area of timeout / fall /
   joint-limit exceeded, if recordable.
4. **Per-env return distribution**: violin plot per iter from
   rollout's per-env returns.

Each plot saved as PNG alongside `final_report.md`, referenced inline.
Cite the papers that motivate each metric (1707.06347 PPO for return,
2109.11978 Rudin for per-env variance, 1901.08652 ANYmal for fall
rate). These are already in the KG.

Adds matplotlib dep to `RewardSculptor/pyproject.toml`.

### §7.9 P2 — Motor-datasheet PDF upload

New backend route: `POST /projects/{slug}/physics/datasheet-pdf` —
accepts multipart PDF. Backend extracts text via `pypdf`, passes to
Claude with a schema like `{joint_name: str, peak_torque_nm: float,
peak_speed_rads: float, gear_ratio: float, rotor_inertia_kgm2: float}`,
then feeds result into the existing `apply_prompt_edit` flow. Returns
normally via the physics-edit commit.

Frontend: new button next to "Motor limits template" on Physics tab —
opens a file picker, uploads the PDF, shows a preview of extracted
params before committing.

### §7.10 P3 — React Query cache invalidation on physics commit

Already fixed the backend file-unlink (reward_jobs.py:228). Frontend
needs to invalidate `qk.physics(slug)` + `qk.preview(slug, angle)` on
physics-edit success so the Overview preview refreshes without a manual
reload. ~10 LoC in `usePhysics.ts`.

## 8. Known gotchas (read before debugging)

Same list as NEW_WINDOW_DIRECTIVE.md §8, plus:

- **sculptor.edit now writes staging files** (`.v<N>.staging.py`) before
  validating + renaming. Tests that assert target file existence need
  to use the actual target, not staging. Staging files are leading-dot
  hidden so `list_versions` glob `v*.py` ignores them.
- **`_build_dummy_inputs` returns torch CPU tensors for schema-style
  contracts**, NOT numpy. Gym-style contracts still get numpy.
- **`_PROBE_SCRIPT` imports torch in the subprocess**. Safe because each
  probe runs in a fresh Python process.
- **arxiv API rate-limits at ~3-5 concurrent requests per minute**.
  Aggressive bulk-ingest will stub-title many papers (see §6.3).
- **`sculptor/kg/query.query_semantic` has `min_similarity=0.0` default
  BUT prompt-edit callers pass 0.35**. Don't change the default.
- **The monotonic iter reducer in RunsTab.tsx uses a useRef Map**;
  iterations never vanish from the timeline once seen.

## 9. Priority matrix

| P | Item | Tier | Complexity | Depends on |
|---|---|---|---|---|
| P0 | §7.1 Expand trajectory capture | core prereq | M | — |
| P0 | §7.2 Eureka reflection block | feature | M | §7.1 |
| P0 | §7.3 Physics realism audit | feature | M | §7.1 |
| P1 | §7.4 Auto-adjust physics | feature | M | §7.3 |
| P1 | §7.5 training_iterations regression | bug | S-M | — |
| P1 | §7.6 Project-tab breakage | bug | M | — |
| P1 | §7.7 Stub-title heal + retry | bug | S | — |
| P2 | §7.8 Report graphs | feature | M-L | §7.1 + matplotlib |
| P2 | §7.9 Datasheet PDF upload | feature | L | — |
| P3 | §7.10 React Query cache invalidation | polish | S | — |

Ask Sam to prioritize within P0 before shipping anything; §7.1 is a
real prereq that §7.2 + §7.3 can't ship without, but §7.4 can wait.

## 10. Ship discipline (per NEW_WINDOW_DIRECTIVE pattern)

After EACH ship:
1. Re-run the three §4 suites. Paste tail into CONTEXT.md.
2. Append Change Log entry to CONTEXT.md with What/Why/How/Verified.
3. Live smoke where applicable.
4. Don't start the next ship until baselines match.

---

## 11. Starter prompt for the new window

Paste this exactly into the new Claude Code window:

```
Read ~/projects/NEXT_WINDOW_DIRECTIVE.md end-to-end. Then:

1. Run the three test suites (§4). Confirm baselines match.
2. Reproduce the `training_iterations=500 → logs show /1500` bug from §6.1
   with a clean restart. Report whether it's a real regression or a
   stale-reload artifact.
3. Generate a plan that addresses §7.1 + §7.2 + §7.3 (the P0 prereq chain
   that unlocks physics realism + Eureka reflection). For each: file paths
   and line ranges, complexity S/M/L, regression risk, tests to add.
4. Do NOT implement anything until I confirm the plan.
5. After I approve, ship test-by-test. Log each in CONTEXT.md. Keep the
   test baselines green after every ship.

Then pause and ask me whether to proceed to P1.
```

---

*End of directive. Index into the new window's prompt.*
