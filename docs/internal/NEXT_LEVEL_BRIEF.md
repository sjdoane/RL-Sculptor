# RL-Sculptor — Next-Level Brief (fresh-session handoff)

You are taking over **RL-Sculptor**, an autonomous "reward engineer" for robot
reinforcement learning. Your mission: **produce a comprehensive, phased plan to
take this project toward research-grade quality, get it approved, then execute
it** — under a strict, quality-assured workflow. Do **NOT** change code before
you have oriented yourself and presented a plan.

---

## 0. Orient FIRST (before touching anything)

Read these in order, then summarize back what the system does + its current
state so we know you understand it:

1. `~/projects/CONTEXT.md` — the canonical changelog. Read the **top ~12 entries**
   (Ships 22a→22s, newest first); they ARE the recent history + design
   rationale. This file is the project's memory.
2. `~/projects/M7_PLAN.md` and `~/projects/MJLAB_PIVOT_DESIGN.md` — architecture
   + the mjlab pivot.
3. Sculptor core: `~/projects/RewardSculptor/sculptor/` — `sculpt.py` (inner
   train→diagnose→edit loop + the mission orchestrator), `mission_runtime.py`
   (success-criterion evaluator + namespace), `decompose.py` (+
   `prompts/decompose_task.md`, `prompts/redecompose_stage.md`), `diagnose.py`,
   `edit.py`, `reward.py`, `adapters/` (`mjlab.py`, `gym_sb3.py`, `base.py`,
   `_mjlab_runner.py`), `kg/`.
4. The app: `~/projects/reward-sculptor-ui/` — `backend/` (FastAPI + in-process
   JobManager + WS streams) and `frontend/` (React 18 + TS + Vite + TanStack
   Query + bespoke "rs-" CSS design system).

Then **establish a green baseline** — run all four gates (§5) — before you
change a line.

---

## 1. What the system is (one paragraph)

Given a **natural-language behavior goal** for a simulated robot, RL-Sculptor
autonomously engineers the reward function + training curriculum to achieve it.
Pipeline: NL goal → an LLM **decomposes** it into an ordered **curriculum** of
learnable sub-skill stages → per stage, an LLM **seeds** a reward function (v0),
then loops **train (PPO) → roll out → diagnose failures (LLM, grounded in a
knowledge graph of ~1,400 RL papers/techniques/failure-modes) → rewrite the
reward (versioned + diffed, contract-checked) → commit** until the stage's
**success_criterion** (a sandboxed Python predicate over rollout metrics) holds.
Stages warm-start from predecessors; a failed stage can be re-decomposed once.
Training runs via **mjlab** (MuJoCo + rsl_rl PPO, GPU) or **gym_sb3** (CPU).
Closest prior art: Eureka, Text2Reward, CurricuLLM — our differentiators are
**KG-grounded reward edits + an automatic curriculum + a closed-loop
diagnose→edit cycle + a full product UI**.

---

## 2. Environment + hard gotchas (these WILL bite you)

- **WSL2 Ubuntu**; repo at `/home/samjd/projects` (Windows UNC
  `\\wsl.localhost\Ubuntu-24.04\home\samjd\projects`).
- **GPU: RTX 5070 Laptop, 8 GiB VRAM** → mjlab auto-caps `num_envs`≈2048; a
  humanoid sculpt-iter is ~25 min. **Compute is THE bottleneck** (see §8).
- The Bash tool's shell is **Windows Git-Bash, NOT WSL**. Wrap commands in
  `wsl bash <<'EOF' … EOF` (quoted heredoc) so `$vars`/quotes aren't mangled by
  the outer shell. `wsl bash -lc '…'` with a `for`-loop silently eats `$var`.
- `uv` is at `/home/samjd/.local/bin/uv` (NOT on the default `bash -lc` PATH —
  prepend it). node/pnpm live under `/home/samjd/.local/share/pnpm`.
  `ANTHROPIC_API_KEY` is in `RewardSculptor/.env`.
- Edit/Write writes files at **0644** → `chmod +x` any `*.sh` after editing.
- Launch the app: `cd reward-sculptor-ui && ./run.sh` (headless-safe as of Ship
  22p; serves UI at `localhost:5173`, backend at `:8000`).

---

## 3. Immediate capabilities (works TODAY)

- **NL goal → adaptive staged curriculum** — stage count now scales with task
  complexity and no longer wastes a stage just standing the robot up (Ship 22s).
- **Per-stage closed loop**: LLM reward seed → PPO train → rollout → LLM
  diagnose (KG-grounded) → reward edit (versioned, diffed, contract-checked) →
  success-criterion eval. The criterion evaluator is a **sandboxed-AST** Python
  predicate over rollout artifacts (`behavior.json`, `trajectory.npz`, and the
  sculptor reward's per-component means).
- **Knowledge graph**: ~1,400 nodes (papers / techniques / failure-modes /
  reward-components / environments). "Add papers" + "Research a topic" ingest +
  extract entities; the graph grounds reward edits + decomposition. Force-
  directed pyvis viewer in the UI.
- **Recovery / robustness**: a criterion referencing a metric the reward didn't
  produce is now recoverable (→ `criterion_not_met`, not a fatal crash); bounded
  re-decomposition feeds the validator error back to the LLM (Ships 22q/22r).
- **Adapters**: `mjlab` (Unitree G1, Go1/Go2, Cartpole, + Menagerie robots),
  `gym_sb3` (HalfCheetah, etc.). `isaac_lab`/`mjx`/`rllib` are scaffolded
  (coming-soon). Robot **library** (MuJoCo Menagerie) + **URDF/MJCF upload**
  with MuJoCo validation.
- **Physics**: MJCF parsing, auto-physics-edit suggestions, realism audits.
- **Full UI**: dashboard, project tabs (overview / rewards / physics / KG / runs
  / reports), live WS event streaming, rollout videos, metric charts.

---

## 4. What's weak / could be improved (be honest about this)

- **Evaluation is NOT research-grade.** Success criteria are LLM-authored
  heuristics; there are **no quantitative behavior metrics vs a spec, no
  baselines** (Eureka / human-designed reward / plain PPO), **no ablations**
  (no-KG / no-curriculum / no-diagnose), **no multi-seed statistics, no
  benchmark task suite**. This is the single biggest gap to "research-grade".
- **Reward/curriculum quality is variable + unmeasured.** The implicit contract
  between reward-component names and the criterion keys is fragile (22r hardened
  it). Decomposition quality just improved (22s) but is not evaluated.
- **KG-grounding effectiveness is unmeasured** — does citing literature actually
  improve reward quality? No A/B exists.
- **Compute-bound**: 8 GiB GPU limits parallelism + speed; runs are long; resume
  exists but there's no remote/cloud training path.
- **Reproducibility gaps**: seed handling, determinism, and full config/version
  logging are not paper-grade.
- **In-process JobManager** (single host, in-memory) — limited cross-restart
  history, no durable job queue.
- **No manipulation / object interaction** — environments are locomotion-only.

---

## 5. The gates — TEST EVERYTHING (exact commands + gotchas)

Keep ALL of these green before and after every change; add tests for new
behavior.

- **Frontend**: `cd reward-sculptor-ui/frontend && pnpm build` (or
  `pnpm typecheck` = `tsc -b`). **NOT `tsc --noEmit`** — that is a no-op
  (solution-style `tsconfig.json`) that always passes and hid 15 real errors.
  Prepend `/home/samjd/.local/share/pnpm:/home/samjd/.local/share/pnpm/nodejs/<ver>/bin`
  to PATH.
- **Backend**: `cd reward-sculptor-ui && uv run pytest backend/tests/ -q -k 'not test_reward_prompt_edit_emits'` → ~305 passed.
- **Sculptor**: `cd RewardSculptor && uv run pytest tests/ -q` → ~369 passed.
  **CWD gotcha**: run from `RewardSculptor/`, NOT `RewardSculptor/sculptor/`
  (wrong dir collects 0 tests and reports "no tests ran" as a false pass).
- **Live smoke**: `./run.sh`, then exercise the affected flow in the UI.

---

## 6. Future capabilities to design toward (the vision)

- **Arbitrary robots**: upload exists (renders the URDF/MJCF); extend it to
  **auto-generate a trainable task** (observation/action/termination spec) for an
  uploaded robot, not just render it.
- **More complex tasks**: multi-phase, long-horizon, periodic + transitional
  skills, longer episodes.
- **Object interaction / manipulation** (the biggest leap): gripping,
  pick-and-place, tool-use, contact-rich tasks. Requires manipulation
  environments + a manipulation adapter (mjlab/Isaac manipulation suites or a
  custom MuJoCo env) + reward/criterion vocabulary for contact/grasp/object-pose.
  Scope carefully.
- **Cross-robot / cross-task skill transfer** (`skill_library` exists — extend +
  evaluate).
- **Quantitative evaluation harness + baselines + a benchmark suite** (the
  research-grade core — see §4).
- **Compute scaling** (§8).

---

## 7. Non-negotiable working rules (the quality bar)

1. **Keep `CONTEXT.md` updated** — append a Change Log entry for every
   meaningful change, in the SAME turn as the change. Don't let commits outrun
   the changelog.
2. **Run the audit-driven loop for every major change**: research (Explore
   agents, fan-out reads) → plan (a Plan agent — lead with the biggest
   hole/risk) → implement → review (parallel agents: code-correctness, design/
   UX, a11y, and for research work a **methodology/statistics-rigor** reviewer)
   → apply CRITICAL/HIGH fixes.
3. **VERIFY every agent's load-bearing claim against the source before acting.**
   Agents produce plausible-but-wrong findings — reject the ones that don't hold.
   (This was repeatedly necessary.)
4. **Test everything** (the four gates, §5) and add tests for new behavior.
   Green before AND after.
5. **Check compatibility**: do NOT break backend route shapes, pydantic models,
   TanStack query keys, WS protocols, or sculptor library contracts unless you
   deliberately and visibly version them.
6. **Think through failure modes** for every change — enumerate how it fails and
   make failures **recoverable + observable, never silently fatal**. The
   criterion/redecomposition work (22q/22r) is the template: one bad key must
   never silently halt a multi-hour run.
7. **Commit per logical ship** (Ship-NN pattern), push to the working branch
   (`ship-20-ux-revamp`; PR #1 auto-updates). End commit messages with
   `Co-Authored-By: Claude <noreply@anthropic.com>`.
8. **Use agents to review every significant decision and to evaluate results
   objectively** — not just to write code.
9. Prefer additive, reversible changes; confirm destructive/irreversible
   actions; **no scope creep**.

---

## 8. External GPU compute (cheap, individual use — the user will pay for this)

Training (mjlab rsl_rl PPO) is the bottleneck on the 8 GiB laptop. The user
wants the cheapest viable way to offload it for **individual** use (not a
cluster). Research + propose, with a cost + setup estimate and the SMALLEST
change to enable it:

- **On-demand / spot GPU**: Vast.ai or RunPod (RTX 4090 ≈ $0.3–0.5/hr, A100
  ≈ $1/hr), Lambda, Paperspace; or **Modal** (serverless, pay-per-second — good
  for bursty individual runs).
- **Minimal architecture**: the per-iteration `adapter.train` step is the
  GPU-heavy part. A remote path dispatches that step to a rented GPU (mjlab +
  the project's config + current reward), syncs back the checkpoint + rollout
  artifacts, and keeps diagnose/edit/KG (LLM + light) local. Alternatively run
  the WHOLE sculptor on the remote box and view the UI over an SSH tunnel —
  evaluate which is simpler.
- Don't over-engineer (individual use, not scale). A containerized sculptor +
  mjlab image + an artifact sync is likely enough.

---

## 9. Your deliverable

1. **Orient** (§0) and confirm a green baseline.
2. **Produce a comprehensive plan** to take RL-Sculptor toward research-grade,
   organized **Now / Next / Later**, each item with: goal, why it matters,
   effort + risk, dependencies, and the gate/experiment that proves it. The plan
   MUST cover:
   - (a) the **evaluation + rigor** work that defines "research-grade" —
     baselines, ablations, multi-seed metrics, a benchmark task suite,
     reproducibility;
   - (b) **capability gaps + future capabilities** (§6) — especially object
     interaction / manipulation;
   - (c) the **cheap-GPU compute path** (§8);
   - (d) **hardening the core** — reward/criterion contract, decomposition
     quality, and an experiment that MEASURES whether KG-grounding helps.
3. **Present the plan for approval first** (use plan mode) — then execute it
   phase by phase under the §7 rules, gates green at each step, `CONTEXT.md`
   updated per ship.

Be ambitious but rigorous. The bar: a system whose central claims
(KG-grounding + auto-curriculum improve LLM reward design) are backed by
**reproducible, baseline-compared, statistically honest** evidence.
