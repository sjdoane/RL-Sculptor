# Ship 20 — UX revamp + cross-tab mission integration

You are taking over an audit-driven multi-ship project. **Your task: implement Ship 20 — UX label cleanup + cross-tab mission integration** in the reward-sculptor-ui React app + sculptor library. This is a UI-heavy ship; spawn a website-designer agent to plan the cross-tab integration before implementing.

---

## 0. Mandatory execution order

Follow this exact pattern. **Do NOT skip any step.** It mirrors Ships 14-19d.

1. **READ §1 first**: it tells you which files to look at and in what order. Do that *before* writing any plan.
2. **RESEARCH** (`Explore` agent + your own reads) — map current Missions/Runs/Rewards/Physics/Overview tab structure, WS event flow, query keys, the MissionDetailDialog shape, the existing iter-display code paths. Produce a file:line-cited report.
3. **WEBSITE-DESIGNER PLANNING (`Plan` agent, framed as designer)** — see §2 goal #5. Hand it the cross-tab integration question + screenshots context + the user's vision (in §3 below). Get back a concrete information-architecture proposal.
4. **DRAFT PLAN v1** combining the small fixes (§2 goals #1-#4) with the designer's cross-tab proposal (§2 goal #5).
5. **PLAN-AUDIT** (`Plan` agent) — critique your v1 plan. Lead with the biggest hole. Incorporate findings into Plan v2.
6. **IMPLEMENT** per Plan v2.
7. **CODE-AUDIT (1-2 `Explore` agents)** — one for correctness/integration, one **specifically framed as a website-design critic** for the UI code (see §2 goal #6). Apply CRITICAL/HIGH fixes + add regression tests.
8. **VERIFY GATES** — see §5.
9. **UPDATE CONTEXT.md** with the landed Ship 20 entry (same format as Ship 19 / Ship 18a).

---

## 1. Required reading order

1. **`~/projects/CONTEXT.md`** — newest-first change log. Read at minimum:
   - The "Ship 19c UX fixes" entry (datasheet + auto-open + iter ribbon).
   - The Ship 19d entry (run-mission config UI + Goal A/B). **You will rename Goal A/Goal B in Ship 20.**
   - Ship 19, Ship 18b, Ship 16-17 entries for the architectural arc.
2. **`~/projects/HANDOFF_SHIP_18B.md`** — the process pattern to follow.
3. **`~/projects/TEST_WORKFLOW_SHIP_18B_19.md`** — how the system is tested end-to-end. Re-run after your changes.
4. The recent commits on the `ship-19-skill-library` branch (the head as of this handoff): `git log --oneline ship-19-skill-library` — there are 4 commits; the last (1dfdc61) is Ship 19d.
5. **Files you'll touch** — read these top-to-bottom before editing:
   - `frontend/src/components/RunMissionDialog.tsx` — Goal A/B labels live here.
   - `frontend/src/components/MissionDetailDialog.tsx` — stage card iter display, structured event panel, log scroller.
   - `frontend/src/components/MissionsTab.tsx` — mission list + row actions.
   - `frontend/src/components/RunsTab.tsx` — the existing tab pattern that missions should integrate with (or replace).
   - `frontend/src/components/RewardsTab.tsx`, `frontend/src/components/PhysicsTab.tsx` — sibling tabs that may need mission-aware extensions.
   - `frontend/src/pages/ProjectDetail.tsx` — the tab container.
   - `RewardSculptor/sculptor/sculpt.py` — `_run_one_stage` (line ~1909) emits `stage_*` events; you may add `effective_max_iterations` to its `stage_started` / `stage_completed_training` payloads.
   - `reward-sculptor-ui/backend/services/mission_jobs.py` — subprocess streamer + JobManager registration. Cross-tab integration may need this to register child Job entries per stage.
   - `reward-sculptor-ui/backend/routes/runs.py` — `list_runs` endpoint + RunSummary shape.

---

## 2. Goals

### Goal #1 — Rename Goal A / Goal B in the UI

The internal CLI/library kwargs stay as `early_stop_on_criterion` and `extend_on_improvement` (don't churn the API). Only the user-facing labels in `RunMissionDialog.tsx` change. The website-designer agent (§5) should pick the final names from these (or propose better ones):

- `Goal A: early-stop on criterion` → candidates: "Finish stage early when goal is met" / "Stop at goal" / "Adaptive early-finish".
- `Goal B: extend on improvement` → candidates: "Keep training while still improving" / "Auto-extend on progress" / "Adaptive extension".

Help text under each option also needs simplification — current text references "Ship 9a" and "Ship 16" internal versions; remove those.

### Goal #2 — Fix the "iters 2/3" display when `iterations_override` is set

**The bug**: user sets `iterations_override=2` in RunMissionDialog. Stage's `max_iterations` was authored by Claude as 3. After the run, MissionDetailDialog's StageCard shows `iters 2/3` — confusing because user expected `iters 2/2`. The actual training ran exactly 2 iters; only the display is wrong.

**Fix**:
- Backend: `_run_one_stage` already computes `max_iters = iterations_override or stage.max_iterations`. Persist this as `effective_max_iterations` on the stage's runtime state (in-memory only — DON'T overwrite Claude's authored `stage.max_iterations`, the persisted JSON field).
- The `stage_started` and `stage_completed_training` events should include `effective_max_iterations` in their payload.
- Frontend: StageCard reads from a new event-derived field; falls back to `stage.max_iterations` when no override is set.
- Add a tooltip when override differs: "Claude allocated 3 iters; this run capped at 2."

Add a regression test in `tests/test_mission_run.py` confirming the events carry `effective_max_iterations`.

### Goal #3 — Investigate why the auto-open dialog still doesn't fire

Sam reports: "the window still didn't immediately pop up when I sent the mission prompt." Ship 19d's `JobSummary → JobDetail` response_model fix should have resolved this. Three hypotheses to verify in order:

1. **uvicorn isn't picking up the route change.** `./run.sh` may not pass `--reload`; FastAPI doesn't auto-reload modified route signatures. Confirm by stopping `./run.sh` cold and restarting.
2. **Optimistic-cache-insert is racing the `invalidateQueries` refetch.** The list refetch lands and overwrites the placeholder before React commits. Add a query-cache observer log in `useCreateMission.onSuccess` or use `setQueryDefaults` to delay the refetch.
3. **`params.mission_slug` isn't on the wire.** Add a regression test in `backend/tests/test_missions.py`:
   ```python
   def test_create_mission_response_includes_params_mission_slug(client, ...):
       resp = client.post(...)
       assert resp.status_code == 202
       assert "params" in resp.json()
       assert resp.json()["params"]["mission_slug"]
   ```

If none of those, dig deeper — but report back rather than implementing speculative fixes.

### Goal #4 — Audit + improve other UI labels

Spend ≤30 minutes on this; don't bikeshed forever. Hand the website-designer agent (§5) a screenshot of the Missions tab + MissionDetailDialog + RunMissionDialog and ask for a labels audit. Specific suspects from Sam's session:

- Missions tab description (`Claude decomposes a goal into a curriculum of stages...`) — too implementation-detail (mentions `.missions/<slug>/mission.json`).
- `MISSION_EXECUTE` / `MISSION_DECOMPOSE` badges — uppercase+underscore reads as internal. Replace with "Running curriculum" / "Decomposing".
- Mission row slug truncation `balance-the-cartpole-at-the-upri` — looks like a parse error; needs an ellipsis.
- `0/3 stages` — ambiguous; switch to "0 of 3 complete" or a progress bar.
- `DRAFT` chip on project — what does it convey?
- RunMissionDialog field help references "rsl_rl iters (mjlab) / env steps (gym)" — non-user vocabulary; replace with "training steps per cycle" or split per-adapter.

The designer agent's output should be a markdown table mapping `current label → proposed label → rationale`.

### Goal #5 — Cross-tab mission integration (the big feature)

**Sam's vision (verbatim quote, paraphrased)**: missions should populate the Runs tab as they execute. Each stage's run becomes a drop-down inside Runs with its iter timeline. Reward changes per stage visible in Rewards (so you can see the rationale per stage). Live videos visible in Overview as they play out. Physics edits applied per stage visible in Physics. Possibly merge the Missions tab into Runs entirely. The exact arrangement is the designer agent's call.

**Process**:

1. **Spawn a `Plan`-type agent** with this prompt:
   > You are a website designer reviewing a research-tooling React app. The user has a Missions tab (Claude-decomposed curriculum) running curricula of stages, and separate Runs / Rewards / Physics / Overview tabs that operate on standalone training runs. The user wants mission stages to populate Runs (drop-down per stage with iter timeline), reward evolution to show per stage in Rewards (so the user sees WHY rewards changed), live training videos to show in Overview as they play, and physics edits per stage to show in Physics. Possibly merge Missions into Runs entirely.
   >
   > Produce: (a) an information-architecture proposal — which tab owns what, what's redundant, what should be unified vs kept separate. (b) interaction flows — what happens when a user clicks a mission row, drills into a stage, navigates back. (c) state ownership — which React Query keys fire when, what re-renders. (d) a labels-and-copy audit (you'll get a list of suspect strings). (e) priority order — which restructure to ship first.
   >
   > Constraints: existing test coverage must remain green; Ships 18b-19d's audit-driven fixes (Ship 18a finding A WS gating, optimistic cache writes, Ship 17 redecomp path) must not regress; no backend mission_run library changes (the orchestrator is stable). Files in scope: ProjectDetail.tsx, MissionsTab.tsx, RunsTab.tsx, RewardsTab.tsx, PhysicsTab.tsx, MissionDetailDialog.tsx, RunMissionDialog.tsx. Backend: mission_jobs.py may grow logic to register per-stage Job entries in JobManager so list_runs surfaces them.
   >
   > Reference files at /home/samjd/projects/reward-sculptor-ui/frontend/src/. Read 3-5 of the listed files before answering. Cite file:line for every concrete proposal. Lead with the BIGGEST architectural decision (merge vs keep separate) and your recommendation with rationale.

2. **Plan-audit the designer's plan.** Spawn another `Plan` agent and have it lead with the biggest hole.

3. **Incorporate audit findings into Plan v2.** Implement.

4. **After implementation, spawn an `Explore` agent framed as design critic**:
   > You are a website-design critic reviewing a React UI implementation. Read the files I changed (list them). Critique: visual hierarchy, accessibility (WCAG AA contrast, role attributes, focus order, screen-reader labels), consistency with the existing tab styles (look at RunsTab/RewardsTab as the reference), label clarity, loading state coverage, error state coverage, mobile breakpoints. Lead with the WORST UX issue. Cite file:line. Don't be polite.

   Apply CRITICAL/HIGH fixes; defer LOW with comments.

### Goal #6 — Final website-design critique pass

Same `Explore`-as-design-critic agent process described in Goal #5 step 4 — but as a final sweep over EVERY UI file you touched in Ship 20, not just the cross-tab files. Apply CRITICAL/HIGH fixes.

---

## 3. Out of scope (do NOT build)

- Backend mission_run library changes (the orchestrator is stable; only mission_jobs.py / list_runs may need new logic to register child Job entries).
- New mission-orchestrator features (Goals C, D, etc.) — Ship 21+.
- Backend mission.json schema changes (don't migrate; add fields to runtime events instead).
- Per-mission-run skill-library overrides UI — Ship 19e.
- New tests for components NOT modified by Ship 20.

If you find yourself wanting to do any of these, **surface as a finding and stop.**

---

## 4. Files you should expect to create / modify

**Estimate** — designer agent's plan will refine:

| File | New / Modify | Purpose |
|---|---|---|
| `frontend/src/components/RunMissionDialog.tsx` | MODIFY | Goal A/B label rename + help text simplification. |
| `frontend/src/components/MissionDetailDialog.tsx` | MODIFY | Display effective_max_iterations; tooltip on override. |
| `frontend/src/components/MissionsTab.tsx` | MODIFY | Label cleanup; possibly merge into RunsTab. |
| `frontend/src/components/RunsTab.tsx` | MODIFY | Surface mission-stage runs as drop-down rows. |
| `frontend/src/components/RewardsTab.tsx` | MODIFY (TBD) | Show per-stage reward evolution if designer agrees. |
| `frontend/src/components/PhysicsTab.tsx` | MODIFY (TBD) | Show per-stage physics edits if designer agrees. |
| `frontend/src/pages/ProjectDetail.tsx` | MODIFY (TBD) | Possibly merge Missions tab into Runs. |
| `RewardSculptor/sculptor/sculpt.py` | MODIFY | Emit `effective_max_iterations` on `stage_started` + `stage_completed_training` events. |
| `reward-sculptor-ui/backend/services/mission_jobs.py` | MODIFY | Register per-stage child Job entries in JobManager (if designer specifies). |
| `reward-sculptor-ui/backend/tests/test_missions.py` | MODIFY | Regression test for `params.mission_slug` on wire. |
| `RewardSculptor/tests/test_mission_run.py` | MODIFY | Regression test for `effective_max_iterations` event payload. |
| `frontend/src/lib/api.ts` | DO NOT MODIFY | already wired; only modify if absolutely required + flagged. |
| `frontend/src/lib/types.ts` | DO NOT MODIFY | only modify if absolutely required + flagged. |

If you discover you must modify api.ts/types.ts, surface it as a finding and propose the minimal change before implementing.

---

## 5. Verification gates (all must pass before Ship 20 lands)

1. **TypeScript**: `cd ~/projects/reward-sculptor-ui/frontend && pnpm tsc --noEmit` returns 0.
2. **Sculptor pytest**: `cd ~/projects/RewardSculptor && uv run pytest tests/ -q` returns 353+ passed (you'll add ≥1 new test for Goal #2's `effective_max_iterations`).
3. **Backend pytest**: `cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q -k 'not test_reward_prompt_edit_emits'` returns 298+ passed (you'll add ≥1 new test for Goal #3's `params.mission_slug`).
4. **Live smoke** (after `./run.sh`): re-run the §5/§6 steps from `~/projects/TEST_WORKFLOW_SHIP_18B_19.md`. Confirm:
   - Auto-open dialog fires immediately on Decompose submit (Goal #3).
   - Stage card shows `iters 2/2` (NOT 2/3) when override is set (Goal #2).
   - Goal A/B labels read like English (Goal #1).
   - Mission stage activity is visible in the redesigned tabs per the designer's plan (Goal #5).

---

## 6. Process rules (extracted from CLAUDE.md / MEMORY)

These are non-negotiable. Read once before starting:

- **Terse, file:line over prose. No emojis.** This applies to your responses to the user, AND to every agent prompt.
- **Confirm destructive ops** (force-push, `rm -rf` outside `runs/`/temp, anything touching AME456). Ask before doing.
- **`./run.sh` is the only entry point.** Every feature must be UI-reachable. CLI is the escape hatch, not the only path.
- **Edit tool strips exec bit** — `chmod +x` after any `*.sh` edit (probably not relevant here).
- **Shell is Windows Git-Bash, NOT WSL.** Every Bash invocation needs `wsl bash <<'EOF' ... EOF` heredoc with `export PATH="/home/samjd/.local/bin:/home/samjd/.local/share/pnpm:$PATH"`.
- **Don't add features beyond what the task requires.** A bug fix doesn't need surrounding cleanup.
- **Default to writing no comments unless the why is non-obvious.**
- **NEVER skip the audit steps.** Plan → plan-audit → plan v2 → implement → code-audit → fixes is the load-bearing pattern.
- **CONTEXT.md change log entry mandatory** when you land Ship 20. Format-match Ship 19's entry: scope, process, files, audit findings + fixes, verified gates, callable surface, what's NOT included.

---

## 7. Recent ship history (so you don't reinvent)

- **Ship 18a**: backend mission orchestration (mission.py, mission_store, routes/missions.py, job_manager mission_decompose/execute kinds).
- **Ship 18b**: frontend Mission UI (MissionsTab, MissionDetailDialog, NewMissionDialog, useMissions, useMissionEvents).
- **Ship 19**: cross-mission skill library (sculptor/skill_library.py, Stage.init_skill_id, decompose-time SKILL_LIBRARY rendering).
- **Ship 19c**: UX fixes — datasheet name-mapping, auto-open dialog (partial — see Goal #3), per-stage iter ribbon.
- **Ship 19d (today)**: run-mission config UI (RunMissionDialog), Goal A (early-stop on criterion), Goal B (extend on improvement). **Goal A/B are the labels you'll rename in Ship 20.**
- Hotfixes:
  - Stage scaffold inheritance (`_inherit_parent_adapter_config`) for both [adapter] and [iteration] sections.
  - JobSummary→JobDetail response_model on POST /missions and POST /run.

---

## 8. Branch + PR conventions

- Branch off `ship-19-skill-library`: `git checkout -b ship-20-ux-revamp ship-19-skill-library`.
- Stage only the files you authored or directly modified (per the established Ship 18b/19 pattern). Don't bundle prior-ships uncommitted work.
- Commit message format: see Ship 19d's commit (1dfdc61) for the style — mention the goals, the audit findings + fixes, verification gates, and prior-ship caveats if applicable.
- After commit: `git push -u origin ship-20-ux-revamp`. `gh` CLI is not installed locally; `gh pr create` won't work — open the PR via the GitHub URL the push prints.

---

## 9. When you're done

1. Verify all 4 gates in §5 pass.
2. Replace nothing in CONTEXT.md; APPEND a new "Ship 20" entry at the top (after the change-log header).
3. Push if the user asks. Otherwise stop and report.

**Start by reading the files in §1, then spawn the website-designer Plan agent for Goal #5. Do NOT skip the audit steps.**
