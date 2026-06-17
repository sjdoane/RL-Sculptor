# Ship 18b — Mission UI (handoff prompt for new context window)

You are taking over an audit-driven multi-ship project. **Your task: implement Ship 18b — the Mission frontend** in the reward-sculptor-ui React app. Backend is already shipped (Ship 18a) with a stable, tested API surface. You do **frontend only**.

---

## 0. Mandatory execution order

Follow this exact pattern. **Do NOT skip any step.** It's the same pattern Ships 14-18a established and the user explicitly wants you to mirror.

1. **RESEARCH (Explore agent)** — map the existing `ProjectDetail` page structure, React Query patterns, WebSocket reconnection patterns elsewhere in the codebase, shadcn primitives in use. You must produce a file:line-cited report before writing any plan.
2. **DRAFT PLAN v1** — write a file-by-file plan with explicit out-of-scope list. Identify risks.
3. **PLAN AUDIT (Plan agent)** — spawn a Plan-type agent to critique your v1 plan. Tell it to lead with the biggest hole. Incorporate findings into **Plan v2** before writing any code.
4. **IMPLEMENT** — write the code per Plan v2. Match existing component conventions (file naming, CSS classes, Tailwind utilities, React Query keys).
5. **CODE AUDIT (1-2 audit agents)** — spawn one or two `Explore`-type agents to review the implementation. One should focus on **correctness + UX edge cases** (loading/error states, empty data, race conditions). A second can focus on **integration + accessibility** (keyboard nav, focus traps, screen-reader labels, mobile breakpoint).
6. **APPLY AUDIT FIXES + REGRESSION TESTS** — fix every CRITICAL and HIGH finding. For MEDIUMs, fix or document. For LOWs, defer with a comment. **Add a regression check for each fix you apply** — usually a small test or a source-inspection guard.
7. **VERIFY GATES** — see §5 below. All four must pass.
8. **UPDATE CONTEXT.md** — replace the "Ship 18b: queued for next-window handoff" section with the actual landed entry (same format as Ship 18a).

---

## 1. Required reading order

Read these files top-to-bottom before doing anything else. The first two will give you nearly the entire context.

1. **`~/projects/CONTEXT.md`** — newest-first change log. The Ship 18a entry (top) is what your work builds on; the Ship 18b entry has the goals you're delivering.
2. **`~/projects/reward-sculptor-ui/backend/models/mission.py`** — pydantic shapes. Frontend types in `frontend/src/lib/types.ts` mirror these and are already done.
3. **`~/projects/reward-sculptor-ui/backend/routes/missions.py`** — REST endpoints + WebSocket route. You're consuming these; the contracts are stable.
4. **`~/projects/reward-sculptor-ui/frontend/src/lib/api.ts`** (search for "Missions (Ship 18a)") — `listMissions`, `getMission`, `createMission`, `runMission`, `deleteMission`, `missionEventsWsUrl` — all wired and tested. **Do not modify these.**
5. **`~/projects/reward-sculptor-ui/frontend/src/lib/types.ts`** (search for "Missions (Ship 18a)") — `MissionSummary`, `MissionDetail`, `StageSchema`, `CreateMissionRequest`, `MissionEvent`, `MissionLifecycleStatus`, `MissionJobKind`. **Do not modify these.**
6. **`~/projects/reward-sculptor-ui/frontend/src/pages/ProjectDetail.tsx`** — the page you're adding a tab to. Tabs primitives at lines 30, 255-257.
7. **One existing tab component** (e.g., `frontend/src/components/RunsTab.tsx` or `RewardsTab.tsx`) — to learn the conventions: React Query keys, error handling, loading states, layout.

---

## 2. Goals (what Ship 18b delivers)

1. **Missions tab** added to `ProjectDetail` alongside Rewards / Physics / Runs / KG / Reports.

2. **Mission list view**: table of missions with these columns:
   - Goal (truncated, full on hover)
   - Lifecycle chip — `ready` / `running` / `completed` / `halted` / `errored` (each a distinct color)
   - Stages count (`current_stage_idx / n_stages`)
   - Created at (relative time, e.g., "3m ago")
   - Actions (Run, Delete)
   Errored missions (corrupt mission.json — surface as `lifecycle="errored"`) show a distinct chip and only the Delete action.

3. **"New mission" dialog** opened by a top-right "New mission" button. Fields:
   - Goal (textarea, 8-2000 chars, required)
   - `mission_slug` override (input, optional, pattern `^[a-z][a-z0-9_-]{0,63}$`)
   - `no_kg` checkbox (default unchecked)
   - Submit → `createMission(slug, body)` → 202. On success: invalidate the list query; show toast.

4. **Mission detail panel** opened by clicking a row. Use a Dialog (no Sheet primitive in the project). Content:
   - Goal + decomposition_rationale at top.
   - Stage list as a vertical tree with parent_stage links; each stage card shows: status chip, name, goal_text, success_criterion (mono font), iterations_used / max_iterations, last metric, redecomposition_attempts (only if >0).
   - Action buttons: Run mission (if `lifecycle == "ready"`), Delete (always, except when `active_job_id`).

5. **Live event stream** (the load-bearing UI piece):
   - Open the WebSocket via `new WebSocket(missionEventsWsUrl(slug, mission_slug))`.
   - **Open AFTER `runMission` succeeds**, not at mount — Ship 18a audit finding A. Pre-mount the WS would attach to no_active_job and close.
   - Render two panels:
     a. Scrolling log panel (last 200 `log_line` events; auto-scroll-pinned-to-bottom unless user scrolls up).
     b. Structured events list — each `stage_*`, `mission_*`, `redecomposition_*`, `feedback_read_degraded` event as a chip / banner with type + key fields.
   - Close cleanly on a `terminal` event or when the user navigates away.

6. **Polling fallback** when no WS is connected: refetch the mission list every 5 s while the tab is visible AND a mission has `active_job_id` set. Stop polling when no missions are active.

---

## 3. Out of scope (do NOT build)

- DAG visualization with d3 / SVG nodes-and-edges. The vertical-tree-with-indents in goal #4 is sufficient.
- Time-lapse mission video (Ship 18c).
- Mid-mission cancellation UI (the backend supports `POST /jobs/{id}/stop`; defer the UI hookup).
- Resume-from-stage selector.
- Per-stage knob overrides.
- New backend endpoints. **You do not modify any backend file.** If you discover the need to, surface it as a finding and stop.

---

## 4. Files you should create / modify

| File | New / Modify | Purpose |
|---|---|---|
| `frontend/src/components/MissionsTab.tsx` | NEW | main tab content (list + actions + opens detail dialog) |
| `frontend/src/components/NewMissionDialog.tsx` | NEW | create-mission form dialog |
| `frontend/src/components/MissionDetailDialog.tsx` | NEW | drill-down view (or fold into MissionsTab if simpler) |
| `frontend/src/lib/useMissionEvents.ts` | NEW | WebSocket subscription hook (handles open-after-run, close-on-terminal) |
| `frontend/src/pages/ProjectDetail.tsx` | MODIFY | add the new tab to the tabs list |
| `frontend/src/lib/api.ts` | DO NOT MODIFY | already wired in Ship 18a |
| `frontend/src/lib/types.ts` | DO NOT MODIFY | already wired in Ship 18a |
| `backend/**` | DO NOT MODIFY | Ship 18a ships the contract you consume |

If you find yourself wanting to modify `api.ts`, `types.ts`, or any backend file, **stop and surface it as a finding**.

---

## 5. Verification gates (all must pass before you mark Ship 18b done)

1. **TypeScript**: `cd ~/projects/reward-sculptor-ui/frontend && pnpm tsc --noEmit` returns 0. No new type errors anywhere.
2. **Backend tests stay green**: `cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q -k 'not test_reward_prompt_edit_emits'` returns **298 passed, 1 deselected** (no regressions from your work).
3. **Sculptor tests stay green**: `cd ~/projects/RewardSculptor && uv run pytest tests/ -q` returns **291 passed, 1 skipped**.
4. **Live smoke** (after `./run.sh`): create a mission via the UI, observe its decompose job complete, run the mission, observe live events stream into the detail panel. If you can't get the live smoke to work end-to-end, the ship isn't done — report what you saw and pause for user input.

---

## 6. Risks the previous ships flagged for you

- **Ship 18a audit finding A (deferred to you)**: the WebSocket route accepts even with no active job and immediately sends `{type: "no_active_job"}` then closes. **DO NOT open the WS at mount.** Open it only inside the `onSuccess` callback of the `runMission` mutation, OR when an existing mission already has `active_job_id` set on its summary. Add a 1-line code comment referencing this finding.

- **WS reconnection on transient disconnect**: If the connection drops mid-execution (network blip, server restart), the user loses the stream. Decision for v1: do NOT auto-reconnect. Show a "WebSocket disconnected — refresh to retry" banner. Auto-reconnect with replay-from-seq is a Ship 18c follow-up.

- **Many `log_line` events flood the UI**: a long-running mission emits thousands of stdout lines. Cap rendered log entries at the last 200 (mirror what the backend's WS replay does). Plus a "show all" mode that fetches from the per-job log file (a Ship 18c if needed).

- **React Query stale-time vs WS push**: a `MissionDetail` query fetched via `useQuery` won't auto-update when a stage transitions from `pending` → `succeeded`. Either (a) invalidate the query when a `stage_succeeded` event arrives on the WS, or (b) set a 5 s polling interval while there's an active job. Pick one and document it.

---

## 7. The user's working style (excerpt from `MEMORY.md`)

- Terse, file:line over prose. No emojis.
- Confirm destructive ops.
- Run-style: `./run.sh` opens the UI; the user expects every feature to be UI-reachable after that command, not via terminal.
- Edit tool strips exec bit — `chmod +x` after any `*.sh` edit (you probably won't touch shell files).

---

## 8. When you're done

1. Verify all four gates in §5 pass.
2. Replace the "Ship 18b: queued for next-window handoff" entry in `~/projects/CONTEXT.md` with the actual landed entry. Format-match the Ship 18a entry: scope, process, files changed, audit findings + fixes, verified gates, callable surface, what's NOT included (Ship 18c).
3. Push if the user asks. Otherwise stop and report.

---

## 9. Quick context links

- Ship 17 entry in CONTEXT.md — re-decomposition events your UI must render
- Ship 16 entry — orchestrator events your UI must render
- Ship 14 entry — `Stage` shape (already mirrored in `types.ts`)

Read CONTEXT.md top-to-bottom for the architectural arc — Ships 14-18a build the system you're putting a UI on. The mission orchestrator is the system. You are surfacing it.

---

**Start by reading the files in §1, then spawn an Explore agent for the research step. Do NOT skip the audit steps.**
