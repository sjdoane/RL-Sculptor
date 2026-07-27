# Mission UI: Results, de-siloing, and Saved-Missions library — QUEUED (2026-07-07)

All items here need backend edits → the dev server runs with
`--reload --reload-dir backend --reload-dir ../RewardSculptor/sculptor`,
so editing restarts uvicorn and would KILL a live mission run. **Do NOT
start any of this until Sam confirms the current jump mission
(`g1-jump-demo` / `from-a-stable-standing-pose-crou`) has finished.**

## Data-safety facts (verified read-only 2026-07-07)
- Project data lives at `~/.local/share/reward-sculptor/projects/` —
  **33 GB, 31 projects, persists across terminal/backend restarts.**
- Terminal/backend restart does NOT delete files. JobManager keeps job
  state **in-memory** (`services/job_manager.py`), so a restart makes
  the UI *forget* live runs (empty view) though every artifact remains
  on disk. Missions ARE re-listed from disk (`mission_store` scans dirs).
- The ONLY destructive path is the UI "reset/delete project" action.
- No `run_store` DB: runs = filesystem (`runs/iter_*`) + in-memory job.

## 1. Results tab populates from missions
- Today: `routes/reports.py` reads only `<project>/runs/`; the Reports
  tab shows a BUILT final_report.md + final.mp4 (timelapse) +
  mission-quality card. Missions store under
  `.missions/<m>/stages/<stage>/runs/iter_*`, so the tab is empty.
- Fix: make the report builder mission-aware — build a per-mission
  report (per-stage sections, each stage's iterations + rollouts +
  reward evolution + the stage's generated metric + criterion result),
  and a stitched full-mission timelapse. Surface a mission picker in the
  Reports tab. Keep standalone-run behavior byte-identical.
- Immediate (run-end) populate for the recording: build the report over
  the finished mission so the tab shows the full jump. Safe once the run
  is done (filesystem + report build, no restart needed for the build
  itself; but the mission-aware builder code is a backend edit → land it
  first, after the run).

## 2. De-silo stages across the mission UI
- Today: MissionDetailDialog + the live/replay viewer are scoped to
  `current_stage_idx` — you can only view the current stage. Completed
  stages' rollouts/rewards/iterations are on disk but unreachable.
- Fix: a **stage selector** (dropdown/tabs) in the mission viewer that
  loads ANY stage's runs — rollout videos, per-iter metrics, reward
  versions, diagnosis — not just the current one. Apply the same
  selector to the Rewards view (per-stage reward evolution) and Reports.
  Root cause is uniform: the mission UI treats each stage as an isolated
  run; give every stage-scoped view a stage argument.

## 3. Saved-Missions library (survives restart; keep favorites)
- Goal: browse persisted missions after any restart; keep ~10-20
  favorites and reopen them. The DATA already persists — this is a
  surfacing + retention feature, not new storage.
- Library view: lists missions found on disk (across projects), each
  with goal, stage outcomes, thumbnails/final video; click to reopen the
  full mission (stages, rollouts, report). Rebuilt from disk on load, so
  a backend restart never hides them.
- **Checkpoint retention (Sam's design, 2026-07-07):**
  - DEFAULT auto-keep per stage: the **final (kept) checkpoint** AND the
    **highest-scoring iteration's checkpoint** (if different). For the
    last stage that final policy = the full learned skill.
  - ALWAYS keep the small viewable artifacts regardless: rollout videos,
    reward code (v0..vN), metrics/reward_trajectory, diagnosis, realism
    audits, generated stage metrics, provenance (llm_calls/kg_retrievals),
    mission.json, the built report.
  - User can **pin additional checkpoints** via a per-iteration checkbox.
  - Non-kept, non-pinned intermediate `checkpoint.pt` files are dropped
    from a saved mission to save space (their video + metrics stay, so
    every iteration is still viewable). ~20 saved missions ≈ a few GB,
    not 60.
  - "Save mission" action = snapshot the mission dir into a durable
    archive (e.g. `~/.local/share/reward-sculptor/saved/<name>/`) applying
    the retention rules, so a later "reset/delete project" can't nuke it.
- Optional later: prune/cleanup UI for the 33 GB of old un-saved runs,
  only on explicit confirmation.

## Execution order (post-run)
1. Back up the current jump mission to `saved/` first (protect the
   footage before any edits).
2. Land #1 (Results-from-missions) + build the jump's report so Sam can
   record the Results tab.
3. Land #2 (stage selector) so the whole mission is viewable together.
4. Land #3 (Saved-Missions library + retention).
Gates each step: backend pytest, sculptor pytest, `pnpm build`.

## IMPLEMENTED (2026-07-07/08) — after a real data-loss incident

A user's successful G1 standing-jump mission (project g1-jump-demo) was
PERMANENTLY DELETED via the "delete project" rmtree path (no trash, no
backup). All of today's jump footage was unrecoverable. That triggered
this whole hardening pass. Architected by a Fable director agent;
implemented by Opus (main) + Sonnet agents. Backend/sculptor gates green
throughout (sculptor 1246 pass/1 skip; backend 426 pass).

Commits on ship-20-ux-revamp:
- a52c0d3 B1 keep-best stage finalization — the stage keeps the BEST iter
  whose rollout meets the criterion (not the last), so a jump that
  regresses to standing keeps the jumping policy. (root cause of "the
  stander got kept".)
- 9ce7fc3 A2 sculptor/archive.py — durable archive w/ retention (best+
  final+pinned checkpoints, all videos/reports/reward-code; heavy
  intermediates dropped).
- 43e9e24 A3 auto-archive hooks in mission_run (per-stage + mission-end,
  completed OR halted) + `sculpt mission-save` CLI. RS_AUTO_ARCHIVE=0
  disables; RS_SAVED_ROOT overrides.
- 906898c A1 non-destructive delete — project + mission delete move to a
  recoverable ~/.local/share/reward-sculptor/.trash/ (RS_TRASH_ROOT);
  the ONLY rmtree left is trash.purge; DELETE project 409s on active
  jobs; GET/restore/purge routes.
- ed0b790 C1 mission-aware Results (build_mission_report + reports
  endpoints + /reports/sources).
- 01d1893 C2 stage disk-truth endpoints (iterations / rollout / env-spec
  per stage) — de-silos completed stages.
- 55287e8 A4 saved-missions library backend (GET /saved, file serving,
  DELETE→trash, POST .../save job).
- Frontend (D1-D4) in progress: trash UI, Results source picker, stage
  selector, Saved Missions page.

Net: a mission now (a) can't be permanently deleted by accident, (b)
auto-saves its footage as it trains, and (c) keeps the good policy even
if a later round regresses. The re-record is durable + reliable.

## 2026-07-08 — stage-persistence deep fix (SHIPPED, ship-20-ux-revamp)

Root-caused and fixed the class of "stage vanishes / lies about status"
bugs the plan above did not cover. Commits 29fc2f2, d6ba7d3, 30c83e5,
d9d55c3, e938fbf, d56883c.

1. Replan retention (sculpt.py:5350 was a destructive splice): a
   redecomposed parent stays in mission.stages as status="superseded"
   (new StageStatus), children insert after it; failure_reason/detail
   now PERSIST on Stage; loop/resume never re-execute superseded.
2. Disk-truth union: mission detail unions mission.stages with orphaned
   stages/<name>/ dirs (restores pre-fix casualties, e.g. crouch_load),
   enriches failure_reason from provenance.json, computes hierarchical
   display_label ("1", "1.1", ... — truncation- and nesting-robust) and
   per-stage metric_status (accepted/rejected/inherited/none).
3. /runs survives backend restarts: mission stage rows are synthesized
   from disk (run_id "disk:<mission>/<stage>") when JobManager is empty;
   Replay + downloads for those rows use mission-stage endpoints.
4. Honest statuses: criterion_not_met renders as amber "Criterion not
   met (trained N iters)", mission-cancel collateral as "Stopped";
   superseded gets a neutral badge + "Re-planned into sub-stages —
   artifacts retained".
5. Results tab: joins the shared ?stage= selection, defaults to the
   viewed/running mission, per-stage disk-truth checkpoint list with
   rollout links + NEW checkpoint download endpoint.
6. Metric choice: stage_metric_required (create-time, blocks runs via
   409 stage-metrics-missing unless proceed_blind), per-stage POST
   .../metric/regenerate (409-guarded), metric chips in the mission
   dialog, New-Mission Advanced hint now derives from the Basic
   per-stage toggle (no more silent "none (blind loop)").

KNOWN GAP (pre-existing, deliberate defer): plain PROJECT-level runs
still have no disk-reconstructed Training-sidebar row after a backend
restart (artifacts remain reachable via Results "Project runs" source).
Needs a project-run disk pane + project-level rollout endpoint.
Also deferred: /runs disk-scan does O(stages×iters) syscalls per GET —
fine single-user, cache if it ever hurts. One pre-existing flaky test:
test_create_mission_returns_202_with_jobsummary (lock-timeout race
under full-suite load).
