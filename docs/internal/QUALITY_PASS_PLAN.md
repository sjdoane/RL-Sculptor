# Quality pass — iter-0 blocker + KG-mandate + surfaced risks

Written 2026-04-22 00:50 in response to Sam's retry that got all the way
through train + rollout + video and then errored inside `edit.apply_edits`
during the v0→v1 pre-flight. Plus his request to make every physics /
reward change force a KG consult, plus the open-ended "fix everything
important".

Priority is unblocking iter 1 first. KG-mandate lands in the same pass
since it's narrow. Tail section is a ranked list of non-blocking
improvements — none land in this pass unless flagged `SHIP`.

---

## Phase A — fix the iter-0→iter-1 pre-flight IndexError   `SHIP`

**Error** (from `_run_job_01b5876ddffebe4e.log`):

```
_pre_validate → _current_reward_component_keys → _call_compute_reward
  → v1.compute_reward → v1.compute_reward_batched
  → qpos = next_state["qpos"]
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis
(`None`) and integer or boolean arrays are valid indices
```

**Root cause**: `sculptor.edit._build_dummy_inputs(contract)` calls
`_dummy_from_space(contract.observation_space_spec)` to synthesize a
dummy state. For gym_sb3 that's a `Box` → returns a zero numpy array.
For mjlab, `observation_space_spec = None` (the contract uses
`state_schema: dict[str, tuple[int,...]]` instead), so the fallback
returns a 1-element array. The reward module's scalar wrapper passes
that array straight into `compute_reward_batched`, which does
`next_state["qpos"]` — hence the IndexError.

**Fix**: `_build_dummy_inputs` checks `contract.state_schema` first.
When present, build a dict of `{key: np.zeros(shape)}` (leading-dim
1 to keep it "one env"). Fall back to the old path when state_schema
is None. info dummies stay numeric — reward modules that accept
tensors already handle `torch.as_tensor([0.0])` in their wrapper.

**Test**: add a unit test that constructs a schema-based contract and
passes it through `_call_compute_reward`. Mjlab `MjlabAdapter.reward_contract()`
is the canonical source of such a contract, so the test can import
`_VELOCITY_STATE_SCHEMA` directly — no GPU needed because v0.py's
`compute_reward_batched` doesn't invoke mujoco.

---

## Phase B — every reward / physics change must consult the KG   `SHIP`

Sam's mandate. Three surfaces that write new reward or MJCF from
natural language; audit of current behavior:

| surface | entry point | currently reads KG? |
|---|---|---|
| reward — sculpt auto-edit per iter | `sculptor.edit.apply_edits` via `sculpt_run` | yes — `diagnose` injects `# LITERATURE CONTEXT` + edits carry `paper_refs`; validator rejects uncited claims |
| reward — prompt edit from Rewards tab | `sculptor.edit.apply_prompt_edit` | **partial** — takes `kg_store` but synthesizes a diagnosis with `literature_context=[]` and never queries it; Claude writes the new reward without any KG context |
| physics — prompt edit from Physics tab | `backend.services.physics.apply_prompt_edit` | **no** — function doesn't know what a KG is |

**B1 — reward prompt edit grounds via KG before handoff**:
`sculptor.edit.apply_prompt_edit` queries
`sculptor.kg.query.query_semantic(user_prompt, top_k=5)` and threads
the matches into the synthetic Diagnosis's `literature_context`.
`apply_edits` already forwards `diagnosis.literature_context` into the
Claude `edit_rewriter` prompt, so no prompt-template changes needed.

**B2 — physics prompt edit injects KG context into system prompt**:
`backend.services.physics.apply_prompt_edit` gains a `kg_store`
parameter. Before calling Claude, query `query_semantic(prompt, top_k=5)`
and render via `sculptor.diagnose._render_kg_context`. Inject a
`# LITERATURE CONTEXT` block into the user message. Return the matched
arxiv_ids on the result dict so the UI can render citations next to
the commit. `reward_jobs.run_physics_prompt_edit_job` opens a
`SculptorKG` against `project_kg_db_path(project_dir)` and passes it
through.

**Tests**:
- `test_reward_prompt_edit_consults_kg` — monkey-patch `query_semantic`,
  assert it was called with the user prompt, and assert the returned
  TechniqueMatch titles appear in the prompt Claude received.
- `test_physics_prompt_edit_consults_kg` — same pattern against the
  physics entry point.

---

## Phase C — additional warnings already surfaced; fix the easy ones

### C1. `reward_spec.json missing` warning at diagnose time   `SHIP`

Log shows:
```
[diagnose] warning: ...runs/iter_0/reward_spec.json missing —
diagnosis prompt will show an empty REWARD_SPEC.
```

`gym_sb3` writes `reward_spec.json` into the iter dir; `mjlab` doesn't.
Diagnose therefore runs on an empty REWARD_SPEC → Claude's failure-mode
analysis is weaker. Fix by teaching `MjlabAdapter.train` to drop a
`reward_spec.json` alongside `checkpoint.pt` that echoes the loaded
reward module's `REWARD_SPEC`. Two lines of write_text. No test gate.

### C2. Resume-from-iter (don't retrain iter 0 when post-train phase fails)   `defer`

Current: any post-train failure (rollout, diagnose, edit) errors the
whole run. Sam then re-clicks Run and pays ~22 min for iter 0
re-training even though `iter_0/checkpoint.pt` is on disk and
`runs/iter_0/rollout/rollout.mp4` may also exist. Fix sketch: before
`adapter.train()`, check for `iter_dir/checkpoint.pt`; if present
AND it parses, skip the training call and reuse it. Behind a
`[iteration].resume_from_disk` toggle (default on). Not this pass
because the sculpt loop's atomicity semantics need careful thought —
partial rollout state could break replayability.

### C3. UI: rollout.mp4 playback in the iter card   `defer`

`runs/iter_<i>/rollout/rollout.mp4` exists (verified this pass) but
the Runs tab doesn't embed an HTML5 `<video>` tag pointing at a
`/projects/<slug>/runs/<id>/iters/<i>/rollout.mp4` endpoint. Sam
asked for behavioral verification earlier — the fastest path is
letting him watch the robot in the tab. Medium effort: needs a
backend route for streaming mp4 + a `<video>` in `IterPanel`.

### C4. `ring-buffer cap` toast / indicator in UI   `defer`

The 20 k client-side event cap in `useRunEvents.ts` silently drops
old events. A `<20k / 20k buffered — older events in log file>`
indicator would avoid the "exactly 20000 entries, is it stuck?"
confusion that hit Sam earlier. Two-line indicator + tooltip.

### C5. GPU-gated rollout regression test   `defer`

`tests/test_mjlab_gpu.py` covers train; doesn't cover rollout. A
test that trains 20 iters on Go1, runs `adapter.rollout(...)`,
asserts `rollout.mp4` exists AND was produced in < 90 s would
catch both the `num_envs=1` regression and the WSL-EGL software
renderer hang. Gated by `@pytest.mark.gpu` so CI doesn't need
CUDA.

### C6. Alerting when sculptor_primary regresses across iters   `defer`

If `iter_<i+1>.primary_metric < iter_<i>.primary_metric`, flag the
iter in the Timeline with a red "regressed" marker + push a
diagnostics banner. Helps Sam spot reward-function collapse
(e.g. Claude accidentally killed the height term) before spending
another 22 min retraining. Needs a metric-history tracker in the
iter merger.

### C7. Checkpoint atomic write   `defer`

`_cmd_train`'s periodic `model_<N>.pt` copy to `checkpoint.pt` is
`shutil.copy` — not atomic. If the backend SIGTERMs mid-copy (Sam
hits Kill), the file on disk can be a truncated torch archive.
Swap for `write-to-tempfile + rename`. Trivial, but low probability
of biting.

---

## Log

- 2026-04-22 00:50 — plan written.
- 2026-04-22 00:55 — Phase A shipped.
- 2026-04-22 01:05 — Phase B1 + B2 shipped.
- 2026-04-22 01:10 — Phase C1 shipped.
## T11 acceptance — overnight 12-iter run (added 2026-04-22 17:00)

Landed as part of the new-window plan's S7 regression gates. This is
a manual checklist Sam runs once per sculptor / mjlab bump (not on
every ship); it validates the claims §7.10 makes about overnight
robustness.

Happy path:
- 12 iters × 22 min = **4.5 h wall-clock target** on an RTX-5070
  Laptop at `steps_per_iter=1500, num_envs=2048`. If the run takes
  meaningfully longer, something regressed in mujoco_warp kernel
  scheduling — check `nvidia-smi dmon` during a single iter.
- Every iter produces: `runs/iter_<i>/checkpoint.pt` (atomic write
  via `.pt.tmp + os.replace` — shipped 2026-04-22 01:40),
  `runs/iter_<i>/rollout/rollout.mp4`, `trajectory.npz`,
  `behavior.json`. Missing any of these should re-trigger only that
  phase on retry (per-phase resume logic).
- `reward_spec.json` drops into each iter dir (mjlab adapter gained
  this in the 2026-04-22 01:10 entry), so diagnose gets a real
  REWARD_SPEC block rather than the empty fallback.

Failure-injection gates:
- **Backend reload mid-run** (`pkill -USR1 uvicorn` or touching
  `backend/main.py`): mjlab subprocess survives because it's in a
  detached process group (`start_new_session=True`). WS reconnects on
  the frontend side within 8 s via `useRunEvents`' retry. UI re-attaches
  via replay — no data loss.
- **Forced SIGKILL of the sculpt subprocess mid-checkpoint-copy**:
  tests `_cmd_train` atomic-write integrity. Partial `checkpoint.pt.tmp`
  gets left behind; the next run's resume logic passes the `torch.load`
  integrity check (because the final `checkpoint.pt` either exists
  complete or doesn't exist at all) and retrains just that iter.
- **Laptop sleep mid-run**: mujoco_warp may or may not survive a
  CUDA-context suspend. Unverified. Workaround: disable sleep via
  `systemctl mask sleep.target suspend.target hibernate.target
  hybrid-sleep.target` on the laptop for the duration of the run.
- **Anthropic API 429 spike**: `max_retries=6` at diagnose.py /
  edit.py / physics.py hot paths gives a ~79%→99.9% run-success
  lift under 1% per-call transient rates (math in the 2026-04-22 01:40
  entry).

To run:
- `cd ~/projects/reward-sculptor-ui && ./run.sh`
- Create / pick a Go1 project. Hit "New run" with `iterations=12`.
- Let it run. Open the Runs tab the next morning. Per §9 this is
  T11 passing.

Degraded outcomes worth flagging (all still-passing per §9 but worth
watching):
- Early-stop triggered before iter 12 due to 3 non-improving iters
  in a row — correct behaviour (see 2026-04-22 05:30 entry), but
  might still indicate a reward-function collapse worth a manual diff
  of v<N-1>.py vs v<N>.py.
- Iter artifact missing but backend still thinks run is "running" —
  backend crash mid-iter. Look at
  `~/.local/share/reward-sculptor/projects/<slug>/runs/_run_job_<id>.log`.

---

- 2026-04-22 01:40 — overnight-reliability pass. Shipped:
  * **Anthropic SDK retries → `max_retries=6`** at [diagnose.py:400](RewardSculptor/sculptor/diagnose.py),
    [edit.py:621](RewardSculptor/sculptor/edit.py),
    [physics.py:560](reward-sculptor-ui/backend/services/physics.py).
    SDK default is 2; bumps give transient 429 / 500 / network blips
    a realistic chance of recovering over a 4-hour run. 24 API calls
    per 12-iter run × P(transient failure) = single biggest error
    source before this pass.
  * **Atomic checkpoint write** in [_mjlab_runner.py](RewardSculptor/sculptor/adapters/_mjlab_runner.py)
    — `shutil.copy(...)` → `copy to .pt.tmp + os.replace`. Prevents a
    truncated `checkpoint.pt` if the backend gets SIGKILL'd mid-copy,
    which would torpedo the new resume-from-iter logic.
  * **NaN / Inf guard at pre-flight** in [edit.py::_call_compute_reward](RewardSculptor/sculptor/edit.py).
    `math.isfinite(reward)` + per-component check. Pre-flight now
    rejects runaway rewards (unguarded divisions, `log(0)`, unclipped
    `exp`) before they reach rsl_rl and poison PPO's gradient — old
    behavior let NaN slip through and crash training 10-30 iters in.
  * **Per-phase resume in `_run_one_iter`** — new `_train_or_resume` +
    `_rollout_or_resume` helpers in [sculpt.py](RewardSculptor/sculptor/sculpt.py).
    If `iter_<i>/checkpoint.pt` is on disk and torch.load-able, skip
    `adapter.train` (~22 min saved). If all three rollout artifacts
    (`rollout.mp4` + `trajectory.npz` + `behavior.json`) exist, skip
    `adapter.rollout`. Corrupt ckpt falls through to a fresh train.
    Emits `[SCULPT-EVENT] phase_skipped` for UI breadcrumbs.
  * **Always-resume from backend** in [run_manager.py](reward-sculptor-ui/backend/services/run_manager.py)
    — `--resume` passed unconditionally. Fresh projects are unaffected
    (start_iter=0 either way); existing projects with partial runs
    pick up at the next unfinished iter, reusing any artifacts still
    on disk.
- **Tests added**: `test_train_or_resume_skips_when_checkpoint_present`,
  `test_train_or_resume_falls_through_on_corrupt_checkpoint`,
  `test_rollout_or_resume_skips_when_artifacts_present` in
  [test_sculpt.py](RewardSculptor/tests/test_sculpt.py).
  Sculptor suite now 97 passed (+3).
- **Rollout video UI**: already wired in [RobotViewer.tsx:354](reward-sculptor-ui/frontend/src/components/RobotViewer.tsx)
  via the existing `/projects/{slug}/runs/{run_id}/iterations/{iter_index}/rollout`
  endpoint. Sam just needs to open the Runs tab after each iter
  finishes — the RobotViewer hangs off of `useLatestRun` + picks up
  new rollout.mp4 files automatically.
