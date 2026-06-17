# End-to-end test workflow — Ship 18b + Ship 19 (+ follow-up UX fixes)

This is the **complete, copy-paste-runnable** smoke test that exercises every load-bearing path in Ships 18b and 19, plus the follow-up UX work just landed:

- **Mission UI** (Ship 18b): Missions tab, NewMissionDialog, MissionDetailDialog with live event stream, Run/Delete, deferred-mount, optimistic cache.
- **Skill library** (Ship 19): per-stage publish on success, decompose-time skill context, cross-mission warm-start, CLI flags.
- **Hotfixes** landed today:
  - **Ship 16 latent bug**: `sculpt_init` writes a gym_sb3-flavored `[adapter].config = { env_id = "CHANGE_ME", ... }` template — fixed by `_inherit_parent_adapter_config` after `sculpt_init` succeeds.
  - **Datasheet name-mapping**: backend `extract_datasheet_pdf` accepts MJCF actuator names + Claude maps datasheet entries to existing rows.
  - **Auto-open MissionDetailDialog**: clicking Decompose now jumps straight into the live event stream — no more 30-90 s of staring at a row badge.
  - **Per-stage iter ribbon** in MissionDetailDialog: `iter 0 0.45 → iter 1 0.62 → iter 2 training…` chips appear under each stage card while training runs.

Total wall-clock: **~15-20 min** on RTX 5070 Laptop. mjlab Cartpole is the only fast-enough mjlab task — every test step uses it.

---

## 0. Robot + task choice

**mjlab Cartpole**, task_id `Mjlab-Cartpole-Balance` (or whatever the robot library entry surfaces; the UI auto-fills).

Why: Cartpole runs ~30 s/cycle on your RTX 5070; G1/Go1 cycles are 20+ min and would make this an overnight test. Both Ship 19's "publish skill" and "warm-start from skill" paths require **two missions on the same `task_id`** so the compatibility key matches.

---

## 1. Pre-flight

Wipe any previous test state and confirm the test suites pass clean on this branch:

```bash
wsl bash <<'EOF'
export PATH="/home/samjd/.local/bin:/home/samjd/.local/share/pnpm:$PATH"

# Confirm we're on the ship-19 branch (latest hotfixes landed here):
cd /home/samjd/projects && git status -sb && git log --oneline -3

# Wipe prior skill-library state so test starts clean:
rm -rf ~/.local/share/sculptor/skills/

# Sanity-check both suites:
cd /home/samjd/projects/RewardSculptor && uv run pytest tests/ -q 2>&1 | tail -3
cd /home/samjd/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q -k 'not test_reward_prompt_edit_emits' 2>&1 | tail -3
EOF
```

**Expect:** branch `ship-19-skill-library`. Sculptor `340 passed, 1 skipped`. Backend `298 passed, 1 deselected`.

---

## 2. Launch the UI

```bash
wsl bash -c "cd /home/samjd/projects/reward-sculptor-ui && ./run.sh"
```

**Expect:** browser opens at `http://localhost:5173`; both backend (`:8000`) and Vite dev server up; no orphan-port warnings.

---

## 3. Create the test project (one-time)

In the UI:

1. **+ New project**.
2. **Display name:** `ship-test`.
3. **Adapter:** `mjlab` (default).
4. **Library robot:** scroll the menagerie grid → click **Cartpole**.
5. **Task:** dropdown auto-fills with the Cartpole task_id. Leave as-is.
6. **num_envs:** `1024`. **device:** `cuda:0`. Submit.

**Expect:**
- Redirect to `/projects/ship-test`.
- Status badge `ready`. Adapter chip `mjlab`. Task fact band shows the Cartpole task_id.

### 3a. Verify the project's config.toml is mjlab-shaped

```bash
wsl bash -c "cat ~/.local/share/reward-sculptor/projects/ship-test/config.toml | head -10"
```

**Expect** (key lines):
```toml
[adapter]
class = "sculptor.adapters.mjlab.MjlabAdapter"
config = { task_id = "Mjlab-Cartpole-Balance", num_envs = 1024, device = "cuda:0" }
```

`task_id` (not `env_id`), `num_envs` (not `n_envs`). If you see the gym shape, the project_store has its own config-write bug and the rest of the test will fail — file it before continuing.

---

## 4. Datasheet upload — verify name mapping fix

This exercises the §A hotfix (datasheet entries map to existing MJCF actuator names).

### 4a. Open Physics tab and inspect actuator names

Click **Physics** tab. The **MJCF summary** panel lists the project's actuators. For Cartpole there's typically one actuator (e.g., `slider`). For a humanoid you'd see ~20.

Open **Motor specs** card (it has a "Motor specs (structured form)" title). The table has one row per existing actuator, all empty.

### 4b. Upload a datasheet

You need a real PDF datasheet for any motor. Options:
- Maxon EC60 / EC90 PDFs from their website.
- A Tmotor / U motor datasheet you have on disk.
- Any motor PDF with extractable text (NOT a scanned image — pypdf needs real text).

Click the **upload** icon next to "Motor specs", select your PDF.

**Expect:**
- Toast: `Extracted N motor spec(s)`.
- The motor-specs card opens automatically.
- **THE FIX**: when the existing MJCF has actuator names that match (e.g., your datasheet says "Hip Pitch" and your MJCF has `left_hip_pitch_actuator`), Claude is now told to use the EXACT MJCF name as the JSON key. The extracted data lands in the EXISTING actuator's row.
- For a Cartpole project (one actuator), there's usually no plausible match — Claude returns `unmapped_*` keys. That's the fallback path; the `unmapped_` prefix tells you to rename or delete those rows manually.

### 4c. Confirm via inspection

Open browser DevTools → Network tab → find the `POST /api/projects/ship-test/physics/datasheet-pdf` request → request body should include a `actuator_names` form field listing your project's actuator names.

If `actuator_names` is missing, the frontend isn't passing them — bug in PhysicsTab.tsx wiring.

---

## 5. Mission 1 — exercises all of Ship 18b + Ship 19's "publish" path

### 5a. Open Missions tab

Click **Missions** tab.

**Expect:** empty state with Sparkles icon + "No missions yet."

### 5b. Decompose a goal

Click **New mission**.

- **Goal** (paste verbatim):
  ```
  Balance the cartpole at the upright pose, keeping the pole within 12 degrees of vertical and the cart within 0.5 m of origin for at least 200 steps.
  ```
- Leave Mission slug blank.
- Leave `--no-kg` unchecked.
- Click **Decompose**.

**Expect (NEW Ship-19c behavior — auto-open):**
- Toast: `Decompose job queued`.
- The NewMissionDialog closes.
- **Immediately**, the MissionDetailDialog opens for the new mission, showing:
  - Title with the goal text and a `running` badge (amber, pulsing).
  - "Decomposing — Claude is building the curriculum" amber banner with a spinner.
  - Live events panel below: `connecting` → `live` (emerald) within ~1 s; `connected` and decompose stdout start streaming into the log panel.

This is the §B fix. **Pre-fix you would have stared at a row pulsing in the missions list with no other feedback for 30-90 s.**

### 5c. Watch the decompose complete

After 30-90 s:

- Live event stream produces `mission_decompose_completed` (default-grey chip — color is queued for Ship 19b UI sweep) followed by `terminal`.
- WS chip flips to `ended`.
- Banner replaced by a stage list — N stages, each with a status chip (`pending`), name, goal_text, success_criterion in mono, parent_stage indented.
- Decomposition rationale block appears at the top.
- Lifecycle badge in the title flips from `running` to `ready` (emerald).

If the WS shows `disconnected — refresh to retry`, the decompose subprocess errored. Inspect the per-job log file:

```bash
wsl bash -c "ls -la ~/.local/share/reward-sculptor/projects/ship-test/.missions/*/_decompose_*.log 2>/dev/null && tail -30 ~/.local/share/reward-sculptor/projects/ship-test/.missions/*/_decompose_*.log"
```

### 5d. Run the mission

Click **Run mission** in the dialog footer.

**Expect (immediately):**
- Toast: `Mission run queued`.
- WS status chip flips back to `live`.
- Active job badge becomes `mission_execute`.
- The `Run mission` button becomes disabled.

**Expect over the next 1-3 min** (1 stage × N iters × ~30 s on Cartpole):

In the structured-events panel (newest at top):
1. `mission_started`
2. `stage_started`
3. `stage_warm_start_resolved` (Ship 16; default-grey)
4. `stage_warm_start_chosen` (NEW, Ship 19) with `source: "none"` (no prior skills)
5. `stage_scaffolded` with `inherited_parent_adapter_config: true` ← **this is the §0 hotfix verification**. If this is `false` or missing, the env_id bug is back.
6. `stage_v1_materialized`
7. `iter_started` events from the inner sculpt_run subprocess
8. `iter_completed` events with primary_metric values

**NEW (§C iter ribbon):** under the stage card, look for a row of small colored chips:
- `iter 0 training…` (amber, pulsing) → flips to `iter 0 0.45` (emerald) when iter completes
- `iter 1 training…` → `iter 1 0.62` → `iter 2 training…` etc.

This is the per-stage iter timeline. The ribbon updates in real time as `iter_started`/`iter_completed` events stream in. **Pre-fix you'd have seen the structured-events list scroll but no compact "this is iter N, metric M" view.**

In the log panel: stdout from the mjlab subprocess scrolls.

### 5e. Stage completes successfully

After the last iter's criterion evaluates true:
- `stage_criterion_evaluated` with `satisfied: true`
- `stage_succeeded` (emerald)
- **`stage_skill_published`** (default-grey, Ship 19's load-bearing event) with the skill_id, final_metric, source_iter_index in description text
- `mission_completed`
- `terminal` → WS chip `ended`

If you see `stage_skill_publish_skipped` instead, read the `reason`:
- `redecomposition_artifact` — Ship 17 sub-stage; expected if the original stage failed and was re-decomposed.
- `adapter_does_not_support_warm_start` — should NOT fire on mjlab; if it does, file a bug.
- `no_metric_history` — sculpt_run returned 0 iters; training failed silently.

### 5f. Verify skill landed

```bash
wsl bash -c "ls -la ~/.local/share/sculptor/skills/ && echo --- && cat ~/.local/share/sculptor/skills/*/metadata.json | head -25"
```

**Expect:**
- One directory like `<12 hex>/` containing `metadata.json` + `checkpoint.pt`.
- Plus `.locks/`.
- `metadata.json`:
  - `"adapter_class": "sculptor.adapters.mjlab.MjlabAdapter"`
  - `"task_id": "Mjlab-Cartpole-Balance"` (matches the project)
  - `"source_mission_goal": "Balance the cartpole at the upright pose, ..."`
  - `"source_iter_index": <0..N-1>` — the BEST-iter index, NOT necessarily the last
  - `"final_metric": <float>` — best across iters
  - `"checkpoint_sha256": <64 hex>` (Ship 19 audit-fix C4: full file hash, not first 8 KB)

---

## 6. Mission 2 — Ship 19's "reuse" path

Still on the same project. Click **New mission** again.

- **Goal:**
  ```
  Balance the cartpole upright while also keeping the cart velocity near zero throughout the episode.
  ```
- Submit.

**Expect (auto-open + decompose with library context):**
- Detail dialog opens immediately, shows the "Decomposing" banner.
- Decompose completes in 30-90 s.
- The **decompose's user content sent to Claude now includes a SKILL_LIBRARY block** listing the skill_id from Mission 1 with its `criterion`, `seed_prompt`, and `final_metric`.

You can verify this by tailing the backend's stdout (in the terminal running `./run.sh`) — look for the heartbeat events from Ship 10:
```
[reward-prompt] KG preview done (...)
[decompose] ...
```

### 6a. Inspect Mission 2's stages for `init_skill_id`

```bash
wsl bash -c "cat ~/.local/share/reward-sculptor/projects/ship-test/.missions/*/mission.json | python3 -m json.tool | grep -A1 init_skill_id"
```

**Two valid outcomes:**
- **A (preferred):** at least one stage has `"init_skill_id": "<12 hex>"` matching the published skill's id. Claude picked up the suggestion.
- **B:** all stages have `"init_skill_id": null`. Claude was offered the library but decided retraining was simpler. Still a valid Ship 19 outcome.

### 6b. Run Mission 2 — confirm warm-start fires (only if Outcome A)

Click **Run mission**.

**Expect (only if Outcome A):**
- For the stage with `init_skill_id` set:
  - `stage_warm_start_chosen` event with `source: "skill_library"` and `source_id: "<skill_id>"`.
  - If the stage ALSO has `parent_stage` set: an additional `warm_start_skipped(reason: "skill_overrides_parent")` event (Ship 19 audit-fix C1 — explicit beats implicit).
  - The `init_policy_path` threaded into the mjlab subprocess points at `~/.local/share/sculptor/skills/<skill_id>/checkpoint.pt`.

**Expect (regardless of outcome):**
- After successful stage completion, a SECOND `stage_skill_published` event with a NEW skill_id (different bytes than Mission 1).
- `~/.local/share/sculptor/skills/` now has 2 directories.

```bash
wsl bash -c "ls ~/.local/share/sculptor/skills/ | grep -v '^\\.'"
```

---

## 7. Negative path: `--no-skill-library` opts out

Quick CLI sanity check (escape hatch from the UI-only stance, but verifies the flag wiring):

```bash
wsl bash <<'EOF'
export PATH="/home/samjd/.local/bin:$PATH"
cd /home/samjd/projects/RewardSculptor

PROJECT=~/.local/share/reward-sculptor/projects/ship-test
BEFORE=$(ls ~/.local/share/sculptor/skills/ 2>/dev/null | grep -v '^\.' | wc -l)
echo "library entries before: $BEFORE"

# This decomposes a NEW mission with the library disabled. Check the
# user-content sent to Claude — it should NOT include a SKILL_LIBRARY
# block. (You can confirm via the backend log if you tail it.)
uv run sculpt mission-init "$PROJECT" \
    --goal "Hold the cartpole upright in a steady stance for the full episode." \
    --no-skill-library

ls "$PROJECT"/.missions/
EOF
```

**Expect:** new mission created. The decompose flow ran without a SKILL_LIBRARY block. If you `mission-run` it with `--no-skill-library` too, no `stage_skill_published` events fire and the library count stays unchanged.

---

## 8. Cleanup

```bash
wsl bash <<'EOF'
rm -rf ~/.local/share/reward-sculptor/projects/ship-test/
rm -rf ~/.local/share/sculptor/skills/
echo "cleaned"
EOF
```

---

## 9. What "passing" looks like — checklist

Tick all that you observe. **Items in bold are the new fixes from today's UX session:**

### Ship 18b (Mission UI)
- [ ] Missions tab visible between Runs and Reports.
- [ ] `New mission` dialog: validates goal length (try `<8` chars → toast errors), submits, closes.
- [ ] **NEW: clicking Decompose immediately opens MissionDetailDialog with a "Decomposing" amber banner + spinner.**
- [ ] **NEW: live decompose stdout streams into the log panel during the 30-90 s wait.**
- [ ] After decompose completes, banner is replaced by stage list with parent_stage indentation.
- [ ] Click `Run mission` → WS status chip flips connecting→live.
- [ ] Structured events stream in real time.
- [ ] **NEW: per-stage iter ribbon under each stage card (`iter 0 training…` → `iter 0 0.45` → `iter 1 training…`).**
- [ ] `stage_succeeded` event fires after the last iter's criterion passes.

### Ship 19 (skill library)
- [ ] **NEW (§0 hotfix): `stage_scaffolded` event payload has `inherited_parent_adapter_config: true`.**
- [ ] `stage_skill_published` event fires after stage success.
- [ ] `~/.local/share/sculptor/skills/<id>/{metadata.json, checkpoint.pt}` lands.
- [ ] metadata's `task_id` matches the project's task_id (Cartpole's, not `env_id` placeholder).
- [ ] Mission 2's mission.json either has `init_skill_id: "<id>"` (A) or all-null (B); both valid.
- [ ] If A: `stage_warm_start_chosen(source="skill_library")` fires; `init_policy_path` points into the library.
- [ ] `--no-skill-library` flag suppresses both rendering and publish.

### Datasheet name-mapping (today's fix)
- [ ] Datasheet upload: Network panel shows `actuator_names` form field in the request body.
- [ ] Extracted entries that match an MJCF actuator land in that actuator's existing row.
- [ ] Unmatchable entries land in `unmapped_*` rows so the user notices.

If any item fails AND the failure isn't audit-acceptable (e.g., Claude chose Outcome B in §6a — that's fine), capture the event sequence + skip-reason from the WS event panel + the per-job log file at `<mission_dir>/_execute_*.log` or `<mission_dir>/_decompose_*.log` and ping me.
