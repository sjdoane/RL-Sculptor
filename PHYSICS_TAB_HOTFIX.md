# Physics tab hotfix — bootstrap for a new Claude Code window

Read this end-to-end before doing anything. It's self-contained: user
profile, system state, architecture, the exact bug Sam reported, the
diagnostic commands to confirm it, a prioritised fix plan, and the
verification gate. The M7 milestone (0-7) is complete; this document
is a follow-up hotfix focused on the Physics tab (M7 Phase 5).

**Companion docs to read in order of priority:**
1. `~/projects/CONTEXT.md` — session history. The **most recent
   entries** describe M7 Phase 5 (physics tab original scope) and
   M7 Phase 6-7 (polish). The two most recent Phase 5 entries explain
   why the Physics tab is "prompt-first, form deferred".
2. `~/projects/M7_PLAN.md` — the original M7 plan. You don't need to
   execute it; M7 is done. But §"MuJoCo physics editor (deferred;
   design complete)" is the Phase 5 design doc.
3. `~/projects/MJLAB_PIVOT_DESIGN.md` — M0 architecture reference.
4. This file — the hotfix plan.

---

## Who you are helping

Sam Doane — USC ME undergrad, AME 456 capstone on a **quadruped
jumping robot with series-elastic actuators (SEAs)**. WSL2 Ubuntu
24.04 on an RTX 5070 Laptop (8 GiB VRAM, sm_120, CUDA 13.0). Python
3.13.5 via uv. `ANTHROPIC_API_KEY` already set in
`~/projects/RewardSculptor/.env` (auto-loaded by `sculptor/__init__.py`).

**Working style** (memory will reinforce):
- Terse. File:line references + metrics over prose.
- Flag gotchas early; state the trade-off when you make a judgment call.
- Confirm before destructive actions.
- No features beyond what the task requires.
- Log every meaningful change to `CONTEXT.md` incrementally.
- No emojis unless asked.

---

## Stack + test health (as of this hotfix)

```
~/projects/
├── AME456/                     # capstone project (NOT modified by sculptor)
├── RewardSculptor/             # library; has pre-extracted kg_preextracted.db (46 papers, 824 KB)
└── reward-sculptor-ui/         # FastAPI + React control panel
```

**Baseline test count:** 188 backend + 89 sculptor + 2 GPU (cached) =
**279 passing**. Frontend typecheck clean. Do NOT regress this.

**Relevant M7 Phase 5 deliverables** (present in code, partly broken):
- [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py)
  with `load_mjcf`, `materialize_mjcf`, `apply_prompt_edit`,
  `_summarize_mjcf`, `_parse_claude_output`.
- [backend/routes/physics.py](reward-sculptor-ui/backend/routes/physics.py)
  with `GET /projects/{slug}/physics`,
  `POST /projects/{slug}/physics/prompt`, and a 501-stubbed
  `PUT /projects/{slug}/physics/fields`.
- [frontend/src/components/PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx)
  — two-column layout: prompt textarea + read-only summary pane.
- Prompt: [sculptor/prompts/physics_editor.md](RewardSculptor/sculptor/prompts/physics_editor.md).

---

## What Sam reported (verbatim symptoms)

From the UI screenshot on the `unitree-go1` project's Physics tab:

1. **Edit rejected** toast: `mujoco rejected the new MJCF: ValueError:
   Error: Error opening file 'assets/hip.stl'`.
2. **Read-only summary pane shows all zeros**:
   - `GRAVITY —`, `INTEGRATOR —`, `SOLVER —` (all dashes).
   - `Joints (0)` with `(none)`.
   - `Actuators (0)` with `(none)`.
   - "Simulation options" row empty.
3. Sam's intent: prompt Claude for physics changes AND see the
   current parameters correctly summarised. Both are broken.

---

## Root cause (single bug, two symptoms)

**Confirmed:** [`backend/services/physics.py::materialize_mjcf`](reward-sculptor-ui/backend/services/physics.py)
copies ONLY the XML file, not the sibling `assets/` directory that
holds the meshes (STL / OBJ) + textures the MJCF references.

Evidence (run these to confirm):

```bash
wsl bash -c 'ls -la /home/samjd/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/'
# Expected: ONLY base.xml. No assets/ sibling. This is the bug.

wsl bash -c 'head -3 /home/samjd/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/base.xml'
# Expected line 2: <compiler angle="radian" meshdir="assets" autolimits="true"/>
# meshdir="assets" → mujoco looks for meshes at <xml_parent>/assets/ → missing → ValueError.
```

**Chain of symptoms from the single root:**

- MJCF has `<compiler meshdir="assets"/>` + `<mesh file="hip.stl"/>`.
- `shutil.copy2(path, local)` in `materialize_mjcf` copies one file.
- `assets/` directory from
  `robot_descriptions/go1_mj_description/assets/` is never copied.
- When mujoco parses the local XML: cannot open `assets/hip.stl` →
  `ValueError`. This hits:
  1. **`_summarize_mjcf`** catches the exception and returns a
     default-empty `MjcfSummary()` (silent failure) → UI shows
     `(none)` / dashes. [physics.py:~153](reward-sculptor-ui/backend/services/physics.py).
  2. **`apply_prompt_edit`** validator (line 410+): `mujoco.MjModel.
     from_xml_string(new_xml)` also fails because Claude's output
     references the same missing meshes → edit rejected.

Once the XML is materialized WITHOUT assets, every future load +
every edit attempt fails. The project is effectively stuck.

**Why Phase 5 didn't catch this:** my tests used a minimal hand-crafted
`<mujoco>...</mujoco>` string with no mesh references. The menagerie
MJCFs are realistic; mesh-chain materialization never got exercised.

---

## Secondary bugs surfaced by the diagnosis

These are all symptoms of the same root; fix them as part of the
primary patch:

1. **`_summarize_mjcf` swallows errors silently.** Users see "0
   joints, 0 actuators" with no hint of why. Should return a
   `parse_error: str` so the UI can show an actionable warning.
2. **`apply_prompt_edit` validator rejection OVERWRITES `new_xml`
   with `old_xml`** (physics.py:415). User can't inspect what
   Claude tried. Keep Claude's output on rejection so a "View
   what Claude wrote" panel can render it.
3. **UI toast is the only surface for rejection.** Sam gets a
   160-char snippet in a fade-out toast, not an inline panel where
   he can read the full reason + see Claude's attempt. PhysicsTab
   should render an inline error card below the prompt.

---

## Fix plan (ordered, ships green before the next)

### Phase H1 — Fix materialize_mjcf to bring assets (~1 hr)

Goal: materializing a library MJCF results in a fully self-contained
copy under `<project>/uploads/robot/` that mujoco can parse without
hitting missing meshes.

- [H1.1] Rewrite [`materialize_mjcf`](reward-sculptor-ui/backend/services/physics.py)
  to copy:
  1. The XML file itself → `<project>/uploads/robot/base.xml`.
  2. **Every sibling subdir referenced by `<compiler meshdir=...>`,
     `<compiler texturedir=...>`, or the defaults** (`assets` +
     `meshes` + `textures`) if present under the source XML's parent
     directory. Use `shutil.copytree(dirs_exist_ok=True)` for
     idempotence.
  3. **Every file explicitly referenced** via
     `<mesh file="...">`, `<hfield file="...">`, `<texture file="...">`,
     and `<include file="...">` whose resolved source path is outside
     the copied subdirs. Parse the XML with `xml.etree.ElementTree`
     (already stdlib) — NO new deps.
  4. **Resolve each relative reference** against the same meshdir /
     texturedir logic mujoco uses: if the path is absolute or contains
     `..` bail out with a warning (don't traverse outside the source
     package).
- [H1.2] Add a small helper `_copy_referenced_assets(xml_path,
  dst_dir)` in physics.py; add unit tests for both the happy path
  (assets dir copied) and edge cases (explicit file reference
  outside `assets/`; include chain; missing source; absolute paths
  rejected).
- [H1.3] **Migration path for existing broken projects**: add an
  endpoint `POST /projects/{slug}/physics/remateriailize` that
  deletes `<project>/uploads/robot/` and re-runs materialize from
  the library source. UI surfaces it as a "Re-materialize MJCF"
  button in the error banner from H3.
- [H1.4] Tests:
  - `test_materialize_copies_assets_dir` — asserts
    `<project>/uploads/robot/assets/*.stl` exists after first edit.
  - `test_materialize_copies_referenced_mesh_outside_assets_dir` —
    crafts a tiny MJCF that references a mesh via explicit path.
  - `test_materialize_idempotent_on_second_call` — second call
    doesn't clobber user edits to XML but does top up missing assets.
  - `test_rematerializing_wipes_local_and_recopies`.

**Verify H1:** Sam re-opens the Physics tab on the unitree-go1
project → clicks Re-materialize → `ls
~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/`
shows `base.xml` + `assets/` with the STL files. Refresh the tab →
summary pane populates with actual joint + actuator counts.

### Phase H2 — Surface parse errors in the summary (~30 min)

- [H2.1] Extend `MjcfSummary` (pydantic + dataclass) with
  `parse_error: Optional[str] = None`. [physics.py:~30](reward-sculptor-ui/backend/services/physics.py).
- [H2.2] `_summarize_mjcf` catches mujoco parse errors and populates
  `parse_error = f"{type(e).__name__}: {e}"` instead of silently
  returning empty fields. Return early after setting it.
- [H2.3] Frontend: [`PhysicsTab.tsx`](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx)
  checks `phys.data.summary.parse_error` and renders an amber warning
  banner above the Monaco view: "MJCF won't parse:
  `<parse_error>`. Click **Re-materialize** to restore missing
  assets." The banner's Re-materialize button calls the H1.3
  endpoint.
- [H2.4] Tests:
  - `test_summary_populates_parse_error_when_mesh_missing` —
    materialize WITHOUT assets, load, assert `parse_error` is set
    and `joints == []`.
  - Frontend test: typecheck clean.

### Phase H3 — Preserve Claude's rejected XML + inline error card (~1 hr)

- [H3.1] [`apply_prompt_edit`](reward-sculptor-ui/backend/services/physics.py)
  — on ALL rejection paths (parse failure, REJECTED: summary,
  validator reject), keep `new_xml = claude_output_xml` (not
  `old_xml`). Diff_lines still reflects old → proposed so the user
  can see what Claude wanted to change.
- [H3.2] Extend the return dict with `rejected_at: Literal["parse" |
  "claude_rejected" | "mujoco_validate" | null]` so the UI can route
  to the right remediation hint.
- [H3.3] Frontend: `PhysicsTab.tsx` renders an inline `RejectionCard`
  below the prompt textarea when the last job's result has
  `committed=False`. Card contents:
  - Rejection reason (full text).
  - `rejected_at` tag.
  - "View what Claude wrote" disclosure → MonacoDiffLazy showing
    old vs Claude's output (side-by-side, read-only).
  - "Retry with a different prompt" button that focuses the textarea.
- [H3.4] Toast simplifies to a brief "Edit rejected — see card below"
  pointing at the inline panel.
- [H3.5] Tests:
  - `test_apply_prompt_edit_keeps_claude_output_on_validator_reject`.
  - `test_apply_prompt_edit_rejected_at_tagged_correctly`.
  - Frontend: typecheck clean + manual smoke.

### Phase H4 — Form-based direct field editing (optional; ~3 hrs)

The M7 plan explicitly deferred form-based editing to a follow-up.
After H1-H3 land, this is the natural time to promote it:

- [H4.1] Implement `PUT /projects/{slug}/physics/fields` (currently
  stubbed 501 at [routes/physics.py:~220](reward-sculptor-ui/backend/routes/physics.py)).
  Body: `{timestep?, gravity_z?, joint_damping: {<name>: float}?,
  actuator_forcerange: {<name>: [min, max]}?, ...}`. Uses
  `mujoco.MjSpec` to parse, mutate, regenerate → same validate +
  commit pipeline as the prompt path.
- [H4.2] Hard bounds per the physics_editor prompt (timestep ∈
  [1e-4, 0.1], damping ≥ 0, etc.). 422 on violation.
- [H4.3] PhysicsTab right column changes from read-only digest to
  inline-editable fields (shadcn Input with debounce). Each field
  has a tooltip with the current value + a unit label.
- [H4.4] Tests (~6 covering: valid edit commits, bounds violations,
  unknown joint name 404, round-trip preserves unrelated fields).

**Priority call:** do H1-H3 first since they unblock the current
surface. H4 is pure enhancement, gate on Sam's ask.

---

## Verification gate per phase

Every phase ships with all four green:

1. Backend tests: `cd ~/projects/reward-sculptor-ui && uv run
   pytest backend/tests/ -q`. H1 target ≥ 193, H2 ≥ 195, H3 ≥ 198,
   H4 ≥ 204.
2. Sculptor tests: `cd ~/projects/RewardSculptor && uv run pytest
   tests/ -q --ignore=tests/test_mjlab_gpu.py` — 89 passed, 1
   skipped (unchanged baseline).
3. Frontend typecheck: `cd ~/projects/reward-sculptor-ui/frontend
   && PATH=$HOME/.local/share/pnpm:$PATH node_modules/.bin/tsc
   --noEmit` — empty output.
4. Manual smoke after H1 lands:
   - Delete `uploads/robot/` on Sam's unitree-go1:
     `rm -rf ~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/`.
   - Restart `./run.sh`.
   - Open Project → Physics tab. Summary populates.
   - Prompt "increase hip damping by 50%" → edit commits.
   - `git log` in the project dir shows a `physics:` commit.

---

## Known gotchas (absorb before editing anything)

These cost me time; saving you from repeating them.

1. **Shell is Git-Bash, not WSL bash.** `$var` expansion leaks
   across the WSL boundary even inside single-quoted `bash -c`.
   Use `wsl bash <<'EOF' ... EOF` heredoc for anything with
   variables / loops / globs. Single-shot no-var commands are fine
   via `wsl bash -c 'literal'`.
2. **`uv` isn't on PATH in the default shell.** Prepend
   `$HOME/.local/bin` in heredocs: `export PATH=$HOME/.local/bin:$PATH`.
3. **Write/Edit tools strip the exec bit** on `*.sh` files (writes
   at 0644). After ANY shell-script edit, follow up with
   `wsl bash -c 'chmod +x <path>'`. Memory entry
   `feedback_exec_bit.md` reinforces this.
4. **Glob over `\\wsl.localhost\…` UNC paths times out.** Use
   `wsl bash -c 'find /home/samjd/…'` for file enumeration.
5. **Backend tests redirect `RS_KG_PATH` per-test** (conftest.py).
   Any new route that reads the KG must use
   `backend.services.kg_store.shared_kg_db_path()` / `project_kg_db_path()`
   so tests stay isolated.
6. **JobManager.submit populates `job._cancel`.** Tests that
   directly inject `Job(...)` into `jm._jobs` must set
   `job._cancel = asyncio.Event()` or the stop endpoint returns
   404.

---

## Critical files (map for the fix)

| Phase | File | Role |
|---|---|---|
| H1 | [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py) (`materialize_mjcf`, new `_copy_referenced_assets`) | asset copy logic |
| H1.3 | [backend/routes/physics.py](reward-sculptor-ui/backend/routes/physics.py) | new remateriailize POST |
| H1 | [backend/tests/test_physics.py](reward-sculptor-ui/backend/tests/test_physics.py) | +4 tests |
| H2 | [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py) `MjcfSummary` + `_summarize_mjcf` | parse_error surfacing |
| H2 | [frontend/src/components/PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) | warning banner + Re-materialize button |
| H2 | [frontend/src/lib/types.ts](reward-sculptor-ui/frontend/src/lib/types.ts) | `MjcfSummary.parse_error` field |
| H3 | [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py) `apply_prompt_edit` | keep Claude XML on reject + `rejected_at` |
| H3 | [frontend/src/components/PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) | inline RejectionCard + "View what Claude wrote" |
| H4 | [backend/routes/physics.py](reward-sculptor-ui/backend/routes/physics.py) | un-stub `PUT /physics/fields` |
| H4 | [backend/services/physics.py](reward-sculptor-ui/backend/services/physics.py) | new `apply_field_edits` via `MjSpec` |
| H4 | [frontend/src/components/PhysicsTab.tsx](reward-sculptor-ui/frontend/src/components/PhysicsTab.tsx) | form inputs + debounced mutation |

Existing utilities to reuse (don't reimplement):
- [`MjcfLoadResult`](reward-sculptor-ui/backend/services/physics.py) — load return shape.
- [`_git_commit`](reward-sculptor-ui/backend/services/physics.py) — git commit helper.
- [`MonacoDiffLazy`](reward-sculptor-ui/frontend/src/components/MonacoLazy.tsx) — side-by-side diff renderer for the "View what Claude wrote" disclosure.
- [`useJob`](reward-sculptor-ui/frontend/src/hooks/useJob.ts) — job-polling hook used by `PhysicsTab` already.

---

## Quick commands

```bash
# Start the UI (Sam's main loop):
cd ~/projects/reward-sculptor-ui && ./run.sh
# Open http://localhost:5173 in Windows browser.

# Backend tests:
cd ~/projects/reward-sculptor-ui && uv run pytest backend/tests/ -q
# Expect: 188 passed baseline → grows with each H phase.

# Sculptor tests (should stay at 89 passed, 1 skipped):
cd ~/projects/RewardSculptor && uv run pytest tests/ -q --ignore=tests/test_mjlab_gpu.py

# Frontend typecheck (node lives in ~/.local/share/pnpm/):
cd ~/projects/reward-sculptor-ui/frontend \
    && PATH=$HOME/.local/share/pnpm:$PATH node_modules/.bin/tsc --noEmit

# Nuke the broken unitree-go1 MJCF so H1 has a clean test bed:
rm -rf ~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/

# Manual verification that materialize-with-assets worked:
wsl bash -c 'ls ~/.local/share/reward-sculptor/projects/unitree-go1/uploads/robot/'
# After H1: base.xml  assets  (at minimum)
```

---

## Starter prompt for the new window

Paste this into a fresh Claude Code session (same WSL repo) after
memory has loaded:

```
Read ~/projects/PHYSICS_TAB_HOTFIX.md end-to-end. Then:

1. Confirm the root cause by running the three diagnostic commands
   from the "Root cause" section.
2. Execute Phase H1 (materialize_mjcf + assets/ copy + tests +
   remateriailize endpoint). Ship green before starting H2.
3. Continue through H2 + H3 in order. H4 only if I greenlight.
4. Log each phase to CONTEXT.md with What/Why/How/Verified.
5. Don't commit — I'll do that myself after smoke-testing in the
   UI.
```

---

*End of plan. Go in phase order. Log as you go. Ask before
destructive changes.*
