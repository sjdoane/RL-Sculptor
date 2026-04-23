# End-to-end verification scenarios (M6)

Five scenarios that exercise the stack end-to-end. Run them after
every material change to the adapter interface, library loader, or
run-launch path. Each has a clear "pass" criterion.

---

## Scenario A — Legacy path (no regression)

**Goal:** the pre-M1 Gymnasium flow still works.

```bash
cd ~/projects/RewardSculptor
uv run sculpt run \
    --config examples/hopper/config.toml \
    "run forward fast without falling" \
    --iterations 3 \
    --dry-run
```

**Pass:** 3 iter_* dirs written under `examples/hopper/runs/`, each
with `diagnosis.json`; CHANGELOG.md gains 3 entries; dry-run completes
in < 60 s wall-clock.

If the CLI is gone or the output structure differs, the dry-run
harness regressed. Check [`sculpt.py`](../sculptor/sculpt.py) + any
recent changes to `GymSB3Adapter`.

---

## Scenario B — mjlab path

**Goal:** library → project → GPU training → checkpoint end-to-end.

**Note.** The spec mentions Unitree Go2; mjlab core v1.3.0 has no Go2
task (the D-guard auto-demotes the Go2 library entry to
`preview_only`). Use **Unitree Go1** — same 18-DoF quadruped class,
registered task, fits 8 GiB VRAM at num_envs=1024.

1. Start the UI: `cd ~/projects/reward-sculptor-ui && ./run.sh`.
2. Open http://localhost:5173 → Library tab.
3. Filter → Category: Quadruped + Training support: mjlab-ready.
4. Click the **Unitree Go1** card → `Create project with this robot`.
5. In the CreateProjectDialog: verify RTX 5070 Laptop is listed, the
   num_envs slider defaults to 2048 (the library recommendation), and
   the VRAM estimate is green (< 70% of free).
6. Set num_envs to **1024** (conservative for this scenario).
7. Submit.

**Backend pass:** HTTP 201; project detail returns `adapter_class =
"sculptor.adapters.mjlab.MjlabAdapter"`, `ready_to_train=true`,
`adapter_config.task_id = "Mjlab-Velocity-Flat-Unitree-Go1"`,
`adapter_config.num_envs = 1024`.

**Live-train pass (optional; budget 15 min):**

```bash
# Drop into the project dir printed in the UI and:
cd ~/.local/share/reward-sculptor/projects/<slug>
uv run --project ~/projects/RewardSculptor \
    sculpt run "run forward fast" \
    --config config.toml \
    --iterations 3
```

`runs/iter_0/checkpoint.pt` should exist after ~90 s per iter (Go1 at
num_envs=1024 on RTX 5070 Laptop).

---

## Scenario C — Coming-soon adapter (Isaac Lab)

**Goal:** coming-soon adapter scaffolds but training is gated.

1. UI → Library → pick any mjlab-ready humanoid (e.g. **Unitree G1**).
2. Click `Create project with this robot`.
3. In the CreateProjectDialog, open the **RL adapter** dropdown.
4. Pick `⏳ Isaac Lab (coming soon)`.

**UI pass:** an amber card appears with the title "Isaac Lab —
scaffolded, not yet implemented", a clickable "Adoption guide" link
pointing at
`https://github.com/sjdoane/RL-Sculptor/blob/main/RewardSculptor/docs/adapters/isaac.md`
(or the local path equivalent), and an "Estimated effort: 4-8 hours"
note. The submit button changes to `Create anyway`.

5. Click `Create anyway`.

**Backend pass:** HTTP 201; `ProjectDetail.adapter_class =
"sculptor.adapters.isaac_lab.IsaacLabAdapter"`;
`adapter_unavailable=true`; `ready_to_train=false`;
`metadata.json::robot_source.adoption_guide_url = "docs/adapters/isaac.md"`.

6. Navigate to the new project's detail page.

**UI pass:** Train button is disabled, tooltip reads "Adapter not yet
implemented — see docs/adapters/isaac.md".

---

## Scenario D — No-GPU path

**Goal:** the entire system degrades gracefully when CUDA is absent.

```bash
# Restart the backend with CUDA hidden:
CUDA_VISIBLE_DEVICES="" uv run uvicorn backend.main:app --port 8000
```

In a second terminal:

```bash
curl http://localhost:8000/system/gpu | jq .cuda_available
# expect: false
```

**UI pass:**
- Settings page GPU panel shows the amber "No NVIDIA GPU detected"
  banner; mjlab/mujoco_warp/rsl_rl dots may still be green (they're
  importable; they just can't reach a device).
- Library tab: mjlab-ready robots still visible but
  CreateProjectDialog's adapter dropdown defaults to `gym_sb3` when
  `cuda_available=false`. Picking `mjlab` shows a red "No CUDA device
  detected" banner in the device dropdown.
- Legacy gymnasium_compatible path (Hopper, Ant, …) still works
  end-to-end.
- Creating an mjlab project from a `mjlab_ready` entry returns HTTP
  412 `/problems/gpu-required` — the CreateProjectDialog surfaces
  this directly.

---

## Scenario E — Custom robot upload

**Goal:** the pre-M1 URDF/MJCF upload flow still works.

1. UI → open any existing project → Robot tab → Upload.
2. Drag a `.xml` MuJoCo model (or the `examples/hopper/` xml from a
   standalone MuJoCo install) into the drop zone.
3. Wait for validation.

**Pass:** the project's preview renderer loads the uploaded model;
`metadata.json::robot_source.kind = "mjcf"`;
`metadata.json::robot_source.model_file` points at
`uploads/robot/<filename>.xml`. The existing contract test
`backend/tests/test_robot.py` covers this end-to-end with mocked
MuJoCo.

---

## Running the scenarios in CI

Scenarios A, C, D, E are CI-friendly (no GPU training required).
Scenario B's live-train step requires `pytest -m gpu` and is gated
behind the GPU runner. Current CI configuration: local-only; see
`.github/workflows/` when wiring up GitHub Actions.
