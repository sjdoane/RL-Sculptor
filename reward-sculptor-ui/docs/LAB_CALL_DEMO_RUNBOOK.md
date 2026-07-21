# RewardSculptor lab-call demo runbook

> **Critical correction (July 21):** The completed `job_434b10c7d3fd8eb2`
> must not be presented as successful parkour. Its robots and waypoint targets
> were in different environment-origin frames; all recorded waypoint indices
> stayed at zero. The source fix invalidates the old tuple by design. Re-author
> and promote the World in the UI, then train a new run before using parkour as
> evidence.

This is the reliable path for the July 21 research-lab demonstration. After
one startup command, project creation, world authoring, launch, monitoring,
steering, stopping, and resuming are all performed in the UI.

## Night-before run

### 1. Start once

In WSL:

```bash
cd ~/projects/reward-sculptor-ui
./run.sh
```

Keep that terminal and the laptop awake, plugged in, and on AC power. The UI
opens at `http://localhost:5173`. Do not edit configuration files for the demo.

### 2. Check readiness in the UI

Open **Settings** and confirm all three:

1. **Anthropic API** says `Connected`. If it does not, paste the key into the
   owner-only field and choose **Save & activate**. It takes effect immediately;
   no restart is needed.
2. **GPU** shows `GeForce RTX 5070 Laptop GPU`, CUDA available, and both
   `mjlab` and `rsl_rl` ready.
3. **Knowledge graph** shows the shared corpus rather than an empty graph.

The saved API key is stored in the local RewardSculptor data directory with
owner-only permissions. The UI and API only display its masked suffix.

### 3. Use the prepared project

Open **Projects → Lab Call — Authored Parkour**. It is configured as:

- Robot: `Unitree Go1`
- Adapter: `mjlab`
- Task: `Mjlab-Velocity-Rough-Unitree-Go1`
- Device: `cuda:0`

The **World** tab should show `Authoritative world tuple` and `Verified for
launch`. The authored World remains `v1`; the atomic tuple-selection version
may be higher because each promoted reward/environment revision advances its
lineage. The scene is a five-element ordered parkour course with three
ascending platforms and two authored gap intervals. The gaps are spacing
between platforms rather than hidden collision geometry, which the World UI
states explicitly.

If the prepared project is unavailable, create it entirely in the UI with the
values above, then follow the world-authoring recipe below.

### 4. Re-create the authored world if needed

Open **World → Author world** and paste this prompt:

> Traverse a parkour course of ascending boxes with gaps, moving forward steadily without falling.

Leave **Robot capability ID** blank so the project robot is used. Choose
**Draft world**. Keep **System decides** for every clarification, moving through
all five pages. Confirm that all eight gates are green:

- schema
- capability
- budget
- build
- initial penetration
- settle
- placement
- reachability

Choose **Preview scene**, inspect the course, then **Apply & promote**. Do not
launch until the World page says `Verified for launch`. Promotion is an atomic
world/task/reward/evaluation tuple; launch rechecks that same tuple server-side.

### 5. Launch the complete showcase run

From **World**, choose **Train this world**. Use these exact settings:

- Run plan: **Overnight showcase**
- Behavior goal:

  > Trot forward steadily across the authored ascending-platform course, clear each gap, and remain upright without falling.

- Mode: `Auto` (the preset turns off pause-for-feedback)
- Sculpt iterations: `4`
- rsl_rl iterations per cycle: `750`
- num_envs override: `1024`
- Device: `cuda:0`
- Episode steps: `500`
- Rollout episodes: `2`
- Seed: `42`
- Video resolution: `960×540`
- Objective fitness metric: `go1_trot`
- Fitness mode: `steer`
- Fitness patience: `4`
- Knowledge graph: enabled; do not select the ablation toggle

The readiness rail must show API key configured, `cuda:0 · mjlab + rsl_rl
ready`, and the authored world tuple verified. The estimated time should be
roughly 2 hours 20 minutes on the demo laptop. The completed rehearsal took
2 hours 21 minutes 34 seconds. Choose **Launch** once and keep the laptop awake
and on AC power.

`go1_trot` is the dependable showcase metric: it scores forward locomotion and
stability while the authored task and environment enforce the ordered course.
The generated-metric path is valuable for a separate experiment, but it adds
LLM-generation and calibration variance that is unnecessary for the live call.

### 6. Monitor without a terminal

The UI switches to **Training** after launch. Watch for this sequence:

1. run accepted with an authored-world selection and launch manifest;
2. reward generation and validation;
3. GPU training for the current outer iteration;
4. rollout capture and video;
5. objective-fitness score, diagnosis, and reward/environment edit;
6. the next iteration starts automatically.

The run header can switch between **Auto** and **Manual**, stop the process, or
resume a previously interrupted run. Completed iteration artifacts are reused
only when the existing checkpoint, rollout, and trajectory match the run.

## Call-day presentation flow

Use this three-minute narrative:

1. **World tab:** show the natural-language prompt, selected robot, materialized
   3D scene, passing gates, train variations, and immutable selection lineage.
2. **Training tab:** show the run plan/readiness rail, live GPU state, iteration
   timeline, rollout video, objective fitness, and the diagnosis/edit loop.
3. **Rewards and Results:** compare reward versions, show literature references,
   metric history, and the best rollout. Emphasize that environment and reward
   changes are versioned together rather than being hidden side effects.

If time permits, open **Robot Library** to show that authoring is capability-
driven and not keyed to Go1 or G1. Gym robots, quadrupeds, humanoids, and arm
robots share the same core world-selection and launch contract.

## Invalidated historical rehearsal

The historical run is `job_434b10c7d3fd8eb2` in **Lab Call — Authored
Parkour**. It completed all four requested cycles and then re-evaluated the
selected policy on fresh seed `90001`, but it is not valid task evidence:

- fitness progressed `0.00159 → 0.20701 → 0.26848 → 0.23281`;
- iteration 4 (displayed as the third new cycle in this continued project) was
  selected as the best atomic artifact tuple;
- the fresh held-out score was `0.25805`, close to the selected score of
  `0.26848`;
- the best rollout stayed upright and showed coherent sustained locomotion;
  the fresh replay reproduced that behavior;
- direct trajectory inspection found zero waypoint advancement in every
  recorded environment because geometry and targets were not translated to
  each environment origin. Do not present its fitness as course progress.

Retain the artifacts as failure provenance, but do not show the old Results
card as successful parkour. A replacement run must use a newly promoted World
tuple compiled after the environment-origin fix.

## Recovery guide

- **Out of memory:** stop from the Training header, relaunch, and change
  `num_envs` from `1024` to `512`. The UI marks the plan `custom` and updates
  the estimate.
- **Laptop sleep, power loss, or backend restart:** start with `./run.sh`, open
  the same project, and relaunch the same settings. Resume reuses only exact-
  matching completed artifacts.
- **World verification turns red:** return to World, choose **Verify integrity**,
  and re-author/promote if the tuple really changed. Do not bypass the gate.
- **LLM failure:** keep the project and run artifacts. Verify the key in Settings,
  then relaunch; no file editing is necessary.
- **Generated metric rejected:** for the live call, select built-in `go1_trot`
  and steer. Rejected generated metrics must not be promoted merely to continue.

## Morning checklist

- Keep the laptop on AC power and disable sleep for the call window.
- Start `./run.sh` and verify the dashboard reports no orphaned active job.
- Open the prepared project and confirm the World tuple is verified.
- Confirm at least one completed rollout video plays in Training or Results.
- Do not re-author or relaunch the completed project immediately before the call.
- Keep one known-good result open in a browser tab as the presentation fallback.
