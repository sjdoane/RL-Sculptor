# RewardSculptor research-lab demo runbook

This runbook is current as of 2026-07-24. The workflow is entirely in the UI
after starting the local app.

> **Evidence correction:** iter 13 is not valid physical slalom evidence.
> Its policy followed a route expressed in the robot's local environment
> frame, while the four rendered collision boxes remained in unshifted global
> coordinates. The measured physical-scene error is 9.90 m for every box.
> Results now marks that policy **Invalid evidence — physical scene mismatch**
> and disables export. Do not present the old video as weaving.

## What is fixed

Three committed slices address the failure rather than reinterpreting it:

- `11e664d` places every authored object at
  `nominal local pose + per-environment origin` on every train/eval reset.
  Commands, sensors, metrics, validators, and rendered physics now share one
  frame. A fixed-object alignment invariant blocks waypoint credit if physical
  geometry drifts from the manifest.
- The same slice adds generic route-aware RSI for authored waypoint sequences.
  Training episodes are split between the real course entrance and positions
  just before later waypoints, facing the next target. Evaluation remains
  frozen and always starts from its authored reset.
- `55865b8` adds the fail-closed Results audit that exposed and invalidated the
  historical rollout.
- `3d92602` adds **Pre-existing motion** to **New run**. A selected
  target-robot clip becomes an immutable phase-indexed tracking prior; the
  behavior prompt may author only a bounded task residual around it. The
  physical world and route RSI remain independent curricula.

Reference clips from another embodiment must first be retargeted and
registered in the target robot's library namespace. RewardSculptor's existing
GMR integration records that provenance. The launch path never silently
borrows a same-named clip from another robot.

## One-time startup

In WSL:

```bash
cd ~/projects/reward-sculptor-ui
./run.sh
```

Then use `http://localhost:5173`. Keep the laptop on AC power, awake, and
cooled.

In **Settings**, confirm:

1. Anthropic API is connected.
2. The RTX 5070, CUDA, `mjlab`, and `rsl_rl` are ready.
3. The knowledge graph is populated.

## Prepared project

Open **Projects → G1 Lab Showcase — Weave and Stop**.

- Project slug: `g1-lab-showcase-weave-and-stop`
- Robot/task: Unitree G1 / `Mjlab-Velocity-Flat-Unitree-G1`
- Device: `cuda:0`
- Current promoted pre-fix tuple: selection v22, reward v7 + env v4
- Objective metric: `gen_003`, accepted prompt-native and observe-only
- Historical iter 13: retained as failure provenance; not exportable evidence
- Fresh proof: required after commit `11e664d`

## World prompt

For a clean project, open **World → Author world** and paste:

> On flat high-traction ground, build a clearly visible slalom using four identical bright orange boxes centered along the +X direction at roughly x=2.0, 3.5, 5.0, and 6.5 metres. Each box should be about 0.45 m wide, 0.45 m deep, and 0.75 m tall, with collision enabled. Give the robot a generous alternating path around them using ordered waypoints approximately at (2.0, +0.85), (3.5, -0.85), (5.0, +0.85), and (6.5, -0.85). Add a large contrasting finish zone centered near (8.0, 0.0). The task is to start upright facing +X, run through every waypoint in order without touching a box, enter the finish zone, come to a complete stop, and remain upright and still there for at least 2 seconds. Randomize floor friction mildly and each box lateral position by at most 0.08 m without closing the path. Preserve generous clearances and high-contrast rendering.

Choose **Draft world**, resolve clarifications, preview, and **Apply & promote**.
Before training, require:

- **Verified for launch**;
- one robot at the course entrance;
- four orange collision boxes on the centerline;
- four alternating waypoint disks;
- one finish disk after the course.

The boxes and route disks must be near the robot in the same preview. If they
are visibly detached, do not launch.

## Behavior and objective prompts

In **New run**, use this behavior goal:

> Start upright facing +X. Run a smooth slalom through waypoint_01, waypoint_02, waypoint_03, and waypoint_04 in exact order, alternating around the four orange boxes with zero robot-box contacts. Enter the finish zone, decelerate, then remain upright and still there continuously for at least 2.0 s. Success requires ordered waypoint completion, no forbidden contact or fall, finish entry, and terminal horizontal base speed <0.12 m/s; elapsed time matters only after physical success.

Under **Objective fitness metric**, use accepted `gen_003` in **observe** mode,
or choose **Generate a metric from this goal at launch** on a new project.
Prompt-native validation does not require a stored trajectory.

### Optional motion-guided discovery

Under the behavior goal, choose **Pre-existing motion → Choose motion**.
Search for a target-robot gait, recovery, jump, or other motion prior and
choose **Use motion**.

The selected `robot/clip` is shown before launch. This does not replace the
slalom task:

- the clip supplies pose/velocity/root/orientation tracking;
- the prompt supplies the bounded novel-task residual;
- route RSI supplies real entrance and mid-course starts;
- the frozen evaluator still requires the complete physical task.

For the first corrected slalom proof, leaving motion blank is the lowest-risk
baseline. Use a treadmill/walk clip in a second run to demonstrate
motion-guided discovery without conflating it with the physical-frame fix.

## Recommended corrected proof run

Choose **Live rehearsal** first:

- Mode: Manual
- Outer sculpt cycles: 2
- PPO iterations per cycle: 350
- Environments: 512
- Device: `cuda:0`
- Episode steps: 1,000
- Rollout episodes: 2
- Seed: 42
- Video: 1920×1080
- Objective metric: `gen_003`
- Fitness mode: observe
- Resume exact promoted tuple: on only when intentionally restoring the last
  accepted tuple

Inspect the first official rollout before committing to an overnight run.
The physical-scene audit must say **aligned**. If the route is learning and
the boxes are visibly co-located, launch **Overnight showcase**:

- Mode: Auto
- Outer sculpt cycles: 4
- PPO iterations per cycle: 750
- Environments: 1,024
- Episode steps: 1,000
- Rollout episodes: 2
- Seed: 42
- Video: 1920×1080
- Fitness patience: 4
- Knowledge graph: enabled

## Acceptance checklist

A finished rollout is showcase evidence only if disk artifacts and video agree:

- physical-scene audit is `aligned`;
- all boxes are within the manifest tolerance in the robot's local frame;
- waypoint index reaches 5 through the actual ordered disks;
- authored success is observed;
- every forbidden-contact channel remains false;
- no sustained fall occurs;
- the robot enters and remains inside the finish disk;
- terminal horizontal speed is below 0.12 m/s;
- one uninterrupted 100-frame/two-second post-completion window satisfies the
  hold threshold;
- the robot remains upright and whole-body quiet;
- the video visibly shows the same course interaction.

Reward return, a zig-zag in empty space, or virtual waypoint counts are not
substitutes. If any item fails, present it as a diagnosed research iteration.

## Three-minute call flow

1. **World:** show the prompt, verified tuple, robot, boxes, route disks,
   finish, randomization, and local-frame placement.
2. **New run:** show the prompt-native objective and optional motion-prior
   picker. Explain the separation between motion prior, physical curriculum,
   and frozen evaluator.
3. **Training/Results:** show the atomic tuple, route-RSI/runtime events,
   physical-scene audit, objective subcomponents, realism audit, and only then
   play a rollout that passes the checklist.

The invalid historical clip is useful failure-analysis material: it
demonstrates why RewardSculptor now audits physical scene/task-frame agreement
instead of trusting a plausible video or local-coordinate metric.

## Recovery

- **OOM:** stop in Training and relaunch with 512 environments.
- **Restart/sleep:** restart `./run.sh`; Resume must log
  `resume_warm_start_resolved` and `warm_start_loaded`.
- **Bad automatic draft:** enable **Resume exact promoted tuple**; hash drift
  fails closed before GPU launch.
- **Reference unavailable:** choose a clip in the target robot namespace or
  retarget/register it first; do not bypass the exact-pair check.
- **World audit red:** re-verify/re-author. Never override an invalid evidence
  result.

## Morning checklist

- App running; no orphaned GPU job.
- World says **Verified for launch**.
- Robot, boxes, route, and finish are visibly co-located.
- Corrected rollout physical-scene audit is aligned.
- Selected video passes every acceptance item.
- Results export is enabled only for valid evidence.
- Keep the verified project open before the call.
