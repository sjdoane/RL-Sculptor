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

Five committed slices address the failure rather than reinterpreting it:

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
- `0822e07` closes the contact-supervision gap exposed by the first aligned
  rollout. Command targets now move to embodiment-safe subtargets inside the
  same authored waypoint disks using the selected robot's declared reach
  geometry and obstacle bounding volume. Every authored forbidden-contact
  pair also installs a direct penalty from its compiled simulator contact
  sensor, so fixed objects no longer appear "contact-clean" merely because
  their velocity is identically zero.

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
- Current promoted tuple: selection v23, reward v7 + env v4
- Objective metric: `gen_003`, accepted prompt-native and observe-only
- Historical iter 13: retained as failure provenance; not exportable evidence
- Iter 14: first physically aligned diagnostic rollout; not accepted because
  all 64 evaluation environments touched at least one box
- Fresh proof: required after commit `63dbc28`

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

## Latest physical diagnosis

Iter 14 is the first trustworthy physical-course result:

- physical-scene audit: `aligned`, maximum error `0.00 m`;
- actual ordered waypoint disks and finish entered: 64/64;
- rendered env 0 visibly alternated around all four physical boxes and held
  still for 113 frames;
- zero forbidden contact: 0/64;
- no sustained fall: 37/64;
- full conjunction: 0/64.

The contact timestamps line up with the rendered obstacle passes. The old
generated `box_disturbance` term inferred contact from object velocity, which
cannot work for fixed boxes. Treat iter 14 as diagnosed evidence that the
frame/RSI correction works, not as a successful demonstration.

Iter 15 validates the direct sensor penalty and clearance subtargets:

- physical-scene audit remains perfectly aligned at `0.00 m`;
- contact-free environments improved from 0/64 to 34/64;
- 62/64 completed the physical route and finish;
- 51/64 met the no-sustained-fall proxy;
- six environments passed the full route/contact/fall/whole-body-hold
  conjunction.

The rendered environment still brushed box 1 for three frames and held only
51 uninterrupted frames. Its trajectory showed why: the shifted safe target
was correct, but the velocity command reused the broad 0.35 m task-predicate
radius and switched toward the next waypoint before reaching the safe side.
Commit `63dbc28` keeps the task predicate frozen at 0.35 m but tightens the
command-only transition radius to 0.14 m whenever clearance subtargets exist.

Iter 16 shows that the tighter target fixed contact but exposed an approach
controller mismatch:

- physical-scene audit remained aligned at `0.00 m`;
- contact-free environments improved again, from 34/64 to **58/64**;
- only 1/64 completed the actual route and finish;
- rendered env 0 was contact-free but stopped after waypoint 3;
- no environment produced a valid 100-frame terminal hold.

The command still carried a 0.28 m/s intermediate speed floor into a 0.14 m
transition radius, so it could overshoot the tight safe subtarget. The
corrected runtime keeps the existing 0.35x floor for ordinary routes and uses
a 0.10x floor only for clearance-adjusted routes. The automatic iter 16
diagnosis also found that promoted reward v7 pays terminal settle income
before ordered completion. UI-authored reward v15 already fixes that exploit
by gating settle income on the reward-visible route-completion proxy and
tightening the finish crossfade.

## Recommended corrected proof run

After the clearance approach-speed correction, choose **Resume** with exact
promoted tuple recovery disabled so the run consumes the already-generated
reward v15 instead of restoring promoted reward v7:

- Mode: Auto
- Outer sculpt cycles: 1
- PPO iterations per cycle: 750
- Environments: 1,024
- Device: `cuda:0`
- Episode steps: 1,000
- Rollout episodes: 2
- Seed: 42
- Video: 1920×1080
- Objective metric: `gen_003`
- Fitness mode: observe
- Resume exact promoted tuple: **off**

Before iteration 0, the Training log must show:

- four embodiment-clearance waypoint adjustments;
- clearance transition radius `0.140 m`, while the task predicate remains
  `0.350 m`;
- clearance intermediate approach speed floor `0.10x` cruise;
- four compiled forbidden-contact sensors;
- forbidden-contact supervision at weight `-8`;
- warm start from iter 16;
- selected reward v15 (route-gated terminal settle), not reward v7;
- physical object placement at local pose + environment origin.

Current proof run: UI job `job_4ff8e2081df13d11`, iter 17, clean launch
commit `5b0c834`, selection v26, tuple
`0a7bc62b2fb1a1134de3a1a02b70e072d71e7a2aba9eb02d036327d6fdf868e6`.
All startup checks above passed and actor+critic loaded from iter 16
checkpoint SHA8 `56d1d91a`. Let this worker finish before auditing or editing
runtime code.

Iter 17 finished and must be presented only as diagnostic evidence. Its scene
audit is aligned and 61/64 environments were contact-free, but only 3/64
completed the physical route and no environment passed the 100-frame hold.
Rendered env 0 stopped beside waypoint 3.

The trajectory proved why: the robot was inside the immutable 0.35 m
waypoint disk and only 2 cm short of the safe steering point, but the command
was still waiting for its separate 0.14 m target ball. The corrected
controller now advances through the authored disk's obstacle-safe outer cap:
inside the frozen disk plus across the typed clearance half-space. It retains
0.025 m of transition slack within the existing 0.05 m clearance margin and
restores the ordinary 0.35x crossing-speed floor. There is no second success
predicate and no robot/task-name keying.

For the next proof run, keep exact recovery **off** and verify:

- reward v16 (route-gate sharpening + dense waypoint capture);
- env v15 (entropy scale 1.0);
- `safe-cap transition inside frozen 0.350 m task predicate`;
- `0.025 m clearance slack`;
- warm start from iter 17;
- every previously required physical alignment, contact, RSI, and command
  supervision line.

Current proof run: UI job `job_99e23f1f888a44a5`, iter 18, clean launch
commit `07ec6dc`, selection v27, tuple
`ca96cf73ebc66aecbd227e62d8c0217bf061492173ed2b82e86fd21d489f5c82`.
The log confirms reward v16, env v15, safe-cap progression, entropy 0.01,
and actor+critic warm start from iter 17 checkpoint SHA8 `91fa1b84`.

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
- Forbidden-contact channels are false throughout the selected rollout.
- Selected video passes every acceptance item.
- Results export is enabled only for valid evidence.
- Keep the verified project open before the call.

## Current authoritative correction after iter 18

Iter 18 is diagnostic evidence only. Its physical-scene audit is aligned and
53/64 environments were contact-free, but **0/64** completed the actual route
and finish. The rendered robot cleared the first three boxes, then parked
between boxes 3 and 4. The cause was not rendering: the immutable raw-disk
objective had advanced while the command still waited for a stricter safe cap,
so reward and command requested different waypoints.

Commit `b883447` replaces that post-success cap with a generic two-phase entry:

1. approach a finite-width stage outside the authored disk on its incoming,
   obstacle-safe side;
2. target and enter the original authored disk, with command and evaluator
   advancing on the same frame.

The stage is derived from typed robot reach, object bounds, route tangent, and
authored tolerance. Route RSI uses the same stages. An early disk entry also
synchronizes immediately, so the immutable objective can never remain ahead of
the command. The evaluator, metric firewall, contact sensors, task predicate,
and atomic tuple rules are unchanged.

Launch the next proof through **New run** with:

- exact promoted-tuple recovery: **off**;
- reward: v17 (speed-qualified capture + finish-qualified settle);
- environment: v15;
- warm start: iter 18 checkpoint SHA8 `ffe80ac9`;
- Auto, one outer cycle, 750 PPO iterations, 1,024 environments, `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- objective `gen_003`, observe-only; no reference-motion prior for this
  continuation.

Before iteration 0, confirm the log reports outside-stage then frozen-disk
waypoint progression, the unchanged 0.350 m authored predicate, four direct
forbidden-contact channels at weight `-8`, 50/50 entrance/midroute RSI,
physical object local pose + environment origin, entropy coefficient 0.01,
and actor+critic warm start from iter 18. Acceptance remains the full checklist
above; iter 18 must never be promoted as showcase success.
