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

Current proof run: UI job `job_0825a00f4219404d`, iter 19, clean launch
commit `cdbec1c`, selection v28, tuple
`b3574e18c09b2fda89467ac50ce0234e6ac744b624b6a7d068b91dd662378c07`.
The log confirms reward v17 SHA `aaaf2a20cf86532a...`, env v15 SHA
`08837c8d2f093bfe...`, all two-phase entry/RSI/contact/alignment invariants,
entropy 0.01, and actor+critic warm start from iter 18 SHA8 `ffe80ac9`.
PPO iteration 0 is active. Let this worker finish before inspecting official
rollout evidence or changing runtime code.

## Current authoritative correction after iter 19

Iter 19 is physically aligned diagnostic evidence, not showcase success. The
video finally shows real interaction with the four rendered boxes, but only
1/64 environments completed the actual ordered route and finish, 44/64 were
contact-free, none completed a 100-frame hold, and rendered env 0 stopped
after waypoint 2. The physical-scene audit remained aligned at `0.00 m`.

The official trajectory localized the failure immediately before the outside
approach stages. During that command-only phase, the base velocity rewards
point toward the safe outside stage while generated reward v17 still points
toward the immutable disk center. Both signals are individually correct in
their own coordinate contracts, but together they create a dense
training-only equilibrium.

Commit `3dfae11` installs a generic clearance-stage reward firewall:

- while the command's typed outside stage is active, only the conflicting
  generated `sculptor_primary` reward and its component diagnostics are
  withheld;
- command tracking, direct contact supervision, survival/failure economics,
  terminal stillness, and native realism priors stay active;
- generated reward returns at full strength for the immutable disk-entry and
  terminal phases;
- no robot, task, object, prompt, or simulator name selects the behavior.

Focused compiler/adapter verification is **67 passed**; Ruff, compileall, and
diff check pass. The automatic reward-v18 edit failed both syntax retries, so
the next UI New run must use reward v17. It may consume newly authored env v16
(entropy scale 1.5), with exact promoted-tuple recovery **off**, and warm-start
the iter 19 checkpoint. Keep the same proof settings: Auto, one cycle, 750 PPO
iterations, 1,024 environments on `cuda:0`, seed 42, two 1,000-step
1920x1080 episodes, and `gen_003` observe-only.

Before PPO iteration 0, require the previous two-phase/RSI/contact/alignment
lines plus:

`installed clearance-stage reward firewall: predicate-centered generated reward withheld during command-only safe approach; command/contact/survival supervision remains active`

All acceptance criteria above remain conjunctive. Never promote iter 19.

Current proof run: UI job `job_86d707964503d576`, iter 20, clean captured
commit `b734809` (firewall `3dfae11`), selection v29, tuple
`1793447534bb34385f3fe43b7cb3ba796582583f6d8d8530f8359c4e51a12710`.
It uses reward v17 SHA `aaaf2a20cf86532a...` and env v16 SHA
`46fa68a262828da90...`. The live log proves actor+critic warm start from iter
19 checkpoint SHA8 `182e00f5`, entropy coefficient 0.015, all prior physical
alignment/RSI/contact/command invariants, and the exact clearance-stage
firewall line. PPO iteration 1/750 is active. Leave the worker alone until
completion, then apply the full acceptance checklist to its official
first-episode-safe evidence.

## Current authoritative correction after iter 20

Iter 20 is aligned diagnostic evidence, not showcase success. The full video
now shows real alternating traversal around the physical orange boxes, and
48/64 environments were contact-free with no sustained falls. Only 3/64
entered the finish after the ordered route, however, and none could satisfy
the uninterrupted 100-frame hold.

This time the official timing—not rendering or reward conflict—is decisive.
Median waypoint-4 entry occurred at 18.56 seconds, and the only three route
completions occurred at 19.50–19.68 seconds. The frozen episode is 20 seconds
and the frozen terminal dwell is two seconds, so the fixed 0.8 m/s cruise
schedule made success impossible even for those completing policies.

Commit `3039879` derives cruise speed generically from the full staged command
path, episode horizon, authored hold, settle reserve, traversal-efficiency
allowance, and installed command-domain cap. It changes no task predicate,
tolerance, hold, evaluator, firewall, or atomic tuple. The exact project now
compiles to:

- staged command path: 12.740 m;
- traversal window: 16.000 s;
- horizon-aware cruise: 1.000 m/s, capped by the command domain.

Focused compiler/adapter verification is **68 passed**; Ruff, compileall, and
diff check pass. Launch the next proof through **New run** with:

- exact promoted-tuple recovery: **off**;
- reward: v17 (reward v18 does not exist because both generated edits failed
  syntax validation);
- environment: v17 (entropy coefficient scale 2.0);
- warm start: iter 20 checkpoint SHA8 `c1dbcce9`;
- Auto, one cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- `gen_003` observe-only and no motion prior for this continuation.

Before PPO iteration 0, require the startup log to report horizon-aware
`1.000 m/s` cruise for the `12.740 m` path and `16.000 s` traversal window,
plus the existing actor+critic warm start, local-frame physical boxes,
two-phase outside-stage→frozen-disk controller, 50/50 route RSI, four direct
contact sensors at `-8`, full command weights, whole-body terminal stillness,
and clearance-stage reward firewall. Keep the full acceptance checklist
conjunctive; never promote iter 20.

Current proof run: UI job `job_ac1eb30cafc3fdee`, iter 21, clean captured
commit `6c541ff`, selection v30 / tuple
`d44cd6a28364eeb08ae086e8739ad220ed09fe0324d3d76ade04607ea4b7d978`.
It uses reward v17 SHA `aaaf2a20cf86532a...` and env v17 SHA
`1d1655a8397c9df9...`. The live log proves every required startup invariant,
including the 1.000 m/s horizon-aware schedule, entropy 0.02, and actor+critic
warm start from iter 20 SHA8 `c1dbcce9`. PPO is active at 1,024 environments.
Leave the worker alone until it finishes; then audit only official artifacts
and the full video against the complete acceptance checklist.

## Current authoritative correction after iter 21

Iter 21 is the first run where the scheduling correction materially solved the
course: **40/64** environments entered all four ordered waypoint predicates
and the finish, the aligned full video visibly weaves around the real orange
boxes, and 18/64 combined route completion with zero forbidden contact.
Physical-scene error remains `0.00 m`. It is still not showcase success:
30/64 were contact-free, box 2 accounted for 30 contact failures, and no
environment produced the required uninterrupted 100-frame hold.

The terminal command explains the hold failure. It retained a 0.35 m/s floor
until finish-disk entry, then jumped to zero; rendered env 0 completed at
17.64 s but continued whole-body corrections and held horizontally quiet for
only 35 frames. Uniform mid-route RSI also devoted only 12.5% of all resets to
the terminal phase.

The corrected generic runtime now:

- brakes with a constant-deceleration square-root profile against distance to
  the immutable finish-predicate boundary;
- crosses that boundary at no more than 0.10 m/s and then commands standing;
- preserves 50% full-route resets and splits the remainder into 25% interior
  recovery plus 25% terminal-approach resets for authored dwell goals;
- records root linear and angular velocity in official trajectories, so the
  2-second whole-body hold can be checked exactly rather than inferred.

No evaluator, task predicate, tolerance, horizon, hold, contact rule, reward
firewall, or tuple invariant changed. Focused compiler/adapter tests are
**68 passed**, with Ruff, compileall, and diff checks clean.

Launch the next proof through normal **Resume** with:

- exact promoted-tuple recovery: **off**;
- reward: v17 (automatic reward v18 failed syntax validation);
- environment: v18 (entropy coefficient scale 1.0);
- warm start: iter 21 checkpoint;
- Auto, one cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- `gen_003` observe-only and no motion prior for this continuation.

Before PPO iteration 0, require the log to report terminal predicate-boundary
braking with entry command `≤0.100 m/s`, 50/25/25 phase-balanced route RSI,
the existing 1.000 m/s horizon-aware cruise, actor+critic warm start, aligned
physical objects, two-phase clearance stages, four direct contact sensors at
`-8`, full velocity-command weights, terminal whole-body stillness, and the
clearance-stage reward firewall. Acceptance remains fully conjunctive; never
present iter 21 as the final result.

Current proof run: UI job `job_94a8897853269309`, iter 22, clean launch
commit `a642572`, selection v31 / tuple
`f9456723bfac3a1911042ad5adac1627ad35beeceecb525d043f2923c34503d8`.
It uses reward v17 SHA `aaaf2a20cf86532a...`, env v18 SHA
`5388a2bb5d094aef...`, and explicitly loads iter 21 checkpoint SHA8
`675b6296`. The UI startup log proves the 1.000 m/s horizon-aware route,
0.100 m/s terminal-boundary entry command, 50/25/25 RSI, physical alignment,
direct contact supervision, full command weights, terminal stillness, and
reward firewall. PPO iteration 1/750 is active. Do not run an intermediate
GPU audit or edit reload-watched core while this worker is alive.

## Current authoritative correction after iter 22

Iter 22 solved most of the physical task across the evaluation batch:
57/64 lanes completed the ordered route, 52/64 avoided every box contact, and
four lanes (`10, 11, 15, 50`) combined route/contact/no-fall with a literal
100-frame whole-body velocity hold. The scene remains aligned at tight error
and the video shows the robot traversing the actual box course. Do not present
iter 22 as final proof: the fixed displayed lane 0 finished while deeply
crouched and flailing, with terminal speed `0.443 m/s`.

The generic runtime now refuses to count a motionless collapse as terminal
stillness. It gates the continuity streak on projected-gravity uprightness and
RMS distance from each robot's own default joint pose, and it exposes
reset-safe `action_rate` plus `joint_vel_rms` channels for generated rewards.
No robot or task name is used.

The Advanced New Run dialog also has an **Evidence environment** field. It is
a transparent, pre-run choice of which parallel evaluation lane the video
tracks; it never changes training, the full 64-lane trajectory, or batch
fitness. `behavior.json` records the requested/resolved index, return, and
percentile. Automatic post-hoc best-lane selection is not implemented.

Verification: core focused suites **140 passed**, backend run suite **48
passed**, frontend TypeScript, scoped Ruff, compileall, and diff check pass.

Launch the next proof entirely in the UI:

- click **New run** (not exact-tuple recovery);
- keep exact promoted-tuple recovery **off** so reward v18 and env v19 are
  consumed;
- Auto, one cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- Evidence environment: **10**;
- `gen_003` observe-only, no new motion prior.

Reward v18 increases posture supervision and caps excessive foot swing; env
v19 reduces entropy to `0.75x`. The run must warm-start actor and critic from
iter 22. Before PPO, verify the existing alignment, staged-route, horizon,
terminal-braking, 50/25/25 RSI, contact `-8`, full command-weight, and reward
firewall lines, plus posture-aware terminal stillness. Acceptance is still
fully conjunctive: the disclosed video lane and official batch evidence must
show the aligned physical boxes, ordered weave and finish, zero forbidden
contact, no sustained fall, upright posture, terminal horizontal speed below
`0.12 m/s`, and 100 uninterrupted post-completion whole-body-quiet frames.

## Iter 28 is aligned, contact-free, and route-incomplete

Iter 28 completed with checkpoint SHA
`8032c012dff39a68b5054f81bf35037bc1a2b691ad466bd3f12f0e126e732edc`.
Its scene audit is aligned at `0.00 m`, realism is `ok`, every one of the 64
official lanes is forbidden-contact-free, and no lane sustains a fall.
Nonetheless, 0/64 complete the ordered actual-disk route, 0/64 reach index 5,
and 0/64 produce any valid post-completion hold. The index distribution is
`{0: 50, 1: 1, 2: 12, 3: 1}`.

Requested/resolved lane 10 (percentile `0.0625`) enters only disks 1 and 2 at
frames `[137, 309]`, advances their `0.350 m` predicates at frames 158 and
573, and ends at index 2, `3.306 m` from finish. The complete video shows the
same partial physical weave and then parking near waypoint 3. This is
diagnostic evidence, not a showcase result.

The measured stall distances expose the remaining geometric issue:
`0.3514 m` and `0.3547 m` versus the immutable `0.3500 m` predicate. A
through-disk target only `0.025 m` inside cannot absorb the policy's ordinary
`0.026–0.030 m` command-tracking lag. The generic controller now preserves
the same obstacle-safe radial clearance and chord but places the target
`0.050 m` inside, at radius `0.300 m`. Disk entry is still the sole route
advancement authority. Focused compiler + adapter suites pass **70 tests**;
scoped Ruff (`F,E9`), compileall, and diff check pass.

For the next New Run, use exact promoted reward v20/env v21 and explicitly
warm-start actor+critic from iter 26, the last checkpoint with real
full-conjunction lanes. Keep Auto, one 750-PPO cycle, 1,024 CUDA environments,
seed 42, two 1,000-step 1920x1080 episodes, lane 10, and `gen_003`
observe-only. Require startup proof of the `0.050 m` inside margin plus all
existing alignment, contact `-8`, full command, 50/25/25 RSI, firewall,
strict posture, horizon, and terminal-braking invariants.

## Live proof: iter 29 robust predicate depth

UI job `job_2ebcb47d3cee200f` is running iter 29 from clean captured commit
`874968e3f644c49488d1440952e42807b7343508`. Selection v38 restores the
exact promoted reward v20/env v21 tuple, and `warm_start_loaded` proves both
actor and critic came from iter 26 checkpoint SHA8 `d5a35ae6`. PPO 0/750 is
active.

The startup record shows the intended generic geometry: each `0.100 m`
outside stage transitions to a clearance-preserving chord whose target is
`0.050 m` inside the unchanged `0.350 m` authored predicate. It also proves
aligned local-frame boxes, four contact sensors at `-8`, full linear/angular
command weights 2, 50/25/25 train-only RSI, the clearance reward firewall,
strict whole-body terminal posture at weight 4, `1.000 m/s` horizon
scheduling, `2.000 m`/`0.050 m/s` terminal braking, and entropy `0.01`.

The run is Auto with one 750-PPO cycle, 1,024 CUDA environments, seed 42, two
1,000-step 1920x1080 episodes, precommitted lane 10, and `gen_003`
observe-only. Leave the worker untouched. After it stops, the lane-10 video
and official batch artifacts must prove every item in the physical acceptance
checklist; no batch partial rate, attractive clip, or prior iteration can
substitute for that conjunction.

Current proof run: UI job `job_0102595ce1cf9e61`, iter 23, clean launch
commit `6ebc857`, selection v32 / tuple
`014c62f4757b1e91d8689afcddd568cd85a8d699778154d22729c8b5a70397fd`.
It uses reward v18 SHA `48c34c86b939d332...`, env v19 SHA
`84c9a5ff58c12ffb...`, and explicitly loads actor plus critic from iter 22
checkpoint SHA8 `e3c665ec`. The UI command precommits evidence environment
**10** and preserves the full 64-lane batch evidence.

The live startup log proves the 1.000 m/s horizon schedule for the 12.740 m
staged path, terminal predicate-boundary braking, 0.268 m typed clearance and
outside approach stages, unchanged 0.350 m frozen disks, 50/25/25 train-only
RSI, local-frame physical boxes, four direct contact sensors at `-8`, both
command weights at `2.0`, the clearance-stage reward firewall, terminal
whole-body stillness at weight `4`, and entropy coefficient `0.0075`. PPO is
active on 1,024 environments. Leave the worker alone until it finishes; then
audit only the official artifacts and the disclosed lane-10 video against the
complete physical acceptance conjunction above.

## Current authoritative correction after iter 23

Iter 23 is not demo evidence. Its physical scene is correctly aligned and
62/64 evaluation lanes avoided every box contact, but **0/64** completed the
route or entered the finish. The disclosed lane-10 video honestly shows the
robot parked beside the first box; it never reached the remainder of the
course. The batch waypoint-index maxima were `{0:48, 1:4, 2:11, 3:1}`.

Reward v18 over-corrected ordinary posture (coefficient `0.15 → 0.30`) and
created a stationary first-waypoint basin. Because `gen_003` is observe-only,
the software must not automatically choose an older checkpoint from its
scores. Commit `62f9a1b` instead adds an explicit, metric-independent
**Warm-start checkpoint** field in **New run → Advanced**:

- enter an iteration number, never a filesystem path;
- the backend accepts only a non-empty checkpoint under this project's exact
  `runs/iter_N` directory and rejects missing, empty, or escaping files;
- the launch log records `warm_start_checkpoint_resolved`, the exact path,
  iteration, and full SHA-256;
- only actor/critic initialization changes. Reward, environment, frozen
  objective, fitness mode, and atomic tuple remain exactly as selected.

Verification for the control: backend run suite **51 passed**, frontend
TypeScript, Ruff, compileall, and diff check pass.

Launch the next proof entirely in the UI:

- open **New run → Advanced**;
- exact promoted-tuple recovery: **off**;
- Warm-start checkpoint: **22**;
- keep current reward v19 and environment v19;
- Auto, one outer cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- Evidence environment: **10**;
- `gen_003` observe-only and no new motion prior.

Before PPO, require `warm_start_checkpoint_resolved` for iter 22 and both
actor and critic load events, then the existing physical alignment, staged
route, horizon-aware cruise, terminal brake, 50/25/25 RSI, contact `-8`, full
command weights, reward firewall, and posture-aware stillness lines. Leave
the GPU worker alone after startup. Final acceptance remains conjunctive:
aligned visible boxes, ordered physical weave and finish, no forbidden
contact or sustained fall, upright/default-like posture, terminal horizontal
speed below `0.12 m/s`, and 100 uninterrupted post-completion frames quiet in
horizontal, angular, joint, and posture channels.

Current proof run: UI job `job_0d2b45c89d3cf056`, iter 24, clean captured
commit `9fda226`. The UI pinned selection v33 / tuple
`2262c85d4d823331ac510211688d81573cf80fd3218c0fa23ee1e5fa0cc5cfa9`,
reward v19 SHA `ee9941bb5cb61778...`, and env v19 SHA
`84c9a5ff58c12ffb...`, with exact promoted recovery off. The new recovery
event records iter 22 checkpoint SHA
`e3c665ecde508fa1ea9f3f12c519b285a5bc8e8116acdb83213fd1a2d8041c21`,
and `warm_start_loaded` confirms both actor and critic. The UI/worker logs
also prove 750 PPO iterations, 1,024 environments, seed 42, lane 10, two
1,000-step 1920×1080 episodes, `gen_003` observe-only, reward-v19 posture
revert, entropy `0.0075`, and every physical controller/contact/firewall
invariant above. PPO is active. Leave the worker untouched until completion.

## Current authoritative correction after iter 24

Iter 24 restored the real task but is still diagnostic, not final evidence.
The physical scene is aligned at `0.00 m`; 63/64 lanes entered every actual
waypoint disk and the finish in order, 49/64 were contact-free, and all 64
avoided a sustained fall. All 15 contact failures were on box 2. The disclosed
lane 10 visibly performs the alternating weave around the real orange boxes,
enters the finish at 15.90 seconds, makes no forbidden contact, and holds
horizontal speed below `0.12 m/s` for 106 uninterrupted frames.

It nevertheless ends in a deep squat with raised arms. Lane 10's terminal
speed channels are quiet (`0.0079 m/s` horizontal, `0.196 rad/s` angular,
`0.304 rad/s` joint RMS), but its default-pose RMS error is `1.003 rad`.
Across the full batch, 18/64 lanes achieved the horizontal hold and **0/64**
achieved the posture-qualified whole-body hold. Never present iter 24 as
success.

The terminal runtime now treats posture as a conjunction rather than a small
bonus. The velocity-based stillness score is multiplied by the geometric mean
of the posture signals generically available on the articulation:
projected-gravity uprightness and RMS distance from its own default joint
pose. This removes the high-reward motionless-collapse basin while preserving
dense gradients, full reward for honest upright stillness, and fail-soft
behavior when a custom/fixed-base robot lacks a posture signal. The change has
no robot/task-name keying and does not alter the frozen objective or world.

Verification: Mjlab adapter suite **52 passed**, scoped Ruff (`F,E9`),
compileall, and diff check pass.

Launch the next proof entirely through **New run -> Advanced**:

- exact promoted-tuple recovery: **off**;
- reward: **v20** (adds a dense base-height gate to settle income);
- environment: **v20** (`0.5x` entropy scale);
- Warm-start checkpoint: **24**;
- one cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920x1080;
- Evidence environment: **10**;
- `gen_003` observe-only, no new motion prior.

Require `warm_start_checkpoint_resolved` for iter 24, actor and critic load,
and the new `multiplicative posture gate` terminal-stillness line before
leaving the worker alone. Also retain the 1.000 m/s horizon schedule,
predicate-boundary brake, two-phase clearance controller, 50/25/25 RSI,
aligned physical boxes, four direct contact sensors at `-8`, full command
weights, and clearance reward firewall. Final acceptance remains the full
physical conjunction; no batch aggregate or visually convincing weave can
substitute for the disclosed lane's posture-qualified 100-frame hold.

## Live proof: iter 25

The run above is active as UI job `job_65bfa68b72389283`, iter 25, from clean
captured commit `8a2d451`. The UI resolved Warm-start checkpoint 24 at full
SHA `c9b0fee59f4898d889d853634535b97e96b0b8830d3d778843e7dc16c4bef238`
and the worker loaded both actor and critic. Iter 25 pins selection v34 /
tuple `4d91af1045ed0afa4290b1a3c0b2284ae8e02394c9dce190bb876f9d44329a02`.
Exact promoted recovery is off; reward v20 and env v20 are authoritative.

The visible startup log confirms:

- terminal stillness uses the multiplicative posture gate at weight 4;
- env-v20 entropy changes `0.01 -> 0.005`;
- all four forbidden-contact sensors remain at `-8`;
- both authored velocity-command terms retain full supervision;
- the safe-stage reward firewall, two-phase 0.268 m clearance entries,
  unchanged 0.350 m task disks, 50/25/25 RSI, aligned boxes, 1.000 m/s
  horizon schedule, and terminal boundary brake remain active;
- PPO iteration 0/750 is running on 1,024 environments with seed 42, evidence
  lane 10, two 1,000-step 1920x1080 episodes, Auto, and `gen_003`
  observe-only.

Leave the live worker untouched. On completion, use Results plus the official
artifacts to verify lane 10 visibly weaves around the co-located boxes,
enters every disk and finish in order, has no forbidden contact or fall, and
holds a default-like upright pose with horizontal, angular, and joint motion
quiet for 100 uninterrupted post-completion frames. Do not substitute a batch
rate or an older video for that disclosed-lane proof.

## Iter 25 is improved diagnostic evidence, not final

Iter 25 completed and its physical scene is aligned at `0.00 m`. The official
trajectory has 60/64 ordered physical route completions, 55/64 contact-free
lanes, 52/64 satisfying both, and no sustained falls. Ten lanes hold
horizontal speed for 100 frames and nine also keep angular and joint motion
quiet, but **0/64** satisfy the default-like posture requirement. Full
conjunction remains **0/64**.

The disclosed lane 10 (requested and resolved, percentile `0.53125`) is
contact-free and enters the actual disks at frames
`[166, 376, 562, 764, 844]`. It achieves 111 uninterrupted velocity-quiet
post-completion frames, ending at `0.105 m/s` horizontal speed,
`0.178 rad/s` angular speed, and `0.329 rad/s` joint RMS. Its terminal pose is
still `0.763 rad` RMS from the robot's own default. The complete video shows
the real weave followed by a deep squat with raised arms. Do not present it
as success.

The next runtime uses a strict smooth posture conjunction. It multiplies the
kinematic stillness score by every available posture factor directly,
instead of taking their geometric mean and thereby diluting a single bad
factor. Missing posture signals remain fail-soft; the objective and physical
task are unchanged. The next UI recovery run must warm-start iter 25, preserve
the current physical controller/contact/firewall invariants, and prove the
same disclosed lane's 100-frame posture-qualified hold before promotion.

## Live proof: iter 26 strict posture conjunction

UI job `job_f61191b6d9080217` is running iter 26 from clean captured commit
`bb8e085`. Exact promoted recovery is off; reward v20 and current environment
v21 are authoritative. The UI resolved iter 25 checkpoint SHA
`d5c1f8552626c2cf4c3e5cffef11edc61ccace0c6564a0fbb433f5ea3702f51b`
and pins selection v35 / tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`.

The launch record confirms:

- env v21 applies `entropy_coef 0.01 -> 0.01`;
- actor and critic loaded from iter 25 (SHA8 `d5c1f855`);
- PPO entered iteration 0/750 on 1,024 environments;
- terminal stillness uses the strict smooth product of every available
  posture factor at weight 4;
- four forbidden-contact sensors remain at -8 and both authored velocity
  command terms retain full supervision;
- the safe-stage firewall, two-phase 0.268 m clearance entries, unchanged
  0.350 m task disks, 50/25/25 RSI, aligned physical boxes, 1.000 m/s horizon
  schedule, and terminal predicate-boundary brake remain active.

The remaining settings are seed 42, evidence lane 10, two 1,000-step
1920x1080 episodes, Auto, and `gen_003` observe-only. Leave the live worker
untouched. On completion, require lane 10—not a batch aggregate or older
video—to visibly weave around the co-located boxes, enter every disk and
finish in order, avoid contact and sustained fall, and maintain a default-like
upright pose with horizontal, angular, and joint motion quiet for 100
uninterrupted post-completion frames inside finish.

## Iter 26 is visually convincing but not yet promotable

Iter 26 completed and preserved checkpoint SHA
`d5a35ae6c0a3f2ca8cc7cc6c5fce076fcb8499cae8e1351c7ed04bd864c54cea`.
The scene audit is aligned at `0.00 m`, every lane enters the five real disks
in order, 62/64 reach index 5, 48/64 are contact-free, and no lane sustains a
fall. The strict posture product is a real improvement: nine lanes achieve
the posture-qualified 100-frame hold and seven satisfy the entire physical
conjunction. Fitness is nevertheless `0.18120` because it aggregates the
frozen 64-lane conjunction; it does not grade only the attractive rendered
lane.

The disclosed lane 10 visibly weaves around the actual boxes and stops in a
much better upright/default-like pose, but it clips box 2 for two frames and
box 3 for one frame. Its root is only `0.396-0.427 m` from those box centers,
showing a physical corner cut. It reaches index 5 at frame `805` and ends
quiet, but its uninterrupted full-body streak is only frames `927-999`:
**73 frames / 1.46 seconds**. Never present iter 26 as final success.

The next generic controller recovery keeps the obstacle-away target after an
outside clearance stage instead of steering back to the unsafe disk center.
The immutable authored disk still advances the route on first entry, so the
fix adds no new objective predicate. Terminal boundary braking also starts
over `2.0 m` and reaches the boundary at at most `0.05 m/s`, providing more
pre-entry settling time. Focused compiler and adapter suites pass
**18 + 52 tests**.

Launch the next proof entirely through **New run → Advanced**:

- exact promoted-tuple recovery: **off**;
- reward/environment: current **v20 / v21**;
- Warm-start checkpoint: **26**;
- one cycle, 750 PPO iterations, 1,024 environments on `cuda:0`;
- 1,000 episode steps, two rollout episodes, seed 42, 1920×1080;
- Evidence environment: **10**;
- Auto, `gen_003` observe-only, no new motion prior.

Before leaving the worker alone, require actor and critic load from iter 26,
the clearance-preserving in-disk target line, `2.0 m` terminal brake with
`0.05 m/s` boundary command, and all existing aligned-scene, physical-box,
contact `-8`, full command-weight, 50/25/25 RSI, strict-posture, and firewall
invariants. Final acceptance remains the disclosed lane's complete physical
conjunction, not visual plausibility or a batch partial-pass count.

## Live proof: iter 27 clearance-preserving recovery

UI job `job_0b8d143e516a7920` is running iter 27 from clean captured commit
`7e39452`. Exact promoted recovery is off; reward v20 and environment v21 are
authoritative. The UI resolved Warm-start checkpoint 26 at full SHA
`d5a35ae6c0a3f2ca8cc7cc6c5fce076fcb8499cae8e1351c7ed04bd864c54cea`
and pins selection v36 / tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`.

The startup record confirms actor+critic recovery and PPO iteration 0/750.
It retains all four typed clearance targets and outside stages under the
controller commit that keeps the safe in-disk target active until the frozen
disk predicate advances. It also confirms the new `2.000 m` terminal brake
and `0.050 m/s` boundary command, 1.000 m/s horizon schedule, 50/25/25 RSI,
aligned physical boxes, contact `-8`, full command weights, strict-product
terminal stillness, entropy `0.01`, and the clearance reward firewall.

Settings are one cycle, 1,024 environments on `cuda:0`, seed 42, two
1,000-step 1920×1080 episodes, evidence lane 10, Auto, and `gen_003`
observe-only. Leave the worker untouched. On completion, require the
disclosed lane—not an older video or batch partial pass—to satisfy the full
aligned weave, zero-contact, ordered-finish, no-fall, upright/default-like,
terminal-speed, and uninterrupted 100-frame whole-body hold conjunction.

## Iter 27 is aligned but route-incomplete

Iter 27 completed and preserved checkpoint SHA
`15d2a8434fe6b9f332760e8a71bb61e261994f41de66650ebcd2fc1f406084b6`.
Its scene audit is aligned at `0.00 m`, realism is `ok`, 63/64 lanes avoid
forbidden contact, and no lane sustains a fall. It is not task evidence:
0/64 complete the ordered route, 0/64 reach index 5, objective fitness is
zero, and the maximum-index distribution is
`{0: 41, 1: 9, 2: 13, 3: 1}`.

The requested and resolved lane 10 (percentile `0.03125`) is contact-free but
enters only waypoint zones 1–3 at frames `[135, 545, 998]`. It advances only
the first two frozen command predicates, never reaches waypoint 4 or finish,
and ends `3.535 m` from finish at index 2 and `0.178 m/s`. The complete video
shows a slow, high-sway partial weave among the boxes, not a weave-and-stop.

The next generic controller correction retains the same typed obstacle-away
radial clearance and outside approach stage, but aims through the authored
disk on the outgoing side of that safe chord. This avoids the measured
near-boundary hover while leaving the unchanged disk predicate as the only
route advancement authority. The command-only target is `0.025 m` inside the
disk; it does not add a success condition. Focused compiler + adapter suites
pass **70 tests**, with scoped Ruff (`F,E9`), compileall, and diff check also
passing.

For the next UI New Run, warm-start iter 27 and require startup proof of the
clearance-preserving through-disk chord together with all prior invariants:
aligned boxes, four contact sensors at `-8`, full velocity-command weights,
50/25/25 train-only RSI, reward firewall, strict whole-body terminal posture,
horizon scheduling, and terminal boundary braking. Final acceptance remains
the disclosed lane's complete physical conjunction.

## Live proof: iter 28 through-disk recovery

UI job `job_e71a1d16d5100f1b` is running iter 28 from clean captured commit
`9195e554b3d15c5ed73c26414d28f22a562eff65`. The UI restored selection v36
before training and iter 28 pins selection v37 with exact promoted tuple
`95afc97b6000593eab01c8e7b374b71dbf20f6e75e647008d322f8270b0b88c7`,
reward v20, and env v21. `warm_start_loaded` proves actor+critic recovery
from iter 27 checkpoint SHA8 `15d2a843`, and PPO 0/750 is active.

The worker startup line must remain visible in the demonstration evidence:
the four typed `0.268 m` obstacle-away entries keep their `0.100 m` outside
stages, then command through each unchanged `0.350 m` authored disk along the
same safe chord with a `0.025 m` inside margin. Disk entry alone advances the
route. All existing safeguards are also live: local-frame box alignment,
four contact sensors at `-8`, full linear/angular command weights 2,
50/25/25 train-only RSI, the clearance-stage reward firewall,
strict-product whole-body terminal stillness at weight 4, `1.000 m/s`
horizon scheduling, `2.000 m`/`0.050 m/s` terminal braking, and entropy
`0.01`.

This is an Auto run with one 750-PPO cycle, 1,024 CUDA environments, seed 42,
two 1,000-step 1920x1080 episodes, precommitted evidence lane 10, and
`gen_003` observe-only. Do not inspect the GPU or change reload-watched core
while the worker is alive. After it stops, require the official lane-10 video
and artifacts to prove scene alignment, visible alternating physical-box
traversal, all actual disks plus finish in order, index 5, zero contact, no
fall, upright/default-like posture, terminal horizontal speed below
`0.12 m/s`, and 100 uninterrupted post-completion whole-body-quiet frames.

## Iter 28 and iter 29 remain diagnostic-only

Iter 28 is aligned and contact-free in all 64 lanes, but no lane completes
the route. The disclosed lane enters only the first two physical waypoint
zones, stops near waypoint 3, and never reaches finish. Its old `0.025 m`
inside target left measured predicate misses of `0.3514 m` and `0.3547 m`
against the immutable `0.3500 m` tolerance.

Iter 29 retested from iter 26 with a `0.050 m` inside target. Its scene is
also exactly aligned and 63/64 lanes are contact-free, but it again produces
0/64 route completions and 0 holds. The maximum-index distribution is
`{0: 41, 1: 1, 2: 21, 4: 1}`. Lane 10 is contact-free, reaches the first
actual zone at frame 121, then parks just outside its frozen predicate at a
minimum distance of `0.3533 m`. The complete video is a first-box approach
followed by stasis, not a weave.

Objective fitness is correctly `0.00000`: success is conjunctive, and no lane
has route completion, finish entry, or the 100-frame hold. The physical-scene
audit remains `aligned` at `0.00 m`; layout is not the failure. Ignore the
generated diagnosis's lane-10 visual-contact inference because all four
official lane-10 contact channels are false.

## Next proof: firewall the complete safe clearance maneuver

The controller has one generic reward conflict left. Predicate-centered
generated reward is withheld while the robot approaches the command-only
outside stage, but was restored immediately after that stage even though the
typed controller was still traversing its obstacle-safe chord into the
immutable disk. The policy can therefore earn contradictory shaping by
parking just outside the predicate.

The corrected per-environment firewall remains active for both approach and
traversal, and ends only when the frozen waypoint predicate advances.
Authored linear/angular command reward, direct contact supervision, survival,
and realism stay active throughout. The disk, tolerance, physical objects,
robot capability, and generated reward are unchanged, and the implementation
contains no robot/task/object-name branch.

Focused compiler + adapter suites pass **70 tests**. Scoped Ruff (`F,E9`),
compileall, and diff check pass. Commit this slice, then launch the next proof
through **New run → Advanced** with exact promoted reward v20/env v21,
actor+critic warm start from iter 26, one 750-PPO cycle, 1,024 CUDA
environments, seed 42, two 1,000-step 1920×1080 episodes, Auto,
`gen_003` observe-only, and a precommitted evidence lane. Final acceptance
remains the complete physical conjunction, never visual plausibility alone.

## Live proof: iter 30 full clearance-maneuver firewall

UI job `job_8d340a6c65c057f4` is running iter 30 from clean commit
`a3d21d4efd2b92edc5a283cd0b18d0e2d2295148`. Selection v39 pins the exact
promoted reward v20/environment v21 tuple. The UI proves actor and critic
loaded from iter 26 checkpoint SHA8 `d5a35ae6`, and the corrected runtime
reports that predicate-centered generated shaping is withheld through both
the command-only safe approach and traversal until each immutable waypoint
advances.

All physical invariants remain installed: the four typed `0.268 m`
obstacle-away entries with `0.100 m` outside stages and safe through-disk
targets, local-frame boxes, four direct contact sensors at weight -8, full
linear/angular command supervision, 50/25/25 train-only RSI, 1.000 m/s
horizon cruise, 2.000 m terminal brake to at most 0.050 m/s, strict-product
whole-body terminal stillness at weight 4, and entropy `0.01`.

This is an Auto run with one 750-PPO cycle, 1,024 CUDA environments, seed 42,
two 1,000-step 1920×1080 episodes, exact promoted recovery on, precommitted
evidence lane 10, and `gen_003` observe-only. Leave the live worker and
reload-watched core untouched. After it stops, require the official scene
audit, all-lane trajectory/fitness, and lane-10 keyframes/full video to prove
the entire aligned, ordered, contact-free, no-fall, upright, terminal-speed,
and uninterrupted 100-frame whole-body-hold conjunction.

## Iter 30 is aligned but parks at waypoint 1

Iter 30 completed and preserved its checkpoint. The scene audit is exactly
aligned, realism is `ok`, and requested/resolved lane 10 is recorded at
percentile `0.796875`. It is diagnostic only: 0/64 lanes complete the route,
52/64 are contact-free, no lane sustains a fall, and no lane achieves any
100-frame terminal hold. The maximum-index distribution is
`{0: 36, 1: 9, 2: 18, 4: 1}`; fitness is correctly zero.

Lane 10 remains contact-free but enters only the first actual waypoint zone,
never advances the frozen waypoint-1 predicate, and ends `6.037 m` from
finish. Its closest approach is `0.3845 m` against the immutable `0.3500 m`
tolerance. The complete video shows the robot reach the safe side of the
first physical box and park there, not traverse the slalom.

The complete clearance-maneuver firewall is active and removes
predicate-centered generated shaping during the maneuver, so the remaining
failure is command geometry rather than scene alignment or lane-10 contact.
An in-disk steering target can still yield a just-outside equilibrium when a
velocity policy trails its command. The next generic controller target
continues along the identical obstacle-safe chord to `0.100 m` beyond the
outgoing disk boundary. It does not change success: the original `0.350 m`
disk still advances the command immediately on first entry.

Focused compiler + adapter suites pass **70 tests**; scoped Ruff (`F,E9`),
compileall, and diff check also pass. Commit the correction before launching
the next UI proof. Recover actor+critic from iter 26, keep exact promoted
reward v20/environment v21, retain the aligned boxes, four direct contact
sensors at `-8`, full velocity-command supervision, 50/25/25 RSI,
full-maneuver firewall, terminal brake, and strict whole-body stillness, and
judge only the disclosed lane's complete physical conjunction.

## Live proof: iter 31 clearance-preserving full-disk traversal

UI job `job_1fae6e454ac140cf` is running iter 31 from clean captured commit
`3150e7a10daa7a9d2154f607ce4ea6f52d921bb9`. Exact promoted recovery restored
selection v39, and iter 31 pins selection v40 with the same hash-verified
reward v20/environment v21 tuple. The worker loaded actor and critic from
iter 26 checkpoint SHA8 `d5a35ae6` and entered PPO iteration 0/750.

The startup line now reports four typed `0.268 m` obstacle-away entries with
their `0.100 m` outside stages and clearance-preserving traversal through
each frozen `0.350 m` disk to a `0.100 m` outgoing margin. It retains the
complete-maneuver generated-reward firewall, all four direct contact sensors
at weight -8, full linear/angular command terms, 50/25/25 train-only RSI,
aligned local-frame boxes, 1.000 m/s horizon scheduling, 2.000 m terminal
braking to at most 0.050 m/s, strict-product whole-body stillness at weight 4,
and entropy `0.01`.

This is an Auto run with one 750-PPO cycle, 1,024 CUDA environments, seed 42,
two 1,000-step 1920×1080 episodes, precommitted evidence lane 10, and
`gen_003` observe-only. Leave the live worker untouched. On completion,
preserve its checkpoint and require the official scene audit, all-lane
trajectory/fitness, and disclosed lane's keyframes/full video to prove every
disk plus finish in order, zero contact, no fall, upright/default-like posture,
terminal horizontal speed below `0.12 m/s`, and 100 uninterrupted
post-completion whole-body-quiet frames.

## Iter 31 result and iter 32 safe-cap recovery

Iter 31 completed and preserved its checkpoint, but it is diagnostic only.
The scene audit is perfectly aligned (`0.0 m` maximum box error), while the
official trajectory has only `1/64` success/index-5 lanes, `63/64`
contact-free lanes, `63/64` no-sustained-fall lanes, and zero 100-frame
whole-body holds. Env 37 was the sole route completion and produced only a
98-frame qualifying quiet streak.

Precommitted lane 10 remained contact-free and upright enough to avoid the
fall gate, but stayed at index 0. It entered only the first actual zone,
stopped at a `0.40619 m` minimum distance from the frozen waypoint-1 center,
and ended `5.7410 m` from finish. The keyframes and full video show approach
to the first maneuver followed by a stationary wide-crouched shuffle between
the first pair of boxes. Do not present this video as a successful weave.

The outgoing steering point was itself outside the raw disk, so a policy could
arc around the predicate and converge on the command without route progress.
The generic recovery commands the embodiment-derived safe radial cap inside
the unchanged disk and applies a `1.0x` cruise floor during the typed
clearance traversal. Convergence therefore implies raw predicate entry, while
the outside stage, obstacle-safe half-space, four contact sensors, reward
firewall, and frozen success definition remain intact.

Verification for the next launch is 70 focused compiler/adapter tests, scoped
Ruff `F,E9`, compileall, and diff check. Launch only through New Run after the
fix commit is clean. Recover actor+critic from iter 26, restore exact promoted
reward v20/environment v21, keep one 750-PPO cycle with 1,024 CUDA
environments, seed 42, two 1,000-step 1920x1080 episodes, Auto,
`gen_003` observe-only, exact promoted recovery on, and precommitted evidence
lane 10. Reapply the complete conjunctive acceptance audit after the worker
stops.

## Live proof: iter 32 full-speed safe-cap traversal

UI job `job_28fa781d092c229e` is running iter 32 from clean captured commit
`697a2ae5bc0c76d9741643a91b5cbf6946363914`. Selection v41 pins the exact
promoted reward v20/environment v21 tuple, and the runner loaded both actor
and critic from iter 26 checkpoint SHA8 `d5a35ae6`.

The startup log reports four typed `0.268 m` obstacle-away caps, `0.100 m`
outside stages, and full-speed traversal to each cap inside the unchanged
`0.350 m` raw disk. The raw disk remains the sole advancement authority.
The full-maneuver reward firewall, four direct contact sensors at -8, full
linear/angular command terms, 50/25/25 RSI, aligned local-frame boxes,
1.000 m/s horizon scheduling, 2.000 m terminal braking to at most
0.050 m/s, strict-product whole-body stillness at weight 4, entropy `0.01`,
and PPO iteration 0/750 are active.

This Auto run uses one 750-PPO cycle, 1,024 CUDA environments, seed 42, two
1,000-step 1920x1080 episodes, precommitted evidence lane 10, and `gen_003`
observe-only. Do not touch reload-watched core or run an intermediate GPU
audit. Once stopped, preserve the checkpoint and require the official scene
audit, all-lane artifacts, and lane-10 keyframes/full video to prove the
entire physical conjunction.

## Iter 32 looks successful but is time-disqualified

Iter 32 completed with a perfectly aligned scene and a real route recovery:
62/64 lanes enter the five actual regions in order, 30/64 reach authored
success/index 5, 56/64 are contact-free, and no lane sustains a fall. The
frozen fit is nevertheless only `0.08365` because the completion gate is
`11/64`, terminal-speed aggregate is `0.19921 m/s`, and **0/64** lanes
provide the separate 100-frame post-success whole-body hold.

Lane 10 is the clearest explanation for a viewer. The full video visibly
shows the robot weave around the real orange boxes and stop in the finish.
The immutable trace agrees on geometry and safety: actual-region entries are
frames `[108, 310, 508, 697, 823]`, index 5 arrives at 842, all four contact
channels stay false, no sustained fall occurs, and final horizontal/angular/
joint speeds are `0.0353 m/s`, `0.1278 rad/s`, and `0.3840 rad/s`. But the
authored 2-second dwell does not declare success until frame 942, and the
strict quiet window starts at 968. Only 31 proof frames remain before the
1,000-step horizon. Never substitute the attractive video for the frozen
conjunction.

The next recovery is generic and timing-only. The former fixed `2.000 m`
terminal brake consumed about two seconds of the same horizon needed for
staged route transitions. Terminal braking is now budgeted from route length,
traversal window, cruise cap, and command-segment count. This staged course
uses a `0.500 m` constant-deceleration span and reaches the unchanged raw
predicate boundary with at most a `0.100 m/s` command, still below the frozen
`0.12 m/s` stop threshold. The raw disk, authored hold, episode length,
scene, contacts, route, and reward authority are unchanged.

Focused compiler/adapter verification is 71 passed; scoped Ruff `F,E9`,
compileall, and diff check pass. After the clean commit, launch only through
New Run, recover actor+critic from iter 32, restore exact promoted reward
v20/environment v21, and retain one 750-PPO cycle, 1,024 CUDA environments,
seed 42, two 1,000-step 1920×1080 episodes, Auto, evidence lane 10,
`gen_003` observe-only, exact promoted recovery, aligned boxes, four contact
sensors at -8, full command weights, 50/25/25 RSI, the full-maneuver reward
firewall, strict-product whole-body stillness weight 4, and entropy `0.01`.

## Live proof: iter 33 horizon-budgeted terminal braking

UI job `job_f3d7e4c744ed6494` is running iter 33 from clean captured commit
`6098ce5cbc10136c1a6c6c4170a2c5442d02d69a`. Exact promoted recovery pins
selection v42 to the hash-verified reward v20/environment v21 tuple. The UI
resolved iter 32 checkpoint SHA8 `34178045`, and the runner confirms
`warm_start_loaded` for both actor and critic.

The startup log proves the generic timing correction is active: the staged
12.740 m route retains its 1.000 m/s cruise, while the terminal brake now
uses the horizon-budgeted 0.500 m span and reaches the unchanged finish
predicate boundary at at most 0.100 m/s. All four typed 0.268 m
obstacle-away entries, 0.100 m outside stages, full-speed in-disk safe caps,
the unchanged 0.350 m raw waypoint predicates, full-maneuver reward firewall,
50/25/25 RSI, aligned local-frame boxes, four direct contact sensors at -8,
full velocity-command supervision, strict-product whole-body stillness at
weight 4, and entropy 0.01 remain installed. PPO iteration 0/750 is active.

This Auto run uses 1,024 CUDA environments, seed 42, two 1,000-step
1920×1080 episodes, exact promoted recovery, precommitted evidence lane 10,
and `gen_003` observe-only. Leave the worker and reload-watched core
untouched. After completion, preserve the checkpoint and require the official
all-lane artifacts, lane-10 keyframes/full video, and Results scene audit to
prove the ordered, aligned, contact-free, no-fall, upright/default-like,
sub-0.12 m/s, uninterrupted 100-frame whole-body-hold conjunction.
