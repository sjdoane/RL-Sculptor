# RewardSculptor research-lab demo runbook

This is the July 22 end-to-end showcase: author a physical world from one
prompt, generate a prompt-native objective validator without a stored
trajectory, train a Unitree G1 against the immutable world tuple, and inspect
the result entirely in the UI.

> **Evidence rule:** do not call the slalom solved until the selected rollout
> proves all five ordered regions, zero forbidden box contacts, finish entry,
> terminal speed below `0.12 m/s`, and a continuous two-second upright hold.
> Reward return and visually plausible walking are not substitutes.

## Current prepared project

Open **Projects → G1 Lab Showcase — Weave and Stop**.

- Project slug: `g1-lab-showcase-weave-and-stop`
- Robot: `Unitree G1`
- Adapter: `mjlab`
- Task: `Mjlab-Velocity-Flat-Unitree-G1`
- Device: `cuda:0`
- Promoted training tuple: `de07325bab038d29…` (selection v18: reward v7 +
  env v4; the frozen world/task/evaluation half is unchanged)
- Evaluation lineage: `world-58560025c10981814943d42e`
- Objective metric: `gen_003` (accepted, prompt-native, observe-only)
- Latest completed recovery job: `dde47f043fe792ec`, iter 9, clean code
  `e1b5d50`; it consumed reward v7 + env v4 and warm-started iter 8.
- No recovery is active while the timestep-invariant continuity fix is being
  committed.

The iter-9 rollout completed the ordered route and actual finish in 63/64
environments, avoided every forbidden contact in 56/64, and had no sustained
falls. Terminal mean speed was 0.09883 m/s, with 52/64 below 0.12 m/s. Ten
environments satisfy the full literal conjunction, including an uninterrupted
100-frame post-completion hold.

It is still not accepted as the call-ready proof because rendered environment
0, although clean and visibly correct through the weave, achieved 85 rather
than 100 consecutive quiet frames and ended at 0.18832 m/s. Small corrective
steps remain visible in the terminal video. The frozen `gen_003` metric reports
14 completions using its 90%-quiet proxy; the runbook applies the stronger
consecutive criterion.

## One-time startup

In WSL, run:

```bash
cd ~/projects/reward-sculptor-ui
./run.sh
```

Keep that terminal open. Keep the laptop awake, plugged into AC power, and on a
cooling surface. Everything after startup is done at `http://localhost:5173`.

In **Settings**, verify:

1. **Anthropic API** says `Connected`.
2. **GPU** reports the RTX 5070 Laptop GPU, CUDA, `mjlab`, and `rsl_rl` ready.
3. **Knowledge graph** is populated.

If the API key is missing, paste it into the owner-only field and choose
**Save & activate**. No app restart is required.

## World authoring, entirely in the UI

The prepared project is already promoted. For a clean re-creation, make a new
G1 project in the UI, open **World → Author world**, leave the capability ID
blank so the project robot is inherited, and paste this exact prompt:

> On flat high-traction ground, build a clearly visible slalom using four identical bright orange boxes centered along the +X direction at roughly x=2.0, 3.5, 5.0, and 6.5 metres. Each box should be about 0.45 m wide, 0.45 m deep, and 0.75 m tall, with collision enabled. Give the robot a generous alternating path around them using ordered waypoints approximately at (2.0, +0.85), (3.5, -0.85), (5.0, +0.85), and (6.5, -0.85). Add a large contrasting finish zone centered near (8.0, 0.0). The task is to start upright facing +X, run through every waypoint in order without touching a box, enter the finish zone, come to a complete stop, and remain upright and still there for at least 2 seconds. Randomize floor friction mildly and each box lateral position by at most 0.08 m without closing the path. Preserve generous clearances and high-contrast rendering.

Choose **Draft world** and keep **System decides** for any clarification unless
the lab specifically wants to discuss one. Confirm all admission gates are
green: schema, capability, budget, build, initial penetration, settle,
placement, and reachability. Then choose **Preview scene → Apply & promote**.

Before training, the World tab must say **Verified for launch**. In the 3D
scene, confirm:

- one G1 at the start;
- four orange collision boxes on the centerline;
- four green waypoint disks alternating left/right;
- one larger green finish disk after the course.

Click `waypoint_01` and show that the inspector reports
`center_m=[2,0.85]`, `kind=disk`, and `radius_m=0.45`. This is the clearest
visual proof that the natural-language task became executable geometry and
task state, not merely a background image.

## Prompt-native objective metric

Open **New run** and paste this exact behavior goal:

> Start upright facing +X. Run a smooth slalom through waypoint_01, waypoint_02, waypoint_03, and waypoint_04 in exact order, alternating around the four orange boxes with zero robot-box contacts. Enter the finish zone, decelerate, then remain upright and still there continuously for at least 2.0 s. Success requires ordered waypoint completion, no forbidden contact or fall, finish entry, and terminal horizontal base speed <0.12 m/s; elapsed time matters only after physical success.

Under **Objective fitness metric**:

1. choose `best-of-3`;
2. choose **Generate from goal** (or **Generate a metric from this goal at
   launch** on a fresh project);
3. wait for generate → validate → review to finish;
4. use the accepted generated metric.

No stored trajectory is required. The independent abstract objective and the
authored-world channels construct the competent validator fixture. A newly
generated metric stays **observe-only** until it earns steer rights through
empirical calibration; leave it in observe mode rather than weakening that
trust boundary. The prepared project already has accepted metric `gen_003`.

## Next one-cycle recovery

Use **Resume** in the Training tab with the exact behavior goal above, Auto
mode, one sculpt iteration, 750 rsl_rl iterations, 1,024 environments,
`cuda:0`, 1,000 episode steps, two rollout episodes, seed 42, 1920×1080 video,
and `gen_003` observe-only. Enable **Resume exact promoted tuple** to reject
all stale, partition-flagged automatic diagnosis drafts and restore selection
v18 (reward v7 + env v4) before continuing from the preserved iter-9
policy. Before letting the cycle continue, verify the Training log contains
all five facts:

- `promoted_tuple_restored ... selection v18 ... reward v7 ... env v4`
- `resume_warm_start_resolved ... runs/iter_9/checkpoint.pt`
- `warm_start_loaded ... load_cfg_keys=[actor, critic]`
- `preserved authored command supervision at full weight`
- `installed authored terminal continuity-aware whole-body stillness`

If any is absent, use **Stop** and diagnose before spending the GPU budget.

### Latest completed exact-tuple recovery

UI job `dde47f043fe792ec`, iter 9, restored selection v17 and pinned selection
v18 with tuple
`de07325bab038d29fa6705148f795d201d8159c42d93b8ddd92c4ec41f2226db`,
reward v7, and env v4 from clean code commit `e1b5d50`. The worker loaded both
actor and critic from `runs/iter_8/checkpoint.pt` and logged full-strength
linear/angular waypoint-command supervision, terminal braking, entropy
coefficient 0.0075, and continuity-aware terminal whole-body stillness at
weight 1.0, `hold_s=2`, and continuity scale 2.

The strict iter-9 audit is summarized above. Its 85-frame rendered hold is a
near miss, not a success. The next recovery corrects the continuity term's
units: because the reward manager multiplies by `step_dt`, streak-potential
gain and loss are now returned as a per-second rate. That keeps the integrated
interruption penalty invariant across simulator timesteps. Do not consume
automatic reward v10 or env v8; both reward edits were partition-gate flagged
and are not the verified promoted tuple. Before PPO starts, verify the log says
`installed authored terminal continuity-aware whole-body stillness
supervision` with hold 2 seconds and continuity scale 2.

## Exact overnight launch settings

From **World**, choose **Train this world**, or open **New run**. Select
**Overnight showcase**, then expand **Advanced** and use:

- Behavior goal: exact text above
- Mode: `Auto`
- Sculpt iterations: `4`
- rsl_rl iterations per cycle: `750`
- Environments: `1024`
- Device: `cuda:0`
- Episode steps: `1000` (20 simulated seconds)
- Rollout episodes: `2`
- Seed: `42`
- Video: `1920×1080`
- Objective metric: accepted generated metric (`gen_003` in the prepared
  project)
- Fitness mode: `observe`
- Fitness patience: `4`
- Resume: enabled
- Resume exact promoted tuple: leave **off** for a normal iteration; turn
  **on** only when intentionally rejecting unpromoted diagnosis drafts and
  continuing from the last accepted atomic tuple
- Knowledge graph: enabled
- Auto-physics on severe: project default

The readiness rail must show a configured API key, CUDA/MJLab/rsl_rl ready,
and the selected world tuple verified. A four-cycle run is an overnight job;
do not rely on a two-hour estimate for this 1,000-step authored-world setup.

Choose **Launch** once. The app moves to **Training** and streams:

1. the exact world selection and launch manifest;
2. reward generation/validation;
3. GPU training;
4. rollout and 1080p video capture;
5. objective metric and realism audit;
6. diagnosis plus the next atomic reward/world selection.

Do not edit reload-watched core files during an active worker. Use the UI Stop
control before any core change. Resume reuses only exact-matching completed
artifacts; it never pretends a partial or drifted artifact is complete.

## Acceptance checklist for the finished run

In **Training** or **Results**, select the chosen iteration and verify the disk
artifacts and visible UI agree:

- waypoint index reaches `5`;
- authored success is observed;
- ordered waypoint verification passes;
- `contact_frac = 0` and every forbidden-contact channel stays false;
- no sustained fall is detected;
- the robot is inside the finish disk during the terminal window;
- terminal horizontal speed is `< 0.12 m/s`;
- one uninterrupted two-second (100-frame) window stays below that speed;
- the robot remains upright during that hold;
- the rollout video visibly shows the same weave, finish, and stop.

If any item fails, present it as a diagnosed research iteration, not as solved.
The metric's dense progress score is useful for ranking partial policies, but
only the conjunctive completion gate establishes success.

## Three-minute call flow

1. **World (about 60 s).** Show the exact prompt, verified tuple, robot and
   task, visible orange obstacles, alternating green task disks, a selected
   waypoint's parameters, train-only randomization, and immutable lineage.
2. **Training (about 75 s).** Show the one-prompt goal, live/completed GPU run,
   iteration timeline, objective trust status, reward diagnosis, physics
   audit, and atomic environment/reward revisions.
3. **Results + Rewards (about 45 s).** Play the selected rollout, show the
   completion subcomponents and contact/hold evidence, compare reward
   versions, and show literature grounding.

If asked about generality, open **Robot Library**. The implementation is keyed
to capabilities and control surfaces, not G1 or Go1 names: humanoids,
quadrupeds, arms, grippers, and future robots use the same world-selection and
validator contracts when their capability descriptors support the task.

## Honest fallback if training is still running

Do not substitute the old Go1 parkour run or call a partial G1 checkpoint a
success. Instead:

1. show the fully verified World tab and interactive task geometry;
2. show the live ordered-learning evidence in Training;
3. show the validator's honest zero completion plus its progress/contact
   subcomponents;
4. explain the identified control conflict and the generic goal-conditioned
   command fix;
5. state which acceptance gate remains unmet.

That is a stronger research demonstration than a cherry-picked video with an
unsupported success claim.

## Invalidated historical Go1 rehearsal

The historical `job_434b10c7d3fd8eb2` in **Lab Call — Authored Parkour** is
failure provenance only. Its robots and waypoint targets were in different
environment-origin frames, and every recorded waypoint index stayed at zero.
Although it completed four cycles and produced coherent forward locomotion,
its fitness is not evidence of platform traversal. Do not show it as solved.

Likewise, the later short Go1 UI plumbing run proved launch, streaming,
rollout, objective evaluation, and cancellation, but terminated after only
eight rollout steps and did not prove parkour. Keep both artifacts for the
failure-analysis story only.

## Recovery

- **Out of memory:** Stop in Training, relaunch with `512` environments, and
  let the UI update the estimate/plan to custom.
- **Sleep, power loss, or backend restart:** rerun `./run.sh`, open the same
  project, and relaunch the same settings with Resume enabled. Resume searches
  the actual preceding iteration artifacts rather than assuming reward and run
  indices are contiguous, so a UI-authored reward-version gap still loads the
  newest valid learned policy. A completed iteration also has an atomic
  `iteration_complete.json`; Resume advances past only contiguous, valid
  completion markers, while a missing or corrupt marker keeps same-iteration
  crash recovery. The Training log must emit
  `resume_warm_start_resolved` followed by `warm_start_loaded`; stop the run if
  a fresh actor is initialized instead.
- **A run errors before PPO iteration 0:** preserve the failed job, inspect its
  traceback, fix and verify the generic runtime contract, then relaunch the
  identical UI configuration. A pre-PPO configuration failure has not produced
  a new learned checkpoint; Resume must still load the last valid promoted
  actor and critic.
- **Reject a bad automatic diagnosis:** open New run → Advanced and enable
  **Resume exact promoted tuple**. The launch must log
  `promoted_tuple_restored` with the expected selection/tuple hash and
  reward/env versions before training starts. A hash mismatch fails closed
  before the GPU process launches; never repair the immutable artifacts in
  place.
- **World verification turns red:** choose **Verify integrity**; re-author and
  promote only if the tuple genuinely changed. Never bypass the gate.
- **Metric generation fails:** use the inline Retry control. Do not promote a
  rejected metric. For a live fallback, show the already accepted `gen_003`
  as observe-only.
- **LLM key missing:** fix it in Settings and relaunch; no file editing is
  required.

## Morning checklist

- Laptop on AC, sleep disabled, cooling unobstructed.
- `./run.sh` running and Dashboard free of orphaned jobs.
- Prepared project opens and World says **Verified for launch**.
- 3D scene visibly contains boxes, alternating waypoint disks, and finish.
- Browser warning/error console is clean.
- At least one completed video plays from Training or Results.
- Completion claims match the objective subcomponents and trajectory.
- Keep the verified project open in one tab before the call.
- Do not re-author or relaunch immediately before presenting.
