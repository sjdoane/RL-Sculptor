# Showcase run — Platform Ascent

A brand-new project, start to finish, touching every subsystem: world
authoring → novel-motion composition → per-mode reward authoring →
training → replay → report. About **80 minutes**, most of it unattended.

## The task

> Run in along flat ground, bound up onto a three-platform ascending
> course, cross the gaps without dropping in, and finish balanced on the
> top platform.

It was chosen because no part of it can be faked:

| What it forces | Why it's the interesting case |
|---|---|
| **A motion no clip contains** | The library has running, one-leg jumps, beam balance and turns — never in one recording. The reference has to be *composed* from four solved clips. That is the whole research direction: solve a novel motion out of pre-existing solved data. |
| **Four distinct phases** | approach / bound / balance / settle become four OGMP-inspired fixed phase windows with separately-authored reward functions. A single posture term cannot cover all four. This is not OGMP's online oracle or predicate-driven mode executor. |
| **A course wider than mjlab's env grid** | The compiled course spans **5.02 × 1.20 m** against mjlab's 2.0 m default `env_spacing`. Every authored world before the fix had its 1024 parallel copies overlapping three deep. You will watch the reconciler print the correction. |
| **Platforms at three heights** | Tops at **0.231 / 0.305 / 0.380 m**. Route-RSI resets have to place the robot *on* the surface it picked, not at flat-ground height. That bug put the G1 shin-deep in solid box. |
| **Train-only randomization** | The author emits two variations — middle platform height `U(0.18, 0.41) m`, second gap length `U(0.31, 0.57) m` — applied to training and never to evaluation. |

---

## Step 0 — start the UI

```bash
cd ~/projects/reward-sculptor-ui && ./run.sh
```

Opens <http://localhost:5173>. Ctrl+C stops both servers.

## Step 1 — new project (1 min)

**Projects → New project → Robot Library → Unitree G1** (Humanoid,
`unitree_g1`, marked READY TO TRAIN). Adapter **mjlab**. Name it:

```
Platform Ascent
```

The project lands on the Overview tab with a **Getting started**
checklist: configure robot → author world → shape reward → train →
export. That checklist is the spine of everything below.

## Step 2 — author the world (~3 min)

**World tab → Author world.** Paste into **Environment prompt**:

```
A narrow obstacle course of raised platforms in a straight line. The robot runs in, hops up onto the first platform, crosses each platform in turn without stepping off the side, and drops back to the ground past the last one.
```

**Draft world.** It returns in about a second — this author is
deterministic, not an LLM call, and it grounds on the vocabulary in the
prompt (`obstacle course`, `platforms`). Rewording into something it
doesn't recognise gives you a bare finish-zone with no course at all,
so keep those words.

You now get the **World builder**: five pages of clarification
questions, each with three choices and a disclosed system default.

- Pages 1–3: per-element `height_m` / `length_m` / `width_m` for the
  three platforms and `depth_m` / `length_m` for the two gaps.
- Page 4: the **train variations** — the randomization ranges.
- Page 5: goal success — `hold_s` and whether waypoints are `ordered`.

**Take the defaults.** They produce the geometry in the table above.
Click **Preview scene** to compile without promoting: expect
`course_elements 5`, breakdown `platform 3 / gap 2`, terrain `plane`,
and all four gates green (schema, capability, budget, build). Then
**Apply & promote**.

Check the World tab afterwards — the 3D viewer shows three ascending
platforms, and the goal is `waypoint_sequence` with `waypoints: auto`
and predicate `sequence_complete`.

## Step 3 — compose the novel motion (~4 min)

**Training tab → New run → Pre-existing motion → Choose motion →
`Compose novel`.**

**Motion name:**

```
platform ascent
```

The dialog derives `clip id: platform-ascent--g1` under the field.

Now four phases. Each row has a label, a clip search, and optional
start/end seconds. Click **Add phase** twice — it starts with two.

| # | Label | Search for | Pick this clip | Start | End |
|---|---|---|---|---|---|
| 1 | `approach` | `run 03` | `run03_poses_100_jpos` | — | — |
| 2 | `bound` | `one leg jump` | `50004_one_leg_jump_poses_60_jpos` | `1.0` | `2.5` |
| 3 | `balance` | `balance on beam 03` | `balance_on_beam03_poses_100_jpos` | `1.0` | `4.0` |
| 4 | `settle` | `left turn 04` | `leftturn04_poses_100_jpos` | `1.0` | `3.0` |

Leave phase 1's start/end blank to take the whole 1.02 s clip.

**Compose.** Expect a result panel reporting:

- **692 frames · 6.92 s · 4 sources**
- three seams, at 0.82 s / 2.12 s / 4.92 s
- worst seam **0.085 rad** max joint jump (4.8°), root-height jump 2.3 mm

Those seam numbers are the point. `strict` composition refuses spans
that don't meet; these pass with room to spare because each source ends
and begins near a comparable pose. The result registers at **tier K,
`certified: false`** — kinematically real frames, but momentum is not
conserved across a seam, so it is a *candidate*, not a validated motion.

## Step 4 — author one reward per mode (~10 min, 4 LLM calls)

**Rewards tab → Per-mode reward** (bottom card). Search:

```
platform ascent
```

Pick `platform-ascent--g1`. The card should read **4 modes at 100 fps**,
because the mode graph is derived from the composition seams, not
guessed:

| mode | window | from |
|---|---|---|
| `approach` | 0.00 – 0.82 s | `run03_poses_100_jpos` |
| `bound` | 0.82 – 2.12 s | `50004_one_leg_jump_poses_60_jpos` |
| `balance` | 2.12 – 4.92 s | `balance_on_beam03_poses_100_jpos` |
| `settle` | 4.92 – 6.92 s | `leftturn04_poses_100_jpos` |

The graph records diagnostic transitions at the immutable composition seam
times. Runtime dispatch is fixed elapsed time: it does not evaluate transition
predicates, retime the clip, or query OGMP's receding-horizon oracle. The UI
uses **OGMP-inspired** deliberately so this showcase cannot overstate the
paper alignment.

1. **Scaffold reward** — instant, no LLM. Writes the dispatch and the
   four windows, leaves four `_mode_*` bodies as stubs.
2. **Author** on each mode in turn — a real Claude call each, 1–3 min.
3. All four `Authored` → **Use for training**. The version list at the
   top of the tab should gain a `v1 SCULPTOR` entry.

**Deliberately break it:** hit **Use for training** after authoring only
two modes. It must refuse and name the unauthored ones — an empty stub
pays zero reward for its slice of every episode, which would silently
train a reward that is blank for 40 % of the motion.

**Then judge it.** Open the authored source. Do `_mode_approach` and
`_mode_balance` actually reward different things — forward velocity
versus centre-of-mass containment — or are they four rewrites of the
same posture term? That is the open research question and I can't
answer it for you.

## Step 5 — pipeline check (~4 min, no LLM spend)

**Training tab → New run → RUN PLAN → Pipeline check.**

**Behavior goal:**

```
Run in along the ground, bound up onto the ascending platform course, cross every platform without dropping into the gaps, and finish standing balanced on the top platform.
```

**Choose motion → `platform-ascent--g1`** so the composed clip becomes
the tracking base.

**Launch.** The `~50 s` label is calibrated on cartpole; G1 with an
authored world is 3–5 min.

**Event log → filter `Log lines`.** These are the lines worth reading:

```
env grid pitch for authored scene: env_spacing 2→6.023 m (course footprint 5.02 x 1.20 m about the origin; …)
constraint budget for authored scene: njmax 300→1536
event:world_route_state_initialization→50% entrance / 25% collision-local interior / 25% terminal-approach starts
```

The first is the fix for the interpenetrating boxes, computed live from
*your* course footprint. The third is the reset that used to drop the
robot through its own platforms. Zero `nefc overflow` lines.

**Overview tab → `Replay`:**

- **one** course of three ascending platforms
- the robot on the ground and on the platform tops — not inside them
- the iteration chip reads **`iter 0`** — a project with nothing on disk
  starts its numbering at 0, not 1
- a short clip, **1–2 s**

That last point is expected. Clip length is *actual episode length* ×
0.02 s, capped at `max_episode_steps`; a 100-iteration policy falls
almost immediately. Anything under 1.0 s gets an amber `truncated`
label — no label means the episode really was that short.

## Step 6 — the real run (~45 min, live LLM)

Same dialog, **Live rehearsal**: 2 outer cycles, 350 rsl_rl iters each,
512 envs, real LLM, 300-step episodes, pauses for feedback. Re-paste the
goal and re-attach the motion prior — the dialog remembers neither.

Per cycle: train → rollout renders → live clip → diagnose → reward edit
proposed → **pause**. The pause is an amber *"Awaiting your feedback ·
after iteration 1"* card with **Continue** / **Continue + go Auto**, not
a hang.

Feedback worth pasting after cycle 1 if the robot never leaves the
ground:

```
The robot is not reaching the first platform — it runs in place or falls before the bound phase. Weight the approach mode's forward progress much harder, and make the bound mode's reward depend on gaining height over the platform edge rather than on posture.
```

## Step 7 — take the results (~2 min)

- **Results tab → Build report** — markdown evidence bundle for the run.
- **Results tab → export** the policy checkpoint (deployment bundle,
  the sim-to-real hand-off).
- **Physics tab** — the MJCF the run actually used, with a prompt box
  that has Claude edit it, and motor limits extractable from a
  datasheet PDF.
- **Knowledge tab** — the papers that grounded the world author. The
  authored world carries its own grounding list (`technique:…`,
  `paper:…`) in `world_v1.json`.

## Optional — mission decomposition instead of a single run

**Training tab → New mission** takes one goal and decomposes it into an
auto-curriculum of stages, each with its own trust-gated objective
metric. Try:

```
Teach the G1 to complete the ascending platform course: first reach the first platform reliably, then chain all three, then finish balanced on top.
```

Decompose is an LLM call plus roughly 1–2 min per stage metric, and
running the mission is several stage runs back to back — budget hours,
not minutes. Worth it only after Step 6 looks right.

---

## What I verified before writing this

Run against the live backend on 2026-07-27, on a throwaway project:

- **World authoring** — this exact prompt, defaults taken: 3 platforms +
  2 gaps, tops 0.231 / 0.305 / 0.380 m, footprint 5.02 × 1.20 m, all
  four admission gates green, `waypoint_sequence` goal, two train-only
  variations. `env_spacing` reconciles 2.0 → 6.023 m.
- **Composition** — these exact four spans: 692 frames, 6.92 s, seams
  0.085 / 0.025 / 0.019 rad, `strict=True` accepted, tier K uncertified.
- **Mode graph** — four modes with the windows in Step 4's table and
  phase-guarded transitions.
- **Training** — a pipeline-check run launched through the same API the
  UI uses, on the world authored above: `status: completed`, zero
  errors, zero `nefc overflow`, and these adjustments in the log —
  `env_spacing 2→6.023 m`, `njmax 300→1536`,
  `reset_base→aligned with course +X`, route RSI installed,
  `command:twist→goal-conditioned waypoint traversal with terminal
  braking`, `horizon-aware cruise 0.800 m/s for 4.423 m command path`,
  and DR on body mass / joint damping / joint armature. The rendered
  rollout is 1.04 s (a 53-step episode) and shows one clean course with
  the robot standing on the ground.

What I did **not** run: Step 4's four Author calls and Step 6's live
rehearsal on this specific project — those are the LLM-spend steps and
they're yours to judge. The per-mode path is verified on another
project; the numbers above are what to compare against.
