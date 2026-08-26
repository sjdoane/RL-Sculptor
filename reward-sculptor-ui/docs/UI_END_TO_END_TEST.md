# End-to-end UI run — do this yourself

One run, start to finish, through the UI. Two stages: a ~4 min stubbed
check that proves the world is fixed, then a ~45 min real cycle-pair with
live LLM calls. Times measured on this machine (RTX 5070 Laptop).

---

## Step 0 — clear the stale iteration (one terminal command)

`sculpt run` always passes `--resume`, and resume **skips training for any
iteration already on disk**. `tracking-first-ui-verification` has
`iter_1` from before the world fix, so a new run would reuse it and show
you the same broken video. There is no UI toggle for this.

```bash
cd ~/.local/share/reward-sculptor/projects/tracking-first-ui-verification/runs && for d in iter_*; do mv "$d" "_prefix_$d"; done && ls
```

Renamed, not deleted — the leading underscore takes it out of the
`iter_*` glob the API lists, and you can move it back any time. The old
run card stays in the Runs tab; its Replay will now say *no rollout on
disk*, which is correct.

Use **this** project, not a new one. A fresh project has no authored
world, so it trains on flat ground and exercises none of the geometry
that was broken.

## Step 1 — start the UI

```bash
cd ~/projects/reward-sculptor-ui && ./run.sh
```

Opens <http://localhost:5173>. Ctrl+C stops both servers. If it was
already open, hard-reload the page once.

Open **Projects → Tracking First UI Verification**.

---

## Step 2 — pipeline check (~4 min, no LLM spend)

**Runs tab → New run.** Under **RUN PLAN** pick **Pipeline check**.

That sets: 1 outer iteration, 100 rsl_rl iters, 256 envs, `--dry-run`
(every LLM call stubbed), 180-step episodes, 1 rollout episode, 960×540.

**Behavior goal** — the box starts empty; paste this:

```
Run in, launch off one leg, and strike with the trailing leg at the apex, tracking the composed novel-running-jump-kick reference throughout.
```

**Pre-existing motion → Choose motion.** Search:

```
running jump kick
```

Pick **`novel-running-jump-kick--g1`** (top hit, 444 frames @ 120 fps,
3.7 s). The card should change to *"Motion prior attached"*. This is the
headline research path — the clip becomes the immutable tracking base and
the goal above can only add a bounded residual on top of it. Skip this
and you're just training a kick from scratch.

Click **Launch**. The card's `~50 s` estimate is calibrated on cartpole;
for G1 with an authored world expect **3–5 min** — about 40 s of env
build, ~3 min of training, then rollout + render.

### What to check while it runs

**Training tab → Event log → filter `Log lines`.** Three lines prove the
world fix is live:

```
env grid pitch for authored scene: env_spacing 2→7.796 m
constraint budget for authored scene: njmax 300→1536
event:world_route_state_initialization→50% entrance / 25% collision-local interior / 25% terminal-approach starts
```

The first is the fix for the interpenetrating boxes. The second is
headroom. The third is the route reset that used to drop the robot
through its own platforms.

There should be **zero** `nefc overflow` lines.

### What to check when it finishes

**Overview tab → `Replay`** (top-right of the viewer, next to `Static`
and `Live`).

- **one** course, not a lattice of overlapping boxes
- the robot standing **on** the ground and the platforms, not sunk into
  them
- overlay reads roughly **`replay · iter 1 · 1.3s`**

**That short duration is the policy, not the player.** Clip length is
`actual episode length × 0.02 s`, capped at `max_episode_steps` — not
always the cap itself. A 100-iteration policy falls at about step 64, so
you get ~1.3 s (measured here: 1.26 s, 63 frames). The old 10.00 s clip
was a 500-step episode that survived to its cap. The genuinely-broken
case you saw was 7 frames, and the viewer now labels anything under
1.0 s `truncated` in amber. No amber label means the episode really ran
that long.

The scene is the whole point of this step. The policy will be bad — 100
iterations is nothing — but the *scene* must be right. If it isn't, stop
here and send me the frame.

---

## Step 3 — the real run (~45 min, live LLM)

Run Step 0's command again if you want cycle 1 to actually train
(the pipeline check just wrote a new `iter_1`).

**Runs tab → New run → Live rehearsal.** That is 2 outer cycles, 350
rsl_rl iters each, 512 envs, real LLM calls, 300-step episodes, 2 rollout
episodes, and it pauses for your feedback between cycles.

Same behavior goal, and attach the same motion prior again — the dialog
does not remember either between launches:

```
Run in, launch off one leg, and strike with the trailing leg at the apex, tracking the composed novel-running-jump-kick reference throughout.
```

The dialog estimates ~1.9 h; that is deliberately conservative. Measured
here it is closer to **20 min per cycle**.

Per cycle you should see, in the Training tab: train → rollout renders →
live clip appears → diagnose → a reward edit proposed → **pause**. Each
cycle's replay caps at 6.00 s (300 × 0.02) but will be shorter for as
long as the policy is still falling early.

**The pause is not a hang.** An amber card appears —
*"Awaiting your feedback · after iteration 1"* — with a text box and
**Continue** / **Continue + go Auto**. Type what you saw in the replay
(or nothing) and press Continue.

Useful feedback to paste for cycle 1, if the replay shows what I expect:

```
The robot is not translating forward — it crouches and holds instead of running in. Weight the approach phase's forward progress much harder relative to posture, and check whether the episode is ending before the strike window is ever reached.
```

After cycle 2, the **Rewards tab** should show a new version and the
Runs tab should show a `v_n → v_n+1` transition. If all proposed edits
get filtered at pre-flight, no new version is written — that has happened
before and is worth reporting, but it is a known behaviour, not a crash.

---

## Optional — per-mode reward authoring (~5 min, 3 LLM calls)

Independent of the run above; no GPU. This is the headline research
feature: split a composed motion into OGMP modes and have Claude write a
separate reward for each phase.

1. **Rewards tab**, bottom card: **Per-mode reward**.
2. Search box: `running jump kick` → **Search**.
3. Pick **`novel-running-jump-kick--g1`** (444 frames @ 120 fps, 3.7 s).
4. Card reads *"3 modes at 120 fps…"*.
5. **Scaffold reward** — instant, no LLM. Writes the dispatch and the
   mode windows (`approach` 0–1.25 s, `launch` 1.25–2.5 s, `strike`
   2.5–3.7 s), leaves three `_mode_*` bodies as stubs.
6. **Author** on each mode in turn — a real Claude call each, 1–3 min.
7. All three `Authored` → **Use for training**. The version list at the
   top of the tab should gain a new `SCULPTOR` entry.

**Worth deliberately breaking:** hit **Use for training** after authoring
only one mode. It should refuse and name the unauthored ones. An empty
stub pays zero reward for its slice of every episode.

**Worth judging:** open the authored source. Do the three `_mode_*`
bodies actually differ, or are they three variations of the same posture
term? I can't answer that for you.

---

## What I'd like back

1. Step 2's replay frame — one course, feet on top?
2. Did Step 3 reach a new reward version, or did pre-flight filter
   everything?
3. Anything ambiguous enough that you had to guess.

## Known-open, so you don't re-report it

- **Every authored-world run before `828fd18` trained in an
  interpenetrating scene** — all five projects with a course. Any
  conclusion drawn from those runs is being re-checked, including
  "the policy learned to crouch and hold" (it couldn't translate; its
  legs were inside solid boxes).
- The training episode isn't tied to the 3.7 s reference clip length, so
  `strike` carries most of the reward mass.
- `recert5` is INFEASIBLE: four wrist joints sit at exactly 0.0000 in
  the reference and `mean_joint_err_rad` averages over all 29 joints.
