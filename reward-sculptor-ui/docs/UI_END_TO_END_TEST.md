# End-to-end UI test — run this yourself

Four tests, in order, shortest first. Each one stands alone: stop after any
of them and you'll still have told me something useful. Times are measured
on this machine (RTX 5070 Laptop), not estimates.

| # | What it proves | Time | Costs |
|---|---|---|---|
| 1 | The last run's artifacts are real and visible | 2 min | nothing |
| 2 | Per-mode reward authoring — the headline feature | ~5 min | 3 LLM calls |
| 3 | Launch → train → rollout → diagnose plumbing | 50 s | nothing (stubbed) |
| 4 | The real loop, end to end | ~1.9 h | GPU + LLM |

## Start

```bash
cd ~/projects/reward-sculptor-ui && ./run.sh
```

That's the only terminal command in this document. It starts both servers
and opens <http://localhost:5173>. Ctrl+C stops everything. If you already
had the UI open, reload the page once — the replay fixes below shipped
after your last session.

---

## Test 1 — see what's already on disk (2 min)

Open **Projects → Tracking First UI Verification**.

**1a. Overview tab → click `Replay`** (top-right of the robot viewer, next
to `Static` and `Live`).

You should see:
- the overlay reads **`replay · iter 1 · 10.00s`**
- one iteration chip along the bottom: **`iter 1`**
- a 10-second video, 1280×720

**This video shows a broken scene, and that is expected — it was rendered
before the fix.** The frame is packed with interpenetrating orange and blue
boxes and the robot is embedded in one of them. That is real: mjlab shares
one model across all 1024 parallel environments and repeats the authored
course at each one, but the environment grid pitch was mjlab's 2.0 m default
while the course reaches 6.8 m forward — so every course overlapped its
neighbours three deep and every robot spawned inside someone else's boxes.
Fixed in `9e3e5cf`; **any run you launch now renders one clean course.** The
old artifacts on disk are kept as-is rather than re-rendered, because the
policy in them was trained in the broken scene and re-rendering would only
make bad training look tidy.

So use this step to confirm the *player* works (10.00 s, iter 1, scrubbable),
and judge the *scene* on a run you launch yourself in Test 3 or 4.

> This is what you were looking at before, and you were right to flag it.
> The viewer was defaulting to a *different* run — a five-day-old
> `four-box-parkour-demo / run_to_course` stage whose rollout is genuinely
> **0.07 s long (7 frames)**. That run predates the constraint-budget fix;
> its episode died after 7 physics steps. Three bugs stacked up to put it on
> screen; all three are fixed in commit `7faa7de`. If a rollout is ever
> that short again, the overlay now says so explicitly —
> `replay · iter 0 · 0.07s truncated` in amber.

**1b. Rewards tab.** Expect `Versions 2`, with **`v1 SCULPTOR 3.5`** and
`v0 HUMAN`. Click `v1` — the source header should say
*"Auto-generated per-mode reward scaffold for clip
`novel-running-jump-kick--g1`"* and `MODE_WINDOWS_S` should list three
modes: `approach` 0–1.25 s, `launch` 1.25–2.5 s, `strike` 2.5–3.7 s.

**1c. Training tab.** The run card for `324d2a0b8a474b02` should read
`COMPLETED`, `iter 1`, `r 3.5`, and:

> `v1 · 5 edits filtered — reward_hacking, static_equilibrium, sparse_reward, reward_saturation`

That line means the diagnoser watched the rollout, correctly identified
reward hacking, proposed 5 fixes — and pre-flight rejected all 5, so no v2
was written. The diagnosis is the working part; the edit gate is what to
scrutinise.

*Optional deep check:* in the Event log, set the filter to **`Log lines`**
and look for `constraint budget for authored scene: njmax 300→1536`. That
is the fix that stopped silent contact-row overflow. There should be zero
`nefc overflow` lines in the whole run.

---

## Test 2 — author a per-mode reward (~5 min, no GPU)

This is the feature to judge. It takes a composed motion, splits it into
OGMP modes, and has Claude write a separate reward for each phase.

1. **Rewards tab**, scroll to the bottom card: **Per-mode reward**.
2. In the search box type `running jump kick`, press **Search**.
3. Pick **`novel-running-jump-kick--g1`** (top hit — "running approach into
   a one-leg jumping kick", 444 frames @ 120 fps, 3.7 s).
4. The card should now read *"3 modes at 120 fps. The windows and the
   dispatch are generated from the automaton; a model writes only one
   mode's terms per call."*
5. Click **Scaffold reward**. Fast, no LLM — it writes the dispatch and the
   mode windows and leaves the three `_mode_*` bodies as stubs. The three
   modes (`approach`, `launch`, `strike`) each get an **Author** button.
6. Click **Author** on each mode in turn. Each is a real Claude call,
   **1–3 min**. The button reads `Authoring…` then `Authored`.
7. When all three say `Authored`, click **Use for training**.

**What to check:** step 7 is the one that used to silently do nothing.
`mode_reward_v*.py` files are not reward versions, so the promote step has
to copy the authored file into the version chain. This project currently
has `v0` and `v1`, so promotion should write **`v2`** — the version list at
the top of the tab should gain a **`v2 SCULPTOR`** entry whose source is
your authored per-mode reward, not the starter `alive_bonus`. If the
version count doesn't change, that's a bug worth reporting.

**Worth deliberately breaking:** click **Use for training** after authoring
only one or two modes. It should *refuse*, naming the unauthored modes — an
empty stub pays zero reward for its slice of the episode, so promoting a
half-authored module would silently train a reward that's blank for part of
every rollout. If it promotes anyway, tell me.

**Also worth judging:** open the authored source and read the three
`_mode_*` bodies. Do the reward terms actually differ per phase, or did
Claude write three variations of the same posture term? That is a research
question I can't answer for you.

---

## Test 3 — 50-second pipeline check

**Runs tab → New run.** Under **RUN PLAN**, pick **Pipeline check**
(`~50 s · stubbed LLM · no GPU commitment`). The estimate line should
update to `Estimated wall-clock: 50 s`. Click **Launch**.

This caps training at 1000 steps and stubs every LLM call, so it exercises
launch → env build → train → rollout → render → diagnose plumbing without
spending anything. Watch the **Training tab**: events should stream, a
rollout should render, and the run should end `COMPLETED`.

**Two things that will confuse you if you don't know them:**

- **Manual mode is ON by default.** The run pauses after each iteration
  with an amber card: *"Awaiting your feedback · after iteration N"*, a
  text box, and **Continue** / **Continue + go Auto**. It is not hung. Type
  what you saw (or nothing) and press Continue.
- **Resume is on.** This project already has `iter_1` on disk, so a new run
  reuses it instead of retraining. The dialog says so explicitly —
  *"Resume is enabled — this project has 1 iter(s) on disk"*. For a truly
  clean run, start a fresh project (**New project** → Robot Library →
  **Unitree G1**, marked READY TO TRAIN).

---

## Test 4 — the real loop (~1.9 h)

Same dialog, pick **Live rehearsal** (`2 real cycles · pauses for
feedback`). Estimate should read ~1.9 h. Launch, then leave the Training
tab open.

Per cycle you should see: train → rollout renders → live clip appears →
diagnose → reward edit proposed → the pause for your feedback. After it
finishes, the Rewards tab should show a new version and the Runs tab should
show the `v_n → v_n+1` transition.

**Overnight showcase** (4 cycles, auto, ~9.2 h) is the same thing without
the pauses. Only worth it once Test 4 looks right.

---

## What I'd most like to hear back

1. Did the Test 1 replay show `iter 1 · 10.00s`, or something else?
2. In Test 2, did **Use for training** produce a new version?
3. Do the three authored mode rewards look meaningfully different from each
   other?
4. Anything that looked broken, ambiguous, or that made you stop and guess.

Screenshots of anything odd are ideal — the run's slug and iteration number
are enough for me to find the artifacts on disk.

## Known-open, so you don't re-report them

- `strike` carries 98.6 % of v1's reward mass — a real property of the
  reward function, and the structural cause is that the training episode
  isn't tied to the 3.7 s reference clip length. But the *behavioural*
  reading I drew from it ("the policy learned to crouch and hold") is no
  longer trustworthy: that policy trained with its legs inside solid boxes,
  so it could not translate even if it wanted to. Needs re-collecting on a
  correctly-pitched scene.
- All five proposed edits were filtered at pre-flight, so v1 never advanced
  to v2 automatically.
- **Every authored-world run before `9e3e5cf` trained in an
  interpenetrating scene** — all five projects with a course, not just the
  parkour mission. Conclusions from those runs are being re-checked. The
  constraint-budget increase I made earlier was treating a symptom of this;
  with the pitch fixed the scene fits the task's own default.
- `recert5` is INFEASIBLE: four wrist joints sit at exactly 0.0000 in the
  reference, and `mean_joint_err_rad` averages uniformly over all 29
  joints.
