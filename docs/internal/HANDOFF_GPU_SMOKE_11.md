# #11 GPU smoke — exact runbook (Sam runs; report the results back)

Status: the metric-quality-laws CODE build (#1–#10, #12, #4, #9) is complete + green
(sculptor 789 / UI 351). `ANTHROPIC_API_KEY` IS available to sculptor (auto-loaded from
`RewardSculptor/.env`). #11's GPU foot-plumbing is already confirmed live (Tier 1 below was
run successfully). Tier 2 (the full kick LOOP) needs a g1-kick PROJECT — none is on disk
(only the `hopper` gym example) — so it must be created (UI, or ask Claude to scaffold one).

Run everything from WSL: `cd ~/projects/RewardSculptor`. (`export WANDB_MODE=disabled` once —
the runner's wandb autologin is the only gotcha when driving it directly.)

---

## TIER 1 — GPU foot-data + live-metric smoke (VERIFIED; ~3 min; no project, no API)
Confirms #5 plumbing + that `spec_g1_kick` runs on real GPU arrays.

```bash
cd ~/projects/RewardSculptor
export WANDB_MODE=disabled
rm -rf /tmp/g1_smoke && mkdir -p /tmp/g1_smoke
.venv/bin/python -m sculptor.adapters._mjlab_runner train \
  --task-id Mjlab-Velocity-Flat-Unitree-G1 --num-envs 512 --max-iterations 5 \
  --output-dir /tmp/g1_smoke/train --device cuda:0
.venv/bin/python -m sculptor.adapters._mjlab_runner rollout \
  --task-id Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-path /tmp/g1_smoke/train/checkpoint.pt \
  --output-dir /tmp/g1_smoke/roll --n-episodes 4 --max-episode-steps 150 --device cuda:0
.venv/bin/python - <<'PY'
import json, numpy as np
from pathlib import Path
from sculptor.eval.spec_metrics import spec_g1_kick
d = Path("/tmp/g1_smoke/roll"); tr = dict(np.load(d/"trajectory.npz"))
for k in ("left_foot_pos_b","right_foot_pos_b","left_foot_contact","right_foot_contact"):
    print(k, "present" if k in tr else "MISSING", tr[k].shape if k in tr else "")
names = json.loads((d/"mjcf_limits.json").read_text()).get("joint_names", [])
beh = json.loads((d/"behavior.json").read_text())
out = spec_g1_kick(tr, beh, {"joint_names": names})
print("spec_score:", round(out["spec_score"],4), "(walker should be ~0)",
      "| direction active:", "kick_direction" in out)
PY
```
PASS if: the four foot keys are present + non-empty, and `spec_score` is small (~0.02) with
`direction active: True`. (Reference run: foot keys present (T,64,3)/(T,64); spec_score 0.0201.)

---

## TIER 2 — full kick LOOP (needs a g1-kick project; this is what exercises #7/#10 live)
The #7 steer-gate and #10 Goodhart-onset are LOOP behaviors — they fire across iterations, so
you need a multi-iter run on a kick task. There is no g1-kick project on disk; create one:
- **Easiest (matches your UI workflow):** `cd ~/projects/reward-sculptor-ui && ./run.sh`, create a
  new run on the **Unitree-G1** mjlab adapter, set **Objective fitness metric = g1_kick**, mode
  **steer**, ~6–8 iterations, and launch. Watch the run timeline events.
- **Or ask Claude to scaffold** a minimal `examples/g1_kick/` (config.toml + a v0 kick reward) so you
  can run the CLI below. (Recommended if you want a clean repeatable artifact.)

CLI form (once a `config.toml` exists):
```bash
export WANDB_MODE=disabled
.venv/bin/python -m sculptor.cli run "kick forward with the left leg from a stance" \
  --config examples/g1_kick/config.toml \
  --fitness-metric g1_kick --fitness-mode steer \
  -n 8 --rollout-episodes 6 --steps-per-iter 80
```

### The four acceptance checks — exactly what to watch in the event stream / run report
1. **Real forward kick scores HIGH** — as a competent kick emerges, that iter's objective fitness
   (`spec_g1_kick.spec_score`) climbs toward ~0.78 (the offline competent reference) with
   `kick_direction` ≈ 1. Seen in the per-iter objective_progress / run report.
2. **One-leg-balance / kick-behind score BELOW floor** — if the loop wanders into a v5 hack, that
   iter's `spec_score` is ~0 (completion gate / signed direction), NOT 0.13–0.38. Inspect that
   iter's `trajectory.npz` + objective score.
3. **#7 steer-gate fires** — look for `realism_audited` events. When a rollout is too aggressive,
   `naturalness_steer_factor` < 1.0 (down-weighted ×0.5) or `naturalness_hard_reject=true` (a
   joint-limit exploit → steer 0); that iter's `steer_fitness` is then below its displayed `fitness`,
   so best-selection skips it. (Naturalness also rides on `objective_progress` for the diagnoser.)
4. **#10 Goodhart-onset fires** — look for an `early_stop` event whose `reason` starts
   `"goodhart onset:"`. It needs ≥4 aligned iters where the metric MAX keeps rising while
   naturalness SUSTAINABLY declines (≥2 of the last 3 iters non-pass). The loop then locks the prior
   best instead of optimizing further into the exploit.

### Optional — exercise Panel A (#9) + the spec audit (#4) live
- `RS_ADVERSARIAL_ARCHETYPES` is now DEFAULT ON: a **generate-at-launch** kick run (fitness =
  "generate at launch") emits a `metric_spec_audit` event (audit-only; never revokes the built-in).
- The #9 review PANEL is library-ready (`metric_gen.generate_objective_metric(review_models=...)`);
  its UI wiring + the VLM keyframe lens land with the post-rollout re-review follow-up.

Report items 1–4 (and the Tier-1 PASS) back and #11 closes.
