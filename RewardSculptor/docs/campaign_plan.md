# E4 Campaign Plan (frozen pre-launch — §Ship 31)

The headline experiment: does KG-grounded curriculum reward-sculpting
beat its ablations and the literature baselines on objective spec
metrics, under honest compute accounting?

## Matrix

| axis | values |
|---|---|
| benchmarks | `cartpole_balance` (sanity), `g1_floss`, `g1_kick`, `go1_trot` |
| conditions | `mission` (full system) · `mission_no_kg` · `full` (no-curriculum) · `eureka` (E3) · `seed_only_matched` (no-diagnose) · `plain_ppo_matched` · `plain_ppo` |
| seeds | 5, paired: 1000, 1017, 1034, 1051, 1068 |
| LLM budget | `iterations = 4` per sculpt-loop job; mission stages early-stop on criterion |
| GPU budget | `steps_per_iter = 600` rsl_rl iterations |
| eureka scale | `eureka_k = 3` × 3 generations (paper used K=16 × 5 — state as a scale delta in any writeup) |

Primary comparisons (paired-difference CIs, Ship-27 machinery):
`mission − mission_no_kg` (the KG question), `mission − full` (the
curriculum question), `mission − eureka` (the literature question),
each read against `total_rl_iterations`.

## Budget (measured 5090 numbers: 0.65 s/iter G1@4096; ~60 s dispatch overhead; ~2 min local rollout; ~2 min LLM per loop-iter)

Per G1-class benchmark, minutes/job × 5 seeds:

| condition | est. /job | ×5 seeds |
|---|---|---|
| mission | ~60 m | 300 m |
| mission_no_kg | ~60 m | 300 m |
| full | ~46 m | 230 m |
| eureka (3×3) | ~85 m | 425 m |
| seed_only_matched | ~29 m | 145 m |
| plain_ppo_matched | ~29 m | 145 m |
| plain_ppo | ~9 m | 45 m |
| **Σ** | | **~26.5 h** |

× 3 heavy benchmarks ≈ 80 h + cartpole ≈ 6 h → **~86 pod-hours ≈ $60
GPU** (3 × 5090 @ $0.69/hr ≈ 29 h wall ≈ 1.2 days) + **~$50–90 LLM**
(decompose/diagnose/edit/eureka-samples ≈ 700–900 Opus calls).
Trim lever if needed: drop `plain_ppo` (unmatched) and `mission_no_kg`
from one heavy benchmark each (−~10 h).

## Sharding (3 × 5090 pods)

One harness process per pod, each with its own `SCULPTOR_REMOTE_*`
env; jobs are disjoint by benchmark so the out-dirs merge trivially.

```bash
# pod A                                   # pod B                       # pod C
sculpt eval run --out ~/rs_campaign \
  -b g1_floss   <ALL_CONDITIONS>          -b g1_kick  <ALL_CONDITIONS>  -b go1_trot -b cartpole_balance <ALL_CONDITIONS>
  --seeds 5 --iterations 4 --steps-per-iter 600 --name e4-campaign
```

All shards write into the SAME `--out` tree (disjoint job dirs);
afterwards `sculpt eval report ~/rs_campaign` re-aggregates everything
into one `campaign_report.json` + `report.html`. Every job is
resumable (`result.json` keys); pod restarts only change host/port env.

Launch detached on Windows (WSL kills `setsid` children when the last
client exits): `Start-Process -WindowStyle Hidden wsl 'bash -c "…"'`.

## Go / no-go checklist (run in order)

1. 3 pods provisioned: `./scripts/provision_remote.sh root@<ip> -p <port> -i ~/.ssh/id_ed25519 -w /workspace/sculptor_remote` (per pod, ~2 min warm).
2. `sculpt remote doctor` green ×3 (version-skew check pins the stack).
3. **Smoke (~$3)**: `sculpt eval run --out /tmp/e4_smoke -b cartpole_balance -c mission -c eureka --seeds 1 --iterations 2 --steps-per-iter 300` on one pod — proves the two not-yet-live-tested condition modes end-to-end (mission: decompose → stages; eureka: sample → train → reflect).
4. Capture-parity warnings empty in the smoke report.
5. `ANTHROPIC_API_KEY` set; spend alert configured at console.anthropic.com.
6. Launch all three shards; check `eval_job_finished` events per shard after the first job (~10–60 min).

## Grounding state at freeze (Ship 30 + 31)

KG: 94 vetted papers / 1452 nodes / 1476 edges / 493 embeddings;
seeds provenance committed (`kg_seeds_campaign.yml`). Anti-
hallucination gates active: floored semantic slices
(`DEFAULT_MIN_PROMPT_SIMILARITY = 0.35`) in decompose/diagnose/edit;
citation verification at BOTH diagnose (drop + `kg_citation_dropped`
event) and edit (hard gate); scored deterministic failure-mode
resolution; criterion↔component reconcile at iter 0 (Ship 25a);
reward post-validation (AST + probe + contract); spec metrics
independent of all LLM-authored criteria (Ship 26).
