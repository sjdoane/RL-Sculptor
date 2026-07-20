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

## Current campaign-integrity rules

`sculpt eval run` now freezes `campaign_charter.json` before the first
training job. The charter pins the effective campaign config, exact benchmark
and condition definitions, analysis/failure policy, executable source tree,
dependency lock, and frozen KG input hash. Every `result.json` and report is
bound to that design hash. A changed seed, budget, source file, task spec,
condition, or analysis rule requires a fresh `--out` directory.

KG-enabled conditions do not read or write the live shared KG during the
campaign. The runner makes one transactionally consistent
`campaign_inputs/kg_base.db` snapshot, then initializes a private writable KG
copy for every `(benchmark, condition, seed)` job. A crash-resume reuses only
that job's copy; no later paired arm can inherit cases learned by an earlier
job.

Output directories containing legacy results but no charter are intentionally
rejected. They may still be inspected as historical artifacts, but cannot be
retroactively represented as pre-registered runs.

The four built-in specs predate the adversarial certificate workflow and are
now surfaced as `legacy_provisional` in events, JSON, CLI warnings, and HTML.
This keeps old smoke campaigns usable without presenting them as A4 reporting
authority. New external campaign entries must carry a verified passing A4
certificate; see `docs/spec_audit.md` and `docs/benchmarks/README.md`.

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

## Safe charter-aware sharding

`sculpt eval shard` is the only supported way to merge multi-process or
multi-pod results into one campaign. Preparation receives the **entire**
benchmark × condition × seed design, creates one global charter, and then
partitions its already-frozen Cartesian product by job assignment. A shard
manifest is not a smaller experiment: it carries the full partition and
references the same charter design, runtime, dependency, and external-input
hashes as every other shard.

Prepare once on the coordinating machine (repeat every `-b` and `-c`; never
prepare separate benchmark subsets):

```bash
sculpt eval shard prepare --out ~/rs_campaign --shards 3 \
  -b cartpole_balance -b g1_floss -b g1_kick -b go1_trot \
  -c mission -c mission_no_kg -c full -c eureka \
  -c seed_only_matched -c plain_ppo_matched -c plain_ppo \
  --seeds 5 --iterations 4 --steps-per-iter 600 --eureka-k 3 \
  --name e4-campaign
```

This creates `~/rs_campaign/shards/shard-000` through `shard-002`.
Transport each complete shard directory to one pod, preserving its internal
layout, and run it there:

```bash
sculpt eval shard run /workspace/shard-000/shard_manifest.json
```

Each worker verifies the full charter against its current source and
dependency hashes before its first job. It also verifies its byte-identical KG
base replica, records a sealed worker runtime identity, and creates a private
writable `inputs/kg.db` copy per assigned job. Re-running the same command is
the crash-resume path: only exact matching `result.json`, charter, manifest,
runtime, dependency, and KG lineage are reused.

Fetch the completed shard directories back without combining their contents,
then merge them at the global root:

```bash
sculpt eval shard merge --out ~/rs_campaign \
  --shard-dir ~/fetched/shard-000 \
  --shard-dir ~/fetched/shard-001 \
  --shard-dir ~/fetched/shard-002
```

Merge rejects foreign charters, altered runtime/dependency or KG hashes,
manifests that differ from the frozen partition, result tuples outside their
assignment, and a `(benchmark, condition, seed)` claimed by multiple shards.
Missing shards are not an integrity error: available verified results are
merged, while `campaign_report.json`, `campaign_merge.json`, and `report.html`
declare `incomplete_coverage` and enumerate every missing tuple. Such partial
aggregates are diagnostic only, not the headline campaign result.

The old pattern—independent `sculpt eval run` commands with different
benchmark lists followed by an ad hoc merge—remains unsafe and unsupported.
Those commands define different charter hashes; the coordinator does not and
cannot reinterpret them as one experiment.

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
