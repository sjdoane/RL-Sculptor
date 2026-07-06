# Handoff — research-grade elevation session (2026-07-06)

Session ended early: the Claude Code monthly spend limit was hit mid-run
(two subagents died on it; main loop finished inline). Everything below
"SHIPPED" is committed on `ship-20-ux-revamp` and gate-verified
(sculptor 1175 passed/1 skip, backend 376 passed, pnpm build green).
Everything below "NOT DONE" is designed but unimplemented — specs
included so the next window starts from zero context.

## SHIPPED (commits c75d8f8..0551a6d)

1. **c75d8f8** — recovered the stranded `claude/optimistic-chatterjee-*`
   branch: multi-seed selection statistics (§7.2), hack-income screen,
   `sculptor/reference.py` (procedural jump clip + RSI derivation),
   RESEARCH_GAP_ANALYSIS.md, selection-stats UI knobs.
2. **f26e2c0** — six UI fixes (spinner timeout+retry, Modal pointer-down
   scrim tracking, Settings grid, runs-sidebar overflow, Knowledge
   polish, Rewards loop explainer). Browser-verified in live preview.
3. **f128ca8** — `sculptor/llm.py`: per-role model registry
   (RS_MODEL_<ROLE> / RS_MODEL_ALL overrides) + llm_calls.jsonl
   provenance archive at every call site (§3.9/§7.8). Role upgrades:
   fable-5 on decompose/diagnose/edit/metric_gen/env_gen AND
   eureka_baseline (model-matched arms); calibration=opus-4-8
   (deliberately author-disjoint, §3.5); kg_*+mjcf=sonnet-5. All ids
   verified live. **Cost note: fable-5 is materially pricier per call
   than opus-4-7 — RS_MODEL_ALL=claude-opus-4-8 is the one-var
   downgrade for budget runs.**
   Plus **best-of-K reward candidates** (§3.3): apply_edits
   n_candidates=K samples strategy-framed rewrites (MINIMAL-DIFF /
   STRUCTURAL / EXPLORATION-FIRST / ROBUSTNESS), full validation each,
   ranked by worst-case honest-vs-exploit replay margin, winner-only
   trains. Config: `[iteration] edit_candidates` (default 1;
   **recommended 3 for the jump project**). Fixed en route: staging
   .pyc mtime+size cache collision (unique staging names).
4. **a97a982** — per-stage trust-gated metrics (decision record
   `docs/internal/MISSION_METRIC_GRANULARITY.md`): mission-init +
   backend decompose generate one metric per stage from stage goal
   text; rejected → mission-metric fallback; mission_run anchors
   mission-dir-relative refs.
5. **021ae85** — `Stage.needs_reference_rsi`: decomposer flags airborne
   stages (prompt rule 9); orchestrator applies clip-derived TRAIN-only
   RSI + paired sunk termination to the stage env before training;
   resume-idempotent. Plus decompose criterion grounding fix
   (available_trajectory_keys per adapter — live smoke caught the
   decomposer citing root_link_pos_w on a gym env).
6. **0551a6d** — partition-gate rename bypass closed (1:1 removed+added
   reject-gate pair with lower value = HARD, §7.9); decompose calls now
   provenance-archived (sink set before decompose_task).
   Note: L3 adversarial archetypes were ALREADY default-ON
   (run_manager.py RS_ADVERSARIAL_ARCHETYPES=1) — memory was stale.

**Live e2e evidence (mission system, WS4):** scratch project
`~/rs_smoke/hopmission`, mission `smoke-hop-stop` — real fable-5
decomposition into 3 warm-start-chained stages; per-stage metric
generation ran the FULL trust pipeline on all 3 and correctly REJECTED
all 3 for the out-of-manifest gym hopper (non-degeneracy + stationarity
axioms firing exactly as designed), graceful fallback, structured
events, mission.json runnable, per-metric llm_calls.jsonl written.
Mission EXECUTION (training) e2e is covered by 128 orchestrator tests
incl. the new RSI + steering-metric ones, but no live training ran.

## NOT DONE (cut off by the spend limit) — next-window specs

**A. G1 standing-jump mission (WS5 live test).** All scaffolding is in.
Runbook:
```bash
cd ~/projects/RewardSculptor && source .venv/bin/activate
export ANTHROPIC_API_KEY=$(grep -oP 'ANTHROPIC_API_KEY\s*=\s*\K\S+' .env)
P=~/.local/share/reward-sculptor/projects/tuck-jump   # or a fresh G1 project
# add to $P/config.toml [iteration]: edit_candidates = 3
python -m sculptor.cli mission-init $P \
  --goal "From a stationary stand, perform a vertical standing jump: crouch, explode upward to visible flight (both feet off the ground), and land back in a stable stand." \
  --slug g1-standing-jump
# inspect .missions/g1-standing-jump/mission.json — expect airborne
# stages flagged needs_reference_rsi + per-stage gen metrics accepted
# (G1 IS in the robot manifest, unlike the hopper smoke)
python -m sculptor.cli mission-run $P g1-standing-jump
```
Expect ~22 min/iter on the 5070 at tuck-jump's config. The decomposer
should produce crouch→launch→land stages; RSI fires on the airborne
ones. Real retargeted clips (better than procedural): Unitree LAFAN1
HF dataset (auth-gated) → convert to `$P/reference/jump.npz`
(root_pos_z + fps is enough — sculptor/reference.py docstring).

**B. KG restructure (WS3) — full spec written, zero code.** Four
upgrades derived from Google's agentic-data guidance (Agentic Data
Cloud/Knowledge Catalog Apr 2026; Context Engineering whitepaper Nov
2025; sources + 12 ranked principles in the session transcript):
  1. Provenance trust tiers on nodes (observed_run > paper_claim >
     llm_extraction > seed), surfaced as `[evidence: …]` tags in
     diagnose/edit prompt context with an explicit precedence rule.
  2. Usage-based enrichment: on run end, "helped" iterations bump
     `useful_citations` on the techniques their kept reward cited;
     query ranking adds a CAPPED boost (≤0.05 similarity units).
  3. Retrieval trajectory logging: kg_retrievals.jsonl per decision
     (pairs with llm_calls.jsonl for §7.1's retrieval-precision
     analysis).
  4. RunCase validity scope: record reward/env versions; recency
     tie-break (0.02 similarity window) in query_cases.
  Constraints: metric-calibration firewall untouchable; old graph.db
  must keep loading.

**C. Held-out eval battery (WS2 §7.4):** perturbation suite
(push/friction/mass deltas on the frozen shared env) + repaired hand
specs, run only on final kept policies. Not started.

**D. Adversarial verify of f128ca8/a97a982** was killed mid-run.
Inline re-checks done: best-of-K tie ranking, all-invalid fallback
propagation, contextvar-in-thread semantics, backend phase-2 mission
round-trip. Still worth a fresh subagent pass next window (charter in
transcript).

**E. E4-v2 (§7.5)** remains the one big spend — now properly gated on
A-D. The eureka arm is already model-matched to the treatment arm.
