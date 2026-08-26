# RL-Sculptor

> An autonomous agent that iterates on reinforcement-learning **reward functions** —
> grounded in a living knowledge graph of research papers, and gated by an
> objective metric it has to *earn the right to trust* before it can steer a run.

[![sculptor tests](https://img.shields.io/badge/sculptor-701%20passed%2C%201%20skipped-brightgreen)](RewardSculptor)
[![backend tests](https://img.shields.io/badge/ui%20backend-345%20passed-brightgreen)](reward-sculptor-ui)
[![python](https://img.shields.io/badge/python-3.13-blue)](RewardSculptor/pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## Why this exists

Reward design is the biggest bottleneck in real-world RL. You pick a weight, train for
hours, watch the agent hack it, tweak a coefficient, retrain. **RL-Sculptor automates
that inner loop:** given a behavior goal and a starter reward, it trains, rolls out,
diagnoses what went wrong, queries a knowledge graph of robotics / locomotion /
manipulation papers for a technique that addresses the diagnosed failure, rewrites the
reward module — every edit carrying an `arxiv_id` — and repeats.

The hard part isn't the rewrite. It's knowing whether the agent is actually getting
*better at the task* or just getting better at *the number you're optimizing*. So the
system's center of gravity is an **objective-metric trust pipeline**: before a generated
metric is allowed to steer a run, it has to pass a layered, mostly-offline gauntlet that
is deliberately hard to game. That pipeline is the project's most active work — see
[The objective-metric trust pipeline](#the-objective-metric-trust-pipeline) below.

---

## The two projects

This is a monorepo with two stacked components:

| Component | What it is | Entry point |
| --- | --- | --- |
| **[`RewardSculptor/`](RewardSculptor)** | The core library + `sculpt` CLI: the train→rollout→diagnose→edit loop, the paper knowledge graph, the adapter ABC, and the metric trust pipeline. Lab-agnostic — drop in an adapter for your RL stack. | `uv run sculpt run …` |
| **[`reward-sculptor-ui/`](reward-sculptor-ui)** | A FastAPI + React/Vite control panel that wraps the sculptor: project lifecycle, robot picker, reward editor (Monaco), KG browser, live run streaming with a 3-mode robot viewer, reports. | `./run.sh` |

Dependency direction: **`reward-sculptor-ui` → `RewardSculptor`** (editable path-install).
Each subproject has its own README with a deep dive:
**[library README](RewardSculptor/README.md)** · **[UI README](reward-sculptor-ui/README.md)**.

For the current lab research direction, exact capability boundary, SONIC/OGMP
reading, and proposal seed, start with the
**[guiding research context](docs/GUIDING_RESEARCH_CONTEXT.md)**.

---

## Architecture in one screen

```
                         behavior goal + starter reward
                                      │
        ┌─────────────────────────────┴──────────────────────────────┐
        │                    sculpt run (inner loop)                  │
        │   train → rollout → diagnose → edit → commit → (repeat)     │
        └───────┬───────────────┬───────────────┬──────────┬─────────┘
                │               │               │          │
                ▼               ▼               ▼          ▼
            checkpoint      keyframes      failure mode   v<n>.py
                                + metrics   (Claude)      (cited edits)
                                              │
            ┌─────────────────────────────────┴───────────────────┐
            │   knowledge graph: arxiv → PDF → Claude extraction   │
            │   Papers · Techniques · FailureModes · Components     │
            │   queried by failure tag + MiniLM semantic cosine     │
            └──────────────────────────────────────────────────────┘

   gating every steer decision:  objective-metric trust pipeline (L0→L5)
```

The diagnoser is the join point — it pulls the current iteration's artifacts up and the
retrieved techniques down, and hands both to the editor. The full per-iteration anatomy
is documented in the [library README](RewardSculptor/README.md#what-s-actually-happening).

---

## The objective-metric trust pipeline

A generated metric should never silently drive a training run. The single biggest lever
in this whole system is *reliably generating an objective metric that actually measures
the target movement* — so a metric has to **earn steer-rights** through a layered,
standardized trust score (`trust ∈ [0,1]`, one threshold, no per-task tuning). Every
layer writes its verdict to `meta.json`; nothing is silent, and "observe-only" always
names the layer that failed.

| Layer | Proves | Cost |
| --- | --- | --- |
| **L0 Validate** | safe, physical-only, deterministic, bounded, non-degenerate | offline |
| **L1 Task-agnostic axioms** | universal invariants (translation / gravity-scale / yaw invariance, monotone-in-uprightness, no-chaos, stationary-no-travel) | offline |
| **L2 Task-derived ladder** | metric ranks an *independently authored*, metric-blind competence ladder monotonically; **K=3** sources must agree (min, not mean) | K LLM calls, no GPU |
| **L3 Cross-metric consensus + adversarial archetypes** | an independent author proposes "gaming policies"; the metric must score every one below a competent positive | few LLM calls, no GPU |
| **L4 VLM grounding** | numeric metric correlates with a general VLM's "matches goal?" rating over rollout keyframes | API budget |
| **L5 Optimization-outcome audit** | training against the metric yields legit, non-hacked behavior — the only non-circular ground truth | short GPU run |

`L0+L1+L2` is the offline steer-rights minimum for a novel task; for the five built-in
families the same math reduces to the original `rho ≥ 0.7` decision (provable no-op, no
regression). Design spec:
[`docs/internal/DESIGN_autonomous_metric_eval.md`](docs/internal/DESIGN_autonomous_metric_eval.md).

The two failure modes the pipeline is built to beat:

- **Circularity** — same LLM authors the metric *and* its grader. Defeated by independent
  metric-blind authorship, K-source agreement, and (at L5) judging *trained behavior*.
- **Goodhart** — looks great offline, gameable when optimized. Defeated by the
  non-degeneracy negatives, the anti-spurious-correlation rank guard, and the L3
  adversarial gaming-archetypes.

---

## Quickstart

### Library (CPU, ~1 min dry run)

```bash
cd RewardSculptor
uv sync
cp .env.example .env                       # add ANTHROPIC_API_KEY
uv run sculpt run \
    --config examples/hopper/config.toml \
    "run forward fast without falling" \
    --iterations 3 --dry-run               # bypasses LLM calls, caps training
```

### Control panel (one command)

```bash
cd reward-sculptor-ui
uv sync
pnpm install --dir frontend
./run.sh                                   # → http://localhost:5173
```

---

## Repository layout

```
RL-Sculptor/
├── README.md                  ← you are here
├── RewardSculptor/            # core library + sculpt CLI + trust pipeline
│   ├── sculptor/              #   diagnose · edit · kg · adapters · eval/
│   ├── examples/hopper/       #   end-to-end CPU demo project
│   ├── docs/                  #   adapters.md, knowledge_graph.md, remote.md, …
│   └── tests/                 #   701 passing, 1 skipped (jax parity)
├── reward-sculptor-ui/        # FastAPI + React/Vite control panel
│   ├── backend/               #   routes · services · 345 passing tests
│   ├── frontend/              #   Vite + React + Tailwind + shadcn/ui
│   └── run.sh                 #   one-command start
├── CONTEXT.md                 # detailed engineering log (per-change changelog)
└── docs/internal/             # design notes, pivots, and session handoffs
```

---

## Tests

```bash
# core library — 701 passed, 1 skipped (jax-only parity test)
cd RewardSculptor && uv run pytest tests/ -q

# UI backend — 345 passed
cd reward-sculptor-ui && uv run pytest backend/tests/ -q

# frontend typecheck
cd reward-sculptor-ui/frontend && pnpm typecheck
```

All Anthropic API calls in the test suites are mocked; GPU tests are skipped on machines
without CUDA.

---

## License

MIT. See each subproject for third-party notices.
