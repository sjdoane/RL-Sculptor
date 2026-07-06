# Reward Sculptor

> An autonomous agent that iterates on reinforcement-learning reward
> functions — grounded in a living knowledge graph of research papers.
> Every edit is cited.

[![tests](https://img.shields.io/badge/tests-63%20passed%2C%201%20skipped-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml) [![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 60-second pitch

Reward design is the biggest bottleneck in real-world RL. You pick a
weight, train for hours, watch the agent hack it, tweak a coefficient,
retrain. Reward Sculptor automates the inner loop: given a behavior
goal and a starter reward, it trains, rolls out, diagnoses what went
wrong, queries a knowledge graph of legged-robotics / locomotion /
manipulation papers for techniques that address the diagnosed failure,
rewrites the reward module, and repeats — producing a time-lapse, a
changelog, and a citation graph along the way.

**Key property**: every reward edit has an `arxiv_id` attached. The
sculptor isn't guessing — it's citing.

Two loops, one system:

- **Inner (iteration)**: `train → rollout → diagnose → edit → train`, 3-5
  minutes per round on CPU Hopper, longer for real tasks.
- **Outer (knowledge graph)**: arxiv seeds → PDFs → Claude extraction →
  techniques / failure-modes / reward-components / environments →
  queryable by failure tag and by semantic similarity.

It's **lab-agnostic**. Drop in an adapter for your RL stack and
sculptor drives it. Ready-to-train + scaffolded adapters:

| Adapter | Status | Notes |
| --- | --- | --- |
| `gym_sb3` | ✓ ready | Gymnasium + Stable-Baselines3 reference path ([docs/adapters.md](docs/adapters.md)) |
| `mjlab` | ✓ ready | mjlab (MuJoCo-Warp) — GPU, manager-based ([design note](../docs/internal/MJLAB_PIVOT_DESIGN.md)) |
| `isaac` | ⏳ scaffolded | Isaac Lab 2.0+ ([adoption guide](docs/adapters/isaac.md), ~4-8 hours) |
| `mjx` | ⏳ scaffolded | Brax / MJX (JAX) ([adoption guide](docs/adapters/mjx.md), ~4-6 hours) |
| `rllib` | ⏳ scaffolded | Ray RLlib ([adoption guide](docs/adapters/rllib.md), ~4-8 hours) |

Implementing a scaffolded adapter takes ~4-8 hours of focused work for
someone already familiar with that stack — each guide has a completion
checklist. **Contributions welcome.** The scaffolded adapters satisfy
the `SculptorAdapter` ABC with sensible `reward_contract()` defaults so
the UI can surface them in the project-creation dropdown without
crashing; `train()` / `rollout()` raise `NotImplementedError` pointing
at the adoption guide.

See the ["Adapting to your RL stack"](#adapting-to-your-rl-stack)
section below for the Brax worked example.

---

## Install

```bash
git clone <this-repo>
cd RewardSculptor
uv sync                            # installs all deps
cp .env.example .env               # add your ANTHROPIC_API_KEY
uv run sculpt --help
```

Requires Python ≥ 3.10 and an Anthropic API key.

**GPU requirements (mjlab adapter only).** NVIDIA GPU with CUDA 12.4+
and ≥ 6 GiB VRAM. Tested daily on RTX 5070 Laptop (8 GiB,
CUDA 13.0). The `gym_sb3` adapter works on any CPU. Full WSL2 setup
in [`docs/wsl_setup.md`](docs/wsl_setup.md).

**Robot library.** 63 robots seeded from MuJoCo Menagerie (via
`robot_descriptions`) — 6 `mjlab_ready` (G1, Go1, ANYmal-C, T1, Yam,
Cartpole) + 5 Gymnasium-builtin (Hopper/Ant/Walker2d/HalfCheetah/
Humanoid) + 52 `preview_only` (available to render, training gated
until an mjlab task lands). See
[`docs/robot_library.md`](docs/robot_library.md) for how to add more.

---

## One-command demo

```bash
uv run sculpt run \
    --config examples/hopper/config.toml \
    "run forward fast without falling" \
    --iterations 10
```

That's it. The sculptor will:

1. Train Hopper-v4 with `examples/hopper/rewards/v0.py` for the configured
   step budget.
2. Roll out 4-6 eval episodes and stash behavior metrics (fall rate,
   episode length, forward velocity).
3. Call Claude Opus 4.7 to diagnose the failure mode from the metrics +
   behavior + 4 keyframes + reward contract.
4. Query the knowledge graph for techniques that address the diagnosed
   failure (by-tag walk + sentence-transformer cosine).
5. Call Claude again to rewrite `rewards/v1.py` end-to-end, citing each
   grounded edit's `arxiv_id`.
6. Validate the new reward module (imports, signature, component dict,
   paper-refs existing in the KG). One retry on schema failure.
7. `git commit`, append `CHANGELOG.md`, update `reports/provenance.json`.
8. One-line status on stdout. Repeat until `--iterations` is exhausted or
   the primary metric has not improved for 3 iterations.

Want to see the loop without burning API credits? Add `--dry-run`:
training is capped at 1000 steps, the diagnose + edit calls return canned
responses, and the full pipeline completes in under a minute. Perfect for
plumbing tests.

When a run finishes:

```bash
uv run sculpt report --config examples/hopper/config.toml --out final.mp4
```

Renders a side-by-side time-lapse (title card + iter 0 / N/2 / N panels
with primary-metric burned in) and writes `reports/final_report.md` with
top-3 impactful edits, a literature map grouped by reward component, and a
summary table of citations added per iteration.

To take a trained policy out for deployment (sim-to-real, another
codebase, a robot):

```bash
uv run sculpt export --config examples/hopper/config.toml --list   # what's exportable
uv run sculpt export --config examples/hopper/config.toml --iter 2
```

Writes `<project>/exports/policy_<name>_iter<N>.zip` — a self-contained
bundle with the raw checkpoint, best-effort ONNX + TorchScript exports of
the actor network (rsl_rl observation normalization baked in when the
checkpoint carries it), the exact reward version + env spec the iteration
trained under, the project config, metrics, and a `DEPLOY.md` loading
recipe. The same bundles are downloadable from the UI's Results tab.

---

## Architecture

```
                          behavior goal
                                 │
                                 ▼
           ┌─────────────────────────────────────────────┐
           │              sculpt run (outer)             │
           └───────┬──────────┬──────────┬───────┬──────┘
                   │          │          │       │
                   ▼          ▼          ▼       ▼
              ┌───────┐  ┌───────┐  ┌──────┐ ┌────────┐
              │ train │  │ roll- │  │ diag-│ │  edit  │
              │       │  │ out   │  │ nose │ │        │
              └───┬───┘  └───┬───┘  └──┬───┘ └────┬───┘
                  │          │         │          │
                  └──────────┴─────────┼──────────┘
                                       │
                                       ▼
                           ┌────────────────────┐
                           │  commit · CHANGELOG │
                           │  provenance · v<n>  │
                           └──────────┬──────────┘
                                      │  loop if not early-stopped
                                      ▼
                               (next iteration)

                               KG retrieval loop
                               ─────────────────
             arxiv seeds ─► ingest ─► PDFs ─► Claude extract ─┐
                                                              │
                                          Papers · Techniques │
                                          FailureModes        │
                                          RewardComponents    │
                                          Environments        │
                                                              ▼
                     ┌──────────────────┬───────────────────┐
                     │ query_techniques │ query_semantic    │
                     │ (by failure tag) │ (MiniLM cosine)   │
                     └────────┬─────────┴─────────┬─────────┘
                              └──────► diagnose ◄─┘
                                          │
                                          ▼
                              (grounded edits with paper_refs)
```

Two independent data flows, one joint:

- The **iteration loop** (top) works per-project and produces `v<n>.py`
  reward modules + per-iteration artifacts.
- The **KG retrieval** (bottom) is cross-project and persistent — ingest
  your seed papers once, reuse across every sculpt run.

The diagnoser is the join point: it pulls the current iteration's
artifacts up and the retrieved techniques down, and hands both to the
editor.

---

## Adapting to your RL stack

Sculptor is lab-agnostic. The full adapter reference with five
injection patterns is in [`docs/adapters.md`](docs/adapters.md) — this
section is a worked example so you can see the shape.

**Scenario**: you train a legged robot in Brax/MJX. Here's a sketch of
the adapter.

```python
# my_lab/brax_adapter.py
from pathlib import Path
import json
import pickle

import jax
import jax.numpy as jp
from brax.training.agents.ppo import train as ppo_train

from sculptor.adapters.base import (
    RewardContract, RolloutResult, SculptorAdapter, TrainResult,
)


class BraxQuadrupedAdapter(SculptorAdapter):
    """Sculptor adapter for a Brax environment whose __init__ accepts
    a `reward_fn` kwarg that's called per env step with JAX arrays."""

    def __init__(self, env_factory, num_envs=2048, episode_length=250):
        self._env_factory = env_factory
        self._num_envs = num_envs
        self._episode_length = episode_length

    def reward_contract(self) -> RewardContract:
        # The env's info dict publishes these scalars per step.
        return RewardContract(
            observation_space_spec=(self._env_factory(None).observation_size,),
            action_space_spec=(self._env_factory(None).action_size,),
            expected_info_keys=[
                "body_z", "body_vz", "tilt", "roll_rate", "pitch_rate",
                "foot_contacts", "height_gain", "xy_drift",
                # NOTE: only list keys your env actually publishes.
            ],
            expected_components=None,  # open-ended — editor picks names
        )

    def train(self, reward_module_path, output_dir, steps, seed):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the generated reward module (sculptor's edit stage wrote it).
        mod = _load_reward_module_by_path(reward_module_path)

        # Preflight: jit-lower the reward on zero-dummies so we fail fast
        # on numpy idioms instead of after a 30-second JAX compile.
        self._jit_preflight(mod.compute_reward)

        # Instantiate the env with Sculptor's reward_fn injected.
        env = self._env_factory(reward_fn=mod.compute_reward)

        # Hand off to Brax PPO.
        make_inference_fn, params, training_metrics = ppo_train(
            environment=env,
            num_timesteps=steps,
            num_envs=self._num_envs,
            episode_length=self._episode_length,
            seed=seed,
        )

        # Persist what Sculptor's contract requires.
        (output_dir / "checkpoint.pkl").write_bytes(pickle.dumps(params))
        (output_dir / "metrics.json").write_text(json.dumps({
            "metrics": {
                "mean_return": float(training_metrics["eval/episode_reward"]),
                "training_steps": int(steps),
                "seed": int(seed),
            },
        }))
        (output_dir / "reward_spec.json").write_text(
            json.dumps(mod.REWARD_SPEC, default=str))

        # Component means from Brax's episode metrics (your env should write
        # each component into env.state.metrics so ppo_train can aggregate).
        component_means = {
            k.replace("eval/component/", ""): float(v)
            for k, v in training_metrics.items()
            if k.startswith("eval/component/")
        }

        return TrainResult(
            checkpoint_path=output_dir / "checkpoint.pkl",
            metrics_dict={"mean_return": float(training_metrics["eval/episode_reward"])},
            component_means=component_means or {"_total": 0.0},
            logs_path=output_dir,
        )

    def rollout(self, checkpoint_path, output_dir, n_episodes):
        # Roll out n_episodes with the saved params and your env.render()
        # pipeline. Must write rollout.mp4, keyframes/, trajectory.npz,
        # behavior.json. See docs/adapters.md §rollout.
        ...
        return RolloutResult(
            video_path=output_dir / "rollout.mp4",
            keyframes_dir=output_dir / "keyframes",
            trajectory_path=output_dir / "trajectory.npz",
            n_episodes=n_episodes,
        )

    def compute_behavior_metrics(self, rollout: RolloutResult):
        # Whatever metrics are specified in config.iteration.behavior_metrics.
        return {
            "max_jump_height": ...,
            "num_takeoffs": ...,
            "mean_tilt_deg": ...,
        }

    def _jit_preflight(self, reward_fn):
        import jax
        dummy = jp.zeros(self._env_factory(None).observation_size)
        # Will raise if the generated reward uses numpy, Python if-on-traced,
        # or un-traceable control flow. Saves you a 30 s JIT compile failure.
        jax.jit(reward_fn).lower(
            dummy, jp.zeros(self._env_factory(None).action_size),
            dummy, {},
        )
```

Three things make a Brax adapter distinct from the Gymnasium reference:

1. **Reward is JAX-pure.** Sculptor's generated reward must use `jp.where`
   / `jax.lax.cond`, never Python `if` on traced values.
   `_jit_preflight` is the escape hatch — two lines of `jax.jit(...).lower()`
   save you hours of "why did PPO silently produce NaN".
2. **Reward is injected at env construction, not at `step` time.** Brax's
   training loop JITs over your env, so you need the reward to be in the
   jitted graph. Pass it into `env_factory(reward_fn=...)` and live with
   the constraint.
3. **Component means come from Brax's metrics pipeline**, not from a
   Python accumulator in a wrapper. Stash each reward component into
   `env.state.metrics` from your reward function, and Brax's PPO
   evaluator aggregates across the env batch for free.

Point `config.adapter.class` at your new adapter class and the rest of
Sculptor — diagnose, edit, provenance, time-lapse — continues to work
unchanged.

For Isaac Gym / Isaac Lab, RLlib, CleanRL, and custom loop patterns, see
[`docs/adapters.md`](docs/adapters.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) §"Contributing an adapter".

---

## What's actually happening

A deep dive into one iteration. This is what `sculpt run` does under the
hood for iteration `i`:

### 1. Training

```python
adapter.train(
    reward_module_path = project/rewards/current.py,   # file-path re-export of latest v<n>
    output_dir         = project/runs/iter_<i>/,
    steps              = config.iteration.steps_per_iter,
    seed               = base_seed + i,
)
```

`current.py` is a pure Python re-export of the latest `v<n>.py` via
`importlib.util.spec_from_file_location` — no symlinks, works
identically on Linux / macOS / Windows. The adapter is responsible for
plumbing that reward module into its stack (see
[§Adapting](#adapting-to-your-rl-stack)).

Required writes under `output_dir`: `checkpoint.*`, `metrics.json`,
`reward_spec.json`. Your adapter's failure modes here are opaque to
Sculptor — it just reads the artifacts back.

### 2. Rollout

```python
adapter.rollout(
    checkpoint_path = iter_<i>/checkpoint.*,
    output_dir      = iter_<i>/rollout/,
    n_episodes      = config.iteration.rollout_episodes,   # default 6
)
```

Writes `rollout.mp4`, `keyframes/*.png`, `trajectory.npz`, `behavior.json`.
The keyframes are 12 evenly-spaced frames from the best-return episode;
the diagnoser will send 4 of them to Claude as image inputs.

### 3. Diagnose (two-stage)

```python
diagnosis = diagnose(
    iter_dir       = iter_<i>/,
    behavior_goal  = "<your natural-language goal>",
    config         = config,
)
```

**Stage 1** (preliminary): Claude Opus 4.7 receives the behavior goal, the
current `REWARD_SPEC`, `metrics.json`, `behavior.json` (keyed by the
adapter's behavior-metric vocabulary — which varies per domain), 4
keyframes, and the reward contract. It returns a strict JSON with:

- `failure_modes`: one or more from a fixed stack-agnostic vocabulary:
  `reward_hacking`, `static_equilibrium`, `premature_termination`,
  `sparse_reward`, `reward_saturation`, `component_imbalance`, `none`.
- `evidence`: 2-4 sentences citing specific numbers from the inputs.
- `confidence`: float in `[0, 1]`.

**Stage 2** (grounded): the KG is queried via
`query_techniques(failure_modes, domain_filter=config.kg.environment_tag)`
(graph walk over `Technique --[ADDRESSES]--> FailureMode` edges, optionally
filtered by the introducing paper's environment tag) UNION-ed with
`query_semantic(behavior_goal)` (all-MiniLM-L6-v2 cosine over Technique
descriptions). Top 6 hits are rendered as markdown in the Stage-2 prompt.
Claude returns proposed edits, each with:

- `target_term`: a `REWARD_SPEC.hyperparameters` key or a new snake_case
  name for an added term.
- `operation`: `increase` / `decrease` / `add` / `remove` / `clip` / `gate`
  / `replace` / `normalize`.
- `rationale`: why. Novel edits must begin `"novel."`.
- `suggested_value`: numeric or short formula.
- `paper_refs`: list of arxiv_ids, or `[]` if novel.
- `requires_env_extension`: true if the ideal edit would need a field
  outside `expected_info_keys` — in which case the editor skips it and
  logs it as a deferred suggestion.

### 4. Apply edits

```python
new_reward_path = apply_edits(
    current_reward_path = project/rewards/v<i>.py,
    diagnosis           = diagnosis,
    new_iter_id         = "v<i+1>",
    reward_contract     = adapter.reward_contract(),
)
```

**Pre-flight** (no API call):

- Every `paper_refs[*]` arxiv_id must exist in the KG. If not → raise.
- Every identifier in `suggested_value` (extracted via `ast.parse(mode="eval")`,
  kwarg names filtered out) must be in `expected_info_keys ∪ current-components ∪
  current-hyperparameters ∪ math-allowlist`. If not → raise.
- Edits flagged `requires_env_extension=True` split off; they're listed in
  the prompt as "DO NOT apply — record in REWARD_SPEC.description".

**LLM call**: Opus 4.7 rewrites the reward module end-to-end. Returns the
full new `v<i+1>.py` source.

**Post-flight**:

- Import the new module. Must define `compute_reward` and `REWARD_SPEC`.
- Call `compute_reward(zero_state, zero_action, zero_next_state, zero_info)`.
  Must return `(numeric, dict[str, numeric])`.
- If `reward_contract.expected_components is not None`, returned keys must
  be a subset.
- `REWARD_SPEC.references[*].arxiv_id` must all exist in the KG.
- `REWARD_SPEC.parent_hash` must match `sha256(v<i>.py)[:16]`.

One retry on failure with the validation errors appended. Second failure
raises.

Finally, `rewards/current.py` is rewritten to re-export from `v<i+1>.py`.

### 5. Commit, changelog, provenance

`git commit` in the project dir (if it's a repo) with message
`"iter <i>: <failure_modes> [<n> edits]"`. Then:

- `CHANGELOG.md`: append a section with the iteration's metrics, failure
  modes, and per-edit rationale + paper_refs.
- `reports/provenance.json`: upsert per-target-term entries
  `[{arxiv_id, citation, iter_introduced, how_used, still_active}, ...]`.
  The `still_active` flag flips to `False` when the target_term is no
  longer in the current reward.

### 6. Early stop

After every iteration, `max(metric[-3:]) ≤ max(metric[:-3])` triggers
early stop. Avoids wasting compute when the sculptor hit a plateau.

### 7. Report

```bash
sculpt report --config … --out final.mp4
```

Selects iterations `1, N/2, N`, pre-renders per-panel labels + title
card as PNGs via PIL (sidestepping ffmpeg drawtext font-path escaping on
Windows), and assembles the mp4 in a single ffmpeg filtergraph pass.
Emits `reports/final_report.md` with:

- Starting vs ending behavior description (from the first and last
  `behavior.json`).
- Top-3 most impactful edits ranked by `metric[i+1] - metric[i]`.
- Literature map: every active reward component → papers that influenced
  it, from `provenance.json`.
- Candidate novel contributions: every applied edit with `paper_refs=[]`
  and rationale starting `novel.`.
- Summary table: `iter | primary_metric | num_references_added | num_novel_edits`.

---

## Project layout

```
RewardSculptor/
├── sculptor/              # the package
│   ├── sculpt.py          # orchestrator (sculpt_run, sculpt_init)
│   ├── diagnose.py        # two-stage LLM diagnoser
│   ├── edit.py            # reward-module rewriter + validator
│   ├── timelapse.py       # final.mp4 + final_report.md builder
│   ├── cli.py             # `sculpt` entry point (typer)
│   ├── adapters/
│   │   ├── base.py        # SculptorAdapter ABC + RewardContract
│   │   └── gym_sb3.py     # reference Gymnasium + SB3 adapter
│   ├── kg/
│   │   ├── schema.py      # typed node/edge dataclasses
│   │   ├── store.py       # sqlite-backed graph store
│   │   ├── ingest.py      # arxiv → PDF → text (no LLM)
│   │   ├── extract.py     # LLM-powered entity extraction
│   │   ├── query.py       # graph walk + semantic search + cite()
│   │   └── viz.py         # pyvis HTML viz (see `sculpt kg viz`)
│   └── prompts/           # all LLM system prompts (tunable per-project)
│       ├── diagnose_preliminary.md
│       ├── diagnose_grounded.md
│       ├── edit_rewriter.md
│       └── kg_extract.md
├── examples/
│   └── hopper/            # end-to-end Hopper-v4 demo project
│       ├── config.toml
│       ├── rewards/v0.py
│       ├── kg_seeds.yml
│       └── reports/...    # populated by `sculpt run`
├── docs/
│   └── adapters.md        # the adapter reference (READ THIS if you're integrating)
├── tests/                 # 63 tests, 1 skipped (jax-only parity test)
└── scripts/               # verification scripts (phase1_smoke, diagnose_demo, etc.)
```

The prompts in `sculptor/prompts/*.md` are loaded at runtime. Override any
of them for a specific project by setting `SCULPTOR_PROMPTS_DIR=/path/to/your/prompts`.

---

## Commands

```bash
sculpt init <dir> --adapter <name>           # scaffold a new project
sculpt run <behavior> --config <cfg> [opts]  # drive the inner loop
sculpt report --config <cfg> --out final.mp4 # time-lapse + markdown report

sculpt kg ingest <seeds.yml>                 # fetch arxiv papers
sculpt kg extract --all                      # LLM-extract entities from ingested papers
sculpt kg list-papers                        # list ingested papers
sculpt kg list-techniques                    # list extracted techniques
sculpt kg stats                              # node/edge/embedding counts
sculpt kg viz --out kg.html                  # interactive graph (see below)
```

`sculpt run` flags:

| Flag              | Default | Behavior                                                                                  |
| ----------------- | :-----: | ----------------------------------------------------------------------------------------- |
| `--iterations N`  |   10    | Iterations this invocation runs.                                                          |
| `--resume`        |  false  | Start after the highest `v<n>.py` in `rewards/`.                                          |
| `--no-kg`         |  false  | Skip KG queries — diagnoser sees an empty literature context. Ablation mode.              |
| `--dry-run`       |  false  | Bypass all LLM calls; cap training at 1000 steps. Under a minute end-to-end.              |

---

## Knowledge graph

The sculptor gets its "street smarts" from a KG of papers you ingest once
per project:

```bash
uv run sculpt kg ingest examples/hopper/kg_seeds.yml
uv run sculpt kg extract --all          # requires ANTHROPIC_API_KEY
uv run sculpt kg stats
```

Seed files are plain YAML:

```yaml
papers:
  - arxiv_id: "1707.06347"
    title: "Proximal Policy Optimization Algorithms"
    rationale: "Training algorithm the Hopper adapter uses."
  - arxiv_id: "1801.00690"
    title: "DeepMind Control Suite"
    rationale: "Bounded reward structure + tolerance kernels."
```

Pick seeds relevant to **your** domain — Sculptor is not hard-coded to
locomotion. Manipulation labs ingest Dex-Net / object-rearrangement
papers; LLM-RL labs ingest RLHF / reward-model survey papers.

Visualize the ingested KG:

```bash
uv run sculpt kg viz --config examples/hopper/config.toml --out kg.html
open kg.html
```

Nodes are colored by type (paper / technique / failure mode / reward
component / environment / result). Nodes cited in the current project's
`reports/provenance.json` get a **gold halo** — these are the specific
papers and techniques that shaped the final reward.

---

## Tests

```bash
uv run pytest tests/ -v
```

63 tests covering: reward-contract validation, Diagnosis JSON parsing,
edit pre-flight + post-flight validation, KG store round-trip, query
fixtures, adapter-contract compliance for `GymSB3Adapter`, sculpt
orchestration (scaffold, dry-run, early-stop, resume), and the time-lapse
builder. All API calls are mocked; GPU tests are skipped on machines
without CUDA.

---

## What's next

Two things the authors are thinking about:

1. **Cross-project KG sharing.** A public community KG of well-annotated
   RL papers keyed by failure mode, so every new adopting lab doesn't
   cold-start extraction.
2. **Adapter contributions.** Isaac Gym, Brax, RLlib, CleanRL — we have
   stub adapters sketched in `docs/adapters.md` but production-quality
   adapters are where lab adoption actually lives. See
   [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT.
