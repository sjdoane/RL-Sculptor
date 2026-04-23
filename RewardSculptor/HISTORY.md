# Reward Sculptor — Working History

A complete log of what has been built, how it's organized, and the state
of everything as of **2026-04-20**. Read this alongside
[`README.md`](README.md), [`docs/adapters.md`](docs/adapters.md), and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## What Reward Sculptor is

An autonomous agent that iterates on reinforcement-learning reward
functions, grounded in a literature-backed knowledge graph. Given a
behavior goal and a starter reward, it trains, rolls out, diagnoses the
failure mode, queries a KG of RL papers for techniques that address that
failure, rewrites the reward module with citations, and repeats —
producing a time-lapse video, a CHANGELOG, and a provenance graph along
the way.

Two loops, one system:

1. **Iteration loop** — per-project. `train → rollout → diagnose →
   apply_edits → git commit`. Drives reward versions `v0.py → v1.py →
   v<N>.py` plus one iter dir per cycle under `<project>/runs/`.
2. **Knowledge graph** — persistent, lab-shared. arxiv seeds → PDFs →
   Claude-extracted Techniques / FailureModes / RewardComponents /
   Environments. Queried by the diagnoser via failure-mode graph walk
   + sentence-transformer cosine semantic search.

Both loops join at the diagnoser, which produces proposed reward edits
with `arxiv_id` citations for each grounded change.

The project is **lab-agnostic**: a stack-specific adapter bridges
Sculptor to your RL framework. The reference adapter is
`GymSB3Adapter` (Gymnasium + Stable-Baselines3); Isaac Gym, Brax, RLlib,
CleanRL, and custom-loop patterns are documented in `docs/adapters.md`.

---

## Project location

`C:\Users\SamJD\OneDrive\Desktop\Projects\RewardSculptor`

Git: scoped to `examples/hopper/` only (that subdir was `git init`ed
for the sculpt-run demo). The RewardSculptor project root itself is not
a git repo by design — it's a working scaffold, and the interesting
commits live inside each sculpted project.

Python: **miniconda 3.13.5** (pinned in `.python-version`) via `uv venv
--python C:/Users/SamJD/miniconda3/python.exe`. See "Gotchas" below for
why.

---

## Build timeline

Each prompt produced the artifacts listed. Everything cumulative —
nothing was rolled back.

### Prompt 0 — Planning
Read the AME456 quadruped repo, designed the sculptor seam, and wrote a
plan (no code). Identified the reward-injection seam in the quadruped
env, proposed the `sculptor/` directory layout, KG query shapes, deps
list, and risks.

### Prompt 1 — Swappable reward v0 (AME456 quadruped)
- Created [`sculptor/__init__.py`](sculptor/__init__.py) +
  [`sculptor/reward.py`](sculptor/reward.py) with a verbatim port of the
  in-env reward from `quadruped_mjx_env.py:579–699` as `compute_reward
  (state, action, next_state, info) -> (reward, components)` + a
  `REWARD_SPEC` dict.
- Wired the AME456 env (`quadruped_mjx_env.py`) to import
  `sculptor.reward.compute_reward` via a sys.path shim.
- [`tests/test_reward_parity.py`](tests/test_reward_parity.py):
  **bit-exact parity** across 100 steps (max `|diff|` = 0.0, tol 1e-6).

### Prompt 2 — Package scaffold + GymSB3Adapter (overnight run)
- uv-initialized the package: [`pyproject.toml`](pyproject.toml),
  `.python-version`, `uv.lock`, `README.md` (stub).
- [`sculptor/adapters/base.py`](sculptor/adapters/base.py) —
  `SculptorAdapter` ABC, `TrainResult`, `RolloutResult`,
  `RewardContract`, `load_adapter()`, 95-line contributor docstring.
- [`sculptor/adapters/gym_sb3.py`](sculptor/adapters/gym_sb3.py) —
  `GymSB3Adapter` + `RewardOverrideWrapper` + `_write_mp4` (subprocess
  ffmpeg via imageio-ffmpeg fallback) + `compute_behavior_metrics` for
  Hopper.
- [`sculptor/cli.py`](sculptor/cli.py) — typer skeleton (`init`, `run`,
  `resume`, `viz`).
- [`sculptor/rewards/__init__.py`](sculptor/rewards/__init__.py),
  [`sculptor/kg/__init__.py`](sculptor/kg/__init__.py) — placeholder
  packages.
- [`examples/hopper/config.toml`](examples/hopper/config.toml),
  [`examples/hopper/rewards/v0.py`](examples/hopper/rewards/v0.py) —
  canonical Hopper reward (`forward_velocity + alive_bonus - ctrl_cost`).
- [`scripts/phase1_smoke.py`](scripts/phase1_smoke.py) +
  [`scripts/phase2_smoke.py`](scripts/phase2_smoke.py) — verification
  scripts.
- Phase 1 gate: 20k-step Hopper-v4 training, `mean_return=+71.08`,
  TrainResult shape validated. Phase 2 gate: 6 eval eps → 344 KB mp4,
  12 keyframes, valid trajectory.npz + behavior.json.
- [`STATUS.md`](STATUS.md), [`NOTES.md`](NOTES.md): overnight-run
  artifacts documenting every default chosen.

**Key decision**: `imageio-ffmpeg` added to the dep set beyond the
originally-specified list because the Windows demo needs a bundled
ffmpeg binary.

### Prompt 3 — KG substrate (no LLM yet)
- [`sculptor/kg/schema.py`](sculptor/kg/schema.py) — dataclasses for
  `Paper`, `Technique`, `FailureMode`, `RewardComponent`, `Environment`,
  `Result` + `Edge` with `Relation` enum (CITES, INTRODUCES, ADDRESSES,
  USES, EVALUATES_ON, REPORTS, IMPROVES_OVER). Natural IDs
  (`paper:1707.06347`, `technique:<slug>`, …).
- [`sculptor/kg/store.py`](sculptor/kg/store.py) — sqlite-backed
  `SculptorKG` class. Two tables (`nodes`, `edges`), upsert semantics,
  `find_nodes`, `neighbors`, `stats`. Default DB path
  `./kg/graph.db`, overrideable via `SCULPTOR_KG_PATH`.
- [`sculptor/kg/ingest.py`](sculptor/kg/ingest.py) — arxiv API for
  metadata (with `_fetch_arxiv_metadata` fallback to seed-YAML on HTTP
  429) + direct CDN download of PDFs via urllib with 60s timeout +
  pypdf text extraction + heuristic abstract/conclusion split.
  Idempotent.
- CLI: `sculpt kg list-papers`, `list-techniques`, `stats`.
- [`examples/hopper/kg_seeds.yml`](examples/hopper/kg_seeds.yml) —
  4 seeds: PPO (1707.06347), OpenAI Gym (1606.01540), Deep RL That
  Matters (1709.06560), DeepMind Control Suite (1801.00690).
- 9 offline tests for schema + store round-trip + ingest heuristics.
- Live ingest verified: 4/4 papers fetched, PDFs + `.txt` sidecars in
  `kg/pdfs/`, re-run is idempotent (`ingested=0 already_present=4`).

### Prompt 4 — KG extract + query (LLM-driven)
- [`sculptor/kg/extract.py`](sculptor/kg/extract.py) — Pydantic-typed
  extraction payload for Techniques / FailureModes / RewardComponents /
  Environments + four relation lists. `messages.parse()` with
  Claude Opus 4.7, adaptive thinking, prompt caching, **one retry on
  parse failure** (as a second user turn — 4.7 forbids assistant
  prefill). `_materialize` upserts nodes + edges with cross-payload
  lookup so later papers can cite earlier papers' entities. Idempotent
  on `Paper.extracted=True`.
- [`sculptor/kg/query.py`](sculptor/kg/query.py) — `TechniqueMatch`
  dataclass, `query_techniques(failure_modes, domain_filter, top_k)`
  (graph walk + fuzzy slug resolution), `query_semantic(text, top_k)`
  (all-MiniLM-L6-v2 cosine over cached embeddings), `cite(arxiv_id)`.
- [`sculptor/kg/store.py`](sculptor/kg/store.py) — added
  `node_embeddings` table + `set_embedding` / `get_embedding` /
  `iter_embeddings` / `has_embedding` / `count_embeddings`; cascade
  delete with parent node.
- CLI: `sculpt kg extract --all`.
- 9 offline tests for materialize / query / cite + a monkey-patched
  `query_semantic` that doesn't load the real model.
- **Live extraction was blocked** in this session (Claude Code OAuth
  token ≠ public API auth, and `ANTHROPIC_API_KEY` wasn't set yet).
- [`scripts/kg_seed_demo.py`](scripts/kg_seed_demo.py) — hand-written
  extraction payloads (what a careful reader would pull from the four
  seed papers) routed through the same `_materialize` codepath. Used
  to populate the KG for downstream diagnose / query demos.

### Prompt 5 — `.env` file + dotenv loader
- [`.env`](.env) — template with `ANTHROPIC_API_KEY=`.
- [`.env.example`](.env.example) — committed placeholder.
- [`.gitignore`](.gitignore) — ignores `.env`, `.venv`, `kg/graph.db`,
  `kg/pdfs/`, `runs/`, Python caches, editor dirs.
- After the key was filled, [`sculptor/__init__.py`](sculptor/__init__.py)
  was updated (Prompt 6) to auto-load `.env` and — critically — treat
  empty-string env vars as unset so Claude Code's default
  `ANTHROPIC_API_KEY=""` doesn't block the `.env` value.

### Prompt 6 — `diagnose.py` (two-stage LLM diagnoser)
- [`sculptor/diagnose.py`](sculptor/diagnose.py) — two-stage flow:
  - **Stage 1** (preliminary): behavior goal + REWARD_SPEC +
    metrics.json + behavior.json + 4 keyframe PNGs + reward_contract +
    behavior-metric vocab (from `config.iteration.behavior_metrics`)
    → failure_modes from a **fixed stack-agnostic vocab**
    (`reward_hacking`, `static_equilibrium`, `premature_termination`,
    `sparse_reward`, `reward_saturation`, `component_imbalance`,
    `none`), evidence, confidence.
  - **Stage 2** (grounded): stage-1 result + top-6 union of
    `query_techniques(..., domain_filter=cfg.kg.environment_tag)` and
    `query_semantic(behavior_goal)` + all original inputs + reward
    contract → proposed_edits with `paper_refs: list[arxiv_id]`.
  - `claude-opus-4-7` + adaptive thinking + `messages.parse` + one
    retry. Writes `iter_dir/diagnosis.json`.
  - `requires_env_extension` flag on each edit so the diagnoser can
    propose ideas that need env extension without emitting ungrounded
    formulas.
- [`scripts/diagnose_demo.py`](scripts/diagnose_demo.py) — fresh 20k
  Hopper train + rollout + diagnose for the behavior goal "run forward
  as fast as possible without falling". Live output correctly
  identified `premature_termination + component_imbalance` with 2
  paper-cited edits and 1 novel edit.
- 2 offline tests (mocked client) + dotenv wiring.

### Prompt 7 — `edit.py` (LLM reward rewriter + validator)
- [`sculptor/edit.py`](sculptor/edit.py) — `apply_edits(current_reward_
  path, diagnosis, new_iter_id, reward_contract) -> Path`. Three
  phases:
  - **Pre-flight** (no API call): every `paper_refs[*]` arxiv_id must
    exist in KG; every identifier in `suggested_value` (extracted via
    `ast.parse(mode="eval")`) must be in `expected_info_keys ∪
    current-components ∪ current-hyperparameters ∪ math-allowlist`.
    Edits flagged `requires_env_extension=True` split off to
    `plan.deferred_edits` and are listed (but not applied) in the
    prompt.
  - **LLM call**: Opus 4.7 rewrites the reward module end-to-end.
  - **Post-flight**: import the generated module, call
    `compute_reward` on zero-dummies, enforce `expected_components`
    subset when declared, check REWARD_SPEC has
    `version/parent_hash/author/description/hyperparameters/references`,
    verify every reference arxiv_id is in the KG.
  - One retry on any failure; second failure raises
    `EditValidationError`. Writes `<rewards_dir>/<new_iter_id>.py` +
    regenerates `<rewards_dir>/current.py` as a by-file-path re-export
    (no symlinks, works cross-platform).
- Stage-2 diagnose prompt updated with the **grounded-field rule** and
  instructions to use `requires_env_extension=true` rather than
  propose ungrounded formulas.
- [`scripts/edit_demo.py`](scripts/edit_demo.py) — hand-built
  Diagnosis (1 DM-Control-cited edit + 1 novel edit) → live v1 with
  correct `references` block + preserved compute_reward signature +
  `parent_hash=sha256(v0.py)[:16]`.
- 12 offline tests covering every validator branch.

### Prompt 8 — `sculpt.py` + CLI orchestrator
- [`sculptor/sculpt.py`](sculptor/sculpt.py) — `sculpt_run(config_path,
  behavior_goal, *, iterations, resume, no_kg, dry_run)` drives the
  full inner loop: per-iter `adapter.train → adapter.rollout → diagnose
  → apply_edits → CHANGELOG append → provenance update → git commit`.
  `sculpt_init(project_dir, adapter)` scaffolds a new project with
  config.toml + rewards/v0.py + kg_seeds.yml + .gitignore + `git init`
  + initial commit.
  - `--dry-run`: bypasses both LLM calls via canned Diagnosis + regex
    reward-bump; caps training at 1000 steps.
  - `--no-kg`: threads `skip_kg=True` into diagnose for ablations.
  - **Early stop**: `max(metric[-3:]) ≤ max(metric[:-3])`.
- CLI: `sculpt run <behavior> --config <cfg> [--iterations N]
  [--resume] [--no-kg] [--dry-run]`, `sculpt init <dir> --adapter
  <name>`.
- `CHANGELOG.md`: auto-appended per iter (reward before/after, metric
  delta, failure modes, evidence, every edit with rationale +
  paper_refs + deferred flag).
- `reports/provenance.json`: per-target-term entries with
  `{arxiv_id, citation, iter_introduced, how_used, still_active}`.
  `still_active` flips to False when the term is removed.
- `reports/metric_history.json`: per-iter primary_metric values (for
  delta rendering + early-stop).
- 10 offline tests (+ a stub adapter so no MuJoCo is pulled in).
- **Live verification**: 3-iter Hopper run at `steps_per_iter=10000`
  produced 3 per-iter git commits, a well-formed CHANGELOG, populated
  provenance, and the diagnose loop correctly diagnosed
  `premature_termination + reward_hacking` each iter. Dry-run 3-iter
  completed in 49s.

### Prompt 9 — `timelapse.py` + `sculpt report`
- [`sculptor/timelapse.py`](sculptor/timelapse.py) — `build_report
  (config_path, out_mp4) -> ReportResult`. Selects iterations
  `1, N/2, N` (zero-indexed + deduped), renders per-panel labels + a
  1440×480 title card as PNGs via PIL (sidestepping ffmpeg's drawtext
  font-path escaping on Windows), then composes the mp4 in a **single
  ffmpeg filter-graph pass**: scale each panel to 480×480 → overlay
  label PNG at bottom → hstack → prepend 4-second title still. Uses
  bundled `imageio-ffmpeg` binary if system ffmpeg is absent. Rollouts
  < 2 KB are treated as sentinels and dropped.
- Emits `<project>/reports/final_report.md` with:
  - Starting vs ending behavior description (from first / last
    `behavior.json`).
  - **Top 3 most impactful edits** ranked by `metric[i+1] − metric[i]`.
  - **Literature map**: per active reward term → citing papers from
    `provenance.json`.
  - **Candidate novel contributions**: applied edits with
    `paper_refs=[]` AND rationale starting with `"novel."` (strict
    spec-compliant filter).
  - **Summary table**: `iter | primary_metric | num_references_added
    | num_novel_edits`.
  - Pointer to the full CHANGELOG.
- CLI: `sculpt report --config <cfg> --out final.mp4`.
- 5 offline tests.
- **Live verification**: ran on the 3-iter Hopper run. Produced
  441 KB `final.mp4` + 4 KB `final_report.md`. Panels from iters
  [0, 1, 2] with `mean_return=+67.455 → +128.183 → +101.901`.

### Prompt 10 — Production docs + kg viz + prompt extraction
- Extracted all Claude system prompts to
  [`sculptor/prompts/*.md`](sculptor/prompts/):
  `diagnose_preliminary.md`, `diagnose_grounded.md`, `edit_rewriter.md`,
  `kg_extract.md`. [`sculptor/prompts/__init__.py`](sculptor/prompts/__init__.py)
  exports `load_prompt(name)` with a `SCULPTOR_PROMPTS_DIR` env-var
  override. `pyproject.toml` `[tool.hatch.build.targets.wheel.force-
  include]` ships the `.md` files in the wheel.
- [`docs/adapters.md`](docs/adapters.md) — exhaustive adapter
  reference. Every method with input/output guarantees, RewardContract
  field-by-field, **five reward-injection patterns** (Gymnasium
  wrapper, Isaac Gym method patching, Brax `reward_fn`, RLlib
  env_creator, custom loop callback), validation checklist.
- [`README.md`](README.md) — production-grade. 60-second pitch, one-
  command demo, dual-loop ASCII architecture diagram, "Adapting to
  your RL stack" with a complete Brax / MJX worked example, "What's
  actually happening" deep dive.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — explicit invitation for
  external labs to contribute Isaac Gym / Brax / RLlib / CleanRL /
  custom adapters. Five-step checklist.
- [`sculptor/kg/viz.py`](sculptor/kg/viz.py) — interactive pyvis graph
  with typed color palette (Paper / Technique / FailureMode /
  RewardComponent / Environment / Result), **gold halo** for nodes in
  the current project's `provenance.json` with `still_active=True`,
  rich HTML tooltips (descriptions, authors, abstract, tags,
  formulas, edge evidence), floating legend.
- CLI: `sculpt kg viz --config <cfg> --out <file.html>`.
- 4 offline viz tests.
- `sculptor/kg/store.py` — added `all_edges()` iterator for the viz.
- **Live verification**: [`examples/hopper/reports/kg.html`](examples/hopper/reports/kg.html)
  rendered (**736 KB, 23 nodes, 21 edges, 1 active — DM Control paper
  gold**). Full pytest: **62 passed, 1 skipped** in 7.3s. Dry-run
  3-iter: **50.97s** end-to-end (under the 60s target).

---

## Current code organization

```
RewardSculptor/
├── .env                         # ANTHROPIC_API_KEY (gitignored)
├── .env.example                 # committed template
├── .gitignore
├── .python-version              # miniconda 3.13.5
├── pyproject.toml               # uv-managed; hatchling; ships prompts/*.md
├── uv.lock
├── README.md                    # production-grade
├── CONTRIBUTING.md              # "Contributing an adapter" invitation
├── HISTORY.md                   # (this file)
├── NEW_CONTEXT_PROMPT.md        # bootstrap for a new Claude session
├── STATUS.md                    # Prompt 2 overnight run log
├── NOTES.md                     # defaults chosen during overnight
│
├── sculptor/                    # the package
│   ├── __init__.py              # .env autoload; empty-env-var handling
│   ├── cli.py                   # typer: init / run / resume / viz / report + kg sub-app
│   ├── sculpt.py                # sculpt_run + sculpt_init orchestrator
│   ├── diagnose.py              # two-stage LLM diagnoser
│   ├── edit.py                  # LLM reward rewriter + pre/post validator
│   ├── timelapse.py             # final.mp4 + final_report.md builder
│   ├── reward.py                # LEGACY v1 quadruped reward (AME456 import target)
│   ├── rewards/                 # shared primitives placeholder
│   ├── prompts/                 # system prompts, runtime-loaded
│   │   ├── __init__.py
│   │   ├── diagnose_preliminary.md
│   │   ├── diagnose_grounded.md
│   │   ├── edit_rewriter.md
│   │   └── kg_extract.md
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py              # SculptorAdapter ABC + dataclasses + load_adapter
│   │   └── gym_sb3.py           # reference adapter
│   └── kg/
│       ├── __init__.py
│       ├── schema.py            # typed nodes/edges + Relation enum
│       ├── store.py             # sqlite + embeddings + all_edges
│       ├── ingest.py            # arxiv → PDF → text (no LLM)
│       ├── extract.py           # LLM entity extraction
│       ├── query.py             # graph walk + semantic + cite
│       └── viz.py               # pyvis interactive graph
│
├── examples/
│   ├── hopper/                  # end-to-end reference project (git repo)
│   │   ├── config.toml          # steps_per_iter=10000 (demo bound)
│   │   ├── kg_seeds.yml
│   │   ├── CHANGELOG.md         # populated by sculpt run
│   │   ├── rewards/
│   │   │   ├── __init__.py
│   │   │   ├── v0.py            # forward_velocity + alive_bonus - ctrl_cost
│   │   │   ├── v1.py / v2.py / v3.py  # sculpt-generated
│   │   │   └── current.py       # re-export of latest v<n>.py
│   │   ├── runs/iter_{0,1,2}/   # checkpoint, metrics, rollout/, diagnosis.json
│   │   └── reports/
│   │       ├── final.mp4        # 441 KB — time-lapse
│   │       ├── final_report.md
│   │       ├── kg.html          # 736 KB — pyvis graph
│   │       ├── metric_history.json
│   │       └── provenance.json
│   └── hopper_test/             # Prompt 10 `sculpt init` demo
│
├── docs/
│   └── adapters.md              # lab-contributor reference
│
├── tests/                       # 62 passed, 1 skipped
│   ├── __init__.py
│   ├── test_adapter_contract.py   # 6 tests — GymSB3Adapter contract
│   ├── test_diagnose.py           # 2 tests — two-stage diagnose (mocked)
│   ├── test_edit.py               # 12 tests — edit pre/post-flight
│   ├── test_kg.py                 # 9 tests — schema + store round-trip
│   ├── test_kg_query.py           # 9 tests — materialize + query
│   ├── test_kg_viz.py             # 4 tests — viz smoke
│   ├── test_load_adapter.py       # 4 tests — config loader
│   ├── test_sculpt.py             # 10 tests — orchestrator
│   ├── test_timelapse.py          # 5 tests — report builder
│   └── test_reward_parity.py      # SKIPPED via importorskip("jax")
│
├── scripts/                     # runnable verification + demos
│   ├── phase1_smoke.py          # Prompt 2 gate
│   ├── phase2_smoke.py          # Prompt 2 gate
│   ├── kg_seed_demo.py          # Prompt 4 — hand-seeded KG demo
│   ├── diagnose_demo.py         # Prompt 6 — live Hopper diagnose
│   └── edit_demo.py             # Prompt 7 — live apply_edits
│
├── kg/                          # local KG store (gitignored content)
│   ├── graph.db                 # sqlite; 23 nodes, 21 edges
│   └── pdfs/                    # 4 arxiv PDFs + .txt sidecars
│
├── runs/                        # artifacts from top-level script runs
│   └── iter_000/                # phase1_smoke + diagnose_demo outputs
│
└── .venv/                       # uv-managed Python env
```

---

## Verification state

### Tests
```
$ uv run pytest tests/
62 passed, 1 skipped, 7 warnings in 7.3s
```
The skip is `test_reward_parity.py` which `importorskip`s JAX (not
installed in this venv; a legacy test for the AME456 quadruped reward).

### Live run artifacts
- Dry-run 3 iters: **50.97s end-to-end** (target: <60s).
- Live 3-iter Hopper: **~5 min** total (25s train + 10s rollout +
  35s diagnose + 30s edit per iter). Final `mean_return` trajectory
  `+67.5 → +128.2 → +101.9`. All 3 iter commits, populated
  CHANGELOG, 5 provenance entries, v1/v2/v3 reward modules, final.mp4
  (441 KB), final_report.md (4 KB), kg.html (736 KB).

### KG state
23 nodes (4 Papers, 6 Techniques, 6 FailureModes, 2 RewardComponents,
5 Environments), 21 edges (6 INTRODUCES, 7 ADDRESSES, 2 USES,
6 EVALUATES_ON). Entities were hand-seeded via `scripts/kg_seed_demo.py`
because **live LLM extraction (`sculpt kg extract --all`) was not
executed in this session** — all other live LLM paths (diagnose, edit)
*were* exercised, so extraction should work once run against the 4
ingested PDFs with the current API key.

---

## Known limitations, gotchas, and worth-exploring items

### Gotchas baked into the code
1. **Claude Code's env exports `ANTHROPIC_API_KEY=""`** (empty string),
   which would otherwise override the `.env` value. `sculptor/__init__.py`
   treats empty-string env vars as unset. Works end-to-end; don't undo.
2. **Windows cp1252 console**: `print()` calls must avoid Unicode
   arrows (`→`, `≤`). Use ASCII (`->`, `<=`). Fixed in scripts/demo
   output.
3. **OneDrive file locks**: uv add sometimes fails to remove the
   `reward_sculptor-0.1.0.dist-info` dir. Use `UV_LINK_MODE=copy` to
   work around.
4. **uv's Python 3.11 has `_ssl.pyd` blocked by Windows Application
   Control** on this machine. Venv is pinned to miniconda 3.13.5 via
   `.python-version`. Don't switch back to 3.11 without re-testing.
5. **arxiv API rate-limits aggressively** after bursts. Ingest has
   fallback: direct PDF CDN URL + seed-YAML metadata when the API
   returns 429. Re-running extracts wait out the rate limit.
6. **Prompts live in `sculptor/prompts/*.md`**, loaded at import time
   via `load_prompt(name)`. Per-project override: set
   `SCULPTOR_PROMPTS_DIR=/path/to/custom/prompts`.
7. **`examples/hopper/config.toml` has `steps_per_iter=10000`** as a
   demo bound. Real Hopper training wants 50000+. The
   `b36ef0f` commit in the hopper project's git history documents
   this.
8. **`sculptor/reward.py` is legacy** — the AME456 quadruped's v1
   reward (with the action_smoothness fix, see the Run 15 post-mortem
   mentioned in user memory). The AME456 env imports from this path;
   deleting it would break that external project. `test_reward_parity.py`
   tests it but skips unless JAX is installed.

### Known limitations
1. **`sculpt kg extract --all` has not been live-executed** in this
   session. Every other LLM path (diagnose, edit) was. The
   extraction-side code is the same pattern and has offline tests, but
   a first live run would confirm the arxiv API + Opus 4.7 combination
   works for all 4 seed papers.
2. **The diagnoser sometimes skips the `"novel."` prefix** on rationales
   for edits it emits with `paper_refs=[]`. The report's "Candidate
   novel contributions" section uses a strict filter
   (`paper_refs=[] AND rationale.startswith("novel.")`) so these
   non-prefixed edits get counted in the Summary table as novel but
   not in the candidates section. By design — makes the diagnoser's
   instruction-follow miss visible — but tighten if needed.
3. **The torso_angle / grounded-field issue** revealed in Prompt 6
   (diagnoser proposed `tolerance(torso_angle, …)` when `torso_angle`
   isn't in the Hopper contract's `expected_info_keys`) is now caught
   by edit.py's pre-flight validator AND the diagnoser's updated Stage-2
   prompt. But the diagnoser *still occasionally emits such edits* —
   the prompt has it set `requires_env_extension=true` instead, which
   works.
4. **Provenance's `still_active` check** reads only REWARD_SPEC.hyper
   parameters of the latest reward (`_alive_reward_keys`). Doesn't
   probe the components dict. Fine for current behavior (edits target
   hyperparameter names) but would need expansion if edits start
   adding components without corresponding weights.
5. **No CI.** pytest runs locally; there's no GitHub Actions workflow.

### Worth-exploring next
1. **Run `sculpt kg extract --all` live** to replace the hand-seeded
   KG entities with real LLM-extracted ones. Compare coverage; if the
   live extraction adds useful entities that the hand-seed missed, the
   literature map in `final_report.md` gets richer.
2. **Brax / MJX adapter** for the AME456 quadruped. `docs/adapters.md`
   has the sketch; actually implementing it would (a) validate the
   adapter contract against a non-Gymnasium stack, (b) close the loop
   on the project's original goal (quadruped reward iteration).
3. **Expanded failure-mode vocabulary**. The 7-label enum is
   intentionally stack-agnostic. A manipulation-domain or LLM-RL
   adapter might want domain-specific labels (`object_ejection`,
   `refusal_to_commit`, etc.). CONTRIBUTING.md discusses the
   upstreaming policy.
4. **Cross-project KG sharing**. The current KG is per-project
   (`kg/graph.db` in cwd). A public curated KG would avoid every
   adopting lab cold-starting extraction.
5. **pyvis → pyvis + click navigation into the paper PDFs** (drop the
   PDF URL in the node tooltip). Tiny polish.

---

## Commands cheat-sheet

```bash
# Install + verify
uv sync
uv run pytest tests/ -v

# Full CLI
uv run sculpt --help
uv run sculpt init <dir> --adapter gym_sb3
uv run sculpt run <behavior> --config <cfg.toml> [--iterations N] [--resume] [--no-kg] [--dry-run]
uv run sculpt report --config <cfg.toml> --out final.mp4
uv run sculpt kg ingest <seeds.yml>
uv run sculpt kg extract --all
uv run sculpt kg stats
uv run sculpt kg list-papers
uv run sculpt kg list-techniques
uv run sculpt kg viz --config <cfg.toml> --out kg.html

# Fast end-to-end sanity
uv run sculpt run "run forward fast" --config examples/hopper/config.toml --iterations 3 --dry-run   # ~50s

# Live demos
uv run python scripts/phase1_smoke.py         # ~30s, trains Hopper 20k steps
uv run python scripts/phase2_smoke.py         # ~30s, train + rollout
uv run python scripts/diagnose_demo.py         # ~70s, live diagnose
uv run python scripts/edit_demo.py             # ~30s, live v0 → v1
uv run python scripts/kg_seed_demo.py          # hand-seed the KG for demos
```

---

## Dependencies

Primary dependencies (pinned minimums in `pyproject.toml`):

| Package               | Purpose                                                    |
| --------------------- | ---------------------------------------------------------- |
| `anthropic`           | Claude API client                                          |
| `gymnasium[mujoco]`   | Reference env (Hopper-v4)                                  |
| `stable-baselines3`   | Reference PPO in the reference adapter                     |
| `torch`               | SB3 backbone + sentence-transformers                       |
| `sentence-transformers` | `query_semantic` (all-MiniLM-L6-v2)                      |
| `arxiv`               | Paper metadata                                             |
| `pypdf`               | PDF → text                                                 |
| `pyvis`               | KG HTML viz                                                |
| `imageio-ffmpeg`      | Bundled ffmpeg for mp4 rendering                           |
| `typer`               | CLI                                                        |
| `python-dotenv`       | `.env` auto-load                                           |
| `pyyaml`              | Seed YAML                                                  |
| `tomli`               | Py310 TOML fallback (3.11+ uses stdlib `tomllib`)          |
| `pytest`              | Tests                                                      |

No `jax` / `brax` / `mjx` — the AME456 quadruped lives in a separate
env. Those are expected to land when the Brax adapter is contributed
(see `docs/adapters.md`).

---

*End of history.*
