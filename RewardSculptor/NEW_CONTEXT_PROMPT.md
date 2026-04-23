# Reward Sculptor — context bootstrap

Paste this verbatim into a new Claude Code conversation so the session
has the same working context as the one that built this project.

---

## Who you are helping

Sam Doane (USC ME undergrad, AME 456 capstone on a quadruped jumping
robot with series-elastic actuators). Primary working directory:

```
C:\Users\SamJD\OneDrive\Desktop\Projects\RewardSculptor
```

A secondary project sits at `C:\Users\SamJD\OneDrive\Desktop\AME456\files`
(the MJX quadruped env). RewardSculptor imports nothing from AME456;
the direction of the dependency is the other way around — AME456's env
file imports `sculptor.reward.compute_reward`.

## What the project is

**Reward Sculptor** is a lab-agnostic autonomous agent that iterates
on RL reward functions, grounded in a knowledge graph of research
papers. Two loops joined at the diagnoser:

1. **Inner iteration loop**: `train → rollout → diagnose → apply_edits
   → commit` — produces `v0.py → v1.py → v<N>.py` reward modules with
   full provenance (git commit, `CHANGELOG.md`, `reports/provenance.json`).
2. **Knowledge graph**: arxiv seeds → PDFs → Claude-extracted
   `Paper` / `Technique` / `FailureMode` / `RewardComponent` /
   `Environment` nodes, queryable by failure mode (graph walk) and
   by semantic similarity (sentence-transformers cosine). The diagnoser
   pulls KG hits and the editor cites `arxiv_id` for every
   literature-grounded edit.

The reference adapter is Gymnasium + Stable-Baselines3 (`GymSB3Adapter`).
External labs contribute adapters for their own stacks — Isaac Gym,
Brax, RLlib, CleanRL, custom loops. The adapter contract is documented
in `docs/adapters.md`.

## First things to do in the new session

1. **Read [`HISTORY.md`](HISTORY.md)** end-to-end. It's the authoritative
   log of what's built, how it's organized, and what's verified.
2. **Skim [`README.md`](README.md)** for the user-facing pitch +
   architecture + Brax adapter worked example.
3. **Read [`docs/adapters.md`](docs/adapters.md)** if the user wants to
   write a new adapter or debug the existing one.
4. **Run the sanity sweep** (takes ~1 minute total):
   ```bash
   cd C:/Users/SamJD/OneDrive/Desktop/Projects/RewardSculptor
   uv run pytest tests/                                                 # 62 passed, 1 skipped
   uv run sculpt run "demo goal" --config examples/hopper/config.toml \
       --iterations 3 --no-kg --dry-run                                 # ~50s, no API cost
   ```
   If either fails, stop and diagnose before doing anything else.

## Architecture in one screen

```
sculpt run --config <cfg.toml> <behavior goal> [--iterations N]
         │
         ▼
  ┌──────────────┐
  │ sculpt_run() │  (sculptor/sculpt.py)
  └──┬───────────┘
     │ per iter i:
     │
     ├─► adapter.train(reward=rewards/current.py, out=runs/iter_<i>/, steps, seed)
     │       writes checkpoint.zip + metrics.json + reward_spec.json
     │
     ├─► adapter.rollout(ckpt, out=runs/iter_<i>/rollout/, n=rollout_episodes)
     │       writes rollout.mp4 + keyframes/*.png + trajectory.npz + behavior.json
     │
     ├─► diagnose(iter_dir, goal, config)
     │       stage 1 (preliminary): reward_spec + metrics + behavior + 4 keyframes
     │                               + reward_contract + behavior-metric vocab
     │                               → failure_modes (fixed 7-label enum)
     │                                 + evidence + confidence
     │       stage 2 (grounded): + KG top-6 via query_techniques + query_semantic
     │                            → proposed_edits each with paper_refs, rationale,
     │                              optional requires_env_extension=true
     │       writes iter_dir/diagnosis.json
     │
     ├─► apply_edits(current_reward, diagnosis, "v<i+1>", reward_contract)
     │       PRE-flight:  paper_refs in KG; grounded-field rule;
     │                    requires_env_extension split off
     │       LLM CALL:    rewrite reward module end-to-end
     │       POST-flight: import + compute_reward zero-dummies + expected_components
     │                    + REWARD_SPEC shape + references in KG
     │       one retry on failure; second failure raises.
     │       writes rewards/v<i+1>.py + regenerates rewards/current.py (re-export)
     │
     ├─► git commit "iter <i>: <failure_modes> [<n> edits]"
     ├─► append CHANGELOG.md, update reports/provenance.json + metric_history.json
     │
     └─► early-stop if max(metric[-3:]) ≤ max(metric[:-3])
```

Everything is driven off `config.toml` at the project root. The reward
module format (`compute_reward(state, action, next_state, info) ->
(reward, components)` + a typed `REWARD_SPEC` dict) is stable across
iterations. The adapter's `reward_contract()` declares what `info` keys
exist and (optionally) pins the component-dict key set; every
generated reward is validated against it.

## Critical gotchas you must know before editing

1. **Claude Code exports `ANTHROPIC_API_KEY=""`** (empty string).
   `sculptor/__init__.py` autoloads `.env` and treats empty-string env
   vars as unset so the `.env` value propagates. Do not "fix" the
   dotenv loader by making it `override=True` — that regresses other
   environments.

2. **Windows cp1252 console** chokes on Unicode arrows in `print()`.
   Use `->`, not `→`, in CLI output. Don't assume Windows terminals
   speak UTF-8.

3. **OneDrive file-locking** sometimes causes `uv add` to fail with
   "Access is denied (os error 5)" on the .dist-info dir. Workaround:
   `export UV_LINK_MODE=copy` (or prefix commands with it).

4. **Python pin**: `.python-version` points at **miniconda 3.13.5**
   (`C:/Users/SamJD/miniconda3/python.exe`). The uv-provided CPython
   3.11 has its `_ssl.pyd` blocked by Windows Application Control on
   this machine, which breaks pytest (anyio imports ssl). Don't switch
   back to 3.11 without allow-listing that DLL first.

5. **Legacy quadruped bridge**: `sculptor/reward.py` is the v1 reward
   for the AME456 quadruped env. It's not part of Reward Sculptor's
   lab-agnostic design — it's there so the AME456 env's
   `from sculptor.reward import compute_reward` keeps working.
   `tests/test_reward_parity.py` exercises it but is gated by
   `pytest.importorskip("jax")`. Don't delete either without
   coordinating with the AME456 project.

6. **Prompts are in `sculptor/prompts/*.md`**, loaded at import time.
   To override per-project (e.g., manipulation-domain phrasing), set
   `SCULPTOR_PROMPTS_DIR=/path/to/your/prompts`. `pyproject.toml`
   `[tool.hatch.build.targets.wheel.force-include]` ships the .md
   files in the built wheel.

7. **arxiv API rate-limits aggressively** after bursts. The ingest
   (`sculptor/kg/ingest.py`) has a fallback: direct PDF CDN URL
   (`https://arxiv.org/pdf/<id>.pdf`) + metadata from the seeds YAML
   when the API returns 429. Re-run after ~5 minutes if you see
   "Rate exceeded".

8. **Grounded-field rule** (edit.py pre-flight): every identifier in
   a proposed-edit's `suggested_value` formula must be in
   `reward_contract.expected_info_keys ∪ current-components ∪
   current-hyperparameters ∪ math-allowlist`. The Stage-2 diagnose
   prompt instructs the model to set `requires_env_extension=true`
   rather than emit an ungrounded formula. Those flagged edits are
   logged in CHANGELOG but **not applied**.

9. **examples/hopper/config.toml has `steps_per_iter=10000`** as a
   demo bound. Real Hopper training wants 50000+. The `b36ef0f` commit
   in the hopper project's git history documents this.

10. **The `examples/hopper/` directory is its own git repo** (`git
    init` was run for the sculpt-run demo). The RewardSculptor project
    root is intentionally NOT a git repo — commits live inside each
    sculpted project.

## Current state snapshot (as of 2026-04-20)

- **62 tests passing, 1 skipped** (`test_reward_parity.py`).
- **Live end-to-end verified**: `sculpt init` → config edit → `sculpt
  run --iterations 3` (live, ~5 min at 10k steps/iter) → `sculpt
  report` (441 KB final.mp4 + 4 KB final_report.md) → `sculpt kg viz`
  (736 KB kg.html).
- **Dry-run 3-iter**: 50.97s end-to-end (target: <60s, ✓).
- **KG has 23 nodes + 21 edges** (hand-seeded via
  `scripts/kg_seed_demo.py` because live extraction wasn't run in the
  previous session; see `HISTORY.md` §"Known limitations" item 1).

## File-reading priority

If the user asks "what does X do?" read in this order:

1. `HISTORY.md` — what was built and in what order
2. `README.md` — user-facing architecture
3. `docs/adapters.md` — adapter contract for lab contributors
4. The relevant module source (all under `sculptor/`)
5. `tests/test_*.py` for the module — the contract is most precisely
   encoded in the tests
6. `scripts/*_demo.py` — runnable live demos

If they ask "how do I run X?" the first answer is almost always in
[`README.md`](README.md) §Commands or the CLI's `--help`:
```bash
uv run sculpt --help
uv run sculpt run --help
uv run sculpt kg --help
```

## What's open / worth exploring

See `HISTORY.md` §"Worth-exploring next". Summary:

1. Run `sculpt kg extract --all` live (never executed — hand-seeded KG
   is the current state).
2. Implement a Brax/MJX adapter for the AME456 quadruped to close the
   original loop.
3. Expand / verify the failure-mode vocabulary beyond the stock 7.
4. CI / GitHub Actions (none currently).
5. Polish the pyvis viz (paper PDF links, force-directed grouping by
   domain tag).

## Communication style Sam prefers

Terse, direct, technically specific. Prefer showing actual file paths +
line numbers + metrics over prose summaries. Flag gotchas early. When
there's a judgment call, say what you picked and why. When a tool or
library doesn't do something, say so plainly rather than dressing it
up.

If you're about to run something destructive (destructive git, rm -rf
outside `runs/` or temp dirs, force-push, anything that touches the
AME456 project), confirm first.

---

*End of bootstrap. Assume everything above is current; cross-check
against `HISTORY.md` for detail and the repo's file tree for ground
truth.*
