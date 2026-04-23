# Contributing to Reward Sculptor

Reward Sculptor only pays its way if it works in *your* stack. The
easiest ways to contribute, in rough order of impact:

1. **Ship an adapter** for your RL framework. See §"Contributing an
   adapter for your RL framework" below — this is the single highest-
   leverage thing an external lab can contribute. Scaffolds with
   ~4-8 hours of adoption work exist for **Isaac Lab**, **Brax / MJX**,
   and **Ray RLlib** — see [`docs/adapters/`](docs/adapters/).
2. **Add a robot to the library.** The UI's Library tab is driven by
   `reward-sculptor-ui/backend/data/robot_library.yml` — a full guide is
   at [`docs/robot_library.md`](docs/robot_library.md). In short:
   pick the `robot_descriptions` loader name, categorize, write a
   1-2 sentence description, render a thumbnail via
   `scripts/generate_library_thumbnails.py --only-slug <new>`, and
   PR the YAML + PNG.
3. **Grow the knowledge graph**. Curate a seeds YAML for your domain
   (manipulation, navigation, LLM RL, HRI, whatever) and open a PR that
   adds it under `examples/<your-domain>/kg_seeds.yml`. A good seed set
   is a 20-line file; we'll take 50 of them gladly.
4. **Improve a prompt.** All Claude prompts live in
   `sculptor/prompts/*.md` — if you find a phrasing that cuts
   hallucination for your domain, PR it (or ship it as an override via
   `SCULPTOR_PROMPTS_DIR`, with notes in your project README).
5. **File a bug.** "The diagnoser keeps flagging `sparse_reward` when
   really it's `reward_hacking`" is exactly the signal we need.

---

## Ground rules

- **Python ≥ 3.10**, `uv` for env management.
- **Tests first.** `uv run pytest tests/` should pass green before you
  PR. New code paths come with tests; new adapters come with a copy of
  `tests/test_adapter_contract.py` parameterized for the new adapter.
- **No LLM calls in tests.** Mock the client. Keep CI deterministic and
  free.
- **Strict typing is optional**, but code that crosses a public API
  (`sculptor.adapters.*`, `sculptor.diagnose.*`, `sculptor.edit.*`)
  should be typed. The rest of the codebase is dynamic; match the
  neighbors.
- **Commits are small and explain why.** `iter <i>: <failure modes>` is
  a template `sculpt run` writes automatically; humans should do better.

---

## Getting started

```bash
git clone <this-repo>
cd RewardSculptor
uv sync
cp .env.example .env      # fill in ANTHROPIC_API_KEY
uv run pytest tests/ -v   # sanity check
```

Hot-path development:

```bash
# Run only the tests relevant to what you're changing.
uv run pytest tests/test_adapter_contract.py -v
uv run pytest tests/test_diagnose.py tests/test_edit.py -v

# Full smoke (under a minute, no API credit spent):
uv run sculpt run \
    --config examples/hopper/config.toml \
    "goal" --iterations 3 --dry-run
```

---

## Contributing an adapter for your RL framework

**This is the most important contribution an external lab can make.**
Sculptor's inner loop (train → rollout → diagnose → edit) is adapter-
agnostic, but the inner loop only *runs* when an adapter wires it to
something concrete. Every new adapter unlocks Sculptor for an entire RL
framework's worth of users.

We are actively inviting:

- **Isaac Gym / Isaac Lab** adapter. Method-patching pattern; GPU-batched
  reward. See `docs/adapters.md` §"Isaac Gym / Isaac Lab".
- **Brax / MJX** adapter. Custom env with `reward_fn` kwarg + JAX-pure
  reward + JIT preflight. See `README.md` §"Adapting to your RL stack"
  for a worked sketch.
- **RLlib** adapter. `env_creator` override; worker remoting caveat.
- **CleanRL** adapter. Custom training loop with a per-step callback.
- **Anything else that speaks RL**. Tianshou, SaLinA, Acme, your own
  in-house rig.

### The five-step checklist

1. **Read the contract.** [`docs/adapters.md`](docs/adapters.md) is the
   reference. Every method signature, every required artifact, every
   validation rule — all there. Do not deviate from the contract without
   discussing it in an issue first.

2. **Write the adapter class.** Standard location:
   `sculptor/adapters/<stack_name>.py`. Subclass `SculptorAdapter`.
   Implement all four abstract methods:
      - `train(reward_module_path, output_dir, steps, seed) -> TrainResult`
      - `rollout(checkpoint_path, output_dir, n_episodes) -> RolloutResult`
      - `compute_behavior_metrics(rollout) -> dict[str, Any]`
      - `reward_contract() -> RewardContract`

   The docstring should name the injection pattern you chose (wrapper /
   method patch / reward_fn / callback) and justify the choice.

3. **Handle the reward module cleanly.** The generated
   `reward_module_path` is a Python file with `compute_reward(state,
   action, next_state, info) -> (reward, components)` and a `REWARD_SPEC`
   dict. Load it via `importlib.util.spec_from_file_location` (the
   reference implementation has a reusable helper) — **never** add its
   directory to `sys.path` and import by name. You don't own the
   name; it changes per iteration.

4. **Ship the contract test.** Copy `tests/test_adapter_contract.py` to
   `tests/test_adapter_<stack>_contract.py` and swap the adapter
   instantiation for yours. All six assertions must pass:
     - config parses
     - adapter instantiates with your config shape
     - `reward_contract()` returns populated `RewardContract`
     - your reward-injection mechanism replaces the env's reward
     - malformed reward modules raise at adapter construction
     - the stub reward's `REWARD_SPEC` has the required keys

   Mock your trainer. No real training happens in tests.

5. **Ship an end-to-end example.** `examples/<stack>_<domain>/` with a
   `config.toml`, `rewards/v0.py`, `kg_seeds.yml`, and a one-page
   README showing someone how to actually run `sculpt run` against your
   adapter. This is the part that makes the adapter useful to someone
   who isn't you.

### What we'll ask when reviewing your PR

- Does the adapter violate the contract? If yes, we'll reject.
- Does `--dry-run` work end-to-end in under a minute? If not, we'll
  reject — that's the smoke test that keeps the project honest.
- Does it handle the edge cases documented in `docs/adapters.md`?
  (Vectorized envs, worker remoting, reward module hot-reloads.)
- Does the contract test pass with no real training? If the test
  requires GPU or a specific simulator installed, mark it with
  `@pytest.mark.skipif` and document the skip reason.
- Is the `examples/` directory runnable by someone who just ran
  `sculpt init` for the first time?

### What we will NOT ask

- That you implement `rollout`/`compute_behavior_metrics` in the first
  PR. Ship `train` + `reward_contract` + the contract test first, then
  layer rollout + metrics in a follow-up PR.
- That you've wired up your adapter to the `GymSB3Adapter` reference
  benchmarks. Use your own benchmarks; every domain has its own.
- Prompt customization. Sculptor's stock prompts work well out of the
  box; tune them if you want to in `sculptor/prompts/`, but it's not a
  prerequisite for the adapter PR.

---

## Adding a prompt variant

If your domain needs a different system prompt — e.g. the default
`diagnose_grounded.md` talks too much about locomotion for your LLM-RL
project — you have two options:

1. **Per-project override** (no code change): drop a replacement
   `diagnose_grounded.md` in your project directory and set
   `SCULPTOR_PROMPTS_DIR=/path/to/your/prompts` before running
   `sculpt run`. The loader picks it up automatically.
2. **Upstream contribution**: if the improved phrasing helps every
   domain, PR a replacement in `sculptor/prompts/` with before/after
   measurements (e.g., "on the hopper bench, Claude's preliminary
   diagnoses went from 70% confidence to 82%").

Either way, keep prompts short. The current prompts are 20-50 lines; if
yours is 200, you're probably solving a prompt-tuning problem by
throwing more text at it.

---

## Extending the failure-mode vocabulary

The diagnoser's failure-mode enum is intentionally small and stack-
agnostic (7 labels). If your domain genuinely needs a new label —
`object_ejection` for manipulation, `map_drift` for navigation — open
an issue first. New labels have to earn their way in: we want every
adopting lab to be able to reuse diagnoses across domains, and a long
tail of lab-specific labels breaks that.

If the label is domain-specific, keep it in your project's prompt
override and ignore the stock enum. You don't need to upstream it.

---

## Reporting bugs

- Open an issue with a minimal reproducer.
- Attach the relevant `runs/iter_*/diagnosis.json` and the failing
  `rewards/v<n>.py` if the bug is in the edit pipeline.
- If the bug is in the KG layer, `sculpt kg stats` output + the seed
  YAML you used.
- Sculptor prints stack traces verbatim; don't abbreviate them.

---

## License

MIT. By contributing, you agree that your contributions will be
licensed under the same.
