# Writing a Sculptor Adapter

An adapter is the **only** piece of Sculptor that talks to your RL stack.
The inner loop (`train → rollout → diagnose → apply_edits`), the KG, the
literature-grounded editor, the changelog, the provenance tracking — all
of that is adapter-agnostic. Swap the adapter and Sculptor drives Isaac
Gym just as happily as it drives Gymnasium + SB3.

This document is the contract, nothing more and nothing less. If you're
implementing an adapter, read every section; everything here is what
Sculptor assumes your adapter guarantees.

## Contents

1. [The contract: what `SculptorAdapter` must do](#the-contract)
2. [`RewardContract` in detail](#rewardcontract-in-detail)
3. [Reward-injection patterns](#reward-injection-patterns)
   * [Gymnasium wrapper (reference)](#gymnasium-wrapper-reference-implementation)
   * [Isaac Gym / Isaac Lab](#isaac-gym--isaac-lab)
   * [Brax / MJX](#brax--mjx)
   * [RLlib](#rllib)
   * [Custom training loops](#custom-training-loops)
4. [Validation checklist](#validation-checklist)

---

## The contract

```python
# sculptor/adapters/base.py

class SculptorAdapter(ABC):
    def train(self, reward_module_path, output_dir, steps, seed) -> TrainResult: ...
    def rollout(self, checkpoint_path, output_dir, n_episodes) -> RolloutResult: ...
    def compute_behavior_metrics(self, rollout) -> dict[str, Any]: ...
    def reward_contract(self) -> RewardContract: ...
```

### `train(reward_module_path, output_dir, steps, seed) -> TrainResult`

| Arg                  | Type       | Guarantees from Sculptor                                   |
| -------------------- | ---------- | ---------------------------------------------------------- |
| `reward_module_path` | `Path`     | Points at a Python module defining `compute_reward(state, action, next_state, info) -> (reward, components)` and a `REWARD_SPEC` dict. **Do not mutate this file.** |
| `output_dir`         | `Path`     | Exists and is writable. Sculptor will read back artifacts named by the contract below; name collisions are your bug. |
| `steps`              | `int`      | The exact training budget for this call. Sculptor uses `steps` to control per-iteration wall-clock. **Do not exceed it.** |
| `seed`               | `int`      | Seed for env + algorithm. Sculptor varies `seed` per iteration so the budget is fair; use it or you'll muddle provenance. |

**Required writes under `output_dir`:**

| File                    | Shape                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| `checkpoint.zip` (or another extension — just match your `TrainResult.checkpoint_path`) | Deserialisable by your own `rollout()` method. |
| `metrics.json`          | JSON dict with at minimum the config's `primary_metric`. The reference `GymSB3Adapter` emits `{"metrics": {...}, "components": {...}}` — any shape works as long as `primary_metric` is findable under one of the top-level dicts. |
| `reward_spec.json`      | A JSON dump of the imported reward module's `REWARD_SPEC`. The diagnoser reads this to know the current hyperparameters. |
| `logs/` (optional)      | Any training-time logs you want persisted. Sculptor uses this only for `TrainResult.logs_path`. |

**`TrainResult` return:**

```python
@dataclass
class TrainResult:
    checkpoint_path: Path        # what rollout() will consume
    metrics_dict: dict[str, float]
    component_means: dict[str, float]   # per-component averages, non-empty
    logs_path: Path
```

Rules:
- `component_means` **must not be empty** — it's how the diagnoser sees which reward terms were active. Empty means "you never called compute_reward", which is a bug.
- Keys in `component_means` must be a subset of the `components` dict the reward module returned.

### `rollout(checkpoint_path, output_dir, n_episodes) -> RolloutResult`

`output_dir` is typically `<iter_dir>/rollout/`. Required writes:

| File                | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `rollout.mp4`       | Stitched eval video. Sculptor's time-lapse builder samples panels from the best episode. |
| `keyframes/*.png`   | ~12 evenly-spaced frames from the best episode. The diagnoser sends 4 of them to Claude. |
| `trajectory.npz`    | Per-step arrays — at minimum `rewards`, `terminations`, `truncations`, `episode_id`. The reference adapter also stashes `obs`, `actions`, `x_velocity`, and a per-step `components_per_step` object array. |
| `behavior.json`     | Output of your `compute_behavior_metrics`. Sculptor treats these as the "did it learn the task?" signal, orthogonal to the scalar return. |

**`RolloutResult` return:**

```python
@dataclass
class RolloutResult:
    video_path: Path
    keyframes_dir: Path
    trajectory_path: Path
    n_episodes: int
```

### `compute_behavior_metrics(rollout) -> dict[str, Any]`

Adapter-defined. Return a flat `dict[str, float | int]` keyed by the names
declared in your project's `config.iteration.behavior_metrics` — those keys
are what the diagnoser sees as the adapter's "behavior vocabulary."

Typical examples:

| Domain                    | `behavior_metrics`                                                            |
| ------------------------- | ----------------------------------------------------------------------------- |
| Gymnasium locomotion      | `max_episode_length`, `mean_forward_velocity`, `fall_rate`                    |
| Legged-robot jumping      | `max_jump_height`, `num_takeoffs`, `mean_tilt_deg_per_step`                   |
| Manipulation              | `success_rate`, `time_to_contact`, `mean_object_displacement`                 |
| Navigation                | `goal_reach_rate`, `path_length`, `collision_count`                           |

Whatever you pick, declare the names in `config.iteration.behavior_metrics`
and emit those keys here. The diagnose prompt will surface them to Claude
as "the adapter's behavior vocabulary."

### `reward_contract() -> RewardContract`

Described in full in the next section — this is what every edit and
diagnosis is validated against.

---

## `RewardContract` in detail

```python
@dataclass
class RewardContract:
    observation_space_spec: Any
    action_space_spec:      Any
    expected_info_keys:     list[str]
    expected_components:    list[str] | None
```

| Field                    | Meaning                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `observation_space_spec` | Any object with `.shape` / `.dtype`. `gymnasium.spaces.Box`, a NumPy dtype-like, or a plain `(shape, dtype)` tuple all work. Sculptor's edit validator uses it to build zero-dummies for `compute_reward(state, action, next_state, info)`. |
| `action_space_spec`      | Same shape constraints.                                                                                                        |
| `expected_info_keys`     | **Load-bearing.** The complete list of field names the reward function is allowed to read from `info`. The editor refuses to generate a reward module that references any key not in this list. Missing keys must be added to the env first. |
| `expected_components`    | Either `None` (open — any component-dict keys are allowed) or an explicit list. When provided, the generated reward module's components dict keys MUST be a subset. Catches typos (`"alive_bouns"`) and renames (`"ctrl"` vs `"ctrl_cost"`). |

**Design philosophy:** the contract is how you enforce "no reward edit can
assume a field I didn't wire up." If your env doesn't expose `torso_angle`,
leave it out of `expected_info_keys`, and the editor will either propose a
different edit or flag `requires_env_extension=true` with a note to you.

**`expected_info_keys` may vary by task within one adapter.** A single
adapter can advertise different info keys per task family when the robot
exposes different sensors. `MjlabAdapter` does this (see
`_info_keys_for_task`): every task gets the base set
`{episode_length, terminated, time_outs, step_dt, base_height, fallen}`, and
the **G1 humanoid additionally** gets per-foot kick channels
`{left_foot_contact, right_foot_contact, left_foot_swing_speed,
right_foot_swing_speed, left_foot_height, right_foot_height}` plus
`base_horizontal_speed`. These are sourced from sensors mjlab already
computes for its own foot reward terms (`feet_ground_contact`,
`foot_height_scan`, site velocities) and are surfaced as `(num_envs,)`
scalars — per-foot data flattened to named keys, zero-filled on tasks
without the named foot sites/sensors. They exist so a sculpted reward can
shape a single-leg kick (balance on one foot, swing the other forward); the
quadruped/Cartpole contracts intentionally omit them so the editor can't
ground a formula the runner would only zero-fill.

---

## Reward-injection patterns

Every stack has its own place to intercept "the number the agent
optimizes." Here are the five common shapes.

### Gymnasium wrapper (reference implementation)

See [`sculptor/adapters/gym_sb3.py`](../sculptor/adapters/gym_sb3.py). The
adapter wraps the env in a `RewardOverrideWrapper` that loads the reward
module once at construction and calls `compute_reward` in `step()`:

```python
class RewardOverrideWrapper(gym.Wrapper):
    def __init__(self, env, reward_module_path: Path):
        super().__init__(env)
        mod = _load_reward_module(reward_module_path)
        self._compute_reward = mod.compute_reward
        self._last_obs = None
        self._component_acc = defaultdict(float)
        self._step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        reward, components = self._compute_reward(
            self._last_obs, action, next_obs, info)
        for k, v in components.items():
            self._component_acc[k] += float(v)
        self._step_count += 1
        info["sculptor_components"] = dict(components)
        self._last_obs = next_obs
        return next_obs, float(reward), terminated, truncated, info
```

Works for CleanRL, Stable-Baselines3, Tianshou, RLkit, Acme, and anything
else that speaks the Gymnasium API. Tradeoff: no domain-specific
optimizations (you can't use the underlying env's vectorized GPU step
while also running a Python reward function per step).

### Isaac Gym / Isaac Lab

Isaac tasks typically expose a `compute_rewards` (or `_compute_reward`)
method on the task class that runs on the GPU over the full env batch.
Patch it at construction time, before the trainer is built:

```python
class IsaacAdapter(SculptorAdapter):
    def train(self, reward_module_path, output_dir, steps, seed):
        reward_mod = _load_reward_module(reward_module_path)

        # Build the task first, then monkey-patch compute_rewards so the
        # framework's own vectorized training loop uses our reward.
        task = build_task(self._cfg, seed=seed)

        original_compute = task.compute_rewards
        components_acc = defaultdict(float)
        step_count = [0]  # mutable cell for closure

        def patched_compute_rewards(actions, *args, **kwargs):
            # Vectorized reward: task.obs_buf / task.info_buf are tensors.
            reward_vec, components_vec = reward_mod.compute_reward_vec(
                state=task.prev_obs_buf, action=actions,
                next_state=task.obs_buf, info=task.info_buf,
            )
            for k, v in components_vec.items():
                components_acc[k] += float(v.mean().item())
            step_count[0] += 1
            return reward_vec

        task.compute_rewards = patched_compute_rewards

        # Now hand off to Isaac's own PPO/RL runner for `steps`.
        runner = RLRunner(task, cfg=self._cfg)
        runner.train(num_iterations=steps // task.num_envs)

        # Persist artifacts: checkpoint.zip, metrics.json, reward_spec.json, ...
        ...
        return TrainResult(
            checkpoint_path=output_dir / "checkpoint.pt",
            metrics_dict={"mean_return": float(runner.best_mean_return)},
            component_means={k: v / max(step_count[0], 1) for k, v in components_acc.items()},
            logs_path=output_dir / "logs",
        )
```

Two things the adapter owes here:

1. **A vectorized `compute_reward_vec`.** You're calling the reward over a
   batch of envs; the sculptor-generated reward must be torch-friendly.
   The simplest route is to ship a small `to_vec(reward_fn)` helper that
   wraps a per-step Python function; the faster route is to require
   vectorized rewards from the start and document that in
   `expected_info_keys`.
2. **Component aggregation across envs.** Take per-env means, then a
   running mean across steps. Sculptor's diagnoser cares about magnitudes,
   not per-env variance.

### Brax / MJX

Brax environments accept a `reward_fn` at construction time. The sculptor
reward must be **JAX-pure** — no Python branching on traced values, only
`jnp.where` / `jax.lax.cond`. The AME456 quadruped env in this repo is a
real-world example; see its `reward_fn=` kwarg seam.

```python
class BraxAdapter(SculptorAdapter):
    def train(self, reward_module_path, output_dir, steps, seed):
        reward_mod = _load_reward_module(reward_module_path)

        # 1. Preflight: lower the reward under jit so we fail FAST on numpy
        #    idioms before the Brax PPO JIT cycle burns 30 s of compilation.
        self._jit_preflight(reward_mod.compute_reward)

        # 2. Build env with the generated reward injected.
        env = self._env_factory(reward_fn=reward_mod.compute_reward)

        # 3. Hand off to Brax PPO.
        from brax.training.agents.ppo import train as ppo_train
        params, metrics = ppo_train(
            environment=env,
            num_timesteps=steps,
            num_envs=self._n_envs,
            seed=seed,
        )

        # 4. Serialize + persist.
        _save_pickle(output_dir / "checkpoint.pkl", params)
        (output_dir / "metrics.json").write_text(json.dumps(metrics))
        (output_dir / "reward_spec.json").write_text(
            json.dumps(reward_mod.REWARD_SPEC, default=str))
        ...
        return TrainResult(...)

    def _jit_preflight(self, reward_fn):
        """Trace the reward under jax.jit to surface non-JAX constructs
        (float() casts, numpy operations, Python `if` on traced values)
        before training runs."""
        import jax
        import jax.numpy as jnp
        dummy_state = self._dummy_env_state()
        jax.jit(reward_fn).lower(
            dummy_state, jnp.zeros(self._action_dim),
            dummy_state, {}
        )  # raises on bad reward
```

The pre-flight JIT-lowering is optional but highly recommended. It catches
~90 % of "my reward worked in Python but Brax training silently produced
garbage" bugs in seconds.

### RLlib

The canonical seam is `env_creator` passed to the trainer. Wrap the env in
a `RewardOverrideWrapper` analog — essentially the Gymnasium pattern, but
wired at RLlib's env-construction boundary so every worker builds a fresh
wrapper:

```python
class RLlibAdapter(SculptorAdapter):
    def train(self, reward_module_path, output_dir, steps, seed):
        def env_creator(env_config):
            base = gym.make(self._env_id)
            return RewardOverrideWrapper(base, reward_module_path)

        register_env("sculpt_env", env_creator)

        config = (
            PPOConfig()
            .environment("sculpt_env")
            .env_runners(num_env_runners=self._n_workers)
            .framework("torch")
            .training(**self._ppo_kwargs)
        )
        algo = config.build()
        for _ in range(steps // config.train_batch_size):
            algo.train()

        # Collect component means across all workers.
        component_means = _aggregate_component_means(
            algo.workers.foreach_worker(lambda w: w.env.get_component_means()))
        ...
        return TrainResult(...)
```

Watch the **worker remoting boundary**. `reward_module_path` must be
reachable from every worker's filesystem. If workers run on remote nodes,
package the reward module into the `runtime_env` or stage it onto shared
storage. Don't inline the source into `env_config` — that'll break prompt
caching on the editor side.

### Custom training loops

Research code often runs its own loop. Intercept at the exact line you
compute "the reward":

```python
class CustomLoopAdapter(SculptorAdapter):
    def train(self, reward_module_path, output_dir, steps, seed):
        reward_mod = _load_reward_module(reward_module_path)
        env, agent = self._build(seed)

        component_acc, step_count = defaultdict(float), 0
        for step in range(steps):
            action = agent.act(env.state)
            next_state, _env_reward, done, info = env.step(action)

            reward, components = reward_mod.compute_reward(
                env.prev_state, action, next_state, info)
            for k, v in components.items():
                component_acc[k] += float(v)
            step_count += 1

            agent.learn(env.prev_state, action, reward, next_state, done)
            env.prev_state = next_state
            if done:
                env.reset()

        # Serialize as usual.
        ...
        return TrainResult(
            ...,
            component_means={k: v / step_count for k, v in component_acc.items()},
            ...
        )
```

Keep the override **stateless** except for the two fields every sculptor
run wants: the components accumulator and an optional `prev_action`
if your reward uses action-smoothness. Sculptor does **not** maintain
any step state on your behalf.

---

## Validation checklist

Before publishing a new adapter, run:

1. **Contract test.** Copy `tests/test_adapter_contract.py` and swap the
   `GymSB3Adapter` instantiation for yours. All six tests should pass:
     - config parses
     - adapter instantiates
     - `reward_contract()` returns a populated `RewardContract`
     - a wrapper around your reward-injection mechanism replaces `env.step`'s reward
     - a reward module missing `compute_reward` raises at adapter construction
     - the `REWARD_SPEC` dict of your stub reward has `version / description / author / parent_hash / hyperparameters / references`
2. **Short training smoke.** Instantiate your adapter, call
   `adapter.train(reward_module_path, tmp_dir, steps=10_000, seed=42)`.
   Assert that `checkpoint_path` exists + is non-empty, `metrics_dict`
   contains the config's `primary_metric`, `component_means` is non-empty.
3. **Rollout smoke.** Call `adapter.rollout(ckpt, tmp_dir/"rollout", 4)`.
   Assert `rollout.mp4` is > 2 KB, `keyframes/` has at least 1 PNG,
   `trajectory.npz` loads with `numpy.load`, `behavior.json` is valid JSON
   with the keys your adapter declares.
4. **End-to-end dry run.** `uv run sculpt run --config your_project/config.toml
   "<goal>" --iterations 2 --dry-run`. Under a minute. Produces two iter
   dirs, one new `v1.py`, a CHANGELOG entry, and a commit per iter.
5. **End-to-end live run.** Same command without `--dry-run`. Confirm the
   diagnoser's edits are actually applied in `v1.py` / `v2.py`, and the
   sculpt report video renders. If an edit's formula references a field
   your env doesn't publish, the diagnoser should set
   `requires_env_extension=true` rather than crash — if it crashes, your
   `expected_info_keys` is probably wider than the truth.

If any of those fail, the adapter is not ready. Sculptor leans hard on the
contract — silent contract violations will show up as mysterious
diagnosis failures three iterations into a run, which is the worst time
to debug them.

---

**Questions about the contract?** Open a discussion in the repo — the
adapter surface is intentionally small and stable, so every new use case
that reveals a missing guarantee is a signal we should tighten the doc.
