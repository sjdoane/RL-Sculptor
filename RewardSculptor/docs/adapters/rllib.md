# Ray RLlib adapter — adoption guide

**Status:** scaffolded (`sculptor.adapters.rllib.RllibAdapter`).
`train()` / `rollout()` raise `NotImplementedError`.
**Estimated effort:** 4-8 hours (worker coordination + checkpoint shape).

Ray RLlib is the mature "framework of frameworks" — supports most
algorithms + distributed training out of the box. The sculptor RLlib
adapter is the natural path for users who already have Ray
infrastructure and want to iterate rewards on top.

## Target versions

- `ray[rllib] >= 2.10`
- Python 3.9-3.13 (RLlib 2.10+ supports 3.13).
- CPU or GPU — RLlib handles device placement internally per-worker.

## Install prerequisites

```bash
cd ~/projects/RewardSculptor
uv add "ray[rllib]>=2.10"
```

Ray ships heavy (~600 MB for `rllib` extras). Expect a longer
`uv sync` than mjlab's cold install.

## Reward injection

RLlib composes envs via `env_creator` callbacks registered on the
algorithm's config. Wrap the base env in a `RewardOverrideWrapper`
analog (see `sculptor/adapters/gym_sb3.py` for the reference pattern):

```python
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        env=lambda env_cfg: RewardOverrideWrapper(
            gym.make(env_cfg["base_env_id"]),
            env_cfg["reward_module_path"],
        ),
        env_config={
            "base_env_id": self.env_id,
            "reward_module_path": str(reward_module_path),
        },
    )
    .framework(self.framework)  # "torch" or "tf2"
    .rollouts(num_rollout_workers=self.num_workers)
)
```

Each rollout worker instantiates the env on its own process; the
reward module is loaded per-worker from the absolute path. Ship the
path via `runtime_env` so workers on other nodes can resolve it:

```python
import ray
ray.init(runtime_env={"py_modules": [str(reward_module_path.parent)]})
```

## Minimal viable train() skeleton

```python
def train(self, reward_module_path, output_dir, steps, seed):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    ray.init(ignore_reinit_error=True, num_cpus=2)
    config = _build_ppo_config(
        env_id=self.env_id,
        reward_module_path=reward_module_path,
        num_workers=self.num_workers,
        num_envs_per_worker=self.num_envs_per_worker,
        seed=seed,
    )
    algo = config.build()

    # RLlib's train() returns a result dict per iteration. Loop until
    # num_env_steps_sampled hits the step budget.
    sampled = 0
    metrics = {}
    while sampled < steps:
        result = algo.train()
        sampled = result["env_runners"]["num_env_steps_sampled_lifetime"]
        metrics = result  # keep latest

    # RLlib checkpoints are DIRECTORIES, not files — adjust TrainResult
    # shape accordingly.
    ckpt_dir = output_dir / "checkpoint"
    algo.save(str(ckpt_dir))

    return TrainResult(
        checkpoint_path=ckpt_dir,  # Path, not .pt/.pkl
        metrics_dict={
            "mean_return": metrics["env_runners"]["episode_reward_mean"],
            "episodes_total": metrics["env_runners"]["episodes_total"],
        },
        component_means={},  # RLlib doesn't surface per-component reward
        logs_path=output_dir / "logs",
    )
```

## Testing strategy

First smoke test: 1-iter RLlib training on `CartPole-v1` (cheap,
CPU-only), assert `result["env_runners"]["episode_reward_mean"]` is a
float and `algo.save()` produces a `checkpoint/` directory with a
`rllib_checkpoint.json` manifest inside.

## Known gotchas

- **Worker remoting.** RLlib spawns rollout workers as Ray actors,
  each in a separate process. Any sculptor in-memory state (loaded
  modules, counters) does NOT cross the actor boundary. Coordination
  happens through the filesystem (reward-module path, logs) or Ray's
  `runtime_env`.
- **Ray init overhead.** `ray.init()` takes 5-15 s. For short training
  budgets, this overhead dominates; document it. Sculptor's inner
  loop does per-iteration subprocess spawn already, so the pain is
  mitigated but not eliminated.
- **Checkpoint directories.** Unlike mjlab / gym_sb3, RLlib saves a
  DIRECTORY not a single file. Sculptor's `TrainResult.checkpoint_path:
  Path` still works (a Path can point at a dir), but `torch.load()`
  won't work — rollout code must use `Algorithm.from_checkpoint`.
- **Single-node scope (v1).** The sculptor RLlib adapter's initial
  implementation targets single-node training (`num_rollout_workers=0`
  + `num_envs_per_worker=4`). Multi-node adds the runtime_env shipping
  concern + shared-filesystem assumptions that aren't worth fighting
  in v1.
- **Deprecated APIs.** RLlib is in heavy flux (2.x removed the old
  `Trainer` class in favour of `Algorithm`). Pin `ray[rllib]` to a
  narrow range in sculptor's pyproject to avoid API drift.

## References

- [Ray — GitHub](https://github.com/ray-project/ray)
- [RLlib — documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [RLlib environments guide](https://docs.ray.io/en/latest/rllib/rllib-env.html)

## Completion checklist

- [ ] Install `ray[rllib]>=2.10` and verify `import ray.rllib` works.
- [ ] Copy `RewardOverrideWrapper` from `gym_sb3.py` into
  `sculptor/adapters/rllib.py` (the RLlib env_creator closure must
  reference it).
- [ ] Implement `RllibAdapter.train()` + `rollout()`.
- [ ] Adjust `TrainResult` semantics: `checkpoint_path` is a directory.
  Rollout code uses `Algorithm.from_checkpoint`.
- [ ] Write `tests/test_rllib_adapter.py` — mocked `ray.init` + a real
  CartPole smoke behind `@pytest.mark.slow` (not gpu-gated; RLlib runs
  on CPU).
- [ ] Flip `ADAPTER_REGISTRY["rllib"].status` to `"ready"`.
