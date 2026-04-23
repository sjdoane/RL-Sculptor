# Brax / MJX adapter — adoption guide

**Status:** scaffolded (`sculptor.adapters.mjx.MjxAdapter`).
`train()` / `rollout()` raise `NotImplementedError`.
**Estimated effort:** 4-6 hours for a JAX-familiar contributor.

Brax is Google DeepMind's JAX-native RL framework; MJX is MuJoCo's
XLA-compiled backend that Brax targets for rigid-body sim. This is the
natural sculptor adapter for anyone already on the JAX stack — MJX is
faster than MuJoCo-CPU but the code paths diverge from torch.

## Target versions

- `brax >= 0.12.0`
- `jax[cuda12] >= 0.7.0` (CUDA 12+ build)
- `mujoco-mjx >= 3.3`
- CUDA 12.x. CPU-only JAX works but is too slow to iterate — expect
  the GPU path.

## Install prerequisites

From the sculptor uv env:

```bash
cd ~/projects/RewardSculptor
uv add "brax>=0.12" "jax[cuda12]>=0.7" "mujoco-mjx>=3.3"
```

MJX uses XLA's CUDA runtime; it coexists with mjlab's Warp-CUDA but
initialising both in the same process is wasteful. The recommended
setup mirrors mjlab: `MjxAdapter.train()` subprocess-spawns a runner
that imports Brax + MJX only once per run.

## Reward injection

Brax envs accept a custom `reward_fn` at construction time. The
sculptor reward module must export `compute_reward_batched` as a
JAX-pure callable — **no Python branching on traced values**. Verify
via a JIT preflight in `train()`:

```python
import jax
@jax.jit
def _preflight(sculpt_fn, dummy_state, dummy_action):
    return sculpt_fn(dummy_state, dummy_action, dummy_state, {})
```

If `_preflight` raises a `TracerError` or similar, surface the trace
error in a pre-flight validation step and abort before spawning the
trainer — same pattern as `sculptor/edit.py`'s post-flight import.

Injection example for a Brax env:

```python
from brax.envs import create

def _reward_fn(state, action, next_state):
    # Call through to the sculpted module. Must be JAX-pure.
    r, _components = sculpt_mod.compute_reward_batched(
        state, action, next_state, {"step": state.info["step"]},
    )
    return r  # shape (num_envs,) on the JIT device

env = create(env_name=self.env_id, reward_fn=_reward_fn)
```

For envs that don't accept `reward_fn` at construction, subclass
`brax.envs.Env` and override `_compute_reward`.

## Minimal viable train() skeleton

```python
def train(self, reward_module_path, output_dir, steps, seed):
    from brax.training.agents.ppo import train as ppo_train

    sculpt_mod = _load_reward_module(reward_module_path)
    env = _make_env_with_reward(self.env_id, sculpt_mod)

    # PPO on Brax is a single function call; params come back as a
    # JAX pytree which we pickle into checkpoint.pkl.
    make_inference_fn, params, _metrics = ppo_train(
        environment=env,
        num_timesteps=steps,
        num_envs=self.num_envs,
        seed=seed,
    )

    import pickle
    (output_dir / "checkpoint.pkl").write_bytes(pickle.dumps(params))
    _write_metrics(output_dir, _metrics)
```

## Testing strategy

The smoke test should JIT-preflight the reward module then run a tiny
(~20k env-step) PPO training and assert:
1. The checkpoint pkl exists and deserialises.
2. The metrics dict includes an `episode_reward_mean` key.
3. No `TracerError` escaped from the reward module.

JAX compilation dominates first-run wall-clock (~60-90 s); the test
budget should reflect this.

## Known gotchas

- **Pure-functional constraint.** Any Python dict of tensors the
  sculptor reward module wants to read MUST be converted to a JAX
  PyTree before the JIT boundary — no `torch.Tensor` survives. Explicit
  `jax.numpy.asarray(v)` conversion inside `_reward_fn`.
- **Determinism.** Brax's scan over timesteps requires a seeded PRNG
  key threaded through every reward call. If `compute_reward_batched`
  uses randomness, accept an `rng` kwarg or be RNG-free. Sculptor's
  default reward template is RNG-free.
- **vmap vs pmap.** Single-GPU training uses vmap; multi-GPU uses
  pmap. Sculptor's scope is single-GPU through M6, so stay on vmap.
- **Numerical parity with MuJoCo-CPU.** MJX is float32; MuJoCo CPU is
  float64. Rewards tuned on CPU may need re-tuning on MJX. Document
  this in the project's README.
- **The AME456 precedent.** `sculptor/reward.py` has a legacy v1
  quadruped reward that was originally targeted at MJX. It's a useful
  reference for the "JAX-pure reward module" shape but is NOT
  currently used by any adapter.

## References

- [Brax — GitHub](https://github.com/google/brax)
- [MuJoCo — GitHub](https://github.com/google-deepmind/mujoco)
- [MJX — documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)

## Completion checklist

- [ ] Install Brax + JAX + mujoco-mjx.
- [ ] Write a JAX-pure reward-template `rewards/v0.py` and verify JIT.
- [ ] Implement `MjxAdapter.train()` with the PPO call above.
- [ ] Implement `rollout()` — single-env, CPU render for reproducibility.
- [ ] Add an MJX-ready column to `robot_library.yml` for applicable
  robots (start with Ant / HalfCheetah / Humanoid — they already have
  Brax envs).
- [ ] Write `tests/test_mjx_adapter.py` (mocked + a small GPU smoke).
- [ ] Flip `ADAPTER_REGISTRY["mjx"].status` to `"ready"`.
