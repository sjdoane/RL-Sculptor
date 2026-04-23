# Isaac Lab adapter — adoption guide

**Status:** scaffolded (`sculptor.adapters.isaac_lab.IsaacLabAdapter`).
`train()` / `rollout()` raise `NotImplementedError`.
**Estimated effort:** 4-8 hours for a senior Isaac Lab user.

Isaac Lab is NVIDIA's RL framework built on Isaac Sim (Omniverse).
It's the closest analogue to mjlab's manager-based API and would be
the natural next adapter for sculptor to grow into. Same shape as
mjlab's reward-injection path, different runtime.

## Target versions

- Isaac Lab `>= 2.0.x` (see the project's `RELEASE.md` for the active
  minor).
- Isaac Sim `>= 4.5`.
- CUDA 12.4+ (matches the mjlab floor).
- Linux x86_64. **WSL2 works for headless training** but Isaac Sim's
  GUI is awkward through WSL2's display forwarding — run headless
  (`HEADLESS=1`) from sculptor.

## Install prerequisites

Isaac Sim is heavy: budget ~60 GB of disk. The recommended install
path is the bundled installer that ships with Isaac Lab:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

The installer sets up Isaac Sim + Isaac Lab in a conda/mamba env. From
sculptor's uv env, you either:

1. **Path-activate** Isaac Lab's python in the sculptor subprocess via
   `cfg.adapter.config.isaaclab_python = "/path/to/isaaclab/.conda/bin/python"`
   and have `IsaacLabAdapter.train()` spawn that interpreter explicitly.
2. **Add as an optional dep** to sculptor via `uv add --optional
   isaac "isaaclab"` once Isaac Lab ships a PyPI wheel.

Option 1 is the current recommendation — it keeps the sculptor env
clean and matches how mjlab handles its heavy CUDA-graph-compiling
dependencies.

## Reward injection

Isaac Lab uses a manager-based RL env identical in shape to mjlab's
(both drew from Isaac Gym's `LeggedRobot`). Patch the task's
`RewardManagerCfg` the same way `sculptor/adapters/_mjlab_runner.py`
does for mjlab:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg

cfg: ManagerBasedRLEnvCfg = load_isaac_task_cfg(task_id)
cfg.rewards.scale_rewards_by_dt = False  # keep raw module output
# Zero task-default terms so the sculpted reward is the entire signal.
for k, term in list(cfg.rewards.items()):
    if term is not None and hasattr(term, "weight"):
        term.weight = 0.0
cfg.rewards["sculptor_primary"] = RewardTermCfg(
    func=SculptorRewardTerm,        # class-based, same shape as mjlab's
    weight=1.0,
    params={"reward_module_path": args.reward_module_path},
)
```

`SculptorRewardTerm` is the same class from the mjlab runner, adapted
to read state off Isaac Lab's `env.scene[robot_name].data` (same
attribute names — joint_pos, joint_vel, root_lin_vel_b, etc.).

## Minimal viable train() skeleton

```python
def train(self, reward_module_path, output_dir, steps, seed):
    # 1. Build Isaac Lab task cfg + inject sculpt reward term (above).
    # 2. Launch Isaac Lab's bundled trainer as a subprocess, pointing
    #    --log-dir at output_dir:
    cmd = [
        self.isaaclab_python,
        "-m", "isaaclab.scripts.reinforcement_learning.rsl_rl.train",
        "--task", self.task,
        "--headless",
        "--num_envs", str(self.num_envs),
        "--max_iterations", str(steps),
        "--log_dir", str(output_dir / "logs"),
    ]
    env = {**os.environ, "HEADLESS": "1", "LIVESTREAM": "0"}
    _run_with_cleanup(cmd, env=env)  # reuse mjlab's subprocess helper
    # 3. Scan output_dir/logs/**/model_<N>.pt for the latest checkpoint
    #    and copy to output_dir/checkpoint.pt.
```

## Testing strategy

First real smoke test should be the mjlab Go1 equivalent — a 100-iter
run at `num_envs=1024` that produces a checkpoint. Mirror
`tests/test_mjlab_gpu.py::test_mjlab_train_produces_checkpoint`:

```python
@pytest.mark.gpu
def test_isaac_train_produces_checkpoint(isaac_go1_checkpoint):
    assert isaac_go1_checkpoint.is_file()
    assert isaac_go1_checkpoint.stat().st_size > 1024
```

Reuse the `tests/conftest.py` fixture-cache pattern; budget 10 min for
first run (Isaac Lab's cold start is heavier than mjlab's).

## Known gotchas

- **Isaac Sim license.** Free for research; commercial users need an
  Omniverse license. The installer prompts.
- **60+ GB disk.** Isaac Sim pulls a lot of assets. Make sure
  `$HOME` has headroom before `./isaaclab.sh --install`.
- **Reset semantics.** Isaac Lab's rsl_rl integration calls `env.reset()`
  at iteration boundaries; the `SculptorRewardTerm.reset(env_ids)` hook
  must zero `prev_state[env_ids]` (subset), not the whole buffer —
  otherwise rewards spike on individual-env timeouts. Same rule as
  mjlab.
- **Device pinning.** Isaac Lab defaults to `cuda:0`; multi-GPU is
  `--device cuda:N` but sculptor's single-GPU scope (post-M6) hasn't
  been exercised on Isaac Lab.
- **Task registry.** Unlike mjlab, Isaac Lab's task names include the
  `v0` suffix. See `isaaclab.utils.task.task_id_to_gym_env`.

## References

- [Isaac Lab — GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab — documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim — install guide](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)

## Completion checklist

- [ ] Install Isaac Lab and verify `isaaclab.sh -p` runs.
- [ ] Implement `SculptorRewardTerm` mirror (copy from
  `sculptor/adapters/_mjlab_runner.py`).
- [ ] Implement `IsaacLabAdapter.train()` + `rollout()` subprocess paths.
- [ ] Extend `robot_library.yml` with `training_support: isaac_ready`
  entries pointing at Isaac Lab-registered tasks.
- [ ] Update the library YAML D-guard in `backend/services/robot_library.py`
  to also cross-reference Isaac Lab's registry.
- [ ] Write `tests/test_isaac_adapter.py` (mocked subprocess) + a
  `@pytest.mark.gpu_isaac` smoke test.
- [ ] Update the root README's adapter-support table.
- [ ] Flip `ADAPTER_REGISTRY["isaac"].status` from `"coming_soon"` to
  `"ready"`.
