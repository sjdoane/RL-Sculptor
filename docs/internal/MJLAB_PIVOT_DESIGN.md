# mjlab Pivot — Design Doc (M0)

Scope: pivot Reward Sculptor's primary adapter from `GymSB3Adapter` to a new
`MjlabAdapter` (GPU MuJoCo-Warp, Isaac-Lab-style manager API), plus a
Menagerie-seeded robot library, GPU-aware UI, coming-soon adapter stubs, and
a migration path for existing Gymnasium projects.

Status: **design only**. No code in this doc. Push back before M2.

Research sources (all URLs verified via WebFetch during synthesis):

- mjlab core — [github.com/mujocolab/mjlab](https://github.com/mujocolab/mjlab), docs [mujocolab.github.io/mjlab/main/](https://mujocolab.github.io/mjlab/main/), PyPI [mjlab 1.3.0](https://pypi.org/project/mjlab/).
- mjlab ecosystem — [mjlab_playground](https://github.com/mujocolab/mjlab_playground), [g1_spinkick_example](https://github.com/mujocolab/g1_spinkick_example), [anymal_c_velocity](https://github.com/mujocolab/anymal_c_velocity).
- mjlab paper — Zakka et al., *mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning*, [arXiv 2601.22074](https://arxiv.org/abs/2601.22074) (2026).
- Menagerie — [github.com/google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (63 robot directories enumerated).
- MuJoCo-Warp — [github.com/google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp), pinned by mjlab to git rev `ea7f05b` (PyPI `mujoco-warp==3.7.0.1`).

---

## 1. mjlab adapter architecture

### 1.1 API surface we're adapting to

**Env instantiation** ([mjlab/envs/manager_based_rl_env.py:173-194](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/envs/manager_based_rl_env.py)):

```python
class ManagerBasedRlEnv:
    is_vector_env = True
    def __init__(self, cfg: ManagerBasedRlEnvCfg, device: str,
                 render_mode: str | None = None, **kwargs): ...
```

Task lookup via registry ([mjlab/tasks/registry.py:22-55](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/registry.py)):

```python
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go1")   # deepcopied
# mutate cfg.rewards / cfg.scene.num_envs here
env = ManagerBasedRlEnv(cfg, device="cuda:0")
```

**Managers loaded in fixed order** by `ManagerBasedRlEnv.load_managers()`
(`manager_based_rl_env.py:295-346`):

| # | Manager | Purpose |
|---|---------|---------|
| 1 | `EventManager` | Domain-randomization / reset / interval / startup / step hooks |
| 2 | `CommandManager` / `NullCommandManager` | Goal-conditioned command generators (velocity targets etc.) |
| 3 | `ActionManager` | Splits policy output into per-entity action terms, applies scale/offset/clip |
| 4 | `ObservationManager` | Concatenates grouped obs terms, applies noise→clip→scale→delay→history |
| 5 | `TerminationManager` | Bool OR of terms; separates `.terminated` from `.time_outs` |
| 6 | `RewardManager` | Weighted sum of `RewardTermCfg` entries, dt-scaled by default |
| 7 | `CurriculumManager` / `NullCurriculumManager` | Per-episode env-param mutation |
| 8 | `MetricsManager` / `NullMetricsManager` | Custom per-step metrics logged as episode averages |
| 9 | `RecorderManager` / `NullRecorderManager` | Pre/post-reset + post-step obs/action trace hooks |

`Scene` is **not** a manager — it's constructed before managers
(`manager_based_rl_env.py:188`). Sensors are tuples on `SceneCfg.sensors`.

**Reward term invocation** ([mjlab/managers/reward_manager.py:119-125](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/managers/reward_manager.py)):

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
```

A reward term is:

```python
def my_reward(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    # returns (num_envs,) float32 on env.device
```

Class form (stateful, with `reset(env_ids)` hook) — `mjlab/envs/mdp/rewards.py:98-128`:

```python
class posture:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv): ...
    def __call__(self, env, **params) -> torch.Tensor: ...
    def reset(self, env_ids): ...   # optional — called on per-env reset
```

`ManagerTermBaseCfg.func` may be a function OR a class; if it's a class the
manager auto-instantiates with `(cfg, env)` (`mjlab/managers/manager_base.py:136-138`).

**Tensor shapes during `.step()`** (all on `env.device`):

| Quantity | Shape | dtype |
|----------|-------|-------|
| `action` input | `(num_envs, total_action_dim)` | `float32` |
| `obs_buf` (group, concatenated) | `(num_envs, obs_dim)` | `float32` |
| `obs_buf` (history flattened) | `(num_envs, obs_dim * history)` | `float32` |
| `reward_buf` | `(num_envs,)` | `float32` |
| `reset_terminated` | `(num_envs,)` | `bool` |
| `reset_time_outs` | `(num_envs,)` | `bool` |

`step()` returns `(obs, reward, terminated, time_outs, extras)` — note it
does NOT return a merged `done`. The sculptor adapter mirrors Gymnasium's
`(terminated, truncated)` split for parity; `RslRlVecEnvWrapper` OR-merges
them for training (`mjlab/rl/vecenv_wrapper.py:73-89`).

### 1.2 Proposed adapter class

```
sculptor/adapters/mjlab.py
    MjlabAdapter(SculptorAdapter)
        __init__(
            task_id: str,                  # e.g. "Mjlab-Velocity-Flat-Unitree-Go1"
            num_envs: int = 1024,          # overridden per-device in §3
            device: str = "auto",          # "auto" | "cpu" | "cuda:N"
            seed: int = 1,
            rsl_rl_kwargs: dict = {},
            render_fps: int = 50,
        )
        train(reward_module_path, output_dir, steps, seed) -> TrainResult
        rollout(checkpoint_path, output_dir, n_episodes) -> RolloutResult
        reward_contract() -> RewardContract
        compute_behavior_metrics(rollout) -> dict
```

Fields set at `__init__`:

- `self._base_cfg` — `load_env_cfg(task_id)` called lazily (deepcopied each
  `train()` so sculpt iterations get fresh cfgs).
- `self._state_schema` — derived once by introspecting a `num_envs=1` env
  (observation-group tensors + action-dim), saved for `reward_contract()`.

### 1.3 Reward injection — the core design problem

**Problem.** The sculptor's reward-module contract is
`compute_reward(state, action, next_state, info) -> (reward, components)` —
a push-based, per-env scalar. mjlab's reward terms are pull-based
(`func(env, **params) -> (num_envs,) tensor`) and must be vectorised.

**Solution.** Two-pronged:

1. **Extend RewardContract** with `supports_batched: bool` and
   `state_schema: dict[str, tuple[int, ...]] | None`. Reward modules that
   target mjlab export both the scalar `compute_reward` (for validation +
   the UI's ComponentProbe) and a `compute_reward_batched` entry point with
   identical keys and shape convention.

2. **A class-based mjlab reward term** in `MjlabAdapter` wraps the reward
   module. The term:

   ```
   class SculptorRewardTerm:
       def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
           self.reward_module = load_reward_module(cfg.params["reward_module_path"])
           self.prev_state = snapshot_state(env)          # initial snapshot
           self.component_sums = {}                        # running episode sums

       def __call__(self, env, **kwargs) -> torch.Tensor:
           next_state = snapshot_state(env)               # dict of (num_envs, D) tensors
           action = env.action_manager.action
           rewards, components = self.reward_module.compute_reward_batched(
               self.prev_state, action, next_state, self._build_info_dict(env)
           )
           self._accumulate_components(components, env.episode_length_buf)
           self.prev_state = {k: v.detach().clone() for k, v in next_state.items()}
           return rewards

       def reset(self, env_ids):
           # called by the reward manager on per-env reset
           for k in self.prev_state: self.prev_state[k][env_ids] = 0
           for k in self.component_sums: self.component_sums[k][env_ids] = 0
   ```

   This term is registered as the **sole** active term:

   ```
   cfg.rewards = {"sculptor_primary": RewardTermCfg(
       func=SculptorRewardTerm,
       weight=1.0,
       params={"reward_module_path": reward_module_path},
   )}
   # all other task-provided reward terms set weight=0.0 (kept for introspection)
   cfg.scale_rewards_by_dt = False   # we want the module's raw per-step reward
   ```

   `scale_rewards_by_dt=False` is critical — otherwise mjlab's default
   `cfg.scale_rewards_by_dt=True` (manager_based_rl_env.py:156) multiplies
   every term by `step_dt`, and the sculpted reward no longer matches what
   the LLM-rewritten module returned.

### 1.4 State snapshot — what goes into `state` / `next_state`

A per-adapter, per-robot dict with fixed keys. For velocity-tracking tasks
(G1, Go1, ANYmal-C) the schema is:

| Key | Shape | Source |
|-----|-------|--------|
| `qpos` | `(N, nq)` | `env.scene["robot"].data.joint_pos` |
| `qvel` | `(N, nv)` | `env.scene["robot"].data.joint_vel` |
| `base_lin_vel_b` | `(N, 3)` | `env.scene["robot"].data.root_lin_vel_b` |
| `base_ang_vel_b` | `(N, 3)` | `env.scene["robot"].data.root_ang_vel_b` |
| `projected_gravity_b` | `(N, 3)` | `env.scene["robot"].data.projected_gravity_b` |
| `actuator_force` | `(N, nu)` | `env.scene["robot"].data.actuator_force` |
| `command_vel` | `(N, 3)` | `env.command_manager.get_command("base_velocity")` |

`info` is a dict of scalars per env:
- `episode_length` — `env.episode_length_buf`
- `time_outs` — `env.termination_manager.time_outs`
- `terminated` — `env.termination_manager.terminated`
- `step_dt` — `env.step_dt` (scalar, broadcast)

For each mjlab task, `MjlabAdapter._state_schema` is a static dict baked
into the class (simpler than introspecting the scene graph at runtime).
Manipulation tasks (Yam) extend with `ee_pose`, `object_poses`. All schema
keys go into `RewardContract.expected_info_keys` so the edit.py pre-flight
validator treats them as grounded.

### 1.5 Train / rollout / behavior metrics

- `train()` — builds the cfg (with sculptor reward term injected), wraps
  with `RslRlVecEnvWrapper`, runs `OnPolicyRunner` (`rsl-rl-lib==5.0.1`,
  the pin mjlab ships). Honors `steps` exactly by computing
  `max_iterations = steps // (cfg.scene.num_envs * cfg.episode_length_s / cfg.sim.dt)`.
  Writes `checkpoint.pt`, `metrics.json`, `reward_spec.json`, `logs/`.
- `rollout()` — reloads cfg with `num_envs=1` + `render_mode="rgb_array"`,
  plays policy deterministically for `n_episodes`, writes `rollout.mp4`,
  `keyframes/`, `trajectory.npz`, `behavior.json`. Rollout runs on CPU by
  default (see §3).
- `compute_behavior_metrics()` — default dict
  `{ mean_return, mean_episode_length, fall_rate, mean_base_forward_velocity,
  command_tracking_error, termination_reason_counts }` for locomotion
  tasks; `{ success_rate, time_to_success, object_pose_error }` for
  manipulation. Per-task overrides via subclass or config.

---

## 2. Parallel envs vs component probe

### 2.1 The probe modes

Two probes, both mandatory for mjlab-bound reward modules, one mandatory
for Gym-bound:

| Mode | Input | Output shape | Caller | Purpose |
|------|-------|--------------|--------|---------|
| `scalar` | zero-dummy dict of per-key `(D,)` arrays + zero `(A,)` action + empty info | scalar reward + `dict[str, float]` | `edit.py` pre-flight; UI ComponentProbe | Validate signature; UI shows contribution breakdown |
| `batched` | dict of per-key `(N, D)` tensors + `(N, A)` action + broadcast info, where `N=4` | `(N,)` tensor + `dict[str, (N,)] tensor` | `MjlabAdapter.train` preflight | Catch shape/device/dtype bugs before 4096-env training |

### 2.2 Reward-module contract (new)

```python
# rewards/v0.py (mjlab-compatible)
from typing import Any
import torch

def compute_reward(state: dict, action, next_state: dict, info: dict) -> tuple[float, dict[str, float]]:
    """Single-env scalar reward. Required by ALL adapters."""
    ...

def compute_reward_batched(
    state: dict[str, torch.Tensor],
    action: torch.Tensor,
    next_state: dict[str, torch.Tensor],
    info: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batched (num_envs,) reward. Required iff REWARD_SPEC['supports_batched']=True."""
    ...

REWARD_SPEC = {
    "version": "v0",
    "parent_hash": None,
    "author": "human",
    "description": "...",
    "hyperparameters": { "alive_bonus": 1.0, ... },
    "references": [],
    "supports_batched": True,          # NEW
}
```

### 2.3 Probe implementation

A new module `sculptor/probe.py`:

```python
def probe_scalar(reward_module_path: Path, contract: RewardContract) -> ProbeResult
def probe_batched(reward_module_path: Path, contract: RewardContract, device: str,
                  num_envs: int = 4) -> ProbeResult
```

Both run in a subprocess (same isolation pattern the UI already uses
for reward validation — protects the main process from import-time
crashes). `ProbeResult` carries `{ ok, reward_scalar | reward_tensor_stats,
components, stderr, import_error }`.

### 2.4 edit.py pre-flight extension

Today: `edit.py` validates the scalar `compute_reward` path. Extension:

1. Read `REWARD_SPEC["supports_batched"]`. If absent → treat as False.
2. If False and the adapter declares `supports_batched=True` in its
   contract → post-flight failure `"mjlab adapter requires supports_batched=True; reward module missing compute_reward_batched"`.
3. If True, run `probe_batched` after `probe_scalar`. Both must pass.

This is a two-line condition in `edit.py` post-flight; the LLM prompt's
`reward_contract` serialisation is extended to include the batched schema
so the rewriter gets the signature right on the first try.

### 2.5 UI ComponentProbe response

Unchanged shape — always scalar. The UI doesn't need batched values; it
just shows the per-component contribution for a canned zero state. The
backend-side probe adapter picks scalar for the UI call, batched for
training-time validation.

---

## 3. Device model

### 3.1 Three concerns, three defaults

| Concern | mjlab adapter | Gymnasium adapter | Preview (MuJoCo offscreen) |
|---------|---------------|-------------------|----------------------------|
| Training | GPU required (cuda:0 default) | CPU default; GPU allowed | — |
| Rollout rendering | CPU default (reproducibility, OSMesa-friendly) | CPU | — |
| Static preview | CPU always (`MUJOCO_GL=egl` or `osmesa`) | CPU always | CPU always |

Rationale:
- mjlab's training path explicitly relies on CUDA graphs with conditional
  execution — **minimum supported CUDA is 12.4**, recommended 12.8 per
  `docs/source/faq.rst:33-37` of mjlab and `pyproject.toml:99` `cu128`
  extra. On the RTX 5070 Laptop (sm_120, Blackwell) this works — the
  user-confirmed `uvx --from mjlab demo` is the proof.
- Gymnasium-SB3 stays CPU-default for parity with the Windows-era
  baseline and because SB3's PPO is CPU-bound on small nets anyway.
- Rollout mp4 rendering is single-env, short, and offscreen — no GPU
  benefit.

### 3.2 Device-resolution precedence (in `MjlabAdapter.__init__`)

```
1. cfg["adapter"]["config"]["device"] == "cuda:N"  -> use that; error-fast if unavailable
2. cfg["adapter"]["config"]["device"] == "cpu"     -> use CPU; warn mjlab is 50-100x slower
3. cfg["adapter"]["config"]["device"] == "auto"    -> prefer cuda:0 if torch.cuda.is_available()
                                                      AND mujoco_warp imports
                                                      AND CUDA runtime >= 12.4
                                                      else fall back to CPU with warning
4. omitted                                          -> "auto"
```

### 3.3 VRAM budget on 8 GB (RTX 5070 Laptop)

mjlab itself doesn't document exact per-robot VRAM; we infer from published
benchmarks in the mjlab paper (Zakka et al. 2026, arXiv 2601.22074) and
the mjlab_playground README showing Go1 getup trained on a 5090 in ~2 min
at `num_envs=4096`. Conservative defaults for 8 GB VRAM:

| Task family | Recommended `num_envs` | Rough peak VRAM |
|-------------|------------------------|-----------------|
| Cartpole (toy) | 4096 | < 1 GB |
| Velocity-Go1, Velocity-ANYmal-C (quadrupeds, ~18 DoF) | 2048 | 3–4 GB |
| Getup-Go1, Getup-T1 (quadruped/humanoid, similar DoF) | 2048 | 3–4 GB |
| Velocity-G1 (humanoid, 29 DoF) | 1024 | 4–5 GB |
| Tracking-G1 (adds motion-buffer state) | 768 | 5–6 GB |
| Spinkick-G1 (heavy tracking + contact) | 512 | 5–6 GB |
| Lift-Cube-Yam (arm, small state) | 2048 | 3–4 GB |
| Lift-Cube-Yam-Rgb / Depth (camera obs) | 512 | 6–7 GB |

**These are proposals, not measured.** First-task-after-M2 for the user is
to run each mjlab-ready task once at the proposed `num_envs` and read
`nvidia-smi` peak usage, then adjust this table. The UI surfaces the
proposed value and the measured one side-by-side once we have data.

### 3.4 UI exposure

**Settings page (`/settings`, existing).** New "GPU & training" panel
with:
- Detected CUDA toolkit version (via `torch.version.cuda` and
  `nvidia-smi`); green check if ≥ 12.4, yellow warn if older, red if
  CUDA missing.
- Detected VRAM (total + currently-used via `torch.cuda.mem_get_info()`).
- mjlab import health (yes/no + error message if no — extends the health
  check in [sculptor_bridge.py:21-26](reward-sculptor-ui/backend/services/sculptor_bridge.py:21) to also check `import mjlab`).
- mujoco_warp import health (separate — partial-install is a known class
  of failure; see mjlab issues #218 / #469).

**Project creation dialog.** When a user picks a mjlab-ready robot:
- `num_envs` slider defaults to the table above, min = 128, max = 4096.
- Tooltip: "Estimated VRAM: ~X.X GB at this `num_envs`. Total: Y.Y GB."
- If CUDA missing: dialog blocks at click — "mjlab requires an NVIDIA
  GPU with CUDA ≥ 12.4 (detected: none). Pick a gymnasium-compatible
  robot instead or install CUDA 12.4+."
- If VRAM headroom tight (estimated > 80% of total): yellow badge "May
  OOM; consider lowering num_envs or closing other GPU-heavy programs."

**Runs tab (existing live log).** GPU VRAM + utilisation mini-chart
streamed via the same WS the run-manager already uses. Row spec: `ts,
vram_used_bytes, sm_utilisation`. Polled server-side every 2 s during a
training run.

---

## 4. Robot library data model

### 4.1 Schema

```yaml
# backend/data/robot_library.yml — loaded at FastAPI startup
robots:
  - slug: unitree_go1
    display_name: "Unitree Go1"
    category: Quadruped
    description: "Unitree Go1 quadruped (~12 kg, 18 DoF)."
    source: menagerie                # menagerie | mjlab_builtin | gymnasium_builtin | custom
    menagerie_package: go1_mj_description
    training_support: mjlab_ready     # mjlab_ready | preview_only | gymnasium_compatible
    preconfigured_tasks:              # only meaningful when mjlab_ready
      - task_id: Mjlab-Velocity-Flat-Unitree-Go1
        display_name: "Velocity tracking (flat)"
        recommended_num_envs: 2048
      - task_id: Mjlab-Velocity-Rough-Unitree-Go1
        display_name: "Velocity tracking (rough)"
        recommended_num_envs: 2048
      - task_id: Mjlab-Getup-Flat-Unitree-Go1
        display_name: "Fall recovery (flat)"
        recommended_num_envs: 2048
    references:
      - kind: paper
        url: https://arxiv.org/abs/2212.03238
        citation: "Margolis & Agrawal, Walk These Ways, CoRL 2022"
      - kind: repo
        url: https://github.com/unitreerobotics/unitree_ros
        citation: "unitreerobotics / unitree_ros"
      # ... (full refs §4.3)
    thumbnail_path: robots/unitree_go1.webp
```

`source`:
- `menagerie` — installed via `pip install robot_descriptions`, URDF/MJCF fetched by package name.
- `mjlab_builtin` — ships inside the mjlab wheel (Cartpole).
- `gymnasium_builtin` — MuJoCo-Gymnasium envs (Hopper, Ant, etc.); no URDF upload needed.
- `custom` — user-uploaded.

### 4.2 Enumerated robots (seeded from research)

**All 63 Menagerie models** plus 1 mjlab_builtin (Cartpole) plus 5
gymnasium_builtin (existing: Hopper, Ant, Walker2d, HalfCheetah,
Humanoid). Total library size at M2: **69 entries**.

#### 4.2.1 `training_support: mjlab_ready` (6 robots, 14 tasks)

| slug | display_name | category | source | tasks | reference count |
|------|--------------|----------|--------|-------|-----------------|
| `cartpole_mjlab` | Cartpole (toy) | Other | mjlab_builtin | Balance, Swingup | 0 |
| `unitree_go1` | Unitree Go1 | Quadruped | menagerie | Velocity-Flat, Velocity-Rough, Getup-Flat | 5 |
| `anybotics_anymal_c` | ANYmal C | Quadruped | menagerie | Velocity-Flat, Velocity-Rough (via `anymal_c_velocity`) | 5 |
| `unitree_g1` | Unitree G1 | Humanoid | menagerie | Velocity-Flat, Velocity-Rough, Tracking-Flat, Tracking-Flat-NoStateEst, Spinkick | 5 |
| `booster_t1` | Booster T1 | Humanoid | menagerie | Getup-Flat | 3 |
| `i2rt_yam` | Yam arm (YAM) | Arm | menagerie | Lift-Cube, Lift-Cube-Rgb, Lift-Cube-Depth, Multi-Cube-Seg | 0 (follow-up) |

#### 4.2.2 `training_support: gymnasium_compatible` (5 robots)

The existing `LIBRARY_ROBOTS` array becomes a subset of the new library,
migrated in place. `env_id` becomes the task-id equivalent.

| slug | display_name | category | env_id | tasks |
|------|--------------|----------|--------|-------|
| `hopper` | Hopper | Other (biped) | `Hopper-v4` | — |
| `ant` | Ant | Quadruped | `Ant-v4` | — |
| `walker2d` | Walker2d | Other (biped) | `Walker2d-v4` | — |
| `halfcheetah` | HalfCheetah | Quadruped | `HalfCheetah-v4` | — |
| `humanoid` | Humanoid | Humanoid | `Humanoid-v4` | — |

#### 4.2.3 `training_support: preview_only` (58 Menagerie robots)

Full enumeration (from Agent A research on mujoco_menagerie `main`):

<details>
<summary>Preview-only robots (click to expand)</summary>

**Arm (19):** `agilex_piper`, `arx_l5`, `dynamixel_2r`, `flexiv_rizon4`,
`franka_emika_panda`, `franka_fr3`, `franka_fr3_v2`, `kinova_gen3`,
`kuka_iiwa_14`, `low_cost_robot_arm`, `rethink_robotics_sawyer`,
`robotstudio_so101`, `trossen_vx300s`, `trossen_wx250s`, `trossen_wxai`,
`trs_so_arm100`, `ufactory_lite6`, `ufactory_xarm7`, `unitree_z1`,
`universal_robots_ur10e`, `universal_robots_ur5e`. *(Yam promoted to
mjlab-ready.)*

**Biomechanical (2):** `flybody`, `ms_human_700`.

**Drone (2):** `bitcraze_crazyflie_2`, `skydio_x2`.

**Hand (8):** `leap_hand`, `robotiq_2f85`, `robotiq_2f85_v4`,
`shadow_dexee`, `shadow_hand`, `tetheria_aero_hand_open`, `umi_gripper`,
`wonik_allegro`.

**Humanoid (9):** `apptronik_apollo`, `berkeley_humanoid`, `fourier_n1`,
`pal_talos`, `pndbotics_adam_lite`, `robotis_op3`, `toddlerbot_2xc`,
`toddlerbot_2xm`, `unitree_h1`. *(G1 and T1 promoted to mjlab-ready.)*

**Mobile (6):** `google_robot`, `hello_robot_stretch`,
`hello_robot_stretch_3`, `pal_tiago`, `pal_tiago_dual`, `robot_soccer_kit`,
`stanford_tidybot`.

**Other (3):** `aloha`, `iit_softfoot`, `realsense_d435i`.

**Quadruped (8):** `agility_cassie`, `anybotics_anymal_b`,
`boston_dynamics_spot`, `google_barkour_v0`, `google_barkour_vb`,
`unitree_a1`, `unitree_go2`. *(Go1 and ANYmal-C promoted to mjlab-ready.)*

</details>

### 4.3 References (KG seeds) for the 6 mjlab-ready robots

All URLs verified in research pass.

**Unitree Go1** (5):
- paper https://arxiv.org/abs/2212.03238 — Margolis & Agrawal, *Walk These Ways*, CoRL 2022
- paper https://arxiv.org/abs/2208.07860 — Smith et al., *A Walk in the Park*, arXiv 2022
- paper https://arxiv.org/abs/2109.11978 — Rudin et al., *Learning to Walk in Minutes* (legged_gym), CoRL 2021
- repo https://github.com/Improbable-AI/walk-these-ways
- repo https://github.com/unitreerobotics/unitree_ros

**ANYmal C** (5):
- paper https://arxiv.org/abs/2010.11251 — Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*, Science Robotics 2020
- paper https://arxiv.org/abs/1901.08652 — Hwangbo et al., *Learning Agile and Dynamic Motor Skills*, Science Robotics 2019
- paper https://arxiv.org/abs/2109.11978 — Rudin et al. (shared with Go1)
- repo https://github.com/ANYbotics/anymal_c_simple_description
- repo https://github.com/google-deepmind/mujoco_menagerie (for the `anybotics_anymal_c` MJCF)

**Unitree G1** (5):
- paper https://arxiv.org/abs/2406.08858 — He et al., *OmniH2O*, CoRL 2024
- paper https://arxiv.org/abs/2412.13196 — Ji et al., *ExBody2*, arXiv 2024
- paper https://arxiv.org/abs/2402.19469 — Radosavovic et al., *Humanoid Locomotion as Next Token Prediction*, NeurIPS 2024
- repo https://github.com/unitreerobotics/unitree_rl_lab
- repo https://github.com/unitreerobotics/unitree_ros

**Booster T1** (3):
- paper https://arxiv.org/abs/2506.15132 — Wang et al., *Booster Gym*, arXiv 2025
- repo https://github.com/BoosterRobotics/booster_gym
- repo https://github.com/BoosterRobotics/booster_assets

**Cartpole (mjlab_builtin)** (0): toy task, no canonical KG seeds.

**Yam arm** (0 for v1): the canonical Yam paper search returned nothing
high-signal in the research pass; the `i2rt-robotics/i2rt` upstream repo
is cited but not a KG seed. Flag as follow-up — user to validate once
M4 lands.

**Shared (mjlab paper)** — every mjlab-ready entry auto-includes:
- paper https://arxiv.org/abs/2601.22074 — Zakka et al., *mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning*, arXiv 2026

### 4.4 Thumbnails

Thumbnails are the Menagerie README images (`{slug}.png` in most cases),
re-encoded to `.webp` and copied to `frontend/public/robots/{slug}.webp`
at build time. One-off script `scripts/build_thumbnails.py` reads
`robot_library.yml`, fetches the Menagerie image by path, encodes at
512×384 quality 80. Cartpole + Gymnasium entries reuse existing
thumbnails. Missing-image fallback: a placeholder svg.

---

## 5. Library → project flow

### 5.1 Backend: project scaffolding decision tree

Current endpoint (frontend calls `usePickLibraryRobot`):
`POST /projects/{slug}/robot/library` with `{ "name": "hopper" }`.

New behavior — the backend loads `robot_library.yml` once at startup and
exposes:

- `GET /library/robots` — returns the whole list (the UI's Library tab
  consumes this instead of the hardcoded `LIBRARY_ROBOTS`).
- `POST /projects/{slug}/robot/library` with
  `{ "robot_slug": "unitree_go1", "task_id": "...", "num_envs": 2048 }`.
  Behavior branches on `robot.training_support`:

```
mjlab_ready
  -> scaffold with adapter class = "sculptor.adapters.mjlab.MjlabAdapter"
  -> adapter config = { task_id, num_envs: picked-or-default, device: "auto" }
  -> kg_seeds.yml = references[] from the library entry + mjlab paper shared seed
  -> fire kg_jobs.run_ingest_extract_job(auto_extract=key_present)

gymnasium_compatible
  -> scaffold with adapter class = "sculptor.adapters.gym_sb3.GymSB3Adapter"
  -> adapter config = { env_id: entry.env_id, n_envs: 4 }
  -> kg_seeds.yml = existing default seeds (sculptor's hand-seeded set)

preview_only
  -> scaffold with adapter class = "sculptor.adapters.preview.PreviewOnlyAdapter" (NEW)
  -> train() raises RuntimeError("This robot has no mjlab task configured. Training disabled.")
  -> rollout() raises RuntimeError likewise
  -> reward_contract() returns a minimal stub so preview + reward-authoring UI still work
  -> kg_seeds.yml = empty
  -> UI Train tab renders disabled with tooltip "No mjlab task configured for this robot. See Settings → adapters to request one."

custom (uploaded URDF — unchanged from today)
  -> scaffold with gymnasium or preview-only adapter based on inference
```

### 5.2 UI: RobotConfig tab rewrite

Current [RobotConfig.tsx:70-99](reward-sculptor-ui/frontend/src/components/RobotConfig.tsx:70) renders all five library
robots as a flat grid. Changes for v1:

1. **Replace `LIBRARY_ROBOTS` constant** with a React Query hook
   `useLibraryRobots()` that fetches `GET /library/robots`.
2. **Add filter bar** above the grid:
   - Category chips (multi-select): Quadruped / Humanoid / Arm / Hand /
     Mobile / Drone / Biomechanical / Other.
   - Training-support chips (multi-select): "mjlab ready" / "Gymnasium" /
     "Preview only". Default selection: first two only (`mjlab_ready |
     gymnasium_compatible`); user can click "Preview only" to surface
     the other ~58.
   - Free-text search box (name / slug / description substring).
3. **Card badges.** Each `RobotCard` gets a badge row:
   - Training-support: green `mjlab ready`, blue `Gymnasium`, grey
     `Preview only`.
   - `robot_descriptions` install status badge (see §9).
4. **Task picker.** When `robot.training_support === "mjlab_ready"` and
   `preconfigured_tasks.length > 1`, clicking the card opens a modal:
   "Pick a task for this robot" + radio list with per-task description
   and recommended `num_envs` slider + GPU-info summary. Single task →
   click card commits directly.
5. **preview_only card** — clicking commits but surfaces an info toast
   "Preview only — training disabled. See Settings for how to add an
   mjlab task for this robot." Train button in the project remains
   disabled with tooltip.

### 5.3 Migration of existing `LIBRARY_ROBOTS`

The five current entries (Hopper, Ant, Walker2d, HalfCheetah, Humanoid)
migrate to the new schema as `source: gymnasium_builtin`,
`training_support: gymnasium_compatible`. No other change to how these
projects scaffold or train. Frontend types file
(`frontend/src/lib/types.ts`) keeps `LibraryRobotEntry` the old shape for
one release, deprecated; a new `LibraryRobotV2` type replaces it in the
library hook. Existing projects (which reference the old names) don't
care — they store `env_id`, not the library slug.

---

## 6. KG seeding from the library

### 6.1 Ingest flow

Current flow (`kg_jobs.run_ingest_extract_job` at
[backend/services/kg_jobs.py:22-121](reward-sculptor-ui/backend/services/kg_jobs.py:22)):
reads `<project>/kg_seeds.yml`, calls `sculptor.kg.ingest.ingest_from_seeds`,
then optionally `extract_all` if `ANTHROPIC_API_KEY` present.

**No change required to the runner.** The library entry's `references`
list is serialised into `kg_seeds.yml` at project-create time, keyed by
arxiv ID extracted from the URL. Sample transform:

```
library entry reference:          kg_seeds.yml entry:
- kind: paper                     - arxiv_id: "2212.03238"
  url: https://arxiv.org/abs/       title: "Walk These Ways"
  2212.03238                        authors: ["Margolis", "Agrawal"]
  citation: "Margolis & Agrawal,    year: 2022
  Walk These Ways, CoRL 2022"
```

For `kind: repo` references: the ingest doesn't handle GitHub repos
today (sculptor's `ingest` is arxiv-only). Repo references are stored in
the library YAML and surfaced in the KG tab's "Related repos" panel
(new, small — §6.3), **not** ingested into the sqlite KG. This keeps
the ingest job scope-identical to today.

### 6.2 Auto-trigger on project create

```
1. scaffold_project(...)             # writes kg_seeds.yml with library refs
2. if robot.training_support == "mjlab_ready":
     job_id = kg_jobs.run_ingest_extract_job(project_dir, auto_extract=key_present)
     return {project, kg_ingest_job: job_id}
3. else if gymnasium_compatible:
     same as today (no auto-ingest — user triggers from KG tab)
```

`auto_extract` follows existing logic: true only when the API key is
present. On mjlab project creation the backend returns the job id so the
frontend can show a toast "Ingesting 5 papers… (job #42) — see KG tab".

### 6.3 KG tab addition: "Related repos" panel

Small read-only panel listing `kind: repo` references from the library
entry. Each row: repo name, GitHub URL, one-line citation. No ingestion,
no KG rows. Purely cosmetic. Skip if the list is empty.

---

## 7. Coming-soon adapter stubs

### 7.1 File layout

```
sculptor/adapters/
    base.py               # existing
    gym_sb3.py            # existing
    mjlab.py              # NEW — M2
    preview.py            # NEW — M2 (used by preview_only library entries)
    isaac_lab.py          # NEW — M5 stub
    mjx.py                # NEW — M5 stub
    rllib.py              # NEW — M5 stub
```

### 7.2 Stub shape (same pattern for all three)

```python
# sculptor/adapters/isaac_lab.py
"""Isaac Lab adapter — STUB.

Status: scaffold only. Claimed interface exists so the Sculptor codebase
type-checks and the UI "adapter picker" has a real class to reference.
train() and rollout() are not implemented.

Missing to complete:
  1. `isaaclab` install + a working CUDA build (tested with 12.4+).
  2. Map Sculptor's `compute_reward(state, action, next_state, info)`
     onto Isaac Lab's reward term API (RewardTermCfg + ManagerBase) —
     this is structurally identical to the mjlab adapter; start there.
  3. Replace `train()` and `rollout()` NotImplementedError bodies.
  4. Extend `tests/` with an Isaac Lab contract test (gate on a
     platform marker — Isaac Lab is CUDA-only, Linux x86_64 only today).

See RewardSculptor/docs/adapters.md § Isaac Lab for the injection
recommendation and the reward-contract mapping.
"""

@dataclass
class IsaacLabAdapter(SculptorAdapter):
    task: str = "Isaac-Velocity-Flat-Unitree-A1-v0"
    num_envs: int = 4096
    device: str = "cuda:0"

    def reward_contract(self) -> RewardContract:
        # Stub — real values require building an Isaac Lab env once.
        # These are defensible defaults derived from the mjlab Go1
        # contract; swap in the real ones before un-stubbing.
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=["qpos", "qvel", "base_lin_vel_b",
                                "base_ang_vel_b", "projected_gravity_b",
                                "command_vel"],
            expected_components=None,
        )

    def compute_behavior_metrics(self, rollout) -> dict:
        return {
            "mean_return": 0.0,
            "mean_episode_length": 0,
            "fall_rate": 0.0,
            "adapter_status": "stub",
        }

    def train(self, *args, **kwargs) -> TrainResult:
        raise NotImplementedError(
            "IsaacLabAdapter.train() is not implemented. "
            "This is a scaffold; see the module docstring for the "
            "completion checklist. Pick the mjlab or gym_sb3 adapter "
            "for this project."
        )

    def rollout(self, *args, **kwargs) -> RolloutResult:
        raise NotImplementedError("IsaacLabAdapter.rollout() — see train().")
```

Same pattern for `MjxAdapter` (wrap Brax/MJX; the existing legacy
`sculptor/reward.py` hints at the injection shape) and `RllibAdapter`
(env_creator callback).

### 7.3 UI adapter picker (new-project dialog)

Radio list: `mjlab` (default for mjlab_ready) · `gymnasium` (default for
gymnasium_compatible) · `isaac_lab` / `mjx` / `rllib` (all disabled,
grey, "Coming soon" badge, tooltip links to the stub file's path in the
GitHub repo). Clicking a disabled option surfaces a toast with the
adapter's module docstring `Missing to complete:` section.

---

## 8. Migration path for existing projects

Your preference: (a) leave old projects on Gymnasium-SB3, with (b) a
manual "Upgrade to mjlab" button as an opt-in. Agreed — here's the
fleshed-out version.

### 8.1 (a) Leave-alone contract

- Existing `config.toml` files continue to declare
  `adapter.class = "sculptor.adapters.gym_sb3.GymSB3Adapter"`.
  No migration script, no backend rewrite. `sculpt run` and the UI
  Train tab keep working via the existing code path.
- The sculptor's `gym_sb3.py` stays in the tree and in the installed
  wheel — no removal, no deprecation warning.
- Reward modules in old projects (`rewards/v*.py`) continue to use the
  scalar-only contract (`supports_batched` defaulting False when missing
  from `REWARD_SPEC`).

### 8.2 (b) Opt-in "Upgrade to mjlab" button

Surfaces in the project's Settings tab, **only when all three hold**:

1. Project's current adapter class is `GymSB3Adapter`.
2. The library entry that seeded the project has
   `training_support == "mjlab_ready"` (i.e. the robot is now eligible).
   For projects scaffolded from Hopper / Ant / Walker2d / HalfCheetah /
   Humanoid this is never true — the button never shows for those.
3. The host has CUDA + mjlab importable.

Click flow:

```
1. confirmation dialog: "Upgrade <project-name> to mjlab?"
    - lists the target task_id and recommended num_envs
    - warns: "Existing reward modules will be re-validated against the
      batched contract. If they fail, upgrade is aborted — no files are
      modified."
    - warns: "Runs under runs/ are kept as historical; the next sculpt
      run starts a fresh run-id."

2. backend performs (in order):
    a. load current rewards/current.py; call probe_batched on it.
       If the module doesn't export compute_reward_batched or it fails,
       abort with a structured error listing what's missing.
    b. deepcopy config.toml to config.toml.gym-backup.
    c. rewrite [adapter] section to MjlabAdapter + task_id + num_envs.
    d. write a new rewards/vN+1.py auto-generated stub that wraps the
       current compute_reward with a trivial compute_reward_batched
       (vmap over the batch). Commit as "upgrade to mjlab: add batched
       reward wrapper".
    e. return the rebuild run-id.

3. frontend: toast + redirect to project's Train tab.
```

Rollback: the `config.toml.gym-backup` is kept for manual restore; no
one-click rollback in v1.

### 8.3 Projects created from scratch on v2+

These get the new adapter directly via §5.1. No migration needed.

---

## 9. Failure modes

For each, listing: detection point, user-visible message, remediation
link. All errors use the existing FastAPI ProblemDetail wrapping.

### 9.1 No NVIDIA GPU, user picks mjlab robot

- **Detect:** `torch.cuda.is_available() == False` AND robot is
  `mjlab_ready`. Checked in the library-pick endpoint BEFORE
  `scaffold_project` runs.
- **Response:** `400 Bad Request`,
  `type=no_cuda_for_mjlab`,
  `detail="mjlab requires an NVIDIA GPU with CUDA ≥ 12.4. Detected: no CUDA runtime. Pick a Gymnasium-compatible robot (Hopper / Ant / …) or install CUDA 12.4+ and restart."`.
- **UI:** dialog stays open, error banner inside the library-pick card.

### 9.2 GPU detected but insufficient VRAM for `num_envs`

- **Detect:** at preflight (before `scaffold_project`). Heuristic: the
  §3.3 table's `num_envs` × per-env memory estimate for the task family
  > `torch.cuda.mem_get_info()[0] * 0.85`.
- **Response:** `409 Conflict`, `type=insufficient_vram`, body carries
  `suggested_num_envs` (the largest power-of-two that fits the 85% bound).
- **UI:** the project-create dialog's num_envs slider jumps to the
  suggestion with a yellow toast "`num_envs` reduced from 4096 → 1024
  to fit 8.0 GB VRAM. You can override in Settings."

### 9.3 GPU detected but wrong CUDA version

- **Detect:** at FastAPI startup. `torch.version.cuda` is `None` or
  `< "12.4"`.
- **Response:** Settings page GPU panel shows red badge "CUDA version
  11.8 detected — mjlab requires 12.4+". Project-create for mjlab
  robots blocked identically to §9.1.
- Surfaced also via `GET /system/info` (existing endpoint) with new
  keys `cuda_version`, `cuda_version_ok`, `recommended_cuda` so the
  Dashboard and other surfaces consume uniformly.

### 9.4 mjlab install failed silently

- **Detect:** extend `sculptor_bridge._sculptor_import_error` coverage
  to also attempt `import mjlab` and `from mjlab.tasks.registry import
  list_tasks`. Store error separately as `_mjlab_import_error`.
- **Response:** Settings GPU panel row "mjlab importable: no — `<error
  class>: <first 200 chars>`". New-project adapter picker disables the
  `mjlab` radio with the same detail in tooltip.
- Extends the existing `sculptor_ok() / sculptor_error()` pattern at
  [sculptor_bridge.py:29-36](reward-sculptor-ui/backend/services/sculptor_bridge.py:29) — mjlab is a separate
  optional subsystem, not a sculptor-critical dep.

### 9.5 `robot_descriptions` not installed

- **Detect:** import probe once at startup. If missing, the library
  `source: menagerie` entries all render with a grey overlay on their
  thumbnail and a badge "Install: `pip install robot_descriptions`".
- Clicking such a card shows a modal with the install command and a
  "Copy to clipboard" button. No auto-install — user runs it in their
  shell.
- Gymnasium-builtin entries unaffected; Cartpole unaffected.

### 9.6 Menagerie model missing meshes after install

- **Detect:** on first use of the robot (static preview render). The
  preview renderer catches MuJoCo's `XML Error: Mesh file not found`
  and returns a structured error.
- **Response:** `422 Unprocessable Content`, `type=menagerie_mesh_missing`,
  `detail="<slug>: mesh '<filename>' not found under robot_descriptions package. Re-install with `pip install --force-reinstall robot_descriptions`."`.
- UI: preview panel shows error card with the command and a retry
  button.

### 9.7 `mujoco_warp` missing after mjlab install

Known partial-install state (mjlab issue #469, #218).

- **Detect:** at startup, separate `import mujoco_warp` attempt. Error
  stored at `_mujoco_warp_import_error`.
- **Response:** Settings GPU panel shows specific message
  "mujoco_warp not importable: `<error>`. Common fix:
  `uv add 'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@ea7f05b'` (the rev mjlab pins)."
- Same disable treatment as §9.4 — mjlab adapter picker is disabled
  with this specific remediation.

---

## 10. Deferrals

Explicit out-of-scope for this pivot series (M0 → M6). Restated from
your prompt with one clarification per:

1. **No Isaac Lab / MJX / RLlib full implementations.** Stubs only per
   §7. Completion is an external contribution opportunity (as already
   called out in `CONTRIBUTING.md`).
2. **No macOS or native Windows training paths.** mjlab's PyPI wheel
   ships no macOS torch+cu128 wheel (issue #306) and native-Windows
   CUDA on sm_120 isn't a tested mjlab target. WSL2/Linux only. The
   Gymnasium-SB3 adapter continues to work on macOS and native
   Windows — it's the fallback for non-Linux dev.
3. **No automatic mesh upgrading for existing custom-uploaded robots.**
   If a uploaded URDF stops rendering after a sculptor release, the
   user re-uploads. Covered separately by the custom-robot upload
   flow.
4. **No simultaneous multi-adapter runs.** One adapter per project;
   switching requires the §8.2 upgrade flow.
5. **No cloud GPU provisioning.** Train path is local-GPU only. UI
   doesn't know how to spawn on RunPod/Vast/Modal.
6. **No multi-GPU training in v1.** mjlab's `--gpu-ids` is a proven
   code path but we're single-GPU-only; the adapter hard-codes
   `device="cuda:0"` or `"cpu"`. Multi-GPU lands post-M6 if the lab
   gets a multi-GPU machine.

---

## Milestone breakdown (proposed for your push-back)

- **M0 — this doc.** You push back; I revise; we freeze.
- **M1** — Robot library foundation. YAML schema, 69 entries seeded,
  `GET /library/robots` endpoint, frontend migrates to the hook (still
  shows old 5 robots as the only enabled ones; mjlab-ready and
  preview-only entries appear but do nothing yet). Thumbnails baked.
  New tests for the library loader.
- **M2 — MjlabAdapter + probe.** `sculptor/adapters/mjlab.py`,
  `sculptor/probe.py`, `RewardContract.supports_batched`, batched
  reward contract, pre-flight + post-flight validator extension. Live
  smoke: Go1 velocity-flat 3-iter dry-run under 60 s.
- **M3 — GPU UI.** Device-resolution, Settings GPU panel, project-
  create dialog with num_envs slider + VRAM estimator, Runs-tab
  GPU mini-chart.
- **M4 — KG seeding on create.** Library → `kg_seeds.yml`
  transformation, auto-trigger ingest on mjlab project creation,
  "Related repos" panel in KG tab.
- **M5 — Adapter stubs + preview-only adapter.** `PreviewOnlyAdapter`,
  `IsaacLabAdapter`, `MjxAdapter`, `RllibAdapter` stubs; UI adapter
  picker with disabled options.
- **M6 — Migration button.** The "Upgrade to mjlab" button in the
  project Settings tab per §8.2, including the auto-generated
  `compute_reward_batched` wrapper.

Post-M6: measured VRAM benchmarks to replace §3.3's estimates, Yam
paper-reference discovery, first full end-to-end live mjlab training run
logged and archived in `HISTORY.md` with timings.

---

## Open questions — I want your read before M1

1. **Cartpole's Menagerie status.** Cartpole is an mjlab-builtin toy, not
   in Menagerie. Do you want it in the library at all, or hidden? I have
   it in §4.2.1 because it's the fastest smoke-test target; happy to
   move it to a dev-only/hidden flag if it clutters the UI for users
   who aren't debugging.
2. **`supports_batched` default.** I'm proposing this defaults to
   `False` in `REWARD_SPEC` so existing reward modules don't break. The
   downside: a user's gym project won't accidentally gain mjlab
   compatibility from a reward rewrite — they'd need to flip the flag.
   Acceptable?
3. **Yam references gap.** No canonical paper found. OK to ship M4
   with Yam references empty + a TODO, or should M4 block on finding
   one?
4. **§3.3 VRAM estimates.** These are rough — I'd prefer to measure
   them rather than publish them. Option: ship M3 with a "Measured:
   pending" label next to each estimate until the measurement task runs.
5. **KG repo references.** I'm storing `kind: repo` references in the
   library YAML but NOT ingesting them into the KG sqlite. A future
   extension could add a lightweight Repo node type (arxiv ID-free) so
   repos participate in queries. Worth threading into this series, or
   park it?
6. **mjlab-ready humanoid on 8 GB VRAM.** My §3.3 proposes 1024 envs for
   G1 velocity and 512 for Spinkick-G1. If these turn out to OOM, the
   user's "first mjlab training run" experience is a bad one. Do we
   want M3 to include an explicit "Start with 256 envs then scale up"
   first-run wizard for humanoids?

*End of design doc.*
