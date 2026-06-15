"""Subprocess runner for MjlabAdapter.

Invoked as `python -m sculptor.adapters._mjlab_runner train|rollout ...`.

Two sub-commands — `train` and `rollout` — each imports mjlab + rsl_rl
(heavy, CUDA-initialising) in-process. Keeping this in a separate module
means the UI / health-check / adapter-instantiation paths never pay the
import cost, per the lazy-import rule in MJLAB_PIVOT_DESIGN §7.

Reward injection: when `--reward-module-path` is passed, the runner
patches the loaded task cfg's `rewards` dict to contain a single active
term (`SculptorRewardTerm`) and zeroes the weight of every task-shipped
term. `cfg.scale_rewards_by_dt` is set to False so the reward module's
raw per-step return survives without dt-scaling. When `--reward-module-path`
is omitted, training uses the task's default reward terms unchanged —
useful for the GPU smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# ── Component capture sink (§7.1 / §7.2 — Eureka-style reward reflection) ──
# Populated by `_cmd_train` before invoking `runner.learn`; consumed by
# `SculptorRewardTerm.__call__`. Module-level so multiple term instances
# in the same subprocess share one sink (rsl_rl's env build may construct
# the term more than once). `None` disables capture — the `__call__` path
# stays allocation-free when no sink is active.
_COMPONENT_SINK: dict[str, list[float]] | None = None


def _record_components(
    sink: dict[str, list[float]] | None,
    components: dict[str, Any],
    info: dict[str, Any],
) -> None:
    """Append mean-per-step values to `sink` for each component + aux info
    signal. Safe to call with `sink=None` (returns immediately). Non-tensor
    values + non-finite means (empty tensor → NaN, ±Inf from a diverging
    reward term) are filtered so downstream `sum/len` aggregation stays
    finite. Aux keys from `info` get a `__` prefix so they can't collide
    with a user-named reward component."""
    if sink is None:
        return
    import math

    for name, tensor in components.items():
        try:
            mean = float(tensor.detach().mean().item())
        except Exception:  # noqa: BLE001 — any non-tensor / weird shape
            continue
        if not math.isfinite(mean):
            continue
        sink.setdefault(str(name), []).append(mean)
    for key in ("episode_length", "terminated", "time_outs"):
        t = info.get(key) if isinstance(info, dict) else None
        if t is None:
            continue
        try:
            val = float(t.detach().mean().item())
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(val):
            continue
        sink.setdefault(f"__{key}", []).append(val)


def _compute_playback_fps(
    step_dt: float,
    render_every: int,
    playback_speed: float,
    cli_fps: float = 0.0,
) -> float:
    """§Ship-7: derive video fps so rollout video duration ==
    sim_duration / playback_speed. Exposed as a free function for unit
    testing; the rollout body calls it identically with runtime args.

    - `step_dt`: physics timestep (seconds). Typically 0.02 (50 Hz).
    - `render_every`: frames captured every N physics steps.
    - `playback_speed`: 1.0 = real-time; 0.5 = slow-mo; 2.0 = 2× fast.
    - `cli_fps`: non-zero overrides the derived value.

    Clamps output to [1, 240] — ffmpeg rejects outside this range.
    """
    step_dt = max(float(step_dt), 1e-4)
    render_every = max(int(render_every), 1)
    playback_speed = max(0.1, min(float(playback_speed), 10.0))
    derived = playback_speed / (step_dt * render_every)
    if cli_fps and cli_fps > 0:
        derived = float(cli_fps)
    return max(1.0, min(derived, 240.0))


def _snapshots_to_trajectory(
    snapshots: list[dict[str, float]],
) -> dict[str, list[float]]:
    """Pivot a list of per-window `{name: mean}` snapshots into a dict of
    `{name: [mean_at_w0, mean_at_w1, ...]}` time-series, filling missing
    values with the previous window's value (or skipping the component if
    it first appeared mid-training). Preserves Eureka Appendix F format."""
    out: dict[str, list[float]] = {}
    if not snapshots:
        return out
    all_keys: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for k in snap.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    for k in all_keys:
        vals: list[float] = []
        last: float | None = None
        for snap in snapshots:
            if k in snap:
                last = float(snap[k])
            if last is not None:
                vals.append(last)
        if vals:
            out[k] = vals
    return out


def _load_reward_module(path: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sculpt_reward_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not spec reward module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a rsl_rl / mjlab config dataclass to a plain dict.

    rsl_rl ships `RslRlOnPolicyRunnerCfg` as a regular dataclass (not a
    pydantic model). `to_dict()` does not exist. Use `dataclasses.asdict`
    as the primary path; fall back to `vars()` for attrs-style classes.
    """
    import dataclasses

    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    try:
        return dict(vars(cfg))
    except TypeError:
        return dict(cfg)  # last resort; raises if not iterable


# Keys in the sculptor state schema for velocity-family tasks. Manipulation
# tasks (Yam) would extend with ee_pose / object_poses — deferred to M4.
_DEFAULT_SCHEMA_KEYS = (
    "qpos", "qvel", "base_lin_vel_b", "base_ang_vel_b",
    "projected_gravity_b", "actuator_force", "command_vel",
)


def _build_sculptor_term_class(schema_keys: tuple[str, ...]):
    """Factory for the reward-term class. Kept inside the function so
    heavy imports (mjlab, torch) only happen when the runner is actually
    invoked with --reward-module-path."""
    import torch

    # §Ship 46: canonical per-foot kick channels, single-sourced from the
    # adapter contract so the keys the runner EMITS can never drift from
    # the keys the contract ADVERTISES (and that edit.py grounds against).
    from sculptor.adapters.mjlab import _G1_INFO_EXTRA

    class SculptorRewardTerm:
        """mjlab reward term that dispatches to the sculpted module.

        Per MJLAB_PIVOT_DESIGN §1.3 — class-based term so we can hold
        `prev_state` across steps and zero it on per-env reset.
        """

        def __init__(self, cfg, env):  # type: ignore[no-untyped-def]
            path = cfg.params["reward_module_path"]
            self._schema_keys = schema_keys
            mod = _load_reward_module(path)
            if not hasattr(mod, "compute_reward_batched"):
                raise AttributeError(
                    f"reward module {path!r} missing compute_reward_batched; "
                    "required when training with MjlabAdapter (set "
                    "REWARD_SPEC['supports_batched']=True and define the "
                    "batched entry point)."
                )
            self._mod = mod
            self._prev = self._snapshot(env)

        @staticmethod
        def _find_articulated_entity(env):
            """mjlab scenes name the articulated entity after the robot
            (Go1 / G1 / ANYmal use 'robot'; Cartpole uses 'cartpole';
            manipulation tasks pick their own). Try 'robot' first for
            the common case, then fall back to scanning the scene for
            the first entity that isn't the terrain / ground plane.
            Raises with a clear message that names what IS in the scene
            if nothing matches — previous behavior was a bare KeyError.
            """
            try:
                return env.scene["robot"]
            except KeyError:
                pass
            # Scan scene keys. mjlab's Scene exposes iteration via keys().
            # Skip known ground-plane / static entities; pick the first
            # articulated one.
            _SKIP = {"terrain", "ground", "plane", "floor", "skybox", "light"}
            # mjlab.scene.Scene doesn't implement .keys(); the entity
            # dict lives at `.entities` (Scene.__getitem__ looks it up
            # internally). `.keys()` would raise AttributeError and the
            # previous except-clause would have masked it into an empty
            # list, then a confusing "Scene keys: []" KeyError downstream.
            keys: list[str] = []
            for attr in ("entities", "_entities"):
                ents = getattr(env.scene, attr, None)
                if isinstance(ents, dict):
                    keys = list(ents.keys())
                    break
            for k in keys:
                if k in _SKIP:
                    continue
                try:
                    ent = env.scene[k]
                except KeyError:
                    continue
                # Duck-type: an articulated entity exposes `.data` with
                # joint_pos / joint_vel. Terrain/ground doesn't.
                if hasattr(ent, "data") and hasattr(ent.data, "joint_pos"):
                    return ent
            raise KeyError(
                f"SculptorRewardTerm could not find an articulated robot "
                f"in env.scene. Tried 'robot' + scan-for-non-terrain. "
                f"Scene keys: {keys!r}. Either rename your robot asset "
                f"to 'robot' in the task config, or extend "
                f"SculptorRewardTerm._find_articulated_entity to know "
                f"about your task."
            )

        def _snapshot(self, env) -> dict[str, "torch.Tensor"]:
            robot = SculptorRewardTerm._find_articulated_entity(env)
            data = robot.data
            # Fixed-base articulations like Cartpole don't have a
            # floating root — `root_link_lin_vel_b` / `projected_gravity_b`
            # either raise or return zeros. Default the missing fields
            # to zero tensors so reward modules written against the
            # locomotion schema don't crash on non-locomotion tasks.
            N = env.num_envs
            dev = env.device

            def _zeros(dim: int) -> torch.Tensor:
                return torch.zeros(N, dim, device=dev)

            def _get(attr: str, fallback_dim: int):
                try:
                    v = getattr(data, attr)
                    return v if v is not None else _zeros(fallback_dim)
                except Exception:  # noqa: BLE001
                    return _zeros(fallback_dim)

            out: dict[str, torch.Tensor] = {}
            for k in self._schema_keys:
                if k == "qpos":
                    out[k] = _get("joint_pos", 1)
                elif k == "qvel":
                    out[k] = _get("joint_vel", 1)
                elif k == "base_lin_vel_b":
                    out[k] = _get("root_link_lin_vel_b", 3)
                elif k == "base_ang_vel_b":
                    out[k] = _get("root_link_ang_vel_b", 3)
                elif k == "projected_gravity_b":
                    out[k] = _get("projected_gravity_b", 3)
                elif k == "actuator_force":
                    out[k] = _get("actuator_force", 1)
                elif k == "command_vel":
                    try:
                        v = env.command_manager.get_command("base_velocity")
                    except Exception:  # noqa: BLE001
                        v = None
                    # `get_command` can return None silently on tasks that
                    # don't have a "base_velocity" command (Cartpole, Yam,
                    # any non-locomotion task). Don't let a None leak into
                    # `self._prev` — reset() would crash indexing into it.
                    out[k] = v if v is not None else _zeros(3)
                else:
                    # Unknown schema key — zero-fill rather than silently
                    # skipping, so `self._prev.keys()` matches the schema
                    # and reset() finds a tensor at every key.
                    out[k] = _zeros(1)
            return out

        def _resolve_foot_handles(self, env, robot):
            """§Ship 46: resolve (once) the contact + height sensors and
            the left/right foot site indices for the per-foot kick
            channels. Returns None unless BOTH a 'left_foot' and a
            'right_foot' site exist — that named-site pair is what fixes
            the per-foot column order shared across the contact `found`,
            height `heights`, and site-velocity tensors (mjlab wires all
            three from the same site list for the G1 biped). Quadrupeds /
            fixed-base tasks lack those exact names → None → foot channels
            stay zero."""
            try:
                names = tuple(robot.site_names)
            except Exception:  # noqa: BLE001
                return None
            idx = {n: i for i, n in enumerate(names)}
            li, ri = idx.get("left_foot"), idx.get("right_foot")
            if li is None or ri is None:
                return None

            def _scene_get(key):
                try:
                    return env.scene[key]
                except Exception:  # noqa: BLE001
                    return None

            return {
                "contact": _scene_get("feet_ground_contact"),
                "height": _scene_get("foot_height_scan"),
                "left_site": li,
                "right_site": ri,
            }

        def _foot_info(self, env, robot, dtype):
            """§Ship 46: per-foot kick channels + base horizontal speed,
            all (N,) tensors. Each signal is independently guarded so a
            missing sensor/site degrades to zeros rather than crashing —
            non-G1 tasks (which never advertise these keys) just carry
            harmless zeros the reward never references."""
            N = env.num_envs
            dev = env.device

            def _zero():
                return torch.zeros(N, device=dev, dtype=dtype)

            out = {k: _zero() for k in _G1_INFO_EXTRA}
            # Base horizontal speed (body-frame xy) — universally
            # meaningful; lets a kick reward penalise travel and lets the
            # diagnoser distinguish standing from walking.
            try:
                v = robot.data.root_link_lin_vel_b
                if v is not None:
                    out["base_horizontal_speed"] = torch.linalg.norm(
                        v[:, :2], dim=-1
                    ).to(dtype)
            except Exception:  # noqa: BLE001
                pass

            if not hasattr(self, "_foot_cache"):
                self._foot_cache = self._resolve_foot_handles(env, robot)
            fc = self._foot_cache
            if fc is None:
                return out

            try:
                found = fc["contact"].data.found  # (N, F)
                if found is not None and found.shape[-1] >= 2:
                    out["left_foot_contact"] = (found[:, 0] > 0).to(dtype)
                    out["right_foot_contact"] = (found[:, 1] > 0).to(dtype)
            except Exception:  # noqa: BLE001
                pass

            try:
                heights = fc["height"].data.heights  # (N, F)
                if heights is not None and heights.shape[-1] >= 2:
                    out["left_foot_height"] = heights[:, 0].to(dtype)
                    out["right_foot_height"] = heights[:, 1].to(dtype)
            except Exception:  # noqa: BLE001
                pass

            try:
                sv = robot.data.site_lin_vel_w  # (N, S, 3)
                li, ri = fc["left_site"], fc["right_site"]
                if sv is not None and sv.shape[1] > max(li, ri):
                    out["left_foot_swing_speed"] = torch.linalg.norm(
                        sv[:, li, :], dim=-1
                    ).to(dtype)
                    out["right_foot_swing_speed"] = torch.linalg.norm(
                        sv[:, ri, :], dim=-1
                    ).to(dtype)
            except Exception:  # noqa: BLE001
                pass

            return out

        def __call__(self, env, **_kwargs) -> "torch.Tensor":
            state = self._snapshot(env)
            action = env.action_manager.action
            robot = SculptorRewardTerm._find_articulated_entity(env)
            data = robot.data
            # Booster-Gym fall-detection signals. `base_height` is the
            # world-frame Z of the root link; `fallen` is True when the
            # body-frame projected gravity's Z component is ≥ 0, which
            # happens when the base is tipped past ~90° (on its side or
            # back — not a recoverable orientation for a quadruped).
            # Reward modules can zero-clip their output on `fallen==True`
            # to prevent the reward-hacking-by-toppling failure mode
            # diagnose surfaced on Sam's overnight run.
            # Fixed-base articulations (Cartpole) don't have these
            # attributes — fall back to zeros rather than crashing, so
            # the reward just ignores the fall-detection signals.
            try:
                base_z = data.root_link_pos_w[:, 2]
            except Exception:  # noqa: BLE001
                base_z = torch.zeros(env.num_envs, device=env.device)
            try:
                proj_g_z_b = data.projected_gravity_b[:, 2]
                fallen = (proj_g_z_b >= 0.0).to(dtype=action.dtype)
            except Exception:  # noqa: BLE001
                fallen = torch.zeros(env.num_envs, device=env.device, dtype=action.dtype)
            info = {
                "episode_length": env.episode_length_buf.float(),
                "terminated": env.termination_manager.terminated.float(),
                "time_outs": env.termination_manager.time_outs.float(),
                "step_dt": torch.full(
                    (env.num_envs,), float(env.step_dt), device=env.device
                ),
                "base_height": base_z,
                "fallen": fallen,
            }
            # §Ship 46: per-foot kick channels (contact / swing speed /
            # height) + base horizontal speed, so a sculpted reward can
            # shape a single-leg kick. Zero-filled on non-biped tasks.
            info.update(self._foot_info(env, robot, action.dtype))
            rewards, _components = self._mod.compute_reward_batched(
                self._prev, action, state, info
            )
            # §7.1 / §7.2: feed the module-level component sink when
            # training has enabled it. No-op when `_COMPONENT_SINK is None`,
            # which keeps rollout + non-sculpt runs allocation-free.
            if _COMPONENT_SINK is not None and isinstance(_components, dict):
                _record_components(_COMPONENT_SINK, _components, info)
            self._prev = {k: v.detach().clone() for k, v in state.items()}
            return rewards

        def reset(self, env_ids):  # type: ignore[no-untyped-def]
            for k in list(self._prev.keys()):
                self._prev[k][env_ids] = 0.0

    return SculptorRewardTerm


def _cmd_train(args: argparse.Namespace) -> None:
    # Lazy heavy imports — stay out of the module top.
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs

    # Reward injection (optional).
    if args.reward_module_path:
        from mjlab.managers.reward_manager import RewardTermCfg

        env_cfg.scale_rewards_by_dt = False
        # Attenuate the mjlab default reward terms to 0.3× instead of
        # zeroing them. The defaults (track_linear_velocity, upright,
        # pose, dof_pos_limits, action_rate_l2, foot_clearance, ...)
        # are carefully-tuned realism priors: stand like a quadruped,
        # respect joint limits, damp high-frequency action commands.
        # Zeroing them (pre-this-pass) meant the sculpted reward had
        # to simultaneously do "be a locomoting dog" AND "achieve the
        # task", with no anti-spasm / anti-topple floor — which is
        # exactly how Sam's overnight v2..v7 reward-hacked by
        # flipping onto the base (sculptor_primary rewarded upward
        # body motion irrespective of how the body got up there).
        # Keep them at 0.3× so the physics-plausible prior dominates
        # fine-grained control while sculptor_primary (weight=1.0 below)
        # dominates the task objective.
        existing = getattr(env_cfg, "rewards", None)
        REALISM_FLOOR_SCALE = 0.3
        if isinstance(existing, dict):
            for k in list(existing.keys()):
                term = existing[k]
                if term is not None and hasattr(term, "weight"):
                    term.weight = float(term.weight) * REALISM_FLOOR_SCALE
        else:
            env_cfg.rewards = {}

        schema_keys = tuple(args.schema_keys.split(",")) if args.schema_keys else _DEFAULT_SCHEMA_KEYS
        SculptorRewardTerm = _build_sculptor_term_class(schema_keys)
        env_cfg.rewards["sculptor_primary"] = RewardTermCfg(
            func=SculptorRewardTerm,
            weight=1.0,
            params={"reward_module_path": args.reward_module_path},
        )
        print(
            f"[runner] injected SculptorRewardTerm; {sum(1 for t in env_cfg.rewards.values() if t and getattr(t, 'weight', 0) == 0)} default terms zeroed",
            file=sys.stderr, flush=True,
        )

    env = ManagerBasedRlEnv(env_cfg, device=args.device)

    # Use mjlab's own runner shim (not rsl_rl's OnPolicyRunner directly —
    # the cfg structure differs). Mirrors mjlab/scripts/train.py:148-165.
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    rl_cfg = load_rl_cfg(args.task_id)
    rl_cfg.max_iterations = args.max_iterations

    agent_cfg_dict = _cfg_to_dict(rl_cfg)

    wrapped = RslRlVecEnvWrapper(
        env, clip_actions=getattr(rl_cfg, "clip_actions", None)
    )

    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(
        wrapped, agent_cfg_dict, str(output_dir / "logs"), args.device
    )

    # §Ship 15: optional warm-start from a pre-trained policy checkpoint.
    # Load ONLY actor+critic weights, skipping optimizer / iteration /
    # RND state. Optimizer skip is important — stale Adam momentum from
    # a previously-different reward degrades new-task learning. Iteration
    # skip keeps `max_iterations` semantics intact (we train
    # num_learning_iterations fresh iters regardless of what the
    # checkpoint thought). rsl_rl's PPO.load honors these keys per
    # rsl_rl/algorithms/ppo.py:444-466.
    if args.load_pretrained_policy:
        import hashlib as _hashlib
        ckpt = Path(args.load_pretrained_policy).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"--load-pretrained-policy not found: {ckpt}"
            )
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        # Cheap forensics: short checksum so Ship-16's mission orchestrator
        # can later verify the path matches its known stage checkpoint.
        source_sha8 = _hashlib.sha256(ckpt.read_bytes()).hexdigest()[:8]
        try:
            _ = runner.load(str(ckpt), load_cfg=load_cfg)
        except (RuntimeError, OSError, EOFError, Exception) as e:
            # Broaden beyond RuntimeError per Ship-15 audit — torch.load
            # can raise UnpicklingError (corrupt file), OSError (bad
            # I/O), EOFError (truncated file), and RuntimeError
            # (state_dict shape mismatch from an obs-space /
            # action-space drift between the source and target tasks).
            # We catch `Exception` as a safety net too — any other
            # error from the load path should still surface with a
            # pointer toward the likely cause.
            raise RuntimeError(
                f"failed to load pretrained policy {ckpt}: "
                f"{type(e).__name__}: {e}. "
                "Likely causes: (a) the checkpoint's task has a "
                "different observation or action space than the "
                "current task_id (warm-start requires matching "
                "obs_groups), (b) the checkpoint file is corrupt "
                "or truncated, or (c) the rsl_rl version that wrote "
                "the checkpoint has drifted from the one loading it."
            ) from e
        print(
            "[SCULPT-EVENT] " + json.dumps({
                "type": "warm_start_loaded",
                "source": str(ckpt),
                "source_sha8": source_sha8,
                "load_cfg_keys": sorted(
                    k for k, v in load_cfg.items() if v
                ),
            }),
            flush=True,
        )

    # Progress poller — watches the logs dir for new model_<N>.pt
    # checkpoints rsl_rl writes every `save_interval` iters and emits
    # `[SCULPT-EVENT] iter_progress` lines. Those flow up to the sculpt
    # CLI's stdout (via the mjlab tee thread) and land on the UI's
    # WebSocket event stream. Daemon thread + 2 s poll = negligible
    # CPU. Without this the UI shows a static "running" badge for the
    # entire ~25 min sculpt iter with no indication progress is real.
    import json as _json
    import re as _re
    import threading as _threading
    import time as _time

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _stop = _threading.Event()
    _ckpt_re = _re.compile(r"^model_(\d+)$")
    _t0 = _time.time()

    # §7.1: activate the module-level component sink only when a sculpted
    # reward is injected — non-injected runs (GPU smoke tests, task-default
    # rewards) have no `SculptorRewardTerm.__call__` feeding the sink, so
    # the file would be empty noise. Snapshots are grabbed by the poll
    # thread at each new `model_<N>.pt` (one per save_interval), giving us
    # the Eureka Appendix F "list of per-checkpoint means" shape.
    global _COMPONENT_SINK
    checkpoint_window_snapshots: list[dict[str, float]] = []
    if args.reward_module_path:
        _COMPONENT_SINK = {}
    else:
        _COMPONENT_SINK = None

    def _progress_poller() -> None:
        last_n = -1
        t0 = _t0
        # Heartbeat at t=0 so the UI gets an immediate "0 / max" tick
        # instead of waiting for rsl_rl's first checkpoint (~25-50 iters
        # into the run on default save_interval).
        print(
            "[SCULPT-EVENT] " + _json.dumps({
                "type": "iter_progress",
                "rl_iter": 0,
                "rl_total": int(args.max_iterations),
                "pct": 0.0,
                "elapsed_s": 0.0,
                "eta_s": None,
            }),
            flush=True,
        )
        while not _stop.is_set():
            try:
                best = -1
                for p in logs_dir.glob("model_*.pt"):
                    m = _ckpt_re.match(p.stem)
                    if m is not None:
                        best = max(best, int(m.group(1)))
                if best > last_n:
                    elapsed = _time.time() - t0
                    pct = (
                        100.0 * best / int(args.max_iterations)
                        if args.max_iterations else 0.0
                    )
                    eta = (
                        elapsed * (int(args.max_iterations) - best) / best
                        if best > 0 else None
                    )
                    print(
                        "[SCULPT-EVENT] " + _json.dumps({
                            "type": "iter_progress",
                            "rl_iter": best,
                            "rl_total": int(args.max_iterations),
                            "pct": round(pct, 1),
                            "elapsed_s": round(elapsed, 1),
                            "eta_s": round(eta, 1) if eta is not None else None,
                        }),
                        flush=True,
                    )
                    # §7.1: snapshot the sink at each checkpoint boundary.
                    # Clear after snapshot so each window holds only the
                    # steps between save_interval N and N+1 — the
                    # Eureka-format list is "window mean over time", not
                    # cumulative-since-start.
                    if _COMPONENT_SINK is not None and _COMPONENT_SINK:
                        snap = {
                            name: sum(vals) / len(vals)
                            for name, vals in _COMPONENT_SINK.items() if vals
                        }
                        if snap:
                            checkpoint_window_snapshots.append(snap)
                        for vals in _COMPONENT_SINK.values():
                            vals.clear()
                    last_n = best
            except Exception:  # noqa: BLE001
                pass
            _stop.wait(2.0)

    _poll_thread = _threading.Thread(target=_progress_poller, daemon=True)
    _poll_thread.start()
    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        _stop.set()
        _poll_thread.join(timeout=3.0)
        # §7.1: capture one final window for any samples accumulated AFTER
        # the last checkpoint (runs with max_iterations < save_interval
        # would otherwise write an empty reward_trajectory.json).
        if _COMPONENT_SINK is not None and _COMPONENT_SINK:
            tail_snap = {
                name: sum(vals) / len(vals)
                for name, vals in _COMPONENT_SINK.items() if vals
            }
            if tail_snap:
                checkpoint_window_snapshots.append(tail_snap)
        # Write reward_trajectory.json (Eureka Appendix F format, one value
        # per save_interval window). Missing file is fine for non-injected
        # runs — diagnose loads it optionally.
        if checkpoint_window_snapshots:
            reward_traj = _snapshots_to_trajectory(checkpoint_window_snapshots)
            try:
                (output_dir / "reward_trajectory.json").write_text(
                    json.dumps(reward_traj, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[runner] warning: could not write reward_trajectory.json: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )
        # Disable the sink so rollout / next-iter training start clean.
        _COMPONENT_SINK = None
        # Final tick at 100 % so the UI snaps to "done" immediately.
        print(
            "[SCULPT-EVENT] " + _json.dumps({
                "type": "iter_progress",
                "rl_iter": int(args.max_iterations),
                "rl_total": int(args.max_iterations),
                "pct": 100.0,
                "elapsed_s": round(_time.time() - _t0, 1),
                "eta_s": 0.0,
            }),
            flush=True,
        )

    # Capture the latest periodic checkpoint rsl_rl wrote under logs/.
    # Avoid runner.save() — it internally calls `wandb.save(path, ...)`
    # via rsl_rl.utils.wandb_utils, which raises even when
    # WANDB_MODE=disabled is set (wandb.save requires an active run).
    ckpt_path = output_dir / "checkpoint.pt"
    import os as _os
    import shutil as _shutil
    logs_dir = output_dir / "logs"
    candidates = sorted(
        logs_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if candidates:
        # Write atomically: copy to a tempfile in the same dir and
        # rename into place. Without this, a SIGKILL / OOM-kill during
        # the copy leaves a truncated checkpoint.pt that torch.load
        # explodes on, which would sabotage resume-from-iter.
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        _shutil.copy(candidates[-1], tmp_ckpt)
        _os.replace(tmp_ckpt, ckpt_path)
    else:
        # No rsl_rl checkpoint appeared — fall back to runner.save and
        # tolerate the wandb exception, leaving partial state on disk.
        try:
            runner.save(str(ckpt_path))
        except Exception as e:  # noqa: BLE001
            print(
                f"[runner] warning: both rsl_rl periodic ckpt and runner.save "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )

    metrics = {
        "task_id": args.task_id,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "device": args.device,
        "reward_injection": bool(args.reward_module_path),
        "checkpoint_path": str(ckpt_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    env.close()
    print(json.dumps({"status": "ok", "checkpoint": str(ckpt_path)}))


def _apply_ground_texture(env_cfg: Any) -> None:
    """§Ship 35: give the rendered floor an IMAGE texture instead of the
    default solid/checker terrain. PURELY COSMETIC and rollout-render only.

    FULLY DEFENSIVE by design — this runs on the training/rollout
    subprocess, so it must NEVER break a rollout: every failure path
    (missing asset, mjlab without `scene.spec_fn`, MjSpec API drift) leaves
    `env_cfg` untouched and the default ground in place. Toggle/override
    via `SCULPTOR_GROUND_TEXTURE` ('0'/'off'/'false' disables; a file path
    overrides the shipped texture). The MjSpec texture→material→geom API is
    verified against mujoco 3.7; on any version drift it silently no-ops.
    """
    import os

    setting = os.environ.get("SCULPTOR_GROUND_TEXTURE", "").strip()
    if setting.lower() in ("0", "off", "false", "no"):
        return
    try:
        from pathlib import Path as _Path

        if setting and _Path(setting).is_file():
            tex_path = setting
        else:
            from importlib.resources import files

            tex_path = str(files("sculptor.assets.textures") / "ground.png")
        if not _Path(tex_path).is_file():
            return
        scene = getattr(env_cfg, "scene", None)
        if scene is None or not hasattr(scene, "spec_fn"):
            return

        import mujoco

        prev_spec_fn = getattr(scene, "spec_fn", None)
        _GROUND_TOKENS = ("terrain", "floor", "ground")

        def _ground_spec_fn(spec: Any) -> None:
            # Chain any pre-existing spec_fn FIRST so we never clobber it.
            if callable(prev_spec_fn):
                prev_spec_fn(spec)
            try:
                tex = spec.add_texture()
                tex.name = "rs_ground_tex"
                tex.type = mujoco.mjtTexture.mjTEXTURE_2D
                tex.file = tex_path
                mat = spec.add_material()
                mat.name = "rs_ground_mat"
                mat.texrepeat = [12, 12]
                mat.reflectance = 0.15
                mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "rs_ground_tex"
                assigned = 0
                for g in spec.geoms:
                    name = (g.name or "").lower()
                    is_plane = getattr(g, "type", None) == mujoco.mjtGeom.mjGEOM_PLANE
                    if is_plane or any(tok in name for tok in _GROUND_TOKENS):
                        g.material = "rs_ground_mat"
                        assigned += 1
                if assigned == 0:
                    print("[runner] ground texture: no floor geom matched; "
                          "leaving default ground", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 — cosmetic; never break rollout
                print(f"[runner] ground texture skipped (spec edit failed): "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)

        scene.spec_fn = _ground_spec_fn
    except Exception as e:  # noqa: BLE001 — cosmetic; never break rollout
        print(f"[runner] ground texture setup skipped: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def _cmd_rollout(args: argparse.Namespace) -> None:
    """Run `n_episodes` rollouts and record a video for behavioral review.

    Designed for interactive iter-boundary feedback, not end-of-training
    evaluation, so the defaults favor wall-clock over sample count:

      * `num_envs = max(n_episodes, 64)` — mujoco_warp's kernel-launch
        overhead dominates at num_envs ≤ 8. The old rollout used
        `num_envs=1` which made every physics step ~500× slower than
        training (training batches 2048 envs per step). One real-world
        consequence: with num_envs=1 on WSL2 a 3000-step rollout ran
        for >60 min CPU-pegged; with num_envs=64 the same workload
        finishes in ~30 s.
      * `render_every = max(1, max_episode_steps // 120)` — WSL2's EGL
        software path does a scene-render in ~200 ms. Rendering every
        step was the OTHER ~500× wall-clock hit. Cap at ~120 frames,
        which is ~2 s of video at 60 fps — enough to eyeball behavior.
      * `[SCULPT-EVENT] rollout_progress` emitted every 25 steps so
        the UI shows progress instead of sitting on the env-setup
        print for the entire rollout duration.

    Episode stats are recorded for the first `n_episodes` envs only;
    extra envs (padding to MIN_ENVS_FOR_WARP) are stepped but ignored.
    When an env hits `done`, we freeze its cumulative return/length —
    the env auto-resets and keeps stepping, but its counters stop.
    """
    import time

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import numpy as np

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(exist_ok=True)

    n_episodes = int(args.n_episodes)
    # mujoco_warp amortizes kernel-launch overhead across envs; anything
    # under ~32 envs runs at <10 % of its peak. 64 is a comfortable
    # minimum that still fits alongside a trained policy in <1 GiB.
    MIN_ENVS_FOR_WARP = 64
    num_envs = max(n_episodes, MIN_ENVS_FOR_WARP)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = num_envs
    # §Ship 35: textured floor in the rendered rollout (cosmetic, guarded).
    _apply_ground_texture(env_cfg)
    env = ManagerBasedRlEnv(
        env_cfg, device=args.device, render_mode="rgb_array"
    )

    # §7.3: snapshot the mujoco model's actuator forceranges + joint
    # ranges. Downstream `sculptor.adapters.realism.audit_rollout` reads
    # these from disk (no XML parsing needed). Attribute names under the
    # mjlab-loaded mujoco model are the standard `mjModel` fields —
    # `actuator_forcerange`, `jnt_range`, `jnt_limited`, plus `name_*adr`
    # offsets into `names` for string lookups. Guard with try/except so
    # a missing attribute (different mjlab version) degrades to an empty
    # limits file, which the audit treats as verdict=unknown.
    limits_snapshot: dict[str, Any] = {
        "actuator_names": [],
        "actuator_forceranges": [],
        "joint_names": [],
        "joint_ranges": [],
    }
    # §Ship 26 (E1): entity-first joint names. The articulation exposes
    # `joint_names` in the SAME order as the persisted joint_pos /
    # joint_vel buffers (both come from the entity's data API) — the
    # mjModel route below includes the floating-base free joint and may
    # order differently, so it must not be the primary source. Spec
    # metrics use these names to select leg / hip / arm joint subsets.
    try:
        _robot = env.scene["robot"]
        _jn = list(getattr(_robot, "joint_names", []) or [])
        if _jn:
            limits_snapshot["joint_names"] = [str(n) for n in _jn]
    except Exception:  # noqa: BLE001 — best-effort, audit tolerates empty
        pass
    try:
        mj_model = getattr(env, "sim", None) or getattr(env, "_sim", None) \
            or getattr(env, "physics", None)
        # mjlab's ManagerBasedRlEnv exposes the underlying mujoco MjModel
        # via several attribute names across versions. Fall back to the
        # articulated entity we already locate for the reward-snapshot.
        m = None
        for attr in ("model", "_model", "mj_model", "_mj_model"):
            m = getattr(mj_model, attr, None) if mj_model is not None else None
            if m is not None:
                break
        if m is None:
            # Secondary route: the articulated entity in the scene usually
            # exposes its own model handle.
            try:
                robot = env.scene["robot"]
            except Exception:  # noqa: BLE001
                robot = None
            if robot is not None:
                for attr in ("model", "_model", "mj_model"):
                    m = getattr(robot, attr, None)
                    if m is not None:
                        break
        if m is not None:
            fr = np.asarray(getattr(m, "actuator_forcerange"), dtype=np.float64)
            jr = np.asarray(getattr(m, "jnt_range"), dtype=np.float64)
            # Names: use mujoco's id→name helpers when present, else
            # leave as positional indices (audit tolerates empty lists).
            def _names(model, count: int, kind: str) -> list[str]:
                import mujoco  # type: ignore[import-untyped]
                out: list[str] = []
                type_enum = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}", None)
                if type_enum is None:
                    return [f"{kind.lower()}_{i}" for i in range(count)]
                for i in range(count):
                    try:
                        nm = mujoco.mj_id2name(model, type_enum, i) or f"{kind.lower()}_{i}"
                    except Exception:  # noqa: BLE001
                        nm = f"{kind.lower()}_{i}"
                    out.append(str(nm))
                return out

            limits_snapshot["actuator_forceranges"] = fr.tolist()
            limits_snapshot["joint_ranges"] = jr.tolist()
            try:
                limits_snapshot["actuator_names"] = _names(m, fr.shape[0], "ACTUATOR")
                # Entity route above is authoritative for joint_names
                # (ordering matches the persisted buffers); only fall
                # back to mjModel names when it produced nothing.
                if not limits_snapshot["joint_names"]:
                    limits_snapshot["joint_names"] = _names(m, jr.shape[0], "JOINT")
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001 — realism audit is best-effort
        print(
            f"[runner] warning: could not snapshot mjcf limits: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )

    # Load policy via mjlab's own runner shim.
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    rl_cfg = load_rl_cfg(args.task_id)
    agent_cfg_dict = _cfg_to_dict(rl_cfg)
    wrapped = RslRlVecEnvWrapper(
        env, clip_actions=getattr(rl_cfg, "clip_actions", None)
    )
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, agent_cfg_dict, None, args.device)
    runner.load(args.checkpoint_path)
    policy = runner.get_inference_policy(device=args.device)

    import torch

    frames: list[np.ndarray] = []
    all_rewards: list[float] = []
    obs, _ = wrapped.reset()

    device = args.device
    ep_return = torch.zeros(num_envs, device=device)
    ep_length = torch.zeros(num_envs, dtype=torch.int, device=device)
    ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    max_steps = int(args.max_episode_steps)
    # §Ship-7 / rollout-video-realtime fix: the previous cap was
    # `render_every = max_steps // 120`, which always produced ~120
    # frames regardless of episode length. Combined with hardcoded
    # fps=50 playback, every video was exactly 2.4 s and a 500-step
    # rollout looked sped up ~4×. Sam's go1-jump-stress run (2026-04-23)
    # showed this as "crazy physics" — partly reward-driven, partly
    # just sped-up playback.
    #
    # Strategy: cap captured frames at MAX_FRAMES (memory + encoding
    # budget) then compute playback fps so video duration ==
    # sim_duration / playback_speed. For short rollouts (max_steps ≤
    # MAX_FRAMES) that means render_every=1 + fps=1/step_dt (real-time,
    # every step); for long rollouts it means render_every>1 + a
    # smaller fps so playback still matches real-time.
    MAX_FRAMES = 500  # bounded render memory ~200 MB at 640x480x3
    playback_speed = max(0.1, min(float(getattr(args, "playback_speed", 1.0) or 1.0), 10.0))
    cli_render_every = int(getattr(args, "render_every", 0) or 0)
    if cli_render_every > 0:
        render_every = cli_render_every
    else:
        render_every = max(1, (max_steps + MAX_FRAMES - 1) // MAX_FRAMES)

    # §7.1: per-step buffers for the expanded trajectory.npz. Each is
    # appended once per step across all `num_envs` (leading dim T added
    # at stack time). Memory footprint at T=500, N=64 is ~3.7 MB per
    # joint-wide field (T*N*num_dofs*4B) — safe on 8 GiB GPU / 16 GiB RAM.
    # Per-term reward decomposition comes from `env.reward_manager` (mjlab
    # default task terms, attenuated to 0.3× when a sculpted reward is
    # injected at train time; here in rollout we load the TRAINED policy
    # and roll it against the task's full default reward manager). The
    # per-term numbers let §7.3's realism audit identify which terms the
    # policy is exploiting.
    joint_pos_buf: list[np.ndarray] = []
    joint_vel_buf: list[np.ndarray] = []
    action_buf: list[np.ndarray] = []
    actuator_force_buf: list[np.ndarray] = []
    projected_gravity_b_buf: list[np.ndarray] = []
    root_link_pos_w_buf: list[np.ndarray] = []
    per_term_reward_buf: dict[str, list[np.ndarray]] = {}

    # Resolve the articulated robot entity once — mirrors the discovery
    # logic in `SculptorRewardTerm._find_articulated_entity` so fixed-base
    # tasks (Cartpole) and scenes with non-default root names still land
    # on the right handle. None-result means the expanded fields are
    # skipped (we still write `rewards` + `episode_id` as before).
    def _find_robot(e):
        try:
            return e.scene["robot"]
        except KeyError:
            pass
        _SKIP = {"terrain", "ground", "plane", "floor", "skybox", "light"}
        ents = getattr(e.scene, "entities", None) or getattr(e.scene, "_entities", None) or {}
        if isinstance(ents, dict):
            for k in ents.keys():
                if k in _SKIP:
                    continue
                try:
                    ent = e.scene[k]
                except KeyError:
                    continue
                if hasattr(ent, "data") and hasattr(ent.data, "joint_pos"):
                    return ent
        return None

    _robot = _find_robot(env)

    def _tensor_to_np(t) -> np.ndarray | None:
        if t is None:
            return None
        try:
            return t.detach().cpu().numpy().astype(np.float32, copy=False)
        except Exception:  # noqa: BLE001
            return None

    print("[SCULPT-EVENT] " + json.dumps({
        "type": "rollout_started",
        "n_episodes": n_episodes,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "render_every": render_every,
    }), flush=True)

    t0 = time.time()
    for step in range(max_steps):
        with torch.inference_mode():
            action = policy(obs)
        obs, rew, dones, _extras = wrapped.step(action)
        # Freeze cumulative return/length on the first `done` per env.
        active = (~ep_done).float()
        ep_return += rew * active
        ep_length += active.int()
        ep_done |= dones.bool()

        # Record env[0]'s step reward as a representative trajectory
        # (preserves the old single-env semantics for downstream
        # analysis in diagnose + reports).
        all_rewards.append(float(rew[0].item()))

        # §7.1: expanded-trajectory capture. Skipped frames (when the
        # entity lookup fails) still let `rewards` + per-term capture
        # proceed. Per-field try/except: different mjlab versions rename
        # data attributes (e.g. `root_link_lin_vel_b` vs `root_lin_vel_b`),
        # and fixed-base articulations legitimately lack `projected_gravity_b`.
        if _robot is not None:
            d = _robot.data
            jp = _tensor_to_np(getattr(d, "joint_pos", None))
            if jp is not None:
                joint_pos_buf.append(jp)
            jv = _tensor_to_np(getattr(d, "joint_vel", None))
            if jv is not None:
                joint_vel_buf.append(jv)
            af = _tensor_to_np(getattr(d, "actuator_force", None))
            if af is not None:
                actuator_force_buf.append(af)
            pg = _tensor_to_np(getattr(d, "projected_gravity_b", None))
            if pg is not None:
                projected_gravity_b_buf.append(pg)
            rp = _tensor_to_np(getattr(d, "root_link_pos_w", None))
            if rp is not None:
                root_link_pos_w_buf.append(rp)
        ap = _tensor_to_np(action)
        if ap is not None:
            action_buf.append(ap)
        try:
            # mjlab's RewardManager exposes per-term per-env values at
            # `_step_reward` (shape N × num_terms) updated inside the
            # env.step call that already completed. `_term_names` is the
            # ordered list of reward-term names. Both are private but
            # stable across the mjlab versions we support (unit tests
            # pin them via `test_realism.py` in Ship 3).
            rm = env.reward_manager
            sr = _tensor_to_np(rm._step_reward)
            term_names = list(rm._term_names)
            if sr is not None and sr.shape[-1] == len(term_names):
                for i, name in enumerate(term_names):
                    per_term_reward_buf.setdefault(str(name), []).append(
                        sr[:, i].astype(np.float32, copy=False)
                    )
        except Exception:  # noqa: BLE001 — reward_manager API drift; skip silently
            pass

        # Render only while env[0]'s first episode is ongoing. After
        # env[0] hits done, mujoco_warp's auto-reset warps it to a
        # fresh initial pose — continuing to record produces a glitchy
        # video where the Cartpole snaps from upright to hanging and
        # back every episode boundary (Issue F from Test 1 2026-04-22).
        # Sam's Cartpole video showed the pole "teleporting" and
        # checkered-floor flashes at the reset frames — this guard
        # stops the video at env[0]'s first terminal so the user sees
        # exactly one clean episode.
        if step % render_every == 0 and not bool(ep_done[0].item()):
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

        if step % 25 == 0:
            done_count = int(ep_done[:n_episodes].sum().item())
            elapsed = time.time() - t0
            pct = 100.0 * done_count / n_episodes if n_episodes else 0.0
            print("[SCULPT-EVENT] " + json.dumps({
                "type": "rollout_progress",
                "step": step,
                "max_steps": max_steps,
                "episodes_done": done_count,
                "n_episodes": n_episodes,
                "pct": round(pct, 1),
                "elapsed_s": round(elapsed, 1),
                "fps": round(step / elapsed, 1) if elapsed > 0 else None,
            }), flush=True)

        if bool(ep_done[:n_episodes].all().item()):
            break

    ep_returns = ep_return[:n_episodes].detach().cpu().tolist()
    ep_lengths = ep_length[:n_episodes].detach().int().cpu().tolist()

    print("[SCULPT-EVENT] " + json.dumps({
        "type": "rollout_done",
        "n_episodes": n_episodes,
        "total_steps": step + 1,
        "frames_recorded": len(frames),
        "elapsed_s": round(time.time() - t0, 1),
    }), flush=True)

    # Write video.
    import subprocess
    import shutil
    import tempfile

    video_path = output_dir / "rollout.mp4"
    if frames:
        # §Ship-7: video fps via _compute_playback_fps (unit-tested).
        try:
            step_dt = float(getattr(env, "step_dt", 0.02) or 0.02)
        except Exception:  # noqa: BLE001
            step_dt = 0.02
        cli_fps = float(getattr(args, "fps", 0.0) or 0.0)
        effective_fps = _compute_playback_fps(
            step_dt=step_dt,
            render_every=render_every,
            playback_speed=playback_speed,
            cli_fps=cli_fps,
        )
        print("[SCULPT-EVENT] " + json.dumps({
            "type": "video_params",
            "step_dt": step_dt,
            "render_every": render_every,
            "n_frames": len(frames),
            "playback_speed": playback_speed,
            "effective_fps": round(effective_fps, 3),
            "video_duration_s": round(len(frames) / effective_fps, 2),
        }), flush=True)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if ffmpeg:
            with tempfile.TemporaryDirectory() as td:
                from PIL import Image
                h, w = frames[0].shape[:2]
                for i, fr in enumerate(frames):
                    Image.fromarray(fr).save(Path(td) / f"f_{i:06d}.png")
                cmd = [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-framerate", f"{effective_fps:.3f}",
                    "-i", str(Path(td) / "f_%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", f"scale={w}:{h}", str(video_path),
                ]
                subprocess.run(cmd, check=False)

        # Keyframes.
        idxs = np.linspace(0, len(frames) - 1, num=min(12, len(frames))).astype(int)
        from PIL import Image
        for i, fi in enumerate(idxs):
            Image.fromarray(frames[fi]).save(keyframes_dir / f"frame_{i:02d}.png")

    # Trajectory.npz + behavior.json. §7.1 adds expanded state/action/
    # per-term-reward fields when the entity resolver + reward_manager API
    # found them; missing fields are simply omitted so consumers must
    # feature-detect. Per-term reward series land as `reward_term__<name>`
    # keys (double-underscore separator avoids collision with `reward_*`
    # user-facing names like `rewards` itself).
    def _stack_if_consistent(buf: list[np.ndarray]) -> np.ndarray | None:
        if not buf:
            return None
        try:
            shape0 = buf[0].shape
            if any(a.shape != shape0 for a in buf):
                return None
            return np.stack(buf, axis=0)
        except Exception:  # noqa: BLE001
            return None

    trajectory = {
        "rewards": np.asarray(all_rewards, dtype=np.float32),
        "episode_id": np.asarray(
            [e for e, L in enumerate(ep_lengths) for _ in range(L)], dtype=np.int32
        ),
    }
    for key, buf in (
        ("joint_pos", joint_pos_buf),
        ("joint_vel", joint_vel_buf),
        ("action", action_buf),
        ("actuator_force", actuator_force_buf),
        ("projected_gravity_b", projected_gravity_b_buf),
        ("root_link_pos_w", root_link_pos_w_buf),
    ):
        arr = _stack_if_consistent(buf)
        if arr is not None:
            trajectory[key] = arr
    for name, arrs in per_term_reward_buf.items():
        stacked = _stack_if_consistent(arrs)
        if stacked is not None:
            trajectory[f"reward_term__{name}"] = stacked
    np.savez_compressed(output_dir / "trajectory.npz", **trajectory)

    # §7.1 / §7.2: per-term time-series as JSON (Eureka Appendix F shape).
    # Complements the npz — diagnose / edit can load this one without
    # numpy as a dep. Values are per-step means across envs.
    rollout_reward_traj: dict[str, list[float]] = {}
    for name, arrs in per_term_reward_buf.items():
        rollout_reward_traj[name] = [float(np.mean(a)) for a in arrs]
    if rollout_reward_traj:
        try:
            (output_dir / "reward_trajectory.json").write_text(
                json.dumps(rollout_reward_traj, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[runner] warning: could not write rollout reward_trajectory.json: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )

    behavior = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "max_episode_length": int(max(ep_lengths)) if ep_lengths else 0,
        # §Ship 26 (E1/M1): capture settings are load-bearing for spec
        # metrics (frequency bands are in cycles/FRAME; episode-length
        # normalization needs the cap). Persisting them lets the eval
        # harness ASSERT capture parity across conditions instead of
        # silently comparing incomparables.
        "step_dt": float(getattr(env, "step_dt", 0.0) or 0.0),
        "max_episode_steps": int(max_steps),
        "rollout_num_envs": int(num_envs),
    }
    (output_dir / "behavior.json").write_text(json.dumps(behavior, indent=2))

    # §7.3: persist the mjcf-limits snapshot taken at env-init time so
    # `realism.audit_rollout` can compute torque-saturation / joint-limit-
    # violation metrics without re-parsing the XML. Missing arrays mean
    # the audit degrades to verdict=unknown (still safe, just uninformative).
    try:
        (output_dir / "mjcf_limits.json").write_text(
            json.dumps(limits_snapshot, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[runner] warning: could not write mjcf_limits.json: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )

    env.close()
    print(json.dumps({"status": "ok", "video": str(video_path)}))


def _cmd_vram_probe(args: argparse.Namespace) -> None:
    """Measure per-env VRAM at num_envs=64, emit coefficient to stdout as JSON."""
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA not available"}))
        sys.exit(0)

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    device = args.device
    baseline = torch.cuda.memory_allocated(device)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(env_cfg, device=device)
    env.reset()
    allocated = torch.cuda.memory_allocated(device)
    per_env_bytes = max(0, (allocated - baseline) / float(args.num_envs))
    coefficient_bytes = per_env_bytes * 1.2  # 20% safety buffer

    free, total = torch.cuda.mem_get_info(device)

    result = {
        "ok": True,
        "task_id": args.task_id,
        "probe_num_envs": args.num_envs,
        "baseline_bytes": int(baseline),
        "allocated_bytes": int(allocated),
        "per_env_bytes": float(per_env_bytes),
        "coefficient_bytes_per_env": float(coefficient_bytes),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }
    env.close()
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(prog="_mjlab_runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--task-id", required=True)
    p_train.add_argument("--reward-module-path", default=None,
                         help="optional path to a reward module with compute_reward_batched")
    p_train.add_argument("--num-envs", type=int, default=1024)
    p_train.add_argument("--max-iterations", type=int, default=100)
    p_train.add_argument("--seed", type=int, default=1)
    p_train.add_argument("--device", default="cuda:0")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--schema-keys", default="",
                         help="comma-separated override for the state-schema keys")
    p_train.add_argument(
        "--load-pretrained-policy", default=None,
        help=(
            "§Ship 15: path to a prior rsl_rl checkpoint (e.g., "
            "runs/iter_N/checkpoint.pt). When set, the runner loads "
            "actor+critic weights from this file BEFORE training begins, "
            "skipping optimizer / iteration / RND state. Used by the "
            "mission orchestrator to chain skills across stages."
        ),
    )

    p_roll = sub.add_parser("rollout")
    p_roll.add_argument("--task-id", required=True)
    p_roll.add_argument("--checkpoint-path", required=True)
    p_roll.add_argument("--output-dir", required=True)
    p_roll.add_argument("--n-episodes", type=int, default=3)
    p_roll.add_argument("--max-episode-steps", type=int, default=500)
    p_roll.add_argument("--device", default="cuda:0")
    # §Ship-7: fps=0 (default) means "compute from env.step_dt so video
    # plays in real time". A non-zero value is a hard override.
    p_roll.add_argument("--fps", type=float, default=0.0)
    # §Ship-7: playback speed multiplier. 1.0 = real-time; 2.0 = 2x
    # faster; 0.5 = half speed. Clamped to [0.1, 10.0] at runtime.
    p_roll.add_argument("--playback-speed", type=float, default=1.0)
    # §Ship-7: advanced override for frame decimation (1 = every step).
    # Default 0 means "pick automatically to cap at 500 captured frames".
    p_roll.add_argument("--render-every", type=int, default=0)

    p_probe = sub.add_parser("vram-probe")
    p_probe.add_argument("--task-id", required=True)
    p_probe.add_argument("--num-envs", type=int, default=64)
    p_probe.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    if args.mode == "train":
        _cmd_train(args)
    elif args.mode == "rollout":
        _cmd_rollout(args)
    elif args.mode == "vram-probe":
        _cmd_vram_probe(args)
    else:
        parser.error(f"unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
