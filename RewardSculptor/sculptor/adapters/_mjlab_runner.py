"""Subprocess runner for MjlabAdapter.

Invoked as `python -m sculptor.adapters._mjlab_runner train|rollout ...`.

Two sub-commands — `train` and `rollout` — each imports mjlab + rsl_rl
(heavy, CUDA-initialising) in-process. Keeping this in a separate module
means the UI / health-check / adapter-instantiation paths never pay the
import cost, per the lazy-import rule in MJLAB_PIVOT_DESIGN §7.

Reward injection: when `--reward-module-path` is passed, the runner
adds `SculptorRewardTerm` and attenuates task-shipped terms to a 0.3x
realism floor, except nominal command-tracking terms whose command was
replaced by an authored task goal. It also adds a task-independent survival
guard and explicit
non-timeout termination penalty. Without those terms, an early policy can
learn to fall immediately to avoid accumulating realism penalties, making a
less-negative return look like progress. `cfg.scale_rewards_by_dt` is set to
False so the reward module's raw per-step return survives without dt-scaling. When
`--reward-module-path` is omitted, training uses the task's default
reward terms unchanged (useful for the GPU smoke test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


# ── Component capture sink (§7.1 / §7.2 — Eureka-style reward reflection) ──
# Populated by `_cmd_train` before invoking `runner.learn`; consumed by
# `SculptorRewardTerm.__call__`. Module-level so multiple term instances
# in the same subprocess share one sink (rsl_rl's env build may construct
# the term more than once). `None` disables capture — the `__call__` path
# stays allocation-free when no sink is active.
_COMPONENT_SINK: dict[str, list[float]] | None = None

# §termination economics: custom rewards are deliberately composed with a
# subset of each task's native realism rewards. Those priors contain penalties
# (action rate, joint limits, collision costs, ...) that can make an untrained
# policy's per-step return negative. If failure ends the episode without a
# penalty, PPO can improve the episodic return simply by failing sooner. A
# constant survival term does not change the ordering of equal-length
# successful trajectories; it only makes premature non-timeout termination an
# economically dominated escape. The terminal penalty adds a clear final-step
# separation and is intentionally larger than a single survival step.
_SCULPTOR_SURVIVAL_WEIGHT = 1.0
_SCULPTOR_FAILURE_WEIGHT = -5.0
_SCULPTOR_TERMINAL_STILLNESS_WEIGHT = 1.0
_SCULPTOR_TERMINAL_CONTINUITY_SCALE = 2.0
_SCULPTOR_FORBIDDEN_CONTACT_WEIGHT = 4.0
_SCULPTOR_FORBIDDEN_CONTACT_WEIGHT_SCALE = 2.0
# A command-only obstacle-clearance stage deliberately targets a point outside
# the immutable task predicate. Predicate-centered generated shaping would pull
# in the opposite direction during that short phase, so withhold it entirely.
# Command tracking, direct contact, survival, and native realism terms are
# separate rewards and remain active.
_CLEARANCE_STAGE_PRIMARY_SCALE = 0.0


def _install_sculptor_termination_economics(
    rewards: dict[str, Any],
    reward_term_cfg: Any,
    mdp: Any,
) -> None:
    """Install robot/task-agnostic survival and failure reward terms.

    ``mdp.is_terminated`` excludes time-limit terminations, so completing a
    full episode is never punished. Fixed-base tasks with no failure
    termination simply receive the same constant horizon offset in every
    trajectory. Names are reserved under the ``sculptor_`` prefix so authored
    reward components cannot shadow the guard.
    """
    rewards["sculptor_survival"] = reward_term_cfg(
        func=mdp.is_alive,
        weight=_SCULPTOR_SURVIVAL_WEIGHT,
    )
    rewards["sculptor_failure"] = reward_term_cfg(
        func=mdp.is_terminated,
        weight=_SCULPTOR_FAILURE_WEIGHT,
    )


def _to_host_numpy(value: Any) -> Any:
    """Convert tensor-like simulator metadata to a host NumPy array.

    mjlab/mujoco-warp models may expose limits as CUDA tensors. Calling
    ``np.asarray`` on those tensors raises instead of copying implicitly.
    Detach and move only values that advertise those tensor operations; plain
    NumPy arrays and lists retain the normal conversion path.
    """
    import numpy as np

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    return np.asarray(candidate)


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


def _freeze_invalid_first_episode_steps(
    values: Any, valid_mask: Any,
) -> Any:
    """Replace post-reset samples with the last first-episode state.

    mjlab auto-resets an environment *inside* ``step`` before returning.  A
    state sampled on a done step therefore belongs to the next episode and
    creates a teleport in the recorded trajectory.  Objective metrics then
    misread the reset displacement as extreme terminal speed and erase real
    course progress.  Keep the rectangular ``(T, N, ...)`` contract while
    making its padding absorbing: after an environment's first invalid state,
    repeat its last valid state rather than stitching in a new attempt.

    ``valid_mask`` is persisted separately so mask-aware consumers can still
    distinguish measured samples from absorbing padding.  A malformed input
    is returned unchanged; rollout telemetry must never crash artifact write.
    """
    import numpy as np

    array = np.asarray(values)
    mask = np.asarray(valid_mask, dtype=bool)
    if array.ndim < 2 or mask.ndim != 2 or array.shape[:2] != mask.shape:
        return array
    frozen = array.copy()
    for step in range(1, array.shape[0]):
        invalid = ~mask[step]
        if np.any(invalid):
            frozen[step, invalid] = frozen[step - 1, invalid]
    return frozen


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
# Registered manipulation tasks extend this base contract through the generic
# capability-discovered recorder in manipulation_telemetry.py.
_DEFAULT_SCHEMA_KEYS = (
    "qpos", "qvel", "base_lin_vel_b", "base_ang_vel_b",
    "projected_gravity_b", "actuator_force", "command_vel",
)


def _episode_relative_base_height(base_z, episode_length, anchor):
    """Return per-env height displacement and the refreshed reset anchor.

    Reference datasets disagree on whether root Z is absolute, ground-relative,
    or normalized to start at zero.  The invariant that transfers across
    embodiments and authored terrain is vertical *change from this episode's
    reset*.  Capture the first observed height after every reset and never
    share it across environments.
    """
    import torch

    if (anchor is None or anchor.shape != base_z.shape
            or anchor.device != base_z.device or anchor.dtype != base_z.dtype):
        anchor = torch.full_like(base_z, float("nan"))
    fresh = (episode_length <= 1.0) | ~torch.isfinite(anchor)
    anchor = torch.where(fresh, base_z.detach(), anchor)
    return base_z - anchor, anchor


def _motion_quality_info(
    action, previous_action, episode_length, joint_vel,
):
    """Return universal smoothness scalars and the next action anchor.

    The reward contract advertises these channels for every mjlab
    articulation, so the runtime must define them without task, robot, or
    joint-name assumptions.  RMS reductions keep their scale comparable as
    embodiments gain joints.  Reset frames are zeroed so an episode boundary
    cannot manufacture a large action-rate penalty.
    """
    import torch

    if (
        previous_action is None
        or tuple(previous_action.shape) != tuple(action.shape)
        or previous_action.device != action.device
        or previous_action.dtype != action.dtype
    ):
        previous_action = torch.zeros_like(action)
    fresh = episode_length <= 1.0
    action_rate = torch.sqrt(
        torch.mean(torch.square(action - previous_action), dim=-1)
    )
    action_rate = torch.where(
        fresh, torch.zeros_like(action_rate), action_rate)

    if (
        joint_vel is None
        or getattr(joint_vel, "ndim", 0) != 2
        or joint_vel.shape[0] != action.shape[0]
    ):
        joint_vel_rms = torch.zeros_like(action_rate)
    else:
        joint_vel_rms = torch.sqrt(
            torch.mean(torch.square(joint_vel), dim=-1)
        ).to(dtype=action.dtype)
        joint_vel_rms = torch.where(
            fresh, torch.zeros_like(joint_vel_rms), joint_vel_rms)
    return {
        "action_rate": action_rate,
        "joint_vel_rms": joint_vel_rms,
    }, action.detach().clone()


def _build_sculptor_term_class(
    schema_keys: tuple[str, ...], robot_capability: Any | None = None,
    world_bundle: Any | None = None,
):
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
            self._robot_capability = robot_capability
            mod = _load_reward_module(path)
            if not hasattr(mod, "compute_reward_batched"):
                raise AttributeError(
                    f"reward module {path!r} missing compute_reward_batched; "
                    "required when training with MjlabAdapter (set "
                    "REWARD_SPEC['supports_batched']=True and define the "
                    "batched entry point)."
                )
            self._mod = mod
            self._world_reward_runtime = None
            if world_bundle is not None:
                from sculptor.world.runtime import TorchWorldRewardRuntime

                self._world_reward_runtime = TorchWorldRewardRuntime(
                    env, catalog=world_bundle.channel_catalog,
                    manifest=world_bundle.manifest)
            self._prev = self._snapshot(env)
            self._base_height_anchor = None
            self._previous_action = None

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
                    # Term names are task configuration, not adapter
                    # semantics. Discover the velocity command by its ranges
                    # so current ``twist`` and legacy ``base_velocity`` tasks
                    # both expose the command the policy actually observes.
                    v = None
                    manager = env.command_manager
                    names = list(dict.fromkeys([
                        "base_velocity", "twist",
                        *list(getattr(manager, "active_terms", ()) or ()),
                    ]))
                    for name in names:
                        try:
                            term_cfg = manager.get_term_cfg(name)
                            ranges = getattr(term_cfg, "ranges", None)
                            if not all(hasattr(ranges, field) for field in (
                                "lin_vel_x", "lin_vel_y", "ang_vel_z",
                            )):
                                continue
                            candidate = manager.get_command(name)
                            if (candidate is not None and candidate.ndim == 2
                                    and candidate.shape[0] == N
                                    and candidate.shape[1] >= 3):
                                v = candidate[:, :3]
                                break
                        except Exception:  # noqa: BLE001
                            continue
                    # `get_command` can return None silently on tasks that
                    # don't have a "base_velocity" command (Cartpole, Yam,
                    # any non-locomotion task). Don't let a None leak into
                    # `self._prev` — reset() would crash indexing into it.
                    out[k] = v if v is not None else _zeros(3)
                elif (self._robot_capability is not None
                      and k in self._robot_capability.reward_state_sources):
                    source = self._robot_capability.reward_state_sources[k]
                    namespace, _, role = source.partition(":")
                    try:
                        if namespace == "site":
                            concrete = self._robot_capability.resolve_site_role(
                                role)
                            indices = {
                                name: index for index, name in
                                enumerate(tuple(robot.site_names))}
                            selected = [indices[name] for name in concrete]
                            values = _get("site_pos_w", 3)[:, selected, :]
                        elif namespace == "body":
                            concrete = self._robot_capability.resolve_role(role)
                            indices = {
                                name: index for index, name in
                                enumerate(tuple(robot.body_names))}
                            selected = [indices[name] for name in concrete]
                            values = _get("body_link_pos_w", 3)[:, selected, :]
                        else:
                            raise KeyError(namespace)
                        out[k] = values.mean(dim=1)
                    except Exception:  # noqa: BLE001
                        shape = self._robot_capability.reward_state_schema.get(
                            k, (1,))
                        out[k] = _zeros(int(shape[-1]))
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
            episode_length = env.episode_length_buf.float()
            base_height_delta, self._base_height_anchor = (
                _episode_relative_base_height(
                    base_z, episode_length, self._base_height_anchor))
            info = {
                "episode_length": episode_length,
                "terminated": env.termination_manager.terminated.float(),
                "time_outs": env.termination_manager.time_outs.float(),
                "step_dt": torch.full(
                    (env.num_envs,), float(env.step_dt), device=env.device
                ),
                "base_height": base_z,
                "base_height_delta": base_height_delta,
                "fallen": fallen,
            }
            motion_info, self._previous_action = _motion_quality_info(
                action,
                self._previous_action,
                episode_length,
                getattr(data, "joint_vel", None),
            )
            info.update(motion_info)
            # §Ship 46: per-foot kick channels (contact / swing speed /
            # height) + base horizontal speed, so a sculpted reward can
            # shape a single-leg kick. Zero-filled on non-biped tasks.
            info.update(self._foot_info(env, robot, action.dtype))
            if self._world_reward_runtime is not None:
                # The runtime object exposes only base/shared_shaping catalog
                # entries; metric_only success truth has no reward-side API.
                info.update(self._world_reward_runtime.sample())
            rewards, _components = self._mod.compute_reward_batched(
                self._prev, action, state, info
            )
            rewards, _components = _apply_clearance_maneuver_reward_firewall(
                env, rewards, _components)
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
            if self._base_height_anchor is not None:
                self._base_height_anchor[env_ids] = float("nan")
            if self._previous_action is not None:
                self._previous_action[env_ids] = 0.0
            if self._world_reward_runtime is not None:
                self._world_reward_runtime.reset(env_ids)

    return SculptorRewardTerm


def _resolve_env_spec(args: argparse.Namespace) -> "dict | None":
    """Resolve the effective env spec for a runner invocation.

    `--env-spec <path>` (a validated per-project JSON, see
    `sculptor.env_spec`) wins; else `--env-profile <name>` names a
    built-in preset expressed in the SAME schema; else None (task
    defaults, byte-identical cfg). An invalid spec FILE fails the run
    loudly — the adapter validates before spawning, so reaching that
    branch means the file changed underneath us or a caller skipped
    validation; training under a half-applied env is never acceptable.
    An unknown profile NAME keeps the historical warn-and-ignore
    contract."""
    from sculptor.env_spec import jump_preset_spec, load_env_spec

    spec_path = getattr(args, "env_spec", "") or ""
    if spec_path:
        return load_env_spec(spec_path)   # raises on unreadable/invalid
    profile = getattr(args, "env_profile", "") or ""
    if not profile or profile == "default":
        return None
    if profile != "jump":
        print(f"[runner] env-profile {profile!r} unknown — ignored",
              file=sys.stderr, flush=True)
        return None
    return jump_preset_spec()


def reset_joints_to_reference(
    env: Any,
    env_ids: Any,
    joint_pos_target=None,  # noqa: ANN001 — torch.Tensor, mjlab-only at call time
    joint_pos_noise: float = 0.0,
    asset_cfg: Any = None,
    joint_pos_traj=None,  # noqa: ANN001 — [K, J] torch.Tensor for phase RSI
    joint_vel_traj=None,  # noqa: ANN001 — [K, J] torch.Tensor for phase RSI
) -> None:
    """Reset every selected joint to an EXPLICIT per-joint target (+ small
    symmetric noise), rather than mjlab's shipped
    `reset_joints_by_offset` (a single UNIFORM range added to the
    STANDING default across ALL joints — confirmed by reading
    `.venv/.../mjlab/envs/mdp/events.py` during recon; it has no
    per-joint target/keyframe parameter at all).

    §REFERENCE_TRAJECTORY_PLAN §8 part 2: a get-up clip's lying posture
    (e.g. bent knees/elbows) is materially different PER JOINT from the
    standing default, so it cannot be expressed as one shared offset.
    This event is the missing mechanism — a genuinely new mjlab event
    term, injected the same way `_apply_env_spec` already injects the
    `sunk` termination term (mjlab's `events`/`terminations` dicts are
    plain `dict[str, EventTermCfg]` / `dict[str, TerminationTermCfg]`
    the adapter is free to add entries to; no mjlab fork required).

    Mirrors `reset_joints_by_offset`'s own shape/clamp/write contract
    (same `soft_joint_pos_limits` clamp, same `write_joint_state_to_sim`
    call) so it composes with the rest of the reset pipeline identically
    — only the "what value do we reset around" question changes (an
    explicit target vector instead of the standing default + a random
    offset).

    §DeepMimic phase RSI (arXiv 1804.02717): when `joint_pos_traj` ([K, J], K
    downsampled reference frames) is given INSTEAD of a single `joint_pos_target`,
    each env samples a random frame k∈[0,K) at reset and initializes from THAT
    frame — so the batch covers the whole motion manifold, not one posture (the
    canonical RSI that "enables parallel learning of the motion phases"). When
    `joint_vel_traj` is also given, the joint VELOCITIES are initialized from the
    same frame too (a dynamic skill's mid-motion pose is meaningless at rest); the
    prior single-target path leaves velocity at the default (zeros).

    `joint_pos_target` / `joint_pos_traj` must already be tensors on `env.device`
    with one element per joint (J) selected by `asset_cfg` — the caller
    (`_apply_env_spec`) resolves/validates J against the robot's actual joint
    count before injecting this event; a mismatch is a clear `ValueError` there,
    never a silent misassignment here.
    """
    import torch

    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.utils.lab_api.math import sample_uniform

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    asset = env.scene[asset_cfg.name]
    default_joint_vel = asset.data.default_joint_vel
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    if joint_pos_traj is not None:
        # Phase RSI: per-env random reference frame.
        pos_traj = joint_pos_traj.to(device=env.device, dtype=torch.float32)
        k = int(pos_traj.shape[0])
        frame = torch.randint(0, max(1, k), (n,), device=env.device)
        joint_pos = pos_traj[frame].clone()                      # [n, J]
        if joint_vel_traj is not None:
            vel_traj = joint_vel_traj.to(device=env.device, dtype=torch.float32)
            joint_vel = vel_traj[frame].clone()                  # [n, J]
    else:
        target = joint_pos_target.to(device=env.device, dtype=torch.float32)
        joint_pos = target.unsqueeze(0).expand(n, -1).clone()
    if joint_pos_noise:
        joint_pos = joint_pos + sample_uniform(
            -float(joint_pos_noise), float(joint_pos_noise),
            joint_pos.shape, env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )


def _apply_world_selection(
    env_cfg: Any, selection_path: str, *, train: bool,
    task_id: str | None = None,
) -> Any | None:
    """Resolve and apply one immutable prompt-authored world tuple.

    Heavy simulator imports remain inside the compiler. An explicit authored
    selection is fail-closed: any hash, schema, capability, or materialized
    evaluation mismatch aborts before the environment or GPU runner exists.
    """
    if not selection_path:
        return None
    from sculptor.world.compiler import apply_world_selection

    bundle = apply_world_selection(
        env_cfg, Path(selection_path).resolve(), train=train,
        runtime_task_id=task_id)
    for adjustment in bundle.runtime_adjustments:
        print(
            f"[runner] authored-world runtime adjustment: {adjustment}",
            file=sys.stderr,
            flush=True,
        )
    return bundle


def _full_weight_authored_command_rewards(world_bundle: Any | None) -> frozenset[str]:
    """Return base reward terms that are part of an authored command contract.

    The generic 0.3 realism floor is appropriate for posture, smoothness, and
    safety priors, but it must not attenuate the simulator's dense supervision
    for a command that the authored World replaced with a task goal.  Doing so
    made lateral waypoint turns three times less important than their nominal
    locomotion objective even though the policy observation already carried the
    correct goal-conditioned command.

    Detect the compiled schema and the compiler's installed runtime adjustment,
    never a robot name or registered simulator task id.  The adjustment check is
    important: a declarative goal alone must not preserve command rewards when
    the selected base environment had no compatible command surface.
    """
    if world_bundle is None:
        return frozenset()
    manifest = getattr(world_bundle, "manifest", None)
    task_shared = getattr(manifest, "task_shared", {})
    goal = task_shared.get("goal", {}) if isinstance(task_shared, Mapping) else {}
    adjustments = tuple(getattr(world_bundle, "runtime_adjustments", ()) or ())
    installed = any(
        "goal-conditioned waypoint traversal" in str(adjustment)
        for adjustment in adjustments
    )
    if isinstance(goal, Mapping) and goal.get("type") == "waypoint_sequence" \
            and installed:
        return frozenset({"track_linear_velocity", "track_angular_velocity"})
    return frozenset()


def _authored_terminal_standing_enabled(world_bundle: Any | None) -> bool:
    """Whether the compiled command contract has a terminal dwell phase."""
    return _authored_terminal_hold_s(world_bundle) > 0.0


def _authored_terminal_hold_s(world_bundle: Any | None) -> float:
    """Return the positive dwell duration installed by an authored command."""
    if not _full_weight_authored_command_rewards(world_bundle):
        return 0.0
    manifest = getattr(world_bundle, "manifest", None)
    task_shared = getattr(manifest, "task_shared", {})
    goal = task_shared.get("goal", {}) if isinstance(task_shared, Mapping) else {}
    success = goal.get("success", {}) if isinstance(goal, Mapping) else {}
    event_sequence = (
        task_shared.get("event_sequence")
        if isinstance(task_shared, Mapping)
        else None
    )
    try:
        hold_s = float(success.get("hold_s", 0.0))
    except (TypeError, ValueError):
        hold_s = 0.0
    if isinstance(event_sequence, Mapping):
        try:
            phases = event_sequence["phases"]
            hold_s = max(
                hold_s,
                float(phases[2]["minimum_hold_s"]),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0
    return hold_s if hold_s > 0.0 else 0.0


#: Task-shipped reward terms that stay on when a reference is attached.
#: The test is "does this constrain what the hardware may do, or does it
#: prescribe what pose the robot holds?" -- limits, self-collision and
#: actuator smoothness are the former and are compatible with ANY reference;
#: `pose`, `upright`, the command-tracking terms and the gait shapers
#: (`foot_clearance`, `air_time`, `foot_slip`, `foot_swing_height`,
#: `angular_momentum`, `body_ang_vel`, `soft_landing`) are the latter and
#: fight the motion being tracked. Matched as substrings so a task that names
#: a term `robot_dof_pos_limits` is still recognised.
_HARDWARE_SAFETY_TERM_MARKERS: tuple[str, ...] = (
    "dof_pos_limits",
    "dof_vel_limits",
    "dof_torque_limits",
    "joint_limits",
    "torque_limits",
    "self_collision",
    "action_rate",
    "action_smoothness",
)


def _is_hardware_safety_term(name: str) -> bool:
    """Whether a task-shipped reward term constrains the hardware rather than
    prescribing a posture (see `_HARDWARE_SAFETY_TERM_MARKERS`)."""
    lowered = name.lower()
    return any(marker in lowered for marker in _HARDWARE_SAFETY_TERM_MARKERS)


def _reward_module_declares(reward_module_path: Any, key: str) -> bool:
    """Read one boolean flag out of a reward module's `REWARD_SPEC`.

    Fail-soft by design: a reward that cannot be imported here will fail
    loudly a few lines later when `SculptorRewardTerm` loads it for real, and
    guessing "tracking" for an unreadable module would silently change which
    task rewards are active."""
    if not reward_module_path:
        return False
    try:
        from sculptor.adapters.base import _import_reward_module

        mod = _import_reward_module(Path(reward_module_path))
        spec = getattr(mod, "REWARD_SPEC", None)
        return bool(isinstance(spec, dict) and spec.get(key))
    except Exception:  # noqa: BLE001 — see docstring
        return False


def _authored_terminal_stillness_weight(
    rewards: Mapping[str, Any],
    authored_command_terms: frozenset[str],
) -> float:
    """Balance terminal supervision against the installed command contract.

    Route-following terms can carry several times the nominal weight of the
    terminal stillness term.  Once the finite route completes, those terms
    become zero-command tracking objectives, but a broad tracking kernel can
    still pay a stepping equilibrium much more than the stricter authored
    dwell signal.  Make the phase-gated terminal objective at least as strong
    as the aggregate command supervision that delivered the robot there.

    The calculation uses only compiled command capabilities and live term
    weights.  It therefore adapts to future velocity-command embodiments
    without robot, simulator-task, or authored-prompt keying.
    """
    command_weight = 0.0
    for name in authored_command_terms:
        term = rewards.get(name)
        try:
            command_weight += abs(float(getattr(term, "weight", 0.0)))
        except (TypeError, ValueError):
            continue
    return max(_SCULPTOR_TERMINAL_STILLNESS_WEIGHT, command_weight)


def _authored_forbidden_contact_sensor_names(
    world_bundle: Any | None,
) -> tuple[str, ...]:
    """Return compiled sensors for every authored forbidden contact pair."""
    if world_bundle is None:
        return ()
    manifest = getattr(world_bundle, "manifest", None)
    task_shared = getattr(manifest, "task_shared", {})
    contacts = (
        task_shared.get("contacts", {})
        if isinstance(task_shared, Mapping)
        else {}
    )
    forbidden = (
        contacts.get("forbidden", ())
        if isinstance(contacts, Mapping)
        else ()
    )
    if not isinstance(forbidden, (list, tuple)):
        return ()
    return tuple(
        f"authored_contact__forbidden__{index}"
        for index, pair in enumerate(forbidden)
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )


def _authored_forbidden_contact_weight(
    rewards: Mapping[str, Any],
    authored_command_terms: frozenset[str],
) -> float:
    """Scale collision avoidance above the command income it must override."""
    command_weight = 0.0
    for name in authored_command_terms:
        term = rewards.get(name)
        try:
            command_weight += abs(float(getattr(term, "weight", 0.0)))
        except (TypeError, ValueError):
            continue
    return max(
        _SCULPTOR_FORBIDDEN_CONTACT_WEIGHT,
        _SCULPTOR_FORBIDDEN_CONTACT_WEIGHT_SCALE * command_weight,
    )


def _authored_forbidden_contact_penalty(
    env: Any, *, sensor_names: tuple[str, ...],
) -> Any:
    """Binary per-environment contact truth from compiled simulator sensors."""
    import torch

    found_any = torch.zeros(
        int(env.num_envs), device=env.device, dtype=torch.bool)
    for name in sensor_names:
        try:
            sensor = env.scene[name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"compiled forbidden-contact sensor is absent: {name}"
            ) from exc
        found = sensor.data.found
        if found.ndim > 1:
            found = torch.any(
                found > 0, dim=tuple(range(1, found.ndim)))
        found_any |= found.to(device=env.device, dtype=torch.bool)
    return found_any.to(dtype=torch.float32)


def _clearance_maneuver_primary_scale(env: Any) -> Any:
    """Return a per-environment firewall for command-only clearance maneuvers.

    The authored task predicate remains immutable, while a capable command
    term may temporarily target obstacle-safe approach and traversal points
    around that predicate. Generated rewards only observe predicate-centered
    channels and therefore cannot distinguish either intentional phase from
    failure to approach the raw goal. Detect the command capability itself and
    suppress only the conflicting generated reward until the immutable
    predicate advances to the next waypoint.

    This is intentionally based on typed runtime state, never an embodiment,
    simulator task id, or prompt-specific name.
    """
    import torch

    scale = torch.ones(
        int(env.num_envs), device=env.device, dtype=torch.float32)
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return scale

    for name in tuple(getattr(manager, "active_terms", ()) or ()):
        try:
            term = manager.get_term(name)
            shifts = getattr(term, "_clearance_shifts")
            waypoint_index = getattr(term, "_waypoint_index")
        except (AttributeError, KeyError, RuntimeError):
            continue

        if (
            not torch.is_tensor(shifts)
            or shifts.ndim < 2
            or shifts.shape[0] < 1
            or shifts.shape[1] < 2
        ):
            continue
        waypoint_index = torch.as_tensor(
            waypoint_index, device=env.device, dtype=torch.long)
        if tuple(waypoint_index.shape) != tuple(scale.shape):
            continue

        valid = (
            (waypoint_index >= 0)
            & (waypoint_index < int(shifts.shape[0]))
        )
        active_index = waypoint_index.clamp(
            min=0, max=int(shifts.shape[0]) - 1)
        active_shifts = shifts.to(
            device=env.device, dtype=torch.float32)[active_index, :2]
        adjusted = torch.linalg.norm(active_shifts, dim=-1) > 1e-6
        maneuver_active = valid & adjusted
        followthrough_pending = getattr(
            term, "_clearance_followthrough_pending", None)
        if torch.is_tensor(followthrough_pending):
            followthrough_pending = followthrough_pending.to(
                device=env.device, dtype=torch.bool)
            if tuple(followthrough_pending.shape) == tuple(scale.shape):
                maneuver_active |= followthrough_pending
        scale = torch.where(
            maneuver_active,
            torch.full_like(scale, _CLEARANCE_STAGE_PRIMARY_SCALE),
            scale,
        )
    return scale


def _apply_clearance_maneuver_reward_firewall(
    env: Any,
    rewards: Any,
    components: Any,
) -> tuple[Any, Any]:
    """Scale generated rewards to match active clearance-maneuver truth."""
    import torch

    scale = _clearance_maneuver_primary_scale(env)

    def _scale_per_env(value: Any) -> Any:
        if (
            not torch.is_tensor(value)
            or value.ndim < 1
            or int(value.shape[0]) != int(scale.shape[0])
        ):
            return value
        broadcast_shape = (int(scale.shape[0]),) + (1,) * (value.ndim - 1)
        return value * scale.to(
            device=value.device, dtype=value.dtype).reshape(broadcast_shape)

    scaled_rewards = _scale_per_env(rewards)
    if isinstance(components, dict):
        components = {
            name: _scale_per_env(value)
            for name, value in components.items()
        }
    return scaled_rewards, components


def _authored_terminal_stillness_state(
    env: Any, *, lin_std: float, ang_std: float, joint_std: float,
    upright_z_max: float = -0.7,
    joint_pos_tolerance: float = 0.6,
    upright_std: float = 0.35,
    joint_pos_std: float = 0.6,
) -> tuple[Any, Any, Any, Any]:
    """Return terminal phase, score, horizontal speed, and whole-body quiet.

    The command term owns phase truth and exposes is_standing_env once its
    finite route completes.  Query that capability generically across active
    commands: no embodiment, simulator task id, or authored prompt name is
    involved.  Before completion this term is identically zero and therefore
    cannot trade route progress for standing early.
    """
    import torch

    standing = torch.zeros(
        int(env.num_envs), device=env.device, dtype=torch.bool)
    manager = getattr(env, "command_manager", None)
    if manager is None:
        zeros = standing.float()
        return (
            standing,
            zeros,
            torch.full_like(zeros, float("inf")),
            standing,
        )
    for name in tuple(getattr(manager, "active_terms", ()) or ()):
        try:
            term = manager.get_term(name)
        except (KeyError, AttributeError):
            continue
        event_sequence_id = getattr(term, "event_sequence_id", "")
        event_phase = getattr(term, "event_phase", None)
        if event_sequence_id:
            # Event-authored tasks earn terminal supervision only in their
            # immutable HOLD phase.  Do not trust a generic standing flag for
            # this path: it may also be true during route-retention command
            # completion, which is intentionally distinct from phase truth.
            violation = getattr(term, "event_sequence_violation", None)
            if (
                not torch.is_tensor(event_phase)
                or tuple(event_phase.shape) != tuple(standing.shape)
                or not torch.is_tensor(violation)
                or tuple(violation.shape) != tuple(standing.shape)
            ):
                raise RuntimeError(
                    "event terminal stillness requires shape-validated "
                    "phase and sequence-violation truth"
                )
            standing |= (
                (event_phase.to(device=env.device, dtype=torch.long) == 2)
                & ~violation.to(device=env.device, dtype=torch.bool)
            )
            continue
        flag = getattr(term, "is_standing_env", None)
        if flag is not None and tuple(flag.shape) == tuple(standing.shape):
            standing |= flag.to(device=env.device, dtype=torch.bool)

    robot = None
    try:
        robot = env.scene["robot"]
    except (KeyError, TypeError):
        pass
    if robot is None:
        for attr in ("entities", "_entities"):
            entities = getattr(env.scene, attr, None)
            if not isinstance(entities, Mapping):
                continue
            robot = next(
                (
                    entity for entity in entities.values()
                    if hasattr(getattr(entity, "data", None), "joint_vel")
                ),
                None,
            )
            if robot is not None:
                break
    if robot is None:
        zeros = standing.float() * 0.0
        return (
            standing,
            zeros,
            torch.full_like(zeros, float("inf")),
            torch.zeros_like(standing),
        )

    data = robot.data
    lin_vel = getattr(data, "root_link_lin_vel_b", None)
    if lin_vel is None:
        lin_vel = getattr(data, "root_link_lin_vel_w", None)
    ang_vel = getattr(data, "root_link_ang_vel_b", None)
    if ang_vel is None:
        ang_vel = getattr(data, "root_link_ang_vel_w", None)
    joint_vel = getattr(data, "joint_vel", None)
    if lin_vel is None or ang_vel is None or joint_vel is None:
        zeros = standing.float() * 0.0
        return (
            standing,
            zeros,
            torch.full_like(zeros, float("inf")),
            torch.zeros_like(standing),
        )

    horizontal_speed = torch.linalg.vector_norm(lin_vel[:, :2], dim=-1)
    angular_speed = torch.linalg.vector_norm(ang_vel, dim=-1)
    joint_rms = torch.sqrt(torch.mean(torch.square(joint_vel), dim=-1))
    whole_body_quiet = (
        (horizontal_speed < float(lin_std))
        & (angular_speed < float(ang_std))
        & (joint_rms < float(joint_std))
    )
    score = (
        0.60 * torch.exp(-torch.square(horizontal_speed / float(lin_std)))
        + 0.25 * torch.exp(-torch.square(angular_speed / float(ang_std)))
        + 0.15 * torch.exp(-torch.square(joint_rms / float(joint_std)))
    )
    posture_factor = torch.ones_like(score)
    posture_signal_available = False

    # A motionless collapse is not an authored upright hold.  Projected
    # gravity and the articulation's own default joint pose are generic
    # posture references available on every floating-base locomotion robot;
    # no embodiment or task identifier is involved.  Gate the kinematic score
    # by the product of every available posture score.  This is a smooth
    # conjunction: the former geometric mean diluted a single failing posture
    # factor (for example a folded articulation under a moderately upright
    # torso), leaving enough terminal income to form a stable crouch optimum.
    # Missing signals remain fail-soft for fixed-base/custom adapters,
    # preserving their old behavior.
    projected_gravity = getattr(data, "projected_gravity_b", None)
    if (
        projected_gravity is not None
        and getattr(projected_gravity, "ndim", 0) == 2
        and projected_gravity.shape[0] == standing.shape[0]
        and projected_gravity.shape[1] >= 3
    ):
        gravity_z = projected_gravity[:, 2]
        upright_score = torch.exp(
            -torch.square((gravity_z + 1.0) / float(upright_std))
        )
        posture_factor *= upright_score
        posture_signal_available = True
        whole_body_quiet &= gravity_z < float(upright_z_max)

    joint_pos = getattr(data, "joint_pos", None)
    default_joint_pos = getattr(data, "default_joint_pos", None)
    if (
        joint_pos is not None
        and default_joint_pos is not None
        and tuple(joint_pos.shape) == tuple(default_joint_pos.shape)
        and getattr(joint_pos, "ndim", 0) == 2
    ):
        joint_pos_rms = torch.sqrt(
            torch.mean(torch.square(joint_pos - default_joint_pos), dim=-1)
        )
        pose_score = torch.exp(
            -torch.square(joint_pos_rms / float(joint_pos_std))
        )
        posture_factor *= pose_score
        posture_signal_available = True
        whole_body_quiet &= joint_pos_rms < float(joint_pos_tolerance)
    if posture_signal_available:
        score *= posture_factor
    return standing, score, horizontal_speed, whole_body_quiet


def _authored_terminal_stillness_reward(
    env: Any, *, lin_std: float, ang_std: float, joint_std: float,
    upright_z_max: float = -0.7,
    joint_pos_tolerance: float = 0.6,
    upright_std: float = 0.35,
    joint_pos_std: float = 0.6,
) -> Any:
    """Dense whole-body stillness, active only after an authored command ends."""
    standing, score, _horizontal_speed, _whole_body_quiet = (
        _authored_terminal_stillness_state(
            env,
            lin_std=lin_std,
            ang_std=ang_std,
            joint_std=joint_std,
            upright_z_max=upright_z_max,
            joint_pos_tolerance=joint_pos_tolerance,
            upright_std=upright_std,
            joint_pos_std=joint_pos_std,
        )
    )
    return score * standing.to(dtype=score.dtype)


def _build_authored_terminal_stillness_term_class():
    """Build stateful dwell supervision with an interruption-sensitive streak.

    A frame-wise stillness score cannot distinguish one uninterrupted dwell
    from many quiet samples separated by corrective steps.  The compiled
    authored goal supplies the required dwell duration.  This term accumulates
    a private per-environment quiet streak only in the command's terminal
    standing phase, rewards increasing consecutive progress, and applies the
    lost progress as an interruption penalty.  Its reset method follows the
    reward manager's selective per-environment reset contract.
    """
    import torch

    class AuthoredTerminalStillnessTerm:
        def __init__(self, cfg, env):  # type: ignore[no-untyped-def]
            # ManagerBase constructs class-backed terms with these exact
            # keyword names (`func(cfg=term_cfg, env=self._env)`).
            del cfg
            self._quiet_streak_s = torch.zeros(
                int(env.num_envs), device=env.device)

        def __call__(
            self,
            env,
            *,
            lin_std: float,
            ang_std: float,
            joint_std: float,
            hold_s: float,
            continuity_scale: float,
            upright_z_max: float = -0.7,
            joint_pos_tolerance: float = 0.6,
            upright_std: float = 0.35,
            joint_pos_std: float = 0.6,
        ):
            standing, score, _horizontal_speed, whole_body_quiet = (
                _authored_terminal_stillness_state(
                    env,
                    lin_std=lin_std,
                    ang_std=ang_std,
                    joint_std=joint_std,
                    upright_z_max=upright_z_max,
                    joint_pos_tolerance=joint_pos_tolerance,
                    upright_std=upright_std,
                    joint_pos_std=joint_pos_std,
                )
            )
            dtype = score.dtype
            if (
                tuple(self._quiet_streak_s.shape) != tuple(standing.shape)
                or self._quiet_streak_s.device != standing.device
                or self._quiet_streak_s.dtype != dtype
            ):
                self._quiet_streak_s = torch.zeros_like(
                    score, device=standing.device, dtype=dtype)

            step_dt = max(float(env.step_dt), 1e-6)
            duration = max(float(hold_s), step_dt)
            previous = self._quiet_streak_s
            # Base translation alone is not a sufficient definition of
            # stillness: a policy can step in place, rotate, or swing joints
            # while remaining under the horizontal-speed threshold.  Require
            # every velocity component already used by the dense whole-body
            # score so an uninterrupted dwell cannot hide a rhythmic sway.
            quiet = standing & whole_body_quiet
            streak = torch.where(
                quiet,
                torch.clamp(previous + step_dt, max=duration),
                torch.zeros_like(previous),
            )
            previous_progress = previous / duration
            progress = streak / duration
            # RewardManager applies scale_by_dt after evaluating this term.
            # Express the potential difference as a per-second rate so its
            # integrated gain/loss is invariant to the simulator timestep.
            # Without `/ step_dt`, a corrective step lost only dt times its
            # accumulated progress and was effectively free at 50 Hz.
            delta_rate = torch.where(
                standing,
                (progress - previous_progress) / step_dt,
                torch.zeros_like(progress),
            )
            continuity = torch.square(progress) + delta_rate
            self._quiet_streak_s = torch.where(
                standing, streak, torch.zeros_like(streak)).detach()
            dense = score * standing.to(dtype=dtype)
            return dense + float(continuity_scale) * continuity

        def reset(self, env_ids):  # type: ignore[no-untyped-def]
            self._quiet_streak_s[env_ids] = 0.0

    return AuthoredTerminalStillnessTerm


def _reward_visible_rollout_evidence(
    trajectory: Mapping[str, Any], catalog: Any, valid_mask: Any,
) -> dict[str, Any]:
    """Summarize batch-wide task evidence without crossing the metric firewall.

    Four rendered keyframes are necessarily ambiguous for courses, contacts,
    and manipulation.  The authored channel catalog already labels which
    arrays reward authoring may see.  Summarize only ``shared_shaping``
    progress and motion channels, over each environment's first episode, so
    the diagnoser can ground its visual interpretation without receiving a
    success predicate, held-out contact, objective score, or other metric-only
    signal.  The selection is semantic (catalog role/access), never keyed to a
    robot, task, goal, or channel name.
    """
    import numpy as np

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        return {}

    def rounded(value: float) -> float:
        return round(float(value), 5)

    summaries: dict[str, Any] = {}
    for spec in tuple(getattr(catalog, "channels", ()) or ()):
        if str(getattr(spec, "access", "")) != "shared_shaping":
            continue
        role = str(getattr(spec, "metric_role", ""))
        producer = str(getattr(spec, "producer", ""))
        if role != "progress" and producer != "entity_state":
            continue
        name = str(getattr(spec, "name", ""))
        # Position and quaternion state add prompt volume but little behavioral
        # evidence; for generic entity state, retain only motion channels.
        if producer == "entity_state" and not name.endswith(
                ("__lin_vel_w", "__ang_vel_w")):
            continue
        raw = trajectory.get(name)
        if raw is None:
            continue
        values = np.asarray(raw)
        if values.ndim < 2 or values.shape[:2] != mask.shape:
            continue
        magnitude = (
            np.linalg.norm(values, axis=-1)
            if values.ndim > 2 else values.astype(np.float64, copy=False)
        )
        if magnitude.shape != mask.shape:
            continue
        per_env: list[np.ndarray] = [
            magnitude[:, env_i][mask[:, env_i]]
            for env_i in range(mask.shape[1])
        ]
        per_env = [series[np.isfinite(series)] for series in per_env
                   if series.size]
        if not per_env or any(not series.size for series in per_env):
            continue
        start = np.asarray([series[0] for series in per_env])
        final = np.asarray([series[-1] for series in per_env])
        minimum = np.asarray([np.min(series) for series in per_env])
        maximum = np.asarray([np.max(series) for series in per_env])
        summaries[name] = {
            "role": role,
            "value": "vector_magnitude" if values.ndim > 2 else "scalar",
            "environments": len(per_env),
            "start_median": rounded(np.median(start)),
            "final_median": rounded(np.median(final)),
            "final_p10": rounded(np.quantile(final, 0.1)),
            "final_p90": rounded(np.quantile(final, 0.9)),
            "final_zero_fraction": rounded(np.mean(np.abs(final) <= 1e-6)),
            "min_over_time_median": rounded(np.median(minimum)),
            "max_over_time_median": rounded(np.median(maximum)),
            "max_over_time_p90": rounded(np.quantile(maximum, 0.9)),
        }
    if not summaries:
        return {}
    return {
        "policy": "shared_shaping channels only; metric_only excluded",
        "episode_scope": "first episode per environment",
        "channels": summaries,
    }


def _load_authored_robot_capability(selection_path: str) -> Any | None:
    if not selection_path:
        return None
    from sculptor.world.capabilities import resolve_robot_capability
    from sculptor.world.project import load_selected_world

    _store, _selection, bundle = load_selected_world(selection_path)
    robot = bundle["world"]["shared"]["robot"]
    return resolve_robot_capability(
        robot["capability_id"],
        required=robot.get("required_capabilities", []),
        extra_paths=([robot["descriptor_path"]]
                     if robot.get("descriptor_path") else []),
    )


def _primary_robot_entity(env_cfg: Any) -> str:
    """The scene's articulated-robot entity NAME — NOT hard-coded 'robot'. The
    locomotion tasks name it 'robot', but cartpole names it 'cartpole', object
    tasks add object entities, etc. A DR / reset event that targets a name the
    scene doesn't have raises KeyError at env startup and crashes training (the
    cartpole smoke-train regression). Prefer 'robot'; else the first non-terrain
    entity; fall back to 'robot' when the scene cfg isn't introspectable."""
    try:
        ents = getattr(getattr(env_cfg, "scene", None), "entities", None)
        names = list(ents.keys()) if hasattr(ents, "keys") else []
        if "robot" in names:
            return "robot"
        for n in names:
            if str(n).lower() not in ("terrain", "ground", "plane", "light"):
                return str(n)
    except Exception:  # noqa: BLE001
        pass
    return "robot"


def _is_pd_actuated(env_cfg: Any, rname: str) -> bool:
    """True iff EVERY actuator on the robot is a PD/position type mjlab's
    `dr.pd_gains` / `dr.effort_limits` support (BuiltinPosition, IdealPd, or an
    XmlActuator in `position` command mode). A motor/velocity/muscle actuator —
    e.g. the cartpole `<motor>` — makes those DR funcs raise TypeError at env
    STARTUP (invisible to cfg-building tests), aborting training. When we cannot
    confirm PD actuation we skip the gain/effort DR axes rather than risk that
    crash (the always-on mass/damping/armature axes are safe regardless)."""
    try:
        acts = env_cfg.scene.entities[rname].articulation.actuators
    except Exception:  # noqa: BLE001
        return False
    if not acts:
        return False
    for a in acts:
        tname = type(a).__name__
        if tname in ("BuiltinPositionActuatorCfg", "IdealPdActuatorCfg"):
            continue
        if tname == "XmlActuatorCfg" and getattr(a, "command_field", None) == "position":
            continue
        return False   # any non-PD actuator → skip the gain/effort DR axes
    return True


#: §always-on baseline physics domain randomization (arXiv 1710.06537 mass +
#: joint damping; RMA 2107.04034). ONLY crash-safe pure model-field SCALE axes —
#: every body has a mass and every dof a (possibly-zero) damping/armature, so
#: scaling them can never fault at env startup on any robot. Kept MODERATE
#: (BeyondMimic 2508.08241: over-wide DR dilutes the objective). Merged for any
#: TRAIN spec that omits them; richer axes (pd_gains, motor strength, CoM,
#: whole-body friction) are opt-in via the env spec / generator.
_DEFAULT_PHYSICS_DR: dict[str, tuple[float, float]] = {
    "body_mass_scale_range": (0.85, 1.15),
    "joint_damping_scale_range": (0.8, 1.2),
    "joint_armature_scale_range": (0.8, 1.2),
}


def _apply_env_spec(env_cfg: Any, spec: "dict | None", *,
                    train: bool = True, task_id: str = "") -> None:
    """§RL_SCULPTOR_AUDIT (env generalization, 2026-07-04): apply a
    validated env spec to the loaded task cfg, before the env is built.
    General successor to the retired jump-only `_apply_env_profile` —
    the spec's `shared` section applies to BOTH train and rollout (the
    policy is evaluated under its training task); its `train` section
    is TRAIN-ONLY curricula (RSI resets, sunk termination, domain
    randomization) so rollout evaluation — and the metric's view of the
    task — is never touched by them.

    The measured rationale for the jump preset's values (dead `fallen`
    signal at 70°, termination-as-escape, RSI/early-termination pairing,
    command-curriculum re-widening) lives in the audit doc's loop-4a /
    loop-6 entries; this function only maps schema semantics onto mjlab
    cfg fields. FULLY DEFENSIVE per-mutation: any cfg-shape drift skips
    that mutation with a warning, never breaks the run.

    `task_id` (the adapter's `--task-id`, e.g.
    "Mjlab-Velocity-Flat-Unitree-G1") is ONLY used to resolve the robot's
    canonical joint order for `train.reset_joint_pos_target`
    (§REFERENCE_TRAJECTORY_PLAN §8 part 2) — the env isn't built yet at
    this point, so the live `Entity.joint_names` isn't available; the
    static manifest (`sculptor.eval.robot_manifest`) is the ground truth
    used instead, same source the pre-run required-joint-roles gate
    already trusts. Omitted only when the spec carries no joint-target
    key (no behavior change for every existing caller)."""
    if not spec:
        # Rollout with no spec stays untouched (evaluation must be
        # un-randomized so the metric sees the TRUE task). A TRAIN call with no
        # spec still gets the always-on physics DR below — the "in any case"
        # guarantee that every training run/stage is domain-randomized.
        if not train:
            return
        spec = {}
    import math

    shared = spec.get("shared") or {}
    train_sec = dict(spec.get("train") or {}) if train else {}
    # §always-on physics DR (Dynamics Randomization arXiv 1710.06537; RMA
    # 2107.04034): fill CRASH-SAFE, MODERATE defaults for any physics axis the
    # spec/LLM omitted, so EVERY train run/stage is dynamics-randomized even with
    # no authored/generated env spec. Only pure model-field SCALE axes go here
    # (mass/damping/armature can never crash on any robot); actuator-shape-
    # dependent axes (pd_gains/effort/CoM/body-friction) stay OPT-IN via the spec
    # so a non-PD or unusual robot can't fault at env startup. Train-only:
    # train_sec is empty on rollout, keeping evaluation un-randomized.
    if train:
        for _k, _v in _DEFAULT_PHYSICS_DR.items():
            train_sec.setdefault(_k, _v)
    applied: list[str] = []

    def _skip(what: str, e: Exception) -> None:
        print(f"[runner] env-spec: {what} skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)

    if shared.get("zero_velocity_commands"):
        try:
            twist = (getattr(env_cfg, "commands", None) or {}).get("twist")
            if twist is not None:
                wrote_c = False
                ranges = getattr(twist, "ranges", None)
                for f in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                    if ranges is not None and hasattr(ranges, f):
                        setattr(ranges, f, (0.0, 0.0))
                        wrote_c = True
                # heading must be None (not (0,0)) — UniformVelocityCommand
                # rejects ANY truthy heading range when heading_command=False
                # (caught live: first loop-4 E2E launch, 2026-07-01).
                if ranges is not None and hasattr(ranges, "heading"):
                    ranges.heading = None
                    wrote_c = True
                for f, v in (("rel_standing_envs", 1.0),
                             ("rel_heading_envs", 0.0),
                             ("rel_forward_envs", 0.0),
                             ("heading_command", False)):
                    if hasattr(twist, f):
                        setattr(twist, f, v)
                        wrote_c = True
                if wrote_c:
                    applied.append("commands:twist→zero/standing")
        except Exception as e:  # noqa: BLE001 — never break a run
            _skip("command zeroing", e)
        # Coupled by construction: the command curriculum RE-WIDENS the
        # zeroed ranges mid-training, silently undoing the zeroing.
        try:
            cur = getattr(env_cfg, "curriculum", None)
            if isinstance(cur, dict) and cur.pop("command_vel", None) is not None:
                applied.append("curriculum:command_vel→removed")
        except Exception as e:  # noqa: BLE001
            _skip("curriculum trim", e)

    # Push events: the train section may override the shared setting
    # (e.g. robustness pushes during training only).
    push = train_sec.get("push_events", shared.get("push_events"))
    if isinstance(push, dict):
        try:
            events = getattr(env_cfg, "events", None)
            if isinstance(events, dict) and "push_robot" in events:
                if not push.get("enabled", True):
                    events.pop("push_robot")
                    applied.append("events:push_robot→removed")
                else:
                    term = events["push_robot"]
                    wrote_p = False
                    iv = push.get("interval_s")
                    if iv is not None and hasattr(term, "interval_range_s"):
                        term.interval_range_s = (float(iv[0]), float(iv[1]))
                        wrote_p = True
                    params = getattr(term, "params", None)
                    vr = (params or {}).get("velocity_range")
                    if isinstance(vr, dict):
                        lin = push.get("linear_mps")
                        if lin is not None:
                            for ax in ("x", "y", "z"):
                                if ax in vr:
                                    vr[ax] = (-float(lin), float(lin))
                                    wrote_p = True
                        ang = push.get("angular_radps")
                        if ang is not None:
                            for ax in ("roll", "pitch", "yaw"):
                                if ax in vr:
                                    vr[ax] = (-float(ang), float(ang))
                                    wrote_p = True
                    if wrote_p:
                        applied.append("events:push_robot→retuned")
        except Exception as e:  # noqa: BLE001
            _skip("push_robot", e)

    if shared.get("orientation_termination_deg") is not None:
        try:
            term = (getattr(env_cfg, "terminations", None) or {}).get("fell_over")
            params = getattr(term, "params", None)
            if isinstance(params, dict) and "limit_angle" in params:
                deg = float(shared["orientation_termination_deg"])
                params["limit_angle"] = math.radians(deg)
                applied.append(f"terminations:fell_over→{deg:g}deg")
        except Exception as e:  # noqa: BLE001
            _skip("fell_over relax", e)

    if shared.get("episode_length_s") is not None:
        try:
            if hasattr(env_cfg, "episode_length_s"):
                env_cfg.episode_length_s = float(shared["episode_length_s"])
                applied.append(f"episode_length_s→{env_cfg.episode_length_s:g}")
        except Exception as e:  # noqa: BLE001
            _skip("episode length", e)

    # ── train-only curricula (skipped entirely when train=False) ──────
    rsi_z = train_sec.get("reset_height_offset_m")
    rsi_vz = train_sec.get("reset_vertical_velocity_mps")
    rsi_vxy = train_sec.get("reset_horizontal_velocity_mps")
    rsi_pitch = train_sec.get("reset_pitch_offset_rad")
    rsi_roll = train_sec.get("reset_roll_offset_rad")
    if (rsi_z is not None or rsi_vz is not None or rsi_vxy is not None
            or rsi_pitch is not None or rsi_roll is not None):
        try:
            reset = (getattr(env_cfg, "events", None) or {}).get("reset_base")
            params = getattr(reset, "params", None)
            if isinstance(params, dict):
                # Height/orientation offsets need pose_range; the
                # velocity keys only write velocity_range — don't couple
                # them to it. Tag applied[] per ACTUAL write so a dead
                # sub-knob is visible to the disclosure below (a spec key
                # the cfg can't honor must never read as applied).
                wrote: list[str] = []
                pose_range = params.get("pose_range")
                if isinstance(pose_range, dict):
                    if rsi_z is not None:
                        pose_range["z"] = (float(rsi_z[0]), float(rsi_z[1]))
                        wrote.append("z")
                    # §REFERENCE_TRAJECTORY_PLAN §8 part 1: mjlab's
                    # reset_root_state_uniform natively reads
                    # pose_range["pitch"]/["roll"] (radians, offset from
                    # the entity's default orientation via quat_mul) —
                    # confirmed in .venv/.../mjlab/envs/mdp/events.py.
                    # A lying get-up start is exactly a large pitch (face
                    # up/down) or roll (on-the-side) offset.
                    if rsi_pitch is not None:
                        pose_range["pitch"] = (
                            float(rsi_pitch[0]), float(rsi_pitch[1]))
                        wrote.append("pitch")
                    if rsi_roll is not None:
                        pose_range["roll"] = (
                            float(rsi_roll[0]), float(rsi_roll[1]))
                        wrote.append("roll")
                if rsi_vz is not None or rsi_vxy is not None:
                    vr = params.get("velocity_range")
                    if not isinstance(vr, dict):
                        vr = {}
                        params["velocity_range"] = vr
                    if rsi_vz is not None:
                        vr["z"] = (float(rsi_vz[0]), float(rsi_vz[1]))
                        wrote.append("vz")
                    if rsi_vxy is not None:
                        vr["x"] = (float(rsi_vxy[0]), float(rsi_vxy[1]))
                        vr["y"] = (float(rsi_vxy[0]), float(rsi_vxy[1]))
                        wrote.append("vxy")
                if wrote:
                    applied.append(f"reset_base→RSI({','.join(wrote)})")
        except Exception as e:  # noqa: BLE001
            _skip("RSI reset", e)

    jp = train_sec.get("reset_joint_position_offset_rad")
    jv = train_sec.get("reset_joint_velocity_radps")
    if jp is not None or jv is not None:
        try:
            reset = (getattr(env_cfg, "events", None) or {}).get(
                "reset_robot_joints")
            params = getattr(reset, "params", None)
            if isinstance(params, dict):
                wrote_j: list[str] = []
                if jp is not None and "position_range" in params:
                    params["position_range"] = (float(jp[0]), float(jp[1]))
                    wrote_j.append("pos")
                if jv is not None and "velocity_range" in params:
                    params["velocity_range"] = (float(jv[0]), float(jv[1]))
                    wrote_j.append("vel")
                if wrote_j:
                    applied.append(
                        f"reset_robot_joints→randomized({','.join(wrote_j)})")
        except Exception as e:  # noqa: BLE001
            _skip("joint reset", e)

    # §REFERENCE_TRAJECTORY_PLAN §8 part 2: per-joint reference-posture
    # reset. mjlab's shipped `reset_joints_by_offset` has NO per-joint
    # target mechanism (a single uniform range from the STANDING
    # default — confirmed by reading events.py during recon), so this
    # injects a NEW event term (`reset_joints_to_reference`, defined
    # above in this module) exactly like the `sunk` termination below is
    # already injected into mjlab's plain `dict[str, ...]` managers.
    # Length is validated against the robot's CANONICAL joint order
    # (`sculptor.eval.robot_manifest`, resolved via `task_id`) here —
    # the env isn't built yet, so this is the only ground truth
    # available at cfg-mutation time. A mismatch is a CLEAR error (never
    # a silent misassignment): the caller already validated the spec
    # schema-wise, but "does this vector match THIS robot" is a
    # robot-specific check the schema layer cannot make.
    # §DeepMimic phase RSI (arXiv 1804.02717): a full downsampled reference
    # TRAJECTORY (reset_joint_pos_trajectory [K][J], + optional _vel_) takes
    # precedence over the single median posture — the reset event then samples a
    # random frame per env and initializes joint pos AND vel from it, so the
    # batch covers the whole motion manifold instead of one pose.
    jpt = train_sec.get("reset_joint_pos_target")
    jpt_traj = train_sec.get("reset_joint_pos_trajectory")
    jvt_traj = train_sec.get("reset_joint_vel_trajectory")
    jpt_noise = train_sec.get("reset_joint_pos_noise_rad")
    if jpt is not None or jpt_traj is not None:
        try:
            from sculptor.eval.robot_manifest import robot_joint_names

            # Joint width comes from the trajectory's frames when present, else
            # the single target.
            width = len(jpt_traj[0]) if jpt_traj is not None else len(jpt)
            canonical = robot_joint_names(task_id)
            if canonical is not None and width != len(canonical):
                raise ValueError(
                    f"reference reset has {width} joints but robot "
                    f"{task_id!r} has {len(canonical)} ({canonical[:3]}...) — "
                    f"refusing to apply a mismatched per-joint reset")
            reset = getattr(env_cfg, "events", None)
            if isinstance(reset, dict):
                import torch

                from mjlab.managers.event_manager import EventTermCfg
                from mjlab.managers.scene_entity_config import SceneEntityCfg

                params: dict[str, Any] = {
                    "joint_pos_noise": float(jpt_noise or 0.0),
                    "asset_cfg": SceneEntityCfg(
                        _primary_robot_entity(env_cfg), joint_names=(".*",)),
                }
                if jpt_traj is not None:
                    params["joint_pos_traj"] = torch.tensor(
                        [[float(x) for x in frame] for frame in jpt_traj],
                        dtype=torch.float32)
                    # Only carry the velocity trajectory when its shape matches
                    # (same K frames, same J) — the reset event indexes it with a
                    # frame sampled from the position traj, so a mismatch would
                    # crash at reset. A validated spec can't reach here mismatched;
                    # this is the defensive backstop for an unvalidated caller.
                    vel_ok = (jvt_traj is not None
                              and len(jvt_traj) == len(jpt_traj)
                              and len(jvt_traj[0]) == width)
                    if vel_ok:
                        params["joint_vel_traj"] = torch.tensor(
                            [[float(x) for x in frame] for frame in jvt_traj],
                            dtype=torch.float32)
                    elif jvt_traj is not None:
                        _skip("phase-RSI velocity trajectory", RuntimeError(
                            "shape mismatch with position trajectory — "
                            "using default joint velocities"))
                    label = (f"phase-RSI {len(jpt_traj)} frames×{width} joints"
                             + (", +vel" if vel_ok else ""))
                else:
                    params["joint_pos_target"] = torch.tensor(
                        [float(x) for x in jpt], dtype=torch.float32)
                    label = f"{width} joints"
                reset["reset_robot_joints_to_reference"] = EventTermCfg(
                    func=reset_joints_to_reference, mode="reset", params=params)
                applied.append(
                    f"events:+reset_robot_joints_to_reference({label})")
        except Exception as e:  # noqa: BLE001
            _skip("reference joint-posture reset", e)

    fr = train_sec.get("friction_range")
    if fr is not None:
        try:
            ev = (getattr(env_cfg, "events", None) or {}).get("foot_friction")
            params = getattr(ev, "params", None)
            if isinstance(params, dict) and "ranges" in params:
                params["ranges"] = (float(fr[0]), float(fr[1]))
                applied.append(f"events:foot_friction→({fr[0]:g},{fr[1]:g})")
        except Exception as e:  # noqa: BLE001
            _skip("friction randomization", e)

    # ── §sim2real physics domain randomization ──────────────────────────────
    # Mass, base CoM, PD gains, motor strength, joint damping/armature, and
    # whole-body friction as startup DR events on the robot (Dynamics
    # Randomization arXiv 1710.06537; RMA 2107.04034; Rapid Locomotion 2205.02824
    # + Walk-These-Ways 2212.03238). This runs on EVERY train/rollout — the
    # world-INDEPENDENT chokepoint both the mission-stage and single-run paths
    # funnel through — so physics DR is applied in every case, not only for
    # authored worlds. `startup` mode (each env samples once → the parallel batch
    # spans the distribution; the per-iteration seed re-rolls it) avoids the
    # expensive per-reset recompute mjlab warns about for mass/armature.
    physics_specs: list[tuple[str, Any, str]] = []
    try:
        from mjlab.envs.mdp import dr as _dr
        from mjlab.managers.event_manager import EventTermCfg as _ETC
        from mjlab.managers.scene_entity_config import SceneEntityCfg as _SEC

        _rname = _primary_robot_entity(env_cfg)
        bodies = _SEC(_rname, body_names=(".*",))
        joints = _SEC(_rname, joint_names=(".*",))
        acts = _SEC(_rname, actuator_names=(".*",))
        geoms = _SEC(_rname, geom_names=(".*",))

        def _rng(key: str) -> "tuple[float, float] | None":
            v = train_sec.get(key)
            return (float(v[0]), float(v[1])) if v is not None else None

        m = _rng("body_mass_scale_range")
        if m is not None:
            physics_specs.append(("env_dr__body_mass", _ETC(
                mode="startup", func=_dr.body_mass,
                params={"asset_cfg": bodies, "operation": "scale", "ranges": m}),
                f"body_mass×{m}"))
        com = train_sec.get("com_offset_m")
        if com is not None and float(com) > 0.0:
            c = float(com)
            physics_specs.append(("env_dr__com", _ETC(
                mode="startup", func=_dr.body_com_offset,
                params={"asset_cfg": bodies, "operation": "add",
                        "ranges": {0: (-c, c), 1: (-c, c), 2: (-c, c)}}),
                f"com±{c:g}m"))
        # pd_gains / effort_limits ONLY support PD/position actuators — a
        # motor/velocity actuator (cartpole) makes them raise at env startup, so
        # gate on the actual actuator type (the install try/except can't catch a
        # runtime startup fault). Real target robots (g1/go1/go2 IdealPd, yam
        # BuiltinPosition) are PD; the axes are simply dropped elsewhere.
        pd_ok = _is_pd_actuated(env_cfg, _rname)
        kp, kd = _rng("pd_kp_scale_range"), _rng("pd_kd_scale_range")
        if (kp is not None or kd is not None) and pd_ok:
            physics_specs.append(("env_dr__pd_gains", _ETC(
                mode="startup", func=_dr.pd_gains,
                params={"asset_cfg": acts, "operation": "scale",
                        "kp_range": kp or (1.0, 1.0), "kd_range": kd or (1.0, 1.0)}),
                f"pd_gains(kp×{kp},kd×{kd})"))
        elif (kp is not None or kd is not None) and not pd_ok:
            _skip("pd_gains DR", RuntimeError(
                f"robot {_rname!r} is not PD-actuated — skipping gain DR"))
        eff = _rng("motor_strength_scale_range")
        if eff is not None and pd_ok:
            physics_specs.append(("env_dr__motor_strength", _ETC(
                mode="startup", func=_dr.effort_limits,
                params={"asset_cfg": acts, "operation": "scale",
                        "effort_limit_range": eff}),
                f"motor_strength×{eff}"))
        elif eff is not None and not pd_ok:
            _skip("motor_strength DR", RuntimeError(
                f"robot {_rname!r} is not PD-actuated — skipping effort DR"))
        dmp = _rng("joint_damping_scale_range")
        if dmp is not None:
            physics_specs.append(("env_dr__joint_damping", _ETC(
                mode="startup", func=_dr.dof_damping,
                params={"asset_cfg": joints, "operation": "scale", "ranges": dmp}),
                f"joint_damping×{dmp}"))
        arm = _rng("joint_armature_scale_range")
        if arm is not None:
            physics_specs.append(("env_dr__joint_armature", _ETC(
                mode="startup", func=_dr.dof_armature,
                params={"asset_cfg": joints, "operation": "scale", "ranges": arm}),
                f"joint_armature×{arm}"))
        bfr = _rng("body_friction_range")
        if bfr is not None:
            physics_specs.append(("env_dr__body_friction", _ETC(
                mode="startup", func=_dr.geom_friction,
                params={"asset_cfg": geoms, "operation": "abs", "ranges": bfr}),
                f"body_friction={bfr}"))
    except Exception as e:  # noqa: BLE001 — mjlab DR API import/build failure
        _skip("physics DR setup", e)

    if physics_specs:
        try:
            events = getattr(env_cfg, "events", None)
            if isinstance(events, dict):
                for name, term, label in physics_specs:
                    events[name] = term
                    applied.append(f"events:+{name}({label})")
            else:
                _skip("physics DR install", RuntimeError("env_cfg has no events"))
        except Exception as e:  # noqa: BLE001
            _skip("physics DR install", e)

    # §get-up RSI fix (2026-07-09): a lying-start reset (large pitch/roll
    # offset from upright) trips the task's own fell-over/bad-orientation
    # termination on the reset itself — observed live: ALL envs terminate
    # at step 0 (Episode_Termination/fell_over = num_envs), so get-up
    # training never runs. `fell_over_termination: False` removes that
    # term for TRAIN only; the sunk-height termination + episode time_out
    # remain the episode enders. Term-name confirmed at the
    # `orientation_termination_deg` site above ("fell_over"); mjlab's
    # `terminations` cfg is a plain dict (same mechanism the `sunk`
    # injection below relies on), so removal is a plain `.pop()`.
    fell_over_off = train_sec.get("fell_over_termination") is False
    if fell_over_off:
        try:
            terms = getattr(env_cfg, "terminations", None)
            if isinstance(terms, dict) and terms.pop("fell_over", None) is not None:
                applied.append("terminations:fell_over→removed")
        except Exception as e:  # noqa: BLE001
            _skip("fell_over removal", e)

    sunk = train_sec.get("min_base_height_termination_m")
    if sunk is not None:
        # Early termination off the recoverable manifold — RSI's required
        # other half (DeepMimic pairing; measured tuck-jump iters 19-20:
        # RSI without it converges to the floor basin). TRAIN-ONLY by
        # schema position — evaluation keeps honest full episodes.
        try:
            terms = getattr(env_cfg, "terminations", None)
            if isinstance(terms, dict):
                from mjlab.envs.mdp.terminations import (
                    root_height_below_minimum,
                )
                from mjlab.managers.termination_manager import (
                    TerminationTermCfg,
                )
                terms["sunk"] = TerminationTermCfg(
                    func=root_height_below_minimum,
                    params={"minimum_height": float(sunk)},
                )
                applied.append(f"terminations:+sunk(base<{float(sunk):g}m)")
        except Exception as e:  # noqa: BLE001
            _skip("sunk termination", e)

    # Dead-knob disclosure: a spec key this task's cfg can't honor is a
    # silent no-op by the never-break-a-run contract — but the sculpt
    # loop's diagnoser iterates on these knobs, so it must be able to
    # SEE that one is dead (e.g. friction_range on a task without a
    # foot_friction event) instead of retuning it blindly forever.
    requested: list[str] = []
    if shared.get("zero_velocity_commands"):
        requested.append("commands:twist→zero/standing")
    if isinstance(push, dict):
        if not push.get("enabled", True):
            requested.append("events:push_robot→removed")
        elif any(push.get(k) is not None
                 for k in ("interval_s", "linear_mps", "angular_radps")):
            # enabled=true with no retune values means "keep pushes" —
            # honored by doing nothing, so it is never a dead knob.
            requested.append("events:push_robot→retuned")
    if shared.get("orientation_termination_deg") is not None:
        requested.append("terminations:fell_over")
    if shared.get("episode_length_s") is not None:
        requested.append("episode_length_s")
    if (rsi_z is not None or rsi_vz is not None or rsi_vxy is not None
            or rsi_pitch is not None or rsi_roll is not None):
        requested.append("reset_base→RSI")
    if jp is not None or jv is not None:
        requested.append("reset_robot_joints→randomized")
    if jpt is not None:
        requested.append("events:+reset_robot_joints_to_reference")
    if fr is not None:
        requested.append("events:foot_friction")
    if fell_over_off:
        requested.append("terminations:fell_over→removed")
    if sunk is not None:
        requested.append("terminations:+sunk")
    dead = [r for r in requested
            if not any(a.startswith(r.split("(")[0].split("→")[0])
                       for a in applied)]
    print(f"[runner] env-spec applied (train={train}): {applied}"
          + (f"; NOT APPLICABLE on this task cfg: {dead}" if dead else ""),
          file=sys.stderr, flush=True)


def _apply_eval_reset(env_cfg: Any, payload: "dict | None", *,
                       task_id: str = "") -> None:
    """§D17: apply a stage-FIXED eval-rollout reset override — a small
    ALLOWLISTED subset of `sculptor.reference.derive_eval_reset`'s
    payload (single deterministic values, not train-iterable ranges).

    Called from `_cmd_rollout` AFTER the existing shared-only
    `_apply_env_spec(..., train=False, ...)` call, so it only ever adds
    a fixed lying-start reset on top of the honest shared/eval task cfg
    — it never reads or is influenced by the diagnoser-iterable
    `env_spec.py` train section. Reuses the SAME cfg-mutation mechanisms
    `_apply_env_spec` uses for train (pose_range height/pitch/roll,
    the injected `reset_joints_to_reference` event, the `fell_over`
    termination pop) so eval genuinely resets the way the derivation
    promises — same code path, just fed a single midpoint value instead
    of a [lo, hi] range. `None`/empty payload is a pure no-op (today's
    standing-start behavior, byte-identical) — the common case for every
    non-get-up stage.

    Every mutation is defensive per-key (mirrors `_apply_env_spec`): a
    cfg-shape drift skips that key with a warning, never breaks the
    rollout. Announces what it actually wrote so the runner log makes a
    reference-derived lying-start eval visible, not silent."""
    if not payload:
        return

    def _skip(what: str, e: Exception) -> None:
        print(f"[runner] eval-reset: {what} skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)

    applied: list[str] = []

    z = payload.get("reset_height_offset_m")
    vz = payload.get("reset_vertical_velocity_mps")
    pitch = payload.get("reset_pitch_offset_rad")
    roll = payload.get("reset_roll_offset_rad")
    if z is not None or vz is not None or pitch is not None or roll is not None:
        try:
            reset = (getattr(env_cfg, "events", None) or {}).get("reset_base")
            params = getattr(reset, "params", None)
            if isinstance(params, dict):
                wrote: list[str] = []
                pose_range = params.get("pose_range")
                if isinstance(pose_range, dict):
                    if z is not None:
                        pose_range["z"] = (float(z), float(z))
                        wrote.append("z")
                    if pitch is not None:
                        pose_range["pitch"] = (float(pitch), float(pitch))
                        wrote.append("pitch")
                    if roll is not None:
                        pose_range["roll"] = (float(roll), float(roll))
                        wrote.append("roll")
                if vz is not None:
                    vr = params.get("velocity_range")
                    if not isinstance(vr, dict):
                        vr = {}
                        params["velocity_range"] = vr
                    vr["z"] = (float(vz), float(vz))
                    wrote.append("vz")
                if wrote:
                    applied.append(f"reset_base→eval_reset({','.join(wrote)})")
        except Exception as e:  # noqa: BLE001
            _skip("eval reset pose/velocity", e)

    jpt = payload.get("reset_joint_pos_target")
    jpt_noise = payload.get("reset_joint_pos_noise_rad")
    if jpt is not None:
        try:
            from sculptor.eval.robot_manifest import robot_joint_names

            canonical = robot_joint_names(task_id)
            if canonical is not None and len(jpt) != len(canonical):
                raise ValueError(
                    f"eval-reset reset_joint_pos_target has {len(jpt)} "
                    f"elements but robot {task_id!r} has "
                    f"{len(canonical)} joints ({canonical[:3]}...) — "
                    f"refusing to apply a mismatched per-joint reset")
            events = getattr(env_cfg, "events", None)
            if isinstance(events, dict):
                import torch

                from mjlab.managers.event_manager import EventTermCfg
                from mjlab.managers.scene_entity_config import SceneEntityCfg

                target_t = torch.tensor(
                    [float(x) for x in jpt], dtype=torch.float32)
                events["reset_robot_joints_to_reference"] = EventTermCfg(
                    func=reset_joints_to_reference,
                    mode="reset",
                    params={
                        "joint_pos_target": target_t,
                        "joint_pos_noise": float(jpt_noise or 0.0),
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=(".*",)),
                    },
                )
                applied.append(
                    f"events:+reset_robot_joints_to_reference"
                    f"({len(jpt)} joints)")
        except Exception as e:  # noqa: BLE001
            _skip("eval reference joint-posture reset", e)

    if payload.get("fell_over_termination") is False:
        try:
            terms = getattr(env_cfg, "terminations", None)
            if isinstance(terms, dict) and terms.pop("fell_over", None) is not None:
                applied.append("terminations:fell_over→removed")
        except Exception as e:  # noqa: BLE001
            _skip("eval fell_over removal", e)

    print(f"[runner] eval reset: reference-derived lying start "
          f"(stage-fixed): {applied}", file=sys.stderr, flush=True)


def _apply_rl_spec(rl_cfg: Any, spec: "dict | None") -> None:
    """PPO exploration adjustment from the env spec's train section —
    `entropy_coef_scale` multiplies the task's default entropy bonus
    (explosive single-burst skills need higher early exploration than
    walking defaults; see the audit doc §1). Train-time only (the
    caller); defensive on cfg-shape drift."""
    if not spec:
        return
    scale = (spec.get("train") or {}).get("entropy_coef_scale")
    if scale is None:
        return
    try:
        algo = getattr(rl_cfg, "algorithm", None)
        cur = getattr(algo, "entropy_coef", None)
        if isinstance(cur, (int, float)) and cur > 0:
            algo.entropy_coef = float(cur) * float(scale)
            print(f"[runner] rl-spec: entropy_coef {cur} → "
                  f"{algo.entropy_coef}", file=sys.stderr, flush=True)
        else:
            # entropy_coef_scale IS diagnoser-iterable — a dead knob
            # must be disclosed like the env-side ones, or the loop
            # retunes it blindly forever.
            print("[runner] rl-spec: entropy_coef_scale NOT APPLICABLE "
                  "(task cfg exposes no positive algorithm.entropy_coef)",
                  file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001 — never break a run
        print(f"[runner] rl-spec skipped: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def _apply_env_profile(env_cfg: Any, profile: str, *, train: bool = True) -> None:
    """Named-preset entry point: resolve `profile` to its env-spec
    instance and route through the general applier. Kept as the seam
    the profile tests pin — the jump preset MUST stay byte-equivalent
    to the retired hardcoded implementation."""
    if not profile or profile == "default":
        return
    if profile != "jump":
        print(f"[runner] env-profile {profile!r} unknown — ignored",
              file=sys.stderr, flush=True)
        return
    from sculptor.env_spec import jump_preset_spec

    _apply_env_spec(env_cfg, jump_preset_spec(), train=train)


def _apply_rl_profile(rl_cfg: Any, profile: str) -> None:
    """Named-preset entry point for the PPO side — see _apply_env_profile."""
    if profile != "jump":
        return
    from sculptor.env_spec import jump_preset_spec

    _apply_rl_spec(rl_cfg, jump_preset_spec())


def _configured_init_std(rl_cfg: Any) -> Optional[float]:
    """The exploration std a *fresh* policy would start from, or None.

    Reads `actor.distribution_cfg["init_std"]` — the value rsl_rl's
    `GaussianDistribution` uses to initialize `std_param`. None when the
    task uses a deterministic or non-Gaussian actor, which is the signal
    to leave a warm-started policy's noise alone.
    """
    dist = getattr(getattr(rl_cfg, "actor", None), "distribution_cfg", None)
    if not isinstance(dist, dict):
        return None
    try:
        init_std = float(dist.get("init_std"))
    except (TypeError, ValueError):
        return None
    return init_std if init_std > 0.0 else None


def _clamp_warm_started_noise(runner: Any, ceiling: float) -> Optional[dict]:
    """Cap a warm-started policy's exploration noise at `ceiling`.

    A warm start loads actor weights from a checkpoint, and for a Gaussian
    policy that includes the learned action-noise std. Across chained sculpt
    iterations that std ratchets *up*: measured on platform-ascent-showcase
    it went 1.05 → 1.39 → 1.71 over three iterations against an `init_std`
    of 1.0, and since mjlab's `action_rate_l2` penalty grows as 2σ² the
    inherited noise alone came to cost more per step than the entire task
    reward paid — every episode ended in `fell_over` and the run read as
    "the reward does nothing".

    Exploration scale is a property of the training run about to start, not
    knowledge carried by the checkpoint — the same reasoning that already
    makes the warm-start path skip the source optimizer's Adam momentum.
    So the bound is one-directional: a policy that converged to *less* noise
    than a fresh one keeps that precision, while anything above the fresh-init
    value is drift and gets clamped back. Returns the before/after summary
    when it changed anything, else None.
    """
    import torch

    distribution = getattr(getattr(runner, "alg", None), "actor", None)
    distribution = getattr(distribution, "distribution", None)
    scalar = getattr(distribution, "std_param", None)
    logged = getattr(distribution, "log_std_param", None)
    param = scalar if scalar is not None else logged
    if param is None:
        return None

    with torch.no_grad():
        std = param.detach() if scalar is not None else param.detach().exp()
        before = float(std.mean())
        if before <= ceiling:
            return None
        if scalar is not None:
            param.detach().clamp_(max=ceiling)
        else:
            param.detach().clamp_(max=float(torch.log(torch.tensor(ceiling))))
        after = scalar if scalar is not None else logged.detach().exp()
        return {"std_before": before, "std_after": float(after.detach().mean()),
                "ceiling": ceiling}


def _install_learning_vitals(runner: Any, total_iters: int) -> bool:
    """Emit per-iteration `learning_vitals` events by wrapping the logger.

    rsl_rl prints everything a person needs to judge a run — mean return,
    episode length, exploration std, and the per-component reward breakdown —
    but only as console text, so the UI sees an hour of unstructured log lines
    behind a percentage bar. Every failure diagnosed on this project so far was
    a number sitting in that text: an inherited action std that made the
    action-rate penalty outgrow the task reward, then the same penalty growing
    back under a tripled entropy bonus. Both are obvious the moment the
    strongest positive and negative components are put side by side.

    Wraps `runner.logger.log` rather than reimplementing it, so the numbers
    reported are exactly the numbers rsl_rl computed. Returns False when the
    runner has no logger to wrap — telemetry must never be a reason a run
    fails to start.
    """
    import statistics

    logger = getattr(runner, "logger", None)
    original = getattr(logger, "log", None)
    if not callable(original):
        return False

    def _mean(buf: Any) -> Optional[float]:
        try:
            return float(statistics.mean(buf)) if len(buf) else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _components() -> dict[str, float]:
        """Mean of each `Episode_Reward/<name>` term over the window.

        Averaged across `ep_extras` the same way rsl_rl averages them for its
        own console block, so the two never disagree. Must be read before the
        wrapped call — `log` clears the buffer on its way out.
        """
        import torch

        acc: dict[str, list[float]] = {}
        for ep in getattr(logger, "ep_extras", None) or []:
            for key, val in ep.items():
                name = str(key)
                if not name.startswith("Episode_Reward/"):
                    continue
                try:
                    v = (float(val.mean()) if isinstance(val, torch.Tensor)
                         else float(val))
                except (TypeError, ValueError):
                    continue
                acc.setdefault(name[len("Episode_Reward/"):], []).append(v)
        return {k: statistics.mean(v) for k, v in acc.items() if v}

    def _logged(it, start_it, total_it, *a, **kw):
        # The wrapped call must happen no matter what: it is the run's own
        # logging, not ours. Anything we add is best-effort on top.
        try:
            # rsl_rl calls this positionally:
            #   log(it, start_it, total_it, collect_time, learn_time,
            #       loss_dict, learning_rate, action_std, rnd_weight, ...)
            # so after the three named parameters `action_std` is a[4].
            # Guarded rather than indexed blindly: a signature change should
            # cost this one field, not the whole event.
            action_std = kw.get("action_std")
            if action_std is None and len(a) >= 5:
                action_std = a[4]
            std = (float(action_std.mean())
                   if hasattr(action_std, "mean") else None)
            comps = _components()
            top = max(comps.items(), key=lambda kv: kv[1], default=None)
            bottom = min(comps.items(), key=lambda kv: kv[1], default=None)
            payload = {
                "type": "learning_vitals",
                "rl_iter": int(it),
                "rl_total": int(total_it or total_iters),
                "mean_reward": _mean(getattr(logger, "rewbuffer", [])),
                "mean_ep_len": _mean(getattr(logger, "lenbuffer", [])),
                "action_std": std,
            }
            # The pair that decides whether the task is worth doing: what pays
            # most, and what costs most. When the cost outgrows the pay, the
            # policy's best move is to stop trying — which reads from outside
            # as "the reward does nothing".
            if top is not None and top[1] > 0:
                payload["top_reward"] = {"term": top[0],
                                         "value": round(top[1], 4)}
            if bottom is not None and bottom[1] < 0:
                payload["top_penalty"] = {"term": bottom[0],
                                          "value": round(bottom[1], 4)}
            print("[SCULPT-EVENT] " + json.dumps(payload), flush=True)
        except Exception as e:  # noqa: BLE001 — never break a run to report on it
            print(f"[runner] learning vitals skipped: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        return original(it, start_it, total_it, *a, **kw)

    logger.log = _logged
    return True


def _completed_iter_progress_event(
    *, max_iterations: int, elapsed_s: float, completed: bool,
) -> dict[str, Any] | None:
    """Return the terminal 100% tick only after ``runner.learn`` succeeds.

    This deliberately returns ``None`` when training raises.  Emitting a
    synthetic max-iteration event from a ``finally`` block made interrupted
    runs look complete in the UI even though only the last periodic
    ``model_N.pt`` snapshot was durable.
    """
    if not completed:
        return None
    return {
        "type": "iter_progress",
        "rl_iter": int(max_iterations),
        "rl_total": int(max_iterations),
        "pct": 100.0,
        "elapsed_s": round(float(elapsed_s), 1),
        "eta_s": 0.0,
    }


def _install_scalar_std_guard(runner: Any, *, minimum: float = 1e-4) -> Any:
    """Keep directly-parameterized Gaussian policy noise positive.

    Some rsl_rl task configs still use ``GaussianDistribution`` with
    ``std_type=\"scalar\"``.  That representation is an unconstrained trainable
    parameter, so a perfectly finite PPO update can move one action's standard
    deviation below zero.  The *next* minibatch then fails inside
    ``torch.normal`` (``normal expects all elements of std >= 0.0``), often near
    the end of an otherwise healthy long run.

    Clamp only the legacy direct ``std_param`` after every optimizer step.  Log
    parameterizations and non-Gaussian distributions have no ``std_param`` and
    remain untouched.  Returning the optimizer hook handle lets the caller
    remove it deterministically when training finishes.
    """
    algorithm = getattr(runner, "alg", None)
    actor = getattr(algorithm, "actor", None)
    distribution = getattr(actor, "distribution", None)
    std_param = getattr(distribution, "std_param", None)
    optimizer = getattr(algorithm, "optimizer", None)
    register_hook = getattr(optimizer, "register_step_post_hook", None)
    if std_param is None:
        return None
    if not callable(register_hook):
        raise RuntimeError(
            "rsl_rl uses an unconstrained scalar policy standard deviation, "
            "but this PyTorch optimizer cannot install the required positivity "
            "guard"
        )

    def _clamp_scalar_std(*_unused: Any) -> None:
        # ``detach`` shares storage without recording the repair in autograd;
        # it also avoids importing torch in this runner's lightweight paths.
        std_param.detach().clamp_(min=float(minimum))

    # A resumed checkpoint may already contain an invalid value, so repair it
    # once before the first sample as well as after every subsequent step.
    _clamp_scalar_std()
    handle = register_hook(_clamp_scalar_std)
    print(
        "[SCULPT-EVENT] " + json.dumps({
            "type": "policy_std_guard_installed",
            "parameterization": "scalar",
            "minimum": float(minimum),
        }),
        flush=True,
    )
    return handle


def _event_observation_extension_width(world_bundle: Any | None) -> int:
    """Width of the manifest-declared event phase observation, or zero."""
    manifest = getattr(world_bundle, "manifest", None)
    task_shared = getattr(manifest, "task_shared", {})
    event_sequence = (
        task_shared.get("event_sequence")
        if isinstance(task_shared, Mapping)
        else None
    )
    phases = (
        event_sequence.get("phases")
        if isinstance(event_sequence, Mapping)
        else None
    )
    return len(phases) if isinstance(phases, list) else 0


def _zero_extend_observation_state_dict(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    extension_width: int,
    role: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Explicitly migrate one feed-forward observation interface.

    Only the first MLP input columns and matching normalizer vectors may grow,
    and only by the manifest-declared event observation width.  Every key and
    every other tensor shape remains exact.  New network columns are zero, so
    the inherited policy is behaviorally identical until PPO learns to consume
    the phase; normalizer defaults are mean=0, std/variance=1.
    """
    import torch

    if extension_width <= 0:
        raise RuntimeError("observation extension width must be positive")
    if set(source) != set(target):
        raise RuntimeError(
            f"{role} checkpoint keys differ from the effective policy contract"
        )
    adapted = dict(source)
    changed: list[str] = []
    input_key = "mlp.0.weight"
    normalizer_padding = {
        "obs_normalizer._mean": 0.0,
        "obs_normalizer._std": 1.0,
        "obs_normalizer._var": 1.0,
    }
    for key in source:
        source_value = source[key]
        target_value = target[key]
        source_shape = tuple(getattr(source_value, "shape", ()))
        target_shape = tuple(getattr(target_value, "shape", ()))
        if source_shape == target_shape:
            continue
        if key == input_key and (
            len(source_shape) == 2
            and len(target_shape) == 2
            and source_shape[0] == target_shape[0]
            and target_shape[1] == source_shape[1] + extension_width
        ):
            padding = torch.zeros(
                (source_shape[0], extension_width),
                device=source_value.device,
                dtype=source_value.dtype,
            )
            adapted[key] = torch.cat((source_value, padding), dim=1)
            changed.append(key)
            continue
        if key in normalizer_padding and (
            len(source_shape) in (1, 2)
            and len(target_shape) == len(source_shape)
            and source_shape[:-1] == target_shape[:-1]
            and target_shape[-1] == source_shape[-1] + extension_width
        ):
            padding_shape = source_shape[:-1] + (extension_width,)
            padding = torch.full(
                padding_shape,
                normalizer_padding[key],
                device=source_value.device,
                dtype=source_value.dtype,
            )
            adapted[key] = torch.cat((source_value, padding), dim=-1)
            changed.append(key)
            continue
        raise RuntimeError(
            f"{role} tensor {key!r} shape {source_shape} cannot satisfy "
            f"effective target shape {target_shape}"
        )
    if input_key not in changed:
        raise RuntimeError(
            f"{role} warm start did not require the declared "
            f"{extension_width}-column event observation extension"
        )
    for key, value in adapted.items():
        if tuple(getattr(value, "shape", ())) != tuple(
                getattr(target[key], "shape", ())):
            raise RuntimeError(
                f"{role} adapted tensor {key!r} still violates target shape"
            )
    return adapted, tuple(changed)


def _prepare_event_observation_warm_start(
    runner: Any,
    checkpoint: Path,
    *,
    output_dir: Path,
    extension_width: int,
    load_role: str,
) -> tuple[Path, dict[str, Any]]:
    """Materialize and hash a provenance-bearing compatible checkpoint."""
    import hashlib
    import torch

    loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if not isinstance(loaded, dict):
        raise RuntimeError("warm-start checkpoint is not a state dictionary")
    algorithm = getattr(runner, "alg", None)
    role_models = {
        "actor": getattr(algorithm, "actor", None),
        "critic": getattr(algorithm, "critic", None),
    }
    requested_roles = (
        ("actor",) if load_role == "actor_only" else ("actor", "critic")
    )
    needs_extension: list[bool] = []
    for role in requested_roles:
        model = role_models[role]
        source_state = loaded.get(f"{role}_state_dict")
        if model is None or not isinstance(source_state, Mapping):
            raise RuntimeError(f"checkpoint/runner has no {role} state")
        source_input = source_state.get("mlp.0.weight")
        target_input = model.state_dict().get("mlp.0.weight")
        source_shape = tuple(getattr(source_input, "shape", ()))
        target_shape = tuple(getattr(target_input, "shape", ()))
        needs_extension.append(source_shape != target_shape)
    if not any(needs_extension):
        for role in requested_roles:
            source_state = loaded[f"{role}_state_dict"]
            target_state = role_models[role].state_dict()
            if set(source_state) != set(target_state) or any(
                tuple(getattr(source_state[key], "shape", ()))
                != tuple(getattr(target_state[key], "shape", ()))
                for key in source_state
            ):
                raise RuntimeError(
                    f"{role} differs from the effective event policy contract"
                )
        return checkpoint, {
            "adapted": False,
            "extension_width": int(extension_width),
            "observation_term": "authored_event_phase",
        }
    if not all(needs_extension):
        raise RuntimeError(
            "actor and critic disagree about the event observation extension"
        )
    adapted_checkpoint = dict(loaded)
    changed: dict[str, list[str]] = {}
    for role in requested_roles:
        model = role_models[role]
        if model is None:
            raise RuntimeError(f"runner has no {role} model")
        state_key = f"{role}_state_dict"
        source_state = loaded.get(state_key)
        if not isinstance(source_state, Mapping):
            raise RuntimeError(f"checkpoint has no {state_key}")
        adapted_state, changed_keys = _zero_extend_observation_state_dict(
            source_state,
            model.state_dict(),
            extension_width=extension_width,
            role=role,
        )
        adapted_checkpoint[state_key] = adapted_state
        changed[role] = list(changed_keys)

    adapted_path = output_dir / "warm_start_event_observation.pt"
    torch.save(adapted_checkpoint, adapted_path)
    digest = hashlib.sha256(adapted_path.read_bytes()).hexdigest()
    return adapted_path, {
        "adapted": True,
        "adapted_checkpoint": str(adapted_path),
        "adapted_checkpoint_sha256": digest,
        "extension_width": int(extension_width),
        "observation_term": "authored_event_phase",
        "padding": {
            "input_columns": 0.0,
            "normalizer_mean": 0.0,
            "normalizer_std": 1.0,
            "normalizer_variance": 1.0,
        },
        "changed_tensors": changed,
    }


def _event_policy_contract_admission_kind(
    admitted_migration: Mapping[str, Any],
    effective_contract: Mapping[str, Any],
    *,
    extension_width: int,
) -> str:
    """Classify the two admitted event warm-start contract relations.

    A schema-3 checkpoint may load directly when its source and target
    contracts are exact.  A schema-2 checkpoint may cross the event interface
    boundary only through the one explicit zero-column migration.  Keeping
    this decision pure makes the remote runner fail closed before inspecting
    or adapting checkpoint tensors.
    """
    expected_migration = {
        "type": "zero_initialized_event_phase_observation",
        "from_schema": 2,
        "to_schema": 3,
        "observation_term": "authored_event_phase",
        "extension_width": int(extension_width),
        "ordered_phase_ids": ["route", "jump", "hold"],
        "optimizer_resume": False,
    }
    if dict(admitted_migration) == expected_migration:
        return "zero_initialized_event_phase_observation"

    effective_schema = effective_contract.get("schema")
    expected_exact = {
        "type": "exact_policy_contract",
        "from_schema": effective_schema,
        "to_schema": effective_schema,
        "optimizer_resume": False,
    }
    if dict(admitted_migration) == expected_exact:
        return "exact_policy_contract"

    raise RuntimeError(
        "runner received an unrecognized event-observation "
        "policy-contract migration"
    )


def _expected_warm_start_checkpoint_sha256() -> str | None:
    """Resolve the immutable source digest pin for every warm-start kind.

    ``SCULPTOR_WARM_START_CHECKPOINT_SHA256`` is the generic launch
    authority used by project checkpoints and interrupted snapshots.  The
    older starting-skill-specific name remains an admitted compatibility
    alias for portable imports.  If both are present they must name the same
    bytes; otherwise the runner fails before ``runner.load``.
    """
    generic = os.environ.get("SCULPTOR_WARM_START_CHECKPOINT_SHA256")
    imported_skill = os.environ.get(
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256"
    )
    if generic and imported_skill and generic != imported_skill:
        raise RuntimeError(
            "generic warm-start and starting-skill checkpoint digest pins "
            "disagree"
        )
    expected = generic or imported_skill
    if expected is None:
        return None
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError(
            "warm-start checkpoint digest pin must be canonical lowercase "
            "SHA-256"
        )
    return expected


def _verify_warm_start_checkpoint_sha256(actual_sha256: str) -> str | None:
    """Verify checkpoint bytes against the launch pin, when one is set."""
    expected = _expected_warm_start_checkpoint_sha256()
    if expected and actual_sha256 != expected:
        raise RuntimeError(
            "pretrained policy digest differs from the immutable "
            "warm-start launch pin: expected "
            f"{expected}, got {actual_sha256}"
        )
    return expected


def _attest_warm_start_policy_contract(
    *,
    world_selection: str | Path,
    extension_width: int,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Rebuild and attest every UI-pinned warm-start policy contract.

    Contract admission is independent of whether checkpoint tensors need an
    event-observation extension.  In particular, an exact schema-2 load still
    has to prove its source receipt, immutable target selection, and declared
    exact migration before ``runner.load`` sees the checkpoint.
    """
    from sculptor.policy_contract import (
        build_project_policy_contract,
        contract_fingerprint,
        policy_contract_migration,
    )

    pin_names = (
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON",
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256",
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256",
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON",
    )
    pins = {name: os.environ.get(name) for name in pin_names}
    contract_pin_present = any(pins.values())
    if contract_pin_present and not all(pins.values()):
        raise RuntimeError("warm-start policy-contract pin is incomplete")
    if not contract_pin_present and extension_width > 0:
        raise RuntimeError(
            "event-interface warm start requires a full immutable source/"
            "target policy-contract receipt; target-only CLI adaptation is "
            "not an admitted compatibility proof"
        )
    if not contract_pin_present:
        return {
            "active": False,
            "contract_pin_present": False,
            "effective_contract": None,
            "effective_contract_sha256": None,
            "admitted_migration": None,
            "admitted_contract_receipt": None,
            "admission_kind": None,
        }

    selection_path = Path(world_selection).expanduser().resolve()
    if not selection_path.is_file():
        raise RuntimeError(
            "warm-start policy-contract attestation requires the immutable "
            f"world selection: {selection_path}"
        )
    project_dir = selection_path.parent.parent
    try:
        actual_contract = build_project_policy_contract(
            project_dir,
            world_selection_path=selection_path,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "failed to rebuild the effective policy contract from the "
            "immutable world selection"
        ) from exc
    actual_contract_sha256 = contract_fingerprint(actual_contract)

    if contract_pin_present:
        try:
            effective_contract = json.loads(str(
                pins["SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON"]
            ))
            admitted_migration = json.loads(str(
                pins["SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON"]
            ))
            admitted_contract_receipt = json.loads(str(
                pins[
                    "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON"
                ]
            ))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "warm-start policy-contract pin is invalid JSON"
            ) from exc
        if not isinstance(effective_contract, dict) or not isinstance(
            admitted_migration, dict
        ):
            raise RuntimeError(
                "warm-start policy-contract pin must contain objects"
            )
        if (
            not isinstance(admitted_contract_receipt, dict)
            or admitted_contract_receipt.get("schema") != 1
            or not isinstance(admitted_contract_receipt.get("source"), dict)
            or not isinstance(admitted_contract_receipt.get("target"), dict)
        ):
            raise RuntimeError(
                "warm-start policy-contract receipt is malformed"
            )
        pinned_contract_sha256 = str(
            pins["SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256"]
        )
        source_contract_sha256 = str(
            pins["SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256"]
        )
        if (
            contract_fingerprint(effective_contract)
            != pinned_contract_sha256
        ):
            raise RuntimeError(
                "effective policy-contract JSON disagrees with its "
                "immutable fingerprint pin"
            )
        if (
            actual_contract != effective_contract
            or actual_contract_sha256 != pinned_contract_sha256
        ):
            raise RuntimeError(
                "effective policy contract rebuilt from the immutable world "
                "selection differs from the pre-queue launch pin"
            )
        admission_kind = _event_policy_contract_admission_kind(
            admitted_migration,
            effective_contract,
            extension_width=extension_width,
        )
        receipt_source = admitted_contract_receipt["source"]
        receipt_target = admitted_contract_receipt["target"]
        receipt_source_contract = receipt_source.get("contract")
        receipt_checkpoint_sha256 = receipt_source.get("checkpoint_sha256")
        if isinstance(receipt_source_contract, dict):
            recomputed_migration = policy_contract_migration(
                receipt_source_contract,
                actual_contract,
            )
            if receipt_source_contract == actual_contract:
                recomputed_compatibility: dict[str, Any] | None = {
                    "type": "exact_policy_contract",
                    "from_schema": receipt_source_contract.get("schema"),
                    "to_schema": actual_contract.get("schema"),
                    "optimizer_resume": False,
                }
            else:
                recomputed_compatibility = recomputed_migration
        else:
            recomputed_compatibility = None
        if (
            not isinstance(receipt_source_contract, dict)
            or recomputed_compatibility is None
            or recomputed_compatibility != admitted_migration
            or contract_fingerprint(receipt_source_contract)
            != source_contract_sha256
            or receipt_source.get("contract_sha256")
            != source_contract_sha256
            or receipt_target.get("contract") != effective_contract
            or receipt_target.get("contract_sha256")
            != actual_contract_sha256
            or admitted_contract_receipt.get("compatibility")
            != admitted_migration
            or (
                receipt_checkpoint_sha256 is not None
                and receipt_checkpoint_sha256
                != source_checkpoint_sha256
            )
        ):
            raise RuntimeError(
                "warm-start policy-contract receipt disagrees with the "
                "admitted runner pins or checkpoint"
            )
    interface = effective_contract.get("event_observation")
    interface_phases = (
        interface.get("ordered_phase_ids")
        if isinstance(interface, dict)
        else None
    )
    if extension_width > 0:
        if (
            interface_phases != ["route", "jump", "hold"]
            or len(interface_phases) != extension_width
        ):
            raise RuntimeError(
                "effective policy contract lacks the runtime event "
                "observation interface"
            )
    elif interface is not None:
        raise RuntimeError(
            "pinned policy contract declares an event observation, but the "
            "applied runtime world has no event interface"
        )

    return {
        "active": True,
        "contract_pin_present": contract_pin_present,
        "effective_contract": effective_contract,
        "effective_contract_sha256": actual_contract_sha256,
        "source_contract_sha256": source_contract_sha256,
        "admitted_migration": admitted_migration,
        "admitted_contract_receipt": admitted_contract_receipt,
        "admission_kind": admission_kind,
    }


def _warm_start_loaded_receipt(
    *,
    requested_source: Path,
    requested_sha256: str,
    loaded_checkpoint: Path,
    loaded_sha256: str,
    load_cfg: Mapping[str, bool],
    effective_policy_contract_sha256: str | None,
    extension_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Earned initialization receipt for the exact bytes runner.load used."""
    adapted = bool(extension_receipt.get("adapted"))
    return {
        "type": "warm_start_loaded",
        # Backwards-readable aliases keep denoting requested source intent.
        "source": str(requested_source),
        "source_sha256": requested_sha256,
        "source_sha8": requested_sha256[:8],
        "requested_source": str(requested_source),
        "requested_source_sha256": requested_sha256,
        "loaded_checkpoint": str(loaded_checkpoint),
        "loaded_checkpoint_sha256": loaded_sha256,
        "loaded_checkpoint_sha8": loaded_sha256[:8],
        "adapted": adapted,
        "derived_from": (
            {
                "source": str(requested_source),
                "source_sha256": requested_sha256,
            }
            if adapted
            else None
        ),
        "policy_contract_migration": (
            "zero_initialized_event_phase_observation"
            if adapted
            else None
        ),
        "effective_policy_contract_sha256": (
            effective_policy_contract_sha256
        ),
        "source_policy_contract_sha256": extension_receipt.get(
            "source_policy_contract_sha256"
        ),
        "admitted_policy_contract_migration": extension_receipt.get(
            "admitted_policy_contract_migration"
        ),
        "policy_contract_receipt": extension_receipt.get(
            "policy_contract_receipt"
        ),
        "policy_contract_receipt_sha256": extension_receipt.get(
            "policy_contract_receipt_sha256"
        ),
        "load_cfg_keys": sorted(
            key for key, value in load_cfg.items() if value
        ),
    }


def _cmd_train(args: argparse.Namespace) -> None:
    # Lazy heavy imports — stay out of the module top.
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs
    # Authored geometry/task semantics are applied first. Legacy EnvSpec then
    # overlays only its separate reset/randomization/optimizer surface.
    world_bundle = _apply_world_selection(
        env_cfg, getattr(args, "world_selection", ""), train=True,
        task_id=args.task_id)
    # §RL_SCULPTOR_AUDIT: per-project env spec (--env-spec file wins over
    # a named --env-profile preset; neither → task defaults, no-op).
    # train=True additionally applies the train-only curricula section.
    env_spec = _resolve_env_spec(args)
    _apply_env_spec(env_cfg, env_spec, train=True, task_id=args.task_id)
    # Authored worlds already carry their admitted actuator profile. Only a
    # legacy registered task may receive the environment-gated compatibility
    # transform, and it must receive it in both training and rollout.
    if world_bundle is None:
        _enforce_actuator_limits(env_cfg)

    # Reward injection (optional).
    if args.reward_module_path:
        from mjlab.envs import mdp as envs_mdp
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
        # Keep them at 0.3× so the physics-plausible prior complements
        # sculptor_primary (weight=1.0 below).  A default reward that tracks a
        # command replaced by the authored World is different: it is now dense
        # TASK supervision, so preserve its nominal weight.  This lets a route
        # command teach lateral turns and heading while the generated reward
        # remains responsible for completion, contact, and terminal-hold intent.
        existing = getattr(env_cfg, "rewards", None)
        REALISM_FLOOR_SCALE = 0.3
        full_weight_terms = _full_weight_authored_command_rewards(world_bundle)
        # When a REFERENCE is attached the realism prior stops being a
        # complement and becomes a competitor: the reference already dictates
        # posture, gait and body motion frame by frame, so `pose` (a nominal-
        # pose regularizer), `upright`, the command-tracking terms and the
        # gait-shaping terms are all pulling the policy AWAY from the motion
        # it is being certified against. Measured on the first Tier-D attempt:
        # 14 task terms at 0.3x against one tracking term at 1.0x produced a
        # policy that reproduced 28% of the reference's joint amplitude and
        # could not beat a static pose.
        #
        # So in tracking mode the floor is narrowed to HARDWARE-SAFETY terms
        # only — limits, self-collision, actuator smoothness — which constrain
        # what the robot may do without prescribing what pose it holds. The
        # anti-falling role the broad floor used to play is covered by
        # `_install_sculptor_termination_economics` below (survival guard +
        # explicit non-timeout termination penalty), which is why dropping
        # `upright` here does not reopen the fall-immediately failure mode.
        tracking = _reward_module_declares(args.reward_module_path,
                                           "reference_tracking")
        if isinstance(existing, dict):
            kept, dropped = [], []
            for k in list(existing.keys()):
                term = existing[k]
                if term is None or not hasattr(term, "weight"):
                    continue
                name = str(k)
                if name in full_weight_terms:
                    scale = 1.0            # authored command == task supervision
                elif tracking and not _is_hardware_safety_term(name):
                    scale = 0.0
                else:
                    scale = REALISM_FLOOR_SCALE
                term.weight = float(term.weight) * scale
                (dropped if scale == 0.0 else kept).append(name)
            if tracking:
                print(
                    f"[runner] reference-tracking reward: narrowed the realism "
                    f"floor to hardware-safety terms. kept={sorted(kept)} "
                    f"dropped={sorted(dropped)}",
                    file=sys.stderr, flush=True)
        else:
            env_cfg.rewards = {}

        schema_keys = tuple(args.schema_keys.split(",")) if args.schema_keys else _DEFAULT_SCHEMA_KEYS
        terminal_hold_s = _authored_terminal_hold_s(world_bundle)
        terminal_standing = terminal_hold_s > 0.0
        terminal_stillness_weight = _authored_terminal_stillness_weight(
            env_cfg.rewards, full_weight_terms)
        forbidden_contact_sensors = (
            _authored_forbidden_contact_sensor_names(world_bundle))
        forbidden_contact_weight = _authored_forbidden_contact_weight(
            env_cfg.rewards, full_weight_terms)
        if forbidden_contact_sensors:
            env_cfg.rewards["sculptor_forbidden_contact"] = RewardTermCfg(
                func=_authored_forbidden_contact_penalty,
                weight=-forbidden_contact_weight,
                params={"sensor_names": forbidden_contact_sensors},
            )
        if terminal_standing:
            AuthoredTerminalStillnessTerm = (
                _build_authored_terminal_stillness_term_class())
            env_cfg.rewards["sculptor_terminal_stillness"] = RewardTermCfg(
                func=AuthoredTerminalStillnessTerm,
                weight=terminal_stillness_weight,
                params={
                    "lin_std": 0.12,
                    "ang_std": 0.5,
                    "joint_std": 1.0,
                    "upright_z_max": -0.7,
                    "joint_pos_tolerance": 0.6,
                    "upright_std": 0.35,
                    "joint_pos_std": 0.6,
                    "hold_s": terminal_hold_s,
                    "continuity_scale": _SCULPTOR_TERMINAL_CONTINUITY_SCALE,
                },
            )
        SculptorRewardTerm = _build_sculptor_term_class(
            schema_keys,
            _load_authored_robot_capability(
                getattr(args, "world_selection", "")),
            world_bundle,
        )
        env_cfg.rewards["sculptor_primary"] = RewardTermCfg(
            func=SculptorRewardTerm,
            weight=1.0,
            params={"reward_module_path": args.reward_module_path},
        )
        _install_sculptor_termination_economics(
            env_cfg.rewards,
            RewardTermCfg,
            envs_mdp,
        )
        print(
            f"[runner] injected SculptorRewardTerm; {sum(1 for t in env_cfg.rewards.values() if t and getattr(t, 'weight', 0) == 0)} default terms zeroed",
            file=sys.stderr, flush=True,
        )
        if full_weight_terms:
            print(
                "[runner] preserved authored command supervision at full weight: "
                + ", ".join(sorted(full_weight_terms)),
                file=sys.stderr,
                flush=True,
            )
        if terminal_standing:
            print(
                "[runner] installed authored terminal continuity-aware "
                "whole-body stillness supervision with strict multiplicative "
                "posture conjunction at weight "
                f"{terminal_stillness_weight:g}, "
                f"hold_s {terminal_hold_s:g}, continuity scale "
                f"{_SCULPTOR_TERMINAL_CONTINUITY_SCALE:g}",
                file=sys.stderr,
                flush=True,
            )
        if forbidden_contact_sensors:
            print(
                "[runner] installed authored forbidden-contact supervision "
                f"at weight {-forbidden_contact_weight:g} from sensors: "
                + ", ".join(forbidden_contact_sensors),
                file=sys.stderr,
                flush=True,
            )
        if any(
            "outside approach stage" in str(adjustment)
            for adjustment in (
                getattr(world_bundle, "runtime_adjustments", ()) or ())
        ):
            print(
                "[runner] installed clearance-maneuver reward firewall: "
                "predicate-centered generated reward withheld through "
                "command-only safe approach and traversal until the frozen "
                "waypoint advances; command/contact/survival supervision "
                "remains active",
                file=sys.stderr,
                flush=True,
            )
        print(
            "[runner] installed termination economics: "
            f"survival={_SCULPTOR_SURVIVAL_WEIGHT:+g}/step, "
            f"non-timeout failure={_SCULPTOR_FAILURE_WEIGHT:+g}",
            file=sys.stderr,
            flush=True,
        )

    env = ManagerBasedRlEnv(env_cfg, device=args.device)

    # Use mjlab's own runner shim (not rsl_rl's OnPolicyRunner directly —
    # the cfg structure differs). Mirrors mjlab/scripts/train.py:148-165.
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls

    rl_cfg = load_rl_cfg(args.task_id)
    rl_cfg.max_iterations = args.max_iterations
    # PPO exploration from the env spec's train section (e.g.
    # entropy_coef_scale). Train-time only — rollout never optimizes.
    _apply_rl_spec(rl_cfg, env_spec)

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
        load_role = str(args.pretrained_load_role or "actor_critic")
        if load_role not in ("actor_only", "actor_critic"):
            raise ValueError(
                "--pretrained-load-role must be actor_only or actor_critic"
            )
        load_cfg = {
            "actor": True,
            "critic": load_role == "actor_critic",
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        source_digest = _hashlib.sha256()
        with ckpt.open("rb") as checkpoint_stream:
            for chunk in iter(lambda: checkpoint_stream.read(1 << 20), b""):
                source_digest.update(chunk)
        source_sha256 = source_digest.hexdigest()
        _verify_warm_start_checkpoint_sha256(source_sha256)
        load_path = ckpt
        effective_contract_sha256: str | None = None
        extension_receipt: dict[str, Any] = {"adapted": False}
        extension_width = _event_observation_extension_width(world_bundle)
        contract_attestation = _attest_warm_start_policy_contract(
            world_selection=getattr(args, "world_selection", ""),
            extension_width=extension_width,
            source_checkpoint_sha256=source_sha256,
        )
        # Exact schema-2 warm starts have no tensor extension, but they still
        # carry the same immutable receipt/selection attestation as event
        # migrations.  Persist that proof before runner.load rather than
        # silently skipping all contract checks when extension_width == 0.
        if contract_attestation["active"] and extension_width == 0:
            effective_contract = contract_attestation[
                "effective_contract"
            ]
            effective_contract_sha256 = contract_attestation[
                "effective_contract_sha256"
            ]
            admitted_migration = contract_attestation[
                "admitted_migration"
            ]
            admitted_contract_receipt = contract_attestation[
                "admitted_contract_receipt"
            ]
            effective_contract_path = (
                output_dir / "warm_start_effective_policy_contract.json"
            )
            effective_contract_path.write_text(
                json.dumps(
                    effective_contract,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            receipt_artifact_path: Path | None = None
            receipt_artifact_sha256: str | None = None
            if admitted_contract_receipt is not None:
                receipt_artifact_path = (
                    output_dir / "warm_start_policy_contract_receipt.json"
                )
                receipt_artifact_path.write_text(
                    json.dumps(
                        admitted_contract_receipt,
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
                receipt_artifact_sha256 = _hashlib.sha256(
                    receipt_artifact_path.read_bytes()
                ).hexdigest()
            extension_receipt.update({
                "extension_width": 0,
                "observation_term": None,
                "source_policy_contract_sha256": contract_attestation[
                    "source_contract_sha256"
                ],
                "admitted_policy_contract_migration": admitted_migration,
                "policy_contract_receipt": (
                    str(receipt_artifact_path)
                    if receipt_artifact_path is not None
                    else None
                ),
                "policy_contract_receipt_sha256": (
                    receipt_artifact_sha256
                ),
            })
            print(
                "[SCULPT-EVENT] " + json.dumps({
                    "type": "warm_start_observation_contract_verified",
                    "source": str(ckpt),
                    "source_sha256": source_sha256,
                    "effective_policy_contract": str(
                        effective_contract_path),
                    "effective_policy_contract_sha256": (
                        effective_contract_sha256),
                    "effective_policy_contract_schema": (
                        effective_contract.get("schema")),
                    **extension_receipt,
                }),
                flush=True,
            )
        if extension_width > 0:
            from sculptor.policy_contract import (
                build_project_policy_contract,
                contract_fingerprint,
            )

            pinned_contract_json = os.environ.get(
                "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON"
            )
            pinned_contract_sha256 = os.environ.get(
                "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256"
            )
            source_contract_sha256 = os.environ.get(
                "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256"
            )
            pinned_migration_json = os.environ.get(
                "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON"
            )
            pinned_receipt_json = os.environ.get(
                "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON"
            )
            contract_pin_present = any((
                pinned_receipt_json,
                pinned_contract_json,
                pinned_contract_sha256,
                source_contract_sha256,
                pinned_migration_json,
            ))
            if contract_pin_present and not all((
                pinned_receipt_json,
                pinned_contract_json,
                pinned_contract_sha256,
                source_contract_sha256,
                pinned_migration_json,
            )):
                raise RuntimeError(
                    "warm-start policy-contract pin is incomplete"
                )
            if contract_pin_present:
                try:
                    effective_contract = json.loads(
                        str(pinned_contract_json)
                    )
                    admitted_migration = json.loads(
                        str(pinned_migration_json)
                    )
                    admitted_contract_receipt = json.loads(
                        str(pinned_receipt_json)
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "warm-start policy-contract pin is invalid JSON"
                    ) from exc
                if not isinstance(effective_contract, dict) or not isinstance(
                    admitted_migration, dict
                ):
                    raise RuntimeError(
                        "warm-start policy-contract pin must contain objects"
                    )
                if (
                    not isinstance(admitted_contract_receipt, dict)
                    or admitted_contract_receipt.get("schema") != 1
                    or not isinstance(
                        admitted_contract_receipt.get("source"), dict
                    )
                    or not isinstance(
                        admitted_contract_receipt.get("target"), dict
                    )
                ):
                    raise RuntimeError(
                        "warm-start policy-contract receipt is malformed"
                    )
                admission_kind = _event_policy_contract_admission_kind(
                    admitted_migration,
                    effective_contract,
                    extension_width=extension_width,
                )
            else:
                # Direct local CLI use retains deterministic contract building.
                # UI/remote launches always carry the pre-queue immutable pin
                # above and therefore never trust an unsynced config rebuild.
                selection_path = Path(args.world_selection).resolve()
                project_dir = selection_path.parent.parent
                effective_contract = build_project_policy_contract(
                    project_dir,
                    world_selection_path=selection_path,
                )
                admitted_migration = None
                admitted_contract_receipt = None
                admission_kind = None
            effective_contract_sha256 = contract_fingerprint(
                effective_contract)
            if (
                pinned_contract_sha256
                and effective_contract_sha256 != pinned_contract_sha256
            ):
                raise RuntimeError(
                    "effective policy contract differs from the immutable "
                    "pre-queue launch pin"
                )
            if admitted_contract_receipt is not None:
                receipt_source = admitted_contract_receipt["source"]
                receipt_target = admitted_contract_receipt["target"]
                receipt_source_contract = receipt_source.get("contract")
                if (
                    not isinstance(receipt_source_contract, dict)
                    or contract_fingerprint(receipt_source_contract)
                    != source_contract_sha256
                    or receipt_target.get("contract") != effective_contract
                    or receipt_target.get("contract_sha256")
                    != effective_contract_sha256
                    or receipt_source.get("contract_sha256")
                    != source_contract_sha256
                    or admitted_contract_receipt.get("compatibility")
                    != admitted_migration
                ):
                    raise RuntimeError(
                        "warm-start policy-contract receipt disagrees with "
                        "the admitted runner pins"
                    )
            interface = effective_contract.get("event_observation")
            if not isinstance(interface, dict) or interface.get(
                "ordered_phase_ids"
            ) != ["route", "jump", "hold"]:
                raise RuntimeError(
                    "effective policy contract lacks the admitted event "
                    "observation interface"
                )
            effective_contract_path = (
                output_dir / "warm_start_effective_policy_contract.json"
            )
            effective_contract_path.write_text(
                json.dumps(
                    effective_contract,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            receipt_artifact_path: Path | None = None
            receipt_artifact_sha256: str | None = None
            if admitted_contract_receipt is not None:
                receipt_artifact_path = (
                    output_dir / "warm_start_policy_contract_receipt.json"
                )
                receipt_artifact_path.write_text(
                    json.dumps(
                        admitted_contract_receipt,
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
                receipt_artifact_sha256 = _hashlib.sha256(
                    receipt_artifact_path.read_bytes()
                ).hexdigest()
            load_path, extension_receipt = (
                _prepare_event_observation_warm_start(
                    runner,
                    ckpt,
                    output_dir=output_dir,
                    extension_width=extension_width,
                    load_role=load_role,
                )
            )
            if contract_pin_present:
                adapted = bool(extension_receipt.get("adapted"))
                if admission_kind == "exact_policy_contract" and adapted:
                    raise RuntimeError(
                        "exact policy-contract warm start unexpectedly "
                        "required an event-observation migration"
                    )
                if (
                    admission_kind
                    == "zero_initialized_event_phase_observation"
                    and not adapted
                ):
                    raise RuntimeError(
                        "declared event-observation migration did not change "
                        "the checkpoint interface"
                    )
            extension_receipt.update({
                "source_policy_contract_sha256": source_contract_sha256,
                "admitted_policy_contract_migration": admitted_migration,
                "policy_contract_receipt": (
                    str(receipt_artifact_path)
                    if receipt_artifact_path is not None
                    else None
                ),
                "policy_contract_receipt_sha256": (
                    receipt_artifact_sha256
                ),
            })
            print(
                "[SCULPT-EVENT] " + json.dumps({
                    "type": (
                        "warm_start_observation_extended"
                        if extension_receipt.get("adapted")
                        else "warm_start_observation_contract_verified"
                    ),
                    "source": str(ckpt),
                    "source_sha256": source_sha256,
                    "effective_policy_contract": str(
                        effective_contract_path),
                    "effective_policy_contract_sha256": (
                        effective_contract_sha256),
                    "effective_policy_contract_schema": (
                        effective_contract.get("schema")),
                    **extension_receipt,
                }),
                flush=True,
            )
        loaded_digest = _hashlib.sha256()
        with load_path.open("rb") as checkpoint_stream:
            for chunk in iter(
                lambda: checkpoint_stream.read(1 << 20), b""
            ):
                loaded_digest.update(chunk)
        loaded_sha256 = loaded_digest.hexdigest()
        try:
            _ = runner.load(str(load_path), load_cfg=load_cfg)
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
            "[SCULPT-EVENT] " + json.dumps(
                _warm_start_loaded_receipt(
                    requested_source=ckpt,
                    requested_sha256=source_sha256,
                    loaded_checkpoint=load_path,
                    loaded_sha256=loaded_sha256,
                    load_cfg=load_cfg,
                    effective_policy_contract_sha256=(
                        effective_contract_sha256),
                    extension_receipt=extension_receipt,
                )
            ),
            flush=True,
        )
        # The actor weights just loaded include the learned exploration std,
        # which ratchets up across chained warm starts until the action-rate
        # penalty outweighs the task reward. Bound it by what a fresh policy
        # would start from; see `_clamp_warm_started_noise`.
        init_std = _configured_init_std(rl_cfg)
        clamped = (None if init_std is None
                   else _clamp_warm_started_noise(runner, init_std))
        if clamped is not None:
            print(
                "[SCULPT-EVENT] " + json.dumps({
                    "type": "warm_start_noise_clamped", **clamped}),
                flush=True,
            )
            print(
                f"[runner] warm start carried action-noise std "
                f"{clamped['std_before']:.3f}, above this task's fresh-init "
                f"{clamped['ceiling']:.3f} — clamped to "
                f"{clamped['std_after']:.3f} so the inherited noise does not "
                f"pay the action-rate penalty for the whole run",
                file=sys.stderr, flush=True,
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
    std_guard_handle = _install_scalar_std_guard(runner)
    # The progress poller above says how far along the run is; this says
    # whether it is going anywhere.
    if not _install_learning_vitals(runner, int(args.max_iterations)):
        print("[runner] learning vitals unavailable: runner exposes no logger",
              file=sys.stderr, flush=True)
    _learn_completed = False
    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
        _learn_completed = True
    finally:
        if std_guard_handle is not None:
            std_guard_handle.remove()
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
        # Final tick at 100% only when the learning call actually returned.
        # On an exception, the last periodic snapshot remains the honest
        # progress authority and the UI must continue to show interruption.
        _terminal_progress = _completed_iter_progress_event(
            max_iterations=int(args.max_iterations),
            elapsed_s=_time.time() - _t0,
            completed=_learn_completed,
        )
        if _terminal_progress is not None:
            print(
                "[SCULPT-EVENT] " + _json.dumps(_terminal_progress),
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

    _write_world_curriculum_stats(env, output_dir)
    env.close()
    print(json.dumps({"status": "ok", "checkpoint": str(ckpt_path)}))


def _write_world_curriculum_stats(env: Any, output_dir: Path) -> None:
    """§env-authoring §10: per-difficulty traversal statistics for the
    diagnoser. mjlab's terrain curriculum promotes each env's row
    (`terrain_levels`) on traversal success, so the end-of-training level
    distribution IS the per-difficulty success summary: mass at high
    levels = the policy earned promotion; mass pinned at level 0 = the
    easiest difficulty is still failing. Fail-soft by contract — plane
    terrain, non-curriculum grids, and legacy envs write nothing."""
    try:
        scene = getattr(getattr(env, "unwrapped", env), "scene", None)
        terrain = getattr(scene, "terrain", None)
        levels_t = getattr(terrain, "terrain_levels", None)
        if levels_t is None:
            return
        levels = [int(v) for v in levels_t.detach().cpu().tolist()]
        if not levels:
            return
        from collections import Counter

        histogram = Counter(levels)
        max_level = getattr(terrain, "max_terrain_level", None)
        stats = {
            "version": 1,
            "num_envs": len(levels),
            "mean_level": round(sum(levels) / len(levels), 3),
            "max_level": int(max_level) if max_level is not None else None,
            "histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        }
        (output_dir / "world_curriculum_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — stats are advisory, never fatal
        print(
            f"[runner] warning: world curriculum stats skipped: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )


_RENDER_DEFAULT_W = 1280
_RENDER_DEFAULT_H = 720


def _configure_rollout_viewer(env_cfg: Any, args: Any) -> None:
    """Rollout-video viewer settings. FULLY DEFENSIVE (same discipline as
    `_apply_ground_texture`): any mjlab ViewerConfig drift no-ops rather
    than break the rollout.

    Two fixes over mjlab defaults:
    * `max_extra_envs = 0` — the default (2) renders NEIGHBORING envs
      behind the tracked one, and those neighbors keep auto-resetting
      mid-episode, so ghost robots teleport around in the background of
      every video (the guard at the render loop only stops recording at
      env[0]'s OWN terminal; it never touched the neighbors).
    * 1280x720 instead of 320x240 — render cost measured on the WSL2
      path is resolution-independent (~200 ms/frame at both), so the
      default was leaving quality on the table for free.
    """
    try:
        viewer = getattr(env_cfg, "viewer", None)
        if viewer is None:
            return
        w = int(getattr(args, "render_width", 0) or 0) or _RENDER_DEFAULT_W
        h = int(getattr(args, "render_height", 0) or 0) or _RENDER_DEFAULT_H
        if hasattr(viewer, "width"):
            viewer.width = max(64, w)
        if hasattr(viewer, "height"):
            viewer.height = max(64, h)
        if hasattr(viewer, "max_extra_envs"):
            viewer.max_extra_envs = 0
        if hasattr(viewer, "env_idx"):
            viewer.env_idx = int(
                getattr(args, "render_env_index", 0) or 0)
    except Exception as e:  # noqa: BLE001 — cosmetics must never kill a rollout
        print(f"[runner] viewer config skipped: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def _hide_untracked_authored_geometry(env: Any, env_idx: int) -> None:
    """Make other environments' authored courses invisible in rollout video.

    `max_extra_envs = 0` above stops mjlab drawing neighbouring ROBOTS, but
    authored course geometry lives in the shared worldbody — one copy per env
    origin, always in the model, always drawn. At training widths that fills
    the frame with a field of identical courses and buries the one the tracked
    robot is actually running, which reads as a broken scene.

    Alpha only. `geom_rgba`/`site_rgba` are render inputs; collision geometry,
    contacts and every observation are untouched, so the video shows the same
    physics it always did — just the tracked environment's slice of it.

    FULLY DEFENSIVE, like the viewer config: cosmetics must never kill a run.
    """
    try:
        import re as _re

        model = env.sim.mj_model
        keep = f"__env_{int(env_idx):04d}"
        hidden = 0
        for array, count, getter in (
            (getattr(model, "geom_rgba", None), model.ngeom, model.geom),
            (getattr(model, "site_rgba", None), model.nsite, model.site),
        ):
            if array is None:
                continue
            for index in range(count):
                name = getter(index).name or ""
                if not _re.match(r"(obstacle|zone)__.*__env_\d{4}$", name):
                    continue
                if name.endswith(keep):
                    continue
                array[index][3] = 0.0
                hidden += 1
        if hidden:
            print(f"[runner] rollout video: hid {hidden} authored geoms/sites "
                  f"belonging to environments other than env {env_idx}",
                  file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001 — cosmetics must never kill a rollout
        print(f"[runner] authored-geometry culling skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)


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


# §actuator-limit enforcement (Sam 2026-06-20): mjlab's robot configs use
# BuiltinPositionActuatorCfg, which STRUCTURALLY drops each motor's velocity_limit
# (the ElectricActuator no-load speed) — so the sim clamps TORQUE (effort_limit) but
# never VELOCITY, and a trained policy drives joints 2-3.7× past the real no-load
# speed (g1-kick-v6: knee p99 43-73 vs 20 rad/s). mjlab already ships the
# research-standard fix — DcMotorActuator, a port of Isaac Lab's DCMotor torque-speed
# model (Rudin et al. 2022 / legged_gym): available torque falls LINEARLY to ~0 at
# velocity_limit (motor back-EMF), so the joint physically cannot be driven past its
# no-load speed. We swap the actuator MODEL on env_cfg (both train + rollout) and
# re-supply the velocity_limit mjlab dropped. Per-pattern no-load speeds, cited from
# mjlab's own robot constants (single source of truth):
#   G1  g1_constants.py:91-178   Go1 go1_constants.py:40-72
def _recover_velocity_limit(actuator_cfg: Any) -> "float | None":
    """The real motor no-load speed (rad/s) for an actuator group, recovered from
    its `target_names_expr` (cited from mjlab's robot constants — same source of
    truth that defines the group). None when no pattern matches (an unknown
    robot/group → the caller leaves it unchanged), so this never invents a limit."""
    from sculptor.world.capabilities import (
        LEGACY_UNITREE_DC_MOTOR_PROFILE,
        actuator_velocity_limit,
    )

    return actuator_velocity_limit(
        actuator_cfg, LEGACY_UNITREE_DC_MOTOR_PROFILE,
    )


def _enforce_actuator_limits(env_cfg: Any) -> None:
    """Swap every `BuiltinPositionActuatorCfg` → `DcMotorActuatorCfg` so the sim
    enforces each motor's VELOCITY (no-load speed) limit via mjlab's research-standard
    torque-speed model, on top of the torque (effort) limit it already clamps. Replaces
    each entity with an owned copy BEFORE the env is built, so it is active in BOTH
    train and rollout without mutating MjLab's shared robot constants.

    Gated by `RS_ENFORCE_ACTUATOR_LIMITS` for legacy registered tasks. Authored worlds
    compose the same versioned profile during admission, so a process flag cannot
    change their pinned physics. Any unknown group is left unchanged with a warning.
    `saturation_effort = effort_limit` gives the conservative triangular envelope."""
    import os
    if os.environ.get("RS_ENFORCE_ACTUATOR_LIMITS", "1").strip().lower() not in (
            "1", "true", "on", "yes"):
        return
    try:
        from sculptor.world.capabilities import (
            LEGACY_UNITREE_DC_MOTOR_PROFILE,
            apply_actuator_profile,
        )

        entities = getattr(getattr(env_cfg, "scene", None), "entities", None) or {}
        items = list(entities.items()) if hasattr(entities, "items") else []
        for ekey, ent in items:
            art = getattr(ent, "articulation", None)
            acts = list(getattr(art, "actuators", None) or []) if art is not None else []
            if not acts:
                continue
            owned, swapped, unresolved = apply_actuator_profile(
                ent, LEGACY_UNITREE_DC_MOTOR_PROFILE, strict=False,
            )
            entities[ekey] = owned
            msg = (f"[runner] actuator-limit enforcement: entity {ekey!r} — "
                   f"{swapped}/{len(acts)} groups → DcMotor (velocity-limited)")
            if unresolved:
                msg += f"; UNRESOLVED (left unchanged): {unresolved}"
            print(msg, file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001 — never break a run
        print(f"[runner] actuator-limit enforcement skipped: {type(e).__name__}: {e}",
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
    requested_render_env_index = int(
        getattr(args, "render_env_index", 0) or 0)
    render_env_index = max(
        0, min(requested_render_env_index, num_envs - 1))
    # Keep the normalized value as the single source for viewer config,
    # trajectory selection, and evidence metadata.
    args.render_env_index = render_env_index

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = num_envs
    # Rollout loads the materialized evaluation artifact from the selected
    # tuple; it never re-samples WorldSpec generators from a seed.
    world_bundle = _apply_world_selection(
        env_cfg, getattr(args, "world_selection", ""), train=False,
        task_id=args.task_id)
    # §RL_SCULPTOR_AUDIT: SAME spec as train, SHARED section only — a
    # policy trained with zero commands / no pushes must be evaluated
    # under that distribution, and the metric arrays must see the
    # un-truncated fall dynamics. train=False: the train-only curricula
    # (RSI resets, sunk termination, domain randomization) NEVER apply
    # here — evaluation starts from the honest task state, or the
    # metric's view (upright_start / return-to-start-height) would be
    # corrupted by mid-air spawns.
    _apply_env_spec(
        env_cfg, _resolve_env_spec(args), train=False, task_id=args.task_id)
    # §D17: stage-FIXED eval-reset override — reference-derived lying
    # start for get-up stages, applied ONLY here (never to training),
    # AFTER the shared-only env-spec above and strictly ADDITIVE to it.
    # Absent --eval-reset (the default, and every non-get-up stage) is a
    # byte-identical no-op. See `_apply_eval_reset`'s docstring and
    # `sculptor.reference.derive_eval_reset` for the full rationale.
    _eval_reset_arg = getattr(args, "eval_reset", "") or ""
    if _eval_reset_arg:
        try:
            _eval_reset_payload = json.loads(
                Path(_eval_reset_arg).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[runner] --eval-reset {_eval_reset_arg!r} unreadable "
                  f"({type(e).__name__}: {e}) — ignored", file=sys.stderr,
                  flush=True)
            _eval_reset_payload = None
        _apply_eval_reset(
            env_cfg, _eval_reset_payload, task_id=args.task_id)
    # §Ship 35: textured floor in the rendered rollout (cosmetic, guarded).
    _apply_ground_texture(env_cfg)
    # 720p + no ghost neighbor envs in the background (cosmetic, guarded).
    _configure_rollout_viewer(env_cfg, args)
    # Authored physics is descriptor-pinned. Legacy registered tasks repeat
    # the same compatibility transform used in training.
    if world_bundle is None:
        _enforce_actuator_limits(env_cfg)
    # §Selection statistics: deterministic eval seeding for repeat rollouts
    # of the SAME checkpoint (multi-seed evaluation / fresh-seed re-eval of
    # the kept best). --seed 0 (default) leaves the legacy RNG state
    # untouched. Reset-event randomization draws from torch's global RNG;
    # cfg.seed is additionally honored when the cfg exposes it.
    _eval_seed = int(getattr(args, "seed", 0) or 0)
    if _eval_seed:
        try:
            import torch
            torch.manual_seed(_eval_seed)
        except Exception:  # noqa: BLE001 — seeding is best-effort
            pass
        np.random.seed(_eval_seed % (2**32 - 1))
        if hasattr(env_cfg, "seed"):
            try:
                env_cfg.seed = _eval_seed
            except Exception:  # noqa: BLE001 — frozen cfg tolerated
                pass
    # §manipulation telemetry: registered (non-authored) tasks get generic
    # object/end-effector/contact/target channels discovered from the scene
    # cfg + capability descriptors — the artifact contract the YAM
    # benchmark manifests name as their first known limitation. Authored
    # runs are excluded: their ChannelCatalog recorder is the authoritative
    # channel contract and the two must never double-write. Contact-sensor
    # injection must precede env construction; every step is fail-soft.
    manip_discovery = None
    if world_bundle is None:
        try:
            from sculptor.adapters.manipulation_telemetry import (
                discover_from_cfg,
                inject_contact_sensors,
            )

            manip_discovery = discover_from_cfg(env_cfg)
            if manip_discovery is not None:
                manip_discovery = inject_contact_sensors(
                    env_cfg, manip_discovery)
        except Exception as e:  # noqa: BLE001 — telemetry must never block
            print(f"[runner] manipulation-telemetry discovery skipped: "
                  f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
            manip_discovery = None
    env = ManagerBasedRlEnv(
        env_cfg, device=args.device, render_mode="rgb_array"
    )
    _hide_untracked_authored_geometry(
        env, int(getattr(args, "render_env_index", 0) or 0))
    world_channel_recorder = None
    if world_bundle is not None:
        from sculptor.world.runtime import (
            WorldChannelRecorder,
            WorldChannelRuntime,
        )

        world_channel_recorder = WorldChannelRecorder(WorldChannelRuntime(
            env, catalog=world_bundle.channel_catalog,
            manifest=world_bundle.manifest))
    manip_recorder = None
    if manip_discovery is not None:
        try:
            from sculptor.adapters.manipulation_telemetry import (
                ManipulationRecorder,
            )

            manip_recorder = ManipulationRecorder(env, manip_discovery)
            print("[SCULPT-EVENT] " + json.dumps({
                "type": "manipulation_telemetry_discovered",
                "capability_id": manip_discovery.capability_id,
                "objects": list(manip_discovery.object_names),
                "ee_site": manip_discovery.ee_site,
                "finger_groups": sorted(manip_discovery.finger_groups),
                "grasp_capable": manip_discovery.grasp_capable,
                "contact_sensors": list(manip_discovery.sensor_names),
            }), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[runner] manipulation-telemetry recorder skipped: "
                  f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
            manip_recorder = None

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
            fr = np.asarray(
                _to_host_numpy(getattr(m, "actuator_forcerange")),
                dtype=np.float64,
            )
            jr = np.asarray(
                _to_host_numpy(getattr(m, "jnt_range")),
                dtype=np.float64,
            )
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
    default_pose_rms_buf: list[np.ndarray] = []
    action_buf: list[np.ndarray] = []
    actuator_force_buf: list[np.ndarray] = []
    # §reports: torque in JOINT space (qfrc_actuator) — aligned with joint_vel/names,
    # unlike `actuator_force` which is in actuator order. Feeds the per-motor
    # torque-vs-limit report.
    joint_torque_buf: list[np.ndarray] = []
    projected_gravity_b_buf: list[np.ndarray] = []
    root_link_pos_w_buf: list[np.ndarray] = []
    root_link_lin_vel_b_buf: list[np.ndarray] = []
    root_link_ang_vel_b_buf: list[np.ndarray] = []
    # §Metric-quality laws (LAW 3/4): per-foot ground contact + foot position
    # in the pelvis frame, persisted to the metric arrays so an objective
    # metric can measure signed forward-kick DIRECTION (anterior foot
    # displacement) and the single-vs-double support SCHEDULE (one-leg-balance
    # veto). Biped-only — stay empty (and are dropped at save time) on tasks
    # whose robot has no left_foot/right_foot site pair.
    left_foot_contact_buf: list[np.ndarray] = []
    right_foot_contact_buf: list[np.ndarray] = []
    left_foot_pos_b_buf: list[np.ndarray] = []
    right_foot_pos_b_buf: list[np.ndarray] = []
    per_term_reward_buf: dict[str, list[np.ndarray]] = {}
    # Post-step mjlab state is already reset for environments whose `done`
    # fired.  These masks let artifact finalization preserve only each env's
    # first episode instead of stitching the reset attempt onto it.
    first_episode_state_valid_buf: list[np.ndarray] = []
    first_episode_action_valid_buf: list[np.ndarray] = []

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

    # §Metric-quality laws (LAW 3/4): resolve the foot sites + ground-contact
    # sensor ONCE, mirroring SculptorRewardTerm._resolve_foot_handles. Returns
    # None unless a left_foot/right_foot site pair exists (the same named-site
    # pair that fixes per-foot column order in mjlab) → biped only. The
    # pelvis-frame foot position uses quat_apply_inverse (the same transform
    # that derives projected_gravity_b: data.py site_pos_w − root_link_pos_w
    # rotated by root_link_quat_w); import is lazy + guarded so a future
    # mjlab rename degrades to "no foot position" rather than crashing the
    # rollout.
    def _resolve_feet(e, robot):
        if robot is None:
            return None
        try:
            names = tuple(robot.site_names)
        except Exception:  # noqa: BLE001
            return None
        idx = {n: i for i, n in enumerate(names)}
        li, ri = idx.get("left_foot"), idx.get("right_foot")
        if li is None or ri is None:
            return None
        try:
            contact = e.scene["feet_ground_contact"]
        except Exception:  # noqa: BLE001
            contact = None
        return {"li": li, "ri": ri, "contact": contact}

    _feet = _resolve_feet(env, _robot)
    try:
        from mjlab.utils.lab_api.math import quat_apply_inverse as _quat_apply_inverse
    except Exception:  # noqa: BLE001
        _quat_apply_inverse = None

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
        active_before = ~ep_done
        with torch.inference_mode():
            action = policy(obs)
        obs, rew, dones, _extras = wrapped.step(action)
        dones_bool = dones.bool()
        state_valid = active_before & (~dones_bool)
        first_episode_state_valid_buf.append(
            state_valid.detach().cpu().numpy().astype(bool, copy=False))
        first_episode_action_valid_buf.append(
            active_before.detach().cpu().numpy().astype(bool, copy=False))
        if world_channel_recorder is not None:
            # Sample every catalogued task observable at the same cadence as
            # base trajectory state. Missing/malformed producers fail the
            # authored rollout instead of silently emitting a partial NPZ.
            world_channel_recorder.append()
            # The recorder owns independent NumPy route/hold state.  Keep it
            # synchronized with mjlab's automatic per-env reset even though
            # finalized first-episode arrays absorb all later samples.
            try:
                done_ids = np.flatnonzero(
                    dones_bool.detach().cpu().numpy().astype(bool, copy=False))
                if done_ids.size:
                    world_channel_recorder.runtime.reset(done_ids)
            except Exception as e:  # noqa: BLE001
                print(f"[runner] authored recorder reset skipped: "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if manip_recorder is not None:
            # mjlab auto-resets before returning from a done step, so that
            # step's scene state belongs to the next episode. Persist the
            # boundary explicitly; metrics must never stitch attempts across
            # it. `ep_done` freezes the mask after each env's first episode.
            try:
                first_done = dones_bool & (~ep_done)
                manip_recorder.append(
                    valid_mask=(~ep_done) & (~dones_bool),
                    terminal_mask=first_done,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[runner] manipulation-telemetry step skipped: "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        # Freeze cumulative return/length on the first `done` per env.
        active = active_before.float()
        ep_return += rew * active
        ep_length += active.int()
        ep_done |= dones_bool

        # The rendered lane is precommitted before rollout. Keep its scalar
        # reward series aligned with the video while the full state tensors
        # continue to cover every evaluation lane.
        all_rewards.append(float(rew[render_env_index].item()))

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
                default_jp = _tensor_to_np(
                    getattr(d, "default_joint_pos", None)
                )
                if default_jp is not None and default_jp.shape == jp.shape:
                    default_pose_rms_buf.append(np.sqrt(np.mean(
                        np.square(jp - default_jp), axis=-1,
                    )).astype(np.float32, copy=False))
            jv = _tensor_to_np(getattr(d, "joint_vel", None))
            if jv is not None:
                joint_vel_buf.append(jv)
            af = _tensor_to_np(getattr(d, "actuator_force", None))
            if af is not None:
                actuator_force_buf.append(af)
            jt = _tensor_to_np(getattr(d, "qfrc_actuator", None))   # §reports: joint-space torque
            if jt is not None:
                joint_torque_buf.append(jt)
            pg = _tensor_to_np(getattr(d, "projected_gravity_b", None))
            if pg is not None:
                projected_gravity_b_buf.append(pg)
            rp = _tensor_to_np(getattr(d, "root_link_pos_w", None))
            if rp is not None:
                root_link_pos_w_buf.append(rp)
            rlv = _tensor_to_np(getattr(d, "root_link_lin_vel_b", None))
            if rlv is not None:
                root_link_lin_vel_b_buf.append(rlv)
            rav = _tensor_to_np(getattr(d, "root_link_ang_vel_b", None))
            if rav is not None:
                root_link_ang_vel_b_buf.append(rav)
            # §Metric-quality laws (LAW 3/4): per-foot contact + pelvis-frame
            # foot position. Each signal is independently guarded so a missing
            # sensor/site degrades to "field absent" (an empty buf is dropped
            # at save time) rather than crashing the rollout — same discipline
            # as _foot_info. Contact columns 0/1 = left/right (mjlab wiring,
            # mirrored from _foot_info); foot position uses the resolved site
            # indices li/ri.
            if _feet is not None:
                fc = _feet["contact"]
                if fc is not None:
                    try:
                        found = fc.data.found  # (N, F)
                        if found is not None and found.shape[-1] >= 2:
                            lc = _tensor_to_np((found[:, 0] > 0).float())
                            rc = _tensor_to_np((found[:, 1] > 0).float())
                            if lc is not None:
                                left_foot_contact_buf.append(lc)
                            if rc is not None:
                                right_foot_contact_buf.append(rc)
                    except Exception:  # noqa: BLE001
                        pass
                if _quat_apply_inverse is not None:
                    try:
                        sp = getattr(d, "site_pos_w", None)        # (N, S, 3)
                        rq = getattr(d, "root_link_quat_w", None)  # (N, 4)
                        rpw = getattr(d, "root_link_pos_w", None)  # (N, 3)
                        li, ri = _feet["li"], _feet["ri"]
                        if (sp is not None and rq is not None and rpw is not None
                                and sp.shape[1] > max(li, ri)):
                            lf_b = _quat_apply_inverse(rq, sp[:, li, :] - rpw)
                            rf_b = _quat_apply_inverse(rq, sp[:, ri, :] - rpw)
                            lfp = _tensor_to_np(lf_b)
                            rfp = _tensor_to_np(rf_b)
                            if lfp is not None:
                                left_foot_pos_b_buf.append(lfp)
                            if rfp is not None:
                                right_foot_pos_b_buf.append(rfp)
                    except Exception:  # noqa: BLE001
                        pass
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

        # Render only while the precommitted lane's first episode is ongoing.
        # After it hits done, mujoco_warp's auto-reset warps it to a
        # fresh initial pose — continuing to record produces a glitchy
        # video where the Cartpole snaps from upright to hanging and
        # back every episode boundary (Issue F from Test 1 2026-04-22).
        # Sam's Cartpole video showed the pole "teleporting" and
        # checkered-floor flashes at the reset frames — this guard
        # stops the video at the lane's first terminal so the user sees
        # exactly one clean episode.
        if (
            step % render_every == 0
            and not bool(ep_done[render_env_index].item())
        ):
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

        tracked_done = bool(ep_done[:n_episodes].all().item())
        rendered_done = bool(ep_done[render_env_index].item())
        if tracked_done and rendered_done:
            break

    ep_returns = ep_return[:n_episodes].detach().cpu().tolist()
    ep_lengths = ep_length[:n_episodes].detach().int().cpu().tolist()
    all_first_episode_returns = ep_return.detach().cpu().tolist()

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
    state_valid_mask = np.stack(
        first_episode_state_valid_buf, axis=0).astype(bool, copy=False)
    action_valid_mask = np.stack(
        first_episode_action_valid_buf, axis=0).astype(bool, copy=False)
    trajectory["first_episode_valid_mask"] = state_valid_mask
    for key, buf in (
        ("joint_pos", joint_pos_buf),
        ("joint_vel", joint_vel_buf),
        ("default_pose_rms", default_pose_rms_buf),
        ("joint_torque", joint_torque_buf),
        ("action", action_buf),
        ("actuator_force", actuator_force_buf),
        ("projected_gravity_b", projected_gravity_b_buf),
        ("root_link_pos_w", root_link_pos_w_buf),
        ("root_link_lin_vel_b", root_link_lin_vel_b_buf),
        ("root_link_ang_vel_b", root_link_ang_vel_b_buf),
        ("left_foot_contact", left_foot_contact_buf),
        ("right_foot_contact", right_foot_contact_buf),
        ("left_foot_pos_b", left_foot_pos_b_buf),
        ("right_foot_pos_b", right_foot_pos_b_buf),
    ):
        arr = _stack_if_consistent(buf)
        if arr is not None:
            valid = action_valid_mask if key == "action" else state_valid_mask
            trajectory[key] = _freeze_invalid_first_episode_steps(arr, valid)
    for name, arrs in per_term_reward_buf.items():
        stacked = _stack_if_consistent(arrs)
        if stacked is not None:
            trajectory[f"reward_term__{name}"] = (
                _freeze_invalid_first_episode_steps(
                    stacked, action_valid_mask))
    if world_channel_recorder is not None:
        world_arrays = world_channel_recorder.finalize()
        trajectory.update({
            key: (
                _freeze_invalid_first_episode_steps(value, state_valid_mask)
                if np.asarray(value).ndim >= 2 else value
            )
            for key, value in world_arrays.items()
        })
    if manip_recorder is not None:
        try:
            manip_arrays = {
                key: value
                for key, value in manip_recorder.finalize().items()
                if key not in trajectory  # existing contract always wins
            }
            trajectory.update(manip_arrays)
            manip_recorder.write_manifest(output_dir, manip_arrays)
        except Exception as e:  # noqa: BLE001
            print(f"[runner] manipulation-telemetry finalize skipped: "
                  f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
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
        # The video lane is declared at launch, never selected post-hoc from
        # rollout outcomes. Record its identity and percentile across every
        # parallel first episode so the evidence remains transparent.
        "rendered_env_index": int(render_env_index),
        "rendered_env_index_requested": int(requested_render_env_index),
        "rendered_env_selection": "precommitted",
        "rendered_episode_return": (
            float(all_first_episode_returns[render_env_index])
            if all_first_episode_returns else None),
        "rendered_episode_percentile": (
            float(np.mean([
                r <= all_first_episode_returns[render_env_index]
                for r in all_first_episode_returns
            ]))
            if all_first_episode_returns else None),
        # §Ship 26 (E1/M1): capture settings are load-bearing for spec
        # metrics (frequency bands are in cycles/FRAME; episode-length
        # normalization needs the cap). Persisting them lets the eval
        # harness ASSERT capture parity across conditions instead of
        # silently comparing incomparables.
        "step_dt": float(getattr(env, "step_dt", 0.0) or 0.0),
        "max_episode_steps": int(max_steps),
        "rollout_num_envs": int(num_envs),
    }
    import hashlib as _behavior_hashlib

    ordered_joint_names = list(limits_snapshot.get("joint_names") or [])
    ordered_joint_names_sha256 = _behavior_hashlib.sha256(
        json.dumps(
            ordered_joint_names,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    behavior["terminal_proof_contract"] = {
        "minimum_hold_s": float(_authored_terminal_hold_s(world_bundle)),
        "minimum_hold_frames": int(round(
            _authored_terminal_hold_s(world_bundle)
            / max(float(getattr(env, "step_dt", 0.02) or 0.02), 1e-9)
        )),
        "horizontal_speed_m_s": 0.12,
        "angular_speed_rad_s": 0.5,
        "joint_speed_rms_rad_s": 1.0,
        "upright_gravity_z_max": -0.7,
        "default_pose_rms_rad": 0.6,
        "default_pose_channel": "default_pose_rms",
        "ordered_joint_names_sha256": ordered_joint_names_sha256,
        "ordered_joint_count": len(ordered_joint_names),
    }
    if world_bundle is not None:
        shaping_evidence = _reward_visible_rollout_evidence(
            trajectory, world_bundle.channel_catalog, state_valid_mask)
        if shaping_evidence:
            behavior["reward_visible_rollout_evidence"] = shaping_evidence
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

    # This event is the backend's cue to inspect/transcode the video.  Emit it
    # only after ffmpeg, trajectory, behavior, and limits files are closed;
    # emitting before encoding raced the UI clip worker against an incomplete
    # MP4 (`moov atom not found`) even though the final video was valid.
    print("[SCULPT-EVENT] " + json.dumps({
        "type": "rollout_done",
        "n_episodes": n_episodes,
        "total_steps": step + 1,
        "frames_recorded": len(frames),
        "elapsed_s": round(time.time() - t0, 1),
    }), flush=True)

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
        "--env-profile", default="",
        help=(
            "§RL_SCULPTOR_AUDIT §4.4: goal-class env alignment applied to "
            "the loaded task cfg before the env is built. '' = task "
            "defaults; 'jump' = zero velocity commands, no push events, "
            "fell_over at 120°, 10 s episodes."
        ),
    )
    p_train.add_argument(
        "--env-spec", default="",
        help=(
            "path to a per-project env spec JSON (sculptor.env_spec "
            "schema). Wins over --env-profile. shared section applies "
            "to train AND rollout; train section is train-only."
        ),
    )
    p_train.add_argument(
        "--world-selection", default="",
        help=(
            "path to a hash-verified prompt-authored selection JSON. "
            "Applies world/task semantics before legacy --env-spec."
        ),
    )
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
    p_train.add_argument(
        "--pretrained-load-role", default="actor_critic",
        choices=("actor_only", "actor_critic"),
        help=(
            "Select which compatible state dictionaries to inherit. "
            "Optimizer/iteration/RND are always reset."
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
    # §RL_SCULPTOR_AUDIT §4.4: must match the train-side spec/profile.
    p_roll.add_argument("--env-profile", default="")
    p_roll.add_argument("--env-spec", default="")
    p_roll.add_argument("--world-selection", default="")
    # §D17: stage-FIXED eval-rollout reset override (a small allowlisted
    # subset of reset keys — height/pitch/roll collapsed to a single
    # deterministic midpoint value, zero reset velocity/noise,
    # fell_over_termination popped). Applied AFTER the existing
    # shared-only --env-spec (which never carries train-only RSI ranges
    # into rollout). Absent (default) = today's behavior, byte-identical.
    p_roll.add_argument(
        "--eval-reset", default="",
        help=(
            "path to a JSON object of allowlisted eval-reset keys "
            "(sculptor.reference.derive_eval_reset output). Applied to "
            "rollout only, after --env-spec/--env-profile; never affects "
            "training."
        ),
    )
    # Video resolution. 0 = the runner default (1280x720 — measured
    # resolution-INDEPENDENT render cost on the WSL2 path: 320x240 and
    # 1280x720 both ~200 ms/frame, so high-res is free).
    p_roll.add_argument("--render-width", type=int, default=0)
    p_roll.add_argument("--render-height", type=int, default=0)
    p_roll.add_argument(
        "--render-env-index", type=int, default=0,
        help=(
            "precommitted parallel rollout lane to render; clamped to the "
            "available evaluation batch and disclosed in behavior.json"
        ),
    )
    # §Selection statistics: deterministic eval seed for repeat rollouts
    # of the same checkpoint. 0 (default) = legacy unseeded behavior.
    p_roll.add_argument("--seed", type=int, default=0)

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
