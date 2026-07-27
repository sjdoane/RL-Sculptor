"""sculptor/adapters/mjlab.py — the mjlab adapter.

mjlab (https://github.com/mujocolab/mjlab) is an Isaac-Lab-style, manager-
based RL framework on MuJoCo-Warp. GPU required. This adapter is the
primary sculptor target as of M2 of MJLAB_PIVOT_DESIGN.

Design notes (see MJLAB_PIVOT_DESIGN §1.2 for rationale):

*   **Subprocess training.** `train()` invokes a dedicated runner at
    `python -m sculptor.adapters._mjlab_runner train ...` rather than
    running in-process. mjlab + mujoco_warp + rsl_rl imports take 15-25 s
    on first CUDA-graph compilation; isolating them per-run keeps the
    UI backend and the pytest collection path fast (the lazy-import rule
    in MJLAB_PIVOT_DESIGN §7 is the counterpart constraint on the
    sculptor_bridge / project-validation endpoints, which must stay
    import-free for mjlab).

*   **Reward injection** lives inside the runner. When a sculpted reward
    module is supplied, the runner registers a class-based
    `SculptorRewardTerm` on the task's cfg and zeroes the default task
    reward terms. The reward module must export `compute_reward_batched
    (state, action, next_state, info) -> (rewards, components)` where
    every input is a dict of `(num_envs, *feature_shape)` torch tensors
    on device, and the output `rewards` is shape `(num_envs,)` on the
    same device. `cfg.scale_rewards_by_dt` is set to False so the raw
    module output survives without multiplication by `step_dt`.

*   **task_id validation** happens inside `__init__` — it imports
    `mjlab.tasks.registry` (which triggers mjlab's full import). Callers
    who must stay import-free (UI health check, project-creation
    validation) must check task_id eligibility against the robot-library
    YAML first, not by instantiating this adapter.

*   **num_envs autocap**: if detected VRAM < 12 GiB at `__init__` time
    and the caller specified `num_envs > 2048`, cap at 2048 with a
    warning. Less conservative than an OOM at iteration 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)


def _run_with_cleanup(
    cmd: list[str],
    env: dict[str, str],
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with process-group isolation + live stdout
    tee so we can kill it cleanly on exception (Ctrl-C on the parent,
    test timeout, etc.) AND the UI sees progress events as they're
    printed (not at subprocess exit).

    Why not plain `subprocess.run`: if the backend (or pytest) gets
    terminated while the mjlab subprocess is mid-training, the child
    inherits our process group. `subprocess.run` does not propagate
    SIGTERM to the child; the child keeps running with CUDA context
    locked, blocking any retry. Using `start_new_session=True` puts the
    child in its own session + pgid, then on any exception we
    `os.killpg` the whole subtree (mjlab + mujoco_warp threads +
    rsl_rl + any child python) in one shot.

    Why the stdout tee thread: the previous `.communicate()`-based
    implementation buffered the entire training subprocess's stdout
    in memory, so rsl_rl's "Learning iteration N/M" prints + our own
    `[SCULPT-EVENT] iter_progress` JSON lines never reached the outer
    sculpt-CLI stdout until the subprocess exited (i.e., per sculpt
    iter, ~25 min of silence on the UI followed by a wall of text at
    the end). We now spawn a reader thread per pipe that streams
    lines to this process's stdout/stderr in real time AND collects
    them so failure paths still get the tail.
    """
    import os
    import signal
    import sys as _sys
    import threading

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        start_new_session=True,
    )

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []

    def _tee(src, buf: list[str], sink) -> None:
        try:
            for line in src:
                sink.write(line)
                sink.flush()
                buf.append(line)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass

    t_out = threading.Thread(
        target=_tee, args=(proc.stdout, stdout_buf, _sys.stdout), daemon=True,
    )
    t_err = threading.Thread(
        target=_tee, args=(proc.stderr, stderr_buf, _sys.stderr), daemon=True,
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
        returncode = proc.returncode
    except BaseException:
        # KeyboardInterrupt, SystemExit, TimeoutExpired, or any other —
        # kill the process group before re-raising so we don't leak
        # mjlab subprocesses across a backend restart.
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait()
        except (ProcessLookupError, OSError):
            pass
        raise
    finally:
        # Let the tee threads drain any trailing output after wait().
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
    return subprocess.CompletedProcess(
        cmd, returncode,
        stdout="".join(stdout_buf),
        stderr="".join(stderr_buf),
    )


_RUNNER_MODULE = "sculptor.adapters._mjlab_runner"

# State schema per task family (MJLAB_PIVOT_DESIGN §1.4). Velocity and
# tracking share the locomotion schema; Yam manipulation extends.
_VELOCITY_STATE_SCHEMA: dict[str, tuple[int, ...]] = {
    "qpos": (18,),               # 18 for Go1 (12 joints + 6 base)
    "qvel": (18,),
    "base_lin_vel_b": (3,),
    "base_ang_vel_b": (3,),
    "projected_gravity_b": (3,),
    "actuator_force": (12,),
    "command_vel": (3,),
}
_G1_STATE_SCHEMA: dict[str, tuple[int, ...]] = {
    "qpos": (29,),               # G1: 23 joints + 6 base
    "qvel": (29,),
    "base_lin_vel_b": (3,),
    "base_ang_vel_b": (3,),
    "projected_gravity_b": (3,),
    "actuator_force": (23,),
    "command_vel": (3,),
}
_INFO_KEYS: list[str] = [
    "episode_length", "terminated", "time_outs", "step_dt",
    # Fall-detection signals — enables reward modules to implement
    # Booster-Gym-style zero-clip-on-fall (arXiv:2506.15132) without
    # reward-hacking by exploiting upward body motion during topples.
    # Claude's iter-7 diagnosis on Sam's overnight run explicitly
    # asked for these, flagging the lack as the blocker preventing
    # a non-reward-hacking jumping reward (see v7.py description in
    # unitree-go1-3/rewards/v7.py:1-28). `base_height` is the world-
    # frame Z of the root link; `fallen` is a bool tensor, True when
    # the base is inverted enough that gravity projects upward in
    # the body frame (robot is clearly not in a recoverable pose).
    # ``base_height_delta`` is measured from each environment's own reset
    # height, so motion priors can track vertical displacement on any robot,
    # terrain elevation, or platform spawn without assuming a nominal height.
    "base_height", "base_height_delta", "fallen",
    # Universal motion-quality channels.  These are embodiment-agnostic
    # reductions over the adapter's canonical action/joint tensors, so reward
    # authoring can respond to flailing without guessing simulator internals.
    # ``action_rate`` is RMS(a_t - a_{t-1}); ``joint_vel_rms`` is the
    # whole-articulation RMS joint velocity.  Both are zero on reset frames.
    "action_rate", "joint_vel_rms",
]

# §Ship 46: extra info keys surfaced for the G1 humanoid so a sculpted
# reward can shape a single-leg KICK. The g1-kick-v3 overnight run
# stalled because the reward could only see base_height/fallen — every
# kick term the diagnoser proposed (swing-foot velocity, single-leg XOR
# contact, foot clearance) had to be deferred for want of these channels
# (the deferral was correct: edit.py grounds reward formulas against
# `expected_info_keys`, and these were absent). mjlab already computes
# all of them for its own foot reward terms (feet_slip / feet_clearance
# / feet_swing_height); the runner surfaces them as (N,) scalars,
# zero-filled on tasks without the named foot sites/sensors. Per-foot
# keys are advertised ONLY for the G1 biped (which has 'left_foot' /
# 'right_foot' sites that fix the per-foot column order across the
# contact / height / site-velocity tensors); other robots keep the
# universal base contract and the runner emits zeros they never reference.
# `base_horizontal_speed` lets the diagnoser tell standing from walking
# (info previously had no base velocity, so a forward walker read as
# "standing still").
_G1_INFO_EXTRA: list[str] = [
    "left_foot_contact", "right_foot_contact",
    "left_foot_swing_speed", "right_foot_swing_speed",
    "left_foot_height", "right_foot_height",
    "base_horizontal_speed",
]


def _info_keys_for_task(task_id: str) -> list[str]:
    """Info-dict keys advertised in the reward contract for a task.
    G1 gains the per-foot kick channels (§Ship 46); all other task
    families use the universal base set."""
    if "G1" in task_id:
        return list(_INFO_KEYS) + list(_G1_INFO_EXTRA)
    return list(_INFO_KEYS)


_CARTPOLE_STATE_SCHEMA: dict[str, tuple[int, ...]] = {
    # Cartpole is a fixed-base articulation: cart-slide joint + pole-
    # hinge joint. No floating root, no actuator_force per-leg, no
    # command_vel. Exposing only qpos + qvel keeps Claude-written
    # rewards sane for this task family.
    "qpos": (2,),  # [cart_position, pole_angle]
    "qvel": (2,),
    "actuator_force": (1,),  # single actuator on the cart
}


def _schema_for_task(task_id: str) -> dict[str, tuple[int, ...]]:
    """Pick a state schema based on the task_id. Currently velocity
    / tracking for G1 and Go1-family quadrupeds + Cartpole for the
    inverted-pendulum sanity test — extend as tasks land."""
    if "G1" in task_id:
        return dict(_G1_STATE_SCHEMA)
    if "Go1" in task_id or "Go2" in task_id or "Anymal" in task_id:
        return dict(_VELOCITY_STATE_SCHEMA)
    if "Cartpole" in task_id or "cartpole" in task_id.lower():
        return dict(_CARTPOLE_STATE_SCHEMA)
    # Default — velocity schema. Manipulation tasks (Yam) will need their
    # own schema in a follow-up.
    return dict(_VELOCITY_STATE_SCHEMA)


@dataclass
class MjlabAdapter(SculptorAdapter):
    """mjlab-backed adapter. Subprocess training, GPU required.

    Config fields flow in from `[adapter].config` in config.toml.
    """

    task_id: str = "Mjlab-Velocity-Flat-Unitree-Go1"
    num_envs: int = 4096
    device: str = "cuda:0"
    max_iterations: int = 1500
    seed: int = 1
    # §RL_SCULPTOR_AUDIT §4.4: goal-class env alignment. "" (default) keeps
    # the mjlab task cfg untouched; "jump" retargets the walking-task
    # mechanics that fight a standing jump (zero velocity commands, no
    # push events, fell_over at 120°, 10 s episodes — see
    # `_mjlab_runner._apply_env_profile`). Applied to BOTH train and
    # rollout so the policy is evaluated under its training distribution.
    env_profile: str = ""
    # §RL_SCULPTOR_AUDIT (env generalization): path to a per-project env
    # spec JSON (sculptor.env_spec schema — the general successor to the
    # named profiles). Wins over `env_profile` when both are set. The
    # spec's `shared` section applies to BOTH train and rollout; its
    # `train` section (RSI resets, sunk termination, domain
    # randomization, PPO exploration) is train-only. Injected by
    # `load_adapter` when the project has `env/current.json`; the FILE's
    # content is re-read by each train/rollout subprocess, so the sculpt
    # loop can iterate the train section between iterations.
    env_spec_path: str = ""
    # Atomic prompt-authored world/task/evaluation/channel tuple. This is
    # separate from env_spec_path by design: the latter remains the legacy
    # diagnoser-managed reset/randomization/optimizer surface.
    world_selection_path: str = ""
    # §D17: path to a stage-FIXED eval-rollout reset override JSON
    # (`sculptor.reference.derive_eval_reset`'s payload, written once at
    # stage-scaffold time to `env/eval_reset.json`). Applied ONLY to
    # rollout evaluation, AFTER the existing shared-only `_apply_env_spec`
    # — a small allowlisted set of reset keys (height/pitch/roll collapsed
    # to a single deterministic value, zero reset velocity/noise,
    # fell_over_termination popped), NEVER the diagnoser-iterable
    # `env_spec_path` train section. Injected by `load_adapter` when the
    # project has `env/eval_reset.json`, same convention as
    # `env_spec_path` above. Empty (default) = today's behavior, byte-
    # identical (eval resets standing-start on every task, get-up stages
    # included).
    eval_reset_path: str = ""
    rsl_rl_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional override for the schema keys emitted by the reward-term
    # state snapshot. If empty, derived from task_id via _schema_for_task.
    schema_keys: Optional[list[str]] = None
    # §Ship 23: optional `[remote]` table (top-level in config.toml,
    # plumbed by load_adapter) — SSH dispatch of train/rollout to a
    # rented GPU. `SCULPTOR_REMOTE_*` env vars override these values
    # (the UI backend's injection path). None/disabled → fully local.
    remote: Optional[dict[str, Any]] = None

    # Populated by __post_init__.
    _validated: bool = field(default=False, init=False, repr=False)
    _remote_exec: Any = field(default=None, init=False, repr=False)
    _world_bundle: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Lazy import — keeps non-mjlab adapters and UI health check
        # import-cost-free. This import DOES incur mjlab's 15-25s cold
        # start; only pays out per MjlabAdapter instantiation.
        from importlib.metadata import version

        import torch

        try:
            from mjlab.tasks.registry import list_tasks
        except ImportError as e:
            raise ImportError(
                f"mjlab not importable (cannot validate task_id={self.task_id!r}): "
                f"{type(e).__name__}: {e}. Install with `uv add mjlab[cu128]`."
            ) from e

        registered = list_tasks()
        if self.task_id not in registered:
            raise ValueError(
                f"task_id={self.task_id!r} is not registered in mjlab; "
                f"known tasks: {sorted(registered)}"
            )

        # §RL_SCULPTOR_AUDIT §4.4: fail fast on an unknown profile here
        # (clear error at adapter construction) instead of a silent
        # no-op warning buried in the training subprocess's stderr.
        if self.env_profile not in ("", "default", "jump"):
            raise ValueError(
                f"env_profile={self.env_profile!r} is not supported; "
                "known profiles: '' (task defaults), 'jump'."
            )
        # §env generalization: fail fast on a missing/invalid env spec —
        # never spawn a GPU subprocess that would die (or half-apply)
        # under a bad spec. Content is validated again at each
        # subprocess spawn (the file is re-read so the loop can iterate
        # it); this init check catches config errors at load time.
        if self.env_spec_path:
            from sculptor.env_spec import load_env_spec

            load_env_spec(self.env_spec_path)  # raises ValueError
            # Pin the path NOW — a relative path resolved again at
            # spawn time (after a cwd change) could validate one file
            # and hand the subprocess another.
            self.env_spec_path = str(Path(self.env_spec_path).resolve())

        if self.world_selection_path:
            from sculptor.world.artifacts import WorldArtifactStore
            from sculptor.world.task_spec import validate_task_spec
            from sculptor.world.world_spec import validate_world_spec

            selection_path = Path(self.world_selection_path).resolve()
            store = WorldArtifactStore(selection_path.parent.parent)
            selection = store.read_selection(selection_path)
            if selection is None:  # defensive: explicit path must exist
                raise ValueError(
                    f"world_selection_path not found: {selection_path}")
            world = store.load_json_ref(selection.refs["world"])
            task = store.load_json_ref(selection.refs["task"])
            world_errors = validate_world_spec(world)
            task_errors = validate_task_spec(task, world=world)
            if world_errors or task_errors:
                details = "; ".join(world_errors + task_errors)
                raise ValueError(
                    f"world selection {selection.tuple_hash[:12]} invalid: "
                    f"{details}")
            self._world_bundle = {
                "selection": selection.to_dict(),
                "world": world,
                "task": task,
                "resolved_eval": store.load_json_ref(
                    selection.refs["resolved_eval"]),
                "channel_catalog": store.load_json_ref(
                    selection.refs["channel_catalog"]),
            }
            self.world_selection_path = str(selection_path)

        # §D17: fail fast on a missing/invalid eval-reset override — same
        # discipline as env_spec_path above. This file is a plain JSON
        # dict of allowlisted reset keys (not an env-spec document), so
        # validation here is just "readable, parses as a JSON object";
        # the runner validates individual key shapes when it applies them.
        if self.eval_reset_path:
            p = Path(self.eval_reset_path)
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"eval_reset_path unreadable at "
                    f"{self.eval_reset_path!r}: {type(e).__name__}: {e}"
                ) from e
            if not isinstance(payload, dict):
                raise ValueError(
                    f"eval_reset_path={self.eval_reset_path!r} must "
                    f"contain a JSON object, got {type(payload).__name__}"
                )
            # Pin the path NOW — same rationale as env_spec_path (a
            # relative path resolved again at spawn time, after a cwd
            # change, could validate one file and hand the subprocess
            # another).
            self.eval_reset_path = str(p.resolve())

        # num_envs autocap on smaller VRAM. Skipped when remote dispatch
        # is enabled — the local VRAM probe measures the wrong GPU (the
        # rented pod has its own, almost always larger, card).
        if (
            not self._remote_enabled()
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            idx = 0
            try:
                idx = int(self.device.split(":")[1]) if ":" in self.device else 0
            except (ValueError, IndexError):
                idx = 0
            free, total = torch.cuda.mem_get_info(idx)
            total_gib = total / (1024 ** 3)
            if total_gib < 12.0 and self.num_envs > 2048:
                import warnings
                warnings.warn(
                    f"MjlabAdapter auto-capping num_envs {self.num_envs} -> 2048 "
                    f"on {total_gib:.1f} GiB GPU (< 12 GiB threshold).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.num_envs = 2048

        self._mjlab_version = version("mjlab")
        self._validated = True

    # ── Remote dispatch (§Ship 23) ──────────────────────────────────────────
    def _remote_config(self):
        """Resolve the effective RemoteConfig (TOML `[remote]` table +
        `SCULPTOR_REMOTE_*` env overrides; env wins). None when nothing
        is configured — the common local case."""
        from sculptor.adapters._remote import RemoteConfig

        return RemoteConfig.from_sources(self.remote, os.environ)

    def _remote_enabled(self) -> bool:
        rcfg = self._remote_config()
        return rcfg is not None and rcfg.enabled

    def _remote_executor(self):
        """Lazily build (and cache) the RemoteExecutor when remote
        dispatch is enabled; None otherwise."""
        rcfg = self._remote_config()
        if rcfg is None or not rcfg.enabled:
            return None
        if self._remote_exec is None or self._remote_exec.cfg != rcfg:
            from sculptor.adapters._remote import RemoteExecutor

            self._remote_exec = RemoteExecutor(rcfg)
        return self._remote_exec

    @staticmethod
    def _remote_device_env(device: str) -> tuple[dict[str, str], str]:
        """Map a remote device selection onto (env, runner_device).

        §Ship 31b (multi-GPU pods): `CUDA_VISIBLE_DEVICES=N` MASKS the
        GPU set — inside the runner the selected card is always
        `cuda:0`. Passing `--device cuda:N` alongside the mask raised
        "invalid device ordinal" for N>0 (latent on single-GPU pods,
        load-bearing on the campaign's 3× PRO 6000 host). The physical
        index lives ONLY in the env mask; the runner argv always says
        cuda:0.

        §Ship 32a: remote hosts are headless — MuJoCo's offscreen
        renderer needs `MUJOCO_GL=egl` or rollout dies with "an OpenGL
        platform library has not been loaded" (caught live: campaign
        first jobs, 2026-06-11; the smoke had rollouts local so the
        path was never exercised). Inert for train (no GL context is
        ever created). Provisioning installs the glvnd front-end
        (libegl1) the EGL path needs."""
        env = {"MUJOCO_GL": "egl"}
        if device.startswith("cuda") and ":" in device:
            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
            return env, "cuda:0"
        return env, device

    # ── Contract ────────────────────────────────────────────────────────────
    def reward_contract(self) -> RewardContract:
        state_schema = _schema_for_task(self.task_id)
        info_keys = _info_keys_for_task(self.task_id)
        info_schema: dict[str, tuple[int, ...]] = {
            key: () for key in info_keys
        }
        channel_catalog = None
        if self._world_bundle is not None:
            from sculptor.world.capabilities import resolve_robot_capability

            robot = self._world_bundle["world"]["shared"]["robot"]
            cap = resolve_robot_capability(
                robot["capability_id"],
                required=robot.get("required_capabilities", []),
                extra_paths=([robot["descriptor_path"]]
                             if robot.get("descriptor_path") else []),
            )
            # Authored projects are descriptor-driven. Legacy projects retain
            # their existing task-family compatibility mapping.
            if cap.reward_state_schema:
                state_schema = dict(cap.reward_state_schema)
            channel_catalog = dict(self._world_bundle["channel_catalog"])
            shared_names = [
                str(channel["name"])
                for channel in channel_catalog.get("channels", [])
                if channel.get("access") == "shared_shaping"
            ]
            info_keys = list(dict.fromkeys(
                list(_INFO_KEYS) + list(cap.reward_info_keys) + shared_names))
            info_schema = {key: () for key in info_keys}
            for channel in channel_catalog.get("channels", []):
                if channel.get("access") != "shared_shaping":
                    continue
                name = str(channel.get("name", ""))
                raw_shape = channel.get("shape", [])
                if not name or not isinstance(raw_shape, list):
                    continue
                # Catalog trajectories are (T, N, *feature). The reward
                # runtime exposes one simulator step, hence (N, *feature).
                trailing = raw_shape[2:]
                if all(
                    isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
                    for dim in trailing
                ):
                    info_schema[name] = tuple(trailing)
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=info_keys,
            expected_components=None,
            supports_batched=True,
            training_device="gpu",
            # Conservative: 6 GiB for 4096-env G1 humanoid; smaller tasks
            # (Go1 / Anymal quadrupeds) fit in less. Per-task-class
            # override could live in a follow-up when VRAM probe data is
            # in hand (MJLAB_PIVOT_DESIGN §3.3).
            min_gpu_memory_gb=6.0 if "G1" in self.task_id else 4.0,
            state_schema=state_schema,
            info_schema=info_schema,
            channel_catalog=channel_catalog,
        )

    # ── Component probe (scalar, subprocess-isolated, mjlab-shaped) ─────────
    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        """Scalar probe for the UI and edit.py pre-flight.

        Constructs zero torch tensors matching `state_schema` and calls
        `compute_reward(state, action, next_state, info)`. Single env
        (leading dim 1), not batched — batched probing is what
        `reward_batched` does during edit post-flight.
        """
        import textwrap

        contract = self.reward_contract()
        schema = contract.state_schema or _schema_for_task(self.task_id)
        action_dim = schema.get("actuator_force", (12,))[0]
        path = Path(reward_module_path).resolve()

        script = textwrap.dedent(
            """\
            import importlib.util, json, sys
            import torch
            path, schema_json, action_dim, info_keys_json, info_schema_json = sys.argv[1:6]
            schema = json.loads(schema_json)
            info_keys = json.loads(info_keys_json)
            info_schema = json.loads(info_schema_json)
            spec = importlib.util.spec_from_file_location('_probe', path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                state = {k: torch.zeros((1, *shape), dtype=torch.float32)
                         for k, shape in schema.items()}
                next_state = {k: torch.zeros((1, *shape), dtype=torch.float32)
                              for k, shape in schema.items()}
                action = torch.zeros((1, int(action_dim)), dtype=torch.float32)
                info = {
                    k: torch.zeros((1, *info_schema.get(k, [])), dtype=torch.float32)
                    for k in info_keys
                }
                out = mod.compute_reward(state, action, next_state, info)
                if not (isinstance(out, tuple) and len(out) == 2):
                    raise TypeError(f'compute_reward must return (reward, components); got {type(out).__name__}')
                reward, components = out
                if not isinstance(components, dict):
                    raise TypeError('components must be a dict')
                # Accept tensor or scalar reward for the scalar probe;
                # coerce to a python float via .item() or float().
                if hasattr(reward, 'item'):
                    total = float(reward.item()) if reward.ndim == 0 else float(reward.flatten()[0].item())
                else:
                    total = float(reward)
                comp_out = {}
                for k, v in components.items():
                    if hasattr(v, 'item'):
                        comp_out[str(k)] = float(v.item()) if v.ndim == 0 else float(v.flatten()[0].item())
                    else:
                        comp_out[str(k)] = float(v)
                print(json.dumps({'ok': True, 'components': comp_out, 'total': total}))
            except Exception as e:
                print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
                sys.exit(0)
            """
        )

        schema_json = json.dumps({k: list(v) for k, v in schema.items()})
        info_keys_json = json.dumps(contract.expected_info_keys)
        info_schema_json = json.dumps({
            key: list(shape)
            for key, shape in (contract.info_schema or {}).items()
        })

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path), schema_json, str(action_dim), info_keys_json,
                    info_schema_json,
                ],
                capture_output=True, text=True, timeout=30.0,
            )
        except subprocess.TimeoutExpired:
            return ComponentProbe(ok=False, error="subprocess timeout (30s)")
        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0 and not stdout:
            stderr = (proc.stderr or "").strip()[-500:]
            return ComponentProbe(ok=False, error=f"subprocess exit {proc.returncode}: {stderr}")
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except (ValueError, IndexError) as e:
            return ComponentProbe(ok=False, error=f"probe stdout not JSON: {e}")
        if not payload.get("ok"):
            return ComponentProbe(ok=False, error=str(payload.get("error") or "unknown error"))
        return ComponentProbe(
            ok=True,
            components=payload.get("components") or {},
            total=float(payload.get("total") or 0.0),
        )

    # ── Batched reward path ─────────────────────────────────────────────────
    def reward_batched(
        self,
        reward_module_path: Path,
        state_batch: Any,
        action_batch: Any,
        next_state_batch: Any,
        info_batch: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Dispatch to the reward module's compute_reward_batched.

        If the module doesn't define compute_reward_batched, fall back to
        the default loop in SculptorAdapter with a runtime warning. For
        mjlab projects this indicates a reward-contract violation and
        should be caught by edit.py post-flight.
        """
        from sculptor.adapters.base import _import_reward_module

        mod = _import_reward_module(reward_module_path)
        if hasattr(mod, "compute_reward_batched"):
            return mod.compute_reward_batched(
                state_batch, action_batch, next_state_batch, info_batch
            )
        return super().reward_batched(
            reward_module_path, state_batch, action_batch,
            next_state_batch, info_batch,
        )

    @staticmethod
    def _load_reward_spec(reward_module_path: Path) -> dict[str, Any]:
        """Read the reward module's `REWARD_SPEC` dict in-process.
        Used by `train` to drop `reward_spec.json` next to the
        checkpoint so downstream diagnose has the reward metadata.
        """
        from sculptor.adapters.base import _import_reward_module

        mod = _import_reward_module(reward_module_path)
        spec = getattr(mod, "REWARD_SPEC", {})
        if not isinstance(spec, dict):
            return {}
        return spec

    # ── Train ───────────────────────────────────────────────────────────────
    def train(
        self,
        reward_module_path: Path,
        output_dir: Path,
        steps: int,
        seed: int,
        *,
        init_policy_path: Optional[Path] = None,
    ) -> TrainResult:
        """Subprocess-train. `steps` is interpreted as `max_iterations`
        for rsl_rl's OnPolicyRunner (one iteration = num_envs *
        num_steps_per_env policy rollouts = ~num_envs * episode_length
        env steps). Caller budgets accordingly.

        §Ship 15: `init_policy_path` is an optional path to a prior
        rsl_rl checkpoint. When set, the runner loads actor+critic
        weights from that checkpoint before training begins. Used by
        the mission orchestrator (Ship 16) to warm-start a new
        stage's training from a previous stage's final policy. The
        source checkpoint MUST be from a compatible task_id — obs and
        action spaces must match or `runner.load` raises.
        """
        reward_module_path = Path(reward_module_path).resolve() if reward_module_path else None  # type: ignore[assignment]
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optionally use steps as max_iterations if the caller overrides.
        max_iterations = int(steps) if steps else self.max_iterations

        env = dict(os.environ)
        # Device pinning without setting any display env vars; mjlab
        # trains headless, we just constrain GPU visibility.
        if self.device.startswith("cuda") and ":" in self.device:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[1]
        # Disable wandb autologin — MjlabOnPolicyRunner calls wandb.init
        # by default and prompts for login when no API key is configured.
        # Sculptor training should be self-contained; users wanting wandb
        # can opt-in via WANDB_MODE=online + WANDB_API_KEY externally.
        env.setdefault("WANDB_MODE", "disabled")

        cmd = [
            sys.executable, "-m", _RUNNER_MODULE, "train",
            "--task-id", self.task_id,
            "--num-envs", str(self.num_envs),
            "--max-iterations", str(max_iterations),
            "--seed", str(int(seed)),
            "--device", self.device,
            "--output-dir", str(output_dir),
        ]
        if reward_module_path is not None:
            cmd += ["--reward-module-path", str(reward_module_path)]
        # §Ship 15: warm-start flag — validated before spawning the
        # subprocess so the user sees a clear error instead of a
        # cryptic subprocess failure buried in stderr.
        if init_policy_path is not None:
            init = Path(init_policy_path).resolve()
            if not init.is_file():
                raise FileNotFoundError(
                    f"init_policy_path not found: {init}"
                )
            cmd += ["--load-pretrained-policy", str(init)]
        # Always pass the per-task schema keys to the subprocess so
        # SculptorRewardTerm uses the correct key set (bug: previously
        # only passed when `self.schema_keys` was explicitly set, so the
        # runner defaulted to the 7-key velocity schema regardless of
        # task_id — caused Cartpole's reward term to store
        # `self._prev["command_vel"] = None` because Cartpole has no
        # `base_velocity` command, then crash on reset()). `schema_keys`
        # override from the user still wins if provided.
        effective_schema_keys = (
            list(self.schema_keys) if self.schema_keys
            else list((self.reward_contract().state_schema
                       or _schema_for_task(self.task_id)).keys())
        )
        cmd += ["--schema-keys", ",".join(effective_schema_keys)]
        if self.env_spec_path:
            cmd += ["--env-spec", str(Path(self.env_spec_path).resolve())]
        elif self.env_profile:
            cmd += ["--env-profile", self.env_profile]
        if self.world_selection_path:
            cmd += [
                "--world-selection",
                str(Path(self.world_selection_path).resolve()),
            ]

        executor = self._remote_executor()
        if executor is not None:
            # §Ship 23: dispatch the runner to the rented GPU. The
            # executor syncs artifacts back into the same local
            # `output_dir` (checkpoint.pt promoted LAST), so everything
            # below — post-checks, metrics, reward_spec.json, resume —
            # runs unchanged.
            from sculptor.adapters._remote import RunnerJob

            device = executor.cfg.device or self.device
            remote_env, runner_device = self._remote_device_env(device)
            options = {
                "--task-id": self.task_id,
                "--num-envs": str(self.num_envs),
                "--max-iterations": str(max_iterations),
                "--seed": str(int(seed)),
                "--device": runner_device,
                "--schema-keys": ",".join(effective_schema_keys),
            }
            if not self.env_spec_path and self.env_profile:
                options["--env-profile"] = self.env_profile
            input_paths: dict[str, Path] = {}
            aux_dir_list: list[Path] = []
            if self.env_spec_path:
                # File input → synced to the pod at its mirror path.
                input_paths["--env-spec"] = Path(self.env_spec_path).resolve()
            if self.world_selection_path:
                selection_path = Path(self.world_selection_path).resolve()
                input_paths["--world-selection"] = selection_path
                # Selection refs and materialized terrain assets are relative
                # to env/. Mirror the complete immutable artifact directory.
                aux_dir_list.append(selection_path.parent)
            if reward_module_path is not None:
                input_paths["--reward-module-path"] = Path(reward_module_path)
                # sculpt passes rewards/current.py — a shim that loads
                # its sibling v<N>.py at import time, so the whole
                # rewards/ dir must exist at its mirror path on the pod.
                aux_dir_list.append(Path(reward_module_path).resolve().parent)
            if init_policy_path is not None:
                input_paths["--load-pretrained-policy"] = Path(init_policy_path).resolve()
            job = RunnerJob(
                subcommand="train",
                options=options,
                input_paths=input_paths,
                output_dir=output_dir,
                # Ordered: completion key (checkpoint.pt — also
                # sculpt.py's resume key) is promoted last.
                required_artifacts=("metrics.json", "checkpoint.pt"),
                remote_env=remote_env,
                aux_dirs=tuple(dict.fromkeys(aux_dir_list)),
            )
            proc = executor.execute(job)
        else:
            proc = _run_with_cleanup(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mjlab runner exited {proc.returncode}\n"
                f"stdout: {(proc.stdout or '')[-2000:]}\n"
                f"stderr: {(proc.stderr or '')[-2000:]}"
            )

        ckpt_path = output_dir / "checkpoint.pt"
        metrics_path = output_dir / "metrics.json"
        if not ckpt_path.exists():
            raise RuntimeError(
                f"mjlab runner did not produce {ckpt_path}\n"
                f"stdout: {(proc.stdout or '')[-1000:]}\n"
                f"stderr: {(proc.stderr or '')[-1000:]}"
            )

        metrics: dict[str, float] = {
            "max_iterations": float(max_iterations),
            "num_envs": float(self.num_envs),
        }
        try:
            payload = json.loads(metrics_path.read_text())
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        except Exception:
            pass

        # Drop `reward_spec.json` so diagnose has the reward's metadata.
        # gym_sb3 writes this via `env_method("get_reward_spec")`; mjlab
        # injects via a subprocess so we load the reward module
        # in-process (read-only) and dump its REWARD_SPEC. Missing this
        # file made diagnose hit Claude with an empty REWARD_SPEC block,
        # weakening the failure-mode analysis.
        if reward_module_path is not None:
            try:
                reward_spec = self._load_reward_spec(Path(reward_module_path))
                (output_dir / "reward_spec.json").write_text(
                    json.dumps(reward_spec, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "mjlab.train: could not write reward_spec.json: %s: %s",
                    type(e).__name__, e,
                )

        return TrainResult(
            checkpoint_path=ckpt_path,
            metrics_dict=metrics,
            component_means={},
            logs_path=output_dir / "logs",
        )

    # ── Rollout ─────────────────────────────────────────────────────────────
    def rollout(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        n_episodes: int,
        *,
        max_episode_steps: int | None = None,
        playback_speed: float | None = None,
        render_every: int | None = None,
        fps: float | None = None,
        render_width: int | None = None,
        render_height: int | None = None,
        render_env_index: int | None = None,
        seed: int | None = None,
    ) -> RolloutResult:
        """§Ship-7: accept rollout-video knobs so the UI can drive them
        without config-file edits.

          * `max_episode_steps` — env steps per rollout (default 500
            matches the runner default). Longer = more episode visible
            but more memory + time.
          * `playback_speed` — video speed multiplier (1.0 = real-time).
          * `render_every` — capture every N-th step; 0/None = auto-cap.
          * `fps` — hard override on playback fps; 0/None = derive from
            env.step_dt * render_every / playback_speed.
          * `render_env_index` — precommitted parallel rollout lane shown
            in the video. Batch metrics still cover every evaluation lane.
          * `seed` — §Selection statistics: deterministic eval seed so
            repeat rollouts of the SAME checkpoint (multi-seed eval,
            fresh-seed re-eval) sample distinct, reproducible resets.
            None = legacy unseeded behavior.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        if self.device.startswith("cuda") and ":" in self.device:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[1]
        env.setdefault("WANDB_MODE", "disabled")

        cmd = [
            sys.executable, "-m", _RUNNER_MODULE, "rollout",
            "--task-id", self.task_id,
            "--checkpoint-path", str(checkpoint_path),
            "--output-dir", str(output_dir),
            "--n-episodes", str(int(n_episodes)),
            "--device", self.device,
        ]
        if max_episode_steps is not None:
            cmd += ["--max-episode-steps", str(int(max_episode_steps))]
        if playback_speed is not None:
            cmd += ["--playback-speed", str(float(playback_speed))]
        if render_every is not None:
            cmd += ["--render-every", str(int(render_every))]
        if fps is not None:
            cmd += ["--fps", str(float(fps))]
        if render_width is not None:
            cmd += ["--render-width", str(int(render_width))]
        if render_height is not None:
            cmd += ["--render-height", str(int(render_height))]
        if render_env_index is not None:
            cmd += ["--render-env-index", str(int(render_env_index))]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        if self.env_spec_path:
            cmd += ["--env-spec", str(Path(self.env_spec_path).resolve())]
        elif self.env_profile:
            cmd += ["--env-profile", self.env_profile]
        if self.eval_reset_path:
            cmd += ["--eval-reset", str(Path(self.eval_reset_path).resolve())]
        if self.world_selection_path:
            cmd += [
                "--world-selection",
                str(Path(self.world_selection_path).resolve()),
            ]

        executor = self._remote_executor()
        if executor is not None and executor.cfg.rollout_remote:
            # §Ship 23: remote rollout is opt-in (`rollout_remote=true`)
            # — rollouts are short, and keeping them local preserves
            # video preview when the pod is flaky.
            from sculptor.adapters._remote import RunnerJob

            device = executor.cfg.device or self.device
            remote_env, runner_device = self._remote_device_env(device)
            options = {
                "--task-id": self.task_id,
                "--n-episodes": str(int(n_episodes)),
                "--device": runner_device,
            }
            if max_episode_steps is not None:
                options["--max-episode-steps"] = str(int(max_episode_steps))
            if playback_speed is not None:
                options["--playback-speed"] = str(float(playback_speed))
            if render_every is not None:
                options["--render-every"] = str(int(render_every))
            if fps is not None:
                options["--fps"] = str(float(fps))
            if render_width is not None:
                options["--render-width"] = str(int(render_width))
            if render_height is not None:
                options["--render-height"] = str(int(render_height))
            if render_env_index is not None:
                options["--render-env-index"] = str(int(render_env_index))
            if seed is not None:
                options["--seed"] = str(int(seed))
            if not self.env_spec_path and self.env_profile:
                options["--env-profile"] = self.env_profile
            rollout_inputs: dict[str, Path] = {
                "--checkpoint-path": checkpoint_path}
            if self.env_spec_path:
                rollout_inputs["--env-spec"] = Path(self.env_spec_path).resolve()
            if self.eval_reset_path:
                rollout_inputs["--eval-reset"] = Path(self.eval_reset_path).resolve()
            rollout_aux_dirs: tuple[Path, ...] = ()
            if self.world_selection_path:
                selection_path = Path(self.world_selection_path).resolve()
                rollout_inputs["--world-selection"] = selection_path
                rollout_aux_dirs = (selection_path.parent,)
            job = RunnerJob(
                subcommand="rollout",
                options=options,
                input_paths=rollout_inputs,
                output_dir=output_dir,
                # Ordered: rollout.mp4 last — sculpt.py's rollout-skip
                # check requires all three non-empty, so a partial sync
                # can never present as a finished rollout.
                required_artifacts=("behavior.json", "trajectory.npz", "rollout.mp4"),
                remote_env=remote_env,
                aux_dirs=rollout_aux_dirs,
            )
            proc = executor.execute(job)
        else:
            proc = _run_with_cleanup(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mjlab rollout runner exited {proc.returncode}\n"
                f"stdout: {(proc.stdout or '')[-1500:]}\n"
                f"stderr: {(proc.stderr or '')[-1500:]}"
            )

        return RolloutResult(
            video_path=output_dir / "rollout.mp4",
            keyframes_dir=output_dir / "keyframes",
            trajectory_path=output_dir / "trajectory.npz",
            n_episodes=n_episodes,
        )

    # ── Reward replay (edit anti-collapse screen) ───────────────────────────
    # Cap on replayed frames — one batched compute_reward_batched call on
    # CPU; 4096 frames × ~30 floats is milliseconds and plenty of episode
    # coverage. Deterministic (evenly-spaced) subsample, never RNG.
    _REPLAY_MAX_FRAMES = 4096

    def build_reward_replay(
        self, rollout_dir: Path,
    ) -> "tuple[Any, Any, Any, dict] | None":
        """§RL_SCULPTOR_AUDIT §4.4 (edit quality): reconstruct reward
        inputs from `trajectory.npz` so edit.py can replay a CANDIDATE
        reward over the archived behavior of the current policy.

        Fidelity notes (vs the live `SculptorRewardTerm.__call__`):
          * exact — qpos/qvel/actuator_force/projected_gravity_b/action,
            info base_height, fallen, per-foot contacts;
          * approximated — foot heights (root_z + pelvis-frame foot z;
            exact only when upright), swing speeds + base_horizontal_speed
            (finite-difference over step_dt), base_lin_vel_b (world-frame
            finite difference; body≈world when upright);
          * zero-filled — base_ang_vel_b, command_vel (the jump profile
            zeroes commands anyway), terminated/time_outs.
        Good enough for the screen's question — "does this reward make
        the current behavior net-negative / zero-credit?" — which is
        dominated by the exact channels. Returns None when the archive
        lacks the core arrays (screen silently skipped)."""
        import numpy as np
        import torch

        rollout_dir = Path(rollout_dir)
        npz_path = rollout_dir / "trajectory.npz"
        if not npz_path.is_file():
            return None
        try:
            z = np.load(npz_path)
        except Exception:  # noqa: BLE001 — unreadable archive → no screen
            return None
        files = set(z.files)
        core = {"joint_pos", "joint_vel", "root_link_pos_w",
                "projected_gravity_b", "action"}
        if not core.issubset(files):
            return None
        jp = z["joint_pos"]
        if jp.ndim != 3 or jp.shape[0] < 3:
            return None
        T, N = jp.shape[:2]

        step_dt = 0.02
        try:
            behavior = json.loads((rollout_dir / "behavior.json").read_text())
            step_dt = float(behavior.get("step_dt") or 0.02) or 0.02
        except Exception:  # noqa: BLE001 — default 50 Hz
            pass

        def _np(key: str) -> "np.ndarray | None":
            return z[key] if key in files else None

        root = z["root_link_pos_w"].astype(np.float32)      # (T, N, 3)
        pg = z["projected_gravity_b"].astype(np.float32)    # (T, N, 3)
        jv = z["joint_vel"].astype(np.float32)
        act = z["action"].astype(np.float32)
        af = _np("actuator_force")
        lfc, rfc = _np("left_foot_contact"), _np("right_foot_contact")
        lfp, rfp = _np("left_foot_pos_b"), _np("right_foot_pos_b")

        # Transitions t -> t+1; frames beyond the cap are subsampled
        # evenly so the whole episode (launch, apex, landing, aftermath)
        # stays represented.
        n_trans = T - 1
        flat_total = n_trans * N
        n_keep = min(flat_total, self._REPLAY_MAX_FRAMES)
        flat_idx = np.linspace(0, flat_total - 1, num=n_keep).astype(np.int64)
        t_idx, env_idx = flat_idx // N, flat_idx % N

        def _pick(arr: "np.ndarray", ts: "np.ndarray") -> "np.ndarray":
            return arr[ts, env_idx]

        # World-frame velocities by finite difference over the transition.
        root_t, root_t1 = _pick(root, t_idx), _pick(root, t_idx + 1)
        root_vel = (root_t1 - root_t) / step_dt                    # (F, 3)

        def _foot_world_z(fp: "np.ndarray | None", ts: "np.ndarray") -> "np.ndarray":
            if fp is None:
                return np.zeros(len(ts), dtype=np.float32)
            return np.maximum(
                0.0, _pick(root, ts)[:, 2] + _pick(fp, ts)[:, 2])

        def _foot_speed(fp: "np.ndarray | None") -> "np.ndarray":
            if fp is None:
                return np.zeros(n_keep, dtype=np.float32)
            p0 = _pick(root, t_idx) + _pick(fp, t_idx)
            p1 = _pick(root, t_idx + 1) + _pick(fp, t_idx + 1)
            return np.linalg.norm((p1 - p0) / step_dt, axis=-1)

        schema = _schema_for_task(self.task_id)

        def _t(arr: "np.ndarray") -> "torch.Tensor":
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

        def _state_at(ts: "np.ndarray") -> dict[str, "torch.Tensor"]:
            out: dict[str, torch.Tensor] = {}
            for key, shape in schema.items():
                if key == "qpos":
                    out[key] = _t(_pick(jp, ts))
                elif key == "qvel":
                    out[key] = _t(_pick(jv, ts))
                elif key == "actuator_force" and af is not None:
                    out[key] = _t(_pick(af, ts))
                elif key == "projected_gravity_b":
                    out[key] = _t(_pick(pg, ts))
                elif key == "base_lin_vel_b":
                    out[key] = _t(root_vel)
                else:
                    out[key] = torch.zeros((n_keep, *shape), dtype=torch.float32)
            return out

        state = _state_at(t_idx)
        next_state = _state_at(t_idx + 1)
        action = _t(_pick(act, t_idx))

        pg_t1 = _pick(pg, t_idx + 1)
        info: dict[str, torch.Tensor] = {
            "episode_length": _t(t_idx.astype(np.float32)),
            "terminated": torch.zeros(n_keep, dtype=torch.float32),
            "time_outs": torch.zeros(n_keep, dtype=torch.float32),
            "step_dt": torch.full((n_keep,), float(step_dt)),
            "base_height": _t(root_t1[:, 2]),
            "fallen": _t((pg_t1[:, 2] >= 0.0).astype(np.float32)),
        }
        if "G1" in self.task_id:
            info.update({
                "left_foot_contact": _t(
                    _pick(lfc, t_idx + 1) if lfc is not None
                    else np.zeros(n_keep, dtype=np.float32)),
                "right_foot_contact": _t(
                    _pick(rfc, t_idx + 1) if rfc is not None
                    else np.zeros(n_keep, dtype=np.float32)),
                "left_foot_height": _t(_foot_world_z(lfp, t_idx + 1)),
                "right_foot_height": _t(_foot_world_z(rfp, t_idx + 1)),
                "left_foot_swing_speed": _t(_foot_speed(lfp)),
                "right_foot_swing_speed": _t(_foot_speed(rfp)),
                "base_horizontal_speed": _t(
                    np.linalg.norm(root_vel[:, :2], axis=-1)),
            })
        return state, action, next_state, info

    # ── Behavior metrics ────────────────────────────────────────────────────
    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict[str, Any]:
        """Read behavior.json written by the rollout runner."""
        behavior_path = rollout.video_path.parent / "behavior.json"
        if not behavior_path.exists():
            return {
                "mean_return": 0.0,
                "mean_episode_length": 0,
                "n_episodes": rollout.n_episodes,
                "adapter_status": "rollout_artifacts_missing",
            }
        payload = json.loads(behavior_path.read_text())
        return {
            "mean_return": float(payload.get("mean_return", 0.0)),
            "mean_episode_length": float(payload.get("mean_episode_length", 0.0)),
            "max_episode_length": int(payload.get("max_episode_length", 0)),
            "n_episodes": int(payload.get("n_episodes", rollout.n_episodes)),
        }


# ── Module-level helpers for backend validation / UI VRAM probe ─────────────

def measure_vram_coefficient(
    task_id: str,
    num_envs: int = 64,
    device: str = "cuda:0",
    cache_file: Path | None = None,
) -> dict[str, Any]:
    """Run the subprocess VRAM probe. Returns the JSON payload the runner
    emits: { ok, task_id, probe_num_envs, per_env_bytes,
    coefficient_bytes_per_env, free_bytes, total_bytes, ... } or
    { ok=False, error }.

    If `cache_file` is provided and already contains a payload whose
    `_cache_key == "<task_id>|<mjlab_version>"`, returns the cached
    result without launching a subprocess. After a successful probe the
    result is persisted to `cache_file`. See MJLAB_PIVOT_DESIGN §3.3.
    """
    # Cache hit check.
    if cache_file is not None:
        cache_file = Path(cache_file)
        if cache_file.is_file():
            try:
                from importlib.metadata import version as _pkg_version
                cache_key = f"{task_id}|{_pkg_version('mjlab')}"
                cached = json.loads(cache_file.read_text())
                if cached.get("_cache_key") == cache_key:
                    return cached
            except Exception:  # noqa: BLE001
                pass

    cmd = [
        sys.executable, "-m", _RUNNER_MODULE, "vram-probe",
        "--task-id", task_id,
        "--num-envs", str(int(num_envs)),
        "--device", device,
    ]
    env = {**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled")}
    proc = _run_with_cleanup(cmd, env=env)
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 and not stdout:
        return {
            "ok": False,
            "error": (
                f"vram-probe exit {proc.returncode}: "
                f"{(proc.stderr or '')[-300:]}"
            ),
        }
    try:
        result = json.loads(stdout.splitlines()[-1])
    except (ValueError, IndexError) as e:
        return {"ok": False, "error": f"probe stdout not JSON: {e}"}

    # Optional cache (MJLAB_PIVOT_DESIGN §3.3). Key = (task_id, mjlab version).
    if cache_file is not None and result.get("ok"):
        try:
            from importlib.metadata import version as _pkg_version
            cache_key = f"{task_id}|{_pkg_version('mjlab')}"
            cached = {**result, "_cache_key": cache_key}
            cache_file = Path(cache_file)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached, indent=2))
        except Exception:  # noqa: BLE001
            pass
    return result


def estimate_vram_static(num_envs: int, policy_params_millions: float = 1.5) -> float:
    """Pre-creation VRAM budget estimate in GiB (MJLAB_PIVOT_DESIGN §9.2).

    Formula: `1.5 GiB + 0.5 MB per env` for mjlab's default policy,
    independent of task. Conservative upper bound used by the UI's
    validation endpoint when no measured coefficient is available.
    """
    per_env_gib = 0.5 / 1024.0  # 0.5 MB -> GiB
    return float(policy_params_millions) + per_env_gib * num_envs
