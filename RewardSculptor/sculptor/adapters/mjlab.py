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
    "base_height", "fallen",
]


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
    rsl_rl_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional override for the schema keys emitted by the reward-term
    # state snapshot. If empty, derived from task_id via _schema_for_task.
    schema_keys: Optional[list[str]] = None

    # Populated by __post_init__.
    _validated: bool = field(default=False, init=False, repr=False)

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

        # num_envs autocap on smaller VRAM.
        if self.device.startswith("cuda") and torch.cuda.is_available():
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

    # ── Contract ────────────────────────────────────────────────────────────
    def reward_contract(self) -> RewardContract:
        return RewardContract(
            observation_space_spec=None,
            action_space_spec=None,
            expected_info_keys=list(_INFO_KEYS),
            expected_components=None,
            supports_batched=True,
            training_device="gpu",
            # Conservative: 6 GiB for 4096-env G1 humanoid; smaller tasks
            # (Go1 / Anymal quadrupeds) fit in less. Per-task-class
            # override could live in a follow-up when VRAM probe data is
            # in hand (MJLAB_PIVOT_DESIGN §3.3).
            min_gpu_memory_gb=6.0 if "G1" in self.task_id else 4.0,
            state_schema=_schema_for_task(self.task_id),
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

        schema = _schema_for_task(self.task_id)
        action_dim = schema.get("actuator_force", (12,))[0]
        path = Path(reward_module_path).resolve()

        script = textwrap.dedent(
            """\
            import importlib.util, json, sys
            import torch
            path, schema_json, action_dim, info_keys_json = sys.argv[1:5]
            schema = json.loads(schema_json)
            info_keys = json.loads(info_keys_json)
            spec = importlib.util.spec_from_file_location('_probe', path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                state = {k: torch.zeros((1, *shape), dtype=torch.float32)
                         for k, shape in schema.items()}
                next_state = {k: torch.zeros((1, *shape), dtype=torch.float32)
                              for k, shape in schema.items()}
                action = torch.zeros((1, int(action_dim)), dtype=torch.float32)
                info = {k: torch.zeros((1,), dtype=torch.float32) for k in info_keys}
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
        info_keys_json = json.dumps(_INFO_KEYS)

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(path), schema_json, str(action_dim), info_keys_json,
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
    ) -> TrainResult:
        """Subprocess-train. `steps` is interpreted as `max_iterations`
        for rsl_rl's OnPolicyRunner (one iteration = num_envs *
        num_steps_per_env policy rollouts = ~num_envs * episode_length
        env steps). Caller budgets accordingly.
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
            else list(_schema_for_task(self.task_id).keys())
        )
        cmd += ["--schema-keys", ",".join(effective_schema_keys)]

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
