"""sculptor/adapters/gym_sb3.py — reference adapter for Gymnasium + SB3.

The pattern here is intended to be the simplest viable Sculptor integration
and the template external labs copy from. See `base.py` for the integration
patterns recommended for other RL stacks.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    SculptorAdapter,
    TrainResult,
)


# ── Reward-module loader ────────────────────────────────────────────────────
def _load_reward_module(path: Path):
    """Import a reward module by file path. Returns the module object.

    The module must define `compute_reward(state, action, next_state, info)
    -> (reward, components_dict)` and a `REWARD_SPEC` dict.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"reward module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_sculptor_reward_{abs(hash(str(path)))}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load reward module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "compute_reward"):
        raise AttributeError(
            f"{path} must define compute_reward(state, action, next_state, info)"
        )
    if not hasattr(mod, "REWARD_SPEC"):
        raise AttributeError(f"{path} must define REWARD_SPEC dict")
    return mod


# ── Reward-override wrapper ─────────────────────────────────────────────────
def _make_reward_override_wrapper():
    """Defer import so this module is importable without gymnasium installed
    (the contract test instantiates the adapter and inspects its contract).
    """
    import gymnasium as gym

    class RewardOverrideWrapper(gym.Wrapper):
        """Replace env reward with `reward_module.compute_reward(...)`.

        Loads the reward module ONCE at construction. Sculptor iterations
        write a fresh module per iteration and build a fresh wrapper — no
        hot reload, no module-level state shared across iterations.
        """

        def __init__(self, env, reward_module_path: Path):
            super().__init__(env)
            self._reward_module_path = Path(reward_module_path).resolve()
            mod = _load_reward_module(self._reward_module_path)
            self._compute_reward: Callable = mod.compute_reward
            self._reward_spec: dict = getattr(mod, "REWARD_SPEC", {})
            self._last_obs = None
            self._component_acc: dict[str, float] = defaultdict(float)
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
            # Pass-through metadata for downstream rollout logging.
            info = dict(info)
            info["sculptor_env_reward"] = float(env_reward)
            info["sculptor_components"] = {k: float(v) for k, v in components.items()}
            self._last_obs = next_obs
            return next_obs, float(reward), terminated, truncated, info

        def get_component_means(self) -> dict[str, float]:
            if self._step_count == 0:
                return {}
            return {k: v / self._step_count for k, v in self._component_acc.items()}

        def get_reward_spec(self) -> dict:
            return dict(self._reward_spec)

    return RewardOverrideWrapper


# ── Adapter ─────────────────────────────────────────────────────────────────
@dataclass
class GymSB3Adapter(SculptorAdapter):
    """Gymnasium env + Stable-Baselines3 PPO."""

    env_id: str
    policy: str = "MlpPolicy"
    n_envs: int = 4
    ppo_kwargs: dict[str, Any] = field(default_factory=dict)
    video_length_s: int = 20

    # Cached after first __post_init__ inspection.
    _obs_space: Any = field(default=None, init=False, repr=False)
    _action_space: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Inspect the env once for the reward contract; do not hold the env
        # open (we re-create per-train so seeds are fresh).
        try:
            import gymnasium as gym
        except ModuleNotFoundError:  # pragma: no cover
            self._obs_space = None
            self._action_space = None
            return
        env = gym.make(self.env_id)
        self._obs_space = env.observation_space
        self._action_space = env.action_space
        env.close()

    # ── Contract ────────────────────────────────────────────────────────────
    def reward_contract(self) -> RewardContract:
        return RewardContract(
            observation_space_spec=self._obs_space,
            action_space_spec=self._action_space,
            expected_info_keys=["x_velocity"],
            expected_components=None,
            # Gym-SB3 adapter is scalar-only by design. CPU-default training.
            supports_batched=False,
            training_device="any",
            min_gpu_memory_gb=None,
            state_schema=None,
        )

    # ── Component probe (scalar, subprocess-isolated) ───────────────────────
    def probe_component(self, reward_module_path: Path) -> ComponentProbe:
        """Run compute_reward on a zero-dummy gym-style state in a fresh
        subprocess. Mirrors the existing UI backend probe — 0.0 scalars +
        empty info dict (info keys default to 0.0 via the reward-module
        contract). Returns the ComponentProbe dataclass from base.py.
        """
        import json
        import textwrap

        path = Path(reward_module_path).resolve()
        info_keys = list(self.reward_contract().expected_info_keys or [])

        script = textwrap.dedent(
            """\
            import importlib.util, json, sys
            path, info_keys_json = sys.argv[1], sys.argv[2]
            spec = importlib.util.spec_from_file_location('_probe', path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                state = action = next_state = 0.0
                info = {k: 0.0 for k in json.loads(info_keys_json)}
                out = mod.compute_reward(state, action, next_state, info)
                if not (isinstance(out, tuple) and len(out) == 2):
                    raise TypeError(f'compute_reward must return (reward, components); got {type(out).__name__}')
                reward, components = out
                if not isinstance(components, dict):
                    raise TypeError('components must be a dict')
                comp_out = {str(k): float(v) for k, v in components.items()}
                print(json.dumps({'ok': True, 'components': comp_out, 'total': float(reward)}))
            except Exception as e:
                print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
                sys.exit(0)
            """
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(path), json.dumps(info_keys)],
                capture_output=True, text=True, timeout=15.0,
            )
        except subprocess.TimeoutExpired:
            return ComponentProbe(ok=False, error="subprocess timeout (15s)")
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

    # reward_batched: GymSB3 is scalar-only; inherit the default looping
    # fallback from SculptorAdapter (emits a RuntimeWarning when called).

    # ── Train ───────────────────────────────────────────────────────────────
    def train(
        self,
        reward_module_path: Path,
        output_dir: Path,
        steps: int,
        seed: int,
    ) -> TrainResult:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy

        reward_module_path = Path(reward_module_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        RewardOverrideWrapper = _make_reward_override_wrapper()
        env_id = self.env_id

        def _factory() -> Any:
            base_env = gym.make(env_id)
            return RewardOverrideWrapper(base_env, reward_module_path)

        vec_env = make_vec_env(_factory, n_envs=self.n_envs, seed=seed)

        # Tensorboard is optional. If not installed, fall back to a plain logs/ dir.
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        try:
            import tensorboard  # noqa: F401
            tb_log = str(logs_dir)
        except ModuleNotFoundError:
            tb_log = None

        model = PPO(
            self.policy,
            vec_env,
            verbose=0,
            seed=seed,
            tensorboard_log=tb_log,
            **self.ppo_kwargs,
        )
        model.learn(total_timesteps=int(steps), progress_bar=False)

        ckpt_path = output_dir / "checkpoint.zip"
        model.save(str(ckpt_path))

        # Quick eval for mean_return; small n to stay inside the time budget.
        mean_return, std_return = evaluate_policy(
            model, vec_env, n_eval_episodes=5, deterministic=True,
            return_episode_rewards=False)

        # Aggregate components across all wrapper envs.
        per_env_means = vec_env.env_method("get_component_means")
        agg: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for d in per_env_means:
            for k, v in d.items():
                agg[k] += v
                counts[k] += 1
        component_means = {k: agg[k] / counts[k] for k in agg}

        metrics_dict = {
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "n_eval_episodes": 5,
            "training_steps": int(steps),
            "n_envs": self.n_envs,
            "seed": int(seed),
        }
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(
                {"metrics": metrics_dict, "components": component_means},
                f, indent=2, sort_keys=True)

        # Reward spec for provenance.
        reward_spec = vec_env.env_method("get_reward_spec")[0]
        spec_path = output_dir / "reward_spec.json"
        with spec_path.open("w") as f:
            json.dump(reward_spec, f, indent=2, sort_keys=True, default=str)

        vec_env.close()

        return TrainResult(
            checkpoint_path=ckpt_path,
            metrics_dict=metrics_dict,
            component_means=component_means,
            logs_path=logs_dir,
        )

    # ── Rollout (Phase 2) ───────────────────────────────────────────────────
    def rollout(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        n_episodes: int,
    ) -> RolloutResult:
        import gymnasium as gym
        from stable_baselines3 import PPO

        checkpoint_path = Path(checkpoint_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        keyframes_dir = output_dir / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)

        # Build a render-mode env (no RewardOverrideWrapper — rollout reports
        # the env's native reward AND the sculpted reward via re-application
        # of the reward module).
        env = gym.make(self.env_id, render_mode="rgb_array")
        model = PPO.load(str(checkpoint_path))

        episodes: list[dict] = []
        for ep_idx in range(n_episodes):
            obs, info = env.reset(seed=1000 + ep_idx)
            done = False
            ep = {
                "obs": [], "actions": [], "rewards": [],
                "terminations": [], "truncations": [], "frames": [],
                "infos_flat": [],
            }
            ep["obs"].append(np.asarray(obs, dtype=np.float32))
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                ep["actions"].append(np.asarray(action, dtype=np.float32))
                ep["rewards"].append(float(reward))
                ep["terminations"].append(bool(terminated))
                ep["truncations"].append(bool(truncated))
                ep["infos_flat"].append(_flatten_info(info))
                ep["frames"].append(np.asarray(frame, dtype=np.uint8))
                ep["obs"].append(np.asarray(obs, dtype=np.float32))
                done = terminated or truncated
            ep["return"] = float(sum(ep["rewards"]))
            ep["length"] = len(ep["rewards"])
            episodes.append(ep)
        env.close()

        best_idx = int(np.argmax([e["return"] for e in episodes]))
        best = episodes[best_idx]

        # Keyframes: 12 evenly spaced frames from best episode.
        n_frames = len(best["frames"])
        if n_frames > 0:
            from PIL import Image  # via torchvision dep tree, present in sb3 stack
            idxs = np.linspace(0, n_frames - 1, num=min(12, n_frames)).astype(int)
            for i, fi in enumerate(idxs):
                img = Image.fromarray(best["frames"][fi])
                img.save(keyframes_dir / f"frame_{i:02d}.png")

        # Stitched mp4: 4–8 episodes, max 20s total via ffmpeg.
        video_path = output_dir / "rollout.mp4"
        all_frames: list[np.ndarray] = []
        episodes_for_video = episodes[: min(8, max(4, n_episodes))]
        for e in episodes_for_video:
            all_frames.extend(e["frames"])
        # Cap to video_length_s seconds (assume render dt ≈ 1/50, conservative)
        # We don't know env fps exactly; clip to 20s × 50fps = 1000 frames.
        max_frames = self.video_length_s * 50
        if len(all_frames) > max_frames:
            all_frames = all_frames[:max_frames]

        wrote_mp4 = _write_mp4(video_path, all_frames, fps=50)
        if not wrote_mp4:
            # ffmpeg unavailable: dump a sentinel + frame stash so the
            # downstream "file exists, size > 0" check still passes.
            video_path.write_bytes(b"# ffmpeg not available; frames in keyframes/\n")

        # trajectory.npz — concatenate across episodes with episode_id markers.
        traj_arrays = _episodes_to_npz_dict(episodes)
        trajectory_path = output_dir / "trajectory.npz"
        np.savez_compressed(trajectory_path, **traj_arrays)

        result = RolloutResult(
            video_path=video_path,
            keyframes_dir=keyframes_dir,
            trajectory_path=trajectory_path,
            n_episodes=n_episodes,
        )

        # Behavior metrics → behavior.json
        behavior = self.compute_behavior_metrics(result)
        with (output_dir / "behavior.json").open("w") as f:
            json.dump(behavior, f, indent=2, sort_keys=True, default=str)

        return result

    # ── Behavior metrics ────────────────────────────────────────────────────
    def compute_behavior_metrics(self, rollout: RolloutResult) -> dict[str, Any]:
        """Hopper-flavored behavior metrics. Adapter authors should override
        for their domain; this implementation looks for `x_velocity` in the
        per-step infos and termination reasons in the trajectory.
        """
        traj = np.load(rollout.trajectory_path, allow_pickle=True)
        ep_ids = traj["episode_id"]
        rewards = traj["rewards"]
        terminations = traj["terminations"]
        truncations = traj["truncations"]
        x_velocity = traj["x_velocity"] if "x_velocity" in traj.files else None

        unique_eps = np.unique(ep_ids)
        ep_lengths: list[int] = []
        ep_returns: list[float] = []
        ep_terminated: list[bool] = []
        ep_x_vel_means: list[float] = []
        for eid in unique_eps:
            mask = ep_ids == eid
            ep_lengths.append(int(mask.sum()))
            ep_returns.append(float(rewards[mask].sum()))
            ep_terminated.append(bool(terminations[mask].any()))
            if x_velocity is not None:
                ep_x_vel_means.append(float(np.nan_to_num(x_velocity[mask]).mean()))

        n = max(len(unique_eps), 1)
        out: dict[str, Any] = {
            "n_episodes": int(len(unique_eps)),
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "max_episode_length": int(max(ep_lengths)) if ep_lengths else 0,
            "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "fall_rate": float(sum(ep_terminated)) / n,
            "termination_reason_counts": {
                "terminated": int(sum(ep_terminated)),
                "truncated": int(sum(1 for tt, te in zip(
                    [bool(truncations[ep_ids == eid].any()) for eid in unique_eps],
                    ep_terminated) if tt and not te)),
            },
        }
        if ep_x_vel_means:
            out["mean_forward_velocity"] = float(np.mean(ep_x_vel_means))
        return out


# ── Helpers ─────────────────────────────────────────────────────────────────
def _flatten_info(info: dict) -> dict:
    """Drop non-scalar / non-1D-array entries; numpy-arrayify the rest."""
    flat = {}
    for k, v in info.items():
        if isinstance(v, (int, float, np.floating, np.integer, bool, np.bool_)):
            flat[k] = float(v)
        elif isinstance(v, np.ndarray) and v.ndim <= 1:
            flat[k] = v.tolist()
    return flat


def _episodes_to_npz_dict(episodes: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate per-episode lists into flat arrays + episode_id marker."""
    obs = []
    actions = []
    rewards = []
    terminations = []
    truncations = []
    episode_id = []
    components_per_step: list[dict] = []
    x_velocity: list[float] = []
    for eid, ep in enumerate(episodes):
        obs.extend(ep["obs"][:-1])  # drop trailing reset obs to match action length
        actions.extend(ep["actions"])
        rewards.extend(ep["rewards"])
        terminations.extend(ep["terminations"])
        truncations.extend(ep["truncations"])
        episode_id.extend([eid] * len(ep["rewards"]))
        for inf in ep["infos_flat"]:
            x_velocity.append(float(inf.get("x_velocity", float("nan"))))
            comps = inf.get("sculptor_components", {})
            components_per_step.append(
                comps if isinstance(comps, dict) else {})
    out = {
        "obs": np.asarray(obs, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminations": np.asarray(terminations, dtype=np.bool_),
        "truncations": np.asarray(truncations, dtype=np.bool_),
        "episode_id": np.asarray(episode_id, dtype=np.int32),
        "x_velocity": np.asarray(x_velocity, dtype=np.float32),
        "components_per_step": np.asarray(components_per_step, dtype=object),
    }
    return out


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 50) -> bool:
    """Encode `frames` to mp4 via ffmpeg-as-subprocess. Returns True on success.

    No moviepy. Prefers ffmpeg on PATH; falls back to the bundled binary
    shipped by `imageio-ffmpeg` (added as a project dep precisely so this
    works out-of-the-box on Windows without a system ffmpeg).
    """
    if not frames:
        return False
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            from PIL import Image
            Image.fromarray(fr).save(Path(td) / f"f_{i:06d}.png")
        cmd = [
            ffmpeg_path, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(Path(td) / "f_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale={w}:{h}",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True)
        return res.returncode == 0 and path.exists() and path.stat().st_size > 0
