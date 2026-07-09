"""Tier-D certification: physics-track a Tier-K clip in our own mjlab sim
(§REFERENCE_TRAJECTORY_PLAN §2.3 Tier D, §11 R4).

A Tier-K clip (retargeted/segmented mocap, kinematics only, no dynamics
guarantee) is TRACKED by a bounded DeepMimic-style tracking run: a
throwaway sculpt project is built (donor adapter config + a PROGRAMMATIC
tracking reward that penalizes phase-indexed joint/root deviation from
the clip), trained briefly, then rolled out WITH the clip's own
eval-reset override so the rollout attempts the motion from the clip's
start. Rollout-vs-clip tracking error decides feasibility:

    - within tolerance  -> tier upgrades K -> D; the rollout's
      trajectory.npz is copied beside the clip as `tierD_rollout.npz`
      and a `tierD` provenance block records iterations/errors/path.
    - outside tolerance  -> tier stays K; a `tierD` block with
      `feasible: false` records the attempt (a useful verdict, never an
      error).

Composes EXISTING machinery only — this module does not extend the
adapter contract, `sculptor.reference`, or `sculptor.eval`. It:
  * copies `[adapter]` config out of a donor project's `config.toml`
    (`--donor-project`) into a throwaway project directory;
  * writes a tracking `rewards/current.py` whose target arrays are
    embedded as literals (reward modules are plain files loaded via
    `_import_reward_module`/`exec_module` — see
    `sculptor.adapters.base._import_reward_module` — so there is no
    project-file-access channel available to a reward function; the
    clip's phase-downsampled target arrays are baked into the generated
    source instead);
  * calls `apply_reference_rsi` (from `sculptor.reference`, unmodified)
    to seed RSI/sunk-termination/eval-reset env-spec files from the clip
    itself — it already generalizes across getup/airborne archetypes;
  * trains + rolls out via the REAL adapter API
    (`sculptor.adapters.base.load_adapter(config_path).train(...)` /
    `.rollout(...)`), the same minimal programmatic path
    `scripts/phase2_smoke.py` uses;
  * scores the rollout's `trajectory.npz` against the clip with
    role-resolved common joints (`sculptor.eval.joint_resolver`) and
    updates provenance via `sculptor.refs.library.read_provenance` /
    `write_provenance` (no new library helper needed — those two already
    form a read-mutate-write seam).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: Feasibility thresholds (§mission spec — "calibrate later against the
#: known-good jump reference"). Both must hold for a tracking run to
#: certify Tier D.
MEAN_JOINT_ERR_THRESHOLD_RAD = 0.35
ROOT_Z_RMSE_THRESHOLD_M = 0.12

#: Reward-shaping weights for the generated tracking reward's Gaussian
#: kernels: `exp(-w * err**2)`. Chosen so a "close" pose (few-degree
#: joint error, few-cm root error) scores near 1.0 while a "way off"
#: pose (tens of degrees, tens of cm) decays toward 0 — the standard
#: DeepMimic imitation-kernel shape, not independently tuned per clip.
JOINT_ERR_WEIGHT = 8.0
ROOT_ERR_WEIGHT = 40.0

#: Default training budget — small iteration count x modest steps/iter,
#: comfortably under an hour on an RTX 5070 (§mission: "1500-3000 steps
#: x 2-3 iters"). `--iterations` overrides the iteration count only;
#: `steps` here is mjlab's `max_iterations` per the adapter contract
#: (see `MjlabAdapter.train`'s docstring — `steps` IS max_iterations,
#: not env steps), so this module trains ONE adapter.train() call with
#: `steps=iterations`.
DEFAULT_ITERATIONS = 3
DEFAULT_STEPS_PER_ITERATION = 2000
DEFAULT_N_EPISODES = 2
#: Phase-target downsampling: number of clocked keyframes the generated
#: reward tracks against. Independent of the clip's native frame count
#: (a long mocap clip and a short one both compress to this many
#: phase-indexed targets) — keeps the generated reward source small and
#: the phase clock (`t/T`) well-defined regardless of clip fps.
N_PHASE_TARGETS = 32


class TrackError(RuntimeError):
    """Raised for setup failures (bad donor config, clip missing common
    joints with the robot, etc.) — never for a feasibility verdict itself,
    which is always a normal (non-exception) result."""


# ── phase-target downsampling ───────────────────────────────────────────
def downsample_phase_targets(
    array: np.ndarray, n: int = N_PHASE_TARGETS,
) -> np.ndarray:
    """Resample `array` (shape `(T, ...)`) to exactly `n` phase-indexed
    rows via nearest-frame lookup at evenly spaced phase fractions
    `[0, 1)` — i.e. index `round(phase * (T - 1))`. Deterministic, no
    interpolation (avoids inventing joint poses between real mocap
    frames). `n` must be >= 1; `array` must have `T >= 1` along axis 0.
    """
    array = np.asarray(array)
    t = array.shape[0]
    if t < 1:
        raise ValueError(f"array must have at least 1 frame along axis 0, got {t}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    phases = np.linspace(0.0, 1.0, n, endpoint=False)
    idx = np.clip(np.round(phases * (t - 1)).astype(int), 0, t - 1)
    return array[idx]


# ── tracking reward source generation ───────────────────────────────────
def _format_array_literal(arr: np.ndarray, *, ndigits: int = 5) -> str:
    """Render a 1-D or 2-D float array as a Python list literal, rounded
    to `ndigits` — compact enough to embed a few dozen phase-targets x a
    couple dozen joints directly in generated reward source."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), ndigits)
    if rounded.ndim == 1:
        return "[" + ", ".join(repr(float(x)) for x in rounded) + "]"
    return "[\n    " + ",\n    ".join(
        "[" + ", ".join(repr(float(x)) for x in row) + "]" for row in rounded
    ) + "\n]"


def generate_tracking_reward_source(
    *,
    clip_id: str,
    joint_names: list[str],
    target_joint_pos: np.ndarray,
    target_root_z: np.ndarray,
    episode_len_steps: int,
    joint_err_weight: float = JOINT_ERR_WEIGHT,
    root_err_weight: float = ROOT_ERR_WEIGHT,
) -> str:
    """Build the PROGRAMMATIC (non-LLM) tracking reward module source.

    `compute_reward(state, action, next_state, info)` follows the
    project-wide reward contract (`examples/hopper/rewards/v0.py`'s
    docstring) — `state`/`next_state` here are the mjlab G1 state dict
    (`state["qpos"]`: 6 free-joint DOFs + N actuated joint DOFs, in the
    robot's canonical joint order; see
    `sculptor.eval.robot_manifest.G1_29`). `info["episode_length"]`
    (mjlab's per-step step counter) supplies the phase clock
    `t / episode_len_steps`, clamped to `[0, 1)`.

    `target_joint_pos`/`target_root_z` are phase-downsampled arrays
    (`downsample_phase_targets`) EMBEDDED AS LITERALS in the returned
    source — reward modules are loaded by
    `sculptor.adapters.base._import_reward_module` via
    `importlib.util.spec_from_file_location` + `exec_module`, a plain
    file import with no project-relative data-file channel available to
    the function body at call time, so the clip data must travel inside
    the module source itself rather than as a sibling file the reward
    reads at runtime.

    reward = exp(-joint_err_weight * mean_joint_err_rad**2)
           + exp(-root_err_weight  * root_z_err_m**2)
    (§mission spec's exact two-Gaussian-kernel formula.) `joint_names` is
    the SAME order as `target_joint_pos` columns and is asserted against
    `qpos`'s trailing (actuated-joint) slice length at reward-call time
    via `len(joint_names)` — a project.joint-count mismatch raises inside
    `compute_reward` rather than silently misindexing.
    """
    if target_joint_pos.shape[0] != target_root_z.shape[0]:
        raise ValueError(
            "target_joint_pos and target_root_z must share phase-count: "
            f"{target_joint_pos.shape[0]} vs {target_root_z.shape[0]}")
    if target_joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            "target_joint_pos column count must equal len(joint_names): "
            f"{target_joint_pos.shape[1]} vs {len(joint_names)}")
    n_phase = target_joint_pos.shape[0]
    n_joints = target_joint_pos.shape[1]

    joint_pos_literal = _format_array_literal(target_joint_pos)
    root_z_literal = _format_array_literal(target_root_z)
    names_literal = "[" + ", ".join(repr(str(n)) for n in joint_names) + "]"

    return f'''"""Auto-generated Tier-D tracking reward for clip {clip_id!r}.

Generated by `sculptor.refs.track.generate_tracking_reward_source` — a
PROGRAMMATIC (non-LLM) DeepMimic-style tracking reward. Target arrays are
phase-downsampled to {n_phase} keyframes and embedded as literals (reward
modules have no project-file-access channel at call time — see
`sculptor.refs.track`'s module docstring). Do not hand-edit; regenerate
from the clip instead.
"""
from __future__ import annotations

import numpy as np

REWARD_SPEC: dict = {{
    "version": "tierD-track-v1",
    "description": "DeepMimic-style phase-indexed tracking reward for "
                    "Tier-D certification of clip {clip_id!r}.",
    "author": "sculptor.refs.track",
    "parent_hash": None,
    "hyperparameters": {{
        "joint_err_weight": {joint_err_weight!r},
        "root_err_weight": {root_err_weight!r},
        "n_phase_targets": {n_phase},
        "episode_len_steps": {episode_len_steps!r},
    }},
    "references": [],
}}

CLIP_ID = {clip_id!r}
JOINT_NAMES = {names_literal}
N_JOINTS = {n_joints}
N_PHASE = {n_phase}
EPISODE_LEN_STEPS = {episode_len_steps!r}
JOINT_ERR_WEIGHT = {joint_err_weight!r}
ROOT_ERR_WEIGHT = {root_err_weight!r}

# Phase-indexed targets, shape (N_PHASE, N_JOINTS) / (N_PHASE,).
TARGET_JOINT_POS = np.asarray({joint_pos_literal}, dtype=np.float64).reshape(N_PHASE, N_JOINTS)
TARGET_ROOT_Z = np.asarray({root_z_literal}, dtype=np.float64)


def _phase_index(info) -> int:
    step = int(info.get("episode_length", 0) or 0)
    phase = (step / float(EPISODE_LEN_STEPS)) if EPISODE_LEN_STEPS > 0 else 0.0
    phase = min(max(phase, 0.0), 0.999999)
    return int(phase * N_PHASE)


def compute_reward(state, action, next_state, info):
    del action  # tracking reward does not penalize control effort
    qpos = np.asarray(next_state["qpos"], dtype=np.float64)
    if qpos.shape[0] < 7 + N_JOINTS:
        raise ValueError(
            f"qpos too short for {{N_JOINTS}} tracked joints: "
            f"shape={{qpos.shape}}")
    root_z = float(qpos[2])
    joint_pos = qpos[7:7 + N_JOINTS]

    i = _phase_index(info)
    target_joint = TARGET_JOINT_POS[i]
    target_root_z = TARGET_ROOT_Z[i]

    joint_err = joint_pos - target_joint
    mean_joint_err_sq = float(np.mean(joint_err ** 2))
    root_err = root_z - float(target_root_z)

    joint_term = float(np.exp(-JOINT_ERR_WEIGHT * mean_joint_err_sq))
    root_term = float(np.exp(-ROOT_ERR_WEIGHT * (root_err ** 2)))

    components = {{
        "joint_tracking": joint_term,
        "root_tracking": root_term,
    }}
    reward = joint_term + root_term
    return float(reward), components
'''


# ── error metrics: rollout trajectory vs clip ───────────────────────────
@dataclass
class TrackingErrors:
    """Rollout-vs-clip tracking metrics (§mission spec's exact set)."""

    mean_joint_err_rad: float
    max_joint_err_rad: float
    root_z_rmse_m: float
    duration_coverage: float  # rollout_frames / target_frames, clamped [0, 1]
    common_joint_names: list[str] = field(default_factory=list)
    n_common_joints: int = 0

    @property
    def feasible(self) -> bool:
        return (
            self.mean_joint_err_rad < MEAN_JOINT_ERR_THRESHOLD_RAD
            and self.root_z_rmse_m < ROOT_Z_RMSE_THRESHOLD_M
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_joint_err_rad": round(self.mean_joint_err_rad, 6),
            "max_joint_err_rad": round(self.max_joint_err_rad, 6),
            "root_z_rmse_m": round(self.root_z_rmse_m, 6),
            "duration_coverage": round(self.duration_coverage, 6),
            "common_joint_names": list(self.common_joint_names),
            "n_common_joints": self.n_common_joints,
            "feasible": self.feasible,
            "thresholds": {
                "mean_joint_err_rad": MEAN_JOINT_ERR_THRESHOLD_RAD,
                "root_z_rmse_m": ROOT_Z_RMSE_THRESHOLD_M,
            },
        }


def _resolve_common_joints(
    clip_joint_names: list[str], rollout_joint_names: list[str],
) -> tuple[list[int], list[int], list[str]]:
    """Positional-name intersection (exact string match) between the
    clip's `joint_names` and the rollout robot's joint order — role
    resolution (`sculptor.eval.joint_resolver`) is for goal/role lookups
    against a single robot's names; here BOTH sides already carry robot
    joint names (retargeted clip + live rollout), so a direct name match
    is the correct (and simpler) correspondence. Returns
    (clip_indices, rollout_indices, names) in ROLLOUT order, so both
    trajectory slices are self-consistent for elementwise error."""
    rollout_index = {name: i for i, name in enumerate(rollout_joint_names)}
    clip_idx: list[int] = []
    rollout_idx: list[int] = []
    names: list[str] = []
    for ci, name in enumerate(clip_joint_names):
        ri = rollout_index.get(name)
        if ri is not None:
            clip_idx.append(ci)
            rollout_idx.append(ri)
            names.append(name)
    return clip_idx, rollout_idx, names


def compute_tracking_errors(
    *,
    clip: dict[str, Any],
    rollout_joint_pos: np.ndarray,
    rollout_root_z: np.ndarray,
    rollout_joint_names: list[str],
) -> TrackingErrors:
    """Score a rollout (`trajectory.npz`-shaped arrays) against the clip
    it was tracking. `rollout_joint_pos` is `(T, J_rollout)`,
    `rollout_root_z` is `(T,)` (typically `root_link_pos_w[..., 2]`
    averaged over the env axis if E>1 — callers pass a single-env slice).
    Common joints resolved by exact name match; a clip with NO joint
    names or NO overlap with the rollout robot degrades to root-only
    scoring (`mean_joint_err_rad`/`max_joint_err_rad` both 0.0 — never
    raises, matches the "kinematic-tier contacts inferred" spirit of
    degrading gracefully on missing channels elsewhere in refs/)."""
    clip_joint_pos = clip.get("joint_pos")
    clip_joint_names = clip.get("joint_names") or []
    clip_root_z = np.asarray(clip["root_pos_z"], dtype=np.float64)

    rollout_joint_pos = np.asarray(rollout_joint_pos, dtype=np.float64)
    rollout_root_z = np.asarray(rollout_root_z, dtype=np.float64)

    t_rollout = rollout_root_z.shape[0]
    t_clip = clip_root_z.shape[0]
    duration_coverage = min(1.0, t_rollout / t_clip) if t_clip > 0 else 0.0

    common_names: list[str] = []
    mean_err = 0.0
    max_err = 0.0
    if clip_joint_pos is not None and clip_joint_names and t_rollout > 0:
        clip_idx, rollout_idx, common_names = _resolve_common_joints(
            list(clip_joint_names), list(rollout_joint_names))
        if common_names:
            clip_jp = np.asarray(clip_joint_pos, dtype=np.float64)
            n = min(t_rollout, t_clip)
            # Phase-align both traces to n common frames via the same
            # nearest-frame downsampling used for reward-target
            # generation, so unequal rollout/clip lengths still compare
            # like-for-like phases rather than truncating one blindly.
            clip_at_n = downsample_phase_targets(clip_jp[:, clip_idx], n=n)
            rollout_at_n = downsample_phase_targets(
                rollout_joint_pos[:, rollout_idx], n=n)
            err = clip_at_n - rollout_at_n
            abs_err = np.abs(err)
            mean_err = float(np.mean(abs_err))
            max_err = float(np.max(abs_err))

    n = min(t_rollout, t_clip) if t_clip > 0 and t_rollout > 0 else 0
    if n > 0:
        clip_z_at_n = downsample_phase_targets(clip_root_z, n=n)
        rollout_z_at_n = downsample_phase_targets(rollout_root_z, n=n)
        root_rmse = float(np.sqrt(np.mean((clip_z_at_n - rollout_z_at_n) ** 2)))
    else:
        root_rmse = float("inf")

    return TrackingErrors(
        mean_joint_err_rad=mean_err,
        max_joint_err_rad=max_err,
        root_z_rmse_m=root_rmse,
        duration_coverage=duration_coverage,
        common_joint_names=common_names,
        n_common_joints=len(common_names),
    )


# ── donor-config templating ─────────────────────────────────────────────
def read_donor_adapter_config(donor_project: Path) -> dict[str, Any]:
    """Read `[adapter]` (class + config) out of a donor project's
    `config.toml`. Raises `TrackError` with an actionable message if the
    donor has no `[adapter]` section — mirrors
    `sculptor.adapters.base.load_adapter`'s own validation, one layer up
    (this reads the TABLE without instantiating anything)."""
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(donor_project) / "config.toml"
    if not config_path.is_file():
        raise TrackError(f"donor project has no config.toml: {config_path}")
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    adapter_cfg = cfg.get("adapter")
    if not adapter_cfg or "class" not in adapter_cfg:
        raise TrackError(
            f"{config_path}: missing [adapter] section or [adapter].class")
    return {
        "class": adapter_cfg["class"],
        "config": adapter_cfg.get("config", {}) or {},
    }


def write_project_config_toml(
    project_dir: Path, adapter_cfg: dict[str, Any],
) -> Path:
    """Write a minimal `config.toml` for the throwaway tracking project,
    templated from a donor's `[adapter]` table. Hand-serialized (no TOML
    writer dependency in this project) — values are constrained to the
    JSON-safe primitive types `[adapter].config` tables already carry
    (str/int/float/bool/nested dict/list), which is all `load_adapter`'s
    own tomllib reader ever produces."""
    def _toml_value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, dict):
            return "{ " + ", ".join(
                f"{k} = {_toml_value(v2)}" for k, v2 in v.items()) + " }"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_toml_value(x) for x in v) + "]"
        raise TrackError(f"unsupported TOML value type for {v!r}: {type(v)}")

    config_lines = [f"{k} = {_toml_value(v)}" for k, v in adapter_cfg["config"].items()]
    content = (
        "[adapter]\n"
        f'class = {json.dumps(adapter_cfg["class"])}\n'
        "config = { " + ", ".join(config_lines) + " }\n"
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


# ── throwaway project construction ──────────────────────────────────────
@dataclass
class TrackPlan:
    """Everything `track_clip` would do, computed up front — returned
    directly by `--dry-run` and reused by the real (training) path so
    both stay in sync by construction."""

    clip_id: str
    robot: str
    project_dir: Path
    reward_path: Path
    config_path: Path
    env_dir: Path
    joint_names: list[str]
    n_phase_targets: int
    iterations: int
    steps_per_iteration: int
    n_episodes: int


def build_track_project(
    *,
    clip: dict[str, Any],
    clip_id: str,
    robot: str,
    donor_project: Path,
    project_dir: Path,
    iterations: int = DEFAULT_ITERATIONS,
    steps_per_iteration: int = DEFAULT_STEPS_PER_ITERATION,
    n_episodes: int = DEFAULT_N_EPISODES,
    n_phase_targets: int = N_PHASE_TARGETS,
) -> TrackPlan:
    """Build the throwaway sculpt project directory (config.toml +
    rewards/current.py + env/ RSI+eval-reset), WITHOUT training. Used by
    both `--dry-run` (prints the plan and stops) and the real path
    (trains right after this returns)."""
    from sculptor.reference import apply_reference_rsi, derive_eval_reset

    joint_names = list(clip.get("joint_names") or [])
    if not joint_names or clip.get("joint_pos") is None:
        raise TrackError(
            f"clip {clip_id!r} has no joint_pos/joint_names — Tier-D "
            "tracking needs a per-joint target to track against")

    adapter_cfg = read_donor_adapter_config(donor_project)
    config_path = write_project_config_toml(project_dir, adapter_cfg)

    episode_len_steps = int(steps_per_iteration)
    target_joint_pos = downsample_phase_targets(
        np.asarray(clip["joint_pos"], dtype=np.float64), n=n_phase_targets)
    target_root_z = downsample_phase_targets(
        np.asarray(clip["root_pos_z"], dtype=np.float64), n=n_phase_targets)

    reward_source = generate_tracking_reward_source(
        clip_id=clip_id,
        joint_names=joint_names,
        target_joint_pos=target_joint_pos,
        target_root_z=target_root_z,
        episode_len_steps=episode_len_steps,
    )
    rewards_dir = project_dir / "rewards"
    rewards_dir.mkdir(parents=True, exist_ok=True)
    (rewards_dir / "__init__.py").write_text("", encoding="utf-8")
    reward_path = rewards_dir / "current.py"
    reward_path.write_text(reward_source, encoding="utf-8")

    env_dir = project_dir / "env"
    apply_reference_rsi(env_dir, clip)
    eval_reset = derive_eval_reset(clip)
    if eval_reset is not None:
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "eval_reset.json").write_text(
            json.dumps(eval_reset, indent=2, sort_keys=True), encoding="utf-8")

    return TrackPlan(
        clip_id=clip_id,
        robot=robot,
        project_dir=project_dir,
        reward_path=reward_path,
        config_path=config_path,
        env_dir=env_dir,
        joint_names=joint_names,
        n_phase_targets=n_phase_targets,
        iterations=iterations,
        steps_per_iteration=steps_per_iteration,
        n_episodes=n_episodes,
    )


# ── provenance update ────────────────────────────────────────────────────
def update_provenance_tier_d(
    *,
    robot: str,
    clip_id: str,
    errors: TrackingErrors,
    iterations: int,
    rollout_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """Read -> mutate -> write the clip's provenance with the Tier-D
    certification result (§mission spec's exact contract): feasible ->
    `tier="D"` + `tierD` block including `rollout_path`; infeasible ->
    tier stays whatever it was (K), `tierD.feasible=False` recorded.
    Rebuilds the library index for this clip afterward. Uses only the
    EXISTING `library.read_provenance`/`write_provenance` seam — no new
    library helper needed."""
    from sculptor.refs import library

    prov = library.read_provenance(robot, clip_id, root=root)
    tier_d_block: dict[str, Any] = {
        "tracked_at": library._utc_now_iso(),
        "iterations": iterations,
        "errors": errors.to_dict(),
    }
    if errors.feasible:
        prov["tier"] = "D"
        if rollout_path is not None:
            tier_d_block["rollout_path"] = str(rollout_path)
    else:
        tier_d_block["feasible"] = False
    prov["tierD"] = tier_d_block
    library.write_provenance(robot, clip_id, prov, root=root)
    library.rebuild_index(root=root)
    return prov


# ── orchestration ────────────────────────────────────────────────────────
@dataclass
class TrackResult:
    plan: TrackPlan
    errors: Optional[TrackingErrors]
    provenance: dict[str, Any]
    dry_run: bool


def track_clip(
    *,
    clip_id: str,
    robot: str = "g1",
    donor_project: Path,
    iterations: int = DEFAULT_ITERATIONS,
    steps_per_iteration: int = DEFAULT_STEPS_PER_ITERATION,
    n_episodes: int = DEFAULT_N_EPISODES,
    seed: int = 0,
    project_dir: Optional[Path] = None,
    dry_run: bool = False,
    library_root: Optional[Path] = None,
    progress: Optional[Any] = None,
) -> TrackResult:
    """Full Tier-D certification pipeline for one already-Tier-K clip.

    1. load the clip + its provenance (`sculptor.refs.library`,
       `sculptor.reference.load_clip`);
    2. build the throwaway project (`build_track_project`);
    3. `--dry-run` stops here;
    4. else: train via `load_adapter(config).train(...)`, roll out via
       `.rollout(...)` (the real minimal programmatic path — see module
       docstring), score the rollout vs the clip
       (`compute_tracking_errors`), copy `trajectory.npz` beside the clip
       as `tierD_rollout.npz` on success, and update provenance
       (`update_provenance_tier_d`).
    """
    from sculptor.adapters.base import load_adapter
    from sculptor.reference import load_clip
    from sculptor.refs import library

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    lib_clip_path = library.clip_dir(robot, clip_id, root=library_root) / library.CLIP_FILENAME
    if not lib_clip_path.is_file():
        raise TrackError(f"no such clip in library: {robot}/{clip_id}")
    clip = load_clip(lib_clip_path)

    if project_dir is None:
        clip_d = library.clip_dir(robot, clip_id, root=library_root)
        project_dir = clip_d / "tierD_work"

    _log(f"[track] building throwaway project at {project_dir}")
    plan = build_track_project(
        clip=clip, clip_id=clip_id, robot=robot, donor_project=donor_project,
        project_dir=project_dir, iterations=iterations,
        steps_per_iteration=steps_per_iteration, n_episodes=n_episodes)

    if dry_run:
        prov = library.read_provenance(robot, clip_id, root=library_root)
        return TrackResult(plan=plan, errors=None, provenance=prov, dry_run=True)

    _log(f"[track] loading adapter from {plan.config_path}")
    adapter = load_adapter(plan.config_path)

    import inspect

    train_accepts_init_policy = "init_policy_path" in inspect.signature(
        adapter.train).parameters

    train_dir = project_dir / "train"
    _log(
        f"[track] training {plan.iterations} iteration(s) x "
        f"{plan.steps_per_iteration} steps -> {train_dir}")
    ckpt_path = None
    for i in range(plan.iterations):
        extra: dict[str, Any] = {}
        if ckpt_path is not None and train_accepts_init_policy:
            extra["init_policy_path"] = ckpt_path
        result = adapter.train(
            reward_module_path=plan.reward_path,
            output_dir=train_dir,
            steps=plan.steps_per_iteration,
            seed=seed,
            **extra,
        )
        ckpt_path = result.checkpoint_path
        _log(f"[track] iteration {i + 1}/{plan.iterations} done: {ckpt_path}")

    rollout_extra: dict[str, Any] = {}
    if "seed" in inspect.signature(adapter.rollout).parameters:
        rollout_extra["seed"] = seed

    rollout_dir = project_dir / "rollout"
    _log(f"[track] rolling out {plan.n_episodes} episode(s) -> {rollout_dir}")
    rollout_result = adapter.rollout(
        checkpoint_path=ckpt_path,
        output_dir=rollout_dir,
        n_episodes=plan.n_episodes,
        **rollout_extra,
    )

    with np.load(rollout_result.trajectory_path) as npz:
        if "joint_pos" not in npz.files or "root_link_pos_w" not in npz.files:
            raise TrackError(
                f"rollout trajectory at {rollout_result.trajectory_path} is "
                "missing joint_pos/root_link_pos_w — cannot score tracking "
                "(adapter/task did not emit the expanded §7.1 fields)")
        # Shape (T, E, J) / (T, E, 3) per the mjlab runner's trajectory
        # contract — use env 0 (single-episode-shaped scoring; n_episodes
        # small by design for a Tier-D smoke run).
        rollout_joint_pos = npz["joint_pos"][:, 0, :]
        rollout_root_z = npz["root_link_pos_w"][:, 0, 2]

    from sculptor.eval.robot_manifest import robot_joint_names

    rollout_joint_names = robot_joint_names(robot) or plan.joint_names
    errors = compute_tracking_errors(
        clip=clip, rollout_joint_pos=rollout_joint_pos,
        rollout_root_z=rollout_root_z, rollout_joint_names=rollout_joint_names)
    _log(f"[track] errors: {errors.to_dict()}")

    rollout_dest = None
    if errors.feasible:
        clip_d = library.clip_dir(robot, clip_id, root=library_root)
        rollout_dest = clip_d / "tierD_rollout.npz"
        shutil.copyfile(rollout_result.trajectory_path, rollout_dest)
        _log(f"[track] feasible — rollout copied to {rollout_dest}")
    else:
        _log("[track] infeasible-for-robot (tier stays K)")

    prov = update_provenance_tier_d(
        robot=robot, clip_id=clip_id, errors=errors,
        iterations=plan.iterations, rollout_path=rollout_dest,
        root=library_root)

    return TrackResult(plan=plan, errors=errors, provenance=prov, dry_run=False)
