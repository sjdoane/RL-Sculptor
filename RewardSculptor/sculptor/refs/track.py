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

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sys

import numpy as np

from sculptor.refs import timing as _timing

#: Feasibility thresholds (§mission spec — "calibrate later against the
#: known-good jump reference"). Both must hold for a tracking run to
#: certify Tier D.
MEAN_JOINT_ERR_THRESHOLD_RAD = 0.35
ROOT_Z_RMSE_THRESHOLD_M = 0.12

#: Above which a clip's root height is read as a WORLD height, below which
#: as an origin-relative excursion. Retargeted AMASS clips zero the root
#: translation, so their `root_pos_z` is "height above the initial pose"
#: and lives near 0 (measured library-wide: 5798 of 6015 g1 clips peak
#: under 0.11 m) while a standing G1 base is ~0.74 m. A humanoid whose
#: base stays under 0.30 m for an ENTIRE clip is not standing at any
#: point, which no locomotion/jump/kick reference does — so this cleanly
#: separates the two conventions. A clip may state the convention
#: outright via `clip["root_frame"]`; the heuristic is only the fallback
#: for the 6015 clips ingested before the field existed.
ORIGIN_RELATIVE_MAX_ROOT_Z_M = 0.30

#: A tracking policy must beat the best CONSTANT pose by this factor to
#: certify. Measured on the first real certification attempt, the absolute
#: joint gate alone did not discriminate at all: the trained policy scored
#: 0.1685 rad, the same rollout played BACKWARDS scored 0.1691, and simply
#: holding the rollout's time-averaged pose scored 0.1624 — better than the
#: policy. A mean-absolute-error threshold is blind to temporal structure,
#: so on a clip whose joint excursions are small relative to the threshold,
#: standing still passes. Requiring the policy to beat "hold one pose"
#: turns the gate back into a test of tracking rather than of posture.
STATIC_BASELINE_RATIO_MAX = 0.80

#: Below this the reference has effectively no joint motion to track, so the
#: static-baseline comparison is vacuous (a constant reference IS tracked by
#: a constant pose) and is skipped rather than failing the clip.
MIN_REFERENCE_MOTION_RAD = 0.02

#: Reward-shaping weights for the generated tracking reward's Gaussian
#: kernels: `exp(-w * err**2)`. Chosen so a "close" pose (few-degree
#: joint error, few-cm root error) scores near 1.0 while a "way off"
#: pose (tens of degrees, tens of cm) decays toward 0 — the standard
#: DeepMimic imitation-kernel shape, not independently tuned per clip.
JOINT_ERR_WEIGHT = 8.0
ROOT_ERR_WEIGHT = 40.0

#: Orientation kernel width, on projected-gravity error (a unit vector, so the
#: error is bounded by 2). Matches the residual generator's existing 4.0 rather
#: than being tuned here — one imitation kernel shape across both paths.
#: OGMP (2403.04205 Eq. 8) weights orientation equally with base position; this
#: reward also tracks joints, so orientation is one of three terms, not half.
ORIENTATION_ERR_WEIGHT = 4.0

#: Default training budget — small iteration count x modest steps/iter,
#: comfortably under an hour on an RTX 5070 (§mission: "1500-3000 steps
#: x 2-3 iters"). `--iterations` overrides the iteration count only;
#: `steps` here is mjlab's `max_iterations` per the adapter contract
#: (see `MjlabAdapter.train`'s docstring — `steps` IS max_iterations,
#: not env steps), so this module trains ONE adapter.train() call with
#: `steps=iterations`.
#: FALLBACK control rate for the build-time phase-clock length. The generated
#: reward prefers the `step_dt` the mjlab runner publishes per step, so this
#: only matters for an adapter that does not report it.
#:
#: Read from the task config, not inferred from a training statistic:
#: `mjlab/tasks/velocity/velocity_env_cfg.py` sets `MujocoCfg.timestep=0.005`
#: (200 Hz physics) with `decimation=4`, so the CONTROL step is 0.005 x 4 =
#: 0.02 s -> 50 Hz. See `sculptor.refs.timing` for the literature basis and
#: for why physics and control rates are separate numbers.
#:
#: Do NOT re-derive this from `__episode_length` in a reward trajectory: that
#: channel is `env.episode_length_buf` averaged over envs (_mjlab_runner:654
#: and :132), i.e. the MEAN PROGRESS of envs at uniformly distributed phases,
#: which sits near half the maximum. Reading it as an episode duration is what
#: produced a spurious "25 Hz" here.
DEFAULT_CONTROL_HZ = _timing.MJLAB_G1_VELOCITY.control_hz

DEFAULT_ITERATIONS = 3
DEFAULT_STEPS_PER_ITERATION = 2000
DEFAULT_N_EPISODES = 2
#: Phase-target downsampling: number of clocked keyframes the generated
#: reward tracks against. Independent of the clip's native frame count
#: (a long mocap clip and a short one both compress to this many
#: phase-indexed targets) — keeps the generated reward source small and
#: the phase clock (`t/T`) well-defined regardless of clip fps.
N_PHASE_TARGETS = 32

# Normal mission rewards are returned to the LLM on every edit.  Sixteen
# targets preserve the reference's phase structure while keeping the immutable
# motion prior comfortably inside the editor's output budget.  Tier-D
# certification keeps the denser 32-target default above.
REFERENCE_REWARD_PHASE_TARGETS = 16

# A locomotion prior must keep producing a gait after its source clip ends;
# clamping a non-zero velocity target against one frozen joint pose creates an
# impossible objective.  Detect a repeatable, translation-dominant suffix from
# kinematics alone (no robot/task-name keys) and loop only that suffix.  Large
# vertical motions such as get-ups and jumps deliberately stay one-shot.
_LOCOMOTION_MIN_TRAVEL_M = 0.50
_LOCOMOTION_MAX_VERTICAL_RANGE_M = 0.20
_LOOP_MIN_PERIOD_S = 0.30
_LOOP_MAX_PERIOD_S = 1.50
_LOOP_MAX_JOINT_RMS_RAD = 0.25
_LOOP_MAX_ROOT_Z_DELTA_M = 0.12
_LOOP_MAX_GRAVITY_RMS = 0.20


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


def select_tracking_phase_window(
    *,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    gravity: Optional[np.ndarray],
    fps: float,
) -> tuple[slice, str]:
    """Choose a one-shot clip or a repeatable locomotion suffix.

    The decision is embodiment-neutral and derived only from the reference:
    meaningful horizontal translation, small vertical excursion, and an
    end-pose match within a plausible gait-period lookback.  Returning a slice
    keeps every target channel aligned.  ``"hold"`` means play once and then
    hold the terminal pose with a zero velocity target; ``"loop"`` means wrap
    the selected suffix indefinitely.
    """
    q = np.asarray(joint_pos, dtype=np.float64)
    root = np.asarray(root_pos, dtype=np.float64)
    g = None if gravity is None else np.asarray(gravity, dtype=np.float64)
    if q.ndim != 2 or root.ndim != 2 or root.shape[1] < 3:
        return slice(None), "hold"
    if q.shape[0] != root.shape[0] or q.shape[0] < 4 or fps <= 0.0:
        return slice(None), "hold"

    horizontal_travel = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    vertical_range = float(np.ptp(root[:, 2]))
    if (horizontal_travel < _LOCOMOTION_MIN_TRAVEL_M
            or vertical_range > _LOCOMOTION_MAX_VERTICAL_RANGE_M):
        return slice(None), "hold"

    min_lookback = max(2, int(round(_LOOP_MIN_PERIOD_S * fps)))
    max_lookback = max(min_lookback + 1, int(round(_LOOP_MAX_PERIOD_S * fps)))
    lo = max(0, q.shape[0] - 1 - max_lookback)
    hi = q.shape[0] - min_lookback
    if hi <= lo:
        return slice(None), "hold"

    candidates = np.arange(lo, hi, dtype=np.int64)
    joint_rms = np.sqrt(np.mean((q[candidates] - q[-1]) ** 2, axis=1))
    root_delta = np.abs(root[candidates, 2] - root[-1, 2])
    if g is not None and g.shape == root.shape:
        gravity_rms = np.sqrt(np.mean((g[candidates] - g[-1]) ** 2, axis=1))
    else:
        gravity_rms = np.zeros_like(joint_rms)
    score = joint_rms + 2.0 * root_delta + 0.5 * gravity_rms
    best_local = int(np.argmin(score))
    start = int(candidates[best_local])
    suffix_travel = float(np.linalg.norm(root[-1, :2] - root[start, :2]))
    if (joint_rms[best_local] > _LOOP_MAX_JOINT_RMS_RAD
            or root_delta[best_local] > _LOOP_MAX_ROOT_Z_DELTA_M
            or gravity_rms[best_local] > _LOOP_MAX_GRAVITY_RMS
            or suffix_travel < 0.15):
        return slice(None), "hold"
    return slice(start, None), "loop"


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


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Unit gravity direction expressed in the body frame, `(N, 3)`.

    OGMP's tracking reward (arXiv 2403.04205 Eq. 8) weights orientation error
    equally with position — `0.475·e^(-5‖er_p‖) + 0.475·e^(-5‖er_o‖)` — but the
    Tier-D reward tracked joints and root height only. Retargeted clips carry
    `root_quat_wxyz`, and mjlab publishes `projected_gravity_b` every step, so
    this is the bridge between them.

    Projected gravity rather than the raw quaternion because it is what mjlab
    already observes, it is yaw-invariant (a heading offset is not an
    orientation error for a clip whose root translation was zeroed), and it is
    the standard humanoid attitude signal. Upright is `[0, 0, -1]`.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(norm > 0.0, norm, 1.0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R^T @ [0, 0, -1] — the negated third ROW of the rotation matrix.
    return np.stack([
        2.0 * (w * y - x * z),
        -2.0 * (y * z + w * x),
        2.0 * (x * x + y * y) - 1.0,
    ], axis=1)


def generate_tracking_reward_source(
    *,
    clip_id: str,
    joint_names: list[str],
    target_joint_pos: np.ndarray,
    target_root_z: np.ndarray,
    episode_len_steps: int,
    duration_s: float = 0.0,
    target_gravity: Optional[np.ndarray] = None,
    joint_err_weight: float = JOINT_ERR_WEIGHT,
    root_err_weight: float = ROOT_ERR_WEIGHT,
    orientation_err_weight: float = ORIENTATION_ERR_WEIGHT,
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

    if target_gravity is not None:
        target_gravity = np.asarray(target_gravity, dtype=np.float64)
        if target_gravity.shape != (n_phase, 3):
            raise ValueError(
                "target_gravity must be (n_phase, 3) to align with the joint "
                f"targets: {target_gravity.shape} vs {(n_phase, 3)}")

    joint_pos_literal = _format_array_literal(target_joint_pos)
    root_z_literal = _format_array_literal(target_root_z)
    names_literal = "[" + ", ".join(repr(str(n)) for n in joint_names) + "]"
    gravity_literal = (
        "np.asarray(" + _format_array_literal(target_gravity)
        + ", dtype=np.float64)" if target_gravity is not None else "None")
    # Zero weight collapses the term to a no-op for clips with no orientation
    # data, so the reward shape is identical to before for those.
    orientation_weight = orientation_err_weight if target_gravity is not None else 0.0

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
    "author": "sculptor",
    "parent_hash": None,
    # MjlabAdapter refuses a reward without this flag AND a
    # `compute_reward_batched` entry point (see the batched section below).
    "supports_batched": True,
    # Tells `_mjlab_runner` a reference motion is attached, so it narrows the
    # task-reward realism floor to hardware-safety terms. Without this the
    # task's `pose`/`upright`/gait terms compete with the very motion being
    # certified (measured: 28% of the reference's joint amplitude reproduced).
    "reference_tracking": True,
    "hyperparameters": {{
        "joint_err_weight": {joint_err_weight!r},
        "root_err_weight": {root_err_weight!r},
        "orientation_err_weight": {orientation_weight!r},
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
# The reference's true duration. The phase clock prefers wall time --
# `episode_length * step_dt`, with `step_dt` published per-step by the mjlab
# runner -- because EPISODE_LEN_STEPS has to assume a control rate at BUILD
# time and that assumption was once just the training budget (2000 PPO updates
# read as env steps). The G1 task steps at 50 Hz (physics 0.005 x decimation 4,
# see `sculptor.refs.timing`); reading step_dt removes the assumption entirely.
REFERENCE_DURATION_S = {duration_s!r}
JOINT_ERR_WEIGHT = {joint_err_weight!r}
ROOT_ERR_WEIGHT = {root_err_weight!r}
# 0.0 when the clip carries no root orientation, which makes the orientation
# term an exact no-op rather than a silently-wrong constant.
ORIENTATION_ERR_WEIGHT = {orientation_weight!r}

# Phase-indexed targets, shape (N_PHASE, N_JOINTS) / (N_PHASE,).
TARGET_JOINT_POS = np.asarray({joint_pos_literal}, dtype=np.float64).reshape(N_PHASE, N_JOINTS)
TARGET_ROOT_Z = np.asarray({root_z_literal}, dtype=np.float64)
# Unit gravity in the body frame, derived from the clip's root_quat_wxyz. Yaw-
# invariant on purpose: retargeting zeroes root translation, so a heading
# offset is not an orientation error. `None` when the clip has no quaternion.
TARGET_GRAVITY = {gravity_literal}


def _phase_index(info) -> int:
    step = int(info.get("episode_length", 0) or 0)
    step_dt = float(info.get("step_dt", 0.0) or 0.0)
    if REFERENCE_DURATION_S > 0.0 and step_dt > 0.0:
        phase = (step * step_dt) / REFERENCE_DURATION_S
    elif EPISODE_LEN_STEPS > 0:
        phase = step / float(EPISODE_LEN_STEPS)
    else:
        phase = 0.0
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
    if TARGET_GRAVITY is not None:
        gravity = np.asarray(
            next_state["projected_gravity_b"], dtype=np.float64).reshape(-1)[-3:]
        orient_err_sq = float(np.mean((gravity - TARGET_GRAVITY[i]) ** 2))
        orient_term = float(np.exp(-ORIENTATION_ERR_WEIGHT * orient_err_sq))
        components["orientation_tracking"] = orient_term
        reward += orient_term
    return float(reward), components


# ── GPU-batched entry point (mjlab) ────────────────────────────────────
# MjlabAdapter trains thousands of envs at once and REQUIRES this entry
# point; without it a Tier-D certification cannot train on the mjlab path
# at all. Same two-Gaussian formula as `compute_reward` above, vectorized.
#
# Two state-layout differences from the scalar path, both forced by mjlab:
#   * joints are the TRAILING N_JOINTS of a per-env `qpos` row, not
#     `qpos[7:7+N]` of a full MuJoCo qpos vector;
#   * root height arrives as `info["base_height"]`, and is compared as a
#     DELTA from the reference's own first frame. Absolute comparison is
#     meaningless for the origin-relative retargeted clips that make up
#     most of the library — their root_z sits near 0 while a standing G1
#     base is ~0.74 m, so an absolute error would saturate the kernel at
#     zero for every frame regardless of how well the motion tracked.
def compute_reward_batched(state, action, next_state, info):
    import torch

    del state, action
    qpos = next_state["qpos"]
    if qpos.shape[-1] < N_JOINTS:
        raise ValueError(
            f"batched qpos has {{qpos.shape[-1]}} columns, fewer than the "
            f"{{N_JOINTS}} tracked joints")
    like = qpos[:, 0]

    step = info.get("episode_length", torch.zeros_like(like))
    step_dt = info.get("step_dt", None)
    if REFERENCE_DURATION_S > 0.0 and step_dt is not None:
        phase = torch.clamp(
            (step * step_dt) / REFERENCE_DURATION_S, 0.0, 0.999999)
    elif EPISODE_LEN_STEPS > 0:
        phase = torch.clamp(step / float(EPISODE_LEN_STEPS), 0.0, 0.999999)
    else:
        phase = torch.zeros_like(like)
    i = torch.clamp((phase * N_PHASE).long(), 0, N_PHASE - 1)

    target_joint = torch.as_tensor(
        TARGET_JOINT_POS, device=qpos.device, dtype=qpos.dtype)[i]
    target_root = torch.as_tensor(
        TARGET_ROOT_Z, device=qpos.device, dtype=qpos.dtype)[i]

    joint_err = qpos[:, -N_JOINTS:] - target_joint
    joint_term = torch.exp(
        -JOINT_ERR_WEIGHT * torch.mean(joint_err ** 2, dim=-1))

    root0 = float(TARGET_ROOT_Z[0])
    base_height = info.get("base_height", torch.zeros_like(like))
    actual_delta = info.get("base_height_delta", base_height - root0)
    root_err = actual_delta - (target_root - root0)
    root_term = torch.exp(-ROOT_ERR_WEIGHT * root_err ** 2)

    total = joint_term + root_term
    components = {{
        "joint_tracking": joint_term,
        "root_tracking": root_term,
    }}
    if TARGET_GRAVITY is not None:
        target_gravity = torch.as_tensor(
            TARGET_GRAVITY, device=qpos.device, dtype=qpos.dtype)[i]
        gravity = next_state["projected_gravity_b"][:, -3:]
        orient_term = torch.exp(-ORIENTATION_ERR_WEIGHT * torch.mean(
            (gravity - target_gravity) ** 2, dim=-1))
        components["orientation_tracking"] = orient_term
        total = total + orient_term
    return total, components
'''


def generate_tracking_residual_reward_source(
    *,
    clip: dict[str, Any],
    clip_id: str,
    version: str = "v0",
    n_phase_targets: int = REFERENCE_REWARD_PHASE_TARGETS,
    residual_max: float = 0.25,
) -> str:
    """Build the tracking-FIRST stage reward used by normal mission runs.

    Unlike :func:`generate_tracking_reward_source` (the small scalar Tier-D
    certification project), this module implements both the scalar authoring
    contract and the GPU-batched mjlab contract.  The reference is converted
    through the same canonical ``clip_to_arrays`` path used by metric trust,
    downsampled into phase targets, and embedded as immutable data.  Subsequent
    LLM edits may add a bounded task residual, but the tracking target hash and
    composition contract are preserved by ``sculptor.edit``.
    """
    from sculptor.refs.convert import clip_to_arrays

    arrays, meta = clip_to_arrays(clip, n_envs=1)
    if "joint_pos" not in arrays or not meta.get("joint_names"):
        raise TrackError(
            f"clip {clip_id!r} has no joint trajectory; cannot construct "
            "a tracking-first stage reward")

    def _raw_target(name: str) -> Optional[np.ndarray]:
        arr = arrays.get(name)
        if arr is None:
            return None
        return np.asarray(arr[:, 0], dtype=np.float64)

    joint_pos_raw = _raw_target("joint_pos")
    joint_vel_raw = _raw_target("joint_vel")
    root_pos_raw = _raw_target("root_link_pos_w")
    gravity_raw = _raw_target("projected_gravity_b")
    assert (joint_pos_raw is not None and joint_vel_raw is not None
            and root_pos_raw is not None)

    fps = float(clip.get("fps") or 30.0)
    phase_window, phase_mode = select_tracking_phase_window(
        joint_pos=joint_pos_raw,
        root_pos=root_pos_raw,
        gravity=gravity_raw,
        fps=fps,
    )
    joint_pos_raw = joint_pos_raw[phase_window]
    joint_vel_raw = joint_vel_raw[phase_window]
    root_pos_raw = root_pos_raw[phase_window]
    if gravity_raw is not None:
        gravity_raw = gravity_raw[phase_window]

    joint_pos = downsample_phase_targets(joint_pos_raw, n=n_phase_targets)
    joint_vel = downsample_phase_targets(joint_vel_raw, n=n_phase_targets)
    root_pos = downsample_phase_targets(root_pos_raw, n=n_phase_targets)
    gravity = (
        downsample_phase_targets(gravity_raw, n=n_phase_targets)
        if gravity_raw is not None else None)

    # Hash exactly the rounded arrays embedded in source.  This is the durable
    # parent→child identity checked after every LLM rewrite.
    rounded_targets = {
        "joint_pos": np.round(joint_pos, 5).tolist(),
        "joint_vel": np.round(joint_vel, 5).tolist(),
        "root_z": np.round(root_pos[:, 2], 5).tolist(),
        "gravity": (
            np.round(gravity, 5).tolist() if gravity is not None else None),
    }
    target_hash = hashlib.sha256(json.dumps(
        rounded_targets, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    n_frames = int(joint_pos_raw.shape[0])
    duration_s = max(1.0 / fps, (n_frames - 1) / fps)
    names_literal = repr([str(name) for name in meta["joint_names"]])
    jp_literal = _format_array_literal(joint_pos)
    jv_literal = _format_array_literal(joint_vel)
    rz_literal = _format_array_literal(root_pos[:, 2])
    gravity_literal = (
        _format_array_literal(gravity) if gravity is not None else "None")
    n_joints = int(joint_pos.shape[1])
    orientation_weight = 0.20 if gravity is not None else 0.0
    root_weight = 0.25 + (0.20 - orientation_weight)

    return f'''"""Reference-tracking base plus bounded task residual.

Generated deterministically from clip {clip_id!r}.  REFERENCE_* data and
``_reference_tracking_*`` functions are the immutable motion prior; reward
editing may only author the small residual task term.
"""
from __future__ import annotations

import numpy as np

REFERENCE_TARGET_SHA256 = {target_hash!r}
REFERENCE_JOINT_NAMES = {names_literal}
REFERENCE_N_PHASES = {n_phase_targets}
REFERENCE_N_JOINTS = {n_joints}
REFERENCE_DURATION_S = {duration_s!r}
REFERENCE_PHASE_MODE = {phase_mode!r}
REFERENCE_JOINT_POS = np.asarray({jp_literal}, dtype=np.float64)
REFERENCE_JOINT_VEL = np.asarray({jv_literal}, dtype=np.float64)
REFERENCE_ROOT_Z = np.asarray({rz_literal}, dtype=np.float64)
REFERENCE_GRAVITY = {'np.asarray(' + gravity_literal + ', dtype=np.float64)' if gravity is not None else 'None'}

REWARD_SPEC: dict = {{
    "version": {version!r},
    "description": "Tracking-first reward generated from the attached reference; "
                   "task-specific residual starts at zero.",
    "author": "sculptor",
    "parent_hash": None,
    "supports_batched": True,
    # Tells `_mjlab_runner` a reference motion is attached, so it narrows the
    # task-reward realism floor to hardware-safety terms. Without this the
    # task's `pose`/`upright`/gait terms compete with the very motion being
    # certified (measured: 28% of the reference's joint amplitude reproduced).
    "reference_tracking": True,
    "composition": {{
        "type": "reference_tracking_residual",
        "reference_clip_id": {clip_id!r},
        "reference_target_sha256": {target_hash!r},
        "tracking_weight": 1.0,
        "residual_max": {float(residual_max)!r},
        "phase_mode": {phase_mode!r},
        "phase_duration_s": {duration_s!r},
        "root_height_frame": "episode_relative",
    }},
    "hyperparameters": {{
        "tracking_weight": 1.0,
        "residual_max": {float(residual_max)!r},
        "joint_position_kernel": 8.0,
        "joint_velocity_kernel": 0.10,
        "root_height_kernel": 40.0,
        "orientation_kernel": 4.0,
        "alive_bonus": 0.05,
    }},
    "grounding": {{
        "tracking_weight": "Reference motion is the structural objective; normalized tracking channels sum to one.",
        "residual_max": "Residual is capped at 25% of the unit tracking base so it cannot replace the motion prior.",
        "joint_position_kernel": "DeepMimic-style Gaussian tracking kernel over measured reference joint positions.",
        "joint_velocity_kernel": "Velocity errors are in rad/s and need a wider Gaussian than position errors.",
        "root_height_kernel": "A 0.1 m root-height miss decays the channel materially without forming a hard cliff.",
        "orientation_kernel": "Projected-gravity error measures body orientation without yaw assumptions.",
        "alive_bonus": "Small positive floor prevents termination-seeking while remaining below task tracking signal.",
    }},
    "references": [],
}}

_W_JOINT_POS = 0.40
_W_JOINT_VEL = 0.15
_W_ROOT = {root_weight!r}
_W_ORIENTATION = {orientation_weight!r}
_TRACKING_WEIGHT = 1.0
_RESIDUAL_MAX = {float(residual_max)!r}
_ALIVE_BONUS = 0.05


def _scalar(info, key, default=0.0):
    value = info.get(key, default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _phase_index_scalar(info):
    elapsed = _scalar(info, "episode_length") * _scalar(info, "step_dt", 0.02)
    if REFERENCE_PHASE_MODE == "loop":
        fraction = (max(elapsed, 0.0) % REFERENCE_DURATION_S) / REFERENCE_DURATION_S
    else:
        fraction = min(max(elapsed / REFERENCE_DURATION_S, 0.0), 0.999999)
    return min(REFERENCE_N_PHASES - 1, int(fraction * REFERENCE_N_PHASES))


def _reference_tracking_numpy(next_state, info):
    qpos = np.asarray(next_state["qpos"], dtype=np.float64).reshape(-1)
    qvel = np.asarray(next_state["qvel"], dtype=np.float64).reshape(-1)
    if qpos.size < REFERENCE_N_JOINTS or qvel.size < REFERENCE_N_JOINTS:
        raise ValueError("state has fewer joints than the attached reference")
    i = _phase_index_scalar(info)
    pos_err = qpos[-REFERENCE_N_JOINTS:] - REFERENCE_JOINT_POS[i]
    elapsed = _scalar(info, "episode_length") * _scalar(info, "step_dt", 0.02)
    target_vel = REFERENCE_JOINT_VEL[i]
    if REFERENCE_PHASE_MODE == "hold" and elapsed >= REFERENCE_DURATION_S:
        target_vel = np.zeros_like(target_vel)
    vel_err = qvel[-REFERENCE_N_JOINTS:] - target_vel
    reference_root_delta = float(REFERENCE_ROOT_Z[i] - REFERENCE_ROOT_Z[0])
    base_height = _scalar(info, "base_height")
    actual_root_delta = _scalar(
        info, "base_height_delta", base_height - float(REFERENCE_ROOT_Z[0]))
    root_err = actual_root_delta - reference_root_delta
    joint_pos = float(np.exp(-8.0 * np.mean(pos_err ** 2)))
    joint_vel = float(np.exp(-0.10 * np.mean(vel_err ** 2)))
    root_height = float(np.exp(-40.0 * root_err ** 2))
    orientation = 1.0
    if REFERENCE_GRAVITY is not None:
        gravity = np.asarray(
            next_state["projected_gravity_b"], dtype=np.float64).reshape(-1)[-3:]
        orientation = float(np.exp(-4.0 * np.mean(
            (gravity - REFERENCE_GRAVITY[i]) ** 2)))
    base = (_W_JOINT_POS * joint_pos + _W_JOINT_VEL * joint_vel
            + _W_ROOT * root_height + _W_ORIENTATION * orientation)
    return base, joint_pos, joint_vel, root_height, orientation


def _residual_task_numpy(state, action, next_state, info):
    """Editable task hook. Return raw residual credit; wrapper bounds it."""
    del state, action, next_state, info
    return 0.0


def compute_reward(state, action, next_state, info):
    base, joint_pos, joint_vel, root_height, orientation = (
        _reference_tracking_numpy(next_state, info))
    residual_task = float(np.clip(
        _residual_task_numpy(state, action, next_state, info),
        0.0, _RESIDUAL_MAX))
    alive = _ALIVE_BONUS
    not_fallen = 1.0 - min(max(_scalar(info, "fallen"), 0.0), 1.0)
    total = (_TRACKING_WEIGHT * base + residual_task + alive) * not_fallen
    return float(total), {{
        "reference_tracking": float(base),
        "tracking_joint_pos": joint_pos,
        "tracking_joint_vel": joint_vel,
        "tracking_root_height": root_height,
        "tracking_orientation": orientation,
        "residual_task": residual_task,
        "alive_bonus": alive * not_fallen,
    }}


def _phase_index_batched(info, like):
    import torch
    step = info.get("episode_length", torch.zeros_like(like))
    dt = info.get("step_dt", torch.full_like(like, 0.02))
    elapsed = torch.clamp(step * dt, min=0.0)
    if REFERENCE_PHASE_MODE == "loop":
        fraction = torch.remainder(elapsed, REFERENCE_DURATION_S) / REFERENCE_DURATION_S
    else:
        fraction = torch.clamp(elapsed / REFERENCE_DURATION_S, 0.0, 0.999999)
    return torch.clamp(
        (fraction * REFERENCE_N_PHASES).long(), 0, REFERENCE_N_PHASES - 1)


def _reference_tracking_batched(next_state, info):
    import torch
    qpos = next_state["qpos"]
    qvel = next_state["qvel"]
    if qpos.shape[-1] < REFERENCE_N_JOINTS or qvel.shape[-1] < REFERENCE_N_JOINTS:
        raise ValueError("state has fewer joints than the attached reference")
    like = qpos[:, 0]
    i = _phase_index_batched(info, like)
    target_pos = torch.as_tensor(
        REFERENCE_JOINT_POS, device=qpos.device, dtype=qpos.dtype)[i]
    target_vel = torch.as_tensor(
        REFERENCE_JOINT_VEL, device=qvel.device, dtype=qvel.dtype)[i]
    elapsed = info.get("episode_length", torch.zeros_like(like)) * info.get(
        "step_dt", torch.full_like(like, 0.02))
    if REFERENCE_PHASE_MODE == "hold":
        target_vel = torch.where(
            (elapsed >= REFERENCE_DURATION_S)[:, None],
            torch.zeros_like(target_vel), target_vel)
    pos_err = qpos[:, -REFERENCE_N_JOINTS:] - target_pos
    vel_err = qvel[:, -REFERENCE_N_JOINTS:] - target_vel
    joint_pos = torch.exp(-8.0 * torch.mean(pos_err ** 2, dim=-1))
    joint_vel = torch.exp(-0.10 * torch.mean(vel_err ** 2, dim=-1))
    target_root = torch.as_tensor(
        REFERENCE_ROOT_Z, device=qpos.device, dtype=qpos.dtype)[i]
    reference_root_delta = target_root - float(REFERENCE_ROOT_Z[0])
    base_height = info.get("base_height", torch.zeros_like(like))
    actual_root_delta = info.get(
        "base_height_delta", base_height - float(REFERENCE_ROOT_Z[0]))
    root_height = torch.exp(-40.0 * (
        actual_root_delta - reference_root_delta) ** 2)
    orientation = torch.ones_like(like)
    if REFERENCE_GRAVITY is not None:
        target_gravity = torch.as_tensor(
            REFERENCE_GRAVITY, device=qpos.device, dtype=qpos.dtype)[i]
        orientation = torch.exp(-4.0 * torch.mean((
            next_state["projected_gravity_b"][:, -3:] - target_gravity) ** 2,
            dim=-1))
    base = (_W_JOINT_POS * joint_pos + _W_JOINT_VEL * joint_vel
            + _W_ROOT * root_height + _W_ORIENTATION * orientation)
    return base, joint_pos, joint_vel, root_height, orientation


def _residual_task_batched(state, action, next_state, info, like):
    """Editable task hook. Return raw per-env credit; wrapper bounds it."""
    import torch
    del state, action, next_state, info
    return torch.zeros_like(like)


def compute_reward_batched(state, action, next_state, info):
    import torch
    base, joint_pos, joint_vel, root_height, orientation = (
        _reference_tracking_batched(next_state, info))
    residual_task = torch.clamp(
        _residual_task_batched(state, action, next_state, info, base),
        0.0, _RESIDUAL_MAX)
    alive = torch.full_like(base, _ALIVE_BONUS)
    not_fallen = 1.0 - torch.clamp(
        info.get("fallen", torch.zeros_like(base)), 0.0, 1.0)
    total = (_TRACKING_WEIGHT * base + residual_task + alive) * not_fallen
    return total, {{
        "reference_tracking": base,
        "tracking_joint_pos": joint_pos,
        "tracking_joint_vel": joint_vel,
        "tracking_root_height": root_height,
        "tracking_orientation": orientation,
        "residual_task": residual_task,
        "alive_bonus": alive * not_fallen,
    }}
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
    #: Which convention `root_z_rmse_m` was measured in — "absolute" (world
    #: heights compared directly) or "origin_relative" (vertical excursions
    #: compared, each trace referenced to its own first frame).
    root_frame: str = "absolute"
    #: The constant world-height difference between the two traces' first
    #: frames. Under "origin_relative" this is the frame offset that was
    #: DIVIDED OUT and is not part of `root_z_rmse_m`; it is reported so a
    #: certificate never hides what it chose not to measure.
    root_z_offset_m: float = 0.0
    #: What the BEST CONSTANT POSE would have scored — the rollout's own
    #: time-averaged pose held for the whole clip. This is the "policy did
    #: nothing" control; `mean_joint_err_rad` must beat it. Defaults to 0.0
    #: meaning "no static control applies" (root-only scoring, or a caller
    #: that predates the field), which skips the comparison the same way a
    #: motionless reference does — `compute_tracking_errors` always supplies
    #: a real value when there are joints to compare.
    static_baseline_err_rad: float = 0.0
    #: How much of the reference's joint motion the rollout actually
    #: reproduced (std over time, rollout / clip). Informational.
    motion_ratio: float = 0.0
    #: RMS error between the rollout's body-frame gravity and the clip's, in
    #: units of a unit vector (so 0 = attitude matched, 2 = inverted). OGMP
    #: (2403.04205 Eq. 8) weights orientation equally with base position, and
    #: this reward now tracks it — but it is deliberately MEASURED, NOT GATED:
    #: nothing has ever passed Tier-D, so there is no evidence for an
    #: achievable threshold and inventing one would be a made-up number.
    #: Gate it once a certified run establishes the range. 0.0 when the clip
    #: has no quaternion or the rollout recorded no gravity channel.
    orientation_err: float = 0.0

    @property
    def beats_static_baseline(self) -> bool:
        """Did the policy do better than holding one pose? Vacuous — and so
        skipped — for a reference with no joint motion to track."""
        if not np.isfinite(self.static_baseline_err_rad):
            return False
        if self.static_baseline_err_rad < MIN_REFERENCE_MOTION_RAD:
            return True
        return (self.mean_joint_err_rad
                <= self.static_baseline_err_rad * STATIC_BASELINE_RATIO_MAX)

    @property
    def feasible(self) -> bool:
        return (
            self.mean_joint_err_rad < MEAN_JOINT_ERR_THRESHOLD_RAD
            and self.root_z_rmse_m < ROOT_Z_RMSE_THRESHOLD_M
            and self.beats_static_baseline
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_joint_err_rad": round(self.mean_joint_err_rad, 6),
            "max_joint_err_rad": round(self.max_joint_err_rad, 6),
            "root_z_rmse_m": round(self.root_z_rmse_m, 6),
            "duration_coverage": round(self.duration_coverage, 6),
            "orientation_err": round(self.orientation_err, 6),
            "common_joint_names": list(self.common_joint_names),
            "n_common_joints": self.n_common_joints,
            "root_frame": self.root_frame,
            "root_z_offset_m": round(self.root_z_offset_m, 6),
            "static_baseline_err_rad": (
                round(self.static_baseline_err_rad, 6)
                if np.isfinite(self.static_baseline_err_rad) else None),
            "beats_static_baseline": self.beats_static_baseline,
            "motion_ratio": round(self.motion_ratio, 6),
            "feasible": self.feasible,
            "thresholds": {
                "mean_joint_err_rad": MEAN_JOINT_ERR_THRESHOLD_RAD,
                "root_z_rmse_m": ROOT_Z_RMSE_THRESHOLD_M,
                "static_baseline_ratio_max": STATIC_BASELINE_RATIO_MAX,
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


#: `env_spec`'s validated bounds for `episode_length_s` (see
#: `sculptor.env_spec`). A clip outside this window cannot cap its episode.
_EPISODE_LENGTH_S_BOUNDS = (2.0, 60.0)


def _cap_episode_to_reference(env_dir: Path, duration_s: float) -> bool:
    """Cap the training episode to the reference's own duration.

    Without this the episode runs its task default (~10 s) while the phase
    clock finishes in `duration_s`, so the tail of every episode is the robot
    holding the reference's last frame — on the 3.70 s composite that was 62%
    of each episode. Worse, `compute_tracking_errors` index-aligns the WHOLE
    rollout against the WHOLE clip, so a rollout 2.7x longer than the
    reference compares mismatched phases and the certificate is meaningless.

    Returns whether the cap was applied; a clip outside `env_spec`'s validated
    `episode_length_s` window is left alone rather than written invalid."""
    lo, hi = _EPISODE_LENGTH_S_BOUNDS
    if not (lo <= duration_s <= hi):
        return False
    spec_path = env_dir / "current.json"
    if not spec_path.is_file():
        return False
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    spec.setdefault("shared", {})["episode_length_s"] = round(float(duration_s), 4)

    from sculptor.env_spec import validate_env_spec

    if validate_env_spec(spec):
        return False  # never write a spec the validator rejects
    payload = json.dumps(spec, indent=2, sort_keys=True)
    spec_path.write_text(payload, encoding="utf-8")
    # `apply_reference_rsi` writes the versioned copy alongside current.json;
    # keep them identical so the run's recorded spec matches what it ran.
    versioned = env_dir / f"{spec.get('meta', {}).get('version', 'v0')}.json"
    if versioned.is_file():
        versioned.write_text(payload, encoding="utf-8")
    return True


def clip_root_frame(clip: dict[str, Any]) -> str:
    """Whether `clip["root_pos_z"]` is a world height or an origin-relative
    excursion. Returns `"absolute"` or `"origin_relative"`.

    An explicit `clip["root_frame"]` wins; otherwise the height band decides
    (see `ORIGIN_RELATIVE_MAX_ROOT_Z_M`). Unknown explicit values fall
    through to the heuristic rather than raising — scoring a rollout must
    never fail closed on a metadata typo."""
    stated = clip.get("root_frame")
    if stated in ("absolute", "origin_relative"):
        return str(stated)
    z = np.asarray(clip.get("root_pos_z", ()), dtype=np.float64)
    if z.size == 0:
        return "absolute"
    return ("origin_relative" if float(np.max(z)) < ORIGIN_RELATIVE_MAX_ROOT_Z_M
            else "absolute")


def compute_tracking_errors(
    *,
    clip: dict[str, Any],
    rollout_joint_pos: np.ndarray,
    rollout_root_z: np.ndarray,
    rollout_joint_names: list[str],
    rollout_gravity: Optional[np.ndarray] = None,
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
    # No common joints -> root-only scoring; there is no joint trace to
    # compare against a static pose, so the control is vacuously satisfied.
    static_err = 0.0
    motion_ratio = 0.0
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
            # The control: what the policy would have scored by holding its
            # own time-averaged pose for the whole clip. If it cannot beat
            # that, it did not track anything.
            static_pose = np.mean(rollout_at_n, axis=0, keepdims=True)
            static_err = float(np.mean(np.abs(clip_at_n - static_pose)))
            clip_motion = float(np.mean(np.std(clip_at_n, axis=0)))
            rollout_motion = float(np.mean(np.std(rollout_at_n, axis=0)))
            motion_ratio = (rollout_motion / clip_motion) if clip_motion > 0 else 0.0

    # Root height must be scored in the SAME frame the reward optimized, or
    # certification measures something the policy was never trained to do.
    # `compute_reward_batched` compares mjlab's `base_height_delta` (measured
    # from each env's own reset anchor) against the reference's excursion from
    # ITS first frame. For an origin-relative clip the two traces therefore
    # differ by a constant ~0.74 m frame offset that is not tracking error at
    # all; comparing them raw made `root_z_rmse_m < 0.12` unsatisfiable for the
    # 96% of the library that is origin-relative, and no clip had ever reached
    # tier D. The offset is divided out here and reported separately as
    # `root_z_offset_m` rather than silently dropped. Absolute clips keep the
    # direct comparison — there a constant offset IS a real tracking error.
    root_frame = clip_root_frame(clip)
    n = min(t_rollout, t_clip) if t_clip > 0 and t_rollout > 0 else 0
    root_offset = 0.0
    if n > 0:
        clip_z_at_n = downsample_phase_targets(clip_root_z, n=n)
        rollout_z_at_n = downsample_phase_targets(rollout_root_z, n=n)
        root_offset = float(rollout_z_at_n[0] - clip_z_at_n[0])
        if root_frame == "origin_relative":
            clip_z_at_n = clip_z_at_n - clip_z_at_n[0]
            rollout_z_at_n = rollout_z_at_n - rollout_z_at_n[0]
        root_rmse = float(np.sqrt(np.mean((clip_z_at_n - rollout_z_at_n) ** 2)))
    else:
        root_rmse = float("inf")

    # Orientation, per OGMP Eq. 8. Measured only — see `TrackingErrors.
    # orientation_err` for why this does not gate.
    orientation_err = 0.0
    clip_quat = clip.get("root_quat_wxyz")
    if clip_quat is not None and rollout_gravity is not None:
        roll_g = np.asarray(rollout_gravity, dtype=np.float64).reshape(
            -1, 3) if np.asarray(rollout_gravity).size else np.zeros((0, 3))
        clip_g = projected_gravity_from_quat(
            np.asarray(clip_quat, dtype=np.float64))
        m = min(roll_g.shape[0], clip_g.shape[0])
        if m > 0:
            diff = (downsample_phase_targets(clip_g, n=m)
                    - downsample_phase_targets(roll_g, n=m))
            orientation_err = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return TrackingErrors(
        mean_joint_err_rad=mean_err,
        max_joint_err_rad=max_err,
        root_z_rmse_m=root_rmse,
        duration_coverage=duration_coverage,
        common_joint_names=common_names,
        n_common_joints=len(common_names),
        root_frame=root_frame,
        root_z_offset_m=root_offset,
        static_baseline_err_rad=static_err,
        motion_ratio=motion_ratio,
        orientation_err=orientation_err,
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
    control_hz: float = DEFAULT_CONTROL_HZ,
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

    # The phase clock must advance with WALL TIME, not with the training
    # budget. This used to read `int(steps_per_iteration)`, but per this
    # module's own docstring `steps` IS mjlab's `max_iterations` — a count of
    # PPO updates, not env steps. With the default 2000 against ~500-step
    # episodes the reference played at a quarter speed and the policy never
    # saw past phase 0.25, i.e. never reached the jump or the kick of a
    # three-phase composite. Deriving the clock from the clip's real duration
    # makes the reference play at true speed.
    fps = float(clip.get("fps") or 0.0) or 30.0
    n_frames = int(np.asarray(clip["root_pos_z"]).shape[0])
    duration_s = n_frames / fps if fps > 0 else 0.0
    episode_len_steps = max(1, int(round(duration_s * float(control_hz))))
    target_joint_pos = downsample_phase_targets(
        np.asarray(clip["joint_pos"], dtype=np.float64), n=n_phase_targets)
    target_root_z = downsample_phase_targets(
        np.asarray(clip["root_pos_z"], dtype=np.float64), n=n_phase_targets)
    # Orientation, per OGMP Eq. 8. Downsample the derived gravity rather than
    # the quaternion: averaging quaternion components across a phase window is
    # not a rotation, while averaging unit gravity vectors is a well-defined
    # (if approximate) direction. Clips without a quaternion get None, which
    # zeroes the term rather than fabricating an upright target.
    quat = clip.get("root_quat_wxyz")
    target_gravity = None
    if quat is not None:
        target_gravity = downsample_phase_targets(
            projected_gravity_from_quat(np.asarray(quat, dtype=np.float64)),
            n=n_phase_targets)
        # Already unit — `projected_gravity_from_quat` normalizes and
        # `downsample_phase_targets` selects nearest frames rather than
        # interpolating. Re-normalizing is a cheap guard that keeps the
        # invariant true if either of those ever changes: mjlab's observed
        # `projected_gravity_b` is unit, and a shrunken target would charge a
        # standing error against a perfectly upright robot.
        norm = np.linalg.norm(target_gravity, axis=1, keepdims=True)
        target_gravity = target_gravity / np.where(norm > 0.0, norm, 1.0)

    # Say out loud whether this reference is even representable at the task's
    # control rate. Both Tier-D timing failures were silent; a phase clock that
    # cannot visit all its targets, or a reference with content above Nyquist,
    # should be visible before the GPU time is spent rather than inferred from
    # a flat learning curve afterwards.
    timing_findings = _timing.validate_timing(
        _timing.MJLAB_G1_VELOCITY,
        reference_fps=fps,
        reference_duration_s=duration_s,
        n_phase_targets=n_phase_targets)
    for finding in timing_findings:
        print(f"[track] timing: {finding}", file=sys.stderr, flush=True)

    reward_source = generate_tracking_reward_source(
        clip_id=clip_id,
        joint_names=joint_names,
        target_joint_pos=target_joint_pos,
        target_root_z=target_root_z,
        episode_len_steps=episode_len_steps,
        duration_s=duration_s,
        target_gravity=target_gravity,
    )
    rewards_dir = project_dir / "rewards"
    rewards_dir.mkdir(parents=True, exist_ok=True)
    (rewards_dir / "__init__.py").write_text("", encoding="utf-8")
    reward_path = rewards_dir / "current.py"
    reward_path.write_text(reward_source, encoding="utf-8")

    env_dir = project_dir / "env"
    apply_reference_rsi(env_dir, clip)
    _cap_episode_to_reference(env_dir, duration_s)
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
    library helper needed.

    §audit-finding close (REFERENCE_BUILD_LOG.md "Audit findings
    deferred" — Tier-D spoofing): the `tierD` block also records
    `clip_content_sha256` (a copy of THIS provenance's `content_sha256`
    at tracking time) and, when feasible, `rollout_sha256` (sha256 of the
    copied rollout artifact's bytes). Together these let
    `verify_tierd_certificate` bind a later "tier D" claim to a
    consistent on-disk artifact chain instead of trusting the `tier`
    field or `tierD.errors.feasible` bool in isolation. Hashing the
    rollout is best-effort (`OSError` -> `rollout_sha256` omitted, never
    raised) so a caller that passes a `rollout_path` which doesn't
    actually exist on disk (e.g. an offline unit test) still gets a
    recorded verdict — `verify_tierd_certificate` treats a missing hash
    as an unverifiable (not fatally-erroring) certificate."""
    from sculptor.refs import library

    prov = library.read_provenance(robot, clip_id, root=root)
    tier_d_block: dict[str, Any] = {
        "tracked_at": library._utc_now_iso(),
        "iterations": iterations,
        "errors": errors.to_dict(),
        "clip_content_sha256": prov.get("content_sha256"),
    }
    if errors.feasible:
        prov["tier"] = "D"
        if rollout_path is not None:
            tier_d_block["rollout_path"] = str(rollout_path)
            try:
                tier_d_block["rollout_sha256"] = library.content_sha256(
                    Path(rollout_path).read_bytes())
            except OSError:
                pass  # artifact unreadable — verify_tierd_certificate will deny cleanly
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
        # Optional: older trajectories predate the channel, and orientation is
        # measured rather than gated, so its absence must not fail a run.
        rollout_gravity = (
            npz["projected_gravity_b"][:, 0, :]
            if "projected_gravity_b" in npz.files else None)

    from sculptor.eval.robot_manifest import robot_joint_names

    rollout_joint_names = robot_joint_names(robot) or plan.joint_names
    errors = compute_tracking_errors(
        clip=clip, rollout_joint_pos=rollout_joint_pos,
        rollout_root_z=rollout_root_z, rollout_joint_names=rollout_joint_names,
        rollout_gravity=rollout_gravity)
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


# ── §REFERENCE_TRAJECTORY_PLAN §6/§10 audit-finding close: verified certs ──
#
# REFERENCE_BUILD_LOG.md "Audit findings deferred": `calibrate_metric_
# against_reference`'s `tier` argument used to come straight from the
# caller (user-writable) — nothing stopped a caller from claiming
# `tier="D"` for a clip that was never tracked. `verify_tierd_certificate`
# is the single choke point that re-derives "is this REALLY Tier D" from
# disk, so a caller can never manufacture steer-rights by passing a
# string.
@dataclass(frozen=True)
class TierDCertificate:
    """The verified facts backing a Tier-D claim for one library clip —
    returned ONLY by `verify_tierd_certificate` after every check in its
    docstring passes. Downstream code (`calibrate_metric_against_
    reference`) may treat the mere EXISTENCE of one of these as "this
    clip is genuinely Tier D"; it must never construct one itself.

    Residual trust assumption (state this honestly, do not oversell it):
    this binds the D-claim to a CONSISTENT on-disk artifact chain
    (provenance.tierD block + a rollout file whose bytes hash to the
    recorded value + a clip content hash that hasn't drifted since
    tracking) — it does NOT cryptographically prevent a determined local
    attacker who can edit provenance.json AND replace the rollout file
    AND recompute matching hashes by hand. That is an ACCEPTABLE gap: the
    threat model here is accidental/stale/hand-edited provenance (e.g. a
    clip re-ingested after certification without re-tracking, or a
    hopeful caller typing `tier="D"`), not an adversary with local disk
    write access and the will to forge a hash chain."""

    robot: str
    clip_id: str
    tracked_at: str
    iterations: int
    mean_joint_err_rad: float
    max_joint_err_rad: float
    root_z_rmse_m: float
    rollout_path: Path
    rollout_sha256: str
    clip_content_sha256: str


def verify_tierd_certificate(
    robot: str,
    clip_id: str,
    *,
    root: Optional[Path] = None,
) -> tuple[Optional[TierDCertificate], Optional[str]]:
    """Re-derive, from disk, whether `robot/clip_id` genuinely earned
    Tier D — never trust a caller-supplied tier string or an in-memory
    claim. Returns `(certificate, None)` when EVERY check below passes,
    else `(None, reason)` with a short human-readable reason (never
    raises — a missing/corrupt/tampered clip is a normal "not
    certified" verdict, exactly like `track_clip`'s own philosophy that
    an infeasible tracking run is a verdict, not an error).

    Checks (ALL required):
      1. `provenance.json` has a `tierD` block with numeric
         `errors.mean_joint_err_rad` / `errors.root_z_rmse_m`.
      2. Those RAW recorded stats are within track.py's OWN feasibility
         thresholds (`MEAN_JOINT_ERR_THRESHOLD_RAD`, `ROOT_Z_RMSE_
         THRESHOLD_M`) — recomputed here, never trusting a stored
         `feasible` bool or the top-level `tier` field in isolation (an
         edited/hand-set `tier="D"` with stale or absent stats fails
         this check).
      3. `provenance.tier == "D"`.
      4. `tierD.rollout_path` RESOLVES INSIDE the library root (`root`,
         defaulting to `library.references_root()`) — rejects a
         hand-edited provenance.json that points the artifact at an
         arbitrary path elsewhere on disk (§F7) — AND the file exists
         on disk AND its SHA-256 matches `tierD.rollout_sha256` — the
         tracking-rollout artifact is what it claims to be.
      5. `tierD.clip_content_sha256` (recorded at tracking time) matches
         the clip's CURRENT `provenance.content_sha256` — the clip was
         not re-ingested/edited after certification without re-tracking.
      6. §F7 (adversarial-audit finding): checks 5 above only compares
         two FIELDS of the SAME provenance.json — a provenance.json
         hand-edited to carry matching-but-wrong hashes in both fields
         would sail through it. This additionally recomputes the
         sha256 of the CURRENT `clip.npz` bytes on disk and requires it
         to equal `tierD.clip_content_sha256` (which, having passed
         check 5, also equals `provenance.content_sha256`) — binding
         the claim to the real clip bytes, not just internally
         self-consistent metadata. A missing/unreadable clip.npz is a
         distinct denial reason, not a crash.

    See `TierDCertificate`'s docstring for the residual trust assumption
    this does and does not cover."""
    from sculptor.refs import library

    prefix = f"{robot}/{clip_id}"
    try:
        prov = library.read_provenance(robot, clip_id, root=root)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return None, f"{prefix}: cannot read provenance: {type(e).__name__}: {e}"

    tier_d = prov.get("tierD")
    if not isinstance(tier_d, dict):
        return None, f"{prefix}: provenance has no tierD block (never tracked)"

    errors_block = tier_d.get("errors")
    if not isinstance(errors_block, dict):
        return None, f"{prefix}: tierD block has no errors stats"
    try:
        mean_joint_err = float(errors_block["mean_joint_err_rad"])
        max_joint_err = float(errors_block.get("max_joint_err_rad", mean_joint_err))
        root_z_rmse = float(errors_block["root_z_rmse_m"])
    except (KeyError, TypeError, ValueError) as e:
        return None, f"{prefix}: tierD.errors missing/invalid numeric stats: {e}"

    # Check 2: recompute feasibility from the RAW stats — never trust a
    # stored `feasible` bool, which could be hand-edited independent of
    # the underlying numbers ("edited tier" tamper).
    if not (mean_joint_err < MEAN_JOINT_ERR_THRESHOLD_RAD
            and root_z_rmse < ROOT_Z_RMSE_THRESHOLD_M):
        return None, (
            f"{prefix}: recorded tracking error is out of tolerance "
            f"(mean_joint_err_rad={mean_joint_err} >= "
            f"{MEAN_JOINT_ERR_THRESHOLD_RAD}? "
            f"{mean_joint_err >= MEAN_JOINT_ERR_THRESHOLD_RAD}; "
            f"root_z_rmse_m={root_z_rmse} >= {ROOT_Z_RMSE_THRESHOLD_M}? "
            f"{root_z_rmse >= ROOT_Z_RMSE_THRESHOLD_M}) — not a valid "
            "Tier-D certificate")

    if prov.get("tier") != "D":
        return None, f"{prefix}: provenance.tier is {prov.get('tier')!r}, not 'D'"

    rollout_path_str = tier_d.get("rollout_path")
    if not rollout_path_str:
        return None, f"{prefix}: tierD block has no rollout_path recorded"
    rollout_path = Path(rollout_path_str)

    # §F7 containment check: `rollout_path` must resolve INSIDE the
    # library root — reject a hand-edited provenance.json that points
    # this at an arbitrary file elsewhere on disk (path traversal via
    # `../`, an absolute path outside root, or a symlink escape).
    # Resolution (not just string prefixing) so `..` segments and
    # symlinks are normalized before the containment check.
    effective_root = Path(root) if root is not None else library.references_root()
    try:
        resolved_root = effective_root.resolve()
        resolved_rollout = rollout_path.resolve()
    except OSError as e:
        return None, f"{prefix}: cannot resolve rollout_path/library root: {e}"
    if resolved_rollout != resolved_root and not resolved_rollout.is_relative_to(
            resolved_root):
        return None, (
            f"{prefix}: tierD.rollout_path {rollout_path} resolves outside "
            f"the library root {effective_root} — refusing (path traversal)")

    if not rollout_path.is_file():
        return None, (
            f"{prefix}: tracking-rollout artifact missing on disk: "
            f"{rollout_path}")

    recorded_rollout_sha = tier_d.get("rollout_sha256")
    if not recorded_rollout_sha:
        return None, f"{prefix}: tierD block has no rollout_sha256 recorded"
    try:
        actual_rollout_sha = library.content_sha256(rollout_path.read_bytes())
    except OSError as e:
        return None, f"{prefix}: cannot read rollout artifact: {e}"
    if actual_rollout_sha != recorded_rollout_sha:
        return None, (
            f"{prefix}: rollout artifact sha256 mismatch (recorded "
            f"{recorded_rollout_sha[:12]}…, actual {actual_rollout_sha[:12]}…)")

    recorded_clip_sha = tier_d.get("clip_content_sha256")
    current_clip_sha = prov.get("content_sha256")
    if not recorded_clip_sha or not current_clip_sha:
        return None, f"{prefix}: missing clip content hash for staleness check"
    if recorded_clip_sha != current_clip_sha:
        return None, (
            f"{prefix}: clip content hash drift — provenance.content_sha256 "
            f"({current_clip_sha[:12]}…) does not match the hash recorded "
            f"at tracking time ({recorded_clip_sha[:12]}…); the clip was "
            "likely re-ingested/edited after certification without "
            "re-tracking")

    # Check 6 (§F7): re-derive from the REAL clip.npz bytes on disk —
    # checks above only compared two fields of the same provenance.json,
    # which a hand-edited file could keep mutually consistent (both
    # wrong) without ever touching the clip. This closes the loop to
    # ground truth.
    clip_path = library.clip_dir(robot, clip_id, root=root) / library.CLIP_FILENAME
    try:
        actual_clip_sha = library.content_sha256(clip_path.read_bytes())
    except OSError as e:
        return None, (
            f"{prefix}: cannot read clip.npz to verify content hash: "
            f"{type(e).__name__}: {e}")
    if actual_clip_sha != recorded_clip_sha:
        return None, (
            f"{prefix}: clip.npz on-disk bytes do not match the recorded "
            f"content hash (recorded {recorded_clip_sha[:12]}…, actual "
            f"{actual_clip_sha[:12]}…) — the clip file was modified after "
            "certification without re-tracking")

    cert = TierDCertificate(
        robot=robot,
        clip_id=clip_id,
        tracked_at=str(tier_d.get("tracked_at", "")),
        iterations=int(tier_d.get("iterations", 0) or 0),
        mean_joint_err_rad=mean_joint_err,
        max_joint_err_rad=max_joint_err,
        root_z_rmse_m=root_z_rmse,
        rollout_path=rollout_path,
        rollout_sha256=actual_rollout_sha,
        clip_content_sha256=current_clip_sha,
    )
    return cert, None
