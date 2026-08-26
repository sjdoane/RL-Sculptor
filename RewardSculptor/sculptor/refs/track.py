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
      trajectory.npz is promoted beside the clip under its SHA-256 identity
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
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import sys

import numpy as np

from sculptor.reference_clock import (
    build_reference_clock,
    reference_clock_from_reward_source,
    reference_playback_duration_s,
    reference_target_sha256,
    validate_reference_clock,
)
from sculptor.runtime_inputs import (
    capture_environment_artifacts,
    environment_artifacts_for_phase,
    validate_environment_artifacts,
)
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

# Allow the unavoidable one-control-sample endpoint quantization described by
# ``duration_coverage`` while requiring effectively the entire reference.
DURATION_COVERAGE_MIN = 0.99

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

# Relative masses for the generated Tier-D imitation terms.  The certificate
# gates joint-position and root-height tracking; projected orientation is
# useful stabilizing supervision but is deliberately measured-only.  Earlier
# equal additive masses let a policy collect most of the available reward by
# staying upright near the correct height while holding the support leg close
# to a static pose.  Give the gated joint trajectory the dominant mass while
# retaining root and orientation guidance.  These are reward-shaping values,
# never certification thresholds.  The generator normalizes them back to the
# historical perfect-return scale (2 without orientation, 3 with it), so the
# change reallocates credit without also changing PPO/value-target scale.
JOINT_TERM_SCALE = 4.0
ROOT_TERM_SCALE = 1.0
ORIENTATION_TERM_SCALE = 0.25

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

# Immutable Tier-D execution evidence.  Tier D is deliberately narrower than a
# general dynamics-feasibility claim: it proves only the gated tracking channels
# below for exact motion bytes executing through one embodiment, task,
# simulator/control cadence, environment input set, and software stack.
# Keep the schema small and explicit so target admission can compare the
# physical boundary without accidentally requiring identical PPO
# hyperparameters from the donor project.
TIER_D_EXECUTION_CONTRACT_SCHEMA = 3
TIER_D_TRUSTED_ADAPTER_CLASS = "sculptor.adapters.mjlab.MjlabAdapter"
TIER_D_CERTIFICATE_SCHEMA = "reward-sculptor-tier-d-certificate-v4"
TIER_D_PREFLIGHT_SCHEMA = "reward-sculptor-tier-d-preflight-v1"
TIER_D_DONOR_INTERFACE_SCHEMA = (
    "reward-sculptor-tier-d-donor-interface-v1"
)
TIER_D_DONOR_INTERFACE_FILENAME = "tier_d_interface_contract.json"
TIER_D_REFERENCE_CADENCE = "generated-target-control-phase-clock-v3"
REFERENCE_TARGET_SAMPLING = "nearest_frame_endpoint_inclusive"
REFERENCE_TARGET_IDENTITY_SCHEMA = "reference-tracking-target-v2"
REFERENCE_TERMINAL_HOLD = "final_pose_zero_joint_velocity"
TIER_D_RUNTIME_ARTIFACT_SCHEMA = "reward-sculptor-tier-d-runtime-artifacts-v2"
RUNNER_RUNTIME_ARTIFACT_SCHEMA = "reward-sculptor-runner-artifacts-v2"
TIER_D_CONTINUATION_SCHEMA = "reward-sculptor-tier-d-continuation-v1"
_TIER_D_VERSION_KEYS = ("torch", "mjlab", "rsl_rl", "adapter")

TIER_D_CERTIFICATION_SCOPE: dict[str, Any] = {
    "schema": "reward-sculptor-tier-d-scope-v1",
    "claim": "exact-schedule joint-position and root-height tracking",
    "gated_evidence": [
        "mean_joint_position_error",
        "root_z_rmse",
        "duration_coverage",
        "non_vacuous_reference_motion",
        "beats_static_pose_baseline",
    ],
    "measured_only": [
        "maximum_joint_position_error",
        "orientation_error",
        "motion_ratio",
    ],
    "not_certified": [
        "root_xy_tracking",
        "contact_safety",
        "collision_avoidance",
        "general_dynamics_feasibility",
    ],
}

DEFAULT_ITERATIONS = 3
DEFAULT_STEPS_PER_ITERATION = 2000
DEFAULT_N_EPISODES = 1
#: Phase-target downsampling: number of clocked keyframes the generated
#: reward tracks against. Independent of the clip's native frame count
#: (a long mocap clip and a short one both compress to this many
#: phase-indexed targets) — keeps the generated reward source small and
#: the phase clock (`t/T`) well-defined regardless of clip fps.
N_PHASE_TARGETS = 32

# A reference may influence a normal mission only through the exact phase
# schedule that earned Tier D.  The editable authoring twin keeps large tables
# out of the model context, so there is no longer a sound reason to reduce the
# live reward to a different 16-target surrogate.
REFERENCE_REWARD_PHASE_TARGETS = N_PHASE_TARGETS

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


def _canonical_sha256(value: Any) -> str:
    """SHA-256 of one JSON value under the repository's canonical encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, *, label: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise TrackError(f"cannot read {label} at {path}: {exc}") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


_SEED_APPLICATION_SCHEMA = "reward-sculptor-seed-application-v1"
_SEED_APPLICATION_KEYS = {
    "schema",
    "applied_seed",
    "python_random",
    "numpy_global",
    "torch_global",
    "env_cfg",
    "rl_cfg",
}


def _is_runtime_seed(value: Any) -> bool:
    """The exact integer domain shared by Python, NumPy, and Torch RNGs."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 2**32 - 1
    )


def _canonical_seed_application(
    value: Any,
    *,
    requested_seed: int,
) -> dict[str, Any]:
    """Validate and detach one observed, fail-closed runtime seed receipt."""
    if not _is_runtime_seed(requested_seed):
        raise TrackError(
            "requested runtime seed is outside the supported 0..4294967295 "
            "domain"
        )
    if not isinstance(value, dict) or set(value) != _SEED_APPLICATION_KEYS:
        raise TrackError("runtime seed-application receipt is non-canonical")
    if value.get("schema") != _SEED_APPLICATION_SCHEMA:
        raise TrackError("runtime seed-application schema is unsupported")
    if value.get("applied_seed") != requested_seed:
        raise TrackError("runtime applied seed differs from the request")
    for key in (
        "python_random", "numpy_global", "torch_global", "env_cfg", "rl_cfg",
    ):
        if not isinstance(value.get(key), bool):
            raise TrackError(f"runtime seed receipt {key} must be boolean")
    if not all(value[key] for key in (
        "python_random", "numpy_global", "torch_global",
    )):
        raise TrackError("runtime did not apply the seed to every core RNG")
    return json.loads(json.dumps(value, allow_nan=False))


def _canonical_application_receipt(
    value: Any,
    *,
    schema: str,
    phase: Optional[str] = None,
) -> dict[str, Any]:
    """Validate one cfg-mutation receipt and require an exact clean apply."""
    expected_keys = {"schema", "requested", "applied", "dead", "errors"}
    if phase is not None:
        expected_keys.add("phase")
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise TrackError("runtime environment-application receipt is non-canonical")
    if value.get("schema") != schema or (
        phase is not None and value.get("phase") != phase
    ):
        raise TrackError("runtime environment-application schema/phase is invalid")
    for key in ("requested", "applied", "dead", "errors"):
        items = value.get(key)
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item for item in items
        ):
            raise TrackError(f"runtime environment receipt {key} is invalid")
    if value["requested"] != sorted(value["requested"]):
        raise TrackError("runtime environment requested fields are non-canonical")
    if value["dead"] or value["errors"]:
        raise TrackError(
            "runtime did not apply every requested environment mutation: "
            f"dead={value['dead']}, errors={value['errors']}"
        )
    return json.loads(json.dumps(value, allow_nan=False))


def _resolved_adapter_signature(
    config_path: Path,
) -> tuple[dict[str, Any], set[str], bool]:
    """Resolve and type-check an adapter constructor without constructing it."""
    import importlib
    import inspect

    configured = _read_adapter_config_file(config_path)
    dotted = configured.get("class")
    if not isinstance(dotted, str) or not dotted:
        raise TrackError("adapter class must be a non-empty dotted path")
    if dotted != TIER_D_TRUSTED_ADAPTER_CLASS:
        raise TrackError(
            "Tier-D certification requires the trusted local adapter "
            f"{TIER_D_TRUSTED_ADAPTER_CLASS!r}; got {dotted!r}"
        )
    _assert_local_tierd_configuration(config_path)
    module_name, separator, class_name = dotted.rpartition(".")
    if not separator or not module_name or not class_name:
        raise TrackError(f"adapter class must be a dotted path, got {dotted!r}")
    try:
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
        from sculptor.adapters.base import SculptorAdapter

        if not isinstance(adapter_class, type) or not issubclass(
            adapter_class, SculptorAdapter,
        ):
            raise TypeError(f"{dotted!r} is not a SculptorAdapter subclass")
        signature = inspect.signature(adapter_class)
        signature.bind(**configured["config"])
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise TrackError(
            "cannot resolve Tier-D adapter/config without construction: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    parameters = set(signature.parameters)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return configured, parameters, accepts_kwargs


def _configured_environment_paths(
    config_path: Path,
) -> dict[str, Path | None]:
    """Resolve the environment files ``load_adapter(config_path)`` selects.

    This deliberately resolves paths without importing or constructing the
    adapter.  In particular, Tier-D dry-run preflight must not execute the
    mjlab adapter's GPU-aware ``__post_init__`` merely to learn which immutable
    files a later runner would consume.
    """
    config_path = Path(config_path).resolve()
    configured = _read_adapter_config_file(config_path)
    if configured.get("class") != TIER_D_TRUSTED_ADAPTER_CLASS:
        raise TrackError(
            "Tier-D environment resolution requires the trusted local "
            f"adapter {TIER_D_TRUSTED_ADAPTER_CLASS!r}"
        )
    _assert_local_tierd_configuration(config_path)
    adapter_config = configured["config"]

    def _path(key: str, conventional_name: str) -> Path | None:
        explicit = adapter_config.get(key)
        if isinstance(explicit, str) and explicit:
            # Match adapter construction: explicit relative paths resolve from
            # the process cwd, while convention paths are project-relative.
            return Path(explicit).expanduser().resolve()
        if explicit is not None:
            raise TrackError(
                f"Tier-D adapter config {key!r} must be a non-empty path"
            )
        conventional = config_path.parent / "env" / conventional_name
        return conventional.resolve() if conventional.is_file() else None

    return {
        "env_spec_path": _path("env_spec_path", "current.json"),
        "eval_reset_path": _path("eval_reset_path", "eval_reset.json"),
        "world_selection_path": _path(
            "world_selection_path", "selection_current.json",
        ),
    }


def _configured_environment_artifacts(config_path: Path) -> dict[str, Any]:
    """Capture and validate exact CPU-readable environment inputs."""
    paths = _configured_environment_paths(config_path)
    env_spec_path = paths["env_spec_path"]
    eval_reset_path = paths["eval_reset_path"]
    try:
        if env_spec_path is not None:
            from sculptor.env_spec import load_env_spec

            load_env_spec(env_spec_path)
        if eval_reset_path is not None:
            payload = json.loads(eval_reset_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("evaluation reset must contain a JSON object")
        return capture_environment_artifacts(**paths)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrackError(f"cannot capture Tier-D environment inputs: {exc}") from exc


def _adapter_environment_artifacts(adapter: Any) -> dict[str, Any]:
    """Capture the resolved paths on the instantiated adapter."""
    try:
        return capture_environment_artifacts(
            env_spec_path=getattr(adapter, "env_spec_path", "") or None,
            eval_reset_path=getattr(adapter, "eval_reset_path", "") or None,
            world_selection_path=(
                getattr(adapter, "world_selection_path", "") or None
            ),
        )
    except ValueError as exc:
        raise TrackError(
            f"cannot capture instantiated Tier-D environment inputs: {exc}"
        ) from exc


def _build_generated_tracker_policy_contract(
    donor_policy_contract: dict[str, Any],
    *,
    reference_clock: dict[str, Any],
) -> dict[str, Any]:
    """Purely bind an exported donor interface to the tracker phase clock.

    Tier-D dry-run is a CPU/data-only preflight.  Importing mjlab merely to
    reconstruct this contract is not data-only: a cold mjlab import currently
    probes ffmpeg through ``subprocess.Popen``.  The donor therefore exports
    its exact base contract once, and this pure transform adds only the
    reference-clock observation used by the generated tracker.
    """
    from sculptor.policy_contract import (
        condition_policy_contract_on_reference_clock,
    )

    try:
        return condition_policy_contract_on_reference_clock(
            donor_policy_contract,
            reference_clock,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackError(
            "cannot bind the exported donor interface to the Tier-D "
            f"reference clock: {exc}"
        ) from exc


def _policy_execution_boundary(
    *, robot: str, policy_contract: dict[str, Any],
) -> dict[str, Any]:
    """Project the generated tracker contract onto Tier-D's execution seam.

    The full policy-contract digest is retained separately as provenance.  It
    is intentionally *not* part of this boundary: a legitimate transfer may
    change observations, network widths, or PPO settings while executing the
    same robot/task/action interface at the same cadence.  The fields below
    are the exact structural and simulator facts covered by the tracking
    compatibility evidence.
    """
    if not isinstance(policy_contract, dict):
        raise TrackError("generated tracker policy contract is unavailable")

    identity = policy_contract.get("identity")
    joints = policy_contract.get("joints")
    actions = policy_contract.get("actions")
    timing = policy_contract.get("timing")
    versions = policy_contract.get("versions")
    if not all(isinstance(block, dict) for block in (
        identity, joints, actions, timing, versions,
    )):
        raise TrackError(
            "generated tracker policy contract is missing identity/joints/"
            "actions/timing/versions"
        )

    adapter_class = identity.get("adapter_class")
    task_id = identity.get("task_id")
    ordered_joints = joints.get("ordered_names")
    ordered_actions = actions.get("ordered_names")
    action_terms = actions.get("term_names")
    action_shape = actions.get("shape")
    if not isinstance(adapter_class, str) or not adapter_class:
        raise TrackError("generated tracker policy contract has no adapter class")
    if adapter_class != TIER_D_TRUSTED_ADAPTER_CLASS:
        raise TrackError(
            "Tier-D execution boundary requires the trusted local adapter "
            f"{TIER_D_TRUSTED_ADAPTER_CLASS!r}"
        )
    if not isinstance(task_id, str) or not task_id:
        raise TrackError("generated tracker policy contract has no task id")
    for label, value in (
        ("ordered joints", ordered_joints),
        ("ordered actions", ordered_actions),
        ("action terms", action_terms),
        ("action shape", action_shape),
    ):
        if not isinstance(value, list) or not value:
            raise TrackError(f"generated tracker policy contract has no {label}")
    if not all(isinstance(name, str) and name for name in ordered_joints):
        raise TrackError(
            "generated tracker policy contract ordered joints are invalid"
        )
    if not all(isinstance(name, str) and name for name in ordered_actions):
        raise TrackError(
            "generated tracker policy contract ordered actions are invalid"
        )
    if not all(isinstance(name, str) and name for name in action_terms):
        raise TrackError(
            "generated tracker policy contract action terms are invalid"
        )
    if not all(
        isinstance(size, int) and not isinstance(size, bool) and size > 0
        for size in action_shape
    ):
        raise TrackError(
            "generated tracker policy contract action shape is invalid"
        )

    try:
        sim_timestep_s = float(timing["sim_timestep_s"])
        decimation = int(timing["decimation"])
        control_dt_s = float(timing["control_dt_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackError(
            "generated tracker policy contract timing is incomplete"
        ) from exc
    if (
        not math.isfinite(sim_timestep_s)
        or not math.isfinite(control_dt_s)
        or sim_timestep_s <= 0.0
        or decimation < 1
        or control_dt_s <= 0.0
    ):
        raise TrackError(
            "generated tracker policy contract timing must be positive"
        )
    expected_control_dt = sim_timestep_s * decimation
    if abs(control_dt_s - expected_control_dt) > 1e-9:
        raise TrackError(
            "generated tracker policy contract control_dt_s does not equal "
            "sim_timestep_s * decimation"
        )

    clean_versions: dict[str, str] = {}
    for key in _TIER_D_VERSION_KEYS:
        value = versions.get(key)
        if not isinstance(value, str) or not value:
            raise TrackError(
                "generated tracker policy contract software version "
                f"{key!r} is unknown"
            )
        clean_versions[key] = value

    if not isinstance(robot, str) or not robot:
        raise TrackError("Tier-D robot identity is empty")
    return {
        "robot": robot,
        "execution_locus": "local",
        "identity": {
            "adapter_class": adapter_class,
            "task_id": task_id,
        },
        "joints": {"ordered_names": list(ordered_joints)},
        "actions": {
            "ordered_names": list(ordered_actions),
            "term_names": list(action_terms),
            "shape": list(action_shape),
        },
        "timing": {
            "sim_timestep_s": sim_timestep_s,
            "decimation": decimation,
            "control_dt_s": control_dt_s,
            "control_hz": 1.0 / control_dt_s,
        },
        "versions": clean_versions,
    }


@dataclass(frozen=True)
class _TierDDonorInterface:
    """One data-only donor interface receipt admitted by Tier-D preflight."""

    donor_project: Path
    policy_contract: dict[str, Any] = field(repr=False)
    donor_config_sha256: str
    certification_config_sha256: str
    receipt_sha256: str


def _read_tierd_donor_interface(
    donor_project: Path,
    *,
    robot: str,
) -> _TierDDonorInterface:
    """Read an exported interface without importing mjlab or adapter code."""
    try:
        donor = Path(donor_project).expanduser().resolve(strict=True)
    except OSError as exc:
        raise TrackError(
            f"cannot resolve Tier-D donor project {donor_project}: {exc}"
        ) from exc
    if not donor.is_dir():
        raise TrackError(f"Tier-D donor project is not a directory: {donor}")
    config_path = donor / "config.toml"
    receipt_path = donor / TIER_D_DONOR_INTERFACE_FILENAME
    if config_path.is_symlink() or receipt_path.is_symlink():
        raise TrackError(
            "Tier-D donor config/interface receipt must not be a symlink"
        )
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise TrackError(
            f"Tier-D donor has no {TIER_D_DONOR_INTERFACE_FILENAME}; export "
            "the trusted donor interface before running CPU preflight with "
            "`sculpt refs export-tierd-interface --donor-project "
            f"{donor}`"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackError(
            f"cannot read Tier-D donor interface receipt: {exc}"
        ) from exc
    expected_keys = {
        "schema",
        "donor_config_sha256",
        "certification_config_sha256",
        "policy_contract",
        "policy_contract_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise TrackError("Tier-D donor interface receipt is non-canonical")
    if receipt.get("schema") != TIER_D_DONOR_INTERFACE_SCHEMA:
        raise TrackError("Tier-D donor interface receipt schema is unsupported")
    try:
        donor_config_sha = _file_sha256(
            config_path, label="donor config.toml",
        )
    except TrackError:
        raise
    if receipt.get("donor_config_sha256") != donor_config_sha:
        raise TrackError(
            "Tier-D donor interface receipt is stale for config.toml"
        )
    certification_config_sha = receipt.get("certification_config_sha256")
    if not _is_sha256(certification_config_sha):
        raise TrackError(
            "Tier-D donor interface receipt has no exact certification "
            "config digest"
        )
    contract = receipt.get("policy_contract")
    if not isinstance(contract, dict) or contract.get("schema") not in {2, 3}:
        raise TrackError(
            "Tier-D donor interface must be an unconditioned schema-2/3 "
            "policy contract"
        )
    if "reference_clock" in contract:
        raise TrackError(
            "Tier-D donor interface must not already contain a reference clock"
        )
    try:
        contract_sha = _canonical_sha256(contract)
    except (TypeError, ValueError) as exc:
        raise TrackError(
            "Tier-D donor policy contract is not canonical JSON"
        ) from exc
    if receipt.get("policy_contract_sha256") != contract_sha:
        raise TrackError("Tier-D donor policy contract digest is stale")
    adapter_cfg = _read_adapter_config_file(config_path)
    task_id = str(
        adapter_cfg.get("config", {}).get("task_id")
        or adapter_cfg.get("config", {}).get("env_id")
        or ""
    )
    identity = contract.get("identity")
    if (
        adapter_cfg.get("class") != TIER_D_TRUSTED_ADAPTER_CLASS
        or not isinstance(identity, dict)
        or identity.get("adapter_class") != TIER_D_TRUSTED_ADAPTER_CLASS
        or identity.get("task_id") != task_id
    ):
        raise TrackError(
            "Tier-D donor interface identity differs from the trusted local "
            "adapter config"
        )
    _assert_local_tierd_configuration(config_path)
    _policy_execution_boundary(robot=robot, policy_contract=contract)
    return _TierDDonorInterface(
        donor_project=donor,
        policy_contract=json.loads(json.dumps(contract, allow_nan=False)),
        donor_config_sha256=donor_config_sha,
        certification_config_sha256=certification_config_sha,
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    )


def export_tierd_donor_interface(donor_project: Path) -> Path:
    """Explicitly export the pure contract later Tier-D dry-runs consume.

    This is deliberately separate from dry-run: exporting inspects the trusted
    mjlab task and may therefore trigger third-party import-time probes.  Once
    exported, all Tier-D CPU preflight work is data-only and the live runner's
    checkpoint sidecar independently corroborates the same conditioned
    contract before any certificate can be published.
    """
    donor = Path(donor_project).expanduser().resolve(strict=True)
    config_path = donor / "config.toml"
    if config_path.is_symlink():
        raise TrackError("Tier-D donor config.toml must not be a symlink")
    _assert_local_tierd_configuration(config_path)
    configured, _parameters, _accepts_kwargs = _resolved_adapter_signature(
        config_path,
    )
    with tempfile.TemporaryDirectory(prefix=".tier-d-interface-") as name:
        staging = Path(name)
        certification_config = write_project_config_toml(staging, configured)
        environment_paths = _configured_environment_paths(certification_config)
        kwargs: dict[str, Any] = {}
        selection_path = environment_paths["world_selection_path"]
        if selection_path is not None:
            kwargs["world_selection_path"] = selection_path
        from sculptor.policy_contract import build_project_policy_contract

        policy_contract = build_project_policy_contract(staging, **kwargs)
        if policy_contract.get("schema") not in {2, 3}:
            raise TrackError(
                "exported donor interface must be an unconditioned schema-2/3 "
                "policy contract"
            )
        receipt = {
            "schema": TIER_D_DONOR_INTERFACE_SCHEMA,
            "donor_config_sha256": _file_sha256(
                config_path, label="donor config.toml",
            ),
            "certification_config_sha256": _file_sha256(
                certification_config,
                label="generated certification config.toml",
            ),
            "policy_contract": policy_contract,
            "policy_contract_sha256": _canonical_sha256(policy_contract),
        }
    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    destination = donor / TIER_D_DONOR_INTERFACE_FILENAME
    if destination.is_symlink():
        raise TrackError("Tier-D donor interface destination must not be a symlink")
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=donor,
            prefix=f".{TIER_D_DONOR_INTERFACE_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(donor, label="Tier-D donor interface")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _tracker_policy_interface_issues(
    policy_contract: Any,
    reference_clock: dict[str, Any],
) -> list[str]:
    """Validate the exact schema-4 clock column retained by Tier-D."""
    if not isinstance(policy_contract, dict):
        return ["generated tracker policy contract is missing"]
    issues: list[str] = []
    if policy_contract.get("schema") != 4:
        issues.append("generated tracker policy contract is not schema 4")
    expected_term = {
        "name": reference_clock["term_name"],
        "source": reference_clock["source"],
        "shape": list(reference_clock["shape"]),
    }
    observations = policy_contract.get("observations")
    if not isinstance(observations, dict):
        return issues + ["generated tracker observation contract is missing"]
    for label, terms_key, shape_key in (
        ("actor", "ordered_terms", "shape"),
        ("critic", "critic_ordered_terms", "critic_shape"),
    ):
        terms = observations.get(terms_key)
        if not isinstance(terms, list) or not terms:
            issues.append(
                f"generated tracker {label} observation terms are missing"
            )
            continue
        if terms[-1] != expected_term:
            issues.append(
                f"generated tracker {label} reference clock is not the exact final "
                "observation term"
            )
        try:
            derived_width = sum(
                math.prod(term["shape"])
                for term in terms
                if isinstance(term, dict)
            )
        except (KeyError, TypeError, ValueError):
            issues.append(
                f"generated tracker {label} observation shapes are invalid"
            )
            continue
        if observations.get(shape_key) != [derived_width]:
            issues.append(
                f"generated tracker {label} observation width is inconsistent"
            )
    return issues


def build_tierd_execution_contract(
    *,
    donor_project: Path,
    certification_config_path: Path,
    clip_id: str,
    robot: str,
    clip: dict[str, Any],
    n_phase_targets: int = N_PHASE_TARGETS,
    policy_contract: Optional[dict[str, Any]] = None,
    reference_clock: Optional[dict[str, Any]] = None,
    environment_artifacts: Optional[dict[str, Any]] = None,
    root_frame_declaration_evidence: Optional[dict[str, Any]] = None,
    root_frame_inheritance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the immutable execution evidence attached to a Tier-D result.

    ``policy_contract`` must be the exact generated tracker project's
    clock-conditioned contract.  Rebuilding it from the donor would bind the
    certificate to the wrong observation interface, so omission fails closed.
    Both donor config bytes and generated certification config bytes are
    retained: the former is source provenance, while the latter proves what
    the adapter actually consumed.
    """
    _assert_local_tierd_configuration(Path(donor_project) / "config.toml")
    _assert_local_tierd_configuration(certification_config_path)
    if environment_artifacts is None:
        environment_artifacts = _configured_environment_artifacts(
            certification_config_path,
        )
    environment_issues = validate_environment_artifacts(environment_artifacts)
    if environment_issues:
        raise TrackError(
            "Tier-D environment input receipt is invalid: "
            + "; ".join(environment_issues)
        )

    if reference_clock is None and isinstance(policy_contract, dict):
        embedded_clock = policy_contract.get("reference_clock")
        if isinstance(embedded_clock, dict):
            reference_clock = embedded_clock
    explicit_root_frame = clip.get("root_frame")
    if explicit_root_frame not in {"absolute", "origin_relative"}:
        raise TrackError(
            "Tier-D reference requires explicit root_frame='absolute' or "
            "'origin_relative'; legacy height-band inference is diagnostic "
            "only, so materialize a new immutable clip before certification"
        )
    if root_frame_declaration_evidence is not None:
        from sculptor.refs import library

        evidence_issues = library.validate_root_frame_declaration_evidence(
            root_frame_declaration_evidence,
            expected_root_frame=explicit_root_frame,
        )
        if evidence_issues:
            raise TrackError(
                "Tier-D root-frame declaration evidence is invalid: "
                + "; ".join(evidence_issues)
            )
    if root_frame_inheritance is not None:
        from sculptor.refs import library

        inheritance_issues = library.validate_root_frame_inheritance_receipt(
            root_frame_inheritance,
            expected_root_frame=explicit_root_frame,
        )
        if inheritance_issues:
            raise TrackError(
                "Tier-D root-frame inheritance is invalid: "
                + "; ".join(inheritance_issues)
            )
    if (
        root_frame_declaration_evidence is not None
        and root_frame_inheritance is not None
    ):
        raise TrackError(
            "Tier-D root-frame authority cannot be both declared and inherited"
        )
    try:
        reference_clock = validate_reference_clock(reference_clock or {})
    except ValueError as exc:
        raise TrackError(
            "Tier-D tracker policy requires an exact reference clock: "
            f"{exc}"
        ) from exc

    if policy_contract is None:
        raise TrackError(
            "Tier-D execution evidence requires the explicit generated "
            "tracker policy contract"
        )

    if policy_contract.get("schema") != 4:
        raise TrackError("Tier-D tracker policy contract must use schema 4")
    try:
        embedded_clock = validate_reference_clock(
            policy_contract.get("reference_clock") or {}
        )
    except ValueError as exc:
        raise TrackError(
            "Tier-D tracker policy contract has no valid reference clock"
        ) from exc
    if embedded_clock != reference_clock:
        raise TrackError(
            "Tier-D tracker policy contract reference clock differs from the "
            "generated reward clock"
        )
    interface_issues = _tracker_policy_interface_issues(
        policy_contract, reference_clock,
    )
    if interface_issues:
        raise TrackError(
            "Tier-D tracker policy interface is invalid: "
            + "; ".join(interface_issues)
        )

    boundary = _policy_execution_boundary(
        robot=robot, policy_contract=policy_contract,
    )
    for label, adapter_cfg in (
        (
            "donor",
            _read_adapter_config_file(Path(donor_project) / "config.toml"),
        ),
        (
            "generated certification",
            _read_adapter_config_file(Path(certification_config_path)),
        ),
    ):
        config_identity = boundary["identity"]
        task_id = str(
            adapter_cfg.get("config", {}).get("task_id")
            or adapter_cfg.get("config", {}).get("env_id")
            or ""
        )
        if adapter_cfg.get("class") != config_identity["adapter_class"]:
            raise TrackError(
                f"{label} config adapter class does not match the donor "
                "policy contract"
            )
        if task_id != config_identity["task_id"]:
            raise TrackError(
                f"{label} config task id does not match the tracker policy contract"
            )
    joint_names = clip.get("joint_names")
    joint_pos = clip.get("joint_pos")
    if not isinstance(joint_names, (list, tuple)) or not joint_names:
        raise TrackError("Tier-D reference has no ordered joint names")
    reference_joints = [str(name) for name in joint_names]
    if reference_joints != boundary["joints"]["ordered_names"]:
        raise TrackError(
            "Tier-D reference ordered joints do not exactly match the donor "
            "policy contract"
        )
    array = np.asarray(joint_pos)
    if array.ndim != 2 or array.shape[0] < 1:
        raise TrackError("Tier-D reference joint_pos must have shape (T, J)")
    if array.shape[1] != len(reference_joints):
        raise TrackError(
            "Tier-D reference joint_pos width does not match ordered joints"
        )
    try:
        fps = float(clip["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackError("Tier-D reference fps is missing/invalid") from exc
    if fps <= 0.0:
        raise TrackError("Tier-D reference fps must be positive")
    if not isinstance(n_phase_targets, int) or n_phase_targets < 1:
        raise TrackError("Tier-D phase target count must be a positive integer")

    frame_count = int(array.shape[0])
    playback_duration_s = reference_playback_duration_s(
        frame_count=frame_count, fps=fps,
    )
    if reference_clock["reference_robot"] != robot:
        raise TrackError(
            "Tier-D reference clock robot differs from the certified robot"
        )
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise TrackError("Tier-D reference clip id is empty")
    if reference_clock["reference_clip_id"] != clip_id:
        raise TrackError(
            "Tier-D reference clock clip id differs from the certified clip"
        )
    if reference_clock["n_phase_targets"] != n_phase_targets:
        raise TrackError(
            "Tier-D reference clock target count differs from the generated "
            "tracking reward"
        )
    if abs(
        float(reference_clock["phase_duration_s"]) - playback_duration_s
    ) > 1e-9:
        raise TrackError(
            "Tier-D reference clock duration differs from the exact sampled "
            "clip duration"
        )
    (
        target_names,
        target_joint_pos,
        target_joint_vel,
        target_root_z,
        target_gravity,
    ) = _tracking_targets_from_clip(clip, n_phase_targets=n_phase_targets)
    if target_names != reference_joints:
        raise TrackError("Tier-D generated target joint order is inconsistent")
    expected_target_sha = reference_target_sha256(
        reference_tracking_target_payload(
            joint_names=target_names,
            target_joint_pos=target_joint_pos,
            target_joint_vel=target_joint_vel,
            target_root_z=target_root_z,
            target_gravity=target_gravity,
            root_frame=clip_root_frame(clip),
        )
    )
    if reference_clock["reference_target_sha256"] != expected_target_sha:
        raise TrackError(
            "Tier-D reference clock target hash differs from the exact clip "
            "tracking tables"
        )
    base: dict[str, Any] = {
        "schema": TIER_D_EXECUTION_CONTRACT_SCHEMA,
        "donor": {
            "config_sha256": _file_sha256(
                Path(donor_project) / "config.toml", label="donor config.toml",
            ),
            "certification_config_sha256": _file_sha256(
                Path(certification_config_path),
                label="generated certification config.toml",
            ),
            "policy_contract": policy_contract,
            "policy_contract_sha256": _canonical_sha256(policy_contract),
        },
        "execution_boundary": boundary,
        "environment_artifacts": json.loads(json.dumps(
            environment_artifacts, allow_nan=False,
        )),
        "reference": {
            "clip_id": clip_id,
            "root_frame": explicit_root_frame,
            "root_frame_declaration_evidence": (
                json.loads(json.dumps(
                    root_frame_declaration_evidence,
                    allow_nan=False,
                ))
                if root_frame_declaration_evidence is not None
                else None
            ),
            "root_frame_inheritance": (
                json.loads(json.dumps(
                    root_frame_inheritance,
                    allow_nan=False,
                ))
                if root_frame_inheritance is not None
                else None
            ),
            "fps": fps,
            "frame_count": frame_count,
            "playback_duration_s": playback_duration_s,
            "ordered_joints": reference_joints,
            "phase_target_count": n_phase_targets,
            "rollout_lane": 0,
            "clock_contract": reference_clock,
            "cadence": {
                "schema": TIER_D_REFERENCE_CADENCE,
                "target_table_sampling": REFERENCE_TARGET_SAMPLING,
                "target_selection": "floor(phase * n_phase_targets)",
                "phase_interval": "[0,1)",
                "clock": reference_clock["clock"],
            },
        },
    }
    base["execution_boundary_sha256"] = _canonical_sha256(boundary)
    base["contract_sha256"] = _canonical_sha256(base)
    issues = validate_tierd_execution_contract(base)
    if issues:
        raise TrackError("invalid Tier-D execution contract: " + "; ".join(issues))
    return base


def _canonical_tierd_initial_checkpoint(value: Any) -> dict[str, Any]:
    """Validate the immutable receipt for a trusted local continuation.

    A continuation is not an optimizer resume claim.  It is an exact,
    policy-contract-compatible tracker checkpoint used to initialize the next
    bounded certification attempt.  The source and retained copies are both
    content addressed so moving the original work directory cannot change the
    scientific identity of the new run.
    """
    if not isinstance(value, dict):
        raise TrackError("Tier-D initial checkpoint receipt must be an object")
    required = {
        "schema",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "source_reward_module_sha256",
        "policy_contract_sha256",
        "policy_contract_sidecar_sha256",
        "source_metrics_sha256",
        "retained_checkpoint_relpath",
        "retained_policy_contract_sidecar_relpath",
        "retained_metrics_relpath",
    }
    if set(value) != required:
        raise TrackError("Tier-D initial checkpoint receipt is non-canonical")
    canonical = {key: value.get(key) for key in sorted(required)}
    if canonical["schema"] != TIER_D_CONTINUATION_SCHEMA:
        raise TrackError("Tier-D initial checkpoint schema is unsupported")
    for key in (
        "checkpoint_sha256",
        "source_reward_module_sha256",
        "policy_contract_sha256",
        "policy_contract_sidecar_sha256",
        "source_metrics_sha256",
    ):
        if not _is_sha256(canonical[key]):
            raise TrackError(f"Tier-D initial checkpoint {key} is invalid")
    if (
        not isinstance(canonical["checkpoint_size_bytes"], int)
        or isinstance(canonical["checkpoint_size_bytes"], bool)
        or canonical["checkpoint_size_bytes"] < 1
    ):
        raise TrackError("Tier-D initial checkpoint size is invalid")
    expected_relpaths = {
        "retained_checkpoint_relpath": "initialization/checkpoint.pt",
        "retained_policy_contract_sidecar_relpath": (
            "initialization/checkpoint.pt.policy_contract.json"
        ),
        "retained_metrics_relpath": "initialization/source_metrics.json",
    }
    for key, expected in expected_relpaths.items():
        if canonical[key] != expected:
            raise TrackError(
                f"Tier-D initial checkpoint {key} is non-canonical"
            )
    return canonical


def bind_tierd_runtime_artifacts(
    execution_contract: dict[str, Any],
    *,
    requested_reward_module_sha256: str,
    train_receipts: list[dict[str, Any]],
    final_checkpoint_sha256: str,
    requested_steps_per_iteration: int,
    requested_seed: int,
    requested_num_envs: int,
    requested_rollout_seed: Optional[int] = None,
    requested_rollout_episodes: int = 1,
    requested_rollout_max_steps: Optional[int] = None,
    requested_rollout_task_id: Optional[str] = None,
    initial_checkpoint: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Bind training outputs and rollout inputs to one Tier-D receipt.

    The base execution contract is intentionally constructible before GPU
    allocation.  Once training finishes, this function adds the runner's
    observed reward/checkpoint chain and the exact final checkpoint that the
    rollout must load, then re-hashes the complete contract.
    """
    base_issues = validate_tierd_execution_contract(execution_contract)
    if base_issues:
        raise TrackError(
            "cannot bind artifacts to invalid Tier-D execution contract: "
            + "; ".join(base_issues)
        )
    if not _is_sha256(requested_reward_module_sha256):
        raise TrackError("Tier-D requested reward module sha256 is invalid")
    if not _is_sha256(final_checkpoint_sha256):
        raise TrackError("Tier-D final checkpoint sha256 is invalid")
    if not isinstance(train_receipts, list) or not train_receipts:
        raise TrackError("Tier-D requires at least one observed training receipt")
    if requested_steps_per_iteration < 1:
        raise TrackError("Tier-D requested steps per iteration must be positive")
    if requested_num_envs < 1:
        raise TrackError("Tier-D requested training env count must be positive")
    environment_artifacts = execution_contract.get("environment_artifacts")
    environment_issues = validate_environment_artifacts(environment_artifacts)
    if environment_issues:
        raise TrackError(
            "Tier-D requested environment artifacts are invalid: "
            + "; ".join(environment_issues)
        )
    try:
        train_environment_artifacts = environment_artifacts_for_phase(
            environment_artifacts, "train",
        )
        rollout_environment_artifacts = environment_artifacts_for_phase(
            environment_artifacts, "rollout",
        )
    except ValueError as exc:  # pragma: no cover - validation above is exact
        raise TrackError(str(exc)) from exc
    if requested_rollout_seed is None:
        requested_rollout_seed = requested_seed
    if requested_rollout_max_steps is None:
        duration_s = float(execution_contract["reference"]["playback_duration_s"])
        control_dt_s = float(
            execution_contract["execution_boundary"]["timing"]["control_dt_s"]
        )
        requested_rollout_max_steps = max(
            1, int(math.ceil(duration_s / control_dt_s)) + 1,
        )
    if requested_rollout_task_id is None:
        requested_rollout_task_id = str(
            execution_contract["execution_boundary"]["identity"]["task_id"]
        )
    if (
        not isinstance(requested_rollout_seed, int)
        or isinstance(requested_rollout_seed, bool)
        or not isinstance(requested_rollout_episodes, int)
        or isinstance(requested_rollout_episodes, bool)
        or requested_rollout_episodes < 1
        or not isinstance(requested_rollout_max_steps, int)
        or isinstance(requested_rollout_max_steps, bool)
        or requested_rollout_max_steps < 1
        or not isinstance(requested_rollout_task_id, str)
        or not requested_rollout_task_id
    ):
        raise TrackError("Tier-D requested rollout settings are invalid")

    canonical_initial = (
        _canonical_tierd_initial_checkpoint(initial_checkpoint)
        if initial_checkpoint is not None
        else None
    )
    if (
        canonical_initial is not None
        and canonical_initial["policy_contract_sha256"]
        != execution_contract["donor"]["policy_contract_sha256"]
    ):
        raise TrackError(
            "Tier-D initial checkpoint policy contract differs from the "
            "generated tracker execution boundary"
        )

    canonical_train: list[dict[str, Any]] = []
    prior_checkpoint_sha: Optional[str] = (
        str(canonical_initial["checkpoint_sha256"])
        if canonical_initial is not None
        else None
    )
    for expected_index, raw in enumerate(train_receipts, start=1):
        if not isinstance(raw, dict):
            raise TrackError("Tier-D training receipt must be an object")
        receipt = {
            "iteration": raw.get("iteration"),
            "schema": raw.get("schema"),
            "phase": raw.get("phase"),
            "reward_module_sha256": raw.get("reward_module_sha256"),
            "requested_max_iterations": raw.get("requested_max_iterations"),
            "requested_seed": raw.get("requested_seed"),
            "requested_num_envs": raw.get("requested_num_envs"),
            "seed_application": raw.get("seed_application"),
            "environment_artifacts": raw.get("environment_artifacts"),
            "env_spec_application": raw.get("env_spec_application"),
            "input_checkpoint_requested_sha256": raw.get(
                "input_checkpoint_requested_sha256"
            ),
            "input_checkpoint_loaded_sha256": raw.get(
                "input_checkpoint_loaded_sha256"
            ),
            "input_checkpoint_load_completed": raw.get(
                "input_checkpoint_load_completed"
            ),
            "output_checkpoint_sha256": raw.get("output_checkpoint_sha256"),
            "output_policy_contract_sha256": raw.get(
                "output_policy_contract_sha256"
            ),
            "output_policy_contract_sidecar_sha256": raw.get(
                "output_policy_contract_sidecar_sha256"
            ),
        }
        if receipt["iteration"] != expected_index:
            raise TrackError("Tier-D training receipt iteration order is invalid")
        if (
            receipt["schema"] != RUNNER_RUNTIME_ARTIFACT_SCHEMA
            or receipt["phase"] != "train"
        ):
            raise TrackError("Tier-D training receipt schema/phase is invalid")
        if receipt["reward_module_sha256"] != requested_reward_module_sha256:
            raise TrackError(
                "Tier-D training consumed reward bytes different from those "
                "requested"
            )
        if receipt["environment_artifacts"] != train_environment_artifacts:
            raise TrackError(
                "Tier-D training consumed environment bytes different from "
                "those requested"
            )
        if (
            receipt["requested_max_iterations"]
            != requested_steps_per_iteration
            or receipt["requested_seed"] != requested_seed
            or receipt["requested_num_envs"] != requested_num_envs
        ):
            raise TrackError(
                "Tier-D observed training settings differ from the request"
            )
        receipt["seed_application"] = _canonical_seed_application(
            receipt["seed_application"], requested_seed=requested_seed,
        )
        receipt["env_spec_application"] = _canonical_application_receipt(
            receipt["env_spec_application"],
            schema="reward-sculptor-env-spec-application-v1",
            phase="train",
        )
        if prior_checkpoint_sha is None:
            expected_input = (None, None, False)
        else:
            expected_input = (
                prior_checkpoint_sha,
                prior_checkpoint_sha,
                True,
            )
        observed_input = (
            receipt["input_checkpoint_requested_sha256"],
            receipt["input_checkpoint_loaded_sha256"],
            receipt["input_checkpoint_load_completed"],
        )
        if observed_input != expected_input:
            raise TrackError(
                "Tier-D checkpoint chain has stale requested/loaded facts"
            )
        if not _is_sha256(receipt["output_checkpoint_sha256"]):
            raise TrackError("Tier-D training checkpoint sha256 is invalid")
        if (
            receipt["output_policy_contract_sha256"]
            != execution_contract["donor"]["policy_contract_sha256"]
            or not _is_sha256(
                receipt["output_policy_contract_sidecar_sha256"]
            )
        ):
            raise TrackError(
                "Tier-D training checkpoint policy contract differs from the "
                "generated tracker execution boundary"
            )
        canonical_train.append(receipt)
        prior_checkpoint_sha = str(receipt["output_checkpoint_sha256"])
    if canonical_train[-1]["output_checkpoint_sha256"] != final_checkpoint_sha256:
        raise TrackError(
            "Tier-D final checkpoint bytes differ from the last training receipt"
        )

    # Canonical JSON round-trip prevents aliases to nested caller-owned values.
    bound = json.loads(json.dumps(execution_contract, allow_nan=False))
    bound.pop("contract_sha256", None)
    bound["runtime_artifacts"] = {
        "schema": TIER_D_RUNTIME_ARTIFACT_SCHEMA,
        "requested_reward_module_sha256": requested_reward_module_sha256,
        "requested_training": {
            "iterations": len(canonical_train),
            "steps_per_iteration": requested_steps_per_iteration,
            "seed": requested_seed,
            "num_envs": requested_num_envs,
        },
        "train_observations": canonical_train,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "rollout_requirements": {
            "reward_module_sha256": requested_reward_module_sha256,
            "checkpoint_sha256": final_checkpoint_sha256,
            "checkpoint_load_completed": True,
            "environment_artifacts": rollout_environment_artifacts,
            "requested_seed": requested_rollout_seed,
            "requested_n_episodes": requested_rollout_episodes,
            "requested_max_episode_steps": requested_rollout_max_steps,
            "requested_task_id": requested_rollout_task_id,
            "requested_lane": int(execution_contract["reference"]["rollout_lane"]),
        },
    }
    if canonical_initial is not None:
        bound["runtime_artifacts"]["initial_checkpoint"] = canonical_initial
    bound["contract_sha256"] = _canonical_sha256(bound)
    issues = validate_tierd_execution_contract(bound)
    if issues:
        raise TrackError(
            "bound Tier-D execution contract is invalid: " + "; ".join(issues)
        )
    return bound


def validate_tierd_execution_contract(contract: Any) -> list[str]:
    """Validate one stored execution receipt without trusting its digests."""
    if not isinstance(contract, dict):
        return ["execution contract is missing"]
    issues: list[str] = []
    if contract.get("schema") != TIER_D_EXECUTION_CONTRACT_SCHEMA:
        issues.append("execution contract schema is unsupported")

    boundary = contract.get("execution_boundary")
    donor = contract.get("donor")
    environment_artifacts = contract.get("environment_artifacts")
    reference = contract.get("reference")
    if not isinstance(boundary, dict):
        issues.append("execution boundary is missing")
    if not isinstance(donor, dict):
        issues.append("donor evidence is missing")
    if not isinstance(reference, dict):
        issues.append("reference cadence evidence is missing")
    issues.extend(validate_environment_artifacts(environment_artifacts))
    if issues:
        return issues

    try:
        rebuilt_boundary = _policy_execution_boundary(
            robot=boundary.get("robot"),
            policy_contract={
                "identity": boundary.get("identity"),
                "joints": boundary.get("joints"),
                "actions": boundary.get("actions"),
                "timing": boundary.get("timing"),
                "versions": boundary.get("versions"),
            },
        )
    except TrackError as exc:
        issues.append(str(exc))
        rebuilt_boundary = None
    if rebuilt_boundary is not None and rebuilt_boundary != boundary:
        issues.append("execution boundary contains non-canonical fields or values")

    for key in (
        "config_sha256", "certification_config_sha256", "policy_contract_sha256",
    ):
        if not _is_sha256(donor.get(key)):
            issues.append(f"donor.{key} is missing/invalid")

    cadence = reference.get("cadence")
    try:
        fps = float(reference["fps"])
        frame_count = int(reference["frame_count"])
        duration_s = float(reference["playback_duration_s"])
        phase_target_count = int(reference["phase_target_count"])
        rollout_lane = int(reference["rollout_lane"])
    except (KeyError, TypeError, ValueError):
        issues.append("reference fps/frame count/duration/phase targets are invalid")
    else:
        if fps <= 0.0 or frame_count < 1 or duration_s <= 0.0:
            issues.append("reference fps/frame count/duration must be positive")
        else:
            try:
                exact_duration_s = reference_playback_duration_s(
                    frame_count=frame_count, fps=fps,
                )
            except ValueError:
                exact_duration_s = -1.0
            if abs(duration_s - exact_duration_s) > 1e-9:
                issues.append(
                    "reference playback duration does not match the sampled "
                    "trajectory interval count"
                )
        if phase_target_count < 1:
            issues.append("reference phase target count must be positive")
        if rollout_lane != 0:
            issues.append("Tier-D rollout lane must be precommitted to lane 0")
        try:
            control_hz = float(
                boundary.get("timing", {}).get("control_hz", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            control_hz = 0.0
            issues.append("execution boundary control cadence is invalid")
        if control_hz < fps / 2.0:
            issues.append("reference fps exceeds the certified control cadence")
        if duration_s > 0.0 and phase_target_count > round(duration_s * control_hz):
            issues.append("reference phase targets exceed available control steps")
    expected_cadence = {
        "schema": TIER_D_REFERENCE_CADENCE,
        "target_table_sampling": REFERENCE_TARGET_SAMPLING,
        "target_selection": "floor(phase * n_phase_targets)",
        "phase_interval": "[0,1)",
        "clock": "per_environment_episode_elapsed_control_time",
    }
    if not isinstance(cadence, dict) or cadence != expected_cadence:
        issues.append("reference cadence schema is missing/unsupported")
    ordered_reference_joints = reference.get("ordered_joints")
    if ordered_reference_joints != boundary.get("joints", {}).get("ordered_names"):
        issues.append("reference ordered joints differ from the execution boundary")
    reference_clip_id = reference.get("clip_id")
    if not isinstance(reference_clip_id, str) or not reference_clip_id:
        issues.append("reference clip id is missing")
    if reference.get("root_frame") not in {"absolute", "origin_relative"}:
        issues.append("reference root frame is missing/unsupported")
    declaration_evidence = reference.get("root_frame_declaration_evidence")
    if declaration_evidence is not None:
        from sculptor.refs import library

        issues.extend(
            library.validate_root_frame_declaration_evidence(
                declaration_evidence,
                expected_root_frame=reference.get("root_frame"),
            )
        )
    inheritance = reference.get("root_frame_inheritance")
    if inheritance is not None:
        from sculptor.refs import library

        issues.extend(
            library.validate_root_frame_inheritance_receipt(
                inheritance,
                expected_root_frame=reference.get("root_frame"),
            )
        )
    if declaration_evidence is not None and inheritance is not None:
        issues.append(
            "reference root-frame authority cannot be both declared and inherited"
        )

    stored_policy_contract = donor.get("policy_contract")
    if not isinstance(stored_policy_contract, dict):
        issues.append("generated tracker policy contract is missing")
    else:
        try:
            stored_clock = validate_reference_clock(
                stored_policy_contract.get("reference_clock") or {}
            )
            reference_clock = validate_reference_clock(
                reference.get("clock_contract") or {}
            )
        except ValueError as exc:
            issues.append(f"reference clock contract is invalid: {exc}")
        else:
            issues.extend(
                _tracker_policy_interface_issues(
                    stored_policy_contract, stored_clock,
                )
            )
            if stored_clock != reference_clock:
                issues.append(
                    "generated tracker policy and reference evidence clocks differ"
                )
            if reference_clock["reference_robot"] != boundary.get("robot"):
                issues.append("reference clock robot differs from execution boundary")
            if reference_clock["reference_clip_id"] != reference_clip_id:
                issues.append("reference clock clip id differs from cadence evidence")
            if reference_clock["n_phase_targets"] != reference.get(
                "phase_target_count"
            ):
                issues.append("reference clock target count differs from cadence")
            try:
                recorded_duration = float(reference.get("playback_duration_s"))
            except (TypeError, ValueError):
                recorded_duration = -1.0
            if abs(
                float(reference_clock["phase_duration_s"]) - recorded_duration
            ) > 1e-9:
                issues.append("reference clock duration differs from cadence evidence")
        try:
            actual_policy_sha = _canonical_sha256(stored_policy_contract)
        except (TypeError, ValueError):
            actual_policy_sha = ""
            issues.append(
                "generated tracker policy contract is not canonical JSON"
            )
        if donor.get("policy_contract_sha256") != actual_policy_sha:
            issues.append("generated tracker policy contract sha256 mismatch")

    runtime_artifacts = contract.get("runtime_artifacts")
    if runtime_artifacts is not None:
        if not isinstance(runtime_artifacts, dict):
            issues.append("runtime artifact evidence must be an object")
        else:
            try:
                expected_train_environment = environment_artifacts_for_phase(
                    environment_artifacts, "train",
                )
            except ValueError:
                expected_train_environment = None
            requested_reward_sha = runtime_artifacts.get(
                "requested_reward_module_sha256"
            )
            final_checkpoint_sha = runtime_artifacts.get(
                "final_checkpoint_sha256"
            )
            if runtime_artifacts.get("schema") != TIER_D_RUNTIME_ARTIFACT_SCHEMA:
                issues.append("runtime artifact evidence schema is unsupported")
            if not _is_sha256(requested_reward_sha):
                issues.append("requested reward module sha256 is missing/invalid")
            if not _is_sha256(final_checkpoint_sha):
                issues.append("final checkpoint sha256 is missing/invalid")
            requested_training = runtime_artifacts.get("requested_training")
            if not isinstance(requested_training, dict):
                issues.append("requested training settings are missing")
                requested_iterations = 0
                requested_steps = 0
                requested_seed = None
                requested_num_envs = 0
            else:
                requested_iterations = requested_training.get("iterations")
                requested_steps = requested_training.get("steps_per_iteration")
                requested_seed = requested_training.get("seed")
                requested_num_envs = requested_training.get("num_envs")
                if (
                    not isinstance(requested_iterations, int)
                    or isinstance(requested_iterations, bool)
                    or requested_iterations < 1
                    or not isinstance(requested_steps, int)
                    or isinstance(requested_steps, bool)
                    or requested_steps < 1
                    or not _is_runtime_seed(requested_seed)
                    or not isinstance(requested_num_envs, int)
                    or isinstance(requested_num_envs, bool)
                    or requested_num_envs < 1
                ):
                    issues.append("requested training settings are invalid")
            observations = runtime_artifacts.get("train_observations")
            initial_checkpoint = runtime_artifacts.get("initial_checkpoint")
            if initial_checkpoint is not None:
                try:
                    canonical_initial = _canonical_tierd_initial_checkpoint(
                        initial_checkpoint
                    )
                except TrackError as exc:
                    issues.append(str(exc))
                    canonical_initial = None
                if (
                    canonical_initial is not None
                    and canonical_initial["policy_contract_sha256"]
                    != donor.get("policy_contract_sha256")
                ):
                    issues.append(
                        "initial checkpoint policy contract differs from donor"
                    )
            else:
                canonical_initial = None
            if not isinstance(observations, list) or not observations:
                issues.append("training runtime observations are missing")
            else:
                if len(observations) != requested_iterations:
                    issues.append(
                        "training observation count differs from requested "
                        "iterations"
                    )
                prior_checkpoint_sha = (
                    canonical_initial["checkpoint_sha256"]
                    if canonical_initial is not None
                    else None
                )
                for index, observation in enumerate(observations, start=1):
                    if not isinstance(observation, dict):
                        issues.append("training runtime observation is invalid")
                        continue
                    try:
                        seed_application = _canonical_seed_application(
                            observation.get("seed_application"),
                            requested_seed=requested_seed,
                        )
                    except TrackError as exc:
                        issues.append(str(exc))
                        seed_application = observation.get("seed_application")
                    try:
                        env_spec_application = _canonical_application_receipt(
                            observation.get("env_spec_application"),
                            schema="reward-sculptor-env-spec-application-v1",
                            phase="train",
                        )
                    except TrackError as exc:
                        issues.append(str(exc))
                        env_spec_application = observation.get(
                            "env_spec_application"
                        )
                    expected_observation = {
                        "iteration": index,
                        "schema": RUNNER_RUNTIME_ARTIFACT_SCHEMA,
                        "phase": "train",
                        "reward_module_sha256": requested_reward_sha,
                        "requested_max_iterations": requested_steps,
                        "requested_seed": requested_seed,
                        "requested_num_envs": requested_num_envs,
                        "seed_application": seed_application,
                        "environment_artifacts": expected_train_environment,
                        "env_spec_application": env_spec_application,
                        "input_checkpoint_requested_sha256": (
                            prior_checkpoint_sha
                        ),
                        "input_checkpoint_loaded_sha256": prior_checkpoint_sha,
                        "input_checkpoint_load_completed": (
                            prior_checkpoint_sha is not None
                        ),
                        "output_checkpoint_sha256": observation.get(
                            "output_checkpoint_sha256"
                        ),
                        "output_policy_contract_sha256": donor.get(
                            "policy_contract_sha256"
                        ),
                        "output_policy_contract_sidecar_sha256": (
                            observation.get(
                                "output_policy_contract_sidecar_sha256"
                            )
                        ),
                    }
                    if observation != expected_observation:
                        issues.append(
                            "training runtime observation contains stale or "
                            "non-canonical fields"
                        )
                    if not _is_sha256(observation.get(
                        "output_checkpoint_sha256"
                    )):
                        issues.append(
                            "training output checkpoint sha256 is invalid"
                        )
                    else:
                        prior_checkpoint_sha = observation[
                            "output_checkpoint_sha256"
                        ]
                    if not _is_sha256(observation.get(
                        "output_policy_contract_sidecar_sha256"
                    )):
                        issues.append(
                            "training policy-contract sidecar sha256 is invalid"
                        )
                if (
                    isinstance(observations[-1], dict)
                    and observations[-1].get("output_checkpoint_sha256")
                    != final_checkpoint_sha
                ):
                    issues.append(
                        "final checkpoint differs from last training observation"
                    )
            rollout_requirements = runtime_artifacts.get("rollout_requirements")
            if not isinstance(rollout_requirements, dict):
                issues.append("rollout runtime requirements are invalid")
            else:
                expected_rollout_keys = {
                    "reward_module_sha256",
                    "checkpoint_sha256",
                    "checkpoint_load_completed",
                    "environment_artifacts",
                    "requested_seed",
                    "requested_n_episodes",
                    "requested_max_episode_steps",
                    "requested_task_id",
                    "requested_lane",
                }
                if set(rollout_requirements) != expected_rollout_keys:
                    issues.append("rollout runtime requirements are non-canonical")
                if (
                    rollout_requirements.get("reward_module_sha256")
                    != requested_reward_sha
                    or rollout_requirements.get("checkpoint_sha256")
                    != final_checkpoint_sha
                    or rollout_requirements.get("checkpoint_load_completed") is not True
                ):
                    issues.append("rollout reward/checkpoint requirements are invalid")
                try:
                    expected_rollout_environment = environment_artifacts_for_phase(
                        environment_artifacts, "rollout",
                    )
                except ValueError:
                    expected_rollout_environment = None
                if rollout_requirements.get(
                    "environment_artifacts"
                ) != expected_rollout_environment:
                    issues.append("rollout environment requirements are invalid")
                requested_rollout_seed = rollout_requirements.get("requested_seed")
                requested_episodes = rollout_requirements.get("requested_n_episodes")
                requested_max_steps = rollout_requirements.get(
                    "requested_max_episode_steps"
                )
                requested_task_id = rollout_requirements.get("requested_task_id")
                requested_lane = rollout_requirements.get("requested_lane")
                if (
                    not _is_runtime_seed(requested_rollout_seed)
                    or not isinstance(requested_episodes, int)
                    or isinstance(requested_episodes, bool)
                    or requested_episodes < 1
                    or not isinstance(requested_max_steps, int)
                    or isinstance(requested_max_steps, bool)
                    or requested_max_steps < 1
                    or requested_task_id
                    != boundary.get("identity", {}).get("task_id")
                    or requested_lane != reference.get("rollout_lane")
                ):
                    issues.append("requested rollout settings are invalid")

    recorded_boundary_sha = contract.get("execution_boundary_sha256")
    try:
        actual_boundary_sha = _canonical_sha256(boundary)
    except (TypeError, ValueError):
        actual_boundary_sha = ""
        issues.append("execution boundary is not canonical JSON")
    if not _is_sha256(recorded_boundary_sha):
        issues.append("execution boundary sha256 is missing/invalid")
    elif recorded_boundary_sha != actual_boundary_sha:
        issues.append("execution boundary sha256 mismatch")

    recorded_contract_sha = contract.get("contract_sha256")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    try:
        actual_contract_sha = _canonical_sha256(unsigned_contract)
    except (TypeError, ValueError):
        actual_contract_sha = ""
        issues.append("execution contract is not canonical JSON")
    if not _is_sha256(recorded_contract_sha):
        issues.append("execution contract sha256 is missing/invalid")
    elif recorded_contract_sha != actual_contract_sha:
        issues.append("execution contract sha256 mismatch")
    return issues


def compare_tierd_target_contract(
    execution_contract: Any,
    target_policy_contract: Any,
    *,
    target_robot: str,
) -> list[str]:
    """Compare certified dynamics evidence to a proposed target project.

    Donor config and full policy-contract digests remain evidence, not an
    equality constraint.  This comparison deliberately admits different
    optimization/network settings while requiring the same robot, task,
    ordered joint/action interface, simulator cadence, and software versions.
    """
    issues = validate_tierd_execution_contract(execution_contract)
    if issues:
        return [f"certified {issue}" for issue in issues]
    try:
        target = _policy_execution_boundary(
            robot=target_robot, policy_contract=target_policy_contract,
        )
    except TrackError as exc:
        return [f"target {exc}"]
    source = execution_contract["execution_boundary"]
    reasons: list[str] = []
    paths = (
        ("robot",),
        ("identity", "adapter_class"),
        ("identity", "task_id"),
        ("joints", "ordered_names"),
        ("actions", "ordered_names"),
        ("actions", "term_names"),
        ("actions", "shape"),
        ("timing", "sim_timestep_s"),
        ("timing", "decimation"),
        ("timing", "control_dt_s"),
        ("versions", "torch"),
        ("versions", "mjlab"),
        ("versions", "rsl_rl"),
        ("versions", "adapter"),
    )
    for path in paths:
        left: Any = source
        right: Any = target
        for key in path:
            left = left.get(key) if isinstance(left, dict) else None
            right = right.get(key) if isinstance(right, dict) else None
        if left != right:
            reasons.append(
                f"{'.'.join(path)} differs (certified {left!r}, target {right!r})"
            )
    return reasons


# ── phase-target downsampling ───────────────────────────────────────────
def downsample_phase_targets(
    array: np.ndarray, n: int = N_PHASE_TARGETS,
) -> np.ndarray:
    """Resample `array` (shape `(T, ...)`) to exactly `n` phase-indexed
    rows via nearest-frame lookup at evenly spaced phase fractions
    `[0, 1]` — i.e. index `round(phase * (T - 1))`. The final table row
    is therefore the exact final clip sample used by terminal holds. With a
    one-row table, that sole row is the final sample for the same reason.
    Deterministic, no interpolation (avoids inventing joint poses between real
    mocap frames). `n` must be >= 1; `array` must have `T >= 1` along axis 0.
    """
    array = np.asarray(array)
    t = array.shape[0]
    if t < 1:
        raise ValueError(f"array must have at least 1 frame along axis 0, got {t}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        # A one-row table is necessarily its own terminal row. Preserve the
        # exact final pose rather than silently treating it as clip entrance.
        return array[[-1]]
    phases = np.linspace(0.0, 1.0, n, endpoint=True)
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


def reference_tracking_target_payload(
    *,
    joint_names: list[str],
    target_joint_pos: np.ndarray,
    target_joint_vel: np.ndarray,
    target_root_z: np.ndarray,
    target_gravity: Optional[np.ndarray],
    root_frame: str,
    phase_mode: str = "hold",
) -> dict[str, Any]:
    """Canonical versioned identity of every executed tracking target.

    Joint velocity and the post-duration hold rule are execution semantics,
    not advisory metadata: the flat mission reward scores velocity and swaps
    its terminal velocity target to zero.  Omitting either let two different
    reward programs advertise the same ``REFERENCE_TARGET_SHA256``.
    """
    if phase_mode not in {"hold", "loop"}:
        raise ValueError("reference target phase_mode must be hold or loop")
    joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
    joint_vel = np.asarray(target_joint_vel, dtype=np.float64)
    root_z = np.asarray(target_root_z, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(
            "reference joint position/velocity targets must share shape (K, J)"
        )
    if root_z.shape != (joint_pos.shape[0],):
        raise ValueError("reference root-z targets must align with phase rows")
    gravity = None
    if target_gravity is not None:
        gravity = np.asarray(target_gravity, dtype=np.float64)
        if gravity.shape != (joint_pos.shape[0], 3):
            raise ValueError(
                "reference gravity targets must have shape (K, 3)"
            )
    return {
        "schema": REFERENCE_TARGET_IDENTITY_SCHEMA,
        "sampling": REFERENCE_TARGET_SAMPLING,
        "phase_mode": phase_mode,
        "terminal_hold": (
            REFERENCE_TERMINAL_HOLD if phase_mode == "hold" else None
        ),
        "joint_names": [str(name) for name in joint_names],
        "joint_pos": np.round(joint_pos, 5).tolist(),
        "joint_vel": np.round(joint_vel, 5).tolist(),
        "root_z": np.round(root_z, 5).tolist(),
        "root_frame": root_frame,
        "gravity": (
            np.round(gravity, 5).tolist() if gravity is not None else None
        ),
    }


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


def _tracking_targets_from_clip(
    clip: dict[str, Any],
    *,
    n_phase_targets: int,
) -> tuple[
    list[str], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]
]:
    """Derive the one authoritative set of Tier-D embedded target tables."""
    joint_names = [str(name) for name in (clip.get("joint_names") or [])]
    joint_pos = np.asarray(clip.get("joint_pos"), dtype=np.float64)
    root_z = np.asarray(clip.get("root_pos_z"), dtype=np.float64)
    if (
        not joint_names
        or joint_pos.ndim != 2
        or joint_pos.shape[1] != len(joint_names)
        or root_z.ndim != 1
        or root_z.shape[0] != joint_pos.shape[0]
    ):
        raise TrackError("Tier-D clip cannot produce aligned tracking targets")
    target_joint_pos = downsample_phase_targets(
        joint_pos, n=n_phase_targets,
    )
    source_joint_vel = clip.get("joint_vel")
    if source_joint_vel is None:
        fps = float(clip.get("fps") or 0.0)
        if joint_pos.shape[0] <= 1:
            source_joint_vel = np.zeros_like(joint_pos)
        elif fps > 0.0:
            source_joint_vel = np.gradient(joint_pos, axis=0) * fps
        else:
            raise TrackError(
                "Tier-D clip without joint_vel requires positive fps"
            )
    source_joint_vel = np.asarray(source_joint_vel, dtype=np.float64)
    if source_joint_vel.shape != joint_pos.shape:
        raise TrackError(
            "Tier-D clip joint_vel must align with joint_pos"
        )
    target_joint_vel = downsample_phase_targets(
        source_joint_vel, n=n_phase_targets,
    )
    target_root_z = downsample_phase_targets(root_z, n=n_phase_targets)
    target_gravity: Optional[np.ndarray] = None
    quat = clip.get("root_quat_wxyz")
    if quat is not None:
        target_gravity = downsample_phase_targets(
            projected_gravity_from_quat(np.asarray(quat, dtype=np.float64)),
            n=n_phase_targets,
        )
        norm = np.linalg.norm(target_gravity, axis=1, keepdims=True)
        target_gravity = target_gravity / np.where(norm > 0.0, norm, 1.0)
    return (
        joint_names,
        target_joint_pos,
        target_joint_vel,
        target_root_z,
        target_gravity,
    )


def build_tierd_reference_clock(
    clip: dict[str, Any],
    *,
    clip_id: str,
    robot: str,
    n_phase_targets: int = N_PHASE_TARGETS,
) -> dict[str, Any]:
    """Build the exact clock/target identity a Tier-D tracker must execute."""
    names, joint_pos, joint_vel, root_z, gravity = _tracking_targets_from_clip(
        clip, n_phase_targets=n_phase_targets,
    )
    target_sha = reference_target_sha256(
        reference_tracking_target_payload(
            joint_names=names,
            target_joint_pos=joint_pos,
            target_joint_vel=joint_vel,
            target_root_z=root_z,
            target_gravity=gravity,
            root_frame=clip_root_frame(clip),
        )
    )
    frame_count = int(np.asarray(clip.get("joint_pos")).shape[0])
    duration_s = reference_playback_duration_s(
        frame_count=frame_count,
        fps=float(clip.get("fps") or 0.0),
    )
    return build_reference_clock(
        clip_id=clip_id,
        robot=robot,
        target_sha256=target_sha,
        phase_mode="hold",
        phase_duration_s=duration_s,
        n_phase_targets=n_phase_targets,
    )


def generate_tracking_reward_source(
    *,
    clip_id: str,
    robot: str,
    joint_names: list[str],
    target_joint_pos: np.ndarray,
    target_root_z: np.ndarray,
    episode_len_steps: int,
    target_joint_vel: Optional[np.ndarray] = None,
    duration_s: float = 0.0,
    target_gravity: Optional[np.ndarray] = None,
    joint_err_weight: float = JOINT_ERR_WEIGHT,
    root_err_weight: float = ROOT_ERR_WEIGHT,
    orientation_err_weight: float = ORIENTATION_ERR_WEIGHT,
    joint_term_scale: float = JOINT_TERM_SCALE,
    root_term_scale: float = ROOT_TERM_SCALE,
    orientation_term_scale: float = ORIENTATION_TERM_SCALE,
    root_frame: str = "origin_relative",
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

    The joint term is a fixed blend of position and velocity kernels. During
    a terminal hold the final pose remains the position target and zero joint
    velocity becomes the velocity target. The root and optional lower-mass
    orientation kernels are normalized with that blended joint term back to
    the historical maximum return. `joint_names` is
    the SAME order as `target_joint_pos` columns and is asserted against
    `qpos`'s trailing (actuated-joint) slice length at reward-call time
    via `len(joint_names)` — a project.joint-count mismatch raises inside
    `compute_reward` rather than silently misindexing.
    """
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("tracking reward clip_id must be non-empty")
    if not isinstance(robot, str) or not robot.strip():
        raise ValueError("tracking reward robot must be non-empty")
    if root_frame not in {"absolute", "origin_relative"}:
        raise ValueError(
            "tracking reward root_frame must be absolute or origin_relative"
        )
    term_scales = {
        "joint_term_scale": joint_term_scale,
        "root_term_scale": root_term_scale,
        "orientation_term_scale": orientation_term_scale,
    }
    for label, value in term_scales.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"tracking reward {label} must be finite and non-negative")
    if float(joint_term_scale) <= 0.0 or float(root_term_scale) <= 0.0:
        raise ValueError(
            "tracking reward joint/root term scales must be positive"
        )
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
    if target_joint_vel is None:
        target_joint_vel = np.zeros_like(target_joint_pos, dtype=np.float64)
    target_joint_vel = np.asarray(target_joint_vel, dtype=np.float64)
    if target_joint_vel.shape != target_joint_pos.shape:
        raise ValueError(
            "target_joint_vel must match target_joint_pos shape: "
            f"{target_joint_vel.shape} vs {target_joint_pos.shape}"
        )

    if target_gravity is not None:
        target_gravity = np.asarray(target_gravity, dtype=np.float64)
        if target_gravity.shape != (n_phase, 3):
            raise ValueError(
                "target_gravity must be (n_phase, 3) to align with the joint "
                f"targets: {target_gravity.shape} vs {(n_phase, 3)}")

    joint_pos_literal = _format_array_literal(target_joint_pos)
    joint_vel_literal = _format_array_literal(target_joint_vel)
    root_z_literal = _format_array_literal(target_root_z)
    names_literal = "[" + ", ".join(repr(str(n)) for n in joint_names) + "]"
    gravity_literal = (
        "np.asarray(" + _format_array_literal(target_gravity)
        + ", dtype=np.float64)" if target_gravity is not None else "None")
    # Zero weight collapses the term to a no-op for clips with no orientation
    # data, so the reward shape is identical to before for those.
    orientation_weight = orientation_err_weight if target_gravity is not None else 0.0
    orientation_mass = (
        float(orientation_term_scale) if target_gravity is not None else 0.0
    )
    term_mass_sum = (
        float(joint_term_scale) + float(root_term_scale) + orientation_mass
    )
    reward_total_scale = 3.0 if target_gravity is not None else 2.0
    joint_term_coefficient = (
        reward_total_scale * float(joint_term_scale) / term_mass_sum
    )
    root_term_coefficient = (
        reward_total_scale * float(root_term_scale) / term_mass_sum
    )
    orientation_term_coefficient = (
        reward_total_scale * orientation_mass / term_mass_sum
    )
    from sculptor.reference_clock import (
        build_reference_clock,
        reference_target_sha256,
    )

    rounded_targets = reference_tracking_target_payload(
        joint_names=joint_names,
        target_joint_pos=target_joint_pos,
        target_joint_vel=target_joint_vel,
        target_root_z=target_root_z,
        target_gravity=target_gravity,
        root_frame=root_frame,
    )
    target_hash = reference_target_sha256(rounded_targets)
    effective_duration_s = (
        float(duration_s)
        if float(duration_s) > 0.0
        else max(1, int(episode_len_steps)) * 0.02
    )
    reference_clock = build_reference_clock(
        clip_id=clip_id,
        robot=robot,
        target_sha256=target_hash,
        phase_mode="hold",
        phase_duration_s=effective_duration_s,
        n_phase_targets=n_phase,
    )

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

REFERENCE_TARGET_SHA256 = {target_hash!r}
REFERENCE_TARGET_IDENTITY_SCHEMA = {REFERENCE_TARGET_IDENTITY_SCHEMA!r}
REFERENCE_TARGET_SAMPLING = {REFERENCE_TARGET_SAMPLING!r}
REFERENCE_PHASE_MODE = "hold"
REFERENCE_TERMINAL_HOLD = {REFERENCE_TERMINAL_HOLD!r}

REWARD_SPEC: dict = {{
    "version": "tierD-track-v2",
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
    # This is part of the policy interface, not reward metadata: actor and
    # critic observe the exact clock used below to select reference targets.
    "reference_clock": {reference_clock!r},
    "root_height_frame": {root_frame!r},
    "hyperparameters": {{
        "joint_err_weight": {joint_err_weight!r},
        "joint_vel_err_weight": 0.10,
        "joint_position_share": 0.75,
        "joint_velocity_share": 0.25,
        "root_err_weight": {root_err_weight!r},
        "orientation_err_weight": {orientation_weight!r},
        "joint_term_scale": {float(joint_term_scale)!r},
        "root_term_scale": {float(root_term_scale)!r},
        "orientation_term_scale": {float(orientation_term_scale)!r},
        "reward_total_scale": {reward_total_scale!r},
        "joint_term_coefficient": {joint_term_coefficient!r},
        "root_term_coefficient": {root_term_coefficient!r},
        "orientation_term_coefficient": {orientation_term_coefficient!r},
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
REFERENCE_DURATION_S = {effective_duration_s!r}
REFERENCE_ROOT_FRAME = {root_frame!r}
JOINT_ERR_WEIGHT = {joint_err_weight!r}
JOINT_VEL_ERR_WEIGHT = 0.10
JOINT_POSITION_SHARE = 0.75
JOINT_VELOCITY_SHARE = 0.25
ROOT_ERR_WEIGHT = {root_err_weight!r}
# 0.0 when the clip carries no root orientation, which makes the orientation
# term an exact no-op rather than a silently-wrong constant.
ORIENTATION_ERR_WEIGHT = {orientation_weight!r}
JOINT_TERM_SCALE = {float(joint_term_scale)!r}
ROOT_TERM_SCALE = {float(root_term_scale)!r}
ORIENTATION_TERM_SCALE = {float(orientation_term_scale)!r}
REWARD_TOTAL_SCALE = {reward_total_scale!r}
JOINT_TERM_COEFFICIENT = {joint_term_coefficient!r}
ROOT_TERM_COEFFICIENT = {root_term_coefficient!r}
ORIENTATION_TERM_COEFFICIENT = {orientation_term_coefficient!r}

# Phase-indexed targets, shape (N_PHASE, N_JOINTS) / (N_PHASE,).
TARGET_JOINT_POS = np.asarray({joint_pos_literal}, dtype=np.float64).reshape(N_PHASE, N_JOINTS)
TARGET_JOINT_VEL = np.asarray({joint_vel_literal}, dtype=np.float64).reshape(N_PHASE, N_JOINTS)
TARGET_ROOT_Z = np.asarray({root_z_literal}, dtype=np.float64)
# Unit gravity in the body frame, derived from the clip's root_quat_wxyz. Yaw-
# invariant on purpose: retargeting zeroes root translation, so a heading
# offset is not an orientation error. `None` when the clip has no quaternion.
TARGET_GRAVITY = {gravity_literal}


def reference_clock_scalar(info) -> float:
    step = int(info.get("episode_length", 0) or 0)
    step_dt = float(info.get("step_dt", 0.0) or 0.0)
    if REFERENCE_DURATION_S > 0.0 and step_dt > 0.0:
        phase = (step * step_dt) / REFERENCE_DURATION_S
    elif EPISODE_LEN_STEPS > 0:
        phase = step / float(EPISODE_LEN_STEPS)
    else:
        phase = 0.0
    return min(max(phase, 0.0), 0.999999)


def _phase_index(info) -> int:
    return int(reference_clock_scalar(info) * N_PHASE)


def reference_clock_batched(info, like):
    import torch

    step = info.get("episode_length", torch.zeros_like(like))
    step_dt = info.get("step_dt", None)
    if REFERENCE_DURATION_S > 0.0 and step_dt is not None:
        phase = torch.clamp(
            (step * step_dt) / REFERENCE_DURATION_S, 0.0, 0.999999)
    elif EPISODE_LEN_STEPS > 0:
        phase = torch.clamp(
            step / float(EPISODE_LEN_STEPS), 0.0, 0.999999)
    else:
        phase = torch.zeros_like(like)
    return (phase + torch.zeros_like(like))[:, None]


def reference_target_index_batched(info, like):
    import torch

    phase = reference_clock_batched(info, like)[:, 0]
    return torch.clamp((phase * N_PHASE).long(), 0, N_PHASE - 1)


def compute_reward(state, action, next_state, info):
    del action  # tracking reward does not penalize control effort
    qpos = np.asarray(next_state["qpos"], dtype=np.float64)
    qvel = np.asarray(next_state["qvel"], dtype=np.float64)
    if qpos.shape[0] < 7 + N_JOINTS or qvel.shape[0] < N_JOINTS:
        raise ValueError(
            f"qpos/qvel too short for {{N_JOINTS}} tracked joints: "
            f"shapes={{qpos.shape}}/{{qvel.shape}}")
    root_z = float(qpos[2])
    joint_pos = qpos[7:7 + N_JOINTS]

    i = _phase_index(info)
    target_joint = TARGET_JOINT_POS[i]
    target_joint_vel = TARGET_JOINT_VEL[i]
    target_root_z = TARGET_ROOT_Z[i]

    step = int(info.get("episode_length", 0) or 0)
    step_dt = float(info.get("step_dt", 0.0) or 0.0)
    terminal_hold = (
        step * step_dt >= REFERENCE_DURATION_S
        if step_dt > 0.0 and REFERENCE_DURATION_S > 0.0
        else EPISODE_LEN_STEPS > 0 and step >= EPISODE_LEN_STEPS
    )
    if REFERENCE_PHASE_MODE == "hold" and terminal_hold:
        target_joint_vel = np.zeros_like(target_joint_vel)

    joint_err = joint_pos - target_joint
    joint_vel = qvel[-N_JOINTS:]
    joint_vel_err = joint_vel - target_joint_vel
    mean_joint_err_sq = float(np.mean(joint_err ** 2))
    mean_joint_vel_err_sq = float(np.mean(joint_vel_err ** 2))
    if REFERENCE_ROOT_FRAME == "absolute":
        root_err = root_z - float(target_root_z)
    else:
        root0 = float(TARGET_ROOT_Z[0])
        actual_delta = float(info.get("base_height_delta", root_z - root0))
        root_err = actual_delta - (float(target_root_z) - root0)

    joint_pos_term = float(np.exp(-JOINT_ERR_WEIGHT * mean_joint_err_sq))
    joint_vel_term = float(np.exp(
        -JOINT_VEL_ERR_WEIGHT * mean_joint_vel_err_sq))
    joint_term = (
        JOINT_POSITION_SHARE * joint_pos_term
        + JOINT_VELOCITY_SHARE * joint_vel_term
    )
    root_term = float(np.exp(-ROOT_ERR_WEIGHT * (root_err ** 2)))

    joint_contribution = JOINT_TERM_COEFFICIENT * joint_term
    root_contribution = ROOT_TERM_COEFFICIENT * root_term
    components = {{
        "joint_tracking": joint_contribution,
        "root_tracking": root_contribution,
    }}
    reward = joint_contribution + root_contribution
    if TARGET_GRAVITY is not None:
        gravity = np.asarray(
            next_state["projected_gravity_b"], dtype=np.float64).reshape(-1)[-3:]
        orient_err_sq = float(np.mean((gravity - TARGET_GRAVITY[i]) ** 2))
        orient_term = float(np.exp(-ORIENTATION_ERR_WEIGHT * orient_err_sq))
        orientation_contribution = ORIENTATION_TERM_COEFFICIENT * orient_term
        components["orientation_tracking"] = orientation_contribution
        reward += orientation_contribution
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
    qvel = next_state["qvel"]
    if qpos.shape[-1] < N_JOINTS or qvel.shape[-1] < N_JOINTS:
        raise ValueError(
            f"batched qpos/qvel have {{qpos.shape[-1]}}/{{qvel.shape[-1]}} "
            f"columns, fewer than the {{N_JOINTS}} tracked joints")
    like = qpos[:, 0]

    i = reference_target_index_batched(info, like)

    target_joint = torch.as_tensor(
        TARGET_JOINT_POS, device=qpos.device, dtype=qpos.dtype)[i]
    target_joint_vel = torch.as_tensor(
        TARGET_JOINT_VEL, device=qvel.device, dtype=qvel.dtype)[i]
    target_root = torch.as_tensor(
        TARGET_ROOT_Z, device=qpos.device, dtype=qpos.dtype)[i]

    step = info.get("episode_length", torch.zeros_like(like))
    step_dt = info.get("step_dt", None)
    if REFERENCE_PHASE_MODE == "hold":
        terminal_hold = (
            step * step_dt >= REFERENCE_DURATION_S
            if step_dt is not None and REFERENCE_DURATION_S > 0.0
            else step >= EPISODE_LEN_STEPS
        )
        target_joint_vel = torch.where(
            terminal_hold[:, None],
            torch.zeros_like(target_joint_vel),
            target_joint_vel,
        )

    joint_err = qpos[:, -N_JOINTS:] - target_joint
    joint_vel_err = qvel[:, -N_JOINTS:] - target_joint_vel
    joint_pos_term = torch.exp(
        -JOINT_ERR_WEIGHT * torch.mean(joint_err ** 2, dim=-1))
    joint_vel_term = torch.exp(
        -JOINT_VEL_ERR_WEIGHT * torch.mean(joint_vel_err ** 2, dim=-1))
    joint_term = (
        JOINT_POSITION_SHARE * joint_pos_term
        + JOINT_VELOCITY_SHARE * joint_vel_term
    )

    root0 = float(TARGET_ROOT_Z[0])
    base_height = info.get("base_height", torch.zeros_like(like))
    if REFERENCE_ROOT_FRAME == "absolute":
        root_err = base_height - target_root
    else:
        actual_delta = info.get("base_height_delta", base_height - root0)
        root_err = actual_delta - (target_root - root0)
    root_term = torch.exp(-ROOT_ERR_WEIGHT * root_err ** 2)

    joint_contribution = JOINT_TERM_COEFFICIENT * joint_term
    root_contribution = ROOT_TERM_COEFFICIENT * root_term
    total = joint_contribution + root_contribution
    components = {{
        "joint_tracking": joint_contribution,
        "root_tracking": root_contribution,
    }}
    if TARGET_GRAVITY is not None:
        target_gravity = torch.as_tensor(
            TARGET_GRAVITY, device=qpos.device, dtype=qpos.dtype)[i]
        gravity = next_state["projected_gravity_b"][:, -3:]
        orient_term = torch.exp(-ORIENTATION_ERR_WEIGHT * torch.mean(
            (gravity - target_gravity) ** 2, dim=-1))
        orientation_contribution = ORIENTATION_TERM_COEFFICIENT * orient_term
        components["orientation_tracking"] = orientation_contribution
        total = total + orientation_contribution
    return total, components
'''


def generate_tracking_residual_reward_source(
    *,
    clip: dict[str, Any],
    clip_id: str,
    robot: str,
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

    if not isinstance(robot, str) or not robot:
        raise TrackError("tracking reward requires an exact robot namespace")
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
    # Runtime tracking consumes the same full, one-shot schedule certified by
    # Tier D.  Cropping or looping may be useful as a separately materialized
    # reference transformation, but silently applying either here would make
    # the runtime clock/targets different from the evidence that admitted it.
    phase_mode = "hold"

    joint_pos = downsample_phase_targets(joint_pos_raw, n=n_phase_targets)
    joint_vel = downsample_phase_targets(joint_vel_raw, n=n_phase_targets)
    root_pos = downsample_phase_targets(root_pos_raw, n=n_phase_targets)
    gravity = (
        downsample_phase_targets(gravity_raw, n=n_phase_targets)
        if gravity_raw is not None else None)

    # Hash exactly the rounded arrays embedded in source.  This is the durable
    # parent→child identity checked after every LLM rewrite.
    root_frame = clip_root_frame(clip)
    target_hash = reference_target_sha256(
        reference_tracking_target_payload(
            joint_names=[str(name) for name in meta["joint_names"]],
            target_joint_pos=joint_pos,
            target_joint_vel=joint_vel,
            target_root_z=root_pos[:, 2],
            target_gravity=gravity,
            root_frame=root_frame,
        )
    )

    n_frames = int(joint_pos_raw.shape[0])
    duration_s = reference_playback_duration_s(
        frame_count=n_frames, fps=fps,
    )
    names_literal = repr([str(name) for name in meta["joint_names"]])
    jp_literal = _format_array_literal(joint_pos)
    jv_literal = _format_array_literal(joint_vel)
    rz_literal = _format_array_literal(root_pos[:, 2])
    gravity_literal = (
        _format_array_literal(gravity) if gravity is not None else "None")
    n_joints = int(joint_pos.shape[1])
    orientation_weight = 0.20 if gravity is not None else 0.0
    root_weight = 0.25 + (0.20 - orientation_weight)
    from sculptor.reference_clock import build_reference_clock

    reference_clock = build_reference_clock(
        clip_id=clip_id,
        robot=robot,
        target_sha256=target_hash,
        phase_mode=phase_mode,
        phase_duration_s=duration_s,
        n_phase_targets=n_phase_targets,
    )

    return f'''"""Reference-tracking base plus bounded task residual.

Generated deterministically from clip {clip_id!r}.  REFERENCE_* data and
``_reference_tracking_*`` functions are the immutable motion prior; reward
editing may only author the small residual task term.
"""
from __future__ import annotations

import numpy as np

REFERENCE_TARGET_SHA256 = {target_hash!r}
REFERENCE_TARGET_IDENTITY_SCHEMA = {REFERENCE_TARGET_IDENTITY_SCHEMA!r}
REFERENCE_TARGET_SAMPLING = {REFERENCE_TARGET_SAMPLING!r}
REFERENCE_TERMINAL_HOLD = {REFERENCE_TERMINAL_HOLD!r}
REFERENCE_JOINT_NAMES = {names_literal}
REFERENCE_N_PHASES = {n_phase_targets}
REFERENCE_N_JOINTS = {n_joints}
REFERENCE_DURATION_S = {duration_s!r}
REFERENCE_PHASE_MODE = {phase_mode!r}
REFERENCE_ROOT_FRAME = {root_frame!r}
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
    "reference_clock": {reference_clock!r},
    "reference_robot": {robot!r},
    "root_height_frame": {root_frame!r},
    "composition": {{
        "type": "reference_tracking_residual",
        "reference_clip_id": {clip_id!r},
        "reference_robot": {robot!r},
        "reference_target_sha256": {target_hash!r},
        "reference_target_identity_schema": {REFERENCE_TARGET_IDENTITY_SCHEMA!r},
        "reference_target_sampling": {REFERENCE_TARGET_SAMPLING!r},
        "terminal_hold": {REFERENCE_TERMINAL_HOLD!r},
        "tracking_weight": 1.0,
        "residual_max": {float(residual_max)!r},
        "phase_mode": {phase_mode!r},
        "phase_duration_s": {duration_s!r},
        "root_height_frame": {root_frame!r},
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


def reference_clock_scalar(info):
    elapsed = _scalar(info, "episode_length") * _scalar(info, "step_dt", 0.02)
    if REFERENCE_PHASE_MODE == "loop":
        fraction = (max(elapsed, 0.0) % REFERENCE_DURATION_S) / REFERENCE_DURATION_S
    else:
        fraction = min(max(elapsed / REFERENCE_DURATION_S, 0.0), 0.999999)
    return fraction


def _phase_index_scalar(info):
    return min(
        REFERENCE_N_PHASES - 1,
        int(reference_clock_scalar(info) * REFERENCE_N_PHASES),
    )


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
    base_height = _scalar(info, "base_height")
    if REFERENCE_ROOT_FRAME == "absolute":
        root_err = base_height - float(REFERENCE_ROOT_Z[i])
    else:
        reference_root_delta = float(
            REFERENCE_ROOT_Z[i] - REFERENCE_ROOT_Z[0])
        actual_root_delta = _scalar(
            info, "base_height_delta",
            base_height - float(REFERENCE_ROOT_Z[0]))
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


def reference_clock_batched(info, like):
    import torch
    step = info.get("episode_length", torch.zeros_like(like))
    dt = info.get("step_dt", torch.full_like(like, 0.02))
    elapsed = torch.clamp(step * dt, min=0.0)
    if REFERENCE_PHASE_MODE == "loop":
        fraction = torch.remainder(elapsed, REFERENCE_DURATION_S) / REFERENCE_DURATION_S
    else:
        fraction = torch.clamp(elapsed / REFERENCE_DURATION_S, 0.0, 0.999999)
    return (fraction + torch.zeros_like(like))[:, None]


def reference_target_index_batched(info, like):
    import torch

    fraction = reference_clock_batched(info, like)[:, 0]
    return torch.clamp(
        (fraction * REFERENCE_N_PHASES).long(), 0, REFERENCE_N_PHASES - 1)


def _phase_index_batched(info, like):
    return reference_target_index_batched(info, like)


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
    base_height = info.get("base_height", torch.zeros_like(like))
    if REFERENCE_ROOT_FRAME == "absolute":
        root_err = base_height - target_root
    else:
        reference_root_delta = target_root - float(REFERENCE_ROOT_Z[0])
        actual_root_delta = info.get(
            "base_height_delta", base_height - float(REFERENCE_ROOT_Z[0]))
        root_err = actual_root_delta - reference_root_delta
    root_height = torch.exp(-40.0 * root_err ** 2)
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
    #: Fraction of the reference's WALL TIME the rollout spans, clamped [0, 1].
    #: Seconds, not frames — a 120 fps clip and a 50 Hz rollout of the same
    #: 3.70 s have 444 vs 185 frames, and the frame ratio reported 41.7%
    #: coverage for a rollout that ran the entire motion.
    duration_coverage: float
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
    #: meaning "no static control evidence is available". Such a result is a
    #: useful diagnostic but cannot earn Tier D: exact tracking certification must
    #: prove temporal tracking on at least one resolved joint.
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
    def has_common_joint_evidence(self) -> bool:
        return (
            isinstance(self.n_common_joints, int)
            and not isinstance(self.n_common_joints, bool)
            and self.n_common_joints > 0
            and len(self.common_joint_names) == self.n_common_joints
            and len(set(self.common_joint_names)) == self.n_common_joints
            and all(
                isinstance(name, str) and bool(name)
                for name in self.common_joint_names
            )
        )

    @property
    def static_baseline_ratio(self) -> float:
        if (
            not np.isfinite(self.mean_joint_err_rad)
            or not np.isfinite(self.static_baseline_err_rad)
            or self.static_baseline_err_rad <= 0.0
        ):
            return float("inf")
        return self.mean_joint_err_rad / self.static_baseline_err_rad

    @property
    def beats_static_baseline(self) -> bool:
        """Did the policy beat a finite, non-vacuous constant-pose control?"""
        if not self.has_common_joint_evidence:
            return False
        if (
            not np.isfinite(self.static_baseline_err_rad)
            or self.static_baseline_err_rad < MIN_REFERENCE_MOTION_RAD
        ):
            return False
        return self.static_baseline_ratio <= STATIC_BASELINE_RATIO_MAX

    @property
    def feasible(self) -> bool:
        return (
            self.mean_joint_err_rad < MEAN_JOINT_ERR_THRESHOLD_RAD
            and self.root_z_rmse_m < ROOT_Z_RMSE_THRESHOLD_M
            and np.isfinite(self.duration_coverage)
            and self.duration_coverage >= DURATION_COVERAGE_MIN
            and self.beats_static_baseline
        )

    def to_dict(self) -> dict[str, Any]:
        mean_joint_err = round(self.mean_joint_err_rad, 6)
        static_baseline_err = (
            round(self.static_baseline_err_rad, 6)
            if np.isfinite(self.static_baseline_err_rad) else None
        )
        static_baseline_ratio = (
            round(mean_joint_err / static_baseline_err, 6)
            if static_baseline_err is not None and static_baseline_err > 0.0
            else None
        )
        return {
            "certification_scope": json.loads(json.dumps(
                TIER_D_CERTIFICATION_SCOPE, allow_nan=False,
            )),
            "mean_joint_err_rad": mean_joint_err,
            "max_joint_err_rad": round(self.max_joint_err_rad, 6),
            "root_z_rmse_m": round(self.root_z_rmse_m, 6),
            "duration_coverage": round(self.duration_coverage, 6),
            "orientation_err": round(self.orientation_err, 6),
            "common_joint_names": list(self.common_joint_names),
            "n_common_joints": self.n_common_joints,
            "root_frame": self.root_frame,
            "root_z_offset_m": round(self.root_z_offset_m, 6),
            "static_baseline_err_rad": static_baseline_err,
            "static_baseline_ratio": static_baseline_ratio,
            "beats_static_baseline": self.beats_static_baseline,
            "motion_ratio": round(self.motion_ratio, 6),
            "feasible": self.feasible,
            "thresholds": {
                "mean_joint_err_rad": MEAN_JOINT_ERR_THRESHOLD_RAD,
                "root_z_rmse_m": ROOT_Z_RMSE_THRESHOLD_M,
                "static_baseline_ratio_max": STATIC_BASELINE_RATIO_MAX,
                "duration_coverage_min": DURATION_COVERAGE_MIN,
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
    control_hz: float = DEFAULT_CONTROL_HZ,
    rollout_samples_are_post_step: bool = False,
    scheduled_target_joint_pos: Optional[np.ndarray] = None,
    scheduled_target_root_z: Optional[np.ndarray] = None,
    scheduled_target_root_anchor: Optional[float] = None,
    scheduled_target_gravity: Optional[np.ndarray] = None,
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
    scheduled_joint = (
        None if scheduled_target_joint_pos is None
        else np.asarray(scheduled_target_joint_pos, dtype=np.float64)
    )
    scheduled_root = (
        None if scheduled_target_root_z is None
        else np.asarray(scheduled_target_root_z, dtype=np.float64)
    )
    scheduled_gravity = (
        None if scheduled_target_gravity is None
        else np.asarray(scheduled_target_gravity, dtype=np.float64)
    )
    if scheduled_joint is not None and scheduled_joint.shape != (
        t_rollout, len(clip_joint_names)
    ):
        raise ValueError(
            "scheduled_target_joint_pos must have exact shape "
            "(rollout_steps, clip_joints)"
        )
    if scheduled_root is not None and scheduled_root.shape != (t_rollout,):
        raise ValueError(
            "scheduled_target_root_z must have exact shape (rollout_steps,)"
        )
    if scheduled_gravity is not None and scheduled_gravity.shape != (
        t_rollout, 3
    ):
        raise ValueError(
            "scheduled_target_gravity must have exact shape (rollout_steps, 3)"
        )
    # How much of the reference's WALL TIME the rollout spans. Frame counts are
    # not comparable across the two: a 120 fps clip and a 50 Hz rollout covering
    # the identical 3.70 s have 444 and 185 frames, and dividing those reported
    # 41.7% coverage for a rollout that in fact ran the whole motion — exactly
    # 50/120. Convert both to seconds first.
    clip_fps = float(clip.get("fps") or 0.0)
    clip_duration_s = (
        reference_playback_duration_s(frame_count=t_clip, fps=clip_fps)
        if clip_fps > 0 else float(t_clip)
    )
    # The mjlab rollout recorder stores one state *after* every valid control
    # transition and excludes the done step because mjlab has already
    # auto-reset that state.  Such a prefix of T post-step samples proves T
    # control intervals, even though the timestamps between the stored samples
    # span only T-1 intervals.  Generic callers may instead provide ordinary
    # state samples including t=0, so preserve the sampled-span convention by
    # default and make the runner's transition semantics explicit at its
    # artifact boundary.
    if control_hz > 0 and t_rollout > 0:
        rollout_duration_s = (
            float(t_rollout) / control_hz
            if rollout_samples_are_post_step
            else reference_playback_duration_s(
                frame_count=t_rollout, fps=control_hz,
            )
        )
    else:
        rollout_duration_s = float(t_rollout)
    duration_coverage = (
        min(1.0, rollout_duration_s / clip_duration_s)
        if clip_duration_s > 0 else 0.0)

    common_names: list[str] = []
    mean_err = 0.0
    max_err = 0.0
    # No common joints -> root-only scoring; there is no joint trace to
    # compare against a static pose, so the control is vacuously satisfied.
    static_err = 0.0
    motion_ratio = 0.0
    # For runner artifacts, reference lookup is by the certified wall clock:
    # sample i is the state after transition i+1, at (i+1)/control_hz.  Never
    # normalize an arbitrarily long/short rollout over the whole clip; doing so
    # made a 2x-slow replay appear to track perfectly.
    timed_clip_indices: Optional[np.ndarray] = None
    if (
        rollout_samples_are_post_step
        and control_hz > 0.0
        and clip_fps > 0.0
        and t_clip > 0
        and t_rollout > 0
    ):
        sample_times_s = (
            np.arange(t_rollout, dtype=np.float64) + 1.0
        ) / control_hz
        timed_clip_indices = np.minimum(
            np.floor(sample_times_s * clip_fps + 1e-12).astype(np.int64),
            t_clip - 1,
        )

    if clip_joint_pos is not None and clip_joint_names and t_rollout > 0:
        clip_idx, rollout_idx, common_names = _resolve_common_joints(
            list(clip_joint_names), list(rollout_joint_names))
        if common_names:
            clip_jp = np.asarray(clip_joint_pos, dtype=np.float64)
            if scheduled_joint is not None:
                clip_at_n = scheduled_joint[:, clip_idx]
                rollout_at_n = rollout_joint_pos[:, rollout_idx]
            elif timed_clip_indices is not None:
                clip_at_n = clip_jp[timed_clip_indices][:, clip_idx]
                rollout_at_n = rollout_joint_pos[:, rollout_idx]
            else:
                n = min(t_rollout, t_clip)
                # Generic, non-runner callers may supply ordinary state samples
                # without timestamps; retain phase alignment for that API.
                clip_at_n = downsample_phase_targets(
                    clip_jp[:, clip_idx], n=n,
                )
                rollout_at_n = downsample_phase_targets(
                    rollout_joint_pos[:, rollout_idx], n=n,
                )
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
        if scheduled_root is not None:
            clip_z_at_n = scheduled_root
            rollout_z_at_n = rollout_root_z
        elif timed_clip_indices is not None:
            clip_z_at_n = clip_root_z[timed_clip_indices]
            rollout_z_at_n = rollout_root_z
        else:
            clip_z_at_n = downsample_phase_targets(clip_root_z, n=n)
            rollout_z_at_n = downsample_phase_targets(rollout_root_z, n=n)
        if root_frame == "origin_relative":
            clip_anchor = (
                float(scheduled_target_root_anchor)
                if scheduled_target_root_anchor is not None
                else float(clip_z_at_n[0])
            )
            root_offset = float(rollout_z_at_n[0] - clip_anchor)
            clip_z_at_n = clip_z_at_n - clip_anchor
            rollout_z_at_n = rollout_z_at_n - rollout_z_at_n[0]
        else:
            root_offset = float(rollout_z_at_n[0] - clip_z_at_n[0])
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
            if scheduled_gravity is not None:
                diff = scheduled_gravity - roll_g
            elif timed_clip_indices is not None:
                diff = clip_g[timed_clip_indices[:roll_g.shape[0]]] - roll_g
            else:
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


_TIER_D_TRAJECTORY_SCHEMA = "reward-sculptor-trajectory-v1"
_TIER_D_VALID_MASK_CONTRACT = {
    "key": "first_episode_valid_mask",
    "semantics": "true_prefix_before_first_done",
    "invalid_state": "frozen_last_valid_sample",
    "state_samples": "post_step_after_valid_transition",
}


def _score_tierd_rollout_artifact(
    path: Path,
    *,
    clip: dict[str, Any],
    execution_contract: dict[str, Any],
    lane: int = 0,
) -> TrackingErrors:
    """Load and score one exact, self-describing Tier-D rollout artifact.

    The valid-mask prefix is the episode. Frozen post-done padding is excluded
    so it cannot forge duration coverage or a stable terminal tail.
    """
    issues = validate_tierd_execution_contract(execution_contract)
    if issues:
        raise TrackError(
            "cannot score rollout against invalid execution contract: "
            + "; ".join(issues)
        )
    boundary = execution_contract["execution_boundary"]
    reference = execution_contract["reference"]
    runtime_artifacts = execution_contract.get("runtime_artifacts")
    if not isinstance(runtime_artifacts, dict):
        raise TrackError(
            "Tier-D rollout scoring requires bound reward/checkpoint runtime "
            "artifacts"
        )
    rollout_requirements = runtime_artifacts.get("rollout_requirements")
    if not isinstance(rollout_requirements, dict):
        raise TrackError("Tier-D rollout requirements are missing")
    expected_joints = list(boundary["joints"]["ordered_names"])
    expected_dt = float(boundary["timing"]["control_dt_s"])
    expected_metadata = {
        "schema": _TIER_D_TRAJECTORY_SCHEMA,
        "layout": ["time", "environment", "feature"],
        "ordered_joint_names": expected_joints,
        "control_dt_s": expected_dt,
        "root_link_pos_w_frame": "world",
        "first_episode_lane": lane,
        "valid_mask": _TIER_D_VALID_MASK_CONTRACT,
        "runtime_artifacts": {
            "schema": RUNNER_RUNTIME_ARTIFACT_SCHEMA,
            "phase": "rollout",
            "reward_module_sha256": rollout_requirements[
                "reward_module_sha256"
            ],
            "checkpoint_sha256": rollout_requirements["checkpoint_sha256"],
            "checkpoint_load_completed": True,
            "environment_artifacts": rollout_requirements[
                "environment_artifacts"
            ],
            "requested_seed": rollout_requirements["requested_seed"],
            "applied_seed": rollout_requirements["requested_seed"],
            "requested_n_episodes": rollout_requirements[
                "requested_n_episodes"
            ],
            "configured_n_episodes": rollout_requirements[
                "requested_n_episodes"
            ],
            "requested_max_episode_steps": rollout_requirements[
                "requested_max_episode_steps"
            ],
            "configured_max_episode_steps": rollout_requirements[
                "requested_max_episode_steps"
            ],
            "requested_task_id": rollout_requirements["requested_task_id"],
            "configured_task_id": rollout_requirements["requested_task_id"],
        },
    }
    try:
        archive = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TrackError(f"cannot load Tier-D rollout artifact {path}: {exc}") from exc
    with archive as npz:
        required = {
            "trajectory_contract_json",
            "first_episode_valid_mask",
            "joint_pos",
            "root_link_pos_w",
        }
        missing = sorted(required - set(npz.files))
        if missing:
            raise TrackError(
                f"Tier-D rollout artifact is missing required channels: {missing}"
            )
        try:
            raw_contract = np.asarray(npz["trajectory_contract_json"])
            if raw_contract.ndim != 0:
                raise ValueError("must be a scalar JSON string")
            observed_metadata = json.loads(str(raw_contract.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrackError(
                f"Tier-D trajectory contract is invalid: {exc}"
            ) from exc
        observed_runtime = (
            observed_metadata.get("runtime_artifacts")
            if isinstance(observed_metadata, dict) else None
        )
        configured_num_envs = (
            observed_runtime.get("configured_num_envs")
            if isinstance(observed_runtime, dict) else None
        )
        if (
            not isinstance(configured_num_envs, int)
            or isinstance(configured_num_envs, bool)
            or configured_num_envs < rollout_requirements["requested_n_episodes"]
        ):
            raise TrackError(
                "Tier-D trajectory contract has no valid configured_num_envs"
            )
        completed_first_episodes = (
            observed_runtime.get("completed_first_episodes")
            if isinstance(observed_runtime, dict) else None
        )
        if (
            not isinstance(completed_first_episodes, int)
            or isinstance(completed_first_episodes, bool)
            or completed_first_episodes < 0
            or completed_first_episodes
            > rollout_requirements["requested_n_episodes"]
        ):
            raise TrackError(
                "Tier-D trajectory contract has invalid completed episode facts"
            )
        try:
            seed_application = _canonical_seed_application(
                observed_runtime.get("seed_application"),
                requested_seed=rollout_requirements["requested_seed"],
            )
        except TrackError as exc:
            raise TrackError(f"Tier-D rollout {exc}") from exc
        try:
            env_spec_application = _canonical_application_receipt(
                observed_runtime.get("env_spec_application"),
                schema="reward-sculptor-env-spec-application-v1",
                phase="rollout",
            )
            eval_reset_application = _canonical_application_receipt(
                observed_runtime.get("eval_reset_application"),
                schema="reward-sculptor-eval-reset-application-v1",
            )
        except TrackError as exc:
            raise TrackError(f"Tier-D rollout {exc}") from exc
        expected_metadata["runtime_artifacts"][
            "seed_application"
        ] = seed_application
        expected_metadata["runtime_artifacts"][
            "env_spec_application"
        ] = env_spec_application
        expected_metadata["runtime_artifacts"][
            "eval_reset_application"
        ] = eval_reset_application
        expected_metadata["runtime_artifacts"][
            "configured_num_envs"
        ] = configured_num_envs
        expected_metadata["runtime_artifacts"][
            "completed_first_episodes"
        ] = completed_first_episodes
        if observed_metadata != expected_metadata:
            raise TrackError(
                "Tier-D trajectory contract differs from the certified "
                "joint order/cadence/root frame/lane/mask semantics"
            )

        joint_pos = np.asarray(npz["joint_pos"])
        root_pos = np.asarray(npz["root_link_pos_w"])
        valid_mask = np.asarray(npz["first_episode_valid_mask"])
        if joint_pos.ndim != 3 or joint_pos.shape[2] != len(expected_joints):
            raise TrackError(
                "Tier-D joint_pos must have exact shape (T, E, ordered_joints)"
            )
        if root_pos.ndim != 3 or root_pos.shape[2] != 3:
            raise TrackError("Tier-D root_link_pos_w must have shape (T, E, 3)")
        if valid_mask.ndim != 2:
            raise TrackError(
                "Tier-D first_episode_valid_mask must have shape (T, E)"
            )
        if joint_pos.shape[:2] != root_pos.shape[:2] or joint_pos.shape[:2] != (
            valid_mask.shape[0], valid_mask.shape[1]
        ):
            raise TrackError(
                "Tier-D rollout state channels and valid mask have mismatched "
                "time/environment dimensions"
            )
        if joint_pos.shape[1] != configured_num_envs:
            raise TrackError(
                "Tier-D configured_num_envs differs from trajectory array shape"
            )
        if lane < 0 or lane >= joint_pos.shape[1]:
            raise TrackError(f"Tier-D precommitted lane {lane} is unavailable")
        if not np.isfinite(joint_pos).all() or not np.isfinite(root_pos).all():
            raise TrackError("Tier-D rollout contains non-finite state values")
        if valid_mask.dtype != np.bool_:
            if not np.isin(valid_mask, (0, 1)).all():
                raise TrackError("Tier-D valid mask must contain only booleans")
        lane_mask = valid_mask[:, lane].astype(bool, copy=False)
        valid_count = int(np.sum(lane_mask))
        if valid_count < 2:
            raise TrackError("Tier-D rollout has fewer than two valid samples")
        if not lane_mask[:valid_count].all() or lane_mask[valid_count:].any():
            raise TrackError(
                "Tier-D valid mask must be one true prefix with no re-entry"
            )
        observed_duration_s = valid_count * expected_dt
        certified_duration_s = float(reference["playback_duration_s"])
        if observed_duration_s > certified_duration_s + expected_dt + 1e-12:
            raise TrackError(
                "Tier-D rollout valid prefix exceeds the certified reference "
                "duration by more than one terminal control step"
            )
        gravity = None
        if "projected_gravity_b" in npz.files:
            gravity_all = np.asarray(npz["projected_gravity_b"])
            if gravity_all.shape != (*joint_pos.shape[:2], 3):
                raise TrackError(
                    "Tier-D projected_gravity_b must have shape (T, E, 3)"
                )
            if not np.isfinite(gravity_all).all():
                raise TrackError(
                    "Tier-D rollout contains non-finite gravity values"
                )
            gravity = gravity_all[:valid_count, lane, :]
        elif clip.get("root_quat_wxyz") is not None:
            raise TrackError(
                "Tier-D rollout omitted gravity tracked by the certified reward"
            )

        if clip.get("root_frame") != reference.get("root_frame"):
            raise TrackError(
                "Tier-D rollout scorer root convention differs from the "
                "certified reference"
            )
        (
            target_names,
            target_joint_pos,
            _target_joint_vel,
            target_root_z,
            target_gravity,
        ) = _tracking_targets_from_clip(
            clip,
            n_phase_targets=int(reference["phase_target_count"]),
        )
        if target_names != expected_joints:
            raise TrackError(
                "Tier-D target-table joint order differs from execution contract"
            )
        # These are the exact numeric literals embedded in the generated
        # reward, not the higher-precision native clip samples.  Certification
        # must score the target schedule the policy actually optimized.
        target_joint_pos = np.round(target_joint_pos, 5)
        target_root_z = np.round(target_root_z, 5)
        if target_gravity is not None:
            target_gravity = np.round(target_gravity, 5)
        phase = np.clip(
            (
                (np.arange(valid_count, dtype=np.float64) + 1.0)
                * expected_dt
            ) / float(reference["playback_duration_s"]),
            0.0,
            0.999999,
        )
        target_indices = np.clip(
            np.floor(phase * target_joint_pos.shape[0]).astype(np.int64),
            0,
            target_joint_pos.shape[0] - 1,
        )
        return compute_tracking_errors(
            clip=clip,
            rollout_joint_pos=joint_pos[:valid_count, lane, :],
            rollout_root_z=root_pos[:valid_count, lane, 2],
            rollout_joint_names=expected_joints,
            rollout_gravity=gravity,
            control_hz=1.0 / expected_dt,
            rollout_samples_are_post_step=True,
            scheduled_target_joint_pos=target_joint_pos[target_indices],
            scheduled_target_root_z=target_root_z[target_indices],
            scheduled_target_root_anchor=float(target_root_z[0]),
            scheduled_target_gravity=(
                target_gravity[target_indices]
                if target_gravity is not None else None
            ),
        )


# ── donor-config templating ─────────────────────────────────────────────
def _read_adapter_config_file(config_path: Path) -> dict[str, Any]:
    """Read one adapter table without instantiating runtime code."""
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(config_path)
    if not config_path.is_file():
        raise TrackError(f"adapter config does not exist: {config_path}")
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    adapter_cfg = cfg.get("adapter")
    if not adapter_cfg or "class" not in adapter_cfg:
        raise TrackError(
            f"{config_path}: missing [adapter] section or [adapter].class"
        )
    return {
        "class": adapter_cfg["class"],
        "config": adapter_cfg.get("config", {}) or {},
    }


def _configured_remote_environment() -> list[str]:
    return sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("SCULPTOR_REMOTE_") and str(value).strip()
    )


def _assert_local_tierd_configuration(config_path: Path) -> None:
    """Refuse remote Tier-D until remote runtime identities are observable.

    A local config hash cannot prove which remote checkout, container, driver,
    simulator, or produced bytes actually executed.  Until the remote runner
    returns those observed identities, certification is local-only.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(config_path)
    try:
        with config_path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise TrackError(f"cannot inspect Tier-D config for remote use: {exc}") from exc
    adapter = payload.get("adapter")
    adapter_config = (
        adapter.get("config")
        if isinstance(adapter, dict) and isinstance(adapter.get("config"), dict)
        else {}
    )
    if "remote" in payload or "remote" in adapter_config:
        raise TrackError(
            "Tier-D remote execution is refused until observed remote runtime "
            "identities are part of the certificate"
        )
    remote_environment = _configured_remote_environment()
    if remote_environment:
        raise TrackError(
            "Tier-D remote execution environment is refused until observed "
            "remote runtime identities are part of the certificate: "
            + ", ".join(remote_environment)
        )


def _assert_local_tierd_adapter(adapter: Any) -> None:
    """Re-check local execution after construction and before each GPU call."""
    observed_class = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
    if observed_class != TIER_D_TRUSTED_ADAPTER_CLASS:
        raise TrackError(
            "constructed Tier-D adapter is not the trusted local "
            f"MjlabAdapter: {observed_class!r}"
        )
    remote_environment = _configured_remote_environment()
    if remote_environment:
        raise TrackError(
            "Tier-D remote environment appeared after preflight: "
            + ", ".join(remote_environment)
        )
    remote_enabled = getattr(adapter, "_remote_enabled", None)
    if callable(remote_enabled):
        try:
            enabled = bool(remote_enabled())
        except Exception as exc:  # noqa: BLE001 - fail closed on authority
            raise TrackError(
                "cannot prove the constructed Tier-D adapter is local: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if enabled:
            raise TrackError(
                "Tier-D remote adapter execution is refused until observed "
                "remote runtime identities are part of the certificate"
            )


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
    own tomllib reader ever produces.

    The generated tracker owns its reference-derived ``env/current.json`` and
    ``env/eval_reset.json``.  Donor-local overrides for those two inputs are
    deliberately removed so adapter convention resolution cannot silently run
    a different reset/spec.  World selection remains an explicit independent
    choice and is preserved.
    """
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

    tracker_config = {
        key: value
        for key, value in adapter_cfg["config"].items()
        if key not in {"env_spec_path", "eval_reset_path"}
    }
    config_lines = [
        f"{key} = {_toml_value(value)}"
        for key, value in tracker_config.items()
    ]
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


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _claim_fresh_tierd_project_dir(
    project_dir: Path,
    *,
    donor_project: Path,
    protected_paths: tuple[Path, ...] = (),
) -> Path:
    """Atomically claim a new work directory outside every retained input."""
    try:
        donor = Path(donor_project).expanduser().resolve(strict=True)
    except OSError as exc:
        raise TrackError(
            f"cannot resolve Tier-D donor project {donor_project}: {exc}"
        ) from exc
    requested = Path(project_dir).expanduser()
    candidate = requested.resolve(strict=False)
    if candidate.exists() or candidate.is_symlink():
        raise TrackError(
            f"Tier-D project_dir must be fresh and non-existing: {candidate}"
        )
    forbidden = (donor, *(Path(path).expanduser().resolve() for path in protected_paths))
    for protected in forbidden:
        if _paths_overlap(candidate, protected):
            raise TrackError(
                "Tier-D project_dir must be distinct from donor/library/source "
                f"paths: {candidate} overlaps {protected}"
            )
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = candidate.parent.resolve(strict=True)
        candidate = resolved_parent / candidate.name
        for protected in forbidden:
            if _paths_overlap(candidate, protected):
                raise TrackError(
                    "Tier-D project_dir resolves into donor/library/source "
                    f"paths: {candidate} overlaps {protected}"
                )
        candidate.mkdir(exist_ok=False)
        claimed = candidate.resolve(strict=True)
    except FileExistsError as exc:
        raise TrackError(
            f"Tier-D project_dir was claimed concurrently: {candidate}"
        ) from exc
    except OSError as exc:
        raise TrackError(
            f"cannot atomically claim Tier-D project_dir {candidate}: {exc}"
        ) from exc
    if claimed != candidate or claimed.is_symlink():  # pragma: no cover - race
        raise TrackError("Tier-D project_dir changed while being claimed")
    return claimed


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
    sim_timing: Optional[_timing.SimTiming] = None,
    protected_paths: tuple[Path, ...] = (),
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

    project_dir = _claim_fresh_tierd_project_dir(
        project_dir,
        donor_project=donor_project,
        protected_paths=protected_paths,
    )

    adapter_cfg = read_donor_adapter_config(donor_project)
    config_path = write_project_config_toml(project_dir, adapter_cfg)
    if sim_timing is None:
        task_id = str(adapter_cfg.get("config", {}).get("task_id") or "")
        sim_timing = _timing.timing_for_task(task_id)
    effective_control_hz = (
        sim_timing.control_hz if sim_timing is not None else float(control_hz)
    )
    if effective_control_hz <= 0.0:
        raise TrackError("tracking project control rate must be positive")

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
    duration_s = reference_playback_duration_s(
        frame_count=n_frames, fps=fps,
    )
    episode_len_steps = max(1, int(round(duration_s * effective_control_hz)))
    (
        target_joint_names,
        target_joint_pos,
        target_joint_vel,
        target_root_z,
        target_gravity,
    ) = _tracking_targets_from_clip(
        clip, n_phase_targets=n_phase_targets,
    )
    if target_joint_names != [str(name) for name in joint_names]:
        raise TrackError("tracking target joint order changed during project build")
    # Orientation, per OGMP Eq. 8. Downsample the derived gravity rather than
    # the quaternion: averaging quaternion components across a phase window is
    # not a rotation, while averaging unit gravity vectors is a well-defined
    # (if approximate) direction. Clips without a quaternion get None, which
    # zeroes the term rather than fabricating an upright target.
    # Say out loud whether this reference is even representable at the task's
    # control rate. Both Tier-D timing failures were silent; a phase clock that
    # cannot visit all its targets, or a reference with content above Nyquist,
    # should be visible before the GPU time is spent rather than inferred from
    # a flat learning curve afterwards.
    timing_findings = (
        _timing.validate_timing(
            sim_timing,
            reference_fps=fps,
            reference_duration_s=duration_s,
            n_phase_targets=n_phase_targets,
        )
        if sim_timing is not None else []
    )
    for finding in timing_findings:
        print(f"[track] timing: {finding}", file=sys.stderr, flush=True)

    reward_source = generate_tracking_reward_source(
        clip_id=clip_id,
        robot=robot,
        joint_names=joint_names,
        target_joint_pos=target_joint_pos,
        target_joint_vel=target_joint_vel,
        target_root_z=target_root_z,
        episode_len_steps=episode_len_steps,
        duration_s=duration_s,
        target_gravity=target_gravity,
        root_frame=clip_root_frame(clip),
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
def _content_addressed_rollout_name(sha256: str) -> str:
    if not _is_sha256(sha256):
        raise TrackError("Tier-D rollout sha256 is invalid")
    return f"tierD_rollout_{sha256}.npz"


def _is_server_owned_rollout_path(
    path: Path,
    *,
    clip_dir: Path,
    sha256: Optional[str] = None,
) -> bool:
    """Require one immutable digest name inside the exact clip directory."""
    try:
        resolved = path.resolve(strict=True)
        resolved_clip_dir = clip_dir.resolve(strict=True)
    except OSError:
        return False
    if resolved.parent != resolved_clip_dir:
        return False
    if sha256 is None or not _is_sha256(sha256):
        return False
    return resolved.name == _content_addressed_rollout_name(sha256)


def _fsync_directory(path: Path, *, label: str) -> None:
    """Persist a directory-entry update before publishing dependent facts."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as exc:
        raise TrackError(f"cannot open {label} directory for fsync: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise TrackError(f"cannot fsync {label} directory: {exc}") from exc
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int, *, label: str) -> None:
    """Persist entries through an already-pinned directory descriptor."""
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise TrackError(f"cannot fsync {label} directory: {exc}") from exc


def _exact_provenance_identity_issues(
    provenance: Any,
    *,
    robot: str,
    clip_id: str,
) -> list[str]:
    """Validate the modern robot-scoped provenance authority itself."""
    from sculptor.refs import library

    if not isinstance(provenance, dict):
        return ["provenance must be a JSON object"]
    issues: list[str] = []
    if provenance.get("schema") != library.PROVENANCE_SCHEMA:
        issues.append(
            "provenance schema is not the current immutable artifact schema"
        )
    if provenance.get("robot") != robot:
        issues.append("provenance.robot does not match its robot-scoped path")
    if provenance.get("clip_id") != clip_id:
        issues.append("provenance.clip_id does not match its clip-scoped path")
    try:
        issues.extend(library.validate_provenance(provenance))
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(
            "provenance validation failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return issues


def _materialize_tierd_rollout_artifact(
    source: Path,
    *,
    clip_dir: Path,
    clip: dict[str, Any],
    execution_contract: dict[str, Any],
    lane: int,
    expected_errors: TrackingErrors,
    library_root: Path,
) -> Path:
    """Atomically retain exact rollout bytes under their immutable identity.

    All writes are relative to a pinned, no-follow clip-directory descriptor.
    A path-level confinement check alone is insufficient because the checked
    directory can be exchanged for a symlink before ``mkstemp`` or ``link``.
    """
    from sculptor.refs import library

    source = Path(source)
    digest = _file_sha256(source, label="Tier-D rollout artifact")
    try:
        resolved_root = Path(library_root).expanduser().resolve(strict=True)
        resolved_clip_dir = Path(clip_dir).resolve(strict=True)
        relative = resolved_clip_dir.relative_to(resolved_root)
        if len(relative.parts) != 2:
            raise ValueError("clip path must be exactly root/robot/clip_id")
        robot = library.validate_robot_namespace(relative.parts[0])
        clip_id = library.validate_clip_id(relative.parts[1])
        admitted_clip_dir = library.require_confined_clip_dir(
            robot, clip_id, root=resolved_root,
        )
        if (
            Path(clip_dir).is_symlink()
            or Path(clip_dir).parent.is_symlink()
            or admitted_clip_dir.resolve(strict=True) != resolved_clip_dir
        ):
            raise ValueError("clip directory is linked or stale")
    except (OSError, TypeError, ValueError) as exc:
        raise TrackError(
            "Tier-D rollout publication path is not a confined retained "
            f"clip directory: {exc}"
        ) from exc

    destination_name = _content_addressed_rollout_name(digest)
    destination = admitted_clip_dir / destination_name
    try:
        with library._pinned_confined_clip_dir(
            robot, clip_id, root=resolved_root,
        ) as (pinned_clip_dir, directory_fd):
            if pinned_clip_dir != admitted_clip_dir:
                raise TrackError(
                    "Tier-D rollout publication coordinate changed before pinning"
                )
            retained_bytes = library._read_regular_file_at(
                directory_fd, destination_name, required=False,
            )
            if retained_bytes is None:
                nofollow = getattr(os, "O_NOFOLLOW", 0)
                if os.name != "posix" or not nofollow:
                    raise TrackError(
                        "secure no-follow Tier-D rollout publication is unavailable"
                    )
                temporary_name: Optional[str] = None
                temporary_fd = -1
                for _attempt in range(32):
                    candidate = (
                        f".{destination_name}.{os.urandom(12).hex()}.tmp"
                    )
                    try:
                        temporary_fd = os.open(
                            candidate,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                            0o600,
                            dir_fd=directory_fd,
                        )
                    except FileExistsError:  # pragma: no cover - entropy
                        continue
                    temporary_name = candidate
                    break
                if temporary_name is None or temporary_fd < 0:
                    raise TrackError(
                        "cannot allocate Tier-D rollout temporary member"
                    )
                try:
                    with os.fdopen(
                        temporary_fd, "wb", closefd=True,
                    ) as target, source.open("rb") as origin:
                        temporary_fd = -1
                        shutil.copyfileobj(origin, target)
                        target.flush()
                        os.fsync(target.fileno())
                    candidate_bytes = library._read_regular_file_at(
                        directory_fd, temporary_name, required=True,
                    )
                    assert candidate_bytes is not None
                    if hashlib.sha256(candidate_bytes).hexdigest() != digest:
                        raise TrackError(
                            "Tier-D rollout changed while being copied"
                        )
                    try:
                        os.link(
                            temporary_name,
                            destination_name,
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        pass
                    _fsync_directory_descriptor(
                        directory_fd, label="Tier-D rollout",
                    )
                finally:
                    if temporary_fd >= 0:
                        os.close(temporary_fd)
                    try:
                        os.unlink(temporary_name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                retained_bytes = library._read_regular_file_at(
                    directory_fd, destination_name, required=True,
                )
            assert retained_bytes is not None
            if hashlib.sha256(retained_bytes).hexdigest() != digest:
                raise TrackError("content-addressed Tier-D rollout path is corrupt")
            if not library._confined_clip_coordinate_matches_fd(
                robot,
                clip_id,
                root=resolved_root,
                expected_fd=directory_fd,
            ):
                raise TrackError(
                    "Tier-D rollout publication coordinate changed during write"
                )
            with tempfile.TemporaryDirectory(
                prefix=".tier-d-retained-score-",
            ) as score_dir:
                score_path = Path(score_dir) / "trajectory.npz"
                score_path.write_bytes(retained_bytes)
                retained_errors = _score_tierd_rollout_artifact(
                    score_path,
                    clip=clip,
                    execution_contract=execution_contract,
                    lane=lane,
                )
            if not library._confined_clip_coordinate_matches_fd(
                robot,
                clip_id,
                root=resolved_root,
                expected_fd=directory_fd,
            ):
                raise TrackError(
                    "Tier-D rollout publication coordinate changed during scoring"
                )
    except TrackError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise TrackError(
            f"cannot securely retain Tier-D rollout artifact: {exc}"
        ) from exc
    if retained_errors.to_dict() != expected_errors.to_dict():
        raise TrackError("retained Tier-D rollout scoring evidence changed")
    return destination


def update_provenance_tier_d(
    *,
    robot: str,
    clip_id: str,
    errors: TrackingErrors,
    iterations: int,
    rollout_path: Optional[Path] = None,
    execution_contract: Optional[dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """Atomically publish one Tier-D verdict and its global index row.

    The provenance is authoritative, while ``index.jsonl`` is shared by every
    robot and clip.  Holding one root-scoped cross-process lock across the full
    read-modify-write and rebuild prevents both stale same-clip overwrites and
    a slower rebuild from dropping another clip's newer row.
    """
    from sculptor.refs import library

    with library.reference_library_mutation_lock(root=root):
        prov = _update_provenance_tier_d_locked(
            robot=robot,
            clip_id=clip_id,
            errors=errors,
            iterations=iterations,
            rollout_path=rollout_path,
            execution_contract=execution_contract,
            root=root,
        )
        library._rebuild_index_unlocked(root=root)
        return prov


def _update_provenance_tier_d_locked(
    *,
    robot: str,
    clip_id: str,
    errors: TrackingErrors,
    iterations: int,
    rollout_path: Optional[Path] = None,
    execution_contract: Optional[dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """Read -> mutate -> write the clip's provenance with the Tier-D
    certification result (§mission spec's exact contract): feasible ->
    `tier="D"` + `tierD` block including `rollout_path`; infeasible ->
    `tier="K"`, with `tierD.feasible=False` recorded.
    The caller owns the global index publication and must hold
    ``reference_library_mutation_lock`` across this function and that rebuild.

    §audit-finding close (REFERENCE_BUILD_LOG.md "Audit findings
    deferred" — Tier-D spoofing): the `tierD` block also records
    `source_content_sha256` (a copy of THIS provenance's source-content
    identity at tracking time), `clip_content_sha256` (sha256 of the exact
    canonical `clip.npz` bytes actually tracked), and, when feasible,
    `rollout_sha256` (sha256 of the copied rollout artifact's bytes). Together
    these let
    `verify_tierd_certificate` bind a later "tier D" claim to a
    consistent on-disk artifact chain instead of trusting the `tier`
    field or `tierD.errors.feasible` bool in isolation. Hashing the
    feasible rollout is strict: unreadable, mutable-name, mismatched, or
    unretained bytes fail before any provenance mutation."""
    from sculptor.refs import library

    try:
        effective_root = Path(root or library.references_root()).expanduser().resolve()
        confined_clip_dir = library.require_confined_clip_dir(
            robot, clip_id, root=effective_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise TrackError(
            "cannot resolve confined Tier-D provenance publication path: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # A feasible rollout without exact execution evidence is still useful
    # diagnostic output, but it is not a Tier-D certificate.  Fail before
    # mutating provenance rather than letting an unscoped "D" claim escape.
    if execution_contract is None and errors.feasible:
        raise TrackError(
            "feasible Tier-D provenance requires an execution contract"
        )
    if errors.feasible and _configured_remote_environment():
        raise TrackError(
            "feasible Tier-D provenance cannot be published while a remote "
            "execution environment is configured"
        )
    if execution_contract is not None:
        issues = validate_tierd_execution_contract(execution_contract)
        if issues:
            raise TrackError(
                "cannot record invalid Tier-D execution contract: "
                + "; ".join(issues)
            )
        certified_robot = execution_contract["execution_boundary"]["robot"]
        if certified_robot != robot:
            raise TrackError(
                f"execution contract robot {certified_robot!r} does not match "
                f"provenance robot {robot!r}"
            )
        certified_reference = execution_contract["reference"]
        if certified_reference.get("clip_id") != clip_id:
            raise TrackError(
                "execution contract clip id does not match provenance clip id"
            )
        if errors.feasible and errors.common_joint_names != certified_reference[
            "ordered_joints"
        ]:
            raise TrackError(
                "feasible Tier-D provenance requires exact full ordered-joint "
                "coverage from the certified reference"
            )
        if errors.feasible:
            runtime_artifacts = execution_contract.get("runtime_artifacts")
            if not isinstance(runtime_artifacts, dict):
                raise TrackError(
                    "feasible Tier-D provenance requires bound runtime artifacts"
                )
            requested_training = runtime_artifacts.get("requested_training")
            if (
                not isinstance(requested_training, dict)
                or requested_training.get("iterations") != iterations
            ):
                raise TrackError(
                    "Tier-D provenance iterations differ from the exact "
                    "training request"
                )

    prov = library.read_provenance(robot, clip_id, root=effective_root)
    provenance_issues = _exact_provenance_identity_issues(
        prov, robot=robot, clip_id=clip_id,
    )
    if provenance_issues:
        raise TrackError(
            "cannot promote invalid or mis-scoped provenance: "
            + "; ".join(provenance_issues)
        )
    declaration_evidence, declaration_issues = (
        library.root_frame_declaration_evidence_from_provenance(prov)
    )
    if declaration_issues and errors.feasible:
        raise TrackError(
            "feasible Tier-D provenance requires structured root-frame "
            "declaration evidence: " + "; ".join(declaration_issues)
        )
    certified_root_frame = (
        execution_contract["reference"].get("root_frame")
        if execution_contract is not None
        else None
    )
    root_frame_inheritance, inheritance_issues = (
        library.root_frame_inheritance_from_provenance(
            prov,
            root=effective_root,
            expected_root_frame=certified_root_frame,
        )
    )
    if inheritance_issues and errors.feasible:
        raise TrackError(
            "feasible Tier-D provenance requires valid root-frame "
            "inheritance: " + "; ".join(inheritance_issues)
        )
    if execution_contract is not None and (
        execution_contract["reference"].get(
            "root_frame_declaration_evidence"
        )
        != declaration_evidence
    ):
        raise TrackError(
            "Tier-D execution contract root-frame declaration evidence "
            "differs from current provenance"
        )
    if execution_contract is not None and (
        execution_contract["reference"].get("root_frame_inheritance")
        != root_frame_inheritance
    ):
        raise TrackError(
            "Tier-D execution contract root-frame inheritance differs from "
            "current parent artifacts"
        )
    retained_rollout_sha: Optional[str] = None
    if errors.feasible:
        if rollout_path is None:
            raise TrackError(
                "feasible Tier-D provenance requires the exact server-owned "
                "content-addressed rollout artifact"
            )
        try:
            actual_rollout = Path(rollout_path).resolve(strict=True)
            retained_rollout_sha = library.content_sha256(
                actual_rollout.read_bytes()
            )
        except OSError as exc:
            raise TrackError(
                f"feasible Tier-D rollout artifact is unreadable: {exc}"
            ) from exc
        if not _is_server_owned_rollout_path(
            actual_rollout,
            clip_dir=confined_clip_dir,
            sha256=retained_rollout_sha,
        ):
            raise TrackError(
                "feasible Tier-D rollout must use its exact server-owned "
                "content-addressed path"
            )
    tier_d_block: dict[str, Any] = {
        "tracked_at": library._utc_now_iso(),
        "iterations": iterations,
        "errors": errors.to_dict(),
        "source_content_sha256": prov.get("source_content_sha256"),
    }
    clip_path = confined_clip_dir / library.CLIP_FILENAME
    try:
        clip_artifact_sha256 = library.content_sha256(
            clip_path.read_bytes()
        )
    except OSError as exc:
        if errors.feasible:
            raise TrackError(
                "feasible Tier-D provenance requires readable exact clip "
                f"bytes at {clip_path}: {exc}"
            ) from exc
    else:
        if prov.get("content_sha256") != clip_artifact_sha256:
            raise TrackError(
                "provenance content_sha256 does not identify the exact clip.npz "
                "bytes; repair/re-ingest before Tier-D certification"
            )
        tier_d_block["clip_content_sha256"] = clip_artifact_sha256
    if errors.feasible:
        assert rollout_path is not None
        assert execution_contract is not None
        try:
            from sculptor.reference import load_clip

            exact_clip = load_clip(clip_path)
            recomputed_errors = _score_tierd_rollout_artifact(
                rollout_path,
                clip=exact_clip,
                execution_contract=execution_contract,
                lane=int(execution_contract["reference"]["rollout_lane"]),
            )
        except (OSError, KeyError, TypeError, ValueError, TrackError) as exc:
            raise TrackError(
                "cannot recompute feasible Tier-D rollout before promotion: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if recomputed_errors.to_dict() != errors.to_dict():
            raise TrackError(
                "feasible Tier-D errors differ from the exact retained rollout"
            )
        errors = recomputed_errors
        tier_d_block["errors"] = errors.to_dict()
    if execution_contract is not None:
        tier_d_block["execution_contract"] = execution_contract
        tier_d_block["execution_contract_sha256"] = execution_contract[
            "contract_sha256"
        ]
        tier_d_block["execution_boundary_sha256"] = execution_contract[
            "execution_boundary_sha256"
        ]
    if errors.feasible:
        prov["tier"] = "D"
        assert rollout_path is not None
        tier_d_block["rollout_path"] = str(Path(rollout_path).resolve())
        try:
            final_rollout_sha = library.content_sha256(
                Path(rollout_path).read_bytes()
            )
        except OSError as exc:
            raise TrackError(
                f"cannot hash exact Tier-D rollout artifact: {exc}"
            ) from exc
        if final_rollout_sha != retained_rollout_sha:
            raise TrackError("Tier-D rollout bytes changed during promotion")
        tier_d_block["rollout_sha256"] = final_rollout_sha
    else:
        # A fresh failed recertification invalidates any older D authority for
        # these mutable library coordinates. Keep the diagnostic block, but do
        # not leave a stale certificate active.
        prov["tier"] = "K"
        tier_d_block["feasible"] = False
    prov["tierD"] = tier_d_block
    library.write_provenance(robot, clip_id, prov, root=effective_root)
    _fsync_directory(
        confined_clip_dir,
        label="Tier-D provenance",
    )
    return prov


def _publish_and_verify_tierd_verdict(
    *,
    robot: str,
    clip_id: str,
    errors: TrackingErrors,
    iterations: int,
    rollout_path: Optional[Path] = None,
    execution_contract: Optional[dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """Publish one tracking verdict and fail closed in one transaction.

    A feasible verdict is written so the normal certificate verifier can
    re-read and recompute the exact on-disk chain.  The same library-wide lock
    remains held through that verification, any required Tier-K invalidation,
    and the sole final index rebuild.  A concurrent recertification therefore
    cannot be overwritten afterward by a stale failed self-check.
    """
    from sculptor.refs import library

    denial: Optional[str] = None
    verification_failed = False
    with library.reference_library_mutation_lock(root=root):
        prov = _update_provenance_tier_d_locked(
            robot=robot,
            clip_id=clip_id,
            errors=errors,
            iterations=iterations,
            rollout_path=rollout_path,
            execution_contract=execution_contract,
            root=root,
        )
        if errors.feasible:
            try:
                certificate, denial = verify_tierd_certificate(
                    robot, clip_id, root=root,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed on self-check
                certificate = None
                denial = f"{type(exc).__name__}: {exc}"
            if certificate is None:
                verification_failed = True
                # Never leave unverified launch authority behind.  This uses
                # the just-written object while the mutation lock still excludes
                # every competing recertification of these coordinates.
                prov["tier"] = "K"
                tier_d = prov.get("tierD")
                if isinstance(tier_d, dict):
                    tier_d["feasible"] = False
                    tier_d["verification_error"] = str(denial or "unknown")
                library.write_provenance(robot, clip_id, prov, root=root)
                _fsync_directory(
                    library.clip_dir(robot, clip_id, root=root),
                    label="Tier-D invalidation provenance",
                )
        library._rebuild_index_unlocked(root=root)

    if verification_failed:
        raise TrackError(
            "Tier-D rollout passed numeric gates but exact certificate "
            f"self-verification failed: {denial or 'unknown reason'}"
        )
    return prov


# ── orchestration ────────────────────────────────────────────────────────
@dataclass
class TrackResult:
    plan: TrackPlan
    errors: Optional[TrackingErrors]
    provenance: dict[str, Any]
    dry_run: bool
    preflight_receipt: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class _TierDTrackingPreflight:
    """CPU-only boundary shared by dry-run and live Tier-D execution."""

    plan: TrackPlan
    policy_contract: dict[str, Any] = field(repr=False)
    execution_contract: dict[str, Any] = field(repr=False)
    requested_reward_sha256: str
    requested_num_envs: int
    receipt: dict[str, Any] = field(repr=False)
    retained_initial_checkpoint_path: Optional[Path] = None
    initial_checkpoint_receipt: Optional[dict[str, Any]] = field(
        default=None, repr=False,
    )


def _prepare_tierd_tracking_preflight(
    *,
    clip: dict[str, Any],
    clip_content_sha256: str,
    clip_id: str,
    robot: str,
    donor_project: Path,
    project_dir: Path,
    iterations: int,
    steps_per_iteration: int,
    n_episodes: int,
    seed: int,
    resume_checkpoint: Optional[Path] = None,
    protected_paths: tuple[Path, ...] = (),
    root_frame_declaration_evidence: Optional[dict[str, Any]] = None,
    root_frame_inheritance: Optional[dict[str, Any]] = None,
) -> _TierDTrackingPreflight:
    """Build and validate the complete pre-GPU Tier-D execution boundary.

    This helper intentionally never calls ``load_adapter``.  The mjlab
    adapter constructor validates CUDA and may query GPU memory; a dry-run is
    instead authoritative over every fact that can be proven from config and
    immutable CPU-readable inputs.  The live path consumes this same result,
    then constructs the adapter and compares its resolved environment inputs
    before any training call.
    """
    try:
        donor_interface = _read_tierd_donor_interface(
            donor_project,
            robot=robot,
        )
        donor_project = donor_interface.donor_project
        donor_policy_contract = donor_interface.policy_contract
        donor_boundary = _policy_execution_boundary(
            robot=robot, policy_contract=donor_policy_contract,
        )
        donor_timing = donor_boundary["timing"]
        sim_timing = _timing.SimTiming(
            physics_dt=float(donor_timing["sim_timestep_s"]),
            decimation=int(donor_timing["decimation"]),
        )
    except Exception as exc:  # noqa: BLE001 - normalized setup failure
        if isinstance(exc, TrackError):
            raise
        raise TrackError(
            "cannot capture donor adapter/interface/config boundary before "
            f"tracking: {type(exc).__name__}: {exc}"
        ) from exc

    plan = build_track_project(
        clip=clip,
        clip_id=clip_id,
        robot=robot,
        donor_project=donor_project,
        project_dir=project_dir,
        iterations=iterations,
        steps_per_iteration=steps_per_iteration,
        n_episodes=n_episodes,
        sim_timing=sim_timing,
        protected_paths=protected_paths,
    )
    try:
        reference_clock = reference_clock_from_reward_source(
            plan.reward_path.read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError) as exc:
        raise TrackError(
            "generated Tier-D reward has no valid immutable reference clock"
        ) from exc
    if reference_clock is None:  # pragma: no cover - generator invariant
        raise TrackError("generated Tier-D reward omitted its reference clock")

    environment_artifacts = _configured_environment_artifacts(plan.config_path)
    try:
        policy_contract = _build_generated_tracker_policy_contract(
            donor_policy_contract,
            reference_clock=reference_clock,
        )
    except Exception as exc:  # noqa: BLE001 - normalized setup failure
        raise TrackError(
            "cannot build clock-conditioned Tier-D tracker policy contract: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    generated_boundary = _policy_execution_boundary(
        robot=robot, policy_contract=policy_contract,
    )
    if generated_boundary != donor_boundary:
        raise TrackError(
            "generated tracker execution boundary differs from the donor "
            "adapter/interface/config boundary"
        )

    execution_contract = build_tierd_execution_contract(
        donor_project=donor_project,
        certification_config_path=plan.config_path,
        clip_id=clip_id,
        robot=robot,
        clip=clip,
        n_phase_targets=plan.n_phase_targets,
        policy_contract=policy_contract,
        reference_clock=reference_clock,
        environment_artifacts=environment_artifacts,
        root_frame_declaration_evidence=(
            root_frame_declaration_evidence
        ),
        root_frame_inheritance=root_frame_inheritance,
    )
    contract_issues = validate_tierd_execution_contract(execution_contract)
    if contract_issues:  # pragma: no cover - builder already fails closed
        raise TrackError(
            "invalid unbound Tier-D execution contract: "
            + "; ".join(contract_issues)
        )
    if (
        _configured_environment_artifacts(plan.config_path)
        != execution_contract["environment_artifacts"]
    ):
        raise TrackError(
            "configured environment inputs changed during Tier-D preflight"
        )
    consumed_config_sha = _file_sha256(
        plan.config_path, label="generated certification config.toml",
    )
    if consumed_config_sha != donor_interface.certification_config_sha256:
        raise TrackError(
            "generated certification config differs from the donor's "
            "exported Tier-D interface receipt"
        )
    if consumed_config_sha != execution_contract["donor"][
        "certification_config_sha256"
    ]:
        raise TrackError(
            "generated certification config changed during Tier-D preflight"
        )
    requested_reward_sha256 = _file_sha256(
        plan.reward_path, label="generated Tier-D reward module",
    )
    requested_num_envs = _read_adapter_config_file(plan.config_path).get(
        "config", {}
    ).get("num_envs")
    if (
        not isinstance(requested_num_envs, int)
        or isinstance(requested_num_envs, bool)
        or requested_num_envs < 1
    ):
        raise TrackError(
            "Tier-D adapter does not expose an exact requested num_envs"
        )

    retained_initial_checkpoint_path: Optional[Path] = None
    initial_checkpoint_receipt: Optional[dict[str, Any]] = None
    if resume_checkpoint is not None:
        retained_initial_checkpoint_path, initial_checkpoint_receipt = (
            _retain_verified_tierd_initial_checkpoint(
                resume_checkpoint,
                project_dir=plan.project_dir,
                expected_policy_contract=policy_contract,
                expected_policy_contract_sha256=execution_contract["donor"][
                    "policy_contract_sha256"
                ],
                expected_environment_artifacts=environment_artifacts_for_phase(
                    environment_artifacts, "train",
                ),
            )
        )

    initialization: dict[str, Any] = {
        "donor_project_role": "adapter_interface_and_config_only",
        "first_tracker_training": "fresh_random_policy",
        "donor_policy_weights_loaded": False,
    }
    if initial_checkpoint_receipt is not None:
        initialization = {
            "donor_project_role": "adapter_interface_and_config_only",
            "first_tracker_training": "verified_tracker_checkpoint",
            "donor_policy_weights_loaded": False,
            "continuation": initial_checkpoint_receipt,
        }
    receipt: dict[str, Any] = {
        "schema": TIER_D_PREFLIGHT_SCHEMA,
        "status": "ready",
        "initialization": initialization,
        "request": {
            "robot": robot,
            "clip_id": clip_id,
            "clip_content_sha256": clip_content_sha256,
            "project_dir": str(plan.project_dir.resolve()),
            "iterations": plan.iterations,
            "steps_per_iteration": plan.steps_per_iteration,
            "n_episodes": plan.n_episodes,
            "seed": int(seed),
            "num_envs": requested_num_envs,
        },
        "artifacts": {
            "reward_module_sha256": requested_reward_sha256,
            "donor_config_sha256": execution_contract["donor"][
                "config_sha256"
            ],
            "certification_config_sha256": execution_contract["donor"][
                "certification_config_sha256"
            ],
            "policy_contract_sha256": execution_contract["donor"][
                "policy_contract_sha256"
            ],
            "donor_interface_receipt_sha256": (
                donor_interface.receipt_sha256
            ),
            "execution_boundary_sha256": execution_contract[
                "execution_boundary_sha256"
            ],
            "unbound_execution_contract_sha256": execution_contract[
                "contract_sha256"
            ],
        },
        "unbound_execution_contract": execution_contract,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return _TierDTrackingPreflight(
        plan=plan,
        policy_contract=policy_contract,
        execution_contract=execution_contract,
        requested_reward_sha256=requested_reward_sha256,
        requested_num_envs=requested_num_envs,
        retained_initial_checkpoint_path=retained_initial_checkpoint_path,
        initial_checkpoint_receipt=initial_checkpoint_receipt,
        receipt=receipt,
    )


def _read_tierd_train_runtime_receipt(
    metrics_path: Path,
    *,
    iteration: int,
) -> dict[str, Any]:
    """Read the subprocess-observed reward/checkpoint/settings receipt."""
    try:
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TrackError(
            f"cannot read Tier-D training runtime receipt: {exc}"
        ) from exc
    raw = payload.get("runtime_artifacts")
    required = {
        "schema",
        "phase",
        "reward_module_sha256",
        "requested_max_iterations",
        "requested_seed",
        "requested_num_envs",
        "seed_application",
        "environment_artifacts",
        "env_spec_application",
        "input_checkpoint_requested_sha256",
        "input_checkpoint_loaded_sha256",
        "input_checkpoint_load_completed",
        "output_checkpoint_sha256",
        "output_policy_contract_sha256",
        "output_policy_contract_sidecar_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise TrackError(
            "Tier-D training runtime receipt is missing or non-canonical"
        )
    return {"iteration": iteration, **raw}


def _verify_checkpoint_policy_contract_sidecar(
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str,
    expected_policy_contract: dict[str, Any],
    expected_policy_contract_sha256: str,
    expected_sidecar_sha256: str,
) -> None:
    """Bind one produced checkpoint to the interface the runner observed."""
    from sculptor.policy_contract import contract_fingerprint

    sidecar_path = Path(str(Path(checkpoint_path)) + ".policy_contract.json")
    if _file_sha256(
        sidecar_path, label="checkpoint policy-contract sidecar",
    ) != expected_sidecar_sha256:
        raise TrackError("checkpoint policy-contract sidecar bytes differ from receipt")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackError(f"cannot read checkpoint policy-contract sidecar: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "checkpoint_sha256", "policy_contract",
        "policy_contract_sha256",
    }:
        raise TrackError("checkpoint policy-contract sidecar is non-canonical")
    observed_contract = payload.get("policy_contract")
    if (
        payload.get("schema") != 1
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or observed_contract != expected_policy_contract
        or payload.get("policy_contract_sha256")
        != expected_policy_contract_sha256
        or not isinstance(observed_contract, dict)
        or contract_fingerprint(observed_contract)
        != expected_policy_contract_sha256
    ):
        raise TrackError(
            "checkpoint policy-contract sidecar differs from the generated "
            "tracker runtime interface"
        )


def _retain_verified_tierd_initial_checkpoint(
    checkpoint_path: Path,
    *,
    project_dir: Path,
    expected_policy_contract: dict[str, Any],
    expected_policy_contract_sha256: str,
    expected_environment_artifacts: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Verify and retain a prior tracker checkpoint before GPU allocation.

    Only an exact runner-produced checkpoint chain is eligible: the adjacent
    policy-contract sidecar must bind the checkpoint bytes to the freshly
    generated tracker interface, and the adjacent metrics receipt must bind
    the same output bytes, sidecar, and training environment.  The three
    inputs are copied into the fresh work directory and re-hashed so a later
    source mutation cannot alter this run.
    """

    def _source_file(path: Path, *, label: str, max_bytes: int) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_symlink():
            raise TrackError(f"{label} must not be a symlink")
        try:
            resolved = candidate.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            raise TrackError(f"cannot resolve {label}: {exc}") from exc
        if not resolved.is_file() or stat.st_size < 1:
            raise TrackError(f"{label} must be a non-empty regular file")
        if stat.st_size > max_bytes:
            raise TrackError(f"{label} exceeds the trusted local size limit")
        return resolved

    source_checkpoint = _source_file(
        checkpoint_path,
        label="Tier-D continuation checkpoint",
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    source_sidecar = _source_file(
        Path(str(source_checkpoint) + ".policy_contract.json"),
        label="Tier-D continuation policy-contract sidecar",
        max_bytes=16 * 1024 * 1024,
    )
    source_metrics = _source_file(
        source_checkpoint.parent / "metrics.json",
        label="Tier-D continuation metrics receipt",
        max_bytes=16 * 1024 * 1024,
    )

    checkpoint_sha = _file_sha256(
        source_checkpoint, label="Tier-D continuation checkpoint",
    )
    sidecar_sha = _file_sha256(
        source_sidecar, label="Tier-D continuation policy-contract sidecar",
    )
    metrics_sha = _file_sha256(
        source_metrics, label="Tier-D continuation metrics receipt",
    )
    runtime = _read_tierd_train_runtime_receipt(source_metrics, iteration=0)
    if (
        runtime["schema"] != RUNNER_RUNTIME_ARTIFACT_SCHEMA
        or runtime["phase"] != "train"
    ):
        raise TrackError(
            "Tier-D continuation runtime receipt schema/phase is invalid"
        )
    if not _is_sha256(runtime["reward_module_sha256"]):
        raise TrackError(
            "Tier-D continuation source reward sha256 is invalid"
        )
    requested_iterations = runtime["requested_max_iterations"]
    requested_seed = runtime["requested_seed"]
    requested_num_envs = runtime["requested_num_envs"]
    if (
        not isinstance(requested_iterations, int)
        or isinstance(requested_iterations, bool)
        or requested_iterations < 1
        or not _is_runtime_seed(requested_seed)
        or not isinstance(requested_num_envs, int)
        or isinstance(requested_num_envs, bool)
        or requested_num_envs < 1
    ):
        raise TrackError(
            "Tier-D continuation runtime training settings are invalid"
        )
    _canonical_seed_application(
        runtime["seed_application"], requested_seed=requested_seed,
    )
    _canonical_application_receipt(
        runtime["env_spec_application"],
        schema="reward-sculptor-env-spec-application-v1",
        phase="train",
    )
    input_requested = runtime["input_checkpoint_requested_sha256"]
    input_loaded = runtime["input_checkpoint_loaded_sha256"]
    input_completed = runtime["input_checkpoint_load_completed"]
    if input_requested is None:
        expected_input = (None, None, False)
    elif _is_sha256(input_requested):
        expected_input = (input_requested, input_requested, True)
    else:
        raise TrackError(
            "Tier-D continuation prior checkpoint sha256 is invalid"
        )
    if (input_requested, input_loaded, input_completed) != expected_input:
        raise TrackError(
            "Tier-D continuation prior checkpoint load receipt is incoherent"
        )
    if runtime["output_checkpoint_sha256"] != checkpoint_sha:
        raise TrackError(
            "Tier-D continuation metrics do not bind the requested checkpoint"
        )
    if runtime["output_policy_contract_sidecar_sha256"] != sidecar_sha:
        raise TrackError(
            "Tier-D continuation metrics do not bind the policy-contract "
            "sidecar"
        )
    if runtime["output_policy_contract_sha256"] != expected_policy_contract_sha256:
        raise TrackError(
            "Tier-D continuation checkpoint uses a different policy contract"
        )
    if runtime["environment_artifacts"] != expected_environment_artifacts:
        raise TrackError(
            "Tier-D continuation checkpoint was trained with different "
            "environment artifacts"
        )
    _verify_checkpoint_policy_contract_sidecar(
        source_checkpoint,
        checkpoint_sha256=checkpoint_sha,
        expected_policy_contract=expected_policy_contract,
        expected_policy_contract_sha256=expected_policy_contract_sha256,
        expected_sidecar_sha256=sidecar_sha,
    )

    retained_dir = project_dir / "initialization"
    try:
        retained_dir.mkdir(exist_ok=False)
        retained_checkpoint = retained_dir / "checkpoint.pt"
        retained_sidecar = Path(
            str(retained_checkpoint) + ".policy_contract.json"
        )
        retained_metrics = retained_dir / "source_metrics.json"
        shutil.copyfile(source_checkpoint, retained_checkpoint)
        shutil.copyfile(source_sidecar, retained_sidecar)
        shutil.copyfile(source_metrics, retained_metrics)
    except OSError as exc:
        raise TrackError(
            f"cannot retain Tier-D continuation artifacts: {exc}"
        ) from exc

    expected_copies = (
        (source_checkpoint, retained_checkpoint, checkpoint_sha),
        (source_sidecar, retained_sidecar, sidecar_sha),
        (source_metrics, retained_metrics, metrics_sha),
    )
    for source, retained, expected_sha in expected_copies:
        if _file_sha256(source, label="Tier-D continuation source") != expected_sha:
            raise TrackError(
                "Tier-D continuation source changed while it was retained"
            )
        if _file_sha256(retained, label="retained Tier-D continuation") != expected_sha:
            raise TrackError(
                "retained Tier-D continuation bytes differ from the source"
            )

    receipt = _canonical_tierd_initial_checkpoint({
        "schema": TIER_D_CONTINUATION_SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": source_checkpoint.stat().st_size,
        "source_reward_module_sha256": runtime["reward_module_sha256"],
        "policy_contract_sha256": expected_policy_contract_sha256,
        "policy_contract_sidecar_sha256": sidecar_sha,
        "source_metrics_sha256": metrics_sha,
        "retained_checkpoint_relpath": "initialization/checkpoint.pt",
        "retained_policy_contract_sidecar_relpath": (
            "initialization/checkpoint.pt.policy_contract.json"
        ),
        "retained_metrics_relpath": "initialization/source_metrics.json",
    })
    return retained_checkpoint, receipt


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
    resume_checkpoint: Optional[Path] = None,
    dry_run: bool = False,
    library_root: Optional[Path] = None,
    progress: Optional[Any] = None,
) -> TrackResult:
    """Full Tier-D certification pipeline for one already-Tier-K clip.

    1. load the clip + its provenance (`sculptor.refs.library`,
       `sculptor.reference.load_clip`);
    2. build the throwaway project and complete its CPU-only donor,
       interface, reference-clock, environment, and unbound execution-contract
       preflight;
    3. `--dry-run` returns that exact receipt without constructing the
       GPU-aware adapter or loading/training any policy weights;
    4. else: instantiate the adapter, verify it resolved the preflight inputs,
       train via `load_adapter(config).train(...)`, roll out via
       `.rollout(...)` (the real minimal programmatic path — see module
       docstring), score the rollout vs the clip
       (`compute_tracking_errors`), copy `trajectory.npz` beside the clip
       under its SHA-256 identity on success, and update provenance
       (`update_provenance_tier_d`).
    """
    from sculptor.reference import load_clip
    from sculptor.refs import library

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    if n_episodes != 1:
        raise TrackError(
            "Tier-D certification currently requires exactly one rollout "
            "lane; multi-lane aggregation is not yet implemented"
        )
    try:
        from sculptor.project_robot import validate_robot_namespace

        robot = validate_robot_namespace(robot)
        library.validate_clip_id(clip_id)
    except (TypeError, ValueError) as exc:
        label = "robot namespace" if "robot namespace" in str(exc) else "clip id"
        raise TrackError(f"invalid {label}: {exc}") from exc
    try:
        effective_library_root = Path(
            library_root or library.references_root()
        ).expanduser().resolve()
        source_clip_dir = library.require_confined_clip_dir(
            robot, clip_id, root=effective_library_root,
        )
        provenance_bytes, clip_bytes, _preview_bytes = (
            library.capture_reference_artifact_snapshot(
                robot, clip_id, root=effective_library_root,
            )
        )
    except FileNotFoundError as exc:
        raise TrackError(f"no such clip in library: {robot}/{clip_id}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise TrackError(
            "cannot capture confined reference artifact before Tier-D "
            f"allocation: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackError(
            "cannot read reference provenance before Tier-D allocation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    provenance_issues = library.validate_provenance(provenance)
    if provenance_issues:
        raise TrackError(
            "invalid reference provenance before Tier-D allocation: "
            + "; ".join(provenance_issues)
        )
    if provenance.get("schema") != library.PROVENANCE_SCHEMA:
        raise TrackError(
            "Tier-D requires migrated provenance schema "
            f"{library.PROVENANCE_SCHEMA}; refusing GPU allocation for "
            f"legacy schema {provenance.get('schema')!r}"
        )
    if (
        provenance.get("robot") != robot
        or provenance.get("clip_id") != clip_id
    ):
        raise TrackError(
            "reference provenance identity does not match requested robot/clip"
        )
    actual_clip_sha = library.content_sha256(clip_bytes)
    if provenance.get("content_sha256") != actual_clip_sha:
        raise TrackError(
            "provenance.content_sha256 does not match exact clip.npz bytes; "
            "repair/re-ingest before Tier-D allocation"
        )
    try:
        with tempfile.TemporaryDirectory(prefix=".tier-d-reference-") as name:
            snapshot_path = Path(name) / library.CLIP_FILENAME
            snapshot_path.write_bytes(clip_bytes)
            clip = load_clip(snapshot_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise TrackError(
            "cannot validate exact captured clip.npz before Tier-D allocation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if clip.get("root_frame") not in {"absolute", "origin_relative"}:
        raise TrackError(
            "Tier-D requires an explicit persisted root_frame before project "
            "or GPU allocation; materialize a new immutable clip with "
            "absolute or origin_relative semantics"
        )

    declaration_evidence, declaration_issues = (
        library.root_frame_declaration_evidence_from_provenance(provenance)
    )
    if declaration_issues:
        raise TrackError(
            "Tier-D root-frame declaration lacks structured evidence: "
            + "; ".join(declaration_issues)
        )
    root_frame_inheritance, inheritance_issues = (
        library.root_frame_inheritance_from_provenance(
            provenance,
            root=effective_library_root,
            expected_root_frame=clip.get("root_frame"),
        )
    )
    if inheritance_issues:
        raise TrackError(
            "Tier-D root-frame inheritance is invalid: "
            + "; ".join(inheritance_issues)
        )

    if project_dir is None:
        project_dir = (
            effective_library_root.parent
            / "tierD_work"
            / f"{robot}-{clip_id}-{uuid.uuid4().hex}"
        )

    _log(f"[track] CPU-preflighting throwaway project at {project_dir}")
    preflight = _prepare_tierd_tracking_preflight(
        clip=clip,
        clip_content_sha256=actual_clip_sha,
        clip_id=clip_id,
        robot=robot,
        donor_project=donor_project,
        project_dir=project_dir,
        iterations=iterations,
        steps_per_iteration=steps_per_iteration,
        n_episodes=n_episodes,
        seed=seed,
        resume_checkpoint=resume_checkpoint,
        protected_paths=(effective_library_root, source_clip_dir),
        root_frame_declaration_evidence=declaration_evidence,
        root_frame_inheritance=root_frame_inheritance,
    )
    plan = preflight.plan

    if dry_run:
        prov = library.read_provenance(
            robot, clip_id, root=effective_library_root,
        )
        return TrackResult(
            plan=plan,
            errors=None,
            provenance=prov,
            dry_run=True,
            preflight_receipt=preflight.receipt,
        )

    from sculptor.adapters.base import load_adapter

    _log(f"[track] loading adapter from {plan.config_path}")
    try:
        adapter = load_adapter(plan.config_path)
    except Exception as exc:  # noqa: BLE001 - normalized setup failure
        raise TrackError(
            "cannot instantiate the preflighted Tier-D adapter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _assert_local_tierd_adapter(adapter)
    policy_contract = preflight.policy_contract
    execution_contract = preflight.execution_contract

    observed_environment_artifacts = _adapter_environment_artifacts(adapter)
    if observed_environment_artifacts != execution_contract[
        "environment_artifacts"
    ]:
        raise TrackError(
            "instantiated adapter environment inputs differ from the exact "
            "Tier-D execution receipt"
        )
    consumed_config_sha = _file_sha256(
        plan.config_path, label="generated certification config.toml",
    )
    recorded_config_sha = execution_contract["donor"][
        "certification_config_sha256"
    ]
    if consumed_config_sha != recorded_config_sha:
        raise TrackError(
            "generated certification config changed while the adapter was "
            "being loaded; refusing a stale Tier-D execution receipt"
        )
    requested_reward_sha = preflight.requested_reward_sha256
    requested_num_envs = preflight.requested_num_envs

    import inspect

    train_accepts_init_policy = "init_policy_path" in inspect.signature(
        adapter.train).parameters
    if (
        preflight.retained_initial_checkpoint_path is not None
        and not train_accepts_init_policy
    ):
        raise TrackError(
            "Tier-D adapter cannot load the verified continuation checkpoint"
        )

    train_dir = project_dir / "train"
    _log(
        f"[track] training {plan.iterations} iteration(s) x "
        f"{plan.steps_per_iteration} steps -> {train_dir}")
    ckpt_path = preflight.retained_initial_checkpoint_path
    retained_initial_sha = (
        preflight.initial_checkpoint_receipt["checkpoint_sha256"]
        if preflight.initial_checkpoint_receipt is not None
        else None
    )
    train_receipts: list[dict[str, Any]] = []
    for i in range(plan.iterations):
        _assert_local_tierd_adapter(adapter)
        if _file_sha256(
            plan.reward_path, label="generated Tier-D reward module",
        ) != requested_reward_sha:
            raise TrackError("Tier-D reward bytes changed before training")
        extra: dict[str, Any] = {}
        if ckpt_path is not None and train_accepts_init_policy:
            extra["init_policy_path"] = ckpt_path
        if (
            i == 0
            and retained_initial_sha is not None
            and _file_sha256(
                ckpt_path, label="retained Tier-D continuation checkpoint",
            ) != retained_initial_sha
        ):
            raise TrackError(
                "retained Tier-D continuation changed before training"
            )
        result = adapter.train(
            reward_module_path=plan.reward_path,
            output_dir=train_dir,
            steps=plan.steps_per_iteration,
            seed=seed,
            **extra,
        )
        if (
            i == 0
            and retained_initial_sha is not None
            and _file_sha256(
                preflight.retained_initial_checkpoint_path,
                label="retained Tier-D continuation checkpoint",
            ) != retained_initial_sha
        ):
            raise TrackError(
                "retained Tier-D continuation changed during training"
            )
        ckpt_path = result.checkpoint_path
        observed_checkpoint_sha = _file_sha256(
            ckpt_path, label="Tier-D training checkpoint",
        )
        if _file_sha256(
            plan.reward_path, label="generated Tier-D reward module",
        ) != requested_reward_sha:
            raise TrackError("Tier-D reward bytes changed during training")
        receipt = _read_tierd_train_runtime_receipt(
            train_dir / "metrics.json", iteration=i + 1,
        )
        if receipt["output_checkpoint_sha256"] != observed_checkpoint_sha:
            raise TrackError(
                "Tier-D training checkpoint differs from the runner receipt"
            )
        _verify_checkpoint_policy_contract_sidecar(
            ckpt_path,
            checkpoint_sha256=observed_checkpoint_sha,
            expected_policy_contract=policy_contract,
            expected_policy_contract_sha256=execution_contract["donor"][
                "policy_contract_sha256"
            ],
            expected_sidecar_sha256=receipt[
                "output_policy_contract_sidecar_sha256"
            ],
        )
        train_receipts.append(receipt)
        _log(f"[track] iteration {i + 1}/{plan.iterations} done: {ckpt_path}")

    if ckpt_path is None:  # pragma: no cover - plan validates positive budget
        raise TrackError("Tier-D training produced no checkpoint")
    final_checkpoint_sha = _file_sha256(
        ckpt_path, label="final Tier-D checkpoint",
    )
    rollout_signature = inspect.signature(adapter.rollout).parameters
    required_rollout_parameters = {"seed", "max_episode_steps"}
    missing_rollout_parameters = sorted(
        required_rollout_parameters - set(rollout_signature)
    )
    if missing_rollout_parameters:
        raise TrackError(
            "Tier-D adapter cannot pin exact rollout settings: missing "
            + ", ".join(missing_rollout_parameters)
        )
    rollout_max_steps = max(
        1,
        int(math.ceil(
            float(execution_contract["reference"]["playback_duration_s"])
            / float(execution_contract["execution_boundary"]["timing"][
                "control_dt_s"
            ])
        )) + 1,
    )
    rollout_task_id = str(
        execution_contract["execution_boundary"]["identity"]["task_id"]
    )
    execution_contract = bind_tierd_runtime_artifacts(
        execution_contract,
        requested_reward_module_sha256=requested_reward_sha,
        train_receipts=train_receipts,
        final_checkpoint_sha256=final_checkpoint_sha,
        requested_steps_per_iteration=plan.steps_per_iteration,
        requested_seed=int(seed),
        requested_num_envs=requested_num_envs,
        requested_rollout_seed=int(seed),
        requested_rollout_episodes=plan.n_episodes,
        requested_rollout_max_steps=rollout_max_steps,
        requested_rollout_task_id=rollout_task_id,
        initial_checkpoint=preflight.initial_checkpoint_receipt,
    )

    rollout_extra: dict[str, Any] = {
        "seed": int(seed),
        "max_episode_steps": rollout_max_steps,
    }

    rollout_dir = project_dir / "rollout"
    _log(f"[track] rolling out {plan.n_episodes} episode(s) -> {rollout_dir}")
    _assert_local_tierd_adapter(adapter)
    if _file_sha256(
        plan.reward_path, label="generated Tier-D reward module",
    ) != requested_reward_sha:
        raise TrackError("Tier-D reward bytes changed before rollout")
    if _file_sha256(
        ckpt_path, label="final Tier-D checkpoint",
    ) != final_checkpoint_sha:
        raise TrackError("Tier-D checkpoint bytes changed before rollout")
    rollout_result = adapter.rollout(
        checkpoint_path=ckpt_path,
        output_dir=rollout_dir,
        n_episodes=plan.n_episodes,
        reward_module_path=plan.reward_path,
        **rollout_extra,
    )
    if _file_sha256(
        plan.reward_path, label="generated Tier-D reward module",
    ) != requested_reward_sha:
        raise TrackError("Tier-D reward bytes changed during rollout")
    if _file_sha256(
        ckpt_path, label="final Tier-D checkpoint",
    ) != final_checkpoint_sha:
        raise TrackError("Tier-D checkpoint bytes changed during rollout")

    errors = _score_tierd_rollout_artifact(
        rollout_result.trajectory_path,
        clip=clip,
        execution_contract=execution_contract,
        lane=0,
    )
    _log(f"[track] errors: {errors.to_dict()}")

    rollout_dest = None
    if errors.feasible:
        try:
            clip_d = library.require_confined_clip_dir(
                robot, clip_id, root=effective_library_root,
            )
        except (OSError, ValueError) as exc:
            raise TrackError(
                "reference publication path changed before Tier-D rollout "
                f"retention: {type(exc).__name__}: {exc}"
            ) from exc
        rollout_dest = _materialize_tierd_rollout_artifact(
            rollout_result.trajectory_path,
            clip_dir=clip_d,
            clip=clip,
            execution_contract=execution_contract,
            lane=0,
            expected_errors=errors,
            library_root=effective_library_root,
        )
        _log(f"[track] tracking gates passed; rollout retained at {rollout_dest}")
    else:
        _log("[track] exact-schedule tracking gates failed (tier stays K)")

    prov = _publish_and_verify_tierd_verdict(
        robot=robot, clip_id=clip_id, errors=errors,
        iterations=plan.iterations, rollout_path=rollout_dest,
        execution_contract=execution_contract,
        root=effective_library_root)

    return TrackResult(
        plan=plan,
        errors=errors,
        provenance=prov,
        dry_run=False,
        preflight_receipt=preflight.receipt,
    )


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
    common_joint_names: tuple[str, ...]
    static_baseline_err_rad: float
    static_baseline_ratio: float
    rollout_path: Path
    rollout_sha256: str
    clip_content_sha256: str
    certification_scope: dict[str, Any] = field(repr=False)
    execution_contract: dict[str, Any] = field(repr=False)
    execution_contract_sha256: str = ""
    execution_boundary_sha256: str = ""
    # Canonical digest of the exact robot/clip identity plus the complete
    # tierD certificate block that passed verification.  This is a receipt for
    # immutable admission state, not a substitute for re-reading the clip and
    # rollout bytes at every execution boundary.
    certificate_sha256: str = ""


class TierDAdmissionError(ValueError):
    """Raised when a reference cannot satisfy an exact Tier-D admission."""


def _tierd_certificate_sha256(
    robot: str,
    clip_id: str,
    tier_d: dict[str, Any],
) -> str:
    """Stable content identity for one verified Tier-D certificate block."""
    payload = {
        "schema": TIER_D_CERTIFICATE_SCHEMA,
        "robot": robot,
        "clip_id": clip_id,
        "tierD": tier_d,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        effective_root = Path(root or library.references_root()).expanduser().resolve()
        confined_clip_dir = library.require_confined_clip_dir(
            robot, clip_id, root=effective_root,
        )
        prov = library.read_provenance(
            robot, clip_id, root=effective_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return None, f"{prefix}: cannot read provenance: {type(e).__name__}: {e}"

    provenance_issues = _exact_provenance_identity_issues(
        prov, robot=robot, clip_id=clip_id,
    )
    if provenance_issues:
        return None, (
            f"{prefix}: invalid or mis-scoped provenance: "
            + "; ".join(provenance_issues)
        )

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
    if not all(np.isfinite(value) for value in (
        mean_joint_err, max_joint_err, root_z_rmse,
    )):
        return None, f"{prefix}: tierD.errors contains non-finite numeric stats"

    # Reject an out-of-tolerance historical/edited record before asking for
    # newer evidence fields, while still requiring every newer field for a
    # successful schema-v3 certificate.
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

    try:
        duration_coverage = float(errors_block["duration_coverage"])
        static_baseline_err = float(errors_block["static_baseline_err_rad"])
        recorded_static_ratio = float(errors_block["static_baseline_ratio"])
    except (KeyError, TypeError, ValueError) as e:
        return None, f"{prefix}: tierD.errors missing/invalid numeric stats: {e}"

    if not all(np.isfinite(value) for value in (
        duration_coverage,
        static_baseline_err,
        recorded_static_ratio,
    )):
        return None, f"{prefix}: tierD.errors contains non-finite numeric stats"
    common_joint_names = errors_block.get("common_joint_names")
    n_common_joints = errors_block.get("n_common_joints")
    if (
        not isinstance(common_joint_names, list)
        or not common_joint_names
        or not all(isinstance(name, str) and name for name in common_joint_names)
        or len(set(common_joint_names)) != len(common_joint_names)
        or not isinstance(n_common_joints, int)
        or isinstance(n_common_joints, bool)
        or n_common_joints != len(common_joint_names)
    ):
        return None, (
            f"{prefix}: Tier-D requires a non-empty, exact common-joint "
            "tracking contract"
        )
    if static_baseline_err < MIN_REFERENCE_MOTION_RAD:
        return None, (
            f"{prefix}: static baseline is vacuous "
            f"({static_baseline_err} < {MIN_REFERENCE_MOTION_RAD}); Tier-D "
            "requires temporal joint-motion evidence"
        )
    recomputed_static_ratio = mean_joint_err / static_baseline_err
    if not math.isclose(
        recorded_static_ratio,
        recomputed_static_ratio,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return None, (
            f"{prefix}: stored static-baseline ratio is stale/tampered "
            f"(recorded {recorded_static_ratio}, recomputed "
            f"{recomputed_static_ratio})"
        )
    recomputed_beats_static = (
        recomputed_static_ratio <= STATIC_BASELINE_RATIO_MAX
    )
    recorded_beats_static = errors_block.get("beats_static_baseline")
    if not isinstance(recorded_beats_static, bool):
        return None, (
            f"{prefix}: tierD.errors has no boolean beats_static_baseline receipt"
        )
    if recorded_beats_static != recomputed_beats_static:
        return None, (
            f"{prefix}: stored beats_static_baseline verdict disagrees with "
            "the recomputed ratio"
        )
    if not recomputed_beats_static:
        return None, (
            f"{prefix}: tracker did not beat the constant-pose baseline "
            f"(ratio={recomputed_static_ratio} > "
            f"{STATIC_BASELINE_RATIO_MAX})"
        )
    if duration_coverage < DURATION_COVERAGE_MIN:
        return None, (
            f"{prefix}: rollout covered only {duration_coverage:.6f} of the "
            f"reference duration; Tier-D requires >= {DURATION_COVERAGE_MIN}"
        )
    if errors_block.get("certification_scope") != TIER_D_CERTIFICATION_SCOPE:
        return None, (
            f"{prefix}: Tier-D certification scope is missing or unsupported"
        )

    if prov.get("tier") != "D":
        return None, f"{prefix}: provenance.tier is {prov.get('tier')!r}, not 'D'"

    execution_contract = tier_d.get("execution_contract")
    execution_issues = validate_tierd_execution_contract(execution_contract)
    if execution_issues:
        return None, (
            f"{prefix}: invalid Tier-D execution evidence: "
            + "; ".join(execution_issues)
        )
    declaration_evidence, declaration_issues = (
        library.root_frame_declaration_evidence_from_provenance(prov)
    )
    if declaration_issues:
        return None, (
            f"{prefix}: invalid root-frame declaration evidence: "
            + "; ".join(declaration_issues)
        )
    root_frame_inheritance, inheritance_issues = (
        library.root_frame_inheritance_from_provenance(
            prov,
            root=effective_root,
            expected_root_frame=execution_contract["reference"].get(
                "root_frame"
            ),
        )
    )
    if inheritance_issues:
        return None, (
            f"{prefix}: invalid root-frame inheritance: "
            + "; ".join(inheritance_issues)
        )
    if execution_contract["reference"].get(
        "root_frame_declaration_evidence"
    ) != declaration_evidence:
        return None, (
            f"{prefix}: Tier-D root-frame declaration evidence is stale"
        )
    if execution_contract["reference"].get(
        "root_frame_inheritance"
    ) != root_frame_inheritance:
        return None, (
            f"{prefix}: Tier-D root-frame inheritance is stale"
        )
    execution_contract_sha = execution_contract["contract_sha256"]
    execution_boundary_sha = execution_contract["execution_boundary_sha256"]
    tracked_at = tier_d.get("tracked_at")
    iterations = tier_d.get("iterations")
    requested_training = execution_contract["runtime_artifacts"].get(
        "requested_training"
    )
    if not isinstance(tracked_at, str) or not tracked_at.strip():
        return None, f"{prefix}: tierD.tracked_at is missing/invalid"
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
        or not isinstance(requested_training, dict)
        or iterations != requested_training.get("iterations")
    ):
        return None, (
            f"{prefix}: tierD.iterations is invalid or differs from the "
            "exact training request"
        )
    if tier_d.get("execution_contract_sha256") != execution_contract_sha:
        return None, (
            f"{prefix}: tierD execution contract sha256 receipt is missing/stale"
        )
    if tier_d.get("execution_boundary_sha256") != execution_boundary_sha:
        return None, (
            f"{prefix}: tierD execution boundary sha256 receipt is missing/stale"
        )
    certified_robot = execution_contract["execution_boundary"]["robot"]
    if certified_robot != robot:
        return None, (
            f"{prefix}: execution evidence robot {certified_robot!r} does not "
            f"match library robot {robot!r}"
        )

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
    if not _is_server_owned_rollout_path(
        resolved_rollout,
        clip_dir=confined_clip_dir,
        sha256=actual_rollout_sha,
    ):
        return None, (
            f"{prefix}: tierD.rollout_path is not the exact server-owned "
            "content-addressed artifact path for this robot/clip"
        )

    recorded_source_sha = tier_d.get("source_content_sha256")
    current_source_sha = prov.get("source_content_sha256")
    if (recorded_source_sha is None) != (current_source_sha is None):
        return None, f"{prefix}: source content hash lineage changed"
    if recorded_source_sha != current_source_sha:
        return None, (
            f"{prefix}: source content hash drift — provenance.source_content_sha256 "
            f"({str(current_source_sha)[:12]}…) does not match the source hash "
            f"recorded at tracking time ({str(recorded_source_sha)[:12]}…); the clip was "
            "likely re-ingested/edited after certification without "
            "re-tracking")

    # Check 6 (§F7): re-derive from the REAL clip.npz bytes on disk —
    # checks above only compared two fields of the same provenance.json,
    # which a hand-edited file could keep mutually consistent (both
    # wrong) without ever touching the clip. This closes the loop to
    # ground truth.
    clip_path = confined_clip_dir / library.CLIP_FILENAME
    try:
        actual_clip_sha = library.content_sha256(clip_path.read_bytes())
    except OSError as e:
        return None, (
            f"{prefix}: cannot read clip.npz to verify content hash: "
            f"{type(e).__name__}: {e}")
    recorded_clip_sha = tier_d.get("clip_content_sha256")
    current_clip_sha = prov.get("content_sha256")
    if not recorded_clip_sha:
        return None, f"{prefix}: tierD block has no exact clip artifact sha256"
    if current_clip_sha != recorded_clip_sha:
        return None, (
            f"{prefix}: provenance clip artifact hash drift — recorded "
            f"{recorded_clip_sha[:12]}…, current "
            f"{str(current_clip_sha)[:12]}…"
        )
    if actual_clip_sha != recorded_clip_sha:
        return None, (
            f"{prefix}: clip.npz on-disk bytes do not match the recorded "
            f"content hash (recorded {recorded_clip_sha[:12]}…, actual "
            f"{actual_clip_sha[:12]}…) — the clip file was modified after "
            "certification without re-tracking")

    # The byte hash above already binds the motion artifact.  Re-read its
    # explicit cadence/interface fields as well so a malformed or stale
    # execution receipt produces an actionable denial rather than a generic
    # digest mismatch downstream.
    try:
        from sculptor.reference import load_clip

        current_clip = load_clip(clip_path)
        current_fps = float(current_clip["fps"])
        current_joints = [str(name) for name in current_clip["joint_names"]]
        current_joint_pos = np.asarray(current_clip["joint_pos"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, (
            f"{prefix}: cannot read clip cadence/interface evidence: "
            f"{type(exc).__name__}: {exc}"
        )
    reference_evidence = execution_contract["reference"]
    if reference_evidence.get("clip_id") != clip_id:
        return None, (
            f"{prefix}: Tier-D execution evidence names a different clip id"
        )
    if common_joint_names != reference_evidence["ordered_joints"]:
        return None, (
            f"{prefix}: tracking errors do not cover the exact full ordered "
            "joint contract"
        )
    if current_joints != reference_evidence["ordered_joints"]:
        return None, (
            f"{prefix}: current clip ordered joints differ from Tier-D evidence"
        )
    current_root_frame = current_clip.get("root_frame")
    if current_root_frame not in {"absolute", "origin_relative"}:
        return None, (
            f"{prefix}: current clip has no explicit persisted root frame; "
            "heuristic frame inference cannot support Tier-D admission"
        )
    if current_root_frame != reference_evidence.get("root_frame"):
        return None, (
            f"{prefix}: current clip root frame differs from Tier-D evidence"
        )
    if current_joint_pos.ndim != 2 or int(current_joint_pos.shape[0]) != int(
        reference_evidence["frame_count"]
    ):
        return None, (
            f"{prefix}: current clip frame count differs from Tier-D evidence"
        )
    if abs(current_fps - float(reference_evidence["fps"])) > 1e-12:
        return None, (
            f"{prefix}: current clip fps/cadence differs from Tier-D evidence"
        )
    try:
        phase_target_count = int(reference_evidence["phase_target_count"])
        (
            target_names,
            target_joint_pos,
            target_joint_vel,
            target_root_z,
            target_gravity,
        ) = _tracking_targets_from_clip(
            current_clip,
            n_phase_targets=phase_target_count,
        )
        expected_target_sha = reference_target_sha256(
            reference_tracking_target_payload(
                joint_names=target_names,
                target_joint_pos=target_joint_pos,
                target_joint_vel=target_joint_vel,
                target_root_z=target_root_z,
                target_gravity=target_gravity,
                root_frame=current_root_frame,
            )
        )
        stored_clock = validate_reference_clock(
            reference_evidence.get("clock_contract") or {}
        )
        fresh_clock = build_reference_clock(
            clip_id=clip_id,
            robot=robot,
            target_sha256=expected_target_sha,
            phase_mode="hold",
            phase_duration_s=reference_playback_duration_s(
                frame_count=int(current_joint_pos.shape[0]),
                fps=current_fps,
            ),
            n_phase_targets=phase_target_count,
        )
    except (KeyError, TypeError, ValueError, TrackError) as exc:
        return None, (
            f"{prefix}: cannot re-derive exact Tier-D tracking target: "
            f"{type(exc).__name__}: {exc}"
        )
    if stored_clock != fresh_clock:
        changed_clock_fields = sorted(
            key for key in set(stored_clock) | set(fresh_clock)
            if stored_clock.get(key) != fresh_clock.get(key)
        )
        return None, (
            f"{prefix}: Tier-D clock differs from the freshly re-derived "
            f"clip schedule (changed fields: {changed_clock_fields})"
        )

    try:
        recomputed_errors = _score_tierd_rollout_artifact(
            rollout_path,
            clip=current_clip,
            execution_contract=execution_contract,
            lane=int(reference_evidence["rollout_lane"]),
        )
    except (KeyError, TypeError, ValueError, TrackError) as exc:
        return None, (
            f"{prefix}: cannot recompute Tier-D evidence from the exact "
            f"rollout: {type(exc).__name__}: {exc}"
        )
    canonical_recomputed = recomputed_errors.to_dict()
    if errors_block != canonical_recomputed:
        changed = sorted({
            *errors_block.keys(), *canonical_recomputed.keys(),
        } - {
            key for key in set(errors_block) & set(canonical_recomputed)
            if errors_block.get(key) == canonical_recomputed.get(key)
        })
        return None, (
            f"{prefix}: stored Tier-D errors are stale/tampered relative to "
            f"the exact rollout (changed fields: {changed})"
        )
    if not recomputed_errors.feasible:
        return None, (
            f"{prefix}: exact rollout no longer satisfies Tier-D feasibility"
        )

    cert = TierDCertificate(
        robot=robot,
        clip_id=clip_id,
        tracked_at=tracked_at,
        iterations=iterations,
        mean_joint_err_rad=recomputed_errors.mean_joint_err_rad,
        max_joint_err_rad=recomputed_errors.max_joint_err_rad,
        root_z_rmse_m=recomputed_errors.root_z_rmse_m,
        common_joint_names=tuple(recomputed_errors.common_joint_names),
        static_baseline_err_rad=recomputed_errors.static_baseline_err_rad,
        static_baseline_ratio=recomputed_errors.static_baseline_ratio,
        rollout_path=rollout_path,
        rollout_sha256=actual_rollout_sha,
        clip_content_sha256=actual_clip_sha,
        certification_scope=json.loads(json.dumps(
            TIER_D_CERTIFICATION_SCOPE, allow_nan=False,
        )),
        execution_contract=execution_contract,
        execution_contract_sha256=execution_contract_sha,
        execution_boundary_sha256=execution_boundary_sha,
        certificate_sha256=_tierd_certificate_sha256(
            robot, clip_id, tier_d,
        ),
    )
    return cert, None


def require_tierd_admission(
    robot: str,
    clip_id: str,
    *,
    expected_clip_sha256: Optional[str] = None,
    expected_certificate_sha256: Optional[str] = None,
    expected_rollout_sha256: Optional[str] = None,
    expected_execution_contract_sha256: Optional[str] = None,
    expected_execution_boundary_sha256: Optional[str] = None,
    root: Optional[Path] = None,
) -> TierDCertificate:
    """Verify Tier D and, when supplied, match immutable admission pins.

    This is the sole raising admission API for launch paths.  Callers may use
    :func:`verify_tierd_certificate` for a read-only availability verdict, but
    mission attachment and every training boundary must come through here so
    stale bytes cannot silently inherit a prior approval.
    """
    certificate, reason = verify_tierd_certificate(robot, clip_id, root=root)
    if certificate is None:
        raise TierDAdmissionError(
            reason or f"{robot}/{clip_id}: no verified Tier-D certificate"
        )

    expected = {
        "clip sha256": expected_clip_sha256,
        "certificate sha256": expected_certificate_sha256,
        "rollout sha256": expected_rollout_sha256,
        "execution contract sha256": expected_execution_contract_sha256,
        "execution boundary sha256": expected_execution_boundary_sha256,
    }
    actual = {
        "clip sha256": certificate.clip_content_sha256,
        "certificate sha256": certificate.certificate_sha256,
        "rollout sha256": certificate.rollout_sha256,
        "execution contract sha256": certificate.execution_contract_sha256,
        "execution boundary sha256": certificate.execution_boundary_sha256,
    }
    for label, pinned in expected.items():
        if pinned is not None and actual[label] != pinned:
            raise TierDAdmissionError(
                f"{robot}/{clip_id}: stale Tier-D admission: pinned {label} "
                f"{pinned}, current {actual[label]}"
            )
    return certificate


def require_tierd_runtime_reference(
    certificate: TierDCertificate,
    reward_source: str,
) -> dict[str, Any]:
    """Require the live reward to execute the schedule that earned Tier D.

    Certification is evidence for one immutable target table and one exact
    per-environment clock.  A runtime reward that downsamples, crops, loops, or
    retimes that table is a new reference and must be materialized/certified
    separately.  Return the validated descriptor for the launch receipt.
    """
    try:
        runtime_clock = reference_clock_from_reward_source(reward_source)
    except (TypeError, ValueError) as exc:
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: live reward has no "
            f"valid reference clock: {exc}"
        ) from exc
    if runtime_clock is None:
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: live reward omitted "
            "the certified reference clock"
        )
    try:
        certified_clock = validate_reference_clock(
            certificate.execution_contract["reference"]["clock_contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: Tier-D certificate "
            f"has no valid clock contract: {exc}"
        ) from exc
    if runtime_clock != certified_clock:
        fields = (
            "reference_clip_id",
            "reference_robot",
            "reference_target_sha256",
            "phase_mode",
            "phase_duration_s",
            "n_phase_targets",
            "clock",
            "term_name",
            "source",
            "shape",
        )
        changed = [
            name for name in fields
            if runtime_clock.get(name) != certified_clock.get(name)
        ]
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: live reward reference "
            "schedule differs from Tier-D evidence in "
            + ", ".join(changed)
        )
    return runtime_clock


def require_tierd_target_compatibility(
    certificate: TierDCertificate,
    target_project: Path,
    *,
    target_robot: str,
    target_policy_contract: Optional[dict[str, Any]] = None,
) -> TierDCertificate:
    """Fail closed unless a target project is inside the certified boundary.

    This helper is intentionally separate from artifact verification so launch
    adapters can build/request their target contract once and pass it here.
    Supplying ``target_policy_contract`` is also the CPU-only test seam; normal
    callers omit it and the canonical project contract is rebuilt from disk.
    """
    if not isinstance(target_robot, str) or not target_robot.strip():
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: target robot identity "
            "is required for Tier-D compatibility admission"
        )
    robot = target_robot.strip()
    if target_policy_contract is None:
        try:
            from sculptor.policy_contract import build_project_policy_contract

            target_policy_contract = build_project_policy_contract(
                Path(target_project),
            )
        except Exception as exc:  # noqa: BLE001 - normalize admission failure
            raise TierDAdmissionError(
                f"{certificate.robot}/{certificate.clip_id}: cannot build "
                f"target execution contract: {type(exc).__name__}: {exc}"
            ) from exc
    reasons = compare_tierd_target_contract(
        certificate.execution_contract,
        target_policy_contract,
        target_robot=robot,
    )
    if reasons:
        raise TierDAdmissionError(
            f"{certificate.robot}/{certificate.clip_id}: target project is "
            "outside the certified Tier-D execution boundary: "
            + "; ".join(reasons)
        )
    return certificate


def require_stage_tierd_admission(
    stage: Any,
    *,
    expected_robot: Optional[str] = None,
    root: Optional[Path] = None,
) -> Optional[TierDCertificate]:
    """Re-admit an attached mission stage against its exact stored pins.

    Unattached stages return ``None``.  An attached legacy/Tier-K/unpinned
    stage fails closed: it must be detached and attached again after earning a
    real Tier-D certificate.  This keeps serialization backward compatible
    without letting legacy display metadata become execution authority.
    """
    clip_id = getattr(stage, "reference_clip_id", None)
    if not clip_id:
        return None
    robot = getattr(stage, "reference_robot", None)
    clip_sha256 = getattr(stage, "reference_clip_sha256", None)
    certificate_sha256 = getattr(
        stage, "reference_certificate_sha256", None,
    )
    execution_contract_sha256 = getattr(
        stage, "reference_execution_contract_sha256", None,
    )
    execution_boundary_sha256 = getattr(
        stage, "reference_execution_boundary_sha256", None,
    )
    missing = [
        label for label, value in (
            ("reference_robot", robot),
            ("reference_clip_sha256", clip_sha256),
            ("reference_certificate_sha256", certificate_sha256),
            ("reference_execution_contract_sha256", execution_contract_sha256),
            ("reference_execution_boundary_sha256", execution_boundary_sha256),
        ) if not value
    ]
    if missing:
        raise TierDAdmissionError(
            f"stage {getattr(stage, 'name', '<unknown>')!r} reference "
            f"{clip_id!r} has no immutable Tier-D admission pin(s): "
            + ", ".join(missing)
        )
    if getattr(stage, "reference_tier", None) != "D":
        raise TierDAdmissionError(
            f"stage {getattr(stage, 'name', '<unknown>')!r} reference "
            f"{robot}/{clip_id} is not recorded as Tier D"
        )
    if expected_robot and str(robot) != expected_robot:
        raise TierDAdmissionError(
            f"stage {getattr(stage, 'name', '<unknown>')!r} reference robot "
            f"pin {robot!r} does not match the active training robot "
            f"{expected_robot!r}"
        )
    return require_tierd_admission(
        str(robot),
        str(clip_id),
        expected_clip_sha256=str(clip_sha256),
        expected_certificate_sha256=str(certificate_sha256),
        expected_execution_contract_sha256=str(execution_contract_sha256),
        expected_execution_boundary_sha256=str(execution_boundary_sha256),
        root=root,
    )


def require_mission_tierd_admissions(
    mission: Any,
    *,
    expected_robot: Optional[str] = None,
    root: Optional[Path] = None,
) -> dict[str, TierDCertificate]:
    """Re-admit every runnable attached reference in a mission."""
    admitted: dict[str, TierDCertificate] = {}
    for stage in getattr(mission, "stages", ()):
        if getattr(stage, "status", "pending") not in ("pending", "training"):
            continue
        certificate = require_stage_tierd_admission(
            stage,
            expected_robot=expected_robot,
            root=root,
        )
        if certificate is not None:
            admitted[str(getattr(stage, "name", ""))] = certificate
    return admitted
