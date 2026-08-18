"""Canonical compatibility contracts for policy transfer.

Task ids are labels, not interface proofs.  This module fingerprints the
ordered tensor/control interface a warm-start actually depends on and compares
every field fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional


CONTRACT_SCHEMA = 2


def contract_fingerprint(contract: dict[str, Any]) -> str:
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _shape_for_observation_term(
    *,
    name: str,
    source: str,
    params: dict[str, Any],
    joint_count: int,
    action_dim: int,
    command_cfg: Any,
    env_cfg: Any,
) -> Optional[list[int]]:
    source = source.lower()
    if "builtin_sensor" in source:
        sensor = str(params.get("sensor_name") or "").lower()
        if "imu" in sensor and any(token in sensor for token in ("lin", "ang", "acc", "gyro")):
            return [3]
    if source in {"projected_gravity", "base_lin_vel", "base_ang_vel"}:
        return [3]
    if source in {"joint_pos_rel", "joint_vel_rel", "joint_pos", "joint_vel"}:
        return [joint_count]
    if source == "last_action":
        return [action_dim]
    if source == "generated_commands":
        command_name = params.get("command_name")
        cfg = command_cfg.get(command_name) if hasattr(command_cfg, "get") else None
        cls = type(cfg).__name__.lower() if cfg is not None else ""
        if "velocitycommand" in cls:
            return [3]
    sensor_shape = _shape_from_sensor_cfg(
        source=source,
        sensor_name=params.get("sensor_name"),
        env_cfg=env_cfg,
    )
    if sensor_shape is not None:
        return sensor_shape
    if any(token in source for token in ("height", "clock", "phase")):
        return [1]
    # A shape hint on a custom term is the generic extension point.
    hinted = params.get("shape") or params.get("observation_shape")
    if isinstance(hinted, (list, tuple)) and hinted and all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in hinted
    ):
        return [int(v) for v in hinted]
    return None


def _shape_from_sensor_cfg(
    *, source: str, sensor_name: Any, env_cfg: Any,
) -> Optional[list[int]]:
    """Infer common sensor-backed observation widths from immutable cfg.

    This reads model element names from ``spec_fn`` but never constructs or
    steps an environment, so compatibility admission remains CPU-only.
    """
    if not isinstance(sensor_name, str) or not sensor_name:
        return None
    scene = getattr(env_cfg, "scene", None)
    sensors = getattr(scene, "sensors", ()) if scene is not None else ()
    sensor = next(
        (item for item in sensors if getattr(item, "name", None) == sensor_name),
        None,
    )
    if sensor is None:
        return None
    if source == "foot_height":
        frames = getattr(sensor, "frame", ())
        frame_count = len(frames) if isinstance(frames, (list, tuple)) else 1
        if str(getattr(sensor, "reduction", "")).lower() == "min":
            return [frame_count]
        pattern = getattr(sensor, "pattern", None)
        sample_count = int(bool(getattr(pattern, "include_center", False)))
        for ring in getattr(pattern, "rings", ()) or ():
            sample_count += int(getattr(ring, "num_samples", 0) or 0)
        return [frame_count * max(sample_count, 1)]
    if source not in {"foot_air_time", "foot_contact", "foot_contact_forces"}:
        return None
    primary = getattr(sensor, "primary", None)
    entity_name = getattr(primary, "entity", None)
    raw_pattern = getattr(primary, "pattern", None)
    patterns = (
        [raw_pattern]
        if isinstance(raw_pattern, str)
        else list(raw_pattern or ())
    )
    entities = getattr(scene, "entities", {}) or {}
    entity = entities.get(entity_name) if hasattr(entities, "get") else None
    spec_fn = getattr(entity, "spec_fn", None)
    if not callable(spec_fn) or not patterns or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        return None
    try:
        # Keep the MjSpec alive while traversing its bound body sequence.
        spec = spec_fn()
        elements = (
            spec.geoms
            if str(getattr(primary, "mode", "")).lower() == "geom"
            else spec.bodies
        )
        names = [
            str(element.name)
            for element in elements
            if getattr(element, "name", None)
        ]
        matches = [
            name for name in names
            if any(re.fullmatch(pattern, name) for pattern in patterns)
        ]
        excludes = tuple(getattr(primary, "exclude", ()) or ())
        matches = [
            name for name in matches
            if not any(re.fullmatch(str(exclude), name) for exclude in excludes)
        ]
    except Exception:
        return None
    if not matches:
        return None
    slots = max(int(getattr(sensor, "num_slots", 1) or 1), 1)
    width = len(matches) * slots
    if source == "foot_contact_forces":
        width *= 3
    return [width]


def build_project_policy_contract(
    project_dir: Path,
    *,
    observed_network: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the current project's complete warm-start target contract.

    Unknown observation shapes raise instead of silently admitting a policy.
    Custom tasks can expose ``params.shape`` on an observation term to extend
    the generic inference without adding a robot-specific branch.
    """
    project_dir = Path(project_dir)
    config = tomllib.loads((project_dir / "config.toml").read_text(encoding="utf-8"))
    adapter = config.get("adapter") or {}
    adapter_class = str(adapter.get("class") or "")
    adapter_cfg = adapter.get("config") or {}
    task_id = str(adapter_cfg.get("task_id") or adapter_cfg.get("env_id") or "")
    if not adapter_class or not task_id:
        raise ValueError("project adapter class/task_id is incomplete")

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from sculptor.eval.robot_manifest import robot_joint_names

    joints = robot_joint_names(task_id)
    if not joints:
        raise ValueError(f"no ordered joint contract for task {task_id!r}")
    joints = [str(name) for name in joints]
    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    actions = getattr(env_cfg, "actions", None) or {}
    action_names = list(actions.keys()) if hasattr(actions, "keys") else []
    if len(action_names) != 1:
        raise ValueError(
            f"task {task_id!r} must expose one ordered joint action term"
        )
    action_dim = len(joints)

    observations = getattr(env_cfg, "observations", None) or {}
    actor_group = (
        observations.get("actor") or observations.get("policy")
        if hasattr(observations, "get") else None
    )
    critic_group = (
        observations.get("critic") or observations.get("value") or actor_group
        if hasattr(observations, "get") else actor_group
    )
    commands = getattr(env_cfg, "commands", None) or {}

    def observation_contract(group: Any, label: str) -> tuple[list[dict[str, Any]], int]:
        terms = getattr(group, "terms", None) if group else None
        if not isinstance(terms, dict) or not terms:
            raise ValueError(
                f"task {task_id!r} has no ordered {label} observation terms"
            )
        rows: list[dict[str, Any]] = []
        for name, term in terms.items():
            func = getattr(term, "func", None)
            source = getattr(func, "__name__", None) if func else None
            params = getattr(term, "params", None) or {}
            shape = _shape_for_observation_term(
                name=str(name),
                source=str(source or name),
                params=dict(params),
                joint_count=len(joints),
                action_dim=action_dim,
                command_cfg=commands,
                env_cfg=env_cfg,
            )
            if shape is None:
                raise ValueError(
                    f"cannot determine {label} observation shape for term "
                    f"{name!r} ({source!r}); warm-start admission is blocked"
                )
            history = int(getattr(term, "history_length", 0) or 0)
            if history > 0:
                shape[0] *= history
            rows.append({"name": str(name), "source": source, "shape": shape})
        return rows, sum(math.prod(row["shape"]) for row in rows)

    obs_rows, obs_dim = observation_contract(actor_group, "actor")
    critic_obs_rows, critic_obs_dim = observation_contract(
        critic_group, "critic",
    )

    actor_cfg = getattr(rl_cfg, "actor", None)
    critic_cfg = getattr(rl_cfg, "critic", None)
    if actor_cfg is None or critic_cfg is None:
        raise ValueError(f"task {task_id!r} runner has no actor/critic model cfg")
    network = observed_network or {}
    observed_obs = network.get("obs_dim")
    observed_action = network.get("action_dim")
    if observed_obs is not None and int(observed_obs) != obs_dim:
        raise ValueError(
            f"checkpoint obs_dim {observed_obs} != task contract {obs_dim}"
        )
    if observed_action is not None and int(observed_action) != action_dim:
        raise ValueError(
            f"checkpoint action_dim {observed_action} != task contract {action_dim}"
        )

    sim = getattr(env_cfg, "sim", None)
    mujoco = getattr(sim, "mujoco", None) if sim else None
    timestep = float(getattr(mujoco, "timestep", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 0) or 0)
    if timestep <= 0 or decimation <= 0:
        raise ValueError(f"task {task_id!r} has no control timing contract")

    actor_hidden = network.get("hidden_dims") or getattr(actor_cfg, "hidden_dims", ())
    actor_activation = network.get("activation") or getattr(actor_cfg, "activation", None)
    recurrent = {
        "type": getattr(actor_cfg, "rnn_type", None),
        "hidden_dim": int(getattr(actor_cfg, "rnn_hidden_dim", 0) or 0),
        "num_layers": int(getattr(actor_cfg, "rnn_num_layers", 0) or 0),
    }
    actor_normalizer = bool(getattr(actor_cfg, "obs_normalization", False))
    critic_normalizer = bool(getattr(critic_cfg, "obs_normalization", False))
    return {
        "schema": CONTRACT_SCHEMA,
        "identity": {"adapter_class": adapter_class, "task_id": task_id},
        "joints": {"ordered_names": joints},
        "observations": {
            "ordered_terms": obs_rows,
            "shape": [obs_dim],
            "critic_ordered_terms": critic_obs_rows,
            "critic_shape": [critic_obs_dim],
        },
        "actions": {
            "ordered_names": joints,
            "term_names": action_names,
            "shape": [action_dim],
        },
        "policy": {
            "actor": {
                "class_name": str(getattr(actor_cfg, "class_name", "")),
                "hidden_dims": [int(v) for v in actor_hidden],
                "activation": str(actor_activation),
                "recurrent": recurrent,
            },
            "critic": {
                "class_name": str(getattr(critic_cfg, "class_name", "")),
                "hidden_dims": [int(v) for v in getattr(critic_cfg, "hidden_dims", ())],
                "activation": str(getattr(critic_cfg, "activation", "")),
                "recurrent": {
                    "type": getattr(critic_cfg, "rnn_type", None),
                    "hidden_dim": int(getattr(critic_cfg, "rnn_hidden_dim", 0) or 0),
                    "num_layers": int(getattr(critic_cfg, "rnn_num_layers", 0) or 0),
                },
            },
            "normalizer": {
                # ``present`` stays as a backwards-readable actor alias.
                "present": actor_normalizer,
                "actor_present": actor_normalizer,
                "critic_present": critic_normalizer,
                "actor_shape": [obs_dim] if actor_normalizer else None,
                "critic_shape": (
                    [critic_obs_dim] if critic_normalizer else None
                ),
            },
        },
        "timing": {
            "sim_timestep_s": timestep,
            "decimation": decimation,
            "control_dt_s": round(timestep * decimation, 9),
        },
        "versions": {
            "torch": _major_minor(_package_version("torch")),
            "mjlab": _package_version("mjlab"),
            "rsl_rl": _package_version("rsl-rl-lib"),
            "adapter": _package_version("reward-sculptor"),
        },
    }


def _major_minor(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    match = re.match(r"^(\d+\.\d+)", value)
    return match.group(1) if match else value


def compare_policy_contracts(
    source: Optional[dict[str, Any]], target: Optional[dict[str, Any]],
) -> list[str]:
    if not isinstance(source, dict):
        return ["source bundle is missing compatibility_contract"]
    if not isinstance(target, dict):
        return ["project target compatibility contract is unavailable"]
    reasons: list[str] = []
    required_paths = (
        ("identity", "adapter_class"), ("identity", "task_id"),
        ("joints", "ordered_names"),
        ("observations", "ordered_terms"), ("observations", "shape"),
        ("observations", "critic_ordered_terms"),
        ("observations", "critic_shape"),
        ("actions", "ordered_names"), ("actions", "term_names"),
        ("actions", "shape"),
        ("policy", "actor"), ("policy", "critic"),
        ("policy", "normalizer"),
        ("timing", "sim_timestep_s"), ("timing", "decimation"),
        ("timing", "control_dt_s"),
        ("versions", "torch"), ("versions", "mjlab"),
        ("versions", "rsl_rl"), ("versions", "adapter"),
    )
    for path in required_paths:
        left: Any = source
        right: Any = target
        for key in path:
            left = left.get(key) if isinstance(left, dict) else None
            right = right.get(key) if isinstance(right, dict) else None
        label = ".".join(path)
        if left is None or right is None:
            reasons.append(f"{label} is missing/unknown")
        elif left != right:
            reasons.append(f"{label} differs")
    if reasons:
        return reasons
    return [] if source == target else ["compatibility contract differs"]


__all__ = [
    "CONTRACT_SCHEMA",
    "build_project_policy_contract",
    "compare_policy_contracts",
    "contract_fingerprint",
]
