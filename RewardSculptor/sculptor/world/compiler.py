"""Deterministic WorldSpec/TaskSpec compilation for mjlab.

The compiler translates only typed semantic data.  Robot-specific geometry
names and limits come from capability descriptors; no task or robot ID is
inspected with substring conditionals.
"""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import hashlib
import json
import io
import math
import re
import struct
import tempfile
from unittest import mock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from sculptor.world.artifacts import (
    WorldArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
)
from sculptor.world.capabilities import (
    RobotCapability,
    SimulatorCapability,
    build_robot_entity_cfg,
    resolve_robot_capability,
    simulator_capability,
)
from sculptor.world.channels import ChannelCatalog, compile_channel_catalog
from sculptor.world.task_spec import validate_task_spec
from sculptor.world.world_spec import validate_world_spec


class WorldCompileError(ValueError):
    """A validated declarative world could not be compiled."""


@dataclass(frozen=True)
class ResolvedPrimitive:
    """One deterministic fixed primitive produced by the course grammar."""

    primitive_id: str
    source_id: str
    shape: str
    position_m: tuple[float, float, float]
    size_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ResolvedEvaluation:
    """Canonical manifest consumed by rollout instead of mutable authored data."""

    manifest_version: int
    world_hash: str
    task_hash: str
    compiler_hash: str
    robot_capability_hash: str
    robot_asset_hash: str
    simulator_capability_hash: str
    dependency_versions: Mapping[str, str]
    runtime_task_id: str | None
    eval_seed: int
    terrain: Mapping[str, Any]
    course: tuple[ResolvedPrimitive, ...]
    objects: Mapping[str, Any]
    zones: Mapping[str, Any]
    task_shared: Mapping[str, Any]
    channel_catalog_hash: str
    compiled_model_hash: str
    materialized_assets: Mapping[str, Any]
    admission: Mapping[str, Any]
    manifest_hash: str

    @classmethod
    def build(cls, **values: Any) -> "ResolvedEvaluation":
        payload = {
            "manifest_version": 1,
            **values,
        }
        payload["course"] = [
            item.to_dict() if isinstance(item, ResolvedPrimitive) else item
            for item in payload.get("course", ())
        ]
        manifest_hash = sha256_bytes(canonical_json_bytes(payload))
        # Normalize course items to ResolvedPrimitive on EVERY construction
        # path. `with_admission` round-trips through to_dict (course →
        # plain dicts) and previously left them as dicts, so the NEXT
        # to_dict crashed — but only for course-bearing worlds, which no
        # admission test covered until the live parkour authoring hit it.
        values["course"] = tuple(
            item if isinstance(item, ResolvedPrimitive)
            else ResolvedPrimitive(**{
                key: (tuple(value) if isinstance(value, list) else value)
                for key, value in dict(item).items()
            })
            for item in values.get("course", ())
        )
        return cls(manifest_version=1, manifest_hash=manifest_hash, **values)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedEvaluation":
        raw = dict(value)
        supplied_hash = str(raw.pop("manifest_hash", ""))
        raw.pop("manifest_version", None)
        raw["course"] = tuple(
            ResolvedPrimitive(**item) for item in raw.get("course", ()))
        manifest = cls.build(**raw)
        if supplied_hash and supplied_hash != manifest.manifest_hash:
            raise WorldCompileError(
                "resolved evaluation manifest hash mismatch: "
                f"{supplied_hash} != {manifest.manifest_hash}")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["course"] = [item.to_dict() for item in self.course]
        return data

    def with_admission(self, admission: Mapping[str, Any]) -> "ResolvedEvaluation":
        data = self.to_dict()
        data.pop("manifest_version", None)
        data.pop("manifest_hash", None)
        data["admission"] = copy.deepcopy(admission)
        return type(self).build(**data)


@dataclass(frozen=True)
class RuntimeContactBinding:
    sensor_name: str
    group: str
    selectors: tuple[str, str]
    resolved: tuple[Mapping[str, Any], Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TaskRuntimePlan:
    """Simulator-independent task bindings plus concrete sensor configs."""

    contacts: tuple[RuntimeContactBinding, ...]
    observation_bindings: Mapping[str, Any]
    reset_bindings: tuple[Mapping[str, Any], ...]
    goal_binding: Mapping[str, Any]
    termination_bindings: Mapping[str, Any]
    sensor_cfgs: tuple[Any, ...] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contacts": [item.to_dict() for item in self.contacts],
            "observation_bindings": copy.deepcopy(self.observation_bindings),
            "reset_bindings": copy.deepcopy(list(self.reset_bindings)),
            "goal_binding": copy.deepcopy(self.goal_binding),
            "termination_bindings": copy.deepcopy(self.termination_bindings),
            "sensor_names": [sensor.name for sensor in self.sensor_cfgs],
        }


@dataclass
class CompiledWorld:
    """In-memory train scene plus the immutable evaluation contract."""

    scene_cfg: Any
    resolved_eval: ResolvedEvaluation
    channel_catalog: ChannelCatalog
    train_variations: tuple[Mapping[str, Any], ...]
    robot: RobotCapability
    simulator: SimulatorCapability
    model_inventory: Mapping[str, int]
    task_runtime: TaskRuntimePlan
    _scene: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class ResolvedWorldBundle:
    """Selection metadata returned to the runner after one verified load."""

    tuple_hash: str
    evaluation_lineage: str
    manifest: ResolvedEvaluation
    channel_catalog: ChannelCatalog
    train: bool
    refs: Mapping[str, Mapping[str, Any]]
    runtime_robot_asset_hash: str | None = None
    runtime_adjustments: tuple[str, ...] = ()


def _root_body_relative(env: Any, target_pos_w: Any) -> Any:
    """Express a world-space target position in the robot root frame.

    Authored observations must be useful across replicated environments and
    arbitrary embodiments.  MuJoCo scene state is world-space (including each
    environment origin), while authored geometry is local to one environment.
    The caller resolves that translation; this helper removes robot root
    translation and orientation without assuming a robot/task name.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot = env.scene["robot"]
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w
    return quat_apply_inverse(root_quat_w, target_pos_w - root_pos_w)


def authored_region_relative_observation(
    env: Any, *, center_m: tuple[float, float, float],
) -> Any:
    """Body-frame vector from the robot root to an authored region center."""
    import torch

    robot_pos = env.scene["robot"].data.root_link_pos_w
    raw_origins = getattr(env.scene, "env_origins", None)
    origins = (
        torch.zeros_like(robot_pos)
        if raw_origins is None
        else raw_origins.to(device=robot_pos.device, dtype=robot_pos.dtype)
    )
    if tuple(origins.shape) != tuple(robot_pos.shape):
        raise WorldCompileError(
            "scene env_origins shape does not match robot root positions")
    center = torch.as_tensor(
        center_m, device=robot_pos.device, dtype=robot_pos.dtype)
    return _root_body_relative(env, origins + center)


def authored_object_relative_observation(
    env: Any, *, object_name: str,
) -> Any:
    """Body-frame vector from the robot root to an authored object root."""
    target = env.scene[object_name].data.root_link_pos_w
    if target.ndim == 3 and target.shape[1] == 1:
        target = target[:, 0]
    return _root_body_relative(env, target)


def authored_end_effector_relative_observation(
    env: Any, *, role_kind: str, role_names: tuple[str, ...],
) -> Any:
    """Body-frame vector to a semantic end-effector site/body centroid."""
    robot = env.scene["robot"]
    if role_kind == "site":
        available = tuple(robot.site_names)
        positions = robot.data.site_pos_w
    elif role_kind == "body":
        available = tuple(robot.body_names)
        positions = robot.data.body_link_pos_w
    else:
        raise WorldCompileError(
            f"unsupported end-effector semantic role kind {role_kind!r}")
    missing = [name for name in role_names if name not in available]
    if missing:
        raise WorldCompileError(
            f"semantic end-effector names are absent at runtime: {missing}")
    indices = [available.index(name) for name in role_names]
    target = positions[:, indices].mean(dim=1)
    return _root_body_relative(env, target)


def _install_task_observations(
    env_cfg: Any,
    runtime: TaskRuntimePlan,
    *,
    zones: Mapping[str, Any],
    robot: RobotCapability,
) -> None:
    """Install validated authored observations into actor and critic groups.

    ``compile_task_runtime`` has always preserved these bindings in the
    manifest, but the overlay previously installed only their sensors.  The
    policy therefore could neither see a waypoint/region nor locate an object,
    even though rewards and rollout metrics could.  Add only fixed-shape,
    embodiment-neutral terms and use semantic capability roles for end
    effectors.  Train and evaluation both call this same chokepoint.
    """
    observations = getattr(env_cfg, "observations", None)
    if not isinstance(observations, dict):
        return

    from mjlab.managers.observation_manager import ObservationTermCfg

    bindings = runtime.observation_bindings
    terms: dict[str, Any] = {}
    # Task-space vectors are clipped before a modest scale so ordinary
    # metre-scale courses enter the policy near O(1) without losing direction.
    vector_kwargs = {"clip": (-20.0, 20.0), "scale": 0.1}

    for name in bindings.get("region_relative", ()):
        zone = zones[str(name)]
        raw_center = list(zone["center_m"])
        center = tuple(float(v) for v in (
            raw_center if len(raw_center) == 3 else [*raw_center, 0.0]
        ))
        terms[f"authored_region__{name}"] = ObservationTermCfg(
            func=authored_region_relative_observation,
            params={"center_m": center},
            **vector_kwargs,
        )

    for name in bindings.get("object_relative", ()):
        terms[f"authored_object__{name}"] = ObservationTermCfg(
            func=authored_object_relative_observation,
            params={"object_name": str(name)},
            **vector_kwargs,
        )

    for role in bindings.get("end_effector_relative", ()):
        role_kind, role_names = robot.resolve_semantic_role(str(role))
        terms[f"authored_end_effector__{role}"] = ObservationTermCfg(
            func=authored_end_effector_relative_observation,
            params={
                "role_kind": role_kind,
                "role_names": tuple(role_names),
            },
            clip=(-5.0, 5.0),
        )

    height = bindings.get("height_scan", {})
    if isinstance(height, Mapping) and height.get("enabled"):
        from mjlab.envs import mdp as envs_mdp

        terms["authored_height_scan"] = ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": str(height["sensor"])},
            clip=(-5.0, 5.0),
            scale=0.2,
        )

    if not terms:
        return
    for group_name in ("actor", "critic", "policy"):
        group = observations.get(group_name)
        group_terms = getattr(group, "terms", None)
        if isinstance(group_terms, dict):
            group_terms.update(copy.deepcopy(terms))


def _materialized_cfg_types() -> tuple[type[Any], type[Any]]:
    """Create lazy terrain adapter types without importing mjlab at CLI import."""
    import mujoco
    import torch
    from mjlab.terrains import TerrainEntity, TerrainEntityCfg

    @dataclass
    class MaterializedTerrainEntityCfg(TerrainEntityCfg):
        # Frozen XML already contains these assets; inherited plane defaults
        # would add duplicate names during Entity initialization.
        textures: tuple[Any, ...] = field(default_factory=tuple)
        materials: tuple[Any, ...] = field(default_factory=tuple)
        lights: tuple[Any, ...] = field(default_factory=tuple)
        terrain_xml_path: str = ""
        terrain_origins_m: tuple[Any, ...] = ()
        frozen_flat_patches: Mapping[str, Any] = field(default_factory=dict)
        frozen_flat_patch_radii_m: Mapping[str, float] = field(
            default_factory=dict)

    class MaterializedTerrainEntity(TerrainEntity):
        """TerrainEntity backed by persisted XML/binary assets, not a seed."""

        cfg: MaterializedTerrainEntityCfg

        def _build_spec(self) -> None:
            path = Path(self.cfg.terrain_xml_path)
            if not path.is_file():
                raise WorldCompileError(
                    f"materialized terrain XML is missing: {path}")
            self._spec = mujoco.MjSpec.from_file(str(path))
            origins = np.asarray(self.cfg.terrain_origins_m, dtype=np.float32)
            if origins.size:
                origins = origins.reshape((*origins.shape[:-1], 3))
                self.terrain_origins = torch.as_tensor(
                    origins, device=self._device, dtype=torch.float32)
                rows, cols = self.terrain_origins.shape[:2]
                ids = torch.arange(
                    self.cfg.num_envs, device=self._device, dtype=torch.long)
                self.terrain_types = ids.remainder(cols)
                self.terrain_levels = torch.div(
                    ids, cols, rounding_mode="floor").remainder(rows)
                self.max_terrain_level = rows
                self.env_origins = self.terrain_origins[
                    self.terrain_levels, self.terrain_types]
            else:
                self.terrain_origins = None
                self._configure_env_origins()
            self._flat_patches = {
                name: torch.as_tensor(
                    values, device=self._device, dtype=torch.float32)
                for name, values in self.cfg.frozen_flat_patches.items()
            }
            self._flat_patch_radii = dict(
                self.cfg.frozen_flat_patch_radii_m)

        def update_env_origins(self, env_ids: Any, move_up: Any,
                               move_down: Any) -> None:
            # Evaluation origin allocation is part of the frozen manifest.
            return None

        def randomize_env_origins(self, env_ids: Any) -> None:
            return None

    return MaterializedTerrainEntityCfg, MaterializedTerrainEntity


def _waypoint_velocity_command_types() -> tuple[type[Any], type[Any]]:
    """Create a goal-conditioned velocity command without eager mjlab imports.

    Authored waypoint tasks previously replaced the base task's command with a
    constant +X request.  That is adequate for a straight staircase, but it is
    actively contradictory for slaloms and any route that turns: the base
    locomotion reward teaches the policy to ignore the next authored target.
    This command retargets the existing velocity-policy interface onto the
    current waypoint and emits zero after sequence completion.  It therefore
    works for every embodiment exposing a velocity command, while tasks with a
    different command surface remain untouched.
    """
    import torch
    from mjlab.tasks.velocity.mdp.velocity_command import (
        UniformVelocityCommand,
        UniformVelocityCommandCfg,
    )

    class WaypointVelocityCommand(UniformVelocityCommand):
        cfg: Any

        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            self._waypoints = torch.as_tensor(
                cfg.waypoints_m, device=self.device, dtype=torch.float32)
            self._waypoint_index = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long)
            raw_starts = getattr(
                env, "_sculptor_waypoint_start_index", None)
            if raw_starts is not None:
                self._waypoint_index.copy_(
                    raw_starts.to(device=self.device, dtype=torch.long).clamp(
                        min=0, max=self._waypoints.shape[0] - 1))

        def _resample_command(self, env_ids: Any) -> None:
            # The configured timer exceeds the episode horizon, so this path is
            # the per-episode reset.  Route RSI publishes one authoritative
            # start index on the environment; evaluation has no such event and
            # therefore always starts at zero.
            raw_starts = getattr(
                self._env, "_sculptor_waypoint_start_index", None)
            if raw_starts is None:
                self._waypoint_index[env_ids] = 0
            else:
                self._waypoint_index[env_ids] = raw_starts[env_ids].to(
                    device=self.device, dtype=torch.long).clamp(
                        min=0, max=self._waypoints.shape[0] - 1)
            self.vel_command_b[env_ids] = 0.0
            self.vel_command_w[env_ids] = 0.0
            self.is_heading_env[env_ids] = False
            self.is_standing_env[env_ids] = False
            self.is_world_env[env_ids] = False
            self.is_forward_env[env_ids] = False

        def _update_command(self) -> None:
            root_pos = self.robot.data.root_link_pos_w
            raw_origins = getattr(self._env.scene, "env_origins", None)
            origins = (
                torch.zeros_like(root_pos)
                if raw_origins is None
                else raw_origins.to(device=root_pos.device, dtype=root_pos.dtype)
            )
            if tuple(origins.shape) != tuple(root_pos.shape):
                raise WorldCompileError(
                    "scene env_origins shape does not match robot root positions")
            local_xy = (root_pos - origins)[:, :2]
            last = self._waypoints.shape[0] - 1
            target = self._waypoints[
                torch.clamp(self._waypoint_index, max=last), :2]
            distance = torch.linalg.norm(target - local_xy, dim=-1)
            reached = (
                (self._waypoint_index < self._waypoints.shape[0])
                & (distance <= float(self.cfg.tolerance_m))
            )
            self._waypoint_index = torch.clamp(
                self._waypoint_index + reached.long(),
                max=self._waypoints.shape[0],
            )
            complete = self._waypoint_index >= self._waypoints.shape[0]

            target = self._waypoints[
                torch.clamp(self._waypoint_index, max=last), :2]
            delta_w = target - local_xy
            distance = torch.linalg.norm(delta_w, dim=-1)
            direction_w = delta_w / torch.clamp(distance[:, None], min=1e-6)
            # Intermediate gates need a non-zero crossing speed, but the
            # terminal target is qualitatively different: the prompted dwell
            # begins immediately after entry.  Start braking over a longer
            # terminal approach so the command does not jump from ~0.4 m/s to
            # zero at the finish tolerance boundary.
            terminal_target = self._waypoint_index == last
            normal_scale = torch.clamp(
                distance / float(self.cfg.slow_radius_m),
                min=float(self.cfg.intermediate_min_speed_scale),
                max=1.0,
            )
            terminal_scale = torch.clamp(
                distance / float(self.cfg.terminal_slow_radius_m),
                min=float(self.cfg.terminal_min_speed_scale), max=1.0)
            speed = float(self.cfg.cruise_speed_mps) * torch.where(
                terminal_target, terminal_scale, normal_scale)
            speed = torch.where(complete, torch.zeros_like(speed), speed)
            velocity_w = direction_w * speed[:, None]

            # The base policy consumes body-frame velocity commands.  Rotate
            # the authored world-space target direction on every step, then
            # request a matching yaw rate so the body visibly follows turns
            # instead of merely crab-walking through the route.
            heading = self.robot.data.heading_w
            cos_h = torch.cos(heading)
            sin_h = torch.sin(heading)
            vx_w = velocity_w[:, 0]
            vy_w = velocity_w[:, 1]
            vx_b = cos_h * vx_w + sin_h * vy_w
            vy_b = -sin_h * vx_w + cos_h * vy_w
            bearing_b = torch.atan2(vy_b, vx_b)

            self.vel_command_b[:, 0] = vx_b
            self.vel_command_b[:, 1] = vy_b
            yaw_rate = torch.clamp(
                float(self.cfg.turn_gain) * bearing_b,
                min=-float(self.cfg.max_yaw_rate),
                max=float(self.cfg.max_yaw_rate),
            )
            # Slow the turn request with the terminal linear approach.  A
            # full-rate yaw command next to the finish made the body circle
            # and shuffle around the target even while its linear command was
            # braking.  Intermediate route turns retain their full authority.
            yaw_scale = torch.where(
                terminal_target, terminal_scale, torch.ones_like(speed))
            self.vel_command_b[:, 2] = yaw_rate * yaw_scale
            self.vel_command_b[complete] = 0.0
            self.vel_command_w[:, :2] = velocity_w
            self.vel_command_w[:, 2] = self.vel_command_b[:, 2]
            # Preserve the semantic distinction offered by the base command:
            # completed authored routes are standing commands, not merely a
            # coincidental all-zero velocity sample.  Consumers that only use
            # the command tensor remain unchanged; future embodiment-specific
            # standing priors can use this flag without task-name keying.
            self.is_standing_env[:] = complete

    @dataclass(kw_only=True)
    class WaypointVelocityCommandCfg(UniformVelocityCommandCfg):
        waypoints_m: tuple[tuple[float, float, float], ...]
        tolerance_m: float = 0.25
        cruise_speed_mps: float = 0.8
        slow_radius_m: float = 0.75
        intermediate_min_speed_scale: float = 0.35
        terminal_slow_radius_m: float = 2.0
        terminal_min_speed_scale: float = 0.35
        turn_gain: float = 2.0
        max_yaw_rate: float = 1.5

        def build(self, env: Any) -> WaypointVelocityCommand:
            return WaypointVelocityCommand(self, env)

    return WaypointVelocityCommandCfg, WaypointVelocityCommand


_MATERIALIZED_TYPES: tuple[type[Any], type[Any]] | None = None


def materialized_terrain_types() -> tuple[type[Any], type[Any]]:
    global _MATERIALIZED_TYPES
    if _MATERIALIZED_TYPES is None:
        _MATERIALIZED_TYPES = _materialized_cfg_types()
    return _MATERIALIZED_TYPES


def install_materialized_terrain_factory() -> None:
    """Teach mjlab Scene to instantiate the frozen-terrain adapter."""
    import mjlab.scene.scene as scene_module

    cfg_type, entity_type = materialized_terrain_types()
    current = scene_module.TerrainEntity
    if getattr(current, "_sculpt_materialized_factory", False):
        return

    def terrain_factory(cfg: Any, device: str) -> Any:
        if isinstance(cfg, cfg_type):
            return entity_type(cfg, device=device)
        return current(cfg, device=device)

    terrain_factory._sculpt_materialized_factory = True  # type: ignore[attr-defined]
    scene_module.TerrainEntity = terrain_factory


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _range(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def _flat_patch_cfgs(world: Mapping[str, Any]) -> dict[str, Any] | None:
    """Request reset-safe patches whenever an object needs terrain placement."""
    from mjlab.terrains import FlatPatchSamplingCfg

    patches: dict[str, Any] = {}
    for name, obj in sorted(world["shared"].get("objects", {}).items()):
        pose = obj.get("nominal", {}).get("pose", {})
        if not str(pose.get("placement", "")).startswith("zone:"):
            continue
        nominal = obj["nominal"]
        if obj["shape"] == "sphere":
            radius = float(nominal["radius_m"])
        elif obj["shape"] == "box":
            radius = math.hypot(*map(float, nominal["size_m"][:2])) / 2
        else:
            radius = float(nominal.get("radius_m", 0.1))
        patches[f"object_{name}"] = FlatPatchSamplingCfg(
            num_patches=8, patch_radius=max(0.05, radius + 0.03),
            max_height_diff=0.03)
    return patches or None


def _terrain_sub_cfg(
    terrain_type: str, nominal: Mapping[str, Any], *, proportion: float,
    size: tuple[float, float], flat_patches: Mapping[str, Any] | None,
) -> Any:
    """Map the closed terrain vocabulary to installed mjlab dataclasses."""
    import mjlab.terrains as terrains

    common = {
        "proportion": float(proportion), "size": size,
        "flat_patch_sampling": dict(flat_patches) if flat_patches else None,
    }
    # The table is simulator vocabulary, not task or robot dispatch.
    builders: dict[str, tuple[str, Callable[[Mapping[str, Any]], dict[str, Any]]]] = {
        "flat": ("BoxFlatTerrainCfg", lambda n: {}),
        "hf_random_uniform": ("HfRandomUniformTerrainCfg", lambda n: {
            "noise_range": _range(n["noise_range_m"]),
            "noise_step": float(n.get("noise_step_m", 0.005)),
            "downsampled_scale": n.get("downsampled_scale_m"),
        }),
        "hf_pyramid_sloped": ("HfPyramidSlopedTerrainCfg", lambda n: {
            "slope_range": tuple(
                math.tan(math.radians(x)) for x in _range(n["slope_deg"])),
            "platform_width": float(n.get("platform_width_m", 1.0)),
            "inverted": bool(n.get("inverted", False)),
        }),
        "hf_wave": ("HfWaveTerrainCfg", lambda n: {
            "amplitude_range": _range(n["amplitude_m"]),
            "num_waves": int(n.get("num_waves", 1)),
        }),
        "hf_discrete_obstacles": (
            "HfDiscreteObstaclesTerrainCfg", lambda n: {
                "obstacle_width_range": _range(n["obstacle_width_m"]),
                "obstacle_height_range": _range(n["obstacle_height_m"]),
                "num_obstacles": int(n["num_obstacles"]),
                "platform_width": float(n.get("platform_width_m", 1.0)),
                "square_obstacles": bool(n.get("square_obstacles", False)),
            }),
        "hf_perlin_noise": ("HfPerlinNoiseTerrainCfg", lambda n: {
            "height_range": _range(n["height_range_m"]),
            "octaves": int(n.get("octaves", 4)),
            "persistence": float(n.get("persistence", 0.5)),
            "lacunarity": float(n.get("lacunarity", 2.0)),
            "scale": float(n.get("scale", 10.0)),
        }),
        "box_random_spread": ("BoxRandomSpreadTerrainCfg", lambda n: {
            "num_boxes": int(n.get("num_boxes", 60)),
            "box_width_range": _range(n.get("box_width_m", (0.3, 1.0))),
            "box_length_range": _range(n.get("box_length_m", (0.3, 1.0))),
            "box_height_range": _range(n.get("box_height_m", (0.05, 1.0))),
            "box_yaw_range": _range(n.get("box_yaw_deg", (0.0, 360.0))),
            "add_floor": bool(n.get("add_floor", True)),
        }),
        "box_random_grid": ("BoxRandomGridTerrainCfg", lambda n: {
            "grid_width": float(n["grid_width_m"]),
            "grid_height_range": _range(n["grid_height_m"]),
            "holes": bool(n.get("holes", False)),
        }),
        "box_random_stairs": ("BoxRandomStairsTerrainCfg", lambda n: {
            "step_width": float(n.get("step_width_m", 0.8)),
            "step_height_range": _range(n.get("step_height_m", (0.1, 0.3))),
        }),
        "box_pyramid_stairs": (
            "BoxPyramidStairsTerrainCfg", lambda n: {
                "step_height_range": _range(n["step_height_m"]),
                "step_width": float(n["step_width_m"]),
                "holes": bool(n.get("holes", False)),
            }),
        "box_open_stairs": ("BoxOpenStairsTerrainCfg", lambda n: {
            "step_height_range": _range(n.get("step_height_m", (0.1, 0.2))),
            "step_width_range": _range(n.get("step_width_m", (0.4, 0.8))),
            "inverted": bool(n.get("inverted", True)),
        }),
        "box_stepping_stones": (
            "BoxSteppingStonesTerrainCfg", lambda n: {
                "stone_size_range": _range(n.get("stone_size_m", (0.4, 0.8))),
                "stone_distance_range": _range(
                    n.get("stone_distance_m", (0.2, 0.5))),
                "stone_height": float(n.get("stone_height_m", 0.2)),
            }),
        "box_narrow_beams": ("BoxNarrowBeamsTerrainCfg", lambda n: {
            "num_beams": int(n.get("num_beams", 16)),
            "beam_width_range": _range(n.get("beam_width_m", (0.2, 0.4))),
            "beam_height": float(n.get("beam_height_m", 0.2)),
            "spacing": float(n.get("spacing_m", 0.8)),
        }),
        "box_tilted_grid": ("BoxTiltedGridTerrainCfg", lambda n: {
            "grid_width": float(n.get("grid_width_m", 1.0)),
            "tilt_range_deg": float(n.get("tilt_deg", 15.0)),
            "height_range": float(n.get("height_m", 0.1)),
        }),
        "box_nested_rings": ("BoxNestedRingsTerrainCfg", lambda n: {
            "num_rings": int(n.get("num_rings", 5)),
            "ring_width_range": _range(n.get("ring_width_m", (0.3, 0.6))),
            "gap_range": _range(n.get("gap_m", (0.0, 0.2))),
            "height_range": _range(n.get("height_m", (0.1, 0.4))),
        }),
    }
    if terrain_type not in builders:
        raise WorldCompileError(f"unsupported terrain type {terrain_type!r}")
    class_name, convert = builders[terrain_type]
    try:
        cls = getattr(terrains, class_name)
        return cls(**common, **convert(nominal))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise WorldCompileError(
            f"terrain {terrain_type!r} could not be compiled: {exc}") from exc


def compile_terrain_cfg(world: Mapping[str, Any]) -> Any:
    """Compile the shared terrain section into one deterministic mjlab cfg."""
    from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

    terrain = world["shared"]["terrain"]
    if terrain["kind"] == "plane":
        return TerrainEntityCfg(terrain_type="plane", num_envs=1)
    layout = terrain["layout"]
    size = tuple(map(float, layout["tile_size_m"]))
    patches = _flat_patch_cfgs(world)
    sub_terrains = {
        name: _terrain_sub_cfg(
            entry["type"], entry["nominal"],
            proportion=float(entry.get("proportion", 1.0)), size=size,
            flat_patches=patches)
        for name, entry in sorted(terrain["sub_terrains"].items())
    }
    curriculum = layout["mode"] == "curriculum_grid"
    cols = len(sub_terrains) if curriculum else int(layout["cols"])
    difficulty = float(terrain["evaluation_difficulty"])
    generator = TerrainGeneratorCfg(
        seed=int(world["shared"]["eval_seed"]), curriculum=curriculum,
        size=size, border_width=float(layout["border_width_m"]),
        num_rows=int(layout["rows"]), num_cols=cols,
        sub_terrains=sub_terrains,
        difficulty_range=(difficulty, difficulty), add_lights=False)
    return TerrainEntityCfg(
        terrain_type="generator", terrain_generator=generator, num_envs=1)


def resolve_course(world: Mapping[str, Any]) -> tuple[ResolvedPrimitive, ...]:
    """Resolve the stable-ID linear course grammar into fixed boxes."""
    result: list[ResolvedPrimitive] = []
    cursor = float(world["shared"].get(
        "obstacles", {}).get("start_offset_m", 0.0))

    def box(source: str, suffix: str, length: float, width: float,
            height: float, *, z: float | None = None) -> None:
        nonlocal cursor
        result.append(ResolvedPrimitive(
            primitive_id=f"{source}__{suffix}", source_id=source, shape="box",
            position_m=(cursor + length / 2, 0.0,
                        height / 2 if z is None else z),
            size_m=(length, width, height)))
        cursor += length

    for entry in world["shared"].get("obstacles", {}).get("course", []):
        source = str(entry["id"])
        kind = entry["element"]
        nominal = entry["nominal"]
        if kind == "gap":
            cursor += float(nominal["length_m"])
        elif kind == "platform":
            box(source, "platform", float(nominal["length_m"]),
                float(nominal.get("width_m", 1.0)),
                float(nominal["height_m"]))
        elif kind == "beam":
            height = float(nominal.get("height_m", 0.15))
            box(source, "beam", float(nominal["length_m"]),
                float(nominal["width_m"]), height)
        elif kind == "wall":
            box(source, "wall", float(nominal.get("thickness_m", 0.15)),
                float(nominal["width_m"]), float(nominal["height_m"]))
        elif kind == "stairs":
            count = int(nominal["num_steps"])
            step_width = float(nominal["step_width_m"])
            step_height = float(nominal["step_height_m"])
            for index in range(count):
                height = step_height * (index + 1)
                box(source, f"step_{index:03d}", step_width,
                    float(nominal.get("width_m", 1.0)), height)
        elif kind == "stepping_stones":
            count = int(nominal["count"])
            stone_size = float(nominal["stone_size_m"])
            spacing = float(nominal["spacing_m"])
            for index in range(count):
                box(source, f"stone_{index:03d}", stone_size,
                    float(nominal.get("width_m", stone_size)),
                    float(nominal.get("height_m", 0.1)))
                if index + 1 < count:
                    cursor += spacing
        cursor += float(nominal.get("gap_after_m", 0.0))
    return tuple(result)


def _resolve_pose(
    obj: Mapping[str, Any], zones: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pose = obj["nominal"]["pose"]
    if "position_m" in pose:
        position = tuple(map(float, pose["position_m"]))
    else:
        zone = zones[str(pose["placement"]).split(":", 1)[1]]
        center = list(map(float, zone["center_m"]))
        if len(center) == 2:
            center.append(float(pose.get("z_m", 0.0)))
        else:
            center[2] = float(pose.get("z_m", center[2]))
        position = tuple(center)
    rotation = tuple(map(float, pose.get(
        "quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))))
    return position, rotation


def resolve_objects(world: Mapping[str, Any]) -> dict[str, Any]:
    zones = world["shared"].get("zones", {})
    resolved: dict[str, Any] = {}
    for name, obj in sorted(world["shared"].get("objects", {}).items()):
        position, rotation = _resolve_pose(obj, zones)
        resolved[name] = {
            "shape": obj["shape"], "fixed": bool(obj.get("fixed", False)),
            "nominal": copy.deepcopy(obj["nominal"]),
            "position_m": list(position),
            "quaternion_wxyz": list(rotation),
        }
    return resolved


def _object_entity(name: str, resolved: Mapping[str, Any]) -> Any:
    import mujoco
    from mjlab.entity import EntityCfg

    nominal = resolved["nominal"]
    shape = resolved["shape"]
    fixed = bool(resolved["fixed"])

    def add_geom(body: Any, *, suffix: str, geom_type: Any,
                 size: tuple[float, float, float],
                 pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        friction = float(nominal.get("friction", 0.8))
        restitution = float(nominal.get("restitution", 0.0))
        geom = body.add_geom(
            name=f"{name}_{suffix}", type=geom_type, size=size, pos=pos,
            mass=float(nominal.get("mass_kg", 1.0)),
            friction=(friction, 0.005, 0.0001))
        # MuJoCo exposes compliant contact damping, not a direct restitution
        # coefficient. This monotonic mapping preserves the declared intent.
        geom.solref[:] = (0.02, max(0.05, 1.0 - restitution))
        if "rgba" in nominal:
            geom.rgba[:] = tuple(map(float, nominal["rgba"]))

    def spec_fn() -> Any:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=name)
        if not fixed:
            body.add_freejoint(name=f"{name}_freejoint")
        if shape == "sphere":
            add_geom(body, suffix="geom", geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=(float(nominal["radius_m"]), 0.0, 0.0))
        elif shape == "box":
            sx, sy, sz = map(float, nominal["size_m"])
            add_geom(body, suffix="geom", geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=(sx / 2, sy / 2, sz / 2))
        elif shape in {"cylinder", "capsule"}:
            geom_type = (mujoco.mjtGeom.mjGEOM_CYLINDER if shape == "cylinder"
                         else mujoco.mjtGeom.mjGEOM_CAPSULE)
            add_geom(body, suffix="geom", geom_type=geom_type,
                     size=(float(nominal["radius_m"]),
                           float(nominal["height_m"]) / 2, 0.0))
        elif shape == "frame":
            opening_width, opening_height = map(float, nominal["opening_m"])
            radius = float(nominal["post_radius_m"])
            depth = float(nominal.get("depth_m", radius * 2))
            for suffix, pos, size in (
                ("left_post", (0.0, -(opening_width / 2 + radius), 0.0),
                 (depth / 2, radius, opening_height / 2 + radius)),
                ("right_post", (0.0, opening_width / 2 + radius, 0.0),
                 (depth / 2, radius, opening_height / 2 + radius)),
                ("crossbar", (0.0, 0.0, opening_height / 2 + radius),
                 (depth / 2, opening_width / 2 + radius, radius)),
            ):
                add_geom(body, suffix=suffix,
                         geom_type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=size, pos=pos)
        else:  # guarded by WorldSpec validation
            raise WorldCompileError(f"unsupported object shape {shape!r}")
        return spec

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(resolved["position_m"]),
            rot=tuple(resolved["quaternion_wxyz"])),
        spec_fn=spec_fn)


def _world_spec_editor(
    course: tuple[ResolvedPrimitive, ...], zones: Mapping[str, Any],
) -> Callable[[Any], None]:
    def edit(spec: Any) -> None:
        import mujoco

        # mjlab offsets each parallel robot by an environment origin. Static
        # authored geometry must be repeated at those same origins; adding one
        # course at global (0, 0) makes every non-central robot see no boxes and
        # sends world-space waypoint rewards toward the wrong environment.
        # TerrainEntity has already emitted these sites before Scene.spec_fn is
        # called, so they are the simulator's authoritative placement source.
        indexed_origins: list[tuple[int, tuple[float, float, float]]] = []
        for site in spec.sites:
            match = re.fullmatch(r"env_origin_(\d+)", str(site.name or ""))
            if match:
                indexed_origins.append((
                    int(match.group(1)), tuple(map(float, site.pos))))
        if not indexed_origins:
            indexed_origins = [(0, (0.0, 0.0, 0.0))]

        # Generator terrains may assign multiple envs to one physical tile.
        # One course per unique origin avoids overlapping duplicate collision
        # geoms while preserving the terrain allocator's exact coordinates.
        origins: list[tuple[int, tuple[float, float, float]]] = []
        seen: set[tuple[float, float, float]] = set()
        for env_index, origin in sorted(indexed_origins):
            key = tuple(round(value, 7) for value in origin)
            if key not in seen:
                seen.add(key)
                origins.append((env_index, origin))

        single = len(origins) == 1
        for env_index, origin in origins:
            suffix = "" if single else f"__env_{env_index:04d}"
            course_body = spec.worldbody.add_body(
                name=f"authored_course{suffix}")
            for primitive_index, primitive in enumerate(course):
                sx, sy, sz = primitive.size_m
                pos = tuple(
                    origin[axis] + primitive.position_m[axis]
                    for axis in range(3)
                )
                # Alternating high-contrast colors make platform boundaries
                # and gaps legible in compressed rollout video.
                rgba = (
                    (0.16, 0.52, 0.92, 1.0)
                    if primitive_index % 2 == 0
                    else (0.95, 0.58, 0.16, 1.0)
                )
                course_body.add_geom(
                    name=f"obstacle__{primitive.primitive_id}{suffix}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos, size=(sx / 2, sy / 2, sz / 2),
                    mass=0.0, friction=(1.0, 0.005, 0.0001), rgba=rgba)
            for name, zone in sorted(zones.items()):
                if zone["kind"] == "disk":
                    local = tuple(map(float, zone["center_m"])) + (0.01,)
                    size = (float(zone["radius_m"]), 0.01, 0.0)
                    geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
                else:
                    local = tuple(map(float, zone["center_m"]))
                    sx, sy, sz = map(float, zone["size_m"])
                    size = (sx / 2, sy / 2, sz / 2)
                    geom_type = mujoco.mjtGeom.mjGEOM_BOX
                center = tuple(origin[axis] + local[axis] for axis in range(3))
                spec.worldbody.add_site(
                    name=f"zone__{name}{suffix}", pos=center, size=size,
                    type=geom_type, group=4, rgba=(0.1, 0.9, 0.1, 0.25))
    return edit


def compile_scene_cfg(
    world: Mapping[str, Any], *, robot: RobotCapability | None = None,
) -> tuple[Any, tuple[ResolvedPrimitive, ...], dict[str, Any]]:
    """Compile shared world geometry into an mjlab SceneCfg."""
    from mjlab.scene import SceneCfg

    course = resolve_course(world)
    objects = resolve_objects(world)
    entities: dict[str, Any] = {
        name: _object_entity(name, resolved)
        for name, resolved in objects.items()
    }
    if robot is not None:
        entities = {"robot": build_robot_entity_cfg(robot), **entities}
    cfg = SceneCfg(
        num_envs=1, terrain=compile_terrain_cfg(world), entities=entities,
        spec_fn=_world_spec_editor(course, world["shared"].get("zones", {})))
    return cfg, course, objects


def _contact_match(
    selector: str, world: Mapping[str, Any], robot: RobotCapability,
) -> tuple[Any, Mapping[str, Any]]:
    from mjlab.sensor import ContactMatch

    kind, raw = selector.split(":", 1)
    if kind == "robot":
        names: list[str] = []
        for role in raw.split("|"):
            if role == "any":
                names.extend((robot.root_body, *(
                    name for resolved in robot.body_roles.values()
                    for name in resolved
                )))
            else:
                namespace, resolved = robot.resolve_semantic_role(role)
                if namespace != "body":
                    raise WorldCompileError(
                        f"contact selector {selector!r} resolves to a site; "
                        "contacts require body roles")
                names.extend(resolved)
        concrete = tuple(dict.fromkeys(names))
        return (ContactMatch(mode="body", pattern=concrete, entity="robot"),
                {"kind": kind, "mode": "body", "entity": "robot",
                 "names": list(concrete)})
    if kind == "object":
        return (ContactMatch(mode="body", pattern=raw, entity=raw),
                {"kind": kind, "mode": "body", "entity": raw,
                 "names": [raw]})
    if kind == "obstacle":
        primitives = [
            primitive for primitive in resolve_course(world)
            if primitive.source_id == raw or primitive.primitive_id == raw
        ]
        names = tuple(
            f"obstacle__{primitive.primitive_id}" for primitive in primitives)
        if not names:
            raise WorldCompileError(
                f"contact selector {selector!r} resolves to no course geom")
        pattern: str | tuple[str, ...] = (
            names[0] if len(names) == 1 else names)
        return (ContactMatch(mode="geom", pattern=pattern),
                {"kind": kind, "mode": "geom", "entity": None,
                 "names": list(names)})
    if kind == "world" and raw == "terrain":
        return (ContactMatch(mode="body", pattern="terrain"),
                {"kind": kind, "mode": "body", "entity": None,
                 "names": ["terrain"]})
    raise WorldCompileError(
        f"contact selector {selector!r} has no colliding simulator entity")


def compile_task_runtime(
    world: Mapping[str, Any], task: Mapping[str, Any],
    robot: RobotCapability,
) -> TaskRuntimePlan:
    """Compile TaskSpec semantics without assuming manager/group names."""
    from mjlab.sensor import (
        ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg,
    )

    sensors: list[Any] = []
    contacts: list[RuntimeContactBinding] = []
    for group in ("desired", "forbidden", "terminate_on"):
        for index, pair in enumerate(
                task["shared"].get("contacts", {}).get(group, ())):
            primary, primary_meta = _contact_match(pair[0], world, robot)
            secondary, secondary_meta = _contact_match(pair[1], world, robot)
            name = f"authored_contact__{group}__{index}"
            sensors.append(ContactSensorCfg(
                name=name, primary=primary, secondary=secondary,
                fields=("found", "force", "dist"), reduce="maxforce",
                secondary_policy="first"))
            contacts.append(RuntimeContactBinding(
                sensor_name=name, group=group,
                selectors=(str(pair[0]), str(pair[1])),
                resolved=(primary_meta, secondary_meta)))

    observations = copy.deepcopy(task["shared"].get("observations", {}))
    terrain_kind = world["shared"]["terrain"]["kind"]
    height_scan = observations.get("height_scan", "auto")
    height_enabled = height_scan is True or (
        height_scan == "auto" and terrain_kind == "generator")
    if height_enabled:
        sensors.append(RayCastSensorCfg(
            name="authored_height_scan",
            frame=ObjRef(type="body", name=robot.root_body, entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.2),
            max_distance=5.0, exclude_parent_body=True,
            include_geom_groups=(0,), debug_vis=False))
    raw_end_effectors = observations.get("end_effector_relative", ())
    end_effectors = (
        ["end_effector"]
        if raw_end_effectors is True or raw_end_effectors == "auto"
        else []
        if raw_end_effectors is False
        else list(raw_end_effectors)
    )
    observation_bindings = {
        "proprioception": observations.get("proprioception", True),
        "height_scan": (
            {"sensor": "authored_height_scan", "enabled": True}
            if height_enabled else {"enabled": False}),
        "object_relative": list(observations.get("object_relative", ())),
        "region_relative": list(observations.get("region_relative", ())),
        "end_effector_relative": end_effectors,
    }
    reset_bindings = tuple(copy.deepcopy(
        task.get("train", {}).get("goal_sampling", ())))
    goal = copy.deepcopy(task["shared"]["goal"])
    goal_binding = {
        "id": goal["id"], "type": goal["type"],
        "predicate": copy.deepcopy(goal["success"]),
        "arguments": {key: copy.deepcopy(goal[key]) for key in
                      ("subject", "region", "target", "waypoints")
                      if key in goal},
    }
    termination = copy.deepcopy(task["shared"].get("termination", {}))
    termination["contact_sensors"] = [
        item.sensor_name for item in contacts if item.group == "terminate_on"]
    return TaskRuntimePlan(
        contacts=tuple(contacts), observation_bindings=observation_bindings,
        reset_bindings=reset_bindings, goal_binding=goal_binding,
        termination_bindings=termination, sensor_cfgs=tuple(sensors))


def _model_bytes(model: Any) -> bytes:
    import mujoco

    with tempfile.TemporaryDirectory(prefix="sculpt-world-model-") as raw:
        path = Path(raw) / "scene.mjb"
        mujoco.mj_saveModel(model, str(path))
        return path.read_bytes()


def _compiled_model_hash(model: Any) -> str:
    """Hash compiled model content without MJB allocator/padding bytes."""
    digest = hashlib.sha256()
    digest.update(bytes(model.names))
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if not isinstance(value, np.ndarray):
            continue
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _robot_asset_hash_from_cfg(robot_cfg: Any) -> str:
    """Fingerprint one unprefixed robot asset by compiled physical content."""
    try:
        entity = robot_cfg.build()
        return _compiled_model_hash(entity.spec.copy().compile())
    except Exception as exc:
        raise WorldCompileError(
            f"robot asset could not be compiled for hashing: {exc}") from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resolved_evaluation(
    world: Mapping[str, Any],
    task: Mapping[str, Any],
    channel_catalog: ChannelCatalog | Mapping[str, Any],
    resolved_eval: ResolvedEvaluation | Mapping[str, Any],
    *,
    asset_base: Path | str,
) -> ResolvedEvaluation:
    """Fail closed unless a frozen evaluation matches the current runtime.

    This is the shared verification boundary for promotion, CLI validation,
    and runner startup.  It validates declarative hashes, installed compiler
    and dependencies, descriptor/asset identity, every frozen file, and the
    semantic content hash of the persisted MJB.
    """
    import mujoco

    manifest = (
        resolved_eval if isinstance(resolved_eval, ResolvedEvaluation)
        else ResolvedEvaluation.from_dict(resolved_eval)
    )
    catalog = (
        channel_catalog if isinstance(channel_catalog, ChannelCatalog)
        else ChannelCatalog.from_dict(channel_catalog)
    )
    if _hash_mapping(world) != manifest.world_hash:
        raise WorldCompileError(
            "selected WorldSpec does not match evaluation manifest")
    if _hash_mapping(task) != manifest.task_hash:
        raise WorldCompileError(
            "selected TaskSpec does not match evaluation manifest")
    if catalog.catalog_hash != manifest.channel_catalog_hash:
        raise WorldCompileError(
            "selected channel catalog does not match evaluation manifest")
    if catalog.world_hash != manifest.world_hash or \
            catalog.task_hash != manifest.task_hash:
        raise WorldCompileError(
            "channel catalog world/task hashes do not match manifest")
    if not bool(manifest.admission.get("ok", False)):
        raise WorldCompileError(
            "selected evaluation manifest has not passed admission gates")

    simulator = simulator_capability()
    if manifest.compiler_hash != simulator.compiler_version:
        raise WorldCompileError(
            "world compiler version differs from admitted evaluation")
    if dict(manifest.dependency_versions) != dict(
            sorted(simulator.dependency_versions.items())):
        raise WorldCompileError(
            "simulator dependency versions differ from admitted evaluation")
    if manifest.simulator_capability_hash != _hash_mapping(
            simulator.to_dict()):
        raise WorldCompileError(
            "simulator capability descriptor differs from admission")

    robot_data = world["shared"]["robot"]
    robot = resolve_robot_capability(
        robot_data["capability_id"],
        required=robot_data.get("required_capabilities", ()),
        extra_paths=([robot_data["descriptor_path"]]
                     if robot_data.get("descriptor_path") else ()),
    )
    if manifest.robot_capability_hash != _hash_mapping(robot.to_dict()):
        raise WorldCompileError(
            "robot capability descriptor differs from admission")
    installed_asset_hash = _robot_asset_hash_from_cfg(
        build_robot_entity_cfg(robot))
    if manifest.robot_asset_hash != installed_asset_hash:
        raise WorldCompileError(
            "installed robot asset differs from admitted evaluation")

    base = Path(asset_base).expanduser().resolve()

    def resolve_asset(relative: str) -> Path:
        candidate = (base / relative).resolve()
        if candidate != base and base not in candidate.parents:
            raise WorldCompileError(
                f"materialized asset escapes evaluation store: {relative}")
        return candidate

    for record in manifest.materialized_assets.get("files", ()):
        asset = resolve_asset(str(record["path"]))
        if not asset.is_file() or _file_hash(asset) != record["sha256"]:
            raise WorldCompileError(
                f"materialized evaluation asset hash mismatch: {asset}")
        if asset.stat().st_size != int(record["bytes"]):
            raise WorldCompileError(
                f"materialized evaluation asset size mismatch: {asset}")

    mjb_relative = manifest.materialized_assets.get("evaluation_mjb")
    expected_mjb_hash = manifest.materialized_assets.get(
        "evaluation_mjb_sha256")
    if not mjb_relative or not expected_mjb_hash:
        raise WorldCompileError(
            "admitted evaluation has no frozen evaluation MJB")
    mjb_path = resolve_asset(str(mjb_relative))
    if not mjb_path.is_file() or _file_hash(mjb_path) != expected_mjb_hash:
        raise WorldCompileError(
            f"materialized evaluation model hash mismatch: {mjb_path}")
    try:
        frozen_model = mujoco.MjModel.from_binary_path(str(mjb_path))
    except Exception as exc:
        raise WorldCompileError(
            f"materialized evaluation model cannot be loaded: {exc}") from exc
    if _compiled_model_hash(frozen_model) != manifest.compiled_model_hash:
        raise WorldCompileError(
            "materialized evaluation model content differs from manifest")
    return manifest


def _materialize_terrain(
    terrain: Any, output_dir: Path, *, manifest_base: Path,
) -> dict[str, Any]:
    """Persist buffer-backed generated terrain as exact file-backed assets."""
    import imageio.v3 as iio
    import mujoco

    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    spec = terrain.spec.copy()
    original_model = spec.compile()

    for index, hfield in enumerate(list(spec.hfields)):
        if len(hfield.userdata) == 0:
            continue
        name = hfield.name
        size = tuple(map(float, hfield.size))
        path = assets / f"hfield_{index:04d}.bin"
        values = np.asarray(hfield.userdata, dtype=np.float32)
        path.write_bytes(
            struct.pack("<ii", int(hfield.nrow), int(hfield.ncol))
            + values.tobytes(order="C"))
        spec.delete(hfield)
        spec.add_hfield(
            name=name, file=str(path.resolve()), size=size,
            content_type="image/vnd.mujoco.hfield")

    for index, texture in enumerate(list(spec.textures)):
        if len(texture.data) == 0:
            continue
        name = texture.name
        tex_type = texture.type
        channels = int(texture.nchannel)
        pixels = np.frombuffer(texture.data, dtype=np.uint8).reshape(
            int(texture.height), int(texture.width), channels)
        path = assets / f"texture_{index:04d}.png"
        iio.imwrite(path, pixels)
        spec.delete(texture)
        spec.add_texture(
            name=name, type=tex_type, file=str(path.resolve()),
            content_type="image/png")

    xml_path = output_dir / "terrain.xml"
    xml = spec.to_xml().replace(
        assets.resolve().as_posix() + "/", "assets/")
    xml_path.write_text(xml, encoding="utf-8")
    frozen_model = spec.compile()
    if frozen_model.nhfield != original_model.nhfield or not np.array_equal(
            frozen_model.hfield_data, original_model.hfield_data):
        raise WorldCompileError(
            "materialized terrain did not preserve exact heightfield data")
    records = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            records.append({
                "path": path.relative_to(manifest_base).as_posix(),
                "sha256": _file_hash(path), "bytes": path.stat().st_size,
            })
    return {
        "terrain_xml": xml_path.relative_to(manifest_base).as_posix(),
        "files": records,
    }


def compile_world(
    world: Mapping[str, Any], task: Mapping[str, Any], *,
    materialize_dir: Path | str | None = None,
    runtime_task_id: str | None = None,
) -> CompiledWorld:
    """Validate, build, and freeze one world/task pair on CPU."""
    import mujoco
    from mjlab.scene import Scene

    world_errors = validate_world_spec(world)
    task_errors = validate_task_spec(task, world=dict(world))
    if world_errors or task_errors:
        errors = [f"world: {item}" for item in world_errors]
        errors.extend(f"task: {item}" for item in task_errors)
        raise WorldCompileError("invalid authored environment:\n- " + "\n- ".join(errors))

    robot_raw = world["shared"]["robot"]
    robot = resolve_robot_capability(
        robot_raw["capability_id"],
        required=robot_raw.get("required_capabilities", ()),
        extra_paths=([robot_raw["descriptor_path"]]
                     if robot_raw.get("descriptor_path") else ()))
    simulator = simulator_capability()
    catalog = compile_channel_catalog(dict(world), dict(task))
    task_runtime = compile_task_runtime(world, task, robot)
    scene_cfg, course, objects = compile_scene_cfg(world, robot=robot)
    scene_cfg.sensors = task_runtime.sensor_cfgs
    counter = iter(range(1_000_000))

    class _DeterministicUuid:
        def __init__(self, index: int) -> None:
            self.hex = hashlib.sha256(
                f"{world['shared']['eval_seed']}:{index}".encode()).hexdigest()[:32]

    try:
        # mjlab otherwise calls uuid4 for heightfield asset names, making the
        # compiled MJB hash differ despite identical physics and seeds.
        with (
            mock.patch(
                "mjlab.terrains.heightfield_terrains.uuid.uuid4",
                side_effect=lambda: _DeterministicUuid(next(counter)),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            scene = Scene(scene_cfg, device="cpu")
            model = scene.compile()
    except Exception as exc:  # mjlab/MuJoCo expose several exception classes
        raise WorldCompileError(f"MuJoCo scene build failed: {exc}") from exc

    model_bytes = _model_bytes(model)
    materialized_assets: dict[str, Any] = {}
    if materialize_dir is not None:
        root = Path(materialize_dir)
        root.mkdir(parents=True, exist_ok=True)
        exact_path = root / "evaluation_scene.mjb"
        exact_path.write_bytes(model_bytes)
        materialized_assets = {
            "evaluation_mjb": exact_path.relative_to(root.parent).as_posix(),
            "evaluation_mjb_sha256": sha256_bytes(model_bytes),
        }
        if scene.terrain is not None and scene_cfg.terrain.terrain_type == "generator":
            materialized_assets.update(_materialize_terrain(
                scene.terrain, root / "terrain", manifest_base=root.parent))

    terrain_resolved: dict[str, Any] = copy.deepcopy(world["shared"]["terrain"])
    if scene.terrain is not None:
        if scene.terrain.terrain_origins is not None:
            terrain_resolved["origins_m"] = (
                scene.terrain.terrain_origins.detach().cpu().tolist())
        terrain_resolved["flat_patches"] = {
            name: values.detach().cpu().tolist()
            for name, values in sorted(scene.terrain.flat_patches.items())
        }
        terrain_resolved["flat_patch_radii_m"] = dict(
            sorted(scene.terrain.flat_patch_radii.items()))

    manifest = ResolvedEvaluation.build(
        world_hash=_hash_mapping(world), task_hash=_hash_mapping(task),
        compiler_hash=simulator.compiler_version,
        robot_capability_hash=_hash_mapping(robot.to_dict()),
        robot_asset_hash=_robot_asset_hash_from_cfg(
            scene_cfg.entities["robot"]),
        simulator_capability_hash=_hash_mapping(simulator.to_dict()),
        dependency_versions=dict(sorted(simulator.dependency_versions.items())),
        runtime_task_id=runtime_task_id,
        eval_seed=int(world["shared"]["eval_seed"]), terrain=terrain_resolved,
        course=course, objects=objects,
        zones=copy.deepcopy(world["shared"].get("zones", {})),
        task_shared=copy.deepcopy(task["shared"]),
        channel_catalog_hash=catalog.catalog_hash,
        compiled_model_hash=_compiled_model_hash(model),
        materialized_assets=materialized_assets,
        admission={"ok": False, "status": "not_run", "violations": []})
    inventory = {
        "bodies": int(model.nbody), "geoms": int(model.ngeom),
        "contacts": int(model.nconmax), "constraints": int(model.neq),
        "heightfields": int(model.nhfield),
        "heightfield_texels": int(sum(
            int(model.hfield_nrow[i]) * int(model.hfield_ncol[i])
            for i in range(model.nhfield))),
        "sites": int(model.nsite), "joints": int(model.njnt),
    }
    return CompiledWorld(
        scene_cfg=scene_cfg, resolved_eval=manifest,
        channel_catalog=catalog,
        train_variations=tuple(copy.deepcopy(
            world.get("train", {}).get("variations", ()))),
        robot=robot, simulator=simulator, model_inventory=inventory,
        task_runtime=task_runtime,
        _scene=scene, _model=model)


def _compose_spec_editors(
    first: Callable[[Any], None] | None,
    second: Callable[[Any], None] | None,
) -> Callable[[Any], None] | None:
    if first is None:
        return second
    if second is None:
        return first

    def composed(spec: Any) -> None:
        first(spec)
        second(spec)
    return composed


def apply_compiled_world(env_cfg: Any, compiled: CompiledWorld) -> None:
    """Overlay a compiled world onto a deep-copied mjlab environment config."""
    scene_cfg = getattr(env_cfg, "scene", env_cfg)
    scene_cfg.terrain = compiled.scene_cfg.terrain
    entities = dict(getattr(scene_cfg, "entities", {}))
    entities.update(compiled.scene_cfg.entities)
    scene_cfg.entities = entities
    scene_cfg.spec_fn = _compose_spec_editors(
        getattr(scene_cfg, "spec_fn", None), compiled.scene_cfg.spec_fn)
    existing_sensors = {
        sensor.name: sensor for sensor in getattr(scene_cfg, "sensors", ())}
    existing_sensors.update({
        sensor.name: sensor for sensor in compiled.task_runtime.sensor_cfgs})
    scene_cfg.sensors = tuple(existing_sensors.values())
    _install_task_observations(
        env_cfg,
        compiled.task_runtime,
        zones=compiled.resolved_eval.zones,
        robot=compiled.robot,
    )


def train_difficulty_span(world: Mapping[str, Any]) -> tuple[float, float] | None:
    """The train-time terrain difficulty span, or None to keep the pinned
    evaluation difficulty.

    Env-authoring §10.1: the within-run mjlab curriculum trains across
    ``train.curriculum.difficulty_range`` (curriculum_grid rows ARE the
    difficulty axis), while evaluation always replays the materialized
    manifest at the single nominal difficulty. Only a well-formed,
    non-degenerate ``[lo, hi]`` on generator terrain in curriculum_grid
    mode yields a span — everything else trains exactly like evaluation.
    """
    terrain = world.get("shared", {}).get("terrain", {})
    if terrain.get("kind") != "generator":
        return None
    if terrain.get("layout", {}).get("mode") != "curriculum_grid":
        return None
    rng = world.get("train", {}).get("curriculum", {}).get("difficulty_range")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return None
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= lo < hi <= 1.0):
        return None
    return lo, hi


def expand_train_terrain_difficulty(
    compiled: CompiledWorld, world: Mapping[str, Any],
) -> bool:
    """Widen the TRAIN scene's generator difficulty to the curriculum span.

    Mutates only the in-memory train scene cfg after ``compile_world`` —
    the evaluation manifest, its materialized assets, and every recorded
    hash were computed from the pinned nominal difficulty and stay
    untouched. Returns True when a span was applied. Under mjlab's
    curriculum mode the atlas rows then interpolate lo→hi and the base
    task's terrain-levels curriculum term promotes/demotes environment
    origins within the run (update_env_origins); evaluation rollouts load
    the frozen materialized terrain, where origin promotion is a no-op.
    """
    span = train_difficulty_span(world)
    if span is None:
        return False
    terrain_cfg = getattr(compiled.scene_cfg, "terrain", None)
    generator = getattr(terrain_cfg, "terrain_generator", None)
    if generator is None:
        return False
    generator.difficulty_range = span
    return True


def _apply_scene_and_runtime(
    env_cfg: Any,
    authored_scene: Any,
    runtime: TaskRuntimePlan,
    *,
    zones: Mapping[str, Any],
    robot: RobotCapability,
) -> None:
    scene_cfg = getattr(env_cfg, "scene", env_cfg)
    scene_cfg.terrain = authored_scene.terrain
    entities = dict(getattr(scene_cfg, "entities", {}))
    entities.update(authored_scene.entities)
    scene_cfg.entities = entities
    scene_cfg.spec_fn = _compose_spec_editors(
        getattr(scene_cfg, "spec_fn", None), authored_scene.spec_fn)
    existing_sensors = {
        sensor.name: sensor for sensor in getattr(scene_cfg, "sensors", ())}
    existing_sensors.update({sensor.name: sensor for sensor in runtime.sensor_cfgs})
    scene_cfg.sensors = tuple(existing_sensors.values())
    _install_task_observations(
        env_cfg, runtime, zones=zones, robot=robot)


def _runtime_robot_hash(env_cfg: Any) -> str | None:
    scene_cfg = getattr(env_cfg, "scene", env_cfg)
    robot_cfg = getattr(scene_cfg, "entities", {}).get("robot")
    if robot_cfg is None:
        return None
    try:
        return _robot_asset_hash_from_cfg(robot_cfg)
    except Exception:
        return None


def _reconcile_terrain_curriculum(env_cfg: Any) -> tuple[str, ...]:
    """Remove curriculum terms incompatible with the overlaid terrain.

    mjlab rough-terrain tasks ship a ``terrain_levels`` curriculum whose
    implementation requires ``scene.terrain.terrain_generator``.  An authored
    plane (and a frozen materialized evaluation terrain) deliberately has no
    live generator, so retaining that base-task term makes the first reset
    assert before training begins.  Detect the dependency from the curriculum
    term/function semantics—not a robot or task identifier—and preserve every
    unrelated curriculum term.
    """
    scene_cfg = getattr(env_cfg, "scene", env_cfg)
    terrain_cfg = getattr(scene_cfg, "terrain", None)
    if getattr(terrain_cfg, "terrain_generator", None) is not None:
        return ()
    curriculum = getattr(env_cfg, "curriculum", None)
    if not isinstance(curriculum, dict):
        return ()

    removed: list[str] = []
    for name, term in tuple(curriculum.items()):
        func = getattr(term, "func", None)
        identifiers = {
            str(name).lower(),
            str(getattr(func, "__name__", "")).lower(),
        }
        requires_live_generator = any(
            value == "terrain_levels" or value.startswith("terrain_levels_")
            for value in identifiers
        )
        if requires_live_generator:
            curriculum.pop(name, None)
            removed.append(
                f"curriculum:{name}→removed(no live terrain generator)"
            )
    return tuple(removed)


def reset_robot_along_waypoint_route(
    env: Any,
    env_ids: Any,
    *,
    waypoints_m: tuple[tuple[float, float, float], ...],
    midroute_probability: float,
    approach_distance_m: tuple[float, float],
    lateral_jitter_m: float,
    asset_name: str = "robot",
) -> None:
    """Train-only route RSI in the same local frame as authored geometry.

    A configurable fraction of resets starts immediately before a later
    waypoint, facing it.  The remaining resets retain the base task's entrance
    pose and therefore preserve full-route learning.  The sampled logical route
    index is published once on ``env`` so command, reward, and metric runtimes
    all resume from the same state; evaluation never installs this event.
    """
    import torch
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs, device=env.device, dtype=torch.long)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    starts = getattr(env, "_sculptor_waypoint_start_index", None)
    if starts is None or tuple(starts.shape) != (int(env.num_envs),):
        starts = torch.zeros(
            int(env.num_envs), device=env.device, dtype=torch.long)
        env._sculptor_waypoint_start_index = starts
    starts[env_ids] = 0

    points = torch.as_tensor(
        waypoints_m, device=env.device, dtype=torch.float32)
    if points.shape[0] <= 1 or env_ids.numel() == 0:
        return
    use_midroute = torch.rand(
        env_ids.numel(), device=env.device) < float(midroute_probability)
    selected_env_ids = env_ids[use_midroute]
    if selected_env_ids.numel() == 0:
        return
    selected_indices = torch.randint(
        1, points.shape[0],
        (selected_env_ids.numel(),),
        device=env.device,
        dtype=torch.long,
    )
    starts[selected_env_ids] = selected_indices

    target = points[selected_indices, :2]
    previous = points[selected_indices - 1, :2]
    direction = target - previous
    direction = direction / torch.clamp(
        torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-6)
    low, high = (float(value) for value in approach_distance_m)
    approach = low + (high - low) * torch.rand(
        selected_env_ids.numel(), 1, device=env.device)
    normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    lateral = (
        2.0 * torch.rand(
            selected_env_ids.numel(), 1, device=env.device) - 1.0
    ) * float(lateral_jitter_m)
    local_xy = target - direction * approach + normal * lateral

    robot = env.scene[asset_name]
    if bool(getattr(robot, "is_fixed_base", False)):
        starts[selected_env_ids] = 0
        return
    root_state = robot.data.default_root_state[selected_env_ids].clone()
    root_state[:, :2] = (
        local_xy + env.scene.env_origins[selected_env_ids, :2])
    yaw = torch.atan2(direction[:, 1], direction[:, 0])
    zeros = torch.zeros_like(yaw)
    root_state[:, 3:7] = quat_mul(
        root_state[:, 3:7],
        quat_from_euler_xyz(zeros, zeros, yaw),
    )
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=selected_env_ids)


def _resolved_waypoint_points(
    manifest: ResolvedEvaluation,
    requested_waypoints: Any,
) -> tuple[tuple[float, float, float], ...]:
    raw_waypoints: list[tuple[float, float, float]] = []
    if requested_waypoints == "auto":
        raw_waypoints = [
            tuple(float(value) for value in primitive.position_m)
            for primitive in manifest.course
        ]
    else:
        by_source = {
            str(primitive.source_id): primitive
            for primitive in manifest.course
        }
        for waypoint_name in requested_waypoints:
            name = str(waypoint_name)
            if name in manifest.zones:
                center = list(manifest.zones[name]["center_m"])
                if len(center) == 2:
                    center.append(0.0)
                raw_waypoints.append(tuple(float(value) for value in center))
            elif name in by_source:
                raw_waypoints.append(tuple(
                    float(value) for value in by_source[name].position_m))
            else:  # frozen compiler validation should make this unreachable
                raise WorldCompileError(
                    f"waypoint {name!r} is absent from the resolved world")
    if not raw_waypoints:
        raise WorldCompileError(
            "waypoint sequence has no resolved command targets")
    return tuple(raw_waypoints)


def _object_horizontal_radius(record: Mapping[str, Any]) -> float:
    """Conservative horizontal bounding radius for an authored object."""
    nominal = record.get("nominal", record)
    shape = str(record.get("shape", ""))
    if shape == "box":
        size = tuple(float(value) for value in nominal.get("size_m", ()))
        if len(size) >= 2:
            return math.hypot(0.5 * size[0], 0.5 * size[1])
    if shape in {"sphere", "cylinder", "capsule"}:
        try:
            return max(0.0, float(nominal.get("radius_m", 0.0)))
        except (TypeError, ValueError):
            return 0.0
    if shape == "frame":
        opening = tuple(float(value) for value in nominal.get("opening_m", ()))
        try:
            post_radius = max(0.0, float(nominal.get("post_radius_m", 0.0)))
        except (TypeError, ValueError):
            post_radius = 0.0
        if len(opening) >= 1:
            return 0.5 * max(opening) + post_radius
    return 0.0


def _forbidden_object_names(manifest: ResolvedEvaluation) -> tuple[str, ...]:
    """Resolve authored forbidden-contact object selectors without name keying."""
    contacts = manifest.task_shared.get("contacts", {})
    pairs = contacts.get("forbidden", ()) if isinstance(contacts, Mapping) else ()
    names: list[str] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)):
            continue
        for selector in pair:
            prefix, separator, raw_name = str(selector).partition(":")
            if separator and prefix == "object" and raw_name in manifest.objects:
                names.append(raw_name)
    return tuple(dict.fromkeys(names))


def _clearance_adjusted_waypoint_points(
    manifest: ResolvedEvaluation,
    requested_waypoints: Any,
    robot: RobotCapability | None,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[str, ...]]:
    """Choose safe command subtargets inside authored waypoint regions.

    A waypoint region describes a set, not an obligation to steer at its
    center.  When that center lies within a forbidden object's collision
    envelope, center tracking can teach a policy to graze the obstacle while
    still satisfying the abstract region predicate.  Move the command target
    away from the nearest conflicting object, but keep it inside both the
    authored region and the goal tolerance.  The clearance comes from the
    selected robot capability's geometry, so this generalizes to quadrupeds,
    humanoids, arms, and future embodiments without robot/task-name branches.
    """
    raw = _resolved_waypoint_points(manifest, requested_waypoints)
    if (
        robot is None
        or not isinstance(requested_waypoints, (list, tuple))
        or not requested_waypoints
    ):
        return raw, ()
    forbidden_names = _forbidden_object_names(manifest)
    if not forbidden_names:
        return raw, ()

    goal = manifest.task_shared.get("goal", {})
    success = goal.get("success", {}) if isinstance(goal, Mapping) else {}
    try:
        tolerance = max(0.0, float(success.get("tolerance_m", 0.0))) or 0.25
    except (TypeError, ValueError):
        tolerance = 0.25
    reach = max(0.0, float(robot.geometry.reach_radius_m))
    clearance_margin_m = 0.05
    adjusted = list(raw)
    notes: list[str] = []

    for index, waypoint_name in enumerate(requested_waypoints):
        name = str(waypoint_name)
        zone = manifest.zones.get(name)
        if not isinstance(zone, Mapping) or zone.get("kind") != "disk":
            continue
        try:
            zone_radius = max(0.0, float(zone.get("radius_m", 0.0)))
        except (TypeError, ValueError):
            continue
        if zone_radius <= 0.0:
            continue
        center = adjusted[index]
        conflicts: list[tuple[float, str, tuple[float, float], float]] = []
        for object_name in forbidden_names:
            record = manifest.objects[object_name]
            nominal = record.get("nominal", record)
            pose = nominal.get("pose", {}) if isinstance(nominal, Mapping) else {}
            position = tuple(float(value) for value in pose.get("position_m", ()))
            if len(position) < 2:
                continue
            delta = (center[0] - position[0], center[1] - position[1])
            distance = math.hypot(*delta)
            required = (
                reach + _object_horizontal_radius(record) + clearance_margin_m
            )
            deficit = required - distance
            if deficit > 1e-6 and distance > 1e-6:
                conflicts.append((deficit, object_name, delta, distance))
        if not conflicts:
            continue

        deficit, object_name, delta, distance = max(conflicts)
        # Staying within 80% of the predicate tolerance makes the adjusted
        # command point itself a valid waypoint completion state.  The region
        # radius provides an independent geometric cap.
        max_shift = min(0.9 * zone_radius, 0.8 * tolerance)
        shift = min(deficit, max_shift)
        if shift <= 1e-6:
            continue
        unit = (delta[0] / distance, delta[1] / distance)
        adjusted[index] = (
            center[0] + unit[0] * shift,
            center[1] + unit[1] * shift,
            center[2],
        )
        notes.append(
            f"waypoint {name!r} shifted {shift:.3f} m inside its region "
            f"for {reach:.3f} m embodiment reach clearance from forbidden "
            f"object {object_name!r}"
        )
    return tuple(adjusted), tuple(notes)


def _reconcile_waypoint_course(
    env_cfg: Any,
    manifest: ResolvedEvaluation,
    *,
    train: bool,
    robot: RobotCapability | None = None,
) -> tuple[str, ...]:
    """Align resets and velocity commands with an authored waypoint course.

    Generic velocity tasks otherwise randomize spawn yaw over 360 degrees and
    issue commands unrelated to the authored route.  Resets stay aligned with
    the course entrance, while compatible velocity-command terms are replaced
    by an embodiment-neutral goal-conditioned command that points at the active
    waypoint and becomes zero after completion. Detect only declared
    goal/config semantics; no robot or registered task identifier participates.
    """
    goal = manifest.task_shared.get("goal", {})
    requested_waypoints = goal.get("waypoints")
    has_explicit_route = (
        isinstance(requested_waypoints, (list, tuple))
        and bool(requested_waypoints)
    )
    if (
        goal.get("type") != "waypoint_sequence"
        or not (manifest.course or has_explicit_route)
    ):
        return ()

    raw_waypoints, clearance_adjustments = (
        _clearance_adjusted_waypoint_points(
            manifest, requested_waypoints, robot)
    )
    try:
        goal_tolerance = max(
            0.0, float(goal.get("success", {}).get("tolerance_m", 0.0))
        ) or 0.25
    except (TypeError, ValueError):
        goal_tolerance = 0.25
    # A shifted target uses the authored region's outer side to preserve
    # obstacle clearance.  Reusing the broad task predicate radius would
    # advance the velocity command before the robot reaches that safe side,
    # causing it to cut diagonally back across the forbidden object.  Keep
    # ordinary routes unchanged; only clearance-adjusted routes get a tighter
    # command transition.  The task predicate itself remains frozen.
    command_tolerance = goal_tolerance
    if clearance_adjustments:
        command_tolerance = min(
            goal_tolerance, max(0.08, 0.4 * goal_tolerance))
    # A command that must enter a tight clearance subtarget cannot retain the
    # base route's 35%-of-cruise speed floor: at the transition boundary that
    # floor can carry the robot past the target faster than the controller can
    # turn.  Let the ordinary distance ramp slow naturally for adjusted
    # routes.  Its positive tolerance still guarantees non-zero crossing
    # speed, while unadjusted routes preserve the established 0.35 floor.
    intermediate_min_speed_scale = 0.10 if clearance_adjustments else 0.35
    adjustments: list[str] = []
    adjustments.extend(
        f"command:forbidden-contact clearance→{item}"
        for item in clearance_adjustments
    )
    if clearance_adjustments:
        adjustments.append(
            "command:forbidden-contact clearance→transition radius "
            f"{command_tolerance:.3f} m (task predicate remains "
            f"{goal_tolerance:.3f} m)"
        )
        adjustments.append(
            "command:forbidden-contact clearance→intermediate approach "
            f"speed floor {intermediate_min_speed_scale:.2f}x cruise"
        )
    events = getattr(env_cfg, "events", None)
    if isinstance(events, dict):
        for name, term in tuple(events.items()):
            params = getattr(term, "params", None)
            pose_range = params.get("pose_range") if isinstance(params, dict) else None
            if not isinstance(pose_range, dict) or not all(
                    axis in pose_range for axis in ("x", "y", "yaw")):
                continue
            asset_cfg = params.get("asset_cfg")
            if (
                asset_cfg is not None
                and str(getattr(asset_cfg, "name", "")) != "robot"
            ):
                continue
            pose_range["x"] = (-0.10, 0.05) if train else (0.0, 0.0)
            pose_range["y"] = (-0.08, 0.08) if train else (0.0, 0.0)
            pose_range["yaw"] = (-0.08, 0.08) if train else (0.0, 0.0)
            adjustments.append(f"event:{name}→aligned with course +X")
        if train:
            from mjlab.managers.event_manager import EventTermCfg

            events["world_route_state_initialization"] = EventTermCfg(
                mode="reset",
                func=reset_robot_along_waypoint_route,
                params={
                    "waypoints_m": raw_waypoints,
                    "midroute_probability": 0.5,
                    "approach_distance_m": (0.25, 0.55),
                    "lateral_jitter_m": 0.12,
                    "asset_name": "robot",
                },
            )
            adjustments.append(
                "event:world_route_state_initialization→50% entrance / "
                "50% collision-local route starts (train only)"
            )

    commands = getattr(env_cfg, "commands", None)
    if isinstance(commands, dict):
        for name, term in commands.items():
            ranges = getattr(term, "ranges", None)
            if not all(hasattr(ranges, field) for field in (
                    "lin_vel_x", "lin_vel_y", "ang_vel_z")):
                continue
            WaypointVelocityCommandCfg, _ = _waypoint_velocity_command_types()
            ranges = copy.deepcopy(ranges)
            ranges.lin_vel_x = (-1.0, 1.0)
            ranges.lin_vel_y = (-1.0, 1.0)
            ranges.ang_vel_z = (-1.5, 1.5)
            if hasattr(ranges, "heading"):
                ranges.heading = None
            commands[name] = WaypointVelocityCommandCfg(
                entity_name=str(getattr(term, "entity_name", "robot")),
                resampling_time_range=(1_000.0, 1_000.0),
                debug_vis=bool(getattr(term, "debug_vis", False)),
                heading_command=False,
                heading_control_stiffness=1.0,
                rel_standing_envs=0.0,
                rel_heading_envs=0.0,
                rel_world_envs=0.0,
                rel_forward_envs=0.0,
                init_velocity_prob=0.0,
                ranges=ranges,
                waypoints_m=raw_waypoints,
                tolerance_m=command_tolerance,
                cruise_speed_mps=0.8,
                intermediate_min_speed_scale=intermediate_min_speed_scale,
                terminal_slow_radius_m=2.0,
            )
            adjustments.append(
                f"command:{name}→goal-conditioned waypoint traversal "
                f"with terminal braking")

    curriculum = getattr(env_cfg, "curriculum", None)
    if isinstance(curriculum, dict):
        for name, term in tuple(curriculum.items()):
            func_name = str(getattr(getattr(term, "func", None), "__name__", ""))
            if str(name) == "command_vel" or func_name == "commands_vel":
                curriculum.pop(name, None)
                adjustments.append(
                    f"curriculum:{name}→removed(goal-conditioned route)")
    return tuple(adjustments)


def apply_world_selection(
    env_cfg: Any, selection_path: Path | str, *, train: bool,
    runtime_task_id: str | None = None,
) -> ResolvedWorldBundle:
    """Hash-verify one selected tuple and apply it once before EnvSpec.

    Asset paths in the evaluation manifest are relative to the manifest JSON's
    directory, allowing remote execution to mirror the complete ``env`` tree.
    Evaluation currently verifies the frozen assets and reuses the manifest's
    resolved geometry.  The runner adapter may replace the generator terrain
    with :class:`MaterializedTerrainEntity` to avoid all regeneration.
    """
    path = Path(selection_path)
    project_dir = path.parent.parent
    store = WorldArtifactStore(project_dir)
    selection = store.read_selection(path)
    if selection is None:
        raise WorldCompileError(f"world selection is missing: {path}")

    def read_ref(kind: str) -> tuple[dict[str, Any], Path]:
        ref_path = store.resolve_ref(selection.refs[kind])
        return json.loads(ref_path.read_text(encoding="utf-8")), ref_path

    world, _ = read_ref("world")
    task, _ = read_ref("task")
    manifest_raw, manifest_path = read_ref("resolved_eval")
    catalog_raw, _ = read_ref("channel_catalog")
    manifest = ResolvedEvaluation.from_dict(manifest_raw)
    catalog = ChannelCatalog.from_dict(catalog_raw)
    manifest = verify_resolved_evaluation(
        world, task, catalog, manifest, asset_base=manifest_path.parent)
    if (manifest.runtime_task_id is not None
            and runtime_task_id != manifest.runtime_task_id):
        raise WorldCompileError(
            "runtime task does not match admitted environment: "
            f"{runtime_task_id!r} != {manifest.runtime_task_id!r}")
    if _hash_mapping(world) != manifest.world_hash:
        raise WorldCompileError("selected WorldSpec does not match evaluation manifest")
    if _hash_mapping(task) != manifest.task_hash:
        raise WorldCompileError("selected TaskSpec does not match evaluation manifest")
    if catalog.catalog_hash != manifest.channel_catalog_hash:
        raise WorldCompileError("selected channel catalog does not match manifest")
    if not bool(manifest.admission.get("ok", False)):
        raise WorldCompileError(
            "selected evaluation manifest has not passed admission gates")
    for record in manifest.materialized_assets.get("files", ()):
        asset = manifest_path.parent / record["path"]
        if not asset.is_file() or _file_hash(asset) != record["sha256"]:
            raise WorldCompileError(
                f"materialized evaluation asset hash mismatch: {asset}")
    mjb_relative = manifest.materialized_assets.get("evaluation_mjb")
    if mjb_relative:
        mjb_path = manifest_path.parent / str(mjb_relative)
        expected = manifest.materialized_assets.get("evaluation_mjb_sha256")
        if not mjb_path.is_file() or (expected and _file_hash(mjb_path) != expected):
            raise WorldCompileError(
                f"materialized evaluation model hash mismatch: {mjb_path}")

    robot_data = world["shared"]["robot"]
    robot = resolve_robot_capability(
        robot_data["capability_id"],
        required=robot_data.get("required_capabilities", ()),
        extra_paths=([robot_data["descriptor_path"]]
                     if robot_data.get("descriptor_path") else ()))
    runtime_robot_hash = _runtime_robot_hash(env_cfg)
    expected_robot_hash = manifest.robot_asset_hash.removeprefix("sha256:")
    if runtime_robot_hash is None:
        raise WorldCompileError(
            "runtime environment has no compilable 'robot' entity")
    if runtime_robot_hash != expected_robot_hash:
        raise WorldCompileError(
            "runtime robot asset does not match evaluation manifest")

    world_dr_applied: tuple[str, ...] = ()
    if train:
        compiled = compile_world(world, task)
        expand_train_terrain_difficulty(compiled, world)
        apply_compiled_world(env_cfg, compiled)
        # Per-episode domain randomization: the authored `train.variations`
        # (box heights, object mass/friction) become mjlab reset events so each
        # env re-samples its layout every episode — the sim-to-real robustness
        # lever. Train-only; evaluation always replays the frozen manifest.
        from sculptor.world.randomization import install_world_randomizations
        world_dr_applied = tuple(install_world_randomizations(env_cfg, world))
    else:
        authored_scene, _, _ = compile_scene_cfg(world)
        runtime = compile_task_runtime(world, task, robot)
        terrain = manifest.terrain
        if terrain.get("kind") == "generator":
            relative_xml = manifest.materialized_assets.get("terrain_xml")
            if not relative_xml:
                raise WorldCompileError(
                    "generated evaluation terrain has no materialized XML")
            install_materialized_terrain_factory()
            cfg_type, _ = materialized_terrain_types()
            authored_scene.terrain = cfg_type(
                terrain_type="generator", terrain_generator=None,
                terrain_xml_path=str(manifest_path.parent / relative_xml),
                terrain_origins_m=tuple(terrain.get("origins_m", ())),
                frozen_flat_patches=terrain.get("flat_patches", {}),
                frozen_flat_patch_radii_m=terrain.get(
                    "flat_patch_radii_m", {}))
        _apply_scene_and_runtime(
            env_cfg,
            authored_scene,
            runtime,
            zones=manifest.zones,
            robot=robot,
        )
    # Entity init poses are authored in environment-local coordinates.  mjlab
    # only applies replicated env origins to fixed mocap and floating entities
    # through an explicit reset event.  Install it for BOTH train and eval so
    # physical objects occupy the same frame as commands, zones, and metrics.
    from sculptor.world.randomization import install_authored_object_resets
    object_placement_applied = tuple(install_authored_object_resets(
        env_cfg,
        manifest.objects,
        world=world,
        train=train,
    ))
    runtime_adjustments = (
        *_reconcile_terrain_curriculum(env_cfg),
        *_reconcile_waypoint_course(
            env_cfg, manifest, train=train, robot=robot),
        *(f"physical scene alignment → {msg}"
          for msg in object_placement_applied),
        *(f"per-episode domain randomization → {msg}"
          for msg in world_dr_applied),
    )
    return ResolvedWorldBundle(
        tuple_hash=selection.tuple_hash,
        evaluation_lineage=selection.evaluation_lineage,
        manifest=manifest, channel_catalog=catalog, train=train,
        refs={kind: dataclasses.asdict(ref)
              for kind, ref in selection.refs.items()},
        runtime_robot_asset_hash=runtime_robot_hash,
        runtime_adjustments=runtime_adjustments)
