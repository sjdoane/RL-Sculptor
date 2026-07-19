"""Robot and simulator capability descriptors.

Compiler behavior is driven by these descriptors, never by task-name
substrings. Descriptors are immutable data and can be extended from project
JSON without adding task-specific branches to the compiler.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class CapabilityError(ValueError):
    """A requested semantic role or capability is not installed."""


@dataclass(frozen=True)
class RobotGeometry:
    standing_height_m: float
    leg_length_m: float
    foot_length_m: float
    reach_radius_m: float
    max_step_height_m: float
    max_gap_m: float


@dataclass(frozen=True)
class RobotCapability:
    capability_id: str
    asset_id: str
    asset_hash: str
    root_body: str
    body_roles: dict[str, tuple[str, ...]]
    site_roles: dict[str, tuple[str, ...]]
    capabilities: frozenset[str]
    geometry: RobotGeometry
    supported_commands: frozenset[str]
    supported_observations: frozenset[str]
    reward_state_schema: dict[str, tuple[int, ...]] = field(
        default_factory=dict)
    reward_state_sources: dict[str, str] = field(default_factory=dict)
    reward_info_keys: tuple[str, ...] = ()
    contact_capacity: int = 24
    augmentation: str | None = None
    asset_factory: str | None = None

    def resolve_role(self, role: str) -> tuple[str, ...]:
        names = self.body_roles.get(role, ())
        if not names:
            raise CapabilityError(
                f"{self.capability_id}: robot role {role!r} is unavailable")
        return names

    def resolve_site_role(self, role: str) -> tuple[str, ...]:
        """Resolve a semantic point role without coupling to asset names."""
        names = self.site_roles.get(role, ())
        if not names:
            raise CapabilityError(
                f"{self.capability_id}: robot site role {role!r} is "
                "unavailable")
        return names

    def resolve_semantic_role(
        self, role: str,
    ) -> tuple[str, tuple[str, ...]]:
        """Resolve a role to its simulator namespace and concrete names.

        Body roles take precedence because they can participate in contacts.
        Site roles describe points such as a tool center or grasp frame.  This
        keeps world/task compilation generic across legged robots, arms, and
        future descriptors without guessing from robot or task names.
        """
        if role in self.body_roles:
            return "body", self.resolve_role(role)
        if role in self.site_roles:
            return "site", self.resolve_site_role(role)
        raise CapabilityError(
            f"{self.capability_id}: robot semantic role {role!r} is "
            "unavailable")

    def require(self, required: Iterable[str]) -> None:
        missing = sorted(set(required) - set(self.capabilities))
        if missing:
            raise CapabilityError(
                f"{self.capability_id}: missing required capabilities "
                f"{missing}; installed={sorted(self.capabilities)}")

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["capabilities"] = sorted(self.capabilities)
        data["supported_commands"] = sorted(self.supported_commands)
        data["supported_observations"] = sorted(self.supported_observations)
        data["body_roles"] = {k: list(v) for k, v in self.body_roles.items()}
        data["site_roles"] = {k: list(v) for k, v in self.site_roles.items()}
        data["reward_state_schema"] = {
            k: list(v) for k, v in self.reward_state_schema.items()}
        data["reward_state_sources"] = dict(self.reward_state_sources)
        data["reward_info_keys"] = list(self.reward_info_keys)
        return data


def build_robot_entity_cfg(capability: RobotCapability) -> Any:
    """Instantiate the robot asset declared by a capability descriptor."""
    target = capability.asset_factory
    if not target or ":" not in target:
        raise CapabilityError(
            f"{capability.capability_id}: descriptor has no importable "
            "asset_factory; exact composed admission is unavailable")
    module_name, attribute = target.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        cfg = factory()
    except Exception as exc:
        raise CapabilityError(
            f"{capability.capability_id}: asset_factory {target!r} failed: "
            f"{exc}") from exc
    if not hasattr(cfg, "build"):
        raise CapabilityError(
            f"{capability.capability_id}: asset_factory {target!r} did not "
            "return an EntityCfg")
    return cfg


@dataclass(frozen=True)
class TerrainCapability:
    schema_type: str
    mjlab_class: str
    parameter_bounds: dict[str, tuple[float, float]] = field(
        default_factory=dict)
    required_parameters: frozenset[str] = frozenset()
    allowed_parameters: frozenset[str] = frozenset()

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(self.parameter_bounds) | self.required_parameters \
            | self.allowed_parameters


@dataclass(frozen=True)
class SimulatorBudget:
    max_geoms: int = 4096
    max_contacts: int = 256
    max_rays: int = 512
    max_constraints: int = 256
    max_heightfield_texels: int = 4_000_000
    max_rollout_channel_bytes: int = 128_000_000


@dataclass(frozen=True)
class SimulatorCapability:
    capability_id: str
    compiler_version: str
    dependency_versions: dict[str, str]
    terrains: dict[str, TerrainCapability]
    entity_shapes: frozenset[str]
    observation_adapters: frozenset[str]
    command_adapters: frozenset[str]
    sensor_types: frozenset[str]
    mutable_model_fields: frozenset[str]
    budget: SimulatorBudget = SimulatorBudget()

    def terrain(self, schema_type: str) -> TerrainCapability:
        try:
            return self.terrains[schema_type]
        except KeyError as exc:
            raise CapabilityError(
                f"{self.capability_id}: terrain type {schema_type!r} is "
                f"unavailable; installed={sorted(self.terrains)}") from exc

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        for terrain in data["terrains"].values():
            terrain["required_parameters"] = sorted(
                terrain["required_parameters"])
            terrain["allowed_parameters"] = sorted(
                terrain["allowed_parameters"])
        for key in ("entity_shapes", "observation_adapters",
                    "command_adapters", "sensor_types",
                    "mutable_model_fields"):
            data[key] = sorted(getattr(self, key))
        return data


def _stable_hash(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _builtin_robots() -> dict[str, RobotCapability]:
    g1_base = RobotCapability(
        capability_id="unitree_g1:base",
        asset_id="unitree_g1",
        asset_hash="mjlab:g1",
        asset_factory="mjlab.asset_zoo.robots:get_g1_robot_cfg",
        root_body="pelvis",
        body_roles={
            "pelvis": ("pelvis",), "torso": ("torso_link",),
            "left_foot": ("left_ankle_roll_link",),
            "right_foot": ("right_ankle_roll_link",),
            "left_end_effector": ("left_wrist_yaw_link",),
            "right_end_effector": ("right_wrist_yaw_link",),
        },
        site_roles={
            "left_foot": ("left_foot",), "right_foot": ("right_foot",),
        },
        capabilities=frozenset(
            {"locomotion", "jump", "kick", "push", "whole_body"}),
        geometry=RobotGeometry(
            standing_height_m=1.28, leg_length_m=0.70, foot_length_m=0.27,
            reach_radius_m=0.75, max_step_height_m=0.55, max_gap_m=0.80),
        supported_commands=frozenset(
            {"base_velocity", "robot_to_region", "waypoint_heading"}),
        supported_observations=frozenset(
            {"proprioception", "height_scan", "object_relative",
             "region_relative"}),
        reward_state_schema={
            "qpos": (29,), "qvel": (29,), "base_lin_vel_b": (3,),
            "base_ang_vel_b": (3,), "projected_gravity_b": (3,),
            "actuator_force": (29,), "command_vel": (3,),
        },
        reward_info_keys=(
            "left_foot_contact", "right_foot_contact",
            "left_foot_swing_speed", "right_foot_swing_speed",
            "left_foot_height", "right_foot_height",
            "base_horizontal_speed",
        ),
    )
    go1 = RobotCapability(
        capability_id="unitree_go1:base", asset_id="unitree_go1",
        asset_hash="mjlab:go1", root_body="trunk",
        asset_factory="mjlab.asset_zoo.robots:get_go1_robot_cfg",
        body_roles={
            "torso": ("trunk",),
            "front_left_foot": ("FL_calf",),
            "front_right_foot": ("FR_calf",),
            "rear_left_foot": ("RL_calf",),
            "rear_right_foot": ("RR_calf",),
            "feet": ("FL_calf", "FR_calf", "RL_calf", "RR_calf"),
        },
        site_roles={},
        capabilities=frozenset({"locomotion", "jump", "push"}),
        geometry=RobotGeometry(
            standing_height_m=0.32, leg_length_m=0.22, foot_length_m=0.06,
            reach_radius_m=0.35, max_step_height_m=0.28, max_gap_m=0.55),
        supported_commands=frozenset(
            {"base_velocity", "robot_to_region", "waypoint_heading"}),
        supported_observations=frozenset(
            {"proprioception", "height_scan", "object_relative",
             "region_relative"}),
        reward_state_schema={
            "qpos": (12,), "qvel": (12,), "base_lin_vel_b": (3,),
            "base_ang_vel_b": (3,), "projected_gravity_b": (3,),
            "actuator_force": (12,), "command_vel": (3,),
        },
    )
    yam = RobotCapability(
        capability_id="yam:parallel_gripper", asset_id="yam",
        asset_hash="mjlab:yam", root_body="arm",
        asset_factory="mjlab.asset_zoo.robots:get_yam_robot_cfg",
        body_roles={
            "base": ("arm",), "wrist": ("link_6",),
            "left_finger": ("link_left_finger", "lf_rot", "lf_down"),
            "right_finger": ("link_right_finger", "rf_rot", "rf_down"),
            "gripper": ("lf_down", "rf_down"),
            "end_effector": ("link_6",),
        },
        site_roles={
            "tool_center": ("tcp_site",), "grasp": ("grasp_site",),
        },
        capabilities=frozenset(
            {"manipulation", "grasp", "push", "lift"}),
        geometry=RobotGeometry(
            standing_height_m=0.0, leg_length_m=0.0, foot_length_m=0.0,
            reach_radius_m=0.75, max_step_height_m=0.0, max_gap_m=0.0),
        supported_commands=frozenset(
            {"end_effector_pose", "object_target", "robot_to_region"}),
        supported_observations=frozenset(
            {"proprioception", "object_relative", "region_relative",
             "end_effector_relative"}),
        reward_state_schema={
            "qpos": (8,), "qvel": (8,), "actuator_force": (7,),
            "end_effector_pos_w": (3,),
        },
        reward_state_sources={
            "end_effector_pos_w": "site:tool_center",
        },
        contact_capacity=32,
    )
    # Do not advertise a synthetic G1 gripper variant here. A capability is
    # installed only after its asset augmentation, actuators, action scaling,
    # collision roles, and reset keyframe compile successfully. Stock mjlab
    # G1 has fixed rubber hands and cannot honestly satisfy `grasp`.
    return {c.capability_id: c for c in (g1_base, go1, yam)}


def _load_robot_file(path: Path) -> RobotCapability:
    data = json.loads(path.read_text(encoding="utf-8"))
    geometry = RobotGeometry(**data.pop("geometry"))
    return RobotCapability(
        geometry=geometry,
        body_roles={k: tuple(v) for k, v in data.pop("body_roles").items()},
        site_roles={k: tuple(v) for k, v in
                    data.pop("site_roles", {}).items()},
        capabilities=frozenset(data.pop("capabilities", [])),
        supported_commands=frozenset(data.pop("supported_commands", [])),
        supported_observations=frozenset(
            data.pop("supported_observations", [])),
        reward_state_schema={
            k: tuple(v) for k, v in
            data.pop("reward_state_schema", {}).items()},
        reward_state_sources={
            str(k): str(v) for k, v in
            data.pop("reward_state_sources", {}).items()},
        reward_info_keys=tuple(data.pop("reward_info_keys", [])),
        **data,
    )


def robot_capabilities(
    extra_paths: Iterable[Path | str] = (),
) -> dict[str, RobotCapability]:
    registry = _builtin_robots()
    for raw_path in extra_paths:
        cap = _load_robot_file(Path(raw_path))
        registry[cap.capability_id] = cap
    return registry


def resolve_robot_capability(
    capability_id: str, *, required: Iterable[str] = (),
    extra_paths: Iterable[Path | str] = (),
) -> RobotCapability:
    registry = robot_capabilities(extra_paths)
    try:
        cap = registry[capability_id]
    except KeyError as exc:
        raise CapabilityError(
            f"unknown robot capability {capability_id!r}; "
            f"installed={sorted(registry)}") from exc
    cap.require(required)
    return cap


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def simulator_capability() -> SimulatorCapability:
    terrains = {
        "flat": TerrainCapability("flat", "BoxFlatTerrainCfg"),
        "hf_random_uniform": TerrainCapability(
            "hf_random_uniform", "HfRandomUniformTerrainCfg",
            {"noise_range_m": (0.0, 0.35), "noise_step_m": (0.001, 0.10)},
            frozenset({"noise_range_m"}),
            frozenset({"downsampled_scale_m"})),
        "hf_pyramid_sloped": TerrainCapability(
            "hf_pyramid_sloped", "HfPyramidSlopedTerrainCfg",
            {"slope_deg": (0.0, 45.0), "platform_width_m": (0.2, 5.0)},
            frozenset({"slope_deg"}), frozenset({"inverted"})),
        "hf_wave": TerrainCapability(
            "hf_wave", "HfWaveTerrainCfg",
            {"amplitude_m": (0.0, 0.5), "num_waves": (1.0, 20.0)},
            frozenset({"amplitude_m"})),
        "hf_discrete_obstacles": TerrainCapability(
            "hf_discrete_obstacles", "HfDiscreteObstaclesTerrainCfg"),
        "hf_perlin_noise": TerrainCapability(
            "hf_perlin_noise", "HfPerlinNoiseTerrainCfg"),
        "box_random_spread": TerrainCapability(
            "box_random_spread", "BoxRandomSpreadTerrainCfg",
            {"num_boxes": (1.0, 200.0), "box_height_m": (0.0, 1.5)}),
        "box_random_grid": TerrainCapability(
            "box_random_grid", "BoxRandomGridTerrainCfg"),
        "box_random_stairs": TerrainCapability(
            "box_random_stairs", "BoxRandomStairsTerrainCfg"),
        "box_pyramid_stairs": TerrainCapability(
            "box_pyramid_stairs", "BoxPyramidStairsTerrainCfg"),
        "box_open_stairs": TerrainCapability(
            "box_open_stairs", "BoxOpenStairsTerrainCfg"),
        "box_stepping_stones": TerrainCapability(
            "box_stepping_stones", "BoxSteppingStonesTerrainCfg"),
        "box_narrow_beams": TerrainCapability(
            "box_narrow_beams", "BoxNarrowBeamsTerrainCfg"),
        "box_tilted_grid": TerrainCapability(
            "box_tilted_grid", "BoxTiltedGridTerrainCfg"),
        "box_nested_rings": TerrainCapability(
            "box_nested_rings", "BoxNestedRingsTerrainCfg"),
    }
    deps = {name: _dependency_version(name) for name in
            ("mjlab", "mujoco", "mujoco-warp")}
    descriptor = SimulatorCapability(
        capability_id="mjlab:1", compiler_version="world-compiler-v1",
        dependency_versions=deps, terrains=terrains,
        entity_shapes=frozenset(
            {"sphere", "box", "cylinder", "capsule", "frame"}),
        observation_adapters=frozenset(
            {"proprioception", "height_scan", "object_relative",
             "region_relative", "end_effector_relative"}),
        command_adapters=frozenset(
            {"base_velocity", "robot_to_region", "waypoint_heading",
             "end_effector_pose", "object_target"}),
        sensor_types=frozenset({"contact", "height", "entity_state"}),
        mutable_model_fields=frozenset(
            {"mass_kg", "friction", "restitution", "size_m", "radius_m"}),
    )
    return dataclasses.replace(
        descriptor, compiler_version="world-compiler-v1:" +
        _stable_hash(descriptor.to_dict())[:12])
