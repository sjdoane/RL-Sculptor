"""Simulator-side producers for a compiled :mod:`ChannelCatalog`.

The runtime is intentionally driven by channel ``producer`` and ``source``
metadata.  It never dispatches on a task prompt, task name, or robot model.
That makes the same object/region/contact machinery usable by locomotion,
mobile-manipulation, and fixed-base arm configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sculptor.world.channels import (
    ChannelCatalog,
    ChannelSpec,
    validate_trajectory_channels,
)


class WorldRuntimeError(RuntimeError):
    """A declared authored-world channel could not be produced exactly."""


def _to_numpy(value: Any, *, dtype: str | None = None) -> np.ndarray:
    if value is None:
        raise WorldRuntimeError("simulator value is missing")
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        pass
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise WorldRuntimeError(f"could not transfer simulator value: {exc}") from exc
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(np.dtype(dtype), copy=False)
    return array


def _first_attr(value: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    raise WorldRuntimeError(
        f"none of the required simulator fields exist: {', '.join(names)}")


def _scene_item(scene: Any, name: str) -> Any:
    try:
        return scene[name]
    except (KeyError, TypeError) as exc:
        raise WorldRuntimeError(f"scene entity/sensor {name!r} is missing") from exc


def _center3(zone: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(zone["center_m"], dtype=np.float32)
    if center.shape == (2,):
        center = np.concatenate((center, np.zeros(1, dtype=np.float32)))
    if center.shape != (3,):
        raise WorldRuntimeError(f"zone center must have 2 or 3 values, got {center}")
    return center


def _signed_distance_to_zone(
    position: np.ndarray, zone: Mapping[str, Any],
) -> np.ndarray:
    """Return negative-inside signed distance for disk and box regions."""
    center = _center3(zone)
    if zone["kind"] == "disk":
        return np.linalg.norm(position[..., :2] - center[:2], axis=-1) - float(
            zone["radius_m"])
    if zone["kind"] == "box":
        half = np.asarray(zone["size_m"], dtype=np.float32) / 2.0
        q = np.abs(position - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside
    raise WorldRuntimeError(f"unsupported zone kind {zone.get('kind')!r}")


@dataclass(frozen=True)
class RuntimeSnapshot:
    """One timestep of catalogued arrays, split by access policy."""

    channels: Mapping[str, np.ndarray]
    reward_info: Mapping[str, np.ndarray]


class WorldChannelRuntime:
    """Produce exact per-step arrays from an instantiated mjlab environment."""

    _ENTITY_FIELDS = {
        "pos_w": ("root_link_pos_w", "root_pos_w", "body_pos_w"),
        "quat_w": ("root_link_quat_w", "root_quat_w", "body_quat_w"),
        "lin_vel_w": (
            "root_link_lin_vel_w", "root_lin_vel_w", "body_lin_vel_w"),
        "ang_vel_w": (
            "root_link_ang_vel_w", "root_ang_vel_w", "body_ang_vel_w"),
    }

    def __init__(
        self, env: Any, *, catalog: ChannelCatalog,
        manifest: Any,
    ) -> None:
        self.env = env
        self.catalog = catalog
        self.manifest = manifest
        self.scene = env.scene
        self.num_envs = int(getattr(env, "num_envs", 0))
        if self.num_envs <= 0:
            raise WorldRuntimeError("environment num_envs must be positive")
        self.step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        if not np.isfinite(self.step_dt) or self.step_dt <= 0:
            raise WorldRuntimeError("environment step_dt must be positive")
        self.task = dict(manifest.task_shared)
        self.goal = dict(self.task.get("goal", {}))
        self.zones = dict(manifest.zones)
        self.objects = dict(manifest.objects)
        self._hold_elapsed = np.zeros(self.num_envs, dtype=np.float32)
        self._waypoint_index = np.zeros(self.num_envs, dtype=np.int32)
        self._last_predicate: np.ndarray | None = None
        self._last_waypoint_distance = np.zeros(self.num_envs, dtype=np.float32)
        self._last_waypoint_complete = np.zeros(self.num_envs, dtype=bool)
        self._waypoints = self._resolve_waypoints()

    def reset(self, env_ids: Any) -> None:
        """Reset private temporal state for simulator-reset environments.

        Replicated simulators auto-reset only the environments that terminate.
        Keeping route progress or hold time across that boundary contaminates
        later episodes and can make recorder output disagree with simulator
        state.  Reset exactly the requested rows so unaffected environments
        retain their in-progress temporal state.
        """
        ids = _to_numpy(env_ids).astype(np.int64, copy=False).reshape(-1)
        if ids.size == 0:
            return
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise WorldRuntimeError(
                "reset env_ids must be within [0, num_envs)"
            )
        self._hold_elapsed[ids] = 0.0
        self._waypoint_index[ids] = 0
        self._last_waypoint_distance[ids] = 0.0
        self._last_waypoint_complete[ids] = False
        if self._last_predicate is not None:
            self._last_predicate[ids] = False

    def _entity_state(self, entity_name: str, suffix: str) -> np.ndarray:
        entity = _scene_item(self.scene, entity_name)
        data = getattr(entity, "data", None)
        if data is None:
            raise WorldRuntimeError(f"scene entity {entity_name!r} has no data")
        raw = _first_attr(data, self._ENTITY_FIELDS[suffix])
        array = _to_numpy(raw)
        # Articulated/free objects expose their root as (N, 3/4).  A backend
        # may expose a one-body axis; resolve only that unambiguous case.
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0]
        return array

    def _robot_position(self) -> np.ndarray:
        return self._entity_state("robot", "pos_w")

    def _object_position(self, name: str) -> np.ndarray:
        return self._entity_state(name, "pos_w")

    def _local_position(self, position: np.ndarray) -> np.ndarray:
        """Translate world-space state into each environment's local frame."""
        raw_origins = getattr(self.scene, "env_origins", None)
        # Single-world backends and lightweight adapter/test scenes may not
        # expose replicated-environment origins.  Their world frame already
        # is the local frame, so the mathematically correct origin is zero.
        origins = (
            np.zeros((self.num_envs, 3), dtype=np.float32)
            if raw_origins is None
            else _to_numpy(raw_origins, dtype=np.float32)
        )
        if origins.shape != (self.num_envs, 3):
            raise WorldRuntimeError(
                f"scene env_origins has shape {origins.shape}, expected "
                f"({self.num_envs}, 3)")
        return position - origins

    def _goal_tolerance(self) -> float:
        success = self.goal.get("success", {})
        return float(success.get("tolerance_m", 0.0))

    def _region_predicate(self, entity: str, region: str) -> np.ndarray:
        position = (
            self._robot_position() if entity == "robot"
            else self._object_position(entity)
        )
        position = self._local_position(position)
        signed = _signed_distance_to_zone(position, self.zones[region])
        return signed <= self._goal_tolerance()

    def _region_distance(self, entity: str, region: str) -> np.ndarray:
        position = (
            self._robot_position() if entity == "robot"
            else self._object_position(entity)
        )
        position = self._local_position(position)
        signed = _signed_distance_to_zone(position, self.zones[region])
        return np.maximum(signed - self._goal_tolerance(), 0.0)

    def _resolve_waypoints(self) -> np.ndarray:
        if self.goal.get("type") != "waypoint_sequence":
            return np.empty((0, 3), dtype=np.float32)
        requested = self.goal.get("waypoints", "auto")
        course = list(self.manifest.course)
        points: list[np.ndarray] = []
        if requested == "auto":
            for primitive in course:
                points.append(np.asarray(primitive.position_m, dtype=np.float32))
        else:
            by_source: dict[str, Any] = {}
            for primitive in course:
                by_source[str(primitive.source_id)] = primitive
            for name in requested:
                if name in self.zones:
                    points.append(_center3(self.zones[name]))
                elif name in by_source:
                    points.append(np.asarray(
                        by_source[name].position_m, dtype=np.float32))
                else:  # schema/compiler should have caught this first
                    raise WorldRuntimeError(f"waypoint {name!r} is unresolved")
        if not points:
            raise WorldRuntimeError("waypoint_sequence compiled no waypoints")
        return np.stack(points).astype(np.float32, copy=False)

    def _update_waypoints(self) -> None:
        if self._waypoints.size == 0:
            return
        position = self._local_position(self._robot_position())
        last = len(self._waypoints) - 1
        target = self._waypoints[np.minimum(self._waypoint_index, last)]
        # Course waypoints are traversal gates, not root-height targets. XY
        # distance forces the robot through each platform footprint while the
        # collidable geometry itself enforces climbing/jumping onto its top.
        distance = np.linalg.norm(position[..., :2] - target[..., :2], axis=-1)
        tolerance = max(0.0, self._goal_tolerance()) or 0.25
        reached = (self._waypoint_index < len(self._waypoints)) & (
            distance <= tolerance)
        self._waypoint_index = np.minimum(
            self._waypoint_index + reached.astype(np.int32),
            len(self._waypoints),
        )
        complete = self._waypoint_index >= len(self._waypoints)
        target = self._waypoints[np.minimum(self._waypoint_index, last)]
        distance = np.linalg.norm(position[..., :2] - target[..., :2], axis=-1)
        self._last_waypoint_distance = np.where(complete, 0.0, distance)
        self._last_waypoint_complete = complete

    def _contact(self, source: Mapping[str, Any]) -> np.ndarray:
        name = (
            f"authored_contact__{source['group']}__{int(source.get('index', 0))}")
        sensor = _scene_item(self.scene, name)
        data = getattr(sensor, "data", None)
        found = getattr(data, "found", None)
        array = _to_numpy(found)
        if array.ndim > 1:
            array = np.any(array > 0, axis=tuple(range(1, array.ndim)))
        return array.astype(bool, copy=False)

    def _configuration_error(self) -> np.ndarray:
        target = self.goal.get("target")
        if isinstance(target, Mapping):
            target = target.get("joint_pos")
        if not isinstance(target, (list, tuple)) or not target:
            raise WorldRuntimeError(
                "configuration_distribution goal requires target joint_pos")
        robot = _scene_item(self.scene, "robot")
        actual = _to_numpy(_first_attr(robot.data, ("joint_pos",)))
        desired = np.asarray(target, dtype=np.float32)
        if actual.shape[-1] != desired.shape[0]:
            raise WorldRuntimeError(
                "configuration target length does not match robot joint state")
        return np.sqrt(np.mean(np.square(actual - desired), axis=-1))

    def _velocity_error(self, subject: str) -> np.ndarray:
        velocity = self._entity_state(subject, "lin_vel_w")
        target = self.goal.get("target")
        threshold = float(self.goal.get("success", {}).get("threshold", 0.0))
        if isinstance(target, (list, tuple)):
            desired = np.asarray(target, dtype=np.float32)
            return np.linalg.norm(velocity - desired, axis=-1)
        speed = np.linalg.norm(velocity, axis=-1)
        desired_speed = float(target) if isinstance(target, (int, float)) else threshold
        return np.abs(speed - desired_speed)

    def _base_predicate(self) -> np.ndarray:
        goal_type = self.goal.get("type")
        if goal_type == "object_to_region":
            return self._region_predicate(
                str(self.goal["subject"]), str(self.goal["region"]))
        if goal_type == "robot_to_region":
            return self._region_predicate("robot", str(self.goal["region"]))
        if goal_type == "waypoint_sequence":
            return self._last_waypoint_complete
        if goal_type == "object_velocity":
            subject = str(self.goal["subject"])
            velocity = self._entity_state(subject, "lin_vel_w")
            speed = np.linalg.norm(velocity, axis=-1)
            threshold = float(self.goal.get("success", {}).get("threshold", 0.0))
            return speed >= threshold
        if goal_type == "configuration_distribution":
            threshold = float(self.goal.get("success", {}).get("threshold", 0.05))
            return self._configuration_error() <= threshold
        raise WorldRuntimeError(f"unsupported goal type {goal_type!r}")

    def _update_success(self) -> np.ndarray:
        predicate = self._base_predicate().astype(bool, copy=False)
        self._last_predicate = predicate
        self._hold_elapsed = np.where(
            predicate, self._hold_elapsed + self.step_dt, 0.0).astype(
                np.float32, copy=False)
        hold_s = float(self.goal.get("success", {}).get("hold_s", 0.0))
        return predicate & (self._hold_elapsed + 1e-7 >= hold_s)

    def _produce(
        self, spec: ChannelSpec, *, success: np.ndarray,
    ) -> np.ndarray:
        source = spec.source
        producer = spec.producer
        if producer == "entity_state":
            suffix = next((key for key in self._ENTITY_FIELDS
                           if spec.name.endswith(f"__{key}")), None)
            if suffix is None:
                raise WorldRuntimeError(
                    f"channel {spec.name!r} has unknown entity-state suffix")
            value = self._entity_state(str(source["entity"]), suffix)
        elif producer == "region_relative":
            center = _center3(self.zones[str(source["region"])])
            value = center - self._local_position(self._robot_position())
        elif producer == "object_region_distance":
            value = self._region_distance(
                str(source["object"]), str(source["region"]))
        elif producer == "robot_region_distance":
            value = self._region_distance("robot", str(source["region"]))
        elif producer in {"object_region_predicate", "robot_region_predicate"}:
            entity = "robot" if producer.startswith("robot") else str(source["object"])
            value = self._region_predicate(entity, str(source["region"]))
        elif producer == "success_hold":
            value = success
        elif producer == "contact_pair":
            value = self._contact(source)
        elif producer == "object_velocity_error":
            value = self._velocity_error(str(source["object"]))
        elif producer == "waypoint_distance":
            value = self._last_waypoint_distance
        elif producer == "waypoint_state":
            value = self._waypoint_index
        elif producer == "configuration_error":
            value = self._configuration_error()
        else:
            raise WorldRuntimeError(
                f"channel {spec.name!r} declares unsupported producer {producer!r}")
        array = _to_numpy(value, dtype=spec.dtype)
        expected_rank = len(spec.shape) - 1
        if array.ndim != expected_rank or array.shape[0] != self.num_envs:
            raise WorldRuntimeError(
                f"channel {spec.name!r} produced shape {array.shape}; "
                f"expected one-step rank {expected_rank} with N={self.num_envs}")
        for index, expected in enumerate(spec.shape[1:]):
            if isinstance(expected, int) and array.shape[index] != expected:
                raise WorldRuntimeError(
                    f"channel {spec.name!r} dimension {index} is "
                    f"{array.shape[index]}, expected {expected}")
        return array

    def sample(self) -> RuntimeSnapshot:
        self._update_waypoints()
        success = self._update_success()
        values = {
            spec.name: self._produce(spec, success=success)
            for spec in self.catalog.channels
        }
        # This is the actual metric firewall.  Reward code is never handed
        # metric_only arrays, even though the recorder persists all channels.
        reward_names = self.catalog.names(reward=True)
        reward_info = {name: value for name, value in values.items()
                       if name in reward_names}
        return RuntimeSnapshot(channels=values, reward_info=reward_info)


class TorchWorldRewardRuntime:
    """GPU-native producer for the reward-visible catalog subset.

    This mirrors the recorder's physical observables but deliberately has no
    API for requesting ``metric_only`` channels.  Keeping the split at object
    construction prevents generated reward code from reaching authoritative
    completion truth through an accidental dictionary merge.
    """

    def __init__(self, env: Any, *, catalog: ChannelCatalog, manifest: Any) -> None:
        import torch

        self.env = env
        self.catalog = catalog
        self.manifest = manifest
        self.scene = env.scene
        self.torch = torch
        self.task = dict(manifest.task_shared)
        self.goal = dict(self.task.get("goal", {}))
        self.zones = dict(manifest.zones)
        self.specs = tuple(
            spec for spec in catalog.channels
            if spec.access in {"base", "shared_shaping"})
        forbidden = {spec.name for spec in catalog.channels
                     if spec.access == "metric_only"}
        if forbidden & {spec.name for spec in self.specs}:  # defensive
            raise WorldRuntimeError("metric-only reward channel exposure")
        self._waypoints = self._resolve_waypoints()
        self._waypoint_index = torch.zeros(
            int(env.num_envs), dtype=torch.long, device=env.device)

    def _entity_state(self, name: str, suffix: str) -> Any:
        entity = _scene_item(self.scene, name)
        value = _first_attr(entity.data, WorldChannelRuntime._ENTITY_FIELDS[suffix])
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        return value

    def _robot_position(self) -> Any:
        return self._entity_state("robot", "pos_w")

    def _local_position(self, position: Any) -> Any:
        raw_origins = getattr(self.scene, "env_origins", None)
        origins = (
            self.torch.zeros_like(position)
            if raw_origins is None
            else raw_origins.to(device=position.device, dtype=position.dtype)
        )
        if origins.shape != position.shape:
            raise WorldRuntimeError(
                f"scene env_origins has shape {tuple(origins.shape)}, "
                f"expected {tuple(position.shape)}")
        return position - origins

    def _zone_center(self, name: str, *, dtype: Any) -> Any:
        return self.torch.as_tensor(
            _center3(self.zones[name]), device=self.env.device, dtype=dtype)

    def _signed_region_distance(self, entity: str, region: str) -> Any:
        position = (
            self._robot_position() if entity == "robot"
            else self._entity_state(entity, "pos_w"))
        position = self._local_position(position)
        zone = self.zones[region]
        center = self._zone_center(region, dtype=position.dtype)
        if zone["kind"] == "disk":
            return self.torch.linalg.norm(
                position[..., :2] - center[:2], dim=-1) - float(zone["radius_m"])
        half = self.torch.as_tensor(
            zone["size_m"], device=position.device, dtype=position.dtype) / 2.0
        q = self.torch.abs(position - center) - half
        outside = self.torch.linalg.norm(self.torch.clamp_min(q, 0.0), dim=-1)
        inside = self.torch.minimum(
            self.torch.max(q, dim=-1).values,
            self.torch.zeros((), device=position.device, dtype=position.dtype))
        return outside + inside

    def _resolve_waypoints(self) -> Any:
        if self.goal.get("type") != "waypoint_sequence":
            return None
        requested = self.goal.get("waypoints", "auto")
        course = list(self.manifest.course)
        raw: list[np.ndarray] = []
        if requested == "auto":
            raw = [np.asarray(item.position_m, dtype=np.float32) for item in course]
        else:
            by_source = {str(item.source_id): item for item in course}
            for name in requested:
                if name in self.zones:
                    raw.append(_center3(self.zones[name]))
                else:
                    raw.append(np.asarray(by_source[name].position_m, dtype=np.float32))
        if not raw:
            raise WorldRuntimeError("waypoint_sequence compiled no waypoints")
        return self.torch.as_tensor(
            np.stack(raw), device=self.env.device, dtype=self.torch.float32)

    def _waypoint_distance(self) -> Any:
        if self._waypoints is None:
            raise WorldRuntimeError("waypoint producer used for a non-waypoint goal")
        position = self._local_position(self._robot_position())
        last = self._waypoints.shape[0] - 1
        target = self._waypoints[
            self.torch.clamp(self._waypoint_index, max=last)]
        distance = self.torch.linalg.norm(
            position[..., :2] - target[..., :2], dim=-1)
        tolerance = max(0.0, float(
            self.goal.get("success", {}).get("tolerance_m", 0.0))) or 0.25
        reached = (
            (self._waypoint_index < self._waypoints.shape[0])
            & (distance <= tolerance))
        self._waypoint_index = self.torch.clamp(
            self._waypoint_index + reached.long(), max=self._waypoints.shape[0])
        complete = self._waypoint_index >= self._waypoints.shape[0]
        target = self._waypoints[
            self.torch.clamp(self._waypoint_index, max=last)]
        distance = self.torch.linalg.norm(
            position[..., :2] - target[..., :2], dim=-1)
        return self.torch.where(complete, self.torch.zeros_like(distance), distance)

    def _contact(self, source: Mapping[str, Any]) -> Any:
        sensor = _scene_item(
            self.scene,
            f"authored_contact__{source['group']}__{int(source['index'])}")
        found = sensor.data.found
        if found.ndim > 1:
            found = self.torch.any(found > 0, dim=tuple(range(1, found.ndim)))
        return found.bool()

    def _configuration_error(self) -> Any:
        target = self.goal.get("target")
        if isinstance(target, Mapping):
            target = target.get("joint_pos")
        robot = _scene_item(self.scene, "robot")
        actual = robot.data.joint_pos
        desired = self.torch.as_tensor(
            target, device=actual.device, dtype=actual.dtype)
        if desired.ndim != 1 or desired.shape[0] != actual.shape[-1]:
            raise WorldRuntimeError("configuration target does not match joint state")
        return self.torch.sqrt(self.torch.mean((actual - desired) ** 2, dim=-1))

    def sample(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        waypoint_distance = None
        if any(spec.producer == "waypoint_distance" for spec in self.specs):
            waypoint_distance = self._waypoint_distance()
        tolerance = float(self.goal.get("success", {}).get("tolerance_m", 0.0))
        for spec in self.specs:
            source = spec.source
            producer = spec.producer
            if producer == "entity_state":
                suffix = next((key for key in WorldChannelRuntime._ENTITY_FIELDS
                               if spec.name.endswith(f"__{key}")), None)
                if suffix is None:
                    raise WorldRuntimeError(f"unknown entity channel {spec.name!r}")
                value = self._entity_state(str(source["entity"]), suffix)
            elif producer == "region_relative":
                position = self._local_position(self._robot_position())
                value = self._zone_center(
                    str(source["region"]), dtype=position.dtype) - position
            elif producer in {"object_region_distance", "robot_region_distance"}:
                entity = "robot" if producer.startswith("robot") else str(
                    source["object"])
                value = self.torch.clamp_min(
                    self._signed_region_distance(entity, str(source["region"]))
                    - tolerance, 0.0)
            elif producer == "contact_pair":
                value = self._contact(source)
            elif producer == "object_velocity_error":
                velocity = self._entity_state(str(source["object"]), "lin_vel_w")
                target = self.goal.get("target")
                if isinstance(target, (list, tuple)):
                    desired = self.torch.as_tensor(
                        target, device=velocity.device, dtype=velocity.dtype)
                    value = self.torch.linalg.norm(velocity - desired, dim=-1)
                else:
                    desired_speed = float(target) if isinstance(
                        target, (int, float)) else float(
                            self.goal.get("success", {}).get("threshold", 0.0))
                    value = self.torch.abs(
                        self.torch.linalg.norm(velocity, dim=-1) - desired_speed)
            elif producer == "waypoint_distance":
                value = waypoint_distance
            elif producer == "configuration_error":
                value = self._configuration_error()
            else:
                # Every predicate/success/index producer is metric_only.  If
                # a future catalog marks one reward-visible, fail closed.
                raise WorldRuntimeError(
                    f"reward channel {spec.name!r} has unsupported producer "
                    f"{producer!r}")
            if value.shape[0] != self.env.num_envs:
                raise WorldRuntimeError(
                    f"reward channel {spec.name!r} has invalid N axis")
            values[spec.name] = value
        return values

    def reset(self, env_ids: Any) -> None:
        self._waypoint_index[env_ids] = 0


class WorldChannelRecorder:
    """Bounded in-memory recorder for catalogued rollout channels."""

    def __init__(self, runtime: WorldChannelRuntime) -> None:
        self.runtime = runtime
        self._buffers: dict[str, list[np.ndarray]] = {
            spec.name: [] for spec in runtime.catalog.channels}

    def append(self) -> RuntimeSnapshot:
        snapshot = self.runtime.sample()
        for name, value in snapshot.channels.items():
            self._buffers[name].append(value)
        return snapshot

    def finalize(self) -> dict[str, np.ndarray]:
        if not self._buffers or not all(self._buffers.values()):
            raise WorldRuntimeError("no authored-world channel samples recorded")
        arrays = {
            name: np.stack(values, axis=0)
            for name, values in self._buffers.items()
        }
        errors = validate_trajectory_channels(
            arrays, self.runtime.catalog,
            catalog_hash=self.runtime.catalog.catalog_hash,
            strict_unknown=True, require_all=True,
        )
        if errors:
            raise WorldRuntimeError(
                "recorded channel contract failed: " + "; ".join(errors))
        arrays["channel_catalog_hash"] = np.asarray(
            self.runtime.catalog.catalog_hash)
        return arrays
