"""Compiled trajectory/reward tensor contract for authored worlds."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from sculptor.world.artifacts import canonical_json_bytes, sha256_bytes

ChannelAccess = Literal["base", "shared_shaping", "metric_only"]
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:__[a-z0-9_]+)*$")
_DTYPES = {"float32", "float64", "int32", "int64", "bool"}
_SHAPE_SYMBOLS = {"T", "N", "J"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# Keep the existing metric surface as a stable compatibility layer.  An
# authored ChannelCatalog extends this tuple; it never replaces or broadens it
# implicitly.  This lives here (rather than importing generated_metric) so the
# world compiler and the metric runtime share one source without an import
# cycle.
BASE_METRIC_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "projected_gravity_b",
    "root_link_pos_w",
    "left_foot_contact",
    "right_foot_contact",
    "left_foot_pos_b",
    "right_foot_pos_b",
)

_CATALOG_KEYS = {
    "catalog_version", "world_hash", "task_hash", "channels", "catalog_hash",
}
_CHANNEL_KEYS = {
    "name", "dtype", "shape", "producer", "access", "metric_role",
    "max_bytes_per_rollout", "source",
}


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    dtype: str
    shape: tuple[str | int, ...]
    producer: str
    access: ChannelAccess
    metric_role: str
    max_bytes_per_rollout: int
    source: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["shape"] = list(self.shape)
        return data


@dataclass(frozen=True)
class ChannelCatalog:
    catalog_version: int
    world_hash: str
    task_hash: str
    channels: tuple[ChannelSpec, ...]
    catalog_hash: str

    @classmethod
    def build(
        cls, *, world_hash: str, task_hash: str,
        channels: Iterable[ChannelSpec],
    ) -> "ChannelCatalog":
        ordered = tuple(sorted(channels, key=lambda item: item.name))
        payload = {
            "catalog_version": 1, "world_hash": world_hash,
            "task_hash": task_hash,
            "channels": [item.to_dict() for item in ordered],
        }
        return cls(
            catalog_version=1, world_hash=world_hash, task_hash=task_hash,
            channels=ordered,
            catalog_hash=sha256_bytes(canonical_json_bytes(payload)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelCatalog":
        unknown = set(value) - _CATALOG_KEYS
        missing = _CATALOG_KEYS - set(value)
        if unknown or missing:
            raise ValueError(
                "invalid channel catalog keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}")
        if value.get("catalog_version") != 1:
            raise ValueError(
                f"unsupported channel catalog version "
                f"{value.get('catalog_version')!r}")
        for key in ("world_hash", "task_hash", "catalog_hash"):
            if not isinstance(value.get(key), str):
                raise ValueError(f"channel catalog {key} must be a string")
        raw_channels = value.get("channels")
        if not isinstance(raw_channels, list):
            raise ValueError("channel catalog channels must be a list")
        for index, item in enumerate(raw_channels):
            if not isinstance(item, Mapping):
                raise ValueError(f"channels[{index}] must be an object")
            item_unknown = set(item) - _CHANNEL_KEYS
            item_missing = (_CHANNEL_KEYS - {"source"}) - set(item)
            if item_unknown or item_missing:
                raise ValueError(
                    f"invalid channels[{index}] keys: "
                    f"missing={sorted(item_missing)}, "
                    f"unknown={sorted(item_unknown)}")
            if not isinstance(item.get("shape"), list):
                raise ValueError(f"channels[{index}].shape must be a list")
            for key in ("name", "dtype", "producer", "access", "metric_role"):
                if not isinstance(item.get(key), str):
                    raise ValueError(f"channels[{index}].{key} must be a string")
            byte_budget = item.get("max_bytes_per_rollout")
            if (not isinstance(byte_budget, int)
                    or isinstance(byte_budget, bool)):
                raise ValueError(
                    f"channels[{index}].max_bytes_per_rollout must be an integer")
            if "source" in item and not isinstance(item["source"], Mapping):
                raise ValueError(f"channels[{index}].source must be an object")
        channels = tuple(ChannelSpec(
            name=item["name"], dtype=item["dtype"],
            shape=tuple(item["shape"]), producer=item["producer"],
            access=item["access"], metric_role=item["metric_role"],
            max_bytes_per_rollout=int(item["max_bytes_per_rollout"]),
            source=dict(item.get("source", {})),
        ) for item in value["channels"])
        catalog = cls.build(
            world_hash=str(value["world_hash"]),
            task_hash=str(value["task_hash"]), channels=channels)
        supplied = value["catalog_hash"]
        if not isinstance(supplied, str) or not _HASH_RE.fullmatch(supplied):
            raise ValueError("catalog_hash must be a lowercase sha256")
        if supplied != catalog.catalog_hash:
            raise ValueError(
                f"channel catalog hash mismatch: {supplied} != "
                f"{catalog.catalog_hash}")
        return catalog

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "world_hash": self.world_hash, "task_hash": self.task_hash,
            "channels": [item.to_dict() for item in self.channels],
            "catalog_hash": self.catalog_hash,
        }

    def names(self, *, reward: bool = False) -> set[str]:
        if reward:
            return {c.name for c in self.channels
                    if c.access in {"base", "shared_shaping"}}
        return {c.name for c in self.channels}

    def allowed_metric_arrays(self) -> tuple[str, ...]:
        """The exact metric array allowlist for this project.

        Base arrays remain available for legacy/body-state metrics.  Project
        arrays include both shared-shaping and metric-only signals because the
        objective metric is the held-out consumer of the latter.
        """
        return tuple(dict.fromkeys((*BASE_METRIC_ARRAYS, *sorted(self.names()))))

    def by_name(self) -> dict[str, ChannelSpec]:
        return {c.name: c for c in self.channels}


def validate_channel_catalog(
    catalog: ChannelCatalog, *, max_total_bytes: int = 128_000_000,
) -> list[str]:
    errors: list[str] = []
    if catalog.catalog_version != 1:
        errors.append(
            f"unsupported catalog_version {catalog.catalog_version!r}")
    for label, value in (
        ("world_hash", catalog.world_hash), ("task_hash", catalog.task_hash),
    ):
        if not _HASH_RE.fullmatch(value):
            errors.append(f"{label}: expected lowercase sha256")
    seen: set[str] = set()
    total = 0
    for channel in catalog.channels:
        path = f"channels.{channel.name}"
        if not _NAME_RE.fullmatch(channel.name):
            errors.append(f"{path}: invalid channel name")
        if channel.name in seen:
            errors.append(f"{path}: duplicate")
        seen.add(channel.name)
        if channel.dtype not in _DTYPES:
            errors.append(f"{path}.dtype: unsupported {channel.dtype!r}")
        if not channel.shape or any(
                not (isinstance(x, int) and not isinstance(x, bool) and x > 0)
                and x not in _SHAPE_SYMBOLS
                for x in channel.shape):
            errors.append(f"{path}.shape: invalid dimensions {channel.shape}")
        if not isinstance(channel.producer, str) or not channel.producer:
            errors.append(f"{path}.producer: must be a non-empty string")
        if not isinstance(channel.metric_role, str) or not channel.metric_role:
            errors.append(f"{path}.metric_role: must be a non-empty string")
        if not isinstance(channel.source, dict):
            errors.append(f"{path}.source: must be an object")
        if channel.access not in {"base", "shared_shaping", "metric_only"}:
            errors.append(f"{path}.access: unsupported {channel.access!r}")
        if channel.name in BASE_METRIC_ARRAYS:
            errors.append(
                f"{path}: project catalog may not redeclare base metric array")
        if (isinstance(channel.max_bytes_per_rollout, bool)
                or channel.max_bytes_per_rollout <= 0):
            errors.append(f"{path}.max_bytes_per_rollout: must be > 0")
        total += max(0, channel.max_bytes_per_rollout)
    if total > max_total_bytes:
        errors.append(
            f"channel budget {total} exceeds max_total_bytes {max_total_bytes}")
    rebuilt = ChannelCatalog.build(
        world_hash=catalog.world_hash, task_hash=catalog.task_hash,
        channels=catalog.channels)
    if rebuilt.catalog_hash != catalog.catalog_hash:
        errors.append("catalog_hash does not match canonical content")
    return errors


def _channel(
    name: str, shape: tuple[str | int, ...], producer: str,
    access: ChannelAccess, role: str, source: dict[str, Any],
    *, bytes_: int = 2_400_000, dtype: str = "float32",
) -> ChannelSpec:
    return ChannelSpec(
        name=name, dtype=dtype, shape=shape, producer=producer,
        access=access, metric_role=role,
        max_bytes_per_rollout=bytes_, source=dict(source))


def compile_channel_catalog(
    world: dict[str, Any], task: dict[str, Any],
) -> ChannelCatalog:
    world_hash = sha256_bytes(canonical_json_bytes(world))
    task_hash = sha256_bytes(canonical_json_bytes(task))
    channels: list[ChannelSpec] = []
    observations = task.get("shared", {}).get("observations", {})
    objects = world.get("shared", {}).get("objects", {})
    zones = world.get("shared", {}).get("zones", {})
    for name in sorted(objects):
        source = {"entity": name}
        channels.extend([
            _channel(f"object__{name}__pos_w", ("T", "N", 3),
                     "entity_state", "shared_shaping", "state", source),
            _channel(f"object__{name}__quat_w", ("T", "N", 4),
                     "entity_state", "shared_shaping", "state", source),
            _channel(f"object__{name}__lin_vel_w", ("T", "N", 3),
                     "entity_state", "shared_shaping", "state", source),
            _channel(f"object__{name}__ang_vel_w", ("T", "N", 3),
                     "entity_state", "shared_shaping", "state", source),
        ])
    for zone_name in sorted(zones):
        if zone_name in observations.get("region_relative", []):
            channels.append(_channel(
                f"region__{zone_name}__relative", ("T", "N", 3),
                "region_relative", "shared_shaping", "progress",
                {"region": zone_name}))
    goal = task.get("shared", {}).get("goal", {})
    goal_id = str(goal.get("id", "goal"))
    if goal.get("type") == "object_to_region":
        subject = str(goal["subject"])
        region = str(goal["region"])
        source = {"object": subject, "region": region, "goal": goal_id}
        channels.extend([
            _channel(
                f"object__{subject}__to_region__{region}__distance",
                ("T", "N"), "object_region_distance", "shared_shaping",
                "progress", source),
            _channel(
                f"goal__{goal_id}__inside", ("T", "N"),
                "object_region_predicate", "metric_only", "completion",
                source, dtype="bool"),
            _channel(
                f"goal__{goal_id}__success", ("T", "N"),
                "success_hold", "metric_only", "completion", source,
                dtype="bool"),
        ])
    elif goal.get("type") == "object_velocity":
        subject = str(goal["subject"])
        source = {"object": subject, "goal": goal_id}
        channels.extend([
            _channel(
                f"goal__{goal_id}__velocity_error", ("T", "N"),
                "object_velocity_error", "shared_shaping", "progress", source),
            _channel(
                f"goal__{goal_id}__success", ("T", "N"),
                "success_hold", "metric_only", "completion", source,
                dtype="bool"),
        ])
    elif goal.get("type") == "robot_to_region":
        region = str(goal["region"])
        source = {"region": region, "goal": goal_id}
        channels.extend([
            _channel(
                f"robot__to_region__{region}__distance", ("T", "N"),
                "robot_region_distance", "shared_shaping", "progress", source),
            _channel(
                f"goal__{goal_id}__inside", ("T", "N"),
                "robot_region_predicate", "metric_only", "completion", source,
                dtype="bool"),
            _channel(
                f"goal__{goal_id}__success", ("T", "N"),
                "success_hold", "metric_only", "completion", source,
                dtype="bool"),
        ])
    elif goal.get("type") == "waypoint_sequence":
        source = {"goal": goal_id}
        channels.extend([
            _channel(
                f"goal__{goal_id}__waypoint_distance", ("T", "N"),
                "waypoint_distance", "shared_shaping", "progress", source),
            _channel(
                f"goal__{goal_id}__waypoint_index", ("T", "N"),
                "waypoint_state", "metric_only", "progress", source,
                dtype="int32"),
            _channel(
                f"goal__{goal_id}__success", ("T", "N"),
                "success_hold", "metric_only", "completion", source,
                dtype="bool"),
        ])
    elif goal.get("type") == "configuration_distribution":
        source = {"goal": goal_id}
        channels.extend([
            _channel(
                f"goal__{goal_id}__configuration_error", ("T", "N"),
                "configuration_error", "shared_shaping", "progress", source),
            _channel(
                f"goal__{goal_id}__success", ("T", "N"),
                "success_hold", "metric_only", "completion", source,
                dtype="bool"),
        ])
    contacts = task.get("shared", {}).get("contacts", {})
    for group in ("desired", "forbidden", "terminate_on"):
        for index, pair in enumerate(contacts.get(group, [])):
            channels.append(_channel(
                f"contact__{group}__{index}", ("T", "N"),
                "contact_pair",
                "shared_shaping" if group == "desired" else "metric_only",
                "contact", {
                    "group": group, "index": index,
                    "selectors": list(pair),
                },
                dtype="bool"))
    catalog = ChannelCatalog.build(
        world_hash=world_hash, task_hash=task_hash, channels=channels)
    errors = validate_channel_catalog(catalog)
    if errors:
        raise ValueError("invalid ChannelCatalog:\n- " + "\n- ".join(errors))
    return catalog


def load_channel_catalog(path: Path | str) -> ChannelCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    catalog = ChannelCatalog.from_dict(raw)
    errors = validate_channel_catalog(catalog)
    if errors:
        raise ValueError("invalid ChannelCatalog:\n- " + "\n- ".join(errors))
    return catalog


def resolve_channel_catalog(
    value: ChannelCatalog | Mapping[str, Any] | Path | str | None,
) -> ChannelCatalog | None:
    """Normalize a public catalog argument and enforce its canonical hash."""
    if value is None:
        return None
    if isinstance(value, ChannelCatalog):
        errors = validate_channel_catalog(value)
        if errors:
            raise ValueError("invalid ChannelCatalog:\n- " + "\n- ".join(errors))
        return value
    if isinstance(value, Mapping):
        catalog = ChannelCatalog.from_dict(value)
        errors = validate_channel_catalog(catalog)
        if errors:
            raise ValueError("invalid ChannelCatalog:\n- " + "\n- ".join(errors))
        return catalog
    return load_channel_catalog(value)


def load_project_channel_catalog(
    project_dir: Path | str,
) -> ChannelCatalog | None:
    """Load the authoritative catalog for an authored project, if present.

    Absence keeps legacy projects on their historical metric surface.  Once a
    selection exists, malformed refs or hashes fail closed instead of silently
    widening the generated-metric allowlist.
    """
    from sculptor.world.artifacts import WorldArtifactStore

    project = Path(project_dir).expanduser().resolve()
    selection_path = project / "env" / "selection_current.json"
    if not selection_path.is_file():
        return None
    store = WorldArtifactStore(project)
    selection = store.read_selection(selection_path)
    if selection is None:
        raise ValueError(
            f"authored world selection is unreadable: {selection_path}")
    if "channel_catalog" not in selection.refs:
        raise ValueError(
            "authored world selection has no channel_catalog ref")
    return resolve_channel_catalog(
        store.load_json_ref(selection.refs["channel_catalog"]))


def validate_trajectory_channels(
    arrays: Mapping[str, Any], catalog: ChannelCatalog, *,
    catalog_hash: str | None = None, strict_unknown: bool = True,
    require_all: bool = True,
    max_total_bytes: int = 128_000_000,
) -> list[str]:
    """Validate an in-memory/NPZ mapping against the exact compiled catalog."""
    errors: list[str] = []
    if catalog_hash is not None and catalog_hash != catalog.catalog_hash:
        errors.append(
            f"trajectory catalog hash {catalog_hash} != {catalog.catalog_hash}")
    declared = catalog.by_name()
    symbol_sizes: dict[str, int] = {}
    actual_total_bytes = 0
    if strict_unknown:
        unknown = set(arrays) - set(declared) - {"channel_catalog_hash"}
        if unknown:
            errors.append(f"trajectory contains undeclared arrays {sorted(unknown)}")
    for name, spec in declared.items():
        if name not in arrays and require_all:
            errors.append(f"trajectory missing declared array {name!r}")
            continue
        if name not in arrays:
            continue
        array = arrays[name]
        actual_total_bytes += int(getattr(array, "nbytes", 0))
        dtype_name = str(getattr(array, "dtype", ""))
        if dtype_name and dtype_name != spec.dtype:
            errors.append(
                f"{name}: dtype {dtype_name} does not match {spec.dtype}")
        shape = tuple(getattr(array, "shape", ()))
        if len(shape) != len(spec.shape):
            errors.append(f"{name}: rank {len(shape)} != {len(spec.shape)}")
            continue
        for index, expected in enumerate(spec.shape):
            if isinstance(expected, int) and shape[index] != expected:
                errors.append(
                    f"{name}: shape[{index}]={shape[index]} != {expected}")
            elif isinstance(expected, str):
                actual = shape[index]
                if actual <= 0:
                    errors.append(
                        f"{name}: symbolic dimension {expected} must be > 0")
                elif expected in symbol_sizes and symbol_sizes[expected] != actual:
                    errors.append(
                        f"{name}: symbolic dimension {expected}={actual} does "
                        f"not match {symbol_sizes[expected]}")
                else:
                    symbol_sizes.setdefault(expected, actual)
        if int(getattr(array, "nbytes", 0)) > spec.max_bytes_per_rollout:
            errors.append(
                f"{name}: byte budget exceeded "
                f"({array.nbytes} > {spec.max_bytes_per_rollout})")
    if actual_total_bytes > max_total_bytes:
        errors.append(
            f"trajectory channel bytes {actual_total_bytes} exceed total budget "
            f"{max_total_bytes}")
    return errors


def catalog_fixture_arrays(
    catalog: ChannelCatalog,
    *,
    time_steps: int,
    num_envs: int,
    case: Literal[
        "far_idle", "edge_camping", "contact_flicker",
        "forbidden_contact", "competent",
    ],
) -> dict[str, Any]:
    """Deterministic semantic fixtures for catalog-aware metric gates.

    These are deliberately task-channel fixtures, not physics rollouts.  They
    make the firewall test the important failure modes encoded by the catalog:
    distance-only edge camping, transient predicate/contact flicker, forbidden
    contact, and a held successful completion.  They are also used to populate
    task-derived calibration ladders with the exact declared array surface.
    """
    import numpy as np

    if time_steps <= 0 or num_envs <= 0:
        raise ValueError("fixture dimensions must be positive")
    out: dict[str, Any] = {}
    shape_values = {"T": time_steps, "N": num_envs, "J": 1}
    for spec in catalog.channels:
        shape = tuple(shape_values.get(dim, dim) for dim in spec.shape)
        dtype = np.dtype(spec.dtype)
        arr = np.zeros(shape, dtype=dtype)
        producer = spec.producer
        role = spec.metric_role

        if producer == "entity_state":
            if spec.name.endswith("__quat_w") and arr.shape[-1:] == (4,):
                arr[..., 0] = 1.0
            elif spec.name.endswith("__pos_w") and arr.shape[-1:] == (3,):
                arr[..., 0] = 1.0 if case == "far_idle" else 0.0
            elif "vel" in spec.name and case == "competent":
                # The object is settled after task completion; velocity is not
                # a proxy for success.
                arr[...] = 0
        elif producer in {"region_relative", "object_region_distance",
                          "robot_region_distance", "waypoint_distance",
                          "object_velocity_error", "configuration_error"}:
            distance = {
                "far_idle": 1.0,
                "edge_camping": 0.005,
                "contact_flicker": 0.0,
                "forbidden_contact": 0.4,
                "competent": 0.0,
            }[case]
            if arr.ndim >= 3:
                arr[..., 0] = distance
            else:
                arr[...] = distance
        elif producer in {"object_region_predicate", "robot_region_predicate"}:
            if case == "competent":
                arr[...] = True
            elif case == "contact_flicker":
                arr[::2] = True
        elif producer == "success_hold":
            if case == "competent":
                # Persisted success is the compiler's hold-qualified predicate.
                arr[...] = True
        elif producer == "contact_pair":
            group = str(spec.source.get("group", ""))
            if group == "desired":
                if case == "competent":
                    arr[...] = True
                elif case == "contact_flicker":
                    arr[::2] = True
            elif group in {"forbidden", "terminate_on"}:
                if case == "forbidden_contact":
                    arr[...] = True
        elif producer == "waypoint_state" and case == "competent":
            arr[...] = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
        elif role == "completion" and case == "competent":
            arr[...] = True
        out[spec.name] = arr
    return out
