"""Strict WorldSpec v2 validation and stable variation addressing."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from sculptor.world.capabilities import (
    CapabilityError, SimulatorCapability, resolve_robot_capability,
    simulator_capability,
)

WORLD_SPEC_VERSION = 2
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOP_KEYS = {"world_spec_version", "meta", "shared", "train"}
_META_KEYS = {
    "version", "parent", "source", "prompt", "grounding",
    "parameter_provenance",
}
_SHARED_KEYS = {
    "eval_seed", "robot", "terrain", "obstacles", "objects", "zones",
}
_TRAIN_KEYS = {"variations", "curriculum"}
_COURSE_ELEMENTS = {
    "platform", "gap", "beam", "wall", "stairs", "stepping_stones",
}
_COURSE_NOMINAL_KEYS: dict[str, set[str]] = {
    "platform": {"height_m", "length_m", "width_m", "gap_after_m"},
    "gap": {"length_m", "width_m", "depth_m"},
    "beam": {"width_m", "length_m", "height_m", "gap_after_m"},
    "wall": {"height_m", "width_m", "thickness_m", "gap_after_m"},
    "stairs": {"step_height_m", "step_width_m", "num_steps", "width_m"},
    "stepping_stones": {
        "stone_size_m", "spacing_m", "height_m", "count", "width_m"},
}
_OBJECT_KEYS = {"shape", "fixed", "nominal"}
_OBJECT_NOMINAL_COMMON = {
    "mass_kg", "friction", "restitution", "pose", "rgba",
}
_OBJECT_SHAPE_KEYS = {
    "sphere": {"radius_m"},
    "box": {"size_m"},
    "cylinder": {"radius_m", "height_m"},
    "capsule": {"radius_m", "height_m"},
    "frame": {"opening_m", "post_radius_m", "depth_m"},
}
_ZONE_KEYS = {
    "disk": {"kind", "center_m", "radius_m", "attach"},
    "box": {"kind", "center_m", "size_m", "attach"},
}
_DIST_KEYS = {
    "uniform": {"kind", "low", "high"},
    "normal": {"kind", "mean", "std", "clip"},
    "choice": {"kind", "values"},
    "uniform_in_region": {"kind", "region"},
}


def _is_num(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(float(value)))


def _strict(
    value: Any, path: str, allowed: set[str], errors: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}: must be an object")
        return {}
    unknown = set(value) - allowed
    if unknown:
        errors.append(f"{path}: unknown keys {sorted(unknown, key=str)}")
    return value


def _number(
    value: Any, path: str, errors: list[str], *,
    lo: float | None = None, hi: float | None = None,
) -> None:
    if not _is_num(value):
        errors.append(f"{path}: must be a finite number")
        return
    number = float(value)
    if lo is not None and number < lo:
        errors.append(f"{path}: {number:g} below {lo:g}")
    if hi is not None and number > hi:
        errors.append(f"{path}: {number:g} above {hi:g}")


def _vector(
    value: Any, path: str, errors: list[str], length: int, *,
    positive: bool = False,
) -> None:
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or not all(_is_num(v) for v in value)):
        errors.append(f"{path}: must be {length} finite numbers")
        return
    if positive and any(float(v) <= 0 for v in value):
        errors.append(f"{path}: every value must be > 0")


def _identifier(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        errors.append(f"{path}: must match {_ID_RE.pattern}")
        return False
    return True


def _validate_distribution(
    value: Any, path: str, errors: list[str], *,
    allow_region: bool = False,
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: must be an object")
        return
    kind = value.get("kind")
    if kind not in _DIST_KEYS or (kind == "uniform_in_region"
                                  and not allow_region):
        errors.append(f"{path}.kind: unsupported distribution {kind!r}")
        return
    _strict(value, path, _DIST_KEYS[kind], errors)
    if kind == "uniform":
        _number(value.get("low"), f"{path}.low", errors)
        _number(value.get("high"), f"{path}.high", errors)
        if (_is_num(value.get("low")) and _is_num(value.get("high"))
                and float(value["low"]) > float(value["high"])):
            errors.append(f"{path}: low must be <= high")
    elif kind == "normal":
        _number(value.get("mean"), f"{path}.mean", errors)
        _number(value.get("std"), f"{path}.std", errors, lo=0.0)
        if "clip" in value:
            _vector(value["clip"], f"{path}.clip", errors, 2)
    elif kind == "choice":
        values = value.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"{path}.values: must be a non-empty list")
    elif kind == "uniform_in_region":
        _identifier(value.get("region"), f"{path}.region", errors)


def _validate_meta(meta: Any, errors: list[str]) -> None:
    meta = _strict(meta, "meta", _META_KEYS, errors)
    if "version" in meta and not isinstance(meta["version"], str):
        errors.append("meta.version: must be a string")
    if "parent" in meta and meta["parent"] is not None \
            and not isinstance(meta["parent"], str):
        errors.append("meta.parent: must be a string or null")
    if "source" in meta and meta["source"] not in {
            "generated", "user", "imported", "default", "repaired"}:
        errors.append("meta.source: unsupported source")
    if "prompt" in meta and not isinstance(meta["prompt"], str):
        errors.append("meta.prompt: must be a string")
    grounding = meta.get("grounding", [])
    if not isinstance(grounding, list) or not all(
            isinstance(x, str) for x in grounding):
        errors.append("meta.grounding: must be a list of node IDs")
    provenance = meta.get("parameter_provenance", {})
    if not isinstance(provenance, dict) or not all(
            isinstance(k, str) and v in {"prompt", "user", "default",
                                         "timeout_default", "repair"}
            for k, v in (provenance.items()
                         if isinstance(provenance, dict) else [])):
        errors.append("meta.parameter_provenance: invalid provenance map")


def _validate_robot(robot: Any, errors: list[str]) -> None:
    robot = _strict(
        robot, "shared.robot",
        {"capability_id", "required_capabilities", "descriptor_path"}, errors)
    capability_id = robot.get("capability_id")
    if not isinstance(capability_id, str) or not capability_id:
        errors.append("shared.robot.capability_id: required string")
        return
    required = robot.get("required_capabilities", [])
    if not isinstance(required, list) or not all(
            isinstance(x, str) and x for x in required):
        errors.append(
            "shared.robot.required_capabilities: must be a list of strings")
        required = []
    descriptor_paths = []
    if robot.get("descriptor_path"):
        descriptor_paths = [robot["descriptor_path"]]
    try:
        resolve_robot_capability(
            capability_id, required=required, extra_paths=descriptor_paths)
    except (CapabilityError, OSError, ValueError) as exc:
        errors.append(f"shared.robot: {exc}")


def _validate_terrain(
    terrain: Any, errors: list[str], sim: SimulatorCapability,
) -> None:
    terrain = _strict(
        terrain, "shared.terrain",
        {"kind", "layout", "evaluation_difficulty", "sub_terrains"},
        errors)
    kind = terrain.get("kind")
    if kind not in {"plane", "generator"}:
        errors.append("shared.terrain.kind: must be plane or generator")
        return
    if kind == "plane":
        if "layout" in terrain or "sub_terrains" in terrain:
            errors.append(
                "shared.terrain: plane cannot declare layout/sub_terrains")
        return
    layout = _strict(
        terrain.get("layout"), "shared.terrain.layout",
        {"mode", "rows", "cols", "tile_size_m", "border_width_m"}, errors)
    mode = layout.get("mode")
    if mode not in {"curriculum_grid", "sampled_grid"}:
        errors.append("shared.terrain.layout.mode: unsupported mode")
    rows = layout.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 64:
        errors.append("shared.terrain.layout.rows: integer in [1,64] required")
    if mode == "curriculum_grid" and "cols" in layout:
        errors.append("shared.terrain.layout.cols: forbidden in curriculum_grid")
    if mode == "sampled_grid":
        cols = layout.get("cols")
        if not isinstance(cols, int) or isinstance(cols, bool) or not 1 <= cols <= 64:
            errors.append(
                "shared.terrain.layout.cols: integer in [1,64] required")
    _vector(layout.get("tile_size_m"),
            "shared.terrain.layout.tile_size_m", errors, 2, positive=True)
    _number(layout.get("border_width_m"),
            "shared.terrain.layout.border_width_m", errors, lo=0.0, hi=20.0)
    _number(terrain.get("evaluation_difficulty"),
            "shared.terrain.evaluation_difficulty", errors, lo=0.0, hi=1.0)
    sub = terrain.get("sub_terrains")
    if not isinstance(sub, dict) or not sub:
        errors.append("shared.terrain.sub_terrains: non-empty object required")
        return
    proportions = 0.0
    for name, entry in sub.items():
        _identifier(name, f"shared.terrain.sub_terrains.{name}", errors)
        path = f"shared.terrain.sub_terrains.{name}"
        entry = _strict(entry, path, {"type", "nominal", "proportion"}, errors)
        terrain_type = entry.get("type")
        try:
            capability = sim.terrain(str(terrain_type))
        except CapabilityError as exc:
            errors.append(f"{path}.type: {exc}")
            continue
        nominal = entry.get("nominal")
        if not isinstance(nominal, dict):
            errors.append(f"{path}.nominal: must be an object")
            continue
        unknown = set(nominal) - set(capability.parameter_names)
        if unknown:
            errors.append(
                f"{path}.nominal: unknown keys "
                f"{sorted(unknown, key=str)}")
        missing = capability.required_parameters - set(nominal)
        if missing:
            errors.append(f"{path}.nominal: missing {sorted(missing)}")
        for parameter, bounds in capability.parameter_bounds.items():
            if parameter not in nominal:
                continue
            raw = nominal[parameter]
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            for value in values:
                _number(value, f"{path}.nominal.{parameter}", errors,
                        lo=bounds[0], hi=bounds[1])
        proportion = entry.get("proportion", 1.0)
        _number(proportion, f"{path}.proportion", errors, lo=0.0, hi=1.0)
        if _is_num(proportion):
            proportions += float(proportion)
    if mode == "sampled_grid" and abs(proportions - 1.0) > 1e-6:
        errors.append(
            "shared.terrain.sub_terrains: sampled_grid proportions must sum to 1")


def _validate_obstacles(obstacles: Any, errors: list[str]) -> None:
    obstacles = _strict(
        obstacles, "shared.obstacles",
        {"course", "layout", "waypoints", "start_offset_m"},
        errors)
    if obstacles.get("layout", "linear") not in {"linear"}:
        errors.append("shared.obstacles.layout: only linear is supported")
    if obstacles.get("waypoints", "auto") not in {"auto", "none"}:
        errors.append("shared.obstacles.waypoints: must be auto or none")
    if "start_offset_m" in obstacles:
        _number(
            obstacles["start_offset_m"],
            "shared.obstacles.start_offset_m", errors, lo=0.0, hi=20.0)
    course = obstacles.get("course", [])
    if not isinstance(course, list):
        errors.append("shared.obstacles.course: must be a list")
        return
    seen: set[str] = set()
    for index, item in enumerate(course):
        path = f"shared.obstacles.course[{index}]"
        item = _strict(item, path, {"id", "element", "nominal"}, errors)
        item_id = item.get("id")
        if _identifier(item_id, f"{path}.id", errors):
            if item_id in seen:
                errors.append(f"{path}.id: duplicate stable ID {item_id!r}")
            seen.add(item_id)
        element = item.get("element")
        if element not in _COURSE_ELEMENTS:
            errors.append(f"{path}.element: unsupported {element!r}")
            continue
        nominal = _strict(
            item.get("nominal"), f"{path}.nominal",
            _COURSE_NOMINAL_KEYS[element], errors)
        for key, value in nominal.items():
            if key in {"num_steps", "count"}:
                if not isinstance(value, int) or isinstance(value, bool) \
                        or not 1 <= value <= 100:
                    errors.append(f"{path}.nominal.{key}: integer in [1,100]")
            else:
                _number(value, f"{path}.nominal.{key}", errors,
                        lo=0.0, hi=20.0)


def _validate_pose(value: Any, path: str, errors: list[str]) -> None:
    pose = _strict(
        value, path, {"position_m", "quaternion_wxyz", "placement", "z_m"},
        errors)
    if "position_m" in pose:
        _vector(pose["position_m"], f"{path}.position_m", errors, 3)
    elif "placement" in pose:
        placement = pose["placement"]
        if not isinstance(placement, str) or not placement.startswith("zone:"):
            errors.append(f"{path}.placement: must be zone:<id>")
        _number(pose.get("z_m", 0.0), f"{path}.z_m", errors, lo=-2.0, hi=20.0)
    else:
        errors.append(f"{path}: position_m or placement is required")
    if "quaternion_wxyz" in pose:
        _vector(pose["quaternion_wxyz"],
                f"{path}.quaternion_wxyz", errors, 4)


def _validate_objects(
    objects: Any, zones: Any, errors: list[str], sim: SimulatorCapability,
) -> None:
    if not isinstance(objects, dict):
        errors.append("shared.objects: must be an object")
        objects = {}
    if not isinstance(zones, dict):
        errors.append("shared.zones: must be an object")
        zones = {}
    for name, obj in objects.items():
        path = f"shared.objects.{name}"
        _identifier(name, path, errors)
        obj = _strict(obj, path, _OBJECT_KEYS, errors)
        shape = obj.get("shape")
        if shape not in sim.entity_shapes:
            errors.append(f"{path}.shape: unsupported {shape!r}")
            continue
        if "fixed" in obj and not isinstance(obj["fixed"], bool):
            errors.append(f"{path}.fixed: must be boolean")
        nominal = _strict(
            obj.get("nominal"), f"{path}.nominal",
            _OBJECT_NOMINAL_COMMON | _OBJECT_SHAPE_KEYS.get(str(shape), set()),
            errors)
        required = {
            "sphere": {"radius_m"}, "box": {"size_m"},
            "cylinder": {"radius_m", "height_m"},
            "capsule": {"radius_m", "height_m"},
            "frame": {"opening_m", "post_radius_m"},
        }.get(str(shape), set()) | {"pose"}
        missing = required - set(nominal)
        if missing:
            errors.append(f"{path}.nominal: missing {sorted(missing)}")
        if "radius_m" in nominal:
            _number(nominal["radius_m"], f"{path}.nominal.radius_m",
                    errors, lo=0.005, hi=5.0)
        if "height_m" in nominal:
            _number(nominal["height_m"], f"{path}.nominal.height_m",
                    errors, lo=0.005, hi=10.0)
        if "size_m" in nominal:
            _vector(nominal["size_m"], f"{path}.nominal.size_m",
                    errors, 3, positive=True)
        if "opening_m" in nominal:
            _vector(nominal["opening_m"], f"{path}.nominal.opening_m",
                    errors, 2, positive=True)
        for key, bounds in {
            "mass_kg": (0.001, 500.0), "friction": (0.0, 5.0),
            "restitution": (0.0, 1.0), "post_radius_m": (0.005, 1.0),
            "depth_m": (0.005, 10.0),
        }.items():
            if key in nominal:
                _number(nominal[key], f"{path}.nominal.{key}", errors,
                        lo=bounds[0], hi=bounds[1])
        if "pose" in nominal:
            _validate_pose(nominal["pose"], f"{path}.nominal.pose", errors)
    for name, zone in zones.items():
        path = f"shared.zones.{name}"
        _identifier(name, path, errors)
        if not isinstance(zone, dict):
            errors.append(f"{path}: must be an object")
            continue
        kind = zone.get("kind")
        if kind not in _ZONE_KEYS:
            errors.append(f"{path}.kind: must be disk or box")
            continue
        zone = _strict(zone, path, _ZONE_KEYS[kind], errors)
        _vector(zone.get("center_m"), f"{path}.center_m", errors,
                2 if kind == "disk" else 3)
        if kind == "disk":
            _number(zone.get("radius_m"), f"{path}.radius_m", errors,
                    lo=0.01, hi=100.0)
        else:
            _vector(zone.get("size_m"), f"{path}.size_m", errors,
                    3, positive=True)
        attach = zone.get("attach")
        if attach is not None and attach not in objects:
            errors.append(f"{path}.attach: unknown object {attach!r}")
    for name, obj in objects.items():
        pose = obj.get("nominal", {}).get("pose", {}) \
            if isinstance(obj, dict) else {}
        placement = pose.get("placement") if isinstance(pose, dict) else None
        if isinstance(placement, str) and placement.startswith("zone:"):
            zone_id = placement.split(":", 1)[1]
            if zone_id not in zones:
                errors.append(
                    f"shared.objects.{name}.nominal.pose.placement: "
                    f"unknown zone {zone_id!r}")


def editable_registry(spec: dict[str, Any]) -> dict[str, str]:
    """Return exact JSON-pointer targets and their declared variation class."""
    registry: dict[str, str] = {}

    def leaves(value: Any, prefix: str, klass: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                leaves(child, f"{prefix}/{key}", klass)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                leaves(child, f"{prefix}/{index}", klass)
        elif _is_num(value):
            registry[prefix] = klass

    shared = spec.get("shared", {})
    if not isinstance(shared, dict):
        return registry
    terrain = shared.get("terrain", {})
    sub_terrains = terrain.get("sub_terrains", {}) \
        if isinstance(terrain, dict) else {}
    if not isinstance(sub_terrains, dict):
        sub_terrains = {}
    for name, entry in sub_terrains.items():
        if not isinstance(entry, dict):
            continue
        leaves(entry.get("nominal", {}),
               f"/shared/terrain/sub_terrains/{name}/nominal",
               "generator_parameter")
    obstacles = shared.get("obstacles", {})
    course = obstacles.get("course", []) if isinstance(obstacles, dict) else []
    if not isinstance(course, list):
        course = []
    for item in course:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id:
            leaves(item.get("nominal", {}),
                   f"/shared/obstacles/course/@{item_id}/nominal",
                   "model_field")
    objects = shared.get("objects", {})
    if not isinstance(objects, dict):
        objects = {}
    for name, obj in objects.items():
        if not isinstance(obj, dict):
            continue
        nominal = obj.get("nominal", {})
        if not isinstance(nominal, dict):
            continue
        for key, value in nominal.items():
            klass = "state" if key == "pose" else "model_field"
            leaves(value, f"/shared/objects/{name}/nominal/{key}", klass)
    return registry


def _validate_train(spec: dict[str, Any], errors: list[str]) -> None:
    train = _strict(spec.get("train"), "train", _TRAIN_KEYS, errors)
    registry = editable_registry(spec)
    variations = train.get("variations", [])
    if not isinstance(variations, list):
        errors.append("train.variations: must be a list")
        variations = []
    seen_ids: set[str] = set()
    seen_targets: set[str] = set()
    for index, variation in enumerate(variations):
        path = f"train.variations[{index}]"
        variation = _strict(
            variation, path,
            {"id", "target", "class", "distribution", "curriculum"}, errors)
        variation_id = variation.get("id")
        if _identifier(variation_id, f"{path}.id", errors):
            if variation_id in seen_ids:
                errors.append(f"{path}.id: duplicate {variation_id!r}")
            seen_ids.add(variation_id)
        target = variation.get("target")
        if not isinstance(target, str):
            errors.append(f"{path}.target: required JSON pointer string")
        else:
            if target in seen_targets:
                errors.append(f"{path}.target: duplicated by another variation")
            seen_targets.add(target)
            expected = registry.get(target)
            if expected is None:
                errors.append(f"{path}.target: not an editable field: {target}")
            if variation.get("class") == "structural":
                errors.append(f"{path}.class: structural variation is forbidden")
            elif expected and variation.get("class") != expected:
                errors.append(
                    f"{path}.class: {variation.get('class')!r} does not "
                    f"match registry class {expected!r}")
        _validate_distribution(
            variation.get("distribution"), f"{path}.distribution", errors)
        curriculum = variation.get("curriculum")
        if curriculum is not None:
            curriculum = _strict(
                curriculum, f"{path}.curriculum", {"axis", "mapping"}, errors)
            if curriculum.get("axis") != "difficulty":
                errors.append(f"{path}.curriculum.axis: must be difficulty")
            if curriculum.get("mapping") not in {"linear", "log", "step"}:
                errors.append(f"{path}.curriculum.mapping: unsupported")
    curriculum = train.get("curriculum", {})
    if curriculum:
        curriculum = _strict(
            curriculum, "train.curriculum",
            {"difficulty_range", "promotion"}, errors)
        difficulty = curriculum.get("difficulty_range")
        _vector(difficulty, "train.curriculum.difficulty_range", errors, 2)
        if (isinstance(difficulty, (list, tuple)) and len(difficulty) == 2
                and all(_is_num(v) for v in difficulty)):
            lo, hi = map(float, difficulty)
            if not 0.0 <= lo <= hi <= 1.0:
                errors.append(
                    "train.curriculum.difficulty_range: must lie in [0,1]")
        promotion = curriculum.get("promotion")
        if promotion is not None:
            promotion = _strict(
                promotion, "train.curriculum.promotion",
                {"signal", "promote_above", "demote_below"}, errors)
            if promotion.get("signal") not in {
                    "traversal_fraction", "waypoint_success", "task_success"}:
                errors.append("train.curriculum.promotion.signal: unsupported")
            _number(promotion.get("promote_above"),
                    "train.curriculum.promotion.promote_above", errors,
                    lo=0.0, hi=1.0)
            _number(promotion.get("demote_below"),
                    "train.curriculum.promotion.demote_below", errors,
                    lo=0.0, hi=1.0)


def validate_world_spec(
    spec: Any, *, simulator: SimulatorCapability | None = None,
) -> list[str]:
    """Return every WorldSpec violation; empty means valid."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"world spec must be an object, got {type(spec).__name__}"]
    _strict(spec, "world", _TOP_KEYS, errors)
    if spec.get("world_spec_version") != WORLD_SPEC_VERSION:
        errors.append(
            f"world_spec_version must be {WORLD_SPEC_VERSION}, "
            f"got {spec.get('world_spec_version')!r}")
    _validate_meta(spec.get("meta"), errors)
    shared = _strict(spec.get("shared"), "shared", _SHARED_KEYS, errors)
    seed = shared.get("eval_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) \
            or not 0 <= seed < 2**32:
        errors.append("shared.eval_seed: integer in [0, 2^32) required")
    _validate_robot(shared.get("robot"), errors)
    sim = simulator or simulator_capability()
    _validate_terrain(shared.get("terrain"), errors, sim)
    _validate_obstacles(shared.get("obstacles", {}), errors)
    _validate_objects(shared.get("objects", {}), shared.get("zones", {}),
                      errors, sim)
    _validate_train(spec, errors)
    return errors


def load_world_spec(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"world spec {path} is unreadable: {exc}") from exc
    errors = validate_world_spec(spec)
    if errors:
        raise ValueError("invalid WorldSpec:\n- " + "\n- ".join(errors))
    return spec


def resolve_pointer(document: dict[str, Any], pointer: str) -> tuple[Any, Any]:
    """Resolve JSON pointer with ``@stable-id`` list segments.

    Returns ``(parent, final_key_or_index)`` so callers can update a value
    transactionally without relying on course array order.
    """
    if not pointer.startswith("/"):
        raise KeyError("pointer must start with /")
    parts = [p.replace("~1", "/").replace("~0", "~")
             for p in pointer.split("/")[1:]]
    if not parts:
        raise KeyError("pointer cannot address the document root")
    current: Any = document
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            if part.startswith("@"):
                stable_id = part[1:]
                matches = [x for x in current
                           if isinstance(x, dict) and x.get("id") == stable_id]
                if len(matches) != 1:
                    raise KeyError(
                        f"stable ID {stable_id!r} resolved {len(matches)} items")
                current = matches[0]
            else:
                current = current[int(part)]
        else:
            raise KeyError(f"cannot traverse {part!r}")
    final: Any = parts[-1]
    if isinstance(current, list) and not final.startswith("@"):
        final = int(final)
    return current, final


def apply_variation_value(
    spec: dict[str, Any], variation_id: str, value: Any,
) -> dict[str, Any]:
    """Return a validated copy with one registered variation value applied."""
    variations = spec.get("train", {}).get("variations", [])
    matches = [v for v in variations if v.get("id") == variation_id]
    if len(matches) != 1:
        raise KeyError(f"unknown/ambiguous variation ID {variation_id!r}")
    target = matches[0]["target"]
    updated = copy.deepcopy(spec)
    parent, key = resolve_pointer(updated, target)
    parent[key] = value
    errors = validate_world_spec(updated)
    if errors:
        raise ValueError("variation produced invalid WorldSpec:\n- "
                         + "\n- ".join(errors))
    return updated
