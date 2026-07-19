"""Strict TaskSpec v1 validation against a concrete WorldSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sculptor.world.capabilities import (
    CapabilityError, resolve_robot_capability, simulator_capability,
)
from sculptor.world.world_spec import _identifier, _is_num, _strict, _vector

TASK_SPEC_VERSION = 1
_TOP_KEYS = {"task_spec_version", "meta", "shared", "train"}
_META_KEYS = {"version", "parent", "source", "prompt", "grounding"}
_SHARED_KEYS = {
    "control_mode", "goal", "contacts", "termination", "observations",
}
_TRAIN_KEYS = {"goal_sampling", "scaffolds"}
_GOAL_TYPES = {
    "object_to_region", "object_velocity", "robot_to_region",
    "waypoint_sequence", "configuration_distribution",
}
_PREDICATES = {
    "inside", "speed_above", "distance_below", "sequence_complete",
    "configuration_match",
}


def _number(
    value: Any, path: str, errors: list[str], *,
    lo: float | None = None, hi: float | None = None,
) -> None:
    if not _is_num(value):
        errors.append(f"{path}: must be a finite number")
        return
    value = float(value)
    if lo is not None and value < lo:
        errors.append(f"{path}: {value:g} below {lo:g}")
    if hi is not None and value > hi:
        errors.append(f"{path}: {value:g} above {hi:g}")


def _world_sets(world: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    shared = world.get("shared", {})
    objects = set(shared.get("objects", {}))
    zones = set(shared.get("zones", {}))
    course = {
        item.get("id") for item in
        shared.get("obstacles", {}).get("course", [])
        if isinstance(item, dict) and item.get("id")
    }
    return objects, zones, course


def parse_contact_selector(
    selector: str, *, world: dict[str, Any], errors: list[str] | None = None,
    path: str = "contact",
) -> tuple[str, tuple[str, ...]] | None:
    """Resolve a semantic contact selector to concrete names.

    Return kind + names. Robot role alternatives use ``|``. The function is
    shared by schema validation and the runtime contact compiler.
    """
    local_errors = errors if errors is not None else []
    if not isinstance(selector, str) or ":" not in selector:
        local_errors.append(f"{path}: selector must be kind:name")
        return None
    kind, raw_name = selector.split(":", 1)
    objects, zones, course = _world_sets(world)
    if kind == "robot":
        robot = world.get("shared", {}).get("robot", {})
        try:
            cap = resolve_robot_capability(
                robot.get("capability_id", ""),
                required=robot.get("required_capabilities", []),
                extra_paths=([robot["descriptor_path"]]
                             if robot.get("descriptor_path") else []),
            )
        except (CapabilityError, OSError, ValueError) as exc:
            local_errors.append(f"{path}: {exc}")
            return None
        names: list[str] = []
        for role in raw_name.split("|"):
            try:
                names.extend(cap.resolve_role(role))
            except CapabilityError as exc:
                local_errors.append(f"{path}: {exc}")
        return ("robot", tuple(dict.fromkeys(names))) if names else None
    if kind == "object":
        if raw_name not in objects:
            local_errors.append(f"{path}: unknown object {raw_name!r}")
            return None
        return "object", (raw_name,)
    if kind == "zone":
        if raw_name not in zones:
            local_errors.append(f"{path}: unknown zone {raw_name!r}")
            return None
        return "zone", (raw_name,)
    if kind == "obstacle":
        if raw_name not in course:
            local_errors.append(f"{path}: unknown obstacle {raw_name!r}")
            return None
        return "obstacle", (raw_name,)
    if kind == "world" and raw_name == "terrain":
        return "world", ("terrain",)
    local_errors.append(f"{path}: unsupported selector {selector!r}")
    return None


def _validate_goal(
    goal: Any, world: dict[str, Any], errors: list[str],
) -> None:
    goal = _strict(
        goal, "shared.goal",
        {"id", "type", "subject", "region", "target", "waypoints",
         "success"}, errors)
    _identifier(goal.get("id"), "shared.goal.id", errors)
    goal_type = goal.get("type")
    if goal_type not in _GOAL_TYPES:
        errors.append(f"shared.goal.type: unsupported {goal_type!r}")
        return
    objects, zones, course = _world_sets(world)
    subject = goal.get("subject")
    region = goal.get("region")
    if goal_type in {"object_to_region", "object_velocity"}:
        if subject not in objects:
            errors.append(f"shared.goal.subject: unknown object {subject!r}")
    if goal_type in {"object_to_region", "robot_to_region"}:
        if region not in zones:
            errors.append(f"shared.goal.region: unknown zone {region!r}")
    if goal_type == "waypoint_sequence":
        waypoints = goal.get("waypoints", "auto")
        if waypoints != "auto":
            if (not isinstance(waypoints, list) or not waypoints
                    or not all(w in course or w in zones for w in waypoints)):
                errors.append(
                    "shared.goal.waypoints: non-empty known ID list or auto")
    success = _strict(
        goal.get("success"), "shared.goal.success",
        {"predicate", "hold_s", "tolerance_m", "threshold", "ordered"},
        errors)
    predicate = success.get("predicate")
    expected = {
        "object_to_region": "inside",
        "object_velocity": "speed_above",
        "robot_to_region": "distance_below",
        "waypoint_sequence": "sequence_complete",
        "configuration_distribution": "configuration_match",
    }.get(str(goal_type))
    if predicate not in _PREDICATES:
        errors.append(f"shared.goal.success.predicate: unsupported {predicate!r}")
    elif expected and predicate != expected:
        errors.append(
            f"shared.goal.success.predicate: {goal_type} requires {expected}")
    _number(success.get("hold_s", 0.0), "shared.goal.success.hold_s",
            errors, lo=0.0, hi=10.0)
    if "tolerance_m" in success:
        _number(success["tolerance_m"],
                "shared.goal.success.tolerance_m", errors, lo=0.0, hi=5.0)
    if "threshold" in success:
        _number(success["threshold"], "shared.goal.success.threshold", errors)
    if "ordered" in success and not isinstance(success["ordered"], bool):
        errors.append("shared.goal.success.ordered: must be boolean")


def _validate_contacts(
    contacts: Any, world: dict[str, Any], errors: list[str],
) -> None:
    contacts = _strict(
        contacts, "shared.contacts",
        {"desired", "forbidden", "terminate_on"}, errors)
    for group in ("desired", "forbidden", "terminate_on"):
        pairs = contacts.get(group, [])
        if not isinstance(pairs, list):
            errors.append(f"shared.contacts.{group}: must be a list")
            continue
        for index, pair in enumerate(pairs):
            path = f"shared.contacts.{group}[{index}]"
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                errors.append(f"{path}: must contain two selectors")
                continue
            for side, selector in enumerate(pair):
                parse_contact_selector(
                    selector, world=world, errors=errors,
                    path=f"{path}[{side}]")


def _validate_termination(value: Any, errors: list[str]) -> None:
    value = _strict(
        value, "shared.termination",
        {"fall", "out_of_bounds_m", "success_ends_episode",
         "episode_length_s"}, errors)
    if value.get("fall", "capability_default") not in {
            "capability_default", "disabled", "torso_contact"}:
        errors.append("shared.termination.fall: unsupported")
    _number(value.get("out_of_bounds_m", 12.0),
            "shared.termination.out_of_bounds_m", errors, lo=0.5, hi=500.0)
    _number(value.get("episode_length_s", 20.0),
            "shared.termination.episode_length_s", errors, lo=1.0, hi=300.0)
    if not isinstance(value.get("success_ends_episode", False), bool):
        errors.append("shared.termination.success_ends_episode: must be boolean")


def _validate_observations(
    value: Any, world: dict[str, Any], errors: list[str],
) -> None:
    value = _strict(
        value, "shared.observations",
        {"proprioception", "height_scan", "object_relative",
         "region_relative", "end_effector_relative"}, errors)
    robot = world.get("shared", {}).get("robot", {})
    try:
        cap = resolve_robot_capability(
            robot.get("capability_id", ""),
            required=robot.get("required_capabilities", []),
            extra_paths=([robot["descriptor_path"]]
                         if robot.get("descriptor_path") else []))
    except (CapabilityError, OSError, ValueError) as exc:
        errors.append(f"shared.observations: {exc}")
        return
    sim = simulator_capability()
    for flag in ("proprioception", "height_scan"):
        raw = value.get(flag, flag == "proprioception")
        if raw not in {True, False, "auto"}:
            errors.append(f"shared.observations.{flag}: boolean or auto")
        if raw in {True, "auto"} and flag not in cap.supported_observations:
            errors.append(
                f"shared.observations.{flag}: robot capability unavailable")
        if raw in {True, "auto"} and flag not in sim.observation_adapters:
            errors.append(
                f"shared.observations.{flag}: simulator adapter unavailable")
    objects, zones, _ = _world_sets(world)
    for key, known in (("object_relative", objects),
                       ("region_relative", zones)):
        names = value.get(key, [])
        if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names):
            errors.append(f"shared.observations.{key}: must be a list")
        else:
            missing = sorted(set(names) - known)
            if missing:
                errors.append(
                    f"shared.observations.{key}: unknown names {missing}")
    end_effector = value.get("end_effector_relative", [])
    if end_effector:
        if "end_effector_relative" not in cap.supported_observations:
            errors.append(
                "shared.observations.end_effector_relative: robot "
                "capability unavailable")
        if "end_effector_relative" not in sim.observation_adapters:
            errors.append(
                "shared.observations.end_effector_relative: simulator "
                "adapter unavailable")
        roles = (["end_effector"] if end_effector is True
                 or end_effector == "auto"
                 else end_effector)
        if not isinstance(roles, list) or not all(
                isinstance(role, str) and role for role in roles):
            errors.append(
                "shared.observations.end_effector_relative: must be a "
                "role list, true, false, or auto")
        else:
            for role in roles:
                try:
                    cap.resolve_semantic_role(role)
                except CapabilityError as exc:
                    errors.append(
                        "shared.observations.end_effector_relative: "
                        f"{exc}")


def _validate_train(value: Any, world: dict[str, Any], errors: list[str]) -> None:
    value = _strict(value, "train", _TRAIN_KEYS, errors)
    sampling = value.get("goal_sampling", [])
    if not isinstance(sampling, list):
        errors.append("train.goal_sampling: must be a list")
        sampling = []
    objects, zones, _ = _world_sets(world)
    for index, item in enumerate(sampling):
        path = f"train.goal_sampling[{index}]"
        item = _strict(
            item, path, {"id", "target", "distribution"}, errors)
        _identifier(item.get("id"), f"{path}.id", errors)
        target = item.get("target")
        if not isinstance(target, str) or not target.startswith("object:") \
                or not target.endswith(".pose"):
            errors.append(f"{path}.target: must be object:<id>.pose")
        elif target[7:-5] not in objects:
            errors.append(f"{path}.target: unknown object")
        dist = item.get("distribution")
        if not isinstance(dist, dict):
            errors.append(f"{path}.distribution: must be an object")
        else:
            unknown = set(dist) - {"kind", "region"}
            if unknown:
                errors.append(f"{path}.distribution: unknown keys {sorted(unknown)}")
            if dist.get("kind") != "uniform_in_region":
                errors.append(
                    f"{path}.distribution.kind: must be uniform_in_region")
            if dist.get("region") not in zones:
                errors.append(f"{path}.distribution.region: unknown zone")
    scaffolds = value.get("scaffolds", [])
    if not isinstance(scaffolds, list) or not all(
            isinstance(x, str) for x in scaffolds):
        errors.append("train.scaffolds: must be a list of scaffold IDs")


def validate_task_spec(spec: Any, *, world: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"task spec must be an object, got {type(spec).__name__}"]
    _strict(spec, "task", _TOP_KEYS, errors)
    if spec.get("task_spec_version") != TASK_SPEC_VERSION:
        errors.append(
            f"task_spec_version must be {TASK_SPEC_VERSION}, "
            f"got {spec.get('task_spec_version')!r}")
    meta = _strict(spec.get("meta"), "meta", _META_KEYS, errors)
    if "version" in meta and not isinstance(meta["version"], str):
        errors.append("meta.version: must be a string")
    shared = _strict(spec.get("shared"), "shared", _SHARED_KEYS, errors)
    control_mode = shared.get("control_mode")
    if control_mode not in {
            "goal_directed", "base_velocity", "waypoint_following"}:
        errors.append(f"shared.control_mode: unsupported {control_mode!r}")
    _validate_goal(shared.get("goal"), world, errors)
    _validate_contacts(shared.get("contacts", {}), world, errors)
    _validate_termination(shared.get("termination", {}), errors)
    _validate_observations(shared.get("observations", {}), world, errors)
    _validate_train(spec.get("train", {}), world, errors)
    return errors


def load_task_spec(
    path: Path | str, *, world: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"task spec {path} is unreadable: {exc}") from exc
    errors = validate_task_spec(spec, world=world)
    if errors:
        raise ValueError("invalid TaskSpec:\n- " + "\n- ".join(errors))
    return spec
