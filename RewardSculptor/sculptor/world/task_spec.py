"""Strict TaskSpec v1 validation against a concrete WorldSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sculptor.world.capabilities import (
    CapabilityError, resolve_robot_capability, simulator_capability,
)
from sculptor.world.world_spec import _identifier, _is_num, _strict

TASK_SPEC_VERSION = 1
_TOP_KEYS = {"task_spec_version", "meta", "shared", "train"}
_META_KEYS = {"version", "parent", "source", "prompt", "grounding"}
_SHARED_KEYS = {
    "control_mode", "goal", "contacts", "termination", "observations",
    "event_sequence",
}
_TRAIN_KEYS = {"goal_sampling", "scaffolds", "event_phase_sampling"}
_GOAL_TYPES = {
    "object_to_region", "object_velocity", "robot_to_region",
    "waypoint_sequence", "configuration_distribution",
}
_PREDICATES = {
    "inside", "speed_above", "distance_below", "sequence_complete",
    "configuration_match",
}

_EVENT_PHASE_IDS = ("route", "jump", "hold")


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
            if role == "any":
                names.extend((cap.root_body, *(
                    name for resolved in cap.body_roles.values()
                    for name in resolved
                )))
            else:
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


def _validate_event_sequence(
    value: Any, world: dict[str, Any], goal: Any, errors: list[str],
) -> None:
    """Validate the small, explicit one-shot event automaton capability.

    This is deliberately not a general predicate language.  The execution
    contract admits one linear ROUTE -> JUMP -> HOLD program whose transitions
    are grounded in authoritative goal completion and two declared support
    contacts.  Extending the event vocabulary requires a new compiler/runtime
    implementation and tests; unknown behavior therefore fails closed.
    """
    if value is None:
        return
    value = _strict(
        value,
        "shared.event_sequence",
        {"id", "phases"},
        errors,
    )
    _identifier(value.get("id"), "shared.event_sequence.id", errors)
    if not isinstance(goal, dict) or goal.get("type") != "waypoint_sequence":
        errors.append(
            "shared.event_sequence: goal_complete currently requires a "
            "waypoint_sequence goal"
        )

    phases = value.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        errors.append(
            "shared.event_sequence.phases: must contain route, jump, hold"
        )
        phases = []
    phase_ids = tuple(
        phase.get("id") if isinstance(phase, dict) else None
        for phase in phases
    )
    if phases and phase_ids != _EVENT_PHASE_IDS:
        errors.append(
            "shared.event_sequence.phases: ordered IDs must be "
            "route, jump, hold"
        )

    if len(phases) == 3:
        route = _strict(
            phases[0],
            "shared.event_sequence.phases[0]",
            {"id", "until"},
            errors,
        )
        route_until = _strict(
            route.get("until"),
            "shared.event_sequence.phases[0].until",
            {"event"},
            errors,
        )
        if route_until.get("event") != "goal_complete":
            errors.append(
                "shared.event_sequence.phases[0].until.event: must be "
                "goal_complete"
            )

        jump = _strict(
            phases[1],
            "shared.event_sequence.phases[1]",
            {"id", "until"},
            errors,
        )
        jump_until = _strict(
            jump.get("until"),
            "shared.event_sequence.phases[1].until",
            {
                "event", "support_contacts", "min_air_time_s",
                "min_height_delta_m",
            },
            errors,
        )
        if jump_until.get("event") != "bilateral_support_cycle":
            errors.append(
                "shared.event_sequence.phases[1].until.event: must be "
                "bilateral_support_cycle"
            )
        support_contacts = jump_until.get("support_contacts")
        if not isinstance(support_contacts, list) or len(support_contacts) != 2:
            errors.append(
                "shared.event_sequence.phases[1].until.support_contacts: "
                "must contain exactly two contact selector pairs"
            )
        else:
            resolved_supports: list[set[str]] = []
            for index, pair in enumerate(support_contacts):
                path = (
                    "shared.event_sequence.phases[1].until."
                    f"support_contacts[{index}]"
                )
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    errors.append(f"{path}: must contain two selectors")
                    continue
                resolved_pair = []
                for side, selector in enumerate(pair):
                    resolved_pair.append(parse_contact_selector(
                        selector,
                        world=world,
                        errors=errors,
                        path=f"{path}[{side}]",
                    ))
                if (
                    resolved_pair[0] is not None
                    and resolved_pair[0][0] == "robot"
                ):
                    resolved_supports.append(set(resolved_pair[0][1]))
                else:
                    errors.append(
                        f"{path}[0]: support primary must be a robot role"
                    )
                if resolved_pair[1] != ("world", ("terrain",)):
                    errors.append(
                        f"{path}[1]: support secondary must be world:terrain"
                    )
            if (
                len(resolved_supports) == 2
                and resolved_supports[0] & resolved_supports[1]
            ):
                errors.append(
                    "shared.event_sequence.phases[1].until.support_contacts: "
                    "robot support roles must resolve to distinct bodies"
                )
        _number(
            jump_until.get("min_air_time_s"),
            "shared.event_sequence.phases[1].until.min_air_time_s",
            errors,
            lo=0.02,
            hi=2.0,
        )
        _number(
            jump_until.get("min_height_delta_m"),
            "shared.event_sequence.phases[1].until.min_height_delta_m",
            errors,
            lo=0.05,
            hi=2.0,
        )

        hold = _strict(
            phases[2],
            "shared.event_sequence.phases[2]",
            {"id", "terminal", "minimum_hold_s"},
            errors,
        )
        if hold.get("terminal") is not True:
            errors.append(
                "shared.event_sequence.phases[2].terminal: must be true"
            )
        _number(
            hold.get("minimum_hold_s"),
            "shared.event_sequence.phases[2].minimum_hold_s",
            errors,
            lo=2.0,
            hi=10.0,
        )

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


def _validate_train(
    value: Any,
    world: dict[str, Any],
    event_sequence: Any,
    errors: list[str],
) -> None:
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
    phase_sampling = value.get("event_phase_sampling")
    if event_sequence is None:
        if phase_sampling is not None:
            errors.append(
                "train.event_phase_sampling requires shared.event_sequence"
            )
        return
    if not isinstance(phase_sampling, dict):
        errors.append("train.event_phase_sampling: must be an object")
        return
    unknown = set(phase_sampling) - set(_EVENT_PHASE_IDS)
    missing = set(_EVENT_PHASE_IDS) - set(phase_sampling)
    if unknown or missing:
        errors.append(
            "train.event_phase_sampling: expected exactly route, jump, hold"
        )
    total = 0.0
    valid = True
    for phase_id in _EVENT_PHASE_IDS:
        raw = phase_sampling.get(phase_id)
        before = len(errors)
        _number(
            raw,
            f"train.event_phase_sampling.{phase_id}",
            errors,
            lo=0.0,
            hi=1.0,
        )
        if len(errors) != before:
            valid = False
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            total += float(raw)
    if valid and abs(total - 1.0) > 1e-6:
        errors.append(
            "train.event_phase_sampling: probabilities must sum to 1, "
            f"got {total:g}"
        )


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
    _validate_event_sequence(
        shared.get("event_sequence"), world, shared.get("goal"), errors)
    _validate_termination(shared.get("termination", {}), errors)
    if (
        shared.get("event_sequence") is not None
        and isinstance(shared.get("termination"), dict)
        and shared["termination"].get("success_ends_episode") is True
    ):
        errors.append(
            "shared.termination.success_ends_episode: event sequences require "
            "false until compound-success termination is implemented"
        )
    _validate_observations(shared.get("observations", {}), world, errors)
    _validate_train(
        spec.get("train", {}),
        world,
        shared.get("event_sequence"),
        errors,
    )
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
