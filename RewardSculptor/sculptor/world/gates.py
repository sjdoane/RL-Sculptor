"""CPU-only admission gates for authored robot environments.

Every gate reports all violations it can establish.  A failure never starts
training, and one violation does not hide independent problems in later
analytic gates.
"""

from __future__ import annotations

import copy
import dataclasses
import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sculptor.world.capabilities import (
    CapabilityError,
    resolve_robot_capability,
    simulator_capability,
)
from sculptor.world.compiler import (
    CompiledWorld,
    WorldCompileError,
    compile_task_runtime,
    compile_world,
    resolve_course,
    resolve_objects,
)
from sculptor.world.observation_geometry import height_scan_ray_count
from sculptor.world.task_spec import validate_task_spec
from sculptor.world.world_spec import validate_world_spec


@dataclass(frozen=True)
class GateViolation:
    gate: str
    code: str
    path: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GateResult:
    gate: str
    ok: bool
    violations: tuple[GateViolation, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate, "ok": self.ok,
            "violations": [item.to_dict() for item in self.violations],
            "evidence": copy.deepcopy(self.evidence),
        }


@dataclass(frozen=True)
class AdmissionReport:
    report_version: int
    ok: bool
    gates: tuple[GateResult, ...]

    @property
    def violations(self) -> tuple[GateViolation, ...]:
        return tuple(
            violation for gate in self.gates for violation in gate.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version, "ok": self.ok,
            "gates": [gate.to_dict() for gate in self.gates],
            "violations": [item.to_dict() for item in self.violations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdmissionReport":
        gates = tuple(GateResult(
            gate=item["gate"], ok=bool(item["ok"]),
            violations=tuple(GateViolation(**violation)
                             for violation in item.get("violations", ())),
            evidence=item.get("evidence", {}),
        ) for item in value.get("gates", ()))
        return cls(report_version=int(value.get("report_version", 1)),
                   ok=bool(value.get("ok", False)), gates=gates)


def _violation(
    gate: str, code: str, path: str, message: str, **details: Any,
) -> GateViolation:
    return GateViolation(
        gate=gate, code=code, path=path, message=message, details=details)


def _schema_gate(
    world: Mapping[str, Any], task: Mapping[str, Any],
) -> GateResult:
    violations = [
        _violation("schema", "invalid_world_spec", "world", error)
        for error in validate_world_spec(world)
    ]
    violations.extend(
        _violation("schema", "invalid_task_spec", "task", error)
        for error in validate_task_spec(task, world=dict(world)))
    return GateResult("schema", not violations, tuple(violations), {
        "world_errors": sum(v.code == "invalid_world_spec" for v in violations),
        "task_errors": sum(v.code == "invalid_task_spec" for v in violations),
    })


def _capability_gate(
    world: Mapping[str, Any], task: Mapping[str, Any],
) -> GateResult:
    violations: list[GateViolation] = []
    robot_data = world.get("shared", {}).get("robot", {})
    try:
        robot = resolve_robot_capability(
            robot_data.get("capability_id", ""),
            required=robot_data.get("required_capabilities", ()),
            extra_paths=([robot_data["descriptor_path"]]
                         if robot_data.get("descriptor_path") else ()))
    except (CapabilityError, OSError, ValueError) as exc:
        return GateResult("capability", False, (
            _violation("capability", "robot_unavailable", "shared.robot", str(exc)),
        ))
    sim = simulator_capability()
    observations = task.get("shared", {}).get("observations", {})
    for name, raw in observations.items():
        requested = raw is True or raw == "auto" or (
            isinstance(raw, (list, tuple)) and bool(raw))
        if requested and name not in robot.supported_observations:
            violations.append(_violation(
                "capability", "observation_unavailable",
                f"shared.observations.{name}",
                f"robot {robot.capability_id} does not expose {name!r}"))
        if requested and name not in sim.observation_adapters:
            violations.append(_violation(
                "capability", "observation_adapter_unavailable",
                f"shared.observations.{name}",
                f"simulator does not expose the {name!r} adapter"))
    command_for_mode = {
        "base_velocity": "base_velocity",
        "waypoint_following": "waypoint_heading",
    }.get(task.get("shared", {}).get("control_mode"))
    if command_for_mode and command_for_mode not in robot.supported_commands:
        violations.append(_violation(
            "capability", "command_unavailable", "shared.control_mode",
            f"robot {robot.capability_id} lacks command {command_for_mode!r}"))
    try:
        runtime = compile_task_runtime(world, task, robot)
    except (CapabilityError, KeyError, TypeError, ValueError) as exc:
        violations.append(_violation(
            "capability", "runtime_binding_failed", "task.shared", str(exc)))
        runtime = None
    contact_count = len(runtime.contacts) if runtime is not None else 0
    if contact_count > min(robot.contact_capacity, sim.budget.max_contacts):
        violations.append(_violation(
            "capability", "contact_capacity_exceeded", "shared.contacts",
            f"{contact_count} contact bindings exceed robot/simulator capacity",
            requested=contact_count,
            capacity=min(robot.contact_capacity, sim.budget.max_contacts)))
    return GateResult("capability", not violations, tuple(violations), {
        "robot_capability_id": robot.capability_id,
        "contact_bindings": contact_count,
    })


def estimate_budget(world: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, int]:
    """Conservative pre-build resource estimate used to avoid unsafe builds."""
    shared = world.get("shared", {})
    terrain = shared.get("terrain", {})
    geoms = 1
    texels = 0
    if terrain.get("kind") == "generator":
        layout = terrain.get("layout", {})
        rows = int(layout.get("rows", 1))
        cols = (len(terrain.get("sub_terrains", {}))
                if layout.get("mode") == "curriculum_grid"
                else int(layout.get("cols", 1)))
        tiles = rows * cols
        width, height = layout.get("tile_size_m", (10.0, 10.0))
        for entry in terrain.get("sub_terrains", {}).values():
            nominal = entry.get("nominal", {})
            horizontal = float(nominal.get("horizontal_scale_m", 0.1))
            if str(entry.get("type", "")).startswith("hf_"):
                texels += tiles * max(2, int(width / horizontal)) * max(
                    2, int(height / horizontal))
        # Primitive sub-terrains can emit many boxes; use the declared maximum
        # when present and a conservative per-tile fallback otherwise.
        per_tile = max((
            int(entry.get("nominal", {}).get(
                "num_boxes", entry.get("nominal", {}).get(
                    "num_obstacles", entry.get("nominal", {}).get(
                        "num_beams", 32))))
            for entry in terrain.get("sub_terrains", {}).values()
        ), default=1)
        geoms = tiles * max(1, per_tile) + 4
    geoms += len(resolve_course(world))
    for obj in shared.get("objects", {}).values():
        geoms += 3 if obj.get("shape") == "frame" else 1
    contacts = sum(len(task.get("shared", {}).get("contacts", {}).get(
        group, ())) for group in ("desired", "forbidden", "terminate_on"))
    height_scan = task.get("shared", {}).get("observations", {}).get(
        "height_scan", "auto")
    rays = height_scan_ray_count() if height_scan is True or (
        height_scan == "auto" and terrain.get("kind") == "generator") else 0
    return {
        "geoms": geoms, "contacts": contacts, "rays": rays,
        "constraints": 0, "heightfield_texels": texels,
    }


def _budget_gate(
    world: Mapping[str, Any], task: Mapping[str, Any],
) -> GateResult:
    sim = simulator_capability()
    estimate = estimate_budget(world, task)
    limits = {
        "geoms": sim.budget.max_geoms,
        "contacts": sim.budget.max_contacts,
        "rays": sim.budget.max_rays,
        "constraints": sim.budget.max_constraints,
        "heightfield_texels": sim.budget.max_heightfield_texels,
    }
    violations = tuple(
        _violation(
            "budget", f"{name}_budget_exceeded", f"budget.{name}",
            f"estimated {name} {estimate[name]} exceeds limit {limit}",
            estimated=estimate[name], limit=limit)
        for name, limit in limits.items() if estimate[name] > limit)
    return GateResult("budget", not violations, violations, {
        "estimate": estimate, "limits": limits})


def _build_gate(compiled: CompiledWorld | None, error: str | None) -> GateResult:
    if error is not None or compiled is None:
        violation = _violation(
            "build", "mujoco_compile_failed", "world",
            error or "world compilation did not produce a model")
        return GateResult("build", False, (violation,))
    inventory = dict(compiled.model_inventory)
    limits = compiled.simulator.budget
    violations: list[GateViolation] = []
    for name, actual, limit in (
        ("geoms", inventory["geoms"], limits.max_geoms),
        ("constraints", inventory["constraints"], limits.max_constraints),
        ("heightfield_texels", inventory["heightfield_texels"],
         limits.max_heightfield_texels),
    ):
        if actual > limit:
            violations.append(_violation(
                "build", f"actual_{name}_budget_exceeded", f"model.{name}",
                f"compiled model {name} {actual} exceeds limit {limit}",
                actual=actual, limit=limit))
    return GateResult("build", not violations, tuple(violations), inventory)


def _initial_penetration_gate(
    compiled: CompiledWorld | None, *, tolerance_m: float,
) -> GateResult:
    if compiled is None:
        return GateResult("initial_penetration", False, (
            _violation("initial_penetration", "build_required", "world",
                       "penetration gate requires a compiled model"),))
    import mujoco

    model = compiled._model
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    contacts: list[dict[str, Any]] = []
    support_bodies = {
        f"robot/{body}"
        for role, bodies in compiled.robot.body_roles.items()
        if "foot" in role
        for body in bodies
    }
    support_tolerance_m = max(tolerance_m, 0.04)
    for index in range(data.ncon):
        contact = data.contact[index]
        penetration = max(0.0, -float(contact.dist))
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = model.body(int(model.geom_bodyid[geom1])).name
        body2 = model.body(int(model.geom_bodyid[geom2])).name
        geom1_name = model.geom(geom1).name
        geom2_name = model.geom(geom2).name
        terrain_support = (
            (body1 in support_bodies and geom2_name.startswith("terrain"))
            or (body2 in support_bodies and geom1_name.startswith("terrain"))
        )
        allowed = support_tolerance_m if terrain_support else tolerance_m
        if penetration <= allowed:
            continue
        contacts.append({
            "penetration_m": penetration,
            "allowed_m": allowed,
            "geom1": geom1_name,
            "geom2": geom2_name,
        })
    violations = tuple(_violation(
        "initial_penetration", "excessive_initial_penetration",
        f"contacts[{index}]",
        f"initial penetration {item['penetration_m']:.4f} m exceeds "
        f"{item['allowed_m']:.4f} m", **item)
        for index, item in enumerate(contacts))
    return GateResult("initial_penetration", not violations, violations, {
        "contact_count": int(data.ncon), "tolerance_m": tolerance_m,
        "support_contact_tolerance_m": support_tolerance_m,
        "worst_penetration_m": max(
            (item["penetration_m"] for item in contacts), default=0.0),
    })


def _settle_gate(
    compiled: CompiledWorld | None, *, steps: int, sink_z_m: float,
) -> GateResult:
    if compiled is None:
        return GateResult("settle", False, (
            _violation("settle", "build_required", "world",
                       "settle gate requires a compiled model"),))
    import mujoco

    model = compiled._model
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    violations: list[GateViolation] = []
    failed_step: int | None = None
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            failed_step = step
            violations.append(_violation(
                "settle", "non_finite_state", "simulation.state",
                f"non-finite state at settle step {step}", step=step))
            break
    free_bodies = [index for index in range(1, model.nbody)
                   if int(model.body_jntnum[index]) > 0]
    for body_id in free_bodies:
        z = float(data.xpos[body_id, 2])
        if z < sink_z_m:
            violations.append(_violation(
                "settle", "body_fell_through", f"bodies.{model.body(body_id).name}",
                f"free body settled below sink threshold ({z:.3f} < {sink_z_m:.3f})",
                z_m=z, sink_z_m=sink_z_m))
    return GateResult("settle", not violations, tuple(violations), {
        "steps": steps, "failed_step": failed_step,
        "free_body_count": len(free_bodies),
        "max_abs_qvel_end": float(np.max(np.abs(data.qvel)))
        if data.qvel.size else 0.0,
    })


def _world_half_extents(world: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[float, float]:
    terrain = world.get("shared", {}).get("terrain", {})
    if terrain.get("kind") == "generator":
        layout = terrain["layout"]
        rows = int(layout["rows"])
        cols = (len(terrain["sub_terrains"])
                if layout["mode"] == "curriculum_grid" else int(layout["cols"]))
        return (rows * float(layout["tile_size_m"][0]) / 2,
                cols * float(layout["tile_size_m"][1]) / 2)
    bound = float(task.get("shared", {}).get(
        "termination", {}).get("out_of_bounds_m", 12.0))
    return bound, bound


def _object_radius(resolved: Mapping[str, Any]) -> float:
    nominal = resolved["nominal"]
    if resolved["shape"] == "sphere":
        return float(nominal["radius_m"])
    if resolved["shape"] == "box":
        return float(np.linalg.norm(np.asarray(nominal["size_m"], dtype=float))) / 2
    if resolved["shape"] in {"capsule", "cylinder"}:
        return float(nominal["radius_m"]) + float(nominal["height_m"]) / 2
    opening = nominal.get("opening_m", (0.5, 0.5))
    return max(map(float, opening)) / 2 + float(nominal.get("post_radius_m", 0.05))


def _placement_gate(
    world: Mapping[str, Any], task: Mapping[str, Any],
    compiled: CompiledWorld | None,
) -> GateResult:
    violations: list[GateViolation] = []
    try:
        objects = resolve_objects(world)
    except (KeyError, TypeError, ValueError) as exc:
        return GateResult("placement", False, (
            _violation("placement", "unresolved_placement", "shared.objects", str(exc)),))
    names = sorted(objects)
    for left_index, left_name in enumerate(names):
        left = objects[left_name]
        left_pos = np.asarray(left["position_m"], dtype=float)
        for right_name in names[left_index + 1:]:
            right = objects[right_name]
            distance = float(np.linalg.norm(
                left_pos - np.asarray(right["position_m"], dtype=float)))
            required = _object_radius(left) + _object_radius(right)
            if distance + 1e-4 < required:
                violations.append(_violation(
                    "placement", "object_overlap",
                    f"shared.objects.{left_name}|{right_name}",
                    f"object bounding volumes overlap ({distance:.3f} < {required:.3f})",
                    distance_m=distance, minimum_m=required))
    half_x, half_y = _world_half_extents(world, task)
    for name, zone in world.get("shared", {}).get("zones", {}).items():
        center = zone.get("center_m", ())
        if len(center) < 2:
            continue
        extent_x = (float(zone["radius_m"]) if zone.get("kind") == "disk"
                    else float(zone["size_m"][0]) / 2)
        extent_y = (float(zone["radius_m"]) if zone.get("kind") == "disk"
                    else float(zone["size_m"][1]) / 2)
        if abs(float(center[0])) + extent_x > half_x or \
                abs(float(center[1])) + extent_y > half_y:
            violations.append(_violation(
                "placement", "zone_outside_world", f"shared.zones.{name}",
                f"zone {name!r} extends outside world bounds",
                world_half_extents_m=[half_x, half_y]))
    if compiled is not None and world.get("shared", {}).get(
            "terrain", {}).get("kind") == "generator":
        required_patch_names = {
            f"object_{name}" for name, obj in world["shared"].get(
                "objects", {}).items()
            if str(obj.get("nominal", {}).get("pose", {}).get(
                "placement", "")).startswith("zone:")}
        available = set(compiled.resolved_eval.terrain.get("flat_patches", {}))
        for patch_name in sorted(required_patch_names - available):
            violations.append(_violation(
                "placement", "flat_patch_missing", "shared.terrain",
                f"generated terrain has no placement patches for {patch_name!r}"))
    return GateResult("placement", not violations, tuple(violations), {
        "objects": len(objects), "zones": len(
            world.get("shared", {}).get("zones", {})),
        "world_half_extents_m": [half_x, half_y],
    })


def _reachability_gate(
    world: Mapping[str, Any], task: Mapping[str, Any],
) -> GateResult:
    violations: list[GateViolation] = []
    robot_data = world.get("shared", {}).get("robot", {})
    try:
        robot = resolve_robot_capability(
            robot_data.get("capability_id", ""),
            required=robot_data.get("required_capabilities", ()),
            extra_paths=([robot_data["descriptor_path"]]
                         if robot_data.get("descriptor_path") else ()))
    except (CapabilityError, OSError, ValueError) as exc:
        return GateResult("reachability", False, (
            _violation("reachability", "robot_geometry_unavailable",
                       "shared.robot", str(exc)),))
    previous_height = 0.0
    for entry in world.get("shared", {}).get(
            "obstacles", {}).get("course", ()):
        nominal = entry.get("nominal", {})
        path = f"shared.obstacles.course.@{entry.get('id', '?')}"
        gap = (float(nominal.get("length_m", 0.0))
               if entry.get("element") == "gap"
               else float(nominal.get("gap_after_m", 0.0)))
        if gap > robot.geometry.max_gap_m:
            violations.append(_violation(
                "reachability", "gap_out_of_envelope", path,
                f"gap {gap:.3f} m exceeds robot envelope "
                f"{robot.geometry.max_gap_m:.3f} m", gap_m=gap,
                max_gap_m=robot.geometry.max_gap_m))
        height = float(nominal.get(
            "height_m", nominal.get("step_height_m", previous_height)))
        delta = abs(height - previous_height)
        if delta > robot.geometry.max_step_height_m:
            violations.append(_violation(
                "reachability", "climb_out_of_envelope", path,
                f"height change {delta:.3f} m exceeds robot envelope "
                f"{robot.geometry.max_step_height_m:.3f} m",
                height_change_m=delta,
                max_step_height_m=robot.geometry.max_step_height_m))
        if entry.get("element") != "gap":
            previous_height = height
        if entry.get("element") == "beam" and float(
                nominal.get("width_m", 0.0)) < robot.geometry.foot_length_m * 0.4:
            violations.append(_violation(
                "reachability", "beam_too_narrow", path,
                "beam width is below the robot support-envelope minimum"))

    goal = task.get("shared", {}).get("goal", {})
    if goal.get("type") == "object_to_region":
        objects = world.get("shared", {}).get("objects", {})
        zones = world.get("shared", {}).get("zones", {})
        subject = objects.get(goal.get("subject"), {})
        region = zones.get(goal.get("region"), {})
        if subject and region:
            radius = _object_radius({
                "shape": subject["shape"], "nominal": subject["nominal"]})
            if region.get("kind") == "disk":
                aperture = float(region["radius_m"])
            else:
                aperture = min(map(float, region["size_m"])) / 2
            if radius > aperture:
                violations.append(_violation(
                    "reachability", "object_does_not_fit_region",
                    f"shared.goal.{goal.get('id', 'goal')}",
                    f"object radius {radius:.3f} m exceeds region aperture "
                    f"{aperture:.3f} m", object_radius_m=radius,
                    region_aperture_m=aperture))
    if "locomotion" not in robot.capabilities and "manipulation" in robot.capabilities:
        try:
            objects = resolve_objects(world)
        except (KeyError, TypeError, ValueError):
            objects = {}
        relevant = set(task.get("shared", {}).get(
            "observations", {}).get("object_relative", ()))
        for name in sorted(relevant & set(objects)):
            distance = float(np.linalg.norm(np.asarray(
                objects[name]["position_m"], dtype=float)))
            if distance > robot.geometry.reach_radius_m:
                violations.append(_violation(
                    "reachability", "object_out_of_reach",
                    f"shared.objects.{name}.nominal.pose",
                    f"object is {distance:.3f} m from the fixed robot base; "
                    f"reach radius is {robot.geometry.reach_radius_m:.3f} m"))
    return GateResult("reachability", not violations, tuple(violations), {
        "robot_capability_id": robot.capability_id,
        "max_step_height_m": robot.geometry.max_step_height_m,
        "max_gap_m": robot.geometry.max_gap_m,
        "reach_radius_m": robot.geometry.reach_radius_m,
    })


def run_admission_gates(
    world: Mapping[str, Any], task: Mapping[str, Any], *,
    materialize_dir: Path | str | None = None,
    settle_steps: int = 120, penetration_tolerance_m: float = 0.02,
    sink_z_m: float = -2.0,
    runtime_task_id: str | None = None,
) -> tuple[AdmissionReport, CompiledWorld | None]:
    """Run the complete bounded gate chain in the current CPU process."""
    schema = _schema_gate(world, task)
    capability = _capability_gate(world, task)
    budget = _budget_gate(world, task)
    compiled: CompiledWorld | None = None
    build_error: str | None = None
    if schema.ok and capability.ok and budget.ok:
        try:
            compiled = compile_world(
                world, task, materialize_dir=materialize_dir,
                runtime_task_id=runtime_task_id)
        except (WorldCompileError, RuntimeError, ValueError) as exc:
            build_error = str(exc)
    else:
        build_error = "build skipped because schema, capability, or budget gate failed"
    build = _build_gate(compiled, build_error)
    penetration = _initial_penetration_gate(
        compiled, tolerance_m=penetration_tolerance_m)
    settle = _settle_gate(compiled, steps=max(1, min(settle_steps, 2000)),
                          sink_z_m=sink_z_m)
    placement = _placement_gate(world, task, compiled)
    reachability = _reachability_gate(world, task)
    gates = (schema, capability, budget, build, penetration, settle,
             placement, reachability)
    report = AdmissionReport(
        report_version=1, ok=all(gate.ok for gate in gates), gates=gates)
    if compiled is not None:
        compiled.resolved_eval = compiled.resolved_eval.with_admission(
            report.to_dict())
    return report, compiled


def _isolated_worker(
    output: Any, world: Mapping[str, Any], task: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> None:
    try:
        report, _ = run_admission_gates(world, task, **kwargs)
        output.put(("ok", report.to_dict()))
    except BaseException:  # subprocess boundary must return actionable detail
        output.put(("error", traceback.format_exc()))


def run_admission_gates_isolated(
    world: Mapping[str, Any], task: Mapping[str, Any], *,
    timeout_s: float = 90.0, **kwargs: Any,
) -> AdmissionReport:
    """Run admission in a fresh process with a hard wall-clock timeout."""
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_isolated_worker,
        args=(output, copy.deepcopy(world), copy.deepcopy(task), kwargs),
        daemon=True)
    process.start()
    process.join(timeout=max(1.0, timeout_s))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        violation = _violation(
            "subprocess", "admission_timeout", "world",
            f"admission exceeded {timeout_s:.1f} seconds", timeout_s=timeout_s)
        return AdmissionReport(1, False, (
            GateResult("subprocess", False, (violation,)),))
    try:
        status, payload = output.get_nowait()
    except queue.Empty:
        status, payload = "error", (
            f"admission subprocess exited {process.exitcode} without a report")
    if status == "ok":
        return AdmissionReport.from_dict(payload)
    violation = _violation(
        "subprocess", "admission_crashed", "world", str(payload),
        exit_code=process.exitcode)
    return AdmissionReport(1, False, (
        GateResult("subprocess", False, (violation,)),))
