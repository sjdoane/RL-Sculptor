"""Generic manipulation telemetry for the mjlab rollout artifact contract.

Registered (non-authored) mjlab tasks — the external benchmark surface —
previously persisted no end-effector pose, object state, gripper
contact, grasp evidence, or target identity, which is exactly why the
YAM manifests are frozen at ``compile_only``. This module closes that
gap **generically**:

- **No robot or task name keying.** The robot is the scene entity whose
  config declares actuators; objects are the passive entities. The
  end-effector, finger groups, and grasp eligibility come from the
  installed :mod:`sculptor.world.capabilities` descriptors, matched
  against the compiled robot asset by semantic-role name sets. A future
  arm inherits all of this by registering a descriptor — never by
  editing this file.
- **Objects are embodiment-neutral.** Object pose/velocity channels are
  recorded for every robot (a quadruped pushing a ball needs them as
  much as a gripper arm); gripper-contact and grasp channels appear only
  when the matched capability actually advertises finger roles and the
  ``grasp`` capability.
- **Honest signals only.** ``grasp__<object>`` is bilateral finger
  contact (left AND right finger groups simultaneously touching the
  object) — a mechanical proxy recorded as evidence, not a claim of
  force-closure. The sidecar manifest states this derivation explicitly
  so a success specification or auditor can weigh it.
- **Target identity is recorded, not inferred.** The task's command term
  is duck-typed against the documented mjlab API: a term carrying
  ``target_selection`` plus ``cfg.entity_names`` yields a per-step
  target index (multi-object); a term whose ``cfg.entity_name`` names a
  discovered object yields a constant index. The 3D command tensor is
  persisted as ``target__pos_w``.

Array naming follows the authored-world ChannelCatalog vocabulary
(``object__<name>__pos_w`` …) so metrics and specs share one vocabulary
across authored and registered tasks. Authored runs keep their catalog
recorder; this module is only engaged when no world selection is pinned,
so the two can never double-write.

Everything here is fail-soft: any discovery or per-step failure degrades
to "channel absent" (feature-detect discipline, mirroring the foot
channels) rather than breaking a rollout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TELEMETRY_MANIFEST_NAME = "manipulation_telemetry.json"
TELEMETRY_SCHEMA_VERSION = 1

#: Candidate mjlab data attributes per root-state suffix (API drift across
#: versions — same tolerance set as the world channel runtime).
_ROOT_FIELDS: dict[str, tuple[str, ...]] = {
    "pos_w": ("root_link_pos_w", "root_pos_w", "body_pos_w"),
    "quat_w": ("root_link_quat_w", "root_quat_w", "body_quat_w"),
    "lin_vel_w": ("root_link_lin_vel_w", "root_lin_vel_w", "body_lin_vel_w"),
    "ang_vel_w": ("root_link_ang_vel_w", "root_ang_vel_w", "body_ang_vel_w"),
}


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    try:  # torch tensor
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:  # noqa: BLE001
        try:
            return np.asarray(value, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None


def _first_attr(data: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(data, name, None)
        if value is not None:
            return value
    return None


# ── discovery ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ManipulationDiscovery:
    """What the scene offers, resolved once per rollout."""

    robot_entity: str
    object_names: tuple[str, ...]          # deterministic (cfg order)
    capability_id: str | None = None
    ee_site: str | None = None             # tool_center → grasp fallback
    ee_body: str | None = None             # end_effector role body
    finger_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict)              # {"left": (...), "right": (...)}
    grasp_capable: bool = False
    sensor_names: tuple[str, ...] = ()     # injected contact sensors

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "robot_entity": self.robot_entity,
            "object_names": list(self.object_names),
            "capability_id": self.capability_id,
            "ee_site": self.ee_site,
            "ee_body": self.ee_body,
            "finger_groups": {
                group: list(bodies)
                for group, bodies in sorted(self.finger_groups.items())},
            "grasp_capable": self.grasp_capable,
            "contact_sensors": list(self.sensor_names),
            "derivations": {
                "grasp": (
                    "bilateral finger contact: left AND right finger-group "
                    "contact with the object in the same step — a mechanical "
                    "grasp proxy, not force-closure verification"),
                "target_object_index": (
                    "per-step index into object_names taken from the task "
                    "command term (target_selection for multi-object "
                    "commands; the command's static entity_name otherwise); "
                    "-1 when the task declares no object target"),
            },
        }


def _robot_model_names(entity_cfg: Any) -> tuple[set[str], set[str]]:
    """Body and site name sets of one entity cfg, via a CPU spec compile."""
    import mujoco

    spec = entity_cfg.spec_fn()
    model = spec.compile()
    bodies = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)}
    sites = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)}
    bodies.discard(None)
    sites.discard(None)
    return bodies, sites


def _match_capability(bodies: set[str], sites: set[str]) -> Any | None:
    """Best installed RobotCapability whose EVERY declared role name exists
    in the compiled robot asset; ties break on capability_id so matching is
    deterministic. None when no descriptor fits (telemetry then records
    objects only)."""
    from sculptor.world.capabilities import robot_capabilities

    best = None
    best_score = -1
    for capability_id in sorted(robot_capabilities()):
        capability = robot_capabilities()[capability_id]
        role_bodies = [
            name for names in capability.body_roles.values() for name in names]
        role_sites = [
            name
            for names in getattr(capability, "site_roles", {}).values()
            for name in names]
        if not role_bodies:
            continue
        if not set(role_bodies) <= bodies:
            continue
        if not set(role_sites) <= sites:
            continue
        score = len(role_bodies) + len(role_sites)
        if score > best_score:
            best, best_score = capability, score
    return best


def discover_from_cfg(env_cfg: Any) -> ManipulationDiscovery | None:
    """Classify scene entities and resolve manipulation roles, before env
    construction (so contact sensors can still be injected).

    Robot = the entity whose cfg declares an articulation with actuators;
    objects = every passive entity. Returns None when the scene has no
    passive objects (nothing to record) or no unambiguous robot.
    """
    entities = dict(getattr(env_cfg.scene, "entities", None) or {})
    if not entities:
        return None

    def _actuated(cfg: Any) -> bool:
        articulation = getattr(cfg, "articulation", None)
        if articulation is None:
            return False
        actuators = getattr(articulation, "actuators", None)
        return bool(actuators)

    robots = [name for name, cfg in entities.items() if _actuated(cfg)]
    if len(robots) != 1:
        return None  # no robot, or ambiguous — do not guess
    robot_name = robots[0]
    object_names = tuple(
        name for name in entities if name != robot_name)
    if not object_names:
        return None

    capability = None
    try:
        bodies, sites = _robot_model_names(entities[robot_name])
        capability = _match_capability(bodies, sites)
    except Exception:  # noqa: BLE001 — descriptor matching is optional
        capability = None

    ee_site = ee_body = None
    finger_groups: dict[str, tuple[str, ...]] = {}
    grasp_capable = False
    capability_id = None
    if capability is not None:
        capability_id = capability.capability_id
        site_roles = dict(getattr(capability, "site_roles", {}) or {})
        for role in ("tool_center", "grasp"):
            names = site_roles.get(role)
            if names:
                ee_site = names[0]
                break
        ee_bodies = capability.body_roles.get("end_effector")
        if ee_bodies:
            ee_body = ee_bodies[0]
        for group, role in (("left", "left_finger"), ("right", "right_finger")):
            names = capability.body_roles.get(role)
            if names:
                finger_groups[group] = tuple(names)
        if not finger_groups:
            gripper = capability.body_roles.get("gripper")
            if gripper:
                finger_groups["gripper"] = tuple(gripper)
        grasp_capable = (
            "grasp" in capability.capabilities
            and "left" in finger_groups and "right" in finger_groups)

    return ManipulationDiscovery(
        robot_entity=robot_name,
        object_names=object_names,
        capability_id=capability_id,
        ee_site=ee_site,
        ee_body=ee_body,
        finger_groups=finger_groups,
        grasp_capable=grasp_capable,
    )


def inject_contact_sensors(
    env_cfg: Any, discovery: ManipulationDiscovery,
) -> ManipulationDiscovery:
    """Declare finger↔object contact sensors on the scene cfg (must run
    BEFORE env construction). Sensors are additive, uniquely named, and
    read-only — they observe contacts, they cannot create them. Returns
    the discovery updated with the injected sensor names."""
    if not discovery.finger_groups:
        return discovery
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    sensors = list(env_cfg.scene.sensors or ())
    existing = {getattr(s, "name", "") for s in sensors}
    names: list[str] = []
    for group, bodies in sorted(discovery.finger_groups.items()):
        for obj in discovery.object_names:
            name = f"manip_contact__{group}__{obj}"
            if name in existing:
                continue
            sensors.append(ContactSensorCfg(
                name=name,
                primary=ContactMatch(
                    mode="subtree", pattern=tuple(bodies),
                    entity=discovery.robot_entity),
                secondary=ContactMatch(mode="body", pattern=".*", entity=obj),
                fields=("found",),
                reduce="maxforce",
                num_slots=1,
            ))
            names.append(name)
    env_cfg.scene.sensors = tuple(sensors)
    return ManipulationDiscovery(
        robot_entity=discovery.robot_entity,
        object_names=discovery.object_names,
        capability_id=discovery.capability_id,
        ee_site=discovery.ee_site,
        ee_body=discovery.ee_body,
        finger_groups=discovery.finger_groups,
        grasp_capable=discovery.grasp_capable,
        sensor_names=tuple(names),
    )


# ── target-command resolution (duck-typed on the mjlab manager API) ───
def _command_terms(env: Any) -> list[Any]:
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return []
    terms = getattr(manager, "_terms", None) or getattr(manager, "terms", None)
    if isinstance(terms, Mapping):
        return list(terms.values())
    if isinstance(terms, (list, tuple)):
        return list(terms)
    return []


def _resolve_target_source(
    env: Any, object_names: tuple[str, ...],
) -> tuple[Any, int | None] | None:
    """(command_term, static_index) — static_index None means the term
    carries a per-step ``target_selection``; -1 static means positions only."""
    for term in _command_terms(env):
        cfg = getattr(term, "cfg", None)
        command = getattr(term, "command", None)
        if command is None:
            continue
        entity_names = tuple(getattr(cfg, "entity_names", ()) or ())
        if (entity_names and hasattr(term, "target_selection")
                and set(entity_names) <= set(object_names)):
            return term, None
        entity_name = getattr(cfg, "entity_name", None)
        if entity_name in object_names:
            return term, object_names.index(entity_name)
    return None


# ── per-step recorder ─────────────────────────────────────────────────
class ManipulationRecorder:
    """Append-per-step buffers for the discovered manipulation channels.

    Every channel is independently feature-detected at construction and
    independently guarded per step; a channel whose buffer ends up empty
    or shape-inconsistent is dropped at finalize — identical discipline
    to the foot-contact channels.
    """

    def __init__(self, env: Any, discovery: ManipulationDiscovery) -> None:
        self.env = env
        self.discovery = discovery
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._objects: dict[str, Any] = {}
        for name in discovery.object_names:
            try:
                self._objects[name] = env.scene[name]
            except Exception:  # noqa: BLE001
                continue
        self._robot = None
        try:
            self._robot = env.scene[discovery.robot_entity]
        except Exception:  # noqa: BLE001
            pass
        self._ee_site_index: int | None = None
        if self._robot is not None and discovery.ee_site:
            try:
                site_names = tuple(self._robot.site_names)
                if discovery.ee_site in site_names:
                    self._ee_site_index = site_names.index(discovery.ee_site)
            except Exception:  # noqa: BLE001
                self._ee_site_index = None
        self._ee_body_index: int | None = None
        if self._robot is not None and discovery.ee_body:
            try:
                body_names = tuple(self._robot.body_names)
                if discovery.ee_body in body_names:
                    self._ee_body_index = body_names.index(discovery.ee_body)
            except Exception:  # noqa: BLE001
                self._ee_body_index = None
        self._sensors: dict[str, Any] = {}
        for name in discovery.sensor_names:
            try:
                self._sensors[name] = env.scene[name]
            except Exception:  # noqa: BLE001
                continue
        self._target = _resolve_target_source(env, discovery.object_names)

    # ── helpers ───────────────────────────────────────────────────
    def _push(self, key: str, value: np.ndarray | None) -> None:
        if value is None:
            return
        self._buffers.setdefault(key, []).append(value)

    def _root_state(self, entity: Any, suffix: str) -> np.ndarray | None:
        data = getattr(entity, "data", None)
        if data is None:
            return None
        array = _to_numpy(_first_attr(data, _ROOT_FIELDS[suffix]))
        if array is None:
            return None
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0]
        return array

    def _sensor_contact(self, sensor: Any) -> np.ndarray | None:
        found = getattr(getattr(sensor, "data", None), "found", None)
        array = _to_numpy(found)
        if array is None:
            return None
        if array.ndim > 1:
            array = np.any(array > 0, axis=tuple(range(1, array.ndim)))
        else:
            array = array > 0
        return array.astype(np.float32, copy=False)

    # ── per step ──────────────────────────────────────────────────
    def append(self) -> None:
        discovery = self.discovery
        if self._robot is not None:
            data = getattr(self._robot, "data", None)
            if self._ee_site_index is not None:
                try:
                    sites = _to_numpy(getattr(data, "site_pos_w", None))
                    if sites is not None and sites.ndim == 3 \
                            and sites.shape[1] > self._ee_site_index:
                        self._push("ee_pos_w", sites[:, self._ee_site_index])
                except Exception:  # noqa: BLE001
                    pass
            if self._ee_body_index is not None:
                try:
                    quats = _to_numpy(_first_attr(
                        data, ("body_link_quat_w", "body_quat_w")))
                    if quats is not None and quats.ndim == 3 \
                            and quats.shape[1] > self._ee_body_index:
                        self._push("ee_quat_w", quats[:, self._ee_body_index])
                except Exception:  # noqa: BLE001
                    pass

        for name, entity in self._objects.items():
            for suffix in ("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"):
                try:
                    self._push(f"object__{name}__{suffix}",
                               self._root_state(entity, suffix))
                except Exception:  # noqa: BLE001
                    continue

        contact_by_object: dict[str, dict[str, np.ndarray]] = {}
        for sensor_name, sensor in self._sensors.items():
            try:
                value = self._sensor_contact(sensor)
            except Exception:  # noqa: BLE001
                value = None
            if value is None:
                continue
            # manip_contact__<group>__<object>
            parts = sensor_name.split("__")
            if len(parts) != 3:
                continue
            _, group, obj = parts
            self._push(f"contact__{group}__{obj}", value)
            contact_by_object.setdefault(obj, {})[group] = value
        if discovery.grasp_capable:
            for obj, groups in contact_by_object.items():
                left, right = groups.get("left"), groups.get("right")
                if left is not None and right is not None:
                    self._push(
                        f"grasp__{obj}",
                        ((left > 0) & (right > 0)).astype(np.float32))

        if self._target is not None:
            term, static_index = self._target
            try:
                command = _to_numpy(getattr(term, "command", None))
                if command is not None and command.ndim == 2 \
                        and command.shape[-1] == 3:
                    self._push("target__pos_w", command)
            except Exception:  # noqa: BLE001
                pass
            try:
                if static_index is None:
                    selection = getattr(term, "target_selection", None)
                    entity_names = tuple(term.cfg.entity_names)
                    raw = _to_numpy(selection)
                    if raw is not None:
                        local = raw.astype(np.int64, copy=False)
                        remap = np.asarray([
                            self.discovery.object_names.index(n)
                            for n in entity_names], dtype=np.int32)
                        valid = (local >= 0) & (local < len(remap))
                        indices = np.where(
                            valid, remap[np.clip(local, 0, len(remap) - 1)],
                            -1).astype(np.int32)
                        self._push("target_object_index",
                                   indices.astype(np.float32))
                else:
                    width = 1
                    command = _to_numpy(getattr(term, "command", None))
                    if command is not None and command.ndim >= 1:
                        width = command.shape[0]
                    self._push("target_object_index", np.full(
                        (width,), float(static_index), dtype=np.float32))
            except Exception:  # noqa: BLE001
                pass

    # ── finalize ──────────────────────────────────────────────────
    def finalize(self) -> dict[str, np.ndarray]:
        import sys

        arrays: dict[str, np.ndarray] = {}
        for key, buffers in self._buffers.items():
            if not buffers:
                continue
            shape0 = buffers[0].shape
            if any(b.shape != shape0 for b in buffers):
                # Transparency over silence: a channel that vanishes must
                # leave a trace of why, or its absence reads as "never
                # discovered" instead of "recorded inconsistently".
                print(f"[manipulation-telemetry] channel {key!r} dropped: "
                      f"inconsistent per-step shapes", file=sys.stderr,
                      flush=True)
                continue
            stacked = np.stack(buffers, axis=0)
            if key == "target_object_index":
                stacked = stacked.astype(np.int32)
            arrays[key] = stacked
        return arrays

    def write_manifest(self, output_dir: Path,
                       arrays: Mapping[str, np.ndarray]) -> None:
        manifest = self.discovery.to_manifest()
        manifest["channels"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(arrays.items())}
        (Path(output_dir) / TELEMETRY_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8")
