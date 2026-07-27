"""Compiled-scene → JSON scene graph for the in-browser 3D world viewer.

The exporter walks a compiled MuJoCo model (the materialized evaluation
MJB of a promoted selection, or the in-memory model of a draft dry-run)
posed at keyframe 0, and emits primitives + meshes + heightfields with
world-frame transforms — the exact scene evaluation replays, never a
re-derivation from the spec. Semantic entities (course elements,
objects, zones, robot, terrain) are indexed from the WorldSpec by the
compiler's stable naming conventions:

  - course geoms:  ``obstacle__<stable_id>__<element>``
  - zone sites:    ``zone__<name>``
  - robot bodies:  ``robot/<body>``
  - object bodies: ``<object_name>`` or ``<object_name>/<body>``
  - terrain:       body/geom ``terrain``

Gaps are course elements with no geometry. For legibility the exporter
emits a synthetic translucent *marker* box spanning the neighbouring
course geoms — clearly tagged ``marker: true`` so the viewer can label
it as spacing, not a solid.
"""
from __future__ import annotations

from typing import Any, Mapping

_ROUND_POS = 5
_ROUND_VERT = 4

#: mujoco geom-type id → viewer primitive name (mjtGeom order).
_GEOM_TYPES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
    4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh",
}
_SITE_TYPES = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}


def _round_list(values: Any, ndigits: int) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _quat_from_xmat(xmat: Any) -> list[float]:
    import mujoco
    import numpy as np

    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=np.float64).reshape(9))
    return _round_list(quat, _ROUND_POS)  # [w, x, y, z]


def _entity_for_geom(geom_name: str, body_name: str) -> tuple[str, str]:
    """(kind, entity_id) classification for one geom."""
    if geom_name.startswith("obstacle__"):
        parts = geom_name.split("__")
        return "course_element", (parts[1] if len(parts) > 1 else geom_name)
    if body_name == "robot" or body_name.startswith("robot/"):
        return "robot", "robot"
    if body_name in ("terrain", "world") and (
            geom_name in ("terrain", "") or geom_name.startswith("terrain")):
        return "terrain", "terrain"
    if "/" in body_name:
        return "object", body_name.split("/", 1)[0]
    if body_name in ("authored_course",):
        return "course_element", geom_name or body_name
    return "object", body_name or geom_name or "unnamed"


def scene_from_model(
    model: Any, world: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize a compiled model (posed at keyframe 0) + WorldSpec
    semantics into the viewer scene graph."""
    import mujoco

    from backend.services.preview_renderer import _posed_data

    data = _posed_data(model)

    def _name(objtype: Any, index: int) -> str:
        return mujoco.mj_id2name(model, objtype, index) or ""

    # ── meshes (deduplicated; geoms reference them by name) ──────────
    meshes: dict[str, dict[str, Any]] = {}
    for mesh_id in range(model.nmesh):
        name = _name(mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh_{mesh_id}"
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vert_start:vert_start + vert_count]
        faces = model.mesh_face[face_start:face_start + face_count]
        meshes[name] = {
            "vertices": [round(float(v), _ROUND_VERT)
                         for v in vertices.reshape(-1)],
            "faces": [int(f) for f in faces.reshape(-1)],
        }

    # ── heightfields ─────────────────────────────────────────────────
    hfields: dict[str, dict[str, Any]] = {}
    for hfield_id in range(model.nhfield):
        name = (_name(mujoco.mjtObj.mjOBJ_HFIELD, hfield_id)
                or f"hfield_{hfield_id}")
        nrow = int(model.hfield_nrow[hfield_id])
        ncol = int(model.hfield_ncol[hfield_id])
        adr = int(model.hfield_adr[hfield_id])
        hfields[name] = {
            "nrow": nrow,
            "ncol": ncol,
            # [radius_x, radius_y, elevation_z, base_z]
            "size": _round_list(model.hfield_size[hfield_id], _ROUND_POS),
            "data": [round(float(v), _ROUND_VERT)
                     for v in model.hfield_data[adr:adr + nrow * ncol]],
        }

    # ── geoms (world-frame pose from the keyframe-posed data) ────────
    geoms: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        geom_name = _name(mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = _name(
            mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        type_id = int(model.geom_type[geom_id])
        kind, entity_id = _entity_for_geom(geom_name, body_name)
        entry: dict[str, Any] = {
            "name": geom_name or f"geom_{geom_id}",
            "body": body_name,
            "type": _GEOM_TYPES.get(type_id, f"type_{type_id}"),
            "size": _round_list(model.geom_size[geom_id], _ROUND_POS),
            "pos": _round_list(data.geom_xpos[geom_id], _ROUND_POS),
            "quat": _quat_from_xmat(data.geom_xmat[geom_id]),
            "rgba": _round_list(model.geom_rgba[geom_id], 4),
            "group": int(model.geom_group[geom_id]),
            "entity_kind": kind,
            "entity_id": entity_id,
        }
        if entry["type"] == "mesh":
            mesh_id = int(model.geom_dataid[geom_id])
            entry["mesh"] = (_name(mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                             or f"mesh_{mesh_id}")
        elif entry["type"] == "hfield":
            hfield_id = int(model.geom_dataid[geom_id])
            entry["hfield"] = (_name(mujoco.mjtObj.mjOBJ_HFIELD, hfield_id)
                               or f"hfield_{hfield_id}")
        geoms.append(entry)

    # ── zone/region sites ────────────────────────────────────────────
    sites: list[dict[str, Any]] = []
    for site_id in range(model.nsite):
        site_name = _name(mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not site_name.startswith("zone__"):
            continue
        sites.append({
            "name": site_name,
            "type": _SITE_TYPES.get(int(model.site_type[site_id]), "box"),
            "size": _round_list(model.site_size[site_id], _ROUND_POS),
            "pos": _round_list(data.site_xpos[site_id], _ROUND_POS),
            "quat": _quat_from_xmat(data.site_xmat[site_id]),
            "rgba": _round_list(model.site_rgba[site_id], 4),
            "entity_kind": "zone",
            "entity_id": site_name.removeprefix("zone__"),
        })

    entities = _entity_index(world, geoms, sites)
    return {
        "meshes": meshes,
        "hfields": hfields,
        "geoms": geoms,
        "sites": sites,
        "entities": entities,
    }


def _variations_by_target_id(world: Mapping[str, Any]) -> dict[str, list]:
    """train.variations grouped by the stable id their pointer targets
    (@<id> course syntax, or the second-to-last path segment for
    terrain/object paths)."""
    grouped: dict[str, list] = {}
    for variation in (world.get("train") or {}).get("variations") or []:
        target = str(variation.get("target") or "")
        entity_id = None
        if "@" in target:
            entity_id = target.split("@", 1)[1].split("/", 1)[0]
        else:
            parts = [p for p in target.split("/") if p]
            for anchor in ("objects", "zones", "sub_terrains"):
                if anchor in parts:
                    idx = parts.index(anchor)
                    if idx + 1 < len(parts):
                        entity_id = parts[idx + 1]
                    break
        key = entity_id or "__world__"
        grouped.setdefault(key, []).append({
            "id": variation.get("id"),
            "target": target,
            "class": variation.get("class"),
            "distribution": variation.get("distribution"),
        })
    return grouped


def _entity_index(
    world: Mapping[str, Any],
    geoms: list[dict[str, Any]],
    sites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One inspectable entry per semantic entity, parameters straight
    from the authored WorldSpec (the artifact IS the source of truth)."""
    shared = world.get("shared") or {}
    variations = _variations_by_target_id(world)
    geoms_of = lambda kind, eid: [  # noqa: E731
        g["name"] for g in geoms
        if g["entity_kind"] == kind and g["entity_id"] == eid]

    entities: list[dict[str, Any]] = []

    course = (shared.get("obstacles") or {}).get("course") or []
    for position, element in enumerate(course):
        element_id = str(element.get("id"))
        element_kind = str(element.get("element"))
        entry: dict[str, Any] = {
            "kind": "course_element",
            "id": element_id,
            "label": f"{element_id} ({element_kind})",
            "element": element_kind,
            "params": dict(element.get("nominal") or {}),
            "variations": variations.get(element_id, []),
            "geoms": geoms_of("course_element", element_id),
        }
        if not entry["geoms"]:
            marker = _gap_marker(course, position, geoms)
            if marker is not None:
                entry["marker"] = marker
        entities.append(entry)

    for name, obj in sorted((shared.get("objects") or {}).items()):
        entities.append({
            "kind": "object",
            "id": name,
            "label": name,
            "params": dict((obj or {}).get("nominal") or {}),
            "variations": variations.get(name, []),
            "geoms": geoms_of("object", name),
        })

    for name, zone in sorted((shared.get("zones") or {}).items()):
        entities.append({
            "kind": "zone",
            "id": name,
            "label": name,
            "params": {k: v for k, v in (zone or {}).items()},
            "variations": variations.get(name, []),
            "geoms": [],
            "sites": [s["name"] for s in sites if s["entity_id"] == name],
        })

    terrain = shared.get("terrain") or {}
    entities.append({
        "kind": "terrain",
        "id": "terrain",
        "label": f"terrain ({terrain.get('kind', 'plane')})",
        "params": {
            key: value for key, value in terrain.items() if key != "kind"},
        "variations": [
            v for eid, vs in variations.items()
            if eid in ((terrain.get("sub_terrains") or {}).keys())
            for v in vs],
        "geoms": geoms_of("terrain", "terrain"),
    })

    robot = shared.get("robot") or {}
    entities.append({
        "kind": "robot",
        "id": "robot",
        "label": str(robot.get("capability_id") or "robot"),
        "params": dict(robot),
        "variations": [],
        "geoms": geoms_of("robot", "robot"),
    })
    return entities


def _gap_marker(
    course: list, position: int, geoms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Translucent marker box spanning the previous and next course geoms
    along x — presentation only, clearly tagged as a marker."""
    def _bounds(index: int) -> tuple[float, float, float] | None:
        if not 0 <= index < len(course):
            return None
        element_id = str(course[index].get("id"))
        best = None
        for geom in geoms:
            if (geom["entity_kind"] == "course_element"
                    and geom["entity_id"] == element_id
                    and geom["type"] == "box"):
                x_low = geom["pos"][0] - geom["size"][0]
                x_high = geom["pos"][0] + geom["size"][0]
                top = geom["pos"][2] + geom["size"][2]
                if best is None:
                    best = [x_low, x_high, top]
                else:
                    best[0] = min(best[0], x_low)
                    best[1] = max(best[1], x_high)
                    best[2] = max(best[2], top)
        return tuple(best) if best else None

    before = _bounds(position - 1)
    after = _bounds(position + 1)
    if before is None or after is None:
        return None
    x_start, x_end = before[1], after[0]
    if x_end <= x_start:
        return None
    top = max(before[2], after[2])
    return {
        "pos": [round((x_start + x_end) / 2, _ROUND_POS), 0.0,
                round(top / 2, _ROUND_POS)],
        "size": [round((x_end - x_start) / 2, _ROUND_POS), 0.5,
                 round(max(top / 2, 0.02), _ROUND_POS)],
    }
