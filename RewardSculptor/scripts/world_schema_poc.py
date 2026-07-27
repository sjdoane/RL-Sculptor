"""scripts/world_schema_poc.py — environment-authoring architecture PoC.

NOT the implementation. Validates three claims of
docs/internal/ENV_AUTHORING_ARCHITECTURE.md against the INSTALLED
mjlab 1.3.0, config-only (no training, no GPU requirement):

  (a) a WorldSpec-shaped dict compiles into mjlab's existing
      TerrainGeneratorCfg / TerrainEntityCfg / EntityCfg / SceneCfg
      machinery: generated rough/random-box terrain + a free ball
      + a fixed box entity + a goal-zone site, all from config;
  (b) the settle gate catches a deliberately-broken variant (ball
      spawned intersecting the fixed goal box → penetration rejection);
  (c) the schema→cfg mapping has no impedance mismatch (names, fields,
      attach prefixes) — printed inventory proves the scene contains
      what the spec says.

Run:  .venv/bin/python scripts/world_schema_poc.py
"""

from __future__ import annotations

import numpy as np

# ── the hand-written WorldSpec (schema per architecture §5) ─────────────
WORLD = {
    "world_spec_version": 2,
    "meta": {"source": "hand", "prompt": "PoC: rough terrain + free ball + goal region"},
    "shared": {
        "eval_seed": 1729,
        "terrain": {
            "kind": "generator",
            "layout": {"mode": "sampled_grid", "rows": 2, "cols": 2,
                       "tile_size_m": [8.0, 8.0], "border_width_m": 2.0},
            "evaluation_difficulty": 0.5,
            "sub_terrains": {
                "rough": {"type": "hf_random_uniform", "proportion": 0.5,
                          "nominal": {"noise_range_m": [0.02, 0.08],
                                      "noise_step_m": 0.02}},
                "boxes": {"type": "box_random_spread", "proportion": 0.5,
                          "nominal": {"num_boxes": 3,
                                      "box_height_m": [0.15, 0.35]}},
            },
        },
        "objects": {
            "ball": {"shape": "sphere",
                     "nominal": {"radius_m": 0.11, "mass_kg": 0.45,
                                 "pose": {"position_m": [1.0, 0.0, 0.6]}}},
            "goal_box": {"shape": "box", "fixed": True,
                         "nominal": {"size_m": [0.05, 0.9, 0.5],
                                     "pose": {
                                         "position_m": [3.0, 0.0, 0.5]}}},
        },
        "zones": {
            "goal_mouth": {"kind": "box", "center_m": [2.8, 0.0, 0.25],
                           "size_m": [0.2, 0.9, 0.25]},
        },
    },
    "train": {
        "curriculum": {"difficulty_range": [0.2, 0.8]},
        "variations": [],
    },
}


# ── minimal validator (bounds + closed vocab; architecture §7.4 gate 1) ──
TERRAIN_TYPES = {"hf_random_uniform", "box_random_spread"}  # PoC subset

def validate(world: dict) -> list[str]:
    errs: list[str] = []
    shared = world.get("shared", {})
    t = shared.get("terrain", {})
    for name, st in t.get("sub_terrains", {}).items():
        nominal = st.get("nominal", {})
        if st["type"] not in TERRAIN_TYPES:
            errs.append(f"terrain.sub_terrains.{name}.type {st['type']!r} "
                        f"not in {sorted(TERRAIN_TYPES)}")
        nr = nominal.get("noise_range_m")
        if nr and not (0.0 <= nr[0] < nr[1] <= 0.35):
            errs.append(f"{name}.noise_range_m {nr} outside [0, 0.35]")
        bh = nominal.get("box_height_m")
        if bh and not (0.0 <= bh[0] < bh[1] <= 0.6):
            errs.append(f"{name}.box_height_m {bh} outside [0, 0.6]")
    curriculum = world.get("train", {}).get("curriculum", {})
    lo, hi = curriculum.get("difficulty_range", [0, 1])
    ed = t.get("evaluation_difficulty", 0.5)
    if not (0 <= lo < hi <= 1):
        errs.append(f"difficulty_range [{lo},{hi}] must be within [0,1]")
    if not (0 <= ed <= 1):
        errs.append(f"evaluation_difficulty {ed} outside [0,1]")
    for oname, o in shared.get("objects", {}).items():
        if o.get("shape") not in ("sphere", "box"):
            errs.append(f"objects.{oname}.shape {o.get('shape')!r} unsupported")
        if not isinstance(o.get("nominal"), dict):
            errs.append(f"objects.{oname}.nominal must be a mapping")
    return errs


# ── compiler (architecture §7.1/§7.2): WorldSpec → mjlab cfg tree ────────
def compile_world(world: dict):
    import mujoco
    from mjlab.entity import EntityCfg
    from mjlab.scene import SceneCfg
    from mjlab.terrains import (
        BoxRandomSpreadTerrainCfg,
        HfRandomUniformTerrainCfg,
        TerrainEntityCfg,
        TerrainGeneratorCfg,
    )

    shared = world["shared"]
    t = shared["terrain"]

    def sub_cfg(st: dict):
        layout = t["layout"]
        nominal = st["nominal"]
        size = tuple(layout["tile_size_m"])
        if st["type"] == "hf_random_uniform":
            return HfRandomUniformTerrainCfg(
                proportion=st["proportion"], size=size,
                noise_range=tuple(nominal["noise_range_m"]),
                noise_step=nominal["noise_step_m"])
        if st["type"] == "box_random_spread":
            return BoxRandomSpreadTerrainCfg(
                proportion=st["proportion"], size=size,
                num_boxes=nominal["num_boxes"],
                box_height_range=tuple(nominal["box_height_m"]))
        raise ValueError(st["type"])

    terrain_cfg = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=int(shared["eval_seed"]),
            size=tuple(t["layout"]["tile_size_m"]),
            num_rows=t["layout"]["rows"], num_cols=t["layout"]["cols"],
            border_width=t["layout"]["border_width_m"],
            sub_terrains={k: sub_cfg(v)
                          for k, v in t["sub_terrains"].items()},
            difficulty_range=(
                t["evaluation_difficulty"], t["evaluation_difficulty"]),
        ),
        num_envs=1,
    )

    def object_entity(name: str, o: dict) -> EntityCfg:
        def spec_fn(o=o, name=name):
            nominal = o["nominal"]
            spec = mujoco.MjSpec()
            body = spec.worldbody.add_body(name=name)
            if not o.get("fixed"):
                body.add_freejoint()
            if o["shape"] == "sphere":
                body.add_geom(
                    name=f"{name}_geom",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[nominal["radius_m"], 0, 0],
                    mass=nominal.get("mass_kg", 0.1))
            else:
                sx, sy, sz = nominal["size_m"]
                body.add_geom(
                    name=f"{name}_geom",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[sx / 2, sy / 2, sz / 2],
                    mass=nominal.get("mass_kg", 1.0))
            return spec

        return EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=tuple(o["nominal"]["pose"]["position_m"])),
            spec_fn=spec_fn,
        )

    entities = {name: object_entity(name, o)
                for name, o in shared.get("objects", {}).items()}

    def add_zones(spec: "mujoco.MjSpec") -> None:
        for zname, z in shared.get("zones", {}).items():
            sx, sy, sz = z["size_m"]
            spec.worldbody.add_site(
                name=f"zone_{zname}",
                pos=list(z["center_m"]),
                size=[sx / 2, sy / 2, sz / 2],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                group=4,  # visual-only group
                rgba=[0.1, 0.9, 0.1, 0.25],
            )

    scene_cfg = SceneCfg(
        num_envs=1,
        terrain=terrain_cfg,
        entities=entities,
        spec_fn=add_zones,
    )
    return scene_cfg


# ── gates 3+4: build + settle (architecture §7.4) ───────────────────────
# PoC design lesson (recorded in the architecture doc §14): GLOBAL
# QUIESCENCE is the wrong settle criterion — a sphere on rough terrain
# legitimately never stops rolling. The discriminating checks are:
#   (i)  init-penetration: after mj_forward at t=0, no contact deeper
#        than a tolerance (catches objects spawned inside geometry);
#   (ii) finite state + no-sink over N steps (catches solver blowups
#        and fall-through).
def build_and_settle(scene_cfg, *, settle_steps=300, pen_tol_m=0.02,
                     sink_z_m=-2.0):
    import mujoco
    from mjlab.scene import Scene

    scene = Scene(scene_cfg, device="cpu")
    spec = scene.spec
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.mj_forward(model, data)
    # (i) init-penetration gate
    worst_pen = 0.0
    for i in range(data.ncon):
        worst_pen = max(worst_pen, float(-data.contact[i].dist))
    if worst_pen > pen_tol_m:
        return scene, model, data, {
            "ok": False,
            "reason": f"init penetration {worst_pen * 100:.1f} cm > "
                      f"{pen_tol_m * 100:.0f} cm (object spawned inside "
                      f"geometry)",
            "init_penetration_m": worst_pen,
        }

    # (ii) finite + no-sink over N steps
    for i in range(settle_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(
                np.isfinite(data.qvel)):
            return scene, model, data, {"ok": False,
                                        "reason": f"NaN at step {i}"}
    free_z = [float(data.body(b).xpos[2]) for b in range(model.nbody)
              if model.body(b).jntnum[0] > 0]
    if free_z and min(free_z) < sink_z_m:
        return scene, model, data, {
            "ok": False,
            "reason": f"object sank to z={min(free_z):.2f} (fell through "
                      f"terrain)"}
    return scene, model, data, {
        "ok": True, "reason": None,
        "init_penetration_m": worst_pen,
        "max_qvel_end": (float(np.max(np.abs(data.qvel)))
                         if data.qvel.size else 0.0),
    }


def inventory(model) -> dict:
    import mujoco
    names = {
        "bodies": [model.body(i).name for i in range(model.nbody)],
        "geoms": model.ngeom,
        "hfields": model.nhfield,
        "sites": [model.site(i).name for i in range(model.nsite)],
        "nq": model.nq,
    }
    return names


def main() -> int:
    import importlib.metadata

    print(f"[poc] mjlab {importlib.metadata.version('mjlab')}, "
          f"mujoco {importlib.metadata.version('mujoco')}")

    # gate 1: validation
    errs = validate(WORLD)
    print(f"[poc] validate: {'OK' if not errs else errs}")
    assert not errs

    # broken variant must FAIL validation (bounds)
    bad = {**WORLD, "shared": {**WORLD["shared"], "terrain": {
        **WORLD["shared"]["terrain"], "evaluation_difficulty": 1.7}}}
    errs = validate(bad)
    print(f"[poc] validate(broken bounds): rejected = {bool(errs)} ({errs})")
    assert errs

    # gates 3+4 on the good spec
    scene_cfg = compile_world(WORLD)
    scene, model, data, settle = build_and_settle(scene_cfg)
    inv = inventory(model)
    print(f"[poc] build: nbody={len(inv['bodies'])} ngeom={inv['geoms']} "
          f"nhfield={inv['hfields']} nq={inv['nq']}")
    print(f"[poc] bodies: {[b for b in inv['bodies'] if b]}")
    print(f"[poc] sites:  {inv['sites']}")
    print(f"[poc] settle: {settle}")
    assert inv["hfields"] >= 1, "generated heightfield missing"
    assert any("ball" in b for b in inv["bodies"]), "ball entity missing"
    assert any("goal_box" in b for b in inv["bodies"]), "fixed goal entity missing"
    assert any("goal_mouth" in s for s in inv["sites"]), "zone site missing"
    assert settle["ok"], f"settle gate failed on GOOD spec: {settle}"

    # (b) deliberately-broken variant: ball inside the fixed goal box;
    # the penetration gate must reject the initial placement.
    broken = {**WORLD, "shared": {**WORLD["shared"], "objects": {
        **WORLD["shared"]["objects"],
        "ball": {**WORLD["shared"]["objects"]["ball"],
                 "nominal": {**WORLD["shared"]["objects"]["ball"]["nominal"],
                             "pose": WORLD["shared"]["objects"]["goal_box"]
                             ["nominal"]["pose"]}},
    }}}
    scene_cfg_b = compile_world(broken)
    _, model_b, _, settle_b = build_and_settle(scene_cfg_b, settle_steps=60)
    print(f"[poc] settle(broken overlap): {settle_b}")
    assert not settle_b["ok"], (
        "settle gate FAILED to catch the overlapping-spawn variant")

    print("[poc] ALL CHECKS PASSED — schema→mjlab compile path is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
