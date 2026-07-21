"""Per-episode domain randomization (sculptor/world/randomization.py).

Everything here is CPU-verifiable: variation parsing, distribution→range
collapse, mjlab event-term wiring, and the box-height write math (exercised on a
real mj_model for name resolution + a fake tensor model bridge). The end-to-end
GPU reset is a separate live smoke test.
"""
from __future__ import annotations

import types

import pytest

from sculptor.world.randomization import (
    Randomization,
    _range_from_distribution,
    install_world_randomizations,
    resolve_world_randomizations,
)


def _parkour_world():
    return {"train": {"variations": [
        {"id": "middle_box_height",
         "target": "/shared/obstacles/course/@box_02/nominal/height_m",
         "class": "model_field",
         "distribution": {"kind": "uniform", "low": 0.15, "high": 0.35}},
        {"id": "second_gap_length",
         "target": "/shared/obstacles/course/@gap_02/nominal/length_m",
         "class": "model_field",
         "distribution": {"kind": "uniform", "low": 0.2, "high": 0.4}},
    ]}}


def _object_world():
    return {"train": {"variations": [
        {"id": "object_mass",
         "target": "/shared/objects/target_object/nominal/mass_kg",
         "distribution": {"kind": "uniform", "low": 0.12, "high": 0.32}},
        {"id": "object_friction",
         "target": "/shared/objects/target_object/nominal/friction",
         "distribution": {"kind": "uniform", "low": 0.5, "high": 1.1}},
    ]}}


# ── distribution → range ─────────────────────────────────────────────────────

def test_range_from_distribution():
    assert _range_from_distribution({"kind": "uniform", "low": 1, "high": 2}) == (1.0, 2.0)
    lo, hi = _range_from_distribution({"kind": "normal", "mean": 1.0, "std": 0.5})
    assert (lo, hi) == pytest.approx((0.0, 2.0))
    # clip narrows the ±2σ span
    assert _range_from_distribution(
        {"kind": "normal", "mean": 1.0, "std": 0.5, "clip": [0.5, 1.6]}
    ) == pytest.approx((0.5, 1.6))
    assert _range_from_distribution(
        {"kind": "choice", "values": [3, 1, 2]}) == (1.0, 3.0)
    # degenerate / unknown → skip
    assert _range_from_distribution({"kind": "uniform", "low": 2, "high": 2}) is None
    assert _range_from_distribution({"kind": "weird"}) is None


# ── parsing ──────────────────────────────────────────────────────────────────

def test_resolve_course_height_only_gap_skipped():
    rands = resolve_world_randomizations(_parkour_world())
    # box height resolves; gap length is not yet wired → skipped (not an error)
    assert rands == [Randomization(
        "middle_box_height", "course_height", "obstacle__box_02", 0.15, 0.35)]


def test_resolve_object_mass_and_friction():
    rands = resolve_world_randomizations(_object_world())
    kinds = {(r.kind, r.target_name) for r in rands}
    assert ("object_mass", "target_object") in kinds
    assert ("object_friction", "target_object") in kinds


def test_resolve_empty_world():
    assert resolve_world_randomizations({}) == []
    assert resolve_world_randomizations({"train": {"variations": []}}) == []


# ── event wiring ─────────────────────────────────────────────────────────────

def test_install_course_event():
    env_cfg = types.SimpleNamespace(events={})
    msgs = install_world_randomizations(env_cfg, _parkour_world())
    assert "world_dr__course" in env_cfg.events
    term = env_cfg.events["world_dr__course"]
    assert term.mode == "reset"
    specs = term.params["specs"]
    assert specs[0]["geom_name"] == "obstacle__box_02"
    assert any("course platform heights" in m for m in msgs)


def test_install_object_events():
    env_cfg = types.SimpleNamespace(events={})
    msgs = install_world_randomizations(env_cfg, _object_world())
    assert "world_dr__object_mass" in env_cfg.events
    assert "world_dr__object_friction" in env_cfg.events
    mass = env_cfg.events["world_dr__object_mass"]
    assert mass.mode == "reset"
    assert mass.params["operation"] == "abs"
    assert mass.params["ranges"] == (0.12, 0.32)
    assert mass.params["asset_cfg"].name == "target_object"
    assert len(msgs) == 2


def test_install_no_events_dict_is_noop():
    # An env_cfg without an events dict must not raise.
    assert install_world_randomizations(types.SimpleNamespace(), _object_world()) == []


# ── box-height write math (real mj_model name resolution + fake model bridge) ─

def _fake_env_with_box(box_name: str, num_envs: int = 4):
    """A minimal env: a real 1-geom mj_model (for name→id) + a fake per-env
    model bridge whose geom_* fields are CPU tensors, mirroring mjlab shapes."""
    import mujoco
    import torch

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="course")
    body.add_geom(name=box_name, type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(0.5, 0.6, 0.1), pos=(1.0, 0.0, 0.1))
    mj_model = spec.compile()
    ngeom = mj_model.ngeom

    model = types.SimpleNamespace(
        geom_size=torch.zeros(num_envs, ngeom, 3),
        geom_pos=torch.zeros(num_envs, ngeom, 3),
        geom_rbound=torch.zeros(num_envs, ngeom),
        geom_aabb=torch.zeros(num_envs, ngeom, 2, 3),
    )
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, box_name)
    model.geom_size[:, gid, 0] = 0.5   # fixed half-x/-y (only z is randomized)
    model.geom_size[:, gid, 1] = 0.6
    env = types.SimpleNamespace(
        num_envs=num_envs, device="cpu",
        sim=types.SimpleNamespace(mj_model=mj_model, model=model))
    return env, gid


def test_randomize_authored_course_writes_grounded_boxes():
    import torch

    from sculptor.world.randomization import randomize_authored_course

    torch.manual_seed(0)
    env, gid = _fake_env_with_box("obstacle__box_02", num_envs=6)
    randomize_authored_course(
        env, None, specs=[{"geom_name": "obstacle__box_02",
                           "low": 0.20, "high": 0.40}])

    half = env.sim.model.geom_size[:, gid, 2]
    pos_z = env.sim.model.geom_pos[:, gid, 2]
    # heights sampled in [0.20, 0.40] → half-height in [0.10, 0.20]
    assert torch.all(half >= 0.10 - 1e-6) and torch.all(half <= 0.20 + 1e-6)
    # box stays grounded: center z == half-height
    assert torch.allclose(pos_z, half)
    # per-env variation actually happened (not one constant)
    assert half.std() > 1e-4
    # broadphase bound recomputed for a box: sqrt(sx²+sy²+sz²)
    expected_rbound = torch.sqrt(0.5**2 + 0.6**2 + half**2)
    assert torch.allclose(env.sim.model.geom_rbound[:, gid], expected_rbound)
    assert torch.allclose(env.sim.model.geom_aabb[:, gid, 1, 2], half)


def test_randomize_authored_course_unknown_geom_is_safe():
    from sculptor.world.randomization import randomize_authored_course

    env, _ = _fake_env_with_box("obstacle__box_02")
    # A geom that does not exist must be skipped, never raise.
    randomize_authored_course(
        env, None, specs=[{"geom_name": "obstacle__nope",
                           "low": 0.2, "high": 0.4}])


def test_randomize_authored_course_hits_all_env_origin_copies():
    """In multi-env training the compiler emits per-origin course copies
    (obstacle__box_02__env_0000, __env_0001, …); the reset event must randomize
    EVERY copy (the base-name-only lookup was a silent no-op in production)."""
    import mujoco
    import torch

    from sculptor.world.randomization import randomize_authored_course

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="course")
    for suffix in ("__env_0000", "__env_0001"):
        body.add_geom(name=f"obstacle__box_02{suffix}",
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=(0.5, 0.6, 0.1), pos=(1.0, 0.0, 0.1))
    mj_model = spec.compile()
    ngeom = mj_model.ngeom
    model = types.SimpleNamespace(
        geom_size=torch.zeros(4, ngeom, 3), geom_pos=torch.zeros(4, ngeom, 3),
        geom_rbound=torch.zeros(4, ngeom), geom_aabb=torch.zeros(4, ngeom, 2, 3))
    model.geom_size[:, :, 0] = 0.5
    model.geom_size[:, :, 1] = 0.6
    env = types.SimpleNamespace(
        num_envs=4, device="cpu",
        sim=types.SimpleNamespace(mj_model=mj_model, model=model))

    torch.manual_seed(1)
    randomize_authored_course(
        env, None, specs=[{"geom_name": "obstacle__box_02",
                           "low": 0.20, "high": 0.40}])

    g0 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle__box_02__env_0000")
    g1 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle__box_02__env_0001")
    for g in (g0, g1):
        half = env.sim.model.geom_size[:, g, 2]
        assert torch.all(half >= 0.10 - 1e-6) and torch.all(half <= 0.20 + 1e-6)
        assert torch.allclose(env.sim.model.geom_pos[:, g, 2], half)  # grounded
    # Every env-origin copy got the SAME per-env sample, so an env reads one
    # randomized height whichever origin it sits at.
    assert torch.allclose(env.sim.model.geom_size[:, g0, 2],
                          env.sim.model.geom_size[:, g1, 2])
