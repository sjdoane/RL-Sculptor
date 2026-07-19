"""Generic manipulation telemetry (the mjlab rollout artifact contract).

Covers the three claims the benchmark manifests depend on:

1. discovery is data-driven — capability descriptors matched by semantic
   role names against the compiled robot asset, robots classified by
   actuation, objects by passivity; no robot/task name strings;
2. the REAL registered YAM lift and multi-cube cfgs discover end-effector,
   finger groups, objects, and injectable contact sensors (CPU spec
   compile only — no env construction, no GPU);
3. the recorder produces honest per-step arrays: object root states,
   bilateral-contact grasp evidence, per-step target identity remapped to
   the discovered object order, and a manifest that states every
   derivation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sculptor.adapters.manipulation_telemetry import (
    TELEMETRY_MANIFEST_NAME,
    ManipulationDiscovery,
    ManipulationRecorder,
    _match_capability,
    discover_from_cfg,
    inject_contact_sensors,
)
from sculptor.world.capabilities import robot_capabilities


# ── 1. capability matching by role names ──────────────────────────────
def _yam_role_names() -> tuple[set[str], set[str]]:
    capability = robot_capabilities()["yam:parallel_gripper"]
    bodies = {n for names in capability.body_roles.values() for n in names}
    sites = {n for names in capability.site_roles.values() for n in names}
    return bodies, sites


def test_capability_matches_on_complete_role_names():
    bodies, sites = _yam_role_names()
    # Extra names in the asset never hurt — the descriptor is a subset.
    bodies |= {"extra_link", "camera_mount"}
    sites |= {"aux_site"}
    match = _match_capability(bodies, sites)
    assert match is not None
    assert match.capability_id == "yam:parallel_gripper"


def test_capability_rejects_incomplete_role_names():
    bodies, sites = _yam_role_names()
    bodies.discard("lf_down")  # one missing finger body → not this asset
    assert _match_capability(bodies, sites) is None
    assert _match_capability({"pelvis"}, set()) is None


# ── 2. discovery on the real registered YAM cfgs ──────────────────────
def _lift_cfg():
    yam = pytest.importorskip(
        "mjlab.tasks.manipulation.config.yam.env_cfgs",
        reason="registered YAM manipulation tasks unavailable")
    return yam


def test_discover_real_yam_lift_cfg():
    yam = _lift_cfg()
    cfg = yam.yam_lift_cube_env_cfg()
    discovery = discover_from_cfg(cfg)
    assert discovery is not None
    assert discovery.robot_entity == "robot"
    assert discovery.object_names == ("cube",)
    assert discovery.capability_id == "yam:parallel_gripper"
    assert discovery.ee_site == "tcp_site"
    assert set(discovery.finger_groups) == {"left", "right"}
    assert discovery.grasp_capable is True

    before = len(cfg.scene.sensors or ())
    discovery = inject_contact_sensors(cfg, discovery)
    assert discovery.sensor_names == (
        "manip_contact__left__cube", "manip_contact__right__cube")
    sensors = {s.name: s for s in cfg.scene.sensors}
    assert len(sensors) == before + 2
    left = sensors["manip_contact__left__cube"]
    assert left.primary.entity == "robot"
    assert left.primary.mode == "subtree"
    assert set(left.primary.pattern) == set(
        robot_capabilities()["yam:parallel_gripper"]
        .body_roles["left_finger"])
    assert left.secondary.entity == "cube"
    # Injection is idempotent — a second call adds nothing.
    again = inject_contact_sensors(cfg, discovery)
    assert len(cfg.scene.sensors) == before + 2
    assert again.sensor_names == ()


def test_discover_real_yam_multi_cube_cfg():
    yam = _lift_cfg()
    cfg = yam.yam_multi_cube_seg_env_cfg(num_cubes=3)
    discovery = discover_from_cfg(cfg)
    assert discovery is not None
    assert discovery.object_names == ("cube_0", "cube_1", "cube_2")
    assert discovery.grasp_capable is True
    discovery = inject_contact_sensors(cfg, discovery)
    assert len(discovery.sensor_names) == 6  # 2 finger groups × 3 cubes


# ── discovery classification (synthetic, no simulator) ────────────────
def _actuated_cfg():
    return SimpleNamespace(
        articulation=SimpleNamespace(actuators=("act",)),
        spec_fn=lambda: (_ for _ in ()).throw(RuntimeError("no spec")))


def _passive_cfg():
    return SimpleNamespace(articulation=None)


def _scene_cfg(entities):
    return SimpleNamespace(scene=SimpleNamespace(
        entities=entities, sensors=()))


def test_discovery_requires_objects_and_one_robot():
    # No passive objects → nothing to record.
    assert discover_from_cfg(_scene_cfg({"robot": _actuated_cfg()})) is None
    # No actuated entity → no robot to attribute telemetry to.
    assert discover_from_cfg(_scene_cfg({"cube": _passive_cfg()})) is None
    # Two actuated entities → ambiguous; never guess.
    assert discover_from_cfg(_scene_cfg({
        "a": _actuated_cfg(), "b": _actuated_cfg(),
        "cube": _passive_cfg()})) is None
    # One robot + passive objects → discovery even when the capability
    # match fails (spec_fn raises): objects still get telemetry.
    discovery = discover_from_cfg(_scene_cfg({
        "robot": _actuated_cfg(), "ball": _passive_cfg()}))
    assert discovery is not None
    assert discovery.object_names == ("ball",)
    assert discovery.capability_id is None
    assert discovery.grasp_capable is False
    # No finger groups → sensor injection is a no-op.
    assert inject_contact_sensors(
        _scene_cfg({}), discovery).sensor_names == ()


# ── 3. recorder semantics on a fake env ───────────────────────────────
N = 4  # fake env count


class _FakeScene:
    def __init__(self, items):
        self._items = items

    def __getitem__(self, key):
        return self._items[key]


def _fake_env(*, multi_target: bool):
    robot_data = SimpleNamespace(
        site_pos_w=np.tile(
            np.asarray([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]], np.float32),
            (N, 1, 1)),
        body_quat_w=np.tile(
            np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32),
            (N, 1, 1)),
    )
    robot = SimpleNamespace(
        data=robot_data,
        site_names=("grasp_site", "tcp_site"),
        body_names=("arm", "link_6"),
    )
    cube = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=np.full((N, 3), 2.0, np.float32),
        root_link_quat_w=np.tile(
            np.asarray([1, 0, 0, 0], np.float32), (N, 1)),
        root_link_lin_vel_w=np.full((N, 3), 0.5, np.float32),
        root_link_ang_vel_w=np.zeros((N, 3), np.float32),
    ))
    other = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=np.zeros((N, 3), np.float32),
        root_link_quat_w=np.tile(
            np.asarray([1, 0, 0, 0], np.float32), (N, 1)),
        root_link_lin_vel_w=np.zeros((N, 3), np.float32),
        root_link_ang_vel_w=np.zeros((N, 3), np.float32),
    ))
    # left touches envs 0/1; right touches envs 1/2 → grasp only in env 1.
    left_sensor = SimpleNamespace(data=SimpleNamespace(
        found=np.asarray([[1], [1], [0], [0]], np.float32)))
    right_sensor = SimpleNamespace(data=SimpleNamespace(
        found=np.asarray([[0], [1], [1], [0]], np.float32)))

    if multi_target:
        command_term = SimpleNamespace(
            cfg=SimpleNamespace(entity_names=("other", "cube")),
            command=np.full((N, 3), 0.4, np.float32),
            target_selection=np.asarray([0, 1, 1, 0], np.int64),
        )
        objects = ("cube", "other")
    else:
        command_term = SimpleNamespace(
            cfg=SimpleNamespace(entity_name="cube"),
            command=np.full((N, 3), 0.4, np.float32),
        )
        objects = ("cube",)

    items = {
        "robot": robot,
        "cube": cube,
        "manip_contact__left__cube": left_sensor,
        "manip_contact__right__cube": right_sensor,
    }
    if multi_target:
        items["other"] = other
    env = SimpleNamespace(
        scene=_FakeScene(items),
        command_manager=SimpleNamespace(_terms={"lift_height": command_term}),
    )
    discovery = ManipulationDiscovery(
        robot_entity="robot",
        object_names=objects,
        capability_id="yam:parallel_gripper",
        ee_site="tcp_site",
        ee_body="link_6",
        finger_groups={"left": ("lf_down",), "right": ("rf_down",)},
        grasp_capable=True,
        sensor_names=(
            "manip_contact__left__cube", "manip_contact__right__cube"),
    )
    return env, discovery


def test_recorder_records_objects_contacts_grasp_and_target(tmp_path: Path):
    env, discovery = _fake_env(multi_target=True)
    recorder = ManipulationRecorder(env, discovery)
    recorder.append()
    recorder.append()
    arrays = recorder.finalize()

    assert arrays["ee_pos_w"].shape == (2, N, 3)
    assert np.allclose(arrays["ee_pos_w"][0, 0], [1.0, 1.0, 1.0])  # tcp site
    assert arrays["ee_quat_w"].shape == (2, N, 4)
    assert np.allclose(arrays["ee_quat_w"][0, 0], [0, 1, 0, 0])  # link_6
    assert arrays["object__cube__pos_w"].shape == (2, N, 3)
    assert arrays["object__cube__lin_vel_w"][0, 0, 0] == pytest.approx(0.5)
    assert arrays["object__other__pos_w"].shape == (2, N, 3)

    assert arrays["contact__left__cube"].shape == (2, N)
    assert arrays["contact__left__cube"][0].tolist() == [1, 1, 0, 0]
    assert arrays["contact__right__cube"][0].tolist() == [0, 1, 1, 0]
    # grasp = bilateral contact — only env 1 has both fingers touching.
    assert arrays["grasp__cube"][0].tolist() == [0, 1, 0, 0]

    # target_selection indexes cfg.entity_names ("other","cube") and must
    # be remapped into the discovered object order ("cube","other").
    assert arrays["target_object_index"].dtype == np.int32
    assert arrays["target_object_index"][0].tolist() == [1, 0, 0, 1]
    assert arrays["target__pos_w"].shape == (2, N, 3)

    recorder.write_manifest(tmp_path, arrays)
    manifest = json.loads(
        (tmp_path / TELEMETRY_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["object_names"] == ["cube", "other"]
    assert manifest["grasp_capable"] is True
    assert "bilateral finger contact" in manifest["derivations"]["grasp"]
    assert manifest["channels"]["grasp__cube"]["shape"] == [2, N]


def test_recorder_static_target_for_single_object_command():
    env, discovery = _fake_env(multi_target=False)
    recorder = ManipulationRecorder(env, discovery)
    recorder.append()
    arrays = recorder.finalize()
    assert arrays["target_object_index"].tolist() == [[0] * N]
    assert arrays["target__pos_w"].shape == (1, N, 3)
