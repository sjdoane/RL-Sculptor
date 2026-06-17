"""§Ship 49: always-correct joint identification — the canonical resolver,
the validation gates that enforce it, and the §3A regression table (a
shuffled / empty / foreign joint-name list must HARD-FAIL or be stable,
never silently mis-score). GPU-free."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sculptor.eval.generated_metric import compute_generated_metric
from sculptor.eval.joint_resolver import (
    JointResolutionError,
    assert_name_axis_contract,
    arm_indices,
    hip_indices,
    leg_sagittal_indices,
    parse_joint_name,
    resolve_joint_roles,
    resolve_or_raise,
    select_joints,
)
from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.eval.robot_manifest import G1_29, GO1_12, robot_joint_names

# Real captured orderings (entity joint_names, = buffer column order).
G1 = list(G1_29)
GO1 = list(GO1_12)


# ── parser ────────────────────────────────────────────────────────────


def test_parse_side_segment_axis():
    d = parse_joint_name("left_hip_roll_joint")
    assert (d.side, d.segment, d.axis) == ("left", "hip", "roll")
    d = parse_joint_name("FR_calf_joint")
    assert (d.side, d.segment, d.axis) == ("front_right", "calf", None)
    d = parse_joint_name("RL_thigh_joint")
    assert (d.side, d.segment, d.axis) == ("rear_left", "thigh", None)
    d = parse_joint_name("waist_pitch_joint")
    assert (d.side, d.segment, d.axis) == (None, "waist", "pitch")
    d = parse_joint_name("right_ankle")            # axis-less single DOF
    assert (d.side, d.segment, d.axis) == ("right", "ankle", None)


# ── group selection (replaces _match_joints) ───────────────────────────


def test_leg_sagittal_excludes_roll_and_yaw_on_g1():
    """The directional fix: sagittal legs = hip PITCH + knee + ankle PITCH,
    never hip roll/yaw (which is how a sideways kick used to earn credit)."""
    idx = leg_sagittal_indices(G1)
    assert idx == [0, 3, 4, 6, 9, 10]
    names = [G1[i] for i in idx]
    assert all("roll" not in n and "yaw" not in n for n in names), names
    assert G1.index("left_hip_roll_joint") not in idx
    assert G1.index("left_hip_yaw_joint") not in idx


def test_hip_and_arm_groups_on_g1():
    assert hip_indices(G1) == [0, 1, 2, 6, 7, 8]              # all hip DOF
    arms = arm_indices(G1)
    assert all(("shoulder" in G1[i]) or ("elbow" in G1[i]) for i in arms)
    assert G1.index("left_elbow_joint") in arms


def test_select_joints_axis_filter_admits_axisless():
    names = ["left_ankle", "left_ankle_roll_joint"]
    # axes={"pitch", None}: the bare ankle (single DOF) qualifies, the roll one
    # does not.
    assert select_joints(names, segments=("ankle",), axes=("pitch", None)) == [0]


# ── strict role resolution ─────────────────────────────────────────────


def test_resolve_roles_unique_on_g1():
    r = resolve_joint_roles(G1, ["left_hip_pitch", "right_knee", "left_ankle_pitch"])
    assert r.ok
    assert r.resolved == {"left_hip_pitch": 0, "right_knee": 9, "left_ankle_pitch": 4}


def test_resolve_roles_ambiguous_axis_required():
    r = resolve_joint_roles(G1, ["left_hip"])     # pitch/roll/yaw all match
    assert not r.ok
    assert r.ambiguous["left_hip"] == [0, 1, 2]


def test_resolve_roles_missing_on_foreign_robot():
    """A G1 metric's roles do not exist on a Go1 — the resolution is empty,
    not a wrong guess."""
    r = resolve_joint_roles(GO1, ["left_knee", "right_knee"])
    assert not r.ok
    assert set(r.missing) == {"left_knee", "right_knee"}


def test_unparseable_role_is_missing_not_ambiguous():
    r = resolve_joint_roles(G1, ["tail_wag"])
    assert r.missing == ["tail_wag"] and not r.ambiguous


def test_resolve_or_raise():
    assert resolve_or_raise(G1, ["left_knee"]) == {"left_knee": 3}
    with pytest.raises(JointResolutionError):
        resolve_or_raise(G1, ["left_hip"])        # ambiguous


# ── adapter order-contract (item 1) ────────────────────────────────────


def test_name_axis_contract_enforced():
    assert_name_axis_contract(G1, 29)             # ok
    with pytest.raises(JointResolutionError):
        assert_name_axis_contract(G1, 28)         # length != joint-axis width


def test_adapter_persists_entity_order():
    """Pin the order-contract at its source: the mjlab runner must capture
    joint_names from the SCENE ENTITY (whose order == the joint_pos/joint_vel
    buffer columns), NOT the mjModel route (which includes the free joint and
    may reorder). Guards against a refactor that silently breaks every
    name-based metric at once (§3A invariant i)."""
    runner = (Path(__file__).resolve().parents[1]
              / "sculptor" / "adapters" / "_mjlab_runner.py").read_text()
    entity = runner.index('_robot.joint_names') if '_robot.joint_names' in runner \
        else runner.index('getattr(_robot, "joint_names"')
    # The entity-first capture documents and establishes the buffer order…
    assert 'SAME order as the persisted joint_pos' in runner
    # …and runs before the mjModel `_names(...)` fallback.
    assert entity < runner.index('def _names(')


# ── manifest ───────────────────────────────────────────────────────────


def test_robot_manifest_resolves_by_hint():
    assert robot_joint_names("Mjlab-Velocity-Flat-Unitree-G1") == list(G1_29)
    assert robot_joint_names("Mjlab-Velocity-Flat-Unitree-Go1") == list(GO1_12)
    assert robot_joint_names("some-unknown-bot") is None
    assert robot_joint_names(None) is None


# ── §3A regression table: runtime never silently mis-scores ────────────

# A role-based metric (the safe pattern the prompt teaches).
ROLE_METRIC = '''import numpy as np
REQUIRED_JOINT_ROLES = ["left_knee", "right_knee", "left_hip_pitch", "right_hip_pitch"]
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); grav = arrays.get("projected_gravity_b")
    if jv is None or grav is None:
        return {"spec_score": 0.0}
    roles = (meta or {}).get("joint_roles", {})
    idx = [roles[r] for r in ("left_knee","right_knee","left_hip_pitch","right_hip_pitch") if r in roles]
    if not idx:
        return {"spec_score": 0.0}
    up = float(np.mean(grav[..., 2] < -0.85))
    peak = float(np.abs(jv[..., idx]).max(axis=2).max(axis=0).mean())
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-peak / 8.0)), 0.0, 1.0))}
'''


def _rollout(tmp: Path, names, label: str) -> Path:
    """A synthetic G1-shaped rollout (29 joints) with a clear left-knee burst,
    written with the given joint_names list (None → empty)."""
    d = tmp / f"roll_{label}"
    d.mkdir(parents=True, exist_ok=True)
    T, E, J = 60, 4, 29
    rng = np.random.default_rng(0)
    jv = rng.normal(0, 0.05, (T, E, J)).astype(np.float32)
    jv[20:25, :, 3] = 9.0                          # left_knee burst (index 3)
    jp = np.cumsum(jv, axis=0).astype(np.float32) * 0.02
    grav = np.zeros((T, E, 3), np.float32); grav[..., 2] = -1.0
    root = np.zeros((T, E, 3), np.float32); root[..., 2] = 0.7
    np.savez(d / "trajectory.npz", joint_pos=jp, joint_vel=jv,
             projected_gravity_b=grav, root_link_pos_w=root)
    (d / "behavior.json").write_text(json.dumps({"step_dt": 0.02}))
    (d / "mjcf_limits.json").write_text(
        json.dumps({"joint_names": [] if names is None else names}))
    return d


def test_3A_correct_names_score_and_resolve(tmp_path):
    p = tmp_path / "m.py"; p.write_text(ROLE_METRIC)
    out = compute_generated_metric(p, _rollout(tmp_path, G1, "ok"))
    assert out["spec_score"] > 0.5, out
    # the resolved indices are the REAL G1 columns — observable, not guessed.
    assert out["joint_roles"]["left_knee"] == 3
    assert out["joint_roles"]["left_hip_pitch"] == 0


def test_3A_empty_names_hard_fail(tmp_path):
    p = tmp_path / "m.py"; p.write_text(ROLE_METRIC)
    out = compute_generated_metric(p, _rollout(tmp_path, None, "empty"))
    assert out["spec_score"] == 0.0
    assert "joint_names" in out["error"]


def test_3A_foreign_robot_hard_fail(tmp_path):
    """The old behaviour scored a foreign Go1 name list ~0.34 silently; now it
    hard-fails because the G1 roles resolve to nothing on a Go1."""
    p = tmp_path / "m.py"; p.write_text(ROLE_METRIC)
    foreign = GO1 + ["x"] * (29 - len(GO1))        # 29 names, none are G1 legs
    out = compute_generated_metric(p, _rollout(tmp_path, foreign, "foreign"))
    assert out["spec_score"] == 0.0
    assert "left_knee" in out["error"]             # names the missing role


def test_3A_wrong_length_hard_fail(tmp_path):
    p = tmp_path / "m.py"; p.write_text(ROLE_METRIC)
    out = compute_generated_metric(p, _rollout(tmp_path, G1[:20], "short"))
    assert out["spec_score"] == 0.0
    assert "contract" in out["error"]


# ── validation gates ───────────────────────────────────────────────────

RAW_INDEX_METRIC = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel")
    if jv is None:
        return {"spec_score": 0.0}
    return {"spec_score": float(np.clip(np.abs(jv[:, :, 0]).mean(), 0.0, 1.0))}
'''

# Reads joint column 0 via the Ellipsis form — passes the static raw-int ban
# but is caught by the permutation gate (index-sensitive).
ELLIPSIS_INDEX_HACK = '''import numpy as np
def compute_spec(arrays, behavior, meta):
    jv = arrays.get("joint_vel"); g = arrays.get("projected_gravity_b")
    if jv is None or g is None:
        return {"spec_score": 0.0}
    up = float((g[..., 2] < -0.85).mean())
    return {"spec_score": float(np.clip(up * (1.0 - np.exp(-np.abs(jv[..., 0]).max() / 8.0)), 0.0, 1.0))}
'''


def test_raw_integer_joint_index_rejected(tmp_path):
    p = tmp_path / "raw.py"; p.write_text(RAW_INDEX_METRIC)
    v = validate_generated_metric(RAW_INDEX_METRIC, p)
    assert not v["ok"]
    assert v["gates"]["no_raw_joint_index"] is False
    assert any("joint-index" in r for r in v["reasons"])


def test_ellipsis_index_hack_caught_by_permutation(tmp_path):
    """A metric that reads a hard-coded joint column via `jv[..., 0]` slips
    the static ban but swings under a consistent joint relabel."""
    p = tmp_path / "ell.py"; p.write_text(ELLIPSIS_INDEX_HACK)
    v = validate_generated_metric(ELLIPSIS_INDEX_HACK, p)
    assert v["gates"]["no_raw_joint_index"] is True      # static ban doesn't fire
    assert v["gates"]["joint_index_robust"] is False     # permutation does
    assert any("index-sensitive" in r for r in v["reasons"])


def test_role_metric_is_permutation_robust(tmp_path):
    """The role-based metric reads by name → invariant under relabel."""
    p = tmp_path / "role.py"; p.write_text(ROLE_METRIC)
    v = validate_generated_metric(
        ROLE_METRIC, p, behavior_goal="repeatedly kick forward with one leg")
    assert v["gates"]["joint_index_robust"] is True, v["reasons"]
    assert v["required_roles"] == ["left_knee", "right_knee",
                                   "left_hip_pitch", "right_hip_pitch"]


def test_required_roles_gate_rejects_on_real_robot(tmp_path):
    """A metric needing a role the robot lacks is rejected pre-project when the
    real joint_names are supplied (here: G1 leg roles against a Go1)."""
    p = tmp_path / "role.py"; p.write_text(ROLE_METRIC)
    v = validate_generated_metric(
        ROLE_METRIC, p, behavior_goal="kick forward",
        robot_joint_names=GO1)
    assert v["gates"]["joint_roles_resolve"] is False
    assert any("joint-roles" in r for r in v["reasons"])


def test_required_roles_gate_passes_on_matching_robot(tmp_path):
    p = tmp_path / "role.py"; p.write_text(ROLE_METRIC)
    v = validate_generated_metric(
        ROLE_METRIC, p, behavior_goal="kick forward",
        robot_joint_names=G1)
    assert v["gates"].get("joint_roles_resolve") is True
