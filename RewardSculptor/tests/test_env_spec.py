"""§RL_SCULPTOR_AUDIT (env generalization, 2026-07-04): the general
per-project env spec — schema validation, the runner's general applier,
jump-preset parity with the retired hardcoded profile, and the
adapter/loader threading.

Offline tests only — cfg mutation is pure attribute manipulation on a
SimpleNamespace fake (same convention as test_env_profile.py, whose
fake is reused here)."""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import pytest

from sculptor import env_spec as es
from sculptor.adapters import _mjlab_runner
from tests.test_env_profile import _fake_velocity_cfg


def _fake_cfg_full() -> SimpleNamespace:
    """The velocity-cfg fake extended with the surfaces the general
    spec can touch beyond the jump preset (joint resets, friction,
    push magnitudes)."""
    cfg = _fake_velocity_cfg()
    cfg.events["reset_robot_joints"] = SimpleNamespace(params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
    })
    cfg.events["push_robot"] = SimpleNamespace(
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }},
    )
    cfg.events["foot_friction"] = SimpleNamespace(params={
        "ranges": (0.3, 1.2),
    })
    # A PD-actuated 'robot' scene entity, so _primary_robot_entity resolves and
    # the actuator-type guard admits the pd_gains / motor_strength DR axes (a
    # non-PD robot like cartpole is exercised in the GPU smoke test).
    class BuiltinPositionActuatorCfg:  # name matches the PD-actuator whitelist
        pass

    cfg.scene = SimpleNamespace(entities={
        "robot": SimpleNamespace(articulation=SimpleNamespace(
            actuators=(BuiltinPositionActuatorCfg(),)))})
    return cfg


# ── Schema validation ──────────────────────────────────────────────────────
def test_jump_preset_validates() -> None:
    assert es.validate_env_spec(es.jump_preset_spec()) == []


def test_minimal_and_empty_sections_validate() -> None:
    assert es.validate_env_spec({"env_spec_version": 1}) == []
    assert es.validate_env_spec(
        {"env_spec_version": 1, "shared": {}, "train": {}}) == []


def test_unknown_keys_rejected_everywhere() -> None:
    errors = es.validate_env_spec({
        "env_spec_version": 1,
        "bogus_top": 1,
        "shared": {"episode_len": 10.0},          # typo'd key
        "train": {"reset_hight_offset_m": [0, 1]},  # typo'd key
    })
    joined = "\n".join(errors)
    assert "bogus_top" in joined
    assert "episode_len" in joined
    assert "reset_hight_offset_m" in joined


def test_train_only_keys_rejected_in_shared() -> None:
    """The section split IS the train/eval isolation guarantee — a
    train-only curriculum key in `shared` must fail validation, or RSI
    spawns would leak into rollout evaluation."""
    errors = es.validate_env_spec({
        "env_spec_version": 1,
        "shared": {"reset_height_offset_m": [0.0, 0.4]},
    })
    assert any("reset_height_offset_m" in e for e in errors)
    errors = es.validate_env_spec({
        "env_spec_version": 1,
        "shared": {"min_base_height_termination_m": 0.3},
    })
    assert any("min_base_height_termination_m" in e for e in errors)


def test_shared_only_keys_rejected_in_train() -> None:
    errors = es.validate_env_spec({
        "env_spec_version": 1,
        "train": {"episode_length_s": 10.0},
    })
    assert any("episode_length_s" in e for e in errors)


def test_bounds_enforced() -> None:
    bad = {
        "env_spec_version": 1,
        "shared": {"episode_length_s": 500.0,           # > 60
                   "orientation_termination_deg": 10.0},  # < 45
        "train": {"entropy_coef_scale": 100.0,           # > 4
                  "reset_height_offset_m": [0.5, 0.1],   # lo > hi
                  "friction_range": [0.0, 5.0]},         # both out
    }
    errors = es.validate_env_spec(bad)
    joined = "\n".join(errors)
    for frag in ("episode_length_s", "orientation_termination_deg",
                 "entropy_coef_scale", "lo 0.5 > hi 0.1", "friction_range"):
        assert frag in joined, f"missing {frag!r} in: {joined}"


def test_non_finite_and_wrong_types_rejected() -> None:
    errors = es.validate_env_spec({
        "env_spec_version": 1,
        "shared": {"zero_velocity_commands": "yes",
                   "episode_length_s": float("nan")},
        "train": {"reset_vertical_velocity_mps": [0.0, "fast"]},
    })
    assert len(errors) >= 3


def test_mixed_key_types_never_raise() -> None:
    """Increment-5 verifier: mixed str+int dict keys made the
    unknown-key `sorted()` raise TypeError. Contract: errors, never
    raises, for ANY input shape."""
    errs = es.validate_env_spec({
        "env_spec_version": 1,
        1: "int-keyed",
        "shared": {2: "x", "episode_length_s": 10.0},
        "train": {3: "y"},
    })
    assert errs and all(isinstance(e, str) for e in errs)


def test_wrong_version_rejected() -> None:
    assert es.validate_env_spec({"env_spec_version": 2}) != []
    assert es.validate_env_spec({}) != []


def test_push_events_subschema() -> None:
    ok = {"env_spec_version": 1,
          "shared": {"push_events": {"enabled": True, "interval_s": [2, 5],
                                     "linear_mps": 0.5,
                                     "angular_radps": 0.3}}}
    assert es.validate_env_spec(ok) == []
    bad = {"env_spec_version": 1,
           "shared": {"push_events": {"interval_s": [2, 5]}}}   # no enabled
    assert es.validate_env_spec(bad) != []
    bad2 = {"env_spec_version": 1,
            "shared": {"push_events": {"enabled": True, "shove": 1}}}
    assert any("shove" in e for e in es.validate_env_spec(bad2))


def test_load_env_spec_raises_with_all_errors(tmp_path) -> None:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({
        "env_spec_version": 1,
        "shared": {"episode_length_s": 500.0},
        "train": {"entropy_coef_scale": 100.0},
    }))
    with pytest.raises(ValueError) as ei:
        es.load_env_spec(p)
    assert "episode_length_s" in str(ei.value)
    assert "entropy_coef_scale" in str(ei.value)
    with pytest.raises(ValueError, match="unreadable"):
        es.load_env_spec(tmp_path / "missing.json")
    (tmp_path / "junk.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        es.load_env_spec(tmp_path / "junk.json")


def test_rsi_requires_early_termination_pairing() -> None:
    """MEASURED invariant (tuck-jump iters 19-20): airborne/upward RSI
    starts without early termination off the recoverable manifold
    regress — the validator enforces the pairing, not just the prompt."""
    base = {"env_spec_version": 1}
    # Airborne height offsets alone → rejected.
    bad = {**base, "train": {"reset_height_offset_m": [0.0, 0.4]}}
    assert any("min_base_height_termination_m" in e
               for e in es.validate_env_spec(bad))
    # Upward spawn velocity alone → rejected.
    bad2 = {**base, "train": {"reset_vertical_velocity_mps": [-0.5, 2.0]}}
    assert any("min_base_height_termination_m" in e
               for e in es.validate_env_spec(bad2))
    # Paired → valid.
    ok = {**base, "train": {"reset_height_offset_m": [0.0, 0.4],
                            "min_base_height_termination_m": 0.3}}
    assert es.validate_env_spec(ok) == []
    # Horizontal-only / downward-only jitter doesn't trigger it.
    ok2 = {**base, "train": {"reset_horizontal_velocity_mps": [-1.0, 1.0]}}
    assert es.validate_env_spec(ok2) == []
    ok3 = {**base, "train": {"reset_vertical_velocity_mps": [-1.0, 0.0]}}
    assert es.validate_env_spec(ok3) == []


def test_fell_over_termination_bool_validated() -> None:
    """`fell_over_termination` must be a real boolean — not truthy/falsy
    junk — and belongs to train only (a shared-section fell_over switch
    would leak into rollout evaluation)."""
    assert es.validate_env_spec(
        {"env_spec_version": 1, "train": {"fell_over_termination": False}}
    ) == []
    assert es.validate_env_spec(
        {"env_spec_version": 1, "train": {"fell_over_termination": True}}
    ) == []
    errors = es.validate_env_spec(
        {"env_spec_version": 1, "train": {"fell_over_termination": "off"}})
    assert any("fell_over_termination" in e for e in errors)
    errors = es.validate_env_spec(
        {"env_spec_version": 1, "shared": {"fell_over_termination": False}})
    assert any("fell_over_termination" in e for e in errors)


def test_fell_over_termination_is_iterable() -> None:
    assert "fell_over_termination" in es.ITERABLE_TRAIN_KEYS


def test_iterable_train_keys_is_the_train_section() -> None:
    """The diagnoser's editable surface is exactly the train section's
    value keys — shared keys must never be iterable mid-run."""
    assert "reset_height_offset_m" in es.ITERABLE_TRAIN_KEYS
    assert "entropy_coef_scale" in es.ITERABLE_TRAIN_KEYS
    assert "episode_length_s" not in es.ITERABLE_TRAIN_KEYS
    assert "orientation_termination_deg" not in es.ITERABLE_TRAIN_KEYS
    assert "zero_velocity_commands" not in es.ITERABLE_TRAIN_KEYS


# ── Jump-preset parity through the general applier ─────────────────────────
def test_jump_preset_parity_with_profile_train() -> None:
    """The preset routed through _apply_env_spec must mutate the cfg
    EXACTLY like the retired hardcoded profile (values pinned by
    test_env_profile.py, which now also runs through this path)."""
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_spec(cfg, es.jump_preset_spec(), train=True)
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (0.0, 0.0)
    assert twist.ranges.heading is None
    assert twist.rel_standing_envs == 1.0
    assert twist.heading_command is False
    assert "command_vel" not in cfg.curriculum
    assert "push_robot" not in cfg.events
    assert cfg.terminations["fell_over"].params["limit_angle"] == (
        pytest.approx(math.radians(120.0)))
    assert cfg.episode_length_s == 10.0
    assert cfg.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.40)
    assert cfg.events["reset_base"].params["velocity_range"]["z"] == (-0.5, 2.0)


def test_jump_preset_parity_rollout_excludes_train_section() -> None:
    cfg = _fake_velocity_cfg()
    _mjlab_runner._apply_env_spec(cfg, es.jump_preset_spec(), train=False)
    # Shared applies…
    assert cfg.episode_length_s == 10.0
    assert "push_robot" not in cfg.events
    # …train-only curricula do NOT.
    assert cfg.events["reset_base"].params["pose_range"]["z"] == (0.01, 0.05)
    assert cfg.events["reset_base"].params["velocity_range"] == {}
    assert "sunk" not in cfg.terminations


def test_none_and_empty_spec_are_noops() -> None:
    for spec in (None, {}):
        cfg = _fake_velocity_cfg()
        _mjlab_runner._apply_env_spec(cfg, spec)
        assert cfg.commands["twist"].ranges.lin_vel_x == (-1.0, 1.0)
        assert cfg.episode_length_s == 20.0


# ── The knobs beyond the jump preset ───────────────────────────────────────
def test_friction_randomization_applies_train_only() -> None:
    spec = {"env_spec_version": 1,
            "train": {"friction_range": [0.2, 1.8]}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    assert cfg.events["foot_friction"].params["ranges"] == (0.2, 1.8)
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert cfg2.events["foot_friction"].params["ranges"] == (0.3, 1.2)


def test_joint_reset_ranges_apply_train_only() -> None:
    spec = {"env_spec_version": 1,
            "train": {"reset_joint_position_offset_rad": [-0.3, 0.3],
                      "reset_joint_velocity_radps": [-1.0, 1.0]}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    rj = cfg.events["reset_robot_joints"].params
    assert rj["position_range"] == (-0.3, 0.3)
    assert rj["velocity_range"] == (-1.0, 1.0)
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert cfg2.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)


def test_horizontal_reset_velocity_applies() -> None:
    spec = {"env_spec_version": 1,
            "train": {"reset_horizontal_velocity_mps": [-0.8, 0.8]}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    vr = cfg.events["reset_base"].params["velocity_range"]
    assert vr["x"] == (-0.8, 0.8)
    assert vr["y"] == (-0.8, 0.8)


def test_orientation_reset_ranges_apply_train_only() -> None:
    """§REFERENCE_TRAJECTORY_PLAN §8 part 1: pitch/roll offsets write
    mjlab's native `pose_range["pitch"/"roll"]` keys, train-only."""
    spec = {"env_spec_version": 1,
            "train": {"reset_pitch_offset_rad": [1.2, 1.6],
                      "reset_roll_offset_rad": [-0.3, 0.3]}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    pr = cfg.events["reset_base"].params["pose_range"]
    assert pr["pitch"] == (1.2, 1.6)
    assert pr["roll"] == (-0.3, 0.3)
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert "pitch" not in cfg2.events["reset_base"].params["pose_range"]
    assert "roll" not in cfg2.events["reset_base"].params["pose_range"]


def test_fell_over_termination_false_removes_the_term() -> None:
    """§get-up RSI fix: `fell_over_termination: False` pops mjlab's
    `terminations["fell_over"]` entry for TRAIN only — a lying-start
    reset trips that termination on itself (root pitch/roll far from
    upright reads as "fallen"), so get-up training never runs unless it
    is removed."""
    spec = {"env_spec_version": 1,
            "train": {"fell_over_termination": False}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    assert "fell_over" in cfg.terminations
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    assert "fell_over" not in cfg.terminations
    # time_out (and any sunk termination) must be untouched.
    assert "time_out" in cfg.terminations


def test_fell_over_termination_true_or_absent_leaves_the_term() -> None:
    """Default behavior (today's) is preserved: explicit True and an
    absent key must both leave `fell_over` in place."""
    spec_true = {"env_spec_version": 1,
                 "train": {"fell_over_termination": True}}
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec_true, train=True)
    assert "fell_over" in cfg.terminations

    spec_absent = {"env_spec_version": 1, "train": {}}
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec_absent, train=True)
    assert "fell_over" in cfg2.terminations


def test_fell_over_termination_false_is_train_only() -> None:
    """`fell_over_termination` lives in `train` and is never read when
    `train=False` (rollout evaluation) — the key isn't even a valid
    `shared` key (see test_fell_over_termination_bool_validated), but
    this pins the ADAPTER side of that isolation too: even if a caller
    passed train=False with a spec carrying the key in train, it must
    not be applied."""
    spec = {"env_spec_version": 1,
            "train": {"fell_over_termination": False}}
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=False)
    assert "fell_over" in cfg.terminations


def test_fell_over_termination_absent_from_cfg_is_a_clean_noop() -> None:
    """Term absent from the task cfg → clean no-op, never a crash (the
    defensive per-mutation contract every other knob in this function
    follows)."""
    spec = {"env_spec_version": 1,
            "train": {"fell_over_termination": False}}
    cfg = _fake_cfg_full()
    del cfg.terminations["fell_over"]
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)   # must not raise
    assert "fell_over" not in cfg.terminations


def test_joint_pos_target_injects_new_reset_event() -> None:
    """§8 part 2: `reset_joint_pos_target` injects a brand-new event term
    (mjlab has no shipped per-joint-target reset) — this only needs the
    real mjlab classes (`EventTermCfg`, `SceneEntityCfg`), not a full env
    build, so it's skipped (not failed) when mjlab isn't installed."""
    pytest.importorskip("mjlab")
    spec = {"env_spec_version": 1,
            "train": {"reset_joint_pos_target": [0.1, -0.2, 0.3],
                      "reset_joint_pos_noise_rad": 0.05}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(
        cfg, spec, train=True, task_id="Mjlab-Velocity-Flat-3dof-Testbot")
    ev = cfg.events["reset_robot_joints_to_reference"]
    assert ev.mode == "reset"
    assert list(ev.params["joint_pos_target"].tolist()) == pytest.approx(
        [0.1, -0.2, 0.3])
    assert ev.params["joint_pos_noise"] == pytest.approx(0.05)
    # Rollout (train=False) must NOT get the injected event.
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert "reset_robot_joints_to_reference" not in cfg2.events


def test_joint_pos_target_length_mismatch_is_a_clear_skip_not_silent() -> None:
    """A target vector whose length disagrees with the robot's CANONICAL
    joint count (resolved via `task_id` against
    `sculptor.eval.robot_manifest`) must never be silently applied — the
    mismatch is logged (defensive-skip contract: never break a run) and
    the event is NOT injected, rather than writing a target that would
    misassign joints."""
    pytest.importorskip("mjlab")
    spec = {"env_spec_version": 1,
            # G1 has 29 joints; this target has 3 — a deliberate mismatch.
            "train": {"reset_joint_pos_target": [0.1, -0.2, 0.3]}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(
        cfg, spec, train=True, task_id="Mjlab-Velocity-Flat-Unitree-G1")
    assert "reset_robot_joints_to_reference" not in cfg.events


def test_joint_pos_target_unknown_robot_applies_without_validation() -> None:
    """An unrecognized `task_id` (not in the static manifest) can't be
    checked at all — `robot_joint_names` returns None, so the length
    check is skipped and the event is applied on trust (same "can't
    reject what we can't model" contract the manifest's own docstring
    states for the pre-run gate)."""
    pytest.importorskip("mjlab")
    spec = {"env_spec_version": 1,
            "train": {"reset_joint_pos_target": [0.1, -0.2, 0.3]}}
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(
        cfg, spec, train=True, task_id="Mjlab-SomeUnknownRobot")
    assert "reset_robot_joints_to_reference" in cfg.events


def test_push_retune_and_train_override() -> None:
    # Shared retune: interval + magnitudes.
    spec = {"env_spec_version": 1,
            "shared": {"push_events": {"enabled": True,
                                       "interval_s": [4.0, 8.0],
                                       "linear_mps": 0.2,
                                       "angular_radps": 0.1}}}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    push = cfg.events["push_robot"]
    assert push.interval_range_s == (4.0, 8.0)
    assert push.params["velocity_range"]["x"] == (-0.2, 0.2)
    assert push.params["velocity_range"]["roll"] == (-0.1, 0.1)

    # Train-only pushes: off at eval, on in training.
    spec2 = {"env_spec_version": 1,
             "shared": {"push_events": {"enabled": False}},
             "train": {"push_events": {"enabled": True, "linear_mps": 0.3}}}
    assert es.validate_env_spec(spec2) == []
    cfg_t = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg_t, spec2, train=True)
    assert "push_robot" in cfg_t.events
    assert cfg_t.events["push_robot"].params["velocity_range"]["x"] == (-0.3, 0.3)
    cfg_r = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg_r, spec2, train=False)
    assert "push_robot" not in cfg_r.events


def test_custom_sunk_height_applies() -> None:
    pytest.importorskip("mjlab")
    spec = {"env_spec_version": 1,
            "train": {"min_base_height_termination_m": 0.22}}
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    assert cfg.terminations["sunk"].params["minimum_height"] == 0.22


def test_rl_spec_entropy_scale() -> None:
    spec = {"env_spec_version": 1, "train": {"entropy_coef_scale": 3.0}}
    algo = SimpleNamespace(entropy_coef=0.01)
    _mjlab_runner._apply_rl_spec(SimpleNamespace(algorithm=algo), spec)
    assert algo.entropy_coef == pytest.approx(0.03)
    # No scale → untouched; missing algo shape → never raises.
    algo2 = SimpleNamespace(entropy_coef=0.01)
    _mjlab_runner._apply_rl_spec(
        SimpleNamespace(algorithm=algo2), {"env_spec_version": 1})
    assert algo2.entropy_coef == 0.01
    _mjlab_runner._apply_rl_spec(SimpleNamespace(), spec)
    _mjlab_runner._apply_rl_spec(None, None)


def test_applier_tolerates_partial_cfg() -> None:
    """Defensive per-mutation contract — a cfg missing every touched
    surface must not raise."""
    full = {"env_spec_version": 1,
            "shared": {"zero_velocity_commands": True,
                       "orientation_termination_deg": 100.0,
                       "episode_length_s": 8.0,
                       "push_events": {"enabled": False}},
            "train": {"reset_height_offset_m": [0.0, 0.2],
                      "min_base_height_termination_m": 0.25,
                      "reset_joint_position_offset_rad": [-0.1, 0.1],
                      "friction_range": [0.4, 1.0],
                      "entropy_coef_scale": 1.5}}
    assert es.validate_env_spec(full) == []
    _mjlab_runner._apply_env_spec(SimpleNamespace(), full, train=True)
    _mjlab_runner._apply_env_spec(
        SimpleNamespace(commands={}, terminations={}), full, train=True)


# ── §D17: _apply_eval_reset — stage-fixed eval-rollout reset override ──────
def test_apply_eval_reset_writes_midpoint_pose_range() -> None:
    """Height/pitch/roll each collapse to a SINGLE deterministic
    (lo, hi) == (v, v) pair — not a train-style range — and vertical
    velocity is likewise pinned to exactly zero."""
    cfg = _fake_cfg_full()
    payload = {
        "reset_height_offset_m": -0.65,
        "reset_vertical_velocity_mps": 0.0,
        "reset_pitch_offset_rad": 1.5708,
        "reset_roll_offset_rad": 0.0,
    }
    _mjlab_runner._apply_eval_reset(cfg, payload)
    pose_range = cfg.events["reset_base"].params["pose_range"]
    assert pose_range["z"] == (-0.65, -0.65)
    assert pose_range["pitch"] == pytest.approx((1.5708, 1.5708))
    assert pose_range["roll"] == (0.0, 0.0)
    vr = cfg.events["reset_base"].params["velocity_range"]
    assert vr["z"] == (0.0, 0.0)


def test_apply_eval_reset_injects_joint_target_with_zero_noise() -> None:
    """The joint-posture reset event is injected exactly like the train
    path's, but with `joint_pos_noise` forced to whatever the payload
    says (derive_eval_reset always emits 0.0 — reproducible eval)."""
    pytest.importorskip("mjlab")
    cfg = _fake_cfg_full()
    payload = {
        "reset_joint_pos_target": [0.1, -0.2, 0.3],
        "reset_joint_pos_noise_rad": 0.0,
    }
    _mjlab_runner._apply_eval_reset(
        cfg, payload, task_id="Mjlab-Velocity-Flat-3dof-Testbot")
    ev = cfg.events["reset_robot_joints_to_reference"]
    assert ev.mode == "reset"
    assert list(ev.params["joint_pos_target"].tolist()) == pytest.approx(
        [0.1, -0.2, 0.3])
    assert ev.params["joint_pos_noise"] == pytest.approx(0.0)


def test_apply_eval_reset_joint_target_length_mismatch_skips() -> None:
    """Same defensive contract as the train-side joint-target injection:
    a length mismatch against the robot's canonical joint count is a
    clear skip, never a silent misassignment."""
    pytest.importorskip("mjlab")
    cfg = _fake_cfg_full()
    payload = {"reset_joint_pos_target": [0.1, -0.2, 0.3]}
    _mjlab_runner._apply_eval_reset(
        cfg, payload, task_id="Mjlab-Velocity-Flat-Unitree-G1")
    assert "reset_robot_joints_to_reference" not in cfg.events


def test_apply_eval_reset_removes_fell_over_termination() -> None:
    cfg = _fake_cfg_full()
    receipt = _mjlab_runner._apply_eval_reset(
        cfg, {"fell_over_termination": False}, strict=True)
    assert "fell_over" not in cfg.terminations
    assert receipt["applied"] == ["fell_over_termination"]
    assert receipt["dead"] == []
    assert receipt["errors"] == []


def test_apply_eval_reset_strict_accepts_fell_over_already_absent() -> None:
    cfg = _fake_cfg_full()
    cfg.terminations.pop("fell_over", None)

    receipt = _mjlab_runner._apply_eval_reset(
        cfg, {"fell_over_termination": False}, strict=True,
    )

    assert receipt["applied"] == ["fell_over_termination"]
    assert receipt["dead"] == []
    assert receipt["errors"] == []


def test_apply_eval_reset_absent_payload_is_byte_identical_noop() -> None:
    """None and {} must leave the cfg completely untouched — the default
    (every non-get-up stage) and the documented no-arg contract."""
    for payload in (None, {}):
        cfg = _fake_cfg_full()
        before_pose = dict(cfg.events["reset_base"].params["pose_range"])
        before_terms = dict(cfg.terminations)
        _mjlab_runner._apply_eval_reset(cfg, payload)
        assert cfg.events["reset_base"].params["pose_range"] == before_pose
        assert cfg.terminations.keys() == before_terms.keys()
        assert "reset_robot_joints_to_reference" not in cfg.events


def test_apply_eval_reset_tolerates_partial_cfg() -> None:
    """Defensive per-mutation contract, same as `_apply_env_spec` —
    a cfg missing every touched surface must not raise."""
    payload = {
        "reset_height_offset_m": -0.5,
        "reset_pitch_offset_rad": 1.2,
        "reset_joint_pos_target": [0.1, 0.2],
        "fell_over_termination": False,
    }
    _mjlab_runner._apply_eval_reset(SimpleNamespace(), payload)
    _mjlab_runner._apply_eval_reset(
        SimpleNamespace(events={}, terminations={}), payload)


def test_apply_eval_reset_strict_rejects_unapplied_fields() -> None:
    payload = {
        "reset_height_offset_m": -0.5,
        "fell_over_termination": False,
    }
    with pytest.raises(RuntimeError, match="not applied exactly"):
        _mjlab_runner._apply_eval_reset(
            SimpleNamespace(), payload, strict=True,
        )


def test_load_explicit_eval_reset_fails_closed(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable/invalid"):
        _mjlab_runner._load_explicit_eval_reset(str(malformed))

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must contain a JSON object"):
        _mjlab_runner._load_explicit_eval_reset(str(non_object))


def test_env_spec_application_receipt_distinguishes_applied_and_dead() -> None:
    spec = {
        "env_spec_version": 1,
        "shared": {"episode_length_s": 7.0},
    }
    applied = _mjlab_runner._apply_env_spec(
        _fake_cfg_full(), spec, train=False,
    )
    assert applied["phase"] == "rollout"
    assert applied["requested"] == ["episode_length_s"]
    assert applied["dead"] == []
    assert applied["errors"] == []

    dead = _mjlab_runner._apply_env_spec(
        SimpleNamespace(), spec, train=False,
    )
    assert dead["requested"] == ["episode_length_s"]
    assert dead["dead"] == ["episode_length_s"]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_runtime_seed_rejects_values_outside_shared_rng_domain(seed: int) -> None:
    with pytest.raises(ValueError, match="0..4294967295"):
        _mjlab_runner._apply_runtime_seed(seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_runtime_seed_accepts_shared_rng_domain_boundaries(seed: int) -> None:
    receipt = _mjlab_runner._apply_runtime_seed(seed)
    assert receipt["applied_seed"] == seed
    assert receipt["python_random"] is True
    assert receipt["numpy_global"] is True
    assert receipt["torch_global"] is True


# ── Runner-side resolution ─────────────────────────────────────────────────
def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"env_spec": "", "env_profile": "", **kw})


def test_resolve_env_spec_precedence(tmp_path) -> None:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"env_spec_version": 1,
                             "shared": {"episode_length_s": 7.0}}))
    # File wins over profile.
    spec = _mjlab_runner._resolve_env_spec(
        _args(env_spec=str(p), env_profile="jump"))
    assert spec["shared"]["episode_length_s"] == 7.0
    # Profile alone → the preset instance.
    spec = _mjlab_runner._resolve_env_spec(_args(env_profile="jump"))
    assert spec == es.jump_preset_spec()
    # Neither → None (task defaults).
    assert _mjlab_runner._resolve_env_spec(_args()) is None
    # Unknown profile → warn-and-ignore (historical contract).
    assert _mjlab_runner._resolve_env_spec(_args(env_profile="backflip")) is None


def test_resolve_env_spec_invalid_file_fails_loudly(tmp_path) -> None:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"env_spec_version": 1,
                             "shared": {"episode_length_s": 9999.0}}))
    with pytest.raises(ValueError, match="episode_length_s"):
        _mjlab_runner._resolve_env_spec(_args(env_spec=str(p)))


# ── Adapter + loader threading ─────────────────────────────────────────────
def _write_valid_spec(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(es.jump_preset_spec()))


def test_adapter_rejects_invalid_env_spec_at_init(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"env_spec_version": 1,
                               "train": {"entropy_coef_scale": 99.0}}))
    with pytest.raises(ValueError, match="entropy_coef_scale"):
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1",
                     env_spec_path=str(bad))
    with pytest.raises(ValueError, match="unreadable"):
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1",
                     env_spec_path=str(tmp_path / "missing.json"))


def _spawn_capture(adapter_kwargs, tmp_path, *, mode):
    """Instantiate the adapter with a patched subprocess runner and
    return the captured argv for train or rollout."""
    from unittest.mock import patch

    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-G1", num_envs=64,
        **adapter_kwargs)
    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa: ANN001
        captured["cmd"] = cmd
        return _FakeCompleted()

    (tmp_path / "checkpoint.pt").write_bytes(b"stub")
    (tmp_path / "metrics.json").write_text("{}")
    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        if mode == "train":
            adapter.train(reward_module_path=None, output_dir=tmp_path,
                          steps=1, seed=1)
        else:
            adapter.rollout(checkpoint_path=tmp_path / "checkpoint.pt",
                            output_dir=tmp_path, n_episodes=1)
    return captured["cmd"]


@pytest.mark.parametrize("mode", ["train", "rollout"])
def test_adapter_passes_env_spec_and_wins_over_profile(tmp_path, mode) -> None:
    pytest.importorskip("mjlab")
    spec_path = tmp_path / "env" / "current.json"
    _write_valid_spec(spec_path)
    cmd = _spawn_capture(
        {"env_spec_path": str(spec_path), "env_profile": "jump"},
        tmp_path, mode=mode)
    assert "--env-spec" in cmd
    assert cmd[cmd.index("--env-spec") + 1] == str(spec_path.resolve())
    assert "--env-profile" not in cmd


def test_load_adapter_injects_project_env_spec(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.base import load_adapter

    (tmp_path / "config.toml").write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-Go1", '
        'num_envs = 64 }\n')
    _write_valid_spec(tmp_path / "env" / "current.json")
    adapter = load_adapter(tmp_path / "config.toml")
    assert adapter.env_spec_path == str(
        (tmp_path / "env" / "current.json").resolve())
    # Without the file: nothing injected (byte-identical old behavior).
    (tmp_path / "env" / "current.json").unlink()
    adapter2 = load_adapter(tmp_path / "config.toml")
    assert adapter2.env_spec_path == ""


# ── §D17: eval_reset_path threading (adapter cmd + loader convention) ──────
def _write_valid_eval_reset(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reset_height_offset_m": -0.65,
        "reset_vertical_velocity_mps": 0.0,
        "fell_over_termination": False,
    }))


def test_adapter_rejects_invalid_eval_reset_at_init(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    bad = tmp_path / "bad.json"
    bad.write_text("not json{{{")
    with pytest.raises(ValueError, match="unreadable"):
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1",
                     eval_reset_path=str(bad))
    with pytest.raises(ValueError, match="unreadable"):
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1",
                     eval_reset_path=str(tmp_path / "missing.json"))
    not_obj = tmp_path / "list.json"
    not_obj.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="JSON object"):
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1",
                     eval_reset_path=str(not_obj))


def test_rollout_passes_eval_reset_flag(tmp_path) -> None:
    """`--eval-reset` is only ever passed to ROLLOUT (there is no train
    equivalent — this is an eval-only override by construction)."""
    pytest.importorskip("mjlab")
    reset_path = tmp_path / "eval_reset.json"
    _write_valid_eval_reset(reset_path)
    cmd = _spawn_capture(
        {"eval_reset_path": str(reset_path)}, tmp_path, mode="rollout")
    assert "--eval-reset" in cmd
    assert cmd[cmd.index("--eval-reset") + 1] == str(reset_path.resolve())


def test_train_never_receives_eval_reset_flag(tmp_path) -> None:
    pytest.importorskip("mjlab")
    reset_path = tmp_path / "eval_reset.json"
    _write_valid_eval_reset(reset_path)
    cmd = _spawn_capture(
        {"eval_reset_path": str(reset_path)}, tmp_path, mode="train")
    assert "--eval-reset" not in cmd


def test_rollout_without_eval_reset_path_omits_flag(tmp_path) -> None:
    pytest.importorskip("mjlab")
    cmd = _spawn_capture({}, tmp_path, mode="rollout")
    assert "--eval-reset" not in cmd


def test_load_adapter_injects_project_eval_reset(tmp_path) -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.base import load_adapter

    (tmp_path / "config.toml").write_text(
        '[adapter]\n'
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        'config = { task_id = "Mjlab-Velocity-Flat-Unitree-Go1", '
        'num_envs = 64 }\n')
    _write_valid_eval_reset(tmp_path / "env" / "eval_reset.json")
    adapter = load_adapter(tmp_path / "config.toml")
    assert adapter.eval_reset_path == str(
        (tmp_path / "env" / "eval_reset.json").resolve())
    # Without the file: nothing injected (byte-identical old behavior).
    (tmp_path / "env" / "eval_reset.json").unlink()
    adapter2 = load_adapter(tmp_path / "config.toml")
    assert adapter2.eval_reset_path == ""


# ── §sim2real physics domain randomization ─────────────────────────────────
def test_physics_dr_knobs_validate() -> None:
    """The new physics-DR axes (mass/gains/motor/damping/armature/CoM/friction)
    validate inside their envelopes and reject outside."""
    spec = {"env_spec_version": 1, "train": {
        "body_mass_scale_range": [0.8, 1.2],
        "com_offset_m": 0.05,
        "pd_kp_scale_range": [0.9, 1.1],
        "pd_kd_scale_range": [0.9, 1.1],
        "motor_strength_scale_range": [0.9, 1.1],
        "joint_damping_scale_range": [0.5, 1.5],
        "joint_armature_scale_range": [0.5, 1.5],
        "body_friction_range": [0.4, 1.2],
    }}
    assert es.validate_env_spec(spec) == []
    # all are diagnoser-tunable across iterations
    for k in ("body_mass_scale_range", "pd_kp_scale_range",
              "motor_strength_scale_range", "joint_damping_scale_range"):
        assert k in es.ITERABLE_TRAIN_KEYS
    # out-of-envelope rejected
    bad = es.validate_env_spec({"env_spec_version": 1,
                                "train": {"body_mass_scale_range": [0.5, 9.0]}})
    assert any("body_mass_scale_range" in e for e in bad)


def test_physics_dr_events_wire_train_only() -> None:
    """Physics-DR knobs become startup dr.* events on the robot for TRAIN, and
    NONE of them touch the rollout cfg (evaluation stays un-randomized)."""
    spec = {"env_spec_version": 1, "train": {
        "body_mass_scale_range": [0.8, 1.2],
        "pd_kp_scale_range": [0.9, 1.1],
        "motor_strength_scale_range": [0.9, 1.1],
        "body_friction_range": [0.4, 1.2],
    }}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    installed = {k for k in cfg.events if k.startswith("env_dr__")}
    assert {"env_dr__body_mass", "env_dr__pd_gains", "env_dr__motor_strength",
            "env_dr__body_friction"} <= installed
    assert cfg.events["env_dr__body_mass"].mode == "startup"
    # rollout: no physics DR events (train-only)
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert not any(k.startswith("env_dr__") for k in cfg2.events)


def test_always_on_physics_dr_defaults_on_train() -> None:
    """Even with NO spec, a TRAIN call gets the crash-safe baseline physics DR
    (mass/damping/armature) — the 'in any case' guarantee — while a rollout with
    no spec stays completely untouched."""
    for empty in (None, {}):
        cfg = _fake_cfg_full()
        _mjlab_runner._apply_env_spec(cfg, empty, train=True)
        installed = {k for k in cfg.events if k.startswith("env_dr__")}
        assert installed == {"env_dr__body_mass", "env_dr__joint_damping",
                             "env_dr__joint_armature"}
    # rollout with no spec: still a no-op (metric sees the true task)
    cfg_eval = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg_eval, None, train=False)
    assert not any(k.startswith("env_dr__") for k in cfg_eval.events)


def test_explicit_physics_dr_overrides_default() -> None:
    """An explicit range wins over the baseline default for the same axis."""
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(
        cfg, {"env_spec_version": 1,
              "train": {"body_mass_scale_range": [0.5, 1.5]}}, train=True)
    assert cfg.events["env_dr__body_mass"].params["ranges"] == (0.5, 1.5)


# ── §DeepMimic phase RSI (full-trajectory random-phase reset + joint_vel) ────
def test_rsi_trajectory_validates_and_is_not_iterable() -> None:
    spec = {"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]],
        "reset_joint_vel_trajectory": [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]],
    }}
    assert es.validate_env_spec(spec) == []
    # ragged (uneven joint width across frames) is rejected
    ragged = es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2], [0.1, 0.2, 0.3]]}})
    assert any("ragged" in e for e in ragged)
    # velocity out of the generous rail is rejected
    fast = es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_vel_trajectory": [[999.0, 0.0]]}})
    assert any("reset_joint_vel_trajectory" in e for e in fast)
    # derived clip data — not a diagnoser-tunable scalar knob
    assert "reset_joint_pos_trajectory" not in es.ITERABLE_TRAIN_KEYS


def test_rsi_trajectory_wires_phase_reset_event() -> None:
    spec = {"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        "reset_joint_vel_trajectory": [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]],
    }}
    assert es.validate_env_spec(spec) == []
    cfg = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg, spec, train=True, task_id="")
    ev = cfg.events["reset_robot_joints_to_reference"]
    assert ev.mode == "reset"
    assert "joint_pos_traj" in ev.params and "joint_vel_traj" in ev.params
    # rollout: reference reset is train-only
    cfg2 = _fake_cfg_full()
    _mjlab_runner._apply_env_spec(cfg2, spec, train=False)
    assert "reset_robot_joints_to_reference" not in cfg2.events


def test_phase_rsi_reset_samples_pos_and_vel_from_same_frame() -> None:
    """Each env resets to a RANDOM reference frame, and its joint velocity comes
    from that SAME frame (not the default zeros the single-target path uses)."""
    import torch
    from types import SimpleNamespace

    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.5, 0.5], [1.0, -1.0, 1.0]])
    vel = torch.tensor([[0.0, 0.0, 0.0], [2.0, -2.0, 1.0], [4.0, -4.0, 2.0]])

    class _Asset:
        data = SimpleNamespace(
            default_joint_vel=torch.zeros(8, 3),
            soft_joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * 3] * 8))

        def write_joint_state_to_sim(self, jp, jv, env_ids, joint_ids):
            self.pos, self.vel = jp.clone(), jv.clone()

    asset = _Asset()
    env = SimpleNamespace(num_envs=8, device="cpu", scene={"robot": asset})
    torch.manual_seed(3)
    _mjlab_runner.reset_joints_to_reference(
        env, None, joint_pos_traj=pos, joint_vel_traj=vel,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)))
    for r in range(8):
        match = [k for k in range(3) if torch.allclose(asset.pos[r], pos[k], atol=1e-4)]
        assert match, asset.pos[r].tolist()
        assert torch.allclose(asset.vel[r], vel[match[0]], atol=1e-4)


def test_pd_gains_dr_skipped_on_non_pd_robot() -> None:
    """pd_gains / motor_strength DR must be SKIPPED (not installed → no GPU
    startup crash) when the robot's actuators are not PD/position — e.g. a
    motor-actuated cartpole."""
    cfg = _fake_cfg_full()

    class XmlActuatorCfg:            # motor actuator (command_field != position)
        command_field = None

    cfg.scene = SimpleNamespace(entities={
        "cartpole": SimpleNamespace(articulation=SimpleNamespace(
            actuators=(XmlActuatorCfg(),)))})
    spec = {"env_spec_version": 1, "train": {
        "pd_kp_scale_range": [0.9, 1.1], "motor_strength_scale_range": [0.9, 1.1],
        "body_mass_scale_range": [0.9, 1.1]}}
    _mjlab_runner._apply_env_spec(cfg, spec, train=True)
    assert "env_dr__pd_gains" not in cfg.events
    assert "env_dr__motor_strength" not in cfg.events
    # mass DR (actuator-independent) still applies on the non-PD robot
    assert "env_dr__body_mass" in cfg.events


def test_rsi_trajectory_pairing_validation() -> None:
    # velocity trajectory without a position trajectory is rejected
    e1 = es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_vel_trajectory": [[0.0, 0.0], [1.0, 1.0]]}})
    assert any("requires reset_joint_pos_trajectory" in e for e in e1)
    # mismatched frame count K rejected
    e2 = es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2], [0.3, 0.4]],
        "reset_joint_vel_trajectory": [[0.0, 0.0]]}})
    assert any("frames" in e and "reset_joint_vel_trajectory" in e for e in e2)
    # mismatched joint width J rejected
    e3 = es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2, 0.3]],
        "reset_joint_vel_trajectory": [[0.0, 0.0]]}})
    assert any("width" in e for e in e3)
    # matched pair validates
    assert es.validate_env_spec({"env_spec_version": 1, "train": {
        "reset_joint_pos_trajectory": [[0.1, 0.2], [0.3, 0.4]],
        "reset_joint_vel_trajectory": [[0.0, 0.0], [1.0, -1.0]]}}) == []
