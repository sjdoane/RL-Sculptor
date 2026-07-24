"""Unit tests for MjlabAdapter + stub adapters.

Covers the mocked / import-only surface. The real-GPU smoke test lives
in tests/test_mjlab_gpu.py behind `@pytest.mark.gpu`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from sculptor.adapters.base import (
    ComponentProbe,
    RewardContract,
    RolloutResult,
    TrainResult,
)


def test_to_host_numpy_moves_tensor_metadata_to_cpu() -> None:
    """CUDA-like simulator metadata is copied to host before NumPy sees it."""
    import numpy as np

    from sculptor.adapters._mjlab_runner import _to_host_numpy

    calls: list[str] = []

    class FakeCudaTensor:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    result = _to_host_numpy(FakeCudaTensor())

    assert calls == ["detach", "cpu", "numpy"]
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


def test_to_host_numpy_keeps_plain_metadata_supported() -> None:
    import numpy as np

    from sculptor.adapters._mjlab_runner import _to_host_numpy

    np.testing.assert_array_equal(_to_host_numpy([[0.0, 1.0]]), [[0.0, 1.0]])


def test_authored_waypoint_command_rewards_keep_nominal_weight() -> None:
    """Only a successfully installed goal command earns full supervision."""
    from sculptor.adapters._mjlab_runner import (
        _full_weight_authored_command_rewards,
    )

    bundle = SimpleNamespace(
        manifest=SimpleNamespace(task_shared={
            "goal": {"type": "waypoint_sequence"},
        }),
        runtime_adjustments=(
            "command:velocity→goal-conditioned waypoint traversal",
        ),
    )
    assert _full_weight_authored_command_rewards(bundle) == frozenset({
        "track_linear_velocity", "track_angular_velocity",
    })

    # A goal declaration without a compatible installed command surface must
    # retain the conservative realism-floor behavior.
    bundle.runtime_adjustments = ()
    assert _full_weight_authored_command_rewards(bundle) == frozenset()

    # Non-navigation authored tasks never inherit locomotion-specific terms.
    bundle.manifest.task_shared["goal"]["type"] = "object_region"
    bundle.runtime_adjustments = (
        "command:velocity→goal-conditioned waypoint traversal",
    )
    assert _full_weight_authored_command_rewards(bundle) == frozenset()


def test_authored_terminal_standing_requires_installed_dwell_command() -> None:
    from sculptor.adapters._mjlab_runner import (
        _authored_terminal_standing_enabled,
    )

    bundle = SimpleNamespace(
        manifest=SimpleNamespace(task_shared={
            "goal": {
                "type": "waypoint_sequence",
                "success": {"hold_s": 2.0},
            },
        }),
        runtime_adjustments=(
            "command:velocity→goal-conditioned waypoint traversal",
        ),
    )
    assert _authored_terminal_standing_enabled(bundle)
    bundle.manifest.task_shared["goal"]["success"]["hold_s"] = 0.0
    assert not _authored_terminal_standing_enabled(bundle)
    bundle.manifest.task_shared["goal"]["success"]["hold_s"] = 2.0
    bundle.runtime_adjustments = ()
    assert not _authored_terminal_standing_enabled(bundle)


def test_authored_terminal_stillness_balances_command_supervision() -> None:
    from sculptor.adapters._mjlab_runner import (
        _authored_terminal_stillness_weight,
    )

    rewards = {
        "track_linear_velocity": SimpleNamespace(weight=2.0),
        "track_angular_velocity": SimpleNamespace(weight=2.0),
        "unrelated_posture": SimpleNamespace(weight=20.0),
    }
    authored_terms = frozenset({
        "track_linear_velocity",
        "track_angular_velocity",
    })

    assert _authored_terminal_stillness_weight(
        rewards, authored_terms) == 4.0
    # Missing, malformed, and zero-weight command terms retain the safe floor.
    assert _authored_terminal_stillness_weight(
        {"track_linear_velocity": SimpleNamespace(weight="bad")},
        authored_terms,
    ) == 1.0


def test_authored_forbidden_contact_supervision_uses_compiled_sensors() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import (
        _authored_forbidden_contact_penalty,
        _authored_forbidden_contact_sensor_names,
        _authored_forbidden_contact_weight,
    )

    bundle = SimpleNamespace(manifest=SimpleNamespace(task_shared={
        "contacts": {
            "forbidden": [
                ["robot:any", "object:first"],
                ["robot:any", "object:second"],
            ],
        },
    }))
    names = _authored_forbidden_contact_sensor_names(bundle)
    assert names == (
        "authored_contact__forbidden__0",
        "authored_contact__forbidden__1",
    )
    assert _authored_forbidden_contact_weight(
        {
            "track_linear_velocity": SimpleNamespace(weight=2.0),
            "track_angular_velocity": SimpleNamespace(weight=2.0),
        },
        frozenset({"track_linear_velocity", "track_angular_velocity"}),
    ) == 8.0

    scene = {
        names[0]: SimpleNamespace(
            data=SimpleNamespace(found=torch.tensor([
                [False], [True], [False],
            ])),
        ),
        names[1]: SimpleNamespace(
            data=SimpleNamespace(found=torch.tensor([
                [False], [False], [True],
            ])),
        ),
    }
    env = SimpleNamespace(num_envs=3, device="cpu", scene=scene)
    penalty = _authored_forbidden_contact_penalty(
        env, sensor_names=names)
    assert penalty.tolist() == [0.0, 1.0, 1.0]


def test_authored_terminal_stillness_is_dense_and_phase_gated() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import (
        _authored_terminal_stillness_reward,
    )

    command = SimpleNamespace(
        is_standing_env=torch.tensor([True, True, False]))

    class CommandManager:
        active_terms = ("route",)

        @staticmethod
        def get_term(name):
            assert name == "route"
            return command

    data = SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        root_link_ang_vel_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        joint_vel=torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]),
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        command_manager=CommandManager(),
        scene={"robot": SimpleNamespace(data=data)},
    )

    reward = _authored_terminal_stillness_reward(
        env, lin_std=0.12, ang_std=0.5, joint_std=1.0)

    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    assert 0.3 < reward[1].item() < 0.5
    assert reward[2].item() == 0.0


def test_authored_terminal_stillness_rewards_continuity_and_resets() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import (
        _build_authored_terminal_stillness_term_class,
    )

    command = SimpleNamespace(
        is_standing_env=torch.tensor([True, True, False]))

    class CommandManager:
        active_terms = ("route",)

        @staticmethod
        def get_term(name):
            assert name == "route"
            return command

    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(3, 3),
        root_link_ang_vel_b=torch.zeros(3, 3),
        joint_vel=torch.zeros(3, 2),
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        step_dt=0.02,
        command_manager=CommandManager(),
        scene={"robot": SimpleNamespace(data=data)},
    )
    term_type = _build_authored_terminal_stillness_term_class()
    # Match ManagerBase's real class-backed term construction contract.
    term = term_type(cfg=SimpleNamespace(), env=env)
    params = {
        "lin_std": 0.12,
        "ang_std": 0.5,
        "joint_std": 1.0,
        "hold_s": 0.1,
        "continuity_scale": 2.0,
    }

    first = term(env, **params)
    second = term(env, **params)
    assert second[0].item() > first[0].item() > 1.0
    assert second[1].item() > first[1].item() > 1.0
    assert first[2].item() == 0.0

    # Reward-manager selective reset clears only the requested environment.
    term.reset(torch.tensor([0]))
    after_reset = term(env, **params)
    torch.testing.assert_close(after_reset[0], first[0])
    assert after_reset[1].item() > second[1].item()
    assert after_reset[2].item() == 0.0

    # A corrective step breaks the uninterrupted dwell and loses accumulated
    # progress instead of retaining credit for a high quiet-sample fraction.
    # The potential loss is a per-second rate because RewardManager scales the
    # returned value by dt; keep it strong enough to survive that integration.
    data.root_link_lin_vel_b[0, 0] = 0.2
    interrupted = term(env, **params)
    assert interrupted[0].item() < -10.0
    assert interrupted[1].item() > second[1].item()

    # In-place stepping and rotation must also break the uninterrupted hold,
    # even when horizontal base translation remains below the task threshold.
    data.root_link_lin_vel_b[0, 0] = 0.0
    data.joint_vel[1, 0] = 2.0
    joint_interrupted = term(env, **params)
    assert joint_interrupted[1].item() < -10.0

    data.joint_vel[1, 0] = 0.0
    term(env, **params)
    data.root_link_ang_vel_b[1, 2] = 1.0
    angular_interrupted = term(env, **params)
    assert angular_interrupted[1].item() < -10.0


def test_rollout_evidence_excludes_metric_only_channels() -> None:
    """Diagnosis receives batch progress, never frozen completion truth."""
    import numpy as np

    from sculptor.adapters._mjlab_runner import (
        _reward_visible_rollout_evidence,
    )

    catalog = SimpleNamespace(channels=(
        SimpleNamespace(
            name="goal__route__distance", access="shared_shaping",
            metric_role="progress", producer="waypoint_distance"),
        SimpleNamespace(
            name="goal__route__success", access="metric_only",
            metric_role="completion", producer="success_hold"),
        SimpleNamespace(
            name="object__box__lin_vel_w", access="shared_shaping",
            metric_role="state", producer="entity_state"),
        SimpleNamespace(
            name="object__box__pos_w", access="shared_shaping",
            metric_role="state", producer="entity_state"),
    ))
    trajectory = {
        "goal__route__distance": np.asarray([
            [3.0, 4.0], [1.0, 2.0], [0.0, 99.0]], dtype=np.float32),
        "goal__route__success": np.ones((3, 2), dtype=bool),
        "object__box__lin_vel_w": np.asarray([
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        ], dtype=np.float32),
        "object__box__pos_w": np.zeros((3, 2, 3), dtype=np.float32),
    }
    # The last row is an auto-reset and must not contaminate the summary.
    valid = np.asarray([[True, True], [True, True], [False, False]])
    evidence = _reward_visible_rollout_evidence(
        trajectory, catalog, valid)

    channels = evidence["channels"]
    assert set(channels) == {
        "goal__route__distance", "object__box__lin_vel_w"}
    assert channels["goal__route__distance"]["final_median"] == 1.5
    assert channels["goal__route__distance"]["final_zero_fraction"] == 0.0
    assert channels["object__box__lin_vel_w"]["max_over_time_median"] == 0.1
    assert "goal__route__success" not in channels

def test_base_reward_contract_default_fields() -> None:
    c = RewardContract(observation_space_spec=None, action_space_spec=None)
    assert c.supports_batched is False
    assert c.training_device == "any"
    assert c.min_gpu_memory_gb is None
    assert c.state_schema is None


def test_scalar_policy_std_guard_clamps_initial_and_optimizer_values() -> None:
    """Legacy rsl_rl scalar exploration must never cross below zero."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import _install_scalar_std_guard

    std_param = torch.nn.Parameter(torch.tensor([-0.25, 0.5]))
    optimizer = torch.optim.SGD([std_param], lr=1.0)
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            actor=SimpleNamespace(
                distribution=SimpleNamespace(std_param=std_param),
            ),
            optimizer=optimizer,
        ),
    )

    handle = _install_scalar_std_guard(runner, minimum=0.01)
    assert handle is not None
    assert torch.all(std_param >= 0.01)

    # A finite optimizer update would drive both values negative without the
    # post-step hook; the guard repairs them before the next action sample.
    std_param.grad = torch.ones_like(std_param)
    optimizer.step()
    assert torch.all(std_param >= 0.01)
    handle.remove()


def test_scalar_policy_std_guard_ignores_other_distributions() -> None:
    """Log-std and non-Gaussian policies remain byte-for-byte untouched."""
    from types import SimpleNamespace

    from sculptor.adapters._mjlab_runner import _install_scalar_std_guard

    runner = SimpleNamespace(
        alg=SimpleNamespace(
            actor=SimpleNamespace(distribution=SimpleNamespace()),
            optimizer=SimpleNamespace(),
        ),
    )
    assert _install_scalar_std_guard(runner) is None


def test_sculpted_reward_installs_non_timeout_termination_economics() -> None:
    """A custom reward cannot improve return merely by ending sooner."""
    from types import SimpleNamespace

    from sculptor.adapters._mjlab_runner import (
        _SCULPTOR_FAILURE_WEIGHT,
        _SCULPTOR_SURVIVAL_WEIGHT,
        _install_sculptor_termination_economics,
    )

    class FakeRewardTermCfg:
        def __init__(self, *, func, weight):
            self.func = func
            self.weight = weight

    def is_alive(_env):
        return "alive"

    def is_terminated(_env):
        return "terminated"

    native = object()
    rewards = {"native_task_term": native}
    mdp = SimpleNamespace(is_alive=is_alive, is_terminated=is_terminated)

    _install_sculptor_termination_economics(
        rewards,
        FakeRewardTermCfg,
        mdp,
    )

    assert rewards["native_task_term"] is native
    assert rewards["sculptor_survival"].func is is_alive
    assert rewards["sculptor_survival"].weight == _SCULPTOR_SURVIVAL_WEIGHT
    assert rewards["sculptor_survival"].weight > 0
    assert rewards["sculptor_failure"].func is is_terminated
    assert rewards["sculptor_failure"].weight == _SCULPTOR_FAILURE_WEIGHT
    assert rewards["sculptor_failure"].weight < -_SCULPTOR_SURVIVAL_WEIGHT


def test_component_probe_dataclass_shape() -> None:
    p = ComponentProbe(ok=True, components={"x": 1.0}, total=1.0, error=None)
    assert p.ok is True
    assert p.components == {"x": 1.0}
    assert p.total == 1.0


def test_gym_sb3_still_satisfies_abc() -> None:
    """Sanity — importing GymSB3Adapter and calling reward_contract
    still works after the ABC extension; contract is scalar-only."""
    from sculptor.adapters.gym_sb3 import GymSB3Adapter

    adapter = GymSB3Adapter(env_id="Hopper-v4", n_envs=2)
    c = adapter.reward_contract()
    assert c.supports_batched is False
    assert c.training_device == "any"


def test_mjlab_adapter_validates_task_id() -> None:
    """Instantiation with an unknown task_id raises ValueError."""
    from sculptor.adapters.mjlab import MjlabAdapter

    with pytest.raises(ValueError, match="not registered in mjlab"):
        MjlabAdapter(task_id="Mjlab-Nope-Not-A-Real-Task")


def test_mjlab_adapter_reward_contract_is_batched() -> None:
    """Happy-path contract shape. Requires mjlab import (task_id
    validation in __init__). Auto-skip if mjlab is missing."""
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1")
    c = adapter.reward_contract()
    assert c.supports_batched is True
    assert c.training_device == "gpu"
    assert c.min_gpu_memory_gb is not None and c.min_gpu_memory_gb > 0
    assert c.state_schema is not None
    assert c.info_schema is not None
    # Keys the sculptor reward-term snapshot emits for velocity tasks.
    expected_keys = {
        "qpos", "qvel", "base_lin_vel_b", "base_ang_vel_b",
        "projected_gravity_b", "actuator_force", "command_vel",
    }
    assert set(c.state_schema.keys()) == expected_keys
    # §Ship 46: per-foot kick channels are G1-only; Go1 keeps the base
    # 6-key info contract (these keys must NOT leak into the quadruped
    # contract, or edit.py would ground formulas the runner zero-fills).
    assert "left_foot_contact" not in (c.expected_info_keys or [])
    assert "base_horizontal_speed" not in (c.expected_info_keys or [])


def test_mjlab_g1_state_schema_differs_from_go1() -> None:
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    g1 = MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-G1")
    go1 = MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1")
    assert g1.reward_contract().state_schema != go1.reward_contract().state_schema


def test_enforce_actuator_limits_swaps_to_dcmotor_with_real_velocity_limits(monkeypatch) -> None:
    """§actuator-limit enforcement: RS_ENFORCE_ACTUATOR_LIMITS=1 swaps every
    BuiltinPositionActuatorCfg → DcMotorActuatorCfg carrying the robot's REAL motor
    no-load speed (G1 knee 20, Go1 calf 20.06), so the sim enforces velocity, not
    just torque. Flag OFF is a no-op (existing runs bit-identical); an unknown
    joint pattern is left unchanged (never invents a limit). Config-only — no GPU."""
    pytest.importorskip("mjlab")
    from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
    from mjlab.tasks.registry import load_env_cfg

    from sculptor.adapters._mjlab_runner import (
        _enforce_actuator_limits,
        _recover_velocity_limit,
    )

    def _acts(cfg):
        return list(cfg.scene.entities["robot"].articulation.actuators)

    # flag explicitly OFF → byte-identical no-op (default is now ON, so set "0")
    monkeypatch.setenv("RS_ENFORCE_ACTUATOR_LIMITS", "0")
    g1_off = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    _enforce_actuator_limits(g1_off)
    assert all(isinstance(a, BuiltinPositionActuatorCfg) for a in _acts(g1_off))

    # default (unset) is ON → all groups swapped, real velocity_limits, fields preserved
    monkeypatch.delenv("RS_ENFORCE_ACTUATOR_LIMITS", raising=False)
    g1 = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    _enforce_actuator_limits(g1)
    g1a = _acts(g1)
    assert g1a and all(isinstance(a, DcMotorActuatorCfg) for a in g1a)
    knee = next(a for a in g1a if any("knee" in p for p in a.target_names_expr))
    assert knee.velocity_limit == 20.0
    assert knee.effort_limit == 139.0 and knee.saturation_effort == 139.0

    go1 = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go1")
    _enforce_actuator_limits(go1)
    calf = next(a for a in _acts(go1) if any("calf" in p for p in a.target_names_expr))
    assert isinstance(calf, DcMotorActuatorCfg) and calf.velocity_limit == 20.06

    # unknown joint pattern → no recoverable limit (caller leaves it unchanged)
    class _Fake:
        target_names_expr = (".*_mystery_joint",)

    assert _recover_velocity_limit(_Fake()) is None


# ── §Ship 46: per-foot kick channels in the G1 info contract ───────────────
def test_info_keys_for_task_adds_foot_channels_for_g1_only() -> None:
    """`_info_keys_for_task` is a pure function (no mjlab import): G1 gets
    the base info keys PLUS the per-foot kick channels; every other task
    family keeps the universal base set."""
    from sculptor.adapters.mjlab import (
        _G1_INFO_EXTRA,
        _INFO_KEYS,
        _info_keys_for_task,
    )

    assert _info_keys_for_task("Mjlab-Velocity-Flat-Unitree-G1") == (
        list(_INFO_KEYS) + list(_G1_INFO_EXTRA)
    )
    for other in ("Mjlab-Velocity-Flat-Unitree-Go1", "Mjlab-Cartpole-Balance"):
        assert _info_keys_for_task(other) == list(_INFO_KEYS)
    # The extras must include the exact channels the kick diagnoser kept
    # deferring (per-foot contact + swing velocity + height) + base travel.
    assert set(_G1_INFO_EXTRA) == {
        "left_foot_contact", "right_foot_contact",
        "left_foot_swing_speed", "right_foot_swing_speed",
        "left_foot_height", "right_foot_height",
        "base_horizontal_speed",
    }


def test_mjlab_g1_reward_contract_exposes_foot_kick_channels() -> None:
    """End-to-end: the G1 contract handed to edit.py/diagnose advertises
    the per-foot kick channels (so a kick formula grounds instead of
    deferring); Go1 does not."""
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter, _G1_INFO_EXTRA

    g1_keys = set(
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-G1")
        .reward_contract().expected_info_keys or []
    )
    assert set(_G1_INFO_EXTRA).issubset(g1_keys)
    assert {"base_height", "fallen"}.issubset(g1_keys)  # base set retained

    go1_keys = set(
        MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1")
        .reward_contract().expected_info_keys or []
    )
    assert not (set(_G1_INFO_EXTRA) & go1_keys)


def test_kick_formula_grounds_under_g1_contract_not_base() -> None:
    """THE CRUX PROOF (no GPU): a kick reward formula referencing the new
    foot channels is GROUNDED under the G1 contract — i.e. edit.py would
    APPLY it, not defer it with requires_env_extension. The same formula
    is UNGROUNDED under the old 6-key base set, proving the contract
    extension is load-bearing (it's what unblocks the kick the g1-kick-v3
    run could never express)."""
    from sculptor.edit import (
        _ALLOWED_MATH,
        _SIGNATURE_ARGS,
        _extract_formula_identifiers,
    )
    from sculptor.adapters.mjlab import _INFO_KEYS, _info_keys_for_task

    # A plausible single-leg kick term: reward forward swing speed of the
    # kicking foot while the other foot is planted, penalising travel.
    formula = (
        "right_foot_swing_speed * left_foot_contact "
        "- 0.5 * base_horizontal_speed"
    )
    idents = _extract_formula_identifiers(formula)

    g1_info = set(_info_keys_for_task("Mjlab-Velocity-Flat-Unitree-G1"))
    g1_allowed = _ALLOWED_MATH | _SIGNATURE_ARGS | g1_info
    assert not (idents - g1_allowed), (
        f"kick formula should ground under G1 contract; ungrounded: "
        f"{sorted(idents - g1_allowed)}"
    )

    base_allowed = _ALLOWED_MATH | _SIGNATURE_ARGS | set(_INFO_KEYS)
    assert idents - base_allowed, (
        "kick formula must be UNGROUNDED under the old base info set — "
        "otherwise the contract extension wasn't the unblocker"
    )


# ── S4 (bug #6.6 / T1): Cartpole fixed-base schema ──────────────────
def test_mjlab_cartpole_schema_is_minimal_fixed_base() -> None:
    """Cartpole ships inside mjlab (source=mjlab_builtin) and is a
    fixed-base articulation — 2 joints, 1 actuator, no floating root.
    `_schema_for_task` MUST return the 3-key cartpole schema for the
    known Cartpole task_ids so Claude-written rewards don't reach for
    `base_lin_vel_b` / `command_vel` (absent from the Cartpole env and
    would raise AttributeError in _snapshot).
    """
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import (
        _CARTPOLE_STATE_SCHEMA,
        _schema_for_task,
    )

    for task_id in (
        "Mjlab-Cartpole-Balance",
        "Mjlab-Cartpole-Swingup",
        "Something-cartpole-lower",  # lower-case fallback branch
    ):
        schema = _schema_for_task(task_id)
        assert schema == dict(_CARTPOLE_STATE_SCHEMA), (
            f"{task_id!r} should dispatch to cartpole schema"
        )
        # Explicit shape contract — guards against _CARTPOLE_STATE_SCHEMA
        # being silently widened later.
        assert schema == {
            "qpos": (2,),
            "qvel": (2,),
            "actuator_force": (1,),
        }


def test_mjlab_cartpole_adapter_reward_contract_is_minimal() -> None:
    """`MjlabAdapter(task_id="Mjlab-Cartpole-Balance")` → `reward_contract`
    returns the 3-key schema. End-to-end seam from adapter construction
    to the reward-module contract Sam's UI hands to Claude."""
    pytest.importorskip("mjlab")
    from sculptor.adapters.mjlab import MjlabAdapter

    adapter = MjlabAdapter(task_id="Mjlab-Cartpole-Balance")
    c = adapter.reward_contract()
    assert c.supports_batched is True
    assert c.state_schema is not None
    assert set(c.state_schema.keys()) == {"qpos", "qvel", "actuator_force"}


def test_mjlab_adapter_train_subprocess_construction() -> None:
    """Mock subprocess.run and verify the CLI args + env passed."""
    pytest.importorskip("mjlab")
    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        num_envs=1024,
        device="cuda:0",
        max_iterations=50,
    )

    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = '{"status": "ok", "checkpoint": "/tmp/out/checkpoint.pt"}'
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa
        captured["cmd"] = cmd
        captured["env"] = env
        return _FakeCompleted()

    output = Path("/tmp/sculptor-mjlab-test-out")
    output.mkdir(exist_ok=True)
    # Also drop a fake checkpoint file so train() post-check passes.
    (output / "checkpoint.pt").write_bytes(b"stub")
    (output / "metrics.json").write_text('{"status": "ok"}')

    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        result = adapter.train(
            reward_module_path=Path("/tmp/v0.py"),
            output_dir=output,
            steps=50,
            seed=7,
        )

    assert isinstance(result, TrainResult)
    assert result.checkpoint_path == output / "checkpoint.pt"
    # CLI shape.
    cmd = captured["cmd"]
    assert "train" in cmd
    assert "sculptor.adapters._mjlab_runner" in cmd
    assert "--task-id" in cmd
    assert "Mjlab-Velocity-Flat-Unitree-Go1" in cmd
    assert "--num-envs" in cmd and "1024" in cmd
    assert "--max-iterations" in cmd and "50" in cmd
    assert "--seed" in cmd and "7" in cmd
    assert "--reward-module-path" in cmd
    # S8-followup regression: --schema-keys MUST be passed so the
    # runner subprocess doesn't fall back to the 7-key velocity default
    # on non-Go1 tasks. Go1 happens to match the default 7 keys, but we
    # still require the flag to be passed explicitly.
    assert "--schema-keys" in cmd
    sk_idx = cmd.index("--schema-keys") + 1
    go1_keys = set(cmd[sk_idx].split(","))
    assert {"qpos", "qvel", "base_lin_vel_b", "command_vel"}.issubset(go1_keys), (
        f"Go1 schema keys: {go1_keys}"
    )
    # CUDA_VISIBLE_DEVICES pinning.
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_mjlab_adapter_passes_cartpole_schema_to_subprocess(tmp_path: Path) -> None:
    """Regression for Cartpole Test 1 failure (2026-04-22): the adapter
    MUST pass `--schema-keys qpos,qvel,actuator_force` for Cartpole task_ids.
    Pre-fix `self.schema_keys` defaulted to None → CLI flag omitted →
    runner used the 7-key velocity default → `SculptorRewardTerm._prev`
    gained a None entry for `command_vel` (Cartpole has no base_velocity
    command) → `reset()` crashed with 'NoneType does not support item
    assignment'."""
    pytest.importorskip("mjlab")
    from sculptor.adapters import mjlab as mjlab_mod
    from unittest.mock import patch

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Cartpole-Balance",
        num_envs=256,
        device="cuda:0",
        max_iterations=10,
    )
    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = '{"status": "ok"}'
        stderr = ""

    def fake_run(cmd, env=None, timeout=None):  # noqa
        captured["cmd"] = cmd
        return _FakeCompleted()

    output = Path("/tmp/sculptor-mjlab-test-out-cp")
    output.mkdir(exist_ok=True)
    (output / "checkpoint.pt").write_bytes(b"stub")
    (output / "metrics.json").write_text('{"status": "ok"}')

    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_run):
        adapter.train(
            reward_module_path=Path("/tmp/v0.py"),
            output_dir=output,
            steps=10,
            seed=1,
        )

    cmd = captured["cmd"]
    assert "--schema-keys" in cmd, (
        "adapter must always pass --schema-keys to avoid the 7-key "
        "velocity default clobbering Cartpole's 3-key schema"
    )
    sk_idx = cmd.index("--schema-keys") + 1
    cp_keys = set(cmd[sk_idx].split(","))
    assert cp_keys == {"qpos", "qvel", "actuator_force"}, (
        f"Cartpole schema keys on CLI: {cp_keys!r} — should be exactly "
        "qpos, qvel, actuator_force (no base_* / command_vel)"
    )
    # Negative assertion: the locomotion-only keys MUST NOT be present.
    for bad_key in ("command_vel", "base_lin_vel_b", "base_ang_vel_b",
                    "projected_gravity_b"):
        assert bad_key not in cp_keys, (
            f"Cartpole schema leaked locomotion key {bad_key!r}; this is "
            "the exact condition that caused the reset() NoneType crash."
        )


def test_mjlab_adapter_reward_batched_uses_compute_reward_batched(
    tmp_path: Path,
) -> None:
    """If the reward module exports compute_reward_batched, MjlabAdapter
    dispatches to it directly (NOT the default scalar-loop fallback)."""
    pytest.importorskip("mjlab")
    pytest.importorskip("torch")
    import torch

    module_path = tmp_path / "v0.py"
    module_path.write_text(
        "import torch\n"
        "REWARD_SPEC = {'version': 'v0', 'supports_batched': True}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return (0.5, {'c': 0.5})\n"
        "def compute_reward_batched(state, action, next_state, info):\n"
        "    n = action.shape[0]\n"
        "    return (torch.ones(n) * 42.0, {'c': torch.ones(n) * 42.0})\n"
    )

    from sculptor.adapters.mjlab import MjlabAdapter
    adapter = MjlabAdapter(task_id="Mjlab-Velocity-Flat-Unitree-Go1")

    state = {"qpos": torch.zeros(4, 18)}
    next_state = {"qpos": torch.zeros(4, 18)}
    action = torch.zeros(4, 12)
    info = {}
    rewards, components = adapter.reward_batched(
        module_path, state, action, next_state, info,
    )
    assert rewards.shape == (4,)
    assert (rewards == 42.0).all()
    assert components["c"].shape == (4,)


def test_stub_adapters_raise_not_implemented() -> None:
    from sculptor.adapters.isaac_lab import IsaacLabAdapter
    from sculptor.adapters.mjx import MjxAdapter
    from sculptor.adapters.rllib import RllibAdapter

    for cls in (IsaacLabAdapter, MjxAdapter, RllibAdapter):
        a = cls()
        with pytest.raises(NotImplementedError):
            a.train(
                reward_module_path=Path("/dev/null"),
                output_dir=Path("/tmp"),
                steps=1,
                seed=1,
            )
        with pytest.raises(NotImplementedError):
            a.rollout(
                checkpoint_path=Path("/dev/null"),
                output_dir=Path("/tmp"),
                n_episodes=1,
            )
        # reward_contract / compute_behavior_metrics return sensible defaults.
        c = a.reward_contract()
        assert isinstance(c, RewardContract)
        metrics = a.compute_behavior_metrics(
            RolloutResult(
                video_path=Path("/tmp/x.mp4"),
                keyframes_dir=Path("/tmp/kf"),
                trajectory_path=Path("/tmp/t.npz"),
                n_episodes=0,
            )
        )
        assert metrics["adapter_status"] == "stub"


def test_estimate_vram_static() -> None:
    from sculptor.adapters.mjlab import estimate_vram_static

    assert estimate_vram_static(num_envs=0) == pytest.approx(1.5)
    assert estimate_vram_static(num_envs=2048) == pytest.approx(1.5 + 2048 * 0.5 / 1024)
    # Conservative estimate for 8 GB VRAM (RTX 5070 Laptop): 1024 envs
    # should fit comfortably.
    est_1024 = estimate_vram_static(num_envs=1024)
    assert est_1024 < 8 * 0.85


def test_run_with_cleanup_kills_subprocess_on_exception(tmp_path: Path) -> None:
    """Pre-M3 gate (A): ensure subprocess gets terminated cleanly when the
    caller raises mid-wait. Uses a fake runner that sleeps 30s — if the
    cleanup path is broken, this test hangs for 30s+; healthy path
    terminates within 2-5s via SIGTERM to the process group."""
    import os
    import threading
    import time

    from sculptor.adapters.mjlab import _run_with_cleanup

    fake_runner = tmp_path / "sleep.py"
    fake_runner.write_text("import time; time.sleep(30)\n")
    cmd = [sys.executable, str(fake_runner)]

    start = time.monotonic()
    captured: dict = {}

    def _run() -> None:
        try:
            _run_with_cleanup(cmd, env=dict(os.environ), timeout=1.0)
        except subprocess.TimeoutExpired as e:
            captured["exc"] = e

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=10.0)
    elapsed = time.monotonic() - start

    assert not t.is_alive(), f"thread still running after {elapsed:.1f}s"
    assert "exc" in captured, "expected TimeoutExpired to propagate"
    # Cleanup path should have killed the child in < 10s (practical upper
    # bound). If cleanup were missing, we'd be blocked on the 30s sleep.
    assert elapsed < 10.0, f"cleanup took {elapsed:.1f}s; expected < 10s"


def test_mjlab_adapter_train_surfaces_subprocess_nonzero_exit(
    tmp_path: Path,
) -> None:
    """Pre-M3 gate (A): a runner failure (non-zero exit) should produce a
    RuntimeError with stdout+stderr preserved for debugging."""
    pytest.importorskip("mjlab")
    from sculptor.adapters import mjlab as mjlab_mod

    adapter = mjlab_mod.MjlabAdapter(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1",
        num_envs=128,
        device="cuda:0",
        max_iterations=1,
    )

    class _FakeProc:
        returncode = 7
        stdout = "fake stdout"
        stderr = "boom: synthetic failure from unit test"

    def fake_cleanup(cmd, env, timeout=None):
        return _FakeProc()

    with patch.object(mjlab_mod, "_run_with_cleanup", side_effect=fake_cleanup):
        with pytest.raises(RuntimeError) as exc_info:
            adapter.train(
                reward_module_path=None,
                output_dir=tmp_path / "out",
                steps=1,
                seed=1,
            )
    msg = str(exc_info.value)
    assert "exited 7" in msg
    assert "synthetic failure" in msg, f"stderr not preserved in error: {msg}"


import subprocess  # noqa: E402 — used by the helper tests above
import sys  # noqa: E402


# ── §7.1: trajectory-capture helpers (CPU-only; exercises the hot path ────
# without requiring mjlab or a GPU) ───────────────────────────────────────

def test_record_components_appends_mean_per_step() -> None:
    """`_record_components` must append one float per component per call,
    computed as the tensor's mean across envs."""
    import torch
    from sculptor.adapters._mjlab_runner import _record_components

    sink: dict[str, list[float]] = {}
    c1 = {"alive": torch.tensor([1.0, 1.0, 1.0]), "upright": torch.tensor([0.4, 0.6, 0.5])}
    i1 = {
        "episode_length": torch.tensor([10.0, 20.0, 30.0]),
        "terminated": torch.tensor([0.0, 0.0, 1.0]),
        "time_outs": torch.tensor([0.0, 1.0, 0.0]),
    }
    _record_components(sink, c1, i1)
    c2 = {"alive": torch.tensor([0.5, 0.5, 0.5]), "upright": torch.tensor([0.8, 0.9, 1.0])}
    i2 = {"episode_length": torch.tensor([15.0]), "terminated": torch.tensor([0.0]),
          "time_outs": torch.tensor([0.0])}
    _record_components(sink, c2, i2)

    assert sink["alive"] == pytest.approx([1.0, 0.5])
    assert sink["upright"] == pytest.approx([0.5, 0.9])
    assert sink["__episode_length"] == pytest.approx([20.0, 15.0])
    assert sink["__terminated"] == pytest.approx([1.0 / 3.0, 0.0])
    assert sink["__time_outs"] == pytest.approx([1.0 / 3.0, 0.0])


def test_record_components_noop_when_sink_none() -> None:
    from sculptor.adapters._mjlab_runner import _record_components

    # No exception, no side effects: the training path must pay zero cost
    # when capture is disabled (non-injected runs, GPU smoke tests).
    _record_components(None, {}, {})


def test_record_components_skips_non_tensor_values() -> None:
    """Mixed-type components dicts (e.g. a user writes `{"alive": 1.0}`
    instead of returning tensors) must not crash the sink — just skip."""
    import torch
    from sculptor.adapters._mjlab_runner import _record_components

    sink: dict[str, list[float]] = {}
    components = {
        "valid": torch.tensor([2.0, 4.0]),
        "bare_float": 3.14,  # not a tensor — must be skipped
        "empty": torch.tensor([]),  # empty mean is NaN — must be skipped
    }
    _record_components(sink, components, {})
    assert sink["valid"] == pytest.approx([3.0])
    # Non-tensor values produce no key.
    assert "bare_float" not in sink
    # Empty tensors produce NaN mean → skipped (ensures float cast
    # doesn't poison the window).
    import math
    for vals in sink.values():
        for v in vals:
            assert math.isfinite(v), f"non-finite snuck into sink: {v}"


def test_snapshots_to_trajectory_pivots_per_component() -> None:
    """`_snapshots_to_trajectory` pivots `list[dict[name, val]]` into
    `dict[name, list[val]]` matching Eureka Appendix F's format."""
    from sculptor.adapters._mjlab_runner import _snapshots_to_trajectory

    snaps = [
        {"alive": 1.0, "upright": 0.5, "__episode_length": 10.0},
        {"alive": 1.2, "upright": 0.6, "__episode_length": 15.0},
        {"alive": 1.1, "upright": 0.55, "__episode_length": 20.0},
    ]
    traj = _snapshots_to_trajectory(snaps)
    assert traj["alive"] == pytest.approx([1.0, 1.2, 1.1])
    assert traj["upright"] == pytest.approx([0.5, 0.6, 0.55])
    assert traj["__episode_length"] == pytest.approx([10.0, 15.0, 20.0])


def test_snapshots_to_trajectory_empty_input_returns_empty_dict() -> None:
    from sculptor.adapters._mjlab_runner import _snapshots_to_trajectory
    assert _snapshots_to_trajectory([]) == {}


def test_snapshots_to_trajectory_fills_missing_keys_with_last_seen() -> None:
    """A component that appears late (e.g. added mid-training) gets its
    first value at its debut window and fills forward after — same shape
    as the keys that were present from window 0."""
    from sculptor.adapters._mjlab_runner import _snapshots_to_trajectory

    snaps = [
        {"alive": 1.0},
        {"alive": 1.2, "upright": 0.5},  # upright debuts here
        {"alive": 1.1, "upright": 0.6},
    ]
    traj = _snapshots_to_trajectory(snaps)
    assert traj["alive"] == pytest.approx([1.0, 1.2, 1.1])
    # upright appears in 2 windows → 2-long series (post-debut only).
    assert traj["upright"] == pytest.approx([0.5, 0.6])


# ── §Ship 46: per-foot kick channels in the runtime info dict ─────────────
# CPU-only — fakes the mjlab sensor/entity API so the hot path is exercised
# without a GPU or the mjlab package.

def test_episode_relative_base_height_is_per_env_and_reset_safe() -> None:
    pytest.importorskip("torch")
    import torch
    from sculptor.adapters._mjlab_runner import _episode_relative_base_height

    anchor = None
    delta, anchor = _episode_relative_base_height(
        torch.tensor([0.74, 1.10]), torch.tensor([1.0, 1.0]), anchor)
    assert torch.allclose(delta, torch.zeros(2))
    delta, anchor = _episode_relative_base_height(
        torch.tensor([0.82, 1.06]), torch.tensor([2.0, 2.0]), anchor)
    assert torch.allclose(delta, torch.tensor([0.08, -0.04]), atol=1e-6)

    # Only env 1 reset; env 0 retains its own original episode anchor.
    anchor[1] = float("nan")
    delta, anchor = _episode_relative_base_height(
        torch.tensor([0.85, 0.66]), torch.tensor([3.0, 1.0]), anchor)
    assert torch.allclose(delta, torch.tensor([0.11, 0.0]), atol=1e-6)


def _make_term():
    """Build a SculptorRewardTerm and bypass __init__ (which needs a real
    env + reward module). _foot_info / _resolve_foot_handles only touch
    `self._foot_cache`, so __new__ is sufficient."""
    from sculptor.adapters._mjlab_runner import _build_sculptor_term_class

    TermClass = _build_sculptor_term_class(("qpos", "qvel"))
    return TermClass.__new__(TermClass)


def test_foot_info_populates_biped_channels() -> None:
    """A biped env (left_foot/right_foot sites + the two named sensors)
    yields real per-foot contact / swing-speed / height + base speed."""
    pytest.importorskip("torch")
    import torch

    N = 3
    term = _make_term()

    class _Data:
        # left foot velocity (3,4,0)->|v|=5; right (0,0,0)->0
        site_lin_vel_w = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]] * N)
        root_link_lin_vel_b = torch.tensor([[3.0, 4.0, 9.0]] * N)  # xy-norm=5

    class _Robot:
        site_names = ("left_foot", "right_foot")
        data = _Data()

    def _sensor(**kw):
        return type("S", (), {"data": type("D", (), kw)})()

    class _Scene:
        _d = {
            "feet_ground_contact": _sensor(found=torch.tensor([[1.0, 0.0]] * N)),
            "foot_height_scan": _sensor(heights=torch.tensor([[0.05, 0.20]] * N)),
        }

        def __getitem__(self, k):
            return self._d[k]

    class _Env:
        num_envs = N
        device = torch.device("cpu")
        scene = _Scene()

    out = term._foot_info(_Env(), _Robot(), torch.float32)
    assert torch.allclose(out["left_foot_contact"], torch.ones(N))
    assert torch.allclose(out["right_foot_contact"], torch.zeros(N))
    assert torch.allclose(out["left_foot_swing_speed"], torch.full((N,), 5.0))
    assert torch.allclose(out["right_foot_swing_speed"], torch.zeros(N))
    assert torch.allclose(out["left_foot_height"], torch.full((N,), 0.05))
    assert torch.allclose(out["right_foot_height"], torch.full((N,), 0.20))
    assert torch.allclose(out["base_horizontal_speed"], torch.full((N,), 5.0))


def test_foot_info_zeros_for_non_biped_but_keeps_base_speed() -> None:
    """A quadruped (no left_foot/right_foot sites) + missing foot sensors
    must degrade to zeros on every per-foot channel — no crash — while
    base_horizontal_speed still computes from the root velocity."""
    pytest.importorskip("torch")
    import torch

    N = 2
    term = _make_term()

    class _Data:
        root_link_lin_vel_b = torch.tensor([[6.0, 8.0, 1.0]] * N)  # xy-norm=10

    class _Robot:
        site_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        data = _Data()

    class _Scene:
        def __getitem__(self, k):
            raise KeyError(k)  # quadruped task lacks the named foot sensors

    class _Env:
        num_envs = N
        device = torch.device("cpu")
        scene = _Scene()

    out = term._foot_info(_Env(), _Robot(), torch.float32)
    for k in (
        "left_foot_contact", "right_foot_contact",
        "left_foot_swing_speed", "right_foot_swing_speed",
        "left_foot_height", "right_foot_height",
    ):
        assert torch.allclose(out[k], torch.zeros(N)), k
    assert torch.allclose(out["base_horizontal_speed"], torch.full((N,), 10.0))


def test_foot_info_keys_match_contract_extra() -> None:
    """The runtime info dict must emit exactly the keys the contract
    advertises for G1 — guards against runner/contract drift."""
    pytest.importorskip("torch")
    import torch
    from sculptor.adapters.mjlab import _G1_INFO_EXTRA

    term = _make_term()

    class _Data:
        root_link_lin_vel_b = None

    class _Robot:
        site_names = ()
        data = _Data()

    class _Scene:
        def __getitem__(self, k):
            raise KeyError(k)

    class _Env:
        num_envs = 1
        device = torch.device("cpu")
        scene = _Scene()

    out = term._foot_info(_Env(), _Robot(), torch.float32)
    assert set(out.keys()) == set(_G1_INFO_EXTRA)


# ── §Ship-7: rollout video fps math ──────────────────────────────────────
def test_compute_playback_fps_real_time_default() -> None:
    """50 Hz sim, render_every=1, playback_speed=1.0 → 50 fps playback.
    Video plays real-time."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=1.0,
    ) == pytest.approx(50.0)


def test_compute_playback_fps_render_every_preserves_real_time() -> None:
    """When render_every > 1 (frames decimated), fps must drop
    proportionally so total video duration still equals sim duration."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    # 50 Hz × render_every=4 → 4 sim steps per frame → 12.5 fps keeps
    # playback real-time.
    assert _compute_playback_fps(
        step_dt=0.02, render_every=4, playback_speed=1.0,
    ) == pytest.approx(12.5)


def test_compute_playback_fps_speed_multiplier() -> None:
    """playback_speed=2.0 → video plays 2× real time (fps doubled)."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=2.0,
    ) == pytest.approx(100.0)
    # 0.5× = slow-mo; video plays half-speed → fps halved.
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=0.5,
    ) == pytest.approx(25.0)


def test_compute_playback_fps_clamps_to_valid_range() -> None:
    """ffmpeg rejects fps outside [1, 240]. Helper must clamp."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    # Unrealistically fast playback → clamped to 240.
    assert _compute_playback_fps(
        step_dt=0.0001, render_every=1, playback_speed=1.0,
    ) == pytest.approx(240.0)
    # Unrealistically slow (e.g. huge render_every) → clamped to 1.
    assert _compute_playback_fps(
        step_dt=1.0, render_every=1000, playback_speed=0.1,
    ) == pytest.approx(1.0)


def test_compute_playback_fps_cli_override_wins() -> None:
    """Non-zero cli_fps replaces the derived value."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    # Derived would be 50; override wins.
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=1.0, cli_fps=24.0,
    ) == pytest.approx(24.0)
    # But clamped: an override >240 snaps back.
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=1.0, cli_fps=500.0,
    ) == pytest.approx(240.0)


def test_compute_playback_fps_clamps_playback_speed() -> None:
    """Very-out-of-range speeds get clamped before the fps math."""
    from sculptor.adapters._mjlab_runner import _compute_playback_fps
    # 100x is above the 10x cap, so fps for step_dt=0.02 lands at
    # min(10.0 / 0.02, 240) = 240.
    assert _compute_playback_fps(
        step_dt=0.02, render_every=1, playback_speed=100.0,
    ) == pytest.approx(240.0)


def test_first_episode_freeze_removes_auto_reset_teleport() -> None:
    """A done-step state is the next episode and must become absorbing padding."""
    import numpy as np

    from sculptor.adapters._mjlab_runner import (
        _freeze_invalid_first_episode_steps,
    )

    root = np.asarray([
        [[0.0, 0.0], [10.0, 0.0]],
        [[1.0, 0.0], [11.0, 0.0]],
        [[0.0, 0.0], [12.0, 0.0]],  # env 0 auto-reset to spawn
        [[0.2, 0.0], [10.0, 0.0]],  # both are now later attempts
    ])
    valid = np.asarray([
        [True, True],
        [True, True],
        [False, True],
        [False, False],
    ])

    frozen = _freeze_invalid_first_episode_steps(root, valid)

    np.testing.assert_array_equal(frozen[:, 0, 0], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(frozen[:, 1, 0], [10.0, 11.0, 12.0, 12.0])
    assert not np.shares_memory(frozen, root)


def test_first_episode_freeze_fails_soft_on_incompatible_mask() -> None:
    import numpy as np

    from sculptor.adapters._mjlab_runner import (
        _freeze_invalid_first_episode_steps,
    )

    values = np.arange(6).reshape(3, 2)
    result = _freeze_invalid_first_episode_steps(
        values, np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(result, values)
