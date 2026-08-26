"""Unit tests for MjlabAdapter + stub adapters.

Covers the mocked / import-only surface. The real-GPU smoke test lives
in tests/test_mjlab_gpu.py behind `@pytest.mark.gpu`.
"""

from __future__ import annotations

import copy
import json
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


def test_runner_reward_receipt_tracks_immutable_selected_module(
    tmp_path: Path,
) -> None:
    """Strict lineage publishes vN bytes even though the loader is current.py."""
    import hashlib

    from sculptor.adapters._mjlab_runner import (
        _capture_runner_reward_artifact,
        _load_reward_module,
        _validate_loaded_reward_artifact,
    )
    from sculptor.edit import _write_current_reexport

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    selected = rewards / "v3.py"
    selected.write_text(
        "REWARD_SPEC = {'reference_clock': {'schema': 1}}\n"
        "def compute_reward(s, a, n, i): return 0.0, {}\n"
        "def compute_reward_batched(s, a, n, i): return a[:, 0], {}\n"
        "def reference_clock_batched(step_count, device): return step_count\n"
        "def reference_target_index_batched(step_count, device): return step_count\n",
        encoding="utf-8",
    )
    _write_current_reexport(rewards, selected)

    artifact = _capture_runner_reward_artifact(
        rewards / "current.py", label="training reward module"
    )
    module = _load_reward_module(str(rewards / "current.py"))
    _validate_loaded_reward_artifact(module, artifact)

    selected_sha256 = hashlib.sha256(selected.read_bytes()).hexdigest()
    current_sha256 = hashlib.sha256(
        (rewards / "current.py").read_bytes()
    ).hexdigest()
    assert artifact["selection_kind"] == "selector"
    assert artifact["selected"]["sha256"] == selected_sha256
    assert artifact["loader"]["sha256"] == current_sha256
    assert selected_sha256 != current_sha256


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


def test_authored_task_horizon_is_reasserted_after_env_overlay() -> None:
    from sculptor.adapters._mjlab_runner import (
        _reassert_authored_task_termination,
    )

    env_cfg = SimpleNamespace(episode_length_s=12.0)
    world_bundle = SimpleNamespace(manifest=SimpleNamespace(task_shared={
        "termination": {"episode_length_s": 24.0},
    }))
    _reassert_authored_task_termination(env_cfg, world_bundle)
    assert env_cfg.episode_length_s == 24.0

    # Registered non-authored tasks keep their own EnvSpec authority.
    env_cfg.episode_length_s = 12.0
    _reassert_authored_task_termination(env_cfg, None)
    assert env_cfg.episode_length_s == 12.0


def test_event_jump_linear_tracking_preserves_xy_without_penalizing_launch(
) -> None:
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _build_event_jump_masked_reward_term,
        _install_authored_event_jump_linear_tracking,
        _install_authored_event_jump_realism_firewall,
    )

    command_term = SimpleNamespace(
        event_sequence_id="route_jump_hold",
        event_phase=torch.tensor([0, 1, 2, 1]),
        event_sequence_violation=torch.tensor([
            False, False, False, True,
        ]),
    )
    velocity_command = torch.zeros((4, 3))

    class CommandManager:
        active_terms = ("route",)

        @staticmethod
        def get_term(name):
            assert name == "route"
            return command_term

        @staticmethod
        def get_command(name):
            assert name == "twist"
            return velocity_command

    actual_velocity = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        command_manager=CommandManager(),
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(
                root_link_lin_vel_b=actual_velocity,
            )),
        },
    )

    def native_tracker(env, *, std, command_name):
        command = env.command_manager.get_command(command_name)
        actual = env.scene["robot"].data.root_link_lin_vel_b
        error = torch.sum(
            torch.square(command[:, :2] - actual[:, :2]), dim=1)
        error += torch.square(actual[:, 2])
        return torch.exp(-error / std**2)

    reward_term = SimpleNamespace(func=native_tracker, weight=2.0)
    rewards = {"track_linear_velocity": reward_term}
    bundle = SimpleNamespace(
        manifest=SimpleNamespace(task_shared={
            "goal": {"type": "waypoint_sequence"},
            "event_sequence": {"phases": [
                {"id": "route"}, {"id": "jump"}, {"id": "hold"},
            ]},
        }),
        runtime_adjustments=(
            "command:velocity→goal-conditioned waypoint traversal",
        ),
    )

    assert _install_authored_event_jump_linear_tracking(rewards, bundle)
    reward = reward_term.func(env, std=0.5, command_name="twist")
    native = native_tracker(env, std=0.5, command_name="twist")

    # ROUTE and HOLD remain exactly native. Valid JUMP keeps only the XY
    # tracking error, so vertical takeoff is no longer punished. A violated
    # JUMP sequence fails closed to the native reward.
    assert reward[0].item() == pytest.approx(native[0].item())
    assert reward[1].item() == pytest.approx(
        torch.exp(torch.tensor(-0.01 / 0.25)).item())
    assert reward[2].item() == pytest.approx(native[2].item())
    assert reward[3].item() == pytest.approx(native[3].item())

    # Typed event truth is authoritative; malformed shapes abort instead of
    # silently reinstating a contradictory grounded-motion reward.
    command_term.event_phase = torch.tensor([0, 1])
    with pytest.raises(RuntimeError, match="malformed phase"):
        reward_term.func(env, std=0.5, command_name="twist")
    command_term.event_phase = torch.tensor([0, 1, 2, 1])

    # Re-installation is idempotent and a non-event world is untouched.
    installed = reward_term.func
    assert _install_authored_event_jump_linear_tracking(rewards, bundle)
    assert reward_term.func is installed
    bundle.manifest.task_shared.pop("event_sequence")
    plain = SimpleNamespace(func=native_tracker, weight=2.0)
    assert not _install_authored_event_jump_linear_tracking(
        {"track_linear_velocity": plain}, bundle)
    assert plain.func is native_tracker

    # The broader firewall masks only grounded-gait priors. Contact/safety
    # terms and landing bookkeeping remain native.
    def ones(env):
        return torch.ones(env.num_envs)

    split_rewards = {
        name: SimpleNamespace(func=ones, weight=1.0)
        for name in (
            "track_linear_velocity",
            "track_angular_velocity",
            "upright",
            "pose",
            "foot_clearance",
            "foot_swing_height",
            "body_ang_vel",
            "angular_momentum",
            "foot_slip",
            "soft_landing",
            "dof_pos_limits",
            "action_rate_l2",
            "self_collisions",
        )
    }
    split_rewards["track_linear_velocity"].func = native_tracker
    masked = _install_authored_event_jump_realism_firewall(
        split_rewards, SimpleNamespace(
            manifest=SimpleNamespace(task_shared={
                "goal": {"type": "waypoint_sequence"},
                "event_sequence": {"phases": [
                    {"id": "route"}, {"id": "jump"}, {"id": "hold"},
                ]},
            }),
            runtime_adjustments=(
                "command:velocity→goal-conditioned waypoint traversal",
            ),
        ),
    )
    assert masked == (
        "angular_momentum",
        "body_ang_vel",
        "foot_clearance",
        "foot_swing_height",
        "pose",
    )
    for name in masked:
        assert getattr(
            split_rewards[name].func,
            "_sculptor_event_jump_masked",
            False,
        )
    for name in (
        "track_angular_velocity", "upright", "foot_slip", "soft_landing",
        "dof_pos_limits", "action_rate_l2", "self_collisions",
    ):
        assert split_rewards[name].func is ones

    # Class-backed terms keep native call/reset behavior; masking happens only
    # after the stateful callable has updated its internal bookkeeping.
    class StatefulTerm:
        def __init__(self, cfg, env):
            self.calls = 0
            self.reset_ids = None

        def __call__(self, env):
            self.calls += 1
            return torch.ones(env.num_envs)

        def reset(self, env_ids):
            self.reset_ids = env_ids

    WrappedStateful = _build_event_jump_masked_reward_term(StatefulTerm)
    stateful = WrappedStateful(None, env)
    stateful_reward = stateful(env)
    assert stateful.calls == 1
    assert stateful_reward.tolist() == [1.0, 0.0, 1.0, 1.0]
    stateful.reset(torch.tensor([1]))
    assert stateful.reset_ids.tolist() == [1]


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


def test_event_terminal_stillness_is_hold_phase_only() -> None:
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _authored_terminal_hold_s,
        _authored_terminal_stillness_reward,
        _authored_terminal_stillness_state,
    )

    command = SimpleNamespace(
        event_sequence_id="route_jump_hold",
        event_phase=torch.tensor([1, 2]),
        event_sequence_violation=torch.tensor([False, False]),
        # This deliberately lies for the JUMP lane; event phase must win.
        is_standing_env=torch.tensor([True, True]),
    )

    class Manager:
        active_terms = ("route",)

        @staticmethod
        def get_term(_name):
            return command

    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.zeros((2, 3)),
        root_link_ang_vel_b=torch.zeros((2, 3)),
        joint_vel=torch.zeros((2, 4)),
    ))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=Manager(),
        scene={"robot": robot},
    )
    standing, _score, _speed, _quiet = _authored_terminal_stillness_state(
        env, lin_std=0.12, ang_std=0.15, joint_std=0.12,
    )
    assert standing.tolist() == [False, True]

    command.event_sequence_violation[1] = True
    standing, _score, _speed, _quiet = _authored_terminal_stillness_state(
        env, lin_std=0.12, ang_std=0.15, joint_std=0.12,
    )
    assert standing.tolist() == [False, False]
    reward = _authored_terminal_stillness_reward(
        env, lin_std=0.12, ang_std=0.15, joint_std=0.12,
    )
    assert reward.tolist() == [0.0, 0.0]

    del command.event_sequence_violation
    with pytest.raises(RuntimeError, match="sequence-violation truth"):
        _authored_terminal_stillness_state(
            env, lin_std=0.12, ang_std=0.15, joint_std=0.12,
        )

    bundle = SimpleNamespace(
        runtime_adjustments=(
            "installed goal-conditioned waypoint traversal command",
        ),
        manifest=SimpleNamespace(task_shared={
        "goal": {
            "type": "waypoint_sequence",
            "success": {"hold_s": 0.0},
        },
        "event_sequence": {"phases": [
            {"id": "route"},
            {"id": "jump"},
            {"id": "hold", "minimum_hold_s": 2.0},
        ]},
    }))
    assert _authored_terminal_hold_s(bundle) == 2.0


def test_event_observation_extension_supports_real_batched_normalizers() -> None:
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _zero_extend_observation_state_dict,
    )

    source = {
        "mlp.0.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "obs_normalizer._mean": torch.full((1, 4), 3.0),
        "obs_normalizer._var": torch.full((1, 4), 4.0),
        "obs_normalizer._std": torch.full((1, 4), 2.0),
        "obs_normalizer.count": torch.tensor(17.0),
        "mlp.0.bias": torch.zeros(2),
    }
    target = {
        **source,
        "mlp.0.weight": torch.zeros((2, 7)),
        "obs_normalizer._mean": torch.zeros((1, 7)),
        "obs_normalizer._var": torch.zeros((1, 7)),
        "obs_normalizer._std": torch.zeros((1, 7)),
    }
    adapted, changed = _zero_extend_observation_state_dict(
        source, target, extension_width=3, role="actor",
    )

    torch.testing.assert_close(
        adapted["mlp.0.weight"][:, :4], source["mlp.0.weight"])
    assert torch.all(adapted["mlp.0.weight"][:, 4:] == 0.0)
    assert torch.all(adapted["obs_normalizer._mean"][:, 4:] == 0.0)
    assert torch.all(adapted["obs_normalizer._var"][:, 4:] == 1.0)
    assert torch.all(adapted["obs_normalizer._std"][:, 4:] == 1.0)
    assert adapted["obs_normalizer.count"].shape == torch.Size([])
    assert set(changed) == {
        "mlp.0.weight",
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    }

    invalid = dict(target)
    invalid["mlp.0.bias"] = torch.zeros(3)
    with pytest.raises(RuntimeError, match="cannot satisfy"):
        _zero_extend_observation_state_dict(
            source, invalid, extension_width=3, role="actor",
        )


def test_ordered_observation_migration_preserves_trailing_reference_phase() -> None:
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _zero_extend_observation_state_dict,
    )

    source_weight = torch.arange(200, dtype=torch.float32).reshape(2, 100)
    source = {
        "mlp.0.weight": source_weight,
        "obs_normalizer._mean": torch.arange(100, dtype=torch.float32).reshape(
            1, 100
        ),
        "obs_normalizer._var": torch.full((1, 100), 4.0),
        "obs_normalizer._std": torch.full((1, 100), 2.0),
        "obs_normalizer.count": torch.tensor(17.0),
        "mlp.0.bias": torch.zeros(2),
    }
    target = {
        **source,
        "mlp.0.weight": torch.zeros((2, 169)),
        "obs_normalizer._mean": torch.zeros((1, 169)),
        "obs_normalizer._var": torch.zeros((1, 169)),
        "obs_normalizer._std": torch.zeros((1, 169)),
    }
    migration = {
        "role": "actor",
        "source_width": 100,
        "target_width": 169,
        "extension_width": 69,
        "preserved_segments": [
            {
                "term_name": "proprioception",
                "source_offset": 0,
                "target_offset": 0,
                "width": 99,
            },
            {
                "term_name": "reference_phase",
                "source_offset": 99,
                "target_offset": 168,
                "width": 1,
            },
        ],
        "inserted_segments": [{
            "term_name": "task_features",
            "target_offset": 99,
            "width": 69,
        }],
    }

    adapted, changed = _zero_extend_observation_state_dict(
        source,
        target,
        extension_width=69,
        role="actor",
        column_migration=migration,
    )

    torch.testing.assert_close(
        adapted["mlp.0.weight"][:, :99], source_weight[:, :99]
    )
    assert torch.all(adapted["mlp.0.weight"][:, 99:168] == 0.0)
    torch.testing.assert_close(
        adapted["mlp.0.weight"][:, 168], source_weight[:, 99]
    )
    torch.testing.assert_close(
        adapted["obs_normalizer._mean"][:, 168],
        source["obs_normalizer._mean"][:, 99],
    )
    assert torch.all(adapted["obs_normalizer._mean"][:, 99:168] == 0.0)
    assert torch.all(adapted["obs_normalizer._var"][:, 99:168] == 1.0)
    assert torch.all(adapted["obs_normalizer._std"][:, 99:168] == 1.0)
    assert "mlp.0.weight" in changed

    malformed = {
        **migration,
        "preserved_segments": [
            migration["preserved_segments"][0],
            {
                **migration["preserved_segments"][1],
                "target_offset": 167,
            },
        ],
    }
    with pytest.raises(RuntimeError, match="overlap or leave unmapped"):
        _zero_extend_observation_state_dict(
            source,
            target,
            extension_width=69,
            role="actor",
            column_migration=malformed,
        )


def test_prepare_warm_start_applies_contract_role_mapping(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _prepare_event_observation_warm_start,
    )

    source_state = {
        "mlp.0.weight": torch.arange(
            200, dtype=torch.float32
        ).reshape(2, 100),
        "mlp.0.bias": torch.zeros(2),
    }
    target_state = {
        "mlp.0.weight": torch.zeros((2, 169)),
        "mlp.0.bias": torch.zeros(2),
    }
    checkpoint = tmp_path / "tracker.pt"
    torch.save({"actor_state_dict": source_state}, checkpoint)
    runner = SimpleNamespace(alg=SimpleNamespace(
        actor=SimpleNamespace(state_dict=lambda: target_state),
        critic=None,
    ))
    actor_migration = {
        "role": "actor",
        "source_width": 100,
        "target_width": 169,
        "extension_width": 69,
        "preserved_segments": [
            {
                "term_name": "proprioception",
                "source_offset": 0,
                "target_offset": 0,
                "width": 99,
            },
            {
                "term_name": "reference_phase",
                "source_offset": 99,
                "target_offset": 168,
                "width": 1,
            },
        ],
        "inserted_segments": [{
            "term_name": "task_features",
            "target_offset": 99,
            "width": 69,
        }],
    }

    adapted_path, receipt = _prepare_event_observation_warm_start(
        runner,
        checkpoint,
        output_dir=tmp_path,
        extension_width=69,
        load_role="actor_only",
        observation_terms=("task_features",),
        role_migrations={"actor": actor_migration},
    )

    adapted = torch.load(adapted_path, weights_only=False, map_location="cpu")
    weight = adapted["actor_state_dict"]["mlp.0.weight"]
    torch.testing.assert_close(weight[:, :99], source_state["mlp.0.weight"][:, :99])
    assert torch.all(weight[:, 99:168] == 0.0)
    torch.testing.assert_close(weight[:, 168], source_state["mlp.0.weight"][:, 99])
    assert receipt["adapted"] is True
    assert receipt["role_migrations"] == {"actor": actor_migration}

    actor_critic_checkpoint = tmp_path / "actor-critic.pt"
    torch.save(
        {
            "actor_state_dict": source_state,
            "critic_state_dict": source_state,
        },
        actor_critic_checkpoint,
    )
    actor_critic_runner = SimpleNamespace(alg=SimpleNamespace(
        actor=SimpleNamespace(state_dict=lambda: target_state),
        critic=SimpleNamespace(state_dict=lambda: target_state),
    ))
    with pytest.raises(RuntimeError, match="no critic mapping"):
        _prepare_event_observation_warm_start(
            actor_critic_runner,
            actor_critic_checkpoint,
            output_dir=tmp_path,
            extension_width=69,
            load_role="actor_critic",
            observation_terms=("task_features",),
            role_migrations={"actor": actor_migration},
        )


def test_reference_clock_is_installed_and_executed_for_actor_and_critic() -> None:
    """The admitted reference clock must reach both live policy interfaces."""
    torch = pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _install_reference_clock_observation,
    )
    from sculptor.reference_clock import build_reference_clock

    clock = build_reference_clock(
        clip_id="hop",
        robot="g1",
        target_sha256="a" * 64,
        phase_mode="hold",
        phase_duration_s=2.0,
        n_phase_targets=32,
    )

    def reference_clock_batched(obs, like):
        phase = obs["episode_length"] * obs["step_dt"] / 2.0
        return phase.clamp(min=0.0, max=1.0).unsqueeze(-1).to(like)

    reward_module = SimpleNamespace(
        REWARD_SPEC={
            "reference_tracking": True,
            "reference_clock": clock,
        },
        reference_clock_batched=reference_clock_batched,
        reference_target_index_batched=lambda _obs, like: like.to(
            dtype=torch.long
        ),
    )
    actor = SimpleNamespace(terms={})
    critic = SimpleNamespace(terms={})
    env_cfg = SimpleNamespace(observations={
        "actor": actor,
        "critic": critic,
        # Alias the actor group as some MJLab tasks do. Installation must not
        # double-add the same mutable term mapping.
        "policy": actor,
    })

    installed = _install_reference_clock_observation(env_cfg, reward_module)

    assert installed == clock
    assert set(actor.terms) == {"reference_phase"}
    assert set(critic.terms) == {"reference_phase"}
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0, 25, 100]),
        step_dt=0.02,
        num_envs=3,
        device="cpu",
    )
    for group in (actor, critic):
        cfg = group.terms["reference_phase"]
        term = cfg.func(cfg, env)
        torch.testing.assert_close(
            term(env), torch.tensor([[0.0], [0.25], [1.0]])
        )


@pytest.mark.parametrize("present_role", ["actor", "critic"])
def test_reference_clock_rejects_a_missing_policy_role(
    present_role: str,
) -> None:
    pytest.importorskip("torch")
    from sculptor.adapters._mjlab_runner import (
        _install_reference_clock_observation,
    )
    from sculptor.reference_clock import build_reference_clock

    reward_module = SimpleNamespace(
        REWARD_SPEC={
            "reference_tracking": True,
            "reference_clock": build_reference_clock(
                clip_id="hop",
                robot="g1",
                target_sha256="a" * 64,
                phase_mode="hold",
                phase_duration_s=2.0,
                n_phase_targets=32,
            ),
        },
        reference_clock_batched=lambda _info, like: like[:, None],
        reference_target_index_batched=lambda _info, like: like.long(),
    )
    env_cfg = SimpleNamespace(observations={
        present_role: SimpleNamespace(terms={}),
    })

    with pytest.raises(RuntimeError, match="actor and critic"):
        _install_reference_clock_observation(env_cfg, reward_module)


def test_rollout_policy_contract_sidecar_must_match_exact_interface(
    tmp_path: Path,
) -> None:
    import hashlib

    from sculptor.adapters._mjlab_runner import (
        _validate_rollout_checkpoint_policy_contract,
        _write_local_checkpoint_policy_contract,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"server-owned checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = {
        "schema": 4,
        "observations": {
            "ordered_terms": [{"name": "phase", "shape": [1]}],
            "critic_ordered_terms": [{"name": "phase", "shape": [1]}],
        },
        "reference_clock": {"reference_target_sha256": "a" * 64},
        "world": {"selection_sha256": "b" * 64},
    }
    _write_local_checkpoint_policy_contract(checkpoint, contract)

    receipt = _validate_rollout_checkpoint_policy_contract(
        checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        expected_contract=contract,
    )
    assert receipt["checkpoint_sha256"] == checkpoint_sha256

    changed_critic = copy.deepcopy(contract)
    changed_critic["observations"]["critic_ordered_terms"][0]["shape"] = [2]
    with pytest.raises(RuntimeError, match="policy contract differs"):
        _validate_rollout_checkpoint_policy_contract(
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_contract=changed_critic,
        )

    changed_world = copy.deepcopy(contract)
    changed_world["world"]["selection_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="policy contract differs"):
        _validate_rollout_checkpoint_policy_contract(
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_contract=changed_world,
        )

    (tmp_path / "checkpoint.pt.policy_contract.json").unlink()
    with pytest.raises(RuntimeError, match="requires the exact local"):
        _validate_rollout_checkpoint_policy_contract(
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_contract=contract,
        )


def test_warm_start_loaded_receipt_names_exact_loaded_bytes(tmp_path: Path) -> None:
    from sculptor.adapters._mjlab_runner import _warm_start_loaded_receipt

    source = tmp_path / "source.pt"
    derived = tmp_path / "warm_start_event_observation.pt"
    receipt = _warm_start_loaded_receipt(
        requested_source=source,
        requested_sha256="a" * 64,
        loaded_checkpoint=derived,
        loaded_sha256="b" * 64,
        load_cfg={
            "actor": True, "critic": True, "optimizer": False,
        },
        effective_policy_contract_sha256="c" * 64,
        extension_receipt={
            "adapted": True,
            "source_policy_contract_sha256": "d" * 64,
            "admitted_policy_contract_migration": {
                "type": "zero_initialized_event_phase_observation",
            },
        },
    )

    assert receipt["requested_source"] == str(source)
    assert receipt["requested_source_sha256"] == "a" * 64
    assert receipt["loaded_checkpoint"] == str(derived)
    assert receipt["loaded_checkpoint_sha256"] == "b" * 64
    assert receipt["policy_contract_migration"] \
        == "zero_initialized_event_phase_observation"
    assert receipt["derived_from"]["source_sha256"] == "a" * 64
    assert receipt["source_policy_contract_sha256"] == "d" * 64
    assert receipt["admitted_policy_contract_migration"] == {
        "type": "zero_initialized_event_phase_observation",
    }
    assert receipt["load_cfg_keys"] == ["actor", "critic"]


def test_generic_warm_start_checkpoint_pin_is_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters._mjlab_runner import (
        _verify_warm_start_checkpoint_sha256,
    )

    digest = "a" * 64
    monkeypatch.setenv("SCULPTOR_WARM_START_CHECKPOINT_SHA256", digest)
    monkeypatch.delenv(
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256", raising=False,
    )

    assert _verify_warm_start_checkpoint_sha256(digest) == digest
    with pytest.raises(RuntimeError, match="warm-start launch pin"):
        _verify_warm_start_checkpoint_sha256("b" * 64)


def test_imported_skill_checkpoint_pin_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters._mjlab_runner import (
        _verify_warm_start_checkpoint_sha256,
    )

    digest = "c" * 64
    monkeypatch.delenv(
        "SCULPTOR_WARM_START_CHECKPOINT_SHA256", raising=False,
    )
    monkeypatch.setenv(
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256", digest,
    )

    assert _verify_warm_start_checkpoint_sha256(digest) == digest


def test_conflicting_or_noncanonical_warm_start_pins_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters._mjlab_runner import (
        _expected_warm_start_checkpoint_sha256,
    )

    monkeypatch.setenv(
        "SCULPTOR_WARM_START_CHECKPOINT_SHA256", "a" * 64,
    )
    monkeypatch.setenv(
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256", "b" * 64,
    )
    with pytest.raises(RuntimeError, match="digest pins disagree"):
        _expected_warm_start_checkpoint_sha256()

    monkeypatch.delenv(
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256", raising=False,
    )
    monkeypatch.setenv(
        "SCULPTOR_WARM_START_CHECKPOINT_SHA256", "A" * 64,
    )
    with pytest.raises(RuntimeError, match="canonical lowercase SHA-256"):
        _expected_warm_start_checkpoint_sha256()


def test_event_policy_contract_admits_exact_schema3_direct_load() -> None:
    from sculptor.adapters._mjlab_runner import (
        _event_policy_contract_admission_kind,
    )

    exact = {
        "type": "exact_policy_contract",
        "from_schema": 3,
        "to_schema": 3,
        "optimizer_resume": False,
    }
    effective = {
        "schema": 3,
        "event_observation": {
            "ordered_phase_ids": ["route", "jump", "hold"],
        },
    }
    assert _event_policy_contract_admission_kind(
        exact,
        effective,
        extension_width=3,
    ) == "exact_policy_contract"

    with pytest.raises(RuntimeError, match="unrecognized"):
        _event_policy_contract_admission_kind(
            {**exact, "optimizer_resume": True},
            effective,
            extension_width=3,
        )


def test_policy_contract_admits_pinned_ordered_observation_insertions() -> None:
    from sculptor.adapters._mjlab_runner import (
        _event_policy_contract_admission_kind,
    )

    def role_migration(role: str, source_width: int, target_width: int) -> dict:
        return {
            "role": role,
            "source_width": source_width,
            "target_width": target_width,
            "extension_width": target_width - source_width,
            "preserved_segments": [{
                "term_name": "source",
                "source_offset": 0,
                "target_offset": 0,
                "width": source_width,
            }],
            "inserted_segments": [{
                "term_name": "task",
                "target_offset": source_width,
                "width": target_width - source_width,
            }],
        }

    migration = {
        "type": "zero_initialized_ordered_observation_insertions",
        "from_schema": 4,
        "to_schema": 4,
        "roles": ["actor", "critic"],
        "extension_width": 69,
        "role_migrations": {
            "actor": role_migration("actor", 100, 169),
            "critic": role_migration("critic", 112, 181),
        },
        "optimizer_resume": False,
    }
    effective = {
        "schema": 4,
        "observations": {"shape": [169], "critic_shape": [181]},
    }

    assert _event_policy_contract_admission_kind(
        migration,
        effective,
        # Runtime-installed clock width is not the task-feature insertion
        # width; the immutable role mapping is the tensor authority.
        extension_width=1,
    ) == "zero_initialized_ordered_observation_insertions"

    malformed = copy.deepcopy(migration)
    malformed["role_migrations"]["actor"]["target_width"] = 168
    with pytest.raises(RuntimeError, match="invalid actor mapping"):
        _event_policy_contract_admission_kind(
            malformed,
            effective,
            extension_width=1,
        )

    actor_only = copy.deepcopy(migration)
    actor_only["roles"] = ["actor"]
    actor_only["role_migrations"].pop("critic")
    assert _event_policy_contract_admission_kind(
        actor_only,
        effective,
        extension_width=1,
    ) == "zero_initialized_ordered_observation_insertions"

    undeclared_critic = copy.deepcopy(actor_only)
    undeclared_critic["role_migrations"]["critic"] = role_migration(
        "critic", 112, 181,
    )
    with pytest.raises(RuntimeError, match="undeclared role"):
        _event_policy_contract_admission_kind(
            undeclared_critic,
            effective,
            extension_width=1,
        )


def test_runner_attests_exact_schema2_pins_without_observation_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sculptor.policy_contract as policy_contract
    from sculptor.adapters._mjlab_runner import (
        _attest_warm_start_policy_contract,
    )

    project = tmp_path / "project"
    selection = project / "env" / "selection_v1.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}\n", encoding="utf-8")
    contract = {
        "schema": 2,
        "identity": {"adapter_class": "MjlabAdapter", "task_id": "g1"},
    }
    contract_sha256 = policy_contract.contract_fingerprint(contract)
    checkpoint_sha256 = "c" * 64
    compatibility = {
        "type": "exact_policy_contract",
        "from_schema": 2,
        "to_schema": 2,
        "optimizer_resume": False,
    }
    receipt = {
        "schema": 1,
        "kind": "starting_skill",
        "source": {
            "checkpoint_sha256": checkpoint_sha256,
            "contract": contract,
            "contract_sha256": contract_sha256,
        },
        "target": {
            "contract": contract,
            "contract_sha256": contract_sha256,
        },
        "compatibility": compatibility,
    }
    pins = {
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON": json.dumps(
            receipt, sort_keys=True),
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON": json.dumps(
            contract, sort_keys=True),
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256": contract_sha256,
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256": contract_sha256,
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON": json.dumps(
            compatibility, sort_keys=True),
    }
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        policy_contract,
        "build_project_policy_contract",
        lambda *_args, **_kwargs: contract,
    )

    admitted = _attest_warm_start_policy_contract(
        world_selection=selection,
        extension_width=0,
        source_checkpoint_sha256=checkpoint_sha256,
    )
    assert admitted["active"]
    assert admitted["admission_kind"] == "exact_policy_contract"
    assert admitted["effective_contract_sha256"] == contract_sha256

    monkeypatch.setenv(
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256", "d" * 64,
    )
    with pytest.raises(RuntimeError, match="receipt disagrees"):
        _attest_warm_start_policy_contract(
            world_selection=selection,
            extension_width=0,
            source_checkpoint_sha256=checkpoint_sha256,
        )


def test_runner_rejects_target_only_cli_event_adaptation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters._mjlab_runner import (
        _attest_warm_start_policy_contract,
    )

    for name in (
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON",
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256",
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256",
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="full immutable policy-contract"):
        _attest_warm_start_policy_contract(
            world_selection=tmp_path / "missing.json",
            extension_width=3,
            source_checkpoint_sha256="c" * 64,
        )


def test_runner_rejects_pinned_target_or_runtime_interface_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sculptor.policy_contract as policy_contract
    from sculptor.adapters._mjlab_runner import (
        _attest_warm_start_policy_contract,
    )

    project = tmp_path / "project"
    selection = project / "env" / "selection_v1.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}\n", encoding="utf-8")
    event_contract = {
        "schema": 3,
        "identity": {"adapter_class": "MjlabAdapter", "task_id": "g1"},
        "event_observation": {
            "ordered_phase_ids": ["route", "jump", "hold"],
        },
    }
    event_sha256 = policy_contract.contract_fingerprint(event_contract)
    compatibility = {
        "type": "exact_policy_contract",
        "from_schema": 3,
        "to_schema": 3,
        "optimizer_resume": False,
    }
    receipt = {
        "schema": 1,
        "source": {
            "contract": event_contract,
            "contract_sha256": event_sha256,
        },
        "target": {
            "contract": event_contract,
            "contract_sha256": event_sha256,
        },
        "compatibility": compatibility,
    }
    for name, value in {
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON": json.dumps(
            receipt, sort_keys=True),
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON": json.dumps(
            event_contract, sort_keys=True),
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256": event_sha256,
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256": event_sha256,
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON": json.dumps(
            compatibility, sort_keys=True),
    }.items():
        monkeypatch.setenv(name, value)

    drifted_contract = {**event_contract, "schema": 2}
    monkeypatch.setattr(
        policy_contract,
        "build_project_policy_contract",
        lambda *_args, **_kwargs: drifted_contract,
    )
    with pytest.raises(RuntimeError, match="differs from the pre-queue"):
        _attest_warm_start_policy_contract(
            world_selection=selection,
            extension_width=0,
            source_checkpoint_sha256="c" * 64,
        )

    monkeypatch.setattr(
        policy_contract,
        "build_project_policy_contract",
        lambda *_args, **_kwargs: event_contract,
    )
    with pytest.raises(RuntimeError, match="runtime world has no event"):
        _attest_warm_start_policy_contract(
            world_selection=selection,
            extension_width=0,
            source_checkpoint_sha256="c" * 64,
        )

    unequal_source = {
        **event_contract,
        "identity": {
            "adapter_class": "MjlabAdapter",
            "task_id": "different-task",
        },
    }
    unequal_source_sha = policy_contract.contract_fingerprint(unequal_source)
    receipt["source"] = {
        "contract": unequal_source,
        "contract_sha256": unequal_source_sha,
    }
    monkeypatch.setenv(
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        json.dumps(receipt, sort_keys=True),
    )
    monkeypatch.setenv(
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256", unequal_source_sha,
    )
    with pytest.raises(RuntimeError, match="receipt disagrees"):
        _attest_warm_start_policy_contract(
            world_selection=selection,
            extension_width=3,
            source_checkpoint_sha256="c" * 64,
        )

    non_prefix_source = {
        "schema": 2,
        "identity": {
            "adapter_class": "MjlabAdapter",
            "task_id": "g1",
        },
    }
    non_prefix_sha = policy_contract.contract_fingerprint(non_prefix_source)
    declared_migration = {
        "type": "zero_initialized_event_phase_observation",
        "from_schema": 2,
        "to_schema": 3,
        "observation_term": "authored_event_phase",
        "extension_width": 3,
        "ordered_phase_ids": ["route", "jump", "hold"],
        "optimizer_resume": False,
    }
    receipt["source"] = {
        "contract": non_prefix_source,
        "contract_sha256": non_prefix_sha,
    }
    receipt["compatibility"] = declared_migration
    monkeypatch.setenv(
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        json.dumps(receipt, sort_keys=True),
    )
    monkeypatch.setenv(
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256", non_prefix_sha,
    )
    monkeypatch.setenv(
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON",
        json.dumps(declared_migration, sort_keys=True),
    )
    with pytest.raises(RuntimeError, match="receipt disagrees"):
        _attest_warm_start_policy_contract(
            world_selection=selection,
            extension_width=3,
            source_checkpoint_sha256="c" * 64,
        )


def test_remote_device_env_forwards_policy_contract_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters.mjlab import MjlabAdapter

    pins = {
        "SCULPTOR_WARM_START_CHECKPOINT_SHA256": "c" * 64,
        "SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256": "c" * 64,
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON": (
            '{"schema":1}'
        ),
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON": '{"schema":3}',
        "SCULPTOR_EFFECTIVE_POLICY_CONTRACT_SHA256": "a" * 64,
        "SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256": "b" * 64,
        "SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON": (
            '{"type":"exact_policy_contract"}'
        ),
    }
    for key, value in pins.items():
        monkeypatch.setenv(key, value)

    remote_env, runner_device = MjlabAdapter._remote_device_env("cuda:2")

    assert runner_device == "cuda:0"
    assert remote_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert {key: remote_env[key] for key in pins} == pins


def test_remote_policy_contract_warm_start_is_explicitly_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor.adapters.mjlab import (
        _guard_remote_policy_contract_warm_start,
    )

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"policy")
    monkeypatch.setenv(
        "SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON",
        '{"schema":1}',
    )
    with pytest.raises(RuntimeError, match="remote mirror paths"):
        _guard_remote_policy_contract_warm_start(checkpoint)
    _guard_remote_policy_contract_warm_start(None)


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


def test_clearance_maneuver_reward_firewall_is_per_env_and_capability_gated(
) -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import (
        _apply_clearance_maneuver_reward_firewall,
        _clearance_maneuver_primary_scale,
    )

    command = SimpleNamespace(
        _clearance_shifts=torch.tensor([
            [0.268, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.268, 0.0, 0.0],
        ]),
        _waypoint_index=torch.tensor([0, 0, 1, 2, 3]),
        _clearance_stage_complete=torch.tensor([
            False, True, False, False, False,
        ]),
        _clearance_followthrough_pending=torch.tensor([
            False, False, True, False, False,
        ]),
    )

    class CommandManager:
        active_terms = ("route", "ordinary")

        @staticmethod
        def get_term(name):
            if name == "route":
                return command
            return SimpleNamespace()

    env = SimpleNamespace(
        num_envs=5,
        device=torch.device("cpu"),
        command_manager=CommandManager(),
    )

    scale = _clearance_maneuver_primary_scale(env)
    # Both the outside approach and the through-disk traversal are command-only
    # phases around the same immutable predicate. Predicate-centered generated
    # shaping remains withheld until that predicate advances.
    assert scale.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    components = {
        "dense": rewards.clone(),
        "matrix": torch.stack((rewards, rewards + 10.0), dim=-1),
        "scalar": torch.tensor(7.0),
        "metadata": "unchanged",
    }
    scaled_rewards, scaled_components = (
        _apply_clearance_maneuver_reward_firewall(
            env, rewards, components))

    assert scaled_rewards.tolist() == [0.0, 0.0, 0.0, 0.0, 5.0]
    assert scaled_components["dense"].tolist() == [
        0.0, 0.0, 0.0, 0.0, 5.0,
    ]
    assert scaled_components["matrix"].tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [5.0, 15.0],
    ]
    assert scaled_components["scalar"].item() == 7.0
    assert scaled_components["metadata"] == "unchanged"

    # A command manager without the typed clearance capability is unchanged.
    plain_env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(
            active_terms=(),
        ),
    )
    assert _clearance_maneuver_primary_scale(
        plain_env).tolist() == [1.0, 1.0]


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


def test_terminal_stillness_rejects_motionless_collapse() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import (
        _authored_terminal_stillness_state,
    )

    command = SimpleNamespace(
        is_standing_env=torch.tensor([True, True, True]))

    class CommandManager:
        active_terms = ("route",)

        @staticmethod
        def get_term(name):
            assert name == "route"
            return command

    default = torch.zeros(3, 4)
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(3, 3),
        root_link_ang_vel_b=torch.zeros(3, 3),
        joint_vel=torch.zeros(3, 4),
        joint_pos=default.clone(),
        default_joint_pos=default,
        projected_gravity_b=torch.tensor([
            [0.0, 0.0, -1.0],   # honest neutral hold
            [0.0, 0.0, -0.5],   # motionless but tipped/crouched
            [0.0, 0.0, -1.0],   # upright base, collapsed articulation
        ]),
    )
    data.joint_pos[2] = 1.0
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        command_manager=CommandManager(),
        scene={"robot": SimpleNamespace(data=data)},
    )

    standing, score, _speed, quiet = _authored_terminal_stillness_state(
        env, lin_std=0.12, ang_std=0.5, joint_std=1.0)

    assert standing.tolist() == [True, True, True]
    assert quiet.tolist() == [True, False, False]
    torch.testing.assert_close(score[0], torch.tensor(1.0))
    # Posture is a strict smooth conjunction, not a small additive bonus or a
    # geometric mean that dilutes one bad factor: a perfectly motionless but
    # tipped or folded body must retain less than one fifth of terminal income.
    assert score[1].item() < 0.2
    assert score[2].item() < 0.2


def test_motion_quality_info_is_reset_safe_and_embodiment_agnostic() -> None:
    torch = pytest.importorskip("torch")

    from sculptor.adapters._mjlab_runner import _motion_quality_info

    action = torch.tensor([
        [1.0, -1.0],
        [2.0, 0.0],
    ])
    previous = torch.zeros_like(action)
    joint_vel = torch.tensor([
        [3.0, 4.0, 0.0],
        [0.0, 6.0, 8.0],
    ])
    episode_length = torch.tensor([1.0, 2.0])

    info, next_anchor = _motion_quality_info(
        action, previous, episode_length, joint_vel)

    assert info["action_rate"][0].item() == 0.0
    assert info["joint_vel_rms"][0].item() == 0.0
    torch.testing.assert_close(
        info["action_rate"][1], torch.sqrt(torch.tensor(2.0)))
    torch.testing.assert_close(
        info["joint_vel_rms"][1], torch.sqrt(torch.tensor(100.0 / 3.0)))
    torch.testing.assert_close(next_anchor, action)
    assert next_anchor.data_ptr() != action.data_ptr()


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


def test_robot_materialization_owns_and_pins_actuator_physics(monkeypatch) -> None:
    """Authored admission and runtime share one immutable actuator model."""
    pytest.importorskip("mjlab")
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.tasks.registry import load_env_cfg

    from sculptor.adapters._mjlab_runner import _enforce_actuator_limits
    from sculptor.world.capabilities import (
        build_base_robot_entity_cfg,
        build_robot_entity_cfg,
        resolve_robot_capability,
    )
    from sculptor.world.compiler import _robot_asset_hash_from_cfg

    capability = resolve_robot_capability("unitree_g1:base")
    assert isinstance(
        capability.actuator_profile.velocity_limits_rad_s, tuple
    )
    first = build_robot_entity_cfg(capability)
    second = build_robot_entity_cfg(capability)
    assert first.articulation is not second.articulation
    assert first.articulation.actuators is not second.articulation.actuators
    assert all(
        isinstance(actuator, DcMotorActuatorCfg)
        for actuator in first.articulation.actuators
    )
    admitted_hash = _robot_asset_hash_from_cfg(first)
    base_hash = _robot_asset_hash_from_cfg(
        build_base_robot_entity_cfg(capability)
    )
    assert base_hash != admitted_hash

    first.articulation.actuators = ()
    assert second.articulation.actuators
    assert _robot_asset_hash_from_cfg(
        build_robot_entity_cfg(capability)
    ) == admitted_hash

    # The legacy registered-task path also transforms copy-on-write and cannot
    # poison later admission factory calls in the same worker process.
    monkeypatch.delenv("RS_ENFORCE_ACTUATOR_LIMITS", raising=False)
    legacy = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    _enforce_actuator_limits(legacy)
    assert _robot_asset_hash_from_cfg(
        legacy.scene.entities["robot"]
    ) == admitted_hash
    assert _robot_asset_hash_from_cfg(
        build_robot_entity_cfg(capability)
    ) == admitted_hash


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


def test_biped_foot_site_selector_preserves_world_positions_and_order() -> None:
    import numpy as np

    from sculptor.adapters._mjlab_runner import (
        _select_biped_foot_site_positions,
    )

    sites = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    selected = _select_biped_foot_site_positions(sites, 3, 1)

    assert selected is not None
    left, right = selected
    np.testing.assert_array_equal(left, sites[:, 3, :])
    np.testing.assert_array_equal(right, sites[:, 1, :])


@pytest.mark.parametrize(
    ("sites_shape", "left_index", "right_index"),
    [
        (None, 0, 1),
        ((2, 3), 0, 1),
        ((2, 2, 4), 0, 1),
        ((2, 2, 3), -1, 1),
        ((2, 2, 3), 0, 0),
        ((2, 2, 3), 0, 2),
    ],
)
def test_biped_foot_site_selector_rejects_ambiguous_or_malformed_tables(
    sites_shape, left_index: int, right_index: int,
) -> None:
    import numpy as np

    from sculptor.adapters._mjlab_runner import (
        _select_biped_foot_site_positions,
    )

    sites = (
        None
        if sites_shape is None
        else np.zeros(sites_shape, dtype=np.float32)
    )
    assert _select_biped_foot_site_positions(
        sites, left_index, right_index,
    ) is None


# ── reference-tracking narrows the realism floor (HANDOFF §10, option 2) ──
#
# A reference dictates posture, gait and body motion frame by frame, so the
# task's own posture/gait priors stop complementing the sculpted reward and
# start competing with it. Measured on the first Tier-D attempt: 14 task terms
# at 0.3x against one tracking term at 1.0x produced a policy that reproduced
# 28% of the reference's joint amplitude and could not beat a static pose.
def test_hardware_safety_terms_survive_reference_tracking() -> None:
    """Limits and smoothness constrain the hardware without prescribing a
    pose, so they are compatible with any reference and stay on."""
    from sculptor.adapters._mjlab_runner import _is_hardware_safety_term

    for name in ("dof_pos_limits", "robot_dof_pos_limits", "dof_vel_limits",
                 "self_collisions", "action_rate_l2", "joint_limits",
                 "dof_torque_limits"):
        assert _is_hardware_safety_term(name), name


def test_posture_and_gait_terms_are_not_treated_as_safety() -> None:
    """These are exactly the terms that fight the reference."""
    from sculptor.adapters._mjlab_runner import _is_hardware_safety_term

    for name in ("pose", "upright", "track_linear_velocity",
                 "track_angular_velocity", "foot_clearance", "air_time",
                 "foot_slip", "foot_swing_height", "angular_momentum",
                 "body_ang_vel", "soft_landing"):
        assert not _is_hardware_safety_term(name), name


def test_the_g1_task_term_split_is_what_we_intend(tmp_path) -> None:
    """Pin the actual split over the real G1 velocity task's 14 terms, so a
    future task-term rename cannot silently re-enable a posture prior."""
    from sculptor.adapters._mjlab_runner import _is_hardware_safety_term

    shipped = [
        "action_rate_l2", "air_time", "angular_momentum", "body_ang_vel",
        "dof_pos_limits", "foot_clearance", "foot_slip", "foot_swing_height",
        "pose", "self_collisions", "soft_landing", "track_angular_velocity",
        "track_linear_velocity", "upright",
    ]
    kept = sorted(n for n in shipped if _is_hardware_safety_term(n))
    assert kept == ["action_rate_l2", "dof_pos_limits", "self_collisions"]


def test_reward_module_flag_is_read_and_fails_soft(tmp_path) -> None:
    """The runner keys off REWARD_SPEC['reference_tracking']; an unreadable
    module must NOT be guessed as tracking, since that would silently change
    which task rewards are active."""
    from sculptor.adapters._mjlab_runner import _reward_module_declares

    tracking = tmp_path / "tracking.py"
    tracking.write_text(
        'REWARD_SPEC = {"reference_tracking": True}\n'
        "def compute_reward(s, a, ns, i):\n    return 0.0, {}\n",
        encoding="utf-8")
    assert _reward_module_declares(tracking, "reference_tracking") is True

    plain = tmp_path / "plain.py"
    plain.write_text(
        'REWARD_SPEC = {"version": "v1"}\n'
        "def compute_reward(s, a, ns, i):\n    return 0.0, {}\n",
        encoding="utf-8")
    assert _reward_module_declares(plain, "reference_tracking") is False

    broken = tmp_path / "broken.py"
    broken.write_text("this is not python(", encoding="utf-8")
    assert _reward_module_declares(broken, "reference_tracking") is False
    assert _reward_module_declares(None, "reference_tracking") is False


def test_reference_tracking_capability_covers_flat_mode_and_task_only() -> None:
    """Flat and mode rewards must drive the same runtime arbitration."""
    from types import SimpleNamespace

    from sculptor.adapters._mjlab_runner import (
        _reward_module_tracks_reference,
        _reward_spec_tracks_reference,
    )

    flat = {"reference_tracking": True}
    tracked_mode = {"tracking_enabled": True}
    task_only_mode = {"tracking_enabled": False}

    assert _reward_spec_tracks_reference(flat) is True
    assert _reward_spec_tracks_reference(tracked_mode) is True
    assert _reward_spec_tracks_reference(task_only_mode) is False
    assert _reward_spec_tracks_reference({"version": "plain"}) is False
    assert _reward_spec_tracks_reference(None) is False
    assert _reward_module_tracks_reference(
        SimpleNamespace(REWARD_SPEC=flat)
    ) is True
    assert _reward_module_tracks_reference(
        SimpleNamespace(REWARD_SPEC=tracked_mode)
    ) is True
    assert _reward_module_tracks_reference(
        SimpleNamespace(REWARD_SPEC=task_only_mode)
    ) is False


def test_generated_tracking_rewards_declare_the_flag() -> None:
    """Both tracking-reward generators must set it, or the runner silently
    keeps the competing posture priors."""
    import numpy as np

    from sculptor.refs.track import generate_tracking_reward_source

    src = generate_tracking_reward_source(
        clip_id="c", robot="g1", joint_names=["j0"],
        target_joint_pos=np.zeros((4, 1)), target_root_z=np.zeros(4),
        episode_len_steps=100, duration_s=2.0)
    ns: dict = {}
    exec(compile(src, "r", "exec"), ns)  # noqa: S102
    assert ns["REWARD_SPEC"]["reference_tracking"] is True


# ── rollout video: only the tracked env's authored geometry ───────────
def _fake_model(n_envs: int, per_env: int = 4):
    """Stand-in for an mjlab model carrying one authored course per env
    origin, the way `_world_spec_editor` emits them."""
    geom_names, site_names = [], []
    for env_index in range(n_envs):
        suffix = f"__env_{env_index:04d}"
        geom_names += [f"obstacle__box_{i:02d}__platform{suffix}"
                       for i in range(per_env)]
        site_names.append(f"zone__start{suffix}")
    geom_names.append("terrain")          # shared, must survive
    geom_names.append("unitree_g1:pelvis")

    geom_rgba = [[0.5, 0.5, 0.5, 1.0] for _ in geom_names]
    site_rgba = [[0.1, 0.9, 0.1, 0.25] for _ in site_names]
    return SimpleNamespace(
        ngeom=len(geom_names), nsite=len(site_names),
        geom_rgba=geom_rgba, site_rgba=site_rgba,
        geom=lambda i: SimpleNamespace(name=geom_names[i]),
        site=lambda i: SimpleNamespace(name=site_names[i]),
        _geom_names=geom_names, _site_names=site_names)


def test_rollout_video_hides_other_envs_authored_courses() -> None:
    """mjlab shares one model across parallel envs, so every env's authored
    course is always in it and always drawn — a field of identical courses that
    buries the one the tracked robot is running. `max_extra_envs=0` only hides
    neighbouring ROBOTS; static geometry needs this."""
    from sculptor.adapters._mjlab_runner import _hide_untracked_authored_geometry

    model = _fake_model(n_envs=8)
    env = SimpleNamespace(sim=SimpleNamespace(mj_model=model))

    _hide_untracked_authored_geometry(env, 3)

    for index, name in enumerate(model._geom_names):
        alpha = model.geom_rgba[index][3]
        if name.endswith("__env_0003") or "__env_" not in name:
            assert alpha == 1.0, f"{name} must stay visible"
        else:
            assert alpha == 0.0, f"{name} must be hidden"
    for index, name in enumerate(model._site_names):
        expected = 0.25 if name.endswith("__env_0003") else 0.0
        assert model.site_rgba[index][3] == expected


def test_rollout_video_culling_touches_alpha_only() -> None:
    """Collision geometry, contacts and observations must be untouched — the
    video has to show the same physics it always did."""
    from sculptor.adapters._mjlab_runner import _hide_untracked_authored_geometry

    model = _fake_model(n_envs=4)
    env = SimpleNamespace(sim=SimpleNamespace(mj_model=model))

    _hide_untracked_authored_geometry(env, 0)

    for row in model.geom_rgba:
        assert row[:3] == [0.5, 0.5, 0.5]   # only the alpha channel moved


def test_rollout_video_culling_is_a_no_op_on_a_single_env_scene() -> None:
    """With one origin `_world_spec_editor` emits unsuffixed names; nothing
    should be hidden, least of all the only course in the scene."""
    from sculptor.adapters._mjlab_runner import _hide_untracked_authored_geometry

    names = ["obstacle__box_01__platform", "zone__start", "terrain"]
    model = SimpleNamespace(
        ngeom=3, nsite=0, geom_rgba=[[1.0, 1.0, 1.0, 1.0]] * 3, site_rgba=[],
        geom=lambda i: SimpleNamespace(name=names[i]),
        site=lambda i: SimpleNamespace(name=""))
    env = SimpleNamespace(sim=SimpleNamespace(mj_model=model))

    _hide_untracked_authored_geometry(env, 0)

    assert all(row[3] == 1.0 for row in model.geom_rgba)


def test_rollout_video_culling_never_raises() -> None:
    """Cosmetics must never kill a rollout — a model that does not look the way
    we expect has to no-op, not except."""
    from sculptor.adapters._mjlab_runner import _hide_untracked_authored_geometry

    _hide_untracked_authored_geometry(SimpleNamespace(), 0)
    _hide_untracked_authored_geometry(
        SimpleNamespace(sim=SimpleNamespace(mj_model=object())), 0)


def test_terminal_progress_never_claims_success_after_interruption() -> None:
    from sculptor.adapters._mjlab_runner import _completed_iter_progress_event

    assert _completed_iter_progress_event(
        max_iterations=750,
        elapsed_s=178.7,
        completed=False,
    ) is None


def test_terminal_progress_reports_completion_after_learn_returns() -> None:
    from sculptor.adapters._mjlab_runner import _completed_iter_progress_event

    assert _completed_iter_progress_event(
        max_iterations=750,
        elapsed_s=178.74,
        completed=True,
    ) == {
        "type": "iter_progress",
        "rl_iter": 750,
        "rl_total": 750,
        "pct": 100.0,
        "elapsed_s": 178.7,
        "eta_s": 0.0,
    }
