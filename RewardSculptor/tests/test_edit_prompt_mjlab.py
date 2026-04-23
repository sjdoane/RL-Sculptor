"""Unit test: edit.py prompt builder emits batched instruction when the
reward contract declares supports_batched=True.

Per MJLAB_PIVOT_DESIGN §2.2, Q #2 — enforced via unit test so that any
future change to _build_user_prompt that accidentally drops the batched
block surfaces immediately without needing a live LLM call.
"""

from __future__ import annotations

from sculptor.adapters.base import RewardContract
from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.edit import _build_user_prompt


def _make_diagnosis() -> Diagnosis:
    return Diagnosis(
        behavior_goal="run forward fast without falling",
        failure_modes=["premature_termination"],
        evidence=["short episode lengths"],
        confidence=0.85,
        proposed_edits=[
            ProposedEdit(
                operation="increase",
                target_term="alive_bonus",
                rationale="increase episode longevity signal",
                suggested_value="alive_bonus * 2",
                paper_refs=[],
                requires_env_extension=False,
            ),
        ],
    )


def test_prompt_includes_batched_block_when_supports_batched_true() -> None:
    contract = RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=["episode_length", "terminated", "time_outs", "step_dt"],
        expected_components=None,
        supports_batched=True,
        training_device="gpu",
        min_gpu_memory_gb=6.0,
        state_schema={
            "qpos": (18,),
            "qvel": (18,),
            "base_lin_vel_b": (3,),
            "command_vel": (3,),
        },
    )
    diagnosis = _make_diagnosis()

    prompt = _build_user_prompt(
        current_source="# v0 reward\nREWARD_SPEC = {}\n"
                       "def compute_reward(s, a, ns, i): return (0.0, {})\n",
        current_version="v0",
        current_references=[],
        new_version="v1",
        parent_hash="abcdef1234567890",
        diagnosis=diagnosis,
        contract=contract,
        citation_map={},
        applicable_edits=diagnosis.proposed_edits,
        deferred_edits=[],
    )

    # Headline batched block and its key instruction must be present.
    assert "BATCHED_CONTRACT" in prompt, "batched block header missing"
    assert "compute_reward_batched" in prompt, (
        "prompt must instruct the LLM to emit compute_reward_batched"
    )
    assert "MUST emit BOTH" in prompt, "must-emit-both instruction missing"
    # REWARD_SPEC key
    assert "REWARD_SPEC['supports_batched'] MUST be True" in prompt
    # State schema has to leak in so the LLM writes correct shapes
    for key in contract.state_schema or {}:
        assert key in prompt, f"state schema key {key!r} missing from prompt"
    # training_device + supports_batched visible in contract header
    assert "supports_batched:   True" in prompt
    assert "training_device:    gpu" in prompt


def test_prompt_skips_batched_block_when_supports_batched_false() -> None:
    contract = RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=["x_velocity"],
        expected_components=None,
        # default supports_batched=False — gym-style
    )
    diagnosis = _make_diagnosis()

    prompt = _build_user_prompt(
        current_source="# v0\nREWARD_SPEC = {}\n"
                       "def compute_reward(s, a, ns, i): return (0.0, {})\n",
        current_version="v0",
        current_references=[],
        new_version="v1",
        parent_hash="abcdef",
        diagnosis=diagnosis,
        contract=contract,
        citation_map={},
        applicable_edits=diagnosis.proposed_edits,
        deferred_edits=[],
    )

    assert "BATCHED_CONTRACT" not in prompt, (
        "batched block must not appear for supports_batched=False contracts"
    )
    assert "compute_reward_batched" not in prompt, (
        "prompt must not mention batched path for scalar-only contracts"
    )
    assert "supports_batched:   False" in prompt
