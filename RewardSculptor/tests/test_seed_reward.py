"""§RL_SCULPTOR_AUDIT §4.4 (loop 4c, gap #4): goal-conditioned starter.

`_maybe_seed_goal_reward` replaces the pristine `sculpt init` constant
alive-bonus template with a goal-conditioned v1 at the start of the
first run. Offline — the LLM is the same stub-client convention as
test_edit.py; validation runs the REAL post-flight stack (probes +
variance pre-screen).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sculptor.kg.store import SculptorKG
from sculptor.sculpt import (
    _SEED_GOAL_CLIP,
    _V0_BATCHED_REWARD,
    _V0_SCALAR_REWARD,
    _is_pristine_starter_reward,
    _maybe_seed_goal_reward,
    _seed_reward_prompt,
)


# ── Fakes (mirroring test_edit.py conventions) ─────────────────────────────
@dataclass
class _FakeContract:
    observation_space_spec: Any
    action_space_spec: Any
    expected_info_keys: list[str]
    expected_components: Any = None
    supports_batched: bool = False
    state_schema: Any = None
    training_device: str = "any"


class _FakeAdapter:
    def __init__(self):
        import gymnasium as gym
        import numpy as np

        self._contract = _FakeContract(
            observation_space_spec=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
            action_space_spec=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            expected_info_keys=["x_velocity"],
        )

    def reward_contract(self):
        return self._contract


class _StubBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _StubResp:
    def __init__(self, text: str):
        self.content = [_StubBlock(text)]


class _StubMessages:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("stub ran out of responses")
        return _StubResp(self._responses.pop(0))


class _StubClient:
    def __init__(self, *responses: str):
        self.messages = _StubMessages(*responses)


class _ExplodingClient:
    """Raises on any use — for asserting the LLM was never consulted /
    that non-EditValidationError failures don't crash the run."""

    class _M:
        def create(self, **kwargs):
            raise ConnectionError("no API key / network down")

    messages = _M()


_VALID_V1 = '''\
"""v1 — goal-conditioned starter (stub)."""
from __future__ import annotations

REWARD_SPEC = {
    "version": "v1",
    "parent_hash": "%HASH%",
    "description": "seeded from the behavior goal",
    "author": "sculptor",
    "hyperparameters": {"forward_weight": 1.0, "alive_bonus": 0.1},
    "references": [],
    "grounding": {"forward_weight": "physics: unit weight on measured velocity"},
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"],
    }
    return sum(components.values()), components
'''


def _seed_project(tmp_path: Path, template: str = _V0_SCALAR_REWARD) -> Path:
    rewards = tmp_path / "rewards"
    rewards.mkdir(parents=True)
    (rewards / "__init__.py").write_text("", encoding="utf-8")
    (rewards / "v0.py").write_text(template, encoding="utf-8")
    return rewards


def _v1_source_for(rewards_dir: Path) -> str:
    import hashlib

    src = (rewards_dir / "v0.py").read_text(encoding="utf-8")
    return _VALID_V1.replace(
        "%HASH%", hashlib.sha256(src.encode("utf-8")).hexdigest()[:16])


# ── _is_pristine_starter_reward ───────────────────────────────────────────
def test_pristine_detects_both_shipped_templates(tmp_path: Path) -> None:
    for i, template in enumerate((_V0_SCALAR_REWARD, _V0_BATCHED_REWARD)):
        p = tmp_path / f"t{i}.py"
        p.write_text(template, encoding="utf-8")
        assert _is_pristine_starter_reward(p) is True


def test_pristine_rejects_real_rewards(tmp_path: Path) -> None:
    # Any of: sculptor author, non-v0 version, extra hyperparameters.
    edited = _V0_SCALAR_REWARD.replace('"author": "human"', '"author": "sculptor"')
    versioned = _V0_SCALAR_REWARD.replace('"version": "v0"', '"version": "v3"')
    grown = _V0_SCALAR_REWARD.replace(
        '"alive_bonus": 1.0,', '"alive_bonus": 1.0, "launch_weight": 5.0,')
    for i, src in enumerate((edited, versioned, grown)):
        p = tmp_path / f"r{i}.py"
        p.write_text(src, encoding="utf-8")
        assert _is_pristine_starter_reward(p) is False
    assert _is_pristine_starter_reward(tmp_path / "missing.py") is False


# ── _seed_reward_prompt ───────────────────────────────────────────────────
def test_seed_prompt_carries_goal_and_stays_bounded() -> None:
    prompt = _seed_reward_prompt("standing tuck jump with a clean landing")
    assert "standing tuck jump" in prompt
    assert "alive-bonus" in prompt
    # apply_prompt_edit rejects prompts > 2000 chars — even a huge goal
    # must stay under after clipping.
    huge = _seed_reward_prompt("jump " * 2000)
    assert len(huge) <= 2000
    assert len(_seed_reward_prompt("x" * 5000)) <= _SEED_GOAL_CLIP + 1100


# ── _maybe_seed_goal_reward ───────────────────────────────────────────────
def test_seed_generates_v1_from_pristine_template(tmp_path: Path) -> None:
    rewards = _seed_project(tmp_path)
    client = _StubClient(_v1_source_for(rewards))
    kg = SculptorKG(tmp_path / "kg.db")
    try:
        out = _maybe_seed_goal_reward(
            rewards_dir=rewards, behavior_goal="hop forward fast",
            adapter=_FakeAdapter(), kg_store=kg, client=client)
    finally:
        kg.close()
    assert out == rewards / "v1.py"
    assert out.is_file()
    assert "goal" in (rewards / "current.py").read_text() or (
        "v1.py" in (rewards / "current.py").read_text())
    # The goal reached the LLM.
    assert "hop forward fast" in client.messages.calls[0]["messages"][0]["content"]


def test_seed_skips_non_pristine_reward_without_llm_call(tmp_path: Path) -> None:
    rewards = _seed_project(
        tmp_path,
        _V0_SCALAR_REWARD.replace('"author": "human"', '"author": "sculptor"'))
    out = _maybe_seed_goal_reward(
        rewards_dir=rewards, behavior_goal="hop",
        adapter=_FakeAdapter(), kg_store=None, client=_ExplodingClient())
    assert out is None
    assert not (rewards / "v1.py").exists()


def test_seed_skips_when_later_versions_exist(tmp_path: Path) -> None:
    rewards = _seed_project(tmp_path)
    (rewards / "v3.py").write_text(_V0_SCALAR_REWARD, encoding="utf-8")
    out = _maybe_seed_goal_reward(
        rewards_dir=rewards, behavior_goal="hop",
        adapter=_FakeAdapter(), kg_store=None, client=_ExplodingClient())
    assert out is None


def test_seed_failure_keeps_template_and_returns_none(
        tmp_path: Path, capsys) -> None:
    rewards = _seed_project(tmp_path)
    out = _maybe_seed_goal_reward(
        rewards_dir=rewards, behavior_goal="hop forward",
        adapter=_FakeAdapter(), kg_store=None, client=_ExplodingClient())
    assert out is None
    assert not (rewards / "v1.py").exists()
    # v0 untouched, failure surfaced honestly.
    assert _is_pristine_starter_reward(rewards / "v0.py")
    captured = capsys.readouterr()
    assert "starter generation failed" in captured.err
    assert "seed_reward_failed" in captured.out


def test_seed_rejects_constant_generation_via_variance_probe(
        tmp_path: Path) -> None:
    """A still-constant 'goal-conditioned' generation must be rejected by
    the dead-reward pre-screen on BOTH attempts → template kept."""
    rewards = _seed_project(tmp_path)
    import hashlib

    src = (rewards / "v0.py").read_text(encoding="utf-8")
    h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    constant = _VALID_V1.replace("%HASH%", h).replace(
        'float(info.get("x_velocity", 0.0))', "0.0")
    client = _StubClient(constant, constant)   # attempt + retry
    out = _maybe_seed_goal_reward(
        rewards_dir=rewards, behavior_goal="hop",
        adapter=_FakeAdapter(), kg_store=None, client=client)
    assert out is None
    assert not (rewards / "v1.py").exists()
    assert len(client.messages.calls) == 2     # bounded: exactly 1 retry
