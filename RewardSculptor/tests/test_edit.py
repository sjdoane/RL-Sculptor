"""tests/test_edit.py — offline tests for `sculptor.edit.apply_edits`.

No live LLM calls. We stub the Anthropic client to return pre-written reward
module sources, then assert the validators catch the right things and the
happy path writes v1.py + current.py correctly.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.edit import (
    EditValidationError,
    _extract_formula_identifiers,
    _pre_validate,
    apply_edits,
)
from sculptor.kg.schema import Paper, make_paper_id
from sculptor.kg.store import SculptorKG


# ── Shared fixtures ────────────────────────────────────────────────────────
V0_REWARD_SOURCE = '''\
"""v0 — canonical Hopper reward."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v0",
    "description": "Hopper: forward_velocity + alive_bonus - ctrl_cost.",
    "author": "human",
    "parent_hash": "",
    "hyperparameters": {
        "forward_weight": 1.0,
        "alive_bonus": 1.0,
        "ctrl_cost_weight": 1e-3,
    },
    "references": [],
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    alive = 1.0
    a = np.asarray(action, dtype=np.float32)
    ctrl_cost = REWARD_SPEC["hyperparameters"]["ctrl_cost_weight"] * float(np.sum(a ** 2))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * alive,
        "ctrl_cost": -ctrl_cost,
    }
    reward = components["forward_velocity"] + components["alive_bonus"] + components["ctrl_cost"]
    return float(reward), components
'''


@dataclass
class _FakeBoxSpace:
    shape: tuple
    dtype: Any = float


@dataclass
class _FakeContract:
    observation_space_spec: Any
    action_space_spec: Any
    expected_info_keys: list[str]
    expected_components: list[str] | None = None


def _hopper_contract(expected_components=None) -> _FakeContract:
    import gymnasium as gym
    import numpy as np

    obs = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return _FakeContract(
        observation_space_spec=obs, action_space_spec=act,
        expected_info_keys=["x_velocity"],
        expected_components=expected_components,
    )


@pytest.fixture
def v0_path(tmp_path: Path) -> Path:
    p = tmp_path / "rewards" / "v0.py"
    p.parent.mkdir(parents=True)
    p.write_text(V0_REWARD_SOURCE, encoding="utf-8")
    (p.parent / "__init__.py").write_text("", encoding="utf-8")
    return p


@pytest.fixture
def kg(tmp_path: Path) -> SculptorKG:
    store = SculptorKG(tmp_path / "kg.db")
    # Only 1801.00690 is ingested — any other cited arxiv_id is "missing".
    store.add_node(Paper(
        id=make_paper_id("1801.00690"), arxiv_id="1801.00690",
        title="DeepMind Control Suite", authors=["Yuval Tassa"], year=2018))
    return store


# ── Stub Anthropic client ──────────────────────────────────────────────────
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


# ── Unit: formula identifier extraction ────────────────────────────────────
def test_extract_formula_identifiers_tolerance_case():
    ids = _extract_formula_identifiers(
        "0.5 * tolerance(torso_angle, bounds=(-0.1, 0.1), margin=0.2)")
    # 'bounds' and 'margin' are kwargs — not Name nodes, excluded.
    assert "torso_angle" in ids
    assert "tolerance" in ids
    assert "bounds" not in ids
    assert "margin" not in ids


def test_extract_formula_identifiers_info_dict_access():
    ids = _extract_formula_identifiers('info["x_velocity"] - 0.01 * np.sum(action ** 2)')
    assert "info" in ids
    assert "x_velocity" in ids  # identifier-like string literal is harvested
    assert "np" in ids
    assert "action" in ids
    # `sum` is reached as attribute access (np.sum) — not a bare Name node;
    # we don't harvest attrs, which is fine because `np` is already allowed.
    assert "sum" not in ids


def test_extract_formula_identifiers_invalid_python_falls_back_to_regex():
    ids = _extract_formula_identifiers("-w * ||a||^2")
    assert "w" in ids
    assert "a" in ids


# ── Pre-flight validation ─────────────────────────────────────────────────
def test_pre_validate_rejects_missing_paper_ref(v0_path, kg):
    import sculptor.edit as edit_mod

    current_mod = edit_mod._load_reward_module(v0_path)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="dummy",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="grounded to citation",
                suggested_value="3.0",
                paper_refs=["9999.99999"],  # NOT in KG
            ),
        ],
        confidence=0.9,
    )
    with pytest.raises(EditValidationError, match="paper_refs not in KG"):
        _pre_validate(diagnosis, _hopper_contract(), current_mod, kg)


def test_pre_validate_partitions_ungrounded_formula_field(v0_path, kg):
    """The torso_angle case — ungrounded formula identifier.

    Post-2026-04-23 partition refactor: a single bad edit no longer
    kills the whole batch. `_pre_validate` partitions it into
    `rejected_edits` with a reason; only an empty `applicable_edits`
    is a hard error. `apply_edits` itself still raises when the
    applicable list is empty (tested in happy-path stubs elsewhere).
    """
    import sculptor.edit as edit_mod

    current_mod = edit_mod._load_reward_module(v0_path)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="upright_tolerance", operation="add",
                rationale="literature-cited torso-upright term",
                suggested_value=(
                    "0.5 * tolerance(torso_angle, bounds=(-0.1, 0.1), margin=0.2)"
                ),
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.9,
    )
    plan = _pre_validate(diagnosis, _hopper_contract(), current_mod, kg)
    assert plan.applicable_edits == []
    assert len(plan.rejected_edits) == 1
    assert plan.rejected_edits[0].target_term == "upright_tolerance"
    assert len(plan.rejection_reasons) == 1
    assert "torso_angle" in plan.rejection_reasons[0]
    assert "requires_env_extension" in plan.rejection_reasons[0]


def test_pre_validate_partitions_modify_op_with_unknown_target_term(
    v0_path, kg
):
    """Diagnoser-common-error: using 'increase' / 'gate' / 'clip' with
    a NEW target_term name. Post-fix, rejected individually; post-fix
    the rejection message suggests using operation='add' as the escape
    hatch (this string lives in `_pre_validate`)."""
    import sculptor.edit as edit_mod

    current_mod = edit_mod._load_reward_module(v0_path)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="unknown_weight", operation="increase",
                rationale="cannot increase what doesn't exist",
                suggested_value="2.0", paper_refs=[],
            ),
        ],
        confidence=0.9,
    )
    plan = _pre_validate(diagnosis, _hopper_contract(), current_mod, kg)
    assert plan.applicable_edits == []
    assert len(plan.rejected_edits) == 1
    assert "unknown_weight" in plan.rejection_reasons[0]
    assert "operation='add'" in plan.rejection_reasons[0]


def test_pre_validate_partitions_mixed_batch_keeps_valid_edits(v0_path, kg):
    """Pins the 2026-04-23 overnight regression: Sam's 10 iters each
    proposed 5 edits, 1-3 of which were ungrounded. Pre-fix,
    `_pre_validate` raised on the first violation — ALL 5 edits
    dropped, `new_reward` stayed at v1 for the whole overnight run.
    Post-fix, the grounded ones survive."""
    import sculptor.edit as edit_mod

    current_mod = edit_mod._load_reward_module(v0_path)
    diagnosis = Diagnosis(
        failure_modes=["reward_hacking", "component_imbalance"], evidence="",
        proposed_edits=[
            # Valid: grounded existing hparam.
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="bump per DM Control",
                suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
            # Invalid: new name with modify-op 'clip'. (Mirrors Sam's
            # iter-2 edit[1]: 'clip' with target='kick_velocity_clip'.)
            ProposedEdit(
                target_term="alive_bonus_clip", operation="clip",
                rationale="cap alive bonus",
                suggested_value="5.0", paper_refs=[],
            ),
            # Valid: 'add' with a novel name is fine.
            ProposedEdit(
                target_term="upright_term", operation="add",
                rationale="novel — tolerance on info-derived thing",
                suggested_value="1.0", paper_refs=[],
            ),
        ],
        confidence=0.9,
    )
    plan = _pre_validate(diagnosis, _hopper_contract(), current_mod, kg)
    assert len(plan.applicable_edits) == 2, (
        f"expected 2 valid edits to survive, got "
        f"{[e.target_term for e in plan.applicable_edits]}"
    )
    assert {e.target_term for e in plan.applicable_edits} == {
        "alive_bonus", "upright_term",
    }
    assert len(plan.rejected_edits) == 1
    assert plan.rejected_edits[0].target_term == "alive_bonus_clip"
    assert "alive_bonus_clip" in plan.rejection_reasons[0]


def test_pre_validate_defers_requires_env_extension(v0_path, kg):
    import sculptor.edit as edit_mod

    current_mod = edit_mod._load_reward_module(v0_path)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            # Applicable
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="grounded edit",
                suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
            # Deferred — would need torso_angle, flagged honestly.
            ProposedEdit(
                target_term="upright_tolerance", operation="add",
                rationale="Would penalize torso tilt once env exposes torso_angle.",
                suggested_value=None,  # no ungrounded formula
                paper_refs=["1801.00690"],
                requires_env_extension=True,
            ),
        ],
        confidence=0.9,
    )
    plan = _pre_validate(diagnosis, _hopper_contract(), current_mod, kg)
    assert len(plan.applicable_edits) == 1
    assert plan.applicable_edits[0].target_term == "alive_bonus"
    assert len(plan.deferred_edits) == 1
    assert plan.deferred_edits[0].requires_env_extension is True
    assert plan.cited_arxiv_ids == ["1801.00690"]
    assert "arXiv:1801.00690" in plan.citation_by_arxiv_id["1801.00690"]


# ── Happy-path apply_edits with a stubbed LLM ─────────────────────────────
_V1_HAPPY_SOURCE = '''\
"""v1 — raise alive_bonus to 3.0 per DM Control bounded-reward guidance."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v1",
    "description": "v1: alive_bonus raised 1.0 -> 3.0 to increase survival pressure.",
    "author": "sculptor",
    "parent_hash": "__PARENT_HASH__",
    "hyperparameters": {
        "forward_weight": 1.0,
        "alive_bonus": 3.0,
        "ctrl_cost_weight": 1e-3,
    },
    "references": [
        {
            "arxiv_id": "1801.00690",
            "citation": "__CITATION__",
            "how_used": "alive_bonus raised per DM Control bounded-reward guidance",
        }
    ],
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    alive = 1.0
    a = np.asarray(action, dtype=np.float32)
    ctrl_cost = REWARD_SPEC["hyperparameters"]["ctrl_cost_weight"] * float(np.sum(a ** 2))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * alive,
        "ctrl_cost": -ctrl_cost,
    }
    reward = components["forward_velocity"] + components["alive_bonus"] + components["ctrl_cost"]
    return float(reward), components
'''


def _render_v1_source(parent_hash: str, citation: str) -> str:
    return _V1_HAPPY_SOURCE.replace("__PARENT_HASH__", parent_hash).replace(
        "__CITATION__", citation)


def test_apply_edits_happy_path(v0_path, kg, monkeypatch):
    import hashlib

    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    citation = cite("1801.00690", store=kg)
    source = _render_v1_source(parent_hash, citation)
    client = _StubClient(source)

    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="Raise alive bonus per DM Control.",
                suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.9,
        behavior_goal="run forward without falling",
    )
    contract = _hopper_contract()

    out_path = apply_edits(
        current_reward_path=v0_path, diagnosis=diagnosis,
        new_iter_id="v1", reward_contract=contract,
        kg_store=kg, client=client,
    )
    assert out_path == v0_path.parent / "v1.py"
    assert out_path.is_file()

    # Import the new module via file path and check the spec.
    spec = importlib.util.spec_from_file_location("v1_check", out_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)
    rs = new_mod.REWARD_SPEC
    assert rs["version"] == "v1"
    assert rs["parent_hash"] == parent_hash
    assert rs["hyperparameters"]["alive_bonus"] == 3.0
    assert rs["references"][0]["arxiv_id"] == "1801.00690"
    assert "arXiv:1801.00690" in rs["references"][0]["citation"]

    # current.py was written and re-exports.
    current_path = v0_path.parent / "current.py"
    assert current_path.is_file()
    # Point-check it loads by path (no `from .v1 ...` relative import).
    cur_spec = importlib.util.spec_from_file_location(
        "current_check", current_path)
    cur_mod = importlib.util.module_from_spec(cur_spec)
    cur_spec.loader.exec_module(cur_mod)
    assert cur_mod.REWARD_SPEC["version"] == "v1"
    assert cur_mod.compute_reward is new_mod.compute_reward or \
        cur_mod.REWARD_SPEC == new_mod.REWARD_SPEC


def test_apply_edits_retries_once_on_post_validation_failure(v0_path, kg):
    """First response is broken; second response fixes it."""
    import hashlib

    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    citation = cite("1801.00690", store=kg)

    broken = '''\
"""v1 — broken: missing REWARD_SPEC."""
def compute_reward(state, action, next_state, info):
    return 0.0, {"a": 0.0}
'''
    good = _render_v1_source(parent_hash, citation)
    client = _StubClient(broken, good)

    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="grounded", suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.9,
    )
    out_path = apply_edits(
        current_reward_path=v0_path, diagnosis=diagnosis,
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
    )
    assert out_path.is_file()
    assert len(client.messages.calls) == 2
    # The retry turn must include the validation error text.
    retry_msg = client.messages.calls[1]["messages"][0]["content"]
    assert "VALIDATION_ERRORS_ON_PREVIOUS_ATTEMPT" in retry_msg
    assert "REWARD_SPEC" in retry_msg


def test_apply_edits_raises_on_second_failure(v0_path, kg):
    broken = '''\
"""still missing REWARD_SPEC."""
def compute_reward(state, action, next_state, info):
    return 0.0, {"a": 0.0}
'''
    client = _StubClient(broken, broken)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="", suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.5,
    )
    with pytest.raises(EditValidationError):
        apply_edits(
            current_reward_path=v0_path, diagnosis=diagnosis,
            new_iter_id="v1", reward_contract=_hopper_contract(),
            kg_store=kg, client=client,
        )


def test_apply_edits_rejects_bad_reference_arxiv_id(v0_path, kg):
    """Generated module cites an arxiv_id that isn't in the KG."""
    import hashlib

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    # Stub emits a source with a fabricated arxiv_id.
    fabricated = _V1_HAPPY_SOURCE.replace("__PARENT_HASH__", parent_hash).replace(
        '"arxiv_id": "1801.00690"', '"arxiv_id": "4242.42424"'
    ).replace("__CITATION__", "Fabricated et al.")
    # Second attempt: same broken source.
    client = _StubClient(fabricated, fabricated)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="", suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.5,
    )
    with pytest.raises(EditValidationError, match="not in KG"):
        apply_edits(
            current_reward_path=v0_path, diagnosis=diagnosis,
            new_iter_id="v1", reward_contract=_hopper_contract(),
            kg_store=kg, client=client,
        )


def test_apply_edits_enforces_expected_components_subset(v0_path, kg):
    import hashlib

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    # Generated module introduces a bogus key not in expected_components.
    source = _V1_HAPPY_SOURCE.replace("__PARENT_HASH__", parent_hash).replace(
        "__CITATION__", "cite").replace(
        '"ctrl_cost": -ctrl_cost,',
        '"ctrl_cost": -ctrl_cost,\n        "bogus_key": 0.1,',
    )
    client = _StubClient(source, source)
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="", suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.5,
    )
    contract = _hopper_contract(
        expected_components=["forward_velocity", "alive_bonus", "ctrl_cost"])
    with pytest.raises(EditValidationError,
                       match="components dict has keys not in expected_components"):
        apply_edits(
            current_reward_path=v0_path, diagnosis=diagnosis,
            new_iter_id="v1", reward_contract=contract,
            kg_store=kg, client=client,
        )


def test_apply_edits_raises_when_all_edits_deferred(v0_path, kg):
    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="upright_tolerance", operation="add",
                rationale="needs torso_angle", suggested_value=None,
                paper_refs=["1801.00690"],
                requires_env_extension=True,
            ),
        ],
        confidence=0.5,
    )
    with pytest.raises(EditValidationError, match="no applicable edits"):
        apply_edits(
            current_reward_path=v0_path, diagnosis=diagnosis,
            new_iter_id="v1", reward_contract=_hopper_contract(),
            kg_store=kg, client=_StubClient(),
        )


# ── Phase A regression: schema-style (mjlab) contract pre-flight ────
def test_build_dummy_inputs_uses_state_schema_when_present(tmp_path):
    """Regression for the iter-0→iter-1 IndexError: `_build_dummy_inputs`
    must hand the reward module a dict keyed by `state_schema` when the
    contract has one. Pre-fix behavior returned a numpy array from the
    fallback space handler, which blew up inside the reward's batched
    path at `next_state["qpos"]`."""
    from sculptor.adapters.base import RewardContract
    from sculptor.edit import _build_dummy_inputs

    contract = RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=["episode_length", "terminated"],
        expected_components=None,
        supports_batched=True,
        training_device="gpu",
        state_schema={
            "qpos": (18,),
            "qvel": (18,),
            "actuator_force": (12,),
            "base_lin_vel_b": (3,),
            "base_ang_vel_b": (3,),
            "projected_gravity_b": (3,),
            "command_vel": (3,),
        },
    )
    state, action, next_state, info = _build_dummy_inputs(contract)
    assert isinstance(state, dict), "state must be a dict for schema contracts"
    assert isinstance(next_state, dict), "next_state must be a dict too"
    assert set(state.keys()) == set(contract.state_schema.keys())
    # Leading dim 1 so scalar-via-batched wrappers work.
    assert state["qpos"].shape == (1, 18)
    assert state["actuator_force"].shape == (1, 12)
    assert isinstance(info, dict)
    assert "episode_length" in info and "terminated" in info


def test_build_dummy_inputs_gym_path_unchanged():
    """Non-mjlab adapters (gym_sb3, no state_schema) still get the flat
    numpy-array dummies. Guard against a regression where the schema
    branch accidentally swallowed the gym path."""
    from sculptor.edit import _build_dummy_inputs

    contract = _hopper_contract()
    state, action, _, info = _build_dummy_inputs(contract)
    assert hasattr(state, "shape") and not isinstance(state, dict)
    # Hopper observation is (11,), action is (3,) — see _FakeBoxSpace
    # usage in _hopper_contract above.
    assert not isinstance(state, dict)
    assert not isinstance(action, dict)
    assert "x_velocity" in info


def test_post_validate_unlinks_staging_on_validation_failure(tmp_path):
    """Test 1 round 3 (2026-04-22): `_post_validate` writes Claude's
    output to a hidden `.staging.py` file FIRST, validates, then
    atomically renames to the target. On any validation failure the
    staging file is unlinked and the target is NEVER created — so the
    "reward rewrite failed" toast + polluted v1.py on disk bug is
    closed. Pre-fix the write hit the target path directly before any
    check, so a failed validation left a broken v1.py that the UI then
    loaded on subsequent tab visits."""
    from sculptor.edit import _post_validate, EditValidationError
    from sculptor.kg.store import SculptorKG

    class _KG:
        def get_node(self, _id):
            return None

    # Source with missing compute_reward → _post_validate raises.
    bad_source = "REWARD_SPEC = {}\n# no compute_reward — rejection\n"
    target = tmp_path / "v1.py"
    staging = target.with_name(f".{target.stem}.staging.py")

    from sculptor.adapters.base import RewardContract
    contract = RewardContract(
        observation_space_spec=None, action_space_spec=None,
        expected_info_keys=[], expected_components=None,
        supports_batched=False, training_device="cpu",
    )
    with pytest.raises(EditValidationError):
        _post_validate(
            bad_source, contract=contract, kg_store=_KG(),
            parent_hash="x" * 16, new_version="v1", write_to=target,
        )
    assert not target.exists(), (
        "target v1.py must not exist after validation failure"
    )
    assert not staging.exists(), (
        "staging file must be unlinked after validation failure"
    )


def test_apply_prompt_edit_threads_kg_arxiv_ids_into_paper_refs(monkeypatch, tmp_path):
    """Issue B (Test 1 2026-04-22): `apply_prompt_edit` must extract
    arxiv_ids from the KG `query_semantic` result's `source_paper_ids`
    and thread them into the synthetic ProposedEdit's `paper_refs`.
    Pre-fix `paper_refs=[]` → empty CITATIONS block in the Claude prompt
    → Claude emitted `references: []` despite populated grounding.
    """
    from sculptor.edit import apply_prompt_edit
    from sculptor.kg import query as kg_query_mod
    from sculptor.kg.query import TechniqueMatch
    from sculptor.kg.schema import Technique

    # Stub query_semantic to return one match with two source papers.
    fake_match = TechniqueMatch(
        technique=Technique(id="technique:foo", name="foo", description="d", tags=[]),
        description="d",
        paper_citation="Foo et al.",
        evidence="e",
        relevance_score=0.7,
        matched_on=["semantic"],
        source_paper_ids=["paper:1234.5678", "paper:2020.99999"],
    )
    monkeypatch.setattr(
        kg_query_mod, "query_semantic",
        lambda *a, **kw: [fake_match],
    )

    # Capture the diagnosis handed to `apply_edits` — that's where
    # paper_refs lives on the synthetic ProposedEdit.
    captured: dict = {}
    from sculptor import edit as edit_mod

    def fake_apply_edits(
        current_reward_path, diagnosis, new_iter_id, reward_contract,
        *, kg_store=None, client=None, on_event=None,
    ):
        captured["paper_refs"] = list(diagnosis.proposed_edits[0].paper_refs)
        captured["lit_context_count"] = len(diagnosis.literature_context)
        return tmp_path / "v1.py"

    monkeypatch.setattr(edit_mod, "apply_edits", fake_apply_edits)

    # Minimal stubs for adapter contract + KG store (not actually used
    # when apply_edits is stubbed out).
    class _FakeContract:
        pass

    rewards = tmp_path / "rewards"
    rewards.mkdir()
    (rewards / "v0.py").write_text(
        "REWARD_SPEC = {}\ndef compute_reward(s,a,n,i): return 0.0, {}\n",
        encoding="utf-8",
    )

    apply_prompt_edit(
        current_reward_path=rewards / "v0.py",
        user_prompt="balance the pole vertically",
        new_iter_id="v1",
        reward_contract=_FakeContract(),
        kg_store=object(),  # non-None triggers KG query branch
    )

    assert captured["paper_refs"] == ["1234.5678", "2020.99999"], (
        f"paper_refs not plumbed from lit_context: {captured['paper_refs']!r}"
    )
    assert captured["lit_context_count"] == 1


def test_post_validate_rejects_grounding_referencing_uncited_arxiv(monkeypatch, tmp_path):
    """Issue B cross-check (Test 1 2026-04-22): if `grounding` cites an
    arxiv_id that is NOT in `references`, `_post_validate` must reject.
    Pre-fix the validator only checked references-⊆-KG, so Claude could
    write `grounding: {w: "arXiv:1707.02286"}` with `references: []` and
    the edit would commit.
    """
    from sculptor.edit import EditValidationError, _post_validate
    from sculptor.kg.store import SculptorKG

    # Minimal KG that asserts ID presence — stub has_node to True for
    # anything so references-in-KG doesn't fire. We're testing the
    # grounding-vs-references consistency check.
    class _FakeKG:
        def get_node(self, _id):
            return object()  # non-None → "in KG"

    source = """
REWARD_SPEC = {
    "version": "v1",
    "parent_hash": "x" * 16,
    "author": "sculptor",
    "description": "t",
    "hyperparameters": {"w_u": 1.0},
    "references": [],
    "grounding": {"w_u": "arXiv:1707.02286 Mnih survey"},
}
def compute_reward(state, action, next_state, info):
    # State-dependent so the dead-reward variance pre-screen doesn't
    # fire first — this test targets the grounding-vs-references check.
    v = float(state[0]) + 1.0
    return v, {"w_u": v}
"""

    # Build a contract that won't trip other validator checks.
    from sculptor.adapters.base import RewardContract
    contract = RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=[],
        expected_components=None,
        supports_batched=False,
        training_device="cpu",
    )

    target = tmp_path / "v1.py"
    with pytest.raises(EditValidationError) as ei:
        _post_validate(
            source, contract=contract, kg_store=_FakeKG(),
            parent_hash="x" * 16, new_version="v1", write_to=target,
        )
    assert "1707.02286" in str(ei.value)
    assert "grounding" in str(ei.value).lower()


def test_query_semantic_filters_by_min_similarity(monkeypatch):
    """Issue G (Test 1 2026-04-22): `query_semantic` with min_similarity
    must drop matches whose cosine score is below the floor. Tuned at
    0.35 for prompt-edit callers — prevents humanoid papers being cited
    for Cartpole rewards on the current 46-paper seed KG."""
    import numpy as np

    from sculptor.kg import query as kg_query_mod
    from sculptor.kg.schema import Technique

    techs = [
        (Technique(id=f"technique:t{i}", name=f"t{i}", description=f"d{i}", tags=[]),
         np.array([1.0, 0.0, 0.0]))
        for i in range(3)
    ]
    monkeypatch.setattr(kg_query_mod, "_ensure_technique_embeddings",
                        lambda store, model: techs)
    # Embed text to a vector that produces cosine=1.0 with [1,0,0].
    monkeypatch.setattr(kg_query_mod, "_embed_text",
                        lambda text, model: np.array([0.25, 0.0, 0.0]))
    # With qv=[0.25,0,0] and pool=[1,0,0], sim = 0.25 < 0.35 floor.

    # Stub store to avoid creating a real one.
    class _StoreStub:
        def neighbors(self, *a, **kw):
            return iter([])
        def close(self):
            pass

    from sculptor.kg.query import query_semantic
    result = query_semantic(
        "cartpole balance",
        top_k=5,
        store=_StoreStub(),
        min_similarity=0.35,
    )
    assert result == [], (
        f"expected empty result (all similarities below 0.35 floor), "
        f"got {result!r}"
    )


def test_build_dummy_inputs_handles_cartpole_schema():
    """S4 (bug #6.6 / T1) — Cartpole's 3-key schema (qpos, qvel,
    actuator_force; no root_link_*, no command_vel) must round-trip
    through `_build_dummy_inputs` without crashing. Previously only
    the 7-key G1/Go1 shape was exercised at unit level; this pins the
    minimal schema shape Claude's v1.py will see as `next_state`."""
    from sculptor.adapters.base import RewardContract
    from sculptor.edit import _build_dummy_inputs

    contract = RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=["episode_length", "terminated"],
        expected_components=None,
        supports_batched=True,
        training_device="gpu",
        state_schema={
            "qpos": (2,),
            "qvel": (2,),
            "actuator_force": (1,),
        },
    )
    state, action, next_state, info = _build_dummy_inputs(contract)
    assert isinstance(state, dict) and isinstance(next_state, dict)
    assert set(state.keys()) == {"qpos", "qvel", "actuator_force"}
    assert state["qpos"].shape == (1, 2)
    assert state["qvel"].shape == (1, 2)
    assert state["actuator_force"].shape == (1, 1)
    # A synthetic Claude-written v1 that reads `next_state["qpos"]`,
    # `next_state["qvel"]`, `next_state["actuator_force"]` must run
    # without IndexError. Exercising the exact call path the pre-flight
    # uses (pre-S4 this was only exercised against the 7-key shape).
    import numpy as np

    qpos = next_state["qpos"]
    qvel = next_state["qvel"]
    force = next_state["actuator_force"]
    # Cartpole-style reward: keep pole angle small + penalize velocity.
    pole_angle = qpos[:, 1]
    pole_vel = qvel[:, 1]
    reward = -(pole_angle**2) - 0.1 * (pole_vel**2) - 0.001 * (force[:, 0] ** 2)
    assert reward.shape == (1,)
    assert np.isfinite(reward).all()


# ── Phase B1 regression: reward prompt-edit consults KG ─────────────
def test_reward_prompt_edit_queries_kg(v0_path, kg, monkeypatch):
    """Reward prompt-edit (Rewards-tab flow) must semantic-search the
    KG on the user's prompt. Pre-fix `apply_prompt_edit` hardcoded
    `literature_context=[]`, so Claude saw no KG context for prompt-
    based edits even though `kg_store` was passed in."""
    from sculptor.edit import apply_prompt_edit
    from sculptor.kg import query as kg_query

    captured_queries: list[str] = []

    def fake_query_semantic(text, top_k=5, *, store=None, model_name=None, min_similarity=0.0):
        captured_queries.append(text)
        return []  # no matches is fine — we care about the call site

    monkeypatch.setattr(kg_query, "query_semantic", fake_query_semantic)

    # Claude-side stub — response isn't what we're testing.
    client = _StubClient(
        "REWARD_SPEC = {}\n"
        "def compute_reward(s, a, n, i): return 1.0, {'alive': 1.0}\n"
    )

    try:
        apply_prompt_edit(
            current_reward_path=v0_path,
            user_prompt="add an action-rate penalty to smooth gait",
            new_iter_id="v1",
            reward_contract=_hopper_contract(),
            kg_store=kg,
            client=client,
        )
    except Exception:  # noqa: BLE001
        # Downstream validators may still reject the stub response —
        # irrelevant to this test. The KG-query call is what we assert.
        pass

    assert captured_queries, (
        "apply_prompt_edit must call query_semantic with the user prompt"
    )
    assert "action-rate" in captured_queries[0]


# ── §7.2: Eureka reward-reflection block in edit prompt ─────────────
def test_edit_rewriter_prompt_contains_dead_component_rule() -> None:
    """The edit_rewriter system prompt must ship the dead-component rule
    so Claude knows how to treat flat components in TRAINING_FEEDBACK."""
    from sculptor.prompts import load_prompt
    prompt = load_prompt("edit_rewriter")
    # Keyword sentinels — tight enough to catch removal, loose enough
    # that wording tweaks don't break the test.
    assert "Dead-component rule" in prompt, (
        "edit_rewriter prompt missing the Dead-component rule section"
    )
    assert "TRAINING_FEEDBACK" in prompt
    assert "5" in prompt and "%" in prompt  # the <5% threshold


def test_apply_edits_injects_training_feedback_when_iter_dir_has_file(
    v0_path, kg, tmp_path,
):
    """When `iter_dir/reward_trajectory.json` exists, `apply_edits` must
    load it and pass it into the editor prompt so the dead-component
    rule can fire."""
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()
    (iter_dir / "reward_trajectory.json").write_text(json.dumps({
        "alive_bonus": [1.0, 1.0, 1.0, 1.0],    # dead signal
        "forward_velocity": [0.1, 0.3, 0.6],
    }))

    # Minimum-viable v1.py that will pass post-validate against
    # _hopper_contract() — carries forward the required spec fields.
    valid_v1 = '''\
"""v1 — stub accepted for TRAINING_FEEDBACK injection test."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v1",
    "parent_hash": "PARENT_HASH_PLACEHOLDER",
    "description": "Injection test.",
    "author": "sculptor",
    "hyperparameters": {"forward_weight": 1.0, "alive_bonus": 1.0, "ctrl_cost_weight": 1e-3},
    "references": [{"arxiv_id": "1801.00690", "citation": "stub", "how_used": "stub"}],
}

def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    a = np.asarray(action, dtype=np.float32)
    ctrl_cost = REWARD_SPEC["hyperparameters"]["ctrl_cost_weight"] * float(np.sum(a ** 2))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * 1.0,
        "ctrl_cost": -ctrl_cost,
    }
    return float(sum(components.values())), components
'''
    # Substitute the real parent_hash — post_validate checks exact match.
    import hashlib
    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()[:16]
    valid_v1 = valid_v1.replace("PARENT_HASH_PLACEHOLDER", parent_hash)
    client = _StubClient(valid_v1)

    diagnosis = Diagnosis(
        failure_modes=["reward_hacking"],
        evidence="alive_bonus flat across training — dead component",
        proposed_edits=[ProposedEdit(
            target_term="alive_bonus", operation="decrease",
            rationale="Per-component trajectory shows alive_bonus is flat at 1.0.",
            suggested_value="0.2", paper_refs=["1801.00690"],
        )],
        literature_context=[], confidence=0.8,
        iter_dir=None, behavior_goal="goal",
    )

    apply_edits(
        current_reward_path=v0_path,
        diagnosis=diagnosis,
        new_iter_id="v1",
        reward_contract=_hopper_contract(),
        kg_store=kg,
        client=client,
        iter_dir=iter_dir,
    )

    # Captured user message must contain the TRAINING_FEEDBACK block.
    assert client.messages.calls, "edit_rewriter was never called"
    user_msg = client.messages.calls[0]["messages"][0]["content"]
    assert "# TRAINING_FEEDBACK" in user_msg
    assert "alive_bonus:" in user_msg
    assert "forward_velocity:" in user_msg


def test_apply_edits_omits_training_feedback_when_iter_dir_none(
    v0_path, kg, tmp_path,
):
    """No iter_dir passed + diagnosis.iter_dir=None → no block. Prompt
    shape matches the pre-§7.2 behavior for untouched call sites."""
    valid_v1 = '''\
"""v1 — baseline injection test."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v1",
    "parent_hash": "PARENT_HASH_PLACEHOLDER",
    "description": "No-feedback test.",
    "author": "sculptor",
    "hyperparameters": {"forward_weight": 1.0, "alive_bonus": 1.0, "ctrl_cost_weight": 1e-3},
    "references": [{"arxiv_id": "1801.00690", "citation": "stub", "how_used": "stub"}],
}

def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    a = np.asarray(action, dtype=np.float32)
    ctrl_cost = REWARD_SPEC["hyperparameters"]["ctrl_cost_weight"] * float(np.sum(a ** 2))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * 1.0,
        "ctrl_cost": -ctrl_cost,
    }
    return float(sum(components.values())), components
'''
    import hashlib
    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()[:16]
    valid_v1 = valid_v1.replace("PARENT_HASH_PLACEHOLDER", parent_hash)
    client = _StubClient(valid_v1)

    diagnosis = Diagnosis(
        failure_modes=[], evidence="no feedback",
        proposed_edits=[ProposedEdit(
            target_term="alive_bonus", operation="decrease",
            rationale="test", suggested_value="0.2",
            paper_refs=["1801.00690"],
        )],
        literature_context=[], confidence=0.5,
        iter_dir=None, behavior_goal="goal",
    )

    apply_edits(
        current_reward_path=v0_path, diagnosis=diagnosis,
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
    )
    user_msg = client.messages.calls[0]["messages"][0]["content"]
    assert "# TRAINING_FEEDBACK" not in user_msg


# ── §Convergence (RL_SCULPTOR_AUDIT loop 3): dead-reward pre-screen ──────


def _variance_probe_contract():
    from sculptor.adapters.base import RewardContract
    return RewardContract(
        observation_space_spec=None,
        action_space_spec=None,
        expected_info_keys=["fallen", "base_height"],
        expected_components=None,
        supports_batched=False,
        training_device="cpu",
    )


def test_variance_probe_rejects_constant_reward():
    """The v0-class degenerate (constant alive-bonus) must be rejected
    before it can burn a GPU iteration training 'stand still'."""
    from types import SimpleNamespace
    from sculptor.edit import EditValidationError, _probe_reward_variance

    mod = SimpleNamespace(
        compute_reward=lambda s, a, ns, info: (1.0, {"alive_bonus": 1.0}))
    with pytest.raises(EditValidationError, match="state-independent"):
        _probe_reward_variance(mod, _variance_probe_contract())


def test_variance_probe_passes_state_dependent_reward():
    from types import SimpleNamespace
    from sculptor.edit import _probe_reward_variance

    def _r(s, a, ns, info):
        v = float(s[0])
        return v, {"height": v, "alive": 1.0}

    _probe_reward_variance(
        SimpleNamespace(compute_reward=_r), _variance_probe_contract())


def test_variance_probe_passes_difference_only_reward():
    """A reward built purely of next-state-minus-state terms is state-
    sensitive even though every UNIFORM fill zeroes it — the asymmetric
    probes must save it from a false reject."""
    from types import SimpleNamespace
    from sculptor.edit import _probe_reward_variance

    def _r(s, a, ns, info):
        v = float(ns[0]) - float(s[0])
        return v, {"ascent": v}

    _probe_reward_variance(
        SimpleNamespace(compute_reward=_r), _variance_probe_contract())


def test_variance_probe_passes_info_gated_reward():
    from types import SimpleNamespace
    from sculptor.edit import _probe_reward_variance

    def _r(s, a, ns, info):
        v = 2.0 * float(info["base_height"])
        return v, {"launch": v}

    _probe_reward_variance(
        SimpleNamespace(compute_reward=_r), _variance_probe_contract())


def test_variance_probe_tolerates_crashing_probes():
    """A reward that crashes on every non-zero fill leaves <2 surviving
    probes — insufficient evidence must PASS, never false-reject."""
    from types import SimpleNamespace
    from sculptor.edit import _probe_reward_variance

    def _r(s, a, ns, info):
        if float(s[0]) != 0.0 or float(ns[0]) != 0.0:
            raise ValueError("guarded")
        return 1.0, {"c": 1.0}

    _probe_reward_variance(
        SimpleNamespace(compute_reward=_r), _variance_probe_contract())


def test_apply_edits_injects_case_memory_block(v0_path, kg, monkeypatch):
    """§2026-07-03 case-memory upgrade: the rewrite prompt carries the same
    CASE MEMORY block the diagnoser sees, so the rewriter doesn't re-make a
    reward mistake a past run already measured as regressing."""
    import hashlib

    from sculptor.kg import cases as kg_cases
    from sculptor.kg.query import cite
    from sculptor.kg.schema import RunCase

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    source = _render_v1_source(parent_hash, cite("1801.00690", store=kg))
    client = _StubClient(source)

    past = kg_cases.CaseMatch(
        case=RunCase(
            id="case:past", task="run forward", symptom="component_imbalance",
            verdict="regressed",
            edits=["increase ctrl_cost_weight"],
            edit_summary=("iter 3: [component_imbalance] → applied: increase "
                          "ctrl_cost_weight → regressed (progress -0.0200)."),
        ),
        relevance_score=0.91,
    )
    captured_q: dict = {}

    def fake_query_cases(text, top_k=3, *, store=None, min_similarity=0.0,
                         model_name=None):
        captured_q["text"] = text
        return [past]

    monkeypatch.setattr(kg_cases, "query_cases", fake_query_cases)

    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[ProposedEdit(
            target_term="alive_bonus", operation="increase",
            rationale="Raise alive bonus.", suggested_value="3.0",
            paper_refs=["1801.00690"])],
        confidence=0.9,
        behavior_goal="run forward without falling",
    )
    apply_edits(
        current_reward_path=v0_path, diagnosis=diagnosis,
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
    )
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "# CASE MEMORY" in prompt
    assert "increase ctrl_cost_weight" in prompt
    assert "[-]" in prompt                       # regressed marker
    # the query keyed on goal + failure modes.
    assert "run forward" in captured_q["text"]
    assert "component_imbalance" in captured_q["text"]


def test_apply_edits_case_memory_failure_is_silent(v0_path, kg, monkeypatch):
    """A broken case query must not block the edit (advisory only)."""
    import hashlib

    from sculptor.kg import cases as kg_cases
    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    client = _StubClient(_render_v1_source(parent_hash, cite("1801.00690", store=kg)))
    monkeypatch.setattr(
        kg_cases, "query_cases",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing")))

    diagnosis = Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[ProposedEdit(
            target_term="alive_bonus", operation="increase",
            rationale="x", suggested_value="3.0",
            paper_refs=["1801.00690"])],
        confidence=0.9, behavior_goal="run",
    )
    out = apply_edits(
        current_reward_path=v0_path, diagnosis=diagnosis,
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
    )
    assert out.is_file()
    assert "# CASE MEMORY" not in client.messages.calls[0]["messages"][0]["content"]


# ── §best-of-K candidate edits ────────────────────────────────────────────
_VK_BATCHED_SOURCE = '''\
"""v1 — best-of-K candidate (forward_weight __W__), with batched path."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v1",
    "description": "candidate forward_weight=__W__",
    "author": "sculptor",
    "parent_hash": "__PARENT_HASH__",
    "hyperparameters": {
        "forward_weight": __W__,
        "alive_bonus": 1.0,
        "ctrl_cost_weight": 1e-3,
    },
    "references": [
        {
            "arxiv_id": "1801.00690",
            "citation": "__CITATION__",
            "how_used": "forward-weight tuning per DM Control guidance",
        }
    ],
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"],
    }
    return components["forward_velocity"] + components["alive_bonus"], components


def compute_reward_batched(state, action, next_state, info):
    import torch
    fwd = torch.as_tensor(info["x_velocity"]).reshape(-1).float()
    w = REWARD_SPEC["hyperparameters"]["forward_weight"]
    components = {
        "forward_velocity": w * fwd,
        "alive_bonus": torch.ones_like(fwd),
    }
    return components["forward_velocity"] + components["alive_bonus"], components
'''


def _render_vk_source(parent_hash: str, citation: str, weight: float) -> str:
    return (
        _VK_BATCHED_SOURCE
        .replace("__PARENT_HASH__", parent_hash)
        .replace("__CITATION__", citation)
        .replace("__W__", repr(weight))
    )


def _replay_batch(x_vel: float, n: int = 64):
    """Minimal (state, action, next_state, info) replay tuple — only
    info matters to the candidate sources above."""
    import torch

    state = torch.zeros((n, 11), dtype=torch.float32)
    action = torch.zeros((n, 3), dtype=torch.float32)
    next_state = torch.zeros((n, 11), dtype=torch.float32)
    info = {"x_velocity": torch.full((n,), float(x_vel))}
    return (state, action, next_state, info)


def _bok_diagnosis() -> Diagnosis:
    return Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus", operation="increase",
                rationale="grounded", suggested_value="3.0",
                paper_refs=["1801.00690"],
            ),
        ],
        confidence=0.9,
        behavior_goal="run forward without falling",
    )


def test_apply_edits_best_of_k_selects_by_hack_margin(v0_path, kg, tmp_path):
    """K=2 with replay evidence: the candidate whose reward separates the
    archived honest rollout from the archived exploit MORE (larger
    forward_weight → larger honest−hack margin) must win, even though it
    is NOT the first valid candidate."""
    import hashlib

    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    citation = cite("1801.00690", store=kg)
    # Candidate 0 (default framing): weak separation (w=0.1 → margin 0.3).
    # Candidate 1 (MINIMAL-DIFF framing): strong separation (w=1.0 → 3.0).
    weak = _render_vk_source(parent_hash, citation, 0.1)
    strong = _render_vk_source(parent_hash, citation, 1.0)
    client = _StubClient(weak, strong)

    iter_dir = tmp_path / "iter_bok"
    iter_dir.mkdir()
    out_path = apply_edits(
        current_reward_path=v0_path, diagnosis=_bok_diagnosis(),
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
        iter_dir=iter_dir,
        replay_inputs=_replay_batch(2.0),
        hack_replays=[{"label": "iter 3 (sit-farm)",
                       "replay_inputs": _replay_batch(-1.0)}],
        n_candidates=2,
    )
    assert out_path.is_file()
    assert len(client.messages.calls) == 2
    # The second call's prompt carries the framing directive.
    assert "MINIMAL-DIFF" in client.messages.calls[1]["messages"][0]["content"]

    spec = importlib.util.spec_from_file_location("v1_bok", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.REWARD_SPEC["hyperparameters"]["forward_weight"] == 1.0

    report = json.loads(
        (iter_dir / "edit_candidates.json").read_text(encoding="utf-8"))
    assert report["selected"] == 1
    assert len(report["candidates"]) == 2
    margins = [c["hack_margin"] for c in report["candidates"]]
    assert margins[1] > margins[0]
    assert (iter_dir / "edit_candidates" / "cand0.py").is_file()
    assert (iter_dir / "edit_candidates" / "cand1.py").is_file()


def test_apply_edits_best_of_k_no_replays_keeps_first_valid(v0_path, kg, tmp_path):
    """No replay evidence → margins are all None → the DEFAULT framing
    (candidate 0) wins on the lowest-index tiebreak."""
    import hashlib

    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    citation = cite("1801.00690", store=kg)
    first = _render_vk_source(parent_hash, citation, 0.5)
    second = _render_vk_source(parent_hash, citation, 2.0)
    client = _StubClient(first, second)

    iter_dir = tmp_path / "iter_bok2"
    iter_dir.mkdir()
    out_path = apply_edits(
        current_reward_path=v0_path, diagnosis=_bok_diagnosis(),
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client, iter_dir=iter_dir,
        n_candidates=2,
    )
    assert out_path.is_file()
    spec = importlib.util.spec_from_file_location("v1_bok2", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.REWARD_SPEC["hyperparameters"]["forward_weight"] == 0.5
    report = json.loads(
        (iter_dir / "edit_candidates.json").read_text(encoding="utf-8"))
    assert report["selected"] == 0


def test_apply_edits_best_of_k_all_invalid_falls_back_to_retry(v0_path, kg):
    """Every candidate fails validation → one repair retry (seeded with
    the first candidate's error), same semantics as the K=1 attempt 2."""
    import hashlib

    from sculptor.kg.query import cite

    parent_hash = hashlib.sha256(
        v0_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]
    citation = cite("1801.00690", store=kg)
    broken = '"""broken — no REWARD_SPEC."""\ndef compute_reward(s, a, n, i):\n    return 0.0, {"a": 0.0}\n'
    good = _render_v1_source(parent_hash, citation)
    client = _StubClient(broken, broken, good)

    out_path = apply_edits(
        current_reward_path=v0_path, diagnosis=_bok_diagnosis(),
        new_iter_id="v1", reward_contract=_hopper_contract(),
        kg_store=kg, client=client,
        n_candidates=2,
    )
    assert out_path.is_file()
    assert len(client.messages.calls) == 3
    retry_msg = client.messages.calls[2]["messages"][0]["content"]
    assert "VALIDATION_ERRORS_ON_PREVIOUS_ATTEMPT" in retry_msg
