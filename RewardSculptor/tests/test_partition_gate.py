"""tests/test_partition_gate.py — the shaping↔metric partition gate (#12).

Pins the deterministic, offline gate that fires at the reward-edit commit point
WHEN an objective metric steers the run. Anchored to g1-kick-v5, where the editor
reward-hacked the metric over 21 iters by lowering a completion gate and farming
the metric's own signal.

Two layers under test:
  * PRE-LLM screen (`screen_edits`) — NON-BLOCKING flags (partition overlap +
    gate-erosion), surfaced to the editor prompt + changelog.
  * POST-LLM check (`gate_threshold_regressions`) — the ONE hard gate: a
    same-named, positive, numerically-lowered completion-gate hparam is rejected;
    rename / removal / ambiguous / sign-ambiguous lowerings are advisory (no
    freeze).
Plus the end-to-end `apply_edits` behavior + the byte-identical no-metric path.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.edit import EditValidationError, apply_edits
from sculptor.eval import partition_gate as pg
from sculptor.eval.spec_metrics import metric_observables
from sculptor.kg.schema import Paper, make_paper_id
from sculptor.kg.store import SculptorKG


KICK_OBS = metric_observables("g1_kick")


# ── Unit: gate_kind (token-matched lexicon) ─────────────────────────────────
@pytest.mark.parametrize("name", [
    "completion_gate", "kick_completion_threshold", "qualify_gate",
    "swing_required", "kick_complete_gate",
])
def test_gate_kind_reject_roots(name):
    assert pg.gate_kind(name) == "reject"


@pytest.mark.parametrize("name", [
    "kick_cycle_weight", "min_torque", "contact_threshold",
    "min_swing_height", "floor_clearance", "gait_cycle_freq",
])
def test_gate_kind_flag_roots_are_ambiguous(name):
    # Direction-ambiguous roots are FLAG, never REJECT — lowering them can be a
    # legitimate tuning edit (gait_cycle_freq, contact_threshold, min_torque).
    assert pg.gate_kind(name) == "flag"


@pytest.mark.parametrize("name", [
    "torque_penalty_weight", "alive_bonus", "forward_weight",
    "terminated", "termination_penalty", "x_velocity",
])
def test_gate_kind_benign_names_are_none(name):
    # Token matching (not substring) keeps `terminated`/`termination` clear of
    # the `term`-less lexicon and avoids false hits on penalty weights.
    assert pg.gate_kind(name) is None


# ── Unit: observable_of / held_out_channels ─────────────────────────────────
def test_observable_of_alias_identity_and_miss():
    assert pg.observable_of("qvel", KICK_OBS) == "joint_vel"
    assert pg.observable_of("left_foot_swing_speed", KICK_OBS) == "joint_vel"
    assert pg.observable_of("left_foot_pos_b", KICK_OBS) == "left_foot_pos_b"
    assert pg.observable_of("projected_gravity_b", KICK_OBS) == "projected_gravity_b"
    # not a metric observable / not in this metric's set
    assert pg.observable_of("x_velocity", KICK_OBS) is None
    assert pg.observable_of("actuator_force", KICK_OBS) is None
    assert pg.observable_of("command_vel", KICK_OBS) is None


def test_observable_of_respects_active_metric_set():
    # base_height aliases root_link_pos_w; flagged only when the metric reads it.
    assert pg.observable_of("base_height", frozenset({"root_link_pos_w"})) == "root_link_pos_w"
    assert pg.observable_of("base_height", frozenset({"joint_vel"})) is None


def test_held_out_channels_lists_arrays_and_reward_aliases():
    held = pg.held_out_channels(KICK_OBS)
    assert "joint_vel" in held                  # metric array itself
    assert "qvel" in held                        # reward alias of joint_vel
    assert "left_foot_swing_speed" in held       # farmed via joint_vel
    assert "left_foot_pos_b" in held
    # an observable the metric does NOT read contributes no alias
    assert "x_velocity" not in held


# ── Unit: screen_edits (NON-BLOCKING flags) ─────────────────────────────────
def _edit(target, op, val=None):
    return ProposedEdit(target_term=target, operation=op, rationale="t",
                        suggested_value=val)


def test_screen_edits_flags_partition_and_gate_erosion_not_benign():
    edits = [
        _edit("forward_kick_speed", "add", "left_foot_swing_speed * 2.0"),  # farm
        _edit("kick_completion_gate", "decrease", "0.3"),                   # erosion
        _edit("torque_penalty_weight", "decrease", "0.01"),                # benign
    ]
    cur = {"kick_completion_gate": 0.8, "torque_penalty_weight": 0.02,
           "alive_bonus": 1.0}
    res = pg.screen_edits(edits, metric_observables=KICK_OBS, current_hparams=cur)
    assert len(res.flagged_edits) == 2          # benign torque edit NOT flagged
    joined = " ".join(res.flag_reasons)
    assert "partition" in joined and "left_foot_swing_speed" in joined
    assert "gate-erosion" in joined and "kick_completion_gate" in joined
    assert res.held_out == ["left_foot_swing_speed"]
    assert res.gate_hparams == ["kick_completion_gate"]


def test_screen_edits_numeric_lowering_of_ambiguous_gate_flags():
    # decrease via a `replace` value lower than current — ambiguous gate → flag.
    edits = [_edit("min_swing_height", "replace", "0.05")]
    cur = {"min_swing_height": 0.1}
    res = pg.screen_edits(edits, metric_observables=KICK_OBS, current_hparams=cur)
    assert len(res.flagged_edits) == 1
    assert "lowered" in res.flag_reasons[0]


def test_screen_edits_no_metric_overlap_when_set_empty():
    edits = [_edit("forward_kick_speed", "add", "left_foot_swing_speed")]
    res = pg.screen_edits(edits, metric_observables=frozenset(), current_hparams={})
    assert res.flagged_edits == []


# ── Unit: gate_threshold_regressions (the hard gate + freeze guards) ─────────
def test_regression_hard_on_completion_gate_lowering():
    parent = {"kick_completion_gate": 0.8, "forward_weight": 1.0}
    child = {"kick_completion_gate": 0.3, "forward_weight": 1.0}
    reg = pg.gate_threshold_regressions(parent, child)
    assert reg.hard and "kick_completion_gate" in reg.hard[0]
    assert reg.advisory == []


def test_regression_rename_is_advisory_not_hard():
    # B1 freeze-guard: a legit rename reads as removed+added — must NOT raise.
    parent = {"kick_completion_gate": 0.8}
    child = {"kick_phase_gate": 0.8}
    reg = pg.gate_threshold_regressions(parent, child)
    assert reg.hard == []
    assert any("removed" in a for a in reg.advisory)


def test_regression_ambiguous_flag_lowering_is_advisory():
    parent = {"min_swing_height": 0.1}
    child = {"min_swing_height": 0.05}
    reg = pg.gate_threshold_regressions(parent, child)
    assert reg.hard == []
    assert any("min_swing_height" in a for a in reg.advisory)


def test_regression_sign_ambiguous_reject_gate_is_advisory():
    # A non-positive parent value makes `<` unreliable → advisory, never hard.
    parent = {"completion_gate": -1.0}
    child = {"completion_gate": -2.0}
    reg = pg.gate_threshold_regressions(parent, child)
    assert reg.hard == []
    assert reg.advisory  # surfaced, but not blocking


def test_regression_raise_or_equal_gate_is_clean():
    parent = {"kick_completion_gate": 0.8}
    assert pg.gate_threshold_regressions(parent, {"kick_completion_gate": 0.9}).hard == []
    assert pg.gate_threshold_regressions(parent, {"kick_completion_gate": 0.8}).hard == []


def test_regression_scientific_notation_and_nonnumeric():
    # float() parses 1e-3; a lowered reject gate in sci-notation is hard.
    parent = {"completion_gate": 1e-2}
    assert pg.gate_threshold_regressions(parent, {"completion_gate": 1e-3}).hard
    # non-numeric child value (prose) → not a numeric regression, no raise.
    reg = pg.gate_threshold_regressions(parent, {"completion_gate": "about half"})
    assert reg.hard == []


# ── Unit: prompt block ──────────────────────────────────────────────────────
def test_prompt_block_empty_without_observables():
    assert pg.build_partition_prompt_block(frozenset(), pg.ScreenResult()) == ""


def test_prompt_block_lists_observables_and_gates():
    screen = pg.ScreenResult(
        flag_reasons=["edit[0]: gate-erosion: ..."],
        gate_hparams=["kick_completion_gate"],
    )
    block = pg.build_partition_prompt_block(KICK_OBS, screen)
    assert "# METRIC_PARTITION" in block
    assert "joint_vel" in block
    assert "kick_completion_gate" in block
    assert "REJECTED" in block  # the post-write enforcement warning


# ── Integration: apply_edits end-to-end ─────────────────────────────────────
@dataclass
class _FakeContract:
    observation_space_spec: Any
    action_space_spec: Any
    expected_info_keys: list[str]
    expected_components: list[str] | None = None


def _kick_contract() -> _FakeContract:
    import gymnasium as gym
    import numpy as np

    obs = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    act = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return _FakeContract(observation_space_spec=obs, action_space_spec=act,
                         expected_info_keys=["x_velocity"])


_PARENT_SOURCE = '''\
"""v0 — kick seed with a completion gate."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "v0",
    "description": "kick seed.",
    "author": "human",
    "parent_hash": "",
    "hyperparameters": {
        "forward_weight": 1.0,
        "alive_bonus": 1.0,
        "kick_completion_gate": 0.8,
    },
    "references": [],
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    gate = REWARD_SPEC["hyperparameters"]["kick_completion_gate"]
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * 1.0,
        "kick_gate": gate,
    }
    return float(sum(components.values())), components
'''


_CHILD_TEMPLATE = '''\
"""__VERSION__ — gate set to __GATE__."""
from __future__ import annotations
import numpy as np

REWARD_SPEC = {
    "version": "__VERSION__",
    "description": "kick_completion_gate -> __GATE__.",
    "author": "sculptor",
    "parent_hash": "__PARENT_HASH__",
    "hyperparameters": {
        "forward_weight": 1.0,
        "alive_bonus": 1.0,
        "__GATE_NAME__": __GATE__,
    },
    "references": [
        {
            "arxiv_id": "1801.00690",
            "citation": "__CITATION__",
            "how_used": "gate tuning",
        }
    ],
}


def compute_reward(state, action, next_state, info):
    fwd = float(info.get("x_velocity", 0.0))
    gate = REWARD_SPEC["hyperparameters"]["__GATE_NAME__"]
    components = {
        "forward_velocity": REWARD_SPEC["hyperparameters"]["forward_weight"] * fwd,
        "alive_bonus": REWARD_SPEC["hyperparameters"]["alive_bonus"] * 1.0,
        "kick_gate": gate,
    }
    return float(sum(components.values())), components
'''


def _render_child(parent_hash, citation, gate, version="v1",
                  gate_name="kick_completion_gate"):
    return (_CHILD_TEMPLATE
            .replace("__VERSION__", version)
            .replace("__PARENT_HASH__", parent_hash)
            .replace("__CITATION__", citation)
            .replace("__GATE_NAME__", gate_name)
            .replace("__GATE__", str(gate)))


class _StubBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _StubResp:
    def __init__(self, text):
        self.content = [_StubBlock(text)]


class _StubMessages:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResp(self._responses.pop(0))


class _StubClient:
    def __init__(self, *responses):
        self.messages = _StubMessages(*responses)


@pytest.fixture
def parent_path(tmp_path: Path) -> Path:
    p = tmp_path / "rewards" / "v0.py"
    p.parent.mkdir(parents=True)
    p.write_text(_PARENT_SOURCE, encoding="utf-8")
    (p.parent / "__init__.py").write_text("", encoding="utf-8")
    return p


@pytest.fixture
def kg(tmp_path: Path) -> SculptorKG:
    store = SculptorKG(tmp_path / "kg.db")
    store.add_node(Paper(
        id=make_paper_id("1801.00690"), arxiv_id="1801.00690",
        title="DeepMind Control Suite", authors=["Yuval Tassa"], year=2018))
    return store


def _diag():
    return Diagnosis(
        failure_modes=["component_imbalance"], evidence="",
        proposed_edits=[ProposedEdit(
            target_term="alive_bonus", operation="increase",
            rationale="raise survival pressure", suggested_value="1.0",
            paper_refs=["1801.00690"])],
        confidence=0.9, behavior_goal="kick a ball forward")


def _parent_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:16]


def test_apply_edits_rejects_completion_gate_lowering(parent_path, kg):
    """PROVE the v5 fix: a reward that LOWERS the completion gate is rejected
    post-write (both attempts lower it → EditValidationError)."""
    from sculptor.kg.query import cite

    ph = _parent_hash(parent_path)
    citation = cite("1801.00690", store=kg)
    lowered = _render_child(ph, citation, 0.3)        # 0.8 -> 0.3
    client = _StubClient(lowered, lowered)            # both attempts erode

    with pytest.raises(EditValidationError, match="partition gate"):
        apply_edits(
            current_reward_path=parent_path, diagnosis=_diag(),
            new_iter_id="v1", reward_contract=_kick_contract(),
            kg_store=kg, client=client, metric_observables=KICK_OBS,
        )
    # The eroded reward must NOT have been committed.
    assert not (parent_path.parent / "v1.py").is_file()


def test_apply_edits_allows_gate_raise_and_writes_report(parent_path, kg, tmp_path):
    """A reward that RAISES the gate commits; the partition report is persisted
    to iter_dir for changelog surfacing on the loop path."""
    from sculptor.kg.query import cite

    ph = _parent_hash(parent_path)
    citation = cite("1801.00690", store=kg)
    raised = _render_child(ph, citation, 0.9)         # 0.8 -> 0.9 (tightened)
    client = _StubClient(raised)
    iter_dir = tmp_path / "iter_0"
    iter_dir.mkdir()

    out = apply_edits(
        current_reward_path=parent_path, diagnosis=_diag(),
        new_iter_id="v1", reward_contract=_kick_contract(),
        kg_store=kg, client=client, iter_dir=iter_dir,
        metric_observables=KICK_OBS,
    )
    assert out.is_file()
    report = iter_dir / "partition_gate.json"
    assert report.is_file()
    import json
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "kick_completion_gate" in data["gate_hparams"]


def test_apply_edits_rename_does_not_freeze(parent_path, kg):
    """B1 freeze-guard: renaming the gate (removed+added, no same-named numeric
    lowering) must commit, not raise."""
    from sculptor.kg.query import cite

    ph = _parent_hash(parent_path)
    citation = cite("1801.00690", store=kg)
    renamed = _render_child(ph, citation, 0.8, gate_name="kick_phase_gate")
    client = _StubClient(renamed)

    out = apply_edits(
        current_reward_path=parent_path, diagnosis=_diag(),
        new_iter_id="v1", reward_contract=_kick_contract(),
        kg_store=kg, client=client, metric_observables=KICK_OBS,
    )
    assert out.is_file()


def test_metric_partition_block_present_only_with_metric(parent_path, kg):
    """The METRIC_PARTITION prompt block appears iff metric_observables is set —
    the no-metric path is byte-identical (no block, gate not enforced)."""
    from sculptor.kg.query import cite

    ph = _parent_hash(parent_path)
    citation = cite("1801.00690", store=kg)

    # With a metric: block present, gate-lowering REJECTED.
    lowered = _render_child(ph, citation, 0.3)
    client_m = _StubClient(lowered, lowered)
    with pytest.raises(EditValidationError):
        apply_edits(
            current_reward_path=parent_path, diagnosis=_diag(),
            new_iter_id="v1", reward_contract=_kick_contract(),
            kg_store=kg, client=client_m, metric_observables=KICK_OBS,
        )
    prompt_m = client_m.messages.calls[0]["messages"][0]["content"]
    assert "# METRIC_PARTITION" in prompt_m

    # Without a metric (None): no block, and the SAME gate-lowering COMMITS.
    client_n = _StubClient(lowered)
    out = apply_edits(
        current_reward_path=parent_path, diagnosis=_diag(),
        new_iter_id="v1", reward_contract=_kick_contract(),
        kg_store=kg, client=client_n, metric_observables=None,
    )
    assert out.is_file()                                  # not rejected
    prompt_n = client_n.messages.calls[0]["messages"][0]["content"]
    assert "# METRIC_PARTITION" not in prompt_n


def test_empty_observable_set_is_byte_identical(parent_path, kg):
    """Cartpole's metric_observables() is the empty frozenset — falsy, so the
    gate must no-op exactly like None: gate-lowering COMMITS, no prompt block,
    no side-file. Locks the byte-identical guarantee against a future refactor
    that swaps the truthiness guard for `is not None`."""
    from sculptor.kg.query import cite

    assert metric_observables("cartpole_balance") == frozenset()  # the premise
    ph = _parent_hash(parent_path)
    citation = cite("1801.00690", store=kg)
    lowered = _render_child(ph, citation, 0.3)            # would be rejected w/ metric
    client = _StubClient(lowered)

    out = apply_edits(
        current_reward_path=parent_path, diagnosis=_diag(),
        new_iter_id="v1", reward_contract=_kick_contract(),
        kg_store=kg, client=client, metric_observables=frozenset(),
    )
    assert out.is_file()                                  # NOT rejected (no-op)
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "# METRIC_PARTITION" not in prompt
