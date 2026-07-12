"""tests/test_decompose.py — Ship 14 coverage for sculptor.mission + decompose.

No real Anthropic or sentence-transformers calls: we stub
`client.messages.parse` and (for KG tests) inject a tmp SculptorKG or
monkeypatch `query_semantic`. The goal is exhaustive coverage of the
validators + Mission serialization, plus one happy-path end-to-end
smoke with a canned Claude response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sculptor.mission import (
    MissionValidationError,
    Mission,
    Stage,
    load_mission,
    save_mission,
    validate_mission,
)
from sculptor.decompose import (
    DecompositionError,
    _DecompositionModel,
    _StageModel,
    decompose_task,
)


# ── Minimal RewardContract stub ───────────────────────────────────────
@dataclass
class _FakeContract:
    expected_info_keys: list[str]
    expected_components: list[str] | None = None
    supports_batched: bool = False


def _default_contract() -> _FakeContract:
    return _FakeContract(
        expected_info_keys=[
            "base_height", "fallen", "terminated", "time_outs",
            "step_dt", "episode_length",
        ],
        expected_components=None,
    )


# ── Anthropic client stub (mirrors test_diagnose._Stub*) ─────────────
class _StubResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, responses: list):
        # Pop one response per `parse()` call; raises if the caller
        # exceeds our prepared list (useful for catching retry loops).
        self._responses = list(responses)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                f"stub ran out of responses after {len(self.calls)} calls"
            )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _StubResponse(nxt)


class _StubClient:
    def __init__(self, *responses):
        self.messages = _StubMessages(list(responses))


# ── Canonical good decomposition ──────────────────────────────────────
def _good_decomp_model() -> _DecompositionModel:
    """A 3-stage G1-kick curriculum Claude could plausibly emit."""
    return _DecompositionModel(
        decomposition_rationale=(
            "Standing is the prerequisite for single-leg stance which is "
            "the prerequisite for kicking. Three stages, each warm-starting "
            "from the prior."
        ),
        stages=[
            _StageModel(
                name="stand",
                goal_text="Hold upright double-stance without falling.",
                success_criterion=(
                    "behavior['mean_return'] > 0.3 and "
                    "behavior['mean_episode_length'] > 500"
                ),
                max_iterations=3,
                parent_stage=None,
                reward_seed_prompt=(
                    "alive_bonus (+0.1 when not fallen), "
                    "torso_upright (exp(-||base_ang_vel_b||^2)*0.3), "
                    "action_rate_penalty (-0.03*||a-a_prev||^2). "
                    "Zero the whole reward when fallen."
                ),
                kg_seed_papers=[],
            ),
            _StageModel(
                name="single_leg_stance",
                goal_text="Hold on one leg for at least 2 seconds.",
                success_criterion=(
                    "components['single_leg_stance'] > 0.4"
                ),
                max_iterations=5,
                parent_stage="stand",
                reward_seed_prompt=(
                    "Keep stand's three terms. Add single_leg_stance "
                    "component: +0.5 when root_link_pos_w[...,2]>0.6 and not "
                    "fallen (single-leg heuristic). Weight 1.0."
                ),
                kg_seed_papers=[],
            ),
            _StageModel(
                name="kick_and_return",
                goal_text=(
                    "From single-leg stance, swing the non-support leg "
                    "forward then return to double-support. Alternate."
                ),
                success_criterion=(
                    "metric > 1.0 and behavior['mean_episode_length'] > 600"
                ),
                max_iterations=6,
                parent_stage="single_leg_stance",
                reward_seed_prompt=(
                    "Keep prior terms. Add kick_cycle component that "
                    "rewards alternating swing-return detected via "
                    "root_link_pos_w oscillation. Weight 0.6."
                ),
                kg_seed_papers=[],
            ),
        ],
    )


# ── 1. Mission / Stage data model ────────────────────────────────────
def test_mission_serialization_roundtrip():
    m = Mission(
        goal="toy",
        stages=[
            Stage(
                name="s1", goal_text="do a thing",
                success_criterion="behavior['mean_return'] > 0",
                max_iterations=3, parent_stage=None,
                reward_seed_prompt="alive_bonus 0.1",
            ),
        ],
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="one-stage toy mission",
    )
    text = m.to_json()
    m2 = Mission.from_json(text)
    assert m2.goal == m.goal
    assert m2.stages[0].name == "s1"
    assert m2.schema_version == m.schema_version
    assert m2.to_dict() == m.to_dict()


def test_mission_save_load_roundtrip(tmp_path: Path):
    m = Mission(
        goal="toy", stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="claude-opus-4-7",
        decomposition_rationale="",
    )
    path = save_mission(m, tmp_path)
    assert path.name == "mission.json"
    m2 = load_mission(tmp_path)
    assert m2.goal == m.goal


def test_mission_rejects_future_schema_version():
    data = {
        "schema_version": 999,
        "goal": "x", "decomposition_model": "x",
        "decomposition_rationale": "",
        "stages": [],
    }
    with pytest.raises(MissionValidationError, match="schema_version=999"):
        Mission.from_dict(data)


def test_stage_from_dict_drops_unknown_keys():
    """Forward-compat: older readers tolerate newer schemas' extra keys."""
    s = Stage.from_dict({
        "name": "s1", "goal_text": "x", "success_criterion": "True",
        "max_iterations": 2, "parent_stage": None,
        "reward_seed_prompt": "alive",
        "some_future_field": "ignored",  # not on the dataclass
    })
    assert s.name == "s1"


# ── 1b. §start_pose — Stage field round-trip + defaults ─────────────
def test_stage_start_pose_defaults_none():
    s = Stage(
        name="s1", goal_text="x", success_criterion="True",
        max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
    )
    assert s.start_pose is None


def test_stage_start_pose_roundtrips_through_dict():
    s = Stage(
        name="s1", goal_text="x", success_criterion="True",
        max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
        start_pose="supine",
    )
    d = s.to_dict()
    assert d["start_pose"] == "supine"
    s2 = Stage.from_dict(d)
    assert s2.start_pose == "supine"


def test_stage_start_pose_older_mission_json_loads_none():
    """Back-compat: a mission.json predating start_pose loads with None
    via from_dict's filter-unknown-keys path (no `start_pose` key at
    all in the payload)."""
    s = Stage.from_dict({
        "name": "s1", "goal_text": "x", "success_criterion": "True",
        "max_iterations": 2, "parent_stage": None,
        "reward_seed_prompt": "alive",
    })
    assert s.start_pose is None


# ── 2. validate_mission — structural rules ──────────────────────────
def test_validate_rejects_empty_stages():
    m = Mission(
        goal="g", stages=[],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(MissionValidationError, match="no stages"):
        validate_mission(m, info_keys=set())


def test_validate_rejects_bad_stage_name():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="BadName",  # capitalized → invalid
                goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(MissionValidationError, match="snake_case"):
        validate_mission(m, info_keys=set())


def test_validate_rejects_duplicate_stage_names():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="a", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
            Stage(
                name="s1", goal_text="b", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(MissionValidationError, match="duplicates"):
        validate_mission(m, info_keys=set())


def test_validate_rejects_first_stage_with_parent():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage="s0",  # there is no s0
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="first.*must cold-start"
    ):
        validate_mission(m, info_keys=set())


def test_validate_rejects_forward_parent_reference():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2,
                parent_stage="s3",  # refers to a later stage
                reward_seed_prompt="alive",
            ),
            Stage(
                name="s2", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
            Stage(
                name="s3", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(MissionValidationError, match="must cold-start"):
        # First stage has a parent → triggers the first-stage rule.
        validate_mission(m, info_keys=set())


def test_validate_rejects_unknown_parent_name():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
            Stage(
                name="s2", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage="nonexistent",
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="does not name an earlier stage"
    ):
        validate_mission(m, info_keys=set())


def test_validate_rejects_unparseable_success_criterion():
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x",
                success_criterion="if foo: else",  # syntax error
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="not a valid Python expression"
    ):
        validate_mission(m, info_keys=set())


def test_validate_rejects_torch_float_method_in_criterion():
    """§Ship 21c regression: Sam's robot-flossing run failed at the
    very last criterion eval after 10+ hours of training because
    Claude's stage criterion used `(arr > 0.65).float().mean() > 0.9`.
    `.float()` is a torch tensor method; the criterion namespace is
    numpy. Now rejected at decompose time before any GPU spent.

    Tests: each torch method we forbid is rejected with a message
    pointing the user to the numpy equivalent.
    """
    cases = [
        ("(trajectory['root_link_pos_w'][..., 2] > 0.65).float().mean() > 0.9", "float"),
        ("(trajectory['joint_pos'] > 0).long().mean() > 0.5", "long"),
        ("trajectory['rewards'].cpu().mean() > 0.0", "cpu"),
        ("trajectory['action'].detach().mean() > 0.0", "detach"),
        ("metric > 0.5 and trajectory['action'].numpy().mean() < 1.0", "numpy"),
    ]
    for crit, method in cases:
        m = Mission(
            goal="g",
            stages=[
                Stage(
                    name="s1", goal_text="x",
                    success_criterion=crit,
                    max_iterations=2, parent_stage=None,
                    reward_seed_prompt="alive",
                ),
            ],
            decomposition_model="x", decomposition_rationale="",
        )
        with pytest.raises(MissionValidationError) as exc_info:
            validate_mission(m, info_keys=set())
        assert "torch tensor method" in str(exc_info.value)
        assert method in str(exc_info.value), (
            f"error should name the offending method {method!r}, got: "
            f"{exc_info.value}"
        )


def test_validate_accepts_numpy_equivalent_of_float_dot_mean():
    """§Ship 21c: the correct numpy idiom passes validation. A bool
    array's `.mean()` returns the fraction-True directly — no cast
    needed. `.astype(float)` also passes for users who want explicit
    casting."""
    for crit in [
        "(trajectory['root_link_pos_w'][..., 2] > 0.65).mean() > 0.9",
        "(trajectory['root_link_pos_w'][..., 2] > 0.65).astype(float).mean() > 0.9",
    ]:
        m = Mission(
            goal="g",
            stages=[
                Stage(
                    name="s1", goal_text="x",
                    success_criterion=crit,
                    max_iterations=2, parent_stage=None,
                    reward_seed_prompt="alive",
                ),
            ],
            decomposition_model="x", decomposition_rationale="",
        )
        # Should not raise.
        validate_mission(m, info_keys=set())


def test_validate_rejects_success_criterion_with_unknown_info_key():
    """Ship 16 namespace update: `info[<key>]` is now checked against
    PERSISTED_TRAJECTORY_KEYS (the runtime-persisted array names),
    NOT the adapter's expected_info_keys. `torso_angle` isn't in
    either set, so the rejection still fires — only the failure
    message changed."""
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x",
                success_criterion="info['torso_angle'] > 0.1",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="not in the persisted"
    ):
        validate_mission(m, info_keys={"base_height", "fallen"})


def test_validate_accepts_bare_identifier_that_could_be_component():
    """Ship 14 accepts unknown bare names — they'll be validated at
    runtime in Ship 16 against the stage's actual reward components."""
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x",
                success_criterion="kick_velocity > 0.5",  # bare name
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    # Should NOT raise — the name "kick_velocity" may be introduced by
    # this stage's reward_seed_prompt; we can't verify until runtime.
    validate_mission(m, info_keys=set())


def test_validate_rejects_max_iterations_out_of_range():
    for bad in (0, 51, -3):
        m = Mission(
            goal="g",
            stages=[
                Stage(
                    name="s1", goal_text="x", success_criterion="True",
                    max_iterations=bad, parent_stage=None,
                    reward_seed_prompt="alive",
                ),
            ],
            decomposition_model="x", decomposition_rationale="",
        )
        with pytest.raises(MissionValidationError, match="max_iterations"):
            validate_mission(m, info_keys=set())


def test_validate_rejects_reward_seed_prompt_outside_char_bounds():
    # Too short
    m = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="xx",  # 2 chars < 3 min
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="reward_seed_prompt length"
    ):
        validate_mission(m, info_keys=set())

    # Too long
    m2 = Mission(
        goal="g",
        stages=[
            Stage(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="x" * 2001,
            ),
        ],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(
        MissionValidationError, match="reward_seed_prompt length"
    ):
        validate_mission(m2, info_keys=set())


# ── 2b. §start_pose — validation + the deterministic force rule ─────
def _stage_with_start_pose(start_pose, needs_reference_rsi=False) -> Stage:
    return Stage(
        name="s1", goal_text="x", success_criterion="True",
        max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
        start_pose=start_pose, needs_reference_rsi=needs_reference_rsi,
    )


def test_validate_rejects_unknown_start_pose_value():
    m = Mission(
        goal="g", stages=[_stage_with_start_pose("levitating")],
        decomposition_model="x", decomposition_rationale="",
    )
    with pytest.raises(MissionValidationError, match="start_pose"):
        validate_mission(m, info_keys=set())


@pytest.mark.parametrize(
    "pose", ["supine", "prone", "sitting", "crouched"],
)
def test_validate_forces_needs_reference_rsi_for_non_standing_start_pose(pose):
    """The DETERMINISTIC FORCE RULE: any non-'standing' start_pose with
    needs_reference_rsi=False gets FORCED True at validation — prompt
    compliance on the separate boolean is not trusted. A warning
    discloses the force (this module's log/warn channel; see
    `_validate_stage_structure`)."""
    stage = _stage_with_start_pose(pose, needs_reference_rsi=False)
    m = Mission(
        goal="g", stages=[stage],
        decomposition_model="x", decomposition_rationale="",
    )
    assert stage.needs_reference_rsi is False
    with pytest.warns(UserWarning, match="needs_reference_rsi"):
        validate_mission(m, info_keys=set())
    assert stage.needs_reference_rsi is True


def test_validate_standing_start_pose_does_not_force_rsi():
    stage = _stage_with_start_pose("standing", needs_reference_rsi=False)
    m = Mission(
        goal="g", stages=[stage],
        decomposition_model="x", decomposition_rationale="",
    )
    validate_mission(m, info_keys=set())
    assert stage.needs_reference_rsi is False


def test_validate_none_start_pose_does_not_force_rsi():
    stage = _stage_with_start_pose(None, needs_reference_rsi=False)
    m = Mission(
        goal="g", stages=[stage],
        decomposition_model="x", decomposition_rationale="",
    )
    validate_mission(m, info_keys=set())
    assert stage.needs_reference_rsi is False


def test_validate_non_standing_start_pose_already_true_does_not_warn(
    recwarn,
):
    """No force needed (and no warning) when the stage already carries
    needs_reference_rsi=True alongside a non-standing start_pose."""
    stage = _stage_with_start_pose("crouched", needs_reference_rsi=True)
    m = Mission(
        goal="g", stages=[stage],
        decomposition_model="x", decomposition_rationale="",
    )
    validate_mission(m, info_keys=set())
    assert stage.needs_reference_rsi is True
    assert len(recwarn) == 0


# ── 3. decompose_task — happy path with stubbed Claude ───────────────
def test_decompose_happy_path_no_kg(monkeypatch):
    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick",
        _default_contract(),
        kg_store=None,
        client=client,
    )
    assert isinstance(mission, Mission)
    assert mission.goal == "Stand on one leg and kick"
    assert len(mission.stages) == 3
    assert mission.stages[0].name == "stand"
    assert mission.stages[0].parent_stage is None
    assert mission.stages[1].parent_stage == "stand"
    assert mission.stages[2].parent_stage == "single_leg_stance"
    # All stages begin pending (runtime fields unset).
    assert all(s.status == "pending" for s in mission.stages)
    assert all(s.final_policy_path is None for s in mission.stages)
    # One parse call consumed.
    assert len(client.messages.calls) == 1


def test_decompose_threads_contract_into_user_content(monkeypatch):
    client = _StubClient(_good_decomp_model())
    decompose_task(
        "x", _default_contract(), kg_store=None, client=client,
    )
    call_kwargs = client.messages.calls[0]
    user_content = call_kwargs["messages"][0]["content"]
    # The info-key list must appear so Claude knows what to cite.
    assert "base_height" in user_content
    assert "fallen" in user_content
    # The contract block header is present.
    assert "REWARD_CONTRACT" in user_content
    # Goal is echoed.
    assert "BEHAVIOR_GOAL" in user_content


def test_decompose_rejects_empty_goal():
    with pytest.raises(DecompositionError, match="goal is empty"):
        decompose_task(
            "   ", _default_contract(),
            client=_StubClient(_good_decomp_model()),
        )


# ── 4. decompose_task validates Claude's output ───────────────────────
def test_decompose_rejects_forward_parent_from_claude():
    bad = _DecompositionModel(
        decomposition_rationale="bad",
        stages=[
            _StageModel(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
            _StageModel(
                # s2's parent is s3 — forward ref, invalid.
                name="s2", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage="s3",
                reward_seed_prompt="alive",
            ),
            _StageModel(
                name="s3", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
    )
    client = _StubClient(bad)
    with pytest.raises(
        MissionValidationError, match="does not name an earlier stage"
    ):
        decompose_task(
            "x", _default_contract(), kg_store=None, client=client,
        )


def test_decompose_rejects_unknown_info_key_from_claude():
    """Ship 16: `info[<key>]` is checked against PERSISTED_TRAJECTORY_KEYS
    (the runtime-persisted trajectory.npz array names), not the adapter's
    expected_info_keys. `hip_flex_vel` isn't persisted, so the rejection
    still fires. The UX improves — error names the real set Claude
    should have drawn from."""
    bad = _DecompositionModel(
        decomposition_rationale="bad",
        stages=[
            _StageModel(
                name="s1", goal_text="x",
                # hip_flex_vel isn't in PERSISTED_TRAJECTORY_KEYS.
                success_criterion="info['hip_flex_vel'].mean() > 3.0",
                max_iterations=3, parent_stage=None,
                reward_seed_prompt="alive",
            ),
        ],
    )
    client = _StubClient(bad)
    with pytest.raises(
        MissionValidationError, match="not in the persisted"
    ):
        decompose_task(
            "x", _default_contract(), kg_store=None, client=client,
        )


def test_decompose_rejects_kg_citation_without_kg_store():
    """If Claude cites a paper but we didn't give it a KG, refuse — we
    can't verify the citation."""
    bad = _DecompositionModel(
        decomposition_rationale="bad",
        stages=[
            _StageModel(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
                kg_seed_papers=["2312.17507"],
            ),
        ],
    )
    client = _StubClient(bad)
    with pytest.raises(
        MissionValidationError,
        match="kg_seed_papers cited without a kg_store",
    ):
        decompose_task(
            "x", _default_contract(), kg_store=None, client=client,
        )


def test_decompose_rejects_kg_citation_not_in_kg(tmp_path, monkeypatch):
    """Claude cited an arxiv_id that isn't in the live KG."""
    # Build a minimal KG with ONE known paper so we can assert the
    # validator catches the OTHER (unknown) one Claude cited.
    from sculptor.kg.store import SculptorKG
    from sculptor.kg.schema import Paper, make_paper_id

    db = tmp_path / "kg.db"
    store = SculptorKG(db)
    known_paper = Paper(
        id=make_paper_id("1234.56789"),
        arxiv_id="1234.56789", title="Known paper", authors=["A"],
        year=2024, abstract="", conclusion_text="",
        ingested_at=0, extracted=True,
    )
    store.add_node(known_paper)

    # Skip the sentence-transformers semantic query (no embeddings in
    # this minimal KG). `_render_kg_context` will gracefully return an
    # empty slice.
    monkeypatch.setattr(
        "sculptor.decompose.query_semantic",
        lambda *a, **kw: [],
        raising=False,
    )
    # query_semantic is imported lazily inside _render_kg_context;
    # patch the canonical module too.
    monkeypatch.setattr(
        "sculptor.kg.query.query_semantic",
        lambda *a, **kw: [],
    )

    bad = _DecompositionModel(
        decomposition_rationale="bad",
        stages=[
            _StageModel(
                name="s1", goal_text="x", success_criterion="True",
                max_iterations=2, parent_stage=None,
                reward_seed_prompt="alive",
                kg_seed_papers=["9999.99999"],  # NOT in KG
            ),
        ],
    )
    client = _StubClient(bad)
    with pytest.raises(
        MissionValidationError, match="not in the KG"
    ):
        decompose_task(
            "x", _default_contract(), kg_store=store, client=client,
        )
    store.close()


def test_decompose_retries_once_on_parse_failure():
    """Mirrors diagnose._parse_with_retry — one retry on first-attempt error."""
    good = _good_decomp_model()
    client = _StubClient(RuntimeError("fake parse error"), good)
    mission = decompose_task(
        "x", _default_contract(), kg_store=None, client=client,
    )
    assert len(mission.stages) == 3
    # Exactly 2 parse calls (initial + retry).
    assert len(client.messages.calls) == 2
    # Retry should include a user-role message carrying the original
    # error string for Claude to self-correct.
    retry_messages = client.messages.calls[1]["messages"]
    assert len(retry_messages) == 2
    assert retry_messages[0]["role"] == "user"
    assert retry_messages[1]["role"] == "user"
    assert "fake parse error" in retry_messages[1]["content"]


# ── 5. Prompt file exists and contains the load-bearing rules ────────
def test_decompose_task_prompt_contains_hard_rules():
    """Guard against accidental prompt edits removing the structural
    rules Claude relies on to produce valid JSON."""
    from sculptor.prompts import load_prompt

    p = load_prompt("decompose_task")
    assert "parent_stage: null" in p
    assert "parent_stage" in p and "earlier" in p.lower()
    # The final-stage-must-satisfy-goal rule.
    assert "last stage" in p.lower() or "final stage" in p.lower()
    # The grounded-field rule for success_criterion.
    assert "expected_info_keys" in p
    # Strict JSON instruction at the end.
    assert "JSON" in p


def test_needs_reference_rsi_description_covers_non_standing_start():
    """§REFERENCE_TRAJECTORY_PLAN §7/§9: the flag must ALSO cover a
    non-standing start state (get-up's first stage), not just
    ballistic/airborne states — both the prompt and the pydantic field
    description."""
    from sculptor.prompts import load_prompt

    p = load_prompt("decompose_task")
    assert "lying" in p.lower()
    assert "standing reset" in p.lower()

    field_desc = _StageModel.model_fields["needs_reference_rsi"].description
    assert "lying" in field_desc.lower() or "standing reset" in field_desc.lower()


# ── §start_pose: _StageModel field + decompose parsing ────────────────
def test_stage_model_start_pose_defaults_none():
    sm = _StageModel(
        name="s1", goal_text="x", success_criterion="True",
        max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
    )
    assert sm.start_pose is None


@pytest.mark.parametrize(
    "raw,expected",
    [("Supine", "supine"), (" prone ", "prone"), ("", None), (None, None)],
)
def test_stage_model_normalizes_start_pose(raw, expected):
    sm = _StageModel(
        name="s1", goal_text="x", success_criterion="True",
        max_iterations=2, parent_stage=None, reward_seed_prompt="alive",
        start_pose=raw,
    )
    assert sm.start_pose == expected


def test_decompose_task_prompt_contains_start_pose_rule():
    """Guard against accidental prompt edits dropping the start_pose
    section — both the JSON schema block and the vocabulary."""
    from sculptor.prompts import load_prompt

    p = load_prompt("decompose_task")
    assert "start_pose" in p
    assert "supine" in p.lower() and "prone" in p.lower()
    assert "crouched" in p.lower()


def test_redecompose_prompt_contains_start_pose_rule():
    from sculptor.prompts import load_prompt

    p = load_prompt("redecompose_stage")
    assert "start_pose" in p
    assert "supine" in p.lower() and "prone" in p.lower()


def test_decompose_parses_start_pose_from_claude(monkeypatch):
    """A stage-level start_pose Claude emits flows through
    decompose_task into the resulting Mission's Stage — including the
    coherent needs_reference_rsi:true a get-up first stage requires."""
    model = _DecompositionModel(
        decomposition_rationale=(
            "One stage: rise from lying on the back to standing."
        ),
        stages=[
            _StageModel(
                name="stand_up",
                goal_text="From lying on your back, rise to standing.",
                success_criterion="behavior['mean_return'] > 0.3",
                max_iterations=4,
                parent_stage=None,
                reward_seed_prompt="alive_bonus + upright + rise_progress",
                kg_seed_papers=[],
                needs_reference_rsi=True,
                start_pose="supine",
            ),
        ],
    )
    client = _StubClient(model)
    mission = decompose_task(
        "Get up off the ground from lying on your back",
        _default_contract(), kg_store=None, client=client,
        attach_references=False,
    )
    assert mission.stages[0].start_pose == "supine"
    assert mission.stages[0].needs_reference_rsi is True


# ── 6. §REFERENCE_TRAJECTORY_PLAN §4/§7: retrieval + attachment ──────
@dataclass
class _FakeMatch:
    clip_id: str
    text: str = ""
    score: float = 1.0
    match_confidence: "float | None" = None
    reason: "str | None" = None
    tier: "str | None" = "K"
    license: "str | None" = "internal"
    n_frames: "int | None" = 50
    fps: "float | None" = 30.0
    duration_s: "float | None" = 2.0
    rerank: str = "deterministic-only"


def _touch_index(tmp_path: Path, monkeypatch) -> Path:
    """§`_retrieve_mission_references`/`_attach_stage_references` both
    early-return when no index.jsonl exists on disk — create an empty one
    under a tmp RS_REFERENCE_ROOT so the retrieval path actually runs."""
    root = tmp_path / "refs_root"
    root.mkdir()
    (root / "index.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))
    return root


def _mk_clip_for_signature():
    from sculptor.reference import make_procedural_jump_clip

    return make_procedural_jump_clip()


def test_attach_references_true_picks_top_match_above_threshold(
    tmp_path, monkeypatch,
):
    """A stage whose retrieval top match has match_confidence >= 0.8 gets
    its reference_clip_id/tier/match_confidence set."""
    _touch_index(tmp_path, monkeypatch)
    clip = _mk_clip_for_signature()
    monkeypatch.setattr(
        "sculptor.reference.load_clip", lambda path: clip)

    def fake_search(query, robot="g1", k=10, **kw):
        return [_FakeMatch(clip_id="getup_high_conf", match_confidence=0.92,
                            tier="D")]

    monkeypatch.setattr("sculptor.refs.retrieve.search", fake_search)

    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=True,
    )
    for stage in mission.stages:
        assert stage.reference_clip_id == "getup_high_conf"
        assert stage.reference_tier == "D"
        assert stage.reference_match_confidence == pytest.approx(0.92)


def test_attach_references_true_leaves_none_below_threshold(
    tmp_path, monkeypatch,
):
    """A low-confidence, non-standout match must NOT be auto-attached —
    the UI picker remains the approval surface."""
    _touch_index(tmp_path, monkeypatch)
    clip = _mk_clip_for_signature()
    monkeypatch.setattr("sculptor.reference.load_clip", lambda path: clip)

    def fake_search(query, robot="g1", k=10, **kw):
        # Two close-scoring candidates, low LLM confidence: neither the
        # confidence bar nor the "clear standout" fallback should fire.
        return [
            _FakeMatch(clip_id="maybe_a", match_confidence=0.4, score=1.0),
            _FakeMatch(clip_id="maybe_b", match_confidence=0.35, score=0.9),
        ]

    monkeypatch.setattr("sculptor.refs.retrieve.search", fake_search)

    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=True,
    )
    for stage in mission.stages:
        assert stage.reference_clip_id is None
        assert stage.reference_tier is None
        assert stage.reference_match_confidence is None


def test_attach_references_false_leaves_stages_untouched(tmp_path, monkeypatch):
    """`attach_references=False` must skip retrieval entirely — even a
    high-confidence match must not be attached, and search() must not
    even be called."""
    _touch_index(tmp_path, monkeypatch)
    called = {"search": False}

    def fake_search(query, robot="g1", k=10, **kw):
        called["search"] = True
        return [_FakeMatch(clip_id="should_not_attach", match_confidence=0.99)]

    monkeypatch.setattr("sculptor.refs.retrieve.search", fake_search)

    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=False,
    )
    assert called["search"] is False
    for stage in mission.stages:
        assert stage.reference_clip_id is None


def test_attach_references_injects_signature_into_decompose_prompt(
    tmp_path, monkeypatch,
):
    """§7: the mission-goal-level retrieval + signature injection happens
    in the SAME decompose call (decompose has no separate criterion-
    authoring call) — assert the rendered user_content carries the
    REFERENCE MOTION SIGNATURES block with real numeric fields."""
    _touch_index(tmp_path, monkeypatch)
    clip = _mk_clip_for_signature()
    monkeypatch.setattr("sculptor.reference.load_clip", lambda path: clip)

    def fake_search(query, robot="g1", k=10, **kw):
        return [_FakeMatch(clip_id="jump_ref_clip", match_confidence=0.9)]

    monkeypatch.setattr("sculptor.refs.retrieve.search", fake_search)

    client = _StubClient(_good_decomp_model())
    decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=True,
    )
    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "REFERENCE MOTION SIGNATURES" in user_content
    assert "jump_ref_clip" in user_content
    assert "duration_s" in user_content
    assert "ground" in user_content.lower()


def test_attach_references_no_index_on_disk_is_a_noop(tmp_path, monkeypatch):
    """No `index.jsonl` on disk at all (fresh RS_REFERENCE_ROOT, never
    ingested) must never raise and must never attach anything — the
    §build-brief guard on `library.index_path().is_file()`."""
    root = tmp_path / "empty_refs_root"
    root.mkdir()
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(root))

    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=True,
    )
    for stage in mission.stages:
        assert stage.reference_clip_id is None


def test_attach_references_retrieval_exception_never_raises(tmp_path, monkeypatch):
    """Any retrieval-layer exception (index corrupt, search() bug, etc.)
    must degrade to "no references" rather than fail decompose_task."""
    _touch_index(tmp_path, monkeypatch)

    def broken_search(query, robot="g1", k=10, **kw):
        raise RuntimeError("simulated retrieval failure")

    monkeypatch.setattr("sculptor.refs.retrieve.search", broken_search)

    client = _StubClient(_good_decomp_model())
    mission = decompose_task(
        "Stand on one leg and kick", _default_contract(),
        kg_store=None, client=client, attach_references=True,
    )
    for stage in mission.stages:
        assert stage.reference_clip_id is None
