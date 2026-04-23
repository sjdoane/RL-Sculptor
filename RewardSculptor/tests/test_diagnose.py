"""tests/test_diagnose.py — offline tests for the two-stage diagnoser.

No API calls. We mock Anthropic's `messages.parse` to return fixed Pydantic
models that match the expected schema, and verify the wiring:
  - iter_dir artifacts are read
  - config.iteration.behavior_metrics is surfaced into the prompt
  - KG retrieval runs with `domain_filter = config.kg.environment_tag`
  - failure_modes propagate into the Diagnosis
  - proposed_edits carry paper_refs
  - diagnosis.json is written to iter_dir
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest

from sculptor.diagnose import (
    Diagnosis,
    ProposedEdit,
    _GroundedModel,
    _PreliminaryModel,
    _ProposedEditModel,
    diagnose,
)
from sculptor.kg.schema import (
    Edge,
    Environment,
    FailureMode,
    Paper,
    Relation,
    Technique,
    make_environment_id,
    make_failure_mode_id,
    make_paper_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG


# ── Test fixtures ──────────────────────────────────────────────────────────
def _png_1x1() -> bytes:
    """Minimal valid 1x1 PNG (no deps)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xFF\x00\x00"  # one row: filter byte + single red pixel
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture
def iter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "iter_000"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({
        "metrics": {"mean_return": 71.1, "std_return": 0.8,
                    "n_eval_episodes": 5, "training_steps": 20000,
                    "n_envs": 4, "seed": 42},
        "components": {"alive_bonus": 1.0,
                       "forward_velocity": 0.12,
                       "ctrl_cost": -0.0015}
    }, sort_keys=True))
    (d / "behavior.json").write_text(json.dumps({
        "n_episodes": 6,
        "mean_return": 70.5,
        "max_episode_length": 43,
        "mean_episode_length": 42.0,
        "fall_rate": 1.0,
        "mean_forward_velocity": 0.68,
        "termination_reason_counts": {"terminated": 6, "truncated": 0}
    }, sort_keys=True))
    (d / "reward_spec.json").write_text(json.dumps({
        "version": "v0",
        "author": "human",
        "parent_hash": None,
        "description": "Canonical Hopper-v4 reward.",
        "hyperparameters": {"forward_weight": 1.0, "alive_bonus": 1.0,
                            "ctrl_cost_weight": 0.001},
        "references": []
    }, sort_keys=True))
    kdir = d / "keyframes"
    kdir.mkdir()
    png = _png_1x1()
    for i in range(4):
        (kdir / f"frame_{i:02d}.png").write_bytes(png)
    return d


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """Minimal TOML pointing at the real Hopper GymSB3Adapter so
    `reward_contract()` returns real spaces."""
    p = tmp_path / "config.toml"
    p.write_text(
        '[target]\n'
        'name = "test_hopper"\n\n'
        '[adapter]\n'
        'class = "sculptor.adapters.gym_sb3.GymSB3Adapter"\n'
        'config = { env_id = "Hopper-v4", n_envs = 1 }\n\n'
        '[kg]\n'
        'environment_tag = "continuous_locomotion"\n\n'
        '[iteration]\n'
        'steps_per_iter = 50000\n'
        'primary_metric = "mean_return"\n'
        'behavior_metrics = ["max_episode_length", "mean_forward_velocity", '
        '"fall_rate"]\n'
    )
    return p


@pytest.fixture
def kg_with_locomotion_tech(tmp_path: Path) -> SculptorKG:
    store = SculptorKG(tmp_path / "kg.db")
    paper = Paper(
        id=make_paper_id("1801.00690"), arxiv_id="1801.00690",
        title="DeepMind Control Suite", authors=["Yuval Tassa"], year=2018,
    )
    store.add_node(paper)
    tech = Technique(
        id=make_technique_id("tolerance_reward_kernel"),
        name="tolerance_reward_kernel",
        description="Smooth tolerance-based reward components that densify sparse signals.",
        tags=["dense_reward", "reward_shaping"],
    )
    store.add_node(tech)
    fm_sparse = FailureMode(
        id=make_failure_mode_id("sparse_reward"),
        name="sparse_reward",
        description="Reward signal is almost always zero.",
    )
    store.add_node(fm_sparse)
    env = Environment(
        id=make_environment_id("Hopper-v4"),
        name="Hopper-v4",
        description="MuJoCo Hopper",
        tags=["continuous_locomotion", "mujoco"],
    )
    store.add_node(env)
    store.add_edge(Edge(src=paper.id, dst=tech.id,
                        relation=Relation.INTRODUCES,
                        data={"evidence": "The suite introduces tolerance rewards."}))
    store.add_edge(Edge(src=tech.id, dst=fm_sparse.id,
                        relation=Relation.ADDRESSES,
                        data={"evidence": "Tolerance kernels densify sparse rewards.",
                              "source_paper_id": paper.id}))
    store.add_edge(Edge(src=paper.id, dst=env.id,
                        relation=Relation.EVALUATES_ON,
                        data={"evidence": "Evaluated on Hopper-v4."}))
    return store


# ── Stub anthropic client ──────────────────────────────────────────────────
class _StubResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, preliminary, grounded):
        self._preliminary = preliminary
        self._grounded = grounded
        self._calls = 0
        self.captured_prompts: list[dict] = []

    def parse(self, **kwargs):
        self.captured_prompts.append(kwargs)
        self._calls += 1
        cls = kwargs["output_format"]
        if cls is _PreliminaryModel:
            return _StubResponse(self._preliminary)
        if cls is _GroundedModel:
            return _StubResponse(self._grounded)
        raise AssertionError(f"unexpected output_format: {cls}")


class _StubClient:
    def __init__(self, preliminary, grounded):
        self.messages = _StubMessages(preliminary, grounded)


# ── Tests ──────────────────────────────────────────────────────────────────
def test_diagnose_happy_path(iter_dir, config, kg_with_locomotion_tech, monkeypatch):
    # Skip the heavy sentence-transformers load in query_semantic.
    monkeypatch.setattr(
        "sculptor.diagnose.query_semantic",
        lambda *a, **kw: [],
    )

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward", "premature_termination"],
        evidence="Episodes end after ~42 steps with fall_rate=1.0 and "
                 "mean_forward_velocity=0.68, indicating the hopper falls "
                 "before sustaining forward motion.",
        confidence=0.82,
    )
    grounded = _GroundedModel(
        proposed_edits=[
            _ProposedEditModel(
                target_term="forward_weight",
                operation="decrease",
                rationale="Down-weight raw velocity until posture terms shape stance.",
                suggested_value="0.6",
                paper_refs=["1801.00690"],
            ),
            _ProposedEditModel(
                target_term="tolerance_torso_upright",
                operation="add",
                rationale="novel. Add a tolerance kernel on the torso angle.",
                suggested_value="tolerance(torso_angle, [-0.2, 0.2], margin=0.5)",
                paper_refs=[],
            ),
        ],
        confidence=0.77,
    )
    client = _StubClient(prelim, grounded)

    d = diagnose(
        iter_dir=iter_dir, behavior_goal="run forward as fast as possible without falling",
        config=config, store=kg_with_locomotion_tech, client=client,
    )

    # Shape
    assert isinstance(d, Diagnosis)
    assert d.failure_modes == ["sparse_reward", "premature_termination"]
    assert "fall_rate" in d.evidence
    assert d.confidence == pytest.approx(0.77)
    assert d.iter_dir == str(iter_dir)
    assert d.behavior_goal == "run forward as fast as possible without falling"

    # Edits
    assert len(d.proposed_edits) == 2
    grounded_edit = d.proposed_edits[0]
    assert isinstance(grounded_edit, ProposedEdit)
    assert grounded_edit.target_term == "forward_weight"
    assert grounded_edit.paper_refs == ["1801.00690"]

    novel_edit = d.proposed_edits[1]
    assert novel_edit.paper_refs == []
    assert novel_edit.rationale.lower().startswith("novel.")

    # KG context: should include the tolerance_reward_kernel technique because
    # sparse_reward is in the failure modes.
    tech_names = [m.technique.name for m in d.literature_context]
    assert "tolerance_reward_kernel" in tech_names

    # Two LLM calls (preliminary + grounded).
    assert client.messages._calls == 2

    # First call was the preliminary: 4 images + 2 text blocks in content.
    prelim_call = client.messages.captured_prompts[0]
    content = prelim_call["messages"][0]["content"]
    assert isinstance(content, list)
    images = [c for c in content if c.get("type") == "image"]
    assert len(images) == 4, "expected 4 keyframes"
    text_blocks = [c for c in content if c.get("type") == "text"]
    text_joined = "\n".join(t["text"] for t in text_blocks)
    # Behavior-metric vocab from config flows into the prompt.
    assert "max_episode_length" in text_joined
    assert "mean_forward_velocity" in text_joined
    assert "fall_rate" in text_joined
    # reward_contract expected_info_keys is shown.
    assert "expected_info_keys" in text_joined
    assert "x_velocity" in text_joined
    # Behavior goal is shown.
    assert "run forward as fast as possible without falling" in text_joined

    # Second call included the KG context.
    grounded_user = client.messages.captured_prompts[1]["messages"][0]["content"]
    assert "LITERATURE CONTEXT" in grounded_user
    assert "tolerance_reward_kernel" in grounded_user
    assert "PRELIMINARY DIAGNOSIS" in grounded_user

    # diagnosis.json written.
    out_path = iter_dir / "diagnosis.json"
    assert out_path.is_file()
    dumped = json.loads(out_path.read_text())
    assert dumped["failure_modes"] == ["sparse_reward", "premature_termination"]
    assert any(e["paper_refs"] == ["1801.00690"] for e in dumped["proposed_edits"])


def test_diagnose_uses_domain_filter_from_config(iter_dir, config, tmp_path, monkeypatch):
    """Papers whose environment doesn't match `config.kg.environment_tag` are filtered out."""
    store = SculptorKG(tmp_path / "kg2.db")
    # Paper + technique + failure_mode + env, but env tag is NOT
    # continuous_locomotion — domain filter should drop the technique.
    paper = Paper(id=make_paper_id("0000.00000"), arxiv_id="0000.00000",
                  title="Unrelated", year=2020)
    tech = Technique(id=make_technique_id("unrelated_technique"),
                     name="unrelated_technique",
                     description="A technique from a different domain.")
    fm = FailureMode(id=make_failure_mode_id("sparse_reward"),
                     name="sparse_reward",
                     description="Sparse reward")
    env = Environment(id=make_environment_id("Atari"), name="Atari",
                      description="Atari 2600", tags=["discrete_control"])
    for node in (paper, tech, fm, env):
        store.add_node(node)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES))
    store.add_edge(Edge(src=tech.id, dst=fm.id, relation=Relation.ADDRESSES,
                        data={"source_paper_id": paper.id}))
    store.add_edge(Edge(src=paper.id, dst=env.id, relation=Relation.EVALUATES_ON))

    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="synthetic", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d = diagnose(iter_dir=iter_dir, behavior_goal="goal",
                 config=config, store=store, client=client)
    # The unrelated technique should be filtered out by the
    # `continuous_locomotion` domain tag from config.
    assert d.literature_context == []


# ── §7.2: Eureka reward-reflection block ──────────────────────────────────
def test_format_training_feedback_renders_eureka_lines() -> None:
    from sculptor.diagnose import _format_training_feedback

    data = {
        "alive": [1.0, 1.0, 1.0],           # dead component (Max == Min)
        "forward": [0.1, 0.3, 0.5, 0.6],    # rising
        "__episode_length": [10.0, 20.0],   # aux stripped prefix
    }
    out = _format_training_feedback(data)
    lines = out.splitlines()
    # One line per key, `__`-prefix stripped.
    assert any(line.startswith("alive:") for line in lines)
    assert any(line.startswith("forward:") for line in lines)
    assert any(line.startswith("episode_length:") for line in lines)
    # Eureka's summary fields present.
    for line in lines:
        assert "Max:" in line and "Mean:" in line and "Min:" in line
    # Alive's Max == Min — the "dead" signature.
    alive_line = next(line for line in lines if line.startswith("alive:"))
    assert "Max: 1.00" in alive_line and "Min: 1.00" in alive_line


def test_format_training_feedback_caps_at_10_samples() -> None:
    """Long series (>10 pts) must be sampled to ≤10 for prompt-size
    bounding, but Max/Mean/Min are computed over the FULL series."""
    from sculptor.diagnose import _format_training_feedback

    # 100-point linear ramp from 0..99.
    data = {"x": [float(i) for i in range(100)]}
    out = _format_training_feedback(data)
    line = out.splitlines()[0]
    # Summary stats from the full list.
    assert "Max: 99.00" in line
    assert "Min: 0.00" in line
    # Values list inside [...] — count commas to get sample count ≤10.
    import re
    m = re.search(r"\[([^\]]+)\]", line)
    assert m is not None
    sample_count = len(m.group(1).split(","))
    assert sample_count <= 10, f"capped list too long: {sample_count}"


def test_format_training_feedback_empty_returns_empty_string() -> None:
    from sculptor.diagnose import _format_training_feedback
    assert _format_training_feedback({}) == ""
    assert _format_training_feedback({"x": []}) == ""
    # Dict with only non-list values → no output.
    assert _format_training_feedback({"x": "not a list"}) == ""


def test_diagnose_injects_training_feedback_block(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """When `<iter_dir>/reward_trajectory.json` is present, both prompts
    must contain a `# TRAINING_FEEDBACK` block with the component lines."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    # §7.1-style reward trajectory file.
    (iter_dir / "reward_trajectory.json").write_text(json.dumps({
        "alive_bonus": [1.0, 1.0, 1.0],      # dead
        "forward_velocity": [0.1, 0.3, 0.5], # rising
        "__episode_length": [10.0, 20.0, 35.0],
    }))

    prelim = _PreliminaryModel(
        failure_modes=["component_imbalance"], evidence="stub", confidence=0.8)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.7)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="test",
             config=config, store=kg_with_locomotion_tech, client=client)

    # Both prompts carry the block.
    prelim_call = client.messages.captured_prompts[0]
    prelim_text = "\n".join(
        c["text"] for c in prelim_call["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# TRAINING_FEEDBACK" in prelim_text
    assert "alive_bonus:" in prelim_text
    assert "forward_velocity:" in prelim_text
    assert "episode_length:" in prelim_text  # __-prefix stripped

    grounded_call = client.messages.captured_prompts[1]
    grounded_text = grounded_call["messages"][0]["content"]
    assert "# TRAINING_FEEDBACK" in grounded_text
    assert "alive_bonus:" in grounded_text


def test_diagnose_omits_training_feedback_block_when_file_missing(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """No reward_trajectory.json (pre-§7.1 iters, non-sculpted runs) must
    NOT inject an empty block — the header `# TRAINING_FEEDBACK` should be
    absent so the prompt shape matches the pre-§7.2 behavior exactly."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    # Pointedly do NOT write reward_trajectory.json.

    prelim = _PreliminaryModel(
        failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="test",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_call = client.messages.captured_prompts[0]
    prelim_text = "\n".join(
        c["text"] for c in prelim_call["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# TRAINING_FEEDBACK" not in prelim_text


def test_diagnose_injects_realism_audit_block_when_severe(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """A severe realism verdict must produce a `# PHYSICS_REALISM_AUDIT`
    block in both preliminary and grounded prompts, citing top joints."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    (iter_dir / "realism_audit.json").write_text(json.dumps({
        "verdict": "severe",
        "torque_saturation_frac": 0.47,
        "any_joint_saturation_max": 0.92,
        "joint_vel_p99_max": 85.0,
        "joint_vel_multiplier_vs_nominal": 2.83,
        "joint_limit_violation_frac": 0.01,
        "top_joints_saturation": [
            {"name": "knee_pitch_left", "value": 0.92},
            {"name": "ankle_roll_right", "value": 0.74},
        ],
        "top_joints_vel": [{"name": "knee_pitch_left", "value": 85.0}],
        "top_joints_limit_violation": [],
        "n_actuators": 23, "n_joints": 29, "n_steps": 500,
    }))

    prelim = _PreliminaryModel(
        failure_modes=["reward_hacking"], evidence="stub", confidence=0.9)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.8)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="test",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# PHYSICS_REALISM_AUDIT" in prelim_text
    assert "SEVERE" in prelim_text
    assert "knee_pitch_left" in prelim_text
    # Overall saturation surfaced.
    assert "0.47" in prelim_text
    grounded_text = client.messages.captured_prompts[1]["messages"][0]["content"]
    assert "# PHYSICS_REALISM_AUDIT" in grounded_text
    assert "SEVERE" in grounded_text


def test_diagnose_omits_realism_block_when_verdict_ok(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """An `ok` verdict is uninformative — must NOT dilute the prompt."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    (iter_dir / "realism_audit.json").write_text(json.dumps({
        "verdict": "ok", "torque_saturation_frac": 0.02,
        "any_joint_saturation_max": 0.03, "joint_vel_p99_max": 5.0,
        "joint_limit_violation_frac": 0.0,
        "top_joints_saturation": [], "top_joints_vel": [],
        "top_joints_limit_violation": [],
        "n_actuators": 12, "n_joints": 18, "n_steps": 500,
    }))

    prelim = _PreliminaryModel(
        failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="t",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# PHYSICS_REALISM_AUDIT" not in prelim_text


def test_diagnose_omits_realism_block_when_file_missing(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    # Deliberately no realism_audit.json.

    prelim = _PreliminaryModel(
        failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="t",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# PHYSICS_REALISM_AUDIT" not in prelim_text


def test_diagnose_prefers_training_side_trajectory_over_rollout_side(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """When BOTH training-side and rollout-side files exist, the training
    one wins (per `_load_training_feedback` precedence)."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    (iter_dir / "reward_trajectory.json").write_text(json.dumps({
        "from_training": [1.0, 2.0, 3.0],
    }))
    rollout = iter_dir / "rollout"
    rollout.mkdir(exist_ok=True)
    (rollout / "reward_trajectory.json").write_text(json.dumps({
        "from_rollout": [9.0, 9.0, 9.0],
    }))

    prelim = _PreliminaryModel(
        failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="t",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "from_training:" in prelim_text
    assert "from_rollout:" not in prelim_text
