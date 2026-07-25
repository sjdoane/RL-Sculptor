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
    KG_TOP_K,
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


# ── §reference-grounded diagnose: REFERENCE MOTION SIGNATURE block ────────
def _write_reference_signature(stage_dir: Path, **overrides) -> None:
    payload = {
        "schema": 1,
        "clip_id": "g1_jump_ref_01",
        "robot": "g1",
        "tier": "K",
        "text": "Reference standing long jump.",
        "signature": {
            "duration_s": 1.8,
            "fps": 30.0,
            "n_frames": 54,
            "root_z": {
                "start": 0.1, "end": 0.72, "min": 0.1, "min_t": 0.0,
                "max": 0.72, "max_t": 1.5,
            },
            "root_velocity_mps": {"min": -0.2, "max": 1.4},
            "phases": [
                {"phase": "rising", "t_start": 0.0, "t_end": 1.5,
                 "z_start": 0.1, "z_end": 0.72},
            ],
        },
    }
    payload.update(overrides)
    (stage_dir / "reference_signature.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_diagnose_injects_reference_signature_block(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """When `<stage_dir>/reference_signature.json` is present (stage_dir =
    config.parent), both preliminary and grounded prompts must carry a
    `# REFERENCE MOTION SIGNATURE` block with the real clip numbers."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    _write_reference_signature(config.parent)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="stub", confidence=0.8)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.7)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="test",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# REFERENCE MOTION SIGNATURE" in prelim_text
    assert "g1_jump_ref_01" in prelim_text
    assert "0.72" in prelim_text  # root_z.max flows through verbatim

    grounded_text = client.messages.captured_prompts[1]["messages"][0]["content"]
    assert "# REFERENCE MOTION SIGNATURE" in grounded_text
    assert "g1_jump_ref_01" in grounded_text


def test_diagnose_omits_reference_signature_block_when_file_missing(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """No reference_signature.json (plain runs, no reference attached) —
    the block must be absent, prompt shape unchanged from before."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    # Deliberately do NOT write reference_signature.json.

    prelim = _PreliminaryModel(failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="test",
             config=config, store=kg_with_locomotion_tech, client=client)

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# REFERENCE MOTION SIGNATURE" not in prelim_text
    grounded_text = client.messages.captured_prompts[1]["messages"][0]["content"]
    assert "# REFERENCE MOTION SIGNATURE" not in grounded_text


def test_diagnose_omits_reference_signature_block_when_corrupt(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """A corrupt/wrong-schema reference_signature.json must silently
    no-op — never crash diagnose(), never inject a partial block."""
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    (config.parent / "reference_signature.json").write_text(
        "{not valid json", encoding="utf-8")

    prelim = _PreliminaryModel(failure_modes=[], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d = diagnose(iter_dir=iter_dir, behavior_goal="test",
                 config=config, store=kg_with_locomotion_tech, client=client)
    assert d is not None

    prelim_text = "\n".join(
        c["text"] for c in client.messages.captured_prompts[0]["messages"][0]["content"]
        if c.get("type") == "text"
    )
    assert "# REFERENCE MOTION SIGNATURE" not in prelim_text


# ── KG-retrieval fix 1: evidence-anchored semantic retrieval ──────────────
def test_diagnose_evidence_anchored_query_merges_first_and_logs(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """When preliminary evidence is present, a SECOND semantic query
    anchored on that evidence text must run, its hits must be merged
    BEFORE the tag/goal hits (deduped, capped at KG_TOP_K), and it must
    be logged under decision="diagnose_evidence"."""
    from sculptor.kg.query import TechniqueMatch
    from sculptor.kg.schema import Technique

    calls: list[str] = []

    def _stub_query_semantic(text, top_k=6, store=None, min_similarity=0.0):
        calls.append(text)
        if "leg drive" in text:
            return [TechniqueMatch(
                technique=Technique(id="technique:evidence_hit", name="evidence_hit"),
                description="d", paper_citation="c", evidence="e",
                relevance_score=0.9)]
        return [TechniqueMatch(
            technique=Technique(id="technique:goal_hit", name="goal_hit"),
            description="d", paper_citation="c", evidence="e",
            relevance_score=0.5)]

    monkeypatch.setattr("sculptor.diagnose.query_semantic", _stub_query_semantic)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"],
        evidence="The hopper planks on forearms without leg drive, failing "
                  "to push off before falling.",
        confidence=0.7,
    )
    grounded = _GroundedModel(proposed_edits=[], confidence=0.6)
    client = _StubClient(prelim, grounded)

    d = diagnose(iter_dir=iter_dir, behavior_goal="run forward",
                 config=config, store=kg_with_locomotion_tech, client=client)

    # Both the goal query and the evidence query ran.
    assert "run forward" in calls
    assert any("leg drive" in c for c in calls)

    names = [m.technique.name for m in d.literature_context]
    assert "evidence_hit" in names
    assert "goal_hit" in names
    assert names.index("evidence_hit") < names.index("goal_hit")
    assert len(d.literature_context) <= KG_TOP_K

    log_path = iter_dir / "kg_retrievals.jsonl"
    assert log_path.is_file()
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    decisions = [r["decision"] for r in records]
    assert "diagnose_evidence" in decisions
    ev_rec = next(r for r in records if r["decision"] == "diagnose_evidence")
    assert "leg drive" in ev_rec["query"]
    assert "technique:evidence_hit" in ev_rec["node_ids"]


def test_diagnose_no_evidence_skips_evidence_query_byte_identical(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """Empty/whitespace-only preliminary evidence must NOT trigger the
    evidence-anchored query or its log record — behavior stays identical
    to before this fix (single semantic call, on the goal text only)."""
    calls: list[str] = []

    def _stub_query_semantic(text, top_k=6, store=None, min_similarity=0.0):
        calls.append(text)
        return []

    monkeypatch.setattr("sculptor.diagnose.query_semantic", _stub_query_semantic)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="   ", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="run forward",
             config=config, store=kg_with_locomotion_tech, client=client)

    # Exactly one semantic call — the static goal query — was made.
    assert calls == ["run forward"]

    log_path = iter_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    decisions = [r["decision"] for r in records]
    assert "diagnose_evidence" not in decisions


# ── KG-retrieval fix 2: free-text failure_descriptors ──────────────────────
def test_preliminary_model_failure_descriptors_absent_defaults_empty():
    p = _PreliminaryModel(failure_modes=["sparse_reward"], evidence="e", confidence=0.5)
    assert p.failure_descriptors == []


def test_preliminary_model_failure_descriptors_round_trip():
    p = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="e", confidence=0.5,
        failure_descriptors=["planks on forearms without leg drive", "no hip extension"])
    assert p.failure_descriptors == [
        "planks on forearms without leg drive", "no hip extension"]


def test_preliminary_model_failure_descriptors_malformed_coerces_to_empty():
    """Old cached/replayed preliminary responses predate this field —
    malformed input must coerce to `[]`, never raise."""
    p = _PreliminaryModel(failure_modes=[], evidence="e", confidence=0.5,
                          failure_descriptors="not a list")
    assert p.failure_descriptors == []

    p2 = _PreliminaryModel(failure_modes=[], evidence="e", confidence=0.5,
                           failure_descriptors=[123, None, "", "   ", "ok phrase"])
    assert p2.failure_descriptors == ["123", "ok phrase"]


def test_diagnose_uses_descriptor_resolved_failure_modes_for_tag_query(
    iter_dir, config, tmp_path, monkeypatch,
):
    """A failure_descriptor must pull in an ADDITIONAL technique via
    query_techniques's extra_failure_node_ids, even when that technique's
    FailureMode isn't reachable through the enum-resolved failure_modes."""
    from sculptor.kg.schema import (
        Edge, Environment, FailureMode, Paper, Relation, Technique,
        make_environment_id, make_failure_mode_id, make_paper_id, make_technique_id,
    )
    from sculptor.kg.store import SculptorKG

    store = SculptorKG(tmp_path / "kg_descriptors.db")
    paper = Paper(id=make_paper_id("2099.00001"), arxiv_id="2099.00001",
                  title="Descriptor Paper", year=2022)
    store.add_node(paper)
    # A FailureMode NOT among the fixed six — only reachable via a
    # descriptor-resolved node id, never via `_resolve_failure_modes`.
    fm_niche = FailureMode(
        id=make_failure_mode_id("forearm_planking"),
        name="forearm_planking",
        description="Robot supports weight on forearms instead of extending legs.")
    store.add_node(fm_niche)
    tech = Technique(
        id=make_technique_id("leg_extension_bonus"),
        name="leg_extension_bonus",
        description="Reward term rewarding knee/hip extension at touchdown.")
    store.add_node(tech)
    env = Environment(id=make_environment_id("Hopper-v4"), name="Hopper-v4",
                      description="MuJoCo Hopper", tags=["continuous_locomotion"])
    store.add_node(env)
    store.add_edge(Edge(src=paper.id, dst=tech.id, relation=Relation.INTRODUCES,
                        data={"evidence": "introduces leg extension bonus"}))
    store.add_edge(Edge(src=tech.id, dst=fm_niche.id, relation=Relation.ADDRESSES,
                        data={"evidence": "fixes forearm planking",
                              "source_paper_id": paper.id}))
    store.add_edge(Edge(src=paper.id, dst=env.id, relation=Relation.EVALUATES_ON))

    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    monkeypatch.setattr(
        "sculptor.kg.query.resolve_failure_modes_semantic",
        lambda store, descriptors, **kw: [fm_niche.id],
    )

    prelim = _PreliminaryModel(
        failure_modes=["premature_termination"],  # does NOT resolve to fm_niche
        evidence="planks on forearms instead of extending the legs",
        failure_descriptors=["planks on forearms without leg drive"],
        confidence=0.6,
    )
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d = diagnose(iter_dir=iter_dir, behavior_goal="jump forward",
                 config=config, store=store, client=client)

    names = [m.technique.name for m in d.literature_context]
    assert "leg_extension_bonus" in names

    log_path = iter_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    desc_rec = next(r for r in records if r["decision"] == "diagnose_descriptors")
    assert "planks on forearms without leg drive" in desc_rec["query"]
    assert fm_niche.id in desc_rec["node_ids"]


def test_diagnose_no_descriptors_skips_descriptor_resolution(
    iter_dir, config, kg_with_locomotion_tech, monkeypatch,
):
    """No failure_descriptors → resolve_failure_modes_semantic is never
    called and no `diagnose_descriptors` log record is written."""
    called = []
    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])
    monkeypatch.setattr(
        "sculptor.kg.query.resolve_failure_modes_semantic",
        lambda *a, **kw: called.append(1) or [],
    )

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="stub", confidence=0.5)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    diagnose(iter_dir=iter_dir, behavior_goal="t",
             config=config, store=kg_with_locomotion_tech, client=client)

    assert called == []
    log_path = iter_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert "diagnose_descriptors" not in [r["decision"] for r in records]


# ── KG-retrieval fix 5: citation grounding annotation ──────────────────────
def test_diagnose_grounded_flag_reflects_this_iters_retrieval(
    iter_dir, config, tmp_path, monkeypatch,
):
    """paper_refs_grounded[aid] is True iff `aid` was among the papers
    THIS iteration's literature_context actually retrieved; existing-
    but-unretrieved refs are kept with grounded=False; nonexistent refs
    are still dropped entirely (pre-existing behavior, unchanged)."""
    from sculptor.kg.schema import (
        Edge, Environment, FailureMode, Paper, Relation, Technique,
        make_environment_id, make_failure_mode_id, make_paper_id, make_technique_id,
    )
    from sculptor.kg.store import SculptorKG

    store = SculptorKG(tmp_path / "kg_grounding.db")
    paper_retrieved = Paper(id=make_paper_id("1111.11111"), arxiv_id="1111.11111",
                            title="Retrieved Paper", year=2019)
    paper_unretrieved = Paper(id=make_paper_id("2222.22222"), arxiv_id="2222.22222",
                              title="Unretrieved Paper", year=2020)
    store.add_node(paper_retrieved)
    store.add_node(paper_unretrieved)

    tech = Technique(id=make_technique_id("tag_tech"), name="tag_tech",
                     description="Addresses sparse reward.")
    store.add_node(tech)
    fm = FailureMode(id=make_failure_mode_id("sparse_reward"), name="sparse_reward",
                     description="Sparse reward")
    store.add_node(fm)
    env = Environment(id=make_environment_id("Hopper-v4"), name="Hopper-v4",
                      description="Hopper", tags=["continuous_locomotion"])
    store.add_node(env)
    store.add_edge(Edge(src=paper_retrieved.id, dst=tech.id, relation=Relation.INTRODUCES))
    store.add_edge(Edge(src=tech.id, dst=fm.id, relation=Relation.ADDRESSES,
                        data={"source_paper_id": paper_retrieved.id}))
    store.add_edge(Edge(src=paper_retrieved.id, dst=env.id, relation=Relation.EVALUATES_ON))

    monkeypatch.setattr("sculptor.diagnose.query_semantic", lambda *a, **kw: [])

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="stub", confidence=0.6)
    grounded = _GroundedModel(
        proposed_edits=[
            _ProposedEditModel(
                target_term="forward_weight", operation="decrease",
                rationale="grounded ref", suggested_value="0.5",
                paper_refs=["1111.11111"]),
            _ProposedEditModel(
                target_term="ctrl_cost_weight", operation="increase",
                rationale="recalled ref, not shown this iter",
                suggested_value="0.01", paper_refs=["2222.22222"]),
            _ProposedEditModel(
                target_term="alive_bonus", operation="increase",
                rationale="fabricated ref", suggested_value="1.0",
                paper_refs=["9999.99999"]),
        ],
        confidence=0.6,
    )
    client = _StubClient(prelim, grounded)

    d = diagnose(iter_dir=iter_dir, behavior_goal="run",
                 config=config, store=store, client=client)

    by_term = {e.target_term: e for e in d.proposed_edits}
    assert by_term["forward_weight"].paper_refs == ["1111.11111"]
    assert by_term["forward_weight"].paper_refs_grounded == {"1111.11111": True}

    assert by_term["ctrl_cost_weight"].paper_refs == ["2222.22222"]
    assert by_term["ctrl_cost_weight"].paper_refs_grounded == {"2222.22222": False}

    assert by_term["alive_bonus"].paper_refs == []
    assert by_term["alive_bonus"].paper_refs_grounded == {}

    dumped = json.loads((iter_dir / "diagnosis.json").read_text())
    assert dumped["literature_context"], "expected at least one retrieved technique"
    assert all(m["grounded"] is True for m in dumped["literature_context"])
    edit_dump = next(e for e in dumped["proposed_edits"] if e["target_term"] == "forward_weight")
    assert edit_dump["paper_refs_grounded"] == {"1111.11111": True}


# ── Fix 3: staleness rotation ───────────────────────────────────────────────
def _build_iter_dir(base: Path, name: str) -> Path:
    """Same artifact shape as the `iter_dir` fixture above, parameterized by
    directory name so a test can build an `iter_<N-1>` / `iter_<N>` pair."""
    d = base / name
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


def _stale_pool(names: list[str]):
    from sculptor.kg.query import TechniqueMatch
    from sculptor.kg.schema import Technique

    return [
        TechniqueMatch(
            technique=Technique(id=f"technique:{name}", name=name),
            description="d", paper_citation="c", evidence="e",
            relevance_score=1.0 - i * 0.01,
        )
        for i, name in enumerate(names)
    ]


_STALE_POOL_NAMES = [
    "tech_a", "tech_b", "tech_c", "tech_d", "tech_e", "tech_f", "tech_g", "tech_h",
]


def _write_prev_diagnosis(
    prev_dir: Path, *, shown: list[tuple[str, str]], cited_arxiv_ids: list[str],
) -> None:
    """`shown`: [(technique_name, source_arxiv_id), ...] — mirrors the real
    `technique_id`/`source_paper_ids` fields Diagnosis.to_dict() persists.

    Uses the raw `technique:{name}` id form (matching `_stale_pool`'s
    hand-built `Technique(id=f"technique:{name}", ...)`) rather than
    `make_technique_id`, which additionally slugifies (e.g. "tech_a" ->
    "technique:tech-a") — real ingest-populated KG techniques go through
    that slugification, but this fixture bypasses ingest entirely, so its
    ids must match what it actually assigns the stub technique objects."""
    from sculptor.kg.schema import make_paper_id

    (prev_dir / "diagnosis.json").write_text(json.dumps({
        "failure_modes": ["sparse_reward"],
        "evidence": "prior evidence",
        "proposed_edits": [{
            "target_term": "forward_weight", "operation": "decrease",
            "rationale": "r", "suggested_value": "0.5",
            "paper_refs": list(cited_arxiv_ids),
            "paper_refs_grounded": {}, "requires_env_extension": False,
        }] if cited_arxiv_ids else [],
        "proposed_env_edits": [],
        "literature_context": [
            {
                "technique": name,
                "technique_id": f"technique:{name}",
                "source_paper_ids": [make_paper_id(arxiv_id)],
                "description": "d", "paper_citation": "c", "evidence": "e",
                "relevance_score": 0.9, "matched_on": ["semantic"],
                "grounded": True,
            }
            for name, arxiv_id in shown
        ],
        "confidence": 0.5,
        "iter_dir": str(prev_dir),
        "behavior_goal": "run forward",
    }))


def test_diagnose_stale_rotation_excludes_uncited_and_refills(
    config, kg_with_locomotion_tech, tmp_path, monkeypatch,
):
    """Stuck iteration (delta<=0): 3 techniques shown last iter, only 1
    cited by a proposed edit -> the 2 uncited ones are excluded from THIS
    iteration's merged retrieval, and the block refills from next-ranked
    matches back up to KG_TOP_K."""
    prev_dir = _build_iter_dir(tmp_path, "iter_0")
    _write_prev_diagnosis(
        prev_dir,
        shown=[("tech_a", "1111.11111"), ("tech_b", "2222.22222"),
               ("tech_c", "3333.33333")],
        cited_arxiv_ids=["1111.11111"],  # only tech_a's paper was cited
    )
    cur_dir = _build_iter_dir(tmp_path, "iter_1")

    def _stub_query_semantic(text, top_k=6, store=None, min_similarity=0.0):
        return _stale_pool(_STALE_POOL_NAMES[:top_k])

    def _stub_query_techniques(failure_modes, domain_filter=None, top_k=5,
                                store=None, extra_failure_node_ids=None):
        return _stale_pool(_STALE_POOL_NAMES[:top_k])

    monkeypatch.setattr("sculptor.diagnose.query_semantic", _stub_query_semantic)
    monkeypatch.setattr("sculptor.diagnose.query_techniques", _stub_query_techniques)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="still stuck", confidence=0.6)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d = diagnose(
        iter_dir=cur_dir, behavior_goal="run forward", config=config,
        store=kg_with_locomotion_tech, client=client,
        objective_progress={"current": 0.3, "best_so_far": 0.3, "last": 0.31,
                            "delta": -0.01},
    )

    names = {m.technique.name for m in d.literature_context}
    assert "tech_b" not in names, "shown-uncited technique must be excluded"
    assert "tech_c" not in names, "shown-uncited technique must be excluded"
    assert "tech_a" in names, "the CITED technique must not be excluded"
    # refilled back up to KG_TOP_K from the enlarged fetch (tech_g/tech_h
    # only appear because the fetch top_k grew to make room for exclusion).
    assert len(d.literature_context) == KG_TOP_K
    assert {"tech_g", "tech_h"} <= names

    log_path = cur_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    rotate_rec = next(r for r in records if r["decision"] == "diagnose_stale_rotate")
    assert "technique:tech_b" in rotate_rec["query"]
    assert "technique:tech_c" in rotate_rec["query"]
    assert "technique:tech_a" not in rotate_rec["query"]


def test_diagnose_stale_rotation_inert_when_delta_positive(
    config, kg_with_locomotion_tech, tmp_path, monkeypatch,
):
    """A previous iteration with shown-uncited techniques must NOT trigger
    exclusion when this iteration's delta is positive — merge stays
    byte-identical to the no-objective_progress case."""
    prev_dir = _build_iter_dir(tmp_path, "iter_0")
    _write_prev_diagnosis(
        prev_dir,
        shown=[("tech_a", "1111.11111"), ("tech_b", "2222.22222"),
               ("tech_c", "3333.33333")],
        cited_arxiv_ids=["1111.11111"],
    )
    cur_dir = _build_iter_dir(tmp_path, "iter_1")

    def _stub_query_semantic(text, top_k=6, store=None, min_similarity=0.0):
        return _stale_pool(_STALE_POOL_NAMES[:top_k])

    def _stub_query_techniques(failure_modes, domain_filter=None, top_k=5,
                                store=None, extra_failure_node_ids=None):
        return _stale_pool(_STALE_POOL_NAMES[:top_k])

    monkeypatch.setattr("sculptor.diagnose.query_semantic", _stub_query_semantic)
    monkeypatch.setattr("sculptor.diagnose.query_techniques", _stub_query_techniques)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="improving", confidence=0.6)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d_positive = diagnose(
        iter_dir=cur_dir, behavior_goal="run forward", config=config,
        store=kg_with_locomotion_tech, client=client,
        objective_progress={"current": 0.4, "best_so_far": 0.4, "last": 0.3,
                            "delta": 0.1},
    )
    names_positive = [m.technique.name for m in d_positive.literature_context]

    cur_dir2 = _build_iter_dir(tmp_path, "iter_2")
    d_none = diagnose(
        iter_dir=cur_dir2, behavior_goal="run forward", config=config,
        store=kg_with_locomotion_tech, client=client,
        objective_progress=None,
    )
    names_none = [m.technique.name for m in d_none.literature_context]

    assert names_positive == names_none == ["tech_a", "tech_b", "tech_c",
                                            "tech_d", "tech_e", "tech_f"]

    log_path = cur_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert "diagnose_stale_rotate" not in [r["decision"] for r in records]


def test_diagnose_stale_rotation_never_drops_below_two_results(
    config, kg_with_locomotion_tech, tmp_path, monkeypatch,
):
    """When the retrievable pool IS the shown-uncited set (excluding all of
    it would starve the block below 2 matches), the floor re-admits enough
    of them — in original priority order — to keep >=2 results."""
    prev_dir = _build_iter_dir(tmp_path, "iter_0")
    _write_prev_diagnosis(
        prev_dir,
        shown=[("tech_a", "1111.11111"), ("tech_b", "2222.22222"),
               ("tech_c", "3333.33333")],
        cited_arxiv_ids=["1111.11111"],  # only tech_a cited -> b, c uncited
    )
    cur_dir = _build_iter_dir(tmp_path, "iter_1")

    # Unlike the refill test above, this store only ever has 3 techniques
    # total (a, b, c) regardless of the requested top_k — there is nothing
    # to refill from.
    def _stub_query_semantic(text, top_k=6, store=None, min_similarity=0.0):
        return _stale_pool(["tech_a", "tech_b", "tech_c"])

    def _stub_query_techniques(failure_modes, domain_filter=None, top_k=5,
                                store=None, extra_failure_node_ids=None):
        return _stale_pool(["tech_a", "tech_b", "tech_c"])

    monkeypatch.setattr("sculptor.diagnose.query_semantic", _stub_query_semantic)
    monkeypatch.setattr("sculptor.diagnose.query_techniques", _stub_query_techniques)

    prelim = _PreliminaryModel(
        failure_modes=["sparse_reward"], evidence="still stuck", confidence=0.6)
    grounded = _GroundedModel(proposed_edits=[], confidence=0.5)
    client = _StubClient(prelim, grounded)

    d = diagnose(
        iter_dir=cur_dir, behavior_goal="run forward", config=config,
        store=kg_with_locomotion_tech, client=client,
        objective_progress={"current": 0.3, "best_so_far": 0.3, "last": 0.31,
                            "delta": -0.01},
    )

    names = [m.technique.name for m in d.literature_context]
    assert len(names) >= 2, "the floor must never let the block drop below 2"
    assert "tech_a" in names, "the cited technique is never excluded"
    # exactly one of {tech_b, tech_c} was re-admitted to meet the floor —
    # evidence_matches is scanned first, so tech_b (encountered before
    # tech_c in that pass) is the one re-admitted.
    assert "tech_b" in names
    assert "tech_c" not in names

    log_path = cur_dir / "kg_retrievals.jsonl"
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    rotate_rec = next(r for r in records if r["decision"] == "diagnose_stale_rotate")
    # only the ACTUALLY-applied exclusion (tech_c) is reported — tech_b was
    # re-admitted by the floor and must not appear as "excluded".
    assert "technique:tech_c" in rotate_rec["query"]
    assert "technique:tech_b" not in rotate_rec["query"]


# ── §Iter-29 regression: held-out observable inventory ──────────────────
# Iter 29 of the g1-lab-showcase campaign burned a full iteration emitting
# `requires_env_extension` for a `box_contact` field that already existed as
# the compiled `contact__forbidden__{0..3}` channels. The diagnoser had been
# told only that held-out channels were "structurally excluded" from its
# evidence — never that they exist or what they are named — so it inferred
# contact from a box-VELOCITY proxy that is blind to a robot leaning on a
# stationary box. These tests pin the inventory that prevents that.
class _FakeChannel:
    def __init__(self, name: str, access: str) -> None:
        self.name = name
        self.access = access


class _FakeCatalog:
    def __init__(self, channels) -> None:
        self.channels = channels


class _FakeContract:
    observation_space_spec = "Box(29,)"
    action_space_spec = "Box(29,)"
    expected_info_keys = ["root_link_pos_w", "joint_pos"]
    expected_components = {"upright": "float"}


def test_reward_contract_names_held_out_channels():
    from sculptor.diagnose import _render_reward_contract

    catalog = _FakeCatalog([
        _FakeChannel("object__slalom_box_01__lin_vel_w", "shared_shaping"),
        _FakeChannel("contact__forbidden__0", "metric_only"),
        _FakeChannel("contact__forbidden__1", "metric_only"),
    ])
    text = _render_reward_contract(_FakeContract(), catalog)

    # The exact channels iter 29 claimed did not exist are now named.
    assert "contact__forbidden__0" in text
    assert "contact__forbidden__1" in text
    assert "held_out_metric_observables" in text
    # Reward-visible channels are NOT relisted as held-out.
    held_block = text.split("held_out_metric_observables")[1]
    assert "object__slalom_box_01__lin_vel_w" not in held_block
    # The two behaviors that actually wasted the iteration are named.
    assert "requires_env_extension" in text
    assert "VELOCITY" in text


def test_reward_contract_unchanged_without_catalog():
    """No authored catalog (legacy/registered tasks) → byte-identical to the
    historical rendering, so this cannot perturb non-authored projects."""
    from sculptor.diagnose import _render_reward_contract

    baseline = (
        "observation_space: Box(29,)\n"
        "action_space:      Box(29,)\n"
        "expected_info_keys: ['root_link_pos_w', 'joint_pos']\n"
        "expected_components: {'upright': 'float'}"
    )
    assert _render_reward_contract(_FakeContract(), None) == baseline
    # A catalog with no metric_only channels is equally inert.
    empty = _FakeCatalog([_FakeChannel("joint_pos", "base")])
    assert _render_reward_contract(_FakeContract(), empty) == baseline


def test_held_out_inventory_never_leaks_values():
    """Names only. Emitting VALUES here would breach the metric firewall the
    partition gate exists to enforce."""
    from sculptor.diagnose import _render_held_out_observables

    catalog = _FakeCatalog([_FakeChannel("goal__g__success", "metric_only")])
    text = _render_held_out_observables(catalog)
    assert "goal__g__success" in text
    assert "may NOT read or reconstruct" in text


def test_catalog_loader_is_fail_soft(tmp_path):
    """A project with no authored world must degrade to None, not raise."""
    from sculptor.diagnose import _load_iter_channel_catalog

    iter_dir = tmp_path / "proj" / "runs" / "iter_0"
    iter_dir.mkdir(parents=True)
    assert _load_iter_channel_catalog(iter_dir) is None
