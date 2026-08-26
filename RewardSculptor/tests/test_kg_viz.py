"""tests/test_kg_viz.py — offline viz smoke tests.

No LLM or network. Builds a small in-memory KG, renders HTML, asserts the
output contains the expected structure (nodes, legend, provenance
highlighting).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.kg.schema import (
    ArtifactAttestation,
    Edge,
    Environment,
    FailureMode,
    ModeExecutionArtifact,
    Paper,
    PolicyArtifact,
    ReferenceMotion,
    Relation,
    RobotEmbodiment,
    SoftwareEnvironment,
    Technique,
    TrainingIteration,
    TrainingRun,
    WorldArtifact,
    make_environment_id,
    make_failure_mode_id,
    make_paper_id,
    make_technique_id,
)
from sculptor.kg.store import SculptorKG
from sculptor.kg.viz import build_kg_html


@pytest.fixture
def kg(tmp_path: Path) -> SculptorKG:
    store = SculptorKG(tmp_path / "kg.db")
    p = Paper(
        id=make_paper_id("1801.00690"), arxiv_id="1801.00690",
        title="DeepMind Control Suite", authors=["Tassa"], year=2018,
        abstract="Bounded smooth tolerance kernels in [0, 1].")
    t = Technique(
        id=make_technique_id("tolerance_kernel"), name="tolerance_kernel",
        description="Smooth reward kernel that saturates in a bounded interval.",
        tags=["dense_reward", "reward_shaping"])
    f = FailureMode(
        id=make_failure_mode_id("sparse_reward"), name="sparse_reward",
        description="Reward signal is almost always zero.")
    e = Environment(
        id=make_environment_id("Hopper-v4"), name="Hopper-v4",
        description="MuJoCo Hopper.", tags=["continuous_locomotion"])
    for n in (p, t, f, e):
        store.add_node(n)
    store.add_edge(Edge(src=p.id, dst=t.id, relation=Relation.INTRODUCES,
                        data={"evidence": "paper introduces tolerance kernel"}))
    store.add_edge(Edge(src=t.id, dst=f.id, relation=Relation.ADDRESSES,
                        data={"evidence": "kernel densifies sparse reward",
                              "source_paper_id": p.id}))
    store.add_edge(Edge(src=p.id, dst=e.id, relation=Relation.EVALUATES_ON,
                        data={}))
    return store


def test_build_kg_html_writes_file(kg, tmp_path: Path):
    out = tmp_path / "kg.html"
    result = build_kg_html(kg, out)
    assert result.out_path == out.resolve()
    assert out.is_file()
    assert out.stat().st_size > 10_000  # real pyvis HTML is ~100KB+
    body = out.read_text(encoding="utf-8")
    # Legend present
    assert "sculpt-legend" in body
    # Every node kind color rendered in the legend
    for color in ("#4fc3f7", "#81c784", "#ef5350", "#ffb74d", "#ffd54f"):
        assert color in body
    # Node IDs land in the graph data
    assert "paper:1801.00690" in body
    assert "technique:tolerance-kernel" in body


def test_build_kg_html_highlights_provenance_nodes(kg, tmp_path: Path):
    provenance = {
        "tolerance_kernel": [{
            "arxiv_id": "1801.00690",
            "citation": "Tassa et al. (2018)",
            "iter_introduced": 1,
            "how_used": "Added per DM Control.",
            "still_active": True,
        }],
        "removed_term": [{
            "arxiv_id": "9999.99999",
            "citation": "fake",
            "iter_introduced": 1,
            "still_active": False,
        }],
    }
    out = tmp_path / "kg.html"
    result = build_kg_html(kg, out, provenance=provenance)
    # 1 paper (1801.00690) + 1 technique (tolerance_kernel) should be gold.
    # The 'removed_term' entry has still_active=False, so it doesn't count.
    assert result.n_active_nodes == 2
    body = out.read_text(encoding="utf-8")
    assert "2 node(s) in gold" in body
    # Active-reason text appears in the tooltip somewhere.
    assert "appears in provenance.json" in body


def test_build_kg_html_handles_empty_provenance(kg, tmp_path: Path):
    out = tmp_path / "kg.html"
    result = build_kg_html(kg, out, provenance={})
    assert result.n_active_nodes == 0
    body = out.read_text(encoding="utf-8")
    assert "OWN run experience" in body   # zero-active note (html-escaped ')


def test_build_kg_html_no_provenance_arg(kg, tmp_path: Path):
    out = tmp_path / "kg.html"
    result = build_kg_html(kg, out)  # provenance=None default
    assert result.n_active_nodes == 0
    assert out.is_file()


def test_build_kg_html_tolerates_dangling_edge(tmp_path: Path):
    """A DANGLING edge (an edge referencing a node that was never persisted — e.g. an
    unhealed `failure:…` stub) must NOT crash the viz. pyvis's add_edge asserts the
    endpoint node exists, which used to 500 the whole live graph.html; the renderer now
    skips the un-drawable edge and renders the consistent subgraph."""
    store = SculptorKG(tmp_path / "kg.db")
    p = Paper(id=make_paper_id("2000.00001"), arxiv_id="2000.00001",
              title="Paper", authors=["A"], year=2020, abstract="x")
    t = Technique(id=make_technique_id("good_tech"), name="good_tech",
                  description="present node", tags=[])
    store.add_node(p)
    store.add_node(t)
    store.add_edge(Edge(src=p.id, dst=t.id, relation=Relation.INTRODUCES, data={}))
    # a DANGLING edge: dst node id was never added to the store.
    store.add_edge(Edge(src=t.id, dst=make_failure_mode_id("reward-saturation"),
                        relation=Relation.ADDRESSES, data={}))

    out = tmp_path / "kg.html"
    res = build_kg_html(store, out)                 # must NOT raise
    assert out.is_file() and out.stat().st_size > 1000
    assert res.n_nodes == 2
    assert res.n_edges == 1                          # the dangling edge was skipped


def test_artifact_lineage_has_distinct_legend_details_and_edge_receipt(
    tmp_path: Path,
):
    store = SculptorKG(tmp_path / "kg.db")
    policy = PolicyArtifact(
        id="policy:" + "a" * 64,
        sha256="a" * 64,
        artifact_format="safetensors",
        size_bytes=1234,
        tensor_inventory_digest="b" * 64,
    )
    motion = ReferenceMotion(
        id="reference:g1:cartwheel:" + "c" * 64,
        sha256="c" * 64,
        fps=50.0,
        frame_count=180,
        joint_names=["left_hip_pitch", "right_hip_pitch"],
    )
    world = WorldArtifact(
        id="world:" + "d" * 64,
        sha256="d" * 64,
        artifact_format="selection_manifest",
    )
    mode = ModeExecutionArtifact(
        id="mode-execution:" + "e" * 64,
        bundle_digest="e" * 64,
        reward_sha256="f" * 64,
        robot="g1",
        clip_id="cartwheel-v3",
        clip_sha256="c" * 64,
        graph_sha256="1" * 64,
        execution_manifest_digest="2" * 64,
        selection_digest="3" * 64,
        context_refs_digest="4" * 64,
        context_refs={"world": "d" * 64},
    )
    attestation = ArtifactAttestation(
        id="attestation:" + "5" * 64,
        manifest_digest="5" * 64,
        trust_status="validated",
        source_format="portable_bundle_v1",
        declared={"robot": "g1", "initialization_modes": ["actor_only"]},
    )
    robot = RobotEmbodiment(
        id="robot:g1:" + "6" * 64,
        slug="g1",
        contract_digest="6" * 64,
        joint_names=["left_hip_pitch", "right_hip_pitch"],
        observation_contract={"policy": {"shape": [128]}},
        action_contract={"shape": [29]},
        control_dt_s=0.02,
    )
    software = SoftwareEnvironment(
        id="software:" + "7" * 64,
        lock_digest="7" * 64,
        versions={"python": "3.11"},
        code_commit="abc123def456",
        code_dirty=False,
        runtime={"adapter": "mjlab"},
    )
    run = TrainingRun(
        id="training-run:g1-evolution:job-1",
        project="g1-evolution",
        run_id="job-1",
        requested_initialization_mode="actor_critic",
        observed_initialization_mode="actor_critic",
        selection_digest="3" * 64,
    )
    iteration = TrainingIteration(
        id="training-iteration:g1-evolution:job-1:0",
        project="g1-evolution",
        run_id="job-1",
        iteration_index=0,
    )
    for node in (
        policy, motion, world, mode, attestation, robot, software, run,
        iteration,
    ):
        store.add_node(node)
    store.add_edge(Edge(
        src=run.id,
        dst=policy.id,
        relation=Relation.INITIALIZED_FROM,
        data={
            "authority": "worker_event",
            "requested": {"roles": ["actor", "critic"]},
            "observed": {
                "roles": ["actor", "critic"],
                "checkpoint_sha256": "a" * 64,
            },
        },
    ))
    store.add_edge(Edge(
        src=run.id, dst=iteration.id, relation=Relation.HAS_ITERATION,
        data={"iteration_index": 0},
    ))

    out = tmp_path / "lineage.html"
    result = build_kg_html(store, out)
    body = out.read_text(encoding="utf-8")

    assert result.n_nodes == 9
    for label in (
        "Policy artifact", "Reference motion", "World artifact",
        "Mode execution", "Artifact attestation", "Robot embodiment",
        "Software environment", "Training run", "Training iteration",
    ):
        assert label in body
    for detail in (
        "tensor inventory digest", "ordered joints", "execution manifest",
        "declared metadata", "contract digest", "requested initialization",
    ):
        assert detail in body
    assert "structured receipt" in body
    assert "authority" in body
    assert "worker_event" in body
    assert "requested" in body
    assert "observed" in body
    assert "checkpoint_sha256" in body
    assert "a" * 64 in body
