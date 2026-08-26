"""Executable claim boundary for OGMP and Preferenced-OGMP support."""

from __future__ import annotations

from sculptor.kg.capabilities import (
    CapabilityStatus,
    OGMP_CAPABILITIES,
    implementation_status_map,
    materialize_ogmp_capability_map,
    mode_api_capability_summary,
    unsupported_mode_diagnostic_keys,
    validate_ogmp_capability_catalog,
)
from sculptor.kg.schema import (
    Edge,
    ImplementationStatus,
    Paper,
    Relation,
    ResearchCapability,
    make_implementation_status_id,
    make_paper_id,
)
from sculptor.kg.store import SculptorKG


EXPECTED_STATUS_MAP = {
    "fixed_linear_phase_window_dispatch": "implemented",
    "per_mode_reward_authoring_and_scope": "implemented",
    "immutable_mode_execution_admission_and_diagnostics": "implemented",
    "transition_guard_declarations": "metadata_only",
    "mode_success_predicate_declarations": "metadata_only",
    "online_receding_horizon_oracle": "unsupported",
    "rho_bounded_permissible_state_exploration": "unsupported",
    "learned_mode_latent_and_task_feedback_conditioning": "unsupported",
    "runtime_predicate_or_branch_transition_execution": "unsupported",
    "preference_conditioned_oracle_or_policy": "unsupported",
}


def _add_papers(store: SculptorKG) -> None:
    store.add_node(
        Paper(
            id=make_paper_id("2403.04205"),
            arxiv_id="2403.04205",
            title="Oracle Guided Multi-mode Policies",
        )
    )
    store.add_node(
        Paper(
            id=make_paper_id("2410.01030"),
            arxiv_id="2410.01030",
            title="Preferenced Oracle Guided Multi-mode Policies",
        )
    )


def test_reviewed_capability_statuses_are_explicit_and_exact() -> None:
    """A claim change must update this reviewer-facing contract explicitly."""

    assert implementation_status_map() == EXPECTED_STATUS_MAP
    assert {spec.status for spec in OGMP_CAPABILITIES} == set(CapabilityStatus)
    assert all(
        not spec.code_evidence
        for spec in OGMP_CAPABILITIES
        if spec.status is CapabilityStatus.UNSUPPORTED
    )


def test_implemented_and_metadata_claim_evidence_symbols_still_exist() -> None:
    """Moving/removing an execution surface invalidates its KG claim."""

    validate_ogmp_capability_catalog(resolve_symbols=True)


def test_mode_disclosures_are_derived_from_the_reviewed_catalog() -> None:
    summary = mode_api_capability_summary()
    assert summary["implementation_status"] == EXPECTED_STATUS_MAP
    assert summary["runtime_transition_guards"] is False
    assert summary["policy_mode_conditioning"] is False
    assert summary["rho_bounded_exploration"] is False
    assert summary["closed_loop_receding_horizon_oracle"] is False
    assert summary["preference_conditioning"] is False
    assert unsupported_mode_diagnostic_keys() == [
        "closed_loop_oracle",
        "rho_bounded_exploration",
        "predicate_or_branch_executor",
        "mode_conditioned_policy",
    ]


def test_capability_map_materializes_explicit_nodes_and_edges(tmp_path) -> None:
    with SculptorKG(tmp_path / "graph.db") as store:
        _add_papers(store)
        counts = materialize_ogmp_capability_map(store)

        assert counts == {
            "capabilities": len(EXPECTED_STATUS_MAP),
            "statuses": 3,
            "paper_edges": 14,
            "status_edges": len(EXPECTED_STATUS_MAP),
        }
        capabilities = store.find_nodes(kind=ResearchCapability.kind)
        statuses = store.find_nodes(kind=ImplementationStatus.kind)
        assert {node.id for node in capabilities} == {
            spec.node_id for spec in OGMP_CAPABILITIES
        }
        assert {node.status for node in statuses} == {
            "implemented",
            "metadata_only",
            "unsupported",
        }
        for spec in OGMP_CAPABILITIES:
            status_edges = store.neighbors(
                spec.node_id,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
                direction="out",
            )
            assert len(status_edges) == 1
            assert status_edges[0][1] == make_implementation_status_id(
                spec.status.value
            )
            paper_edges = store.neighbors(
                spec.node_id,
                relation=Relation.GROUNDS_CAPABILITY,
                direction="in",
            )
            assert {other for _edge, other in paper_edges} == {
                make_paper_id(arxiv_id) for arxiv_id in spec.paper_arxiv_ids
            }

        # A previous mapping cannot leave a second, contradictory status edge.
        first = OGMP_CAPABILITIES[0]
        wrong_status = make_implementation_status_id("unsupported")
        store.add_edge(
            Edge(
                src=first.node_id,
                dst=wrong_status,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
            )
        )
        materialize_ogmp_capability_map(store)
        status_edges = store.neighbors(
            first.node_id,
            relation=Relation.HAS_IMPLEMENTATION_STATUS,
            direction="out",
        )
        assert [other for _edge, other in status_edges] == [
            make_implementation_status_id("implemented")
        ]

        # Idempotent materialization leaves one node and one status edge each.
        materialize_ogmp_capability_map(store)
        assert len(store.find_nodes(kind=ResearchCapability.kind)) == len(
            EXPECTED_STATUS_MAP
        )


def test_materialization_requires_real_paper_nodes(tmp_path) -> None:
    from pytest import raises

    with SculptorKG(tmp_path / "graph.db") as store:
        with raises(ValueError, match="before its paper nodes exist"):
            materialize_ogmp_capability_map(store)
