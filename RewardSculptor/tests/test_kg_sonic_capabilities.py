"""Source pinning and product-boundary tests for SONIC paper claims."""

from __future__ import annotations

from sculptor.kg.capabilities import CapabilityStatus
from sculptor.kg.schema import (
    PROVENANCE_PAPER_CLAIM,
    Edge,
    ImplementationStatus,
    Paper,
    Relation,
    ResearchCapability,
    make_implementation_status_id,
    make_paper_id,
    make_research_capability_id,
    row_to_node,
)
from sculptor.kg.sonic_capabilities import (
    SONIC_ARXIV_ID,
    SONIC_CAPABILITIES,
    SONIC_CATALOG_OWNER,
    SONIC_PAPER_VERSION,
    SONIC_SOURCE_URL,
    SonicCapabilityMapError,
    materialize_sonic_capability_map,
    sonic_capability_by_key,
    validate_sonic_capability_catalog,
)
from sculptor.kg.store import SculptorKG

EXPECTED_KEYS = {
    "sonic_universal_controller_contract",
    "sonic_fsq_interface_and_training_loss",
    "sonic_scale_and_training_recipe",
    "sonic_public_bones_seed_release",
    "sonic_motion_tracking_reward_design",
    "sonic_domain_randomization_ranges",
    "sonic_separate_kinematic_motion_planner",
    "sonic_vla_interface",
    "sonic_evaluation_protocol_results_and_limits",
}


def _add_sonic_paper(store: SculptorKG) -> None:
    store.add_node(
        Paper(
            id=make_paper_id(SONIC_ARXIV_ID),
            arxiv_id=SONIC_ARXIV_ID,
            title=(
                "SONIC: Supersizing Motion Tracking for Natural Humanoid "
                "Whole-Body Control"
            ),
            source_url=SONIC_SOURCE_URL,
        )
    )


def _edge_snapshot(store: SculptorKG) -> list[tuple[object, ...]]:
    return sorted(
        (
            edge.src,
            edge.dst,
            edge.relation.value,
            tuple(sorted(edge.data.items())),
            edge.created_at,
        )
        for edge in store.all_edges()
    )


def test_research_capability_parameters_are_backward_compatible() -> None:
    legacy = row_to_node(
        "capability:legacy",
        ResearchCapability.kind,
        {"name": "Legacy capability"},
    )
    assert isinstance(legacy, ResearchCapability)
    assert legacy.parameters == {}


def test_sonic_catalog_is_source_pinned_and_never_claims_implementation() -> None:
    validate_sonic_capability_catalog()

    assert {spec.key for spec in SONIC_CAPABILITIES} == EXPECTED_KEYS
    assert all(
        spec.status is CapabilityStatus.UNSUPPORTED
        for spec in SONIC_CAPABILITIES
    )
    assert all(not spec.code_evidence for spec in SONIC_CAPABILITIES)
    assert all(
        spec.source_locator.startswith(f"{SONIC_SOURCE_URL}#")
        for spec in SONIC_CAPABILITIES
    )


def test_sonic_catalog_exposes_stable_exact_parameter_contract() -> None:
    controller = sonic_capability_by_key(
        "sonic_universal_controller_contract"
    ).parameters
    assert controller["controlled_dof"] == 29
    assert controller["control_rate_hz"] == 50
    assert controller["action_semantics"] == (
        "desired_joint_positions_tracked_by_joint_level_pd"
    )
    assert controller["proprioceptive_history_steps"] == 10
    assert controller["future_frame_intervals_s"] == {
        "robot": 0.1,
        "human": 0.02,
        "hybrid": 0.1,
    }
    assert controller["reported_actor_inputs_include_camera_observations"] is False
    assert controller["action_distribution"] == "diagonal_gaussian"

    fsq = sonic_capability_by_key(
        "sonic_fsq_interface_and_training_loss"
    ).parameters
    assert fsq["token_count"] == 2
    assert fsq["token_dimensions"] == 32
    assert fsq["quantization_levels_per_dimension"] == 32
    assert fsq["flattened_body_token_dimensions"] == 64
    assert fsq["separate_hand_joint_dimensions"] == 14
    assert fsq["hand_dimensions_are_part_of_body_token"] is False
    assert fsq["total_loss"] == "L_ppo + L_recon + L_token + L_cycle"
    assert fsq["critic_hidden_dimensions"] == [
        4096,
        4096,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
    ]

    training = sonic_capability_by_key(
        "sonic_scale_and_training_recipe"
    ).parameters
    assert training["filtered_hours"] == 611
    assert training["training_clips"] == 317189
    assert training["training_frames"] == ">100000000"
    assert training["largest_model_parameters"] == 42000000
    assert training["training_iterations"] == 50000
    assert training["largest_run_gpus"] == 128
    assert training["ppo"]["entropy_coefficient"] == 0.013
    assert training["ppo"]["adaptive_learning_rate_range"] == [
        0.00001,
        0.0002,
    ]
    assert training["adaptive_sampling"] == {
        "bin_size_s": 1,
        "failure_rate_cap_beta": 200,
        "blend_alpha": 0.1,
    }

    bones = sonic_capability_by_key(
        "sonic_public_bones_seed_release"
    ).parameters
    assert bones["dataset_url"] == (
        "https://huggingface.co/datasets/bones-studio/seed"
    )
    assert bones["annotated_motion_sequences"] == 142220
    assert bones["duration_hours"] == 288
    assert bones["actors"] == 522
    assert bones["is_complete_sonic_611_hour_training_corpus"] is False
    assert bones["original_sequences"] == 71132
    assert bones["mirrored_sequences"] == 71088
    assert bones["source_rate_hz"] == 120
    assert bones["formats_reported_by_dataset_card"] == {
        "soma_uniform": "BVH",
        "soma_proportional": "BVH",
        "unitree_g1_mujoco_compatible": "CSV",
    }
    assert bones["access_contract"]["files_and_content_gated"] is True
    assert bones["dataset_card_snapshot"]["revision"] == (
        "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
    )
    assert bones["split_constraint"] == (
        "keep_each_original_and_its_mirror_in_the_same_data_split"
    )

    reward = sonic_capability_by_key(
        "sonic_motion_tracking_reward_design"
    ).parameters
    assert reward["tracking_terms"]["end_effector_position"]["weight"] == 2.0
    assert reward["tracking_terms"]["end_effector_position"]["scale"] == 0.1
    assert reward["penalty_terms"]["joint_limit"]["weight"] == -10.0
    assert reward["penalty_terms"]["undesired_contacts"] == {
        "equation": (
            "sum_c_not_in_ankles_or_wrists(1[||contact_force_c||>1.0N])"
        ),
        "contact_force_threshold_n": 1.0,
        "allowed_contact_links": ["ankles", "wrists"],
        "weight": -0.1,
    }

    randomization = sonic_capability_by_key(
        "sonic_domain_randomization_ranges"
    ).parameters
    assert randomization["physical_parameters"][
        "static_friction_coefficient"
    ] == [0.3, 1.6]
    assert randomization["root_velocity_perturbations"][
        "push_duration_s"
    ] == [1.0, 3.0]
    assert randomization["target_motion_perturbations"][
        "position_jitter"
    ]["z"] == [-0.01, 0.01]
    assert randomization["target_motion_perturbations"][
        "position_jitter"
    ]["displayed_vector_expression"] == "U[-0.05,0.05]^3"

    planner = sonic_capability_by_key(
        "sonic_separate_kinematic_motion_planner"
    ).parameters
    assert planner["latent_space_distinct_from_fsq"] is True
    assert planner["temporal_downsample"] == 4
    assert planner["segment_duration_s"] == [0.8, 2.4]
    assert planner["replan_period_s"] == 0.1
    assert planner["root_trajectory_filter"] == {
        "model": "critically_damped_spring",
        "filtered_quantities": [
            "pelvis_x",
            "pelvis_y",
            "projected_pelvis_heading",
        ],
        "position_damping_coefficient": "5*ln(2)",
        "heading_damping_coefficient": "20*ln(2)",
        "velocity_target_horizon_s": 1.0,
    }

    vla = sonic_capability_by_key("sonic_vla_interface").parameters
    assert vla["whole_body_action_dimensions"] == 78
    assert vla["whole_body_action_breakdown"] == {
        "universal_motion_token": 64,
        "hand_joint_angles": 14,
    }
    assert vla["reported_five_task_average_success_percent"] == 75

    evaluation = sonic_capability_by_key(
        "sonic_evaluation_protocol_results_and_limits"
    ).parameters
    assert evaluation["test_sets"]["phuma"] == {
        "motions": 68326,
        "different_retargeting_pipeline": True,
    }
    assert evaluation["cross_dataset_comparison_is_not_data_matched"] is True
    assert evaluation["global_root_position_is_not_tracked"] is True
    assert evaluation["physical_robot_evaluation"]["successful_sequences"] == 123


def test_sonic_materialization_requires_exact_paper_node(tmp_path) -> None:
    import pytest

    with SculptorKG(tmp_path / "graph.db") as store:
        with pytest.raises(
            SonicCapabilityMapError,
            match="before its exact paper node exists",
        ):
            materialize_sonic_capability_map(store)
        assert store.count_nodes() == 0
        assert store.count_edges() == 0


def test_sonic_materialization_is_typed_grounded_and_idempotent(
    tmp_path,
) -> None:
    with SculptorKG(tmp_path / "graph.db") as store:
        _add_sonic_paper(store)
        counts = materialize_sonic_capability_map(store)

        assert counts == {
            "capabilities": len(SONIC_CAPABILITIES),
            "statuses": 1,
            "paper_edges": len(SONIC_CAPABILITIES),
            "status_edges": len(SONIC_CAPABILITIES),
            "stale_capabilities_removed": 0,
            "stale_edges_removed": 0,
        }
        for spec in SONIC_CAPABILITIES:
            node = store.get_node(spec.node_id)
            assert isinstance(node, ResearchCapability)
            assert node.provenance == PROVENANCE_PAPER_CLAIM
            assert node.code_evidence == []
            assert node.parameters["catalog_owner"] == SONIC_CATALOG_OWNER
            assert node.parameters["source_arxiv_id"] == SONIC_ARXIV_ID
            assert node.parameters["source_version"] == SONIC_PAPER_VERSION
            assert node.parameters["source_locator"] == spec.source_locator

            status_edges = store.neighbors(
                spec.node_id,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
                direction="out",
            )
            assert len(status_edges) == 1
            assert status_edges[0][1] == make_implementation_status_id(
                "unsupported"
            )
            grounding_edges = store.neighbors(
                spec.node_id,
                relation=Relation.GROUNDS_CAPABILITY,
                direction="in",
            )
            assert len(grounding_edges) == 1
            grounding, source_id = grounding_edges[0]
            assert source_id == make_paper_id(SONIC_ARXIV_ID)
            assert grounding.data == {
                "catalog_owner": SONIC_CATALOG_OWNER,
                "paper_role": spec.paper_role,
                "source_locator": spec.source_locator,
                "paper_version": SONIC_PAPER_VERSION,
            }

        before = _edge_snapshot(store)
        repeated = materialize_sonic_capability_map(store)
        after = _edge_snapshot(store)
        assert repeated["stale_capabilities_removed"] == 0
        assert repeated["stale_edges_removed"] == 0
        assert before == after


def test_sonic_materialization_reconciles_only_owned_stale_graph_state(
    tmp_path,
) -> None:
    with SculptorKG(tmp_path / "graph.db") as store:
        _add_sonic_paper(store)
        other_paper_id = make_paper_id("0000.00000")
        store.add_node(
            Paper(
                id=other_paper_id,
                arxiv_id="0000.00000",
                title="Unrelated paper",
            )
        )
        materialize_sonic_capability_map(store)

        active = SONIC_CAPABILITIES[0]
        implemented_id = make_implementation_status_id("implemented")
        store.add_node(
            ImplementationStatus(
                id=implemented_id,
                status="implemented",
                definition="Contradictory test status",
            )
        )
        store.add_edge(
            Edge(
                src=active.node_id,
                dst=implemented_id,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
            )
        )
        store.add_edge(
            Edge(
                src=other_paper_id,
                dst=active.node_id,
                relation=Relation.GROUNDS_CAPABILITY,
                data={"paper_role": "stale_wrong_source"},
            )
        )

        stale_id = make_research_capability_id("sonic_retired_test_claim")
        store.add_node(
            ResearchCapability(
                id=stale_id,
                name="Retired SONIC claim",
                parameters={"catalog_owner": SONIC_CATALOG_OWNER},
            )
        )
        store.add_edge(
            Edge(
                src=make_paper_id(SONIC_ARXIV_ID),
                dst=stale_id,
                relation=Relation.GROUNDS_CAPABILITY,
                data={"catalog_owner": SONIC_CATALOG_OWNER},
            )
        )
        store.add_edge(
            Edge(
                src=stale_id,
                dst=make_implementation_status_id("unsupported"),
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
                data={"catalog_owner": SONIC_CATALOG_OWNER},
            )
        )
        dangling_id = make_research_capability_id(
            "sonic_retired_dangling_claim"
        )
        store.add_edge(
            Edge(
                src=make_paper_id(SONIC_ARXIV_ID),
                dst=dangling_id,
                relation=Relation.GROUNDS_CAPABILITY,
                data={"catalog_owner": SONIC_CATALOG_OWNER},
            )
        )

        unrelated_id = make_research_capability_id("unrelated_capability")
        store.add_node(
            ResearchCapability(
                id=unrelated_id,
                name="Unrelated capability",
                parameters={"catalog_owner": "some_other_catalog"},
            )
        )
        store.add_edge(
            Edge(
                src=other_paper_id,
                dst=unrelated_id,
                relation=Relation.GROUNDS_CAPABILITY,
                data={"catalog_owner": "some_other_catalog"},
            )
        )

        counts = materialize_sonic_capability_map(store)

        assert counts["stale_capabilities_removed"] == 1
        assert counts["stale_edges_removed"] >= 5
        assert store.get_node(stale_id) is None
        assert store.neighbors(
            make_paper_id(SONIC_ARXIV_ID),
            relation=Relation.GROUNDS_CAPABILITY,
            direction="out",
        ) != []
        assert all(
            other_id != dangling_id
            for _edge, other_id in store.neighbors(
                make_paper_id(SONIC_ARXIV_ID),
                relation=Relation.GROUNDS_CAPABILITY,
                direction="out",
            )
        )
        assert [
            other_id
            for _edge, other_id in store.neighbors(
                active.node_id,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
                direction="out",
            )
        ] == [make_implementation_status_id("unsupported")]
        assert [
            other_id
            for _edge, other_id in store.neighbors(
                active.node_id,
                relation=Relation.GROUNDS_CAPABILITY,
                direction="in",
            )
        ] == [make_paper_id(SONIC_ARXIV_ID)]
        assert isinstance(store.get_node(unrelated_id), ResearchCapability)
        assert store.neighbors(
            unrelated_id,
            relation=Relation.GROUNDS_CAPABILITY,
            direction="in",
        )
