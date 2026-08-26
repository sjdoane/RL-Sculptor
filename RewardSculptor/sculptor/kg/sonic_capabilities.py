"""Reviewed SONIC v4 claims with an explicit product-capability boundary.

The SONIC paper describes a controller, token interface, planner, and VLA
integration.  None of those mechanisms is currently executed by
RewardSculptor.  This catalog therefore records the paper's exact, useful
parameters while giving every capability the separate ``unsupported``
implementation status.  A paper claim is never promoted into a product claim
merely because its parameters are queryable in the knowledge graph.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sculptor.kg.capabilities import STATUS_DEFINITIONS, CapabilityStatus
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
)

if TYPE_CHECKING:  # pragma: no cover
    from sculptor.kg.store import SculptorKG


SONIC_ARXIV_ID = "2511.07820"
SONIC_PAPER_VERSION = "v4"
SONIC_SOURCE_URL = "https://arxiv.org/html/2511.07820v4"
SONIC_CATALOG_OWNER = "sonic_2511.07820v4_reviewed_capabilities"


@dataclass(frozen=True)
class SonicCapabilitySpec:
    """One source-pinned paper mechanism or evaluation claim."""

    key: str
    name: str
    description: str
    paper_role: str
    source_locator: str
    parameters: dict[str, Any]
    scope: str = "external_paper_mechanism"
    status: CapabilityStatus = CapabilityStatus.UNSUPPORTED
    code_evidence: tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return make_research_capability_id(self.key)


SONIC_CAPABILITIES: tuple[SonicCapabilitySpec, ...] = (
    SonicCapabilitySpec(
        key="sonic_universal_controller_contract",
        name="SONIC universal motion-tracking controller contract",
        description=(
            "The paper's 29-DoF Unitree G1 policy tracks local-frame robot, "
            "human, or hybrid motion commands and emits desired joint "
            "positions for the robot's joint-level PD controller."
        ),
        paper_role="primary_source_controller_contract",
        source_locator=f"{SONIC_SOURCE_URL}#S3.SS2",
        parameters={
            "controlled_dof": 29,
            "control_rate_hz": 50,
            "action_dimensions": 29,
            "action_distribution": "diagonal_gaussian",
            "action_semantics": (
                "desired_joint_positions_tracked_by_joint_level_pd"
            ),
            "proprioceptive_history_steps": 10,
            "proprioceptive_quantities": [
                "joint_positions",
                "joint_velocities",
                "root_angular_velocity",
                "root_frame_gravity",
                "previous_action",
            ],
            "motion_command_types": ["robot", "human", "hybrid"],
            "state_reference_frame": "robot_local",
            "rotation_representation": "6d",
            "future_command_frames": {
                "robot": 10,
                "human": 10,
                "hybrid": 10,
            },
            "future_frame_intervals_s": {
                "robot": 0.1,
                "human": 0.02,
                "hybrid": 0.1,
            },
            "control_decoder_inputs": [
                "universal_motion_token",
                "proprioceptive_state",
            ],
            "reported_actor_inputs_include_camera_observations": False,
            "critic_training": {
                "asymmetric": True,
                "privileged_quantities": [
                    "base_linear_velocity",
                    "full_body_link_positions",
                    "full_body_link_orientations",
                    "noise_free_observations",
                ],
            },
            "tracking_scope": {
                "local_motion_tracking": True,
                "global_root_trajectory_tracking": False,
            },
            "source_sections": [
                "3.2 Motion Tracking Formulation",
                "3.2 Universal Control Policy",
                "Table S1",
                "S1.2 Deployment Architecture",
            ],
        },
    ),
    SonicCapabilitySpec(
        key="sonic_fsq_interface_and_training_loss",
        name="SONIC universal FSQ interface and joint training objective",
        description=(
            "Three specialized encoders align synchronized robot, human, "
            "and hybrid commands in a shared two-token FSQ representation; "
            "PPO and three auxiliary losses are optimized jointly."
        ),
        paper_role="primary_source_token_interface_and_training_objective",
        source_locator=f"{SONIC_SOURCE_URL}#S3.SS2.SSS0.Px2",
        parameters={
            "quantizer": "finite_scalar_quantization",
            "default_configuration": "FSQ-32-32",
            "token_count": 2,
            "token_dimensions": 32,
            "quantization_levels_per_dimension": 32,
            "flattened_body_token_dimensions": 64,
            "separate_hand_joint_dimensions": 14,
            "hand_dimensions_are_part_of_body_token": False,
            "input_encoders": {
                "robot": "future_robot_joint_positions_and_velocities",
                "human": "future_smpl_3d_joint_positions",
                "hybrid": (
                    "current_head_and_hand_keypoints_plus_future_lower_body_"
                    "robot_motion"
                ),
            },
            "encoder_architecture": "MLP",
            "encoder_hidden_dimensions": [2048, 1024, 512, 512],
            "control_decoder_architecture": "MLP",
            "control_decoder_hidden_dimensions": [
                4096,
                4096,
                2048,
                2048,
                1024,
                1024,
                512,
                512,
            ],
            "robot_motion_decoder_architecture": "MLP",
            "robot_motion_decoder_paper_label": "Decoder (refs)",
            "robot_motion_decoder_hidden_dimensions": [
                2048,
                1024,
                512,
                512,
            ],
            "critic_architecture": "MLP",
            "critic_hidden_dimensions": [
                4096,
                4096,
                2048,
                2048,
                1024,
                1024,
                512,
                512,
            ],
            "total_loss": "L_ppo + L_recon + L_token + L_cycle",
            "loss_terms": {
                "L_recon": (
                    "||D_r(z_r)-g_r||^2 + ||D_r(z_h)-g_r||^2 + "
                    "||D_r(z_m)-g_r||^2"
                ),
                "L_token": (
                    "||z_r-z_h||^2 + ||z_r-z_m||^2 + ||z_m-z_h||^2"
                ),
                "L_cycle": "||E_r(D_r(z_h))-z_r||^2",
            },
            "gradient_routing": {
                "ppo_updates": [
                    "encoders",
                    "fsq_quantizer",
                    "control_decoder",
                    "critic",
                ],
                "auxiliary_losses_update": [
                    "encoders",
                    "robot_motion_decoder",
                ],
                "quantizer_estimator": "straight_through",
            },
            "source_sections": [
                "3.2 Universal Control Policy",
                "Equations 1-4",
                "Table S1",
                "3.6 Quantizer Design and Configuration",
            ],
        },
    ),
    SonicCapabilitySpec(
        key="sonic_scale_and_training_recipe",
        name="SONIC scale and training recipe",
        description=(
            "The largest reported tracker combines the filtered motion "
            "corpus, a 42M-parameter network, distributed Isaac Lab PPO, "
            "adaptive failure-bin sampling, and motion-command domain "
            "randomization."
        ),
        paper_role="primary_source_scale_and_training_recipe",
        source_locator=f"{SONIC_SOURCE_URL}#S2.SS1",
        parameters={
            "source_motion_hours_approx": 700,
            "filtered_hours": 611,
            "training_clips": 317189,
            "training_frames": ">100000000",
            "training_frame_rate_hz": 50,
            "motion_categories": 33,
            "training_subcategories": 8447,
            "largest_model_parameters": 42000000,
            "model_parameter_sweep": [1200000, 16000000, 42000000],
            "training_iterations": 50000,
            "largest_run_gpus": 128,
            "largest_run_gpu_hours_approx": 21000,
            "largest_run_wall_days": 7,
            "simulator": "Isaac Lab",
            "parallel_environments_per_gpu": 4096,
            "ppo": {
                "rollout_steps_per_environment": 24,
                "learning_epochs": 5,
                "mini_batches": 4,
                "discount_gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_parameter": 0.2,
                "entropy_coefficient": 0.013,
                "value_loss_coefficient": 1.0,
                "actor_learning_rate": 0.00002,
                "critic_learning_rate": 0.001,
                "max_gradient_norm": 0.1,
                "desired_kl": 0.01,
                "adaptive_learning_rate_range": [0.00001, 0.0002],
                "initial_policy_noise_std": 0.05,
                "actor_std_clamp": [0.001, 0.5],
            },
            "adaptive_sampling": {
                "bin_size_s": 1,
                "failure_rate_cap_beta": 200,
                "blend_alpha": 0.1,
            },
            "domain_randomization_families": [
                "friction_and_restitution",
                "default_joint_positions",
                "base_center_of_mass_offset",
                "root_velocity_pushes",
                "target_motion_commands",
            ],
            "source_sections": [
                "2.1 Scaling Up Motion Tracking",
                "3.1 Humanoid Motion Dataset",
                "3.2 Training",
                "Table S2",
            ],
        },
    ),
    SonicCapabilitySpec(
        key="sonic_public_bones_seed_release",
        name="SONIC-linked BONES-SEED public motion-data release",
        description=(
            "The paper identifies BONES-SEED as a public-repository subset "
            "of its motion-capture collection, with synchronized human and "
            "Unitree G1 representations plus language and temporal labels; "
            "the current dataset card gates file access behind its license."
        ),
        paper_role="primary_source_public_dataset_release",
        source_locator=f"{SONIC_SOURCE_URL}#S3.SS1.SSS0.Px2",
        scope="external_dataset_release",
        parameters={
            "dataset_name": "BONES-SEED",
            "dataset_url": "https://huggingface.co/datasets/bones-studio/seed",
            "dataset_card_snapshot": {
                "revision": "2f59b2077b9da34dd4e43618e705c7cb962c9a66",
                "last_modified": "2026-05-03T15:03:12Z",
                "reviewed_on": "2026-08-24",
            },
            "relationship_to_sonic_corpus": (
                "substantial_public_subset_of_motion_capture_collection"
            ),
            "is_complete_sonic_611_hour_training_corpus": False,
            "annotated_motion_sequences": 142220,
            "original_sequences": 71132,
            "mirrored_sequences": 71088,
            "duration_hours": 288,
            "source_rate_hz": 120,
            "actors": 522,
            "formats_reported_by_paper": ["SOMA", "Unitree G1"],
            "formats_reported_by_dataset_card": {
                "soma_uniform": "BVH",
                "soma_proportional": "BVH",
                "unitree_g1_mujoco_compatible": "CSV",
            },
            "annotations_reported_by_paper": [
                "natural_language_descriptions",
                "temporal_segmentation_labels",
                "actor_information",
            ],
            "dataset_card_metadata": {
                "main_metadata_rows": 142220,
                "main_metadata_columns": 51,
                "mirror_flag_field": "is_mirror",
                "metadata_only_path": (
                    "hf://datasets/bones-studio/seed/metadata/"
                    "seed_metadata_v003.parquet"
                ),
            },
            "access_contract": {
                "repository_public": True,
                "files_and_content_gated": True,
                "login_required": True,
                "contact_information_shared_on_acceptance": True,
                "license": "BONES-SEED License",
            },
            "split_constraint": (
                "keep_each_original_and_its_mirror_in_the_same_data_split"
            ),
            "paper_reported_facts": [
                "annotated_motion_sequences",
                "duration_hours",
                "actors",
                "formats_reported_by_paper",
                "annotations_reported_by_paper",
            ],
            "dataset_card_reported_facts": [
                "dataset_card_snapshot",
                "original_sequences",
                "mirrored_sequences",
                "source_rate_hz",
                "formats_reported_by_dataset_card",
                "dataset_card_metadata",
                "access_contract",
            ],
            "source_sections": [
                "3.1 Public Data Release",
                "Data, Code and Materials Availability",
                "Reference 56",
            ],
            "additional_source_locators": [
                (
                    "https://huggingface.co/datasets/bones-studio/seed/"
                    "tree/2f59b2077b9da34dd4e43618e705c7cb962c9a66"
                )
            ],
            "applicability_boundary": (
                "motion_data_and_annotations_not_a_dynamics_certificate_"
                "controller_checkpoint_or_complete_sonic_training_corpus"
            ),
            "admission_boundary": (
                "120hz_g1_csv_requires_hostile_data_validation_exact_joint_"
                "root_frame_and_cadence_contracts_new_identity_for_retiming_"
                "and_tier_d_certification_before_research_training"
            ),
        },
    ),
    SonicCapabilitySpec(
        key="sonic_motion_tracking_reward_design",
        name="SONIC motion-tracking reward and penalty design",
        description=(
            "Table S3 reports seven exponential tracking terms and five "
            "penalties used jointly across the SONIC scaling runs."
        ),
        paper_role="primary_source_reward_design",
        source_locator=f"{SONIC_SOURCE_URL}#A0.T3",
        parameters={
            "combination": "weighted_sum_of_tracking_rewards_and_penalties",
            "tracking_terms": {
                "root_position": {
                    "equation": (
                        "exp(-||p_current_root-p_goal_root||_2^2/0.3^2)"
                    ),
                    "scale": 0.3,
                    "weight": 0.5,
                },
                "root_orientation": {
                    "equation": (
                        "exp(-||o_current_root-o_goal_root||_2^2/0.4^2)"
                    ),
                    "scale": 0.4,
                    "weight": 0.5,
                },
                "body_link_position_root_relative": {
                    "equation": (
                        "exp(-(1/|B|)*sum_b(||p_current_rel[b]-"
                        "p_goal_rel[b]||_2^2)/0.3^2)"
                    ),
                    "scale": 0.3,
                    "weight": 1.0,
                },
                "body_link_orientation_root_relative": {
                    "equation": (
                        "exp(-(1/|B|)*sum_b(||o_current_rel[b]-"
                        "o_goal_rel[b]||_2^2)/0.4^2)"
                    ),
                    "scale": 0.4,
                    "weight": 1.0,
                },
                "body_link_linear_velocity": {
                    "equation": (
                        "exp(-(1/|B|)*sum_b(||v_current[b]-"
                        "v_goal[b]||_2^2)/1.0^2)"
                    ),
                    "scale": 1.0,
                    "weight": 1.0,
                },
                "body_link_angular_velocity": {
                    "equation": (
                        "exp(-(1/|B|)*sum_b(||omega_current[b]-"
                        "omega_goal[b]||_2^2)/3.14^2)"
                    ),
                    "scale": 3.14,
                    "weight": 1.0,
                },
                "end_effector_position": {
                    "equation": (
                        "exp(-(1/5)*sum_k(||p_current[k]-"
                        "p_goal[k]||_2^2)/0.1^2)"
                    ),
                    "keypoints": [
                        "head",
                        "left_wrist",
                        "right_wrist",
                        "left_ankle",
                        "right_ankle",
                    ],
                    "scale": 0.1,
                    "weight": 2.0,
                },
            },
            "penalty_terms": {
                "action_rate": {
                    "equation": "||action_t-action_t_minus_1||_2^2",
                    "weight": -0.1,
                },
                "joint_limit": {
                    "equation": (
                        "sum_j(1[joint_position_j outside min_max_limits])"
                    ),
                    "weight": -10.0,
                },
                "undesired_contacts": {
                    "equation": (
                        "sum_c_not_in_ankles_or_wrists(1[||contact_force_c||"
                        ">1.0N])"
                    ),
                    "contact_force_threshold_n": 1.0,
                    "allowed_contact_links": ["ankles", "wrists"],
                    "weight": -0.1,
                },
                "anti_shake_angular_velocity": {
                    "equation": (
                        "sum_k_in_wrists_and_head(||omega_k||_2^2*"
                        "1[||omega_k||>1.5])"
                    ),
                    "links": ["wrists", "head"],
                    "threshold": 1.5,
                    "weight": -0.005,
                },
                "feet_acceleration": {
                    "equation": (
                        "sum_k_in_ankles(||q_double_dot_t_k||_2^2)"
                    ),
                    "links": ["ankles"],
                    "weight": -0.0000025,
                },
            },
            "notation": {
                "B": "tracked_body_links",
                "K": "head_both_wrists_and_both_ankles",
                "goal_superscript": "target",
                "current_superscript": "policy_state",
                "rel": "relative_to_root_frame",
            },
            "source_sections": ["Table S3"],
            "applicability_boundary": (
                "reported_joint_sonic_recipe_not_isolated_evidence_that_each_"
                "term_or_weight_is_optimal_or_portable_to_another_task"
            ),
        },
    ),
    SonicCapabilitySpec(
        key="sonic_domain_randomization_ranges",
        name="SONIC domain-randomization and command-perturbation ranges",
        description=(
            "Table S4 reports the physical randomization, external-push, and "
            "target-command perturbation distributions used during training."
        ),
        paper_role="primary_source_domain_randomization",
        source_locator=f"{SONIC_SOURCE_URL}#A0.T4",
        parameters={
            "distribution": "uniform_inclusive_range_as_reported",
            "sampling_cadence": "not_reported_in_table_s4",
            "cross_parameter_independence": "not_reported_in_table_s4",
            "physical_parameters": {
                "static_friction_coefficient": [0.3, 1.6],
                "dynamic_friction_coefficient": [0.3, 1.2],
                "restitution_coefficient": [0.0, 0.5],
                "default_joint_position_additive_offset": [-0.01, 0.01],
                "base_com_offset": {
                    "x": [-0.075, 0.075],
                    "y": [-0.1, 0.1],
                    "z": [-0.1, 0.1],
                },
            },
            "root_velocity_perturbations": {
                "linear_velocity": {
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                    "z": [-0.2, 0.2],
                },
                "push_duration_s": [1.0, 3.0],
                "angular_velocity": {
                    "roll": [-0.52, 0.52],
                    "pitch": [-0.52, 0.52],
                    "yaw": [-0.78, 0.78],
                },
            },
            "target_motion_perturbations": {
                "position_jitter": {
                    "displayed_vector_expression": "U[-0.05,0.05]^3",
                    "axis_specific_parenthetical": {
                        "x_y": "plus_or_minus_0.05",
                        "z": "plus_or_minus_0.01",
                    },
                    "x": [-0.05, 0.05],
                    "y": [-0.05, 0.05],
                    "z": [-0.01, 0.01],
                },
                "orientation_jitter": {
                    "roll": [-0.1, 0.1],
                    "pitch": [-0.1, 0.1],
                    "yaw": [-0.2, 0.2],
                },
                "linear_velocity_jitter": {
                    "displayed_vector_expression": "U[-0.5,0.5]^3",
                    "axis_specific_parenthetical": {
                        "x_y": "plus_or_minus_0.5",
                        "z": "plus_or_minus_0.2",
                    },
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                    "z": [-0.2, 0.2],
                },
                "angular_velocity_jitter": {
                    "roll": [-0.52, 0.52],
                    "pitch": [-0.52, 0.52],
                    "yaw": [-0.78, 0.78],
                },
                "joint_position_jitter": [-0.1, 0.1],
            },
            "source_sections": ["3.2 Domain Randomization", "Table S4"],
            "applicability_boundary": (
                "sonic_g1_training_ranges_not_validated_defaults_for_another_"
                "robot_task_simulator_or_controller"
            ),
        },
    ),
    SonicCapabilitySpec(
        key="sonic_separate_kinematic_motion_planner",
        name="SONIC separate generative kinematic motion planner",
        description=(
            "A separate latent masked-token in-betweening model converts "
            "commands and endpoint keyframes into short-horizon kinematic "
            "reference motions for the tracking controller."
        ),
        paper_role="primary_source_separate_kinematic_planner",
        source_locator=f"{SONIC_SOURCE_URL}#S3.SS3",
        parameters={
            "separate_from_tracking_policy": True,
            "planner_output": "kinematic_reference_motion",
            "latent_space_distinct_from_fsq": True,
            "training_task": "autoregressive_motion_inbetweening",
            "training_data_shared_with_tracker": True,
            "motion_representation": [
                "pelvis_relative_joint_positions",
                "global_joint_rotations",
            ],
            "temporal_downsample": 4,
            "segment_duration_s": [0.8, 2.4],
            "segment_duration_selected_by_planner": True,
            "endpoint_constraint_frames": 4,
            "generation_method": "iterative_masked_token_prediction",
            "training_mask_fraction_range": [1.0, 0.0],
            "inference_token_finalization_schedule": (
                "1-cos((pi/2)*(iteration/max_iterations))"
            ),
            "replan_period_s": 0.1,
            "immediate_replan_on_command_update": True,
            "inference_latency_ms": {
                "standard_laptop": "<5",
                "jetson_orin": "~12",
            },
            "root_trajectory_filter": {
                "model": "critically_damped_spring",
                "filtered_quantities": [
                    "pelvis_x",
                    "pelvis_y",
                    "projected_pelvis_heading",
                ],
                "position_damping_coefficient": "5*ln(2)",
                "heading_damping_coefficient": "20*ln(2)",
                "velocity_target_horizon_s": 1.0,
            },
            "source_sections": [
                "2.2 Interactive Motion Control",
                "3.3 Generative Kinematic Motion Planner",
                "Equations 5-8",
            ],
        },
    ),
    SonicCapabilitySpec(
        key="sonic_vla_interface",
        name="SONIC VLA action interfaces",
        description=(
            "The reported GR00T N1.5 integrations place vision-language "
            "reasoning upstream of SONIC: one interface emits 3-point "
            "teleoperation commands, while whole-body tasks emit a 64D body "
            "token plus 14 hand-joint values."
        ),
        paper_role="primary_source_vla_interface_and_results",
        source_locator=f"{SONIC_SOURCE_URL}#S2.SS5",
        parameters={
            "upstream_vla_model": "GR00T N1.5",
            "sonic_actor_direct_camera_input": False,
            "three_point_interface_outputs": [
                "head_se3_pose",
                "left_wrist_se3_pose",
                "right_wrist_se3_pose",
                "hand_joint_angles",
                "waist_height",
                "locomotion_mode",
                "desired_root_velocity_and_heading",
            ],
            "whole_body_action_dimensions": 78,
            "whole_body_action_breakdown": {
                "universal_motion_token": 64,
                "hand_joint_angles": 14,
            },
            "reported_task_rows": {
                "apple_to_plate": {
                    "interface": "three_point",
                    "training_trajectories": 300,
                    "trials": 20,
                    "success_percent": 90,
                },
                "object_pickup_carrot": {
                    "interface": "whole_body",
                    "shared_multi_object_training_trajectories": 3900,
                    "trials": 20,
                    "success_percent": 75,
                },
                "object_pickup_scrub": {
                    "interface": "whole_body",
                    "shared_multi_object_training_trajectories": 3900,
                    "trials": 20,
                    "success_percent": 95,
                },
                "open_trash_can_with_foot": {
                    "interface": "whole_body",
                    "training_trajectories": 200,
                    "trials": 10,
                    "success_percent": 70,
                },
                "soda_can_to_trash_can": {
                    "interface": "whole_body",
                    "multi_object_training_trajectories": 1000,
                    "trials": 10,
                    "success_percent": 60,
                },
                "drill_and_box_relocation": {
                    "interface": "whole_body",
                    "training_trajectories": 300,
                    "trials": 10,
                    "success_percent": 70,
                },
            },
            "reported_five_task_average_success_percent": 75,
            "three_task_action_space_ablation_average_success_percent": {
                "fsq_token": 68,
                "explicit_smpl_pose": 27,
            },
            "source_sections": [
                "2.5 Foundation-Model-Driven Loco-manipulation",
                "Table 1",
                "S1.4 VLA Integration Details",
            ],
        },
    ),
    SonicCapabilitySpec(
        key="sonic_evaluation_protocol_results_and_limits",
        name="SONIC evaluation protocol, reported results, and limitations",
        description=(
            "The paper evaluates local motion tracking on held-out and "
            "external motion sets plus 124 physical-robot sequences, while "
            "explicitly limiting its claims on global trajectory tracking, "
            "formal safety, energy efficiency, and extreme motions."
        ),
        paper_role="primary_source_evaluation_and_limitations",
        source_locator=f"{SONIC_SOURCE_URL}#S2.SS1",
        scope="external_paper_evaluation_claim",
        parameters={
            "primary_evaluation_simulator": "Isaac Lab",
            "baseline_comparison_simulator": "MuJoCo",
            "test_sets": {
                "test_content": {"motions": 7016, "hours": 15},
                "test_repetition": {"motions": 9395, "hours": 12},
                "phuma": {
                    "motions": 68326,
                    "different_retargeting_pipeline": True,
                },
            },
            "failure_threshold": {
                "root_or_end_effector_height_deviation_m": 0.25,
            },
            "tracking_metric_scope": "local_root_relative",
            "global_root_position_is_not_tracked": True,
            "mujoco_cross_dataset_success_percent": {
                "test_content": 98.5,
                "test_repetition": 99.2,
                "phuma": 97.2,
            },
            "cross_dataset_comparison_is_not_data_matched": True,
            "reported_local_mpjpe_mm": 23.7,
            "physical_robot_evaluation": {
                "motion_sequences": 124,
                "successful_sequences": 123,
                "success_percent": 99.2,
                "local_mpjpe_mm": 25.7,
            },
            "paper_stated_limits": [
                "no_formal_treatment_of_safety",
                "no_formal_treatment_of_energy_efficiency",
                "may_lose_balance_under_extreme_conditions_or_dynamic_motions",
                "sustained_or_complex_ground_contact_remains_challenging",
            ],
            "reported_example_failures": [
                "zombie_crawl",
                "cross_legged_sit",
            ],
            "source_sections": [
                "2.1 Motion Tracking",
                "2.6 Discussion",
                "3 Study Design",
                "S2.2 Qualitative Analysis of Success and Failure Motions",
            ],
            "additional_source_locators": [
                f"{SONIC_SOURCE_URL}#S2.SS6",
                f"{SONIC_SOURCE_URL}#A0.SS2.SSS2",
            ],
        },
    ),
)


class SonicCapabilityMapError(ValueError):
    """The source-pinned SONIC catalog is incomplete or inconsistent."""


_REQUIRED_PARAMETER_KEYS: dict[str, set[str]] = {
    "sonic_universal_controller_contract": {
        "controlled_dof",
        "control_rate_hz",
        "action_semantics",
        "proprioceptive_history_steps",
    },
    "sonic_fsq_interface_and_training_loss": {
        "token_count",
        "token_dimensions",
        "quantization_levels_per_dimension",
        "flattened_body_token_dimensions",
        "separate_hand_joint_dimensions",
        "total_loss",
    },
    "sonic_scale_and_training_recipe": {
        "filtered_hours",
        "training_clips",
        "training_frames",
        "largest_model_parameters",
        "training_iterations",
    },
    "sonic_public_bones_seed_release": {
        "dataset_url",
        "annotated_motion_sequences",
        "duration_hours",
        "actors",
        "formats_reported_by_paper",
    },
    "sonic_motion_tracking_reward_design": {
        "tracking_terms",
        "penalty_terms",
        "notation",
        "applicability_boundary",
    },
    "sonic_domain_randomization_ranges": {
        "physical_parameters",
        "root_velocity_perturbations",
        "target_motion_perturbations",
        "applicability_boundary",
    },
    "sonic_separate_kinematic_motion_planner": {
        "latent_space_distinct_from_fsq",
        "temporal_downsample",
        "segment_duration_s",
        "replan_period_s",
    },
    "sonic_vla_interface": {
        "whole_body_action_dimensions",
        "whole_body_action_breakdown",
        "reported_task_rows",
    },
    "sonic_evaluation_protocol_results_and_limits": {
        "test_sets",
        "failure_threshold",
        "cross_dataset_comparison_is_not_data_matched",
        "paper_stated_limits",
    },
}


def sonic_capability_by_key(key: str) -> SonicCapabilitySpec:
    for spec in SONIC_CAPABILITIES:
        if spec.key == key:
            return spec
    raise KeyError(key)


def validate_sonic_capability_catalog() -> None:
    """Fail closed when reviewed source, status, or parameter keys drift."""

    keys = [spec.key for spec in SONIC_CAPABILITIES]
    if len(keys) != len(set(keys)):
        raise SonicCapabilityMapError("SONIC capability keys are not unique")
    if set(keys) != set(_REQUIRED_PARAMETER_KEYS):
        raise SonicCapabilityMapError(
            "SONIC capability inventory drifted from its reviewed categories"
        )
    for spec in SONIC_CAPABILITIES:
        if spec.status is not CapabilityStatus.UNSUPPORTED:
            raise SonicCapabilityMapError(
                f"SONIC capability {spec.key!r} must remain unsupported until "
                "reviewed executable evidence exists"
            )
        if spec.code_evidence:
            raise SonicCapabilityMapError(
                f"unsupported SONIC capability {spec.key!r} cannot claim "
                "RewardSculptor code evidence"
            )
        if not spec.paper_role.startswith("primary_source_"):
            raise SonicCapabilityMapError(
                f"SONIC capability {spec.key!r} has an invalid paper role"
            )
        if not spec.source_locator.startswith(f"{SONIC_SOURCE_URL}#"):
            raise SonicCapabilityMapError(
                f"SONIC capability {spec.key!r} is not pinned to arXiv v4"
            )
        missing = _REQUIRED_PARAMETER_KEYS[spec.key] - spec.parameters.keys()
        if missing:
            raise SonicCapabilityMapError(
                f"SONIC capability {spec.key!r} is missing parameters: "
                f"{sorted(missing)!r}"
            )
        try:
            json.dumps(spec.parameters, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise SonicCapabilityMapError(
                f"SONIC capability {spec.key!r} has non-JSON parameters"
            ) from exc


def _owned_parameters(node: ResearchCapability) -> bool:
    return node.parameters.get("catalog_owner") == SONIC_CATALOG_OWNER


def _ensure_edge(store: SculptorKG, edge: Edge) -> bool:
    """Write ``edge`` only when its data changed, preserving idempotence."""

    for current, other_id in store.neighbors(
        edge.src,
        relation=edge.relation,
        direction="out",
    ):
        if other_id == edge.dst and current.data == edge.data:
            return False
    store.add_edge(edge)
    return True


def materialize_sonic_capability_map(store: SculptorKG) -> dict[str, int]:
    """Materialize the reviewed SONIC v4 capability/status subgraph.

    The exact SONIC paper node must already exist.  Materialization removes
    stale nodes and edges owned by an older version of this catalog, deletes
    contradictory implementation-status edges, and leaves unrelated graph
    content untouched.
    """

    validate_sonic_capability_catalog()
    paper_id = make_paper_id(SONIC_ARXIV_ID)
    paper = store.get_node(paper_id)
    if not isinstance(paper, Paper) or paper.arxiv_id != SONIC_ARXIV_ID:
        raise SonicCapabilityMapError(
            "cannot materialize SONIC capability map before its exact paper "
            f"node exists: {paper_id!r}"
        )

    status_id = make_implementation_status_id(
        CapabilityStatus.UNSUPPORTED.value
    )
    expected_capability_ids = {spec.node_id for spec in SONIC_CAPABILITIES}
    expected_owned_edges = {
        (paper_id, spec.node_id, Relation.GROUNDS_CAPABILITY)
        for spec in SONIC_CAPABILITIES
    } | {
        (spec.node_id, status_id, Relation.HAS_IMPLEMENTATION_STATUS)
        for spec in SONIC_CAPABILITIES
    }
    stale_capabilities_removed = 0
    stale_edges_removed = 0

    with store.transaction():
        for edge in list(store.all_edges()):
            identity = (edge.src, edge.dst, edge.relation)
            if (
                edge.data.get("catalog_owner") == SONIC_CATALOG_OWNER
                and identity not in expected_owned_edges
            ):
                stale_edges_removed += int(
                    store.delete_edge(edge.src, edge.dst, edge.relation)
                )

        for node in list(
            store.find_nodes(kind=ResearchCapability.kind)
        ):
            if (
                _owned_parameters(node)
                and node.id not in expected_capability_ids
            ):
                stale_capabilities_removed += int(store.delete_node(node.id))

        store.add_node(
            ImplementationStatus(
                id=status_id,
                status=CapabilityStatus.UNSUPPORTED.value,
                definition=STATUS_DEFINITIONS[CapabilityStatus.UNSUPPORTED],
            )
        )

        for spec in SONIC_CAPABILITIES:
            parameters = copy.deepcopy(spec.parameters)
            parameters.update(
                {
                    "catalog_owner": SONIC_CATALOG_OWNER,
                    "source_arxiv_id": SONIC_ARXIV_ID,
                    "source_version": SONIC_PAPER_VERSION,
                    "source_locator": spec.source_locator,
                }
            )
            store.add_node(
                ResearchCapability(
                    id=spec.node_id,
                    name=spec.name,
                    description=spec.description,
                    scope=spec.scope,
                    code_evidence=[],
                    provenance=PROVENANCE_PAPER_CLAIM,
                    parameters=parameters,
                )
            )

            for current, other_id in list(
                store.neighbors(
                    spec.node_id,
                    relation=Relation.HAS_IMPLEMENTATION_STATUS,
                    direction="out",
                )
            ):
                if other_id != status_id:
                    stale_edges_removed += int(
                        store.delete_edge(
                            current.src,
                            current.dst,
                            current.relation,
                        )
                    )
            for current, other_id in list(
                store.neighbors(
                    spec.node_id,
                    relation=Relation.GROUNDS_CAPABILITY,
                    direction="in",
                )
            ):
                if other_id != paper_id:
                    stale_edges_removed += int(
                        store.delete_edge(
                            current.src,
                            current.dst,
                            current.relation,
                        )
                    )

            _ensure_edge(
                store,
                Edge(
                    src=spec.node_id,
                    dst=status_id,
                    relation=Relation.HAS_IMPLEMENTATION_STATUS,
                    data={"catalog_owner": SONIC_CATALOG_OWNER},
                ),
            )
            _ensure_edge(
                store,
                Edge(
                    src=paper_id,
                    dst=spec.node_id,
                    relation=Relation.GROUNDS_CAPABILITY,
                    data={
                        "catalog_owner": SONIC_CATALOG_OWNER,
                        "paper_role": spec.paper_role,
                        "source_locator": spec.source_locator,
                        "paper_version": SONIC_PAPER_VERSION,
                    },
                ),
            )

    return {
        "capabilities": len(SONIC_CAPABILITIES),
        "statuses": 1,
        "paper_edges": len(SONIC_CAPABILITIES),
        "status_edges": len(SONIC_CAPABILITIES),
        "stale_capabilities_removed": stale_capabilities_removed,
        "stale_edges_removed": stale_edges_removed,
    }


validate_sonic_capability_catalog()
