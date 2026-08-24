"""Reviewed OGMP capability claims as a small, materializable graph.

Paper extraction answers what a paper says; it must not answer whether this
repository executes the paper's mechanism.  This module is the deliberately
small bridge between those two questions.  Every capability has exactly one of
three statuses:

``implemented``
    The named executable path affects training or rollout behavior.
``metadata_only``
    The concept is represented, validated, or reported, but does not control
    runtime handover, policy observations, or selection.
``unsupported``
    The paper concept is not implemented by the current runtime.

The catalog is also the source for API and rollout-disclosure fields.  That
keeps the knowledge graph, researcher-facing receipt, and persisted diagnostic
from drifting into three independent product claims.
"""

from __future__ import annotations

import enum
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sculptor.kg.schema import (
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


OGMP_ARXIV_ID = "2403.04205"
PREFERENCED_OGMP_ARXIV_ID = "2410.01030"


class CapabilityStatus(str, enum.Enum):
    IMPLEMENTED = "implemented"
    METADATA_ONLY = "metadata_only"
    UNSUPPORTED = "unsupported"


STATUS_DEFINITIONS: dict[CapabilityStatus, str] = {
    CapabilityStatus.IMPLEMENTED: (
        "Consumed by an executable training or rollout path and therefore "
        "able to affect runtime behavior."
    ),
    CapabilityStatus.METADATA_ONLY: (
        "Represented, validated, or reported, but not an authority for "
        "runtime handover, policy input, reward dispatch, or selection."
    ),
    CapabilityStatus.UNSUPPORTED: (
        "A paper concept that the current RewardSculptor runtime does not "
        "execute."
    ),
}


@dataclass(frozen=True)
class CapabilitySpec:
    key: str
    name: str
    description: str
    status: CapabilityStatus
    paper_arxiv_ids: tuple[str, ...]
    paper_role: str
    scope: str
    code_evidence: tuple[str, ...] = ()
    api_flag: str | None = None
    diagnostic_key: str | None = None

    @property
    def node_id(self) -> str:
        return make_research_capability_id(self.key)


# Do not add a capability merely because a schema has a suggestive field.  An
# implemented claim below needs reviewed executable evidence; an unsupported
# claim intentionally has none.  ``paper_role`` prevents the narrower fixed
# phase-window adaptation from being misread as a mechanism OGMP introduced.
OGMP_CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        key="fixed_linear_phase_window_dispatch",
        name="Fixed linear phase-window dispatch",
        description=(
            "Derive ordered windows from one immutable composed clip and "
            "dispatch phase-specific reward code by per-environment elapsed "
            "episode time."
        ),
        status=CapabilityStatus.IMPLEMENTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID,),
        paper_role="rewardsculptor_adaptation_boundary",
        scope="rewardsculptor_adaptation",
        code_evidence=(
            "sculptor.modes:modes_from_composition",
            "sculptor.modes:build_mode_execution_manifest",
            "sculptor.mode_rewards:generate_mode_reward_scaffold",
        ),
    ),
    CapabilitySpec(
        key="per_mode_reward_authoring_and_scope",
        name="Per-mode reward authoring and scope",
        description=(
            "Author and validate separate reward bodies, then mask each body "
            "to its fixed execution window."
        ),
        status=CapabilityStatus.IMPLEMENTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID,),
        paper_role="ogmp_inspired_adaptation",
        scope="rewardsculptor_adaptation",
        code_evidence=(
            "sculptor.mode_rewards:author_mode",
            "sculptor.mode_rewards:promote_mode_reward",
            "sculptor.mode_rewards:validate_mode_reward_source",
        ),
    ),
    CapabilitySpec(
        key="immutable_mode_execution_admission_and_diagnostics",
        name="Immutable mode execution admission and diagnostics",
        description=(
            "Bind reference, graph, emitted schedule, robot, reward, and "
            "context before training; report digest-bound per-window rollout "
            "evidence without granting it fitness authority."
        ),
        status=CapabilityStatus.IMPLEMENTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID,),
        paper_role="rewardsculptor_reproducibility_extension",
        scope="rewardsculptor_extension",
        code_evidence=(
            "sculptor.mode_rewards:build_mode_reward_binding",
            "sculptor.mode_rewards:mode_reward_binding_errors",
            "sculptor.eval.mode_metrics:build_mode_diagnostics",
        ),
    ),
    CapabilitySpec(
        key="transition_guard_declarations",
        name="Transition guard declarations",
        description=(
            "Phase and predicate guards can be stored, structurally "
            "validated, and checked against completed rollout evidence; they "
            "do not drive runtime handover."
        ),
        status=CapabilityStatus.METADATA_ONLY,
        paper_arxiv_ids=(OGMP_ARXIV_ID, PREFERENCED_OGMP_ARXIV_ID),
        paper_role="paper_mechanism_represented_as_metadata",
        scope="paper_mechanism",
        code_evidence=(
            "sculptor.modes:Guard",
            "sculptor.modes:validate_mode_graph",
            "sculptor.eval.mode_metrics:check_transitions",
        ),
    ),
    CapabilitySpec(
        key="mode_success_predicate_declarations",
        name="Mode success-predicate declarations",
        description=(
            "A mode can retain a success-predicate string for authoring and "
            "diagnostic context, but that field is not populated by "
            "composition and is not production readiness evidence."
        ),
        status=CapabilityStatus.METADATA_ONLY,
        paper_arxiv_ids=(PREFERENCED_OGMP_ARXIV_ID,),
        paper_role="hybrid_automaton_extension_metadata",
        scope="rewardsculptor_extension",
        code_evidence=(
            "sculptor.modes:Mode",
            "sculptor.eval.mode_metrics:mode_goal_text",
        ),
    ),
    CapabilitySpec(
        key="online_receding_horizon_oracle",
        name="Online receding-horizon oracle",
        description=(
            "Query a closed-loop oracle from the current state to regenerate "
            "a finite-horizon state reference online."
        ),
        status=CapabilityStatus.UNSUPPORTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID, PREFERENCED_OGMP_ARXIV_ID),
        paper_role="paper_mechanism",
        scope="paper_mechanism",
        api_flag="closed_loop_receding_horizon_oracle",
        diagnostic_key="closed_loop_oracle",
    ),
    CapabilitySpec(
        key="rho_bounded_permissible_state_exploration",
        name="Rho-bounded permissible-state exploration",
        description=(
            "Bound policy exploration around the oracle reference by rho and "
            "terminate trajectories that leave the permissible state set."
        ),
        status=CapabilityStatus.UNSUPPORTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID, PREFERENCED_OGMP_ARXIV_ID),
        paper_role="paper_mechanism",
        scope="paper_mechanism",
        api_flag="rho_bounded_exploration",
        diagnostic_key="rho_bounded_exploration",
    ),
    CapabilitySpec(
        key="learned_mode_latent_and_task_feedback_conditioning",
        name="Learned mode-latent and task-feedback policy conditioning",
        description=(
            "Condition one policy on a learned task-vital mode latent, clock, "
            "and task feedback rather than dispatching reward by time alone."
        ),
        status=CapabilityStatus.UNSUPPORTED,
        paper_arxiv_ids=(OGMP_ARXIV_ID, PREFERENCED_OGMP_ARXIV_ID),
        paper_role="paper_mechanism",
        scope="paper_mechanism",
        api_flag="policy_mode_conditioning",
        diagnostic_key="mode_conditioned_policy",
    ),
    CapabilitySpec(
        key="runtime_predicate_or_branch_transition_execution",
        name="Runtime predicate or branch transition execution",
        description=(
            "Use environment feedback or predicates as the live authority "
            "for nonlinear mode handover and branching."
        ),
        status=CapabilityStatus.UNSUPPORTED,
        paper_arxiv_ids=(PREFERENCED_OGMP_ARXIV_ID,),
        paper_role="paper_mechanism",
        scope="paper_mechanism",
        api_flag="runtime_transition_guards",
        diagnostic_key="predicate_or_branch_executor",
    ),
    CapabilitySpec(
        key="preference_conditioned_oracle_or_policy",
        name="Preference-conditioned oracle or policy",
        description=(
            "Select or condition loco-manipulation behavior with an explicit "
            "user/task preference signal."
        ),
        status=CapabilityStatus.UNSUPPORTED,
        paper_arxiv_ids=(PREFERENCED_OGMP_ARXIV_ID,),
        paper_role="paper_mechanism",
        scope="paper_mechanism",
        api_flag="preference_conditioning",
    ),
)


class CapabilityMapError(ValueError):
    """The reviewed capability map is internally inconsistent."""


def capability_by_key(key: str) -> CapabilitySpec:
    for spec in OGMP_CAPABILITIES:
        if spec.key == key:
            return spec
    raise KeyError(key)


def validate_ogmp_capability_catalog(*, resolve_symbols: bool = False) -> None:
    """Fail closed if the reviewed mapping is ambiguous or stale.

    ``resolve_symbols`` imports each evidence target and verifies the named
    attribute still exists.  CI enables it; ordinary API reads avoid the
    import cost because they consume the already-reviewed immutable tuple.
    """

    keys = [spec.key for spec in OGMP_CAPABILITIES]
    if len(keys) != len(set(keys)):
        raise CapabilityMapError("OGMP capability keys are not unique")
    if {spec.status for spec in OGMP_CAPABILITIES} != set(CapabilityStatus):
        raise CapabilityMapError(
            "OGMP map must exercise implemented, metadata_only, and "
            "unsupported statuses"
        )
    for spec in OGMP_CAPABILITIES:
        if not spec.paper_arxiv_ids:
            raise CapabilityMapError(
                f"capability {spec.key!r} has no grounding paper"
            )
        unknown_papers = set(spec.paper_arxiv_ids) - {
            OGMP_ARXIV_ID,
            PREFERENCED_OGMP_ARXIV_ID,
        }
        if unknown_papers:
            raise CapabilityMapError(
                f"capability {spec.key!r} names unknown papers: "
                f"{sorted(unknown_papers)!r}"
            )
        if spec.status is CapabilityStatus.UNSUPPORTED:
            if spec.code_evidence:
                raise CapabilityMapError(
                    f"unsupported capability {spec.key!r} cannot claim "
                    "executable evidence"
                )
        elif not spec.code_evidence:
            raise CapabilityMapError(
                f"{spec.status.value} capability {spec.key!r} needs "
                "executable evidence"
            )
        if resolve_symbols:
            for reference in spec.code_evidence:
                _resolve_evidence_symbol(reference)


def _resolve_evidence_symbol(reference: str) -> object:
    try:
        module_name, qualified_name = reference.split(":", 1)
        value: object = importlib.import_module(module_name)
        for part in qualified_name.split("."):
            value = getattr(value, part)
        return value
    except (AttributeError, ImportError, ValueError) as exc:
        raise CapabilityMapError(
            f"capability evidence symbol {reference!r} is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def implementation_status_map() -> dict[str, str]:
    return {spec.key: spec.status.value for spec in OGMP_CAPABILITIES}


def unsupported_mode_diagnostic_keys() -> list[str]:
    """Legacy-stable disclosure keys persisted in mode diagnostics."""

    by_key = {
        spec.diagnostic_key: spec
        for spec in OGMP_CAPABILITIES
        if spec.diagnostic_key is not None
    }
    order = (
        "closed_loop_oracle",
        "rho_bounded_exploration",
        "predicate_or_branch_executor",
        "mode_conditioned_policy",
    )
    if set(by_key) != set(order):
        raise CapabilityMapError(
            "mode diagnostic disclosure keys drifted from the persisted "
            f"contract: {sorted(by_key)!r}"
        )
    if any(
        by_key[key].status is not CapabilityStatus.UNSUPPORTED for key in order
    ):
        raise CapabilityMapError(
            "a mode diagnostic not_implemented key is no longer unsupported"
        )
    return list(order)


def mode_api_capability_summary() -> dict[str, object]:
    """Researcher-facing summary used by the existing modes endpoint."""

    flags = {
        spec.api_flag: spec.status is CapabilityStatus.IMPLEMENTED
        for spec in OGMP_CAPABILITIES
        if spec.api_flag is not None
    }
    return {
        "kind": "phase_window_reference_scaffold",
        "paper_alignment": "ogmp_inspired",
        "dispatch_authority": "episode_time_window",
        "reference_generator": "fixed_composed_clip",
        **flags,
        "implementation_status": implementation_status_map(),
        "summary": (
            "Fixed composite-reference windows gate phase-specific reward "
            "terms. Transition guards are inspectable metadata; they do not "
            "currently drive the policy or runtime handover."
        ),
    }


def materialize_ogmp_capability_map(store: SculptorKG) -> dict[str, int]:
    """Write the reviewed capability/status subgraph into ``store``.

    Both paper nodes must already exist.  This function never fabricates or
    overwrites literature metadata merely to avoid a dangling edge.  Repeated
    materialization is idempotent, and stale status edges are removed before
    the one authoritative status is written.
    """

    validate_ogmp_capability_catalog()
    paper_ids = {
        make_paper_id(arxiv_id)
        for spec in OGMP_CAPABILITIES
        for arxiv_id in spec.paper_arxiv_ids
    }
    missing = sorted(
        paper_id
        for paper_id in paper_ids
        if not isinstance(store.get_node(paper_id), Paper)
    )
    if missing:
        raise CapabilityMapError(
            "cannot materialize OGMP capability map before its paper nodes "
            f"exist: {missing!r}"
        )

    status_nodes = {
        status: ImplementationStatus(
            id=make_implementation_status_id(status.value),
            status=status.value,
            definition=STATUS_DEFINITIONS[status],
        )
        for status in CapabilityStatus
    }
    with store.transaction():
        for node in status_nodes.values():
            store.add_node(node)
        for spec in OGMP_CAPABILITIES:
            node = ResearchCapability(
                id=spec.node_id,
                name=spec.name,
                description=spec.description,
                scope=spec.scope,
                code_evidence=list(spec.code_evidence),
            )
            store.add_node(node)
            for edge, other_id in store.neighbors(
                node.id,
                relation=Relation.HAS_IMPLEMENTATION_STATUS,
                direction="out",
            ):
                if other_id != status_nodes[spec.status].id:
                    store.delete_edge(edge.src, edge.dst, edge.relation)
            store.add_edge(
                Edge(
                    src=node.id,
                    dst=status_nodes[spec.status].id,
                    relation=Relation.HAS_IMPLEMENTATION_STATUS,
                )
            )
            for arxiv_id in spec.paper_arxiv_ids:
                store.add_edge(
                    Edge(
                        src=make_paper_id(arxiv_id),
                        dst=node.id,
                        relation=Relation.GROUNDS_CAPABILITY,
                        data={"paper_role": spec.paper_role},
                    )
                )
    return {
        "capabilities": len(OGMP_CAPABILITIES),
        "statuses": len(status_nodes),
        "paper_edges": sum(
            len(spec.paper_arxiv_ids) for spec in OGMP_CAPABILITIES
        ),
        "status_edges": len(OGMP_CAPABILITIES),
    }


validate_ogmp_capability_catalog()
