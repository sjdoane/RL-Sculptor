"""sculptor/kg/schema.py — Node / edge dataclasses for the knowledge graph.

Design notes
------------
- **Dataclass-per-node-type** rather than a single generic row. Gives us typed
  fields and lets `dataclasses.asdict` drive the JSON serialization in the
  store. New node types are added by writing a new `@dataclass` + a string
  literal for `.kind`.
- **Natural IDs** (e.g. `paper:1707.06347`, `technique:ramiel-r-keep`) are
  preferred over UUIDs so idempotent re-ingest is trivial: the id collision
  IS the dedupe check. The store treats `id` as the primary key.
- **Edges are typed by `Relation` enum, not free strings**, so the diagnoser
  and editor can pattern-match on `relation == Relation.ADDRESSES` rather
  than `== "addresses"`. Unknown relation strings on disk surface as
  deserialization errors instead of silent typos.
- **Node.kind** duplicates the class name so the store can round-trip a row
  through `SculptorKG.get_node` without importing every dataclass class.

IDs must be globally unique across node types. Use the helpers
`make_paper_id`, `make_technique_id`, etc. to build them consistently.

- **§Agentic-data upgrade 1 (provenance trust tiers)**: the node kinds the
  diagnoser/decomposer actually retrieve and render to Claude (Paper,
  Technique, FailureMode, RewardComponent, RunCase) carry a `provenance`
  field — one of "observed_run" | "attested_artifact" | "paper_claim" |
  "llm_extraction" | "seed" — so a retrieval-time renderer can tell Claude
  WHERE a claim came
  from (this system's own runs vs. a paper's claims vs. an LLM inference
  vs. a hand-seeded fact). Stored in the JSON data blob like any other
  field, so no store-schema change is needed. Backward compatible for
  free: `row_to_node` calls `cls(id=node_id, **data)`, and a dataclass
  field with a default is simply omitted from `**data` on old rows — the
  type's own default fires. Per-type defaults (see each dataclass):
  RunCase → "observed_run" (it IS a recorded run); Technique /
  FailureMode / RewardComponent → "paper_claim" (materialized from
  extraction with paper evidence attached); Paper → "seed" (the 46-paper
  KG seed set / ingest entry point).
"""

from __future__ import annotations

import dataclasses
import enum
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ── Provenance trust tiers ───────────────────────────────────────────────────
#: §Agentic-data upgrade 1. Free-string rather than an enum (like Relation)
#: because provenance is advisory metadata for prompt rendering, not a
#: graph-walk key — an unrecognized value on disk should degrade gracefully
#: (render as "llm-inferred", the least-trusted tag) rather than raise.
PROVENANCE_OBSERVED_RUN = "observed_run"
PROVENANCE_ATTESTED_ARTIFACT = "attested_artifact"
PROVENANCE_PAPER_CLAIM = "paper_claim"
PROVENANCE_LLM_EXTRACTION = "llm_extraction"
PROVENANCE_SEED = "seed"

#: Compact per-item tag rendered inline in prompt context (diagnose.py's
#: literature block, cases.py's case-memory block) so Claude can see, at a
#: glance, WHERE each claim came from. `observed_run` gets an emphatic tag
#: ("in THIS project's own runs") because it is the highest-trust tier —
#: see the header line each renderer prepends: observations outrank paper
#: claims when they conflict.
_EVIDENCE_TAGS: dict[str, str] = {
    PROVENANCE_OBSERVED_RUN: "[evidence: observed run]",
    PROVENANCE_ATTESTED_ARTIFACT: "[evidence: content-attested artifact]",
    PROVENANCE_PAPER_CLAIM: "[evidence: paper]",
    PROVENANCE_SEED: "[evidence: seed]",
    PROVENANCE_LLM_EXTRACTION: "[evidence: llm-inferred]",
}


def evidence_tag(provenance: str | None) -> str:
    """Compact `[evidence: ...]` tag for a node's provenance value.
    Unrecognized/None provenance (e.g. a future value not yet known to
    this module) degrades to the LEAST-trusted tag rather than raising —
    provenance is advisory rendering metadata, not a graph-integrity
    constraint."""
    return _EVIDENCE_TAGS.get(
        provenance or "", _EVIDENCE_TAGS[PROVENANCE_LLM_EXTRACTION])


#: Trust ranking for provenance merges (higher = more trusted). Unknown
#: values rank lowest, same as the rendering fallback above.
_PROVENANCE_TRUST: dict[str, int] = {
    PROVENANCE_OBSERVED_RUN: 4,
    PROVENANCE_ATTESTED_ARTIFACT: 3,
    PROVENANCE_PAPER_CLAIM: 2,
    PROVENANCE_SEED: 1,
    PROVENANCE_LLM_EXTRACTION: 0,
}


def merge_provenance(existing: str | None, incoming: str | None) -> str:
    """Pick the MORE-trusted of two provenance values when merging node
    data (e.g. a diagnoser-flagged FailureMode stub later attested by a
    paper extraction upgrades llm_extraction -> paper_claim; the reverse
    never downgrades). None/unknown values rank as least-trusted."""
    e = existing or PROVENANCE_LLM_EXTRACTION
    i = incoming or PROVENANCE_LLM_EXTRACTION
    return e if _PROVENANCE_TRUST.get(e, 0) >= _PROVENANCE_TRUST.get(i, 0) else i


# ── Relation enum ───────────────────────────────────────────────────────────
class Relation(str, enum.Enum):
    CITES          = "CITES"
    INTRODUCES     = "INTRODUCES"
    ADDRESSES      = "ADDRESSES"
    USES           = "USES"
    EVALUATES_ON   = "EVALUATES_ON"
    REPORTS        = "REPORTS"
    IMPROVES_OVER  = "IMPROVES_OVER"
    INSTANTIATES   = "INSTANTIATES"   # §Ship 37: RunCase → FailureMode
    INITIALIZED_FROM = "INITIALIZED_FROM"  # TrainingRun → PolicyArtifact
    TRACKS         = "TRACKS"         # TrainingRun/PolicyArtifact → ReferenceMotion
    EXECUTES_IN    = "EXECUTES_IN"    # TrainingRun → World/SoftwareEnvironment
    DECLARES_TARGET = "DECLARES_TARGET"  # Artifact → embodiment named by metadata
    COMPATIBLE_WITH = "COMPATIBLE_WITH"  # Artifact → exactly validated contract
    DERIVED_FROM   = "DERIVED_FROM"   # PolicyArtifact → PolicyArtifact
    ATTESTS        = "ATTESTS"        # ArtifactAttestation → immutable artifact
    PRODUCED       = "PRODUCED"       # TrainingRun → output PolicyArtifact
    USES_MODE_EXECUTION = "USES_MODE_EXECUTION"  # TrainingRun → ModeExecutionArtifact
    HAS_ITERATION  = "HAS_ITERATION"  # TrainingRun → TrainingIteration
    GROUNDS_CAPABILITY = "GROUNDS_CAPABILITY"  # Paper → ResearchCapability
    HAS_IMPLEMENTATION_STATUS = "HAS_IMPLEMENTATION_STATUS"  # ResearchCapability → ImplementationStatus


# ── Node dataclasses ────────────────────────────────────────────────────────
@dataclass
class Paper:
    kind: ClassVar[str] = "Paper"
    id: str
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    conclusion_text: str = ""
    #: Campaign curation metadata. Unlike PDF-derived claims, these fields
    #: record why the human/system chose this source and make unextracted
    #: hybrid-corpus papers retrievable by domain.
    rationale: str = ""
    tags: list[str] = field(default_factory=list)
    tier: str | None = None
    source_url: str = ""
    full_text_path: str | None = None
    ingested_at: float = field(default_factory=time.time)
    extracted: bool = False  # set True after LLM extraction lands in a later prompt
    #: §Agentic-data upgrade 1: papers are the KG's seed literature set.
    provenance: str = PROVENANCE_SEED


@dataclass
class Technique:
    """A named technique or method (e.g. 'reference state initialization',
    'quadratic height reward', 'CPG residual action')."""
    kind: ClassVar[str] = "Technique"
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    #: §Agentic-data upgrade 1: materialized from LLM extraction over a
    #: paper's text, with paper evidence attached at the edge level.
    provenance: str = PROVENANCE_PAPER_CLAIM
    #: §Agentic-data upgrade 2 (usage-based enrichment): incremented each
    #: time a KEPT ("helped") reward edit cites this technique's
    #: introducing paper — a coarse, capped signal that this technique has
    #: actually paid off in THIS project's own runs, not just been
    #: proposed. See query.py's ranking boost for the (deliberately small)
    #: cap rationale.
    useful_citations: int = 0
    #: §KG-retrieval fix 4 (outcome-stats ranking): per-FailureMode
    #: helped/regressed tallies from this project's OWN RunCase verdicts —
    #: `{<failure_mode_node_id>: {"helped": int, "regressed": int}}`.
    #: Written by `kg.cases.record_run_cases`, read by
    #: `kg.query.query_techniques`'s ordering boost. Unlike
    #: `useful_citations` (any "helped" iteration citing this technique's
    #: paper) this is scoped PER failure mode, so a technique that helps
    #: with one failure but regresses another doesn't wash out.
    outcome_stats: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class FailureMode:
    """A named RL-training failure pattern (e.g. 'walking_not_jumping',
    'tumbling_jump', 'flat_reward_plateau')."""
    kind: ClassVar[str] = "FailureMode"
    id: str
    name: str
    description: str = ""
    symptoms: list[str] = field(default_factory=list)
    environment_tag: str | None = None  # e.g. "continuous_locomotion"
    #: §Agentic-data upgrade 1: extracted from a paper's text (some are
    #: diagnoser-flagged stubs — see cases.py — but the type default
    #: reflects the common paper-derived case).
    provenance: str = PROVENANCE_PAPER_CLAIM


@dataclass
class RewardComponent:
    """A reusable reward-shaping component (e.g. 'ctrl_cost', 'alive_bonus',
    'imitation_exp_kernel'). Formula is a human-readable expression."""
    kind: ClassVar[str] = "RewardComponent"
    id: str
    name: str
    description: str = ""
    formula: str | None = None
    hyperparameters: dict[str, float] = field(default_factory=dict)
    #: §Agentic-data upgrade 1: materialized from paper extraction.
    provenance: str = PROVENANCE_PAPER_CLAIM


@dataclass
class Environment:
    """A benchmark env or env family (e.g. 'Hopper-v4', 'DeepMind Control
    Suite', 'Isaac Gym ANYmal')."""
    kind: ClassVar[str] = "Environment"
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Result:
    """A reported experimental datapoint: paper P got metric M = V on env E."""
    kind: ClassVar[str] = "Result"
    id: str
    paper_id: str
    metric_name: str
    value: float | None = None
    environment_id: str | None = None
    notes: str = ""


@dataclass
class ResearchCapability:
    """One paper concept or deliberately narrower system capability.

    This node describes *what* a mechanism means. Whether RewardSculptor
    executes it is represented by a separate :class:`ImplementationStatus`
    edge, so paper extraction can never accidentally turn a literature claim
    into a product claim. ``code_evidence`` names reviewed executable symbols
    for implemented or metadata-only capabilities and is empty for unsupported
    concepts by construction.
    """

    kind: ClassVar[str] = "ResearchCapability"
    id: str
    name: str
    description: str = ""
    scope: str = "paper_mechanism"
    code_evidence: list[str] = field(default_factory=list)
    provenance: str = PROVENANCE_SEED
    #: Machine-queryable parameters reported by the grounding source.  The
    #: empty default keeps capability rows written before this field was
    #: introduced readable and avoids turning an absent parameter inventory
    #: into an implementation claim.
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationStatus:
    """Definition node for a capability's current implementation status.

    The values are not interchangeable: ``implemented`` affects runtime
    behavior, ``metadata_only`` is retained or inspected but is not an
    execution authority, and ``unsupported`` is absent from the runtime.
    """

    kind: ClassVar[str] = "ImplementationStatus"
    id: str
    status: str
    definition: str
    provenance: str = PROVENANCE_SEED


@dataclass
class RunCase:
    """§Ship 37: a recorded run-iteration learning — a failure observed in a
    sculpt iteration, the edit tried in response, and whether the OBJECTIVE
    fitness then improved or regressed. Lets the diagnoser retrieve "what was
    tried before on this failure / task" so the same dead-end isn't repeated
    ("the same failure can't happen twice"). Distinct from the literature
    nodes: cases are this system's OWN experience — fast-moving, run-scoped,
    retrieved by semantic similarity like Techniques, kept in a separate silo
    so transient run artifacts never pollute the published-knowledge graph."""
    kind: ClassVar[str] = "RunCase"
    id: str
    task: str                                  # the behavior goal
    robot: str = ""                            # env / robot tag (optional)
    project: str = ""                          # project/stage scope (optional)
    symptom: str = ""                          # short failure description
    failure_modes: list[str] = field(default_factory=list)
    edit_summary: str = ""                     # what was changed in response
    fitness_before: float | None = None
    fitness_after: float | None = None
    fitness_delta: float | None = None
    verdict: str = "unknown"                   # helped|regressed|neutral|unknown
    # §2026-07-03 case-content upgrade. The original case carried only
    # "responded with N edit(s)" — a future diagnoser retrieving it
    # learned nothing actionable (WHICH edit failed? WHAT was the
    # behavior?). All optional so pre-upgrade rows load unchanged.
    #: Compact applied-edit identities, e.g. ["decrease stance_weight",
    #: "add flight_bonus"] — what a future run must not blindly repeat.
    edits: list[str] = field(default_factory=list)
    #: Paper references credited by this case. Persisted so a resumed run
    #: can update/reverse counters when an unknown verdict becomes measured,
    #: without double-counting an idempotent re-record.
    references: list[str] = field(default_factory=list)
    #: Attribution accounting schema. Rows written before references were
    #: persisted load as 0, allowing one conservative migration on re-record.
    attribution_version: int = 0
    #: Dense sub-success progress (§Convergence): the channel that ranks
    #: iters when the completion-gated fitness is 0.0 everywhere. Without
    #: it, every case from a below-gate run was verdict-'neutral' noise.
    progress_before: float | None = None
    progress_after: float | None = None
    progress_delta: float | None = None
    #: Salient physical numbers from the metric's component breakdown
    #: (e.g. apex_gain_m_mean, frac_launched) — the behavior signature
    #: that lets retrieval + the prompt distinguish "stand-still farm"
    #: from "tumble-bounce" instead of lumping both as reward_hacking.
    behavior: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    #: §Agentic-data upgrade 1: a RunCase IS an observed run by construction.
    provenance: str = PROVENANCE_OBSERVED_RUN
    #: §Agentic-data upgrade 4 (freshness metadata). The reward file version
    #: (e.g. "v3") and env-spec version (e.g. "v2") ACTIVE when this
    #: iteration trained — lets a future diagnoser (or a human) tell how
    #: stale a retrieved case is relative to the project's current reward
    #: / env. Populated in `record_run_cases` from whatever the run-history
    #: record already carries (`IterOutcome.reward_path_trained` /
    #: `.env_spec_trained`); None when unavailable (old rows, blind runs
    #: with no trained-reward path recorded).
    reward_version: str | None = None
    env_spec_version: str | None = None
    #: §Env-authoring §10 (world-tuple identity): the atomic selection's
    #: tuple_hash plus the world/task artifact versions ACTIVE while this
    #: iteration trained. Lets retrieval and humans judge how stale a case
    #: is relative to the project's current authored world, and ties the
    #: measured outcome to one exact evaluation lineage. Read fail-soft
    #: from the iteration's pinned selection file; None for legacy
    #: (non-authored) runs and pre-upgrade rows.
    world_tuple_hash: str | None = None
    world_version: int | None = None
    task_version: int | None = None


# ── Artifact lineage ────────────────────────────────────────────────────────
# These nodes deliberately carry factual, content-attested metadata only.  A
# bundle digest proves which bytes were admitted; it does *not* prove that the
# policy is capable, safe, or hardware-ready.  Behavioral claims remain
# Results/RunCases backed by observed rollouts.


@dataclass
class PolicyArtifact:
    """Byte-intrinsic facts for one immutable policy/checkpoint.

    Aliases, declared targets, trust decisions, and allowed transfer modes are
    deliberately absent: two manifests can make different claims about the
    same bytes. Those contextual claims belong to :class:`ArtifactAttestation`.
    """
    kind: ClassVar[str] = "PolicyArtifact"
    id: str
    sha256: str
    artifact_format: str
    size_bytes: int | None = None
    tensor_inventory_digest: str | None = None
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class ReferenceMotion:
    """Byte-intrinsic facts for a content-addressed reference trajectory."""
    kind: ClassVar[str] = "ReferenceMotion"
    id: str
    sha256: str
    fps: float | None = None
    frame_count: int | None = None
    joint_names: list[str] = field(default_factory=list)
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class WorldArtifact:
    """Byte-intrinsic facts for a world snapshot used by a training run."""
    kind: ClassVar[str] = "WorldArtifact"
    id: str
    sha256: str
    artifact_format: str = ""
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class ModeExecutionArtifact:
    """Exact reward/automaton bundle admitted immediately before training.

    This is an immutable *configuration* fact, not a behavioral claim.  The
    identity covers the reward bytes, independently re-derived mode graph,
    emitted execution-manifest digest, reference bytes/namespace, and every
    non-circular authored context ref that the reward binding names.
    """
    kind: ClassVar[str] = "ModeExecutionArtifact"
    id: str
    bundle_digest: str
    reward_sha256: str
    robot: str
    clip_id: str
    clip_sha256: str
    graph_sha256: str
    execution_manifest_digest: str
    selection_digest: str
    context_refs_digest: str
    context_refs: dict[str, str] = field(default_factory=dict)
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class ArtifactAttestation:
    """One admission/manifest claim about immutable artifact bytes.

    This node is keyed by the manifest digest, so a later import cannot mutate
    the meaning of an existing content node. ``declared`` is metadata, never an
    observed capability; validated compatibility is represented by an edge.
    """
    kind: ClassVar[str] = "ArtifactAttestation"
    id: str
    manifest_digest: str
    trust_status: str
    source_format: str
    declared: dict[str, Any] = field(default_factory=dict)
    admitted_at: float = field(default_factory=time.time)
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class RobotEmbodiment:
    """Ordered robot I/O contract against which compatibility is checked."""
    kind: ClassVar[str] = "RobotEmbodiment"
    id: str
    slug: str
    contract_digest: str
    joint_names: list[str] = field(default_factory=list)
    observation_contract: dict[str, Any] = field(default_factory=dict)
    action_contract: dict[str, Any] = field(default_factory=dict)
    control_dt_s: float | None = None
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class SoftwareEnvironment:
    """Content-addressed software/run-context identity for one invocation.

    ``captured_source_sha256`` is a legacy optional field. Exact
    ``run_context.json`` bytes are per-run edge evidence, not node identity,
    because they contain a volatile capture time. ``code_diff_digest`` hashes
    the normalized tracked diff plus untracked bytes and prevents two dirty
    worktrees at the same commit from collapsing into one node. Older rows
    remain readable because all added fields are optional.
    """
    kind: ClassVar[str] = "SoftwareEnvironment"
    id: str
    lock_digest: str
    versions: dict[str, str] = field(default_factory=dict)
    capture_schema: int | None = None
    captured_source_sha256: str | None = None
    code_commit: str | None = None
    code_dirty: bool | None = None
    code_tree_digest: str | None = None
    code_diff_digest: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    provenance: str = PROVENANCE_ATTESTED_ARTIFACT


@dataclass
class TrainingRun:
    """A concrete training invocation, linked to every starting artifact."""
    kind: ClassVar[str] = "TrainingRun"
    id: str
    project: str
    run_id: str
    requested_initialization_mode: str
    observed_initialization_mode: str | None = None
    code_commit: str | None = None
    selection_digest: str | None = None
    created_at: float = field(default_factory=time.time)
    provenance: str = PROVENANCE_OBSERVED_RUN


@dataclass
class TrainingIteration:
    """One ordered execution unit inside a concrete training invocation.

    Run-level edges remain useful summaries, but they cannot distinguish two
    iterations that used different policy inputs, worlds, or mode executors.
    This node is the collision-safe join point for those observed facts.
    """
    kind: ClassVar[str] = "TrainingIteration"
    id: str
    project: str
    run_id: str
    iteration_index: int
    provenance: str = PROVENANCE_OBSERVED_RUN


@dataclass
class Edge:
    src: str
    dst: str
    relation: Relation
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ── ID helpers ──────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    return _SLUG_RE.sub("-", s.strip().lower()).strip("-")


def make_paper_id(arxiv_id: str) -> str:
    return f"paper:{arxiv_id.strip()}"


def make_technique_id(name: str) -> str:
    return f"technique:{_slugify(name)}"


def make_failure_mode_id(name: str) -> str:
    return f"failure:{_slugify(name)}"


def make_reward_component_id(name: str) -> str:
    return f"component:{_slugify(name)}"


def make_environment_id(name: str) -> str:
    return f"environment:{_slugify(name)}"


def make_research_capability_id(name: str) -> str:
    return f"capability:{_slugify(name)}"


def make_implementation_status_id(status: str) -> str:
    return f"implementation-status:{_slugify(status)}"


def make_result_id(paper_id: str, metric_name: str, environment_id: str | None) -> str:
    """Content-addressed: same (paper, metric, env) always produces the same id."""
    env_slug = _slugify(environment_id or "none")
    metric_slug = _slugify(metric_name)
    paper_slug = paper_id.replace("paper:", "")
    return f"result:{paper_slug}|{metric_slug}|{env_slug}"


def make_run_case_id(task: str, iter_index: int, nonce: str) -> str:
    """§Ship 37: unique per (task, iter, run). `nonce` (a per-run token)
    distinguishes runs so cases ACCUMULATE across runs rather than overwrite —
    the whole point is to build an experience base over time."""
    return f"case:{_slugify(task)[:24]}:{int(iter_index)}:{_slugify(nonce)[:12]}"


def _artifact_id(kind: str, sha256: str) -> str:
    digest = sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{kind} sha256 must be 64 lowercase hex characters")
    return f"{kind}:{digest}"


def make_policy_artifact_id(sha256: str) -> str:
    return _artifact_id("policy", sha256)


def make_reference_motion_id(sha256: str) -> str:
    return _artifact_id("motion", sha256)


def make_world_artifact_id(sha256: str) -> str:
    return _artifact_id("world", sha256)


def make_mode_execution_artifact_id(bundle_digest: str) -> str:
    return _artifact_id("mode-execution", bundle_digest)


def make_artifact_attestation_id(manifest_digest: str) -> str:
    return _artifact_id("attestation", manifest_digest)


def make_robot_embodiment_id(slug: str, contract_digest: str) -> str:
    return f"robot:{_slugify(slug)}:{_artifact_id('contract', contract_digest).split(':', 1)[1]}"


def make_software_environment_id(lock_digest: str) -> str:
    return _artifact_id("software", lock_digest)


def make_training_run_id(project: str, run_id: str) -> str:
    return f"training-run:{_slugify(project)}:{_slugify(run_id)}"


def make_training_iteration_id(
    project: str, run_id: str, iteration_index: int,
) -> str:
    if isinstance(iteration_index, bool) or int(iteration_index) < 0:
        raise ValueError("training iteration index must be a non-negative integer")
    return (
        f"training-iteration:{_slugify(project)}:{_slugify(run_id)}:"
        f"{int(iteration_index)}"
    )


# ── Serialization helpers ───────────────────────────────────────────────────
NODE_TYPES: dict[str, type] = {
    cls.kind: cls  # type: ignore[attr-defined]
    for cls in (
        Paper, Technique, FailureMode, RewardComponent, Environment, Result,
        ResearchCapability, ImplementationStatus,
        RunCase, PolicyArtifact, ReferenceMotion, WorldArtifact,
        ModeExecutionArtifact,
        ArtifactAttestation,
        RobotEmbodiment, SoftwareEnvironment, TrainingRun, TrainingIteration,
    )
}


def node_to_row(node: Any) -> tuple[str, str, dict[str, Any]]:
    """Return (id, kind, data_dict) for storage. The `id`/`kind` fields are
    hoisted to their own columns; the rest goes in the JSON blob."""
    if not dataclasses.is_dataclass(node):
        raise TypeError(f"node must be a dataclass, got {type(node).__name__}")
    d = dataclasses.asdict(node)
    kind = getattr(type(node), "kind", None) or d.pop("kind", None)
    if kind is None:
        raise ValueError(f"node {node!r} has no kind")
    node_id = d.pop("id", None)
    if not node_id:
        raise ValueError(f"node {node!r} has no id")
    return node_id, kind, d


def row_to_node(node_id: str, kind: str, data: dict[str, Any]) -> Any:
    """Inverse of `node_to_row`. Looks up the dataclass type by `kind`.

    Forward-compatible on FIELDS: keys in `data` that this code version's
    dataclass doesn't know are DROPPED (a row written by a newer schema
    must not make older readers' `get_node` raise TypeError — symmetric
    with the backward-compat direction, where a missing key falls back to
    the field default). Unknown KINDS still raise: a whole node type this
    code can't represent is not safely partial-readable."""
    cls = NODE_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"unknown node kind: {kind!r}")
    known = {f.name for f in dataclasses.fields(cls)} - {"id"}
    extra = data.keys() - known
    if extra:
        data = {k: v for k, v in data.items() if k in known}
    return cls(id=node_id, **data)


def edge_to_row(edge: Edge) -> tuple[str, str, str, dict[str, Any]]:
    """Return (src, dst, relation, extra_dict) for storage."""
    return edge.src, edge.dst, edge.relation.value, {
        "data": edge.data,
        "created_at": edge.created_at,
    }


def row_to_edge(src: str, dst: str, relation: str, extra: dict[str, Any]) -> Edge:
    return Edge(
        src=src,
        dst=dst,
        relation=Relation(relation),
        data=extra.get("data", {}) or {},
        created_at=float(extra.get("created_at", 0.0)),
    )
