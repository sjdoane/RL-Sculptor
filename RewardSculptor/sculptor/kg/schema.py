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
  field — one of "observed_run" | "paper_claim" | "llm_extraction" |
  "seed" — so a retrieval-time renderer can tell Claude WHERE a claim came
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
    PROVENANCE_OBSERVED_RUN: "[evidence: observed in THIS project's own runs]",
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


# ── Serialization helpers ───────────────────────────────────────────────────
NODE_TYPES: dict[str, type] = {
    cls.kind: cls  # type: ignore[attr-defined]
    for cls in (Paper, Technique, FailureMode, RewardComponent, Environment,
                Result, RunCase)
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
    """Inverse of `node_to_row`. Looks up the dataclass type by `kind`."""
    cls = NODE_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"unknown node kind: {kind!r}")
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
