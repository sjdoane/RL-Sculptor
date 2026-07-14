"""sculptor/diagnose.py — two-stage LLM diagnoser grounded in the KG.

Stage 1 ("preliminary"):
    behavior_goal + current REWARD_SPEC + metrics.json + behavior.json +
    4 keyframe PNGs + reward_contract + the adapter's behavior-metric vocab
    → failure_modes (from a fixed enum), evidence, confidence,
    failure_descriptors (free-text, optional — see below).

Stage 2 ("grounded"):
    preliminary result + KG context (top-KG_TOP_K union of
    query_techniques(failure_modes ∪ descriptor-resolved FailureMode ids,
    domain_filter=config.kg.environment_tag), an EVIDENCE-anchored
    query_semantic(preliminary.evidence) when evidence is non-empty, and
    the STATIC query_semantic(behavior_goal)) + original inputs +
    reward_contract → proposed_edits. Every edit must cite paper_refs
    (arxiv_ids) when literature-grounded, or mark itself `novel.` and
    leave paper_refs empty. Kept refs are additionally annotated
    `grounded=True/False`: True iff the arxiv_id was among the papers
    THIS iteration's retrieved literature_context actually showed Claude
    (vs. merely existing somewhere in the KG — the existence check alone
    doesn't distinguish "shown" from "recalled").

    NOTE on `query_semantic(behavior_goal)`: this query is STATIC per
    stage (behavior_goal never changes across a stage's iterations), so
    on its own it retrieves identical literature every call. The
    evidence-anchored query above is what makes retrieval track what's
    actually going wrong THIS iteration.

Both calls use the registry model (`sculptor.llm.model_for("diagnose")`)
with adaptive thinking, strict JSON via `messages.parse`, and one retry
on parse/validation failure.

The diagnoser is stack-agnostic: failure modes are the SAME six across RL
domains (`reward_hacking`, `static_equilibrium`, `premature_termination`,
`sparse_reward`, `reward_saturation`, `component_imbalance`, plus `none`),
but the behavior-metric vocabulary it sees (e.g. `max_episode_length`,
`fall_rate` for Hopper; `max_jump_height` for a quadruped) is read from
`config.iteration.behavior_metrics` at call time.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from sculptor.adapters.base import load_adapter
from sculptor.kg.query import TechniqueMatch, query_semantic, query_techniques
from sculptor.kg.schema import evidence_tag, make_paper_id
from sculptor.kg.store import SculptorKG
from sculptor.llm import log_llm_call, model_for, response_text_blocks


MODEL_ID = model_for("diagnose")
MAX_TOKENS = 8192
N_KEYFRAMES_SENT = 4         # the spec calls for exactly 4 keyframes per call
KG_TOP_K = 6                 # union size of KG context shown to the grounded call
RETRY_REMINDER = (
    "Your previous response did not validate against the schema. "
    "Return JSON only — no preamble, no markdown code fence."
)

FailureModeLit = Literal[
    "reward_hacking",
    "static_equilibrium",
    "premature_termination",
    "sparse_reward",
    "reward_saturation",
    "component_imbalance",
    "none",
]
EditOpLit = Literal[
    "increase", "decrease", "add", "remove", "clip", "gate", "replace", "normalize"
]


# ── Pydantic shapes (what Claude is constrained to emit) ──────────────────
class _PreliminaryModel(BaseModel):
    failure_modes: list[FailureModeLit] = Field(default_factory=list)
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: §KG-retrieval fix 2: 2-4 short FREE-TEXT phrases naming the SPECIFIC
    #: observed failure (e.g. "planks on forearms without leg drive") —
    #: additive detail alongside `failure_modes`, which stays restricted to
    #: the fixed six-plus-none vocabulary. Optional: absent or malformed
    #: input (not a list, non-string entries, etc.) coerces to `[]` rather
    #: than raising, so old cached/replayed preliminary responses (recorded
    #: before this field existed) still parse.
    failure_descriptors: list[str] = Field(default_factory=list)

    @field_validator("failure_descriptors", mode="before")
    @classmethod
    def _coerce_failure_descriptors(cls, v):  # noqa: ANN001, ANN205
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]


class _ProposedEditModel(BaseModel):
    target_term: str = Field(
        description=(
            "Name of the reward hyperparameter or component to change. "
            "Must be a key in REWARD_SPEC.hyperparameters for existing terms, "
            "or a new snake_case name for an added term."
        )
    )
    operation: EditOpLit
    rationale: str
    suggested_value: Optional[str] = Field(
        default=None,
        description=(
            "New numeric value as a string, or a short formula description "
            "for add/replace operations. Null when the edit type doesn't "
            "carry a value (e.g., 'remove'). MUST only reference fields in "
            "reward_contract.expected_info_keys or existing reward components."
        ),
    )
    paper_refs: list[str] = Field(
        default_factory=list,
        description="arXiv IDs (e.g., '1801.00690') supporting this edit. "
                    "Empty when the edit is novel.",
    )
    requires_env_extension: bool = Field(
        default=False,
        description=(
            "Set to true when the ideal edit would need a NEW field in "
            "info / env — a field NOT in reward_contract.expected_info_keys. "
            "Populate rationale with what field is needed and why, and leave "
            "suggested_value null (or describe the ideal formula in prose "
            "without using the ungrounded field as code). The editor stage "
            "skips these edits; they serve as recorded proposals for the "
            "adapter author to expose new env state."
        ),
    )


#: §env generalization 3/4 — the diagnoser's env-adaptation surface is
#: EXACTLY the env spec's train section (shared/eval keys are frozen per
#: run and structurally absent here). Single-sourced from env_spec so
#: the constraint can't drift.
from sculptor.env_spec import ITERABLE_TRAIN_KEYS as _ENV_ITERABLE_KEYS

_EnvParamLit = Literal[tuple(sorted(_ENV_ITERABLE_KEYS))]  # type: ignore[valid-type]


class _ProposedEnvEditModel(BaseModel):
    parameter: _EnvParamLit = Field(
        description=(
            "TRAIN-ONLY env-spec parameter to change (the # ENV_SPEC "
            "block lists current values + hard bounds)."
        )
    )
    new_value: str = Field(
        description=(
            "New value as stringified JSON matching the parameter's "
            "shape: a number (e.g. \"0.25\") or a [lo, hi] pair "
            "(e.g. \"[0.0, 0.4]\")."
        )
    )
    rationale: str

    @field_validator("new_value", mode="before")
    @classmethod
    def _coerce_json_value(cls, v):  # noqa: ANN001, ANN205
        """Accept a bare number / pair the model emitted unstringified —
        coercing here saves a full parse-retry round-trip."""
        if isinstance(v, (int, float, bool, list)):
            return json.dumps(v)
        return v


class _GroundedModel(BaseModel):
    proposed_edits: list[_ProposedEditModel] = Field(default_factory=list)
    proposed_env_edits: list[_ProposedEnvEditModel] = Field(
        default_factory=list,
        description=(
            "0-2 changes to the TRAINING-ONLY environment curriculum "
            "(only when the user message contains an # ENV_SPEC block "
            "and the diagnosed failure is a training-distribution "
            "pathology; empty otherwise)."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)


# ── Public dataclasses ────────────────────────────────────────────────────
@dataclass
class ProposedEdit:
    target_term: str
    operation: str
    rationale: str
    suggested_value: str | None
    paper_refs: list[str] = field(default_factory=list)
    #: §KG-retrieval fix 5: per-arxiv_id grounding annotation for the KEPT
    #: `paper_refs` above — True iff that arxiv_id was among the papers
    #: THIS iteration's retrieved literature_context actually showed
    #: Claude, False if it's cited-and-exists-in-the-KG but wasn't shown
    #: (Claude recalled it rather than being grounded on it this iter).
    #: A separate dict (not a paper_refs shape change) because paper_refs
    #: is consumed as a flat `list[str]` throughout edit.py/sculpt.py/
    #: timelapse.py (KG-existence validation, citation joins, provenance
    #: tracking) — changing its element type would ripple across all of
    #: them for no behavioral gain.
    paper_refs_grounded: dict[str, bool] = field(default_factory=dict)
    requires_env_extension: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ProposedEnvEdit:
    """§env generalization 3/4: a diagnoser-proposed change to the env
    spec's TRAIN section — the environment-curriculum counterpart of
    ProposedEdit. Applied (validated + bounded) by
    `env_spec.apply_env_edits`, takes effect the NEXT iteration."""

    parameter: str
    new_value: str
    rationale: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Diagnosis:
    failure_modes: list[str]
    evidence: str
    proposed_edits: list[ProposedEdit] = field(default_factory=list)
    proposed_env_edits: list[ProposedEnvEdit] = field(default_factory=list)
    literature_context: list[TechniqueMatch] = field(default_factory=list)
    confidence: float = 0.0
    iter_dir: str | None = None
    behavior_goal: str | None = None

    def to_dict(self) -> dict:
        return {
            "failure_modes": list(self.failure_modes),
            "evidence": self.evidence,
            "proposed_edits": [e.to_dict() for e in self.proposed_edits],
            "proposed_env_edits": [
                e.to_dict() for e in self.proposed_env_edits],
            "literature_context": [
                {
                    "technique": m.technique.name,
                    # §Fix 3 (staleness rotation): the technique's node id
                    # and its introducing papers' node ids, so a LATER
                    # iteration's diagnose() can tell whether THIS iter's
                    # shown-but-uncited techniques should be rotated out
                    # without re-querying the KG. Additive fields — no
                    # existing consumer indexes this dict by exact key set.
                    "technique_id": m.technique.id,
                    "source_paper_ids": list(m.source_paper_ids),
                    "description": m.description,
                    "paper_citation": m.paper_citation,
                    "evidence": m.evidence,
                    "relevance_score": m.relevance_score,
                    "matched_on": m.matched_on,
                    # §KG-retrieval fix 5: literature_context entries are,
                    # by definition, RETRIEVED (shown to Claude this
                    # iteration) — the uniform `grounded` field lets
                    # downstream consumers treat this list and
                    # proposed_edits[].paper_refs_grounded consistently.
                    "grounded": True,
                }
                for m in self.literature_context
            ],
            "confidence": self.confidence,
            "iter_dir": self.iter_dir,
            "behavior_goal": self.behavior_goal,
        }


# ── Config helpers ────────────────────────────────────────────────────────
def _parse_config(config: Path | str | dict) -> dict:
    if isinstance(config, dict):
        return config
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with Path(config).open("rb") as f:
        return tomllib.load(f)


def _behavior_metrics_from_config(cfg: dict) -> list[str]:
    iter_cfg = cfg.get("iteration", {}) or {}
    return list(iter_cfg.get("behavior_metrics", []))


def _env_tag_from_config(cfg: dict) -> str | None:
    return (cfg.get("kg", {}) or {}).get("environment_tag")


# ── Iter-dir loading ──────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[diagnose] warning: failed to read {path.name}: {e}",
              file=sys.stderr, flush=True)
        return {}


def _find_artifact(iter_dir: Path, name: str) -> Path:
    """Look for `name` in iter_dir/, then iter_dir/rollout/. Returns the
    first match, or iter_dir/name as a fallback (the caller's _load_json
    tolerates missing files)."""
    direct = iter_dir / name
    if direct.is_file():
        return direct
    nested = iter_dir / "rollout" / name
    if nested.is_file():
        return nested
    return direct


def _pick_keyframes(iter_dir: Path, n: int = N_KEYFRAMES_SENT) -> list[Path]:
    # Train + rollout may land in different subdirs — check both.
    for candidate in (iter_dir / "keyframes", iter_dir / "rollout" / "keyframes"):
        if candidate.is_dir():
            frames = sorted(candidate.glob("*.png"))
            if frames:
                break
    else:
        return []
    if len(frames) <= n:
        return frames
    import numpy as np

    idxs = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
    return [frames[i] for i in idxs]


def _encode_image(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


# ── Prompt rendering ──────────────────────────────────────────────────────
from sculptor.prompts import load_prompt

_PRELIM_SYSTEM = load_prompt("diagnose_preliminary")
_GROUNDED_SYSTEM = load_prompt("diagnose_grounded")


def _load_realism_audit(iter_dir: Path) -> dict:
    """Load `<iter_dir>/realism_audit.json` (§7.3). Returns `{}` when the
    file is missing or parseable — the prompt formatter then emits no
    block and the diagnose prompt shape matches the pre-§7.3 behavior."""
    path = iter_dir / "realism_audit.json"
    if not path.is_file():
        return {}
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _format_realism_audit(audit: dict) -> str:
    """Render the audit dict as a prompt block. Only emits when the
    verdict is `mild` or `severe` — an `ok` / `unknown` verdict would
    just dilute the prompt with no actionable signal."""
    if not isinstance(audit, dict):
        return ""
    verdict = str(audit.get("verdict") or "")
    if verdict not in ("mild", "severe"):
        return ""
    lines: list[str] = [
        f"verdict: {verdict.upper()}",
        f"torque_saturation_frac: {audit.get('torque_saturation_frac', 'n/a')} "
        f"(worst single joint: {audit.get('any_joint_saturation_max', 'n/a')})",
        f"joint_vel_p99_max: {audit.get('joint_vel_p99_max', 'n/a')} rad/s "
        f"({audit.get('joint_vel_multiplier_vs_nominal', 'n/a')}× nominal 30 rad/s)",
        f"joint_limit_violation_frac: {audit.get('joint_limit_violation_frac', 'n/a')}",
    ]
    top_sat = audit.get("top_joints_saturation") or []
    if top_sat:
        entries = ", ".join(
            f"{j.get('name', '?')}={j.get('value', 0.0):.2f}" for j in top_sat
        )
        lines.append(f"top_saturated_joints: {entries}")
    top_vel = audit.get("top_joints_vel") or []
    if top_vel:
        entries = ", ".join(
            f"{j.get('name', '?')}={j.get('value', 0.0):.2f}rad/s" for j in top_vel
        )
        lines.append(f"top_vel_joints: {entries}")
    return "\n".join(lines)


def _load_training_feedback(iter_dir: Path) -> dict:
    """Load the per-component time-series that §7.1 writes.

    Prefers the training-side `<iter_dir>/reward_trajectory.json` (one
    value per save_interval window — Eureka Appendix F format) and falls
    back to the rollout-side file at `<iter_dir>/rollout/reward_trajectory.json`
    when training didn't emit one (non-sculpted run, or pre-§7.1 iter).
    Returns `{}` when neither file is present or parseable, which keeps
    the prompt-formatter a no-op on those iters."""
    for candidate in (
        iter_dir / "reward_trajectory.json",
        iter_dir / "rollout" / "reward_trajectory.json",
    ):
        if candidate.is_file():
            payload = _load_json(candidate)
            if isinstance(payload, dict) and payload:
                return payload
    return {}


def _format_training_feedback(data: dict) -> str:
    """Render a `{component: [v0, v1, ...]}` dict in Eureka Appendix F
    format — one line per key, values list followed by Max/Mean/Min.

    Caps the shown list at 10 evenly-spaced points (Eureka's default)
    to keep the prompt bounded. `__` prefix from §7.1's aux signals
    (`__episode_length`, `__terminated`, `__time_outs`) is stripped for
    display so the LLM sees readable labels."""
    if not isinstance(data, dict) or not data:
        return ""
    lines: list[str] = []
    for name, raw_vals in data.items():
        if not isinstance(raw_vals, list) or not raw_vals:
            continue
        try:
            vals = [float(v) for v in raw_vals]
        except (TypeError, ValueError):
            continue
        if not vals:
            continue
        if len(vals) > 10:
            step = (len(vals) - 1) / 9.0
            idxs = [int(round(i * step)) for i in range(10)]
            # Deduplicate while preserving order (short lists with many
            # samples can collapse).
            seen: set[int] = set()
            unique_idxs: list[int] = []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    unique_idxs.append(i)
            shown = [vals[i] for i in unique_idxs]
        else:
            shown = list(vals)
        shown_strs = [f"{v:.2f}" for v in shown]
        display_name = name[2:] if name.startswith("__") else name
        line = (
            f"{display_name}: [{', '.join(shown_strs)}], "
            f"Max: {max(vals):.2f}, Mean: {sum(vals) / len(vals):.2f}, "
            f"Min: {min(vals):.2f}"
        )
        lines.append(line)
    return "\n".join(lines)


def _render_reward_contract(contract) -> str:
    obs = getattr(contract, "observation_space_spec", None)
    act = getattr(contract, "action_space_spec", None)
    return (
        f"observation_space: {obs}\n"
        f"action_space:      {act}\n"
        f"expected_info_keys: {list(contract.expected_info_keys)}\n"
        f"expected_components: {contract.expected_components}"
    )


def _render_kg_context(matches: list[TechniqueMatch]) -> str:
    """§Agentic-data upgrade 1: each technique gets a compact
    `[evidence: ...]` provenance tag (techniques are paper_claim by
    default — materialized from extraction) and the block's header states
    the trust-tier rule so Claude weighs a conflicting case-memory
    observation (rendered above this block, see diagnose()) correctly."""
    if not matches:
        return "(no matches from the knowledge graph)"
    lines = [
        "# LITERATURE CONTEXT (top KG matches)",
        "# Observations from this system's own runs (CASE MEMORY, above) "
        "outrank paper claims below when they conflict.",
        "",
    ]
    for m in matches:
        # `provenance` is a NEW field (agentic-data upgrade 1) — read it
        # defensively so Technique-shaped objects that predate it (duck-
        # typed fakes, older pickled rows) degrade to the least-trusted
        # tag instead of crashing the whole diagnose context build
        # (evidence_tag(None) → llm-inferred tier by design).
        lines.append(
            f"## {m.technique.name} "
            f"{evidence_tag(getattr(m.technique, 'provenance', None))}"
        )
        lines.append(f"- source: {m.paper_citation}")
        if m.matched_on:
            lines.append(f"- matched_on: {m.matched_on}")
        lines.append(f"- relevance: {m.relevance_score:.3f}")
        desc = (m.description or "").strip()
        if desc:
            lines.append(f"- description: {desc}")
        ev = (m.evidence or "").strip()
        if ev:
            lines.append(f"- evidence: {ev}")
        lines.append("")
    return "\n".join(lines)


def _stale_uncited_technique_ids(
    iter_dir: Path, objective_progress: dict | None,
) -> set[str]:
    """§Fix 3 (staleness rotation): when the diagnoser is STUCK (this
    iter's fitness delta-vs-previous is <=0), find technique ids the
    PREVIOUS iteration's literature block showed but whose papers no
    proposed edit actually cited — re-showing uncited literature to a
    stuck diagnoser is dead weight.

    Inert (returns an empty set) whenever:
      * `objective_progress` is None/empty, or its `delta` is None or > 0
        (no regression signal, or nothing to compare against yet);
      * `iter_dir`'s name isn't the `iter_<N>` convention, or N == 0
        (no previous iteration can exist);
      * the previous iteration's `diagnosis.json` is missing or
        unparseable (best-effort — never raises).
    """
    if not objective_progress:
        return set()
    delta = objective_progress.get("delta")
    # Non-numeric delta (malformed progress dict) must read as "not stuck",
    # not raise out of the unguarded diagnose call.
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta > 0:
        return set()
    name = iter_dir.name
    if not name.startswith("iter_"):
        return set()
    try:
        idx = int(name[len("iter_"):])
    except ValueError:
        return set()
    if idx <= 0:
        return set()
    prev_path = iter_dir.parent / f"iter_{idx - 1}" / "diagnosis.json"
    if not prev_path.is_file():
        return set()
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — malformed prior diagnosis: rule inert
        return set()
    if not isinstance(prev, dict):
        return set()
    shown = prev.get("literature_context")
    if not isinstance(shown, list) or not shown:
        return set()

    cited_paper_ids: set[str] = set()
    for e in (prev.get("proposed_edits") or []):
        if not isinstance(e, dict):
            continue
        for aid in (e.get("paper_refs") or []):
            cited_paper_ids.add(make_paper_id(str(aid)))

    excluded: set[str] = set()
    for entry in shown:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("technique_id")
        if not tid:
            continue
        src_ids = entry.get("source_paper_ids") or []
        if not any(pid in cited_paper_ids for pid in src_ids):
            excluded.add(tid)
    return excluded


def _build_preliminary_user_content(
    behavior_goal: str,
    reward_spec: dict,
    metrics: dict,
    behavior: dict,
    behavior_metric_names: list[str],
    contract_text: str,
    keyframes: list[Path],
    training_feedback: dict | None = None,
    realism_audit: dict | None = None,
    objective_progress: dict | None = None,
    human_note: str | None = None,
    reference_signature: dict | None = None,
) -> list[dict]:
    # §Ship 33: optional OBJECTIVE TASK PROGRESS block — a held-out,
    # ground-truth fitness scalar (higher = better) for THIS iter's
    # rollout, plus best-so-far / last / delta. Omitted (None) keeps the
    # blind-navigation behavior. When present it makes the diagnose→edit
    # search fitness-guided (the mechanism Eureka's ablations show is
    # indispensable) instead of navigating only by self-authored signals.
    # §Ship 39 (H1): a human watching the rollout video can inject a free-text
    # observation that the model can't see (subtle gait artifacts, "it's
    # cheating by leaning on the wall", etc.). Rendered FIRST + emphatically —
    # the human has ground-truth eyes on behavior the metrics may miss.
    human_block = ""
    if human_note and str(human_note).strip():
        human_block = (
            "# USER OBSERVATION (a human watched this iteration's rollout "
            "video and is steering you — weight this HEAVILY; they can see "
            "behavior the metrics and keyframes miss):\n"
            f"{str(human_note).strip()}\n\n"
        )

    objective_block = ""
    if objective_progress:
        cur = objective_progress.get("current")
        best = objective_progress.get("best_so_far")
        last = objective_progress.get("last")
        delta = objective_progress.get("delta")
        components = objective_progress.get("components")
        reverted = objective_progress.get("reverted_to_best")
        objective_block = (
            "# OBJECTIVE_TASK_PROGRESS\n"
            "# Ground-truth task fitness in [0,1] (higher is better), "
            "measured on this iteration's rollout independently of the "
            "reward you are editing. This is the bar that actually "
            "matters — prioritize edits you expect to RAISE it. A high "
            "reward with low/stalled fitness means the reward is being "
            "optimized for the wrong thing (reward hacking).\n"
            f"current={cur}  best_so_far={best}  "
            f"previous={last}  delta_vs_previous={delta}\n"
        )
        # §Ship 36 (F2): physical sub-measurements of the fitness above, so
        # you can localize WHAT is wrong rather than only THAT it fell (e.g.
        # high burst speed but low uprightness = violent-but-falling; high
        # kick_events with a low score = the legacy ratio is under-counting).
        if components:
            objective_block += (
                "# fitness component breakdown (physical sub-measurements):\n"
                f"{json.dumps(components, sort_keys=True)}\n"
            )
        # §Ship 36 (F1): the prior edit regressed fitness and was rolled back.
        if reverted:
            objective_block += (
                "# NOTE: your PREVIOUS edit REGRESSED fitness, so the reward "
                "was reverted to the best-so-far version before this "
                "iteration trained. Do NOT repeat that edit — diagnose why it "
                "lowered the objective and propose a DIFFERENT direction.\n"
            )
        objective_block += "\n"
    # §7.2: Eureka-format reward trajectory injected between metrics and
    # REWARD_CONTRACT. Block is omitted when the file is missing (pre-§7.1
    # iters, non-sculpted runs) so stage-1 diagnose stays structurally
    # identical when no per-component data is available.
    feedback_block = ""
    formatted = _format_training_feedback(training_feedback or {})
    if formatted:
        feedback_block = (
            "# TRAINING_FEEDBACK\n"
            "# Per-component + success/episode-length time-series across "
            "training (one value per save_interval window). Use these to "
            "spot dead components (max - min < 5% of max → cannot be "
            "optimized by RL) and component imbalance (one term's Max "
            "dwarfs everything else).\n"
            f"{formatted}\n\n"
        )
    # §7.3: physics-realism audit — only emitted on mild/severe verdicts
    # so healthy runs don't dilute the prompt.
    realism_block = ""
    realism_text = _format_realism_audit(realism_audit or {})
    if realism_text:
        realism_block = (
            "# PHYSICS_REALISM_AUDIT\n"
            "# Post-rollout check on torque saturation, joint velocity, "
            "and joint-limit violation. Severe verdict + reward_hacking "
            "failure mode usually means the MJCF permits unrealistic "
            "actuator response the policy is exploiting — flag it in "
            "evidence so the physics-edit step can address it.\n"
            f"{realism_text}\n\n"
        )
    # §reference-grounded diagnose: when the stage has a resolved reference
    # clip, the mission scaffold wrote its kinematic signature to
    # `<stage_dir>/reference_signature.json`. Rendered here (right next to
    # the training/behavior stats) so the model can compare rollout numbers
    # against the competent-motion numbers side by side instead of
    # inventing targets. Absent reference_signature -> "" -> byte-identical
    # to pre-this-change prompts.
    reference_block = ""
    if reference_signature:
        from sculptor.reference_context import render_reference_signature_block

        rendered = render_reference_signature_block(reference_signature)
        if rendered:
            reference_block = f"{rendered}\n\n"
    header = (
        f"# BEHAVIOR GOAL\n{behavior_goal}\n\n"
        f"{human_block}"
        f"# REWARD_SPEC\n{json.dumps(reward_spec, indent=2, sort_keys=True, default=str)}\n\n"
        f"# metrics.json\n{json.dumps(metrics, indent=2, sort_keys=True, default=str)}\n\n"
        f"{objective_block}"
        f"{feedback_block}"
        f"{realism_block}"
        f"{reference_block}"
        f"# ADAPTER BEHAVIOR METRIC VOCABULARY\n{behavior_metric_names}\n\n"
        f"# behavior.json\n{json.dumps(behavior, indent=2, sort_keys=True, default=str)}\n\n"
        f"# REWARD_CONTRACT\n{contract_text}\n\n"
        f"# KEYFRAMES ({len(keyframes)} evenly-spaced frames from the best eval episode)\n"
    )
    content: list[dict] = [{"type": "text", "text": header}]
    for kf in keyframes:
        content.append(_encode_image(kf))
    content.append({"type": "text", "text": "Emit the preliminary diagnosis JSON now."})
    return content


def _render_env_spec_block(env_spec: dict | None) -> str:
    """§env generalization 3/4: the # ENV_SPEC block for the grounded
    prompt — current train-section values, the editable parameter set
    with hard bounds (single-sourced from the validator's tables), and
    the frozen shared section for context. Empty string when no spec is
    active (the model is instructed to emit no env edits then)."""
    if not isinstance(env_spec, dict):
        return ""
    from sculptor.env_spec import _TRAIN_RANGES, _TRAIN_SCALARS

    train = env_spec.get("train") or {}
    shared = env_spec.get("shared") or {}
    bounds_lines = [
        f"  {k}: scalar within [{lo:g}, {hi:g}]"
        for k, (lo, hi) in sorted(_TRAIN_SCALARS.items())
    ] + [
        f"  {k}: [lo, hi] pair, both within [{lo:g}, {hi:g}]"
        for k, (lo, hi) in sorted(_TRAIN_RANGES.items())
    ]
    return (
        "# ENV_SPEC\n"
        "# The project's environment spec. `train` is the TRAINING-ONLY\n"
        "# curriculum surface you may edit via proposed_env_edits (takes\n"
        "# effect next iteration; evaluation rollouts NEVER see it).\n"
        "# `shared` defines the evaluated task — frozen for this run.\n"
        f"active version: {(env_spec.get('meta') or {}).get('version', '?')}\n"
        f"train (editable): {json.dumps(train, sort_keys=True)}\n"
        f"shared (frozen): {json.dumps(shared, sort_keys=True)}\n"
        "editable parameters + hard bounds:\n"
        + "\n".join(bounds_lines) + "\n\n"
    )


def _build_grounded_user_content(
    behavior_goal: str,
    reward_spec: dict,
    metrics: dict,
    behavior: dict,
    contract_text: str,
    preliminary: _PreliminaryModel,
    kg_context: str,
    training_feedback: dict | None = None,
    realism_audit: dict | None = None,
    env_spec: dict | None = None,
    reference_signature: dict | None = None,
) -> str:
    feedback_block = ""
    formatted = _format_training_feedback(training_feedback or {})
    if formatted:
        feedback_block = (
            "# TRAINING_FEEDBACK\n"
            "# Per-component time-series across training. Dead components "
            "(near-constant values) and component imbalance (one term's Max "
            "dominates) are the two patterns to ground edits against.\n"
            f"{formatted}\n\n"
        )
    realism_block = ""
    realism_text = _format_realism_audit(realism_audit or {})
    if realism_text:
        realism_block = (
            "# PHYSICS_REALISM_AUDIT\n"
            "# Post-rollout torque/velocity/limit audit. Severe + reward_hacking "
            "means MJCF is likely exploited; mild means reward shape can fix it.\n"
            f"{realism_text}\n\n"
        )
    # §reference-grounded diagnose: same block the preliminary call sees
    # (see _build_preliminary_user_content) — the grounded call proposes
    # edits, so it needs the same measured-vs-reference numbers to derive
    # targets/thresholds from rather than inventing them.
    reference_block = ""
    if reference_signature:
        from sculptor.reference_context import render_reference_signature_block

        rendered = render_reference_signature_block(reference_signature)
        if rendered:
            reference_block = f"{rendered}\n\n"
    return (
        f"# BEHAVIOR GOAL\n{behavior_goal}\n\n"
        f"# REWARD_SPEC\n{json.dumps(reward_spec, indent=2, sort_keys=True, default=str)}\n\n"
        f"# metrics.json\n{json.dumps(metrics, indent=2, sort_keys=True, default=str)}\n\n"
        f"{feedback_block}"
        f"{realism_block}"
        f"{reference_block}"
        f"{_render_env_spec_block(env_spec)}"
        f"# behavior.json\n{json.dumps(behavior, indent=2, sort_keys=True, default=str)}\n\n"
        f"# REWARD_CONTRACT\n{contract_text}\n\n"
        f"# PRELIMINARY DIAGNOSIS\n"
        f"failure_modes: {preliminary.failure_modes}\n"
        f"evidence: {preliminary.evidence}\n"
        f"confidence: {preliminary.confidence:.2f}\n\n"
        f"{kg_context}\n\n"
        f"Emit the grounded proposed_edits JSON now."
    )


# ── LLM calls with one-retry parse ────────────────────────────────────────
def _log_user_text(user_content: Any) -> str:
    """Text view of a user-content payload for the provenance archive.
    Image blocks (base64 keyframes) are elided to a placeholder — they
    would bloat llm_calls.jsonl by MBs per call without adding replay
    value (the keyframe PNGs are already archived in the iter dir)."""
    if isinstance(user_content, str):
        return user_content
    parts: list[str] = []
    for blk in user_content or []:
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
        elif isinstance(blk, dict):
            parts.append(f"[{blk.get('type', 'block')} block omitted]")
    return "\n".join(parts)


def _parse_with_retry(client, *, model_cls, system_prompt, user_content,
                      stage: Optional[str] = None):
    """messages.parse with one retry on parse / validation failure."""
    meta = {"stage": stage} if stage else None
    try:
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_format=model_cls,
        )
        log_llm_call(
            "diagnose", MODEL_ID, system=system_prompt,
            user=_log_user_text(user_content),
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None),
            meta={**(meta or {}), "attempt": 1})
        return resp.parsed_output
    except Exception as first_err:
        # 4.7 forbids assistant prefill — retry via a second user turn.
        retry_reminder = f"{RETRY_REMINDER} Previous error: {first_err!s}"
        retry_messages = [
            {"role": "user", "content": user_content},
            {"role": "user", "content": retry_reminder},
        ]
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=retry_messages,
            output_format=model_cls,
        )
        log_llm_call(
            "diagnose", MODEL_ID, system=system_prompt,
            user=f"{_log_user_text(user_content)}\n\n{retry_reminder}",
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None),
            meta={**(meta or {}), "attempt": 2})
        return resp.parsed_output


# ── Public entry ──────────────────────────────────────────────────────────
def diagnose(
    iter_dir: Path | str,
    behavior_goal: str,
    config: Path | str | dict,
    *,
    store: SculptorKG | None = None,
    client=None,
    skip_kg: bool = False,
    objective_progress: dict | None = None,
    human_note: str | None = None,
) -> Diagnosis:
    """Two-stage literature-grounded diagnosis of a training iteration.

    `skip_kg=True` bypasses both KG queries and passes an empty
    literature_context to the grounded call — used for the `--no-kg`
    ablation mode in `sculpt run`.

    `objective_progress` (§Ship 33) — optional ground-truth task-fitness
    summary (current / best_so_far / last / delta) for this iter; when
    given it is surfaced to stage-1 so the diagnosis is fitness-guided.
    None preserves the original blind behavior.
    """
    iter_dir = Path(iter_dir).resolve()
    cfg = _parse_config(config)

    # 1. Load iter artifacts (train writes to iter_dir; rollout may write to
    # iter_dir/rollout/ — we check both).
    metrics = _load_json(_find_artifact(iter_dir, "metrics.json"))
    behavior = _load_json(_find_artifact(iter_dir, "behavior.json"))
    reward_spec = _load_json(_find_artifact(iter_dir, "reward_spec.json"))
    if not reward_spec:
        # adapter writes {"metrics": {...}, "components": {...}} under metrics.json
        # and REWARD_SPEC under reward_spec.json — if missing we soldier on with {}.
        print(f"[diagnose] warning: {iter_dir / 'reward_spec.json'} missing — "
              "diagnosis prompt will show an empty REWARD_SPEC.",
              file=sys.stderr, flush=True)
    # §7.2: Eureka reward-reflection data. Optional — pre-§7.1 iters
    # won't have the file, in which case the prompt omits the block.
    training_feedback = _load_training_feedback(iter_dir)
    # §7.3: realism audit. Emitted for mild/severe verdicts only.
    realism_audit = _load_realism_audit(iter_dir)
    keyframes = _pick_keyframes(iter_dir, N_KEYFRAMES_SENT)
    behavior_metric_names = _behavior_metrics_from_config(cfg)
    env_tag = _env_tag_from_config(cfg)
    # §reference-grounded diagnose: the stage dir is the config file's
    # parent directory. The mission scaffold writes
    # <stage_dir>/reference_signature.json for every stage whose reference
    # clip resolves; a plain (non-mission) run or a stage with no reference
    # attached simply has no file, and `load_reference_signature` returns
    # None on any problem (missing/corrupt/wrong schema) — never raises.
    reference_signature: dict | None = None
    if not isinstance(config, dict):
        from sculptor.reference_context import load_reference_signature

        reference_signature = load_reference_signature(Path(config))

    # 2. Load the adapter to get the reward_contract.
    adapter = load_adapter(Path(config)) if not isinstance(config, dict) else None
    if adapter is None:
        raise ValueError(
            "diagnose() requires a config path that resolves to a SculptorAdapter; "
            "passing a dict isn't enough because we need reward_contract() from the "
            "instantiated adapter.")
    contract = adapter.reward_contract()
    contract_text = _render_reward_contract(contract)

    # §env generalization 3/4: the active env spec — but ONLY when it is
    # the loop-MANAGED per-project file (env/current.json next to the
    # config), because that is the surface apply_env_edits writes to. An
    # explicit config env_spec_path pinned elsewhere is static
    # configuration: showing it as editable would invite proposals that
    # can never land (or, worse, land on a file training doesn't read).
    # None → no # ENV_SPEC block; any env edits the model hallucinates
    # are dropped at packing below.
    env_spec: dict | None = None
    _env_spec_path = str(getattr(adapter, "env_spec_path", "") or "")
    if _env_spec_path:
        # Full-path resolve on BOTH sides (matches sculpt.py's condition
        # exactly — a symlinked env/ dir must not split the surface from
        # the apply target).
        _managed_spec = (
            Path(config).resolve().parent / "env" / "current.json").resolve()
        if Path(_env_spec_path).resolve() == _managed_spec:
            try:
                from sculptor.env_spec import load_env_spec

                env_spec = load_env_spec(_env_spec_path)
            except Exception as e:  # noqa: BLE001 — context is advisory here
                print(f"[diagnose] env spec unreadable ({e}) — no env-edit "
                      "surface this iter.", file=sys.stderr, flush=True)

    # 3. Anthropic client.
    if client is None:
        import anthropic
        # max_retries=6 (SDK default is 2) so transient 429 / 500 /
        # network blips during an overnight 12-iter run don't error
        # the whole thing — the Anthropic SDK retries with exponential
        # backoff internally (200ms, 400ms, 800ms, ...).
        client = anthropic.Anthropic(max_retries=6)

    # 4. Stage 1 — preliminary diagnosis.
    prelim_user = _build_preliminary_user_content(
        behavior_goal=behavior_goal,
        reward_spec=reward_spec,
        metrics=metrics,
        behavior=behavior,
        behavior_metric_names=behavior_metric_names,
        contract_text=contract_text,
        keyframes=keyframes,
        training_feedback=training_feedback,
        realism_audit=realism_audit,
        objective_progress=objective_progress,
        human_note=human_note,
        reference_signature=reference_signature,
    )
    preliminary: _PreliminaryModel = _parse_with_retry(
        client, model_cls=_PreliminaryModel,
        system_prompt=_PRELIM_SYSTEM, user_content=prelim_user,
        stage="preliminary",
    )

    # 5. KG retrieval — union of tag-based and semantic matches, deduped.
    #    Skipped when `skip_kg=True` (used for `--no-kg` ablations).
    owns_store = (store is None) and not skip_kg
    store = store or (None if skip_kg else SculptorKG())
    try:
        if skip_kg:
            kg_matches: list[TechniqueMatch] = []
            case_context = ""
        else:
            from sculptor.kg.query import DEFAULT_MIN_PROMPT_SIMILARITY

            # §Fix 3 (staleness rotation): when stuck, the excluded set is
            # ADDED to each source query's top_k so the merge below can
            # refill from the next-ranked results and still reach
            # KG_TOP_K where the underlying pool allows it. Empty set
            # (the common case, and always the case when not stuck) keeps
            # this byte-identical to KG_TOP_K.
            _stale_exclude_ids = _stale_uncited_technique_ids(
                iter_dir, objective_progress)
            _kg_fetch_top_k = KG_TOP_K + len(_stale_exclude_ids)

            fm_keywords = [fm for fm in preliminary.failure_modes if fm != "none"]

            # §KG-retrieval fix 2: resolve this iter's free-text
            # `failure_descriptors` onto FailureMode nodes via embedding
            # similarity (complementing the fixed 6-value enum fuzzy
            # resolution `_resolve_failure_modes` already does for
            # `fm_keywords`). Guarded exactly like the semantic query
            # below — a broken/missing embedder degrades to enum-only
            # tag context, never fails the diagnose call.
            extra_failure_ids: list[str] = []
            if preliminary.failure_descriptors:
                try:
                    from sculptor.kg.query import resolve_failure_modes_semantic

                    extra_failure_ids = resolve_failure_modes_semantic(
                        store, preliminary.failure_descriptors)
                except Exception as e:  # noqa: BLE001
                    print(f"[diagnose] descriptor FailureMode resolution "
                          f"failed ({e}) — enum-only tag context.",
                          file=sys.stderr, flush=True)
                    extra_failure_ids = []

            tag_matches = query_techniques(
                fm_keywords, domain_filter=env_tag, top_k=_kg_fetch_top_k,
                store=store, extra_failure_node_ids=extra_failure_ids,
            ) if (fm_keywords or extra_failure_ids) else []
            try:
                # §Ship 31: floored — an unfloored semantic slice feeds
                # tangential techniques into the grounded prompt and
                # Claude dutifully cites them (Issue G).
                sem_matches = query_semantic(
                    behavior_goal, top_k=_kg_fetch_top_k, store=store,
                    min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY)
            except Exception as e:  # noqa: BLE001
                print(f"[diagnose] semantic query failed ({e}) — tag-only context.",
                      file=sys.stderr, flush=True)
                sem_matches = []

            # §KG-retrieval fix 1: the goal-anchored `sem_matches` query
            # above is STATIC per stage (behavior_goal never changes
            # across a stage's iterations), so on its own it retrieves
            # identical literature every call. When stage-1 produced
            # actual evidence prose, ALSO run a semantic query anchored
            # on THIS iteration's evidence (+ failure-mode labels) so
            # retrieval tracks what's actually going wrong this iter.
            # Empty/whitespace evidence keeps behavior byte-identical to
            # before this fix (no extra query, no extra log record).
            evidence_matches: list[TechniqueMatch] = []
            _evidence_text = (preliminary.evidence or "").strip()
            _evidence_query = ""
            if _evidence_text:
                _evidence_query = _evidence_text[:400]
                if fm_keywords:
                    _evidence_query += " | " + ", ".join(fm_keywords)
                try:
                    evidence_matches = query_semantic(
                        _evidence_query, top_k=_kg_fetch_top_k, store=store,
                        min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY)
                    for m in evidence_matches:
                        # Distinguishes these hits from the static goal
                        # query in the rendered prompt / retrieval log —
                        # ranking/merge logic doesn't depend on this tag.
                        m.matched_on = ["semantic_evidence"]
                except Exception as e:  # noqa: BLE001
                    print(f"[diagnose] evidence-anchored semantic query "
                          f"failed ({e}) — goal/tag context only.",
                          file=sys.stderr, flush=True)
                    evidence_matches = []

            # Merge order: EVIDENCE-QUERY HITS FIRST (this iter's specific
            # failure), then tag hits (enum + descriptor FailureModes),
            # then the static goal hits — deduped by technique id,
            # truncated to KG_TOP_K total (unchanged). §Fix 3: a technique
            # id in `_stale_exclude_ids` is skipped here (not counted
            # toward KG_TOP_K) so the next-ranked match refills its slot;
            # skipped hits are kept in `_stale_excluded_hits` so the
            # ≥2-result floor below can re-admit them if exclusion would
            # otherwise starve the block.
            seen: set[str] = set()
            kg_matches = []
            _stale_excluded_hits: list[TechniqueMatch] = []
            _stale_excluded_seen: set[str] = set()
            for m in evidence_matches + tag_matches + sem_matches:
                if m.technique.id in seen:
                    continue
                if m.technique.id in _stale_exclude_ids:
                    if m.technique.id not in _stale_excluded_seen:
                        _stale_excluded_seen.add(m.technique.id)
                        _stale_excluded_hits.append(m)
                    continue
                seen.add(m.technique.id)
                kg_matches.append(m)
                if len(kg_matches) >= KG_TOP_K:
                    break

            # §Fix 3 floor: NEVER let staleness rotation exclude so much
            # that fewer than 2 literature matches remain — re-admit
            # excluded hits (original evidence>tag>goal priority order)
            # until the floor is met, and drop the re-admitted ids from
            # the exclusion set actually reported below.
            if _stale_exclude_ids and len(kg_matches) < 2:
                for m in _stale_excluded_hits:
                    if len(kg_matches) >= 2:
                        break
                    if m.technique.id in seen:
                        continue
                    kg_matches.append(m)
                    seen.add(m.technique.id)
                    _stale_exclude_ids.discard(m.technique.id)

            # §Agentic-data upgrade 3: retrieval trajectory log — durable
            # record of what the technique retrieval surfaced for this
            # iter's failure/behavior query. `iter_dir` is already in
            # scope here (this iteration's artifact directory) so it's
            # passed explicitly rather than falling back to llm_log_dir().
            from sculptor.kg.retrieval_log import log_retrieval

            log_retrieval(
                "diagnose", behavior_goal, kg_matches, out_dir=iter_dir)
            if _stale_exclude_ids:
                # §Fix 3: fires once per stuck iteration that actually had
                # >=1 shown-uncited technique to rotate out.
                log_retrieval(
                    decision="diagnose_stale_rotate",
                    query=",".join(sorted(_stale_exclude_ids)),
                    matches=kg_matches, out_dir=iter_dir)
                print(
                    f"[diagnose] stale-rotation: excluded "
                    f"{len(_stale_exclude_ids)} uncited technique(s) from "
                    "the prior iteration's literature block (stuck, "
                    "delta<=0) and refilled from next-ranked matches.",
                    file=sys.stderr, flush=True)
            if _evidence_text:
                log_retrieval(
                    decision="diagnose_evidence", query=_evidence_query,
                    matches=evidence_matches, out_dir=iter_dir)
            if extra_failure_ids:
                # Records the descriptor text as the query and the
                # resolved FailureMode NODES as matches (log_retrieval's
                # `_extract_match` falls back to `.id` for anything
                # without `.technique`/`.case`, which a FailureMode has).
                _fm_nodes = [
                    n for n in (store.get_node(nid) for nid in extra_failure_ids)
                    if n is not None
                ]
                log_retrieval(
                    decision="diagnose_descriptors",
                    query=" | ".join(preliminary.failure_descriptors),
                    matches=_fm_nodes, out_dir=iter_dir)

            # §Ship 37: case-memory — this system's OWN past runs on similar
            # tasks/failures (what was tried + whether it helped), additive to
            # the literature context. Floored like the semantic query so only
            # genuinely-similar cases reach the prompt.
            try:
                from sculptor.kg.cases import _render_case_context, query_cases
                from sculptor.kg.query import (
                    DEFAULT_MIN_PROMPT_SIMILARITY as _MIN_SIM,
                )
                _case_q = behavior_goal + (
                    " | " + ", ".join(fm_keywords) if fm_keywords else "")
                _case_matches = query_cases(
                    _case_q, top_k=3, store=store, min_similarity=_MIN_SIM)
                log_retrieval(
                    "diagnose", _case_q, _case_matches, out_dir=iter_dir)
                case_context = _render_case_context(_case_matches)
            except Exception as e:  # noqa: BLE001 — case memory is advisory
                print(f"[diagnose] case-memory query failed ({e}) — skipped.",
                      file=sys.stderr, flush=True)
                case_context = ""

        # 6. Stage 2 — grounded edits.
        grounded_user = _build_grounded_user_content(
            behavior_goal=behavior_goal,
            reward_spec=reward_spec,
            metrics=metrics,
            behavior=behavior,
            contract_text=contract_text,
            preliminary=preliminary,
            kg_context=(
                (case_context + "\n\n" if case_context else "")
                + _render_kg_context(kg_matches)
            ),
            training_feedback=training_feedback,
            realism_audit=realism_audit,
            env_spec=env_spec,
            reference_signature=reference_signature,
        )
        grounded: _GroundedModel = _parse_with_retry(
            client, model_cls=_GroundedModel,
            system_prompt=_GROUNDED_SYSTEM, user_content=grounded_user,
            stage="grounded",
        )

        # §Ship 31 (anti-hallucination): verify citations at the SOURCE.
        # A fabricated arxiv_id in proposed_edits would otherwise ride
        # into the edit phase and hard-fail its KG validation gate —
        # burning a retry (or the iteration) on a reference the model
        # invented. Unknown ids are DROPPED here (the edit degrades to
        # novel/uncited), observably via a kg_citation_dropped event.
        _dropped_refs: dict[str, list[str]] = {}
        if grounded.proposed_edits:
            from sculptor.kg.schema import make_paper_id

            for e in grounded.proposed_edits:
                if not e.paper_refs:
                    continue
                keep: list[str] = []
                missing: list[str] = []
                for aid in e.paper_refs:
                    known = (
                        store is not None
                        and store.get_node(make_paper_id(str(aid))) is not None
                    )
                    (keep if known else missing).append(str(aid))
                if missing:
                    _dropped_refs.setdefault(e.target_term, []).extend(missing)
                    e.paper_refs = keep
        if _dropped_refs:
            print("[SCULPT-EVENT] " + json.dumps({
                "type": "kg_citation_dropped",
                "dropped": _dropped_refs,
                "reason": "cited arxiv_id not present in the KG",
            }, default=str), flush=True)

        # §KG-retrieval fix 5: citation grounding annotation. A KEPT
        # paper_ref (survived the existence check above) is `grounded`
        # iff its arxiv_id is among the papers THIS iteration's retrieved
        # literature_context (`kg_matches`) actually showed Claude —
        # False means it exists in the KG but Claude recalled it rather
        # than being shown it (still a valid, kept citation).
        _retrieved_arxiv_ids: set[str] = set()
        for m in kg_matches:
            for pid in (m.source_paper_ids or []):
                if isinstance(pid, str) and pid.startswith("paper:"):
                    _retrieved_arxiv_ids.add(pid[len("paper:"):])
    finally:
        if owns_store and store is not None:
            store.close()

    # 7. Pack into Diagnosis dataclass and persist to disk.
    diagnosis = Diagnosis(
        failure_modes=[str(fm) for fm in preliminary.failure_modes],
        evidence=preliminary.evidence,
        proposed_edits=[
            ProposedEdit(
                target_term=e.target_term,
                operation=e.operation,
                rationale=e.rationale,
                suggested_value=e.suggested_value,
                paper_refs=list(e.paper_refs),
                paper_refs_grounded={
                    aid: (aid in _retrieved_arxiv_ids) for aid in e.paper_refs
                },
                # BUG FIX (2026-07-04, found during env generalization
                # 3/4): the flag was silently dropped here since Ship 48
                # — deferred edits lost requires_env_extension on the
                # REAL path, so apply_edits tried (and failed) their
                # ungrounded formulas and the never-silent
                # requires_env_extension event could not fire.
                requires_env_extension=bool(e.requires_env_extension),
            )
            for e in grounded.proposed_edits
        ],
        # §env generalization 3/4: env-curriculum proposals. Only
        # meaningful when a spec is active — without one there is no
        # # ENV_SPEC block, and anything the model emitted anyway is
        # dropped here (no surface to apply it to).
        proposed_env_edits=(
            [
                ProposedEnvEdit(
                    parameter=str(e.parameter),
                    new_value=str(e.new_value),
                    rationale=e.rationale,
                )
                for e in grounded.proposed_env_edits
            ]
            if env_spec is not None else []
        ),
        literature_context=kg_matches,
        confidence=min(preliminary.confidence, grounded.confidence),
        iter_dir=str(iter_dir),
        behavior_goal=behavior_goal,
    )
    out_path = iter_dir / "diagnosis.json"
    try:
        out_path.write_text(
            json.dumps(diagnosis.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[diagnose] warning: could not write {out_path}: {e}",
              file=sys.stderr, flush=True)
    return diagnosis


# ── Pretty-printer for scripts ────────────────────────────────────────────
def print_diagnosis(d: Diagnosis, *, stream=sys.stdout) -> None:
    w = stream.write
    w("=" * 72 + "\n")
    w(f"Diagnosis for {d.iter_dir}\n")
    w(f"goal: {d.behavior_goal}\n")
    w(f"confidence: {d.confidence:.2f}\n")
    w("=" * 72 + "\n")
    w(f"failure_modes: {d.failure_modes}\n")
    w("\n")
    w("evidence:\n")
    for line in (d.evidence or "").splitlines() or [""]:
        w(f"  {line}\n")
    w("\n")
    w(f"literature_context ({len(d.literature_context)} hit(s)):\n")
    for i, m in enumerate(d.literature_context, 1):
        w(f"  {i}. {m.technique.name}  score={m.relevance_score:.3f}  "
          f"{m.paper_citation}\n")
    w("\n")
    w(f"proposed_edits ({len(d.proposed_edits)}):\n")
    for i, e in enumerate(d.proposed_edits, 1):
        w(f"  {i}. [{e.operation}] {e.target_term}")
        if e.suggested_value is not None:
            w(f"  -> {e.suggested_value}")
        w("\n")
        w(f"     rationale: {e.rationale}\n")
        if e.paper_refs:
            w(f"     paper_refs: {e.paper_refs}\n")
        else:
            w(f"     paper_refs: (novel)\n")
