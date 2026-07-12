"""sculptor/decompose.py — Claude-backed task decomposition (Ship 14).

Public entry: `decompose_task(goal, reward_contract, *, kg_store=None,
client=None, model=MODEL_ID) -> Mission`.

Claude receives the goal + the adapter's reward_contract + a slice of the
KG literature, and returns a JSON curriculum of stages. Each stage is
validated against the grounded-field rules in `sculptor.mission` before
the Mission is returned; validation failure raises
`MissionValidationError` rather than silently truncating or retrying —
the orchestrator (Ship 16) can implement retry on top.

Architectural references:
  - CurricuLLM (arXiv:2409.18382) — decomposition + per-stage reward +
    warm-start; we mirror the curriculum-generation LLM role here.
  - sculptor.diagnose._parse_with_retry — the existing pydantic-parse +
    one-retry pattern we reuse for this LLM call.

Ship 14 scope: implement the Claude call, validators, and Mission
construction. NO orchestrator, NO training integration, NO UI. Those
land in Ships 15-18.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from sculptor.llm import log_llm_call, model_for, response_text_blocks
from sculptor.mission import (
    MAX_SEED_PROMPT_CHARS,
    MIN_SEED_PROMPT_CHARS,
    Mission,
    MissionValidationError,
    Stage,
    validate_mission,
)
from sculptor.prompts import load_prompt


MODEL_ID = model_for("decompose")
MAX_TOKENS = 8000
# Cap on how many KG techniques we surface to the decomposer. Matches
# the `KG_TOP_K` used in diagnose.py for symmetry — bigger slices add
# noise without measurably helping curriculum design.
KG_CONTEXT_TOP_K = 8


# ── Pydantic shapes (what Claude is constrained to emit) ─────────────
class _StageModel(BaseModel):
    name: str = Field(
        description=(
            "snake_case identifier ≤ 32 chars, letter-first. Used as a "
            "filesystem + reward-component namespace."
        )
    )
    goal_text: str = Field(
        description="Natural-language description of what THIS stage achieves."
    )
    success_criterion: str = Field(
        description=(
            "Python boolean expression evaluated on rollout trajectory. "
            "May reference info['<expected_info_key>'], reward components "
            "introduced by this stage's reward_seed_prompt, and standard "
            "math helpers. See decompose_task.md for the full rules."
        )
    )
    max_iterations: int = Field(
        ge=1, le=50,
        description="Sculpt-run iteration budget before escalating to fail/re-decompose.",
    )
    parent_stage: Optional[str] = Field(
        default=None,
        description=(
            "Name of an earlier stage whose final policy warm-starts this one. "
            "None for cold-start stages (the first stage MUST be None)."
        ),
    )
    reward_seed_prompt: str = Field(
        min_length=MIN_SEED_PROMPT_CHARS,
        max_length=MAX_SEED_PROMPT_CHARS,
        description=(
            "NL reward spec passed to apply_prompt_edit as this stage's "
            "v0 seed. Must obey the grounded-field rule."
        ),
    )
    kg_seed_papers: list[str] = Field(
        default_factory=list,
        description="arXiv IDs (e.g., '2312.17507') from the KG slice provided.",
    )
    # §Ship 19: optional cross-mission skill-library reference. The
    # decomposer may pick a `skill_id` from the SKILL_LIBRARY block
    # (when one is rendered into the user content) to warm-start this
    # stage from a prior trained policy. Empty / whitespace strings
    # are normalized to None so a Claude response of `""` is treated
    # the same as omitting the field (audit fix H4).
    init_skill_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional skill_id from the SKILL_LIBRARY context block. "
            "Setting this warm-starts the stage from a prior mission's "
            "trained policy. Must match an id in the rendered library "
            "slice; unknown ids are rejected at validation time."
        ),
    )

    # §Ship 38: optional per-stage objective metric. Lets a multi-phase
    # curriculum steer each phase by ITS OWN objective instead of one
    # uniform mission metric. Conservative by design — only a correct fit.
    steering_metric: Optional[str] = Field(
        default=None,
        description=(
            "OPTIONAL. The name of a built-in spec metric (see "
            "AVAILABLE_FITNESS_METRICS) that DIRECTLY and CORRECTLY measures "
            "THIS stage's sub-goal for THIS robot. Set it ONLY when a listed "
            "metric clearly fits the stage; otherwise OMIT (null) — the "
            "mission-level metric / success criterion is used. A wrong-robot "
            "or loosely-related metric MIS-steers the stage; when unsure, null."
        ),
    )

    # §JUMP_SCAFFOLD: DeepMimic-style reference-state initialization for
    # hard-exploration stages. Every published G1 jump rides a reference
    # motion (RESEARCH_GAP_ANALYSIS §4.5); RSI is the cheapest transfer —
    # start a fraction of training episodes INSIDE the airborne/landing
    # phase so the policy experiences apex → descent → touchdown long
    # before it can produce a launch. When true, the orchestrator derives
    # a validated train-only RSI curriculum from a reference clip
    # (project clip if present, procedural jump otherwise) and applies it
    # to THIS stage's env spec before training. Rollout evaluation is
    # untouched (schema-level shared/train split).
    needs_reference_rsi: bool = Field(
        default=False,
        description=(
            "Set true for a stage whose core skill involves ballistic/"
            "airborne states the robot cannot yet reach (jump launch, "
            "flight, landing), OR whose START state is far from the "
            "robot's default standing reset (lying, sitting, crouched, "
            "fallen) — e.g. a get-up mission's first stage MUST set this, "
            "since the env resets standing and the stage is otherwise "
            "untrainable. Starts a fraction of training episodes in those "
            "states via a validated reference curriculum. False for "
            "grounded skills that start from the default standing pose "
            "(standing, walking, crouching). MUST be true whenever "
            "`start_pose` below is anything other than 'standing' — "
            "validation forces it true even if you leave it false, so set "
            "it explicitly for consistency."
        ),
    )

    # §start_pose: the physical configuration the robot is in at THIS
    # stage's episode start. Locked-vocabulary companion to
    # `needs_reference_rsi` — see that field's updated description.
    start_pose: Optional[str] = Field(
        default=None,
        description=(
            "The physical configuration the robot is in at THIS stage's "
            "episode start. One of 'supine' (lying on back, face up), "
            "'prone' (lying on front, face down), 'sitting', 'crouched', "
            "'standing', or null (defaults to standing behavior at "
            "validation, but prefer 'standing' explicitly when you know "
            "it). Derive this from the MISSION GOAL and THIS STAGE's "
            "semantics — phrases like 'starting prone', 'from a seated "
            "position', 'get up off the ground', 'lying on your back' "
            "must flow into the matching stage's start_pose. Standing "
            "unless the mission or stage goal says otherwise. A "
            "MULTI-STAGE get-up curriculum's sub-goals often have "
            "DIFFERENT start poses as the motion progresses stage to "
            "stage (e.g. stage 1 supine -> stage 2 crouched -> stage 3 "
            "standing) — set EACH stage's start_pose to what THAT "
            "stage's episode actually begins from, not the mission's "
            "overall starting pose. Any value other than 'standing' "
            "means this stage is untrainable from the env's default "
            "reset and REQUIRES needs_reference_rsi=true (see that "
            "field)."
        ),
    )

    @field_validator("init_skill_id", mode="before")
    @classmethod
    def _normalize_init_skill_id(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("steering_metric", mode="before")
    @classmethod
    def _normalize_steering_metric(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("start_pose", mode="before")
    @classmethod
    def _normalize_start_pose(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().lower()
        return s or None


class _DecompositionModel(BaseModel):
    decomposition_rationale: str
    stages: list[_StageModel]


# §Ship 17 — same shape with stricter sub-stage count bounds. Pydantic
# enforces 2-8 at parse time so an out-of-range Claude response triggers
# the _parse_with_retry path before any downstream validation.
class _RedecompositionModel(BaseModel):
    decomposition_rationale: str
    stages: list[_StageModel] = Field(min_length=2, max_length=8)


class DecompositionError(MissionValidationError):
    """Raised on unrecoverable decomposition-call failure.

    Subclasses MissionValidationError so callers that catch the parent
    class get both syntactic (schema) and semantic (validation) errors
    uniformly. Semantic-only callers can still catch
    MissionValidationError.
    """


# ── KG context rendering ─────────────────────────────────────────────
def _render_kg_context(
    goal: str,
    kg_store: Any,
    top_k: int = KG_CONTEXT_TOP_K,
) -> tuple[str, list[str]]:
    """Build the KG literature slice passed to Claude.

    Returns (markdown_block, available_arxiv_ids). The arxiv_id list
    is used by `_validate_kg_seed_papers` to reject citations Claude
    invents rather than reusing from the slice.
    """
    if kg_store is None:
        return (
            "# KG LITERATURE CONTEXT\n"
            "(no KG store supplied — kg_seed_papers must be empty)\n",
            [],
        )
    try:
        from sculptor.kg.query import (
            DEFAULT_MIN_PROMPT_SIMILARITY,
            query_semantic,
        )
        # §Ship 31: floored — same Issue-G rationale as diagnose/edit.
        matches = query_semantic(
            goal, top_k=top_k, store=kg_store,
            min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY)
        # §Agentic-data upgrade 3: retrieval trajectory log. No iter_dir
        # context here (decompose runs before any iteration exists), so
        # out_dir=None falls back to sculptor.llm.llm_log_dir().
        from sculptor.kg.retrieval_log import log_retrieval

        log_retrieval("decompose", goal, matches, out_dir=None)
    except Exception as e:  # noqa: BLE001
        print(
            f"[decompose] KG query failed ({type(e).__name__}: {e}) — "
            "decomposer will see an empty literature slice.",
            file=sys.stderr, flush=True,
        )
        return (
            "# KG LITERATURE CONTEXT\n"
            "(KG query failed; kg_seed_papers must be empty)\n",
            [],
        )

    if not matches:
        return (
            "# KG LITERATURE CONTEXT\n"
            "(no techniques matched the goal above the similarity floor; "
            "kg_seed_papers should be empty)\n",
            [],
        )

    lines = ["# KG LITERATURE CONTEXT",
             f"Top {len(matches)} techniques by semantic similarity to the goal:",
             ""]
    arxiv_ids: set[str] = set()
    for m in matches:
        for pid in m.source_paper_ids or []:
            # source_paper_ids are internal IDs like "paper:2312.17507".
            if pid.startswith("paper:"):
                arxiv_ids.add(pid[len("paper:"):])
        lines.append(
            f"- **{m.technique.name}** (sim={m.relevance_score:.2f}) — "
            f"{(m.description or '').strip()[:200]}"
        )
        if m.paper_citation:
            lines.append(f"  • {m.paper_citation}")
    lines.append("")
    lines.append(
        "You may cite any of the following arXiv IDs in kg_seed_papers "
        f"(citing others will fail validation): {sorted(arxiv_ids)}"
    )
    return "\n".join(lines), sorted(arxiv_ids)


def _render_contract(contract: Any) -> str:
    info_keys = list(getattr(contract, "expected_info_keys", None) or [])
    components = getattr(contract, "expected_components", None)
    supports_batched = bool(getattr(contract, "supports_batched", False))
    components_text = (
        "OPEN (adapter does not constrain component names)"
        if components is None else sorted(components)
    )
    # §criterion grounding (smoke-hop-stop finding, 2026-07-06): the
    # decomposer wrote `trajectory['root_link_pos_w']` criteria for a
    # gym Hopper project — that key exists only on mjlab articulated
    # envs, so every criterion eval raises CriterionMissingKeyError and
    # the stage can never early-stop. Tell the decomposer which
    # trajectory keys THIS adapter actually persists. supports_batched
    # is the honest available proxy for "mjlab articulated env".
    if supports_batched:
        # §root-height channel (smoke-hop-stop finding, 2026-07-06):
        # `root_height` is a DERIVED, unambiguous 1-D per-step root z the
        # runtime synthesizes from `root_link_pos_w`; criteria on base/root
        # height MUST use it (raw `root_link_pos_w[..., 2]` is the full
        # (T, E) grid across every env — `.any()` fires on teleport spikes).
        traj_keys = sorted(
            {"rewards", "episode_id", "joint_pos", "joint_vel", "action",
             "actuator_force", "projected_gravity_b", "root_link_pos_w",
             "root_height"})
        traj_note = (
            "  (base/root-HEIGHT criteria MUST use trajectory['root_height'] "
            "— a 1-D per-step root z; do NOT write root_link_pos_w[..., 2], "
            "which is the (T, E) grid over ALL envs)\n"
        )
    else:
        traj_keys = ["episode_id", "rewards"]
        traj_note = (
            "  (this adapter persists ONLY these — do NOT reference "
            "joint_pos / root_link_pos_w / other articulated-env keys "
            "in success criteria; use components/behavior instead)\n"
        )
    return (
        "# REWARD_CONTRACT\n"
        f"expected_info_keys: {sorted(info_keys)}\n"
        f"expected_components: {components_text}\n"
        f"supports_batched:   {supports_batched}\n"
        f"available_trajectory_keys: {traj_keys}\n"
        f"{traj_note}"
    )


def _build_user_content(
    goal: str,
    contract: Any,
    kg_context: str,
    skill_context: str = "",
    fitness_metrics_block: str = "",
    reference_context: str = "",
) -> str:
    skill_block = f"{skill_context}\n\n" if skill_context else ""
    metrics_block = f"{fitness_metrics_block}\n\n" if fitness_metrics_block else ""
    reference_block = f"{reference_context}\n\n" if reference_context else ""
    return (
        f"# BEHAVIOR_GOAL\n{goal}\n\n"
        f"{_render_contract(contract)}\n"
        f"{kg_context}\n\n"
        f"{metrics_block}"
        f"{skill_block}"
        f"{reference_block}"
        "Emit the decomposition JSON now. Ordered stages only, "
        "topologically valid parent_stage refs, final stage must satisfy "
        "the behavior goal end-to-end."
    )


# ── §REFERENCE_TRAJECTORY_PLAN §3/§4/§7: mission-goal-level retrieval ────
#: Retrieval-confidence floor for AUTO-attaching a match to a stage
#: without human review (§4.3's "auto-accept only above a high confidence
#: bar" + §7's build brief's ">= 0.8"). Below this, a match is surfaced in
#: the prompt context for GROUNDING only — the stage fields stay None so
#: the UI picker remains the approval surface for anything less certain.
REFERENCE_ATTACH_CONFIDENCE = 0.8
#: How many mission-goal-level matches to retrieve for prompt grounding.
_REFERENCE_RETRIEVE_K = 3


def _retrieve_mission_references(
    goal: str, robot_hint: Optional[str],
) -> list[Any]:
    """§decompose_task's single LLM call drafts `success_criterion`
    alongside `goal_text`/etc in the SAME `_DecompositionModel` parse
    (see module docstring) — there is no separate later call to ground.
    So retrieval happens ONCE, up front, against the MISSION's
    `goal_text` (not per-stage — stages don't exist yet), and the top
    matches' kinematic signatures are injected into the SAME decompose
    prompt with instructions to ground stage criteria in them.

    Returns `[]` on ANY failure (no library index on disk, retrieval
    exception, import failure) — reference grounding is a strict
    ADDITION, never a decompose blocker. Never raises."""
    try:
        from sculptor.refs import library, retrieve

        robot = _reference_robot_slug(robot_hint)
        if not library.index_path().is_file():
            return []  # §build brief: only retrieve when an index exists on disk
        matches = retrieve.search(goal, robot=robot, k=_REFERENCE_RETRIEVE_K)
        return list(matches or [])
    except Exception as e:  # noqa: BLE001 — retrieval must never block decompose
        print(
            f"[decompose] reference retrieval failed "
            f"({type(e).__name__}: {e}) — decomposer proceeds without "
            "reference grounding.", file=sys.stderr, flush=True,
        )
        return []


#: Same robot-hint -> library-slug convention as `mission_metrics._robot_slug`
#: (kept local — decompose.py must not import mission_metrics, which would
#: create a cycle back through sculptor.mission). Default "g1" mirrors the
#: `robot: str = "g1"` default used throughout `sculptor.refs`.
_REFERENCE_ROBOT_SLUGS: tuple[str, ...] = ("go2", "go1", "g1")


def _reference_robot_slug(robot_hint: Optional[str]) -> str:
    h = str(robot_hint or "").lower()
    for slug in _REFERENCE_ROBOT_SLUGS:
        if slug in h:
            return slug
    return "g1"


def _load_reference_clip(clip_id: str, robot: str) -> Optional[dict]:
    """Load one library clip by id, or None on any failure (missing/
    corrupt clip on disk must never crash decompose)."""
    try:
        from sculptor.reference import load_clip
        from sculptor.refs import library

        path = library.clip_dir(robot, clip_id) / library.CLIP_FILENAME
        return load_clip(path)
    except Exception:  # noqa: BLE001 — a bad clip degrades to no signature
        return None


def _render_reference_context(matches: list[Any], robot: str) -> str:
    """Render retrieved reference matches into a `REFERENCE MOTION
    SIGNATURES` block: kinematic signatures for grounding stage
    success_criterion thresholds (§7), plus each match's retrieval
    metadata (clip_id, match_confidence, tier) so the decomposer can see
    WHICH clips are confident enough to be worth grounding against.
    Empty string when there are no matches or nothing loads."""
    if not matches:
        return ""
    from sculptor.refs.convert import kinematic_signature

    lines = [
        "# REFERENCE MOTION SIGNATURES",
        "Real mocap/retargeted clips retrieved for this goal. GROUND every "
        "numeric success_criterion threshold (heights, durations, "
        "velocities) that corresponds to one of these clips' motion in its "
        "actual signature values instead of guessing. Also use a "
        "signature's orientation/phase data to judge whether a stage's "
        "START state is far from the robot's default standing reset — see "
        "the needs_reference_rsi rule.",
        "",
    ]
    rendered_any = False
    for m in matches:
        clip_id = getattr(m, "clip_id", None)
        if not clip_id:
            continue
        clip = _load_reference_clip(clip_id, robot)
        if clip is None:
            continue
        try:
            sig = kinematic_signature(clip)
        except Exception:  # noqa: BLE001 — one bad clip must not block others
            continue
        rendered_any = True
        lines.append(
            f"## clip_id: {clip_id} "
            f"(match_confidence={getattr(m, 'match_confidence', None)}, "
            f"tier={getattr(m, 'tier', None)})"
        )
        lines.append(json.dumps(sig, indent=2, default=str))
        lines.append("")
    return "\n".join(lines) if rendered_any else ""


def _render_fitness_metrics_block() -> str:
    """§Ship 38: the list of built-in objective metrics the decomposer may
    assign per stage (conservatively)."""
    from sculptor.eval.spec_metrics import spec_metric_names

    return (
        "# AVAILABLE_FITNESS_METRICS\n"
        "Optional per-stage objectives you MAY assign to a stage's "
        "`steering_metric` when one DIRECTLY measures that stage's sub-goal "
        f"for this robot: {spec_metric_names()}. They are robot-specific "
        "(g1_* are humanoid; go1_trot is a quadruped gait; cartpole_balance "
        "is cartpole). Assign ONLY a correct fit; otherwise leave it null."
    )


# ── Ship 19: skill-library context rendering ─────────────────────────
def _render_skill_library_context(handle: Any) -> tuple[str, list[str]]:
    """Build the SKILL_LIBRARY block injected into Claude's user content.

    `handle` is a `sculptor.skill_library.SkillLibraryHandle` (typed
    structurally to avoid the heavy import at module load); when None
    or when its library has no compatible records, we emit a small
    "no compatible skills" stub so Claude is told the field is
    available but empty (so an `init_skill_id` would be rejected by
    `_validate_skill_ids`).

    Returns (markdown_block, available_skill_ids).
    """
    if handle is None:
        return "", []
    try:
        records = handle.list_for_decompose(top_k=5)
    except Exception as e:  # noqa: BLE001
        # Library read failures must NOT poison the whole decompose
        # call (a corrupt skill record shouldn't block missioning).
        print(
            f"[decompose] skill-library read failed "
            f"({type(e).__name__}: {e}) — Claude sees no skills.",
            file=sys.stderr, flush=True,
        )
        return (
            "# SKILL_LIBRARY\n"
            "(skill-library read failed; init_skill_id must be null)\n",
            [],
        )
    if not records:
        return (
            "# SKILL_LIBRARY\n"
            f"(no compatible skills for adapter={handle.adapter_class}, "
            f"task_id={handle.task_id}; init_skill_id must be null)\n",
            [],
        )
    lines = [
        "# SKILL_LIBRARY",
        f"Top {len(records)} compatible prior-mission skills "
        f"(adapter={handle.adapter_class}, task_id={handle.task_id}). "
        "Reference one by id in any stage's `init_skill_id` to warm-start "
        "from its trained policy. Citations are validated; unknown "
        "skill_ids will be rejected.",
        "",
    ]
    available_ids: list[str] = []
    for r in records:
        available_ids.append(r.skill_id)
        seed = (r.reward_seed_prompt or "").strip().replace("\n", " ")
        if len(seed) > 240:
            seed = seed[:240] + "…"
        lines.append(
            f"- **{r.skill_id}** (final_metric={r.final_metric:.3f}, "
            f"iters={r.iterations_used}, "
            f"from_stage={r.source_stage_name!r}, "
            f"from_mission_goal={(r.source_mission_goal or '')[:80]!r})"
        )
        lines.append(f"  • criterion: `{r.success_criterion}`")
        lines.append(f"  • seed_prompt: {seed}")
    lines.append("")
    lines.append(
        "May reference any of these skill_ids: " + str(available_ids)
    )
    return "\n".join(lines), available_ids


def _validate_skill_ids(
    stages: list[Stage],
    available_skill_ids: set[str],
) -> None:
    """Reject any stage whose `init_skill_id` is not in the rendered
    library slice. Empty / None init_skill_id is fine. This catches
    Claude inventing or misremembering an id (audit fix to BIGGEST
    HOLE — without this validator a stale id would silently surface
    as `skill_warm_start_skipped(reason="skill_not_found")` at
    runtime, which is the WRONG layer to catch a decomposition bug).
    """
    offenders: list[tuple[str, str]] = []
    for s in stages:
        if s.init_skill_id and s.init_skill_id not in available_skill_ids:
            offenders.append((s.name, s.init_skill_id))
    if offenders:
        raise MissionValidationError(
            "init_skill_id references unknown skills: "
            + ", ".join(f"{n}→{sid!r}" for n, sid in offenders)
            + ". Re-decompose restricting init_skill_id to the rendered "
            "SKILL_LIBRARY slice, or omit the field."
        )


# ── Claude call (pydantic-parse + one retry) ────────────────────────
def _parse_with_retry(
    client: Any,
    system_prompt: str,
    user_content: str,
    *,
    output_format: Any = None,
    model: str = MODEL_ID,
    max_tokens: int = MAX_TOKENS,
):
    """Mirror of diagnose._parse_with_retry — one retry on parse failure.

    `output_format` defaults to `_DecompositionModel` so existing
    `decompose_task` callers stay unchanged. Ship 17 passes
    `_RedecompositionModel` for stricter sub-stage count bounds.
    """
    if output_format is None:
        output_format = _DecompositionModel
    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_format=output_format,
        )
        log_llm_call(
            "decompose", model, system=system_prompt, user=user_content,
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None), meta={"attempt": 1})
        return resp.parsed_output
    except Exception as first_err:  # noqa: BLE001
        retry_reminder = (
            "Previous response failed parse / validation: "
            f"{first_err!s}. Emit ONLY the strict JSON matching "
            "the schema — no prose, no markdown fences."
        )
        retry_messages = [
            {"role": "user", "content": user_content},
            {"role": "user", "content": retry_reminder},
        ]
        resp = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=retry_messages,
            output_format=output_format,
        )
        log_llm_call(
            "decompose", model, system=system_prompt,
            user=f"{user_content}\n\n{retry_reminder}",
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None), meta={"attempt": 2})
        return resp.parsed_output


# ── Validation helpers (on top of mission.validate_mission) ──────────
def _validate_kg_seed_papers(
    stages: list[Stage],
    available_arxiv_ids: set[str],
    kg_store: Any,
) -> None:
    """Ensure every cited arxiv_id is actually in the KG.

    `available_arxiv_ids` is what `_render_kg_context` surfaced — Claude
    should restrict citations to that slice, but we also verify against
    the live KG in case Claude substitutes a paper it knows offhand.
    """
    if kg_store is None:
        # If no store provided, reject any citations — we can't verify them.
        offenders: list[tuple[str, str]] = [
            (s.name, aid) for s in stages for aid in s.kg_seed_papers
        ]
        if offenders:
            raise MissionValidationError(
                "kg_seed_papers cited without a kg_store: "
                + ", ".join(f"{n}→{a}" for n, a in offenders)
            )
        return

    from sculptor.kg.schema import make_paper_id

    offenders = []
    for stage in stages:
        for aid in stage.kg_seed_papers:
            if kg_store.get_node(make_paper_id(aid)) is None:
                offenders.append((stage.name, aid))
    if offenders:
        raise MissionValidationError(
            "kg_seed_papers reference arxiv_ids not in the KG: "
            + ", ".join(f"{n}→{a}" for n, a in offenders)
            + ". Re-decompose restricting citations to the provided KG slice, "
            "or ingest the missing papers first."
        )


def _stages_from_model(stages_in: list[_StageModel]) -> list[Stage]:
    return [
        Stage(
            name=s.name,
            goal_text=s.goal_text,
            success_criterion=s.success_criterion,
            max_iterations=s.max_iterations,
            parent_stage=s.parent_stage,
            reward_seed_prompt=s.reward_seed_prompt,
            kg_seed_papers=list(s.kg_seed_papers or []),
            init_skill_id=s.init_skill_id,
            steering_metric=s.steering_metric,
            needs_reference_rsi=bool(s.needs_reference_rsi),
            start_pose=s.start_pose,
        )
        for s in stages_in
    ]


def _validate_steering_metrics(stages: list[Stage]) -> None:
    """§Ship 38: a decomposer-authored `steering_metric` must name a KNOWN
    built-in spec metric. Generated-metric paths are assigned by the UI /
    backend (which resolves + calibrates them), never invented by the
    decomposer. An unknown name triggers the parse-retry path rather than a
    fail-fast crash deep in mission_run."""
    from sculptor.eval.spec_metrics import spec_metric_names

    known = set(spec_metric_names())
    offenders = [
        (s.name, s.steering_metric) for s in stages
        if s.steering_metric and s.steering_metric not in known
    ]
    if offenders:
        raise MissionValidationError(
            "steering_metric references unknown spec metrics: "
            + ", ".join(f"{n}→{m!r}" for n, m in offenders)
            + f". Use one of {sorted(known)} or omit the field."
        )


def _attach_stage_references(stages: list[Stage], robot: str) -> None:
    """§REFERENCE_TRAJECTORY_PLAN §4/§7/§9: for each drafted stage, run
    refs retrieval against ITS OWN `goal_text` and attach the top match's
    fields (`reference_clip_id`/`tier`/`match_confidence`) ONLY when
    `match_confidence` is not None and >= `REFERENCE_ATTACH_CONFIDENCE`,
    OR when the deterministic-only layer's top score is a clear standout
    (>= 2x the second-place score, mirroring the "clear standout" build
    brief — a single strong deterministic hit with no close competitor is
    trustworthy even without an LLM confidence number). Otherwise the
    stage's reference fields stay None — the UI picker remains the
    approval surface for anything less certain.

    Runs AFTER stages are drafted+parsed (so it sees the LLM's actual
    per-stage goal_text), mutates `stages` in place. Any retrieval
    failure for a stage (or globally, e.g. no library index) leaves that
    stage's fields untouched — never raises."""
    for stage in stages:
        try:
            from sculptor.refs import library, retrieve

            if not library.index_path().is_file():
                return  # no library at all — skip every remaining stage too
            matches = retrieve.search(
                stage.goal_text, robot=robot, k=2)
        except Exception:  # noqa: BLE001 — retrieval must never block decompose
            continue
        if not matches:
            continue
        top = matches[0]
        confidence = getattr(top, "match_confidence", None)
        is_clear_standout = False
        if confidence is None and len(matches) >= 2:
            second_score = getattr(matches[1], "score", 0.0) or 0.0
            top_score = getattr(top, "score", 0.0) or 0.0
            is_clear_standout = second_score > 0 and top_score >= 2 * second_score
        elif confidence is None:
            is_clear_standout = True  # only one candidate at all — no competitor to beat
        if (confidence is not None and confidence >= REFERENCE_ATTACH_CONFIDENCE) or (
                confidence is None and is_clear_standout):
            stage.reference_clip_id = getattr(top, "clip_id", None)
            stage.reference_tier = getattr(top, "tier", None)
            stage.reference_match_confidence = confidence


# ── Public entry ─────────────────────────────────────────────────────
def decompose_task(
    goal: str,
    reward_contract: Any,
    *,
    kg_store: Any = None,
    client: Any = None,
    model: str = MODEL_ID,
    skill_library_handle: Any = None,
    robot_hint: Optional[str] = None,
    attach_references: bool = True,
) -> Mission:
    """Ask Claude to decompose `goal` into a Mission curriculum.

    Parameters
    ----------
    goal : the user's complex behavior goal.
    reward_contract : the adapter's RewardContract (has
        expected_info_keys + expected_components).
    kg_store : optional SculptorKG. When provided, top-k semantic matches
        are rendered into the decomposer's context and stage
        citations are verified against the live KG. When None, the
        decomposer is told to emit empty `kg_seed_papers` lists.
    client : optional Anthropic client (for testing / reuse). If None,
        constructs a client with max_retries=2 + timeout=240 s (same
        envelope as the post-Ship-13 edit path).
    model : Anthropic model id. Defaults to the registry's "decompose"
        role (`sculptor.llm.model_for`).
    skill_library_handle : optional `SkillLibraryHandle` (Ship 19).
        When provided, Claude sees up to 5 prior-mission skills
        compatible with the handle's (adapter_class, task_id) pair
        and may set a stage's `init_skill_id` to warm-start from one.
        Unknown ids are rejected at validation time.
    robot_hint : optional adapter/task_id-shaped robot hint (e.g.
        "Mjlab-Velocity-Flat-Unitree-G1"), used ONLY to resolve which
        reference-library robot slug to query (§REFERENCE_TRAJECTORY_PLAN
        §4); mapped via the same substring convention as
        `sculptor.eval.robot_manifest.robot_joint_names`, defaulting to
        "g1". Unrelated to the LLM call itself.
    attach_references : (default True) §REFERENCE_TRAJECTORY_PLAN §4/§7.
        When True: (a) retrieves top mission-goal-level reference matches
        UP FRONT and injects their kinematic signatures into THIS SAME
        decompose call's prompt so `success_criterion` thresholds can be
        grounded in real numbers (decompose's stage-drafting and
        criterion-authoring happen in ONE `_DecompositionModel` parse —
        there is no separate later call to ground); (b) AFTER stages are
        drafted, retrieves per-stage and attaches
        `reference_clip_id`/`reference_tier`/`reference_match_confidence`
        to a stage only on a high-confidence or clear-standout match
        (`_attach_stage_references`). Set False to disable both (e.g. in
        tests, or when the caller wants a bare decomposition). A missing
        library index or any retrieval failure is always a no-op, never
        an error, regardless of this flag.

    Returns
    -------
    Mission with every Stage in `status="pending"`. Caller (the Ship-16
    orchestrator) handles training.

    Raises
    ------
    MissionValidationError / DecompositionError on any structural or
    grounding violation. Callers can retry by invoking this function
    again — no internal retry loop beyond the single parse retry.
    """
    goal = (goal or "").strip()
    if not goal:
        raise DecompositionError("goal is empty")

    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    system_prompt = load_prompt("decompose_task")
    kg_context, available_ids = _render_kg_context(goal, kg_store)
    skill_context, available_skill_ids = _render_skill_library_context(
        skill_library_handle,
    )
    robot = _reference_robot_slug(robot_hint)
    reference_context = ""
    if attach_references:
        mission_matches = _retrieve_mission_references(goal, robot_hint)
        reference_context = _render_reference_context(mission_matches, robot)
    user_content = _build_user_content(
        goal, reward_contract, kg_context, skill_context=skill_context,
        fitness_metrics_block=_render_fitness_metrics_block(),
        reference_context=reference_context,
    )

    parsed = _parse_with_retry(
        client, system_prompt, user_content, model=model,
    )

    stages = _stages_from_model(parsed.stages)
    if attach_references:
        _attach_stage_references(stages, robot)
    mission = Mission(
        goal=goal,
        stages=stages,
        decomposition_model=model,
        decomposition_rationale=parsed.decomposition_rationale,
    )

    # Structural validation.
    info_keys = set(getattr(reward_contract, "expected_info_keys", None) or [])
    validate_mission(mission, info_keys=info_keys)
    # KG citation verification.
    _validate_kg_seed_papers(stages, set(available_ids), kg_store)
    # Ship 19: skill-library citation verification.
    _validate_skill_ids(stages, set(available_skill_ids))
    # §Ship 38: per-stage steering-metric verification.
    _validate_steering_metrics(stages)

    return mission


# ── Ship 17: per-stage re-decomposition on criterion failure ─────────
# Dataclass shape for the diagnostic payload `redecompose_stage` consumes.
# `mission_run` builds this from the failing stage's sculpt_result + iter
# artifacts and hands it across so this module stays sculpt-free at
# import time (no circular import).
@dataclass
class StageTrainingFeedback:
    final_reward_source: str               # verbatim v<n>.py text
    last_iter_diagnosis: dict              # diagnosis.json content
    last_iter_namespace: dict              # behavior + components dicts
    metric_history: list[float]            # per-iter primary_metric
    last_3_iter_components: list[dict]     # one dict per iter, last 3
    failure_reason: str                    # StageResult.failure_reason
    criterion_error: Optional[str] = None  # if criterion eval crashed


def _render_training_feedback_block(
    feedback: "StageTrainingFeedback",
) -> str:
    """Markdown-fence the feedback into a Claude-readable block. Order
    matters: the reviewer-flagged-most-important signal (final reward
    source) goes FIRST so prompt-cache benefits from common prefixes
    across re-decomposition calls within a single mission."""
    lines = ["# TRAINING_FEEDBACK"]
    lines.append("\n## final_reward_source")
    lines.append("```python")
    src = (feedback.final_reward_source or "").strip()
    # Cap at ~6 KB so a pathologically-large reward module doesn't
    # blow Claude's context. Realistic reward modules are ~1-3 KB.
    if len(src) > 6000:
        src = src[:6000] + "\n# ... (truncated, original was "
        src += f"{len(feedback.final_reward_source)} chars)"
    lines.append(src)
    lines.append("```")

    lines.append("\n## last_iter_diagnosis")
    lines.append(json.dumps(feedback.last_iter_diagnosis, indent=2, default=str))

    lines.append("\n## last_iter_namespace")
    # Strip trajectory arrays; only keep behavior + components scalars.
    namespace = {
        "behavior": feedback.last_iter_namespace.get("behavior", {}),
        "components": feedback.last_iter_namespace.get("components", {}),
        "metric": feedback.last_iter_namespace.get("metric"),
    }
    lines.append(json.dumps(namespace, indent=2, default=str))

    lines.append("\n## metric_history (per iter)")
    lines.append(json.dumps(feedback.metric_history, default=str))

    lines.append("\n## last_3_iter_components")
    lines.append(json.dumps(feedback.last_3_iter_components, indent=2, default=str))

    lines.append(f"\n## failure_reason\n{feedback.failure_reason}")
    if feedback.criterion_error:
        lines.append(f"\n## criterion_error\n{feedback.criterion_error}")
    return "\n".join(lines)


def _build_redecompose_user_content(
    original_stage: Stage,
    reward_contract: Any,
    kg_context: str,
    feedback: "StageTrainingFeedback",
) -> str:
    return (
        "# ORIGINAL_FAILED_STAGE\n"
        f"name: {original_stage.name}\n"
        f"goal_text: {original_stage.goal_text}\n"
        f"success_criterion: {original_stage.success_criterion}\n"
        f"max_iterations: {original_stage.max_iterations}\n"
        f"parent_stage: {original_stage.parent_stage}\n"
        f"reward_seed_prompt: {original_stage.reward_seed_prompt}\n\n"
        f"{_render_contract(reward_contract)}\n"
        f"{kg_context}\n\n"
        f"{_render_training_feedback_block(feedback)}\n\n"
        "Emit the redecomposition JSON now. 2-8 sub-stages whose names "
        f"all start with {original_stage.name}__r1_, where the LAST "
        "sub-stage's success_criterion is byte-identical to the failed "
        "stage's, and the FIRST sub-stage's parent_stage equals the "
        f"failed stage's parent_stage ({original_stage.parent_stage!r})."
    )


def _truncate_for_name_budget(name: str, suffix_len: int) -> str:
    """Stage names cap at 32 chars (mission._STAGE_NAME_RE). Sub-stage
    names take the form `{name}__r1_<i>[_v<n>]`. Truncate `name` so
    `name + suffix_len` fits in 32 chars. Conservative: leave 4 extra
    chars for collision-disambiguator suffixes."""
    max_base = 32 - suffix_len - 4
    if len(name) <= max_base:
        return name
    # Truncate at the last underscore boundary if possible for readability.
    truncated = name[:max_base]
    last_us = truncated.rfind("_")
    if last_us >= max_base - 6:  # only respect boundary if it's "near the end"
        truncated = truncated[:last_us]
    return truncated


_MAX_NAME_LEN = 32  # mirrors mission._STAGE_NAME_RE's char cap
_MAX_COLLISION_ATTEMPTS = 100  # audit-fix safety cap


def _resolve_unique_name(
    proposed: str, existing_names: set[str],
) -> str:
    """Append `_v2`, `_v3`, ... suffixes until `proposed` is unique
    within `existing_names`. Used to disambiguate sub-stage names that
    collide with already-extant mission stages (e.g., a prior
    re-decomposition's leftovers).

    Audit-fix (finding #C): cap the loop at `_MAX_COLLISION_ATTEMPTS`
    and validate the final length against `_MAX_NAME_LEN` so a
    pathological mission with 100+ pre-existing collisions OR a
    too-long base name surfaces a clear error rather than silently
    producing an invalid (>32 char) stage name.
    """
    candidate = proposed
    if candidate not in existing_names:
        if len(candidate) > _MAX_NAME_LEN:
            raise MissionValidationError(
                f"redecomposition: proposed name {candidate!r} "
                f"exceeds {_MAX_NAME_LEN}-char limit even before "
                f"collision resolution; failed stage's base name is "
                f"too long. Truncate the parent stage name."
            )
        return candidate
    for n in range(2, _MAX_COLLISION_ATTEMPTS + 2):
        candidate = f"{proposed}_v{n}"
        if candidate not in existing_names:
            if len(candidate) > _MAX_NAME_LEN:
                raise MissionValidationError(
                    f"redecomposition: collision-resolved name "
                    f"{candidate!r} exceeds {_MAX_NAME_LEN}-char "
                    f"limit. The failed stage's base name leaves "
                    f"insufficient headroom for the `_vN` "
                    f"disambiguator. Truncate the parent stage name."
                )
            return candidate
    raise MissionValidationError(
        f"redecomposition: hit {_MAX_COLLISION_ATTEMPTS} consecutive "
        f"collisions on candidate {proposed!r}; mission state is "
        f"pathological. Manually reduce the number of similarly-"
        f"named stages and retry."
    )


def redecompose_stage(
    mission: Any,                   # Mission — Any to dodge circular import
    failed_stage_idx: int,
    *,
    feedback: "StageTrainingFeedback",
    reward_contract: Any,
    kg_store: Any = None,
    client: Any = None,
    model: str = MODEL_ID,
    prior_attempt_error: Optional[str] = None,
) -> list[Stage]:
    """Ask Claude to split a failed stage into 2-8 simpler sub-stages.

    Returns a list of `Stage` objects ready for the orchestrator to
    splice into `mission.stages` at `failed_stage_idx`. Each returned
    Stage has `redecomposition_attempts=1` set so it can't be re-split
    again — Ship 17 bounds the curriculum tree at one level.

    Validates Claude's response:
      * Sub-stage count ∈ [2, 8] (enforced by `_RedecompositionModel`).
      * `name` starts with `{failed.name}__r1_` (after truncation).
      * `name` is unique within `mission.stages` (collision-resolved).
      * First sub-stage's `parent_stage == failed.parent_stage`.
      * Each subsequent sub-stage's `parent_stage == prev.name` (linear).
      * Last sub-stage's `success_criterion == failed.success_criterion`
        (byte-equal after `.strip()`).

    §D21 Fix 1 (post-mortem D20, docs/internal/REFERENCE_BUILD_LOG.md):
    every sub-stage UNCONDITIONALLY inherits the failed stage's
    `reference_clip_id`/`reference_tier`/`reference_match_confidence` —
    the sub-stages are phases of the SAME motion the failed stage was
    attempting, and the clip is also what stage-metric generation reads.
    Pre-D21, `redecompose_stage` dropped the binding entirely (no
    retrieval pass, unlike `decompose_task`), so a sub-stage with
    `needs_reference_rsi=True` and no clip fell through the scaffold's
    precedence chain to the WRONG task class (procedural jump RSI on a
    get-up mission — the D20 incident).

    Additionally: when the failed stage's on-disk stage dir contains
    `env/eval_reset.json` (§D17 — a non-standing eval start means THE
    START STATE IS THE TASK), every sub-stage's `needs_reference_rsi` is
    FORCED to True, overriding the LLM's per-sub-stage choice — a
    get-up sub-stage that scaffolds without RSI silently reverts to the
    default standing reset, which is exactly the defect this fix closes.
    The stage-dir lookup degrades gracefully (no force, no crash) if
    `mission.stage_dir` raises or the dir doesn't exist. When NO
    eval_reset.json exists, the LLM decides each sub-stage's flag as
    before (see the comment on the `needs_reference_rsi=` line below —
    that rule exists for jump decompositions, where a redecomposition
    typically splits ONE airborne stage into a grounded precursor plus a
    later airborne sub-stage, and a blanket force would deny the
    grounded precursor's correctly-False flag).

    Raises `MissionValidationError` on any violation. Caller halts the
    mission on failure (do NOT retry Claude — same envelope as
    `decompose_task`).
    """
    failed_stage: Stage = mission.stages[failed_stage_idx]

    # §D21 Fix 1: does the failed stage have a stage-FIXED eval reset
    # (§D17 get-up archetype)? If so, the start state IS the task, and
    # every sub-stage MUST scaffold with reference RSI regardless of
    # what the redecompose LLM chose per sub-stage. Best-effort: any
    # failure resolving the stage dir (mission_dir unset, dir missing,
    # ...) just leaves this False — the LLM's per-sub-stage choice wins,
    # same as today.
    force_reference_rsi = False
    try:
        _failed_stage_dir = mission.stage_dir(failed_stage.name)
        force_reference_rsi = (
            _failed_stage_dir / "env" / "eval_reset.json"
        ).is_file()
    except Exception:  # noqa: BLE001 — degrade gracefully, never crash
        force_reference_rsi = False

    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    system_prompt = load_prompt("redecompose_stage")
    # KG context tied to the ORIGINAL goal, not the diagnosis — same
    # papers should be relevant since the high-level task is unchanged.
    kg_context, available_arxiv_ids = _render_kg_context(
        failed_stage.goal_text, kg_store,
    )
    user_content = _build_redecompose_user_content(
        failed_stage, reward_contract, kg_context, feedback,
    )
    if prior_attempt_error:
        # Ship 22r: a previous redecomposition draft was rejected by the
        # mission validator (e.g. a sub-stage criterion referenced a key the
        # rollout doesn't persist). Feed the exact error back so Claude fixes
        # the offending sub-stage instead of the caller halting the mission.
        user_content += (
            "\n\n## PREVIOUS REDECOMPOSITION ATTEMPT WAS REJECTED\n"
            f"{prior_attempt_error}\n\n"
            "Fix the offending sub-stage. EVERY success_criterion may ONLY "
            "reference keys that exist: `behavior[<BEHAVIOR_KEY>]`, "
            "`components[<name your reward_seed_prompt defines>]`, "
            "`trajectory[<PERSISTED_TRAJECTORY_KEY>]` / `info[...]`, or `metric`. "
            "Do NOT use base_height / fallen or any non-persisted key — derive "
            "base height from `trajectory['root_link_pos_w'][..., 2]` and an "
            "upright/fallen proxy from `trajectory['projected_gravity_b'][..., 2]`. "
            "Use `<dict>.get(key, default)` for any component you are unsure the "
            "reward emits."
        )
    parsed = _parse_with_retry(
        client, system_prompt, user_content,
        output_format=_RedecompositionModel, model=model,
    )

    # Build sub-stages with the right parent chain + name discipline.
    sub_stages: list[Stage] = []
    existing_names = {s.name for s in mission.stages}
    # Drop the failed name from collision set — it'll be replaced.
    existing_names.discard(failed_stage.name)

    name_prefix_base = _truncate_for_name_budget(
        failed_stage.name, suffix_len=len("__r1_") + 2,  # __r1_<digits>
    )

    parent_chain = failed_stage.parent_stage
    expected_criterion = failed_stage.success_criterion.strip()

    for i, model_stage in enumerate(parsed.stages):
        # Force the canonical naming convention regardless of what
        # Claude emitted — guards against prompt drift.
        canonical = f"{name_prefix_base}__r1_{i}"
        canonical = _resolve_unique_name(canonical, existing_names)
        existing_names.add(canonical)

        # Validate the linear-parent-chain rule.
        if i == 0:
            if model_stage.parent_stage != failed_stage.parent_stage:
                raise MissionValidationError(
                    f"redecomposition: sub-stage[0].parent_stage="
                    f"{model_stage.parent_stage!r} must equal failed "
                    f"stage's parent_stage={failed_stage.parent_stage!r}"
                )
        else:
            prev_canonical = sub_stages[-1].name
            if model_stage.parent_stage not in (
                prev_canonical, parsed.stages[i - 1].name,
            ):
                # Allow either: Claude's emitted previous-sub-name OR our
                # canonical version of it (Claude won't have known the
                # disambiguator we may have applied).
                raise MissionValidationError(
                    f"redecomposition: sub-stage[{i}].parent_stage="
                    f"{model_stage.parent_stage!r} must reference the "
                    f"previous sub-stage in the chain "
                    f"({prev_canonical!r} or "
                    f"{parsed.stages[i - 1].name!r})"
                )

        # Last sub-stage must have byte-equal success_criterion.
        is_last = (i == len(parsed.stages) - 1)
        if is_last and model_stage.success_criterion.strip() != expected_criterion:
            raise MissionValidationError(
                f"redecomposition: last sub-stage's success_criterion "
                f"must be byte-identical (after strip) to the failed "
                f"stage's. Failed: {expected_criterion!r}; "
                f"got: {model_stage.success_criterion.strip()!r}"
            )

        sub_stages.append(Stage(
            name=canonical,
            goal_text=model_stage.goal_text,
            success_criterion=model_stage.success_criterion,
            max_iterations=model_stage.max_iterations,
            parent_stage=(
                failed_stage.parent_stage if i == 0
                else sub_stages[-1].name
            ),
            reward_seed_prompt=model_stage.reward_seed_prompt,
            kg_seed_papers=list(model_stage.kg_seed_papers or []),
            # §Ship 38: sub-stages inherit the failed stage's objective so a
            # re-decomposed phase keeps steering by the same metric.
            steering_metric=failed_stage.steering_metric,
            # §JUMP_SCAFFOLD refinement (2026-07-06): per-sub-stage RSI —
            # the redecompose LLM decides each sub-stage's flag (rule 10),
            # NOT force-inherit from the failed parent. Re-decomposition
            # usually splits ONE airborne stage into a GROUNDED precursor
            # (RSI false — else needless resets are wasted on states it
            # doesn't need) plus a later airborne sub-stage (RSI true); a
            # blanket `or failed_stage.needs_reference_rsi` denied the
            # grounded precursors that gain.
            #
            # §D21 Fix 1: EXCEPT when the failed stage had a stage-fixed
            # eval reset (§D17 — the lying/low start IS the task, not
            # curriculum). In that case every sub-stage MUST scaffold
            # with reference RSI or it silently reverts to a standing
            # default reset (the D20 incident) — override the LLM here.
            needs_reference_rsi=(
                True if force_reference_rsi else bool(
                    getattr(model_stage, "needs_reference_rsi", False))
            ),
            # §start_pose: PER-SUB-STAGE, model-chosen — unlike the
            # reference-clip fields below, start_pose is NOT force-
            # inherited from the failed parent. Sub-stages of a
            # redecomposed get-up stage typically progress through
            # DIFFERENT physical configurations (e.g. r1_0 supine ->
            # r1_2 crouched), so the model picks each one; the D21
            # force-rule above (and `validate_mission`'s force-rule on
            # start_pose != "standing") still applies per sub-stage
            # regardless of what the model set here.
            start_pose=getattr(model_stage, "start_pose", None),
            # §D21 Fix 1: unconditional inheritance — sub-stages are
            # phases of the SAME motion the failed stage was attempting,
            # and stage-metric generation reads this same clip.
            reference_clip_id=failed_stage.reference_clip_id,
            reference_tier=failed_stage.reference_tier,
            reference_match_confidence=failed_stage.reference_match_confidence,
            redecomposition_attempts=1,  # bound at one level
        ))

    # KG citation verification (same as decompose_task).
    _validate_kg_seed_papers(sub_stages, set(available_arxiv_ids), kg_store)
    return sub_stages


# ── §Ship 25a (H1): criterion ↔ reward-component reconciliation ──────
class _ReconcileModel(BaseModel):
    """Claude's response schema for a criterion rewrite."""

    rationale: str
    success_criterion: str


def _gate_reconciled_criterion(
    stage: Stage,
    new_criterion: str,
    available_components: list[str],
) -> None:
    """All validation gates a reconciled criterion must pass. Raises
    MissionValidationError with an actionable message (fed back to
    Claude on the retry attempt)."""
    from sculptor.mission_runtime import extract_components_keys

    if not new_criterion:
        raise MissionValidationError(
            "reconcile_criterion: Claude returned an empty criterion"
        )
    if new_criterion == stage.success_criterion.strip():
        raise MissionValidationError(
            "reconcile_criterion: rewrite is identical to the original — "
            "the missing-key references were not fixed"
        )

    # Same static gate the decomposer's criteria pass through, run on a
    # COPY so a rejected rewrite can't leave the stage half-mutated.
    from dataclasses import replace as _dc_replace

    from sculptor.mission import _validate_success_criterion

    candidate = _dc_replace(stage, success_criterion=new_criterion)
    _validate_success_criterion(candidate, set())

    # ALSO run the runtime's unsafe-AST gate statically (the decompose-
    # time validator checks subscript keys + torch idioms but defers
    # node-level safety — lambdas etc. — to eval time; a reconciled
    # criterion must not fail LATER for a reason knowable NOW).
    import ast as _ast

    from sculptor.mission_runtime import (
        BARE_IDENTIFIERS,
        CriterionEvalError,
        _validate_criterion_ast,
    )

    try:
        _validate_criterion_ast(
            _ast.parse(new_criterion, mode="eval"),
            namespace_keys=set(BARE_IDENTIFIERS) | {
                "behavior", "components", "trajectory", "info",
            },
        )
    except (CriterionEvalError, SyntaxError) as e:
        raise MissionValidationError(
            f"reconcile_criterion: rewrite failed the runtime safety "
            f"gate: {e}"
        ) from e

    still_missing = (
        extract_components_keys(new_criterion) - set(available_components)
    )
    if still_missing:
        raise MissionValidationError(
            f"reconcile_criterion: rewrite still hard-references absent "
            f"component keys {sorted(still_missing)} "
            f"(available: {sorted(available_components)})"
        )


def reconcile_criterion(
    stage: Stage,
    *,
    missing_keys: list[str],
    available_components: list[str],
    client: Any = None,
    model: str = MODEL_ID,
) -> tuple[str, str]:
    """Rewrite a stage's success_criterion whose hard
    `components['<name>']` references don't exist in the materialized
    reward — caught at iter 0 by the probe instead of burning the whole
    stage budget before `criterion_not_met` fires at eval time.

    Returns `(new_criterion, rationale)`. One validation-feedback retry:
    a rewrite that fails the gates gets a second attempt carrying the
    exact gate error (mirrors redecompose's `prior_attempt_error`).
    Raises MissionValidationError when both attempts fail.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    from sculptor.mission_runtime import (
        BEHAVIOR_KEYS,
        PERSISTED_TRAJECTORY_KEYS,
    )

    system_prompt = load_prompt("reconcile_criterion")
    payload: dict[str, Any] = {
        "stage_name": stage.name,
        "stage_goal": stage.goal_text,
        "current_criterion": stage.success_criterion,
        "missing_component_keys": sorted(missing_keys),
        "available_component_keys": sorted(available_components),
        "available_behavior_keys": sorted(BEHAVIOR_KEYS),
        "available_trajectory_keys": sorted(PERSISTED_TRAJECTORY_KEYS),
        "reward_seed_prompt": stage.reward_seed_prompt,
    }

    last_error: Optional[str] = None
    for _attempt in range(2):
        if last_error is not None:
            payload["prior_attempt_error"] = last_error
        parsed = _parse_with_retry(
            client, system_prompt, json.dumps(payload, indent=2),
            output_format=_ReconcileModel, model=model,
        )
        new_criterion = parsed.success_criterion.strip()
        try:
            _gate_reconciled_criterion(
                stage, new_criterion, available_components,
            )
            return new_criterion, parsed.rationale
        except MissionValidationError as e:
            last_error = str(e)
    raise MissionValidationError(
        f"reconcile_criterion: rewrite failed validation twice; "
        f"last error: {last_error}"
    )
