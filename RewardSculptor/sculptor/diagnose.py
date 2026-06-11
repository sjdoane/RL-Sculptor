"""sculptor/diagnose.py — two-stage LLM diagnoser grounded in the KG.

Stage 1 ("preliminary"):
    behavior_goal + current REWARD_SPEC + metrics.json + behavior.json +
    4 keyframe PNGs + reward_contract + the adapter's behavior-metric vocab
    → failure_modes (from a fixed enum), evidence, confidence.

Stage 2 ("grounded"):
    preliminary result + KG context (top-6 union of
    query_techniques(failure_modes, domain_filter=config.kg.environment_tag)
    and query_semantic(behavior_goal)) + original inputs + reward_contract
    → proposed_edits. Every edit must cite paper_refs (arxiv_ids) when
    literature-grounded, or mark itself `novel.` and leave paper_refs empty.

Both calls use claude-opus-4-7 with adaptive thinking, strict JSON via
`messages.parse`, and one retry on parse/validation failure.

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

from pydantic import BaseModel, Field

from sculptor.adapters.base import load_adapter
from sculptor.kg.query import TechniqueMatch, query_semantic, query_techniques
from sculptor.kg.store import SculptorKG


MODEL_ID = "claude-opus-4-7"
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


class _GroundedModel(BaseModel):
    proposed_edits: list[_ProposedEditModel] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ── Public dataclasses ────────────────────────────────────────────────────
@dataclass
class ProposedEdit:
    target_term: str
    operation: str
    rationale: str
    suggested_value: str | None
    paper_refs: list[str] = field(default_factory=list)
    requires_env_extension: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Diagnosis:
    failure_modes: list[str]
    evidence: str
    proposed_edits: list[ProposedEdit] = field(default_factory=list)
    literature_context: list[TechniqueMatch] = field(default_factory=list)
    confidence: float = 0.0
    iter_dir: str | None = None
    behavior_goal: str | None = None

    def to_dict(self) -> dict:
        return {
            "failure_modes": list(self.failure_modes),
            "evidence": self.evidence,
            "proposed_edits": [e.to_dict() for e in self.proposed_edits],
            "literature_context": [
                {
                    "technique": m.technique.name,
                    "description": m.description,
                    "paper_citation": m.paper_citation,
                    "evidence": m.evidence,
                    "relevance_score": m.relevance_score,
                    "matched_on": m.matched_on,
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
    if not matches:
        return "(no matches from the knowledge graph)"
    lines = ["# LITERATURE CONTEXT (top KG matches)", ""]
    for m in matches:
        lines.append(f"## {m.technique.name}")
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
) -> list[dict]:
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
    header = (
        f"# BEHAVIOR GOAL\n{behavior_goal}\n\n"
        f"# REWARD_SPEC\n{json.dumps(reward_spec, indent=2, sort_keys=True, default=str)}\n\n"
        f"# metrics.json\n{json.dumps(metrics, indent=2, sort_keys=True, default=str)}\n\n"
        f"{feedback_block}"
        f"{realism_block}"
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
    return (
        f"# BEHAVIOR GOAL\n{behavior_goal}\n\n"
        f"# REWARD_SPEC\n{json.dumps(reward_spec, indent=2, sort_keys=True, default=str)}\n\n"
        f"# metrics.json\n{json.dumps(metrics, indent=2, sort_keys=True, default=str)}\n\n"
        f"{feedback_block}"
        f"{realism_block}"
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
def _parse_with_retry(client, *, model_cls, system_prompt, user_content):
    """messages.parse with one retry on parse / validation failure."""
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
        return resp.parsed_output
    except Exception as first_err:
        # 4.7 forbids assistant prefill — retry via a second user turn.
        retry_messages = [
            {"role": "user", "content": user_content},
            {"role": "user",
             "content": f"{RETRY_REMINDER} Previous error: {first_err!s}"},
        ]
        resp = client.messages.parse(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=retry_messages,
            output_format=model_cls,
        )
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
) -> Diagnosis:
    """Two-stage literature-grounded diagnosis of a training iteration.

    `skip_kg=True` bypasses both KG queries and passes an empty
    literature_context to the grounded call — used for the `--no-kg`
    ablation mode in `sculpt run`.
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

    # 2. Load the adapter to get the reward_contract.
    adapter = load_adapter(Path(config)) if not isinstance(config, dict) else None
    if adapter is None:
        raise ValueError(
            "diagnose() requires a config path that resolves to a SculptorAdapter; "
            "passing a dict isn't enough because we need reward_contract() from the "
            "instantiated adapter.")
    contract = adapter.reward_contract()
    contract_text = _render_reward_contract(contract)

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
    )
    preliminary: _PreliminaryModel = _parse_with_retry(
        client, model_cls=_PreliminaryModel,
        system_prompt=_PRELIM_SYSTEM, user_content=prelim_user,
    )

    # 5. KG retrieval — union of tag-based and semantic matches, deduped.
    #    Skipped when `skip_kg=True` (used for `--no-kg` ablations).
    owns_store = (store is None) and not skip_kg
    store = store or (None if skip_kg else SculptorKG())
    try:
        if skip_kg:
            kg_matches: list[TechniqueMatch] = []
        else:
            fm_keywords = [fm for fm in preliminary.failure_modes if fm != "none"]
            tag_matches = query_techniques(
                fm_keywords, domain_filter=env_tag, top_k=KG_TOP_K, store=store,
            ) if fm_keywords else []
            try:
                # §Ship 31: floored — an unfloored semantic slice feeds
                # tangential techniques into the grounded prompt and
                # Claude dutifully cites them (Issue G).
                from sculptor.kg.query import DEFAULT_MIN_PROMPT_SIMILARITY

                sem_matches = query_semantic(
                    behavior_goal, top_k=KG_TOP_K, store=store,
                    min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY)
            except Exception as e:  # noqa: BLE001
                print(f"[diagnose] semantic query failed ({e}) — tag-only context.",
                      file=sys.stderr, flush=True)
                sem_matches = []

            seen: set[str] = set()
            kg_matches = []
            for m in tag_matches + sem_matches:
                if m.technique.id in seen:
                    continue
                seen.add(m.technique.id)
                kg_matches.append(m)
                if len(kg_matches) >= KG_TOP_K:
                    break

        # 6. Stage 2 — grounded edits.
        grounded_user = _build_grounded_user_content(
            behavior_goal=behavior_goal,
            reward_spec=reward_spec,
            metrics=metrics,
            behavior=behavior,
            contract_text=contract_text,
            preliminary=preliminary,
            kg_context=_render_kg_context(kg_matches),
            training_feedback=training_feedback,
            realism_audit=realism_audit,
        )
        grounded: _GroundedModel = _parse_with_retry(
            client, model_cls=_GroundedModel,
            system_prompt=_GROUNDED_SYSTEM, user_content=grounded_user,
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
            )
            for e in grounded.proposed_edits
        ],
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
