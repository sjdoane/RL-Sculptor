"""Central LLM model registry + call-provenance archive.

§RESEARCH_GAP_ANALYSIS §3.9/§7.8: the mutation operator is a
closed-source LLM, so replayability of every decision (model id, prompt
hash, full request/response) is the only reproducibility story a paper
can offer — and it cannot be retrofitted onto past runs. This module is
the single place that answers two questions:

  1. *Which model serves which role?* — `model_for(role)`. Every module
     that talks to the Anthropic API initializes its MODEL_ID from the
     role table below instead of hardcoding, so a per-role upgrade (or a
     cheap-model override for smoke tests) is one env var, not a source
     edit. Precedence: `RS_MODEL_<ROLE>` > `RS_MODEL_ALL` > role default.

  2. *What did each call actually say?* — `log_llm_call(...)`. When a
     sink is configured (`set_llm_log_dir`), every call appends one JSON
     line with timestamps, model id, sha256 hashes, and the full
     system/user/response text to `<dir>/llm_calls.jsonl`. Unconfigured
     → no-op; a failing sink NEVER raises into the pipeline (provenance
     is advisory, the loop is not).

Role assignments (2026-07-09 pass — fable-5 across the board, one
principled exception):

  * fable-5 on EVERY in-system role: decomposition, diagnosis, reward
    writing, metric authoring/review, env-spec generation, KG extraction/
    research, MJCF editing, reference reranking, eureka baseline (the
    baseline arm must be model-matched to the treatment arm or an E4-v2
    comparison confounds search topology with model strength).
  * calibration (competence-ladder + gaming-archetype author) is the ONE
    exception — DELIBERATELY a different model (opus-4-8) from the metric
    author: §RESEARCH_GAP_ANALYSIS §3.5 — if author and ladder share one
    model's blind spot, L2 agreement passes without catching it. Keep
    these model-disjoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

#: role → default model id. Roles are the stable names run_context.py
#: records; renaming one breaks provenance comparability across runs.
ROLE_DEFAULTS: dict[str, str] = {
    "decompose": "claude-fable-5",
    "diagnose": "claude-fable-5",
    "edit": "claude-fable-5",
    "metric_gen": "claude-fable-5",
    "metric_review": "claude-fable-5",
    # calibration is the ONE deliberate exception to the all-fable-5
    # policy (2026-07-09): §RESEARCH_GAP_ANALYSIS §3.5 requires the
    # ladder/archetype author to be model-DISJOINT from metric_gen so a
    # shared blind spot can't pass L2 agreement — metric_gen is fable-5,
    # so calibration must not be.
    "calibration": "claude-opus-4-8",
    "eureka_baseline": "claude-fable-5",
    "env_gen": "claude-fable-5",
    "kg_extract": "claude-fable-5",
    "kg_research": "claude-fable-5",
    "mjcf_editor": "claude-fable-5",
    # reference-clip retrieval reranking (§R1_BUILD_SPEC decision 7):
    # OPTIONAL role — retrieve.py's deterministic layer is always-on and
    # never blocked by this call.
    "reference_rerank": "claude-fable-5",
    # §D24 F1 (docs/internal/REFERENCE_BUILD_LOG.md D23/D24): proposes
    # the goal-aligned SUB-SPAN of a reference clip (`sculptor.refs.
    # spans.select_reference_span`). Follows the D1 all-fable-5 default
    # — this role is not the metric author/ladder pair D1 keeps
    # model-disjoint, so no exception applies.
    "span_select": "claude-fable-5",
    # §D24 F2 (docs/internal/REFERENCE_BUILD_LOG.md D23/D24): re-grounds a
    # stage's blind-authored `success_criterion` in the CROPPED reference
    # clip's real kinematic signature, right after a span attaches
    # (`sculptor.decompose.ground_stage_criterion`). Same D1 all-fable-5
    # default — not the metric author/ladder pair kept model-disjoint.
    "criterion_ground": "claude-fable-5",
    # §D28 F-SYNTH (docs/internal/REFERENCE_BUILD_LOG.md D28): sketches a
    # LAST-RESORT synthetic exemplar (`sculptor.refs.synth.
    # synthesize_reference_clip`) when no real reference clip matches a
    # stage's goal closely enough to certify against. Same D1 all-fable-5
    # default — not the metric author/ladder pair kept model-disjoint.
    "exemplar_synth": "claude-fable-5",
}

#: §LAW 9 review panel: author-DISJOINT by construction — none of these
#: may equal ROLE_DEFAULTS["metric_gen"], so the panel never contains
#: the metric's own author model (blind-spot diversification).
REVIEW_PANEL_DEFAULTS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)


def model_for(role: str) -> str:
    """Resolve the model id for `role`. Precedence:
    `RS_MODEL_<ROLE>` (upper-cased role) > `RS_MODEL_ALL` > table default.
    Unknown roles fall back to RS_MODEL_ALL or the edit default — new
    call sites should be added to ROLE_DEFAULTS, but an unknown role
    must not crash a run."""
    per_role = os.environ.get(f"RS_MODEL_{role.upper()}")
    if per_role:
        return per_role
    every = os.environ.get("RS_MODEL_ALL")
    if every:
        return every
    return ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["edit"])


def review_panel_models() -> list[str]:
    """Models for the metric review panel. `RS_MODEL_REVIEW_PANEL`
    (comma-separated) overrides. The author-exclusion filter itself
    lives in metric_gen (it also excludes at runtime whatever model
    authored the candidate under review)."""
    raw = os.environ.get("RS_MODEL_REVIEW_PANEL")
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    return list(REVIEW_PANEL_DEFAULTS)


# ── Provenance archive ──────────────────────────────────────────────────
_LOG_DIR: ContextVar[Optional[str]] = ContextVar("sculptor_llm_log_dir",
                                                 default=None)
LOG_BASENAME = "llm_calls.jsonl"


def set_llm_log_dir(path: Path | str | None) -> None:
    """Point the provenance sink at a directory (typically the current
    iteration dir, mission dir, or metric out_dir). None disables."""
    _LOG_DIR.set(str(path) if path is not None else None)


def llm_log_dir() -> Optional[Path]:
    raw = _LOG_DIR.get()
    return Path(raw) if raw else None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def log_llm_call(
    role: str,
    model: str,
    *,
    system: str = "",
    user: str = "",
    response_text: str = "",
    usage: Any = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Append one provenance record. No sink configured → no-op. Any
    failure (disk full, non-serializable usage, …) is swallowed —
    provenance must never take down a training run."""
    out_dir = llm_log_dir()
    if out_dir is None:
        return
    try:
        rec: dict[str, Any] = {
            "ts": time.time(),
            "role": role,
            "model": model,
            "system_sha256": _sha256(system),
            "user_sha256": _sha256(user),
            "response_sha256": _sha256(response_text),
            "system": system,
            "user": user,
            "response": response_text,
        }
        if usage is not None:
            try:
                rec["usage"] = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                }
            except Exception:  # noqa: BLE001 — usage shape is SDK-owned
                pass
        if meta:
            rec["meta"] = meta
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / LOG_BASENAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — advisory by contract (docstring)
        pass


def response_text_blocks(resp: Any) -> str:
    """Join the text blocks of an Anthropic response (skipping thinking
    blocks) — shared helper so provenance records match what call sites
    actually parsed."""
    chunks: list[str] = []
    for block in getattr(resp, "content", ()) or ():
        if getattr(block, "type", None) == "text":
            chunks.append(getattr(block, "text", ""))
    return "".join(chunks)
