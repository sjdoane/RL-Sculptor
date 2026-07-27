"""sculptor/kg/retrieval_log.py — §Agentic-data upgrade 3: retrieval
trajectory logging.

Mirrors `sculptor.llm.log_llm_call`'s provenance-archive pattern: append
one JSON line per retrieval decision to `<out_dir>/kg_retrievals.jsonl`.
The point is a durable, greppable trail of WHAT the KG returned for WHICH
query at WHICH decision point (diagnose / decompose / ...) — useful for
after-the-fact audits of why a diagnosis or decomposition cited (or
missed) a particular technique/case, without instrumenting every call
site with bespoke logging.

Design mirrors `llm.py`'s provenance sink exactly:
  * no sink configured (out_dir is None AND `llm_log_dir()` is None) ->
    silent no-op;
  * any failure (disk full, unwritable dir, non-serializable match
    object, ...) is swallowed — retrieval logging is advisory, it must
    never take down a diagnose/decompose call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

LOG_BASENAME = "kg_retrievals.jsonl"


def _extract_match(match: Any) -> dict[str, Any]:
    """Defensively pull (node_id, score) out of a TechniqueMatch/CaseMatch
    (or anything duck-typed close to one). Never raises — an unexpected
    shape just yields a best-effort partial record."""
    node_id = None
    score = None
    try:
        technique = getattr(match, "technique", None)
        if technique is not None:
            node_id = getattr(technique, "id", None)
        if node_id is None:
            case = getattr(match, "case", None)
            if case is not None:
                node_id = getattr(case, "id", None)
        if node_id is None:
            # Fall back to anything that looks like a node itself.
            node_id = getattr(match, "id", None)
    except Exception:  # noqa: BLE001 — defensive extraction, never raises
        node_id = None
    try:
        score = getattr(match, "relevance_score", None)
    except Exception:  # noqa: BLE001
        score = None
    return {"node_id": node_id, "score": score}


def log_retrieval(
    decision: str,
    query: str,
    matches: Iterable[Any] | None,
    out_dir: Path | str | None = None,
) -> None:
    """Append one JSON line recording a retrieval decision. NEVER raises.

    `out_dir`: directory to write `kg_retrievals.jsonl` into. When None,
    falls back to `sculptor.llm.llm_log_dir()` (the current iteration /
    mission provenance sink); when that is also None, this is a no-op —
    exactly the `log_llm_call` contract.

    `matches`: a list of TechniqueMatch / CaseMatch (or any object
    exposing `.technique.id` / `.case.id` and `.relevance_score`) —
    extracted defensively via `getattr` so an unexpected shape degrades to
    partial data rather than raising.
    """
    try:
        target_dir: Path | None
        if out_dir is not None:
            target_dir = Path(out_dir)
        else:
            from sculptor.llm import llm_log_dir

            target_dir = llm_log_dir()
        if target_dir is None:
            return

        node_ids: list[Any] = []
        scores: list[Any] = []
        for m in matches or []:
            extracted = _extract_match(m)
            node_ids.append(extracted["node_id"])
            scores.append(extracted["score"])

        rec = {
            "ts": time.time(),
            "decision": decision,
            "query": query,
            "node_ids": node_ids,
            "scores": scores,
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        with (target_dir / LOG_BASENAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — advisory by contract (docstring)
        pass
