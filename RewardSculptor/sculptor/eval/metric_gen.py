"""Generate + review an auto objective metric from a behavior goal
(§Ship 35). Mirrors the eval/eureka.py LLM pattern (sample → strip fences)
and the diagnose.py structured-review pattern (messages.parse).

Pipeline: GENERATE (LLM, physical-quantity metric) → VALIDATE (the
must-have gates in metric_validate) → on failure, REGENERATE with the
gate failures fed back (bounded attempts) → independent REVIEW (a fresh
LLM context that never saw the reward) → record a verdict + the
archetype scores to `meta.json`. A metric is `accepted` only if it both
passes validation AND the reviewer approves. Acceptance does NOT grant
steer-rights — that requires calibration (metric_calibration); accepted
metrics run observe-only until calibrated.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from sculptor.eval.metric_validate import validate_generated_metric
from sculptor.prompts import load_prompt

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 8000
_FENCE_RE = re.compile(r"```(?:[A-Za-z]+)?\s*\n(.*?)```", re.DOTALL)


class MetricReview(BaseModel):
    """Independent reviewer's structured verdict."""

    approved: bool
    concerns: list[str] = []
    summary: str = ""


def _strip_code(text: str) -> str:
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        code = [b for b in blocks if "def compute_spec" in b]
        return (code[0] if code else max(blocks, key=len)).strip() + "\n"
    return (text or "").strip() + "\n"


def _sample_source(client: Any, system_prompt: str, user_content: str,
                   *, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    chunks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return _strip_code("\n".join(chunks))


def _review_metric(
    client: Any, model: str, goal: str, source: str,
    archetype_scores: dict,
) -> dict[str, Any]:
    system_prompt = load_prompt("review_objective_metric")
    user = json.dumps({
        "behavior_goal": goal,
        "metric_source": source,
        "archetype_scores": archetype_scores,
    }, indent=2, default=str)
    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user}],
            output_format=MetricReview,
        )
        r = resp.parsed_output
        return {"approved": bool(r.approved),
                "concerns": list(r.concerns), "summary": r.summary}
    except Exception as e:  # noqa: BLE001 — a failed review is a non-approval
        return {"approved": False,
                "concerns": [f"review call failed: {type(e).__name__}: {e}"],
                "summary": ""}


def generate_objective_metric(
    behavior_goal: str,
    out_dir: Path | str,
    *,
    robot_hint: Optional[str] = None,
    client: Any = None,
    model: str = MODEL_ID,
    max_attempts: int = 3,
    review: bool = True,
) -> dict[str, Any]:
    """Generate, validate, (regenerate on failure,) and review an objective
    metric for `behavior_goal`. Writes `metric.py` + `meta.json` to
    `out_dir` and returns the full record. NEVER raises on a bad
    candidate — a rejected metric is recorded with `accepted=False`."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=2, timeout=240.0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / "metric.py"
    system_prompt = load_prompt("gen_objective_metric")
    base_user = json.dumps(
        {"behavior_goal": behavior_goal, "robot_hint": robot_hint},
        indent=2, default=str,
    )

    attempts: list[dict[str, Any]] = []
    source = ""
    validation: Optional[dict[str, Any]] = None
    for attempt in range(max(1, max_attempts)):
        user = base_user
        if attempt > 0 and validation is not None:
            user = (
                base_user
                + "\n\nThe previous attempt FAILED these validation gates:\n"
                + json.dumps(validation["reasons"], indent=2)
                + "\nFix ALL of them. Output ONLY the corrected module."
            )
        try:
            source = _sample_source(client, system_prompt, user, model=model)
        except Exception as e:  # noqa: BLE001 — API failure = failed attempt
            attempts.append({"attempt": attempt,
                             "api_error": f"{type(e).__name__}: {e}"})
            continue
        metric_path.write_text(source, encoding="utf-8")
        validation = validate_generated_metric(source, metric_path)
        attempts.append({"attempt": attempt, "ok": validation["ok"],
                         "reasons": validation["reasons"]})
        if validation["ok"]:
            break

    passed = bool(validation and validation["ok"])
    review_out: Optional[dict[str, Any]] = None
    if passed and review:
        review_out = _review_metric(
            client, model, behavior_goal, source,
            (validation or {}).get("archetype_scores", {}),
        )

    accepted = bool(passed and (review_out is None or review_out.get("approved")))
    record = {
        "accepted": accepted,
        "validation_passed": passed,
        "metric_path": str(metric_path),
        "source": source,
        "validation": validation,
        "review": review_out,
        "attempts": attempts,
        "behavior_goal": behavior_goal,
        "robot_hint": robot_hint,
        "model": model,
        # steer-rights are earned later via calibration; observe-only until then.
        "calibrated": False,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record
