"""Optional LLM world author (hybrid: LLM proposes, local validators gate).

The offline author (`sculptor.world.author`) routes a prompt through four keyword
templates — so it cannot ground "a ball and a soccer goal", "four boxes then a
ramp", or anything else outside that set. This module implements the
`AuthorModel` protocol with a language model, so the FULL WorldSpec-v2 vocabulary
(spheres, boxes, cylinders, capsules, goal frames, ~14 terrain generators, zones,
train.variations) becomes reachable from natural language.

It is deliberately NOT trusted: `WorldAuthor.author` runs the model's output
through the exact same local, deterministic gates as the offline path —
`validate_world_spec` / `validate_task_spec`, the closed output-key set, the
robot-capability lock, and the required-capability check — and the compiler admits
it only if every hash/capability/materialization check passes. So a hallucinated
or unsafe spec is rejected, never admitted. `hybrid_author_environment` adds the
belt-and-braces fallback: if the model errors OR its spec fails the gates, it
falls back to the deterministic offline author.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Mapping, Optional

from sculptor.llm import log_llm_call, model_for, response_text_blocks
from sculptor.prompts import load_prompt
from sculptor.world.author import (
    AuthoringDraft,
    AuthoringError,
    _jsonable,
    author_environment,
)

_MAX_TOKENS = 16000
_ALLOWED_KEYS = ("world_spec", "task_spec", "parameter_provenance")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the authoring JSON from a model reply — a fenced ```json block when
    present, else the outermost {...} object. Raises AuthoringError (never a bare
    JSONDecodeError) so the caller's gate/fallback handles it uniformly."""
    blocks = _JSON_FENCE_RE.findall(text or "")
    candidate = blocks[0] if blocks else None
    if candidate is None:
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        candidate = text[start:end + 1] if 0 <= start < end else None
    if candidate is None:
        raise AuthoringError("author model returned no JSON object")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise AuthoringError(f"author model returned invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise AuthoringError("author model JSON must be an object")
    return data


class LLMWorldAuthor:
    """`AuthorModel` backed by a language model.

    `client` is an Anthropic-style client (injected in tests / by the backend);
    when omitted a default `anthropic.Anthropic` is constructed lazily so import
    stays cheap. `model_id` overrides the `world_author` role model.
    """

    def __init__(self, client: Any = None, *, model_id: Optional[str] = None) -> None:
        self._client = client
        self._model_id = model_id or model_for("world_author")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        self._client = anthropic.Anthropic(max_retries=2, timeout=240.0)
        return self._client

    def generate_authoring(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        system = load_prompt("gen_world_author")
        user = json.dumps(_jsonable(request), indent=2, default=str)
        resp = self._get_client().messages.create(
            model=self._model_id, max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system, messages=[{"role": "user", "content": user}])
        log_llm_call(
            "world_author", self._model_id, system=system, user=user,
            response_text=response_text_blocks(resp),
            usage=getattr(resp, "usage", None))
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise AuthoringError(
                f"world authoring truncated at max_tokens={_MAX_TOKENS} — "
                "the spec is incomplete")
        data = _extract_json(response_text_blocks(resp))
        # Return only the closed key set; WorldAuthor re-checks this too, but
        # trimming here keeps a chatty model from tripping the unknown-key gate.
        return {k: data[k] for k in _ALLOWED_KEYS if k in data}


def hybrid_author_environment(
    prompt: str, *, client: Any = None, model_id: Optional[str] = None,
    **kwargs: Any,
) -> AuthoringDraft:
    """Author with the LLM first; fall back to the deterministic offline author
    if the model errors or its spec fails the local gates.

    Every path returns a draft that PASSED `validate_world_spec` /
    `validate_task_spec` — the LLM never bypasses a gate. `client=None` and no
    configured model still work: the LLM attempt short-circuits to offline.
    """
    model = LLMWorldAuthor(client=client, model_id=model_id)
    try:
        return author_environment(prompt, model=model, **kwargs)
    except AuthoringError as exc:
        # The model errored or produced a spec the local validators rejected —
        # never admit it; fall back to the deterministic templates.
        print(f"[world-author] LLM author rejected ({exc}); "
              "falling back to offline templates", file=sys.stderr, flush=True)
        return author_environment(prompt, **kwargs)
