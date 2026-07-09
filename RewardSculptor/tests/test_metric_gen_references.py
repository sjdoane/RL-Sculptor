"""tests/test_metric_gen_references.py — §REFERENCE_TRAJECTORY_PLAN §7.

Kinematic-signature grounding of `sculptor.eval.metric_gen
.generate_objective_metric`'s authoring prompt. No live LLM calls: the
Anthropic client is a hand-rolled stub whose `messages.create` captures
the exact kwargs it was called with (mirrors the `_StubClient` pattern
in `test_decompose.py`, adapted to the `.create` + `.content` shape
`_sample_source` consumes rather than `.parse` + `.parsed_output`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sculptor.eval.metric_gen import generate_objective_metric
from sculptor.reference import make_procedural_jump_clip


# ── stub Anthropic client (messages.create / .content / .stop_reason) ───
class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _StubResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [_TextBlock(text)]
        self.stop_reason = stop_reason
        self.usage = None


# A metric source that FAILS validation trivially (no compute_spec) — the
# signature-injection tests only care about the PROMPT that was sent, not
# whether the sampled candidate is accepted, so keep the fixture minimal
# and let validation reject it (bounded max_attempts=1, review=False).
_TRIVIAL_SOURCE = "import numpy as np\n# no compute_spec on purpose\n"


class _CapturingClient:
    """Records every `messages.create(**kwargs)` call and always returns
    `_TRIVIAL_SOURCE`. `n_candidates`/`max_attempts` control how many
    times `create` fires; this stub never runs out (unlike the parse-
    based `_StubClient` elsewhere) since metric_gen's retry loop count
    is fully caller-controlled via `max_attempts`."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(_TRIVIAL_SOURCE)


def _user_content(call: dict[str, Any]) -> str:
    return call["messages"][0]["content"]


def _mk_clip() -> dict:
    return make_procedural_jump_clip()


# ── signature-block injection ───────────────────────────────────────────
def test_references_none_omits_signature_block(tmp_path: Path):
    client = _CapturingClient()
    generate_objective_metric(
        "stand up", tmp_path, client=client, max_attempts=1, review=False)
    assert len(client.calls) == 1
    assert "REFERENCE MOTION SIGNATURE" not in _user_content(client.calls[0])


def test_references_present_injects_signature_block_with_clip_id_and_numbers(
    tmp_path: Path,
):
    clip = _mk_clip()
    client = _CapturingClient()
    rec = generate_objective_metric(
        "jump up and land", tmp_path, client=client,
        max_attempts=1, review=False,
        references=[("jump_demo_clip", clip)],
    )
    assert len(client.calls) == 1
    content = _user_content(client.calls[0])
    assert "REFERENCE MOTION SIGNATURE" in content
    assert "jump_demo_clip" in content
    # The competent-motion grounding instruction is present.
    assert "scoring" in content.lower() and "ground" in content.lower()
    # Real numeric fields from kinematic_signature appear (duration_s,
    # root_z block) — not just the clip id.
    assert "duration_s" in content
    assert "root_z" in content

    # Result dict records what was used.
    assert rec["references_used"] == ["jump_demo_clip"]
    assert "jump_demo_clip" in rec["reference_signatures"]
    assert "duration_s" in rec["reference_signatures"]["jump_demo_clip"]


def test_multiple_references_all_rendered(tmp_path: Path):
    clip_a = _mk_clip()
    clip_b = make_procedural_jump_clip(apex_gain_m=0.1)
    client = _CapturingClient()
    rec = generate_objective_metric(
        "jump up and land", tmp_path, client=client,
        max_attempts=1, review=False,
        references=[("clip_a", clip_a), ("clip_b", clip_b)],
    )
    content = _user_content(client.calls[0])
    assert "clip_a" in content and "clip_b" in content
    assert rec["references_used"] == ["clip_a", "clip_b"]
    assert set(rec["reference_signatures"]) == {"clip_a", "clip_b"}


def test_invalid_reference_clip_records_error_without_crashing(tmp_path: Path):
    """A clip that fails `validate_clip` (kinematic_signature raises) must
    not crash generation — it's recorded under `_error` and skipped in
    the rendered block, everything else proceeds normally."""
    bad_clip = {"root_pos_z": "not-an-array", "fps": 30.0}
    client = _CapturingClient()
    rec = generate_objective_metric(
        "jump up and land", tmp_path, client=client,
        max_attempts=1, review=False,
        references=[("broken_clip", bad_clip)],
    )
    assert rec["references_used"] == ["broken_clip"]
    assert "_error" in rec["reference_signatures"]["broken_clip"]


# ── references threaded into validate_generated_metric ─────────────────
def test_references_passed_through_to_validation(tmp_path: Path, monkeypatch):
    """`generate_objective_metric` must forward `references=` into
    `validate_generated_metric` (already-existing param on that
    function) so candidates certify against the reference-anchored
    gates, per §REFERENCE_TRAJECTORY_PLAN §5."""
    captured: list[Any] = []

    import sculptor.eval.metric_gen as mg

    real_validate = mg.validate_generated_metric

    def spy_validate(*args, **kwargs):
        captured.append(kwargs.get("references"))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(mg, "validate_generated_metric", spy_validate)

    clip = _mk_clip()
    client = _CapturingClient()
    generate_objective_metric(
        "jump up and land", tmp_path, client=client,
        max_attempts=1, review=False,
        references=[("jump_demo_clip", clip)],
    )
    assert len(captured) == 1
    refs = captured[0]
    assert refs is not None and refs[0][0] == "jump_demo_clip"


# ── retry loop feeds reference-gate failures back like other gates ─────
def test_retry_loop_feeds_back_validation_reasons_with_references(
    tmp_path: Path,
):
    """A first attempt that fails validation (trivial source, no
    compute_spec) must have its gate `reasons` fed into the SECOND
    attempt's prompt — the existing feedback mechanism — regardless of
    whether references are attached. This proves the reference path
    didn't fork the retry-feedback wiring."""
    clip = _mk_clip()
    client = _CapturingClient()
    rec = generate_objective_metric(
        "jump up and land", tmp_path, client=client,
        max_attempts=2, review=False,
        references=[("jump_demo_clip", clip)],
    )
    assert len(client.calls) == 2
    second_prompt = _user_content(client.calls[1])
    assert "FAILED these validation gates" in second_prompt
    # The reference block is still present on the retry (base_user carries it).
    assert "jump_demo_clip" in second_prompt
    assert rec["accepted"] is False
