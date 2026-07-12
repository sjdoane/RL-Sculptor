"""tests/test_criterion_ground.py — §D24 F2 (docs/internal/
REFERENCE_BUILD_LOG.md D23/D24): criterion re-grounding.

`sculptor.decompose.ground_stage_criterion` re-derives a stage's blind-
authored `success_criterion` from the CROPPED reference clip's real
kinematic signature once a span attaches, closing the D23 defect: the
criterion was authored in the SAME LLM call as goal_text, before any
per-stage clip was known, so `root_height > 0.35` was demanded of a
stage whose goal-aligned span never exceeds ~0.16 m.

No real Anthropic calls: `_StubClient` (reused from test_decompose.py)
stands in for the LLM. Uses the REAL fixture clip at
fixtures/torso_righting_satup/reference_clip.npz (the exact clip D23
diagnosed), cropped to the canonical [0, 8.5]s torso_righting span
(same convention as test_reference_spans.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sculptor.decompose import (
    _CriterionGroundModel,
    _clip_to_trajectory_namespace,
    _conjunct_is_component_free,
    _mechanically_verify_criterion_on_clip,
    _split_top_level_and,
    ground_stage_criterion,
)
from sculptor.mission import Stage
from sculptor.reference import load_clip
from sculptor.refs.spans import crop_span
from tests.test_decompose import _StubClient

FIXTURE_CLIP = (
    Path(__file__).parent / "fixtures" / "torso_righting_satup" / "reference_clip.npz"
)

#: The actual D23 defect: authored blind, before any clip was known —
#: demands root_height > 0.35, a height the goal-aligned span (pelvis
#: stays ~0.10-0.16 m) never reaches.
D23_ORIGINAL_CRITERION = (
    "(trajectory['root_height'] > 0.35).mean() > 0.25 and "
    "(trajectory['projected_gravity_b'][..., 2] < -0.5).mean() > 0.25 and "
    "components.get('righting_progress', 0.0) > 0.2"
)

#: A corrected criterion grounded in the CROPPED span's real numbers
#: (g_z-based uprightness instead of the unreachable height demand) —
#: verified (offline, this file) to pass on the fixture's [0, 8.5]s span.
CORRECTED_CRITERION = (
    "(trajectory['projected_gravity_b'][..., 2] < -0.55).mean() > 0.25 and "
    "components.get('righting_progress', 0.0) > 0.2"
)


@pytest.fixture(scope="module")
def cropped_clip() -> dict:
    clip = load_clip(FIXTURE_CLIP)
    return crop_span(clip, 0.0, 8.5)


def _stage(criterion: str = D23_ORIGINAL_CRITERION) -> Stage:
    return Stage(
        name="torso_righting",
        goal_text="Right the torso from lying flat to an upright seated posture.",
        success_criterion=criterion,
        max_iterations=8,
        parent_stage=None,
        reward_seed_prompt="upright_progress (rewards g_z becoming more negative)",
    )


# ── mechanical verifier (pure function, no LLM) ─────────────────────────
def test_original_d23_criterion_fails_mechanical_verification(cropped_clip):
    """Pins the D23 defect itself: the ORIGINAL criterion's height
    conjunct fails on the very clip it was meant to certify."""
    reason = _mechanically_verify_criterion_on_clip(
        D23_ORIGINAL_CRITERION, cropped_clip)
    assert reason is not None
    assert "root_height" in reason


def test_corrected_criterion_passes_mechanical_verification(cropped_clip):
    reason = _mechanically_verify_criterion_on_clip(
        CORRECTED_CRITERION, cropped_clip)
    assert reason is None


def test_components_only_criterion_is_vacuously_accepted(cropped_clip):
    """No mechanically-checkable conjunct exists (the whole expression is
    a single fail-closed `components...` clause) — accepted on trust,
    nothing to verify."""
    reason = _mechanically_verify_criterion_on_clip(
        "components.get('righting_progress', 0.0) > 0.2", cropped_clip)
    assert reason is None


def test_fail_open_components_default_is_rejected(cropped_clip):
    """LIVE FINDING (first real criterion_ground call): the rewrite came
    back with `components.get('righting_progress', 1.0) > 0.2` — a
    default that makes the conjunct vacuously True whenever the
    component is missing, silently deleting one leg of the criterion.
    Components conjuncts must be FAIL-CLOSED: evaluated with an empty
    components dict they must NOT pass."""
    reason = _mechanically_verify_criterion_on_clip(
        "components.get('righting_progress', 1.0) > 0.2", cropped_clip)
    assert reason is not None
    assert "FAIL-OPEN" in reason

    # Mixed criterion: a healthy trajectory conjunct does not rescue a
    # fail-open components conjunct.
    reason = _mechanically_verify_criterion_on_clip(
        "(trajectory['root_height'] > 0.05).mean() > 0.1 and "
        "components.get('x', 2.0) > 1.0",
        cropped_clip)
    assert reason is not None
    assert "FAIL-OPEN" in reason

    # components['x'] raises on missing — fail-closed, still skipped.
    reason = _mechanically_verify_criterion_on_clip(
        "components['righting_progress'] > 0.2", cropped_clip)
    assert reason is None


def test_unparseable_criterion_fails_verification(cropped_clip):
    reason = _mechanically_verify_criterion_on_clip("(((", cropped_clip)
    assert reason is not None
    assert "unparseable" in reason


def test_split_top_level_and_splits_bare_and():
    import ast

    tree = ast.parse("a > 0 and b > 0 and c > 0", mode="eval")
    conjuncts = _split_top_level_and(tree)
    assert len(conjuncts) == 3


def test_split_top_level_and_single_clause_is_one_conjunct():
    import ast

    tree = ast.parse("a > 0 or b > 0", mode="eval")
    conjuncts = _split_top_level_and(tree)
    assert len(conjuncts) == 1


def test_conjunct_is_component_free_detects_components_name():
    import ast

    tree = ast.parse("components.get('x', 0.0) > 0.2", mode="eval")
    assert _conjunct_is_component_free(tree.body) is False

    tree2 = ast.parse("trajectory['root_height'].mean() > 0.1", mode="eval")
    assert _conjunct_is_component_free(tree2.body) is True


def test_clip_to_trajectory_namespace_has_root_height_and_gravity(cropped_clip):
    ns = _clip_to_trajectory_namespace(cropped_clip)
    assert "root_height" in ns["trajectory"]
    assert "projected_gravity_b" in ns["trajectory"]
    assert ns["trajectory"] is ns["info"]  # alias, matches the real namespace
    assert ns["components"] == {}
    assert ns["behavior"] == {}


def test_clip_to_trajectory_namespace_omits_gravity_without_quat():
    ns = _clip_to_trajectory_namespace({"root_pos_z": [0.1, 0.2, 0.3], "fps": 30.0})
    assert "root_height" in ns["trajectory"]
    assert "projected_gravity_b" not in ns["trajectory"]


# ── ground_stage_criterion (mocked LLM via _StubClient) ────────────────
def test_ground_stage_criterion_adopts_a_correct_rewrite(cropped_clip):
    stage = _stage()
    client = _StubClient(_CriterionGroundModel(
        rationale="root_height>0.35 is unreachable on this sub-span; "
                   "g_z uprightness is what the span actually measures",
        success_criterion=CORRECTED_CRITERION,
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is True
    assert stage.success_criterion == CORRECTED_CRITERION
    assert "g_z" in result["rationale"] or "root_height" in result["rationale"]

    # The stub got the cropped signature + goal in its payload.
    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "torso_righting" in user_content
    assert "root_z" in user_content  # kinematic_signature's own key


def test_ground_stage_criterion_rejects_a_rewrite_that_fails_its_own_exemplar(
    cropped_clip,
):
    """The realistic failure mode: Claude's rewrite still demands a
    height the reference span never reaches — REJECTED, original kept,
    reason recorded. This is the guard closing the D23 class: a wrong
    re-grounded criterion must be HARDER to adopt than doing nothing."""
    stage = _stage()
    original = stage.success_criterion
    bad_rewrite = "(trajectory['root_height'] > 0.5).mean() > 0.1"
    client = _StubClient(_CriterionGroundModel(
        rationale="tightened the height bar", success_criterion=bad_rewrite,
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is False
    assert "root_height" in result["rationale"]
    assert stage.success_criterion == original  # untouched


def test_ground_stage_criterion_skips_components_only_conjuncts(cropped_clip):
    """A rewrite whose ONLY conjunct references `components` has nothing
    mechanically checkable — accepted (adopted) rather than rejected for
    lack of verification."""
    stage = _stage()
    client = _StubClient(_CriterionGroundModel(
        rationale="height was never the right signal for this stage",
        success_criterion="components.get('righting_progress', 0.0) > 0.3",
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is True
    assert stage.success_criterion == "components.get('righting_progress', 0.0) > 0.3"


def test_ground_stage_criterion_llm_failure_keeps_original(cropped_clip):
    """`_parse_with_retry` retries once internally on ANY exception
    (including a network failure, not just a parse error) — queue the
    same failure twice so the outer `ground_stage_criterion` call sees
    the retry exhaust and re-raise."""
    stage = _stage()
    original = stage.success_criterion
    client = _StubClient(
        RuntimeError("connection reset"), RuntimeError("connection reset"))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is False
    assert "connection reset" in result["rationale"]
    assert stage.success_criterion == original


def test_ground_stage_criterion_no_change_returns_not_adopted(cropped_clip):
    """The model may legitimately conclude nothing needs fixing — the
    original criterion is echoed back; this is NOT an adoption (no
    mutation happened, mirrors reconcile_criterion's identical-rewrite
    handling, just non-fatal here instead of a validation error)."""
    stage = _stage()
    client = _StubClient(_CriterionGroundModel(
        rationale="already consistent with the signature",
        success_criterion=stage.success_criterion,
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is False
    assert stage.success_criterion == D23_ORIGINAL_CRITERION


def test_ground_stage_criterion_empty_rewrite_keeps_original(cropped_clip):
    stage = _stage()
    client = _StubClient(_CriterionGroundModel(
        rationale="nothing to change", success_criterion="   ",
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is False
    assert stage.success_criterion == D23_ORIGINAL_CRITERION


def test_ground_stage_criterion_unsafe_rewrite_fails_static_gate(cropped_clip):
    """A rewrite that passes the mechanical clip-check but fails the
    static safety validator (lambda, disallowed node) must still be
    rejected — the mechanical check is ADDITIVE to, not a replacement
    for, the existing safety gates."""
    stage = _stage()
    client = _StubClient(_CriterionGroundModel(
        rationale="sneaky", success_criterion="(lambda: True)()",
    ))
    result = ground_stage_criterion(stage, cropped_clip, client=client)
    assert result["adopted"] is False
    assert stage.success_criterion == D23_ORIGINAL_CRITERION


def test_ground_stage_criterion_signature_failure_keeps_original():
    """A clip that can't be signature'd (e.g. missing fps) degrades to
    keeping the original — never raises, never calls the LLM."""
    stage = _stage()
    original = stage.success_criterion
    client = _StubClient(_CriterionGroundModel(
        rationale="should never be reached", success_criterion="metric > 0",
    ))
    result = ground_stage_criterion(stage, {}, client=client)
    assert result["adopted"] is False
    assert stage.success_criterion == original
    assert client.messages.calls == []  # never reached the LLM call
