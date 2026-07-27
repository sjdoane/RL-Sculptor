"""Hybrid LLM world author (sculptor/world/llm_author.py).

Verifies the plumbing + safety WITHOUT a live model: a fake client returns canned
replies. The invariant is that a draft is only ever returned if it PASSED the local
validators — a bad/errored model falls back to the deterministic offline author.
"""
from __future__ import annotations

import copy
import json
import types

import pytest

from sculptor.world.author import AuthoringError, author_environment
from sculptor.world.llm_author import (
    LLMWorldAuthor,
    _extract_json,
    hybrid_author_environment,
)
from sculptor.world.world_spec import validate_world_spec


def _reply(text: str, *, stop_reason: str = "end_turn"):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, usage=None)


class _FakeClient:
    """Minimal Anthropic-shaped client. `reply` is text or an Exception to raise."""

    def __init__(self, reply, *, stop_reason="end_turn"):
        self._reply = reply
        self._stop_reason = stop_reason
        self.messages = types.SimpleNamespace(create=self._create)
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return _reply(self._reply, stop_reason=self._stop_reason)


_PARKOUR = "Learn parkour over a course of boxes."


def _valid_spec_json(robot="unitree_g1:base"):
    """A real, gate-passing world/task pair (authored offline, replayed as if the
    LLM produced it) — same robot + provenance so the local gates admit it."""
    d = author_environment(_PARKOUR, robot_capability_id=robot)
    return json.dumps({
        "world_spec": d.world_spec, "task_spec": d.task_spec,
        "parameter_provenance": d.world_spec["meta"]["parameter_provenance"]})


# ── _extract_json ────────────────────────────────────────────────────────────

def test_extract_json_fenced_and_bare_and_invalid():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose {"a": 2} more') == {"a": 2}
    with pytest.raises(AuthoringError, match="no JSON object"):
        _extract_json("no json here")
    with pytest.raises(AuthoringError, match="invalid JSON"):
        _extract_json('{"a": bad}')


# ── LLMWorldAuthor.generate_authoring ────────────────────────────────────────

def test_generate_authoring_returns_closed_key_set():
    client = _FakeClient('{"world_spec":{"x":1},"task_spec":{"y":2},"junk":9}')
    author = LLMWorldAuthor(client=client, model_id="fake")
    out = author.generate_authoring({"prompt": "hi"})
    assert set(out) == {"world_spec", "task_spec"}   # "junk" dropped


def test_generate_authoring_truncation_raises():
    client = _FakeClient('{"world_spec":{}}', stop_reason="max_tokens")
    author = LLMWorldAuthor(client=client, model_id="fake")
    with pytest.raises(AuthoringError, match="truncated"):
        author.generate_authoring({"prompt": "hi"})


# ── hybrid: LLM used when its spec passes the gates ──────────────────────────

def test_hybrid_uses_llm_spec_when_valid():
    # The LLM returns a VALID spec carrying a distinctive box height the offline
    # template never produces; the hybrid must return THAT spec (LLM path used).
    seed = author_environment(_PARKOUR, robot_capability_id="unitree_g1:base")
    ws = copy.deepcopy(seed.world_spec)
    for c in ws["shared"]["obstacles"]["course"]:
        if c["id"] == "box_02":
            c["nominal"]["height_m"] = 0.199        # distinctive marker
    client = _FakeClient(json.dumps({
        "world_spec": ws, "task_spec": seed.task_spec,
        "parameter_provenance": seed.world_spec["meta"]["parameter_provenance"]}))
    draft = hybrid_author_environment(
        _PARKOUR, client=client, robot_capability_id="unitree_g1:base")
    assert client.calls == 1
    assert validate_world_spec(draft.world_spec) == []
    heights = {c["id"]: c["nominal"].get("height_m")
               for c in draft.world_spec["shared"]["obstacles"]["course"]}
    assert heights["box_02"] == 0.199               # the LLM value survived


# ── hybrid: fall back to offline when the model output fails the gates ────────

def test_hybrid_falls_back_when_llm_spec_invalid():
    # Model returns structurally-broken specs → local gate rejects → offline used.
    client = _FakeClient('{"world_spec":{"nope":1},"task_spec":{"nope":1}}')
    draft = hybrid_author_environment(
        "Learn parkour over a course of boxes.", client=client,
        robot_capability_id="unitree_g1:base")
    assert validate_world_spec(draft.world_spec) == []          # offline result
    assert draft.world_spec["shared"]["obstacles"]["course"]


def test_hybrid_rejects_valid_but_wrong_explicit_course_count():
    """Schema-valid geometry may still contradict an explicit prompt fact.

    The model returns the nominal three-platform template for a four-box
    request.  The semantic gate must reject it, and hybrid fallback must
    compile the requested four platforms rather than promote the drift.
    """
    seed = author_environment(_PARKOUR, robot_capability_id="unitree_g1:base")
    client = _FakeClient(json.dumps({
        "world_spec": seed.world_spec,
        "task_spec": seed.task_spec,
        "parameter_provenance": seed.world_spec["meta"]["parameter_provenance"],
    }))
    prompt = (
        "Build a parkour course with four progressively taller, "
        "high-friction boxes in a straight line."
    )

    draft = hybrid_author_environment(
        prompt, client=client, robot_capability_id="unitree_g1:base")

    platforms = [
        item for item in draft.world_spec["shared"]["obstacles"]["course"]
        if item["element"] == "platform"
    ]
    assert client.calls == 1
    assert len(platforms) == 4


def test_hybrid_rejects_slalom_that_drops_ordered_terminal_goal():
    prompt = (
        "Run a slalom around four boxes through ordered waypoints without "
        "touching them, then stop in the finish zone for 2 seconds."
    )
    seed = author_environment(prompt, robot_capability_id="unitree_g1:base")
    wrong_task = copy.deepcopy(seed.task_spec)
    wrong_task["shared"]["goal"] = {
        "id": "reach_finish", "type": "robot_to_region", "region": "finish",
        "success": {
            "predicate": "distance_below", "hold_s": 2.0,
            "tolerance_m": 0.35,
        },
    }
    client = _FakeClient(json.dumps({
        "world_spec": seed.world_spec,
        "task_spec": wrong_task,
        "parameter_provenance": seed.world_spec["meta"]["parameter_provenance"],
    }))

    draft = hybrid_author_environment(
        prompt, client=client, robot_capability_id="unitree_g1:base")

    assert client.calls == 1
    assert draft.task_spec["shared"]["goal"]["type"] == "waypoint_sequence"
    assert draft.task_spec["shared"]["goal"]["waypoints"][-1] == "finish"


def test_hybrid_falls_back_when_client_errors():
    client = _FakeClient(RuntimeError("api down"))
    draft = hybrid_author_environment(
        "Learn parkour over a course of boxes.", client=client,
        robot_capability_id="unitree_g1:base")
    assert validate_world_spec(draft.world_spec) == []          # offline result


def test_hybrid_rejects_robot_capability_swap():
    # A model that tries to swap the robot is rejected by the capability lock,
    # so the hybrid falls back to offline (never honors the swap).
    tampered = json.loads(_valid_spec_json())
    tampered["world_spec"]["shared"]["robot"]["capability_id"] = "yam:parallel_gripper"
    client = _FakeClient(json.dumps(tampered))
    draft = hybrid_author_environment(
        "Learn parkour over a course of boxes.", client=client,
        robot_capability_id="unitree_g1:base")
    assert draft.world_spec["shared"]["robot"]["capability_id"] == "unitree_g1:base"
