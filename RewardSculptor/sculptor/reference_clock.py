"""Immutable policy interface for phase-indexed reference tracking."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping


REFERENCE_CLOCK_SCHEMA = 1
REFERENCE_CLOCK_TERM = "reference_phase"
REFERENCE_CLOCK_SOURCE = "reference_clock_observation"
REFERENCE_CLOCK_KIND = "per_environment_episode_elapsed_control_time"
REFERENCE_CLOCK_WIDTH = 1

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_PHASE_MODES = frozenset({"hold", "loop"})


def reference_playback_duration_s(*, frame_count: int, fps: float) -> float:
    """Return the exact sampled-trajectory duration used by every clock.

    A clip with ``N`` samples has ``N - 1`` sample intervals.  A one-sample
    static reference is assigned one sample period so its normalized clock is
    still well-defined and strictly positive.
    """
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 1
    ):
        raise ValueError("reference frame_count must be a positive integer")
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("reference fps must be positive")
    return max(1.0, float(frame_count - 1)) / float(fps)


def reference_target_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical identity of the exact embedded target tables."""
    canonical = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_reference_clock(
    *,
    clip_id: str,
    robot: str,
    target_sha256: str,
    phase_mode: str,
    phase_duration_s: float,
    n_phase_targets: int,
) -> dict[str, Any]:
    """Build and validate the one-column reference observation descriptor."""
    return validate_reference_clock({
        "schema": REFERENCE_CLOCK_SCHEMA,
        "term_name": REFERENCE_CLOCK_TERM,
        "source": REFERENCE_CLOCK_SOURCE,
        "encoding": "normalized_phase",
        "shape": [REFERENCE_CLOCK_WIDTH],
        "clock": REFERENCE_CLOCK_KIND,
        "reset_semantics": "per_environment_episode_reset",
        "phase_mode": phase_mode,
        "phase_duration_s": float(phase_duration_s),
        "n_phase_targets": int(n_phase_targets),
        "reference_clip_id": clip_id,
        "reference_robot": robot,
        "reference_target_sha256": target_sha256,
    })


def validate_reference_clock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical copy, rejecting missing, extra, or vague evidence."""
    if not isinstance(value, Mapping):
        raise ValueError("reference clock must be a mapping")
    expected_keys = {
        "schema", "term_name", "source", "encoding", "shape", "clock",
        "reset_semantics", "phase_mode", "phase_duration_s",
        "n_phase_targets", "reference_clip_id", "reference_robot",
        "reference_target_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError(
            "reference clock fields differ from schema 1: "
            f"missing={sorted(expected_keys - set(value))}, "
            f"extra={sorted(set(value) - expected_keys)}"
        )
    fixed = {
        "schema": REFERENCE_CLOCK_SCHEMA,
        "term_name": REFERENCE_CLOCK_TERM,
        "source": REFERENCE_CLOCK_SOURCE,
        "encoding": "normalized_phase",
        "shape": [REFERENCE_CLOCK_WIDTH],
        "clock": REFERENCE_CLOCK_KIND,
        "reset_semantics": "per_environment_episode_reset",
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise ValueError(f"reference clock {key} must be {expected!r}")
    if value.get("phase_mode") not in _PHASE_MODES:
        raise ValueError("reference clock phase_mode must be hold or loop")
    duration = value.get("phase_duration_s")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise ValueError("reference clock phase_duration_s must be positive")
    n_targets = value.get("n_phase_targets")
    if (
        not isinstance(n_targets, int)
        or isinstance(n_targets, bool)
        or n_targets <= 0
    ):
        raise ValueError("reference clock n_phase_targets must be positive")
    for key in ("reference_clip_id", "reference_robot"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"reference clock {key} must be non-empty text")
    target_sha = value.get("reference_target_sha256")
    if not isinstance(target_sha, str) or not _SHA256_RE.fullmatch(target_sha):
        raise ValueError(
            "reference clock reference_target_sha256 must be lowercase SHA-256"
        )
    return copy.deepcopy(dict(value))


def reference_clock_from_reward_spec(
    reward_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read a generated reward's policy interface, failing closed if claimed."""
    if not isinstance(reward_spec, Mapping):
        raise ValueError("reward spec must be a mapping")
    raw = reward_spec.get("reference_clock")
    declares_reference = bool(
        reward_spec.get("reference_tracking")
        or reward_spec.get("tracking_enabled")
    )
    if raw is None:
        if declares_reference:
            raise ValueError(
                "reference-tracking reward has no reference_clock descriptor"
            )
        return None
    if not declares_reference:
        raise ValueError("reward exposes reference_clock without tracking")
    if not isinstance(raw, Mapping):
        raise ValueError("reward reference_clock must be a mapping")
    return validate_reference_clock(raw)


def reference_clock_from_reward_source(source: str) -> dict[str, Any] | None:
    """Read a generated reward descriptor without importing uploaded code."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("reward source must be non-empty text")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("reward source is not valid Python syntax") from exc
    reward_spec: Any = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "REWARD_SPEC"
            for target in targets
        ):
            continue
        try:
            reward_spec = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward REWARD_SPEC must be literal data") from exc
    if reward_spec is None:
        # Legacy/non-reference rewards are allowed to omit REWARD_SPEC. A
        # literal spec that claims tracking remains fail-closed in
        # ``reference_clock_from_reward_spec`` below.
        return None
    return reference_clock_from_reward_spec(reward_spec)


def reference_clock_from_module(module: Any) -> dict[str, Any] | None:
    """Validate a loaded reward module's descriptor and executable surface."""
    reward_spec = getattr(module, "REWARD_SPEC", None)
    # Legacy/non-reference rewards predate REWARD_SPEC and remain valid.  A
    # module that does publish a spec is still validated strictly below.
    if reward_spec is None:
        return None
    descriptor = reference_clock_from_reward_spec(reward_spec)
    if descriptor is None:
        return None
    for name in ("reference_clock_batched", "reference_target_index_batched"):
        if not callable(getattr(module, name, None)):
            raise ValueError(
                f"reference-tracking reward is missing callable {name}"
            )
    return descriptor


__all__ = [
    "REFERENCE_CLOCK_KIND", "REFERENCE_CLOCK_SCHEMA",
    "REFERENCE_CLOCK_SOURCE", "REFERENCE_CLOCK_TERM", "REFERENCE_CLOCK_WIDTH",
    "build_reference_clock", "reference_clock_from_module",
    "reference_clock_from_reward_source", "reference_clock_from_reward_spec",
    "reference_playback_duration_s", "reference_target_sha256",
    "validate_reference_clock",
]
