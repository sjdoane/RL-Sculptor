"""Per-project environment specification — the general task-env
adaptation layer (§RL_SCULPTOR_AUDIT, 2026-07-04 env generalization).

Replaces the single hardcoded ``env_profile="jump"`` preset with
declarative, validated, per-project DATA. The schema speaks physical
semantics (reset ranges, termination thresholds, push events, friction
randomization, episode length, PPO exploration); each adapter maps
those semantics onto its own task cfg (see
``_mjlab_runner._apply_env_spec``). Values are chosen per project —
generated from the behavior goal, or iterated by the diagnoser — so
nothing here is robot- or task-specific.

Two scopes, one firm invariant each:

``shared``
    Applied to BOTH training and rollout evaluation. The policy is
    evaluated under its training task (commands, terminations, episode
    length). FROZEN for the duration of a sculpt run — rollout-side
    changes mid-run would make the metric incomparable across
    iterations.

``train``
    TRAIN-ONLY curricula (reference-state initialization ranges, early
    termination off the recoverable manifold, domain randomization,
    exploration). Never applied to rollout, so the metric's view of the
    task — honest starts, honest physics, full episodes — is never
    corrupted. This is the section the diagnoser may edit between
    iterations.

Validation discipline mirrors the reward/metric pipelines: strict
schema (unknown keys REJECTED — a typo must fail loudly, not silently
no-op), hard per-field bounds, cross-field invariants (ranges
well-ordered). ``validate_env_spec`` returns every violation at once so
a generator gets complete feedback in one retry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ENV_SPEC_VERSION = 1

# ── Bounds tables ──────────────────────────────────────────────────────────
# Each entry: (lo, hi) hard bounds for scalars, or for [lo, hi] range
# fields the allowed envelope both endpoints must lie in. These are
# SAFETY rails (a spec outside them is a generation/edit bug), not
# tuning guidance — generators pick task-appropriate values inside.
_SHARED_SCALARS: dict[str, tuple[float, float]] = {
    "orientation_termination_deg": (45.0, 179.0),
    "episode_length_s": (2.0, 60.0),
}
_TRAIN_SCALARS: dict[str, tuple[float, float]] = {
    "min_base_height_termination_m": (0.02, 1.0),
    "entropy_coef_scale": (0.25, 4.0),
}
_TRAIN_RANGES: dict[str, tuple[float, float]] = {
    # Offsets ADDED to the robot's default reset state (mjlab
    # reset_root_state_uniform semantics), so they are robot-relative.
    "reset_height_offset_m": (0.0, 1.5),
    "reset_vertical_velocity_mps": (-3.0, 4.0),
    "reset_horizontal_velocity_mps": (-3.0, 3.0),
    # Joint-space reset offsets (radians / rad/s) around defaults.
    "reset_joint_position_offset_rad": (-1.5, 1.5),
    "reset_joint_velocity_radps": (-10.0, 10.0),
    # Startup domain randomization of foot-geom friction.
    "friction_range": (0.05, 2.5),
}
_PUSH_KEYS = {"enabled", "interval_s", "linear_mps", "angular_radps"}
_PUSH_INTERVAL_ENVELOPE = (0.5, 30.0)
_PUSH_LINEAR_MAX = 2.0
_PUSH_ANGULAR_MAX = 3.0

_SHARED_KEYS = set(_SHARED_SCALARS) | {"zero_velocity_commands", "push_events"}
_TRAIN_KEYS = set(_TRAIN_SCALARS) | set(_TRAIN_RANGES) | {"push_events"}
_TOP_KEYS = {"env_spec_version", "meta", "shared", "train"}

#: Train-section keys the diagnoser may propose changes to between
#: iterations (everything in ``train``; ``shared`` is frozen per run).
ITERABLE_TRAIN_KEYS: frozenset[str] = frozenset(
    set(_TRAIN_SCALARS) | set(_TRAIN_RANGES)
)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def _check_range(name: str, v: Any, envelope: tuple[float, float],
                 errors: list[str]) -> None:
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(_is_num(x) for x in v)):
        errors.append(f"{name}: must be a [lo, hi] pair of numbers, got {v!r}")
        return
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        errors.append(f"{name}: lo {lo} > hi {hi}")
    for x in (lo, hi):
        if not (envelope[0] <= x <= envelope[1]):
            errors.append(
                f"{name}: {x} outside hard bounds [{envelope[0]}, {envelope[1]}]")


def _check_scalar(name: str, v: Any, bounds: tuple[float, float],
                  errors: list[str]) -> None:
    if not _is_num(v):
        errors.append(f"{name}: must be a finite number, got {v!r}")
        return
    if not (bounds[0] <= float(v) <= bounds[1]):
        errors.append(
            f"{name}: {v} outside hard bounds [{bounds[0]}, {bounds[1]}]")


def _check_push(name: str, v: Any, errors: list[str]) -> None:
    if not isinstance(v, dict):
        errors.append(f"{name}: must be an object, got {v!r}")
        return
    unknown = set(v) - _PUSH_KEYS
    if unknown:
        errors.append(f"{name}: unknown keys {sorted(unknown)}")
    if "enabled" not in v or not isinstance(v["enabled"], bool):
        errors.append(f"{name}: requires boolean 'enabled'")
    if "interval_s" in v:
        _check_range(f"{name}.interval_s", v["interval_s"],
                     _PUSH_INTERVAL_ENVELOPE, errors)
    if "linear_mps" in v:
        _check_scalar(f"{name}.linear_mps", v["linear_mps"],
                      (0.0, _PUSH_LINEAR_MAX), errors)
    if "angular_radps" in v:
        _check_scalar(f"{name}.angular_radps", v["angular_radps"],
                      (0.0, _PUSH_ANGULAR_MAX), errors)


def validate_env_spec(spec: Any) -> list[str]:
    """Every violation in one pass (so a generator retry gets complete
    feedback). Empty list == valid."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"env spec must be a JSON object, got {type(spec).__name__}"]
    unknown = set(spec) - _TOP_KEYS
    if unknown:
        errors.append(f"unknown top-level keys {sorted(unknown)}")
    if spec.get("env_spec_version") != ENV_SPEC_VERSION:
        errors.append(
            f"env_spec_version must be {ENV_SPEC_VERSION}, "
            f"got {spec.get('env_spec_version')!r}")
    if "meta" in spec and not isinstance(spec["meta"], dict):
        errors.append("meta: must be an object")

    shared = spec.get("shared", {})
    if not isinstance(shared, dict):
        errors.append("shared: must be an object")
        shared = {}
    unknown = set(shared) - _SHARED_KEYS
    if unknown:
        errors.append(
            f"shared: unknown keys {sorted(unknown)} "
            f"(allowed: {sorted(_SHARED_KEYS)})")
    if ("zero_velocity_commands" in shared
            and not isinstance(shared["zero_velocity_commands"], bool)):
        errors.append("shared.zero_velocity_commands: must be a boolean")
    for k, bounds in _SHARED_SCALARS.items():
        if k in shared:
            _check_scalar(f"shared.{k}", shared[k], bounds, errors)
    if "push_events" in shared:
        _check_push("shared.push_events", shared["push_events"], errors)

    train = spec.get("train", {})
    if not isinstance(train, dict):
        errors.append("train: must be an object")
        train = {}
    unknown = set(train) - _TRAIN_KEYS
    if unknown:
        errors.append(
            f"train: unknown keys {sorted(unknown)} "
            f"(allowed: {sorted(_TRAIN_KEYS)})")
    for k, bounds in _TRAIN_SCALARS.items():
        if k in train:
            _check_scalar(f"train.{k}", train[k], bounds, errors)
    for k, envelope in _TRAIN_RANGES.items():
        if k in train:
            _check_range(f"train.{k}", train[k], envelope, errors)
    if "push_events" in train:
        _check_push("train.push_events", train["push_events"], errors)

    # Cross-field invariant (MEASURED requirement, tuck-jump iters
    # 19-20: RSI without early termination off the recoverable manifold
    # REGRESSES — failed episodes' floor data dominates PPO's
    # distribution): airborne/upward reference-state starts REQUIRE the
    # min-base-height termination. Horizontal-only or downward-only
    # reset jitter doesn't trigger it.
    def _hi(v: Any) -> float:
        try:
            return float(v[1])
        except (TypeError, ValueError, IndexError):
            return 0.0

    rsi_airborne = (_hi(train.get("reset_height_offset_m")) > 0.0
                    or _hi(train.get("reset_vertical_velocity_mps")) > 0.0)
    if rsi_airborne and not _is_num(
            train.get("min_base_height_termination_m")):
        errors.append(
            "train: reset_height_offset_m / upward "
            "reset_vertical_velocity_mps (reference-state initialization "
            "into the air) REQUIRE train.min_base_height_termination_m — "
            "measured: RSI without early termination off the recoverable "
            "manifold regresses (floor data dominates training)")
    return errors


def load_env_spec(path: Path | str) -> dict:
    """Read + validate an env spec JSON file. Raises ValueError with the
    full violation list (or the JSON error) — an invalid spec must fail
    loudly before any GPU time is spent under it."""
    p = Path(path)
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"env spec unreadable at {p}: {type(e).__name__}: {e}")
    errors = validate_env_spec(spec)
    if errors:
        raise ValueError(
            f"env spec invalid at {p}:\n  - " + "\n  - ".join(errors))
    return spec


# ── Project-side versioning ────────────────────────────────────────────────
# A project's env specs live in `<project>/env/` as `v<N>.json` plus
# `current.json` — an exact COPY of the active version (its identity is
# `meta.version` inside the file; no symlinks, so the layout survives
# Windows/WSL round-trips and remote pod sync). Mirrors the rewards/
# convention: monotonic version numbers, current repointed by
# keep-best/revert.

def find_latest_env_version(env_dir: Path) -> int:
    """Highest existing v<N>.json number, or -1 when none exist."""
    latest = -1
    if Path(env_dir).is_dir():
        for p in Path(env_dir).glob("v*.json"):
            stem = p.stem
            if stem.startswith("v") and stem[1:].isdigit():
                latest = max(latest, int(stem[1:]))
    return latest


def write_env_spec_version(env_dir: Path, spec: dict) -> Path:
    """Persist `spec` as the next v<N>.json and repoint current.json at
    it. Validates FIRST — an invalid spec is never written to disk.
    `meta.version` is stamped into the file (the identity current.json
    carries)."""
    env_dir = Path(env_dir)
    to_write = json.loads(json.dumps(spec))   # deep copy, JSON-clean
    errors = validate_env_spec(to_write)
    if errors:
        raise ValueError(
            "refusing to persist invalid env spec:\n  - "
            + "\n  - ".join(errors))
    n = find_latest_env_version(env_dir) + 1
    meta = to_write.setdefault("meta", {})
    meta["version"] = f"v{n}"
    env_dir.mkdir(parents=True, exist_ok=True)
    path = env_dir / f"v{n}.json"
    text = json.dumps(to_write, indent=2, sort_keys=True)
    # tmp+replace for BOTH files — a crash mid-write must not leave a
    # truncated version file (which would burn its number and fail any
    # later revert onto it) any more than a truncated current.json.
    tmp_v = env_dir / f"v{n}.json.tmp"
    tmp_v.write_text(text, encoding="utf-8")
    tmp_v.replace(path)
    tmp = env_dir / "current.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(env_dir / "current.json")
    return path


def read_current_env_spec(env_dir: Path) -> "dict | None":
    """The active spec (validated), or None when the project has no env
    spec. An invalid current.json raises — never silently train under
    a half-readable spec."""
    p = Path(env_dir) / "current.json"
    if not p.is_file():
        return None
    return load_env_spec(p)


def repoint_env_current(env_dir: Path, version: str) -> Path:
    """Repoint current.json at an existing v<N>.json (keep-best /
    revert). Raises if the version file is missing or invalid."""
    env_dir = Path(env_dir)
    src = env_dir / f"{version}.json"
    spec = load_env_spec(src)   # raises on missing/invalid
    text = json.dumps(spec, indent=2, sort_keys=True)
    tmp = env_dir / "current.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(env_dir / "current.json")
    return env_dir / "current.json"


def apply_env_edits(env_dir: Path, edits: list) -> dict:
    """§env generalization 3/4: apply diagnoser-proposed TRAIN-section
    edits to the project's active env spec — the environment counterpart
    of reward apply_edits, with the same validation discipline.

    Each edit is `{parameter, new_value, rationale}` (duck-typed:
    attributes or dict keys). Per-edit gates: parameter must be in
    ITERABLE_TRAIN_KEYS (the shared/eval section is structurally
    unreachable), new_value must parse as JSON, and the spec must
    validate AFTER the edit (bounds + shape) — a failing edit is
    rejected with its reasons and the remaining edits still apply.
    Net changes are persisted as the next v<N>.json (current.json
    repointed); no valid edits → nothing written.

    Returns {"applied": [...], "rejected": [(param, reason)], "new_version":
    "v<N>" | None, "path": str | None}."""
    result: dict[str, Any] = {
        "applied": [], "rejected": [], "new_version": None, "path": None}

    def _field(e: Any, name: str) -> Any:
        if isinstance(e, dict):
            return e.get(name)
        return getattr(e, name, None)

    spec = read_current_env_spec(env_dir)   # raises on invalid current
    if spec is None:
        for e in edits:
            result["rejected"].append(
                (str(_field(e, "parameter")), "no active env spec"))
        return result

    work = json.loads(json.dumps(spec))
    changed = False
    for e in edits:
        param = str(_field(e, "parameter") or "")
        raw = _field(e, "new_value")
        if param not in ITERABLE_TRAIN_KEYS:
            result["rejected"].append(
                (param, f"not an iterable train-section parameter "
                        f"(allowed: {sorted(ITERABLE_TRAIN_KEYS)})"))
            continue
        try:
            value = json.loads(str(raw))
        except (TypeError, ValueError):
            result["rejected"].append(
                (param, f"new_value {raw!r} is not valid JSON"))
            continue
        prev_train = json.loads(json.dumps(work.get("train") or {}))
        work.setdefault("train", {})[param] = value
        errors = validate_env_spec(work)
        if errors:
            work["train"] = prev_train   # revert just this edit
            result["rejected"].append((param, "; ".join(errors)))
            continue
        result["applied"].append(f"{param}={json.dumps(value)}")
        changed = True

    if changed:
        meta = work.setdefault("meta", {})
        meta["source"] = "diagnoser"
        meta["parent"] = (spec.get("meta") or {}).get("version")
        rationales = [
            " ".join(str(_field(e, "rationale") or "").split())[:200]
            for e in edits
        ]
        meta["rationale"] = " | ".join(r for r in rationales if r)[:1000]
        path = write_env_spec_version(env_dir, work)
        result["new_version"] = path.stem
        result["path"] = str(path)
    return result


def jump_preset_spec() -> dict:
    """The former hardcoded ``env_profile="jump"`` expressed as an
    instance of the general mechanism. Values byte-match the retired
    ``_apply_env_profile`` mutations (parity-tested); see the audit
    doc's loop-4a/loop-6 entries for the measured rationale behind
    each number."""
    return {
        "env_spec_version": ENV_SPEC_VERSION,
        "meta": {"source": "preset:jump"},
        "shared": {
            "zero_velocity_commands": True,
            "orientation_termination_deg": 120.0,
            "episode_length_s": 10.0,
            "push_events": {"enabled": False},
        },
        "train": {
            "reset_height_offset_m": [0.0, 0.40],
            "reset_vertical_velocity_mps": [-0.5, 2.0],
            "min_base_height_termination_m": 0.30,
            "entropy_coef_scale": 2.0,
        },
    }
