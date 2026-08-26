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
    # §sim2real DR (RMA arXiv 2107.04034; Dynamics Randomization 1710.06537):
    # per-link center-of-mass offset magnitude (m), sampled as (−off, +off) on
    # each of x/y/z per body at env startup. The real robot's link CoMs are never
    # exactly the CAD values, so the policy learns robustness to the shift. NOTE
    # this perturbs EVERY link independently (not just the trunk), so keep it
    # SMALL (~0.02–0.05) — a large per-link CoM shift is heavy DR. 0.0 == off.
    "com_offset_m": (0.0, 0.3),
    # Uniform per-joint noise magnitude (radians) added on top of
    # `reset_joint_pos_target` when present (§REFERENCE_TRAJECTORY_PLAN
    # §8 part 2 — get-up posture resets need SOME variation per episode,
    # same DeepMimic RSI rationale as the root-state ranges above, but a
    # single non-negative magnitude rather than a [lo, hi] pair since
    # it's applied symmetrically around each joint's target).
    "reset_joint_pos_noise_rad": (0.0, 1.0),
}
_TRAIN_RANGES: dict[str, tuple[float, float]] = {
    # Offsets ADDED to the robot's default reset state (mjlab
    # reset_root_state_uniform semantics), so they are robot-relative.
    # Envelope covers BOTH directions: positive for airborne/jump-style
    # RSI (above standing) and negative for get-up-style RSI (a lying
    # start is well below the standing default root height — G1 standing
    # ≈0.78 m, lying ≈0.1 m, so an offset near -0.7 m must be
    # expressible). §REFERENCE_TRAJECTORY_PLAN §8.
    "reset_height_offset_m": (-1.0, 1.5),
    "reset_vertical_velocity_mps": (-3.0, 4.0),
    "reset_horizontal_velocity_mps": (-3.0, 3.0),
    # Joint-space reset offsets (radians / rad/s) around defaults.
    "reset_joint_position_offset_rad": (-1.5, 1.5),
    "reset_joint_velocity_radps": (-10.0, 10.0),
    # Startup domain randomization of foot-geom friction.
    "friction_range": (0.05, 2.5),
    # §sim2real physics DR — the core dynamics-randomization axes the
    # literature says to randomize FIRST for zero-shot transfer (Dynamics
    # Randomization, arXiv 1710.06537: mass, damping, friction, motor gains;
    # RMA, 2107.04034: payload/mass, motor strength, friction; Rapid Locomotion,
    # 2205.02824 + Walk-These-Ways, 2212.03238: Kp/Kd + motor strength). Each is
    # a multiplicative [lo, hi] SCALE about the nominal model value, sampled
    # per-env at startup (so the 4096-env batch spans the distribution and the
    # per-iteration seed re-rolls it each round). Keep ranges MODERATE
    # (BeyondMimic 2508.08241: randomize only genuinely-uncertain params — too
    # wide dilutes the control objective and yields an overly-conservative
    # policy, FailureMode too_wide_dr_ranges). Absent == off.
    "body_mass_scale_range": (0.5, 2.0),          # link masses ×
    "pd_kp_scale_range": (0.5, 2.0),              # position-gain (stiffness) ×
    "pd_kd_scale_range": (0.5, 2.0),              # velocity-gain (damping) ×
    "motor_strength_scale_range": (0.5, 2.0),     # actuator effort limit ×
    "joint_damping_scale_range": (0.0, 5.0),      # joint (dof) damping ×
    "joint_armature_scale_range": (0.0, 5.0),     # rotor armature/inertia ×
    # Whole-body (not just foot) contact friction, ABSOLUTE tangential
    # coefficient — feet already covered by friction_range; this adds torso/
    # knee/arm contact friction so falls and braces transfer too.
    "body_friction_range": (0.05, 2.5),
    # Root-orientation reset offsets (radians), mjlab
    # `reset_root_state_uniform`'s `pose_range["roll"|"pitch"]` semantics
    # — a get-up clip's lying start is exactly a large pitch (face-up/
    # face-down) or roll (on-the-side) offset from the standing default.
    # Full ±pi envelope: a lying start can be oriented anywhere from
    # upright to fully inverted. §REFERENCE_TRAJECTORY_PLAN §8 part 1.
    "reset_pitch_offset_rad": (-math.pi, math.pi),
    "reset_roll_offset_rad": (-math.pi, math.pi),
}
#: Per-joint reference-posture reset target (§8 part 2): an explicit
#: `[q_0, ..., q_{J-1}]` qpos vector (radians) the adapter writes at
#: reset via a NEW custom event (`reset_joints_to_reference` — mjlab's
#: shipped `reset_joints_by_offset` only supports a single UNIFORM
#: offset range applied to every joint from the standing default, no
#: per-joint target; see `_mjlab_runner.py`'s injection for the mjlab
#: event-manager mechanism this rides). Length is validated against the
#: robot's own joint count at APPLY time (the schema layer doesn't know
#: the robot), so here only per-element bounds + a generous max-length
#: safety rail are enforced. Absent == today's behavior (no per-joint
#: reset). Paired scalar `reset_joint_pos_noise_rad` lives in
#: `_TRAIN_SCALARS` above.
_JOINT_TARGET_ELEMENT_BOUNDS = (-6.4, 6.4)   # generous; true joint limits
                                              # are enforced by mjlab's
                                              # own soft_joint_pos_limits
                                              # clamp at apply time.
_JOINT_TARGET_MAX_LEN = 64
_TRAIN_JOINT_TARGET_KEYS = {"reset_joint_pos_target"}
#: §DeepMimic phase RSI (arXiv 1804.02717): downsampled reference trajectories —
#: `[K][J]` matrices (K frames spanning the motion). `reset_joints_to_reference`
#: samples a random frame per env each reset, so the batch covers the whole
#: motion manifold, and initializes joint VELOCITY from the same frame (a dynamic
#: skill's mid-motion pose is meaningless at rest). Derived data, so NOT in
#: ITERABLE_TRAIN_KEYS (the diagnoser tunes scalar knobs, not the reference clip).
_TRAJ_MAX_FRAMES = 64
_JOINT_VEL_ELEMENT_BOUNDS = (-40.0, 40.0)   # generous rad/s rail; mjlab clamps
_TRAIN_TRAJECTORY_KEYS = {
    "reset_joint_pos_trajectory", "reset_joint_vel_trajectory"}
#: Boolean train-only switches. `fell_over_termination` (default True =
#: today's behavior, i.e. the task's standard orientation/bad-pose
#: termination stays active) — a GET-UP stage's lying reset (root
#: pitched/rolled far from upright, e.g. |pitch| ~ pi/2) IS exactly what
#: that termination is designed to detect as "fallen," so it fires on
#: every env at reset and the episode never runs (observed live:
#: Episode_Termination/fell_over = 2048.0 with all 2048 envs terminating
#: at step 0). Setting this False lets the adapter remove that term for
#: TRAIN only — rollout evaluation is unaffected (this key lives in
#: `train`, never `shared`), and the sunk-height termination + episode
#: time_out remain the episode enders. See
#: `_mjlab_runner._apply_env_spec` for the removal mechanism and
#: `reference.py`'s get-up branch for where this is derived.
_TRAIN_BOOLS = {"fell_over_termination"}
_PUSH_KEYS = {"enabled", "interval_s", "linear_mps", "angular_radps"}
_PUSH_INTERVAL_ENVELOPE = (0.5, 30.0)
_PUSH_LINEAR_MAX = 2.0
_PUSH_ANGULAR_MAX = 3.0

_SHARED_KEYS = set(_SHARED_SCALARS) | {"zero_velocity_commands", "push_events"}
_TRAIN_KEYS = (
    set(_TRAIN_SCALARS) | set(_TRAIN_RANGES) | _TRAIN_JOINT_TARGET_KEYS
    | _TRAIN_TRAJECTORY_KEYS | _TRAIN_BOOLS | {"push_events"}
)
_TOP_KEYS = {"env_spec_version", "meta", "shared", "train"}

#: Train-section keys the diagnoser may propose changes to between
#: iterations (everything in ``train``; ``shared`` is frozen per run).
ITERABLE_TRAIN_KEYS: frozenset[str] = frozenset(
    set(_TRAIN_SCALARS) | set(_TRAIN_RANGES) | _TRAIN_JOINT_TARGET_KEYS
    | _TRAIN_BOOLS
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


def _check_joint_target(name: str, v: Any, errors: list[str]) -> None:
    """`reset_joint_pos_target`: a non-empty `[q_0, ..., q_{J-1}]` vector
    of finite radians. J is robot-specific and unknown at this layer —
    only per-element bounds and a generous max-length safety rail are
    enforced here; the adapter validates J against the robot's actual
    joint count (or resolves per-name) at apply time and raises a clear
    error on mismatch rather than silently misassigning joints."""
    if (not isinstance(v, (list, tuple)) or len(v) == 0
            or len(v) > _JOINT_TARGET_MAX_LEN
            or not all(_is_num(x) for x in v)):
        errors.append(
            f"{name}: must be a non-empty list of <= "
            f"{_JOINT_TARGET_MAX_LEN} finite numbers, got {v!r}")
        return
    lo, hi = _JOINT_TARGET_ELEMENT_BOUNDS
    for x in v:
        if not (lo <= float(x) <= hi):
            errors.append(
                f"{name}: element {x} outside hard bounds [{lo}, {hi}]")


def _check_joint_trajectory(name: str, v: Any,
                            elem_bounds: tuple[float, float],
                            errors: list[str]) -> None:
    """`reset_joint_pos_trajectory` / `reset_joint_vel_trajectory`: a `[K][J]`
    matrix — K downsampled reference frames, each a per-joint vector — for
    DeepMimic phase RSI (§reset_joints_to_reference samples a random frame per
    env). J is validated against the robot at apply time (same as the single
    target); here only frame-count, rectangularity, and per-element bounds are
    checked."""
    if (not isinstance(v, (list, tuple)) or not (1 <= len(v) <= _TRAJ_MAX_FRAMES)):
        errors.append(
            f"{name}: must be a list of 1..{_TRAJ_MAX_FRAMES} frames, got {v!r}")
        return
    lo, hi = elem_bounds
    width: int | None = None
    for i, frame in enumerate(v):
        if (not isinstance(frame, (list, tuple)) or len(frame) == 0
                or len(frame) > _JOINT_TARGET_MAX_LEN
                or not all(_is_num(x) for x in frame)):
            errors.append(
                f"{name}[{i}]: must be a non-empty <= {_JOINT_TARGET_MAX_LEN}"
                f"-vector of finite numbers, got {frame!r}")
            return
        if width is None:
            width = len(frame)
        elif len(frame) != width:
            errors.append(
                f"{name}: ragged — frame {i} has {len(frame)} joints, "
                f"expected {width}")
            return
        for x in frame:
            if not (lo <= float(x) <= hi):
                errors.append(
                    f"{name}[{i}]: element {x} outside bounds [{lo}, {hi}]")
                return


def _check_push(name: str, v: Any, errors: list[str]) -> None:
    if not isinstance(v, dict):
        errors.append(f"{name}: must be an object, got {v!r}")
        return
    unknown = set(v) - _PUSH_KEYS
    if unknown:
        errors.append(f"{name}: unknown keys {sorted(unknown, key=str)}")
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
        errors.append(f"unknown top-level keys {sorted(unknown, key=str)}")
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
            f"shared: unknown keys {sorted(unknown, key=str)} "
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
            f"train: unknown keys {sorted(unknown, key=str)} "
            f"(allowed: {sorted(_TRAIN_KEYS)})")
    for k, bounds in _TRAIN_SCALARS.items():
        if k in train:
            _check_scalar(f"train.{k}", train[k], bounds, errors)
    for k, envelope in _TRAIN_RANGES.items():
        if k in train:
            _check_range(f"train.{k}", train[k], envelope, errors)
    for k in _TRAIN_JOINT_TARGET_KEYS:
        if k in train:
            _check_joint_target(f"train.{k}", train[k], errors)
    for k in _TRAIN_TRAJECTORY_KEYS:
        if k in train:
            bounds = (_JOINT_VEL_ELEMENT_BOUNDS if "vel" in k
                      else _JOINT_TARGET_ELEMENT_BOUNDS)
            _check_joint_trajectory(f"train.{k}", train[k], bounds, errors)
    # §RSI trajectory pairing: the reset event samples ONE frame index from the
    # position trajectory and indexes the VELOCITY trajectory with it, so the two
    # must share frame count K and joint width J — and a velocity trajectory with
    # no position trajectory is meaningless (would be silently ignored). Reject
    # both here so a reset-time crash / silent no-op becomes a clear schema error.
    pos_t = train.get("reset_joint_pos_trajectory")
    vel_t = train.get("reset_joint_vel_trajectory")
    if vel_t is not None:
        if pos_t is None:
            errors.append(
                "train.reset_joint_vel_trajectory: requires "
                "reset_joint_pos_trajectory (velocity RSI needs the paired "
                "position frames)")
        elif (isinstance(pos_t, (list, tuple))
              and isinstance(vel_t, (list, tuple))):
            if len(pos_t) != len(vel_t):
                errors.append(
                    f"train: reset_joint_vel_trajectory has {len(vel_t)} "
                    f"frames but reset_joint_pos_trajectory has {len(pos_t)}")
            elif (pos_t and vel_t and isinstance(pos_t[0], (list, tuple))
                  and isinstance(vel_t[0], (list, tuple))
                  and len(pos_t[0]) != len(vel_t[0])):
                errors.append(
                    f"train: reset_joint_vel_trajectory joint width "
                    f"{len(vel_t[0])} != reset_joint_pos_trajectory width "
                    f"{len(pos_t[0])}")
    for k in _TRAIN_BOOLS:
        if k in train and not isinstance(train[k], bool):
            errors.append(f"train.{k}: must be a boolean, got {train[k]!r}")
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
        except (TypeError, ValueError, LookupError):
            # LookupError covers KeyError too — a dict-valued "range"
            # (e.g. {"lo":0,"hi":1} from an LLM edit) must land in the
            # error list via the range checks above, not crash here.
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


def apply_env_edits(env_dir: Path, edits: list, *,
                    author: str = "diagnoser") -> dict:
    """§env generalization 3/4: apply proposed TRAIN-section edits to the
    project's active env spec — the environment counterpart of reward
    apply_edits, with the same validation discipline.

    `author` is recorded as the new version's `meta.source`. It defaults
    to the sculpt loop's diagnoser, which is where these edits originated;
    a hand edit from the UI passes its own so the version history says who
    actually made the change rather than crediting the loop for it.

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
    applied_rationales: list[str] = []
    # Order-independence for the RSI↔early-termination pairing: apply
    # min_base_height_termination_m FIRST so a batch proposing both the
    # sunk key and RSI ranges succeeds regardless of emission order
    # (validated after EACH edit, so RSI-before-sunk would otherwise
    # reject the RSI half of a correct pair).
    edits = sorted(
        edits,
        key=lambda e: 0 if str(_field(e, "parameter") or "")
        == "min_base_height_termination_m" else 1)
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
        train_now = work.get("train") or {}
        if (param in train_now and train_now[param] == value
                and isinstance(value, bool)
                == isinstance(train_now[param], bool)):
            # A no-op must not burn a version number — and the diagnoser
            # should hear that its edit changed nothing.
            result["rejected"].append(
                (param, f"no change — already {json.dumps(value)}"))
            continue
        prev_train = json.loads(json.dumps(train_now))
        work.setdefault("train", {})[param] = value
        errors = validate_env_spec(work)
        if errors:
            work["train"] = prev_train   # revert just this edit
            result["rejected"].append((param, "; ".join(errors)))
            continue
        result["applied"].append(f"{param}={json.dumps(value)}")
        rationale = " ".join(str(_field(e, "rationale") or "").split())[:200]
        if rationale:
            applied_rationales.append(rationale)
        changed = True

    if changed:
        meta = work.setdefault("meta", {})
        meta["source"] = author
        meta["parent"] = (spec.get("meta") or {}).get("version")
        meta["rationale"] = " | ".join(applied_rationales)[:1000]
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
