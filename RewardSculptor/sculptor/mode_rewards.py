"""sculptor/mode_rewards.py — per-mode reward authoring over a ModeGraph.

`sculptor.modes` writes down the hybrid automaton (OGMP, arXiv 2403.04205;
docs/RESEARCH_DIRECTION.md §4) but stops short of authoring reward code — its
own docstring says so. This module is that missing half: it turns a validated
`ModeGraph` into a reward module in which **each mode owns its own terms, paid
only inside its own window**.

Why gating matters more than it sounds
--------------------------------------
The failure this fixes is not hypothetical. A single scalar reward summed over
a whole episode makes every term compete with every other term for the same
policy: a "keep the torso upright" term authored for the landing phase is still
being paid during a flight phase where the reference is deliberately pitched
over, so the policy is punished for tracking the motion it was asked to track.
Measured on this repo's own Tier-D path, an unscoped task reward left the
policy reproducing **28% of the reference's joint amplitude** — it had found
that standing still scored better than moving. Scoping the reward to hardware
safety alone took that to **85%**. Per-mode gating is the same fix applied one
level finer: within a single behavior rather than across the task.

Why a scaffold instead of asking for one module
-----------------------------------------------
The obvious approach — prompt an LLM for a reward that handles all the modes —
puts the gating logic itself inside generated code, where it is unverifiable
and silently wrong when the phase clock is off. Both real Tier-D failures in
this repo were clock bugs, not reward bugs (a phase clock driven by the
TRAINING BUDGET, then a control rate assumed rather than read). So the clock
and the gating are generated *here*, deterministically, from the graph; the LLM
only fills one function body per mode. That keeps the existing authoring path —
`sculptor.edit.apply_prompt_edit`, its KG grounding, its repair retries, and
the objective-metric gauntlet — working unchanged, one mode at a time.

The scaffold also emits `MODE_WINDOWS_S` and per-mode reward components, which
is what a per-mode metric needs to score a mode's own slice of a rollout
instead of averaging its degeneracy away across the episode.

Wall time, not step counts
--------------------------
Windows are seconds and the clock reads `info["step_dt"]`, published per-step
by the mjlab runner. `EPISODE_LEN_STEPS`-style step math has to assume a
control rate at BUILD time, and that assumption has been wrong twice here. See
`sculptor.refs.timing` for the rates and why they are stated rather than
inferred.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Mapping, Optional, Sequence

from sculptor.edit import SYSTEM_PROMPT_MAX_CHARS as _SYSTEM_PROMPT_MAX_CHARS
from sculptor.modes import Mode, ModeGraph, ModeError, validate_mode_graph

#: Identifier-safe form of a mode name, for the generated function names.
_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")

#: Emitted per-mode function prefix. Kept explicit so `validate_mode_reward_source`
#: and the per-mode metric layer agree on one naming convention.
MODE_FN_PREFIX = "_mode_"

#: Component key a mode's contribution is reported under. Per-mode metrics key
#: off this, so it is part of the contract rather than a formatting detail.
MODE_COMPONENT_PREFIX = "mode_"

#: Hard ceiling on an authoring prompt. This is a SYSTEM-composed prompt, not
#: the Rewards-tab text box, so it budgets against `SYSTEM_PROMPT_MAX_CHARS`
#: rather than the human bound — a mission brief naming the course and the
#: readable goal channels does not fit in 2000 chars, and authoring a mode
#: blind to the mission is what produced four modes of terms that a robot
#: standing in front of the platforms maximized. Stated here so
#: `mode_authoring_prompt` can budget for it instead of discovering it after
#: the KG query has already run. Imported rather than restated: two copies of a
#: limit is how the last one drifted.
MAX_PROMPT_CHARS = _SYSTEM_PROMPT_MAX_CHARS

#: Suffix of a mode's batched twin. mjlab dispatches to `compute_reward_batched`
#: and treats its absence as a contract violation (`adapters/mjlab.py:670`), so
#: the batched half is not optional for the Tier-D path — it IS the training
#: path. See `sculpt.py:3540`.
BATCHED_FN_SUFFIX = "_batched"


def mode_ident(name: str) -> str:
    """Function-name-safe form of a mode name.

    Mode names come from a composed reference's provenance and are free text
    ("running approach", "one-leg kick"). Collisions after sanitizing would
    silently make two modes share a function body, so `validate_mode_graph`'s
    uniqueness check is re-applied on the SANITIZED names in
    `generate_mode_reward_scaffold` rather than assumed.
    """
    ident = _IDENT_RE.sub("_", str(name)).strip("_").lower()
    if not ident:
        raise ModeError(f"mode name {name!r} has no identifier-safe characters")
    if ident[0].isdigit():
        ident = f"m_{ident}"
    return ident


def mode_windows_s(graph: ModeGraph) -> dict[str, tuple[float, float]]:
    """`{mode_name: (start_s, end_s)}` — the window each mode's terms are paid
    over. Thin wrapper over `modes.mode_phase_windows` so this module has one
    stated source for the windows it bakes into generated code."""
    from sculptor.modes import mode_phase_windows

    return mode_phase_windows(graph)


def scale_windows(
    windows: Mapping[str, tuple[float, float]], time_scale: float,
) -> dict[str, tuple[float, float]]:
    """Stretch clip-time mode windows onto the episode's clock.

    `mode_phase_windows` returns CLIP seconds — literally `frame / clip_fps`.
    That number knows nothing about how long the episode runs or how far the
    course reaches, and for an authored world the two disagree badly.

    Measured on platform-ascent-showcase: a 6.92 s composite gating a 20 s
    episode. `active_mode` clamps everything past the last window into the
    terminal mode, so 75% of every episode was paid as `settle` — a mode whose
    terms reward stillness — while the robot was still standing on flat ground
    in front of the first box. Worse, the entry window (`approach`, 0.82 s)
    demanded 1.31 m of travel, which needs 1.60 m/s against a runtime command
    cap of 1.0 m/s: unreachable inside its own window, by construction. Across
    three sculpt iterations the policy never mounted the platform, and every
    `bound.*`/`settle.*` term read 0.00 because the robot was never in the
    state a mode expected at the time that mode expected it.

    Standing still was the reward-optimal policy. Scaling the automaton to span
    the episode removes both failures: the entry window becomes long enough to
    physically reach the first waypoint, and there is no clamp region left for
    the terminal mode to swallow.
    """
    if time_scale <= 0.0 or abs(time_scale - 1.0) < 1e-12:
        return {n: (float(lo), float(hi)) for n, (lo, hi) in windows.items()}
    return {
        n: (round(float(lo) * time_scale, 4), round(float(hi) * time_scale, 4))
        for n, (lo, hi) in windows.items()
    }


def clip_time_scale(
    windows: Mapping[str, tuple[float, float]], horizon_s: Optional[float],
) -> float:
    """`horizon_s / clip_duration`, or 1.0 when there is no horizon to fit.

    1.0 keeps the historical clip-time behaviour byte-for-byte, so a caller
    that cannot determine an episode horizon (no authored world) is unchanged.
    """
    if not horizon_s or horizon_s <= 0.0:
        return 1.0
    clip_duration_s = max((float(hi) for _, hi in windows.values()),
                          default=0.0)
    if clip_duration_s <= 0.0:
        return 1.0
    return float(horizon_s) / clip_duration_s


def _format_windows_literal(windows: Mapping[str, tuple[float, float]],
                            idents: Mapping[str, str]) -> str:
    rows = [
        f"    {name!r}: ({lo!r}, {hi!r}),  # {MODE_FN_PREFIX}{idents[name]}"
        for name, (lo, hi) in windows.items()
    ]
    return "{\n" + "\n".join(rows) + "\n}"


def _stub_body(mode: Mode, goal: str) -> str:
    """The placeholder a mode's function starts as.

    Returns 0.0 rather than a plausible-looking guess: an unauthored mode must
    be visibly unauthored. `validate_mode_reward_source` reports which modes
    are still stubs, so a half-authored graph cannot reach training looking
    complete.
    """
    goal_line = goal.strip().replace("\n", " ") or "(no goal text supplied)"
    return (
        f'    """{mode.name}: {goal_line}\n\n'
        f"    Frames [{mode.frame_range[0]}, {mode.frame_range[1]}) of the "
        f"reference.\n"
        f"    UNAUTHORED STUB — returns no credit. Author this body via\n"
        f"    `sculptor.edit.apply_prompt_edit` against this file.\n"
        f'    """\n'
        f"    del state, action, next_state, info\n"
        f"    return 0.0, {{}}\n"
    )


def _stub_body_batched(mode: Mode) -> str:
    """The placeholder a mode's BATCHED function starts as.

    Returns `like` — the caller's zero tensor — so an unauthored mode pays
    nothing at the right shape and dtype without importing torch. Same
    `UNAUTHORED STUB` marker as the scalar half, because a mode whose scalar
    body is authored and whose batched body is not would evaluate correctly in
    replay and pay zero in training: the exact silent failure this marker
    exists to make loud.
    """
    return (
        f'    """{mode.name} (batched twin of '
        f"{MODE_FN_PREFIX}{mode_ident(mode.name)}).\n\n"
        f"    `like` is a zeros tensor of shape (num_envs,) — build terms from\n"
        f"    it so device and dtype follow the env rather than being assumed.\n"
        f"    UNAUTHORED STUB — returns no credit.\n"
        f'    """\n'
        f"    del state, action, next_state, info\n"
        f"    return like, {{}}\n"
    )


def _tracking_block(clip: Mapping[str, Any], n_phase: int,
                    placeholder: bool = False,
                    time_scale: float = 1.0,
                    tracking_weight: float = 1.0) -> tuple[str, str, str]:
    """`(constants, scalar_fns, batched_fns)` for the reference-tracking
    backbone, or three empty strings when `clip` is None.

    This is the same two-Gaussian-plus-orientation formula
    `sculptor.refs.track` emits, deliberately reproduced rather than improved
    on: it is the version that took a Tier-D rollout from 28% of the
    reference's joint amplitude to 85%, and a per-mode reward is a worse place
    to try a new tracking formulation than a flat one.

    The tracking clock stays GLOBAL — phase over the whole composite — while
    the authored terms are mode-gated. Re-anchoring the phase at each mode
    boundary would be a different (and arguably better) design, but it would
    also mean the backbone is no longer the thing that has been measured, so
    it is left for after a per-mode clip certifies.

    `placeholder=True` emits the same code with the phase tables replaced by
    `np.zeros(...)` — see `authoring_twin_source` for why that exists and why
    the fake numbers are harmless.

    `tracking_weight` scales the whole backbone. It exists because the backbone
    is three exp kernels summed unweighted, so it pays up to 3.0/step, and all
    three are near-maximal for a robot standing upright at nominal height in a
    neutral pose. Measured on the platform-ascent-showcase run: the backbone
    paid 2.28/step while every authored mode term together paid 0.34 — a 7:1
    ratio in favour of the style prior. Standing still scored 2.62/step against
    roughly 2.5 for actually running the course, so the optimal policy was to
    stand in front of the first platform, which is exactly what it did for
    three iterations. On a task-grounded project the imitation clip is a style
    prior and the mission is the objective, so the caller passes a weight that
    caps the backbone at one mode's worth of reward. 1.0 (the default) leaves
    pure-imitation projects byte-for-byte as they were.
    """
    import numpy as np

    from sculptor.refs.track import (JOINT_ERR_WEIGHT, ORIENTATION_ERR_WEIGHT,
                                     ROOT_ERR_WEIGHT, _format_array_literal,
                                     downsample_phase_targets,
                                     projected_gravity_from_quat)

    joint_names = list(clip.get("joint_names") or [])
    if not joint_names or clip.get("joint_pos") is None:
        raise ModeError(
            "clip has no joint_pos/joint_names — a tracking backbone needs a "
            "per-joint target. Scaffold without `clip` for stubs only.")

    fps = float(clip.get("fps") or 0.0) or 30.0
    n_frames = int(np.asarray(clip["root_pos_z"]).shape[0])
    # Scaled by the SAME factor as the mode windows. If only one of the two
    # were stretched, the mode gate and the phase clock would disagree and the
    # backbone would be tracking a different instant than the mode being paid.
    duration_s = (n_frames / fps if fps > 0 else 0.0) * float(time_scale)
    joint_pos = downsample_phase_targets(
        np.asarray(clip["joint_pos"], dtype=np.float64), n=n_phase)
    root_z = downsample_phase_targets(
        np.asarray(clip["root_pos_z"], dtype=np.float64), n=n_phase)

    quat = clip.get("root_quat_wxyz")
    gravity_literal = "None"
    orientation_weight = 0.0
    if quat is not None:
        gravity = downsample_phase_targets(
            projected_gravity_from_quat(np.asarray(quat, dtype=np.float64)),
            n=n_phase)
        norm = np.linalg.norm(gravity, axis=1, keepdims=True)
        gravity = gravity / np.where(norm > 0.0, norm, 1.0)
        gravity_literal = (
            f"np.asarray({_format_array_literal(gravity)}, dtype=np.float64)"
            f".reshape(N_PHASE, 3)")
        orientation_weight = ORIENTATION_ERR_WEIGHT

    joint_literal = (f"np.asarray({_format_array_literal(joint_pos)}, "
                     f"dtype=np.float64).reshape(N_PHASE, N_JOINTS)")
    root_literal = (f"np.asarray({_format_array_literal(root_z)}, "
                    f"dtype=np.float64)")
    banner = ("Paid in EVERY mode. The automaton decides which TASK terms "
              "apply; it does\n# not change what the robot is supposed to be "
              "tracking, so the backbone is\n# mode-independent by "
              "construction. Same formula as `sculptor.refs.track`.")
    if placeholder:
        joint_literal = "np.zeros((N_PHASE, N_JOINTS), dtype=np.float64)"
        root_literal = "np.zeros(N_PHASE, dtype=np.float64)"
        if quat is not None:
            gravity_literal = (
                "np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), "
                "(N_PHASE, 1))")
        banner = ("PLACEHOLDER TARGETS — this module is an authoring "
                  "scaffold, not a\n# trainable reward. The real reference "
                  "tables live in the module these\n# authored bodies are "
                  "grafted into; only the mode functions are copied\n# "
                  "across, so these values are never used to train anything.")

    constants = f'''import numpy as np

# ── reference-tracking backbone ────────────────────────────────────────
# {banner}
JOINT_NAMES = {joint_names!r}
N_JOINTS = {len(joint_names)}
N_PHASE = {n_phase}
REFERENCE_DURATION_S = {duration_s!r}
JOINT_ERR_WEIGHT = {JOINT_ERR_WEIGHT!r}
ROOT_ERR_WEIGHT = {ROOT_ERR_WEIGHT!r}
# 0.0 when the clip carries no root orientation, making the term an exact
# no-op rather than a silently-wrong upright target.
ORIENTATION_ERR_WEIGHT = {orientation_weight!r}
# Scales the WHOLE backbone, components included, so what is reported is what
# is paid. The three kernels are each <= 1 and each is near-maximal for a robot
# standing upright at nominal height, so at 1.0 the backbone pays ~2.3/step for
# doing nothing — more than any authored mode could earn for doing the task.
TRACKING_W = {tracking_weight!r}

TARGET_JOINT_POS = {joint_literal}
TARGET_ROOT_Z = {root_literal}
TARGET_GRAVITY = {gravity_literal}
'''

    scalar = '''
def _phase_index(info) -> int:
    """Where in the WHOLE composite we are — not where in the mode.

    The tracking target is the reference's own frame at this instant, and the
    reference does not restart per mode.
    """
    t = _elapsed_s(info)
    phase = (t / REFERENCE_DURATION_S) if REFERENCE_DURATION_S > 0.0 else 0.0
    return int(min(max(phase, 0.0), 0.999999) * N_PHASE)


def _tracking(next_state, info):
    """(value, components) for the reference-tracking backbone, scalar path.

    Handles BOTH qpos layouts, because both really occur:

      * a full MuJoCo vector (7 free-joint DOFs + N actuated), from replay
        against a raw MuJoCo state;
      * an mjlab per-env row of just the N actuated joints — which is what the
        G1 task's reward contract declares (`qpos: (29,)`), and what
        `sculptor.edit`'s pre-flight probe builds. Slicing `qpos[7:7+N]` here
        raised on that contract.

    Taking the TRAILING N_JOINTS is correct for both (they coincide when the
    vector is exactly 7+N long) and matches the batched path, so the two cannot
    drift apart on the joint term.
    """
    qpos = np.asarray(next_state["qpos"], dtype=np.float64).reshape(-1)
    if qpos.shape[0] < N_JOINTS:
        raise ValueError(
            f"qpos has {qpos.shape[0]} entries, fewer than the {N_JOINTS} "
            f"tracked joints")
    i = _phase_index(info)
    joint_err = qpos[-N_JOINTS:] - TARGET_JOINT_POS[i]
    joint_term = float(np.exp(-JOINT_ERR_WEIGHT * np.mean(joint_err ** 2)))
    # Root height: prefer what the runner publishes, since mjlab's
    # `base_height_delta` is measured from the ROBOT's own episode start and is
    # the only anchor that means anything for origin-relative retargeted clips.
    # Fall back to absolute `qpos[2]` only when qpos really is a full MuJoCo
    # vector; a joints-only row has no root at index 2 to read.
    root0 = float(TARGET_ROOT_Z[0])
    if "base_height_delta" in info or "base_height" in info:
        actual_delta = float(info.get(
            "base_height_delta", float(info.get("base_height", root0)) - root0))
        root_err = actual_delta - (float(TARGET_ROOT_Z[i]) - root0)
    elif qpos.shape[0] >= 7 + N_JOINTS:
        root_err = float(qpos[2]) - float(TARGET_ROOT_Z[i])
    else:
        root_err = 0.0
    root_term = float(np.exp(-ROOT_ERR_WEIGHT * root_err ** 2))
    components = {"joint_tracking": joint_term, "root_tracking": root_term}
    if TARGET_GRAVITY is not None:
        gravity = np.asarray(
            next_state["projected_gravity_b"], dtype=np.float64).reshape(-1)[-3:]
        components["orientation_tracking"] = float(
            np.exp(-ORIENTATION_ERR_WEIGHT * np.mean(
                (gravity - TARGET_GRAVITY[i]) ** 2)))
    # Weighted before anything is returned, so the reported components are the
    # amounts actually added to the reward. Scaling only the total would leave
    # the diagnosis loop comparing an unpaid backbone against paid mode terms.
    components = {k: v * TRACKING_W for k, v in components.items()}
    return float(sum(components.values())), components

'''

    batched = '''
def _tracking_batched(next_state, info, like):
    """The training path's backbone.

    Joints are the TRAILING N_JOINTS of a per-env `qpos` row — the G1 task's
    contract declares `qpos: (29,)`, actuated joints only. Root height arrives
    as `info["base_height"]` and is compared as a DELTA from the reference's
    own first frame: retargeting zeroes root translation, so the clip's root_z
    sits near 0 while a standing G1 base is ~0.74 m, and an absolute comparison
    would saturate the kernel at zero for every frame no matter how well the
    motion tracked. `_tracking` follows both conventions, so the two paths
    agree wherever they can.
    """
    import torch

    qpos = next_state["qpos"]
    if qpos.shape[-1] < N_JOINTS:
        raise ValueError(
            f"batched qpos has {qpos.shape[-1]} columns, fewer than the "
            f"{N_JOINTS} tracked joints")
    t = _elapsed_s_batched(like, info)
    phase = torch.clamp(t / REFERENCE_DURATION_S, 0.0, 0.999999) \\
        if REFERENCE_DURATION_S > 0.0 else torch.zeros_like(like)
    i = torch.clamp((phase * N_PHASE).long(), 0, N_PHASE - 1)

    target_joint = torch.as_tensor(
        TARGET_JOINT_POS, device=qpos.device, dtype=qpos.dtype)[i]
    joint_err = qpos[:, -N_JOINTS:] - target_joint
    joint_term = torch.exp(
        -JOINT_ERR_WEIGHT * torch.mean(joint_err ** 2, dim=-1))

    target_root = torch.as_tensor(
        TARGET_ROOT_Z, device=qpos.device, dtype=qpos.dtype)[i]
    root0 = float(TARGET_ROOT_Z[0])
    base_height = info.get("base_height", torch.zeros_like(like))
    actual_delta = info.get("base_height_delta", base_height - root0)
    root_term = torch.exp(-ROOT_ERR_WEIGHT * (actual_delta - (target_root - root0)) ** 2)

    components = {"joint_tracking": joint_term, "root_tracking": root_term}
    if TARGET_GRAVITY is not None:
        target_gravity = torch.as_tensor(
            TARGET_GRAVITY, device=qpos.device, dtype=qpos.dtype)[i]
        gravity = next_state["projected_gravity_b"][:, -3:]
        components["orientation_tracking"] = torch.exp(
            -ORIENTATION_ERR_WEIGHT * torch.mean(
                (gravity - target_gravity) ** 2, dim=-1))
    # Same weighting as the scalar path, applied the same way, so replay and
    # training cannot disagree about how much the backbone is worth.
    components = {k: v * TRACKING_W for k, v in components.items()}
    total = torch.zeros_like(like)
    for v in components.values():
        total = total + v
    return total, components

'''
    return constants, scalar, batched


def generate_mode_reward_scaffold(
    graph: ModeGraph,
    *,
    behavior_goal: str = "",
    goal_by_mode: Optional[Mapping[str, str]] = None,
    clip_id: str = "",
    clip: Optional[Mapping[str, Any]] = None,
    n_phase: int = 32,
    placeholder_targets: bool = False,
    horizon_s: Optional[float] = None,
    tracking_weight: float = 1.0,
) -> str:
    """Emit a reward module whose mode gating is correct by construction.

    Each mode gets a `_mode_<ident>(state, action, next_state, info)` returning
    `(float, dict)` and a `_mode_<ident>_batched(..., like)` returning
    `(tensor, dict)`. `compute_reward` pays ONLY the mode whose window contains
    the current wall-clock time; `compute_reward_batched` does the same PER ENV,
    which matters because mjlab's envs reset independently and are therefore in
    different modes at the same step. Both report `mode_<ident>` plus
    `active_mode_index`, so a per-mode metric can slice a rollout by mode
    without re-deriving the automaton.

    Pass `clip` (a loaded reference, `sculptor.reference.load_clip`) to include
    the reference-tracking backbone. Without it every mode is a stub paying
    zero, so the module only becomes trainable once every mode is authored —
    and even then it has nothing telling the policy to follow the reference.
    With it the scaffold is trainable IMMEDIATELY (it is the tracking reward
    that already works), and authoring adds mode-specific task terms on top.
    That layering is OGMP's own shape: one oracle to track throughout, a
    per-mode objective on top of it.

    Raises `ModeError` when the graph is structurally invalid, when two mode
    names collide once sanitized to identifiers, or when `clip` carries no
    per-joint target to track.
    """
    errors = validate_mode_graph(graph)
    if errors:
        raise ModeError("; ".join(errors))

    idents: dict[str, str] = {}
    for m in graph.modes:
        ident = mode_ident(m.name)
        if ident in idents.values():
            clash = next(k for k, v in idents.items() if v == ident)
            raise ModeError(
                f"modes {clash!r} and {m.name!r} both sanitize to {ident!r}; "
                "rename one — two modes sharing a function body is silent")
        idents[m.name] = ident

    # `horizon_s` is the episode the automaton has to cover. Absent it the
    # windows stay in clip time, which is only right when the episode happens
    # to be the clip's length — see `scale_windows` for what that cost live.
    windows = mode_windows_s(graph)
    time_scale = clip_time_scale(windows, horizon_s)
    windows = scale_windows(windows, time_scale)
    # A horizon we could not use is not a horizon. Normalize so 0.0/negative
    # record identically to "no authored world" instead of leaving a number in
    # the spec that no window was ever fitted to.
    horizon_s = float(horizon_s) if horizon_s and horizon_s > 0.0 else None
    goals = dict(goal_by_mode or {})
    order = [m.name for m in graph.modes]

    fns = "\n\n".join(
        f"def {MODE_FN_PREFIX}{idents[m.name]}(state, action, next_state, info):\n"
        + _stub_body(m, goals.get(m.name, ""))
        + f"\n\ndef {MODE_FN_PREFIX}{idents[m.name]}{BATCHED_FN_SUFFIX}"
          "(state, action, next_state, info, like):\n"
        + _stub_body_batched(m)
        for m in graph.modes
    )
    dispatch = ",\n".join(
        f"    {name!r}: {MODE_FN_PREFIX}{idents[name]}" for name in order)
    dispatch_b = ",\n".join(
        f"    {name!r}: {MODE_FN_PREFIX}{idents[name]}{BATCHED_FN_SUFFIX}"
        for name in order)

    has_track = clip is not None
    track_const, track_scalar, track_batched = (
        _tracking_block(clip, int(n_phase), bool(placeholder_targets),
                        time_scale=time_scale,
                        tracking_weight=float(tracking_weight))
        if has_track else ("", "", ""))
    # The calls are spliced rather than always emitted-and-zeroed: a stubs-only
    # scaffold must not require `qpos` in the contract, since it does not read
    # the state at all.
    track_call = ("    track_value, track_components = _tracking(next_state, info)\n"
                  "    value = value + track_value\n"
                  "    out.update(track_components)\n") if has_track else ""
    track_call_b = (
        "    track_total, track_components = _tracking_batched(\n"
        "        next_state, info, like)\n"
        "    total = total + track_total\n"
        "    components.update(track_components)\n") if has_track else ""

    return f'''"""Auto-generated per-mode reward scaffold{f" for clip {clip_id!r}" if clip_id else ""}.

Generated by `sculptor.mode_rewards.generate_mode_reward_scaffold` from a
validated `ModeGraph`. The mode gating below is DERIVED, not authored — both
Tier-D failures in this repo were phase-clock bugs, so the clock is not left to
generated code. Author the `{MODE_FN_PREFIX}*` bodies — TWO per mode, a scalar
one and a `{BATCHED_FN_SUFFIX}` twin, because mjlab only ever calls the batched
path — and leave the dispatch alone; regenerate from the graph rather than
hand-editing windows.

Behavior goal: {behavior_goal.strip() or "(unspecified)"}
"""
from __future__ import annotations

REWARD_SPEC: dict = {{
    "version": "mode-reward-v1",
    "description": "Per-mode reward over a {len(graph.modes)}-mode automaton.",
    "author": "sculptor",
    "parent_hash": None,
    "supports_batched": True,
    # Consumed by the per-mode metric layer: it slices a rollout by these
    # windows instead of scoring the episode as one undifferentiated blob.
    "mode_windows_s": {{{", ".join(f"{n!r}: {list(w)}" for n, w in windows.items())}}},
    # Which composed clip this automaton came from. Load-bearing after
    # promotion: `promote_mode_reward` copies this module into the version
    # chain, and without the id the UI could not re-open the per-mode panel on
    # its own reward — it had to make the user search the clip library for it
    # by hand, twice.
    "reference_clip_id": {clip_id!r},
    # The episode the automaton was stretched to cover, and by how much. 1.0
    # means the windows are raw clip seconds — correct only when the episode
    # IS the clip's length. Recorded so a reader can tell which clock a
    # promoted reward is on without re-deriving it from the windows.
    "episode_horizon_s": {horizon_s!r},
    "clip_time_scale": {round(time_scale, 6)!r},
    # How much the imitation backbone is worth relative to the authored task
    # terms. Below 1.0 the clip is a style prior and the mission is the
    # objective; at 1.0 it is the other way round.
    "tracking_weight": {float(tracking_weight)!r},
    "hyperparameters": {{}},
    "references": [],
}}

#: Mode name -> (start_s, end_s), half-open. Seconds because the clock reads
#: `info["step_dt"]`; a step count would have to assume a control rate at build
#: time, which is exactly the assumption that broke Tier-D twice.
MODE_WINDOWS_S: dict = {_format_windows_literal(windows, idents)}

#: Authoring order — also the order `active_mode_index` refers to.
MODE_ORDER: list = {order!r}

REFERENCE_FPS = {graph.fps!r}

#: Only used when the runner publishes no `step_dt`. The mjlab G1 tasks run
#: 0.005 s physics with decimation 4 (see `sculptor.refs.timing`), so this is
#: the real control rate rather than a round number.
DEFAULT_STEP_DT = 0.02

{track_const}

def _elapsed_s(info) -> float:
    """Wall time into the episode. `step_dt` is published per-step by the mjlab
    runner; falling back to a fixed rate would silently mis-gate every mode, so
    the fallback is the G1 task's real 0.02 s (50 Hz) rather than a guess."""
    step = float(info.get("episode_length", 0) or 0)
    step_dt = float(info.get("step_dt", 0.0) or 0.0) or DEFAULT_STEP_DT
    return step * step_dt


def active_mode(info) -> str:
    """Which mode owns the current instant.

    Time past the terminal window stays in the terminal mode: an episode
    running long is still *in* the last mode, not outside the automaton
    (matches `sculptor.modes.mode_at_frame`). Before the first window it is the
    entry mode, so there is never an instant with no owner.
    """
    t = _elapsed_s(info)
    for name in MODE_ORDER:
        lo, hi = MODE_WINDOWS_S[name]
        if lo <= t < hi:
            return name
    return MODE_ORDER[-1] if t >= MODE_WINDOWS_S[MODE_ORDER[-1]][0] \\
        else MODE_ORDER[0]


def _batch_like(action, next_state, info):
    """A zeros tensor of shape (num_envs,) to build every batched term from.

    Derived rather than assumed. `sculptor.edit`'s pre-flight probe builds
    `state`/`next_state` from the contract's state schema, so no single key is
    guaranteed present; `episode_length` is tried first because the mjlab runner
    publishes it PER ENV (`adapters/_mjlab_runner.py:649`), which is the fact
    that makes per-env mode masking meaningful at all.
    """
    import torch

    for v in (info.get("episode_length"), action,
              *tuple((next_state or {{}}).values())):
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > 0:
            dtype = v.dtype if v.is_floating_point() else torch.float32
            return torch.zeros(v.shape[0], device=v.device, dtype=dtype)
    raise ValueError(
        "compute_reward_batched cannot determine the batch size: no tensor in "
        "info['episode_length'], action, or next_state")


def _elapsed_s_batched(like, info):
    """Per-env wall time into the episode — the batched twin of `_elapsed_s`.

    Envs reset independently, so at any given step they are at DIFFERENT points
    in the automaton. That is the whole reason this is a tensor: a scalar clock
    would put every env in the same mode and pay the wrong terms to most of
    them.
    """
    import torch

    step = info.get("episode_length", None)
    if not torch.is_tensor(step):
        step = torch.as_tensor(float(step or 0.0), device=like.device,
                               dtype=like.dtype)
    dt = info.get("step_dt", None)
    if not torch.is_tensor(dt):
        dt = torch.as_tensor(float(dt or 0.0), device=like.device,
                             dtype=like.dtype)
    dt = dt.to(like.dtype)
    dt = torch.where(dt > 0.0, dt, torch.full_like(dt, DEFAULT_STEP_DT))
    # `+ like` is a broadcast to (num_envs,), not arithmetic: `like` is zeros.
    return step.to(like.dtype) * dt + like


def _mode_masks(like, info):
    """One boolean mask per mode, in MODE_ORDER; exactly one True per env.

    Deliberately mirrors `active_mode` term for term — first matching window
    wins, leftovers go to the terminal mode when past its start and to the entry
    mode otherwise. A rollout scored by the scalar path and trained by this one
    must never disagree about which mode owns an instant.
    """
    import torch

    t = _elapsed_s_batched(like, info)
    matched = torch.zeros_like(t, dtype=torch.bool)
    masks = []
    for name in MODE_ORDER:
        lo, hi = MODE_WINDOWS_S[name]
        m = (t >= lo) & (t < hi) & (~matched)
        matched = matched | m
        masks.append(m)
    leftover = ~matched
    past_end = t >= MODE_WINDOWS_S[MODE_ORDER[-1]][0]
    masks[-1] = masks[-1] | (leftover & past_end)
    masks[0] = masks[0] | (leftover & (~past_end))
    return masks

{track_scalar}{track_batched}

{fns}


_MODE_FNS = {{
{dispatch},
}}

_MODE_FNS_BATCHED = {{
{dispatch_b},
}}


def compute_reward(state, action, next_state, info):
    """Tracking backbone (if any) plus ONLY the active mode's terms.

    Terms authored for one mode are not paid during another — that scoping is
    the entire point of the automaton. An episode-level sum would let a term
    written for landing punish the policy throughout a flight phase where the
    reference is deliberately pitched over. The backbone is exempt because it
    is not a task term: what the robot should be tracking does not change with
    the mode.
    """
    name = active_mode(info)
    value, components = _MODE_FNS[name](state, action, next_state, info)
    value = float(value)
    out = {{f"{MODE_COMPONENT_PREFIX}{{name}}": value,
           "active_mode_index": float(MODE_ORDER.index(name))}}
    for k, v in (components or {{}}).items():
        out[f"{{name}}.{{k}}"] = float(v)
{track_call}    return value, out


def compute_reward_batched(state, action, next_state, info):
    """The training path. Pay each env only the mode it is currently in.

    Every mode's function is evaluated for every env and then masked, which is
    what makes this vectorizable. The masking uses `torch.where` and NOT
    `mask * value` on purpose: a mode's terms are only defined inside its own
    window, and `0.0 * nan == nan`, so a multiply would let one out-of-window
    env poison the whole batch's reward. `where` discards the unselected branch
    outright, while a nan produced INSIDE the window still surfaces — which is
    the direction you want a numerical bug to fail in.
    """
    import torch

    like = _batch_like(action, next_state, info)
    masks = _mode_masks(like, info)
    zero = torch.zeros_like(like)

    total = zero.clone()
    index = zero.clone()
    components = {{}}
    for j, name in enumerate(MODE_ORDER):
        mask = masks[j]
        value, parts = _MODE_FNS_BATCHED[name](
            state, action, next_state, info, like)
        value = torch.as_tensor(
            value, device=like.device, dtype=like.dtype) + like
        paid = torch.where(mask, value, zero)
        total = total + paid
        index = index + torch.where(mask, torch.full_like(like, float(j)), zero)
        components[f"{MODE_COMPONENT_PREFIX}{{name}}"] = paid
        for k, v in (parts or {{}}).items():
            v = torch.as_tensor(v, device=like.device, dtype=like.dtype) + like
            components[f"{{name}}.{{k}}"] = torch.where(mask, v, zero)
    components["active_mode_index"] = index
{track_call_b}    return total, components
'''


def _fn_span(source: str, fn_name: str) -> Optional[tuple[int, int]]:
    """`(start, end)` character span of `fn_name`'s whole `def` block.

    Parsed rather than pattern-matched. The graft is load-bearing — it is how
    an authored mode reaches the trainable module — and a regex has to guess
    where a function ends from blank lines, which is a property of whatever
    formatting the model happened to emit. `ast` knows.

    Falls back to a regex only when the source does not parse, so a caller
    inspecting a half-written file still gets an answer instead of an
    exception.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        m = re.search(
            rf"^def {re.escape(fn_name)}\(.*?\):\n.*?"
            rf"(?=\n+def |\n+_MODE_FNS|\Z)",
            source, re.M | re.S)
        return (m.start(), m.end()) if m else None

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            # `lineno` is 1-based and excludes decorators, which the generated
            # module never uses; `end_lineno` is inclusive.
            return offsets[node.lineno - 1], offsets[node.end_lineno]
    return None


def _module_bindings(source: str) -> dict[str, tuple[int, int]]:
    """Module-level name -> character span of the statement that binds it.

    Only top-level statements count: a name bound inside a function is that
    function's business, not the module's.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    out: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        span = (offsets[node.lineno - 1], offsets[node.end_lineno])
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            # Every name the statement binds, not just the bare-Name targets.
            # `_SETTLE_LO, _SETTLE_HI = 4.92, 6.92` is how a model naturally
            # writes a window pair, and skipping tuple targets meant
            # `_carry_helpers` did not know the module defined those names —
            # so the graft dropped the statement and the batched probe died on
            # `NameError: name '_SETTLE_HI' is not defined`.
            names = [n.id for t in node.targets
                     for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.asname or a.name.split(".")[0] for a in node.names]
        for name in names:
            out.setdefault(name, span)
    return out


def _free_names(source: str, fn_name: str) -> set[str]:
    """Names `fn_name` reads that it does not itself bind.

    Approximate on purpose — a name that is both a parameter and a global read
    would be missed, which the generated code never does. What it has to catch
    is the case that matters: a call to a helper defined elsewhere.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return set()
    bound = {a.arg for a in fn.args.args}
    bound |= {a.arg for a in fn.args.kwonlyargs}
    read: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return read - bound


def _carry_helpers(base: str, authored: str, fn_names: Sequence[str]) -> str:
    """Append helpers the authored bodies call but `base` does not define.

    A model asked for one function often writes two — the first real authoring
    run came back with the mode calling an `_info_b` it had defined at module
    level, which the graft dropped, and the module then failed the batched
    probe with `NameError: name '_info_b' is not defined`. Prohibiting helpers
    in the prompt would be one answer; carrying them is the better one, since
    a helper shared by the scalar and batched halves is exactly how you keep
    the two from drifting.

    Only names that are (a) read by a grafted body, (b) undefined in `base`
    and (c) defined at module level in `authored` are carried, so nothing here
    can overwrite the dispatch, the windows or another mode. Transitive: a
    helper that calls a helper brings it along.
    """
    import builtins

    donor = _module_bindings(authored)
    have = set(_module_bindings(base)) | set(dir(builtins))
    wanted: list[str] = []
    queue = [n for fn in fn_names for n in sorted(_free_names(base, fn))]
    while queue:
        name = queue.pop(0)
        if name in have or name in wanted or name not in donor:
            continue
        wanted.append(name)
        queue.extend(sorted(_free_names(authored, name)))
    if not wanted:
        return base
    # Emit in the donor's own order so a helper that references another still
    # reads top-down, and so the result is stable across runs. Dedupe by span:
    # one statement can bind several names (`_LO, _HI = ...`), and emitting it
    # once per name would redefine it — harmless for a constant, a silent
    # double-definition for anything else.
    wanted.sort(key=lambda n: donor[n][0])
    spans = list(dict.fromkeys(donor[n] for n in wanted))
    blocks = [authored[lo:hi].rstrip("\n") for lo, hi in spans]
    return (base.rstrip("\n") + "\n\n\n"
            + "\n\n\n".join(blocks) + "\n")


def _body_after(source: str, fn_name: str) -> Optional[str]:
    """Source of `fn_name`'s body (everything after the `def` line), or None."""
    span = _fn_span(source, fn_name)
    if span is None:
        return None
    block = source[span[0]:span[1]]
    head, sep, body = block.partition(":\n")
    return body if sep else None


def _strip_prose(source: str, keep_prefix: str = MODE_FN_PREFIX) -> str:
    """Drop comments and docstrings, except from functions named `keep_prefix*`.

    Only used on the authoring twin. `apply_prompt_edit` rewrites the whole
    module, so every byte of prose in it is prose the model has to reproduce
    out of a bounded output budget — and the first two real runs both died on
    that budget. The per-mode docstrings are kept because they are the part the
    model is being asked to act on; the explanations of how the dispatch works
    are for humans reading the real module, and the real module still has them.

    Returns `source` unchanged if it does not parse, so this can never be the
    thing that breaks an authoring run.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body.pop(0)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith(keep_prefix):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # A function whose only statement is its docstring would become an
            # empty body, which is a SyntaxError.
            node.body = body[1:] if len(body) > 1 else [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def authoring_twin_source(
    graph: ModeGraph,
    *,
    clip: Optional[Mapping[str, Any]] = None,
    behavior_goal: str = "",
    clip_id: str = "",
    n_phase: int = 4,
    horizon_s: Optional[float] = None,
    tracking_weight: float = 1.0,
) -> str:
    """A small, literal-free stand-in to run `apply_prompt_edit` against.

    Two measured problems make authoring against the real module unworkable,
    and this fixes both:

    1. `apply_prompt_edit` rewrites the WHOLE module. A backbone-carrying
       scaffold is ~27 KB of which most is a 32x29 table of float literals, and
       the model expands what it rewrites — the first real run truncated inside
       the table (`SyntaxError: '[' was never closed`). The tables here are
       `np.zeros(...)`, so there is nothing to mangle and nothing to spend
       output tokens on.
    2. `sculptor.edit` rejects a reward that returns the same value across its
       state probes — rightly, since a constant reward gives PPO no gradient.
       But a per-mode module is *deliberately* zero outside the active mode, so
       a stubs-only twin reads as constant-0 and is rejected for a property it
       is supposed to have. The placeholder backbone is state-dependent (it
       reads qpos, height and gravity), so the twin passes that gate on the
       same grounds the real module does.

    The fake numbers are harmless because only `_mode_*` functions are grafted
    back (`graft_mode_bodies`); no line written here reaches training. Pass the
    real `clip` so the twin has the right `N_JOINTS` and duration — an authored
    body that indexes joints must see the real joint count.
    """
    # `horizon_s` and `tracking_weight` must match the module the bodies are
    # grafted back into, or the twin gates on different windows — and reports a
    # different backbone magnitude — than the real thing.
    return _strip_prose(generate_mode_reward_scaffold(
        graph, behavior_goal=behavior_goal, clip_id=clip_id, clip=clip,
        n_phase=n_phase, placeholder_targets=True, horizon_s=horizon_s,
        tracking_weight=tracking_weight))


def _horizon_of(source: str) -> Optional[float]:
    """The episode horizon a scaffold was fitted to, read back off its own
    windows. Recovering it from the source keeps the twin in step with the
    module even for scaffolds written before `episode_horizon_s` was recorded.
    """
    windows = windows_in_source(source)
    if not windows:
        return None
    span = max((hi for _, hi in windows.values()), default=0.0)
    return span if span > 0.0 else None


def _tracking_weight_of(source: str) -> float:
    """`TRACKING_W` as the module declares it; 1.0 for modules predating it.

    Read off the module rather than re-derived, for the same reason as
    `_horizon_of`: the twin has to describe the module that exists on disk, not
    the one today's generator would produce.
    """
    match = re.search(r"^TRACKING_W\s*=\s*([0-9.eE+-]+)\s*$", source, re.M)
    if not match:
        return 1.0
    try:
        return float(match.group(1))
    except ValueError:
        return 1.0


def graft_mode_bodies(base: str, authored: str,
                      mode_names: Sequence[str]) -> str:
    """Copy the named modes' functions out of `authored` into `base`.

    Why this exists: `apply_prompt_edit` rewrites the WHOLE module, and a
    scaffold carrying the tracking backbone is ~27 KB of which most is a
    32x29 table of float literals. Asking a model to reproduce those verbatim
    on every edit is a corruption waiting to happen — and it happened on the
    first real authoring run, which came back
    `SyntaxError: '[' was never closed`.

    So authoring runs against a stubs-only twin (small, no literals) and the
    result is transplanted here. That is the same division of labour the rest
    of this module already draws: the deterministic parts are generated, the
    model writes function bodies and nothing else. It also means a model that
    edits the dispatch, the windows or another mode's body simply has that
    edit dropped, rather than it having to be caught downstream.

    Raises `ModeError` when a named mode is missing from either side.
    """
    out = base
    grafted: list[str] = []
    for name in mode_names:
        ident = mode_ident(name)
        for fn in (MODE_FN_PREFIX + ident,
                   MODE_FN_PREFIX + ident + BATCHED_FN_SUFFIX):
            src_span = _fn_span(authored, fn)
            dst_span = _fn_span(out, fn)
            if src_span is None:
                raise ModeError(
                    f"authored module has no {fn} — nothing to graft for mode "
                    f"{name!r}")
            if dst_span is None:
                raise ModeError(f"target module has no {fn} to replace")
            out = (out[:dst_span[0]]
                   + authored[src_span[0]:src_span[1]]
                   + out[dst_span[1]:])
            grafted.append(fn)
    return _carry_helpers(out, authored, grafted)


def _component_names(source: str, fn_name: str) -> list[str]:
    """String keys of the dict literals inside `fn_name` — its components."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return []
    names: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if k.value not in names:
                        names.append(k.value)
    return names


def summarize_authored_modes(twin: str, authored: str,
                             mode_names: Sequence[str]) -> str:
    """Note what the already-authored modes pay, without their bodies.

    Authoring is one mode per call, so by the third call the module carries two
    finished modes — and `apply_prompt_edit` makes the model reproduce all of
    it. Grafting the real neighbours in was the first attempt at telling the
    model what they own; it put the twin back over the output budget and both
    attempts at mode 'approach' came back truncated
    (`'(' was never closed`).

    What the model actually needs from a neighbour is which terms are already
    paid, so it does not pay them twice — not how they are computed. That is a
    line of component names. The summary never leaves the twin: the graft back
    into the real module copies one mode, this one.
    """
    out = twin
    for name in mode_names:
        ident = mode_ident(name)
        scalar = MODE_FN_PREFIX + ident
        batched = scalar + BATCHED_FN_SUFFIX
        comps = _component_names(authored, scalar) or ["(unnamed)"]
        listed = ", ".join(comps)
        pairs = ", ".join(f"{c!r}: 0.0" for c in comps if c != "(unnamed)")
        for fn, sig, ret in (
            (scalar, "(state, action, next_state, info)",
             f"    return 0.0, {{{pairs}}}\n"),
            (batched, "(state, action, next_state, info, like)",
             f"    return like, {{{', '.join(f'{c!r}: like' for c in comps if c != '(unnamed)')}}}\n"),
        ):
            span = _fn_span(out, fn)
            if span is None:
                continue
            block = (f"def {fn}{sig}:\n"
                     f"    \"\"\"ALREADY AUTHORED — pays {listed}. Shown so you do "
                     f"not pay these again; do not edit, it is discarded.\"\"\"\n"
                     f"    del state, action, next_state, info\n"
                     f"{ret}")
            out = out[:span[0]] + block + out[span[1]:]
    return out


def authored_modes(source: str, *, require_batched: bool = True) -> dict[str, bool]:
    """`{mode_name: is_authored}` for a scaffold's source.

    A stub is detected by its marker, not by comparing against a regenerated
    scaffold — an author may legitimately reformat a body, and re-deriving the
    scaffold to diff against it would call every such edit a stub.

    Both halves must be authored by default. A mode whose scalar body is written
    and whose batched body is still a stub *evaluates* correctly in replay and
    pays exactly zero in training, since mjlab only ever calls the batched path
    (`adapters/mjlab.py:670`) — a discrepancy that looks like a bad reward
    rather than a missing one. Pass `require_batched=False` only for adapters
    that call the scalar path (`sculpt.py:3485` decides which those are).
    """
    order_match = re.search(r"^MODE_ORDER: list = (\[.*?\])$", source, re.M | re.S)
    if not order_match:
        return {}
    try:
        names = eval(order_match.group(1), {"__builtins__": {}})  # noqa: S307
    except Exception:  # noqa: BLE001 — malformed scaffold, not a crash site
        return {}

    out: dict[str, bool] = {}
    for name in names:
        try:
            ident = mode_ident(name)
        except ModeError:
            continue
        scalar = _body_after(source, MODE_FN_PREFIX + ident)
        done = scalar is not None and "UNAUTHORED STUB" not in scalar
        if require_batched:
            batched = _body_after(
                source, MODE_FN_PREFIX + ident + BATCHED_FN_SUFFIX)
            done = done and batched is not None \
                and "UNAUTHORED STUB" not in batched
        out[name] = done
    return out


def validate_mode_reward_source(source: str, graph: ModeGraph) -> list[str]:
    """Every structural problem at once, mirroring `validate_mode_graph`.

    Checks that the module still matches the automaton it was generated from —
    a graph that gained or renamed a mode after the scaffold was written would
    otherwise train with a mode whose terms are never paid, which is precisely
    the silent dead end `validate_mode_graph`'s reachability check exists to
    prevent one level up.

    Unauthored stubs are reported but are NOT errors on their own: authoring is
    incremental by design (one mode per `apply_prompt_edit` call), so a
    partially authored scaffold is a normal intermediate state. Callers that
    require completeness should check `authored_modes` explicitly.
    """
    errors: list[str] = []
    if "def compute_reward(" not in source:
        errors.append("source defines no compute_reward")
    if "def compute_reward_batched(" not in source:
        errors.append(
            "source defines no compute_reward_batched — mjlab dispatches to "
            "it and treats its absence as a reward-contract violation, so the "
            "modes would never be paid during training")
    if "MODE_WINDOWS_S" not in source:
        errors.append("source has no MODE_WINDOWS_S — mode gating is missing")

    for m in graph.modes:
        try:
            ident = mode_ident(m.name)
        except ModeError as e:
            errors.append(str(e))
            continue
        if f"def {MODE_FN_PREFIX}{ident}(" not in source:
            errors.append(
                f"mode {m.name!r}: no {MODE_FN_PREFIX}{ident} function — its "
                "terms could never be paid")
        if f"def {MODE_FN_PREFIX}{ident}{BATCHED_FN_SUFFIX}(" not in source:
            errors.append(
                f"mode {m.name!r}: no "
                f"{MODE_FN_PREFIX}{ident}{BATCHED_FN_SUFFIX} function — its "
                "terms would be paid in replay and silently skipped in "
                "training")

    errors.extend(_window_agreement_errors(source, graph))
    return errors


def windows_in_source(source: str) -> dict[str, tuple[float, float]]:
    """`MODE_WINDOWS_S` as the module actually declares it, or {} if unreadable.

    Parsed, not string-matched: the literal is no longer predictable from the
    graph alone once the automaton has been fitted to an episode horizon.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        if not any(isinstance(t, ast.Name) and t.id == "MODE_WINDOWS_S"
                   for t in targets):
            continue
        try:
            raw = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    return {}


def _declared_time_scale(source: str) -> Optional[float]:
    """`REWARD_SPEC['clip_time_scale']`, or None if the module predates it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        if not any(isinstance(t, ast.Name) and t.id == "REWARD_SPEC"
                   for t in targets):
            continue
        try:
            spec = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        value = spec.get("clip_time_scale") if isinstance(spec, dict) else None
        return float(value) if isinstance(value, (int, float)) else None
    return None


def _window_agreement_errors(source: str, graph: ModeGraph) -> list[str]:
    """Do the module's windows still describe THIS graph?

    This used to demand the clip-time literal verbatim. That conflated two
    different things: whether the scaffold belongs to this automaton, and which
    clock it is on. A scaffold fitted to an episode horizon is not stale — its
    windows are the graph's windows times one shared factor — so check the
    property that actually matters: same modes, same order, same proportions,
    no window moved by hand relative to its neighbours.
    """
    want = mode_windows_s(graph)
    got = windows_in_source(source)
    if not got:
        return ["source has an unreadable MODE_WINDOWS_S — mode gating cannot "
                "be verified against the automaton"]
    missing = [n for n in want if n not in got]
    if missing:
        return [f"mode {n!r}: no window in MODE_WINDOWS_S — the scaffold is "
                f"stale relative to the graph; regenerate it" for n in missing]

    # The scale must be the one the module DECLARES, not merely some uniform
    # one. Otherwise a scaffold built from a proportionally different graph —
    # the same clip re-exported at another fps, say — is numerically
    # indistinguishable from a horizon fit and would slip through. A module
    # that declares nothing is a pre-horizon scaffold and must still be exact.
    declared = _declared_time_scale(source)
    span = max((hi for _, hi in want.values()), default=0.0)
    got_span = max((hi for n, (_, hi) in got.items() if n in want), default=0.0)
    observed = (got_span / span) if span > 0 and got_span > 0 else 1.0
    scale = declared if declared is not None else 1.0
    if abs(observed - scale) > 1e-3:
        return [
            f"MODE_WINDOWS_S spans {got_span:g}s where this automaton spans "
            f"{span:g}s — a factor of {observed:.4f}, but the module declares "
            f"{scale:.4f}. The scaffold is stale relative to the graph; "
            f"regenerate it."]
    errors: list[str] = []
    for name, (lo, hi) in want.items():
        glo, ghi = got[name]
        # 1 ms, comfortably above the 4-dp rounding the scaler applies.
        if abs(glo - lo * scale) > 1e-3 or abs(ghi - hi * scale) > 1e-3:
            errors.append(
                f"mode {name!r}: window {(round(glo, 4), round(ghi, 4))} is not "
                f"{(lo, hi)} scaled by {scale:.4f} — the scaffold is stale "
                f"relative to the graph, or a window was moved by hand; "
                f"regenerate it")
    return errors


def _render_authoring_prompt(graph: ModeGraph, mode_name: str,
                             goal_text: str, mode_text: str,
                             windows: Optional[
                                 Mapping[str, tuple[float, float]]] = None,
                             task_brief: str = "",
                             ) -> str:
    """The prompt body. Split out so `mode_authoring_prompt` can measure the
    fixed part before deciding how much free text fits.

    `windows` overrides the graph's clip-time windows. The prompt states the
    mode's window on purpose, so once a scaffold has been fitted to an episode
    horizon it must state the FITTED window — otherwise the model reasons about
    a 0.82 s approach while the module pays one lasting 2.37 s.

    `task_brief` is the world and goal the robot is being trained in, rendered
    by the caller (`sculptor.mode_rewards.task_brief`). It is empty for a pure
    imitation project. Without it the prompt describes a clip and nothing else,
    so a model can only write terms over proprioception — gait quality,
    uprightness, stillness — every one of which a robot standing in front of
    the course maximizes. On the platform-ascent-showcase run that produced
    four modes' worth of terms, none of which mentioned the platforms, and the
    policy stood still for three iterations. The goal channels were published
    by the runner the whole time; nothing told the author they existed.
    """
    mode = graph.mode(mode_name)
    windows = dict(windows) if windows else mode_windows_s(graph)
    lo, hi = windows[mode_name]
    order = [m.name for m in graph.modes]
    i = order.index(mode_name)

    neighbours = []
    if i > 0:
        neighbours.append(
            f"after {order[i - 1]!r} (ends {windows[order[i - 1]][1]:g}s)")
    if i + 1 < len(order):
        neighbours.append(
            f"before {order[i + 1]!r} (starts {windows[order[i + 1]][0]:g}s)")
    neighbour_line = "; ".join(neighbours) if neighbours else "the only mode"

    ident = mode_ident(mode_name)
    return (
        f"Author `{MODE_FN_PREFIX}{ident}` and its batched twin "
        f"`{MODE_FN_PREFIX}{ident}{BATCHED_FN_SUFFIX}` — the reward for mode "
        f"{mode_name!r} ONLY. Both must compute the SAME quantity: the scalar "
        f"one is scored in replay, the batched one is what trains, and "
        f"authoring one without the other silently pays nothing where it "
        f"matters.\n\n"
        f"Overall behavior: {goal_text or '(unspecified)'}\n"
        f"This mode's job: {mode_text}\n"
        f"Window: {lo:g}s-{hi:g}s (reference frames "
        f"[{mode.frame_range[0]}, {mode.frame_range[1]}) at {graph.fps:g} fps); "
        f"{neighbour_line}.\n\n"
        f"{task_brief}"
        "Scope:\n"
        "- Called ONLY inside that window. Do not re-detect the phase — the "
        "dispatch already did.\n"
        "- Do not reward or penalize what the neighbouring modes own. A term "
        "that is right here but reads as a global constraint is the main "
        "failure.\n"
        "- Return `(value, components)`; snake_case component names, the SAME "
        "ones in both halves. They are reported as `<mode>.<name>` and scored "
        "against this window's slice of the rollout.\n"
        "- Leave `compute_reward`, `compute_reward_batched`, `MODE_WINDOWS_S`, "
        "`MODE_ORDER`, `_mode_masks` and the other modes untouched.\n"
        f"- A shared helper is fine — define it at module level, name it "
        f"`_{ident}_<what>`, and both halves may call it. Anything you add "
        "elsewhere is discarded.\n"
        "\nBatched half — signature `(state, action, next_state, info, like)`:\n"
        "- `like` is zeros of shape (num_envs,); build terms from it "
        "(`torch.zeros_like(like)`, `torch.full_like(like, c)`) so device and "
        "dtype follow the env.\n"
        "- It runs for EVERY env and is masked afterwards, so it must be "
        "finite outside this window too: no unguarded div, log or index that "
        "assumes time is inside it.\n"
        "- Return tensors of shape (num_envs,), not floats.\n"
        "- Arithmetic on a bool tensor raises; use `mask.float()`."
    )


#: Cap on how many course elements the brief lists. A long course would eat the
#: prompt budget `_fit_free_text` divides up, and the shape of the first dozen
#: is what a mode's terms actually need.
MAX_BRIEF_ELEMENTS = 12


def _element_line(element: Mapping[str, Any]) -> str:
    """One course element as `kind id: k=v, k=v`, in the units the spec uses."""
    nominal = element.get("nominal") or {}
    dims = ", ".join(
        f"{k.removesuffix('_m')} {float(v):g} m"
        for k, v in sorted(nominal.items())
        if isinstance(v, (int, float)))
    kind = str(element.get("element") or "element")
    ident = str(element.get("id") or "")
    return f"  {kind} {ident}: {dims}" if dims else f"  {kind} {ident}"


#: Catalog `access` value that means "a reward term may read this". Everything
#: else is published for metrics only and is NOT in the contract's
#: `expected_info_keys`, so `sculptor.edit`'s grounding check rejects any term
#: that reads it. Naming them in the brief anyway — as off-limits — is
#: deliberate: a model told only about the distance channel invents a
#: `waypoint_index` read on its own, and fails validation after the model call.
SHAPING_ACCESS = "shared_shaping"


def task_brief(task: Optional[Mapping[str, Any]],
               world: Optional[Mapping[str, Any]],
               channels: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    """The mission section of an authoring prompt, or "" when there is none.

    Renders three things the author cannot otherwise know: the course geometry,
    the goal channels the runner publishes into `info` (by their exact keys and
    split by whether a reward may read them), and the magnitude discipline that
    keeps a stand-still policy from out-earning a working one.

    `channels` is the project's channel catalog (`channel_catalog["channels"]`).
    Without it the brief still describes the course and the goal, but names no
    channel — better than naming one the contract does not ground.

    Empty for a project with no goal — a pure imitation reward should not be
    told to chase a course that does not exist.
    """
    shared = dict((task or {}).get("shared") or task or {})
    goal = dict(shared.get("goal") or {})
    goal_id = str(goal.get("id") or "").strip()
    goal_type = str(goal.get("type") or "").strip()
    if not goal_id or not goal_type:
        return ""

    lines = ["MISSION — what this reward is FOR. The reference clip is a style",
             "prior; the world below is the objective."]

    obstacles = dict(
        ((world or {}).get("shared") or world or {}).get("obstacles") or {})
    course = list(obstacles.get("course") or [])
    if course:
        start = obstacles.get("start_offset_m")
        head = f"Course ({obstacles.get('layout') or 'linear'})"
        if isinstance(start, (int, float)):
            head += f", robot starts {float(start):g} m before it"
        lines.append(head + ", in order:")
        lines.extend(_element_line(e) for e in course[:MAX_BRIEF_ELEMENTS])
        if len(course) > MAX_BRIEF_ELEMENTS:
            lines.append(f"  … and {len(course) - MAX_BRIEF_ELEMENTS} more")

    success = dict(goal.get("success") or {})
    goal_line = f"Goal: {goal_type} {goal_id!r}"
    if success.get("predicate"):
        goal_line += f" — {success['predicate']}"
    if isinstance(success.get("hold_s"), (int, float)):
        goal_line += f", held {float(success['hold_s']):g}s"
    lines.append(goal_line + ".")

    mine = [c for c in (channels or [])
            if str((c.get("source") or {}).get("goal") or "") == goal_id]
    readable = [c for c in mine if c.get("access") == SHAPING_ACCESS]
    metric_only = [c for c in mine if c.get("access") != SHAPING_ACCESS]

    if readable:
        lines += [
            "",
            "Task channels your terms MAY read. They are in `info` every step;",
            "use `info.get(<key>)` and give both halves the same fallback:",
        ]
        lines += [f"  {c.get('name')} — {c.get('metric_role') or 'task signal'}"
                  for c in readable]
    if metric_only:
        lines += [
            "",
            "Published for METRICS ONLY — reading one is rejected as ungrounded,",
            "because it is not in the reward contract's info keys:",
        ]
        lines += [f"  {c.get('name')}" for c in metric_only]

    progress = next((c.get("name") for c in readable
                     if c.get("metric_role") == "progress"), None)
    if progress:
        lines += [
            "",
            f"REQUIRED: include a dense progress term built on `{progress}`,",
            "and make it the LARGEST term in this mode. Dense means it pays for",
            "the first centimetre closed, not on arrival — an arrival-gated",
            "bonus is unreachable from a policy that has never arrived.",
        ]
    lines += [
        "",
        "Anything a MOTIONLESS robot can max out (upright, stillness, pose,",
        "gait shape) must be gated on progress or kept small: standing still",
        "already collects ~2/step from the tracking backbone and the base",
        "locomotion terms, so an ungated stationary term competes with the",
        "mission instead of serving it.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fit_free_text(behavior_goal: str, mode_goal: str, *,
                   fixed: int) -> tuple[str, str]:
    """Squeeze the two caller-supplied strings into what's left of the budget.

    `apply_prompt_edit` hard-rejects a prompt over `MAX_PROMPT_CHARS`
    (`edit.py:2042`), and a behavior goal is free text a user typed — so
    without this, a long goal fails at the *end* of the authoring call, after
    the KG query, with an error about a character count rather than about the
    goal. Truncating here is deterministic and visible in `--print-prompt`.
    """
    goal = " ".join(str(behavior_goal or "").split())
    mode = " ".join(str(mode_goal or "").split())
    room = MAX_PROMPT_CHARS - fixed
    if len(goal) + len(mode) <= room:
        return goal, mode
    # The mode's own job is the more specific of the two, so it keeps its half
    # of the budget outright and the overall goal absorbs the truncation.
    half = max(0, room // 2)
    if len(mode) > half:
        mode = mode[:max(0, half - 1)].rstrip() + "…"
    goal_room = max(0, room - len(mode))
    if len(goal) > goal_room:
        goal = goal[:max(0, goal_room - 1)].rstrip() + "…"
    return goal, mode


def mode_authoring_prompt(
    graph: ModeGraph,
    mode_name: str,
    *,
    behavior_goal: str = "",
    mode_goal: str = "",
    windows: Optional[Mapping[str, tuple[float, float]]] = None,
    task_brief: str = "",
) -> str:
    """The per-mode authoring instruction for `apply_prompt_edit`.

    Deliberately states the mode's window and its neighbours: the most common
    way per-mode authoring goes wrong is a term that is correct for its mode but
    written as though it applied to the whole episode (e.g. "keep both feet on
    the ground" authored for an approach mode, which then reads as a global
    constraint to anyone editing later). Naming the neighbours makes the scope
    explicit in the prompt rather than implicit in the gating.

    The result is guaranteed to fit `MAX_PROMPT_CHARS`; see `_fit_free_text`
    for why that is enforced here rather than left to the caller.
    """
    mode = graph.mode(mode_name)          # KeyError here is a caller bug
    goal_text, mode_text = _fit_free_text(
        behavior_goal, mode_goal or mode.name,
        fixed=len(_render_authoring_prompt(graph, mode_name, "", "", windows,
                                           task_brief)))
    return _render_authoring_prompt(graph, mode_name, goal_text, mode_text,
                                    windows, task_brief)


__all__ = [
    "AUTHOR_MAX_TOKENS",
    "BATCHED_FN_SUFFIX",
    "MAX_PROMPT_CHARS",
    "MODE_COMPONENT_PREFIX",
    "MODE_FN_PREFIX",
    "ModeAuthorError",
    "authored_modes",
    "author_mode",
    "authoring_twin_source",
    "generate_mode_reward_scaffold",
    "graft_mode_bodies",
    "mode_authoring_prompt",
    "mode_ident",
    "mode_windows_s",
    "probe_info_keys",
    "probe_reward_module",
    "promote_mode_reward",
    "summarize_authored_modes",
    "task_brief",
    "validate_mode_reward_source",
]


# ── the authoring call itself ─────────────────────────────────────────────
#
# Lives here rather than in `cli.py` because there are two callers — the CLI
# and the UI's mode-author job — and the sequence below is not a formality.
# Every step of it was added because a real run failed without it: the twin
# because a model mangled a 32x29 float table, the summary because carrying
# finished neighbours blew the output budget, the helper carry because a model
# asked for one function wrote two, the info-key gate because a reward reading
# a key the env never sends pays a constant and passes everything else.
class ModeAuthorError(ModeError):
    """An authoring attempt that produced nothing usable."""


class _RecordingInfo(dict):
    """A dict that remembers which keys were actually read."""

    def __init__(self, data):
        super().__init__(data)
        self.seen: set = set()

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)


def probe_reward_module(path, contract) -> Optional[str]:
    """Run `sculptor.edit`'s own reward probes on a file, or say why not.

    Needed because authoring runs against a twin: `apply_prompt_edit` validated
    THAT, and the grafted module is what trains. Re-probing is what makes the
    graft safe rather than merely convenient.
    """
    import importlib.util
    from pathlib import Path as _Path

    from sculptor.edit import (EditValidationError, _call_compute_reward,
                               _call_compute_reward_batched)

    path = _Path(path)
    try:
        spec = importlib.util.spec_from_file_location(f"_probe_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — surfaced as a message, not a crash
        return f"module does not import: {type(e).__name__}: {e}"
    for probe in (_call_compute_reward, _call_compute_reward_batched):
        try:
            probe(mod, contract)
        except EditValidationError as e:
            return str(e)
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
    return None


def probe_info_keys(path, contract, mode_name: str) -> Optional[str]:
    """Reject a mode that reads an `info` key the env never publishes.

    This is the failure the other probes cannot see. The first real authoring
    run reached for ten info keys through a helper doing `info.get(key, 0.0)` —
    every one happened to be real, but had one not been, that term would have
    paid a constant 0.0 for the whole of training while the module imported,
    ran and validated perfectly. A reward whose terms silently evaluate to a
    constant is the exact shape of a gameable reward.

    Recorded at runtime rather than read off the source, so a key reached
    through a helper, a loop or an f-string is caught the same as a literal.
    """
    import importlib.util
    from pathlib import Path as _Path

    from sculptor.edit import _build_dummy_inputs

    path = _Path(path)
    spec = importlib.util.spec_from_file_location(f"_keyprobe_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    declared = set(contract.expected_info_keys or [])
    seen: set = set()
    ident = mode_ident(mode_name)

    s, a, ns, info = _build_dummy_inputs(contract)
    rec = _RecordingInfo(info)
    fn = getattr(mod, MODE_FN_PREFIX + ident, None)
    if fn is not None:
        try:
            fn(s, a, ns, rec)
        except Exception:  # noqa: BLE001 — crashes are the other probes' job
            pass
        seen |= rec.seen

    if bool(getattr(contract, "supports_batched", False)):
        # The batched dispatch runs EVERY mode and masks afterwards, so one
        # call covers this mode whatever the dummy elapsed time says.
        import torch

        schema = dict(contract.state_schema or {})
        n = 2
        state = {k: torch.zeros((n, *sh), dtype=torch.float32)
                 for k, sh in schema.items()}
        action = torch.zeros((n, int(schema.get("actuator_force", (1,))[0])),
                             dtype=torch.float32)
        ischema = getattr(contract, "info_schema", None) or {}
        brec = _RecordingInfo({
            k: torch.zeros((n, *tuple(ischema.get(k, ()))), dtype=torch.float32)
            for k in declared})
        bfn = getattr(mod, MODE_FN_PREFIX + ident + BATCHED_FN_SUFFIX, None)
        if bfn is not None:
            try:
                bfn(state, action, state, brec, torch.zeros(n))
            except Exception:  # noqa: BLE001
                pass
            seen |= brec.seen

    unknown = sorted(k for k in seen if k not in declared)
    if not unknown:
        return None
    return (f"reads info keys this env does not publish: {', '.join(unknown)}. "
            f"Every such read returns the fallback forever, so the term is a "
            f"constant. Available keys: {', '.join(sorted(declared))}")


#: Token ceiling for a mode-authoring call. The `edit.MAX_TOKENS` default of
#: 16000 truncated attempt 1 of EVERY real authoring run — adaptive thinking on
#: "write a single-leg-takeoff reward" is expensive and what is left has to
#: carry the whole module back. Raised here rather than there, because
#: `edit.MAX_TOKENS` has a 240s HTTP timeout calibrated against it for the
#: training-mission path, which is not what needed more room.
AUTHOR_MAX_TOKENS = 32000


def author_mode(*, source: str, graph: ModeGraph, mode: str, contract,
                clip: Optional[Mapping[str, Any]] = None,
                clip_id: str = "", behavior_goal: str = "",
                mode_goal: str = "", kg_store=None,
                mission: str = "",
                max_tokens: int = AUTHOR_MAX_TOKENS,
                on_event=None) -> dict:
    """Author one mode's bodies into `source`, leaving every other mode alone.

    One mode per call on purpose: the scaffold's gating is already correct, so
    the only thing a model can get wrong is the terms of a single mode. That
    keeps the blast radius of a bad edit to one window rather than the whole
    behavior, and lets each mode clear the metric gauntlet separately.

    `mission` is the rendered `task_brief` for the project's world and goal, or
    "" for pure imitation. It is what makes an authored term able to mention
    the course at all.

    Returns `{"source", "prompt", "authored", "pending"}`. Raises
    `ModeAuthorError` with a caller-presentable message on any rejection;
    `sculptor.edit.EditValidationError` propagates unchanged.
    """
    import tempfile
    from pathlib import Path as _Path

    from sculptor.edit import apply_prompt_edit

    try:
        graph.mode(mode)
    except KeyError:
        names = ", ".join(m.name for m in graph.modes)
        raise ModeAuthorError(
            f"no mode {mode!r} in this automaton; have: {names}") from None

    stale = validate_mode_reward_source(source, graph)
    if stale:
        # Authoring into a scaffold that no longer matches the automaton would
        # write terms for a window that has since moved. Regenerate instead.
        raise ModeAuthorError(
            "scaffold no longer matches the automaton: " + "; ".join(stale))

    # State the module's OWN windows, not the graph's. They differ whenever the
    # scaffold was fitted to an episode horizon, and the prompt's whole job is
    # to tell the model which slice of the episode its terms are paid over.
    prompt = mode_authoring_prompt(
        graph, mode, behavior_goal=behavior_goal, mode_goal=mode_goal,
        windows=windows_in_source(source) or None, task_brief=mission)

    twin_source = authoring_twin_source(
        graph, clip=clip, behavior_goal=behavior_goal, clip_id=clip_id,
        horizon_s=_horizon_of(source),
        tracking_weight=_tracking_weight_of(source))
    already = [n for n, ok in authored_modes(source).items() if ok]
    if already:
        twin_source = summarize_authored_modes(twin_source, source, already)

    with tempfile.TemporaryDirectory(prefix="rs_mode_author_") as tmp:
        twin = _Path(tmp) / "v0.py"
        twin.write_text(twin_source, encoding="utf-8")
        edited = apply_prompt_edit(
            current_reward_path=twin, user_prompt=prompt, new_iter_id="v1",
            reward_contract=contract, kg_store=kg_store,
            max_tokens=max_tokens, max_prompt_chars=MAX_PROMPT_CHARS,
            on_event=on_event)
        edited_source = _Path(edited).read_text(encoding="utf-8")

    grafted = graft_mode_bodies(source, edited_source, [mode])

    stale = validate_mode_reward_source(grafted, graph)
    if stale:
        raise ModeAuthorError("grafted module invalid: " + "; ".join(stale))

    with tempfile.TemporaryDirectory(prefix="rs_mode_probe_") as tmp:
        probe_path = _Path(tmp) / "probe.py"
        probe_path.write_text(grafted, encoding="utf-8")
        err = probe_reward_module(probe_path, contract)
        if err:
            # The twin cleared the probes; the grafted module is what trains.
            raise ModeAuthorError(
                f"grafted module failed the reward contract probe: {err}")
        err = probe_info_keys(probe_path, contract, mode)
        if err:
            raise ModeAuthorError(err)

    authored = authored_modes(grafted)
    if not authored.get(mode):
        # Accepted by the reward gates but the bodies were not actually
        # filled — a silent no-op, worse than a rejection because the next
        # call would move on to the following mode.
        raise ModeAuthorError(
            f"{mode} still reads as an unauthored stub — the edit was accepted "
            f"but did not fill its bodies. Re-run, or check the prompt reached "
            f"the right function.")
    return {"source": grafted, "prompt": prompt, "authored": authored,
            "pending": [n for n, ok in authored.items() if not ok]}


def reward_spec_from_source(source: str) -> dict:
    """`REWARD_SPEC` read out of a reward module without importing it.

    Read rather than imported because a per-mode module builds its numpy
    target tables at import time, and every caller here only wants a couple of
    scalars out of the header.

    Regex is not an option: the scaffold writes the dict across many lines with
    double-quoted keys, and `_rewrite_reward_spec` — which every promotion runs
    — replaces it with a single-line `repr`, so the same key appears as
    `"reference_clip_id":` before promotion and `'reference_clip_id':` after.
    Matching one shape silently returned nothing for the other.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in tree.body:
        named = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "REWARD_SPEC"
                    for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REWARD_SPEC"
        )
        if not named or node.value is None:
            continue
        try:
            spec = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        return spec if isinstance(spec, dict) else {}
    return {}


def _rewrite_reward_spec(source: str, updates: Mapping[str, Any]) -> str:
    """Return `source` with `REWARD_SPEC`'s keys updated in the literal.

    Rewritten in place rather than mutated at import: readers of a reward
    module (`reward_store._extract_reward_spec`, the version list, the diff
    view) `ast.literal_eval` the dict literal and never execute the module, so
    a `REWARD_SPEC[...] = ...` line appended at the bottom would be invisible
    to every one of them.
    """
    import ast

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign):
            target = next((t for t in node.targets
                           if isinstance(t, ast.Name) and t.id == "REWARD_SPEC"),
                          None)
        elif (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "REWARD_SPEC"):
            target = node.target
        if target is None or node.value is None:
            continue
        spec = ast.literal_eval(node.value)
        if not isinstance(spec, dict):
            raise ModeAuthorError("REWARD_SPEC is not a dict literal")
        spec.update(updates)
        start, end = offsets[node.lineno - 1], offsets[node.end_lineno]
        # `repr` of a dict of literals stays literal-evaluable, which is the
        # only property the readers need.
        return (source[:start] + f"REWARD_SPEC: dict = {spec!r}\n"
                + source[end:])
    raise ModeAuthorError("no module-level REWARD_SPEC assignment to update")


def promote_mode_reward(path, *, contract=None, allow_unauthored: bool = False,
                        author: str = "sculptor") -> dict:
    """Copy a mode-reward module into the project's reward version chain.

    Without this the feature does nothing. Authoring writes
    `rewards/mode_reward_v<n>.py`, but only `v<n>.py` is a version: it is what
    `reward_store.list_versions` matches, what the Rewards tab lists, and
    `rewards/current.py` — the module every adapter actually imports — points
    at one. Author a per-mode reward and press Run without this step and the
    run trains whatever `current.py` pointed at before, silently.

    Refuses a module with unauthored stubs unless `allow_unauthored`. A stub
    pays nothing, so promoting a half-authored module trains a reward that is
    blank across part of the episode — the same silent-underpay failure the
    `UNAUTHORED STUB` marker exists to make loud. Scaffolds are legitimately
    trainable on their tracking backbone alone, which is why this is a flag
    rather than a hard refusal.
    """
    import hashlib
    import re as _re
    from pathlib import Path as _Path

    from sculptor.edit import _write_current_reexport

    path = _Path(path)
    rewards_dir = path.parent
    source = path.read_text(encoding="utf-8")

    authored = authored_modes(source)
    unauthored = [n for n, ok in authored.items() if not ok]
    if unauthored and not allow_unauthored:
        raise ModeAuthorError(
            f"{path.name} has {len(unauthored)} unauthored stub(s): "
            f"{', '.join(unauthored)}. Each pays nothing, so promoting this "
            f"trains a reward that is blank for that part of the episode. "
            f"Author them, or pass allow_unauthored to train the tracking "
            f"backbone alone.")

    if contract is not None:
        err = probe_reward_module(path, contract)
        if err:
            raise ModeAuthorError(f"{path.name} fails the reward contract: {err}")

    versions = sorted(
        int(m.group(1))
        for p in rewards_dir.iterdir()
        for m in [_re.fullmatch(r"v(\d+)\.py", p.name)] if m)
    n = (versions[-1] + 1) if versions else 0
    parent_hash = ""
    if versions:
        parent_hash = hashlib.sha256(
            (rewards_dir / f"v{versions[-1]}.py").read_text(encoding="utf-8")
            .encode("utf-8")).hexdigest()[:16]

    # Which mode-reward file this version came from, and its exact bytes.
    # Promotion rewrites REWARD_SPEC, so the copy never digests equal to its
    # source and "is the promoted version still this file?" cannot be answered
    # by comparing them. Without an answer the UI could only assume, and it
    # assumed "yes" — so after re-authoring, the button that promotes was
    # disabled behind a reward two versions stale, with no way to advance it.
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    promoted = _rewrite_reward_spec(source, {
        "version": f"v{n}", "parent_hash": parent_hash, "author": author,
        "source_filename": path.name, "source_sha256": source_sha256})
    dest = rewards_dir / f"v{n}.py"
    dest.write_text(promoted, encoding="utf-8")
    _write_current_reexport(rewards_dir, dest)
    return {"version": n, "path": str(dest), "filename": dest.name,
            "unauthored": unauthored, "parent_hash": parent_hash,
            "source_filename": path.name, "source_sha256": source_sha256}
