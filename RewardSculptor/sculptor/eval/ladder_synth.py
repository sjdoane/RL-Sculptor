"""§Ship 51: deterministic competence-ladder synthesizer (L2 — the unblocker).

A generated objective metric earns STEER-RIGHTS on a NOVEL task by ranking
an INDEPENDENTLY-AUTHORED competence ladder. The author (a separate, metric-
blind LLM call, `gen_competence_ladder.md`) never writes numpy — it emits a
`CompetenceLadder` = a competence axis + ordered rung specs (low→high
competence) using a fixed, bounded MOTION VOCABULARY. This module is the
DETERMINISTIC half: it renders each rung spec to a physical rollout
(joint_pos / joint_vel / projected_gravity_b / root_link_pos_w), so the
ground truth stays PHYSICAL and the author can never bias the rendering.

Why a generic vocabulary (vs the hand-coded `_ladder` per family): a novel
task has no built-in ground truth. The author maps the task's competence
axis onto renderable PHYSICAL primitives — posture (uprightness, height),
progress (forward/lateral travel, hops), and coordinated joint motion
(GROUPS of joints oscillating / bursting / held, with anti-/in-phase
structure). Targeting is by canonical ROLE (Ship-49 resolver), never a raw
column, so a group like "arms" drives every shoulder+elbow on THIS robot —
the fix for the single-joint dilution that scored a floss ladder 0.0.

Key conventions (matched to spec_metrics / _archetypes / _ladder byte-for-
byte so a rendered rung reads identically to the built-in ladders):
  * gravity z < −0.85 ≈ upright; the `uprightness` scalar is rendered as the
    FRACTION of frames upright (smooth grading — a constant intermediate
    tilt would step the uprightness() gate at the threshold).
  * joint_pos is the integral of joint_vel (cumsum); +x is forward; root z is
    base height.
  * a renderer-built DEGENERATE ANCHOR (rung 0, NOT author-chosen) gives every
    ladder a real incompetent floor, so the calibration gate can require the
    metric to separate competent from degenerate, not merely rank.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from sculptor.eval.joint_resolver import resolve_joint_roles, select_joints

T, E, DT = 120, 4, 0.02
_BEHAVIOR = {"max_episode_steps": T, "rollout_num_envs": E, "step_dt": DT}

_VALID_AXES = ("pitch", "roll", "yaw")
_BURST_WIN = 5
_HOP_WIN = 20

# §fold-and-return primitive: a single 0→1→0 arc over the whole rollout (dips at
# mid, returns to start). The pelvis follows base_height − fold_depth·_FOLD_ARC and
# `fold`-mode groups follow offset + amplitude·_FOLD_ARC IN PHASE, so a squat /
# toe-touch / deep bow / floor-touch-and-rise renders as a coherent DOWN-AND-UP cycle.
# The arc is forced-symmetric (start == end), so this is for true dip-AND-RETURN goals
# ONLY — a one-way height change (sit-to-stand) is a base_height_m [start,end] ramp,
# not a fold. Byte-for-byte the dip-and-return that metric_validate._selectivity_probe
# (C1) scores, so the task-derived calibration ladder and the L0 validation probe
# agree on what a fold competence looks like (the real toe-touch metric scores ≈0.83).
_FOLD_ARC = (1.0 - np.cos(2.0 * np.pi * np.arange(T) / T)) / 2.0


# ── the motion vocabulary (what the blind author fills) ──────────────


class RoleQuery(BaseModel):
    """Which joints a group drives — a predicate over canonical roles,
    resolved by the Ship-49 resolver. Never a raw column index."""

    segments: list[str] = Field(default_factory=list)
    # `null` (None) admits single-DOF joints; default pitch+single-DOF =
    # sagittal plane (matches LEG_SAGITTAL_AXES).
    axes: list[Optional[str]] = Field(default_factory=lambda: ["pitch", None])
    sides: Optional[list[str]] = None

    @field_validator("axes", mode="before")
    @classmethod
    def _coerce_axes(cls, v):
        """§round-13: the blind gaming/ladder author commonly emits `axes: null`
        (a whole-set null, not a list of nulls). The strict `list[Optional[str]]`
        type REJECTS that, raising a ValidationError that drops the ENTIRE
        GamingArchetypeSet — which made the adversarial gate fail OPEN (ran=False
        → not enforced → false grant). Coerce a top-level null to the default
        (pitch + single-DOF) so one malformed field degrades to the default, never
        nukes the whole set; a stray scalar is wrapped into a 1-element list."""
        if v is None:
            return ["pitch", None]
        if isinstance(v, (str, type(None))):
            return [v]
        return v


class Group(BaseModel):
    """A coordinated set of joints in one motion mode."""

    name: str = "g"
    role_query: RoleQuery = Field(default_factory=RoleQuery)
    mode: str = "oscillate"            # oscillate | burst | hold | fold
    amplitude_rad: float = 0.0
    period_frames: int = 25
    phase: float = 0.0
    within_group_phase_spread: float = 0.0
    offset_rad: float = 0.0
    peak_radps: float = 0.0
    burst_count: int = 0


class Coordination(BaseModel):
    group_a: str = ""
    group_b: str = ""
    relation: str = "anti_phase"      # anti_phase | in_phase | phase_lag
    lag_frames: int = 0


# A posture scalar may be a constant or a [start, end] ramp over the rollout.
Scalar = Union[float, list[float]]


class MotionSpec(BaseModel):
    """One rung. All numeric fields are CLAMPED at render time (never raises);
    an unrenderable/unknown target is dropped and recorded, never silent."""

    uprightness: Scalar = 1.0          # [0,1] fraction of frames upright
    base_height_m: Scalar = 0.7        # [0,1.2]
    forward_speed_mps: float = 0.0     # [-2,2], +x
    lateral_speed_mps: float = 0.0     # [-2,2]
    hop_height_m: float = 0.0          # [0,0.6]
    hop_count: int = 0                 # [0,8]
    fold_depth_m: float = 0.0          # [0,0.6] pelvis DIP-AND-RETURN depth (squat/
    #                                    toe-touch/deep bow/floor-touch); 0 = no fold
    groups: list[Group] = Field(default_factory=list)
    coordination: list[Coordination] = Field(default_factory=list)
    tremor: float = 0.0                # [0,2] high-freq jitter (degeneracy)
    noise: float = 0.0                 # [0,0.2] gaussian on joint_vel
    degenerate_axis: bool = False      # author honesty: goal needs yaw/contact/force
    degenerate_reason: str = ""


class CompetenceLadder(BaseModel):
    """What the author returns: a named competence axis + ascending rungs."""

    competence_axis: str = ""
    rungs: list[MotionSpec] = Field(default_factory=list)


# ── §Ship 53: adversarial gaming archetypes (L3) ─────────────────────


class GamingArchetype(BaseModel):
    """One GAMING POLICY: an OFF-GOAL behavior a naive metric might score high
    (stand-and-flail, fall-rhythmically, walk-away, vibrate). A `motion` is the
    SAME renderable `MotionSpec` vocabulary as a ladder rung — gaming policies
    are just MotionSpecs that do NOT achieve the goal. `strategy` is the blind
    author's one-line rationale for HOW it tries to fool a proxy."""

    name: str = "gaming"
    strategy: str = ""
    motion: MotionSpec = Field(default_factory=MotionSpec)


class GamingArchetypeSet(BaseModel):
    """What the adversarial author returns: a restated goal + a few gaming
    policies. The author is metric-BLIND (sees only goal/robot/joint_names)."""

    goal_restated: str = ""
    archetypes: list[GamingArchetype] = Field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def _ramp(scalar: Any, lo: float, hi: float) -> np.ndarray:
    """A constant or [start,end] ramp → per-frame (T,) array, clamped."""
    if isinstance(scalar, (list, tuple)) and len(scalar) >= 2:
        a, b = _clip(scalar[0], lo, hi), _clip(scalar[1], lo, hi)
        return np.linspace(a, b, T)
    return np.full(T, _clip(scalar if not isinstance(scalar, (list, tuple))
                            else (scalar[0] if scalar else hi), lo, hi))


def _resolve_targets(joint_names: list[str], q: RoleQuery) -> list[int]:
    axes = [a for a in (q.axes or ["pitch", None]) if a in _VALID_AXES or a is None]
    segs = [s for s in (q.segments or [])] or None
    sides = q.sides or None
    if segs is None:
        return []
    return select_joints(joint_names, segments=segs,
                         axes=axes or None, sides=sides)


def _upright_gravity(up_frac: np.ndarray) -> np.ndarray:
    """Render the uprightness FRACTION: gravity points down (g_z=−1, upright)
    on a deterministic subset of frames whose density equals `up_frac`, and
    sideways (g_x=1, toppled) on the rest. So `uprightness()` (the fraction of
    frames with g_z<−0.85) ≈ the scalar — smooth grading, unlike a constant
    intermediate tilt which would STEP the gate at the −0.85 threshold. A
    [start,end] ramp yields a per-frame-varying upright density."""
    g = np.zeros((T, E, 3))
    grid = (np.arange(T) % 100) / 100.0        # interleaved threshold grid
    upright = grid < up_frac                    # (T,) bool
    g[upright, :, 2] = -1.0
    g[~upright, :, 0] = 1.0
    return g


def render_rung(
    spec: MotionSpec, joint_names: list[str], *, rung_index: int = 0,
) -> tuple[dict, dict, dict]:
    """Render one rung spec to (arrays, behavior, meta). Pure-numpy,
    deterministic given (spec, rung_index). Never raises — bad fields clamp,
    unknown targets drop (recorded in meta).

    A `fold_depth_m` rung dips the pelvis and returns (the inverse of a hop) and a
    `fold`-mode group flexes its joints and returns IN PHASE — the dip-and-return a
    monotone `_ramp` base_height cannot express, so a squat / toe-touch / deep bow
    metric can rank a competence ladder and earn steer-rights."""
    J = len(joint_names)
    t = np.arange(T)
    jp = np.zeros((T, E, J))
    burst_jv = np.zeros((T, E, J))

    # gravity from the uprightness fraction (smooth grading).
    up = _ramp(spec.uprightness, 0.0, 1.0)
    g = _upright_gravity(up)

    # root: height (− fold dip, + hops) and travel.
    root = np.zeros((T, E, 3))
    root[..., 2] = _ramp(spec.base_height_m, 0.0, 1.2)[:, None]
    # §fold: a single pelvis dip that returns to start (the INVERSE of a hop —
    # base_height is a monotone ramp, so a fold needs its own primitive). Applied
    # before the hop block so a (rare) fold+hop composes; a fold task has hop_count
    # 0, so this is byte-identical for every non-fold rung (fold_depth_m default 0).
    fd = _clip(spec.fold_depth_m, 0.0, 0.6)
    if fd > 0:
        root[..., 2] -= (fd * _FOLD_ARC)[:, None]
        # a pelvis can't dip below the floor: clamp ≥0 so a deep fold from a low base
        # height degrades to a physical posture, never a negative z (degrade-to-safe).
        root[..., 2] = np.clip(root[..., 2], 0.0, None)
    hh = _clip(spec.hop_height_m, 0.0, 0.6)
    hc = int(_clip(spec.hop_count, 0, 8))
    if hh > 0 and hc > 0:
        z = root[:, 0, 2].copy()
        period = max(T // hc, _HOP_WIN + 1)
        for c in range(hc):
            s0 = c * period + 5
            for k in range(_HOP_WIN):
                if s0 + k < T:
                    z[s0 + k] += hh * np.sin(np.pi * k / _HOP_WIN)
        root[..., 2] = z[:, None]
    root[..., 0] = np.cumsum(np.full(T, _clip(spec.forward_speed_mps, -2, 2) * DT))[:, None]
    root[..., 1] = np.cumsum(np.full(T, _clip(spec.lateral_speed_mps, -2, 2) * DT))[:, None]

    # groups: resolve targets, apply motion. Record resolution counts.
    group_phase: dict[str, float] = {}
    resolved_counts: dict[str, int] = {}
    by_name: dict[str, Group] = {}
    for gr in spec.groups[:8]:
        idxs = _resolve_targets(joint_names, gr.role_query)
        resolved_counts[gr.name] = len(idxs)
        by_name[gr.name] = gr
        group_phase[gr.name] = _clip(gr.phase, -100, 100)
    # coordination adjusts group_b phase relative to group_a.
    coord_dropped = []
    for co in spec.coordination[:6]:
        if co.group_a in group_phase and co.group_b in by_name:
            base_ph = group_phase[co.group_a]
            if co.relation == "anti_phase":
                group_phase[co.group_b] = base_ph + np.pi
            elif co.relation == "in_phase":
                group_phase[co.group_b] = base_ph
            elif co.relation == "phase_lag":
                gb = by_name[co.group_b]
                per = max(int(_clip(gb.period_frames, 4, 120)), 4)
                group_phase[co.group_b] = base_ph + 2 * np.pi * _clip(
                    co.lag_frames, 0, 60) / per
        else:
            coord_dropped.append({"a": co.group_a, "b": co.group_b})

    for gr in spec.groups[:8]:
        idxs = _resolve_targets(joint_names, gr.role_query)
        if not idxs:
            continue
        amp = _clip(gr.amplitude_rad, 0.0, 1.5)
        per = max(int(_clip(gr.period_frames, 4, 120)), 4)
        off = _clip(gr.offset_rad, -1.5, 1.5)
        spread = _clip(gr.within_group_phase_spread, 0.0, 6.283)
        ph = group_phase.get(gr.name, _clip(gr.phase, -100, 100))
        if gr.mode == "oscillate":
            for n, i in enumerate(idxs):
                jp[:, :, i] += (off + amp * np.sin(
                    2 * np.pi * t / per + ph + n * spread))[:, None]
        elif gr.mode == "hold":
            for i in idxs:
                jp[:, :, i] += off
        elif gr.mode == "burst":
            peak = _clip(gr.peak_radps, 0.0, 15.0)
            cnt = int(_clip(gr.burst_count, 0, 8)) or 3
            bp = max(T // cnt, _BURST_WIN + 1)
            for c in range(cnt):
                s0 = c * bp + 10
                for i in idxs:
                    burst_jv[s0:s0 + _BURST_WIN, :, i] += peak
        elif gr.mode == "fold":
            # flex one direction and return, IN PHASE with the pelvis dip (the same
            # 0→1→0 arc): off → off+amp at mid → off — a fold/squat/toe-touch/bow
            # posture (joint ROM = amplitude_rad). phase/spread/period are
            # intentionally unused: a fold is one synchronized posture arc over the
            # whole rollout, not a periodic motion, so it stays coherent with the dip.
            for i in idxs:
                jp[:, :, i] += (off + amp * _FOLD_ARC)[:, None]

    # degradation: tremor (high-freq, on group joints or whole body) + noise.
    tremor = _clip(spec.tremor, 0.0, 2.0)
    if tremor > 0:
        targets = sorted({i for gr in spec.groups[:8]
                          for i in _resolve_targets(joint_names, gr.role_query)})
        trem = (tremor * np.sin(2 * np.pi * t / 3.0))[:, None, None]
        if targets:
            jp[:, :, targets] += trem
        else:
            jp += trem
    noise = _clip(spec.noise, 0.0, 0.2)
    if noise > 0:
        rng = np.random.default_rng(1000 + rung_index)
        jp += rng.normal(0.0, noise, (T, E, J))

    # consistency: joint_pos = integral of joint_vel.
    jv = burst_jv + np.gradient(jp, axis=0) / DT
    jp = jp + np.cumsum(burst_jv, axis=0) * DT

    arrays = {"joint_pos": jp, "joint_vel": jv,
              "projected_gravity_b": g, "root_link_pos_w": root}
    meta = {"joint_names": list(joint_names),
            "groups_resolved_counts": resolved_counts,
            "coordination_dropped": coord_dropped}
    return arrays, dict(_BEHAVIOR), meta


# The renderer-built degenerate anchor (rung 0): FALLEN (uprightness 0),
# jittering, no structured motion, no travel — a universal incompetent floor.
# Fully fallen so every uprightness-gated metric crushes it to ~0 (large,
# clean separation from the competent top rung), while a flail/motion rewarder
# still scores the tremor high and so FAILS the separation gate — exactly the
# metric we want to deny steer-rights.
_ANCHOR = MotionSpec(uprightness=0.0, base_height_m=0.5, tremor=1.6)


def _feature_vector(arrays: dict, meta: dict) -> np.ndarray:
    """A small physical fingerprint of a rendered rung, to detect a
    degenerate (near-identical-rungs) ladder."""
    root = arrays["root_link_pos_w"]
    g = arrays["projected_gravity_b"]
    jp = arrays["joint_pos"]
    fwd = float(root[-1, :, 0].mean() - root[0, :, 0].mean())
    lat = float(root[-1, :, 1].mean() - root[0, :, 1].mean())
    z = float(root[..., 2].mean())
    up = float((g[..., 2] < -0.85).mean())
    amp = float((jp.max(axis=0) - jp.min(axis=0)).mean())
    return np.array([fwd, lat, z, up, amp], dtype=np.float64)


def render_ladder(
    specs: list[MotionSpec], joint_names: list[str],
) -> dict[str, Any]:
    """Render an author's ascending rungs with the renderer-built degenerate
    anchor PREPENDED (rung 0). Returns
    `{rungs: [(arrays, behavior, meta)], n, degenerate, reason,
      groups_resolved_counts, feature_vectors}`. Never raises."""
    full = [_ANCHOR] + list(specs or [])
    rendered: list[tuple] = []
    feats: list[np.ndarray] = []
    counts: dict[str, int] = {}
    any_degenerate_axis = any(getattr(s, "degenerate_axis", False) for s in (specs or []))
    zero_joint_group = False
    for i, sp in enumerate(full):
        arrays, beh, meta = render_rung(sp, joint_names, rung_index=i)
        rendered.append((arrays, beh, meta))
        feats.append(_feature_vector(arrays, meta))
        for name, n in meta.get("groups_resolved_counts", {}).items():
            counts[name] = n
            if n == 0:
                zero_joint_group = True

    reason = None
    degenerate = False
    if any_degenerate_axis:
        degenerate = True
        reason = ("ladder_degenerate: author marked the competence axis "
                  "inexpressible from physical arrays (yaw/contacts/forces)")
    elif len(full) < 4:
        degenerate = True
        reason = "ladder_degenerate: too few rungs to rank"
    else:
        fmat = np.array(feats)
        # distinct rungs?
        distinct = len({tuple(np.round(f, 3)) for f in feats})
        maxd = 0.0
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                maxd = max(maxd, float(np.linalg.norm(feats[a] - feats[b])))
        if maxd < 1e-3 or distinct < 3:
            degenerate = True
            reason = ("ladder_degenerate: rendered rungs are physically "
                      "near-identical (competence axis not observable)")
        elif zero_joint_group:
            reason = "warning: a group resolved 0 joints on this robot"

    return {"rungs": rendered, "n": len(rendered), "degenerate": degenerate,
            "reason": reason, "groups_resolved_counts": counts,
            "zero_joint_group": zero_joint_group}
