"""Canonical joint resolver (§Ship 49 — always-correct joint identification).

Every objective metric ultimately reads a SUBSET of the (T, E, J) joint
arrays — the swing leg for a kick, the hips and arms for a floss. Before
Ship 49 that subset was chosen by ad-hoc lowercase SUBSTRING matching,
reinvented in every built-in (`spec_metrics._match_joints`) and every
generated metric. That has two silent failure modes, both reproduced on the
live g1-kick-v4 rollouts:

  * **Imprecise.** `("hip","knee","ankle")` grabs ALL 12 G1 leg joints —
    hip PITCH (the forward-kick DOF), hip ROLL (sideways abduction) and hip
    YAW alike. A metric averaging over that set scores a SIDEWAYS kick the
    same as a forward one (the g1-kick-v4 "kicks but sideways" gap).
  * **Unguarded.** A shuffled or foreign `joint_names` list still produces a
    plausible-looking score instead of failing — on real v4 rollouts a Go1
    name list scored a stuck G1 policy 0.34 while the correct names scored
    0.00.

This module is the ONE audited place that maps a robot's `joint_names`
(in the SAME column order as the joint_pos/joint_vel buffers — the adapter
order-contract, see tests/test_joint_resolver.py) to canonical FUNCTIONAL
ROLES, direction-aware (pitch / roll / yaw, left / right, front / rear).
A role either resolves to exactly one joint or is reported missing /
ambiguous — it never silently picks the wrong joint.

Two resolution surfaces:
  * `resolve_joint_roles(names, roles)` — each role → exactly one index, or
    flagged. The strict path used by the runtime + the validation gate.
  * `select_joints(names, segments=…, axes=…, sides=…)` — predicate group
    selection (e.g. every sagittal leg joint) for metrics that average over
    a set. Replaces `_match_joints`.

Naming is anchored on FUNCTION, not spelling: tokens are normalised through
synonym tables (`thigh`→hip-flexion stays `thigh`; `calf`/`shin`→`calf`;
`torso`/`spine`→`waist`) so the same role vocabulary spans MJCF/URDF/SMPL
variants. Cross-robot SEMANTIC equivalence (a Go1 `calf` ≈ a G1 `knee`)
is deliberately NOT collapsed here — that belongs to the per-robot manifest
(`robot_manifest.py`); the resolver stays purely structural so it can never
guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# ── canonical vocabulary ─────────────────────────────────────────────

#: Canonical body sides. Bipeds use left/right; quadrupeds add the
#: front/rear quadrant.
_CANON_SIDES = frozenset({
    "left", "right",
    "front_left", "front_right", "rear_left", "rear_right",
})

#: Fused single-token quadrant codes (Unitree Go1/Go2/B2, ANYmal HL/HR…).
_FUSED_SIDE = {
    "fl": "front_left", "fr": "front_right",
    "rl": "rear_left", "rr": "rear_right",
    "hl": "rear_left", "hr": "rear_right",     # H = hind
    "lf": "front_left", "rf": "front_right",
    "lh": "rear_left", "rh": "rear_right",
}

#: Raw name token → canonical segment. First matching token (in name order)
#: wins. Anchored on function: a Go1 `thigh` is the hip-flexion link, a
#: `calf`/`shin` the knee link — kept literal so the manifest, not the
#: resolver, decides cross-robot equivalence.
_SEGMENT_SYNONYMS = {
    "hip": "hip",
    "thigh": "thigh",
    "knee": "knee",
    "calf": "calf", "shin": "calf",
    "ankle": "ankle",
    "foot": "foot", "feet": "foot", "toe": "foot",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "wrist": "wrist",
    "waist": "waist", "torso": "waist", "spine": "waist", "trunk": "waist",
    "neck": "neck", "head": "neck",
}

#: Rotation-axis qualifiers.
_AXES = frozenset({"pitch", "roll", "yaw"})

#: Tokens that carry no anatomical meaning — dropped before parsing.
_NOISE_TOKENS = frozenset({"joint", "link", "the", "j"})

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokens(name: str) -> list[str]:
    """Lowercase a raw joint name and split into meaningful tokens."""
    raw = _TOKEN_SPLIT.split(str(name).lower())
    return [t for t in raw if t and t not in _NOISE_TOKENS]


@dataclass(frozen=True)
class JointDescriptor:
    """A joint name parsed into canonical (side, segment, axis). Any field
    may be None when the name does not encode it (a single-DOF joint has no
    axis; a centreline joint like `waist_pitch` has no side)."""

    side: Optional[str]
    segment: Optional[str]
    axis: Optional[str]
    raw: str = ""

    @property
    def resolved(self) -> bool:
        """A descriptor is usable iff it names at least a segment."""
        return self.segment is not None


def _detect_side(toks: Sequence[str]) -> Optional[str]:
    s = set(toks)
    # Fused quadrant code (fr/fl/rr/rl/hl/hr) — most specific, check first.
    for t in toks:
        if t in _FUSED_SIDE:
            return _FUSED_SIDE[t]
    # front/rear (+ hind/back) combined with left/right as separate tokens.
    front_rear = (
        "front" if "front" in s
        else "rear" if (s & {"rear", "hind", "back"}) else None
    )
    left_right = "left" if "left" in s else "right" if "right" in s else None
    if front_rear and left_right:
        return f"{front_rear}_{left_right}"
    if left_right:
        return left_right
    # Single-letter biped prefix (`L_hip`, `R_shoulder`) — lowest priority,
    # only when exactly one of l/r is present so it can't misfire.
    if "l" in s and "r" not in s:
        return "left"
    if "r" in s and "l" not in s:
        return "right"
    return None


def _detect_segment(toks: Sequence[str]) -> Optional[str]:
    for t in toks:
        seg = _SEGMENT_SYNONYMS.get(t)
        if seg is not None:
            return seg
    return None


def _detect_axis(toks: Sequence[str]) -> Optional[str]:
    for t in toks:
        if t in _AXES:
            return t
    return None


def parse_joint_name(name: str) -> JointDescriptor:
    """Parse one raw joint name into its canonical (side, segment, axis)."""
    toks = _tokens(name)
    return JointDescriptor(
        side=_detect_side(toks),
        segment=_detect_segment(toks),
        axis=_detect_axis(toks),
        raw=str(name),
    )


def parse_role(role: str) -> JointDescriptor:
    """Parse a requested ROLE string (e.g. `left_hip_pitch`, `right_knee`,
    `front_left_calf`) the same way a joint name is parsed. An unspecified
    field is a wildcard at match time."""
    return parse_joint_name(role)


def _axis_matches(role_axis: Optional[str], joint_axis: Optional[str]) -> bool:
    """Axis match rule. A role with no axis is a wildcard. A SINGLE-DOF joint
    (no axis qualifier in its name) is assumed to be the joint's primary
    flexion DOF and satisfies any requested axis — so a `left_ankle_pitch`
    role still maps to a robot that exposes a plain `left_ankle`. A joint
    that DOES name an axis must match it exactly, which is what keeps a
    `hip_pitch` role off a `hip_roll` joint."""
    if role_axis is None or joint_axis is None:
        return True
    return role_axis == joint_axis


def _candidates(
    descriptors: Sequence[JointDescriptor], role: JointDescriptor, *,
    lenient: bool = False,
) -> list[int]:
    """Indices of joints matching `role`. `lenient` drops the axis constraint
    — used only when scoring the synthetic archetype battery on the
    fixed 12-joint biped, which has no roll/yaw columns, so an anatomically
    valid roll/yaw role can still run there."""
    out: list[int] = []
    for i, d in enumerate(descriptors):
        if role.side is not None and d.side != role.side:
            continue
        if role.segment is not None and d.segment != role.segment:
            continue
        if not lenient and not _axis_matches(role.axis, d.axis):
            continue
        out.append(i)
    return out


@dataclass
class RoleResolution:
    """Outcome of resolving a set of roles against a robot's joint names."""

    resolved: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    ambiguous: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.ambiguous

    def problems(self) -> list[str]:
        msgs: list[str] = []
        for r in self.missing:
            msgs.append(f"role {r!r} matches no joint")
        for r, idxs in self.ambiguous.items():
            msgs.append(f"role {r!r} is ambiguous (matches indices {idxs})")
        return msgs


class JointResolutionError(ValueError):
    """A required role could not be resolved to exactly one joint."""


def resolve_joint_roles(
    names: Sequence[str], roles: Iterable[str], *, lenient: bool = False,
) -> RoleResolution:
    """Map each requested role to the index of the SINGLE joint that fills it.

    A role that matches no joint is `missing`; a role that matches several
    (e.g. `left_hip` on a robot exposing hip pitch/roll/yaw) is `ambiguous`
    — the caller must name the axis. Never raises; inspect `.ok` /
    `.problems()`. The order-contract (names aligned with the joint axis) is
    the caller's to uphold — see `assert_name_axis_contract`."""
    descriptors = [parse_joint_name(n) for n in names]
    res = RoleResolution()
    for role in roles:
        rd = parse_role(role)
        # A role must name at least a segment — an unparseable role (no
        # recognised body part) would otherwise wildcard-match every joint
        # and read as "ambiguous"; report it as missing/invalid instead.
        if rd.segment is None:
            res.missing.append(role)
            continue
        cand = _candidates(descriptors, rd, lenient=lenient)
        if not cand:
            res.missing.append(role)
        elif len(cand) > 1:
            res.ambiguous[role] = cand
        else:
            res.resolved[role] = cand[0]
    return res


def resolve_or_raise(
    names: Sequence[str], roles: Iterable[str], *, lenient: bool = False,
) -> dict[str, int]:
    """Strict resolver: return `{role: index}` or raise
    `JointResolutionError` listing every missing/ambiguous role."""
    res = resolve_joint_roles(names, roles, lenient=lenient)
    if not res.ok:
        raise JointResolutionError("; ".join(res.problems()))
    return res.resolved


def select_joints(
    names: Sequence[str], *,
    segments: Optional[Iterable[str]] = None,
    axes: Optional[Iterable[str]] = None,
    sides: Optional[Iterable[str]] = None,
) -> list[int]:
    """Predicate group selection — every joint whose canonical fields fall in
    the given sets. `None` for a field is a wildcard. `axes` MAY include
    `None` to admit single-DOF (axis-less) joints alongside named axes — e.g.
    `axes={"pitch", None}` selects the sagittal-plane leg joints (hip pitch,
    knee, ankle pitch) while excluding hip roll/yaw. Replaces the old
    substring `_match_joints`; returns sorted indices."""
    seg_set = set(segments) if segments is not None else None
    side_set = set(sides) if sides is not None else None
    axis_set = set(axes) if axes is not None else None
    out: list[int] = []
    for i, n in enumerate(names):
        d = parse_joint_name(n)
        if seg_set is not None and d.segment not in seg_set:
            continue
        if side_set is not None and d.side not in side_set:
            continue
        if axis_set is not None and d.axis not in axis_set:
            continue
        out.append(i)
    return out


def assert_name_axis_contract(names: Sequence[str], n_joints: int) -> None:
    """The invariant every name-based metric silently relies on: the
    `joint_names` list is exactly as long as the joint axis of the
    (T, E, J) buffers (and therefore in the same column order — the adapter
    persists both from the entity's data API; see _mjlab_runner ~920). A
    mismatch means the names cannot be trusted to index the buffers, so a
    caller must DEGRADE (drop names) rather than silently mis-score. Raises
    `JointResolutionError` on violation."""
    if len(names) != n_joints:
        raise JointResolutionError(
            f"joint_names length {len(names)} != joint-axis width {n_joints} "
            f"— names are not aligned with the buffer columns; refusing to "
            f"index joints by an unverified name list"
        )


# ── named role groups the built-in metrics select ───────────────────
#: Sagittal-plane leg joints: hip pitch + knee + ankle pitch (and any
#: single-DOF hip/knee/ankle/thigh/calf), EXCLUDING hip roll/yaw and ankle
#: roll. This is the forward-kick / forward-gait DOF set — selecting it is
#: what makes a kick metric score a forward kick above a sideways one.
LEG_SAGITTAL_SEGMENTS = ("hip", "thigh", "knee", "calf", "ankle")
LEG_SAGITTAL_AXES = ("pitch", None)

#: Hip joints (any axis) — the floss "swing the hips" group.
HIP_SEGMENTS = ("hip",)
#: Arm joints — shoulder + elbow, the floss "arms in opposition" group.
ARM_SEGMENTS = ("shoulder", "elbow")


def leg_sagittal_indices(names: Sequence[str]) -> list[int]:
    """Forward (sagittal-plane) leg joints — see LEG_SAGITTAL_*."""
    return select_joints(
        names, segments=LEG_SAGITTAL_SEGMENTS, axes=LEG_SAGITTAL_AXES)


def hip_indices(names: Sequence[str]) -> list[int]:
    return select_joints(names, segments=HIP_SEGMENTS)


def arm_indices(names: Sequence[str]) -> list[int]:
    return select_joints(names, segments=ARM_SEGMENTS)
