"""sculptor/modes.py — stages as OGMP modes with explicit transition guards.

The lab review's framework (docs/RESEARCH_DIRECTION.md §4; OGMP, arXiv
2403.04205) formalizes a multi-phase behavior as a **hybrid automaton**: a set
of *modes*, each bundling one sub-behavior, connected by *transitions* whose
*guards* say when control hands over. The reviewer's point was that this repo's
stage decomposition had rediscovered that structure informally — stages are
modes, but the handover between them was never written down.

Writing it down is what this module adds, and it buys three things the implicit
version cannot:

1. **Per-mode reward scope.** A term authored for "launch" should not be paid
   during "land". A mode owns a phase window, so a reward can be gated to it
   instead of every term competing over the whole episode — which is precisely
   the interference that makes a single scalar reward fight itself.
2. **Per-mode and per-transition gating.** The objective-metric gauntlet can
   run against each mode and each guard separately, so a degenerate sub-motion
   is caught where it happens rather than being averaged away in an
   episode-level score.
3. **An explicit failure surface.** A guard that never fires is a *diagnosable*
   event ("the policy never left approach"), where an implicit stage boundary
   just yields a low score with no stated reason.

The natural source is a composed reference (`sculptor.refs.compose`): its
provenance already records, per segment, which clip contributed which frames
and where the seams landed. One composed segment is one mode, and the seam
between two segments is exactly the transition between them — so
`modes_from_composition` reads a hybrid automaton straight out of a clip that
already exists, rather than asking an LLM to invent one.

Scope, stated honestly: this module defines and validates the structure and
derives it from a composition. It does **not** author per-mode reward code or
run the metric gauntlet per mode — those consume this schema and are separate
work (see HANDOFF §9). Guards here are *phase* predicates (time/progress
through the reference), which is what a composed reference can support;
state-predicate guards are a superset this schema leaves room for via
`guard.kind`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: Guard kinds this schema understands. `phase` is derivable from a composed
#: reference today; `predicate` is the escape hatch for a state-space
#: condition an author writes explicitly. Unknown kinds are rejected rather
#: than ignored — a guard nobody evaluates is worse than a missing one.
GUARD_KINDS = ("phase", "predicate")

MODES_SCHEMA_VERSION = 1


class ModeError(ValueError):
    """Raised when a mode/transition structure is not a valid automaton."""


@dataclass(frozen=True)
class Guard:
    """When control may leave a mode.

    `kind="phase"`: fires once normalized progress through the mode's own
    reference window reaches `at_phase` (0-1).
    `kind="predicate"`: fires when `expression` evaluates true. The expression
    is NOT evaluated here — it is carried for the same isolated evaluator that
    already runs mission success criteria, and must never be eval'd inline.
    """

    kind: str = "phase"
    at_phase: Optional[float] = None
    expression: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items()
                if v is not None}


@dataclass(frozen=True)
class Mode:
    """One bundled sub-behavior.

    `frame_range` is a half-open `[start, end)` window into the mode's
    reference clip — the frames this mode is responsible for tracking.
    """

    name: str
    frame_range: tuple[int, int]
    reference_clip_id: Optional[str] = None
    source_clip_id: Optional[str] = None
    reward_terms: tuple[str, ...] = ()
    success_predicate: Optional[str] = None

    @property
    def n_frames(self) -> int:
        return int(self.frame_range[1] - self.frame_range[0])

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["frame_range"] = list(self.frame_range)
        d["reward_terms"] = list(self.reward_terms)
        return d


@dataclass(frozen=True)
class Transition:
    """A directed handover between two modes."""

    from_mode: str
    to_mode: str
    guard: Guard = field(default_factory=Guard)

    def to_dict(self) -> dict[str, Any]:
        return {"from_mode": self.from_mode, "to_mode": self.to_mode,
                "guard": self.guard.to_dict()}


@dataclass(frozen=True)
class ModeGraph:
    """A validated hybrid automaton over one behavior."""

    modes: tuple[Mode, ...]
    transitions: tuple[Transition, ...]
    fps: float
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def entry(self) -> Mode:
        return self.modes[0]

    @property
    def terminal(self) -> Mode:
        return self.modes[-1]

    def mode(self, name: str) -> Mode:
        for m in self.modes:
            if m.name == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODES_SCHEMA_VERSION,
            "fps": self.fps,
            "modes": [m.to_dict() for m in self.modes],
            "transitions": [t.to_dict() for t in self.transitions],
            "source": dict(self.source),
        }


def validate_mode_graph(graph: ModeGraph) -> list[str]:
    """Every structural violation at once (mirrors `validate_clip` /
    `validate_env_spec` style, so a generator retry gets complete feedback).

    Checks the properties that make the automaton *runnable*: unique named
    modes, non-empty ordered frame windows, transitions that reference real
    modes, well-formed guards, and — the load-bearing one — that every mode
    after the entry is actually reachable. An unreachable mode is a silent
    dead end: its reward is authored, gated, and never paid.
    """
    errors: list[str] = []
    if not graph.modes:
        return ["mode graph has no modes"]

    seen: set[str] = set()
    for i, m in enumerate(graph.modes):
        if not m.name or not isinstance(m.name, str):
            errors.append(f"mode[{i}]: name must be a non-empty string")
        elif m.name in seen:
            errors.append(f"mode[{i}]: duplicate mode name {m.name!r}")
        else:
            seen.add(m.name)
        lo, hi = m.frame_range
        if int(hi) <= int(lo):
            errors.append(
                f"mode {m.name!r}: frame_range must be ordered and non-empty, "
                f"got [{lo}, {hi})")
        if int(lo) < 0:
            errors.append(f"mode {m.name!r}: frame_range start must be >= 0")

    if graph.fps <= 0:
        errors.append(f"fps must be positive, got {graph.fps}")

    for i, t in enumerate(graph.transitions):
        if t.from_mode not in seen:
            errors.append(
                f"transition[{i}]: from_mode {t.from_mode!r} is not a mode")
        if t.to_mode not in seen:
            errors.append(
                f"transition[{i}]: to_mode {t.to_mode!r} is not a mode")
        if t.from_mode == t.to_mode:
            errors.append(
                f"transition[{i}]: self-transition on {t.from_mode!r}")
        g = t.guard
        if g.kind not in GUARD_KINDS:
            errors.append(
                f"transition[{i}]: unknown guard kind {g.kind!r} "
                f"(expected one of {list(GUARD_KINDS)})")
        elif g.kind == "phase":
            if g.at_phase is None:
                errors.append(
                    f"transition[{i}]: phase guard requires 'at_phase'")
            elif not (0.0 < float(g.at_phase) <= 1.0):
                errors.append(
                    f"transition[{i}]: at_phase must be in (0, 1], "
                    f"got {g.at_phase}")
        elif g.kind == "predicate" and not (g.expression or "").strip():
            errors.append(
                f"transition[{i}]: predicate guard requires an expression")

    # Reachability. A mode nobody can enter is a dead end whose reward is
    # authored and gated but never paid — the exact silent failure this
    # formalization exists to surface.
    reachable = {graph.modes[0].name}
    changed = True
    while changed:
        changed = False
        for t in graph.transitions:
            if t.from_mode in reachable and t.to_mode not in reachable:
                reachable.add(t.to_mode)
                changed = True
    for m in graph.modes[1:]:
        if m.name not in reachable:
            errors.append(
                f"mode {m.name!r} is unreachable from the entry mode "
                f"{graph.modes[0].name!r}")
    return errors


def modes_from_composition(
    clip: Mapping[str, Any],
    *,
    clip_id: Optional[str] = None,
    guard_at_phase: float = 1.0,
) -> ModeGraph:
    """Read a hybrid automaton out of a composed reference clip.

    A composite's provenance already records the ordered segments and the seam
    frames between them, so the automaton is *derived*, not invented: one
    segment is one mode, and each seam is the transition between the pair it
    separates. Mode frame windows are the composed clip's own frames, so a
    per-mode tracking reward can index the reference directly.

    `guard_at_phase` defaults to 1.0 — hand over at the end of the mode's
    window, matching what the cross-fade already does physically. Lower it to
    hand over early (a guard that fires mid-window overlaps the modes, which
    is legal and sometimes what a blend wants).

    Raises `ModeError` when `clip` carries no composition provenance, or when
    the derived graph does not validate.
    """
    meta = clip.get("meta") or {}
    comp = meta.get("composition")
    if not isinstance(comp, Mapping):
        raise ModeError(
            "clip has no meta.composition — modes_from_composition reads a "
            "COMPOSED reference (see sculptor.refs.compose). For a single-clip "
            "stage there is one mode and no transition to derive.")
    segments = comp.get("segments") or []
    if len(segments) < 2:
        raise ModeError(
            f"composition records {len(segments)} segment(s); a mode graph "
            "needs at least 2")

    seams = [int(s) for s in (comp.get("seam_frames") or [])]
    n_frames = int(len(clip["root_pos_z"]))
    # Segment i spans [boundary[i], boundary[i+1]) in the COMPOSED clip.
    boundaries = [0, *seams, n_frames]
    if len(boundaries) != len(segments) + 1:
        raise ModeError(
            f"composition has {len(segments)} segments but {len(seams)} seam "
            "frame(s); expected exactly one fewer seam than segments")

    modes: list[Mode] = []
    used: set[str] = set()
    for i, seg in enumerate(segments):
        raw = str(seg.get("label") or f"mode_{i + 1}")
        name = raw.strip().lower().replace(" ", "_") or f"mode_{i + 1}"
        # Labels come from user input and are not required to be unique.
        base, k = name, 2
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        modes.append(Mode(
            name=name,
            frame_range=(boundaries[i], boundaries[i + 1]),
            reference_clip_id=clip_id,
            source_clip_id=seg.get("source_id"),
        ))

    transitions = [
        Transition(from_mode=modes[i].name, to_mode=modes[i + 1].name,
                   guard=Guard(kind="phase", at_phase=float(guard_at_phase)))
        for i in range(len(modes) - 1)
    ]

    graph = ModeGraph(
        modes=tuple(modes), transitions=tuple(transitions),
        fps=float(clip["fps"]),
        source={
            "kind": "composition",
            "clip_id": clip_id,
            "parent_clip_ids": [m.source_clip_id for m in modes],
            "seam_frames": seams,
            "derived_by": "sculptor.modes.modes_from_composition",
        },
    )
    errors = validate_mode_graph(graph)
    if errors:
        raise ModeError(
            "derived mode graph is invalid:\n  - " + "\n  - ".join(errors))
    return graph


def mode_at_frame(graph: ModeGraph, frame: int) -> Mode:
    """Which mode owns `frame`. Frames past the end belong to the terminal
    mode — an episode running long is still *in* the last mode, not outside
    the automaton."""
    for m in graph.modes:
        if m.frame_range[0] <= frame < m.frame_range[1]:
            return m
    return graph.terminal if frame >= graph.terminal.frame_range[0] \
        else graph.entry


def mode_phase_windows(graph: ModeGraph) -> dict[str, tuple[float, float]]:
    """Per-mode `(start_s, end_s)` in seconds — the window a per-mode reward
    term should be paid over, and the window a per-mode metric should score."""
    return {
        m.name: (round(m.frame_range[0] / graph.fps, 4),
                 round(m.frame_range[1] / graph.fps, 4))
        for m in graph.modes
    }
