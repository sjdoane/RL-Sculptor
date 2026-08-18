"""sculptor/modes.py — an OGMP-inspired phase graph over a fixed reference.

The distinction in that title is deliberate. Original OGMP (arXiv 2403.04205)
uses an online, closed-loop, receding-horizon oracle; rho-bounded permissible
state exploration; task-parameterized modes; and one policy conditioned on a
learned mode latent, clock, and task feedback. This module implements none of
those mechanisms. It derives a finite set of **clip-time phase windows** from
the seams of one composed, open-loop reference. The generated reward dispatches
by episode time. Transition guards are validated and reportable metadata, not
the runtime handover authority or a policy observation.

That smaller abstraction is still useful, and writing it down buys three
things the previous implicit stage decomposition could not:

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

Scope, stated honestly: this module defines, validates, and derives the phase
graph. ``sculptor.mode_rewards`` consumes it to generate a time-windowed reward
module. Production rollouts persist digest-bound, diagnostic-only per-window
evidence through ``sculptor.eval.mode_metrics``; the generated/validated/
calibrated per-mode objective gauntlet remains an explicit research workflow,
not a fitness or selection authority. Predicate guards are an
extension point carried to the isolated criterion evaluator; the generated
training reward does not execute them. Any UI/API using this schema must expose
those capability limits rather than calling the result paper-faithful OGMP.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: Guard kinds this schema understands. `phase` is derivable from a composed
#: reference today; `predicate` is the escape hatch for a state-space
#: condition an author writes explicitly. Unknown kinds are rejected rather
#: than ignored — a guard nobody evaluates is worse than a missing one.
GUARD_KINDS = ("phase", "predicate")

MODES_SCHEMA_VERSION = 1
MODE_EXECUTION_SCHEMA_VERSION = 2
LEGACY_MODE_EXECUTION_SCHEMA_VERSION = 1

#: The generated reward reads ``episode_length * step_dt`` independently for
#: each vectorized environment.  Naming that clock in the manifest keeps an
#: evaluator from silently interpreting the same numeric windows as clip time.
PER_ENV_EPISODE_ELAPSED_S = "per_env_episode_elapsed_s"


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


@dataclass(frozen=True)
class ModeExecutionManifest:
    """The exact phase schedule a generated reward executes.

    A :class:`ModeGraph` owns reference-frame ranges at the reference's
    certified cadence.  Those ranges are never silently stretched to consume a
    longer episode: unused budget is represented as an explicit terminal-hold
    mode.  This immutable record is the join between authoring and execution:
    exact emitted windows, their order and clock, plus a digest of the graph
    they came from.  Evaluation therefore scores the same schedule training
    executed instead of re-deriving or retiming it.

    ``windows_s`` is a tuple rather than a mutable mapping so a caller cannot
    change the evaluation schedule after validating it.  Version 2 also names
    the certified clip duration and the post-clip terminal hold separately.
    That distinction is load-bearing: extending only the terminal window is a
    hold, while stretching every window is an uncertified retiming.

    ``to_dict`` exposes the ergonomic mapping stored in ``REWARD_SPEC``.
    """

    mode_order: tuple[str, ...]
    windows_s: tuple[tuple[str, float, float], ...]
    time_basis: str
    graph_sha256: str
    certified_clip_duration_s: Optional[float] = None
    terminal_hold_s: float = 0.0
    schema_version: int = MODE_EXECUTION_SCHEMA_VERSION

    @property
    def window_map(self) -> dict[str, tuple[float, float]]:
        return {name: (lo, hi) for name, lo, hi in self.windows_s}

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "mode_order": list(self.mode_order),
            "windows_s": {
                name: [lo, hi] for name, lo, hi in self.windows_s
            },
            "time_basis": self.time_basis,
            "graph_sha256": self.graph_sha256,
        }
        # Preserve the normalized byte identity of schema-1 manifests. They
        # remain readable as zero-hold legacy schedules, but every newly built
        # manifest is schema 2 and records the distinction explicitly.
        if self.schema_version >= 2:
            value["certified_clip_duration_s"] = (
                self.certified_clip_duration_s
            )
            value["terminal_hold_s"] = self.terminal_hold_s
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModeExecutionManifest":
        """Parse the data-only form stored in a generated reward.

        Parsing is deliberately strict.  A partial/legacy object must not look
        like an authoritative schedule; callers can explicitly build a raw
        graph-time manifest instead.
        """
        try:
            order = tuple(str(name) for name in value["mode_order"])
            raw_windows = value["windows_s"]
            if not isinstance(raw_windows, Mapping):
                raise TypeError("windows_s must be a mapping")
            if set(raw_windows) != set(order):
                raise ValueError(
                    "windows_s keys must exactly match mode_order: "
                    f"got {sorted(str(k) for k in raw_windows)!r}, "
                    f"expected {sorted(order)!r}"
                )
            windows = tuple(
                (name, float(raw_windows[name][0]), float(raw_windows[name][1]))
                for name in order
            )
            schema_version = int(value["schema_version"])
            certified_duration = None
            terminal_hold = 0.0
            if schema_version >= 2:
                certified_duration = float(
                    value["certified_clip_duration_s"]
                )
                terminal_hold = float(value["terminal_hold_s"])
            return cls(
                mode_order=order,
                windows_s=windows,
                time_basis=str(value["time_basis"]),
                graph_sha256=str(value["graph_sha256"]),
                certified_clip_duration_s=certified_duration,
                terminal_hold_s=terminal_hold,
                schema_version=schema_version,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModeError(
                f"invalid mode execution manifest: {type(exc).__name__}: {exc}"
            ) from exc


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


def mode_graph_sha256(graph: ModeGraph) -> str:
    """Content digest of the complete, data-only mode graph.

    Sorting mapping keys makes the digest independent of insertion order while
    retaining mode/transition list order, which is semantically significant.
    Refusing non-JSON source metadata is safer than stringifying it into a hash
    that another process cannot reproduce.
    """
    try:
        payload = json.dumps(
            graph.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModeError(
            f"mode graph cannot be content-addressed: {type(exc).__name__}: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def build_mode_execution_manifest(
    graph: ModeGraph,
    *,
    windows_s: Optional[Mapping[str, tuple[float, float]]] = None,
    terminal_hold_s: float = 0.0,
    time_basis: str = PER_ENV_EPISODE_ELAPSED_S,
) -> ModeExecutionManifest:
    """Build and validate the exact schedule a reward/evaluator will execute."""
    order = tuple(mode.name for mode in graph.modes)
    raw_windows = mode_phase_windows(graph)
    try:
        terminal_hold_s = float(terminal_hold_s)
    except (TypeError, ValueError) as exc:
        raise ModeError(
            "cannot build mode execution manifest: terminal_hold_s must be "
            "a finite non-negative number"
        ) from exc
    if not math.isfinite(terminal_hold_s) or terminal_hold_s < 0.0:
        raise ModeError(
            "cannot build mode execution manifest: terminal_hold_s must be "
            f"finite and >= 0, got {terminal_hold_s!r}"
        )
    resolved = dict(windows_s or raw_windows)
    if windows_s is None and order and terminal_hold_s > 0.0:
        terminal_name = order[-1]
        lo, hi = resolved[terminal_name]
        resolved[terminal_name] = (lo, hi + terminal_hold_s)
    if set(resolved) != set(order):
        raise ModeError(
            "cannot build mode execution manifest: windows_s keys must exactly "
            f"match mode order; got {sorted(str(k) for k in resolved)!r}, "
            f"expected {sorted(order)!r}"
        )
    try:
        rows = tuple(
            (name, float(resolved[name][0]), float(resolved[name][1]))
            for name in order
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModeError(
            f"cannot build mode execution manifest: {type(exc).__name__}: {exc}"
        ) from exc
    manifest = ModeExecutionManifest(
        mode_order=order,
        windows_s=rows,
        time_basis=str(time_basis),
        graph_sha256=mode_graph_sha256(graph),
        certified_clip_duration_s=max(
            (float(hi) for _lo, hi in raw_windows.values()), default=0.0
        ),
        terminal_hold_s=terminal_hold_s,
    )
    errors = validate_mode_execution_manifest(manifest, graph)
    if errors:
        raise ModeError(
            "invalid mode execution manifest:\n  - " + "\n  - ".join(errors)
        )
    return manifest


def validate_mode_execution_manifest(
    manifest: ModeExecutionManifest, graph: ModeGraph,
) -> list[str]:
    """Every mismatch between an execution schedule and its claimed graph."""
    errors = validate_mode_graph(graph)
    if errors:
        return errors
    if manifest.schema_version not in {
        LEGACY_MODE_EXECUTION_SCHEMA_VERSION,
        MODE_EXECUTION_SCHEMA_VERSION,
    }:
        errors.append(
            f"execution manifest schema {manifest.schema_version} is unsupported; "
            f"expected {LEGACY_MODE_EXECUTION_SCHEMA_VERSION} or "
            f"{MODE_EXECUTION_SCHEMA_VERSION}"
        )
    if manifest.time_basis != PER_ENV_EPISODE_ELAPSED_S:
        errors.append(
            f"execution manifest time_basis {manifest.time_basis!r} is unsupported; "
            f"expected {PER_ENV_EPISODE_ELAPSED_S!r}"
        )

    expected_order = tuple(mode.name for mode in graph.modes)
    if manifest.mode_order != expected_order:
        errors.append(
            f"execution manifest mode_order {list(manifest.mode_order)!r} does "
            f"not match graph order {list(expected_order)!r}"
        )
    row_names = tuple(name for name, _lo, _hi in manifest.windows_s)
    if row_names != manifest.mode_order:
        errors.append(
            f"execution manifest windows are ordered {list(row_names)!r}, not "
            f"{list(manifest.mode_order)!r}"
        )
    for name, lo, hi in manifest.windows_s:
        if not math.isfinite(lo) or not math.isfinite(hi):
            errors.append(
                f"execution manifest window for {name!r} must be finite, got "
                f"({lo!r}, {hi!r})"
            )
        elif hi <= lo:
            errors.append(
                f"execution manifest window for {name!r} must be non-empty, got "
                f"({lo!r}, {hi!r})"
            )
    raw_windows = mode_phase_windows(graph)
    emitted = manifest.window_map
    terminal_name = expected_order[-1]
    raw_duration = max(
        (float(hi) for _lo, hi in raw_windows.values()), default=0.0
    )
    if manifest.schema_version >= 2:
        if (
            manifest.certified_clip_duration_s is None
            or not math.isfinite(manifest.certified_clip_duration_s)
            or not math.isclose(
                manifest.certified_clip_duration_s,
                raw_duration,
                abs_tol=1e-3,
            )
        ):
            errors.append(
                "execution manifest certified_clip_duration_s does not match "
                f"the graph: expected {raw_duration:g}, got "
                f"{manifest.certified_clip_duration_s!r}"
            )
        if (
            not math.isfinite(manifest.terminal_hold_s)
            or manifest.terminal_hold_s < 0.0
        ):
            errors.append(
                "execution manifest terminal_hold_s must be finite and >= 0, "
                f"got {manifest.terminal_hold_s!r}"
            )
    else:
        # A v1 manifest did not have a field capable of distinguishing a hold
        # from a retiming. Treat it as an exact certified-cadence/zero-hold
        # schedule. New rewards always write v2.
        if manifest.terminal_hold_s != 0.0:
            errors.append("schema-1 execution manifests cannot declare a hold")

    for name, (raw_lo, raw_hi) in raw_windows.items():
        emitted_lo, emitted_hi = emitted.get(name, (float("nan"),) * 2)
        expected_hi = raw_hi
        if name == terminal_name and manifest.schema_version >= 2:
            expected_hi = raw_hi + manifest.terminal_hold_s
        if (
            not math.isclose(emitted_lo, raw_lo, abs_tol=1e-3)
            or not math.isclose(emitted_hi, expected_hi, abs_tol=1e-3)
        ):
            expectation = (
                f"certified-cadence window ({raw_lo:g}, {raw_hi:g})"
                if name != terminal_name or manifest.schema_version < 2
                else "certified terminal window "
                f"({raw_lo:g}, {raw_hi:g}) plus explicit "
                f"{manifest.terminal_hold_s:g}s hold"
            )
            errors.append(
                f"execution manifest window for {name!r} is "
                f"({emitted_lo:g}, {emitted_hi:g}), not the {expectation}; "
                "materialize and "
                "re-certify a retimed reference instead"
            )
    expected_digest = mode_graph_sha256(graph)
    if manifest.graph_sha256 != expected_digest:
        errors.append(
            "execution manifest graph_sha256 does not match this graph: "
            f"expected {expected_digest}, got {manifest.graph_sha256}"
        )
    return errors


def validate_phase_window_execution_graph(graph: ModeGraph) -> list[str]:
    """Can the current elapsed-time reward execute this graph *exactly*?

    The schema intentionally represents richer automata for future controllers,
    but today's generated reward has no stateful transition executor: it moves
    through contiguous windows in declaration order.  Reject every graph whose
    transition semantics would otherwise be silently ignored.
    """
    errors = validate_mode_graph(graph)
    if errors:
        return errors

    expected_edges = [
        (graph.modes[i].name, graph.modes[i + 1].name)
        for i in range(len(graph.modes) - 1)
    ]
    actual_edges = [(t.from_mode, t.to_mode) for t in graph.transitions]
    if actual_edges != expected_edges:
        errors.append(
            "elapsed-time mode rewards require exactly one ordered, adjacent "
            f"transition per boundary: expected {expected_edges!r}, got "
            f"{actual_edges!r}"
        )

    for index, transition in enumerate(graph.transitions):
        if transition.guard.kind != "phase":
            errors.append(
                f"transition[{index}] {transition.from_mode!r}->"
                f"{transition.to_mode!r} uses {transition.guard.kind!r}; the "
                "elapsed-time reward cannot execute state-dependent guards"
            )
        elif transition.guard.at_phase is None or not math.isclose(
            float(transition.guard.at_phase), 1.0, rel_tol=0.0, abs_tol=1e-12,
        ):
            errors.append(
                f"transition[{index}] {transition.from_mode!r}->"
                f"{transition.to_mode!r} fires at phase "
                f"{transition.guard.at_phase!r}; the elapsed-time reward hands "
                "over only at the emitted window boundary (phase 1.0)"
            )

    for earlier, later in zip(graph.modes, graph.modes[1:]):
        if int(earlier.frame_range[1]) != int(later.frame_range[0]):
            errors.append(
                f"mode windows must be contiguous for elapsed-time execution: "
                f"{earlier.name!r} ends at frame {earlier.frame_range[1]} but "
                f"{later.name!r} starts at frame {later.frame_range[0]}"
            )
    return errors
