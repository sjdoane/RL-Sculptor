"""sculptor/eval/mode_metrics.py — the objective-metric gauntlet, PER MODE and
PER TRANSITION.

`sculptor.modes` writes the hybrid automaton down (OGMP, arXiv 2403.04205;
docs/RESEARCH_DIRECTION.md §4) and its docstring names the two things the
formalization is supposed to buy that the implicit stage decomposition could
not. This module is the half that collects them:

  * **Per-mode gating.** A metric authored for one mode scores ONLY that mode's
    slice of a rollout. An episode-level score averages a mode away: a
    five-mode behavior in which "grasp" is completely degenerate but the other
    four are fine still reports ~0.8, and nothing in the pipeline says which
    fifth is broken. Scored per mode it reports 0.0 on `grasp` — the failure is
    located, not diluted.
  * **An explicit failure surface.** A transition guard that never fires is a
    *statable* event ("the policy never left approach"), where an implicit
    stage boundary yields a low number with no stated cause.

This is not a hypothetical improvement. The iter-29 finding in HANDOFF §9 is a
29-iteration campaign that regressed to `order_ok_frac ~0.0` under a single
episode-level score, with each recovery iteration adding another global
constraint. Nothing in that loop could say *which part of the route* had
collapsed, because nothing scored a part.

One recorded clock is the common currency
-----------------------------------------
A generated reward publishes a `ModeExecutionManifest` whose windows are in
SECONDS on the exact per-environment episode clock training used. That is
load-bearing, not formatting: a 2 s reference may be followed by an explicit
18 s terminal hold, so re-deriving only raw clip windows would omit behavior
the terminal reward trained. A
`ModeGraph`'s `fps` is the REFERENCE rate (60 or 120 fps for retargeted mocap);
a rollout's time axis is the CONTROL rate (50 Hz on mjlab). Converting either
through frame COUNT requires assuming a rate at build time, and that assumption
has been wrong twice in this repo. So the rollout's own `step_dt` is READ,
never assumed, and the reward's manifest is consumed when available. Legacy
callers without a manifest retain raw graph-time behavior explicitly rather
than masquerading as evidence from a horizon-fitted reward.

What this module deliberately does NOT gate
-------------------------------------------
`check_transitions` reports whether each guard fired; it does not fail anything
on the answer. Guard firing is a property of the ROLLOUT (did the episode last
long enough / did the predicate hold), not of the METRIC, and this gauntlet
gates metrics. A short episode is a policy result to diagnose, not evidence
that a metric is bad. Likewise `coverage` — the fraction of a mode's window the
rollout actually reached — is REPORTED as a number and left ungated: there is
no calibration data behind any particular coverage floor, and inventing one
would be worse than showing the number (see the same reasoning behind
`TrackingErrors.orientation_err` in `refs/track.py`, which reports and does not
gate for exactly this reason).

The gating that does happen is the EXISTING gauntlet, re-pointed at one mode at
a time: `validate_generated_metric` with the mode's own goal and the mode's own
cropped reference, and `calibrate_task_derived` with the mode's own goal so the
gaming archetypes and the competence ladder are authored for the sub-behavior.
No threshold in `metric_validate` / `metric_axioms` / `metric_gen` is changed,
loosened, or special-cased from here, and no synthetic positive is added to the
fixed battery: a metric that is hard to game episode-wide can still be trivially
gamed inside one mode, and the way to find that out is to run the unmodified
gate against the mode, not to build a softer one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from sculptor.modes import (
    Mode,
    ModeError,
    ModeExecutionManifest,
    ModeGraph,
    Transition,
    build_mode_execution_manifest,
    mode_graph_sha256,
    mode_phase_windows,
    validate_mode_execution_manifest,
    validate_mode_graph,
)

#: Key under which a per-mode metric's artifacts are written by
#: `generate_mode_metrics`. One directory per mode keeps `metric.py` /
#: `meta.json` / `llm_calls.jsonl` at the exact paths every existing consumer
#: (metric_store, the UI, calibrate_task_derived) already expects.
MODE_DIR_PREFIX = "mode_"

#: Guard-firing verdicts. `None` is a THIRD state and is never collapsed into
#: False: "the predicate was not evaluated because no rollout namespace was
#: supplied" and "the predicate evaluated false" are different findings, and
#: silently reporting the first as the second is how an abstain becomes a
#: fabricated pass/fail.
FIRED_UNKNOWN = None

#: Persisted rollout evidence emitted by the production sculpt loop. This is
#: intentionally a diagnostics schema, not a success-certificate schema.
MODE_DIAGNOSTICS_SCHEMA_VERSION = 1


class ModeMetricError(ValueError):
    """Raised when a rollout cannot be resolved against a mode graph.

    Distinct from `modes.ModeError` (a malformed automaton) — by the time this
    module runs the graph has already validated; what fails here is the join
    between the graph and a concrete rollout (an unreadable clock, a frame
    range the clip cannot support).
    """


def resolve_mode_execution_manifest(
    graph: ModeGraph,
    execution_manifest: Optional[
        ModeExecutionManifest | Mapping[str, Any]
    ] = None,
) -> ModeExecutionManifest:
    """The validated schedule shared with the generated training reward.

    ``None`` preserves the pre-manifest/raw-clip behavior for older callers.
    Once a reward has been scaffolded, callers should pass its
    ``REWARD_SPEC['mode_execution_manifest']``; that is the only schedule known
    to match what training executed, including an explicit terminal hold.
    """
    try:
        manifest = (
            build_mode_execution_manifest(graph)
            if execution_manifest is None
            else ModeExecutionManifest.from_dict(execution_manifest)
            if isinstance(execution_manifest, Mapping)
            else execution_manifest
        )
        errors = validate_mode_execution_manifest(manifest, graph)
    except (ModeError, TypeError, ValueError) as exc:
        raise ModeMetricError(str(exc)) from exc
    if errors:
        raise ModeMetricError(
            "mode execution manifest does not match the graph:\n  - "
            + "\n  - ".join(errors)
        )
    return manifest


# ── the clock ────────────────────────────────────────────────────────────


def resolve_step_dt(
    behavior: Optional[Mapping[str, Any]] = None,
    *,
    step_dt: Optional[float] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> float:
    """Seconds per rollout frame, READ from the rollout — never assumed.

    Resolution order: an explicit `step_dt` argument, then `behavior["step_dt"]`
    (the key every adapter and every synthetic fixture in this package already
    publishes — see `ladder_synth._BEHAVIOR`, `refs.convert.clip_to_arrays`),
    then `meta["step_dt"]`.

    Raises `ModeMetricError` when none resolves. A default of 0.02 would be
    right for mjlab and wrong for a 60 fps reference-derived fixture, and the
    failure is invisible: every mode still gets a score, they are just the wrong
    slices, attributed to the wrong modes. This repo has twice shipped a phase
    clock that silently ran at the wrong rate (HANDOFF §10); refusing to guess
    is the cheapest way not to do it a third time.
    """
    for candidate, origin in (
        (step_dt, "step_dt argument"),
        ((behavior or {}).get("step_dt"), "behavior['step_dt']"),
        ((meta or {}).get("step_dt"), "meta['step_dt']"),
    ):
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError) as e:
            raise ModeMetricError(
                f"{origin} is not a number: {candidate!r}") from e
        if not math.isfinite(value) or value <= 0.0:
            raise ModeMetricError(
                f"{origin} must be a positive finite number, got {value!r}")
        return value
    raise ModeMetricError(
        "cannot resolve the rollout's step_dt: pass step_dt=..., or supply a "
        "behavior dict carrying 'step_dt'. A per-mode window is in SECONDS "
        "(the reference fps and the control rate differ), so it cannot be "
        "converted to frames without the rollout's own timestep — and "
        "defaulting one silently mis-attributes every mode's score.")


def frame_at(time_s: float, step_dt: float) -> int:
    """The rollout frame index a wall-clock offset lands on.

    `round`, not `floor`: a window boundary at 1.6 s with dt 0.02 is frame 80,
    and floating-point accumulation in the seconds (`mode_phase_windows` rounds
    to 4 decimals) otherwise drops it to 79 about half the time — a one-frame
    boundary jitter that makes two adjacent modes overlap or leave a hole
    depending on the clip's fps.
    """
    return int(round(float(time_s) / float(step_dt)))


# ── per-mode slices of a rollout ─────────────────────────────────────────


@dataclass(frozen=True)
class ModeSlice:
    """One mode's window resolved against a concrete rollout.

    `lo`/`hi` are the CLAMPED half-open frame bounds actually available in the
    rollout; `requested_hi` is where the window would have ended had the
    episode run long enough. Keeping both is what makes truncation visible: a
    slice of 40 frames is a very different finding depending on whether 40 or
    400 were asked for.
    """

    name: str
    start_s: float
    end_s: float
    lo: int
    hi: int
    requested_lo: int
    requested_hi: int
    rollout_frames: int

    @property
    def n_frames(self) -> int:
        return max(0, self.hi - self.lo)

    @property
    def entered(self) -> bool:
        """Did the rollout reach this mode at all? A mode with zero frames was
        never entered, which is a categorically different result from a mode
        that was entered and scored badly — see `score_modes`."""
        return self.n_frames > 0

    @property
    def shorter_than_one_step(self) -> bool:
        """Is this mode's window too short to contain a single control step?

        A 3-frame mode of a 120 fps reference is 0.025 s, which at a 50 Hz
        control rate is one step — and a 2-frame one is less. Such a mode
        cannot be scored at this control rate, and that is a DIFFERENT finding
        from "the policy never got here": one is a decomposition that the
        control rate cannot represent, the other is a policy that stalled.
        (Same family as the Nyquist finding in HANDOFF §10 — a 50 Hz controller
        cannot represent a 120 fps reference above 25 Hz.)
        """
        return self.requested_hi <= self.requested_lo

    @property
    def coverage(self) -> float:
        """Fraction of the requested window the rollout actually reached.

        REPORTED, never gated: there is no dataset behind any particular
        coverage floor, and a partially-covered mode is a diagnostic about the
        policy's episode length, not a defect in the metric being validated.
        """
        requested = max(1, self.requested_hi - self.requested_lo)
        return round(self.n_frames / requested, 4)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update({"n_frames": self.n_frames, "entered": self.entered,
                  "coverage": self.coverage,
                  "shorter_than_one_step": self.shorter_than_one_step})
        return d


def mode_slices(
    graph: ModeGraph, *, rollout_frames: int, step_dt: float,
    execution_manifest: Optional[
        ModeExecutionManifest | Mapping[str, Any]
    ] = None,
) -> list[ModeSlice]:
    """Resolve every mode's window against a rollout of `rollout_frames`.

    Windows come from the reward's execution manifest when supplied.  That
    distinction is load-bearing: a 2 s reference can have an 18 s terminal
    hold, and scoring only raw graph windows would omit the phase training
    paid. With no manifest, legacy callers retain raw
    graph-time behavior. Slices are clamped into `[0, rollout_frames]`; a mode
    whose window starts past the rollout yields an EMPTY slice rather than an
    error, because "the policy never got here" is a result to report.
    """
    if rollout_frames < 0:
        raise ModeMetricError(f"rollout_frames must be >= 0, got {rollout_frames}")
    step_dt = resolve_step_dt(step_dt=step_dt)
    manifest = resolve_mode_execution_manifest(graph, execution_manifest)
    windows = manifest.window_map
    out: list[ModeSlice] = []
    for name in manifest.mode_order:
        start_s, end_s = windows[name]
        req_lo, req_hi = frame_at(start_s, step_dt), frame_at(end_s, step_dt)
        lo = min(max(req_lo, 0), rollout_frames)
        hi = min(max(req_hi, lo), rollout_frames)
        out.append(ModeSlice(
            name=name, start_s=start_s, end_s=end_s,
            lo=lo, hi=hi, requested_lo=req_lo, requested_hi=req_hi,
            rollout_frames=rollout_frames))
    return out


def rollout_frames_of(arrays: Mapping[str, Any]) -> int:
    """Length of the rollout's time axis, taken from the arrays themselves.

    Every persisted metric array is `(T, E, ...)` or `(T, E)` with T first (the
    contract `spec_metrics` and `generated_metric` both document). Disagreeing
    lengths are a corrupt artifact, not something to silently min() over —
    scoring a mode against ragged channels would produce a number nobody can
    interpret.
    """
    lengths: set[int] = set()
    for value in arrays.values():
        try:
            array = np.asarray(value)
        except Exception:  # noqa: BLE001 — a non-array entry is not a channel
            continue
        if array.ndim >= 1:
            lengths.add(int(array.shape[0]))
    if not lengths:
        raise ModeMetricError("no time-indexed arrays to measure a rollout against")
    if len(lengths) > 1:
        raise ModeMetricError(
            f"rollout arrays disagree on their time axis: {sorted(lengths)}")
    return lengths.pop()


def slice_arrays(
    arrays: Mapping[str, Any], window: ModeSlice,
) -> dict[str, np.ndarray]:
    """The rollout arrays restricted to one mode's frames (axis 0)."""
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        out[key] = array[window.lo:window.hi] if array.ndim >= 1 else array
    return out


def slice_behavior(
    behavior: Mapping[str, Any], window: ModeSlice,
) -> dict[str, Any]:
    """The behavior dict a metric should see when scoring one mode's slice.

    `max_episode_steps` and `mean_episode_length` are overridden to the SLICE's
    own length. Leaving them at the episode's values silently breaks any metric
    that normalizes by episode length or reads a window as a fraction of it —
    it would compute "the last 20% of the episode" while looking at a slice
    that is 20% of the episode long, i.e. at the wrong frames entirely. The
    remaining keys (`rollout_num_envs`, `step_dt`, and any adapter extras) pass
    through unchanged.

    Honest boundary: episode-level aggregates a caller may have put in
    `behavior` (`mean_return`, `n_episodes`) still describe the WHOLE episode.
    They are not sliceable — a per-mode return is not recoverable from a scalar
    mean — so they are left alone rather than scaled into a fiction.
    """
    out = dict(behavior)
    out["max_episode_steps"] = window.n_frames
    if "mean_episode_length" in out:
        try:
            out["mean_episode_length"] = min(
                float(out["mean_episode_length"]), float(window.n_frames))
        except (TypeError, ValueError):
            out["mean_episode_length"] = window.n_frames
    return out


def score_modes(
    fn: Callable[..., Mapping[str, Any]],
    arrays: Mapping[str, Any],
    behavior: Mapping[str, Any],
    meta: Mapping[str, Any],
    graph: ModeGraph,
    *,
    step_dt: Optional[float] = None,
    metrics_by_mode: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
    execution_manifest: Optional[
        ModeExecutionManifest | Mapping[str, Any]
    ] = None,
) -> dict[str, Any]:
    """Score a rollout PER MODE — the whole point of the formalization.

    `fn` is the fallback metric (typically the episode-level one, so a caller
    can ask "what would this metric say about each mode?"). `metrics_by_mode`
    overrides it per mode, which is the real configuration: each mode's own
    generated metric scoring its own slice.

    Returns `{"episode": …, "modes": {name: entry}, "worst_mode": …,
    "worst_mode_gap": …, "step_dt": …}`. Each entry carries the mode's slice,
    its `spec_score`, the metric's full component dict, and — for a mode the
    rollout never reached — `scored: False` with `score: None`.

    **An unentered mode is not a zero.** Returning 0.0 for a mode the episode
    ended before reaching would make "the sub-behavior was performed and it was
    degenerate" indistinguishable from "the sub-behavior never happened", which
    are opposite diagnoses: the first says fix the reward for that mode, the
    second says the policy is stalling in an earlier one. `check_transitions`
    is the companion answer for why it never got there.

    `worst_mode_gap` = episode score − worst scored mode. It is a MEASUREMENT,
    deliberately ungated: a large gap is exactly the "a degenerate sub-motion
    averaged away by an episode score" signature this module exists to expose,
    but there is no evidence for where to draw a line on it, so the number is
    reported and the judgment is left to a human or to the per-mode gates.

    Never raises on a metric that crashes — a raising metric records an `error`
    and scores `nan`, mirroring `metric_validate._score`'s archetype error path
    so a broken mode cannot take down a report about the other four.
    """
    errors = validate_mode_graph(graph)
    if errors:
        raise ModeMetricError(
            "refusing to score against an invalid mode graph:\n  - "
            + "\n  - ".join(errors))
    dt = resolve_step_dt(behavior, step_dt=step_dt, meta=meta)
    n_frames = rollout_frames_of(arrays)
    manifest = resolve_mode_execution_manifest(graph, execution_manifest)
    slices = mode_slices(
        graph, rollout_frames=n_frames, step_dt=dt,
        execution_manifest=manifest,
    )

    def _run(metric, sub_arrays, sub_behavior) -> tuple[float, dict, Optional[str]]:
        try:
            out = metric(sub_arrays, sub_behavior, dict(meta))
        except Exception as e:  # noqa: BLE001 — one bad mode must not blank the report
            return float("nan"), {}, f"{type(e).__name__}: {e}"
        if not isinstance(out, Mapping) or "spec_score" not in out:
            return (float("nan"), {},
                    "metric did not return a mapping with spec_score")
        try:
            score = float(out.get("spec_score"))
        except (TypeError, ValueError):
            return float("nan"), {}, "spec_score is not a number"
        if not np.isfinite(score):
            # Recorded as an error rather than carried as a nan: a nan is not
            # valid JSON, and "the metric returned nan here" is a finding worth
            # a sentence, not a silent hole in the report.
            return float("nan"), dict(out), "spec_score is not finite"
        return score, dict(out), None

    episode_score, episode_components, episode_error = _run(
        fn, dict(arrays), dict(behavior))

    entries: dict[str, dict[str, Any]] = {}
    for window in slices:
        metric = (metrics_by_mode or {}).get(window.name, fn)
        entry: dict[str, Any] = {
            "slice": window.to_dict(),
            "scored": False,
            "score": None,
            "components": {},
            "error": None,
        }
        if window.shorter_than_one_step:
            entry["error"] = (
                f"mode {window.name!r} spans "
                f"{window.end_s - window.start_s:.4f}s, shorter than one "
                f"control step ({dt}s) — this decomposition cannot be scored "
                f"at this control rate; the mode boundary is finer than the "
                f"controller can resolve")
        elif not window.entered:
            entry["error"] = (
                f"mode {window.name!r} was never entered: its window starts at "
                f"{window.start_s:.3f}s (frame {window.requested_lo}) but the "
                f"rollout is {n_frames} frames ({n_frames * dt:.3f}s) long")
        elif metric is None:
            entry["error"] = f"no metric supplied for mode {window.name!r}"
        else:
            score, components, err = _run(
                metric, slice_arrays(arrays, window),
                slice_behavior(behavior, window))
            ok = bool(err is None and np.isfinite(score))
            entry.update({
                # `scored` is a plain Python bool, not numpy's — this record is
                # serialized into meta.json/report payloads, and `np.True_`
                # neither round-trips through json nor satisfies `is True`.
                "scored": ok,
                "score": float(score) if ok else None,
                "components": components,
                "error": err,
            })
        entries[window.name] = entry

    scored = {name: e["score"] for name, e in entries.items() if e["scored"]}
    worst_mode = min(scored, key=lambda k: scored[k]) if scored else None
    gap = (
        round(float(episode_score) - float(scored[worst_mode]), 6)
        if worst_mode is not None and np.isfinite(episode_score) else None
    )
    return {
        "episode": (float(episode_score) if np.isfinite(episode_score) else None),
        "episode_components": episode_components,
        "episode_error": episode_error,
        "modes": entries,
        "worst_mode": worst_mode,
        "worst_mode_gap": gap,
        "unentered_modes": [n for n, e in entries.items()
                            if not e["slice"]["entered"]],
        "step_dt": dt,
        "rollout_frames": n_frames,
        "mode_execution_manifest": manifest.to_dict(),
    }


# ── per-transition: did each guard actually fire? ────────────────────────


def guard_fire_time_s(
    graph: ModeGraph, transition: Transition,
    execution_manifest: Optional[
        ModeExecutionManifest | Mapping[str, Any]
    ] = None,
) -> Optional[float]:
    """Wall-clock offset at which a PHASE guard hands over, or None for a
    predicate guard (whose firing is a state condition, not a time).

    A phase guard's `at_phase` is normalized progress through the FROM mode's
    own window (`modes.Guard`), so the fire time is
    `start_s + at_phase · (end_s − start_s)` — not a fraction of the episode.
    """
    if transition.guard.kind != "phase" or transition.guard.at_phase is None:
        return None
    windows = resolve_mode_execution_manifest(
        graph, execution_manifest,
    ).window_map
    start_s, end_s = windows[transition.from_mode]
    return round(start_s + float(transition.guard.at_phase) * (end_s - start_s), 6)


def check_transitions(
    graph: ModeGraph,
    *,
    rollout_frames: int,
    step_dt: float,
    namespace: Optional[Mapping[str, Any]] = None,
    execution_manifest: Optional[
        ModeExecutionManifest | Mapping[str, Any]
    ] = None,
) -> list[dict[str, Any]]:
    """Did each transition guard ACTUALLY FIRE in this rollout?

    This is the "explicit failure surface" `modes.py` promises. Per transition:

      * **phase guard** — fired iff the rollout has a frame at or after the
        guard's fire time. Not fired is a directly statable diagnosis: *"the
        approach→grasp guard fires at 2.40 s but the rollout ended at 1.80 s —
        the policy never left approach."* Under an implicit stage boundary the
        same rollout produced a low episode number with no stated cause.
      * **predicate guard** — evaluated by `mission_runtime`'s ISOLATED
        evaluator, the same seccomp/rlimit worker that runs mission success
        criteria. `modes.Guard` states the expression must never be eval'd
        inline and this module does not; re-implementing an evaluator would
        fork a security boundary that currently has one audit surface.
        With no `namespace` the verdict is `None` (abstain) — never a silent
        pass or fail.

    **Deliberately not a gate.** Whether a guard fired is a fact about the
    ROLLOUT, not about a metric, and this package's gates judge metrics. A
    policy that stalls in mode 1 should be diagnosed, not used as evidence that
    mode 1's metric is bad. Callers wanting to fail a run on a stalled guard can
    do so from `fired`; nothing here decides it for them.

    Honest boundary worth stating plainly: a phase guard is a CLOCK. "Fired"
    means the episode lasted long enough to reach the handover time — it does
    NOT mean the sub-behavior succeeded. What the mode achieved is what
    `score_modes` measures; the two answer different questions and a report
    should show both.
    """
    errors = validate_mode_graph(graph)
    if errors:
        raise ModeMetricError(
            "refusing to check transitions against an invalid mode graph:\n  - "
            + "\n  - ".join(errors))
    dt = resolve_step_dt(step_dt=step_dt)
    duration_s = rollout_frames * dt
    manifest = resolve_mode_execution_manifest(graph, execution_manifest)
    windows = manifest.window_map
    out: list[dict[str, Any]] = []
    for index, transition in enumerate(graph.transitions):
        entry: dict[str, Any] = {
            "index": index,
            "from_mode": transition.from_mode,
            "to_mode": transition.to_mode,
            "guard": transition.guard.to_dict(),
            "kind": transition.guard.kind,
            "fired": FIRED_UNKNOWN,
            "reason": "",
            "rollout_frames": rollout_frames,
            "rollout_duration_s": round(duration_s, 6),
        }
        to_lo = frame_at(windows[transition.to_mode][0], dt)
        entry["to_mode_entered"] = to_lo < rollout_frames
        if transition.guard.kind == "phase":
            fire_s = guard_fire_time_s(
                graph, transition, execution_manifest=manifest,
            )
            fire_frame = frame_at(fire_s, dt) if fire_s is not None else None
            entry["fire_time_s"] = fire_s
            entry["fire_frame"] = fire_frame
            if fire_frame is None:
                entry["reason"] = (
                    "phase guard carries no at_phase — the graph should not "
                    "have validated; treating as not evaluable")
            else:
                fired = fire_frame < rollout_frames
                entry["fired"] = fired
                entry["reason"] = (
                    f"guard fires at {fire_s:.3f}s (frame {fire_frame}); the "
                    f"rollout is {rollout_frames} frames "
                    f"({duration_s:.3f}s) long — "
                    + ("reached" if fired else
                       f"NEVER REACHED: the policy never left "
                       f"{transition.from_mode!r}"))
        elif transition.guard.kind == "predicate":
            expression = (transition.guard.expression or "").strip()
            if namespace is None:
                entry["reason"] = (
                    f"predicate guard {expression!r} not evaluated: no rollout "
                    "namespace supplied (build one with "
                    "sculptor.mission_runtime._build_criterion_namespace) — "
                    "abstained, not failed")
            else:
                fired, reason = _evaluate_predicate_guard(expression, namespace)
                entry["fired"] = fired
                entry["reason"] = reason
        else:  # pragma: no cover — validate_mode_graph rejects unknown kinds
            entry["reason"] = f"unknown guard kind {transition.guard.kind!r}"
        out.append(entry)
    return out


def _evaluate_predicate_guard(
    expression: str, namespace: Mapping[str, Any],
) -> tuple[Optional[bool], str]:
    """Run one predicate guard through the mission-criterion sandbox.

    Imported lazily and deliberately: `sculptor.mission_runtime` reaches back
    into `sculptor.eval.generated_metric` for the hardened worker, so a
    module-level import here would close a cycle. The private name is used
    knowingly — there is exactly one isolated expression evaluator in this
    codebase and a second one would be a second thing to audit. (A public alias
    on `mission_runtime` would be the tidy fix; noted in HANDOFF rather than
    edited, since that file belongs to another workstream.)

    Returns `(fired, reason)`; `fired` is None when evaluation itself failed —
    an evaluator crash is absence of evidence, not a false guard.
    """
    if not expression:
        return None, "predicate guard has an empty expression"
    try:
        from sculptor.mission_runtime import _evaluate_success_criterion
    except Exception as e:  # noqa: BLE001 — no evaluator is an abstain
        return None, f"criterion evaluator unavailable: {type(e).__name__}: {e}"
    try:
        fired = bool(_evaluate_success_criterion(expression, dict(namespace)))
    except Exception as e:  # noqa: BLE001 — a failed eval is not a verdict
        return None, (f"predicate guard {expression!r} could not be evaluated: "
                      f"{type(e).__name__}: {e}")
    return fired, (
        f"predicate guard {expression!r} evaluated "
        + ("TRUE — the transition condition held"
           if fired else "FALSE — the transition condition never held"))


# ── per-mode goals, references, and prompt context ───────────────────────


def _canonical_json_sha256(value: Any) -> str:
    """Content digest for a finite, data-only JSON value."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModeMetricError(
            "mode diagnostics are not canonical JSON: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def mode_diagnostics_digest(record: Mapping[str, Any]) -> str:
    """Recompute the record digest (the digest field itself is excluded)."""
    payload = dict(record)
    payload.pop("diagnostic_digest", None)
    return _canonical_json_sha256(payload)


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ModeMetricError(f"{label} must be a lowercase SHA-256")
    return digest


def _first_episode_mask(
    arrays: Mapping[str, Any], behavior: Mapping[str, Any],
) -> np.ndarray:
    """Validate the all-lane, first-episode support of an official rollout.

    Phase windows run on a per-environment episode clock. The mjlab rollout
    freezes state after each lane's first termination and persists this mask;
    ignoring it would count frozen post-termination samples as time spent in a
    later mode. Production evidence therefore requires the mask rather than
    guessing episode boundaries from an aggregate behavior scalar.
    """
    raw = arrays.get("first_episode_valid_mask")
    if raw is None:
        raise ModeMetricError(
            "official phase diagnostics require first_episode_valid_mask; "
            "without it, per-environment resets cannot be separated"
        )
    mask_raw = np.asarray(raw)
    if mask_raw.ndim != 2 or not mask_raw.shape[0] or not mask_raw.shape[1]:
        raise ModeMetricError(
            "first_episode_valid_mask must have non-empty shape (T, E), got "
            f"{mask_raw.shape}"
        )
    if mask_raw.dtype != np.bool_:
        try:
            numeric = np.asarray(mask_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ModeMetricError(
                "first_episode_valid_mask is not boolean/0-or-1"
            ) from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ModeMetricError(
                "first_episode_valid_mask contains values other than 0 or 1"
            )
    mask = mask_raw.astype(bool, copy=False)
    # A first-episode mask is a prefix of True values for each lane. False to
    # True means reset samples were stitched into the same claimed episode.
    if mask.shape[0] > 1 and np.any((~mask[:-1]) & mask[1:]):
        raise ModeMetricError(
            "first_episode_valid_mask re-enters after termination; the "
            "trajectory stitches multiple episodes"
        )
    declared_envs = behavior.get("rollout_num_envs")
    if declared_envs is not None:
        try:
            declared_envs = int(declared_envs)
        except (TypeError, ValueError) as exc:
            raise ModeMetricError(
                "behavior['rollout_num_envs'] is not an integer"
            ) from exc
        if declared_envs != mask.shape[1]:
            raise ModeMetricError(
                "behavior rollout_num_envs disagrees with trajectory: "
                f"{declared_envs} != {mask.shape[1]}"
            )
    return mask


def _finite_masked_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Small, JSON-safe telemetry summary over valid lane/frame samples."""
    selected = np.asarray(values)[mask]
    if selected.size == 0:
        return {"sample_count": 0, "finite_fraction": None, "mean": None,
                "p10": None, "median": None, "p90": None,
                "min": None, "max": None}
    flat = np.asarray(selected, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    finite_fraction = float(finite.size / flat.size)
    if finite.size == 0:
        return {"sample_count": int(flat.size), "finite_fraction": 0.0,
                "mean": None, "p10": None, "median": None, "p90": None,
                "min": None, "max": None}
    return {
        "sample_count": int(flat.size),
        "finite_fraction": round(finite_fraction, 6),
        "mean": round(float(np.mean(finite)), 8),
        "p10": round(float(np.quantile(finite, 0.10)), 8),
        "median": round(float(np.median(finite)), 8),
        "p90": round(float(np.quantile(finite, 0.90)), 8),
        "min": round(float(np.min(finite)), 8),
        "max": round(float(np.max(finite)), 8),
    }


def build_mode_diagnostics(
    graph: ModeGraph,
    *,
    execution_manifest: ModeExecutionManifest | Mapping[str, Any],
    mode_binding: Mapping[str, Any],
    arrays: Mapping[str, Any],
    behavior: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build digest-bound all-lane phase diagnostics for one rollout.

    The record is deliberately not a calibrated success or selection
    authority. It states which fixed elapsed-time windows each first episode
    reached, whether each clock handover was reachable per lane, and the
    environment reward telemetry observed inside each window. This is the
    production evidence supported by today's phase-window scaffold; it does
    not claim an OGMP oracle, predicate executor, or mode-conditioned policy.

    Both joins fail closed: the execution manifest must validate against the
    supplied canonical graph snapshot, and the promoted reward binding must
    name the same graph and normalized manifest digest.
    """
    manifest = resolve_mode_execution_manifest(graph, execution_manifest)
    graph_digest = mode_graph_sha256(graph)
    try:
        from sculptor.mode_rewards import mode_execution_manifest_digest

        manifest_digest = mode_execution_manifest_digest(manifest.to_dict())
    except (ModeError, TypeError, ValueError) as exc:
        raise ModeMetricError(str(exc)) from exc

    required_binding_keys = {
        "schema", "clip_id", "robot", "clip_sha256", "graph_sha256",
        "context_refs", "execution_manifest_digest",
    }
    if not isinstance(mode_binding, Mapping):
        raise ModeMetricError("promoted phase reward is missing mode_binding")
    actual_binding_keys = set(mode_binding)
    if actual_binding_keys != required_binding_keys:
        raise ModeMetricError(
            "mode_binding fields are incomplete or unsupported: expected "
            f"{sorted(required_binding_keys)!r}, got "
            f"{sorted(str(k) for k in actual_binding_keys)!r}"
        )
    if mode_binding.get("schema") != 1:
        raise ModeMetricError("mode_binding schema must be 1")
    if mode_binding.get("graph_sha256") != graph_digest:
        raise ModeMetricError(
            "mode_binding graph_sha256 is stale: expected "
            f"{graph_digest}, got {mode_binding.get('graph_sha256')}"
        )
    if mode_binding.get("execution_manifest_digest") != manifest_digest:
        raise ModeMetricError(
            "mode_binding execution_manifest_digest is stale: expected "
            f"{manifest_digest}, got "
            f"{mode_binding.get('execution_manifest_digest')}"
        )
    clip_sha = _require_sha256(
        mode_binding.get("clip_sha256"), label="mode_binding clip_sha256"
    )
    if not str(mode_binding.get("clip_id") or ""):
        raise ModeMetricError("mode_binding clip_id is missing")
    if not str(mode_binding.get("robot") or ""):
        raise ModeMetricError("mode_binding robot is missing")
    context_refs = mode_binding.get("context_refs")
    if not isinstance(context_refs, Mapping):
        raise ModeMetricError("mode_binding context_refs must be a mapping")
    for kind, digest in context_refs.items():
        _require_sha256(digest, label=f"mode_binding context_refs[{kind!r}]")

    required_provenance = (
        "reward_sha256", "trajectory_sha256", "behavior_sha256",
        "reference_clip_sha256", "artifact_tuple_sha256",
    )
    normalized_provenance = dict(provenance)
    for key in required_provenance:
        normalized_provenance[key] = _require_sha256(
            normalized_provenance.get(key), label=f"provenance {key}"
        )
    if normalized_provenance["reference_clip_sha256"] != clip_sha:
        raise ModeMetricError(
            "provenance reference_clip_sha256 does not match mode_binding"
        )

    valid_mask = _first_episode_mask(arrays, behavior)
    rollout_frames, n_envs = map(int, valid_mask.shape)
    step_dt = resolve_step_dt(behavior)
    # The caller supplies only official time-indexed channels. A ragged
    # channel is corruption, not something to silently truncate to the mask.
    normalized_arrays: dict[str, np.ndarray] = {}
    for key, raw in arrays.items():
        array = np.asarray(raw)
        if array.ndim < 1:
            continue
        if array.shape[0] != rollout_frames:
            raise ModeMetricError(
                f"trajectory channel {key!r} has {array.shape[0]} frames, "
                f"expected {rollout_frames}"
            )
        normalized_arrays[str(key)] = array

    rendered_env = behavior.get("rendered_env_index")
    if rendered_env is not None:
        try:
            rendered_env = int(rendered_env)
        except (TypeError, ValueError) as exc:
            raise ModeMetricError(
                "behavior['rendered_env_index'] is not an integer"
            ) from exc
        if not 0 <= rendered_env < n_envs:
            raise ModeMetricError(
                f"rendered_env_index {rendered_env} is outside 0..{n_envs - 1}"
            )

    windows = mode_slices(
        graph,
        rollout_frames=rollout_frames,
        step_dt=step_dt,
        execution_manifest=manifest,
    )
    reward_terms = {
        key: array for key, array in normalized_arrays.items()
        if key.startswith("reward_term__")
    }
    mode_entries: dict[str, Any] = {}
    for window in windows:
        requested_frames = max(1, window.requested_hi - window.requested_lo)
        window_mask = valid_mask[window.lo:window.hi]
        valid_per_env = window_mask.sum(axis=0, dtype=np.int64)
        coverage_per_env = np.clip(
            valid_per_env.astype(np.float64) / requested_frames, 0.0, 1.0
        )
        entered = valid_per_env > 0
        completed = valid_per_env >= requested_frames
        term_summaries: dict[str, Any] = {}
        for key, array in reward_terms.items():
            if array.ndim < 2 or tuple(array.shape[:2]) != tuple(valid_mask.shape):
                raise ModeMetricError(
                    f"reward telemetry {key!r} must begin with (T, E)="
                    f"{tuple(valid_mask.shape)}, got {array.shape}"
                )
            term_summaries[key[len("reward_term__"):]] = _finite_masked_summary(
                array[window.lo:window.hi], window_mask
            )
        entry: dict[str, Any] = {
            "slice": window.to_dict(),
            "entered_env_count": int(np.count_nonzero(entered)),
            "entered_env_fraction": round(float(np.mean(entered)), 6),
            "completed_window_env_count": int(np.count_nonzero(completed)),
            "completed_window_env_fraction": round(float(np.mean(completed)), 6),
            "valid_frames_per_env": [int(v) for v in valid_per_env],
            "coverage_per_env": [round(float(v), 6) for v in coverage_per_env],
            "mean_env_coverage": round(float(np.mean(coverage_per_env)), 6),
            "reward_terms": term_summaries,
        }
        if rendered_env is not None:
            entry["rendered_env"] = {
                "index": rendered_env,
                "entered": bool(entered[rendered_env]),
                "completed_window": bool(completed[rendered_env]),
                "valid_frames": int(valid_per_env[rendered_env]),
                "coverage": round(float(coverage_per_env[rendered_env]), 6),
            }
        mode_entries[window.name] = entry

    transitions = check_transitions(
        graph,
        rollout_frames=rollout_frames,
        step_dt=step_dt,
        execution_manifest=manifest,
    )
    for transition in transitions:
        fire_frame = transition.get("fire_frame")
        per_env = np.zeros(n_envs, dtype=bool)
        if isinstance(fire_frame, int) and 0 <= fire_frame < rollout_frames:
            per_env = np.any(valid_mask[fire_frame:], axis=0)
        transition["recording_reached_guard"] = transition.pop("fired", None)
        transition["fired_env_count"] = int(np.count_nonzero(per_env))
        transition["fired_env_fraction"] = round(float(np.mean(per_env)), 6)
        transition["fired_per_env"] = [bool(v) for v in per_env]
        if rendered_env is not None:
            transition["rendered_env_fired"] = bool(per_env[rendered_env])

    from sculptor.kg.capabilities import unsupported_mode_diagnostic_keys

    record: dict[str, Any] = {
        "schema_version": MODE_DIAGNOSTICS_SCHEMA_VERSION,
        "authority": {
            "classification": "diagnostic_only",
            "calibrated_success_authority": False,
            "fitness_or_selection_authority": False,
            "execution_model": "fixed_linear_phase_windows",
            "not_implemented": unsupported_mode_diagnostic_keys(),
        },
        "provenance": normalized_provenance,
        "mode_binding": dict(mode_binding),
        "canonical_graph": graph.to_dict(),
        "graph_sha256": graph_digest,
        "mode_execution_manifest": manifest.to_dict(),
        "execution_manifest_digest": manifest_digest,
        "rollout": {
            "frames": rollout_frames,
            "environments": n_envs,
            "step_dt": step_dt,
            "first_episode_valid_fraction": round(float(np.mean(valid_mask)), 6),
            "rendered_env_index": rendered_env,
            "time_indexed_channels": sorted(normalized_arrays),
        },
        "modes": mode_entries,
        "transitions": transitions,
    }
    record["diagnostic_digest"] = mode_diagnostics_digest(record)
    return record


def mode_goal_text(
    mode: Mode, *, mode_goals: Optional[Mapping[str, str]] = None,
) -> str:
    """The behavior goal a mode's OWN metric is generated and validated against.

    Explicitly NOT the episode goal, and explicitly not the episode goal with
    the mode name prepended. `metric_validate.resolve_behavior_family` is a
    WORD-level keyword match over the whole goal string, so "approach — run up
    and kick the ball" resolves the *approach* mode to family `kick`; the
    approach mode's non-degeneracy would then be anchored against a kick
    positive and a legitimate run-up metric would be false-rejected. The
    episode goal is still given to the author, as CONTEXT, through
    `mode_prompt_context` — where it informs the prose without steering family
    resolution.

    `mode_goals` is the caller's chance to say what a mode actually means; a
    mode named `mode_2` carries no semantics. Falling back to the humanized
    mode name is honest but thin, and `generate_mode_metrics` records which of
    the two was used so a report never implies more grounding than there was.
    """
    supplied = (mode_goals or {}).get(mode.name)
    if supplied and str(supplied).strip():
        return str(supplied).strip()
    return re.sub(r"[_\s]+", " ", mode.name).strip() or mode.name


def mode_prompt_context(
    graph: ModeGraph,
    mode: Mode,
    *,
    episode_goal: str = "",
    mode_goals: Optional[Mapping[str, str]] = None,
) -> str:
    """The per-mode DATA block appended to the authoring/review prompt.

    The mode-scoping RULES live in `prompts/gen_mode_metric.md`; this is the
    graph-derived data those rules apply to — which mode, its window, what runs
    before and after it, and which guards enter and leave it. Splitting them
    keeps a rule in exactly one place, the same reason `_review_one` layers a
    per-lens focus on one shared rubric rather than forking the rubric.
    """
    windows = mode_phase_windows(graph)
    start_s, end_s = windows[mode.name]
    order = [m.name for m in graph.modes]
    position = order.index(mode.name)
    incoming = [t for t in graph.transitions if t.to_mode == mode.name]
    outgoing = [t for t in graph.transitions if t.from_mode == mode.name]
    payload = {
        "mode": mode.name,
        "mode_goal": mode_goal_text(mode, mode_goals=mode_goals),
        "position": f"{position + 1} of {len(order)}",
        "mode_order": order,
        "window_seconds": [start_s, end_s],
        "window_reference_frames": list(mode.frame_range),
        "reference_fps": graph.fps,
        "preceded_by": order[position - 1] if position else None,
        "followed_by": order[position + 1] if position + 1 < len(order) else None,
        "guards_in": [t.to_dict() for t in incoming],
        "guards_out": [t.to_dict() for t in outgoing],
        "episode_goal_for_context_only": episode_goal,
    }
    return (
        "# THIS MODE (data — the rules above apply to exactly this)\n"
        + json.dumps(payload, indent=2, default=str)
    )


def mode_reference_clip(
    clip: Mapping[str, Any], mode: Mode, *, clip_id: Optional[str] = None,
) -> dict[str, Any]:
    """Crop a reference clip to one mode's own frames.

    This is what makes reference-anchored validation per-mode rather than
    per-episode. `_validate_references` scores a metric against the reference
    and its perturbation suite (reversal, freeze, shuffle, truncations); run
    against the WHOLE composite those gates ask "can this metric recognize the
    full three-phase motion", which a per-mode metric should fail — it is only
    responsible for its own phase. Cropped, each mode's metric is anchored
    against the segment it actually scores, and the perturbation suite becomes
    "can it tell this phase from this phase reversed/frozen/shuffled".

    The crop is by the mode's `frame_range`, which `modes_from_composition`
    derives from the composition's own seam frames — so the boundaries are the
    clip's real seams, not a guess.

    Raises `ModeMetricError` if the range does not fit the clip or the cropped
    result is not a valid clip; `sculptor.reference.validate_clip` is the judge,
    so a future time-indexed channel that this crop forgot shows up as a loud
    validation failure instead of a silently ragged clip.
    """
    # `_TIME_KEYS` / `_PASSTHROUGH_KEYS` are imported rather than restated so
    # there is ONE list of which channels are frame-indexed. Restating it here
    # would silently mis-crop any channel added to perturb.py later.
    from sculptor.refs.perturb import _PASSTHROUGH_KEYS, _TIME_KEYS
    from sculptor.reference import validate_clip

    if clip.get("root_pos_z") is None:
        raise ModeMetricError(
            "cannot crop a clip with no 'root_pos_z' — that key defines the "
            "clip's frame count (sculptor.reference.validate_clip requires it)")
    lo, hi = int(mode.frame_range[0]), int(mode.frame_range[1])
    source_len = int(np.asarray(clip["root_pos_z"]).shape[0])
    if lo < 0 or hi > source_len:
        raise ModeMetricError(
            f"mode {mode.name!r} spans frames [{lo}, {hi}) but the clip has "
            f"{source_len} frames")
    if hi - lo < 2:
        raise ModeMetricError(
            f"mode {mode.name!r} spans {hi - lo} frame(s); a reference clip "
            "needs at least 2 to carry a velocity")

    out: dict[str, Any] = {}
    for key in _PASSTHROUGH_KEYS:
        if key in clip:
            out[key] = clip[key]
    for key in _TIME_KEYS:
        value = clip.get(key)
        if value is not None:
            out[key] = np.asarray(value)[lo:hi]

    # Provenance: a cropped clip that forgets it was cropped is indistinguishable
    # from a short clip, and every downstream record (signatures, certificates)
    # would then attribute the mode's numbers to the whole reference. Copy the
    # meta rather than mutating the caller's clip.
    meta = dict(out.get("meta") or {})
    meta["mode_crop"] = {
        "mode": mode.name,
        "frame_range": [lo, hi],
        "source_clip_id": clip_id or meta.get("clip_id"),
        "source_frames": source_len,
        "cropped_by": "sculptor.eval.mode_metrics.mode_reference_clip",
    }
    out["meta"] = meta

    errors = validate_clip(out)
    if errors:
        raise ModeMetricError(
            f"cropping {mode.name!r} produced an invalid clip:\n  - "
            + "\n  - ".join(errors))
    return out


# ── the gauntlet, one mode at a time ─────────────────────────────────────


def generate_mode_metrics(
    graph: ModeGraph,
    out_dir: Path | str,
    *,
    episode_goal: str = "",
    mode_goals: Optional[Mapping[str, str]] = None,
    reference_clip: Optional[Mapping[str, Any]] = None,
    reference_clip_id: Optional[str] = None,
    robot_hint: Optional[str] = None,
    client: Any = None,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    **generate_kwargs: Any,
) -> dict[str, Any]:
    """Generate one objective metric PER MODE, each in its own directory.

    Each mode goes through the UNMODIFIED `generate_objective_metric` — the same
    validation chain, the same independent review panel, the same best-of-N — with
    three things swapped: the mode's own goal (see `mode_goal_text` for why the
    episode goal must not be spliced into it), the mode's cropped slice of the
    reference (`mode_reference_clip`), and a mode-scoping prompt appendix.

    Artifacts land in `<out_dir>/mode_<name>/{metric.py,meta.json}`, the layout
    every existing consumer already reads, so a per-mode metric is a first-class
    metric rather than a new artifact shape.

    Never raises on one mode's failure: a mode whose generation dies records its
    error and the remaining modes still run. A four-mode behavior where one
    metric cannot be authored is a partial result worth having.
    """
    from sculptor.eval.metric_gen import generate_objective_metric
    from sculptor.prompts import load_prompt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        rules = load_prompt("gen_mode_metric")
    except FileNotFoundError as e:  # pragma: no cover — shipped alongside
        raise ModeMetricError(f"mode-metric prompt is missing: {e}") from e
    try:
        review_rules: Optional[str] = load_prompt("review_mode_metric")
    except FileNotFoundError:  # pragma: no cover — optional reviewer appendix
        review_rules = None

    records: dict[str, Any] = {}
    for mode in graph.modes:
        goal = mode_goal_text(mode, mode_goals=mode_goals)
        mode_dir = out_dir / f"{MODE_DIR_PREFIX}{mode.name}"
        entry: dict[str, Any] = {
            "mode": mode.name,
            "goal": goal,
            "goal_source": ("supplied" if (mode_goals or {}).get(mode.name)
                            else "derived_from_mode_name"),
            "out_dir": str(mode_dir),
            "accepted": False,
            "error": None,
            "record": None,
        }
        references: Optional[list[tuple[str, dict]]] = None
        if reference_clip is not None:
            try:
                cropped = mode_reference_clip(
                    reference_clip, mode, clip_id=reference_clip_id)
            except Exception as e:  # noqa: BLE001 — a bad crop is not fatal
                entry["reference_error"] = f"{type(e).__name__}: {e}"
            else:
                ref_id = f"{reference_clip_id or 'reference'}#{mode.name}"
                references = [(ref_id, cropped)]
                entry["reference_clip_id"] = ref_id
        appendix = rules + "\n\n" + mode_prompt_context(
            graph, mode, episode_goal=episode_goal, mode_goals=mode_goals)
        try:
            entry["record"] = generate_objective_metric(
                goal, mode_dir, robot_hint=robot_hint, client=client,
                references=references, on_event=on_event,
                prompt_appendix=appendix, review_appendix=review_rules,
                **generate_kwargs)
            entry["accepted"] = bool(entry["record"].get("accepted"))
            entry["metric_path"] = entry["record"].get("metric_path")
        except Exception as e:  # noqa: BLE001 — one mode's failure is not the run's
            entry["error"] = f"{type(e).__name__}: {e}"
        records[mode.name] = entry
    return {
        "modes": records,
        "n_accepted": sum(1 for e in records.values() if e["accepted"]),
        "n_modes": len(records),
        "episode_goal": episode_goal,
    }


def validate_mode_metrics(
    sources: Mapping[str, tuple[str, Path | str]],
    graph: ModeGraph,
    *,
    mode_goals: Optional[Mapping[str, str]] = None,
    reference_clip: Optional[Mapping[str, Any]] = None,
    reference_clip_id: Optional[str] = None,
    robot_hint: Optional[str] = None,
    **validate_kwargs: Any,
) -> dict[str, Any]:
    """Run the EXISTING validation gauntlet against each mode's metric.

    `sources` maps mode name → `(source_text, module_path)`. Every mode is put
    through `validate_generated_metric` unchanged, with the mode's own goal and
    (when a reference is supplied) the mode's own cropped segment as the
    reference-anchored evidence.

    Nothing here relaxes, widens, or special-cases a gate, and nothing adds a
    synthetic positive to the fixed battery. That last one is not a stylistic
    preference: an unconditionally-added positive archetype has already, once,
    masked the `still`/`upright_flail` negatives and rescued a gameable metric
    in this file's own history. A mode is scoped by its GOAL — which is what
    selects the family, the abstract program, and therefore the battery — and
    by its cropped reference. If a per-mode metric fails a gate, the metric is
    wrong.

    Overall `ok` requires every mode to pass, and a mode with no metric supplied
    is a failure, not an omission — a mode whose metric was never authored is
    exactly the silent dead end `validate_mode_graph`'s reachability check
    exists to prevent one level up.
    """
    from sculptor.eval.metric_validate import validate_generated_metric

    results: dict[str, Any] = {}
    for mode in graph.modes:
        goal = mode_goal_text(mode, mode_goals=mode_goals)
        entry: dict[str, Any] = {"mode": mode.name, "goal": goal, "ok": False}
        supplied = sources.get(mode.name)
        if supplied is None:
            entry["reasons"] = [
                f"[mode] no metric supplied for mode {mode.name!r} — an "
                "unauthored mode is scored by nothing, which is the silent "
                "dead end per-mode gating exists to remove"]
            results[mode.name] = entry
            continue
        source, module_path = supplied
        references: Optional[list[tuple[str, dict]]] = None
        if reference_clip is not None:
            try:
                cropped = mode_reference_clip(
                    reference_clip, mode, clip_id=reference_clip_id)
            except Exception as e:  # noqa: BLE001 — recorded, never fatal
                entry["reference_error"] = f"{type(e).__name__}: {e}"
            else:
                references = [
                    (f"{reference_clip_id or 'reference'}#{mode.name}", cropped)]
        try:
            validation = validate_generated_metric(
                source, module_path, behavior_goal=goal, robot_hint=robot_hint,
                references=references, **validate_kwargs)
        except Exception as e:  # noqa: BLE001 — mirrors the never-raise contract
            entry["reasons"] = [f"[mode] validation raised: {type(e).__name__}: {e}"]
            results[mode.name] = entry
            continue
        entry.update({
            "ok": bool(validation.get("ok")),
            "validation": validation,
            "reasons": validation.get("reasons", []),
            "family": validation.get("family"),
        })
        results[mode.name] = entry
    return {
        "ok": bool(results) and all(e["ok"] for e in results.values()),
        "modes": results,
        "failed_modes": [n for n, e in results.items() if not e["ok"]],
    }


def calibrate_mode_metrics(
    metric_paths: Mapping[str, Path | str],
    graph: ModeGraph,
    *,
    mode_goals: Optional[Mapping[str, str]] = None,
    robot_hint: Optional[str] = None,
    client: Any = None,
    adversarial: bool = True,
    **calibrate_kwargs: Any,
) -> dict[str, Any]:
    """Author and score a competence ladder AND gaming archetypes PER MODE.

    Item 3 of the per-mode brief, and the least obvious of the four: a metric
    that is hard to game across a whole episode can be trivially gameable inside
    one mode. An episode-wide gaming archetype has to fool every phase at once;
    a mode-scoped one only has to fool a two-second window, which is a far
    weaker requirement and therefore a far sharper test. The same holds for the
    ladder — "more competent at the whole behavior" is a coarse axis, while
    "more competent at the grasp" is the axis the grasp metric actually claims
    to measure.

    Mechanically this is `calibrate_task_derived` per mode with the mode's own
    goal, which is what makes the blind ladder author and the blind gaming
    author write for the sub-behavior. `adversarial=True` by default here (it
    defaults off in `calibrate_task_derived`, where flipping it would change an
    established grant path): the per-mode gate is new surface, so it starts at
    the stricter setting rather than being loosened into one later.

    Steer-rights stay exactly where they were. This function REPORTS each
    mode's grant decision; the firewall that keeps an uncalibrated metric
    observe-only is untouched.
    """
    from sculptor.eval.metric_calibration import calibrate_task_derived

    results: dict[str, Any] = {}
    for mode in graph.modes:
        goal = mode_goal_text(mode, mode_goals=mode_goals)
        entry: dict[str, Any] = {"mode": mode.name, "goal": goal, "ok": False}
        path = metric_paths.get(mode.name)
        if path is None:
            entry["reason"] = (
                f"no metric path for mode {mode.name!r} — nothing to calibrate")
            results[mode.name] = entry
            continue
        try:
            calibration = calibrate_task_derived(
                path, goal, robot_hint, client=client,
                adversarial=adversarial, **calibrate_kwargs)
        except Exception as e:  # noqa: BLE001 — calibrate_task_derived's own
            # contract is never-raise; a raise here is a caller/plumbing bug and
            # is recorded as an observe-only reason rather than killing the run.
            entry["reason"] = f"calibration raised: {type(e).__name__}: {e}"
            results[mode.name] = entry
            continue
        entry.update({
            "ok": bool(calibration.get("ok")),
            "calibration": calibration,
            "reason": calibration.get("reason"),
            "spearman": calibration.get("spearman"),
            "gameable": bool(
                (calibration.get("adversarial") or {}).get("gameable")),
        })
        results[mode.name] = entry
    return {
        "ok": bool(results) and all(e["ok"] for e in results.values()),
        "modes": results,
        "granted_modes": [n for n, e in results.items() if e["ok"]],
        "gameable_modes": [n for n, e in results.items() if e.get("gameable")],
    }


# ── the report a human reads ─────────────────────────────────────────────


def mode_gauntlet_report(
    graph: ModeGraph,
    *,
    episode_goal: str = "",
    scores: Optional[Mapping[str, Any]] = None,
    transitions: Optional[Sequence[Mapping[str, Any]]] = None,
    validation: Optional[Mapping[str, Any]] = None,
    calibration: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the per-mode pieces into one record keyed by MODE.

    The pieces are produced independently (scoring needs a rollout, validation
    needs sources, calibration needs an LLM) and any subset may be absent; a
    missing piece is recorded as absent rather than as a pass. Keying by mode is
    the point — the whole complaint about episode-level scoring is that a
    failure has no address, so the report gives every finding one.
    """
    per_mode: dict[str, dict[str, Any]] = {}
    for mode in graph.modes:
        entry: dict[str, Any] = {"mode": mode.name}
        score_entry = ((scores or {}).get("modes") or {}).get(mode.name)
        if score_entry is not None:
            entry["score"] = score_entry.get("score")
            entry["scored"] = score_entry.get("scored")
            entry["entered"] = (score_entry.get("slice") or {}).get("entered")
            entry["coverage"] = (score_entry.get("slice") or {}).get("coverage")
            entry["shorter_than_one_step"] = (
                (score_entry.get("slice") or {}).get("shorter_than_one_step"))
            entry["score_error"] = score_entry.get("error")
        val_entry = ((validation or {}).get("modes") or {}).get(mode.name)
        if val_entry is not None:
            entry["validation_ok"] = val_entry.get("ok")
            entry["validation_reasons"] = val_entry.get("reasons", [])
        cal_entry = ((calibration or {}).get("modes") or {}).get(mode.name)
        if cal_entry is not None:
            entry["calibration_ok"] = cal_entry.get("ok")
            entry["calibration_reason"] = cal_entry.get("reason")
            entry["gameable"] = cal_entry.get("gameable")
        per_mode[mode.name] = entry
    return {
        "episode_goal": episode_goal,
        "mode_order": [m.name for m in graph.modes],
        "modes": per_mode,
        "transitions": list(transitions or []),
        "episode_score": (scores or {}).get("episode"),
        "worst_mode": (scores or {}).get("worst_mode"),
        "worst_mode_gap": (scores or {}).get("worst_mode_gap"),
        "unentered_modes": (scores or {}).get("unentered_modes", []),
        "unfired_guards": [
            f"{t['from_mode']}->{t['to_mode']}"
            for t in (transitions or []) if t.get("fired") is False
        ],
        "failed_modes": (validation or {}).get("failed_modes", []),
        "gameable_modes": (calibration or {}).get("gameable_modes", []),
        "have": {
            "scores": scores is not None,
            "transitions": transitions is not None,
            "validation": validation is not None,
            "calibration": calibration is not None,
        },
    }


def render_mode_report(report: Mapping[str, Any]) -> str:
    """A plain-text per-mode report — WHICH mode failed, and why.

    Formatted for the same place a human already reads iteration output. The
    ordering is deliberate: modes in graph order (so the reader walks the
    behavior forward), then transitions, then the episode number LAST — an
    episode score printed first is the thing that hid the failure, and leading
    with it invites the same mistake in prose that it caused in code.
    """
    lines: list[str] = []
    goal = report.get("episode_goal") or ""
    lines.append(f"PER-MODE GAUNTLET{f' — {goal}' if goal else ''}")
    lines.append("")
    for name in report.get("mode_order", []):
        entry = (report.get("modes") or {}).get(name, {})
        bits: list[str] = []
        if entry.get("shorter_than_one_step"):
            # Distinct from NEVER ENTERED: the decomposition is finer than the
            # controller can resolve, which is a decomposition problem, not a
            # policy one.
            bits.append("SHORTER THAN ONE CONTROL STEP")
        elif entry.get("entered") is False:
            bits.append("NEVER ENTERED")
        elif entry.get("score") is not None:
            bits.append(f"score {float(entry['score']):.3f}")
            coverage = entry.get("coverage")
            if coverage is not None and coverage < 1.0:
                bits.append(f"covered {float(coverage) * 100:.0f}% of its window")
        if entry.get("validation_ok") is not None:
            bits.append("gates PASS" if entry["validation_ok"] else "gates FAIL")
        if entry.get("calibration_ok") is not None:
            bits.append("calibrated" if entry["calibration_ok"]
                        else "observe-only")
        if entry.get("gameable"):
            bits.append("GAMEABLE within this mode")
        lines.append(f"  {name}: " + (", ".join(bits) or "no result"))
        for reason in (entry.get("validation_reasons") or [])[:4]:
            lines.append(f"      - {reason}")
        if entry.get("score_error"):
            lines.append(f"      - {entry['score_error']}")
        if entry.get("calibration_reason") and not entry.get("calibration_ok"):
            lines.append(f"      - {entry['calibration_reason']}")
    if report.get("transitions"):
        lines.append("")
        lines.append("  transitions:")
        for transition in report["transitions"]:
            fired = transition.get("fired")
            mark = {True: "fired", False: "NEVER FIRED"}.get(fired, "not evaluated")
            lines.append(
                f"    {transition['from_mode']} -> {transition['to_mode']} "
                f"[{transition.get('kind')}]: {mark}")
            if fired is not True:
                lines.append(f"      - {transition.get('reason', '')}")
    episode = report.get("episode_score")
    if episode is not None:
        lines.append("")
        lines.append(f"  episode score: {float(episode):.3f}")
        gap = report.get("worst_mode_gap")
        worst = report.get("worst_mode")
        if gap is not None and worst:
            lines.append(
                f"  worst mode {worst!r} is {float(gap):.3f} below it — the "
                f"gap an episode-level score averages away (reported, not "
                f"gated)")
    return "\n".join(lines)
