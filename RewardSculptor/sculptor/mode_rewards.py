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

import re
from typing import Any, Mapping, Optional

from sculptor.modes import Mode, ModeGraph, ModeError, validate_mode_graph

#: Identifier-safe form of a mode name, for the generated function names.
_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")

#: Emitted per-mode function prefix. Kept explicit so `validate_mode_reward_source`
#: and the per-mode metric layer agree on one naming convention.
MODE_FN_PREFIX = "_mode_"

#: Component key a mode's contribution is reported under. Per-mode metrics key
#: off this, so it is part of the contract rather than a formatting detail.
MODE_COMPONENT_PREFIX = "mode_"


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


def generate_mode_reward_scaffold(
    graph: ModeGraph,
    *,
    behavior_goal: str = "",
    goal_by_mode: Optional[Mapping[str, str]] = None,
    clip_id: str = "",
) -> str:
    """Emit a reward module whose mode gating is correct by construction.

    Each mode gets a `_mode_<ident>(state, action, next_state, info)` returning
    `(float, dict)`. `compute_reward` pays ONLY the mode whose window contains
    the current wall-clock time, and reports it under
    `mode_<ident>` plus `active_mode_index` so a per-mode metric can slice a
    rollout by mode without re-deriving the automaton.

    Raises `ModeError` when the graph is structurally invalid or when two mode
    names collide once sanitized to identifiers.
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

    windows = mode_windows_s(graph)
    goals = dict(goal_by_mode or {})
    order = [m.name for m in graph.modes]

    fns = "\n\n".join(
        f"def {MODE_FN_PREFIX}{idents[m.name]}(state, action, next_state, info):\n"
        + _stub_body(m, goals.get(m.name, ""))
        for m in graph.modes
    )
    dispatch = ",\n".join(
        f"    {name!r}: {MODE_FN_PREFIX}{idents[name]}" for name in order)

    return f'''"""Auto-generated per-mode reward scaffold{f" for clip {clip_id!r}" if clip_id else ""}.

Generated by `sculptor.mode_rewards.generate_mode_reward_scaffold` from a
validated `ModeGraph`. The mode gating below is DERIVED, not authored — both
Tier-D failures in this repo were phase-clock bugs, so the clock is not left to
generated code. Author the `{MODE_FN_PREFIX}*` bodies (one per mode) and leave
the dispatch alone; regenerate from the graph rather than hand-editing windows.

Behavior goal: {behavior_goal.strip() or "(unspecified)"}
"""
from __future__ import annotations

REWARD_SPEC: dict = {{
    "version": "mode-reward-v1",
    "description": "Per-mode reward over a {len(graph.modes)}-mode automaton.",
    "author": "sculptor",
    "parent_hash": None,
    "supports_batched": False,
    # Consumed by the per-mode metric layer: it slices a rollout by these
    # windows instead of scoring the episode as one undifferentiated blob.
    "mode_windows_s": {{{", ".join(f"{n!r}: {list(w)}" for n, w in windows.items())}}},
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


def _elapsed_s(info) -> float:
    """Wall time into the episode. `step_dt` is published per-step by the mjlab
    runner; falling back to a fixed rate would silently mis-gate every mode, so
    the fallback is the G1 task's real 0.02 s (50 Hz) rather than a guess."""
    step = float(info.get("episode_length", 0) or 0)
    step_dt = float(info.get("step_dt", 0.0) or 0.0) or 0.02
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


{fns}


_MODE_FNS = {{
{dispatch},
}}


def compute_reward(state, action, next_state, info):
    """Pay only the active mode.

    Terms authored for one mode are not paid during another — that scoping is
    the entire point of the automaton. An episode-level sum would let a term
    written for landing punish the policy throughout a flight phase where the
    reference is deliberately pitched over.
    """
    name = active_mode(info)
    value, components = _MODE_FNS[name](state, action, next_state, info)
    value = float(value)
    out = {{f"{MODE_COMPONENT_PREFIX}{{name}}": value,
           "active_mode_index": float(MODE_ORDER.index(name))}}
    for k, v in (components or {{}}).items():
        out[f"{{name}}.{{k}}"] = float(v)
    return value, out
'''


def authored_modes(source: str) -> dict[str, bool]:
    """`{mode_name: is_authored}` for a scaffold's source.

    A stub is detected by its marker, not by comparing against a regenerated
    scaffold — an author may legitimately reformat a body, and re-deriving the
    scaffold to diff against it would call every such edit a stub.
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
        body = re.search(
            rf"^def {re.escape(MODE_FN_PREFIX + ident)}\(.*?\):\n(.*?)(?=\n\ndef |\n\n_MODE_FNS)",
            source, re.M | re.S)
        out[name] = bool(body) and "UNAUTHORED STUB" not in body.group(1)
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

    windows = mode_windows_s(graph)
    for name, (lo, hi) in windows.items():
        if f"{name!r}: ({lo!r}, {hi!r})" not in source:
            errors.append(
                f"mode {name!r}: window ({lo}, {hi}) not found in source — the "
                "scaffold is stale relative to the graph; regenerate it")
    return errors


def mode_authoring_prompt(
    graph: ModeGraph,
    mode_name: str,
    *,
    behavior_goal: str = "",
    mode_goal: str = "",
) -> str:
    """The per-mode authoring instruction for `apply_prompt_edit`.

    Deliberately states the mode's window and its neighbours: the most common
    way per-mode authoring goes wrong is a term that is correct for its mode but
    written as though it applied to the whole episode (e.g. "keep both feet on
    the ground" authored for an approach mode, which then reads as a global
    constraint to anyone editing later). Naming the neighbours makes the scope
    explicit in the prompt rather than implicit in the gating.
    """
    mode = graph.mode(mode_name)          # KeyError here is a caller bug
    windows = mode_windows_s(graph)
    lo, hi = windows[mode_name]
    order = [m.name for m in graph.modes]
    i = order.index(mode_name)
    prev_m = order[i - 1] if i > 0 else None
    next_m = order[i + 1] if i + 1 < len(order) else None

    neighbours = []
    if prev_m:
        neighbours.append(f"preceded by {prev_m!r} (ends at {windows[prev_m][1]}s)")
    if next_m:
        neighbours.append(f"followed by {next_m!r} (starts at {windows[next_m][0]}s)")
    neighbour_line = ("; ".join(neighbours)) if neighbours else "the only mode"

    return (
        f"Author the body of `{MODE_FN_PREFIX}{mode_ident(mode_name)}` — the "
        f"reward for mode {mode_name!r} ONLY.\n\n"
        f"Overall behavior: {behavior_goal.strip() or '(unspecified)'}\n"
        f"This mode's job: {mode_goal.strip() or mode.name}\n"
        f"Window: {lo}s to {hi}s "
        f"(reference frames [{mode.frame_range[0]}, {mode.frame_range[1]}) at "
        f"{graph.fps} fps); {neighbour_line}.\n\n"
        "Scope rules:\n"
        "- This function is called ONLY inside that window. Do not add terms "
        "that belong to another mode, and do not try to detect the phase "
        "yourself — the dispatch already did.\n"
        "- Do not reward or penalize anything the neighbouring modes own. A "
        "term that is right for this mode but reads as a global constraint is "
        "the main failure here.\n"
        "- Return `(value, components)` where components names are snake_case; "
        "they are reported as `<mode>.<name>` and a per-mode metric scores "
        "them against this window's slice of the rollout.\n"
        "- Leave `compute_reward`, `MODE_WINDOWS_S`, `MODE_ORDER` and the other "
        "modes' functions untouched."
    )


__all__ = [
    "MODE_COMPONENT_PREFIX",
    "MODE_FN_PREFIX",
    "authored_modes",
    "generate_mode_reward_scaffold",
    "mode_authoring_prompt",
    "mode_ident",
    "mode_windows_s",
    "validate_mode_reward_source",
]
