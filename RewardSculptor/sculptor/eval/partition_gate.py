"""sculptor/eval/partition_gate.py — deterministic shaping↔metric partition gate.

An offline, LLM-free guard that runs at the reward-edit commit point
(`sculptor.edit.apply_edits`) WHEN an objective metric is steering the run. Two
jobs, both grounded in the g1-kick-v5 reward-hack — the editor lowered a
completion gate and farmed the metric's own signal over 21 iterations until the
metric climbed while the real behavior degraded:

  1. PARTITION (non-blocking FLAG + prompt warning). Surface the reward-side
     channels that read the SAME physical observables the active metric SCORES
     ("held-out test surface") so the editor shapes physics proxies instead of
     teaching to the test.
  2. GATE-EROSION.
       * pre-LLM (non-blocking FLAG + prompt warning): a proposed edit that
         lowers / removes a completion or qualification gate.
       * post-LLM (HARD — the only blocking gate): the emitted reward LOWERED a
         same-named, positive-valued completion-gate hyperparameter. This is the
         load-bearing enforcement — the v5 whack-a-mole, killed deterministically.

DESIGN (after a 4-lens adversarial design review, §Ship 54-pre):
  * NOTHING here drops/rejects a ProposedEdit pre-LLM — a pre-LLM reject
    re-introduces the 2026-04-23 loop-freeze (drop every edit → no reward update)
    and is bypassable anyway, because the rewriter emits free-form source it is
    NOT bound to apply literally. The single hard gate is post-LLM and is
    restricted to the unambiguous case so a false positive is near-impossible.
  * The lexicon is TOKEN-matched on the snake_case split, NOT raw substring
    (so `min` matches the token `min`, never `terminated`/`administer`).
  * REJECT lexicon = unambiguous completion-gate roots only; FLAG lexicon =
    direction-ambiguous roots (floor / min / threshold / cycle) — flagged, never
    hard-failed (e.g. `gait_cycle_freq`, `contact_threshold`, `min_torque` are
    legitimately lowerable).
  * Numeric comparison uses `float()` only; a parse failure or a non-positive
    current value ⇒ NOT a hard regression (penalty/magnitude sign-semantics are
    offline-undecidable — a penalty made "more negative" is a strengthening).

KNOWN UNCOVERED VECTORS (documented residual risk; out of scope for this slice):
  * a gate lowered as an INLINE literal in `compute_reward` (not in
    REWARD_SPEC.hyperparameters) — the hparam-dict diff cannot see it.
  * a farm-able ADDED term referencing a scored observable — surfaced as a FLAG +
    prompt warning, not post-write-enforced.
  * gate-WIDENING of an excluded tolerance/deadband term to admit degenerate
    sub-motions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


# ── Reward-side name → metric observable it reads ────────────────────────────
# The reward (mjlab state_schema + info keys) and the metric (trajectory arrays)
# share several observable names outright (`projected_gravity_b`) and alias the
# rest. A reward edit that touches a name mapping to an observable the ACTIVE
# metric scores is "reaching into the test." Foot swing-speed / height are farmed
# via `joint_vel` (the metric's burst channel), NOT the per-foot `*_pos_b`
# excursion channel — so they alias to joint_vel.
REWARD_TO_OBSERVABLE: dict[str, str] = {
    "qvel": "joint_vel",
    "qpos": "joint_pos",
    "base_height": "root_link_pos_w",
    "base_horizontal_speed": "root_link_pos_w",
    "base_lin_vel_b": "root_link_pos_w",
    "fallen": "projected_gravity_b",
    "left_foot_contact": "left_foot_contact",
    "right_foot_contact": "right_foot_contact",
    "left_foot_pos_b": "left_foot_pos_b",
    "right_foot_pos_b": "right_foot_pos_b",
    "left_foot_swing_speed": "joint_vel",
    "right_foot_swing_speed": "joint_vel",
    "left_foot_height": "joint_vel",
    "right_foot_height": "joint_vel",
}

# ── Gate lexicons (TOKEN-matched against the snake_case split) ────────────────
# REJECT: unambiguous completion-gate roots. Lowering one of these ALWAYS eases
#   qualification, so a numeric lowering of a same-named positive value is the
#   single hard-failing case.
GATE_REJECT_TOKENS: frozenset[str] = frozenset({
    "gate", "completion", "complete", "completed",
    "qualify", "qualified", "qualification",
    "require", "required", "requirement", "requires",
})
# FLAG: direction-ambiguous roots that collide with benign hparams
#   (`gait_cycle_freq`, `contact_threshold`, `min_torque`, `floor_friction`).
#   Flagged for the editor + changelog, but NEVER hard-failed.
GATE_FLAG_TOKENS: frozenset[str] = frozenset({
    "floor", "min", "minimum", "threshold", "thresh", "cycle",
})
# Deliberately EXCLUDED from both: tolerance / margin / deadband / band. They are
# direction-ambiguous (lowering a tolerance TIGHTENS a gate), and `tolerance` is
# a dm_control shaping helper in edit.py's _ALLOWED_MATH — treating it as a gate
# name would false-positive on legitimate `tolerance(...)` calls.


# ── Containers ───────────────────────────────────────────────────────────────
@dataclass
class ScreenResult:
    """Pre-LLM screen of the proposed edits (advisory only — never drops)."""

    flagged_edits: list = field(default_factory=list)       # ProposedEdit
    flag_reasons: list[str] = field(default_factory=list)
    held_out: list[str] = field(default_factory=list)       # reward names touched
    gate_hparams: list[str] = field(default_factory=list)   # current gate hparams


@dataclass
class RegressionResult:
    """Post-LLM diff of parent vs emitted gate hyperparameters."""

    hard: list[str] = field(default_factory=list)       # caller RAISES on these
    advisory: list[str] = field(default_factory=list)   # caller warns, never raises


# ── Token / lexicon helpers ──────────────────────────────────────────────────
def _tokens(name: Any) -> set[str]:
    """Lowercase snake_case tokens of `name` (numbers kept, empties dropped)."""
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def gate_kind(name: Any) -> Optional[str]:
    """`"reject"` if `name` names an unambiguous completion gate, `"flag"` if it
    names a direction-ambiguous gate/threshold, else `None`. Token-matched."""
    toks = _tokens(name)
    if toks & GATE_REJECT_TOKENS:
        return "reject"
    if toks & GATE_FLAG_TOKENS:
        return "flag"
    return None


def _parse_float(value: Any) -> Optional[float]:
    """`float(value)` or None on any parse failure (free-form prose, units…)."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# ── Partition: reward name ↔ metric observable ───────────────────────────────
def observable_of(name: Any, metric_observables: Iterable[str]) -> Optional[str]:
    """The metric observable a reward-side `name` reads, IF the active metric
    actually scores it — else None. Exact-name alias first, then identity (the
    name IS a metric array). No fuzzy matching: the flag must be precise."""
    mo = frozenset(metric_observables or ())
    nl = str(name).strip().lower()
    if nl in REWARD_TO_OBSERVABLE:
        obs = REWARD_TO_OBSERVABLE[nl]
        return obs if obs in mo else None
    if nl in mo:
        return nl
    return None


def held_out_channels(metric_observables: Iterable[str]) -> list[str]:
    """Every reward-side channel name that reads an observable the active metric
    scores — the held-out surface to warn the editor about (whether or not any
    proposed edit touched it)."""
    mo = frozenset(metric_observables or ())
    out: set[str] = {o for o in mo}
    for rname, obs in REWARD_TO_OBSERVABLE.items():
        if obs in mo:
            out.add(rname)
    return sorted(out)


def _names_in_edit(target_term: Any, suggested_value: Any) -> set[str]:
    """Identifier-like names in an edit's target_term + suggested_value. A plain
    word-scan (no AST) — sufficient because we only test membership against the
    small known alias/observable vocabulary."""
    names = {str(target_term).strip()} if target_term else set()
    if suggested_value:
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(suggested_value)))
    return {n for n in names if n}


def _numeric_lowering(
    edit: Any, current_hparams: Mapping[str, Any],
) -> Optional[tuple[float, float]]:
    """`(old, new)` when an edit proposes a parseable numeric LOWERING of a known
    positive hparam (for the pre-LLM advisory flag), else None."""
    cur = current_hparams.get(getattr(edit, "target_term", None))
    new = _parse_float(getattr(edit, "suggested_value", None))
    curf = _parse_float(cur)
    if curf is None or new is None:
        return None
    if curf > 0 and new < curf:
        return (curf, new)
    return None


# ── Pre-LLM screen (advisory; never drops an edit) ───────────────────────────
def screen_edits(
    edits: Iterable[Any],
    *,
    metric_observables: Iterable[str],
    current_hparams: Optional[Mapping[str, Any]] = None,
) -> ScreenResult:
    """Flag proposed edits that (a) touch a held-out metric observable, or
    (b) lower/remove a completion/qualification gate. NON-BLOCKING: the flagged
    edits stay applicable; the report drives the editor-prompt warning and the
    changelog. The single hard gate is `gate_threshold_regressions`, post-LLM."""
    mo = frozenset(metric_observables or ())
    hparams = dict(current_hparams or {})
    res = ScreenResult()
    res.gate_hparams = sorted(h for h in hparams if gate_kind(h))
    held: set[str] = set()

    for i, e in enumerate(edits):
        reasons: list[str] = []

        # (a) partition overlap
        overlaps: list[tuple[str, str]] = []
        for nm in _names_in_edit(
            getattr(e, "target_term", None), getattr(e, "suggested_value", None)
        ):
            obs = observable_of(nm, mo)
            if obs is not None:
                held.add(nm)
                overlaps.append((nm, obs))
        if overlaps:
            reasons.append(
                "partition: "
                + ", ".join(f"{nm}→{obs}" for nm, obs in sorted(set(overlaps)))
            )

        # (b) gate erosion
        kind = gate_kind(getattr(e, "target_term", None))
        op = str(getattr(e, "operation", "") or "")
        if kind and op in {"decrease", "remove"}:
            reasons.append(
                f"gate-erosion: operation={op!r} on gate term "
                f"{getattr(e, 'target_term', None)!r}"
            )
        elif kind:
            low = _numeric_lowering(e, hparams)
            if low is not None:
                reasons.append(
                    f"gate-erosion: {getattr(e, 'target_term', None)!r} "
                    f"lowered {low[0]:g}→{low[1]:g}"
                )

        if reasons:
            res.flagged_edits.append(e)
            res.flag_reasons.append(f"edit[{i}]: " + "; ".join(reasons))

    res.held_out = sorted(held)
    return res


# ── Post-LLM regression check (the only hard gate) ───────────────────────────
def gate_threshold_regressions(
    parent_hparams: Optional[Mapping[str, Any]],
    new_hparams: Optional[Mapping[str, Any]],
) -> RegressionResult:
    """Compare parent vs emitted REWARD_SPEC.hyperparameters for gate erosion.

    HARD (caller raises):
      * a hparam present in BOTH whose name is an unambiguous completion
        gate (REJECT lexicon), whose parent value is POSITIVE, and whose
        emitted value is numerically LOWER. This is the v5 whack-a-mole,
        and the clearest case to block.
      * §7.9 rename-bypass close: EXACTLY ONE positive reject-lexicon gate
        was REMOVED and EXACTLY ONE reject-lexicon gate was ADDED, and the
        added value is LOWER than the removed one — a rename+lower is the
        same erosion wearing a new name. One-to-one pairing only: with
        multiple removed/added candidates the mapping is ambiguous and it
        stays advisory (a legitimate multi-gate refactor must not freeze
        the loop).
    ADVISORY (caller warns, never raises):
      * a same-named lowering of a FLAG-lexicon (direction-ambiguous) gate;
      * a REMOVED gate hparam with no unambiguous rename pairing;
      * a sign-ambiguous (non-positive parent value) reject-lexicon lowering.
    """
    parent = dict(parent_hparams or {})
    new = dict(new_hparams or {})
    res = RegressionResult()
    removed_reject: list[tuple[str, float]] = []
    for name, pval in parent.items():
        kind = gate_kind(name)
        if not kind:
            continue
        if name not in new:
            pf = _parse_float(pval)
            if kind == "reject" and pf is not None and pf > 0:
                removed_reject.append((name, pf))
            res.advisory.append(f"{name!r} (gate hparam) removed in the new reward")
            continue
        pf = _parse_float(pval)
        nf = _parse_float(new[name])
        if pf is None or nf is None or nf >= pf:
            continue
        if kind == "reject" and pf > 0:
            res.hard.append(f"{name!r} {pf:g}→{nf:g}")
        else:
            res.advisory.append(f"{name!r} {pf:g}→{nf:g} (ambiguous gate — lowered)")

    # §7.9 rename-bypass: pair the single removed reject gate with the
    # single added reject gate, compare values.
    added_reject: list[tuple[str, float]] = []
    for name, nval in new.items():
        if name in parent or gate_kind(name) != "reject":
            continue
        nf = _parse_float(nval)
        if nf is not None:
            added_reject.append((name, nf))
    if len(removed_reject) == 1 and len(added_reject) == 1:
        (old_name, old_v), (new_name, new_v) = removed_reject[0], added_reject[0]
        if new_v < old_v:
            res.hard.append(
                f"{old_name!r} {old_v:g} renamed to {new_name!r} {new_v:g} "
                f"(removed+added completion gate pair with a LOWER value — "
                f"rename does not launder gate erosion)")
    return res


# ── Editor-prompt block ──────────────────────────────────────────────────────
def build_partition_prompt_block(
    metric_observables: Iterable[str],
    screen: ScreenResult,
) -> str:
    """The `# METRIC_PARTITION` user-prompt block (self-contained — carries its
    own instructions so the shared system prompt stays unchanged, keeping the
    no-metric path byte-identical). Empty string only when there is nothing to
    say (never on the metric-steered path, which always lists observables)."""
    mo = sorted(frozenset(metric_observables or ()))
    if not mo:
        return ""
    held = held_out_channels(mo)
    lines: list[str] = [
        "# METRIC_PARTITION (objective-metric integrity — read before editing)",
        "An objective metric SCORES this run as a HELD-OUT judge. Keep the reward "
        "(shaping) and the metric (test) separate: shape with physics "
        "first-principles, never reproduce the metric's own thresholds or "
        "scoring composition. The g1-kick-v5 run reward-hacked by doing exactly "
        "that — it lowered a completion gate and farmed the metric's signal.",
        "",
        "The metric SCORES these physical observables (the held-out surface):",
        "  " + ", ".join(mo),
        "These reward-side channels read the SAME observables — use them as "
        "PROXIES (instantaneous physical shaping) only, do NOT mirror the "
        "metric's gates/composition:",
        "  " + ", ".join(held),
    ]
    if screen.gate_hparams:
        lines += [
            "",
            "Do NOT LOWER or REMOVE these completion/qualification gate "
            "hyperparameters — a lower gate lets degenerate sub-motions qualify "
            "(the v5 stall). A new reward that numerically lowers a completion "
            "gate will be REJECTED post-write:",
            "  " + ", ".join(screen.gate_hparams),
        ]
    if screen.flag_reasons:
        lines += [
            "",
            "The diagnoser proposed edits that touch the above (applied, but "
            "FLAGGED — re-derive them as honest physical proxies, do not erode a "
            "gate):",
        ]
        lines += [f"  - {r}" for r in screen.flag_reasons]
    return "\n".join(lines) + "\n\n"
