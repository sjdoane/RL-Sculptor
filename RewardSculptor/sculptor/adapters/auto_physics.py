"""sculptor/adapters/auto_physics.py — §7.4 auto-adjust physics prompt synthesis.

When §7.3's realism audit returns `verdict == "severe"`, the sculpt
loop calls `synthesize_auto_physics_prompt(audit)` to produce a
natural-language MJCF-edit prompt that the existing physics-editor
flow (`backend/services/physics.py::apply_prompt_edit`) can consume.

Pure stdlib — unit-tested CPU-only, no Claude calls, no mjlab.

The prompt is KG-primed: paper_refs in the prompt's "Cite relevant
papers" line come from the `_CANONICAL_PHYSICS_PAPERS` list below,
which is a tiny curated set known to be in the shared KG as of
2026-04-22. If any are missing, Claude's `references[]` emission will
fall back to "novel." — the edit still lands, just uncited.
"""

from __future__ import annotations

from typing import Any


# Papers that virtually any physics-realism edit can cite. Each is in
# the shared KG per the 2026-04-22 cartwheel-session ingest pass.
_CANONICAL_PHYSICS_PAPERS: tuple[tuple[str, str], ...] = (
    ("2312.17507", "Actuator-Constrained RL — learning with realistic torque + speed bounds"),
    ("1901.08652", "ANYmal actuator-net — end-to-end motor dynamics model for sim-to-real"),
    ("2410.08650", "Extended Friction — stick/slip + viscous / Coulomb friction in MJCF"),
)


def _format_top_joints(joints: list[dict] | None) -> str:
    """Render up to 3 top-offending joints as `'name'=0.47`."""
    if not joints:
        return "(no joint-level data)"
    parts: list[str] = []
    for j in joints[:3]:
        name = str(j.get("name") or "?")
        val = j.get("value")
        if isinstance(val, (int, float)):
            parts.append(f"'{name}'={float(val):.2f}")
        else:
            parts.append(f"'{name}'")
    return ", ".join(parts)


def synthesize_auto_physics_prompt(audit: dict[str, Any]) -> str:
    """Build a ready-to-apply NL prompt for the physics-editor flow
    given a realism audit dict (as produced by
    `sculptor.adapters.realism.audit_rollout`).

    The prompt names:
      - the three headline metrics with their actual values,
      - the top saturated / high-velocity joints by name,
      - specific MJCF knobs to tighten (forcerange, armature, damping),
      - canonical KG paper_refs to cite.

    Returns a string in the 200-600 char range — fits comfortably under
    the physics-editor's 2000-char limit while carrying enough numbers
    for Claude to make concrete edits.
    """
    verdict = str(audit.get("verdict", "unknown"))
    overall_sat = audit.get("torque_saturation_frac")
    worst_sat = audit.get("any_joint_saturation_max")
    vel_p99 = audit.get("joint_vel_p99_max")
    vel_mult = audit.get("joint_vel_multiplier_vs_nominal")
    limit_viol = audit.get("joint_limit_violation_frac")
    top_sat = _format_top_joints(audit.get("top_joints_saturation"))
    top_vel = _format_top_joints(audit.get("top_joints_vel"))

    def _pct(x: Any) -> str:
        if isinstance(x, (int, float)):
            return f"{float(x) * 100:.1f}%"
        return "n/a"

    def _num(x: Any, unit: str = "") -> str:
        if isinstance(x, (int, float)):
            return f"{float(x):.2f}{unit}"
        return "n/a"

    citations = ", ".join(f"arXiv:{aid}" for aid, _ in _CANONICAL_PHYSICS_PAPERS)

    lines = [
        f"The last RL rollout exploited unrealistic actuator response "
        f"(realism audit verdict={verdict.upper()}). Tighten the MJCF so "
        f"the policy can't continue doing this.",
        "",
        "Evidence from the rollout:",
        f"  - torque saturation: {_pct(overall_sat)} of (step, env, joint) "
        f"triples, worst single joint {_pct(worst_sat)}",
        f"  - 99th-percentile joint velocity: {_num(vel_p99, ' rad/s')} "
        f"({_num(vel_mult, '')}× nominal 30 rad/s)",
        f"  - joint-limit violation: {_pct(limit_viol)} of (step, env, joint) triples",
        f"  - top saturated joints: {top_sat}",
        f"  - top high-velocity joints: {top_vel}",
        "",
        "Apply the following in order of priority:",
        "  1. Tighten `<actuator forcerange>` on the saturated joints to "
        "values representative of real hardware (peak stall torque from "
        "the motor datasheet — if unknown, halve the current range).",
        "  2. Increase `<joint armature>` to model rotor inertia (typical "
        "0.001-0.02 kg·m² for a small servomotor through 40:1 gearbox).",
        "  3. Increase `<joint damping>` to model viscous friction + "
        "gear backlash (typical 0.1-2.0 Nm·s/rad for a loaded joint).",
        "  4. If high velocity is the primary issue, add or tighten the "
        "velocity limit via a position-servo `<velocity kv=>` term.",
        "",
        f"Cite: {citations}. Keep the commit message ≤80 chars and "
        f"prefix with 'auto-physics-fix:'.",
    ]
    return "\n".join(lines)


def should_auto_adjust_physics(
    audit: dict[str, Any] | None,
    *,
    auto_adjust_enabled: bool,
) -> bool:
    """Returns True iff an auto-adjust should fire for this audit.

    Fires only on verdict=severe AND the project's config has
    `[iteration].auto_adjust_physics = true`. Explicit-kwarg naming keeps
    the call site self-documenting. Mild / ok / unknown verdicts are
    NEVER auto-adjusted — the UI should show the suggestion for mild
    but not act on it automatically.
    """
    if not auto_adjust_enabled:
        return False
    if not isinstance(audit, dict):
        return False
    return str(audit.get("verdict", "")).lower() == "severe"
