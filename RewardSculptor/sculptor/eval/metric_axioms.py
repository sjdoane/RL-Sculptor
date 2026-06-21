"""§Ship 50: L1 task-agnostic axioms — universal physical invariants.

L0 (`metric_validate`) proves a metric DISCRIMINATES: its family positive
beats a fixed battery of degenerate negatives. That is a comparison of
DIFFERENT, confounded archetypes. L1 is complementary and methodologically
stronger: it changes ONE physical variable on the metric's own operating
point and asserts the score moves in the physically-correct direction — a
CONTROLLED PERTURBATION, so the only thing that changed is the thing tested.
L0 structurally cannot express several of these: every L0 archetype spawns
at the world origin, with UNIT gravity, travelling along +x — so a metric
that leaks absolute position, raw gravity magnitude, or an absolute heading
passes all of L0 unnoticed.

Six axioms, each universal (or precisely scoped) and empirically tuned so
every hand-authored built-in and every GOOD_* reference metric passes while
real Goodhart leaks are caught:

  INVARIANCES (universal; physical symmetries — the score must NOT change):
  I1 translation_invariant   — offset the base world position; competence is
     relative, never an absolute coordinate.
  I2 gravity_scale_invariant — scale the gravity vector; uprightness is a unit
     DIRECTION, not a sensor magnitude.
  I3 yaw_rotation_invariant  — rotate heading about vertical; gravity is
     yaw-blind and goals here are heading-agnostic.

  MONOTONICITIES (the score must not move the WRONG way):
  M1 uprightness_monotone (universal) — sweep the body from vertical toward
     toppled; the score must be NON-INCREASING (never reward falling). The
     sweep tilts toward HORIZONTAL (never inversion), so a handstand/inverted
     task is not false-rejected.
  M2 no_reward_for_chaos (universal) — inject whole-body joint-velocity chaos;
     the score must not rise by more than `tol_chaos`. Catches a metric that
     rewards raw motion/energy (the FLAIL_REWARDER leak).
  M3 stationary_no_travel (kick/floss/jump) — add forward base travel; a
     stationary skill must not reward walking away (the g1-kick Goodhart).

Operating point = the positive archetype the metric scores HIGHEST (its own
"best" behavior). This self-scopes the gate: a novel-orientation task (e.g. a
handstand metric) scores all of this upright biped battery ~0, so the
perturbations stay ~0 and the axioms pass vacuously — L1 never false-rejects
a task whose competent behavior the synthetic battery does not represent.

Pure-numpy, offline, deterministic (fixed RNG), no GPU/API. Returns a
per-axiom verdict for the standardized trust score (Ship-52) and the UI;
`validate_generated_metric` folds `ok` into its gate set so a metric that
violates a universal physical law is rejected, not merely flagged.

Design provenance: the invariance axioms came out of a physics-invariance
design panel (2026-06-15); the chaos tolerance and the sweep range were
calibrated empirically against the GOOD_*/built-in reference metrics.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from sculptor.eval.generated_metric import inject_joint_roles
from sculptor.eval.metric_validate import (
    _NAMES_12,
    _STATIONARY_FAMILIES,
    _archetypes,
)

_BEHAVIOR = {"max_episode_steps": 120, "rollout_num_envs": 4, "step_dt": 0.02}

#: Positive archetypes a metric might score high; the operating point is the
#: best of these. (The negatives are L0's job.)
_POSITIVE_KEYS = ("active", "active_kick", "active_floss", "active_jump")

#: For the stationarity axiom, the positive that represents each stationary
#: family's competent behavior.
_FAMILY_POSITIVE = {
    "kick": "active_kick",
    "floss": "active_floss",
    "jump": "active_jump",
}

#: Body-tilt sweep (degrees from vertical) for the uprightness axiom: upright →
#: horizontal. Tilts toward the +x side (gravity z stays ≤ 0, never inverts),
#: so an inversion task (handstand wants z > 0) is never false-rejected.
_TILT_SWEEP_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
_YAW_DEG = 37.0
_TRANSLATE_XY = (100.0, -50.0)
_GRAVITY_SCALE = 9.81
_INVARIANCE_TOL = 1e-6
_MONO_TOL = 1e-6

# ── controlled perturbations (pure numpy, deterministic) ─────────────


def _copy(arrays: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in arrays.items()}


def _tilt(arrays: dict, deg: float) -> dict:
    """Set body-frame gravity to a pure tilt of `deg` from vertical toward
    +x: g = (sin, 0, −cos). deg=0 upright, deg=90 lying on its side. Never
    inverts (z ≤ 0), so a handstand/inversion task stays unrewarded-flat
    rather than being penalised."""
    b = _copy(arrays)
    th = np.deg2rad(deg)
    g = np.zeros_like(b["projected_gravity_b"])
    g[..., 0] = np.sin(th)
    g[..., 2] = -np.cos(th)
    b["projected_gravity_b"] = g
    return b


def _inject_chaos(arrays: dict, *, amp: float = 4.0) -> dict:
    """Add large deterministic whole-body joint-velocity noise (and its
    integrated position) ON TOP of the behavior. Competence is structure, not
    energy — a competent-behavior metric does not score this higher."""
    b = _copy(arrays)
    rng = np.random.default_rng(50)            # fixed → the gate is deterministic
    noise = rng.normal(0.0, amp, b["joint_vel"].shape)
    b["joint_vel"] = b["joint_vel"] + noise
    b["joint_pos"] = b["joint_pos"] + np.cumsum(noise, axis=0) * _BEHAVIOR["step_dt"]
    return b


def _add_travel(arrays: dict, *, speed: float = 0.05) -> dict:
    """Add steady forward base translation without touching the joints — for a
    STATIONARY skill, walking away is never improvement."""
    b = _copy(arrays)
    root = b["root_link_pos_w"]
    t = np.arange(root.shape[0])
    root[..., 0] = root[..., 0] + (t * speed)[:, None]
    return b


def _translate(arrays: dict) -> dict:
    """Offset the base world position by a large constant (x,y); z untouched so
    height gates are preserved. Competence is relative — any displacement/
    velocity computation differences this out."""
    b = _copy(arrays)
    b["root_link_pos_w"][..., 0] += _TRANSLATE_XY[0]
    b["root_link_pos_w"][..., 1] += _TRANSLATE_XY[1]
    return b


def _scale_gravity(arrays: dict) -> dict:
    """Scale the gravity vector (sensor-units change). Uprightness is a unit
    DIRECTION; a metric keying on raw |g| is silently miscalibrated."""
    b = _copy(arrays)
    b["projected_gravity_b"] = b["projected_gravity_b"] * _GRAVITY_SCALE
    return b


def _yaw_rotate(arrays: dict, deg: float = _YAW_DEG) -> dict:
    """Rotate the horizontal plane of gravity AND root position by a yaw angle:
    the same rollout with a different starting heading. Gravity is yaw-blind
    (z untouched) and the goals here are heading-agnostic, so a competent
    metric is invariant; one that leaks an absolute heading is not."""
    b = _copy(arrays)
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    for key in ("projected_gravity_b", "root_link_pos_w"):
        if key in b and isinstance(b[key], np.ndarray) and b[key].shape[-1] >= 2:
            x = b[key][..., 0].copy()
            y = b[key][..., 1].copy()
            b[key][..., 0] = c * x - s * y
            b[key][..., 1] = s * x + c * y
    return b


def _eval(fn: Callable[..., dict], arrays: dict, meta: dict) -> float:
    try:
        out = fn(arrays, _BEHAVIOR, meta)
        return float(out.get("spec_score", float("nan")))
    except Exception:  # noqa: BLE001 — a crash under perturbation = nan, handled
        return float("nan")


def check_metric_axioms(
    fn: Callable[..., dict],
    *,
    family: Optional[str] = None,
    required_roles: Optional[Sequence[str]] = None,
    torso_target: str = "upright",
    tol_chaos: float = 0.50,
    tol_travel: float = 0.10,
    inv_tol: float = _INVARIANCE_TOL,
) -> dict[str, Any]:
    """Run the L1 axioms on a loaded metric `fn`. Returns
    `{ok, axioms: {name: bool}, reasons: [...], details: {...}}`. Never raises.

    `family` (from `resolve_behavior_family`) scopes the stationarity axiom.
    `required_roles` are injected (lenient) so a role-based metric reads the
    right synthetic columns. `torso_target` (from `resolve_torso_target`, LAW
    13) scopes the M1 uprightness-monotonicity axiom: it runs ONLY for an
    `"upright"` skill (the default), and is SKIPPED for a `"horizontal"`
    (flip/dive/roll/crawl) or `"any"` (handstand/cartwheel) target — a metric
    that competently rewards a non-upright torso must not be penalised for it.
    The chaos tolerance (0.5) clears the most peak-sensitive GOOD_* metric
    (~0.33 worst-case) while catching a pure energy rewarder (~1.0); the
    invariances are exact (1e-6)."""
    meta = {"joint_names": list(_NAMES_12)}
    inject_joint_roles(meta, list(required_roles or []), lenient=True)
    arche = _archetypes()

    axioms: dict[str, bool] = {}
    reasons: list[str] = []
    details: dict[str, Any] = {}

    pos_scores = {k: _eval(fn, arche[k], meta) for k in _POSITIVE_KEYS}
    details["positive_scores"] = {
        k: (round(v, 4) if np.isfinite(v) else None) for k, v in pos_scores.items()
    }
    finite_pos = {k: v for k, v in pos_scores.items() if np.isfinite(v)}
    best_key = max(finite_pos, key=finite_pos.get) if finite_pos else None
    details["operating_point"] = best_key

    if best_key is None:
        # The metric is nan on every positive — L0's bounded gate already fails
        # it and there is nothing meaningful to perturb.
        return {"ok": True, "axioms": {}, "reasons": [],
                "details": {**details, "skipped": "no finite positive"}}

    base = arche[best_key]
    base_score = finite_pos[best_key]
    details["operating_score"] = round(base_score, 4)

    def _delta(perturbed: dict) -> float:
        s = _eval(fn, perturbed, meta)
        return s - base_score if np.isfinite(s) else float("nan")

    # ── invariances (exact) ──────────────────────────────────────────
    for name, pert, human in (
        ("translation_invariant", _translate,
         "shifting the world origin (+100,−50) m"),
        ("gravity_scale_invariant", _scale_gravity,
         f"scaling gravity ×{_GRAVITY_SCALE}"),
        ("yaw_rotation_invariant", lambda a: _yaw_rotate(a, _YAW_DEG),
         f"rotating heading {int(_YAW_DEG)}°"),
    ):
        d = _delta(pert(base))
        ok = (not np.isfinite(d)) or abs(d) <= inv_tol
        axioms[name] = bool(ok)
        details[name + "_delta"] = round(d, 6) if np.isfinite(d) else None
        if not ok:
            reasons.append(
                f"[axiom:{name}] {human} CHANGED the score {base_score:.3f}→"
                f"{base_score + d:.3f} (Δ{d:+.4f}) — the metric reads a "
                f"frame/units artifact, not physical competence")

    # ── M1 uprightness monotonicity (sweep) — §LAW 13: upright skills ONLY ──
    # Tilt the operating point from vertical toward horizontal and require the
    # score not to RISE. A flip/dive/roll (torso_target="horizontal") or a
    # handstand/cartwheel ("any") is competently non-upright, so a metric that
    # scores a tilted torso higher must NOT be penalised — skip the sweep.
    if torso_target == "upright":
        sweep = [_eval(fn, _tilt(base, deg), meta) for deg in _TILT_SWEEP_DEG]
        details["uprightness_sweep"] = [
            round(v, 4) if np.isfinite(v) else None for v in sweep]
        mono = True
        for i in range(len(sweep) - 1):
            a, b2 = sweep[i], sweep[i + 1]
            if np.isfinite(a) and np.isfinite(b2) and b2 > a + _MONO_TOL:
                mono = False
                reasons.append(
                    f"[axiom:uprightness] tilting the base from {int(_TILT_SWEEP_DEG[i])}°"
                    f" to {int(_TILT_SWEEP_DEG[i + 1])}° RAISED the score "
                    f"{a:.3f}→{b2:.3f} — the metric rewards falling, not the behavior")
                break
        axioms["uprightness_monotone"] = bool(mono)
    else:
        details["uprightness_monotone_skipped"] = f"torso_target={torso_target}"

    # ── M2 no reward for chaos / raw energy ───────────────────────────
    d_cha = _delta(_inject_chaos(base))
    a2 = (not np.isfinite(d_cha)) or d_cha <= tol_chaos
    axioms["no_reward_for_chaos"] = bool(a2)
    details["chaos_delta"] = round(d_cha, 4) if np.isfinite(d_cha) else None
    if not a2:
        reasons.append(
            f"[axiom:chaos] injecting whole-body flail RAISED the score "
            f"{base_score:.3f}→{base_score + d_cha:.3f} (Δ{d_cha:+.3f} > "
            f"{tol_chaos}) — the metric rewards raw motion/energy, which is "
            f"gameable by flailing")

    # ── M3 stationary skills must not reward travel (scoped) ──────────
    if family in _STATIONARY_FAMILIES:
        fk = _FAMILY_POSITIVE.get(family)
        fc = pos_scores.get(fk, float("nan")) if fk else float("nan")
        if fk is not None and np.isfinite(fc):
            s_tv = _eval(fn, _add_travel(arche[fk]), meta)
            d_tv = s_tv - fc if np.isfinite(s_tv) else float("nan")
            a3 = (not np.isfinite(d_tv)) or d_tv <= tol_travel
            axioms["stationary_no_travel"] = bool(a3)
            details["travel_delta"] = round(d_tv, 4) if np.isfinite(d_tv) else None
            if not a3:
                reasons.append(
                    f"[axiom:stationarity] adding base travel to the {family} "
                    f"positive RAISED the score {fc:.3f}→{s_tv:.3f} (Δ{d_tv:+.3f} "
                    f"> {tol_travel}) — a stationary skill must not reward "
                    f"walking away")

    return {"ok": all(axioms.values()), "axioms": axioms,
            "reasons": reasons, "details": details}
