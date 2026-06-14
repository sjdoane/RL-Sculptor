"""Validation gates for AUTO-GENERATED objective metrics (§Ship 35).

An LLM-authored metric is untrusted code that will score rollouts. Before
it may be used at all (even observe-only), it must clear the MUST-HAVE
gates here; before it may STEER a run it must additionally pass
calibration (see metric_calibration). These gates implement the red-team's
non-negotiable list:

  1. AST safety         — no imports except numpy; no exec/eval/open/dunder
  2. Array-contract     — references only persisted physical arrays
  3. Determinism        — identical output on 3 repeated runs
  4. Bounded [0,1]      — finite, in-range, never raises on diverse inputs
  5. Non-degeneracy     — discriminates: a dead-still / fallen policy must
                          score BELOW an active-upright one, with spread

These are a SMELL TEST, not a proof of task-validity — that comes from
calibration against the hand-authored ground-truth metrics. The metric is
constrained to physical quantities (the whole point: an objective signal,
not LLM judgment).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sculptor.eval.generated_metric import (
    ALLOWED_ARRAYS,
    GENERATED_FN_NAME,
    load_generated_metric,
)

#: Modules a generated metric may import (numpy only — it is a pure
#: physical-quantity function).
_ALLOWED_IMPORTS = {"numpy"}
#: Names that must never appear (code-exec / IO / introspection vectors).
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "Path",
    "importlib", "pickle", "marshal",
}

T, E, J = 120, 4, 12
_NAMES_12 = [
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_elbow", "right_elbow", "torso", "neck",
]


def _ast_safety(source: str) -> list[str]:
    """Return a list of safety violations (empty = safe)."""
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    problems.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                problems.append(f"forbidden import-from: {node.module}")
        elif isinstance(node, ast.Name) and (
                node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            # §Ship 35 review: also reject ANY dunder NAME (e.g.
            # __builtins__ is in every module namespace without an import
            # and reaches eval/exec).
            problems.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute access: {node.attr}")
    return problems


def _referenced_array_keys(source: str) -> set[str]:
    """Extract the string keys used to index the `arrays` mapping —
    `arrays["k"]` / `arrays.get("k")`. Constrains the metric to the
    persisted-array contract."""
    keys: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return keys
    for node in ast.walk(tree):
        # arrays["k"]
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "arrays"):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
        # arrays.get("k")
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "arrays"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


# ── synthetic archetype rollouts (non-degeneracy gate) ───────────────


def _upright_g() -> np.ndarray:
    g = np.zeros((T, E, 3), dtype=np.float64)
    g[..., 2] = -1.0
    return g


def _archetypes() -> dict[str, dict]:
    """Four archetype rollouts spanning a crude competence axis. A valid
    task metric should score `active` strictly above `still` and `fallen`,
    with non-trivial spread."""
    rng = np.random.default_rng(0)
    t = np.arange(T)

    def arrays(joint_pos, joint_vel, gravity, root):
        return {"joint_pos": joint_pos, "joint_vel": joint_vel,
                "projected_gravity_b": gravity, "root_link_pos_w": root}

    # dead-still upright
    jp0 = np.zeros((T, E, J)); jv0 = rng.normal(0, 0.01, (T, E, J))
    root0 = np.zeros((T, E, 3)); root0[..., 2] = 0.5
    still = arrays(jp0, jv0, _upright_g(), root0)

    # fallen (gravity sideways), thrashing
    g_fall = np.zeros((T, E, 3)); g_fall[..., 0] = 1.0
    jvf = rng.normal(0, 3.0, (T, E, J))
    rootf = np.zeros((T, E, 3)); rootf[..., 2] = 0.1
    fallen = arrays(rng.normal(0, 1, (T, E, J)), jvf, g_fall, rootf)

    # chaotic upright (random large motion, no structure, no travel)
    jpc = rng.normal(0, 1.5, (T, E, J)); jvc = rng.normal(0, 5.0, (T, E, J))
    rootc = np.zeros((T, E, 3)); rootc[..., 2] = 0.5
    chaotic = arrays(jpc, jvc, _upright_g(), rootc)

    # active: upright, smooth periodic joints, steady forward travel
    jpa = np.zeros((T, E, J))
    for jj in range(J):
        jpa[:, :, jj] = (0.4 * np.sin(2 * np.pi * t / 25 + jj))[:, None]
    jva = np.gradient(jpa, axis=0)
    roota = np.zeros((T, E, 3)); roota[..., 2] = 0.5
    roota[..., 0] = (t * 0.04)[:, None]   # forward
    active = arrays(jpa, jva, _upright_g(), roota)

    # §Ship 36: upright_flail — large, fast limb oscillation while standing
    # still with ZERO travel: the "stand still and flail" reward-hack Sam's
    # G1 kick run fell into. A valid task metric must score this BELOW
    # `active`; a metric that merely rewards motion magnitude will not, and
    # the non-degeneracy gate now rejects it.
    jpf = np.zeros((T, E, J))
    for jj in range(J):
        jpf[:, :, jj] = (1.2 * np.sin(2 * np.pi * t / 6.0 + jj))[:, None]
    jvf2 = np.gradient(jpf, axis=0)
    rootf2 = np.zeros((T, E, 3)); rootf2[..., 2] = 0.5   # upright, no travel
    upright_flail = arrays(jpf, jvf2, _upright_g(), rootf2)

    return {"still": still, "fallen": fallen, "chaotic": chaotic,
            "active": active, "upright_flail": upright_flail}


def _score(fn, arrays, meta) -> float:
    out = fn(arrays, {"max_episode_steps": T, "rollout_num_envs": E,
                      "step_dt": 0.02}, meta)
    return float(out.get("spec_score", float("nan")))


def validate_generated_metric(
    source: str,
    module_path: Path | str,
    *,
    spread_min: float = 0.1,
) -> dict[str, Any]:
    """Run all MUST-HAVE gates on a generated metric. `source` is the
    module text (for static gates); `module_path` is where it's been
    written (for the runtime gates). Returns
    `{ok: bool, gates: {name: bool}, reasons: [...], archetype_scores: {}}`.
    Never raises — a crashing metric is a failed gate, not an exception."""
    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. AST safety
    safety = _ast_safety(source)
    gates["ast_safety"] = not safety
    reasons += [f"[safety] {s}" for s in safety]

    # contract: must define compute_spec
    gates["defines_compute_spec"] = (GENERATED_FN_NAME in source)
    if GENERATED_FN_NAME not in source:
        reasons.append(f"[contract] missing def {GENERATED_FN_NAME}(arrays, behavior, meta)")

    # 2. Array-contract
    bad_keys = _referenced_array_keys(source) - set(ALLOWED_ARRAYS)
    gates["array_contract"] = not bad_keys
    if bad_keys:
        reasons.append(f"[contract] references unavailable arrays: {sorted(bad_keys)} "
                       f"(allowed: {list(ALLOWED_ARRAYS)})")

    # Static gates must pass before we exec the module.
    if not (gates["ast_safety"] and gates["defines_compute_spec"]):
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}}

    try:
        fn = load_generated_metric(module_path)
    except Exception as e:  # noqa: BLE001
        gates["loads"] = False
        reasons.append(f"[load] {type(e).__name__}: {e}")
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}}
    gates["loads"] = True

    meta = {"joint_names": _NAMES_12}
    arche = _archetypes()
    scores: dict[str, float] = {}

    # 3 + 4: determinism + bounded/finite, over every archetype.
    determ = True
    bounded = True
    for name, arrays in arche.items():
        try:
            s1 = _score(fn, arrays, meta)
            s2 = _score(fn, arrays, meta)
            s3 = _score(fn, arrays, meta)
        except Exception as e:  # noqa: BLE001
            bounded = False
            reasons.append(f"[run] raised on '{name}': {type(e).__name__}: {e}")
            scores[name] = float("nan")
            continue
        scores[name] = s1
        if not (s1 == s2 == s3):
            determ = False
            reasons.append(f"[determinism] '{name}' varied across runs: {s1},{s2},{s3}")
        if not (np.isfinite(s1) and 0.0 <= s1 <= 1.0):
            bounded = False
            reasons.append(f"[bounds] '{name}' out of [0,1] or non-finite: {s1}")
    gates["determinism"] = determ
    gates["bounded"] = bounded

    # 5: non-degeneracy — active must beat still + fallen, with spread.
    nondegen = True
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if len(finite) < 3:
        nondegen = False
        reasons.append("[nondegeneracy] too few finite archetype scores")
    else:
        spread = max(finite.values()) - min(finite.values())
        if spread < spread_min:
            nondegen = False
            reasons.append(f"[nondegeneracy] near-constant metric "
                           f"(spread {spread:.3f} < {spread_min}) — no signal")
        a = finite.get("active", 0.0)
        # §Ship 36: `upright_flail` added — a motion-magnitude metric that
        # rewards standing-and-flailing (the observed G1 kick hack) now fails.
        for low in ("still", "fallen", "upright_flail"):
            if low in finite and finite[low] >= a:
                nondegen = False
                reasons.append(f"[nondegeneracy] '{low}' ({finite[low]:.3f}) scores "
                               f">= active ({a:.3f}) — rewards the wrong behavior")
    gates["nondegeneracy"] = nondegen

    ok = all(gates.values())
    return {"ok": ok, "gates": gates, "reasons": reasons,
            "archetype_scores": scores}
