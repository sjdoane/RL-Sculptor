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
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from sculptor.eval.generated_metric import (
    ALLOWED_ARRAYS,
    GENERATED_FN_NAME,
    inject_joint_roles,
    load_generated_module,
    read_required_roles,
)
from sculptor.eval.joint_resolver import resolve_joint_roles

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

#: §Ship 41: behavior family → the hand-authored ground-truth metric a
#: generated metric of that family calibrates against (metric_calibration).
#: Used by the launch-time auto-calibration path; exported for reuse.
FAMILY_TO_BUILTIN = {
    "kick": "g1_kick",
    "floss": "g1_floss",
    "jump": "g1_jump",
    "locomotion": "go1_trot",
    "cartpole": "cartpole_balance",
}


def resolve_behavior_family(
    behavior_goal: Optional[str], robot_hint: Optional[str] = None,
) -> Optional[str]:
    """§Ship 41: map a natural-language behavior goal to a behavior family
    (`kick`/`floss`/`jump`/`locomotion`/`cartpole`) so the non-degeneracy gate
    can anchor a non-locomotion metric against a behavior-APPROPRIATE positive
    archetype instead of a hard-coded forward-walker (the false-rejection bug).

    WORD-level (not substring) keyword match: real goals are paraphrased (the
    on-disk kick goal does not equal the benchmark string), but every benchmark
    goal contains a clean family word. `None` → no family matched. The family
    selects the CALIBRATION ground truth (and the returned label); it does NOT
    narrow the non-degeneracy gate."""
    g = (behavior_goal or "").lower()
    tokens = set(re.findall(r"[a-z]+", g))

    def has(*words: str) -> bool:
        return any(w in tokens for w in words)

    # §Ship 41 review: WORD matching — substring "hop" matched "Hopper" (a
    # locomotion example) and "strike" matched the idiom "strike a balance",
    # false-rejecting good metrics. "bound"/"strike" dropped ("bound" is a
    # quadruped GAIT; "strike" is too idiomatic — "kick" is the clear token).
    if has("kick", "kicks", "kicking"):
        return "kick"
    if has("floss", "flossing", "opposition", "antiphase") or "anti-phase" in g:
        return "floss"
    if has("jump", "jumps", "jumping", "hop", "hops", "hopping",
           "leap", "leaps", "leaping"):
        return "jump"
    if has("trot", "trotting", "walk", "walking", "forward", "gait",
           "locomote", "locomotion", "run", "running", "march", "marching",
           "stride", "striding"):
        return "locomotion"
    if has("balance", "balancing", "cartpole"):
        return "cartpole"
    # Robot-family fallback: a quadruped goal with no behavior word is almost
    # always locomotion.
    rh = (robot_hint or "").lower()
    if any(w in rh for w in ("go1", "go2", "quadruped")):
        return "locomotion"
    return None


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


def _is_int_const(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return True
    # `-1` parses as UnaryOp(USub, Constant) — count it too.
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and _is_int_const(node.operand))


def _raw_joint_index_violations(source: str) -> list[str]:
    """§Ship 49: flag a HARD-CODED integer index into a joint axis — the
    `x[:, :, 0]` form — which silently reads the wrong joint the moment the
    robot or joint order changes (the §3A failure). Metrics must select
    joints via name-resolved indices (`meta['joint_roles']`). The Ellipsis
    form `x[..., 2]` is NOT flagged: that's the convention for the 3-vector
    gravity/root axes (`projected_gravity_b[..., 2]` = uprightness), which
    are not joint axes."""
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return problems
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        sl = node.slice
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 3 \
                and isinstance(sl.elts[0], ast.Slice) \
                and isinstance(sl.elts[1], ast.Slice) \
                and _is_int_const(sl.elts[2]):
            problems.append(
                "hard-coded integer joint index `[:, :, N]` — select joints "
                "by name via meta['joint_roles'], not a literal column")
    return problems


# §Ship 49: a non-trivial relabelling of the 12-joint archetype axis. The
# permutation-robustness gate applies it CONSISTENTLY to joint_names AND the
# joint_pos/joint_vel columns — a name/role-resolving metric is invariant
# (it follows the names to the same physical joints), an index-hardcoding one
# swings (column 0 is now a different joint).
def _permute_joint_arrays(
    arrays: dict, perm: list[int],
) -> dict:
    out = {}
    for k, v in arrays.items():
        if k in ("joint_pos", "joint_vel") and getattr(v, "ndim", 0) >= 3 \
                and v.shape[2] == len(perm):
            out[k] = v[:, :, perm]
        else:
            out[k] = v
    return out


# ── synthetic archetype rollouts (non-degeneracy gate) ───────────────


def _upright_g() -> np.ndarray:
    g = np.zeros((T, E, 3), dtype=np.float64)
    g[..., 2] = -1.0
    return g


def _archetypes() -> dict[str, dict]:
    """Synthetic archetype rollouts spanning a competence axis. Negatives
    (`still`/`fallen`/`chaotic`/`upright_flail`) plus a POSITIVE per behavior
    family (`active` locomotion, `active_kick`, `active_floss`, `active_jump`).
    A valid task metric scores its family positive strictly above the negatives
    with non-trivial spread (the gate picks the family in
    `validate_generated_metric`)."""
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

    # §Ship 41: behavior-family POSITIVE archetypes — a COMPETENT example of
    # each non-locomotion behavior, so a kick/floss/jump metric is measured
    # against its own behavior rather than the forward-walker `active`. All are
    # upright, stationary, at STANDING height (z≈0.7 — a kick metric's height
    # gate needs ≥0.65; z=0.5 would leave a good metric below the spread floor).
    def _standing_root() -> np.ndarray:
        r = np.zeros((T, E, 3)); r[..., 2] = 0.7
        return r

    # active_kick: discrete leg-velocity bursts (left hip-pitch/knee/ankle =
    # indices 0/2/4), stance leg quiet — a clean, repeated, stationary kick.
    jvk = np.zeros((T, E, J))
    for start in range(20, T, 40):            # 3 discrete kicks
        for jdx in (0, 2, 4):
            jvk[start:start + 5, :, jdx] = 8.0
    jpk = np.cumsum(jvk, axis=0) * 0.02       # consistent integrated position
    active_kick = arrays(jpk, jvk, _upright_g(), _standing_root())

    # active_floss: anti-phase hip↔arm oscillation. SLOW (period 25, like the
    # g1_floss ladder) so a motion-MAGNITUDE metric cannot mistake it for the
    # fast `upright_flail` negative — flossing is structure, not speed.
    jpfl = np.zeros((T, E, J))
    hip = 0.4 * np.sin(2 * np.pi * t / 25)
    arm = 0.4 * np.sin(2 * np.pi * t / 25 + np.pi)
    for jdx in (0, 1):                        # hips
        jpfl[:, :, jdx] = hip[:, None]
    for jdx in (6, 7, 8, 9):                  # shoulders + elbows
        jpfl[:, :, jdx] = arm[:, None]
    jvfl = np.gradient(jpfl, axis=0)
    active_floss = arrays(jpfl, jvfl, _upright_g(), _standing_root())

    # active_jump: repeated vertical hops (crouch→launch→apex→land) with knee
    # extension bursts, upright, ZERO horizontal travel. Has real leg motion so
    # a stillness-rewarder cannot mistake it for a quiet stance.
    zj = np.full(T, 0.55)                     # crouched baseline
    jvj = np.zeros((T, E, J))
    for start in range(15, T, 35):            # ~3 hops
        for k in range(20):
            if start + k < T:
                zj[start + k] = 0.55 + 0.45 * np.sin(np.pi * k / 20)  # apex ≈1.0
                if k < 6:                     # launch: knees extend
                    for jdx in (2, 3):
                        jvj[start + k, :, jdx] = 6.0
    jpj = np.cumsum(jvj, axis=0) * 0.02
    rootj = np.zeros((T, E, 3)); rootj[..., 2] = zj[:, None]
    active_jump = arrays(jpj, jvj, _upright_g(), rootj)

    # §Ship 47: a realistic forward WALKER — upright, at STANDING height
    # (z≈0.70), travelling forward with FAST alternating hip/knee gait
    # swings (peak ≈6 rad/s, well above any kick threshold). This is the
    # exact Goodhart confound that stalled g1-kick-v3: a non-kicking gait
    # that a naive kick metric scores high (the on-disk gen_005 metric
    # scores it ~0.50). The existing `active` archetype could NOT catch
    # this — its joint velocities are ~0.1 rad/s (smooth, small-amplitude)
    # AND it sits at z=0.5, so a height/threshold-gated kick metric scores
    # it ~0. A valid STATIONARY-skill metric (kick/floss/jump) must score
    # the walker LOW; the family-scoped ceiling in validate_generated_metric
    # enforces it. NOT a negative for locomotion (there a walker is the
    # target) — see _STATIONARY_FAMILIES.
    jpw = np.zeros((T, E, J))
    phase = 2 * np.pi * 1.5 * t * 0.02       # 1.5 Hz gait
    jpw[:, :, 0] = (0.64 * np.sin(phase))[:, None]            # left hip pitch
    jpw[:, :, 2] = (0.51 * np.sin(phase))[:, None]            # left knee
    jpw[:, :, 1] = (0.64 * np.sin(phase + np.pi))[:, None]    # right hip pitch
    jpw[:, :, 3] = (0.51 * np.sin(phase + np.pi))[:, None]    # right knee
    jvw = np.gradient(jpw, axis=0) / 0.02
    rootw = np.zeros((T, E, 3)); rootw[..., 2] = 0.70
    rootw[..., 0] = (t * 0.04)[:, None]      # forward travel
    walker = arrays(jpw, jvw, _upright_g(), rootw)

    return {"still": still, "fallen": fallen, "chaotic": chaotic,
            "active": active, "upright_flail": upright_flail,
            "active_kick": active_kick, "active_floss": active_floss,
            "active_jump": active_jump, "walker": walker}


def _score(fn, arrays, meta) -> float:
    out = fn(arrays, {"max_episode_steps": T, "rollout_num_envs": E,
                      "step_dt": 0.02}, meta)
    return float(out.get("spec_score", float("nan")))


# §Ship 47: skills performed from a roughly stationary base. For these a
# forward WALKER is a Goodhart distractor that must score LOW; for the
# locomotion family a walker IS the target, and for an unresolved family
# (None) we don't gate (the calibration firewall is the task-validity check),
# so the smell-test never false-rejects an ambiguous goal.
_STATIONARY_FAMILIES = frozenset({"kick", "floss", "jump"})


def validate_generated_metric(
    source: str,
    module_path: Path | str,
    *,
    spread_min: float = 0.1,
    distractor_ceiling: float = 0.3,
    behavior_goal: Optional[str] = None,
    robot_hint: Optional[str] = None,
    robot_joint_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run all MUST-HAVE gates on a generated metric. `source` is the
    module text (for static gates); `module_path` is where it's been
    written (for the runtime gates). Returns
    `{ok, gates, reasons, archetype_scores, family, required_roles}`.
    Never raises — a crashing metric is a failed gate, not an exception.

    §Ship 41: `behavior_goal`/`robot_hint` (optional) resolve a behavior
    FAMILY so the non-degeneracy gate anchors a non-locomotion metric
    (kick/jump/floss) against a behavior-appropriate positive archetype
    instead of a hard-coded forward-walker. Both default `None` → today's
    behavior (any of the four positives may anchor the metric).

    §Ship 49: `robot_joint_names` (optional — the ACTUAL robot's joint_names,
    sourced from the manifest at launch) gates the metric's declared
    `REQUIRED_JOINT_ROLES`: every role must resolve to exactly one joint on
    THIS robot, or the metric is rejected pre-project. Plus two new gates that
    run regardless: a static ban on hard-coded integer joint indices and a
    permutation-robustness check (a metric that reads joints by index rather
    than name swings when the joint axis is relabelled)."""
    gates: dict[str, bool] = {}
    reasons: list[str] = []
    family = resolve_behavior_family(behavior_goal, robot_hint)
    required_roles: list[str] = []

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

    # §Ship 49: static ban on hard-coded integer joint indices.
    raw_idx = _raw_joint_index_violations(source)
    gates["no_raw_joint_index"] = not raw_idx
    reasons += [f"[joint-index] {r}" for r in raw_idx]

    # Static gates must pass before we exec the module.
    if not (gates["ast_safety"] and gates["defines_compute_spec"]):
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}, "family": family,
                "required_roles": required_roles}

    try:
        mod = load_generated_module(module_path)
        fn = getattr(mod, GENERATED_FN_NAME, None)
        if not callable(fn):
            raise ValueError(f"no callable {GENERATED_FN_NAME}()")
    except Exception as e:  # noqa: BLE001
        gates["loads"] = False
        reasons.append(f"[load] {type(e).__name__}: {e}")
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}, "family": family,
                "required_roles": required_roles}
    gates["loads"] = True
    required_roles = read_required_roles(mod)

    # §Ship 49: required-roles gate. When the ACTUAL robot's joint_names are
    # known (manifest at launch), every declared role must resolve to exactly
    # one joint on this robot — else reject pre-project ("metric needs
    # swing_hip_pitch; robot exposes no matching joint"). When the robot is
    # unknown we cannot reject (the runtime resolution + permutation gate are
    # the backstop), so the gate passes informationally.
    if required_roles and robot_joint_names:
        rr = resolve_joint_roles(list(robot_joint_names), required_roles)
        gates["joint_roles_resolve"] = rr.ok
        if not rr.ok:
            reasons += [f"[joint-roles] {p}" for p in rr.problems()]

    # Archetypes run on the synthetic 12-joint biped. Inject the metric's
    # declared roles LENIENTLY (the synthetic body has no roll/yaw columns,
    # so an anatomically valid roll/yaw role still maps to its segment) so a
    # role-based metric can score the battery.
    meta = {"joint_names": list(_NAMES_12)}
    inject_joint_roles(meta, required_roles, lenient=True)
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

    # §Ship 49: permutation-robustness — relabel the joint axis (names AND the
    # joint_pos/joint_vel columns, CONSISTENTLY) and re-score. A metric that
    # reads joints by NAME/role follows the relabelling to the same physical
    # joints → invariant; a metric that hard-codes a column reads a different
    # joint → its score swings. This is exactly the §3A shuffle experiment
    # turned into a gate (an index-sensitive metric silently mis-scores the
    # moment the real robot's joint order differs from the synthetic battery).
    perm = list(range(J - 1, -1, -1))           # reverse: a non-trivial relabel
    pmeta = {"joint_names": [_NAMES_12[perm[i]] for i in range(J)]}
    inject_joint_roles(pmeta, required_roles, lenient=True)
    robust = True
    for name, arrays in arche.items():
        base = scores.get(name, float("nan"))
        if not np.isfinite(base):
            continue
        try:
            ps = _score(fn, _permute_joint_arrays(arrays, perm), pmeta)
        except Exception as e:  # noqa: BLE001 — a crash under relabel = not robust
            robust = False
            reasons.append(f"[robustness] raised under joint relabel on "
                           f"'{name}': {type(e).__name__}: {e}")
            break
        if not np.isfinite(ps) or abs(ps - base) > 1e-6:
            robust = False
            reasons.append(
                f"[robustness] index-sensitive joint access: '{name}' scored "
                f"{base:.4f} but {ps:.4f} after a consistent joint relabel — "
                f"the metric reads joints by column, not by name "
                f"(use meta['joint_roles'])")
            break
    gates["joint_index_robust"] = robust

    # 5: non-degeneracy — the metric must score SOME competent behavior above
    # EVERY degenerate one, with spread. §Ship 41: positives span all behavior
    # families (locomotion `active` + kick/floss/jump) so a non-locomotion
    # metric isn't measured against a forward-walker. §Ship 41 review: the
    # resolved `family` does NOT NARROW this smell-test — narrowing falsely
    # rejected good metrics whose goal mis-resolved (e.g. "Hopper"→jump, or a
    # compound "walk forward and kick"). The metric passes if ANY positive
    # beats the negatives; `family` only selects the calibration ground truth
    # downstream (the firewall enforces task-validity). NEGATIVES that must
    # lose now include `chaotic` (upright random thrashing — the HIGHEST peak
    # joint speed of any archetype), so a peak-speed reward-hack (which scores
    # chaotic above the real positives) is rejected — closing the
    # stand-and-thrash bypass the review found.
    positive_keys = ("active", "active_kick", "active_floss", "active_jump")
    negative_keys = ("still", "fallen", "upright_flail", "chaotic")

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
        pos = {k: finite[k] for k in positive_keys if k in finite}
        if pos:
            best_key = max(pos, key=lambda k: pos[k])
            best_pos = pos[best_key]
        else:
            best_key, best_pos = None, float("-inf")
            nondegen = False
            reasons.append("[nondegeneracy] no positive archetype scored finite")
        for low in negative_keys:
            if low in finite and finite[low] >= best_pos:
                nondegen = False
                reasons.append(
                    f"[nondegeneracy] '{low}' ({finite[low]:.3f}) scores >= the "
                    f"best positive '{best_key}' ({best_pos:.3f}) — rewards the "
                    f"wrong behavior")
        # §Ship 47: stationary-skill walker ceiling. For kick/floss/jump a
        # forward WALKER must score below an ABSOLUTE ceiling — a metric that
        # gives a walker substantial credit is rewarding locomotion, not the
        # skill (the g1-kick-v3 0.59 Goodhart: gen_005 scored a non-kicking
        # walker ~0.59, yet its `active_kick` 0.90 kept it above every negative
        # via the relative check above, so only an absolute ceiling catches
        # it). Scoped to stationary families so a locomotion metric (walker =
        # target) and an unresolved goal (family None) are never false-rejected.
        if (family in _STATIONARY_FAMILIES and "walker" in finite
                and finite["walker"] > distractor_ceiling):
            nondegen = False
            reasons.append(
                f"[nondegeneracy] forward-walker 'walker' ({finite['walker']:.3f}) "
                f"scores above the {distractor_ceiling} ceiling for the stationary "
                f"'{family}' skill — the metric rewards walking, not the behavior")
    gates["nondegeneracy"] = nondegen

    ok = all(gates.values())
    return {"ok": ok, "gates": gates, "reasons": reasons,
            "archetype_scores": scores, "family": family,
            "required_roles": required_roles}
