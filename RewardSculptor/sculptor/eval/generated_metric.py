"""Auto-generated, per-task objective metrics (§Ship 35).

A generated metric is a small Python module that MIRRORS the hand-authored
spec contract in `spec_metrics.py`:

    def compute_spec(arrays, behavior, meta) -> dict[str, float]
        # returns {"spec_score": float in [0,1], ...sub-components}

computed PURELY from the persisted physical rollout arrays (joint_pos,
joint_vel, projected_gravity_b, root_link_pos_w + behavior.json +
joint_names) — NEVER from LLM judgment. It is generated from the NL goal,
then put through a validation chain (safety/contract/determinism/bounds +
a monotonicity audit) and an independent LLM review, and must EARN
steer-rights via calibration (Spearman vs a hand-authored ground-truth
metric) before it is allowed to drive selection. Until then it runs
OBSERVE-ONLY (computed + displayed, no influence). This module is the
RUNTIME side (load + compute + resolve); generation lives in `metric_gen`,
validation in `metric_validate`, calibration in `metric_calibration`.

SECURITY (§round-9/10): the metric is UNTRUSTED LLM-authored code, and `metric_validate.
_ast_safety` is the containment gate — hardened to block the numpy IO/pickle surface,
native submodules, `def __reduce__`-style gadgets, dunder tokens in string literals, and
`allow_pickle`, plus the round-28/29 frame/generator/coroutine/cython introspection prefix
family (`gi_`/`cr_`/`ag_`/`f_`/`tb_`/`func_`) that reaches the live builtins dict. It is
enforced inside `load_generated_module` (§round-10), the single chokepoint EVERY exec path
shares (validation, the per-rollout runtime scorer, calibration, the best-of-N discriminator,
role-read) — so the gate runs before EVERY exec, not only at validation time. That is still a
STATIC allow/deny gate, NOT a proof of containment, and the rounds-27→29 hardening EMPIRICALLY
confirmed it cannot be proven complete: two CRITICAL escape CLASSES were reproduced in
consecutive rounds (round-28 `gi_frame.f_builtins`, round-29 numpy-cython `func_globals`),
each slipping a freshly-patched denylist. The DURABLE containment (the convergence path) is to
run the module exec + `compute_spec` in a restricted SUBPROCESS.

  DESIGN CONSTRAINT (round-29, code-verified): the cheap in-process "empty/curated
  `__builtins__`" variant is NOT viable here — numpy's ndarray methods (`.mean`/`.max`/`.std`)
  internally call `__import__`, so an emptied `__builtins__` breaks legitimate metrics.
  (`mission_runtime._evaluate_success_criterion` deliberately KEEPS real `__builtins__` for
  exactly this reason and relies on a strict no-attribute-access AST validator instead — an
  approach a numpy-computation metric cannot use.) So fix B must be a true PROCESS sandbox with
  real builtins INSIDE but the process locked down OUTSIDE: a separate process (crash/hang/
  memory isolation; `RLIMIT_CPU`/`RLIMIT_AS`/timeout), restricted cwd, and — for containment
  against `os.system`/`unlink`/network rather than just robustness — a syscall filter
  (seccomp-bpf) or `bubblewrap`/`nsjail`. It also requires reading `REQUIRED_JOINT_ROLES`
  STATICALLY (AST, no exec) since a top-level exploit runs at module-exec time. This is a
  dedicated reviewed increment (high blast radius over the score path; must stay score-identical
  vs the 937-test corpus). Until it lands, the comprehensive denylist above is the live gate and
  the realistic threat is low (the metric source is SYSTEM-generated from the user's own goal,
  not adversarial third-party input).
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from sculptor.eval.joint_resolver import (
    assert_name_axis_contract,
    resolve_joint_roles,
)
from sculptor.eval.spec_metrics import _CAPTURE_KEYS, _SPEC_FNS, make_spec_fitness_fn
from sculptor.world.channels import (
    BASE_METRIC_ARRAYS,
    ChannelCatalog,
    resolve_channel_catalog,
    validate_trajectory_channels,
)

#: The function a generated metric module must define.
GENERATED_FN_NAME = "compute_spec"

#: §Ship 49: optional module-level constant a generated metric declares to
#: name the canonical joint ROLES it needs (e.g.
#: `REQUIRED_JOINT_ROLES = ["left_hip_pitch", "left_knee"]`). The runtime
#: resolves these against the rollout's live `joint_names` and hands the
#: metric `meta["joint_roles"] = {role: index}` — so the metric never does
#: its own brittle name matching, and a role that can't be resolved HARD-
#: FAILS loudly instead of silently scoring the wrong joint.
REQUIRED_ROLES_ATTR = "REQUIRED_JOINT_ROLES"

#: Physical rollout arrays a generated metric MAY read (the full contract —
#: the validator enforces a metric references only these). Mirrors the
#: spec_metrics.py array contract; kept here as the single allow-list a
#: generated metric is constrained to.
_LEGACY_ALLOWED_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "projected_gravity_b",
    "root_link_pos_w",
    # §Metric-quality laws (LAW 3/4): per-foot ground contact (T, E) and
    # foot position in the PELVIS frame (T, E, 3) — anterior (x) component
    # is the signed forward-kick direction; the contact schedule
    # distinguishes a brief kick from a sustained one-leg balance. Present
    # only for biped tasks whose robot exposes left_foot/right_foot sites
    # (the runner omits them otherwise — guard with arrays.get + None).
    "left_foot_contact",
    "right_foot_contact",
    "left_foot_pos_b",
    "right_foot_pos_b",
)
# Guard the externally-visible legacy tuple while moving its canonical
# definition to the catalog module shared by compiler and runtime.
assert _LEGACY_ALLOWED_ARRAYS == BASE_METRIC_ARRAYS
ALLOWED_ARRAYS = BASE_METRIC_ARRAYS


def resolved_allowed_arrays(
    channel_catalog: ChannelCatalog | dict[str, Any] | Path | str | None = None,
) -> tuple[str, ...]:
    """Resolve the exact base-plus-project metric surface."""
    catalog = resolve_channel_catalog(channel_catalog)
    return catalog.allowed_metric_arrays() if catalog else ALLOWED_ARRAYS


def _catalog_hash_from_npz(npz: Any) -> str | None:
    if "channel_catalog_hash" not in npz.files:
        return None
    raw = np.asarray(npz["channel_catalog_hash"])
    if raw.size != 1:
        raise ValueError("channel_catalog_hash must be a scalar")
    value = raw.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("ascii", "strict")
    return str(value)


def load_generated_module(module_path: Path | str):
    """Import a generated-metric module and return the MODULE. Uses a unique
    module name so re-loading an edited metric never hits a stale
    sys.modules entry.

    §round-10 SECURITY: the static safety gate (`_ast_safety`) is enforced HERE,
    at the single chokepoint EVERY exec path shares (the per-rollout runtime
    scorer, calibration, the best-of-N discriminator, role-read) — so untrusted
    LLM-authored source is screened on every path, not only at validation time. A
    violation refuses to exec (raise) rather than running arbitrary code; callers
    that must never raise (compute_generated_metric) already wrap this in
    try/except and degrade to an honest 0.0. (`_ast_safety` is imported lazily:
    metric_validate imports this function, so a top-level import would cycle.)"""
    from sculptor.eval.metric_validate import _ast_safety  # lazy — avoid import cycle
    module_path = Path(module_path)
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ImportError(f"cannot read generated metric at {module_path}: {e}")
    violations = _ast_safety(source)
    if violations:
        raise ImportError(
            f"refusing to exec untrusted metric {module_path}: "
            f"_ast_safety violations: {violations}")
    spec = importlib.util.spec_from_file_location(
        f"_genmetric_{module_path.stem}_{abs(hash(str(module_path)))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generated metric at {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_generated_metric(module_path: Path | str) -> Callable[..., dict]:
    """Import a generated-metric module and return its `compute_spec`."""
    mod = load_generated_module(module_path)
    fn = getattr(mod, GENERATED_FN_NAME, None)
    if not callable(fn):
        raise ValueError(
            f"generated metric {module_path} lacks a callable "
            f"{GENERATED_FN_NAME}()"
        )
    return fn


def read_required_roles_static(source: str) -> list[str]:
    """§Fix-B down-payment: extract `REQUIRED_JOINT_ROLES = ["...", ...]` from metric SOURCE
    by STATIC AST parse — NO exec. The contract is a top-level assignment of a LITERAL list/
    tuple of strings; anything else (absent, non-literal, non-string elements) → `[]` (treated
    as undeclared → the legacy self-matching path). Reading roles used to require exec'ing the
    untrusted module (`load_generated_module`) just to `getattr` a constant — a top-level
    exploit ran at that exec. Parsing the literal statically removes that exec entirely (the
    module is still gated by `_ast_safety` + exec'd in the sandbox when actually SCORED)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in tree.body:                       # top-level statements only
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == REQUIRED_ROLES_ATTR for t in targets):
            continue
        rhs = node.value
        if not isinstance(rhs, (ast.List, ast.Tuple)) or rhs is None:
            return []                            # non-literal RHS → undeclared (legacy path)
        roles: list[str] = []
        for el in rhs.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, str):
                roles.append(el.value)
            else:
                return []                        # any non-string-literal element → undeclared
        return roles
    return []


def read_required_roles(module_or_path) -> list[str]:
    """The canonical joint roles a metric declares via `REQUIRED_JOINT_ROLES`
    (empty list when undeclared — a legacy metric that does its own matching).

    Accepts a path OR a loaded module; BOTH resolve to the module SOURCE and parse it
    STATICALLY (`read_required_roles_static`, NO exec — §Fix-B: never exec untrusted code
    just to read metadata). §round-30 [FALSE GRANT] fix: the path branch was static but the
    module branch read the LIVE `getattr`, so a reassigned/mutated `REQUIRED_JOINT_ROLES`
    was VALIDATED under the static first-literal (calibrate_task_derived reads via path) yet
    DEPLOYED under the live last-binding (compute_generated_metric read via module) — a
    metric validated as a knee-reader was granted while deploying as a shoulder-reader. A
    single static source of truth makes validated == deployed. (Falls back to live `getattr`
    only when a module exposes no readable `__file__` — a synthetic/in-memory module.)"""
    src_path: Optional[Path] = None
    if isinstance(module_or_path, (str, Path)):
        src_path = Path(module_or_path)
    else:
        f = getattr(module_or_path, "__file__", None)
        if f:
            src_path = Path(f)
    if src_path is not None:
        try:
            return read_required_roles_static(src_path.read_text(encoding="utf-8"))
        except OSError:
            return []
    # No source available (synthetic module) — live getattr fallback.
    roles = getattr(module_or_path, REQUIRED_ROLES_ATTR, None)
    if not roles:
        return []
    try:
        return [str(r) for r in roles]
    except TypeError:
        return []


def inject_joint_roles(
    meta: dict[str, Any], required_roles: list[str], *, lenient: bool = False,
) -> Optional[str]:
    """Resolve `required_roles` against `meta["joint_names"]` and set
    `meta["joint_roles"] = {role: index}` IN PLACE. Returns None on success,
    or a human-readable error string when a role is missing/ambiguous (the
    caller then HARD-FAILS the score rather than letting the metric read the
    wrong joint). A metric with no required roles is a no-op."""
    if not required_roles:
        meta.setdefault("joint_roles", {})
        return None
    names = list(meta.get("joint_names") or [])
    if not names:
        return (f"metric requires joint roles {required_roles} but the "
                f"rollout persisted no joint_names")
    res = resolve_joint_roles(names, required_roles, lenient=lenient)
    meta["joint_roles"] = dict(res.resolved)
    if not res.ok:
        return "; ".join(res.problems())
    return None


def compute_generated_metric(
    module_path: Path | str,
    rollout_dir: Path | str,
    *,
    behavior: Optional[dict] = None,
    channel_catalog: ChannelCatalog | dict[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Run a generated metric on a rollout dir. Mirrors
    `compute_spec_metrics`' defensive loading: NEVER raises — a bad/missing
    artifact or a crashing metric yields `{"spec_score": 0.0, "error": ...}`
    so the loop aggregates an honest zero instead of dying."""
    rollout_dir = Path(rollout_dir)
    try:
        catalog = resolve_channel_catalog(channel_catalog)
        if behavior is None:
            bpath = rollout_dir / "behavior.json"
            behavior = (
                json.loads(bpath.read_text(encoding="utf-8"))
                if bpath.is_file() else {}
            )
        meta: dict[str, Any] = {}
        limits_path = rollout_dir / "mjcf_limits.json"
        if limits_path.is_file():
            try:
                limits = json.loads(limits_path.read_text(encoding="utf-8"))
                names = limits.get("joint_names") or []
                if names:
                    meta["joint_names"] = [str(n) for n in names]
            except Exception:  # noqa: BLE001 — names are an upgrade, not a dep
                pass
        arrays: dict[str, np.ndarray] = {}
        npz_path = rollout_dir / "trajectory.npz"
        if npz_path.is_file():
            with np.load(npz_path) as z:
                trajectory_catalog_hash = _catalog_hash_from_npz(z)
                if catalog is not None:
                    if trajectory_catalog_hash is None:
                        raise ValueError(
                            "trajectory is missing channel_catalog_hash for "
                            f"catalog {catalog.catalog_hash}")
                    if trajectory_catalog_hash != catalog.catalog_hash:
                        raise ValueError(
                            "trajectory channel catalog hash mismatch: "
                            f"{trajectory_catalog_hash} != {catalog.catalog_hash}")
                # Load every ALLOWED array that's present — the generated
                # metric may use any subset; missing ones simply aren't
                # there (the validator forbids referencing absent arrays).
                for k in resolved_allowed_arrays(catalog):
                    if k in z.files:
                        arrays[k] = z[k]
                if catalog is not None:
                    project_arrays = {
                        name: arrays[name] for name in catalog.names()
                        if name in arrays
                    }
                    channel_errors = validate_trajectory_channels(
                        project_arrays, catalog,
                        catalog_hash=trajectory_catalog_hash,
                        strict_unknown=True, require_all=True)
                    if channel_errors:
                        raise ValueError(
                            "invalid catalog trajectory channels: "
                            + "; ".join(channel_errors))
        mod = load_generated_module(module_path)
        fn = getattr(mod, GENERATED_FN_NAME, None)
        if not callable(fn):
            return {"spec_score": 0.0,
                    "error": f"metric lacks a callable {GENERATED_FN_NAME}()"}
        # §Ship 49: resolve the metric's declared joint roles against THIS
        # rollout's live joint_names and hand them over as meta["joint_roles"].
        # The name↔buffer order-contract is asserted first (a names list that
        # doesn't span the joint axis is untrustworthy → degrade, don't index
        # by it); an unresolvable role HARD-FAILS to an honest, observable 0.0
        # instead of silently scoring a wrong/foreign joint.
        roles = read_required_roles(mod)
        if roles:
            jarr = arrays.get("joint_pos")
            if jarr is None:
                jarr = arrays.get("joint_vel")
            names = list(meta.get("joint_names") or [])
            if jarr is not None and names:
                try:
                    assert_name_axis_contract(names, jarr.shape[2])
                except Exception as e:  # noqa: BLE001 — contract break → drop names
                    meta["joint_names"] = []
                    return {"spec_score": 0.0,
                            "error": f"joint-name contract: {e}"}
            role_err = inject_joint_roles(meta, roles)
            if role_err:
                return {"spec_score": 0.0,
                        "joint_roles": meta.get("joint_roles", {}),
                        "error": f"unresolved joint roles: {role_err}"}
        out = fn(arrays, behavior, meta)
        if not isinstance(out, dict) or "spec_score" not in out:
            return {"spec_score": 0.0,
                    "error": "metric did not return a dict with spec_score"}
        score = float(out.get("spec_score", 0.0) or 0.0)
        if not np.isfinite(score):
            return {"spec_score": 0.0, "error": "spec_score not finite"}
        out["spec_score"] = float(np.clip(score, 0.0, 1.0))
        capture = {k: behavior.get(k) for k in _CAPTURE_KEYS if k in behavior}
        result = {**out, "capture": capture}
        # §Ship 49: surface the resolved role→index map so the diagnoser /
        # realism audit can verify the SAME joints were read each run (drift
        # across robots/adapters is then observable, not silent).
        if meta.get("joint_roles"):
            result.setdefault("joint_roles", dict(meta["joint_roles"]))
        return result
    except Exception as e:  # noqa: BLE001 — zero, observably
        return {"spec_score": 0.0, "error": f"{type(e).__name__}: {e}"}


def make_generated_fitness_fn(
    module_path: Path | str,
    *,
    channel_catalog: ChannelCatalog | dict[str, Any] | Path | str | None = None,
) -> Callable[[Any], float]:
    """`fitness_fn(iter_dir) -> float` for a generated metric module —
    scores `iter_dir/rollout` (0.0 on any failure)."""
    module_path = Path(module_path)

    def _fitness(iter_dir: Any) -> float:
        result = compute_generated_metric(
            module_path, Path(iter_dir) / "rollout",
            channel_catalog=channel_catalog)
        return float(result.get("spec_score", 0.0) or 0.0)

    def _detail(iter_dir: Any) -> dict:
        # §Ship 36 (F2): full component breakdown for the diagnoser. Rides
        # on the fitness fn (no new threaded param). Never raises.
        try:
            return compute_generated_metric(
                module_path, Path(iter_dir) / "rollout",
                channel_catalog=channel_catalog)
        except Exception:  # noqa: BLE001 — breakdown is advisory, never fatal
            return {}

    def _detail_dir(rollout_dir: Any) -> dict:
        # §Selection statistics: score an ARBITRARY rollout dir (multi-seed
        # evaluation rolls into `rollout_eval_<k>/` beside `rollout/`;
        # fresh-seed re-eval into `rollout_fresh_<j>/`). Never raises.
        try:
            return compute_generated_metric(
                module_path, Path(rollout_dir),
                channel_catalog=channel_catalog)
        except Exception:  # noqa: BLE001 — advisory, never fatal
            return {}

    _fitness.detail = _detail  # type: ignore[attr-defined]
    _fitness.detail_dir = _detail_dir  # type: ignore[attr-defined]
    # §Ship 54-pre (#12): the metric's held-out observable surface for the
    # shaping↔metric partition gate. Parse which ALLOWED_ARRAYS the module
    # actually references (precise flags); fall back to the full contract on any
    # read failure. Never raises — attribute is advisory.
    _fitness.metric_observables = _generated_metric_observables(  # type: ignore[attr-defined]
        module_path, channel_catalog=channel_catalog)
    catalog = resolve_channel_catalog(channel_catalog)
    _fitness.channel_catalog_hash = (  # type: ignore[attr-defined]
        catalog.catalog_hash if catalog else None)
    return _fitness


def _generated_metric_observables(
    module_path: Path | str,
    *,
    channel_catalog: ChannelCatalog | dict[str, Any] | Path | str | None = None,
) -> frozenset[str]:
    """The subset of `ALLOWED_ARRAYS` a generated metric's source references —
    its held-out surface for the partition gate. Conservative: on any read
    failure, return the full `ALLOWED_ARRAYS` contract (the metric is permitted
    to read any of them)."""
    try:
        src = Path(module_path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — fall back to the full contract
        return frozenset(resolved_allowed_arrays(channel_catalog))
    allowed = resolved_allowed_arrays(channel_catalog)
    referenced = frozenset(a for a in allowed if a in src)
    return referenced or frozenset(allowed)


def resolve_fitness_fn(
    spec: str,
    *,
    channel_catalog: ChannelCatalog | dict[str, Any] | Path | str | None = None,
) -> Callable[[Any], float]:
    """Resolve a fitness spec to a `fitness_fn(iter_dir) -> float`. `spec`
    is either a built-in spec-metric name (e.g. "go1_trot") or a filesystem
    path to a generated-metric .py module. Raises on anything else (fail
    fast before GPU work)."""
    if spec in _SPEC_FNS:
        return make_spec_fitness_fn(spec)
    p = Path(spec)
    if p.suffix == ".py" and p.is_file():
        return make_generated_fitness_fn(p, channel_catalog=channel_catalog)
    raise KeyError(
        f"unknown fitness metric {spec!r}: not a built-in "
        f"{sorted(_SPEC_FNS)} and not a generated-metric .py path that exists"
    )
