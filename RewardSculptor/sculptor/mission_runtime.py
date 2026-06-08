"""sculptor/mission_runtime.py — Ship 16 mission runtime.

Hosts the runtime pieces the orchestrator (`sculpt.mission_run`) needs:

  * `_build_criterion_namespace(iter_outcome)` — loads `behavior.json` +
    `trajectory.npz` from the iter's rollout dir, packages them into the
    namespace a success_criterion Python expression can reference.
  * `_evaluate_success_criterion(criterion, namespace)` — ast-parsed
    safe-eval that rejects attribute access, function calls outside the
    math allow-list, and any access to `__builtins__`.
  * `StageResult` / `MissionResult` dataclasses — shape of what
    `mission_run` returns to the caller.

Keeping this separate from `sculpt.py` both for size reasons (sculpt.py
is already 1500+ LOC) and because the criterion evaluator is a pure
function that's easy to unit-test in isolation.

Success-criterion namespace (consumable inside the criterion expression):
  - `metric`            — scalar = `IterOutcome.primary_metric`.
  - `behavior[<key>]`   — scalar field from `behavior.json` (e.g.,
                          `mean_return`, `mean_episode_length`,
                          `max_episode_length`, `n_episodes`).
  - `components[<name>]` — mean of a SCULPTOR reward component the
                          reward_seed_prompt introduces. Read from the
                          training-side `reward_trajectory.json` (the file
                          diagnose surfaces to Claude), merged over the
                          ENVIRONMENT's intrinsic `trajectory.npz
                          ["reward_term__<name>"]` terms (which differ on
                          mjlab, where the sculptor reward is layered on the
                          task's own terms).
  - `trajectory[<key>]` — raw per-step numpy array from trajectory.npz.
  - `info[<key>]`       — ALIAS for `trajectory[<key>]`. Kept for Ship 14
                          prompt compatibility; the two names resolve to
                          the same backing dict.

Design reference: CurricuLLM (arXiv:2409.18382) §3.3 — their evaluator LLM
judges stage policies by looking at rollout trajectory statistics. We
narrow this to a deterministic Python expression for Ship 16; an LLM
evaluator is a later follow-up (Ship 20+).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── Criterion namespace: keys statically validated at decompose time ─
#
# These are the trajectory.npz keys the mjlab runner persists today
# (see _mjlab_runner.py `_cmd_rollout` around line 1078-1094). A
# criterion using `info['<key>']` or `trajectory['<key>']` is checked
# against this set at decompose-time so Claude can't write criteria
# that reference fields the runner doesn't produce.
#
# NOTE: this is deliberately a hardcoded SUPERSET, not the adapter's
# `expected_info_keys`. The adapter's info_keys describes what the
# per-step reward function sees at RUNTIME; trajectory.npz persists a
# different (sometimes smaller) subset. Don't conflate the two.
PERSISTED_TRAJECTORY_KEYS: frozenset[str] = frozenset({
    # Always present.
    "rewards",
    "episode_id",
    # Optional (present only when the env exposes an articulated entity
    # + reward_manager — i.e., mjlab humanoids / quadrupeds, not
    # Cartpole). The evaluator handles missing keys with a clear error.
    "joint_pos", "joint_vel", "action",
    "actuator_force", "projected_gravity_b", "root_link_pos_w",
})

# Scalar fields persisted in behavior.json.
BEHAVIOR_KEYS: frozenset[str] = frozenset({
    "n_episodes", "mean_return",
    "mean_episode_length", "max_episode_length",
})

# Bare identifiers the criterion can reference as non-subscript names.
# `metric` maps to IterOutcome.primary_metric. Math helpers mirror the
# allow-list used in sculptor.edit._ALLOWED_MATH to avoid drift.
BARE_IDENTIFIERS: frozenset[str] = frozenset({
    "metric",
    # Math helpers — safe, deterministic, no side effects.
    "abs", "min", "max", "sum", "len", "round", "float", "int", "bool",
    # Tensor-ish methods that numpy arrays expose as attributes; we'll
    # allow `.mean()`, `.max()`, etc. via the ast.Attribute checker.
})

# Method names allowed on array objects inside the criterion. Anything
# NOT on this list is blocked, preventing e.g. `.tobytes()`,
# `.view(...)`, `.__reduce__()` and other pickle / serialization
# escape hatches.
SAFE_ATTRIBUTE_METHODS: frozenset[str] = frozenset({
    "mean", "sum", "max", "min", "std", "any", "all",
    # `.item()` is valid numpy — extracts a Python scalar from a
    # 0-d array; safe.
    "item", "shape", "size",
    # `dict.get(key, default)` — lets a criterion soft-probe a key that
    # the reward MAY not have produced yet, e.g.
    # `components.get('hip_sway_osc', 0.0) > 0.5`, instead of a bare
    # `components['hip_sway_osc']` that KeyErrors mid-mission. Safe:
    # the AST walker already restricts the receiver to namespace dicts.
    "get",
    # §Ship 21c: `.astype(...)` is the numpy way to do what users
    # familiar with torch reach for via `.float()`. Allow it so
    # `(arr > 0.5).astype(float).mean()` works as a fallback when
    # users want explicit casting (`.mean()` on a bool array
    # already returns a float, so `.astype` is rarely necessary).
    "astype",
    # §Ship 21c: `.float()` REMOVED. It was a torch tensor method
    # that crashed at eval time on numpy arrays — Sam's robot-
    # flossing run failed at the very last criterion check after
    # 10+ hours because Claude wrote `(arr > 0.65).float().mean()`.
    # The decompose-time validator now rejects `.float()` early
    # (mission.py _validate_success_criterion); this set is the
    # belt-and-suspenders runtime guard.
})


# ── Result types ─────────────────────────────────────────────────────
@dataclass
class StageResult:
    """Outcome of one stage's `sculpt_run` + success-criterion eval.

    `status` mirrors `Stage.status` but represents the post-run terminal
    state: "succeeded" / "failed" / "skipped". "skipped" is reserved for
    Ship-17's fail-through; Ship-16 only emits succeeded/failed.
    """

    stage_name: str
    status: str
    iterations_used: int
    final_policy_path: Optional[str]
    final_reward_path: Optional[str]
    criterion_satisfied: bool
    criterion_error: Optional[str] = None
    last_iter_metric: Optional[float] = None
    # Free-form reason string when status == "failed" (e.g.,
    # "no_checkpoint", "criterion_not_met", "training_errored").
    failure_reason: Optional[str] = None


@dataclass
class MissionResult:
    """Aggregated mission-run output. Caller-facing shape."""

    mission_goal: str
    stage_results: list[StageResult] = field(default_factory=list)
    completed: bool = False                    # every stage succeeded
    halted_at_stage: Optional[str] = None      # first stage that failed
    halted_reason: Optional[str] = None


# ── Safe-eval infrastructure ─────────────────────────────────────────
class CriterionEvalError(RuntimeError):
    """Raised by `_evaluate_success_criterion` on any semantic or safety
    violation (bad identifier, unsafe attribute, unresolvable subscript,
    etc.). Caller should mark the stage failed with this as the reason.
    """


class CriterionMissingKeyError(CriterionEvalError):
    """The criterion subscripted a namespace dict (behavior / components /
    info / trajectory) with a key the adapter / reward did NOT produce this
    iter, e.g. `components['hip_sway_osc']` when the reward never emitted a
    `hip_sway_osc` term.

    This is semantically distinct from a *broken* criterion: the measured
    quantity is simply absent, which means the criterion is NOT satisfied —
    not that it is unparseable or unsafe. Callers route this to the
    recoverable `criterion_not_met` (re-decomposable) outcome rather than the
    fatal `criterion_errored`, so one mistyped/early-referenced key can't
    halt a multi-hour mission.
    """


def _validate_criterion_ast(
    tree: ast.Expression,
    *,
    namespace_keys: set[str],
) -> None:
    """Walk the AST, reject anything unsafe or ungrounded.

    Rules:
      * Only ast.Name, ast.Subscript, ast.Attribute, ast.BoolOp,
        ast.UnaryOp, ast.BinOp, ast.Compare, ast.Call, ast.Constant are
        allowed at any depth. Reject ast.Lambda, ast.FunctionDef, etc.
      * ast.Name identifiers must be in `namespace_keys`.
      * ast.Attribute access is limited to SAFE_ATTRIBUTE_METHODS.
      * ast.Call targets must be either bare identifiers in
        `namespace_keys` (the math helpers) or method calls on allowed
        attributes.
    """
    ALLOWED_NODES = (
        ast.Expression, ast.Name, ast.Subscript, ast.Attribute,
        ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Call,
        ast.Constant, ast.Index, ast.Slice, ast.Load, ast.Store,
        # ast.Tuple is needed for `trajectory[..., 2]` and similar
        # multi-axis subscripts (the index `..., 2` parses as a tuple).
        # Tuple by itself is harmless — element nodes are still walked.
        ast.Tuple,
        # boolean / arithmetic / comparison ops
        ast.And, ast.Or, ast.Not,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.USub, ast.UAdd,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    # Belt-and-suspenders: the isinstance(ALLOWED_NODES) check above
    # already rejects unknown nodes, but make the intent explicit for
    # nodes a future Python-syntax addition might introduce. If
    # comprehensions / walrus / lambdas / starred unpacking ever land
    # in `ALLOWED_NODES`, this guard fires to remind the editor that
    # they were considered and rejected.
    REJECTED_NODES = (
        ast.Lambda, ast.NamedExpr, ast.Starred,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
    )

    for node in ast.walk(tree):
        if isinstance(node, REJECTED_NODES):
            raise CriterionEvalError(
                f"criterion contains explicitly-rejected AST node "
                f"{type(node).__name__}. Comprehensions, lambdas, "
                f"walrus, starred-unpack, and assignment are not "
                f"allowed in criteria — express your check as a "
                f"single boolean expression."
            )
        if not isinstance(node, ALLOWED_NODES):
            raise CriterionEvalError(
                f"criterion contains disallowed AST node "
                f"{type(node).__name__}. Only arithmetic / comparisons / "
                f"boolean ops / subscripts / safe attribute access are "
                f"permitted. No lambdas, no assignment, no imports."
            )
        if isinstance(node, ast.Attribute):
            if node.attr not in SAFE_ATTRIBUTE_METHODS:
                raise CriterionEvalError(
                    f"criterion uses disallowed attribute "
                    f".{node.attr!r}. Allowed: "
                    f"{sorted(SAFE_ATTRIBUTE_METHODS)}"
                )
        if isinstance(node, ast.Name):
            if node.id not in namespace_keys:
                raise CriterionEvalError(
                    f"criterion references unknown identifier "
                    f"{node.id!r}. Available in namespace: "
                    f"{sorted(namespace_keys)}"
                )


def _evaluate_success_criterion(
    criterion: str, namespace: dict[str, Any],
) -> bool:
    """Ast-parse + validate + eval the criterion against the namespace.

    Returns a bool. Raises `CriterionEvalError` on any parse / safety /
    runtime failure. `namespace` is a flat dict: keys are the identifier
    names the criterion can reference; values are whatever the criterion
    binds (scalars, numpy arrays, dicts for subscript access).

    The eval is run with `__builtins__={}`, so Python's built-in
    `__import__`, `open`, `eval`, etc. are unreachable. Names not in
    `namespace` raise NameError rather than resolving to a builtin.
    """
    try:
        tree = ast.parse(criterion, mode="eval")
    except SyntaxError as e:
        raise CriterionEvalError(
            f"criterion is not a valid Python expression: {e}"
        ) from e

    _validate_criterion_ast(tree, namespace_keys=set(namespace.keys()))

    try:
        compiled = compile(tree, filename="<criterion>", mode="eval")
        # Safety note: we DON'T zero out __builtins__ because numpy's
        # ndarray methods (.mean, .max, .std) internally call
        # `__import__` to resolve dependencies, and an empty
        # __builtins__ dict turns that into a KeyError. The primary
        # safety layer is the AST walker above — it already rejects
        # ast.Call to names outside the namespace, ast.Attribute
        # outside SAFE_ATTRIBUTE_METHODS, ast.Lambda, ast.FunctionDef,
        # etc. Letting eval see real __builtins__ only lets code
        # ALREADY validated-as-safe run; anything dangerous would
        # have been caught at parse time.
        eval_globals = {"__builtins__": __builtins__}
        result = eval(compiled, eval_globals, namespace)
    except AttributeError as e:  # noqa: BLE001 — re-raised
        # §Ship 21c: torch idioms (.float(), .long(), .cpu(), etc.)
        # crash here on numpy arrays. Decompose-time validator catches
        # most; this surfaces a friendlier message for any that escape
        # (legacy mission.json from before the validator landed, or
        # torch idioms inside a lambda the validator hasn't probed).
        msg = str(e)
        torch_methods = (
            "float", "long", "double", "bool", "byte", "short", "half",
            "cpu", "cuda", "to", "detach", "numpy", "requires_grad",
        )
        for method in torch_methods:
            if f"'{method}'" in msg or f" {method}" in msg:
                raise CriterionEvalError(
                    f"criterion uses torch tensor method `.{method}()` "
                    f"but the namespace is numpy. Replace with "
                    f"`.astype(float)` (for `.float()`) or drop the "
                    f"cast — a bool array's `.mean()` already returns "
                    f"the fraction-True. Original: {type(e).__name__}: {e}"
                ) from e
        raise CriterionEvalError(
            f"criterion raised at eval time: {type(e).__name__}: {e}"
        ) from e
    except KeyError as e:  # noqa: BLE001 — re-raised as our type
        # The criterion subscripted a namespace dict with a key the reward
        # / adapter did not produce. Distinct from a broken criterion: the
        # quantity is just absent → "not satisfied", and recoverable via
        # re-decomposition. Surface the missing key + what WAS available so
        # the re-decomposer (and the user) can pick a real key.
        missing = e.args[0] if e.args else str(e)
        available: dict[str, Any] = {}
        for _name in ("behavior", "components", "info", "trajectory"):
            _val = namespace.get(_name)
            if isinstance(_val, dict):
                available[_name] = sorted(_val.keys())
        avail_str = (
            "; ".join(f"{k}={v}" for k, v in available.items())
            or "(no dict-valued namespace entries loaded)"
        )
        raise CriterionMissingKeyError(
            f"criterion references key {missing!r}, which the reward/adapter "
            f"did not produce this iter. Available keys — {avail_str}. "
            f"Reference only keys your reward_seed_prompt actually defines, "
            f"or use `<dict>.get({missing!r}, <default>)` for a soft check."
        ) from e
    except Exception as e:  # noqa: BLE001 — re-raised as our type
        raise CriterionEvalError(
            f"criterion raised at eval time: {type(e).__name__}: {e}"
        ) from e

    # Coerce numpy scalars / 0-d arrays to Python bool.
    try:
        return bool(result)
    except ValueError as e:
        # numpy raises ValueError on multi-element array bool coercion:
        # "The truth value of an array with more than one element is
        # ambiguous." Surface a hint rather than the raw numpy text so
        # Claude / the user sees the common fix.
        if "ambiguous" in str(e).lower() or "more than one" in str(e).lower():
            raise CriterionEvalError(
                f"criterion result is a multi-element array "
                f"({getattr(result, 'shape', '?')}); wrap with .all(), "
                f".any(), .mean(), or compare to a scalar to reduce "
                f"to a single boolean. Original: {e}"
            ) from e
        raise CriterionEvalError(
            f"criterion result is not boolean-coercible: "
            f"{type(result).__name__} {result!r}. Wrap in a comparison."
        ) from e
    except Exception as e:  # noqa: BLE001
        raise CriterionEvalError(
            f"criterion result is not boolean-coercible: "
            f"{type(result).__name__} {result!r}. Wrap in a comparison."
        ) from e


# ── Namespace construction from iter artifacts ───────────────────────
def _load_behavior_json(iter_dir: Path) -> dict[str, Any]:
    """Find behavior.json in either rollout/ or the iter_dir root.

    `sculpt._load_iter_artifacts` (line 249-251) already uses this
    fallback order — mirror it here so we pick up the same file.
    """
    candidates = (
        iter_dir / "rollout" / "behavior.json",
        iter_dir / "behavior.json",
    )
    for p in candidates:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                raise CriterionEvalError(
                    f"behavior.json at {p} is not valid JSON: {e}"
                ) from e
    raise CriterionEvalError(
        f"no behavior.json under {iter_dir} — rollout may not have run, "
        f"or the adapter didn't produce the expected artifact shape."
    )


def _load_trajectory_arrays(
    iter_dir: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Load trajectory.npz and split its keys into (trajectory, components).

    Returns:
      trajectory: dict of `<persisted_key>` → numpy array (shape varies).
                  Only keys in PERSISTED_TRAJECTORY_KEYS are included.
      components: dict of `<component_name>` → mean of
                  `reward_term__<component_name>` array.

    Missing trajectory.npz is NOT fatal — some adapters (Cartpole in
    particular) produce behavior.json only. Callers that need per-step
    data will get a NameError from the evaluator if they try to access
    `info[...]` / `trajectory[...]` when the file was absent.
    """
    traj_path = iter_dir / "rollout" / "trajectory.npz"
    if not traj_path.is_file():
        traj_path = iter_dir / "trajectory.npz"
    if not traj_path.is_file():
        return {}, {}

    try:
        import numpy as np
        with np.load(traj_path, allow_pickle=False) as npz:
            trajectory: dict[str, Any] = {}
            components: dict[str, float] = {}
            for key in npz.files:
                if key.startswith("reward_term__"):
                    name = key[len("reward_term__"):]
                    try:
                        components[name] = float(np.asarray(npz[key]).mean())
                    except Exception:  # noqa: BLE001
                        pass
                elif key in PERSISTED_TRAJECTORY_KEYS:
                    trajectory[key] = np.asarray(npz[key])
                # other keys silently ignored — future adapter additions
                # won't fail here; they just won't be exposed to criteria.
    except Exception as e:  # noqa: BLE001
        raise CriterionEvalError(
            f"failed to read trajectory.npz at {traj_path}: "
            f"{type(e).__name__}: {e}"
        ) from e
    return trajectory, components


def _load_sculptor_components(iter_dir: Path) -> dict[str, float]:
    """Per-component MEANS of the SCULPTOR reward's components — the terms a
    stage's `reward_seed_prompt` introduces, which is exactly what a
    criterion's `components[<name>]` is documented to reference.

    These live in the training-side `<iter_dir>/reward_trajectory.json`
    (Eureka Appendix-F format: `{component: [values]}`, the same file
    `diagnose._load_training_feedback` reads). This is DISTINCT from the
    ENVIRONMENT's intrinsic reward terms, which the rollout persists as
    `reward_term__*` in trajectory.npz. On mjlab the sculptor reward is
    layered on top of the task's own terms, so the two sets differ — and the
    custom components (e.g. `hip_sway_osc`) appear ONLY here, never in
    `reward_term__*`. Without merging these in, `components['hip_sway_osc']`
    KeyErrors at criterion-eval time even though the reward computed it
    (the bug behind the floss/kicking mission halts).

    The rollout-side `rollout/reward_trajectory.json` is deliberately NOT
    read here — on mjlab it holds the ENV terms (already exposed via
    `reward_term__*`), not the sculptor components.

    `__`-prefixed aux keys (__episode_length, __terminated, __time_outs) are
    skipped. Returns {} when the file is absent/unparseable, so callers fall
    back to whatever `reward_term__*` provided."""
    path = iter_dir / "reward_trajectory.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for name, raw in payload.items():
        if name.startswith("__"):
            continue
        try:
            vals = [float(v) for v in raw] if isinstance(raw, list) else [float(raw)]
        except (TypeError, ValueError):
            continue
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


def _build_criterion_namespace(
    iter_dir: Path,
    primary_metric: Optional[float],
) -> dict[str, Any]:
    """Package the iter's on-disk artifacts into the flat namespace
    `_evaluate_success_criterion` consumes.

    `info` is a deliberate alias for `trajectory` (see module docstring
    for rationale). The two names bind the SAME dict — updating one
    reflects in the other.
    """
    behavior = _load_behavior_json(iter_dir)
    trajectory, components = _load_trajectory_arrays(iter_dir)
    # `components[<name>]` is documented as the SCULPTOR reward's components
    # (the terms the reward_seed_prompt introduces). On mjlab those land in
    # reward_trajectory.json, NOT trajectory.npz's `reward_term__*` (which are
    # the ENVIRONMENT's intrinsic terms). Merge them in with precedence so a
    # criterion like `components['hip_sway_osc']` resolves instead of
    # KeyError-ing. (On gym_sb3 the two coincide; the merge is idempotent.)
    components = {**components, **_load_sculptor_components(iter_dir)}

    namespace: dict[str, Any] = {
        "metric": primary_metric,
        "behavior": behavior,
        "components": components,
        "trajectory": trajectory,
        "info": trajectory,  # alias, Ship-14 prompt compatibility
    }
    # Math helpers — safe, built from Python's standard callables.
    namespace.update({
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "float": float, "int": int, "bool": bool,
    })
    return namespace
