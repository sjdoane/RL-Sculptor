"""sculptor/edit.py — LLM reward-module rewriter.

apply_edits() turns a Diagnosis into a new reward module:

  1. PRE-flight validation (before any LLM call):
       - Every `paper_refs` arxiv_id must exist in the KG.
       - Every `target_term` and every identifier referenced in
         `suggested_value` must be either in `reward_contract.expected_info_keys`
         or in the current reward module's component / hyperparameter keys,
         OR in the small allowlist of math/helper names.
         Edits flagged `requires_env_extension=true` are recorded for the
         adapter author and skipped.
       Hard errors raise `EditValidationError` before we spend any API credits.

  2. LLM call (`claude-opus-4-7`, adaptive thinking). Model receives the
     current source + the filtered diagnosis + the reward contract + the
     citation text for every cited paper_ref, and must return the full new
     Python source of the reward module. No AST manipulation on our side.

  3. POST-flight validation:
       - Import the generated file.
       - compute_reward and REWARD_SPEC exist.
       - compute_reward(dummy_state, dummy_action, dummy_next_state, dummy_info)
         returns (numeric, dict[str, numeric]).
       - If `reward_contract.expected_components` is not None, the returned
         components dict's keys must be a subset.
       - REWARD_SPEC.references is a list of dicts with arxiv_id / citation /
         how_used, and every arxiv_id exists in the KG.
       - REWARD_SPEC.parent_hash is present and non-empty.
     On failure: retry with the validation errors appended to the prompt, up
     to `RS_EDIT_REPAIR_RETRIES` extra attempts (env var; default 1, i.e. 2
     attempts total — unset reproduces the original single-retry behavior
     exactly). If every attempt still fails post-flight validation, the
     LAST attempt's `EditValidationError` is raised. Callers (sculpt.py's
     stage-v1 materialization and per-iteration edit loop) already catch
     `EditValidationError` at their boundary and degrade to a clean
     stage/iteration failure rather than crashing the process — see
     `sculpt.py::_run_one_stage` (v1_materialization_errored) and
     `sculpt.py`'s per-iteration `apply_edits` call (edit skipped, iter
     proceeds unmodified).

  4. Write to `<rewards_dir>/<new_iter_id>.py` (e.g. `…/v1.py`) and rewrite
     `<rewards_dir>/current.py` to load-by-path the new file.

Returns the path to the written v<n>.py.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.eval import partition_gate
from sculptor.kg.query import cite
from sculptor.kg.schema import make_paper_id
from sculptor.kg.store import SculptorKG
from sculptor.llm import log_llm_call, model_for


MODEL_ID = model_for("edit")
MAX_TOKENS = 16000

#: Rough bytes-per-token for Python source. Used only to size the output
#: ceiling, so it is deliberately conservative (real Python runs ~3.5-4).
_BYTES_PER_TOKEN = 3.5

#: A whole-module rewrite must be able to emit the WHOLE module plus its
#: changes. `MAX_TOKENS` is a fine ceiling for a 500-line reward and a hard
#: wall for a generated one: a per-mode reward over a 3-mode automaton is
#: ~1000 lines / ~12.4k tokens, of which ~2.5k is inlined reference tables
#: (TARGET_JOINT_POS alone is 8.3 KB) the editor has to restate verbatim. The
#: first sculpt run over one died with `response was cut off at the 16000-token
#: ceiling`, so the loop could train the reward but never evolve it.
_REWRITE_HEADROOM = 1.6


#: Per-request HTTP ceiling at `MAX_TOKENS`, and the seconds-per-token it
#: implies. 240s for 16000 tokens is the calibrated pair (see the client
#: construction in `apply_edits`); anything above that ceiling gets
#: proportionally longer, floored at the original 240s so every existing call
#: site keeps its tuned budget exactly.
BASE_HTTP_TIMEOUT_S = 240.0


def _rewrite_http_timeout_s(max_tokens: int | None) -> float:
    """HTTP timeout that matches the output ceiling it has to carry."""
    limit = int(max_tokens or MAX_TOKENS)
    return max(BASE_HTTP_TIMEOUT_S, BASE_HTTP_TIMEOUT_S * limit / MAX_TOKENS)


def _rewrite_token_ceiling(source: str) -> int:
    """Output ceiling for rewriting `source` in full.

    Scales with the module so large generated rewards stay editable, and never
    returns less than `MAX_TOKENS` — small rewards keep their existing budget
    byte-for-byte.
    """
    needed = int(len(source) / _BYTES_PER_TOKEN * _REWRITE_HEADROOM)
    return max(MAX_TOKENS, needed)
RETRY_REMINDER_PREFIX = (
    "Your previous response failed validation. Fix the following and return "
    "ONLY the complete new reward.py source as plain Python — no markdown "
    "fences, no commentary."
)

# §RS_EDIT_REPAIR_RETRIES: number of EXTRA repair attempts after attempt 1
# when the LLM's generated reward module fails post-flight validation
# (SyntaxError, missing compute_reward/REWARD_SPEC, etc — see
# `_post_validate`). Default 1 (2 attempts total) reproduces the
# long-standing behavior exactly. Raise via the env var when a
# stage/project is hitting `v1_materialization_errored` /
# `apply_edits skipped` from a persistently flaky prompt and a couple
# more repair rounds are worth the extra API spend; the LAST attempt's
# EditValidationError is always re-raised to the caller if every
# attempt (1 + retries) fails post-flight validation.
_DEFAULT_EDIT_REPAIR_RETRIES = 1


def _edit_repair_retries() -> int:
    """Read `RS_EDIT_REPAIR_RETRIES` (extra attempts beyond attempt 1).
    Unset/invalid/negative → the default of 1. Read live (not cached) so
    tests and callers can override per-process without reloading the
    module."""
    raw = os.environ.get("RS_EDIT_REPAIR_RETRIES")
    if raw is None:
        return _DEFAULT_EDIT_REPAIR_RETRIES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_EDIT_REPAIR_RETRIES
    return n if n >= 0 else _DEFAULT_EDIT_REPAIR_RETRIES


class EditValidationError(Exception):
    """Raised when an edit cannot be applied safely (pre- or post-LLM)."""


# Matches arxiv IDs as they appear inside grounding dict values —
# bare `1234.5678` or `arXiv:1234.5678` (case-insensitive prefix).
# Captures the ID only, no prefix. Used by _post_validate's
# grounding ↔ references cross-check.
_ARXIV_IN_TEXT_RE = re.compile(
    r"(?:arxiv:)?(\d{4}\.\d{4,5})", flags=re.IGNORECASE
)


# ── Small allowlist of identifiers that don't need to be "grounded" ───────
_ALLOWED_MATH = {
    # python
    "abs", "min", "max", "sum", "pow", "float", "int", "bool", "str",
    "True", "False", "None", "len",
    # numpy-ish / common math
    "np", "numpy", "math", "sqrt", "exp", "log", "log2", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "pi", "e", "inf", "nan", "clip", "maximum", "minimum",
    "mean", "std", "var", "sum_", "square", "sign",
    # reward-shaping helpers that appear in the literature
    "tolerance", "sigmoid", "softplus", "huber", "gauss", "gaussian",
    "smooth_min", "smooth_max", "relu", "tanh", "lerp",
    # operators that get parsed as names in some weird formulas
    "and", "or", "not",
}
_SIGNATURE_ARGS = {"state", "action", "next_state", "info"}
_RESERVED_SPEC_KEYS = {
    "version", "description", "author", "parent_hash",
    "hyperparameters", "references",
}


# ── Data containers ──────────────────────────────────────────────────────
@dataclass
class EditPlan:
    """Filtered result of pre-flight validation."""

    applicable_edits: list[ProposedEdit]
    deferred_edits: list[ProposedEdit]   # requires_env_extension=True
    # Individual edits the diagnoser proposed but `_pre_validate` dropped
    # (ungrounded target_term or suggested_value). Parallel lists —
    # `rejected_edits[i]` was rejected for `rejection_reasons[i]`. Kept
    # non-fatal so a partially-grounded batch (e.g., 3 of 5 valid) still
    # drives an iteration forward; only an EMPTY `applicable_edits` is
    # a hard error now. See 2026-04-23 overnight regression.
    rejected_edits: list[ProposedEdit]
    rejection_reasons: list[str]
    cited_arxiv_ids: list[str]
    citation_by_arxiv_id: dict[str, str]
    # §Ship 54-pre (#12 shaping↔metric partition gate). NON-BLOCKING flags: a
    # flagged edit STAYS applicable (unlike rejected_edits) — it touches a
    # held-out metric observable or proposes lowering a completion gate. The
    # flags drive the editor-prompt warning + changelog; the only HARD gate is
    # the post-LLM `gate_threshold_regressions` check. Empty unless an objective
    # metric is steering the run (metric_observables passed) — byte-identical
    # otherwise. `screen` carries the full ScreenResult for the prompt builder.
    flagged_edits: list[ProposedEdit] = field(default_factory=list)
    flag_reasons: list[str] = field(default_factory=list)
    screen: Any = None


# ── Identifier extraction from proposal formulas ─────────────────────────
def _extract_formula_identifiers(formula: str | None) -> set[str]:
    """Return the set of bare-name identifiers in `formula`.

    Uses `ast.parse(mode="eval")` so keyword-argument names (e.g.
    `margin=0.2`) are NOT counted as identifiers. Falls back to a conservative
    regex if the formula can't be parsed (proposal formulas are free-form text).
    """
    if not formula:
        return set()
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        # str constants like info["x_velocity"] — harvest string-literal keys
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # only treat short, identifier-like strings as grounded-field refs
            val = node.value
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", val):
                names.add(val)
    return names


# ── Reward module introspection ───────────────────────────────────────────
def _load_reward_module(path: Path, name_hint: str = "_sculptor_current_reward"):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Unique module name so reloading between pre/post validation is clean.
    mod_name = f"{name_hint}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not spec reward module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _dummy_from_space(space) -> Any:
    """Build a minimal but valid input value from a Gymnasium-style space."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Box):
        shape = tuple(space.shape) if space.shape else ()
        return np.zeros(shape, dtype=np.float32)
    if isinstance(space, gym.spaces.Discrete):
        return 0
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros(space.shape, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.shape, dtype=np.int8)
    # Fallback: treat like a Box of shape (1,)
    return np.zeros((1,), dtype=np.float32)


def _build_dummy_inputs(
    contract, *, info_leading_batch: bool = True,
) -> tuple[Any, Any, Any, dict]:
    """Synthesize dummy inputs matching the reward's expected shape.

    Two contract flavors:
      * gym-style — `observation_space_spec` is a `gym.spaces.Box` (or
        other Space). Pre-flight feeds flat numpy arrays; the reward
        module consumes them directly.
      * schema-style (mjlab + any `supports_batched=True` adapter) —
        `state_schema: dict[str, tuple[int, ...]]`. The reward's
        `compute_reward_batched` does `state[key]`, so a numpy array
        blows up with `IndexError: only integers ... are valid indices`
        at `state["qpos"]`. Pre-flight must hand it a dict of TORCH
        tensors — numpy dicts would propagate through Claude's typical
        scalar-wrapper-that-dispatches-to-batched pattern and then
        `torch.cos(numpy_array)` crashes with `TypeError: cos(): input
        must be Tensor, not numpy.ndarray` (observed live: Test 1
        round 3, 2026-04-22).

    Prefer the schema when present; fall back to the gym path otherwise.
    Leading dim is 1 so scalar wrappers that delegate to the batched
    path (and reshape with `[0]`) still work.
    """
    schema = getattr(contract, "state_schema", None)
    if isinstance(schema, dict) and schema:
        # Torch CPU tensors so downstream `torch.cos`, `.reshape`,
        # `.item()` etc. all work. Training adapters hand batched
        # reward modules CUDA tensors at runtime; pre-flight uses CPU
        # because this runs in the backend process (which may not have
        # an initialized CUDA context).
        import torch
        state = {
            k: torch.zeros((1, *shape), dtype=torch.float32)
            for k, shape in schema.items()
        }
        next_state = {
            k: torch.zeros((1, *shape), dtype=torch.float32)
            for k, shape in schema.items()
        }
        # Action tensor: size from actuator_force if present (mjlab
        # convention), else a 1-vec fallback.
        action_dim = int(schema.get("actuator_force", (1,))[0])
        action = torch.zeros((1, action_dim), dtype=torch.float32)
        # Info as dict of torch scalars — Claude's batched path often
        # does `info["fallen"].to(device)` or arithmetic with it.
        info_keys = list(contract.expected_info_keys or [])
        info_schema = getattr(contract, "info_schema", None) or {}
        info: dict[str, Any] = {}
        for key in info_keys:
            feature_shape = tuple(info_schema.get(key, ()))
            if info_leading_batch:
                shape = (1, *feature_shape)
            else:
                # Legacy scalar rewards consumed a feature vector directly,
                # but still expected scalar fields as one-element tensors.
                shape = feature_shape or (1,)
            info[key] = torch.zeros(shape, dtype=torch.float32)
        return state, action, next_state, info
    # Gym-style path unchanged (gym_sb3 uses numpy throughout).
    state = _dummy_from_space(contract.observation_space_spec)
    next_state = _dummy_from_space(contract.observation_space_spec)
    action = _dummy_from_space(contract.action_space_spec)
    info: dict[str, float] = {k: 0.0 for k in (contract.expected_info_keys or [])}
    return state, action, next_state, info


def _call_compute_reward(
    mod, contract, *, info_leading_batch: bool = True,
) -> tuple[float, dict]:
    s, a, ns, info = _build_dummy_inputs(
        contract, info_leading_batch=info_leading_batch)
    try:
        out = mod.compute_reward(s, a, ns, info)
    except Exception as e:  # noqa: BLE001 — module bug → retryable
        # §Ship 31b follow-up: a crash INSIDE the generated module is a
        # module-quality failure the LLM can fix on retry — it must
        # surface as EditValidationError, not leak raw (observed live:
        # E4 smoke, `float()` on a multi-element tensor inside
        # compute_reward raised ValueError, skipped the retry loop, and
        # halted the whole mission stage).
        raise EditValidationError(
            f"compute_reward crashed on dummy inputs: "
            f"{type(e).__name__}: {e}. Note state/next_state values are "
            f"shape-(1, …) torch tensors for batched contracts — use "
            f"`.item()` only on single-element tensors, index before "
            f"converting, and guard div/log/sqrt on zeros."
        ) from e
    if not isinstance(out, tuple) or len(out) != 2:
        raise EditValidationError(
            f"compute_reward must return (reward, components) tuple; got {type(out).__name__}")
    reward, components = out
    if not isinstance(components, dict):
        raise EditValidationError(
            f"components must be a dict; got {type(components).__name__}")
    if len(components) == 0:
        raise EditValidationError("components dict is empty")
    for k, v in components.items():
        if not isinstance(k, str):
            raise EditValidationError(f"component key must be str; got {type(k).__name__}")
        if not isinstance(v, (int, float, np.floating, np.integer, bool, np.bool_)):
            try:
                float(v)
            except Exception:
                raise EditValidationError(
                    f"component {k!r} is not numeric ({type(v).__name__})")
    try:
        reward_f = float(reward)
    except Exception:
        raise EditValidationError(
            f"reward must be numeric; got {type(reward).__name__}")
    # Guard against runaway / exploding rewards — NaN or ±Inf on the
    # scalar path means the batched path will poison PPO's gradient
    # and crash training a few rsl_rl iters in. Reject at pre-flight
    # so Claude has to fix it before the reward reaches the env.
    import math
    if not math.isfinite(reward_f):
        raise EditValidationError(
            f"reward is non-finite ({reward_f!r}); the compute_reward "
            "scalar path is producing NaN or Inf on zero inputs, which "
            "would make PPO's gradient explode. Likely causes: "
            "unguarded division, log/sqrt on zero, or an unclipped "
            "exp. Bound the offending term."
        )
    for k, v in components.items():
        try:
            cf = float(v)
        except Exception:
            continue
        if not math.isfinite(cf):
            raise EditValidationError(
                f"component {k!r} is non-finite ({cf!r}); same as above "
                "but scoped to this term — bound it before merging.")
    return reward_f, components


def _probe_reward_variance(mod, contract) -> None:
    """§Convergence (RL_SCULPTOR_AUDIT loop 3): offline dead-reward pre-screen.

    Evaluate `compute_reward` over a battery of diverse deterministic
    inputs (state/action/info all filled with each probe value); if the
    TOTAL reward AND every component are constant across every probe,
    the reward is state-independent — the v0-class degenerate (constant
    alive-bonus) that gives PPO zero gradient and burns a full GPU
    iteration training "stand still". Reject so the retry loop
    regenerates with this feedback.

    Deliberately conservative: a reward with even ONE state-sensitive
    term passes; probes that crash are skipped (a reward may
    legitimately guard exotic magnitudes — the zeros probe has already
    validated); fewer than 2 surviving probes → pass (insufficient
    evidence beats a false reject)."""
    # (state, action, next_state, info) fill values per probe. The last
    # two are ASYMMETRIC (state != next_state) so a reward built purely
    # of difference terms (next_z - z, displacement shaping) still shows
    # variance and is not false-rejected.
    fills = (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (-0.5, -0.5, -0.5, -0.5),
        (0.0, 0.5, 1.0, 0.5),
        (0.5, 1.0, 0.0, 1.0),
    )
    totals: list[float] = []
    component_rows: list[dict[str, float]] = []
    for bs, ba, bns, binfo in fills:
        s, a, ns, info = _build_dummy_inputs(contract)

        def _fill(x, b):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return torch.full_like(x, float(b))
            except Exception:  # noqa: BLE001 — torch absent → numpy path
                pass
            if isinstance(x, np.ndarray):
                out = x.copy()
                out.fill(b)
                return out
            if isinstance(x, dict):
                return {k: _fill(v, b) for k, v in x.items()}
            if isinstance(x, (int, float)):
                return float(b)
            return x

        try:
            out = mod.compute_reward(
                _fill(s, bs), _fill(a, ba), _fill(ns, bns), _fill(info, binfo))
            reward, components = out
            row = {}
            for k, v in dict(components).items():
                try:
                    row[str(k)] = float(v)
                except Exception:  # noqa: BLE001 — non-scalar component
                    continue
            totals.append(float(reward))
            component_rows.append(row)
        except Exception:  # noqa: BLE001 — guarded reward → skip this probe
            continue
    if len(totals) < 2:
        return
    if any(not np.isfinite(t) for t in totals):
        return  # finiteness is _call_compute_reward's job, not ours
    tol = 1e-9
    if max(totals) - min(totals) > tol:
        return
    shared = set(component_rows[0])
    for row in component_rows[1:]:
        shared &= set(row)
    for k in shared:
        vals = [row[k] for row in component_rows]
        if max(vals) - min(vals) > tol:
            return
    raise EditValidationError(
        "reward is state-independent: compute_reward returned the "
        f"IDENTICAL total ({totals[0]!r}) and identical per-component "
        f"values across {len(totals)} diverse input probes (varied "
        "state/action/next_state/info fills, including asymmetric "
        "state-vs-next_state). A constant reward gives PPO zero "
        "gradient — the policy will learn to stand still. Every reward "
        "must contain at least one term that responds to the physical "
        "state (height, contacts, joint motion, velocity...) so "
        "improving the behavior changes the reward."
    )


# §RL_SCULPTOR_AUDIT §4.4 (edit quality): reject floor for the replay
# screen. A mean per-step total below this on the archived rollout's
# NON-FALLEN frames means living costs more than terminating — with a
# reachable termination the optimal policy ends episodes ASAP (the
# v6/v8 instant-fall collapses). Slightly negative means (transient
# action costs) are tolerated.
_REPLAY_MEAN_FLOOR = -0.05
# Fewer surviving (finite, non-fallen) frames than this → insufficient
# evidence; pass rather than false-reject (mirrors _probe_reward_variance).
_REPLAY_MIN_FRAMES = 32


def _replay_reward_summary(mod, replay_inputs) -> "dict | None":
    """Replay a reward module over archived rollout inputs (built by
    `adapter.build_reward_replay`). Returns
    `{mean_alive, n_alive, component_means}` or None when the module
    can't be replayed (no batched path / crash / too few frames) —
    callers treat None as "no evidence"."""
    if not replay_inputs or not hasattr(mod, "compute_reward_batched"):
        return None
    try:
        import torch

        state, action, next_state, info = replay_inputs
        with torch.no_grad():
            out = mod.compute_reward_batched(state, action, next_state, info)
        rewards, components = out
        rewards = rewards.reshape(-1).float()
        fallen = info.get("fallen")
        alive = (
            (fallen.reshape(-1) < 0.5)
            if isinstance(fallen, torch.Tensor)
            else torch.ones_like(rewards, dtype=torch.bool)
        )
        alive &= torch.isfinite(rewards)
        n_alive = int(alive.sum().item())
        if n_alive < _REPLAY_MIN_FRAMES:
            return None
        comp_means: dict[str, float] = {}
        if isinstance(components, dict):
            for k, v in components.items():
                try:
                    comp_means[str(k)] = float(
                        v.reshape(-1).float()[alive].mean().item())
                except Exception:  # noqa: BLE001 — non-tensor component
                    continue
        return {
            "mean_alive": float(rewards[alive].mean().item()),
            "n_alive": n_alive,
            "component_means": comp_means,
        }
    except Exception:  # noqa: BLE001 — replay is advisory evidence
        return None


def _screen_reward_on_replay(mod, replay_inputs, parent_summary=None) -> None:
    """§RL_SCULPTOR_AUDIT §4.4 (edit quality): anti-collapse screen.

    Replays the CANDIDATE reward on the archived rollout of the policy
    it will train (the current best behavior). Rejects when the mean
    per-step total over non-fallen frames is meaningfully negative:
    with a reachable episode termination, a net-negative living reward
    makes immediate self-termination the optimum — both diagnoser edits
    in the tuck-jump E2E (v6, v8-class) collapsed this way, burning a
    GPU hour each. The reject message carries per-component means on
    those frames so the retry can rebalance the exact offending terms."""
    if not replay_inputs:
        return
    summary = _replay_reward_summary(mod, replay_inputs)
    if summary is None:
        return
    mean_alive = summary["mean_alive"]
    if mean_alive >= _REPLAY_MEAN_FLOOR:
        return
    comp_s = ", ".join(
        f"{k}={v:+.3f}" for k, v in sorted(
            summary["component_means"].items(), key=lambda kv: kv[1]))
    parent_s = ""
    if parent_summary is not None:
        parent_s = (
            f" The PARENT reward averages {parent_summary['mean_alive']:+.3f} "
            f"on the same frames, so this is a property of your edit, not "
            f"of the rollout."
        )
    raise EditValidationError(
        "reward-collapse screen: replaying this module on the archived "
        "rollout of the CURRENT policy (the behavior your edit must "
        f"refine, not destroy) gives a mean per-step TOTAL of "
        f"{mean_alive:+.3f} across {summary['n_alive']} non-fallen "
        f"frames.{parent_s} A net-negative living reward with a "
        "reachable episode termination teaches the policy to end "
        "episodes as fast as possible (deliberate falling) — this "
        "exact mechanism produced instant-fall policies twice. "
        f"Per-component means on those frames: {comp_s or '(none)'}. "
        "Rebalance so the per-step total stays >= 0 in commonly-visited "
        "non-fallen states: shrink new penalties (aim <= ~0.1/step in "
        "ordinary poses), keep paying for the partial behavior the "
        "policy already achieves, and make an exploit UNPROFITABLE "
        "relative to the intended behavior rather than absolutely "
        "negative."
    )


# §Hack-income regression screen (RESEARCH_GAP_ANALYSIS §4.1; CARD's
# Trajectory Preference Evaluation, arXiv:2410.14660, adapted to archived
# exploits): once the diagnoser has CAUGHT a reward-hacking iteration,
# that iteration's rollout is a standing demonstration of the exploit.
# No future candidate may pay it meaningfully MORE per step than the
# PARENT (edit base) does — a caught hack must become monotonically less
# profitable across edits, never re-opened. Compared parent-vs-candidate
# on the SAME frames, so honest credit that incidentally overlaps the
# exploit (e.g. flight credit on a tumble) passes as long as the edit
# didn't RAISE it; only the delta the edit introduced can reject.
# Tolerance below absorbs float noise + incidental term coupling.
_HACK_INCOME_ABS_TOL = 0.05
_HACK_INCOME_REL_TOL = 0.10


def _screen_hack_income(mod, hack_replays) -> None:
    """Reject a candidate that raises the per-step income of a KNOWN
    (diagnosed reward_hacking) archived exploit above its parent's.

    `hack_replays`: list of `{label, replay_inputs, parent_summary}`
    dicts (built in sculpt.py; parent summaries computed by apply_edits
    with the same `_replay_reward_summary` the candidate is measured
    with). Entries without a parent summary or an unreplayable candidate
    are skipped — no evidence, no reject."""
    for hr in hack_replays or []:
        parent = hr.get("parent_summary")
        if not parent:
            continue
        cand = _replay_reward_summary(mod, hr.get("replay_inputs"))
        if cand is None:
            continue
        p_mean = float(parent["mean_alive"])
        allowed = p_mean + max(_HACK_INCOME_ABS_TOL,
                               _HACK_INCOME_REL_TOL * abs(p_mean))
        if cand["mean_alive"] <= allowed:
            continue
        comp_s = ", ".join(
            f"{k}={v:+.3f}" for k, v in sorted(
                cand["component_means"].items(),
                key=lambda kv: -kv[1])[:6])
        raise EditValidationError(
            f"hack-income screen: {hr.get('label', 'a prior iteration')} "
            "was diagnosed as REWARD HACKING and its rollout is archived "
            "as a known exploit. Replaying your candidate on those exact "
            f"frames pays {cand['mean_alive']:+.3f}/step vs the parent "
            f"reward's {p_mean:+.3f}/step — this edit makes a CAUGHT "
            "exploit MORE profitable, re-opening it. Top-paying "
            f"components on the exploit frames: {comp_s or '(none)'}. "
            "Gate those terms on the requirement the exploit skips "
            "(orientation / foot contact / height band) so the exploit "
            "earns LESS than it did, while keeping the intended behavior "
            "paid the same."
        )


# ── §best-of-K candidate edits (RESEARCH_GAP_ANALYSIS §3.3 / COULD) ──────
# One diagnosis → K candidate rewrites under DIVERSE STRATEGY FRAMINGS,
# screened offline (the full _post_validate stack), ranked on replay
# evidence, and only the winner trains. GPU cost is unchanged (still one
# training per iteration); LLM cost is ×K on the edit call only. This is
# the cheap form of best-of-K selection: the treatment arm's K candidates
# come from ONE grounded diagnosis under different edit strategies, vs
# Eureka's blind resampling. Framing (not model or temperature) carries
# the diversity: the same strongest model explores distinct regions of
# edit space, mirroring metric_gen's best-of-N FRAMING pattern.
_EDIT_FRAMINGS: tuple[str, ...] = (
    # Candidate 1: no suffix — byte-identical to the single-shot prompt.
    "",
    "\n\n# STRATEGY DIRECTIVE (candidate framing)\n"
    "MINIMAL-DIFF: make the SMALLEST coherent change that addresses the "
    "diagnosis. Prefer retuning existing magnitudes, thresholds and "
    "gates over adding terms; add a new term only if the diagnosis "
    "cannot be addressed without one. Keep the component structure "
    "recognizably the parent's.",
    "\n\n# STRATEGY DIRECTIVE (candidate framing)\n"
    "STRUCTURAL: rethink the term structure around the diagnosis. "
    "Consider phase decomposition (e.g. contact/launch/flight/landing "
    "gating), removing dead or fighting terms, and re-staging credit so "
    "each phase pays only when its preconditions hold. Stay within the "
    "same contract and cited techniques; do not relax completion gates.",
    "\n\n# STRATEGY DIRECTIVE (candidate framing)\n"
    "EXPLORATION-FIRST: prioritize making the hard-to-reach states "
    "reachable and worth visiting (shaping toward the diagnosis's "
    "missing behavior) over polishing already-achieved behavior. Keep "
    "already-earned credit intact so the policy does not abandon what "
    "it can do.",
    "\n\n# STRATEGY DIRECTIVE (candidate framing)\n"
    "ROBUSTNESS: assume the policy will try to exploit any unguarded "
    "credit. Audit every term for degenerate maximizers and gate them; "
    "prefer bounded, saturating credit over unbounded linear credit.",
)


#: §best-of-K: monotonic staging-name counter — see the staging-name
#: comment in `_post_validate` for why this must be unique per call.
_STAGING_COUNTER = itertools.count()


def _framing_name(index: int) -> str:
    """Short label for a framing ("default", "MINIMAL-DIFF", …) for
    logs + the candidate report."""
    if index == 0:
        return "default"
    try:
        # Framing shape: "\n\n# STRATEGY DIRECTIVE …\nNAME: directive…"
        return _EDIT_FRAMINGS[index].strip().splitlines()[1].split(":")[0]
    except Exception:  # noqa: BLE001 — label only
        return f"framing_{index}"


def _candidate_hack_margin(mod, replay_inputs, hack_replays) -> "float | None":
    """Offline discrimination score for ranking valid candidates: the
    WORST-CASE gap between what the candidate pays the archived honest
    best rollout and what it pays each archived (diagnosed) exploit,
    per step. Higher = sharper separation of honest behavior from known
    gaming (CARD-TPE-style order preservation, arXiv:2410.14660, on
    this project's own replay evidence). None = no evidence (no replays
    / unreplayable candidate) — callers rank None below any float."""
    if not replay_inputs or not hack_replays:
        return None
    honest = _replay_reward_summary(mod, replay_inputs)
    if honest is None:
        return None
    margins: list[float] = []
    for hr in hack_replays:
        cand = _replay_reward_summary(mod, hr.get("replay_inputs"))
        if cand is None:
            continue
        margins.append(float(honest["mean_alive"]) - float(cand["mean_alive"]))
    return min(margins) if margins else None


def _call_compute_reward_batched(mod, contract) -> None:
    """§Ship 31b: execute the BATCHED path pre-flight (N=2 zero
    tensors, runtime-faithful float info). The scalar probe runs pure
    Python where `1.0 - (x <= 1)` is legal; the same expression on
    tensors crashes ("Subtraction with a bool tensor") — caught live
    in the E4 smoke AFTER training started, burning the stage. The
    training path must be exercised by validation, not first executed
    on a rented GPU."""
    if not bool(getattr(contract, "supports_batched", False)):
        return
    if not hasattr(mod, "compute_reward_batched"):
        return  # presence is enforced elsewhere for batched contracts
    import torch

    schema = dict(contract.state_schema or {})
    n = 2
    state = {k: torch.zeros((n, *shape), dtype=torch.float32)
             for k, shape in schema.items()}
    next_state = {k: torch.zeros((n, *shape), dtype=torch.float32)
                  for k, shape in schema.items()}
    action_dim = int(schema.get("actuator_force", (1,))[0])
    action = torch.zeros((n, action_dim), dtype=torch.float32)
    info_schema = getattr(contract, "info_schema", None) or {}
    info = {
        key: torch.zeros(
            (n, *tuple(info_schema.get(key, ()))), dtype=torch.float32,
        )
        for key in (contract.expected_info_keys or [])
    }
    try:
        out = mod.compute_reward_batched(state, action, next_state, info)
    except Exception as e:  # noqa: BLE001 — surface as validation error
        raise EditValidationError(
            f"compute_reward_batched crashed on zero inputs (N=2): "
            f"{type(e).__name__}: {e}. This is the TRAINING path — fix "
            f"the batched implementation (common causes: arithmetic on "
            f"a bool comparison result — use `(~mask).float()` or "
            f"`mask.logical_not().float()`; shape mismatches; assuming "
            f"a device)."
        ) from e
    if not (isinstance(out, tuple) and len(out) == 2):
        raise EditValidationError(
            "compute_reward_batched must return (rewards, components); "
            f"got {type(out).__name__}")
    rewards, components = out
    if not hasattr(rewards, "shape") or tuple(rewards.shape) != (n,):
        raise EditValidationError(
            f"compute_reward_batched rewards must have shape ({n},); got "
            f"{getattr(rewards, 'shape', type(rewards).__name__)}")
    if not isinstance(components, dict) or not components:
        raise EditValidationError(
            "compute_reward_batched components must be a non-empty dict")
    if not torch.isfinite(rewards).all():
        raise EditValidationError(
            "compute_reward_batched produced non-finite rewards on zero "
            "inputs — bound the offending term (unguarded div/log/exp).")


def _current_reward_component_keys(current_module, contract) -> set[str]:
    try:
        _, components = _call_compute_reward(current_module, contract)
    except EditValidationError as exact_shape_error:
        # Migration-only escape hatch: a parent authored before info_schema
        # existed may understand a vector feature as shape (3,) but not the
        # runtime-faithful single-env batch shape (1, 3). We still need its
        # component names in order to prompt a repair. Newly generated code
        # never gets this fallback: _post_validate below uses the strict
        # exact-shape call plus the N=2 batched probe.
        info_schema = getattr(contract, "info_schema", None) or {}
        if not any(tuple(shape) for shape in info_schema.values()):
            raise
        try:
            _, components = _call_compute_reward(
                current_module, contract, info_leading_batch=False)
        except EditValidationError:
            raise exact_shape_error
    return set(components.keys())


def _current_reward_hparam_keys(current_module) -> set[str]:
    spec = getattr(current_module, "REWARD_SPEC", {}) or {}
    hparams = spec.get("hyperparameters", {}) or {}
    return set(hparams.keys())


def _current_reward_hparams(current_module) -> dict[str, Any]:
    """The parent reward's `REWARD_SPEC.hyperparameters` name→value map (not just
    keys). Source for the §Ship 54-pre partition gate's post-LLM gate-erosion
    check (`partition_gate.gate_threshold_regressions`)."""
    spec = getattr(current_module, "REWARD_SPEC", {}) or {}
    hparams = spec.get("hyperparameters", {}) or {}
    return dict(hparams) if isinstance(hparams, dict) else {}


def _current_reward_version(current_module) -> str:
    spec = getattr(current_module, "REWARD_SPEC", {}) or {}
    return str(spec.get("version", "v0"))


def _current_reward_references(current_module) -> list[dict]:
    spec = getattr(current_module, "REWARD_SPEC", {}) or {}
    refs = spec.get("references", []) or []
    # Normalize to list[dict]; tolerate legacy bare-string arxiv_ids.
    out: list[dict] = []
    for r in refs:
        if isinstance(r, str):
            out.append({"arxiv_id": r, "citation": "", "how_used": ""})
        elif isinstance(r, dict):
            out.append(r)
    return out


_REFERENCE_KERNEL_FUNCTIONS = (
    "_scalar",
    "_phase_index_scalar",
    "_reference_tracking_numpy",
    "_phase_index_batched",
    "_reference_tracking_batched",
    "compute_reward",
    "compute_reward_batched",
)

_REFERENCE_KERNEL_GLOBALS = {
    "_W_JOINT_POS", "_W_JOINT_VEL", "_W_ROOT", "_W_ORIENTATION",
    "_TRACKING_WEIGHT", "_RESIDUAL_MAX", "_ALIVE_BONUS",
}


def _reference_tracking_contract(mod) -> "dict[str, Any] | None":
    spec = getattr(mod, "REWARD_SPEC", {}) or {}
    composition = spec.get("composition") if isinstance(spec, dict) else None
    if (not isinstance(composition, dict)
            or composition.get("type") != "reference_tracking_residual"):
        return None
    return dict(composition)


def _reference_kernel_hash(source: str) -> "str | None":
    """Stable AST hash of immutable targets, kernels, and composition."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    by_name = {
        node.name: node for node in tree.body if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if any(name not in by_name for name in _REFERENCE_KERNEL_FUNCTIONS):
        return None
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node
    immutable_names = sorted(
        name for name in assignments
        if name.startswith("REFERENCE_") or name in _REFERENCE_KERNEL_GLOBALS)
    if (not any(name.startswith("REFERENCE_") for name in immutable_names)
            or not _REFERENCE_KERNEL_GLOBALS.issubset(immutable_names)):
        return None
    payload = "\n".join(
        ast.dump(by_name[name], annotate_fields=True, include_attributes=False)
        for name in _REFERENCE_KERNEL_FUNCTIONS
    )
    payload += "\n" + "\n".join(
        ast.dump(assignments[name], annotate_fields=True, include_attributes=False)
        for name in immutable_names
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_reference_tracking_contract(
    *, mod, source: str, components: dict, parent: dict[str, Any],
    parent_kernel_hash: str,
) -> None:
    """Keep the motion prior immutable while allowing bounded task residuals."""
    child = _reference_tracking_contract(mod)
    if child is None:
        raise EditValidationError(
            "tracking-first contract removed: preserve "
            "REWARD_SPEC.composition.type='reference_tracking_residual'")
    for key in (
        "reference_clip_id", "reference_target_sha256", "phase_mode",
        "phase_duration_s", "root_height_frame",
    ):
        if child.get(key) != parent.get(key):
            raise EditValidationError(
                f"tracking-first contract changed {key}: preserve the attached "
                "reference identity exactly")
    try:
        parent_weight = float(parent["tracking_weight"])
        child_weight = float(child["tracking_weight"])
        residual_max = float(child["residual_max"])
    except (KeyError, TypeError, ValueError) as e:
        raise EditValidationError(
            f"tracking-first composition has invalid numeric fields: {e}") from e
    if abs(child_weight - parent_weight) > 1e-9:
        raise EditValidationError(
            "tracking-first contract changed tracking_weight; the reference "
            "base may not be weakened or amplified by an LLM edit")
    if residual_max < 0.0 or residual_max > 0.35 * child_weight:
        raise EditValidationError(
            f"tracking residual_max={residual_max:g} must be within [0, "
            f"{0.35 * child_weight:g}] (<=35% of tracking_weight)")
    required_components = {
        "reference_tracking", "tracking_joint_pos", "tracking_joint_vel",
        "tracking_root_height", "tracking_orientation", "residual_task",
    }
    missing = sorted(required_components - set(components))
    if missing:
        raise EditValidationError(
            f"tracking-first reward dropped required components: {missing}")
    try:
        residual_probe = float(components["residual_task"])
    except (TypeError, ValueError) as e:
        raise EditValidationError(
            f"residual_task is not scalar/numeric on the scalar path: {e}") from e
    if not (0.0 <= residual_probe <= residual_max + 1e-9):
        raise EditValidationError(
            f"residual_task={residual_probe:g} escapes [0, residual_max="
            f"{residual_max:g}] on validation inputs")

    target_hash = getattr(mod, "REFERENCE_TARGET_SHA256", None)
    if target_hash != parent.get("reference_target_sha256"):
        raise EditValidationError(
            "REFERENCE_TARGET_SHA256 changed or disappeared; preserve the "
            "attached motion targets exactly")
    try:
        targets = {
            "joint_pos": np.round(np.asarray(
                mod.REFERENCE_JOINT_POS, dtype=np.float64), 5).tolist(),
            "joint_vel": np.round(np.asarray(
                mod.REFERENCE_JOINT_VEL, dtype=np.float64), 5).tolist(),
            "root_z": np.round(np.asarray(
                mod.REFERENCE_ROOT_Z, dtype=np.float64), 5).tolist(),
            "gravity": (
                np.round(np.asarray(
                    mod.REFERENCE_GRAVITY, dtype=np.float64), 5).tolist()
                if mod.REFERENCE_GRAVITY is not None else None),
        }
    except (AttributeError, TypeError, ValueError) as e:
        raise EditValidationError(
            f"reference target arrays changed or disappeared: {e}") from e
    actual_hash = hashlib.sha256(json.dumps(
        targets, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if actual_hash != target_hash:
        raise EditValidationError(
            "reference target arrays no longer match REFERENCE_TARGET_SHA256")
    if _reference_kernel_hash(source) != parent_kernel_hash:
        raise EditValidationError(
            "immutable reference tracking kernel changed; restore the parent's "
            "phase clock and _reference_tracking_* functions and edit only the "
            "bounded residual")


# ── KG validation helpers ─────────────────────────────────────────────────
def _missing_paper_ids(arxiv_ids: Iterable[str], kg_store: SculptorKG) -> list[str]:
    missing: list[str] = []
    for aid in arxiv_ids:
        if not aid:
            continue
        if kg_store.get_node(make_paper_id(aid)) is None:
            missing.append(aid)
    return missing


def _citation_map(arxiv_ids: Iterable[str], kg_store: SculptorKG) -> dict[str, str]:
    return {aid: cite(aid, store=kg_store) for aid in arxiv_ids if aid}


# ── PRE-flight validation ─────────────────────────────────────────────────
def _pre_validate(
    diagnosis: Diagnosis,
    contract,
    current_module,
    kg_store: SculptorKG,
    *,
    metric_observables: "frozenset[str] | None" = None,
    current_hparams: "dict[str, Any] | None" = None,
) -> EditPlan:
    """Partition proposed edits into applicable / deferred / rejected.

    Pre-2026-04-23 this raised `EditValidationError` on the first
    ungrounded edit, killing the entire batch. In Sam's overnight run,
    1-3 of 5 edits per iter were ungrounded (diagnoser inventing a new
    `target_term` with a `clip`/`gate` op, or referencing a raw state
    like `qvel`), and the other 2-4 grounded edits got dropped too —
    10 iterations, zero reward updates, v0 → v1 then frozen.

    Now partitions: one bad edit drops itself with a logged reason but
    the valid ones proceed. Only an EMPTY applicable list still raises.
    `paper_refs not in KG` also still raises (separate concern: KG
    hygiene vs. diagnoser formula quality).
    """
    # 0. Split deferred vs candidate.
    candidate: list[ProposedEdit] = []
    deferred: list[ProposedEdit] = []
    for e in diagnosis.proposed_edits:
        if getattr(e, "requires_env_extension", False):
            deferred.append(e)
        else:
            candidate.append(e)

    # 1. All paper_refs across the WHOLE proposal must exist in KG —
    #    KG hygiene is an env-level concern and shouldn't be bypassed
    #    by partitioning.
    all_refs = sorted({
        aid for e in candidate for aid in (e.paper_refs or [])
    })
    missing = _missing_paper_ids(all_refs, kg_store)
    if missing:
        raise EditValidationError(
            "paper_refs not in KG: " + ", ".join(missing) + ". "
            "Ingest them first with `sculpt kg ingest` + "
            "`sculpt kg extract --all`."
        )

    # 2. Grounded-field rule: target_term + formula identifiers must be in
    #    the allowed set.
    info_keys = set(contract.expected_info_keys or [])
    component_keys = _current_reward_component_keys(current_module, contract)
    hparam_keys = _current_reward_hparam_keys(current_module)
    grounded = info_keys | component_keys | hparam_keys
    allowed = _ALLOWED_MATH | _SIGNATURE_ARGS | grounded

    applicable: list[ProposedEdit] = []
    rejected: list[ProposedEdit] = []
    rejection_reasons: list[str] = []
    modify_ops = {"increase", "decrease", "remove", "clip", "gate",
                  "replace", "normalize"}

    for i, e in enumerate(candidate):
        violations: list[str] = []

        # 2a. target_term: for modify-ops (existing term), must be grounded.
        #     For 'add', allow a new snake_case name (not in grounded).
        if e.operation in modify_ops:
            if e.target_term not in grounded:
                violations.append(
                    f"operation={e.operation!r} target_term="
                    f"{e.target_term!r} is not a known hyperparameter, "
                    f"component, or info key. Known: "
                    f"hparams={sorted(hparam_keys)} "
                    f"components={sorted(component_keys)} "
                    f"info_keys={sorted(info_keys)}. "
                    f"To introduce a new term, use operation='add'."
                )
        # (operation='add' accepts any fresh snake_case target_term.)

        # 2b. Formula identifiers must all be in `allowed`.
        identifiers = _extract_formula_identifiers(e.suggested_value)
        ungrounded = sorted(n for n in identifiers if n not in allowed)
        if ungrounded:
            violations.append(
                f"operation={e.operation!r} target_term="
                f"{e.target_term!r} suggested_value references ungrounded "
                f"name(s) {ungrounded}. Allowed info_keys="
                f"{sorted(info_keys)}; allowed components="
                f"{sorted(component_keys)}; allowed hparams="
                f"{sorted(hparam_keys)}. Add the field to the env's info "
                "or flag the edit with requires_env_extension=true in the "
                "diagnosis."
            )

        if violations:
            rejected.append(e)
            rejection_reasons.append(
                f"edit[{i}]: " + "; ".join(violations)
            )
        else:
            applicable.append(e)

    # 3. §Ship 54-pre (#12) shaping↔metric partition screen — NON-BLOCKING.
    #    Only when an objective metric is steering the run (metric_observables
    #    passed). Flags edits that touch a held-out metric observable or propose
    #    lowering a completion gate; they STAY applicable. Byte-identical when
    #    metric_observables is None (the gym_sb3 / blind / prompt-edit paths).
    flagged: list[ProposedEdit] = []
    flag_reasons: list[str] = []
    screen = None
    if metric_observables:
        screen = partition_gate.screen_edits(
            applicable,
            metric_observables=metric_observables,
            current_hparams=current_hparams or {},
        )
        flagged = list(screen.flagged_edits)
        flag_reasons = list(screen.flag_reasons)

    return EditPlan(
        applicable_edits=applicable,
        deferred_edits=deferred,
        rejected_edits=rejected,
        rejection_reasons=rejection_reasons,
        cited_arxiv_ids=all_refs,
        citation_by_arxiv_id=_citation_map(all_refs, kg_store),
        flagged_edits=flagged,
        flag_reasons=flag_reasons,
        screen=screen,
    )


# ── Prompt ────────────────────────────────────────────────────────────────
from sculptor.prompts import load_prompt

_EDIT_SYSTEM = load_prompt("edit_rewriter")


def _build_user_prompt(
    *,
    current_source: str,
    current_version: str,
    current_references: list[dict],
    new_version: str,
    parent_hash: str,
    diagnosis: Diagnosis,
    contract,
    citation_map: dict[str, str],
    applicable_edits: list[ProposedEdit],
    deferred_edits: list[ProposedEdit],
    training_feedback: dict | None = None,
    metric_observables: "frozenset[str] | None" = None,
    screen: Any = None,
    case_context: str = "",
    reference_signature: dict | None = None,
) -> str:
    edits_json = [
        {
            "operation": e.operation,
            "target_term": e.target_term,
            "rationale": e.rationale,
            "suggested_value": e.suggested_value,
            "paper_refs": list(e.paper_refs or []),
        }
        for e in applicable_edits
    ]
    deferred_json = [
        {
            "operation": e.operation,
            "target_term": e.target_term,
            "rationale": e.rationale,
            "paper_refs": list(e.paper_refs or []),
            "note": "requires_env_extension — NOT being applied in this iteration",
        }
        for e in deferred_edits
    ]
    expected_components = (
        contract.expected_components if contract.expected_components is not None
        else "OPEN")

    # Batched-contract block — only present when the adapter declares
    # supports_batched=True (mjlab / Isaac Lab). The LLM must emit BOTH
    # scalar `compute_reward` AND `compute_reward_batched` in this mode.
    # See MJLAB_PIVOT_DESIGN §2.2 for the rationale and the unit test in
    # tests/test_edit_prompt_mjlab.py for the guarantee.
    batched_block = ""
    supports_batched = bool(getattr(contract, "supports_batched", False))
    if supports_batched:
        schema = getattr(contract, "state_schema", None) or {}
        info_schema = getattr(contract, "info_schema", None) or {}
        training_device = getattr(contract, "training_device", "any")
        schema_serialized = {k: list(v) for k, v in schema.items()}
        info_schema_serialized = {
            key: list(info_schema.get(key, ()))
            for key in (contract.expected_info_keys or [])
        }
        batched_block = (
            "# BATCHED_CONTRACT (supports_batched=True)\n"
            "This adapter trains on GPU with parallel environments. Your "
            "reward module MUST emit BOTH of the following module-level "
            "callables:\n"
            "  * compute_reward(state, action, next_state, info) -> "
            "(float, dict[str, float])  # scalar path, required for "
            "validation and the UI probe\n"
            "  * compute_reward_batched(state, action, next_state, info) "
            "-> (torch.Tensor, dict[str, torch.Tensor])  # batched path, "
            "used during training\n\n"
            f"Batched-path argument shapes ({training_device} tensors, "
            "float32, leading dim N = num_envs):\n"
            f"  state / next_state: dict[str, Tensor] with per-key "
            f"feature shapes = {json.dumps(schema_serialized, sort_keys=True)}\n"
            f"  action: Tensor shape (N, action_dim)\n"
            "  info:   dict[str, Tensor] with per-key feature shapes "
            "below (each runtime tensor is (N, *feature_shape); [] means "
            "a scalar per env):\n"
            f"{json.dumps(info_schema_serialized, sort_keys=True)}\n\n"
            "Never reshape a vector-valued info channel to (N,). Reduce "
            "it intentionally (for example, a relative-position or velocity "
            "3-vector usually needs a norm over dim=-1).\n\n"
            "Output shapes:\n"
            "  rewards:     Tensor shape (N,) on the same device as inputs\n"
            "  components:  dict[str, Tensor of shape (N,)]\n\n"
            "REWARD_SPEC['supports_batched'] MUST be True. The batched "
            "path and the scalar path must return the same values when "
            "N=1 (modulo tensor<->float coercion); edit.py post-flight "
            "validates this.\n\n"
        )

    # §7.2: mirror the Eureka reward-reflection block from diagnose.py into
    # the editor prompt so the rewrite step has the SAME per-component
    # trajectory data the diagnoser saw when it proposed the edits. This
    # is what lets the "dead-component rule" (below) be concrete — the
    # LLM can reference specific max-min spans rather than guessing.
    feedback_block = ""
    if training_feedback:
        from sculptor.diagnose import _format_training_feedback

        formatted = _format_training_feedback(training_feedback)
        if formatted:
            feedback_block = (
                "# TRAINING_FEEDBACK\n"
                "# Per-component time-series from the training run that "
                "produced this diagnosis (Eureka Appendix F format, "
                "one value per save_interval window). Apply the "
                "dead-component rule from the system prompt against "
                "these numbers — do NOT silently preserve a flat term.\n"
                f"{formatted}\n\n"
            )

    # §Ship 54-pre (#12): the METRIC_PARTITION block — present ONLY when an
    # objective metric is steering the run. Self-contained (carries its own
    # rules) so the shared system prompt is untouched and the no-metric path is
    # byte-identical (empty string → identical f-string bytes).
    partition_block = ""
    if metric_observables:
        partition_block = partition_gate.build_partition_prompt_block(
            metric_observables,
            screen if screen is not None else partition_gate.ScreenResult(),
        )

    # §2026-07-03 case-memory upgrade: the REWRITER is where "don't repeat
    # the same reward mistake" actually lands — the diagnoser proposes,
    # but the rewriter picks formulas + magnitudes. Same block the
    # diagnoser sees; empty string when no cases match / no KG.
    case_block = f"{case_context}\n\n" if case_context else ""

    # §reference-grounded edit: same "REFERENCE MOTION SIGNATURE" block
    # diagnose.py now shows — the rewriter is where numeric targets/
    # thresholds actually get written into code, so it needs the real
    # competent-motion numbers too, not just the diagnosis's prose.
    # Absent (no reference clip for this stage) → "" → byte-identical.
    reference_block = ""
    if reference_signature:
        from sculptor.reference_context import render_reference_signature_block

        rendered = render_reference_signature_block(reference_signature)
        if rendered:
            reference_block = f"{rendered}\n\n"

    return (
        f"# NEW_VERSION\n{new_version}\n\n"
        f"# PARENT_VERSION\n{current_version}\n\n"
        f"# PARENT_HASH\n{parent_hash}\n\n"
        f"# BEHAVIOR_GOAL\n{diagnosis.behavior_goal or '(not supplied)'}\n\n"
        f"{case_block}"
        f"# DIAGNOSIS\n"
        f"failure_modes: {diagnosis.failure_modes}\n"
        f"evidence: {diagnosis.evidence}\n"
        f"confidence: {diagnosis.confidence:.2f}\n\n"
        f"{feedback_block}"
        f"{reference_block}"
        f"# REWARD_CONTRACT\n"
        f"observation_space: {contract.observation_space_spec}\n"
        f"action_space:      {contract.action_space_spec}\n"
        f"expected_info_keys: {list(contract.expected_info_keys)}\n"
        f"EXPECTED_COMPONENTS: {expected_components}\n"
        f"supports_batched:   {supports_batched}\n"
        f"training_device:    {getattr(contract, 'training_device', 'any')}\n\n"
        f"{batched_block}"
        f"{partition_block}"
        f"# APPLICABLE_EDITS (apply these)\n"
        f"{json.dumps(edits_json, indent=2, sort_keys=True, default=str)}\n\n"
        f"# DEFERRED_EDITS (requires_env_extension=true; DO NOT apply — "
        f"record in REWARD_SPEC.description)\n"
        f"{json.dumps(deferred_json, indent=2, sort_keys=True, default=str)}\n\n"
        f"# CITATIONS (use these verbatim in REWARD_SPEC.references[].citation)\n"
        f"{json.dumps(citation_map, indent=2, sort_keys=True)}\n\n"
        f"# PREVIOUS_REFERENCES (preserve entries whose term still exists)\n"
        f"{json.dumps(current_references, indent=2, sort_keys=True, default=str)}\n\n"
        f"# CURRENT_REWARD_SOURCE\n```python\n{current_source}\n```\n\n"
        f"Emit the new reward module source now."
    )


# ── LLM call + source extraction ──────────────────────────────────────────
def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # strip first line (```python / ```) and trailing ```
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip() + "\n"


def _call_llm(
    client,
    system_prompt: str,
    user_content: str,
    *,
    on_event=None,
    attempt: int = 1,
    max_tokens: int | None = None,
) -> str:
    """One rewrite call. `max_tokens` overrides `MAX_TOKENS` for callers whose
    module is long enough that the default truncates — see `apply_edits`."""
    if on_event is not None:
        on_event({
            "type": "log_line",
            "text": f"[edit] LLM request start (attempt {attempt}, "
                    f"user_prompt_chars={len(user_content)})",
        })
    limit = int(max_tokens or MAX_TOKENS)
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=limit,
        thinking={"type": "adaptive"},
        cache_control={"type": "ephemeral"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    # Collect all text blocks (skip thinking blocks).
    chunks: list[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            chunks.append(block.text)
    log_llm_call(
        "edit", MODEL_ID, system=system_prompt, user=user_content,
        response_text="".join(chunks), usage=getattr(resp, "usage", None),
        meta={"attempt": attempt})
    if not chunks:
        raise EditValidationError("LLM returned no text blocks")
    if getattr(resp, "stop_reason", None) == "max_tokens":
        # Say so, rather than letting the half-written module surface as a
        # baffling `SyntaxError: '(' was never closed`. Raised inside the
        # repair-retry loop's `try`, so the reminder reaches the next attempt.
        raise EditValidationError(
            f"response was cut off at the {limit}-token ceiling — the module "
            f"is incomplete, not wrong. Emit the SAME module more concisely: "
            f"no commentary, short docstrings, and do not restate unchanged "
            f"code you could have left alone.")
    out = _strip_markdown_fence("".join(chunks))
    if on_event is not None:
        on_event({
            "type": "log_line",
            "text": f"[edit] LLM response received "
                    f"(attempt {attempt}, chars={len(out)})",
        })
    return out


# ── POST-flight validation ────────────────────────────────────────────────
def _post_validate(new_source: str, *, contract, kg_store: SculptorKG,
                   parent_hash: str, new_version: str,
                   write_to: Path,
                   parent_hparams: "dict[str, Any] | None" = None,
                   parent_tracking: "dict[str, Any] | None" = None,
                   parent_tracking_kernel_hash: "str | None" = None,
                   metric_observables: "frozenset[str] | None" = None,
                   replay_inputs=None,
                   replay_parent: "dict | None" = None,
                   hack_replays: "list[dict] | None" = None,
                   promote: bool = True) -> Any:
    """Write source, import, validate, return the imported module.

    Raises EditValidationError on any failure (caller decides whether to retry).

    Atomic write discipline: write to a `.pending` staging file first,
    validate it, then atomically rename to `write_to`. On any failure
    the staging file is unlinked. Pre-fix (Test 1 round 3 2026-04-22),
    `_post_validate` wrote directly to `write_to` BEFORE validating,
    so a failed validation left a broken v<n>.py on disk — the UI then
    showed a "reward rewrite failed" toast but subsequent loads read
    the polluted file. `_load_reward_module` imports by path, so the
    staging file is loadable even though its name isn't canonical.

    `promote=False` (§best-of-K): run the FULL validation stack but
    never touch `write_to` — the staging file is unlinked on success
    too, and the imported module is returned for offline ranking. The
    winner is re-validated with `promote=True`, so the on-disk invariant
    ("v<n>.py == validated output") is enforced by the same code path
    either way.
    """
    write_to.parent.mkdir(parents=True, exist_ok=True)
    # Staging filename MUST end in `.py` so `importlib.util.spec_from_
    # file_location` auto-detects a Python loader. Leading dot marks it
    # as a hidden file so `list_versions` glob `v*.py` ignores it.
    # e.g. `v1.py` → `.v1.staging3.py`. The per-invocation counter is
    # load-bearing (§best-of-K): candidates staged to the SAME filename
    # can collide in CPython's mtime+size-keyed bytecode cache — two
    # same-length sources written within mtime granularity execute the
    # FIRST candidate's stale .pyc for the second candidate (observed:
    # both candidates reported the first one's hack margin).
    staging = write_to.with_name(
        f".{write_to.stem}.staging{next(_STAGING_COUNTER)}.py")
    staging.write_text(new_source, encoding="utf-8")
    try:
        mod = _load_reward_module(staging, name_hint="_sculptor_new_reward")
    except SyntaxError as e:
        staging.unlink(missing_ok=True)
        raise EditValidationError(f"SyntaxError in generated module: {e}") from e
    except Exception as e:  # noqa: BLE001
        staging.unlink(missing_ok=True)
        raise EditValidationError(
            f"Import error in generated module: {type(e).__name__}: {e}") from e

    # Everything below validates the imported module. On any raise, the
    # staging file gets cleaned up (bare try/except to also cover
    # unexpected exceptions from _call_compute_reward, etc.).
    try:
        # Required attrs
        if not hasattr(mod, "compute_reward"):
            raise EditValidationError("generated module lacks compute_reward")
        if not hasattr(mod, "REWARD_SPEC"):
            raise EditValidationError("generated module lacks REWARD_SPEC")

        # Call compute_reward with dummies.
        reward, components = _call_compute_reward(mod, contract)
        # §Ship 31b: also execute the BATCHED (training) path — the
        # scalar probe alone let tensor-only crashes reach the GPU.
        _call_compute_reward_batched(mod, contract)
        # §Convergence (RL_SCULPTOR_AUDIT loop 3): dead-reward pre-screen —
        # a state-independent (constant) reward is rejected BEFORE it can
        # burn a GPU iteration training "stand still".
        _probe_reward_variance(mod, contract)
        # §RL_SCULPTOR_AUDIT §4.4 (loop 4b): anti-collapse screen — the
        # candidate must not make the archived current-best behavior
        # net-negative (suicide-by-termination attractor). No-op when the
        # caller supplied no replay inputs.
        _screen_reward_on_replay(mod, replay_inputs, replay_parent)
        # §Hack-income regression screen: a caught exploit must never be
        # made MORE profitable by an edit. No-op when no prior iteration
        # was diagnosed reward_hacking (empty/None list).
        _screen_hack_income(mod, hack_replays)

        # expected_components subset check
        if contract.expected_components is not None:
            expected = set(contract.expected_components)
            actual = set(components.keys())
            stray = sorted(actual - expected)
            if stray:
                raise EditValidationError(
                    f"components dict has keys not in expected_components: {stray}. "
                    f"Expected subset of {sorted(expected)}.")

        # REWARD_SPEC schema
        spec = mod.REWARD_SPEC
        if not isinstance(spec, dict):
            raise EditValidationError("REWARD_SPEC must be a dict")
        missing_keys = _RESERVED_SPEC_KEYS - set(spec.keys())
        if missing_keys:
            raise EditValidationError(
                f"REWARD_SPEC missing required keys: {sorted(missing_keys)}")
        if not spec.get("parent_hash"):
            raise EditValidationError("REWARD_SPEC.parent_hash is empty")
        if spec.get("version") != new_version:
            raise EditValidationError(
                f"REWARD_SPEC.version={spec.get('version')!r} should be "
                f"{new_version!r}")
        if spec.get("parent_hash") != parent_hash:
            raise EditValidationError(
                f"REWARD_SPEC.parent_hash does not match expected "
                f"{parent_hash!r} (got {spec.get('parent_hash')!r})")

        # A stage with an attached reference is tracking-FIRST by construction.
        # The LLM may edit only compute_reward's bounded residual: reference
        # arrays, phase clock, kernels, identity, and relative scale are frozen.
        if parent_tracking is not None:
            if not parent_tracking_kernel_hash:
                raise EditValidationError(
                    "parent tracking reward is missing its immutable kernel")
            _validate_reference_tracking_contract(
                mod=mod,
                source=new_source,
                components=components,
                parent=parent_tracking,
                parent_kernel_hash=parent_tracking_kernel_hash,
            )

        # References shape + every arxiv_id present in KG.
        refs = spec.get("references", None)
        if not isinstance(refs, list):
            raise EditValidationError("REWARD_SPEC.references must be a list")
        ref_ids: list[str] = []
        for i, r in enumerate(refs):
            if not isinstance(r, dict):
                raise EditValidationError(
                    f"REWARD_SPEC.references[{i}] must be a dict, got {type(r).__name__}")
            for req in ("arxiv_id", "citation", "how_used"):
                if req not in r:
                    raise EditValidationError(
                        f"REWARD_SPEC.references[{i}] missing field {req!r}")
            ref_ids.append(str(r["arxiv_id"]))
        missing_refs = _missing_paper_ids(ref_ids, kg_store)
        if missing_refs:
            raise EditValidationError(
                f"REWARD_SPEC.references cite arxiv_ids not in KG: "
                f"{missing_refs}")

        # Grounding ↔ references cross-check.
        grounding = spec.get("grounding") or {}
        if isinstance(grounding, dict):
            grounding_ids: set[str] = set()
            for v in grounding.values():
                if not isinstance(v, str):
                    continue
                for m in _ARXIV_IN_TEXT_RE.finditer(v):
                    grounding_ids.add(m.group(1))
            unreferenced = sorted(grounding_ids - set(ref_ids))
            if unreferenced:
                raise EditValidationError(
                    f"REWARD_SPEC.grounding cites arxiv_id(s) {unreferenced} "
                    f"but they are missing from REWARD_SPEC.references. Add "
                    f"a full reference entry for each, or remove the arxiv_id "
                    f"from grounding (physics first-principles text is OK)."
                )

        # §Ship 54-pre (#12) shaping↔metric partition gate — the ONE HARD gate.
        # Runs ONLY when an objective metric is steering the run; compares the
        # emitted hyperparameters against the parent's. A same-named, positive,
        # numerically-LOWERED completion-gate hparam (the g1-kick-v5 whack-a-mole)
        # is a hard reject → existing retry-once → the iter drops the edit.
        # REMOVED / renamed / ambiguous-gate / sign-ambiguous lowerings are
        # ADVISORY (logged, never raised) so a legitimate refactor can't freeze
        # the loop. Byte-identical when metric_observables is None.
        if metric_observables and parent_hparams is not None:
            new_hparams = spec.get("hyperparameters")
            if isinstance(new_hparams, dict):
                reg = partition_gate.gate_threshold_regressions(
                    parent_hparams, new_hparams)
                for adv in reg.advisory:
                    print(f"[edit] partition-gate advisory: {adv}",
                          file=sys.stderr, flush=True)
                if reg.hard:
                    raise EditValidationError(
                        "metric partition gate: the new reward LOWERS a "
                        "completion/qualification gate the active objective "
                        "metric relies on — " + "; ".join(reg.hard) + ". A lower "
                        "gate lets degenerate sub-motions qualify (this is how "
                        "g1-kick-v5 reward-hacked). Keep or RAISE the gate; "
                        "improve the behavior instead of easing the bar."
                    )
    except Exception:
        # Any validation failure — unlink the staging file so the old
        # v<n>.py (if any) stays authoritative. Preserves the invariant
        # "rewards/v<n>.py on disk == validated Claude output".
        staging.unlink(missing_ok=True)
        raise

    # All checks passed. §best-of-K validation-only mode: discard the
    # staging file, hand the module back for ranking; write_to untouched.
    if not promote:
        staging.unlink(missing_ok=True)
        return mod

    # All checks passed — atomically promote staging to the target name
    # so `current.py` + the rewards list see the new version only after
    # full validation. `os.replace` is atomic on POSIX and Win32.
    import os as _os
    _os.replace(staging, write_to)
    return mod


# ── Public entry ─────────────────────────────────────────────────────────
def apply_edits(
    current_reward_path: Path | str,
    diagnosis: Diagnosis,
    new_iter_id: str,
    reward_contract,
    *,
    kg_store: SculptorKG | None = None,
    client=None,
    on_event=None,
    iter_dir: Path | str | None = None,
    metric_observables: "frozenset[str] | None" = None,
    replay_inputs=None,
    hack_replays: "list[dict] | None" = None,
    n_candidates: int = 1,
    max_tokens: int | None = None,
) -> Path:
    """Produce a new reward module from `diagnosis` applied to
    `current_reward_path`. Writes `<rewards_dir>/<new_iter_id>.py` and
    rewrites `<rewards_dir>/current.py` to load that file by path.

    `n_candidates`: §best-of-K (RESEARCH_GAP_ANALYSIS §3.3). 1 (default)
    = the unchanged single-shot-with-retry path, byte-identical. K>1
    samples K candidates under the `_EDIT_FRAMINGS` strategy directives,
    validates each through the FULL post-flight stack (promote=False),
    ranks the valid ones by `_candidate_hack_margin` (ties / no evidence
    → lowest candidate index, i.e. the unbiased default framing), and
    promotes only the winner. All-invalid falls back to the existing
    one-retry repair on the first candidate's errors. A candidate report
    (framings, verdicts, margins, source hashes) is persisted to
    `<iter_dir>/edit_candidates.json` plus per-candidate sources under
    `<iter_dir>/edit_candidates/` when `iter_dir` is given.

    `replay_inputs`: §RL_SCULPTOR_AUDIT §4.4 (loop 4b). Optional
    `(state, action, next_state, info)` batch reconstructed from the
    archived rollout the candidate must not destroy (built by
    `adapter.build_reward_replay`). When supplied, post-flight replays
    the candidate on it and rejects net-negative-living rewards (the
    suicide-by-termination collapse); None (default) skips the screen.

    `on_event`: optional callable taking a dict. Called at load-bearing
    transitions (pre_validate start/done, LLM request start/response,
    post_validate done, committed). When None (default), no events
    fire — preserves existing sculpt-run call sites unchanged.

    `metric_observables`: §Ship 54-pre (#12). The set of physical observables
    the ACTIVE objective metric scores (e.g. `{"joint_vel", "left_foot_pos_b",
    ...}` for g1_kick). When supplied, the shaping↔metric partition gate fires:
    proposed edits touching a held-out observable / lowering a completion gate
    are FLAGGED into the editor prompt + changelog (non-blocking), and a new
    reward that numerically lowers a completion-gate hyperparameter is REJECTED
    post-write. When None (gym_sb3 / blind / prompt-edit paths), the gate is a
    complete no-op — byte-identical to the prior behavior.
    """
    current_reward_path = Path(current_reward_path).resolve()
    rewards_dir = current_reward_path.parent
    target_path = rewards_dir / f"{new_iter_id}.py"
    if target_path == current_reward_path:
        raise EditValidationError(
            f"new_iter_id {new_iter_id!r} would overwrite the current reward "
            f"file at {current_reward_path}.")

    owns_store = kg_store is None
    kg_store = kg_store or SculptorKG()
    try:
        current_module = _load_reward_module(current_reward_path)
        current_source = current_reward_path.read_text(encoding="utf-8")
        # The model has to emit this whole module back, so the ceiling has to
        # fit it. An explicit caller value still wins.
        max_tokens = max_tokens or _rewrite_token_ceiling(current_source)
        current_version = _current_reward_version(current_module)
        current_references = _current_reward_references(current_module)
        parent_tracking = _reference_tracking_contract(current_module)
        parent_tracking_kernel_hash = (
            _reference_kernel_hash(current_source)
            if parent_tracking is not None else None)
        parent_hash = hashlib.sha256(
            current_source.encode("utf-8")).hexdigest()[:16]
        # §Ship 54-pre (#12): parent hparam VALUES for the post-LLM partition
        # gate. Only consulted when an objective metric is steering the run.
        parent_hparams = _current_reward_hparams(current_module)
        gate_parent_hparams = (
            parent_hparams if metric_observables else None)
        # §RL_SCULPTOR_AUDIT §4.4 (loop 4b): the PARENT's replay summary —
        # baseline for the anti-collapse screen's reject message ("the
        # parent averages +X on the same frames"). None when replay is
        # off or the parent itself can't be replayed.
        replay_parent = (
            _replay_reward_summary(current_module, replay_inputs)
            if replay_inputs else None)
        # §Hack-income regression screen: the PARENT's income on each
        # archived exploit — the baseline the candidate must not exceed.
        # Computed here (not in sculpt.py) so parent and candidate are
        # measured by the exact same replay code path. An unreplayable
        # parent leaves parent_summary None → that entry is skipped.
        if hack_replays:
            hack_replays = [dict(hr) for hr in hack_replays]
            for hr in hack_replays:
                hr["parent_summary"] = _replay_reward_summary(
                    current_module, hr.get("replay_inputs"))

        # Pre-flight.
        if on_event is not None:
            on_event({"type": "log_line", "text": "[edit] pre-validate start"})
        plan = _pre_validate(
            diagnosis=diagnosis, contract=reward_contract,
            current_module=current_module, kg_store=kg_store,
            metric_observables=metric_observables,
            current_hparams=parent_hparams)
        if on_event is not None:
            on_event({
                "type": "log_line",
                "text": (
                    f"[edit] pre-validate done "
                    f"(applicable={len(plan.applicable_edits)}, "
                    f"deferred={len(plan.deferred_edits)}, "
                    f"rejected={len(plan.rejected_edits)})"
                ),
            })
            # Surface each rejection so the UI + changelog show which
            # edits the diagnoser got wrong. Critical for user-facing
            # iteration visibility — Sam's overnight hit this on all
            # 10 iters and the silent all-or-nothing behaviour made it
            # look like the reward function just "wasn't being edited."
            for reason in plan.rejection_reasons:
                on_event({
                    "type": "log_line",
                    "text": f"[edit] rejected: {reason}",
                })
            # Structured event so run_manager can persist these and
            # surface them as per-iter chips / a "rejected edits" tab.
            if plan.rejected_edits:
                on_event({
                    "type": "edits_rejected",
                    "count": len(plan.rejected_edits),
                    "reasons": list(plan.rejection_reasons),
                })
            # §Ship 54-pre (#12): partition flags — NON-BLOCKING (the edit
            # stays applicable). Mirrors the rejection surfacing for UI parity.
            if plan.flag_reasons:
                for reason in plan.flag_reasons:
                    on_event({
                        "type": "log_line",
                        "text": f"[edit] partition flag: {reason}",
                    })
                on_event({
                    "type": "edits_partition_flagged",
                    "count": len(plan.flagged_edits),
                    "reasons": list(plan.flag_reasons),
                })

        # §Ship 54-pre (#12): partition flags ALSO go to stderr (always
        # visible — the sculpt loop path passes no on_event, sculpt.py:1261).
        for reason in plan.flag_reasons:
            print(f"[edit] partition flag: {reason}",
                  file=sys.stderr, flush=True)

        if not plan.applicable_edits:
            raise EditValidationError(
                "no applicable edits — every proposed edit was filtered out "
                f"(deferred={len(plan.deferred_edits)}, "
                f"rejected={len(plan.rejected_edits)}). "
                f"Rejection reasons: {plan.rejection_reasons}"
            )

        # LLM client.
        #
        # timeout=240s: per-request HTTP ceiling. Claude Opus with
        # adaptive thinking + 16K max_tokens on a ~7K-char prompt
        # genuinely takes 180-240s; observed 2026-04-23 at 12:57 a
        # real call completed in 204s. Without this ceiling the SDK's
        # default 600s lets one wedged call eat the whole reward-
        # prompt-job budget. max_retries=2 keeps the SDK's exponential
        # backoff envelope tight enough for a user-facing workflow
        # (was 6 — 10+ min backoff is fine for CLI batch, not for a
        # UI Rewards-tab button).
        #
        # The 240s figure is calibrated against MAX_TOKENS. Raising the
        # ceiling for a large module without raising this just relocates the
        # failure from "response was cut off" to APITimeoutError — measured,
        # not predicted: the first replay of the per-mode edit at a 22,541
        # ceiling died on the 240s wall. Scale them together.
        if client is None:
            import anthropic
            client = anthropic.Anthropic(
                max_retries=2, timeout=_rewrite_http_timeout_s(max_tokens))

        # §7.2: load Eureka-format reward trajectory if present so the
        # rewrite prompt shows the SAME per-component data the diagnoser
        # saw. Prefer explicit `iter_dir` arg over diagnosis.iter_dir for
        # callers that carry the path independently (sculpt_run); fall back
        # to diagnosis.iter_dir which apply_edits() used to ignore.
        training_feedback: dict = {}
        iter_dir_path: Path | None = None
        if iter_dir is not None:
            iter_dir_path = Path(iter_dir)
        elif diagnosis.iter_dir:
            iter_dir_path = Path(diagnosis.iter_dir)
        if iter_dir_path is not None and iter_dir_path.is_dir():
            try:
                from sculptor.diagnose import _load_training_feedback
                training_feedback = _load_training_feedback(iter_dir_path)
            except Exception:  # noqa: BLE001 — never block edit on feedback load
                training_feedback = {}

        # §2026-07-03: recall this system's OWN past outcomes on similar
        # tasks/failures into the rewrite prompt (the diagnoser already
        # sees the same block). Advisory + best-effort: no KG / no model /
        # no matches → empty block, prompt byte-identical.
        case_context = ""
        try:
            from sculptor.kg.cases import _render_case_context, query_cases
            from sculptor.kg.query import DEFAULT_MIN_PROMPT_SIMILARITY

            _case_q = (diagnosis.behavior_goal or "") + " | " + ", ".join(
                diagnosis.failure_modes or [])
            if _case_q.strip(" |"):
                case_context = _render_case_context(query_cases(
                    _case_q, top_k=3, store=kg_store,
                    min_similarity=DEFAULT_MIN_PROMPT_SIMILARITY))
        except Exception as e:  # noqa: BLE001 — case memory is advisory
            print(f"[edit] case-memory query failed ({e}) — skipped.",
                  file=sys.stderr, flush=True)

        # §reference-grounded edit: `<stage_dir>/rewards/vN.py` is
        # `current_reward_path`, so the stage dir is `parents[1]`. Resolved
        # defensively — a layout that doesn't match (no rewards/ parent, or
        # no reference file) just leaves this None and the prompt is
        # byte-identical to before this change. `load_reference_signature`
        # itself never raises, but the `.parents[1]` index lookup can on a
        # pathological short path, hence the broad except here too.
        reference_signature: dict | None = None
        try:
            from sculptor.reference_context import load_reference_signature

            reference_signature = load_reference_signature(
                current_reward_path.parents[1])
        except Exception:  # noqa: BLE001 — advisory context, never blocks an edit
            reference_signature = None

        # Build prompt.
        user_prompt = _build_user_prompt(
            current_source=current_source,
            current_version=current_version,
            current_references=current_references,
            new_version=new_iter_id,
            parent_hash=parent_hash,
            diagnosis=diagnosis,
            contract=reward_contract,
            citation_map=plan.citation_by_arxiv_id,
            applicable_edits=plan.applicable_edits,
            deferred_edits=plan.deferred_edits,
            training_feedback=training_feedback,
            metric_observables=metric_observables,
            screen=plan.screen,
            case_context=case_context,
            reference_signature=reference_signature,
        )
        if on_event is not None:
            on_event({
                "type": "log_line",
                "text": f"[edit] prompt built (chars={len(user_prompt)})",
            })

        # §best-of-K: sample K framed candidates, validate all, promote
        # the best-ranked. Falls through to the single-shot path below
        # when K<=1 (byte-identical) or when a winner was promoted.
        winner_promoted = False
        if n_candidates and n_candidates > 1:
            k = min(int(n_candidates), len(_EDIT_FRAMINGS))
            cand_records: list[dict[str, Any]] = []
            first_error: EditValidationError | None = None
            for ci in range(k):
                framed_prompt = user_prompt + _EDIT_FRAMINGS[ci]
                if on_event is not None:
                    on_event({
                        "type": "log_line",
                        "text": f"[edit] candidate {ci + 1}/{k} "
                                f"({_framing_name(ci)})",
                    })
                rec: dict[str, Any] = {"index": ci, "valid": False,
                                       "hack_margin": None}
                try:
                    cand_source = _call_llm(
                        client, _EDIT_SYSTEM, framed_prompt,
                        on_event=on_event, attempt=1,
                        max_tokens=max_tokens,
                    )
                    rec["source"] = cand_source
                    rec["source_sha256"] = hashlib.sha256(
                        cand_source.encode("utf-8")).hexdigest()[:16]
                    cand_mod = _post_validate(
                        cand_source, contract=reward_contract,
                        kg_store=kg_store, parent_hash=parent_hash,
                        new_version=new_iter_id, write_to=target_path,
                        parent_hparams=gate_parent_hparams,
                        parent_tracking=parent_tracking,
                        parent_tracking_kernel_hash=parent_tracking_kernel_hash,
                        metric_observables=metric_observables,
                        replay_inputs=replay_inputs,
                        replay_parent=replay_parent,
                        hack_replays=hack_replays,
                        promote=False,
                    )
                    rec["valid"] = True
                    rec["hack_margin"] = _candidate_hack_margin(
                        cand_mod, replay_inputs, hack_replays)
                except EditValidationError as e:
                    rec["error"] = str(e)
                    if first_error is None:
                        first_error = e
                cand_records.append(rec)

            valid = [r for r in cand_records if r["valid"]]
            if valid:
                # Rank: evidence beats no-evidence; larger margin beats
                # smaller; ties keep the LOWEST index (default framing).
                winner = max(
                    valid,
                    key=lambda r: (
                        r["hack_margin"] is not None,
                        r["hack_margin"] if r["hack_margin"] is not None
                        else float("-inf"),
                        -r["index"],
                    ),
                )
                new_source = winner["source"]
                _post_validate(
                    new_source, contract=reward_contract, kg_store=kg_store,
                    parent_hash=parent_hash, new_version=new_iter_id,
                    write_to=target_path,
                    parent_hparams=gate_parent_hparams,
                    parent_tracking=parent_tracking,
                    parent_tracking_kernel_hash=parent_tracking_kernel_hash,
                    metric_observables=metric_observables,
                    replay_inputs=replay_inputs,
                    replay_parent=replay_parent,
                    hack_replays=hack_replays,
                )
                winner_promoted = True
                if on_event is not None:
                    on_event({
                        "type": "edit_candidates_ranked",
                        "n": k,
                        "valid": len(valid),
                        "selected": winner["index"],
                        "margins": [r["hack_margin"] for r in cand_records],
                    })
                _write_candidate_report(
                    iter_dir, new_iter_id, cand_records, winner["index"])
            else:
                # Every candidate failed validation — fall through to the
                # single-shot retry below, seeded with the first error
                # (same repair semantics as the K=1 path's attempt 2).
                _write_candidate_report(
                    iter_dir, new_iter_id, cand_records, None)
                if on_event is not None:
                    on_event({
                        "type": "log_line",
                        "text": f"[edit] all {k} candidates rejected — "
                                "falling back to repair retry",
                    })
                assert first_error is not None
                raise_after_retry = first_error
                retry_user = (
                    user_prompt
                    + "\n\n# RETRY\n"
                    + RETRY_REMINDER_PREFIX
                    + "\n\n## VALIDATION_ERRORS_ON_PREVIOUS_ATTEMPT\n"
                    + str(raise_after_retry)
                )
                new_source = _call_llm(
                    client, _EDIT_SYSTEM, retry_user,
                    on_event=on_event, attempt=2,
                    max_tokens=max_tokens,
                )
                _post_validate(
                    new_source, contract=reward_contract, kg_store=kg_store,
                    parent_hash=parent_hash, new_version=new_iter_id,
                    write_to=target_path,
                    parent_hparams=gate_parent_hparams,
                    parent_tracking=parent_tracking,
                    parent_tracking_kernel_hash=parent_tracking_kernel_hash,
                    metric_observables=metric_observables,
                    replay_inputs=replay_inputs,
                    replay_parent=replay_parent,
                    hack_replays=hack_replays,
                )
                winner_promoted = True

        # Attempt 1 (single-shot path; skipped when best-of-K promoted).
        # §RS_EDIT_REPAIR_RETRIES: bounded repair-retry loop. Attempt 1 is
        # unconditional; on EditValidationError we re-prompt with the
        # validation errors appended, up to `_edit_repair_retries()`
        # additional attempts (default 1, i.e. 2 total attempts — byte-
        # identical to the pre-knob behavior). The LAST attempt's
        # EditValidationError is re-raised to the caller (sculpt.py /
        # apply_prompt_edit callers already catch it and fail the stage
        # cleanly rather than letting it crash the process).
        if not winner_promoted:
            max_retries = _edit_repair_retries()
            last_err: EditValidationError | None = None
            attempt_prompt = user_prompt
            for attempt in range(1, max_retries + 2):
                try:
                    new_source = _call_llm(
                        client, _EDIT_SYSTEM, attempt_prompt,
                        on_event=on_event, attempt=attempt,
                        max_tokens=max_tokens,
                    )
                    if on_event is not None:
                        on_event({
                            "type": "log_line",
                            "text": f"[edit] post-validate (attempt {attempt})",
                        })
                    _post_validate(
                        new_source, contract=reward_contract, kg_store=kg_store,
                        parent_hash=parent_hash, new_version=new_iter_id,
                        write_to=target_path,
                        parent_hparams=gate_parent_hparams,
                        parent_tracking=parent_tracking,
                        parent_tracking_kernel_hash=parent_tracking_kernel_hash,
                        metric_observables=metric_observables,
                        replay_inputs=replay_inputs,
                        replay_parent=replay_parent,
                        hack_replays=hack_replays,
                    )
                    last_err = None
                    break
                except EditValidationError as err:
                    last_err = err
                    if attempt >= max_retries + 1:
                        break
                    print(
                        f"[edit] attempt {attempt} failed: {err}. Retrying "
                        f"({max_retries + 1 - attempt} attempt(s) left).",
                        file=sys.stderr, flush=True)
                    if on_event is not None:
                        on_event({
                            "type": "log_line",
                            "text": (
                                f"[edit] attempt {attempt} rejected: {err}; "
                                "retrying"
                            ),
                        })
                    attempt_prompt = (
                        user_prompt
                        + "\n\n# RETRY\n"
                        + RETRY_REMINDER_PREFIX
                        + "\n\n## VALIDATION_ERRORS_ON_PREVIOUS_ATTEMPT\n"
                        + str(err)
                    )
            if last_err is not None:
                raise last_err

        _write_current_reexport(rewards_dir, target_path)
        # §Ship 54-pre (#12): persist the partition-gate report next to the
        # iter so the sculpt loop (which passes no on_event) can surface it in
        # the changelog. Written ONLY when a metric steers AND there is
        # something to report — byte-identical otherwise.
        if metric_observables and (plan.flag_reasons or plan.screen is not None):
            _write_partition_report(iter_dir, new_iter_id, plan, metric_observables)
        if on_event is not None:
            on_event({
                "type": "log_line",
                "text": f"[edit] committed {new_iter_id}.py",
            })
        return target_path
    finally:
        if owns_store:
            kg_store.close()


def apply_prompt_edit(
    current_reward_path: Path | str,
    user_prompt: str,
    new_iter_id: str,
    reward_contract,
    *,
    kg_store: "SculptorKG | None" = None,
    client=None,
    on_event=None,
    max_tokens: int | None = None,
) -> Path:
    """One-shot reward rewrite from a user's natural-language prompt.

    Skips the full sculpt diagnose step — the user prompt IS the
    steering signal. Synthesizes a minimal `Diagnosis` with a single
    `add`-op `ProposedEdit` whose `rationale` carries the user prompt
    verbatim; the Claude-side edit_rewriter prompt consumes it and
    writes a new `rewards/v<n>.py`.

    The `add` operation is chosen deliberately so `_pre_validate`'s
    grounded-term check doesn't reject the synthetic target_term —
    `add` allows fresh snake_case names.

    Parameters mirror `apply_edits`. Returns the path to the new
    reward file.
    """
    from sculptor.diagnose import Diagnosis, ProposedEdit

    user_prompt = (user_prompt or "").strip()
    if len(user_prompt) < 3:
        raise EditValidationError(
            "prompt must be at least 3 characters"
        )
    if len(user_prompt) > 2000:
        raise EditValidationError(
            f"prompt must be ≤ 2000 chars (got {len(user_prompt)})"
        )

    # Always consult the KG — users editing reward functions through
    # the Rewards-tab prompt have historically bypassed literature
    # grounding entirely (pre-this-pass `literature_context=[]` was a
    # no-op). Semantic-search the KG on the user's prompt, slot the
    # top matches into the synthetic diagnosis so they flow through
    # `apply_edits`'s existing edit_rewriter prompt as literature
    # context. Falls back to empty context when KG is empty / embedding
    # model can't load — the edit still proceeds, just uncited.
    # Similarity threshold for prompt-edit KG matches. Below this floor
    # the match is considered off-topic — prefer physics first-principles
    # grounding over tangential citations. Tuned empirically against
    # Sam's 46-paper seed KG: 0.35 drops "humanoid paper cited for
    # Cartpole" matches while keeping genuinely-relevant Go1 matches for
    # locomotion prompts. Issue G from Test 1 (2026-04-22).
    _MIN_PROMPT_EDIT_SIMILARITY = 0.35

    lit_context = []
    if kg_store is not None:
        if on_event is not None:
            on_event({"type": "log_line", "text": "[prompt-edit] KG query start"})
        try:
            from sculptor.kg.query import query_semantic

            lit_context = list(query_semantic(
                user_prompt, top_k=5, store=kg_store,
                min_similarity=_MIN_PROMPT_EDIT_SIMILARITY,
            ))
            if on_event is not None:
                on_event({
                    "type": "log_line",
                    "text": (
                        f"[prompt-edit] KG query done ({len(lit_context)} "
                        f"matches above sim={_MIN_PROMPT_EDIT_SIMILARITY:.2f})"
                    ),
                })
        except Exception as e:  # noqa: BLE001
            # Embedding model unavailable, KG empty, or any other
            # transient issue — don't block the edit, just skip the
            # grounding pass.
            import logging
            logging.getLogger(__name__).warning(
                "apply_prompt_edit: KG consultation failed: %s: %s",
                type(e).__name__, e,
            )
            if on_event is not None:
                on_event({
                    "type": "log_line",
                    "text": f"[prompt-edit] KG query failed: {type(e).__name__}; proceeding uncited",
                })

    # Extract arxiv_ids from the filtered lit_context and thread them
    # through `paper_refs` so the existing citation_map / CITATIONS
    # prompt-block machinery fires. Pre-fix `paper_refs=[]` → empty
    # CITATIONS block → Claude saw zero KG content → `references: []`
    # in the emitted REWARD_SPEC despite the grounding dict being rich.
    # Issue B from Test 1 (2026-04-22). `source_paper_ids` items are in
    # `make_paper_id()` form (`paper:<arxiv_id>`) — strip the prefix.
    prompt_paper_refs: list[str] = sorted({
        pid[len("paper:"):]
        for match in lit_context
        for pid in (match.source_paper_ids or [])
        if pid.startswith("paper:")
    })

    synthetic = Diagnosis(
        failure_modes=[],
        evidence=(
            "Human-requested reward rewrite (prompt-on-Rewards-tab). "
            "No diagnosis step; the user prompt below is the sole "
            "steering signal — the edit_rewriter prompt gets "
            f"{len(lit_context)} KG match(es) as literature grounding "
            f"({len(prompt_paper_refs)} arxiv_ids threaded into CITATIONS)."
        ),
        proposed_edits=[
            ProposedEdit(
                target_term="human_prompt",
                operation="add",
                rationale=user_prompt,
                suggested_value="",  # empty → no identifier check in 2b
                paper_refs=prompt_paper_refs,
                requires_env_extension=False,
            )
        ],
        literature_context=lit_context,
        confidence=1.0,
        iter_dir=None,
        behavior_goal=user_prompt,
    )
    return apply_edits(
        current_reward_path=current_reward_path,
        diagnosis=synthetic,
        new_iter_id=new_iter_id,
        reward_contract=reward_contract,
        kg_store=kg_store,
        client=client,
        on_event=on_event,
        max_tokens=max_tokens,
    )


def _write_partition_report(
    iter_dir: Path | str | None,
    new_iter_id: str,
    plan: EditPlan,
    metric_observables: "frozenset[str]",
) -> None:
    """§Ship 54-pre (#12): persist the partition-gate report to
    `<iter_dir>/partition_gate.json` so the sculpt loop (no on_event on the
    apply_edits call) can surface it in the changelog. Never raises — a
    reporting failure must not break a committed reward edit."""
    if iter_dir is None:
        return
    try:
        d = Path(iter_dir)
        if not d.is_dir():
            return
        screen = plan.screen
        report = {
            "version": new_iter_id,
            "metric_observables": sorted(metric_observables),
            "flag_reasons": list(plan.flag_reasons),
            "flagged_edit_count": len(plan.flagged_edits),
            "held_out": list(getattr(screen, "held_out", []) or []),
            "gate_hparams": list(getattr(screen, "gate_hparams", []) or []),
        }
        (d / "partition_gate.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — reporting is advisory, never fatal
        print(f"[edit] partition report write skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)


def _write_candidate_report(
    iter_dir: Path | str | None,
    new_iter_id: str,
    cand_records: "list[dict[str, Any]]",
    selected: "int | None",
) -> None:
    """§best-of-K: persist the candidate slate — every framing's verdict,
    margin and source — to `<iter_dir>/edit_candidates.json` + the raw
    candidate sources under `<iter_dir>/edit_candidates/`. This is the
    per-iteration paper trail for "which strategies were considered and
    why this one won" (provenance for the selection decision, not just
    the winning artifact). Never raises."""
    if iter_dir is None:
        return
    try:
        d = Path(iter_dir)
        if not d.is_dir():
            return
        src_dir = d / "edit_candidates"
        src_dir.mkdir(exist_ok=True)
        rows = []
        for rec in cand_records:
            source = rec.get("source")
            if source:
                (src_dir / f"cand{rec['index']}.py").write_text(
                    source, encoding="utf-8")
            rows.append({
                "index": rec["index"],
                "framing": _framing_name(rec["index"]),
                "valid": rec["valid"],
                "hack_margin": rec["hack_margin"],
                "source_sha256": rec.get("source_sha256"),
                "error": rec.get("error"),
            })
        report = {
            "version": new_iter_id,
            "selected": selected,
            "candidates": rows,
        }
        (d / "edit_candidates.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — reporting is advisory, never fatal
        print(f"[edit] candidate report write skipped: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)


def _write_current_reexport(rewards_dir: Path, latest: Path) -> None:
    """Rewrite `<rewards_dir>/current.py` so `compute_reward`,
    `REWARD_SPEC`, and (when present) `compute_reward_batched` point
    at `latest`. Uses file-path import so the re-export works regardless
    of how current.py itself is loaded.

    `compute_reward_batched` is re-exported conditionally because
    `MjlabAdapter` (and any adapter with `RewardContract.supports_batched
    = True`) looks it up as a module-level binding on current.py — a
    bare `compute_reward = _mod.compute_reward` re-export hid the
    batched entry point and caused the mjlab runner to AttributeError
    inside `env.load_managers()`. Scalar-only adapters (gym_sb3) don't
    look for it, so the `hasattr` guard keeps both worlds working.
    """
    current = rewards_dir / "current.py"
    src = f'''"""Auto-generated by sculptor.edit. Re-exports the latest reward.

Latest: {latest.name}
"""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LATEST = _HERE / {latest.name!r}

_spec = importlib.util.spec_from_file_location(
    "sculptor_reward_latest", str(_LATEST))
if _spec is None or _spec.loader is None:
    raise ImportError(f"could not load latest reward from {{_LATEST}}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_reward = _mod.compute_reward
REWARD_SPEC = _mod.REWARD_SPEC

__all__ = ["compute_reward", "REWARD_SPEC"]

# Re-export the batched entry point when the latest reward defines one
# (required by MjlabAdapter + any supports_batched=True adapter).
if hasattr(_mod, "compute_reward_batched"):
    compute_reward_batched = _mod.compute_reward_batched
    __all__.append("compute_reward_batched")
'''
    # ``current.py`` is a mutable convenience pointer (the promoted world
    # selection remains authoritative), but readers still must never observe a
    # truncated module if the process dies during a rewrite.  Match the env
    # pointer's tmp+replace discipline.
    tmp = rewards_dir / "current.py.tmp"
    tmp.write_text(src, encoding="utf-8")
    tmp.replace(current)
