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
     On failure: one retry with the validation errors appended to the prompt.
     Second failure raises `EditValidationError`.

  4. Write to `<rewards_dir>/<new_iter_id>.py` (e.g. `…/v1.py`) and rewrite
     `<rewards_dir>/current.py` to load-by-path the new file.

Returns the path to the written v<n>.py.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sculptor.diagnose import Diagnosis, ProposedEdit
from sculptor.kg.query import cite
from sculptor.kg.schema import make_paper_id
from sculptor.kg.store import SculptorKG


MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 16000
RETRY_REMINDER_PREFIX = (
    "Your previous response failed validation. Fix the following and return "
    "ONLY the complete new reward.py source as plain Python — no markdown "
    "fences, no commentary."
)


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


def _build_dummy_inputs(contract) -> tuple[Any, Any, Any, dict]:
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
        info = {k: torch.zeros((1,), dtype=torch.float32) for k in info_keys}
        return state, action, next_state, info
    # Gym-style path unchanged (gym_sb3 uses numpy throughout).
    state = _dummy_from_space(contract.observation_space_spec)
    next_state = _dummy_from_space(contract.observation_space_spec)
    action = _dummy_from_space(contract.action_space_spec)
    info: dict[str, float] = {k: 0.0 for k in (contract.expected_info_keys or [])}
    return state, action, next_state, info


def _call_compute_reward(mod, contract) -> tuple[float, dict]:
    s, a, ns, info = _build_dummy_inputs(contract)
    out = mod.compute_reward(s, a, ns, info)
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


def _current_reward_component_keys(current_module, contract) -> set[str]:
    _, components = _call_compute_reward(current_module, contract)
    return set(components.keys())


def _current_reward_hparam_keys(current_module) -> set[str]:
    spec = getattr(current_module, "REWARD_SPEC", {}) or {}
    hparams = spec.get("hyperparameters", {}) or {}
    return set(hparams.keys())


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

    return EditPlan(
        applicable_edits=applicable,
        deferred_edits=deferred,
        rejected_edits=rejected,
        rejection_reasons=rejection_reasons,
        cited_arxiv_ids=all_refs,
        citation_by_arxiv_id=_citation_map(all_refs, kg_store),
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
        training_device = getattr(contract, "training_device", "any")
        schema_serialized = {k: list(v) for k, v in schema.items()}
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
            f"  info:   dict[str, Tensor of shape (N,)] with keys "
            f"{list(contract.expected_info_keys)}\n\n"
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

    return (
        f"# NEW_VERSION\n{new_version}\n\n"
        f"# PARENT_VERSION\n{current_version}\n\n"
        f"# PARENT_HASH\n{parent_hash}\n\n"
        f"# BEHAVIOR_GOAL\n{diagnosis.behavior_goal or '(not supplied)'}\n\n"
        f"# DIAGNOSIS\n"
        f"failure_modes: {diagnosis.failure_modes}\n"
        f"evidence: {diagnosis.evidence}\n"
        f"confidence: {diagnosis.confidence:.2f}\n\n"
        f"{feedback_block}"
        f"# REWARD_CONTRACT\n"
        f"observation_space: {contract.observation_space_spec}\n"
        f"action_space:      {contract.action_space_spec}\n"
        f"expected_info_keys: {list(contract.expected_info_keys)}\n"
        f"EXPECTED_COMPONENTS: {expected_components}\n"
        f"supports_batched:   {supports_batched}\n"
        f"training_device:    {getattr(contract, 'training_device', 'any')}\n\n"
        f"{batched_block}"
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
) -> str:
    if on_event is not None:
        on_event({
            "type": "log_line",
            "text": f"[edit] LLM request start (attempt {attempt}, "
                    f"user_prompt_chars={len(user_content)})",
        })
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
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
    if not chunks:
        raise EditValidationError("LLM returned no text blocks")
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
                   write_to: Path) -> Any:
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
    """
    write_to.parent.mkdir(parents=True, exist_ok=True)
    # Staging filename MUST end in `.py` so `importlib.util.spec_from_
    # file_location` auto-detects a Python loader. Leading dot marks it
    # as a hidden file so `list_versions` glob `v*.py` ignores it.
    # e.g. `v1.py` → `.v1.staging.py`.
    staging = write_to.with_name(f".{write_to.stem}.staging.py")
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
    except Exception:
        # Any validation failure — unlink the staging file so the old
        # v<n>.py (if any) stays authoritative. Preserves the invariant
        # "rewards/v<n>.py on disk == validated Claude output".
        staging.unlink(missing_ok=True)
        raise

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
) -> Path:
    """Produce a new reward module from `diagnosis` applied to
    `current_reward_path`. Writes `<rewards_dir>/<new_iter_id>.py` and
    rewrites `<rewards_dir>/current.py` to load that file by path.

    `on_event`: optional callable taking a dict. Called at load-bearing
    transitions (pre_validate start/done, LLM request start/response,
    post_validate done, committed). When None (default), no events
    fire — preserves existing sculpt-run call sites unchanged.
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
        current_version = _current_reward_version(current_module)
        current_references = _current_reward_references(current_module)
        parent_hash = hashlib.sha256(
            current_source.encode("utf-8")).hexdigest()[:16]

        # Pre-flight.
        if on_event is not None:
            on_event({"type": "log_line", "text": "[edit] pre-validate start"})
        plan = _pre_validate(
            diagnosis=diagnosis, contract=reward_contract,
            current_module=current_module, kg_store=kg_store)
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
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=2, timeout=240.0)

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
        )
        if on_event is not None:
            on_event({
                "type": "log_line",
                "text": f"[edit] prompt built (chars={len(user_prompt)})",
            })

        # Attempt 1.
        try:
            new_source = _call_llm(
                client, _EDIT_SYSTEM, user_prompt,
                on_event=on_event, attempt=1,
            )
            if on_event is not None:
                on_event({"type": "log_line", "text": "[edit] post-validate (attempt 1)"})
            _post_validate(
                new_source, contract=reward_contract, kg_store=kg_store,
                parent_hash=parent_hash, new_version=new_iter_id,
                write_to=target_path,
            )
        except EditValidationError as first_err:
            print(f"[edit] first attempt failed: {first_err}. Retrying once.",
                  file=sys.stderr, flush=True)
            if on_event is not None:
                on_event({
                    "type": "log_line",
                    "text": f"[edit] attempt 1 rejected: {first_err}; retrying",
                })
            # Attempt 2.
            retry_user = (
                user_prompt
                + "\n\n# RETRY\n"
                + RETRY_REMINDER_PREFIX
                + "\n\n## VALIDATION_ERRORS_ON_PREVIOUS_ATTEMPT\n"
                + str(first_err)
            )
            new_source = _call_llm(
                client, _EDIT_SYSTEM, retry_user,
                on_event=on_event, attempt=2,
            )
            if on_event is not None:
                on_event({"type": "log_line", "text": "[edit] post-validate (attempt 2)"})
            _post_validate(
                new_source, contract=reward_contract, kg_store=kg_store,
                parent_hash=parent_hash, new_version=new_iter_id,
                write_to=target_path,
            )

        _write_current_reexport(rewards_dir, target_path)
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
    )


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
    current.write_text(src, encoding="utf-8")
