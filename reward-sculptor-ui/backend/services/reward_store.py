"""Reward-version filesystem IO + AST-based validation.

The UI never asks sculptor to execute user-supplied reward code in-
process — we parse the source with `ast`, evaluate the REWARD_SPEC dict
literal via `ast.literal_eval`, and run `compute_reward` on zero-
dummies inside a subprocess. This keeps the backend safe against
arbitrary Python the user (or a future third-party frontend) might
post.

Used by: routes/rewards.py (GET list/detail, PUT manual edit).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.models.reward import (
    ComponentProbe,
    ManualEditRequest,
    RewardRef,
    RewardSpec,
    RewardVersionDetail,
    RewardVersionSummary,
)


class RewardValidationError(Exception):
    """Raised by post-flight validation. Carries a list of violations
    and optional remediation hints."""

    def __init__(self, violations: list[str], suggestions: Optional[list[str]] = None):
        super().__init__("; ".join(violations) or "reward validation failed")
        self.violations = violations
        self.suggestions = suggestions or []


class ConcurrencyError(Exception):
    """Raised by PUT when expected_parent_version != current latest."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected_parent_version={expected} but current latest is v{current}"
        )
        self.expected = expected
        self.current = current


# ── version discovery ────────────────────────────────────────────────
_V_RE = re.compile(r"^v(\d+)\.py$")


def list_versions(rewards_dir: Path) -> list[tuple[int, Path]]:
    """Return [(n, path), ...] sorted ascending for every v<n>.py."""
    if not rewards_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in rewards_dir.iterdir():
        m = _V_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def latest_version(rewards_dir: Path) -> Optional[tuple[int, Path]]:
    vs = list_versions(rewards_dir)
    return vs[-1] if vs else None


# ── AST-safe REWARD_SPEC extraction ──────────────────────────────────
def _extract_reward_spec(source: str) -> tuple[dict[str, Any], Optional[str]]:
    """Return `(spec_dict, error_or_None)`.

    Parses the module AST, finds a module-level `REWARD_SPEC = {...}`
    assignment, and `ast.literal_eval`s the dict. Safe against arbitrary
    code execution; anything that isn't a plain literal dict fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {}, f"SyntaxError: {e}"

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "REWARD_SPEC":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError) as e:
                        return {}, (
                            f"REWARD_SPEC is not a plain dict literal: {e}"
                        )
                    if not isinstance(value, dict):
                        return {}, "REWARD_SPEC must be a dict literal"
                    return value, None
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            if node.target.id == "REWARD_SPEC" and node.value is not None:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as e:
                    return {}, (
                        f"REWARD_SPEC is not a plain dict literal: {e}"
                    )
                if not isinstance(value, dict):
                    return {}, "REWARD_SPEC must be a dict literal"
                return value, None

    return {}, "no module-level REWARD_SPEC assignment found"


def _compute_reward_signature_ok(source: str) -> Optional[str]:
    """Return None if compute_reward(state, action, next_state, info) is
    defined at module scope; an error string otherwise."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            args = [a.arg for a in node.args.args]
            expected = ["state", "action", "next_state", "info"]
            if args != expected:
                return (
                    f"compute_reward must accept exactly {expected}, "
                    f"got {args}"
                )
            return None
    return "no module-level compute_reward function found"


def _spec_from_dict(raw: dict[str, Any]) -> RewardSpec:
    refs_raw = raw.get("references") or []
    refs: list[RewardRef] = []
    for r in refs_raw:
        if isinstance(r, dict) and r.get("arxiv_id"):
            refs.append(
                RewardRef(
                    arxiv_id=str(r["arxiv_id"]),
                    citation=str(r.get("citation") or ""),
                )
            )
    hparams_raw = raw.get("hyperparameters") or {}
    hparams: dict[str, float] = {}
    if isinstance(hparams_raw, dict):
        for k, v in hparams_raw.items():
            try:
                hparams[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    grounding_raw = raw.get("grounding") or {}
    grounding: dict[str, str] = {}
    if isinstance(grounding_raw, dict):
        for k, v in grounding_raw.items():
            # Coerce everything to strings so the UI always gets
            # renderable text (Claude occasionally emits a list or
            # arxiv-id prefix). Skip empty values.
            s = str(v).strip()
            if s:
                grounding[str(k)] = s
    return RewardSpec(
        version=str(raw.get("version") or ""),
        parent_hash=str(raw.get("parent_hash") or ""),
        author=str(raw.get("author") or "unknown"),
        description=str(raw.get("description") or ""),
        hyperparameters=hparams,
        references=refs,
        grounding=grounding,
    )


# ── component probe (subprocess-isolated) ────────────────────────────
# Two-mode probe: scalar (gym_sb3 — state/action/next_state are plain
# floats) or schema-based (mjlab — state/next_state are dicts matching
# the adapter's RewardContract.state_schema). Pre-fix the probe only
# did scalar mode, so a Cartpole v1 that reads state["qpos"] raised
# `AttributeError: 'float' object has no attribute 'items'` inside the
# subprocess and the UI surfaced it as a Probe failure (bug #A, Test 1
# 2026-04-22).
_PROBE_SCRIPT = textwrap.dedent(
    """\
    import importlib.util, json, sys, traceback
    path = sys.argv[1]
    state_schema_json = sys.argv[2] if len(sys.argv) > 2 else ""
    info_keys_json = sys.argv[3] if len(sys.argv) > 3 else ""
    spec = importlib.util.spec_from_file_location('_probe', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if state_schema_json:
            # Schema mode (mjlab / any supports_batched adapter). Claude
            # reward modules nearly always call torch ops on the dict
            # values (torch.cos, .device, .to(), .reshape, .item(), ...)
            # so the probe must hand them real torch tensors. Nested
            # Python lists (pre-fix, Test 1 round 3) crashed with
            # `AttributeError: 'list' object has no attribute 'device'`.
            import torch
            schema = json.loads(state_schema_json)
            def _zeros_tensor(shape):
                # Leading dim 1 for single-env probe. torch.zeros so
                # downstream torch.cos / .device / .to() all work.
                return torch.zeros((1, *shape), dtype=torch.float32)
            state = {k: _zeros_tensor(v) for k, v in schema.items()}
            next_state = {k: _zeros_tensor(v) for k, v in schema.items()}
            action_dim = int(schema.get('actuator_force', [1])[0]) if schema.get('actuator_force') else 1
            action = torch.zeros((1, action_dim), dtype=torch.float32)
            info_keys = json.loads(info_keys_json) if info_keys_json else []
            info = {k: torch.zeros((1,), dtype=torch.float32) for k in info_keys}
        else:
            # Scalar mode (gym_sb3 / unschema'd).
            state = action = next_state = 0.0
            info = {}
        out = mod.compute_reward(state, action, next_state, info)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError(f'compute_reward must return (reward, components); got {type(out).__name__}')
        reward, components = out
        if not isinstance(components, dict):
            raise TypeError(f'components must be a dict; got {type(components).__name__}')
        # For schema mode, reward / component values may be tensors/lists;
        # flatten to a scalar via float() on the first element.
        def _scalar(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
            # Tensor or list → try .item() then first-element coercion.
            item = getattr(v, 'item', None)
            if callable(item):
                try:
                    return float(item())
                except Exception:
                    pass
            try:
                # torch.Tensor / np.array / list[float] — take first.
                return float(list(v)[0] if hasattr(v, '__iter__') else v)
            except Exception:
                return 0.0
        total = _scalar(reward)
        comp_out = {str(k): _scalar(v) for k, v in components.items()}
        print(json.dumps({'ok': True, 'components': comp_out, 'total': total}))
    except Exception as e:
        print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
        sys.exit(0)
    """
)


def probe_components(
    source_path: Path,
    state_schema: Optional[dict[str, tuple[int, ...]]] = None,
    info_keys: Optional[list[str]] = None,
) -> ComponentProbe:
    """Run compute_reward in a fresh Python subprocess to avoid polluting
    the backend's module cache with user-supplied code.

    `state_schema`: when provided (mjlab adapter), the probe builds
    `state` and `next_state` as nested-list dicts matching the schema
    shape with leading-dim 1 — so Cartpole's `state["qpos"][0]` works
    end-to-end. When None (gym_sb3 or adapter unavailable), scalar 0.0
    is used for all three (pre-fix default).
    """
    argv = [sys.executable, "-c", _PROBE_SCRIPT, str(source_path)]
    if state_schema:
        # Serialize with tuple → list (JSON-safe). Keys stay strings.
        schema_json = json.dumps({k: list(v) for k, v in state_schema.items()})
        info_json = json.dumps(list(info_keys or []))
        argv += [schema_json, info_json]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except subprocess.TimeoutExpired:
        return ComponentProbe(ok=False, error="subprocess timeout (15s)")
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 and not stdout:
        stderr = (proc.stderr or "").strip()[-500:]
        return ComponentProbe(ok=False, error=f"subprocess exit {proc.returncode}: {stderr}")
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (ValueError, IndexError) as e:
        return ComponentProbe(ok=False, error=f"probe stdout not JSON: {e}")
    if not payload.get("ok"):
        return ComponentProbe(ok=False, error=str(payload.get("error") or "unknown error"))
    return ComponentProbe(
        ok=True,
        components=payload.get("components") or {},
        total=float(payload.get("total") or 0.0),
    )


# ── read API ──────────────────────────────────────────────────────────
def summarize_version(
    rewards_dir: Path, version: int, source_path: Path, metric_history: list[float | None],
) -> RewardVersionSummary:
    source = source_path.read_text(encoding="utf-8")
    spec_dict, _err = _extract_reward_spec(source)
    spec = _spec_from_dict(spec_dict) if spec_dict else RewardSpec(version=f"v{version}")
    author_raw = (spec_dict.get("author") or "").strip().lower()
    author: str = (
        "human" if author_raw == "human"
        else "sculptor" if author_raw in ("sculptor", "reward-sculptor")
        else "unknown"
    )
    parent_version = _parent_version_for(spec_dict.get("parent_hash"), rewards_dir, version)

    # metric_history indexes correspond to sculpt iterations; each
    # iter_<i> produces v<i+1>. So v<n>.py's metric is at history[n-1].
    metric = None
    delta = None
    if version >= 1 and len(metric_history) >= version:
        curr = metric_history[version - 1]
        prev = metric_history[version - 2] if version >= 2 else None
        if isinstance(curr, (int, float)):
            metric = float(curr)
        if isinstance(curr, (int, float)) and isinstance(prev, (int, float)):
            delta = float(curr) - float(prev)

    created_at = datetime.fromtimestamp(
        source_path.stat().st_mtime, tz=timezone.utc
    )

    return RewardVersionSummary(
        version=version,
        file_name=source_path.name,
        author=author,  # type: ignore[arg-type]
        parent_hash=spec.parent_hash,
        parent_version=parent_version,
        description=spec.description,
        iter_introduced=(version - 1) if author == "sculptor" and version > 0 else None,
        n_references=len(spec.references),
        primary_metric=metric,
        metric_delta=delta,
        created_at=created_at,
    )


def _parent_version_for(
    parent_hash: Any, rewards_dir: Path, this_version: int
) -> Optional[int]:
    if not isinstance(parent_hash, str) or not parent_hash:
        return this_version - 1 if this_version > 0 else None
    # sha256 of parent source[:16] — check each earlier version.
    for n, path in list_versions(rewards_dir):
        if n >= this_version:
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if h == parent_hash:
            return n
    return this_version - 1 if this_version > 0 else None


def load_detail(
    rewards_dir: Path, version: int, metric_history: list[float | None],
) -> Optional[RewardVersionDetail]:
    for n, path in list_versions(rewards_dir):
        if n == version:
            summary = summarize_version(rewards_dir, n, path, metric_history)
            source = path.read_text(encoding="utf-8")
            spec_dict, _ = _extract_reward_spec(source)
            spec = _spec_from_dict(spec_dict) if spec_dict else RewardSpec(
                version=f"v{n}"
            )
            # For adapters with a schema-based state (mjlab → dict of
            # shape tuples), pass the schema to the probe so Claude's
            # v<n> doesn't AttributeError inside the subprocess when it
            # reads `state["qpos"]` on scalar 0.0. Falls back silently
            # to scalar probe if adapter loading fails (gym_sb3 /
            # coming-soon adapters / misconfigured project).
            state_schema, info_keys = _resolve_probe_schema(rewards_dir)
            probe = probe_components(path, state_schema, info_keys)
            return RewardVersionDetail(
                **summary.model_dump(),
                source=source,
                spec=spec,
                components_probe=probe,
            )
    return None


def _resolve_probe_schema(
    rewards_dir: Path,
) -> tuple[Optional[dict[str, tuple[int, ...]]], Optional[list[str]]]:
    """Load the project's adapter and extract `state_schema` +
    `expected_info_keys` for the probe. Returns `(None, None)` on any
    failure — the probe then uses scalar-mode dummies, which matches
    gym_sb3 semantics.

    Kept in reward_store.py rather than the route because `load_detail`
    is also called from `apply_manual_edit`'s post-write probe.
    """
    config_path = rewards_dir.parent / "config.toml"
    if not config_path.is_file():
        return None, None
    try:
        from sculptor.adapters.base import load_adapter

        adapter = load_adapter(config_path)
        contract = adapter.reward_contract()
        schema = getattr(contract, "state_schema", None)
        info_keys = list(getattr(contract, "expected_info_keys", None) or [])
        if schema:
            return dict(schema), info_keys
        return None, info_keys or None
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).debug(
            "_resolve_probe_schema: falling back to scalar probe (%s: %s)",
            type(e).__name__, e,
        )
        return None, None


# ── PUT validation + write ───────────────────────────────────────────
@dataclass
class ManualEditResult:
    new_version: int
    new_path: Path


def apply_manual_edit(
    *,
    rewards_dir: Path,
    body: ManualEditRequest,
    kg_arxiv_ids: set[str],
) -> ManualEditResult:
    """Validate + write a manual-edit reward version.

    Raises:
      FileNotFoundError — no parent version exists at all.
      ConcurrencyError — expected_parent_version != current latest.
      RewardValidationError — AST / signature / parent_hash / references
        check failed, or subprocess probe returned an error.
    """
    latest = latest_version(rewards_dir)
    if latest is None:
        raise FileNotFoundError(f"no v*.py in {rewards_dir}")
    current_n, current_path = latest
    if body.expected_parent_version != current_n:
        raise ConcurrencyError(body.expected_parent_version, current_n)

    violations: list[str] = []
    suggestions: list[str] = []

    # Syntactic checks.
    sig_err = _compute_reward_signature_ok(body.source)
    if sig_err:
        violations.append(sig_err)
        suggestions.append(
            "Keep the `def compute_reward(state, action, next_state, info):` "
            "signature verbatim — it's the hard contract with sculptor."
        )

    spec_dict, spec_err = _extract_reward_spec(body.source)
    if spec_err:
        violations.append(f"REWARD_SPEC: {spec_err}")

    if not violations:
        required = (
            "version", "parent_hash", "author", "description",
            "hyperparameters", "references",
        )
        missing = [k for k in required if k not in spec_dict]
        if missing:
            violations.append(f"REWARD_SPEC is missing keys: {missing}")

        author_raw = (spec_dict.get("author") or "").strip().lower()
        if author_raw != "human":
            violations.append(
                f"REWARD_SPEC.author must be 'human' for manual edits; got {author_raw!r}"
            )
            suggestions.append(
                "Set REWARD_SPEC[\"author\"] = \"human\" — sculptor-written "
                "versions are reserved for LLM-driven iterations."
            )

        expected_parent_hash = hashlib.sha256(
            current_path.read_bytes()
        ).hexdigest()[:16]
        if str(spec_dict.get("parent_hash") or "") != expected_parent_hash:
            violations.append(
                f"REWARD_SPEC.parent_hash must be sha256(v{current_n}.py)[:16] "
                f"= {expected_parent_hash}"
            )

        refs = spec_dict.get("references") or []
        if isinstance(refs, list):
            for r in refs:
                aid = r.get("arxiv_id") if isinstance(r, dict) else None
                if aid and aid not in kg_arxiv_ids:
                    violations.append(
                        f"reference arxiv_id={aid!r} is not in the project KG"
                    )
                    suggestions.append(
                        f"Ingest {aid} via POST /kg/seeds before citing it."
                    )

    if violations:
        raise RewardValidationError(violations, suggestions)

    # Write to staging, probe, then atomically rename.
    new_n = current_n + 1
    staging = rewards_dir / f"_v{new_n}.pending.py"
    staging.write_text(body.source, encoding="utf-8")

    state_schema, info_keys = _resolve_probe_schema(rewards_dir)
    probe = probe_components(staging, state_schema, info_keys)
    if not probe.ok:
        staging.unlink(missing_ok=True)
        raise RewardValidationError(
            [f"compute_reward probe failed: {probe.error}"],
            [
                "Make sure compute_reward runs with zero-valued dummies — "
                "no list indexing on info without a default.",
            ],
        )

    final_path = rewards_dir / f"v{new_n}.py"
    staging.replace(final_path)

    # Rewrite current.py to re-export the new version.
    _write_current_reexport(rewards_dir, final_path)
    return ManualEditResult(new_version=new_n, new_path=final_path)


def _write_current_reexport(rewards_dir: Path, target: Path) -> None:
    """Mirror of sculptor/edit.py's current.py pattern: importlib spec
    by file path so there's no symlink, works cross-platform."""
    relative = target.name  # same dir
    content = (
        '"""Auto-generated. Re-exports the latest reward version."""\n'
        "from __future__ import annotations\n\n"
        "import importlib.util\n"
        "from pathlib import Path\n\n"
        f"_TARGET = Path(__file__).resolve().parent / {relative!r}\n"
        "_spec = importlib.util.spec_from_file_location("
        '"_sculpt_current_reward", _TARGET)\n'
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "assert _spec.loader is not None\n"
        "_spec.loader.exec_module(_mod)\n\n"
        "compute_reward = _mod.compute_reward\n"
        "REWARD_SPEC = _mod.REWARD_SPEC\n"
    )
    (rewards_dir / "current.py").write_text(content, encoding="utf-8")
