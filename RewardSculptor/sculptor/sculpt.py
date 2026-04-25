"""sculptor/sculpt.py — inner-loop orchestrator + project scaffolder.

`sculpt run` drives one sculpt cycle end-to-end per iteration:

    1. adapter.train(reward_module_path=<project>/rewards/current.py,
                     output_dir=<project>/runs/iter_<i>/, steps, seed)
    2. adapter.rollout(checkpoint, output_dir=<project>/runs/iter_<i>/rollout/)
    3. diagnose(iter_dir, behavior_goal, config)
           -> <project>/runs/iter_<i>/diagnosis.json
    4. apply_edits(current_reward, diagnosis, iter_<i+1>, adapter.reward_contract())
           -> <project>/rewards/v<i+1>.py  (and rewrites current.py)
    5. git commit in the project dir (if it's a git repo)
    6. append CHANGELOG.md, update reports/provenance.json

Flags:
  --iterations N      total iterations this invocation runs (default 10)
  --resume            start at the next iteration after the highest v<n>.py
  --no-kg             pass empty literature_context to the diagnoser
                      (KG queries are skipped — pure ablation mode)
  --dry-run           bypass the LLM in diagnose + apply_edits, cap training
                      steps at 1000. Used for plumbing smoke tests.

Early-stop: if the adapter's primary_metric does not exceed its best-so-far
for `early_stop_patience` consecutive iterations (default 3), the loop
halts. §Ship-9a: `early_stop_enabled = false` under `[iteration]` turns
the check off entirely — preferred for long overnight runs where a
transient metric dip can mask real behavioral improvement.

`sculpt init <project_dir> --adapter <name>` scaffolds a fresh project with
config.toml, rewards/v0.py, kg_seeds.yml, .gitignore, and an initial git
commit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sculptor.adapters.base import SculptorAdapter, load_adapter
from sculptor.diagnose import Diagnosis, ProposedEdit, diagnose as run_diagnose
from sculptor.edit import (
    EditValidationError,
    _write_current_reexport,
    apply_edits,
)


# ── Structured-event markers (additive; dual-write) ──────────────────────
# Consumers (e.g. the reward-sculptor-ui backend) parse lines of the form
#   [SCULPT-EVENT] {"type": "...", ...}
# out of this CLI's stdout. The markers are ADDITIVE — every existing
# human-readable print stays as-is — so downstream terminals and scripts
# that already grep `[sculpt]` lines keep working unchanged.
_EVENT_TAG = "[SCULPT-EVENT]"


def _emit_event(payload: dict[str, Any]) -> None:
    try:
        line = _EVENT_TAG + " " + json.dumps(payload, default=str)
    except Exception:  # noqa: BLE001 — never let event emission crash the run
        return
    print(line, flush=True)


# ── Public result shape ──────────────────────────────────────────────────
@dataclass
class IterOutcome:
    iter_index: int
    iter_dir: Path
    reward_path_before: Path
    reward_path_after: Path | None
    primary_metric: float | None
    behavior: dict[str, Any]
    failure_modes: list[str]
    edit_count: int


@dataclass
class SculptRunResult:
    iterations_run: int
    completed_iters: list[IterOutcome] = field(default_factory=list)
    early_stopped: bool = False
    early_stop_reason: str = ""
    primary_metric_history: list[float] = field(default_factory=list)

    @property
    def final_reward_path(self) -> Path | None:
        for outcome in reversed(self.completed_iters):
            if outcome.reward_path_after:
                return outcome.reward_path_after
        return None


# ── Config / paths ───────────────────────────────────────────────────────
def _parse_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py310 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as f:
        return tomllib.load(f)


def _project_paths(config_path: Path) -> dict[str, Path]:
    project = config_path.resolve().parent
    return {
        "project": project,
        "rewards": project / "rewards",
        "runs": project / "runs",
        "reports": project / "reports",
    }


def _find_latest_reward_version(rewards_dir: Path) -> tuple[int, Path]:
    """Return (highest_n, path_to_v<n>.py). Falls back to v0.py if nothing
    higher exists. Raises if even v0.py is missing."""
    best_n = -1
    best_path: Path | None = None
    for p in rewards_dir.glob("v*.py"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n = n
                best_path = p
    if best_path is None:
        raise FileNotFoundError(
            f"no v<n>.py reward file in {rewards_dir} — run `sculpt init` first.")
    return best_n, best_path


def _ensure_current_py(rewards_dir: Path) -> Path:
    """Make sure rewards/current.py exists and re-exports the latest v<n>.py."""
    current = rewards_dir / "current.py"
    _, latest = _find_latest_reward_version(rewards_dir)
    if not current.is_file():
        _write_current_reexport(rewards_dir, latest)
    return current


# ── Git helpers (optional) ───────────────────────────────────────────────
def _is_git_repo(project: Path) -> bool:
    r = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "--git-dir"],
        capture_output=True)
    return r.returncode == 0


def _git_add_commit(project: Path, message: str) -> bool:
    """Stage all and commit. Returns True on success, False otherwise."""
    if not _is_git_repo(project):
        return False
    try:
        subprocess.run(["git", "-C", str(project), "add", "."],
                       check=True, capture_output=True)
        # Only commit if there's something staged.
        check = subprocess.run(
            ["git", "-C", str(project), "diff", "--cached", "--quiet"],
            capture_output=True)
        if check.returncode == 0:
            return False  # nothing to commit
        subprocess.run(
            ["git", "-C", str(project), "commit", "-m", message],
            check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:  # pragma: no cover
        sys.stderr.write(f"[sculpt] git commit failed: {e.stderr!r}\n")
        return False


def _scoped_git_commit(project: Path, *, paths: list[str], message: str) -> bool:
    """§Ship-8b: stage ONLY the given relative paths (not the whole tree)
    and commit. Prevents auto-apply commits from bundling in-flight
    reward / run artifacts that are dirty mid-iter. Returns True on
    success, False otherwise (not a repo, nothing to stage, git error,
    unsafe path).

    §Ship-8c hotfix (critique 4): path-traversal defense. Reject any
    absolute path or any path whose normalized form escapes `project`.
    Defense-in-depth — today's only caller passes the literal
    `"uploads/robot"` but a future caller could be less careful.
    Also strip newlines from `message` so git one-line log stays clean.
    """
    if not _is_git_repo(project):
        return False
    project_real = project.resolve()
    safe_paths: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_absolute():
            sys.stderr.write(
                f"[sculpt] scoped git commit refused absolute path: {raw!r}\n"
            )
            return False
        # Resolve through the project root and re-check containment.
        try:
            resolved = (project_real / p).resolve()
            resolved.relative_to(project_real)
        except (ValueError, OSError):
            sys.stderr.write(
                f"[sculpt] scoped git commit refused out-of-tree path: {raw!r}\n"
            )
            return False
        safe_paths.append(str(p))

    # Single-line commit summary — git -m tolerates newlines but the log
    # becomes ugly. Keep first ~200 chars of the first line.
    safe_msg = str(message).splitlines()[0] if message else "commit"
    safe_msg = safe_msg[:200]

    try:
        subprocess.run(
            ["git", "-C", str(project), "add", "--", *safe_paths],
            check=True, capture_output=True,
        )
        check = subprocess.run(
            ["git", "-C", str(project), "diff", "--cached", "--quiet"],
            capture_output=True,
        )
        if check.returncode == 0:
            return False  # nothing staged in those paths
        subprocess.run(
            ["git", "-C", str(project), "commit", "-m", safe_msg],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:  # pragma: no cover
        sys.stderr.write(
            f"[sculpt] scoped git commit failed: {e.stderr!r}\n"
        )
        return False


# ── Dry-run stubs ────────────────────────────────────────────────────────
def _dry_run_diagnose(
    iter_dir: Path, behavior_goal: str,
) -> Diagnosis:
    """Canned Diagnosis for --dry-run. Does NOT hit the LLM."""
    metrics = _load_json_if_present(iter_dir / "metrics.json")
    behavior = _load_json_if_present(iter_dir / "rollout" / "behavior.json")
    if not behavior:
        behavior = _load_json_if_present(iter_dir / "behavior.json")
    evidence = (
        f"Dry-run stub diagnosis. "
        f"mean_return={metrics.get('metrics', {}).get('mean_return', 'n/a')}, "
        f"behavior={behavior!r}."
    )
    return Diagnosis(
        failure_modes=["component_imbalance"],
        evidence=evidence,
        proposed_edits=[
            ProposedEdit(
                target_term="alive_bonus",
                operation="increase",
                rationale=(
                    "novel. Dry-run canned edit — bumps alive_bonus by 0.5 "
                    "so every iteration produces a distinct reward file."
                ),
                suggested_value="bumped",
                paper_refs=[],
            ),
        ],
        literature_context=[],
        confidence=0.5,
        iter_dir=str(iter_dir),
        behavior_goal=behavior_goal,
    )


def _dry_run_apply_edits(
    current_reward_path: Path,
    new_iter_id: str,
) -> Path:
    """Regex-only reward bump for --dry-run. Does NOT hit the LLM and
    intentionally skips the full validation path (it's a plumbing test)."""
    src = current_reward_path.read_text(encoding="utf-8")
    parent_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]

    def _sub_spec(pattern: str, replacement: str, text: str) -> str:
        return re.sub(pattern, replacement, text, count=1)

    # version / parent_hash / author
    new_src = _sub_spec(
        r'("version"\s*:\s*)"[^"]+"',
        rf'\1"{new_iter_id}"', src)
    if '"parent_hash"' in new_src:
        new_src = _sub_spec(
            r'("parent_hash"\s*:\s*)"[^"]*"',
            rf'\1"{parent_hash}"', new_src)
    else:
        new_src = _sub_spec(
            r'("version"\s*:\s*)"[^"]+"(,?)',
            rf'\1"{new_iter_id}"\2\n    "parent_hash": "{parent_hash}",',
            new_src)
    new_src = _sub_spec(
        r'("author"\s*:\s*)"[^"]+"',
        r'\1"sculptor"', new_src)

    # Bump alive_bonus by 0.5 if present, else skip silently.
    def _bump_alive(m):
        val = float(m.group(1))
        return f'"alive_bonus": {val + 0.5}'
    new_src = re.sub(r'"alive_bonus"\s*:\s*([\d.eE+-]+)', _bump_alive, new_src)

    target = current_reward_path.parent / f"{new_iter_id}.py"
    target.write_text(new_src, encoding="utf-8")
    _write_current_reexport(current_reward_path.parent, target)
    return target


def _find_materialized_mjcf(project: Path) -> Path | None:
    """Return `<project>/uploads/robot/*.xml` if a materialized MJCF
    exists (committed to the project by a prior physics-edit or
    bootstrap). §Ship-8b: sculpt's auto-apply path requires a local
    MJCF — library-only projects fall back to the UI click-through."""
    upload_dir = project / "uploads" / "robot"
    if not upload_dir.is_dir():
        return None
    xmls = sorted(upload_dir.glob("*.xml"))
    if not xmls:
        return None
    return xmls[0]


def _maybe_apply_auto_physics_edit(
    *,
    project: Path,
    iter_index: int,
    prompt_text: str,
    audit: dict[str, Any],
    kg_store,
) -> None:
    """§Ship-8b: inline physics edit when `auto_adjust_physics` +
    severe verdict + materialized MJCF + ANTHROPIC_API_KEY all line
    up. Best-effort — every failure is logged + event-emitted without
    blocking the loop.
    """
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        _emit_event({
            "type": "physics_auto_apply_skipped",
            "iter": iter_index,
            "reason": "ANTHROPIC_API_KEY not set — suggestion only.",
        })
        return
    mjcf_path = _find_materialized_mjcf(project)
    if mjcf_path is None:
        _emit_event({
            "type": "physics_auto_apply_skipped",
            "iter": iter_index,
            "reason": (
                "no materialized MJCF under uploads/robot/ — open the "
                "Physics tab once to materialize (or click 'apply "
                "physics fix' in Runs), then subsequent iters auto-apply."
            ),
        })
        return
    try:
        from sculptor.adapters.mjcf_editor import apply_mjcf_edit
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: auto-physics import failed: "
            f"{type(e).__name__}: {e}\n"
        )
        return

    # §Ship-8c hotfix (critique medium-2): emit a started event so the
    # UI can disable the "apply physics fix" chip while sculpt is doing
    # the same operation — otherwise a user click races against our own
    # apply and the filelock serializes them with no UI feedback.
    _emit_event({
        "type": "physics_auto_apply_started",
        "iter": iter_index,
        "mjcf_path": str(mjcf_path),
    })

    try:
        result = apply_mjcf_edit(
            mjcf_path=mjcf_path,
            user_prompt=prompt_text,
            adapter_hint="auto-physics (sculpt-side)",
            kg_store=kg_store,
            write=True,
        )
    except Exception as e:  # noqa: BLE001 — never block the training loop
        _emit_event({
            "type": "physics_auto_apply_errored",
            "iter": iter_index,
            "reason": f"{type(e).__name__}: {e}",
        })
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: physics auto-apply errored: "
            f"{type(e).__name__}: {e}\n"
        )
        return

    if not result.get("committed"):
        _emit_event({
            "type": "physics_auto_apply_rejected",
            "iter": iter_index,
            "rejected_at": result.get("rejected_at"),
            "rejected_reason": result.get("rejected_reason"),
        })
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: physics auto-apply rejected at "
            f"{result.get('rejected_at')!r}: {result.get('rejected_reason')}\n"
        )
        return

    # Success — git-commit the change so the next iter's resume logic
    # sees a clean project state. §Ship-8b hotfix (critique critical-3):
    # scope `git add` to `uploads/robot/` only so in-flight reward /
    # run artifacts don't get bundled into the auto-physics commit.
    summary = result.get("summary", "auto-physics edit")
    _scoped_git_commit(
        project,
        paths=["uploads/robot"],
        message=f"auto-physics (iter {iter_index}): {summary}",
    )
    _emit_event({
        "type": "physics_auto_applied",
        "iter": iter_index,
        "summary": summary,
        "diff_lines": len(result.get("diff_lines") or []),
        "kg_citations": result.get("kg_citations") or [],
    })
    sys.stderr.write(
        f"[sculpt] iter {iter_index}: physics auto-applied. Summary: {summary}\n"
    )


def _load_json_if_present(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ── Metric extraction ────────────────────────────────────────────────────
def _extract_primary_metric(
    metrics_json: dict, behavior_json: dict, primary_key: str,
) -> float | None:
    """Scan metrics.json + behavior.json for the `primary_metric` key."""
    # Adapter's metrics.json is wrapped: {"metrics": {...}, "components": {...}}
    for bag in (
        metrics_json,
        metrics_json.get("metrics", {}) or {},
        metrics_json.get("components", {}) or {},
        behavior_json,
    ):
        if isinstance(bag, dict) and primary_key in bag:
            try:
                return float(bag[primary_key])
            except Exception:  # noqa: BLE001
                return None
    return None


# ── CHANGELOG.md ─────────────────────────────────────────────────────────
def _fmt_delta(curr: float | None, prev: float | None) -> str:
    if curr is None:
        return "n/a"
    if prev is None:
        return f"{curr:+.4f} (Δ —)"
    return f"{curr:+.4f} (Δ {curr - prev:+.4f} vs prev)"


def _append_changelog(
    project: Path,
    iter_index: int,
    reward_path_before: Path,
    reward_path_after: Path | None,
    primary_metric: float | None,
    previous_metric: float | None,
    primary_key: str,
    behavior_metric_names: list[str],
    behavior: dict,
    diagnosis: Diagnosis,
) -> Path:
    path = project / "CHANGELOG.md"
    if not path.is_file():
        path.write_text(
            "# Sculpt Changelog\n\n"
            "Auto-appended by `sculpt run`. Each entry records the iteration "
            "outcome, the diagnosed failure modes, and the edits applied to "
            "produce the next reward version.\n\n",
            encoding="utf-8")

    lines: list[str] = []
    lines.append(f"## Iteration {iter_index}\n")
    lines.append(f"- **Reward before**: `{reward_path_before.name}`")
    if reward_path_after:
        lines.append(f"- **Reward after**:  `{reward_path_after.name}`")
    lines.append(f"- **Primary metric** (`{primary_key}`): "
                 f"{_fmt_delta(primary_metric, previous_metric)}")
    if behavior_metric_names:
        parts: list[str] = []
        for k in behavior_metric_names:
            v = behavior.get(k)
            if v is None:
                continue
            try:
                parts.append(f"`{k}`={float(v):.4g}")
            except Exception:  # noqa: BLE001
                parts.append(f"`{k}`={v}")
        if parts:
            lines.append(f"- **Behavior metrics**: {', '.join(parts)}")
    lines.append(f"- **Failure modes**: "
                 f"{', '.join(diagnosis.failure_modes) or '(none)'}")
    lines.append(f"- **Diagnosis confidence**: {diagnosis.confidence:.2f}")
    if diagnosis.evidence:
        ev = diagnosis.evidence.strip().replace("\n", " ")
        lines.append(f"- **Evidence**: {ev}")

    if diagnosis.proposed_edits:
        lines.append("- **Edits**:")
        for e in diagnosis.proposed_edits:
            deferred = " *(deferred — requires env extension)*" if \
                getattr(e, "requires_env_extension", False) else ""
            sv = f" → `{e.suggested_value}`" if e.suggested_value else ""
            lines.append(f"  - [{e.operation}] `{e.target_term}`{sv}{deferred}")
            lines.append(f"    - *rationale*: {e.rationale}")
            if e.paper_refs:
                lines.append(
                    f"    - *paper_refs*: "
                    f"{', '.join('arXiv:' + a for a in e.paper_refs)}")
            else:
                lines.append("    - *paper_refs*: (novel)")
    else:
        lines.append("- **Edits**: (none)")
    lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ── provenance.json ──────────────────────────────────────────────────────
def _load_provenance(project: Path) -> dict[str, list[dict]]:
    p = project / "reports" / "provenance.json"
    return _load_json_if_present(p) or {}


def _write_provenance(
    project: Path, provenance: dict[str, list[dict]]
) -> Path:
    p = project / "reports" / "provenance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    return p


def _update_provenance(
    project: Path,
    iter_index: int,
    diagnosis: Diagnosis,
    new_reward_path: Path | None,
    reward_contract,
    kg_store=None,
) -> Path:
    """Append one entry per applied edit that cited a paper, then recompute
    `still_active` for every entry based on whether its `target_term` still
    appears in the latest reward's components-or-hyperparameters."""
    provenance = _load_provenance(project)

    # Each applied (non-deferred) edit with paper_refs contributes entries.
    for e in diagnosis.proposed_edits:
        if getattr(e, "requires_env_extension", False):
            continue
        if not e.paper_refs:
            continue
        bucket = provenance.setdefault(e.target_term, [])
        existing_ids = {entry.get("arxiv_id") for entry in bucket}
        for aid in e.paper_refs:
            if aid in existing_ids:
                continue
            citation = ""
            if kg_store is not None:
                try:
                    from sculptor.kg.query import cite
                    citation = cite(aid, store=kg_store)
                except Exception:  # noqa: BLE001
                    pass
            bucket.append({
                "arxiv_id": aid,
                "citation": citation,
                "iter_introduced": iter_index + 1,
                "how_used": e.rationale,
                "still_active": True,
            })

    # Recompute still_active: read the LATEST reward and see which target
    # names are still around (as components or hyperparameters).
    alive_keys = _alive_reward_keys(new_reward_path) if new_reward_path else set()
    if alive_keys:
        for key, entries in provenance.items():
            still = key in alive_keys
            for entry in entries:
                entry["still_active"] = bool(still)

    return _write_provenance(project, provenance)


def _alive_reward_keys(reward_path: Path) -> set[str]:
    """Union of REWARD_SPEC.hyperparameters keys and component keys returned
    by compute_reward on zero dummies. Returns the empty set on any error."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"_sculpt_alive_{abs(hash(str(reward_path)))}", reward_path)
        if spec is None or spec.loader is None:
            return set()
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rs = getattr(mod, "REWARD_SPEC", {}) or {}
        hparams = set((rs.get("hyperparameters") or {}).keys())
        # We cannot build contract-shaped dummies here (no contract in
        # scope), so skip component probing — hparam set is good enough for
        # still_active bookkeeping since sculpt edits target hparam names.
        return hparams
    except Exception:  # noqa: BLE001
        return set()


# ── Per-phase resume helpers ─────────────────────────────────────────────
def _train_or_resume(
    *, adapter, iter_index: int, iter_dir: Path,
    reward_module_path: Path, steps: int, seed: int,
    init_policy_path: Optional[Path] = None,
):
    """Skip `adapter.train` when `iter_dir/checkpoint.pt` is already on
    disk and loads successfully — the expensive phase (≥ 22 min for
    mjlab) is idempotent given the same reward and seed, so reusing a
    prior run's artifact saves all that wall-clock when any downstream
    phase (rollout, diagnose, edit) failed on the previous attempt.
    Returns a `TrainResult`-compatible object either way.

    §Ship 15: `init_policy_path` is an optional path to a pre-trained
    rsl_rl checkpoint. When set AND the adapter's `train` signature
    accepts `init_policy_path`, it's forwarded so the runner warm-starts
    actor+critic weights from that checkpoint. When the adapter DOESN'T
    accept the kwarg, or when this iter's own checkpoint is already on
    disk (resume path wins), `warm_start_skipped` is emitted so callers
    (Ship 16 orchestrator) can distinguish "warm-started" from "resumed
    from prior partial attempt" rather than the two silently collapsing.
    """
    from sculptor.adapters.base import TrainResult

    # Most adapters save to `checkpoint.pt`; gym_sb3 uses `checkpoint.zip`.
    # Check for either so we can resume across both paths.
    for ext in ("pt", "zip"):
        ckpt = iter_dir / f"checkpoint.{ext}"
        if not ckpt.is_file() or ckpt.stat().st_size == 0:
            continue
        # Integrity check: torch can parse the `.pt`; we treat `.zip`
        # (SB3) as opaque since we don't have a standalone validator.
        ok = True
        if ext == "pt":
            try:
                import torch as _torch
                _torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 — corrupt checkpoint, fall through
                ok = False
        if not ok:
            continue
        # Assemble best-effort metrics from prior metrics.json.
        metrics: dict[str, float] = {}
        try:
            metrics = json.loads((iter_dir / "metrics.json").read_text())
            metrics = {k: float(v) for k, v in metrics.items()
                       if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            pass
        _emit_event({
            "type": "phase_skipped",
            "iter": iter_index, "phase": "train",
            "reason": "checkpoint already on disk",
            "checkpoint": str(ckpt),
        })
        # §Ship 15: caller requested warm-start, but iter's own
        # checkpoint already exists on disk — resume path wins.
        # Emit so Ship 16 can tell "I warm-started" apart from
        # "I resumed an in-flight iter".
        if init_policy_path is not None:
            _emit_event({
                "type": "warm_start_skipped",
                "iter": iter_index,
                "reason": "local_checkpoint_wins",
                "source": str(init_policy_path),
            })
        return TrainResult(
            checkpoint_path=ckpt,
            metrics_dict=metrics,
            component_means={},
            logs_path=iter_dir / "logs",
        )

    # Fresh training run — no checkpoint or all candidates corrupt.
    train_kwargs: dict[str, Any] = dict(
        reward_module_path=reward_module_path,
        output_dir=iter_dir,
        steps=steps,
        seed=seed,
    )
    # §Ship 15: introspect adapter.train so adapters that don't yet
    # support warm-start (gym_sb3 / mjx / rllib) don't TypeError on
    # the new kwarg. If a caller passed init_policy_path but the
    # adapter dropped it, emit so the orchestrator sees the silent
    # no-op instead of assuming warm-start happened.
    #
    # Introspection note (audit finding, Ship 15 review): a `**kwargs`
    # catch-all in the adapter's signature would silently discard the
    # kwarg while `"init_policy_path" in sig.parameters` returns False.
    # Accept either an explicit named param OR a VAR_KEYWORD param —
    # the former is the preferred contract, the latter is "adapter
    # MIGHT support it, let's try and let the adapter decide."
    if init_policy_path is not None:
        import inspect
        sig = inspect.signature(adapter.train)
        has_explicit = "init_policy_path" in sig.parameters
        has_var_kwarg = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_explicit or has_var_kwarg:
            train_kwargs["init_policy_path"] = init_policy_path
        else:
            _emit_event({
                "type": "warm_start_skipped",
                "iter": iter_index,
                "reason": "adapter_does_not_support",
                "source": str(init_policy_path),
                "adapter": type(adapter).__name__,
            })
    return adapter.train(**train_kwargs)


def _rollout_or_resume(
    *, adapter, iter_index: int, rollout_dir: Path,
    checkpoint_path: Path, n_episodes: int,
    max_episode_steps: int | None = None,
    playback_speed: float | None = None,
    render_every: int | None = None,
    fps: float | None = None,
) -> None:
    """Skip `adapter.rollout` when the three artifacts it produces
    (`rollout.mp4` + `trajectory.npz` + `behavior.json`) are ALL on
    disk. Partial rollouts re-run cleanly; we don't try to merge.

    §Ship-7: video knobs (max_episode_steps / playback_speed / etc.)
    are threaded through adapter.rollout; adapters that don't support
    them (gym_sb3) silently ignore via `**kwargs`.
    """
    required = (
        rollout_dir / "rollout.mp4",
        rollout_dir / "trajectory.npz",
        rollout_dir / "behavior.json",
    )
    if all(p.is_file() and p.stat().st_size > 0 for p in required):
        _emit_event({
            "type": "phase_skipped",
            "iter": iter_index, "phase": "rollout",
            "reason": "rollout artifacts already on disk",
        })
        return
    # Pass video knobs only to adapters that declare them — older
    # adapter.rollout signatures (gym_sb3) don't accept the new kwargs
    # and would TypeError. Introspect once.
    import inspect
    sig = inspect.signature(adapter.rollout)
    extra: dict[str, Any] = {}
    for name, value in (
        ("max_episode_steps", max_episode_steps),
        ("playback_speed", playback_speed),
        ("render_every", render_every),
        ("fps", fps),
    ):
        if value is not None and name in sig.parameters:
            extra[name] = value
    adapter.rollout(
        checkpoint_path=checkpoint_path,
        output_dir=rollout_dir,
        n_episodes=n_episodes,
        **extra,
    )


# ── One iteration ────────────────────────────────────────────────────────
def _run_one_iter(
    *,
    iter_index: int,
    adapter: SculptorAdapter,
    project: Path,
    rewards_dir: Path,
    runs_dir: Path,
    config_path: Path,
    behavior_goal: str,
    cfg: dict,
    no_kg: bool,
    dry_run: bool,
    kg_store,
    seed: int,
    init_policy_path: Optional[Path] = None,
) -> IterOutcome:
    iter_cfg = cfg.get("iteration", {}) or {}
    primary_key = str(iter_cfg.get("primary_metric", "mean_return"))
    behavior_metric_names: list[str] = list(iter_cfg.get("behavior_metrics", []))
    steps = int(iter_cfg.get("steps_per_iter", 50_000))
    if dry_run:
        steps = min(1000, steps)

    iter_dir = runs_dir / f"iter_{iter_index}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    reward_path_before = _ensure_current_py(rewards_dir)
    latest_n, latest_reward_file = _find_latest_reward_version(rewards_dir)

    _emit_event({
        "type": "iter_started",
        "iter": iter_index,
        "steps": steps,
        "reward_version_before": latest_n,
        "dry_run": bool(dry_run),
        "no_kg": bool(no_kg),
        # §Ship 15: None unless caller requested warm-start for this
        # iter. Ship 16 orchestrator uses this to correlate intent
        # with the `warm_start_loaded` event the subprocess emits
        # later — "caller expected warm-start AND subprocess loaded
        # the ckpt" vs "caller expected warm-start but subprocess
        # didn't emit warm_start_loaded" (= silent drop → bug).
        "warm_start_source": (
            str(init_policy_path) if init_policy_path is not None
            else None
        ),
    })

    # 1. Train — resume from on-disk checkpoint when one is already
    # present (overnight reliability: if a prior run errored at any
    # post-train phase, `iter_<i>/checkpoint.pt` is still on disk and
    # represents ~22 min of GPU work we must not redo).
    t0 = time.time()
    train_result = _train_or_resume(
        adapter=adapter,
        iter_index=iter_index,
        iter_dir=iter_dir,
        reward_module_path=reward_path_before,
        steps=steps,
        seed=seed,
        init_policy_path=init_policy_path,
    )
    train_s = time.time() - t0

    # 2. Rollout — use the checkpoint path the adapter actually wrote.
    # Different adapters use different extensions: gym_sb3 writes
    # `checkpoint.zip` (SB3 convention), mjlab writes `checkpoint.pt`
    # (torch.save format). Previously hardcoded `checkpoint.zip` here,
    # which made iter 1 explode with FileNotFoundError the moment
    # mjlab's iter 0 finished training.
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir(exist_ok=True)
    checkpoint_path = (
        train_result.checkpoint_path
        if train_result is not None and train_result.checkpoint_path is not None
        else iter_dir / "checkpoint.zip"
    )
    _rollout_or_resume(
        adapter=adapter,
        iter_index=iter_index,
        rollout_dir=rollout_dir,
        checkpoint_path=checkpoint_path,
        n_episodes=int(iter_cfg.get("rollout_episodes", 6)),
        # §Ship-7: rollout video knobs — default None means runner defaults
        # (500 steps, playback 1x, auto render_every + fps). Each key
        # overrides independently when set in config.toml or passed via
        # the `sculpt run` CLI flags.
        max_episode_steps=iter_cfg.get("max_episode_steps"),
        playback_speed=iter_cfg.get("playback_speed"),
        render_every=iter_cfg.get("render_every"),
        fps=iter_cfg.get("rollout_fps"),
    )

    # §7.3: physics-realism audit. Reads the expanded `trajectory.npz`
    # (§7.1) + `mjcf_limits.json` that the rollout runner drops next to
    # it, computes torque-saturation / joint-vel / joint-limit metrics,
    # and persists `iter_<N>/realism_audit.json` so diagnose + UI can
    # surface the verdict. Best-effort — failures (missing file, shape
    # drift) return verdict=unknown rather than crashing the loop.
    audit_result: dict[str, Any] | None = None
    try:
        from sculptor.adapters.realism import audit_rollout
        audit_result = audit_rollout(
            trajectory_path=rollout_dir / "trajectory.npz",
            limits_path=rollout_dir / "mjcf_limits.json",
        )
        (iter_dir / "realism_audit.json").write_text(
            json.dumps(audit_result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        _emit_event({
            "type": "realism_audited",
            "iter": iter_index,
            "verdict": audit_result.get("verdict", "unknown"),
            "torque_saturation_frac": audit_result.get("torque_saturation_frac"),
            "any_joint_saturation_max": audit_result.get("any_joint_saturation_max"),
            "joint_vel_p99_max": audit_result.get("joint_vel_p99_max"),
            "joint_limit_violation_frac": audit_result.get("joint_limit_violation_frac"),
            "top_joints_saturation": audit_result.get("top_joints_saturation") or [],
        })
    except Exception as e:  # noqa: BLE001 — audit must not block the loop
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: realism audit skipped — "
            f"{type(e).__name__}: {e}\n"
        )

    # §7.4 / §Ship-8b: compute + emit the auto-physics SUGGESTION here
    # (UI surfaces the chip ASAP, before diagnose finishes). The actual
    # apply happens AFTER diagnose (see below) so Claude's reward
    # edits reason about the PRE-edit MJCF that produced the audit,
    # not a post-edit one.
    auto_physics_prompt: str | None = None
    try:
        from sculptor.adapters.auto_physics import (
            should_auto_adjust_physics, synthesize_auto_physics_prompt,
        )
        auto_adjust_enabled = bool(iter_cfg.get("auto_adjust_physics", False))
        if should_auto_adjust_physics(
            audit_result, auto_adjust_enabled=auto_adjust_enabled,
        ):
            auto_physics_prompt = synthesize_auto_physics_prompt(audit_result or {})
            _emit_event({
                "type": "physics_edit_suggested",
                "iter": iter_index,
                "prompt": auto_physics_prompt,
                "verdict": (audit_result or {}).get("verdict"),
                "top_joints_saturation": (
                    (audit_result or {}).get("top_joints_saturation") or []
                ),
            })
            sys.stderr.write(
                f"[sculpt] iter {iter_index}: auto-physics suggestion ready "
                f"(feature-flag on, verdict=severe). Prompt:\n"
                f"{auto_physics_prompt}\n\n"
            )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: auto-physics suggestion skipped — "
            f"{type(e).__name__}: {e}\n"
        )

    # 3. Diagnose
    if dry_run:
        diagnosis = _dry_run_diagnose(iter_dir, behavior_goal)
        # Persist it (diagnose writes this file when the real fn runs).
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diagnosis.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8")
    else:
        diagnosis = run_diagnose(
            iter_dir=iter_dir, behavior_goal=behavior_goal,
            config=config_path, store=kg_store, skip_kg=no_kg,
        )

    # Metrics + behavior for the primary_metric / changelog
    metrics_json = _load_json_if_present(iter_dir / "metrics.json")
    behavior_json = _load_json_if_present(rollout_dir / "behavior.json")
    primary_metric = _extract_primary_metric(
        metrics_json, behavior_json, primary_key)

    # §7.4 / §Ship-8b: now apply the physics edit, AFTER diagnose so
    # Claude's diagnosis reasons about the MJCF that actually produced
    # the audit. The Claude-generated physics edit lands BEFORE iter
    # N+1's training spawns, so the next rollout uses the new MJCF.
    if auto_physics_prompt and not dry_run:
        _maybe_apply_auto_physics_edit(
            project=project,
            iter_index=iter_index,
            prompt_text=auto_physics_prompt,
            audit=audit_result or {},
            kg_store=kg_store,
        )

    # 4. Apply edits → v<n+1>.py
    new_iter_tag = f"v{latest_n + 1}"
    new_reward_path: Path | None = None
    try:
        if dry_run:
            new_reward_path = _dry_run_apply_edits(
                current_reward_path=latest_reward_file,
                new_iter_id=new_iter_tag)
        else:
            # Filter out fully-deferred — apply_edits raises on empty
            # applicable_edits, which we want to surface clearly.
            # §7.2: pass iter_dir so apply_edits can load
            # `reward_trajectory.json` and inject it into the rewrite
            # prompt (same data the diagnoser saw).
            new_reward_path = apply_edits(
                current_reward_path=latest_reward_file,
                diagnosis=diagnosis,
                new_iter_id=new_iter_tag,
                reward_contract=adapter.reward_contract(),
                kg_store=kg_store,
                iter_dir=iter_dir,
            )
    except EditValidationError as e:
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: apply_edits skipped — "
            f"{type(e).__name__}: {e}\n")

    # 5. Previous metric for delta display
    history_path = project / "reports" / "metric_history.json"
    prior = _load_json_if_present(history_path).get("history", [])

    # 6. CHANGELOG
    _append_changelog(
        project=project, iter_index=iter_index,
        reward_path_before=reward_path_before,
        reward_path_after=new_reward_path,
        primary_metric=primary_metric,
        previous_metric=(prior[-1] if prior else None),
        primary_key=primary_key,
        behavior_metric_names=behavior_metric_names,
        behavior=behavior_json, diagnosis=diagnosis,
    )

    # Persist metric history for the next iter's delta
    prior.append(primary_metric if primary_metric is not None else 0.0)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"primary_metric": primary_key, "history": prior},
                   indent=2, default=str),
        encoding="utf-8")

    # 7. provenance
    _update_provenance(
        project=project, iter_index=iter_index, diagnosis=diagnosis,
        new_reward_path=new_reward_path, reward_contract=adapter.reward_contract(),
        kg_store=kg_store,
    )

    # 8. Git commit
    summary = ", ".join(diagnosis.failure_modes or ["(no failure modes)"])
    edit_count = sum(
        1 for e in diagnosis.proposed_edits
        if not getattr(e, "requires_env_extension", False))
    commit_msg = (
        f"iter {iter_index}: {summary} "
        f"[{edit_count} edit{'s' if edit_count != 1 else ''}"
        + (" + dry-run" if dry_run else "")
        + (" + no-kg" if no_kg else "")
        + "]"
    )
    _git_add_commit(project, commit_msg)

    reward_version_after: int | None = None
    if new_reward_path is not None:
        m = re.fullmatch(r"v(\d+)", new_reward_path.stem)
        if m:
            reward_version_after = int(m.group(1))

    # paper_refs across all applied (non-deferred) edits this iter.
    applied_paper_refs: list[str] = []
    for e in diagnosis.proposed_edits:
        if getattr(e, "requires_env_extension", False):
            continue
        applied_paper_refs.extend(e.paper_refs or [])

    prev_metric = prior[-2] if len(prior) >= 2 else None
    metric_delta: float | None = None
    if primary_metric is not None and isinstance(prev_metric, (int, float)):
        metric_delta = float(primary_metric) - float(prev_metric)

    _emit_event({
        "type": "iter_completed",
        "iter": iter_index,
        "failure_modes": list(diagnosis.failure_modes),
        "edit_count": edit_count,
        "primary_metric": primary_metric,
        "metric_delta": metric_delta,
        "reward_version_before": latest_n,
        "reward_version_after": reward_version_after,
        "paper_refs": sorted(set(applied_paper_refs)),
    })

    return IterOutcome(
        iter_index=iter_index,
        iter_dir=iter_dir,
        reward_path_before=reward_path_before,
        reward_path_after=new_reward_path,
        primary_metric=primary_metric,
        behavior=behavior_json,
        failure_modes=list(diagnosis.failure_modes),
        edit_count=edit_count,
    )


# ── Early stop ───────────────────────────────────────────────────────────
def _is_metric_still_improving(
    history: list[float],
    *,
    threshold: float = 0.05,
    abs_floor: float = 0.05,
) -> bool:
    """§Ship-19d Goal B: detect whether the metric trend on the
    second half of `history` clears the first half by at least
    `threshold` * |prior_best| or `abs_floor`, whichever is larger.

    Designed for SHORT histories (typical 3-12 iters per stage).
    Compares "best in recent half" vs "best in prior half" (NOT
    means) so a single bad iter at the tail doesn't suppress
    extension when the policy IS improving on its peak runs. Robust
    to single noisy spikes in either direction.

    Returns False for histories shorter than 4 iters (insufficient
    signal — extension should NOT fire on tiny stages).

    Examples (threshold=0.05, abs_floor=0.05):
      [0.1, 0.2, 0.3, 0.4]                 → True  (0.4 > 0.2 + max(0.01, 0.05))
      [0.5, 0.5, 0.51, 0.52]               → False (0.52 > 0.5 + max(0.025, 0.05) is False)
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.6]       → True  (0.6 > 0.5 + 0.05)
      [-2.0, -1.5, -1.0, -0.5]             → True  (-0.5 > -1.5 + max(0.075, 0.05))
      [0.9, 0.8, 0.7, 0.6]                 → False (regressing)
      [0.4, 0.5]                           → False (history too short)
    """
    n = len(history)
    if n < 4:
        return False
    half = n // 2
    prior = history[:half]
    recent = history[half:]
    if not prior or not recent:
        return False
    prior_best = max(prior)
    recent_best = max(recent)
    floor = max(abs(prior_best) * float(threshold), float(abs_floor))
    return recent_best > prior_best + floor


def _should_early_stop(
    history: list[float | None],
    patience: int = 3,
    *,
    enabled: bool = True,
) -> bool:
    """True when early-stop should fire now. `enabled=False` forces
    False regardless of history (used when the user disables early-stop
    from the UI). `patience` must be ≥ 1; patience=0 also disables
    (history window of zero is meaningless).

    §Ship-9a: Sam's feedback was that a flat or slightly-decreasing
    primary_metric can mask genuine behavioral improvement (reward
    hacking → sculptor removes the exploit → metric drops but motion
    looks better). Exposing both knobs lets a long overnight run
    complete on the requested iter budget rather than truncating.
    """
    if not enabled or patience < 1:
        return False
    clean = [v for v in history if v is not None]
    if len(clean) < patience + 1:
        return False
    recent_max = max(clean[-patience:])
    prior_max = max(clean[:-patience])
    return recent_max <= prior_max


# ── Public entry: run + init ─────────────────────────────────────────────
def sculpt_run(
    config_path: Path | str,
    behavior_goal: str,
    *,
    iterations: int = 10,
    resume: bool = False,
    no_kg: bool = False,
    dry_run: bool = False,
    steps_per_iter: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    playback_speed: Optional[float] = None,
    render_every: Optional[int] = None,
    rollout_fps: Optional[float] = None,
    rollout_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    auto_adjust_physics: Optional[bool] = None,
    early_stop_enabled: Optional[bool] = None,
    early_stop_patience: Optional[int] = None,
    init_policy_path: Optional[Path | str] = None,
    per_iter_callback: Optional[Callable[["IterOutcome"], Optional[str]]] = None,
) -> SculptRunResult:
    """§Ship-19d: `per_iter_callback` is fired AFTER each iter's
    artifacts are persisted. Returning `None` keeps the loop running;
    returning a non-empty string is interpreted as an early-stop
    reason and breaks the loop after recording that iter as
    completed. Used by mission_run to early-stop a stage the moment
    its success_criterion is satisfied (Goal A) — a no-op for plain
    sculpt_run callers that don't pass it. Distinct from the
    metric-plateau early-stop at lines 1311+: that one looks at the
    history shape; this one is a goal-aware exit signal."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    # §Ship 15: validate init_policy_path at entry so callers see a
    # clear error before any expensive config / adapter loading.
    # Normalized into `init_ckpt` here and threaded through the iter
    # loop below; only iter == start_iter actually uses it.
    #
    # Empty-string defensiveness (audit finding, Ship 15 review):
    # `Path("").resolve() == Path.cwd()`, so a literal `""` from JSON /
    # YAML / form-encoded callers would bypass the None check and then
    # mis-validate as cwd (which may or may not be a file). Treat
    # empty / whitespace-only strings as None explicitly.
    init_ckpt: Optional[Path] = None
    if init_policy_path is not None:
        raw = str(init_policy_path).strip()
        if raw:
            init_ckpt = Path(raw).expanduser().resolve()
            if not init_ckpt.is_file():
                raise FileNotFoundError(
                    f"init_policy_path not found: {init_ckpt}. "
                    "Pass a valid rsl_rl checkpoint or None."
                )

    cfg = _parse_toml(config_path)
    # `--steps-per-iter` CLI override wins over the config.toml value.
    # Wired from the backend's `training_iterations` run-param so UI
    # edits to "rsl_rl iters / cycle" actually reach `_run_one_iter`.
    # Pre-fix the UI field was silently dropped (backend run_manager
    # only forwarded 4 of the 8 NewRunRequest fields).
    if steps_per_iter is not None:
        iter_cfg = dict(cfg.get("iteration") or {})
        iter_cfg["steps_per_iter"] = int(steps_per_iter)
        cfg["iteration"] = iter_cfg
        print(
            f"[sculpt] CLI override: steps_per_iter={steps_per_iter} "
            f"(was {(_parse_toml(config_path).get('iteration') or {}).get('steps_per_iter', '(unset)')} in config.toml)",
            file=sys.stderr, flush=True,
        )
    # §Ship-7: per-run overrides for rollout video / RL knobs. Each
    # override merges into `cfg["iteration"]` so `_run_one_iter` reads a
    # single unified source. None means "use config.toml / runner default".
    _OVERRIDE_KEYS = (
        ("max_episode_steps", max_episode_steps),
        ("playback_speed", playback_speed),
        ("render_every", render_every),
        ("rollout_fps", rollout_fps),
        ("rollout_episodes", rollout_episodes),
        ("seed", seed),
        ("auto_adjust_physics", auto_adjust_physics),
        ("early_stop_enabled", early_stop_enabled),
        ("early_stop_patience", early_stop_patience),
    )
    for key, val in _OVERRIDE_KEYS:
        if val is None:
            continue
        iter_cfg = dict(cfg.get("iteration") or {})
        iter_cfg[key] = val
        cfg["iteration"] = iter_cfg
        print(
            f"[sculpt] CLI override: {key}={val!r}",
            file=sys.stderr, flush=True,
        )
    paths = _project_paths(config_path)
    project = paths["project"]
    rewards_dir = paths["rewards"]
    runs_dir = paths["runs"]
    paths["reports"].mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not rewards_dir.is_dir():
        raise FileNotFoundError(
            f"no rewards/ dir in {project} — run `sculpt init` first.")

    adapter = load_adapter(config_path)

    # Resolve KG store once (shared across iterations). Skip entirely in
    # --no-kg to avoid side-effects on an absent/empty DB.
    kg_store = None
    if not no_kg:
        from sculptor.kg.store import SculptorKG
        kg_store = SculptorKG()

    # Start iter index
    latest_n_before_loop, _ = _find_latest_reward_version(rewards_dir)
    start_iter = latest_n_before_loop if resume else 0
    if not resume and start_iter != 0:
        sys.stderr.write(
            f"[sculpt] warning: rewards/ already has v{start_iter}.py but "
            f"--resume was not passed. Running fresh starting at iter 0.\n")
        start_iter = 0

    end_iter = start_iter + iterations
    # §Ship-7: `[iteration].seed` override (from CLI) wins over legacy
    # top-level `seed` key. Same base_seed + i shifting per iter so
    # resumes and fresh runs stay deterministic.
    _iter_cfg_for_seed = cfg.get("iteration") or {}
    base_seed = int(_iter_cfg_for_seed.get("seed", cfg.get("seed", 42)))

    print(f"[sculpt] project={project}")
    print(f"[sculpt] iterations={iterations} start_iter={start_iter} "
          f"end_iter={end_iter}")
    print(f"[sculpt] no_kg={no_kg} dry_run={dry_run}")

    _emit_event({
        "type": "run_started",
        "project": str(project),
        "iterations": int(iterations),
        "start_iter": int(start_iter),
        "end_iter": int(end_iter),
        "no_kg": bool(no_kg),
        "dry_run": bool(dry_run),
        "behavior_goal": behavior_goal,
    })

    # §Ship 15: `init_ckpt` normalized at function entry. Applies ONLY
    # to the FIRST iter of this run (iter == start_iter). Subsequent
    # iters start fresh; iter-to-iter warm-start within a single
    # sculpt_run is a separate behavioral decision deferred past Ship 16.
    result = SculptRunResult(iterations_run=0)
    try:
        for i in range(start_iter, end_iter):
            t0 = time.time()
            outcome = _run_one_iter(
                iter_index=i, adapter=adapter,
                project=project, rewards_dir=rewards_dir, runs_dir=runs_dir,
                config_path=config_path, behavior_goal=behavior_goal, cfg=cfg,
                no_kg=no_kg, dry_run=dry_run, kg_store=kg_store,
                seed=base_seed + i,
                init_policy_path=(init_ckpt if i == start_iter else None),
            )
            elapsed = time.time() - t0
            result.completed_iters.append(outcome)
            result.primary_metric_history.append(
                outcome.primary_metric if outcome.primary_metric is not None else 0.0)
            result.iterations_run += 1

            # One-line status.
            fm_short = ",".join(outcome.failure_modes) or "none"
            metric_s = (f"{outcome.primary_metric:+.3f}"
                        if outcome.primary_metric is not None else "n/a")
            new_reward_s = (outcome.reward_path_after.name
                            if outcome.reward_path_after else "—")
            print(
                f"[sculpt] iter {outcome.iter_index:>3d}  "
                f"metric={metric_s}  fm={fm_short}  "
                f"edits={outcome.edit_count}  "
                f"new_reward={new_reward_s}  "
                f"t={elapsed:.1f}s",
                flush=True,
            )

            # §Ship-19d Goal A: optional per-iter callback. Mission
            # orchestrator passes one that evaluates the stage's
            # success_criterion on this iter's artifacts. Returning a
            # non-empty string is the early-stop reason; we record
            # the iter as completed (already in `result`) and break.
            # Wrapped in try/except so a buggy caller can't crash
            # sculpt_run mid-iter.
            if per_iter_callback is not None:
                try:
                    callback_reason = per_iter_callback(outcome)
                except Exception as e:  # noqa: BLE001
                    callback_reason = None
                    sys.stderr.write(
                        f"[sculpt] per_iter_callback raised "
                        f"{type(e).__name__}: {e} — ignoring\n"
                    )
                if callback_reason:
                    result.early_stopped = True
                    result.early_stop_reason = str(callback_reason)
                    _emit_event({
                        "type": "early_stop",
                        "at_iter": outcome.iter_index,
                        "reason": result.early_stop_reason,
                        "source": "per_iter_callback",
                    })
                    break

            # §Ship-9a: honor configurable early-stop knobs from
            # [iteration]. Defaults preserve the pre-Ship-9 behavior
            # (enabled=true, patience=3). `or 3` for patience means an
            # explicit `None` in the config falls back to the safe
            # default rather than silently disabling the check.
            _iter_cfg_now = cfg.get("iteration") or {}
            _es_enabled = bool(_iter_cfg_now.get("early_stop_enabled", True))
            _es_patience = int(_iter_cfg_now.get("early_stop_patience") or 3)
            if _should_early_stop(
                result.primary_metric_history,
                patience=_es_patience,
                enabled=_es_enabled,
            ):
                result.early_stopped = True
                result.early_stop_reason = (
                    f"no improvement in {cfg.get('iteration', {}).get('primary_metric', 'mean_return')} "
                    f"over the last {_es_patience} iterations")
                print(f"[sculpt] early-stop: {result.early_stop_reason}")
                _emit_event({
                    "type": "early_stop",
                    "at_iter": outcome.iter_index,
                    "reason": result.early_stop_reason,
                    "patience": _es_patience,
                })
                break
    finally:
        if kg_store is not None:
            kg_store.close()

    _emit_event({
        "type": "run_completed",
        "iterations_run": int(result.iterations_run),
        "early_stopped": bool(result.early_stopped),
        "early_stop_reason": result.early_stop_reason or None,
        "final_reward": (result.final_reward_path.name if result.final_reward_path else None),
        "primary_metric_history": list(result.primary_metric_history),
    })
    return result


# ── sculpt init: scaffolder ──────────────────────────────────────────────
_ADAPTER_SHORT_NAMES = {
    "gym_sb3": "sculptor.adapters.gym_sb3.GymSB3Adapter",
    "mjlab": "sculptor.adapters.mjlab.MjlabAdapter",
}

# Adapter classes whose runtime calls `compute_reward_batched` on the reward
# module. `sculpt_init` writes the batched template for these; everything
# else gets the scalar template. Update when a new adapter adds batched
# support (see `RewardContract.supports_batched` in adapters/base.py).
_BATCHED_ADAPTER_CLASSES = frozenset({
    "sculptor.adapters.mjlab.MjlabAdapter",
})


def _adapter_needs_batched_template(adapter: str) -> bool:
    """True when the named adapter requires `compute_reward_batched`.

    Accepts either a short name (e.g. "mjlab") or a dotted class path
    (e.g. "sculptor.adapters.mjlab.MjlabAdapter").
    """
    dotted = _ADAPTER_SHORT_NAMES.get(adapter, adapter)
    return dotted in _BATCHED_ADAPTER_CLASSES


_V0_SCALAR_REWARD = '''\
"""v0 — starter reward generated by `sculpt init`. Replace with your domain.

Signature contract:

    compute_reward(state, action, next_state, info)
        -> (reward: float, components: dict[str, float])
"""

from __future__ import annotations


REWARD_SPEC: dict = {
    "version": "v0",
    "parent_hash": "",
    "description": (
        "Starter reward. Returns a constant alive_bonus so the adapter's "
        "preflight validator passes. Edit this file (or let `sculpt run` "
        "edit it) to encode your target behavior."
    ),
    "author": "human",
    "hyperparameters": {
        "alive_bonus": 1.0,
    },
    "references": [],
}


def compute_reward(state, action, next_state, info):
    alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
    components = {"alive_bonus": alive}
    return alive, components
'''

_V0_BATCHED_REWARD = '''\
"""v0 — starter reward generated by `sculpt init` for GPU/batched adapters.

Signature contract (MjlabAdapter and any adapter with
`RewardContract.supports_batched = True`):

    compute_reward_batched(state, action, next_state, info)
        -> (rewards: Tensor[N], components: dict[str, Tensor[N]])

    compute_reward(state, action, next_state, info)
        -> (reward: float, components: dict[str, float])

`compute_reward_batched` is the real training path — called every step
for N parallel envs on the GPU. `compute_reward` is the scalar fallback
used by the adapter's preflight validator and the UI component-probe;
keep it behaviorally consistent with the batched path. Both must exist.
"""

from __future__ import annotations


REWARD_SPEC: dict = {
    "version": "v0",
    "parent_hash": "",
    "description": (
        "Starter reward for a batched/GPU adapter. Returns a constant "
        "alive_bonus per env so the adapter's preflight validator passes. "
        "Edit this file (or let `sculpt run` edit it) to encode your "
        "target behavior."
    ),
    "author": "human",
    "supports_batched": True,
    "hyperparameters": {
        "alive_bonus": 1.0,
    },
    "references": [],
}


def compute_reward(state, action, next_state, info):
    """Scalar path — called by the preflight validator and UI probe."""
    alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
    return alive, {"alive_bonus": alive}


def compute_reward_batched(state, action, next_state, info):
    """Batched path — called per-step on N parallel envs during training.

    All tensors live on the env's device (typically `cuda:0`). Returns
    `(rewards, components)` where rewards has shape `(N,)` and every
    entry in components is a `(N,)` tensor on the same device/dtype.
    """
    import torch

    n = action.shape[0]
    alive = float(REWARD_SPEC["hyperparameters"]["alive_bonus"])
    rewards = torch.full((n,), alive, device=action.device, dtype=action.dtype)
    return rewards, {"alive_bonus": rewards.clone()}
'''


def _v0_template_for(adapter: str) -> str:
    """Return the correct v0.py source for the given adapter."""
    return (
        _V0_BATCHED_REWARD
        if _adapter_needs_batched_template(adapter)
        else _V0_SCALAR_REWARD
    )

_CONFIG_TEMPLATE = '''\
[target]
name = "{name}"

[adapter]
class = "{adapter_dotted}"
config = {{ env_id = "CHANGE_ME", n_envs = 4, ppo_kwargs = {{ learning_rate = 3e-4, n_steps = 2048 }} }}

[kg]
seeds_path = "kg_seeds.yml"
environment_tag = "CHANGE_ME"  # e.g., continuous_locomotion, manipulation, navigation

[iteration]
steps_per_iter = 50000
primary_metric = "mean_return"
behavior_metrics = []  # e.g., ["max_episode_length", "fall_rate"]
rollout_episodes = 6
# §7.4 / §Ship-7: when true, sculpt emits a `physics_edit_suggested`
# event on severe realism audits (§7.3) carrying a ready-to-apply MJCF
# prompt. The Runs-tab chip routes the prompt into the Physics tab for
# one-click application. Defaults to true so new projects get the
# full loop out of the box; flip to false for comparison runs.
auto_adjust_physics = true
# §Ship-9a: early-stop knobs. `early_stop_patience` is the number of
# consecutive iterations with no primary_metric improvement before
# the run truncates; `early_stop_enabled = false` disables the check
# entirely (useful for long overnight runs where metric transients
# can mask real behavioral progress).
early_stop_enabled = true
early_stop_patience = 3
# §Ship-7: rollout video knobs. All optional — leaving them unset makes
# the runner pick real-time playback with a 500-step episode cap.
# max_episode_steps = 500     # env steps per rollout episode
# playback_speed    = 1.0     # 1.0 = real-time; 0.5 = slow-mo; 2.0 = 2x fast
# render_every      = 0       # 0 = auto-cap at 500 captured frames
# rollout_fps       = 0       # 0 = derive from env.step_dt * render_every
# seed              = 42      # base RNG seed; iter i uses seed + i
'''

_KG_SEEDS_TEMPLATE = '''\
# Per-project seed papers for the Sculptor knowledge graph.
# Pick 3-10 papers relevant to YOUR domain. `arxiv_id` is required;
# `title` and `rationale` are human-facing notes.
#
# Run `sculpt kg ingest {path}` to fetch PDFs + extract entities.

papers: []
'''

_GITIGNORE_TEMPLATE = """\
# Secrets
.env
.env.*
!.env.example

# Python / uv
.venv/
__pycache__/
*.py[cod]
.pytest_cache/

# Sculptor local data
kg/graph.db
kg/pdfs/
runs/
reports/metric_history.json

# OS / editor
.DS_Store
Thumbs.db
"""


def _extract_toml_section(toml_text: str, section: str) -> Optional[str]:
    """Return the body of a top-level `[section]` (lines after the
    header up to but not including the next `[other_section]`/
    `[[array_table]]` header). Returns None if the section is
    missing. Plain-text manipulation — sculptor doesn't depend on
    `tomli_w` and the configs we touch are flat enough that string-
    level extraction is safe."""
    lines = toml_text.splitlines(keepends=True)
    start = None
    target = f"[{section}]"
    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        s = lines[j].lstrip()
        if s.startswith("["):  # next section header
            end = j
            break
    body = "".join(lines[start:end])
    if not body.endswith("\n"):
        body += "\n"
    return body


def _replace_toml_section(
    toml_text: str, section: str, new_body: str,
) -> str:
    """Return `toml_text` with the body of `[section]` replaced by
    `new_body`. If the section is missing, returns the input
    unchanged."""
    lines = toml_text.splitlines(keepends=True)
    start = None
    target = f"[{section}]"
    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i + 1
            break
    if start is None:
        return toml_text
    end = len(lines)
    for j in range(start, len(lines)):
        s = lines[j].lstrip()
        if s.startswith("["):
            end = j
            break
    return "".join(lines[:start]) + new_body + "".join(lines[end:])


def _inherit_parent_adapter_config(
    *, stage_dir: Path, project_dir: Path,
) -> bool:
    """Hotfix for a latent Ship 16 bug: `sculpt_init` writes a hard-
    coded gym_sb3-flavored `[adapter].config = { env_id = "CHANGE_ME",
    n_envs = 4, ppo_kwargs = {...} }` regardless of the parent
    project's actual adapter. For mjlab projects, that template
    fails the moment `_run_one_stage` calls `load_adapter` (the keys
    `env_id` / `n_envs` / `ppo_kwargs` are not valid kwargs for
    `MjlabAdapter.__init__`).

    Fix: after `sculpt_init` scaffolds a stage dir, copy the parent
    project's `[adapter]` section (class + config inline-table) over
    the stage's so the stage inherits the correct `task_id` /
    `num_envs` / `device` (mjlab) or whatever the parent set
    (gym_sb3 / others).

    Returns True if the stage's `[adapter]` section changed.
    Tolerates missing project / stage configs by returning False
    without raising — caller's load_adapter will surface the
    original error in that pathological case.
    """
    project_config = project_dir / "config.toml"
    stage_config = stage_dir / "config.toml"
    if not project_config.is_file() or not stage_config.is_file():
        return False
    try:
        parent_text = project_config.read_text(encoding="utf-8")
        stage_text = stage_config.read_text(encoding="utf-8")
    except OSError:
        return False
    parent_adapter_body = _extract_toml_section(parent_text, "adapter")
    if parent_adapter_body is None:
        return False
    new_stage_text = _replace_toml_section(
        stage_text, "adapter", parent_adapter_body,
    )
    # §Ship-19c: ALSO inherit `[iteration]` so the project-level
    # steps_per_iter / primary_metric / early-stop knobs propagate
    # to stages. Without this, sculpt_init's generic template forces
    # 50000 steps_per_iter for every stage, blowing wall-clock on
    # short tasks like Cartpole. Sam's first live run hit this.
    parent_iter_body = _extract_toml_section(parent_text, "iteration")
    if parent_iter_body is not None:
        new_stage_text = _replace_toml_section(
            new_stage_text, "iteration", parent_iter_body,
        )
    if new_stage_text == stage_text:
        return False
    stage_config.write_text(new_stage_text, encoding="utf-8")
    return True


def sculpt_init(project_dir: Path | str, adapter: str) -> Path:
    """Scaffold a new Sculptor project. Returns the created project dir."""
    project_dir = Path(project_dir).resolve()
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(
            f"{project_dir} exists and is not empty — refusing to overwrite.")
    project_dir.mkdir(parents=True, exist_ok=True)

    adapter_dotted = _ADAPTER_SHORT_NAMES.get(adapter, adapter)
    name = project_dir.name

    # config.toml
    (project_dir / "config.toml").write_text(
        _CONFIG_TEMPLATE.format(name=name, adapter_dotted=adapter_dotted),
        encoding="utf-8")

    # rewards/
    rewards_dir = project_dir / "rewards"
    rewards_dir.mkdir()
    (rewards_dir / "__init__.py").write_text("", encoding="utf-8")
    v0 = rewards_dir / "v0.py"
    v0.write_text(_v0_template_for(adapter), encoding="utf-8")
    _write_current_reexport(rewards_dir, v0)

    # kg_seeds.yml
    (project_dir / "kg_seeds.yml").write_text(
        _KG_SEEDS_TEMPLATE.format(path="kg_seeds.yml"), encoding="utf-8")

    # .gitignore
    (project_dir / ".gitignore").write_text(
        _GITIGNORE_TEMPLATE, encoding="utf-8")

    # reports/ + runs/ placeholders (tracked via .gitkeep)
    (project_dir / "reports").mkdir()
    (project_dir / "reports" / ".gitkeep").write_text("", encoding="utf-8")
    (project_dir / "runs").mkdir()
    (project_dir / "runs" / ".gitkeep").write_text("", encoding="utf-8")

    # git init + initial commit (best-effort; no-op if git unavailable)
    if shutil.which("git") is not None:
        try:
            subprocess.run(
                ["git", "-C", str(project_dir), "init", "-q"],
                check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(project_dir), "add", "."],
                check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(project_dir), "commit", "-m", "sculpt init"],
                check=True, capture_output=True)
        except subprocess.CalledProcessError as e:  # pragma: no cover
            sys.stderr.write(
                f"[sculpt init] git init/commit failed: {e.stderr!r}\n")
    return project_dir


# ── Ship 16: Mission orchestrator ────────────────────────────────────
def _is_stage_scaffolded(stage_dir: Path) -> bool:
    """Non-destructive check — returns True iff the dir has the files a
    minimum sculpt project needs. Used by `mission_run` for idempotent
    stage scaffolding (don't re-invoke `sculpt_init` on resume).
    """
    return (
        stage_dir.is_dir()
        and (stage_dir / "config.toml").is_file()
        and (stage_dir / "rewards" / "v0.py").is_file()
    )


def _resolve_stage_final_checkpoint(
    sculpt_result: "SculptRunResult",
) -> Optional[Path]:
    """Pick the checkpoint path from the last completed iter.

    mjlab writes `checkpoint.pt`, gym_sb3 writes `checkpoint.zip` — glob
    for both. Returns None if no iter produced a checkpoint (caller
    marks the stage failed).
    """
    if not sculpt_result.completed_iters:
        return None
    last_iter = sculpt_result.completed_iters[-1]
    for ext in ("pt", "zip"):
        p = last_iter.iter_dir / f"checkpoint.{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _verify_stage_adapter_matches(
    stage_dir: Path, expected_short_name: str,
) -> None:
    """Audit-fix guard for resume: confirm the stage's existing
    config.toml's `[adapter].class` matches `expected_short_name`'s
    dotted path. Raises RuntimeError on mismatch so the orchestrator
    fails the stage clearly rather than silently training under the
    wrong adapter.
    """
    config_path = stage_dir / "config.toml"
    if not config_path.is_file():
        return  # not scaffolded yet; caller will do that

    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"failed to parse {config_path}: {type(e).__name__}: {e}"
        ) from e

    on_disk_dotted = ((cfg.get("adapter") or {}).get("class") or "").strip()
    expected_dotted = _ADAPTER_SHORT_NAMES.get(
        expected_short_name, expected_short_name,
    )
    # Only enforce when the on-disk class is a real Python dotted path
    # ("a.b.c" or longer). Bare names like "stubbed" are test stubs the
    # `load_adapter` factory can't resolve anyway — the test harness
    # monkeypatches `load_adapter` directly. This keeps the check
    # useful (catches "you scaffolded under gym_sb3, now you're trying
    # to run under mjlab") without false-positive on every tmp test.
    if on_disk_dotted and "." in on_disk_dotted and on_disk_dotted != expected_dotted:
        raise RuntimeError(
            f"adapter mismatch in {config_path}: on-disk "
            f"{on_disk_dotted!r} != expected {expected_dotted!r} "
            f"(caller passed adapter_short_name={expected_short_name!r}). "
            f"Mission resume would train under the wrong adapter. "
            f"Either pass the matching adapter_short_name or "
            f"re-scaffold the stage."
        )


def _atomic_save_mission(mission, mission_dir: Path) -> None:
    """Write mission.json via tmp+rename so a SIGKILL mid-save can't
    leave a corrupted JSON Mission.from_json would explode on. Pairs
    with `filelock` below to serialize concurrent writers."""
    import os as _os
    tmp = mission_dir / ".mission.json.tmp"
    tmp.write_text(mission.to_json(), encoding="utf-8")
    _os.replace(tmp, mission_dir / "mission.json")


def mission_run(
    mission,
    *,
    adapter_short_name: str,
    kg_store=None,
    on_event: Optional[Any] = None,
    iterations_override: Optional[int] = None,
    steps_per_iter: Optional[int] = None,
    seed: Optional[int] = None,
    skill_library_handle: Optional[Any] = None,
    early_stop_on_criterion: bool = False,
    criterion_stability_window: int = 1,
    extend_on_improvement: bool = False,
    max_extensions_per_stage: int = 1,
    extension_factor: float = 0.5,
    extension_improvement_threshold: float = 0.05,
):
    """§Ship-19d Goals A + B: optional adaptive iteration control.

    `early_stop_on_criterion` (Goal A) — when True, mission_run wraps
    each stage's sculpt_run with a per-iter callback that evaluates
    the stage's success_criterion on the freshly-completed iter's
    artifacts. If the criterion holds for `criterion_stability_window`
    consecutive iters (default 1; bump to 2 for noisier metrics),
    the stage exits early. Cuts wall-clock dramatically on stages
    that learn faster than the human-author allocated `max_iterations`
    for. Default OFF (preserves Ship 16 behavior of always running
    the full budget then evaluating once at the end).

    `extend_on_improvement` (Goal B) — when True, after a stage's
    sculpt_run completes max_iterations and the criterion failed BUT
    the metric history shows the policy is still improving (best of
    last K iters > best of prior K iters by `extension_improvement_
    threshold` * |prior_best| or fixed +0.05 floor), invoke
    sculpt_run again in resume mode for `extension_factor *
    max_iterations` more iters. Cap total extensions at
    `max_extensions_per_stage` (default 1; max 3 to prevent runaway).
    Default OFF — adaptive extension changes the user's iteration
    contract and should be opt-in.

    Both flags are independent and may be combined: a stage that
    early-stops via Goal A never reaches Goal B's logic; a stage
    that runs to budget without satisfying the criterion enters
    Goal B's trend check (if enabled). The two goals are compatible
    with Ship 17's re-decomposition path — extension fires BEFORE
    re-decomposition (give the policy more iters first; only
    re-decompose if extension also fails).
    """
    """Orchestrate a full mission: decompose → per-stage scaffold → v1
    seeding → sculpt_run → success-criterion eval → advance or halt.

    Parameters
    ----------
    mission : Mission
        Result of `decompose_task`. MUST have `mission_dir` set — save
        the mission first via `save_mission` so subsequent stage dirs
        have a parent.
    adapter_short_name : str
        One of the keys in `_ADAPTER_SHORT_NAMES` (e.g., "mjlab",
        "gym_sb3"). Each stage's `config.toml` is scaffolded with this
        adapter.
    kg_store : SculptorKG | None
        Threaded through to `apply_prompt_edit` for each stage's v1
        materialization, and to every `sculpt_run` call for citation
        grounding.
    on_event : callable | None
        Optional event sink. `mission_run` emits mission-level events
        (mission_started / stage_started / ...) AND the events
        `sculpt_run` already emits for its inner loop — the latter are
        wrapped to add a `stage_name` field so Ship 18's UI can
        correlate per-stage iter events.
    iterations_override / steps_per_iter / seed : override stage.max_iterations
        and pass-through to sculpt_run. `None` honors the stage's value.

    Returns
    -------
    MissionResult
        Completion status + per-stage outcomes. `completed=True` iff
        every stage succeeded. `halted_at_stage` names the first stage
        that failed, if any.

    Raises
    ------
    RuntimeError : mission.mission_dir is None, or the file lock on
        `<mission_dir>/.lock` is already held by another process.
    """
    from filelock import FileLock, Timeout as _FileLockTimeout

    from sculptor.mission import save_mission
    from sculptor.mission_runtime import (
        CriterionEvalError,
        MissionResult,
        StageResult,
        _build_criterion_namespace,
        _evaluate_success_criterion,
    )

    if mission.mission_dir is None:
        raise RuntimeError(
            "mission.mission_dir is None — call save_mission(mission, path) "
            "before mission_run so stage dirs resolve."
        )
    mission_dir = Path(mission.mission_dir).resolve()
    mission_dir.mkdir(parents=True, exist_ok=True)
    (mission_dir / "stages").mkdir(exist_ok=True)

    def _emit(payload: dict) -> None:
        """Emit via the public module-level _emit_event AND the caller's
        optional on_event sink, so tests / UI hooks can observe without
        parsing stdout."""
        _emit_event(payload)
        if on_event is not None:
            try:
                on_event(payload)
            except Exception:  # noqa: BLE001 — never let user callback crash run
                pass

    result = MissionResult(mission_goal=mission.goal)

    lock_path = mission_dir / ".lock"
    # Audit fix: 10s timeout (was 1s) — slow filesystems (NFS / WSL
    # interop) can legitimately take >1s to acquire. Stale lock
    # files are NOT a problem with `filelock` because the OS-level
    # advisory lock isn't held when the prior process exits, even
    # if the file persists on disk.
    lock = FileLock(str(lock_path), timeout=10.0)
    try:
        lock.acquire()
    except _FileLockTimeout as e:
        raise RuntimeError(
            f"another mission_run is holding the lock at {lock_path}. "
            f"Concurrent runs on the same mission would corrupt "
            f"mission.json and stage artifacts."
        ) from e

    try:
        _emit({
            "type": "mission_started",
            "mission_dir": str(mission_dir),
            "goal": mission.goal,
            "n_stages": len(mission.stages),
        })

        # §Ship 17: refactored from for-loop to while-loop indexed by
        # `mission.current_stage_idx` so mid-iteration splices (sub-
        # stage redecomposition replacing the failed stage in place)
        # are safe. The for-loop's iterator would have cached the old
        # list state and skipped past the inserted sub-stages.
        all_succeeded = True
        while mission.current_stage_idx < len(mission.stages):
            stage_idx = mission.current_stage_idx
            stage = mission.stages[stage_idx]

            # Resume: skip stages that already succeeded on disk.
            if stage.status == "succeeded":
                _emit({
                    "type": "stage_skipped",
                    "stage_name": stage.name,
                    "reason": "already_succeeded",
                })
                result.stage_results.append(StageResult(
                    stage_name=stage.name, status="succeeded",
                    iterations_used=stage.iterations_used,
                    final_policy_path=stage.final_policy_path,
                    final_reward_path=stage.final_reward_path,
                    criterion_satisfied=True,
                    last_iter_metric=stage.best_metric,
                ))
                mission.current_stage_idx += 1
                continue

            stage_res = _run_one_stage(
                mission=mission,
                mission_dir=mission_dir,
                stage=stage,
                stage_idx=stage_idx,
                adapter_short_name=adapter_short_name,
                kg_store=kg_store,
                emit=_emit,
                iterations_override=iterations_override,
                steps_per_iter=steps_per_iter,
                seed=seed,
                skill_library_handle=skill_library_handle,
                early_stop_on_criterion=early_stop_on_criterion,
                criterion_stability_window=criterion_stability_window,
                extend_on_improvement=extend_on_improvement,
                max_extensions_per_stage=max_extensions_per_stage,
                extension_factor=extension_factor,
                extension_improvement_threshold=(
                    extension_improvement_threshold
                ),
            )
            result.stage_results.append(stage_res)

            # Persist mission.json AFTER every stage transition so
            # resume sees the latest state.
            _atomic_save_mission(mission, mission_dir)

            if stage_res.status == "succeeded":
                mission.current_stage_idx += 1
                continue

            # Failure path. Try Ship 17 redecomposition before halting.
            spliced = _maybe_redecompose_and_splice(
                mission=mission,
                mission_dir=mission_dir,
                failed_stage_idx=stage_idx,
                stage_res=stage_res,
                kg_store=kg_store,
                emit=_emit,
            )
            if spliced:
                # current_stage_idx was rewound by the splice helper to
                # point at the first new sub-stage. Drop the failed
                # stage's StageResult — it'll be replaced when the
                # sub-stages run. Loop continues without advancing.
                result.stage_results.pop()
                continue

            all_succeeded = False
            result.halted_at_stage = stage.name
            result.halted_reason = (
                stage_res.failure_reason or "criterion_not_met"
            )
            _emit({
                "type": "mission_halted",
                "stage_name": stage.name,
                "reason": result.halted_reason,
            })
            break

        if all_succeeded and mission.current_stage_idx >= len(mission.stages):
            result.completed = True

        _emit({
            "type": "mission_completed" if result.completed
                    else "mission_halted_terminal",
            "completed": result.completed,
            "halted_at_stage": result.halted_at_stage,
            "halted_reason": result.halted_reason,
        })
    except KeyboardInterrupt:
        result.halted_reason = "interrupted"
        _emit({
            "type": "mission_halted",
            "reason": "interrupted",
        })
        raise
    finally:
        lock.release()
        # Audit fix: do NOT unlink the .lock file. On Windows / WSL,
        # `filelock` can hold the file handle past `release()`, making
        # the unlink fail silently and confusing future debuggers
        # ("why is .lock here if no one's holding it?"). The OS-level
        # advisory lock is released cleanly; the file persisting is
        # cosmetic and `FileLock` will re-acquire it on the next run.

    return result


def _run_one_stage(
    *,
    mission,
    mission_dir: Path,
    stage,
    stage_idx: int,
    adapter_short_name: str,
    kg_store,
    emit: Any,
    iterations_override: Optional[int],
    steps_per_iter: Optional[int],
    seed: Optional[int],
    skill_library_handle: Optional[Any] = None,
    early_stop_on_criterion: bool = False,
    criterion_stability_window: int = 1,
    extend_on_improvement: bool = False,
    max_extensions_per_stage: int = 1,
    extension_factor: float = 0.5,
    extension_improvement_threshold: float = 0.05,
):
    """Helper called by `mission_run` per stage. Kept separate so the
    orchestrator stays readable — `mission_run` is about flow; this
    function is about "one stage, cradle to grave." """
    from sculptor.mission_runtime import (
        CriterionEvalError,
        StageResult,
        _build_criterion_namespace,
        _evaluate_success_criterion,
    )

    stage_dir = mission.stage_dir(stage.name)
    # §Ship 20 Goal #2: compute the effective max iterations BEFORE
    # emitting stage_started so the UI can render `iters X/effectiveY`
    # accurately when the user passed `iterations_override`. Without
    # this the dialog reads `stage.max_iterations` (Claude's authored
    # budget) and shows nonsense like `iters 2/3` when the run was
    # capped at 2. Don't mutate stage.max_iterations — that's the
    # persisted authored value; effective_max_iterations is runtime.
    #
    # Semantics note: this is the BASELINE cap for the first sculpt_run
    # call. Goal B (extend_on_improvement) extensions issue separate
    # `stage_extended` events with their own additional_iters; they do
    # NOT update this value. The UI surfaces extensions as their own
    # chips — the cap shown on the stage card is what the user
    # explicitly configured. If the actual `iterations_used` exceeds
    # the cap, that's an explicit Goal B opt-in; the user expects it.
    effective_max_iterations = iterations_override or stage.max_iterations
    emit({
        "type": "stage_started",
        "stage_name": stage.name,
        "stage_index": stage_idx,
        "stage_dir": str(stage_dir),  # audit fix: UI symmetry with stage_scaffolded
        "goal_text": stage.goal_text,
        "parent_stage": stage.parent_stage,
        "max_iterations": stage.max_iterations,
        "effective_max_iterations": effective_max_iterations,
    })
    stage.status = "training"
    stage.started_at = _utc_now_iso()

    # 1. Resolve parent checkpoint (None for first stage / parent-not-trained).
    # Audit fix: distinguish "no parent / parent untrained" from "parent
    # ckpt was deleted externally" — the latter silently degrades to
    # cold-start without this branch, which would invalidate the whole
    # curriculum's warm-start chain.
    parent_ckpt, parent_status = mission.parent_checkpoint_status_of(stage.name)
    emit({
        "type": "stage_warm_start_resolved",
        "stage_name": stage.name,
        "stage_index": stage_idx,
        "parent_stage": stage.parent_stage,
        "parent_checkpoint": str(parent_ckpt) if parent_ckpt else None,
        "parent_status": parent_status,
    })
    if parent_status == "parent_ckpt_missing":
        emit({
            "type": "warm_start_skipped",
            "stage_name": stage.name,
            "reason": "parent_ckpt_missing",
            "detail": (
                f"parent stage {stage.parent_stage!r} recorded "
                f"final_policy_path but the file is gone — child stage "
                f"will train cold-start, defeating the curriculum's "
                f"warm-start chain. Restore the file or re-run the "
                f"parent stage."
            ),
        })

    # §Ship 19: resolve cross-mission skill warm-start (if any).
    # Decision: when `stage.init_skill_id` is explicitly set by the
    # decomposer AND the handle resolves it to a real checkpoint,
    # the SKILL wins over the parent ckpt — "explicit beats implicit"
    # (audit fix C1: CurricuLLM's premise is that prior-mission
    # specialized policies often beat re-using the current mission's
    # parent which trained on a different reward shape). When skill
    # resolution fails / is absent, fall back to parent_ckpt as
    # before — Ship 16 behavior preserved.
    skill_ckpt: Optional[Path] = None
    if skill_library_handle is not None and stage.init_skill_id:
        skill_ckpt = skill_library_handle.maybe_load_for_stage(stage, emit)

    if skill_ckpt is not None:
        warm_start_path: Optional[Path] = skill_ckpt
        warm_start_source = "skill_library"
        warm_start_source_id: Optional[str] = stage.init_skill_id
        if parent_ckpt is not None:
            emit({
                "type": "warm_start_skipped",
                "stage_name": stage.name,
                "reason": "skill_overrides_parent",
                "skill_id": stage.init_skill_id,
                "parent_stage": stage.parent_stage,
            })
    elif parent_ckpt is not None:
        warm_start_path = parent_ckpt
        warm_start_source = "parent_stage"
        warm_start_source_id = stage.parent_stage
    else:
        warm_start_path = None
        warm_start_source = "none"
        warm_start_source_id = None

    emit({
        "type": "stage_warm_start_chosen",
        "stage_name": stage.name,
        "stage_index": stage_idx,
        "source": warm_start_source,
        "source_id": warm_start_source_id,
        "checkpoint": str(warm_start_path) if warm_start_path else None,
    })

    # 2. Scaffold stage dir idempotently. `stage_dir` was resolved
    # above for the stage_started event (audit fix: UI symmetry).
    if not _is_stage_scaffolded(stage_dir):
        try:
            sculpt_init(stage_dir, adapter_short_name)
        except Exception as e:  # noqa: BLE001
            return _fail_stage(
                stage, "scaffold_errored",
                f"{type(e).__name__}: {e}", emit,
            )
        # Hotfix for Ship 16 latent bug exposed by Ship 18b/19's first
        # real mjlab mission run: `sculpt_init` writes a gym_sb3-
        # flavored adapter config (env_id / n_envs / ppo_kwargs)
        # regardless of the parent project's actual adapter. For
        # mjlab the keys are simply wrong and the next load_adapter
        # call raises TypeError. Copy the parent's [adapter] section
        # over the freshly-scaffolded stage's config so it inherits
        # the right task_id / num_envs / device.
        try:
            project_root = mission_dir.parent.parent
            inherited = _inherit_parent_adapter_config(
                stage_dir=stage_dir, project_dir=project_root,
            )
        except Exception as e:  # noqa: BLE001 — best-effort; surfaced via stage_scaffolded payload
            inherited = False
            emit({
                "type": "stage_scaffold_inherit_warning",
                "stage_name": stage.name,
                "error": f"{type(e).__name__}: {e}",
            })
        emit({
            "type": "stage_scaffolded",
            "stage_name": stage.name,
            "stage_dir": str(stage_dir),
            "inherited_parent_adapter_config": bool(inherited),
        })
    else:
        # Audit fix: detect adapter mismatch on resume. If the stage
        # was scaffolded under one adapter and the caller now passes
        # another, sculpt_run would silently use the on-disk one.
        try:
            _verify_stage_adapter_matches(stage_dir, adapter_short_name)
        except RuntimeError as e:
            return _fail_stage(
                stage, "adapter_mismatch",
                f"{type(e).__name__}: {e}", emit,
            )

    # 3. Materialize v1 from the stage's reward_seed_prompt.
    try:
        from sculptor.edit import apply_prompt_edit
        latest_n, latest_reward_file = _find_latest_reward_version(
            stage_dir / "rewards",
        )
        # Only materialize v1 if we haven't already (resume case).
        if latest_n == 0:
            from sculptor.adapters.base import load_adapter
            adapter_for_contract = load_adapter(stage_dir / "config.toml")
            apply_prompt_edit(
                current_reward_path=latest_reward_file,
                user_prompt=stage.reward_seed_prompt,
                new_iter_id=f"v{latest_n + 1}",
                reward_contract=adapter_for_contract.reward_contract(),
                kg_store=kg_store,
            )
            emit({
                "type": "stage_v1_materialized",
                "stage_name": stage.name,
                "reward_path": str(stage_dir / "rewards" / "v1.py"),
            })
    except Exception as e:  # noqa: BLE001
        return _fail_stage(
            stage, "v1_materialization_errored",
            f"apply_prompt_edit failed: {type(e).__name__}: {e}",
            emit,
        )

    # 4. Run the per-stage training loop via existing sculpt_run.
    # §Ship 19: warm_start_path = skill (if explicitly chosen by the
    # decomposer) OR parent_ckpt (Ship 16 default) OR None (cold).
    # §Ship-19d: Goal A wraps each iter with a criterion-eval
    # callback; Goal B runs additional sculpt_run passes (resume
    # mode) when metric is still improving at end-of-budget.
    from sculptor.mission_runtime import (
        CriterionEvalError,
        _build_criterion_namespace,
        _evaluate_success_criterion,
    )

    # §Ship 20 Goal #2: re-use the value already computed for
    # stage_started's payload above. `max_iters` is the authoritative
    # iterations cap passed to sculpt_run; `effective_max_iterations`
    # is the same number, surfaced via WS so the UI can label
    # accurately.
    max_iters = effective_max_iterations

    per_iter_cb: Optional[Callable[[Any], Optional[str]]] = None
    if early_stop_on_criterion:
        # Closure-local consecutive-pass counter; bumps on each iter
        # whose artifacts satisfy the stage's criterion. Resets the
        # moment any iter fails (so a noisy spike doesn't trigger).
        # When the count hits `criterion_stability_window`, return a
        # stop-reason; sculpt_run records the iter as completed and
        # breaks its loop.
        _consecutive_passes = {"n": 0}
        _stage_criterion = stage.success_criterion

        def _criterion_callback(outcome: Any) -> Optional[str]:
            try:
                namespace = _build_criterion_namespace(
                    iter_dir=Path(outcome.iter_dir),
                    primary_metric=outcome.primary_metric,
                )
                ok = _evaluate_success_criterion(
                    _stage_criterion, namespace,
                )
            except (CriterionEvalError, Exception):  # noqa: BLE001
                # A criterion that errors during the stream is
                # inconclusive — DO NOT short-circuit. The
                # post-run eval at step 6 will surface the error
                # via `_fail_stage` if it persists at the last iter.
                _consecutive_passes["n"] = 0
                return None
            if ok:
                _consecutive_passes["n"] += 1
                if (
                    _consecutive_passes["n"]
                    >= max(1, criterion_stability_window)
                ):
                    return (
                        f"criterion_satisfied at iter "
                        f"{outcome.iter_index} "
                        f"(stability_window="
                        f"{criterion_stability_window})"
                    )
            else:
                _consecutive_passes["n"] = 0
            return None

        per_iter_cb = _criterion_callback

    extensions_used = 0
    sculpt_result: Any = None
    try:
        sculpt_result = sculpt_run(
            config_path=stage_dir / "config.toml",
            behavior_goal=stage.goal_text,
            iterations=max_iters,
            steps_per_iter=steps_per_iter,
            seed=seed,
            init_policy_path=warm_start_path,
            per_iter_callback=per_iter_cb,
        )
    except KeyboardInterrupt:
        stage.status = "failed"
        stage.finished_at = _utc_now_iso()
        raise
    except Exception as e:  # noqa: BLE001
        return _fail_stage(
            stage, "training_errored",
            f"sculpt_run raised: {type(e).__name__}: {e}",
            emit,
        )

    # §Ship-19d Goal B: extension loop. Only entered when (a) the
    # caller opted in, (b) Goal A's callback did NOT fire (criterion
    # not yet satisfied), (c) the metric trend looks promising, and
    # (d) we haven't exhausted the extension budget. Each pass
    # resumes from the previous run's last checkpoint.
    def _criterion_satisfied_now(sr: Any) -> bool:
        completed = list(getattr(sr, "completed_iters", []) or [])
        if not completed:
            return False
        last = completed[-1]
        try:
            ns = _build_criterion_namespace(
                iter_dir=Path(last.iter_dir),
                primary_metric=last.primary_metric,
            )
            return bool(_evaluate_success_criterion(
                stage.success_criterion, ns,
            ))
        except Exception:  # noqa: BLE001
            return False

    while extend_on_improvement:
        # Goal A's per-iter callback already short-circuited.
        if (
            sculpt_result is not None
            and sculpt_result.early_stopped
            and "criterion_satisfied" in (
                sculpt_result.early_stop_reason or ""
            )
        ):
            break
        # Already passing? No need to extend.
        if _criterion_satisfied_now(sculpt_result):
            break
        # Patience-based metric-plateau early-stop fired — the metric
        # ISN'T improving by definition. Don't extend; that's exactly
        # the case Goal B should NOT fire on.
        if (
            sculpt_result is not None
            and sculpt_result.early_stopped
        ):
            emit({
                "type": "stage_extension_skipped",
                "stage_name": stage.name,
                "reason": "metric_plateau_early_stop",
            })
            break
        if extensions_used >= max(0, max_extensions_per_stage):
            emit({
                "type": "stage_extension_exhausted",
                "stage_name": stage.name,
                "extensions_used": extensions_used,
                "max_extensions_per_stage": max_extensions_per_stage,
            })
            break
        history = list(
            getattr(sculpt_result, "primary_metric_history", []) or [],
        )
        if not _is_metric_still_improving(
            history, threshold=extension_improvement_threshold,
        ):
            emit({
                "type": "stage_extension_skipped",
                "stage_name": stage.name,
                "reason": "no_improvement_trend",
                "history_len": len(history),
            })
            break
        extra_iters = max(2, int(round(max_iters * extension_factor)))
        extensions_used += 1
        emit({
            "type": "stage_extended",
            "stage_name": stage.name,
            "extension_count": extensions_used,
            "additional_iters": extra_iters,
            "reason": "metric_still_improving",
        })
        try:
            sculpt_result = sculpt_run(
                config_path=stage_dir / "config.toml",
                behavior_goal=stage.goal_text,
                iterations=extra_iters,
                steps_per_iter=steps_per_iter,
                seed=seed,
                init_policy_path=None,  # resume picks up from local ckpt
                resume=True,
                per_iter_callback=per_iter_cb,
            )
        except KeyboardInterrupt:
            stage.status = "failed"
            stage.finished_at = _utc_now_iso()
            raise
        except Exception as e:  # noqa: BLE001
            return _fail_stage(
                stage, "extension_errored",
                f"sculpt_run extension raised: "
                f"{type(e).__name__}: {e}",
                emit,
            )

    stage.iterations_used = sculpt_result.iterations_run
    emit({
        "type": "stage_completed_training",
        "stage_name": stage.name,
        "iterations_run": sculpt_result.iterations_run,
        "early_stopped": sculpt_result.early_stopped,
        # §Ship 20 Goal #2: surface the cap that was actually enforced
        # so the UI can finalize `iters X/effectiveY` post-run, even
        # if the stage_started event fell off the WS event window.
        "effective_max_iterations": effective_max_iterations,
    })

    # 5. Derive final_policy_path from the last iter's checkpoint.
    final_ckpt = _resolve_stage_final_checkpoint(sculpt_result)
    if final_ckpt is None:
        return _fail_stage(
            stage, "no_checkpoint",
            "training completed but no checkpoint.pt / .zip was "
            "produced in the last iter — successor stages can't "
            "warm-start from a missing file.",
            emit,
        )
    stage.final_policy_path = str(final_ckpt)
    stage.final_reward_path = (
        str(sculpt_result.final_reward_path)
        if sculpt_result.final_reward_path else None
    )

    # 6. Evaluate success criterion on the last iter's namespace.
    last_iter = sculpt_result.completed_iters[-1]
    stage.best_metric = last_iter.primary_metric
    try:
        namespace = _build_criterion_namespace(
            iter_dir=last_iter.iter_dir,
            primary_metric=last_iter.primary_metric,
        )
        criterion_ok = _evaluate_success_criterion(
            stage.success_criterion, namespace,
        )
    except CriterionEvalError as e:
        emit({
            "type": "stage_criterion_evaluated",
            "stage_name": stage.name,
            "satisfied": False,
            "error": str(e),
        })
        return _fail_stage(
            stage, "criterion_errored", str(e), emit,
            criterion_error=str(e),
        )

    emit({
        "type": "stage_criterion_evaluated",
        "stage_name": stage.name,
        "criterion": stage.success_criterion,
        "satisfied": bool(criterion_ok),
        "last_iter_metric": last_iter.primary_metric,
    })

    if criterion_ok:
        stage.status = "succeeded"
        stage.finished_at = _utc_now_iso()
        emit({
            "type": "stage_succeeded",
            "stage_name": stage.name,
            "iterations_used": stage.iterations_used,
            "final_policy_path": stage.final_policy_path,
            "last_iter_metric": stage.best_metric,
        })
        # §Ship 19: publish a skill record to the cross-mission library
        # AFTER stage success. Gates (stage status, redecomposition
        # attempts, adapter capability, history non-empty) are
        # enforced inside `maybe_publish` so the per-skip reason is
        # observable. Re-loading the adapter here is cheap (already
        # done at v1 materialization above) and lets us pass it for
        # the warm-start-support introspection check. Library / IO
        # errors emit `stage_skill_publish_skipped` and DO NOT break
        # the stage's success outcome.
        if skill_library_handle is not None:
            try:
                from sculptor.adapters.base import load_adapter
                _adapter_for_publish = load_adapter(stage_dir / "config.toml")
                skill_library_handle.maybe_publish(
                    stage=stage,
                    mission=mission,
                    adapter=_adapter_for_publish,
                    sculpt_result=sculpt_result,
                    emit=emit,
                )
            except Exception as e:  # noqa: BLE001
                emit({
                    "type": "stage_skill_publish_skipped",
                    "stage_name": stage.name,
                    "reason": "publish_call_errored",
                    "error": f"{type(e).__name__}: {e}",
                })
        return StageResult(
            stage_name=stage.name, status="succeeded",
            iterations_used=stage.iterations_used,
            final_policy_path=stage.final_policy_path,
            final_reward_path=stage.final_reward_path,
            criterion_satisfied=True,
            last_iter_metric=stage.best_metric,
        )

    return _fail_stage(
        stage, "criterion_not_met",
        f"success_criterion {stage.success_criterion!r} evaluated False "
        f"on last iter (metric={last_iter.primary_metric}).",
        emit,
    )


def _fail_stage(
    stage, reason: str, detail: str, emit,
    *,
    criterion_error: Optional[str] = None,
):
    """Shared failure path used by every early-return in `_run_one_stage`."""
    from sculptor.mission_runtime import StageResult

    stage.status = "failed"
    stage.finished_at = _utc_now_iso()
    emit({
        "type": "stage_failed",
        "stage_name": stage.name,
        "reason": reason,
        "detail": detail,
    })
    return StageResult(
        stage_name=stage.name, status="failed",
        iterations_used=stage.iterations_used,
        final_policy_path=stage.final_policy_path,
        final_reward_path=stage.final_reward_path,
        criterion_satisfied=False,
        criterion_error=criterion_error,
        last_iter_metric=stage.best_metric,
        failure_reason=reason,
    )


def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ── Ship 17: redecomposition splice ─────────────────────────────────
# Only criterion failures are re-decomposable. Infrastructure-class
# failures (training_errored, no_checkpoint, adapter_mismatch,
# scaffold_errored, v1_materialization_errored) signal env / code
# issues that re-decomposition can't fix.
_REDECOMPOSABLE_REASONS: frozenset[str] = frozenset({"criterion_not_met"})


def _build_stage_training_feedback(
    stage,
    stage_res,
    sculpt_result,
):
    """Pack the diagnostic info Claude needs to design a re-decomposition.

    Reads from disk:
      - the final reward module's source (verbatim Python).
      - the last iter's diagnosis.json.
      - the last 3 iters' component means (re-running
        `_load_trajectory_arrays` for each).

    All best-effort — partial reads degrade to empty values rather
    than crash the redecomposition flow.
    """
    from sculptor.decompose import StageTrainingFeedback
    from sculptor.mission_runtime import (
        _build_criterion_namespace,
        _load_trajectory_arrays,
    )

    # Final reward source.
    final_reward_source = ""
    if stage_res.final_reward_path:
        try:
            final_reward_source = Path(
                stage_res.final_reward_path
            ).read_text(encoding="utf-8")
        except OSError:
            pass

    # Last iter's diagnosis + namespace.
    last_iter = (
        sculpt_result.completed_iters[-1]
        if sculpt_result and sculpt_result.completed_iters else None
    )
    last_iter_diagnosis: dict = {}
    last_iter_namespace: dict = {
        "behavior": {}, "components": {}, "metric": stage_res.last_iter_metric,
    }
    if last_iter is not None:
        diag_path = Path(last_iter.iter_dir) / "diagnosis.json"
        if diag_path.is_file():
            try:
                last_iter_diagnosis = json.loads(
                    diag_path.read_text(encoding="utf-8"),
                )
            except Exception:  # noqa: BLE001
                last_iter_diagnosis = {}
        try:
            ns = _build_criterion_namespace(
                Path(last_iter.iter_dir),
                primary_metric=last_iter.primary_metric,
            )
            last_iter_namespace = {
                "behavior": ns.get("behavior", {}),
                "components": ns.get("components", {}),
                "metric": ns.get("metric"),
            }
        except Exception:  # noqa: BLE001
            pass

    # Last-3-iter components (one dict per iter).
    last_3_components: list[dict] = []
    if sculpt_result and sculpt_result.completed_iters:
        for outcome in sculpt_result.completed_iters[-3:]:
            try:
                _, comps = _load_trajectory_arrays(Path(outcome.iter_dir))
                last_3_components.append({
                    "iter": outcome.iter_index,
                    "components": comps,
                })
            except Exception:  # noqa: BLE001
                last_3_components.append({
                    "iter": outcome.iter_index, "components": {},
                })

    metric_history = (
        list(sculpt_result.primary_metric_history)
        if sculpt_result and sculpt_result.primary_metric_history else []
    )

    return StageTrainingFeedback(
        final_reward_source=final_reward_source,
        last_iter_diagnosis=last_iter_diagnosis,
        last_iter_namespace=last_iter_namespace,
        metric_history=metric_history,
        last_3_iter_components=last_3_components,
        failure_reason=stage_res.failure_reason or "criterion_not_met",
        criterion_error=stage_res.criterion_error,
    )


def _repoint_downstream_children(
    stages: list, old_parent_name: str, new_parent_name: str,
    *, slice_start: int,
) -> int:
    """Walk `stages[slice_start:]` and rewrite any `parent_stage ==
    old_parent_name` → `new_parent_name`. Returns the count of
    re-pointings done. Critical reviewer-flagged step: without this,
    `validate_mission` raises on the spliced graph because the failed
    stage's name no longer exists.
    """
    n = 0
    for s in stages[slice_start:]:
        if s.parent_stage == old_parent_name:
            s.parent_stage = new_parent_name
            n += 1
    return n


def _maybe_redecompose_and_splice(
    *,
    mission,
    mission_dir: Path,
    failed_stage_idx: int,
    stage_res,
    kg_store,
    emit,
) -> bool:
    """Try to re-decompose the failed stage into 2-8 sub-stages and
    splice them into `mission.stages`. Returns True on success, False
    if redecomposition was skipped or failed.

    Side effects on success:
      * `mission.stages[failed_stage_idx:failed_stage_idx+1]` replaced
        by sub-stages.
      * Downstream children's `parent_stage` references re-pointed
        from `failed.name` to the LAST sub-stage's name.
      * `mission.current_stage_idx` set to `failed_stage_idx` so the
        while-loop processes the first sub-stage next.
      * mission.json persisted atomically.
      * `stage_redecomposed` event emitted.

    On failure: emits `stage_redecomposition_failed` and returns False.
    Caller halts the mission cleanly (no Claude retry — same envelope
    as decompose_task).
    """
    from sculptor.decompose import (
        DecompositionError,
        StageTrainingFeedback,
        redecompose_stage,
    )
    from sculptor.mission import MissionValidationError, validate_mission

    failed_stage = mission.stages[failed_stage_idx]

    # Trigger conditions: only criterion failures, and only if the
    # stage hasn't been re-decomposed before.
    if failed_stage.redecomposition_attempts >= 1:
        emit({
            "type": "redecomposition_skipped",
            "stage_name": failed_stage.name,
            "reason": "budget_exhausted",
            "detail": (
                f"stage already used its redecomposition budget "
                f"({failed_stage.redecomposition_attempts}); "
                "halting per Ship 17's one-level cap."
            ),
        })
        return False
    reason = stage_res.failure_reason or ""
    if reason not in _REDECOMPOSABLE_REASONS:
        emit({
            "type": "redecomposition_skipped",
            "stage_name": failed_stage.name,
            "reason": "non_curriculum_failure",
            "detail": (
                f"failure_reason={reason!r} signals an env/code issue "
                f"(not a curriculum mismatch). Re-decomposition won't "
                f"help; halting."
            ),
        })
        return False

    # Build training feedback.
    # The stage's training was wrapped by `_run_one_stage`; we don't
    # have the SculptRunResult in scope here — `stage_res` is the
    # post-train summary. Best-effort reconstruct from on-disk state.
    feedback = _build_stage_training_feedback(
        failed_stage, stage_res, sculpt_result=None,
    )
    # The reconstruct from `stage_res` alone misses metric_history; the
    # caller path doesn't currently thread `sculpt_result` here, so
    # we pull the history from the stage's iter dirs as a fallback.
    if not feedback.metric_history:
        feedback.metric_history = _scan_iter_metric_history(
            mission, failed_stage,
        )

    emit({
        "type": "stage_redecomposition_started",
        "stage_name": failed_stage.name,
        "stage_index": failed_stage_idx,
        "trigger_reason": reason,
    })

    # Audit-fix (Ship 17 review, finding #E): surface partial-feedback
    # state so the user knows when Claude saw incomplete training
    # context (vs. seeing complete context with empty signals).
    missing_signals: list[str] = []
    if not feedback.final_reward_source:
        missing_signals.append("final_reward_source")
    if not feedback.last_iter_diagnosis:
        missing_signals.append("last_iter_diagnosis")
    if not feedback.metric_history:
        missing_signals.append("metric_history")
    if not feedback.last_3_iter_components:
        missing_signals.append("last_3_iter_components")
    if missing_signals:
        emit({
            "type": "feedback_read_degraded",
            "stage_name": failed_stage.name,
            "missing_signals": missing_signals,
            "detail": (
                f"Claude will see partial training feedback for this "
                f"redecomposition. Missing: {missing_signals}. "
                f"Result quality may degrade."
            ),
        })

    # Resolve adapter contract from the stage's config.toml.
    try:
        from sculptor.adapters.base import load_adapter
        stage_dir = mission.stage_dir(failed_stage.name)
        adapter = load_adapter(stage_dir / "config.toml")
        reward_contract = adapter.reward_contract()
    except Exception as e:  # noqa: BLE001
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "adapter_load_failed",
            "detail": f"{type(e).__name__}: {e}",
        })
        return False

    # Call Claude.
    try:
        sub_stages = redecompose_stage(
            mission, failed_stage_idx,
            feedback=feedback,
            reward_contract=reward_contract,
            kg_store=kg_store,
        )
    except (MissionValidationError, DecompositionError) as e:
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "validation_failed",
            "detail": f"{type(e).__name__}: {e}",
        })
        return False
    except Exception as e:  # noqa: BLE001
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "claude_call_errored",
            "detail": f"{type(e).__name__}: {e}",
        })
        return False

    if not sub_stages:
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "empty_substages",
        })
        return False

    last_sub_name = sub_stages[-1].name

    # Splice: replace the failed stage with sub-stages, repoint downstream.
    mission.stages[failed_stage_idx:failed_stage_idx + 1] = sub_stages
    repointed = _repoint_downstream_children(
        mission.stages, failed_stage.name, last_sub_name,
        slice_start=failed_stage_idx + len(sub_stages),
    )

    # Validate the spliced mission. If this raises, restore the failed
    # stage and halt. Do NOT leave the mission in a half-spliced state.
    info_keys = set(getattr(reward_contract, "expected_info_keys", None) or [])
    try:
        validate_mission(mission, info_keys=info_keys)
    except MissionValidationError as e:
        # Roll back the splice.
        mission.stages[failed_stage_idx:failed_stage_idx + len(sub_stages)] = [failed_stage]
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "spliced_mission_invalid",
            "detail": f"{type(e).__name__}: {e}",
        })
        return False

    # Reviewer-flagged: persist current_stage_idx BEFORE the splice so
    # a crash-resume starts at the first new sub-stage rather than
    # skipping ahead.
    mission.current_stage_idx = failed_stage_idx

    # Audit-fix (Ship 17 review, finding #A): if `_atomic_save_mission`
    # fails (disk full / permissions / EIO), the in-memory mission has
    # the new sub-stages but on-disk still shows the old failed stage.
    # On resume we'd re-train the failed stage and re-call Claude for
    # a (possibly different) re-decomposition — divergence from user
    # expectations. Rollback the in-memory splice on save failure so
    # in-memory and on-disk stay consistent, and emit a clear event.
    try:
        _atomic_save_mission(mission, mission_dir)
    except OSError as e:
        # Roll back in-memory splice + parent re-pointing.
        mission.stages[failed_stage_idx:failed_stage_idx + len(sub_stages)] = [failed_stage]
        # Restore downstream children's parent_stage to the failed name.
        for s in mission.stages[failed_stage_idx + 1:]:
            if s.parent_stage == last_sub_name:
                s.parent_stage = failed_stage.name
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": "save_failed",
            "detail": (
                f"splice succeeded in-memory but persisting "
                f"mission.json failed ({type(e).__name__}: {e}); "
                f"rolled back to old state. Free disk / fix "
                f"permissions and re-run."
            ),
        })
        return False

    emit({
        "type": "stage_redecomposed",
        "original_stage_name": failed_stage.name,
        "stage_index": failed_stage_idx,
        "sub_stage_names": [s.name for s in sub_stages],
        "downstream_children_repointed": repointed,
    })
    return True


def _scan_iter_metric_history(mission, stage) -> list[float]:
    """Best-effort: read each `runs/iter_N/metrics.json` (or rollout/
    behavior.json if metrics is missing) for the stage's project dir
    and return the per-iter primary_metric series. Used by the
    redecomposition feedback when SculptRunResult isn't in scope."""
    history: list[float] = []
    try:
        stage_dir = mission.stage_dir(stage.name)
    except Exception:  # noqa: BLE001
        return history
    runs_dir = stage_dir / "runs"
    if not runs_dir.is_dir():
        return history
    # Audit-fix (Ship 17 review, finding #B): sort by numeric suffix.
    # Use `+inf` as the fallback for malformed names (e.g.,
    # `iter_10_backup`) so corrupted dirs sort to the END and don't
    # shadow legitimate iter_0..iter_N entries. Pre-fix `else -1`
    # would have ranked them BEFORE iter_0, polluting the metric
    # history Claude sees.
    def _iter_sort_key(p: Path) -> float:
        suffix = p.name.split("_")[-1]
        return int(suffix) if suffix.isdigit() else float("inf")

    iter_dirs = sorted(
        (p for p in runs_dir.iterdir()
         if p.is_dir() and p.name.startswith("iter_")),
        key=_iter_sort_key,
    )
    for d in iter_dirs:
        for path in (d / "rollout" / "behavior.json", d / "behavior.json"):
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    val = payload.get("mean_return")
                    if isinstance(val, (int, float)):
                        history.append(float(val))
                        break
                except Exception:  # noqa: BLE001
                    pass
    return history


def regenerate_reward_template(project_dir: Path | str) -> Path:
    """Rewrite `<project>/rewards/v0.py` to match the adapter in config.toml.

    Used to recover a project whose `v0.py` was scaffolded under the wrong
    adapter contract — e.g. an mjlab project that shipped the scalar
    `compute_reward`-only template and fails at training time with
    `AttributeError: … missing compute_reward_batched`.

    Semantics:
      - Reads `[adapter].class` from `config.toml` to pick the template.
      - OVERWRITES `rewards/v0.py` in place (git history preserves the old).
      - Rewrites `rewards/current.py` to re-export `v0.py`.
      - Raises `FileNotFoundError` if config.toml or rewards/ is missing.

    Returns the path to the rewritten `v0.py`.
    """
    project_dir = Path(project_dir).resolve()
    config_path = project_dir / "config.toml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no config.toml in {project_dir} — not a sculptor project.")
    rewards_dir = project_dir / "rewards"
    if not rewards_dir.is_dir():
        raise FileNotFoundError(
            f"no rewards/ dir in {project_dir} — run `sculpt init` first.")

    cfg = _parse_toml(config_path)
    adapter_class = (cfg.get("adapter") or {}).get("class", "")
    v0 = rewards_dir / "v0.py"
    v0.write_text(_v0_template_for(adapter_class), encoding="utf-8")
    # If iteration has produced v1+, keep current.py pointing at the latest;
    # regenerating v0 on an already-iterated project should not regress
    # current.py. Typical use is on broken projects with only v0.py —
    # _find_latest_reward_version then returns v0 itself.
    _, latest_path = _find_latest_reward_version(rewards_dir)
    _write_current_reexport(rewards_dir, latest_path)
    return v0
