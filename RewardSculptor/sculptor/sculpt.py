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
from typing import Any, Optional

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
):
    """Skip `adapter.train` when `iter_dir/checkpoint.pt` is already on
    disk and loads successfully — the expensive phase (≥ 22 min for
    mjlab) is idempotent given the same reward and seed, so reusing a
    prior run's artifact saves all that wall-clock when any downstream
    phase (rollout, diagnose, edit) failed on the previous attempt.
    Returns a `TrainResult`-compatible object either way.
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
        return TrainResult(
            checkpoint_path=ckpt,
            metrics_dict=metrics,
            component_means={},
            logs_path=iter_dir / "logs",
        )

    # Fresh training run — no checkpoint or all candidates corrupt.
    return adapter.train(
        reward_module_path=reward_module_path,
        output_dir=iter_dir,
        steps=steps,
        seed=seed,
    )


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
) -> SculptRunResult:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

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
