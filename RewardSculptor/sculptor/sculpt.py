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

Metric-plateau auto-kill is disabled: reward changes can fundamentally
alter the meaning of the adapter's primary_metric from one iteration to
the next, so a short no-improvement window is not a reliable halt signal.
Mission runs may still stop a stage when an explicit success criterion is
satisfied via the per-iteration callback.

`sculpt init <project_dir> --adapter <name>` scaffolds a fresh project with
config.toml, rewards/v0.py, kg_seeds.yml, .gitignore, and an initial git
commit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
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
    apply_prompt_edit,
)
from sculptor.llm import set_llm_log_dir


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
    #: §Ship 33: ground-truth task fitness on this iter's rollout when a
    #: `fitness_fn` is supplied to sculpt_run; None in the blind default.
    fitness: float | None = None
    #: §Metric-quality laws (LAW 7): the naturalness-GATED fitness used for
    #: STEER selection (best-by-fitness, early-stop) — distinct from `fitness`,
    #: the true, displayed task score. A joint-limit exploit → 0.0; a 'severe'
    #: (vel/torque) rollout is down-weighted. Defaults to `fitness` (no audit /
    #: ok verdict / observe mode), so selection is unchanged for clean iters.
    steer_fitness: float | None = None
    #: §Metric-quality laws (LAW 11): the naturalness steer-factor this iter
    #: (1.0 natural, <1 down-weighted, 0 joint-limit exploit) — tracked across
    #: iters for the Goodhart-onset early-stop (metric rising while naturalness
    #: falls = gaming). 1.0 when no audit ran (byte-identical default).
    naturalness_factor: float | None = None
    #: §Ship 33: the reward version FILE actually trained this iter (the
    #: input v<n> that `fitness` measures) — distinct from reward_path_after
    #: (the edit produced FROM it, untested until the next iter). Best-by-
    #: fitness selection keeps THIS, not the untested edit.
    reward_path_trained: Path | None = None
    #: §Ship 36 (F1): True when this iter REVERTED to the best-so-far reward
    #: (the prior iter regressed). Consumed by §Ship 37 case-memory: if iter
    #: N+1 reverted, iter N's edit was discarded and never measured, so its
    #: forward fitness delta must NOT be attributed (verdict stays 'unknown').
    reverted_to_best: bool = False
    #: §Convergence (RL_SCULPTOR_AUDIT §4.1): DENSE sub-success progress in
    #: [0,1] from the metric's optional `progress_score` key (min over the
    #: same saturating channels WITHOUT the completion gate). Used ONLY to
    #: RANK candidates below success — never granted as task success, never
    #: shown as the fitness. None when the metric doesn't emit it.
    progress: float | None = None
    #: The naturalness-GATED progress (mirrors `steer_fitness` for the dense
    #: channel) — an unnatural rollout cannot rank up via progress either.
    steer_progress: float | None = None
    #: §2026-07-03 case-memory upgrade: compact identities of the edits
    #: APPLIED this iter ("<operation> <target_term>", deferred excluded)
    #: so the KG run-case records WHAT was tried, not just how many.
    applied_edits: list[str] = field(default_factory=list)
    #: The metric's physical component breakdown for this iter's rollout
    #: (same dict the diagnoser sees) — the case-memory behavior signature.
    fitness_components: dict[str, Any] | None = None
    #: §env generalization 3/4: the env-spec version (e.g. "v2") ACTIVE
    #: while this iter trained — the environment half of the (reward, env)
    #: training config that `fitness` measured. None when the project has
    #: no env spec. Keep-best/revert repoint env/current.json to the best
    #: iter's version, exactly like the reward flow.
    env_spec_trained: str | None = None
    #: §Selection statistics: the checkpoint this iter's rollout(s) were
    #: evaluated from — retained so the end-of-run fresh-seed re-eval can
    #: re-roll the KEPT best on seeds never used for selection (the
    #: report-of-max discipline from Empirical Design in RL).
    checkpoint_path: Path | None = None


@dataclass
class SculptRunResult:
    iterations_run: int
    completed_iters: list[IterOutcome] = field(default_factory=list)
    early_stopped: bool = False
    early_stop_reason: str = ""
    primary_metric_history: list[float] = field(default_factory=list)
    #: §Ship 33: per-iter objective fitness (parallel to completed_iters)
    #: and the best iter selected on it — empty/None in the blind default.
    fitness_history: list[float] = field(default_factory=list)
    best_fitness: float | None = None
    best_fitness_iter: int | None = None
    #: §Convergence (RL_SCULPTOR_AUDIT §4.1): per-iter dense progress
    #: (parallel to fitness_history; 0.0 when the metric emits none) and the
    #: steer-progress of the best iter — the LEXICOGRAPHIC tie-break for
    #: best-by-fitness selection when spec fitness ties (e.g. all-zero
    #: below the completion gate).
    progress_history: list[float] = field(default_factory=list)
    best_progress: float | None = None
    #: §env generalization 3/4: env-spec version active at the best iter
    #: (the environment half of what best_fitness measured); None when the
    #: project has no env spec.
    best_env_spec: str | None = None
    #: §Selection statistics: the kept-best pair re-evaluated on FRESH
    #: rollout seeds after the run (median / per-seed raw scores). The
    #: selected `best_fitness` is a max statistic over the run; this is
    #: the unbiased report number. None when fresh eval didn't run.
    best_fitness_fresh: float | None = None
    fresh_fitness_per_seed: list[float] = field(default_factory=list)

    @property
    def final_reward_path(self) -> Path | None:
        for outcome in reversed(self.completed_iters):
            if outcome.reward_path_after:
                return outcome.reward_path_after
        return None

    @property
    def best_reward_path(self) -> Path | None:
        """§Ship 33: the reward kept by best-by-fitness selection (Eureka's
        'return the best across iterations' rule) — the best iter's
        TRAINED reward (the version `fitness` actually measured), NOT its
        untested edit. Falls back to the last reward when unavailable."""
        if self.best_fitness_iter is not None:
            for outcome in self.completed_iters:
                if (outcome.iter_index == self.best_fitness_iter
                        and outcome.reward_path_trained):
                    return outcome.reward_path_trained
        return self.final_reward_path


# ── Selection statistics ─────────────────────────────────────────────────
# §RL_SCULPTOR_AUDIT §6 (noise-floor bests) + RESEARCH_GAP_ANALYSIS §7.2:
# the lexicographic (steer_fitness, steer_progress) comparison used for
# keep-best/revert is made on NOISY scalars. Measured on tuck-jump E2E
# runs 1-2: behaviorally-identical re-rolls of the same reward differ by
# 1e-7..4e-6 in progress (sensor-noise ramps), which minted "new bests"
# below display precision, reset fitness_patience, and armed reverts on
# noise-level dips. The epsilon makes a progress tie-break require
# clearing that measured band; spec fitness comparisons are unchanged
# (the completion gate is a sharp 0/positive signal, not a noise ramp).
# progress_epsilon=0.0 restores the exact pre-epsilon behavior.
def _lex_improved(cur: tuple, best: tuple, progress_epsilon: float) -> bool:
    """New-best test: spec decides; progress breaks ties only when it
    clears the noise band."""
    if cur[0] != best[0]:
        return cur[0] > best[0]
    return cur[1] > best[1] + progress_epsilon


def _lex_regressed(cur: tuple, best: tuple, progress_epsilon: float) -> bool:
    """Strict-regression test (arms revert). Symmetric to `_lex_improved`:
    a progress dip inside the noise band is a TIE (build forward, patience
    counts), not a regression — reverting on seed noise re-trains the
    incumbent and discards the corrective edit, the exact deadlock the
    tie-no-revert fix (loop 1) removed."""
    if cur[0] != best[0]:
        return cur[0] < best[0]
    return cur[1] < best[1] - progress_epsilon


def _median(values: list[float]) -> float | None:
    """Median over per-seed evaluation scores. Chosen over mean (one
    diverged seed shouldn't drag the estimate) and over IQM (needs ≥4
    samples to differ from the median; eval_seeds is typically 2-5)."""
    if not values:
        return None
    import statistics
    return float(statistics.median(values))


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


def _current_reward_target(rewards_dir: Path) -> Optional[Path]:
    """The v<n>.py that rewards/current.py ACTUALLY re-exports — i.e. the
    reward training runs. Parses the `_LATEST = _HERE / 'v<n>.py'` line
    that `_write_current_reexport` emits (the only writer of current.py).

    §2026-07-04 run-boundary fix: current.py's target and the highest
    v<n>.py on disk DIVERGE whenever a run ends with best-by-fitness
    selection repointing current.py at an older version and a later run
    resumes (tuck-jump iter 16: trained v14 via current.py while the
    loop recorded reward_path_trained=v16 AND applied the diagnosis to
    v16's source — so keep-best then kept a never-trained file). Every
    consumer of "what trained this iter" must resolve through here.

    Accepts BOTH generated formats — this module's
    `_LATEST = _HERE / 'v<n>.py'` AND the UI backend's
    `_TARGET = Path(__file__).resolve().parent / 'v<n>.py'`
    (reward-sculptor-ui reward_store.py; ported from the parallel
    worktree fix, which caught that a UI-rewritten current.py would
    otherwise silently fall back to the buggy latest-version behavior).
    Returns None when current.py is missing or hand-edited into an
    unrecognizable shape (callers fall back to the latest version)."""
    current = rewards_dir / "current.py"
    if not current.is_file():
        return None
    try:
        text = current.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"/\s*(['\"])(v\d+\.py)\1", text)
    if not m:
        return None
    target = rewards_dir / m.group(2)
    return target if target.is_file() else None


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


def _load_partition_gate_report(iter_dir: Path | None) -> dict:
    """§Ship 54-pre (#12): read the partition-gate report apply_edits writes to
    `<iter_dir>/partition_gate.json` (present only when an objective metric
    steers AND there was something to flag). Empty dict otherwise."""
    if iter_dir is None:
        return {}
    return _load_json_if_present(Path(iter_dir) / "partition_gate.json")


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
    partition_summary: dict | None = None,
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

    # §Ship 54-pre (#12): the shaping↔metric partition gate's flags for this
    # iter (present only when an objective metric steered the run and something
    # was flagged). Records WHICH edits reach into the metric's held-out surface
    # or proposed easing a completion gate — the v5-class signal made visible.
    if partition_summary and partition_summary.get("flag_reasons"):
        lines.append("- **Metric partition gate**:")
        for r in partition_summary.get("flag_reasons", []):
            lines.append(f"  - {r}")

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
    render_width: int | None = None,
    render_height: int | None = None,
    seed: int | None = None,
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
        ("render_width", render_width),
        ("render_height", render_height),
        # §Selection statistics: distinct eval seeds per repeat rollout —
        # adapters that don't declare `seed` (gym_sb3) silently skip it.
        ("seed", seed),
    ):
        if value is not None and name in sig.parameters:
            extra[name] = value
    adapter.rollout(
        checkpoint_path=checkpoint_path,
        output_dir=rollout_dir,
        n_episodes=n_episodes,
        **extra,
    )


def _collect_hack_replays(adapter, runs_dir: Path, iter_index: int,
                          limit: int = 2) -> list[dict]:
    """§Hack-income regression screen (edit.py `_screen_hack_income`;
    CARD arXiv:2410.14660 TPE adapted): archived exploit replays.

    Scans PRIOR iteration dirs for a `reward_hacking` diagnosis whose
    rollout is still on disk, newest first, and builds a reward replay
    for each (≤ `limit` — replay construction is the expensive step).
    Every failure path is skipped silently: the screen is advisory
    hardening; a missing artifact must never block the edit."""
    out: list[dict] = []
    try:
        candidates: list[tuple[int, Path]] = []
        for d in Path(runs_dir).glob("iter_*"):
            try:
                k = int(d.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if k >= iter_index:
                continue
            diag = d / "diagnosis.json"
            roll = d / "rollout"
            if not diag.is_file() or not (roll / "trajectory.npz").is_file():
                continue
            try:
                fm = (json.loads(diag.read_text(encoding="utf-8"))
                      .get("failure_modes") or [])
            except Exception:  # noqa: BLE001 — unreadable diagnosis skipped
                continue
            if "reward_hacking" in fm:
                candidates.append((k, roll))
        for k, roll in sorted(candidates, reverse=True)[:max(0, limit)]:
            try:
                replay = adapter.build_reward_replay(roll)
            except Exception:  # noqa: BLE001 — per-replay best-effort
                replay = None
            if replay:
                out.append({"label": f"iter {k}", "replay_inputs": replay})
    except Exception:  # noqa: BLE001 — collection is best-effort
        return out
    return out


# ── One iteration ────────────────────────────────────────────────────────
def _read_control_file(control_file: Optional[Path]) -> dict:
    """§Ship 39 (H1): read the interactive control sidecar (mode / resume /
    feedback / stop) the backend writes. Missing / unreadable / partially
    written → {} (treated as 'auto'; never blocks the run)."""
    if control_file is None:
        return {}
    try:
        data = json.loads(Path(control_file).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — a partial/absent file just means "auto"
        return {}


def _pause_for_feedback(
    control_file: Optional[Path],
    iter_index: int,
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[Optional[str], bool]:
    """§Ship 39 (H1): at an iteration boundary, honor the control file.

    Returns (human_note_for_next_iter, should_stop):
      * mode != "manual" → no pause (auto); (None, False).
      * "stop" set        → (None, True): the loop ends cleanly.
      * mode == "manual"  → emit `awaiting_feedback`, then BLOCK polling the
        file until the human bumps `resume_token` (optionally with `feedback`),
        flips `mode` to auto, sets `stop`, or `timeout` elapses (→ auto-resume
        so a dead client can't pin the GPU forever).
    The human's feedback is returned so the NEXT iteration's diagnose uses it
    ("review iteration N on the video → steer iteration N+1")."""
    ctrl = _read_control_file(control_file)
    if ctrl.get("stop"):
        return None, True
    if ctrl.get("mode") != "manual":
        return None, False
    start_token = int(ctrl.get("resume_token", 0) or 0)
    _emit_event({"type": "awaiting_feedback", "iter": iter_index})
    waited = 0.0
    step = max(0.05, float(poll_interval))
    while waited < timeout:
        time.sleep(step)
        waited += step
        ctrl = _read_control_file(control_file)
        if ctrl.get("stop"):
            return None, True
        if ctrl.get("mode") != "manual":
            _emit_event({"type": "feedback_resumed", "iter": iter_index,
                         "reason": "auto_mode"})
            # §Ship 39 review (MEDIUM): carry feedback ONLY if this was an
            # explicit resume (token bumped — e.g. "Continue + go Auto"). A
            # BARE mode flip (the toggle) must NOT re-inject a stale note left
            # in the sidecar by a prior iteration's resume.
            note = ((ctrl.get("feedback") or None)
                    if int(ctrl.get("resume_token", 0) or 0) > start_token
                    else None)
            return note, False
        if int(ctrl.get("resume_token", 0) or 0) > start_token:
            _emit_event({"type": "feedback_resumed", "iter": iter_index,
                         "reason": "human"})
            return (ctrl.get("feedback") or None), False
    _emit_event({"type": "feedback_timeout", "iter": iter_index,
                 "timeout_s": timeout})
    return None, False


def _fitness_components_for_prompt(detail: dict | None) -> dict | None:
    """§Ship 36 (F2): reduce a metric's full result dict to the numeric
    sub-components worth showing the diagnoser. Drops the top-line score
    (shown separately as `current`), bookkeeping keys, and any non-finite /
    non-scalar entry. Returns None when nothing useful remains."""
    if not detail:
        return None
    skip = {"spec_score", "spec_name", "capture", "error"}
    out: dict[str, Any] = {}
    for k, v in detail.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                out[k] = round(fv, 5)
    return out or None


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
    fitness_fn: Optional[Callable[[Path], float]] = None,
    prior_fitness: Optional[dict] = None,
    fitness_observe_only: bool = False,
    revert_base: Optional[Path] = None,
    env_revert_version: Optional[str] = None,
    human_note: Optional[str] = None,
) -> IterOutcome:
    iter_cfg = cfg.get("iteration", {}) or {}
    primary_key = str(iter_cfg.get("primary_metric", "mean_return"))
    behavior_metric_names: list[str] = list(iter_cfg.get("behavior_metrics", []))
    steps = int(iter_cfg.get("steps_per_iter", 50_000))
    if dry_run:
        steps = min(1000, steps)

    iter_dir = runs_dir / f"iter_{iter_index}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    # §llm provenance: every LLM call this iteration makes (diagnose /
    # edit / env / physics) archives to iter_dir/llm_calls.jsonl. The
    # contextvar is last-set-wins across the single-threaded pipeline.
    set_llm_log_dir(iter_dir)

    reward_path_before = _ensure_current_py(rewards_dir)
    latest_n, latest_reward_file = _find_latest_reward_version(rewards_dir)

    # §2026-07-04 run-boundary fix: training uses current.py, so the edit
    # base and the trained-reward record must follow current.py's ACTUAL
    # re-export target — not the highest v<n>.py on disk. The two diverge
    # when a previous run's best-by-fitness selection repointed current.py
    # at an older version and this run resumed: the old code then applied
    # this iter's diagnosis to the wrong source AND recorded (and could
    # keep-best) a reward that never trained (tuck-jump iter 16 trained
    # v14 while everything downstream said v16). The new edit is still
    # numbered v<latest_n+1> (monotonic numbering, trained content).
    trained_target = _current_reward_target(rewards_dir)
    if trained_target is None:
        # A generated current.py whose target file was DELETED would
        # crash mid-train on import — repair the re-export to the latest
        # version so training, records, and edits agree again. (A
        # hand-written, unrecognizable current.py is left alone: training
        # uses it as-is and the records fall back to the latest version.)
        cur_text = ""
        try:
            cur_text = (rewards_dir / "current.py").read_text(
                encoding="utf-8")
        except OSError:
            pass
        if "Auto-generated by sculptor.edit" in cur_text:
            _write_current_reexport(rewards_dir, latest_reward_file)
        trained_target = latest_reward_file

    # §Ship 36 (F1): revert-on-regression. By default this iter trains on
    # current.py and edits from the reward it re-exports. When the caller
    # passes `revert_base` (the prior iter regressed fitness), repoint
    # current.py at the best-so-far reward so BOTH training and the edit
    # base use it — best-first search instead of compounding a drifting
    # edit.
    edit_base = trained_target
    reward_path_trained = trained_target
    reverted_to_best = False
    if revert_base is not None and Path(revert_base).is_file():
        _write_current_reexport(rewards_dir, Path(revert_base))
        reward_path_before = rewards_dir / "current.py"
        edit_base = Path(revert_base)
        reward_path_trained = Path(revert_base)
        reverted_to_best = True
        # §Ship 36 review (MEDIUM): a stale checkpoint from a prior crashed
        # attempt at THIS iter index was trained on the DEGRADED reward;
        # `_train_or_resume` would reuse it ("resume wins") and silently skip
        # retraining on the reverted reward — defeating the revert's whole
        # point. Invalidate it so training actually re-runs on the best-so-far
        # reward. (No-op on the normal fresh path: no checkpoint exists yet.)
        for _ext in ("pt", "zip"):
            _stale = iter_dir / f"checkpoint.{_ext}"
            try:
                if _stale.is_file():
                    _stale.unlink()
            except OSError:  # pragma: no cover — best-effort cleanup
                pass
        _emit_event({
            "type": "reward_reverted_to_best",
            "iter": iter_index,
            "reverted_to": Path(revert_base).name,
        })

    # §env generalization 3/4: the environment half of the revert — when
    # the prior iter strictly regressed, train under the env spec the
    # best iter used, not a possibly-degraded diagnoser edit. Reverting
    # only the reward while keeping a bad env change would attribute the
    # env's damage to the reward. Best-effort: a missing/invalid target
    # version logs and trains under the current spec.
    #
    # The loop ITERATES only the MANAGED per-project spec — the one the
    # adapter actually trains under AND that lives at env/current.json.
    # An explicit config env_spec_path pointing anywhere else is static
    # configuration: no revert, no version record, no diagnoser edits
    # (otherwise apply/record would target project/env while training
    # reads the pinned file — silent divergence).
    env_dir = project / "env"
    _active_spec_path = str(getattr(adapter, "env_spec_path", "") or "")
    try:
        env_managed = bool(_active_spec_path) and (
            Path(_active_spec_path).resolve()
            == (env_dir / "current.json").resolve())
    except OSError:  # pragma: no cover — unresolvable path
        env_managed = False
    if env_managed and revert_base is not None and env_revert_version:
        try:
            from sculptor.env_spec import (
                read_current_env_spec, repoint_env_current)

            _cur = read_current_env_spec(env_dir)
            _cur_v = ((_cur.get("meta") or {}).get("version")
                      if _cur else None)
            if _cur_v != env_revert_version:
                repoint_env_current(env_dir, env_revert_version)
                _emit_event({
                    "type": "env_spec_reverted",
                    "iter": iter_index,
                    "reverted_to": env_revert_version,
                })
        except Exception as e:  # noqa: BLE001 — revert is best-effort
            sys.stderr.write(
                f"[sculpt] iter {iter_index}: env-spec revert to "
                f"{env_revert_version} failed ({type(e).__name__}: {e}) — "
                f"training under the current spec\n")

    # Record the env-spec version this iter actually trains under (the
    # environment half of the training config; None = no spec/defaults
    # or an unmanaged pinned spec).
    env_spec_trained: Optional[str] = None
    if env_managed:
        try:
            from sculptor.env_spec import read_current_env_spec as _read_env

            _cur_spec = _read_env(env_dir)
            if _cur_spec:
                env_spec_trained = (_cur_spec.get("meta") or {}).get("version")
                # Snapshot the exact spec into the iter dir (parallel to
                # reward_spec.json) so policy export / audits can recover
                # the environment half of the training config even after
                # the project's current.json has moved on. First write
                # wins: on crash-resume, _train_or_resume reuses the
                # already-trained checkpoint, and current.json may have
                # been repointed by a later apply_env_edits — rewriting
                # would pair the checkpoint with a spec it never saw.
                _snap = iter_dir / "env_spec.json"
                if not _snap.exists():
                    _snap.write_text(
                        json.dumps(_cur_spec, indent=2, sort_keys=True,
                                   default=str),
                        encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — invalid spec fails later, loudly
            sys.stderr.write(
                f"[sculpt] iter {iter_index}: env spec unreadable "
                f"({type(e).__name__}: {e})\n")

    # §2026-07-04: report the version that actually TRAINS this iter
    # (current.py's target / the revert base), not the disk maximum —
    # the two diverge at run boundaries after best-selection.
    _m_trained = re.fullmatch(r"v(\d+)", Path(reward_path_trained).stem)
    _emit_event({
        "type": "iter_started",
        "iter": iter_index,
        "steps": steps,
        "reward_version_before": (
            int(_m_trained.group(1)) if _m_trained else latest_n),
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
        render_width=iter_cfg.get("render_width"),
        render_height=iter_cfg.get("render_height"),
    )

    # §Ship 33: objective task fitness on this rollout (ground truth,
    # higher = better) — computed BEFORE diagnose so the diagnosis is
    # fitness-guided. fitness_fn is supplied by the eval harness
    # (spec_metric) only when --fitness-in-loop is on; None keeps the
    # blind default. A crash here must never kill the iter — honest None.
    iter_fitness: float | None = None
    iter_progress: float | None = None
    objective_progress: dict | None = None
    fitness_components: dict | None = None
    # §Selection statistics: extra eval-rollout dirs (multi-seed) + the
    # per-seed raw scores. Populated only when eval_seeds > 1 AND the
    # fitness fn exposes `detail_dir`; the realism audit below extends
    # its naturalness check over these dirs (min factor = safe direction).
    extra_eval_dirs: list[Path] = []
    fitness_per_seed: list[float] = []
    progress_per_seed: list[float] = []
    if fitness_fn is not None:
        # §Ship 36 (F2): prefer the `.detail` accessor (rides on the fitness
        # fn) to get the FULL component breakdown in one compute; fall back to
        # the plain float for callers that supply a bare fitness_fn.
        detail_fn = getattr(fitness_fn, "detail", None)
        try:
            if detail_fn is not None:
                detail = detail_fn(iter_dir) or {}
                iter_fitness = float(detail.get("spec_score", 0.0) or 0.0)
                fitness_components = _fitness_components_for_prompt(detail)
                # §Convergence (RL_SCULPTOR_AUDIT §4.1): the metric's OPTIONAL
                # dense progress channel (min of pre-gate saturating channels).
                # Advisory ranking signal only — absent/malformed → None.
                pr = detail.get("progress_score")
                if (isinstance(pr, (int, float)) and not isinstance(pr, bool)
                        and math.isfinite(float(pr))):
                    iter_progress = min(1.0, max(0.0, float(pr)))
            else:
                iter_fitness = float(fitness_fn(iter_dir))
        except Exception as e:  # noqa: BLE001 — fitness is advisory, never fatal
            sys.stderr.write(
                f"[sculpt] iter {iter_index}: fitness_fn raised "
                f"{type(e).__name__}: {e} — treating as unavailable\n"
            )
            iter_fitness = None
        # §Selection statistics (RESEARCH_GAP_ANALYSIS §7.2): the keep-best
        # decision was previously made on ONE rollout batch — a noisy
        # scalar compared with strict `>`. With `eval_seeds = K > 1` in
        # [iteration], the same checkpoint is re-rolled K-1 more times on
        # distinct seeds and the selection scores become the MEDIAN over
        # seeds (robust to one diverged roll). The primary `rollout/` stays
        # the diagnoser's view (keyframes / video / components); extras
        # land in `rollout_eval_<k>/`. Each extra is best-effort: a failed
        # re-roll is skipped, never fatal. Requires the fitness fn's
        # `detail_dir` accessor (spec + generated metrics both have it).
        _eval_seeds = 1
        try:
            _eval_seeds = max(1, int(iter_cfg.get("eval_seeds", 1) or 1))
        except Exception:  # noqa: BLE001 — a malformed knob keeps N=1
            _eval_seeds = 1
        detail_dir_fn = getattr(fitness_fn, "detail_dir", None)
        if (iter_fitness is not None and _eval_seeds > 1
                and detail_dir_fn is not None):
            fitness_per_seed = [float(iter_fitness)]
            progress_per_seed = [float(iter_progress or 0.0)]
            for k in range(1, _eval_seeds):
                eval_dir = iter_dir / f"rollout_eval_{k}"
                try:
                    eval_dir.mkdir(exist_ok=True)
                    _rollout_or_resume(
                        adapter=adapter,
                        iter_index=iter_index,
                        rollout_dir=eval_dir,
                        checkpoint_path=checkpoint_path,
                        n_episodes=int(iter_cfg.get("rollout_episodes", 6)),
                        max_episode_steps=iter_cfg.get("max_episode_steps"),
                        playback_speed=iter_cfg.get("playback_speed"),
                        render_every=iter_cfg.get("render_every"),
                        fps=iter_cfg.get("rollout_fps"),
                        # Deterministic, disjoint from training seeds
                        # (config seed + iter) and the fresh-eval band
                        # (90k+): reproducible re-rolls, distinct per k.
                        seed=10_000 + iter_index * 100 + k,
                    )
                    d = detail_dir_fn(eval_dir) or {}
                    fitness_per_seed.append(
                        float(d.get("spec_score", 0.0) or 0.0))
                    pr_k = d.get("progress_score")
                    progress_per_seed.append(
                        min(1.0, max(0.0, float(pr_k)))
                        if (isinstance(pr_k, (int, float))
                            and not isinstance(pr_k, bool)
                            and math.isfinite(float(pr_k)))
                        else 0.0)
                    extra_eval_dirs.append(eval_dir)
                except Exception as e:  # noqa: BLE001 — per-seed best-effort
                    sys.stderr.write(
                        f"[sculpt] iter {iter_index}: eval seed {k} "
                        f"skipped — {type(e).__name__}: {e}\n")
            if len(fitness_per_seed) > 1:
                iter_fitness = _median(fitness_per_seed)
                if iter_progress is not None:
                    iter_progress = _median(progress_per_seed)
        if iter_fitness is not None:
            pf = prior_fitness or {}
            best_so_far = pf.get("best_so_far")
            last = pf.get("last")
            progress = {
                "current": round(iter_fitness, 5),
                "best_so_far": (round(best_so_far, 5)
                                if best_so_far is not None else None),
                "last": round(last, 5) if last is not None else None,
                "delta": (round(iter_fitness - last, 5)
                          if last is not None else None),
                # §Ship 36 (F2): physical sub-measurements so the diagnoser
                # can localize WHAT is wrong, not just THAT fitness fell.
                "components": fitness_components,
                # §Ship 36 (F1): tell the diagnoser the prior edit regressed
                # and was reverted — so it proposes a DIFFERENT direction.
                "reverted_to_best": reverted_to_best,
            }
            # §Selection statistics: per-seed dispersion for the diagnoser —
            # "fitness 0.4 (seeds: 0.0/0.4/0.9)" reads very differently from
            # a stable 0.4; the LLM should know when behavior is bimodal.
            if len(fitness_per_seed) > 1:
                progress["eval_seeds"] = len(fitness_per_seed)
                progress["fitness_per_seed"] = [
                    round(v, 5) for v in fitness_per_seed]
            # Always emit for DISPLAY (chart/chip/A/B), even in observe mode.
            _emit_event({
                "type": "iter_fitness",
                "iter": iter_index,
                "fitness": round(iter_fitness, 5),
                "progress": (round(iter_progress, 5)
                             if iter_progress is not None else None),
                "best_so_far": progress["best_so_far"],
                "delta_vs_previous": progress["delta"],
                "observe_only": bool(fitness_observe_only),
                # §Selection statistics: null/absent when eval_seeds=1.
                "eval_seeds": (len(fitness_per_seed)
                               if len(fitness_per_seed) > 1 else None),
                "fitness_per_seed": ([round(v, 5) for v in fitness_per_seed]
                                     if len(fitness_per_seed) > 1 else None),
            })
            # §Ship 35: in observe mode the diagnoser must NOT see fitness —
            # the signal stays passive (no influence on the run).
            objective_progress = None if fitness_observe_only else progress

    # §7.3: physics-realism audit. Reads the expanded `trajectory.npz`
    # (§7.1) + `mjcf_limits.json` that the rollout runner drops next to
    # it, computes torque-saturation / joint-vel / joint-limit metrics,
    # and persists `iter_<N>/realism_audit.json` so diagnose + UI can
    # surface the verdict. Best-effort — failures (missing file, shape
    # drift) return verdict=unknown rather than crashing the loop.
    audit_result: dict[str, Any] | None = None
    # §LAW 7: naturalness channel — a SEPARATE signal that gates STEER credit
    # without touching the displayed fitness. Init to a no-op pass so a failed/
    # absent audit never suppresses steering, and the steer value defaults to
    # the true fitness (byte-identical to the pre-LAW-7 loop).
    naturalness: dict[str, Any] | None = None
    iter_steer_fitness: float | None = iter_fitness
    iter_steer_progress: float | None = iter_progress
    iter_naturalness_factor: float = 1.0   # §LAW 11: tracked for Goodhart-onset
    try:
        from sculptor.adapters.realism import (
            audit_rollout, naturalness_channel, steer_fitness as _steer_fitness)
        audit_result = audit_rollout(
            trajectory_path=rollout_dir / "trajectory.npz",
            limits_path=rollout_dir / "mjcf_limits.json",
        )
        # §LAW 7: derive the naturalness decision + the naturalness-gated steer
        # fitness. A joint-limit exploit earns no steer credit; a severe
        # (vel/torque) rollout is down-weighted. STEER mode only — in observe
        # mode the verdict is recorded but never alters selection.
        naturalness = naturalness_channel(audit_result)
        audit_result["naturalness"] = naturalness
        iter_naturalness_factor = float(naturalness.get("steer_factor", 1.0) or 1.0)
        # §Selection statistics: when multi-seed eval ran, audit EVERY
        # extra rollout and take the MINIMUM steer factor — the safe
        # direction (an exploit visible on any seed earns no steer
        # credit; median-ing naturalness would let a 1-in-K joint-limit
        # exploit through). Best-effort per dir.
        for _ed in extra_eval_dirs:
            try:
                _a = audit_rollout(
                    trajectory_path=_ed / "trajectory.npz",
                    limits_path=_ed / "mjcf_limits.json",
                )
                _f = float(
                    naturalness_channel(_a).get("steer_factor", 1.0) or 1.0)
                if _f < iter_naturalness_factor:
                    iter_naturalness_factor = _f
                    naturalness = dict(naturalness)
                    naturalness["min_factor_source"] = _ed.name
                    naturalness["steer_factor"] = _f
            except Exception:  # noqa: BLE001 — extra audits advisory
                pass
        if not fitness_observe_only:
            iter_steer_fitness = _steer_fitness(iter_fitness, naturalness)
            # §Convergence: gate the dense progress channel identically — an
            # unnatural rollout cannot rank up via progress either.
            iter_steer_progress = _steer_fitness(iter_progress, naturalness)
        if objective_progress is not None:
            objective_progress["naturalness"] = naturalness
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
            # §LAW 7: surface the naturalness decision for the UI chip + diagnoser.
            "naturalness_flag": naturalness.get("flag"),
            "naturalness_hard_reject": bool(naturalness.get("hard_reject")),
            "naturalness_steer_factor": naturalness.get("steer_factor"),
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
            objective_progress=objective_progress,
            human_note=human_note,
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

    # §RL_SCULPTOR_AUDIT §4.4 (loop 4b): build the edit anti-collapse
    # replay from the rollout whose behavior the next edit must refine —
    # THIS iter's rollout by default; the best-so-far iter's rollout when
    # this iter strictly regressed the lexicographic (steer, progress)
    # key (a collapsed rollout is not the behavior to protect). Adapters
    # without replay support return None → the screen is skipped.
    replay_inputs = None
    if not dry_run:
        replay_dir = rollout_dir
        pf = prior_fitness or {}
        best_sf = pf.get("best_so_far")
        best_dir = pf.get("best_iter_dir")
        if best_sf is not None and best_dir:
            cur_key = ((iter_steer_fitness
                        if iter_steer_fitness is not None else 0.0),
                       (iter_steer_progress
                        if iter_steer_progress is not None else 0.0))
            best_key = (best_sf, pf.get("best_progress") or 0.0)
            cand = Path(best_dir) / "rollout"
            if cur_key < best_key and cand.is_dir():
                replay_dir = cand
        try:
            replay_inputs = adapter.build_reward_replay(replay_dir)
        except Exception as e:  # noqa: BLE001 — screen is best-effort
            sys.stderr.write(
                f"[sculpt] iter {iter_index}: reward replay build skipped — "
                f"{type(e).__name__}: {e}\n")

    # §Hack-income regression screen: archived exploits (prior iters the
    # diagnoser flagged reward_hacking) the candidate must not re-open.
    # `hack_income_screen = false` in [iteration] disables.
    hack_replays: list[dict] = []
    if not dry_run and bool(iter_cfg.get("hack_income_screen", True)):
        hack_replays = _collect_hack_replays(adapter, runs_dir, iter_index)
        if hack_replays:
            _emit_event({
                "type": "hack_screen_active",
                "iter": iter_index,
                "exploits": [hr["label"] for hr in hack_replays],
            })

    # 4. Apply edits → v<n+1>.py
    new_iter_tag = f"v{latest_n + 1}"
    new_reward_path: Path | None = None
    try:
        if dry_run:
            new_reward_path = _dry_run_apply_edits(
                current_reward_path=edit_base,
                new_iter_id=new_iter_tag)
        else:
            # Filter out fully-deferred — apply_edits raises on empty
            # applicable_edits, which we want to surface clearly.
            # §7.2: pass iter_dir so apply_edits can load
            # `reward_trajectory.json` and inject it into the rewrite
            # prompt (same data the diagnoser saw).
            new_reward_path = apply_edits(
                current_reward_path=edit_base,
                diagnosis=diagnosis,
                new_iter_id=new_iter_tag,
                reward_contract=adapter.reward_contract(),
                kg_store=kg_store,
                iter_dir=iter_dir,
                # §Ship 54-pre (#12): hand the active metric's held-out
                # observable surface to the shaping↔metric partition gate.
                # None when no objective metric steers (blind run) → gate
                # no-ops, byte-identical.
                metric_observables=getattr(
                    fitness_fn, "metric_observables", None),
                # §RL_SCULPTOR_AUDIT §4.4 (loop 4b): anti-collapse replay.
                replay_inputs=replay_inputs,
                # §Hack-income regression screen: caught exploits stay caught.
                hack_replays=hack_replays or None,
                # §best-of-K (RESEARCH_GAP_ANALYSIS §3.3): K framed candidate
                # rewrites per diagnosis, offline-screened + margin-ranked;
                # only the winner trains. 1 (default) = single-shot path.
                n_candidates=int(iter_cfg.get("edit_candidates", 1)),
            )
    except EditValidationError as e:
        sys.stderr.write(
            f"[sculpt] iter {iter_index}: apply_edits skipped — "
            f"{type(e).__name__}: {e}\n")

    # §env generalization 3/4: apply diagnoser-proposed env-curriculum
    # edits (train section only, validated + bounded) — the environment
    # counterpart of apply_edits above. The new spec version takes
    # effect NEXT iter's training, exactly like the reward edit. Skipped
    # in dry-run (no LLM ran) and when the diagnosis proposed none.
    applied_env_edits: list[str] = []
    if diagnosis.proposed_env_edits and not dry_run:
        if not env_managed:
            # Diagnose only shows the # ENV_SPEC surface for managed
            # specs, so this is belt-and-braces — but never silent.
            _emit_event({
                "type": "env_spec_updated",
                "iter": iter_index,
                "new_version": None,
                "applied": [],
                "rejected": [
                    {"parameter": getattr(e, "parameter", "?"),
                     "reason": "env spec is not loop-managed "
                               "(explicit env_spec_path)"}
                    for e in diagnosis.proposed_env_edits
                ],
            })
        else:
            try:
                from sculptor.env_spec import apply_env_edits

                env_edit_result = apply_env_edits(
                    env_dir, diagnosis.proposed_env_edits)
                applied_env_edits = list(env_edit_result.get("applied") or [])
                _emit_event({
                    "type": "env_spec_updated",
                    "iter": iter_index,
                    "new_version": env_edit_result.get("new_version"),
                    "applied": applied_env_edits,
                    "rejected": [
                        {"parameter": p, "reason": r[:300]}
                        for p, r in (env_edit_result.get("rejected") or [])
                    ],
                })
            except Exception as e:  # noqa: BLE001 — env edits are advisory
                sys.stderr.write(
                    f"[sculpt] iter {iter_index}: env-spec edits skipped — "
                    f"{type(e).__name__}: {e}\n")
                _emit_event({
                    "type": "env_spec_updated",
                    "iter": iter_index,
                    "new_version": None,
                    "applied": [],
                    "rejected": [{
                        "parameter": "*",
                        "reason": f"{type(e).__name__}: {e}"[:300]}],
                })

    # §Ship 54-pre (#12): surface the partition-gate report (written by
    # apply_edits into iter_dir/partition_gate.json when a metric steers and
    # there is something to flag) on the loop path, which passes no on_event.
    partition_summary = _load_partition_gate_report(iter_dir)
    if partition_summary:
        _emit_event({
            "type": "partition_gate",
            "iter": iter_index,
            "flagged": int(partition_summary.get("flagged_edit_count", 0) or 0),
            "reasons": list(partition_summary.get("flag_reasons", []) or []),
        })

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
        partition_summary=partition_summary,
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
        # §2026-07-04: the version that actually trained (mirrors
        # iter_started) — not the disk maximum.
        "reward_version_before": (
            int(_m_trained.group(1)) if _m_trained else latest_n),
        "reward_version_after": reward_version_after,
        "paper_refs": sorted(set(applied_paper_refs)),
    })

    # §Ship 48: never-silent env-extension signal. The diagnoser flags edits
    # it WANTS but can't ground because the adapter doesn't expose the needed
    # field (requires_env_extension); pre-Ship-48 these dead-ended in the
    # changelog and the user never saw that the run was structurally blocked
    # (the g1-kick-v3 stall: every kick term deferred for want of per-foot
    # channels, iter after iter). Emit it so the Runs tab can chip "this skill
    # needs adapter channels X". Modeled on physics_edit_suggested, but
    # informational only — an env extension is a code change, never auto-applied.
    deferred_edits = [
        e for e in diagnosis.proposed_edits
        if getattr(e, "requires_env_extension", False)
    ]
    if deferred_edits:
        _emit_event({
            "type": "requires_env_extension",
            "iter": iter_index,
            "terms": [getattr(e, "target_term", "") for e in deferred_edits],
            "rationales": [getattr(e, "rationale", "") for e in deferred_edits],
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
        fitness=iter_fitness,
        steer_fitness=iter_steer_fitness,
        naturalness_factor=iter_naturalness_factor,
        reward_path_trained=reward_path_trained,
        reverted_to_best=reverted_to_best,
        progress=iter_progress,
        steer_progress=iter_steer_progress,
        # §2026-07-03 case-memory upgrade: what was actually tried +
        # the behavior signature, so the KG run-case is actionable.
        applied_edits=[
            f"{getattr(e, 'operation', '?')} {getattr(e, 'target_term', '?')}"
            for e in diagnosis.proposed_edits
            if not getattr(e, "requires_env_extension", False)
        ] + [
            # §env generalization 3/4: env-curriculum edits ride the same
            # case-memory channel so the KG learns environment lessons too.
            f"env: {a}" for a in applied_env_edits
        ],
        fitness_components=fitness_components,
        env_spec_trained=env_spec_trained,
        checkpoint_path=checkpoint_path,
    )


# ── Early stop ───────────────────────────────────────────────────────────
def detect_goodhart_onset(
    fitness_history: list[float],
    naturalness_history: list[float],
    *,
    window: int = 3,
    rise_tol: float = 0.02,
    min_unnatural: int = 2,
) -> Optional[str]:
    """§Metric-quality laws (LAW 11): Goodhart-ONSET — the objective metric is
    still CLIMBING while the policy has SUSTAINABLY *become* less natural, i.e.
    it is climbing the PROXY by gaming it. Lock the prior best rather than
    optimize further into the exploit. Returns a stop-reason string, or None.

    Pure + deterministic (offline-testable). Needs >= window+1 aligned points.
    Fires ONLY when, over the last `window` iters, ALL of:
      (a) the metric ROSE — recent MAX exceeds the prior MAX by >= `rise_tol`
          (max-based, so a mid-window dip-then-recover does NOT read as a rise);
      (b) >= `min_unnatural` of those iters are NON-PASS (naturalness steer-
          factor < 1.0 — 'severe'/exploit): a SUSTAINED loss of naturalness, not
          one transient aggressive frame (LAW 7 down-weights a SINGLE severe iter
          precisely because an aggressive-but-valid kick can momentarily exceed
          3× nominal joint speed); AND
      (c) the recent window is, on average, LESS natural than everything before
          it — the policy is *becoming* less natural (a decline), not merely a
          run that was always aggressive (which (c) lets through).
    With the no-audit default (naturalness all 1.0) the count is 0, so onset
    NEVER fires — byte-identical to the pre-LAW-11 loop."""
    n = min(len(fitness_history), len(naturalness_history))
    if n < window + 1:
        return None
    recent_f = [float(x) for x in fitness_history[n - window:n]]
    prior_f = [float(x) for x in fitness_history[:n - window]]
    recent_nat = [float(x) for x in naturalness_history[n - window:n]]
    prior_nat = [float(x) for x in naturalness_history[:n - window]]
    rose = (max(recent_f) - max(prior_f)) >= rise_tol
    n_unnatural = sum(1 for v in recent_nat if v < 1.0 - 1e-9)
    declined = (sum(recent_nat) / len(recent_nat)) < (
        sum(prior_nat) / len(prior_nat)) - 1e-9
    if rose and n_unnatural >= min_unnatural and declined:
        return (f"goodhart onset: objective metric still climbing "
                f"(max {max(prior_f):.3f}→{max(recent_f):.3f}) while "
                f"{n_unnatural}/{window} recent iters became less natural "
                f"(severe or mild) — the policy is climbing the proxy by becoming "
                f"less natural; locking the prior best")
    return None


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
    """Compatibility shim for the removed metric-plateau auto-kill.

    Older callers/tests import this helper and older configs may still
    contain `early_stop_*` fields. Keep the symbol and parameters stable,
    but never halt based on primary-metric history.
    """
    return False


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
    render_width: Optional[int] = None,
    render_height: Optional[int] = None,
    rollout_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    auto_adjust_physics: Optional[bool] = None,
    early_stop_enabled: Optional[bool] = None,
    early_stop_patience: Optional[int] = None,
    init_policy_path: Optional[Path | str] = None,
    per_iter_callback: Optional[Callable[["IterOutcome"], Optional[str]]] = None,
    fitness_fn: Optional[Callable[[Path], float]] = None,
    fitness_patience: int = 2,
    fitness_target: Optional[float] = None,
    fitness_observe_only: bool = False,
    fitness_revert: bool = True,
    eval_seeds: Optional[int] = None,
    progress_epsilon: Optional[float] = None,
    fresh_eval_seeds: Optional[int] = None,
    control_file: Optional[Path | str] = None,
    feedback_timeout: float = 3600.0,
    feedback_poll_interval: float = 2.0,
) -> SculptRunResult:
    """§Ship-19d: `per_iter_callback` is fired AFTER each iter's
    artifacts are persisted. Returning `None` keeps the loop running;
    returning a non-empty string is interpreted as an early-stop
    reason and breaks the loop after recording that iter as
    completed. Used by mission_run to early-stop a stage the moment
    its success_criterion is satisfied (Goal A) — a no-op for plain
    sculpt_run callers that don't pass it. Distinct from the
    metric-plateau early-stop at lines 1311+: that one looks at the
    history shape; this one is a goal-aware exit signal.

    §Ship 33 — `fitness_fn(iter_dir) -> float` (optional) supplies a
    held-out, ground-truth task fitness (higher = better) per iteration.
    When given, the loop (1) surfaces it to the diagnoser so the
    diagnose→edit search is fitness-guided, (2) keeps the BEST-by-fitness
    reward as `current.py` at the end (Eureka's 'return best across
    iterations' rule, vs the blind default of keeping the last), and
    (3) early-stops after `fitness_patience` iters with no new best, or
    once `fitness_target` is reached. None preserves the original blind
    behavior exactly. This is the apples-to-apples fix for the eval
    asymmetry where eureka selected on fitness and sculpt could not.

    §Ship 35 — `fitness_observe_only=True` (the "observe" mode) makes the
    fitness signal PURELY PASSIVE: it is still computed and emitted every
    iter (so the UI can display + chart it, and a blind-vs-guided A/B is
    visible) but it does NOT influence the run — the diagnoser does not
    see it, no best-by-fitness `current.py` repoint, no fitness early-stop.
    `best_fitness`/`fitness_history` are still tracked for DISPLAY only.
    This is the safe default for auto-generated metrics that have not yet
    earned steer-rights via calibration (see eval/metric_calibration).

    §Ship 36 — `fitness_revert=True` (steer mode only) turns the diagnose→
    edit search into best-first hill-climbing: when an iter does NOT set a
    new best fitness, the NEXT iter reverts current.py (its training + edit
    base) to the best-so-far reward instead of compounding edits on the
    degraded latest. This is the missing 'edit accept/reject' from Ship 33:
    without it, a single bad edit drifts the reward and the policy hacks it
    deeper each iter (Sam's G1 kick run: best at iter 1, flailing by iter 3).
    Observe-only NEVER reverts (the signal must stay passive)."""
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
        ("render_width", render_width),
        ("render_height", render_height),
        ("rollout_episodes", rollout_episodes),
        ("seed", seed),
        ("auto_adjust_physics", auto_adjust_physics),
        # §Selection statistics: multi-seed eval + noise band + fresh
        # re-eval of the kept best. config.toml [iteration] keys of the
        # same names; None = config/default.
        ("eval_seeds", eval_seeds),
        ("progress_epsilon", progress_epsilon),
        ("fresh_eval_seeds", fresh_eval_seeds),
    )
    # `early_stop_enabled` / `early_stop_patience` remain accepted by the
    # public function and CLI for compatibility, but metric-plateau auto-kill
    # is disabled and the values are intentionally ignored.
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
    # §Selection statistics: the progress-tie noise band for keep-best /
    # revert (see `_lex_improved`/`_lex_regressed`). Default 1e-5 sits
    # above the measured seed-noise ramp (1e-7..4e-6, audit §6) and well
    # below every real progress signal seen (≥7.65e-3). 0.0 restores the
    # exact strict-`>` behavior.
    try:
        _pe_raw = (cfg.get("iteration") or {}).get("progress_epsilon", 1e-5)
        _progress_epsilon = max(0.0, float(1e-5 if _pe_raw is None else _pe_raw))
    except Exception:  # noqa: BLE001 — malformed knob keeps the default
        _progress_epsilon = 1e-5
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
        # §Ship 24 (R1): seed surfaced at run start so any result can
        # be tied back to its seed plan from the event stream alone.
        "base_seed": int(base_seed),
    })

    # §Ship 24 (R1): reproducibility snapshot — code/project git SHAs,
    # config (raw sha256 + effective post-override dict), prompt
    # hashes, LLM model ids, package versions, seed plan. Written to
    # reports/run_context.json; failures are observable, never fatal.
    try:
        from sculptor.run_context import capture_run_context, write_run_context

        _rc = capture_run_context(
            project, config_path,
            parsed_config=cfg,
            behavior_goal=behavior_goal,
            base_seed=int(base_seed),
            iterations=int(iterations),
            start_iter=int(start_iter),
        )
        _rc_path = write_run_context(paths["reports"], _rc)
        _emit_event({
            "type": "run_context_captured",
            "path": str(_rc_path),
            "code_sha": (_rc.get("code_git") or {}).get("sha"),
            "code_dirty": (_rc.get("code_git") or {}).get("dirty"),
            "config_sha256": (_rc.get("config") or {}).get("sha256"),
            "base_seed": int(base_seed),
        })
    except Exception as _rc_err:  # noqa: BLE001 — capture must not kill a run
        _emit_event({
            "type": "run_context_capture_failed",
            "error": f"{type(_rc_err).__name__}: {_rc_err}",
        })

    # §Ship 15: `init_ckpt` normalized at function entry. Applies ONLY
    # to the FIRST iter of this run (iter == start_iter). Subsequent
    # iters start fresh; iter-to-iter warm-start within a single
    # sculpt_run is a separate behavioral decision deferred past Ship 16.
    result = SculptRunResult(iterations_run=0)
    #: §Ship 33: iters since the best fitness improved (plateau early-stop).
    iters_since_best = 0
    #: §LAW 11: per-iter naturalness steer-factor, aligned with
    #: result.fitness_history, for the Goodhart-onset early-stop.
    naturalness_history: list[float] = []
    #: §Ship 36 (F1): edit base for the NEXT iter when the last one regressed
    #: fitness (steer mode only); None = build forward from the latest reward.
    revert_base: Optional[Path] = None
    #: §Ship 39 (H1): human feedback to inject into the NEXT iter's diagnose
    #: (captured at the interactive pause). None = no note this iter.
    pending_human_note: Optional[str] = None
    _control_path: Optional[Path] = Path(control_file) if control_file else None

    # §RL_SCULPTOR_AUDIT §4.4 (loop 4c, gap #4): goal-conditioned starter.
    # If iteration 0 is about to train the pristine `sculpt init` constant
    # alive-bonus template and this run carries a behavior goal, generate a
    # goal-conditioned v1 first (one bounded LLM call; every failure path
    # keeps the template and proceeds). Skipped on resume (start_iter > 0
    # implies real rewards exist) and in dry-run (no LLM).
    if start_iter == 0 and not dry_run and str(behavior_goal or "").strip():
        # §env generalization 2/4: goal-conditioned env spec first (the
        # environment the seeded reward will train in), then the seeded
        # reward. Both are bounded-LLM, once-per-project, fail-open.
        _maybe_seed_env_spec(
            project=project, behavior_goal=behavior_goal, adapter=adapter)
        _maybe_seed_goal_reward(
            rewards_dir=rewards_dir, behavior_goal=behavior_goal,
            adapter=adapter, kg_store=kg_store)

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
                fitness_fn=fitness_fn,
                prior_fitness=(
                    {"best_so_far": result.best_fitness,
                     # §RL_SCULPTOR_AUDIT §4.4 (loop 4b): best PROGRESS +
                     # the best iter's dir so _run_one_iter can source the
                     # edit anti-collapse replay from the best-so-far
                     # rollout when this iter strictly regressed.
                     "best_progress": result.best_progress,
                     "best_iter_dir": (
                         str(runs_dir / f"iter_{result.best_fitness_iter}")
                         if result.best_fitness_iter is not None else None),
                     "last": (result.fitness_history[-1]
                              if result.fitness_history else None)}
                    if fitness_fn is not None else None
                ),
                fitness_observe_only=fitness_observe_only,
                revert_base=revert_base,
                # §env generalization 3/4: revert the env spec alongside
                # the reward — the (reward, env) pair is the training
                # config keep-best/revert operates on.
                env_revert_version=(
                    result.best_env_spec if revert_base is not None else None),
                human_note=pending_human_note,
            )
            elapsed = time.time() - t0
            result.completed_iters.append(outcome)
            result.primary_metric_history.append(
                outcome.primary_metric if outcome.primary_metric is not None else 0.0)
            result.iterations_run += 1

            # §Ship 33: track objective fitness + best-by-fitness selection.
            strictly_regressed = False
            if fitness_fn is not None:
                fit = outcome.fitness if outcome.fitness is not None else 0.0
                result.fitness_history.append(fit)   # TRUE task fitness (display / A-B)
                result.progress_history.append(       # §Convergence: dense channel
                    outcome.progress if outcome.progress is not None else 0.0)
                naturalness_history.append(           # §LAW 11: aligned naturalness trend
                    outcome.naturalness_factor
                    if outcome.naturalness_factor is not None else 1.0)
                # §LAW 7: best-by-fitness selection uses the naturalness-GATED
                # steer value, so a joint-limit exploit (steer 0) or a severe-
                # unnatural iter (down-weighted) cannot become the steered-toward
                # checkpoint. The recorded history above stays the true score.
                steer = (outcome.steer_fitness
                         if outcome.steer_fitness is not None else fit)
                # §Convergence (RL_SCULPTOR_AUDIT §4.1): LEXICOGRAPHIC key —
                # spec fitness decides; the dense progress channel ONLY breaks
                # ties. Below the completion gate (spec 0.0 everywhere, the
                # tuck-jump failure) progress is the only ranking signal; the
                # moment any iter clears the gate, spec dominates again.
                sprog = (outcome.steer_progress
                         if outcome.steer_progress is not None
                         else (outcome.progress
                               if outcome.progress is not None else 0.0))
                cur_key = (steer, sprog)
                best_key = ((result.best_fitness, result.best_progress or 0.0)
                            if result.best_fitness is not None else None)
                # §Selection statistics: `progress_epsilon` (default 1e-5,
                # config [iteration]) is the measured noise band of the
                # dense channel — a tie-break must CLEAR it to mint a new
                # best, and a dip inside it is a tie, not a regression
                # (E2E run 2: two best-selections were decided by
                # sub-display-precision seed noise; audit §6). Spec
                # fitness comparisons are unaffected. 0.0 = old behavior.
                if best_key is None or _lex_improved(
                        cur_key, best_key, _progress_epsilon):
                    result.best_fitness = steer
                    result.best_progress = sprog
                    result.best_fitness_iter = outcome.iter_index
                    # §env generalization 3/4: the env half of the best
                    # (reward, env) training config — revert + end-of-run
                    # selection restore BOTH together.
                    result.best_env_spec = outcome.env_spec_trained
                    iters_since_best = 0
                else:
                    iters_since_best += 1
                    strictly_regressed = _lex_regressed(
                        cur_key, best_key, _progress_epsilon)

            # §Ship 36 (F1): set the next iter's edit base. If this iter set a
            # new best, keep building forward (revert_base=None — the best IS
            # the latest). If it STRICTLY regressed, point the next iter at the
            # best-so-far reward so the search doesn't compound a bad edit.
            # §Convergence deadlock fix: a TIE with best (common when the
            # metric reads 0.0 below its completion gate) is NOT a regression —
            # reverting on ties pinned tuck-jump to v0 forever, retraining the
            # same reward while every corrective edit was generated but never
            # trained. Ties keep building forward; patience still counts them.
            # Steer mode only; observe-only never reverts (signal stays passive).
            if (fitness_fn is not None and not fitness_observe_only
                    and fitness_revert):
                revert_base = (
                    result.best_reward_path
                    if strictly_regressed and result.best_reward_path is not None
                    else None
                )

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

            # §Ship 33: fitness-driven early-stop (only when fitness_fn
            # is supplied AND no callback already stopped us). Target hit
            # → stop at success; `fitness_patience` iters with no new best
            # → stop wasting GPU climbing a plateau (the go1_trot runs
            # over-iterated past their best: best 0.247 but final 0.166).
            if (fitness_fn is not None and not fitness_observe_only
                    and not result.early_stopped):
                reason = None
                if (fitness_target is not None
                        and result.best_fitness is not None
                        and result.best_fitness >= fitness_target):
                    reason = (f"fitness_target {fitness_target} reached "
                              f"(best={result.best_fitness:.3f} at iter "
                              f"{result.best_fitness_iter})")
                elif iters_since_best >= max(1, fitness_patience):
                    reason = (f"fitness plateau: no new best in "
                              f"{iters_since_best} iters "
                              f"(best={result.best_fitness:.3f} at iter "
                              f"{result.best_fitness_iter})")
                # §LAW 11: Goodhart-onset — the objective metric is rising while
                # naturalness falls (the policy is gaming the proxy). Lock the
                # prior best rather than optimize into the exploit. Checked only
                # if no target/plateau reason already fired.
                onset_source = "fitness"
                if reason is None:
                    onset = detect_goodhart_onset(
                        result.fitness_history, naturalness_history)
                    if onset:
                        reason, onset_source = onset, "goodhart_onset"
                if reason:
                    result.early_stopped = True
                    result.early_stop_reason = reason
                    _emit_event({
                        "type": "early_stop",
                        "at_iter": outcome.iter_index,
                        "reason": reason,
                        "source": onset_source,
                    })
                    break

            # §Ship 39 (H1): interactive pause-for-feedback at the iteration
            # boundary — only when a control file is wired, we're not already
            # stopping, and this isn't the last iteration (nothing left to
            # steer). Default (no control file) is fully automated and
            # byte-identical to before. A buggy/locked file degrades to auto.
            if (_control_path is not None and not result.early_stopped
                    and i < end_iter - 1):
                try:
                    _note, _stop = _pause_for_feedback(
                        _control_path, outcome.iter_index,
                        timeout=feedback_timeout,
                        poll_interval=feedback_poll_interval)
                except Exception as _fb_err:  # noqa: BLE001 — never block on a bug
                    _note, _stop = None, False
                    sys.stderr.write(
                        f"[sculpt] feedback pause raised "
                        f"{type(_fb_err).__name__}: {_fb_err} — continuing\n")
                if _stop:
                    result.early_stopped = True
                    result.early_stop_reason = "stopped by user (interactive)"
                    _emit_event({
                        "type": "early_stop",
                        "at_iter": outcome.iter_index,
                        "reason": result.early_stop_reason,
                        "source": "user",
                    })
                    break
                pending_human_note = _note

    finally:
        if kg_store is not None:
            kg_store.close()

    # §Ship 33: keep the BEST-by-fitness reward as current.py (Eureka's
    # 'return best across iterations'). Only when fitness was tracked and
    # the best iter isn't already the last produced reward; otherwise the
    # blind default (current.py = latest v<n>) stands untouched.
    # §Ship 35: observe-only mode NEVER repoints — the fitness signal must
    # not influence which reward is kept (that's the whole point of observe).
    if (fitness_fn is not None and not fitness_observe_only
            and result.best_fitness_iter is not None):
        best_path = result.best_reward_path
        if best_path is not None and best_path.is_file():
            try:
                _write_current_reexport(rewards_dir, best_path)
                _emit_event({
                    "type": "best_reward_selected",
                    "iter": int(result.best_fitness_iter),
                    "fitness": (round(result.best_fitness, 5)
                                if result.best_fitness is not None else None),
                    "progress": (round(result.best_progress, 5)
                                 if result.best_progress is not None else None),
                    "reward": best_path.name,
                })
            except Exception as e:  # noqa: BLE001 — selection is best-effort
                sys.stderr.write(
                    f"[sculpt] best-by-fitness current.py rewrite failed: "
                    f"{type(e).__name__}: {e}\n"
                )
        # §env generalization 3/4: keep the env half of the best training
        # config too — repoint env/current.json at the best iter's spec
        # version so the project resumes (and re-trains) under it.
        if result.best_env_spec is not None:
            try:
                from sculptor.env_spec import (
                    read_current_env_spec, repoint_env_current)

                _env_dir = project / "env"
                _cur = read_current_env_spec(_env_dir)
                _cur_v = ((_cur.get("meta") or {}).get("version")
                          if _cur else None)
                if _cur_v != result.best_env_spec:
                    repoint_env_current(_env_dir, result.best_env_spec)
                    _emit_event({
                        "type": "best_env_spec_selected",
                        "iter": int(result.best_fitness_iter),
                        "env_spec": result.best_env_spec,
                    })
            except Exception as e:  # noqa: BLE001 — selection is best-effort
                sys.stderr.write(
                    f"[sculpt] best env-spec repoint failed: "
                    f"{type(e).__name__}: {e}\n")

    # §Selection statistics (RESEARCH_GAP_ANALYSIS §7.2d): re-evaluate the
    # KEPT best on fresh rollout seeds never used for selection. The
    # selected best_fitness is a max statistic over the run's evaluations
    # (Empirical Design in RL flags report-of-max as a pitfall) — this
    # records the unbiased number beside it. Advisory: never changes the
    # selection, never raises; `fresh_eval_seeds = 0` disables.
    if (fitness_fn is not None and not fitness_observe_only
            and result.best_fitness_iter is not None):
        try:
            _fresh_raw = (cfg.get("iteration") or {}).get(
                "fresh_eval_seeds", 1)
            _fresh_n = max(0, int(1 if _fresh_raw is None else _fresh_raw))
        except Exception:  # noqa: BLE001
            _fresh_n = 1
        _detail_dir_fn = getattr(fitness_fn, "detail_dir", None)
        _best_out = next(
            (o for o in result.completed_iters
             if o.iter_index == result.best_fitness_iter), None)
        if (_fresh_n > 0 and _detail_dir_fn is not None
                and _best_out is not None
                and _best_out.checkpoint_path is not None
                and Path(_best_out.checkpoint_path).is_file()):
            _fresh_scores: list[float] = []
            for _j in range(_fresh_n):
                _fresh_dir = _best_out.iter_dir / f"rollout_fresh_{_j}"
                try:
                    _fresh_dir.mkdir(exist_ok=True)
                    _rollout_or_resume(
                        adapter=adapter,
                        iter_index=_best_out.iter_index,
                        rollout_dir=_fresh_dir,
                        checkpoint_path=Path(_best_out.checkpoint_path),
                        n_episodes=int((cfg.get("iteration") or {}).get(
                            "rollout_episodes", 6)),
                        # Disjoint from training seeds and the in-loop
                        # eval band (10k+): deterministic held-out seeds.
                        seed=90_001 + 131 * _j,
                    )
                    _d = _detail_dir_fn(_fresh_dir) or {}
                    _fresh_scores.append(
                        float(_d.get("spec_score", 0.0) or 0.0))
                except Exception as e:  # noqa: BLE001 — per-seed best-effort
                    sys.stderr.write(
                        f"[sculpt] fresh eval seed {_j} skipped — "
                        f"{type(e).__name__}: {e}\n")
            if _fresh_scores:
                result.fresh_fitness_per_seed = _fresh_scores
                result.best_fitness_fresh = _median(_fresh_scores)
                _emit_event({
                    "type": "best_fresh_eval",
                    "iter": int(result.best_fitness_iter),
                    "fitness_fresh": round(result.best_fitness_fresh, 5),
                    "per_seed": [round(v, 5) for v in _fresh_scores],
                    # Side-by-side: the (max-statistic) selected value.
                    "selected_fitness": (
                        round(result.best_fitness, 5)
                        if result.best_fitness is not None else None),
                })

    # §Ship 37: persist run-learnings to the KG case-memory so future
    # diagnoses can avoid repeating past mistakes ("the same failure can't
    # happen twice"). §2026-07-03: EVERY run records now (was: fitness-
    # tracked only) — a blind run's "what was tried" is itself memory;
    # verdicts stay 'unknown' without a measurable signal. Best-effort —
    # reopens its own store (the loop's kg_store is closed in the finally
    # above) and never lets a logging failure affect the run.
    if not no_kg and result.completed_iters:
        try:
            from sculptor.kg.cases import record_run_cases
            from sculptor.kg.store import SculptorKG

            # §usage-based enrichment: which papers each iteration's KEPT
            # reward actually cited — "helped" verdicts bump the cited
            # techniques' useful_citations (retrieval learns from what
            # got accepted, not just what was retrieved). Best-effort:
            # an unreadable reward file skips that iter's references.
            _iter_refs: dict[int, list[str]] = {}
            for _oc in result.completed_iters:
                _rp = getattr(_oc, "reward_path_trained", None) or getattr(
                    _oc, "reward_path_after", None)
                if not _rp or not Path(_rp).is_file():
                    continue
                try:
                    from sculptor.edit import (
                        _current_reward_references,
                        _load_reward_module,
                    )
                    _ids = [
                        str(r.get("arxiv_id"))
                        for r in _current_reward_references(
                            _load_reward_module(Path(_rp)))
                        if r.get("arxiv_id")]
                    if _ids:
                        _iter_refs[int(_oc.iter_index)] = _ids
                except Exception:  # noqa: BLE001 — enrichment is advisory
                    continue

            _cstore = SculptorKG()
            try:
                _n_cases = record_run_cases(
                    _cstore, task=behavior_goal, result=result,
                    iter_references=_iter_refs or None)
                if _n_cases:
                    _emit_event({"type": "run_cases_recorded", "count": _n_cases})
            finally:
                _cstore.close()
        except Exception as e:  # noqa: BLE001 — case logging is best-effort
            sys.stderr.write(
                f"[sculpt] run-case logging failed: {type(e).__name__}: {e}\n")

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


def _is_pristine_starter_reward(reward_path: Path) -> bool:
    """§RL_SCULPTOR_AUDIT §4.4 (loop 4c): True when `reward_path` is the
    untouched `sculpt init` starter template — a constant alive-bonus.
    Detected by its REWARD_SPEC signature (author 'human', version 'v0',
    exactly one hyperparameter `alive_bonus`, 'Starter reward' in the
    description) rather than byte comparison, so a template that merely
    aged across sculptor versions still counts. Any real reward — human-
    written or sculpted — fails at least one check. Unreadable/broken
    module → False (never trigger generation on something we can't read)."""
    try:
        from sculptor.edit import _load_reward_module

        spec = getattr(_load_reward_module(Path(reward_path)), "REWARD_SPEC", None)
        if not isinstance(spec, dict):
            return False
        return (
            str(spec.get("author", "")) == "human"
            and str(spec.get("version", "")) == "v0"
            and set((spec.get("hyperparameters") or {}).keys()) == {"alive_bonus"}
            and "Starter reward" in str(spec.get("description", ""))
        )
    except Exception:  # noqa: BLE001
        return False


# Goal text is clipped to leave room for the fixed guidance inside
# apply_prompt_edit's 2000-char prompt ceiling.
_SEED_GOAL_CLIP = 900


def _seed_reward_prompt(behavior_goal: str) -> str:
    """User-prompt for the goal-conditioned starter generation. Carries
    the goal + iteration-0 design rules; the edit_rewriter system prompt
    (incl. the net-positive-living / progress-preservation laws) and the
    reward contract ride along via apply_prompt_edit."""
    goal = " ".join(str(behavior_goal).split())[:_SEED_GOAL_CLIP]
    return (
        f"GOAL-CONDITIONED STARTER. Behavior goal: {goal}\n"
        "The current v0 is the scaffold placeholder — a constant "
        "alive-bonus with zero gradient (policies trained on it stand "
        "still and its high mean_return makes it an attractor). Replace "
        "it with the INITIAL shaping reward for this goal, designed for "
        "iteration-0 training from a randomly-initialized policy:\n"
        "- decompose the goal into 2-4 physical phases and give EACH a "
        "dense, bounded term (saturating ramps anchored at values a "
        "fresh policy already reaches — no cliff thresholds above its "
        "reach);\n"
        "- keep a small alive bonus (~0.1) so shaping dominates, and "
        "zero every bonus when info['fallen'] fires;\n"
        "- keep the per-step total >= 0 in ordinary non-fallen poses;\n"
        "- read ONLY the adapter's expected_info_keys / state schema."
    )


def _maybe_seed_goal_reward(
    *,
    rewards_dir: Path,
    behavior_goal: str,
    adapter: SculptorAdapter,
    kg_store=None,
    client=None,
) -> Optional[Path]:
    """§RL_SCULPTOR_AUDIT §4.4 (loop 4c, gap #4): goal-conditioned
    starter. When the reward that iteration 0 would train is still the
    pristine `sculpt init` constant-alive-bonus template and the run has
    a behavior goal, generate `v1.py` from the goal via
    `apply_prompt_edit` (full post-flight stack: probes, variance
    pre-screen — which rejects a still-constant generation — and one
    internal retry). Exactly one LLM call (+1 retry) once per project.

    Deliberately NOT done at `sculpt init` time: project creation must
    stay instant and API-key-free; the first run already spends LLM
    calls, so the seed rides on it. ANY failure (validation, no API
    key, network) logs + returns None and the run proceeds on the
    template — the dead-reward pre-screen still guards later edits."""
    latest_n, latest = _find_latest_reward_version(rewards_dir)
    if latest_n != 0 or not _is_pristine_starter_reward(latest):
        return None
    _emit_event({
        "type": "seed_reward_started",
        "behavior_goal": str(behavior_goal)[:200],
    })
    try:
        new_path = apply_prompt_edit(
            current_reward_path=latest,
            user_prompt=_seed_reward_prompt(behavior_goal),
            new_iter_id="v1",
            reward_contract=adapter.reward_contract(),
            kg_store=kg_store,
            client=client,
        )
    except Exception as e:  # noqa: BLE001 — seed is best-effort, run continues
        sys.stderr.write(
            f"[sculpt] goal-conditioned starter generation failed "
            f"({type(e).__name__}: {e}) — iteration 0 trains the v0 "
            f"template instead.\n")
        _emit_event({
            "type": "seed_reward_failed",
            "error": f"{type(e).__name__}: {e}",
        })
        return None
    _emit_event({
        "type": "seed_reward_generated",
        "reward": new_path.name,
    })
    print(f"[sculpt] goal-conditioned starter written: {new_path.name} "
          f"(replaces the constant alive-bonus template for iter 0)",
          flush=True)
    return new_path

def _maybe_seed_env_spec(
    *,
    project: Path,
    behavior_goal: str,
    adapter: SculptorAdapter,
    client=None,
) -> Optional[Path]:
    """§RL_SCULPTOR_AUDIT (env generalization 2/4): goal-conditioned
    environment adaptation at first run. When the project has no env
    spec yet — and made no explicit env choice in config.toml (an
    `env_profile` or `env_spec_path` there is respected, not
    overridden) — generate one from the behavior goal via the bounded
    `sculptor.env_gen` pipeline (≤2 LLM calls, full validate_env_spec
    gate) and activate it for THIS run by pointing the already-built
    adapter at `env/current.json`. Subsequent runs pick it up via the
    `load_adapter` convention. ANY failure (no API key, network,
    validation twice) logs + emits `env_spec_failed` and the run
    proceeds on task defaults — never blocks.

    Same rationale as `_maybe_seed_goal_reward` for running at first
    run rather than `sculpt init`: project creation stays instant and
    API-key-free; the first run already spends LLM calls."""
    if not hasattr(adapter, "env_spec_path"):
        return None   # adapter family without env-spec support (gym_sb3)
    if getattr(adapter, "env_spec_path", "") or getattr(
            adapter, "env_profile", ""):
        return None   # explicit config choice stands
    env_dir = project / "env"
    if (env_dir / "current.json").is_file():
        return None   # already generated (or hand-written)
    task_id = str(getattr(adapter, "task_id", ""))
    _emit_event({
        "type": "env_spec_started",
        "behavior_goal": str(behavior_goal)[:200],
        "task_id": task_id,
    })
    try:
        from sculptor.env_gen import generate_env_spec
        from sculptor.env_spec import write_env_spec_version

        spec = generate_env_spec(
            behavior_goal=behavior_goal, task_id=task_id, client=client)
        path = write_env_spec_version(env_dir, spec)
    except Exception as e:  # noqa: BLE001 — generation is best-effort
        sys.stderr.write(
            f"[sculpt] env-spec generation failed "
            f"({type(e).__name__}: {e}) — training on task defaults.\n")
        _emit_event({
            "type": "env_spec_failed",
            "error": f"{type(e).__name__}: {e}",
        })
        return None
    adapter.env_spec_path = str((env_dir / "current.json").resolve())
    _emit_event({
        "type": "env_spec_generated",
        "version": path.stem,
        "shared": spec.get("shared") or {},
        "train": spec.get("train") or {},
        "reasoning": (spec.get("meta") or {}).get("reasoning", "")[:500],
    })
    print(f"[sculpt] goal-conditioned env spec written: env/{path.name} "
          f"(shared={sorted((spec.get('shared') or {}).keys())}, "
          f"train={sorted((spec.get('train') or {}).keys())})",
          flush=True)
    return path


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
# Legacy compatibility fields. Metric-plateau auto-kill is disabled and
# these values are ignored by sculpt_run; keep them so older UI/API clients
# and config files continue to parse without migration.
early_stop_enabled = false
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


def _toml_scalar(v: Any) -> str:
    """Serialize a primitive/list as a TOML literal. Mirrors the
    inline-table style `sculpt_init` / the backend's set_adapter_section
    emit (sculptor deliberately avoids a `tomli_w` dependency; the
    configs we touch are flat/inline-table only)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    if v is None:
        return '""'
    return _toml_scalar(str(v))


def _apply_stage_run_overrides(
    stage_config: Path,
    *,
    edit_candidates: Optional[int] = None,
    num_envs: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """§MISSION_RUN_PARITY: inject per-launch knobs that have no
    sculpt_run keyword into the stage's config.toml AFTER it's
    scaffolded + inherited the parent's sections.

      * `edit_candidates` → an `edit_candidates = N` line under
        `[iteration]` (read by `_run_one_iter` via
        `iter_cfg.get("edit_candidates", 1)`). Flat key, upserted.
      * `num_envs` / `device` → the `[adapter].config` inline table
        (mjlab reads these; they are NOT sculpt_run kwargs). Parsed via
        tomllib + re-emitted so the mjlab inline-table form survives.

    All-None is a no-op (a plain mission run stays byte-identical). Best
    effort: a malformed config is left untouched — load_adapter surfaces
    the original error downstream, exactly as before this override."""
    if edit_candidates is None and num_envs is None and device is None:
        return
    if not stage_config.is_file():
        return
    try:
        text = stage_config.read_text(encoding="utf-8")
    except OSError:
        return

    # ── [iteration].edit_candidates (flat key upsert) ──
    if edit_candidates is not None:
        body = _extract_toml_section(text, "iteration")
        if body is None:
            # No [iteration] section (unusual — the template has one).
            # Append a fresh one so the knob is honored.
            text = (
                text.rstrip("\n")
                + f"\n\n[iteration]\nedit_candidates = {int(edit_candidates)}\n"
            )
        else:
            line = f"edit_candidates = {int(edit_candidates)}\n"
            pat = re.compile(r"^\s*edit_candidates\s*=.*$", re.MULTILINE)
            if pat.search(body):
                new_body = pat.sub(line.rstrip("\n"), body, count=1)
            else:
                new_body = body.rstrip("\n") + "\n" + line
            text = _replace_toml_section(text, "iteration", new_body)

    # ── [adapter].config.{num_envs,device} (inline-table re-emit) ──
    if num_envs is not None or device is not None:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover — py310
                import tomli as tomllib  # type: ignore[no-redef]
            cfg = tomllib.loads(text)
            adapter = cfg.get("adapter") or {}
            adapter_class = adapter.get("class") or ""
            inner = dict(adapter.get("config") or {})
            if num_envs is not None:
                inner["num_envs"] = int(num_envs)
            if device is not None:
                inner["device"] = str(device)
            inline = (
                "{ " + ", ".join(
                    f"{k} = {_toml_scalar(v)}" for k, v in inner.items()
                ) + " }"
                if inner else "{}"
            )
            new_block = (
                f'[adapter]\nclass = "{adapter_class}"\nconfig = {inline}\n'
            )
            # Replace the entire [adapter] section (header → next section
            # header or EOF). Mirrors project_store.set_adapter_section.
            pat = re.compile(
                r"^\[adapter\].*?(?=^\[[^\]]+\]|\Z)",
                re.MULTILINE | re.DOTALL,
            )
            text, n = pat.subn(new_block + "\n", text, count=1)
            if n == 0:
                text = text.rstrip() + "\n\n" + new_block
        except Exception:  # noqa: BLE001 — malformed config: leave as-is.
            pass

    try:
        stage_config.write_text(text, encoding="utf-8")
    except OSError:
        pass


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


def _iter_checkpoint(iter_dir: Path) -> Optional[Path]:
    """The checkpoint file (mjlab `.pt` / gym_sb3 `.zip`) in an iter dir,
    or None. Size>0 guarded (a half-written checkpoint is not warm-
    startable)."""
    for ext in ("pt", "zip"):
        p = Path(iter_dir) / f"checkpoint.{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _resolve_stage_final_checkpoint(
    sculpt_result: "SculptRunResult",
) -> Optional[Path]:
    """Pick the checkpoint path from the last completed iter.

    mjlab writes `checkpoint.pt`, gym_sb3 writes `checkpoint.zip`. Returns
    None if the last iter produced no checkpoint. Retained for callers
    that still want the last-iter policy; stage finalization now uses
    `_select_stage_final_iter` (keep-best) instead.
    """
    if not sculpt_result.completed_iters:
        return None
    return _iter_checkpoint(sculpt_result.completed_iters[-1].iter_dir)


def _iter_fitness_key(o: "IterOutcome") -> tuple[float, float, float]:
    """Lexicographic keep-best key for an iteration, mirroring the loop's
    own `(steer_fitness, steer_progress)` selection with primary_metric as
    the final tiebreak. Missing channels sort lowest."""
    NEG = float("-inf")
    sf = o.steer_fitness if o.steer_fitness is not None else (
        o.fitness if o.fitness is not None else NEG)
    sp = o.steer_progress if o.steer_progress is not None else (
        o.progress if o.progress is not None else NEG)
    pm = o.primary_metric if o.primary_metric is not None else NEG
    return (sf, sp, pm)


def _select_stage_final_iter(
    candidates: "list[IterOutcome]",
    stage,
) -> "tuple[Optional[IterOutcome], bool, str, Optional[str], bool]":
    """§keep-best finalization (B1). Choose the iteration a stage keeps as
    its final policy, so a late regression can't discard a good one.

    Returns `(selected, criterion_ok, source, criterion_error,
    error_is_missing_key)`.

    Rule: among candidates that have a checkpoint AND whose rollout
    satisfies the stage criterion, keep the one with the highest fitness
    key (ties → newest iter_index). If none pass, keep the
    highest-fitness candidate that has a checkpoint anyway (so warm-start /
    re-decomposition inherit the STRONGEST policy, not the last one) and
    report criterion failure. A genuine criterion bug (unsafe AST / dtype)
    raises the SAME error on every iter → no pass → surfaced as
    `criterion_errored`; a missing-key (metric absent) is a plain
    non-pass → `criterion_not_met`. Never raises mid-scan.
    """
    from sculptor.mission_runtime import (
        CriterionEvalError,
        CriterionMissingKeyError,
        _build_criterion_namespace,
        _evaluate_success_criterion,
    )

    # Dedup by iter_dir, keeping the latest occurrence (extension passes
    # can re-report the same on-disk iter).
    by_dir: "dict[str, IterOutcome]" = {}
    for o in candidates:
        by_dir[str(o.iter_dir)] = o
    with_ckpt = [o for o in by_dir.values() if _iter_checkpoint(o.iter_dir)]
    if not with_ckpt:
        return (None, False, "last", None, False)

    passing: "list[IterOutcome]" = []
    had_fitness = False
    last_error: Optional[str] = None   # any criterion error message seen
    only_missing_key = True            # False once a genuine (bug) error seen
    saw_any_error = False
    for o in with_ckpt:
        if o.steer_fitness is not None or o.fitness is not None:
            had_fitness = True
        try:
            ns = _build_criterion_namespace(
                iter_dir=Path(o.iter_dir), primary_metric=o.primary_metric)
            if _evaluate_success_criterion(stage.success_criterion, ns):
                passing.append(o)
        except CriterionEvalError as e:
            saw_any_error = True
            last_error = str(e)
            if not isinstance(e, CriterionMissingKeyError):
                only_missing_key = False
        except Exception:  # noqa: BLE001 — a flaky iter is a non-pass, not fatal
            pass

    if passing:
        winner = max(
            passing, key=lambda o: (_iter_fitness_key(o), o.iter_index))
        source = "criterion+fitness" if had_fitness else "criterion_newest"
        return (winner, True, source, None, False)

    # Nobody passed — keep the strongest policy for the successor anyway.
    winner = max(with_ckpt, key=lambda o: (_iter_fitness_key(o), o.iter_index))
    # Preserve the genuine-bug vs missing-key/plain-fail distinction the
    # old single-iter path drew: a broken criterion still surfaces as
    # criterion_errored; a missing key carries its message through to the
    # recoverable criterion_not_met so re-decompose feedback names the key.
    if saw_any_error and not only_missing_key:
        return (winner, False, "fitness_fallback", last_error, False)
    if saw_any_error:  # only missing-key errors
        return (winner, False, "fitness_fallback", last_error, True)
    # Criterion evaluated cleanly to False on every iter — a plain miss,
    # no missing key.
    return (winner, False, "fitness_fallback", None, False)


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


def _archive_mission_snapshot(
    mission_dir: Path, *, emit, final: bool = False,
) -> None:
    """§durable auto-save (A3): incrementally snapshot the mission into
    the restart-/delete-proof `saved/` archive (best+final+pinned
    checkpoints + every video/report/metric; heavy intermediates
    dropped — see sculptor.archive). Set `RS_AUTO_ARCHIVE=0` to disable.
    NEVER fatal — a failed archive emits an event and the mission
    proceeds. project_slug is the dir two levels up
    (`<project>/.missions/<mission>`)."""
    import os as _os
    if _os.environ.get("RS_AUTO_ARCHIVE", "1") != "1":
        return
    try:
        from sculptor.archive import archive_mission, saved_root

        project_slug = Path(mission_dir).parent.parent.name
        res = archive_mission(
            Path(mission_dir), saved_root(),
            project_slug=project_slug, incremental=True)
        emit({
            "type": "mission_archived" if final else "mission_stage_archived",
            "path": str(res.entry_dir),
            "total_bytes": int(getattr(res, "total_bytes", 0) or 0),
        })
    except Exception as e:  # noqa: BLE001 — archiving must never break a run
        emit({
            "type": "mission_archive_failed",
            "error": f"{type(e).__name__}: {e}",
        })


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
    fitness_metric: Optional[str] = None,
    fitness_target: Optional[float] = None,
    fitness_patience: int = 2,
    fitness_observe_only: bool = False,
    fitness_revert: bool = True,
    # §MISSION_RUN_PARITY: per-launch knobs mirrored from NewRunDialog and
    # applied uniformly to EVERY stage's training. All None = the stage's
    # inherited config wins (a plain mission run stays byte-identical).
    edit_candidates: Optional[int] = None,
    rollout_episodes: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    playback_speed: Optional[float] = None,
    render_width: Optional[int] = None,
    render_height: Optional[int] = None,
    num_envs: Optional[int] = None,
    device: Optional[str] = None,
):
    """§Ship-19d Goals A + B: optional adaptive iteration control.

    §Ship 34 — `fitness_metric` (a spec-metric name, e.g. "go1_trot")
    turns on fitness-in-the-loop for EVERY stage: each stage's sculpt_run
    is guided by that objective (best-by-fitness selection + plateau
    early-stop + fitness shown to the diagnoser). The SAME metric is
    applied uniformly to all stages, which is sound for single-skill
    missions (every curriculum sub-goal serves the one task) but not for
    a curriculum whose early stages have an unrelated objective. `None`
    (default) keeps the blind, criterion-only behavior. `fitness_target`
    /`fitness_patience` are forwarded to each stage's sculpt_run.

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
    # §llm provenance: decomposition-adjacent calls (redecompose, metric
    # resolution) archive to the mission dir until a stage's iteration
    # sink takes over in _run_one_iter (last-set-wins).
    set_llm_log_dir(mission_dir)

    # §Ship 34/38: resolve objective fitness fns. The mission-level
    # `fitness_metric` is the default; §Ship 38 lets each Stage carry its own
    # `steering_metric` that OVERRIDES it for that stage — what makes a true
    # multi-phase curriculum sound (a "balance on one leg" stage and a "kick"
    # stage want DIFFERENT objectives; Ship 34's uniform-per-mission metric
    # could not). Pre-resolve ALL distinct refs up front (fail-fast before any
    # GPU work) and cache them so a generated-metric module loads once.
    from sculptor.eval import resolve_fitness_fn as _resolve_fitness_fn
    from sculptor.mission_metrics import resolve_stage_metric_ref

    # §MISSION_METRIC_GRANULARITY: stage metrics generated at decompose
    # time are stored as MISSION-DIR-RELATIVE paths (portable mission.json,
    # inside the 128-char validator bound) — anchor them here before
    # resolution. Spec names + absolute paths pass through untouched.
    def _anchored(ref: Optional[str]) -> Optional[str]:
        return resolve_stage_metric_ref(ref, mission_dir) if ref else ref

    _fitness_fn_cache: dict[str, Callable[[Path], float]] = {}
    # Skip terminal stages that will never run again (resume): succeeded and
    # superseded — don't let a since-deleted generated-metric path on one of
    # them fail-fast the whole resumed mission (§Ship 38 review L2).
    for _ref in [_anchored(fitness_metric)] + [
        _anchored(getattr(s, "steering_metric", None)) for s in mission.stages
        if getattr(s, "status", None) not in ("succeeded", "superseded")
    ]:
        if _ref and _ref not in _fitness_fn_cache:
            _fitness_fn_cache[_ref] = _resolve_fitness_fn(_ref)  # fail-fast on bad ref

    def _fitness_fn_for_stage(stage) -> Optional[Callable[[Path], float]]:
        ref = _anchored(
            getattr(stage, "steering_metric", None) or fitness_metric)
        return _fitness_fn_cache.get(ref) if ref else None

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

        # §Ship 24 (R1): mission-level provenance.json — one context
        # per resume (code may differ between resumes) + one record
        # per executed stage, appended below. Never fatal.
        try:
            from sculptor.run_context import init_mission_provenance

            init_mission_provenance(
                mission_dir,
                goal=mission.goal,
                n_stages=len(mission.stages),
                seed=seed,
                extra={"adapter_short_name": adapter_short_name},
            )
        except Exception as _prov_err:  # noqa: BLE001
            _emit({
                "type": "run_context_capture_failed",
                "scope": "mission",
                "error": f"{type(_prov_err).__name__}: {_prov_err}",
            })

        # §Ship 17: refactored from for-loop to while-loop indexed by
        # `mission.current_stage_idx` so mid-iteration splices (sub-
        # stage redecomposition replacing the failed stage in place)
        # are safe. The for-loop's iterator would have cached the old
        # list state and skipped past the inserted sub-stages.
        all_succeeded = True
        _prov_record_warned = False  # §Ship 24: once-per-mission warning
        # §Ship 25b: decomposition-quality counters for telemetry.json.
        _n_stages_at_start = len(mission.stages)
        _redecompositions = 0
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

            # §mission-persistence increment 1: defensive resume guard.
            # By construction `current_stage_idx` is set to land on the
            # first CHILD immediately after a splice (never on the
            # retained, superseded parent itself), but a superseded
            # stage is terminal and must never be executed — advance
            # past it if somehow encountered (e.g. a hand-edited or
            # older-format mission.json pointing current_stage_idx at
            # it, or a future code path that doesn't maintain the
            # invariant). No StageResult is recorded for it — it
            # already got one (status="failed") on the run that
            # superseded it.
            if stage.status == "superseded":
                _emit({
                    "type": "stage_skipped",
                    "stage_name": stage.name,
                    "reason": "superseded",
                })
                mission.current_stage_idx += 1
                continue

            # §Ship 38: pick THIS stage's objective metric (its own
            # steering_metric, else the mission-level default) and surface it.
            _stage_metric_ref = (
                getattr(stage, "steering_metric", None) or fitness_metric)
            if _stage_metric_ref:
                _emit({
                    "type": "stage_fitness_metric",
                    "stage_name": stage.name,
                    "metric": _stage_metric_ref,
                    "source": ("stage"
                               if getattr(stage, "steering_metric", None)
                               else "mission"),
                })

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
                fitness_fn=_fitness_fn_for_stage(stage),
                fitness_target=fitness_target,
                fitness_patience=fitness_patience,
                fitness_observe_only=fitness_observe_only,
                fitness_revert=fitness_revert,
                # §MISSION_RUN_PARITY: forward the per-launch knobs.
                edit_candidates=edit_candidates,
                rollout_episodes=rollout_episodes,
                max_episode_steps=max_episode_steps,
                playback_speed=playback_speed,
                render_width=render_width,
                render_height=render_height,
                num_envs=num_envs,
                device=device,
            )
            result.stage_results.append(stage_res)

            # Persist mission.json AFTER every stage transition so
            # resume sees the latest state.
            _atomic_save_mission(mission, mission_dir)

            # §durable auto-save (A3): snapshot the mission into the
            # restart-/delete-proof archive AS EACH STAGE FINISHES, so a
            # user never loses footage to a later crash, regression, or
            # an accidental project delete. Incremental → one entry per
            # run. Retention (best+final+pinned checkpoints, all videos)
            # lives in sculptor.archive. NEVER fatal to the mission.
            _archive_mission_snapshot(mission_dir, emit=_emit)

            # §Ship 24 (R1): per-stage provenance record (never fatal,
            # but observable — warn ONCE per mission if records are
            # silently failing, e.g. a corrupted provenance.json).
            try:
                from sculptor.run_context import record_stage_in_provenance

                _rec_ok = record_stage_in_provenance(mission_dir, {
                    "stage_name": stage_res.stage_name,
                    "stage_idx": int(stage_idx),
                    "status": stage_res.status,
                    "iterations_used": stage_res.iterations_used,
                    "criterion_satisfied": stage_res.criterion_satisfied,
                    "last_iter_metric": stage_res.last_iter_metric,
                    "failure_reason": getattr(stage_res, "failure_reason", None),
                    "final_policy_path": str(stage_res.final_policy_path or ""),
                    "final_reward_path": str(stage_res.final_reward_path or ""),
                })
                if _rec_ok is None and not _prov_record_warned:
                    _prov_record_warned = True
                    _emit({
                        "type": "run_context_capture_failed",
                        "scope": "mission_stage_record",
                        "error": "provenance.json unreadable — stage "
                                 "records are not being persisted",
                    })
            except Exception:  # noqa: BLE001
                pass

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
                _redecompositions += 1  # §Ship 25b telemetry
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

        # §Ship 25b (H2): decomposition-quality telemetry (never
        # fatal). Written BEFORE the terminal mission event — consumers
        # treat mission_completed/halted_terminal as stream-end.
        _write_mission_telemetry(
            mission, mission_dir, result,
            n_stages_at_start=_n_stages_at_start,
            redecompositions=_redecompositions,
            emit=_emit,
        )

        # §durable auto-save (A3): final archive snapshot on BOTH the
        # completed AND halted paths — a halted mission still has footage
        # worth keeping (e.g. the jump succeeded but the landing stage
        # didn't). Incremental → folds into the same run entry.
        _archive_mission_snapshot(mission_dir, emit=_emit, final=True)

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


def _write_mission_telemetry(
    mission,
    mission_dir: Path,
    result,
    *,
    n_stages_at_start: int,
    redecompositions: int,
    emit: Any,
) -> None:
    """§Ship 25b (H2): decomposition-quality telemetry — stage counts,
    stage-success rate, redecompose rate, iteration spend — written to
    `<mission_dir>/telemetry.json` and aggregated per-project at
    `reports/mission_quality.json` (one record per mission slug,
    replaced on re-run). 22s changed decomposition behavior with no
    measurement; this is the measurement. Never fatal.

    NOT merged into reports/metric_history.json (the plan's original
    sketch): that file's `{primary_metric, history:[floats]}` shape is
    consumed by sculpt's own delta/early-stop logic — mixing mission
    dicts in would break readers. Separate file, additive surface.
    """
    try:
        from datetime import datetime, timezone

        from sculptor.run_context import write_json_atomic

        executed = list(result.stage_results)
        succeeded = [r for r in executed if r.status == "succeeded"]
        record = {
            "schema": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "mission_slug": Path(mission_dir).name,
            "goal": mission.goal,
            "n_stages_at_start": int(n_stages_at_start),
            "n_stages_final": len(mission.stages),
            "stages_executed": len(executed),
            "stages_succeeded": len(succeeded),
            "stage_success_rate": (
                round(len(succeeded) / len(executed), 4) if executed else None
            ),
            "redecompositions": int(redecompositions),
            "iterations_total": sum(
                int(r.iterations_used or 0) for r in executed
            ),
            "completed": bool(result.completed),
            "halted_reason": result.halted_reason,
            "per_stage": [
                {
                    "name": r.stage_name,
                    "status": r.status,
                    "iterations_used": r.iterations_used,
                    "criterion_satisfied": r.criterion_satisfied,
                }
                for r in executed
            ],
        }
        write_json_atomic(Path(mission_dir) / "telemetry.json", record)

        # Project-level aggregate — only under the real
        # `<project>/.missions/<slug>` layout (tests drive mission_run
        # against bare tmp dirs).
        agg_path = None
        if Path(mission_dir).parent.name == ".missions":
            project = Path(mission_dir).parent.parent
            agg_path = project / "reports" / "mission_quality.json"
            try:
                doc = json.loads(agg_path.read_text(encoding="utf-8"))
                if not isinstance(doc, dict) or not isinstance(
                    doc.get("missions"), list,
                ):
                    raise ValueError("bad shape")
            except Exception:  # noqa: BLE001 — first write or corrupt
                doc = {"schema": 1, "missions": []}
            doc["missions"] = [
                m for m in doc["missions"]
                if m.get("mission_slug") != record["mission_slug"]
            ] + [record]
            write_json_atomic(agg_path, doc)

        emit({
            "type": "mission_telemetry_written",
            "mission_slug": record["mission_slug"],
            "stage_success_rate": record["stage_success_rate"],
            "redecompositions": record["redecompositions"],
            "aggregate_path": str(agg_path) if agg_path else None,
        })
    except Exception as e:  # noqa: BLE001 — telemetry must not fail a mission
        emit({
            "type": "mission_telemetry_failed",
            "error": f"{type(e).__name__}: {e}",
        })


def _reconcile_stage_criterion_if_needed(
    *,
    stage,
    stage_dir: Path,
    adapter,
    mission,
    mission_dir: Path,
    emit: Any,
) -> None:
    """§Ship 25a (H1): probe the freshly-materialized reward and check
    every HARD `components['<name>']` reference in the stage criterion
    against the component names the reward actually produces. On
    mismatch: emit `criterion_keys_mismatch`, then ask Claude to
    rewrite the criterion onto real keys (`criterion_reconciled`) —
    catching at iter 0 what would otherwise burn the stage's whole
    budget before `criterion_not_met` fired at eval time (the 22q/22r
    silent-failure vector).

    NEVER raises: this runs inside the v1-materialization try block,
    and a reconciliation hiccup must not fail the stage — the runtime
    CriterionMissingKeyError path still catches survivors at eval.
    """
    try:
        from sculptor.mission_runtime import extract_components_keys

        keys = extract_components_keys(stage.success_criterion)
        if not keys:
            return

        _, latest_reward = _find_latest_reward_version(stage_dir / "rewards")
        probe_fn = getattr(adapter, "probe_component", None)
        if probe_fn is None:
            emit({
                "type": "criterion_keys_unverified",
                "stage_name": stage.name,
                "reason": "adapter has no probe_component",
            })
            return
        try:
            probe = probe_fn(latest_reward)
        except Exception as probe_err:  # noqa: BLE001
            emit({
                "type": "criterion_keys_unverified",
                "stage_name": stage.name,
                "reason": f"probe raised: {type(probe_err).__name__}: {probe_err}",
            })
            return
        if not getattr(probe, "ok", False) or not getattr(probe, "components", None):
            err = getattr(probe, "error", None)
            emit({
                "type": "criterion_keys_unverified",
                "stage_name": stage.name,
                "reason": (
                    f"probe failed: {err}" if err
                    else "probe returned no components"
                ),
            })
            return

        # Env intrinsic terms (`reward_term__*` in trajectory.npz) are
        # merged into `components` at EVAL time (Ship 22r) — a
        # redecomposed sub-stage's criterion may legitimately reference
        # them. The probe only sees the sculptor module's terms, so
        # union in the parent stage's observed env terms to avoid
        # rewriting a criterion that would actually have worked.
        env_terms = _parent_env_component_terms(mission, stage)
        available = sorted(set(probe.components) | env_terms)
        missing = sorted(keys - set(available))
        if not missing:
            emit({
                "type": "criterion_keys_validated",
                "stage_name": stage.name,
                "keys": sorted(keys),
                "env_terms_considered": sorted(env_terms),
            })
            return

        emit({
            "type": "criterion_keys_mismatch",
            "stage_name": stage.name,
            "missing": missing,
            "available": available,
            "criterion": stage.success_criterion,
        })

        import os as _os

        if not (_os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            # Observable skip — the mismatch event above already told
            # the user exactly what to fix by hand.
            emit({
                "type": "criterion_reconcile_skipped",
                "stage_name": stage.name,
                "reason": "no ANTHROPIC_API_KEY",
            })
            return

        from sculptor.decompose import reconcile_criterion

        old_criterion = stage.success_criterion
        new_criterion, rationale = reconcile_criterion(
            stage,
            missing_keys=missing,
            available_components=available,
        )
        stage.success_criterion = new_criterion
        try:
            _atomic_save_mission(mission, mission_dir)
        except BaseException:
            # Don't let a failed save leave memory≠disk: this run would
            # evaluate the NEW criterion while a later resume saw the
            # old one (and the failure event below would lie).
            stage.success_criterion = old_criterion
            raise
        emit({
            "type": "criterion_reconciled",
            "stage_name": stage.name,
            "old_criterion": old_criterion,
            "new_criterion": new_criterion,
            "missing": missing,
            "rationale": rationale,
        })
    except Exception as e:  # noqa: BLE001 — recoverable by design
        emit({
            "type": "criterion_reconcile_failed",
            "stage_name": stage.name,
            "error": f"{type(e).__name__}: {e}",
        })


def _parent_env_component_terms(mission, stage) -> set[str]:
    """Best-effort: env intrinsic reward-term names (`reward_term__*`)
    observed in the parent stage's latest rollout trajectory — the
    terms the eval-time namespace will merge into `components` anyway.
    Empty set on any failure (first stages have no parent; missing
    artifacts are normal)."""
    try:
        parent = getattr(stage, "parent_stage", None)
        if not parent:
            return set()
        runs = sorted(
            (mission.stage_dir(parent) / "runs").glob(
                "iter_*/rollout/trajectory.npz"
            )
        )
        if not runs:
            return set()
        import numpy as np

        prefix = "reward_term__"
        with np.load(runs[-1]) as z:
            return {
                k[len(prefix):] for k in z.files if k.startswith(prefix)
            }
    except Exception:  # noqa: BLE001 — advisory only
        return set()


#: Reference-library robot slugs the on-disk library is keyed by
#: (`sculptor.refs.library.clip_dir(robot, clip_id)`). Mirrors
#: `mission_metrics._ROBOT_SLUGS` — kept as a small local copy rather
#: than an import since `mission_metrics.py` is a metric-generation
#: module or this one has no other dependency on, and the mapping is a
#: single-source-of-truth-worthy but tiny (3-entry) constant.
_STAGE_REFERENCE_ROBOT_SLUGS: tuple[str, ...] = ("go2", "go1", "g1")


def _stage_reference_robot_slug(*, stage_dir: Path, project_root: Path) -> str:
    """Resolve the bare robot slug (`"g1"`, `"go1"`, ...) the reference
    library keys clips by, from the stage's (or the project's, as a
    fallback for a not-yet-scaffolded stage) `[adapter].config.task_id`
    — the same adapter/task_id-shaped string used everywhere else
    (`"Mjlab-Velocity-Flat-Unitree-G1"`). Unknown/unreadable/absent
    always falls back to `"g1"` (the only populated library robot as of
    §R1_BUILD_SPEC — mirrors `sculptor.refs`'s own `robot: str = "g1"`
    default throughout)."""
    task_id = ""
    for config_path in (stage_dir / "config.toml", project_root / "config.toml"):
        if not config_path.is_file():
            continue
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover
                import tomli as tomllib  # type: ignore[no-redef]
            with config_path.open("rb") as f:
                cfg = tomllib.load(f)
            task_id = str(
                ((cfg.get("adapter") or {}).get("config") or {})
                .get("task_id", "") or "")
        except Exception:  # noqa: BLE001 — best-effort resolution only
            task_id = ""
        if task_id:
            break
    hint = task_id.lower()
    for slug in _STAGE_REFERENCE_ROBOT_SLUGS:
        if slug in hint:
            return slug
    return "g1"


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
    fitness_fn: Optional[Callable[[Path], float]] = None,
    fitness_target: Optional[float] = None,
    fitness_patience: int = 2,
    fitness_observe_only: bool = False,
    fitness_revert: bool = True,
    # §MISSION_RUN_PARITY: per-launch knobs mirrored from NewRunDialog so
    # a mission run reaches parity with a standalone sculpt run. All
    # None = defer to the stage's inherited [iteration]/[adapter] config.
    edit_candidates: Optional[int] = None,
    rollout_episodes: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    playback_speed: Optional[float] = None,
    render_width: Optional[int] = None,
    render_height: Optional[int] = None,
    num_envs: Optional[int] = None,
    device: Optional[str] = None,
):
    """Helper called by `mission_run` per stage. Kept separate so the
    orchestrator stays readable — `mission_run` is about flow; this
    function is about "one stage, cradle to grave."

    §Ship 34: `fitness_fn`/`fitness_target`/`fitness_patience` are
    forwarded verbatim to this stage's sculpt_run call(s) so the stage's
    inner loop is fitness-guided (None = blind, unchanged).

    §MISSION_RUN_PARITY: the video/rollout knobs (`rollout_episodes`,
    `max_episode_steps`, `playback_speed`, `render_width`,
    `render_height`) are passed straight through to sculpt_run (which
    merges them into [iteration]). `edit_candidates` (best-of-K edit
    search) has no sculpt_run kwarg, so it is injected into the stage's
    [iteration] config after scaffolding; `num_envs`/`device` are
    injected into the stage's [adapter] config (they live there, not in
    sculpt_run's signature). All None = the stage's inherited config
    value wins, so a plain mission run is byte-identical."""
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
    # §Ship 20a: persist on the stage so the UI's `rounds X/Y` display
    # is correct even after the WS event window evicts stage_started.
    # Set BEFORE the emit so the on-disk save (which fires on stage
    # completion / failure / interrupt) captures the right value
    # regardless of where the stage's lifecycle terminates.
    stage.effective_max_iterations = effective_max_iterations
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

    # §MISSION_RUN_PARITY: inject per-launch knobs that sculpt_run has no
    # kwarg for (edit_candidates → [iteration]; num_envs/device →
    # [adapter].config). Runs on BOTH the fresh-scaffold and resume paths
    # so a resumed stage still honors the override. All-None = no-op.
    if edit_candidates is not None or num_envs is not None or device is not None:
        _apply_stage_run_overrides(
            stage_dir / "config.toml",
            edit_candidates=edit_candidates,
            num_envs=num_envs,
            device=device,
        )
        emit({
            "type": "stage_run_overrides_applied",
            "stage_name": stage.name,
            "edit_candidates": edit_candidates,
            "num_envs": num_envs,
            "device": device,
        })

    # 2.5 §JUMP_SCAFFOLD (generalized §REFERENCE_TRAJECTORY_PLAN §8 part
    # 2): decomposer-flagged reference-state initialization. Derive a
    # validated TRAIN-ONLY RSI curriculum from a reference clip and
    # persist it as this stage's next env-spec version before any
    # training. Clip preference, most to least specific:
    #   1. `stage.reference_clip_id` — a library clip explicitly
    #      ATTACHED to THIS stage (§R1_BUILD_SPEC decision 10; covers
    #      get-up clips, not just jumps — the whole point of this
    #      increment). Loaded from `sculptor.refs.library` the same way
    #      `mission_metrics._load_stage_reference` already does for
    #      metric generation.
    #   2. a real (converted mocap) clip at <project>/reference/jump.npz
    #      when present (pre-existing behavior, byte-identical).
    #   3. the analytic procedural jump (final fallback, pre-existing).
    # Failure at ANY step is non-fatal — RSI is curriculum assistance,
    # not correctness; the stage trains without it (falls through to the
    # next-lower-precedence source rather than aborting the stage).
    if getattr(stage, "needs_reference_rsi", False):
        try:
            from sculptor.env_spec import read_current_env_spec
            from sculptor.reference import (
                apply_reference_rsi,
                load_clip,
                make_procedural_jump_clip,
            )

            stage_env_dir = stage_dir / "env"
            current_spec = read_current_env_spec(stage_env_dir)
            already = (
                str(((current_spec or {}).get("meta") or {})
                    .get("source", "")).startswith("reference:"))
            if already:
                # Resume idempotency: don't stack a new env version per
                # resume — the reference curriculum is already in force.
                pass
            else:
                clip = None
                clip_src = None
                clip_load_error: Optional[str] = None
                stage_clip_id = getattr(stage, "reference_clip_id", None)
                if stage_clip_id:
                    try:
                        from sculptor.refs import library as refs_library

                        robot = _stage_reference_robot_slug(
                            stage_dir=stage_dir, project_root=(
                                mission_dir.parent.parent))
                        stage_clip_path = (
                            refs_library.clip_dir(robot, stage_clip_id)
                            / refs_library.CLIP_FILENAME)
                        clip = load_clip(stage_clip_path)
                        clip_src = f"library:{robot}/{stage_clip_id}"
                    except Exception as e:  # noqa: BLE001 — fall back below
                        clip_load_error = f"{type(e).__name__}: {e}"
                        emit({
                            "type": "stage_reference_clip_load_failed",
                            "stage_name": stage.name,
                            "reference_clip_id": stage_clip_id,
                            "error": clip_load_error,
                        })
                if clip is None:
                    project_root = mission_dir.parent.parent
                    clip_path = project_root / "reference" / "jump.npz"
                    if clip_path.is_file():
                        clip, clip_src = load_clip(clip_path), str(clip_path)
                    else:
                        clip, clip_src = (
                            make_procedural_jump_clip(), "procedural:jump")
                spec_path = apply_reference_rsi(stage_env_dir, clip)
                emit({
                    "type": "stage_reference_rsi_applied",
                    "stage_name": stage.name,
                    "clip": clip_src,
                    "env_spec": str(spec_path),
                    "stage_clip_load_error": clip_load_error,
                })
        except Exception as e:  # noqa: BLE001 — curriculum, not correctness
            emit({
                "type": "stage_reference_rsi_failed",
                "stage_name": stage.name,
                "error": f"{type(e).__name__}: {e}",
            })

    # 3. Materialize v1 from the stage's reward_seed_prompt.
    #
    # §edit-degrades-gracefully: `apply_prompt_edit` exhausts its repair
    # retries (attempt 1 + `RS_EDIT_REPAIR_RETRIES`, see sculptor/edit.py)
    # and raises `EditValidationError` when the LLM's generated reward
    # module is invalid on every attempt (e.g. SyntaxError, then a module
    # lacking `compute_reward` — a real incident, 2026-07-08). That is NOT
    # transient (the LLM calls themselves succeeded) and is not
    # recoverable by retrying `_run_one_stage` itself, so it's handled the
    # same way as any other v1-materialization failure below: the stage
    # (and ONLY this stage) is marked failed via `_fail_stage`, which
    # returns a normal `StageResult` rather than letting the exception
    # unwind past `_run_one_stage`. The caller (`mission_run`'s while-loop,
    # sculpt.py ~L3848) appends that StageResult, persists mission.json,
    # archives the mission snapshot, and writes the terminal
    # `mission_halted_terminal` event — i.e. a clean failed-stage terminal
    # state, not an uncaught-exception process crash. Earlier stages'
    # on-disk artifacts (rewards/rollouts/checkpoints) and their
    # `StageResult`s (already appended in prior loop iterations) are
    # untouched by this branch — see the "earlier-stage data preserved"
    # note at the top of `_run_one_stage` and `mission_run`'s per-stage
    # loop (sculpt.py ~L3812-3897).
    #
    # `EditValidationError` is caught explicitly (not just via the
    # `except Exception` below) so this specific, expected failure mode is
    # self-documenting at the boundary; any OTHER exception from this
    # block (adapter load errors, disk errors, etc.) still degrades the
    # same way via the broader catch.
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
            # §Ship 25a (H1): validate the criterion's hard
            # `components[...]` references against the JUST-
            # materialized reward, and reconcile mismatches at iter 0
            # instead of burning the stage budget. Never fatal.
            _reconcile_stage_criterion_if_needed(
                stage=stage,
                stage_dir=stage_dir,
                adapter=adapter_for_contract,
                mission=mission,
                mission_dir=mission_dir,
                emit=emit,
            )
    except EditValidationError as e:
        return _fail_stage(
            stage, "v1_materialization_errored",
            f"apply_prompt_edit failed: {type(e).__name__}: {e}",
            emit,
        )
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
        CriterionMissingKeyError,
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
            # §MISSION_RUN_PARITY: rollout-video knobs forwarded straight
            # through (sculpt_run merges them into [iteration]). None =
            # inherited config wins.
            max_episode_steps=max_episode_steps,
            playback_speed=playback_speed,
            render_width=render_width,
            render_height=render_height,
            rollout_episodes=rollout_episodes,
            init_policy_path=warm_start_path,
            per_iter_callback=per_iter_cb,
            fitness_fn=fitness_fn,
            fitness_target=fitness_target,
            fitness_patience=fitness_patience,
            fitness_observe_only=fitness_observe_only,
            fitness_revert=fitness_revert,
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

    # §keep-best finalization (B1): accumulate EVERY pass's iters. Each
    # Goal-B extension REPLACES `sculpt_result`, so the final result only
    # holds the extension's iters — but the best (e.g. jumping) policy may
    # live in an earlier pass. Union them so selection sees them all.
    all_iters: "list[IterOutcome]" = list(sculpt_result.completed_iters or [])

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
        # Any non-criterion sculpt_run early-stop means the stage already
        # chose to halt. Metric-plateau auto-kill no longer fires here.
        if (
            sculpt_result is not None
            and sculpt_result.early_stopped
        ):
            emit({
                "type": "stage_extension_skipped",
                "stage_name": stage.name,
                "reason": "sculpt_run_early_stop",
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
                # §MISSION_RUN_PARITY: same knobs on the Goal-B extension
                # passes so extended iters render identically.
                max_episode_steps=max_episode_steps,
                playback_speed=playback_speed,
                render_width=render_width,
                render_height=render_height,
                rollout_episodes=rollout_episodes,
                init_policy_path=None,  # resume picks up from local ckpt
                resume=True,
                per_iter_callback=per_iter_cb,
                fitness_fn=fitness_fn,
                fitness_target=fitness_target,
                fitness_patience=fitness_patience,
                fitness_observe_only=fitness_observe_only,
                fitness_revert=fitness_revert,
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
        # §keep-best (B1): fold this extension pass's iters into the
        # candidate pool.
        all_iters.extend(sculpt_result.completed_iters or [])

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

    # 5-6. §keep-best finalization (B1): select the BEST iter whose
    # rollout satisfies the criterion (highest fitness) — NOT the last
    # iter. This is what makes a jump stage survive a late collapse to
    # standing: the jumping policy is kept, evaluated, and warm-started
    # from, even when a later iter regressed. `all_iters` unions every
    # training + extension pass.
    if not all_iters:
        return _fail_stage(
            stage, "no_checkpoint",
            "training completed but produced no iterations.",
            emit,
        )
    (selected, criterion_ok, selection_source,
     criterion_error, err_missing_key) = _select_stage_final_iter(
        all_iters, stage)
    if selected is None:
        return _fail_stage(
            stage, "no_checkpoint",
            "training completed but no iteration produced a "
            "checkpoint.pt / .zip — successor stages can't warm-start "
            "from a missing file.",
            emit,
        )
    final_ckpt = _iter_checkpoint(selected.iter_dir)
    stage.final_policy_path = str(final_ckpt)
    stage.final_reward_path = (
        str(sculpt_result.final_reward_path)
        if sculpt_result.final_reward_path else None
    )
    stage.best_metric = selected.primary_metric
    stage.selected_iter_index = selected.iter_index
    stage.selection_source = selection_source
    emit({
        "type": "stage_final_selection",
        "stage_name": stage.name,
        "iter": selected.iter_index,
        "source": selection_source,
        "criterion_pass": bool(criterion_ok),
        "metric": selected.primary_metric,
        "fitness": selected.steer_fitness if selected.steer_fitness is not None
        else selected.fitness,
        "n_candidates": len({str(o.iter_dir) for o in all_iters}),
    })

    # A genuine criterion bug (unsafe AST / dtype) raised on every iter →
    # no pass with a real error surfaced → criterion_errored (unrecoverable,
    # as before). A plain miss / missing-key stays the recoverable
    # `criterion_not_met`. `final_policy_path` already points at the
    # strongest policy either way, so warm-start / re-decompose inherit it.
    if not criterion_ok and criterion_error is not None and not err_missing_key:
        emit({
            "type": "stage_criterion_evaluated",
            "stage_name": stage.name,
            "satisfied": False,
            "error": criterion_error,
            "missing_key": False,
        })
        return _fail_stage(
            stage, "criterion_errored", criterion_error, emit,
            criterion_error=criterion_error,
        )

    emit({
        "type": "stage_criterion_evaluated",
        "stage_name": stage.name,
        "criterion": stage.success_criterion,
        "satisfied": bool(criterion_ok),
        "last_iter_metric": selected.primary_metric,
        # True only when the miss was a missing key (recoverable, names the
        # key for re-decompose feedback); False for a clean criterion-false.
        "missing_key": err_missing_key,
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
            selected_iter_index=stage.selected_iter_index,
            selection_source=stage.selection_source,
        )

    return _fail_stage(
        stage, "criterion_not_met",
        f"success_criterion {stage.success_criterion!r} was not met by any "
        f"trained iteration (kept iter {selected.iter_index}, "
        f"metric={selected.primary_metric}).",
        emit,
        # Preserve the missing-key detail (if any) so re-decompose feedback
        # can point Claude at the real absent key.
        criterion_error=criterion_error,
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
    # §mission-persistence increment 1: persist the failure onto the
    # stage itself (mission.json), not just the ephemeral StageResult /
    # provenance.json. Without this the reason was lost the moment the
    # stage was superseded by redecomposition or the process exited.
    stage.failure_reason = reason
    stage.failure_detail = detail
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
        # §keep-best (B1): carry the selection (None on pre-selection
        # failures like training_errored / no_checkpoint).
        selected_iter_index=getattr(stage, "selected_iter_index", None),
        selection_source=getattr(stage, "selection_source", None),
    )


def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ── Ship 17: redecomposition splice ─────────────────────────────────
# Criterion-level failures are re-decomposable — re-authoring the stage
# (new sub-stages with corrected criteria / seed prompts) can fix them:
#   - criterion_not_met:  the goal threshold wasn't reached. A criterion
#     that referenced a metric the reward never produced is rerouted here
#     (see _run_one_stage step 6) — the quantity was absent, so the goal
#     was not met, and a redecompose can pick a real key / add the term.
#   - criterion_errored:  the criterion itself was malformed (bad dtype,
#     unsafe AST) and slipped past decompose-time validation; a rewrite
#     can correct it.
# Both are bounded by the 1-attempt redecomposition cap, so a persistently
# broken/unmet stage still halts after ONE recovery try rather than wasting
# the whole multi-hour mission on the first stumble. Infrastructure-class
# failures (training_errored, no_checkpoint, adapter_mismatch,
# scaffold_errored, v1_materialization_errored) signal env / code issues
# re-decomposition can't fix and are deliberately excluded.
_REDECOMPOSABLE_REASONS: frozenset[str] = frozenset(
    {"criterion_not_met", "criterion_errored"}
)

# Ship 22r: within a single stage's ONE redecomposition (the per-stage cap
# above is enforced via `redecomposition_attempts`), how many times to ask
# Claude for a VALID sub-stage draft. A draft can be rejected by the mission
# validator (e.g. a sub-stage criterion that references a non-persisted
# trajectory key like base_height). Rather than halt the whole mission on the
# first bad draft, re-ask with the exact validator error fed back. 2 = one
# initial draft + one corrective retry; bounded so a hopeless redecomposition
# still halts quickly instead of looping.
_REDECOMPOSE_MAX_ATTEMPTS = 2


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
        by `[failed_stage] + sub_stages` — §mission-persistence
        increment 1: the failed stage is RETAINED (marked
        "superseded"), not discarded. Its trained iterations stay
        visible in every UI view / report; only its runnable slot is
        taken over by the children.
      * Downstream children's `parent_stage` references re-pointed
        from `failed.name` to the LAST sub-stage's name.
      * `mission.current_stage_idx` set to `failed_stage_idx + 1` (the
        first sub-stage now sits one slot AFTER the retained,
        superseded failed stage) so the while-loop processes the first
        sub-stage next.
      * Any sub-stage lacking its own `steering_metric` inherits the
        failed stage's (metric inheritance); `stage_metric_inherited`
        emitted when this fires.
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

    info_keys = set(getattr(reward_contract, "expected_info_keys", None) or [])

    # Ship 22r: a redecomposition DRAFT can itself be invalid — e.g. a
    # sub-stage criterion referencing a non-persisted trajectory key
    # (base_height) or a component the reward never emits. Pre-22r the first
    # bad draft halted the whole mission, discarding the (often multi-hour)
    # training already invested in this stage. Instead, retry a bounded
    # number of times, feeding the EXACT validator error back to Claude so it
    # corrects the offending sub-stage. Snapshot the pre-splice graph so each
    # retry starts from a clean state.
    saved_stages = list(mission.stages)
    saved_parents = [(s, s.parent_stage) for s in mission.stages]

    def _restore_pre_splice() -> None:
        mission.stages[:] = saved_stages
        for _s, _p in saved_parents:
            _s.parent_stage = _p

    prior_error: Optional[str] = None
    last_reason = "spliced_mission_invalid"
    last_detail = ""
    sub_stages: list = []
    last_sub_name = ""
    repointed = 0
    spliced_ok = False

    def _emit_retry(attempt: int) -> None:
        emit({
            "type": "stage_redecomposition_retry",
            "stage_name": failed_stage.name,
            "attempt": attempt + 1,
            "max_attempts": _REDECOMPOSE_MAX_ATTEMPTS,
            "reason": last_reason,
            "detail": last_detail,
        })

    for attempt in range(_REDECOMPOSE_MAX_ATTEMPTS):
        # Call Claude (the prior attempt's validator error, if any, is fed
        # back into the prompt so Claude fixes the offending sub-stage).
        try:
            sub_stages = redecompose_stage(
                mission, failed_stage_idx,
                feedback=feedback,
                reward_contract=reward_contract,
                kg_store=kg_store,
                prior_attempt_error=prior_error,
            )
        except (MissionValidationError, DecompositionError) as e:
            last_reason, last_detail = "validation_failed", f"{type(e).__name__}: {e}"
            prior_error = last_detail
            _emit_retry(attempt)
            continue
        except Exception as e:  # noqa: BLE001 — non-validation errors don't retry
            emit({
                "type": "stage_redecomposition_failed",
                "stage_name": failed_stage.name,
                "reason": "claude_call_errored",
                "detail": f"{type(e).__name__}: {e}",
            })
            return False

        if not sub_stages:
            last_reason, last_detail = "empty_substages", "Claude returned no sub-stages."
            prior_error = last_detail
            _emit_retry(attempt)
            continue

        last_sub_name = sub_stages[-1].name
        # Splice: RETAIN the failed stage (in place, ahead of its
        # children) and insert sub-stages immediately after it.
        # §mission-persistence increment 1: previously this replaced
        # the failed stage outright (`mission.stages[idx:idx+1] =
        # sub_stages`), which discarded a stage that could carry many
        # hours of trained iterations from every UI view / report the
        # moment it failed. `failed_stage.status` is only flipped to
        # "superseded" once the splice is confirmed valid (below) —
        # NOT here — because `_restore_pre_splice` restores the
        # ORIGINAL `Stage` objects by reference (`saved_stages =
        # list(mission.stages)` copies the list, not the objects), so
        # an in-place status mutation made before a failed validation
        # attempt would survive the "rollback" and corrupt the retry.
        mission.stages[failed_stage_idx:failed_stage_idx + 1] = (
            [failed_stage] + sub_stages
        )
        repointed = _repoint_downstream_children(
            mission.stages, failed_stage.name, last_sub_name,
            slice_start=failed_stage_idx + 1 + len(sub_stages),
        )
        # Validate the spliced mission. If invalid, restore the clean
        # pre-splice graph and retry with the error fed back.
        try:
            validate_mission(mission, info_keys=info_keys)
        except MissionValidationError as e:
            _restore_pre_splice()
            last_reason, last_detail = "spliced_mission_invalid", f"{type(e).__name__}: {e}"
            prior_error = last_detail
            _emit_retry(attempt)
            continue

        spliced_ok = True
        break

    if not spliced_ok:
        emit({
            "type": "stage_redecomposition_failed",
            "stage_name": failed_stage.name,
            "reason": last_reason,
            "detail": (
                f"redecomposition produced no valid mission after "
                f"{_REDECOMPOSE_MAX_ATTEMPTS} attempt(s); last error: {last_detail}"
            ),
        })
        return False

    # §mission-persistence increment 1: the splice validated cleanly —
    # commit the failed stage to its terminal "superseded" state.
    # failure_reason / failure_detail were already set by `_fail_stage`
    # and are left untouched (that's the record of WHY it was
    # superseded). Terminal + never re-entered by the while-loop.
    failed_stage.status = "superseded"

    # §mission-persistence increment 1: metric inheritance. A sub-stage
    # the redecomposer left without its own `steering_metric` inherits
    # the superseded parent's, so it keeps steering by the same
    # objective rather than silently falling back to the mission-level
    # default. (In practice `redecompose_stage` already sets every
    # sub-stage's steering_metric = failed_stage.steering_metric
    # unconditionally — see decompose.py — so this is a defense-in-
    # depth backstop for that behavior changing, not the primary path.)
    _inherited_to: list[str] = []
    for _sub in sub_stages:
        if not getattr(_sub, "steering_metric", None) and failed_stage.steering_metric:
            _sub.steering_metric = failed_stage.steering_metric
            _inherited_to.append(_sub.name)
    if _inherited_to:
        emit({
            "type": "stage_metric_inherited",
            "from_stage": failed_stage.name,
            "to_stages": _inherited_to,
        })

    # Reviewer-flagged: persist current_stage_idx BEFORE the splice so
    # a crash-resume starts at the first new sub-stage rather than
    # skipping ahead. §mission-persistence increment 1: the retained,
    # superseded failed stage now occupies `failed_stage_idx`, so the
    # first sub-stage is one slot later.
    mission.current_stage_idx = failed_stage_idx + 1

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
        # Roll back in-memory splice + parent re-pointing + status +
        # any inherited metric + current_stage_idx.
        mission.stages[failed_stage_idx:failed_stage_idx + 1 + len(sub_stages)] = [failed_stage]
        failed_stage.status = "failed"
        for _sub in sub_stages:
            if _sub.name in _inherited_to:
                _sub.steering_metric = None
        mission.current_stage_idx = failed_stage_idx
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
