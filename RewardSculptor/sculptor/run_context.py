"""sculptor/run_context.py — reproducibility capture (§Ship 24 / R1).

Every `sculpt run` records the full context needed to reproduce (or
honestly explain) a result: code + project git SHAs, a config snapshot,
prompt hashes, LLM model ids, package versions, and the seed plan.
Missions additionally keep a `provenance.json` in the mission dir with
one record per executed stage.

Design rules:
  * Capture NEVER crashes a run — callers wrap in try/except and emit a
    `run_context_capture_failed` event; every helper here is also
    individually tolerant (a missing git binary yields `"unavailable"`,
    not an exception).
  * Volatile fields are isolated (`captured_at`) so tests can assert
    capture determinism by comparing everything else.
  * This is DISTINCT from `reports/provenance.json` (per-reward-term
    citation provenance, sculpt.py `_update_provenance`) — that file's
    shape is consumed by the UI and is not touched here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

#: Bumped on shape changes so downstream analysis can dispatch.
SCHEMA_VERSION = 1

#: Packages whose versions matter for reproducing a run. Tolerant
#: lookup — absent packages are simply omitted.
_TRACKED_PACKAGES = (
    "reward-sculptor",  # sculptor's own dist version (pip installs)
    "torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib",
    "numpy", "scipy", "gymnasium", "stable-baselines3", "anthropic",
    "sentence-transformers", "imageio", "imageio-ffmpeg",
)

#: Modules that talk to the Anthropic API and the constant carrying
#: their model id. Temperature is not set anywhere in the pipeline —
#: recorded as "sdk_default" (and would be recorded per-module if that
#: ever changes).
_LLM_MODULES = (
    ("decompose", "sculptor.decompose", "MODEL_ID"),
    ("diagnose", "sculptor.diagnose", "MODEL_ID"),
    ("edit", "sculptor.edit", "MODEL_ID"),
    ("kg_extract", "sculptor.kg.extract", "MODEL_ID"),
    ("kg_research", "sculptor.kg.research", "_MODEL"),
    ("mjcf_editor", "sculptor.adapters.mjcf_editor", "_MODEL_ID"),
)


# ── small tolerant collectors ────────────────────────────────────────


def _git_info(repo_dir: Path, must_track: Optional[Path] = None) -> dict[str, Any]:
    """{sha, branch, dirty} for the repo containing `repo_dir`, or
    {"available": False} when git/repo is absent.

    `must_track`: a file that must be TRACKED by the found repo for the
    answer to count. Guards the code-repo lookup: a sculptor pip-
    installed into `.venv/` INSIDE an adopter's project repo would
    otherwise report the adopter's SHA as sculptor's code revision.
    """
    def _run(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0:
            return None
        return r.stdout.strip()

    sha = _run("rev-parse", "HEAD")
    if sha is None:
        return {"available": False}
    if must_track is not None:
        tracked = _run("ls-files", "--error-unmatch", str(must_track))
        if tracked is None:
            return {"available": False, "reason": "installed_outside_repo"}
    status = _run("status", "--porcelain")
    return {
        "available": True,
        "sha": sha,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # None (not False) when `git status` itself failed — an unknown
        # dirty state must never be recorded as a definitive "clean".
        "dirty": bool(status) if status is not None else None,
    }


def _code_repo_dir() -> Path:
    """Directory of the sculptor package itself — `_git_info` on it
    resolves the CODE revision even when sculptor lives in a parent
    monorepo (git -C walks up to the enclosing repo)."""
    return Path(__file__).resolve().parent


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version

    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 — absent is fine
            continue
    return out


def _prompt_hashes() -> dict[str, Any]:
    """sha256 per prompt .md (honors SCULPTOR_PROMPTS_DIR overrides —
    a tuned prompt is a different experiment)."""
    try:
        from sculptor.prompts import _prompts_dir, available_prompts, load_prompt

        d = _prompts_dir()
        hashes = {
            name: hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
            for name in available_prompts()
        }
        return {"dir": str(d), "sha256": hashes}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _llm_specs() -> dict[str, Any]:
    import importlib

    out: dict[str, Any] = {}
    for key, module_name, attr in _LLM_MODULES:
        try:
            mod = importlib.import_module(module_name)
            out[key] = {
                "model_id": str(getattr(mod, attr)),
                "temperature": "sdk_default",
            }
        except Exception as e:  # noqa: BLE001 — optional deps may be absent
            out[key] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _sculptor_env() -> dict[str, Any]:
    """Behavior-affecting environment, without secrets: the NAMES of
    set SCULPTOR_* vars, plus where training ran (remote host) — but
    never key paths' contents or API keys."""
    names = sorted(k for k in os.environ if k.startswith("SCULPTOR_"))
    return {
        "sculptor_vars_set": names,
        "remote_enabled": os.environ.get("SCULPTOR_REMOTE_ENABLED") in
        ("1", "true", "yes", "on"),
        "remote_host": os.environ.get("SCULPTOR_REMOTE_HOST") or None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        # mkstemp creates 0600; match the 0644 of sibling artifacts so
        # rsync/file-serving paths see consistent permissions.
        try:
            os.fchmod(fd, 0o644)
        except (OSError, AttributeError):  # non-POSIX
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# ── run-level capture ────────────────────────────────────────────────


def capture_run_context(
    project: Path,
    config_path: Optional[Path] = None,
    *,
    parsed_config: Optional[Mapping[str, Any]] = None,
    behavior_goal: Optional[str] = None,
    base_seed: Optional[int] = None,
    iterations: Optional[int] = None,
    start_iter: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Collect the full reproducibility context for one sculpt run.

    `parsed_config` is the ALREADY-parsed toml dict (post CLI
    overrides) — recorded alongside the raw file's sha256 so both "what
    was on disk" and "what actually ran" are pinned.
    """
    ctx: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "captured_at": _utc_now(),
        "behavior_goal": behavior_goal,
        "code_git": _git_info(
            _code_repo_dir(), must_track=_code_repo_dir() / "__init__.py",
        ),
        "project_git": _git_info(Path(project)),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "prompts": _prompt_hashes(),
        "llm": _llm_specs(),
        "seeds": {
            "base_seed": base_seed,
            "train": "base_seed + iter_index (sculpt.py iteration loop)",
            "rollout": "no explicit seed — deterministic given checkpoint",
            "decompose": "LLM sampling only (model_id + prompt hash above)",
        },
        "env": _sculptor_env(),
        "argv": list(sys.argv),
        "iterations": iterations,
        "start_iter": start_iter,
    }
    if config_path is not None:
        config_path = Path(config_path)
        cfg_entry: dict[str, Any] = {"path": str(config_path)}
        try:
            cfg_entry["sha256"] = hashlib.sha256(
                config_path.read_bytes()).hexdigest()
        except OSError as e:
            cfg_entry["sha256_error"] = str(e)
        if parsed_config is not None:
            cfg_entry["effective"] = dict(parsed_config)
        ctx["config"] = cfg_entry
    if extra:
        ctx["extra"] = dict(extra)
    return ctx


#: Fields legitimately differing between two captures of the same
#: state — everything else must match for a capture to be reproducible.
VOLATILE_KEYS = ("captured_at",)


def write_run_context(reports_dir: Path, ctx: Mapping[str, Any]) -> Path:
    return write_json_atomic(Path(reports_dir) / "run_context.json", ctx)


# ── mission-level provenance ─────────────────────────────────────────


def mission_provenance_path(mission_dir: Path) -> Path:
    return Path(mission_dir) / "provenance.json"


def init_mission_provenance(
    mission_dir: Path,
    *,
    goal: str,
    n_stages: int,
    seed: Optional[int],
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the mission-level provenance header. Idempotent across
    resumes: an existing file keeps its `created_at` + stage records
    and gets a fresh context appended to `contexts` (each resume may
    run different code — record every one)."""
    path = mission_provenance_path(mission_dir)
    ctx = capture_run_context(
        Path(mission_dir), None,
        behavior_goal=goal, base_seed=seed, extra=extra,
    )
    doc: dict[str, Any]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not a dict")
    except Exception:  # noqa: BLE001 — first run or corrupt → fresh
        doc = {
            "schema": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "goal": goal,
            "n_stages_initial": n_stages,
            "contexts": [],
            "stages": [],
        }
    doc["goal"] = goal
    contexts = doc.setdefault("contexts", [])
    contexts.append(ctx)
    doc.setdefault("stages", [])
    write_json_atomic(path, doc)
    return path


def record_stage_in_provenance(
    mission_dir: Path, stage_record: Mapping[str, Any],
) -> Optional[Path]:
    """Append one stage outcome record. Returns None (never raises) if
    the provenance file is unreadable — provenance must not break a
    mission."""
    path = mission_provenance_path(mission_dir)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return None
    except Exception:  # noqa: BLE001
        return None
    rec = dict(stage_record)
    rec.setdefault("recorded_at", _utc_now())
    doc.setdefault("stages", []).append(rec)
    try:
        write_json_atomic(path, doc)
    except OSError:
        return None
    return path
