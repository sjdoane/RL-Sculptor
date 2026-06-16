"""The ONLY module in the backend that imports from `sculptor.*`.

Every other backend module interacts with sculptor through functions
here. Enforces the architectural constraint that sculptor source is
read-only from the UI's perspective — no edits to sculptor ever
originate from code outside this file.

The import is defensive: if sculptor cannot be imported (missing env,
broken install, Python/ABI mismatch) the backend continues to run and
reports the failure via `/health`.

Lazy-import discipline (MJLAB_PIVOT_DESIGN §7)
----------------------------------------------
mjlab, mujoco_warp, and rsl_rl MUST NOT be eagerly imported at backend
startup — their cold import is 15-25 s (CUDA kernel compilation) which
would blow the FastAPI cold-start budget. Availability is probed with
`importlib.util.find_spec()` only. Actual imports happen inside the
MjlabAdapter's methods (subprocess runner).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

_sculptor_import_error: Optional[str] = None
_sculpt_init = None

try:
    from sculptor.sculpt import sculpt_init as _sculpt_init_fn  # type: ignore

    _sculpt_init = _sculpt_init_fn
except Exception as e:  # noqa: BLE001 — deliberate broad catch
    _sculptor_import_error = f"{type(e).__name__}: {e}"


def sculptor_ok() -> bool:
    """True when the sculptor package is importable."""
    return _sculptor_import_error is None


def sculptor_error() -> Optional[str]:
    """Human-readable import error string, or None if sculptor loaded."""
    return _sculptor_import_error


def scaffold_project(project_dir: Path, adapter: str = "gym_sb3") -> Path:
    """Scaffold a fresh Sculptor project at `project_dir`.

    Delegates to `sculptor.sculpt.sculpt_init`, which:
      - writes config.toml, rewards/v0.py, rewards/current.py,
        kg_seeds.yml, .gitignore, reports/, runs/
      - best-effort git init + initial commit (if git is on PATH)

    Raises RuntimeError if sculptor is unavailable, FileExistsError if
    the target dir exists and is non-empty.
    """
    if not sculptor_ok():
        raise RuntimeError(
            f"sculptor library is unavailable: {sculptor_error()}"
        )
    assert _sculpt_init is not None  # narrowed by sculptor_ok()
    return _sculpt_init(project_dir, adapter=adapter)


# ── mjlab / mujoco_warp / rsl_rl availability (find_spec, no import) ────────
def mjlab_available() -> bool:
    """True if `import mjlab` would succeed. Uses find_spec only; does
    NOT import mjlab (cold-start cost 15-25 s)."""
    return importlib.util.find_spec("mjlab") is not None


def mujoco_warp_available() -> bool:
    """True if `import mujoco_warp` would succeed. find_spec only."""
    return importlib.util.find_spec("mujoco_warp") is not None


def rsl_rl_available() -> bool:
    """True if `import rsl_rl` would succeed. find_spec only."""
    return importlib.util.find_spec("rsl_rl") is not None


# ── GPU info (torch is a hard sculptor dep; safe to import) ─────────────────
def gpu_info() -> dict[str, Any]:
    """Return torch.cuda status + per-device memory + pynvml-backed live
    telemetry when available (utilization, temperature, driver version).
    Does NOT import mjlab. Called by GET /system/gpu and by
    project-create validation for mjlab projects.

    M4: extends the M2 shape with `utilization_percent`, `temperature_c`,
    `used_memory_bytes`, `driver_version`, and `pynvml_available`. All
    nvml calls are cached for 2 s by the gpu_monitor service.
    """
    out: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "driver_version": None,
        "pynvml_available": False,
        "mjlab_available": mjlab_available(),
        "mujoco_warp_available": mujoco_warp_available(),
        "rsl_rl_available": rsl_rl_available(),
    }
    try:
        import torch
    except ImportError as e:
        out["import_error"] = f"{type(e).__name__}: {e}"
        return out

    out["torch_available"] = True
    out["torch_version"] = torch.__version__
    out["cuda_version"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    if not out["cuda_available"]:
        return out

    # Prefer gpu_monitor's live snapshot (pynvml + torch-augmented). If
    # pynvml init fails, fall back to the torch-only path.
    from backend.services.gpu_monitor import get_live_snapshot
    snap = get_live_snapshot()
    out["pynvml_available"] = bool(snap.get("pynvml_available", False))
    out["driver_version"] = snap.get("driver_version")
    out["device_count"] = int(snap.get("device_count", 0))
    out["devices"] = list(snap.get("devices") or [])
    return out


def cuda_version_ok(min_major: int = 12, min_minor: int = 4) -> bool:
    """Check torch-reported CUDA version meets minimum. mjlab requires
    CUDA 12.4+ per its FAQ (conditional execution in CUDA graphs).

    Uses `packaging.version.parse` (NOT string comparison) — string
    compare fails at version boundaries like "9.0" vs "12.4" where
    lexical ordering would say "9.0" > "12.4".
    """
    from packaging.version import InvalidVersion
    from packaging.version import parse as parse_version

    info = gpu_info()
    v = info.get("cuda_version")
    if not isinstance(v, str):
        return False
    try:
        return parse_version(v) >= parse_version(f"{min_major}.{min_minor}")
    except InvalidVersion:
        return False


# ── §Ship 35: auto-generated objective metrics ────────────────────────
def generate_objective_metric(
    behavior_goal: str,
    out_dir: "Path",
    *,
    robot_hint: Optional[str] = None,
    review: bool = True,
    on_event=None,
) -> dict:
    """Generate + validate + review an objective fitness metric (the only
    module allowed to import sculptor.eval). Returns the full record.

    §Ship 40: `on_event` (optional) streams `{stage, attempt, max, message}`
    progress so the UI can show live generation progress."""
    from sculptor.eval import generate_objective_metric as _gen

    return _gen(behavior_goal, out_dir, robot_hint=robot_hint, review=review,
                on_event=on_event)


def calibrate_objective_metric(
    metric_path: "Path", builtin_name: str, *, threshold: float = 0.7,
) -> dict:
    """Calibrate a generated metric against a hand-authored ground-truth
    metric (Spearman over a competence ladder). Earns steer-rights on ok."""
    from sculptor.eval import calibrate_metric as _cal

    return _cal(metric_path, builtin_name, threshold=threshold)


def calibrate_task_derived_metric(
    metric_path: "Path", behavior_goal: str,
    robot_hint: Optional[str] = None, *, client: Any = None, k_sources: int = 3,
) -> dict:
    """§Ship 51: earn steer-rights on a NOVEL task (no built-in ground truth)
    by ranking K independently-authored competence ladders. `client` is
    injectable for tests (else a real Anthropic client is constructed)."""
    from sculptor.eval import calibrate_task_derived as _cal

    return _cal(metric_path, behavior_goal, robot_hint,
                client=client, k_sources=k_sources)


def finalize_calibration(calibration: dict, validation: Optional[dict]) -> dict:
    """§Ship 52: the unified grant + standardized trust score over both
    calibration paths. Returns `{calibrated: bool, trust: {...}}`. The grant is
    byte-identical to today for the 5 built-ins; trust is a display confidence."""
    from sculptor.eval import compute_trust, grant_decision

    return {
        "calibrated": bool(grant_decision(calibration, validation)),
        "trust": compute_trust(calibration, validation),
    }


def builtin_spec_metric_names() -> list[str]:
    """The hand-authored ground-truth metric names (for calibration UI)."""
    from sculptor.eval import spec_metric_names

    return spec_metric_names()


def resolve_calibration_builtin(
    behavior_goal: str, robot_hint: Optional[str] = None,
) -> Optional[str]:
    """§Ship 44: map a behavior goal to the hand-authored built-in a
    launch-generated metric should calibrate against (kick→g1_kick,
    floss→g1_floss, jump→g1_jump, locomotion→go1_trot, balance→cartpole).
    Returns None when no family matches (the metric then stays observe-only —
    the firewall never lets an uncalibrated metric steer). Never raises."""
    try:
        from sculptor.eval.metric_validate import (
            FAMILY_TO_BUILTIN,
            resolve_behavior_family,
        )

        fam = resolve_behavior_family(behavior_goal, robot_hint)
        builtin = FAMILY_TO_BUILTIN.get(fam) if fam else None
        if builtin and builtin in builtin_spec_metric_names():
            return builtin
    except Exception:  # noqa: BLE001 — resolution is best-effort
        pass
    return None
