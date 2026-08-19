"""Auto-generated, per-task objective metrics (§Ship 35).

A generated metric is a small Python module that MIRRORS the hand-authored
spec contract in `spec_metrics.py`:

    def compute_spec(arrays, behavior, meta) -> dict[str, float]
        # returns {"spec_score": float in [0,1], ...sub-components}

computed PURELY from the persisted physical rollout arrays (official
first-episode validity, joint_pos, joint_vel, projected_gravity_b,
root_link_pos_w, root_link_ang_vel_b + behavior.json + joint_names) — NEVER
from LLM judgment. It is generated from the NL goal,
then put through a validation chain (safety/contract/determinism/bounds +
a monotonicity audit) and an independent LLM review, and must EARN
steer-rights via calibration (Spearman vs a hand-authored ground-truth
metric) before it is allowed to drive selection. Until then it runs
OBSERVE-ONLY (computed + displayed, no influence). This module is the
RUNTIME side (load + compute + resolve); generation lives in `metric_gen`,
validation in `metric_validate`, calibration in `metric_calibration`.

SECURITY (§round-9/10/31): generated metrics are UNTRUSTED LLM-authored code.  The
static `metric_validate._ast_safety` gate remains defense in depth, but is not the
containment boundary: every accepted source snapshot is now compiled and executed only
in a dedicated worker process.  The worker has a private cwd, a secret-free environment,
memory/CPU/file-size/fd rlimits, a parent-enforced wall timeout, and (fail-closed on Linux)
a seccomp filter denying filesystem, network, process-spawn, tracing, and kernel-program
syscalls.  Requests and results use bounded JSON/raw-array IPC; pickle is never used.
`REQUIRED_JOINT_ROLES` is parsed from that same immutable source snapshot without exec.

The worker deliberately retains real builtins: numpy ndarray methods internally need
`__import__`, so a curated-builtins pseudo-sandbox breaks legitimate metrics.  Real
builtins are safe only because they live behind the OS process/syscall boundary.  This
module-like proxy preserves existing validation/calibration call sites while ensuring no
generated module top-level code or `compute_spec` body executes in the parent process.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import types
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

#: Maximum wall time for one generated metric invocation.  Metrics operate on
#: persisted arrays and should be vectorized; exceeding this is a hard failure,
#: not an invitation to stall a campaign worker.
METRIC_CALL_TIMEOUT_SECONDS = 3.0

#: Imports and numpy initialization happen before the syscall filter is loaded,
#: so worker startup gets a wider (still bounded) allowance than a metric call.
METRIC_STARTUP_TIMEOUT_SECONDS = 15.0

_MAX_SANDBOX_FRAME_BYTES = 512 * 1024 * 1024
_MAX_SANDBOX_RESPONSE_BYTES = 1024 * 1024


class MetricSandboxError(RuntimeError):
    """Base error raised at the generated-metric process boundary."""


class MetricSandboxUnavailable(MetricSandboxError):
    """The required OS containment boundary could not be established."""


class MetricSandboxTimeout(MetricSandboxError):
    """A generated metric exceeded its wall-time budget and was terminated."""


class MetricSandboxExecutionError(MetricSandboxError):
    """The sandboxed metric raised or returned an invalid wire result."""


# Preserve the small set of ordinary numerical/shape errors that callers use
# diagnostically.  OS/security exceptions intentionally remain wrapped as
# MetricSandboxExecutionError so a denied escape is unmistakable.
_REMOTE_BUILTIN_EXCEPTIONS: dict[str, type[Exception]] = {
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "OverflowError": OverflowError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}

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
    "first_episode_valid_mask",
    "joint_pos",
    "joint_vel",
    # Ordered-joint RMS deviation from the environment-owned default pose.
    # Optional for legacy rollouts; metrics must guard it with arrays.get.
    "default_pose_rms",
    "projected_gravity_b",
    "root_link_pos_w",
    "root_link_ang_vel_b",
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


def _sandbox_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            raise ValueError("metadata array is too large for metric sandbox IPC")
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("metric sandbox worker closed its protocol stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_sandbox_frame(stream: Any) -> bytes:
    size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if size > _MAX_SANDBOX_RESPONSE_BYTES:
        raise MetricSandboxError(
            f"metric sandbox response exceeds {_MAX_SANDBOX_RESPONSE_BYTES} bytes")
    return _read_exact(stream, size)


def _write_sandbox_frame(stream: Any, payload: bytes) -> None:
    if len(payload) > _MAX_SANDBOX_FRAME_BYTES:
        raise MetricSandboxError(
            f"metric sandbox request exceeds {_MAX_SANDBOX_FRAME_BYTES} bytes")
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _encode_metric_request(
    arrays: dict[str, np.ndarray], behavior: dict, meta: dict,
) -> bytes:
    """Encode numeric arrays without pickle and JSON-encode the small metadata."""
    raw_parts: list[bytes] = []
    array_headers: list[dict[str, Any]] = []
    offset = 0
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise MetricSandboxError(
                f"object dtype is forbidden for metric array {name!r}")
        raw = array.tobytes(order="C")
        array_headers.append({
            "name": str(name),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": len(raw),
        })
        raw_parts.append(raw)
        offset += len(raw)
    header = json.dumps(
        {"arrays": array_headers, "behavior": behavior, "meta": meta},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_sandbox_json_default,
    ).encode("utf-8")
    payload = struct.pack(">Q", len(header)) + header + b"".join(raw_parts)
    if len(payload) > _MAX_SANDBOX_FRAME_BYTES:
        raise MetricSandboxError(
            f"metric inputs exceed {_MAX_SANDBOX_FRAME_BYTES} byte IPC limit")
    return payload


class _SandboxedGeneratedMetric:
    """Callable proxy whose module load and invocations stay in one worker."""

    def __init__(self, source: str, source_path: Path) -> None:
        self._source = source
        self._source_path = source_path
        self._source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self._process: subprocess.Popen[bytes] | None = None
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.Lock()
        self._isolation: str | None = None
        self._resource_limits: dict[str, int | None] = {}

    @property
    def sandbox_info(self) -> dict[str, Any]:
        return {
            "isolation": self._isolation,
            "source_sha256": self._source_sha256,
            "call_timeout_seconds": METRIC_CALL_TIMEOUT_SECONDS,
            "resource_limits": dict(self._resource_limits),
        }

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_locked()
        worker = Path(__file__).with_name("_metric_sandbox_worker.py")
        if not worker.is_file():
            raise MetricSandboxUnavailable(
                f"metric sandbox worker is missing: {worker}")
        self._workdir = tempfile.TemporaryDirectory(
            prefix="rewardsculptor-metric-")
        # Never inherit API tokens, cloud credentials, or campaign secrets.
        # The absolute interpreter/worker paths make PATH unnecessary.
        environment = {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "LC_ALL": "C.UTF-8",
        }
        # Required by Python on Windows; harmlessly absent on POSIX.  The
        # Linux deployment is stricter because it requires seccomp below.
        for name in ("SYSTEMROOT", "WINDIR"):
            if value := os.environ.get(name):
                environment[name] = value
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self._workdir.name,
                env=environment,
                bufsize=0,
                close_fds=True,
            )
            load_payload = json.dumps(
                {"source": self._source}, ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            response = self._exchange_locked(
                load_payload, METRIC_STARTUP_TIMEOUT_SECONDS, "startup")
            ready = json.loads(response.decode("utf-8"))
            if not isinstance(ready, dict) or not ready.get("ok"):
                detail = ready.get("error", "invalid startup response") \
                    if isinstance(ready, dict) else "invalid startup response"
                raise MetricSandboxExecutionError(str(detail))
            isolation = ready.get("isolation")
            if sys.platform.startswith("linux") \
                    and isolation != "process+rlimit+seccomp":
                raise MetricSandboxUnavailable(
                    "generated metrics require seccomp isolation on Linux; "
                    f"worker reported {isolation!r}")
            self._isolation = str(isolation)
            limits = ready.get("resource_limits")
            self._resource_limits = dict(limits) if isinstance(limits, dict) else {}
        except Exception:
            self._stop_locked()
            raise

    def _exchange_locked(self, payload: bytes, timeout: float, phase: str) -> bytes:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise MetricSandboxError("metric sandbox worker is not running")
        result: list[bytes] = []
        errors: list[BaseException] = []

        def _io() -> None:
            try:
                _write_sandbox_frame(process.stdin, payload)
                result.append(_read_sandbox_frame(process.stdout))
            except BaseException as exc:  # crossed back to the controlling thread
                errors.append(exc)

        io_thread = threading.Thread(
            target=_io, name="metric-sandbox-ipc", daemon=True)
        io_thread.start()
        io_thread.join(timeout)
        if io_thread.is_alive():
            self._stop_locked()
            io_thread.join(1.0)
            raise MetricSandboxTimeout(
                f"generated metric {phase} exceeded {timeout:.3g}s wall timeout")
        if errors:
            self._stop_locked()
            raise MetricSandboxError(
                f"metric sandbox {phase} protocol failed: {errors[0]}") from errors[0]
        if not result:
            self._stop_locked()
            raise MetricSandboxError(
                f"metric sandbox {phase} returned no response")
        return result[0]

    def __call__(
        self, arrays: dict[str, np.ndarray], behavior: dict, meta: dict,
    ) -> dict:
        payload = _encode_metric_request(arrays, behavior, meta)
        with self._lock:
            self._start_locked()
            response = self._exchange_locked(
                payload, METRIC_CALL_TIMEOUT_SECONDS, "call")
        try:
            decoded = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetricSandboxExecutionError(
                f"worker returned invalid JSON: {exc}") from exc
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            detail = decoded.get("error", "invalid worker response") \
                if isinstance(decoded, dict) else "invalid worker response"
            if isinstance(decoded, dict):
                remote_type = decoded.get("error_type")
                exception_type = _REMOTE_BUILTIN_EXCEPTIONS.get(str(remote_type))
                if exception_type is not None:
                    raise exception_type(str(decoded.get("error_message", detail)))
            raise MetricSandboxExecutionError(str(detail))
        result = decoded.get("result")
        if not isinstance(result, dict):
            raise MetricSandboxExecutionError(
                "generated metric did not return a JSON object")
        return result

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        workdir, self._workdir = self._workdir, None
        if workdir is not None:
            workdir.cleanup()
        self._isolation = None
        self._resource_limits = {}

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


def load_generated_module(module_path: Path | str):
    """Load a generated metric as a module-like sandbox proxy.

    The source is read exactly once, statically screened, and passed as an
    immutable snapshot to the worker.  Starting the worker eagerly preserves
    import-time failure semantics while keeping all module exec outside the
    parent.  (`_ast_safety` is lazy-imported to avoid an import cycle.)
    """
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
    metric = _SandboxedGeneratedMetric(source, module_path.resolve())
    metric.start()
    return types.SimpleNamespace(
        __file__=str(module_path.resolve()),
        _metric_source_snapshot=source,
        __metric_sandbox__=metric,
        compute_spec=metric,
    )


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
    # A sandbox proxy carries the exact source snapshot that was screened and
    # sent to its worker.  Prefer it over re-reading the path: validated roles
    # and executed code must remain one atomic artifact even if a file is
    # replaced after load.
    snapshot = getattr(module_or_path, "_metric_source_snapshot", None)
    if isinstance(snapshot, str):
        return read_required_roles_static(snapshot)
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
    metric_meta: dict[str, Any] = {}
    try:
        loaded_meta = json.loads(
            (module_path.parent / "meta.json").read_text(encoding="utf-8")
        )
        if isinstance(loaded_meta, dict):
            metric_meta = loaded_meta
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    try:
        metric_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    except OSError:
        metric_sha256 = None

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
    # Persisted by sculpt.py into each fitness.json. These attributes are
    # provenance only; they do not grant the generated metric steering rights.
    _fitness.metric_id = (  # type: ignore[attr-defined]
        str(metric_meta.get("id") or module_path.parent.name)
    )
    _fitness.metric_version = (  # type: ignore[attr-defined]
        str(metric_meta["version"])
        if metric_meta.get("version") is not None else None
    )
    _fitness.metric_source = "generated"  # type: ignore[attr-defined]
    _fitness.metric_sha256 = metric_sha256  # type: ignore[attr-defined]
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
