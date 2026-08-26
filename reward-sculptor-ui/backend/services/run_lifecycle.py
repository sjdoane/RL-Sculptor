"""Pure lifecycle proof for every sculpt worker, independent of the KG.

Process exit and cumulative metric-history length are observations, not proof
that the newly requested outer-iteration plan completed.  This accumulator
accepts only the worker's ordered run/iteration events and an independently
authorized early-stop source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RunLifecycleError(ValueError):
    """Raised when authoritative worker lifecycle evidence contradicts."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_RECEIPTS_DIR = "_run_receipts"


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError(
            f"lifecycle evidence is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_user_stop_authorization(value: Any, run_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    disclosed = value.get("authorization_sha256")
    unsigned = dict(value)
    unsigned.pop("authorization_sha256", None)
    try:
        digest_matches = (
            isinstance(disclosed, str)
            and _SHA256_RE.fullmatch(disclosed) is not None
            and _canonical_sha256(unsigned) == disclosed
        )
    except RunLifecycleError:
        return False
    return (
        digest_matches
        and value.get("schema") == 1
        and value.get("authority") == "server_control_sidecar_stop"
        and value.get("run_id") == run_id
        and value.get("control_file") == f"_control_{run_id}.json"
        and value.get("stop") is True
        and type(value.get("control_bytes")) is int
        and value["control_bytes"] > 0
        and _SHA256_RE.fullmatch(
            str(value.get("control_sha256") or "")
        ) is not None
        and type(value.get("resume_token")) is int
        and value["resume_token"] >= 0
    )


def _valid_lifecycle_proof(proof: Any) -> bool:
    """Re-validate the self-digest and structural lifecycle authority."""
    if not isinstance(proof, dict):
        return False
    disclosed_sha = proof.get("proof_sha256")
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    try:
        digest_matches = (
            isinstance(disclosed_sha, str)
            and _SHA256_RE.fullmatch(disclosed_sha) is not None
            and _canonical_sha256(unsigned) == disclosed_sha
        )
    except RunLifecycleError:
        return False
    if (
        not digest_matches
        or proof.get("schema") != 1
        or proof.get("authority") != "worker_iteration_lifecycle_verified"
        or not isinstance(proof.get("run_id"), str)
        or not proof["run_id"]
        or _SHA256_RE.fullmatch(
            str(proof.get("run_started_event_sha256") or "")
        ) is None
    ):
        return False
    plan = proof.get("iteration_plan")
    event_digests = proof.get("iter_completed_event_sha256")
    if not isinstance(plan, dict) or not isinstance(event_digests, list):
        return False
    requested = plan.get("requested")
    completed = plan.get("completed")
    allowed = plan.get("allowed_early_stop_sources")
    early_stop = plan.get("early_stop")
    user_authorization = plan.get("user_stop_authorization")
    if (
        not isinstance(requested, list)
        or not requested
        or not isinstance(completed, list)
        or not completed
        or not isinstance(allowed, list)
        or any(
            not isinstance(source, str)
            or source not in {"fitness", "goodhart_onset", "user"}
            for source in allowed
        )
        or len(set(allowed)) != len(allowed)
        or any(type(index) is not int or index < 0 for index in requested)
        or any(type(index) is not int or index < 0 for index in completed)
    ):
        return False
    start = requested[0]
    if requested != list(range(start, start + len(requested))):
        return False
    if completed != requested[: len(completed)]:
        return False
    if (
        len(event_digests) != len(completed)
        or any(
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for digest in event_digests
        )
    ):
        return False
    if completed == requested:
        return early_stop is None and (
            user_authorization is None
            or (
                "user" in allowed
                and _valid_user_stop_authorization(
                    user_authorization, proof["run_id"]
                )
            )
        )
    if "user" in allowed:
        if not _valid_user_stop_authorization(
            user_authorization, proof["run_id"]
        ):
            return False
    elif user_authorization is not None:
        return False
    return (
        isinstance(early_stop, dict)
        and early_stop.get("at_iter") == completed[-1]
        and early_stop.get("source") in allowed
        and isinstance(early_stop.get("reason"), str)
        and bool(early_stop["reason"].strip())
        and _SHA256_RE.fullmatch(
            str(early_stop.get("event_sha256") or "")
        ) is not None
    )


def verified_lifecycle_completed_iterations(
    proof: Any, *, run_id: str,
) -> tuple[int, ...] | None:
    """Return completed indices only from an exact proof for ``run_id``.

    Timeline reconstruction must not infer completion from a later
    ``iter_started`` event when this proof is absent or belongs to another
    run.
    """
    if (
        not isinstance(run_id, str)
        or not run_id
        or not _valid_lifecycle_proof(proof)
        or proof.get("run_id") != run_id
    ):
        return None
    return tuple(proof["iteration_plan"]["completed"])


def build_terminal_run_receipt(
    *,
    project_slug: str,
    lifecycle_proof: dict[str, Any],
    iteration_receipts: list[dict[str, Any]],
    started_at: str | None,
    completed_at: str,
) -> dict[str, Any]:
    """Build a restart-safe terminal receipt bound to schema-3 iterations."""
    if not project_slug or not _valid_lifecycle_proof(lifecycle_proof):
        raise RunLifecycleError("terminal receipt has no valid lifecycle proof")
    completed = lifecycle_proof["iteration_plan"]["completed"]
    if len(iteration_receipts) != len(completed):
        raise RunLifecycleError(
            "terminal receipt requires every completed iteration receipt"
        )
    for index, receipt in zip(completed, iteration_receipts):
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != 3
            or receipt.get("iter_index") != index
            or _SHA256_RE.fullmatch(
                str(receipt.get("marker_sha256") or "")
            ) is None
        ):
            raise RunLifecycleError(
                "terminal receipt contains an invalid iteration receipt"
            )
    if not isinstance(completed_at, str) or not completed_at.strip():
        raise RunLifecycleError("terminal receipt requires completed_at")
    receipt = {
        "schema": 1,
        "authority": "backend_run_terminal_verified",
        "status": "completed",
        "project_slug": project_slug,
        "run_id": lifecycle_proof["run_id"],
        "started_at": started_at,
        "completed_at": completed_at,
        "lifecycle_proof": lifecycle_proof,
        "iteration_receipts": iteration_receipts,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def terminal_run_receipt_path(project_dir: Path, run_id: str) -> Path:
    """Use a digest filename so even a malformed run id cannot traverse."""
    if not isinstance(run_id, str) or not run_id:
        raise RunLifecycleError("terminal receipt requires a run id")
    filename = hashlib.sha256(run_id.encode("utf-8")).hexdigest() + ".json"
    return Path(project_dir) / "runs" / TERMINAL_RECEIPTS_DIR / filename


def write_terminal_run_receipt(
    project_dir: Path, receipt: dict[str, Any],
) -> Path:
    """Atomically persist one server-owned terminal lifecycle receipt."""
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str):
        raise RunLifecycleError("terminal receipt requires a run id")
    path = terminal_run_receipt_path(Path(project_dir), run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RunLifecycleError("terminal receipt directory cannot be a link")
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                receipt,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def verify_terminal_run_receipt(
    project_dir: Path,
    receipt_path: Path,
    *,
    project_slug: str,
) -> dict[str, Any] | None:
    """Reverify a terminal receipt and every bound iteration manifest."""
    project_dir = Path(project_dir)
    receipt_path = Path(receipt_path)
    expected_root = project_dir / "runs" / TERMINAL_RECEIPTS_DIR
    try:
        if (
            expected_root.is_symlink()
            or not expected_root.is_dir()
            or receipt_path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.resolve(strict=True).parent
            != expected_root.resolve(strict=True)
        ):
            return None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return None
    if not isinstance(receipt, dict):
        return None
    disclosed_sha = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    lifecycle_proof = receipt.get("lifecycle_proof")
    if (
        receipt.get("schema") != 1
        or receipt.get("authority") != "backend_run_terminal_verified"
        or receipt.get("status") != "completed"
        or receipt.get("project_slug") != project_slug
        or not isinstance(disclosed_sha, str)
        or _SHA256_RE.fullmatch(disclosed_sha) is None
        or not _valid_lifecycle_proof(lifecycle_proof)
    ):
        return None
    try:
        if _canonical_sha256(unsigned) != disclosed_sha:
            return None
    except RunLifecycleError:
        return None
    run_id = receipt.get("run_id")
    if (
        not isinstance(run_id, str)
        or lifecycle_proof.get("run_id") != run_id
        or receipt_path.name != terminal_run_receipt_path(
            project_dir, run_id,
        ).name
    ):
        return None
    iteration_receipts = receipt.get("iteration_receipts")
    completed = lifecycle_proof["iteration_plan"]["completed"]
    if (
        not isinstance(iteration_receipts, list)
        or len(iteration_receipts) != len(completed)
    ):
        return None
    try:
        from sculptor.run_manifests import verify_iteration_completion_marker

        for index, expected in zip(completed, iteration_receipts):
            if verify_iteration_completion_marker(
                project_dir / "runs" / f"iter_{index}"
            ) != expected:
                return None
    except Exception:  # noqa: BLE001 - restart authority fails closed
        return None
    return receipt


@dataclass
class RunLifecycleSession:
    run_id: str
    expected_iterations: tuple[int, ...]
    allowed_early_stop_sources: tuple[str, ...] = ()
    _run_started_sha256: str | None = field(default=None, init=False, repr=False)
    _started: list[int] = field(default_factory=list, init=False, repr=False)
    _completed: dict[int, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _early_stop: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _user_stop_authorization: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _fatal_errors: list[str] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.run_id:
            raise RunLifecycleError("run lifecycle requires a run id")
        if not self.expected_iterations:
            raise RunLifecycleError("run lifecycle requires a non-empty plan")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.expected_iterations
        ):
            raise RunLifecycleError(
                "run lifecycle iteration indices must be non-negative integers"
            )
        start = self.expected_iterations[0]
        if self.expected_iterations != tuple(
            range(start, start + len(self.expected_iterations))
        ):
            raise RunLifecycleError(
                "run lifecycle iteration plan must be ordered and contiguous"
            )
        allowed = tuple(sorted(set(self.allowed_early_stop_sources)))
        if any(
            source not in {"fitness", "goodhart_onset"}
            for source in allowed
        ):
            raise RunLifecycleError(
                "run lifecycle early-stop source lacks launch authority"
            )
        self.allowed_early_stop_sources = allowed

    def authorize_user_stop(self, receipt: dict[str, Any]) -> None:
        """Admit ``source=user`` only from a re-read server sidecar."""
        expected_file = f"_control_{self.run_id}.json"
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != 1
            or receipt.get("authority") != "server_control_sidecar_stop"
            or receipt.get("run_id") != self.run_id
            or receipt.get("control_file") != expected_file
            or receipt.get("stop") is not True
            or type(receipt.get("control_bytes")) is not int
            or receipt["control_bytes"] <= 0
            or _SHA256_RE.fullmatch(
                str(receipt.get("control_sha256") or "")
            ) is None
            or type(receipt.get("resume_token")) is not int
            or receipt["resume_token"] < 0
        ):
            raise RunLifecycleError(
                "user stop lacks an exact server control-sidecar authorization"
            )
        authorization = dict(receipt)
        authorization["authorization_sha256"] = _canonical_sha256(receipt)
        if (
            self._user_stop_authorization is not None
            and self._user_stop_authorization != authorization
        ):
            raise RunLifecycleError(
                "worker user stop has conflicting sidecar authorizations"
            )
        self._user_stop_authorization = authorization

    def observe_event(self, event: dict[str, Any]) -> None:
        try:
            self._observe_event(event)
        except Exception as exc:
            failure = f"{str(event.get('type') or 'unknown')}: {exc}"
            if failure not in self._fatal_errors:
                self._fatal_errors.append(failure)
            raise

    def _observe_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "run_started":
            expected_start = self.expected_iterations[0]
            expected_end = expected_start + len(self.expected_iterations)
            iterations = event.get("iterations")
            start_iter = event.get("start_iter")
            end_iter = event.get("end_iter")
            if (
                type(iterations) is not int
                or type(start_iter) is not int
                or type(end_iter) is not int
                or iterations != len(self.expected_iterations)
                or start_iter != expected_start
                or end_iter != expected_end
            ):
                raise RunLifecycleError(
                    "worker run plan differs from the pre-spawn iteration plan"
                )
            digest = _canonical_sha256(event)
            if self._run_started_sha256 not in {None, digest}:
                raise RunLifecycleError(
                    "worker emitted conflicting run_started evidence"
                )
            self._run_started_sha256 = digest
            return

        if event_type == "iter_started":
            if self._run_started_sha256 is None:
                raise RunLifecycleError(
                    "worker iteration began before the exact run plan"
                )
            if self._early_stop is not None:
                raise RunLifecycleError(
                    "worker began an iteration after its terminal early stop"
                )
            index = event.get("iter")
            next_position = len(self._started)
            if next_position >= len(self.expected_iterations):
                raise RunLifecycleError(
                    "worker started more iterations than requested"
                )
            expected = self.expected_iterations[next_position]
            if type(index) is not int or index != expected:
                raise RunLifecycleError(
                    f"worker started iteration {index!r}, expected {expected}"
                )
            if self._started and self._started[-1] not in self._completed:
                raise RunLifecycleError(
                    "worker started a new iteration before completing the prior one"
                )
            self._started.append(expected)
            return

        if event_type == "iter_completed":
            index = event.get("iter")
            if (
                type(index) is not int
                or not self._started
                or index != self._started[-1]
            ):
                raise RunLifecycleError(
                    "worker completion has no matching terminal iter_started"
                )
            digest = _canonical_sha256(event)
            prior = self._completed.get(index)
            if prior is not None and prior != digest:
                raise RunLifecycleError(
                    "worker emitted conflicting iter_completed evidence"
                )
            self._completed[index] = digest
            return

        if event_type == "early_stop":
            index = event.get("at_iter")
            source = event.get("source")
            reason = event.get("reason")
            if (
                type(index) is not int
                or not self._started
                or index != self._started[-1]
                or index not in self._completed
            ):
                raise RunLifecycleError(
                    "early stop has no completed terminal iteration"
                )
            authorized_sources = set(self.allowed_early_stop_sources)
            if self._user_stop_authorization is not None:
                authorized_sources.add("user")
            if (
                source not in authorized_sources
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise RunLifecycleError(
                    "early stop was not independently authorized at launch"
                )
            evidence = {
                "at_iter": index,
                "source": source,
                "reason": reason.strip(),
                "event_sha256": _canonical_sha256(event),
            }
            if self._early_stop is not None and self._early_stop != evidence:
                raise RunLifecycleError(
                    "worker emitted conflicting early-stop evidence"
                )
            self._early_stop = evidence

    def finalize_proof(self) -> dict[str, Any]:
        if self._fatal_errors:
            raise RunLifecycleError(
                "worker lifecycle contains rejected evidence: "
                + " | ".join(self._fatal_errors)
            )
        if self._run_started_sha256 is None:
            raise RunLifecycleError("worker emitted no exact run_started plan")
        if not self._started:
            raise RunLifecycleError("worker completed no requested iteration")
        if list(self._completed) != self._started:
            raise RunLifecycleError(
                "worker has an iteration without exact completion evidence"
            )
        completed_plan = tuple(self._started)
        expected_prefix = self.expected_iterations[: len(completed_plan)]
        if completed_plan != expected_prefix:
            raise RunLifecycleError(
                "worker completed iterations outside the requested plan"
            )
        if completed_plan != self.expected_iterations:
            if (
                self._early_stop is None
                or self._early_stop["at_iter"] != completed_plan[-1]
            ):
                raise RunLifecycleError(
                    "worker did not complete the requested plan or an "
                    "authorized early stop"
                )
        effective_sources = list(self.allowed_early_stop_sources)
        if self._user_stop_authorization is not None:
            effective_sources.append("user")
        iteration_plan = {
            "requested": list(self.expected_iterations),
            "completed": list(completed_plan),
            "allowed_early_stop_sources": effective_sources,
            "early_stop": self._early_stop,
        }
        if self._user_stop_authorization is not None:
            iteration_plan["user_stop_authorization"] = (
                self._user_stop_authorization
            )
        receipt = {
            "schema": 1,
            "authority": "worker_iteration_lifecycle_verified",
            "run_id": self.run_id,
            "run_started_event_sha256": self._run_started_sha256,
            "iteration_plan": iteration_plan,
            "iter_completed_event_sha256": [
                self._completed[index] for index in completed_plan
            ],
        }
        receipt["proof_sha256"] = _canonical_sha256(receipt)
        return receipt


__all__ = [
    "RunLifecycleError",
    "RunLifecycleSession",
    "TERMINAL_RECEIPTS_DIR",
    "build_terminal_run_receipt",
    "terminal_run_receipt_path",
    "verified_lifecycle_completed_iterations",
    "verify_terminal_run_receipt",
    "write_terminal_run_receipt",
]
