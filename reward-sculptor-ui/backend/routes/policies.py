"""Trained-policy endpoints — listing + deployment-bundle export.

  GET /projects/{slug}/policies                      — exportable iterations
  GET /projects/{slug}/policies/{iter_index}/export  — build + download zip

Disk is the source of truth (runs/iter_*/checkpoint.*), so policies stay
listable and exportable after a backend restart even though JobManager
runs are in-memory. An optional `run_id` query scopes both endpoints to a
mission stage's own runs tree (.missions/<m>/stages/<s>/runs) via the same
resolution the runs router uses.

Bundle building is synchronous — a bundle is a zip of small files plus an
ONNX/TorchScript trace of a ~500k-param MLP, a few seconds at worst
(same trade-off as POST /reports/build).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse

from backend.models.project import ProblemDetail
from backend.models.run import PolicyRecoverySnapshot, PolicySummary
from backend.routes.runs import _find_run, _resolve_run_root
from backend.services.iteration_completion import is_completed_iteration
from backend.services.job_manager import JobManager
from backend.services.project_store import ProjectStore

router = APIRouter(tags=["policies"])


def get_store(request: Request) -> ProjectStore:
    return request.app.state.project_store


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _problem(code: int, title: str, **extra: Any) -> JSONResponse:
    body = ProblemDetail(title=title, status=code, **{
        k: v for k, v in extra.items() if k in {"detail", "type", "instance"}
    }).model_dump()
    body.update({k: v for k, v in extra.items() if k not in body})
    return JSONResponse(
        status_code=code, content=body,
        media_type="application/problem+json")


def _resolve_roots(
    slug: str,
    run_id: Optional[str],
    store: ProjectStore,
    jobs: JobManager,
) -> tuple[Optional[Path], Optional[Path], Optional[JSONResponse]]:
    """(project_dir, runs_root, error). Mirrors the runs router's
    resolution so stage runs export from their own runs tree."""
    detail = store.get(slug)
    if detail is None:
        return None, None, _problem(
            404, "project not found",
            detail=f"no project with slug {slug!r}",
            type="/problems/not-found")
    project_dir = Path(detail.project_dir)
    if run_id is None:
        return project_dir, project_dir / "runs", None
    job = _find_run(jobs, slug, run_id)
    if job is None:
        return None, None, _problem(
            404, "run not found",
            detail=f"no run with id {run_id!r} in project {slug!r} "
                   "(runs are in-memory; after a backend restart omit "
                   "run_id to export from the project's runs on disk)",
            type="/problems/not-found")
    return project_dir, _resolve_run_root(job, project_dir), None


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _selection_candidates(
    runs_root: Path,
) -> tuple[Optional[int], Optional[str], dict[int, dict[str, Any]]]:
    """Read a canonical keep-best receipt without guessing from recency.

    A partially-written or hand-edited document is not enough to preselect a
    policy: the top-level index/source and matching candidate marker must all
    agree. This is intentionally stricter than merely finding the newest
    checkpoint.
    """
    doc = _load_json_object(runs_root.parent / "reports" / "selection.json")
    selected = doc.get("selected_iter_index")
    source = doc.get("selection_source")
    rows = doc.get("candidates")
    candidates: dict[int, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("iter_index")
            if isinstance(index, int) and not isinstance(index, bool):
                candidates[index] = row
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or not isinstance(source, str)
        or not source.strip()
        or candidates.get(selected, {}).get("selected") is not True
    ):
        return None, None, candidates
    return selected, source.strip(), candidates


_EVIDENCE_SEMANTICS: dict[str, tuple[str, float]] = {
    "route_complete_frac": ("gte", 1.0),
    "actual_route_complete_frac": ("gte", 1.0),
    "order_ok_frac": ("gte", 1.0),
    "success_seen_frac": ("gte", 1.0),
    "completion_gate": ("gte", 1.0),
    "contact_free_frac": ("gte", 1.0),
    "forbidden_contact_free_frac": ("gte", 1.0),
    "contact_frac": ("lte", 0.0),
    "forbidden_contact_count": ("lte", 0.0),
    "strict_hold_frac": ("gte", 1.0),
    "hold_ok_frac": ("gte", 1.0),
    "full_hold_frac": ("gte", 1.0),
    "hold_frac": ("gte", 1.0),
    "strict_hold_count": ("gte", 1.0),
    "ch_hold": ("gte", 1.0),
}


def _comparison_passed(value: float, comparison: str, threshold: float) -> bool:
    if comparison == "gte":
        return value >= threshold
    if comparison == "lte":
        return value <= threshold
    return value == threshold


def _evidence_value(
    components: dict[str, Any],
    keys: tuple[str, ...],
    *,
    semantics: Optional[dict[str, tuple[str, float, str]]] = None,
) -> Optional[dict[str, Any]]:
    for key in keys:
        value = components.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        if key.endswith("_frac") or key in {"completion_gate"}:
            kind = "fraction"
        elif "frame" in key:
            kind = "frames"
        elif "count" in key:
            kind = "count"
        elif "score" in key or key.startswith("ch_"):
            kind = "score"
        else:
            kind = "value"
        result: dict[str, Any] = {
            "key": key, "value": float(value), "kind": kind,
            "comparison": None, "threshold": None, "passed": None,
            "semantics_source": None,
        }
        declared = (semantics or {}).get(key)
        if declared is None and key in _EVIDENCE_SEMANTICS:
            comparison, threshold = _EVIDENCE_SEMANTICS[key]
            declared = (
                comparison, threshold,
                "reward-sculptor-objective-evidence-semantics-v1",
            )
        if declared is not None:
            comparison, threshold, source = declared
            result.update({
                "comparison": comparison,
                "threshold": float(threshold),
                "passed": _comparison_passed(
                    float(value), comparison, float(threshold),
                ),
                "semantics_source": source,
            })
        return result
    return None


def _metric_text(metric: dict[str, Any], key: str) -> Optional[str]:
    value = metric.get(key)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return None
        text = str(value).strip()
        return text or None
    return None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _checkpoint_identity(
    runs_root: Path,
    row: dict[str, Any],
) -> tuple[str, int]:
    """Hash the exact server-owned checkpoint selected by a listing row.

    ``list_exportable_iters`` is an inventory helper, not an identity
    authority. Re-resolve and validate its path here so a policy is never
    presented as selectable from a filename, symlink, stale byte count, or a
    path outside its immutable iteration directory.
    """

    iter_index = int(row["iter_index"])
    checkpoint_name = row.get("checkpoint")
    if checkpoint_name not in {"checkpoint.pt", "checkpoint.zip"}:
        raise ValueError("checkpoint member is not an admitted filename")

    root = runs_root.resolve()
    iter_dir = (root / f"iter_{iter_index}").resolve(strict=True)
    if iter_dir.parent != root or not iter_dir.is_dir():
        raise ValueError("iteration directory escapes the runs root")

    checkpoint = iter_dir / checkpoint_name
    if checkpoint.is_symlink():
        raise ValueError("checkpoint symlinks are not admitted")
    resolved = checkpoint.resolve(strict=True)
    if resolved.parent != iter_dir or not resolved.is_file():
        raise ValueError("checkpoint escapes its iteration directory")

    size = resolved.stat().st_size
    listed_size = row.get("checkpoint_bytes")
    if (
        not isinstance(listed_size, int)
        or isinstance(listed_size, bool)
        or listed_size <= 0
        or listed_size != size
    ):
        raise ValueError("checkpoint byte count changed during listing")

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), size


def _rollout_lane_receipt(iter_dir: Path) -> dict[str, Any]:
    """Read the worker-authored identity of the rendered evidence lane.

    The launch request is not evidence that the requested lane was actually
    rendered. Only ``behavior.json`` can resolve that fact, and a clamp or
    missing percentile must remain visible rather than being filled from run
    parameters or an arbitrary rollout row.
    """
    behavior = _load_json_object(iter_dir / "rollout" / "behavior.json")

    def _index(key: str) -> Optional[int]:
        value = behavior.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            return value
        return None

    requested = _index("rendered_env_index_requested")
    resolved = _index("rendered_env_index")
    selection_raw = behavior.get("rendered_env_selection")
    selection = (
        selection_raw.strip()
        if isinstance(selection_raw, str) and selection_raw.strip()
        else None
    )
    percentile_raw = behavior.get("rendered_episode_percentile")
    percentile = (
        float(percentile_raw)
        if (
            isinstance(percentile_raw, (int, float))
            and not isinstance(percentile_raw, bool)
            and math.isfinite(float(percentile_raw))
            and 0.0 <= float(percentile_raw) <= 1.0
        )
        else None
    )

    raw_fields_present = any(
        key in behavior
        for key in (
            "rendered_env_index_requested",
            "rendered_env_index",
            "rendered_env_selection",
            "rendered_episode_percentile",
        )
    )
    if not raw_fields_present:
        status = "unavailable"
    elif requested is not None and resolved is not None and requested != resolved:
        status = "mismatch"
    elif selection is not None and selection != "precommitted":
        status = "mismatch"
    elif (
        requested is not None
        and resolved is not None
        and requested == resolved
        and selection == "precommitted"
        and percentile is not None
    ):
        status = "verified"
    else:
        status = "incomplete"

    return {
        "lane_evidence_status": status,
        "requested_evidence_env_index": requested,
        "resolved_evidence_env_index": resolved,
        "resolved_episode_percentile": percentile,
        "evidence_lane_selection": selection,
    }


def _objective_proof_decision(
    *,
    route: Optional[dict[str, Any]],
    contact: Optional[dict[str, Any]],
    hold: Optional[dict[str, Any]],
    criterion_status: str,
    lane_receipt: dict[str, Any],
    rollout_available: bool,
    metric_identity_complete: bool,
) -> tuple[str, list[str]]:
    """Interpret objective evidence using only declared comparisons.

    Presence is not success. Unknown component names and frame counts without
    a worker-authored threshold remain incomplete rather than being guessed.
    """
    blockers: list[str] = []
    failed = False
    for label, evidence in (
        ("route", route), ("forbidden contact", contact),
        ("terminal hold", hold),
    ):
        if evidence is None:
            blockers.append(f"{label} evidence is missing")
        elif evidence.get("passed") is False:
            failed = True
            blockers.append(
                f"{label} evidence failed its declared comparison"
            )
        elif evidence.get("passed") is not True:
            blockers.append(
                f"{label} evidence has no explicit comparison semantics"
            )
    if criterion_status == "failed":
        failed = True
        blockers.append("objective criterion failed")
    elif criterion_status != "passed":
        blockers.append("objective criterion was not recorded")
    lane_status = lane_receipt.get("lane_evidence_status")
    if lane_status == "mismatch":
        failed = True
        blockers.append("requested and resolved evidence lane disagree")
    elif lane_status != "verified":
        blockers.append("worker-authored evidence lane receipt is incomplete")
    if not rollout_available:
        blockers.append("rollout artifact is missing")
    if not metric_identity_complete:
        blockers.append("exact objective metric identity is incomplete")
    if failed:
        return "failed", blockers
    return ("passed", []) if not blockers else ("incomplete", blockers)


def _policy_receipt_fields(
    runs_root: Path,
    row: dict[str, Any],
    *,
    selected_iter: Optional[int],
    selection_source: Optional[str],
    selection_candidates: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    iter_index = int(row["iter_index"])
    iter_dir = runs_root / f"iter_{iter_index}"
    fitness_doc = _load_json_object(iter_dir / "fitness.json")
    components = fitness_doc.get("components")
    if not isinstance(components, dict):
        components = {}
    metric = fitness_doc.get("metric")
    if not isinstance(metric, dict):
        metric = {}
    behavior = _load_json_object(iter_dir / "rollout" / "behavior.json")
    terminal_contract = behavior.get("terminal_proof_contract")
    hold_semantics: dict[str, tuple[str, float, str]] = {}
    if isinstance(terminal_contract, dict):
        minimum_hold_frames = terminal_contract.get("minimum_hold_frames")
        if (
            isinstance(minimum_hold_frames, (int, float))
            and not isinstance(minimum_hold_frames, bool)
            and math.isfinite(float(minimum_hold_frames))
            and float(minimum_hold_frames) > 0
        ):
            for key in ("hold_frames", "proof_frames"):
                hold_semantics[key] = (
                    "gte", float(minimum_hold_frames),
                    "behavior.terminal_proof_contract.minimum_hold_frames",
                )

    route = _evidence_value(components, (
        "route_complete_frac", "actual_route_complete_frac",
        "order_ok_frac", "success_seen_frac", "completion_gate",
    ))
    contact = _evidence_value(components, (
        "contact_free_frac", "forbidden_contact_free_frac",
        "contact_frac", "forbidden_contact_count",
    ))
    contact_gate = components.get("contact_evidence_ok")
    if (
        contact is not None
        and isinstance(contact_gate, (int, float))
        and not isinstance(contact_gate, bool)
        and math.isfinite(float(contact_gate))
        and float(contact_gate) <= 0
    ):
        contact["passed"] = False
        contact["semantics_source"] = (
            f"{contact.get('semantics_source') or 'unavailable'}"
            "+contact_evidence_ok"
        )
    hold = _evidence_value(components, (
        "strict_hold_frac", "hold_ok_frac", "full_hold_frac", "hold_frac",
        "strict_hold_count", "hold_frames", "proof_frames", "ch_hold",
    ), semantics=hold_semantics)
    evidence_count = sum(value is not None for value in (route, contact, hold))
    evidence_status = (
        "complete" if evidence_count == 3
        else "partial" if evidence_count > 0
        else "unavailable"
    )

    candidate = selection_candidates.get(iter_index, {})
    criterion = candidate.get("criterion_pass")
    criterion_status = (
        "passed" if criterion is True
        else "failed" if criterion is False
        else "not_recorded"
    )
    lane_receipt = _rollout_lane_receipt(iter_dir)
    rollout_available = _nonempty_file(
        iter_dir / "rollout" / "rollout.mp4"
    )
    metric_id = _metric_text(metric, "id")
    metric_version = _metric_text(metric, "version")
    metric_source = _metric_text(metric, "source")
    metric_sha256 = _metric_text(metric, "sha256")
    metric_identity_complete = bool(
        metric_id
        and metric_source
        and metric_sha256
        and len(metric_sha256) == 64
        and all(char in "0123456789abcdef" for char in metric_sha256)
    )
    objective_proof_status, objective_proof_blockers = (
        _objective_proof_decision(
            route=route,
            contact=contact,
            hold=hold,
            criterion_status=criterion_status,
            lane_receipt=lane_receipt,
            rollout_available=rollout_available,
            metric_identity_complete=metric_identity_complete,
        )
    )
    return {
        "deployable": True,
        "metric_id": metric_id,
        "metric_version": metric_version,
        "metric_source": metric_source,
        "metric_sha256": metric_sha256,
        "criterion_status": criterion_status,
        "evidence_status": evidence_status,
        "route_evidence": route,
        "contact_evidence": contact,
        "hold_evidence": hold,
        "objective_proof_status": objective_proof_status,
        "objective_proof_blockers": objective_proof_blockers,
        **lane_receipt,
        "rollout_available": rollout_available,
        "selected": selected_iter == iter_index,
        "selection_source": (
            selection_source if selected_iter == iter_index else None
        ),
    }


# ── GET /projects/{slug}/policies ─────────────────────────────────────
@router.get(
    "/projects/{slug}/policies",
    response_model=list[PolicySummary],
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def list_policies(
    slug: str,
    run_id: Optional[str] = None,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    _, runs_root, err = _resolve_roots(slug, run_id, store, jobs)
    if err is not None:
        return err
    try:
        from sculptor.export import list_exportable_iters
    except Exception as e:  # noqa: BLE001
        return _problem(
            503, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable")
    selected, selection_source, candidates = _selection_candidates(runs_root)
    policies: list[PolicySummary] = []
    for row in list_exportable_iters(runs_root):
        if not is_completed_iteration(
            runs_root / f"iter_{int(row['iter_index'])}"
        ):
            continue
        try:
            checkpoint_sha256, checkpoint_bytes = _checkpoint_identity(
                runs_root, row,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return _problem(
                409,
                "policy checkpoint identity unavailable",
                detail=(
                    "The retained checkpoint could not be pinned to its "
                    "server-owned iteration bytes. Refresh or repair the "
                    f"iteration before selecting it ({type(exc).__name__})."
                ),
                type="/problems/policy-checkpoint-identity-unavailable",
            )
        pinned_row = dict(row)
        pinned_row["checkpoint_bytes"] = checkpoint_bytes
        policies.append(PolicySummary(
            **pinned_row,
            checkpoint_sha256=checkpoint_sha256,
            **_policy_receipt_fields(
                runs_root,
                row,
                selected_iter=selected,
                selection_source=selection_source,
                selection_candidates=candidates,
            ),
        ))
    return policies


# ── GET /projects/{slug}/policies/recovery-snapshots ────────────────────
@router.get(
    "/projects/{slug}/policies/recovery-snapshots",
    response_model=list[PolicyRecoverySnapshot],
    responses={
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
    },
)
def list_recovery_snapshots(
    slug: str,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    """List attested, unevaluated PPO saves from interrupted local runs.

    This route is deliberately separate from ``/policies`` and export.  It
    returns opaque ids and digests only; the server-owned checkpoint path is
    never part of the browser contract.
    """
    detail = store.get(slug)
    if detail is None:
        return _problem(
            404,
            "project not found",
            detail=f"no project with slug {slug!r}",
            type="/problems/not-found",
        )
    if jobs.has_active_sculpt_run(slug):
        return _problem(
            409,
            "recovery snapshots are unavailable while training is active",
            detail=(
                "wait for the active project worker to stop before selecting "
                "an interrupted PPO snapshot"
            ),
            type="/problems/recovery-snapshot-active",
        )
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    return [
        PolicyRecoverySnapshot(**row)
        for row in discover_recovery_snapshots(Path(detail.project_dir))
    ]


# ── GET /projects/{slug}/policies/{iter_index}/export ─────────────────
@router.get(
    "/projects/{slug}/policies/{iter_index}/export",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/zip": {}}},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        500: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
def export_policy(
    slug: str,
    iter_index: int,
    run_id: Optional[str] = None,
    store: ProjectStore = Depends(get_store),
    jobs: JobManager = Depends(get_job_manager),
) -> Any:
    project_dir, runs_root, err = _resolve_roots(slug, run_id, store, jobs)
    if err is not None:
        return err
    if iter_index < 0:
        return _problem(
            404, "iteration not found",
            detail="iter_index must be >= 0", type="/problems/not-found")
    iter_dir = runs_root / f"iter_{iter_index}"
    has_checkpoint = any(
        _nonempty_file(iter_dir / name)
        for name in ("checkpoint.pt", "checkpoint.zip")
    )
    if has_checkpoint and not is_completed_iteration(iter_dir):
        return _problem(
            409,
            "policy evaluation is incomplete",
            detail=(
                f"iter {iter_index} preserved a checkpoint but has no valid "
                "completion marker or full legacy rollout/fitness evidence; "
                "use the interrupted-snapshot recovery flow instead"
            ),
            type="/problems/policy-evaluation-incomplete",
        )
    try:
        from sculptor.export import ExportError, export_policy_bundle
    except Exception as e:  # noqa: BLE001
        return _problem(
            503, "sculptor unavailable",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/sculptor-unavailable")

    try:
        result = export_policy_bundle(
            project_dir, iter_index=iter_index, runs_root=runs_root)
    except ExportError as e:
        return _problem(
            404, "no exportable checkpoint",
            detail=str(e), type="/problems/not-found")
    except Exception as e:  # noqa: BLE001 — degrade loudly, never 500-blank
        return _problem(
            500, "export failed",
            detail=f"{type(e).__name__}: {e}",
            type="/problems/export-failed")

    return FileResponse(
        result.bundle_path,
        media_type="application/zip",
        filename=result.bundle_path.name,
    )
