"""Immutable readiness receipts for per-mode objective evidence.

The core per-mode gauntlet is deliberately independent from reward shaping,
but until this service existed the production UI could not answer whether the
active phase reward had any per-mode objective evidence at all.  This module
does not manufacture evidence: it re-attests the exact active clip, robot,
reward, selection, graph, and execution manifest, then records the gauntlet's
honest absent/observe-only state as a content-addressed receipt.

Future metric generation can replace the ``absent`` evidence payload with
validated and calibrated artifacts without changing this authority contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Optional


class ModeEvidenceError(ValueError):
    """The active project state cannot support an exact evidence receipt."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_created_at(value: Optional[float]) -> float:
    created = time.time() if value is None else float(value)
    if not math.isfinite(created) or created < 0:
        raise ModeEvidenceError("created_at must be a finite non-negative value")
    return created


def _project_relative(project_dir: Path, path: str | Path, *, label: str) -> str:
    """Return a stable project-relative identity and reject path escape."""
    try:
        return Path(path).resolve().relative_to(project_dir).as_posix()
    except (OSError, ValueError) as exc:
        raise ModeEvidenceError(f"{label} must stay inside the project") from exc


def _resolve_authority(
    project_dir: Path,
    *,
    expected_clip_id: str,
    expected_robot: str,
) -> tuple[dict[str, Any], Any, list[str]]:
    """Resolve the exact active phase-reward authority without importing it."""
    from sculptor.eval.mode_metrics import mode_gauntlet_report
    from sculptor.mode_rewards import (
        MODE_BINDING_CONTEXT_REFS,
        authored_modes,
        mode_execution_manifest_digest,
        mode_reward_binding_errors,
        reward_spec_from_source,
    )
    from sculptor.modes import mode_graph_sha256, modes_from_composition
    from sculptor.reference import load_clip
    from sculptor.reference_authority import (
        ActiveReferenceAuthorityError,
        resolve_active_reference_authority,
    )
    from sculptor.refs.library import CLIP_FILENAME, clip_dir
    from sculptor.world.artifacts import WorldArtifactStore, file_sha256

    project_dir = Path(project_dir).resolve()
    try:
        active = resolve_active_reference_authority(project_dir / "rewards")
    except ActiveReferenceAuthorityError as exc:
        raise ModeEvidenceError(str(exc)) from exc
    if active is None or active.kind != "mode_reference":
        raise ModeEvidenceError(
            "current.py does not select an exact promoted per-mode reward"
        )
    if active.reference_clip_id != expected_clip_id:
        raise ModeEvidenceError(
            "the panel clip is not the active reward clip: "
            f"{expected_clip_id!r} != {active.reference_clip_id!r}"
        )
    if active.reference_robot != expected_robot:
        raise ModeEvidenceError(
            "the panel robot is not the active reward robot: "
            f"{expected_robot!r} != {active.reference_robot!r}"
        )

    reward_path = Path(active.reward_path)
    try:
        reward_source = reward_path.read_text(encoding="utf-8")
        spec = reward_spec_from_source(reward_source)
    except (OSError, TypeError, ValueError) as exc:
        raise ModeEvidenceError(f"cannot parse active reward: {exc}") from exc

    clip_path = clip_dir(expected_robot, expected_clip_id) / CLIP_FILENAME
    if not clip_path.is_file():
        raise ModeEvidenceError(f"active reference clip is missing: {clip_path}")
    clip_sha256 = file_sha256(clip_path)
    if clip_sha256 != active.reference_clip_sha256:
        raise ModeEvidenceError(
            "active reference bytes no longer match the promoted reward binding"
        )
    try:
        clip = load_clip(clip_path)
        graph = modes_from_composition(clip, clip_id=expected_clip_id)
    except (OSError, TypeError, ValueError) as exc:
        raise ModeEvidenceError(f"cannot derive the active mode graph: {exc}") from exc

    graph_sha256 = mode_graph_sha256(graph)
    manifest = spec.get("mode_execution_manifest")
    if not isinstance(manifest, dict):
        raise ModeEvidenceError("active reward has no execution manifest")
    try:
        manifest_sha256 = mode_execution_manifest_digest(manifest)
    except (TypeError, ValueError) as exc:
        raise ModeEvidenceError(f"active execution manifest is invalid: {exc}") from exc

    blockers: list[str] = []
    selection_payload: dict[str, Any]
    context_refs: dict[str, str] = {}
    selection_path = project_dir / "env" / "selection_current.json"
    try:
        selection = WorldArtifactStore(project_dir).read_selection(selection_path)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        selection = None
        selection_payload = {
            "present": selection_path.is_file(),
            "valid": False,
            "sha256": file_sha256(selection_path) if selection_path.is_file() else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        blockers.append("the authoritative selection is unreadable or hash-drifted")
    else:
        if selection is None:
            selection_payload = {
                "present": False,
                "valid": False,
                "sha256": None,
                "error": None,
            }
            blockers.append("the project has no authoritative selection")
        else:
            refs = {
                key: {
                    "version": ref.version,
                    "path": ref.path,
                    "sha256": ref.sha256,
                }
                for key, ref in sorted(selection.refs.items())
            }
            selection_payload = {
                "present": True,
                "valid": True,
                "selection_version": selection.selection_version,
                "tuple_hash": selection.tuple_hash,
                "evaluation_lineage": selection.evaluation_lineage,
                "sha256": file_sha256(selection_path),
                "refs": refs,
            }
            context_refs = {
                key: selection.refs[key].sha256
                for key in MODE_BINDING_CONTEXT_REFS
                if key in selection.refs
            }
            reward_ref = selection.refs.get("reward")
            if reward_ref is None or reward_ref.sha256 != active.reward_sha256:
                blockers.append(
                    "selection_current does not pin the active reward bytes"
                )

    binding_errors = mode_reward_binding_errors(
        spec,
        clip_id=expected_clip_id,
        robot=expected_robot,
        clip_sha256=clip_sha256,
        context_refs=context_refs,
        graph=graph,
        reward_source=reward_source,
    )
    if binding_errors:
        raise ModeEvidenceError(
            "active reward binding is stale: " + "; ".join(binding_errors)
        )

    authored = authored_modes(reward_source)
    unauthored = [name for name, done in authored.items() if not done]
    if unauthored:
        blockers.append(
            "the active reward still has unauthored modes: " + ", ".join(unauthored)
        )

    authority = {
        "schema": 1,
        "clip_id": expected_clip_id,
        "reference_robot": expected_robot,
        "clip_sha256": clip_sha256,
        "reward_path": _project_relative(
            project_dir, reward_path, label="active reward path"
        ),
        "reward_sha256": active.reward_sha256,
        "selector_path": _project_relative(
            project_dir, active.selector_path, label="reward selector path"
        ),
        "selector_sha256": active.selector_sha256,
        "graph_sha256": graph_sha256,
        "execution_manifest_sha256": manifest_sha256,
        "mode_binding": spec.get("mode_binding"),
        "selection": selection_payload,
    }
    authority["context_sha256"] = _digest(authority)
    return authority, mode_gauntlet_report(graph), blockers


def build_readiness_receipt(
    project_dir: Path,
    *,
    expected_clip_id: str,
    expected_robot: str,
    created_at: Optional[float] = None,
) -> dict[str, Any]:
    """Build an honest receipt for the production evidence currently present.

    No per-mode objective metric registry exists yet, so this receipt is
    necessarily observe-only.  Crucially, the absence is represented by the
    same exact authority a future validated/calibrated receipt must use.
    """
    authority, report, blockers = _resolve_authority(
        project_dir,
        expected_clip_id=expected_clip_id,
        expected_robot=expected_robot,
    )
    blockers = list(blockers)
    blockers.append(
        "no generated, validated, and calibrated per-mode objective metric set "
        "is registered for this exact execution context"
    )
    receipt: dict[str, Any] = {
        "schema": 1,
        "created_at": _safe_created_at(created_at),
        "authority": authority,
        "trust_status": "observe_only",
        "evidence_status": "absent",
        "fitness_or_selection_authority": False,
        "training_consumer_active": False,
        "blockers": blockers,
        "gauntlet": report,
        "next_action": (
            "Generate one objective metric per mode, validate each against its "
            "own reference slice, calibrate each competence ladder, then bind "
            "those immutable artifacts to this context digest."
        ),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def _receipt_root(project_dir: Path) -> Path:
    return Path(project_dir) / "mode_evidence"


def persist_readiness_receipt(project_dir: Path, receipt: dict[str, Any]) -> Path:
    """Persist one receipt at a digest-derived path; never overwrite bytes."""
    expected = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if not isinstance(expected, str) or expected != _digest(unsigned):
        raise ModeEvidenceError("receipt digest does not match its content")
    context = ((receipt.get("authority") or {}).get("context_sha256"))
    if not isinstance(context, str) or len(context) != 64:
        raise ModeEvidenceError("receipt has no exact authority context digest")
    path = _receipt_root(project_dir) / context / f"{expected}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(receipt) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ModeEvidenceError("immutable receipt path contains different bytes")
    return path


def latest_receipt(project_dir: Path, context_sha256: str) -> Optional[dict[str, Any]]:
    """Read the newest valid receipt for one exact context."""
    root = _receipt_root(project_dir) / context_sha256
    candidates: list[dict[str, Any]] = []
    if root.is_dir():
        for path in root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                unsigned = dict(record)
                claimed = unsigned.pop("receipt_sha256")
                if claimed != path.stem or claimed != _digest(unsigned):
                    continue
                candidates.append(record)
            except (OSError, TypeError, ValueError, KeyError):
                continue
    return max(candidates, key=lambda row: float(row.get("created_at", 0))) if candidates else None


def status(
    project_dir: Path,
    *,
    expected_clip_id: str,
    expected_robot: str,
) -> dict[str, Any]:
    """Return exact-context status; absence is a first-class result."""
    current = build_readiness_receipt(
        project_dir,
        expected_clip_id=expected_clip_id,
        expected_robot=expected_robot,
    )
    context = current["authority"]["context_sha256"]
    persisted = latest_receipt(project_dir, context)
    if persisted is None:
        current["recorded"] = False
        return current
    return dict(persisted, recorded=True)
