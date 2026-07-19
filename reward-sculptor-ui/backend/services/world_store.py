"""Environment-authoring store: prompt → draft + clarifications →
admitted atomic world tuple, plus selection/lineage reads.

Statefulness contract: ``apply_clarifications`` needs the ACTUAL
``AuthoringDraft`` whose ``draft_hash``/``question_set_hash`` match the
submission. The offline author is deterministic, so this store persists
the authoring INPUTS (prompt / capability / grounding) together with the
draft dict under ``<project>/worlds/<session>/`` and re-authors at apply
time, verifying the recomputed ``draft_hash`` against the persisted one
— a mismatch (changed capability descriptors, upgraded author contract)
raises ``StaleDraftError`` instead of silently applying answers to a
different question set.

Only the offline deterministic author is wired here; if an LLM author
model is ever added, the re-author trick stops working and the draft
must be held server-side instead.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

_SESSIONS_DIRNAME = "worlds"

#: Abandoned drafts are pruned on the next author() call once this old.
_SESSION_TTL_S = 14 * 24 * 3600.0


class StaleDraftError(RuntimeError):
    """The re-authored draft no longer matches the persisted session."""


class UnknownSessionError(KeyError):
    """No persisted authoring session with that id."""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _sessions_dir(project_dir: Path) -> Path:
    return Path(project_dir) / _SESSIONS_DIRNAME


def _session_path(project_dir: Path, session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe or safe != session_id:
        raise UnknownSessionError(session_id)
    return _sessions_dir(project_dir) / safe / "session.json"


def _prune_stale_sessions(project_dir: Path) -> None:
    """Best-effort TTL prune of abandoned authoring sessions (verifier
    finding: nothing else ever removed them). Applied sessions carry an
    applied.json but are pruned too — the promoted tuple in env/ is the
    durable record; the session dir is only working state."""
    import shutil

    root = _sessions_dir(project_dir)
    if not root.is_dir():
        return
    cutoff = time.time() - _SESSION_TTL_S
    for entry in root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:  # pragma: no cover — prune must never block authoring
            continue


def author(
    project_dir: Path,
    prompt: str,
    *,
    robot_capability_id: str | None = None,
    kg_grounding: bool = True,
) -> dict[str, Any]:
    """Author a draft + clarification plan; persist the session inputs."""
    from sculptor.world.author import author_environment
    from sculptor.world.grounding import (
        gather_grounding,
        grounding_context,
        grounding_ids,
    )

    _prune_stale_sessions(project_dir)
    items = gather_grounding(prompt) if kg_grounding else ()
    ids = grounding_ids(items)
    context = grounding_context(items)
    draft = author_environment(
        prompt, robot_capability_id=robot_capability_id,
        grounding=ids, grounding_context=context)

    session_id = draft.draft_hash[:24]
    session_dir = _sessions_dir(project_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(session_dir / "session.json", {
        "session_id": session_id,
        "draft_hash": draft.draft_hash,
        "capability_id": draft.capability_id,
        "created_at": time.time(),
        "inputs": {
            "prompt": prompt,
            "robot_capability_id": robot_capability_id,
            "grounding": ids,
            "grounding_context": context,
        },
    })
    _atomic_write_json(session_dir / "draft.json", draft.to_dict())

    return {
        "session_id": session_id,
        "draft_hash": draft.draft_hash,
        "capability_id": draft.capability_id,
        "clarification_plan": draft.clarification_plan.to_dict(),
        "underspecification_report":
            draft.underspecification_report.to_dict(),
        "kg_grounding": ids,
    }


def _load_session(project_dir: Path, session_id: str) -> dict[str, Any]:
    path = _session_path(project_dir, session_id)
    if not path.is_file():
        raise UnknownSessionError(session_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _reauthor(session: Mapping[str, Any]):
    from sculptor.world.author import author_environment

    inputs = session.get("inputs") or {}
    return author_environment(
        str(inputs.get("prompt") or ""),
        robot_capability_id=inputs.get("robot_capability_id") or None,
        grounding=tuple(inputs.get("grounding") or ()),
        grounding_context=tuple(inputs.get("grounding_context") or ()),
    )


def apply(
    project_dir: Path,
    session_id: str,
    answers: Iterable[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Apply clarification answers and atomically admit + promote.

    Unanswered questions take their disclosed system default (the
    clarifier protocol's "system decides" option) with ``default``
    provenance; explicit answers record ``user`` provenance. Unknown
    question ids raise ``ValueError`` (a stale/foreign answer set must
    not be silently ignored).
    """
    from sculptor.world.author import (
        CLARIFICATION_VERSION,
        ClarificationAnswer,
        ClarificationSubmission,
        apply_clarifications,
    )
    from sculptor.world.project import (
        WorldProjectService,
        evaluation_lineage_for,
    )

    session = _load_session(project_dir, session_id)
    draft = _reauthor(session)
    if draft.draft_hash != session.get("draft_hash"):
        raise StaleDraftError(
            "authored draft no longer reproduces this session (capability "
            "descriptors or author contract changed); re-author first")

    chosen: dict[str, str] = {}
    for entry in answers:
        question_id = str(entry["question_id"])
        if question_id in chosen:
            raise ValueError(
                f"duplicate answer for clarification question {question_id!r}")
        chosen[question_id] = str(entry["choice_id"])
    built: list[ClarificationAnswer] = []
    for question in draft.clarification_plan.questions:
        choice_id = chosen.pop(question.question_id, None)
        if choice_id is None or choice_id == "system_default":
            built.append(ClarificationAnswer(
                question.question_id, "system_default", source="default"))
        else:
            built.append(ClarificationAnswer(
                question.question_id, choice_id, source="user"))
    if chosen:
        raise ValueError(
            f"unknown clarification question(s): {sorted(chosen)}")

    submission = ClarificationSubmission(
        version=CLARIFICATION_VERSION,
        draft_hash=draft.draft_hash,
        question_set_hash=draft.clarification_plan.question_set_hash,
        answers=tuple(built),
    )
    applied = apply_clarifications(draft, submission)
    # §6.1: lineage keys on the SHARED evaluation design only — train /
    # meta / provenance differences preserve the fitness baseline.
    lineage = evaluation_lineage_for(applied.world_spec, applied.task_spec)
    admitted = WorldProjectService(project_dir).admit_and_promote(
        world=applied.world_spec, task=applied.task_spec,
        clarifications=applied.clarification_ledger,
        evaluation_lineage=lineage,
        rejected_session_id=f"ui-{session_id}",
    )
    result = {
        "ok": True,
        "session_id": session_id,
        "capability_id": draft.capability_id,
        "result_hash": applied.result_hash,
        "evaluation_lineage": lineage,
        "selection": admitted.promoted.selection.to_dict(),
        "admission": admitted.admission,
        "asset_dir": str(admitted.asset_dir),
        "clarification_answers": len(
            applied.clarification_ledger.get("answers", [])),
    }
    _atomic_write_json(
        _sessions_dir(project_dir) / session_id / "applied.json", result)
    return result


def selection(project_dir: Path) -> dict[str, Any] | None:
    """The authoritative promoted tuple, shaped for display; None when the
    project has no authored world yet."""
    path = Path(project_dir) / "env" / "selection_current.json"
    if not path.is_file():
        return None
    from sculptor.world.project import load_selected_world

    _store, selected, bundle = load_selected_world(path)
    world = bundle["world"]
    task = bundle["task"]
    shared = world.get("shared", {})
    return {
        "selection": selected.to_dict(),
        "world_meta": world.get("meta", {}),
        "task_meta": task.get("meta", {}),
        "shared_summary": {
            "terrain_kind": (shared.get("terrain") or {}).get("kind"),
            "objects": sorted((shared.get("objects") or {}).keys()),
            "zones": sorted((shared.get("zones") or {}).keys()),
            "course_elements": len(
                (shared.get("obstacles") or {}).get("course") or []),
            "robot": (shared.get("robot") or {}).get("capability_id"),
        },
        "goal": (task.get("shared") or {}).get("goal", {}),
        "train_variations": [
            {"id": v.get("id"), "target": v.get("target"),
             "class": v.get("class"), "distribution": v.get("distribution")}
            for v in (world.get("train") or {}).get("variations") or []
        ],
    }


def lineage(project_dir: Path) -> list[dict[str, Any]]:
    """Every immutable promoted selection, oldest first — the CAD-style
    version history of the project's world tuple."""
    import re

    env_dir = Path(project_dir) / "env"
    if not env_dir.is_dir():
        return []
    entries: list[tuple[int, dict[str, Any]]] = []
    for path in env_dir.glob("selection_v*.json"):
        match = re.fullmatch(r"selection_v([0-9]+)\.json", path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append((int(match.group(1)), {
            "selection_version": data.get("selection_version"),
            "created_at": data.get("created_at"),
            "tuple_hash": data.get("tuple_hash"),
            "evaluation_lineage": data.get("evaluation_lineage"),
            "refs": {
                kind: {"version": ref.get("version")}
                for kind, ref in (data.get("refs") or {}).items()
            },
        }))
    return [entry for _, entry in sorted(entries, key=lambda item: item[0])]
