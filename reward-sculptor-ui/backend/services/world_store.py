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

The optional LLM author (enabled by ``RS_WORLD_LLM_AUTHOR=1``) is
non-deterministic, so the re-author trick alone would break. It is made
reproducible by CAPTURING the model's declarative output at author time and
persisting it in the session; ``_reauthor`` then REPLAYS that exact output
through the same ``AuthorModel`` seam, so the recomputed ``draft_hash`` still
matches. A model failure / invalid spec falls back to the deterministic offline
author (nothing captured → offline replay), so a bad model never blocks authoring.
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


def project_capability_id(project_dir: Path) -> str | None:
    """The registered robot capability matching the project's configured
    robot (metadata.json robot_source.library_slug ↔ capability asset_id);
    None when the project has no library robot or no capability maps to it
    (gym robots, uploads)."""
    meta_path = Path(project_dir) / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    robot_source = meta.get("robot_source") if isinstance(meta, dict) else None
    if not isinstance(robot_source, Mapping):
        return None
    slug = str(robot_source.get("library_slug") or "")
    if not slug:
        return None
    from sculptor.world.capabilities import robot_capabilities

    matches = sorted(
        cap_id for cap_id, cap in robot_capabilities().items()
        if getattr(cap, "asset_id", None) == slug)
    return matches[0] if matches else None


def author(
    project_dir: Path,
    prompt: str,
    *,
    robot_capability_id: str | None = None,
    kg_grounding: bool = True,
) -> dict[str, Any]:
    """Author a draft + clarification plan; persist the session inputs.

    A blank robot defaults to the PROJECT's configured robot when a
    capability descriptor maps to it — the world the run trains under
    must not silently swap the robot out from under the project. The
    author's prompt-based auto-selection only applies when the project
    robot has no capability mapping."""
    from sculptor.world.author import AuthoringError, author_environment
    from sculptor.world.grounding import (
        gather_grounding,
        grounding_context,
        grounding_ids,
    )

    _prune_stale_sessions(project_dir)
    project_capability = project_capability_id(project_dir)
    if robot_capability_id is None:
        robot_capability_id = project_capability
    items = gather_grounding(prompt) if kg_grounding else ()
    ids = grounding_ids(items)
    context = grounding_context(items)

    # Optional LLM author (RS_WORLD_LLM_AUTHOR): capture its declarative output so
    # apply-time re-author can replay it deterministically; fall back to offline on
    # any failure/invalid spec (the local validators gate the model output).
    authoring_output = None
    model = _llm_author_model()
    if model is not None:
        capture = _CaptureModel(model)
        try:
            draft = author_environment(
                prompt, model=capture, robot_capability_id=robot_capability_id,
                grounding=ids, grounding_context=context)
            authoring_output = capture.captured
        except AuthoringError:
            authoring_output = None
            draft = author_environment(
                prompt, robot_capability_id=robot_capability_id,
                grounding=ids, grounding_context=context)
    else:
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
            # Present only when the LLM author produced the draft; drives the
            # deterministic replay in _reauthor (None → offline re-author).
            "authoring_output": authoring_output,
        },
    })
    _atomic_write_json(session_dir / "draft.json", draft.to_dict())

    return {
        "session_id": session_id,
        "draft_hash": draft.draft_hash,
        "capability_id": draft.capability_id,
        "project_capability_id": project_capability,
        "robot_matches_project": (
            None if project_capability is None
            else draft.capability_id == project_capability),
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


def _llm_author_model():
    """The optional LLM world author, or None. Enabled by RS_WORLD_LLM_AUTHOR
    ∈ {1,true,yes}. Isolated in one function so tests can monkeypatch it with a
    fake model without touching the environment or the API."""
    if os.environ.get("RS_WORLD_LLM_AUTHOR", "").strip().lower() not in (
            "1", "true", "yes", "on"):
        return None
    from sculptor.world.llm_author import LLMWorldAuthor

    return LLMWorldAuthor()


class _CaptureModel:
    """Wrap an AuthorModel, recording its declarative output so apply-time
    re-author can replay it verbatim (the LLM is non-deterministic; the captured
    output makes the draft-hash contract reproducible)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.captured: dict[str, Any] | None = None

    def generate_authoring(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        out = dict(self._inner.generate_authoring(request))
        self.captured = out
        return out


class _ReplayModel:
    """Return a persisted authoring output verbatim — a deterministic AuthorModel
    that reproduces the exact draft the LLM produced at author time."""

    def __init__(self, captured: Mapping[str, Any]) -> None:
        self._captured = dict(captured)

    def generate_authoring(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._captured


def _reauthor(session: Mapping[str, Any]):
    from sculptor.world.author import author_environment

    inputs = session.get("inputs") or {}
    # Replay the captured LLM output when present; else deterministic offline.
    captured = inputs.get("authoring_output")
    model = _ReplayModel(captured) if isinstance(captured, Mapping) else None
    return author_environment(
        str(inputs.get("prompt") or ""),
        model=model,
        robot_capability_id=inputs.get("robot_capability_id") or None,
        grounding=tuple(inputs.get("grounding") or ()),
        grounding_context=tuple(inputs.get("grounding_context") or ()),
    )


def _clarified_bundle(
    project_dir: Path,
    session_id: str,
    answers: Iterable[Mapping[str, str]],
):
    """Re-author the persisted session, verify the draft hash, and apply
    the answer set. Shared by apply (admit + promote) and preview_draft
    (gated dry-run) so the two can never diverge on answer semantics:
    unanswered questions take their disclosed system default with
    ``default`` provenance, explicit answers record ``user``, and unknown
    question ids raise ``ValueError``."""
    from sculptor.world.author import (
        CLARIFICATION_VERSION,
        ClarificationAnswer,
        ClarificationSubmission,
        apply_clarifications,
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
    return draft, apply_clarifications(draft, submission)


def apply(
    project_dir: Path,
    session_id: str,
    answers: Iterable[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Apply clarification answers and atomically admit + promote."""
    from sculptor.world.project import (
        WorldProjectService,
        evaluation_lineage_for,
    )

    draft, applied = _clarified_bundle(project_dir, session_id, answers)
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


def preview_draft(
    project_dir: Path,
    session_id: str,
    answers: Iterable[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Gated dry-run of an authoring session: apply the answers, run the
    FULL admission gate chain, and return the gate report plus the
    compiled scene graph — without promoting anything.

    This is the iterative build loop: adjust answers → preview the real
    compiled scene → promote (via ``apply``) when satisfied. All
    artifacts live under ``worlds/<session>/preview_<hash>/`` — the
    authoritative ``env/`` tuple is untouched, so the evaluation freeze
    and atomic-selection invariants cannot be affected. Results are
    cached per (draft, answers) since the author + compiler are
    deterministic.
    """
    import hashlib

    from sculptor.world.gates import run_admission_gates
    from sculptor.world.project import WorldProjectService

    canonical = json.dumps(
        sorted(
            ({"question_id": str(a["question_id"]),
              "choice_id": str(a["choice_id"])} for a in answers),
            key=lambda a: a["question_id"]),
        sort_keys=True)
    session_dir = _sessions_dir(project_dir) / session_id
    draft, applied = _clarified_bundle(
        project_dir, session_id, json.loads(canonical))
    cache_key = hashlib.sha256(
        (draft.draft_hash + canonical).encode()).hexdigest()[:16]
    preview_dir = session_dir / f"preview_{cache_key}"
    cached = preview_dir / "preview.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))

    service = WorldProjectService(project_dir)
    report, compiled = run_admission_gates(
        applied.world_spec, applied.task_spec,
        materialize_dir=preview_dir / "assets",
        runtime_task_id=service._runtime_task_id())

    scene = None
    if compiled is not None and compiled._model is not None:
        from backend.services.scene_export import scene_from_model

        scene = scene_from_model(compiled._model, applied.world_spec)

    shared = applied.world_spec.get("shared") or {}
    course = (shared.get("obstacles") or {}).get("course") or []
    breakdown: dict[str, int] = {}
    for element in course:
        kind = str(element.get("element") or "?")
        breakdown[kind] = breakdown.get(kind, 0) + 1
    result = {
        "ok": bool(report.ok),
        "session_id": session_id,
        "result_hash": applied.result_hash,
        "admission": report.to_dict(),
        "scene": scene,
        "summary": {
            "robot": (shared.get("robot") or {}).get("capability_id"),
            "terrain_kind": (shared.get("terrain") or {}).get("kind"),
            "course_elements": len(course),
            "course_breakdown": breakdown,
            "objects": sorted((shared.get("objects") or {}).keys()),
            "zones": sorted((shared.get("zones") or {}).keys()),
        },
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cached, result)
    return result


def edit_variations(
    project_dir: Path,
    edits: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Human-scale train-variation edits from the World UI — the same
    §10.2 primitive the diagnoser uses: per-edit validation + rollback,
    meta.version bump, full re-admission, atomic promotion under the
    EXISTING evaluation lineage, and a hard proof the frozen evaluation
    did not move. Returns the sculptor result dict (applied / rejected /
    new selection)."""
    from sculptor.world.project import (
        WorldVariationEdit,
        apply_world_variation_edits,
    )

    built = [
        WorldVariationEdit(
            variation_id=str(edit["variation_id"]),
            new_distribution=dict(edit["distribution"]),
            rationale=str(edit.get("rationale") or "ui-edit"),
        )
        for edit in edits
    ]
    if not built:
        raise ValueError("no edits supplied")
    return apply_world_variation_edits(Path(project_dir), built)


def scene(project_dir: Path) -> dict[str, Any]:
    """Scene graph of the PROMOTED selection's materialized evaluation
    model — the exact compiled scene fitness is scored on, cached per
    selection version."""
    env_dir = Path(project_dir) / "env"
    selection_path = env_dir / "selection_current.json"
    if not selection_path.is_file():
        raise FileNotFoundError("project has no authored world selection")
    from sculptor.world.project import load_selected_world

    _store, selected, bundle = load_selected_world(selection_path)
    cached = env_dir / f"world_scene_v{selected.selection_version}.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))

    manifest = bundle["resolved_eval"]
    mjb_relative = (manifest.get("materialized_assets") or {}).get(
        "evaluation_mjb")
    if not mjb_relative:
        raise FileNotFoundError(
            "selection has no materialized evaluation model")
    import mujoco

    from backend.services.scene_export import scene_from_model

    model = mujoco.MjModel.from_binary_path(str(env_dir / mjb_relative))
    payload = scene_from_model(model, bundle["world"])
    payload["selection_version"] = selected.selection_version
    payload["tuple_hash"] = selected.tuple_hash
    _atomic_write_json(cached, payload)
    return payload


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
    course = (shared.get("obstacles") or {}).get("course") or []
    breakdown: dict[str, int] = {}
    for element in course:
        kind = str(element.get("element") or "?")
        breakdown[kind] = breakdown.get(kind, 0) + 1
    world_robot = (shared.get("robot") or {}).get("capability_id")
    project_capability = project_capability_id(project_dir)
    selection_receipt = selected.to_dict()
    payload = {
        "selection": selection_receipt,
        "world_meta": world.get("meta", {}),
        "task_meta": task.get("meta", {}),
        "shared_summary": {
            "terrain_kind": (shared.get("terrain") or {}).get("kind"),
            "objects": sorted((shared.get("objects") or {}).keys()),
            "zones": sorted((shared.get("zones") or {}).keys()),
            "course_elements": len(course),
            "course_breakdown": breakdown,
            "robot": world_robot,
            "project_capability_id": project_capability,
            "robot_matches_project": (
                None if project_capability is None
                else world_robot == project_capability),
        },
        "goal": (task.get("shared") or {}).get("goal", {}),
        "train_variations": [
            {"id": v.get("id"), "target": v.get("target"),
             "class": v.get("class"), "distribution": v.get("distribution")}
            for v in (world.get("train") or {}).get("variations") or []
        ],
        "clarifications": _clarification_summary(
            bundle.get("clarifications") or {}),
    }
    event_program = _event_program_summary(task, selection_receipt)
    if event_program is not None:
        payload["event_program"] = event_program
    return payload


def _event_program_summary(
    task: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project the admitted linear event program for researcher display.

    The selected TaskSpec remains the authority.  This adapter deliberately
    copies its exact transition predicates and pins them back to the immutable
    task artifact/selection tuple; it does not infer a program from a task
    name, prompt, or goal shape.  Legacy tasks therefore keep the old response
    shape with no ``event_program`` key.
    """
    shared = task.get("shared")
    if not isinstance(shared, Mapping):
        return None
    event_sequence = shared.get("event_sequence")
    if not isinstance(event_sequence, Mapping):
        return None
    phases = event_sequence.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        return None
    if not all(isinstance(phase, Mapping) for phase in phases):
        return None

    route, jump, hold = phases
    route_until = route.get("until")
    jump_until = jump.get("until")
    if not isinstance(route_until, Mapping) \
            or not isinstance(jump_until, Mapping):
        return None
    minimum_hold_s = hold.get("minimum_hold_s")
    task_ref = (selection_receipt.get("refs") or {}).get("task")
    return {
        "id": event_sequence.get("id"),
        "ordered_phase_ids": [phase.get("id") for phase in phases],
        "transition_spec": [
            {
                "from": route.get("id"),
                "to": jump.get("id"),
                "when": dict(route_until),
            },
            {
                "from": jump.get("id"),
                "to": hold.get("id"),
                "when": dict(jump_until),
            },
            {
                "from": hold.get("id"),
                "to": "terminal",
                "when": {
                    "event": "minimum_hold_elapsed",
                    "minimum_hold_s": minimum_hold_s,
                },
            },
        ],
        "minimum_air_time_s": jump_until.get("min_air_time_s"),
        "minimum_height_delta_m": jump_until.get("min_height_delta_m"),
        "support_selectors": jump_until.get("support_contacts"),
        "terminal_hold_duration_s": minimum_hold_s,
        "episode_length_s": (shared.get("termination") or {}).get(
            "episode_length_s"),
        "train_only_phase_sampling": (task.get("train") or {}).get(
            "event_phase_sampling"),
        "evaluation_start_phase": route.get("id"),
        "observation_extension": {
            "term": "authored_event_phase",
            "encoding": "one_hot",
            "width": 3,
        },
        "provenance": {
            "selection_version": selection_receipt.get("selection_version"),
            "selection_tuple_hash": selection_receipt.get("tuple_hash"),
            "task_artifact": task_ref,
        },
    }


def _clarification_summary(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance ledger shaped for display: who decided every
    load-bearing parameter — the user or the disclosed system default."""
    answers = [a for a in (ledger.get("answers") or []) if isinstance(a, dict)]
    by_source: dict[str, int] = {}
    for answer in answers:
        source = str(answer.get("source") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "answer_sources": by_source,
        "answers": [
            {
                "question_id": answer.get("question_id"),
                "parameter_path": answer.get("parameter_path"),
                "choice_id": answer.get("choice_id"),
                "source": answer.get("source"),
                "value": answer.get("value"),
            }
            for answer in answers
        ],
    }


def validate(project_dir: Path) -> dict[str, Any]:
    """Integrity check of the authoritative tuple, UI-triggerable.

    Re-verifies the selection's tuple hash, every artifact's recorded
    sha256 against the bytes on disk (tamper evidence — the artifacts
    are immutable by contract), and re-runs WorldSpec/TaskSpec schema
    validation on the loaded bundle."""
    from sculptor.world.artifacts import file_sha256
    from sculptor.world.project import load_selected_world
    from sculptor.world.task_spec import validate_task_spec
    from sculptor.world.world_spec import validate_world_spec

    path = Path(project_dir) / "env" / "selection_current.json"
    if not path.is_file():
        raise FileNotFoundError("project has no authored world selection")
    try:
        store, selected, bundle = load_selected_world(path)
    except Exception as exc:  # noqa: BLE001 — corruption is the finding
        return {"ok": False, "selection_version": None, "tuple_hash": None,
                "errors": [f"{type(exc).__name__}: {exc}"]}
    errors: list[str] = []
    for kind, ref in sorted(selected.refs.items()):
        try:
            actual = file_sha256(store.resolve_ref(ref))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kind}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if actual != ref.sha256:
            errors.append(
                f"{kind}: artifact bytes do not match the recorded hash "
                "(tampered or corrupted)")
    errors.extend(
        f"WorldSpec: {e}" for e in validate_world_spec(bundle["world"]))
    errors.extend(
        f"TaskSpec: {e}"
        for e in validate_task_spec(bundle["task"], world=bundle["world"]))
    shared = bundle["world"].get("shared") or {}
    world_robot = (shared.get("robot") or {}).get("capability_id")
    project_robot = project_capability_id(project_dir)
    return {
        "ok": not errors,
        "selection_version": selected.selection_version,
        "tuple_hash": selected.tuple_hash,
        "errors": errors,
        "robot_matches_project": (
            None if project_robot is None else world_robot == project_robot
        ),
        "world_robot": world_robot,
        "project_robot": project_robot,
    }


def training_preflight(project_dir: Path) -> dict[str, Any] | None:
    """Re-attest the promoted world and its robot before training.

    The authored world is an executable training input, so integrity alone is
    insufficient: its robot must also be the robot configured by the project.
    Keep this as the single backend authority used both when the HTTP request
    is admitted and again by the queued worker immediately before spawn.

    ``None`` means the project has no promoted authored world and therefore
    retains the existing built-in-scene behavior.
    """
    selection_path = Path(project_dir) / "env" / "selection_current.json"
    if not selection_path.is_file():
        return None

    # Derive integrity, tuple identity, and robot compatibility from the same
    # loaded bundle. One snapshot cannot mix hashes from one promotion with
    # robot metadata from a concurrent promotion.
    return dict(validate(project_dir))


def curriculum(project_dir: Path) -> dict[str, Any]:
    """Terrain-curriculum progression of the most recent run — one entry
    per iteration that exported world_curriculum_stats.json (mean level
    rising across iterations = the policy earning promotion)."""
    import re as _re

    runs_root = Path(project_dir) / "runs"
    if not runs_root.is_dir():
        return {"run": None, "iterations": []}
    stats_files = list(
        runs_root.glob("**/iter_*/world_curriculum_stats.json"))
    if not stats_files:
        return {"run": None, "iterations": []}

    def _run_dir(stats_path: Path) -> Path:
        return stats_path.parent.parent

    latest_run = max(
        {_run_dir(p) for p in stats_files},
        key=lambda d: d.stat().st_mtime)
    iterations: list[dict[str, Any]] = []
    for stats_path in stats_files:
        if _run_dir(stats_path) != latest_run:
            continue
        match = _re.fullmatch(r"iter_([0-9]+)", stats_path.parent.name)
        if not match:
            continue
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        iterations.append({
            "iter": int(match.group(1)),
            "mean_level": data.get("mean_level"),
            "max_level": data.get("max_level"),
            "num_envs": data.get("num_envs"),
            "histogram": data.get("histogram"),
        })
    iterations.sort(key=lambda entry: entry["iter"])
    return {"run": latest_run.name, "iterations": iterations}


def preview(
    project_dir: Path,
    *,
    angle: str = "iso",
    regenerate: bool = False,
) -> Path:
    """Render (and cache) the current selection's materialized evaluation
    scene. The MJB is the exact compiled model evaluation replays —
    what you see is what fitness is scored on. Cached per selection
    version + angle; a new promotion naturally gets a fresh render."""
    from backend.services.preview_renderer import render_model_preview
    from sculptor.world.project import load_selected_world

    env_dir = Path(project_dir) / "env"
    selection_path = env_dir / "selection_current.json"
    if not selection_path.is_file():
        raise FileNotFoundError("project has no authored world selection")
    _store, selected, bundle = load_selected_world(selection_path)
    manifest = bundle["resolved_eval"]
    mjb_relative = (manifest.get("materialized_assets") or {}).get(
        "evaluation_mjb")
    if not mjb_relative:
        raise FileNotFoundError(
            "selection has no materialized evaluation model")
    out = env_dir / f"world_preview_v{selected.selection_version}_{angle}.png"
    if out.is_file() and not regenerate:
        return out
    return render_model_preview(env_dir / mjb_relative, out, angle)


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
            # The Act-III proof surfaced in the UI: identical across every
            # entry unless a SHARED design field changed.
            "eval_model_hash": _eval_model_hash(
                env_dir, (data.get("refs") or {}).get("resolved_eval")),
        }))
    return [entry for _, entry in sorted(entries, key=lambda item: item[0])]


def _eval_model_hash(
    env_dir: Path, eval_ref: Mapping[str, Any] | None,
) -> str | None:
    """compiled_model_hash from a selection's resolved-eval manifest;
    fail-soft None (lineage listing must never break on one bad file)."""
    if not eval_ref:
        return None
    relative = str(eval_ref.get("path") or "")
    if not relative:
        return None
    # ArtifactRef paths may be recorded relative to the env dir or the
    # project dir depending on the writing base — accept either.
    for candidate in (env_dir / relative, env_dir.parent / relative):
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = manifest.get("compiled_model_hash")
        return str(value) if value else None
    return None
