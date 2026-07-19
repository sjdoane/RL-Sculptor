"""Project-level authoring transaction and immutable tuple promotion."""

from __future__ import annotations

import copy
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sculptor.env_spec import (
    read_current_env_spec,
    validate_env_spec,
    write_env_spec_version,
)
from sculptor.world.artifacts import (
    ArtifactRef,
    ArtifactSelection,
    WorldArtifactStore,
)
from sculptor.world.channels import ChannelCatalog
from sculptor.world.task_spec import validate_task_spec
from sculptor.world.world_spec import validate_world_spec


class WorldPromotionError(ValueError):
    """An authored bundle is incomplete or failed a promotion invariant."""


@dataclass(frozen=True)
class WorldVariationEdit:
    """One diagnoser-scale edit: retarget a registered ``train.variations``
    entry (by stable ID — env-authoring §10.2) to a new distribution."""

    variation_id: str
    new_distribution: Mapping[str, Any]
    rationale: str = ""


#: ResolvedEvaluation fields that define what evaluation *is*. A train-only
#: variation edit re-admits the tuple, and every one of these must come out
#: identical — world/task/catalog hashes legitimately change, evaluation
#: must not (§6.1: only a shared-field change may move the baseline).
_EVAL_INVARIANT_KEYS = (
    "eval_seed", "terrain", "course", "objects", "zones", "task_shared",
    "robot_asset_hash", "compiled_model_hash",
)


def _asset_signature(materialized: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """(basename, sha256) pairs — asset dirs are versioned, bytes must not be."""
    materialized = materialized or {}
    signature = sorted(
        (Path(str(record.get("path", ""))).name, str(record.get("sha256", "")))
        for record in materialized.get("files", ())
    )
    signature.append(
        ("evaluation_mjb", str(materialized.get("evaluation_mjb_sha256", ""))))
    return signature


def _eval_invariance_errors(
    old: Mapping[str, Any], new: Mapping[str, Any],
) -> list[str]:
    errors = [
        f"evaluation field {key!r} changed"
        for key in _EVAL_INVARIANT_KEYS if old.get(key) != new.get(key)
    ]
    if _asset_signature(old.get("materialized_assets")) != _asset_signature(
            new.get("materialized_assets")):
        errors.append("materialized evaluation asset bytes changed")
    return errors


def _bump_meta_version(meta: dict[str, Any]) -> None:
    old = str(meta.get("version") or "v1")
    match = re.fullmatch(r"v([0-9]+)", old)
    meta["parent"] = old
    meta["version"] = f"v{int(match.group(1)) + 1}" if match else "v2"


@dataclass(frozen=True)
class PromotedWorld:
    selection: ArtifactSelection
    world_ref: ArtifactRef
    task_ref: ArtifactRef
    resolved_eval_ref: ArtifactRef
    channel_catalog_ref: ArtifactRef
    clarifications_ref: ArtifactRef


@dataclass(frozen=True)
class AdmittedWorld:
    """Result of the gate/materialize/atomic-promotion transaction."""

    promoted: PromotedWorld
    admission: Mapping[str, Any]
    asset_dir: Path


def _reward_version(project_dir: Path) -> Path:
    rewards = project_dir / "rewards"
    current = rewards / "current.py"
    if current.is_file():
        try:
            text = current.read_text(encoding="utf-8")
            match = re.search(r"/\s*(['\"])(v[0-9]+\.py)\1", text)
            if match:
                target = rewards / match.group(2)
                if target.is_file():
                    return target.resolve()
        except OSError:
            pass
    candidates: list[tuple[int, Path]] = []
    for path in rewards.glob("v*.py"):
        match = re.fullmatch(r"v([0-9]+)\.py", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise WorldPromotionError(
            f"no immutable reward v<N>.py in {rewards}; run sculpt init")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def _env_spec_version(project_dir: Path) -> Path:
    env_dir = project_dir / "env"
    spec = read_current_env_spec(env_dir)
    if spec is None:
        spec = {
            "env_spec_version": 1,
            "meta": {
                "version": "pending",
                "parent": None,
                "source": "default",
                "rationale": (
                    "Neutral EnvSpec created for atomic world selection; "
                    "WorldSpec owns geometry and task semantics."),
            },
            "shared": {},
            "train": {},
        }
        errors = validate_env_spec(spec)
        if errors:
            raise WorldPromotionError(
                "internal neutral EnvSpec invalid: " + "; ".join(errors))
        return write_env_spec_version(env_dir, spec).resolve()
    version = str((spec.get("meta") or {}).get("version", ""))
    if not re.fullmatch(r"v[0-9]+", version):
        raise WorldPromotionError(
            f"active EnvSpec has invalid immutable version {version!r}")
    path = env_dir / f"{version}.json"
    if not path.is_file():
        raise WorldPromotionError(
            f"active EnvSpec version file is missing: {path}")
    return path.resolve()


class WorldProjectService:
    """Validate, persist, and atomically promote an authored world bundle."""

    def __init__(self, project_dir: Path | str):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.store = WorldArtifactStore(self.project_dir)

    def current_bundle(self) -> dict[str, Any] | None:
        return self.store.current_bundle()

    def _runtime_task_id(self) -> str | None:
        config = self.project_dir / "config.toml"
        if not config.is_file():
            return None
        try:
            payload = tomllib.loads(config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise WorldPromotionError(
                f"project config cannot bind authored world: {exc}") from exc
        task_id = ((payload.get("adapter") or {}).get("config") or {}).get(
            "task_id")
        if task_id is None:
            return None
        if not isinstance(task_id, str) or not task_id.strip():
            raise WorldPromotionError(
                "adapter.config.task_id must be a non-empty string")
        return task_id.strip()

    def _next_asset_dir(self) -> Path:
        """Return a never-reused evaluation asset directory.

        Rejected or interrupted admission attempts can leave useful forensic
        assets without a resolved-eval JSON.  Account for both forms so a
        later attempt never overwrites them.
        """
        best = 0
        for path in self.store.env_dir.iterdir():
            match = re.fullmatch(
                r"resolved_eval(?:_assets)?_v([1-9][0-9]*)(?:\.json)?",
                path.name,
            )
            if match:
                best = max(best, int(match.group(1)))
        return self.store.env_dir / f"resolved_eval_assets_v{best + 1}"

    def admit_and_promote(
        self, *, world: Mapping[str, Any], task: Mapping[str, Any],
        clarifications: Mapping[str, Any], evaluation_lineage: str,
        rejected_session_id: str | None = None,
        require_eval_invariance_with: Mapping[str, Any] | None = None,
    ) -> AdmittedWorld:
        """Run all CPU gates, freeze eval assets, then commit one tuple.

        A failed gate cannot alter the authoritative selection.  Its complete
        report and declarative inputs are retained under ``env/rejected``.

        ``require_eval_invariance_with``: a prior ResolvedEvaluation dict.
        When given, the freshly compiled evaluation must be identical in
        every evaluation-defining field and materialized asset byte —
        the guard that makes train-only edits provably unable to move the
        evaluation baseline (§6.1). Drift rejects the whole admission
        before anything is promoted.
        """
        from sculptor.world.gates import run_admission_gates

        with self.store.locked():
            asset_dir = self._next_asset_dir()
            report, compiled = run_admission_gates(
                world, task, materialize_dir=asset_dir,
                runtime_task_id=self._runtime_task_id())
            drift: list[str] = []
            if (report.ok and compiled is not None
                    and require_eval_invariance_with is not None):
                drift = _eval_invariance_errors(
                    dict(require_eval_invariance_with),
                    compiled.resolved_eval.to_dict())
            if not report.ok or compiled is None or drift:
                session_id = rejected_session_id or (
                    "admission-" + re.sub(
                        r"[^a-zA-Z0-9_-]", "-", evaluation_lineage)[:48])
                self.store.write_rejected_draft(session_id, {
                    "world_spec": dict(world), "task_spec": dict(task),
                    "clarifications": dict(clarifications),
                    "admission": report.to_dict(),
                    "evaluation_lineage": evaluation_lineage,
                    "eval_invariance_violations": drift,
                })
                if drift:
                    raise WorldPromotionError(
                        "train-only edit moved evaluation: " + "; ".join(drift))
                summary = "; ".join(
                    f"{item.gate}/{item.code}: {item.message}"
                    for item in report.violations)
                raise WorldPromotionError(
                    "authored environment failed admission"
                    + (f": {summary}" if summary else ""))
            promoted = self.promote(
                world=world, task=task,
                resolved_eval=compiled.resolved_eval.to_dict(),
                channel_catalog=compiled.channel_catalog,
                clarifications=clarifications,
                evaluation_lineage=evaluation_lineage,
            )
            return AdmittedWorld(
                promoted=promoted, admission=report.to_dict(),
                asset_dir=asset_dir)

    def promote(
        self, *, world: Mapping[str, Any], task: Mapping[str, Any],
        resolved_eval: Mapping[str, Any],
        channel_catalog: ChannelCatalog | Mapping[str, Any],
        clarifications: Mapping[str, Any], evaluation_lineage: str,
    ) -> PromotedWorld:
        from sculptor.world.compiler import (
            ResolvedEvaluation,
            WorldCompileError,
            verify_resolved_evaluation,
        )

        with self.store.locked():
            world_dict = dict(world)
            task_dict = dict(task)
            errors = validate_world_spec(world_dict)
            errors.extend(validate_task_spec(task_dict, world=world_dict))
            if errors:
                raise WorldPromotionError(
                    "authored bundle failed schema validation:\n- "
                    + "\n- ".join(errors))

            if isinstance(channel_catalog, ChannelCatalog):
                catalog = channel_catalog
                catalog_dict = channel_catalog.to_dict()
            else:
                catalog_dict = dict(channel_catalog)
                catalog = ChannelCatalog.from_dict(catalog_dict)

            resolved_dict = dict(resolved_eval)
            try:
                manifest = ResolvedEvaluation.from_dict(resolved_dict)
                verify_resolved_evaluation(
                    world_dict, task_dict, catalog, manifest,
                    asset_base=self.store.env_dir)
            except (WorldCompileError, TypeError, ValueError, OSError) as exc:
                raise WorldPromotionError(
                    f"resolved evaluation verification failed: {exc}") from exc

            world_ref = self.store.write_json("world", world_dict)
            task_ref = self.store.write_json("task", task_dict)
            eval_ref = self.store.write_json("resolved_eval", resolved_dict)
            catalog_ref = self.store.write_json(
                "channel_catalog", catalog_dict)
            clarification_ref = self.store.write_json(
                "clarifications", dict(clarifications))

            reward_path = _reward_version(self.project_dir)
            env_spec_path = _env_spec_version(self.project_dir)
            refs: dict[str, ArtifactRef | Path] = {
                "reward": ArtifactRef.from_path(
                    "reward", reward_path.stem, reward_path,
                    base=self.project_dir),
                "env_spec": ArtifactRef.from_path(
                    "env_spec", env_spec_path.stem, env_spec_path,
                    base=self.project_dir),
                "world": world_ref,
                "task": task_ref,
                "resolved_eval": eval_ref,
                "channel_catalog": catalog_ref,
                "clarifications": clarification_ref,
            }
            selection = self.store.promote(
                refs, evaluation_lineage=evaluation_lineage)
            return PromotedWorld(
                selection=selection, world_ref=world_ref, task_ref=task_ref,
                resolved_eval_ref=eval_ref,
                channel_catalog_ref=catalog_ref,
                clarifications_ref=clarification_ref)


def apply_world_variation_edits(
    project_dir: Path | str,
    edits: Iterable[WorldVariationEdit],
    *,
    service: WorldProjectService | None = None,
) -> dict[str, Any]:
    """Apply diagnoser-scale edits to registered ``train.variations`` by
    stable ID (env-authoring §10.2) and re-promote one atomic tuple.

    Contract, mirroring the legacy ``apply_env_edits``:
      - each edit resolves its variation by ID, is rejected on unknown ID
        or no-op, and is individually rolled back when the edited world
        fails schema/task validation — later edits still get their try;
      - a net change bumps ``meta.version`` (parent-linked), re-runs the
        FULL admission gate chain, and promotes with the EXISTING
        evaluation lineage — train-only edits never reset the fitness
        baseline;
      - the re-compiled evaluation is hard-verified identical to the
        current one (``require_eval_invariance_with``): an edit that
        somehow moves evaluation rejects the whole promotion.

    Returns ``{"applied": [...], "rejected": [...], "selection": dict|None,
    "world_version": str|None}``; ``selection`` is None when nothing
    applied (the authoritative selection is untouched).
    """
    service = service or WorldProjectService(project_dir)
    # Hold the store lock across read-edit-admit so a concurrent promotion
    # (e.g. UI re-authoring during a run) cannot land between the bundle
    # read and this promotion and be silently clobbered by a train edit
    # applied to its stale parent. The store's FileLock is reentrant, so
    # admit_and_promote's own locked() below nests safely.
    with service.store.locked():
        return _apply_world_variation_edits_locked(service, edits)


def _apply_world_variation_edits_locked(
    service: WorldProjectService,
    edits: Iterable[WorldVariationEdit],
) -> dict[str, Any]:
    bundle = service.current_bundle()
    if not bundle:
        raise WorldPromotionError(
            "no authored world selection exists; run sculpt world author")
    world = copy.deepcopy(bundle["world"])
    task = bundle["task"]
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for edit in edits:
        variation_id = str(edit.variation_id)
        new_distribution = copy.deepcopy(dict(edit.new_distribution))
        candidate = copy.deepcopy(world)
        variations = candidate.get("train", {}).get("variations", [])
        target = next(
            (entry for entry in variations
             if str(entry.get("id")) == variation_id), None)
        if target is None:
            known = sorted(str(entry.get("id")) for entry in variations)
            rejected.append({
                "variation_id": variation_id,
                "reason": f"unknown variation id (registered: {known})"})
            continue
        old_distribution = copy.deepcopy(target.get("distribution"))
        if new_distribution == old_distribution:
            rejected.append({
                "variation_id": variation_id,
                "reason": "no-op: distribution unchanged"})
            continue
        target["distribution"] = new_distribution
        errors = validate_world_spec(candidate)
        errors.extend(validate_task_spec(dict(task), world=candidate))
        if errors:
            rejected.append({
                "variation_id": variation_id,
                "reason": "; ".join(errors)})
            continue
        world = candidate
        applied.append({
            "variation_id": variation_id,
            "old_distribution": old_distribution,
            "new_distribution": new_distribution,
            "rationale": str(edit.rationale or ""),
        })

    if not applied:
        return {"applied": [], "rejected": rejected,
                "selection": None, "world_version": None}

    _bump_meta_version(world.setdefault("meta", {}))
    selection_info = bundle.get("selection") or {}
    lineage = str(selection_info.get("evaluation_lineage") or "")
    admitted = service.admit_and_promote(
        world=world, task=task,
        clarifications=bundle.get("clarifications") or {},
        evaluation_lineage=lineage,
        rejected_session_id="world-variation-edit",
        require_eval_invariance_with=bundle.get("resolved_eval") or {},
    )
    return {
        "applied": applied, "rejected": rejected,
        "selection": admitted.promoted.selection.to_dict(),
        "world_version": admitted.promoted.world_ref.version,
    }


def load_selected_world(
    selection_path: Path | str,
) -> tuple[WorldArtifactStore, ArtifactSelection, dict[str, Any]]:
    """Resolve a pinned tuple once and return its verified JSON bundle."""
    path = Path(selection_path).expanduser().resolve()
    store = WorldArtifactStore(path.parent.parent)
    selection = store.read_selection(path)
    if selection is None:
        raise FileNotFoundError(path)
    bundle = {
        key: store.load_json_ref(selection.refs[key])
        for key in (
            "world", "task", "resolved_eval", "channel_catalog",
            "clarifications", "env_spec",
        )
    }
    bundle["reward_path"] = str(store.resolve_ref(selection.refs["reward"]))
    bundle["selection"] = selection.to_dict()
    return store, selection, bundle
