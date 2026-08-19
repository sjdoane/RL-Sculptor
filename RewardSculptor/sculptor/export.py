"""Policy export — self-contained deployment bundles for sim-to-real.

A bundle is a single zip that carries everything needed to reproduce or
deploy one trained iteration:

    policy_<project>_iter<N>.zip
    ├── manifest.json        # schema, provenance, dims, hashes, deployment
    ├── checkpoint.pt|zip    # the raw trained checkpoint, byte-exact
    ├── policy.onnx          # actor network, ONNX (best-effort)
    ├── policy_ts.pt         # actor network, TorchScript (best-effort)
    ├── inference.py         # runnable hardware skeleton (obs order + action
    │                        #   formula + rate baked in; 2 robot-SDK seams)
    ├── reward/
    │   ├── reward_spec.json # REWARD_SPEC snapshot the iter trained under
    │   └── v<n>.py          # exact reward source (when resolvable)
    ├── env_spec.json        # env spec the iter trained under (see manifest
    │                        #   env_spec_source for how exact the match is)
    ├── config.toml          # project config snapshot
    ├── metrics.json         # the iter's training/eval metrics
    ├── run_context.json     # package versions / SHAs / seeds (if recorded)
    └── DEPLOY.md            # loading recipes + sim→real hardware contract

The manifest ``deployment`` block is the sim→real interface contract the raw
network cannot carry: the joint ORDER the action/obs vectors index (from the
robot manifest), the action→joint-target formula + per-joint scale + default
pose, the control rate (sim dt × decimation), and the ordered observation
layout — all read best-effort from the mjlab task cfg, degrading to the joint
order + a flag when mjlab is not importable.

Checkpoint formats understood:

* mjlab / rsl_rl ``checkpoint.pt`` — ``{actor_state_dict, critic_state_dict,
  optimizer_state_dict, iter, infos}`` where the actor is an MLP stored
  under ``mlp.<i>.{weight,bias}`` plus ``distribution.std_param``. The
  actor architecture is reconstructed *from the state dict itself* (layer
  shapes are ground truth); the activation comes from the mjlab task cfg
  when importable, else defaults to ``elu`` and the manifest says so.
* stable-baselines3 ``checkpoint.zip`` — bundled byte-exact; ONNX /
  TorchScript export is attempted via the documented SB3 recipe when
  stable_baselines3 is importable.

Everything network-related is best-effort by design: the raw checkpoint +
DEPLOY.md recipe are always present, so a bundle is never useless even
when torch/onnx/sb3 are missing from the environment doing the export.

This deployment ZIP is intentionally not a portable upload.  Use
``export_starting_skill_bundle`` (CLI: ``sculptor export --portable``) for the
closed, data-only ``.rskill`` format accepted at the hostile-upload boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

EXPORT_SCHEMA_VERSION = 1
DEPLOYMENT_BUNDLE_KIND = "reward-sculptor-deployment-bundle"
PORTABLE_SKILL_SCHEMA_VERSION = 2
PORTABLE_SKILL_BUNDLE_KIND = "reward-sculptor-starting-skill"

_PORTABLE_ROBOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REFERENCE_ARCHIVE_MAX_BYTES = 512 * 1024**2
_REFERENCE_EXPANDED_MAX_BYTES = 1024**3
_REFERENCE_MEMBER_MAX = 128

_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")
_MLP_KEY_RE = re.compile(r"^mlp\.(\d+)\.(weight|bias)$")


class ExportError(RuntimeError):
    """Raised when a bundle cannot be built at all (no checkpoint etc.)."""


@dataclass
class ExportResult:
    bundle_path: Path
    manifest: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ── discovery ──────────────────────────────────────────────────────────────

def list_exportable_iters(runs_root: Path | str) -> list[dict[str, Any]]:
    """Iterations under ``runs_root`` that have a trained checkpoint on disk.

    Returns one dict per iteration, sorted by index:
    ``{iter_index, checkpoint, checkpoint_bytes, primary_metric, fitness,
    reward_version}`` — the scalar fields are None when the sidecar files
    are missing/unreadable. Disk is the source of truth (JobManager runs
    are in-memory and vanish on backend restart; these artifacts don't).
    """
    root = Path(runs_root)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        m = _ITER_DIR_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        ckpt = _find_checkpoint(d)
        if ckpt is None:
            continue
        metrics = _load_json(d / "metrics.json") or {}
        spec = _load_json(d / "reward_spec.json") or {}
        behavior = _load_json(d / "rollout" / "behavior.json") or {}
        fitness_doc = _load_json(d / "fitness.json") or {}
        primary = None
        mm = metrics.get("metrics")
        if isinstance(mm, dict):
            v = mm.get("mean_return")
            if isinstance(v, (int, float)):
                primary = float(v)
        # Current objective-fitness runs persist the firewall score beside
        # the iteration in fitness.json.  behavior.json is rollout telemetry
        # and only older runs stored a fitness scalar there.  Prefer the
        # authoritative current file while retaining the legacy fallback.
        fitness = fitness_doc.get("fitness")
        if not isinstance(fitness, (int, float)):
            fitness = behavior.get("fitness")
        out.append({
            "iter_index": int(m.group(1)),
            "checkpoint": ckpt.name,
            "checkpoint_bytes": ckpt.stat().st_size,
            "primary_metric": primary,
            "fitness": fitness if isinstance(fitness, (int, float)) else None,
            "reward_version": spec.get("version"),
        })
    out.sort(key=lambda r: r["iter_index"])
    return out


def _find_checkpoint(iter_dir: Path) -> Optional[Path]:
    for name in ("checkpoint.pt", "checkpoint.zip"):
        p = iter_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ── bundle builder ─────────────────────────────────────────────────────────

def export_policy_bundle(
    project_dir: Path | str,
    *,
    iter_index: Optional[int] = None,
    runs_root: Path | str | None = None,
    out_path: Path | str | None = None,
) -> ExportResult:
    """Build a self-contained deployment bundle for one trained iteration.

    ``iter_index=None`` picks the latest iteration that has a checkpoint.
    ``runs_root`` defaults to ``<project>/runs`` (mission stages pass their
    own ``.missions/<m>/stages/<s>/runs``). ``out_path`` defaults to
    ``<project>/exports/policy_<project>_iter<N>.zip``.
    """
    project = Path(project_dir).resolve()
    root = Path(runs_root).resolve() if runs_root else project / "runs"

    avail = list_exportable_iters(root)
    if not avail:
        raise ExportError(f"no exportable iterations under {root}")
    if iter_index is None:
        iter_index = avail[-1]["iter_index"]
    if not any(r["iter_index"] == iter_index for r in avail):
        raise ExportError(
            f"iter {iter_index} has no checkpoint under {root} "
            f"(available: {[r['iter_index'] for r in avail]})")

    iter_dir = root / f"iter_{iter_index}"
    ckpt = _find_checkpoint(iter_dir)
    assert ckpt is not None  # guaranteed by list_exportable_iters
    warnings: list[str] = []

    if out_path is None:
        exports_dir = project / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        out = exports_dir / f"policy_{project.name}_iter{iter_index}.zip"
    else:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rs_export_") as td:
        stage = Path(td)

        files: list[tuple[Path, str]] = [(ckpt, ckpt.name)]

        # Reward: spec snapshot + exact source when resolvable.
        reward_spec = _load_json(iter_dir / "reward_spec.json")
        reward_version = (reward_spec or {}).get("version")
        if reward_spec is not None:
            files.append((iter_dir / "reward_spec.json",
                          "reward/reward_spec.json"))
        else:
            warnings.append("reward_spec.json missing from the iter dir")
        if isinstance(reward_version, str):
            # reward_spec.json is on-disk data — validate before it becomes
            # a path component (a hostile "../../x" must not read outside
            # the project once this is HTTP-triggered).
            if re.fullmatch(r"v\d+", reward_version):
                cand = project / "rewards" / f"{reward_version}.py"
                if cand.is_file():
                    files.append((cand, f"reward/{cand.name}"))
                else:
                    warnings.append(
                        f"reward source {cand.name} not found under rewards/")
            else:
                warnings.append(
                    f"reward version {reward_version!r} is not v<n> — "
                    "reward source not bundled")

        # Env spec: exact per-iter snapshot when present (written at train
        # time), else the project's current spec, flagged as approximate.
        env_spec_source = None
        iter_env = iter_dir / "env_spec.json"
        cur_env = project / "env" / "current.json"
        if iter_env.is_file():
            files.append((iter_env, "env_spec.json"))
            env_spec_source = "iter_snapshot"
        elif cur_env.is_file():
            files.append((cur_env, "env_spec.json"))
            env_spec_source = "project_current"
            warnings.append(
                "env_spec.json is the project's CURRENT spec — this run "
                "predates per-iter env snapshots, so the exact trained "
                "version is not guaranteed")

        # Project config + iter metrics + reproducibility context.
        for src, arc in (
            (project / "config.toml", "config.toml"),
            (iter_dir / "metrics.json", "metrics.json"),
            (iter_dir / "run_context.json", "run_context.json"),
        ):
            if src.is_file():
                files.append((src, arc))
            elif arc == "config.toml":
                warnings.append("config.toml missing from the project dir")

        # Neural-net exports (best-effort, never fatal).
        net_meta: dict[str, Any] = {}
        if ckpt.suffix == ".pt":
            net_meta = _export_rsl_rl_actor(
                ckpt, project, stage, files, warnings)
            # The actor reconstruction performs the strict architecture and
            # normalizer checks. Only expose a trainable interchange payload
            # after those checks prove the policy surface is understood; a
            # safetensors file is memory-safe, but that alone does not make an
            # unknown architecture semantically loadable.
            if "obs_dim" in net_meta and "action_dim" in net_meta:
                trainable_meta = _export_rsl_rl_safetensors(
                    ckpt, stage, files, warnings,
                )
                if trainable_meta:
                    net_meta["trainable_checkpoint"] = trainable_meta
        elif ckpt.suffix == ".zip":
            net_meta = _export_sb3_actor(ckpt, stage, files, warnings)

        # Sim→real hardware contract (joint order, action scale/offset, control
        # rate, obs layout) — the interface the raw network cannot carry.
        deployment = _deployment_contract(project, net_meta, warnings)
        compatibility_contract = None
        compatibility_contract_digest = None
        if net_meta.get("trainable_checkpoint"):
            try:
                from sculptor.policy_contract import (
                    build_project_policy_contract,
                    contract_fingerprint,
                )

                compatibility_contract = build_project_policy_contract(
                    project, observed_network=net_meta,
                )
                compatibility_contract_digest = contract_fingerprint(
                    compatibility_contract,
                )
            except Exception as exc:  # noqa: BLE001 — export remains usable
                warnings.append(
                    "trainable compatibility contract unavailable "
                    f"({type(exc).__name__}: {exc}); import will be blocked"
                )

        manifest: dict[str, Any] = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "kind": DEPLOYMENT_BUNDLE_KIND,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sculptor_version": _sculptor_version(),
            "project": project.name,
            "iter_index": iter_index,
            "checkpoint": {
                "file": ckpt.name,
                "format": "rsl_rl" if ckpt.suffix == ".pt" else "sb3",
                "sha256": _sha256(ckpt),
            },
            "reward_version": reward_version,
            "env_spec_source": env_spec_source,
            "network": net_meta,
            "deployment": deployment,
            "compatibility_contract": compatibility_contract,
            "compatibility_contract_digest": compatibility_contract_digest,
            "warnings": warnings,
        }

        deploy_md = _render_deploy_md(manifest)
        (stage / "DEPLOY.md").write_text(deploy_md, encoding="utf-8")
        files.append((stage / "DEPLOY.md", "DEPLOY.md"))

        # Runnable inference skeleton (obs order + action formula + rate baked
        # in; only the two robot-SDK seams are left for the operator).
        (stage / "inference.py").write_text(
            _render_inference_py(manifest), encoding="utf-8")
        files.append((stage / "inference.py", "inference.py"))

        # File table with hashes (manifest lists everything but itself).
        manifest["files"] = [
            {"path": arc, "sha256": _sha256(src), "bytes": src.stat().st_size}
            for src, arc in files
        ]

        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8")

        tmp_zip = stage / "bundle.zip"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            for src, arc in files:
                zf.write(src, arc)
        # Atomic-ish move into place (same-filesystem rename when possible).
        shutil.move(str(tmp_zip), str(out))

    return ExportResult(bundle_path=out, manifest=manifest, warnings=warnings)


def _portable_iteration_policy_contract(
    project: Path,
    *,
    runs_root: Path | str | None,
    iter_index: int,
    observed_network: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Attest the policy interface against the iteration-owned world tuple.

    Historical checkpoints must never inherit a newly promoted project world.
    The iteration's artifact tuple is therefore required to match, byte for
    canonical byte, the immutable ``selection_vN`` that it names before that
    selection can define the portable policy contract.
    """
    from sculptor.policy_contract import (
        build_project_policy_contract,
        contract_fingerprint,
    )
    from sculptor.world.artifacts import (
        WorldArtifactStore,
        canonical_json_bytes,
        file_sha256,
    )

    root = Path(runs_root).resolve() if runs_root else project / "runs"
    tuple_path = root / f"iter_{iter_index}" / "artifact_tuple.json"
    if not tuple_path.is_file():
        raise ExportError(
            "portable starting-skill export requires the iteration-owned "
            f"artifact tuple: {tuple_path}"
        )
    try:
        tuple_payload = json.loads(tuple_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportError(
            f"iteration artifact tuple is unreadable: {tuple_path}"
        ) from exc
    if not isinstance(tuple_payload, dict):
        raise ExportError("iteration artifact tuple must be a JSON object")
    selection_version = tuple_payload.get("selection_version")
    if (
        not isinstance(selection_version, int)
        or isinstance(selection_version, bool)
        or selection_version <= 0
    ):
        raise ExportError(
            "iteration artifact tuple has no valid selection_version"
        )

    selection_path = project / "env" / f"selection_v{selection_version}.json"
    try:
        selection = WorldArtifactStore(project).read_selection(selection_path)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ExportError(
            "portable starting-skill source selection failed verification: "
            f"{exc}"
        ) from exc
    if selection is None:
        raise ExportError(
            f"immutable source selection is absent: {selection_path}"
        )
    if canonical_json_bytes(tuple_payload) != canonical_json_bytes(
        selection.to_dict()
    ):
        raise ExportError(
            "iteration artifact tuple does not exactly match its immutable "
            "source selection"
        )
    try:
        contract = build_project_policy_contract(
            project,
            observed_network=observed_network,
            world_selection_path=selection_path,
        )
        contract_digest = contract_fingerprint(contract)
    except Exception as exc:  # noqa: BLE001 - normalize researcher boundary
        raise ExportError(
            "iteration-owned policy compatibility contract is unavailable "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    receipt = {
        "schema": 1,
        "iter_index": iter_index,
        "artifact_tuple_sha256": file_sha256(tuple_path),
        "selection_version": selection.selection_version,
        "selection_sha256": file_sha256(selection_path),
        "tuple_hash": selection.tuple_hash,
        "compatibility_contract_digest": contract_digest,
    }
    return contract, contract_digest, receipt


def export_starting_skill_bundle(
    project_dir: Path | str,
    *,
    iter_index: Optional[int] = None,
    runs_root: Path | str | None = None,
    out_path: Path | str | None = None,
    robot_slug: Optional[str] = None,
    legacy_origin_job_log: Path | str | None = None,
    legacy_source_selection: Path | str | None = None,
    legacy_observed_selection: Path | str | None = None,
) -> ExportResult:
    """Build a strict data-only `.rskill` policy-transfer artifact.

    Deployment exports intentionally retain the raw checkpoint, generated
    Python, TorchScript/ONNX, reward source, and environment snapshots.  Those
    files are useful to a trusted operator but are forbidden at an untrusted
    upload boundary.  This exporter reuses the local checkpoint conversion and
    contract checks, then emits only canonical safetensors plus a descriptor-
    complete portable manifest.
    """
    project = Path(project_dir).resolve()
    if out_path is not None and Path(out_path).suffix.lower() != ".rskill":
        raise ExportError("portable starting-skill output must end in .rskill")
    try:
        from sculptor.project_robot import resolve_project_reference_robot

        project_robot = resolve_project_reference_robot(project)
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(
            "portable starting-skill export requires the project's exact "
            f"reference robot namespace: {exc}"
        ) from exc
    if robot_slug and robot_slug != project_robot:
        raise ExportError(
            "portable robot identity conflicts with the project's canonical "
            f"reference robot namespace ({robot_slug!r} != {project_robot!r})"
        )
    resolved_robot_slug = project_robot
    with tempfile.TemporaryDirectory(prefix="rs_portable_export_") as td:
        stage = Path(td)
        deployment_result = export_policy_bundle(
            project,
            iter_index=iter_index,
            runs_root=runs_root,
            out_path=stage / "deployment.zip",
        )
        source_manifest = deployment_result.manifest
        source_network = source_manifest.get("network") or {}
        trainable = source_network.get("trainable_checkpoint")
        if not isinstance(trainable, dict):
            raise ExportError(
                "iteration cannot produce a portable starting skill: "
                "canonical safetensors and an exact compatibility contract "
                "are required"
            )
        contract, contract_digest, source_tuple_receipt = (
            _portable_iteration_policy_contract(
                project,
                runs_root=runs_root,
                iter_index=int(source_manifest["iter_index"]),
                observed_network=source_network,
            )
        )
        weights_name = trainable.get("file")
        roles = trainable.get("policy_roles")
        if (
            weights_name != "policy/weights.safetensors"
            or not isinstance(roles, list)
            or "actor" not in roles
            or any(role not in {"actor", "critic"} for role in roles)
        ):
            raise ExportError(
                "iteration produced an unsupported portable policy payload"
            )
        identity = contract.get("identity") or {}
        adapter_class = identity.get("adapter_class")
        task_id = identity.get("task_id")
        if not isinstance(adapter_class, str) or not adapter_class:
            raise ExportError("portable policy contract has no adapter identity")
        if not isinstance(task_id, str) or not task_id:
            raise ExportError("portable policy contract has no task identity")

        try:
            with zipfile.ZipFile(deployment_result.bundle_path, "r") as source:
                weights = source.read(weights_name)
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            raise ExportError(
                "deployment conversion did not produce readable safetensors"
            ) from exc
        weights_sha = hashlib.sha256(weights).hexdigest()
        from sculptor.compatibility_provenance import (
            LEGACY_CONFIG_MEMBER,
            LEGACY_LOG_MEMBER,
            LEGACY_OBSERVED_SELECTION_MEMBER,
            LEGACY_SOURCE_SELECTION_MEMBER,
            ORIGIN_CONTRACT_MEMBER,
            build_legacy_reconstructed_provenance,
            build_origin_persisted_provenance,
            canonical_json_bytes,
            evidence_member_paths,
            provenance_fingerprint,
        )

        legacy_paths = (
            legacy_origin_job_log,
            legacy_source_selection,
            legacy_observed_selection,
        )
        if any(path is not None for path in legacy_paths) and not all(
            path is not None for path in legacy_paths
        ):
            raise ExportError(
                "legacy contract reconstruction requires --legacy-origin-job-log, "
                "--legacy-source-selection, and --legacy-observed-selection together"
            )
        provenance_payloads: dict[str, bytes]
        if all(path is not None for path in legacy_paths):
            try:
                origin_log = Path(legacy_origin_job_log).resolve().read_bytes()
                source_config = (project / "config.toml").read_bytes()
                source_selection = Path(legacy_source_selection).resolve().read_bytes()
                observed_selection = Path(
                    legacy_observed_selection
                ).resolve().read_bytes()
                contract_provenance = build_legacy_reconstructed_provenance(
                    origin_log=origin_log,
                    source_config=source_config,
                    source_selection=source_selection,
                    observed_selection=observed_selection,
                    contract=contract,
                    policy_roles=list(roles),
                    iter_index=int(source_manifest["iter_index"]),
                )
            except (OSError, ValueError) as exc:
                raise ExportError(
                    "legacy compatibility-contract evidence failed verification: "
                    f"{exc}"
                ) from exc
            provenance_payloads = {
                LEGACY_LOG_MEMBER: origin_log,
                LEGACY_CONFIG_MEMBER: source_config,
                LEGACY_SOURCE_SELECTION_MEMBER: source_selection,
                LEGACY_OBSERVED_SELECTION_MEMBER: observed_selection,
            }
        else:
            root = Path(runs_root).resolve() if runs_root else project / "runs"
            origin_contract_path = (
                root
                / f"iter_{int(source_manifest['iter_index'])}"
                / "warm_start_effective_policy_contract.json"
            )
            if not origin_contract_path.is_file():
                raise ExportError(
                    "portable policy export requires the contract sidecar persisted "
                    "when training ran. This historical iteration has no origin "
                    "sidecar; use all three explicit --legacy-* evidence options "
                    "to request a visibly disclosed actor/critic-only reconstruction."
                )
            try:
                origin_contract_bytes = origin_contract_path.read_bytes()
                origin_contract = json.loads(origin_contract_bytes.decode("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ExportError(
                    "origin policy-contract sidecar is unreadable"
                ) from exc
            if (
                not isinstance(origin_contract, dict)
                or canonical_json_bytes(origin_contract) != canonical_json_bytes(contract)
            ):
                raise ExportError(
                    "origin policy-contract sidecar does not match the exported "
                    "compatibility contract"
                )
            contract_provenance = build_origin_persisted_provenance(
                contract_bytes=origin_contract_bytes,
                policy_roles=list(roles),
            )
            provenance_payloads = {ORIGIN_CONTRACT_MEMBER: origin_contract_bytes}
        contract_provenance_digest = provenance_fingerprint(contract_provenance)
        portable_warnings = [
            "Portable transfer excludes the raw checkpoint, optimizer state, "
            "reward/environment source, Python, ONNX, and TorchScript; use the "
            "separate deployment ZIP for trusted deployment workflows."
        ]
        starting_skill: dict[str, Any] = {
            "name": f"{project.name} iter {source_manifest['iter_index']}",
            "weights_file": weights_name,
            "policy_roles": list(roles),
            "adapter_class": adapter_class,
            "task_id": task_id,
        }
        starting_skill["robot_slug"] = str(resolved_robot_slug)
        manifest: dict[str, Any] = {
            "schema_version": PORTABLE_SKILL_SCHEMA_VERSION,
            "kind": PORTABLE_SKILL_BUNDLE_KIND,
            "created_at": source_manifest.get("created_at"),
            "sculptor_version": source_manifest.get("sculptor_version"),
            "project": project.name,
            "iter_index": source_manifest["iter_index"],
            "starting_skill": starting_skill,
            "deployment": {
                "task_id": task_id,
                "robot_slug": str(resolved_robot_slug),
            },
            "checkpoint": {
                "format": (source_manifest.get("checkpoint") or {}).get("format"),
                "sha256": (source_manifest.get("checkpoint") or {}).get("sha256"),
                "included": False,
            },
            "network": {
                key: value
                for key, value in source_network.items()
                if key != "exports"
            },
            "compatibility_contract": contract,
            "compatibility_contract_digest": contract_digest,
            "compatibility_contract_provenance": contract_provenance,
            "compatibility_contract_provenance_digest": (
                contract_provenance_digest
            ),
            "source_artifact_tuple": source_tuple_receipt,
            "warnings": portable_warnings,
            "files": [
                {
                    "path": weights_name,
                    "sha256": weights_sha,
                    "bytes": len(weights),
                },
                *[
                    {
                        "path": member_name,
                        "sha256": hashlib.sha256(
                            provenance_payloads[member_name]
                        ).hexdigest(),
                        "bytes": len(provenance_payloads[member_name]),
                    }
                    for member_name in evidence_member_paths(contract_provenance)
                ],
            ],
        }

        if out_path is None:
            exports_dir = project / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            out = exports_dir / (
                f"skill_{project.name}_iter{source_manifest['iter_index']}.rskill"
            )
        else:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = stage / "portable-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        pending = stage / "portable.rskill"
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, "manifest.json")
            archive.writestr(weights_name, weights)
            for member_name in evidence_member_paths(contract_provenance):
                archive.writestr(member_name, provenance_payloads[member_name])
        shutil.move(str(pending), str(out))
    return ExportResult(
        bundle_path=out,
        manifest=manifest,
        warnings=portable_warnings,
    )


def export_reference_starting_skill_bundle(
    *,
    robot_slug: str,
    clip_id: str,
    out_path: Path | str | None = None,
    name: str | None = None,
    references_root: Path | str | None = None,
) -> ExportResult:
    """Export one exact library trajectory as a data-only ``.rskill``.

    The reference library remains the authority: the caller selects a
    robot-scoped ``(robot_slug, clip_id)`` identity, and this function refuses
    to substitute an index row, a filename, or target-project metadata.  Both
    source files are revalidated and content-attested immediately before the
    archive is written.  The output contains only ``manifest.json``,
    ``motion/clip.npz``, and ``motion/provenance.json``; controller, world,
    policy, Python, pickle, and raw checkpoint bytes are never included.

    A reference-only import is still a candidate.  Target-project Tier-D
    certification/admission happens later at run launch and is intentionally
    not claimed by this transfer artifact.
    """
    from sculptor import reference
    from sculptor.refs import library as refs
    from sculptor.skill_bundle import reference_source_provenance_sha256

    if (
        not isinstance(robot_slug, str)
        or not _PORTABLE_ROBOT_RE.fullmatch(robot_slug)
    ):
        raise ExportError(
            "reference robot must be a safe stable library identifier"
        )
    try:
        refs.validate_clip_id(clip_id)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid reference clip identity: {exc}") from exc

    library_root = Path(
        references_root if references_root is not None else refs.references_root()
    ).expanduser().resolve()
    source_dir = (library_root / robot_slug / clip_id).resolve()
    try:
        source_dir.relative_to(library_root)
    except ValueError as exc:
        raise ExportError(
            "reference identity resolves outside the configured library"
        ) from exc

    clip_path = source_dir / refs.CLIP_FILENAME
    provenance_path = source_dir / refs.PROVENANCE_FILENAME
    if not clip_path.is_file():
        raise ExportError(
            f"reference clip bytes are missing for {robot_slug}/{clip_id}"
        )
    if not provenance_path.is_file():
        raise ExportError(
            f"reference provenance is missing for {robot_slug}/{clip_id}"
        )
    if provenance_path.stat().st_size > 2 * 1024**2:
        raise ExportError("reference provenance exceeds the 2 MiB limit")

    try:
        provenance_bytes = provenance_path.read_bytes()
        provenance = _load_strict_json_object(
            provenance_bytes, label="reference provenance",
        )
    except (OSError, ValueError) as exc:
        raise ExportError(f"invalid reference provenance: {exc}") from exc
    provenance_errors = refs.validate_provenance(provenance)
    if provenance_errors:
        raise ExportError(
            "invalid reference provenance: " + "; ".join(provenance_errors)
        )
    if provenance.get("schema") != refs.PROVENANCE_SCHEMA:
        raise ExportError(
            "reference provenance schema is unsupported "
            f"({provenance.get('schema')!r})"
        )
    if provenance.get("robot") != robot_slug:
        raise ExportError(
            "reference provenance robot does not match the selected library "
            f"identity ({provenance.get('robot')!r} != {robot_slug!r})"
        )
    if provenance.get("clip_id") != clip_id:
        raise ExportError(
            "reference provenance clip_id does not match the selected library "
            f"identity ({provenance.get('clip_id')!r} != {clip_id!r})"
        )

    expected_clip_sha = provenance.get("content_sha256")
    if (
        not isinstance(expected_clip_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_clip_sha)
    ):
        raise ExportError(
            "reference provenance content_sha256 must be lowercase SHA-256"
        )
    _validate_reference_clip_container(clip_path)
    actual_clip_sha = _sha256(clip_path)
    if actual_clip_sha != expected_clip_sha:
        raise ExportError(
            "reference clip digest does not match provenance "
            f"({actual_clip_sha} != {expected_clip_sha})"
        )
    try:
        reference.load_clip(clip_path)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ExportError(f"reference clip is invalid: {exc}") from exc

    if name is None:
        display_name = f"{robot_slug}/{clip_id} reference"
    elif not isinstance(name, str) or not name.strip():
        raise ExportError("starting-skill name must be non-empty")
    else:
        display_name = name.strip()
    if len(display_name) > 160 or any(ord(char) < 32 for char in display_name):
        raise ExportError(
            "starting-skill name must be at most 160 characters with no "
            "control characters"
        )

    if out_path is None:
        exports_dir = library_root / "exports"
        out = exports_dir / f"reference_{robot_slug}_{clip_id}.rskill"
    else:
        out = Path(out_path).expanduser()
    if out.suffix.lower() != ".rskill":
        raise ExportError("reference starting-skill output must end in .rskill")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    clip_size = clip_path.stat().st_size
    source_provenance_sha = reference_source_provenance_sha256(provenance)
    warnings = [
        "This upload registers a reference candidate only. Training remains "
        "blocked until the selected target re-verifies an exact Tier-D "
        "execution contract and evidence chain."
    ]
    manifest: dict[str, Any] = {
        "schema_version": PORTABLE_SKILL_SCHEMA_VERSION,
        "kind": PORTABLE_SKILL_BUNDLE_KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": f"reference:{robot_slug}/{clip_id}",
        "starting_skill": {
            "name": display_name,
            "robot_slug": robot_slug,
        },
        "deployment": {"robot_slug": robot_slug},
        "reference": {
            "robot_slug": robot_slug,
            "clip_id": clip_id,
            "clip_member": "motion/clip.npz",
            "provenance_member": "motion/provenance.json",
            "content_sha256": actual_clip_sha,
            "provenance_sha256": provenance_sha,
            "source_provenance_sha256": source_provenance_sha,
            "tier_at_export": provenance.get("tier", "K"),
        },
        "warnings": warnings,
        "files": [
            {
                "path": "motion/clip.npz",
                "sha256": actual_clip_sha,
                "bytes": clip_size,
            },
            {
                "path": "motion/provenance.json",
                "sha256": provenance_sha,
                "bytes": len(provenance_bytes),
            },
        ],
    }
    manifest_bytes = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    fd, pending_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent,
    )
    os.close(fd)
    pending = Path(pending_name)
    try:
        with zipfile.ZipFile(
            pending, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True,
        ) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            _write_zip_member_verified(
                archive,
                clip_path,
                "motion/clip.npz",
                expected_sha256=actual_clip_sha,
                expected_bytes=clip_size,
            )
            archive.writestr("motion/provenance.json", provenance_bytes)
        with pending.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(pending, out)
    except Exception:
        pending.unlink(missing_ok=True)
        raise

    return ExportResult(bundle_path=out, manifest=manifest, warnings=warnings)


def _export_rsl_rl_safetensors(
    ckpt: Path,
    stage: Path,
    files: list[tuple[Path, str]],
    warnings: list[str],
) -> dict[str, Any]:
    """Export the trainable actor/critic maps without Python pickle.

    The raw checkpoint remains in the deployment bundle for local recovery,
    but an importer never needs to deserialize it: it consumes this
    safetensors member and constructs a fresh server-owned checkpoint.
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        warnings.append("safetensors unavailable — bundle is deployment-only")
        return {}
    try:
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:  # local, server-produced historical checkpoint
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — export remains best-effort
        warnings.append(
            "could not create trainable safetensors export "
            f"({type(exc).__name__}: {exc})"
        )
        return {}
    if not isinstance(payload, dict):
        return {}
    allowed = (
        "actor_state_dict",
        "critic_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
    )
    tensors: dict[str, Any] = {}
    roles: list[str] = []
    for group in allowed:
        state = payload.get(group)
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if isinstance(key, str) and torch.is_tensor(value):
                tensors[f"{group}::{key}"] = value.detach().cpu().contiguous()
        if group == "actor_state_dict" and any(
            key.startswith(f"{group}::") for key in tensors
        ):
            roles.append("actor")
        if group == "critic_state_dict" and any(
            key.startswith(f"{group}::") for key in tensors
        ):
            roles.append("critic")
    # Historical rsl_rl checkpoints may keep EmpiricalNormalization in
    # ``obs_norm_state_dict`` rather than the newer explicit actor group.
    # The strict actor-export gate above has already validated this mapping;
    # preserve its tensor stats under the canonical data-only group name.
    legacy_norm = payload.get("obs_norm_state_dict")
    if isinstance(legacy_norm, dict):
        for key, value in legacy_norm.items():
            if isinstance(key, str) and torch.is_tensor(value):
                tensors[
                    f"actor_obs_normalizer_state_dict::{key}"
                ] = value.detach().cpu().contiguous()
    if "actor" not in roles:
        warnings.append(
            "checkpoint has no tensor-only actor_state_dict; "
            "bundle cannot be imported as a starting skill"
        )
        return {}
    path = stage / "policy_weights.safetensors"
    try:
        save_file(
            tensors,
            str(path),
            metadata={"format": "reward-sculptor-rsl-rl-v1"},
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            "safetensors export failed "
            f"({type(exc).__name__}: {exc})"
        )
        return {}
    arcname = "policy/weights.safetensors"
    files.append((path, arcname))
    return {
        "file": arcname,
        "format": "reward-sculptor-rsl-rl-v1",
        "policy_roles": roles,
    }


# ── rsl_rl (.pt) actor reconstruction ──────────────────────────────────────

def _export_rsl_rl_actor(
    ckpt: Path, project: Path, stage: Path,
    files: list[tuple[Path, str]], warnings: list[str],
) -> dict[str, Any]:
    """Rebuild the actor MLP from the checkpoint state dict and export it.

    Layer sizes come from the state dict (ground truth). The activation
    comes from the mjlab task cfg when resolvable; else assumed ``elu``
    (rsl_rl's default) and flagged in the manifest.
    """
    try:
        import torch
    except ImportError:
        warnings.append("torch not importable — raw checkpoint only")
        return {}

    try:
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 — older ckpts pickle extras in infos
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — corrupt ckpt must not kill export
        warnings.append(f"checkpoint unreadable ({type(e).__name__}: {e}) "
                        "— raw checkpoint only")
        return {}
    if not isinstance(payload, dict):
        warnings.append(
            f"checkpoint is a {type(payload).__name__}, not the rsl_rl dict "
            "format — raw checkpoint only")
        return {}
    actor_sd = payload.get("actor_state_dict")
    if not isinstance(actor_sd, dict):
        warnings.append("no actor_state_dict in checkpoint — raw only "
                        f"(keys: {sorted(payload)})")
        return {}

    # Observation normalization is part of the policy function. When the
    # checkpoint carries rsl_rl EmpiricalNormalization state we BAKE it
    # into the exported graph — y = (x - mean) / (std + eps), eps=1e-2,
    # applied before the MLP exactly as rsl_rl's MLPModel does. Unknown
    # normalizer shapes refuse the nn export (a bare-MLP export taking
    # raw obs where training used normalized ones is silently wrong).
    norm = _extract_obs_normalizer(payload, actor_sd, warnings)
    if norm == "refuse":
        return {"obs_normalization": True}

    layers = _mlp_layers_from_state_dict(actor_sd)
    if not layers:
        warnings.append("actor_state_dict has no mlp.<i> Linear layers — "
                        "unsupported architecture, raw checkpoint only")
        return {}
    reason = _mlp_structure_problem(actor_sd, layers)
    if reason:
        warnings.append(
            f"actor mlp structure not reconstructable ({reason}) — "
            "raw checkpoint only")
        return {}

    obs_dim = layers[0][1]
    act_dim = layers[-1][2]
    hidden = [out for (_, _, out) in layers[:-1]]

    activation, activation_assumed = _resolve_activation(project, warnings)
    if activation not in _ACTIVATIONS:
        # Never guess: an unmapped activation (lrelu/silu/mish/crelu...)
        # built as ELU would ship a silently wrong policy that even the
        # trace parity check can't catch (it compares against the same
        # wrongly-built module).
        warnings.append(
            f"activation {activation!r} from the task cfg has no known "
            "torch equivalent here — raw checkpoint only")
        return {}

    model = _build_mlp(layers, activation, norm=norm)
    # Load only the mlp.* weights; distribution params are not part of the
    # deterministic deployment path (we export the mean action).
    mlp_sd = {k: v for k, v in actor_sd.items() if _MLP_KEY_RE.match(k)}
    try:
        model.mlp.load_state_dict(
            {k.removeprefix("mlp."): v for k, v in mlp_sd.items()},
            strict=True)
    except Exception as e:  # noqa: BLE001 — degrade, never die
        warnings.append(
            f"actor weights did not load into the reconstructed MLP "
            f"({type(e).__name__}: {e}) — raw checkpoint only")
        return {}
    model.eval()

    if norm is not None and not _verify_normalizer_parity(
            model, norm, obs_dim, warnings):
        return {"obs_normalization": True}

    meta: dict[str, Any] = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "hidden_dims": hidden,
        "activation": activation,
        "activation_assumed": activation_assumed,
        "output": "mean_action",
        "obs_normalization_baked": norm is not None,
        "exports": [],
    }

    example = torch.zeros(1, obs_dim)

    # TorchScript.
    try:
        with torch.no_grad():
            ts = torch.jit.trace(model, example)
            # Sanity: traced output must match the eager module.
            if not torch.allclose(ts(example), model(example), atol=1e-6):
                raise RuntimeError("traced output != eager output")
        ts_path = stage / "policy_ts.pt"
        ts.save(str(ts_path))
        files.append((ts_path, "policy_ts.pt"))
        meta["exports"].append("policy_ts.pt")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"TorchScript export failed ({type(e).__name__}: {e})")

    # ONNX.
    try:
        onnx_path = stage / "policy.onnx"
        _onnx_export(model, example, onnx_path)
        files.append((onnx_path, "policy.onnx"))
        meta["exports"].append("policy.onnx")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ONNX export failed ({type(e).__name__}: {e})")

    return meta


def _mlp_layers_from_state_dict(sd: dict) -> list[tuple[int, int, int]]:
    """``[(seq_index, in_features, out_features), ...]`` sorted by index."""
    layers = []
    for k, v in sd.items():
        m = _MLP_KEY_RE.match(k)
        if m and m.group(2) == "weight" and getattr(v, "ndim", 0) == 2:
            layers.append((int(m.group(1)), int(v.shape[1]), int(v.shape[0])))
    layers.sort()
    return layers


def _mlp_structure_problem(
    actor_sd: dict, layers: list[tuple[int, int, int]],
) -> str | None:
    """Reject state dicts the Linear+activation rebuild can't represent.

    None = fine. A reason string = bail to raw-only. Guards:
    * a parameterized ``mlp.<i>`` module that is NOT a 2-D Linear weight
      (e.g. LayerNorm) would either strict-load-fail or be silently
      replaced by an activation;
    * consecutive Linear dims must chain (out_i == in_{i+1});
    * index gaps > 2 would make the rebuild stack multiple activations
      (ELU∘ELU ≠ ELU) where the original had dropout/identity modules.
    """
    linear_idx = {i for (i, _, _) in layers}
    for k in actor_sd:
        m = _MLP_KEY_RE.match(k)
        if m and int(m.group(1)) not in linear_idx:
            return f"parameterized non-Linear module at mlp.{m.group(1)}"
    for (i0, _, out0), (i1, in1, _) in zip(layers, layers[1:]):
        if out0 != in1:
            return f"dims don't chain (mlp.{i0} out={out0} vs mlp.{i1} in={in1})"
        if i1 - i0 > 2:
            return f"index gap {i1 - i0} between mlp.{i0} and mlp.{i1}"
    return None


_ACTIVATIONS = {
    "elu": "ELU", "relu": "ReLU", "tanh": "Tanh", "selu": "SELU",
    "leaky_relu": "LeakyReLU", "lrelu": "LeakyReLU", "sigmoid": "Sigmoid",
    "gelu": "GELU", "softplus": "Softplus", "silu": "SiLU", "mish": "Mish",
    # NOT here on purpose: "crelu" (doubles width — no module swap is right).
}


_RSL_RL_NORM_EPS = 1e-2  # EmpiricalNormalization default (not in the sd)


def _extract_obs_normalizer(payload: dict, actor_sd: dict, warnings: list):
    """Observation-normalizer stats to bake into the export.

    Returns None (no normalizer), a ``{"mean", "std"}`` dict of 1-D
    tensors, or the string ``"refuse"`` when normalizer-ish state exists
    but doesn't match the known rsl_rl EmpiricalNormalization shape —
    exporting a bare MLP in that case would be silently wrong.
    """
    known_prefix = "obs_normalizer."
    known = {"_mean", "_std", "_var", "count"}

    embedded = {
        k.removeprefix(known_prefix): v
        for k, v in actor_sd.items()
        if isinstance(k, str) and k.startswith(known_prefix)
    }
    payload_norm = payload.get("obs_norm_state_dict")
    other_norm_keys = [
        k for k in payload
        if isinstance(k, str) and "norm" in k.lower()
        and k != "obs_norm_state_dict"
    ] + [
        k for k in actor_sd
        if isinstance(k, str) and "norm" in k.lower()
        and not k.startswith(known_prefix)
    ]

    def _refuse(detail: str):
        warnings.append(
            f"checkpoint carries observation-normalizer state the exporter "
            f"doesn't recognise ({detail}) — a bare-MLP export would be "
            "numerically wrong, so only the raw checkpoint is bundled. "
            "Apply the normalizer stats before the MLP when deploying.")
        return "refuse"

    if other_norm_keys:
        return _refuse(f"keys: {sorted(set(other_norm_keys))}")

    source = None
    if embedded:
        if not set(embedded) <= known or not {"_mean", "_std"} <= set(embedded):
            return _refuse(f"obs_normalizer keys: {sorted(embedded)}")
        source = embedded
    elif payload_norm is not None:
        # Present but not a usable dict → refuse, never a bare-MLP export.
        if not (isinstance(payload_norm, dict) and payload_norm):
            return _refuse(
                f"obs_norm_state_dict is {type(payload_norm).__name__}"
                f"{' (empty)' if payload_norm == {} else ''}")
        stripped = {
            (k.removeprefix(known_prefix)
             if isinstance(k, str) else k): v
            for k, v in payload_norm.items()
        }
        if not {"_mean", "_std"} <= set(stripped):
            return _refuse(f"obs_norm_state_dict keys: {sorted(payload_norm)}")
        source = stripped
    if source is None:
        return None
    try:
        mean = source["_mean"].reshape(-1).to(dtype=_f32())
        std = source["_std"].reshape(-1).to(dtype=_f32())
    except Exception as e:  # noqa: BLE001 — non-tensor stats → raw-only
        return _refuse(f"normalizer stats not tensors ({type(e).__name__}: {e})")
    return {"mean": mean, "std": std}


def _f32():
    import torch

    return torch.float32


def _verify_normalizer_parity(
    model, norm: dict, obs_dim: int, warnings: list,
) -> bool:
    """When rsl_rl is importable, check the baked (x-mean)/(std+eps) against
    the REAL EmpiricalNormalization loaded with the same stats. Guards
    against the installed rsl_rl's DEFAULT eps/formula differing from the
    1e-2 baked here — a run trained with a non-default eps override is
    undetectable (eps is a constructor arg, not checkpointed)."""
    import torch

    try:
        from rsl_rl.modules.normalization import EmpiricalNormalization
    except ImportError:
        warnings.append(
            "rsl_rl not importable — baked obs normalization uses the "
            f"documented default eps={_RSL_RL_NORM_EPS} unverified")
        return True
    try:
        ref = EmpiricalNormalization(obs_dim)
        ref._mean = norm["mean"].reshape(1, -1).clone()
        ref._std = norm["std"].reshape(1, -1).clone()
        x = torch.randn(4, obs_dim, generator=torch.Generator().manual_seed(1))
        baked = (x - norm["mean"]) / (norm["std"] + _RSL_RL_NORM_EPS)
        if not torch.allclose(ref(x), baked, atol=1e-6):
            warnings.append(
                "baked obs normalization disagrees with rsl_rl's "
                "EmpiricalNormalization — raw checkpoint only")
            return False
    except Exception as e:  # noqa: BLE001 — parity check must not kill export
        warnings.append(
            f"obs-normalizer parity check errored ({type(e).__name__}: {e}) "
            "— raw checkpoint only")
        return False
    return True


def _build_mlp(
    layers: list[tuple[int, int, int]], activation: str,
    norm: "dict | None" = None,
):
    """nn.Sequential whose child indices mirror the checkpoint's ``mlp.<i>``
    keys, so ``load_state_dict`` maps 1:1 (activations occupy the gaps).
    ``norm`` bakes (x - mean) / (std + eps) in front of the MLP."""
    import torch.nn as nn

    act_cls = getattr(nn, _ACTIVATIONS.get(activation, "ELU"))
    max_idx = layers[-1][0]
    linear_at = {i: (fin, fout) for (i, fin, fout) in layers}
    mods = []
    for i in range(max_idx + 1):
        if i in linear_at:
            fin, fout = linear_at[i]
            mods.append(nn.Linear(fin, fout))
        else:
            mods.append(act_cls())
    model = nn.Sequential(*mods)
    # Wrap under attribute name "mlp" so keys line up.
    return _Actor(model, norm=norm)


def _resolve_activation(project: Path, warnings: list[str]) -> tuple[str, bool]:
    """(activation, assumed?) — exact from the mjlab task cfg when possible."""
    task_id = None
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11
            import tomli as tomllib  # type: ignore[no-redef]
        with (project / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
        task_id = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
    except Exception:  # noqa: BLE001
        pass
    if isinstance(task_id, str) and task_id:
        try:
            from mjlab.tasks.registry import load_rl_cfg

            rl_cfg = load_rl_cfg(task_id)
            actor_cfg = getattr(rl_cfg, "actor", None)
            act = getattr(actor_cfg, "activation", None)
            if isinstance(act, str) and act:
                return act, False
        except Exception as e:  # noqa: BLE001
            warnings.append(
                f"could not read activation from mjlab task cfg "
                f"({type(e).__name__}: {e}) — assuming elu")
    else:
        warnings.append("no task_id in config.toml — assuming elu activation")
    return "elu", True


def _task_id_from_config(project: Path) -> Optional[str]:
    """The adapter's mjlab `task_id` from the project config.toml (or None)."""
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11
            import tomli as tomllib  # type: ignore[no-redef]
        with (project / "config.toml").open("rb") as f:
            cfg = tomllib.load(f)
        tid = ((cfg.get("adapter") or {}).get("config") or {}).get("task_id")
        return tid if isinstance(tid, str) and tid else None
    except Exception:  # noqa: BLE001
        return None


def _resolve_per_joint(
    pattern_map: dict, joint_names: list[str],
) -> dict[str, Any]:
    """mjlab joint-name-pattern → value maps (e.g. action scale, default pose)
    resolved against the ordered `joint_names`. Each joint takes the value of
    the LAST pattern that matches it (mjlab's more-specific-later convention);
    a literal (non-regex) key falls back to exact match."""
    import re

    out: dict[str, Any] = {}
    for jn in joint_names:
        chosen: Any = None
        for pat, val in pattern_map.items():
            try:
                if re.fullmatch(str(pat), jn) or re.search(str(pat), jn):
                    chosen = val
            except re.error:
                if str(pat) == jn:
                    chosen = val
        if chosen is not None:
            out[jn] = float(chosen) if isinstance(chosen, (int, float)) else chosen
    return out


def _deployment_contract(
    project: Path, net_meta: dict[str, Any], warnings: list[str],
) -> dict[str, Any]:
    """The sim→real interface contract a hardware controller needs but the raw
    checkpoint/ONNX cannot carry: the joint ORDER the action/obs vectors index,
    the action target formula + per-joint scale + default pose, the control
    rate, and the ordered observation layout.

    `joint_names` (order) come from the captured robot manifest and are ALWAYS
    present for a known robot. The rest is read best-effort from the mjlab task
    cfg (like `_resolve_activation`); when mjlab is not importable the contract
    still ships the joint order + a flag naming what is missing, never a wrong
    guess."""
    from sculptor.eval.robot_manifest import robot_joint_names

    task_id = _task_id_from_config(project)
    joint_names = robot_joint_names(task_id) if task_id else None
    contract: dict[str, Any] = {
        "task_id": task_id,
        "joint_names": joint_names,
        "joint_order_source": (
            "reward-sculptor robot manifest (captured from real rollouts)"
            if joint_names else "unavailable — robot not in the manifest"),
        "action": {
            "output": "mean_action",
            "target_formula": (
                "q_target[j] = default_joint_pos[j] + scale[j] * action[j]"),
            "note": "action indices align 1:1 with joint_names order",
        },
    }
    if joint_names is None:
        warnings.append(
            "deployment: robot not in the joint manifest — the bundle ships "
            "network dims but no joint-order / action-scale mapping")
        contract["available"] = False
        return contract

    if not task_id:
        contract["available"] = False
        return contract
    try:
        from mjlab.tasks.registry import load_env_cfg

        ec = load_env_cfg(task_id)
        # Control rate.
        sim = getattr(ec, "sim", None)
        mj = getattr(sim, "mujoco", None) if sim else None
        dt = float(getattr(mj, "timestep", 0.0) or 0.0)
        dec = int(getattr(ec, "decimation", 0) or 0)
        if dt > 0 and dec > 0:
            contract["control"] = {
                "sim_timestep_s": dt, "decimation": dec,
                "control_dt_s": round(dt * dec, 6),
                "control_hz": round(1.0 / (dt * dec), 4),
            }
        grav = getattr(mj, "gravity", None) if mj else None
        if grav is not None:
            contract["gravity"] = [float(x) for x in grav]

        # Action term: scale + default-offset + clip.
        actions = getattr(ec, "actions", None) or {}
        for name, term in (actions.items() if hasattr(actions, "items") else []):
            scale = getattr(term, "scale", None)
            contract["action"]["term"] = name
            contract["action"]["type"] = type(term).__name__
            contract["action"]["use_default_offset"] = bool(
                getattr(term, "use_default_offset", False))
            clip = getattr(term, "clip", None)
            if clip is not None:
                contract["action"]["clip"] = clip
            if isinstance(scale, dict):
                contract["action"]["scale"] = _resolve_per_joint(
                    scale, joint_names)
            elif isinstance(scale, (int, float)):
                contract["action"]["scale"] = {
                    jn: float(scale) for jn in joint_names}
            break

        # Default joint pose (the offset the action rides on).
        scene = getattr(ec, "scene", None)
        ents = getattr(scene, "entities", None) if scene else None
        robot = ents.get("robot") if hasattr(ents, "get") else None
        init = getattr(robot, "init_state", None) if robot else None
        jp = getattr(init, "joint_pos", None) if init else None
        if isinstance(jp, dict):
            contract["default_joint_pos"] = _resolve_per_joint(jp, joint_names)

        # Observation layout — the ORDERED terms the actor vector concatenates.
        obs = getattr(ec, "observations", None) or {}
        group = obs.get("actor") or obs.get("policy") if hasattr(obs, "get") else None
        terms = getattr(group, "terms", None) if group else None
        if isinstance(terms, dict):
            layout = []
            for tname, tcfg in terms.items():
                func = getattr(tcfg, "func", None)
                layout.append({
                    "name": tname,
                    "source": getattr(func, "__name__", None) if func else None,
                    "scale": getattr(tcfg, "scale", None),
                })
            contract["observation"] = {
                "note": "concatenated in this order into the actor obs vector",
                "obs_dim": net_meta.get("obs_dim"),
                "terms": layout,
            }
        contract["available"] = True
        contract["source"] = f"mjlab task cfg '{task_id}'"
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        warnings.append(
            f"deployment: could not read the mjlab task cfg for {task_id!r} "
            f"({type(e).__name__}: {e}) — shipping joint order only")
        contract["available"] = False
    return contract


def _render_inference_py(manifest: dict[str, Any]) -> str:
    """A runnable inference skeleton parameterized by the deployment contract:
    it encodes the EXACT obs order, action formula, per-joint scale, default
    pose and control rate — the hard, error-prone part — and marks the two
    robot-SDK seams (`read_robot_state`, `send_joint_targets`) the operator
    fills in for their specific hardware."""
    dep = manifest.get("deployment") or {}
    net = manifest.get("network") or {}
    obs_dim = net.get("obs_dim", "OBS_DIM")
    hz = (dep.get("control") or {}).get("control_hz", 50.0)
    obs_terms = (dep.get("observation") or {}).get("terms") or []
    obs_lines = "\n".join(
        f"    #   {i}: {t.get('name')}  (source: {t.get('source')})"
        for i, t in enumerate(obs_terms)) or "    #   (obs layout unavailable — see env_spec.json)"
    return f'''#!/usr/bin/env python3
"""Real-hardware inference skeleton for a Reward Sculptor policy bundle.

Generated for project {manifest.get('project')!r}, iter {manifest.get('iter_index')}.
Everything hardware-INDEPENDENT is filled in from manifest.json (joint order,
action scale, default pose, observation layout, control rate). You only wire the
two SDK seams marked `TODO(hardware)` below to your robot's driver.

    python inference.py            # dry-run: zeros in, prints target angles

SAFETY: before closing the loop on a real robot, (1) verify joint_names order
matches your SDK exactly, (2) ramp to default_joint_pos slowly, (3) keep an
e-stop in reach. The policy was trained in sim; unverified transfer can be
violent.
"""
import json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "manifest.json").read_text())
DEP = M.get("deployment", {{}})
JOINT_NAMES = DEP.get("joint_names") or []
SCALE = DEP.get("action", {{}}).get("scale", {{}})
DEFAULT_POS = DEP.get("default_joint_pos", {{}})
USE_OFFSET = DEP.get("action", {{}}).get("use_default_offset", True)
CONTROL_HZ = {hz}
OBS_DIM = {obs_dim}

scale_vec = np.array([SCALE.get(j, 1.0) for j in JOINT_NAMES], dtype=np.float32)
default_vec = np.array([DEFAULT_POS.get(j, 0.0) for j in JOINT_NAMES], dtype=np.float32)


def load_policy():
    """Prefer ONNX (obs-normalization is baked in — feed RAW obs)."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(HERE / "policy.onnx"))
        return lambda obs: sess.run(["action"], {{"obs": obs[None].astype(np.float32)}})[0][0]
    except Exception:
        import torch
        net = torch.jit.load(str(HERE / "policy_ts.pt")).eval()
        return lambda obs: net(torch.as_tensor(obs[None], dtype=torch.float32)).detach().numpy()[0]


def read_robot_state():
    """TODO(hardware): return the RAW observation vector, length OBS_DIM,
    concatenated in EXACTLY this order (see manifest deployment.observation):
{obs_lines}
    Joint pos/vel are RELATIVE to default_joint_pos, in JOINT_NAMES order."""
    return np.zeros(OBS_DIM, dtype=np.float32)


def send_joint_targets(q_target):
    """TODO(hardware): command position targets (rad), one per JOINT_NAMES entry,
    to your PD controller. q_target is a float array aligned with JOINT_NAMES."""
    print("q_target:", np.round(q_target, 4).tolist())


def main():
    policy = load_policy()
    dt = 1.0 / CONTROL_HZ
    while True:
        t0 = time.perf_counter()
        obs = read_robot_state()
        action = np.asarray(policy(obs), dtype=np.float32)
        q_target = default_vec + scale_vec * action if USE_OFFSET else scale_vec * action
        send_joint_targets(q_target)
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    # Dry run: one step with a zero observation.
    policy = load_policy()
    obs = read_robot_state()
    action = np.asarray(policy(obs), dtype=np.float32)
    q = default_vec + scale_vec * action if USE_OFFSET else scale_vec * action
    print(f"obs_dim={{obs.size}} action_dim={{action.size}} control_hz={{CONTROL_HZ}}")
    print("first target angles:", np.round(q, 4).tolist()[:8], "...")
'''


def _onnx_export(model, example, onnx_path: Path) -> None:
    import torch

    try:
        torch.onnx.export(
            model, (example,), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:
        # Older/newer torch without the dynamo kwarg.
        torch.onnx.export(
            model, (example,), str(onnx_path),
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        )
    # Structural validation when onnx is importable (onnxruntime not
    # required); a malformed file must not ship silently.
    try:
        import onnx

        onnx.checker.check_model(str(onnx_path))
    except ImportError:
        pass


# ── stable-baselines3 (.zip) ───────────────────────────────────────────────

def _export_sb3_actor(
    ckpt: Path, stage: Path,
    files: list[tuple[Path, str]], warnings: list[str],
) -> dict[str, Any]:
    """Best-effort ONNX/TorchScript for an SB3 PPO policy (documented SB3
    recipe: mlp_extractor → action_net, deterministic mean action)."""
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError:
        warnings.append("stable_baselines3/torch not importable — "
                        "raw checkpoint only")
        return {}
    try:
        model = PPO.load(str(ckpt), device="cpu")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"SB3 load failed ({type(e).__name__}: {e}) — "
                        "raw checkpoint only")
        return {}

    try:
        import gymnasium.spaces as _spaces
    except ImportError:  # pragma: no cover - gym classic
        import gym.spaces as _spaces  # type: ignore[no-redef]
    if not (isinstance(model.observation_space, _spaces.Box)
            and isinstance(model.action_space, _spaces.Box)):
        warnings.append(
            f"SB3 spaces {type(model.observation_space).__name__}/"
            f"{type(model.action_space).__name__} — only Box→Box policies "
            "get an ONNX/TorchScript export; raw checkpoint bundled")
        return {}

    policy = model.policy
    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])

    # predict(deterministic=True) clips the mean action to the space
    # bounds — the export must match it exactly.
    low_np, high_np = model.action_space.low, model.action_space.high
    import numpy as _np

    clip = bool(_np.isfinite(low_np).all() and _np.isfinite(high_np).all())
    low = torch.as_tensor(low_np, dtype=torch.float32)
    high = torch.as_tensor(high_np, dtype=torch.float32)

    class _SB3Actor(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.register_buffer("low", low)
            self.register_buffer("high", high)
            self.clip = clip

        def forward(self, obs):
            feats = self.p.extract_features(
                obs, self.p.features_extractor)
            latent_pi = self.p.mlp_extractor.forward_actor(feats)
            action = self.p.action_net(latent_pi)
            if self.clip:
                action = torch.clamp(action, self.low, self.high)
            return action

    actor = _SB3Actor(policy).eval()
    example = torch.zeros(1, obs_dim)
    meta: dict[str, Any] = {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "output": "mean_action",
        "action_clipped_to_space": clip,
        "exports": [],
    }
    # VecNormalize stats live OUTSIDE checkpoint.zip. If the project has
    # them on disk we can't fold them in here — say so rather than let a
    # normalized-obs policy run on raw observations.
    if (ckpt.parent / "vecnormalize.pkl").is_file():
        warnings.append(
            "vecnormalize.pkl found next to the checkpoint — the exported "
            "network expects NORMALIZED observations; apply the "
            "VecNormalize stats at deployment")
    try:
        with torch.no_grad():
            ts = torch.jit.trace(actor, example)
        ts_path = stage / "policy_ts.pt"
        ts.save(str(ts_path))
        files.append((ts_path, "policy_ts.pt"))
        meta["exports"].append("policy_ts.pt")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"TorchScript export failed ({type(e).__name__}: {e})")
    try:
        onnx_path = stage / "policy.onnx"
        _onnx_export(actor, example, onnx_path)
        files.append((onnx_path, "policy.onnx"))
        meta["exports"].append("policy.onnx")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ONNX export failed ({type(e).__name__}: {e})")
    return meta


# ── misc ───────────────────────────────────────────────────────────────────

def _load_strict_json_object(data: bytes, *, label: str) -> dict[str, Any]:
    """Parse a bounded JSON object without duplicate keys or non-finite data."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains forbidden constant {value!r}")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be an object")
    return parsed


def _validate_reference_clip_container(path: Path) -> None:
    """Bound the nested NPZ before NumPy allocates any of its arrays."""
    size = path.stat().st_size
    if size <= 0:
        raise ExportError("reference clip is empty")
    if size > _REFERENCE_ARCHIVE_MAX_BYTES:
        raise ExportError(
            "reference clip exceeds the 512 MiB portable-export limit"
        )
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > _REFERENCE_MEMBER_MAX:
                raise ExportError("reference clip has too many NPZ members")
            seen: set[str] = set()
            expanded = 0
            for info in infos:
                if info.flag_bits & 0x1:
                    raise ExportError("reference clip contains encrypted data")
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                file_kind = unix_mode & 0o170000
                if file_kind not in (0, 0o100000):
                    raise ExportError(
                        "reference clip contains a link or special NPZ member"
                    )
                member = info.filename
                if (
                    not member
                    or "\\" in member
                    or member.startswith("/")
                    or any(part in ("", ".", "..") for part in member.split("/"))
                ):
                    raise ExportError(
                        f"reference clip contains unsafe NPZ member {member!r}"
                    )
                if info.is_dir() or not member.endswith(".npy"):
                    raise ExportError(
                        f"reference clip contains non-array member {member!r}"
                    )
                folded = member.casefold()
                if folded in seen:
                    raise ExportError(
                        f"reference clip contains colliding member {member!r}"
                    )
                seen.add(folded)
                expanded += int(info.file_size)
                if expanded > _REFERENCE_EXPANDED_MAX_BYTES:
                    raise ExportError(
                        "expanded reference clip exceeds the 1 GiB limit"
                    )
    except zipfile.BadZipFile as exc:
        raise ExportError("reference clip is not a valid NPZ container") from exc


def _write_zip_member_verified(
    archive: zipfile.ZipFile,
    source_path: Path,
    archive_name: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Stream one file into the ZIP and attest the bytes actually written."""
    digest = hashlib.sha256()
    written = 0
    with source_path.open("rb") as source, archive.open(
        archive_name, "w", force_zip64=True,
    ) as destination:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            destination.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if written != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ExportError(
            "reference clip changed while it was being exported; retry from "
            "a stable library snapshot"
        )


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _sculptor_version() -> str:
    try:
        from importlib.metadata import version

        return version("reward-sculptor")
    except Exception:  # noqa: BLE001
        return "unknown"


def _render_deploy_md(manifest: dict[str, Any]) -> str:
    net = manifest.get("network") or {}
    fmt = manifest["checkpoint"]["format"]
    obs = net.get("obs_dim", "?")
    act = net.get("action_dim", "?")
    activation = net.get("activation", "elu")
    lines = [
        f"# Policy bundle — {manifest['project']} · iter {manifest['iter_index']}",
        "",
        f"Exported by Reward Sculptor {manifest['sculptor_version']} "
        f"on {manifest['created_at']}.",
        "",
        "| | |",
        "|---|---|",
        f"| Checkpoint format | `{fmt}` |",
        f"| Observation dim | `{obs}` |",
        f"| Action dim | `{act}` |",
        f"| Reward version | `{manifest.get('reward_version')}` |",
        f"| Env spec source | `{manifest.get('env_spec_source')}` |",
        "",
        "The network outputs the **mean action** of the Gaussian policy "
        "(deterministic deployment). Observation layout, scaling, and "
        "action post-processing must match training — the sim→real contract "
        "below (and `manifest.json` → `deployment`) captures exactly that.",
        "",
    ]
    dep = manifest.get("deployment") or {}
    if dep.get("joint_names"):
        jn = dep["joint_names"]
        act = dep.get("action") or {}
        ctrl = dep.get("control") or {}
        lines += [
            "## Sim→real hardware contract",
            "",
            "**`inference.py` is a runnable skeleton** with all of this baked "
            "in — you only fill the two `TODO(hardware)` seams (`read_robot_"
            "state`, `send_joint_targets`) for your robot's SDK.",
            "",
            f"- **Joints ({len(jn)})**, in the order the action/obs vectors "
            f"index (source: {dep.get('joint_order_source')}):",
            "  ```",
            "  " + ", ".join(jn),
            "  ```",
        ]
        if ctrl.get("control_hz"):
            lines.append(
                f"- **Control rate:** {ctrl['control_hz']} Hz "
                f"(sim dt {ctrl.get('sim_timestep_s')}s × decimation "
                f"{ctrl.get('decimation')} = {ctrl.get('control_dt_s')}s).")
        if act.get("target_formula"):
            off = ("uses the default-pose offset" if act.get("use_default_offset")
                   else "does NOT use a default-pose offset")
            lines.append(
                f"- **Action → joint targets:** `{act['target_formula']}` "
                f"(this policy {off}).")
        if dep.get("default_joint_pos") is not None:
            lines.append(
                "- **`default_joint_pos` + per-joint `action` `scale`** are in "
                "`manifest.json` → `deployment` — apply them exactly.")
        if (dep.get("observation") or {}).get("terms"):
            order = " → ".join(
                t.get("name") for t in dep["observation"]["terms"])
            lines.append(
                f"- **Observation order** (concatenate in this order): {order}.")
        if not dep.get("available", True):
            lines.append(
                "- ⚠️ The mjlab task cfg was not readable at export time, so "
                "action scale / default pose / obs layout may be absent — see "
                "`env_spec.json` and `config.toml`, and the warnings below.")
        lines.append("")
    if net.get("obs_normalization_baked") and net.get("exports"):
        lines += [
            "Observation normalization — `(x - mean) / (std + 0.01)`, the "
            "rsl_rl EmpiricalNormalization the policy trained with — is "
            "**baked into** `policy.onnx` / `policy_ts.pt`: feed them RAW "
            "observations. The raw checkpoint keeps the normalizer as "
            "separate state you must apply yourself.",
            "",
        ]
    if "policy.onnx" in (net.get("exports") or []):
        lines += [
            "## ONNX",
            "```python",
            "import onnxruntime as ort",
            "import numpy as np",
            'sess = ort.InferenceSession("policy.onnx")',
            f"obs = np.zeros((1, {obs}), dtype=np.float32)  # your observation",
            'action = sess.run(["action"], {"obs": obs})[0]',
            "```",
            "",
        ]
    if "policy_ts.pt" in (net.get("exports") or []):
        lines += [
            "## TorchScript",
            "```python",
            "import torch",
            'policy = torch.jit.load("policy_ts.pt").eval()',
            f"action = policy(torch.zeros(1, {obs}))",
            "```",
            "",
        ]
    if fmt == "rsl_rl":
        hidden = net.get("hidden_dims", [])
        lines += [
            "## Raw checkpoint (rsl_rl format)",
            "```python",
            "import torch",
            'ckpt = torch.load("checkpoint.pt", map_location="cpu",'
            " weights_only=False)",
            'actor_sd = ckpt["actor_state_dict"]  # mlp.<i>.weight/.bias',
            f"# MLP: {obs} -> {' -> '.join(str(h) for h in hidden)} -> {act},"
            f" activation={activation}",
            "# Or load into a full rsl_rl runner:",
            "#   runner.load(path, load_cfg={'actor': True, 'critic': True,",
            "#               'optimizer': False, 'iteration': False, 'rnd': False})",
            "```",
        ]
    else:
        lines += [
            "## Raw checkpoint (stable-baselines3)",
            "```python",
            "from stable_baselines3 import PPO",
            'model = PPO.load("checkpoint.zip", device="cpu")',
            "action, _ = model.predict(obs, deterministic=True)",
            "```",
        ]
    if manifest.get("warnings"):
        lines += ["", "## Export warnings", ""]
        lines += [f"- {w}" for w in manifest["warnings"]]
    lines.append("")
    return "\n".join(lines)


def _Actor(mlp, norm=None):  # noqa: N802 — lazy-torch factory
    """Deterministic actor head: obs -> mean action, weights under ``mlp``
    so checkpoint keys (``mlp.<i>.weight``) load 1:1. When ``norm`` is
    given, applies rsl_rl's (x - mean) / (std + eps) before the MLP."""
    import torch
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, seq: nn.Sequential):
            super().__init__()
            self.mlp = seq
            self.normalize = norm is not None
            if norm is not None:
                self.register_buffer("obs_mean", norm["mean"].clone())
                self.register_buffer("obs_std", norm["std"].clone())
            else:
                # Registered regardless so TorchScript tracing sees stable
                # attribute types; unused when normalize is False.
                self.register_buffer("obs_mean", torch.zeros(1))
                self.register_buffer("obs_std", torch.ones(1))

        def forward(self, obs):
            if self.normalize:
                obs = (obs - self.obs_mean) / (self.obs_std + _RSL_RL_NORM_EPS)
            return self.mlp(obs)

    return Actor(mlp)
