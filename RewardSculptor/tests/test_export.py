"""tests/test_export.py — policy export bundles.

Exercises the bundle builder against hand-built projects on disk using the
real Go1 rsl_rl checkpoint fixture (network reconstruction + ONNX/
TorchScript) and degraded projects (missing sidecars, corrupt checkpoints)
to prove the raw-checkpoint fallback never dies. Numeric parity between
the TorchScript export and a hand-built reference MLP is asserted exactly
— a silently-wrong exported policy is the one failure mode this feature
must never have.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sculptor.cli import app
from sculptor.export import (
    DEPLOYMENT_BUNDLE_KIND,
    ExportError,
    export_policy_bundle,
    export_reference_starting_skill_bundle,
    export_starting_skill_bundle,
    list_exportable_iters,
)
from sculptor.reference import save_clip
from sculptor.refs import library as reference_library
from sculptor.skill_bundle import (
    BUNDLE_KIND,
    ImportTarget,
    SkillBundleError,
    StartingSkillBundleImporter,
)
from sculptor.skill_library import SkillLibrary
from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

FIXTURES = Path(__file__).parent / "fixtures"
GO1_CKPT = FIXTURES / "go1_smoke_checkpoint.pt"


# ── helpers ────────────────────────────────────────────────────────────────

def _make_project(
    tmp_path: Path,
    *,
    iters: list[int] = (0,),
    checkpoint: Path | None = GO1_CKPT,
    reward_version: str = "v1",
    with_env_current: bool = True,
    with_iter_env: bool = False,
    task_id: str | None = "Mjlab-Velocity-Flat-Unitree-Go1",
) -> Path:
    project = tmp_path / "proj"
    rewards = project / "rewards"
    rewards.mkdir(parents=True)
    (rewards / f"{reward_version}.py").write_text(
        f"REWARD_SPEC = {{'version': {reward_version!r}}}\n"
        "def compute_reward(state, action, next_state, info):\n"
        "    return 0.0, {}\n",
        encoding="utf-8")
    cfg = '[target]\nname = "proj"\n\n[adapter]\n' \
          'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
    if task_id:
        cfg += f'config = {{ task_id = "{task_id}" }}\n'
    (project / "config.toml").write_text(cfg, encoding="utf-8")
    (project / "metadata.json").write_text(json.dumps({
        "robot_source": {
            "kind": "library",
            "library_slug": "unitree_go1",
            "reference_robot": "go1",
        },
    }), encoding="utf-8")
    env_spec = {
        "env_spec_version": 1,
        "meta": {"version": "v2"},
        "shared": {},
        "train": {},
    }
    if with_env_current:
        env = project / "env"
        env.mkdir()
        (env / "current.json").write_text(json.dumps(env_spec))
    for i in iters:
        it = project / "runs" / f"iter_{i}"
        it.mkdir(parents=True)
        if checkpoint is not None:
            shutil.copy(checkpoint, it / "checkpoint.pt")
        (it / "reward_spec.json").write_text(
            json.dumps({"version": reward_version, "author": "sculptor"}))
        (it / "metrics.json").write_text(
            json.dumps({"metrics": {"mean_return": 10.0 + i}}))
        if with_iter_env:
            (it / "env_spec.json").write_text(json.dumps(env_spec))
    if iters:
        store = WorldArtifactStore(project)
        env_version = project / "env" / "v1.json"
        env_version.write_text(json.dumps(env_spec), encoding="utf-8")
        refs = {
            "reward": ArtifactRef.from_path(
                "reward", reward_version, rewards / f"{reward_version}.py",
                base=project,
            ),
            "env_spec": ArtifactRef.from_path(
                "env_spec", "v1", env_version, base=project,
            ),
            "world": store.write_json("world", {"shared": {}}),
            "task": store.write_json("task", {"shared": {}}),
            "resolved_eval": store.write_json("resolved_eval", {}),
            "channel_catalog": store.write_json("channel_catalog", {}),
            "clarifications": store.write_json("clarifications", {}),
        }
        selection = store.promote(refs, evaluation_lineage="test-source")
        selection_path = project / "env" / (
            f"selection_v{selection.selection_version}.json"
        )
        for i in iters:
            shutil.copyfile(
                selection_path,
                project / "runs" / f"iter_{i}" / "artifact_tuple.json",
            )
    return project


def _make_library_reference(
    root: Path,
    *,
    robot: str = "g1",
    clip_id: str = "parkour_seed",
) -> tuple[Path, Path, dict]:
    clip_dir = reference_library.clip_dir(robot, clip_id, root=root)
    clip_path = clip_dir / reference_library.CLIP_FILENAME
    n_frames = 30
    time = np.arange(n_frames, dtype=np.float64) / 30.0
    save_clip(clip_path, {
        "root_pos_z": np.full(n_frames, 0.78),
        "root_pos_xy": np.stack((0.3 * time, np.zeros_like(time)), axis=1),
        "fps": 30.0,
        "joint_pos": np.zeros((n_frames, 2)),
        "joint_names": ["j0", "j1"],
        "meta": {"source": "test_export.reference"},
    })
    provenance = reference_library.make_provenance(
        clip_id=clip_id,
        robot=robot,
        source={"kind": "unit-test", "dataset": "synthetic"},
        license="CC0-1.0",
        attribution="RewardSculptor test fixture",
        content_sha256_=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        fps_source=30.0,
        labels=["parkour", "complex-motion"],
        text="A complex G1 parkour seed",
    )
    provenance_path = reference_library.write_provenance(
        robot, clip_id, provenance, root=root,
    )
    return clip_path, provenance_path, provenance


def _use_base_policy_contract_for_synthetic_world(
    monkeypatch: pytest.MonkeyPatch, project: Path,
) -> None:
    """Keep exporter tests independent from full world-admission fixtures.

    ``_make_project`` intentionally writes minimal world tuple members because
    these tests exercise archive construction, robot namespacing, and hostile
    import.  Policy-contract/world-overlay behavior has its own real admitted
    fixtures.  Delegate to the real base-task builder while still asserting
    that the exporter supplied the immutable iteration selection it owns.
    """
    from sculptor.policy_contract import build_project_policy_contract

    absent_selection = project / "env" / "synthetic-base-selection-absent.json"

    def build_base_contract(
        source_project: Path,
        *,
        observed_network=None,
        world_selection_path=None,
    ):
        assert Path(source_project) == project
        assert world_selection_path is not None
        return build_project_policy_contract(
            source_project,
            observed_network=observed_network,
            world_selection_path=absent_selection,
        )

    monkeypatch.setattr(
        "sculptor.policy_contract.build_project_policy_contract",
        build_base_contract,
    )


def _persist_origin_policy_contract(
    project: Path,
    *,
    iter_index: int = 0,
):
    """Model the training-time sidecar required by portable policy export."""
    from sculptor.policy_contract import build_project_policy_contract

    deployment = export_policy_bundle(project, iter_index=iter_index)
    tuple_payload = json.loads(
        (project / "runs" / f"iter_{iter_index}" / "artifact_tuple.json")
        .read_text(encoding="utf-8")
    )
    selection_path = project / "env" / (
        f"selection_v{int(tuple_payload['selection_version'])}.json"
    )
    contract = build_project_policy_contract(
        project,
        observed_network=deployment.manifest["network"],
        world_selection_path=selection_path,
    )
    sidecar = (
        project
        / "runs"
        / f"iter_{iter_index}"
        / "warm_start_effective_policy_contract.json"
    )
    sidecar.write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return deployment


# ── discovery ──────────────────────────────────────────────────────────────

def test_list_exportable_iters_empty_and_missing(tmp_path):
    assert list_exportable_iters(tmp_path / "nope") == []
    (tmp_path / "runs").mkdir()
    assert list_exportable_iters(tmp_path / "runs") == []


def test_list_exportable_iters_skips_checkpointless(tmp_path):
    project = _make_project(tmp_path, iters=[0, 2])
    # iter_1 exists but has no checkpoint → excluded
    (project / "runs" / "iter_1").mkdir()
    rows = list_exportable_iters(project / "runs")
    assert [r["iter_index"] for r in rows] == [0, 2]
    assert rows[0]["checkpoint"] == "checkpoint.pt"
    assert rows[0]["reward_version"] == "v1"
    assert rows[1]["primary_metric"] == pytest.approx(12.0)


def test_list_exportable_iters_ignores_empty_checkpoint(tmp_path):
    project = _make_project(tmp_path, iters=[0], checkpoint=None)
    (project / "runs" / "iter_0" / "checkpoint.pt").touch()
    assert list_exportable_iters(project / "runs") == []


def test_list_exportable_iters_prefers_authoritative_fitness_file(tmp_path):
    project = _make_project(tmp_path, iters=[0])
    iteration = project / "runs" / "iter_0"
    (iteration / "fitness.json").write_text(json.dumps({"fitness": 0.42}))
    rollout = iteration / "rollout"
    rollout.mkdir()
    (rollout / "behavior.json").write_text(json.dumps({"fitness": 0.11}))

    rows = list_exportable_iters(project / "runs")

    assert rows[0]["fitness"] == pytest.approx(0.42)


# ── bundle building (rsl_rl path, real fixture) ────────────────────────────

def test_export_bundle_contents_and_manifest(tmp_path):
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)

    assert res.bundle_path.is_file()
    assert res.bundle_path.parent == project / "exports"
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert {
        "manifest.json", "checkpoint.pt", "policy.onnx", "policy_ts.pt",
        "reward/reward_spec.json", "reward/v1.py", "env_spec.json",
        "config.toml", "metrics.json", "DEPLOY.md",
    } <= names

    assert manifest["schema_version"] == 2
    assert manifest["artifact_purpose"] == "reproducibility"
    assert manifest["deployment_status"] == "not_certified"
    assert manifest["deployment_authority"]["status"] == "not_certified"
    assert manifest["compatibility_contract"] is None
    assert manifest["compatibility_contract_digest"] is None
    assert any(
        "compatibility contract intentionally omitted" in item
        for item in res.warnings
    )
    assert any("NOT DEPLOYMENT CERTIFIED" in item for item in res.warnings)
    assert manifest["iter_index"] == 0
    assert manifest["reward_version"] == "v1"
    assert manifest["checkpoint"]["format"] == "rsl_rl"
    net = manifest["network"]
    assert net["obs_dim"] == 48
    assert net["action_dim"] == 12
    assert net["hidden_dims"] == [512, 256, 128]
    assert net["output"] == "mean_action"
    # every listed file carries a sha256
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])


def test_export_rejects_source_replacement_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-size atomic replacement cannot race snapshot capture."""
    from sculptor import export as export_module

    project = _make_project(tmp_path)
    checkpoint = project / "runs" / "iter_0" / "checkpoint.pt"
    output = tmp_path / "published.zip"
    real_copy = export_module._copy_export_source
    mutated = False

    def copy_then_replace(source, source_handle, destination_handle):
        nonlocal mutated
        copied = real_copy(source, source_handle, destination_handle)
        if not mutated and Path(source) == checkpoint:
            replacement = checkpoint.with_name("checkpoint.replacement")
            replacement.write_bytes(b"x" * checkpoint.stat().st_size)
            os.replace(replacement, checkpoint)
            mutated = True
        return copied

    monkeypatch.setattr(
        export_module,
        "_copy_export_source",
        copy_then_replace,
    )

    with pytest.raises(ExportError, match="changed during capture"):
        export_policy_bundle(project, out_path=output)

    assert mutated is True
    assert not output.exists()
    assert not list(output.parent.glob(".rs_export_*"))


def test_export_rejects_symlinked_iteration_parent(
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path)
    iteration = project / "runs" / "iter_0"
    backing = project / "runs" / "iteration_backing"
    iteration.rename(backing)
    iteration.symlink_to(backing, target_is_directory=True)
    output = tmp_path / "published.zip"

    with pytest.raises(
        ExportError,
        match="traverses a symlink or non-directory",
    ):
        export_policy_bundle(project, out_path=output)

    assert not output.exists()


def test_export_rejects_output_aliasing_source_artifact(
    tmp_path: Path,
) -> None:
    project = _make_project(tmp_path)
    checkpoint = project / "runs" / "iter_0" / "checkpoint.pt"
    original_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(ExportError, match="destination aliases a source"):
        export_policy_bundle(project, out_path=checkpoint)

    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_sha256


def test_export_metadata_uses_captured_config_after_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sculptor import export as export_module

    project = _make_project(tmp_path)
    config_path = project / "config.toml"
    original_config = config_path.read_bytes()
    real_actor_export = export_module._export_rsl_rl_actor

    def mutate_source_then_export(
        checkpoint,
        captured_config,
        stage,
        files,
        warnings,
    ):
        config_path.write_text(
            '[adapter]\nconfig = { task_id = "mutated-live-task" }\n',
            encoding="utf-8",
        )
        return real_actor_export(
            checkpoint,
            captured_config,
            stage,
            files,
            warnings,
        )

    monkeypatch.setattr(
        export_module,
        "_export_rsl_rl_actor",
        mutate_source_then_export,
    )

    result = export_policy_bundle(project, out_path=tmp_path / "captured.zip")

    assert result.manifest["deployment"]["task_id"] == (
        "Mjlab-Velocity-Flat-Unitree-Go1"
    )
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.read("config.toml") == original_config


def test_portable_starting_skill_is_data_only_and_importable(
    tmp_path, monkeypatch,
):
    project = _make_project(tmp_path)
    _use_base_policy_contract_for_synthetic_world(monkeypatch, project)
    deployment = _persist_origin_policy_contract(project)
    portable = export_starting_skill_bundle(
        project, robot_slug="go1",
    )

    assert deployment.manifest["kind"] == DEPLOYMENT_BUNDLE_KIND
    assert portable.bundle_path.suffix == ".rskill"
    assert portable.manifest["kind"] == BUNDLE_KIND
    with zipfile.ZipFile(portable.bundle_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert names == {
        "manifest.json",
        "policy/weights.safetensors",
        "provenance/origin_policy_contract.json",
    }
    assert manifest["checkpoint"]["included"] is False
    assert {
        "checkpoint.pt", "policy_ts.pt", "policy.onnx", "inference.py",
        "DEPLOY.md", "config.toml", "env_spec.json",
    }.isdisjoint(names)

    contract = manifest["compatibility_contract"]
    identity = contract["identity"]
    with pytest.raises(
        SkillBundleError,
        match="unsupported portable bundle member|deployment ZIPs are not portable",
    ):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "deployment-rejection-library")
        ).import_archive(
            deployment.bundle_path,
            target=ImportTarget(
                adapter_class=identity["adapter_class"],
                task_id=identity["task_id"],
                robot_slug="go1",
                compatibility_contract=contract,
            ),
        )

    imported = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "portable-library")
    ).import_archive(
        portable.bundle_path,
        target=ImportTarget(
            adapter_class=identity["adapter_class"],
            task_id=identity["task_id"],
            robot_slug="go1",
            compatibility_contract=contract,
        ),
    )
    assert imported.record.policy_roles[0] == "actor"
    assert imported.record.original_checkpoint_sha256 == (
        deployment.manifest["checkpoint"]["sha256"]
    )
    assert imported.record.compatibility_contract_provenance_status == (
        "origin_persisted"
    )


def test_portable_export_fails_closed_without_origin_contract_sidecar(
    tmp_path, monkeypatch,
):
    project = _make_project(tmp_path)
    _use_base_policy_contract_for_synthetic_world(monkeypatch, project)

    with pytest.raises(ExportError, match="persisted when training ran"):
        export_starting_skill_bundle(project, robot_slug="go1")


def test_portable_export_requires_rskill_extension(tmp_path):
    project = _make_project(tmp_path)
    with pytest.raises(ExportError, match="must end in .rskill"):
        export_starting_skill_bundle(
            project, out_path=tmp_path / "not-portable.zip", robot_slug="go1",
        )


def test_portable_export_derives_and_cross_checks_project_robot(
    tmp_path, monkeypatch,
):
    project = _make_project(tmp_path)
    _use_base_policy_contract_for_synthetic_world(monkeypatch, project)
    (project / "metadata.json").write_text(json.dumps({
        # Catalog identity and policy/reference identity are deliberately
        # different for this legacy G1 project.
        "robot_source": {
            "kind": "library", "library_slug": "unitree_g1",
        },
    }))

    _persist_origin_policy_contract(project)

    result = export_starting_skill_bundle(project)
    assert result.manifest["starting_skill"]["robot_slug"] == "g1"
    assert result.manifest["deployment"]["robot_slug"] == "g1"
    identity = result.manifest["compatibility_contract"]["identity"]
    imported = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "g1-portable-library")
    ).import_archive(
        result.bundle_path,
        target=ImportTarget(
            adapter_class=identity["adapter_class"],
            task_id=identity["task_id"],
            robot_slug="g1",
            compatibility_contract=result.manifest["compatibility_contract"],
        ),
    )
    assert imported.receipt["selectable"] is True
    assert imported.record.robot_slug == "g1"

    export_starting_skill_bundle(project, robot_slug="g1")
    with pytest.raises(ExportError, match="canonical reference robot"):
        export_starting_skill_bundle(project, robot_slug="unitree_g1")


def test_portable_export_contract_is_owned_by_iteration_tuple(
    tmp_path, monkeypatch,
):
    project = _make_project(tmp_path)
    store = WorldArtifactStore(project)
    source = store.read_selection()
    assert source is not None
    source_path = project / "env" / (
        f"selection_v{source.selection_version}.json"
    )

    # Promote a different current tuple after the checkpoint already owns its
    # immutable source tuple. The portable contract must still be built from
    # the historical selection, never this mutable current pointer.
    current_refs = dict(source.refs)
    current_refs["task"] = store.write_json(
        "task", {"shared": {"marker": "new-current-world"}},
    )
    current = store.promote(
        current_refs, evaluation_lineage="test-new-current",
    )
    current_path = project / "env" / (
        f"selection_v{current.selection_version}.json"
    )

    def fake_contract(
        _project, *, observed_network=None, world_selection_path=None,
    ):
        selected = (
            Path(world_selection_path)
            if world_selection_path is not None
            else current_path
        )
        return {
            "schema": 2,
            "identity": {
                "adapter_class": "sculptor.adapters.mjlab.MjlabAdapter",
                "task_id": "Mjlab-Velocity-Flat-Unitree-Go1",
            },
            "source_selection": selected.name,
        }

    monkeypatch.setattr(
        "sculptor.policy_contract.build_project_policy_contract",
        fake_contract,
    )
    _persist_origin_policy_contract(project, iter_index=0)
    result = export_starting_skill_bundle(project, iter_index=0)

    assert result.manifest["compatibility_contract"][
        "source_selection"
    ] == source_path.name
    receipt = result.manifest["source_artifact_tuple"]
    assert receipt["selection_version"] == source.selection_version
    assert receipt["tuple_hash"] == source.tuple_hash
    assert len(receipt["artifact_tuple_sha256"]) == 64
    assert len(receipt["selection_sha256"]) == 64
    assert receipt["compatibility_contract_digest"] == (
        result.manifest["compatibility_contract_digest"]
    )


def test_portable_export_rejects_missing_iteration_tuple(tmp_path):
    project = _make_project(tmp_path)
    (project / "runs" / "iter_0" / "artifact_tuple.json").unlink()

    with pytest.raises(ExportError, match="iteration-owned artifact tuple"):
        export_starting_skill_bundle(project, iter_index=0)


def test_reference_starting_skill_round_trips_through_importer(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source-references"
    clip_path, provenance_path, _ = _make_library_reference(source_root)
    out = tmp_path / "exports" / "g1-parkour.rskill"

    exported = export_reference_starting_skill_bundle(
        robot_slug="g1",
        clip_id="parkour_seed",
        out_path=out,
        name="G1 parkour exploration seed",
        references_root=source_root,
    )

    assert exported.bundle_path == out.resolve()
    assert exported.manifest["starting_skill"] == {
        "name": "G1 parkour exploration seed",
        "robot_slug": "g1",
    }
    assert exported.manifest["reference"]["content_sha256"] == hashlib.sha256(
        clip_path.read_bytes()
    ).hexdigest()
    assert exported.manifest["reference"]["provenance_sha256"] == hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    assert any(
        "separate target-project Tier-D exact-schedule tracking evidence job"
        in warning
        for warning in exported.warnings
    )
    assert any(
        "launch only re-verifies" in warning
        for warning in exported.warnings
    )
    with zipfile.ZipFile(out) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "motion/clip.npz",
            "motion/provenance.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        for descriptor in manifest["files"]:
            payload = archive.read(descriptor["path"])
            assert descriptor["bytes"] == len(payload)
            assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()

    imported_root = tmp_path / "imported-references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(imported_root))
    imported = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "starting-skill-library")
    ).import_archive(
        out,
        target=ImportTarget(
            adapter_class="target.Adapter",
            task_id="Target-Task",
            robot_slug="g1",
            compatibility_contract=None,
        ),
    )
    assert imported.record.policy_roles == []
    assert imported.record.initialization_modes == ["reference_only"]
    assert imported.record.reference_robot == "g1"
    assert imported.record.reference_clip_id == "parkour_seed"
    assert imported.receipt["authorization"]["status"] == "candidate"
    assert imported.receipt["authorization"]["training_authorized"] is False
    assert (
        imported_root / "g1" / "parkour_seed" / "clip.npz"
    ).read_bytes() == clip_path.read_bytes()


@pytest.mark.parametrize(
    ("robot", "clip_id", "message"),
    [
        ("../g1", "parkour_seed", "safe stable library identifier"),
        ("g1", "../parkour_seed", "invalid reference clip identity"),
        ("g1/other", "parkour_seed", "safe stable library identifier"),
    ],
)
def test_reference_export_rejects_hostile_library_identity(
    tmp_path, robot, clip_id, message,
):
    with pytest.raises(ExportError, match=message):
        export_reference_starting_skill_bundle(
            robot_slug=robot,
            clip_id=clip_id,
            out_path=tmp_path / "skill.rskill",
            references_root=tmp_path / "references",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("robot", "h1", "provenance robot does not match"),
        ("clip_id", "different_clip", "provenance clip_id does not match"),
        ("robot", None, "missing required field: robot"),
        ("clip_id", None, "missing required field: clip_id"),
    ],
)
def test_reference_export_requires_exact_provenance_identity(
    tmp_path, field, value, message,
):
    source_root = tmp_path / "references"
    _, provenance_path, provenance = _make_library_reference(source_root)
    if value is None:
        provenance.pop(field)
    else:
        provenance[field] = value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ExportError, match=message):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=tmp_path / "skill.rskill",
            references_root=source_root,
        )


def test_reference_export_rejects_missing_and_duplicate_provenance(
    tmp_path,
):
    source_root = tmp_path / "references"
    _, provenance_path, _ = _make_library_reference(source_root)
    provenance_path.unlink()
    with pytest.raises(ExportError, match="provenance is missing"):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=tmp_path / "missing.rskill",
            references_root=source_root,
        )

    _, provenance_path, _ = _make_library_reference(source_root)
    raw = provenance_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"robot": "g1"', '"robot": "g1", "robot": "g1"', 1,
    )
    provenance_path.write_text(raw, encoding="utf-8")
    with pytest.raises(ExportError, match="duplicate key 'robot'"):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=tmp_path / "duplicate.rskill",
            references_root=source_root,
        )


def test_reference_export_rejects_clip_digest_drift_and_missing_bytes(
    tmp_path,
):
    source_root = tmp_path / "references"
    clip_path, _, _ = _make_library_reference(source_root)
    clip_path.write_bytes(clip_path.read_bytes() + b"drift")
    with pytest.raises(ExportError, match="digest does not match provenance"):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=tmp_path / "drift.rskill",
            references_root=source_root,
        )

    clip_path.unlink()
    with pytest.raises(ExportError, match="clip bytes are missing"):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=tmp_path / "missing.rskill",
            references_root=source_root,
        )


def test_reference_export_is_atomic_when_archive_write_fails(
    tmp_path, monkeypatch,
):
    import sculptor.export as export_module

    source_root = tmp_path / "references"
    _make_library_reference(source_root)
    out = tmp_path / "existing.rskill"
    out.write_bytes(b"previous complete export")

    def fail_on_clip(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(export_module, "_write_zip_member_verified", fail_on_clip)
    with pytest.raises(OSError, match="simulated write failure"):
        export_reference_starting_skill_bundle(
            robot_slug="g1",
            clip_id="parkour_seed",
            out_path=out,
            references_root=source_root,
        )
    assert out.read_bytes() == b"previous complete export"
    assert list(tmp_path.glob(".existing.rskill.*.tmp")) == []


def test_refs_export_skill_cli_uses_exact_identity_and_clear_candidate_copy(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "references"
    _make_library_reference(source_root)
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(source_root))
    out = tmp_path / "cli" / "parkour.rskill"

    result = CliRunner().invoke(app, [
        "refs", "export-skill",
        "--robot", "g1",
        "--clip", "parkour_seed",
        "--out", str(out),
        "--name", "Research parkour seed",
    ])

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "pinned g1/parkour_seed" in result.output
    assert "candidate only" in result.output
    assert (
        "separate sculpt refs track Tier-D exact-schedule tracking evidence job"
        in result.output
    )
    assert "before live launch" in result.output
    assert "launch only re-verifies" in result.output


def test_deployment_contract_and_inference_script(tmp_path):
    """The raw bundle records a best-effort policy interface without turning
    that metadata or the illustrative inference hooks into certification."""
    project = _make_project(tmp_path)  # real Go1 task_id
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        infer_src = zf.read("inference.py").decode()

    assert "inference.py" in names
    dep = manifest["deployment"]
    assert dep["available"] is True
    # joint ORDER the action/obs vectors index — the Go1 12-DOF layout
    jn = dep["joint_names"]
    assert len(jn) == 12
    assert jn[0] == "FR_hip_joint" and jn[3] == "FL_hip_joint"
    # control rate 0.005 s * decimation 4 = 50 Hz
    assert dep["control"]["control_hz"] == pytest.approx(50.0)
    # action → joint-target contract
    assert dep["action"]["use_default_offset"] is True
    assert "default_joint_pos" in dep["action"]["target_formula"]
    assert set(dep["action"]["scale"]) == set(jn)
    assert set(dep["default_joint_pos"]) == set(jn)
    assert dep["default_joint_pos"]["FR_thigh_joint"] == pytest.approx(0.9)
    # ordered observation layout
    obs_terms = [t["name"] for t in dep["observation"]["terms"]]
    assert "projected_gravity" in obs_terms and "joint_pos" in obs_terms
    # the inference skeleton is parameterized, not a stub
    assert "read_robot_state" in infer_src and "send_joint_targets" in infer_src
    assert "default_vec + scale_vec * action" in infer_src
    assert manifest["deployment_status"] == "not_certified"


def test_deployment_contract_degrades_without_task_id(tmp_path):
    """No task_id → no joint manifest: the bundle still ships inference.py and a
    deployment block flagged unavailable (never a wrong guess)."""
    project = _make_project(tmp_path, task_id=None)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "inference.py" in zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    dep = manifest["deployment"]
    assert dep["available"] is False
    assert dep["joint_names"] is None
    assert any("deployment" in w for w in manifest["warnings"])


def test_export_picks_latest_iter_by_default(tmp_path):
    project = _make_project(tmp_path, iters=[0, 3, 7])
    res = export_policy_bundle(project)
    assert res.manifest["iter_index"] == 7
    assert res.bundle_path.name.endswith("_iter7.zip")


def test_export_explicit_iter_and_out_path(tmp_path):
    project = _make_project(tmp_path, iters=[0, 3])
    out = tmp_path / "elsewhere" / "bundle.zip"
    res = export_policy_bundle(project, iter_index=3, out_path=out)
    assert res.bundle_path == out
    assert out.is_file()
    assert res.manifest["iter_index"] == 3


def test_export_missing_iter_raises(tmp_path):
    project = _make_project(tmp_path, iters=[0])
    with pytest.raises(ExportError, match="iter 5 has no checkpoint"):
        export_policy_bundle(project, iter_index=5)


def test_export_no_iters_raises(tmp_path):
    project = _make_project(tmp_path, iters=[])
    with pytest.raises(ExportError, match="no exportable iterations"):
        export_policy_bundle(project)


def test_export_prefers_iter_env_snapshot(tmp_path):
    project = _make_project(tmp_path, with_iter_env=True)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] == "iter_snapshot"
    assert not any("CURRENT spec" in w for w in res.warnings)


def test_export_falls_back_to_current_env_with_warning(tmp_path):
    project = _make_project(tmp_path, with_iter_env=False)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] == "project_current"
    assert any("CURRENT spec" in w for w in res.warnings)


def test_export_without_env_spec_at_all(tmp_path):
    project = _make_project(tmp_path, with_env_current=False)
    res = export_policy_bundle(project)
    assert res.manifest["env_spec_source"] is None
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "env_spec.json" not in zf.namelist()


def test_export_runs_root_override(tmp_path):
    """Mission stages keep their runs under .missions/<m>/stages/<s>/runs."""
    project = _make_project(tmp_path, iters=[])
    stage_runs = project / ".missions" / "m1" / "stages" / "s1" / "runs"
    it = stage_runs / "iter_2"
    it.mkdir(parents=True)
    shutil.copy(GO1_CKPT, it / "checkpoint.pt")
    (it / "reward_spec.json").write_text(json.dumps({"version": "v1"}))
    res = export_policy_bundle(project, runs_root=stage_runs)
    assert res.manifest["iter_index"] == 2


# ── network reconstruction correctness ─────────────────────────────────────

def test_torchscript_matches_hand_built_reference(tmp_path):
    """The exported policy must be numerically identical to the checkpoint
    weights run through the known Go1 architecture (48→512→256→128→12, ELU)."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        zf.extractall(tmp_path / "unpacked")

    ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    ref = nn.Sequential(
        nn.Linear(48, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12))
    ref.load_state_dict({
        k.removeprefix("mlp."): v
        for k, v in ckpt["actor_state_dict"].items()
        if k.startswith("mlp.")})
    obs = torch.randn(8, 48, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(ts(obs), ref(obs))


def test_onnx_structurally_valid(tmp_path):
    onnx = pytest.importorskip("onnx")
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        zf.extractall(tmp_path / "unpacked")
    model = onnx.load(str(tmp_path / "unpacked" / "policy.onnx"))
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "obs"
    assert model.graph.output[0].name == "action"


# ── degraded checkpoints: raw-only fallback, never fatal ───────────────────

def test_corrupt_checkpoint_still_bundles_raw(tmp_path):
    project = _make_project(tmp_path, checkpoint=None)
    it = project / "runs" / "iter_0"
    (it / "checkpoint.pt").write_bytes(b"not a torch file at all")
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
    assert "checkpoint.pt" in names
    assert "policy.onnx" not in names
    assert any("unreadable" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_non_mlp_actor_still_bundles_raw(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    it = project / "runs" / "iter_0"
    torch.save(
        {"actor_state_dict": {"rnn.weight_ih_l0": torch.zeros(4, 4)}},
        it / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("unsupported architecture" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "checkpoint.pt" in zf.namelist()


def test_missing_task_id_assumes_elu(tmp_path):
    project = _make_project(tmp_path, task_id=None)
    res = export_policy_bundle(project)
    net = res.manifest["network"]
    assert net["activation"] == "elu"
    assert net["activation_assumed"] is True
    assert any("assuming elu" in w for w in res.warnings)


def test_missing_reward_source_warns(tmp_path):
    project = _make_project(tmp_path)
    (project / "rewards" / "v1.py").unlink()
    res = export_policy_bundle(project)
    assert any("not found under rewards/" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
    assert "reward/reward_spec.json" in names
    assert "reward/v1.py" not in names


def test_deploy_md_mentions_dims_and_recipes(tmp_path):
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    with zipfile.ZipFile(res.bundle_path) as zf:
        md = zf.read("DEPLOY.md").decode("utf-8")
    assert "onnxruntime" in md
    assert "torch.jit.load" in md
    assert "mean action" in md
    assert "48" in md and "12" in md
    assert "NOT DEPLOYMENT CERTIFIED" in md
    assert "do not prove" in md


def test_minimal_self_digested_qualified_receipt_fails_closed(tmp_path):
    project = _make_project(tmp_path)
    checkpoint = project / "runs" / "iter_0" / "checkpoint.pt"
    forged = {
        "schema": 1,
        "status": "qualified",
        "purpose": "reproducibility",
        "iter_index": 0,
        "checks": {
            "origin_lineage": {
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            },
        },
        "blockers": [],
    }
    forged["authority_digest"] = hashlib.sha256(json.dumps(
        forged,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()).hexdigest()

    result = export_policy_bundle(project, authority_receipt=forged)

    assert result.manifest["deployment_status"] == "not_certified"
    assert "malformed" in result.manifest["deployment_authority"]["blockers"][0]


# ── silently-wrong-policy guards (review findings 1-4) ─────────────────────

def test_non_dict_payload_bundles_raw(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    torch.save(torch.zeros(3), project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("not the rsl_rl dict format" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_known_obs_normalizer_is_baked_into_export(tmp_path):
    """rsl_rl EmpiricalNormalization state embedded in the actor sd must be
    BAKED into the exported graph: TS(raw obs) == MLP(normalizer(raw obs))
    computed with the real rsl_rl module."""
    torch = pytest.importorskip("torch")
    rsl_norm = pytest.importorskip("rsl_rl.modules.normalization")
    import torch.nn as nn

    project = _make_project(tmp_path, checkpoint=None)
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    gen = torch.Generator().manual_seed(7)
    mean = torch.randn(48, generator=gen)
    std = torch.rand(48, generator=gen) + 0.5
    ckpt["actor_state_dict"] = dict(ckpt["actor_state_dict"])
    ckpt["actor_state_dict"]["obs_normalizer._mean"] = mean.reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer._std"] = std.reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer._var"] = (std ** 2).reshape(1, -1)
    ckpt["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(100)
    torch.save(ckpt, project / "runs" / "iter_0" / "checkpoint.pt")

    res = export_policy_bundle(project)
    assert res.manifest["network"]["obs_normalization_baked"] is True
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "unpacked")
    assert {"policy.onnx", "policy_ts.pt"} <= names

    # Reference: REAL rsl_rl normalizer + hand-built MLP.
    ref_norm = rsl_norm.EmpiricalNormalization(48)
    ref_norm._mean = mean.reshape(1, -1).clone()
    ref_norm._std = std.reshape(1, -1).clone()
    ref_mlp = nn.Sequential(
        nn.Linear(48, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12))
    ref_mlp.load_state_dict({
        k.removeprefix("mlp."): v
        for k, v in ckpt["actor_state_dict"].items()
        if k.startswith("mlp.")})
    ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
    obs = torch.randn(8, 48, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(ts(obs), ref_mlp(ref_norm(obs)), atol=1e-6)


@pytest.mark.parametrize("bad_norm", [
    True,                       # truthy non-dict
    {},                         # present but empty
    {"_mean": [0.0] * 48, "_std": [1.0] * 48},  # right keys, non-tensor values
])
def test_malformed_obs_norm_state_refuses(tmp_path, bad_norm):
    """A present-but-unusable obs_norm_state_dict must refuse the network
    export — falling through to a bare un-normalized MLP is silently wrong."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    ckpt["obs_norm_state_dict"] = bad_norm
    torch.save(ckpt, project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert res.manifest["network"] == {"obs_normalization": True}
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "policy.onnx" not in zf.namelist()
        assert "policy_ts.pt" not in zf.namelist()


def test_unknown_normalizer_refuses_network_export(tmp_path):
    """Normalizer-ish state the exporter doesn't recognise → raw-only, never
    a bare-MLP guess."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    ckpt = torch.load(GO1_CKPT, map_location="cpu", weights_only=False)
    ckpt["critic_obs_norm_stats"] = {"mean": torch.zeros(48)}
    torch.save(ckpt, project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert res.manifest["network"] == {"obs_normalization": True}
    assert any("normalizer" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        assert "policy.onnx" not in zf.namelist()
        assert "policy_ts.pt" not in zf.namelist()


def test_unknown_activation_refuses_network_export(tmp_path, monkeypatch):
    """An activation with no known torch equivalent must bail to raw-only —
    building with a guessed module ships a silently wrong policy."""
    import sculptor.export as ex

    monkeypatch.setattr(
        ex, "_resolve_activation", lambda project, warnings: ("crelu", False))
    project = _make_project(tmp_path)
    res = export_policy_bundle(project)
    assert any("no known torch equivalent" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_parameterized_non_linear_module_bails(tmp_path):
    """LayerNorm-style 1-D params at an mlp index must not be silently
    replaced by an activation."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.1.weight": torch.zeros(8),   # LayerNorm-ish
        "mlp.1.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 8), "mlp.2.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("non-Linear module at mlp.1" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_unchained_dims_bail(tmp_path):
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 16), "mlp.2.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("dims don't chain" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_large_index_gap_bails(tmp_path):
    """A gap > 2 would stack multiple activations (ELU∘ELU ≠ ELU) where the
    original had dropout/identity modules — refuse instead."""
    torch = pytest.importorskip("torch")
    project = _make_project(tmp_path, checkpoint=None)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.3.weight": torch.zeros(2, 8), "mlp.3.bias": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd},
               project / "runs" / "iter_0" / "checkpoint.pt")
    res = export_policy_bundle(project)
    assert any("index gap" in w for w in res.warnings)
    assert res.manifest["network"] == {}


def test_traversal_reward_version_not_bundled(tmp_path):
    """A hostile reward_spec.json version must not become a path component."""
    project = _make_project(tmp_path)
    it = project / "runs" / "iter_0"
    secret = tmp_path / "secret.py"
    secret.write_text("SECRET = 1\n")
    (it / "reward_spec.json").write_text(
        json.dumps({"version": "../../secret"}))
    res = export_policy_bundle(project)
    assert any("not v<n>" in w for w in res.warnings)
    with zipfile.ZipFile(res.bundle_path) as zf:
        for name in zf.namelist():
            assert "secret" not in name
            if name.startswith("reward/") and name.endswith(".py"):
                raise AssertionError("no reward source should be bundled")


# ── sb3 path (offline; uses the hopper example checkpoint if present) ──────

HOPPER_CKPT = (
    Path(__file__).parent.parent
    / "examples" / "hopper" / "runs" / "iter_1" / "checkpoint.zip")


@pytest.mark.skipif(not HOPPER_CKPT.is_file(), reason="hopper example absent")
def test_sb3_bundle_exports_and_matches_predict(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")

    project = _make_project(tmp_path, checkpoint=None, task_id=None)
    it = project / "runs" / "iter_0"
    shutil.copy(HOPPER_CKPT, it / "checkpoint.zip")
    res = export_policy_bundle(project)
    assert res.manifest["checkpoint"]["format"] == "sb3"
    with zipfile.ZipFile(res.bundle_path) as zf:
        names = set(zf.namelist())
        zf.extractall(tmp_path / "unpacked")
    assert "checkpoint.zip" in names
    if "policy_ts.pt" in names:  # best-effort export succeeded
        model = sb3.PPO.load(str(HOPPER_CKPT), device="cpu")
        obs = np.random.RandomState(0).randn(
            1, model.observation_space.shape[0]).astype(np.float32)
        ref, _ = model.predict(obs, deterministic=True)
        ts = torch.jit.load(str(tmp_path / "unpacked" / "policy_ts.pt")).eval()
        with torch.no_grad():
            out = ts(torch.from_numpy(obs)).numpy()
        assert np.abs(ref - out).max() < 1e-6
