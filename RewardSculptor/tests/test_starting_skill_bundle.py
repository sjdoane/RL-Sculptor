from __future__ import annotations

import hashlib
import json
import pickle
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from filelock import FileLock
from safetensors.torch import save_file

import sculptor.skill_bundle as skill_bundle
from sculptor.compatibility_provenance import (
    LEGACY_RECONSTRUCTED,
    ORIGIN_CONTRACT_MEMBER,
    ORIGIN_PERSISTED,
    CompatibilityProvenanceError,
    build_launch_acknowledgement_receipt,
    build_origin_persisted_provenance,
    provenance_fingerprint,
)
from sculptor.policy_contract import contract_fingerprint
from sculptor.reference import save_clip
from sculptor.refs import library as reference_library
from sculptor.skill_bundle import (
    BUNDLE_KIND,
    DEPLOYMENT_BUNDLE_KIND,
    ImportTarget,
    SkillBundleError,
    StartingSkillBundleImporter,
    compatibility_for,
)
from sculptor.skill_library import SkillLibrary, SkillLibraryError


def _contract() -> dict:
    return {
        "schema": 1,
        "identity": {"adapter_class": "A", "task_id": "T"},
        "joints": {"ordered_names": ["j0", "j1"]},
        "observations": {
            "ordered_terms": [{"name": "q", "source": "q", "shape": [4]}],
            "shape": [4],
            "critic_ordered_terms": [
                {"name": "q", "source": "q", "shape": [4]}
            ],
            "critic_shape": [4],
        },
        "actions": {
            "ordered_names": ["j0", "j1"], "term_names": ["joint_pos"],
            "shape": [2],
        },
        "policy": {
            "actor": {"class_name": "MLP", "hidden_dims": [8], "activation": "elu", "recurrent": {"type": None, "hidden_dim": 0, "num_layers": 0}},
            "critic": {"class_name": "MLP", "hidden_dims": [8], "activation": "elu", "recurrent": {"type": None, "hidden_dim": 0, "num_layers": 0}},
            "normalizer": {
                "present": False,
                "actor_present": False,
                "critic_present": False,
                "actor_shape": None,
                "critic_shape": None,
            },
        },
        "timing": {"sim_timestep_s": 0.002, "decimation": 10, "control_dt_s": 0.02},
        "versions": {"torch": "2.11", "mjlab": "1.3.0", "rsl_rl": "5.0.1", "adapter": "0.1.0"},
    }


def _safe_tensors(*, critic: bool = True) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "actor_state_dict::mlp.0.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "actor_state_dict::mlp.0.bias": torch.zeros(8),
        "actor_state_dict::mlp.2.weight": torch.ones(2, 8),
        "actor_state_dict::mlp.2.bias": torch.zeros(2),
        "actor_state_dict::distribution.std_param": torch.ones(2),
    }
    if critic:
        tensors.update({
            "critic_state_dict::mlp.0.weight": torch.ones(8, 4),
            "critic_state_dict::mlp.0.bias": torch.zeros(8),
            "critic_state_dict::mlp.2.weight": torch.ones(1, 8),
            "critic_state_dict::mlp.2.bias": torch.zeros(1),
        })
    return tensors


def _safe_weights(
    path: Path, *, tensors: dict[str, torch.Tensor] | None = None,
) -> None:
    save_file(
        tensors or _safe_tensors(),
        str(path), metadata={"format": "reward-sculptor-rsl-rl-v1"},
    )


def _bundle(
    tmp_path: Path,
    *,
    members: dict[str, bytes] | None = None,
    bad_descriptor: bool = False,
    bad_descriptor_member: str = "policy/weights.safetensors",
    tensors: dict[str, torch.Tensor] | None = None,
    contract: dict | None = None,
) -> Path:
    weights = tmp_path / "weights.safetensors"
    _safe_weights(weights, tensors=tensors)
    contract = contract or _contract()
    contract_bytes = json.dumps(contract, sort_keys=True).encode("utf-8")
    payloads = {
        "policy/weights.safetensors": weights.read_bytes(),
        ORIGIN_CONTRACT_MEMBER: contract_bytes,
        **(members or {}),
    }
    contract_provenance = build_origin_persisted_provenance(
        contract_bytes=contract_bytes,
        policy_roles=["actor", "critic"],
    )
    files = []
    for name, body in payloads.items():
        digest = hashlib.sha256(body).hexdigest()
        if bad_descriptor and name == bad_descriptor_member:
            digest = "0" * 64
        files.append({"path": name, "sha256": digest, "bytes": len(body)})
    manifest = {
        "schema_version": 2,
        "kind": BUNDLE_KIND,
        "project": "source",
        "iter_index": 4,
        "starting_skill": {
            "name": "G1 seed",
            "weights_file": "policy/weights.safetensors",
            "policy_roles": ["actor", "critic"],
            "adapter_class": "A",
            "task_id": "T",
            "robot_slug": "g1",
        },
        "deployment": {"task_id": "T"},
        "compatibility_contract": contract,
        "compatibility_contract_digest": contract_fingerprint(contract),
        "compatibility_contract_provenance": contract_provenance,
        "compatibility_contract_provenance_digest": provenance_fingerprint(
            contract_provenance
        ),
        "files": files,
    }
    path = tmp_path / "skill.rskill"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, body in payloads.items():
            archive.writestr(name, body)
    return path


def _reference_payloads(
    tmp_path: Path, *, robot: str = "g1", clip_id: str = "complex_seed",
) -> dict[str, bytes]:
    n = 24
    fps = 30.0
    t = np.arange(n, dtype=np.float64) / fps
    clip = {
        "root_pos_z": np.full(n, 0.78),
        "root_pos_xy": np.stack([0.25 * t, np.zeros_like(t)], axis=1),
        "fps": fps,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["j0", "j1"],
        "meta": {"source": "unit-test:starting-skill"},
    }
    clip_path = save_clip(tmp_path / "source-reference.npz", clip)
    provenance = reference_library.make_provenance(
        clip_id=clip_id,
        robot=robot,
        source={"kind": "unit-test"},
        license="CC0",
        attribution="unit test",
        content_sha256_=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        fps_source=fps,
        text="complex motion seed",
    )
    return {
        "motion/clip.npz": clip_path.read_bytes(),
        "motion/provenance.json": json.dumps(provenance).encode("utf-8"),
    }


def _reference_only_bundle(tmp_path: Path, payloads: dict[str, bytes]) -> Path:
    files = [
        {
            "path": name,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
        for name, body in payloads.items()
    ]
    manifest = {
        "schema_version": 2,
        "kind": BUNDLE_KIND,
        "project": "reference-only source",
        "starting_skill": {"name": "Complex reference", "robot_slug": "g1"},
        "files": files,
    }
    path = tmp_path / "reference-only.rskill"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, body in payloads.items():
            archive.writestr(name, body)
    return path


def _read_archive_manifest(path: Path) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(path, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
    return manifest, members


def _rewrite_archive_manifest(
    path: Path, manifest: dict | str | bytes,
) -> None:
    _, members = _read_archive_manifest(path)
    if isinstance(manifest, dict):
        manifest = json.dumps(manifest)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", manifest)
        for name, body in members.items():
            output.writestr(name, body)


def _assert_no_import_residue(
    *, library: SkillLibrary, reference_root: Path, kg_path: Path,
) -> None:
    assert reference_library.references_root() == reference_root
    assert list(library) == []
    assert not reference_library.clip_dir("g1", "complex_seed").exists()
    assert not reference_library.index_path().exists()
    assert not kg_path.exists()


def _target() -> ImportTarget:
    return ImportTarget(
        adapter_class="A", task_id="T", robot_slug="g1",
        compatibility_contract=_contract(),
    )


def _target_for(contract: dict) -> ImportTarget:
    return ImportTarget(
        adapter_class="A", task_id="T", robot_slug="g1",
        compatibility_contract=contract,
    )


def _normalizer_tensors(
    role: str, width: int,
) -> dict[str, torch.Tensor]:
    group = f"{role}_obs_normalizer_state_dict"
    return {
        f"{group}::_mean": torch.zeros(1, width),
        f"{group}::_var": torch.ones(1, width),
        f"{group}::_std": torch.ones(1, width),
        f"{group}::count": torch.zeros((), dtype=torch.int64),
    }


def test_safe_bundle_round_trip(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    imported = StartingSkillBundleImporter(library).import_archive(
        _bundle(tmp_path), target=_target(),
    )
    record = imported.record
    assert imported.receipt["compatible"] is True
    assert imported.receipt["selectable"] is True
    assert imported.receipt["training_authorized"] is False
    authorization = imported.receipt["authorization"]
    assert authorization["status"] == "candidate"
    assert authorization["receipt_scope"] == "structural_selectability_only"
    assert authorization["training_authorized"] is False
    assert any(
        "warm_start_loaded" in gate
        for gate in authorization["mode_gates"]["actor_only"]
    )
    assert imported.receipt["trust"]["source_format"] == "safetensors"
    assert imported.receipt["components"]["world"]["bytes_retained"] is False
    assert imported.receipt["components"]["world"]["activatable"] is False
    assert record.initialization_modes == ["actor_only", "actor_critic"]
    checkpoint = library.checkpoint_path_for(record)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload) >= {"actor_state_dict", "critic_state_dict"}
    assert record.compatibility_contract_digest == contract_fingerprint(_contract())


def test_trainable_bundle_without_contract_origin_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    archive = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename not in {"manifest.json", ORIGIN_CONTRACT_MEMBER}
        }
    manifest.pop("compatibility_contract_provenance")
    manifest.pop("compatibility_contract_provenance_digest")
    manifest["files"] = [
        row for row in manifest["files"]
        if row["path"] != ORIGIN_CONTRACT_MEMBER
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            output.writestr(name, body)

    with pytest.raises(SkillBundleError) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(archive, target=_target())
    assert rejected.value.code == "compatibility_contract_provenance_required"


def test_origin_contract_provenance_is_rederived_from_retained_bytes(
    tmp_path: Path,
) -> None:
    archive = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
    provenance = manifest["compatibility_contract_provenance"]
    provenance["evidence"]["origin_policy_contract"]["sha256"] = "0" * 64
    manifest["compatibility_contract_provenance_digest"] = (
        provenance_fingerprint(provenance)
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            output.writestr(name, body)

    with pytest.raises(
        SkillBundleError, match="provenance descriptor mismatch",
    ) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(archive, target=_target())
    assert rejected.value.code == "compatibility_contract_provenance_invalid"


def test_origin_contract_evidence_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    archive = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
    original = members[ORIGIN_CONTRACT_MEMBER]
    duplicate = b'{"schema":1,' + original[1:]
    members[ORIGIN_CONTRACT_MEMBER] = duplicate
    evidence = manifest["compatibility_contract_provenance"]["evidence"][
        "origin_policy_contract"
    ]
    evidence.update({
        "sha256": hashlib.sha256(duplicate).hexdigest(),
        "bytes": len(duplicate),
    })
    manifest["compatibility_contract_provenance_digest"] = provenance_fingerprint(
        manifest["compatibility_contract_provenance"]
    )
    for descriptor in manifest["files"]:
        if descriptor["path"] == ORIGIN_CONTRACT_MEMBER:
            descriptor.update({
                "sha256": hashlib.sha256(duplicate).hexdigest(),
                "bytes": len(duplicate),
            })
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            output.writestr(name, body)

    with pytest.raises(
        SkillBundleError, match="duplicate JSON key 'schema'",
    ) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(archive, target=_target())
    assert rejected.value.code == "compatibility_contract_provenance_invalid"


def test_legacy_reconstructed_launch_requires_scoped_acknowledgement() -> None:
    digest = "a" * 64
    with pytest.raises(
        CompatibilityProvenanceError, match="explicit acknowledgement",
    ):
        build_launch_acknowledgement_receipt(
            status=LEGACY_RECONSTRUCTED,
            provenance_digest=digest,
            acknowledged=False,
            initialization_mode="actor_critic",
        )

    receipt = build_launch_acknowledgement_receipt(
        status=LEGACY_RECONSTRUCTED,
        provenance_digest=digest,
        acknowledged=True,
        initialization_mode="actor_critic",
    )
    assert receipt == {
        "schema": 1,
        "status": LEGACY_RECONSTRUCTED,
        "provenance_digest": digest,
        "acknowledged": True,
        "initialization_mode": "actor_critic",
        "optimizer_resume": False,
        "exact_resume": False,
    }

    with pytest.raises(
        CompatibilityProvenanceError, match="origin-persisted",
    ):
        build_launch_acknowledgement_receipt(
            status=ORIGIN_PERSISTED,
            provenance_digest=digest,
            acknowledged=True,
            initialization_mode="actor_only",
        )
    with pytest.raises(
        CompatibilityProvenanceError, match="actor/critic initialization only",
    ):
        build_launch_acknowledgement_receipt(
            status=LEGACY_RECONSTRUCTED,
            provenance_digest=digest,
            acknowledged=True,
            initialization_mode="full_resume",
        )


def test_trainable_bundle_cannot_inherit_robot_identity_from_target(
    tmp_path: Path,
) -> None:
    archive = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
    manifest["starting_skill"].pop("robot_slug")
    manifest["deployment"].pop("robot_slug", None)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            output.writestr(name, body)

    with pytest.raises(
        SkillBundleError, match="target-project identity cannot substitute",
    ) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(archive, target=_target())
    assert rejected.value.code == "robot_identity_required"


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("task_id", "task_identity_required"),
        ("adapter_class", "adapter_identity_required"),
    ],
)
def test_trainable_bundle_cannot_inherit_policy_interface_from_target(
    tmp_path: Path, field: str, code: str,
) -> None:
    archive = _bundle(tmp_path)
    with zipfile.ZipFile(archive, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
    manifest["starting_skill"].pop(field)
    if field == "task_id":
        manifest["deployment"].pop("task_id", None)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            output.writestr(name, body)

    with pytest.raises(SkillBundleError) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(archive, target=_target())
    assert rejected.value.code == code
    assert "target-project identity cannot substitute" in str(rejected.value)


def test_extra_actor_tensor_key_is_rejected(tmp_path: Path) -> None:
    tensors = _safe_tensors()
    tensors["actor_state_dict::unused_projection.weight"] = torch.zeros(2, 2)
    with pytest.raises(SkillBundleError, match="tensor inventory mismatch.*extra"):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            _bundle(tmp_path, tensors=tensors), target=_target(),
        )


def test_missing_stochastic_distribution_parameter_is_rejected(
    tmp_path: Path,
) -> None:
    tensors = _safe_tensors()
    tensors.pop("actor_state_dict::distribution.std_param")
    with pytest.raises(
        SkillBundleError,
        match=r"tensor inventory mismatch.*distribution\.std_param",
    ):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            _bundle(tmp_path, tensors=tensors), target=_target(),
        )


def test_wrong_policy_tensor_dtype_is_rejected(tmp_path: Path) -> None:
    tensors = _safe_tensors()
    tensors["actor_state_dict::mlp.0.weight"] = tensors[
        "actor_state_dict::mlp.0.weight"
    ].to(torch.float64)
    with pytest.raises(SkillBundleError, match="dtype torch.float64"):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            _bundle(tmp_path, tensors=tensors), target=_target(),
        )


def test_wrong_normalizer_shape_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    contract["policy"]["normalizer"].update({
        "present": True,
        "actor_present": True,
        "actor_shape": [4],
    })
    tensors = {**_safe_tensors(), **_normalizer_tensors("actor", 4)}
    tensors["actor_obs_normalizer_state_dict::_mean"] = torch.zeros(4)
    with pytest.raises(SkillBundleError, match="_mean shape .*supported shape"):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            _bundle(tmp_path, tensors=tensors, contract=contract),
            target=_target_for(contract),
        )


def test_incomplete_normalizer_inventory_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    contract["policy"]["normalizer"].update({
        "present": True,
        "actor_present": True,
        "actor_shape": [4],
    })
    tensors = {**_safe_tensors(), **_normalizer_tensors("actor", 4)}
    tensors.pop("actor_obs_normalizer_state_dict::_var")
    with pytest.raises(
        SkillBundleError, match=r"tensor inventory mismatch.*_var",
    ):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            _bundle(tmp_path, tensors=tensors, contract=contract),
            target=_target_for(contract),
        )


def test_exact_separate_normalizer_is_materialized_for_native_rsl_rl(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["policy"]["normalizer"].update({
        "present": True,
        "actor_present": True,
        "actor_shape": [4],
    })
    tensors = {**_safe_tensors(), **_normalizer_tensors("actor", 4)}
    library = SkillLibrary(tmp_path / "library")
    record = StartingSkillBundleImporter(library).import_archive(
        _bundle(tmp_path, tensors=tensors, contract=contract),
        target=_target_for(contract),
    ).record
    checkpoint = torch.load(
        library.checkpoint_path_for(record), map_location="cpu", weights_only=True,
    )
    assert "actor_obs_normalizer_state_dict" not in checkpoint
    assert set(checkpoint["actor_state_dict"]) >= {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
    }


def test_descriptor_hash_mismatch_rejected(tmp_path: Path) -> None:
    importer = StartingSkillBundleImporter(SkillLibrary(tmp_path / "library"))
    with pytest.raises(SkillBundleError, match="descriptor digest/size mismatch"):
        importer.import_archive(
            _bundle(tmp_path, bad_descriptor=True), target=_target(),
        )


def test_inert_world_descriptor_is_also_attested(tmp_path: Path) -> None:
    importer = StartingSkillBundleImporter(SkillLibrary(tmp_path / "library"))
    with pytest.raises(SkillBundleError, match="descriptor digest/size mismatch"):
        importer.import_archive(
            _bundle(
                tmp_path,
                members={"world/manifest.json": b"{}"},
                bad_descriptor=True,
                bad_descriptor_member="world/manifest.json",
            ),
            target=_target(),
        )


def test_source_world_declaration_must_be_json_data(tmp_path: Path) -> None:
    with pytest.raises(
        SkillBundleError, match="world/manifest.json is not valid UTF-8 JSON",
    ):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(
            _bundle(
                tmp_path,
                members={"world/manifest.json": b"not-json"},
            ),
            target=_target(),
        )


@pytest.mark.parametrize(
    ("suffix", "match"),
    [
        (', "kind": "reward-sculptor-starting-skill"}', "duplicate JSON key"),
        (', "strict_number": NaN}', "non-finite JSON number"),
    ],
)
def test_manifest_strict_json_failure_leaves_no_import_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    match: str,
) -> None:
    reference_root = tmp_path / "references"
    kg_path = tmp_path / "kg.sqlite"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    library = SkillLibrary(tmp_path / "skills")
    archive = _reference_only_bundle(tmp_path, _reference_payloads(tmp_path))
    manifest, _ = _read_archive_manifest(archive)
    raw = json.dumps(manifest)[:-1] + suffix
    _rewrite_archive_manifest(archive, raw)

    with pytest.raises(SkillBundleError, match=match):
        StartingSkillBundleImporter(library).import_archive(
            archive, target=_target(),
        )

    _assert_no_import_residue(
        library=library, reference_root=reference_root, kg_path=kg_path,
    )


@pytest.mark.parametrize(
    ("member_name", "payload", "match"),
    [
        (
            "world/manifest.json",
            b'{"terrain": "flat", "terrain": "rough"}',
            "duplicate JSON key",
        ),
        (
            "controller/controller.json",
            b'{"kind": "reference_tracker", "kind": "residual_policy"}',
            "duplicate JSON key",
        ),
        (
            "world/manifest.json",
            b'{"gravity": Infinity}',
            "non-finite JSON number",
        ),
        (
            "world/manifest.json",
            b'{"overflow": 1e999}',
            "non-finite JSON number",
        ),
    ],
)
def test_all_declarative_json_members_use_strict_parser(
    tmp_path: Path, member_name: str, payload: bytes, match: str,
) -> None:
    library = SkillLibrary(tmp_path / "skills")
    with pytest.raises(SkillBundleError, match=match):
        StartingSkillBundleImporter(library).import_archive(
            _bundle(tmp_path, members={member_name: payload}),
            target=_target(),
        )
    assert list(library) == []


def test_reference_provenance_duplicate_key_leaves_no_import_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "references"
    kg_path = tmp_path / "kg.sqlite"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    payloads = _reference_payloads(tmp_path)
    raw = payloads["motion/provenance.json"].decode("utf-8")
    payloads["motion/provenance.json"] = (
        raw[:-1] + ', "robot": "g1"}'
    ).encode("utf-8")
    library = SkillLibrary(tmp_path / "skills")

    with pytest.raises(SkillBundleError, match="duplicate JSON key"):
        StartingSkillBundleImporter(library).import_archive(
            _reference_only_bundle(tmp_path, payloads), target=_target(),
        )

    _assert_no_import_residue(
        library=library, reference_root=reference_root, kg_path=kg_path,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("warnings", "not-a-list", "warnings must be a list of strings"),
        ("iter_index", 1.5, "iter_index must be an integer"),
    ],
)
def test_late_manifest_validation_precedes_reference_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
) -> None:
    reference_root = tmp_path / "references"
    kg_path = tmp_path / "kg.sqlite"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    library = SkillLibrary(tmp_path / "skills")
    archive = _reference_only_bundle(tmp_path, _reference_payloads(tmp_path))
    manifest, _ = _read_archive_manifest(archive)
    manifest[field] = value
    _rewrite_archive_manifest(archive, manifest)

    with pytest.raises(SkillBundleError, match=match):
        StartingSkillBundleImporter(library).import_archive(
            archive, target=_target(),
        )

    _assert_no_import_residue(
        library=library, reference_root=reference_root, kg_path=kg_path,
    )


def test_publish_failure_rolls_back_reference_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "references"
    kg_path = tmp_path / "kg.sqlite"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    library = SkillLibrary(tmp_path / "skills")
    archive = _reference_only_bundle(tmp_path, _reference_payloads(tmp_path))

    def fail_publish(**_kwargs: object) -> None:
        assert reference_library.clip_dir("g1", "complex_seed").is_dir()
        assert reference_library.index_path().is_file()
        raise SkillLibraryError("synthetic publish failure")

    monkeypatch.setattr(library, "publish_imported_checkpoint", fail_publish)
    with pytest.raises(SkillLibraryError, match="synthetic publish failure"):
        StartingSkillBundleImporter(library).import_archive(
            archive, target=_target(),
        )

    _assert_no_import_residue(
        library=library, reference_root=reference_root, kg_path=kg_path,
    )


def test_failed_reference_import_cannot_clobber_concurrent_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transaction lock covers snapshot, install, publish, and rollback."""
    reference_root = tmp_path / "references"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_archive = _reference_only_bundle(
        first_dir,
        _reference_payloads(first_dir, clip_id="concurrent_first"),
    )
    second_archive = _reference_only_bundle(
        second_dir,
        _reference_payloads(second_dir, clip_id="concurrent_second"),
    )
    shared_library_root = tmp_path / "skills"
    first_library = SkillLibrary(shared_library_root)
    second_library = SkillLibrary(shared_library_root)
    first_installed = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []

    def fail_first_publish(**_kwargs: object) -> None:
        assert reference_library.clip_dir(
            "g1", "concurrent_first",
        ).is_dir()
        first_installed.set()
        assert release_first.wait(timeout=5)
        raise SkillLibraryError("synthetic concurrent publish failure")

    monkeypatch.setattr(
        first_library, "publish_imported_checkpoint", fail_first_publish,
    )

    def import_first() -> None:
        try:
            StartingSkillBundleImporter(first_library).import_archive(
                first_archive, target=_target(),
            )
        except BaseException as exc:
            first_errors.append(exc)

    def import_second() -> None:
        second_started.set()
        try:
            StartingSkillBundleImporter(second_library).import_archive(
                second_archive, target=_target(),
            )
        except BaseException as exc:
            second_errors.append(exc)
        finally:
            second_finished.set()

    first_thread = threading.Thread(target=import_first)
    second_thread = threading.Thread(target=import_second)
    first_thread.start()
    assert first_installed.wait(timeout=5)
    second_thread.start()
    assert second_started.wait(timeout=5)

    # The second importer cannot mutate the reference library while the first
    # transaction still owns the snapshot it may need to restore.
    assert not second_finished.wait(timeout=0.2)
    assert not reference_library.clip_dir(
        "g1", "concurrent_second",
    ).exists()

    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(first_errors) == 1
    assert isinstance(first_errors[0], SkillLibraryError)
    assert "synthetic concurrent publish failure" in str(first_errors[0])
    assert second_errors == []
    assert not reference_library.clip_dir(
        "g1", "concurrent_first",
    ).exists()
    assert reference_library.clip_dir(
        "g1", "concurrent_second",
    ).is_dir()
    assert [
        (row["robot"], row["clip_id"])
        for row in reference_library.read_index()
    ] == [("g1", "concurrent_second")]


def test_reference_import_lock_timeout_is_clean_and_nonmutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setattr(
        skill_bundle, "_REFERENCE_TRANSACTION_LOCK_TIMEOUT_S", 0.05,
    )
    archive = _reference_only_bundle(
        tmp_path, _reference_payloads(tmp_path),
    )
    library = SkillLibrary(tmp_path / "skills")
    lock_path = (
        reference_root / skill_bundle._REFERENCE_TRANSACTION_LOCK_FILENAME
    )

    with FileLock(str(lock_path)):
        with pytest.raises(
            SkillBundleError, match="reference library is busy",
        ) as rejected:
            StartingSkillBundleImporter(library).import_archive(
                archive, target=_target(),
            )

    assert rejected.value.code == "reference_library_busy"
    assert list(library) == []
    assert not reference_library.clip_dir("g1", "complex_seed").exists()
    assert not reference_library.index_path().exists()


def test_manifest_identifiers_and_sha_fields_are_exactly_validated(
    tmp_path: Path,
) -> None:
    bad_robot = _bundle(tmp_path)
    manifest, _ = _read_archive_manifest(bad_robot)
    manifest["starting_skill"]["robot_slug"] = "../g1"
    _rewrite_archive_manifest(bad_robot, manifest)
    with pytest.raises(SkillBundleError, match="safe stable string") as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "robot-skills")
        ).import_archive(bad_robot, target=_target())
    assert rejected.value.code == "robot_identity_invalid"

    bad_checkpoint_sha = _bundle(tmp_path)
    manifest, _ = _read_archive_manifest(bad_checkpoint_sha)
    manifest["checkpoint"] = {"sha256": "g" * 64}
    _rewrite_archive_manifest(bad_checkpoint_sha, manifest)
    with pytest.raises(
        SkillBundleError, match="checkpoint.sha256.*hexadecimal SHA-256",
    ):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "sha-skills")
        ).import_archive(bad_checkpoint_sha, target=_target())

    bad_descriptor_sha = _bundle(tmp_path)
    manifest, _ = _read_archive_manifest(bad_descriptor_sha)
    manifest["files"][0]["sha256"] = "0" * 63
    _rewrite_archive_manifest(bad_descriptor_sha, manifest)
    with pytest.raises(
        SkillBundleError, match="descriptor sha256.*hexadecimal SHA-256",
    ):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "descriptor-skills")
        ).import_archive(bad_descriptor_sha, target=_target())


def test_reference_robot_identifier_is_not_coerced_from_non_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "references"
    kg_path = tmp_path / "kg.sqlite"
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(reference_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    payloads = _reference_payloads(tmp_path)
    provenance = json.loads(payloads["motion/provenance.json"])
    provenance["robot"] = 1
    payloads["motion/provenance.json"] = json.dumps(provenance).encode("utf-8")
    library = SkillLibrary(tmp_path / "skills")

    with pytest.raises(SkillBundleError, match="robot must be a string"):
        StartingSkillBundleImporter(library).import_archive(
            _reference_only_bundle(tmp_path, payloads), target=_target(),
        )

    _assert_no_import_residue(
        library=library, reference_root=reference_root, kg_path=kg_path,
    )


@pytest.mark.parametrize("bad_name", ["../escape", "/absolute"])
def test_path_traversal_rejected(tmp_path: Path, bad_name: str) -> None:
    path = _bundle(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(bad_name, b"x")
    with pytest.raises(SkillBundleError, match="unsafe archive member"):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            path, target=_target(),
        )


def test_case_colliding_member_rejected(tmp_path: Path) -> None:
    path = _bundle(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("POLICY/WEIGHTS.SAFETENSORS", b"x")
    with pytest.raises(SkillBundleError, match="case-colliding"):
        StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
            path, target=_target(),
        )


def test_uploaded_pickle_member_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "pickle_executed"

    class Payload:
        def __reduce__(self):
            return (Path.write_text, (marker, "executed"))

    pickle_bytes = pickle.dumps(Payload())
    path = _bundle(tmp_path, members={"checkpoint.pt": pickle_bytes})
    calls = 0

    def forbidden_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("uploaded PT reached torch.load")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(
        SkillBundleError, match="unsupported portable bundle member.*checkpoint.pt",
    ) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(path, target=_target())
    assert rejected.value.code == "unsupported_member"
    assert calls == 0
    assert not marker.exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "controller.py",
        "checkpoint.pth",
        "checkpoint.pkl",
        "policy_ts.pt",
        "policy.onnx",
        "native/controller.dll",
        "notes.txt",
    ],
)
def test_executable_serialized_and_unknown_members_are_rejected(
    tmp_path: Path, member_name: str,
) -> None:
    with pytest.raises(SkillBundleError, match="unsupported portable bundle member"):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(
            _bundle(tmp_path, members={member_name: b"untrusted bytes"}),
            target=_target(),
        )


def test_deployment_bundle_kind_cannot_be_renamed_to_rskill(
    tmp_path: Path,
) -> None:
    path = _bundle(tmp_path)
    with zipfile.ZipFile(path, "r") as source:
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "manifest.json"
        }
        manifest = json.loads(source.read("manifest.json"))
    manifest["kind"] = DEPLOYMENT_BUNDLE_KIND
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            archive.writestr(name, body)

    with pytest.raises(
        SkillBundleError, match="deployment ZIPs are not portable",
    ) as rejected:
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "library")
        ).import_archive(path, target=_target())
    assert rejected.value.code == "deployment_bundle_not_portable"


def test_checkpoint_tamper_is_detected_before_use(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    record = StartingSkillBundleImporter(library).import_archive(
        _bundle(tmp_path), target=_target(),
    ).record
    checkpoint = library.checkpoint_path_for(record)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(SkillLibraryError, match="size mismatch"):
        library.checkpoint_path_for(record)


def test_missing_contract_blocks_policy_compatibility(tmp_path: Path) -> None:
    record = StartingSkillBundleImporter(SkillLibrary(tmp_path / "library")).import_archive(
        _bundle(tmp_path), target=_target(),
    ).record
    receipt = compatibility_for(
        record,
        ImportTarget(adapter_class="A", task_id="T", robot_slug="g1"),
    )
    assert receipt["status"] == "incompatible"
    assert any("target compatibility contract" in reason for reason in receipt["reasons"])


@pytest.mark.parametrize(
    ("migration_type", "expected"),
    [
        (
            "zero_initialized_reference_clock_observation",
            "reference-clock observation extension",
        ),
        (
            "zero_initialized_observation_extensions",
            "event-phase and reference-clock observation extensions",
        ),
        (
            "zero_initialized_event_phase_observation",
            "event-phase observation extension",
        ),
    ],
)
def test_policy_migration_copy_names_the_actual_interface_extension(
    migration_type: str,
    expected: str,
) -> None:
    assert skill_bundle._policy_migration_observation_label(
        {"type": migration_type}
    ) == expected


def test_reference_clock_migration_gate_is_not_described_as_event_only(
    tmp_path: Path,
) -> None:
    record = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "library")
    ).import_archive(_bundle(tmp_path), target=_target()).record
    authorization = skill_bundle._authorization_for(
        record,
        {
            "allowed_initialization_modes": ["actor_only"],
            "reasons": [],
            "policy_contract_migration": {
                "type": "zero_initialized_reference_clock_observation",
            },
        },
    )

    gates = authorization["mode_gates"]["actor_only"]
    assert any(
        "zero-initialized reference-clock observation extension" in gate
        for gate in gates
    )
    assert all("event-phase" not in gate for gate in gates)


def test_reference_only_bundle_needs_no_policy_or_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "reference-library"))
    payloads = _reference_payloads(tmp_path)
    imported = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "skill-library")
    ).import_archive(
        _reference_only_bundle(tmp_path, payloads), target=_target(),
    )

    record = imported.record
    assert record.adapter_class == "reference_trajectory"
    assert record.task_id == "reference_trajectory"
    assert record.policy_roles == []
    assert record.initialization_modes == ["reference_only"]
    assert record.compatibility_contract is None
    assert record.checkpoint_filename == ""
    assert record.reference_sha256 == hashlib.sha256(
        payloads["motion/clip.npz"]
    ).hexdigest()
    assert imported.receipt["compatible"] is True
    assert imported.receipt["selectable"] is True
    assert imported.receipt["training_authorized"] is False
    authorization = imported.receipt["authorization"]
    reference_gates = authorization["mode_gates"]["reference_only"]
    assert any("separate Tier-D" in gate for gate in reference_gates)
    assert any("before live launch" in gate for gate in reference_gates)
    assert any(
        "re-attest" in gate and "at launch" in gate
        for gate in reference_gates
    )
    assert "separate Tier-D" in authorization["detail"]
    assert "launch only re-verifies" in authorization["detail"]
    assert imported.receipt["compatibility"]["allowed_initialization_modes"] == [
        "reference_only"
    ]
    assert imported.receipt["trust"]["status"] == "validated"
    admission = imported.receipt["components"]["reference"]["admission"]
    assert admission["status"] == "registered_candidate"
    assert admission["training_authorized"] is False
    assert "separate target-specific Tier-D" in admission["next_gate"]
    assert "before live launch" in admission["next_gate"]
    assert "launch only re-verifies" in admission["next_gate"]
    with pytest.raises(SkillLibraryError, match="no trainable checkpoint"):
        SkillLibrary(tmp_path / "skill-library").checkpoint_path_for(record)


def test_exact_reference_bundle_reimport_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "reference-library"))
    payloads = _reference_payloads(tmp_path)
    archive = _reference_only_bundle(tmp_path, payloads)
    library = SkillLibrary(tmp_path / "skill-library")
    importer = StartingSkillBundleImporter(library)

    first = importer.import_archive(archive, target=_target()).record
    from sculptor.refs import library as reference_library

    local_provenance = reference_library.read_provenance(
        "g1", "complex_seed",
    )
    local_provenance["tier"] = "D"
    local_provenance["tierD"] = {
        "tracked_at": "local-certification-overlay",
        "certificate_sha256": "f" * 64,
    }
    reference_library.write_provenance(
        "g1", "complex_seed", local_provenance,
    )
    second = importer.import_archive(archive, target=_target()).record

    assert second.skill_id == first.skill_id
    assert second.identity_digest == first.identity_digest
    assert second.created_at == first.created_at
    assert second.reference_provenance_sha256 == first.reference_provenance_sha256
    preserved = reference_library.read_provenance("g1", "complex_seed")
    assert preserved["tier"] == "D"
    assert preserved["tierD"]["tracked_at"] == "local-certification-overlay"


def test_reference_id_collision_checks_canonical_provenance_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "reference-library"))
    payloads = _reference_payloads(tmp_path)
    importer = StartingSkillBundleImporter(SkillLibrary(tmp_path / "skills"))
    importer.import_archive(
        _reference_only_bundle(tmp_path, payloads), target=_target(),
    )

    changed = dict(payloads)
    provenance = json.loads(changed["motion/provenance.json"])
    provenance["attribution"] = "different scientific provenance"
    changed["motion/provenance.json"] = json.dumps(provenance).encode("utf-8")
    with pytest.raises(
        SkillBundleError, match="canonical provenance identity differs",
    ):
        importer.import_archive(
            _reference_only_bundle(tmp_path, changed), target=_target(),
        )


def test_compound_manifest_components_produce_distinct_skill_ids(
    tmp_path: Path,
) -> None:
    library = SkillLibrary(tmp_path / "library")
    importer = StartingSkillBundleImporter(library)
    first = importer.import_archive(_bundle(tmp_path), target=_target()).record
    second = importer.import_archive(
        _bundle(tmp_path, members={"world/manifest.json": b"{}"}),
        target=_target(),
    ).record

    assert second.skill_id != first.skill_id
    assert second.identity_digest != first.identity_digest
    assert second.source_weights_sha256 == first.source_weights_sha256


def test_truncated_skill_id_collision_never_overwrites_existing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sculptor.skill_library as skill_library_module

    monkeypatch.setattr(
        skill_library_module, "derive_skill_id", lambda *_args: "a" * 12,
    )
    library = SkillLibrary(tmp_path / "library")
    importer = StartingSkillBundleImporter(library)
    first = importer.import_archive(_bundle(tmp_path), target=_target()).record
    first_metadata = (
        library.root / first.skill_id / "metadata.json"
    ).read_bytes()

    with pytest.raises(SkillLibraryError, match="immutable skill-id collision"):
        importer.import_archive(
            _bundle(tmp_path, members={"world/manifest.json": b"{}"}),
            target=_target(),
        )
    assert (
        library.root / first.skill_id / "metadata.json"
    ).read_bytes() == first_metadata


def test_reference_and_policy_admission_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "reference-library"))
    imported = StartingSkillBundleImporter(
        SkillLibrary(tmp_path / "skill-library")
    ).import_archive(
        _bundle(tmp_path, members=_reference_payloads(tmp_path)),
        target=_target(),
    )
    incompatible_contract = _contract()
    incompatible_contract["actions"] = {
        **incompatible_contract["actions"],
        "shape": [3],
    }
    result = compatibility_for(
        imported.record,
        ImportTarget(
            adapter_class="A",
            task_id="T",
            robot_slug="g1",
            compatibility_contract=incompatible_contract,
        ),
    )
    assert result["status"] == "partially_compatible"
    assert result["allowed_initialization_modes"] == ["reference_only"]
    assert result["mode_reasons"]["actor_only"]
    assert result["mode_reasons"]["reference_only"] == []


def test_reference_robot_path_traversal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "reference-library"))
    payloads = _reference_payloads(tmp_path, robot="g1")
    provenance = json.loads(payloads["motion/provenance.json"])
    provenance["robot"] = "../escape"
    payloads["motion/provenance.json"] = json.dumps(provenance).encode("utf-8")
    with pytest.raises(SkillBundleError, match="safe stable identifier"):
        StartingSkillBundleImporter(
            SkillLibrary(tmp_path / "skill-library")
        ).import_archive(
            _reference_only_bundle(tmp_path, payloads), target=_target(),
        )
