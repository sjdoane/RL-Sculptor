from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from sculptor.policy_contract import (
    build_project_policy_contract,
    build_skill_warm_start_contract_receipt,
    contract_fingerprint,
)
from sculptor.reference import save_clip
from sculptor.refs import library as reference_library
from sculptor.skill_bundle import BUNDLE_KIND
from sculptor.kg.schema import (
    ArtifactAttestation,
    PolicyArtifact,
    Relation,
    WorldArtifact,
)
from sculptor.kg.store import SculptorKG
from sculptor.skill_library import ENV_LIBRARY_ROOT, SkillLibrary


ADAPTER = "sculptor.adapters.mjlab.MjlabAdapter"
TASK = "Mjlab-Velocity-Flat-Unitree-G1"


def _project(client: TestClient) -> tuple[str, Path]:
    response = client.post(
        "/projects",
        json={"name": "Starting skill", "behavior_goal": "adapt a G1 skill"},
    )
    assert response.status_code == 201, response.text
    slug = response.json()["slug"]
    store = client.app.state.project_store
    store.set_adapter_section(
        slug, ADAPTER,
        {"task_id": TASK, "num_envs": 16, "device": "cpu"},
    )
    store.write_robot_source(slug, {
        "kind": "library",
        "library_slug": "g1",
        "library_name": "g1",
        "training_support": "ready",
    })
    detail = store.get(slug)
    assert detail is not None
    return slug, Path(detail.project_dir)


def _state_tensors(contract: dict) -> dict[str, torch.Tensor]:
    obs_dim = int(contract["observations"]["shape"][0])
    critic_obs_dim = int(contract["observations"]["critic_shape"][0])
    action_dim = int(contract["actions"]["shape"][0])
    actor_hidden = contract["policy"]["actor"]["hidden_dims"]
    critic_hidden = contract["policy"]["critic"]["hidden_dims"]
    tensors: dict[str, torch.Tensor] = {}
    for group, dims in (
        ("actor_state_dict", [obs_dim, *actor_hidden, action_dim]),
        ("critic_state_dict", [critic_obs_dim, *critic_hidden, 1]),
    ):
        for layer, (width_in, width_out) in enumerate(zip(dims, dims[1:])):
            seq = layer * 2
            tensors[f"{group}::mlp.{seq}.weight"] = torch.zeros(width_out, width_in)
            tensors[f"{group}::mlp.{seq}.bias"] = torch.zeros(width_out)
    tensors["actor_state_dict::distribution.std_param"] = torch.ones(
        action_dim
    )
    normalizer = contract["policy"]["normalizer"]
    for role, present, width in (
        ("actor", normalizer["actor_present"], obs_dim),
        ("critic", normalizer["critic_present"], critic_obs_dim),
    ):
        if present:
            group = f"{role}_obs_normalizer_state_dict"
            tensors[f"{group}::_mean"] = torch.zeros(1, width)
            tensors[f"{group}::_var"] = torch.ones(1, width)
            tensors[f"{group}::_std"] = torch.ones(1, width)
            tensors[f"{group}::count"] = torch.ones((), dtype=torch.int64)
    return tensors


def _event_contract(source: dict) -> dict:
    target = copy.deepcopy(source)
    target["schema"] = 3
    target["event_observation"] = {
        "schema": 1,
        "term_name": "authored_event_phase",
        "encoding": "one_hot",
        "ordered_phase_ids": ["route", "jump", "hold"],
    }
    phase = {
        "name": "authored_event_phase",
        "source": "authored_event_phase_observation",
        "shape": [3],
    }
    observations = target["observations"]
    observations["ordered_terms"].append(copy.deepcopy(phase))
    observations["critic_ordered_terms"].append(copy.deepcopy(phase))
    observations["shape"][0] += 3
    observations["critic_shape"][0] += 3
    normalizer = target["policy"]["normalizer"]
    if normalizer["actor_present"]:
        normalizer["actor_shape"][0] += 3
    if normalizer["critic_present"]:
        normalizer["critic_shape"][0] += 3
    return target


def _bundle(
    tmp_path: Path,
    project_dir: Path,
    *,
    contract: dict | None = None,
) -> Path:
    contract = contract or build_project_policy_contract(project_dir)
    weights = tmp_path / "weights.safetensors"
    save_file(
        _state_tensors(contract), str(weights),
        metadata={"format": "reward-sculptor-rsl-rl-v1"},
    )
    members = {
        "policy/weights.safetensors": weights.read_bytes(),
        "world/manifest.json": b'{"kind":"staged-only"}',
    }
    files = [
        {
            "path": name,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
        for name, body in members.items()
    ]
    manifest = {
        "schema_version": 2,
        "kind": BUNDLE_KIND,
        "project": "outside-lab-g1",
        "starting_skill": {
            "name": "G1 complex motion seed",
            "weights_file": "policy/weights.safetensors",
            "policy_roles": ["actor", "critic"],
            "adapter_class": ADAPTER,
            "task_id": TASK,
            "robot_slug": "g1",
        },
        "deployment": {"task_id": TASK, "robot_slug": "g1"},
        "compatibility_contract": contract,
        "compatibility_contract_digest": contract_fingerprint(contract),
        "files": files,
    }
    path = tmp_path / "g1.rskill"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def _reference_only_bundle(tmp_path: Path) -> Path:
    n = 24
    fps = 30.0
    t = np.arange(n, dtype=np.float64) / fps
    clip_path = save_clip(tmp_path / "g1-reference.npz", {
        "root_pos_z": np.full(n, 0.78),
        "root_pos_xy": np.stack([0.2 * t, np.zeros_like(t)], axis=1),
        "fps": fps,
        "joint_pos": np.zeros((n, 2)),
        "joint_names": ["j0", "j1"],
        "meta": {"source": "unit-test:reference-only"},
    })
    provenance = reference_library.make_provenance(
        clip_id="complex_seed",
        robot="g1",
        source={"kind": "unit-test"},
        license="CC0",
        attribution="unit test",
        content_sha256_=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        fps_source=fps,
        text="complex G1 motion",
    )
    members = {
        "motion/clip.npz": clip_path.read_bytes(),
        "motion/provenance.json": json.dumps(provenance).encode("utf-8"),
    }
    manifest = {
        "schema_version": 2,
        "kind": BUNDLE_KIND,
        "project": "reference-only source",
        "starting_skill": {"name": "Complex G1 reference", "robot_slug": "g1"},
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
            for name, body in members.items()
        ],
    }
    path = tmp_path / "reference-only.rskill"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def test_import_list_receipt_and_world_stays_inactive(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        )
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert receipt["compatible"] is True
    assert receipt["compatibility"]["allowed_initialization_modes"] == [
        "actor_only", "actor_critic",
    ]
    assert receipt["trust"]["status"] == "sanitized"
    assert receipt["trust"]["source_format"] == "safetensors"
    assert receipt["components"]["world"]["included"] is True
    assert (
        receipt["components"]["world"]["status"]
        == "digest_recorded_bytes_discarded"
    )
    assert receipt["components"]["world"]["bytes_retained"] is False
    assert receipt["components"]["world"]["activatable"] is False
    assert receipt["components"]["world"]["sha256"]
    assert receipt["skill"]["tensor_contract_verified"] is True
    assert receipt["skill"]["tensor_signature_sha256"]
    assert receipt["skill"]["world_bundle_sha256"]
    assert not (project_dir / "env" / "selection_current.json").exists()

    # Admission attests the uploaded safetensors identity, while the retained
    # server-owned policy is an explicit conversion derivative. Discarded
    # world bytes remain declaration metadata, never an executable world node.
    from backend.services.kg_store import project_kg_db_path

    kg_path = project_kg_db_path(project_dir)
    with SculptorKG(kg_path) as kg:
        assert len(kg.find_nodes(kind=ArtifactAttestation.kind)) == 1
        policies = kg.find_nodes(kind=PolicyArtifact.kind)
        assert len(policies) == 2
        assert {policy.sha256 for policy in policies} == {
            receipt["skill"]["source_weights_sha256"],
            receipt["skill"]["checkpoint_sha256"],
        }
        assert len(kg.find_nodes(kind=WorldArtifact.kind)) == 0
        assert kg.count_edges(Relation.ATTESTS) == 1
        assert kg.count_edges(Relation.DERIVED_FROM) == 1
        assert kg.count_edges(Relation.DECLARES_TARGET) == 1
        assert kg.count_edges(Relation.COMPATIBLE_WITH) == 1
        assert kg.count_edges(Relation.EXECUTES_IN) == 0
        attestation = kg.find_nodes(kind=ArtifactAttestation.kind)[0]
        attests_edge, source_id = kg.neighbors(
            attestation.id, relation=Relation.ATTESTS,
        )[0]
        assert attests_edge.data == {
            "role": "uploaded_source_policy",
            "retained": False,
        }
        source = kg.get_node(source_id)
        assert source.sha256 == receipt["skill"]["source_weights_sha256"]
        converted = next(
            policy for policy in policies
            if policy.sha256 == receipt["skill"]["checkpoint_sha256"]
        )
        derived_edge, derived_source_id = kg.neighbors(
            converted.id, relation=Relation.DERIVED_FROM,
        )[0]
        assert derived_source_id == source.id
        assert derived_edge.data["authority"] == "sanitized_safetensors_conversion"
        assert derived_edge.data["source_retained"] is False
        assert derived_edge.data["output_retained"] is True
        assert derived_edge.data["tensor_signature_sha256"] == receipt["skill"][
            "tensor_signature_sha256"
        ]

    # Re-importing the identical compound manifest is an idempotent replay.
    with bundle.open("rb") as handle:
        replay = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        )
    assert replay.status_code == 201, replay.text
    with SculptorKG(kg_path) as kg:
        assert len(kg.find_nodes(kind=ArtifactAttestation.kind)) == 1
        assert len(kg.find_nodes(kind=PolicyArtifact.kind)) == 2
        assert kg.count_edges(Relation.ATTESTS) == 1
        assert kg.count_edges(Relation.DERIVED_FROM) == 1
        assert kg.count_edges(Relation.COMPATIBLE_WITH) == 1

    listed = client.get(f"/projects/{slug}/starting-skills")
    assert listed.status_code == 200
    assert listed.json()["skills"][0]["skill"]["skill_id"] == receipt["skill"]["skill_id"]


def test_rejected_bundle_creates_no_import_lineage(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    response = client.post(
        f"/projects/{slug}/starting-skills",
        files={"bundle": ("tampered.rskill", b"not a zip", "application/zip")},
    )
    assert response.status_code == 400

    from backend.services.kg_store import project_kg_db_path

    with SculptorKG(project_kg_db_path(project_dir)) as kg:
        assert len(kg.find_nodes(kind=ArtifactAttestation.kind)) == 0
        assert len(kg.find_nodes(kind=PolicyArtifact.kind)) == 0
        assert kg.count_edges(Relation.ATTESTS) == 0


def test_kg_publication_failure_rolls_back_skill_reference_and_graph(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """Import/list admission is one transaction with KG attestation."""
    skills_root = tmp_path / "skills"
    references_root = tmp_path / "references"
    kg_path = tmp_path / "lineage.db"
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(skills_root))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(references_root))
    monkeypatch.setenv("RS_KG_PATH", str(kg_path))
    slug, _project_dir = _project(client)
    bundle = _reference_only_bundle(tmp_path)

    original_add_edge = SculptorKG.add_edge
    calls = 0

    def _fail_first_edge(self, edge, *, upsert=True):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected KG edge failure")
        return original_add_edge(self, edge, upsert=upsert)

    monkeypatch.setattr(SculptorKG, "add_edge", _fail_first_edge)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={
                "bundle": (
                    "reference-only.rskill", handle, "application/zip",
                )
            },
        )

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "lineage_publication_failed"
    assert list(SkillLibrary(skills_root)) == []
    assert not reference_library.clip_dir("g1", "complex_seed").exists()
    with SculptorKG(kg_path) as kg:
        assert kg.count_nodes() == 0
        assert kg.count_edges() == 0


def test_deployment_zip_filename_is_not_accepted_as_portable_upload(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("deployment.zip", handle, "application/zip")},
        )

    assert response.status_code == 400
    problem = response.json()
    assert problem["code"] == "invalid_bundle"
    assert "deployment .zip" in problem["detail"]
    assert list(SkillLibrary(tmp_path / "skills")) == []


def test_launch_rechecks_imported_checkpoint_digest(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()
    skill_id = receipt["skill"]["skill_id"]
    library = SkillLibrary()
    record = library.load(skill_id)
    assert record is not None
    checkpoint = library.checkpoint_path_for(record)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")

    response = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": skill_id,
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": "actor_only",
        },
    )
    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/starting-skill-integrity"


def test_full_resume_is_not_claimed_for_sanitized_import(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()
    response = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": "full_resume",
        },
    )
    assert response.status_code == 412
    assert response.json()["type"] == "/problems/starting-skill-mode"


@pytest.mark.parametrize("reference_only", [False, True])
def test_starting_skill_launch_requires_exact_project_robot_identity(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_only: bool,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "references"))
    slug, project_dir = _project(client)
    bundle = (
        _reference_only_bundle(tmp_path)
        if reference_only
        else _bundle(tmp_path, project_dir)
    )
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("candidate.rskill", handle, "application/zip")},
        ).json()

    metadata_path = project_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["robot_source"] = {}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    response = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "adapt the admitted starting point",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": (
                "reference_only" if reference_only else "actor_only"
            ),
        },
    )

    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/starting-skill-project-contract"
    assert "exact robot identity is missing" in response.json()["detail"]


@pytest.mark.parametrize("reference_only", [False, True])
def test_unresolved_project_robot_keeps_import_inspectable_but_blocks_selection(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_only: bool,
) -> None:
    """An unknown embodiment is an inspectable authoring state, not a target.

    Policy and motion-only bundles may still be quarantined, validated, and
    listed so a researcher can inspect their immutable receipts.  No mode may
    become selectable until the project owns a canonical robot namespace, and
    a crafted launch request is rejected by the same authority.
    """
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "references"))
    slug, project_dir = _project(client)
    bundle = (
        _reference_only_bundle(tmp_path)
        if reference_only
        else _bundle(tmp_path, project_dir)
    )
    metadata_path = project_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["robot_source"] = {}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with bundle.open("rb") as handle:
        imported = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("candidate.rskill", handle, "application/zip")},
        )

    assert imported.status_code == 201, imported.text
    receipt = imported.json()
    assert receipt["selectable"] is False
    assert receipt["compatible"] is False
    assert receipt["training_authorized"] is False
    assert receipt["authorization"]["status"] == "blocked"
    assert receipt["compatibility"]["allowed_initialization_modes"] == []
    assert receipt["compatibility"]["reason_codes"] == [
        "project_robot_unresolved"
    ]
    assert any(
        reason.startswith("project_robot_unresolved:")
        for reason in receipt["compatibility"]["reasons"]
    )

    listed = client.get(f"/projects/{slug}/starting-skills")
    assert listed.status_code == 200, listed.text
    listed_receipt = listed.json()["skills"][0]
    assert listed_receipt["skill"]["skill_id"] == receipt["skill"]["skill_id"]
    assert listed_receipt["selectable"] is False
    assert listed_receipt["compatibility"]["reason_codes"] == [
        "project_robot_unresolved"
    ]

    launched = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "inspect but do not execute this imported skill",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": (
                "reference_only" if reference_only else "actor_only"
            ),
        },
    )
    assert launched.status_code == 412, launched.text
    assert launched.json()["code"] == "project_robot_unresolved"
    assert (
        launched.json()["type"]
        == "/problems/starting-skill-project-contract"
    )


def test_reference_only_bundle_imports_without_policy_contract(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "references"))
    slug, _project_dir = _project(client)
    bundle = _reference_only_bundle(tmp_path)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("reference-only.rskill", handle, "application/zip")},
        )
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert receipt["skill"]["policy_roles"] == []
    assert receipt["skill"]["checkpoint_format"] == "none"
    assert receipt["skill"]["compatibility_contract"] is None
    assert receipt["compatibility"]["allowed_initialization_modes"] == [
        "reference_only"
    ]
    assert receipt["trust"]["status"] == "validated"


def test_imported_reference_only_tier_k_cannot_authorize_live_training(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    from backend.services import world_store

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "references"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    slug, project_dir = _project(client)
    bundle = _reference_only_bundle(tmp_path)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("reference-only.rskill", handle, "application/zip")},
        ).json()
    selection = project_dir / "env" / "selection_current.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(world_store, "validate", lambda _p: {"ok": True})

    launched = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "adapt this complex G1 reference",
            "iterations": 1,
            "dry_run": False,
            "acknowledge_blind_fitness": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": "reference_only",
        },
    )
    assert launched.status_code == 412, launched.text
    assert launched.json()["type"] == "/problems/reference-feasibility"


def test_manifest_digest_pin_rejects_stale_starting_skill_selection(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()

    response = client.post(
        f"/projects/{slug}/runs",
        json={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": "0" * 64,
            "initialization_mode": "actor_only",
        },
    )
    assert response.status_code == 412, response.text
    assert response.json()["type"] == "/problems/starting-skill-stale"
    assert response.json()["actual_manifest_digest"] == (
        receipt["skill"]["manifest_digest"]
    )


def test_worker_reloads_and_rejects_manifest_replacement_before_subprocess(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    from backend.services.job_manager import Job
    from backend.services.run_manager import run_sculpt_job

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()
    skill_id = receipt["skill"]["skill_id"]
    metadata_path = SkillLibrary().root / skill_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifest_digest"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    runner = run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": skill_id,
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "initialization_mode": "actor_only",
        },
    )
    job = Job(
        job_id="manifest_reload",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    cancel = asyncio.Event()
    with pytest.raises(
        RuntimeError,
        match="failed immutable metadata revalidation before launch",
    ):
        asyncio.run(runner(job, cancel))
    assert any(
        event.get("type") == "starting_skill_revalidation_failed"
        and event.get("reason") == "missing_or_invalid_immutable_metadata"
        for event in job.events
    )
    assert not any(
        event["type"] == "starting_skill_manifest_mismatch"
        for event in job.events
    )


def test_worker_rechecks_target_contract_after_queueing_before_subprocess(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A queued skill cannot launch against a silently changed project."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        receipt = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()

    _target, target_receipt = run_manager.resolve_starting_skill_target(
        project_dir, require_policy_contract=True,
    )
    metadata_path = project_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["robot_source"] = {
        "kind": "library", "library_slug": "go1",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    subprocess_called = False

    async def _must_not_spawn(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("subprocess crossed the target-contract boundary")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _must_not_spawn)
    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": receipt["skill"]["skill_id"],
            "expected_starting_skill_manifest_digest": (
                receipt["skill"]["manifest_digest"]
            ),
            "starting_skill_target_receipt": target_receipt,
            "initialization_mode": "actor_critic",
        },
    )
    job = Job(
        job_id="target_contract_drift",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    with pytest.raises(RuntimeError, match="project target changed"):
        asyncio.run(runner(job, asyncio.Event()))

    assert subprocess_called is False
    event = next(
        item for item in job.events
        if item.get("type") == "starting_skill_target_contract_mismatch"
    )
    assert event["expected_target"]["robot_slug"] == "g1"
    assert event["actual_target"]["robot_slug"] == "go1"
    assert event["source"] == "worker_launch"


def test_exact_schema3_skill_without_full_receipt_blocks_before_subprocess(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services import run_manager
    from backend.services.job_manager import Job

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    contract = _event_contract(build_project_policy_contract(project_dir))
    bundle = _bundle(tmp_path, project_dir, contract=contract)
    with bundle.open("rb") as handle:
        imported = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        ).json()
    record = SkillLibrary().load(imported["skill"]["skill_id"])
    assert record is not None
    target_payload = {
        "adapter_class": ADAPTER,
        "task_id": TASK,
        "robot_slug": "g1",
        "compatibility_contract": contract,
    }
    target_receipt = {
        "schema": 1,
        "adapter_class": ADAPTER,
        "task_id": TASK,
        "robot_slug": "g1",
        "policy_contract_required": True,
        "policy_contract_sha256": contract_fingerprint(contract),
    }
    monkeypatch.setattr(
        run_manager,
        "resolve_starting_skill_target",
        lambda *_args, **_kwargs: (target_payload, target_receipt),
    )
    subprocess_called = False

    async def _must_not_spawn(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("subprocess crossed the policy receipt boundary")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _must_not_spawn)
    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "extend an event-conditioned G1 policy",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": record.skill_id,
            "expected_starting_skill_manifest_digest": record.manifest_digest,
            "starting_skill_target_receipt": target_receipt,
            "initialization_mode": "actor_critic",
        },
    )
    job = Job(
        job_id="missing_exact_schema3_receipt",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    with pytest.raises(RuntimeError, match="receipt changed after"):
        asyncio.run(runner(job, asyncio.Event()))

    assert subprocess_called is False
    assert any(
        event.get("type") == "warm_start_policy_contract_failed"
        for event in job.events
    )


def test_imported_policy_reaches_exact_worker_load_and_verified_lineage(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """Positive UI-import -> worker -> observed-load -> KG integration."""
    from backend.services import run_manager
    from backend.services.job_manager import Job

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_KG_PATH", str(tmp_path / "lineage.db"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        )
    assert response.status_code == 201, response.text
    receipt = response.json()
    record = SkillLibrary().load(receipt["skill"]["skill_id"])
    assert record is not None
    checkpoint = SkillLibrary().checkpoint_path_for(record).resolve()
    target_payload, target_receipt = run_manager.resolve_starting_skill_target(
        project_dir, require_policy_contract=True,
    )
    assert target_payload["task_id"] == TASK
    policy_receipt = build_skill_warm_start_contract_receipt(
        skill_id=record.skill_id,
        manifest_digest=str(record.manifest_digest),
        checkpoint_sha256=record.checkpoint_sha256,
        tensor_signature_sha256=record.tensor_signature_sha256,
        source_contract=record.compatibility_contract or {},
        target_contract=target_payload["compatibility_contract"] or {},
        target_receipt=target_receipt,
    )

    # Import admission alone never claims the policy was loaded by a worker.
    with SculptorKG(tmp_path / "lineage.db") as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 0

    load_event = {
        "type": "warm_start_loaded",
        "source": str(checkpoint),
        "source_sha256": record.checkpoint_sha256,
        "source_sha8": record.checkpoint_sha256[:8],
        "load_cfg_keys": ["actor", "critic"],
        "source_policy_contract_sha256": policy_receipt["source"][
            "contract_sha256"
        ],
        "admitted_policy_contract_migration": policy_receipt[
            "compatibility"
        ],
    }

    class _Stdout:
        def __init__(self) -> None:
            self.lines = [
                (run_manager.EVENT_TAG + " " + json.dumps(load_event) + "\n")
                .encode("utf-8")
            ]

        async def readline(self) -> bytes:
            await asyncio.sleep(0)
            return self.lines.pop(0) if self.lines else b""

    class _Proc:
        pid = 4242
        returncode = None

        def __init__(self) -> None:
            self.stdout = _Stdout()

        async def wait(self) -> int:
            await asyncio.sleep(0.02)
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    captured: dict[str, object] = {}

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        captured["env"] = kwargs["env"]
        return _Proc()

    async def _wait_for_cancel(*args, **_kwargs):
        cancel = next(arg for arg in args if isinstance(arg, asyncio.Event))
        await cancel.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(run_manager, "_fs_watcher", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_heartbeat", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_kill_on_cancel", _wait_for_cancel)

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "evolve the imported complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": record.skill_id,
            "expected_starting_skill_manifest_digest": record.manifest_digest,
            "starting_skill_target_receipt": target_receipt,
            "warm_start_policy_contract_receipt": policy_receipt,
            "initialization_mode": "actor_critic",
        },
    )
    job = Job(
        job_id="imported_runtime_proof",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    result = asyncio.run(runner(job, asyncio.Event()))
    assert result["return_code"] == 0

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--init-policy") + 1] == str(checkpoint)
    assert cmd[cmd.index("--init-policy-mode") + 1] == "actor_critic"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SCULPTOR_STARTING_SKILL_MANIFEST_DIGEST"] == record.manifest_digest
    assert env["SCULPTOR_STARTING_SKILL_CONTRACT_DIGEST"] == (
        record.compatibility_contract_digest
    )
    assert env["SCULPTOR_STARTING_SKILL_TENSOR_SIGNATURE"] == (
        record.tensor_signature_sha256
    )
    assert env["SCULPTOR_STARTING_SKILL_CHECKPOINT_SHA256"] == (
        record.checkpoint_sha256
    )
    assert json.loads(
        env["SCULPTOR_WARM_START_POLICY_CONTRACT_RECEIPT_JSON"]
    ) == policy_receipt
    assert json.loads(env["SCULPTOR_EFFECTIVE_POLICY_CONTRACT_JSON"]) == (
        policy_receipt["target"]["contract"]
    )
    assert json.loads(env["SCULPTOR_POLICY_CONTRACT_MIGRATION_JSON"]) == (
        policy_receipt["compatibility"]
    )
    assert env["SCULPTOR_SOURCE_POLICY_CONTRACT_SHA256"] == (
        policy_receipt["source"]["contract_sha256"]
    )
    assert any(event.get("type") == "starting_skill_resolved" for event in job.events)
    assert any(event.get("type") == "warm_start_loaded" for event in job.events)

    with SculptorKG(tmp_path / "lineage.db") as kg:
        assert kg.count_edges(Relation.INITIALIZED_FROM) == 1


def test_selected_imported_policy_requires_runtime_load_receipt_even_on_rc_zero(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A clean subprocess exit cannot substitute for exact load evidence."""
    from backend.services import run_manager
    from backend.services.artifact_lineage import RunLineageSession
    from backend.services.job_manager import Job

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    slug, project_dir = _project(client)
    bundle = _bundle(tmp_path, project_dir)
    with bundle.open("rb") as handle:
        response = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        )
    assert response.status_code == 201, response.text
    record = SkillLibrary().load(response.json()["skill"]["skill_id"])
    assert record is not None
    target_payload, target_receipt = run_manager.resolve_starting_skill_target(
        project_dir, require_policy_contract=True,
    )
    policy_receipt = build_skill_warm_start_contract_receipt(
        skill_id=record.skill_id,
        manifest_digest=str(record.manifest_digest),
        checkpoint_sha256=record.checkpoint_sha256,
        tensor_signature_sha256=record.tensor_signature_sha256,
        source_contract=record.compatibility_contract or {},
        target_contract=target_payload["compatibility_contract"] or {},
        target_receipt=target_receipt,
    )

    class _Stdout:
        async def readline(self) -> bytes:
            await asyncio.sleep(0)
            return b""

    class _Proc:
        pid = 4243
        returncode = None
        stdout = _Stdout()

        async def wait(self) -> int:
            await asyncio.sleep(0.02)
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    async def _fake_exec(*_args, **_kwargs):
        return _Proc()

    async def _wait_for_cancel(*args, **_kwargs):
        cancel = next(arg for arg in args if isinstance(arg, asyncio.Event))
        await cancel.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(run_manager, "_fs_watcher", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_heartbeat", _wait_for_cancel)
    monkeypatch.setattr(run_manager, "_kill_on_cancel", _wait_for_cancel)

    def _must_not_publish_outputs(_self):
        pytest.fail("unproven initialization must not publish output lineage")

    monkeypatch.setattr(
        RunLineageSession, "record_outputs", _must_not_publish_outputs,
    )

    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "adapt this complex motion",
            "iterations": 1,
            "dry_run": True,
            "starting_skill_id": record.skill_id,
            "expected_starting_skill_manifest_digest": record.manifest_digest,
            "starting_skill_target_receipt": target_receipt,
            "warm_start_policy_contract_receipt": policy_receipt,
            "initialization_mode": "actor_only",
        },
    )
    job = Job(
        job_id="missing_runtime_proof",
        kind="sculpt_run",
        project_slug=slug,
        status="running",
    )
    result = asyncio.run(runner(job, asyncio.Event()))

    assert result["return_code"] == 0
    assert job.status == "errored"
    assert job.params["error_classification"]["kind"] == (
        "starting_skill_load_unproven"
    )
    assert not any(event.get("type") == "run_completed" for event in job.events)
    assert any(
        event.get("error_kind") == "starting_skill_load_unproven"
        for event in job.events
    )
    assert any(
        event.get("type") == "lineage_outputs_quarantined"
        and event.get("reason") == "starting_skill_load_unproven"
        for event in job.events
    )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ({"source_sha256": "0" * 64}, "full digest"),
        ({"load_cfg_keys": ["actor", "critic"]}, "expected exactly"),
        ({"source": "other"}, "not selected"),
    ],
)
def test_runtime_load_receipt_rejects_wrong_digest_roles_or_source(
    tmp_path: Path,
    mutation: dict[str, object],
    error_match: str,
) -> None:
    from backend.services.run_manager import _verify_starting_skill_load_event

    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    event: dict[str, object] = {
        "type": "warm_start_loaded",
        "source": str(checkpoint),
        "source_sha256": digest,
        "load_cfg_keys": ["actor"],
    }
    event.update(mutation)
    if event.get("source") == "other":
        event["source"] = str(other)

    with pytest.raises(ValueError, match=error_match):
        _verify_starting_skill_load_event(
            event,
            expected_checkpoint=checkpoint,
            expected_sha256=digest,
            initialization_mode="actor_only",
        )


def test_reference_only_import_and_listing_survive_project_contract_failure(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    import sculptor.policy_contract as policy_contract

    monkeypatch.setenv(ENV_LIBRARY_ROOT, str(tmp_path / "skills"))
    monkeypatch.setenv("RS_REFERENCE_ROOT", str(tmp_path / "references"))
    slug, project_dir = _project(client)
    policy_bundle = _bundle(tmp_path, project_dir)
    reference_bundle = _reference_only_bundle(tmp_path)

    def _broken_contract(_project_dir: Path) -> dict:
        raise KeyError("critic term foot_air_time")

    monkeypatch.setattr(
        policy_contract, "build_project_policy_contract", _broken_contract,
    )
    with reference_bundle.open("rb") as handle:
        imported = client.post(
            f"/projects/{slug}/starting-skills",
            files={
                "bundle": ("reference-only.rskill", handle, "application/zip")
            },
        )
    assert imported.status_code == 201, imported.text
    assert imported.json()["compatible"] is True
    assert imported.json()["compatibility"]["allowed_initialization_modes"] == [
        "reference_only"
    ]

    listed = client.get(f"/projects/{slug}/starting-skills")
    assert listed.status_code == 200, listed.text
    assert listed.json()["skills"][0]["compatible"] is True

    with policy_bundle.open("rb") as handle:
        blocked = client.post(
            f"/projects/{slug}/starting-skills",
            files={"bundle": ("g1.rskill", handle, "application/zip")},
        )
    assert blocked.status_code == 412, blocked.text
    assert blocked.json()["code"] == "project_contract_missing"
    assert "critic term foot_air_time" in blocked.json()["detail"]
