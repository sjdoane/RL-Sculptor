"""Policy listing + export-bundle download routes.

Uses a tiny hand-built rsl_rl-format checkpoint (torch save of
{actor_state_dict: mlp.<i>.*}) so the full bundle path — including the
ONNX/TorchScript best-effort exports — runs in-process without GPU or
mjlab task registry (no task_id in config → export assumes elu and says
so in the manifest; that's fine for route-level tests).
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sculptor.run_manifests import (
    build_completion_manifest,
    build_rollout_input_manifest,
    build_train_input_manifest,
    manifest_sha256,
    write_json_atomic,
)
from sculptor.sculpt import _write_iteration_completion_marker

torch = pytest.importorskip("torch")


class _ReceiptAdapter:
    env_id = "policy-route-test"
    robot = "unit"
    control_dt = 0.02


def _attest_iteration(project_dir: Path, iteration: int) -> None:
    iter_dir = project_dir / "runs" / f"iter_{iteration}"
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir(exist_ok=True)
    checkpoint = iter_dir / "checkpoint.pt"
    for path, content in (
        (rollout_dir / "rollout.mp4", b"immutable-rollout"),
        (rollout_dir / "trajectory.npz", b"immutable-trajectory"),
    ):
        if not path.is_file():
            path.write_bytes(content)
    behavior_path = rollout_dir / "behavior.json"
    if not behavior_path.is_file():
        behavior_path.write_text("{}", encoding="utf-8")
    reward = project_dir / "rewards" / "v0.py"
    adapter = _ReceiptAdapter()
    request = build_train_input_manifest(
        adapter=adapter,
        iteration=iteration,
        reward_module_path=reward,
        steps=100,
        seed=iteration,
        init_policy_path=None,
        init_policy_mode="actor_critic",
    )
    train_input = {
        **request,
        "request_manifest_sha256": manifest_sha256(request),
        "effective_initialization": {
            "mode": None,
            "policy": None,
            "forwarded_to_adapter": False,
        },
    }
    rollout_input = build_rollout_input_manifest(
        adapter=adapter,
        iteration=iteration,
        checkpoint_path=checkpoint,
        reward_module_path=reward,
        n_episodes=1,
        seed=iteration,
        max_episode_steps=None,
        playback_speed=None,
        render_every=None,
        fps=None,
        render_width=None,
        render_height=None,
        render_env_index=None,
    )
    write_json_atomic(iter_dir / "train_request_manifest.json", request)
    write_json_atomic(iter_dir / "train_input_manifest.json", train_input)
    write_json_atomic(
        iter_dir / "train_completion_manifest.json",
        build_completion_manifest(train_input, [checkpoint]),
    )
    write_json_atomic(
        rollout_dir / "rollout_input_manifest.json", rollout_input,
    )
    write_json_atomic(
        rollout_dir / "rollout_completion_manifest.json",
        build_completion_manifest(
            rollout_input,
            [
                rollout_dir / "rollout.mp4",
                rollout_dir / "trajectory.npz",
                behavior_path,
            ],
        ),
    )
    _write_iteration_completion_marker(
        iter_dir,
        iter_index=iteration,
        checkpoint_path=checkpoint,
        reward_version_before=0,
        reward_version_after=None,
        world_selection_hash=None,
    )


def _install_verified_origin_lineage(
    project_dir: Path, iteration: int,
) -> None:
    from sculptor.policy_contract import contract_fingerprint
    from sculptor.world.artifacts import ArtifactRef, WorldArtifactStore

    iter_dir = project_dir / "runs" / f"iter_{iteration}"
    env_spec = project_dir / "env" / "unit_env_spec.json"
    env_spec.parent.mkdir(exist_ok=True)
    env_spec.write_text("{}\n", encoding="utf-8")
    store = WorldArtifactStore(project_dir)
    refs = {
        "reward": ArtifactRef.from_path(
            "reward", "v0", project_dir / "rewards" / "v0.py",
            base=project_dir,
        ),
        "env_spec": ArtifactRef.from_path(
            "env_spec", "v1", env_spec, base=project_dir,
        ),
        "world": store.write_json("world", {"shared": {}}),
        "task": store.write_json("task", {"shared": {}}),
        "resolved_eval": store.write_json(
            "resolved_eval", {"objects": {}, "zones": {}},
        ),
        "channel_catalog": store.write_json("channel_catalog", {}),
        "clarifications": store.write_json("clarifications", {}),
    }
    selection = store.promote(refs, evaluation_lineage="policy-test")
    selection_path = project_dir / "env" / (
        f"selection_v{selection.selection_version}.json"
    )
    (iter_dir / "artifact_tuple.json").write_bytes(selection_path.read_bytes())

    checkpoint = iter_dir / "checkpoint.pt"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = {"schema": 1, "policy_interface": {"obs_dim": 4, "action_dim": 2}}
    contract_sha = contract_fingerprint(contract)
    sidecar = Path(str(checkpoint) + ".policy_contract.json")
    sidecar.write_text(json.dumps({
        "schema": 1,
        "checkpoint_sha256": checkpoint_sha,
        "policy_contract": contract,
        "policy_contract_sha256": contract_sha,
    }), encoding="utf-8")
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    metrics = json.loads((iter_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({
        "checkpoint_path": str(checkpoint.resolve()),
        "policy_contract_sidecar": str(sidecar.resolve()),
        "runtime_artifacts": {
            "output_checkpoint_sha256": checkpoint_sha,
            "output_policy_contract_sha256": contract_sha,
            "output_policy_contract_sidecar_sha256": sidecar_sha,
            "environment_artifacts": {
                "world_selection": {
                    "present": True,
                    "tuple_hash": selection.tuple_hash,
                    "refs": selection.to_dict()["refs"],
                },
            },
        },
    })
    (iter_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8",
    )


def _make_project(client: TestClient) -> str:
    r = client.post(
        "/projects",
        json={"name": "Poly", "iteration_budget": 3, "behavior_goal": "hop"},
    )
    assert r.status_code == 201, r.text
    return r.json()["slug"]


def _npy_bytes() -> bytes:
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (1,), }"
    preamble_size = 10
    padding = (-(preamble_size + len(header) + 1)) % 16
    encoded = (header + (" " * padding) + "\n").encode("latin1")
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(encoded))
        + encoded
        + b"\x00"
    )


def _mp4_box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _minimal_structural_mp4() -> bytes:
    track = _mp4_box(
        b"trak",
        _mp4_box(b"tkhd", b"\x00" * 16) + _mp4_box(b"mdia", b""),
    )
    movie = _mp4_box(
        b"moov", _mp4_box(b"mvhd", b"\x00" * 16) + track,
    )
    return (
        _mp4_box(b"ftyp", b"isom\x00\x00\x00\x00isom")
        + movie
        + _mp4_box(b"mdat", b"\x00")
    )


def _plant_iter(
    project_dir: Path, i: int, *, metric: float = 10.0,
    reward_version: str = "v0", fitness: float | None = None,
    fitness_doc: dict | None = None, rollout: bool = False,
    behavior_doc: dict | None = None,
    completed: bool = True, legacy_complete: bool = False,
) -> None:
    it = project_dir / "runs" / f"iter_{i}"
    it.mkdir(parents=True, exist_ok=True)
    sd = {
        "mlp.0.weight": torch.zeros(8, 4), "mlp.0.bias": torch.zeros(8),
        "mlp.2.weight": torch.zeros(2, 8), "mlp.2.bias": torch.zeros(2),
        "distribution.std_param": torch.zeros(2),
    }
    torch.save({"actor_state_dict": sd, "iter": 5}, it / "checkpoint.pt")
    (it / "reward_spec.json").write_text(
        json.dumps({"version": reward_version}))
    (it / "metrics.json").write_text(
        json.dumps({"metrics": {"mean_return": metric}}))
    if fitness_doc is not None:
        (it / "fitness.json").write_text(json.dumps(fitness_doc))
    elif fitness is not None:
        (it / "fitness.json").write_text(json.dumps({"fitness": fitness}))
    if rollout:
        (it / "rollout").mkdir()
        (it / "rollout" / "rollout.mp4").write_bytes(b"immutable-rollout")
    if behavior_doc is not None:
        rollout_dir = it / "rollout"
        rollout_dir.mkdir(exist_ok=True)
        (rollout_dir / "behavior.json").write_text(
            json.dumps(behavior_doc), encoding="utf-8",
        )
    if legacy_complete:
        rollout_dir = it / "rollout"
        rollout_dir.mkdir(exist_ok=True)
        (rollout_dir / "behavior.json").write_text(
            json.dumps({"episodes": 1}), encoding="utf-8",
        )
        with zipfile.ZipFile(
            rollout_dir / "trajectory.npz", "w", zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("trajectory.npy", _npy_bytes())
        (rollout_dir / "rollout.mp4").write_bytes(
            _minimal_structural_mp4()
        )
        if not (it / "fitness.json").is_file():
            (it / "fitness.json").write_text(
                json.dumps({"fitness": metric}), encoding="utf-8",
            )
    if completed:
        _attest_iteration(project_dir, i)


def test_list_policies_empty(client: TestClient, tmp_projects_root: Path):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/policies")
    assert r.status_code == 200
    assert r.json() == []


def test_list_policies_unknown_project(client: TestClient):
    r = client.get("/projects/nope/policies")
    assert r.status_code == 404


def test_list_policies_returns_disk_iters(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    _plant_iter(pdir, 0, metric=5.0)
    _plant_iter(pdir, 2, metric=9.5, reward_version="v2", fitness=0.37)
    r = client.get(f"/projects/{slug}/policies")
    assert r.status_code == 200
    rows = r.json()
    assert [row["iter_index"] for row in rows] == [0, 2]
    assert rows[1]["primary_metric"] == pytest.approx(9.5)
    assert rows[1]["reward_version"] == "v2"
    assert rows[1]["fitness"] == pytest.approx(0.37)
    assert rows[0]["checkpoint"] == "checkpoint.pt"
    assert rows[0]["checkpoint_bytes"] > 0
    assert rows[0]["checkpoint_sha256"] == hashlib.sha256(
        (pdir / "runs" / "iter_0" / "checkpoint.pt").read_bytes()
    ).hexdigest()
    assert rows[0]["deployable"] is False
    assert rows[0]["artifact_purpose"] == "reproducibility"
    assert rows[0]["completion_authority"] == "attested"
    assert rows[0]["deployment_status"] == "not_certified"
    assert rows[0]["deployment_blockers"]


def test_policy_listing_fails_closed_when_checkpoint_identity_escapes(
    client: TestClient,
    tmp_projects_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 0)

    from sculptor import export as export_module

    rows = export_module.list_exportable_iters(project_dir / "runs")
    escaped = dict(rows[0])
    escaped["checkpoint"] = "../checkpoint.pt"
    monkeypatch.setattr(
        export_module,
        "list_exportable_iters",
        lambda _runs_root: [escaped],
    )

    response = client.get(f"/projects/{slug}/policies")

    assert response.status_code == 409
    assert response.json()["type"] == (
        "/problems/policy-checkpoint-identity-unavailable"
    )


def test_policy_listing_excludes_preserved_but_unevaluated_checkpoint(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 2, completed=False)

    listed = client.get(f"/projects/{slug}/policies")
    assert listed.status_code == 200
    assert listed.json() == []

    exported = client.get(f"/projects/{slug}/policies/2/export")
    assert exported.status_code == 409
    assert exported.json()["type"] == (
        "/problems/policy-evaluation-incomplete"
    )
    assert "interrupted-snapshot recovery" in exported.json()["detail"]


def test_policy_listing_rejects_unhashed_completion_marker(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 2, completed=False)
    iter_dir = project_dir / "runs" / "iter_2"
    checkpoint = (iter_dir / "checkpoint.pt").resolve()
    (iter_dir / "iteration_complete.json").write_text(json.dumps({
        "schema": 1,
        "state": "completed",
        "iter": 2,
        "checkpoint": str(checkpoint),
    }), encoding="utf-8")

    listed = client.get(f"/projects/{slug}/policies")
    assert listed.status_code == 200
    assert listed.json() == []


@pytest.mark.parametrize("tamper", ["content", "byte-count"])
def test_policy_listing_rejects_completion_marker_checkpoint_mismatch(
    client: TestClient, tmp_projects_root: Path, tamper: str,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 2)
    iter_dir = project_dir / "runs" / "iter_2"
    checkpoint = iter_dir / "checkpoint.pt"
    marker_path = iter_dir / "iteration_complete.json"
    if tamper == "content":
        mutated = bytearray(checkpoint.read_bytes())
        mutated[-1] ^= 1
        checkpoint.write_bytes(mutated)
    else:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["checkpoint_bytes"] += 1
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    listed = client.get(f"/projects/{slug}/policies")
    assert listed.status_code == 200
    assert listed.json() == []


def test_policy_listing_rejects_schema3_phase_output_tamper(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 2)
    (project_dir / "runs" / "iter_2" / "rollout" / "behavior.json").write_text(
        '{"changed": true}', encoding="utf-8",
    )

    listed = client.get(f"/projects/{slug}/policies")

    assert listed.status_code == 200
    assert listed.json() == []


def test_policy_listing_retains_explicit_full_legacy_completion(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(
        project_dir,
        1,
        fitness=0.5,
        completed=False,
        legacy_complete=True,
    )

    response = client.get(f"/projects/{slug}/policies")
    assert response.status_code == 200
    assert [row["iter_index"] for row in response.json()] == [1]
    row = response.json()[0]
    assert row["completion_authority"] == "legacy_recovery"
    assert row["deployable"] is False
    assert "recovery-only" in " ".join(row["deployment_blockers"])


@pytest.mark.parametrize("corrupt", ["trajectory", "video"])
def test_policy_listing_rejects_corrupt_legacy_binary_evidence(
    client: TestClient, tmp_projects_root: Path, corrupt: str,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(
        project_dir,
        1,
        fitness=0.5,
        completed=False,
        legacy_complete=True,
    )
    rollout_dir = project_dir / "runs" / "iter_1" / "rollout"
    target = (
        rollout_dir / "trajectory.npz"
        if corrupt == "trajectory"
        else rollout_dir / "rollout.mp4"
    )
    target.write_bytes(b"nonempty but corrupt")

    response = client.get(f"/projects/{slug}/policies")
    assert response.status_code == 200
    assert response.json() == []


def test_policy_listing_uses_evidenced_selection_not_newest(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    _plant_iter(
        pdir,
        4,
        fitness_doc={
            "fitness": 0.94,
            "metric": {
                "id": "weave-stop",
                "version": "v3",
                "source": "generated",
                "sha256": "c" * 64,
            },
            "components": {
                "order_ok_frac": 1.0,
                "contact_evidence_ok": 1.0,
                "contact_frac": 0.0,
                "ch_hold": 1.0,
            },
        },
        rollout=True,
        behavior_doc={
            "rendered_env_index_requested": 10,
            "rendered_env_index": 10,
            "rendered_env_selection": "precommitted",
            "rendered_episode_percentile": 0.8125,
        },
    )
    _plant_iter(
        pdir,
        5,
        fitness_doc={
            "fitness": 0.98,
            "components": {
                "order_ok_frac": 0.0,
                "contact_evidence_ok": 1.0,
                "contact_frac": 0.0,
                "ch_hold": 0.0,
            },
        },
    )
    (pdir / "reports").mkdir(exist_ok=True)
    (pdir / "reports" / "selection.json").write_text(json.dumps({
        "selected_iter_index": 4,
        "selection_source": "objective_criterion",
        "candidates": [
            {"iter_index": 4, "selected": True, "criterion_pass": True},
            {"iter_index": 5, "selected": False, "criterion_pass": False},
        ],
    }))

    response = client.get(f"/projects/{slug}/policies")
    assert response.status_code == 200, response.text
    rows = {row["iter_index"]: row for row in response.json()}

    assert rows[4]["selected"] is True
    assert rows[4]["selection_source"] == "objective_criterion"
    assert rows[4]["criterion_status"] == "passed"
    assert rows[4]["metric_id"] == "weave-stop"
    assert rows[4]["metric_version"] == "v3"
    assert rows[4]["metric_sha256"] == "c" * 64
    assert rows[4]["evidence_status"] == "complete"
    assert rows[4]["route_evidence"] == {
        "key": "order_ok_frac", "value": 1.0, "kind": "fraction",
        "comparison": "gte", "threshold": 1.0, "passed": True,
        "semantics_source": (
            "reward-sculptor-objective-evidence-semantics-v1"
        ),
    }
    assert rows[4]["contact_evidence"]["key"] == "contact_frac"
    assert rows[4]["contact_evidence"]["comparison"] == "lte"
    assert rows[4]["contact_evidence"]["threshold"] == 0.0
    assert rows[4]["contact_evidence"]["passed"] is True
    assert rows[4]["hold_evidence"]["key"] == "ch_hold"
    assert rows[4]["hold_evidence"]["passed"] is True
    assert rows[4]["objective_proof_status"] == "passed"
    assert rows[4]["objective_proof_blockers"] == []
    assert rows[4]["lane_evidence_status"] == "verified"
    assert rows[4]["requested_evidence_env_index"] == 10
    assert rows[4]["resolved_evidence_env_index"] == 10
    assert rows[4]["resolved_episode_percentile"] == pytest.approx(0.8125)
    assert rows[4]["evidence_lane_selection"] == "precommitted"
    assert rows[4]["rollout_available"] is True

    assert rows[5]["selected"] is False
    assert rows[5]["selection_source"] is None
    assert rows[5]["criterion_status"] == "failed"
    assert rows[5]["objective_proof_status"] == "failed"
    assert rows[5]["rollout_available"] is True


def test_report_selection_authority_requires_selection_and_passed_evidence(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    from backend.routes.policies import report_selection_authority

    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(
        project_dir,
        0,
        fitness_doc={
            "fitness": 1.0,
            "metric": {
                "id": "objective",
                "version": "v1",
                "source": "generated",
                "sha256": "a" * 64,
            },
            "components": {
                "order_ok_frac": 1.0,
                "contact_frac": 0.0,
                "ch_hold": 1.0,
            },
        },
        behavior_doc={
            "rendered_env_index_requested": 0,
            "rendered_env_index": 0,
            "rendered_env_selection": "precommitted",
            "rendered_episode_percentile": 0.5,
        },
    )
    iter_dir = project_dir / "runs" / "iter_0"
    reward_path = project_dir / "rewards" / "v0.py"
    artifact_tuple_path = iter_dir / "artifact_tuple.json"
    artifact_tuple = {
        "refs": {
            "reward": {
                "path": "rewards/v0.py",
                "sha256": hashlib.sha256(
                    reward_path.read_bytes()
                ).hexdigest(),
            },
        },
    }
    artifact_tuple_path.write_text(
        json.dumps(artifact_tuple), encoding="utf-8",
    )

    missing = report_selection_authority(project_dir)
    assert missing["status"] == "unavailable"
    assert missing["selected_iter_index"] is None

    reports = project_dir / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "selection.json").write_text(json.dumps({
        "selected_iter_index": 0,
        "selection_source": "objective_criterion",
        "candidates": [
            {"iter_index": 0, "selected": True, "criterion_pass": True},
        ],
    }), encoding="utf-8")
    verified = report_selection_authority(project_dir)
    assert verified["status"] == "verified"
    assert verified["selected_iter_index"] == 0
    assert verified["objective_evidence_receipt"][
        "objective_proof_status"
    ] == "passed"
    assert verified["claim_inputs"]["artifact_tuple_sha256"]
    assert verified["claim_inputs"]["selected_reward_path"] == "rewards/v0.py"
    assert verified["claim_inputs"]["selected_reward_sha256"] == (
        artifact_tuple["refs"]["reward"]["sha256"]
    )

    mismatched_tuple = json.loads(json.dumps(artifact_tuple))
    mismatched_tuple["refs"]["reward"]["sha256"] = "0" * 64
    artifact_tuple_path.write_text(
        json.dumps(mismatched_tuple), encoding="utf-8",
    )
    mismatched = report_selection_authority(project_dir)
    assert mismatched["status"] == "unavailable"
    assert any("selected reward bytes" in blocker for blocker in mismatched["blockers"])

    artifact_tuple_path.write_text(
        json.dumps(artifact_tuple), encoding="utf-8",
    )

    fitness = json.loads((iter_dir / "fitness.json").read_text(encoding="utf-8"))
    fitness["components"]["order_ok_frac"] = 0.0
    (iter_dir / "fitness.json").write_text(json.dumps(fitness), encoding="utf-8")
    failed = report_selection_authority(project_dir)
    assert failed["status"] == "unavailable"
    assert failed["selected_iter_index"] is None
    assert any("objective proof" in blocker for blocker in failed["blockers"])


def test_policy_is_qualified_only_with_exact_origin_and_all_gates(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(
        project_dir,
        0,
        fitness_doc={
            "fitness": 1.0,
            "metric": {
                "id": "objective",
                "version": "v1",
                "source": "generated",
                "sha256": "b" * 64,
            },
            "components": {
                "order_ok_frac": 1.0,
                "contact_frac": 0.0,
                "ch_hold": 1.0,
            },
        },
        behavior_doc={
            "rendered_env_index_requested": 0,
            "rendered_env_index": 0,
            "rendered_env_selection": "precommitted",
            "rendered_episode_percentile": 0.5,
        },
    )
    _install_verified_origin_lineage(project_dir, 0)
    reports = project_dir / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "selection.json").write_text(json.dumps({
        "selected_iter_index": 0,
        "selection_source": "objective_criterion",
        "candidates": [
            {"iter_index": 0, "selected": True, "criterion_pass": True},
        ],
    }), encoding="utf-8")

    response = client.get(f"/projects/{slug}/policies")

    assert response.status_code == 200, response.text
    row = response.json()[0]
    assert row["completion_authority"] == "attested"
    assert row["objective_proof_status"] == "passed"
    assert row["physical_scene_status"] == "not_applicable"
    assert row["lineage_status"] == "verified"
    assert row["deployment_status"] == "qualified"
    assert row["deployable"] is True
    assert row["deployment_blockers"] == []

    exported = client.get(f"/projects/{slug}/policies/0/export")
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["deployment_status"] == "qualified"
    assert manifest["deployment_authority"]["checks"][
        "origin_lineage"
    ]["origin_receipt_sha256"] == row["origin_receipt_sha256"]


def test_policy_objective_proof_fails_on_present_but_failing_channels(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(
        project_dir,
        3,
        fitness_doc={
            "fitness": 0.99,
            "metric": {
                "id": "weave-stop",
                "version": "v3",
                "source": "generated",
                "sha256": "d" * 64,
            },
            "components": {
                "actual_route_complete_frac": 0.0,
                "forbidden_contact_count": 2.0,
                "hold_frames": 0.0,
            },
        },
        rollout=True,
        behavior_doc={
            "rendered_env_index_requested": 10,
            "rendered_env_index": 10,
            "rendered_env_selection": "precommitted",
            "rendered_episode_percentile": 0.5,
            "terminal_proof_contract": {"minimum_hold_frames": 100},
        },
    )
    (project_dir / "reports").mkdir(exist_ok=True)
    (project_dir / "reports" / "selection.json").write_text(json.dumps({
        "selected_iter_index": 3,
        "selection_source": "objective_criterion",
        "candidates": [
            {"iter_index": 3, "selected": True, "criterion_pass": True},
        ],
    }))

    response = client.get(f"/projects/{slug}/policies")

    assert response.status_code == 200, response.text
    row = response.json()[0]
    assert row["evidence_status"] == "complete"
    assert row["route_evidence"]["passed"] is False
    assert row["contact_evidence"]["passed"] is False
    assert row["hold_evidence"] == {
        "key": "hold_frames", "value": 0.0, "kind": "frames",
        "comparison": "gte", "threshold": 100.0, "passed": False,
        "semantics_source": (
            "behavior.terminal_proof_contract.minimum_hold_frames"
        ),
    }
    assert row["criterion_status"] == "passed"
    assert row["objective_proof_status"] == "failed"
    assert row["objective_proof_blockers"] == [
        "route evidence failed its declared comparison",
        "forbidden contact evidence failed its declared comparison",
        "terminal hold evidence failed its declared comparison",
    ]


@pytest.mark.parametrize(
    ("behavior_doc", "expected_status"),
    [
        ({}, "unavailable"),
        ({
            "rendered_env_index_requested": 10,
            "rendered_env_index": 10,
            "rendered_env_selection": "precommitted",
        }, "incomplete"),
        ({
            "rendered_env_index_requested": 10,
            "rendered_env_index": 9,
            "rendered_env_selection": "precommitted",
            "rendered_episode_percentile": 0.5,
        }, "mismatch"),
    ],
)
def test_policy_lane_receipt_fails_closed_without_exact_worker_fields(
    client: TestClient,
    tmp_projects_root: Path,
    behavior_doc: dict,
    expected_status: str,
) -> None:
    slug = _make_project(client)
    _plant_iter(
        tmp_projects_root / slug,
        0,
        rollout=True,
        behavior_doc=behavior_doc,
    )

    response = client.get(f"/projects/{slug}/policies")

    assert response.status_code == 200, response.text
    row = response.json()[0]
    assert row["lane_evidence_status"] == expected_status
    if expected_status != "verified":
        assert row["lane_evidence_status"] != "verified"


def test_export_downloads_zip_bundle(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    _plant_iter(pdir, 1)
    r = client.get(f"/projects/{slug}/policies/1/reproducibility")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert "iter1.zip" in r.headers.get("content-disposition", "")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert {"manifest.json", "checkpoint.pt", "DEPLOY.md",
            "reward/reward_spec.json", "config.toml"} <= names
    assert manifest["iter_index"] == 1
    assert manifest["checkpoint"]["format"] == "rsl_rl"
    assert manifest["artifact_purpose"] == "reproducibility"
    assert manifest["deployment_status"] == "not_certified"
    # tiny MLP: dims inferred from the state dict
    assert manifest["network"]["obs_dim"] == 4
    assert manifest["network"]["action_dim"] == 2
    # bundle also persisted under the project for reuse
    assert (pdir / "exports").is_dir()
    assert any((pdir / "exports").glob("*.zip"))


def test_deployment_export_blocks_nonqualified_policy_without_building_bundle(
    client: TestClient, tmp_projects_root: Path,
) -> None:
    slug = _make_project(client)
    project_dir = tmp_projects_root / slug
    _plant_iter(project_dir, 1)

    response = client.get(f"/projects/{slug}/policies/1/export")

    assert response.status_code == 409
    assert response.json()["type"] == (
        "/problems/policy-not-deployment-qualified"
    )
    assert response.json()["blockers"]
    assert not (project_dir / "exports").exists()


def test_export_missing_iter_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies/7/export")
    assert r.status_code == 404
    assert "iter 7" in r.json()["detail"]


def test_export_no_checkpoints_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    r = client.get(f"/projects/{slug}/policies/0/reproducibility")
    assert r.status_code == 404
    assert "no retained checkpoint" in r.json()["detail"]


def test_export_negative_iter_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies/-3/export")
    assert r.status_code == 404


def test_export_unknown_run_id_404(
    client: TestClient, tmp_projects_root: Path,
):
    slug = _make_project(client)
    _plant_iter(tmp_projects_root / slug, 0)
    r = client.get(f"/projects/{slug}/policies?run_id=ghost")
    assert r.status_code == 404
    r = client.get(f"/projects/{slug}/policies/0/export?run_id=ghost")
    assert r.status_code == 404


def test_export_corrupt_checkpoint_still_ships_raw(
    client: TestClient, tmp_projects_root: Path,
):
    """A checkpoint torch can't read must still produce a bundle with the
    raw file + recipe (never a 500)."""
    slug = _make_project(client)
    pdir = tmp_projects_root / slug
    it = pdir / "runs" / "iter_0"
    it.mkdir(parents=True)
    (it / "checkpoint.pt").write_bytes(b"garbage")
    (it / "iteration_complete.json").write_text(json.dumps({
        "schema": 2,
        "state": "completed",
        "iter": 0,
        "checkpoint": str((it / "checkpoint.pt").resolve()),
        "checkpoint_sha256": hashlib.sha256(b"garbage").hexdigest(),
        "checkpoint_bytes": len(b"garbage"),
    }), encoding="utf-8")
    r = client.get(f"/projects/{slug}/policies/0/reproducibility")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "checkpoint.pt" in names
    assert "policy.onnx" not in names
    assert any("unreadable" in w for w in manifest["warnings"])
