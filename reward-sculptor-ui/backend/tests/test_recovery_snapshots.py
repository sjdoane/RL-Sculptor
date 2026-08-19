"""Interrupted PPO snapshot discovery, admission, and worker tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def _event(payload: dict) -> str:
    return "[SCULPT-EVENT] " + json.dumps(payload, separators=(",", ":"))


def _legacy_npy_bytes() -> bytes:
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (1,), }"
    padding = (-(10 + len(header) + 1)) % 16
    encoded = (header + (" " * padding) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + b"\x00"


def _legacy_mp4_box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _legacy_structural_mp4() -> bytes:
    track = _legacy_mp4_box(
        b"trak",
        _legacy_mp4_box(b"tkhd", b"\x00" * 16)
        + _legacy_mp4_box(b"mdia", b""),
    )
    return (
        _legacy_mp4_box(b"ftyp", b"isom\x00\x00\x00\x00isom")
        + _legacy_mp4_box(
            b"moov", _legacy_mp4_box(b"mvhd", b"\x00" * 16) + track,
        )
        + _legacy_mp4_box(b"mdat", b"\x00")
    )


def _plant_interrupted_snapshot(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_step: int = 50,
    observed_step: int = 58,
    completed: bool = False,
    include_final_checkpoint: bool = False,
    training_completed: bool = False,
    contract_event_type: str = "warm_start_observation_extended",
    warm_start_role: str | None = "actor_critic",
    symlink_model: bool = False,
) -> tuple[Path, bytes]:
    """Plant the minimum exact evidence used by legacy reconstruction."""
    from backend.services import recovery_snapshots
    from sculptor.policy_contract import contract_fingerprint

    iteration = 2
    iter_dir = project_dir / "runs" / f"iter_{iteration}"
    logs_dir = iter_dir / "logs"
    env_dir = project_dir / "env"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_bytes = f"actor-and-critic-at-ppo-{model_step}".encode()
    model_path = logs_dir / f"model_{model_step}.pt"
    if symlink_model:
        outside = project_dir.parent / "outside-model.pt"
        outside.write_bytes(checkpoint_bytes)
        model_path.symlink_to(outside)
    else:
        model_path.write_bytes(checkpoint_bytes)
    # model_0 is intentionally never a recoverable trained snapshot.
    (logs_dir / "model_0.pt").write_bytes(b"initial-policy")
    if completed or include_final_checkpoint:
        final_checkpoint = iter_dir / "checkpoint.pt"
        final_checkpoint.write_bytes(checkpoint_bytes)
    if completed:
        (iter_dir / "iteration_complete.json").write_text(json.dumps({
            "schema": 2,
            "state": "completed",
            "iter": iteration,
            "checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_bytes": len(checkpoint_bytes),
        }), encoding="utf-8")

    contract = {
        "schema": 3,
        "identity": {
            "adapter_class": "test.Adapter", "task_id": "test-task",
        },
        "policy": {"roles": ["actor", "critic"]},
        "versions": {
            "torch": "2.11",
            "mjlab": "1.3.0",
            "rsl_rl": "5.0.1",
            "adapter": "0.1.0",
        },
    }
    contract_path = iter_dir / "warm_start_effective_policy_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_sha = contract_fingerprint(contract)

    selection_doc = {
        "selection_version": 6,
        "tuple_hash": "f" * 64,
        "refs": {},
        "created_at": 1.0,
        "evaluation_lineage": "fixture",
    }
    selection_path = env_dir / "selection_v6.json"
    selection_path.write_text(json.dumps(selection_doc), encoding="utf-8")
    stale_tuple = dict(selection_doc)
    stale_tuple["selection_version"] = 4
    stale_tuple["tuple_hash"] = "e" * 64
    (iter_dir / "artifact_tuple.json").write_text(
        json.dumps(stale_tuple), encoding="utf-8",
    )

    fake_selection = SimpleNamespace(
        selection_version=6,
        tuple_hash="f" * 64,
        to_dict=lambda: dict(selection_doc),
    )

    def _read_selection(_self, path):
        return fake_selection if Path(path).resolve() == selection_path.resolve() else None

    monkeypatch.setattr(
        recovery_snapshots.WorldArtifactStore, "read_selection", _read_selection,
    )
    monkeypatch.setattr(
        recovery_snapshots,
        "build_project_policy_contract",
        lambda *_args, **_kwargs: dict(contract),
    )

    context_event: list[str] = []
    if warm_start_role is None:
        config_path = project_dir / "config.toml"
        config_path.write_text("[target]\nname = 'fixture'\n", encoding="utf-8")
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        reports_dir = project_dir / "reports"
        reports_dir.mkdir()
        context_path = reports_dir / "run_context.json"
        context_path.write_text(json.dumps({
            "schema": 3,
            "start_iter": iteration,
            "iterations": 1,
            "code_git": {"sha": "1" * 40, "dirty": False},
            "config": {
                "path": str(config_path.resolve()),
                "sha256": config_sha,
                "effective": {
                    "adapter": {
                        "class": "test.Adapter",
                        "config": {"task_id": "test-task"},
                    },
                },
            },
            "seeds": {"base_seed": 42},
            "packages": {
                "torch": "2.11.0",
                "mjlab": "1.3.0",
                "rsl-rl-lib": "5.0.1",
                "reward-sculptor": "0.1.0",
            },
        }), encoding="utf-8")
        context_event = [_event({
            "type": "run_context_captured",
            "path": str(context_path.resolve()),
            "code_sha": "1" * 40,
            "code_dirty": False,
            "config_sha256": config_sha,
            "base_seed": 42,
        })]

    log_path = project_dir / "runs" / "_run_job_deadbeefcafebabe.log"
    log_path.write_text("\n".join([
        _event({
            "type": "artifact_tuple_pinned",
            "iter": iteration,
            "tuple_hash": "f" * 64,
            "selection": "selection_v6.json",
        }),
        *context_event,
        _event({
            "type": "iter_started",
            "iter": iteration,
            "steps": 750,
            **(
                {"warm_start_source": "/server-owned/source.pt"}
                if warm_start_role is not None
                else {}
            ),
        }),
        *(
            [
                _event({
                    "type": contract_event_type,
                    "effective_policy_contract": str(contract_path.resolve()),
                    "effective_policy_contract_sha256": contract_sha,
                }),
                _event({
                    "type": "warm_start_loaded",
                    "source": "/server-owned/source.pt",
                    "source_sha256": "a" * 64,
                    "load_cfg_keys": (
                        ["actor"]
                        if warm_start_role == "actor_only"
                        else ["actor", "critic"]
                    ),
                }),
            ]
            if warm_start_role is not None
            else []
        ),
        _event({
            "type": "learning_vitals",
            "rl_iter": observed_step,
            "rl_total": 750,
        }),
        *(
            [
                _event({
                    "type": "iter_progress", "rl_iter": 750, "rl_total": 750,
                }),
                json.dumps({
                    "status": "ok",
                    "checkpoint": str((iter_dir / "checkpoint.pt").resolve()),
                }, separators=(",", ":")),
            ]
            if training_completed
            else []
        ),
        "RuntimeError: mjlab rollout runner exited 1",
    ]) + "\n", encoding="utf-8")
    return model_path, checkpoint_bytes


def _make_project(client: TestClient, name: str = "Recovery") -> tuple[str, Path]:
    response = client.post(
        "/projects",
        json={"name": name, "iteration_budget": 3, "behavior_goal": "hop"},
    )
    assert response.status_code == 201, response.text
    slug = response.json()["slug"]
    detail = client.app.state.project_store.get(slug)  # type: ignore[attr-defined]
    assert detail is not None
    return slug, Path(detail.project_dir)


def test_discovery_materializes_one_immutable_nonzero_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "project"
    origin, expected = _plant_interrupted_snapshot(project_dir, monkeypatch)

    rows = discover_recovery_snapshots(project_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["ppo_step"] == 50
    assert row["last_observed_ppo_iteration"] == 58
    assert row["provenance_status"] == "legacy_reconstructed"
    assert row["source_job_id"] == "job_deadbeefcafebabe"
    assert row["checkpoint_sha256"] == hashlib.sha256(expected).hexdigest()
    assert all("path" not in key for key in row)

    cached, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=row["snapshot_id"],
        checkpoint_sha256=row["checkpoint_sha256"],
        receipt_digest=row["receipt_digest"],
    )
    assert cached.read_bytes() == expected
    assert cached != origin
    assert cached.parent.parent.name == "_recovery"
    assert receipt["source"]["matches_pinned_selection"] is False
    evidence_paths = [
        Path(receipt["checkpoint"]["path"]),
        Path(receipt["source"]["effective_policy_contract_path"]),
        Path(receipt["source"]["selection_path"]),
        Path(receipt["source"]["artifact_tuple_path"]),
        Path(receipt["source"]["log_path"]),
    ]
    assert {path.parent for path in evidence_paths} == {cached.parent}
    assert all(path.is_file() for path in evidence_paths)


def test_post_training_rollout_failure_admits_only_latest_exact_contract_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "post-training-rollout-failure"
    origin, expected = _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        model_step=749,
        observed_step=749,
        include_final_checkpoint=True,
        training_completed=True,
        contract_event_type="warm_start_observation_contract_verified",
    )
    logs_dir = origin.parent
    (logs_dir / "model_100.pt").write_bytes(b"periodic-100")
    (logs_dir / "model_700.pt").write_bytes(b"periodic-700")

    rows = discover_recovery_snapshots(project_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["ppo_step"] == 749
    assert row["last_observed_ppo_iteration"] == 750
    assert row["checkpoint_sha256"] == hashlib.sha256(expected).hexdigest()
    assert len(list(
        (project_dir / "runs" / "_recovery").glob("*/receipt.json")
    )) == 1

    _, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=row["snapshot_id"],
        checkpoint_sha256=row["checkpoint_sha256"],
        receipt_digest=row["receipt_digest"],
    )
    assert receipt["source"]["selection_version"] == 6
    assert receipt["source"]["tuple_hash"] == "f" * 64
    assert receipt["source"]["matches_pinned_selection"] is False
    assert receipt["checkpoint"]["origin_path"] == str(origin)

    # Discovery is stable: cached authority suppresses lower periodic saves.
    assert discover_recovery_snapshots(project_dir) == rows
    assert len(list(
        (project_dir / "runs" / "_recovery").glob("*/receipt.json")
    )) == 1


def test_post_training_actor_only_warm_start_admits_produced_actor_critic_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "actor-only-post-training-failure"
    _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        model_step=749,
        observed_step=749,
        include_final_checkpoint=True,
        training_completed=True,
        contract_event_type="warm_start_observation_contract_verified",
        warm_start_role="actor_only",
    )

    rows = discover_recovery_snapshots(project_dir)
    assert len(rows) == 1
    _, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=rows[0]["snapshot_id"],
        checkpoint_sha256=rows[0]["checkpoint_sha256"],
        receipt_digest=rows[0]["receipt_digest"],
    )
    assert receipt["source"]["producer_initialization_roles"] == ["actor"]
    assert receipt["source"]["effective_policy_contract_authority"] == (
        "warm_start_effective_contract_event"
    )


def test_post_training_scratch_run_caches_exact_contract_context_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "scratch-post-training-failure"
    _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        model_step=749,
        observed_step=749,
        include_final_checkpoint=True,
        training_completed=True,
        warm_start_role=None,
    )

    rows = discover_recovery_snapshots(project_dir)
    assert len(rows) == 1
    _, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=rows[0]["snapshot_id"],
        checkpoint_sha256=rows[0]["checkpoint_sha256"],
        receipt_digest=rows[0]["receipt_digest"],
    )
    source = receipt["source"]
    assert source["producer_initialization_roles"] == []
    assert source["effective_policy_contract_authority"] == (
        "scratch_run_context_selection"
    )
    recovery_dir = Path(receipt["checkpoint"]["path"]).parent
    assert Path(source["run_context_path"]) == recovery_dir / "run_context.json"
    assert Path(source["run_config_path"]) == recovery_dir / "config.toml"
    assert Path(source["effective_policy_contract_path"]) == (
        recovery_dir / "effective_policy_contract.json"
    )


def test_latest_snapshot_uses_unique_same_iteration_retry_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "same-iteration-retry"
    _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        model_step=749,
        observed_step=749,
        include_final_checkpoint=True,
        training_completed=True,
    )
    current_log = project_dir / "runs" / "_run_job_deadbeefcafebabe.log"
    old_log = project_dir / "runs" / "_run_job_earlierretry.log"
    current_text = current_log.read_text(encoding="utf-8")
    successful_handoff = json.dumps({
        "status": "ok",
        "checkpoint": str(
            (project_dir / "runs" / "iter_2" / "checkpoint.pt").resolve()
        ),
    }, separators=(",", ":"))
    old_log.write_text(
        current_text.replace('"rl_iter":749', '"rl_iter":58').replace(
            successful_handoff, "RuntimeError: mjlab runner exited 1",
        ),
        encoding="utf-8",
    )

    rows = discover_recovery_snapshots(project_dir)
    assert len(rows) == 1
    assert rows[0]["source_job_id"] == "job_deadbeefcafebabe"
    _, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=rows[0]["snapshot_id"],
        checkpoint_sha256=rows[0]["checkpoint_sha256"],
        receipt_digest=rows[0]["receipt_digest"],
    )
    assert receipt["source"]["last_observed_ppo_step"] == 750


def test_partial_snapshot_rejects_synthetic_outer_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    project_dir = tmp_path / "synthetic-outer-progress"
    _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        model_step=700,
        observed_step=58,
    )
    log_path = project_dir / "runs" / "_run_job_deadbeefcafebabe.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(_event({
            "type": "iter_progress", "rl_iter": 750, "rl_total": 750,
        }) + "\n")

    assert discover_recovery_snapshots(project_dir) == []


def test_full_legacy_completion_suppresses_new_recovery_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    project_dir = tmp_path / "legacy-completed"
    _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        include_final_checkpoint=True,
    )
    iter_dir = project_dir / "runs" / "iter_2"
    rollout_dir = iter_dir / "rollout"
    rollout_dir.mkdir()
    (rollout_dir / "behavior.json").write_text(
        json.dumps({"episodes": 2}), encoding="utf-8",
    )
    with zipfile.ZipFile(
        rollout_dir / "trajectory.npz", "w", zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("trajectory.npy", _legacy_npy_bytes())
    (rollout_dir / "rollout.mp4").write_bytes(_legacy_structural_mp4())
    (iter_dir / "fitness.json").write_text(
        json.dumps({"fitness": 0.25}), encoding="utf-8",
    )

    assert discover_recovery_snapshots(project_dir) == []


@pytest.mark.parametrize("mode", ["model0", "completed", "symlink"])
def test_discovery_rejects_untrained_completed_and_linked_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    project_dir = tmp_path / mode
    model_path, _ = _plant_interrupted_snapshot(
        project_dir,
        monkeypatch,
        completed=mode == "completed",
        symlink_model=mode == "symlink",
    )
    if mode == "model0":
        model_path.unlink()
    assert discover_recovery_snapshots(project_dir) == []


def test_discovery_never_writes_through_a_linked_recovery_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    project_dir = tmp_path / "linked-cache"
    _plant_interrupted_snapshot(project_dir, monkeypatch)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (project_dir / "runs" / "_recovery").symlink_to(outside, target_is_directory=True)

    assert discover_recovery_snapshots(project_dir) == []
    assert list(outside.iterdir()) == []


def test_cached_receipt_survives_origin_mutation_deletion_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        RecoverySnapshotError,
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "stale"
    origin, expected = _plant_interrupted_snapshot(project_dir, monkeypatch)
    row = discover_recovery_snapshots(project_dir)[0]

    with pytest.raises(RecoverySnapshotError, match="checkpoint changed"):
        resolve_recovery_snapshot(
            project_dir,
            snapshot_id=row["snapshot_id"],
            checkpoint_sha256="0" * 64,
            receipt_digest=row["receipt_digest"],
        )
    with pytest.raises(RecoverySnapshotError, match="receipt changed"):
        resolve_recovery_snapshot(
            project_dir,
            snapshot_id=row["snapshot_id"],
            checkpoint_sha256=row["checkpoint_sha256"],
            receipt_digest="0" * 64,
        )

    iter_dir = project_dir / "runs" / "iter_2"
    origin.write_bytes(b"different-origin-checkpoint")
    (iter_dir / "warm_start_effective_policy_contract.json").unlink()
    (project_dir / "env" / "selection_v6.json").unlink()
    (iter_dir / "artifact_tuple.json").unlink()
    (project_dir / "runs" / "_run_job_deadbeefcafebabe.log").unlink()
    (iter_dir / "checkpoint.pt").write_bytes(b"completed-later")
    (iter_dir / "iteration_complete.json").write_text(
        "{}", encoding="utf-8",
    )

    rediscovered = discover_recovery_snapshots(project_dir)
    assert rediscovered == [row]
    cached, receipt = resolve_recovery_snapshot(
        project_dir,
        snapshot_id=row["snapshot_id"],
        checkpoint_sha256=row["checkpoint_sha256"],
        receipt_digest=row["receipt_digest"],
    )
    assert cached.read_bytes() == expected
    assert receipt["receipt_digest"] == row["receipt_digest"]


def test_resolution_rejects_mutated_cached_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import (
        RecoverySnapshotError,
        discover_recovery_snapshots,
        resolve_recovery_snapshot,
    )

    project_dir = tmp_path / "cached-mutation"
    _plant_interrupted_snapshot(project_dir, monkeypatch)
    row = discover_recovery_snapshots(project_dir)[0]
    receipt_path = (
        project_dir / "runs" / "_recovery" / row["snapshot_id"]
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    Path(receipt["source"]["log_path"]).write_text(
        "mutated cached evidence", encoding="utf-8",
    )
    with pytest.raises(RecoverySnapshotError, match="cached worker log changed"):
        resolve_recovery_snapshot(
            project_dir,
            snapshot_id=row["snapshot_id"],
            checkpoint_sha256=row["checkpoint_sha256"],
            receipt_digest=row["receipt_digest"],
        )


def test_discovery_rejects_ambiguous_legacy_producer_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services.recovery_snapshots import discover_recovery_snapshots

    project_dir = tmp_path / "ambiguous-producer"
    _plant_interrupted_snapshot(project_dir, monkeypatch)
    original_log = project_dir / "runs" / "_run_job_deadbeefcafebabe.log"
    shutil.copyfile(
        original_log,
        project_dir / "runs" / "_run_job_secondproducer.log",
    )
    assert discover_recovery_snapshots(project_dir) == []
    assert not list((project_dir / "runs").glob("_recovery/*/receipt.json"))


def test_recovery_endpoint_is_separate_and_exposes_no_server_paths(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug, project_dir = _make_project(client, "Recovery Endpoint")
    _plant_interrupted_snapshot(project_dir, monkeypatch)

    policies = client.get(f"/projects/{slug}/policies")
    recovery = client.get(f"/projects/{slug}/policies/recovery-snapshots")
    assert policies.status_code == 200
    assert policies.json() == []
    assert recovery.status_code == 200, recovery.text
    assert len(recovery.json()) == 1
    assert not any("path" in key for key in recovery.json()[0])


def test_recovery_endpoint_rejects_selection_while_project_worker_is_active(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug, project_dir = _make_project(client, "Recovery Active Guard")
    _plant_interrupted_snapshot(project_dir, monkeypatch)
    client.app.state.job_manager.register_passive_job(  # type: ignore[attr-defined]
        "sculpt_run", slug,
    )

    response = client.get(f"/projects/{slug}/policies/recovery-snapshots")
    assert response.status_code == 409
    assert response.json()["type"] == "/problems/recovery-snapshot-active"


def test_launch_rejects_mixed_sources_and_missing_acknowledgements(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.routes import runs as runs_routes

    slug, _ = _make_project(client, "Recovery Admission")
    ref = {
        "snapshot_id": "snap_" + "a" * 24,
        "checkpoint_sha256": "b" * 64,
        "receipt_digest": "c" * 64,
    }
    mixed = client.post(f"/projects/{slug}/runs", json={
        "behavior_goal": "continue a difficult motion",
        "iterations": 1,
        "dry_run": True,
        "warm_start_iteration": 2,
        "warm_start_snapshot": {
            **ref, "acknowledge_interrupted_snapshot": True,
        },
    })
    assert mixed.status_code == 412
    assert mixed.json()["title"] == "choose one starting policy source"

    missing = client.post(f"/projects/{slug}/runs", json={
        "behavior_goal": "continue a difficult motion",
        "iterations": 1,
        "dry_run": True,
        "warm_start_snapshot": ref,
    })
    assert missing.status_code == 412
    assert missing.json()["type"] == (
        "/problems/recovery-snapshot-acknowledgement"
    )

    legacy_receipt = {"provenance_status": "legacy_reconstructed"}
    monkeypatch.setattr(
        runs_routes,
        "resolve_recovery_snapshot",
        lambda *_args, **_kwargs: (Path("/cached/checkpoint.pt"), legacy_receipt),
    )
    no_legacy = client.post(f"/projects/{slug}/runs", json={
        "behavior_goal": "continue a difficult motion",
        "iterations": 1,
        "dry_run": True,
        "warm_start_snapshot": {
            **ref, "acknowledge_interrupted_snapshot": True,
        },
    })
    assert no_legacy.status_code == 412
    assert no_legacy.json()["type"] == (
        "/problems/recovery-snapshot-provenance"
    )


def test_route_admits_attested_snapshot_as_actor_critic_only(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.routes import runs as runs_routes
    from backend.services import world_store

    slug, project_dir = _make_project(client, "Recovery Route Success")
    (project_dir / "env").mkdir(exist_ok=True)
    (project_dir / "env" / "selection_current.json").write_text(
        "{}", encoding="utf-8",
    )
    monkeypatch.setattr(
        world_store,
        "training_preflight",
        lambda _project_dir: {
            "ok": True,
            "selection_version": 6,
            "tuple_hash": "f" * 64,
            "world_robot": None,
            "project_robot": None,
            "robot_matches_project": True,
            "errors": [],
        },
    )
    snapshot_id = "snap_" + "a" * 24
    snapshot_receipt = {
        "snapshot_id": snapshot_id,
        "receipt_digest": "c" * 64,
        "provenance_status": "legacy_reconstructed",
    }
    monkeypatch.setattr(
        runs_routes,
        "resolve_recovery_snapshot",
        lambda *_args, **_kwargs: (
            project_dir / "runs" / "_recovery" / "checkpoint.pt",
            snapshot_receipt,
        ),
    )
    contract_receipt = {
        "source": {"contract_sha256": "d" * 64},
        "target": {
            "selection_path": str(project_dir / "env" / "selection_v6.json"),
            "contract_sha256": "d" * 64,
        },
        "compatibility": {"type": "exact_policy_contract"},
    }
    monkeypatch.setattr(
        runs_routes,
        "build_recovery_snapshot_warm_start_contract_receipt",
        lambda *_args, **_kwargs: contract_receipt,
    )

    async def _runner(_job, _cancel):
        return {"return_code": 0}

    monkeypatch.setattr(
        runs_routes,
        "run_sculpt_job",
        lambda **_kwargs: _runner,
    )
    response = client.post(f"/projects/{slug}/runs", json={
        "behavior_goal": "continue a difficult motion",
        "iterations": 1,
        "dry_run": True,
        "initialization_mode": "actor_critic",
        "warm_start_snapshot": {
            "snapshot_id": snapshot_id,
            "checkpoint_sha256": "b" * 64,
            "receipt_digest": "c" * 64,
            "acknowledge_interrupted_snapshot": True,
            "acknowledge_legacy_reconstructed_snapshot": True,
        },
    })
    assert response.status_code == 202, response.text
    jobs = client.app.state.job_manager.list(  # type: ignore[attr-defined]
        kind="sculpt_run", project_slug=slug,
    )
    assert len(jobs) == 1
    assert jobs[0].params["initialization_mode"] == "actor_critic"
    assert jobs[0].params["recovery_snapshot_receipt"] == snapshot_receipt
    assert jobs[0].params["warm_start_policy_contract_receipt"] == (
        contract_receipt
    )

    rejected_mode = client.post(f"/projects/{slug}/runs", json={
        "behavior_goal": "continue a difficult motion",
        "iterations": 1,
        "dry_run": True,
        "initialization_mode": "actor_only",
        "warm_start_snapshot": {
            "snapshot_id": snapshot_id,
            "checkpoint_sha256": "b" * 64,
            "receipt_digest": "c" * 64,
            "acknowledge_interrupted_snapshot": True,
            "acknowledge_legacy_reconstructed_snapshot": True,
        },
    })
    assert rejected_mode.status_code == 412
    assert rejected_mode.json()["type"] == (
        "/problems/recovery-snapshot-initialization-mode"
    )


@pytest.mark.parametrize(
    "roles",
    [
        ["actor", "actor", "critic"],
        ["actor", "critic", "optimizer"],
    ],
)
def test_recovery_load_receipt_rejects_duplicate_or_extra_roles(
    tmp_path: Path, roles: list[str],
) -> None:
    from backend.services.run_manager import _verify_starting_skill_load_event

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact-recovery")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="expected exactly"):
        _verify_starting_skill_load_event(
            {
                "type": "warm_start_loaded",
                "source": str(checkpoint),
                "source_sha256": digest,
                "loaded_checkpoint": str(checkpoint),
                "loaded_checkpoint_sha256": digest,
                "adapted": False,
                "load_cfg_keys": roles,
            },
            expected_checkpoint=checkpoint,
            expected_sha256=digest,
            initialization_mode="actor_critic",
            require_unadapted=True,
        )


def test_recovery_load_receipt_rejects_other_loaded_bytes_or_adaptation(
    tmp_path: Path,
) -> None:
    from backend.services.run_manager import _verify_starting_skill_load_event

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact-recovery")
    other = tmp_path / "adapted.pt"
    other.write_bytes(b"different-loaded-bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    other_digest = hashlib.sha256(other.read_bytes()).hexdigest()
    base = {
        "type": "warm_start_loaded",
        "source": str(checkpoint),
        "source_sha256": digest,
        "loaded_checkpoint": str(other),
        "loaded_checkpoint_sha256": other_digest,
        "load_cfg_keys": ["actor", "critic"],
    }
    with pytest.raises(ValueError, match="did not load the selected"):
        _verify_starting_skill_load_event(
            {**base, "adapted": False},
            expected_checkpoint=checkpoint,
            expected_sha256=digest,
            initialization_mode="actor_critic",
            require_unadapted=True,
        )
    with pytest.raises(ValueError, match="cannot be adapted"):
        _verify_starting_skill_load_event(
            {
                **base,
                "adapted": True,
                "derived_from": {
                    "source": str(checkpoint.resolve()),
                    "source_sha256": digest,
                },
                "policy_contract_migration": (
                    "zero_initialized_event_phase_observation"
                ),
            },
            expected_checkpoint=checkpoint,
            expected_sha256=digest,
            initialization_mode="actor_critic",
            require_unadapted=True,
        )


def test_recovery_same_iteration_retry_accepts_exact_local_checkpoint(
    tmp_path: Path,
) -> None:
    from backend.services.run_manager import (
        _verify_local_checkpoint_reuse_events,
    )

    project_dir = tmp_path / "project"
    selected = (
        project_dir / "runs" / "_recovery" / ("snap_" + "a" * 24)
    )
    selected = selected / "checkpoint.pt"
    local = project_dir / "runs" / "iter_2" / "checkpoint.pt"
    selected.parent.mkdir(parents=True)
    local.parent.mkdir(parents=True)
    checkpoint_bytes = b"exact-actor-and-critic"
    selected.write_bytes(checkpoint_bytes)
    local.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    phase_event = {
        "type": "phase_skipped",
        "iter": 2,
        "phase": "train",
        "reason": "checkpoint already on disk",
        "checkpoint": str(local),
    }
    skip_event = {
        "type": "warm_start_skipped",
        "iter": 2,
        "reason": "local_checkpoint_wins",
        "source": str(selected),
    }

    receipt = _verify_local_checkpoint_reuse_events(
        phase_event,
        skip_event,
        expected_checkpoint=selected,
        expected_sha256=digest,
        initialization_mode="actor_critic",
        project_dir=project_dir,
    )
    assert receipt["reuse_kind"] == "content_equivalent_local_checkpoint"
    assert receipt["load_cfg_keys"] == ["actor", "critic"]
    assert receipt["loaded_checkpoint_sha256"] == digest

    local.write_bytes(b"different")
    with pytest.raises(ValueError, match="differ from the selected"):
        _verify_local_checkpoint_reuse_events(
            phase_event,
            skip_event,
            expected_checkpoint=selected,
            expected_sha256=digest,
            initialization_mode="actor_critic",
            project_dir=project_dir,
        )


def test_worker_revalidates_snapshot_and_forwards_exact_actor_critic_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services import recovery_snapshots, run_manager
    from backend.services.job_manager import Job
    from sculptor import policy_contract

    project_dir = tmp_path / "worker"
    cached = project_dir / "runs" / "_recovery" / "snap" / "checkpoint.pt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"immutable-snapshot")
    checkpoint_sha = hashlib.sha256(cached.read_bytes()).hexdigest()
    snapshot_id = "snap_" + "a" * 24
    snapshot_receipt = {
        "snapshot_id": snapshot_id,
        "receipt_digest": "c" * 64,
        "provenance_status": "legacy_reconstructed",
        "checkpoint": {"ppo_step": 50},
        "source": {
            "iteration": 2,
            "last_observed_ppo_step": 58,
        },
    }
    contract = {"schema": 3, "policy": {"roles": ["actor", "critic"]}}
    contract_receipt = {
        "source": {
            "contract": contract,
            "contract_sha256": "d" * 64,
        },
        "target": {
            "selection_path": str(project_dir / "env" / "selection_v6.json"),
            "contract": contract,
            "contract_sha256": "d" * 64,
        },
        "compatibility": {"type": "exact_policy_contract"},
    }
    monkeypatch.setattr(
        recovery_snapshots,
        "resolve_recovery_snapshot",
        lambda *_args, **_kwargs: (cached, snapshot_receipt),
    )
    monkeypatch.setattr(
        policy_contract,
        "build_recovery_snapshot_warm_start_contract_receipt",
        lambda *_args, **_kwargs: contract_receipt,
    )

    captured: dict[str, object] = {}

    class _Spawned(Exception):
        pass

    async def _fake_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        captured["env"] = dict(kwargs["env"])
        raise _Spawned()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    runner = run_manager.run_sculpt_job(
        project_dir=project_dir,
        run_params={
            "behavior_goal": "continue a difficult motion",
            "iterations": 1,
            "initialization_mode": "actor_critic",
            "warm_start_snapshot": {
                "snapshot_id": snapshot_id,
                "checkpoint_sha256": checkpoint_sha,
                "receipt_digest": "c" * 64,
                "acknowledge_interrupted_snapshot": True,
                "acknowledge_legacy_reconstructed_snapshot": True,
            },
            "recovery_snapshot_receipt": snapshot_receipt,
            "warm_start_policy_contract_receipt": contract_receipt,
        },
    )
    job = Job(
        job_id="job_worker_test",
        kind="sculpt_run",
        project_slug="worker",
        status="running",
    )
    job._cancel = asyncio.Event()
    with pytest.raises(_Spawned):
        asyncio.run(runner(job, job._cancel))

    cmd = captured["cmd"]
    env = captured["env"]
    assert isinstance(cmd, list) and isinstance(env, dict)
    assert cmd[cmd.index("--init-policy") + 1] == str(cached)
    assert cmd[cmd.index("--init-policy-mode") + 1] == "actor_critic"
    assert env["SCULPTOR_WARM_START_CHECKPOINT_SHA256"] == checkpoint_sha
    event = next(
        item for item in job.events
        if item.get("type") == "warm_start_snapshot_resolved"
    )
    assert event["load_cfg_keys"] == ["actor", "critic"]
    assert event["optimizer_resume"] is False
