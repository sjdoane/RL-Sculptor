"""Interrupted PPO snapshot discovery, admission, and worker tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def _event(payload: dict) -> str:
    return "[SCULPT-EVENT] " + json.dumps(payload, separators=(",", ":"))


def _plant_interrupted_snapshot(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_step: int = 50,
    completed: bool = False,
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

    checkpoint_bytes = b"actor-and-critic-at-ppo-50"
    model_path = logs_dir / f"model_{model_step}.pt"
    if symlink_model:
        outside = project_dir.parent / "outside-model.pt"
        outside.write_bytes(checkpoint_bytes)
        model_path.symlink_to(outside)
    else:
        model_path.write_bytes(checkpoint_bytes)
    # model_0 is intentionally never a recoverable trained snapshot.
    (logs_dir / "model_0.pt").write_bytes(b"initial-policy")
    if completed:
        (iter_dir / "checkpoint.pt").write_bytes(b"completed-policy")

    contract = {
        "schema": 3,
        "adapter": {"class": "test.Adapter", "task_id": "test-task"},
        "policy": {"roles": ["actor", "critic"]},
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

    log_path = project_dir / "runs" / "_run_job_deadbeefcafebabe.log"
    log_path.write_text("\n".join([
        _event({
            "type": "artifact_tuple_pinned",
            "iter": iteration,
            "tuple_hash": "f" * 64,
            "selection": "selection_v6.json",
        }),
        _event({"type": "iter_started", "iter": iteration, "steps": 750}),
        _event({
            "type": "warm_start_observation_extended",
            "effective_policy_contract": str(contract_path.resolve()),
            "effective_policy_contract_sha256": contract_sha,
        }),
        _event({
            "type": "warm_start_loaded",
            "source": "/server-owned/source.pt",
            "source_sha256": "a" * 64,
            "load_cfg_keys": ["actor", "critic"],
        }),
        _event({
            "type": "learning_vitals", "rl_iter": 58, "rl_total": 750,
        }),
        "RuntimeError: mjlab runner exited 1",
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
