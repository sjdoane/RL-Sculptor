"""Exact-context, observe-only production receipts for the mode gauntlet."""

from __future__ import annotations

from pathlib import Path


def _authority() -> dict:
    return {
        "schema": 1,
        "clip_id": "combo",
        "reference_robot": "g1",
        "clip_sha256": "1" * 64,
        "reward_path": "/project/rewards/v2.py",
        "reward_sha256": "2" * 64,
        "selector_path": "/project/rewards/current.py",
        "selector_sha256": "3" * 64,
        "graph_sha256": "4" * 64,
        "execution_manifest_sha256": "5" * 64,
        "mode_binding": {"schema": 1},
        "selection": {"present": True, "valid": True, "tuple_hash": "6" * 64},
        "context_sha256": "7" * 64,
    }


def test_readiness_receipt_is_immutable_and_never_claims_fitness(
    tmp_path: Path, monkeypatch,
) -> None:
    from backend.services import mode_evidence

    monkeypatch.setattr(
        mode_evidence,
        "_resolve_authority",
        lambda *_args, **_kwargs: (
            _authority(),
            {
                "mode_order": ["approach", "jump"],
                "have": {
                    "scores": False,
                    "transitions": False,
                    "validation": False,
                    "calibration": False,
                },
            },
            [],
        ),
    )

    receipt = mode_evidence.build_readiness_receipt(
        tmp_path,
        expected_clip_id="combo",
        expected_robot="g1",
        created_at=10.0,
    )
    assert receipt["trust_status"] == "observe_only"
    assert receipt["evidence_status"] == "absent"
    assert receipt["fitness_or_selection_authority"] is False
    assert receipt["training_consumer_active"] is False
    assert receipt["gauntlet"]["have"]["validation"] is False

    path = mode_evidence.persist_readiness_receipt(tmp_path, receipt)
    assert path.name == f"{receipt['receipt_sha256']}.json"
    assert mode_evidence.latest_receipt(tmp_path, "7" * 64) == receipt
    # The same bytes are idempotent; no mutable alias is written.
    assert mode_evidence.persist_readiness_receipt(tmp_path, receipt) == path


def test_mode_evidence_routes_expose_status_and_record_action(
    client, monkeypatch, tmp_path: Path,
) -> None:
    from backend.routes import mode_evidence as route

    created = client.post(
        "/projects", json={"name": "Mode Evidence", "adapter": "gym_sb3"}
    )
    assert created.status_code == 201
    slug = created.json()["slug"]
    record = {
        "schema": 1,
        "created_at": 10.0,
        "recorded": False,
        "receipt_sha256": "8" * 64,
        "authority": _authority(),
        "trust_status": "observe_only",
        "evidence_status": "absent",
        "fitness_or_selection_authority": False,
        "training_consumer_active": False,
        "blockers": ["no metric set"],
        "gauntlet": {"mode_order": ["approach"], "have": {}},
        "next_action": "generate metrics",
    }
    monkeypatch.setattr(route.mode_evidence, "status", lambda *_a, **_k: record)
    response = client.get(
        f"/projects/{slug}/mode-evidence?clip_id=combo&robot=g1"
    )
    assert response.status_code == 200
    assert response.json()["fitness_or_selection_authority"] is False

    monkeypatch.setattr(
        route.mode_evidence, "build_readiness_receipt", lambda *_a, **_k: record
    )
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(
        route.mode_evidence, "persist_readiness_receipt", lambda *_a, **_k: receipt_path
    )
    response = client.post(
        f"/projects/{slug}/mode-evidence/receipt?clip_id=combo&robot=g1"
    )
    assert response.status_code == 201
    assert response.json()["recorded"] is True
    assert response.json()["receipt_path"] == str(receipt_path)
