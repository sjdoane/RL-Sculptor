"""Fail-closed target admission at direct and mission reference boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _write_robot(
    project,
    slug: str = "g1",
    *,
    reference_robot: str | None = None,
) -> None:
    robot_source = {"library_slug": slug}
    if reference_robot is not None:
        robot_source["reference_robot"] = reference_robot
    (project / "metadata.json").write_text(
        json.dumps({"robot_source": robot_source}),
        encoding="utf-8",
    )


def _certificate():
    return SimpleNamespace(
        clip_content_sha256="1" * 64,
        rollout_sha256="2" * 64,
        certificate_sha256="3" * 64,
        execution_contract_sha256="4" * 64,
        execution_boundary_sha256="5" * 64,
    )


def test_stage_robot_resolution_has_no_implicit_g1_fallback(tmp_path) -> None:
    from sculptor.sculpt import _stage_reference_robot_slug

    with pytest.raises(ValueError, match="exact robot identity"):
        _stage_reference_robot_slug(
            stage_dir=tmp_path / "stage",
            project_root=tmp_path,
        )

    _write_robot(tmp_path, "t1", reference_robot="t1")
    assert _stage_reference_robot_slug(
        stage_dir=tmp_path / "stage",
        project_root=tmp_path,
    ) == "t1"


def test_stage_robot_resolution_uses_explicit_catalog_namespace(tmp_path) -> None:
    from sculptor.sculpt import _stage_reference_robot_slug

    _write_robot(tmp_path, "unitree_g1", reference_robot="g1")
    assert _stage_reference_robot_slug(
        stage_dir=tmp_path / "stage",
        project_root=tmp_path,
    ) == "g1"


def test_direct_reference_admission_rejects_cross_robot_before_tierd(
    tmp_path, monkeypatch,
) -> None:
    from sculptor.refs.track import TierDAdmissionError
    from sculptor.sculpt import _admit_reference_motion_for_target

    _write_robot(tmp_path, "g1")
    monkeypatch.setattr(
        "sculptor.refs.track.require_tierd_admission",
        lambda *_a, **_kw: pytest.fail("cross-robot input must fail first"),
    )
    with pytest.raises(TierDAdmissionError, match="does not match"):
        _admit_reference_motion_for_target(
            project=tmp_path,
            reference_robot="t1",
            reference_clip_id="flip",
        )


def test_direct_reference_admission_rejects_target_contract_drift(
    tmp_path, monkeypatch,
) -> None:
    from sculptor.refs.track import TierDAdmissionError
    from sculptor.sculpt import _admit_reference_motion_for_target

    _write_robot(tmp_path)
    certificate = _certificate()
    monkeypatch.setattr(
        "sculptor.refs.track.require_tierd_admission",
        lambda robot, clip_id: certificate,
    )

    def _reject(*_args, **_kwargs):
        raise TierDAdmissionError("identity.task_id differs after queue")

    monkeypatch.setattr(
        "sculptor.refs.track.require_tierd_target_compatibility",
        _reject,
    )
    with pytest.raises(TierDAdmissionError, match="task_id differs"):
        _admit_reference_motion_for_target(
            project=tmp_path,
            reference_robot="g1",
            reference_clip_id="flip",
        )


def test_direct_reference_admission_emits_complete_execution_evidence(
    tmp_path, monkeypatch,
) -> None:
    from sculptor.sculpt import (
        _admit_reference_motion_for_target,
        _reference_feasibility_admission_event,
    )

    _write_robot(tmp_path)
    certificate = _certificate()
    monkeypatch.setattr(
        "sculptor.refs.track.require_tierd_admission",
        lambda robot, clip_id: certificate,
    )
    observed = {}

    def _compatible(cert, project, *, target_robot, target_policy_contract=None):
        observed.update(
            cert=cert,
            project=project,
            target_robot=target_robot,
            target_policy_contract=target_policy_contract,
        )
        return cert

    monkeypatch.setattr(
        "sculptor.refs.track.require_tierd_target_compatibility",
        _compatible,
    )
    target_robot, admitted = _admit_reference_motion_for_target(
        project=tmp_path,
        reference_robot="g1",
        reference_clip_id="flip",
    )
    event = _reference_feasibility_admission_event(
        certificate=admitted,
        reference_robot="g1",
        target_robot=target_robot,
        reference_clip_id="flip",
    )

    assert observed == {
        "cert": certificate,
        "project": tmp_path,
        "target_robot": "g1",
        "target_policy_contract": None,
    }
    assert event == {
        "type": "reference_feasibility_admitted",
        "source": "sculpt_run_worker",
        "reference_robot": "g1",
        "target_robot": "g1",
        "reference_clip_id": "flip",
        "clip_sha256": "1" * 64,
        "rollout_sha256": "2" * 64,
        "certificate_sha256": "3" * 64,
        "execution_contract_sha256": "4" * 64,
        "execution_boundary_sha256": "5" * 64,
    }
