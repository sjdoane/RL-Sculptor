"""Exact project robot-namespace mapping regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.services.project_robot import resolve_project_reference_robot


def _project_metadata(
    tmp_path: Path,
    *,
    library_slug: str | None,
    reference_robot: str | None,
) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source: dict[str, str] = {"kind": "library"}
    if library_slug is not None:
        source["library_slug"] = library_slug
    if reference_robot is not None:
        source["reference_robot"] = reference_robot
    (project_dir / "metadata.json").write_text(
        json.dumps({"robot_source": source}), encoding="utf-8"
    )
    return project_dir


def _write_adapter_config(project_dir: Path) -> None:
    (project_dir / "config.toml").write_text(
        "[adapter]\n"
        'class = "sculptor.adapters.mjlab.MjlabAdapter"\n'
        "[adapter.config]\n"
        'task_id = "Mjlab-Velocity-Flat-Unitree-G1"\n',
        encoding="utf-8",
    )


def test_explicit_catalog_namespace_is_authoritative(tmp_path: Path) -> None:
    project_dir = _project_metadata(
        tmp_path, library_slug="unitree_g1", reference_robot="g1"
    )
    assert resolve_project_reference_robot(project_dir) == "g1"


def test_run_target_uses_same_reference_namespace(tmp_path: Path) -> None:
    from backend.services.run_manager import resolve_starting_skill_target

    project_dir = _project_metadata(
        tmp_path, library_slug="unitree_g1", reference_robot="g1"
    )
    _write_adapter_config(project_dir)

    target, receipt = resolve_starting_skill_target(
        project_dir, require_policy_contract=False
    )
    assert target["robot_slug"] == "g1"
    assert receipt["robot_slug"] == "g1"


def test_legacy_allowlisted_project_is_preserved(tmp_path: Path) -> None:
    project_dir = _project_metadata(
        tmp_path, library_slug="unitree_go1", reference_robot=None
    )
    assert resolve_project_reference_robot(project_dir) == "go1"


def test_legacy_library_name_is_preserved_only_through_allowlist(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "metadata.json").write_text(
        json.dumps({"robot_source": {"library_name": "hopper"}}),
        encoding="utf-8",
    )
    assert resolve_project_reference_robot(project_dir) == "hopper"


def test_unknown_legacy_slug_does_not_use_task_or_g1_default(
    tmp_path: Path,
) -> None:
    project_dir = _project_metadata(
        tmp_path, library_slug="research_humanoid", reference_robot=None
    )
    with pytest.raises(ValueError, match="no explicit reference robot"):
        resolve_project_reference_robot(project_dir)


def test_contradictory_explicit_namespace_fails_closed(tmp_path: Path) -> None:
    project_dir = _project_metadata(
        tmp_path, library_slug="unitree_g1", reference_robot="go1"
    )
    with pytest.raises(ValueError, match="contradicts"):
        resolve_project_reference_robot(project_dir)


def test_project_api_exposes_exact_reference_namespace(
    client: TestClient,
) -> None:
    created = client.post(
        "/projects", json={"name": "Reference namespace", "adapter": "gym_sb3"}
    )
    assert created.status_code == 201, created.text
    slug = created.json()["slug"]
    client.app.state.project_store.write_robot_source(  # type: ignore[attr-defined]
        slug,
        {
            "kind": "library",
            "library_slug": "unitree_g1",
            "reference_robot": "g1",
        },
    )

    detail = client.get(f"/projects/{slug}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["library_slug"] == "unitree_g1"
    assert detail.json()["reference_robot"] == "g1"
