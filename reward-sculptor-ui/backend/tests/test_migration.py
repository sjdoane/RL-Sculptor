"""Tests for migration_warning detection (M6.1).

A legacy project whose config.toml references an adapter no longer in
`ADAPTER_REGISTRY` surfaces `migration_warning` on GET /projects and
/projects/{slug}. Projects on current adapters do not.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _spawn_project(tmp_projects_root: Path, slug: str, adapter_class: str) -> None:
    """Hand-craft a minimal project directory that looks like a legacy
    scaffold, avoiding sculpt_init (which only knows gym_sb3)."""
    import json
    pdir = tmp_projects_root / slug
    pdir.mkdir(parents=True)
    (pdir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 1,
            "slug": slug,
            "display_name": slug.replace("-", " ").title(),
            "description": "",
            "created_at": "2024-01-01T00:00:00+00:00",
            "last_run_id": None,
            "last_run_status": None,
            "last_run_at": None,
            "robot_source": {"kind": "library"},
            "iteration_budget": 20,
            "behavior_goal": "",
            "ui_state": {},
        }),
        encoding="utf-8",
    )
    (pdir / "config.toml").write_text(
        f'[adapter]\nclass = "{adapter_class}"\nconfig = {{ }}\n',
        encoding="utf-8",
    )


def test_gym_sb3_project_has_no_migration_warning(
    client: TestClient, tmp_projects_root: Path
) -> None:
    _spawn_project(tmp_projects_root, "gym-project", "sculptor.adapters.gym_sb3.GymSB3Adapter")
    r = client.get("/projects")
    assert r.status_code == 200
    match = next((p for p in r.json() if p["slug"] == "gym-project"), None)
    assert match is not None
    assert match.get("migration_warning") in (None, "")


def test_mjlab_project_has_no_migration_warning(
    client: TestClient, tmp_projects_root: Path
) -> None:
    _spawn_project(tmp_projects_root, "mjlab-project", "sculptor.adapters.mjlab.MjlabAdapter")
    r = client.get("/projects")
    assert r.status_code == 200
    match = next((p for p in r.json() if p["slug"] == "mjlab-project"), None)
    assert match is not None
    assert match.get("migration_warning") in (None, "")


def test_coming_soon_adapter_no_migration_warning(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """Coming-soon adapters are IN the registry (status=coming_soon).
    They should NOT trigger migration_warning — they're forthcoming,
    not legacy."""
    _spawn_project(tmp_projects_root, "isaac-project", "sculptor.adapters.isaac_lab.IsaacLabAdapter")
    r = client.get("/projects")
    match = next((p for p in r.json() if p["slug"] == "isaac-project"), None)
    assert match is not None
    assert match.get("migration_warning") in (None, "")


def test_unknown_adapter_triggers_migration_warning(
    client: TestClient, tmp_projects_root: Path
) -> None:
    """An adapter class_path not present in any ADAPTER_REGISTRY entry
    is the trigger for migration_warning."""
    _spawn_project(
        tmp_projects_root,
        "legacy-phantom",
        "sculptor.adapters.legacy_foo.FooAdapter",  # not in registry
    )
    r = client.get("/projects")
    assert r.status_code == 200
    match = next((p for p in r.json() if p["slug"] == "legacy-phantom"), None)
    assert match is not None
    assert match.get("migration_warning") is not None
    assert "no longer registered" in match["migration_warning"]


def test_detail_endpoint_surfaces_migration_warning(
    client: TestClient, tmp_projects_root: Path
) -> None:
    _spawn_project(
        tmp_projects_root,
        "legacy-phantom-2",
        "sculptor.adapters.legacy_bar.BarAdapter",
    )
    r = client.get("/projects/legacy-phantom-2")
    assert r.status_code == 200
    body = r.json()
    assert body.get("migration_warning") is not None
    assert "no longer registered" in body["migration_warning"]
